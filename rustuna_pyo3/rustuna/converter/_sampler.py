from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

import rustuna
from rustuna._rustuna import ToRustStorage
from rustuna.converter import (
    to_rustuna_directions,
    to_rustuna_distribution,
    to_rustuna_distributions,
)
from rustuna.converter._storage import ToRustunaStorage
from rustuna.converter._trial import to_rustuna_state

if TYPE_CHECKING:
    from rustuna.samplers import SamplerProtocol
    from rustuna.storages import StorageProtocol


class ToOptunaSampler(BaseSampler):
    """Adapt a Rustuna sampler to Optuna's ``BaseSampler`` interface.

    Args:
        sampler: The Rustuna sampler to adapt.

    Example:
        Use a Rustuna sampler in an Optuna study.

        ```python
        import optuna
        import rustuna
        from rustuna.converter import ToOptunaSampler

        def objective(trial: optuna.Trial) -> float:
            # Define your objective function.
            return trial.suggest_float("x", -1, 1) ** 2

        sampler = ToOptunaSampler(rustuna.samplers.TPESampler())
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=10)
        ```
    """

    def __init__(self, sampler: SamplerProtocol) -> None:
        self._sampler = sampler
        self._inter_section_search_space = IntersectionSearchSpace()
        self._storage: ToRustStorage | None = None

    def _get_storage(self, storage: BaseStorage) -> StorageProtocol:
        if self._storage is None:
            self._storage = ToRustStorage(ToRustunaStorage(storage))
        return self._storage

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """Sample jointly distributed parameters using the Rustuna sampler."""
        if search_space == {}:
            return {}

        ctx = rustuna.samplers.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        # PyObjectStorage syncs its Rustuna-side cache with the underlying Optuna storage
        # only when get_trials() is called. In this flow, trials are created and completed
        # through the Optuna storage directly, so sync explicitly here; otherwise Rustuna
        # samplers reading the cache (e.g. get_cached_trial) would observe stale or missing
        # trials.
        # TODO(c-bata): Consider how to remove this redundant storage.get_trials() call.
        storage.get_trials(study._study_id, states=[rustuna.trial.TrialState.COMPLETE])
        rustuna_search_space = to_rustuna_distributions(search_space)
        internal_params = self._sampler.sample_joint(ctx, storage, rustuna_search_space)
        external_params: dict[str, Any] = {}
        for param_name in internal_params:
            distribution = search_space[param_name]
            external_param_value = distribution.to_external_repr(
                internal_params[param_name]
            )
            external_params[param_name] = external_param_value
        return external_params

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        """Run pre-processing in the Rustuna sampler before search-space inference."""
        ctx = rustuna.samplers.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        self._sampler.before_trial(ctx, storage)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample one parameter using the Rustuna sampler."""
        ctx = rustuna.samplers.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        distribution = to_rustuna_distribution(param_distribution)
        internal_param = self._sampler.sample_independent(
            ctx, storage, param_name, distribution
        )
        return param_distribution.to_external_repr(internal_param)

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        """Infer the relative search space supported by the Rustuna sampler."""
        if not self._sampler.support_joint_sampling:
            return {}

        # TODO(y0z): Support study.get_joint_search_space insead of using Optuna Python API
        # search_space = study.get_joint_search_space(study._study_id)
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._inter_section_search_space.calculate(
            study, use_cache=True
        ).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Notify the Rustuna sampler that a trial has finished."""
        ctx = rustuna.samplers.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=to_rustuna_directions(study._directions),
        )
        storage = self._get_storage(study._storage)
        self._sampler.after_trial(
            ctx,
            storage,
            to_rustuna_state(state),
            list(values) if values is not None else None,
        )
