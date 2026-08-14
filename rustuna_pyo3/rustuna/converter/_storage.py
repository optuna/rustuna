from __future__ import annotations

import copy
import json
import threading
import typing
import uuid
from collections.abc import Container, Iterable, Sequence

import optuna
from optuna.distributions import BaseDistribution
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX, BaseStorage
from optuna.study import StudyDirection
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial, TrialState

import rustuna

from ._attrs import to_optuna_attrs, to_rustuna_attrs
from ._direction import to_optuna_directions
from ._distribution import to_optuna_distribution, to_rustuna_distribution
from ._frozen_study import to_frozen_study, to_persisted_study
from ._trial import (
    _OPTUNA_CONSTRAINTS_KEY,
    _RUSTUNA_CONSTRAINTS_KEY,
    FrozenTrialLike,
    to_frozen_trial,
    to_optuna_state,
    to_persisted_trial,
    to_rustuna_state,
)

if typing.TYPE_CHECKING:
    from typing import Any

    from optuna._typing import JSONSerializable

    from rustuna import CategoricalChoiceType
    from rustuna._rustuna import Distribution


logger = optuna.logging.get_logger(__name__)


class ToRustunaStorage:
    """Adapt an Optuna storage to Rustuna's storage protocol.

    Args:
        storage: The Optuna storage to adapt.

    Example:
        Use an Optuna SQLite-backed storage with a Rustuna study.

        ```python
        import optuna
        import rustuna
        from rustuna.converter import ToRustunaStorage

        def objective(trial: rustuna.Trial) -> float:
            # Define your objective function.
            return trial.suggest_float("x", -1, 1) ** 2

        optuna_storage = optuna.storages.RDBStorage("sqlite:///study.db")
        storage = ToRustunaStorage(optuna_storage)
        study = rustuna.create_study(storage=storage)
        study.optimize(objective, n_trials=10)
        ```
    """

    def __init__(self, storage: BaseStorage) -> None:
        self._storage = storage
        self._trial_id_to_study_id: dict[int, int] = {}
        self._lock = threading.Lock()

    def create_new_study(
        self, study_name: str, directions: list[rustuna.study.StudyDirection]
    ) -> rustuna.study.PersistedStudy:
        optuna_directions = to_optuna_directions(directions)
        study_id = self._storage.create_new_study(optuna_directions, study_name)
        return rustuna.study.PersistedStudy(
            id=study_id,
            name=study_name,
            directions=directions,
            user_attrs={},
            system_attrs={},
        )

    def create_new_trial(
        self, study_id: int, template_trial: rustuna.trial.PersistedTrial | None = None
    ) -> rustuna.trial.PersistedTrial:
        template_frozen = to_frozen_trial(template_trial) if template_trial else None
        if template_frozen is not None and template_trial is not None:
            if template_trial.constraints:
                template_frozen.system_attrs[_RUSTUNA_CONSTRAINTS_KEY] = dict(
                    template_trial.constraints
                )
        trial_id = self._storage.create_new_trial(
            study_id, template_trial=template_frozen
        )
        trial = self._storage.get_trial(trial_id)
        with self._lock:
            self._trial_id_to_study_id[trial_id] = study_id
        return to_persisted_trial(trial, study_id)

    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None:
        self._storage.set_trial_param(
            trial_id, name, value, to_optuna_distribution(distribution)
        )

    def set_trial_state_values(
        self,
        trial_id: int,
        state: rustuna.trial.TrialState,
        values: None | list[float] = None,
    ) -> None:
        self._storage.set_trial_state_values(
            trial_id,
            to_optuna_state(state),
            values=values,
        )

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        self._storage.set_trial_intermediate_value(trial_id, step, intermediate_value)

    def get_studies(self) -> list[rustuna.study.PersistedStudy]:
        frozen_studies = self._storage.get_all_studies()
        return [to_persisted_study(s) for s in frozen_studies]

    def get_study(self, study_id: int) -> rustuna.study.PersistedStudy:
        frozen_studies = self._storage.get_all_studies()
        for s in frozen_studies:
            if s._study_id == study_id:
                return to_persisted_study(s)
        raise KeyError(f"Study {study_id} not found")

    def get_trials(
        self,
        study_id: int,
        *,
        states: list[rustuna.trial.TrialState] | None = None,
    ) -> list[rustuna.trial.PersistedTrial]:
        optuna_states = None
        if states is not None:
            optuna_states = tuple(to_optuna_state(state) for state in states)
        frozen_trials = self._storage.get_all_trials(study_id, states=optuna_states)

        persisted_trials: list[rustuna.trial.PersistedTrial] = []
        with self._lock:
            for t in frozen_trials:
                persisted_trials.append(to_persisted_trial(t, study_id))
                self._trial_id_to_study_id[t._trial_id] = study_id
        return persisted_trials

    def get_n_trials(
        self,
        study_id: int,
        *,
        states: Sequence[rustuna.trial.TrialState] | None = None,
    ) -> int:
        if states is None:
            return self._storage.get_n_trials(study_id)
        optuna_states = tuple(to_optuna_state(state) for state in states)
        if not optuna_states:
            return 0
        if len(optuna_states) == 1:
            return self._storage.get_n_trials(study_id, state=optuna_states[0])
        return len(self._storage.get_all_trials(study_id, states=optuna_states))

    def get_trial(self, trial_id: int) -> rustuna.trial.PersistedTrial:
        with self._lock:
            study_id = self._trial_id_to_study_id.get(trial_id, -1)
        if study_id == -1:
            logger.warning(
                f"Failed to get study_id while converting to PersistedTrial({trial_id=})"
                "due to the implementation restriction"
            )
        frozen_trial = self._storage.get_trial(trial_id)
        return to_persisted_trial(frozen_trial, study_id=study_id)

    def get_cached_trial(self, trial_id: int) -> rustuna.trial.PersistedTrial:
        return self.get_trial(trial_id)

    def discard_trials(self, trial_ids: list[int]) -> None:
        pass

    def may_omit_trials(self) -> bool:
        return False

    def delete_study(self, study_id: int) -> None:
        self._storage.delete_study(study_id)

    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int:
        return self._storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number
        )

    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_study_system_attr(study_id, key, value)

    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_study_user_attr(study_id, key, value)

    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_trial_system_attr(trial_id, key, value)

    def set_trial_constraints(
        self, trial_id: int, constraints: dict[str, float]
    ) -> None:
        sorted_constraints = sorted(constraints.items())
        self._storage.set_trial_system_attr(
            trial_id,
            _OPTUNA_CONSTRAINTS_KEY,
            [value for _, value in sorted_constraints],
        )
        self._storage.set_trial_system_attr(
            trial_id, _RUSTUNA_CONSTRAINTS_KEY, constraints
        )

    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_trial_user_attr(trial_id, key, value)

    def get_study_user_attr(self, study_id: int, key: str) -> str:
        value = self._storage.get_study_user_attrs(study_id)[key]
        return json.dumps(value) if not isinstance(value, str) else value

    def get_study_system_attr(self, study_id: int, key: str) -> str:
        value = self._storage.get_study_system_attrs(study_id)[key]
        return json.dumps(value) if not isinstance(value, str) else value

    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None:
        key = f"optuna_category_labels:{param_name}"
        self._storage.set_study_system_attr(study_id, key, choices)

    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None:
        key = f"optuna_category_labels:{param_name}"
        value = self._storage.get_study_system_attrs(study_id).get(key)
        if value is None:
            return None
        assert isinstance(value, list)
        assert len(value) == cardinality
        return typing.cast("list[CategoricalChoiceType]", value)


class ToOptunaStorage(BaseStorage):
    """Adapt a Rustuna storage to Optuna's ``BaseStorage`` interface.

    Args:
        storage: The Rustuna storage to adapt.

    Example:
        Use a Rustuna journal storage with an Optuna study.

        ```python
        import optuna
        import rustuna
        from rustuna.converter import ToOptunaStorage

        def objective(trial: optuna.Trial) -> float:
            # Define your objective function.
            return trial.suggest_float("x", -1, 1) ** 2

        rustuna_storage = rustuna.storages.JournalFileStorage("study.log")
        storage = ToOptunaStorage(rustuna_storage)
        study = optuna.create_study(storage=storage)
        study.optimize(objective, n_trials=10)
        ```
    """

    def __init__(self, storage: rustuna.storages.StorageProtocol) -> None:
        self._storage = storage
        self._trial_cache: dict[int, FrozenTrialLike] = {}

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        rustuna_directions: list[rustuna.study.StudyDirection] = []
        for d in directions:
            if d == StudyDirection.MINIMIZE:
                rustuna_directions.append(rustuna.study.StudyDirection.MINIMIZE)
            elif d == StudyDirection.MAXIMIZE:
                rustuna_directions.append(rustuna.study.StudyDirection.MAXIMIZE)
            else:
                raise ValueError("Unexpected Study Direction")

        study_name = study_name or DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4())
        try:
            study = self._storage.create_new_study(study_name, rustuna_directions)
        except rustuna.exceptions.DuplicatedStudyError as e:
            raise optuna.exceptions.DuplicatedStudyError(f"{study_name} is duplicated")
        return study.id

    def delete_study(self, study_id: int) -> None:
        self._storage.delete_study(study_id=study_id)

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        self._storage.set_study_user_attrs(study_id, to_rustuna_attrs({key: value}))

    def set_study_system_attr(
        self, study_id: int, key: str, value: JSONSerializable
    ) -> None:
        self._storage.set_study_system_attrs(study_id, to_rustuna_attrs({key: value}))

    def get_study_id_from_name(self, study_name: str) -> int:
        for study in self._storage.get_studies():
            if study.name == study_name:
                return study.id
        raise KeyError(f"Study({study_name=}) not found")

    def get_study_name_from_id(self, study_id: int) -> str:
        # TODO(c-bata): Raise KeyError if study not found.
        return self._storage.get_study(study_id).name

    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
        rustuna_directions = self._storage.get_study(study_id).directions
        return to_optuna_directions(rustuna_directions)

    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
        study = self._storage.get_study(study_id)
        return to_optuna_attrs(study.user_attrs)

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        study = self._storage.get_study(study_id)
        return to_optuna_attrs(study.system_attrs)

    def get_all_studies(self) -> list[FrozenStudy]:
        studies: list[FrozenStudy] = []
        for study in self._storage.get_studies():
            studies.append(to_frozen_study(study))
        return studies

    def create_new_trial(
        self, study_id: int, template_trial: FrozenTrial | None = None
    ) -> int:
        persisted_trial_template: rustuna.trial.PersistedTrial | None = None
        if template_trial is not None:
            for param_name, distribution in template_trial.distributions.items():
                if isinstance(
                    distribution, optuna.distributions.CategoricalDistribution
                ):
                    self._storage.set_category_labels(
                        study_id,
                        param_name,
                        list(distribution.choices),
                    )
            persisted_trial_template = to_persisted_trial(template_trial, study_id)
        trial = self._storage.create_new_trial(study_id, persisted_trial_template)
        return trial._trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        rustuna_distribution = to_rustuna_distribution(distribution)
        try:
            self._storage.set_trial_param(
                trial_id,
                param_name,
                rustuna_distribution,
                param_value_internal,
            )
        except rustuna.exceptions.UpdateFinishedTrialError as e:
            raise optuna.exceptions.UpdateFinishedTrialError(str(e)) from e

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        rustuna_state = to_rustuna_state(state)
        if values is None and state == TrialState.COMPLETE:
            values = [0.0]
        values = list(values) if values is not None else None

        try:
            self._storage.set_trial_state_values(trial_id, rustuna_state, values)
        except rustuna.exceptions.UpdateFinishedTrialError as e:
            raise optuna.exceptions.UpdateFinishedTrialError(str(e)) from e
        # TODO(c-bata): Consider adding an atomic state-claim API to prevent
        # multiple workers from claiming the same WAITING trial.
        return True

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        try:
            self._storage.set_trial_intermediate_value(
                trial_id, step, intermediate_value
            )
        except rustuna.exceptions.UpdateFinishedTrialError as e:
            raise optuna.exceptions.UpdateFinishedTrialError(str(e)) from e

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        try:
            self._storage.set_trial_user_attrs(trial_id, to_rustuna_attrs({key: value}))
        except rustuna.exceptions.UpdateFinishedTrialError as e:
            raise optuna.exceptions.UpdateFinishedTrialError(str(e)) from e

    def set_trial_system_attr(
        self, trial_id: int, key: str, value: JSONSerializable
    ) -> None:
        try:
            if key == _OPTUNA_CONSTRAINTS_KEY and isinstance(value, (list, tuple)):
                self._storage.set_trial_constraints(
                    trial_id,
                    {str(index): float(v) for index, v in enumerate(value)},
                )
                return
            if key == _RUSTUNA_CONSTRAINTS_KEY and isinstance(value, dict):
                self._storage.set_trial_constraints(
                    trial_id,
                    {str(name): float(v) for name, v in value.items()},
                )
                return
            self._storage.set_trial_system_attrs(
                trial_id, to_rustuna_attrs({key: value})
            )
        except rustuna.exceptions.UpdateFinishedTrialError as e:
            raise optuna.exceptions.UpdateFinishedTrialError(str(e)) from e

    def get_trial(self, trial_id: int) -> FrozenTrial:
        persisted_trial = self._storage.get_trial(trial_id)
        return to_frozen_trial(persisted_trial)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        rustuna_states: list[rustuna.trial.TrialState] | None = None
        if states is not None:
            assert isinstance(states, Iterable), (
                "ToOptunaStorage assumes that states is Iterable to make this faster"
            )
            states_list = list(states)
            if not states_list:
                return []
            rustuna_states = [to_rustuna_state(s) for s in states_list]
        rustuna_trials = self._storage.get_trials(study_id, states=rustuna_states)
        trials: list[FrozenTrial] = []
        for t in rustuna_trials:
            cached = self._trial_cache.get(t._trial_id)
            if cached is not None:
                trials.append(cached)
            else:
                frozen = FrozenTrialLike(t)
                if t.state.is_finished():
                    self._trial_cache[t._trial_id] = frozen
                trials.append(frozen)
        if deepcopy:
            return copy.deepcopy(trials)
        return trials

    def get_n_trials(
        self,
        study_id: int,
        state: tuple[TrialState, ...] | TrialState | None = None,
    ) -> int:
        if state is None:
            return self._storage.get_n_trials(study_id)
        states = state if isinstance(state, tuple) else (state,)
        return self._storage.get_n_trials(
            study_id,
            states=[to_rustuna_state(s) for s in states],
        )
