from __future__ import annotations

from collections.abc import Container
from typing import Any

import optuna
from optuna import Study as _OptunaStudy

import rustuna

from ._storage import ToOptunaStorage
from ._trial import (
    to_frozen_trial,
    to_rustuna_state_map,
)


class ToOptunaStudy(_OptunaStudy):
    """Expose a Rustuna study through Optuna's ``Study`` interface.

    This adapter allows Optuna APIs that consume a study, such as visualization
    functions, to operate on a study created by Rustuna. Trial data is converted
    to Optuna's ``FrozenTrial`` representation, while study operations are
    forwarded to the underlying Rustuna storage through `ToOptunaStorage`.

    Args:
        study: The Rustuna study to expose through the Optuna API.

    Example:
        Use a Rustuna study with an Optuna visualization function.

        ```python
        import rustuna
        from optuna.visualization import plot_optimization_history
        from rustuna.converter import ToOptunaStudy

        def objective(trial: rustuna.Trial) -> float:
            # Define your objective function.
            return trial.suggest_float("x", -1, 1) ** 2

        rustuna_study = rustuna.create_study()
        rustuna_study.optimize(objective, n_trials=10)
        optuna_study = ToOptunaStudy(rustuna_study)
        fig = plot_optimization_history(optuna_study)
        ```
    """

    def __init__(self, study: rustuna.Study):
        self._rustuna_study = study
        super().__init__(
            study.study_name,
            ToOptunaStorage(study._storage),
        )

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Container[optuna.trial.TrialState] | None = None,
    ) -> list[optuna.trial.FrozenTrial]:
        rustuna_states: list[rustuna.trial.TrialState] | None = None
        if states is not None:
            rustuna_states = []
            for optuna_state, rustuna_state in to_rustuna_state_map.items():
                if optuna_state not in states:
                    continue
                rustuna_states.append(rustuna_state)
        trials = self._rustuna_study.get_trials(states=rustuna_states)
        return [to_frozen_trial(t, use_frozen_trial_like=not deepcopy) for t in trials]

    @property
    def user_attrs(self) -> dict[str, Any]:
        return self._rustuna_study.user_attrs
