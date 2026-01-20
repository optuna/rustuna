from __future__ import annotations

import copy
import json
import threading
import typing
import uuid
from collections.abc import Container, Sequence

import optuna
from optuna.distributions import BaseDistribution
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX, BaseStorage
from optuna.study import StudyDirection
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial, TrialState

import rustuna

from ._distribution import to_optuna_distribution, to_rustuna_distribution
from ._study import to_frozen_study, to_optuna_directions, to_persisted_study
from ._trial import (
    to_frozen_trial,
    to_optuna_state,
    to_persisted_trial,
    to_rustuna_state,
)

if typing.TYPE_CHECKING:
    from typing import Any

    from optuna._typing import JSONSerializable

    from rustuna import CategoricalChoiceType


logger = optuna.logging.get_logger(__name__)


class ToRustunaStorage:
    def __init__(self, storage: BaseStorage, is_distributed: bool = False) -> None:
        self._storage = storage
        self._is_distributed = is_distributed
        self._trial_id_to_study_id: dict[int, int] = {}
        self._lock = threading.Lock()

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

    def create_new_study(
        self, study_name: str, directions: list[rustuna.StudyDirection]
    ) -> rustuna.PersistedStudy:
        optuna_directions = to_optuna_directions(directions)
        study_id = self._storage.create_new_study(optuna_directions, study_name)
        return rustuna.PersistedStudy(
            id=study_id,
            name=study_name,
            directions=directions,
            user_attrs={},
            system_attrs={},
        )

    def create_new_trial(self, study_id: int) -> rustuna.PersistedTrial:
        trial_id = self._storage.create_new_trial(study_id)
        trial = self._storage.get_trial(trial_id)
        with self._lock:
            self._trial_id_to_study_id[trial_id] = study_id
        return to_persisted_trial(trial, study_id)

    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: rustuna.Distribution,
        value: float,
    ) -> None:
        self._storage.set_trial_param(
            trial_id, name, value, to_optuna_distribution(distribution)
        )

    def set_trial_state_values(
        self,
        trial_id: int,
        state: rustuna.TrialState,
        values: None | list[float] = None,
    ) -> None:
        self._storage.set_trial_state_values(
            trial_id,
            to_optuna_state(state),
            values=values,
        )

    def get_studies(self) -> list[rustuna.PersistedStudy]:
        frozen_studies = self._storage.get_all_studies()
        return [to_persisted_study(s) for s in frozen_studies]

    def get_study(self, study_id: int) -> rustuna.PersistedStudy:
        frozen_studies = self._storage.get_all_studies()
        for s in frozen_studies:
            if s._study_id == study_id:
                return to_persisted_study(s)
        raise KeyError(f"Study {study_id} not found")

    def get_trials(self, study_id: int) -> list[rustuna.PersistedTrial]:
        frozen_trials = self._storage.get_all_trials(study_id)

        persisted_trials: list[rustuna.PersistedTrial] = []
        with self._lock:
            for t in frozen_trials:
                persisted_trials.append(to_persisted_trial(t, study_id))
                self._trial_id_to_study_id[t._trial_id] = study_id
        return persisted_trials

    def get_trial(self, trial_id: int) -> rustuna.PersistedTrial:
        with self._lock:
            study_id = self._trial_id_to_study_id.get(trial_id, -1)
        if study_id == -1:
            logger.warning(
                f"Failed to get study_id while converting to PersistedTrial({trial_id=})"
                "due to the implementation restriction"
            )
        frozen_trial = self._storage.get_trial(trial_id)
        return to_persisted_trial(frozen_trial, study_id=study_id)

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

    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        for key, value in attrs.items():
            self._storage.set_trial_user_attr(trial_id, key, value)

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
    ) -> list[CategoricalChoiceType]:
        key = f"optuna_category_labels:{param_name}"
        value = self._storage.get_study_system_attrs(study_id).get(key)
        assert isinstance(value, list)
        assert len(value) == cardinality
        return typing.cast("list[CategoricalChoiceType]", value)


class ToOptunaStorage(BaseStorage):
    def __init__(self, storage: rustuna.StorageProtocol) -> None:
        self._storage = storage

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        rustuna_directions: list[rustuna.StudyDirection] = []
        for d in directions:
            if d == StudyDirection.MINIMIZE:
                rustuna_directions.append(rustuna.StudyDirection.MINIMIZE)
            elif d == StudyDirection.MAXIMIZE:
                rustuna_directions.append(rustuna.StudyDirection.MAXIMIZE)
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
        self._storage.set_study_user_attrs(study_id, {key: json.dumps(value)})

    def set_study_system_attr(
        self, study_id: int, key: str, value: JSONSerializable
    ) -> None:
        self._storage.set_study_system_attrs(study_id, {key: json.dumps(value)})

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
        return {key: json.loads(value) for key, value in study.user_attrs.items()}

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        study = self._storage.get_study(study_id)
        return {key: json.loads(value) for key, value in study.system_attrs.items()}

    def get_all_studies(self) -> list[FrozenStudy]:
        studies: list[FrozenStudy] = []
        for study in self._storage.get_studies():
            studies.append(to_frozen_study(study))
        return studies

    def create_new_trial(
        self, study_id: int, template_trial: FrozenTrial | None = None
    ) -> int:
        trial = self._storage.create_new_trial(study_id)
        trial_id = trial.id

        if template_trial is None:
            return trial_id
        if template_trial.user_attrs:
            user_attrs = {
                k: json.dumps(v) for k, v in template_trial.user_attrs.items()
            }
            self._storage.set_trial_user_attrs(trial_id, user_attrs)
        if template_trial.system_attrs:
            system_attrs = {
                k: json.dumps(v) for k, v in template_trial.system_attrs.items()
            }
            self._storage.set_trial_system_attrs(trial_id, system_attrs)
        if template_trial.distributions and template_trial.params:
            for param_name in template_trial.distributions:
                optuna_distribution = template_trial.distributions[param_name]
                distribution = to_rustuna_distribution(optuna_distribution)
                value = optuna_distribution.to_internal_repr(
                    template_trial.params[param_name]
                )
                self._storage.set_trial_param(trial_id, param_name, distribution, value)
        if template_trial.intermediate_values:
            for step in template_trial.intermediate_values:
                self._storage.set_trial_intermediate_value(
                    trial_id, step, template_trial.intermediate_values[step]
                )
        rustuna_state = to_rustuna_state(template_trial.state)
        self._storage.set_trial_state_values(
            trial_id, rustuna_state, template_trial.values
        )
        return trial_id

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
        # TODO(c-bata): Add support for pop waiting trial
        return False

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
            self._storage.set_trial_user_attrs(trial_id, {key: json.dumps(value)})
        except rustuna.exceptions.UpdateFinishedTrialError as e:
            raise optuna.exceptions.UpdateFinishedTrialError(str(e)) from e

    def set_trial_system_attr(
        self, trial_id: int, key: str, value: JSONSerializable
    ) -> None:
        try:
            self._storage.set_trial_system_attrs(trial_id, {key: json.dumps(value)})
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
        rustuna_trials = self._storage.get_trials(study_id)
        trials = [to_frozen_trial(t) for t in rustuna_trials]
        if states is not None:
            trials = [t for t in trials if t.state in states]
        if deepcopy:
            return copy.deepcopy(trials)
        return trials
