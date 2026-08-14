from __future__ import annotations

import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from optuna.distributions import FloatDistribution
from optuna.exceptions import UpdateFinishedTrialError
from optuna.study import StudyDirection
from optuna.testing.pytest_storages import StorageTestCase
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState
from pytest import FixtureRequest

import rustuna
from rustuna.converter import ToOptunaStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from optuna.storages import BaseStorage


@pytest.fixture(params=["sqlite3", "journal-file"])
def storage(request: FixtureRequest) -> Generator[BaseStorage, None, None]:
    with tempfile.TemporaryDirectory() as workdir:
        rustuna_storage: rustuna.storages.StorageProtocol
        if request.param == "sqlite3":
            file_path = f"{workdir}/test.db"
            rustuna_storage = rustuna.storages.SQLite3Storage(
                file_path, create_database=True
            )
        else:
            file_path = f"{workdir}/test.journal"
            rustuna_storage = rustuna.storages.JournalFileStorage(file_path)
        yield ToOptunaStorage(rustuna_storage)


class TestRustunaStorage(StorageTestCase):
    def test_delete_study(self, storage: BaseStorage) -> None:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        storage.create_new_trial(study_id)
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 1

        # TODO(c-bata): Check study_id
        # with pytest.raises(KeyError):
        #     # Deletion of non-existent study.
        #     storage.delete_study(study_id + 1)

        storage.delete_study(study_id)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 0

        # storage.delete_study(study_id)
        # with pytest.raises(KeyError):
        #     # Double free.
        #     storage.delete_study(study_id)

    def test_get_all_trials_uses_cache_diff(self, storage: BaseStorage) -> None:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id0 = storage.create_new_trial(study_id)
        trials1 = storage.get_all_trials(study_id)
        assert len(trials1) == 1
        assert {t._trial_id for t in trials1} == {trial_id0}

        trial_id1 = storage.create_new_trial(study_id)
        trials2 = storage.get_all_trials(study_id)
        assert len(trials2) == 2
        assert {t._trial_id for t in trials2} == {trial_id0, trial_id1}

    def test_constraints_use_optuna_system_attrs(self, storage: BaseStorage) -> None:
        assert isinstance(storage, ToOptunaStorage)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id = storage.create_new_trial(study_id)

        storage._storage.set_trial_constraints(
            trial_id, {"c1": -1.0, "c0": 1.0}
        )
        trial = storage.get_trial(trial_id)
        assert trial.system_attrs["constraints"] == [1.0, -1.0]
        assert "rustuna:constraints" not in trial.system_attrs

        optuna_trial_id = storage.create_new_trial(study_id)
        storage.set_trial_system_attr(optuna_trial_id, "constraints", [2.0, -2.0])
        rustuna_trial = storage._storage.get_trial(optuna_trial_id)
        assert rustuna_trial.constraints == {"0": 2.0, "1": -2.0}

    @pytest.mark.skip("Rustuna cannot store objective values for failed state")
    def test_get_trial(self, storage: BaseStorage) -> None:
        super().test_get_trial(storage)

    @pytest.mark.skip("Rustuna cannot store objective values for failed state")
    def test_get_all_trials(self, storage: BaseStorage) -> None:
        super().test_get_all_trials(storage)

    @pytest.mark.skip("Rustuna's params cannot support the order of params")
    @pytest.mark.parametrize("param_names", [["a", "b"], ["b", "a"]])
    def test_get_all_trials_params_order(
        self, storage: BaseStorage, param_names: list[str]
    ) -> None: ...

    @pytest.mark.skip("Rustuna storages do not support pickle serialization")
    def test_pickle_storage(self, storage: BaseStorage) -> None: ...
