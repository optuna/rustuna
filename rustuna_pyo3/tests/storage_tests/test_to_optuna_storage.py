from __future__ import annotations

import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from optuna.distributions import FloatDistribution
from optuna.exceptions import UpdateFinishedTrialError
from optuna.study import StudyDirection
from optuna.testing.pytest_storages import StorageTestCase, _setup_studies
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


def _skip_non_finite_attrs_on_journal(storage: BaseStorage) -> None:
    assert isinstance(storage, ToOptunaStorage)
    if isinstance(storage._storage, rustuna.storages.JournalFileStorage):
        pytest.skip(
            "Journal logs must be strict JSON, so NaN and Infinity are rejected."
        )


class TestRustunaStorage(StorageTestCase):
    def test_get_all_studies(self, storage: BaseStorage) -> None:
        expected_frozen_studies, _ = _setup_studies(
            storage, n_study=10, n_trial=10, seed=46
        )
        frozen_studies = storage.get_all_studies()
        assert len(frozen_studies) == len(expected_frozen_studies)
        for _, expected_frozen_study in expected_frozen_studies.items():
            frozen_study = next(
                s
                for s in frozen_studies
                if s.study_name == expected_frozen_study.study_name
            )
            assert frozen_study.direction == expected_frozen_study.direction
            assert frozen_study.study_name == expected_frozen_study.study_name
            assert frozen_study.user_attrs == expected_frozen_study.user_attrs
            # Rustuna stores categorical choices as internal system attributes.
            system_attrs = {
                key: value
                for key, value in frozen_study.system_attrs.items()
                if not key.startswith("category_labels:")
            }
            assert system_attrs == expected_frozen_study.system_attrs

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

    def test_set_and_get_study_user_attrs_for_floats(
        self, storage: BaseStorage
    ) -> None:
        _skip_non_finite_attrs_on_journal(storage)
        super().test_set_and_get_study_user_attrs_for_floats(storage)

    def test_set_and_get_trial_user_attr_for_floats(self, storage: BaseStorage) -> None:
        _skip_non_finite_attrs_on_journal(storage)
        super().test_set_and_get_trial_user_attr_for_floats(storage)

    def test_set_and_get_study_system_attrs_for_floats(
        self, storage: BaseStorage
    ) -> None:
        _skip_non_finite_attrs_on_journal(storage)
        super().test_set_and_get_study_system_attrs_for_floats(storage)

    def test_set_and_get_trial_system_attr_for_floats(
        self, storage: BaseStorage
    ) -> None:
        _skip_non_finite_attrs_on_journal(storage)
        super().test_set_and_get_trial_system_attr_for_floats(storage)

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
