from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest
from optuna.storages import InMemoryStorage, JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend
from pytest import FixtureRequest

import rustuna
from rustuna.converter._storage import ToRustunaStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from rustuna.storages import StorageProtocol


@pytest.fixture(
    params=[
        "inmemory",
        "sqlite3",
        "journal-file",
        "optuna-inmemory",
        "optuna-rdb-sqlite3",
        "optuna-journal-file",
    ]
)
def storage(request: FixtureRequest) -> Generator[StorageProtocol, None, None]:
    if request.param == "inmemory":
        yield rustuna.storages.InMemoryStorage()
        return
    if request.param == "optuna-inmemory":
        yield ToRustunaStorage(InMemoryStorage())
        return
    if request.param == "optuna-rdb-sqlite3":
        yield ToRustunaStorage(RDBStorage("sqlite://"))
        return

    with tempfile.TemporaryDirectory() as workdir:
        if request.param == "sqlite3":
            file_path = f"{workdir}/test.db"
            yield rustuna.storages.SQLite3Storage(file_path, create_database=True)
        elif request.param == "optuna-journal-file":
            file_path = f"{workdir}/test.journal"
            yield ToRustunaStorage(JournalStorage(JournalFileBackend(file_path)))
        else:
            file_path = f"{workdir}/test.journal"
            yield rustuna.storages.JournalFileStorage(file_path)


def test_get_study_attr_methods(storage: StorageProtocol) -> None:
    study = storage.create_new_study(
        "example study", [rustuna.study.StudyDirection.MINIMIZE]
    )
    storage.set_study_user_attrs(
        study.id,
        {
            "study_user_attr": "study_user_attr",
        },
    )
    storage.set_study_system_attrs(
        study.id,
        {
            "study_system_attr": "study_system_attr",
        },
    )
    assert storage.get_study_user_attr(study.id, "study_user_attr") == "study_user_attr"
    assert (
        storage.get_study_system_attr(study.id, "study_system_attr")
        == "study_system_attr"
    )
    storage.set_study_user_attrs(
        study.id,
        {
            "study_user_attr": "updated",
        },
    )
    assert storage.get_study_user_attr(study.id, "study_user_attr") == "updated"


def test_study_get_user_attr_method(storage: StorageProtocol) -> None:
    study = rustuna.create_study(storage=storage)
    study.set_user_attr("key", 1)
    assert study.get_user_attr("key") == 1
    assert study.get_user_attr("not_found") is None
    assert study.get_user_attr("not_found", default=1) == 1


def test_persisted_trial_user_attrs_get(storage: StorageProtocol) -> None:
    study = rustuna.create_study(storage=storage)
    trial = study.ask()
    trial.set_user_attr("key", 1)
    study.tell(trial.number, 0.0)
    persisted = study.get_trials()[0]
    assert persisted.user_attrs.get("key") == 1
    assert persisted.user_attrs.get("not_found") is None
    assert persisted.user_attrs.get("not_found", 1) == 1


def test_get_trials_with_states_filter(storage: StorageProtocol) -> None:
    study = rustuna.create_study(storage=storage)

    completed = study.ask()
    running = study.ask()
    failed = study.ask()

    study.tell(completed.number, 1.0)
    study.tell(failed.number, state=rustuna.trial.TrialState.FAIL)

    completed_trials = storage.get_trials(
        study._study_id, states=[rustuna.trial.TrialState.COMPLETE]
    )
    assert len(completed_trials) == 1
    assert completed_trials[0].number == completed.number

    running_trials = storage.get_trials(
        study._study_id, states=[rustuna.trial.TrialState.RUNNING]
    )
    assert len(running_trials) == 1
    assert running_trials[0].number == running.number

    finished_trials = storage.get_trials(
        study._study_id,
        states=[rustuna.trial.TrialState.COMPLETE, rustuna.trial.TrialState.FAIL],
    )
    assert {trial.number for trial in finished_trials} == {
        completed.number,
        failed.number,
    }


def test_get_n_trials_with_states_filter(storage: StorageProtocol) -> None:
    study = rustuna.create_study(storage=storage)

    completed = study.ask()
    running = study.ask()
    failed = study.ask()

    study.tell(completed.number, 1.0)
    study.tell(failed.number, state=rustuna.trial.TrialState.FAIL)

    assert storage.get_n_trials(study._study_id) == 3
    assert (
        storage.get_n_trials(
            study._study_id,
            states=[rustuna.trial.TrialState.COMPLETE],
        )
        == 1
    )
    assert (
        storage.get_n_trials(
            study._study_id,
            states=[rustuna.trial.TrialState.RUNNING],
        )
        == 1
    )
    assert (
        storage.get_n_trials(
            study._study_id,
            states=[rustuna.trial.TrialState.COMPLETE, rustuna.trial.TrialState.FAIL],
        )
        == 2
    )
    assert storage.get_n_trials(study._study_id, states=[]) == 0
