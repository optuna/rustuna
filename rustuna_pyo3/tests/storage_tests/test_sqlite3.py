import tempfile

import optuna
import pytest
from optuna.storages import RDBStorage

import rustuna


def test_optimize_with_sqlite3() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.db"
        storage = rustuna.storages.SQLite3Storage(file_path, create_database=True)
        study = rustuna.create_study(storage=storage)

        def objective(trial: optuna.Trial | rustuna.Trial) -> float:
            x = trial.suggest_float("x", 1, 10, log=True)
            y = trial.suggest_int("y", -10, 10)
            trial.suggest_categorical("z", [True, False, "foo", 10])
            return x**2 + y

        study.optimize(objective, 10)
        assert len(study.trials) == 10


def test_use_optuna_db() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/test.db"

        # Create a database file
        RDBStorage(f"sqlite:///{file_path}")

        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        assert not storage.may_omit_trials()
        study = rustuna.create_study(storage=storage)

        def objective(trial: optuna.Trial | rustuna.Trial) -> float:
            x = trial.suggest_float("x", 1, 10, log=True)
            y = trial.suggest_int("y", -10, 10)
            trial.suggest_categorical("z", [True, False, "foo", 10])
            return x**2 + y

        study.optimize(objective, 10)
        assert len(study.trials) == 10


def test_sqlite3_storage_can_apply_discard() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        file_path = f"{workdir}/discarded.db"
        storage = rustuna.storages.SQLite3Storage(file_path, apply_discard=True)
        study = rustuna.create_study(storage=storage, study_name="example")

        first = study.ask()
        second = study.ask()
        first_persisted = study.tell(first.number, 1.0)
        second_persisted = study.tell(second.number, 2.0)

        storage.discard_trials([first_persisted._trial_id])

        omitted_trials = storage.get_trials(study._study_id)
        assert [trial._trial_id for trial in omitted_trials] == [
            second_persisted._trial_id
        ]
        with pytest.raises(rustuna.exceptions.TrialDiscarded, match="Trial discarded"):
            storage.get_trial(first_persisted._trial_id)

        reloaded_storage = rustuna.storages.SQLite3Storage(
            file_path, apply_discard=True
        )
        assert [
            trial._trial_id for trial in reloaded_storage.get_trials(study._study_id)
        ] == [second_persisted._trial_id]
