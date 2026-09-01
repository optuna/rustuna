from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import optuna
import pytest

import rustuna
from rustuna.converter import ToRustunaStorage

if TYPE_CHECKING:
    from rustuna._rustuna import Distribution


def test_optimize_with_optuna_storage():
    def objective(trial: rustuna.Trial | optuna.Trial) -> float:
        return trial.suggest_float("x", -10, 10) ** 2

    storage = ToRustunaStorage(optuna.storages.RDBStorage("sqlite://"))
    rustuna_study = rustuna.create_study(storage=storage)
    rustuna_study.optimize(objective, n_trials=10)


def test_duplicate_study_error_with_optuna_storage_bridge():
    storage = ToRustunaStorage(optuna.storages.InMemoryStorage())
    rustuna.create_study(storage=storage, study_name="dup-study")

    with pytest.raises(rustuna.exceptions.DuplicatedStudyError):
        rustuna.create_study(storage=storage, study_name="dup-study")


def test_resume_optimization():
    def objective(trial: rustuna.Trial | optuna.Trial) -> float:
        return trial.suggest_float("x", -10, 10) ** 2

    # Sample 10 trials with Optuna
    optuna_storage = optuna.storages.RDBStorage("sqlite://")
    optuna_study = optuna.create_study(storage=optuna_storage)
    optuna_study.optimize(objective, n_trials=10)
    assert len(optuna_study.trials) == 10

    # Resume the optimization with Rustuna
    rustuna_storage = ToRustunaStorage(optuna_storage)
    rustuna_study = rustuna.load_study(
        study_name=optuna_study.study_name, storage=rustuna_storage
    )
    rustuna_study.optimize(objective, n_trials=10)
    assert len(rustuna_study.trials) == 20

    # Resume the optimization with Optuna
    optuna_study = optuna.load_study(
        study_name=optuna_study.study_name, storage=optuna_storage
    )
    optuna_study.optimize(objective, n_trials=10)
    assert len(optuna_study.trials) == 30


def test_trial_datetimes_are_preserved_in_rustuna_storage_cache():
    optuna_storage = optuna.storages.InMemoryStorage()
    optuna_study = optuna.create_study(storage=optuna_storage)
    optuna_study.optimize(lambda _: 1.0, n_trials=1)

    rustuna_study = rustuna.load_study(
        study_name=optuna_study.study_name,
        storage=ToRustunaStorage(optuna_storage),
    )

    optuna_trial = optuna_study.trials[0]
    rustuna_trial = rustuna_study.trials[0]
    assert rustuna_trial.datetime_start == optuna_trial.datetime_start
    assert rustuna_trial.datetime_complete == optuna_trial.datetime_complete


def _run_optimize(sqlite3_filepath: str) -> None:
    optuna_storage = optuna.storages.RDBStorage(f"sqlite:///{sqlite3_filepath}")
    storage = ToRustunaStorage(optuna_storage)
    study = rustuna.load_study(storage=storage, study_name="test_study")
    study.optimize(lambda t: t.suggest_float("x", -10, 10) ** 2, n_trials=10)


def test_rdb_storage_sqlite3():
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
        storage = ToRustunaStorage(optuna.storages.RDBStorage(f"sqlite:///{f.name}"))
        study = rustuna.create_study(storage=storage, study_name="test_study")
        _run_optimize(f.name)
        assert len(storage.get_studies()) == 1


def test_multi_process():
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
        storage = ToRustunaStorage(optuna.storages.RDBStorage(f"sqlite:///{f.name}"))
        study = rustuna.create_study(storage=storage, study_name="test_study")

        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_run_optimize, f.name) for _ in range(5)]

        studies = storage.get_studies()
        assert len(studies) == 1
        trials = storage.get_trials(study_id=studies[0].id)
        assert len(trials) == 10 * 5


class DummyJointSampler:
    def __init__(self) -> None:
        self.sample_joint_is_called = False

    @property
    def support_joint_sampling(self) -> bool:
        return True

    def before_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
    ) -> None:
        return None

    def sample_joint(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]:
        # TODO(c-bata): Avoid calling sample_joint if the search space is empty.
        if ctx.trial_number == 0:
            return {}

        self.sample_joint_is_called = True
        params = {}
        for name, distribution in search_space.items():
            dic = distribution.to_dict()
            if dic["type"] == "FloatDistribution":
                params[name] = dic["low"]
            elif dic["type"] == "IntDistribution":
                params[name] = dic["low"]
            elif dic["type"] == "CategoricalDistribution":
                params[name] = 0.0
            else:
                assert False, "Unreachable code"
        return params

    def sample_independent(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        name: str,
        distribution: Distribution,
    ) -> float:
        dic = distribution.to_dict()
        if dic["type"] == "FloatDistribution":
            return dic["low"]
        if dic["type"] == "IntDistribution":
            return dic["low"]
        if dic["type"] == "CategoricalDistribution":
            return 0.0
        assert False, "Unreachable code"


def test_storage_cache_joint_search_space():
    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10.0, 10.0)
        y = trial.suggest_int("y", -10, 10)
        z = trial.suggest_categorical("z", ["A", 1, 0.5, None, False])
        return x**2 + y**2

    sampler = DummyJointSampler()
    storage = ToRustunaStorage(optuna.storages.RDBStorage("sqlite://"))
    rustuna_study = rustuna.create_study(storage=storage, sampler=sampler)
    rustuna_study.optimize(objective, n_trials=10)
    assert sampler.sample_joint_is_called
