from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scipy.stats import qmc as scipy_qmc  # type: ignore[import-untyped]

import rustuna

if TYPE_CHECKING:
    from rustuna._rustuna import Distribution


class DummyIndependentSampler:
    def __init__(self) -> None:
        pass

    @property
    def support_joint_sampling(self) -> bool:
        return False

    def sample_joint(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]:
        assert False, "Unreachable code"

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
        elif dic["type"] == "IntDistribution":
            return dic["low"]
        elif dic["type"] == "CategoricalDistribution":
            return 0
        assert False, "Unreachable code"

    def after_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        state: rustuna.trial.TrialState,
        values: list[float] | None = None,
    ) -> None:
        return None


class DummyJointSampler:
    def __init__(self) -> None:
        pass

    @property
    def support_joint_sampling(self) -> bool:
        return True

    def sample_joint(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]:
        params = {}
        for name, distribution in search_space.items():
            dic = distribution.to_dict()
            if dic["type"] == "FloatDistribution":
                params[name] = dic["low"]
            elif dic["type"] == "IntDistribution":
                params[name] = dic["low"]
            elif dic["type"] == "CategoricalDistribution":
                params[name] = 0
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
        elif dic["type"] == "IntDistribution":
            return dic["low"]
        elif dic["type"] == "CategoricalDistribution":
            return 0
        assert False, "Unreachable code"

    def after_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        state: rustuna.trial.TrialState,
        values: list[float] | None = None,
    ) -> None:
        return None


class RecordingSampler(DummyIndependentSampler):
    def __init__(self) -> None:
        self.after_trial_calls: list[
            tuple[int, int, rustuna.trial.TrialState, list[float] | None]
        ] = []

    def after_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        state: rustuna.trial.TrialState,
        values: list[float] | None = None,
    ) -> None:
        self.after_trial_calls.append((ctx.study_id, ctx.trial_number, state, values))


class FailingAfterTrialSampler(DummyIndependentSampler):
    def after_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        state: rustuna.trial.TrialState,
        values: list[float] | None = None,
    ) -> None:
        raise RuntimeError("after_trial failed")


@pytest.mark.parametrize("sampler", [DummyIndependentSampler(), DummyJointSampler()])
def test_custom_sampler(sampler: rustuna.samplers.SamplerProtocol) -> None:
    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        value = (x - 2) ** 2 + (y + 5) ** 2
        return value

    study = rustuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)


def test_study_constructor_accepts_custom_sampler() -> None:
    storage = rustuna.storages.InMemoryStorage()
    persisted_study = storage.create_new_study(
        "direct-study", [rustuna.study.StudyDirection.MINIMIZE]
    )
    study = rustuna.study.Study(
        persisted_study.id,
        persisted_study.name,
        persisted_study.directions,
        storage,
        DummyIndependentSampler(),
    )

    study.optimize(lambda trial: trial.suggest_float("x", -1.0, 1.0), n_trials=1)

    assert study.trials[0].state == rustuna.trial.TrialState.COMPLETE


def test_custom_sampler_after_trial_is_called() -> None:
    sampler = RecordingSampler()

    study = rustuna.create_study(sampler=sampler)
    study.optimize(lambda trial: trial.suggest_float("x", -1.0, 1.0), n_trials=3)

    assert len(sampler.after_trial_calls) == 3
    for study_id, trial_number, state, values in sampler.after_trial_calls:
        assert study_id == study._study_id
        assert trial_number >= 0
        assert state == rustuna.trial.TrialState.COMPLETE
        assert values is not None
        assert len(values) == 1


def test_custom_sampler_after_trial_failure_still_persists_trial() -> None:
    study = rustuna.create_study(sampler=FailingAfterTrialSampler())
    trial = study.ask()

    with pytest.raises(RuntimeError, match="Failed to tell"):
        study.tell(trial.number, 1.0)

    persisted = study.trials[0]
    assert persisted.state == rustuna.trial.TrialState.COMPLETE
    assert persisted.values == [1.0]


@pytest.mark.parametrize("dimension", [1, 2, 3, 10, 40])
def test_qmc_sampler_reproduces_the_scipy_sobol_sequence(dimension: int) -> None:
    # Optuna's QMCSampler draws its points from scipy.stats.qmc.Sobol, so matching SciPy is what
    # makes the two samplers comparable. Names are zero-padded because Rustuna assigns Sobol'
    # dimensions in sorted parameter order.
    names = [f"p{i:03d}" for i in range(dimension)]

    def objective(trial: rustuna.Trial) -> float:
        # Suggesting over [0, 1] makes the parameter value the raw sequence coordinate.
        return sum(trial.suggest_float(name, 0.0, 1.0) for name in names)

    study = rustuna.create_study(sampler=rustuna.samplers.QMCSampler())
    study.optimize(objective, n_trials=17)

    # The first trial has no joint search space yet and falls back to independent sampling.
    expected = scipy_qmc.Sobol(d=dimension, scramble=False).random(16)
    sampled = [[trial.params[name] for name in names] for trial in study.trials[1:]]
    assert sampled == pytest.approx(expected)
