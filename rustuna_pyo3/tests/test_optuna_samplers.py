from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

import optuna
import pytest
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from optuna.samplers import BaseSampler
from optuna.testing.pytest_samplers import (
    BasicSamplerTestCase,
    MultiObjectiveSamplerTestCase,
    RelativeSamplerTestCase,
)

import rustuna
from rustuna.converter import ToOptunaSampler

if TYPE_CHECKING:
    from rustuna._rustuna import Distribution


class TestTpeSampler(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: ToOptunaSampler(
            rustuna.samplers.TPESampler(n_startup_trials=0, multivariate=True)
        )


class TestNSGAIISampler(BasicSamplerTestCase, MultiObjectiveSamplerTestCase):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: ToOptunaSampler(rustuna.samplers.NSGAIISampler())


class TestQMCSampler(BasicSamplerTestCase, RelativeSamplerTestCase):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: ToOptunaSampler(rustuna.samplers.QMCSampler())

    @pytest.mark.parametrize("n_jobs", [1])
    def test_trial_relative_params(
        self, n_jobs: int, sampler: Callable[[], BaseSampler]
    ) -> None:
        # n_jobs > 1 is excluded because ToRustStorage deadlocks under concurrent access: its
        # Python-facing methods take the storage lock while holding the GIL, while the Rust-side
        # Storage implementation reaches back into Python while holding that same lock. This is
        # not specific to QMCSampler; TPESampler deadlocks the same way with enough threads and
        # trials. QMCSampler only makes it easier to hit because it touches the storage on every
        # joint sample to reserve a sequence index.
        super().test_trial_relative_params(n_jobs, sampler)


class TestCmaEsSampler(BasicSamplerTestCase, RelativeSamplerTestCase):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: ToOptunaSampler(rustuna.samplers.CmaEsSampler())

    def test_sample_relative_categorical(
        self, sampler: Callable[[], BaseSampler]
    ) -> None:
        # CmaEsSampler does not support categorical distributions. They are sampled
        # independently after being excluded from the relative search space.
        pass

    @pytest.mark.parametrize("x_distribution", [])
    def test_sample_relative_mixed(
        self, sampler: Callable[[], BaseSampler], x_distribution: BaseDistribution
    ) -> None:
        # CmaEsSampler only samples numerical parameters relatively. Categorical
        # parameters are excluded from the relative search space and sampled independently.
        pass


class RecordingSampler:
    @property
    def support_joint_sampling(self) -> bool:
        return False

    def __init__(self) -> None:
        self.after_trial_calls: list[
            tuple[int, int, rustuna.trial.TrialState, list[float] | None]
        ] = []

    def sample_joint(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]:
        raise AssertionError("sample_joint must not be called")

    def sample_independent(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        name: str,
        distribution: Distribution,
    ) -> float:
        distribution_dict = distribution.to_dict()
        if distribution_dict["type"] == "FloatDistribution":
            return distribution_dict["low"]
        if distribution_dict["type"] == "IntDistribution":
            return distribution_dict["low"]
        if distribution_dict["type"] == "CategoricalDistribution":
            return 0.0
        raise AssertionError("Unreachable code")

    def after_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        state: rustuna.trial.TrialState,
        values: Sequence[float] | None = None,
    ) -> None:
        self.after_trial_calls.append(
            (
                ctx.study_id,
                ctx.trial_number,
                state,
                list(values) if values is not None else None,
            )
        )


class FailingAfterTrialSampler(RecordingSampler):
    def after_trial(
        self,
        ctx: rustuna.samplers.SamplerContext,
        storage: rustuna.storages.StorageProtocol,
        state: rustuna.trial.TrialState,
        values: Sequence[float] | None = None,
    ) -> None:
        raise RuntimeError("after_trial failed")


def test_to_optuna_sampler_after_trial_is_called() -> None:
    sampler = RecordingSampler()
    study = optuna.create_study(sampler=ToOptunaSampler(sampler), direction="minimize")

    study.optimize(lambda trial: trial.suggest_float("x", -1.0, 1.0), n_trials=2)

    assert len(sampler.after_trial_calls) == 2
    for study_id, trial_number, state, values in sampler.after_trial_calls:
        assert study_id == study._study_id
        assert trial_number >= 0
        assert state == rustuna.trial.TrialState.COMPLETE
        assert values is not None
        assert len(values) == 1


def test_to_optuna_sampler_after_trial_failure_still_persists_trial() -> None:
    sampler = FailingAfterTrialSampler()
    study = optuna.create_study(sampler=ToOptunaSampler(sampler), direction="minimize")
    trial = study.ask()

    with pytest.raises(RuntimeError, match="after_trial failed"):
        study.tell(trial, 1.0)

    persisted = study.trials[0]
    assert persisted.state == optuna.trial.TrialState.COMPLETE
    assert persisted.values == [1.0]
