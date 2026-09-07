import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import rustuna
from rustuna.trial import TrialState


@pytest.mark.parametrize("apply_discard", [True, False])
def test_cmaes_sampler(apply_discard: bool) -> None:
    storage = rustuna.storages.InMemoryStorage(apply_discard=apply_discard)
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler, storage=storage)

    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    study.optimize(objective, n_trials=10)

    if not apply_discard:
        assert len(study.get_trials(states=[TrialState.COMPLETE])) == 10


def test_cmaes_sampler_samples_independently_from_multiple_threads() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1)
    storage = rustuna.storages.InMemoryStorage()
    study = rustuna.create_study(sampler=sampler, storage=storage)
    ctx = rustuna.samplers.SamplerContext(
        study_id=study._study_id,
        trial_number=0,
        trial_id=0,
        directions=study.directions,
    )
    distribution = rustuna.distributions.FloatDistribution(-1, 1)

    def sample(_: int) -> float:
        return sampler.sample_independent(ctx, storage, "x", distribution)

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(sample, range(100)))

    assert all(-1 <= value <= 1 for value in values)


def test_cmaes_sampler_with_failed_trials() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        if trial.number % 3 == 0:
            raise ValueError("failed trial")
        return x**2 + y**2

    study.optimize(objective, n_trials=9, catch=ValueError)

    states = [trial.state for trial in study.trials]
    assert states.count(rustuna.trial.TrialState.COMPLETE) == 6
    assert states.count(rustuna.trial.TrialState.FAIL) == 3


def test_cmaes_sampler_ask_and_tell_with_abandoned_trial() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def suggest(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    abandoned = study.ask()
    suggest(abandoned)  # This trial is never told.

    for _ in range(6):
        trial = study.ask()
        study.tell(trial.number, values=suggest(trial))

    completed = study.get_trials(states=[rustuna.trial.TrialState.COMPLETE])
    assert len(completed) == 6


def test_cmaes_sampler_is_a_concrete_class() -> None:
    sampler = rustuna.samplers.CmaEsSampler(seed=1)
    assert isinstance(sampler, rustuna.samplers.CmaEsSampler)

    study = rustuna.create_study(sampler=sampler)
    assert isinstance(study.sampler, rustuna.samplers.CmaEsSampler)


def test_cmaes_sampler_missing_dependency_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cmaes", None)
    sampler = rustuna.samplers.CmaEsSampler(seed=1, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    with pytest.raises(ImportError, match="cmaes"):
        study.optimize(objective, n_trials=2)


def test_cmaes_sampler_with_single_value_parameters() -> None:
    """Test that CMAES correctly handles single-value parameters.

    Single-value parameters should be included in the search space and
    correctly transformed/untransformed by SearchSpaceTransform.
    """
    sampler = rustuna.samplers.CmaEsSampler(seed=42, popsize=4)
    study = rustuna.create_study(sampler=sampler)

    def objective(trial: rustuna.Trial) -> float:
        # Normal parameters
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)

        # Single-value parameters (should be handled correctly)
        z = trial.suggest_float("z", 5.0, 5.0)  # single value: 5.0
        w = trial.suggest_int("w", 3, 3)  # single value: 3

        return x**2 + y**2 + z + w

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(
        trial.state == rustuna.trial.TrialState.COMPLETE for trial in study.trials
    )

    # Verify single-value parameters were always constant
    for trial in study.trials:
        z_val = trial.params["z"]
        w_val = trial.params["w"]
        assert z_val == 5.0, f"z should always be 5.0, got {z_val}"
        assert w_val == 3, f"w should always be 3, got {w_val}"
