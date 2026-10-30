from __future__ import annotations

import pytest

import rustuna

from . import (
    TrialQueueFactory,
    TrialQueueType,
    parametrize_trial_queue,
)


@parametrize_trial_queue
def test_enqueue_trial_basic(queue_type: TrialQueueType) -> None:
    """Test basic enqueue_trial with float parameters."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 1.5, "y": 2.5})

        trial = study.ask()
        assert trial.suggest_float("x", 0.0, 10.0) == 1.5
        assert trial.suggest_float("y", 0.0, 10.0) == 2.5


@parametrize_trial_queue
def test_enqueue_trial_int(queue_type: TrialQueueType) -> None:
    """Test enqueue_trial with integer parameters."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 10, "y": 20})

        trial = study.ask()
        assert trial.suggest_int("x", 0, 100) == 10
        assert trial.suggest_int("y", 0, 100) == 20


@parametrize_trial_queue
def test_enqueue_trial_categorical(queue_type: TrialQueueType) -> None:
    """Test enqueue_trial with categorical parameters."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": "a", "y": "b"})

        trial = study.ask()
        assert trial.suggest_categorical("x", ["a", "b", "c"]) == "a"
        assert trial.suggest_categorical("y", ["a", "b", "c"]) == "b"


@parametrize_trial_queue
def test_enqueue_trial_mixed(queue_type: TrialQueueType) -> None:
    """Test enqueue_trial with mixed parameter types."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 1.5, "y": 10, "z": "a"})

        trial = study.ask()
        assert trial.suggest_float("x", 0.0, 10.0) == 1.5
        assert trial.suggest_int("y", 0, 100) == 10
        assert trial.suggest_categorical("z", ["a", "b", "c"]) == "a"


@parametrize_trial_queue
def test_multiple_enqueued_trials(queue_type: TrialQueueType) -> None:
    """Test multiple trials enqueued in FIFO order."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 1.0})
        study.enqueue_trial({"x": 2.0})
        study.enqueue_trial({"x": 3.0})

        trial1 = study.ask()
        assert trial1.suggest_float("x", 0.0, 10.0) == 1.0

        trial2 = study.ask()
        assert trial2.suggest_float("x", 0.0, 10.0) == 2.0

        trial3 = study.ask()
        assert trial3.suggest_float("x", 0.0, 10.0) == 3.0


@parametrize_trial_queue
def test_enqueue_fallback_on_out_of_range(queue_type: TrialQueueType) -> None:
    """Test that out-of-range enqueued values fall back to sampler."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 100.0})

        trial = study.ask()
        value = trial.suggest_float("x", 0.0, 10.0)
        assert 0.0 <= value <= 10.0


@parametrize_trial_queue
def test_enqueue_then_normal_ask(queue_type: TrialQueueType) -> None:
    """Test that normal sampling works after enqueued trials are exhausted."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 5.0})

        trial1 = study.ask()
        assert trial1.suggest_float("x", 0.0, 10.0) == 5.0
        study.tell(trial1.number, 1.0)

        trial2 = study.ask()
        value = trial2.suggest_float("x", 0.0, 10.0)
        assert 0.0 <= value <= 10.0
        study.tell(trial2.number, 2.0)


@parametrize_trial_queue
def test_enqueue_unspecified_param_sampled(queue_type: TrialQueueType) -> None:
    """Test that unspecified parameters are sampled normally."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 5.0})

        trial = study.ask()
        assert trial.suggest_float("x", 0.0, 10.0) == 5.0
        y = trial.suggest_float("y", 0.0, 10.0)
        assert 0.0 <= y <= 10.0


@parametrize_trial_queue
def test_enqueue_trial_with_user_attrs(queue_type: TrialQueueType) -> None:
    """Test enqueue_trial with user attributes."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)
        study.enqueue_trial({"x": 5.0}, user_attrs={"memo": "test"})

        trial = study.ask()
        assert trial.suggest_float("x", 0.0, 10.0) == 5.0
        assert trial.user_attrs == {"memo": "test"}


@parametrize_trial_queue
def test_queue_basic_enqueue_dequeue(queue_type: "TrialQueueType") -> None:
    """Test basic queue creation and enqueue/dequeue operations."""
    with TrialQueueFactory(queue_type) as factory:
        assert factory.queue is not None
        factory.queue.enqueue(1)
        factory.queue.enqueue(2)
        assert factory.queue.dequeue() == 1
        assert factory.queue.dequeue() == 2


@pytest.mark.parametrize(
    "queue_type",
    ["in_memory", "directory", "sqlite3"],
    ids=["InMemoryTrialQueue", "DirectoryTrialQueue", "SQLite3TrialQueue"],
)
def test_native_queue_empty_returns_none(queue_type: TrialQueueType) -> None:
    with TrialQueueFactory(queue_type) as factory:
        assert factory.queue is not None
        assert factory.queue.dequeue() is None


@parametrize_trial_queue
def test_enqueue_fifo_order_preserved(queue_type: "TrialQueueType") -> None:
    """Test that FIFO order is strictly preserved for enqueued trials."""
    with TrialQueueFactory(queue_type) as factory:
        study = rustuna.create_study(trial_queue=factory.queue)

        # Enqueue 10 trials with sequential values
        for i in range(10):
            study.enqueue_trial({"x": float(i)})

        # Verify they are returned in insertion order
        for i in range(10):
            trial = study.ask()
            assert trial.suggest_float("x", -100.0, 100.0) == float(i)
