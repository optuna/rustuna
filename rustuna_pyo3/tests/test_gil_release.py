"""Progress tests: the GIL must be released during Rust sampling/storage work.

Mechanism shared by both tests: a worker thread runs a busy loop of
suggest/ask/tell while the main thread tries to set a flag the worker
watches. If the worker held the GIL throughout the run (which takes well
under the 5s forced-switch interval), the main thread could not set the
flag until the worker finished, so the worker would never observe it. The
worker observing the flag mid-run therefore proves the main thread got the
GIL while the worker was inside the detached Rust work.
"""

import sys
import threading

import rustuna


def test_optimize_releases_gil() -> None:
    # The worker runs a busy optimize; its detach points (the internal
    # ask/tell and the suggest_* calls inside the objective) are what let
    # the main thread run mid-optimize.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(5.0)
    try:
        study = rustuna.create_study(
            sampler=rustuna.samplers.TPESampler(seed=3, n_startup_trials=10)
        )
        first_trial_started = threading.Event()
        progress: dict[str, int | bool | None] = {"flag": False, "seen_at": None}

        def objective(trial: rustuna.Trial) -> float:
            x = trial.suggest_float("x", -5.0, 5.0)
            y = trial.suggest_float("y", -5.0, 5.0)
            z = trial.suggest_int("z", 0, 100)
            first_trial_started.set()
            if progress["flag"] and progress["seen_at"] is None:
                progress["seen_at"] = trial.number
            return x * x + y * y + z

        # daemon: if a deadlock regression keeps the worker alive past the
        # join timeout below, the test fails but pytest can still exit.
        worker = threading.Thread(
            target=lambda: study.optimize(objective, n_trials=2000), daemon=True
        )
        worker.start()
        assert first_trial_started.wait(timeout=60)
        progress["flag"] = True  # only reachable mid-run if the GIL was released
        worker.join(timeout=120)
        assert not worker.is_alive()
        assert progress["seen_at"] is not None
    finally:
        sys.setswitchinterval(old_interval)


def test_ask_tell_release_gil() -> None:
    # Same mechanism, but driving the ask/tell pymethods directly — a
    # separate code path from the detach boundaries inside optimize.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(5.0)
    try:
        study = rustuna.create_study(
            sampler=rustuna.samplers.TPESampler(seed=3, n_startup_trials=10)
        )
        first_iteration_started = threading.Event()
        progress: dict[str, int | bool | None] = {"flag": False, "seen_at": None}

        def ask_tell_loop() -> None:
            for i in range(2000):
                trial = study.ask()
                x = trial.suggest_float("x", -5.0, 5.0)
                y = trial.suggest_float("y", -5.0, 5.0)
                study.tell(trial.number, x * x + y * y)
                first_iteration_started.set()
                if progress["flag"] and progress["seen_at"] is None:
                    progress["seen_at"] = i

        # daemon: see test_optimize_releases_gil.
        worker = threading.Thread(target=ask_tell_loop, daemon=True)
        worker.start()
        assert first_iteration_started.wait(timeout=60)
        progress["flag"] = True  # only reachable mid-run if the GIL was released
        worker.join(timeout=120)
        assert not worker.is_alive()
        assert progress["seen_at"] is not None
    finally:
        sys.setswitchinterval(old_interval)
