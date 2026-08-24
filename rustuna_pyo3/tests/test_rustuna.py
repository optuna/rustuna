import pytest

import rustuna


def test_load_study_with_trial_queue():
    storage = rustuna.storages.InMemoryStorage()
    created = rustuna.create_study(study_name="queued-study", storage=storage)

    template = rustuna.trial.PersistedTrial(
        trial_id=0,
        study_id=created._study_id,
        number=0,
        state=rustuna.trial.TrialState.WAITING,
        system_attrs={"fixed_params:x": "f:0x4014000000000000"},
    )
    trial = storage.create_new_trial(created._study_id, template)

    queue = rustuna.trial_queue.InMemoryTrialQueue()
    queue.enqueue(trial._trial_id)

    loaded = rustuna.load_study(
        study_name="queued-study",
        storage=storage,
        trial_queue=queue,
    )

    asked = loaded.ask()
    assert asked.number == trial.number
    assert asked.suggest_float("x", 0.0, 10.0) == 5.0
    loaded.trial_queue.enqueue(123)
    assert queue.dequeue() == 123


def test_study_trial_queue_property():
    queue = rustuna.trial_queue.InMemoryTrialQueue()
    study = rustuna.create_study(trial_queue=queue)

    study.trial_queue.enqueue(123)

    assert queue.dequeue() == 123


def test_optimize():
    study = rustuna.create_study()

    def objective(trial: rustuna.Trial):
        x = trial.suggest_float("x", -5.0, 5)
        y = trial.suggest_int("y", 0, 10)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert -5.0 <= x <= 5.0
        assert 0 <= y <= 10
        assert z in ["foo", "bar"]
        trial.set_user_attr("key1", "value1")
        assert trial.user_attrs == {"key1": "value1"}
        trial.set_user_attrs({"key1": "updated", "key2": "value2"})
        assert trial.user_attrs == {"key1": "updated", "key2": "value2"}
        return x * 2 + y

    study.optimize(objective, n_trials=10)

    assert len(study.best_trial.internal_params) == 3
    assert len(study.best_trial.distributions) == 3
    assert len(study.best_trial.params) == 3
    assert study.best_trial.value is not None


def test_trial_storage_inside_objective():
    storage = rustuna.storages.InMemoryStorage()
    study = rustuna.create_study(storage=storage)

    def objective(trial: rustuna.Trial):
        assert trial.storage is storage
        assert len(trial.storage.get_studies()) == 1
        return trial.suggest_float("x", -1.0, 1.0)

    study.optimize(objective, n_trials=1)


def test_concrete_inmemory_storage_class():
    storage = rustuna.storages.InMemoryStorage()
    study = rustuna.create_study(storage=storage)

    assert study._storage is storage
    assert len(storage.get_studies()) == 1


def test_study_get_trials_filters_by_states():
    study = rustuna.create_study()
    failed_trial = study.ask()
    completed_trial = study.ask()
    study.tell(failed_trial.number, state=rustuna.trial.TrialState.FAIL)
    study.tell(completed_trial.number, values=1.0)

    filtered = study.get_trials(
        states=[rustuna.trial.TrialState.FAIL, rustuna.trial.TrialState.COMPLETE]
    )

    assert len(filtered) == 2
    assert [trial.state for trial in filtered] == [
        rustuna.trial.TrialState.FAIL,
        rustuna.trial.TrialState.COMPLETE,
    ]


def test_optimize_catch_exception():
    study = rustuna.create_study()

    def objective(trial: rustuna.Trial):
        if trial.number % 2 == 0:
            raise ValueError("expected failure")
        return 1.0

    study.optimize(objective, n_trials=2, catch=ValueError)
    study.optimize(objective, n_trials=2, catch=(Exception, RuntimeError))

    assert len(study.trials) == 4
    assert study.trials[0].state == rustuna.trial.TrialState.FAIL
    assert study.trials[1].state == rustuna.trial.TrialState.COMPLETE
    assert study.trials[2].state == rustuna.trial.TrialState.FAIL
    assert study.trials[3].state == rustuna.trial.TrialState.COMPLETE


def test_optimize_reraises_uncaught_exception():
    study = rustuna.create_study()

    def objective(_trial: rustuna.Trial):
        raise ValueError("expected failure")

    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1)

    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1, catch=(RuntimeError,))

    assert study.trials[0].state == rustuna.trial.TrialState.FAIL


def test_optimize_trial_pruned():
    study = rustuna.create_study()

    def objective(_trial: rustuna.Trial):
        raise rustuna.exceptions.TrialPruned()

    study.optimize(objective, n_trials=1)

    assert study.trials[0].state == rustuna.trial.TrialState.PRUNED


def test_optimize_multi_objective():
    study = rustuna.create_study(directions=["minimize", "minimize"])

    def objective(trial):
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_int("y", 0, 3)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert 0 <= x <= 5.0
        assert 0 <= y <= 3
        assert z in ["foo", "bar"]

        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    study.optimize(objective, n_trials=10)

    trial = study.trials[-1]
    assert len(trial.internal_params) == 3
    assert len(trial.distributions) == 3
    assert len(trial.params) == 3
    assert len(trial.values) == 2


def test_create_study_with_study_direction():
    study = rustuna.create_study(direction=rustuna.study.StudyDirection.MAXIMIZE)
    assert study.directions == [rustuna.study.StudyDirection.MAXIMIZE]

    study = rustuna.create_study(
        directions=[
            rustuna.study.StudyDirection.MINIMIZE,
            rustuna.study.StudyDirection.MAXIMIZE,
        ]
    )
    assert study.directions == [
        rustuna.study.StudyDirection.MINIMIZE,
        rustuna.study.StudyDirection.MAXIMIZE,
    ]


def test_study():
    study = rustuna.create_study(study_name="example")

    assert study.study_name == "example"

    study.set_user_attr("key1", "value1")
    assert study.user_attrs == {"key1": "value1"}
    study.set_user_attrs({"key1": "updated", "key2": "value2"})
    assert study.user_attrs == {"key1": "updated", "key2": "value2"}

    assert len(study._storage.get_studies()) == 1


def test_study_get_user_attr():
    study = rustuna.create_study(study_name="test_get_user_attr")

    study.set_user_attr("name", "alice")
    assert study.get_user_attr("name") == "alice"

    assert study.get_user_attr("missing") is None
    assert study.get_user_attr("missing", default=False) is False
    assert study.get_user_attr("missing", default="fallback") == "fallback"

    import json

    study.set_user_attr("flag", json.dumps(True))
    assert study.get_user_attr("flag") == "true"
    assert study.get_user_attr("flag", decoder=json.loads) is True

    study.set_user_attr("config", json.dumps({"lr": 0.01}))
    assert study.get_user_attr("config", decoder=json.loads) == {"lr": 0.01}

    assert study.get_user_attr("no_key", decoder=json.loads, default=False) is False


def test_create_study_load_if_exists_true():
    storage = rustuna.storages.InMemoryStorage()
    first = rustuna.create_study(
        storage=storage,
        study_name="load-if-exists",
        direction="minimize",
    )
    second = rustuna.create_study(
        storage=storage,
        study_name="load-if-exists",
        direction="maximize",
        load_if_exists=True,
    )

    assert first._study_id == second._study_id
    assert second.directions == [rustuna.study.StudyDirection.MINIMIZE]


def test_create_study_load_if_exists_false():
    storage = rustuna.storages.InMemoryStorage()
    rustuna.create_study(storage=storage, study_name="load-if-exists")

    with pytest.raises(rustuna.exceptions.DuplicatedStudyError):
        rustuna.create_study(
            storage=storage,
            study_name="load-if-exists",
            load_if_exists=False,
        )


def test_copy_study():
    from_storage = rustuna.storages.InMemoryStorage()
    to_storage = rustuna.storages.InMemoryStorage()
    from_study = rustuna.create_study(
        storage=from_storage,
        study_name="copy-source",
        directions=["maximize", "minimize"],
    )
    from_study._storage.set_study_system_attrs(from_study._study_id, {"sys": "value"})
    from_study.set_user_attr("user", "value")
    from_study.optimize(
        lambda trial: (
            trial.suggest_float("x0", 0, 1),
            trial.suggest_float("x1", 0, 1),
        ),
        n_trials=3,
    )

    rustuna.copy_study(
        from_study_name="copy-source",
        from_storage=from_storage,
        to_storage=to_storage,
    )
    to_study = rustuna.load_study(study_name="copy-source", storage=to_storage)

    assert to_study.study_name == from_study.study_name
    assert to_study.directions == from_study.directions
    assert to_study.user_attrs == from_study.user_attrs
    assert (
        to_study._storage.get_study(to_study._study_id).system_attrs
        == from_study._storage.get_study(from_study._study_id).system_attrs
    )
    assert len(to_study.trials) == len(from_study.trials)


def test_copy_study_to_study_name():
    from_storage = rustuna.storages.InMemoryStorage()
    to_storage = rustuna.storages.InMemoryStorage()
    rustuna.create_study(storage=from_storage, study_name="foo")
    rustuna.create_study(storage=to_storage, study_name="foo")

    with pytest.raises(rustuna.exceptions.DuplicatedStudyError):
        rustuna.copy_study(
            from_study_name="foo",
            from_storage=from_storage,
            to_storage=to_storage,
        )

    rustuna.copy_study(
        from_study_name="foo",
        from_storage=from_storage,
        to_storage=to_storage,
        to_study_name="bar",
    )
    rustuna.load_study(study_name="bar", storage=to_storage)


def test_suggest_categorical():
    study = rustuna.create_study()

    def objective(trial):
        x = trial.suggest_categorical("x", ["foo", 1, 1.0, None, True, False])

        assert x in ["foo", 1, 1.0, None, True, False]
        return 0.0

    study.optimize(objective, n_trials=10)


def test_integer_objective_value():
    study = rustuna.create_study()

    def objective(trial):
        x = trial.suggest_int("x", 0, 10)

        return x

    study.optimize(objective, n_trials=10)


def test_importance():
    study = rustuna.create_study()

    def objective(trial):
        x = trial.suggest_float("x", -5.0, 5)
        y = trial.suggest_int("y", 0, 10)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert -5.0 <= x <= 5.0
        assert 0 <= y <= 10
        assert z in ["foo", "bar"]
        return x * 2 + y

    study.optimize(objective, n_trials=10)
    rustuna.importance.get_param_importances(study)


def test_ask():
    study = rustuna.create_study(study_name="example")
    trial = study.ask()

    # Suggest APIs
    x = trial.suggest_float("x", 0, 1)
    assert 0 <= x <= 1
    y = trial.suggest_int("y", 0, 10)
    assert 0 <= y <= 10
    z = trial.suggest_categorical("z", ["foo", "bar"])
    assert z in ["foo", "bar"]

    # User Attrs
    x = trial.suggest_float("x", 0, 1)
    trial.set_user_attr("foo", "bar")

    # Tell
    study.tell(trial.number, 1.0)


def test_trial_state():
    state = rustuna.trial.TrialState.COMPLETE
    assert state.is_finished()


def test_persisted_trial():
    trial = rustuna.trial.PersistedTrial(
        trial_id=2,
        study_id=1,
        number=2,
        state=rustuna.trial.TrialState.COMPLETE,
        values=[0.5],
    )
    assert trial.study_id == 1
    assert trial.number == 2
    assert trial.state.is_finished()
    assert trial.values == [0.5]

    pytest.raises(
        ValueError,
        lambda: rustuna.trial.PersistedTrial(
            trial_id=2, study_id=1, number=2, state=rustuna.trial.TrialState.COMPLETE
        ),
    )


def test_sample():
    samplers = [
        rustuna.samplers.TPESampler(),
        rustuna.samplers.RandomSampler(),
    ]
    for sampler in samplers:
        storage = rustuna.storages.InMemoryStorage()
        study = rustuna.create_study(sampler=sampler, storage=storage)
        trial = study.ask()
        ctx = rustuna.samplers.SamplerContext(
            study_id=study._study_id,
            trial_number=trial.number,
            trial_id=trial._trial_id,
            directions=study.directions,
        )
        value = sampler.sample_independent(
            ctx, storage, "x", rustuna.distributions.FloatDistribution(0, 1)
        )
        assert 0 <= value <= 1


def test_get_pareto_front():
    study = rustuna.create_study(directions=["minimize", "minimize"])

    def objective(trial):
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_int("y", 0, 3)
        z = trial.suggest_categorical("z", ["foo", "bar"])

        assert 0 <= x <= 5.0
        assert 0 <= y <= 3
        assert z in ["foo", "bar"]

        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    study.optimize(objective, n_trials=10)

    trials = study.best_trials
    assert len(trials) > 0
    assert len(trials) <= 10


def test_tell_infeasible_values_without_state_marks_trial_failed():
    study = rustuna.create_study()
    trial = study.ask()
    with pytest.warns(UserWarning, match="did not match the number of the objectives"):
        study.tell(trial.number, [1.0, 2.0])
    assert study.trials[trial.number].state == rustuna.trial.TrialState.FAIL

    # The study must remain fully usable afterwards.
    ok = study.ask()
    study.tell(ok.number, 1.0)
    assert study.best_trial.number == ok.number


def test_tell_infeasible_values_with_complete_state_raises():
    study = rustuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="did not match the number of the objectives"):
        study.tell(trial.number, [1.0, 2.0], state=rustuna.trial.TrialState.COMPLETE)
    assert study.trials[trial.number].state == rustuna.trial.TrialState.RUNNING


def test_tell_nan_values():
    study = rustuna.create_study()
    trial = study.ask()
    with pytest.warns(UserWarning, match="is not acceptable"):
        study.tell(trial.number, float("nan"))
    assert study.trials[trial.number].state == rustuna.trial.TrialState.FAIL

    trial = study.ask()
    with pytest.raises(ValueError, match="is not acceptable"):
        study.tell(trial.number, float("nan"), state=rustuna.trial.TrialState.COMPLETE)


def test_tell_empty_values_marks_trial_failed():
    study = rustuna.create_study()
    trial = study.ask()
    with pytest.warns(UserWarning, match="did not match the number of the objectives"):
        study.tell(trial.number, [])
    assert study.trials[trial.number].state == rustuna.trial.TrialState.FAIL


def test_optimize_with_infeasible_values_continues_with_failed_trials():
    study = rustuna.create_study()
    with pytest.warns(UserWarning, match="did not match the number of the objectives"):
        study.optimize(lambda t: (1.0, 2.0), n_trials=3)
    assert len(study.trials) == 3
    assert all(t.state == rustuna.trial.TrialState.FAIL for t in study.trials)


def test_best_trial_on_multi_objective_study_raises_without_breaking_study():
    study = rustuna.create_study(directions=["minimize", "minimize"])
    trial = study.ask()
    study.tell(trial.number, [1.0, 2.0])
    with pytest.raises(RuntimeError):
        study.best_trial
    # The storage lock must not be poisoned by the failure above.
    assert len(study.trials) == 1
    assert len(study.best_trials) == 1


def test_add_trial_with_mismatched_values_raises_value_error():
    study = rustuna.create_study()
    trial = rustuna.trial.create_trial(values=[1.0, 2.0])
    with pytest.raises(ValueError, match="different from the number of objectives"):
        study.add_trial(trial)
