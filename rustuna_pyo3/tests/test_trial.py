import pytest

import rustuna


def test_create_trial() -> None:
    trial = rustuna.trial.create_trial(
        params={"x": 1.5, "y": "foo"},
        distributions={
            "x": rustuna.distributions.FloatDistribution(0.0, 10.0),
            "y": rustuna.distributions.CategoricalDistribution(["foo", "bar"]),
        },
        value=5.0,
        user_attrs={"user": "attr"},
        system_attrs={"system": "attr"},
        constraints={"c0": 1.0},
    )

    assert trial._trial_id == 0
    assert trial.study_id == 0
    assert trial.number == 0
    assert trial.state == rustuna.trial.TrialState.COMPLETE
    assert trial.values == [5.0]
    assert trial.params == {"x": 1.5, "y": "foo"}
    assert trial.user_attrs["user"] == "attr"
    assert trial.system_attrs["system"] == "attr"
    assert trial.constraints == {"c0": 1.0}
    assert trial.datetime_start is not None
    assert trial.datetime_complete is not None

    study = rustuna.create_study()
    study.add_trial(trial)
    persisted = study.trials[0]
    assert persisted.number == 0
    assert persisted.value == 5.0


def test_create_trial_distributions() -> None:
    trial = rustuna.trial.create_trial(
        params={"lr": 0.01, "n_layers": 3, "activation": "relu"},
        distributions={
            "lr": rustuna.distributions.FloatDistribution(1e-5, 1e-1),
            "n_layers": rustuna.distributions.IntDistribution(1, 5),
            "activation": rustuna.distributions.CategoricalDistribution(
                ["relu", "tanh", "sigmoid"]
            ),
        },
        value=0.95,
    )
    assert trial.state == rustuna.trial.TrialState.COMPLETE
    assert trial.params["lr"] == pytest.approx(0.01)
    assert trial.params["n_layers"] == 3
    assert trial.params["activation"] == "relu"
    assert trial.internal_params["lr"] == pytest.approx(0.01)
    assert trial.internal_params["n_layers"] == 3.0
    assert trial.internal_params["activation"] == 0.0  # index of "relu"


def test_create_trial_unknown_param_raises() -> None:
    with pytest.raises(Exception):
        rustuna.trial.create_trial(
            params={"x": 1.0, "unknown": 99.0},
            distributions={"x": rustuna.distributions.FloatDistribution(0.0, 10.0)},
            value=1.0,
        )


def test_persisted_trial_new_distributions() -> None:
    trial = rustuna.trial.PersistedTrial(
        trial_id=10,
        study_id=1,
        number=5,
        state=rustuna.trial.TrialState.COMPLETE,
        params={"lr": 0.001, "n_layers": 2, "activation": "tanh"},
        distributions={
            "lr": rustuna.distributions.FloatDistribution(1e-5, 1e-1),
            "n_layers": rustuna.distributions.IntDistribution(1, 5),
            "activation": rustuna.distributions.CategoricalDistribution(
                ["relu", "tanh", "sigmoid"]
            ),
        },
        values=[0.9, 0.8],
        user_attrs={"note": "test"},
        system_attrs={"sys": "val"},
    )
    assert trial._trial_id == 10
    assert trial.study_id == 1
    assert trial.number == 5
    assert trial.values == [0.9, 0.8]
    assert trial.params["lr"] == pytest.approx(0.001)
    assert trial.params["n_layers"] == 2
    assert trial.params["activation"] == "tanh"
    assert trial.internal_params["lr"] == pytest.approx(0.001)
    assert trial.internal_params["n_layers"] == 2.0
    assert trial.internal_params["activation"] == 1.0  # index of "tanh"
    assert trial.user_attrs["note"] == "test"
    assert trial.system_attrs["sys"] == "val"


def test_persisted_trial_new_value_and_values_error() -> None:
    with pytest.raises(Exception):
        rustuna.trial.PersistedTrial(
            trial_id=0,
            study_id=0,
            number=0,
            state=rustuna.trial.TrialState.COMPLETE,
            value=1.0,
            values=[1.0],
        )


def test_constraints() -> None:
    study = rustuna.create_study()

    def objective(trial: rustuna.Trial) -> float:
        trial.set_constraints({"c0": 5.0, "c1": 10.0})
        return 0.0

    study.optimize(objective, n_trials=1)
    assert study.trials[0].constraints["c0"] == 5.0
    assert study.trials[0].constraints["c1"] == 10.0
    assert not any(key.startswith("constraints") for key in study.trials[0].system_attrs)
