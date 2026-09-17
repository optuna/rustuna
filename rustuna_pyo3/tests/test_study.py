import rustuna


def test_study_best_trial_minimize_direction():
    study = rustuna.create_study(direction="minimize")
    for value in [1.0, 9.0, 3.0]:
        trial = study.ask()
        study.tell(trial.number, value)

    assert study.best_trial.number == 0
    assert study.best_trial.value == 1.0


def test_study_best_trial_maximize_direction():
    study = rustuna.create_study(direction="maximize")
    for value in [1.0, 9.0, 3.0]:
        trial = study.ask()
        study.tell(trial.number, value)

    assert study.best_trial.number == 1
    assert study.best_trial.value == 9.0
