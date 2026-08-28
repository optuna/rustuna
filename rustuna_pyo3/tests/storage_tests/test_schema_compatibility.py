"""Test compatibility between Optuna's and Rustuna's storage schemas.

## Schema Differences between Optuna's JournalStorage and Rustuna's JournalStorage

Optuna's JournalStorage serializes attribute values as JSON before writing them to
the journal.  Consequently, a string value is JSON-encoded a second time in the
journal record.  Rustuna's Storage API accepts only string attributes, so Rustuna
stores its attribute maps directly and does not apply this extra JSON encoding.
Supporting both representations in both directions would not provide complete
compatibility, because Optuna can also write non-string values such as integers and
floats.  Rustuna therefore intentionally does not import Optuna's typed user and
system attribute values.

Rustuna writes string attributes to the Rustuna-specific ``user_attr_str`` and
``system_attr_str`` fields.  It also writes ``{"rustuna": null}`` to Optuna's
``user_attr`` or ``system_attr`` field.  The dummy value is required by Optuna's
journal replay code, while the Rustuna-specific fields are ignored by Optuna.  This
allows Optuna to replay a Rustuna journal without claiming that the attribute values
are fully interoperable.  The same rule applies to study and trial attributes.
"""

from __future__ import annotations

import os.path
import tempfile
import typing

import optuna
import pytest
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

import rustuna
from rustuna.converter import ToOptunaStorage, ToRustunaStorage


class Suite(typing.Protocol):
    directions: list[typing.Literal["minimize", "maximize"]]

    def objective(
        self, trial: optuna.Trial | rustuna.Trial
    ) -> float | tuple[float, float]: ...

    def assert_trials(
        self,
        trials: list[optuna.trial.FrozenTrial] | list[rustuna.trial.PersistedTrial],
    ) -> None: ...

    def assert_before_resume(
        self, trials: list[rustuna.trial.PersistedTrial]
    ) -> None: ...


class SuiteAttr:
    directions = ["minimize"]

    def objective(self, trial: optuna.Trial | rustuna.Trial) -> float:
        if isinstance(trial, optuna.Trial):
            trial.set_user_attr("optuna", {"a": 1})
            trial.storage.set_trial_system_attr(trial._trial_id, "optuna", {"b": 2})
        else:
            trial.set_user_attrs({"rustuna": "foo"})
            trial.storage.set_trial_system_attrs(trial._trial_id, {"rustuna": "bar"})
        return 1.0

    def assert_trials(
        self,
        trials: list[optuna.trial.FrozenTrial] | list[rustuna.trial.PersistedTrial],
    ) -> None:
        assert len(trials) == 20
        for trial in trials[-10:]:
            if isinstance(trial, optuna.trial.FrozenTrial):
                assert trial.user_attrs["optuna"] == {"a": 1}
                assert trial.system_attrs["optuna"] == {"b": 2}
            elif "optuna_attr:rustuna" in trial.user_attrs:
                assert trial.user_attrs["optuna_attr:rustuna"] == '"foo"'
                assert trial.system_attrs["optuna_attr:rustuna"] == '"bar"'
            else:
                assert trial.user_attrs["rustuna"] == "foo"
                assert trial.system_attrs["rustuna"] == "bar"

    def assert_before_resume(self, trials: list[rustuna.trial.PersistedTrial]) -> None:
        pass


class SuiteSingleObjective:
    directions = ["minimize"]

    def objective(self, trial: optuna.Trial | rustuna.Trial) -> float:
        return trial.suggest_float("x", 1.0, 10.0, log=True)

    def assert_trials(
        self,
        trials: list[optuna.trial.FrozenTrial] | list[rustuna.trial.PersistedTrial],
    ) -> None:
        assert len(trials) == 20
        for trial in trials:
            assert trial.state == (
                optuna.trial.TrialState.COMPLETE
                if isinstance(trial, optuna.trial.FrozenTrial)
                else rustuna.trial.TrialState.COMPLETE
            )
            assert set(trial.params) == {"x"}
            assert set(trial.distributions) == {"x"}
            x = trial.params["x"]
            assert isinstance(x, (int, float))
            assert 1.0 <= x <= 10.0
            assert trial.values == [x]

    def assert_before_resume(self, trials: list[rustuna.trial.PersistedTrial]) -> None:
        pass


class TestSuiteParam:
    directions = ["minimize"]

    def objective(self, trial: optuna.Trial | rustuna.Trial) -> float:
        x = trial.suggest_float("x", 1.0, 10.0, log=True)
        trial.suggest_int("y", 0, 10, step=2)
        trial.suggest_categorical("z", ["red", "green"])
        return x

    def assert_trials(
        self,
        trials: list[optuna.trial.FrozenTrial] | list[rustuna.trial.PersistedTrial],
    ) -> None:
        assert len(trials) == 20
        for trial in trials:
            assert trial.state == (
                optuna.trial.TrialState.COMPLETE
                if isinstance(trial, optuna.trial.FrozenTrial)
                else rustuna.trial.TrialState.COMPLETE
            )
            assert set(trial.params) == {"x", "y", "z"}
            assert set(trial.distributions) == {"x", "y", "z"}
            x = trial.params["x"]
            y = trial.params["y"]
            assert isinstance(x, (int, float))
            assert isinstance(y, int)
            assert 1.0 <= x <= 10.0
            assert y in {0, 2, 4, 6, 8, 10}
            assert trial.params["z"] in {"red", "green"}
            assert trial.values == [x]
            if isinstance(trial, optuna.trial.FrozenTrial):
                distribution = trial.distributions["z"]
                assert isinstance(
                    distribution, optuna.distributions.CategoricalDistribution
                )
                assert distribution.choices == ("red", "green")
            else:
                distribution = typing.cast(typing.Any, trial.distributions["z"])
                assert distribution.to_dict()["choices"] == [
                    "red",
                    "green",
                ]

    def assert_before_resume(self, trials: list[rustuna.trial.PersistedTrial]) -> None:
        for trial in trials:
            distribution = typing.cast(typing.Any, trial.distributions["z"])
            assert distribution.to_dict()["choices"] == [
                "red",
                "green",
            ]


class TestSuiteMultiObjective:
    directions = ["minimize", "maximize"]

    def objective(self, trial: optuna.Trial | rustuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 1.0, 10.0, log=True)
        y = trial.suggest_float("y", 0.0, 1.0)
        return x, y

    def assert_trials(
        self,
        trials: list[optuna.trial.FrozenTrial] | list[rustuna.trial.PersistedTrial],
    ) -> None:
        assert len(trials) == 20
        for trial in trials:
            assert trial.state == (
                optuna.trial.TrialState.COMPLETE
                if isinstance(trial, optuna.trial.FrozenTrial)
                else rustuna.trial.TrialState.COMPLETE
            )
            assert set(trial.params) == {"x", "y"}
            assert set(trial.distributions) == {"x", "y"}
            x = trial.params["x"]
            y = trial.params["y"]
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert 1.0 <= x <= 10.0
            assert 0.0 <= y <= 1.0
            assert trial.values is not None
            assert len(trial.values) == 2
            assert trial.values[0] == x
            assert trial.values[1] == y

    def assert_before_resume(self, trials: list[rustuna.trial.PersistedTrial]) -> None:
        pass


parametrize_test_suite = pytest.mark.parametrize(
    "suite",
    [
        pytest.param(SuiteAttr(), id="attrs"),
        pytest.param(SuiteSingleObjective(), id="single_objective"),
        pytest.param(TestSuiteParam(), id="params"),
        pytest.param(TestSuiteMultiObjective(), id="multi_objective"),
    ],
)


def assert_optuna_directions(
    directions: list[optuna.study.StudyDirection], suite: Suite
) -> None:
    expected = {
        "minimize": optuna.study.StudyDirection.MINIMIZE,
        "maximize": optuna.study.StudyDirection.MAXIMIZE,
    }
    assert directions == [expected[direction] for direction in suite.directions]


def assert_rustuna_directions(
    directions: list[rustuna.study.StudyDirection], suite: Suite
) -> None:
    expected = {
        "minimize": rustuna.study.StudyDirection.MINIMIZE,
        "maximize": rustuna.study.StudyDirection.MAXIMIZE,
    }
    assert directions == [expected[direction] for direction in suite.directions]


def get_optuna_storage(backend: str, base_dir: str) -> optuna.storages.BaseStorage:
    if backend == "journal":
        file_path = os.path.join(base_dir, "test.journal")
        return JournalStorage(JournalFileBackend(file_path))
    if backend == "sqlite3":
        file_path = os.path.join(base_dir, "test.sqlite3")
        return RDBStorage(f"sqlite:///{file_path}")
    raise ValueError(f"Unknown backend: {backend}")


def get_rustuna_storage(
    backend: str, base_dir: str, create_database: bool
) -> rustuna.storages.StorageProtocol:
    if backend == "journal":
        file_path = os.path.join(base_dir, "test.journal")
        return rustuna.storages.JournalFileStorage(file_path)
    if backend == "sqlite3":
        file_path = os.path.join(base_dir, "test.sqlite3")
        return rustuna.storages.SQLite3Storage(
            file_path, create_database=create_database
        )
    raise ValueError(f"Unknown backend: {backend}")


@parametrize_test_suite
@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
@pytest.mark.parametrize(
    "first_variant,second_variant",
    [
        ("direct", "via_to_optuna"),
        ("via_to_optuna", "direct"),
    ],
)
def test_optuna_api_resume_with_compat_storage(
    suite: Suite, backend: str, first_variant: str, second_variant: str
) -> None:
    if (
        backend == "sqlite3"
        and isinstance(suite, (SuiteAttr, TestSuiteParam))
        and first_variant == "via_to_optuna"
        and second_variant == "direct"
    ):
        pytest.skip("Optuna cannot directly read Rustuna's SQLite3 schema.")

    study_name = "compat-optuna"
    with tempfile.TemporaryDirectory() as workdir:
        # Start optimization via Optuna API
        if first_variant == "direct":
            first_storage = get_optuna_storage(backend, workdir)
        elif first_variant == "via_to_optuna":
            first_storage = ToOptunaStorage(
                get_rustuna_storage(backend, workdir, create_database=True)
            )
        else:
            raise ValueError(f"Unknown optuna variant: {first_variant}")

        first_study = optuna.create_study(
            storage=first_storage,
            study_name=study_name,
            directions=suite.directions,
            sampler=optuna.samplers.RandomSampler(),
        )
        assert_optuna_directions(first_study.directions, suite)
        first_study.optimize(suite.objective, n_trials=10)

        # Resume optimization via Optuna API
        if second_variant == "direct":
            second_storage = get_optuna_storage(backend, workdir)
        elif second_variant == "via_to_optuna":
            second_storage = ToOptunaStorage(
                get_rustuna_storage(backend, workdir, create_database=False)
            )
        else:
            raise ValueError(f"Unknown optuna variant: {second_variant}")

        second_study = optuna.load_study(
            storage=second_storage,
            study_name=study_name,
            sampler=optuna.samplers.RandomSampler(),
        )
        assert_optuna_directions(second_study.directions, suite)
        second_study.optimize(suite.objective, n_trials=10)

        suite.assert_trials(second_study.trials)


@parametrize_test_suite
@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
@pytest.mark.parametrize(
    "first_variant,second_variant",
    [
        ("direct", "via_to_rustuna"),
        ("via_to_rustuna", "direct"),
    ],
)
def test_rustuna_api_resume_with_compat_storage(
    suite: Suite, backend: str, first_variant: str, second_variant: str
) -> None:
    if (
        backend == "sqlite3"
        and isinstance(suite, (SuiteAttr, TestSuiteParam))
        and first_variant == "direct"
        and second_variant == "via_to_rustuna"
    ):
        pytest.skip("ToRustunaStorage cannot read Rustuna's raw SQLite3 attributes.")

    study_name = "compat-rustuna"
    with tempfile.TemporaryDirectory() as workdir:
        # Start optimization via Rustuna API
        first_storage: rustuna.storages.StorageProtocol
        if first_variant == "direct":
            first_storage = get_rustuna_storage(backend, workdir, create_database=True)
        elif first_variant == "via_to_rustuna":
            first_storage = ToRustunaStorage(get_optuna_storage(backend, workdir))
        else:
            raise ValueError(f"Unknown rustuna variant: {first_variant}")

        first_study = rustuna.create_study(
            storage=first_storage,
            study_name=study_name,
            directions=suite.directions,
            sampler=rustuna.samplers.RandomSampler(),
        )
        assert_rustuna_directions(first_study.directions, suite)
        first_study.optimize(suite.objective, n_trials=10)

        # Resume optimization via Rustuna API
        second_storage: rustuna.storages.StorageProtocol
        if second_variant == "direct":
            second_storage = get_rustuna_storage(
                backend, workdir, create_database=False
            )
        elif second_variant == "via_to_rustuna":
            second_storage = ToRustunaStorage(get_optuna_storage(backend, workdir))
        else:
            raise ValueError(f"Unknown rustuna variant: {second_variant}")

        second_study = rustuna.load_study(
            storage=second_storage,
            study_name=study_name,
            sampler=rustuna.samplers.RandomSampler(),
        )
        assert_rustuna_directions(second_study.directions, suite)
        second_study.optimize(suite.objective, n_trials=10)

        observed_study = rustuna.load_study(
            storage=second_storage,
            study_name=study_name,
            sampler=rustuna.samplers.RandomSampler(),
        )
        assert_rustuna_directions(observed_study.directions, suite)
        suite.assert_trials(observed_study.trials)


@parametrize_test_suite
@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
def test_optuna_to_rustuna_resume(suite: Suite, backend: str) -> None:
    study_name = "compat-optuna-to-rustuna"
    with tempfile.TemporaryDirectory() as workdir:
        optuna_storage = get_optuna_storage(backend, workdir)
        optuna_study = optuna.create_study(
            storage=optuna_storage,
            study_name=study_name,
            directions=suite.directions,
            sampler=optuna.samplers.RandomSampler(),
        )
        assert_optuna_directions(optuna_study.directions, suite)
        optuna_study.optimize(suite.objective, n_trials=10)

        rustuna_storage = get_rustuna_storage(backend, workdir, create_database=False)
        rustuna_study = rustuna.load_study(
            storage=rustuna_storage,
            study_name=study_name,
            sampler=rustuna.samplers.RandomSampler(),
        )
        assert_rustuna_directions(rustuna_study.directions, suite)
        suite.assert_before_resume(rustuna_study.trials)
        optuna_trial_count = len(optuna_study.trials)
        rustuna_study.optimize(suite.objective, n_trials=10)

        suite.assert_trials(rustuna_study.trials)
        if backend == "journal" and isinstance(suite, SuiteAttr):
            for trial in rustuna_study.trials[:10]:
                assert trial.user_attrs == {}
                assert trial.system_attrs == {}
        assert optuna_trial_count == 10


@parametrize_test_suite
@pytest.mark.parametrize("backend", ["journal", "sqlite3"])
def test_rustuna_to_optuna_resume(suite: Suite, backend: str) -> None:
    if backend == "sqlite3" and isinstance(suite, (SuiteAttr, TestSuiteParam)):
        pytest.skip("Optuna cannot read Rustuna's raw SQLite3 attributes.")

    study_name = "compat-rustuna-to-optuna"
    with tempfile.TemporaryDirectory() as workdir:
        rustuna_storage = get_rustuna_storage(backend, workdir, create_database=True)
        rustuna_study = rustuna.create_study(
            storage=rustuna_storage,
            study_name=study_name,
            directions=suite.directions,
            sampler=rustuna.samplers.RandomSampler(),
        )
        assert_rustuna_directions(rustuna_study.directions, suite)
        rustuna_study.optimize(suite.objective, n_trials=10)

        optuna_storage = get_optuna_storage(backend, workdir)
        optuna_study = optuna.load_study(
            storage=optuna_storage,
            study_name=study_name,
            sampler=optuna.samplers.RandomSampler(),
        )
        assert_optuna_directions(optuna_study.directions, suite)
        optuna_study.optimize(suite.objective, n_trials=10)

        suite.assert_trials(optuna_study.trials)
        if isinstance(suite, SuiteAttr):
            for trial in optuna_study.trials[:10]:
                assert trial.user_attrs == {"rustuna": None}
                assert trial.system_attrs["rustuna"] is None
