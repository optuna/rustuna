import datetime
import enum
from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from typing import Any, Literal, TypedDict, TypeVar, overload

from ._protocols import (
    CachedStorageBackend,
    SamplerProtocol,
    StorageProtocol,
    TrialQueueProtocol,
)

_T = TypeVar("_T")

CategoricalChoiceType = float | int | str | bool | None
DistributionDict = (
    FloatDistributionDict | IntDistributionDict | CategoricalDistributionDict
)

FloatDistributionDict = TypedDict(
    "FloatDistributionDict",
    {
        "type": Literal["FloatDistribution"],
        "low": float,
        "high": float,
        "log": bool,
        "step": float | None,
    },
)
IntDistributionDict = TypedDict(
    "IntDistributionDict",
    {
        "type": Literal["IntDistribution"],
        "low": int,
        "high": int,
        "log": bool,
        "step": int | None,
    },
)
CategoricalDistributionDict = TypedDict(
    "CategoricalDistributionDict",
    {
        "type": Literal["CategoricalDistribution"],
        "choices": list[CategoricalChoiceType],
    },
)

# Distribution
class Distribution:
    """Parameter distribution for hyperparameter optimization."""

    @classmethod
    def float(
        cls, low: float, high: float, log: bool = False, step: float | None = None
    ) -> Distribution: ...
    @classmethod
    def int(
        cls, low: int, high: int, log: bool = False, step: int | None = None
    ) -> Distribution: ...
    @classmethod
    def categorical(cls, choices: list[CategoricalChoiceType]) -> Distribution: ...
    def to_dict(self) -> DistributionDict:
        """Convert the distribution to a dictionary.

        Returns:
            Dictionary representation of the distribution.
        """

# Trial
class Trial:
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the [Study.optimize][rustuna.study.Study.optimize] method; hence library users do not care about
    instantiation of this object.
    """

    _trial_id: int
    number: int
    study_id: int

    @property
    def storage(self) -> StorageProtocol:
        """Return the storage associated with this trial."""

    @property
    def user_attrs(self) -> dict[str, str]:
        """Return the user attributes."""

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Suggest a float parameter value.

        Args:
            name: A parameter name.
            low: Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high: Upper endpoint of the range of suggested values. ``high`` is included in the range.
            step: A discretization step. If specified, the parameter value is rounded to a multiple of this step.
            log: If True, suggest values from a log scale. This flag is incompatible with ``step``.

        Returns:
            A suggested float value.
        """
    def suggest_int(
        self, name: str, low: int, high: int, step: int | None = None, log: bool = False
    ) -> int:
        """Suggest an integer parameter value.

        Args:
            name: A parameter name.
            low: Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high: Upper endpoint of the range of suggested values. ``high`` is included in the range.
            step: A discretization step.
            log: If True, suggest values from a log scale. This flag is incompatible with ``step`` > 1.

        Returns:
            A suggested integer value.
        """
    def suggest_categorical(
        self, name: str, choices: list[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        """Suggest a categorical parameter value.

        Args:
            name: A parameter name.
            choices: Parameter value candidates.

        Returns:
            A suggested value.
        """
        ...
    def set_user_attr(self, key: str, value: str) -> None:
        """Set a user attribute to the trial.

        Note:
            Unlike Optuna, Rustuna accepts only str values for user attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.
        """
    def set_user_attrs(self, attrs: dict[str, str]) -> None:
        """Set user attributes to the trial.

        Note:
            Unlike Optuna, Rustuna accepts only str values for user attributes.

        Args:
            attrs: A dictionary object.
        """
    def set_constraint(self, key: str, value: float) -> None:
        """Set a constraint to the trial.

        See [Trial.set_constraints][rustuna.trial.Trial.set_constraints] for the detailed
        behavior of the constraints.

        Args:
            key: A key string of the constraint.
            value: A value of the constraint.

        Raises:
            RuntimeError: If `value` is NaN. Nothing is recorded in that case.
        """
    def set_constraints(self, constraints: dict[str, float]) -> None:
        """Set constraints to the trial.

        A trial is feasible when every one of its constraint values is zero or less, and
        infeasible when any of them is positive. How the infeasible trials are compared
        with each other is up to the sampler; see the documentation of each sampler that
        supports constraints, such as [TPESampler][rustuna.samplers.TPESampler] and
        [NSGAIISampler][rustuna.samplers.NSGAIISampler].

        A constraint name that is already set on the trial cannot be overwritten. If any of
        the names in `constraints` is already set, a warning is emitted and none of the
        values is recorded, not even the ones whose names are not set yet.

        Args:
            constraints: A dictionary object mapping each constraint name to its value.

        Raises:
            RuntimeError: If any of the values is NaN. The constraints are validated
                before anything is stored, so none of the values in `constraints` is
                recorded in that case.
        """

class AttrsDictView(Mapping[str, str]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __getitem__(self, key: str) -> str: ...
    @overload
    def get(self, key: str, default: None = ...) -> str | None: ...
    @overload
    def get(self, key: str, default: str) -> str: ...
    @overload
    def get(self, key: str, default: _T) -> str | _T: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[str]: ...
    def items(self) -> ItemsView[str, str]: ...
    def to_dict(self) -> dict[str, str]: ...

class TrialState(enum.IntEnum):
    """State of a trial."""

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    WAITING = 3
    FAIL = 4

    def is_finished(self) -> bool:
        """Return True if the trial state is a finished state.

        Returns:
            True if the trial state is COMPLETE, PRUNED, or FAIL.
        """

class PersistedTrial:
    """Status and results of a Trial.

    Attributes:
        number: Unique and consecutive number of Trial for each Study.
        study_id: An associated study's id.
        state: TrialState of the Trial.
        values: Sequence of objective values of the Trial.
        value: An objective value of the Trial (only available when single objective optimization).
        params: Dictionary that contains suggested parameters.
        distributions: Dictionary that contains the distributions of params.
        intermediate_values: Dictionary that contains reported intermediate values.
        user_attrs: Dictionary that contains the attributes of the Trial set with set_user_attr.
        system_attrs: Dictionary that contains the attributes of the Trial set with set_system_attr.
        internal_params: Dictionary that contains internal representations of the parameters.
        datetime_start: Datetime where the Trial started, as timezone-naive local time.
        datetime_complete: Datetime where the Trial finished, as timezone-naive local time.
    """

    def __init__(
        self,
        *,
        trial_id: int,
        study_id: int,
        number: int,
        state: TrialState,
        value: float | None = None,
        values: list[float] | None = None,
        params: dict[str, CategoricalChoiceType] | None = None,
        distributions: dict[str, Distribution] | None = None,
        intermediate_values: dict[int, float] | None = None,
        user_attrs: dict[str, str] | None = None,
        system_attrs: dict[str, str] | None = None,
        datetime_start: datetime.datetime | None = None,
        datetime_complete: datetime.datetime | None = None,
    ) -> None: ...
    @property
    def _trial_id(self) -> int: ...
    @property
    def study_id(self) -> int: ...
    @property
    def number(self) -> int: ...
    @property
    def state(self) -> TrialState: ...
    @property
    def values(self) -> list[float] | None: ...
    @property
    def value(self) -> float | None: ...
    @property
    def distributions(self) -> dict[str, Distribution]: ...
    @property
    def intermediate_values(self) -> dict[int, float]: ...
    @property
    def user_attrs(self) -> AttrsDictView: ...
    @property
    def system_attrs(self) -> AttrsDictView: ...
    @property
    def constraints(self) -> dict[str, float]: ...
    @property
    def internal_params(self) -> dict[str, float]: ...
    @property
    def params(self) -> dict[str, CategoricalChoiceType]: ...
    @property
    def datetime_start(self) -> datetime.datetime | None: ...
    @property
    def datetime_complete(self) -> datetime.datetime | None: ...
    def get_user_attr(
        self,
        key: str,
        *,
        decoder: Callable[[str], Any] | None = None,
        default: Any = None,
    ) -> Any:
        """Get a single user attribute value by key.

        ``PersistedTrial`` internally may hold a reference to the storage backend
        rather than a copy of all attribute data.  Accessing ``trial.user_attrs``
        therefore triggers a full attribute fetch from the storage.  This method
        retrieves only the requested key, which is both faster and uses less
        memory when you need just one attribute.

        It also provides a decoder callable and a default value for missing
        keys, simplifying a common migration pattern from Optuna.

        Args:
            key: The attribute key to look up.
            decoder: An optional callable to transform the stored string value
                (e.g. ``int``, ``float``, ``json.loads``).  When ``None``, the
                raw string is returned.
            default: Value to return when the key does not exist.
                Defaults to None.

        Returns:
            The attribute value (transformed by *decoder* if provided), or
            *default* if the key is not found.
        """

# Study
ObjectiveFuncType = Callable[[Trial], float | tuple[float, ...]]

def create_trial(
    *,
    state: TrialState = TrialState.COMPLETE,
    value: float | None = None,
    values: Sequence[float] | None = None,
    params: dict[str, Any] | None = None,
    distributions: dict[str, Distribution] | None = None,
    intermediate_values: dict[int, float] | None = None,
    user_attrs: dict[str, str] | None = None,
    system_attrs: dict[str, str] | None = None,
) -> PersistedTrial:
    """Create a low-level PersistedTrial object.

    This is intended to mirror Optuna's ``create_trial`` helper for preparing
    trials that will later be passed to ``Study.add_trial``.

    Args:
        state: Trial state.
        value: Trial objective value. Must not be specified together with ``values``.
        values: Sequence of trial objective values. Must not be specified together
            with ``value``.
        params: Dictionary with suggested parameter values in external representation.
        distributions: Dictionary with parameter distributions.
        user_attrs: Dictionary with user attributes.
        system_attrs: Dictionary with system attributes.

    Returns:
        A PersistedTrial object.
    """

def create_study(
    *,
    study_name: str | None = None,
    storage: StorageProtocol | None = None,
    sampler: SamplerProtocol | None = None,
    direction: Literal["minimize"] | Literal["maximize"] | StudyDirection | None = None,
    directions: Sequence[Literal["minimize"] | Literal["maximize"] | StudyDirection]
    | None = None,
    load_if_exists: bool = False,
    trial_queue: TrialQueueProtocol | None = None,
) -> Study:
    """Create a new study.

    Args:
        study_name: Study's name. If omitted, a unique name is generated automatically.
        storage: Storage object to persist study data. If None, InMemoryStorage is used.
        sampler: Sampler object for parameter suggestion. If None, TPESampler is used.
        direction: Direction of optimization. Either 'minimize', 'maximize', or a
            ``StudyDirection``.
            Cannot be specified together with ``directions``.
        directions: Directions of optimization for multi-objective optimization.
            Each direction can be specified as 'minimize', 'maximize', or a
            ``StudyDirection``.
            Cannot be specified together with ``direction``.
        load_if_exists: If True, return an existing study when ``study_name`` already exists.
        trial_queue: Trial queue object for managing trial execution order. If None,
            InMemoryTrialQueue is used.

    Returns:
        A Study object.
    """

def load_study(
    *,
    study_name: str | None = None,
    storage: StorageProtocol | None = None,
    sampler: SamplerProtocol | None = None,
    trial_queue: TrialQueueProtocol | None = None,
) -> Study:
    """Load an existing study.

    Args:
        study_name: Study's name. If None, the most recently created study is loaded.
        storage: Storage object. If None, raises an error.
        sampler: Sampler object for parameter suggestion. If None, TPESampler is used.
        trial_queue: Trial queue object for managing trial execution order. If None,
            InMemoryTrialQueue is used.
    """

def copy_study(
    *,
    from_study_name: str,
    from_storage: StorageProtocol,
    to_storage: StorageProtocol,
    to_study_name: str | None = None,
) -> None:
    """Copy a study to another storage."""

class PedAnovaImportanceEvaluator:
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm.

    PED-ANOVA fits Parzen estimators to completed trials in the top
    `target_quantile` fraction. The importance can be interpreted as how important each
    hyperparameter is for achieving performance within that fraction.

    For further information about the PED-ANOVA algorithm, please refer to the following paper:

    - [PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces](https://arxiv.org/abs/2304.10255) (IJCAI 2023)

    For further information on how conditional parameters are handled, please refer to the
    following paper:

    - [Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces](https://arxiv.org/abs/2601.20800) (KDD 2026)

    `target_quantile` and `region_quantile` correspond to the parameters
    $\\gamma'$ and $\\gamma$ in the original paper, respectively.

    Note:
        The performance of PED-ANOVA depends on how many trials to consider above
        `target_quantile`. To stabilize the analysis, it is preferable to include at least
        5 trials above `target_quantile`.

        Please also refer to the original implementations:

        - [PED-ANOVA](https://github.com/nabenabe0928/local-anova)
        - [condPED-ANOVA](https://github.com/kAIto47802/condPED-ANOVA)

    Args:
        target_quantile:
            Compute the importance of achieving a top-`target_quantile` objective value.
            For example, `target_quantile=0.1` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

        region_quantile:
            Define the region where we compute the importance. For example,
            `region_quantile=0.5` means that we compute the importance in the region where
            trials achieve top-50% performance. If `region_quantile=1.0`, the importance is
            computed in the whole search space.

        evaluate_on_local:
            Whether to measure the importance in the local or global space. If `True`,
            the importances indicate how important each parameter is during optimization.
            Meanwhile, `evaluate_on_local=False` gives the importances in the whole search
            space and `region_quantile` has no effect. `evaluate_on_local=True` is
            especially useful when users modify the search space during optimization.

    Example:
        An example of using PED-ANOVA is as follows:

        ```python
        import rustuna
        from rustuna.importance import PedAnovaImportanceEvaluator


        def objective(trial: rustuna.trial.Trial) -> float:
            x1 = trial.suggest_float("x1", -10, 10)
            x2 = trial.suggest_float("x2", -10, 10)
            return x1 + x2 / 1000


        study = rustuna.create_study()
        study.optimize(objective, n_trials=100)
        evaluator = PedAnovaImportanceEvaluator()
        importance = rustuna.importance.get_param_importances(study, evaluator=evaluator)
        ```

    """

    def __init__(
        self,
        *,
        target_quantile: float = 0.1,
        region_quantile: float = 1.0,
        evaluate_on_local: bool = True,
    ) -> None: ...
    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[PersistedTrial], float] | None = None,
    ) -> dict[str, float]:
        """Evaluate parameter importances based on completed trials in the given study.

        Note:
            This method is not meant to be called by library users. Use
            [get_param_importances][rustuna.importance.get_param_importances] to evaluate
            parameter importances from user code.

        Args:
            study:
                An optimized study.
            params:
                A list of names of parameters to assess. If `None`, all parameters that appear
                in completed trials, including conditional parameters, are assessed.
            target:
                A function that returns the value used to evaluate importances. If `None`,
                objective values are used for single-objective optimization. For multi-objective
                optimization, this argument must be specified to return a single float value for
                each trial. `PedAnovaImportanceEvaluator` assumes lower `target` values are better.

        Returns:
            A `dict` where the keys are parameter names and the values are assessed importances.

        """

def get_param_importances(
    study: Study,
    *,
    evaluator: PedAnovaImportanceEvaluator | None = None,
    params: list[str] | None = None,
    target: Callable[[PersistedTrial], float] | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Evaluate parameter importances using PED-ANOVA based on completed trials in the given study.

    The parameter importances are returned as a dictionary whose keys are parameter names and
    whose values are their importances.
    The importances are represented by non-negative floating point numbers, where higher values
    mean that the parameters are more important.
    By default, the sum of the importance values is normalized to 1.0.

    By default, this function uses `PedAnovaImportanceEvaluator`.
    For details on this evaluator, please refer to the following papers:

    - [PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces](https://arxiv.org/abs/2304.10255) (IJCAI 2023)
    - [Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces](https://arxiv.org/abs/2601.20800) (KDD 2026)

    When using this evaluator in your project, please consider citing both papers.

    If `params` is `None`, all parameters that appear in completed trials are assessed,
    including conditional parameters. If `params` is specified, only the specified parameters
    are assessed.

    Note:
        If `params` is specified as an empty list, an empty dictionary is returned.

    Args:
        study:
            An optimized study.
        evaluator:
            A `PedAnovaImportanceEvaluator` object. If `None`, a default
            `PedAnovaImportanceEvaluator` is used.
        params:
            A list of names of parameters to assess. If `None`, all parameters that appear in
            completed trials are assessed, including conditional parameters.
        target:
            A function that returns the value used to evaluate importances.
            If `None`, objective values are used for single-objective optimization.
            For multi-objective optimization, this argument must be specified to return
            a single float value for each trial.
        normalize:
            A boolean option to specify whether the sum of the importance values should be
            normalized to 1.0.
            Defaults to `True`.

    Returns:
        A `dict` where the keys are parameter names and the values are assessed importances.

    Example:
        ```python
        import rustuna


        def objective(trial: rustuna.trial.Trial) -> float:
            x = trial.suggest_int("x", 0, 2)
            y = trial.suggest_float("y", -1.0, 1.0)
            z = trial.suggest_float("z", 0.0, 1.5)
            return x**2 + y**3 - z**4


        sampler = rustuna.samplers.RandomSampler(seed=42)
        study = rustuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=100)

        importances = rustuna.importance.get_param_importances(study)
        ```

    """

class Study:
    """A study corresponds to an optimization task, i.e., a set of trials.

    This object provides interfaces to run a new Trial, access trials' history,
    and manage the study's state.
    """
    def __init__(
        self,
        study_id: int,
        name: str,
        directions: list[StudyDirection],
        storage: StorageProtocol,
        sampler: SamplerProtocol,
    ) -> None: ...
    def ask(self) -> Trial:
        """Create a new trial from which hyperparameters can be suggested.

        This method is part of an alternative to [Study.optimize][rustuna.study.Study.optimize]
        that allows controlling the execution of trials from user code.

        Returns:
            A Trial object.
        """
    def tell(
        self,
        number: int,
        values: int | float | Sequence[int | float] | None = None,
        state: TrialState | None = None,
    ) -> PersistedTrial:
        """Finish a trial created with [Study.ask][rustuna.study.Study.ask].

        Args:
            number: Trial number returned by the trial.
            values: Objective value(s). If None, the trial is marked as failed.
            state: State to set on the trial. If None, COMPLETE is used when values is not None.

        Returns:
            A PersistedTrial object.
        """
    def optimize(
        self,
        objective: ObjectiveFuncType,
        n_trials: int,
        catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    ) -> None:
        """Optimize an objective function.

        Args:
            objective: A callable that takes a Trial object and returns a float value or a sequence of float values.
            n_trials: The number of trials to run.
            catch: Exception class or tuple of exception classes that should fail the trial and allow optimization to continue.
        """
    def enqueue_trial(
        self,
        params: dict[str, Any],
        user_attrs: dict[str, str] | None = None,
        # TODO(c-bata): Add support for skip_if_exists option
        # skip_if_exists: bool = False,
    ) -> None:
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters will be evaluated in your objective function.

        Args:
            params: Parameter values to pass your objective function.
            user_attrs: A dictionary of user attributes other than params.
        """
    def add_trial(self, trial: PersistedTrial) -> None:
        """Add trial to study.

        Args:
            trial: Trial to add.
        """
    def set_user_attr(
        self,
        key: str,
        value: str,
    ) -> None:
        """Set a user attribute to the study.

        Note:
            Unlike Optuna, Rustuna accepts only str values for user attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        Example:
            ```python
            import rustuna

            study = rustuna.create_study()
            study.set_user_attr("objective function", "quadratic function")

            assert study.user_attrs == {
                "objective function": "quadratic function",
            }
            ```
        """
    def set_user_attrs(
        self,
        attrs: dict[str, str],
    ) -> None:
        """Set user attributes to the study.

        Note:
            Unlike Optuna, Rustuna accepts only str values for user attributes.

        Args:
            attrs: A dictionary object.

        Example:
            ```python
            import rustuna

            study = rustuna.create_study()
            study.set_user_attrs({
                "objective function", "quadratic function"
            })

            assert study.user_attrs == {
                "objective function": "quadratic function",
            }
            ```
        """
    def get_user_attr(
        self,
        key: str,
        *,
        decoder: Callable[[str], Any] | None = None,
        default: Any = None,
    ) -> Any:
        """Get a single user attribute value by key.

        This method fetches only the specified attribute from the storage backend,
        avoiding the overhead of loading all user attributes via ``study.user_attrs``.
        It also provides a decoder callable and a default value for missing keys,
        simplifying a common migration pattern from Optuna.

        Args:
            key: The attribute key to look up.
            decoder: An optional callable to transform the stored string value
                (e.g. ``int``, ``float``, ``json.loads``).  When ``None``, the
                raw string is returned.
            default: Value to return when the key does not exist.
                Defaults to None.

        Returns:
            The attribute value (transformed by *decoder* if provided), or
            *default* if the key is not found.
        """
    def get_trials(
        self,
        *,
        states: Sequence[TrialState] | None = None,
    ) -> list[PersistedTrial]:
        """Return trials in the study filtered by states if specified."""
    @property
    def _study_id(self) -> int:
        """Return the study ID."""
    @property
    def study_name(self) -> str:
        """Return the study name."""
    @property
    def directions(self) -> list[StudyDirection]:
        """Return the optimization directions."""
    @property
    def user_attrs(self) -> dict[str, str]:
        """Return the user attributes."""

    @property
    def best_trial(self) -> PersistedTrial:
        """Return the best trial in the single-objective study."""
    @property
    def trials(self) -> list[PersistedTrial]:
        """Return all trials in the study"""
    @property
    def best_trials(self) -> list[PersistedTrial]:
        """Return the Pareto front trials in the multi-objective study."""
    @property
    def _storage(self) -> StorageProtocol:
        """Return the storage object."""
    @property
    def sampler(self) -> SamplerProtocol:
        """Return the storage object."""
    @property
    def trial_queue(self) -> TrialQueueProtocol:
        """Return the trial queue object."""

class StudyDirection(enum.IntEnum):
    """Direction of optimization."""

    MINIMIZE = 0
    MAXIMIZE = 1

class PersistedStudy:
    """Persisted study information.

    Attributes:
        id: Study ID.
        name: Study name.
        directions: Optimization directions.
        user_attrs: Dictionary of user attributes.
        system_attrs: Dictionary of system attributes.
    """
    def __init__(
        self,
        id: int,
        name: str,
        directions: list[StudyDirection],
        user_attrs: dict[str, str] | None = None,
        system_attrs: dict[str, str] | None = None,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def directions(self) -> list[StudyDirection]: ...
    @property
    def user_attrs(self) -> dict[str, str]: ...
    @property
    def system_attrs(self) -> dict[str, str]: ...

## Storage

class CachedStorage:
    """Wrap a Python CachedStorageBackend with Rustuna's in-memory cache."""
    def __init__(self, backend: CachedStorageBackend) -> None: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def delete_study(self, study_id: int) -> None: ...
    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial: ...
    def set_trial_param(
        self, trial_id: int, name: str, distribution: Distribution, value: float
    ) -> None: ...
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None: ...
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None: ...
    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: None | list[float] = None
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(
        self, study_id: int, *, states: list[TrialState] | None = None
    ) -> list[PersistedTrial]: ...
    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int: ...
    def get_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_cached_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int: ...
    def get_study_user_attr(self, study_id: int, key: str) -> str: ...
    def get_study_system_attr(self, study_id: int, key: str) -> str: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None: ...
    def discard_trials(self, trial_ids: list[int]) -> None: ...
    def may_omit_trials(self) -> bool: ...

class ToRustStorage:
    """Wrapper to convert a StorageProtocol implementation to Rust Storage trait.

    This class wraps a Python object implementing StorageProtocol and makes it
    usable as a Rust Storage trait implementation.

    Args:
        storage: A Python object implementing StorageProtocol.

    Note:
        This class is not intended for direct use by end users. It is used internally
        by rustuna converters (e.g., ToOptunaSampler) to bridge Python StorageProtocol
        implementations with Rust components.
    """

    def __init__(self, storage: StorageProtocol) -> None: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def delete_study(self, study_id: int) -> None: ...
    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial: ...
    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None: ...
    def set_trial_state_values(
        self,
        trial_id: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(
        self, study_id: int, *, states: list[TrialState] | None = None
    ) -> list[PersistedTrial]: ...
    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int: ...
    def get_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_cached_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int: ...
    def get_study_user_attr(self, study_id: int, key: str) -> str: ...
    def get_study_system_attr(self, study_id: int, key: str) -> str: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None: ...
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None: ...
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None: ...
    def discard_trials(self, trial_ids: list[int]) -> None: ...
    def may_omit_trials(self) -> bool: ...

class InMemoryStorage:
    """Create an in-memory storage.

    Args:
        apply_discard: If True, apply discard_trials() and omit discarded trials from subsequent reads.
    """
    def __init__(self, *, apply_discard: bool = False) -> None: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def delete_study(self, study_id: int) -> None: ...
    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial: ...
    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None: ...
    def set_trial_state_values(
        self,
        trial_id: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(
        self, study_id: int, *, states: list[TrialState] | None = None
    ) -> list[PersistedTrial]: ...
    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int: ...
    def get_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_cached_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int: ...
    def get_study_user_attr(self, study_id: int, key: str) -> str: ...
    def get_study_system_attr(self, study_id: int, key: str) -> str: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None: ...
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None: ...
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None: ...
    def discard_trials(self, trial_ids: list[int]) -> None: ...
    def may_omit_trials(self) -> bool: ...

class JournalFileStorage:
    """Create a Journal storage with its file backend.

    Args:
        file_path: Path to the journal log file.
        apply_discard: If True, apply discard operations when reading the storage. Journal logs are written regardless of this option.
    """
    def __init__(
        self,
        file_path: str,
        *,
        apply_discard: bool = False,
    ) -> None: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def delete_study(self, study_id: int) -> None: ...
    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial: ...
    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None: ...
    def set_trial_state_values(
        self,
        trial_id: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(
        self, study_id: int, *, states: list[TrialState] | None = None
    ) -> list[PersistedTrial]: ...
    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int: ...
    def get_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_cached_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int: ...
    def get_study_user_attr(self, study_id: int, key: str) -> str: ...
    def get_study_system_attr(self, study_id: int, key: str) -> str: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None: ...
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None: ...
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None: ...
    def discard_trials(self, trial_ids: list[int]) -> None: ...
    def may_omit_trials(self) -> bool: ...

class SQLite3Storage:
    """Create a SQLite3 storage.

    Args:
        file_path: Path to the SQLite3 database file.
        create_database: If True, initialize the database when it is missing.
        apply_discard: If True, omit discarded trials from subsequent reads. ``discard_trials()``
            marks the trials in the database regardless of this option, so a storage opened with
            ``apply_discard=False`` still records discards for other readers to apply. Discards
            need a Rustuna-specific column on the ``trials`` table; it is added by
            ``create_database``, and enabling this option on a database that lacks it raises an
            error instead of silently ignoring discards. Discards applied by another process are
            picked up on the next read, except when that process' clock lags behind.
    """
    def __init__(
        self,
        file_path: str,
        *,
        create_database: bool = True,
        apply_discard: bool = False,
    ) -> None: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def delete_study(self, study_id: int) -> None: ...
    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial: ...
    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None: ...
    def set_trial_state_values(
        self,
        trial_id: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(
        self, study_id: int, *, states: list[TrialState] | None = None
    ) -> list[PersistedTrial]: ...
    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int: ...
    def get_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_cached_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int: ...
    def get_study_user_attr(self, study_id: int, key: str) -> str: ...
    def get_study_system_attr(self, study_id: int, key: str) -> str: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None: ...
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None: ...
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None: ...
    def discard_trials(self, trial_ids: list[int]) -> None: ...
    def may_omit_trials(self) -> bool: ...

# Sampler
class SamplerContext:
    """Context information for samplers.

    Attributes:
        study_id: Study ID.
        trial_number: Trial number.
        directions: Optimization directions.
    """

    study_id: int
    trial_number: int
    trial_id: int
    directions: list[StudyDirection]

    def __init__(
        self,
        *,
        study_id: int,
        trial_number: int,
        trial_id: int,
        directions: list[StudyDirection],
    ) -> None: ...

class RandomSampler:
    """Sampler for random search.

    Args:
        seed: Random seed. If None, a random seed is used.
    """
    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None: ...
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        name: str,
        distribution: Distribution,
    ) -> float: ...
    def before_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
    ) -> None: ...
    def after_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        state: TrialState,
        values: list[float] | None = None,
    ) -> None: ...

class TPESampler:
    """Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

    On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) `l(x)` to
    the set of parameter values associated with the good objective values, and another GMM
    `g(x)` to the remaining parameter values. It chooses the parameter value `x` that
    maximizes the ratio `l(x)/g(x)`. For multi-objective optimization, it uses non-domination
    ranks and hypervolume contributions to determine good and poor observations.

    For further information about the TPE algorithm, please refer to the following papers:

    - [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
    - [Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures](http://proceedings.mlr.press/v28/bergstra13.pdf)
    - [Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance](https://arxiv.org/abs/2304.11127)

    For multi-objective TPE (MOTPE), please refer to the following papers:

    - [Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization Problems](https://doi.org/10.1145/3377930.3389817)
    - [Multiobjective Tree-Structured Parzen Estimator](https://doi.org/10.1613/jair.1.13188)

    Example:
        ```python
        import rustuna

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        sampler = rustuna.TPESampler(seed=42)
        study = rustuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=100)
        ```

    Args:
        seed: Seed for random number generator. If `None`, a random seed is used.
        n_startup_trials: The random sampling is used instead of the TPE algorithm until
            the given number of trials finish in the same study. Defaults to `10`.
        multivariate: If `True`, the multivariate TPE samples all parameters jointly, which is
            reported to outperform the independent TPE. If `False`, parameters are sampled
            independently. If `None` (the default), the mode is selected automatically to match
            Optuna: multivariate for single-objective studies and independent for multi-objective
            studies.

    Note:
        By default (`multivariate=None`) multivariate TPE is used for single-objective
        optimization and independent TPE is used for multi-objective optimization, matching
        Optuna's `TPESampler`. In multivariate mode, TPE samples all non-conditional parameters
        jointly, which is reported to outperform independent sampling. See
        [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](http://proceedings.mlr.press/v80/falkner18a.html)
        for more details.

    Note:
        Constraints set via [Trial.set_constraints][rustuna.trial.Trial.set_constraints] are
        taken into account when the observations are split into the good half, which `l(x)`
        is fitted to, and the poor half, which `g(x)` is fitted to. Feasible trials, whose
        constraint values are all zero or less, are always preferred over infeasible ones,
        so the good half is filled with the best feasible trials first, and only the
        remaining slots, if any, are filled with infeasible trials. The infeasible trials
        are ordered by their total violation, i.e. the sum of their positive constraint
        values, and the ones violating the constraints the least come first.

        Which feasible trials are the best is decided exactly as in the unconstrained case:
        by the objective value for single-objective studies, and by the non-domination rank
        and the hypervolume contribution for multi-objective ones.
    """
    def __init__(
        self,
        *,
        seed: int | None = None,
        n_startup_trials: int = 10,
        multivariate: bool | None = None,
    ) -> None: ...
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        name: str,
        distribution: Distribution,
    ) -> float: ...
    def before_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
    ) -> None: ...
    def after_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        state: TrialState,
        values: list[float] | None = None,
    ) -> None: ...

class NSGAIISampler:
    """Sampler using NSGA-II (Non-dominated Sorting Genetic Algorithm II) algorithm.

    NSGA-II is an evolutionary algorithm designed for multi-objective optimization. It maintains a
    population of candidates across generations, and uses non-dominated sorting to rank solutions
    and crowding distance to preserve diversity among Pareto-optimal solutions. Each generation,
    new candidates are generated via crossover and mutation of selected parents, and an elite
    selection strategy retains the best individuals for the next generation.

    For further information about the NSGA-II algorithm, please refer to the following paper:

    - [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://ieeexplore.ieee.org/document/996017)

    Example:
        ```python
        import rustuna

        def objective(trial):
            x = trial.suggest_float("x", -5, 5)
            y = trial.suggest_float("y", -5, 5)
            return x**2, y**2

        sampler = rustuna.NSGAIISampler(seed=42)
        study = rustuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=100)
        ```

    Args:
        seed: Seed for random number generator. If `None`, a random seed is used.
        population_size: Number of individuals in the population. Defaults to `50`.
        mutation_prob: Probability of mutating each parameter of a candidate.
        crossover_prob: Probability of performing crossover between two parents. Defaults to `0.9`.
        swapping_prob: Probability of swapping each parameter value during crossover.
            Defaults to `0.5`.

    Note:
        Constraints set via [Trial.set_constraints][rustuna.trial.Trial.set_constraints] are
        taken into account by replacing the dominance relation of the non-dominated sort with
        constrained domination. A trial is feasible when its constraint values are all zero
        or less, and its total violation is the sum of its positive constraint values. A
        trial `a` constrained-dominates a trial `b` when

        * both are feasible and `a` dominates `b` in the usual Pareto sense,
        * `a` is feasible and `b` is not, or
        * both are infeasible and the total violation of `a` is smaller than that of `b`.

        Feasible trials therefore always form the earlier fronts, and the infeasible ones
        are ranked by how much they violate the constraints. The crowding distance used
        within a front is unchanged.
    """
    def __init__(
        self,
        *,
        seed: int | None = None,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
    ) -> None: ...
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        name: str,
        distribution: Distribution,
    ) -> float: ...
    def before_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
    ) -> None: ...
    def after_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        state: TrialState,
        values: list[float] | None = None,
    ) -> None: ...

class CmaEsSampler:
    """Sampler using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.

    This sampler is backed by Python's `cmaes` package. The optimizer state is held only
    in memory. Therefore, sampler instances in separate processes optimize independently.

    Categorical parameters are not supported by CMA-ES. They are excluded from the joint
    search space and sampled independently.

    Args:
        seed: Random seed for CMA-ES. If `None`, the backend chooses a random seed.
        popsize: CMA-ES population size. If `None`, the backend default is used.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        popsize: int | None = None,
    ) -> None: ...
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        name: str,
        distribution: Distribution,
    ) -> float: ...
    def before_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
    ) -> None: ...
    def after_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        state: TrialState,
        values: list[float] | None = None,
    ) -> None: ...

# Trial Queue
class DirectoryTrialQueue:
    """A directory-based trial queue.

    This queue uses the filesystem to persist trial IDs and provides multi-process
    safety through atomic file operations. The queue is stored in two subdirectories
    under the base directory: 'pending/' for queued trials and 'processing/' for
    trials being processed.

    Args:
        base_dir: Base directory path for the queue. Should be study-specific
            (e.g., '{storage_dir}/queue/{study_id}/') to ensure isolation between studies.
    """

    def __init__(self, base_dir: str) -> None: ...
    def enqueue(self, trial_id: int) -> None: ...
    def dequeue(self) -> int | None: ...

class InMemoryTrialQueue:
    """An in-memory TrialQueue implementation.

    This queue stores trial IDs in memory and does not persist across process restarts.
    Suitable for single-process optimization or when persistence is not required.
    """

    def __init__(self) -> None: ...
    def enqueue(self, trial_id: int) -> None: ...
    def dequeue(self) -> int | None: ...

class SQLite3TrialQueue:
    """An SQLite3 based TrialQueue implementation.

    This queue uses SQLite to persist trial IDs with ACID guarantees. Multiple queues
    can share the same database file, with namespace used for isolation.

    Args:
        db_path: Path to the SQLite database file.
        namespace: Namespace to isolate trials for this queue.
    """

    def __init__(self, db_path: str, namespace: str) -> None: ...
    def enqueue(self, trial_id: int) -> None: ...
    def dequeue(self) -> int | None: ...
