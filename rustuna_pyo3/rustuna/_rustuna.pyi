import datetime
import enum
from collections.abc import Callable
from typing import Literal, Protocol, TypedDict

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
    ) -> Distribution:
        """Create a float distribution.

        Args:
            low: Lower bound.
            high: Upper bound.
            log: If True, sample from a log scale.
            step: Discretization step.

        Returns:
            A float distribution.
        """

    @classmethod
    def int(
        cls, low: int, high: int, log: bool = False, step: int | None = None
    ) -> Distribution:
        """Create an integer distribution.

        Args:
            low: Lower bound.
            high: Upper bound.
            log: If True, sample from a log scale.
            step: Discretization step.

        Returns:
            An integer distribution.
        """

    @classmethod
    def categorical(cls, choices: list[CategoricalChoiceType]) -> Distribution:
        """Create a categorical distribution.

        Args:
            choices: List of candidate values.

        Returns:
            A categorical distribution.
        """

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
    the :func:`Study.optimize` method; hence library users do not care about
    instantiation of this object.
    """

    id: int
    number: int
    study_id: int

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

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.
        """

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

    This object has the same methods as :class:`Trial`, but is not associated with,
    nor has any references to a :class:`Study`.

    Attributes:
        number: Unique and consecutive number of Trial for each Study.
        state: TrialState of the Trial.
        values: Sequence of objective values of the Trial.
        params: Dictionary that contains suggested parameters.
        distributions: Dictionary that contains the distributions of params.
        user_attrs: Dictionary that contains the attributes of the Trial set with set_user_attr.
        system_attrs: Dictionary that contains the attributes of the Trial set with set_system_attr.
        internal_params: Dictionary that contains internal representations of the parameters.
        datetime_start: Datetime where the Trial started.
        datetime_complete: Datetime where the Trial finished.
    """

    def __init__(
        self,
        trial_id: int,
        study_id: int,
        number: int,
        state: TrialState,
        values: list[float] | None = None,
        internal_params: dict[str, float] | None = None,
        distributions: dict[str, Distribution] | None = None,
        user_attrs: dict[str, str] | None = None,
        system_attrs: dict[str, str] | None = None,
        datetime_start: datetime.datetime | None = None,
        datetime_complete: datetime.datetime | None = None,
        intermediate_values: dict[int, float] | None = None,
        id: int | None = None,
    ) -> None: ...
    @property
    def id(self) -> int: ...
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
    def distributions(self) -> dict[str, Distribution]: ...
    @property
    def user_attrs(self) -> dict[str, str]: ...
    @property
    def system_attrs(self) -> dict[str, str]: ...
    @property
    def internal_params(self) -> dict[str, float]: ...
    @property
    def params(self) -> dict[str, CategoricalChoiceType]: ...
    @property
    def datetime_start(self) -> datetime.datetime | None: ...
    @property
    def datetime_complete(self) -> datetime.datetime | None: ...
    @property
    def intermediate_values(self) -> dict[int, float]: ...

# Study
ObjectiveFuncType = Callable[[Trial], float | tuple[float, ...]]

def create_study(
    *,
    study_name: str | None = None,
    storage: Storage | StorageProtocol | None = None,
    sampler: Sampler | SamplerProtocol | None = None,
    direction: Literal["minimize"] | Literal["maximize"] | None = None,
    directions: list[Literal["minimize"] | Literal["maximize"]] | None = None,
) -> Study:
    """Create a new study.

    Args:
        study_name: Study's name. If omitted, a unique name is generated automatically.
        storage: Storage object to persist study data. If None, InMemoryStorage is used.
        sampler: Sampler object for parameter suggestion. If None, TPESampler is used.
        direction: Direction of optimization. Either 'minimize' or 'maximize'.
            Cannot be specified together with ``directions``.
        directions: Directions of optimization for multi-objective optimization.
            Cannot be specified together with ``direction``.

    Returns:
        A Study object.
    """

def load_study(
    *,
    study_name: str | None = None,
    storage: Storage | StorageProtocol | None = None,
    sampler: Sampler | SamplerProtocol | None = None,
) -> Study:
    """Load an existing study.

    Args:
        study_name: Study's name. If None, the most recently created study is loaded.
        storage: Storage object. If None, raises an error.
        sampler: Sampler object for parameter suggestion. If None, TPESampler is used.
    """

def get_param_importance(study: Study) -> list[list[float]]: ...

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
        storage: Storage,
        sampler: Sampler,
    ) -> None: ...
    def ask(self) -> Trial:
        """Create a new trial from which hyperparameters can be suggested.

        This method is part of an alternative to :func:`~Study.optimize` that allows controlling
        the execution of trials from user code.

        Returns:
            A Trial object.
        """
    def tell(
        self,
        number: int,
        values: float | None = None,
        state: TrialState | None = None,
    ) -> Trial:
        """Finish a trial created with :func:`~Study.ask`.

        Args:
            number: Trial number returned by the trial.
            values: Objective value(s). If None, the trial is marked as failed.
            state: State to set on the trial. If None, COMPLETE is used when values is not None.

        Returns:
            A Trial object.
        """
    def optimize(self, objective: ObjectiveFuncType, n_trials: int) -> None:
        """Optimize an objective function.

        Args:
            objective: A callable that takes a Trial object and returns a float value or a sequence of float values.
            n_trials: The number of trials to run.
        """
    @property
    def id(self) -> int:
        """Return the study ID."""
    @property
    def directions(self) -> list[StudyDirection]:
        """Return the optimization directions."""

    @property
    def best_trial(self) -> PersistedTrial:
        """Return the best trial in the single-objective study."""
    @property
    def trials(self) -> list[PersistedTrial]:
        """Return all trials in the study"""
    @property
    def best_trials(self) -> list[PersistedTrial]:
        """Return the Pareto front trials in the multi-objective study."""

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
class StorageProtocol(Protocol):
    """Protocol for storage implementations.

    This protocol defines the interface that storage backends must implement
    to persist optimization history.
    """
    @property
    def is_distributed(self) -> bool: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def delete_study(self, study_id: int) -> None: ...
    def create_new_trial(self, study_id: int) -> PersistedTrial: ...
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
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(self, study_id: int) -> list[PersistedTrial]: ...
    def get_trial(self, trial_id: int) -> PersistedTrial: ...
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None: ...
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
    ) -> list[CategoricalChoiceType]: ...

class Storage:
    """Storage for persisting optimization history."""

    @classmethod
    def in_memory(cls) -> StorageProtocol:
        """Create an in-memory storage.

        Returns:
            An in-memory storage instance.
        """
    @classmethod
    def sqlite3(
        cls, file_path: str, *, create_database: bool = False
    ) -> StorageProtocol:
        """Create a SQLite3 storage.

        Args:
            file_path: Path to the SQLite3 database file.
            create_database: If True, create the database file if it does not exist.

        Returns:
            A SQLite3 storage instance.
        """
    @classmethod
    def journal_file(
        cls,
        file_path: str,
    ) -> StorageProtocol:
        """Create a Journal storage with its file backend.

        Args:
            file_path: Path to the journal log file.

        Returns:
            A Journal storage instance.
        """
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy:
        """Create a new study.

        Args:
            study_name: Name of the study.
            directions: Optimization directions.

        Returns:
            The created study.
        """
    def delete_study(self, study_id: int) -> None:
        """Delete a study.

        Args:
            study_id: ID of the study to delete.
        """
    def create_new_trial(self, study_id: int) -> PersistedTrial:
        """Create a new trial in the specified study.

        Args:
            study_id: ID of the study.

        Returns:
            The created trial.
        """
    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None:
        """Set a parameter value for a trial.

        Args:
            trial_id: ID of the trial.
            name: Parameter name.
            distribution: Parameter distribution.
            value: Internal representation of the parameter value.
        """
    def set_trial_state_values(
        self,
        trial_id: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None:
        """Set the state and values of a trial.

        Args:
            trial_id: ID of the trial.
            state: New state of the trial.
            values: Objective values (required when state is COMPLETE).
        """
    def get_studies(self) -> list[PersistedStudy]:
        """Get all studies.

        Returns:
            List of all studies.
        """
    def get_study(self, study_id: int) -> PersistedStudy:
        """Get a study by ID.

        Args:
            study_id: ID of the study.

        Returns:
            The study.
        """
    def get_trials(self, study_id: int) -> list[PersistedTrial]:
        """Get all trials in a study.

        Args:
            study_id: ID of the study.

        Returns:
            List of all trials in the study.
        """
    def get_n_trials(self, study_id: int) -> int:
        """Get the number of trials in a study.

        Args:
            study_id: ID of the study.

        Returns:
            Number of trials.
        """
    def get_trial(self, trial_id: int) -> PersistedTrial:
        """Get a trial by ID.

        Args:
            trial_id: ID of the trial.

        Returns:
            The trial.
        """
    def get_trial_by_number(self, study_id: int, trial_number: int) -> PersistedTrial:
        """Get a trial by study ID and trial number.

        Args:
            study_id: ID of the study.
            trial_number: Number of the trial within the study.

        Returns:
            The trial.
        """
    def get_trial_id_from_study_id_trial_number(
        self, study_id: int, trial_number: int
    ) -> int:
        """Get a trial ID from study ID and trial number.

        Args:
            study_id: ID of the study.
            trial_number: Number of the trial within the study.

        Returns:
            ID of the trial.
        """
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        """Set system attributes of a study.

        Args:
            study_id: ID of the study.
            attrs: System attributes to set.
        """
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        """Set user attributes of a study.

        Args:
            study_id: ID of the study.
            attrs: User attributes to set.
        """
    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        """Set system attributes of a trial.

        Args:
            trial_id: ID of the trial.
            attrs: System attributes to set.
        """
    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        """Set user attributes of a trial.

        Args:
            trial_id: ID of the trial.
            attrs: User attributes to set.
        """
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        choices: list[CategoricalChoiceType],
    ) -> None:
        """Set category labels for a categorical parameter.

        Args:
            study_id: ID of the study.
            param_name: Name of the categorical parameter.
            choices: List of category labels.
        """
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
        cardinality: int,
    ) -> list[CategoricalChoiceType] | None:
        """Get category labels for a categorical parameter.

        Args:
            study_id: ID of the study.
            param_name: Name of the categorical parameter.
            cardinality: Number of categories.

        Returns:
            List of category labels, or None if not set.
        """
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        """Set an intermediate value for a trial.

        Args:
            trial_id: ID of the trial.
            step: Step at which the intermediate value is reported.
            intermediate_value: Intermediate objective value.
        """

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

class SamplerProtocol(Protocol):
    """Protocol for sampler implementations.

    This protocol defines the interface that samplers must implement
    to suggest parameter values.
    """
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: Storage,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: Storage,
        name: str,
        distribution: Distribution,
    ) -> float: ...

class Sampler:
    """Factory class for creating sampler instances."""

    @classmethod
    def tpe(cls, seed: int | None = None) -> Sampler:
        """Create a Tree-structured Parzen Estimator sampler.

        Args:
            seed: Random seed. If None, a random seed is used.

        Returns:
            A TPE sampler instance.
        """
    @classmethod
    def random(cls, seed: int | None = None) -> Sampler:
        """Create a random sampler.

        Args:
            seed: Random seed. If None, a random seed is used.

        Returns:
            A random sampler instance.
        """
    @classmethod
    def nsgaii(
        cls,
        seed: int | None = None,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
    ) -> Sampler:
        """Create an NSGA-II sampler for multi-objective optimization.

        Args:
            seed: Random seed. If None, a random seed is used.
            population_size: Population size.
            mutation_prob: Mutation probability. If None, 1.0 / len(search_space) is used.
            crossover_prob: Crossover probability.
            swapping_prob: Swapping probability for crossover.

        Returns:
            An NSGA-II sampler instance.
        """
    @property
    def support_joint_sampling(self) -> bool:
        """Return True if the sampler supports joint parameter sampling."""

    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: Storage,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]:
        """Sample multiple parameters simultaneously.

        Args:
            ctx: Sampler context.
            storage: Storage object.
            search_space: Parameter distributions.

        Returns:
            Suggested parameter values (Optuna's internal representation).
        """
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: Storage,
        name: str,
        distribution: Distribution,
    ) -> float:
        """Sample a single parameter independently.

        Args:
            ctx: Sampler context.
            storage: Storage object.
            name: Parameter name.
            distribution: Parameter distribution.

        Returns:
            Suggested parameter value (Optuna's internal representation).
        """

# Private APIs for rustuna.optuna package.
def _get_param_importance_from_list(
    features: list[list[float]],
    targets: list[float],
    n_trees: int,
) -> list[float]: ...
