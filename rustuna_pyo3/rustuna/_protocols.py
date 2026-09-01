from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rustuna import CategoricalChoiceType
    from rustuna._rustuna import Distribution
    from rustuna.samplers import SamplerContext
    from rustuna.study import PersistedStudy, StudyDirection
    from rustuna.trial import PersistedTrial, TrialState


class SamplerProtocol(Protocol):
    """Protocol for sampler implementations.

    This protocol defines the interface that samplers must implement
    to suggest parameter values.
    """

    @property
    def support_joint_sampling(self) -> bool:
        """Return True if the sampler supports joint parameter sampling."""

    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
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

    def before_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
    ) -> None:
        """Run sampler pre-processing before search-space inference.

        The newly created trial is available as ``storage.get_trial(ctx.trial_id)``.
        The target study is available as ``storage.get_study(ctx.study_id)``.

        Args:
            ctx: Sampler context.
            storage: Storage object.
        """

    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
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

    def after_trial(
        self,
        ctx: SamplerContext,
        storage: StorageProtocol,
        state: TrialState,
        values: list[float] | None = None,
    ) -> None:
        """Run sampler post-processing after a trial finishes.

        Args:
            ctx: Sampler context.
            storage: Storage object.
            state: Final trial state.
            values: Final objective values. ``None`` unless the trial completed.
        """


class TrialQueueProtocol(Protocol):
    """Protocol for trial queue implementations.

    This protocol defines the interface implemented by trial queue objects.
    """

    def enqueue(self, trial_id: int) -> None:
        """Add a trial ID to the queue.

        Args:
            trial_id: The trial ID to enqueue.
        """

    def dequeue(self) -> int | None:
        """Remove and return the next trial ID from the queue.

        Returns:
            The next trial ID in FIFO order, or None when the queue is empty.
        """


class StorageProtocol(Protocol):
    """Protocol for storage implementations.

    This protocol defines the interface that storage backends must implement
    to persist optimization history.
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

    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial:
        """Create a new trial in the specified study.

        Args:
            study_id: ID of the study.
            template_trial: Template PersistedTrial with default user-attributes,
                system-attributes, intermediate-values, and a state.

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

    def get_trials(
        self, study_id: int, *, states: list[TrialState] | None = None
    ) -> list[PersistedTrial]:
        """Get all trials in a study.

        Args:
            study_id: ID of the study.
            states: Optional trial states to filter by.

        Returns:
            List of all trials in the study.
        """

    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int:
        """Get the number of trials in a study.

        Args:
            study_id: ID of the study.
            states: Optional trial states to filter by.

        Returns:
            Number of trials in the study.
        """

    def get_trial(self, trial_id: int) -> PersistedTrial:
        """Get a trial by ID.

        Args:
            trial_id: ID of the trial.

        Returns:
            The trial.
        """

    def get_cached_trial(self, trial_id: int) -> PersistedTrial:
        """Get a cached trial by ID without synchronizing with backends.

        Args:
            trial_id: ID of the trial.

        Returns:
            The trial.
        """

    def get_trial_number_from_id(self, trial_id: int) -> int:
        """Get the trial number from a trial ID.

        Args:
            trial_id: ID of the trial.

        Returns:
            Number of the trial within its study.
        """

    def get_study_user_attr(self, study_id: int, key: str) -> str:
        """Get a single user attribute of a study.

        Args:
            study_id: ID of the study.
            key: Attribute key.

        Returns:
            The attribute value as a string.
        """

    def get_study_system_attr(self, study_id: int, key: str) -> str:
        """Get a single system attribute of a study.

        Args:
            study_id: ID of the study.
            key: Attribute key.

        Returns:
            The attribute value as a string.
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

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        """Set an intermediate value for a trial.

        Args:
            trial_id: ID of the trial.
            step: Step at which the intermediate value is reported.
            intermediate_value: Intermediate objective value.
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

    def discard_trials(self, trial_ids: list[int]) -> None:
        """Discard trials from the storage view.

        Args:
            trial_ids: IDs of trials to discard.
        """

    def may_omit_trials(self) -> bool:
        """Return True if this storage view may omit discarded trials."""


class CachedStorageBackend(Protocol):
    """Protocol for Python backends used by Rustuna's ``CachedStorage``."""

    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy:
        """Create a new study and return its persisted representation."""

    def delete_study(self, study_id: int) -> None:
        """Delete a study."""

    def create_new_trial(
        self,
        study_id: int,
        template_trial: PersistedTrial | None = None,
    ) -> PersistedTrial:
        """Create a trial and return its persisted representation."""

    def set_trial_param(
        self,
        trial_id: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None:
        """Persist a trial parameter."""

    def set_trial_state_values(
        self,
        trial_id: int,
        state: TrialState,
        values: list[float] | None = None,
    ) -> None:
        """Persist a trial state and its objective values."""

    def get_studies(self) -> list[PersistedStudy]:
        """Return all persisted studies."""

    def get_study(self, study_id: int) -> PersistedStudy:
        """Return a persisted study."""

    def get_trial(self, trial_id: int) -> PersistedTrial:
        """Return a persisted trial."""

    def get_n_trials(
        self, study_id: int, *, states: Sequence[TrialState] | None = None
    ) -> int:
        """Return the number of persisted trials."""

    def get_study_user_attr(self, study_id: int, key: str) -> str:
        """Return a study user attribute."""

    def get_study_system_attr(self, study_id: int, key: str) -> str:
        """Return a study system attribute."""

    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        """Persist study system attributes."""

    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None:
        """Persist study user attributes."""

    def set_trial_system_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        """Persist trial system attributes."""

    def set_trial_user_attrs(self, trial_id: int, attrs: dict[str, str]) -> None:
        """Persist trial user attributes."""

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        """Persist an intermediate trial value."""

    def discard_trials(self, trial_ids: list[int]) -> None:
        """Persist discarded trial IDs."""

    def may_omit_trials(self) -> bool:
        """Return whether reads omit discarded trials."""

    def get_trials_diff(
        self,
        study_id: int,
        included_numbers: list[int],
        trial_number_greater_than: int,
    ) -> list[PersistedTrial]:
        """Return trials that are missing or may have changed in the cache."""
