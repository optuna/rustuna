from __future__ import annotations

import copy
import datetime
import json
import typing
import warnings
from typing import cast, overload

import optuna
from optuna import distributions
from optuna.trial import FrozenTrial

import rustuna

from ._distribution import (
    to_optuna_distributions,
    to_rustuna_distribution,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from optuna._typing import JSONSerializable
    from optuna.distributions import BaseDistribution
    from optuna.trial import TrialState


to_rustuna_state_map = {
    optuna.trial.TrialState.RUNNING: rustuna.TrialState.RUNNING,
    optuna.trial.TrialState.COMPLETE: rustuna.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL: rustuna.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED: rustuna.TrialState.PRUNED,
    optuna.trial.TrialState.WAITING: rustuna.TrialState.WAITING,
}
# TODO(c-bata): Make rustuna.TrialState hashable.
# to_optuna_state_map = {v: k for k, v in to_optuna_state_map.items()}


def to_optuna_state(state: rustuna.TrialState) -> optuna.trial.TrialState:
    if state == rustuna.TrialState.RUNNING:
        return optuna.trial.TrialState.RUNNING
    elif state == rustuna.TrialState.COMPLETE:
        return optuna.trial.TrialState.COMPLETE
    elif state == rustuna.TrialState.FAIL:
        return optuna.trial.TrialState.FAIL
    elif state == rustuna.TrialState.PRUNED:
        return optuna.trial.TrialState.PRUNED
    elif state == rustuna.TrialState.WAITING:
        return optuna.trial.TrialState.WAITING
    else:
        raise KeyError(f"Unknown state: {state}")


def to_rustuna_state(state: optuna.trial.TrialState) -> rustuna.TrialState:
    return to_rustuna_state_map[state]


def to_persisted_trial(
    trial: optuna.trial.FrozenTrial,
    study_id: int,
) -> rustuna.PersistedTrial:
    optuna_system_attrs = trial.system_attrs.copy()

    internal_params: dict[str, float] = {}
    distributions: dict[str, rustuna.Distribution] = {}
    for param_name in trial.distributions:
        optuna_distribution = trial.distributions[param_name]
        distributions[param_name] = to_rustuna_distribution(optuna_distribution)
        internal_params[param_name] = optuna_distribution.to_internal_repr(
            trial.params[param_name]
        )

    return rustuna.PersistedTrial(
        trial_id=trial._trial_id,
        study_id=study_id,
        number=trial.number,
        state=to_rustuna_state(trial.state),
        values=trial.values,
        internal_params=internal_params,
        distributions=distributions,
        user_attrs={k: json.dumps(v) for k, v in trial.user_attrs.items()},
        system_attrs={k: json.dumps(v) for k, v in optuna_system_attrs.items()},
        intermediate_values=trial.intermediate_values,
        datetime_start=trial.datetime_start,
        datetime_complete=trial.datetime_complete,
    )


class _LazyJSONAttrs(dict[str, typing.Any]):
    def __init__(self, raw: dict[str, str]) -> None:
        super().__init__(raw)
        self._decoded_keys: set[str] = set()

    def __getitem__(self, key: str) -> typing.Any:
        if key in self._decoded_keys:
            return super().__getitem__(key)
        raw = super().__getitem__(key)
        decoded = json.loads(raw)
        super().__setitem__(key, decoded)
        self._decoded_keys.add(key)
        return decoded

    def __setitem__(self, key: str, value: typing.Any) -> None:
        super().__setitem__(key, value)
        self._decoded_keys.add(key)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._decoded_keys.discard(key)

    def get(self, key: str, default: typing.Any = None) -> typing.Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):  # type: ignore[override]
        return {key: self[key] for key in self}.items()

    def values(self):  # type: ignore[override]
        return {key: self[key] for key in self}.values()

    def __repr__(self) -> str:
        return repr(dict(self.items()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            if len(self) != len(other):
                return False
            for key in self:
                if other.get(key) != self[key]:
                    return False
            return True
        return NotImplemented


class FrozenTrialLike(FrozenTrial):
    def __init__(self, persisted_trial: rustuna.PersistedTrial) -> None:
        self._persisted_trial = persisted_trial

        # Pre-cache frequently accessed lightweight fields to avoid repeated conversions
        self.__trial_id: int = persisted_trial._trial_id
        self.__number: int = persisted_trial.number
        self.__state: TrialState = to_optuna_state(persisted_trial.state)

        # The following fields are defined to support property.setter (lazy evaluation)
        self.__values: list[float] | None = None
        self.__intermediate_values: dict[int, float] | None = None
        self.__datetime_start: datetime.datetime | None = None
        self.__datetime_complete: datetime.datetime | None = None
        self.__params: dict[str, Any] | None = None
        self.__distributions: dict[str, BaseDistribution] | None = None
        self.__user_attrs: dict[str, Any] | None = None
        self.__system_attrs: dict[str, Any] | None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenTrial):
            return NotImplemented
        return all(
            [
                self._trial_id == other._trial_id,
                self.number == other.number,
                self.state == other.state,
                self.params == other.params,
                self.distributions == other.distributions,
                self.values == other.values,
                self.intermediate_values == other.intermediate_values,
                self.user_attrs == other.user_attrs,
                self.system_attrs == other.system_attrs,
                self.datetime_start == other.datetime_start,
                self.datetime_complete == other.datetime_complete,
            ]
        )

    @property
    def _trial_id(self) -> int:
        return self._persisted_trial._trial_id

    @_trial_id.setter
    def _trial_id(self, value: int) -> None:
        self.__trial_id = value

    @property
    def number(self) -> int:
        return self.__number

    @number.setter
    def number(self, value: int) -> None:
        self.__number = value

    @property
    def state(self) -> TrialState:
        return self.__state

    @state.setter
    def state(self, value: TrialState) -> None:
        self.__state = value

    @property
    def value(self) -> float | None:
        values = self.__values or self._persisted_trial.values
        if values is None:
            return None
        if len(values) > 1:
            raise RuntimeError(
                "This attribute is not available during multi-objective optimization."
            )
        return values[0]

    @value.setter
    def value(self, v: float | None) -> None:
        if self.__values is not None:
            if len(self.__values) > 1:
                raise RuntimeError(
                    "This attribute is not available during multi-objective optimization."
                )

        if v is not None:
            self.__values = [v]
        else:
            self.__values = None

    # These `_get_values`, `_set_values`, and `values = property(_get_values, _set_values)` are
    # defined to pass the mypy.
    def _get_values(self) -> list[float] | None:
        return self.__values or self._persisted_trial.values

    def _set_values(self, v: Sequence[float] | None) -> None:
        if v is not None:
            self.__values = list(v)
        else:
            self.__values = None

    values = property(_get_values, _set_values)

    @property
    def datetime_start(self) -> datetime.datetime | None:
        if self.__datetime_start is not None:
            return self.__datetime_start
        return self._persisted_trial.datetime_start

    @datetime_start.setter
    def datetime_start(self, value: datetime.datetime | None) -> None:
        self.__datetime_start = value

    @property
    def datetime_complete(self) -> datetime.datetime | None:
        if self.__datetime_complete is not None:
            return self.__datetime_complete
        return self._persisted_trial.datetime_complete

    @datetime_complete.setter
    def datetime_complete(self, value: datetime.datetime | None) -> None:
        self.__datetime_complete = value

    @property
    def params(self) -> dict[str, Any]:
        return self.__params or self._persisted_trial.params

    @params.setter
    def params(self, params: dict[str, Any]) -> None:
        self.__params = params

    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        return self.__distributions or to_optuna_distributions(
            self._persisted_trial.distributions
        )

    @distributions.setter
    def distributions(self, value: dict[str, BaseDistribution]) -> None:
        self.__distributions = value

    @property
    def user_attrs(self) -> dict[str, Any]:
        if self.__user_attrs is not None:
            return self.__user_attrs

        user_attrs = self._persisted_trial.user_attrs
        return _LazyJSONAttrs(user_attrs)

    @user_attrs.setter
    def user_attrs(self, value: dict[str, Any]) -> None:
        self.__user_attrs = value

    @property
    def system_attrs(self) -> dict[str, Any]:
        if self.__system_attrs is not None:
            return self.__system_attrs
        system_attrs = {
            key: self._persisted_trial.system_attrs[key]
            for key in self._persisted_trial.system_attrs
            if not key.startswith("category_labels:")
        }
        self.__system_attrs = _LazyJSONAttrs(system_attrs)
        return self.__system_attrs

    @system_attrs.setter
    def system_attrs(self, value: Mapping[str, JSONSerializable]) -> None:
        self.__system_attrs = cast("dict[str, Any]", value)

    @property
    def intermediate_values(self) -> dict[int, float]:
        if self.__intermediate_values is not None:
            return self.__intermediate_values
        return self._persisted_trial.intermediate_values

    @intermediate_values.setter
    def intermediate_values(self, values: dict[int, float]) -> None:
        self._intermediate_values = values

    @property
    def last_step(self) -> int | None:
        """Return the maximum step of :attr:`intermediate_values` in the trial.

        Returns:
            The maximum step of intermediates.
        """

        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

    @property
    def duration(self) -> datetime.timedelta | None:
        """Return the elapsed time taken to complete the trial.

        Returns:
            The duration.
        """

        if self.datetime_start and self.datetime_complete:
            return self.datetime_complete - self.datetime_start
        else:
            return None

    def __reduce__(self) -> str | tuple[Any, ...]:
        """Convert to a real FrozenTrial for pickling."""
        frozen_trial = FrozenTrial(
            number=self.number,
            state=self.state,
            value=None,
            values=self.values,
            datetime_start=self.datetime_start,
            datetime_complete=self.datetime_complete,
            params=self.params,
            distributions=self.distributions,
            user_attrs=self.user_attrs,
            system_attrs=self.system_attrs,
            intermediate_values=self.intermediate_values,
            trial_id=self._trial_id,
        )
        return frozen_trial.__reduce__()

    def __copy__(self) -> FrozenTrialLike:
        copied = FrozenTrialLike(self._persisted_trial)
        copied.__trial_id = self.__trial_id
        copied.__number = self.__number
        copied.__state = self.__state
        copied.__values = self.__values
        copied.__intermediate_values = self.__intermediate_values
        copied.__datetime_start = self.__datetime_start
        copied.__datetime_complete = self.__datetime_complete
        copied.__params = self.__params
        copied.__distributions = self.__distributions
        copied.__user_attrs = self.__user_attrs
        copied.__system_attrs = self.__system_attrs
        return copied

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenTrialLike:
        cached = memo.get(id(self))
        if cached is not None:
            return cached
        copied = FrozenTrialLike(self._persisted_trial)
        memo[id(self)] = copied
        copied.__trial_id = self.__trial_id
        copied.__number = self.__number
        copied.__state = self.__state
        copied.__values = copy.deepcopy(self.__values, memo)
        copied.__intermediate_values = copy.deepcopy(self.__intermediate_values, memo)
        copied.__datetime_start = self.__datetime_start
        copied.__datetime_complete = self.__datetime_complete
        copied.__params = copy.deepcopy(self.__params, memo)
        copied.__distributions = copy.deepcopy(self.__distributions, memo)
        copied.__user_attrs = copy.deepcopy(self.__user_attrs, memo)
        copied.__system_attrs = copy.deepcopy(self.__system_attrs, memo)
        return copied

    def set_user_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def _validate(self) -> None:
        raise NotImplementedError

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        if name not in self.params:
            raise ValueError(
                "The value of the parameter '{}' is not found. Please set it at "
                "the construction of the FrozenTrial object.".format(name)
            )

        value = self.params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(value)
        if not distribution._contains(param_value_in_internal_repr):
            warnings.warn(
                "The value {} of the parameter '{}' is out of "
                "the range of the distribution {}.".format(value, name, distribution)
            )

        if name in self._distributions:
            distributions.check_distribution_compatibility(
                self._distributions[name], distribution
            )

        self.distributions[name] = distribution
        return value


def to_frozen_trial(
    persisted_trial: rustuna.PersistedTrial,
    *,
    use_frozen_trial_like: bool = True,
) -> FrozenTrial:
    ft_like = FrozenTrialLike(persisted_trial)
    if use_frozen_trial_like:
        return ft_like
    return FrozenTrial(
        number=ft_like.number,
        state=ft_like.state,
        value=None,
        values=ft_like.values,
        datetime_start=ft_like.datetime_start,
        datetime_complete=ft_like.datetime_complete,
        params=ft_like.params,
        distributions=ft_like.distributions,
        user_attrs=ft_like.user_attrs,
        system_attrs=ft_like.system_attrs,
        intermediate_values=ft_like.intermediate_values,
        trial_id=ft_like._trial_id,
    )
