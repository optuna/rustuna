from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Optuna exposes attrs as `dict[str, Any]`. Rustuna's storages created from Python store
# attribute values in Optuna's schema, so values are passed as they are. Constraint attrs
# are the exception: Rustuna stores each constraint under a `constraints:<i>` key, while
# Optuna stores them as a `constraints` list.
#
# Rustuna may also store internal system attrs, such as categorical labels; callers
# exposing attrs through Optuna are responsible for filtering those Rustuna-specific
# attributes when necessary.
_CONSTRAINTS_KEY = "constraints"
_CONSTRAINTS_PREFIX = f"{_CONSTRAINTS_KEY}:"


def to_optuna_trial_system_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in attrs.items():
        if key.startswith(_CONSTRAINTS_PREFIX):
            # Rustuna stores constraints as strings, while Optuna exposes them as floats.
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def to_rustuna_trial_system_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(attrs)
    # Convert Optuna's legacy constraint to Rustuna constraints
    if _CONSTRAINTS_KEY in converted:
        constraints = converted.pop(_CONSTRAINTS_KEY)
        for i, c in enumerate(constraints):
            converted[_CONSTRAINTS_PREFIX + str(i)] = c
    return converted
