from __future__ import annotations

from optuna.study._frozen import FrozenStudy

import rustuna

from ._direction import to_optuna_directions, to_rustuna_directions


def to_frozen_study(study: rustuna.study.PersistedStudy) -> FrozenStudy:
    """Convert a Rustuna persisted study to an Optuna frozen study."""
    return FrozenStudy(
        study_name=study.name,
        study_id=study.id,
        direction=None,
        directions=to_optuna_directions(study.directions),
        user_attrs=study.user_attrs,
        system_attrs=study.system_attrs,
    )


def to_persisted_study(study: FrozenStudy) -> rustuna.study.PersistedStudy:
    """Convert an Optuna frozen study to a Rustuna persisted study."""
    return rustuna.study.PersistedStudy(
        id=study._study_id,
        name=study.study_name,
        directions=to_rustuna_directions(study.directions),
        user_attrs=study.user_attrs,
        system_attrs=study.system_attrs,
    )
