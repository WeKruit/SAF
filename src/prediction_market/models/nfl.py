"""NFL classic logistic-feature pipeline proof of concept."""

from __future__ import annotations

import math


def nfl_logistic_features(
    *,
    score_differential: int | float,
    seconds_remaining: int | float,
    possession_is_home: bool,
    home_timeouts: int,
    away_timeouts: int,
) -> dict[str, float]:
    values = (
        float(score_differential),
        float(seconds_remaining),
        float(home_timeouts),
        float(away_timeouts),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NFL state features must be finite")
    if values[1] < 0 or home_timeouts < 0 or away_timeouts < 0:
        raise ValueError("remaining time and timeouts must be nonnegative")
    if type(possession_is_home) is not bool:
        raise ValueError("possession_is_home must be boolean")
    return {
        "score_differential": values[0],
        "seconds_remaining": values[1],
        "possession_is_home": 1.0 if possession_is_home else 0.0,
        "home_timeouts": values[2],
        "away_timeouts": values[3],
    }


__all__ = ["nfl_logistic_features"]
