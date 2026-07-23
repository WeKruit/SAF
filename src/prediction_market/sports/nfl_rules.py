"""Pinned NFL clock and timeout rules for 2015--2025 state reduction."""

from __future__ import annotations

import re
from typing import Literal


SeasonType = Literal["REG", "POST"]

NFL_RULE_SEASONS = frozenset(range(2015, 2026))
_NFLVERSE_SEASON_RE = re.compile(r"^game_nflverse_([0-9]{4})_")


class NFLRulesError(ValueError):
    """A state falls outside the pinned NFL rules snapshot."""


def season_from_game_id(game_id: str) -> int:
    """Return the NFL season encoded by a canonical nflverse game id."""

    if type(game_id) is not str:
        raise NFLRulesError("game_id must encode an NFL season")
    match = _NFLVERSE_SEASON_RE.match(game_id)
    if match is None:
        raise NFLRulesError("game_id must encode an NFL season")
    season = int(match.group(1))
    if season not in NFL_RULE_SEASONS:
        raise NFLRulesError("NFL rule season must be between 2015 and 2025")
    return season


def validate_season_type(season_type: str) -> SeasonType:
    """Validate the native nflverse regular/postseason discriminator."""

    if season_type not in {"REG", "POST"}:
        raise NFLRulesError("season_type must be REG or POST")
    return season_type


def overtime_period_seconds(season: int, season_type: str) -> int:
    """Return the maximum clock for one overtime period."""

    if season not in NFL_RULE_SEASONS:
        raise NFLRulesError("NFL rule season must be between 2015 and 2025")
    validated_type = validate_season_type(season_type)
    if validated_type == "POST":
        return 900
    return 900 if season <= 2016 else 600


def timeout_allotment(season_type: str, period: int) -> int:
    """Return the per-team timeout allotment for the containing half."""

    validated_type = validate_season_type(season_type)
    if type(period) is not int or period < 1:
        raise NFLRulesError("period must be a positive integer")
    if period <= 4:
        return 3
    return 2 if validated_type == "REG" else 3


def timeout_reset_allotment(
    season_type: str,
    previous_period: int,
    next_period: int,
) -> int | None:
    """Return the required reset value at a half boundary, if any."""

    validated_type = validate_season_type(season_type)
    if next_period != previous_period + 1:
        raise NFLRulesError("timeout reset periods must be adjacent")
    if (previous_period, next_period) == (2, 3):
        return 3
    if validated_type == "REG":
        return 2 if (previous_period, next_period) == (4, 5) else None
    if next_period >= 5 and next_period % 2 == 1:
        return 3
    return None


def postseason_ot_timeout_offset(season_type: str, period: int) -> int:
    """Return nflverse's rules-derived postseason OT counter offset."""

    validated_type = validate_season_type(season_type)
    return 1 if validated_type == "POST" and period >= 5 else 0


def normalize_native_timeout_remaining(
    value: int,
    *,
    season_type: str,
    period: int,
) -> int:
    """Normalize one native timeout counter and fail outside rule bounds."""

    if type(value) is not int:
        raise NFLRulesError("native timeout counter must be an integer")
    normalized = value + postseason_ot_timeout_offset(season_type, period)
    maximum = timeout_allotment(season_type, period)
    if not 0 <= normalized <= maximum:
        raise NFLRulesError(
            "normalized timeout counter is outside the rules allotment"
        )
    return normalized


__all__ = [
    "NFL_RULE_SEASONS",
    "NFLRulesError",
    "SeasonType",
    "normalize_native_timeout_remaining",
    "overtime_period_seconds",
    "postseason_ot_timeout_offset",
    "season_from_game_id",
    "timeout_allotment",
    "timeout_reset_allotment",
    "validate_season_type",
]
