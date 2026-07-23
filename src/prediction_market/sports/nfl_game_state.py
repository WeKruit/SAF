"""Immutable, deterministic NFL play-level game-state transitions.

The nflverse adapter in this module is deliberately an *offline observation*
adapter.  It reads a play's start-state row and the following observation row.
It never reads final-game score columns, EPA/WPA, win probabilities, or drive
result fields into either the state or the normalized event.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.sports.event_envelopes import (
    validate_static_sport_observation_bundle,
)
from prediction_market.sports.nfl_rules import (
    NFLRulesError,
    normalize_native_timeout_remaining,
    overtime_period_seconds,
    season_from_game_id,
    timeout_allotment,
    timeout_reset_allotment,
    validate_season_type,
)


NFL_SPORT = "nfl"
NFLVERSE_OBSERVATION_MODE = "offline"
NFLVERSE_LEAKAGE_FIELDS = frozenset(
    {
        "away_score",
        "def_wp",
        "def_wpa",
        "drive_ended_with_score",
        "drive_result",
        "ep",
        "epa",
        "fixed_drive_result",
        "home_score",
        "home_wp",
        "result",
        "series_result",
        "series_success",
        "success",
        "total",
        "vegas_home_wp",
        "vegas_wp",
        "wp",
        "wpa",
    }
)

_EVENT_ID_RE = re.compile(r"^evt_[0-9a-f]{64}$")
_GAME_ID_RE = re.compile(r"^game_[A-Za-z0-9][A-Za-z0-9._:-]*$")
_LEGAL_SCORE_DELTAS = frozenset({0, 1, 2, 3, 6})


class NFLGameStateError(ValueError):
    """An NFL state, event, or transition failed closed validation."""


def _require_bool(value: object, field: str) -> None:
    if type(value) is not bool:
        raise NFLGameStateError(f"{field} must be boolean")


def _require_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise NFLGameStateError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise NFLGameStateError(f"{field} must be at least {minimum}{upper}")


def _require_text(value: object, field: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise NFLGameStateError(f"{field} must be a canonical nonempty string")
    if "\x00" in value:
        raise NFLGameStateError(f"{field} must not contain NUL")


def _require_game_id(value: object, field: str) -> None:
    if type(value) is not str or _GAME_ID_RE.fullmatch(value) is None:
        raise NFLGameStateError(
            f"{field} must be canonical game_<id>"
        )


def _rule_context(game_id: str, season_type: str) -> tuple[int, str]:
    try:
        return season_from_game_id(game_id), validate_season_type(season_type)
    except NFLRulesError as exc:
        raise NFLGameStateError(str(exc)) from exc


def _validate_clock(
    game_id: str,
    season_type: str,
    period: object,
    period_seconds_remaining: object,
    game_seconds_remaining: object,
    *,
    context: str,
) -> None:
    season, validated_season_type = _rule_context(game_id, season_type)
    _require_int(period, f"{context}.period", minimum=1)
    assert type(period) is int
    if validated_season_type == "REG" and period > 5:
        raise NFLGameStateError(
            f"{context}.period exceeds regular-season overtime"
        )
    maximum_period_seconds = (
        900
        if period <= 4
        else overtime_period_seconds(season, validated_season_type)
    )
    _require_int(
        period_seconds_remaining,
        f"{context}.period_seconds_remaining",
        minimum=0,
        maximum=maximum_period_seconds,
    )
    _require_int(
        game_seconds_remaining,
        f"{context}.game_seconds_remaining",
        minimum=0,
        maximum=3600,
    )
    assert type(period_seconds_remaining) is int
    assert type(game_seconds_remaining) is int
    if period <= 4:
        expected_game_seconds = (4 - period) * 900 + period_seconds_remaining
        if game_seconds_remaining != expected_game_seconds:
            raise NFLGameStateError(
                f"{context} regulation clocks are inconsistent"
            )
    elif game_seconds_remaining not in {0, period_seconds_remaining}:
        raise NFLGameStateError(
            f"{context} overtime game clock must be zero or the period clock"
        )


def _validate_football_values(
    *,
    season_type: str,
    period: int,
    down: object,
    distance: object,
    yardline_100: object,
    home_score: object,
    away_score: object,
    home_timeouts_remaining: object,
    away_timeouts_remaining: object,
    context: str,
) -> None:
    if down is None:
        if distance is not None:
            raise NFLGameStateError(f"{context}.distance requires a down")
    else:
        _require_int(down, f"{context}.down", minimum=1, maximum=4)
        if distance is None:
            raise NFLGameStateError(f"{context}.down requires distance")
        _require_int(distance, f"{context}.distance", minimum=1, maximum=99)
        if yardline_100 is None:
            raise NFLGameStateError(f"{context}.down requires yardline_100")
    if yardline_100 is not None:
        _require_int(
            yardline_100,
            f"{context}.yardline_100",
            minimum=0,
            maximum=100,
        )
    if (
        down is not None
        and type(distance) is int
        and type(yardline_100) is int
        and distance > yardline_100
    ):
        raise NFLGameStateError(
            f"{context}.distance cannot extend beyond the opponent end zone"
        )
    _require_int(home_score, f"{context}.home_score", minimum=0)
    _require_int(away_score, f"{context}.away_score", minimum=0)
    try:
        maximum_timeouts = timeout_allotment(season_type, period)
    except NFLRulesError as exc:
        raise NFLGameStateError(str(exc)) from exc
    for field_name, value in (
        ("home_timeouts_remaining", home_timeouts_remaining),
        ("away_timeouts_remaining", away_timeouts_remaining),
    ):
        _require_int(value, f"{context}.{field_name}", minimum=0)
        if type(value) is int and value > maximum_timeouts:
            word = "three" if maximum_timeouts == 3 else "two"
            raise NFLGameStateError(
                f"{context}.{field_name} exceeds the {word}-timeout allotment"
            )


def _validate_terminal(
    *,
    terminal: bool,
    period: int,
    period_seconds_remaining: int,
    game_seconds_remaining: int,
    home_score: int,
    away_score: int,
    context: str,
) -> None:
    if not terminal:
        return
    if period < 4:
        raise NFLGameStateError(f"{context}.terminal is impossible before period 4")
    if period == 4:
        if game_seconds_remaining != 0:
            raise NFLGameStateError(
                f"{context}.terminal regulation state requires an expired clock"
            )
        if home_score == away_score:
            raise NFLGameStateError(
                f"{context}.terminal tied regulation state requires overtime"
            )
    elif home_score == away_score and period_seconds_remaining != 0:
        raise NFLGameStateError(
            f"{context}.terminal tied overtime state requires an expired clock"
        )


@dataclass(frozen=True, slots=True)
class NFLGameState:
    """Complete observable NFL state immediately before the next event."""

    sport: str
    game_id: str
    sequence: int
    terminal: bool
    season_type: str
    home_team: str
    away_team: str
    period: int
    period_seconds_remaining: int
    game_seconds_remaining: int
    source_play_id: str
    source_order_sequence: int
    context_source_play_id: str
    context_source_order_sequence: int
    suspended: bool
    drive_id: str | None
    play_clock_seconds: int | None
    possession_team: str | None
    down: int | None
    distance: int | None
    yardline_100: int | None
    goal_to_go: bool
    home_score: int
    away_score: int
    home_timeouts_remaining: int
    away_timeouts_remaining: int
    last_event_id: str | None = None

    def __post_init__(self) -> None:
        if self.sport != NFL_SPORT:
            raise NFLGameStateError("state.sport must be nfl")
        _require_game_id(self.game_id, "state.game_id")
        _require_int(self.sequence, "state.sequence", minimum=0)
        _require_bool(self.terminal, "state.terminal")
        _rule_context(self.game_id, self.season_type)
        _require_text(self.home_team, "state.home_team")
        _require_text(self.away_team, "state.away_team")
        if self.home_team == self.away_team:
            raise NFLGameStateError("home_team and away_team must differ")
        _require_text(self.source_play_id, "state.source_play_id")
        _require_int(
            self.source_order_sequence,
            "state.source_order_sequence",
            minimum=0,
        )
        _require_text(
            self.context_source_play_id,
            "state.context_source_play_id",
        )
        _require_int(
            self.context_source_order_sequence,
            "state.context_source_order_sequence",
            minimum=0,
        )
        _require_bool(self.suspended, "state.suspended")
        if self.drive_id is not None:
            _require_text(self.drive_id, "state.drive_id")
        if self.play_clock_seconds is not None:
            _require_int(
                self.play_clock_seconds,
                "state.play_clock_seconds",
                minimum=0,
                maximum=40,
            )
        _require_bool(self.goal_to_go, "state.goal_to_go")
        if self.possession_team is not None and self.possession_team not in {
            self.home_team,
            self.away_team,
        }:
            raise NFLGameStateError(
                "state.possession_team must be home_team, away_team, or None"
            )
        if self.down is not None and self.possession_team is None:
            raise NFLGameStateError("state.down requires possession_team")
        _validate_clock(
            self.game_id,
            self.season_type,
            self.period,
            self.period_seconds_remaining,
            self.game_seconds_remaining,
            context="state",
        )
        _validate_football_values(
            season_type=self.season_type,
            period=self.period,
            down=self.down,
            distance=self.distance,
            yardline_100=self.yardline_100,
            home_score=self.home_score,
            away_score=self.away_score,
            home_timeouts_remaining=self.home_timeouts_remaining,
            away_timeouts_remaining=self.away_timeouts_remaining,
            context="state",
        )
        if self.last_event_id is not None and _EVENT_ID_RE.fullmatch(
            self.last_event_id
        ) is None:
            raise NFLGameStateError(
                "state.last_event_id must be a real evt_<64hex> identifier"
            )
        _validate_terminal(
            terminal=self.terminal,
            period=self.period,
            period_seconds_remaining=self.period_seconds_remaining,
            game_seconds_remaining=self.game_seconds_remaining,
            home_score=self.home_score,
            away_score=self.away_score,
            context="state",
        )


@dataclass(frozen=True, slots=True)
class NFLPlayEvent:
    """One normalized offline NFL play/administrative observation."""

    sport: str
    game_id: str
    sequence: int
    event_id: str
    season_type: str
    source_play_id: str
    source_order_sequence: int
    observation_mode: str
    play_type: str | None
    play_type_nfl: str | None
    description: str | None
    period: int | None
    period_seconds_remaining: int | None
    game_seconds_remaining: int | None
    next_source_play_id: str
    next_source_order_sequence: int
    context_source_play_id: str
    context_source_order_sequence: int
    source_window_play_ids: tuple[str, ...]
    source_window_order_sequences: tuple[int, ...]
    lifecycle_action: Literal["none", "suspend", "resume"]
    clock_carry_forward: bool
    next_drive_id: str | None
    next_play_clock_seconds: int | None
    possession_team: str | None
    down: int | None
    distance: int | None
    yardline_100: int | None
    goal_to_go: bool
    home_score: int
    away_score: int
    home_timeouts_remaining: int
    away_timeouts_remaining: int
    first_down: bool
    turnover: bool
    possession_changed: bool
    score: bool
    timeout_observed: bool
    timeout_observed_team: str | None
    timeout_kind: Literal["none", "administrative", "play_attached"]
    timeout_charge_team: str | None
    quarter_end: bool
    clock_correction: bool
    clock_correction_observed_period_seconds_remaining: int | None
    clock_correction_observed_game_seconds_remaining: int | None
    carry_forward_context: bool
    period_changed: bool
    terminal: bool
    quality_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.sport != NFL_SPORT:
            raise NFLGameStateError("event.sport must be nfl")
        _require_game_id(self.game_id, "event.game_id")
        _require_int(self.sequence, "event.sequence", minimum=1)
        _rule_context(self.game_id, self.season_type)
        _require_text(self.source_play_id, "event.source_play_id")
        _require_int(
            self.source_order_sequence,
            "event.source_order_sequence",
            minimum=0,
        )
        if self.observation_mode != NFLVERSE_OBSERVATION_MODE:
            raise NFLGameStateError(
                "event.observation_mode must be explicit offline observation"
            )
        if self.play_type is not None:
            _require_text(self.play_type, "event.play_type")
        if self.play_type_nfl is not None:
            _require_text(self.play_type_nfl, "event.play_type_nfl")
        if self.description is not None:
            _require_text(self.description, "event.description")
        _require_text(self.next_source_play_id, "event.next_source_play_id")
        _require_int(
            self.next_source_order_sequence,
            "event.next_source_order_sequence",
            minimum=0,
        )
        if self.next_source_order_sequence <= self.source_order_sequence:
            raise NFLGameStateError(
                "event next source order must strictly increase"
            )
        _require_text(
            self.context_source_play_id,
            "event.context_source_play_id",
        )
        _require_int(
            self.context_source_order_sequence,
            "event.context_source_order_sequence",
            minimum=0,
        )
        if (
            type(self.source_window_play_ids) is not tuple
            or type(self.source_window_order_sequences) is not tuple
            or len(self.source_window_play_ids) < 2
            or len(self.source_window_play_ids)
            != len(self.source_window_order_sequences)
            or len(set(self.source_window_play_ids))
            != len(self.source_window_play_ids)
        ):
            raise NFLGameStateError(
                "event source window must contain aligned source identities"
            )
        for play_id in self.source_window_play_ids:
            _require_text(play_id, "event.source_window_play_ids[]")
        for order in self.source_window_order_sequences:
            _require_int(
                order,
                "event.source_window_order_sequences[]",
                minimum=0,
            )
        if (
            self.source_window_play_ids[0] != self.source_play_id
            or self.source_window_order_sequences[0]
            != self.source_order_sequence
            or self.source_window_play_ids[1] != self.next_source_play_id
            or self.source_window_order_sequences[1]
            != self.next_source_order_sequence
            or any(
                later <= earlier
                for earlier, later in zip(
                    self.source_window_order_sequences,
                    self.source_window_order_sequences[1:],
                )
            )
            or (
                not self.carry_forward_context
                and (
                    self.source_window_play_ids[-1]
                    != self.context_source_play_id
                    or self.source_window_order_sequences[-1]
                    != self.context_source_order_sequence
                )
            )
        ):
            raise NFLGameStateError(
                "event source window does not match canonical transition order"
            )
        if self.lifecycle_action not in {"none", "suspend", "resume"}:
            raise NFLGameStateError(
                "event.lifecycle_action must be none, suspend, or resume"
            )
        derived_lifecycle_action = "none"
        if self.play_type_nfl == "COMMENT" and self.description is not None:
            if self.description.startswith("The game has been suspended."):
                derived_lifecycle_action = "suspend"
            elif self.description.startswith("The game has resumed."):
                derived_lifecycle_action = "resume"
        if self.lifecycle_action != derived_lifecycle_action:
            raise NFLGameStateError(
                "event lifecycle_action must match the bounded COMMENT source"
            )
        _require_bool(self.clock_carry_forward, "event.clock_carry_forward")
        if self.next_drive_id is not None:
            _require_text(self.next_drive_id, "event.next_drive_id")
        if self.next_play_clock_seconds is not None:
            _require_int(
                self.next_play_clock_seconds,
                "event.next_play_clock_seconds",
                minimum=0,
                maximum=40,
            )
        _require_bool(self.goal_to_go, "event.goal_to_go")
        if self.possession_team is not None:
            _require_text(self.possession_team, "event.possession_team")
        elif self.down is not None:
            raise NFLGameStateError("event.down requires possession_team")
        lifecycle = self.lifecycle_action != "none"
        clock_values = (
            self.period,
            self.period_seconds_remaining,
            self.game_seconds_remaining,
        )
        if lifecycle:
            if not self.clock_carry_forward or any(
                value is not None for value in clock_values
            ):
                raise NFLGameStateError(
                    "lifecycle events must carry forward all clock fields"
                )
        else:
            if self.clock_carry_forward or any(
                value is None for value in clock_values
            ):
                raise NFLGameStateError(
                    "non-lifecycle events require a complete clock"
                )
            assert self.period is not None
            assert self.period_seconds_remaining is not None
            assert self.game_seconds_remaining is not None
            _validate_clock(
                self.game_id,
                self.season_type,
                self.period,
                self.period_seconds_remaining,
                self.game_seconds_remaining,
                context="event",
            )
        _validate_football_values(
            season_type=self.season_type,
            period=1 if self.period is None else self.period,
            down=self.down,
            distance=self.distance,
            yardline_100=self.yardline_100,
            home_score=self.home_score,
            away_score=self.away_score,
            home_timeouts_remaining=self.home_timeouts_remaining,
            away_timeouts_remaining=self.away_timeouts_remaining,
            context="event",
        )
        for field in (
            "first_down",
            "turnover",
            "possession_changed",
            "score",
            "timeout_observed",
            "quarter_end",
            "clock_correction",
            "carry_forward_context",
            "period_changed",
            "terminal",
        ):
            _require_bool(getattr(self, field), f"event.{field}")
        if self.timeout_observed:
            _require_text(
                self.timeout_observed_team,
                "event.timeout_observed_team",
            )
        elif self.timeout_observed_team is not None:
            raise NFLGameStateError(
                "event.timeout_observed_team requires an observed timeout"
            )
        if self.timeout_charge_team is not None:
            _require_text(
                self.timeout_charge_team,
                "event.timeout_charge_team",
            )
            if not self.timeout_observed:
                raise NFLGameStateError(
                    "event.timeout_charge_team requires an observed timeout"
                )
        if self.timeout_kind not in {
            "none",
            "administrative",
            "play_attached",
        }:
            raise NFLGameStateError(
                "event.timeout_kind must be none, administrative, or play_attached"
            )
        administrative_source = (
            self.play_type_nfl == "TIMEOUT"
            and self.play_type == "no_play"
        )
        expected_timeout_kind = (
            "none"
            if not self.timeout_observed
            else "administrative"
            if administrative_source
            else "play_attached"
        )
        if self.timeout_kind != expected_timeout_kind:
            raise NFLGameStateError(
                "event.timeout_kind administrative classification requires "
                "TIMEOUT + no_play source fields"
            )
        if self.timeout_kind == "administrative":
            if not self.carry_forward_context:
                raise NFLGameStateError(
                    "administrative timeout must carry football context"
                )
            if self.first_down or self.turnover or self.possession_changed or self.score:
                raise NFLGameStateError(
                    "administrative timeout may only charge a timeout"
                )
        elif self.timeout_kind == "play_attached" and self.carry_forward_context:
            raise NFLGameStateError(
                "play-attached timeout must apply observed play context"
            )
        if self.carry_forward_context and (
            self.first_down or self.turnover or self.possession_changed
        ):
            raise NFLGameStateError(
                "a context-carry event cannot claim a contextual transition"
            )
        if lifecycle:
            prefix = (
                "The game has been suspended."
                if self.lifecycle_action == "suspend"
                else "The game has resumed."
            )
            if self.description is None or not self.description.startswith(prefix):
                raise NFLGameStateError(
                    "lifecycle action requires its bounded source description"
                )
            if (
                not self.carry_forward_context
                or self.first_down
                or self.turnover
                or self.possession_changed
                or self.score
                or self.timeout_observed
                or self.timeout_charge_team is not None
                or self.period_changed
                or self.terminal
            ):
                raise NFLGameStateError(
                    "lifecycle event may only update suspension state"
                )
        else:
            assert self.period is not None
            assert self.period_seconds_remaining is not None
            assert self.game_seconds_remaining is not None
            _validate_terminal(
                terminal=self.terminal,
                period=self.period,
                period_seconds_remaining=self.period_seconds_remaining,
                game_seconds_remaining=self.game_seconds_remaining,
                home_score=self.home_score,
                away_score=self.away_score,
                context="event",
            )
        if (
            type(self.event_id) is not str
            or _EVENT_ID_RE.fullmatch(self.event_id) is None
        ):
            raise NFLGameStateError(
                "event.event_id must be supplied by EventEnvelopeV0"
            )
        if (
            type(self.quality_flags) is not tuple
            or len(self.quality_flags) != len(set(self.quality_flags))
            or tuple(sorted(self.quality_flags)) != self.quality_flags
        ):
            raise NFLGameStateError(
                "event.quality_flags must be a sorted unique tuple"
            )
        for flag in self.quality_flags:
            _require_text(flag, "event.quality_flags[]")
        inserted_timeout_flag = "source_order_inserted_timeout"
        if any(flag != inserted_timeout_flag for flag in self.quality_flags):
            raise NFLGameStateError(
                "event quality flags contain an unsupported NFL flag"
            )
        if (
            (inserted_timeout_flag in self.quality_flags)
            != self.clock_correction
        ):
            raise NFLGameStateError(
                "event clock correction and quality flag must agree"
            )
        correction_observations = (
            self.clock_correction_observed_period_seconds_remaining,
            self.clock_correction_observed_game_seconds_remaining,
        )
        if self.clock_correction:
            if any(value is None for value in correction_observations):
                raise NFLGameStateError(
                    "clock correction requires the observed source clock"
                )
            for field_name, value in zip(
                (
                    "clock_correction_observed_period_seconds_remaining",
                    "clock_correction_observed_game_seconds_remaining",
                ),
                correction_observations,
                strict=True,
            ):
                assert value is not None
                _require_int(value, f"event.{field_name}", minimum=0)
        elif any(value is not None for value in correction_observations):
            raise NFLGameStateError(
                "observed correction clock requires a correction"
            )
        if self.clock_correction and (
            self.timeout_kind != "administrative"
            or not _is_inserted_native_play_boundary(
                self.source_play_id,
                self.next_source_play_id,
            )
        ):
            raise NFLGameStateError(
                "clock correction requires an inserted administrative timeout"
            )


def _is_inserted_native_play_boundary(
    source_play_id: str,
    next_source_play_id: str,
) -> bool:
    if not source_play_id.isdigit() or not next_source_play_id.isdigit():
        return False
    return int(source_play_id) > int(next_source_play_id)


def _period_clock_maximum(
    game_id: str,
    season_type: str,
    period: int,
) -> int:
    season, validated_season_type = _rule_context(game_id, season_type)
    return (
        900
        if period <= 4
        else overtime_period_seconds(season, validated_season_type)
    )


def _validate_clock_transition(state: NFLGameState, event: NFLPlayEvent) -> None:
    if event.timeout_kind == "administrative" and event.terminal:
        raise NFLGameStateError(
            "administrative timeout cannot end the game"
        )
    if event.clock_carry_forward:
        if event.period_changed or event.clock_correction:
            raise NFLGameStateError(
                "clock-carry event cannot change period or correct clock"
            )
        return
    assert event.period is not None
    assert event.period_seconds_remaining is not None
    assert event.game_seconds_remaining is not None
    period_delta = event.period - state.period
    if period_delta not in {0, 1}:
        raise NFLGameStateError(
            "event period must equal or immediately follow the state period"
        )
    if event.period_changed != (period_delta == 1):
        raise NFLGameStateError("event.period_changed does not match period transition")
    if period_delta == 0:
        clock_increased = (
            event.period_seconds_remaining > state.period_seconds_remaining
            or event.game_seconds_remaining > state.game_seconds_remaining
        )
        if clock_increased:
            raise NFLGameStateError("event clock moved backwards within a period")
        if event.clock_correction and (
            event.period_seconds_remaining
            != state.period_seconds_remaining
            or event.game_seconds_remaining
            != state.game_seconds_remaining
        ):
            raise NFLGameStateError(
                "clock correction must carry the canonical state clock"
            )
        if event.clock_correction:
            observed_period = (
                event.clock_correction_observed_period_seconds_remaining
            )
            observed_game = (
                event.clock_correction_observed_game_seconds_remaining
            )
            assert observed_period is not None
            assert observed_game is not None
            if (
                observed_period <= state.period_seconds_remaining
                and observed_game <= state.game_seconds_remaining
            ):
                raise NFLGameStateError(
                    "clock correction observed source clock did not increase"
                )
        elif (
            event.clock_correction_observed_period_seconds_remaining
            is not None
            or event.clock_correction_observed_game_seconds_remaining
            is not None
        ):
            raise NFLGameStateError(
                "observed correction clock requires a correction"
            )
        return
    if event.clock_correction:
        raise NFLGameStateError(
            "event.clock_correction requires a same-period transition"
        )
    if event.period_seconds_remaining != _period_clock_maximum(
        state.game_id,
        state.season_type,
        event.period,
    ):
        raise NFLGameStateError(
            "new period must start at the pinned full period clock"
        )
    contextless_period_source = (
        event.timeout_kind == "administrative"
        or event.play_type_nfl == "END_QUARTER"
    )
    if contextless_period_source:
        if state.period_seconds_remaining != 0:
            raise NFLGameStateError(
                "contextless period advance requires an expired prior clock"
            )
    elif not event.quarter_end:
        raise NFLGameStateError(
            "real-play period advance requires native quarter_end proof"
        )


def _validate_score_transition(state: NFLGameState, event: NFLPlayEvent) -> None:
    home_delta = event.home_score - state.home_score
    away_delta = event.away_score - state.away_score
    if home_delta < 0 or away_delta < 0:
        raise NFLGameStateError("event score cannot decrease")
    if home_delta and away_delta:
        raise NFLGameStateError("only one team may score in one NFL play event")
    score_delta = home_delta + away_delta
    if score_delta not in _LEGAL_SCORE_DELTAS:
        raise NFLGameStateError(
            "event score delta is not a legal single-play NFL score"
        )
    if event.score != (score_delta > 0):
        raise NFLGameStateError("event.score does not match the score transition")


def _validate_possession_transition(
    state: NFLGameState, event: NFLPlayEvent
) -> None:
    if event.possession_team is not None and event.possession_team not in {
        state.home_team,
        state.away_team,
    }:
        raise NFLGameStateError("event.possession_team is not a game participant")
    possession_changed = (
        state.possession_team is not None
        and event.possession_team is not None
        and state.possession_team != event.possession_team
    )
    if event.possession_changed != possession_changed:
        raise NFLGameStateError(
            "event.possession_changed does not match the possession transition"
        )
    if possession_changed and event.down not in {None, 1}:
        raise NFLGameStateError(
            "a possession switch must start at first down or a dead-ball state"
        )
    if event.turnover and not possession_changed:
        raise NFLGameStateError("event.turnover requires a possession switch")
    if event.first_down:
        if possession_changed:
            raise NFLGameStateError(
                "event.first_down cannot belong to a possession switch"
            )
        if event.down != 1 and not (event.score and event.down is None):
            raise NFLGameStateError(
                "event.first_down must produce first down or a scoring dead ball"
            )


def _derived_timeout_charge_team(
    state: NFLGameState,
    *,
    target_period: int,
    home_timeouts_remaining: int,
    away_timeouts_remaining: int,
) -> str | None:
    reset_allotment: int | None = None
    if target_period == state.period + 1:
        try:
            reset_allotment = timeout_reset_allotment(
                state.season_type,
                state.period,
                target_period,
            )
        except NFLRulesError as exc:
            raise NFLGameStateError(str(exc)) from exc
    baseline_home = (
        state.home_timeouts_remaining
        if reset_allotment is None
        else reset_allotment
    )
    baseline_away = (
        state.away_timeouts_remaining
        if reset_allotment is None
        else reset_allotment
    )
    home_delta = home_timeouts_remaining - baseline_home
    away_delta = away_timeouts_remaining - baseline_away
    if home_delta > 0 or away_delta > 0:
        raise NFLGameStateError("timeouts exceed the transition allotment")
    if home_delta < -1 or away_delta < -1:
        raise NFLGameStateError("one event cannot consume multiple team timeouts")
    if home_delta < 0 and away_delta < 0:
        raise NFLGameStateError("one event cannot consume both teams' timeouts")
    if home_delta == -1:
        return state.home_team
    if away_delta == -1:
        return state.away_team
    return None


def _validate_timeout_transition(
    state: NFLGameState, event: NFLPlayEvent
) -> None:
    if (
        event.timeout_observed
        and event.timeout_observed_team
        not in {state.home_team, state.away_team}
    ):
        raise NFLGameStateError(
            "observed timeout team is not a game participant"
        )
    target_period = state.period if event.period is None else event.period
    timeout_charge_team = _derived_timeout_charge_team(
        state,
        target_period=target_period,
        home_timeouts_remaining=event.home_timeouts_remaining,
        away_timeouts_remaining=event.away_timeouts_remaining,
    )
    if event.timeout_charge_team != timeout_charge_team:
        raise NFLGameStateError(
            "event.timeout_charge_team does not match the counter delta"
        )
    if (
        timeout_charge_team is not None
        and event.timeout_observed_team != timeout_charge_team
    ):
        raise NFLGameStateError(
            "charged timeout team must match the observed timeout team"
        )


def reduce(state: NFLGameState, event: NFLPlayEvent) -> NFLGameState:
    """Apply one normalized event, rejecting any impossible transition."""

    if not isinstance(state, NFLGameState):
        raise TypeError("state must be an NFLGameState")
    if not isinstance(event, NFLPlayEvent):
        raise TypeError("event must be an NFLPlayEvent")
    if state.sport != event.sport:
        raise NFLGameStateError("state and event sport differ")
    if state.game_id != event.game_id:
        raise NFLGameStateError("state and event game_id differ")
    if state.season_type != event.season_type:
        raise NFLGameStateError("season_type cannot change within a game")
    if event.source_play_id != state.source_play_id:
        raise NFLGameStateError(
            "event source_play_id must match current state"
        )
    if event.source_order_sequence != state.source_order_sequence:
        raise NFLGameStateError(
            "event source_order_sequence must match current state"
        )
    if event.next_source_order_sequence <= event.source_order_sequence:
        raise NFLGameStateError("next source order must strictly increase")
    if state.terminal:
        raise NFLGameStateError("cannot apply an event after terminal state")
    if event.sequence != state.sequence + 1:
        raise NFLGameStateError("event sequence must immediately follow state")
    if state.suspended and event.lifecycle_action == "none":
        raise NFLGameStateError(
            "a suspended game can only accept a resume lifecycle event"
        )
    if event.carry_forward_context and (
        event.context_source_play_id != state.context_source_play_id
        or event.context_source_order_sequence
        != state.context_source_order_sequence
    ):
        raise NFLGameStateError(
            "context-carry event context source must match current state"
        )
    _validate_clock_transition(state, event)
    _validate_score_transition(state, event)
    _validate_possession_transition(state, event)
    _validate_timeout_transition(state, event)

    if event.lifecycle_action == "suspend":
        if state.suspended:
            raise NFLGameStateError("game is already suspended")
        target_suspended = True
    elif event.lifecycle_action == "resume":
        if not state.suspended:
            raise NFLGameStateError("game is not suspended")
        target_suspended = False
    else:
        target_suspended = state.suspended

    if event.carry_forward_context:
        context_source_play_id = state.context_source_play_id
        context_source_order_sequence = (
            state.context_source_order_sequence
        )
        next_drive_id = state.drive_id
        next_play_clock_seconds = state.play_clock_seconds
        possession_team = state.possession_team
        down = state.down
        distance = state.distance
        yardline_100 = state.yardline_100
        goal_to_go = state.goal_to_go
    else:
        context_source_play_id = event.context_source_play_id
        context_source_order_sequence = (
            event.context_source_order_sequence
        )
        next_drive_id = event.next_drive_id
        next_play_clock_seconds = event.next_play_clock_seconds
        possession_team = event.possession_team
        down = event.down
        distance = event.distance
        yardline_100 = event.yardline_100
        goal_to_go = event.goal_to_go

    if event.clock_carry_forward:
        period = state.period
        period_seconds_remaining = state.period_seconds_remaining
        game_seconds_remaining = state.game_seconds_remaining
    else:
        assert event.period is not None
        assert event.period_seconds_remaining is not None
        assert event.game_seconds_remaining is not None
        period = event.period
        period_seconds_remaining = event.period_seconds_remaining
        game_seconds_remaining = event.game_seconds_remaining

    return NFLGameState(
        sport=state.sport,
        game_id=state.game_id,
        sequence=event.sequence,
        terminal=event.terminal,
        season_type=state.season_type,
        home_team=state.home_team,
        away_team=state.away_team,
        period=period,
        period_seconds_remaining=period_seconds_remaining,
        game_seconds_remaining=game_seconds_remaining,
        source_play_id=event.next_source_play_id,
        source_order_sequence=event.next_source_order_sequence,
        context_source_play_id=context_source_play_id,
        context_source_order_sequence=context_source_order_sequence,
        suspended=target_suspended,
        drive_id=next_drive_id,
        play_clock_seconds=next_play_clock_seconds,
        possession_team=possession_team,
        down=down,
        distance=distance,
        yardline_100=yardline_100,
        goal_to_go=goal_to_go,
        home_score=event.home_score,
        away_score=event.away_score,
        home_timeouts_remaining=event.home_timeouts_remaining,
        away_timeouts_remaining=event.away_timeouts_remaining,
        last_event_id=event.event_id,
    )


class NFLGameStateReducer:
    """Common-protocol adapter for the NFL module reducer."""

    __slots__ = ()

    sport = NFL_SPORT
    reducer_id = "REDUCER-NFL-PLAY-STATE"
    reducer_version = "v3"

    def reduce(
        self,
        state: NFLGameState,
        event: NFLPlayEvent,
    ) -> NFLGameState:
        return reduce(state, event)


NFL_GAME_STATE_REDUCER = NFLGameStateReducer()


def _unwrap_scalar(value: object) -> object:
    as_py = getattr(value, "as_py", None)
    return as_py() if callable(as_py) else value


def _is_missing(value: object) -> bool:
    value = _unwrap_scalar(value)
    if value is None:
        return True
    if isinstance(value, Real) and not isinstance(value, Integral):
        return not math.isfinite(float(value))
    if isinstance(value, Decimal):
        return not value.is_finite()
    try:
        unequal = value != value
        return bool(unequal)
    except (TypeError, ValueError):
        return False


def _row_value(
    row: Mapping[str, object],
    field: str,
    *,
    required: bool,
) -> object | None:
    if field not in row:
        if required:
            raise NFLGameStateError(f"nflverse row requires {field}")
        return None
    value = _unwrap_scalar(row[field])
    if _is_missing(value):
        if required:
            raise NFLGameStateError(f"nflverse row requires nonmissing {field}")
        return None
    return value


def _normalized_row(row: object, context: str) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{context} must be a mapping-like nflverse row")
    return row


def _row_int(
    row: Mapping[str, object],
    field: str,
    *,
    required: bool = True,
) -> int | None:
    value = _row_value(row, field, required=required)
    if value is None:
        return None
    if isinstance(value, bool):
        raise NFLGameStateError(f"nflverse {field} must be an integer observation")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Decimal):
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
    elif isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    raise NFLGameStateError(f"nflverse {field} must be an integer observation")


def _row_text(
    row: Mapping[str, object],
    field: str,
    *,
    required: bool = True,
) -> str | None:
    value = _row_value(row, field, required=required)
    if value is None:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise NFLGameStateError(
            f"nflverse {field} must be a canonical nonempty string"
        )
    return value


def _row_indicator(
    row: Mapping[str, object],
    field: str,
    *,
    default: bool,
) -> bool:
    value = _row_value(row, field, required=False)
    if value is None:
        return default
    if type(value) is bool:
        return value
    if isinstance(value, Integral) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, Real):
        number = float(value)
        if number in {0.0, 1.0}:
            return bool(int(number))
    raise NFLGameStateError(f"nflverse {field} must be a binary observation")


def _source_play_id(row: Mapping[str, object]) -> str:
    value = _row_value(row, "play_id", required=True)
    assert value is not None
    if isinstance(value, bool):
        raise NFLGameStateError("nflverse play_id must be a stable scalar")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if type(value) is str and value and value == value.strip():
        return value
    raise NFLGameStateError("nflverse play_id must be a stable scalar")


def _source_order_sequence(row: Mapping[str, object]) -> int:
    value = _row_int(row, "order_sequence")
    assert value is not None
    if value < 0:
        raise NFLGameStateError(
            "nflverse order_sequence must be nonnegative"
        )
    return value


def _lifecycle_action(
    row: Mapping[str, object],
) -> Literal["none", "suspend", "resume"]:
    if _row_text(row, "play_type_nfl", required=False) != "COMMENT":
        return "none"
    description = _row_text(row, "desc", required=False)
    if (
        description is not None
        and description.startswith("The game has been suspended.")
    ):
        return "suspend"
    if (
        description is not None
        and description.startswith("The game has resumed.")
    ):
        return "resume"
    return "none"


def _timeout_kind(
    row: Mapping[str, object],
) -> Literal["none", "administrative", "play_attached"]:
    timeout_observed = _row_indicator(row, "timeout", default=False)
    if not timeout_observed:
        return "none"
    return (
        "administrative"
        if _administrative_timeout_source(row)
        else "play_attached"
    )


def _administrative_timeout_source(row: Mapping[str, object]) -> bool:
    return (
        _row_text(row, "play_type_nfl", required=False) == "TIMEOUT"
        and _row_text(row, "play_type", required=False) == "no_play"
    )


def _stable_optional_scalar(
    row: Mapping[str, object],
    field: str,
) -> str | None:
    value = _row_value(row, field, required=False)
    if value is None:
        return None
    if isinstance(value, bool):
        raise NFLGameStateError(f"nflverse {field} must be a stable scalar")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if type(value) is str and value and value == value.strip():
        return value
    raise NFLGameStateError(f"nflverse {field} must be a stable scalar")


def _play_clock_seconds(row: Mapping[str, object]) -> int | None:
    value = _row_value(row, "play_clock", required=False)
    if value is None:
        return None
    if type(value) is str:
        if not value.isdigit():
            raise NFLGameStateError("nflverse play_clock must be integer seconds")
        number = int(value)
    elif isinstance(value, Integral) and not isinstance(value, bool):
        number = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        number = int(value)
    else:
        raise NFLGameStateError("nflverse play_clock must be integer seconds")
    if not 0 <= number <= 40:
        raise NFLGameStateError("nflverse play_clock must be in [0, 40]")
    return number


def _quarter_seconds(row: Mapping[str, object]) -> int:
    observed = _row_int(
        row,
        "quarter_seconds_remaining",
        required=False,
    )
    if observed is not None:
        return observed
    clock = _row_text(row, "time", required=True)
    assert clock is not None
    parts = clock.split(":")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise NFLGameStateError("nflverse time must be canonical MM:SS")
    minutes, seconds = (int(part) for part in parts)
    if minutes > 15 or seconds > 59:
        raise NFLGameStateError("nflverse time must be a valid period clock")
    return minutes * 60 + seconds


def _canonical_nflverse_game_id(value: str | None) -> str:
    if value is None:
        raise NFLGameStateError("nflverse row requires game_id")
    candidate = (
        value
        if value.startswith("game_nflverse_")
        else f"game_nflverse_{value}"
    )
    _require_game_id(candidate, "nflverse game_id")
    return candidate


def _home_away_scores(
    row: Mapping[str, object],
    *,
    after_play: bool,
) -> tuple[int, int]:
    home_team = _row_text(row, "home_team")
    away_team = _row_text(row, "away_team")
    posteam = _row_text(row, "posteam", required=False)
    posteam_field = "posteam_score_post" if after_play else "posteam_score"
    defteam_field = "defteam_score_post" if after_play else "defteam_score"

    if posteam is None:
        if (
            _row_value(row, posteam_field, required=False) is not None
            or _row_value(row, defteam_field, required=False) is not None
        ):
            raise NFLGameStateError(
                "nflverse no-possession row cannot carry team-relative scores"
            )
        home_score = _row_int(row, "total_home_score")
        away_score = _row_int(row, "total_away_score")
        assert home_score is not None
        assert away_score is not None
        return home_score, away_score

    if posteam not in {home_team, away_team}:
        raise NFLGameStateError(
            "nflverse posteam must identify the home or away team"
        )
    posteam_score = _row_int(row, posteam_field)
    defteam_score = _row_int(row, defteam_field)
    assert posteam_score is not None
    assert defteam_score is not None
    if posteam == home_team:
        scores = (posteam_score, defteam_score)
    else:
        scores = (defteam_score, posteam_score)

    if after_play:
        total_scores = (
            _row_int(row, "total_home_score"),
            _row_int(row, "total_away_score"),
        )
        if scores != total_scores:
            raise NFLGameStateError(
                "nflverse team-relative post-play scores disagree with totals"
            )
    return scores


def _snapshot_from_row(
    row: Mapping[str, object],
    *,
    clock_carry_forward: bool = False,
) -> dict[str, object]:
    game_id = _canonical_nflverse_game_id(_row_text(row, "game_id"))
    season_type = _row_text(row, "season_type")
    assert season_type is not None
    _rule_context(game_id, season_type)
    period = _row_int(row, "qtr")
    assert period is not None
    down = _row_int(row, "down", required=False)
    distance = (
        _row_int(row, "ydstogo", required=True) if down is not None else None
    )
    home_score, away_score = _home_away_scores(row, after_play=False)
    native_home_timeouts = _row_int(row, "home_timeouts_remaining")
    native_away_timeouts = _row_int(row, "away_timeouts_remaining")
    assert native_home_timeouts is not None
    assert native_away_timeouts is not None
    try:
        home_timeouts = normalize_native_timeout_remaining(
            native_home_timeouts,
            season_type=season_type,
            period=period,
        )
        away_timeouts = normalize_native_timeout_remaining(
            native_away_timeouts,
            season_type=season_type,
            period=period,
        )
    except NFLRulesError as exc:
        raise NFLGameStateError(str(exc)) from exc
    return {
        "game_id": game_id,
        "season_type": season_type,
        "home_team": _row_text(row, "home_team"),
        "away_team": _row_text(row, "away_team"),
        "period": None if clock_carry_forward else period,
        "period_seconds_remaining": (
            None if clock_carry_forward else _quarter_seconds(row)
        ),
        "game_seconds_remaining": (
            None
            if clock_carry_forward
            else _row_int(row, "game_seconds_remaining")
        ),
        "source_play_id": _source_play_id(row),
        "source_order_sequence": _source_order_sequence(row),
        "context_source_play_id": _source_play_id(row),
        "context_source_order_sequence": _source_order_sequence(row),
        "drive_id": _stable_optional_scalar(row, "fixed_drive"),
        "play_clock_seconds": _play_clock_seconds(row),
        "possession_team": _row_text(row, "posteam", required=False),
        "down": down,
        "distance": distance,
        "yardline_100": _row_int(row, "yardline_100", required=False),
        "goal_to_go": _row_indicator(row, "goal_to_go", default=False),
        # Team-relative scores are the canonical start-of-play observation.
        # `total_*` is end-of-play on scoring rows and cannot define this state.
        "home_score": home_score,
        "away_score": away_score,
        "home_timeouts_remaining": home_timeouts,
        "away_timeouts_remaining": away_timeouts,
    }


def state_from_nflverse_row(
    row: Mapping[str, object],
    *,
    sequence: int = 0,
    terminal: bool = False,
    last_event_id: str | None = None,
) -> NFLGameState:
    """Build one immutable state from a single offline nflverse observation."""

    normalized = _normalized_row(row, "row")
    snapshot = _snapshot_from_row(normalized)
    return NFLGameState(
        sport=NFL_SPORT,
        sequence=sequence,
        terminal=terminal,
        suspended=False,
        last_event_id=last_event_id,
        **snapshot,
    )


def _infer_terminal(row: Mapping[str, object]) -> bool:
    for field in ("terminal", "game_end"):
        if field in row and not _is_missing(row[field]):
            return _row_indicator(row, field, default=False)
    description = _row_text(row, "desc", required=False)
    if description is None:
        return False
    return description.upper() in {"END GAME", "END OF GAME"}


def _turnover_observed(row: Mapping[str, object]) -> bool:
    if _row_indicator(row, "interception", default=False):
        return True
    if _row_indicator(row, "fumble_lost", default=False):
        return True
    return False


def nflverse_transition_payload(
    state: NFLGameState,
    source_row: Mapping[str, object],
    successor_rows: tuple[Mapping[str, object], ...],
    *,
    sequence: int = 1,
) -> dict[str, object]:
    """Normalize one source row plus its complete causal successor window."""

    if not isinstance(state, NFLGameState):
        raise TypeError("state must be an NFLGameState")
    pre = _normalized_row(source_row, "source_row")
    if type(successor_rows) is not tuple or not successor_rows:
        raise NFLGameStateError(
            "successor_rows must be a nonempty canonical tuple"
        )
    successors = tuple(
        _normalized_row(row, "successor_rows[]") for row in successor_rows
    )
    lifecycle_action = _lifecycle_action(pre)
    pre_snapshot = _snapshot_from_row(
        pre,
        clock_carry_forward=lifecycle_action != "none",
    )
    successor_snapshots = tuple(
        _snapshot_from_row(
            row,
            clock_carry_forward=_lifecycle_action(row) != "none",
        )
        for row in successors
    )
    post = successors[0]
    post_snapshot = successor_snapshots[0]
    for field in ("game_id", "season_type", "home_team", "away_team"):
        if getattr(state, field) != pre_snapshot[field]:
            raise NFLGameStateError(
                f"state and nflverse source observation differ on {field}"
            )
        if any(
            pre_snapshot[field] != snapshot[field]
            for snapshot in successor_snapshots
        ):
            raise NFLGameStateError(
                f"nflverse source window differs on {field}"
            )
    if (
        state.source_play_id != pre_snapshot["source_play_id"]
        or state.source_order_sequence
        != pre_snapshot["source_order_sequence"]
    ):
        raise NFLGameStateError(
            "state source identity must match the nflverse source row"
        )

    source_window_snapshots = (pre_snapshot, *successor_snapshots)
    source_window_orders = tuple(
        int(snapshot["source_order_sequence"])
        for snapshot in source_window_snapshots
    )
    source_window_play_ids = tuple(
        str(snapshot["source_play_id"])
        for snapshot in source_window_snapshots
    )
    if any(
        later <= earlier
        for earlier, later in zip(
            source_window_orders,
            source_window_orders[1:],
        )
    ):
        raise NFLGameStateError(
            "nflverse source window order must strictly increase"
        )

    source_timeout_kind = _timeout_kind(pre)
    source_administrative_timeout = _administrative_timeout_source(pre)
    source_quarter_end = _row_indicator(
        pre,
        "quarter_end",
        default=False,
    )
    clock_correction = bool(
        source_administrative_timeout
        and int(pre_snapshot["period"]) == state.period
        and _is_inserted_native_play_boundary(
            str(pre_snapshot["source_play_id"]),
            str(post_snapshot["source_play_id"]),
        )
        and (
            int(pre_snapshot["period_seconds_remaining"])
            > state.period_seconds_remaining
            or int(pre_snapshot["game_seconds_remaining"])
            > state.game_seconds_remaining
        )
    )
    carry_forward_context = bool(
        lifecycle_action != "none"
        or source_administrative_timeout
        or source_quarter_end
    )
    if carry_forward_context:
        if len(successors) != 1:
            raise NFLGameStateError(
                "context-carry source requires only its immediate successor"
            )
        context_snapshot: Mapping[str, object] = {
            "drive_id": state.drive_id,
            "play_clock_seconds": state.play_clock_seconds,
            "possession_team": state.possession_team,
            "down": state.down,
            "distance": state.distance,
            "yardline_100": state.yardline_100,
            "goal_to_go": state.goal_to_go,
        }
        context_source_play_id = state.context_source_play_id
        context_source_order_sequence = (
            state.context_source_order_sequence
        )
    else:
        complete_positions = tuple(
            index
            for index, snapshot in enumerate(successor_snapshots)
            if snapshot["possession_team"] is not None
        )
        if complete_positions != (len(successor_snapshots) - 1,):
            raise NFLGameStateError(
                "successor window must end at its first complete context"
            )
        context_snapshot = successor_snapshots[-1]
        context_source_play_id = str(
            context_snapshot["source_play_id"]
        )
        context_source_order_sequence = int(
            context_snapshot["source_order_sequence"]
        )

    post_lifecycle_action = _lifecycle_action(post)
    post_administrative_timeout = _administrative_timeout_source(post)
    if lifecycle_action != "none":
        clock_snapshot = None
    elif clock_correction:
        clock_snapshot = {
            "period": state.period,
            "period_seconds_remaining": (
                state.period_seconds_remaining
            ),
            "game_seconds_remaining": state.game_seconds_remaining,
        }
    elif (
        source_administrative_timeout
        or post_lifecycle_action != "none"
        or (
            post_administrative_timeout
            and post_snapshot["period"] == pre_snapshot["period"]
        )
    ):
        clock_snapshot = pre_snapshot
    else:
        clock_snapshot = post_snapshot
    pre_possession = state.possession_team
    post_possession = context_snapshot["possession_team"]
    possession_changed = (
        pre_possession is not None
        and post_possession is not None
        and pre_possession != post_possession
    )
    home_score, away_score = _home_away_scores(pre, after_play=True)
    score = (
        pre_snapshot["home_score"] != home_score
        or pre_snapshot["away_score"] != away_score
    )
    target_period = (
        state.period
        if clock_snapshot is None
        else int(clock_snapshot["period"])
    )
    period_changed = state.period != target_period
    timeout_observed = _row_indicator(pre, "timeout", default=False)
    timeout_observed_team = (
        _row_text(pre, "timeout_team") if timeout_observed else None
    )
    if timeout_observed:
        target_home_timeouts = int(
            pre_snapshot["home_timeouts_remaining"]
        )
        target_away_timeouts = int(
            pre_snapshot["away_timeouts_remaining"]
        )
    elif period_changed:
        try:
            reset_allotment = timeout_reset_allotment(
                state.season_type,
                state.period,
                target_period,
            )
        except NFLRulesError as exc:
            raise NFLGameStateError(str(exc)) from exc
        target_home_timeouts = (
            state.home_timeouts_remaining
            if reset_allotment is None
            else reset_allotment
        )
        target_away_timeouts = (
            state.away_timeouts_remaining
            if reset_allotment is None
            else reset_allotment
        )
    else:
        target_home_timeouts = state.home_timeouts_remaining
        target_away_timeouts = state.away_timeouts_remaining
    timeout_charge_team = _derived_timeout_charge_team(
        state,
        target_period=target_period,
        home_timeouts_remaining=target_home_timeouts,
        away_timeouts_remaining=target_away_timeouts,
    )

    observed_first_down = _row_indicator(pre, "first_down", default=False)
    first_down = observed_first_down and not possession_changed and (
        context_snapshot["down"] == 1
        or (score and context_snapshot["down"] is None)
    )
    turnover = _turnover_observed(pre) and possession_changed

    quality_flags = (
        ("source_order_inserted_timeout",)
        if clock_correction
        else ()
    )
    target_terminal = _infer_terminal(post)
    _require_bool(target_terminal, "terminal")
    if lifecycle_action != "none":
        target_terminal = False
    return {
        "sport": NFL_SPORT,
        "game_id": str(post_snapshot["game_id"]),
        "sequence": sequence,
        "season_type": str(pre_snapshot["season_type"]),
        "source_play_id": _source_play_id(pre),
        "source_order_sequence": int(
            pre_snapshot["source_order_sequence"]
        ),
        "observation_mode": NFLVERSE_OBSERVATION_MODE,
        "play_type": _row_text(pre, "play_type", required=False),
        "play_type_nfl": _row_text(
            pre,
            "play_type_nfl",
            required=False,
        ),
        "description": _row_text(pre, "desc", required=False),
        "period": (
            None
            if clock_snapshot is None
            else int(clock_snapshot["period"])
        ),
        "period_seconds_remaining": (
            None
            if clock_snapshot is None
            else int(clock_snapshot["period_seconds_remaining"])
        ),
        "game_seconds_remaining": (
            None
            if clock_snapshot is None
            else int(clock_snapshot["game_seconds_remaining"])
        ),
        "next_source_play_id": str(post_snapshot["source_play_id"]),
        "next_source_order_sequence": int(
            post_snapshot["source_order_sequence"]
        ),
        "context_source_play_id": context_source_play_id,
        "context_source_order_sequence": (
            context_source_order_sequence
        ),
        "source_window_play_ids": list(source_window_play_ids),
        "source_window_order_sequences": list(source_window_orders),
        "lifecycle_action": lifecycle_action,
        "clock_carry_forward": lifecycle_action != "none",
        "next_drive_id": (
            None
            if context_snapshot["drive_id"] is None
            else str(context_snapshot["drive_id"])
        ),
        "next_play_clock_seconds": (
            None
            if context_snapshot["play_clock_seconds"] is None
            else int(context_snapshot["play_clock_seconds"])
        ),
        "possession_team": (
            None
            if context_snapshot["possession_team"] is None
            else str(context_snapshot["possession_team"])
        ),
        "down": (
            None
            if context_snapshot["down"] is None
            else int(context_snapshot["down"])
        ),
        "distance": (
            None
            if context_snapshot["distance"] is None
            else int(context_snapshot["distance"])
        ),
        "yardline_100": (
            None
            if context_snapshot["yardline_100"] is None
            else int(context_snapshot["yardline_100"])
        ),
        "goal_to_go": bool(context_snapshot["goal_to_go"]),
        "home_score": home_score,
        "away_score": away_score,
        "home_timeouts_remaining": target_home_timeouts,
        "away_timeouts_remaining": target_away_timeouts,
        "first_down": first_down,
        "turnover": turnover,
        "possession_changed": possession_changed,
        "score": score,
        "timeout_observed": timeout_observed,
        "timeout_observed_team": timeout_observed_team,
        "timeout_kind": source_timeout_kind,
        "timeout_charge_team": timeout_charge_team,
        "quarter_end": source_quarter_end,
        "clock_correction": clock_correction,
        "clock_correction_observed_period_seconds_remaining": (
            int(pre_snapshot["period_seconds_remaining"])
            if clock_correction
            else None
        ),
        "clock_correction_observed_game_seconds_remaining": (
            int(pre_snapshot["game_seconds_remaining"])
            if clock_correction
            else None
        ),
        "carry_forward_context": carry_forward_context,
        "period_changed": period_changed,
        "terminal": target_terminal,
        "quality_flags": list(sorted(quality_flags)),
    }


def event_from_nflverse_envelope(
    envelope: EventEnvelopeV0,
    *,
    program_root: str | Path,
    raw_parents: tuple[EventEnvelopeV0, ...],
) -> NFLPlayEvent:
    """Construct an NFL event only from a fully bound normalized envelope."""

    validated = validate_static_sport_observation_bundle(
        program_root,
        envelope,
        raw_parents=raw_parents,
        expected_experiment_id="X-11",
        expected_dataset_id="DS-NFLVERSE",
        expected_source_system="nflverse",
        expected_source_stream="play_by_play",
        expected_native_namespace="nflverse.play",
    )
    payload = dict(validated.payload)
    expected_payload_fields = {
        item.name for item in fields(NFLPlayEvent)
    } - {"event_id"}
    if set(payload) != expected_payload_fields:
        raise NFLGameStateError(
            "normalized nflverse payload fields are incomplete or unexpected"
        )
    if payload["sport"] != NFL_SPORT:
        raise NFLGameStateError("normalized nflverse payload sport must be nfl")

    canonical_game_id = validated.canonical_refs.game_id
    if canonical_game_id is None or not canonical_game_id.startswith(
        "game_nflverse_"
    ):
        raise NFLGameStateError(
            "normalized nflverse envelope requires a canonical nflverse game_id"
        )
    payload["quality_flags"] = tuple(payload["quality_flags"])
    payload["source_window_play_ids"] = tuple(
        payload["source_window_play_ids"]
    )
    payload["source_window_order_sequences"] = tuple(
        payload["source_window_order_sequences"]
    )
    native_game_id = canonical_game_id.removeprefix("game_nflverse_")
    expected_native_ids = tuple(
        f"{native_game_id}:{play_id}"
        for play_id in payload["source_window_play_ids"]
    )
    parent_native_ids = tuple(
        parent.native_refs[0].native_id for parent in raw_parents
    )
    envelope_native_ids = {
        reference.native_id for reference in validated.native_refs
    }
    if (
        len(raw_parents) != len(expected_native_ids)
        or parent_native_ids != expected_native_ids
        or envelope_native_ids != set(expected_native_ids)
    ):
        raise NFLGameStateError(
            "normalized nflverse source window does not match raw native identity"
        )

    event = NFLPlayEvent(event_id=validated.event_id, **payload)
    if event.game_id != canonical_game_id:
        raise NFLGameStateError(
            "normalized nflverse event game does not match envelope game"
        )
    return event


# Descriptive alias for callers that use "observation" rather than "row".
state_from_nflverse_observation = state_from_nflverse_row


__all__ = [
    "NFL_GAME_STATE_REDUCER",
    "NFLGameState",
    "NFLGameStateReducer",
    "NFLGameStateError",
    "NFLPlayEvent",
    "NFLVERSE_LEAKAGE_FIELDS",
    "NFLVERSE_OBSERVATION_MODE",
    "NFL_SPORT",
    "event_from_nflverse_envelope",
    "nflverse_transition_payload",
    "reduce",
    "state_from_nflverse_observation",
    "state_from_nflverse_row",
]
