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
from typing import Any

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.sports.event_envelopes import (
    validate_static_sport_observation_bundle,
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


def _validate_clock(
    period: object,
    period_seconds_remaining: object,
    game_seconds_remaining: object,
    *,
    context: str,
) -> None:
    _require_int(period, f"{context}.period", minimum=1)
    _require_int(
        period_seconds_remaining,
        f"{context}.period_seconds_remaining",
        minimum=0,
        maximum=900,
    )
    _require_int(
        game_seconds_remaining,
        f"{context}.game_seconds_remaining",
        minimum=0,
        maximum=3600,
    )
    assert type(period) is int
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
    _require_int(
        home_timeouts_remaining,
        f"{context}.home_timeouts_remaining",
        minimum=0,
        maximum=3,
    )
    _require_int(
        away_timeouts_remaining,
        f"{context}.away_timeouts_remaining",
        minimum=0,
        maximum=3,
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
    home_team: str
    away_team: str
    period: int
    period_seconds_remaining: int
    game_seconds_remaining: int
    source_play_id: str
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
        _require_text(self.home_team, "state.home_team")
        _require_text(self.away_team, "state.away_team")
        if self.home_team == self.away_team:
            raise NFLGameStateError("home_team and away_team must differ")
        _require_text(self.source_play_id, "state.source_play_id")
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
            self.period,
            self.period_seconds_remaining,
            self.game_seconds_remaining,
            context="state",
        )
        _validate_football_values(
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
    source_play_id: str
    observation_mode: str
    play_type: str | None
    description: str | None
    period: int
    period_seconds_remaining: int
    game_seconds_remaining: int
    next_source_play_id: str
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
    timeout: bool
    timeout_team: str | None
    period_changed: bool
    terminal: bool
    quality_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.sport != NFL_SPORT:
            raise NFLGameStateError("event.sport must be nfl")
        _require_game_id(self.game_id, "event.game_id")
        _require_int(self.sequence, "event.sequence", minimum=1)
        _require_text(self.source_play_id, "event.source_play_id")
        if self.observation_mode != NFLVERSE_OBSERVATION_MODE:
            raise NFLGameStateError(
                "event.observation_mode must be explicit offline observation"
            )
        if self.play_type is not None:
            _require_text(self.play_type, "event.play_type")
        if self.description is not None:
            _require_text(self.description, "event.description")
        _require_text(self.next_source_play_id, "event.next_source_play_id")
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
        _validate_clock(
            self.period,
            self.period_seconds_remaining,
            self.game_seconds_remaining,
            context="event",
        )
        _validate_football_values(
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
            "timeout",
            "period_changed",
            "terminal",
        ):
            _require_bool(getattr(self, field), f"event.{field}")
        if self.timeout:
            _require_text(self.timeout_team, "event.timeout_team")
        elif self.timeout_team is not None:
            raise NFLGameStateError(
                "event.timeout_team must be None when timeout is false"
            )
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


def _validate_clock_transition(state: NFLGameState, event: NFLPlayEvent) -> None:
    period_delta = event.period - state.period
    if period_delta not in {0, 1}:
        raise NFLGameStateError(
            "event period must equal or immediately follow the state period"
        )
    if event.period_changed != (period_delta == 1):
        raise NFLGameStateError("event.period_changed does not match period transition")
    if period_delta == 0:
        if (
            event.period_seconds_remaining > state.period_seconds_remaining
            or event.game_seconds_remaining > state.game_seconds_remaining
        ):
            raise NFLGameStateError("event clock moved backwards within a period")
    elif (
        state.period < 4
        and event.game_seconds_remaining > state.game_seconds_remaining
    ):
        raise NFLGameStateError("event regulation clock moved backwards")


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


def _validate_timeout_transition(
    state: NFLGameState, event: NFLPlayEvent
) -> None:
    home_delta = (
        event.home_timeouts_remaining - state.home_timeouts_remaining
    )
    away_delta = (
        event.away_timeouts_remaining - state.away_timeouts_remaining
    )
    reset_allowed = event.period_changed and (
        (state.period == 2 and event.period == 3)
        or (state.period >= 4 and event.period == state.period + 1)
    )
    if (home_delta > 0 or away_delta > 0) and not reset_allowed:
        raise NFLGameStateError("timeouts may increase only at a rules reset")
    if state.period == 2 and event.period == 3:
        if (
            home_delta > 0
            and event.home_timeouts_remaining != 3
            or away_delta > 0
            and event.away_timeouts_remaining != 3
        ):
            raise NFLGameStateError("halftime timeout reset must restore three")
    if home_delta < -1 or away_delta < -1:
        raise NFLGameStateError("one event cannot consume multiple team timeouts")
    if home_delta < 0 and away_delta < 0:
        raise NFLGameStateError("one event cannot consume both teams' timeouts")
    timeout_team: str | None = None
    if home_delta == -1:
        timeout_team = state.home_team
    elif away_delta == -1:
        timeout_team = state.away_team
    if event.timeout != (timeout_team is not None):
        raise NFLGameStateError("event.timeout does not match timeout counters")
    if event.timeout_team != timeout_team:
        raise NFLGameStateError(
            "event.timeout_team does not match the timeout transition"
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
    if state.terminal:
        raise NFLGameStateError("cannot apply an event after terminal state")
    if event.sequence != state.sequence + 1:
        raise NFLGameStateError("event sequence must immediately follow state")

    _validate_clock_transition(state, event)
    _validate_score_transition(state, event)
    _validate_possession_transition(state, event)
    _validate_timeout_transition(state, event)

    return NFLGameState(
        sport=state.sport,
        game_id=state.game_id,
        sequence=event.sequence,
        terminal=event.terminal,
        home_team=state.home_team,
        away_team=state.away_team,
        period=event.period,
        period_seconds_remaining=event.period_seconds_remaining,
        game_seconds_remaining=event.game_seconds_remaining,
        source_play_id=event.next_source_play_id,
        drive_id=event.next_drive_id,
        play_clock_seconds=event.next_play_clock_seconds,
        possession_team=event.possession_team,
        down=event.down,
        distance=event.distance,
        yardline_100=event.yardline_100,
        goal_to_go=event.goal_to_go,
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
    reducer_version = "v1"

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


def _snapshot_from_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    down = _row_int(row, "down", required=False)
    distance = (
        _row_int(row, "ydstogo", required=True) if down is not None else None
    )
    return {
        "game_id": _canonical_nflverse_game_id(
            _row_text(row, "game_id")
        ),
        "home_team": _row_text(row, "home_team"),
        "away_team": _row_text(row, "away_team"),
        "period": _row_int(row, "qtr"),
        "period_seconds_remaining": _quarter_seconds(row),
        "game_seconds_remaining": _row_int(row, "game_seconds_remaining"),
        "source_play_id": _source_play_id(row),
        "drive_id": _stable_optional_scalar(row, "fixed_drive"),
        "play_clock_seconds": _play_clock_seconds(row),
        "possession_team": _row_text(row, "posteam", required=False),
        "down": down,
        "distance": distance,
        "yardline_100": _row_int(row, "yardline_100", required=False),
        "goal_to_go": _row_indicator(row, "goal_to_go", default=False),
        # nflverse explicitly defines these as scores at the start of the play.
        # `home_score` and `away_score` are final-game values and never fallback.
        "home_score": _row_int(row, "total_home_score"),
        "away_score": _row_int(row, "total_away_score"),
        "home_timeouts_remaining": _row_int(
            row, "home_timeouts_remaining"
        ),
        "away_timeouts_remaining": _row_int(
            row, "away_timeouts_remaining"
        ),
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
    pre_row: Mapping[str, object],
    post_row: Mapping[str, object],
    *,
    sequence: int = 1,
    terminal: bool | None = None,
    quality_flags: tuple[str, ...] = (),
) -> dict[str, object]:
    """Normalize consecutive nflverse rows into a complete envelope payload."""

    pre = _normalized_row(pre_row, "pre_row")
    post = _normalized_row(post_row, "post_row")
    pre_snapshot = _snapshot_from_row(pre)
    post_snapshot = _snapshot_from_row(post)
    for field in ("game_id", "home_team", "away_team"):
        if pre_snapshot[field] != post_snapshot[field]:
            raise NFLGameStateError(
                f"nflverse pre/post observations differ on {field}"
            )

    pre_possession = pre_snapshot["possession_team"]
    post_possession = post_snapshot["possession_team"]
    possession_changed = (
        pre_possession is not None
        and post_possession is not None
        and pre_possession != post_possession
    )
    score = (
        pre_snapshot["home_score"] != post_snapshot["home_score"]
        or pre_snapshot["away_score"] != post_snapshot["away_score"]
    )
    period_changed = pre_snapshot["period"] != post_snapshot["period"]
    home_timeout_decreased = (
        post_snapshot["home_timeouts_remaining"]
        < pre_snapshot["home_timeouts_remaining"]
    )
    away_timeout_decreased = (
        post_snapshot["away_timeouts_remaining"]
        < pre_snapshot["away_timeouts_remaining"]
    )
    # nflverse rows are pre-play snapshots.  A flag on ``pre`` describes the
    # play represented by that row, while the reducer commits only state
    # changes verified in the following snapshot.  Timeout counters therefore
    # define the transition; copying the pre-row flag shifts timeouts by one
    # row and emits an unreducible event.
    timeout_observed = home_timeout_decreased or away_timeout_decreased
    timeout_team: str | None = None
    if home_timeout_decreased and not away_timeout_decreased:
        timeout_team = str(pre_snapshot["home_team"])
    elif away_timeout_decreased and not home_timeout_decreased:
        timeout_team = str(pre_snapshot["away_team"])

    observed_first_down = _row_indicator(pre, "first_down", default=False)
    first_down = observed_first_down and not possession_changed and (
        post_snapshot["down"] == 1
        or (score and post_snapshot["down"] is None)
    )
    turnover = _turnover_observed(pre) and possession_changed

    if (
        type(quality_flags) is not tuple
        or len(quality_flags) != len(set(quality_flags))
    ):
        raise NFLGameStateError("quality_flags must be a unique tuple")
    target_terminal = _infer_terminal(post) if terminal is None else terminal
    _require_bool(target_terminal, "terminal")
    return {
        "sport": NFL_SPORT,
        "game_id": str(post_snapshot["game_id"]),
        "sequence": sequence,
        "source_play_id": _source_play_id(pre),
        "observation_mode": NFLVERSE_OBSERVATION_MODE,
        "play_type": _row_text(pre, "play_type", required=False),
        "description": _row_text(pre, "desc", required=False),
        "period": int(post_snapshot["period"]),
        "period_seconds_remaining": int(
            post_snapshot["period_seconds_remaining"]
        ),
        "game_seconds_remaining": int(post_snapshot["game_seconds_remaining"]),
        "next_source_play_id": str(post_snapshot["source_play_id"]),
        "next_drive_id": (
            None
            if post_snapshot["drive_id"] is None
            else str(post_snapshot["drive_id"])
        ),
        "next_play_clock_seconds": (
            None
            if post_snapshot["play_clock_seconds"] is None
            else int(post_snapshot["play_clock_seconds"])
        ),
        "possession_team": (
            None
            if post_snapshot["possession_team"] is None
            else str(post_snapshot["possession_team"])
        ),
        "down": (
            None if post_snapshot["down"] is None else int(post_snapshot["down"])
        ),
        "distance": (
            None
            if post_snapshot["distance"] is None
            else int(post_snapshot["distance"])
        ),
        "yardline_100": (
            None
            if post_snapshot["yardline_100"] is None
            else int(post_snapshot["yardline_100"])
        ),
        "goal_to_go": bool(post_snapshot["goal_to_go"]),
        "home_score": int(post_snapshot["home_score"]),
        "away_score": int(post_snapshot["away_score"]),
        "home_timeouts_remaining": int(
            post_snapshot["home_timeouts_remaining"]
        ),
        "away_timeouts_remaining": int(
            post_snapshot["away_timeouts_remaining"]
        ),
        "first_down": first_down,
        "turnover": turnover,
        "possession_changed": possession_changed,
        "score": score,
        "timeout": timeout_observed,
        "timeout_team": timeout_team,
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
    if len(raw_parents) != 2:
        raise NFLGameStateError(
            "an nflverse transition requires exactly two raw row parents"
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
    native_game_id = canonical_game_id.removeprefix("game_nflverse_")
    ordered_parents = tuple(
        sorted(
            raw_parents,
            key=lambda parent: int(parent.lineage.raw_record_ordinal),
        )
    )
    expected_native_ids = (
        f"{native_game_id}:{payload['source_play_id']}",
        f"{native_game_id}:{payload['next_source_play_id']}",
    )
    actual_native_ids = tuple(
        parent.native_refs[0].native_id for parent in ordered_parents
    )
    if actual_native_ids != expected_native_ids:
        raise NFLGameStateError(
            "normalized nflverse payload does not match raw native play identity"
        )

    payload["quality_flags"] = tuple(payload["quality_flags"])
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
