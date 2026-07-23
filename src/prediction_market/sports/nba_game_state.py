"""Immutable NBA game-state transitions for the governed X-06 harness.

This module is deliberately limited to source-observed state.  It contains no
win probability, next-possession probability, future final-score label, or
other predictive output.  Until O-005 is green, its only authorized
observation mode is the registered synthetic X-06 contract fixture.
"""

from __future__ import annotations

import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from typing import Literal


NBA_SPORT = "nba"
NBA_OBSERVATION_MODE = "synthetic_fixture"
NBA_REDUCER_ID = "saf.nba.game_state"
NBA_REDUCER_VERSION = "v0"
NBAEventKind = Literal[
    "score",
    "foul",
    "timeout",
    "possession",
    "period",
    "terminal",
]

_EVENT_ID_RE = re.compile(r"evt_[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"game_[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_EVENT_KINDS = frozenset(
    {"score", "foul", "timeout", "possession", "period", "terminal"}
)


class NBAGameStateError(ValueError):
    """An NBA state, event, or transition violated a fail-closed invariant."""


def _require_text(value: object, field_name: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise NBAGameStateError(
            f"{field_name} must be a canonical nonempty string"
        )


def _require_int(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise NBAGameStateError(f"{field_name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise NBAGameStateError(
            f"{field_name} must be at least {minimum}{upper}"
        )


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise NBAGameStateError(f"{field_name} must be boolean")


def _period_clock_max(period: int) -> int:
    return 720_000 if period <= 4 else 300_000


def _validate_observed_clock(
    *,
    period: object,
    clock_ms: object,
    shot_clock_ms: object,
    live_ball: object,
    possession_team: object,
    context: str,
) -> None:
    _require_int(period, f"{context}.period", minimum=1, maximum=20)
    assert type(period) is int
    _require_int(
        clock_ms,
        f"{context}.clock_ms",
        minimum=0,
        maximum=_period_clock_max(period),
    )
    _require_bool(live_ball, f"{context}.live_ball")
    if shot_clock_ms is not None:
        _require_int(
            shot_clock_ms,
            f"{context}.shot_clock_ms",
            minimum=0,
            maximum=24_000,
        )
        if possession_team is None:
            raise NBAGameStateError(
                f"{context}.shot_clock_ms requires a possession team"
            )
    if live_ball and (possession_team is None or shot_clock_ms is None):
        raise NBAGameStateError(
            f"{context}.live_ball requires possession and shot clock"
        )


@dataclass(frozen=True, slots=True)
class NBAGameState:
    """Complete source-observed NBA state immediately after one event."""

    sport: str
    game_id: str
    sequence: int
    terminal: bool
    state_kind: str
    observation_mode: str
    home_team: str
    away_team: str
    period: int
    clock_ms: int
    home_score: int
    away_score: int
    possession_team: str | None
    home_team_fouls: int
    away_team_fouls: int
    home_timeouts_remaining: int
    away_timeouts_remaining: int
    home_in_bonus: bool
    away_in_bonus: bool
    shot_clock_ms: int | None
    live_ball: bool
    last_event_id: str | None
    terminal_reason: str | None

    def __post_init__(self) -> None:
        if self.sport != NBA_SPORT:
            raise NBAGameStateError("state.sport must be nba")
        if (
            type(self.game_id) is not str
            or _GAME_ID_RE.fullmatch(self.game_id) is None
        ):
            raise NBAGameStateError("state.game_id must be canonical game_<id>")
        _require_int(self.sequence, "state.sequence", minimum=0)
        _require_bool(self.terminal, "state.terminal")
        if self.state_kind != "source_observed":
            raise NBAGameStateError(
                "state.state_kind must be source_observed"
            )
        if self.observation_mode != NBA_OBSERVATION_MODE:
            raise NBAGameStateError(
                "state.observation_mode must be synthetic_fixture while O-005 is blocked"
            )
        _require_text(self.home_team, "state.home_team")
        _require_text(self.away_team, "state.away_team")
        if self.home_team == self.away_team:
            raise NBAGameStateError("home_team and away_team must differ")
        if self.possession_team not in {None, self.home_team, self.away_team}:
            raise NBAGameStateError(
                "state.possession_team must be a game team or None"
            )
        _validate_observed_clock(
            period=self.period,
            clock_ms=self.clock_ms,
            shot_clock_ms=self.shot_clock_ms,
            live_ball=self.live_ball,
            possession_team=self.possession_team,
            context="state",
        )
        for field_name in (
            "home_score",
            "away_score",
            "home_team_fouls",
            "away_team_fouls",
        ):
            _require_int(getattr(self, field_name), f"state.{field_name}", minimum=0)
        for field_name in (
            "home_timeouts_remaining",
            "away_timeouts_remaining",
        ):
            _require_int(
                getattr(self, field_name),
                f"state.{field_name}",
                minimum=0,
                maximum=7,
            )
        _require_bool(self.home_in_bonus, "state.home_in_bonus")
        _require_bool(self.away_in_bonus, "state.away_in_bonus")
        if self.last_event_id is not None and _EVENT_ID_RE.fullmatch(
            self.last_event_id
        ) is None:
            raise NBAGameStateError(
                "state.last_event_id must be an external EventEnvelope ID"
            )
        if self.terminal:
            if self.period < 4 or self.clock_ms != 0:
                raise NBAGameStateError(
                    "terminal NBA state requires period >= 4 and an expired clock"
                )
            if self.home_score == self.away_score:
                raise NBAGameStateError("terminal NBA state cannot be tied")
            if self.live_ball or self.shot_clock_ms is not None:
                raise NBAGameStateError(
                    "terminal NBA state cannot retain a live ball or shot clock"
                )
            if self.possession_team is not None:
                raise NBAGameStateError(
                    "terminal NBA state cannot retain possession"
                )
            if self.terminal_reason is None:
                raise NBAGameStateError(
                    "terminal NBA state requires terminal_reason"
                )
        elif self.terminal_reason is not None:
            raise NBAGameStateError(
                "non-terminal NBA state cannot contain terminal_reason"
            )
        if self.terminal_reason is not None:
            _require_text(self.terminal_reason, "state.terminal_reason")


@dataclass(frozen=True, slots=True)
class NBAGameEvent:
    """One newly observed synthetic X-06 event.

    ``event_id`` must already have been assigned by ``EventEnvelopeV0``.  The
    event contains post-event clock/ball observations, not a future possession
    or final-game label.
    """

    sport: str
    game_id: str
    sequence: int
    event_id: str
    observation_mode: str
    kind: NBAEventKind
    period: int
    clock_ms: int
    shot_clock_ms: int | None
    live_ball: bool
    home_in_bonus_after: bool
    away_in_bonus_after: bool
    team_id: str | None
    points: int | None
    possession_team_after: str | None
    terminal_reason: str | None

    def __post_init__(self) -> None:
        if self.sport != NBA_SPORT:
            raise NBAGameStateError("event.sport must be nba")
        if (
            type(self.game_id) is not str
            or _GAME_ID_RE.fullmatch(self.game_id) is None
        ):
            raise NBAGameStateError("event.game_id must be canonical game_<id>")
        _require_int(self.sequence, "event.sequence", minimum=1)
        if (
            type(self.event_id) is not str
            or _EVENT_ID_RE.fullmatch(self.event_id) is None
        ):
            raise NBAGameStateError(
                "event.event_id must be an external EventEnvelope evt_<sha256>"
            )
        if self.observation_mode != NBA_OBSERVATION_MODE:
            raise NBAGameStateError(
                "event.observation_mode must be synthetic_fixture while O-005 is blocked"
            )
        if self.kind not in _EVENT_KINDS:
            raise NBAGameStateError("event.kind is not supported")
        if self.team_id is not None:
            _require_text(self.team_id, "event.team_id")
        if self.possession_team_after is not None:
            _require_text(
                self.possession_team_after,
                "event.possession_team_after",
            )
        _validate_observed_clock(
            period=self.period,
            clock_ms=self.clock_ms,
            shot_clock_ms=self.shot_clock_ms,
            live_ball=self.live_ball,
            possession_team=self.possession_team_after,
            context="event",
        )
        _require_bool(
            self.home_in_bonus_after,
            "event.home_in_bonus_after",
        )
        _require_bool(
            self.away_in_bonus_after,
            "event.away_in_bonus_after",
        )

        if self.kind == "score":
            if self.team_id is None:
                raise NBAGameStateError("score event requires team_id")
            _require_int(self.points, "score event points", minimum=1, maximum=3)
        elif self.points is not None:
            raise NBAGameStateError("only a score event may contain points")

        if self.kind in {"foul", "timeout"} and self.team_id is None:
            raise NBAGameStateError(f"{self.kind} event requires team_id")
        if self.kind in {"possession", "period"} and (
            self.possession_team_after is None
        ):
            raise NBAGameStateError(
                f"{self.kind} event requires possession_team_after"
            )
        if self.kind == "terminal":
            if self.terminal_reason is None:
                raise NBAGameStateError(
                    "terminal event requires terminal_reason"
                )
            if self.live_ball or self.shot_clock_ms is not None:
                raise NBAGameStateError(
                    "terminal event cannot contain a live ball or shot clock"
                )
            if self.possession_team_after is not None:
                raise NBAGameStateError(
                    "terminal event cannot retain possession"
                )
        elif self.terminal_reason is not None:
            raise NBAGameStateError(
                "only a terminal event may contain terminal_reason"
            )
        if self.terminal_reason is not None:
            _require_text(self.terminal_reason, "event.terminal_reason")


def _validate_transition_identity(
    state: NBAGameState,
    event: NBAGameEvent,
) -> None:
    if state.terminal:
        raise NBAGameStateError("terminal state cannot consume another event")
    if state.game_id != event.game_id:
        raise NBAGameStateError("state and event game_id must match")
    if event.sequence != state.sequence + 1:
        raise NBAGameStateError(
            "event sequence must be exactly state sequence + 1"
        )
    if state.observation_mode != event.observation_mode:
        raise NBAGameStateError("state and event observation mode must match")
    for field_name in ("team_id", "possession_team_after"):
        value = getattr(event, field_name)
        if value is not None and value not in {state.home_team, state.away_team}:
            raise NBAGameStateError(f"event.{field_name} is not a game team")


def reduce_nba_game_state(
    state: NBAGameState,
    event: NBAGameEvent,
) -> NBAGameState:
    """Apply exactly one observed NBA transition."""

    _validate_transition_identity(state, event)

    if event.kind == "period":
        if event.period != state.period + 1:
            raise NBAGameStateError(
                "period event must advance exactly one period"
            )
    else:
        if event.period != state.period:
            raise NBAGameStateError(
                "only a period event may change period"
            )
        if event.clock_ms > state.clock_ms:
            raise NBAGameStateError(
                "same-period event clock cannot move backwards"
            )

    home_score = state.home_score
    away_score = state.away_score
    home_fouls = state.home_team_fouls
    away_fouls = state.away_team_fouls
    home_timeouts = state.home_timeouts_remaining
    away_timeouts = state.away_timeouts_remaining

    if event.kind == "score":
        assert event.points is not None
        if event.team_id == state.home_team:
            home_score += event.points
        else:
            away_score += event.points
    elif event.kind == "foul":
        if event.team_id == state.home_team:
            home_fouls += 1
        else:
            away_fouls += 1
    elif event.kind == "timeout":
        if event.team_id == state.home_team:
            if home_timeouts == 0:
                raise NBAGameStateError(
                    "home timeout event cannot consume a missing timeout"
                )
            home_timeouts -= 1
        else:
            if away_timeouts == 0:
                raise NBAGameStateError(
                    "away timeout event cannot consume a missing timeout"
                )
            away_timeouts -= 1
    elif event.kind == "period":
        home_fouls = 0
        away_fouls = 0

    if event.kind == "period":
        if event.home_in_bonus_after or event.away_in_bonus_after:
            raise NBAGameStateError(
                "period event must reset both observed bonus flags"
            )
    else:
        if state.home_in_bonus and not event.home_in_bonus_after:
            raise NBAGameStateError(
                "home bonus cannot clear within a period"
            )
        if state.away_in_bonus and not event.away_in_bonus_after:
            raise NBAGameStateError(
                "away bonus cannot clear within a period"
            )
        if event.kind != "foul" and (
            event.home_in_bonus_after != state.home_in_bonus
            or event.away_in_bonus_after != state.away_in_bonus
        ):
            raise NBAGameStateError(
                "only a foul event may change an observed bonus flag"
            )
        if event.kind == "foul":
            if (
                event.team_id == state.home_team
                and event.home_in_bonus_after != state.home_in_bonus
            ):
                raise NBAGameStateError(
                    "home-team foul cannot change home bonus"
                )
            if (
                event.team_id == state.away_team
                and event.away_in_bonus_after != state.away_in_bonus
            ):
                raise NBAGameStateError(
                    "away-team foul cannot change away bonus"
                )

    terminal = event.kind == "terminal"
    if terminal:
        if event.period < 4 or event.clock_ms != 0:
            raise NBAGameStateError(
                "terminal event requires period >= 4 and an expired clock"
            )
        if home_score == away_score:
            raise NBAGameStateError(
                "terminal event cannot end an NBA game tied"
            )

    return NBAGameState(
        sport=NBA_SPORT,
        game_id=state.game_id,
        sequence=event.sequence,
        terminal=terminal,
        state_kind="source_observed",
        observation_mode=state.observation_mode,
        home_team=state.home_team,
        away_team=state.away_team,
        period=event.period,
        clock_ms=event.clock_ms,
        home_score=home_score,
        away_score=away_score,
        possession_team=event.possession_team_after,
        home_team_fouls=home_fouls,
        away_team_fouls=away_fouls,
        home_timeouts_remaining=home_timeouts,
        away_timeouts_remaining=away_timeouts,
        home_in_bonus=event.home_in_bonus_after,
        away_in_bonus=event.away_in_bonus_after,
        shot_clock_ms=event.shot_clock_ms,
        live_ball=event.live_ball,
        last_event_id=event.event_id,
        terminal_reason=event.terminal_reason,
    )


reduce = reduce_nba_game_state


@dataclass(frozen=True, slots=True)
class NBAGameStateReducer:
    sport: str = NBA_SPORT
    reducer_id: str = NBA_REDUCER_ID
    reducer_version: str = NBA_REDUCER_VERSION

    def reduce(
        self,
        state: NBAGameState,
        event: NBAGameEvent,
    ) -> NBAGameState:
        return reduce_nba_game_state(state, event)


NBA_GAME_STATE_REDUCER = NBAGameStateReducer()


def _nearest_rank(sorted_samples: list[int], fraction: float) -> int:
    return sorted_samples[max(0, math.ceil(fraction * len(sorted_samples)) - 1)]


def benchmark_nba_reducer_only(
    state: NBAGameState,
    event: NBAGameEvent,
    *,
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, object]:
    """Measure only ``reduce(state, event)``; no model inference is present."""

    _require_int(
        warmup_iterations,
        "warmup_iterations",
        minimum=1,
    )
    _require_int(
        timed_iterations,
        "timed_iterations",
        minimum=1_000,
    )
    for _ in range(warmup_iterations):
        NBA_GAME_STATE_REDUCER.reduce(state, event)

    samples: list[int] = []
    for _ in range(timed_iterations):
        started = time.perf_counter_ns()
        NBA_GAME_STATE_REDUCER.reduce(state, event)
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise NBAGameStateError("perf_counter_ns moved backwards")
        samples.append(elapsed)

    ordered = sorted(samples)
    return {
        "measurement": "reducer_only",
        "timer": "time.perf_counter_ns",
        "unit": "nanoseconds",
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "p50_ns": _nearest_rank(ordered, 0.50),
        "p95_ns": _nearest_rank(ordered, 0.95),
        "p99_ns": _nearest_rank(ordered, 0.99),
        "max_ns": ordered[-1],
        "mean_ns": sum(ordered) // len(ordered),
        "includes_model_inference": False,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


__all__ = [
    "NBA_GAME_STATE_REDUCER",
    "NBA_OBSERVATION_MODE",
    "NBA_REDUCER_ID",
    "NBA_REDUCER_VERSION",
    "NBA_SPORT",
    "NBAGameEvent",
    "NBAGameState",
    "NBAGameStateError",
    "NBAGameStateReducer",
    "benchmark_nba_reducer_only",
    "reduce",
    "reduce_nba_game_state",
]
