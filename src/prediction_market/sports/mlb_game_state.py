"""Immutable, deterministic MLB game-state primitives."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal


MLB_SPORT = "mlb"
CHADWICK_CWEVENT_VERSION = "0.10.0"
_EVENT_ID_RE = re.compile(r"evt_[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"game_[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_RETROSHEET_GAME_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_CWEVENT_VERSION_RE = re.compile(
    r"Chadwick expanded event descriptor, version ([0-9]+\.[0-9]+\.[0-9]+)"
)
CWEVENT_FIELD_MAP: Mapping[str, tuple[str, int]] = MappingProxyType(
    {
        "game_id": ("GAME_ID", 0),
        "inning": ("INN_CT", 2),
        "batting_home": ("BAT_HOME_ID", 3),
        "outs_before": ("OUTS_CT", 4),
        "balls_before": ("BALLS_CT", 5),
        "strikes_before": ("STRIKES_CT", 6),
        "away_score_before": ("AWAY_SCORE_CT", 8),
        "home_score_before": ("HOME_SCORE_CT", 9),
        "batter_id": ("BAT_ID", 10),
        "pitcher_id": ("PIT_ID", 14),
        "runner_on_first": ("BASE1_RUN_ID", 26),
        "runner_on_second": ("BASE2_RUN_ID", 27),
        "runner_on_third": ("BASE3_RUN_ID", 28),
        "event_text": ("EVENT_TX", 29),
        "lineup_slot": ("BAT_LINEUP_ID", 33),
        "event_code": ("EVENT_CD", 34),
        "batter_event": ("BAT_EVENT_FL", 35),
        "outs_on_play": ("EVENT_OUTS_CT", 40),
        "batter_destination": ("BAT_DEST_ID", 58),
        "runner_on_first_destination": ("RUN1_DEST_ID", 59),
        "runner_on_second_destination": ("RUN2_DEST_ID", 60),
        "runner_on_third_destination": ("RUN3_DEST_ID", 61),
        "end_game": ("GAME_END_FL", 79),
        "event_index": ("EVENT_ID", 96),
    }
)
HalfInning = Literal["top", "bottom"]
BaseState = tuple[str | None, str | None, str | None]


class MLBGameStateError(ValueError):
    """A parsed MLB observation cannot produce a trustworthy game state."""


def _require_text(value: object, field_name: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise MLBGameStateError(f"{field_name} must be a canonical nonempty string")


def _require_int(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise MLBGameStateError(f"{field_name} is outside its legal range")


def _validate_bases(value: object, field_name: str) -> BaseState:
    if type(value) is not tuple or len(value) != 3:
        raise MLBGameStateError(f"{field_name} bases must be a three-item tuple")
    for runner in value:
        if runner is not None:
            _require_text(runner, f"{field_name} runner")
    runners = tuple(runner for runner in value if runner is not None)
    if len(set(runners)) != len(runners):
        raise MLBGameStateError(f"{field_name} runner IDs must be unique")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MLBScore:
    away: int
    home: int

    def __post_init__(self) -> None:
        _require_int(self.away, "away score", minimum=0)
        _require_int(self.home, "home score", minimum=0)


@dataclass(frozen=True, slots=True)
class RunnerAdvance:
    runner_id: str
    start_base: int
    destination: int

    def __post_init__(self) -> None:
        _require_text(self.runner_id, "runner_id")
        if type(self.start_base) is not int or not 0 <= self.start_base <= 3:
            raise MLBGameStateError("runner start base must be in 0..3")
        if type(self.destination) is not int or not 0 <= self.destination <= 4:
            raise MLBGameStateError("runner destination must be in 0..4")
        if self.destination not in {0, 4} and self.destination < self.start_base:
            raise MLBGameStateError("runner destination cannot move backward")


@dataclass(frozen=True, slots=True)
class InningTransition:
    inning: int
    half: HalfInning
    batting_team: str
    fielding_team: str
    batter_id: str
    pitcher_id: str
    lineup_slot: int
    balls: int
    strikes: int

    def __post_init__(self) -> None:
        _require_int(self.inning, "transition inning", minimum=1)
        if self.half not in {"top", "bottom"}:
            raise MLBGameStateError("transition half must be top or bottom")
        _require_text(self.batting_team, "transition batting_team")
        _require_text(self.fielding_team, "transition fielding_team")
        if self.batting_team == self.fielding_team:
            raise MLBGameStateError("transition teams must differ")
        _require_text(self.batter_id, "transition batter_id")
        _require_text(self.pitcher_id, "transition pitcher_id")
        _require_int(self.lineup_slot, "transition lineup slot", minimum=1, maximum=9)
        _require_int(self.balls, "transition balls", minimum=0, maximum=3)
        _require_int(self.strikes, "transition strikes", minimum=0, maximum=2)


@dataclass(frozen=True, slots=True)
class MLBPlayEvent:
    sport: str
    game_id: str
    sequence: int
    event_id: str
    inning: int
    half: HalfInning
    outs_before: int
    bases_before: BaseState
    score_before: MLBScore
    batting_team: str
    fielding_team: str
    batter_id: str
    pitcher_id: str
    balls_before: int
    strikes_before: int
    lineup_slot_before: int
    play_type: str
    runs: tuple[str, ...]
    outs: tuple[str, ...]
    runner_destinations: tuple[RunnerAdvance, ...]
    next_batter_id: str | None
    next_pitcher_id: str | None
    next_balls: int | None
    next_strikes: int | None
    next_lineup_slot: int | None
    inning_transition: InningTransition | None = None
    terminal: bool = False
    source_parser: str = "parsed_observation"
    source_parser_version: str = "v1"
    source_event_index: int | None = None

    def __post_init__(self) -> None:
        if self.sport != MLB_SPORT:
            raise MLBGameStateError("event sport must be mlb")
        if type(self.game_id) is not str or _GAME_ID_RE.fullmatch(self.game_id) is None:
            raise MLBGameStateError("event game_id must be canonical")
        _require_int(self.sequence, "event sequence", minimum=1)
        if type(self.event_id) is not str or _EVENT_ID_RE.fullmatch(self.event_id) is None:
            raise MLBGameStateError("event_id must be an external evt_<lowercase sha256>")
        _require_int(self.inning, "event inning", minimum=1)
        if self.half not in {"top", "bottom"}:
            raise MLBGameStateError("event half must be top or bottom")
        _require_int(self.outs_before, "event outs", minimum=0, maximum=2)
        _validate_bases(self.bases_before, "event")
        if not isinstance(self.score_before, MLBScore):
            raise MLBGameStateError("event score must be MLBScore")
        for value, name in (
            (self.batting_team, "event batting_team"),
            (self.fielding_team, "event fielding_team"),
            (self.batter_id, "event batter_id"),
            (self.pitcher_id, "event pitcher_id"),
            (self.play_type, "event play_type"),
        ):
            _require_text(value, name)
        _require_int(self.balls_before, "event balls", minimum=0, maximum=3)
        _require_int(self.strikes_before, "event strikes", minimum=0, maximum=2)
        _require_int(
            self.lineup_slot_before,
            "event lineup slot",
            minimum=1,
            maximum=9,
        )
        if type(self.runs) is not tuple or type(self.outs) is not tuple:
            raise MLBGameStateError("event runs and outs must be tuples")
        for runner in (*self.runs, *self.outs):
            _require_text(runner, "event run/out runner")
        if type(self.runner_destinations) is not tuple or any(
            not isinstance(advance, RunnerAdvance)
            for advance in self.runner_destinations
        ):
            raise MLBGameStateError(
                "runner_destinations must be a tuple of RunnerAdvance"
            )
        if type(self.terminal) is not bool:
            raise MLBGameStateError("event terminal must be boolean")
        _require_text(self.source_parser, "source_parser")
        _require_text(self.source_parser_version, "source_parser_version")
        if self.source_event_index is not None:
            _require_int(
                self.source_event_index,
                "source_event_index",
                minimum=1,
            )
        next_values = (
            self.next_batter_id,
            self.next_pitcher_id,
            self.next_balls,
            self.next_strikes,
            self.next_lineup_slot,
        )
        if self.terminal or self.inning_transition is not None:
            if any(value is not None for value in next_values):
                raise MLBGameStateError(
                    "terminal/transition event cannot contain next plate appearance"
                )
        else:
            if any(value is None for value in next_values):
                raise MLBGameStateError(
                    "nonterminal event requires the next plate appearance"
                )
            _require_text(self.next_batter_id, "next_batter_id")
            _require_text(self.next_pitcher_id, "next_pitcher_id")
            _require_int(self.next_balls, "next balls", minimum=0, maximum=3)
            _require_int(self.next_strikes, "next strikes", minimum=0, maximum=2)
            _require_int(
                self.next_lineup_slot,
                "next lineup slot",
                minimum=1,
                maximum=9,
            )
        advances_by_runner = {
            advance.runner_id: advance for advance in self.runner_destinations
        }
        if len(advances_by_runner) != len(self.runner_destinations):
            raise MLBGameStateError("runner destinations must have unique runner IDs")
        destination_runs = {
            advance.runner_id
            for advance in self.runner_destinations
            if advance.destination == 4
        }
        if set(self.runs) != destination_runs or len(set(self.runs)) != len(self.runs):
            raise MLBGameStateError("explicit runs must match home destinations")
        destination_outs = {
            advance.runner_id
            for advance in self.runner_destinations
            if advance.destination == 0
        }
        if set(self.outs) != destination_outs or len(set(self.outs)) != len(self.outs):
            raise MLBGameStateError("explicit outs must match out destinations")


@dataclass(frozen=True, slots=True)
class MLBGameState:
    sport: str
    game_id: str
    sequence: int
    inning: int
    half: HalfInning
    outs: int
    bases: BaseState
    score: MLBScore
    away_team: str
    home_team: str
    batting_team: str
    fielding_team: str
    batter_id: str
    pitcher_id: str
    balls: int
    strikes: int
    lineup_slot: int
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.sport != MLB_SPORT:
            raise MLBGameStateError("state sport must be mlb")
        if type(self.game_id) is not str or _GAME_ID_RE.fullmatch(self.game_id) is None:
            raise MLBGameStateError("state game_id must be canonical")
        _require_int(self.sequence, "state sequence", minimum=0)
        _require_int(self.inning, "state inning", minimum=1)
        if self.half not in {"top", "bottom"}:
            raise MLBGameStateError("state half must be top or bottom")
        if type(self.terminal) is not bool:
            raise MLBGameStateError("state terminal must be boolean")
        _require_int(
            self.outs,
            "state outs",
            minimum=0,
            maximum=3 if self.terminal else 2,
        )
        _validate_bases(self.bases, "state")
        if not isinstance(self.score, MLBScore):
            raise MLBGameStateError("state score must be MLBScore")
        for value, name in (
            (self.away_team, "away_team"),
            (self.home_team, "home_team"),
            (self.batting_team, "batting_team"),
            (self.fielding_team, "fielding_team"),
            (self.batter_id, "batter_id"),
            (self.pitcher_id, "pitcher_id"),
        ):
            _require_text(value, name)
        if self.away_team == self.home_team:
            raise MLBGameStateError("away and home teams must differ")
        expected_batting = self.away_team if self.half == "top" else self.home_team
        expected_fielding = self.home_team if self.half == "top" else self.away_team
        if self.batting_team != expected_batting:
            raise MLBGameStateError("batting team is inconsistent with half inning")
        if self.fielding_team != expected_fielding:
            raise MLBGameStateError("fielding team is inconsistent with half inning")
        if not self.terminal and self.batter_id in self.bases:
            raise MLBGameStateError("current batter cannot also be a base runner")
        _require_int(self.balls, "state balls", minimum=0, maximum=3)
        _require_int(self.strikes, "state strikes", minimum=0, maximum=2)
        _require_int(self.lineup_slot, "state lineup slot", minimum=1, maximum=9)


def initial_state(
    *,
    game_id: str,
    away_team: str,
    home_team: str,
    batter_id: str,
    pitcher_id: str,
    lineup_slot: int = 1,
) -> MLBGameState:
    """Create the state immediately before the first plate appearance."""

    return MLBGameState(
        sport=MLB_SPORT,
        game_id=game_id,
        sequence=0,
        inning=1,
        half="top",
        outs=0,
        bases=(None, None, None),
        score=MLBScore(away=0, home=0),
        away_team=away_team,
        home_team=home_team,
        batting_team=away_team,
        fielding_team=home_team,
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        balls=0,
        strikes=0,
        lineup_slot=lineup_slot,
        terminal=False,
    )


def retrosheet_game_id(native_game_id: str) -> str:
    """Return the common canonical ID for one Retrosheet game."""

    _require_text(native_game_id, "Retrosheet game_id")
    if native_game_id.startswith("game_"):
        if _GAME_ID_RE.fullmatch(native_game_id) is None:
            raise MLBGameStateError("Retrosheet game_id must be canonical")
        return native_game_id
    if _RETROSHEET_GAME_ID_RE.fullmatch(native_game_id) is None:
        raise MLBGameStateError("Retrosheet game_id contains unsupported characters")
    return f"game_retrosheet_{native_game_id}"


def require_cwevent_version(executable: str = "cwevent") -> str:
    """Require the exact Chadwick version used by the declared field map."""

    _require_text(executable, "cwevent executable")
    resolved = shutil.which(executable)
    if resolved is None:
        raise MLBGameStateError("Chadwick cwevent executable is not installed")
    try:
        completed = subprocess.run(
            [resolved],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MLBGameStateError("Chadwick cwevent version cannot be inspected") from exc
    banner = completed.stdout + completed.stderr
    match = _CWEVENT_VERSION_RE.search(banner)
    if match is None:
        raise MLBGameStateError("Chadwick cwevent version banner is missing")
    version = match.group(1)
    if version != CHADWICK_CWEVENT_VERSION:
        raise MLBGameStateError(
            "Chadwick cwevent version does not match the declared field map"
        )
    return version


def _cwevent_value(row: Mapping[str, object], logical_name: str) -> object:
    if not isinstance(row, Mapping):
        raise MLBGameStateError("cwevent observation must be a mapping")
    field_name = CWEVENT_FIELD_MAP[logical_name][0]
    if field_name not in row:
        raise MLBGameStateError(f"cwevent observation lacks {field_name}")
    return row[field_name]


def _cwevent_text(
    row: Mapping[str, object],
    logical_name: str,
    *,
    optional: bool = False,
) -> str | None:
    value = _cwevent_value(row, logical_name)
    if optional and value == "":
        return None
    _require_text(value, f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]}")
    assert type(value) is str
    return value


def _cwevent_int(
    row: Mapping[str, object],
    logical_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = _cwevent_value(row, logical_name)
    if type(value) is int:
        parsed = value
    elif type(value) is str and re.fullmatch(r"-?[0-9]+", value):
        parsed = int(value)
    else:
        raise MLBGameStateError(
            f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]} must be an integer"
        )
    _require_int(
        parsed,
        f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]}",
        minimum=minimum,
        maximum=maximum,
    )
    return parsed


def _cwevent_flag(row: Mapping[str, object], logical_name: str) -> bool:
    value = _cwevent_value(row, logical_name)
    if type(value) is bool:
        return value
    if type(value) is int and value in {0, 1}:
        return bool(value)
    if type(value) is str and value in {"T", "F"}:
        return value == "T"
    raise MLBGameStateError(
        f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]} must be T or F"
    )


def _cwevent_snapshot(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
) -> dict[str, object]:
    native_game_id = _cwevent_text(row, "game_id")
    assert native_game_id is not None
    batting_home = _cwevent_int(
        row,
        "batting_home",
        minimum=0,
        maximum=1,
    )
    half: HalfInning = "bottom" if batting_home else "top"
    return {
        "game_id": retrosheet_game_id(native_game_id),
        "inning": _cwevent_int(row, "inning", minimum=1),
        "half": half,
        "outs": _cwevent_int(row, "outs_before", minimum=0, maximum=2),
        "bases": (
            _cwevent_text(row, "runner_on_first", optional=True),
            _cwevent_text(row, "runner_on_second", optional=True),
            _cwevent_text(row, "runner_on_third", optional=True),
        ),
        "score": MLBScore(
            away=_cwevent_int(row, "away_score_before", minimum=0),
            home=_cwevent_int(row, "home_score_before", minimum=0),
        ),
        "batting_team": home_team if batting_home else away_team,
        "fielding_team": away_team if batting_home else home_team,
        "batter_id": _cwevent_text(row, "batter_id"),
        "pitcher_id": _cwevent_text(row, "pitcher_id"),
        "balls": _cwevent_int(row, "balls_before", minimum=0, maximum=3),
        "strikes": _cwevent_int(row, "strikes_before", minimum=0, maximum=2),
        "lineup_slot": _cwevent_int(
            row,
            "lineup_slot",
            minimum=1,
            maximum=9,
        ),
    }


def _require_snapshot_matches_state(
    snapshot: Mapping[str, object],
    state: MLBGameState,
) -> None:
    for field_name in (
        "game_id",
        "inning",
        "half",
        "outs",
        "bases",
        "score",
        "batting_team",
        "fielding_team",
        "batter_id",
        "pitcher_id",
        "balls",
        "strikes",
        "lineup_slot",
    ):
        if snapshot[field_name] != getattr(state, field_name):
            raise MLBGameStateError(
                f"cwevent {field_name} does not match the supplied state"
            )


def state_from_cwevent_row(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
    cwevent_version: str,
    sequence: int = 0,
) -> MLBGameState:
    """Build the exact state observed immediately before one cwevent row."""

    if cwevent_version != CHADWICK_CWEVENT_VERSION:
        raise MLBGameStateError(
            "Chadwick cwevent version does not match the declared field map"
        )
    _require_int(sequence, "state sequence", minimum=0)
    snapshot = _cwevent_snapshot(
        row,
        away_team=away_team,
        home_team=home_team,
    )
    return MLBGameState(
        sport=MLB_SPORT,
        sequence=sequence,
        away_team=away_team,
        home_team=home_team,
        terminal=False,
        **snapshot,
    )


def _canonical_destination(value: int) -> int:
    if not 0 <= value <= 6:
        raise MLBGameStateError("cwevent runner destination must be in 0..6")
    return 4 if value >= 4 else value


def event_from_cwevent_rows(
    state: MLBGameState,
    play_row: Mapping[str, object],
    next_row: Mapping[str, object] | None,
    *,
    event_id: str,
    cwevent_version: str,
) -> MLBPlayEvent:
    """Adapt explicit cwevent 0.10.0 fields; never parse event-file grammar."""

    if cwevent_version != CHADWICK_CWEVENT_VERSION:
        raise MLBGameStateError(
            "Chadwick cwevent version does not match the declared field map"
        )
    if not isinstance(state, MLBGameState):
        raise MLBGameStateError("cwevent adapter requires MLBGameState")
    pre = _cwevent_snapshot(
        play_row,
        away_team=state.away_team,
        home_team=state.home_team,
    )
    _require_snapshot_matches_state(pre, state)

    advances: list[RunnerAdvance] = []
    if _cwevent_flag(play_row, "batter_event"):
        advances.append(
            RunnerAdvance(
                runner_id=state.batter_id,
                start_base=0,
                destination=_canonical_destination(
                    _cwevent_int(
                        play_row,
                        "batter_destination",
                        minimum=0,
                        maximum=6,
                    )
                ),
            )
        )
    elif _cwevent_int(
        play_row,
        "batter_destination",
        minimum=0,
        maximum=6,
    ) != 0:
        raise MLBGameStateError(
            "cwevent non-batter event cannot advance the batter"
        )

    runner_destination_fields = (
        "runner_on_first_destination",
        "runner_on_second_destination",
        "runner_on_third_destination",
    )
    for start_base, (runner_id, destination_name) in enumerate(
        zip(state.bases, runner_destination_fields, strict=True),
        start=1,
    ):
        destination = _canonical_destination(
            _cwevent_int(
                play_row,
                destination_name,
                minimum=0,
                maximum=6,
            )
        )
        if runner_id is None:
            if destination != 0:
                raise MLBGameStateError(
                    "cwevent cannot advance a runner from an empty base"
                )
            continue
        advances.append(
            RunnerAdvance(
                runner_id=runner_id,
                start_base=start_base,
                destination=destination,
            )
        )

    runs = tuple(
        advance.runner_id for advance in advances if advance.destination == 4
    )
    outs = tuple(
        advance.runner_id for advance in advances if advance.destination == 0
    )
    observed_outs = _cwevent_int(
        play_row,
        "outs_on_play",
        minimum=0,
        maximum=3,
    )
    if len(outs) != observed_outs:
        raise MLBGameStateError(
            "cwevent EVENT_OUTS_CT does not match runner destinations"
        )

    terminal = _cwevent_flag(play_row, "end_game")
    if terminal and next_row is not None:
        raise MLBGameStateError("terminal cwevent row must not have a next row")
    if not terminal and next_row is None:
        raise MLBGameStateError("nonterminal cwevent row requires the next row")
    next_snapshot = (
        None
        if next_row is None
        else _cwevent_snapshot(
            next_row,
            away_team=state.away_team,
            home_team=state.home_team,
        )
    )
    outs_after = state.outs + len(outs)
    transition = (
        InningTransition(
            inning=int(next_snapshot["inning"]),
            half=next_snapshot["half"],  # type: ignore[arg-type]
            batting_team=str(next_snapshot["batting_team"]),
            fielding_team=str(next_snapshot["fielding_team"]),
            batter_id=str(next_snapshot["batter_id"]),
            pitcher_id=str(next_snapshot["pitcher_id"]),
            lineup_slot=int(next_snapshot["lineup_slot"]),
            balls=int(next_snapshot["balls"]),
            strikes=int(next_snapshot["strikes"]),
        )
        if next_snapshot is not None and outs_after == 3
        else None
    )
    event_text = _cwevent_text(play_row, "event_text")
    event_code = _cwevent_int(play_row, "event_code", minimum=0)
    source_event_index = _cwevent_int(
        play_row,
        "event_index",
        minimum=1,
    )
    event = MLBPlayEvent(
        sport=MLB_SPORT,
        game_id=state.game_id,
        sequence=state.sequence + 1,
        event_id=event_id,
        inning=state.inning,
        half=state.half,
        outs_before=state.outs,
        bases_before=state.bases,
        score_before=state.score,
        batting_team=state.batting_team,
        fielding_team=state.fielding_team,
        batter_id=state.batter_id,
        pitcher_id=state.pitcher_id,
        balls_before=state.balls,
        strikes_before=state.strikes,
        lineup_slot_before=state.lineup_slot,
        play_type=f"{event_code}:{event_text}",
        runs=runs,
        outs=outs,
        runner_destinations=tuple(advances),
        next_batter_id=(
            None
            if terminal or transition is not None
            else str(next_snapshot["batter_id"])
        ),
        next_pitcher_id=(
            None
            if terminal or transition is not None
            else str(next_snapshot["pitcher_id"])
        ),
        next_balls=(
            None
            if terminal or transition is not None
            else int(next_snapshot["balls"])
        ),
        next_strikes=(
            None
            if terminal or transition is not None
            else int(next_snapshot["strikes"])
        ),
        next_lineup_slot=(
            None
            if terminal or transition is not None
            else int(next_snapshot["lineup_slot"])
        ),
        inning_transition=transition,
        terminal=terminal,
        source_parser="chadwick.cwevent",
        source_parser_version=cwevent_version,
        source_event_index=source_event_index,
    )
    reduced = reduce_mlb_state(state, event)
    if next_snapshot is not None:
        _require_snapshot_matches_state(next_snapshot, reduced)
    return event


def reduce_mlb_state(state: MLBGameState, event: MLBPlayEvent) -> MLBGameState:
    """Apply one already-parsed play observation without mutating prior state."""

    if not isinstance(state, MLBGameState) or not isinstance(event, MLBPlayEvent):
        raise MLBGameStateError("reducer requires MLBGameState and MLBPlayEvent")
    if state.terminal:
        raise MLBGameStateError("terminal state cannot accept another event")
    if state.sport != MLB_SPORT or event.sport != MLB_SPORT:
        raise MLBGameStateError("sport must be mlb")
    if event.game_id != state.game_id:
        raise MLBGameStateError("event game_id does not match state game_id")
    if event.sequence != state.sequence + 1:
        raise MLBGameStateError("event sequence must be contiguous")
    if event.inning != state.inning:
        raise MLBGameStateError("event inning does not match state inning")
    if event.half != state.half:
        raise MLBGameStateError("event half does not match state half")
    if event.outs_before != state.outs:
        raise MLBGameStateError("event outs_before does not match state outs")
    if event.bases_before != state.bases:
        raise MLBGameStateError("event bases_before does not match state bases")
    if event.score_before != state.score:
        raise MLBGameStateError("event score_before does not match state score")
    if event.batting_team != state.batting_team:
        raise MLBGameStateError("event batting_team does not match state")
    if event.fielding_team != state.fielding_team:
        raise MLBGameStateError("event fielding_team does not match state")
    if event.batter_id != state.batter_id:
        raise MLBGameStateError("event batter_id does not match state batter")
    if event.pitcher_id != state.pitcher_id:
        raise MLBGameStateError("event pitcher_id does not match state pitcher")
    if event.balls_before != state.balls:
        raise MLBGameStateError("event balls_before does not match state balls")
    if event.strikes_before != state.strikes:
        raise MLBGameStateError("event strikes_before does not match state strikes")
    if event.lineup_slot_before != state.lineup_slot:
        raise MLBGameStateError("event lineup slot does not match state lineup")

    start_bases = [advance.start_base for advance in event.runner_destinations]
    if len(set(start_bases)) != len(start_bases):
        raise MLBGameStateError("runner origins must be unique")
    occupied_destinations = [
        advance.destination
        for advance in event.runner_destinations
        if 1 <= advance.destination <= 3
    ]
    if len(set(occupied_destinations)) != len(occupied_destinations):
        raise MLBGameStateError("runner base destinations must be unique")
    moving_bases = {base for base in start_bases if base}
    for advance in event.runner_destinations:
        if advance.start_base == 0:
            if advance.runner_id != state.batter_id:
                raise MLBGameStateError("batter runner origin does not match state")
        elif state.bases[advance.start_base - 1] != advance.runner_id:
            raise MLBGameStateError("base runner origin does not match state")
        if (
            1 <= advance.destination <= 3
            and state.bases[advance.destination - 1] is not None
            and advance.destination not in moving_bases
        ):
            raise MLBGameStateError("runner destination is already occupied")

    outs_after = state.outs + len(event.outs)
    if outs_after > 3:
        raise MLBGameStateError("a play cannot produce more than three outs")
    if event.inning_transition is not None and outs_after != 3:
        raise MLBGameStateError("inning transition requires the third out")
    if outs_after == 3 and event.inning_transition is None and not event.terminal:
        raise MLBGameStateError("third out requires an inning transition or terminal")
    if event.inning_transition is not None and event.terminal:
        raise MLBGameStateError("terminal event cannot also transition innings")
    if event.inning_transition is not None:
        transition = event.inning_transition
        expected_transition = (
            (
                state.inning,
                "bottom",
                state.home_team,
                state.away_team,
            )
            if state.half == "top"
            else (
                state.inning + 1,
                "top",
                state.away_team,
                state.home_team,
            )
        )
        observed_transition = (
            transition.inning,
            transition.half,
            transition.batting_team,
            transition.fielding_team,
        )
        if observed_transition != expected_transition:
            raise MLBGameStateError(
                "inning transition must advance to the immediate next half"
            )

    bases = list(state.bases)
    for advance in event.runner_destinations:
        if advance.start_base:
            bases[advance.start_base - 1] = None
    for advance in event.runner_destinations:
        if 1 <= advance.destination <= 3:
            bases[advance.destination - 1] = advance.runner_id
    score = (
        MLBScore(away=state.score.away + len(event.runs), home=state.score.home)
        if state.half == "top"
        else MLBScore(away=state.score.away, home=state.score.home + len(event.runs))
    )
    if event.terminal:
        if state.inning < 9:
            raise MLBGameStateError("terminal event cannot occur before inning nine")
        if score.away == score.home:
            raise MLBGameStateError("terminal event cannot leave a tied score")
        if state.half == "top" and (
            outs_after != 3 or score.home <= score.away
        ):
            raise MLBGameStateError(
                "top-half terminal event requires third out with home leading"
            )
        if (
            state.half == "bottom"
            and outs_after < 3
            and score.home <= score.away
        ):
            raise MLBGameStateError(
                "bottom-half terminal event before three outs must be a walkoff"
            )
        return replace(
            state,
            sequence=event.sequence,
            outs=state.outs + len(event.outs),
            bases=(bases[0], bases[1], bases[2]),
            score=score,
            terminal=True,
        )
    if event.inning_transition is not None:
        transition = event.inning_transition
        return replace(
            state,
            sequence=event.sequence,
            inning=transition.inning,
            half=transition.half,
            outs=0,
            bases=(None, None, None),
            score=score,
            batting_team=transition.batting_team,
            fielding_team=transition.fielding_team,
            batter_id=transition.batter_id,
            pitcher_id=transition.pitcher_id,
            balls=transition.balls,
            strikes=transition.strikes,
            lineup_slot=transition.lineup_slot,
        )
    return replace(
        state,
        sequence=event.sequence,
        outs=state.outs + len(event.outs),
        bases=(bases[0], bases[1], bases[2]),
        score=score,
        batter_id=event.next_batter_id,
        pitcher_id=event.next_pitcher_id,
        balls=event.next_balls,
        strikes=event.next_strikes,
        lineup_slot=event.next_lineup_slot,
    )


reduce = reduce_mlb_state


@dataclass(frozen=True, slots=True)
class MLBGameStateReducer:
    """Common-protocol adapter for the pure MLB reducer."""

    sport: str = field(default=MLB_SPORT, init=False)
    reducer_id: str = field(
        default="mlb.retrosheet.play-reducer",
        init=False,
    )
    reducer_version: str = field(default="v1", init=False)

    def reduce(
        self,
        state: MLBGameState,
        event: MLBPlayEvent,
    ) -> MLBGameState:
        return reduce_mlb_state(state, event)


MLB_GAME_STATE_REDUCER = MLBGameStateReducer()


__all__ = [
    "CHADWICK_CWEVENT_VERSION",
    "CWEVENT_FIELD_MAP",
    "InningTransition",
    "MLB_GAME_STATE_REDUCER",
    "MLB_SPORT",
    "MLBGameState",
    "MLBGameStateError",
    "MLBGameStateReducer",
    "MLBPlayEvent",
    "MLBScore",
    "RunnerAdvance",
    "event_from_cwevent_rows",
    "initial_state",
    "reduce",
    "reduce_mlb_state",
    "require_cwevent_version",
    "retrosheet_game_id",
    "state_from_cwevent_row",
]
