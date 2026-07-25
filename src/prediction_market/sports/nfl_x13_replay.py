"""Deterministic, game-generic NFL event reconstruction for X-13."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Literal

from prediction_market.contracts import canonical_json_bytes, canonical_sha256

PlayFamily = Literal[
    "run",
    "pass",
    "sack",
    "punt",
    "kickoff",
    "field_goal",
    "kneel",
    "spike",
    "try",
]
SourceTimeStatus = Literal["point", "interval", "missing"]
ParticipationStatus = Literal["present_complete", "present_partial", "missing"]
ReviewResult = Literal["reversed", "affirmed", "unknown", "none"]
PenaltyStatus = Literal[
    "offsetting",
    "declined",
    "accepted_no_play",
    "accepted",
    "unknown",
    "none",
]
EpisodeType = Literal[
    "run",
    "pass",
    "sack",
    "punt",
    "kickoff",
    "field_goal",
    "kneel",
    "spike",
    "try",
    "touchdown",
    "score_turnover",
    "safety",
    "administrative",
    "other",
]


@dataclass(frozen=True, slots=True)
class GameStateSnapshotV1:
    """Source-observed football state; unknown fields remain ``None``."""

    quarter: int | None
    clock_seconds_remaining: int | None
    game_seconds_remaining: int | None
    home_team: str | None
    away_team: str | None
    home_score: int | None
    away_score: int | None
    possession_team: str | None
    offense_team: str | None
    defense_team: str | None
    drive_id: str | None
    down: int | None
    distance: int | None
    yardline_100: int | None
    goal_to_go: bool | None
    direction: str | None
    home_timeouts_remaining: int | None
    away_timeouts_remaining: int | None
    terminal: bool | None


@dataclass(frozen=True, slots=True)
class CanonicalGameEventV1:
    """One canonically ordered NFL source row without fabricated evidence."""

    game_id: str
    play_id: str
    order_sequence: int
    raw_ordinal: int
    source_time_start_utc: str | None
    source_time_end_utc: str | None
    source_time_status: SourceTimeStatus
    quarter: int | None
    clock_seconds_remaining: int | None
    game_seconds_remaining: int | None
    home_team: str | None
    away_team: str | None
    home_score: int | None
    away_score: int | None
    score_margin: int | None
    possession_team: str | None
    offense_team: str | None
    defense_team: str | None
    drive_id: str | None
    down: int | None
    distance: int | None
    yardline_100: int | None
    goal_to_go: bool | None
    direction: str | None
    description: str | None
    field_coordinate_0_100: int | None
    field_orientation_semantics: str
    play_family: PlayFamily | None
    player_roles: dict[str, str]
    formation: str | None
    offense_personnel: str | None
    defense_personnel: str | None
    offense_names: tuple[str, ...]
    defense_names: tuple[str, ...]
    participation_status: ParticipationStatus
    between_play_roster_change: dict[str, object] | None
    timeout: bool
    timeout_team: str | None
    home_timeouts_remaining: int | None
    away_timeouts_remaining: int | None
    penalty: bool
    penalty_status: PenaltyStatus
    penalty_team: str | None
    penalty_yards: int | None
    penalty_type: str | None
    touchdown: bool
    pass_touchdown: bool
    rush_touchdown: bool
    return_touchdown: bool
    safety: bool
    blocked_kick: bool
    turnover: bool
    passing_yards: int | None
    rushing_yards: int | None
    receiving_yards: int | None
    reception: bool
    review: bool
    review_result: ReviewResult
    reviewed_play_id: str | None
    game_end: bool
    no_play: bool
    deleted: bool
    pre_state: GameStateSnapshotV1
    post_state: GameStateSnapshotV1 | None


@dataclass(frozen=True, slots=True)
class FinalizedEpisodeV1:
    """A deterministic episode composed from one or more canonical rows."""

    episode_id: str
    game_id: str
    play_ids: tuple[str, ...]
    episode_type: EpisodeType
    penalty_status: PenaltyStatus
    nullified: bool
    audit_only: bool
    turnover: bool
    scoring_team: str | None
    score_points: int
    blocked_kick: bool
    review_result: ReviewResult
    pre_state: GameStateSnapshotV1
    post_state: GameStateSnapshotV1 | None


@dataclass(frozen=True, slots=True)
class CumulativeStatLedgerEntryV1:
    """Cumulative box-score state after one canonical source row."""

    game_id: str
    through_play_id: str
    through_order_sequence: int
    through_raw_ordinal: int
    correction_of_play_id: str | None
    player_stats: dict[str, dict[str, int]]
    team_scores: dict[str, int | None]
    period_scores: dict[int, dict[str, int]]


@dataclass(frozen=True, slots=True)
class X13GameLedgerV1:
    """Integrated canonical events, episodes, and cumulative stat ledger."""

    events: tuple[CanonicalGameEventV1, ...]
    episodes: tuple[FinalizedEpisodeV1, ...]
    stat_ledger: tuple[CumulativeStatLedgerEntryV1, ...]
    events_sha256: str
    artifact_sha256: str


def _records(value: object) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        converted = value.to_dict(orient="records")
        if type(converted) is not list:
            raise TypeError("DataFrame-like payload must yield record dictionaries")
        value = converted
    if isinstance(value, Mapping):
        nested = value.get("rows")
        if nested is None:
            raise TypeError("mapping payload must contain rows")
        value = nested
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("payload must be a record sequence or DataFrame")
    records: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("every payload row must be a mapping")
        records.append(dict(row))
    return records


def _stable_scalar(value: object, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a stable scalar")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and float(value).is_integer():
        return str(int(value))
    if type(value) is str and value and value == value.strip():
        return value
    raise ValueError(f"{field} must be a stable scalar")


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise TypeError(f"{field} must be an integer")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _optional_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return None
    return _integer(value, field=field)


def _is_missing(value: object) -> bool:
    return value is None or (
        isinstance(value, Real) and not math.isfinite(float(value))
    )


def _optional_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    if type(value) is str and value and value == value.strip():
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and float(value).is_integer():
        return str(int(value))
    raise ValueError("optional source text must be a stable scalar")


def _indicator(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if type(value) is bool:
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return default
        if number in {0.0, 1.0}:
            return bool(int(number))
    raise ValueError("source indicator must be binary")


def _names(value: object) -> tuple[str, ...]:
    if _is_missing(value):
        return ()
    if type(value) is str:
        return tuple(name for name in value.split(";") if name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(name) for name in value if name is not None)
    raise ValueError("participation names must be a sequence or semicolon text")


def _participation_field(
    row: Mapping[str, object],
    field: str,
) -> object:
    nested = row.get("participation")
    if isinstance(nested, Mapping) and field in nested:
        return nested[field]
    return row.get(field)


def _play_family(row: Mapping[str, object]) -> PlayFamily | None:
    if _indicator(row.get("qb_kneel")):
        return "kneel"
    if _indicator(row.get("qb_spike")):
        return "spike"
    if _indicator(row.get("sack")):
        return "sack"
    if _indicator(row.get("two_point_attempt")) or _indicator(
        row.get("extra_point_attempt")
    ):
        return "try"
    if _indicator(row.get("field_goal_attempt")):
        return "field_goal"
    if _indicator(row.get("punt_attempt")):
        return "punt"
    if _indicator(row.get("kickoff_attempt")):
        return "kickoff"
    if _indicator(row.get("rush_attempt")) or row.get("play_type") == "run":
        return "run"
    if _indicator(row.get("pass_attempt")) or row.get("play_type") == "pass":
        return "pass"
    return None


def _source_time(
    row: Mapping[str, object],
) -> tuple[str | None, str | None, SourceTimeStatus]:
    start = row.get("source_time_start_utc", row.get("time_of_day"))
    end = row.get("source_time_end_utc")
    if _is_missing(start) and _is_missing(end):
        return None, None, "missing"
    if type(start) is not str or not start:
        raise ValueError("source time start must be nonempty text")
    if _is_missing(end):
        return start, start, "point"
    if type(end) is not str or not end:
        raise ValueError("source time end must be nonempty text")
    return start, end, "point" if start == end else "interval"


def _teams_and_scores(
    row: Mapping[str, object],
    *,
    after: bool,
) -> tuple[str | None, str | None, int | None, int | None]:
    home_team = _optional_text(row.get("home_team"))
    away_team = _optional_text(row.get("away_team"))
    posteam = _optional_text(row.get("posteam"))
    suffix = "_post" if after else ""
    offense_score = _optional_integer(
        row.get(f"posteam_score{suffix}"), field=f"posteam_score{suffix}"
    )
    defense_score = _optional_integer(
        row.get(f"defteam_score{suffix}"), field=f"defteam_score{suffix}"
    )
    if (
        home_team is not None
        and away_team is not None
        and posteam in {home_team, away_team}
        and offense_score is not None
        and defense_score is not None
    ):
        if posteam == home_team:
            return home_team, away_team, offense_score, defense_score
        return home_team, away_team, defense_score, offense_score
    home_field = "total_home_score_post" if after else "total_home_score"
    away_field = "total_away_score_post" if after else "total_away_score"
    home_value = row.get(home_field)
    away_value = row.get(away_field)
    if not after:
        home_value = row.get("home_score", home_value)
        away_value = row.get("away_score", away_value)
    return (
        home_team,
        away_team,
        _optional_integer(home_value, field=home_field),
        _optional_integer(away_value, field=away_field),
    )


def _state_snapshot(
    source: Mapping[str, object],
    *,
    after: bool,
) -> GameStateSnapshotV1 | None:
    nested_key = "post_state" if after else "pre_state"
    nested = source.get(nested_key)
    if isinstance(nested, Mapping):
        row = nested
    elif after:
        has_post_state = any(
            not _is_missing(source.get(field))
            for field in (
                "posteam_score_post",
                "defteam_score_post",
                "total_home_score_post",
                "total_away_score_post",
            )
        )
        if not has_post_state:
            return None
        home_team, away_team, home_score, away_score = _teams_and_scores(
            source, after=True
        )
        return GameStateSnapshotV1(
            quarter=None,
            clock_seconds_remaining=None,
            game_seconds_remaining=None,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            possession_team=None,
            offense_team=None,
            defense_team=None,
            drive_id=None,
            down=None,
            distance=None,
            yardline_100=None,
            goal_to_go=None,
            direction=None,
            home_timeouts_remaining=None,
            away_timeouts_remaining=None,
            terminal=_indicator(source.get("game_end"), default=False)
            if source.get("game_end") is not None
            else None,
        )
    else:
        row = source

    home_team, away_team, home_score, away_score = _teams_and_scores(row, after=False)
    offense_value = row.get("posteam")
    if offense_value is None:
        offense_value = row.get("offense_team")
    if offense_value is None:
        offense_value = row.get("possession_team")
    offense_team = _optional_text(offense_value)
    possession_value = row.get("possession_team")
    possession_team = _optional_text(
        offense_team if possession_value is None else possession_value
    )
    defense_value = row.get("defteam")
    if defense_value is None:
        defense_value = row.get("defense_team")
    defense_team = _optional_text(defense_value)
    if (
        defense_team is None
        and offense_team is not None
        and home_team is not None
        and away_team is not None
    ):
        defense_team = away_team if offense_team == home_team else home_team
    goal_value = row.get("goal_to_go")
    terminal_value = row.get("terminal", row.get("game_end"))
    return GameStateSnapshotV1(
        quarter=_optional_integer(
            row.get("qtr", row.get("quarter", row.get("period"))),
            field="qtr",
        ),
        clock_seconds_remaining=_optional_integer(
            row.get(
                "quarter_seconds_remaining",
                row.get(
                    "clock_seconds_remaining",
                    row.get("period_seconds_remaining"),
                ),
            ),
            field="quarter_seconds_remaining",
        ),
        game_seconds_remaining=_optional_integer(
            row.get("game_seconds_remaining"),
            field="game_seconds_remaining",
        ),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        possession_team=possession_team,
        offense_team=offense_team,
        defense_team=defense_team,
        drive_id=_optional_text(row.get("fixed_drive", row.get("drive_id"))),
        down=_optional_integer(row.get("down"), field="down"),
        distance=_optional_integer(
            row.get("ydstogo", row.get("distance")), field="distance"
        ),
        yardline_100=_optional_integer(row.get("yardline_100"), field="yardline_100"),
        goal_to_go=None if _is_missing(goal_value) else _indicator(goal_value),
        direction=_optional_text(row.get("play_direction", row.get("direction"))),
        home_timeouts_remaining=_optional_integer(
            row.get("home_timeouts_remaining"),
            field="home_timeouts_remaining",
        ),
        away_timeouts_remaining=_optional_integer(
            row.get("away_timeouts_remaining"),
            field="away_timeouts_remaining",
        ),
        terminal=None if _is_missing(terminal_value) else _indicator(terminal_value),
    )


def _player_roles(row: Mapping[str, object]) -> dict[str, str]:
    role_fields = (
        ("passer", "passer_player_name"),
        ("rusher", "rusher_player_name"),
        ("receiver", "receiver_player_name"),
        ("kicker", "kicker_player_name"),
        ("punter", "punter_player_name"),
        ("punt_returner", "punt_returner_player_name"),
        ("kickoff_returner", "kickoff_returner_player_name"),
        ("interceptor", "interception_player_name"),
        ("touchdown_scorer", "td_player_name"),
        ("safety_scorer", "safety_player_name"),
        ("blocked_player", "blocked_player_name"),
    )
    return {
        role: text
        for role, field in role_fields
        if (text := _optional_text(row.get(field))) is not None
    }


def _field_coordinate(
    row: Mapping[str, object],
    pre_state: GameStateSnapshotV1,
) -> tuple[int | None, str]:
    coordinate = _optional_integer(
        row.get("field_coordinate_0_100"),
        field="field_coordinate_0_100",
    )
    if coordinate is not None and not 0 <= coordinate <= 100:
        raise ValueError("field_coordinate_0_100 must be between 0 and 100")
    semantics = _optional_text(row.get("field_orientation_semantics"))
    if semantics is None:
        if coordinate is not None:
            semantics = "canonical_team_end_orientation_not_tracking_coordinate"
        elif pre_state.possession_team is None or pre_state.yardline_100 is None:
            semantics = "unavailable_without_possession_and_yardline"
        else:
            semantics = "unavailable_without_source_field_coordinate"
    return coordinate, semantics


def _penalty_status(row: Mapping[str, object], *, no_play: bool) -> PenaltyStatus:
    if not _indicator(row.get("penalty")):
        return "none"
    explicit = _optional_text(row.get("penalty_status"))
    description = (_optional_text(row.get("desc")) or "").lower()
    if explicit == "offsetting" or "offsetting" in description:
        return "offsetting"
    if explicit == "declined" or "declined" in description:
        return "declined"
    if explicit == "accepted_no_play" or no_play:
        return "accepted_no_play"
    if explicit == "accepted" or "accepted" in description:
        return "accepted"
    return "unknown"


def _turnover(row: Mapping[str, object]) -> bool:
    return (
        any(
            _indicator(row.get(field))
            for field in (
                "turnover",
                "interception",
                "fumble_lost",
                "fourth_down_failed",
            )
        )
        or _optional_text(row.get("series_result")) == "Turnover on downs"
    )


def _blocked_kick(row: Mapping[str, object]) -> bool:
    result = (_optional_text(row.get("field_goal_result")) or "").lower()
    return (
        _indicator(row.get("punt_blocked"))
        or result == "blocked"
        or _optional_text(row.get("blocked_player_name")) is not None
    )


def _review(
    row: Mapping[str, object],
) -> tuple[bool, ReviewResult]:
    observed = _indicator(row.get("replay_or_challenge")) or _indicator(
        row.get("review")
    )
    raw_result = _optional_text(
        row.get("replay_or_challenge_result", row.get("review_result"))
    )
    if not observed and raw_result is None:
        return False, "none"
    result = (raw_result or "").lower()
    if "reversed" in result:
        return True, "reversed"
    if result in {"affirmed", "upheld"}:
        return True, "affirmed"
    return True, "unknown"


def _roster_change(
    previous: tuple[tuple[str, ...], tuple[str, ...]] | None,
    current: tuple[tuple[str, ...], tuple[str, ...]],
) -> dict[str, object] | None:
    if previous is None:
        return None
    previous_offense, previous_defense = previous
    current_offense, current_defense = current
    if set(previous_offense) == set(current_offense) and set(previous_defense) == set(
        current_defense
    ):
        return None
    return {
        "semantics": (
            "set difference between consecutive observed participation rows; "
            "timing is bounded between plays"
        ),
        "offense_entering_names": tuple(
            sorted(set(current_offense) - set(previous_offense))
        ),
        "offense_leaving_names": tuple(
            sorted(set(previous_offense) - set(current_offense))
        ),
        "defense_entering_names": tuple(
            sorted(set(current_defense) - set(previous_defense))
        ),
        "defense_leaving_names": tuple(
            sorted(set(previous_defense) - set(current_defense))
        ),
    }


def canonicalize_game_events(
    raw_rows: object,
    participation_rows: object | None = None,
) -> tuple[CanonicalGameEventV1, ...]:
    """Normalize fixture records and sort by the X-13 raw-order contract."""

    rows = _records(raw_rows)
    participation_index: dict[tuple[str, str], dict[str, Any]] = {}
    if participation_rows is not None:
        for row in _records(participation_rows):
            game_id = _stable_scalar(
                row.get("nflverse_game_id", row.get("game_id")),
                field="participation game_id",
            )
            play_id = _stable_scalar(row.get("play_id"), field="participation play_id")
            key = (game_id, play_id)
            if key in participation_index:
                raise ValueError("participation key must be unique")
            participation_index[key] = row

    indexed: list[tuple[tuple[str, int, int], dict[str, Any]]] = []
    raw_keys: set[tuple[str, int, int]] = set()
    for input_ordinal, row in enumerate(rows):
        game_id = _stable_scalar(row.get("game_id"), field="game_id")
        order = _integer(row.get("order_sequence"), field="order_sequence")
        raw_ordinal = _integer(
            row.get("_raw_record_ordinal", row.get("raw_ordinal", input_ordinal)),
            field="raw_ordinal",
        )
        key = (game_id, order, raw_ordinal)
        if key in raw_keys:
            raise ValueError("raw order key must be unique")
        raw_keys.add(key)
        indexed.append((key, row))

    events: list[CanonicalGameEventV1] = []
    previous_rosters: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for (game_id, order, raw_ordinal), row in sorted(indexed, key=lambda item: item[0]):
        play_id = _stable_scalar(row.get("play_id"), field="play_id")
        participation = participation_index.get(
            (game_id, play_id),
            row,
        )
        offense_names = _names(_participation_field(participation, "offense_names"))
        defense_names = _names(_participation_field(participation, "defense_names"))
        participation_status: ParticipationStatus
        if not offense_names and not defense_names:
            participation_status = "missing"
        elif len(offense_names) == len(defense_names) == 11:
            participation_status = "present_complete"
        else:
            participation_status = "present_partial"
        start, end, time_status = _source_time(row)
        pre_state = _state_snapshot(row, after=False)
        assert pre_state is not None
        post_state = _state_snapshot(row, after=True)
        score_margin = None
        if (
            pre_state.offense_team is not None
            and pre_state.home_score is not None
            and pre_state.away_score is not None
            and pre_state.home_team is not None
        ):
            score_margin = (
                pre_state.home_score - pre_state.away_score
                if pre_state.offense_team == pre_state.home_team
                else pre_state.away_score - pre_state.home_score
            )
        roster = (offense_names, defense_names)
        roster_change = (
            None
            if participation_status == "missing"
            else _roster_change(previous_rosters.get(game_id), roster)
        )
        if participation_status != "missing":
            previous_rosters[game_id] = roster
        no_play = row.get("play_type") == "no_play" or _indicator(
            row.get("no_play"), default=False
        )
        review, review_result = _review(row)
        field_coordinate, field_semantics = _field_coordinate(row, pre_state)
        events.append(
            CanonicalGameEventV1(
                game_id=game_id,
                play_id=play_id,
                order_sequence=order,
                raw_ordinal=raw_ordinal,
                source_time_start_utc=start,
                source_time_end_utc=end,
                source_time_status=time_status,
                quarter=pre_state.quarter,
                clock_seconds_remaining=pre_state.clock_seconds_remaining,
                game_seconds_remaining=pre_state.game_seconds_remaining,
                home_team=pre_state.home_team,
                away_team=pre_state.away_team,
                home_score=pre_state.home_score,
                away_score=pre_state.away_score,
                score_margin=score_margin,
                possession_team=pre_state.possession_team,
                offense_team=pre_state.offense_team,
                defense_team=pre_state.defense_team,
                drive_id=pre_state.drive_id,
                down=pre_state.down,
                distance=pre_state.distance,
                yardline_100=pre_state.yardline_100,
                goal_to_go=pre_state.goal_to_go,
                direction=pre_state.direction,
                description=_optional_text(row.get("desc", row.get("description"))),
                field_coordinate_0_100=field_coordinate,
                field_orientation_semantics=field_semantics,
                play_family=_play_family(row),
                player_roles=_player_roles(row),
                formation=_optional_text(
                    _participation_field(participation, "offense_formation")
                ),
                offense_personnel=_optional_text(
                    _participation_field(participation, "offense_personnel")
                ),
                defense_personnel=_optional_text(
                    _participation_field(participation, "defense_personnel")
                ),
                offense_names=offense_names,
                defense_names=defense_names,
                participation_status=participation_status,
                between_play_roster_change=roster_change,
                timeout=_indicator(row.get("timeout")),
                timeout_team=_optional_text(row.get("timeout_team")),
                home_timeouts_remaining=pre_state.home_timeouts_remaining,
                away_timeouts_remaining=pre_state.away_timeouts_remaining,
                penalty=_indicator(row.get("penalty")),
                penalty_status=_penalty_status(row, no_play=no_play),
                penalty_team=_optional_text(row.get("penalty_team")),
                penalty_yards=_optional_integer(
                    row.get("penalty_yards"), field="penalty_yards"
                ),
                penalty_type=_optional_text(row.get("penalty_type")),
                touchdown=_indicator(row.get("touchdown")),
                pass_touchdown=_indicator(row.get("pass_touchdown")),
                rush_touchdown=_indicator(row.get("rush_touchdown")),
                return_touchdown=_indicator(row.get("return_touchdown")),
                safety=_indicator(row.get("safety")),
                blocked_kick=_blocked_kick(row),
                turnover=_turnover(row),
                passing_yards=_optional_integer(
                    row.get("passing_yards"), field="passing_yards"
                ),
                rushing_yards=_optional_integer(
                    row.get("rushing_yards"), field="rushing_yards"
                ),
                receiving_yards=_optional_integer(
                    row.get("receiving_yards"), field="receiving_yards"
                ),
                reception=_indicator(row.get("complete_pass")),
                review=review,
                review_result=review_result,
                reviewed_play_id=_optional_text(
                    row.get("reviewed_play_id", row.get("reversed_play_id"))
                ),
                game_end=_indicator(row.get("game_end")),
                no_play=no_play,
                deleted=_indicator(
                    row.get("play_deleted", row.get("deleted")), default=False
                ),
                pre_state=pre_state,
                post_state=post_state,
            )
        )
    return tuple(events)


_PENALTY_PRIORITY: dict[PenaltyStatus, int] = {
    "none": -1,
    "unknown": 0,
    "accepted": 1,
    "accepted_no_play": 2,
    "declined": 3,
    "offsetting": 4,
}


def _finalize_episode(
    events: Sequence[CanonicalGameEventV1],
) -> FinalizedEpisodeV1:
    first = events[0]
    last_observed_post_state = next(
        (
            event.post_state
            for event in reversed(events)
            if event.post_state is not None
        ),
        None,
    )
    valid_events = tuple(
        event for event in events if not event.deleted and not event.no_play
    )
    turnover = any(event.turnover for event in valid_events)
    if any(event.review_result == "reversed" for event in events):
        review_result: ReviewResult = "reversed"
    elif any(event.review_result == "affirmed" for event in events):
        review_result = "affirmed"
    elif any(event.review_result == "unknown" for event in events):
        review_result = "unknown"
    else:
        review_result = "none"
    episode_post_state = (
        events[-1].post_state
        if review_result == "reversed"
        else last_observed_post_state
    )
    team_points: dict[str, int] = {}
    for event in valid_events:
        if event.post_state is None:
            continue
        for team, before, after in (
            (
                event.home_team,
                event.pre_state.home_score,
                event.post_state.home_score,
            ),
            (
                event.away_team,
                event.pre_state.away_score,
                event.post_state.away_score,
            ),
        ):
            if (
                team is not None
                and before is not None
                and after is not None
                and after > before
            ):
                team_points[team] = team_points.get(team, 0) + after - before
    score_points = sum(team_points.values())
    scoring_team = next(iter(team_points)) if len(team_points) == 1 else None
    if review_result == "reversed":
        score_points = 0
        scoring_team = None
    if score_points and turnover:
        episode_type: EpisodeType = "score_turnover"
    elif any(event.safety for event in valid_events):
        episode_type = "safety"
    elif any(event.touchdown for event in events):
        episode_type: EpisodeType = "touchdown"
    elif valid_events and valid_events[0].play_family is not None:
        episode_type = valid_events[0].play_family
    elif all(event.deleted for event in events):
        episode_type = "administrative"
    else:
        episode_type = "administrative"
    penalty_status = max(
        (event.penalty_status for event in events),
        key=_PENALTY_PRIORITY.__getitem__,
    )
    audit_only = all(event.deleted for event in events)
    material = {
        "schema_version": "FinalizedEpisodeV1",
        "game_id": first.game_id,
        "source_keys": [
            [event.play_id, event.order_sequence, event.raw_ordinal] for event in events
        ],
    }
    return FinalizedEpisodeV1(
        episode_id="episode_" + canonical_sha256(material).removeprefix("sha256:"),
        game_id=first.game_id,
        play_ids=tuple(event.play_id for event in events),
        episode_type=episode_type,
        penalty_status=penalty_status,
        nullified=not audit_only
        and (
            review_result == "reversed"
            or (
                any(not event.deleted for event in events)
                and all(event.no_play or event.deleted for event in events)
            )
        ),
        audit_only=audit_only,
        turnover=turnover,
        scoring_team=scoring_team,
        score_points=score_points,
        blocked_kick=any(event.blocked_kick for event in valid_events),
        review_result=review_result,
        pre_state=first.pre_state,
        post_state=episode_post_state,
    )


def build_finalized_episodes(
    events: Sequence[CanonicalGameEventV1],
) -> tuple[FinalizedEpisodeV1, ...]:
    """Apply the fixed X-13 touchdown-bundle boundary rule."""

    ordered = tuple(
        sorted(
            events,
            key=lambda event: (
                event.game_id,
                event.order_sequence,
                event.raw_ordinal,
            ),
        )
    )
    grouped: list[list[CanonicalGameEventV1]] = []
    touchdown_group: list[CanonicalGameEventV1] | None = None
    for event in ordered:
        if touchdown_group is not None:
            if (
                event.game_id != touchdown_group[0].game_id
                or event.deleted
                or event.game_end
                or event.play_family == "kickoff"
            ):
                grouped.append(touchdown_group)
                touchdown_group = None
            elif event.play_family == "try" or event.play_family is None:
                touchdown_group.append(event)
                continue
            else:
                grouped.append(touchdown_group)
                touchdown_group = None
        if event.touchdown and not event.no_play and not event.deleted:
            touchdown_group = [event]
        elif (
            event.review and not event.deleted and event.play_family is None and grouped
        ):
            target_index: int | None = None
            for index in range(len(grouped) - 1, -1, -1):
                candidate = grouped[index]
                if candidate[0].game_id != event.game_id:
                    break
                if any(item.deleted or item.game_end for item in candidate):
                    break
                if any(
                    item.play_family is not None
                    and not item.no_play
                    and not item.deleted
                    for item in candidate
                ):
                    target_index = index
                    break
            if target_index is None:
                grouped.append([event])
            else:
                merged = [
                    item for candidate in grouped[target_index:] for item in candidate
                ]
                merged.append(event)
                grouped[target_index:] = [merged]
        else:
            grouped.append([event])
    if touchdown_group is not None:
        grouped.append(touchdown_group)
    return tuple(_finalize_episode(group) for group in grouped)


def _empty_player_line() -> dict[str, int]:
    return {
        "passing_yards": 0,
        "passing_touchdowns": 0,
        "rushing_yards": 0,
        "rushing_touchdowns": 0,
        "receiving_yards": 0,
        "receptions": 0,
        "receiving_touchdowns": 0,
        "return_touchdowns": 0,
        "touchdowns": 0,
    }


def _player_line(
    player_stats: dict[str, dict[str, int]],
    player: str,
) -> dict[str, int]:
    return player_stats.setdefault(player, _empty_player_line())


def _apply_player_stats(
    event: CanonicalGameEventV1,
    player_stats: dict[str, dict[str, int]],
) -> None:
    passer = event.player_roles.get("passer")
    receiver = event.player_roles.get("receiver")
    rusher = event.player_roles.get("rusher")
    if passer is not None:
        line = _player_line(player_stats, passer)
        line["passing_yards"] += event.passing_yards or 0
        if event.pass_touchdown:
            line["passing_touchdowns"] += 1
    if receiver is not None:
        line = _player_line(player_stats, receiver)
        line["receiving_yards"] += event.receiving_yards or 0
        line["receptions"] += int(event.reception)
        if event.pass_touchdown:
            line["receiving_touchdowns"] += 1
            line["touchdowns"] += 1
    if rusher is not None:
        line = _player_line(player_stats, rusher)
        line["rushing_yards"] += event.rushing_yards or 0
        if event.rush_touchdown:
            line["rushing_touchdowns"] += 1
            line["touchdowns"] += 1
    if event.return_touchdown:
        scorer = event.player_roles.get("touchdown_scorer")
        if scorer is not None:
            line = _player_line(player_stats, scorer)
            line["return_touchdowns"] += 1
            line["touchdowns"] += 1


@dataclass(frozen=True, slots=True)
class _AppliedStatContribution:
    player_stats: dict[str, dict[str, int]]
    team_scores_before: dict[str, int | None]
    period: int | None
    period_points: dict[str, int]


def _merge_player_stats(
    cumulative: dict[str, dict[str, int]],
    delta: Mapping[str, Mapping[str, int]],
    *,
    sign: int,
) -> None:
    for player, changes in delta.items():
        line = _player_line(cumulative, player)
        for statistic, value in changes.items():
            line[statistic] += sign * value


def build_cumulative_stat_ledger(
    events: Sequence[CanonicalGameEventV1],
) -> tuple[CumulativeStatLedgerEntryV1, ...]:
    """Build deterministic cumulative player and team score snapshots."""

    ordered = tuple(
        sorted(
            events,
            key=lambda event: (
                event.game_id,
                event.order_sequence,
                event.raw_ordinal,
            ),
        )
    )
    entries: list[CumulativeStatLedgerEntryV1] = []
    current_game: str | None = None
    player_stats: dict[str, dict[str, int]] = {}
    team_scores: dict[str, int | None] = {}
    period_scores: dict[int, dict[str, int]] = {}
    applied_by_play: dict[str, _AppliedStatContribution] = {}
    last_live_play_id: str | None = None
    for event in ordered:
        if event.game_id != current_game:
            current_game = event.game_id
            player_stats = {}
            team_scores = {
                team: score
                for team, score in sorted(
                    (
                        pair
                        for pair in (
                            (event.home_team, event.pre_state.home_score),
                            (event.away_team, event.pre_state.away_score),
                        )
                        if pair[0] is not None
                    ),
                    key=lambda pair: pair[0],
                )
            }
            period_scores = {}
            applied_by_play = {}
            last_live_play_id = None
        teams = tuple(sorted(team_scores))
        if event.quarter is not None:
            period_scores.setdefault(
                event.quarter,
                {team: 0 for team in teams},
            )
        correction_of_play_id: str | None = None
        if event.review_result == "reversed" and not event.deleted:
            target = event.reviewed_play_id or last_live_play_id
            if target is not None:
                correction_of_play_id = target
                contribution = applied_by_play.pop(target, None)
                if contribution is not None:
                    _merge_player_stats(
                        player_stats,
                        contribution.player_stats,
                        sign=-1,
                    )
                    for team, score in contribution.team_scores_before.items():
                        team_scores[team] = score
                    if contribution.period is not None:
                        period_line = period_scores[contribution.period]
                        for team, points in contribution.period_points.items():
                            period_line[team] -= points
            if event.post_state is not None:
                for team, score in (
                    (event.home_team, event.post_state.home_score),
                    (event.away_team, event.post_state.away_score),
                ):
                    if team is not None and score is not None:
                        team_scores[team] = score
        if (
            not event.no_play
            and not event.deleted
            and event.review_result != "reversed"
        ):
            player_delta: dict[str, dict[str, int]] = {}
            _apply_player_stats(event, player_delta)
            _merge_player_stats(player_stats, player_delta, sign=1)
            team_scores_before: dict[str, int | None] = {}
            period_points: dict[str, int] = {}
            if event.post_state is not None:
                for team, before, after in (
                    (
                        event.home_team,
                        event.pre_state.home_score,
                        event.post_state.home_score,
                    ),
                    (
                        event.away_team,
                        event.pre_state.away_score,
                        event.post_state.away_score,
                    ),
                ):
                    if team is None:
                        continue
                    team_scores_before[team] = before
                    if after is not None:
                        team_scores[team] = after
                    if (
                        event.quarter is not None
                        and before is not None
                        and after is not None
                        and after > before
                    ):
                        points = after - before
                        period_scores[event.quarter][team] += points
                        period_points[team] = points
            if event.play_family is not None:
                last_live_play_id = event.play_id
                applied_by_play[event.play_id] = _AppliedStatContribution(
                    player_stats=player_delta,
                    team_scores_before=team_scores_before,
                    period=event.quarter,
                    period_points=period_points,
                )
        entries.append(
            CumulativeStatLedgerEntryV1(
                game_id=event.game_id,
                through_play_id=event.play_id,
                through_order_sequence=event.order_sequence,
                through_raw_ordinal=event.raw_ordinal,
                correction_of_play_id=correction_of_play_id,
                player_stats={
                    player: dict(line) for player, line in sorted(player_stats.items())
                },
                team_scores=dict(sorted(team_scores.items())),
                period_scores={
                    period: dict(sorted(scores.items()))
                    for period, scores in sorted(period_scores.items())
                },
            )
        )
    return tuple(entries)


def _payload_value(
    payload: object,
    field: str,
    default: object = None,
) -> object:
    if isinstance(payload, Mapping):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _native_game_id(value: object) -> str:
    game_id = _stable_scalar(value, field="game_id")
    return game_id.removeprefix("game_nflverse_")


def _flatten_replay_payload(payload: object) -> list[dict[str, Any]]:
    steps_value = _payload_value(payload, "steps")
    if steps_value is None:
        return _records(payload)
    if not isinstance(steps_value, Sequence):
        raise TypeError("replay steps must be a sequence")
    native_game_id = _native_game_id(
        _payload_value(
            payload,
            "native_game_id",
            _payload_value(payload, "canonical_game_id"),
        )
    )
    home_team = _optional_text(_payload_value(payload, "home_team"))
    away_team = _optional_text(_payload_value(payload, "away_team"))
    rows: list[dict[str, Any]] = []
    for ordinal, untrusted_step in enumerate(steps_value):
        if not isinstance(untrusted_step, Mapping):
            raise TypeError("every replay step must be a mapping")
        step = dict(untrusted_step)
        rows.append(
            {
                "game_id": native_game_id,
                "play_id": step.get("source_play_id", step.get("play_id")),
                "order_sequence": step.get(
                    "source_order_sequence",
                    step.get("order_sequence"),
                ),
                "_raw_record_ordinal": step.get(
                    "raw_ordinal",
                    step.get("_raw_record_ordinal", ordinal),
                ),
                "time_of_day": step.get(
                    "source_time_utc",
                    step.get("time_of_day"),
                ),
                "home_team": home_team,
                "away_team": away_team,
                "post_state": step.get("next_state"),
            }
        )
    return rows


def _flatten_participation_payload(payload: object) -> list[dict[str, Any]]:
    outer_game_id = _payload_value(payload, "native_game_id")
    if outer_game_id is None:
        outer_game_id = _payload_value(payload, "canonical_game_id")
    rows = _records(payload)
    flattened: list[dict[str, Any]] = []
    for row in rows:
        state = row.get("state")
        event = row.get("event")
        if not isinstance(state, Mapping) and not isinstance(event, Mapping):
            result = dict(row)
        else:
            result = dict(state) if isinstance(state, Mapping) else {}
            if isinstance(event, Mapping):
                flags = event.get("flags")
                actors = event.get("actors")
                if isinstance(flags, Mapping):
                    result.update(flags)
                if isinstance(actors, Mapping):
                    result.update(actors)
                for source_field, target_field in (
                    ("play_type", "play_type"),
                    ("play_type_nfl", "play_type_nfl"),
                    ("description", "desc"),
                    ("timeout_team", "timeout_team"),
                ):
                    if source_field in event:
                        result[target_field] = event[source_field]
            result.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"state", "event"}
                }
            )
        if result.get("game_id") is None:
            candidate = result.get("nflverse_game_id", outer_game_id)
            if candidate is not None:
                result["game_id"] = _native_game_id(candidate)
        if "source_time_utc" in result and "time_of_day" not in result:
            result["time_of_day"] = result["source_time_utc"]
        flattened.append(result)
    return flattened


def _integrated_rows(
    replay_payload: object,
    participation_payload: object | None,
) -> list[dict[str, Any]]:
    replay_rows = _flatten_replay_payload(replay_payload)
    if participation_payload is None:
        return replay_rows
    participation_rows = _flatten_participation_payload(participation_payload)
    replay_game_ids = {
        _native_game_id(row.get("game_id"))
        for row in replay_rows
        if row.get("game_id") is not None
    }
    if len(replay_game_ids) == 1:
        only_game_id = next(iter(replay_game_ids))
        for row in participation_rows:
            row.setdefault("game_id", only_game_id)
    replay_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in replay_rows:
        key = (
            _native_game_id(row.get("game_id")),
            _stable_scalar(row.get("play_id"), field="play_id"),
        )
        replay_by_key[key] = row
    merged: list[dict[str, Any]] = []
    for participation in participation_rows:
        key = (
            _native_game_id(participation.get("game_id")),
            _stable_scalar(participation.get("play_id"), field="play_id"),
        )
        replay = replay_by_key.pop(key, {})
        post_state = replay.get("post_state")
        row = dict(replay)
        row.update(participation)
        row["game_id"] = key[0]
        if post_state is not None:
            row["post_state"] = post_state
        merged.append(row)
    merged.extend(replay_by_key.values())
    return merged


def build_x13_game_ledger(
    replay_payload: object,
    participation_payload: object | None = None,
) -> X13GameLedgerV1:
    """Integrate existing replay/personnel payloads into the pure X-13 ledger."""

    events = canonicalize_game_events(
        _integrated_rows(replay_payload, participation_payload)
    )
    episodes = build_finalized_episodes(events)
    stat_ledger = build_cumulative_stat_ledger(events)
    events_hash = canonical_events_sha256(events)
    ledger_material: list[dict[str, Any]] = []
    for entry in stat_ledger:
        encoded = asdict(entry)
        encoded["period_scores"] = {
            str(period): scores for period, scores in entry.period_scores.items()
        }
        ledger_material.append(encoded)
    material = {
        "schema_version": "X13GameLedgerV1",
        "events": [asdict(event) for event in events],
        "episodes": [asdict(episode) for episode in episodes],
        "stat_ledger": ledger_material,
    }
    return X13GameLedgerV1(
        events=events,
        episodes=episodes,
        stat_ledger=stat_ledger,
        events_sha256=events_hash,
        artifact_sha256=canonical_sha256(material),
    )


def canonical_events_bytes(events: Sequence[CanonicalGameEventV1]) -> bytes:
    """Return stable canonical bytes for an already normalized event sequence."""

    material = {
        "schema_version": "CanonicalGameEventV1",
        "events": [asdict(event) for event in events],
    }
    return canonical_json_bytes(material)


def canonical_events_sha256(
    events: Sequence[CanonicalGameEventV1],
) -> str:
    """Return the canonical event sequence hash."""

    material = {
        "schema_version": "CanonicalGameEventV1",
        "events": [asdict(event) for event in events],
    }
    return canonical_sha256(material)


__all__ = [
    "CanonicalGameEventV1",
    "CumulativeStatLedgerEntryV1",
    "FinalizedEpisodeV1",
    "GameStateSnapshotV1",
    "X13GameLedgerV1",
    "build_cumulative_stat_ledger",
    "build_finalized_episodes",
    "build_x13_game_ledger",
    "canonical_events_bytes",
    "canonical_events_sha256",
    "canonicalize_game_events",
]
