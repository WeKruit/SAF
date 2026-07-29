"""Deterministic one-game NFL fact extraction for X-16 review.

The extractor starts from byte-frozen nflverse play-by-play and participation
rows.  It describes sports facts only: it does not read market reactions,
train a model, or infer injuries from participation changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = "EpisodeFactV3"
CLAIM_BOUNDARY = "SPORTS_FACT_DESCRIPTION_ONLY_NO_MARKET_REACTION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INJURY_RE = re.compile(
    r"(?P<team>[A-Z]{2,3})-(?P<number>\d+)-(?P<name>[A-Za-z][A-Za-z.'’\- ]*?)"
    r"\s+was injured during the play",
    flags=re.IGNORECASE,
)
_RETURN_RE = re.compile(
    r"Injury Update:\s*(?P<team>[A-Z]{2,3})-(?P<number>\d+)-"
    r"(?P<name>[A-Za-z][A-Za-z.'’\- ]*?)\s+has returned to the game",
    flags=re.IGNORECASE,
)


class NFLFactExtractionError(ValueError):
    """Frozen rows cannot support a deterministic fact description."""


@dataclass(frozen=True, slots=True)
class GameFactTables:
    events: pd.DataFrame
    tags: pd.DataFrame
    players: pd.DataFrame
    availability: pd.DataFrame
    injury_evidence: pd.DataFrame
    factor_hits: pd.DataFrame
    factor_coverage: pd.DataFrame
    reconciliation: pd.DataFrame
    registry_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise NFLFactExtractionError(f"{field} must be a SHA-256 digest")
    return value


def _present(value: object) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    return not bool(missing) if isinstance(missing, bool) else True


def _text(value: object) -> str | None:
    if not _present(value):
        return None
    result = str(value).strip()
    return result or None


def _number(value: object) -> float | None:
    if not _present(value) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    result = _number(value)
    return None if result is None else int(result)


def _indicator(value: object) -> bool:
    result = _number(value)
    if result is not None:
        return bool(result)
    text = _text(value)
    return False if text is None else text.lower() in {"true", "yes", "y"}


def _stable_play_id(value: object) -> str:
    number = _number(value)
    if number is not None and number.is_integer():
        return str(int(number))
    text = _text(value)
    if text is None:
        raise NFLFactExtractionError("play_id is missing")
    return text


def _split(value: object) -> list[str]:
    text = _text(value)
    return [] if text is None else [item.strip() for item in text.split(";") if item.strip()]


def _parallel(values: Sequence[list[str]], field: str) -> list[tuple[str, ...]]:
    lengths = {len(value) for value in values}
    if len(lengths) > 1:
        raise NFLFactExtractionError(f"{field} participation arrays are misaligned")
    return list(zip(*values, strict=True))


def _fixed_field_coordinate(label: object, *, away_team: str, home_team: str) -> float | None:
    text = _text(label)
    if text is None:
        return None
    if text == "50":
        return 50.0
    match = re.fullmatch(r"([A-Z]{2,3})\s+(\d{1,2})", text)
    if match is None:
        return None
    team, yard = match.group(1), int(match.group(2))
    if not 0 <= yard <= 50:
        return None
    if team == away_team:
        return float(yard)
    if team == home_team:
        return float(100 - yard)
    return None


def _pre_scores(row: Mapping[str, object]) -> tuple[int | None, int | None]:
    home = _text(row.get("home_team"))
    posteam = _text(row.get("posteam"))
    posteam_score = _integer(row.get("posteam_score"))
    defteam_score = _integer(row.get("defteam_score"))
    if home is not None and posteam_score is not None and defteam_score is not None:
        return (
            (posteam_score, defteam_score)
            if posteam == home
            else (defteam_score, posteam_score)
        )
    return (
        _integer(row.get("total_home_score")),
        _integer(row.get("total_away_score")),
    )


def _primary_action(row: Mapping[str, object]) -> str:
    native = (_text(row.get("play_type_nfl")) or "").upper()
    play_type = (_text(row.get("play_type")) or "").lower()
    if native in {"GAME_START", "END_QUARTER", "END_GAME", "END_HALF"}:
        return "ADMIN"
    if native == "TIMEOUT":
        return "TIMEOUT"
    if native == "PENALTY" or (play_type == "no_play" and _indicator(row.get("penalty"))):
        return "PENALTY"
    if (
        play_type in {"extra_point", "two_point_conversion"}
        or native in {"XP_KICK", "PAT2"}
        or _text(row.get("two_point_conv_result")) is not None
    ):
        return "TRY"
    if native == "SACK" or _indicator(row.get("sack")):
        return "SACK"
    if play_type == "pass" or native in {"PASS", "INTERCEPTION"}:
        return "PASS"
    if play_type in {"run", "qb_kneel"} or native == "RUSH":
        return "KNEEL" if _indicator(row.get("qb_kneel")) else "RUN"
    if play_type == "qb_spike":
        return "SPIKE"
    if play_type == "punt" or native == "PUNT":
        return "PUNT"
    if play_type == "kickoff" or native == "KICK_OFF":
        return "KICKOFF"
    if play_type == "field_goal" or native == "FIELD_GOAL":
        return "FIELD_GOAL"
    if play_type == "no_play":
        return "ADMIN"
    return "ADMIN"


def _visual_end_coordinate(
    row: Mapping[str, object],
    *,
    action: str,
    tags: Sequence[str],
    pre_coordinate: float | None,
    next_state_coordinate: float | None,
    posteam: str | None,
    home_team: str,
    away_team: str,
) -> tuple[float | None, str]:
    """Return a transparent display coordinate without parsing description text."""

    if pre_coordinate is None:
        return None, "UNAVAILABLE"
    tag_set = set(tags)
    has_enforced_penalty = (
        "PENALTY_ACCEPTED" in tag_set
        and "PENALTY_DECLINED" not in tag_set
    )
    if (
        action in {"PASS", "RUN", "SACK", "KNEEL", "SPIKE"}
        and "TURNOVER" not in tag_set
        and not has_enforced_penalty
    ):
        yards = _number(row.get("yards_gained"))
        if yards is not None and posteam in {home_team, away_team}:
            sign = 1.0 if posteam == away_team else -1.0
            return (
                max(0.0, min(100.0, pre_coordinate + sign * yards)),
                "DERIVED_FROM_STRUCTURED_YARDS_GAINED_ON_TEAM_NORMALIZED_FIELD",
            )
    if next_state_coordinate is not None and (
        "TURNOVER" in tag_set
        or action in {"PUNT", "KICKOFF"}
        or has_enforced_penalty
    ):
        return (
            next_state_coordinate,
            "NEXT_OBSERVED_STRUCTURED_SCRIMMAGE_STATE_NOT_EXACT_BALL_TRAJECTORY",
        )
    return None, "UNAVAILABLE"


def _transition_direction_semantics(
    *,
    action: str,
    tags: Sequence[str],
    posteam: str | None,
    home_team: str,
    away_team: str,
) -> tuple[str | None, str]:
    if action in {"KICKOFF", "PUNT"}:
        return (
            "BIDIRECTIONAL",
            "KICK_AND_POSSIBLE_RETURN_NET_STATE_ONLY",
        )
    if "TURNOVER" in set(tags):
        return (
            "BIDIRECTIONAL",
            "OFFENSE_THEN_POSSIBLE_RETURN_NET_STATE_ONLY",
        )
    if posteam == home_team:
        return "LEFT", "SINGLE_PHASE_TEAM_NORMALIZED"
    if posteam == away_team:
        return "RIGHT", "SINGLE_PHASE_TEAM_NORMALIZED"
    return None, "UNAVAILABLE"


def _penalty_tags(row: Mapping[str, object], description: str) -> set[str]:
    if not _indicator(row.get("penalty")):
        return set()
    tags = {"PENALTY"}
    lower = description.lower()
    if "declined" in lower:
        tags.add("PENALTY_DECLINED")
    if "offsetting" in lower:
        tags.add("PENALTY_OFFSETTING")
    if row.get("play_type") == "no_play" or "no play" in lower:
        tags.add("PENALTY_ACCEPTED_NO_PLAY")
    elif "declined" not in lower or "enforced" in lower:
        tags.add("PENALTY_ACCEPTED")
    return tags


def _outcome_tags(
    row: Mapping[str, object],
    *,
    next_possession: str | None,
) -> tuple[str, ...]:
    tags: set[str] = set()
    description = _text(row.get("desc")) or ""
    play_type = _text(row.get("play_type"))
    action = _primary_action(row)
    no_play = play_type == "no_play"
    if action == "PASS":
        tags.add("PASS_ATTEMPT")
        if _indicator(row.get("complete_pass")):
            tags.add("COMPLETE_PASS")
        if _indicator(row.get("incomplete_pass")):
            tags.add("INCOMPLETE_PASS")
    if action == "RUN":
        tags.add("RUSH_ATTEMPT")
        if _indicator(row.get("qb_scramble")):
            tags.add("QB_SCRAMBLE")
    if action == "SACK":
        tags.add("SACK")
    if action == "KNEEL":
        tags.add("QB_KNEEL")
    if action == "SPIKE":
        tags.add("QB_SPIKE")
    if _indicator(row.get("first_down")) and not no_play:
        tags.add("FIRST_DOWN")
    for field, tag in (
        ("third_down_converted", "THIRD_DOWN_CONVERTED"),
        ("third_down_failed", "THIRD_DOWN_FAILED"),
        ("fourth_down_converted", "FOURTH_DOWN_CONVERTED"),
        ("fourth_down_failed", "FOURTH_DOWN_FAILED"),
    ):
        if _indicator(row.get(field)) and not no_play:
            tags.add(tag)
    if _indicator(row.get("touchdown")) and not no_play:
        tags.update(("SCORE", "TOUCHDOWN"))
    for field, tag in (
        ("pass_touchdown", "PASS_TOUCHDOWN"),
        ("rush_touchdown", "RUSH_TOUCHDOWN"),
        ("return_touchdown", "RETURN_TOUCHDOWN"),
    ):
        if _indicator(row.get(field)) and not no_play:
            tags.add(tag)
    turnover = False
    if _indicator(row.get("interception")) and not no_play:
        tags.add("INTERCEPTION")
        turnover = True
    if _indicator(row.get("fumble")) and not no_play:
        tags.add("FUMBLE")
    if _indicator(row.get("fumble_lost")) and not no_play:
        tags.add("FUMBLE_LOST")
        turnover = True
    if _indicator(row.get("fourth_down_failed")) and not no_play:
        tags.add("TURNOVER_ON_DOWNS")
        turnover = True
    if turnover:
        tags.add("TURNOVER")
    if (
        "TOUCHDOWN" in tags
        and (
            turnover
            or (
                _text(row.get("td_team")) is not None
                and _text(row.get("td_team")) == _text(row.get("defteam"))
            )
        )
    ):
        tags.add("DEFENSIVE_TOUCHDOWN")
    if _indicator(row.get("safety")) and not no_play:
        tags.update(("SAFETY", "SCORE"))
    if _indicator(row.get("punt_blocked")):
        tags.add("BLOCKED_PUNT")
    result = (_text(row.get("field_goal_result")) or "").lower()
    if result:
        tags.add("FIELD_GOAL_" + result.upper())
        if result == "made":
            tags.add("SCORE")
    extra = (_text(row.get("extra_point_result")) or "").lower()
    if extra:
        tags.add("EXTRA_POINT_" + extra.upper())
        if extra == "good":
            tags.add("SCORE")
    two_point = (_text(row.get("two_point_conv_result")) or "").lower()
    if two_point:
        tags.add("TWO_POINT_" + two_point.upper())
        if two_point in {"success", "successful", "good"}:
            tags.add("SCORE")
    for field, tag in (
        ("touchback", "TOUCHBACK"),
        ("punt_inside_twenty", "PUNT_INSIDE_20"),
        ("punt_in_endzone", "PUNT_IN_ENDZONE"),
        ("punt_out_of_bounds", "PUNT_OUT_OF_BOUNDS"),
        ("punt_downed", "PUNT_DOWNED"),
        ("punt_fair_catch", "PUNT_FAIR_CATCH"),
        ("kickoff_inside_twenty", "KICKOFF_INSIDE_20"),
        ("kickoff_in_endzone", "KICKOFF_IN_ENDZONE"),
        ("kickoff_out_of_bounds", "KICKOFF_OUT_OF_BOUNDS"),
        ("kickoff_downed", "KICKOFF_DOWNED"),
        ("kickoff_fair_catch", "KICKOFF_FAIR_CATCH"),
        ("own_kickoff_recovery", "ONSIDE_RECOVERY"),
    ):
        if _indicator(row.get(field)):
            tags.add(tag)
    if action == "PUNT":
        non_return_endings = {
            "TOUCHBACK",
            "PUNT_FAIR_CATCH",
            "PUNT_DOWNED",
            "PUNT_OUT_OF_BOUNDS",
        }
        if (
            _number(row.get("return_yards")) is not None
            and not non_return_endings.intersection(tags)
        ):
            tags.add("PUNT_RETURN")
        if "TOUCHBACK" in tags:
            tags.add("PUNT_TOUCHBACK")
    if action == "KICKOFF":
        if (
            _number(row.get("return_yards")) is not None
            and "TOUCHBACK" not in tags
            and "KICKOFF_FAIR_CATCH" not in tags
        ):
            tags.add("KICKOFF_RETURN")
        if "TOUCHBACK" in tags:
            tags.add("KICKOFF_TOUCHBACK")
    tags.update(_penalty_tags(row, description))
    if no_play:
        tags.add("NO_PLAY")
    if _indicator(row.get("replay_or_challenge")):
        tags.add("REVIEWED")
        review_result = (_text(row.get("replay_or_challenge_result")) or "").lower()
        if review_result:
            tags.add("REVIEW_" + review_result.upper())
            if review_result == "reversed":
                tags.add("REVERSED")
    if _indicator(row.get("timeout")) or action == "TIMEOUT":
        tags.add("TIMEOUT")
    if _INJURY_RE.search(description):
        tags.add("CONFIRMED_IN_GAME_INJURY")
    if _RETURN_RE.search(description):
        tags.add("CONFIRMED_RETURN_TO_GAME")
    posteam = _text(row.get("posteam"))
    if turnover or (
        posteam is not None
        and next_possession is not None
        and next_possession != posteam
        and action not in {"ADMIN", "TIMEOUT", "PENALTY"}
    ):
        tags.add("POSSESSION_CHANGE")
    return tuple(sorted(tags))


def _actor_team(
    row: Mapping[str, object],
    *,
    action: str,
    tags: Sequence[str],
    home_team: str,
    away_team: str,
) -> str | None:
    explicit = _text(row.get("actor_team"))
    if explicit in {home_team, away_team}:
        return explicit
    if "DEFENSIVE_TOUCHDOWN" in tags or "INTERCEPTION" in tags:
        return _text(row.get("defteam"))
    if action == "KICKOFF":
        return _text(row.get("defteam"))
    return _text(row.get("posteam"))


def _beneficiary_team(
    row: Mapping[str, object],
    *,
    tags: Sequence[str],
    final_sports_outcome_eligible: bool,
    home_team: str,
    away_team: str,
) -> tuple[str | None, str]:
    if not final_sports_outcome_eligible:
        return None, "UNRESOLVED"
    tag_set = set(tags)
    if "SCORE" in tag_set:
        td_team = _text(row.get("td_team"))
        if td_team in {home_team, away_team}:
            return td_team, "RESOLVED_FINAL_SPORTS_RULE"
        team = (
            _text(row.get("defteam"))
            if "SAFETY" in tag_set or "DEFENSIVE_TOUCHDOWN" in tag_set
            else _text(row.get("posteam"))
        )
        if team in {home_team, away_team}:
            return team, "RESOLVED_FINAL_SPORTS_RULE"
    if tag_set.intersection({"INTERCEPTION", "FUMBLE_LOST", "TURNOVER_ON_DOWNS"}):
        recovery_team = _text(row.get("fumble_recovery_1_team"))
        team = (
            recovery_team
            if recovery_team in {home_team, away_team}
            else _text(row.get("defteam"))
        )
        if team in {home_team, away_team}:
            return team, "RESOLVED_FINAL_SPORTS_RULE"
    return None, "UNRESOLVED"


def _is_home(team: str | None, *, home_team: str, away_team: str) -> bool | None:
    if team == home_team:
        return True
    if team == away_team:
        return False
    return None


def _canonical_utc(value: object, field: str) -> pd.Timestamp:
    text = _text(value)
    if text is None:
        raise NFLFactExtractionError(f"{field} is missing")
    try:
        timestamp = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise NFLFactExtractionError(f"{field} must be a timestamp") from exc
    if timestamp.tzinfo is None:
        raise NFLFactExtractionError(f"{field} must include a UTC offset")
    return timestamp.tz_convert("UTC")


def _timestamp_text(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _source_interval(
    row: Mapping[str, object],
) -> tuple[str | None, str | None, str, str | None]:
    source_time = _text(row.get("time_of_day"))
    if source_time is None:
        return None, None, "UNAVAILABLE", None
    start = _canonical_utc(source_time, "time_of_day")
    fractional = re.search(r"\.(\d+)(?:Z|[+-]\d{2}:?\d{2})\Z", source_time)
    digits = 0 if fractional is None else len(fractional.group(1))
    if digits == 0:
        resolution, width = "SECOND", pd.Timedelta(seconds=1)
    elif digits <= 3:
        resolution, width = "MILLISECOND", pd.Timedelta(milliseconds=1)
    elif digits <= 6:
        resolution, width = "MICROSECOND", pd.Timedelta(microseconds=1)
    else:
        resolution, width = "NANOSECOND", pd.Timedelta(nanoseconds=1)
    end = start + width
    return (
        _timestamp_text(start),
        _timestamp_text(end),
        resolution,
        _timestamp_text(end),
    )


_EXPLICIT_ROLES = (
    ("PASSER", "passer_player_id", "passer_player_name"),
    ("RECEIVER", "receiver_player_id", "receiver_player_name"),
    ("RUSHER", "rusher_player_id", "rusher_player_name"),
    ("INTERCEPTOR", "interception_player_id", "interception_player_name"),
    ("PUNT_RETURNER", "punt_returner_player_id", "punt_returner_player_name"),
    ("KICKOFF_RETURNER", "kickoff_returner_player_id", "kickoff_returner_player_name"),
    ("PUNTER", "punter_player_id", "punter_player_name"),
    ("KICKER", "kicker_player_id", "kicker_player_name"),
    ("TOUCHDOWN_SCORER", "td_player_id", "td_player_name"),
    ("FUMBLER", "fumbled_1_player_id", "fumbled_1_player_name"),
    ("FUMBLE_RECOVERER", "fumble_recovery_1_player_id", "fumble_recovery_1_player_name"),
    ("FORCED_FUMBLE", "forced_fumble_player_1_player_id", "forced_fumble_player_1_player_name"),
    ("SACKER", "sack_player_id", "sack_player_name"),
    ("HALF_SACK", "half_sack_1_player_id", "half_sack_1_player_name"),
    ("HALF_SACK", "half_sack_2_player_id", "half_sack_2_player_name"),
    ("BLOCKER", "blocked_player_id", "blocked_player_name"),
    ("PASS_DEFENSE", "pass_defense_1_player_id", "pass_defense_1_player_name"),
    ("PASS_DEFENSE", "pass_defense_2_player_id", "pass_defense_2_player_name"),
    ("TACKLER", "solo_tackle_1_player_id", "solo_tackle_1_player_name"),
    ("TACKLER", "solo_tackle_2_player_id", "solo_tackle_2_player_name"),
    ("ASSIST_TACKLER", "assist_tackle_1_player_id", "assist_tackle_1_player_name"),
    ("ASSIST_TACKLER", "assist_tackle_2_player_id", "assist_tackle_2_player_name"),
)


def _participant_records(
    participation: Mapping[str, object],
    *,
    unit: str,
    team: str,
) -> list[dict[str, object]]:
    prefix = unit.lower()
    values = _parallel(
        [
            _split(participation.get(f"{prefix}_players")),
            _split(participation.get(f"{prefix}_names")),
            _split(participation.get(f"{prefix}_positions")),
            _split(participation.get(f"{prefix}_numbers")),
        ],
        f"{team}.{unit}",
    )
    return [
        {
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "jersey_number": number,
            "team": team,
            "unit": unit,
        }
        for player_id, player_name, position, number in values
    ]


def _registry_factors(registry: Mapping[str, object]) -> list[Mapping[str, object]]:
    factors = registry.get("factors")
    if not isinstance(factors, list):
        raise NFLFactExtractionError("factor registry must contain a factor list")
    result: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for factor in factors:
        if not isinstance(factor, Mapping):
            raise NFLFactExtractionError("factor definition must be an object")
        factor_id = _text(factor.get("factor_id"))
        if factor_id is None or factor_id in seen:
            raise NFLFactExtractionError("factor IDs must be present and unique")
        seen.add(factor_id)
        result.append(factor)
    return result


def _predicate_matches(predicate: object, event: Mapping[str, object]) -> bool:
    if not isinstance(predicate, Mapping):
        raise NFLFactExtractionError("factor predicate must be an object")
    if "all" in predicate:
        children = predicate["all"]
        return isinstance(children, list) and all(
            _predicate_matches(child, event) for child in children
        )
    if "any" in predicate:
        children = predicate["any"]
        return isinstance(children, list) and any(
            _predicate_matches(child, event) for child in children
        )
    if "not" in predicate:
        return not _predicate_matches(predicate["not"], event)
    field = _text(predicate.get("field"))
    operator = (_text(predicate.get("operator")) or "").upper()
    if field is None:
        raise NFLFactExtractionError("predicate field is missing")
    actual = event.get(field)
    expected = predicate.get("value")
    if field == "outcome_tags" and isinstance(actual, str):
        actual = json.loads(actual)
    if operator == "EQ":
        return actual == expected
    if operator == "NE":
        return actual != expected
    if operator == "CONTAINS":
        return isinstance(actual, (list, tuple, set, str)) and expected in actual
    if operator == "IN":
        return isinstance(expected, list) and actual in expected
    if operator in {"GT", "GTE", "LT", "LTE"}:
        left, right = _number(actual), _number(expected)
        if left is None or right is None:
            return False
        return {
            "GT": left > right,
            "GTE": left >= right,
            "LT": left < right,
            "LTE": left <= right,
        }[operator]
    raise NFLFactExtractionError(f"unsupported factor operator: {operator}")


def build_game_fact_tables(
    pbp_rows: pd.DataFrame,
    participation_rows: pd.DataFrame,
    *,
    factor_registry: Mapping[str, object],
    pbp_source_sha256: str,
    participation_source_sha256: str,
) -> GameFactTables:
    """Describe one game and evaluate deterministic registered factor predicates."""

    pbp_sha = _require_sha(pbp_source_sha256, "pbp_source_sha256")
    participation_sha = _require_sha(
        participation_source_sha256, "participation_source_sha256"
    )
    if not isinstance(pbp_rows, pd.DataFrame) or pbp_rows.empty:
        raise NFLFactExtractionError("pbp_rows must be a non-empty DataFrame")
    if not isinstance(participation_rows, pd.DataFrame):
        raise NFLFactExtractionError("participation_rows must be a DataFrame")
    game_ids = {_text(value) for value in pbp_rows["game_id"]}
    if None in game_ids or len(game_ids) != 1:
        raise NFLFactExtractionError("pbp_rows must contain exactly one game")
    game_id = next(iter(game_ids))
    assert game_id is not None
    raw = pbp_rows.copy()
    raw["_play_id"] = raw["play_id"].map(_stable_play_id)
    if raw["_play_id"].duplicated().any():
        raise NFLFactExtractionError("play_id must be unique within the game")
    raw["_order"] = pd.to_numeric(raw["order_sequence"], errors="raise").astype(int)
    raw = raw.sort_values(["_order", "_play_id"], kind="mergesort").reset_index(drop=True)

    participation_index: dict[str, dict[str, object]] = {}
    for value in participation_rows.to_dict("records"):
        row_game = _text(value.get("nflverse_game_id", value.get("game_id")))
        if row_game != game_id:
            continue
        play_id = _stable_play_id(value.get("play_id"))
        if play_id in participation_index:
            raise NFLFactExtractionError("participation play_id must be unique")
        participation_index[play_id] = value

    home_team = _text(raw.iloc[0].get("home_team"))
    away_team = _text(raw.iloc[0].get("away_team"))
    if home_team is None or away_team is None or home_team == away_team:
        raise NFLFactExtractionError("game teams are invalid")

    records = raw.to_dict("records")
    next_possession: list[str | None] = [None] * len(records)
    next_structured_state: list[dict[str, object] | None] = [None] * len(records)
    future: str | None = None
    future_state: dict[str, object] | None = None
    for index in range(len(records) - 1, -1, -1):
        next_possession[index] = future
        next_structured_state[index] = future_state
        current = _text(records[index].get("posteam"))
        if current is not None:
            future = current
        current_yardline = _text(records[index].get("yrdln"))
        if current is not None and current_yardline is not None:
            future_state = {
                "play_id": _stable_play_id(records[index].get("play_id")),
                "possession": current,
                "yardline": current_yardline,
                "coordinate": _fixed_field_coordinate(
                    current_yardline,
                    away_team=away_team,
                    home_team=home_team,
                ),
            }

    event_rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []
    player_rows: list[dict[str, object]] = []
    injury_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    roster_lookup: dict[tuple[str, str], tuple[str, str]] = {}
    roster_snapshots: list[tuple[int, str, str, str, set[str]]] = []
    pending_score_sequence: tuple[str, str | None] | None = None

    for index, row in enumerate(records):
        play_id = str(row["_play_id"])
        event_id = f"{game_id}:episode:{play_id}"
        participation = participation_index.get(play_id)
        posteam = _text(row.get("posteam"))
        defteam = _text(row.get("defteam"))
        participants: list[dict[str, object]] = []
        if participation is not None and posteam is not None and defteam is not None:
            participants.extend(
                _participant_records(participation, unit="OFFENSE", team=posteam)
            )
            participants.extend(
                _participant_records(participation, unit="DEFENSE", team=defteam)
            )
            for participant in participants:
                roster_lookup[
                    (str(participant["team"]), str(participant["jersey_number"]))
                ] = (
                    str(participant["player_id"]),
                    str(participant["player_name"]),
                )
            for unit, team in (("OFFENSE", posteam), ("DEFENSE", defteam)):
                ids = {
                    str(item["player_id"])
                    for item in participants
                    if item["unit"] == unit
                }
                roster_snapshots.append(
                    (int(row["_order"]), event_id, team, unit, ids)
                )

        pre_home, pre_away = _pre_scores(row)
        post_home = _integer(row.get("total_home_score"))
        post_away = _integer(row.get("total_away_score"))
        action = _primary_action(row)
        tags = _outcome_tags(row, next_possession=next_possession[index])
        no_play = _text(row.get("play_type")) == "no_play"
        deleted = _indicator(row.get("play_deleted"))
        information_status = "FINAL"
        (
            source_interval_start,
            source_interval_end,
            source_resolution,
            known_at,
        ) = _source_interval(row)
        consolidated_adjudication = (
            _indicator(row.get("replay_or_challenge"))
            or _text(row.get("replay_or_challenge_result")) is not None
        )
        adjudication_sequence_id = None
        if consolidated_adjudication:
            known_at = None
        score_sequence_id: str | None = None
        if "TOUCHDOWN" in tags and not (no_play or deleted):
            score_sequence_id = f"{game_id}:score:{play_id}"
            pending_score_sequence = (score_sequence_id, _text(row.get("td_team")) or posteam)
        elif action == "TRY" and pending_score_sequence is not None:
            pending_id, pending_team = pending_score_sequence
            if pending_team is None or posteam == pending_team:
                score_sequence_id = pending_id
            pending_score_sequence = None
        elif action not in {"ADMIN", "TIMEOUT", "PENALTY"}:
            pending_score_sequence = None
        admin = action in {"ADMIN", "TIMEOUT", "PENALTY"}
        final_sports_outcome_eligible = (
            not (no_play or deleted or admin)
        )
        factor_eligible = final_sports_outcome_eligible
        stage_b_information_event_eligible = (
            not deleted
            and known_at is not None
        )
        actor_team = _actor_team(
            row,
            action=action,
            tags=tags,
            home_team=home_team,
            away_team=away_team,
        )
        beneficiary_team, beneficiary_resolution_status = _beneficiary_team(
            row,
            tags=tags,
            final_sports_outcome_eligible=final_sports_outcome_eligible,
            home_team=home_team,
            away_team=away_team,
        )
        yardline = _number(row.get("yardline_100"))
        score_margin_home = (
            None
            if pre_home is None or pre_away is None
            else pre_home - pre_away
        )
        score_margin_offense = (
            None
            if score_margin_home is None or posteam is None
            else (
                score_margin_home
                if posteam == home_team
                else -score_margin_home
            )
        )
        vegas_before = _number(row.get("vegas_home_wp"))
        vegas_delta = _number(row.get("vegas_home_wpa"))
        pre_coordinate = _fixed_field_coordinate(
            row.get("yrdln"), away_team=away_team, home_team=home_team
        )
        next_state = next_structured_state[index]
        visual_end_coordinate, visual_end_semantics = _visual_end_coordinate(
            row,
            action=action,
            tags=tags,
            pre_coordinate=pre_coordinate,
            next_state_coordinate=(
                _number(next_state.get("coordinate"))
                if next_state is not None
                else None
            ),
            posteam=posteam,
            home_team=home_team,
            away_team=away_team,
        )
        transition_direction, transition_direction_semantics = (
            _transition_direction_semantics(
                action=action,
                tags=tags,
                posteam=posteam,
                home_team=home_team,
                away_team=away_team,
            )
        )
        posteam_semantics = {
            "KICKOFF": "NFLVERSE_POSTEAM_IS_RECEIVING_TEAM",
            "PUNT": "NFLVERSE_POSTEAM_IS_PUNTING_TEAM_BEFORE_RETURN",
            "TRY": "NFLVERSE_POSTEAM_IS_ATTEMPTING_TEAM",
            "FIELD_GOAL": "NFLVERSE_POSTEAM_IS_KICKING_TEAM",
        }.get(action, "NFLVERSE_POSTEAM_IS_SCRIMMAGE_OFFENSE")
        event = {
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": CLAIM_BOUNDARY,
            "game_id": game_id,
            "event_id": event_id,
            "raw_play_id": play_id,
            "play_id": play_id,
            "atomic_information_episode_id": event_id,
            "score_sequence_id": score_sequence_id,
            "adjudication_sequence_id": adjudication_sequence_id,
            "information_status": information_status,
            "stage_b_information_event_eligible": (
                stage_b_information_event_eligible
            ),
            "final_sports_outcome_eligible": final_sports_outcome_eligible,
            "source_interval_start": source_interval_start,
            "source_interval_end": source_interval_end,
            "source_resolution": source_resolution,
            "source_interval_semantics": "[START,END)",
            "known_at": known_at,
            "order_sequence": int(row["_order"]),
            "source_time_utc": _text(row.get("time_of_day")),
            "source_time_semantics": "NFLVERSE_SOURCE_POINT_NOT_LOCAL_RECEIVE_TIME",
            "quarter": _integer(row.get("qtr")),
            "game_clock": _text(row.get("time")),
            "game_seconds_remaining": _integer(row.get("game_seconds_remaining")),
            "home_team": home_team,
            "away_team": away_team,
            "actor_team": actor_team,
            "beneficiary_team": beneficiary_team,
            "actor_is_home": _is_home(
                actor_team, home_team=home_team, away_team=away_team
            ),
            "beneficiary_is_home": _is_home(
                beneficiary_team, home_team=home_team, away_team=away_team
            ),
            "possession_is_home": _is_home(
                posteam, home_team=home_team, away_team=away_team
            ),
            "beneficiary_resolution_status": beneficiary_resolution_status,
            "possession_before": posteam,
            "posteam_semantics": posteam_semantics,
            "defense_team": defteam,
            "next_observed_possession": next_possession[index],
            "offense_direction": transition_direction,
            "transition_direction_semantics": transition_direction_semantics,
            "field_orientation_semantics": (
                "TEAM_NORMALIZED_AWAY_END_LEFT_HOME_END_RIGHT_NOT_PHYSICAL_STADIUM_DIRECTION"
            ),
            "pre_home_score": pre_home,
            "pre_away_score": pre_away,
            "post_home_score": post_home,
            "post_away_score": post_away,
            "score_margin_home": score_margin_home,
            "score_margin_offense": score_margin_offense,
            "down": _integer(row.get("down")),
            "distance": _integer(row.get("ydstogo")),
            "goal_to_go": _indicator(row.get("goal_to_go")),
            "pre_yardline": _text(row.get("yrdln")),
            "post_yardline": _text(row.get("end_yard_line")),
            "pre_field_coordinate_0_100": pre_coordinate,
            "post_field_coordinate_0_100": _fixed_field_coordinate(
                row.get("end_yard_line"), away_team=away_team, home_team=home_team
            ),
            "next_observed_state_play_id": (
                _text(next_state.get("play_id")) if next_state is not None else None
            ),
            "next_observed_possession_state": (
                _text(next_state.get("possession"))
                if next_state is not None
                else None
            ),
            "next_observed_yardline": (
                _text(next_state.get("yardline")) if next_state is not None else None
            ),
            "next_observed_field_coordinate_0_100": (
                _number(next_state.get("coordinate"))
                if next_state is not None
                else None
            ),
            "visual_end_field_coordinate_0_100": visual_end_coordinate,
            "visual_end_semantics": visual_end_semantics,
            "yardline_100": yardline,
            "pre_red_zone": yardline is not None and yardline <= 20,
            "tied": score_margin_home == 0 if score_margin_home is not None else None,
            "one_score_game": (
                abs(score_margin_home) <= 8 if score_margin_home is not None else None
            ),
            "drive_id": _text(row.get("drive")),
            "series_id": _text(row.get("series")),
            "primary_action": action,
            "outcome_tags": json.dumps(tags, separators=(",", ":")),
            "yards_gained": _number(row.get("yards_gained")),
            "air_yards": _number(row.get("air_yards")),
            "yards_after_catch": _number(row.get("yards_after_catch")),
            "return_yards": _number(row.get("return_yards")),
            "pass_length": _text(row.get("pass_length")),
            "pass_location": _text(row.get("pass_location")),
            "run_location": _text(row.get("run_location")),
            "run_gap": _text(row.get("run_gap")),
            "kick_distance": _number(row.get("kick_distance")),
            "field_goal_result": _text(row.get("field_goal_result")),
            "extra_point_result": _text(row.get("extra_point_result")),
            "two_point_result": _text(row.get("two_point_conv_result")),
            "penalty_team": _text(row.get("penalty_team")),
            "penalty_type": _text(row.get("penalty_type")),
            "penalty_yards": _number(row.get("penalty_yards")),
            "review_result": _text(row.get("replay_or_challenge_result")),
            "timeout_team": _text(row.get("timeout_team")),
            "home_timeouts_remaining": _integer(row.get("home_timeouts_remaining")),
            "away_timeouts_remaining": _integer(row.get("away_timeouts_remaining")),
            "description": _text(row.get("desc")),
            "participation_status": (
                "MISSING"
                if participation is None
                else (
                    "PRESENT_COMPLETE_11V11"
                    if _integer(participation.get("n_offense")) == 11
                    and _integer(participation.get("n_defense")) == 11
                    else "PRESENT_PARTIAL"
                )
            ),
            "factor_eligible": factor_eligible,
            "row_disposition": (
                "DELETED"
                if deleted
                else "NULLIFIED_NO_PLAY"
                if no_play
                else "ADMINISTRATIVE"
                if admin
                else "LIVE_PLAY"
            ),
            "home_wp_before_diagnostic": vegas_before,
            "home_wp_after_diagnostic": (
                None
                if vegas_before is None or vegas_delta is None
                else vegas_before + vegas_delta
            ),
            "home_wp_delta_diagnostic": vegas_delta,
            "reference_semantics": "NFLVERSE_VEGAS_WP_DIAGNOSTIC_NOT_GROUND_TRUTH",
            "source_hashes": json.dumps(
                [pbp_sha, participation_sha],
                separators=(",", ":"),
            ),
            "pbp_source_sha256": pbp_sha,
            "participation_source_sha256": participation_sha,
        }
        event_rows.append(event)
        tag_rows.extend(
            {
                "game_id": game_id,
                "event_id": event_id,
                "play_id": play_id,
                "tag": tag,
                "pbp_source_sha256": pbp_sha,
            }
            for tag in tags
        )
        seen_players: set[tuple[str, str, str]] = set()
        for role, id_field, name_field in _EXPLICIT_ROLES:
            player_id = _text(row.get(id_field))
            player_name = _text(row.get(name_field))
            if player_id is None and player_name is None:
                continue
            key = (role, player_id or "", player_name or "")
            if key in seen_players:
                continue
            seen_players.add(key)
            player_rows.append(
                {
                    "game_id": game_id,
                    "event_id": event_id,
                    "play_id": play_id,
                    "role": role,
                    "player_id": player_id,
                    "player_name": player_name,
                    "team": (
                        defteam
                        if role
                        in {
                            "INTERCEPTOR",
                            "FORCED_FUMBLE",
                            "SACKER",
                            "HALF_SACK",
                            "PASS_DEFENSE",
                            "TACKLER",
                            "ASSIST_TACKLER",
                        }
                        else posteam
                    ),
                    "position": None,
                    "jersey_number": None,
                    "evidence_class": "EXPLICIT_PBP_ROLE",
                    "source_sha256": pbp_sha,
                }
            )
        for participant in participants:
            role = f"{participant['unit']}_PARTICIPANT"
            key = (
                role,
                str(participant["player_id"]),
                str(participant["player_name"]),
            )
            if key in seen_players:
                continue
            seen_players.add(key)
            player_rows.append(
                {
                    "game_id": game_id,
                    "event_id": event_id,
                    "play_id": play_id,
                    "role": role,
                    **participant,
                    "evidence_class": "OBSERVED_PLAY_PARTICIPATION_RESEARCH_ONLY",
                    "source_sha256": participation_sha,
                }
            )
        reconciliation_rows.append(
            {
                "game_id": game_id,
                "raw_play_id": play_id,
                "event_id": event_id,
                "atomic_information_episode_id": event_id,
                "raw_row_preserved": True,
                "row_disposition": event["row_disposition"],
                "factor_eligible": factor_eligible,
                "exclusion_reason": (
                    None
                    if factor_eligible
                    else str(event["row_disposition"]).lower()
                ),
                "pbp_source_sha256": pbp_sha,
            }
        )

    # Build the full roster map before resolving source-declared injury names.
    for row, event in zip(records, event_rows, strict=True):
        description = _text(row.get("desc")) or ""
        for pattern, evidence_type, status in (
            (_INJURY_RE, "CONFIRMED_IN_GAME_INJURY", "INJURED"),
            (_RETURN_RE, "CONFIRMED_RETURN_TO_GAME", "RETURNED"),
        ):
            for match in pattern.finditer(description):
                team = match.group("team").upper()
                number = match.group("number")
                source_name = match.group("name").strip()
                resolved = roster_lookup.get((team, number))
                injury_rows.append(
                    {
                        "game_id": game_id,
                        "event_id": event["event_id"],
                        "play_id": event["play_id"],
                        "source_time_utc": event["source_time_utc"],
                        "evidence_type": evidence_type,
                        "status": status,
                        "team": team,
                        "jersey_number": number,
                        "source_player_name": source_name,
                        "player_id": None if resolved is None else resolved[0],
                        "player_name": None if resolved is None else resolved[1],
                        "identity_status": (
                            "RESOLVED_FROM_GAME_PARTICIPATION"
                            if resolved is not None
                            else "UNRESOLVED_SOURCE_IDENTITY"
                        ),
                        "evidence_grade": "SOURCE_TEXT_DECLARED",
                        "evidence_semantics": (
                            "nflverse PBP explicitly declares injury status; "
                            "never inferred from roster differences"
                        ),
                        "raw_evidence_text": description,
                        "pbp_source_sha256": pbp_sha,
                        "participation_source_sha256": participation_sha,
                    }
                )

    availability_rows: list[dict[str, object]] = []
    previous: dict[tuple[str, str], tuple[str, set[str]]] = {}
    for _, event_id, team, unit, player_ids in sorted(roster_snapshots):
        key = (team, unit)
        prior = previous.get(key)
        if prior is not None:
            prior_event, prior_ids = prior
            for status, ids in (
                ("OBSERVED_EXIT", sorted(prior_ids - player_ids)),
                ("OBSERVED_ENTRY", sorted(player_ids - prior_ids)),
            ):
                availability_rows.extend(
                    {
                        "game_id": game_id,
                        "team": team,
                        "unit": unit,
                        "player_id": player_id,
                        "availability_observation": status,
                        "interval_start_event_id": prior_event,
                        "interval_end_event_id": event_id,
                        "evidence_grade": "BETWEEN_OBSERVED_PARTICIPATION_ROWS",
                        "evidence_semantics": (
                            "set difference between consecutive observed snaps for "
                            "the same team and unit; not an injury or exact substitution time"
                        ),
                        "participation_source_sha256": participation_sha,
                    }
                    for player_id in ids
                )
        previous[key] = (event_id, player_ids)

    events = pd.DataFrame(event_rows)
    tags = pd.DataFrame(tag_rows)
    players = pd.DataFrame(player_rows)
    injury_evidence = pd.DataFrame(injury_rows)
    availability = pd.DataFrame(availability_rows)
    reconciliation = pd.DataFrame(reconciliation_rows)
    registry_sha = _canonical_sha(factor_registry)
    hit_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    event_records = events.to_dict("records")
    for factor in _registry_factors(factor_registry):
        factor_id = str(factor["factor_id"])
        status = _text(factor.get("status")) or "DRAFT"
        scope = (_text(factor.get("scope")) or "LIVE_ONLY").upper()
        matching: list[Mapping[str, object]] = []
        if status == "ACTIVE":
            for event in event_records:
                if scope == "LIVE_ONLY" and not bool(event["factor_eligible"]):
                    continue
                if _predicate_matches(factor.get("predicate"), event):
                    matching.append(event)
                    hit_rows.append(
                        {
                            "game_id": game_id,
                            "event_id": event["event_id"],
                            "play_id": event["play_id"],
                            "factor_id": factor_id,
                            "factor_version": _text(factor.get("version")) or "v1",
                            "registry_sha256": registry_sha,
                            "pbp_source_sha256": pbp_sha,
                            "predicate_evidence": json.dumps(
                                factor.get("predicate"),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
        coverage_rows.append(
            {
                "factor_id": factor_id,
                "factor_version": _text(factor.get("version")) or "v1",
                "status": status,
                "scope": scope,
                "event_count": len(matching),
                "game_count": 1 if matching else 0,
                "availability_status": (
                    "AVAILABLE"
                    if status == "ACTIVE"
                    else "NOT_RUN_NON_ACTIVE"
                ),
                "registry_sha256": registry_sha,
            }
        )
    factor_hits = pd.DataFrame(hit_rows)
    factor_coverage = pd.DataFrame(coverage_rows)
    return GameFactTables(
        events=events,
        tags=tags,
        players=players,
        availability=availability,
        injury_evidence=injury_evidence,
        factor_hits=factor_hits,
        factor_coverage=factor_coverage,
        reconciliation=reconciliation,
        registry_sha256=registry_sha,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "GameFactTables",
    "NFLFactExtractionError",
    "SCHEMA_VERSION",
    "build_game_fact_tables",
]
