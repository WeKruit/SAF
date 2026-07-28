"""Deterministic PIT sports-feature construction for the NFL Factor Lab.

This module consumes rows only after their immutable source objects have been
verified by the caller.  It never reads the large association table: market
observations are joined later against the resulting play/episode keys.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Callable

from prediction_market.sports.nfl_factor_lab_cleaning import (
    SportsRowCleaningError,
    SportsRowDispositionLedgerV2,
    SportsRowDispositionV2,
    build_sports_row_disposition_ledger,
    semantic_rows_sha256,
)
from prediction_market.sports.nfl_factor_lab_types import (
    BetweenPlayRosterChangeV1,
    PlayerRoleV1,
    RollingFeatureV1,
    SportsFeatureRowV1,
    SportsFeatureViewV1,
)


class SportsFeatureBuildError(ValueError):
    """Verified inputs cannot produce a PIT-safe SportsFeatureViewV1."""


NFL_FACTOR_LAB_PBP_COLUMNS = (
    "game_id",
    "play_id",
    "order_sequence",
    "season",
    "season_type",
    "week",
    "game_date",
    "time_of_day",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "qtr",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "posteam_score",
    "defteam_score",
    "total_home_score",
    "total_away_score",
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "drive",
    "fixed_drive",
    "desc",
    "play_type",
    "pass_attempt",
    "rush_attempt",
    "qb_scramble",
    "sack",
    "qb_kneel",
    "qb_spike",
    "yards_gained",
    "first_down",
    "success",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "return_touchdown",
    "field_goal_attempt",
    "field_goal_result",
    "kick_distance",
    "extra_point_attempt",
    "extra_point_result",
    "two_point_attempt",
    "two_point_conv_result",
    "defensive_two_point_attempt",
    "defensive_two_point_conv",
    "interception",
    "fumble",
    "fumble_lost",
    "fumble_recovery_1_team",
    "fumble_recovery_1_yards",
    "fourth_down_converted",
    "fourth_down_failed",
    "punt_attempt",
    "punt_blocked",
    "punt_in_endzone",
    "punt_out_of_bounds",
    "punt_downed",
    "punt_inside_twenty",
    "punt_fair_catch",
    "kickoff_attempt",
    "kickoff_inside_twenty",
    "kickoff_in_endzone",
    "kickoff_out_of_bounds",
    "kickoff_downed",
    "kickoff_fair_catch",
    "own_kickoff_recovery",
    "own_kickoff_recovery_td",
    "touchback",
    "return_team",
    "return_yards",
    "safety",
    "penalty",
    "penalty_team",
    "penalty_yards",
    "penalty_type",
    "timeout",
    "timeout_team",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "replay_or_challenge",
    "replay_or_challenge_result",
    "play_deleted",
    "shotgun",
    "no_huddle",
    "offense_formation",
    "offense_personnel",
    "defense_personnel",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "kicker_player_id",
    "kicker_player_name",
    "punter_player_id",
    "punter_player_name",
    "punt_returner_player_id",
    "punt_returner_player_name",
    "kickoff_returner_player_id",
    "kickoff_returner_player_name",
    "interception_player_id",
    "interception_player_name",
    "fumbled_1_player_id",
    "fumbled_1_player_name",
    "fumble_recovery_1_player_id",
    "fumble_recovery_1_player_name",
    "solo_tackle_1_player_id",
    "solo_tackle_1_player_name",
    "complete_pass",
    "receiving_yards",
    "wp",
    "vegas_wp",
    "wpa",
    "vegas_home_wpa",
    "epa",
    "qb_epa",
    "xpass",
    "pass_oe",
    "cp",
    "cpoe",
    "yac_epa",
    "xyac_epa",
)


def _records(value: object, *, name: str) -> list[dict[str, Any]]:
    if hasattr(value, "to_pylist"):
        value = value.to_pylist()
    elif hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict(orient="records")
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise SportsFeatureBuildError(f"{name} must be a record sequence")
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise SportsFeatureBuildError(f"{name} contains a non-record")
        rows.append(dict(row))
    return rows


def _stable_id(value: object, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise SportsFeatureBuildError(f"{field} is missing")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    if type(value) is str and value and value == value.strip():
        return value
    raise SportsFeatureBuildError(f"{field} is not a stable scalar")


def _text(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise SportsFeatureBuildError("text field has a non-string value")
    stripped = value.strip()
    return stripped or None


def _integer(value: object, *, field: str, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise SportsFeatureBuildError(f"{field} is missing")
        return None
    if isinstance(value, bool):
        raise SportsFeatureBuildError(f"{field} is not an integer")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        number = float(value)
        if number.is_integer():
            return int(number)
    raise SportsFeatureBuildError(f"{field} is not an integer")


def _nonnegative_source_integer(
    value: object, *, field: str
) -> int | None:
    """Interpret a negative provider sentinel as explicitly missing.

    nflverse currently contains negative timeout-remaining values on a small
    number of overtime rows.  They are not a valid football state and must not
    be silently clamped to zero.
    """

    observed = _integer(value, field=field)
    return None if observed is not None and observed < 0 else observed


def _number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SportsFeatureBuildError(f"{field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _indicator(value: object, *, field: str) -> bool:
    if value is None:
        return False
    if type(value) is bool:
        return value
    number = _integer(value, field=field, required=True)
    if number not in {0, 1}:
        raise SportsFeatureBuildError(f"{field} must be binary")
    return number == 1


def _optional_indicator(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    return _indicator(value, field=field)


def _parse_utc(value: object, *, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise SportsFeatureBuildError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise SportsFeatureBuildError(
            f"{field} must be an RFC3339 UTC timestamp"
        ) from error
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_flags(row: Mapping[str, object], *, nullified: bool) -> tuple[str, ...]:
    flags: set[str] = set()
    play_type = _text(row.get("play_type"))
    if play_type:
        flags.add(play_type.replace(" ", "_"))
    binary_fields = {
        "pass_attempt": "pass",
        "rush_attempt": "rush",
        "qb_scramble": "scramble",
        "sack": "sack",
        "qb_kneel": "kneel",
        "qb_spike": "spike",
        "first_down": "first_down",
        "success": "success",
        "touchdown": "touchdown",
        "pass_touchdown": "pass_touchdown",
        "rush_touchdown": "rush_touchdown",
        "return_touchdown": "return_touchdown",
        "field_goal_attempt": "field_goal_attempt",
        "extra_point_attempt": "pat_attempt",
        "two_point_attempt": "two_point_attempt",
        "defensive_two_point_attempt": "defensive_two_point_attempt",
        "defensive_two_point_conv": "defensive_two_point_conversion",
        "interception": "interception",
        "fumble": "fumble",
        "fumble_lost": "fumble_lost",
        "fourth_down_converted": "fourth_down_conversion",
        "fourth_down_failed": "fourth_down_failure",
        "punt_attempt": "punt",
        "punt_blocked": "blocked_punt",
        "punt_inside_twenty": "punt_inside_twenty",
        "punt_fair_catch": "punt_fair_catch",
        "kickoff_attempt": "kickoff",
        "touchback": "touchback",
        "safety": "safety",
        "penalty": "penalty",
        "timeout": "timeout",
        "replay_or_challenge": "review",
        "play_deleted": "deleted",
    }
    for field, flag in binary_fields.items():
        if _indicator(row.get(field), field=field):
            flags.add(flag)
    categorical_fields = {
        "field_goal_result": "field_goal",
        "extra_point_result": "pat",
        "two_point_conv_result": "two_point",
        "replay_or_challenge_result": "review",
    }
    for field, prefix in categorical_fields.items():
        value = _text(row.get(field))
        if value:
            flags.add(f"{prefix}_{value.lower().replace(' ', '_')}")
    if _text(row.get("fumble_recovery_1_team")):
        flags.add("fumble_recovery")
    if nullified:
        flags.add("nullified")
        consequence_flags = {
            "touchdown",
            "pass_touchdown",
            "rush_touchdown",
            "return_touchdown",
            "interception",
            "fumble_lost",
            "fourth_down_conversion",
            "fourth_down_failure",
            "safety",
            "defensive_two_point_conversion",
        }
        flags.difference_update(consequence_flags)
        flags = {
            flag
            for flag in flags
            if not flag.startswith(("field_goal_", "pat_", "two_point_"))
        }
    return tuple(sorted(flags))


_ROLE_FIELDS = (
    ("passer", "passer_player_id", "passer_player_name"),
    ("rusher", "rusher_player_id", "rusher_player_name"),
    ("receiver", "receiver_player_id", "receiver_player_name"),
    ("kicker", "kicker_player_id", "kicker_player_name"),
    ("punter", "punter_player_id", "punter_player_name"),
    ("punt_returner", "punt_returner_player_id", "punt_returner_player_name"),
    ("kickoff_returner", "kickoff_returner_player_id", "kickoff_returner_player_name"),
    ("interceptor", "interception_player_id", "interception_player_name"),
    ("fumbler", "fumbled_1_player_id", "fumbled_1_player_name"),
    (
        "fumble_recoverer",
        "fumble_recovery_1_player_id",
        "fumble_recovery_1_player_name",
    ),
    ("solo_tackler", "solo_tackle_1_player_id", "solo_tackle_1_player_name"),
)


def _player_roles(row: Mapping[str, object]) -> tuple[PlayerRoleV1, ...]:
    result: list[PlayerRoleV1] = []
    for role, id_field, name_field in _ROLE_FIELDS:
        player_id = _text(row.get(id_field))
        player_name = _text(row.get(name_field))
        if player_id is not None or player_name is not None:
            result.append(
                PlayerRoleV1(
                    role=role,
                    player_id=player_id,
                    player_name=player_name,
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.role,
                item.player_id or "",
                item.player_name or "",
            ),
        )
    )


def _field_zone(yardline_100: int | None) -> str | None:
    if yardline_100 is None:
        return None
    if yardline_100 <= 2:
        return "goal_line"
    if yardline_100 <= 20:
        return "red_zone"
    if yardline_100 < 50:
        return "opponent"
    if yardline_100 == 50:
        return "midfield"
    return "own"


def _historical_exclusion_reasons(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    reasons: set[str] = set()
    play_type = (_text(row.get("play_type")) or "").lower()
    description = " ".join(
        (_text(row.get("desc", row.get("description"))) or "").upper().split()
    )
    live_action = bool(
        play_type
        in {
            "pass",
            "run",
            "qb_kneel",
            "qb_spike",
            "sack",
            "punt",
            "field_goal",
            "kickoff",
            "extra_point",
            "two_point_attempt",
        }
        or any(
            _indicator(row.get(field), field=field)
            for field in (
                "pass_attempt",
                "rush_attempt",
                "sack",
                "qb_kneel",
                "qb_spike",
                "punt_attempt",
                "field_goal_attempt",
                "kickoff_attempt",
                "extra_point_attempt",
                "two_point_attempt",
            )
        )
    )
    if play_type == "no_play" or _indicator(row.get("no_play"), field="no_play"):
        reasons.add("no_play")
    if (
        _indicator(row.get("nullified"), field="nullified")
        or _indicator(
            row.get("episode_nullified"),
            field="episode_nullified",
        )
    ):
        reasons.add("nullified")
    if (
        _indicator(row.get("play_deleted"), field="play_deleted")
        or _indicator(row.get("deleted"), field="deleted")
    ):
        reasons.add("deleted")
    review_result = (
        _text(row.get("replay_or_challenge_result"))
        or _text(row.get("review_result"))
        or ""
    ).lower()
    if (
        _indicator(row.get("replay_or_challenge"), field="replay_or_challenge")
        or _indicator(row.get("review"), field="review")
        or (review_result and review_result != "none")
    ):
        reasons.add("review")
    if "reversed" in review_result:
        reasons.add("reversed")
    if (
        description == "GAME"
        or description.startswith("END ")
        or description.startswith("OT END ")
        or _indicator(row.get("game_end"), field="game_end")
        or _indicator(row.get("timeout"), field="timeout")
        or (_indicator(row.get("penalty"), field="penalty") and not live_action)
    ):
        reasons.add("administrative")
    if row.get("time_of_day") is None and row.get("source_time_start_utc") is None:
        reasons.add("source_time_missing")
    if not live_action:
        reasons.add("not_live_play")
    return tuple(sorted(reasons))


class _HistoryIndex:
    """Small in-memory index of completed immutable PBP games."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> None:
        games: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            game_id = _stable_id(row.get("game_id"), field="historical game_id")
            games[game_id].append(dict(row))
        self.games: dict[str, tuple[dict[str, Any], ...]] = {}
        self.ends: dict[str, datetime] = {}
        self.seasons: dict[str, int] = {}
        self.raw_row_counts: dict[str, int] = {}
        self.excluded_row_counts: dict[str, int] = {}
        self.audit_rows: dict[str, tuple[dict[str, object], ...]] = {}
        self._completed_game_ids_cache: dict[
            tuple[datetime, str, str, str, int | None, int | None],
            tuple[str, ...],
        ] = {}
        for game_id, game_rows in games.items():
            times: list[datetime] = []
            for row in game_rows:
                value = (
                    row.get("source_time_end_utc")
                    or row.get("source_time_start_utc")
                    or row.get("time_of_day")
                )
                if value is not None:
                    times.append(
                        _parse_utc(
                            value,
                            field="historical source time",
                        )
                    )
            seasons = {
                _integer(row.get("season"), field="historical season", required=True)
                for row in game_rows
            }
            if len(seasons) != 1 or None in seasons:
                raise SportsFeatureBuildError(
                    f"historical game {game_id} has inconsistent season"
                )
            ordered = tuple(
                sorted(
                    game_rows,
                    key=lambda row: (
                        _integer(
                            row.get("order_sequence", row.get("play_id")),
                            field="historical order_sequence",
                            required=True,
                        ),
                        _stable_id(row.get("play_id"), field="historical play_id"),
                    ),
                )
            )
            eligible: list[dict[str, Any]] = []
            audit_rows: list[dict[str, object]] = []
            for row in ordered:
                reasons = _historical_exclusion_reasons(row)
                row_eligible = not reasons
                if row_eligible:
                    eligible.append(row)
                audit_rows.append(
                    {
                        "game_id": game_id,
                        "play_id": _stable_id(
                            row.get("play_id"),
                            field="historical play_id",
                        ),
                        "factor_eligible": row_eligible,
                        "exclusion_reasons": list(reasons),
                        "source_hash": semantic_rows_sha256(
                            {
                                "schema_version": "HistoricalSportsSourceRowV2",
                                "row": row,
                            }
                        ),
                    }
                )
            self.games[game_id] = tuple(eligible)
            if times:
                self.ends[game_id] = max(times)
            self.seasons[game_id] = int(next(iter(seasons)))
            self.raw_row_counts[game_id] = len(ordered)
            self.excluded_row_counts[game_id] = len(ordered) - len(eligible)
            self.audit_rows[game_id] = tuple(audit_rows)

    def completed_game_ids(
        self,
        *,
        before: datetime,
        current_game_id: str,
        entity_id: str,
        entity_type: str,
        season: int | None,
        last_games: int | None,
    ) -> tuple[str, ...]:
        cache_key = (
            before,
            current_game_id,
            entity_type,
            entity_id,
            season,
            last_games,
        )
        cached = self._completed_game_ids_cache.get(cache_key)
        if cached is not None:
            return cached
        selected: list[str] = []
        for game_id, rows in self.games.items():
            if game_id == current_game_id:
                continue
            end = self.ends.get(game_id)
            if end is None or end >= before:
                continue
            if season is not None and self.seasons[game_id] != season:
                continue
            if entity_type == "team":
                present = any(
                    entity_id in {row.get("posteam"), row.get("defteam")}
                    for row in rows
                )
            else:
                present = any(
                    entity_id
                    in {
                        row.get("passer_player_id"),
                        row.get("rusher_player_id"),
                        row.get("receiver_player_id"),
                        row.get("punter_player_id"),
                        row.get("punt_returner_player_id"),
                        row.get("kickoff_returner_player_id"),
                    }
                    for row in rows
                )
            if present:
                selected.append(game_id)
        selected.sort(key=lambda item: (self.ends[item], item))
        if last_games is not None:
            selected = selected[-last_games:]
        result = tuple(selected)
        self._completed_game_ids_cache[cache_key] = result
        return result

    def rows_for(self, game_ids: Sequence[str]) -> list[dict[str, Any]]:
        return [row for game_id in game_ids for row in self.games[game_id]]

    def audit(
        self,
        *,
        before: datetime,
        current_game_id: str,
        entity_ids: Sequence[str] = (),
    ) -> dict[str, str | int]:
        entities = frozenset(entity_ids)
        completed = tuple(
            sorted(
                game_id
                for game_id, end in self.ends.items()
                if end < before
                and game_id != current_game_id
                and (
                    not entities
                    or any(
                        entities
                        & {
                            _text(row.get("posteam")),
                            _text(row.get("defteam")),
                        }
                        for row in self.games[game_id]
                    )
                )
            )
        )
        semantic_rows = [
            row
            for game_id in completed
            for row in self.audit_rows[game_id]
        ]
        same_game_rows = [
            row
            for row in self.audit_rows.get(current_game_id, ())
        ]
        return {
            "history_cutoff_utc": _utc_text(before),
            "history_completed_game_count": len(completed),
            "history_eligible_row_count": sum(
                len(self.games[game_id]) for game_id in completed
            ),
            "history_excluded_row_count": sum(
                self.excluded_row_counts[game_id] for game_id in completed
            ),
            "history_same_game_excluded_row_count": len(same_game_rows),
            "history_support_sha256": semantic_rows_sha256(
                {
                    "schema_version": "PITHistorySupportV2",
                    "cutoff_utc": _utc_text(before),
                    "current_game_id": current_game_id,
                    "rows": semantic_rows,
                    "same_game_excluded_rows": same_game_rows,
                }
            ),
        }


MetricSelector = Callable[[Mapping[str, object]], tuple[float, float] | None]


def _ratio(
    rows: Sequence[Mapping[str, object]],
    selector: MetricSelector,
) -> tuple[float | None, int]:
    numerator = 0.0
    denominator = 0.0
    observations = 0
    for row in rows:
        contribution = selector(row)
        if contribution is None:
            continue
        child_numerator, child_denominator = contribution
        if not (
            math.isfinite(child_numerator)
            and math.isfinite(child_denominator)
            and child_denominator >= 0
        ):
            raise SportsFeatureBuildError("rolling metric contribution is invalid")
        numerator += child_numerator
        denominator += child_denominator
        observations += int(child_denominator)
    return (
        (None if denominator == 0 else numerator / denominator),
        observations,
    )


def _mean_numeric(
    rows: Sequence[Mapping[str, object]],
    *,
    predicate: Callable[[Mapping[str, object]], bool],
    field: str,
) -> tuple[float | None, int]:
    values = [
        value
        for row in rows
        if predicate(row) and (value := _number(row.get(field), field=field)) is not None
    ]
    return (None if not values else sum(values) / len(values), len(values))


def _scrimmage(row: Mapping[str, object]) -> bool:
    return (
        _indicator(row.get("pass_attempt"), field="pass_attempt")
        or _indicator(row.get("rush_attempt"), field="rush_attempt")
        or _indicator(row.get("sack"), field="sack")
    ) and not _indicator(row.get("qb_kneel"), field="qb_kneel")


def _dropback(row: Mapping[str, object]) -> bool:
    return _indicator(row.get("pass_attempt"), field="pass_attempt") or _indicator(
        row.get("sack"), field="sack"
    )


def _team_metric_selectors(
    team: str,
) -> dict[str, MetricSelector]:
    def offense(row: Mapping[str, object]) -> bool:
        return row.get("posteam") == team

    def defense(row: Mapping[str, object]) -> bool:
        return row.get("defteam") == team

    return {
        "team_pass_epa_per_play": lambda row: (
            (_number(row.get("epa"), field="epa") or 0.0, 1.0)
            if offense(row)
            and _dropback(row)
            and _number(row.get("epa"), field="epa") is not None
            else None
        ),
        "team_rush_epa_per_play": lambda row: (
            (_number(row.get("epa"), field="epa") or 0.0, 1.0)
            if offense(row)
            and _indicator(row.get("rush_attempt"), field="rush_attempt")
            and not _indicator(row.get("qb_kneel"), field="qb_kneel")
            and _number(row.get("epa"), field="epa") is not None
            else None
        ),
        "offense_success_rate": lambda row: (
            (float(_indicator(row.get("success"), field="success")), 1.0)
            if offense(row) and _scrimmage(row)
            else None
        ),
        "defense_success_rate": lambda row: (
            (float(_indicator(row.get("success"), field="success")), 1.0)
            if defense(row) and _scrimmage(row)
            else None
        ),
        "explosive_rate": lambda row: (
            (
                float(
                    (
                        _indicator(row.get("pass_attempt"), field="pass_attempt")
                        and (_number(row.get("yards_gained"), field="yards_gained") or 0)
                        >= 20
                    )
                    or (
                        _indicator(row.get("rush_attempt"), field="rush_attempt")
                        and (_number(row.get("yards_gained"), field="yards_gained") or 0)
                        >= 10
                    )
                ),
                1.0,
            )
            if offense(row) and _scrimmage(row)
            else None
        ),
        "turnover_rate": lambda row: (
            (
                float(
                    _indicator(row.get("interception"), field="interception")
                    or _indicator(row.get("fumble_lost"), field="fumble_lost")
                ),
                1.0,
            )
            if offense(row) and _scrimmage(row)
            else None
        ),
        "sack_rate": lambda row: (
            (float(_indicator(row.get("sack"), field="sack")), 1.0)
            if offense(row) and _dropback(row)
            else None
        ),
        "fourth_down_aggressiveness": lambda row: (
            (
                float(
                    _scrimmage(row)
                    and not _indicator(
                        row.get("field_goal_attempt"), field="field_goal_attempt"
                    )
                ),
                1.0,
            )
            if offense(row)
            and _integer(row.get("down"), field="down") == 4
            else None
        ),
        "field_goal_make_rate": lambda row: (
            (float(row.get("field_goal_result") == "made"), 1.0)
            if offense(row)
            and _indicator(row.get("field_goal_attempt"), field="field_goal_attempt")
            else None
        ),
        "punt_gross_yards": lambda row: (
            (_number(row.get("kick_distance"), field="kick_distance") or 0.0, 1.0)
            if offense(row)
            and _indicator(row.get("punt_attempt"), field="punt_attempt")
            and _number(row.get("kick_distance"), field="kick_distance") is not None
            else None
        ),
        "return_yards_per_return": lambda row: (
            (_number(row.get("return_yards"), field="return_yards") or 0.0, 1.0)
            if row.get("return_team") == team
            and _number(row.get("return_yards"), field="return_yards") is not None
            else None
        ),
    }


def _red_zone_td_rate(
    rows: Sequence[Mapping[str, object]], team: str
) -> tuple[float | None, int]:
    drives: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("posteam") != team:
            continue
        drive = row.get("fixed_drive", row.get("drive"))
        if drive is None:
            continue
        game_id = _stable_id(row.get("game_id"), field="rolling game_id")
        drives[(game_id, _stable_id(drive, field="rolling drive"))].append(row)
    qualifying = [
        drive_rows
        for drive_rows in drives.values()
        if any(
            (yardline := _integer(row.get("yardline_100"), field="yardline_100"))
            is not None
            and yardline <= 20
            for row in drive_rows
        )
    ]
    if not qualifying:
        return None, 0
    touchdowns = sum(
        any(_indicator(row.get("touchdown"), field="touchdown") for row in drive_rows)
        for drive_rows in qualifying
    )
    return touchdowns / len(qualifying), len(qualifying)


def _player_metric_selectors(
    *,
    role: PlayerRoleV1,
    team: str,
) -> dict[str, MetricSelector]:
    player_id = role.player_id
    if player_id is None:
        return {}

    if role.role == "passer":
        return {
            "qb_epa_per_dropback": lambda row: (
                (_number(row.get("qb_epa"), field="qb_epa") or 0.0, 1.0)
                if row.get("passer_player_id") == player_id
                and _dropback(row)
                and _number(row.get("qb_epa"), field="qb_epa") is not None
                else None
            ),
            "qb_cpoe": lambda row: (
                (_number(row.get("cpoe"), field="cpoe") or 0.0, 1.0)
                if row.get("passer_player_id") == player_id
                and _number(row.get("cpoe"), field="cpoe") is not None
                else None
            ),
            "qb_sack_rate": lambda row: (
                (float(_indicator(row.get("sack"), field="sack")), 1.0)
                if row.get("passer_player_id") == player_id and _dropback(row)
                else None
            ),
        }
    if role.role == "rusher":
        return {
            "rusher_epa_per_attempt": lambda row: (
                (_number(row.get("epa"), field="epa") or 0.0, 1.0)
                if row.get("rusher_player_id") == player_id
                and _indicator(row.get("rush_attempt"), field="rush_attempt")
                and _number(row.get("epa"), field="epa") is not None
                else None
            ),
            "rusher_success_rate": lambda row: (
                (float(_indicator(row.get("success"), field="success")), 1.0)
                if row.get("rusher_player_id") == player_id
                and _indicator(row.get("rush_attempt"), field="rush_attempt")
                else None
            ),
        }
    if role.role == "receiver":
        return {
            "receiver_target_share": lambda row: (
                (float(row.get("receiver_player_id") == player_id), 1.0)
                if row.get("posteam") == team
                and _indicator(row.get("pass_attempt"), field="pass_attempt")
                else None
            ),
            "receiver_catch_rate": lambda row: (
                (float(_indicator(row.get("complete_pass"), field="complete_pass")), 1.0)
                if row.get("receiver_player_id") == player_id
                and _indicator(row.get("pass_attempt"), field="pass_attempt")
                else None
            ),
            "receiver_yards_per_target": lambda row: (
                (
                    _number(row.get("receiving_yards"), field="receiving_yards")
                    or 0.0,
                    1.0,
                )
                if row.get("receiver_player_id") == player_id
                and _indicator(row.get("pass_attempt"), field="pass_attempt")
                else None
            ),
        }
    if role.role == "punter":
        return {
            "punter_gross_yards": lambda row: (
                (_number(row.get("kick_distance"), field="kick_distance") or 0.0, 1.0)
                if row.get("punter_player_id") == player_id
                and _number(row.get("kick_distance"), field="kick_distance") is not None
                else None
            )
        }
    if role.role == "kicker":
        return {
            "kicker_field_goal_make_rate": lambda row: (
                (float(row.get("field_goal_result") == "made"), 1.0)
                if row.get("kicker_player_id") == player_id
                and _indicator(
                    row.get("field_goal_attempt"), field="field_goal_attempt"
                )
                else None
            )
        }
    if role.role in {"punt_returner", "kickoff_returner"}:
        id_field = (
            "punt_returner_player_id"
            if role.role == "punt_returner"
            else "kickoff_returner_player_id"
        )
        return {
            "returner_yards_per_return": lambda row: (
                (_number(row.get("return_yards"), field="return_yards") or 0.0, 1.0)
                if row.get(id_field) == player_id
                and _number(row.get("return_yards"), field="return_yards") is not None
                else None
            )
        }
    return {}


def _feature_pair(
    *,
    metric_id: str,
    entity_type: str,
    entity_id: str,
    season_rows: Sequence[Mapping[str, object]],
    trailing_rows: Sequence[Mapping[str, object]],
    selector: MetricSelector,
    low_support_observations: int,
) -> RollingFeatureV1:
    season_value, season_count = _ratio(season_rows, selector)
    trailing_value, trailing_count = _ratio(trailing_rows, selector)
    return RollingFeatureV1(
        metric_id=metric_id,
        entity_type=entity_type,  # type: ignore[arg-type]
        entity_id=entity_id,
        season_to_date_value=season_value,
        season_to_date_observations=season_count,
        trailing_8_value=trailing_value,
        trailing_8_observations=trailing_count,
        low_support=trailing_count < low_support_observations,
    )


def _rolling_features(
    *,
    history: _HistoryIndex,
    before: datetime,
    current_game_id: str,
    season: int,
    team: str,
    opponent_team: str,
    roles: Sequence[PlayerRoleV1],
    low_support_observations: int,
    team_cache: dict[
        tuple[str, datetime, int, str],
        tuple[RollingFeatureV1, ...],
    ],
    player_cache: dict[
        tuple[str, datetime, int, str, str, str],
        tuple[RollingFeatureV1, ...],
    ],
) -> tuple[RollingFeatureV1, ...]:
    features: list[RollingFeatureV1] = []
    for team_id in (team, opponent_team):
        team_cache_key = (current_game_id, before, season, team_id)
        cached_team = team_cache.get(team_cache_key)
        if cached_team is None:
            cached_team = _team_rolling_features(
                history=history,
                before=before,
                current_game_id=current_game_id,
                season=season,
                team=team_id,
                low_support_observations=low_support_observations,
            )
            team_cache[team_cache_key] = cached_team
        features.extend(cached_team)
    seen_players: set[tuple[str, str]] = set()
    for role in roles:
        if role.player_id is None:
            continue
        player_key = (role.role, role.player_id)
        if player_key in seen_players:
            continue
        seen_players.add(player_key)
        cache_key = (
            current_game_id,
            before,
            season,
            team,
            role.role,
            role.player_id,
        )
        cached_player = player_cache.get(cache_key)
        if cached_player is None:
            cached_player = _player_rolling_features(
                history=history,
                before=before,
                current_game_id=current_game_id,
                season=season,
                team=team,
                role=role,
                low_support_observations=low_support_observations,
            )
            player_cache[cache_key] = cached_player
        features.extend(cached_player)
    return tuple(
        sorted(
            features,
            key=lambda item: (item.entity_type, item.entity_id, item.metric_id),
        )
    )


def _team_rolling_features(
    *,
    history: _HistoryIndex,
    before: datetime,
    current_game_id: str,
    season: int,
    team: str,
    low_support_observations: int,
) -> tuple[RollingFeatureV1, ...]:
    team_season_games = history.completed_game_ids(
        before=before,
        current_game_id=current_game_id,
        entity_id=team,
        entity_type="team",
        season=season,
        last_games=None,
    )
    team_trailing_games = history.completed_game_ids(
        before=before,
        current_game_id=current_game_id,
        entity_id=team,
        entity_type="team",
        season=None,
        last_games=8,
    )
    season_rows = history.rows_for(team_season_games)
    trailing_rows = history.rows_for(team_trailing_games)
    features: list[RollingFeatureV1] = [
        _feature_pair(
            metric_id=metric_id,
            entity_type="team",
            entity_id=team,
            season_rows=season_rows,
            trailing_rows=trailing_rows,
            selector=selector,
            low_support_observations=low_support_observations,
        )
        for metric_id, selector in _team_metric_selectors(team).items()
    ]
    season_redzone, season_redzone_count = _red_zone_td_rate(season_rows, team)
    trailing_redzone, trailing_redzone_count = _red_zone_td_rate(
        trailing_rows, team
    )
    features.append(
        RollingFeatureV1(
            metric_id="red_zone_td_rate",
            entity_type="team",
            entity_id=team,
            season_to_date_value=season_redzone,
            season_to_date_observations=season_redzone_count,
            trailing_8_value=trailing_redzone,
            trailing_8_observations=trailing_redzone_count,
            low_support=trailing_redzone_count < low_support_observations,
        )
    )
    return tuple(
        sorted(
            features,
            key=lambda item: (item.entity_type, item.entity_id, item.metric_id),
        )
    )


def _player_rolling_features(
    *,
    history: _HistoryIndex,
    before: datetime,
    current_game_id: str,
    season: int,
    team: str,
    role: PlayerRoleV1,
    low_support_observations: int,
) -> tuple[RollingFeatureV1, ...]:
    if role.player_id is None:
        return ()
    selectors = _player_metric_selectors(role=role, team=team)
    if not selectors:
        return ()
    player_season_games = history.completed_game_ids(
        before=before,
        current_game_id=current_game_id,
        entity_id=role.player_id,
        entity_type="player",
        season=season,
        last_games=None,
    )
    player_trailing_games = history.completed_game_ids(
        before=before,
        current_game_id=current_game_id,
        entity_id=role.player_id,
        entity_type="player",
        season=None,
        last_games=8,
    )
    season_rows = history.rows_for(player_season_games)
    trailing_rows = history.rows_for(player_trailing_games)
    return tuple(
        sorted(
            (
                _feature_pair(
                    metric_id=metric_id,
                    entity_type="player",
                    entity_id=role.player_id,
                    season_rows=season_rows,
                    trailing_rows=trailing_rows,
                    selector=selector,
                    low_support_observations=low_support_observations,
                )
                for metric_id, selector in selectors.items()
            ),
            key=lambda item: (item.entity_type, item.entity_id, item.metric_id),
        )
    )


_REGISTERED_FACTOR_COLUMNS = frozenset(
    {
        "abs_score_margin_focal",
        "abs_cpoe",
        "abs_pass_oe",
        "abs_punt_execution_surprise",
        "abs_yac_surprise",
        "defensive_two_point_conv",
        "down_attempt_failed",
        "episode_is_score",
        "episode_is_turnover",
        "explosive_play",
        "extra_point_result",
        "field_goal_result",
        "first_down",
        "fourth_down_decision_regret",
        "fourth_down_failed",
        "fumble_lost",
        "interception",
        "kickoff_touchback",
        "late_close_state",
        "rolling_focal_offense_success_rate",
        "rolling_focal_pass_epa_per_play",
        "rolling_focal_rush_epa_per_play",
        "rolling_focal_sack_rate",
        "rolling_focal_turnover_rate",
        "rolling_offense_success_diff",
        "rolling_pass_epa_diff",
        "rolling_qb_cpoe",
        "rolling_qb_epa_per_dropback",
        "rolling_receiver_target_share",
        "rolling_rush_epa_diff",
        "rolling_rusher_epa_per_attempt",
        "pass",
        "pass_touchdown",
        "penalty",
        "pre_distance",
        "pre_down",
        "pre_quarter",
        "pre_red_zone",
        "punt",
        "punt_inside_twenty",
        "qb_kneel",
        "qb_scramble",
        "qb_spike",
        "replay_or_challenge",
        "replay_result",
        "return_touchdown",
        "rush",
        "rush_touchdown",
        "sack",
        "safety",
        "sequence_answer_score",
        "sequence_back_to_back_turnovers",
        "sequence_blocked_kick_return",
        "sequence_red_zone_empty_drive",
        "sequence_review_reversal",
        "sequence_td_try_finalized",
        "sequence_two_minute_drive",
        "success",
        "timeout",
        "two_point_conv_result",
        "yards_gained",
    }
)
_NATIVE_FACTOR_COLUMNS = frozenset(
    {
        "between_play_roster_change",
        "defensive_two_point_attempt",
        "defensive_two_point_conversion",
        "fumble_recovery_team",
        "game_seconds_remaining",
        "goal_to_go",
        "kickoff_out_of_bounds",
        "own_kickoff_recovery",
        "penalty_status",
        "play_type",
        "punt_blocked",
        "punt_downed",
        "punt_fair_catch",
        "punt_out_of_bounds",
        "quarter",
        "return_yards",
        "score_margin_focal",
        "touchback",
        "vegas_wp_focal",
        "yardline_100",
    }
)
MARKET_FACTOR_COLUMNS_V1 = (
    "pre_event_actual_trade",
    "staleness_seconds",
    "venue",
)
REGISTERED_FACTOR_COLUMNS_V1 = tuple(
    sorted(_REGISTERED_FACTOR_COLUMNS | _NATIVE_FACTOR_COLUMNS)
)


def _episode_index(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    by_play: dict[tuple[str, str], dict[str, Any]] = {}
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        episode = dict(source)
        game_id = _stable_id(episode.get("game_id"), field="episode game_id")
        episode_id = _stable_id(
            episode.get("episode_id"), field="episode episode_id"
        )
        play_ids = episode.get("play_ids")
        if not isinstance(play_ids, Sequence) or isinstance(
            play_ids, (str, bytes, bytearray)
        ) or not play_ids:
            raise SportsFeatureBuildError("episode play_ids must be nonempty")
        for play_value in play_ids:
            play_id = _stable_id(play_value, field="episode play_id")
            key = (game_id, play_id)
            if key in by_play:
                raise SportsFeatureBuildError(
                    "one play is assigned to multiple finalized episodes"
                )
            by_play[key] = episode
            members[episode_id].append(episode)
    return by_play, members


def _canonical_event_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = (
            _stable_id(row.get("game_id"), field="canonical game_id"),
            _stable_id(row.get("play_id"), field="canonical play_id"),
        )
        if key in result:
            raise SportsFeatureBuildError("canonical events contain a duplicate key")
        result[key] = row
    return result


def _between_play_roster_change(
    value: object,
) -> BetweenPlayRosterChangeV1 | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SportsFeatureBuildError(
            "between_play_roster_change must be an object or null"
        )
    expected_semantics = (
        "set difference between consecutive observed participation rows; "
        "timing is bounded between plays"
    )
    semantics = value.get("semantics")
    if semantics != expected_semantics:
        raise SportsFeatureBuildError(
            "between_play_roster_change has widened or unknown semantics"
        )
    return BetweenPlayRosterChangeV1(
        offense_entering_names=_string_tuple(
            value.get("offense_entering_names"),
            field="offense_entering_names",
        ),
        offense_leaving_names=_string_tuple(
            value.get("offense_leaving_names"),
            field="offense_leaving_names",
        ),
        defense_entering_names=_string_tuple(
            value.get("defense_entering_names"),
            field="defense_entering_names",
        ),
        defense_leaving_names=_string_tuple(
            value.get("defense_leaving_names"),
            field="defense_leaving_names",
        ),
        semantics=expected_semantics,
    )


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is str:
        values: Sequence[object] = [
            child.strip()
            for child in value.split(";")
            if child.strip()
        ]
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        values = value
    else:
        raise SportsFeatureBuildError(f"{field} must be an array")
    result = tuple(sorted({_text(child) for child in values} - {None}))
    return tuple(str(child) for child in result)


def _ordered_strings(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is str:
        values: Sequence[object] = [
            child.strip()
            for child in value.split(";")
            if child.strip()
        ]
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        values = value
    else:
        raise SportsFeatureBuildError(f"{field} must be an array")
    result: list[str] = []
    for child in values:
        parsed = _text(child)
        if parsed is not None:
            result.append(parsed)
    if len(set(result)) != len(result):
        raise SportsFeatureBuildError(f"{field} must contain unique values")
    return tuple(result)


ParticipantPair = tuple[str | None, str | None]


def _participation_containers(
    source: Mapping[str, object],
    canonical: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    result: list[Mapping[str, object]] = []
    for parent in (canonical, source):
        nested = parent.get("participation")
        if isinstance(nested, Mapping):
            result.append(nested)
        result.append(parent)
    return tuple(result)


def _first_participation_values(
    containers: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> tuple[str, ...]:
    for container in containers:
        for field in fields:
            values = _ordered_strings(container.get(field), field=field)
            if values:
                return values
    return ()


def _participant_pairs(
    source: Mapping[str, object],
    canonical: Mapping[str, object],
    *,
    side: str,
) -> tuple[ParticipantPair, ...]:
    containers = _participation_containers(source, canonical)
    id_fields = (
        f"{side}_players",
        f"{side}_player_ids",
        f"{side}_gsis_ids",
    )
    name_fields = (
        f"{side}_names",
        f"{side}_player_names",
    )
    for container in containers:
        ids = _first_participation_values((container,), id_fields)
        names = _first_participation_values((container,), name_fields)
        if ids and names:
            if len(ids) != len(names):
                raise SportsFeatureBuildError(
                    f"{side} participation IDs and names are misaligned"
                )
            return tuple(
                sorted(zip(ids, names, strict=True), key=lambda pair: pair[0])
            )
    ids = _first_participation_values(containers, id_fields)
    names = _first_participation_values(containers, name_fields)
    if ids and names:
        if len(ids) != len(names):
            raise SportsFeatureBuildError(
                f"{side} participation IDs and names are misaligned"
            )
        return tuple(
            sorted(zip(ids, names, strict=True), key=lambda pair: pair[0])
        )
    if ids:
        return tuple((player_id, None) for player_id in sorted(ids))
    if names:
        return tuple((None, player_name) for player_name in sorted(names))
    return ()


def _participation_roles(
    *,
    offense: Sequence[ParticipantPair],
    defense: Sequence[ParticipantPair],
) -> tuple[PlayerRoleV1, ...]:
    roles = [
        PlayerRoleV1(
            role=f"{side}_participant",
            player_id=player_id,
            player_name=player_name,
        )
        for side, pairs in (("offense", offense), ("defense", defense))
        for player_id, player_name in pairs
    ]
    return tuple(
        sorted(
            roles,
            key=lambda role: (
                role.role,
                role.player_id or "",
                role.player_name or "",
            ),
        )
    )


def _canonical_team(
    source: Mapping[str, object],
    canonical: Mapping[str, object],
    *,
    field: str,
    source_field: str,
) -> str | None:
    value = _text(canonical.get(field))
    if value is not None:
        return value
    state = canonical.get("pre_state")
    if isinstance(state, Mapping):
        value = _text(state.get(field))
        if value is not None:
            return value
    return _text(source.get(source_field))


def _unit_phase(row: Mapping[str, object]) -> str:
    if _indicator(row.get("kickoff_attempt"), field="kickoff_attempt"):
        return "kickoff"
    if _indicator(row.get("punt_attempt"), field="punt_attempt"):
        return "punt"
    if _indicator(row.get("field_goal_attempt"), field="field_goal_attempt"):
        return "field_goal"
    if (
        _indicator(row.get("extra_point_attempt"), field="extra_point_attempt")
        or _indicator(row.get("two_point_attempt"), field="two_point_attempt")
    ):
        return "try"
    return "scrimmage"


def _participant_identity(pair: ParticipantPair) -> str:
    player_id, player_name = pair
    if player_id is not None:
        return f"id:{player_id}"
    assert player_name is not None
    return f"name:{player_name}"


def _participant_display(pair: ParticipantPair) -> str:
    player_id, player_name = pair
    value = player_name or player_id
    assert value is not None
    return value


def _roster_delta(
    previous_offense: Sequence[ParticipantPair],
    previous_defense: Sequence[ParticipantPair],
    current_offense: Sequence[ParticipantPair],
    current_defense: Sequence[ParticipantPair],
) -> BetweenPlayRosterChangeV1 | None:
    if not all(
        (
            previous_offense,
            previous_defense,
            current_offense,
            current_defense,
        )
    ):
        return None

    def changes(
        previous: Sequence[ParticipantPair],
        current: Sequence[ParticipantPair],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        previous_by_id = {
            _participant_identity(pair): pair for pair in previous
        }
        current_by_id = {
            _participant_identity(pair): pair for pair in current
        }
        entering = tuple(
            sorted(
                _participant_display(current_by_id[key])
                for key in current_by_id.keys() - previous_by_id.keys()
            )
        )
        leaving = tuple(
            sorted(
                _participant_display(previous_by_id[key])
                for key in previous_by_id.keys() - current_by_id.keys()
            )
        )
        return entering, leaving

    offense_entering, offense_leaving = changes(
        previous_offense,
        current_offense,
    )
    defense_entering, defense_leaving = changes(
        previous_defense,
        current_defense,
    )
    if not any(
        (
            offense_entering,
            offense_leaving,
            defense_entering,
            defense_leaving,
        )
    ):
        return None
    return BetweenPlayRosterChangeV1(
        offense_entering_names=offense_entering,
        offense_leaving_names=offense_leaving,
        defense_entering_names=defense_entering,
        defense_leaving_names=defense_leaving,
    )


def _participation_transitions(
    ordered: Sequence[Mapping[str, object]],
    *,
    canonical_by_play: Mapping[tuple[str, str], Mapping[str, object]],
    dispositions: Mapping[tuple[str, str], SportsRowDispositionV2],
) -> dict[
    tuple[str, str],
    tuple[str | None, BetweenPlayRosterChangeV1 | None],
]:
    previous: dict[
        str,
        tuple[
            str | None,
            str | None,
            str,
            tuple[ParticipantPair, ...],
            tuple[ParticipantPair, ...],
        ],
    ] = {}
    result: dict[
        tuple[str, str],
        tuple[str | None, BetweenPlayRosterChangeV1 | None],
    ] = {}
    for source in ordered:
        game_id = _stable_id(source.get("game_id"), field="game_id")
        play_id = _stable_id(source.get("play_id"), field="play_id")
        disposition = dispositions[(game_id, play_id)]
        if not disposition.live_play_outcome_eligible:
            continue
        canonical = canonical_by_play.get((game_id, play_id), {})
        offense_team = _canonical_team(
            source,
            canonical,
            field="offense_team",
            source_field="posteam",
        )
        defense_team = _canonical_team(
            source,
            canonical,
            field="defense_team",
            source_field="defteam",
        )
        phase = _unit_phase(source)
        offense = _participant_pairs(source, canonical, side="offense")
        defense = _participant_pairs(source, canonical, side="defense")
        classification: str | None = None
        roster_change: BetweenPlayRosterChangeV1 | None = None
        prior = previous.get(game_id)
        if prior is not None:
            (
                prior_offense_team,
                prior_defense_team,
                prior_phase,
                prior_offense,
                prior_defense,
            ) = prior
            if (
                prior_offense_team != offense_team
                or prior_defense_team != defense_team
                or prior_phase != phase
            ):
                classification = "phase_transition"
            else:
                roster_change = _roster_delta(
                    prior_offense,
                    prior_defense,
                    offense,
                    defense,
                )
                if roster_change is not None:
                    classification = "roster_change"
        result[(game_id, play_id)] = (classification, roster_change)
        previous[game_id] = (
            offense_team,
            defense_team,
            phase,
            offense,
            defense,
        )
    return result


def _episode_state_team(
    episode: Mapping[str, object],
    canonical: Mapping[str, object],
) -> str | None:
    for parent in (canonical.get("pre_state"), episode.get("pre_state")):
        if isinstance(parent, Mapping):
            for field in ("possession_team", "offense_team"):
                if (team := _text(parent.get(field))) is not None:
                    return team
    return None


def _sequence_context(
    selected: Sequence[Mapping[str, object]],
    episodes_by_play: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, bool]]:
    by_game: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        by_game[_stable_id(row.get("game_id"), field="game_id")].append(row)
    result: dict[tuple[str, str], dict[str, bool]] = {}
    for game_id, game_rows in by_game.items():
        ordered = sorted(
            game_rows,
            key=lambda row: (
                _integer(
                    row.get("order_sequence", row.get("play_id")),
                    field="order_sequence",
                    required=True,
                ),
                _stable_id(row.get("play_id"), field="play_id"),
            ),
        )
        drive_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in ordered:
            drive = row.get("fixed_drive", row.get("drive"))
            if drive is not None:
                drive_rows[_stable_id(drive, field="drive")].append(row)
        empty_redzone_final: set[str] = set()
        for children in drive_rows.values():
            reached = any(
                (yardline := _integer(row.get("yardline_100"), field="yardline_100"))
                is not None
                and yardline <= 20
                for row in children
            )
            scored_td = any(
                _indicator(row.get("touchdown"), field="touchdown")
                for row in children
            )
            made_field_goal = any(
                row.get("field_goal_result") == "made"
                for row in children
            )
            if reached and not scored_td and not made_field_goal:
                empty_redzone_final.add(
                    _stable_id(children[-1].get("play_id"), field="play_id")
                )
        last_score_team: str | None = None
        previous_turnover = False
        for row in ordered:
            play_id = _stable_id(row.get("play_id"), field="play_id")
            episode = episodes_by_play[(game_id, play_id)]
            nullified = bool(episode.get("nullified")) or bool(
                episode.get("audit_only")
            ) or _indicator(row.get("play_deleted"), field="play_deleted")
            score = (
                _indicator(row.get("touchdown"), field="touchdown")
                or row.get("field_goal_result") == "made"
                or _indicator(row.get("safety"), field="safety")
            ) and not nullified
            score_team = _text(
                episode.get("scoring_team")
                or (
                    row.get("defteam")
                    if _indicator(row.get("safety"), field="safety")
                    else row.get("posteam")
                )
            )
            turnover = (
                _indicator(row.get("interception"), field="interception")
                or _indicator(row.get("fumble_lost"), field="fumble_lost")
                or _indicator(
                    row.get("fourth_down_failed"), field="fourth_down_failed"
                )
            ) and not nullified
            flags = _event_flags(row, nullified=nullified)
            result[(game_id, play_id)] = {
                "sequence_answer_score": bool(
                    score
                    and score_team is not None
                    and last_score_team is not None
                    and score_team != last_score_team
                ),
                "sequence_back_to_back_turnovers": bool(
                    turnover and previous_turnover
                ),
                "sequence_blocked_kick_return": bool(
                    ("blocked_punt" in flags or "field_goal_blocked" in flags)
                    and (
                        "return_touchdown" in flags
                        or _indicator(row.get("return_touchdown"), field="return_touchdown")
                        or (
                            _number(
                                row.get("return_yards"),
                                field="return_yards",
                            )
                            or 0
                        )
                        > 0
                    )
                ),
                "sequence_red_zone_empty_drive": play_id in empty_redzone_final,
                "sequence_review_reversal": (
                    row.get("replay_or_challenge_result") == "reversed"
                ),
                "sequence_td_try_finalized": False,
                "sequence_two_minute_drive": (
                    (
                        _integer(
                            row.get("half_seconds_remaining"),
                            field="half_seconds_remaining",
                        )
                        or 0
                    )
                    <= 120
                    and _scrimmage(row)
                ),
            }
            if score:
                last_score_team = score_team
                previous_turnover = False
            elif turnover:
                previous_turnover = True
        episode_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in ordered:
            play_id = _stable_id(row.get("play_id"), field="play_id")
            episode_id = _stable_id(
                episodes_by_play[(game_id, play_id)].get("episode_id"),
                field="episode_id",
            )
            episode_rows[episode_id].append(row)
        for children in episode_rows.values():
            has_td = any(
                _indicator(row.get("touchdown"), field="touchdown")
                for row in children
            )
            has_try = any(
                _indicator(row.get("extra_point_attempt"), field="extra_point_attempt")
                or _indicator(row.get("two_point_attempt"), field="two_point_attempt")
                for row in children
            )
            if has_td and has_try:
                for row in children:
                    key = (game_id, _stable_id(row.get("play_id"), field="play_id"))
                    result[key]["sequence_td_try_finalized"] = True
            has_reversal = any(
                row.get("replay_or_challenge_result") == "reversed"
                for row in children
            )
            if has_reversal:
                for row in children:
                    key = (
                        game_id,
                        _stable_id(row.get("play_id"), field="play_id"),
                    )
                    result[key]["sequence_review_reversal"] = True
    return result


def _game_scores(
    row: Mapping[str, object],
    episode: Mapping[str, object],
    canonical: Mapping[str, object],
) -> tuple[int, int]:
    home_team = _text(row.get("home_team"))
    away_team = _text(row.get("away_team"))
    if home_team is None or away_team is None or home_team == away_team:
        raise SportsFeatureBuildError("game team identity is missing")
    # nflverse total_home_score/total_away_score are post-play values.  The
    # factor contract is pre-event, so orient the possession/defense scores
    # that are frozen immediately before the snap instead.
    posteam = _text(row.get("posteam"))
    post_score = _integer(row.get("posteam_score"), field="posteam_score")
    def_score = _integer(row.get("defteam_score"), field="defteam_score")
    if posteam is not None and post_score is not None and def_score is not None:
        return (
            (post_score, def_score)
            if posteam == home_team
            else (def_score, post_score)
        )
    for state in (canonical.get("pre_state"), episode.get("pre_state")):
        if not isinstance(state, Mapping):
            continue
        if (
            _text(state.get("home_team")) != home_team
            or _text(state.get("away_team")) != away_team
        ):
            continue
        home_score = _integer(state.get("home_score"), field="home_score")
        away_score = _integer(state.get("away_score"), field="away_score")
        if home_score is not None and away_score is not None:
            return home_score, away_score
    raise SportsFeatureBuildError("pre-play game score is unavailable")


def _factor_values(
    *,
    row: Mapping[str, object],
    episode: Mapping[str, object],
    sequence: Mapping[str, bool],
    quarter: int,
    game_seconds_remaining: int | None,
    distance: int | None,
    down: int | None,
    yardline_100: int | None,
    score_margin_focal: int,
    yards_gained: float | None,
    cpoe: float | None,
    pass_oe: float | None,
    yac_surprise: float | None,
    nullified: bool,
    rolling: Sequence[RollingFeatureV1],
    focal_team: str,
    defense_team: str,
) -> tuple[tuple[str, str | int | float | bool | None], ...]:
    trailing = {
        (item.entity_type, item.entity_id, item.metric_id):
        item.trailing_8_value
        for item in rolling
        if not item.low_support
    }
    focal_metric = lambda metric: trailing.get(("team", focal_team, metric))
    defense_metric = lambda metric: trailing.get(
        ("team", defense_team, metric)
    )

    def difference(metric: str) -> float | None:
        focal = focal_metric(metric)
        defense = defense_metric(metric)
        return (
            None
            if focal is None or defense is None
            else float(focal) - float(defense)
        )

    player_metric = lambda metric: next(
        (
            item.trailing_8_value
            for item in rolling
            if item.entity_type == "player"
            and item.metric_id == metric
            and item.trailing_8_value is not None
            and not item.low_support
        ),
        None,
    )
    touchdown = _indicator(row.get("touchdown"), field="touchdown")
    field_goal_made = row.get("field_goal_result") == "made"
    safety = _indicator(row.get("safety"), field="safety")
    turnover = (
        _indicator(row.get("interception"), field="interception")
        or _indicator(row.get("fumble_lost"), field="fumble_lost")
        or _indicator(row.get("fourth_down_failed"), field="fourth_down_failed")
        or bool(episode.get("turnover"))
    ) and not nullified
    score = (
        touchdown
        or field_goal_made
        or safety
        or (_integer(episode.get("score_points"), field="episode score_points") or 0)
        > 0
    ) and not nullified
    pass_play = _indicator(row.get("pass_attempt"), field="pass_attempt")
    rush_play = _indicator(row.get("rush_attempt"), field="rush_attempt")
    first_down = _indicator(row.get("first_down"), field="first_down")
    success = _indicator(row.get("success"), field="success")
    red_zone = yardline_100 is not None and yardline_100 <= 20
    explosive = bool(
        (pass_play and (yards_gained or 0) >= 20)
        or (rush_play and (yards_gained or 0) >= 10)
    )
    values: dict[str, str | int | float | bool | None] = {
        "abs_cpoe": None if cpoe is None else abs(cpoe),
        "abs_pass_oe": None if pass_oe is None else abs(pass_oe),
        "abs_punt_execution_surprise": None,
        "abs_score_margin_focal": abs(score_margin_focal),
        "abs_yac_surprise": (
            None if yac_surprise is None else abs(yac_surprise)
        ),
        "defensive_two_point_conv": _indicator(
            row.get("defensive_two_point_conv"),
            field="defensive_two_point_conv",
        ),
        "down_attempt_failed": bool(
            down in {3, 4}
            and (pass_play or rush_play or _indicator(row.get("sack"), field="sack"))
            and not first_down
            and not score
            and not nullified
        ),
        "episode_is_score": score,
        "episode_is_turnover": turnover,
        "explosive_play": explosive,
        "extra_point_result": _text(row.get("extra_point_result")),
        "field_goal_result": _text(row.get("field_goal_result")),
        "first_down": first_down,
        "fourth_down_decision_regret": None,
        "fourth_down_failed": (
            _indicator(row.get("fourth_down_failed"), field="fourth_down_failed")
            and not nullified
        ),
        "fumble_lost": (
            _indicator(row.get("fumble_lost"), field="fumble_lost")
            and not nullified
        ),
        "interception": (
            _indicator(row.get("interception"), field="interception")
            and not nullified
        ),
        "kickoff_touchback": bool(
            _indicator(row.get("kickoff_attempt"), field="kickoff_attempt")
            and _indicator(row.get("touchback"), field="touchback")
        ),
        "late_close_state": bool(
            quarter >= 4
            and game_seconds_remaining is not None
            and game_seconds_remaining <= 300
            and abs(score_margin_focal) <= 8
        ),
        "rolling_focal_offense_success_rate": focal_metric(
            "offense_success_rate"
        ),
        "rolling_focal_pass_epa_per_play": focal_metric(
            "team_pass_epa_per_play"
        ),
        "rolling_focal_rush_epa_per_play": focal_metric(
            "team_rush_epa_per_play"
        ),
        "rolling_focal_sack_rate": focal_metric("sack_rate"),
        "rolling_focal_turnover_rate": focal_metric("turnover_rate"),
        "rolling_offense_success_diff": difference("offense_success_rate"),
        "rolling_pass_epa_diff": difference("team_pass_epa_per_play"),
        "rolling_qb_cpoe": player_metric("qb_cpoe"),
        "rolling_qb_epa_per_dropback": player_metric(
            "qb_epa_per_dropback"
        ),
        "rolling_receiver_target_share": player_metric(
            "receiver_target_share"
        ),
        "rolling_rush_epa_diff": difference("team_rush_epa_per_play"),
        "rolling_rusher_epa_per_attempt": player_metric(
            "rusher_epa_per_attempt"
        ),
        "pass": pass_play,
        "pass_touchdown": (
            _indicator(row.get("pass_touchdown"), field="pass_touchdown")
            and not nullified
        ),
        "penalty": _indicator(row.get("penalty"), field="penalty"),
        "pre_distance": distance,
        "pre_down": down,
        "pre_quarter": quarter,
        "pre_red_zone": red_zone,
        "punt": _indicator(row.get("punt_attempt"), field="punt_attempt"),
        "punt_inside_twenty": _indicator(
            row.get("punt_inside_twenty"), field="punt_inside_twenty"
        ),
        "qb_kneel": _indicator(row.get("qb_kneel"), field="qb_kneel"),
        "qb_scramble": _indicator(row.get("qb_scramble"), field="qb_scramble"),
        "qb_spike": _indicator(row.get("qb_spike"), field="qb_spike"),
        "replay_or_challenge": _indicator(
            row.get("replay_or_challenge"), field="replay_or_challenge"
        ),
        "replay_result": _text(row.get("replay_or_challenge_result")),
        "return_touchdown": (
            _indicator(row.get("return_touchdown"), field="return_touchdown")
            and not nullified
        ),
        "rush": rush_play,
        "rush_touchdown": (
            _indicator(row.get("rush_touchdown"), field="rush_touchdown")
            and not nullified
        ),
        "sack": _indicator(row.get("sack"), field="sack"),
        "safety": safety and not nullified,
        "success": success and not nullified,
        "timeout": _indicator(row.get("timeout"), field="timeout"),
        "two_point_conv_result": _text(row.get("two_point_conv_result")),
        "yards_gained": yards_gained,
        "fourth_down_decision_regret_availability": (
            "blocked_missing_nfl4th_offline_asset"
        ),
        "punt_execution_surprise_availability": (
            "blocked_missing_nfl4th_punt_wp_asset"
        ),
    }
    values.update(sequence)
    missing = _REGISTERED_FACTOR_COLUMNS - values.keys()
    if missing:
        raise SportsFeatureBuildError(
            "registered factor columns are missing: " + ", ".join(sorted(missing))
        )
    return tuple(sorted(values.items()))


def _binding_field(binding: object, field: str) -> object:
    if isinstance(binding, Mapping):
        return binding.get(field)
    return getattr(binding, field, None)


def _binding_rows(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        rows: list[object] = []
        for game_id, binding in value.items():
            if isinstance(binding, Mapping):
                rows.append(
                    {
                        **binding,
                        "native_game_id": binding.get(
                            "native_game_id",
                            game_id,
                        ),
                    }
                )
            elif type(binding) is str or isinstance(binding, datetime):
                rows.append(
                    {
                        "native_game_id": game_id,
                        "scheduled_at_utc": binding,
                    }
                )
            else:
                rows.append(binding)
        return rows
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return list(value)
    raise SportsFeatureBuildError(
        "frozen_game_bindings must be a record sequence or mapping"
    )


def _binding_cutoffs(
    *,
    target_games: set[str],
    kickoff_times_utc: Mapping[str, str] | None,
    frozen_game_bindings: object,
) -> tuple[
    dict[str, datetime],
    dict[str, tuple[str, str, str]],
]:
    bindings = _binding_rows(frozen_game_bindings)
    scheduled_by_game: dict[str, datetime | None] = {}
    if bindings:
        for binding in bindings:
            game_id = _stable_id(
                _binding_field(binding, "native_game_id"),
                field="frozen binding native_game_id",
            )
            if game_id in scheduled_by_game:
                raise SportsFeatureBuildError(
                    "frozen_game_bindings contains a duplicate game"
                )
            scheduled = _binding_field(binding, "scheduled_at_utc")
            if scheduled is None:
                scheduled_by_game[game_id] = None
            elif isinstance(scheduled, datetime):
                if scheduled.tzinfo is None:
                    raise SportsFeatureBuildError(
                        "frozen binding scheduled_at_utc must be timezone-aware"
                    )
                scheduled_by_game[game_id] = scheduled.astimezone(timezone.utc)
            else:
                scheduled_by_game[game_id] = _parse_utc(
                    scheduled,
                    field=f"frozen binding scheduled_at_utc {game_id}",
                )
        if set(scheduled_by_game) != target_games:
            raise SportsFeatureBuildError(
                "frozen_game_bindings must exactly cover selected games"
            )
    provided = {} if kickoff_times_utc is None else dict(kickoff_times_utc)
    if provided and set(provided) != target_games:
        raise SportsFeatureBuildError(
            "kickoff_times_utc must exactly cover selected games"
        )
    cutoffs: dict[str, datetime] = {}
    semantics: dict[str, tuple[str, str, str]] = {}
    for game_id in sorted(target_games):
        scheduled = scheduled_by_game.get(game_id)
        if scheduled is not None:
            cutoffs[game_id] = scheduled
            semantics[game_id] = (
                "scheduled_kickoff",
                "scheduled_kickoff_available",
                "GREEN",
            )
            continue
        fallback = provided.get(game_id)
        if fallback is None:
            if bindings:
                raise SportsFeatureBuildError(
                    "scheduled kickoff unavailable and no explicit "
                    f"first-live-source cutoff was provided for {game_id}"
                )
            raise SportsFeatureBuildError(
                "kickoff_times_utc must exactly cover selected games"
            )
        cutoffs[game_id] = _parse_utc(
            fallback,
            field=f"cutoff {game_id}",
        )
        semantics[game_id] = (
            (
                "first_live_source_time"
                if bindings
                else "provided_cutoff_not_proven_scheduled"
            ),
            (
                "scheduled_kickoff_unavailable"
                if bindings
                else "scheduled_kickoff_not_bound"
            ),
            "DATA_GAP",
        )
    return cutoffs, semantics


def build_sports_feature_view(
    *,
    selected_pbp_rows: Sequence[Mapping[str, object]],
    finalized_episode_rows: Sequence[Mapping[str, object]],
    historical_pbp_rows: Sequence[Mapping[str, object]],
    kickoff_times_utc: Mapping[str, str] | None = None,
    source_hashes: Mapping[str, str],
    canonical_event_rows: Sequence[Mapping[str, object]] = (),
    frozen_game_bindings: object = (),
    row_disposition_ledger: SportsRowDispositionLedgerV2 | None = None,
    low_support_observations: int = 8,
) -> SportsFeatureViewV1:
    """Build a deterministic PIT view without rescanning market associations.

    ``historical_pbp_rows`` must come from byte-verified 2022--2025 nflverse
    partitions.  A completed game enters a target row only when its last
    source timestamp is strictly earlier than the target kickoff.
    """

    if (
        isinstance(low_support_observations, bool)
        or not isinstance(low_support_observations, int)
        or low_support_observations <= 0
    ):
        raise SportsFeatureBuildError(
            "low_support_observations must be a positive integer"
        )
    selected = _records(selected_pbp_rows, name="selected_pbp_rows")
    if not selected:
        raise SportsFeatureBuildError("selected_pbp_rows must not be empty")
    episode_rows = _records(
        finalized_episode_rows, name="finalized_episode_rows"
    )
    history_rows = _records(
        historical_pbp_rows, name="historical_pbp_rows"
    )
    canonical_rows = _records(canonical_event_rows, name="canonical_event_rows")
    try:
        expected_ledger = build_sports_row_disposition_ledger(
            selected_pbp_rows=selected,
            finalized_episode_rows=episode_rows,
            source_hashes=source_hashes,
            canonical_event_rows=canonical_rows,
        )
    except SportsRowCleaningError as error:
        raise SportsFeatureBuildError(str(error)) from error
    if row_disposition_ledger is None:
        disposition_ledger = expected_ledger
    else:
        if not isinstance(
            row_disposition_ledger,
            SportsRowDispositionLedgerV2,
        ):
            raise SportsFeatureBuildError(
                "row_disposition_ledger must be SportsRowDispositionLedgerV2"
            )
        if row_disposition_ledger.rows_sha256 != expected_ledger.rows_sha256:
            raise SportsFeatureBuildError(
                "row_disposition_ledger does not match selected semantic rows"
            )
        disposition_ledger = row_disposition_ledger
    dispositions_by_play: dict[
        tuple[str, str],
        SportsRowDispositionV2,
    ] = {}
    for disposition in disposition_ledger.rows:
        key = (disposition.game_id, disposition.raw_play_id)
        if key in dispositions_by_play:
            raise SportsFeatureBuildError(
                "typed feature view cannot represent duplicate raw play keys"
            )
        dispositions_by_play[key] = disposition
    episodes_by_play, _episode_members = _episode_index(episode_rows)
    canonical_by_play = _canonical_event_index(canonical_rows)
    selected_keys = {
        (
            _stable_id(row.get("game_id"), field="game_id"),
            _stable_id(row.get("play_id"), field="play_id"),
        )
        for row in selected
    }
    missing_episodes = selected_keys - episodes_by_play.keys()
    if missing_episodes:
        raise SportsFeatureBuildError(
            "selected play is missing its finalized episode"
        )
    extra_episodes = episodes_by_play.keys() - selected_keys
    if extra_episodes:
        raise SportsFeatureBuildError(
            "finalized episode references a play outside selected PBP"
        )
    target_games = {game_id for game_id, _play_id in selected_keys}
    kickoff_by_game, cutoff_semantics_by_game = _binding_cutoffs(
        target_games=target_games,
        kickoff_times_utc=kickoff_times_utc,
        frozen_game_bindings=frozen_game_bindings,
    )
    history = _HistoryIndex(history_rows)
    sequences = _sequence_context(selected, episodes_by_play)

    drive_state: dict[tuple[str, str], tuple[int, float]] = {}
    team_rolling_cache: dict[
        tuple[str, datetime, int, str],
        tuple[RollingFeatureV1, ...],
    ] = {}
    player_rolling_cache: dict[
        tuple[str, datetime, int, str, str, str],
        tuple[RollingFeatureV1, ...],
    ] = {}
    history_audit_cache: dict[
        tuple[str, datetime, tuple[str, ...]],
        dict[str, str | int],
    ] = {}
    output: list[SportsFeatureRowV1] = []
    ordered = sorted(
        selected,
        key=lambda row: (
            _stable_id(row.get("game_id"), field="game_id"),
            _integer(
                row.get("order_sequence", row.get("play_id")),
                field="order_sequence",
                required=True,
            ),
            _stable_id(row.get("play_id"), field="play_id"),
        ),
    )
    participation_transitions = _participation_transitions(
        ordered,
        canonical_by_play=canonical_by_play,
        dispositions=dispositions_by_play,
    )
    for source in ordered:
        game_id = _stable_id(source.get("game_id"), field="game_id")
        play_id = _stable_id(source.get("play_id"), field="play_id")
        disposition = dispositions_by_play[(game_id, play_id)]
        if not disposition.factor_eligible:
            continue
        episode = episodes_by_play[(game_id, play_id)]
        canonical = canonical_by_play.get((game_id, play_id), {})
        home_team = _text(source.get("home_team"))
        away_team = _text(source.get("away_team"))
        if home_team is None or away_team is None or home_team == away_team:
            raise SportsFeatureBuildError("selected game team identity is invalid")
        focal_team = (
            _text(source.get("posteam"))
            or _episode_state_team(episode, canonical)
            or _text(canonical.get("timeout_team", source.get("timeout_team")))
            or _text(canonical.get("penalty_team", source.get("penalty_team")))
        )
        if focal_team is None:
            # Game-start/end administrative rows have no meaningful focal
            # orientation and remain in the canonical replay, not this factor view.
            continue
        if focal_team not in {home_team, away_team}:
            raise SportsFeatureBuildError("focal team is not a selected game team")
        defense_team = _text(source.get("defteam"))
        if defense_team is None:
            defense_team = away_team if focal_team == home_team else home_team
        season = _integer(source.get("season"), field="season", required=True)
        week = _integer(source.get("week"), field="week", required=True)
        quarter = _integer(source.get("qtr"), field="qtr", required=True)
        assert season is not None and week is not None and quarter is not None
        home_score, away_score = _game_scores(source, episode, canonical)
        score_margin_focal = (
            home_score - away_score
            if focal_team == home_team
            else away_score - home_score
        )
        distance = _integer(source.get("ydstogo"), field="ydstogo")
        down = _integer(source.get("down"), field="down")
        yardline = _integer(source.get("yardline_100"), field="yardline_100")
        game_seconds = _integer(
            source.get("game_seconds_remaining"),
            field="game_seconds_remaining",
        )
        drive_value = source.get("fixed_drive", source.get("drive"))
        drive_id = (
            None
            if drive_value is None
            else _stable_id(drive_value, field="drive")
        )
        drive_key = None if drive_id is None else (game_id, drive_id)
        prior_drive_count, prior_drive_yards = (
            (0, 0.0) if drive_key is None else drive_state.get(drive_key, (0, 0.0))
        )
        nullified = bool(episode.get("nullified")) or bool(
            episode.get("audit_only")
        ) or _indicator(source.get("play_deleted"), field="play_deleted")
        offense_participants = _participant_pairs(
            source,
            canonical,
            side="offense",
        )
        defense_participants = _participant_pairs(
            source,
            canonical,
            side="defense",
        )
        roles = tuple(
            sorted(
                (
                    *_player_roles(source),
                    *_participation_roles(
                        offense=offense_participants,
                        defense=defense_participants,
                    ),
                ),
                key=lambda role: (
                    role.role,
                    role.player_id or "",
                    role.player_name or "",
                ),
            )
        )
        rolling = _rolling_features(
            history=history,
            before=kickoff_by_game[game_id],
            current_game_id=game_id,
            season=season,
            team=focal_team,
            opponent_team=defense_team,
            roles=roles,
            low_support_observations=low_support_observations,
            team_cache=team_rolling_cache,
            player_cache=player_rolling_cache,
        )
        yac_epa = _number(source.get("yac_epa"), field="yac_epa")
        xyac_epa = _number(source.get("xyac_epa"), field="xyac_epa")
        yac_surprise = (
            None
            if yac_epa is None or xyac_epa is None
            else yac_epa - xyac_epa
        )
        cpoe = _number(source.get("cpoe"), field="cpoe")
        pass_oe = _number(source.get("pass_oe"), field="pass_oe")
        yards_gained = _number(source.get("yards_gained"), field="yards_gained")
        vegas_home_wpa = _number(
            source.get("vegas_home_wpa"), field="vegas_home_wpa"
        )
        realized_delta = (
            None
            if vegas_home_wpa is None
            else (
                vegas_home_wpa
                if focal_team == home_team
                else -vegas_home_wpa
            )
        )
        source_time = canonical.get(
            "source_time_start_utc",
            source.get("source_time_start_utc", source.get("time_of_day")),
        )
        source_time_text = (
            None
            if source_time is None
            else _utc_text(_parse_utc(source_time, field="time_of_day"))
        )
        formation = _text(
            canonical.get("formation", source.get("offense_formation"))
        )
        offense_personnel = _text(
            canonical.get("offense_personnel", source.get("offense_personnel"))
        )
        defense_personnel = _text(
            canonical.get("defense_personnel", source.get("defense_personnel"))
        )
        base_factor_values = _factor_values(
            row=source,
            episode=episode,
            sequence=sequences[(game_id, play_id)],
            quarter=quarter,
            game_seconds_remaining=game_seconds,
            distance=distance,
            down=down,
            yardline_100=yardline,
            score_margin_focal=score_margin_focal,
            yards_gained=yards_gained,
            cpoe=cpoe,
            pass_oe=pass_oe,
            yac_surprise=yac_surprise,
            nullified=nullified,
            rolling=rolling,
            focal_team=focal_team,
            defense_team=defense_team,
        )
        history_audit_key = (
            game_id,
            kickoff_by_game[game_id],
            tuple(sorted({focal_team, defense_team})),
        )
        history_audit = history_audit_cache.get(history_audit_key)
        if history_audit is None:
            history_audit = history.audit(
                before=kickoff_by_game[game_id],
                current_game_id=game_id,
                entity_ids=history_audit_key[2],
            )
            history_audit_cache[history_audit_key] = history_audit
        (
            cutoff_semantics,
            scheduled_kickoff_status,
            sports_data_status,
        ) = cutoff_semantics_by_game[game_id]
        (
            participation_transition,
            between_play_roster_change,
        ) = participation_transitions.get((game_id, play_id), (None, None))
        factor_values = tuple(
            sorted(
                (
                    *base_factor_values,
                    (
                        "live_play_outcome_eligible",
                        disposition.live_play_outcome_eligible,
                    ),
                    (
                        "administrative_event_eligible",
                        disposition.administrative_event_eligible,
                    ),
                    ("factor_eligible", disposition.factor_eligible),
                    (
                        "history_cutoff_semantics",
                        cutoff_semantics,
                    ),
                    (
                        "scheduled_kickoff_status",
                        scheduled_kickoff_status,
                    ),
                    ("sports_data_status", sports_data_status),
                    (
                        "participation_transition",
                        participation_transition,
                    ),
                    *tuple(history_audit.items()),
                ),
                key=lambda item: item[0],
            )
        )
        output.append(
            SportsFeatureRowV1(
                game_id=game_id,
                play_id=play_id,
                episode_id=_stable_id(
                    episode.get("episode_id"), field="episode_id"
                ),
                order_sequence=int(
                    _integer(
                        source.get("order_sequence", source.get("play_id")),
                        field="order_sequence",
                        required=True,
                    )
                ),
                season=season,
                season_type=_text(source.get("season_type")) or "REG",
                week=week,
                kickoff_at_utc=_utc_text(kickoff_by_game[game_id]),
                source_time_utc=source_time_text,
                description=_text(source.get("desc")),
                home_team=home_team,
                away_team=away_team,
                focal_team=focal_team,
                possession_team=focal_team,
                offense_team=focal_team,
                defense_team=defense_team,
                quarter=quarter,
                quarter_seconds_remaining=_integer(
                    source.get("quarter_seconds_remaining"),
                    field="quarter_seconds_remaining",
                ),
                half_seconds_remaining=_integer(
                    source.get("half_seconds_remaining"),
                    field="half_seconds_remaining",
                ),
                game_seconds_remaining=game_seconds,
                home_score=home_score,
                away_score=away_score,
                score_margin_focal=score_margin_focal,
                tied=score_margin_focal == 0,
                one_score_game=abs(score_margin_focal) <= 8,
                multi_score_game=abs(score_margin_focal) > 8,
                down=down,
                distance=distance,
                yardline_100=yardline,
                field_zone=_field_zone(yardline),
                red_zone=(None if yardline is None else yardline <= 20),
                goal_to_go=_optional_indicator(
                    source.get("goal_to_go"), field="goal_to_go"
                ),
                drive_id=drive_id,
                drive_play_index=(
                    None if drive_id is None else prior_drive_count + 1
                ),
                drive_yards_before=(
                    None if drive_id is None else prior_drive_yards
                ),
                drive_progression_yards_before=(
                    None if drive_id is None else prior_drive_yards
                ),
                home_timeouts_remaining=_nonnegative_source_integer(
                    source.get("home_timeouts_remaining"),
                    field="home_timeouts_remaining",
                ),
                away_timeouts_remaining=_nonnegative_source_integer(
                    source.get("away_timeouts_remaining"),
                    field="away_timeouts_remaining",
                ),
                play_type=_text(source.get("play_type")),
                event_flags=_event_flags(source, nullified=nullified),
                event_eligible=disposition.factor_eligible,
                episode_nullified=bool(episode.get("nullified")),
                episode_audit_only=bool(episode.get("audit_only")),
                player_roles=roles,
                formation=formation,
                offense_personnel=offense_personnel,
                defense_personnel=defense_personnel,
                offense_player_ids=tuple(
                    sorted(
                        player_id
                        for player_id, _player_name in offense_participants
                        if player_id is not None
                    )
                ),
                defense_player_ids=tuple(
                    sorted(
                        player_id
                        for player_id, _player_name in defense_participants
                        if player_id is not None
                    )
                ),
                offense_player_names=tuple(
                    sorted(
                        player_name
                        for _player_id, player_name in offense_participants
                        if player_name is not None
                    )
                ),
                defense_player_names=tuple(
                    sorted(
                        player_name
                        for _player_id, player_name in defense_participants
                        if player_name is not None
                    )
                ),
                shotgun=_optional_indicator(
                    source.get("shotgun"), field="shotgun"
                ),
                no_huddle=_optional_indicator(
                    source.get("no_huddle"), field="no_huddle"
                ),
                yards_gained=yards_gained,
                wp_focal=_number(source.get("wp"), field="wp"),
                vegas_wp_focal=_number(
                    source.get("vegas_wp"), field="vegas_wp"
                ),
                wpa_focal=_number(source.get("wpa"), field="wpa"),
                realized_delta_wp_focal=realized_delta,
                epa=_number(source.get("epa"), field="epa"),
                qb_epa=_number(source.get("qb_epa"), field="qb_epa"),
                xpass=_number(source.get("xpass"), field="xpass"),
                pass_oe=pass_oe,
                cp=_number(source.get("cp"), field="cp"),
                cpoe=cpoe,
                yac_epa=yac_epa,
                xyac_epa=xyac_epa,
                yac_surprise=yac_surprise,
                rolling_features=rolling,
                factor_values=factor_values,
                defensive_two_point_attempt=_optional_indicator(
                    source.get("defensive_two_point_attempt"),
                    field="defensive_two_point_attempt",
                ),
                defensive_two_point_conversion=_optional_indicator(
                    source.get("defensive_two_point_conv"),
                    field="defensive_two_point_conv",
                ),
                kicker_player_id=_text(source.get("kicker_player_id")),
                kicker_player_name=_text(source.get("kicker_player_name")),
                penalty_team=_text(
                    canonical.get("penalty_team", source.get("penalty_team"))
                ),
                penalty_type=_text(
                    canonical.get("penalty_type", source.get("penalty_type"))
                ),
                penalty_yards=_number(
                    canonical.get(
                        "penalty_yards", source.get("penalty_yards")
                    ),
                    field="penalty_yards",
                ),
                penalty_status=_text(canonical.get("penalty_status")),
                timeout_team=_text(
                    canonical.get("timeout_team", source.get("timeout_team"))
                ),
                punt_blocked=_optional_indicator(
                    source.get("punt_blocked"), field="punt_blocked"
                ),
                punt_inside_twenty=_optional_indicator(
                    source.get("punt_inside_twenty"),
                    field="punt_inside_twenty",
                ),
                punt_in_endzone=_optional_indicator(
                    source.get("punt_in_endzone"), field="punt_in_endzone"
                ),
                punt_out_of_bounds=_optional_indicator(
                    source.get("punt_out_of_bounds"),
                    field="punt_out_of_bounds",
                ),
                punt_downed=_optional_indicator(
                    source.get("punt_downed"), field="punt_downed"
                ),
                punt_fair_catch=_optional_indicator(
                    source.get("punt_fair_catch"),
                    field="punt_fair_catch",
                ),
                kickoff_inside_twenty=_optional_indicator(
                    source.get("kickoff_inside_twenty"),
                    field="kickoff_inside_twenty",
                ),
                kickoff_in_endzone=_optional_indicator(
                    source.get("kickoff_in_endzone"),
                    field="kickoff_in_endzone",
                ),
                kickoff_out_of_bounds=_optional_indicator(
                    source.get("kickoff_out_of_bounds"),
                    field="kickoff_out_of_bounds",
                ),
                kickoff_downed=_optional_indicator(
                    source.get("kickoff_downed"), field="kickoff_downed"
                ),
                kickoff_fair_catch=_optional_indicator(
                    source.get("kickoff_fair_catch"),
                    field="kickoff_fair_catch",
                ),
                own_kickoff_recovery=_optional_indicator(
                    source.get("own_kickoff_recovery"),
                    field="own_kickoff_recovery",
                ),
                own_kickoff_recovery_touchdown=_optional_indicator(
                    source.get("own_kickoff_recovery_td"),
                    field="own_kickoff_recovery_td",
                ),
                touchback=_optional_indicator(
                    source.get("touchback"), field="touchback"
                ),
                return_team=_text(source.get("return_team")),
                return_yards=_number(
                    source.get("return_yards"), field="return_yards"
                ),
                fumble_recovery_team=_text(
                    source.get("fumble_recovery_1_team")
                ),
                fumble_recovery_yards=_number(
                    source.get("fumble_recovery_1_yards"),
                    field="fumble_recovery_1_yards",
                ),
                between_play_roster_change=between_play_roster_change,
            )
        )
        if drive_key is not None and not nullified:
            drive_state[drive_key] = (
                prior_drive_count + 1,
                prior_drive_yards + (yards_gained or 0.0),
            )
    if not output:
        raise SportsFeatureBuildError(
            "selected PBP contained no focal-team feature rows"
        )
    view_source_hashes = dict(source_hashes)
    ledger_hash_name = "sports_row_disposition_ledger_v2"
    existing_ledger_hash = view_source_hashes.get(ledger_hash_name)
    if (
        existing_ledger_hash is not None
        and existing_ledger_hash != disposition_ledger.rows_sha256
    ):
        raise SportsFeatureBuildError(
            "source_hashes contains a conflicting disposition-ledger hash"
        )
    view_source_hashes[ledger_hash_name] = disposition_ledger.rows_sha256
    return SportsFeatureViewV1.build(
        rows=output,
        source_hashes=view_source_hashes,
    )


__all__ = [
    "MARKET_FACTOR_COLUMNS_V1",
    "NFL_FACTOR_LAB_PBP_COLUMNS",
    "REGISTERED_FACTOR_COLUMNS_V1",
    "SportsFeatureBuildError",
    "build_sports_feature_view",
]
