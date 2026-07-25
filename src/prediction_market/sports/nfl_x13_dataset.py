"""Byte-verified source loader for the frozen NFL X-13 twenty-game sample."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_IDS,
    canonical_sha256,
)
from prediction_market.sports.x11 import (
    X11_FROZEN_PARTITION_ALLOWLIST,
    X11_NFLVERSE_VERSION,
)
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)


_PBP_DATASET_ID = "DS-NFLVERSE"
_PBP_OBJECT_SHA256, _PBP_SCHEMA_SHA256 = X11_FROZEN_PARTITION_ALLOWLIST[2025]
_PARTICIPATION_DATASET_ID = "DS-NFLVERSE-PARTICIPATION"
_PARTICIPATION_OBJECT_SHA256 = (
    "sha256:3b26b9487267f011fdbafbce133800ce964c963fdb8b27725abf65663cc46e0f"
)
_PARTICIPATION_SCHEMA_SHA256 = (
    "sha256:9bf2e2bce89ff54b88151df66f3aa7dede804fa12688bcb80b9fb89cb01429c2"
)
_PBP_COLUMNS = (
    "game_id",
    "play_id",
    "order_sequence",
    "time_of_day",
    "time",
    "season_type",
    "qtr",
    "quarter_seconds_remaining",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "side_of_field",
    "fixed_drive",
    "goal_to_go",
    "play_clock",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_score",
    "defteam_score",
    "posteam_score_post",
    "defteam_score_post",
    "total_home_score",
    "total_away_score",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "play_type",
    "play_type_nfl",
    "desc",
    "pass_attempt",
    "rush_attempt",
    "sack",
    "punt_attempt",
    "kickoff_attempt",
    "field_goal_attempt",
    "field_goal_result",
    "punt_blocked",
    "qb_kneel",
    "qb_spike",
    "extra_point_attempt",
    "two_point_attempt",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "return_touchdown",
    "safety",
    "first_down",
    "complete_pass",
    "interception",
    "fumble_lost",
    "fourth_down_failed",
    "timeout",
    "timeout_team",
    "penalty",
    "replay_or_challenge",
    "replay_or_challenge_result",
    "series_result",
    "quarter_end",
    "play_deleted",
    "passer_player_name",
    "rusher_player_name",
    "receiver_player_name",
    "kicker_player_name",
    "punter_player_name",
    "punt_returner_player_name",
    "kickoff_returner_player_name",
    "interception_player_name",
    "td_player_name",
    "safety_player_name",
    "blocked_player_name",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
)
_PARTICIPATION_COLUMNS = (
    "nflverse_game_id",
    "play_id",
    "offense_formation",
    "offense_personnel",
    "defense_personnel",
    "offense_players",
    "defense_players",
    "n_offense",
    "n_defense",
    "offense_names",
    "defense_names",
    "offense_positions",
    "defense_positions",
)
_EXPECTED_SAMPLE_AUDIT = {
    "row_count": 3_699,
    "game_count": 20,
    "touchdowns": 124,
    "made_field_goals": 98,
    "interceptions": 38,
    "lost_fumbles": 28,
    "turnover_on_downs": 25,
    "reviews": 31,
    "reversed_reviews": 22,
    "safeties": 4,
    "return_touchdowns": 9,
    "overtime_game_count": 4,
    "tie_game_count": 1,
}


class X13DatasetError(ValueError):
    """The frozen source objects cannot prove the registered X-13 sample."""


@dataclass(frozen=True, slots=True)
class X13SourceBundle:
    pbp_rows_by_game: Mapping[str, tuple[dict[str, Any], ...]]
    participation_rows_by_game: Mapping[str, tuple[dict[str, Any], ...]]
    pbp_object_sha256: str
    pbp_manifest_sha256: str
    participation_object_sha256: str
    participation_manifest_sha256: str
    selected_rows_sha256: str
    sample_audit: Mapping[str, int]
    full_11v11_participation_rows: int


def _source_hash_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return int(value)
        return Decimal(str(value))
    if isinstance(value, Mapping):
        return {
            str(key): _source_hash_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_source_hash_value(child) for child in value]
    return value


def _exact_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise X13DatasetError(f"{field} must be an integer")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        number = float(value)
        if number.is_integer():
            return int(number)
    raise X13DatasetError(f"{field} must be an integer")


def _indicator(value: object) -> int:
    if value is None:
        return 0
    number = _exact_integer(value, "indicator")
    if number not in {0, 1}:
        raise X13DatasetError("indicator must be binary")
    return number


def canonical_team_field_orientation(
    *,
    home_team: str,
    away_team: str,
    possession_team: str,
    yardline_100: int | float,
) -> dict[str, object]:
    """Map team-relative field state to a stable, explicitly non-tracking field."""

    if (
        type(home_team) is not str
        or not home_team
        or type(away_team) is not str
        or not away_team
        or home_team == away_team
    ):
        raise X13DatasetError("home and away team identity must be explicit")
    if possession_team not in {home_team, away_team}:
        raise X13DatasetError("possession team is not one of the game teams")
    yards = _exact_integer(yardline_100, "yardline_100")
    if not 0 <= yards <= 100:
        raise X13DatasetError("yardline_100 must be between 0 and 100")
    if possession_team == away_team:
        direction = "right"
        coordinate = 100 - yards
    else:
        direction = "left"
        coordinate = yards
    return {
        "direction": direction,
        "field_coordinate_0_100": coordinate,
        "semantics": "canonical_team_end_orientation_not_tracking_coordinate",
    }


def summarize_frozen_sample(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Summarize only the frozen, predeclared event counts used by selection."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise X13DatasetError("sample rows must be an array")
    game_ids: set[str] = set()
    overtime_games: set[str] = set()
    max_scores: dict[str, list[int]] = {}
    counts = {
        "row_count": 0,
        "touchdowns": 0,
        "made_field_goals": 0,
        "interceptions": 0,
        "lost_fumbles": 0,
        "turnover_on_downs": 0,
        "reviews": 0,
        "reversed_reviews": 0,
        "safeties": 0,
        "return_touchdowns": 0,
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise X13DatasetError("sample row must be an object")
        game_id = row.get("game_id")
        if type(game_id) is not str or not game_id:
            raise X13DatasetError("sample row lacks game_id")
        game_ids.add(game_id)
        counts["row_count"] += 1
        counts["touchdowns"] += _indicator(row.get("touchdown"))
        counts["made_field_goals"] += int(row.get("field_goal_result") == "made")
        counts["interceptions"] += _indicator(row.get("interception"))
        counts["lost_fumbles"] += _indicator(row.get("fumble_lost"))
        counts["turnover_on_downs"] += _indicator(row.get("fourth_down_failed"))
        counts["reviews"] += _indicator(row.get("replay_or_challenge"))
        counts["reversed_reviews"] += int(
            row.get("replay_or_challenge_result") == "reversed"
        )
        counts["safeties"] += _indicator(row.get("safety"))
        counts["return_touchdowns"] += _indicator(row.get("return_touchdown"))
        quarter = row.get("qtr")
        if quarter is not None and _exact_integer(quarter, "qtr") > 4:
            overtime_games.add(game_id)
        scores = max_scores.setdefault(game_id, [0, 0])
        home_score = row.get("total_home_score")
        away_score = row.get("total_away_score")
        if home_score is not None:
            scores[0] = max(scores[0], _exact_integer(home_score, "home score"))
        if away_score is not None:
            scores[1] = max(scores[1], _exact_integer(away_score, "away score"))
    counts["game_count"] = len(game_ids)
    counts["overtime_game_count"] = len(overtime_games)
    counts["tie_game_count"] = sum(home == away for home, away in max_scores.values())
    return counts


def _manifest_for_object(
    store_root: Path,
    *,
    dataset_id: str,
    object_sha256: str,
    license_ref: str,
) -> Path:
    digest = object_sha256.removeprefix("sha256:")
    candidates: list[Path] = []
    for path in (store_root / "manifests").rglob("*.manifest.json"):
        try:
            document = json.loads(path.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise X13DatasetError("raw manifest tree contains unreadable JSON") from error
        if (
            document.get("dataset_id") == dataset_id
            and document.get("object_sha256") == object_sha256
            and document.get("license_ref") == license_ref
            and Path(str(document.get("native_object_path", ""))).name
            == f"{digest}.parquet"
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise X13DatasetError(
            f"{dataset_id} object must resolve to exactly one manifest"
        )
    return candidates[0]


def _verified_parquet(
    *,
    program_root: Path,
    store_root: Path,
    dataset_id: str,
    object_sha256: str,
    schema_sha256: str,
    expected_license_status: str,
    expected_license_ref: str,
    expected_version: str | None,
    columns: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    path = _manifest_for_object(
        store_root,
        dataset_id=dataset_id,
        object_sha256=object_sha256,
        license_ref=expected_license_ref,
    )
    try:
        verified = read_verified_static_object(
            path,
            store_root=store_root,
            program_root=program_root,
        )
    except StaticStoreError as error:
        raise X13DatasetError(f"{dataset_id} failed immutable verification") from error
    record = verified.record
    if (
        record.dataset != dataset_id
        or record.manifest.object_sha256 != object_sha256
        or record.manifest.schema_fingerprint != schema_sha256
        or record.manifest.license_ref != expected_license_ref
        or record.manifest.license_status != expected_license_status
        or (expected_version is not None and record.version != expected_version)
    ):
        raise X13DatasetError(f"{dataset_id} identity or governance status changed")
    try:
        table = pq.read_table(pa.BufferReader(verified.object_bytes), columns=list(columns))
    except (pa.ArrowException, OSError, ValueError) as error:
        raise X13DatasetError(f"{dataset_id} is not the expected Parquet schema") from error
    return (
        [dict(row) for row in table.to_pylist()],
        record.manifest.manifest_sha256,
    )


def _group_pbp(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, tuple[dict[str, Any], ...]]:
    selected: dict[str, list[dict[str, Any]]] = {
        game_id: [] for game_id in X13_GAME_IDS
    }
    for raw_ordinal, source in enumerate(rows):
        game_id = source.get("game_id")
        if game_id not in selected:
            continue
        row = dict(source)
        row["_raw_record_ordinal"] = raw_ordinal
        row["game_end"] = int(
            row.get("play_type") == "end_game"
            or str(row.get("desc") or "").upper() in {"END GAME", "END OF GAME"}
        )
        posteam = row.get("posteam")
        yardline = row.get("yardline_100")
        if posteam is not None and yardline is not None:
            orientation = canonical_team_field_orientation(
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                possession_team=str(posteam),
                yardline_100=yardline,
            )
            row["play_direction"] = orientation["direction"]
            row["field_coordinate_0_100"] = orientation[
                "field_coordinate_0_100"
            ]
            row["field_orientation_semantics"] = orientation["semantics"]
        else:
            row["play_direction"] = None
            row["field_coordinate_0_100"] = None
            row["field_orientation_semantics"] = (
                "unavailable_without_possession_and_yardline"
            )
        selected[str(game_id)].append(row)

    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for game_id in X13_GAME_IDS:
        game_rows = selected[game_id]
        if not game_rows:
            raise X13DatasetError(f"frozen PBP has no rows for {game_id}")
        ordered = sorted(
            game_rows,
            key=lambda row: (
                _exact_integer(row["order_sequence"], "order_sequence"),
                _exact_integer(row["_raw_record_ordinal"], "raw ordinal"),
            ),
        )
        orders = [
            _exact_integer(row["order_sequence"], "order_sequence")
            for row in ordered
        ]
        if any(later <= earlier for earlier, later in zip(orders, orders[1:])):
            raise X13DatasetError(f"{game_id} native order is not strictly increasing")
        result[game_id] = tuple(ordered)
    return MappingProxyType(result)


def _group_participation(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, tuple[dict[str, Any], ...]], int]:
    selected: dict[str, list[dict[str, Any]]] = {
        game_id: [] for game_id in X13_GAME_IDS
    }
    seen: set[tuple[str, int]] = set()
    complete = 0
    for source in rows:
        game_id = source.get("nflverse_game_id")
        if game_id not in selected:
            continue
        play_id = _exact_integer(source.get("play_id"), "participation play_id")
        key = (str(game_id), play_id)
        if key in seen:
            raise X13DatasetError("participation contains a duplicate game/play key")
        seen.add(key)
        row = dict(source)
        selected[str(game_id)].append(row)
        if row.get("n_offense") == 11 and row.get("n_defense") == 11:
            complete += 1
    for game_id, game_rows in selected.items():
        if not game_rows:
            raise X13DatasetError(f"participation has no rows for {game_id}")
        game_rows.sort(key=lambda row: _exact_integer(row["play_id"], "play_id"))
    return (
        MappingProxyType(
            {game_id: tuple(selected[game_id]) for game_id in X13_GAME_IDS}
        ),
        complete,
    )


def load_x13_source_bundle(
    *,
    program_root: str | Path,
    raw_store_root: str | Path,
) -> X13SourceBundle:
    """Verify both original objects and extract exactly the registered games."""

    program = Path(program_root).resolve()
    store = Path(raw_store_root).resolve()
    pbp_rows, pbp_manifest_sha256 = _verified_parquet(
        program_root=program,
        store_root=store,
        dataset_id=_PBP_DATASET_ID,
        object_sha256=_PBP_OBJECT_SHA256,
        schema_sha256=_PBP_SCHEMA_SHA256,
        expected_license_status="approved",
        expected_license_ref="I-018",
        expected_version=X11_NFLVERSE_VERSION,
        columns=_PBP_COLUMNS,
    )
    participation_rows, participation_manifest_sha256 = _verified_parquet(
        program_root=program,
        store_root=store,
        dataset_id=_PARTICIPATION_DATASET_ID,
        object_sha256=_PARTICIPATION_OBJECT_SHA256,
        schema_sha256=_PARTICIPATION_SCHEMA_SHA256,
        expected_license_status="research_only",
        expected_license_ref="O-009",
        expected_version="release-asset-2025-observed-20260724",
        columns=_PARTICIPATION_COLUMNS,
    )
    pbp_by_game = _group_pbp(pbp_rows)
    participation_by_game, complete_11v11 = _group_participation(
        participation_rows
    )
    selected_rows = [
        row for game_id in X13_GAME_IDS for row in pbp_by_game[game_id]
    ]
    sample_audit = summarize_frozen_sample(selected_rows)
    if sample_audit != _EXPECTED_SAMPLE_AUDIT:
        raise X13DatasetError(
            "frozen 20-game sample counts changed: "
            f"expected {_EXPECTED_SAMPLE_AUDIT!r}, got {sample_audit!r}"
        )
    participation_count = sum(len(rows) for rows in participation_by_game.values())
    if participation_count != 3_425 or complete_11v11 != 3_413:
        raise X13DatasetError(
            "frozen participation coverage changed from 3425 / 3413 full 11v11"
        )
    return X13SourceBundle(
        pbp_rows_by_game=pbp_by_game,
        participation_rows_by_game=participation_by_game,
        pbp_object_sha256=_PBP_OBJECT_SHA256,
        pbp_manifest_sha256=pbp_manifest_sha256,
        participation_object_sha256=_PARTICIPATION_OBJECT_SHA256,
        participation_manifest_sha256=participation_manifest_sha256,
        selected_rows_sha256=canonical_sha256(
            _source_hash_value(selected_rows)
        ),
        sample_audit=MappingProxyType(sample_audit),
        full_11v11_participation_rows=complete_11v11,
    )


__all__ = [
    "X13DatasetError",
    "X13SourceBundle",
    "canonical_team_field_orientation",
    "load_x13_source_bundle",
    "summarize_frozen_sample",
]
