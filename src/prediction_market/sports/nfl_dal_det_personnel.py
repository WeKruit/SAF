"""Byte-verified DAL–DET play-event and 11v11 personnel reconstruction.

This is an offline historical reconstruction.  It joins the frozen nflverse
play-by-play object to the separately preserved FTN participation release on
the stable ``nflverse_game_id + play_id`` key.  It never assigns a timestamp
to a substitution: changes are explicitly set differences between two
consecutive observed play rosters.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prediction_market.contracts import canonical_json_bytes, canonical_sha256
from prediction_market.sports.nfl_dal_det_audit import (
    CANONICAL_GAME_ID,
    NATIVE_GAME_ID,
    DALDETAuditError,
)
from prediction_market.sports.nfl_game_replay import _load_verified_game_rows
from prediction_market.static_store import StaticStoreError, read_verified_static_object


_PARTICIPATION_DATASET_ID = "DS-NFLVERSE-PARTICIPATION"
_PARTICIPATION_OBJECT_SHA256 = "sha256:3b26b9487267f011fdbafbce133800ce964c963fdb8b27725abf65663cc46e0f"
_PARTICIPATION_MANIFEST_SHA256 = "sha256:c5b2f2d56b3e04ce8ce6674e09a2571eef39bfefaeb16d2c06da50a332b1e291"
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


class DALDETPersonnelError(ValueError):
    """The separately preserved personnel object cannot safely join PBP."""


def _required_text(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise DALDETPersonnelError(f"{field} must be nonempty text")
    return value


def _play_id(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DALDETPersonnelError(f"{field} must be an integer play id")
    if int(value) != value or int(value) < 0:
        raise DALDETPersonnelError(f"{field} must be an integer play id")
    return int(value)


def _split_semicolon(value: object, field: str) -> list[str]:
    text = _required_text(value, field)
    result = text.split(";")
    if any(not item for item in result):
        raise DALDETPersonnelError(f"{field} contains an empty player field")
    return result


def _manifest_path(store_root: Path) -> Path:
    digest = _PARTICIPATION_MANIFEST_SHA256.removeprefix("sha256:")
    matches = tuple((store_root / "manifests").glob(f"**/{digest}.manifest.json"))
    if len(matches) != 1:
        raise DALDETPersonnelError("participation manifest must resolve exactly once")
    return matches[0]


def _load_verified_participation_rows(*, program_root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    store_root = program_root / "var" / "raw"
    try:
        verified = read_verified_static_object(
            _manifest_path(store_root), store_root=store_root, program_root=program_root
        )
    except StaticStoreError as error:
        raise DALDETPersonnelError("participation object failed immutable verification") from error
    record = verified.record
    if (
        record.dataset != _PARTICIPATION_DATASET_ID
        or record.manifest.object_sha256 != _PARTICIPATION_OBJECT_SHA256
        or record.manifest.manifest_sha256 != _PARTICIPATION_MANIFEST_SHA256
        or "research_only" != record.manifest.license_status
    ):
        raise DALDETPersonnelError("participation object identity or license boundary changed")

    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(pa.BufferReader(verified.object_bytes), columns=list(_PARTICIPATION_COLUMNS))
    except (pa.ArrowException, OSError, ValueError) as error:
        raise DALDETPersonnelError("participation object is not readable as the expected Parquet schema") from error
    rows = [dict(row) for row in table.to_pylist() if row.get("nflverse_game_id") == NATIVE_GAME_ID]
    if not rows:
        raise DALDETPersonnelError("participation object has no DAL–DET rows")
    return rows, {
        "object_sha256": _PARTICIPATION_OBJECT_SHA256,
        "manifest_sha256": _PARTICIPATION_MANIFEST_SHA256,
        "license_status": record.manifest.license_status,
    }


def _event(source: Mapping[str, object]) -> dict[str, Any]:
    flags = {
        name: bool(source.get(name))
        for name in (
            "pass_attempt", "rush_attempt", "punt_attempt", "field_goal_attempt",
            "kickoff_attempt", "extra_point_attempt", "two_point_attempt", "touchdown",
            "pass_touchdown", "rush_touchdown", "return_touchdown", "safety",
            "first_down", "interception", "fumble_lost", "timeout", "quarter_end",
        )
    }
    actors = {
        key: value
        for key in (
            "passer_player_name", "rusher_player_name", "receiver_player_name",
            "kicker_player_name", "punter_player_name", "punt_returner_player_name",
            "kickoff_returner_player_name",
        )
        if (value := source.get(key)) is not None
    }
    return {
        "play_type": source.get("play_type"),
        "play_type_nfl": source.get("play_type_nfl"),
        "description": source.get("desc"),
        "timeout_team": source.get("timeout_team"),
        "flags": flags,
        "actors": actors,
    }


def _state(source: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "qtr", "quarter_seconds_remaining", "game_seconds_remaining", "home_team",
        "away_team", "posteam", "defteam", "fixed_drive", "down", "ydstogo",
        "yardline_100", "goal_to_go", "total_home_score", "total_away_score",
        "home_timeouts_remaining", "away_timeouts_remaining",
    )
    state: dict[str, object] = {}
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            raise DALDETPersonnelError(f"state {key} cannot be boolean")
        # The frozen Parquet schema exposes NFL integral state quantities as
        # doubles.  Canonical artifacts store their semantic integer value,
        # never a binary float representation.
        if isinstance(value, float):
            if not value.is_integer():
                raise DALDETPersonnelError(f"state {key} must be integral")
            value = int(value)
        state[key] = value
    return state


def _participation(source: Mapping[str, object]) -> dict[str, object]:
    offense_ids = _split_semicolon(source.get("offense_players"), "offense_players")
    defense_ids = _split_semicolon(source.get("defense_players"), "defense_players")
    offense_names = _split_semicolon(source.get("offense_names"), "offense_names")
    defense_names = _split_semicolon(source.get("defense_names"), "defense_names")
    offense_positions = _split_semicolon(source.get("offense_positions"), "offense_positions")
    defense_positions = _split_semicolon(source.get("defense_positions"), "defense_positions")
    if any(len(values) != 11 for values in (offense_ids, defense_ids, offense_names, defense_names, offense_positions, defense_positions)):
        raise DALDETPersonnelError("participation row must contain complete 11v11 named roster")
    if source.get("n_offense") != 11 or source.get("n_defense") != 11:
        raise DALDETPersonnelError("participation count must be exactly 11v11")
    return {
        "offense_gsis_ids": offense_ids,
        "defense_gsis_ids": defense_ids,
        "offense_names": offense_names,
        "defense_names": defense_names,
        "offense_positions": offense_positions,
        "defense_positions": defense_positions,
        "offense_personnel": source.get("offense_personnel"),
        "defense_personnel": source.get("defense_personnel"),
        "offense_formation": source.get("offense_formation"),
    }


def _roster_delta(previous: Mapping[str, object] | None, current: Mapping[str, object]) -> dict[str, object] | None:
    if previous is None:
        return None
    result: dict[str, object] = {"semantics": "set difference between consecutive participation rows; not an exact substitution timestamp"}
    for side in ("offense", "defense"):
        before = set(previous[f"{side}_gsis_ids"])
        after = set(current[f"{side}_gsis_ids"])
        result[f"{side}_entering_gsis_ids"] = sorted(after - before)
        result[f"{side}_leaving_gsis_ids"] = sorted(before - after)
    return result


def build_dal_det_event_personnel_timeline(*, program_root: str | Path) -> dict[str, Any]:
    """Build the full timestamped PBP × 11v11 review timeline for DAL–DET."""

    root = Path(program_root).resolve()
    pbp_rows, pbp_object_sha256, _ = _load_verified_game_rows(program_root=root, native_game_id=NATIVE_GAME_ID)
    participation_rows, participation_evidence = _load_verified_participation_rows(program_root=root)
    by_play: dict[int, Mapping[str, object]] = {}
    for source in participation_rows:
        play_id = _play_id(source.get("play_id"), "participation play_id")
        if play_id in by_play:
            raise DALDETPersonnelError("participation key must be unique")
        by_play[play_id] = source

    rows: list[dict[str, Any]] = []
    previous_participation: Mapping[str, object] | None = None
    for source in pbp_rows:
        play_id = _play_id(source.get("play_id"), "PBP play_id")
        participation_source = by_play.get(play_id)
        if participation_source is None:
            continue
        parsed_participation = _participation(participation_source)
        rows.append({
            "play_id": play_id,
            "order_sequence": _play_id(source.get("order_sequence"), "PBP order_sequence"),
            "source_time_utc": source.get("time_of_day"),
            "state": _state(source),
            "event": _event(source),
            "participation": parsed_participation,
            "inferred_between_play_roster_change": _roster_delta(previous_participation, parsed_participation),
        })
        previous_participation = parsed_participation
    if len(rows) != len(participation_rows):
        raise DALDETPersonnelError("every participation row must join exactly one PBP row")
    if any(later["order_sequence"] <= earlier["order_sequence"] for earlier, later in zip(rows, rows[1:])):
        raise DALDETPersonnelError("joined PBP order must be strictly increasing")
    if any(row["source_time_utc"] is None for row in rows):
        raise DALDETPersonnelError("joined event rows require a source timestamp")
    full_11v11_rows = sum(
        len(row["participation"]["offense_gsis_ids"]) == 11 and len(row["participation"]["defense_gsis_ids"]) == 11
        for row in rows
    )
    report: dict[str, Any] = {
        "artifact_type": "nfl_play_event_personnel_timeline",
        "artifact_version": "v1",
        "canonical_game_id": CANONICAL_GAME_ID,
        "owner": "C+D2+I",
        "inputs": {
            "pbp_object_sha256": pbp_object_sha256,
            "participation_object_sha256": participation_evidence["object_sha256"],
            "participation_manifest_sha256": participation_evidence["manifest_sha256"],
        },
        "join": {
            "join_key": "nflverse game_id + play_id",
            "pbp_game_rows": len(pbp_rows),
            "participation_rows": len(participation_rows),
            "joined_rows": len(rows),
            "full_11v11_rows": full_11v11_rows,
            "administrative_or_nonparticipation_pbp_rows": len(pbp_rows) - len(rows),
            "missing_pbp_for_participation": 0,
        },
        "evidence_boundary": {
            "historical_only": True,
            "exact_substitution_timestamp_available": False,
            "market_alignment_performed": False,
            "reaction_or_alpha_claim": False,
        },
        "rows": rows,
    }
    report["artifact_sha256"] = canonical_sha256({key: value for key, value in report.items() if key != "artifact_sha256"})
    return report


def write_dal_det_event_personnel_timeline(
    report: Mapping[str, object], *, output_path: str | Path
) -> None:
    """Atomically write the derived timeline outside immutable raw storage."""

    path = Path(output_path)
    if path.suffix != ".json":
        raise DALDETPersonnelError("personnel timeline output must be JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(dict(report)) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DALDETPersonnelError",
    "build_dal_det_event_personnel_timeline",
    "write_dal_det_event_personnel_timeline",
]
