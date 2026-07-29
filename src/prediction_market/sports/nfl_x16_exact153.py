"""Governed, streaming publication of the NFL X-13 exact-153 fact batch."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_historical_sports_clean import (
    VerifiedSource,
    discover_main_checkout,
    verify_frozen_2025_sources,
)
from prediction_market.sports.nfl_x16_fact_extraction import (
    GameFactTables,
    NFLFactExtractionError,
    build_game_fact_tables,
)


EXPERIMENT_ID: Final[str] = "X-13"
EXPECTED_GOVERNANCE_MANIFEST_FILE_SHA256: Final[str] = (
    "sha256:545f7e082055cd5070d907da9eacbdb735dd14351c19c5a83d912764bb595a37"
)
EXPECTED_GOVERNANCE_OBJECT_SHA256: Final[str] = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
EXPECTED_DEVELOPMENT_GAME_COUNT: Final[int] = 153
EXPECTED_HOLDOUT_GAME_COUNT: Final[int] = 81
EXPECTED_DEVELOPMENT_WEEKS: Final[tuple[int, ...]] = tuple(range(1, 13))
EXPECTED_HOLDOUT_WEEKS: Final[tuple[int, ...]] = tuple(range(13, 19))
DEFAULT_GOVERNANCE_MANIFEST: Final[Path] = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-registry/manifests/"
    "545f7e082055cd5070d907da9eacbdb735dd14351c19c5a83d912764bb595a37"
    ".manifest.json"
)
DEFAULT_FACTOR_REGISTRY: Final[Path] = Path(
    "registries/factors/nfl_factor_registry_v4.json"
)
EXPECTED_FACTOR_COUNT: Final[int] = 59
LEGACY_PUBLICATION_ID: Final[str] = "exact-153-facts-v3"
PUBLICATION_ID: Final[str] = "exact-153-facts-v4"
BUILDER_VERSION: Final[str] = "nfl_x13_exact153_fact_publication_v4"
SINGLE_GAME_MANIFEST_SCHEMA: Final[str] = (
    "nfl_x13_exact153_single_game_fact_manifest_v4"
)
SEMANTIC_BATCH_SCHEMA: Final[str] = "nfl_x13_exact153_semantic_batch_v4"
BATCH_INDEX_SCHEMA: Final[str] = "nfl_x13_exact153_fact_batch_index_v4"
CLAIM_BOUNDARY: Final[str] = (
    "DEVELOPMENT_SPORTS_FACTS_ONLY; no market reaction, holdout reaction, "
    "causality, execution, or alpha claim"
)

_CANONICAL_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "claim_boundary",
    "game_id",
    "event_id",
    "raw_play_id",
    "play_id",
    "atomic_information_episode_id",
    "score_sequence_id",
    "adjudication_sequence_id",
    "information_status",
    "stage_b_information_event_eligible",
    "final_sports_outcome_eligible",
    "source_interval_start",
    "source_interval_end",
    "source_resolution",
    "source_interval_semantics",
    "known_at",
    "order_sequence",
    "source_time_utc",
    "source_time_semantics",
    "quarter",
    "game_clock",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "actor_team",
    "beneficiary_team",
    "actor_is_home",
    "beneficiary_is_home",
    "possession_is_home",
    "beneficiary_resolution_status",
    "possession_before",
    "posteam_semantics",
    "defense_team",
    "next_observed_possession",
    "next_observed_possession_semantics",
    "offense_direction",
    "transition_direction_semantics",
    "field_orientation_semantics",
    "pre_home_score",
    "pre_away_score",
    "post_home_score",
    "post_away_score",
    "score_margin_home",
    "score_margin_offense",
    "down",
    "distance",
    "goal_to_go",
    "pre_yardline",
    "post_yardline",
    "pre_field_coordinate_0_100",
    "post_field_coordinate_0_100",
    "next_observed_state_play_id",
    "next_observed_possession_state",
    "next_observed_yardline",
    "next_observed_field_coordinate_0_100",
    "visual_end_field_coordinate_0_100",
    "visual_end_semantics",
    "yardline_100",
    "pre_red_zone",
    "tied",
    "one_score_game",
    "drive_id",
    "series_id",
    "primary_action",
    "outcome_tags",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "return_yards",
    "pass_length",
    "pass_location",
    "run_location",
    "run_gap",
    "kick_distance",
    "field_goal_result",
    "extra_point_result",
    "two_point_result",
    "penalty_team",
    "penalty_type",
    "penalty_yards",
    "review_result",
    "timeout_team",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "description",
    "participation_status",
    "factor_eligible",
    "row_disposition",
    "home_wp_before_diagnostic",
    "home_wp_after_diagnostic",
    "home_wp_delta_diagnostic",
    "reference_semantics",
    "source_hashes",
    "pbp_source_sha256",
    "participation_source_sha256",
)
TABLE_SCHEMAS: Final[Mapping[str, tuple[str, ...]]] = {
    "canonical_factor_events": _CANONICAL_EVENT_COLUMNS,
    "factor_event_tags": (
        "game_id",
        "event_id",
        "play_id",
        "tag",
        "pbp_source_sha256",
    ),
    "factor_event_players": (
        "game_id",
        "event_id",
        "play_id",
        "role",
        "player_id",
        "player_name",
        "team",
        "unit",
        "position",
        "jersey_number",
        "evidence_class",
        "source_sha256",
    ),
    "player_availability_events": (
        "game_id",
        "team",
        "unit",
        "player_id",
        "availability_observation",
        "interval_start_event_id",
        "interval_end_event_id",
        "evidence_grade",
        "evidence_semantics",
        "participation_source_sha256",
    ),
    "injury_evidence": (
        "game_id",
        "event_id",
        "play_id",
        "source_time_utc",
        "evidence_type",
        "status",
        "team",
        "jersey_number",
        "source_player_name",
        "player_id",
        "player_name",
        "identity_status",
        "evidence_grade",
        "evidence_semantics",
        "raw_evidence_text",
        "pbp_source_sha256",
        "participation_source_sha256",
    ),
    "factor_hits": (
        "game_id",
        "event_id",
        "play_id",
        "factor_id",
        "factor_version",
        "registry_sha256",
        "pbp_source_sha256",
        "predicate_evidence",
    ),
    "factor_coverage_audit": (
        "factor_id",
        "factor_version",
        "status",
        "scope",
        "event_count",
        "game_count",
        "availability_status",
        "registry_sha256",
    ),
    "sports_row_reconciliation": (
        "game_id",
        "raw_play_id",
        "event_id",
        "atomic_information_episode_id",
        "raw_row_preserved",
        "row_disposition",
        "factor_eligible",
        "exclusion_reason",
        "pbp_source_sha256",
    ),
}
_EVENT_BOOLEAN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "stage_b_information_event_eligible",
        "final_sports_outcome_eligible",
        "actor_is_home",
        "beneficiary_is_home",
        "possession_is_home",
        "goal_to_go",
        "pre_red_zone",
        "tied",
        "one_score_game",
        "factor_eligible",
    }
)
_EVENT_INTEGER_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "order_sequence",
        "quarter",
        "game_seconds_remaining",
        "pre_home_score",
        "pre_away_score",
        "post_home_score",
        "post_away_score",
        "score_margin_home",
        "score_margin_offense",
        "down",
        "distance",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
    }
)
_EVENT_FLOAT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "pre_field_coordinate_0_100",
        "post_field_coordinate_0_100",
        "next_observed_field_coordinate_0_100",
        "visual_end_field_coordinate_0_100",
        "yardline_100",
        "yards_gained",
        "air_yards",
        "yards_after_catch",
        "return_yards",
        "kick_distance",
        "penalty_yards",
        "home_wp_before_diagnostic",
        "home_wp_after_diagnostic",
        "home_wp_delta_diagnostic",
    }
)
_COLUMN_KIND_OVERRIDES: Final[Mapping[tuple[str, str], str]] = {
    **{
        ("canonical_factor_events", column): "boolean"
        for column in _EVENT_BOOLEAN_COLUMNS
    },
    **{
        ("canonical_factor_events", column): "Int64"
        for column in _EVENT_INTEGER_COLUMNS
    },
    **{
        ("canonical_factor_events", column): "Float64"
        for column in _EVENT_FLOAT_COLUMNS
    },
    ("factor_coverage_audit", "event_count"): "Int64",
    ("factor_coverage_audit", "game_count"): "Int64",
    ("sports_row_reconciliation", "raw_row_preserved"): "boolean",
    ("sports_row_reconciliation", "factor_eligible"): "boolean",
}
TABLE_PANDAS_DTYPES: Final[Mapping[str, Mapping[str, str]]] = {
    name: {
        column: _COLUMN_KIND_OVERRIDES.get((name, column), "string")
        for column in columns
    }
    for name, columns in TABLE_SCHEMAS.items()
}
_ARROW_TYPES: Final[Mapping[str, pa.DataType]] = {
    "string": pa.string(),
    "boolean": pa.bool_(),
    "Int64": pa.int64(),
    "Float64": pa.float64(),
}
TABLE_ARROW_SCHEMAS: Final[Mapping[str, pa.Schema]] = {
    name: pa.schema(
        [
            pa.field(column, _ARROW_TYPES[TABLE_PANDAS_DTYPES[name][column]])
            for column in columns
        ]
    )
    for name, columns in TABLE_SCHEMAS.items()
}
_TABLE_ATTRIBUTES: Final[tuple[tuple[str, str], ...]] = (
    ("canonical_factor_events", "events"),
    ("factor_event_tags", "tags"),
    ("factor_event_players", "players"),
    ("player_availability_events", "availability"),
    ("injury_evidence", "injury_evidence"),
    ("factor_hits", "factor_hits"),
    ("factor_coverage_audit", "factor_coverage"),
    ("sports_row_reconciliation", "reconciliation"),
)


class NFLExact153PublicationError(RuntimeError):
    """An exact-153 authority or publication invariant failed closed."""


@dataclass(frozen=True, slots=True)
class Exact153Authority:
    manifest_path: Path
    manifest_file_sha256: str
    object_path: Path
    object_sha256: str
    development_game_ids: tuple[str, ...]
    final_holdout_game_ids: tuple[str, ...]

    def require_development(self, game_id: str) -> None:
        if game_id in self.development_game_ids:
            return
        if game_id in self.final_holdout_game_ids:
            raise NFLExact153PublicationError(
                f"holdout game cannot enter exact-153 output: {game_id}"
            )
        raise NFLExact153PublicationError(f"unknown game is not governed: {game_id}")


@dataclass(frozen=True, slots=True)
class PublishedGameFactBundle:
    game_id: str
    bundle_sha256: str
    manifest_path: Path
    manifest_sha256: str
    counts: Mapping[str, int]
    table_semantic_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PublishedExact153FactBatch:
    game_count: int
    batch_sha256: str
    semantic_batch_sha256: str
    index_path: Path
    index_sha256: str
    games: tuple[PublishedGameFactBundle, ...]
    aggregate_counts: Mapping[str, int]


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    raise NFLExact153PublicationError(
        f"value is not canonical-JSON serializable: {type(value).__name__}"
    )


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            _canonical_value(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _extractor_semantic_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise NFLExact153PublicationError(
                f"content-addressed object collision: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise NFLExact153PublicationError(
                    f"content-addressed publish race: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise NFLExact153PublicationError(f"{label} is not valid JSON") from exc
    if type(payload) is not dict:
        raise NFLExact153PublicationError(f"{label} must be a JSON object")
    return payload, encoded


def _resolve_under(root: Path, relative: str | Path, *, label: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise NFLExact153PublicationError(f"{label} escapes its declared root")
    return candidate


def _require_v4_publication_path(
    *,
    output_root: Path,
    candidate: str | Path,
    expected_prefix: tuple[str, ...],
    label: str,
    require_relative: bool = False,
) -> Path:
    declared = Path(candidate)
    if require_relative and declared.is_absolute():
        raise NFLExact153PublicationError(f"{label} is not a V4 publication path")
    resolved = _resolve_under(output_root, declared, label=label)
    relative = resolved.relative_to(output_root.resolve())
    if (
        relative.parts[: len(expected_prefix)] != expected_prefix
        or (require_relative and declared.parts != relative.parts)
    ):
        raise NFLExact153PublicationError(f"{label} is not a V4 publication path")
    return resolved


def verify_exact153_authority(
    *,
    project_root: str | Path,
    governance_manifest_path: str | Path | None = None,
) -> Exact153Authority:
    """Verify the frozen expansion registry and return its exact cohort identities."""

    root = Path(project_root).resolve()
    manifest_path = (
        _resolve_under(root, DEFAULT_GOVERNANCE_MANIFEST, label="governance manifest")
        if governance_manifest_path is None
        else Path(governance_manifest_path).resolve()
    )
    manifest, manifest_bytes = _read_json_object(
        manifest_path, label="governance manifest"
    )
    manifest_sha = _sha256_bytes(manifest_bytes)
    if manifest_sha != EXPECTED_GOVERNANCE_MANIFEST_FILE_SHA256:
        raise NFLExact153PublicationError("governance manifest bytes/hash mismatch")
    if (
        "experiment_id" in manifest
        and manifest.get("experiment_id") != EXPERIMENT_ID
    ):
        raise NFLExact153PublicationError(
            "governance manifest experiment identity mismatch"
        )
    if (
        manifest.get("schema") != "nfl_factor_expansion_registry_manifest_v2"
        or manifest.get("candidate_game_count")
        != EXPECTED_DEVELOPMENT_GAME_COUNT + EXPECTED_HOLDOUT_GAME_COUNT
        or manifest.get("development_game_count")
        != EXPECTED_DEVELOPMENT_GAME_COUNT
        or manifest.get("final_holdout_game_count") != EXPECTED_HOLDOUT_GAME_COUNT
        or manifest.get("development_reaction_access") is not False
        or manifest.get("final_holdout_reaction_access") is not False
        or manifest.get("object_sha256") != EXPECTED_GOVERNANCE_OBJECT_SHA256
    ):
        raise NFLExact153PublicationError("governance manifest contract mismatch")
    authority_root = manifest_path.parent.parent
    object_path = _resolve_under(
        authority_root,
        str(manifest.get("object_path", "")),
        label="governance object",
    )
    authority_object, object_bytes = _read_json_object(
        object_path, label="governance object"
    )
    if (
        len(object_bytes) != manifest.get("byte_length")
        or _sha256_bytes(object_bytes) != EXPECTED_GOVERNANCE_OBJECT_SHA256
        or authority_object.get("schema") != "nfl_factor_expansion_registry_v2"
        or authority_object.get("experiment_id") != EXPERIMENT_ID
        or authority_object.get("status") != "FROZEN"
    ):
        if authority_object.get("experiment_id") != EXPERIMENT_ID:
            raise NFLExact153PublicationError(
                "governance object experiment identity mismatch"
            )
        raise NFLExact153PublicationError("governance object bytes/hash contract mismatch")
    split_lock = authority_object.get("split_lock")
    if type(split_lock) is not dict:
        raise NFLExact153PublicationError("governance split_lock is absent")
    assignments = split_lock.get("game_assignments")
    development_lock = split_lock.get("development")
    holdout_lock = split_lock.get("final_holdout")
    if (
        type(assignments) is not list
        or type(development_lock) is not dict
        or type(holdout_lock) is not dict
        or development_lock.get("game_count") != EXPECTED_DEVELOPMENT_GAME_COUNT
        or tuple(development_lock.get("weeks", ())) != EXPECTED_DEVELOPMENT_WEEKS
        or development_lock.get("reaction_access") is not False
        or holdout_lock.get("game_count") != EXPECTED_HOLDOUT_GAME_COUNT
        or tuple(holdout_lock.get("weeks", ())) != EXPECTED_HOLDOUT_WEEKS
        or holdout_lock.get("reaction_access") is not False
    ):
        raise NFLExact153PublicationError("governance split contract mismatch")
    development_rows: list[tuple[int, str]] = []
    holdout_rows: list[tuple[int, str]] = []
    for row in assignments:
        if (
            type(row) is not dict
            or type(row.get("game_id")) is not str
            or type(row.get("week")) is not int
        ):
            raise NFLExact153PublicationError("governance assignment is malformed")
        cohort = row.get("cohort")
        value = (int(row["week"]), str(row["game_id"]))
        if cohort == "development":
            if value[0] not in EXPECTED_DEVELOPMENT_WEEKS:
                raise NFLExact153PublicationError("development week is outside weeks 1-12")
            development_rows.append(value)
        elif cohort == "final_holdout":
            if value[0] not in EXPECTED_HOLDOUT_WEEKS:
                raise NFLExact153PublicationError("holdout week is outside weeks 13-18")
            holdout_rows.append(value)
        else:
            raise NFLExact153PublicationError("governance assignment cohort is unknown")
    development_ids = tuple(game_id for _, game_id in sorted(development_rows))
    holdout_ids = tuple(game_id for _, game_id in sorted(holdout_rows))
    if (
        len(development_ids) != EXPECTED_DEVELOPMENT_GAME_COUNT
        or len(set(development_ids)) != EXPECTED_DEVELOPMENT_GAME_COUNT
        or len(holdout_ids) != EXPECTED_HOLDOUT_GAME_COUNT
        or len(set(holdout_ids)) != EXPECTED_HOLDOUT_GAME_COUNT
    ):
        raise NFLExact153PublicationError("governance exact cohort counts are invalid")
    if set(development_ids).intersection(holdout_ids):
        raise NFLExact153PublicationError(
            "development and final holdout must be disjoint"
        )
    return Exact153Authority(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        object_path=object_path,
        object_sha256=EXPECTED_GOVERNANCE_OBJECT_SHA256,
        development_game_ids=development_ids,
        final_holdout_game_ids=holdout_ids,
    )


def _verify_factor_registry(
    project_root: Path,
    factor_registry_path: str | Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    path = (
        _resolve_under(
            project_root,
            DEFAULT_FACTOR_REGISTRY,
            label="factor registry",
        )
        if factor_registry_path is None
        else Path(factor_registry_path).resolve()
    )
    registry, encoded = _read_json_object(path, label="factor registry")
    factors = registry.get("factors")
    if (
        registry.get("schema") != "NFLFactorRegistryV4"
        or registry.get("status") != "AUTHORITATIVE"
        or registry.get("version") != "v4"
        or type(factors) is not list
    ):
        raise NFLExact153PublicationError(
            "factor registry schema/status/version mismatch"
        )
    identities: list[dict[str, str]] = []
    seen_factor_ids: set[str] = set()
    for factor in factors:
        if type(factor) is not dict:
            raise NFLExact153PublicationError("factor registry identity is malformed")
        factor_id = factor.get("factor_id")
        factor_version = factor.get("version")
        if (
            type(factor_id) is not str
            or not factor_id
            or factor_id in seen_factor_ids
            or type(factor_version) is not str
            or not factor_version
        ):
            raise NFLExact153PublicationError("factor registry identity is malformed")
        seen_factor_ids.add(factor_id)
        identities.append(
            {"factor_id": factor_id, "version": factor_version}
        )
    if len(identities) != EXPECTED_FACTOR_COUNT:
        raise NFLExact153PublicationError(
            f"factor registry must bind exactly {EXPECTED_FACTOR_COUNT} identities"
        )
    return registry, {
        "schema": registry["schema"],
        "status": registry["status"],
        "version": registry["version"],
        "factor_count": len(identities),
        "factor_identity_sha256": _extractor_semantic_sha256(identities),
        "file_sha256": _sha256_bytes(encoded),
        "semantic_sha256": _extractor_semantic_sha256(registry),
    }


def _source_binding(source: VerifiedSource) -> dict[str, object]:
    return {
        "dataset_id": source.dataset_id,
        "logical_manifest_sha256": source.manifest_logical_sha256,
        "manifest_file_sha256": source.manifest_file_sha256,
        "object_sha256": source.object_sha256,
        "byte_length": source.byte_length,
        "license_status": source.license_status,
    }


def _builder_code_sha256() -> str:
    extractor_source = inspect.getsourcefile(build_game_fact_tables)
    if extractor_source is None:
        raise NFLExact153PublicationError("extractor source file is unavailable")
    return _canonical_sha256(
        {
            "builder_version": BUILDER_VERSION,
            "publication_source_sha256": _sha256_file(Path(__file__)),
            "extractor_source_sha256": _sha256_file(Path(extractor_source)),
        }
    )


def _stable_play_id(value: object, *, label: str) -> str:
    if value is None or value is pd.NA:
        raise NFLExact153PublicationError(f"{label} play_id is missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise NFLExact153PublicationError(f"{label} play_id is missing")
        if number.is_integer():
            return str(int(number))
    text = str(value).strip()
    if not text:
        raise NFLExact153PublicationError(f"{label} play_id is missing")
    return text


def _load_game_frames(
    pbp_source: VerifiedSource,
    participation_source: VerifiedSource,
    game_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equality-read one governed game and enforce raw source integrity."""

    try:
        pbp = pd.read_parquet(
            pbp_source.object_path,
            filters=[("game_id", "==", game_id)],
        )
        participation = pd.read_parquet(
            participation_source.object_path,
            filters=[("nflverse_game_id", "==", game_id)],
        )
    except Exception as exc:
        raise NFLExact153PublicationError(
            f"cannot equality-read frozen sources for {game_id}"
        ) from exc
    required_pbp = {"game_id", "play_id"}
    required_participation = {"nflverse_game_id", "play_id"}
    if pbp.empty or not required_pbp.issubset(pbp.columns):
        raise NFLExact153PublicationError(
            f"raw PBP must be nonempty for governed game: {game_id}"
        )
    if not required_participation.issubset(participation.columns):
        raise NFLExact153PublicationError(
            f"participation source lacks identity columns: {game_id}"
        )
    if set(pbp["game_id"].astype(str)) != {game_id}:
        raise NFLExact153PublicationError(
            f"PBP equality read leaked another game: {game_id}"
        )
    if (
        not participation.empty
        and set(participation["nflverse_game_id"].astype(str)) != {game_id}
    ):
        raise NFLExact153PublicationError(
            f"participation equality read leaked another game: {game_id}"
        )
    pbp_keys = [
        (game_id, _stable_play_id(value, label="PBP"))
        for value in pbp["play_id"]
    ]
    participation_keys = [
        (game_id, _stable_play_id(value, label="participation"))
        for value in participation["play_id"]
    ]
    if len(pbp_keys) != len(set(pbp_keys)):
        raise NFLExact153PublicationError(
            f"PBP has duplicate (game_id, play_id): {game_id}"
        )
    if len(participation_keys) != len(set(participation_keys)):
        raise NFLExact153PublicationError(
            f"participation has duplicate (game_id, play_id): {game_id}"
        )
    orphans = set(participation_keys).difference(pbp_keys)
    if orphans:
        raise NFLExact153PublicationError(
            f"participation has orphan play rows: {game_id}"
        )
    return pbp, participation


def _ordered_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise NFLExact153PublicationError(f"{name} is not a DataFrame")
    if frame.columns.duplicated().any():
        raise NFLExact153PublicationError(f"{name} has duplicate columns")
    expected = TABLE_SCHEMAS.get(name)
    if expected is None:
        raise NFLExact153PublicationError(f"unknown published table: {name}")
    if name == "canonical_factor_events" and frame.empty:
        raise NFLExact153PublicationError(
            "canonical_factor_events must be nonempty"
        )
    unexpected = set(frame.columns).difference(expected)
    missing = set(expected).difference(frame.columns)
    allowed_missing = (
        {"unit"} if name == "factor_event_players" else set()
    )
    if unexpected or (not frame.empty and missing.difference(allowed_missing)):
        raise NFLExact153PublicationError(f"{name} schema drifted")
    frame = frame.reindex(columns=list(expected))
    try:
        frame = frame.astype(TABLE_PANDAS_DTYPES[name])
    except (TypeError, ValueError) as exc:
        raise NFLExact153PublicationError(
            f"{name} values violate the fixed physical schema"
        ) from exc
    records = frame.to_dict(orient="records")
    order = sorted(
        range(len(records)),
        key=lambda index: _canonical_bytes(records[index]),
    )
    return frame.iloc[order].reset_index(drop=True)


def _normalize_tables(tables: GameFactTables) -> GameFactTables:
    normalized = {
        attribute: _ordered_frame(name, getattr(tables, attribute))
        for name, attribute in _TABLE_ATTRIBUTES
    }
    return GameFactTables(
        events=normalized["events"],
        tags=normalized["tags"],
        players=normalized["players"],
        availability=normalized["availability"],
        injury_evidence=normalized["injury_evidence"],
        factor_hits=normalized["factor_hits"],
        factor_coverage=normalized["factor_coverage"],
        reconciliation=normalized["reconciliation"],
        registry_sha256=tables.registry_sha256,
    )


def _canonical_frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            str(column): _canonical_value(row[column])
            for column in frame.columns
        }
        for row in frame.to_dict(orient="records")
    ]


def _validate_referential_integrity(
    *,
    game_id: str,
    pbp: pd.DataFrame,
    participation: pd.DataFrame,
    tables: GameFactTables,
) -> dict[str, int | str]:
    raw_ids = tuple(
        _stable_play_id(value, label="PBP") for value in pbp["play_id"]
    )
    event_key_columns = (
        "game_id",
        "event_id",
        "raw_play_id",
        "play_id",
        "atomic_information_episode_id",
    )
    if tables.events.loc[:, list(event_key_columns)].isna().any().any():
        raise NFLExact153PublicationError(
            f"parent event identity is null: {game_id}"
        )
    duplicate_event_id_rows = int(
        tables.events.duplicated(["event_id"], keep=False).sum()
    )
    duplicate_atomic_rows = int(
        tables.events.duplicated(
            ["game_id", "atomic_information_episode_id"], keep=False
        ).sum()
    )
    if duplicate_event_id_rows or duplicate_atomic_rows:
        raise NFLExact153PublicationError(
            f"parent event identity is duplicate: {game_id}"
        )
    raw_event_ids = tuple(tables.events["raw_play_id"].astype(str))
    event_records = tables.events.loc[:, list(event_key_columns)].to_dict(
        orient="records"
    )
    parent_by_event: dict[str, tuple[str, str]] = {}
    parent_by_raw: dict[str, tuple[str, str]] = {}
    for row in event_records:
        event_id = str(row["event_id"])
        raw_play_id = str(row["raw_play_id"])
        play_id = str(row["play_id"])
        atomic_id = str(row["atomic_information_episode_id"])
        if play_id != raw_play_id:
            raise NFLExact153PublicationError(
                f"parent event play/raw mapping mismatch: {game_id}"
            )
        parent_by_event[event_id] = (raw_play_id, atomic_id)
        parent_by_raw[raw_play_id] = (event_id, atomic_id)
    reconciliation_raw_ids = tuple(
        tables.reconciliation["raw_play_id"].astype(str)
    )
    if (
        set(tables.events["game_id"].astype(str)) != {game_id}
        or len(raw_event_ids) != len(set(raw_event_ids))
        or len(tables.events) != len(raw_ids)
        or set(raw_event_ids) != set(raw_ids)
    ):
        raise NFLExact153PublicationError(
            f"extractor must emit exactly one unique event per raw key: {game_id}"
        )
    reconciliation_keys = (
        "game_id",
        "raw_play_id",
        "event_id",
        "atomic_information_episode_id",
    )
    if (
        tables.reconciliation.loc[:, list(reconciliation_keys)]
        .isna()
        .any()
        .any()
        or tables.reconciliation.duplicated(
            ["game_id", "raw_play_id"], keep=False
        ).any()
        or tables.reconciliation.duplicated(
            list(reconciliation_keys), keep=False
        ).any()
    ):
        raise NFLExact153PublicationError(
            f"reconciliation has duplicate or null link keys: {game_id}"
        )
    if (
        set(tables.reconciliation["game_id"].astype(str)) != {game_id}
        or len(reconciliation_raw_ids) != len(set(reconciliation_raw_ids))
        or len(tables.reconciliation) != len(raw_ids)
        or set(reconciliation_raw_ids) != set(raw_ids)
        or not tables.reconciliation["raw_row_preserved"]
        .fillna(False)
        .eq(True)
        .all()
    ):
        raise NFLExact153PublicationError(
            f"extractor reconciliation has silent loss: {game_id}"
        )
    reconciliation_mismatches = 0
    for row in tables.reconciliation.loc[
        :, list(reconciliation_keys)
    ].to_dict(orient="records"):
        expected = parent_by_raw.get(str(row["raw_play_id"]))
        observed = (
            str(row["event_id"]),
            str(row["atomic_information_episode_id"]),
        )
        if expected != observed:
            reconciliation_mismatches += 1
    if reconciliation_mismatches:
        raise NFLExact153PublicationError(
            f"reconciliation parent mapping mismatch: {game_id}"
        )
    child_specs: tuple[
        tuple[str, pd.DataFrame, tuple[str, ...]], ...
    ] = (
        (
            "factor_event_tags",
            tables.tags,
            ("game_id", "event_id", "play_id", "tag"),
        ),
        (
            "factor_event_players",
            tables.players,
            (
                "game_id",
                "event_id",
                "play_id",
                "role",
                "player_id",
                "player_name",
            ),
        ),
        (
            "injury_evidence",
            tables.injury_evidence,
            (
                "game_id",
                "event_id",
                "play_id",
                "evidence_type",
                "status",
                "team",
                "jersey_number",
                "source_player_name",
                "player_id",
            ),
        ),
        (
            "factor_hits",
            tables.factor_hits,
            (
                "game_id",
                "event_id",
                "play_id",
                "factor_id",
                "factor_version",
            ),
        ),
    )
    child_link_mismatches = 0
    duplicate_child_rows = 0
    for name, frame, unique_columns in child_specs:
        duplicate_child_rows += int(
            frame.duplicated(list(unique_columns), keep=False).sum()
        )
        if frame.empty:
            continue
        if (
            frame[["game_id", "event_id", "play_id"]]
            .isna()
            .any()
            .any()
            or set(frame["game_id"].astype(str)) != {game_id}
        ):
            raise NFLExact153PublicationError(
                f"{name} has a null or cross-game link: {game_id}"
            )
        for row in frame[["event_id", "play_id"]].to_dict(orient="records"):
            parent = parent_by_event.get(str(row["event_id"]))
            if parent is None or parent[0] != str(row["play_id"]):
                child_link_mismatches += 1
    availability_unique = (
        "game_id",
        "team",
        "unit",
        "player_id",
        "availability_observation",
        "interval_start_event_id",
        "interval_end_event_id",
    )
    duplicate_child_rows += int(
        tables.availability.duplicated(
            list(availability_unique), keep=False
        ).sum()
    )
    if not tables.availability.empty:
        if (
            tables.availability[
                [
                    "game_id",
                    "interval_start_event_id",
                    "interval_end_event_id",
                ]
            ]
            .isna()
            .any()
            .any()
            or set(tables.availability["game_id"].astype(str)) != {game_id}
        ):
            raise NFLExact153PublicationError(
                f"player_availability_events has a null or cross-game link: {game_id}"
            )
        for column in ("interval_start_event_id", "interval_end_event_id"):
            child_link_mismatches += int(
                (
                    ~tables.availability[column]
                    .astype(str)
                    .isin(parent_by_event)
                ).sum()
            )
    duplicate_child_rows += int(
        tables.factor_coverage.duplicated(
            ["factor_id", "factor_version"], keep=False
        ).sum()
    )
    if duplicate_child_rows:
        raise NFLExact153PublicationError(
            f"child table has duplicate primary/link keys: {game_id}"
        )
    if child_link_mismatches:
        raise NFLExact153PublicationError(
            f"child event/play foreign key link mapping mismatch: {game_id}"
        )
    return {
        "raw_pbp_rows": len(pbp),
        "participation_rows": len(participation),
        "event_rows": len(tables.events),
        "reconciliation_rows": len(tables.reconciliation),
        "pbp_duplicate_key_rows": 0,
        "participation_duplicate_key_rows": 0,
        "participation_orphan_rows": 0,
        "raw_rows_silently_dropped": 0,
        "event_fk_orphan_rows": 0,
        "duplicate_event_id_rows": 0,
        "duplicate_atomic_episode_rows": 0,
        "reconciliation_mapping_mismatch_rows": 0,
        "duplicate_child_primary_rows": 0,
        "child_link_mapping_mismatch_rows": 0,
        "publication_gate": "PASS",
    }


def _parquet_bytes(name: str, frame: pd.DataFrame) -> tuple[bytes, str]:
    try:
        table = pa.Table.from_pandas(
            frame,
            schema=TABLE_ARROW_SCHEMAS[name],
            preserve_index=False,
            safe=True,
        )
    except (KeyError, TypeError, ValueError, pa.ArrowException) as exc:
        raise NFLExact153PublicationError(
            f"{name} cannot be encoded with its fixed Arrow schema"
        ) from exc
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )
    return (
        sink.getvalue().to_pybytes(),
        _sha256_bytes(table.schema.serialize().to_pybytes()),
    )


def _publish_table(
    *,
    output_root: Path,
    game_id: str,
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload, schema_fingerprint = _parquet_bytes(name, frame)
    object_sha = _sha256_bytes(payload)
    digest = object_sha.removeprefix("sha256:")
    relative = (
        Path(PUBLICATION_ID)
        / "single-game"
        / game_id
        / "objects"
        / "sha256"
        / digest[:2]
        / f"{digest}.parquet"
    )
    target = _resolve_under(output_root, relative, label=f"{name} object")
    _atomic_publish(target, payload)
    semantic_sha = _canonical_sha256(
        {
            "name": name,
            "schema_columns": list(frame.columns),
            "rows": _canonical_frame_records(frame),
        }
    )
    return {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "pandas_dtypes": {
            column: str(frame[column].dtype) for column in frame.columns
        },
        "schema_fingerprint": schema_fingerprint,
        "semantic_rows_sha256": semantic_sha,
    }


def _aggregate_counts(
    games: Sequence[PublishedGameFactBundle],
) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for game in games:
        for name, value in game.counts.items():
            aggregate[name] = aggregate.get(name, 0) + int(value)
    return dict(sorted(aggregate.items()))


def _authority_binding(authority: Exact153Authority) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "manifest_file_sha256": authority.manifest_file_sha256,
        "object_sha256": authority.object_sha256,
        "development_game_count": len(authority.development_game_ids),
        "final_holdout_game_count": len(authority.final_holdout_game_ids),
    }


def _publish_game_bundle(
    *,
    output_root: Path,
    game_id: str,
    authority: Exact153Authority,
    source_bindings: Mapping[str, object],
    registry_binding: Mapping[str, object],
    builder_code_sha256: str,
    tables: GameFactTables,
    audit: Mapping[str, int | str],
) -> PublishedGameFactBundle:
    descriptors = [
        _publish_table(
            output_root=output_root,
            game_id=game_id,
            name=name,
            frame=getattr(tables, attribute),
        )
        for name, attribute in _TABLE_ATTRIBUTES
    ]
    counts: dict[str, int] = {
        str(key): int(value)
        for key, value in audit.items()
        if isinstance(value, int)
    }
    counts.update(
        {
            f"{descriptor['name']}_rows": int(descriptor["row_count"])
            for descriptor in descriptors
        }
    )
    table_semantics = {
        str(descriptor["name"]): str(descriptor["semantic_rows_sha256"])
        for descriptor in descriptors
    }
    table_semantic_sha = _canonical_sha256(table_semantics)
    material: dict[str, object] = {
        "schema": SINGLE_GAME_MANIFEST_SCHEMA,
        "publication_id": PUBLICATION_ID,
        "experiment_id": EXPERIMENT_ID,
        "cohort": "development",
        "game_id": game_id,
        "claim_boundary": CLAIM_BOUNDARY,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "authority": _authority_binding(authority),
        "sources": dict(source_bindings),
        "factor_registry": dict(registry_binding),
        "builder_version": BUILDER_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "counts": dict(sorted(counts.items())),
        "referential_integrity": dict(audit),
        "table_semantic_sha256": table_semantic_sha,
        "tables": descriptors,
        "publication_gate": "PASS",
    }
    bundle_sha = _canonical_sha256(material)
    manifest = {**material, "bundle_sha256": bundle_sha}
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    digest = manifest_sha.removeprefix("sha256:")
    relative = (
        Path(PUBLICATION_ID)
        / "single-game"
        / game_id
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    manifest_path = _resolve_under(
        output_root, relative, label=f"{game_id} manifest"
    )
    _atomic_publish(manifest_path, manifest_bytes)
    return _verify_game_bundle(
        output_root=output_root,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        expected_game_id=game_id,
        authority=authority,
        source_bindings=source_bindings,
        registry_binding=registry_binding,
        builder_code_sha256=builder_code_sha256,
    )


def _verify_game_bundle(
    *,
    output_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_game_id: str,
    authority: Exact153Authority,
    source_bindings: Mapping[str, object],
    registry_binding: Mapping[str, object],
    builder_code_sha256: str,
) -> PublishedGameFactBundle:
    manifest_path = _require_v4_publication_path(
        output_root=output_root,
        candidate=manifest_path,
        expected_prefix=(
            PUBLICATION_ID,
            "single-game",
            expected_game_id,
            "manifests",
        ),
        label=f"{expected_game_id} fact manifest",
    )
    manifest, encoded = _read_json_object(
        manifest_path, label=f"{expected_game_id} fact manifest"
    )
    if _sha256_bytes(encoded) != expected_manifest_sha256:
        raise NFLExact153PublicationError(
            f"single-game manifest hash mismatch: {expected_game_id}"
        )
    bundle_sha = manifest.get("bundle_sha256")
    material = dict(manifest)
    material.pop("bundle_sha256", None)
    if (
        manifest.get("schema")
        != SINGLE_GAME_MANIFEST_SCHEMA
        or manifest.get("publication_id") != PUBLICATION_ID
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("game_id") != expected_game_id
        or manifest.get("cohort") != "development"
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("market_data_read") is not False
        or manifest.get("holdout_reaction_accessed") is not False
        or manifest.get("authority") != _authority_binding(authority)
        or manifest.get("sources") != dict(source_bindings)
        or manifest.get("factor_registry") != dict(registry_binding)
        or manifest.get("builder_version") != BUILDER_VERSION
        or manifest.get("builder_code_sha256") != builder_code_sha256
        or bundle_sha != _canonical_sha256(material)
    ):
        raise NFLExact153PublicationError(
            f"single-game manifest contract mismatch: {expected_game_id}"
        )
    descriptors = manifest.get("tables")
    if type(descriptors) is not list or [
        descriptor.get("name") if type(descriptor) is dict else None
        for descriptor in descriptors
    ] != [name for name, _ in _TABLE_ATTRIBUTES]:
        raise NFLExact153PublicationError(
            f"single-game table inventory mismatch: {expected_game_id}"
        )
    table_semantics: dict[str, str] = {}
    counts = manifest.get("counts")
    if type(counts) is not dict:
        raise NFLExact153PublicationError(
            f"single-game counts are absent: {expected_game_id}"
        )
    for descriptor in descriptors:
        assert type(descriptor) is dict
        name = str(descriptor["name"])
        object_path = _require_v4_publication_path(
            output_root=output_root,
            candidate=str(descriptor.get("object_path", "")),
            expected_prefix=(
                PUBLICATION_ID,
                "single-game",
                expected_game_id,
                "objects",
            ),
            label=f"{expected_game_id}.{name}",
            require_relative=True,
        )
        if (
            not object_path.is_file()
            or object_path.stat().st_size != descriptor.get("byte_length")
            or _sha256_file(object_path) != descriptor.get("object_sha256")
        ):
            raise NFLExact153PublicationError(
                f"published table object hash mismatch: {expected_game_id}.{name}"
            )
        try:
            frame = pd.read_parquet(object_path)
            parquet_schema = pq.read_schema(object_path)
        except Exception as exc:
            raise NFLExact153PublicationError(
                f"published table is unreadable: {expected_game_id}.{name}"
            ) from exc
        if (
            len(frame) != descriptor.get("row_count")
            or list(frame.columns) != descriptor.get("schema_columns")
            or {
                column: str(frame[column].dtype) for column in frame.columns
            }
            != descriptor.get("pandas_dtypes")
            or parquet_schema.remove_metadata() != TABLE_ARROW_SCHEMAS[name]
            or _sha256_bytes(parquet_schema.serialize().to_pybytes())
            != descriptor.get("schema_fingerprint")
            or pq.ParquetFile(object_path).metadata.num_rows
            != descriptor.get("row_count")
        ):
            raise NFLExact153PublicationError(
                f"published table schema/count mismatch: {expected_game_id}.{name}"
            )
        observed_semantic = _canonical_sha256(
            {
                "name": name,
                "schema_columns": list(frame.columns),
                "rows": _canonical_frame_records(frame),
            }
        )
        if observed_semantic != descriptor.get("semantic_rows_sha256"):
            raise NFLExact153PublicationError(
                f"published table semantic hash mismatch: {expected_game_id}.{name}"
            )
        table_semantics[name] = observed_semantic
    if manifest.get("table_semantic_sha256") != _canonical_sha256(table_semantics):
        raise NFLExact153PublicationError(
            f"combined table semantic hash mismatch: {expected_game_id}"
        )
    normalized_counts = {str(key): int(value) for key, value in counts.items()}
    for descriptor in descriptors:
        name = str(descriptor["name"])
        if normalized_counts.get(f"{name}_rows") != int(descriptor["row_count"]):
            raise NFLExact153PublicationError(
                f"published table count audit mismatch: {expected_game_id}.{name}"
            )
    return PublishedGameFactBundle(
        game_id=expected_game_id,
        bundle_sha256=str(bundle_sha),
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha256,
        counts=dict(sorted(normalized_counts.items())),
        table_semantic_sha256=dict(sorted(table_semantics.items())),
    )


def _publish_batch_index(
    *,
    output_root: Path,
    authority: Exact153Authority,
    games: Sequence[PublishedGameFactBundle],
    source_bindings: Mapping[str, object],
    registry_binding: Mapping[str, object],
    builder_code_sha256: str,
) -> PublishedExact153FactBatch:
    ordered_ids = tuple(game.game_id for game in games)
    if (
        len(games) != len(authority.development_game_ids)
        or len(set(ordered_ids)) != len(authority.development_game_ids)
        or ordered_ids != authority.development_game_ids
    ):
        raise NFLExact153PublicationError(
            "batch publication requires the exact development set in authority order"
        )
    manifest_paths = tuple(
        _require_v4_publication_path(
            output_root=output_root,
            candidate=game.manifest_path,
            expected_prefix=(
                PUBLICATION_ID,
                "single-game",
                game.game_id,
                "manifests",
            ),
            label=f"{game.game_id} batch child manifest",
        )
        .relative_to(output_root.resolve())
        .as_posix()
        for game in games
    )
    aggregate = _aggregate_counts(games)
    semantic_material = {
        "schema": SEMANTIC_BATCH_SCHEMA,
        "publication_id": PUBLICATION_ID,
        "experiment_id": EXPERIMENT_ID,
        "authority": _authority_binding(authority),
        "sources": dict(source_bindings),
        "factor_registry": dict(registry_binding),
        "builder_version": BUILDER_VERSION,
        "games": [
            {
                "game_id": game.game_id,
                "table_semantic_sha256": dict(game.table_semantic_sha256),
            }
            for game in games
        ],
    }
    semantic_batch_sha = _canonical_sha256(semantic_material)
    material: dict[str, object] = {
        "schema": BATCH_INDEX_SCHEMA,
        "publication_id": PUBLICATION_ID,
        "experiment_id": EXPERIMENT_ID,
        "cohort": "development",
        "game_count": len(games),
        "claim_boundary": CLAIM_BOUNDARY,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "authority": _authority_binding(authority),
        "sources": dict(source_bindings),
        "factor_registry": dict(registry_binding),
        "builder_version": BUILDER_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "aggregate_counts": aggregate,
        "semantic_batch_sha256": semantic_batch_sha,
        "games": [
            {
                "game_id": game.game_id,
                "bundle_sha256": game.bundle_sha256,
                "manifest_path": manifest_path,
                "manifest_sha256": game.manifest_sha256,
                "counts": dict(game.counts),
                "table_semantic_sha256": dict(game.table_semantic_sha256),
            }
            for game, manifest_path in zip(games, manifest_paths, strict=True)
        ],
        "publication_gate": "PASS",
    }
    batch_sha = _canonical_sha256(material)
    index = {**material, "batch_sha256": batch_sha}
    encoded = _canonical_bytes(index)
    index_sha = _sha256_bytes(encoded)
    digest = index_sha.removeprefix("sha256:")
    relative = (
        Path(PUBLICATION_ID)
        / "batches"
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.batch-index.json"
    )
    index_path = _resolve_under(output_root, relative, label="batch index")
    _atomic_publish(index_path, encoded)
    return PublishedExact153FactBatch(
        game_count=len(games),
        batch_sha256=batch_sha,
        semantic_batch_sha256=semantic_batch_sha,
        index_path=index_path,
        index_sha256=index_sha,
        games=tuple(games),
        aggregate_counts=aggregate,
    )


def publish_exact153_fact_batch(
    *,
    project_root: str | Path,
    output_root: str | Path,
    governance_manifest_path: str | Path | None = None,
    factor_registry_path: str | Path | None = None,
    workers: int = 1,
) -> PublishedExact153FactBatch:
    """Publish the governed development fact batch."""

    if type(workers) is not int or workers != 1:
        raise NFLExact153PublicationError("workers must equal 1")
    destination = Path(output_root).resolve()
    if LEGACY_PUBLICATION_ID in destination.parts:
        raise NFLExact153PublicationError(
            f"output_root cannot be the legacy {LEGACY_PUBLICATION_ID} namespace"
        )
    root = Path(project_root).resolve()
    authority = verify_exact153_authority(
        project_root=root,
        governance_manifest_path=governance_manifest_path,
    )
    registry, registry_binding = _verify_factor_registry(
        root, factor_registry_path
    )
    data_root = discover_main_checkout(root)
    pbp_source, participation_source = verify_frozen_2025_sources(data_root)
    if (
        pbp_source.dataset_id != "DS-NFLVERSE"
        or participation_source.dataset_id != "DS-NFLVERSE-PARTICIPATION"
    ):
        raise NFLExact153PublicationError("frozen source dataset identity mismatch")
    source_bindings = {
        "pbp": _source_binding(pbp_source),
        "participation": _source_binding(participation_source),
    }
    builder_code_sha = _builder_code_sha256()
    games: list[PublishedGameFactBundle] = []
    for game_id in authority.development_game_ids:
        authority.require_development(game_id)
        pbp, participation = _load_game_frames(
            pbp_source, participation_source, game_id
        )
        try:
            extracted = build_game_fact_tables(
                pbp,
                participation,
                factor_registry=registry,
                pbp_source_sha256=pbp_source.object_sha256,
                participation_source_sha256=participation_source.object_sha256,
            )
        except NFLFactExtractionError as exc:
            raise NFLExact153PublicationError(
                f"fact extraction failed: {game_id}"
            ) from exc
        if extracted.registry_sha256 != registry_binding["semantic_sha256"]:
            raise NFLExact153PublicationError(
                f"extractor registry semantic hash mismatch: {game_id}"
            )
        tables = _normalize_tables(extracted)
        audit = _validate_referential_integrity(
            game_id=game_id,
            pbp=pbp,
            participation=participation,
            tables=tables,
        )
        games.append(
            _publish_game_bundle(
                output_root=destination,
                game_id=game_id,
                authority=authority,
                source_bindings=source_bindings,
                registry_binding=registry_binding,
                builder_code_sha256=builder_code_sha,
                tables=tables,
                audit=audit,
            )
        )
        del pbp, participation, extracted, tables
        gc.collect()
    return _publish_batch_index(
        output_root=destination,
        authority=authority,
        games=games,
        source_bindings=source_bindings,
        registry_binding=registry_binding,
        builder_code_sha256=builder_code_sha,
    )


__all__ = [
    "BATCH_INDEX_SCHEMA",
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "EXPERIMENT_ID",
    "PUBLICATION_ID",
    "SEMANTIC_BATCH_SCHEMA",
    "SINGLE_GAME_MANIFEST_SCHEMA",
    "Exact153Authority",
    "NFLExact153PublicationError",
    "PublishedExact153FactBatch",
    "PublishedGameFactBundle",
    "TABLE_SCHEMAS",
    "publish_exact153_fact_batch",
    "verify_exact153_authority",
]
