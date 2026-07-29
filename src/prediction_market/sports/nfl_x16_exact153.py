"""Governed, streaming publication of the NFL X-16 exact-153 fact batch."""

from __future__ import annotations

import gc
import hashlib
import inspect
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

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
    "registries/factors/nfl_factor_registry_v3_draft.json"
)
BUILDER_VERSION: Final[str] = "nfl_x16_exact153_fact_publication_v1"
CLAIM_BOUNDARY: Final[str] = (
    "DEVELOPMENT_SPORTS_FACTS_ONLY; no market reaction, holdout reaction, "
    "causality, execution, or alpha claim"
)

TABLE_SCHEMAS: Final[Mapping[str, tuple[str, ...]]] = {
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
_EVENT_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "event_id",
    "raw_play_id",
    "atomic_information_episode_id",
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
        or authority_object.get("status") != "FROZEN"
    ):
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
    if (
        registry.get("schema") != "NFLFactorRegistryV3"
        or registry.get("status") != "DRAFT_EXPERT_REVIEW"
        or type(registry.get("version")) is not str
        or type(registry.get("factors")) is not list
    ):
        raise NFLExact153PublicationError("factor registry schema/status mismatch")
    return registry, {
        "schema": registry["schema"],
        "status": registry["status"],
        "version": registry["version"],
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
    if expected is not None:
        unexpected = set(frame.columns).difference(expected)
        if unexpected:
            raise NFLExact153PublicationError(f"{name} schema drifted")
        frame = frame.reindex(columns=list(expected))
    elif name == "canonical_factor_events":
        missing = set(_EVENT_REQUIRED_COLUMNS).difference(frame.columns)
        if frame.empty or missing:
            raise NFLExact153PublicationError(
                "canonical_factor_events schema is incomplete"
            )
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
    event_ids = set(tables.events["event_id"].astype(str))
    raw_event_ids = tuple(tables.events["raw_play_id"].astype(str))
    atomic_ids = tuple(
        tables.events["atomic_information_episode_id"].astype(str)
    )
    reconciliation_raw_ids = tuple(
        tables.reconciliation["raw_play_id"].astype(str)
    )
    if (
        set(tables.events["game_id"].astype(str)) != {game_id}
        or len(raw_event_ids) != len(set(raw_event_ids))
        or len(atomic_ids) != len(set(atomic_ids))
        or len(tables.events) != len(raw_ids)
        or set(raw_event_ids) != set(raw_ids)
    ):
        raise NFLExact153PublicationError(
            f"extractor must emit exactly one unique event per raw key: {game_id}"
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
    foreign_key_orphans = 0
    for name, frame in (
        ("factor_event_tags", tables.tags),
        ("factor_event_players", tables.players),
        ("injury_evidence", tables.injury_evidence),
        ("factor_hits", tables.factor_hits),
        ("sports_row_reconciliation", tables.reconciliation),
    ):
        if not frame.empty:
            if set(frame["game_id"].astype(str)) != {game_id}:
                raise NFLExact153PublicationError(
                    f"{name} leaked another game: {game_id}"
                )
            foreign_key_orphans += int(
                (~frame["event_id"].astype(str).isin(event_ids)).sum()
            )
    if not tables.availability.empty:
        if set(tables.availability["game_id"].astype(str)) != {game_id}:
            raise NFLExact153PublicationError(
                f"player_availability_events leaked another game: {game_id}"
            )
        for column in ("interval_start_event_id", "interval_end_event_id"):
            foreign_key_orphans += int(
                (~tables.availability[column].astype(str).isin(event_ids)).sum()
            )
    if foreign_key_orphans:
        raise NFLExact153PublicationError(
            f"child event foreign key violation: {game_id}"
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
        "publication_gate": "PASS",
    }


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    table = pa.Table.from_pandas(frame, preserve_index=False)
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
    payload, schema_fingerprint = _parquet_bytes(frame)
    object_sha = _sha256_bytes(payload)
    digest = object_sha.removeprefix("sha256:")
    relative = (
        Path("single-game")
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
        "schema": "nfl_x16_exact153_single_game_fact_manifest_v1",
        "experiment_id": "X-16",
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
        Path("single-game")
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
        != "nfl_x16_exact153_single_game_fact_manifest_v1"
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
        object_path = _resolve_under(
            output_root,
            str(descriptor.get("object_path", "")),
            label=f"{expected_game_id}.{name}",
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
        except Exception as exc:
            raise NFLExact153PublicationError(
                f"published table is unreadable: {expected_game_id}.{name}"
            ) from exc
        if (
            len(frame) != descriptor.get("row_count")
            or list(frame.columns) != descriptor.get("schema_columns")
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
    aggregate = _aggregate_counts(games)
    semantic_material = {
        "authority": _authority_binding(authority),
        "sources": dict(source_bindings),
        "factor_registry": dict(registry_binding),
        "builder_version": BUILDER_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "aggregate_counts": aggregate,
        "games": [
            {
                "game_id": game.game_id,
                "bundle_sha256": game.bundle_sha256,
                "table_semantic_sha256": dict(game.table_semantic_sha256),
            }
            for game in games
        ],
    }
    semantic_batch_sha = _canonical_sha256(semantic_material)
    material: dict[str, object] = {
        "schema": "nfl_x16_exact153_fact_batch_index_v1",
        "experiment_id": "X-16",
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
                "manifest_path": game.manifest_path.relative_to(
                    output_root
                ).as_posix(),
                "manifest_sha256": game.manifest_sha256,
                "counts": dict(game.counts),
                "table_semantic_sha256": dict(game.table_semantic_sha256),
            }
            for game in games
        ],
        "publication_gate": "PASS",
    }
    batch_sha = _canonical_sha256(material)
    index = {**material, "batch_sha256": batch_sha}
    encoded = _canonical_bytes(index)
    index_sha = _sha256_bytes(encoded)
    digest = index_sha.removeprefix("sha256:")
    relative = (
        Path("batches")
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
    destination = Path(output_root).resolve()
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
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "Exact153Authority",
    "NFLExact153PublicationError",
    "PublishedExact153FactBatch",
    "PublishedGameFactBundle",
    "TABLE_SCHEMAS",
    "publish_exact153_fact_batch",
    "verify_exact153_authority",
]
