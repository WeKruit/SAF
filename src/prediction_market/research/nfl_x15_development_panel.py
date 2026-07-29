"""Verified exact-153 inputs and per-game Stage B panel publication.

The historical development artifacts support two deliberately separate target
contracts:

* ``HistoricalTradesOnlyProbabilityPanelV1`` is a retrospective source-time
  diagnostic.  Direction uses the ADR-0006 fixed cross-venue materiality
  threshold of one probability point.  It does not assert a venue tick or
  continuous market availability.
* ``VenueReactionPanelV3`` is confirmatory and is built only when explicit
  historical per-contract tick-rule and market-continuity evidence is supplied.

The publisher verifies every manifest and object by SHA-256 and processes one
game at a time.  It never opens final-holdout reaction data or fetches upstream
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.research import nfl_x15_landmarks as _landmarks


DIAGNOSTIC_SCHEMA = "HistoricalTradesOnlyProbabilityPanelV1"
DIAGNOSTIC_CLAIM_BOUNDARY = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)
DIAGNOSTIC_TARGET_CONTRACT = "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
CONFIRMATORY_CLAIM_BOUNDARY = _landmarks.CLAIM_BOUNDARY
DIRECTION_THRESHOLD_PROBABILITY = 0.01
DIRECTION_THRESHOLD_SEMANTICS = (
    "FIXED_CROSS_VENUE_RESEARCH_MATERIALITY_NOT_TICK"
)
MARKET_CONTINUITY_SUPPORT = "UNKNOWN"
VENUE_TICK_SUPPORT = "UNSUPPORTED"
BUILDER_VERSION = "nfl-x15-development-panel-v1"

DEFAULT_FACTS_BATCH = Path(
    "artifacts/market-observation/nfl/x13/exact-153-facts-v3/"
    "batches/manifests/sha256/db/"
    "db2fb125ea4d5e7844e22b27d6643e6f1c1ddc48db527d347ca789faacb38acd"
    ".batch-index.json"
)
DEFAULT_FACTS_BATCH_FILE_SHA256 = (
    "sha256:db2fb125ea4d5e7844e22b27d6643e6f1c1ddc48db527d347ca789faacb38acd"
)
DEFAULT_STAGE_A_BATCH = Path(
    "artifacts/market-observation/nfl/x15/stage-a-reference-v1/"
    "batches/manifests/sha256/50/"
    "50040cc83d44f5a62d70cbac92d2aa4d8064bbd4b9c3b36f79fe578bd72a2182"
    ".batch-index.json"
)
DEFAULT_STAGE_A_BATCH_FILE_SHA256 = (
    "sha256:50040cc83d44f5a62d70cbac92d2aa4d8064bbd4b9c3b36f79fe578bd72a2182"
)
DEFAULT_MARKET_BATCH = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-development-market/kalshi-native-time-v3/exact-153/"
    "batches/manifests/sha256/b2/"
    "b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341"
    ".batch-index.json"
)
DEFAULT_MARKET_BATCH_FILE_SHA256 = (
    "sha256:b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341"
)
DEFAULT_AUTHORITY_MANIFEST = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-registry/manifests/"
    "545f7e082055cd5070d907da9eacbdb735dd14351c19c5a83d912764bb595a37"
    ".manifest.json"
)
DEFAULT_AUTHORITY_MANIFEST_FILE_SHA256 = (
    "sha256:545f7e082055cd5070d907da9eacbdb735dd14351c19c5a83d912764bb595a37"
)
DEFAULT_AUTHORITY_OBJECT = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-registry/objects/sha256/22/"
    "226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
    ".json"
)
DEFAULT_AUTHORITY_OBJECT_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "historical-trades-only-development-panel-v1"
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIAGNOSTIC_PANEL_GRAIN = (
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
)
_MARKET_CONTINUITY_COLUMNS = {
    "game_id",
    "atomic_information_episode_id",
    "continuity_verified_until_utc",
    "suspension_time_utc",
    "continuity_gap_time_utc",
}


class DevelopmentPanelError(RuntimeError):
    """A frozen input or per-game publication invariant failed."""


@dataclass(frozen=True, slots=True)
class _DiagnosticMark:
    status: str
    trade_ids: tuple[str, ...]
    trade_id_set_sha256: str | None
    source_time: pd.Timestamp | None
    price: float
    staleness_seconds: float
    observation_count: int
    observed_size: float
    semantics: str

    @property
    def observed(self) -> bool:
        return self.status == "OBSERVED"


@dataclass(frozen=True, slots=True)
class DevelopmentSourceSpec:
    """Pinned source artifacts for one complete development cohort."""

    facts_batch_path: Path
    facts_batch_file_sha256: str
    stage_a_batch_path: Path
    stage_a_batch_file_sha256: str
    market_batch_path: Path
    market_batch_file_sha256: str
    authority_manifest_path: Path
    authority_manifest_file_sha256: str
    authority_object_path: Path
    authority_object_sha256: str
    expected_game_count: int = 153


@dataclass(frozen=True, slots=True)
class VenueConfirmatoryEvidence:
    """Historical evidence required by the confirmatory V3 builder."""

    venue: str
    tick_rule_id: str
    tick_rules: pd.DataFrame
    market_continuity: pd.DataFrame
    tick_rule_source_sha256: str
    continuity_source_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedBatch:
    path: Path
    file_sha256: str
    root: Path
    document: Mapping[str, Any]
    games: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class VerifiedDevelopmentSources:
    """Small verified descriptors; no per-game Parquet is retained."""

    project_root: Path
    spec: DevelopmentSourceSpec
    facts: _VerifiedBatch
    stage_a: _VerifiedBatch
    market: _VerifiedBatch
    cohort_metadata: pd.DataFrame
    cohort_authority_sha256: str
    cohort_mapping_sha256: str


@dataclass(frozen=True, slots=True)
class PublishedDevelopmentGame:
    game_id: str
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    confirmatory_venue_count: int
    diagnostic_venue_count: int


@dataclass(frozen=True, slots=True)
class DevelopmentPanelPublication:
    output_root: Path
    batch_manifest_path: Path
    batch_manifest_sha256: str
    batch_sha256: str
    game_count: int
    cohort_authority_sha256: str
    cohort_mapping_sha256: str
    games: tuple[PublishedDevelopmentGame, ...]


def default_development_source_spec() -> DevelopmentSourceSpec:
    """Return the exact frozen development source pins."""

    return DevelopmentSourceSpec(
        facts_batch_path=DEFAULT_FACTS_BATCH,
        facts_batch_file_sha256=DEFAULT_FACTS_BATCH_FILE_SHA256,
        stage_a_batch_path=DEFAULT_STAGE_A_BATCH,
        stage_a_batch_file_sha256=DEFAULT_STAGE_A_BATCH_FILE_SHA256,
        market_batch_path=DEFAULT_MARKET_BATCH,
        market_batch_file_sha256=DEFAULT_MARKET_BATCH_FILE_SHA256,
        authority_manifest_path=DEFAULT_AUTHORITY_MANIFEST,
        authority_manifest_file_sha256=DEFAULT_AUTHORITY_MANIFEST_FILE_SHA256,
        authority_object_path=DEFAULT_AUTHORITY_OBJECT,
        authority_object_sha256=DEFAULT_AUTHORITY_OBJECT_SHA256,
        expected_game_count=153,
    )


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise DevelopmentPanelError(f"{label} must be a SHA-256 identifier")
    return value


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, set):
        return [_canonical_value(child) for child in sorted(value, key=repr)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical_value(child) for child in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return str(value)


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _canonical_value(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise DevelopmentPanelError("value is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentPanelError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DevelopmentPanelError(f"non-JSON number: {value}")


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_reject_constant,
        )
    except DevelopmentPanelError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentPanelError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise DevelopmentPanelError(f"{label} must be a JSON object")
    return value


def _resolve_under(root: Path, value: str | Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise DevelopmentPanelError(f"{label} contains '..'")
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (resolved_root / candidate).resolve()
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise DevelopmentPanelError(f"{label} escapes project root") from exc
    return resolved


def _stable_read(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_length: int | None = None,
) -> bytes:
    expected = _require_sha256(expected_sha256, label=f"{label}.sha256")
    try:
        before = os.stat(path, follow_symlinks=False)
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        if expected_length is not None and before.st_size != expected_length:
            raise OSError("byte length mismatch")
        with path.open("rb") as stream:
            payload = stream.read()
        after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise DevelopmentPanelError(f"{label} is not a stable regular file") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise DevelopmentPanelError(f"{label} changed while reading")
    if _sha256_bytes(payload) != expected:
        raise DevelopmentPanelError(f"{label} SHA-256 mismatch")
    return payload


def _verify_semantic_hash(
    document: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> None:
    expected = _require_sha256(document.get(field), label=f"{label}.{field}")
    material = dict(document)
    material.pop(field)
    if _canonical_sha256(material) != expected:
        raise DevelopmentPanelError(f"{label} {field} mismatch")


def _batch_root(path: Path) -> Path:
    try:
        return path.parents[4]
    except IndexError as exc:
        raise DevelopmentPanelError("batch path is not under a dataset root") from exc


def _verify_batch(
    *,
    project_root: Path,
    path: Path,
    expected_file_sha256: str,
    expected_schema: str,
    expected_game_count: int,
    label: str,
    market_style: bool,
) -> _VerifiedBatch:
    resolved = _resolve_under(project_root, path, label=f"{label} batch")
    payload = _stable_read(
        resolved,
        label=f"{label} batch",
        expected_sha256=expected_file_sha256,
    )
    document = _strict_json(payload, label=f"{label} batch")
    _verify_semantic_hash(document, field="batch_sha256", label=f"{label} batch")
    if (
        document.get("schema") != expected_schema
        or document.get("cohort") != "development"
        or document.get("game_count") != expected_game_count
        or document.get("publication_gate") != "PASS"
    ):
        raise DevelopmentPanelError(f"{label} batch contract mismatch")
    if market_style:
        if document.get("final_holdout_access") != "CLOSED":
            raise DevelopmentPanelError(f"{label} final holdout is not closed")
    elif (
        document.get("holdout_reaction_accessed") is not False
        or document.get("market_data_read") is not False
    ):
        raise DevelopmentPanelError(f"{label} violates no-holdout/no-market input")
    entries = document.get("games")
    if type(entries) is not list or len(entries) != expected_game_count:
        raise DevelopmentPanelError(f"{label} batch game list is incomplete")
    by_game: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if type(entry) is not dict:
            raise DevelopmentPanelError(f"{label} game descriptor is invalid")
        game_id = entry.get("game_id")
        if type(game_id) is not str or not game_id.strip() or game_id in by_game:
            raise DevelopmentPanelError(f"{label} game identity is invalid")
        _require_sha256(
            entry.get("manifest_sha256"),
            label=f"{label}.{game_id}.manifest_sha256",
        )
        _require_sha256(
            entry.get("bundle_sha256"),
            label=f"{label}.{game_id}.bundle_sha256",
        )
        if type(entry.get("manifest_path")) is not str:
            raise DevelopmentPanelError(f"{label}.{game_id}.manifest_path is invalid")
        by_game[game_id] = MappingProxyType(entry)
    return _VerifiedBatch(
        path=resolved,
        file_sha256=expected_file_sha256,
        root=_batch_root(resolved),
        document=MappingProxyType(document),
        games=MappingProxyType(by_game),
    )


def _verify_authority(
    *,
    project_root: Path,
    spec: DevelopmentSourceSpec,
) -> tuple[pd.DataFrame, str]:
    manifest_path = _resolve_under(
        project_root,
        spec.authority_manifest_path,
        label="cohort authority manifest",
    )
    manifest_payload = _stable_read(
        manifest_path,
        label="cohort authority manifest",
        expected_sha256=spec.authority_manifest_file_sha256,
    )
    manifest = _strict_json(manifest_payload, label="cohort authority manifest")
    object_path = _resolve_under(
        project_root,
        spec.authority_object_path,
        label="cohort authority object",
    )
    object_payload = _stable_read(
        object_path,
        label="cohort authority object",
        expected_sha256=spec.authority_object_sha256,
        expected_length=manifest.get("byte_length"),
    )
    authority = _strict_json(object_payload, label="cohort authority object")
    expected_holdout_count = 81 if spec.expected_game_count == 153 else 0
    if (
        manifest.get("object_sha256") != spec.authority_object_sha256
        or manifest.get("development_game_count") != spec.expected_game_count
        or manifest.get("final_holdout_game_count") != expected_holdout_count
        or authority.get("schema") != "nfl_factor_expansion_registry_v2"
    ):
        raise DevelopmentPanelError("cohort authority contract mismatch")
    split_lock = authority.get("split_lock")
    if type(split_lock) is not dict:
        raise DevelopmentPanelError("cohort authority split_lock is missing")
    assignments = split_lock.get("game_assignments")
    if type(assignments) is not list:
        raise DevelopmentPanelError("cohort authority assignments are missing")
    rows: list[dict[str, object]] = []
    for assignment in assignments:
        if type(assignment) is not dict:
            raise DevelopmentPanelError("cohort assignment is invalid")
        if assignment.get("cohort") != "development":
            continue
        game_id = assignment.get("game_id")
        week = assignment.get("week")
        if (
            type(game_id) is not str
            or not game_id.strip()
            or type(week) is not int
            or not 1 <= week <= 12
        ):
            raise DevelopmentPanelError("development cohort assignment is invalid")
        rows.append(
            {
                "game_id": game_id,
                "nfl_week": week,
                "cohort": "development",
                "authority_sha256": spec.authority_object_sha256,
            }
        )
    metadata = pd.DataFrame(rows)
    if (
        len(metadata) != spec.expected_game_count
        or metadata["game_id"].nunique() != spec.expected_game_count
    ):
        raise DevelopmentPanelError("authority development mapping is incomplete")
    mapping_material = [
        {
            "cohort": "development",
            "game_id": str(row["game_id"]),
            "nfl_week": int(row["nfl_week"]),
        }
        for row in metadata.sort_values("game_id", kind="mergesort").to_dict(
            "records"
        )
    ]
    mapping_sha = _sha256_bytes(
        json.dumps(
            mapping_material,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return (
        metadata.sort_values("game_id", kind="mergesort").reset_index(drop=True),
        mapping_sha,
    )


def verify_development_sources(
    *,
    project_root: Path,
    source_spec: DevelopmentSourceSpec | None = None,
) -> VerifiedDevelopmentSources:
    """Verify the three exact development batches and frozen week authority."""

    project_root = project_root.resolve()
    spec = source_spec or default_development_source_spec()
    if type(spec.expected_game_count) is not int or spec.expected_game_count <= 0:
        raise DevelopmentPanelError("expected_game_count must be positive")
    facts = _verify_batch(
        project_root=project_root,
        path=spec.facts_batch_path,
        expected_file_sha256=spec.facts_batch_file_sha256,
        expected_schema="nfl_x13_exact153_fact_batch_index_v1",
        expected_game_count=spec.expected_game_count,
        label="X13 facts",
        market_style=False,
    )
    stage_a = _verify_batch(
        project_root=project_root,
        path=spec.stage_a_batch_path,
        expected_file_sha256=spec.stage_a_batch_file_sha256,
        expected_schema="nfl_x15_stage_a_batch_index_v1",
        expected_game_count=spec.expected_game_count,
        label="Stage A",
        market_style=False,
    )
    market = _verify_batch(
        project_root=project_root,
        path=spec.market_batch_path,
        expected_file_sha256=spec.market_batch_file_sha256,
        expected_schema="nfl_expansion_development_market_batch_index_v1",
        expected_game_count=spec.expected_game_count,
        label="market",
        market_style=True,
    )
    metadata, mapping_sha = _verify_authority(
        project_root=project_root,
        spec=spec,
    )
    expected_games = frozenset(metadata["game_id"].astype(str))
    for label, batch in (
        ("X13 facts", facts),
        ("Stage A", stage_a),
        ("market", market),
    ):
        if frozenset(batch.games) != expected_games:
            raise DevelopmentPanelError(
                f"{label} game set differs from verified authority"
            )
    return VerifiedDevelopmentSources(
        project_root=project_root,
        spec=spec,
        facts=facts,
        stage_a=stage_a,
        market=market,
        cohort_metadata=metadata,
        cohort_authority_sha256=spec.authority_object_sha256,
        cohort_mapping_sha256=mapping_sha,
    )


def _verify_game_manifest(
    *,
    batch: _VerifiedBatch,
    descriptor: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], Path]:
    game_id = str(descriptor["game_id"])
    manifest_path = _resolve_under(
        batch.root,
        str(descriptor["manifest_path"]),
        label=f"{label}.{game_id}.manifest",
    )
    payload = _stable_read(
        manifest_path,
        label=f"{label}.{game_id}.manifest",
        expected_sha256=str(descriptor["manifest_sha256"]),
    )
    manifest = _strict_json(payload, label=f"{label}.{game_id}.manifest")
    if (
        manifest.get("game_id") != game_id
        or manifest.get("cohort") != "development"
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("holdout_reaction_accessed") not in {False, None}
        or manifest.get("bundle_sha256") != descriptor.get("bundle_sha256")
    ):
        raise DevelopmentPanelError(f"{label}.{game_id} manifest contract mismatch")
    _verify_semantic_hash(
        manifest,
        field="bundle_sha256",
        label=f"{label}.{game_id}.manifest",
    )
    return manifest, manifest_path


def _table_descriptor(
    manifest: Mapping[str, Any],
    *,
    table_name: str,
    market_style: bool,
    label: str,
) -> Mapping[str, Any]:
    if market_style:
        stages = manifest.get("stages")
        descriptor = stages.get(table_name) if type(stages) is dict else None
    else:
        tables = manifest.get("tables")
        matches = (
            [
                row
                for row in tables
                if type(row) is dict and row.get("name") == table_name
            ]
            if type(tables) is list
            else []
        )
        descriptor = matches[0] if len(matches) == 1 else None
    if type(descriptor) is not dict:
        raise DevelopmentPanelError(f"{label}.{table_name} descriptor is missing")
    return descriptor


def _read_verified_table(
    *,
    batch: _VerifiedBatch,
    manifest: Mapping[str, Any],
    table_name: str,
    market_style: bool,
    label: str,
) -> pd.DataFrame:
    descriptor = _table_descriptor(
        manifest,
        table_name=table_name,
        market_style=market_style,
        label=label,
    )
    object_sha = _require_sha256(
        descriptor.get("object_sha256"),
        label=f"{label}.{table_name}.object_sha256",
    )
    byte_length = descriptor.get("byte_length")
    row_count = descriptor.get("row_count")
    schema_columns = descriptor.get("schema_columns")
    if (
        type(descriptor.get("object_path")) is not str
        or type(byte_length) is not int
        or byte_length <= 0
        or type(row_count) is not int
        or row_count < 0
    ):
        raise DevelopmentPanelError(f"{label}.{table_name} descriptor is invalid")
    object_path = _resolve_under(
        batch.root,
        str(descriptor["object_path"]),
        label=f"{label}.{table_name}.object",
    )
    _stable_read(
        object_path,
        label=f"{label}.{table_name}.object",
        expected_sha256=object_sha,
        expected_length=byte_length,
    )
    try:
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != row_count:
            raise DevelopmentPanelError(
                f"{label}.{table_name} Parquet row count mismatch"
            )
        frame = parquet.read().to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise DevelopmentPanelError(
            f"{label}.{table_name} is not readable Parquet"
        ) from exc
    if type(schema_columns) is list and list(frame.columns) != schema_columns:
        raise DevelopmentPanelError(f"{label}.{table_name} schema mismatch")
    return frame


def _adapt_market(
    *,
    game_id: str,
    home_team: str,
    observations: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_inventory = {
        "game_id",
        "logical_market_id",
        "outcome",
        "venue",
        "raw_contract_id",
        "family",
        "period",
        "kind",
        "analysis_eligible",
    }
    required_observations = {
        "game_id",
        "observation_id",
        "venue",
        "logical_market_id",
        "outcome",
        "kind",
        "native_source_time_utc",
        "price",
        "size",
        "provenance",
    }
    missing_inventory = required_inventory - set(inventory.columns)
    missing_observations = required_observations - set(observations.columns)
    if missing_inventory or missing_observations:
        raise DevelopmentPanelError(
            "market input missing columns: "
            f"inventory={sorted(missing_inventory)}, "
            f"observations={sorted(missing_observations)}"
        )
    scoped_inventory = inventory.loc[
        inventory["game_id"].astype(str).eq(game_id)
        & inventory["family"].astype(str).eq("moneyline")
        & inventory["period"].astype(str).eq("full_game")
        & inventory["kind"].astype(str).eq("primitive")
        & inventory["analysis_eligible"].eq(True)  # noqa: E712
    ].copy()
    identity = ["game_id", "venue", "logical_market_id", "outcome"]
    if scoped_inventory.empty or scoped_inventory.duplicated(identity).any():
        raise DevelopmentPanelError(
            f"{game_id} contract inventory identity is missing or ambiguous"
        )
    if scoped_inventory["raw_contract_id"].isna().any():
        raise DevelopmentPanelError(f"{game_id} raw contract identity is missing")
    scoped_inventory["raw_contract_id"] = scoped_inventory[
        "raw_contract_id"
    ].astype(str)
    if scoped_inventory["raw_contract_id"].str.strip().eq("").any():
        raise DevelopmentPanelError(f"{game_id} raw contract identity is empty")
    home_counts = (
        scoped_inventory.assign(
            _is_home=scoped_inventory["outcome"].astype(str).eq(home_team)
        )
        .groupby("venue", sort=True)["_is_home"]
        .sum()
    )
    if home_counts.empty or not home_counts.eq(1).all():
        raise DevelopmentPanelError(
            f"{game_id} requires exactly one home contract per venue"
        )

    scoped_observations = observations.loc[
        observations["game_id"].astype(str).eq(game_id)
        & observations["kind"].astype(str).eq("trade")
        & observations["provenance"].astype(str).eq("observed")
    ].copy()
    joined = scoped_observations.merge(
        scoped_inventory.loc[:, [*identity, "raw_contract_id"]],
        on=identity,
        how="left",
        validate="many_to_one",
        indicator="_contract_join",
    )
    if not joined["_contract_join"].eq("both").all():
        raise DevelopmentPanelError(
            f"{game_id} observations do not have exact outcome-token identity"
        )
    kalshi = joined["venue"].astype(str).eq("kalshi")
    if (
        "raw_market_id" in joined.columns
        and not joined.loc[kalshi, "raw_market_id"]
        .astype(str)
        .eq(joined.loc[kalshi, "raw_contract_id"].astype(str))
        .all()
    ):
        raise DevelopmentPanelError(
            f"{game_id} Kalshi ticker differs from exact raw_contract_id"
        )
    market_rows = pd.DataFrame(
        {
            "trade_id": joined["observation_id"].astype(str),
            "game_id": joined["game_id"].astype(str),
            "venue": joined["venue"].astype(str),
            "contract_id": joined["raw_contract_id"].astype(str),
            "source_time_utc": joined["native_source_time_utc"],
            "price": joined["price"],
            "size": joined["size"],
            "kind": "trade",
            "provenance": "observed",
        }
    )
    if market_rows["trade_id"].duplicated().any():
        raise DevelopmentPanelError(f"{game_id} observation_id is not unique")
    contract_metadata = pd.DataFrame(
        {
            "game_id": scoped_inventory["game_id"].astype(str),
            "venue": scoped_inventory["venue"].astype(str),
            "contract_id": scoped_inventory["raw_contract_id"].astype(str),
            "outcome_team": scoped_inventory["outcome"].astype(str),
            "home_team": home_team,
            "market_family": "moneyline",
            "contract_role": np.where(
                scoped_inventory["outcome"].astype(str).eq(home_team),
                "ACTUAL_HOME_OUTCOME",
                "DERIVED_AWAY_COMPLEMENT",
            ),
        }
    )
    identity_audit = scoped_inventory.loc[
        :, ["game_id", "venue", "logical_market_id", "outcome", "raw_contract_id"]
    ].rename(columns={"raw_contract_id": "exact_contract_id"})
    identity_audit["mapping_key"] = (
        identity_audit["venue"].astype(str)
        + "|"
        + identity_audit["logical_market_id"].astype(str)
        + "|"
        + identity_audit["outcome"].astype(str)
    )
    identity_audit["identity_status"] = "EXACT"
    identity_audit["raw_market_id_used_as_contract_identity"] = False
    return (
        market_rows.sort_values(
            ["venue", "contract_id", "source_time_utc", "trade_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        contract_metadata.sort_values(
            ["venue", "contract_id"], kind="mergesort"
        ).reset_index(drop=True),
        identity_audit.sort_values(
            ["venue", "logical_market_id", "outcome"], kind="mergesort"
        ).reset_index(drop=True),
    )


def _sports_continuity(
    eligible_facts: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_id",
        "atomic_information_episode_id",
        "source_interval_start",
        "source_interval_end",
    }
    if not required.issubset(eligible_facts.columns) or eligible_facts.empty:
        raise DevelopmentPanelError("eligible facts cannot build sports continuity")
    facts = eligible_facts.copy()
    for column in ("source_interval_start", "source_interval_end"):
        facts[column] = pd.to_datetime(facts[column], utc=True, errors="raise")
    game_end = facts["source_interval_end"].max()
    facts = facts.sort_values(
        ["source_interval_start", "atomic_information_episode_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    facts["next_salient_event_time_utc"] = facts["source_interval_start"].shift(-1)
    facts["game_end_time_utc"] = game_end
    facts["sports_continuity_source"] = "VERIFIED_FINALIZED_SPORTS_SOURCE_INTERVALS"
    return facts.loc[
        :,
        [
            "game_id",
            "atomic_information_episode_id",
            "next_salient_event_time_utc",
            "game_end_time_utc",
            "sports_continuity_source",
        ],
    ]


def _sports_clean(
    row: pd.Series,
    endpoint_time: pd.Timestamp,
) -> tuple[bool, str]:
    next_event = row["next_salient_event_time_utc"]
    game_end = row["game_end_time_utc"]
    candidates: list[tuple[pd.Timestamp, str]] = []
    if pd.notna(next_event) and next_event <= endpoint_time:
        candidates.append(
            (
                next_event,
                "NEXT_FINALIZED_INFORMATION_EVENT_AT_OR_BEFORE_H",
            )
        )
    if pd.notna(game_end) and game_end <= endpoint_time:
        candidates.append((game_end, "GAME_END_AT_OR_BEFORE_H"))
    if candidates:
        return False, min(candidates, key=lambda value: (value[0], value[1]))[1]
    return True, "SPORTS_WINDOW_CLEAN"


def _diagnostic_direction(delta: float) -> str:
    if delta >= DIRECTION_THRESHOLD_PROBABILITY:
        return "UP"
    if delta <= -DIRECTION_THRESHOLD_PROBABILITY:
        return "DOWN"
    return "NO_MOVE"


def _diagnostic_select_mark(
    trades: _landmarks._TradeIndex,
    *,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    target_time: pd.Timestamp,
) -> _DiagnosticMark:
    """Select the latest source-time bucket and aggregate same-time fills.

    The NFL event interval remains unordered.  After that interval, all actual
    trades at the latest observed native source timestamp are one mark and are
    aggregated by positive-size VWAP.  This never invents a within-timestamp
    ordering or forwards a stale trade.
    """

    interval_start_ns = interval_start.value
    interval_end_ns = interval_end.value
    target_ns = target_time.value
    first_candidate = int(
        np.searchsorted(trades.times_ns, interval_end_ns, side="left")
    )
    after_target = int(np.searchsorted(trades.times_ns, target_ns, side="right"))
    if first_candidate == after_target:
        first_overlap = int(
            np.searchsorted(trades.times_ns, interval_start_ns, side="left")
        )
        after_overlap = int(
            np.searchsorted(trades.times_ns, interval_end_ns, side="left")
        )
        status = (
            "ORDER_AMBIGUOUS"
            if first_overlap < after_overlap
            else "NO_ACTUAL_TRADE"
        )
        return _DiagnosticMark(
            status=status,
            trade_ids=(),
            trade_id_set_sha256=None,
            source_time=None,
            price=math.nan,
            staleness_seconds=math.nan,
            observation_count=0,
            observed_size=0.0,
            semantics="NO_POST_EVENT_SOURCE_TIMESTAMP_BUCKET",
        )
    latest_ns = int(trades.times_ns[after_target - 1])
    first_latest = int(np.searchsorted(trades.times_ns, latest_ns, side="left"))
    after_latest = int(np.searchsorted(trades.times_ns, latest_ns, side="right"))
    latest_time = pd.Timestamp(latest_ns, tz="UTC")
    staleness = float((target_ns - latest_ns) / 1_000_000_000)
    if staleness > _landmarks.MAX_STALENESS_SECONDS:
        return _DiagnosticMark(
            status="STALE",
            trade_ids=(),
            trade_id_set_sha256=None,
            source_time=latest_time,
            price=math.nan,
            staleness_seconds=staleness,
            observation_count=after_latest - first_latest,
            observed_size=0.0,
            semantics="LATEST_SOURCE_TIMESTAMP_STALE",
        )
    sizes = trades.sizes[first_latest:after_latest]
    prices = trades.prices[first_latest:after_latest]
    if (
        len(sizes) == 0
        or not np.isfinite(sizes).all()
        or not np.isfinite(prices).all()
        or np.any(sizes <= 0)
    ):
        return _DiagnosticMark(
            status="INVALID_NONPOSITIVE_SIZE",
            trade_ids=(),
            trade_id_set_sha256=None,
            source_time=latest_time,
            price=math.nan,
            staleness_seconds=staleness,
            observation_count=len(sizes),
            observed_size=0.0,
            semantics="LATEST_SOURCE_TIMESTAMP_INVALID_SIZE",
        )
    ids = tuple(sorted(map(str, trades.trade_ids[first_latest:after_latest])))
    total_size = float(sizes.sum())
    price = float(np.dot(prices, sizes) / total_size)
    id_hash = _sha256_bytes(
        json.dumps(
            list(ids),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return _DiagnosticMark(
        status="OBSERVED",
        trade_ids=ids,
        trade_id_set_sha256=id_hash,
        source_time=latest_time,
        price=price,
        staleness_seconds=staleness,
        observation_count=len(ids),
        observed_size=total_size,
        semantics="LATEST_SOURCE_TIMESTAMP_SIZE_WEIGHTED_VWAP",
    )


def _diagnostic_panel(
    *,
    facts: pd.DataFrame,
    references: pd.DataFrame,
    factor_hits: pd.DataFrame,
    market_rows: pd.DataFrame,
    contracts: pd.DataFrame,
    cohort: pd.DataFrame,
    sports_continuity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_facts, eligible_facts, _ = _landmarks._validate_facts(facts)
    refs = _landmarks._validate_references(references)
    hits = _landmarks._validate_factor_hits(factor_hits, all_facts)
    market = _landmarks._validate_market(market_rows)
    cohort_row = cohort.iloc[0]
    reference_index = {
        (str(row["game_id"]), str(row["atomic_information_episode_id"])): pd.Series(
            row
        )
        for row in refs.to_dict("records")
    }
    continuity_index = {
        (str(row["game_id"]), str(row["atomic_information_episode_id"])): pd.Series(
            row
        )
        for row in sports_continuity.to_dict("records")
    }
    hits_by_event = {
        (str(game_id), str(event_id)): group
        for (game_id, event_id), group in hits.groupby(
            ["game_id", "event_id"], sort=False
        )
    }
    membership = hits.merge(
        all_facts.loc[
            :, ["game_id", "event_id", "atomic_information_episode_id"]
        ],
        on=["game_id", "event_id"],
        how="inner",
        validate="many_to_one",
    ).loc[
        :,
        [
            "game_id",
            "atomic_information_episode_id",
            "factor_id",
            "factor_version",
            "registry_sha256",
            "pbp_source_sha256",
            "predicate_evidence",
        ],
    ]
    membership = membership.sort_values(
        [
            "game_id",
            "atomic_information_episode_id",
            "factor_id",
            "factor_version",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    home_contracts = contracts.loc[
        contracts["contract_role"].eq("ACTUAL_HOME_OUTCOME")
    ]
    observed_indices = {
        (str(game_id), str(venue), str(contract_id)): _landmarks._trade_index(group)
        for (game_id, venue, contract_id), group in market.groupby(
            ["game_id", "venue", "contract_id"], sort=False
        )
    }
    empty_index = _landmarks._trade_index(market.iloc[0:0])
    factor_ids = tuple(sorted(hits["factor_id"].astype(str).unique()))
    event_tags = tuple(
        sorted(
            {
                tag
                for value in all_facts["outcome_tags"]
                for tag in _landmarks._parse_event_tags(
                    value, label="episode_facts.outcome_tags"
                )
            }
        )
    )
    rows: list[dict[str, object]] = []
    for raw_fact in eligible_facts.sort_values(
        ["source_interval_start", "atomic_information_episode_id"],
        kind="mergesort",
    ).to_dict("records"):
        fact = pd.Series(raw_fact)
        key = (
            str(fact["game_id"]),
            str(fact["atomic_information_episode_id"]),
        )
        reference = reference_index.get(key)
        continuity = continuity_index[key]
        fact_hits = hits_by_event.get(
            (str(fact["game_id"]), str(fact["event_id"])),
            hits.iloc[0:0],
        )
        hit_ids = set(fact_hits["factor_id"].astype(str))
        tags = set(
            _landmarks._parse_event_tags(
                fact["outcome_tags"], label="episode_facts.outcome_tags"
            )
        )
        multi_hot = {
            **{f"factor__{factor_id}": factor_id in hit_ids for factor_id in factor_ids},
            **{f"event_tag__{tag}": tag in tags for tag in event_tags},
        }
        factor_provenance = [
            {
                "factor_id": str(hit["factor_id"]),
                "factor_version": str(hit["factor_version"]),
                "source_event_id": str(hit["event_id"]),
                "source_play_id": str(hit["play_id"]),
                "source_hash": str(hit["pbp_source_sha256"]),
                "registry_sha256": str(hit["registry_sha256"]),
                "predicate_evidence": json.loads(str(hit["predicate_evidence"])),
            }
            for hit in fact_hits.to_dict("records")
        ]
        for contract in home_contracts.to_dict("records"):
            trade_index = observed_indices.get(
                (
                    str(fact["game_id"]),
                    str(contract["venue"]),
                    str(contract["contract_id"]),
                ),
                empty_index,
            )
            interval_start = fact["source_interval_start"]
            interval_end = fact["source_interval_end"]
            marks_l = {
                seconds: _diagnostic_select_mark(
                    trade_index,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    target_time=interval_end + pd.Timedelta(seconds=seconds),
                )
                for seconds in _landmarks.LANDMARK_SECONDS
            }
            marks_h = {
                seconds: _diagnostic_select_mark(
                    trade_index,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    target_time=interval_end + pd.Timedelta(seconds=seconds),
                )
                for seconds in _landmarks.ENDPOINT_SECONDS
            }
            for landmark_seconds in _landmarks.LANDMARK_SECONDS:
                landmark_time = interval_end + pd.Timedelta(
                    seconds=landmark_seconds
                )
                mark_l = marks_l[landmark_seconds]
                stage_a_status, stage_a = _landmarks._reference_at_l(
                    reference, landmark_time=landmark_time
                )
                count_30, size_30 = _landmarks._activity(
                    trade_index, landmark_time=landmark_time, seconds=30
                )
                count_60, size_60 = _landmarks._activity(
                    trade_index, landmark_time=landmark_time, seconds=60
                )
                reference_gap = (
                    mark_l.price - float(stage_a["p_after_home"])
                    if mark_l.observed
                    and stage_a_status == "AVAILABLE"
                    and stage_a["p_after_home"] is not None
                    else math.nan
                )
                decision_features = {
                    "schema_version": DIAGNOSTIC_SCHEMA,
                    "target_contract": DIAGNOSTIC_TARGET_CONTRACT,
                    "game_id": str(fact["game_id"]),
                    "atomic_information_episode_id": str(
                        fact["atomic_information_episode_id"]
                    ),
                    "venue": str(contract["venue"]),
                    "actual_home_contract_id": str(contract["contract_id"]),
                    "landmark_seconds": landmark_seconds,
                    "event_interval_start": interval_start,
                    "event_interval_end": interval_end,
                    "direction_threshold_probability": (
                        DIRECTION_THRESHOLD_PROBABILITY
                    ),
                    "direction_threshold_semantics": (
                        DIRECTION_THRESHOLD_SEMANTICS
                    ),
                    "mark_l_trade_ids": list(mark_l.trade_ids),
                    "mark_l_trade_id_set_sha256": mark_l.trade_id_set_sha256,
                    "mark_l_source_time_utc": mark_l.source_time,
                    "mark_l_price": mark_l.price if mark_l.observed else None,
                    "mark_l_staleness_seconds": (
                        mark_l.staleness_seconds if mark_l.observed else None
                    ),
                    "mark_l_observation_count": mark_l.observation_count,
                    "mark_l_observed_size": mark_l.observed_size,
                    "mark_l_semantics": mark_l.semantics,
                    "prior_30s_actual_trade_count": count_30,
                    "prior_30s_actual_trade_size": size_30,
                    "prior_60s_actual_trade_count": count_60,
                    "prior_60s_actual_trade_size": size_60,
                    "stage_a_status": stage_a_status,
                    **stage_a,
                    "reference_gap_at_landmark": reference_gap,
                    "multi_hot_features": multi_hot,
                    "factor_feature_provenance": factor_provenance,
                    "fact_features": {
                        column: fact.get(column)
                        for column in _landmarks._FACT_FEATURE_COLUMNS
                    },
                }
                decision_json = _landmarks._canonical_json(decision_features)
                for endpoint_seconds in _landmarks.ENDPOINT_SECONDS:
                    if endpoint_seconds <= landmark_seconds:
                        continue
                    endpoint_time = interval_end + pd.Timedelta(
                        seconds=endpoint_seconds
                    )
                    mark_h = marks_h[endpoint_seconds]
                    sports_clean_h, sports_reason = _sports_clean(
                        continuity, endpoint_time
                    )
                    decision_eligible = mark_l.observed
                    actual_trade_observed_h = mark_h.observed
                    target_eligible = bool(
                        decision_eligible
                        and sports_clean_h
                        and actual_trade_observed_h
                    )
                    delta = (
                        mark_h.price - mark_l.price
                        if target_eligible
                        else math.nan
                    )
                    direction = (
                        _diagnostic_direction(delta)
                        if target_eligible
                        else "UNOBSERVED"
                    )
                    magnitude = (
                        abs(delta)
                        if direction in {"UP", "DOWN"}
                        else math.nan
                    )
                    if not decision_eligible:
                        attrition_reason = f"LANDMARK_{mark_l.status}"
                    elif not sports_clean_h:
                        attrition_reason = sports_reason
                    elif not actual_trade_observed_h:
                        attrition_reason = f"ENDPOINT_{mark_h.status}"
                    else:
                        attrition_reason = "ELIGIBLE"
                    rows.append(
                        {
                            "schema_version": DIAGNOSTIC_SCHEMA,
                            "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
                            "target_contract": DIAGNOSTIC_TARGET_CONTRACT,
                            "game_id": str(fact["game_id"]),
                            "nfl_week": int(cohort_row["nfl_week"]),
                            "cohort_authority_sha256": str(
                                cohort_row["authority_sha256"]
                            ),
                            "atomic_information_episode_id": str(
                                fact["atomic_information_episode_id"]
                            ),
                            "venue": str(contract["venue"]),
                            "actual_home_contract_id": str(contract["contract_id"]),
                            "home_team": str(fact["home_team"]),
                            "away_team": str(fact["away_team"]),
                            "target_orientation": "ACTUAL_HOME_OUTCOME",
                            "source_interval_start": interval_start,
                            "source_interval_end": interval_end,
                            "source_interval_semantics": "[START,END)",
                            "landmark_seconds": landmark_seconds,
                            "endpoint_seconds": endpoint_seconds,
                            "landmark_utc": landmark_time,
                            "endpoint_utc": endpoint_time,
                            "mark_l_trade_ids_json": json.dumps(
                                list(mark_l.trade_ids),
                                separators=(",", ":"),
                            ),
                            "mark_l_trade_id_set_sha256": (
                                mark_l.trade_id_set_sha256
                            ),
                            "mark_l_source_time_utc": mark_l.source_time,
                            "mark_l_price": (
                                mark_l.price if mark_l.observed else math.nan
                            ),
                            "mark_l_staleness_seconds": (
                                mark_l.staleness_seconds
                                if mark_l.observed
                                else math.nan
                            ),
                            "mark_l_observation_count": (
                                mark_l.observation_count
                            ),
                            "mark_l_observed_size": mark_l.observed_size,
                            "mark_l_semantics": mark_l.semantics,
                            "mark_h_trade_ids_json": json.dumps(
                                list(mark_h.trade_ids),
                                separators=(",", ":"),
                            ),
                            "mark_h_trade_id_set_sha256": (
                                mark_h.trade_id_set_sha256
                            ),
                            "mark_h_source_time_utc": mark_h.source_time,
                            "mark_h_price": (
                                mark_h.price if mark_h.observed else math.nan
                            ),
                            "mark_h_staleness_seconds": (
                                mark_h.staleness_seconds
                                if mark_h.observed
                                else math.nan
                            ),
                            "mark_h_observation_count": (
                                mark_h.observation_count
                            ),
                            "mark_h_observed_size": mark_h.observed_size,
                            "mark_h_semantics": mark_h.semantics,
                            "sports_clean_h": sports_clean_h,
                            "sports_clean_reason": sports_reason,
                            "actual_trade_observed_h": actual_trade_observed_h,
                            "decision_eligible": decision_eligible,
                            "target_eligible": target_eligible,
                            "delta_l_h": delta,
                            "direction": direction,
                            "conditional_magnitude": magnitude,
                            "direction_threshold_probability": (
                                DIRECTION_THRESHOLD_PROBABILITY
                            ),
                            "direction_threshold_semantics": (
                                DIRECTION_THRESHOLD_SEMANTICS
                            ),
                            "venue_tick_support": VENUE_TICK_SUPPORT,
                            "market_continuity_support": (
                                MARKET_CONTINUITY_SUPPORT
                            ),
                            "stage_a_status": stage_a_status,
                            "reference_status": (
                                "MISSING"
                                if reference is None
                                else str(reference["reference_status"])
                            ),
                            "p_before_home": stage_a["p_before_home"],
                            "p_after_home": stage_a["p_after_home"],
                            "reference_delta_home": stage_a[
                                "reference_delta_home"
                            ],
                            "reference_gap_at_landmark": reference_gap,
                            "prior_30s_actual_trade_count": count_30,
                            "prior_30s_actual_trade_size": size_30,
                            "prior_60s_actual_trade_count": count_60,
                            "prior_60s_actual_trade_size": size_60,
                            "decision_features_json": decision_json,
                            "decision_feature_sha256": (
                                _landmarks._sha256_text(decision_json)
                            ),
                            "attrition_reason": attrition_reason,
                        }
                    )
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise DevelopmentPanelError("diagnostic panel unexpectedly empty")
    if panel.duplicated(list(_DIAGNOSTIC_PANEL_GRAIN)).any():
        raise DevelopmentPanelError("diagnostic panel grain is not unique")
    return (
        panel.sort_values(list(_DIAGNOSTIC_PANEL_GRAIN), kind="mergesort").reset_index(
            drop=True
        ),
        membership,
    )


def _confirmatory_evidence_audit(
    *,
    game_id: str,
    venues: Sequence[str],
    evidence: Mapping[tuple[str, str], VenueConfirmatoryEvidence],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for venue in sorted(set(venues)):
        packet = evidence.get((game_id, venue))
        if packet is None:
            rows.append(
                {
                    "game_id": game_id,
                    "venue": venue,
                    "diagnostic_status": "PASS",
                    "diagnostic_claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
                    "direction_threshold_probability": (
                        DIRECTION_THRESHOLD_PROBABILITY
                    ),
                    "direction_threshold_semantics": (
                        DIRECTION_THRESHOLD_SEMANTICS
                    ),
                    "venue_tick_support": VENUE_TICK_SUPPORT,
                    "market_continuity_support": MARKET_CONTINUITY_SUPPORT,
                    "confirmatory_status": "FAIL_CLOSED",
                    "confirmatory_reason": (
                        "MISSING_HISTORICAL_TICK_RULE_AND_"
                        "MARKET_CONTINUITY_EVIDENCE"
                    ),
                    "tick_rule_source_sha256": pd.NA,
                    "continuity_source_sha256": pd.NA,
                    "platform_rule_reference_only": (
                        "KALSHI_2025_PLATFORM_MINIMUM_1_CENT_NOT_"
                        "PER_CONTRACT_SNAPSHOT"
                        if venue == "kalshi"
                        else "NONE"
                    ),
                }
            )
            continue
        if packet.venue != venue:
            raise DevelopmentPanelError(
                f"{game_id}.{venue} evidence venue is inconsistent"
            )
        for value, label in (
            (packet.tick_rule_source_sha256, "tick_rule_source_sha256"),
            (packet.continuity_source_sha256, "continuity_source_sha256"),
        ):
            _require_sha256(value, label=f"{game_id}.{venue}.{label}")
        rows.append(
            {
                "game_id": game_id,
                "venue": venue,
                "diagnostic_status": "PASS",
                "diagnostic_claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
                "direction_threshold_probability": DIRECTION_THRESHOLD_PROBABILITY,
                "direction_threshold_semantics": DIRECTION_THRESHOLD_SEMANTICS,
                "venue_tick_support": "SUPPORTED_BY_EXPLICIT_PACKET",
                "market_continuity_support": "SUPPORTED_BY_EXPLICIT_PACKET",
                "confirmatory_status": "PASS",
                "confirmatory_reason": "EXPLICIT_HISTORICAL_EVIDENCE",
                "tick_rule_source_sha256": packet.tick_rule_source_sha256,
                "continuity_source_sha256": packet.continuity_source_sha256,
                "platform_rule_reference_only": "NONE",
            }
        )
    return pd.DataFrame(rows)


def _confirmatory_continuity(
    *,
    sports_continuity: pd.DataFrame,
    packet: VenueConfirmatoryEvidence,
) -> pd.DataFrame:
    market = packet.market_continuity.copy(deep=True)
    missing = _MARKET_CONTINUITY_COLUMNS - set(market.columns)
    if missing:
        raise DevelopmentPanelError(
            f"{packet.venue} continuity evidence missing: {sorted(missing)}"
        )
    if market.duplicated(
        ["game_id", "atomic_information_episode_id"]
    ).any():
        raise DevelopmentPanelError(
            f"{packet.venue} continuity evidence identity is not unique"
        )
    merged = sports_continuity.merge(
        market.loc[:, sorted(_MARKET_CONTINUITY_COLUMNS)],
        on=["game_id", "atomic_information_episode_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(sports_continuity):
        raise DevelopmentPanelError(
            f"{packet.venue} continuity evidence is incomplete"
        )
    return merged.loc[
        :,
        [
            "game_id",
            "atomic_information_episode_id",
            "next_salient_event_time_utc",
            "suspension_time_utc",
            "game_end_time_utc",
            "continuity_gap_time_utc",
            "continuity_verified_until_utc",
        ],
    ]


def _build_confirmatory(
    *,
    facts: pd.DataFrame,
    references: pd.DataFrame,
    factor_hits: pd.DataFrame,
    market_rows: pd.DataFrame,
    contracts: pd.DataFrame,
    cohort: pd.DataFrame,
    cohort_authority_sha256: str,
    sports_continuity: pd.DataFrame,
    packet: VenueConfirmatoryEvidence,
) -> _landmarks.VenueReactionPanelV3:
    venue = packet.venue
    venue_contracts = contracts.loc[contracts["venue"].eq(venue)].copy()
    venue_contracts["tick_rule_id"] = packet.tick_rule_id
    venue_market = market_rows.loc[market_rows["venue"].eq(venue)].copy()
    if venue_market.empty:
        raise DevelopmentPanelError(
            f"{venue} has no observed trades for confirmatory builder"
        )
    per_game_mapping = [
        {
            "cohort": "development",
            "game_id": str(cohort.iloc[0]["game_id"]),
            "nfl_week": int(cohort.iloc[0]["nfl_week"]),
        }
    ]
    mapping_sha = _sha256_bytes(
        json.dumps(
            per_game_mapping,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return _landmarks.build_venue_reaction_panel_v3(
        episode_facts=facts,
        stage_a_references=references,
        factor_hits=factor_hits,
        market_rows=venue_market,
        contract_metadata=venue_contracts,
        tick_rules=packet.tick_rules,
        continuity=_confirmatory_continuity(
            sports_continuity=sports_continuity,
            packet=packet,
        ),
        cohort_metadata=cohort,
        expected_cohort_authority_sha256=cohort_authority_sha256,
        expected_cohort_mapping_sha256=mapping_sha,
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DevelopmentPanelError(f"content-addressed collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise DevelopmentPanelError(f"content-addressed publish race: {path}")


def _publish_table(
    *,
    output_root: Path,
    game_id: str,
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload = _parquet_bytes(frame)
    digest = _sha256_bytes(payload)
    hexadecimal = digest.removeprefix("sha256:")
    relative = (
        Path("single-game")
        / game_id
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.parquet"
    )
    path = output_root / relative
    _atomic_publish(path, payload)
    schema = pa.Table.from_pandas(frame, preserve_index=False).schema
    return {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": digest,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": _sha256_bytes(str(schema).encode("utf-8")),
        "semantic_rows_sha256": _canonical_sha256(frame.to_dict("records")),
    }


def _publish_game(
    *,
    verified: VerifiedDevelopmentSources,
    output_root: Path,
    game_id: str,
    evidence: Mapping[tuple[str, str], VenueConfirmatoryEvidence],
) -> PublishedDevelopmentGame:
    facts_manifest, _ = _verify_game_manifest(
        batch=verified.facts,
        descriptor=verified.facts.games[game_id],
        label="X13 facts",
    )
    stage_a_manifest, _ = _verify_game_manifest(
        batch=verified.stage_a,
        descriptor=verified.stage_a.games[game_id],
        label="Stage A",
    )
    market_manifest, _ = _verify_game_manifest(
        batch=verified.market,
        descriptor=verified.market.games[game_id],
        label="market",
    )
    facts = _read_verified_table(
        batch=verified.facts,
        manifest=facts_manifest,
        table_name="canonical_factor_events",
        market_style=False,
        label=f"X13 facts.{game_id}",
    )
    hits = _read_verified_table(
        batch=verified.facts,
        manifest=facts_manifest,
        table_name="factor_hits",
        market_style=False,
        label=f"X13 facts.{game_id}",
    )
    references = _read_verified_table(
        batch=verified.stage_a,
        manifest=stage_a_manifest,
        table_name="reference_observations",
        market_style=False,
        label=f"Stage A.{game_id}",
    )
    observations = _read_verified_table(
        batch=verified.market,
        manifest=market_manifest,
        table_name="actual_market_observations",
        market_style=True,
        label=f"market.{game_id}",
    )
    inventory = _read_verified_table(
        batch=verified.market,
        manifest=market_manifest,
        table_name="contract_inventory",
        market_style=True,
        label=f"market.{game_id}",
    )
    game_ids = {
        *facts["game_id"].astype(str).unique(),
        *references["game_id"].astype(str).unique(),
        *observations["game_id"].astype(str).unique(),
        *inventory["game_id"].astype(str).unique(),
    }
    if game_ids != {game_id}:
        raise DevelopmentPanelError(f"{game_id} input tables cross game boundaries")
    teams = facts.loc[:, ["home_team", "away_team"]].drop_duplicates()
    if len(teams) != 1:
        raise DevelopmentPanelError(f"{game_id} team identity is inconsistent")
    home_team = str(teams.iloc[0]["home_team"])
    market_rows, contracts, identity_audit = _adapt_market(
        game_id=game_id,
        home_team=home_team,
        observations=observations,
        inventory=inventory,
    )
    _, eligible_facts, fact_attrition = _landmarks._validate_facts(facts)
    sports_continuity = _sports_continuity(eligible_facts)
    cohort = verified.cohort_metadata.loc[
        verified.cohort_metadata["game_id"].eq(game_id)
    ].reset_index(drop=True)
    diagnostic_panel, factor_membership = _diagnostic_panel(
        facts=facts,
        references=references,
        factor_hits=hits,
        market_rows=market_rows,
        contracts=contracts,
        cohort=cohort,
        sports_continuity=sports_continuity,
    )
    venues = tuple(sorted(contracts["venue"].astype(str).unique()))
    source_audit = _confirmatory_evidence_audit(
        game_id=game_id,
        venues=venues,
        evidence=evidence,
    )
    confirmatory_results: list[_landmarks.VenueReactionPanelV3] = []
    for venue in source_audit.loc[
        source_audit["confirmatory_status"].eq("PASS"), "venue"
    ].astype(str):
        confirmatory_results.append(
            _build_confirmatory(
                facts=facts,
                references=references,
                factor_hits=hits,
                market_rows=market_rows,
                contracts=contracts,
                cohort=cohort,
                cohort_authority_sha256=verified.cohort_authority_sha256,
                sports_continuity=sports_continuity,
                packet=evidence[(game_id, venue)],
            )
        )
    tables: list[tuple[str, pd.DataFrame]] = [
        ("diagnostic_panel", diagnostic_panel),
        ("source_evidence_audit", source_audit),
        ("contract_identity_audit", identity_audit),
        ("sports_continuity", sports_continuity),
        ("factor_membership", factor_membership),
        ("fact_attrition", fact_attrition),
    ]
    if confirmatory_results:
        tables.extend(
            [
                (
                    "confirmatory_panel",
                    pd.concat(
                        [result.panel for result in confirmatory_results],
                        ignore_index=True,
                    ),
                ),
                (
                    "confirmatory_attrition",
                    pd.concat(
                        [result.attrition for result in confirmatory_results],
                        ignore_index=True,
                    ),
                ),
                (
                    "confirmatory_complement_diagnostics",
                    pd.concat(
                        [
                            result.complement_diagnostics
                            for result in confirmatory_results
                        ],
                        ignore_index=True,
                    ),
                ),
            ]
        )
    descriptors = [
        _publish_table(
            output_root=output_root,
            game_id=game_id,
            name=name,
            frame=frame,
        )
        for name, frame in tables
    ]
    material: dict[str, object] = {
        "schema": "nfl_x15_development_game_panel_manifest_v1",
        "builder_version": BUILDER_VERSION,
        "game_id": game_id,
        "cohort": "development",
        "diagnostic_claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "diagnostic_target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "confirmatory_claim_boundary": CONFIRMATORY_CLAIM_BOUNDARY,
        "direction_threshold_probability": DIRECTION_THRESHOLD_PROBABILITY,
        "direction_threshold_semantics": DIRECTION_THRESHOLD_SEMANTICS,
        "diagnostic_venue_count": len(venues),
        "confirmatory_venue_count": len(confirmatory_results),
        "publication_gate": "PASS",
        "holdout_reaction_accessed": False,
        "sources": {
            "facts_manifest_sha256": verified.facts.games[game_id][
                "manifest_sha256"
            ],
            "stage_a_manifest_sha256": verified.stage_a.games[game_id][
                "manifest_sha256"
            ],
            "market_manifest_sha256": verified.market.games[game_id][
                "manifest_sha256"
            ],
            "cohort_authority_sha256": verified.cohort_authority_sha256,
            "cohort_mapping_sha256": verified.cohort_mapping_sha256,
        },
        "tables": descriptors,
    }
    material["bundle_sha256"] = _canonical_sha256(material)
    payload = _canonical_bytes(material)
    manifest_sha = _sha256_bytes(payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    relative = (
        Path("single-game")
        / game_id
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    manifest_path = output_root / relative
    _atomic_publish(manifest_path, payload)
    return PublishedDevelopmentGame(
        game_id=game_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        bundle_sha256=str(material["bundle_sha256"]),
        confirmatory_venue_count=len(confirmatory_results),
        diagnostic_venue_count=len(venues),
    )


def publish_exact153_development_panel(
    *,
    project_root: Path,
    output_root: Path | None = None,
    source_spec: DevelopmentSourceSpec | None = None,
    confirmatory_evidence: Mapping[
        tuple[str, str], VenueConfirmatoryEvidence
    ] | None = None,
    game_ids: Sequence[str] | None = None,
) -> DevelopmentPanelPublication:
    """Verify inputs and publish one game at a time without pooled in-memory data."""

    verified = verify_development_sources(
        project_root=project_root,
        source_spec=source_spec,
    )
    output_root = _resolve_under(
        verified.project_root,
        output_root or DEFAULT_OUTPUT_ROOT,
        label="development panel output root",
    )
    evidence = confirmatory_evidence or {}
    expected_games = set(verified.cohort_metadata["game_id"].astype(str))
    selected = (
        tuple(sorted(expected_games))
        if game_ids is None
        else tuple(sorted(set(map(str, game_ids))))
    )
    if not selected or not set(selected).issubset(expected_games):
        raise DevelopmentPanelError(
            "selected games must be a nonempty subset of verified development"
        )
    published: list[PublishedDevelopmentGame] = []
    for game_id in selected:
        published.append(
            _publish_game(
                verified=verified,
                output_root=output_root,
                game_id=game_id,
                evidence=evidence,
            )
        )
    material: dict[str, object] = {
        "schema": "nfl_x15_development_panel_batch_index_v1",
        "builder_version": BUILDER_VERSION,
        "cohort": "development",
        "verified_development_game_count": len(expected_games),
        "published_game_count": len(published),
        "partial_publication": len(published) != len(expected_games),
        "diagnostic_claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "diagnostic_target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "confirmatory_claim_boundary": CONFIRMATORY_CLAIM_BOUNDARY,
        "direction_threshold_probability": DIRECTION_THRESHOLD_PROBABILITY,
        "direction_threshold_semantics": DIRECTION_THRESHOLD_SEMANTICS,
        "holdout_reaction_accessed": False,
        "cohort_authority_sha256": verified.cohort_authority_sha256,
        "cohort_mapping_sha256": verified.cohort_mapping_sha256,
        "source_batch_file_sha256s": {
            "facts": verified.facts.file_sha256,
            "stage_a": verified.stage_a.file_sha256,
            "market": verified.market.file_sha256,
        },
        "games": [
            {
                "game_id": game.game_id,
                "manifest_path": game.manifest_path.relative_to(
                    output_root
                ).as_posix(),
                "manifest_sha256": game.manifest_sha256,
                "bundle_sha256": game.bundle_sha256,
                "confirmatory_venue_count": game.confirmatory_venue_count,
                "diagnostic_venue_count": game.diagnostic_venue_count,
            }
            for game in published
        ],
        "publication_gate": "PASS",
    }
    material["batch_sha256"] = _canonical_sha256(material)
    payload = _canonical_bytes(material)
    manifest_sha = _sha256_bytes(payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    relative = (
        Path("batches")
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.batch-index.json"
    )
    manifest_path = output_root / relative
    _atomic_publish(manifest_path, payload)
    return DevelopmentPanelPublication(
        output_root=output_root,
        batch_manifest_path=manifest_path,
        batch_manifest_sha256=manifest_sha,
        batch_sha256=str(material["batch_sha256"]),
        game_count=len(published),
        cohort_authority_sha256=verified.cohort_authority_sha256,
        cohort_mapping_sha256=verified.cohort_mapping_sha256,
        games=tuple(published),
    )


__all__ = [
    "BUILDER_VERSION",
    "CONFIRMATORY_CLAIM_BOUNDARY",
    "DEFAULT_OUTPUT_ROOT",
    "DIAGNOSTIC_CLAIM_BOUNDARY",
    "DIAGNOSTIC_SCHEMA",
    "DIAGNOSTIC_TARGET_CONTRACT",
    "DIRECTION_THRESHOLD_PROBABILITY",
    "DIRECTION_THRESHOLD_SEMANTICS",
    "DevelopmentPanelError",
    "DevelopmentPanelPublication",
    "DevelopmentSourceSpec",
    "PublishedDevelopmentGame",
    "VenueConfirmatoryEvidence",
    "VerifiedDevelopmentSources",
    "default_development_source_spec",
    "publish_exact153_development_panel",
    "verify_development_sources",
]
