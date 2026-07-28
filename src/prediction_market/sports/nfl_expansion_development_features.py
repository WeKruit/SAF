"""Reaction-closed sports features for the frozen X-13 development split.

This module intentionally reads no prediction-market data.  It verifies the
frozen 2025 nflverse objects and the already-published sports-clean bundles,
builds one PIT sports feature view with the existing Factor Lab builder, then
publishes immutable per-game Parquet objects and an exact 153-game batch index.
The 81-game final holdout is structurally rejected.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_factor_lab_cleaning import (
    build_sports_row_disposition_ledger,
)
from prediction_market.sports.nfl_factor_lab_features import (
    NFL_FACTOR_LAB_PBP_COLUMNS,
    build_sports_feature_view,
)
from prediction_market.sports.nfl_factor_lab_types import SportsFeatureViewV1
from prediction_market.sports.nfl_historical_sports_clean import (
    DEFAULT_OUTPUT_ROOT as SPORTS_CLEAN_OUTPUT_ROOT,
)
from prediction_market.sports.nfl_historical_sports_clean import (
    discover_main_checkout,
    verify_frozen_2025_sources,
)

SCHEMA_VERSION = "nfl_expansion_development_sports_features_v1"
EXPECTED_DEVELOPMENT_COUNT = 153
EXPECTED_FINAL_HOLDOUT_COUNT = 81
EXPECTED_GOVERNANCE_OBJECT_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
EXPECTED_GOVERNANCE_MANIFEST_SHA256 = (
    "sha256:545f7e082055cd5070d907da9eacbdb735dd14351c19c5a83d912764bb595a37"
)
EXPECTED_FACTOR_REGISTRY_SHA256 = (
    "sha256:295e4e48237f8523fe3153f28b5b28adc9f6e1d3a219568047da5c250ed243c1"
)
EXPECTED_SPORTS_CLEAN_BATCH_SHA256 = (
    "sha256:4ce0759112084e45a7670c4436e240568215bc825c3133c40827061094d2c934"
)
EXPECTED_PBP_OBJECT_SHA256 = (
    "sha256:3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a"
)
EXPECTED_PARTICIPATION_OBJECT_SHA256 = (
    "sha256:3b26b9487267f011fdbafbce133800ce964c963fdb8b27725abf65663cc46e0f"
)
DEFAULT_GOVERNANCE_MANIFEST = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-registry/manifests/"
    "545f7e082055cd5070d907da9eacbdb735dd14351c19c5a83d912764bb595a37"
    ".manifest.json"
)
DEFAULT_FACTOR_REGISTRY = Path(
    "registries/factors/nfl_factor_registry_v2.json"
)
DEFAULT_SPORTS_CLEAN_BATCH_INDEX = Path(
    "artifacts/market-observation/nfl/x13/"
    "sports-clean-expansion/reaction-closed-v1/batches/manifests/sha256/3f/"
    "3fee02261ae6aeeccbfaa03114ebdbdfd44e96eb7dad1501de5f39c2bce0271b"
    ".batch-index.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-development-sports-features"
)
MARKET_JOIN_FIELDS = frozenset(
    {"pre_event_actual_trade", "staleness_seconds"}
)


class ExpansionDevelopmentFeatureError(RuntimeError):
    """A frozen input, PIT boundary, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class PublishedDevelopmentGameFeature:
    game_id: str
    bundle_sha256: str
    manifest_path: Path
    manifest_sha256: str
    object_path: Path
    object_sha256: str
    row_count: int
    low_support_row_count: int
    reused_checkpoint: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PublishedDevelopmentFeatureBatch:
    scope: str
    game_count: int
    batch_sha256: str
    index_path: Path
    index_sha256: str
    games: tuple[PublishedDevelopmentGameFeature, ...]
    aggregate_row_count: int
    aggregate_low_support_row_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _FrozenInputs:
    development_game_ids: tuple[str, ...]
    final_holdout_game_ids: tuple[str, ...]
    governance_object_sha256: str
    governance_manifest_sha256: str
    factor_registry_sha256: str
    sports_clean_batch_sha256: str
    factor_definition_count: int
    sports_predicate_fields: tuple[str, ...]
    market_join_fields: tuple[str, ...]
    sports_clean_refs: Mapping[str, Mapping[str, object]]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExpansionDevelopmentFeatureError(
            f"JSON input is unreadable: {path}"
        ) from error
    if type(value) is not dict:
        raise ExpansionDevelopmentFeatureError(
            f"JSON input must be an object: {path}"
        )
    return value


def _resolve_under(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ExpansionDevelopmentFeatureError(
            f"path escapes frozen root: {relative}"
        ) from error
    return path


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_length: int | None = None,
) -> None:
    if not path.is_file():
        raise ExpansionDevelopmentFeatureError(f"required file is absent: {path}")
    if expected_length is not None and path.stat().st_size != expected_length:
        raise ExpansionDevelopmentFeatureError(
            f"byte length mismatch: {path}"
        )
    if _sha256_file(path) != expected_sha256:
        raise ExpansionDevelopmentFeatureError(f"SHA-256 mismatch: {path}")


def _atomic_json(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    expected = _sha256_bytes(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(raw)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if path.exists():
        if path.read_bytes() != raw:
            temporary.unlink(missing_ok=True)
            raise ExpansionDevelopmentFeatureError(
                f"immutable JSON target differs: {path}"
            )
        temporary.unlink()
    else:
        os.replace(temporary, path)
    _verify_file(path, expected_sha256=expected, expected_length=len(raw))
    return expected


def _decode_nested_records(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        for field, value in tuple(row.items()):
            if not isinstance(value, str) or value[:1] not in {"[", "{"}:
                continue
            try:
                row[field] = json.loads(value)
            except json.JSONDecodeError:
                pass
        result.append(row)
    return result


def _parquet_value(value: object) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    if hasattr(value, "item"):
        return _parquet_value(value.item())
    raise ExpansionDevelopmentFeatureError(
        f"unsupported Parquet value: {type(value).__name__}"
    )


def _parquet_table(
    records: Sequence[Mapping[str, object]],
) -> pa.Table:
    if not records:
        raise ExpansionDevelopmentFeatureError(
            "cannot publish an empty feature object"
        )
    columns = tuple(records[0])
    if any(tuple(row) != columns for row in records):
        raise ExpansionDevelopmentFeatureError(
            "feature rows do not share one ordered schema"
        )
    return pa.Table.from_pylist(
        [
            {field: _parquet_value(value) for field, value in row.items()}
            for row in records
        ]
    )


def _builder_code_sha256() -> str:
    sources: list[dict[str, str]] = []
    for function in (
        build_sports_feature_view,
        build_sports_row_disposition_ledger,
        materialize_expansion_development_features,
    ):
        source = inspect.getsourcefile(function)
        if source is None:
            raise ExpansionDevelopmentFeatureError(
                f"builder source is unavailable: {function.__name__}"
            )
        path = Path(source)
        sources.append(
            {
                "function": function.__name__,
                "module": function.__module__,
                "file_sha256": _sha256_file(path),
            }
        )
    return _canonical_sha256(sources)


def _verify_frozen_inputs(project_root: Path) -> _FrozenInputs:
    governance_manifest_path = _resolve_under(
        project_root, DEFAULT_GOVERNANCE_MANIFEST
    )
    _verify_file(
        governance_manifest_path,
        expected_sha256=EXPECTED_GOVERNANCE_MANIFEST_SHA256,
    )
    governance_manifest = _read_json(governance_manifest_path)
    if (
        governance_manifest.get("schema")
        != "nfl_factor_expansion_registry_manifest_v2"
        or governance_manifest.get("object_sha256")
        != EXPECTED_GOVERNANCE_OBJECT_SHA256
    ):
        raise ExpansionDevelopmentFeatureError("governance manifest drifted")
    governance_path = _resolve_under(
        governance_manifest_path.parent.parent,
        str(governance_manifest["object_path"]),
    )
    _verify_file(
        governance_path,
        expected_sha256=EXPECTED_GOVERNANCE_OBJECT_SHA256,
        expected_length=int(governance_manifest["byte_length"]),
    )
    governance = _read_json(governance_path)
    if (
        governance.get("schema") != "nfl_factor_expansion_registry_v2"
        or governance.get("status") != "FROZEN"
        or governance.get("factor_registry_version") != "v2"
    ):
        raise ExpansionDevelopmentFeatureError("governance object is not frozen v2")
    split_lock = governance.get("split_lock")
    if not isinstance(split_lock, Mapping):
        raise ExpansionDevelopmentFeatureError("governance split lock is absent")
    assignments = split_lock.get("game_assignments")
    if not isinstance(assignments, list):
        raise ExpansionDevelopmentFeatureError(
            "governance game assignments are absent"
        )
    development = tuple(
        sorted(
            str(row["game_id"])
            for row in assignments
            if isinstance(row, Mapping) and row.get("cohort") == "development"
        )
    )
    final_holdout = tuple(
        sorted(
            str(row["game_id"])
            for row in assignments
            if isinstance(row, Mapping) and row.get("cohort") == "final_holdout"
        )
    )
    if (
        len(development) != EXPECTED_DEVELOPMENT_COUNT
        or len(set(development)) != EXPECTED_DEVELOPMENT_COUNT
        or len(final_holdout) != EXPECTED_FINAL_HOLDOUT_COUNT
        or len(set(final_holdout)) != EXPECTED_FINAL_HOLDOUT_COUNT
        or set(development) & set(final_holdout)
    ):
        raise ExpansionDevelopmentFeatureError("frozen split identity drifted")
    if (
        split_lock.get("development", {}).get("reaction_access") is not False
        or split_lock.get("final_holdout", {}).get("reaction_access") is not False
    ):
        raise ExpansionDevelopmentFeatureError("reaction access is not CLOSED")

    registry_path = _resolve_under(project_root, DEFAULT_FACTOR_REGISTRY)
    _verify_file(
        registry_path, expected_sha256=EXPECTED_FACTOR_REGISTRY_SHA256
    )
    registry = _read_json(registry_path)
    definitions = registry.get("factor_definitions")
    if (
        registry.get("schema") != "nfl_factor_registry_v2"
        or registry.get("version") != "v2"
        or not isinstance(definitions, list)
        or len(definitions) != 104
    ):
        raise ExpansionDevelopmentFeatureError("factor registry v2 drifted")
    predicate_fields = {
        str(clause["field"])
        for definition in definitions
        if isinstance(definition, Mapping)
        for clause in definition.get("predicate", ())
        if isinstance(clause, Mapping) and isinstance(clause.get("field"), str)
    }

    sports_index_path = _resolve_under(
        project_root, DEFAULT_SPORTS_CLEAN_BATCH_INDEX
    )
    sports_index = _read_json(sports_index_path)
    if (
        sports_index.get("schema")
        != "nfl_historical_sports_clean_batch_index_v1"
        or sports_index.get("batch_sha256")
        != EXPECTED_SPORTS_CLEAN_BATCH_SHA256
        or sports_index.get("publication_gate") != "PASS"
        or sports_index.get("reaction_access") != "CLOSED"
        or sports_index.get("published_game_count") != 234
    ):
        raise ExpansionDevelopmentFeatureError("sports-clean batch drifted")
    game_refs = sports_index.get("games")
    if not isinstance(game_refs, list):
        raise ExpansionDevelopmentFeatureError("sports-clean game refs are absent")
    refs = {
        str(row["game_id"]): dict(row)
        for row in game_refs
        if isinstance(row, Mapping) and isinstance(row.get("game_id"), str)
    }
    if len(refs) != 234 or not set(development + final_holdout) <= set(refs):
        raise ExpansionDevelopmentFeatureError(
            "sports-clean batch does not cover the frozen split"
        )
    return _FrozenInputs(
        development_game_ids=development,
        final_holdout_game_ids=final_holdout,
        governance_object_sha256=EXPECTED_GOVERNANCE_OBJECT_SHA256,
        governance_manifest_sha256=EXPECTED_GOVERNANCE_MANIFEST_SHA256,
        factor_registry_sha256=EXPECTED_FACTOR_REGISTRY_SHA256,
        sports_clean_batch_sha256=EXPECTED_SPORTS_CLEAN_BATCH_SHA256,
        factor_definition_count=len(definitions),
        sports_predicate_fields=tuple(
            sorted(predicate_fields - MARKET_JOIN_FIELDS)
        ),
        market_join_fields=tuple(sorted(predicate_fields & MARKET_JOIN_FIELDS)),
        sports_clean_refs=refs,
    )


def _load_verified_sports_clean_game(
    project_root: Path,
    frozen: _FrozenInputs,
    game_id: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    root = _resolve_under(project_root, SPORTS_CLEAN_OUTPUT_ROOT)
    ref = frozen.sports_clean_refs[game_id]
    manifest_path = _resolve_under(root, str(ref["manifest_path"]))
    _verify_file(
        manifest_path,
        expected_sha256=str(ref["manifest_sha256"]),
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("game_id") != game_id
        or manifest.get("bundle_sha256") != ref.get("bundle_sha256")
        or manifest.get("reaction_access") != "CLOSED"
        or manifest.get("publication_gate") != "PASS"
    ):
        raise ExpansionDevelopmentFeatureError(
            f"sports-clean single-game manifest drifted: {game_id}"
        )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or (
        inputs.get("pbp_object_sha256") != EXPECTED_PBP_OBJECT_SHA256
        or inputs.get("participation_object_sha256")
        != EXPECTED_PARTICIPATION_OBJECT_SHA256
    ):
        raise ExpansionDevelopmentFeatureError(
            f"sports-clean source binding drifted: {game_id}"
        )
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ExpansionDevelopmentFeatureError(
            f"sports-clean stages are absent: {game_id}"
        )

    def load(stage_name: str) -> list[dict[str, object]]:
        descriptor = stages.get(stage_name)
        if not isinstance(descriptor, Mapping):
            raise ExpansionDevelopmentFeatureError(
                f"sports-clean stage is absent: {game_id}/{stage_name}"
            )
        path = _resolve_under(root, str(descriptor["object_path"]))
        _verify_file(
            path,
            expected_sha256=str(descriptor["object_sha256"]),
            expected_length=int(descriptor["byte_length"]),
        )
        table = pq.read_table(path)
        if table.num_rows != int(descriptor["row_count"]):
            raise ExpansionDevelopmentFeatureError(
                f"sports-clean row count drifted: {game_id}/{stage_name}"
            )
        return _decode_nested_records(table.to_pylist())

    return (
        manifest,
        load("canonical_events"),
        load("finalized_episodes"),
        load("sports_row_disposition"),
    )


def _first_observed_source_event(
    game_id: str,
    canonical_rows: Sequence[Mapping[str, object]],
) -> str:
    observed = sorted(
        str(row["source_time_start_utc"])
        for row in canonical_rows
        if row.get("game_id") == game_id
        and isinstance(row.get("source_time_start_utc"), str)
    )
    if not observed:
        raise ExpansionDevelopmentFeatureError(
            f"game has no observed source event: {game_id}"
        )
    return observed[0]


def _disposition_reconciliation(
    *,
    selected_rows: Sequence[Mapping[str, object]],
    canonical_rows: Sequence[Mapping[str, object]],
    episode_rows: Sequence[Mapping[str, object]],
    published_rows: Sequence[Mapping[str, object]],
    source_hashes: Mapping[str, str],
) -> str:
    rebuilt = build_sports_row_disposition_ledger(
        selected_pbp_rows=selected_rows,
        finalized_episode_rows=episode_rows,
        canonical_event_rows=canonical_rows,
        source_hashes=source_hashes,
    ).to_records()
    fields = (
        "game_id",
        "raw_play_id",
        "canonical_play_id",
        "episode_id",
        "row_type",
        "live_play_outcome_eligible",
        "administrative_event_eligible",
        "factor_eligible",
        "disposition",
        "exclusion_reasons",
    )

    def projection(rows: Sequence[Mapping[str, object]]) -> list[tuple[object, ...]]:
        return [tuple(row.get(field) for field in fields) for row in rows]

    if projection(rebuilt) != projection(published_rows):
        raise ExpansionDevelopmentFeatureError(
            "published sports disposition eligibility differs from "
            "the feature builder"
        )
    return "PUBLISHED_ELIGIBILITY_FIELDS_MATCH"


def _checkpoint_path(output_root: Path, game_id: str) -> Path:
    return output_root / ".work" / "checkpoints" / f"{game_id}.json"


def _verify_published_game(
    output_root: Path,
    *,
    game_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    input_fingerprint: str,
) -> PublishedDevelopmentGameFeature:
    _verify_file(manifest_path, expected_sha256=manifest_sha256)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema")
        != "nfl_expansion_development_single_game_feature_manifest_v1"
        or manifest.get("game_id") != game_id
        or manifest.get("input_fingerprint") != input_fingerprint
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("reaction_access") != "CLOSED"
        or manifest.get("cutoff_semantics")
        != "first_observed_source_event_utc"
    ):
        raise ExpansionDevelopmentFeatureError(
            f"published feature manifest is not reusable: {game_id}"
        )
    descriptor = manifest.get("feature_view")
    if not isinstance(descriptor, Mapping):
        raise ExpansionDevelopmentFeatureError(
            f"feature descriptor is absent: {game_id}"
        )
    object_path = _resolve_under(output_root, str(descriptor["object_path"]))
    _verify_file(
        object_path,
        expected_sha256=str(descriptor["object_sha256"]),
        expected_length=int(descriptor["byte_length"]),
    )
    if pq.ParquetFile(object_path).metadata.num_rows != int(
        descriptor["row_count"]
    ):
        raise ExpansionDevelopmentFeatureError(
            f"feature Parquet row count drifted: {game_id}"
        )
    return PublishedDevelopmentGameFeature(
        game_id=game_id,
        bundle_sha256=str(manifest["bundle_sha256"]),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        object_path=object_path,
        object_sha256=str(descriptor["object_sha256"]),
        row_count=int(descriptor["row_count"]),
        low_support_row_count=int(manifest["low_support_row_count"]),
        reused_checkpoint=True,
        elapsed_seconds=0.0,
    )


def _load_checkpoint(
    output_root: Path,
    *,
    game_id: str,
    input_fingerprint: str,
) -> PublishedDevelopmentGameFeature | None:
    path = _checkpoint_path(output_root, game_id)
    if not path.is_file():
        return None
    checkpoint = _read_json(path)
    if (
        checkpoint.get("schema")
        != "nfl_expansion_development_feature_checkpoint_v1"
        or checkpoint.get("game_id") != game_id
        or checkpoint.get("input_fingerprint") != input_fingerprint
    ):
        return None
    manifest_path = _resolve_under(
        output_root, str(checkpoint["manifest_path"])
    )
    return _verify_published_game(
        output_root,
        game_id=game_id,
        manifest_path=manifest_path,
        manifest_sha256=str(checkpoint["manifest_sha256"]),
        input_fingerprint=input_fingerprint,
    )


def _publish_game(
    *,
    output_root: Path,
    game_id: str,
    view: SportsFeatureViewV1,
    cutoff_utc: str,
    sports_clean_manifest: Mapping[str, object],
    frozen: _FrozenInputs,
    builder_code_sha256: str,
    input_fingerprint: str,
    disposition_reconciliation: str,
) -> PublishedDevelopmentGameFeature:
    started = time.perf_counter()
    records = view.to_records()
    table = _parquet_table(records)
    work = output_root / ".work" / "objects"
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / f"{uuid.uuid4().hex}.parquet"
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        object_sha256 = _sha256_file(temporary)
        digest = object_sha256.removeprefix("sha256:")
        relative_object = (
            Path("single-game")
            / game_id
            / "objects"
            / "sha256"
            / digest[:2]
            / f"{digest}.parquet"
        )
        object_path = _resolve_under(output_root, relative_object)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if object_path.exists():
            _verify_file(
                object_path,
                expected_sha256=object_sha256,
                expected_length=temporary.stat().st_size,
            )
            temporary.unlink()
        else:
            os.replace(temporary, object_path)
    finally:
        temporary.unlink(missing_ok=True)

    output_fields = set(records[0])
    missing_sports_fields = sorted(
        set(frozen.sports_predicate_fields) - output_fields
    )
    if missing_sports_fields:
        raise ExpansionDevelopmentFeatureError(
            f"factor sports fields are absent for {game_id}: "
            f"{missing_sports_fields}"
        )
    low_support_rows = sum(
        any(
            isinstance(child, Mapping) and child.get("low_support") is True
            for child in row.get("rolling_features", ())
        )
        for row in records
    )
    feature_descriptor = {
        "stage": "sports_feature_view",
        "schema": "SportsFeatureViewV1",
        "grain": "one factor-eligible play joined to one finalized episode",
        "object_path": relative_object.as_posix(),
        "object_sha256": object_sha256,
        "byte_length": object_path.stat().st_size,
        "row_count": len(records),
        "rows_sha256": view.rows_sha256,
        "schema_fingerprint": _sha256_bytes(
            table.schema.serialize().to_pybytes()
        ),
    }
    manifest_material = {
        "schema": (
            "nfl_expansion_development_single_game_feature_manifest_v1"
        ),
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "X-13",
        "game_id": game_id,
        "cohort": "development",
        "reaction_access": "CLOSED",
        "market_data_read": False,
        "source_scope": "sports_only",
        "claim_boundary": (
            "SPORTS FEATURE CONSTRUCTION ONLY; no market reaction, "
            "latency, causality, execution, or alpha conclusion"
        ),
        "input_fingerprint": input_fingerprint,
        "builder_code_sha256": builder_code_sha256,
        "cutoff_utc": cutoff_utc,
        "cutoff_semantics": "first_observed_source_event_utc",
        "cutoff_claim": (
            "PIT conservative proxy; not a scheduled kickoff or measured "
            "receive timestamp"
        ),
        "history_scope": (
            "same frozen 2025 nflverse object only; target game excluded; "
            "completed games must end strictly before cutoff"
        ),
        "inputs": {
            "governance_object_sha256": frozen.governance_object_sha256,
            "governance_manifest_sha256": frozen.governance_manifest_sha256,
            "factor_registry_sha256": frozen.factor_registry_sha256,
            "sports_clean_batch_sha256": frozen.sports_clean_batch_sha256,
            "sports_clean_bundle_sha256": sports_clean_manifest[
                "bundle_sha256"
            ],
            "pbp_object_sha256": EXPECTED_PBP_OBJECT_SHA256,
            "participation_object_sha256": (
                EXPECTED_PARTICIPATION_OBJECT_SHA256
            ),
        },
        "factor_coverage": {
            "factor_definition_count": frozen.factor_definition_count,
            "sports_predicate_field_count": len(
                frozen.sports_predicate_fields
            ),
            "sports_predicate_fields": list(frozen.sports_predicate_fields),
            "market_join_fields_not_constructed": list(
                frozen.market_join_fields
            ),
            "sports_field_gate": "PASS",
        },
        "disposition_reconciliation": disposition_reconciliation,
        "low_support_row_count": low_support_rows,
        "feature_view": feature_descriptor,
        "publication_gate": "PASS",
    }
    bundle_sha256 = _canonical_sha256(manifest_material)
    manifest = {**manifest_material, "bundle_sha256": bundle_sha256}
    raw = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(raw)
    manifest_digest = manifest_sha256.removeprefix("sha256:")
    relative_manifest = (
        Path("single-game")
        / game_id
        / "manifests"
        / "sha256"
        / manifest_digest[:2]
        / f"{manifest_digest}.manifest.json"
    )
    manifest_path = _resolve_under(output_root, relative_manifest)
    observed = _atomic_json(manifest_path, manifest)
    if observed != manifest_sha256:
        raise ExpansionDevelopmentFeatureError("manifest hash drifted")
    checkpoint = {
        "schema": "nfl_expansion_development_feature_checkpoint_v1",
        "game_id": game_id,
        "input_fingerprint": input_fingerprint,
        "bundle_sha256": bundle_sha256,
        "manifest_path": relative_manifest.as_posix(),
        "manifest_sha256": manifest_sha256,
    }
    checkpoint_path = _checkpoint_path(output_root, game_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint_path.parent / (
        f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_checkpoint.write_bytes(_canonical_bytes(checkpoint))
    os.replace(temporary_checkpoint, checkpoint_path)
    verified = _verify_published_game(
        output_root,
        game_id=game_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        input_fingerprint=input_fingerprint,
    )
    return PublishedDevelopmentGameFeature(
        **{
            field: getattr(verified, field)
            for field in (
                "game_id",
                "bundle_sha256",
                "manifest_path",
                "manifest_sha256",
                "object_path",
                "object_sha256",
                "row_count",
                "low_support_row_count",
            )
        },
        reused_checkpoint=False,
        elapsed_seconds=time.perf_counter() - started,
    )


def _publish_batch(
    output_root: Path,
    *,
    scope: str,
    games: Sequence[PublishedDevelopmentGameFeature],
    frozen: _FrozenInputs,
    input_fingerprint: str,
) -> tuple[str, Path, str, int, int]:
    ordered = tuple(sorted(games, key=lambda item: item.game_id))
    rows = sum(game.row_count for game in ordered)
    low_support = sum(game.low_support_row_count for game in ordered)
    material = {
        "schema": "nfl_expansion_development_feature_batch_index_v1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "X-13",
        "scope": scope,
        "cohort": "development",
        "reaction_access": "CLOSED",
        "market_data_read": False,
        "input_fingerprint": input_fingerprint,
        "requested_game_count": len(ordered),
        "published_game_count": len(ordered),
        "failed_game_count": 0,
        "aggregate_feature_row_count": rows,
        "aggregate_low_support_row_count": low_support,
        "governance_object_sha256": frozen.governance_object_sha256,
        "factor_registry_sha256": frozen.factor_registry_sha256,
        "sports_clean_batch_sha256": frozen.sports_clean_batch_sha256,
        "games": [
            {
                "game_id": game.game_id,
                "bundle_sha256": game.bundle_sha256,
                "manifest_path": game.manifest_path.relative_to(
                    output_root
                ).as_posix(),
                "manifest_sha256": game.manifest_sha256,
                "object_sha256": game.object_sha256,
                "row_count": game.row_count,
                "low_support_row_count": game.low_support_row_count,
            }
            for game in ordered
        ],
        "publication_gate": "PASS",
        "claim_boundary": (
            "SPORTS FEATURE CONSTRUCTION ONLY; final holdout was not built "
            "and no market data was read"
        ),
    }
    batch_sha256 = _canonical_sha256(material)
    index = {**material, "batch_sha256": batch_sha256}
    raw = _canonical_bytes(index)
    index_sha256 = _sha256_bytes(raw)
    digest = index_sha256.removeprefix("sha256:")
    relative = (
        Path("batches")
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.batch-index.json"
    )
    path = _resolve_under(output_root, relative)
    if _atomic_json(path, index) != index_sha256:
        raise ExpansionDevelopmentFeatureError("batch index hash drifted")
    return batch_sha256, path, index_sha256, rows, low_support


def materialize_expansion_development_features(
    *,
    project_root: Path,
    data_root: Path | None = None,
    output_root: Path | None = None,
    game_ids: Sequence[str] | None = None,
    workers: int = 4,
    progress: Any | None = None,
) -> PublishedDevelopmentFeatureBatch:
    """Build one bounded game or all 153 frozen development games."""

    started = time.perf_counter()
    project_root = project_root.resolve()
    data_root = (
        discover_main_checkout(project_root)
        if data_root is None
        else data_root.resolve()
    )
    output_root = (
        _resolve_under(project_root, DEFAULT_OUTPUT_ROOT)
        if output_root is None
        else output_root.resolve()
    )
    if workers < 1 or workers > 4:
        raise ExpansionDevelopmentFeatureError(
            "workers must be between 1 and 4"
        )
    frozen = _verify_frozen_inputs(project_root)
    requested = (
        frozen.development_game_ids
        if game_ids is None
        else tuple(sorted(str(game_id) for game_id in game_ids))
    )
    if not requested or len(requested) != len(set(requested)):
        raise ExpansionDevelopmentFeatureError(
            "requested development game IDs must be nonempty and unique"
        )
    unknown = sorted(set(requested) - set(frozen.development_game_ids))
    if unknown:
        raise ExpansionDevelopmentFeatureError(
            f"final holdout or unknown game is forbidden: {unknown}"
        )
    pbp_source, participation_source = verify_frozen_2025_sources(data_root)
    if (
        pbp_source.object_sha256 != EXPECTED_PBP_OBJECT_SHA256
        or participation_source.object_sha256
        != EXPECTED_PARTICIPATION_OBJECT_SHA256
    ):
        raise ExpansionDevelopmentFeatureError("frozen 2025 source drifted")
    builder_code_sha256 = _builder_code_sha256()
    input_fingerprint = _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "builder_code_sha256": builder_code_sha256,
            "governance_object_sha256": frozen.governance_object_sha256,
            "factor_registry_sha256": frozen.factor_registry_sha256,
            "sports_clean_batch_sha256": frozen.sports_clean_batch_sha256,
            "pbp_object_sha256": pbp_source.object_sha256,
            "participation_object_sha256": participation_source.object_sha256,
            "history_scope": "same_frozen_2025_object_only",
            "cutoff_semantics": "first_observed_source_event_utc",
            "low_support_observations": 8,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    published: dict[str, PublishedDevelopmentGameFeature] = {}
    pending: list[str] = []
    for game_id in requested:
        reusable = _load_checkpoint(
            output_root,
            game_id=game_id,
            input_fingerprint=input_fingerprint,
        )
        if reusable is None:
            pending.append(game_id)
        else:
            published[game_id] = reusable
            if progress is not None:
                progress(
                    {
                        "stage": "checkpoint",
                        "game_id": game_id,
                        "status": "REUSED",
                    }
                )

    if pending:
        clean: dict[
            str,
            tuple[
                dict[str, Any],
                list[dict[str, object]],
                list[dict[str, object]],
                list[dict[str, object]],
            ],
        ] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _load_verified_sports_clean_game,
                    project_root,
                    frozen,
                    game_id,
                ): game_id
                for game_id in pending
            }
            for future in as_completed(futures):
                game_id = futures[future]
                clean[game_id] = future.result()
                if progress is not None:
                    progress(
                        {
                            "stage": "sports-clean-load",
                            "game_id": game_id,
                            "status": "VERIFIED",
                        }
                    )

        pbp_schema = set(
            pq.ParquetFile(pbp_source.object_path).schema_arrow.names
        )
        selected_columns = sorted(
            set(NFL_FACTOR_LAB_PBP_COLUMNS) & pbp_schema
        )
        selected_rows = pq.read_table(
            pbp_source.object_path,
            columns=selected_columns,
            filters=[("game_id", "in", pending)],
        ).to_pylist()
        selected_by_game: dict[str, list[dict[str, object]]] = {
            game_id: [] for game_id in pending
        }
        for row in selected_rows:
            selected_by_game[str(row["game_id"])].append(dict(row))
        if any(not selected_by_game[game_id] for game_id in pending):
            raise ExpansionDevelopmentFeatureError(
                "frozen PBP lacks one requested development game"
            )

        all_canonical: list[dict[str, object]] = []
        all_episodes: list[dict[str, object]] = []
        cutoffs: dict[str, str] = {}
        disposition_status: dict[str, str] = {}
        feature_source_hashes = {
            "factor_registry_v2": frozen.factor_registry_sha256,
            "governance_v2": frozen.governance_object_sha256,
            "nflverse_2025_participation_object": (
                participation_source.object_sha256
            ),
            "nflverse_2025_pbp_object": pbp_source.object_sha256,
            "sports_clean_batch": frozen.sports_clean_batch_sha256,
        }
        for game_id in pending:
            _manifest, canonical, episodes, dispositions = clean[game_id]
            all_canonical.extend(canonical)
            all_episodes.extend(episodes)
            cutoffs[game_id] = _first_observed_source_event(
                game_id, canonical
            )
            disposition_status[game_id] = _disposition_reconciliation(
                selected_rows=selected_by_game[game_id],
                canonical_rows=canonical,
                episode_rows=episodes,
                published_rows=dispositions,
                source_hashes=feature_source_hashes,
            )

        historical_rows = pq.read_table(
            pbp_source.object_path,
            columns=selected_columns,
        ).to_pylist()
        combined_view = build_sports_feature_view(
            selected_pbp_rows=selected_rows,
            finalized_episode_rows=all_episodes,
            historical_pbp_rows=historical_rows,
            kickoff_times_utc=cutoffs,
            source_hashes=feature_source_hashes,
            canonical_event_rows=all_canonical,
            frozen_game_bindings=[
                {
                    "native_game_id": game_id,
                    "scheduled_at_utc": None,
                }
                for game_id in pending
            ],
            low_support_observations=8,
        )
        rows_by_game: dict[str, list[Any]] = {
            game_id: [] for game_id in pending
        }
        for row in combined_view.rows:
            rows_by_game[row.game_id].append(row)
        if any(not rows_by_game[game_id] for game_id in pending):
            raise ExpansionDevelopmentFeatureError(
                "feature builder emitted an empty development game"
            )

        def publish_one(game_id: str) -> PublishedDevelopmentGameFeature:
            view = SportsFeatureViewV1.build(
                rows=rows_by_game[game_id],
                source_hashes=feature_source_hashes,
            )
            result = _publish_game(
                output_root=output_root,
                game_id=game_id,
                view=view,
                cutoff_utc=cutoffs[game_id],
                sports_clean_manifest=clean[game_id][0],
                frozen=frozen,
                builder_code_sha256=builder_code_sha256,
                input_fingerprint=input_fingerprint,
                disposition_reconciliation=disposition_status[game_id],
            )
            if progress is not None:
                progress(
                    {
                        "stage": "feature-publish",
                        "game_id": game_id,
                        "status": "PASS",
                        "row_count": result.row_count,
                    }
                )
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(publish_one, game_id): game_id
                for game_id in pending
            }
            for future in as_completed(futures):
                result = future.result()
                published[result.game_id] = result

    ordered = tuple(published[game_id] for game_id in requested)
    if len(ordered) != len(requested):
        raise ExpansionDevelopmentFeatureError(
            "not every requested development game was published"
        )
    forbidden_paths = [
        output_root / "single-game" / game_id
        for game_id in frozen.final_holdout_game_ids
        if (output_root / "single-game" / game_id).exists()
    ]
    if forbidden_paths:
        raise ExpansionDevelopmentFeatureError(
            "final holdout feature output exists; refusing publication"
        )
    is_full = requested == frozen.development_game_ids
    scope = "development-153" if is_full else f"bounded-{len(requested)}"
    if is_full and len(ordered) != EXPECTED_DEVELOPMENT_COUNT:
        raise ExpansionDevelopmentFeatureError(
            "full development batch is not exactly 153 games"
        )
    (
        batch_sha256,
        index_path,
        index_sha256,
        row_count,
        low_support_count,
    ) = _publish_batch(
        output_root,
        scope=scope,
        games=ordered,
        frozen=frozen,
        input_fingerprint=input_fingerprint,
    )
    return PublishedDevelopmentFeatureBatch(
        scope=scope,
        game_count=len(ordered),
        batch_sha256=batch_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
        games=ordered,
        aggregate_row_count=row_count,
        aggregate_low_support_row_count=low_support_count,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "ExpansionDevelopmentFeatureError",
    "PublishedDevelopmentFeatureBatch",
    "PublishedDevelopmentGameFeature",
    "materialize_expansion_development_features",
]
