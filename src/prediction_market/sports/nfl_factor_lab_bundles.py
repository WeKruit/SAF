"""Content-addressed single-game and cross-game NFL Factor Lab bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from prediction_market.immutable_object_store import (
    ArtifactRefV1,
    ImmutableObjectStore,
    content_addressed_key,
    mirror_artifact,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"20[0-9]{2}_[0-9]{2}_[A-Z]+_[A-Z]+\Z")
_BUILDER_VERSION = "nfl-factor-lab-bundle-v2"
_SINGLE_STAGE_GRAINS: dict[str, tuple[str, ...]] = {
    "sports_row_disposition": ("game_id", "raw_play_id"),
    "canonical_events": ("game_id", "play_id"),
    "finalized_episodes": ("game_id", "episode_id"),
    "stat_ledger": ("game_id", "through_play_id"),
    "pit_features": ("game_id", "play_id"),
    "contract_inventory": ("game_id", "logical_market_id", "outcome"),
    "market_observations": ("game_id", "observation_id"),
    "market_cleaning_audit": ("game_id", "audit_row_id"),
    "game_time_intervals": ("game_id", "interval_id"),
    "market_time_intervals": ("game_id", "interval_id"),
    "source_timeline": ("game_id", "timeline_id"),
    "timeline_audit": ("game_id", "timeline_audit_id"),
    "association_attrition": ("game_id", "association_id"),
    "reaction_paths": (
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "horizon_seconds",
    ),
    "factor_observations": (
        "factor_id",
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "horizon_seconds",
    ),
    "factor_exclusion_audit": ("game_id", "audit_id"),
    "single_game_factor_results": (
        "factor_id",
        "game_id",
        "market_family",
        "horizon_seconds",
    ),
}
_CROSS_STAGE_GRAINS: dict[str, tuple[str, ...]] = {
    "cross_game_factor_results": (
        "factor_id",
        "market_family",
        "horizon_seconds",
    ),
    "game_contribution": (
        "factor_id",
        "market_family",
        "horizon_seconds",
        "game_id",
    ),
    "leave_one_game_out": (
        "factor_id",
        "market_family",
        "horizon_seconds",
        "game_id",
    ),
    "venue_consistency": (
        "factor_id",
        "market_family",
        "horizon_seconds",
        "venue",
    ),
    "expert_review_queue": (
        "factor_id",
        "market_family",
        "horizon_seconds",
    ),
}


class FactorLabBundleError(ValueError):
    """A V2 Factor Lab bundle failed identity or semantic verification."""


@dataclass(frozen=True, slots=True)
class PublishedSingleGameBundleV2:
    manifest_path: Path
    bundle_sha256: str
    game_id: str
    stage_paths: dict[str, Path]
    source_hashes: tuple[str, ...]
    code_lock_sha256: str


@dataclass(frozen=True, slots=True)
class PublishedCrossGameBundleV2:
    manifest_path: Path
    bundle_sha256: str
    game_ids: tuple[str, ...]
    stage_paths: dict[str, Path]
    holdout_reaction_accessed: bool
    code_lock_sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_length += len(chunk)
    return "sha256:" + digest.hexdigest(), byte_length


def _require_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FactorLabBundleError(f"{label} must be a lowercase SHA-256")
    return value


def _source_hashes(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted(values))
    if not result or len(result) != len(set(result)):
        raise FactorLabBundleError("source_hashes must be nonempty and unique")
    for value in result:
        _require_hash(value, label="source hash")
    return result


def _canonical_frame(
    frame: pd.DataFrame,
    *,
    grain: tuple[str, ...],
    expected_game_id: str | None,
) -> tuple[pd.DataFrame, pa.Table]:
    if not isinstance(frame, pd.DataFrame):
        raise FactorLabBundleError("stage must be a DataFrame")
    missing = set(grain) - set(frame.columns)
    if missing:
        raise FactorLabBundleError(
            "stage grain columns are missing: " + ", ".join(sorted(missing))
        )
    if expected_game_id is not None:
        if "game_id" not in frame:
            raise FactorLabBundleError("single-game stage has no game_id")
        if frame["game_id"].isna().any():
            raise FactorLabBundleError("single-game stage has null game_id")
        observed = set(frame["game_id"].astype(str))
        if observed != ({expected_game_id} if len(frame) else set()):
            raise FactorLabBundleError("single-game stage contains another game")
    if frame.duplicated(list(grain)).any():
        raise FactorLabBundleError("stage contains duplicate grain")
    ordered_columns = [*grain, *sorted(set(frame.columns) - set(grain))]
    ordered = frame.loc[:, ordered_columns].sort_values(
        list(grain),
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    table = pa.Table.from_pandas(ordered, preserve_index=False)
    table = table.replace_schema_metadata(None).combine_chunks()
    return ordered, table


def _table_semantic_sha256(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return _sha256_bytes(sink.getvalue().to_pybytes())


def _schema_sha256(table: pa.Table) -> str:
    return _sha256_bytes(table.schema.serialize().to_pybytes())


def _stage_identity(
    frames: Mapping[str, pd.DataFrame],
    *,
    grains: Mapping[str, tuple[str, ...]],
    expected_game_id: str | None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pa.Table],
    dict[str, dict[str, object]],
]:
    if set(frames) != set(grains):
        missing = set(grains) - set(frames)
        extra = set(frames) - set(grains)
        raise FactorLabBundleError(
            "stage set differs; missing="
            f"{sorted(missing)!r}; extra={sorted(extra)!r}"
        )
    ordered_frames: dict[str, pd.DataFrame] = {}
    tables: dict[str, pa.Table] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for name in sorted(grains):
        ordered, table = _canonical_frame(
            frames[name],
            grain=grains[name],
            expected_game_id=expected_game_id,
        )
        ordered_frames[name] = ordered
        tables[name] = table
        descriptors[name] = {
            "grain": list(grains[name]),
            "row_count": table.num_rows,
            "schema_fingerprint": _schema_sha256(table),
            "semantic_rows_sha256": _table_semantic_sha256(table),
        }
    return ordered_frames, tables, descriptors


def _keys(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    stage: str,
    nullable: bool = False,
) -> set[tuple[object, ...]]:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise FactorLabBundleError(
            f"{stage} foreign-key columns are missing: "
            + ", ".join(sorted(missing))
        )
    values = frame.loc[:, list(columns)]
    if nullable:
        values = values.dropna(how="any")
    elif values.isna().any(axis=None):
        raise FactorLabBundleError(f"{stage} has null foreign-key values")
    return set(values.itertuples(index=False, name=None))


def _require_subset(
    *,
    stage: str,
    child: set[tuple[object, ...]],
    parent: set[tuple[object, ...]],
    relationship: str,
) -> None:
    missing = child - parent
    if missing:
        example = sorted((repr(value) for value in missing))[0]
        raise FactorLabBundleError(
            f"{stage} {relationship} foreign key is invalid: {example}"
        )


def _validate_single_stage_fks(
    frames: Mapping[str, pd.DataFrame],
    *,
    game_id: str,
) -> None:
    canonical_play_ids = _keys(
        frames["canonical_events"],
        ("play_id",),
        stage="canonical_events",
    )
    episode_ids = _keys(
        frames["finalized_episodes"],
        ("episode_id",),
        stage="finalized_episodes",
    )
    contract_keys = _keys(
        frames["contract_inventory"],
        ("logical_market_id", "outcome"),
        stage="contract_inventory",
    )
    observation_ids = _keys(
        frames["market_observations"],
        ("observation_id",),
        stage="market_observations",
    )

    disposition = frames["sports_row_disposition"]
    _require_subset(
        stage="sports_row_disposition",
        child=_keys(
            disposition,
            ("canonical_play_id",),
            stage="sports_row_disposition",
            nullable=True,
        ),
        parent=canonical_play_ids,
        relationship="canonical play",
    )
    _require_subset(
        stage="sports_row_disposition",
        child=_keys(
            disposition,
            ("episode_id",),
            stage="sports_row_disposition",
            nullable=True,
        ),
        parent=episode_ids,
        relationship="episode",
    )
    _require_subset(
        stage="stat_ledger",
        child=_keys(
            frames["stat_ledger"],
            ("through_play_id",),
            stage="stat_ledger",
        ),
        parent=canonical_play_ids,
        relationship="canonical play",
    )
    _require_subset(
        stage="pit_features",
        child=_keys(
            frames["pit_features"],
            ("play_id",),
            stage="pit_features",
        ),
        parent=canonical_play_ids,
        relationship="canonical play",
    )

    market_observations = frames["market_observations"]
    _require_subset(
        stage="market_observations",
        child=_keys(
            market_observations,
            ("logical_market_id", "outcome"),
            stage="market_observations",
        ),
        parent=contract_keys,
        relationship="contract",
    )
    market_audit = frames["market_cleaning_audit"]
    if "included" not in market_audit:
        raise FactorLabBundleError(
            "market_cleaning_audit foreign-key columns are missing: included"
        )
    included_audit = market_audit[market_audit["included"].eq(True)]
    _require_subset(
        stage="market_cleaning_audit",
        child=_keys(
            included_audit,
            ("observation_id",),
            stage="market_cleaning_audit",
        ),
        parent=observation_ids,
        relationship="included observation",
    )

    game_intervals = frames["game_time_intervals"]
    _require_subset(
        stage="game_time_intervals",
        child=_keys(
            game_intervals,
            ("episode_id",),
            stage="game_time_intervals",
        ),
        parent=episode_ids,
        relationship="episode",
    )
    market_intervals = frames["market_time_intervals"]
    _require_subset(
        stage="market_time_intervals",
        child=_keys(
            market_intervals,
            ("observation_id",),
            stage="market_time_intervals",
        ),
        parent=observation_ids,
        relationship="observation",
    )
    if {"logical_market_id", "outcome"}.issubset(market_intervals):
        _require_subset(
            stage="market_time_intervals",
            child=_keys(
                market_intervals,
                ("logical_market_id", "outcome"),
                stage="market_time_intervals",
            ),
            parent=contract_keys,
            relationship="contract",
        )

    game_interval_ids = _keys(
        game_intervals,
        ("interval_id",),
        stage="game_time_intervals",
    )
    market_interval_ids = _keys(
        market_intervals,
        ("interval_id",),
        stage="market_time_intervals",
    )
    timeline = frames["source_timeline"]
    if "layer" not in timeline:
        raise FactorLabBundleError(
            "source_timeline foreign-key columns are missing: layer"
        )
    layers = set(timeline["layer"].dropna().astype(str))
    if layers - {"G", "M"}:
        raise FactorLabBundleError("source_timeline has an invalid layer")
    _require_subset(
        stage="source_timeline",
        child=_keys(
            timeline[timeline["layer"].eq("G")],
            ("interval_id",),
            stage="source_timeline",
        ),
        parent=game_interval_ids,
        relationship="Layer G interval",
    )
    _require_subset(
        stage="source_timeline",
        child=_keys(
            timeline[timeline["layer"].eq("M")],
            ("interval_id",),
            stage="source_timeline",
        ),
        parent=market_interval_ids,
        relationship="Layer M interval",
    )
    timeline_ids = _keys(
        timeline,
        ("timeline_id",),
        stage="source_timeline",
    )
    _require_subset(
        stage="timeline_audit",
        child=_keys(
            frames["timeline_audit"],
            ("timeline_id",),
            stage="timeline_audit",
        ),
        parent=timeline_ids,
        relationship="timeline",
    )

    for stage in (
        "association_attrition",
        "reaction_paths",
        "factor_observations",
    ):
        frame = frames[stage]
        _require_subset(
            stage=stage,
            child=_keys(frame, ("episode_id",), stage=stage),
            parent=episode_ids,
            relationship="episode",
        )
        _require_subset(
            stage=stage,
            child=_keys(
                frame,
                ("logical_market_id", "outcome"),
                stage=stage,
            ),
            parent=contract_keys,
            relationship="contract",
        )

    reaction_key = (
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "horizon_seconds",
    )
    _require_subset(
        stage="factor_observations",
        child=_keys(
            frames["factor_observations"],
            reaction_key,
            stage="factor_observations",
        ),
        parent=_keys(
            frames["reaction_paths"],
            reaction_key,
            stage="reaction_paths",
        ),
        relationship="reaction path",
    )

    factor_observations = frames["factor_observations"]
    single_results = frames["single_game_factor_results"]
    result_key = ("factor_id", "market_family", "horizon_seconds")
    if "market_family" not in factor_observations:
        raise FactorLabBundleError(
            "factor_observations foreign-key columns are missing: market_family"
        )
    _require_subset(
        stage="single_game_factor_results",
        child=_keys(single_results, result_key, stage="single_game_factor_results"),
        parent=_keys(factor_observations, result_key, stage="factor_observations"),
        relationship="factor observation",
    )

    for stage, frame in frames.items():
        if len(frame) and set(frame["game_id"].astype(str)) != {game_id}:
            raise FactorLabBundleError(
                f"{stage} contains rows outside expected game {game_id}"
            )


def _validate_cross_stage_fks(
    frames: Mapping[str, pd.DataFrame],
    *,
    singles: Sequence[PublishedSingleGameBundleV2],
) -> None:
    frozen_games = set(X13_GAME_IDS)
    result_key = ("factor_id", "market_family", "horizon_seconds")
    result_keys = _keys(
        frames["cross_game_factor_results"],
        result_key,
        stage="cross_game_factor_results",
    )

    single_result_keys: set[tuple[object, ...]] = set()
    for single in singles:
        result_frame = pd.read_parquet(
            single.stage_paths["single_game_factor_results"]
        )
        single_result_keys.update(
            _keys(
                result_frame,
                result_key,
                stage="single_game_factor_results",
            )
        )
    _require_subset(
        stage="single_game_factor_results",
        child=single_result_keys,
        parent=result_keys,
        relationship="cross-game factor result",
    )

    for stage in ("game_contribution", "leave_one_game_out"):
        frame = frames[stage]
        observed_games = set(frame["game_id"].dropna().astype(str))
        if observed_games != frozen_games:
            raise FactorLabBundleError(
                f"{stage} does not exactly cover the frozen 20 games"
            )
        _require_subset(
            stage=stage,
            child=_keys(frame, result_key, stage=stage),
            parent=result_keys,
            relationship="cross-game factor result",
        )
    for stage in ("venue_consistency", "expert_review_queue"):
        _require_subset(
            stage=stage,
            child=_keys(frames[stage], result_key, stage=stage),
            parent=result_keys,
            relationship="cross-game factor result",
        )


def _atomic_publish(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(stage)
        return
    os.replace(stage, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_manifest(path: Path, *, schema: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabBundleError("bundle manifest is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise FactorLabBundleError("bundle manifest is not canonical")
    if value.get("schema") != schema:
        raise FactorLabBundleError("bundle manifest schema differs")
    return value


def _write_stage_files(
    stage_root: Path,
    tables: Mapping[str, pa.Table],
    descriptors: dict[str, dict[str, object]],
) -> None:
    for name, table in tables.items():
        path = stage_root / f"{name}_v2.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        digest, byte_length = _sha256_path(path)
        stored_table = (
            pq.read_table(path)
            .replace_schema_metadata(None)
            .combine_chunks()
        )
        descriptors[name].update(
            {
                "path": path.name,
                "sha256": digest,
                "byte_length": byte_length,
                "row_count": stored_table.num_rows,
                "schema_fingerprint": _schema_sha256(stored_table),
                "semantic_rows_sha256": _table_semantic_sha256(
                    stored_table
                ),
            }
        )


def _identity_stage_descriptors(
    descriptors: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            key: descriptor[key]
            for key in (
                "grain",
                "row_count",
                "schema_fingerprint",
                "semantic_rows_sha256",
            )
        }
        for name, descriptor in sorted(descriptors.items())
    }


def build_single_game_bundle(
    *,
    game_id: str,
    stage_frames: Mapping[str, pd.DataFrame],
    source_hashes: Sequence[str],
    factor_registry_sha256: str,
    code_lock_sha256: str,
    output_root: str | Path,
) -> PublishedSingleGameBundleV2:
    """Publish all formal stages for one game in one atomic bundle."""

    if not isinstance(game_id, str) or _GAME_ID_RE.fullmatch(game_id) is None:
        raise FactorLabBundleError("game_id is invalid")
    sources = _source_hashes(source_hashes)
    registry_hash = _require_hash(
        factor_registry_sha256, label="factor registry hash"
    )
    code_hash = _require_hash(code_lock_sha256, label="code lock hash")
    _ordered, tables, descriptors = _stage_identity(
        stage_frames,
        grains=_SINGLE_STAGE_GRAINS,
        expected_game_id=game_id,
    )
    _validate_single_stage_fks(_ordered, game_id=game_id)
    root = Path(output_root)
    game_root = root / game_id
    game_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".single-game-v2-", dir=game_root))
    try:
        physical_descriptors = json.loads(json.dumps(descriptors))
        _write_stage_files(stage, tables, physical_descriptors)
        identity = {
            "schema": "SingleGameAnalysisBundleIdentityV2",
            "experiment_id": "X-13",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "builder_version": _BUILDER_VERSION,
            "game_id": game_id,
            "factor_registry_sha256": registry_hash,
            "code_lock_sha256": code_hash,
            "source_hashes": list(sources),
            "stages": _identity_stage_descriptors(
                physical_descriptors
            ),
        }
        bundle_sha = _sha256_bytes(_canonical_json(identity))
        target = game_root / bundle_sha.replace(":", "-")
        if target.exists():
            return verify_single_game_bundle(target / "manifest.json")
        manifest = {
            "schema": "SingleGameAnalysisBundleV2",
            "experiment_id": "X-13",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "builder_version": _BUILDER_VERSION,
            "bundle_sha256": bundle_sha,
            "game_id": game_id,
            "factor_registry_sha256": registry_hash,
            "code_lock_sha256": code_hash,
            "source_hashes": list(sources),
            "holdout_reaction_accessed": False,
            "stages": physical_descriptors,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _atomic_publish(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_single_game_bundle(target / "manifest.json")


def _verify_stage_files(
    root: Path,
    descriptors: object,
    *,
    grains: Mapping[str, tuple[str, ...]],
    expected_game_id: str | None,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    if not isinstance(descriptors, dict) or set(descriptors) != set(grains):
        raise FactorLabBundleError("bundle stage set differs")
    result: dict[str, Path] = {}
    ordered_frames: dict[str, pd.DataFrame] = {}
    for name in sorted(grains):
        descriptor = descriptors[name]
        if not isinstance(descriptor, dict):
            raise FactorLabBundleError("stage descriptor is invalid")
        path_value = descriptor.get("path")
        if (
            not isinstance(path_value, str)
            or "/" in path_value
            or "\\" in path_value
        ):
            raise FactorLabBundleError("stage path is invalid")
        path = root / path_value
        if path.is_symlink() or not path.is_file():
            raise FactorLabBundleError("stage file is missing")
        file_sha, byte_length = _sha256_path(path)
        if (
            file_sha != descriptor.get("sha256")
            or byte_length != descriptor.get("byte_length")
        ):
            raise FactorLabBundleError("stage file hash or length differs")
        stored_table = (
            pq.read_table(path)
            .replace_schema_metadata(None)
            .combine_chunks()
        )
        if (
            stored_table.num_rows != descriptor.get("row_count")
            or _schema_sha256(stored_table)
            != descriptor.get("schema_fingerprint")
            or _table_semantic_sha256(stored_table)
            != descriptor.get("semantic_rows_sha256")
        ):
            raise FactorLabBundleError("stage semantic identity differs")
        # Pandas is used only for grain and cross-stage referential checks.
        # Reconstructing Arrow from pandas is not an identity operation:
        # nullable integers, for example, can become doubles on readback.
        ordered, _validation_table = _canonical_frame(
            stored_table.to_pandas(),
            grain=grains[name],
            expected_game_id=expected_game_id,
        )
        if descriptor.get("grain") != list(grains[name]):
            raise FactorLabBundleError("stage grain differs")
        result[name] = path
        ordered_frames[name] = ordered
    return result, ordered_frames


def verify_single_game_bundle(
    manifest_path: str | Path,
) -> PublishedSingleGameBundleV2:
    path = Path(manifest_path)
    value = _read_manifest(path, schema="SingleGameAnalysisBundleV2")
    bundle_sha = _require_hash(value.get("bundle_sha256"), label="bundle hash")
    game_id = value.get("game_id")
    if (
        value.get("experiment_id") != "X-13"
        or value.get("claim_boundary") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or value.get("builder_version") != _BUILDER_VERSION
        or value.get("holdout_reaction_accessed") is not False
        or not isinstance(game_id, str)
        or _GAME_ID_RE.fullmatch(game_id) is None
        or path.parent.name != bundle_sha.replace(":", "-")
        or path.parent.parent.name != game_id
    ):
        raise FactorLabBundleError("single-game bundle identity differs")
    sources = _source_hashes(value.get("source_hashes", ()))
    descriptors = value.get("stages")
    stage_paths, ordered_frames = _verify_stage_files(
        path.parent,
        descriptors,
        grains=_SINGLE_STAGE_GRAINS,
        expected_game_id=game_id,
    )
    _validate_single_stage_fks(ordered_frames, game_id=game_id)
    identity_descriptors: dict[str, dict[str, object]] = {}
    assert isinstance(descriptors, dict)
    for name in sorted(descriptors):
        descriptor = descriptors[name]
        assert isinstance(descriptor, dict)
        identity_descriptors[name] = {
            key: descriptor[key]
            for key in (
                "grain",
                "row_count",
                "schema_fingerprint",
                "semantic_rows_sha256",
            )
        }
    identity = {
        "schema": "SingleGameAnalysisBundleIdentityV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "builder_version": _BUILDER_VERSION,
        "game_id": game_id,
        "factor_registry_sha256": value.get("factor_registry_sha256"),
        "code_lock_sha256": value.get("code_lock_sha256"),
        "source_hashes": list(sources),
        "stages": identity_descriptors,
    }
    if _sha256_bytes(_canonical_json(identity)) != bundle_sha:
        raise FactorLabBundleError("single-game bundle hash differs")
    _require_hash(value.get("factor_registry_sha256"), label="registry hash")
    code_lock_sha256 = _require_hash(
        value.get("code_lock_sha256"), label="code lock hash"
    )
    return PublishedSingleGameBundleV2(
        manifest_path=path,
        bundle_sha256=bundle_sha,
        game_id=game_id,
        stage_paths=stage_paths,
        source_hashes=sources,
        code_lock_sha256=code_lock_sha256,
    )


def build_cross_game_bundle(
    *,
    single_game_manifests: Sequence[str | Path],
    cross_game_frames: Mapping[str, pd.DataFrame],
    batch_spec_sha256: str,
    factor_registry_sha256: str,
    code_lock_sha256: str,
    output_root: str | Path,
) -> PublishedCrossGameBundleV2:
    """Publish cross-game tables only after verifying every single-game input."""

    if not single_game_manifests:
        raise FactorLabBundleError("single-game manifests are required")
    singles = tuple(
        verify_single_game_bundle(path) for path in single_game_manifests
    )
    game_ids = tuple(sorted(item.game_id for item in singles))
    if len(game_ids) != len(set(game_ids)):
        raise FactorLabBundleError("single-game bundle list has duplicate games")
    if game_ids != X13_GAME_IDS:
        raise FactorLabBundleError(
            "cross-game bundle must contain the exact frozen 20 games"
        )
    ordered_frames, tables, descriptors = _stage_identity(
        cross_game_frames,
        grains=_CROSS_STAGE_GRAINS,
        expected_game_id=None,
    )
    _validate_cross_stage_fks(ordered_frames, singles=singles)
    batch_hash = _require_hash(batch_spec_sha256, label="batch spec hash")
    registry_hash = _require_hash(
        factor_registry_sha256, label="factor registry hash"
    )
    code_hash = _require_hash(code_lock_sha256, label="code lock hash")
    if any(single.code_lock_sha256 != code_hash for single in singles):
        raise FactorLabBundleError(
            "single-game and cross-game code locks differ"
        )
    root = Path(output_root).resolve()
    single_refs = [
        {
            "game_id": item.game_id,
            "bundle_sha256": item.bundle_sha256,
            "manifest_relative_path": os.path.relpath(
                item.manifest_path.resolve(),
                root,
            ),
            "manifest_sha256": _sha256_path(item.manifest_path)[0],
        }
        for item in singles
    ]
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cross-game-v2-", dir=root))
    try:
        physical_descriptors = json.loads(json.dumps(descriptors))
        _write_stage_files(stage, tables, physical_descriptors)
        identity = {
            "schema": "CrossGameAnalysisBundleIdentityV2",
            "experiment_id": "X-13",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "builder_version": _BUILDER_VERSION,
            "batch_spec_sha256": batch_hash,
            "factor_registry_sha256": registry_hash,
            "code_lock_sha256": code_hash,
            "game_ids": list(game_ids),
            "single_game_bundles": single_refs,
            "stages": _identity_stage_descriptors(
                physical_descriptors
            ),
        }
        bundle_sha = _sha256_bytes(_canonical_json(identity))
        target = root / bundle_sha.replace(":", "-")
        if target.exists():
            return verify_cross_game_bundle(target / "manifest.json")
        manifest = {
            "schema": "CrossGameAnalysisBundleV2",
            "experiment_id": "X-13",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "builder_version": _BUILDER_VERSION,
            "bundle_sha256": bundle_sha,
            "batch_spec_sha256": batch_hash,
            "factor_registry_sha256": registry_hash,
            "code_lock_sha256": code_hash,
            "game_ids": list(game_ids),
            "single_game_bundles": single_refs,
            "holdout_reaction_accessed": False,
            "stages": physical_descriptors,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _atomic_publish(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_cross_game_bundle(target / "manifest.json")


def verify_cross_game_bundle(
    manifest_path: str | Path,
) -> PublishedCrossGameBundleV2:
    path = Path(manifest_path)
    value = _read_manifest(path, schema="CrossGameAnalysisBundleV2")
    bundle_sha = _require_hash(value.get("bundle_sha256"), label="bundle hash")
    code_lock_sha256 = _require_hash(
        value.get("code_lock_sha256"), label="code lock hash"
    )
    game_ids_value = value.get("game_ids")
    if (
        value.get("experiment_id") != "X-13"
        or value.get("claim_boundary") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or value.get("builder_version") != _BUILDER_VERSION
        or value.get("holdout_reaction_accessed") is not False
        or path.parent.name != bundle_sha.replace(":", "-")
        or not isinstance(game_ids_value, list)
        or not all(isinstance(game, str) for game in game_ids_value)
    ):
        raise FactorLabBundleError("cross-game bundle identity differs")
    game_ids = tuple(game_ids_value)
    if game_ids != tuple(sorted(game_ids)) or len(game_ids) != len(set(game_ids)):
        raise FactorLabBundleError("cross-game game IDs are not canonical")
    if game_ids != X13_GAME_IDS:
        raise FactorLabBundleError(
            "cross-game bundle does not contain the exact frozen 20 games"
        )
    refs = value.get("single_game_bundles")
    if not isinstance(refs, list) or len(refs) != len(game_ids):
        raise FactorLabBundleError("single-game bundle references differ")
    verified_refs: list[dict[str, object]] = []
    verified_singles: list[PublishedSingleGameBundleV2] = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise FactorLabBundleError("single-game reference is invalid")
        relative_value = ref.get("manifest_relative_path")
        if (
            not isinstance(relative_value, str)
            or not relative_value
            or Path(relative_value).is_absolute()
        ):
            raise FactorLabBundleError(
                "single-game manifest relative path is invalid"
            )
        single_path = (path.parent.parent / relative_value).resolve()
        manifest_sha = _sha256_path(single_path)[0] if single_path.is_file() else None
        if manifest_sha != ref.get("manifest_sha256"):
            raise FactorLabBundleError("single-game manifest hash differs")
        single = verify_single_game_bundle(single_path)
        if (
            single.game_id != ref.get("game_id")
            or single.bundle_sha256 != ref.get("bundle_sha256")
        ):
            raise FactorLabBundleError("single-game reference identity differs")
        if single.code_lock_sha256 != code_lock_sha256:
            raise FactorLabBundleError(
                "single-game and cross-game code locks differ"
            )
        verified_refs.append(dict(ref))
        verified_singles.append(single)
    if tuple(sorted(str(ref["game_id"]) for ref in verified_refs)) != game_ids:
        raise FactorLabBundleError("single-game references do not cover game IDs")
    descriptors = value.get("stages")
    stage_paths, ordered_frames = _verify_stage_files(
        path.parent,
        descriptors,
        grains=_CROSS_STAGE_GRAINS,
        expected_game_id=None,
    )
    _validate_cross_stage_fks(
        ordered_frames,
        singles=tuple(verified_singles),
    )
    assert isinstance(descriptors, dict)
    identity_descriptors: dict[str, dict[str, object]] = {}
    for name in sorted(descriptors):
        descriptor = descriptors[name]
        assert isinstance(descriptor, dict)
        identity_descriptors[name] = {
            key: descriptor[key]
            for key in (
                "grain",
                "row_count",
                "schema_fingerprint",
                "semantic_rows_sha256",
            )
        }
    identity = {
        "schema": "CrossGameAnalysisBundleIdentityV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "builder_version": _BUILDER_VERSION,
        "batch_spec_sha256": value.get("batch_spec_sha256"),
        "factor_registry_sha256": value.get("factor_registry_sha256"),
        "code_lock_sha256": value.get("code_lock_sha256"),
        "game_ids": list(game_ids),
        "single_game_bundles": verified_refs,
        "stages": identity_descriptors,
    }
    if _sha256_bytes(_canonical_json(identity)) != bundle_sha:
        raise FactorLabBundleError("cross-game bundle hash differs")
    _require_hash(value.get("batch_spec_sha256"), label="batch spec hash")
    _require_hash(value.get("factor_registry_sha256"), label="registry hash")
    return PublishedCrossGameBundleV2(
        manifest_path=path,
        bundle_sha256=bundle_sha,
        game_ids=game_ids,
        stage_paths=stage_paths,
        holdout_reaction_accessed=False,
        code_lock_sha256=code_lock_sha256,
    )


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise FactorLabBundleError(
            "bundle artifact is outside the declared project root"
        ) from error


def bundle_artifact_refs(
    bundle: PublishedSingleGameBundleV2 | PublishedCrossGameBundleV2,
    *,
    project_root: str | Path,
    s3_bucket: str,
    s3_prefix: str,
) -> tuple[ArtifactRefV1, ...]:
    """Map a verified bundle to the frozen X-13 S3 key convention."""

    root = Path(project_root)
    if isinstance(bundle, PublishedSingleGameBundleV2):
        verified: PublishedSingleGameBundleV2 | PublishedCrossGameBundleV2 = (
            verify_single_game_bundle(bundle.manifest_path)
        )
        namespace = f"research/nfl/x13/single-game/{verified.game_id}"
    elif isinstance(bundle, PublishedCrossGameBundleV2):
        verified = verify_cross_game_bundle(bundle.manifest_path)
        namespace = "research/nfl/x13/cross-game"
    else:
        raise FactorLabBundleError("bundle type is unsupported")

    paths = [verified.manifest_path, *verified.stage_paths.values()]
    refs: list[ArtifactRefV1] = []
    for path in sorted(paths):
        digest, byte_length = _sha256_path(path)
        suffix = path.suffix.lower()
        if path == verified.manifest_path:
            key = (
                f"{s3_prefix}/{namespace}/manifests/"
                f"{verified.bundle_sha256.removeprefix('sha256:')}.json"
            )
            media_type = "application/json"
        else:
            key = content_addressed_key(
                prefix=s3_prefix,
                namespace=namespace,
                sha256=digest,
                suffix=suffix,
            )
            media_type = (
                "application/x-parquet"
                if suffix == ".parquet"
                else "application/octet-stream"
            )
        refs.append(
            ArtifactRefV1(
                sha256=digest,
                byte_length=byte_length,
                media_type=media_type,
                local_relative_path=_relative_to_project(path, root),
                s3_bucket=s3_bucket,
                s3_key=key,
            )
        )
    return tuple(sorted(refs, key=lambda ref: ref.s3_key))


def mirror_bundle(
    bundle: PublishedSingleGameBundleV2 | PublishedCrossGameBundleV2,
    *,
    project_root: str | Path,
    s3_bucket: str,
    s3_prefix: str,
    object_store: ImmutableObjectStore,
) -> tuple[ArtifactRefV1, ...]:
    """Mirror every bundle object and verify the remote readback."""

    root = Path(project_root)
    refs = bundle_artifact_refs(
        bundle,
        project_root=root,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )
    for ref in refs:
        mirror_artifact(
            root / ref.local_relative_path,
            ref=ref,
            object_store=object_store,
        )
    return refs


__all__ = [
    "FactorLabBundleError",
    "PublishedCrossGameBundleV2",
    "PublishedSingleGameBundleV2",
    "bundle_artifact_refs",
    "build_cross_game_bundle",
    "build_single_game_bundle",
    "mirror_bundle",
    "verify_cross_game_bundle",
    "verify_single_game_bundle",
]
