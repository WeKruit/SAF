"""Content-addressed publication for X-13 development matched reactions.

The runner reads only the already-published exact-153 development batch.  It
verifies every referenced single-game manifest, byte-verifies the selected
factor/market objects, adapts each selected game independently, then aggregates
those development rows for the matched-control analysis.  It never opens the
sealed holdout and never fetches an upstream source.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_x13_matched_reaction import (
    CLAIM_BOUNDARY,
    MATCH_FEATURES,
    MatchedReactionTablesV1,
    MatchedReactionSpecV1,
    _EXACT_KEYS,
    _balance_matches,
    _build_cross_game_results,
    _build_manifest,
    _build_single_game_results,
    _build_source_time_paths,
    _build_venue_consistency,
    _hash_sequence,
    _match_rows,
    _matched_observation_effects,
    _semantic_rows_sha256,
    _validate_observations,
    adapt_published_x13_factor_observations,
    build_matched_reaction_tables,
)


SCHEMA = "nfl_x13_matched_reaction_publication_v1"
BUILDER_VERSION = "nfl-x13-matched-reaction-publication-v1"
EXPECTED_EXACT153_GAMES = 153
DEFAULT_MAX_PROJECTED_PEAK_BYTES = 8 * 1024**3
DEFAULT_MAX_PROJECTED_SECONDS = 30 * 60


class MatchedReactionPublicationError(RuntimeError):
    """An immutable input or publication invariant failed closed."""


@dataclass(frozen=True, slots=True)
class MatchedReactionPublication:
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    selected_game_count: int
    selected_game_ids: tuple[str, ...]
    full_run_status: str
    holdout_reaction_accessed: bool = False


@dataclass(frozen=True, slots=True)
class _OutOfCoreMatchedResult:
    matched_controls_path: Path
    match_audit_path: Path
    matched_pair_rows: int
    match_audit_rows: int
    match_balance: pd.DataFrame
    source_time_paths: pd.DataFrame
    single_game_results: pd.DataFrame
    cross_game_results: pd.DataFrame
    venue_consistency: pd.DataFrame


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise MatchedReactionPublicationError(f"{field} is not a SHA-256")
    return value


def _safe_relative(root: Path, relative: object, *, field: str) -> Path:
    if type(relative) is not str or not relative:
        raise MatchedReactionPublicationError(f"{field} must be a path")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise MatchedReactionPublicationError(f"{field} escapes input root")
    return candidate


def _read_verified_json(
    path: Path,
    *,
    expected_sha256: str,
    field: str,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MatchedReactionPublicationError(f"{field} is not a regular file")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise MatchedReactionPublicationError(f"{field} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatchedReactionPublicationError(
            f"{field} is not valid JSON"
        ) from error
    if type(value) is not dict:
        raise MatchedReactionPublicationError(f"{field} must be an object")
    return value


def _stage_descriptor(
    manifest: Mapping[str, object],
    *,
    game_id: str,
    stage: str,
) -> dict[str, object]:
    stages = manifest.get("stages")
    if type(stages) is not dict or type(stages.get(stage)) is not dict:
        raise MatchedReactionPublicationError(
            f"{game_id} manifest lacks {stage}"
        )
    descriptor = dict(stages[stage])
    _require_sha(
        descriptor.get("object_sha256"),
        field=f"{game_id}.{stage}.object_sha256",
    )
    if (
        type(descriptor.get("row_count")) is not int
        or descriptor["row_count"] < 0
        or type(descriptor.get("byte_length")) is not int
        or descriptor["byte_length"] < 0
    ):
        raise MatchedReactionPublicationError(
            f"{game_id}.{stage} has invalid sizes"
        )
    return descriptor


def _verify_parquet(
    input_root: Path,
    descriptor: Mapping[str, object],
    *,
    field: str,
) -> pd.DataFrame:
    path = _safe_relative(
        input_root, descriptor.get("object_path"), field=f"{field}.object_path"
    )
    if path.is_symlink() or not path.is_file():
        raise MatchedReactionPublicationError(f"{field} object is missing")
    if path.stat().st_size != descriptor["byte_length"]:
        raise MatchedReactionPublicationError(f"{field} byte length mismatch")
    if _sha256_file(path) != descriptor["object_sha256"]:
        raise MatchedReactionPublicationError(f"{field} SHA-256 mismatch")
    try:
        frame = pq.read_table(path).to_pandas()
    except Exception as error:
        raise MatchedReactionPublicationError(
            f"{field} parquet is unreadable"
        ) from error
    if len(frame) != descriptor["row_count"]:
        raise MatchedReactionPublicationError(f"{field} row count mismatch")
    return frame


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        sink,
        compression="zstd",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise MatchedReactionPublicationError(
                f"content-addressed collision: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise MatchedReactionPublicationError(
                f"content-addressed publish race: {path}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _publish_object(
    output_root: Path,
    *,
    logical_name: str,
    payload: bytes,
    suffix: str,
    row_count: int,
) -> dict[str, object]:
    digest = _sha256_bytes(payload)
    hexadecimal = digest.removeprefix("sha256:")
    path = (
        output_root
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.{suffix}"
    )
    _publish_immutable(path, payload)
    return {
        "logical_name": logical_name,
        "object_path": str(path.relative_to(output_root)),
        "object_sha256": digest,
        "byte_length": len(payload),
        "row_count": row_count,
        "format": suffix,
    }


def _publish_file_object(
    output_root: Path,
    *,
    logical_name: str,
    source_path: Path,
    suffix: str,
    row_count: int,
) -> dict[str, object]:
    """Publish a completed out-of-core object without reading it into RAM."""

    digest = _sha256_file(source_path)
    hexadecimal = digest.removeprefix("sha256:")
    path = (
        output_root
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.{suffix}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != source_path.stat().st_size
            or _sha256_file(path) != digest
        ):
            raise MatchedReactionPublicationError(
                f"content-addressed collision: {path}"
            )
    else:
        os.link(source_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "logical_name": logical_name,
        "object_path": str(path.relative_to(output_root)),
        "object_sha256": digest,
        "byte_length": path.stat().st_size,
        "row_count": row_count,
        "format": suffix,
    }


def _append_parquet(
    writer: pq.ParquetWriter | None,
    path: Path,
    frame: pd.DataFrame,
) -> pq.ParquetWriter | None:
    if frame.empty:
        return writer
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(
            path,
            table.schema,
            compression="zstd",
            version="2.6",
            use_dictionary=True,
            write_statistics=True,
        )
    elif table.schema != writer.schema:
        table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


def _load_batch(
    *,
    input_root: Path,
    batch_index_path: Path,
) -> tuple[dict[str, object], str, list[dict[str, object]], list[dict[str, object]]]:
    batch_index_path = batch_index_path.resolve()
    if not batch_index_path.is_relative_to(input_root.resolve()):
        raise MatchedReactionPublicationError("batch index escapes input root")
    filename = batch_index_path.name
    suffix = ".batch-index.json"
    if not filename.endswith(suffix):
        raise MatchedReactionPublicationError(
            "batch index path is not content-addressed"
        )
    expected = "sha256:" + filename[: -len(suffix)]
    _require_sha(expected, field="batch index path")
    batch = _read_verified_json(
        batch_index_path,
        expected_sha256=expected,
        field="batch index",
    )
    material = dict(batch)
    batch_sha = _require_sha(
        material.pop("batch_sha256", None), field="batch.batch_sha256"
    )
    if (
        batch.get("schema")
        != "nfl_expansion_development_market_batch_index_v1"
        or batch.get("experiment_id") != "X-13"
        or batch.get("cohort") != "development"
        or batch.get("game_count") != EXPECTED_EXACT153_GAMES
        or batch.get("publication_gate") != "PASS"
        or batch.get("final_holdout_access") != "CLOSED"
        or batch_sha != _sha256_bytes(_canonical_bytes(material))
    ):
        raise MatchedReactionPublicationError(
            "exact-153 development batch contract mismatch"
        )
    game_rows = batch.get("games")
    if type(game_rows) is not list or len(game_rows) != EXPECTED_EXACT153_GAMES:
        raise MatchedReactionPublicationError(
            "exact-153 game references are incomplete"
        )
    manifests: list[dict[str, object]] = []
    input_hashes: list[dict[str, object]] = [
        {
            "kind": "batch_index",
            "game_id": None,
            "sha256": expected,
            "path": str(batch_index_path.relative_to(input_root)),
            "byte_verified": True,
        }
    ]
    seen: set[str] = set()
    for game_ref in game_rows:
        if type(game_ref) is not dict or type(game_ref.get("game_id")) is not str:
            raise MatchedReactionPublicationError("game reference is malformed")
        game_id = game_ref["game_id"]
        if game_id in seen:
            raise MatchedReactionPublicationError("duplicate game reference")
        seen.add(game_id)
        manifest_sha = _require_sha(
            game_ref.get("manifest_sha256"),
            field=f"{game_id}.manifest_sha256",
        )
        manifest_path = _safe_relative(
            input_root,
            game_ref.get("manifest_path"),
            field=f"{game_id}.manifest_path",
        )
        manifest = _read_verified_json(
            manifest_path,
            expected_sha256=manifest_sha,
            field=f"{game_id} manifest",
        )
        if manifest.get("game_id") != game_id:
            raise MatchedReactionPublicationError(
                f"{game_id} manifest identity mismatch"
            )
        factor = _stage_descriptor(
            manifest, game_id=game_id, stage="factor_observations"
        )
        market = _stage_descriptor(
            manifest, game_id=game_id, stage="actual_market_observations"
        )
        manifests.append(
            {
                "game_id": game_id,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_sha,
                "factor_observations": factor,
                "actual_market_observations": market,
            }
        )
        input_hashes.extend(
            [
                {
                    "kind": "single_game_manifest",
                    "game_id": game_id,
                    "sha256": manifest_sha,
                    "path": str(manifest_path.relative_to(input_root)),
                    "byte_verified": True,
                },
                {
                    "kind": "factor_observations",
                    "game_id": game_id,
                    "sha256": factor["object_sha256"],
                    "path": factor["object_path"],
                    "byte_verified": False,
                },
                {
                    "kind": "actual_market_observations",
                    "game_id": game_id,
                    "sha256": market["object_sha256"],
                    "path": market["object_path"],
                    "byte_verified": False,
                },
            ]
        )
    return batch, expected, manifests, input_hashes


def publish_matched_reaction_smoke(
    *,
    input_root: str | Path,
    batch_index_path: str | Path,
    output_root: str | Path,
    game_ids: Sequence[str] | None = None,
    bootstrap_samples: int = 10_000,
    max_projected_peak_bytes: int = DEFAULT_MAX_PROJECTED_PEAK_BYTES,
    max_projected_seconds: float = DEFAULT_MAX_PROJECTED_SECONDS,
) -> MatchedReactionPublication:
    """Publish a verified smoke and a fail-closed exact-153 feasibility gate."""

    started = time.perf_counter()
    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()
    batch, batch_index_sha, manifests, input_hashes = _load_batch(
        input_root=input_root_path,
        batch_index_path=Path(batch_index_path),
    )
    by_game = {str(row["game_id"]): row for row in manifests}
    selected = tuple(
        sorted(game_ids) if game_ids is not None else sorted(by_game)[:2]
    )
    if len(selected) != 2 or len(set(selected)) != 2:
        raise MatchedReactionPublicationError(
            "smoke requires exactly two distinct development games"
        )
    unknown = sorted(set(selected).difference(by_game))
    if unknown:
        raise MatchedReactionPublicationError(
            "smoke games are outside exact-153: " + ", ".join(unknown)
        )

    load_started = time.perf_counter()
    treated_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    selected_coverage: list[dict[str, object]] = []
    selected_hashes = set(selected)
    for input_hash in input_hashes:
        if input_hash["game_id"] in selected_hashes:
            input_hash["byte_verified"] = True
    for game_id in selected:
        row = by_game[game_id]
        factor = _verify_parquet(
            input_root_path,
            row["factor_observations"],
            field=f"{game_id}.factor_observations",
        )
        market = _verify_parquet(
            input_root_path,
            row["actual_market_observations"],
            field=f"{game_id}.actual_market_observations",
        )
        adapted = adapt_published_x13_factor_observations(factor, market)
        if set(adapted.treated_observations["game_id"]) != {game_id}:
            raise MatchedReactionPublicationError(
                f"{game_id} adapted identity mismatch"
            )
        treated_frames.append(adapted.treated_observations)
        control_frames.append(adapted.routine_control_observations)
        selected_coverage.append(
            {
                "game_id": game_id,
                "factor_observation_rows": len(factor),
                "actual_market_observation_rows": len(market),
                "treated_rows": len(adapted.treated_observations),
                "routine_control_rows": len(
                    adapted.routine_control_observations
                ),
                "factor_object_sha256": row["factor_observations"][
                    "object_sha256"
                ],
                "market_object_sha256": row[
                    "actual_market_observations"
                ]["object_sha256"],
            }
        )
    load_seconds = time.perf_counter() - load_started
    treated = pd.concat(treated_frames, ignore_index=True)
    controls = pd.concat(control_frames, ignore_index=True)
    spec = MatchedReactionSpecV1(bootstrap_samples=bootstrap_samples)
    analysis_started = time.perf_counter()
    result = build_matched_reaction_tables(treated, controls, spec=spec)
    analysis_seconds = time.perf_counter() - analysis_started

    table_frames = {
        "matched_controls": result.matched_controls,
        "match_audit": result.match_audit,
        "match_balance": result.match_balance,
        "source_time_paths": result.source_time_paths,
        "single_game_results": result.single_game_results,
        "cross_game_results": result.cross_game_results,
        "venue_consistency": result.venue_consistency,
        "stage_manifest": result.manifest,
    }
    selected_table_memory = sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        for frame in table_frames.values()
    )
    total_factor_rows = sum(
        int(row["factor_observations"]["row_count"]) for row in manifests
    )
    treated_scale = total_factor_rows / max(len(treated), 1)
    projected_table_bytes = math.ceil(selected_table_memory * treated_scale)
    projected_peak_bytes = projected_table_bytes * 3
    projected_seconds = analysis_seconds * treated_scale
    full_run_status = (
        "FEASIBLE_WITH_CURRENT_IN_MEMORY_RUNNER"
        if projected_peak_bytes <= max_projected_peak_bytes
        and projected_seconds <= max_projected_seconds
        else "BLOCKED_IN_MEMORY_ARCHITECTURE"
    )
    feasibility = {
        "schema": "nfl_x13_matched_reaction_feasibility_v1",
        "selected_game_ids": list(selected),
        "selected_treated_rows": len(treated),
        "selected_control_rows": len(controls),
        "selected_matched_pairs": len(result.matched_controls),
        "selected_table_memory_bytes": selected_table_memory,
        "exact153_factor_rows": total_factor_rows,
        "projected_exact153_matched_pairs": math.ceil(
            len(result.matched_controls) * treated_scale
        ),
        "projected_exact153_table_bytes": projected_table_bytes,
        "projected_exact153_peak_bytes": projected_peak_bytes,
        "projected_exact153_analysis_seconds": projected_seconds,
        # macOS reports ru_maxrss in bytes; Linux reports KiB.  Mislabeling
        # this value would make the exact-153 feasibility gate meaningless.
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "memory_gate_bytes": max_projected_peak_bytes,
        "runtime_gate_seconds": max_projected_seconds,
        "full_run_status": full_run_status,
        "full_run_started": False,
        "reason": (
            "two-game linear projection exceeds the bounded in-memory "
            "publication gate; an out-of-core partitioned matcher is required"
            if full_run_status == "BLOCKED_IN_MEMORY_ARCHITECTURE"
            else "two-game projection remains within the bounded gate"
        ),
    }
    runtime = {
        "schema": "nfl_x13_matched_reaction_runtime_v1",
        "load_and_adapter_seconds": load_seconds,
        "analysis_seconds": analysis_seconds,
        "publication_seconds_excluded": True,
        "selected_game_count": len(selected),
        "bootstrap_samples": bootstrap_samples,
    }
    coverage = {
        "schema": "nfl_x13_matched_reaction_coverage_v1",
        "batch_game_count": len(manifests),
        "selected_game_count": len(selected),
        "selected_games": selected_coverage,
        "treated_rows": len(treated),
        "routine_control_rows": len(controls),
        "matched_pair_rows": len(result.matched_controls),
        "match_audit_rows": len(result.match_audit),
        "source_time_path_rows": len(result.source_time_paths),
        "single_game_result_rows": len(result.single_game_results),
        "cross_game_result_rows": len(result.cross_game_results),
        "holdout_reaction_accessed": False,
    }

    object_records: list[dict[str, object]] = []
    for logical_name, frame in table_frames.items():
        object_records.append(
            _publish_object(
                output_root_path,
                logical_name=logical_name,
                payload=_parquet_bytes(frame),
                suffix="parquet",
                row_count=len(frame),
            )
        )
    for logical_name, value in (
        ("runtime", runtime),
        ("coverage", coverage),
        ("feasibility", feasibility),
        ("input_hashes", {"inputs": input_hashes}),
    ):
        object_records.append(
            _publish_object(
                output_root_path,
                logical_name=logical_name,
                payload=_canonical_bytes(value),
                suffix="json",
                row_count=1,
            )
        )

    material = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "cohort": "development",
        "publication_mode": "TWO_GAME_SMOKE_AND_EXACT153_FEASIBILITY",
        "selected_game_ids": list(selected),
        "batch_index_sha256": batch_index_sha,
        "batch_sha256": batch["batch_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "spec": asdict(spec),
        "objects": sorted(object_records, key=lambda row: row["logical_name"]),
        "full_run_status": full_run_status,
        "full_run_started": False,
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
    }
    bundle_sha = _sha256_bytes(_canonical_bytes(material))
    manifest = {**material, "bundle_sha256": bundle_sha}
    manifest_payload = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        output_root_path
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    _publish_immutable(manifest_path, manifest_payload)
    runtime["total_seconds_including_publication"] = (
        time.perf_counter() - started
    )
    return MatchedReactionPublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        bundle_sha256=bundle_sha,
        selected_game_count=2,
        selected_game_ids=selected,
        full_run_status=full_run_status,
    )


def _full_preflight(
    manifests: Sequence[Mapping[str, object]],
    *,
    max_projected_peak_bytes: int,
    max_projected_seconds: float,
) -> dict[str, object]:
    """Derive a descriptor-only gate before opening any game Parquet object."""

    if (
        type(max_projected_peak_bytes) is not int
        or max_projected_peak_bytes < 1
        or not isinstance(max_projected_seconds, (int, float))
        or not math.isfinite(float(max_projected_seconds))
        or max_projected_seconds <= 0
    ):
        raise MatchedReactionPublicationError("full feasibility gate is invalid")
    input_bytes = sum(
        int(row[stage]["byte_length"])
        for row in manifests
        for stage in ("factor_observations", "actual_market_observations")
    )
    input_rows = sum(
        int(row[stage]["row_count"])
        for row in manifests
        for stage in ("factor_observations", "actual_market_observations")
    )
    # The in-memory runner holds the factor/market inputs, canonical adapted
    # inputs, and matched tables concurrently.  This is the same 3x peak
    # convention used by the existing smoke feasibility projection.
    projected_peak_bytes = input_bytes * 3
    # A conservative metadata-only I/O-and-adaptation allowance.  It is a
    # gate, not a claim about realized throughput; runtime is recorded later.
    projected_seconds = max(1.0, input_rows / 100_000.0)
    if (
        projected_peak_bytes > max_projected_peak_bytes
        or projected_seconds > float(max_projected_seconds)
    ):
        raise MatchedReactionPublicationError(
            "exact-153 full feasibility gate refuses this in-memory run"
        )
    return {
        "schema": "nfl_x13_matched_reaction_full_preflight_v1",
        "game_count": len(manifests),
        "input_bytes": input_bytes,
        "input_rows": input_rows,
        "projected_peak_bytes": projected_peak_bytes,
        "projected_seconds": projected_seconds,
        "memory_gate_bytes": max_projected_peak_bytes,
        "runtime_gate_seconds": float(max_projected_seconds),
        "status": "PASSED",
    }


def build_partitioned_matched_reaction_tables(
    treated_observations: pd.DataFrame,
    routine_control_observations: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1 | None = None,
) -> MatchedReactionTablesV1:
    """Run the V1 matcher in exact-control partitions without changing results.

    The partition key is the complete V1 exact-control key.  It deliberately
    excludes game ID, so every partition retains the ordinary same-game
    preference *and* the cross-game fallback candidate pool.  All post-match
    balance, effects, and bootstrap inference stays global and calls the same
    V1 helpers as the single-run builder.
    """

    active_spec = spec or MatchedReactionSpecV1()
    treated = _validate_observations(
        treated_observations, label="treated", spec=active_spec
    )
    controls = _validate_observations(
        routine_control_observations, label="control", spec=active_spec
    )
    if treated["routine_control_eligible"].any():
        raise MatchedReactionPublicationError(
            "treated observations cannot be routine controls"
        )
    if not controls["routine_control_eligible"].all():
        raise MatchedReactionPublicationError(
            "control observations must be explicitly routine eligible"
        )
    input_source_hashes = tuple(
        sorted(
            {
                source_hash
                for frame in (treated, controls)
                for value in frame["source_hashes"]
                for source_hash in _hash_sequence(
                    value, field="input source_hashes"
                )
            }
        )
    )
    exact_keys = (
        "venue",
        "market_family",
        "outcome_orientation",
        "delay_seconds",
        "horizon_seconds",
    )
    matches_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for key, treated_partition in treated.groupby(
        list(exact_keys), sort=True, dropna=False
    ):
        control_mask = pd.Series(True, index=controls.index)
        for field, value in zip(exact_keys, key, strict=True):
            control_mask &= controls[field].eq(value)
        matches, audit = _match_rows_cached(
            treated_partition,
            controls[control_mask].copy(),
            spec=active_spec,
        )
        matches_parts.append(matches)
        audit_parts.append(audit)
    matches = (
        pd.concat(matches_parts, ignore_index=True)
        if any(not frame.empty for frame in matches_parts)
        else pd.DataFrame()
    )
    audit = (
        pd.concat(audit_parts, ignore_index=True)
        if audit_parts
        else pd.DataFrame()
    )
    matches, audit, balance = _balance_matches(
        matches, audit, spec=active_spec
    )
    effects = _matched_observation_effects(matches, spec=active_spec)
    source_time_paths = _build_source_time_paths(effects, spec=active_spec)
    single_game = _build_single_game_results(effects, spec=active_spec)
    cross_game = _build_cross_game_results(single_game, spec=active_spec)
    venue_consistency = _build_venue_consistency(cross_game, spec=active_spec)
    tables = {
        "matched_controls": matches,
        "match_audit": audit,
        "match_balance": balance,
        "source_time_paths": source_time_paths,
        "single_game_results": single_game,
        "cross_game_results": cross_game,
        "venue_consistency": venue_consistency,
    }
    return MatchedReactionTablesV1(
        matched_controls=matches,
        match_audit=audit,
        match_balance=balance,
        source_time_paths=source_time_paths,
        single_game_results=single_game,
        cross_game_results=cross_game,
        venue_consistency=venue_consistency,
        manifest=_build_manifest(
            tables,
            spec=active_spec,
            input_source_hashes=input_source_hashes,
        ),
    )


def _match_rows_cached(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact V1 matching with reusable factor/game control matrices.

    The original implementation rebuilt and standardized the identical
    control pool for every treated row.  This implementation preserves the
    V1 pool, distance, tie-break, and audit semantics while caching each
    factor/scope pool inside an already exact-key partition.
    """

    routine = controls[controls["routine_control_eligible"]].copy()
    ordered_treated = treated.sort_values(
        [
            "factor_id",
            "game_id",
            "episode_id",
            "venue",
            "delay_seconds",
            "horizon_seconds",
            "observation_id",
        ],
        kind="mergesort",
    )
    factor_base: dict[str, pd.DataFrame] = {}
    factor_episode_keys: dict[str, set[tuple[str, str]]] = {}
    factor_exposed_count: dict[str, int] = {}
    same_game_cache: dict[tuple[str, str], pd.DataFrame] = {}
    pool_cache: dict[
        tuple[str, str, str],
        tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    def base_for(factor_id: str) -> pd.DataFrame:
        if factor_id not in factor_base:
            exposed = routine["hit_factor_ids"].map(
                lambda values: factor_id in values
            ).to_numpy(dtype=bool)
            factor_exposed_count[factor_id] = int(exposed.sum())
            factor_base[factor_id] = routine.loc[~exposed].copy()
            factor_episode_keys[factor_id] = set(
                zip(
                    factor_base[factor_id]["game_id"].astype(str),
                    factor_base[factor_id]["episode_id"].astype(str),
                    strict=True,
                )
            )
        return factor_base[factor_id]

    def same_game_for(
        factor_id: str,
        game_id: str,
        base: pd.DataFrame,
    ) -> pd.DataFrame:
        key = (factor_id, game_id)
        if key not in same_game_cache:
            same_game_cache[key] = base[
                base["game_id"].astype(str).eq(game_id)
            ].copy()
        return same_game_cache[key]

    def cached_pool(
        factor_id: str,
        scope: str,
        game_id: str,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        key = (factor_id, scope, game_id if scope == "SAME_GAME" else "")
        if key not in pool_cache:
            matrix = frame.loc[:, MATCH_FEATURES].to_numpy(dtype=float)
            pool_cache[key] = (
                frame,
                matrix,
                np.median(matrix, axis=0),
                np.std(matrix, axis=0, ddof=0),
            )
        return pool_cache[key]

    match_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for row in ordered_treated.itertuples(index=False):
        factor_id = str(row.factor_id)
        base = base_for(factor_id)
        game_id = str(row.game_id)
        episode_id = str(row.episode_id)
        has_same_episode = (
            game_id,
            episode_id,
        ) in factor_episode_keys[factor_id]
        if has_same_episode:
            same_episode = base["game_id"].astype(str).eq(
                game_id
            ) & base["episode_id"].astype(str).eq(episode_id)
            exact_pool = base.loc[~same_episode]
            same_game = exact_pool[
                exact_pool["game_id"].astype(str).eq(game_id)
            ]
        else:
            exact_pool = base
            same_game = same_game_for(factor_id, game_id, base)
        if len(same_game) >= spec.minimum_controls:
            pool = same_game
            scope = "SAME_GAME"
        elif len(exact_pool) >= spec.minimum_controls:
            pool = exact_pool
            scope = "CROSS_GAME_FALLBACK"
        else:
            audit_rows.append(
                {
                    "treated_observation_id": row.observation_id,
                    "factor_id": factor_id,
                    "factor_version": row.factor_version,
                    "factor_family": row.factor_family,
                    "game_id": row.game_id,
                    "episode_id": row.episode_id,
                    "venue": row.venue,
                    "market_family": row.market_family,
                    "outcome_orientation": row.outcome_orientation,
                    "delay_seconds": row.delay_seconds,
                    "horizon_seconds": row.horizon_seconds,
                    "same_game_control_count": len(same_game),
                    "exact_control_count": len(exact_pool),
                    "factor_exposed_control_count": factor_exposed_count[factor_id],
                    "selected_control_count": 0,
                    "match_scope": "NONE",
                    "maximum_absolute_smd": math.nan,
                    "match_status": "UNMATCHED_MIN_CONTROLS",
                    "claim_boundary": spec.claim_boundary,
                }
            )
            continue

        # A factor-hit row from the same episode is normally removed by the
        # factor exposure mask.  If a source violates that expectation, keep
        # exact semantics and avoid reusing a pool that includes it.
        if has_same_episode:
            matrix = pool.loc[:, MATCH_FEATURES].to_numpy(dtype=float)
            center = np.median(matrix, axis=0)
            scale = np.std(matrix, axis=0, ddof=0)
            cached_frame = pool
        else:
            cached_frame, matrix, center, scale = cached_pool(
                factor_id,
                scope,
                game_id,
                pool,
            )
        treated_values = np.asarray(
            [float(getattr(row, field)) for field in MATCH_FEATURES],
            dtype=float,
        )
        active_scale = scale.copy()
        zero = active_scale <= 0
        active_scale[zero] = np.maximum(
            np.abs(treated_values[zero] - center[zero]),
            1.0,
        )
        distance = np.sqrt(
            np.sum(((matrix - treated_values) / active_scale) ** 2, axis=1)
        )
        order = np.lexsort(
            (
                cached_frame["episode_id"].astype(str).to_numpy(),
                cached_frame["game_id"].astype(str).to_numpy(),
                cached_frame["observation_id"].astype(str).to_numpy(),
                distance,
            )
        )
        positions = order[: spec.nearest_controls]
        selected = cached_frame.iloc[positions]
        selected_distances = distance[positions]
        count = len(selected)
        for control, match_distance in zip(
            selected.itertuples(index=False),
            selected_distances,
            strict=True,
        ):
            match_rows.append(
                {
                    "treated_observation_id": row.observation_id,
                    "control_observation_id": control.observation_id,
                    "factor_id": factor_id,
                    "factor_version": row.factor_version,
                    "factor_family": row.factor_family,
                    "game_id": row.game_id,
                    "control_game_id": control.game_id,
                    "episode_id": row.episode_id,
                    "control_episode_id": control.episode_id,
                    "logical_market_id": row.logical_market_id,
                    "venue": row.venue,
                    "market_family": row.market_family,
                    "outcome_orientation": row.outcome_orientation,
                    "delay_seconds": row.delay_seconds,
                    "horizon_seconds": row.horizon_seconds,
                    "treated_effect": float(row.effect),
                    "control_effect": float(control.effect),
                    "pair_effect": float(row.effect - control.effect),
                    "treated_maximum_excursion": float(row.maximum_excursion),
                    "control_maximum_excursion": float(
                        control.maximum_excursion
                    ),
                    "control_hit_factor_ids": tuple(control.hit_factor_ids),
                    "match_distance": float(match_distance),
                    "match_weight": 1.0 / count,
                    "match_scope": scope,
                    "treated_source_hashes": _hash_sequence(
                        row.source_hashes,
                        field="treated source_hashes",
                    ),
                    "control_source_hashes": _hash_sequence(
                        control.source_hashes,
                        field="control source_hashes",
                    ),
                    **{
                        f"treated_{feature}": float(getattr(row, feature))
                        for feature in MATCH_FEATURES
                    },
                    **{
                        f"control_{feature}": float(getattr(control, feature))
                        for feature in MATCH_FEATURES
                    },
                    "balance_pass": True,
                    "claim_boundary": spec.claim_boundary,
                }
            )
        audit_rows.append(
            {
                "treated_observation_id": row.observation_id,
                "factor_id": factor_id,
                "factor_version": row.factor_version,
                "factor_family": row.factor_family,
                "game_id": row.game_id,
                "episode_id": row.episode_id,
                "venue": row.venue,
                "market_family": row.market_family,
                "outcome_orientation": row.outcome_orientation,
                "delay_seconds": row.delay_seconds,
                "horizon_seconds": row.horizon_seconds,
                "same_game_control_count": len(same_game),
                "exact_control_count": len(exact_pool),
                "factor_exposed_control_count": factor_exposed_count[factor_id],
                "selected_control_count": count,
                "match_scope": scope,
                "maximum_absolute_smd": math.nan,
                "match_status": "MATCHED",
                "claim_boundary": spec.claim_boundary,
            }
        )
    return pd.DataFrame(match_rows), pd.DataFrame(audit_rows)


def _build_partitioned_out_of_core(
    treated_observations: pd.DataFrame,
    routine_control_observations: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
    temporary_root: Path,
) -> _OutOfCoreMatchedResult:
    """Analyze exact-key partitions while streaming the two wide tables."""

    treated = _validate_observations(
        treated_observations,
        label="treated",
        spec=spec,
    )
    controls = _validate_observations(
        routine_control_observations,
        label="control",
        spec=spec,
    )
    if treated["routine_control_eligible"].any():
        raise MatchedReactionPublicationError(
            "treated observations cannot be routine controls"
        )
    if not controls["routine_control_eligible"].all():
        raise MatchedReactionPublicationError(
            "control observations must be explicitly routine eligible"
        )

    exact_keys = tuple(_EXACT_KEYS)
    control_partitions = {
        tuple(key): child.copy()
        for key, child in controls.groupby(
            list(exact_keys),
            sort=True,
            dropna=False,
        )
    }
    matches_path = temporary_root / "matched_controls.parquet"
    audit_path = temporary_root / "match_audit.parquet"
    matches_writer: pq.ParquetWriter | None = None
    audit_writer: pq.ParquetWriter | None = None
    balance_parts: list[pd.DataFrame] = []
    effect_parts: list[pd.DataFrame] = []
    matched_pair_rows = 0
    match_audit_rows = 0
    try:
        for key, treated_partition in treated.groupby(
            list(exact_keys),
            sort=True,
            dropna=False,
        ):
            normalized_key = tuple(key)
            controls_partition = control_partitions.get(normalized_key)
            if controls_partition is None:
                controls_partition = controls.iloc[0:0].copy()
            matches, audit = _match_rows_cached(
                treated_partition,
                controls_partition,
                spec=spec,
            )
            matches, audit, balance = _balance_matches(
                matches,
                audit,
                spec=spec,
            )
            effects = _matched_observation_effects(matches, spec=spec)
            matches_writer = _append_parquet(
                matches_writer,
                matches_path,
                matches,
            )
            audit_writer = _append_parquet(
                audit_writer,
                audit_path,
                audit,
            )
            matched_pair_rows += len(matches)
            match_audit_rows += len(audit)
            if not balance.empty:
                balance_parts.append(balance)
            if not effects.empty:
                effect_parts.append(effects)
            del matches, audit, balance, effects
            gc.collect()
    finally:
        if matches_writer is not None:
            matches_writer.close()
        if audit_writer is not None:
            audit_writer.close()
    if matches_writer is None:
        matches_path.write_bytes(_parquet_bytes(pd.DataFrame()))
    if audit_writer is None:
        audit_path.write_bytes(_parquet_bytes(pd.DataFrame()))

    match_balance = (
        pd.concat(balance_parts, ignore_index=True)
        if balance_parts
        else pd.DataFrame()
    )
    effects = (
        pd.concat(effect_parts, ignore_index=True)
        if effect_parts
        else _matched_observation_effects(pd.DataFrame(), spec=spec)
    )
    source_time_paths = _build_source_time_paths(effects, spec=spec)
    single_game = _build_single_game_results(effects, spec=spec)
    cross_game = _build_cross_game_results(single_game, spec=spec)
    venue_consistency = _build_venue_consistency(cross_game, spec=spec)
    return _OutOfCoreMatchedResult(
        matched_controls_path=matches_path,
        match_audit_path=audit_path,
        matched_pair_rows=matched_pair_rows,
        match_audit_rows=match_audit_rows,
        match_balance=match_balance,
        source_time_paths=source_time_paths,
        single_game_results=single_game,
        cross_game_results=cross_game,
        venue_consistency=venue_consistency,
    )


def publish_matched_reaction_full(
    *,
    input_root: str | Path,
    batch_index_path: str | Path,
    output_root: str | Path,
    bootstrap_samples: int = 10_000,
    max_projected_peak_bytes: int = DEFAULT_MAX_PROJECTED_PEAK_BYTES,
    max_projected_seconds: float = DEFAULT_MAX_PROJECTED_SECONDS,
) -> MatchedReactionPublication:
    """Run and publish the complete verified exact-153 development cohort.

    No two-game/sample selection exists on this path.  The resource gate is
    evaluated from signed descriptors before any Parquet object is opened, and
    the batch pointer is published only after all 153 objects, adaptation, and
    cross-game analysis complete successfully.
    """

    started = time.perf_counter()
    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()
    batch, batch_index_sha, manifests, input_hashes = _load_batch(
        input_root=input_root_path,
        batch_index_path=Path(batch_index_path),
    )
    if len(manifests) != EXPECTED_EXACT153_GAMES:
        raise MatchedReactionPublicationError("full runner requires exact-153 manifests")
    preflight = _full_preflight(
        manifests,
        max_projected_peak_bytes=max_projected_peak_bytes,
        max_projected_seconds=max_projected_seconds,
    )
    treated_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    coverage_games: list[dict[str, object]] = []
    verified_games: set[str] = set()
    load_started = time.perf_counter()
    for row in sorted(manifests, key=lambda value: str(value["game_id"])):
        game_id = str(row["game_id"])
        factor = _verify_parquet(
            input_root_path,
            row["factor_observations"],
            field=f"{game_id}.factor_observations",
        )
        market = _verify_parquet(
            input_root_path,
            row["actual_market_observations"],
            field=f"{game_id}.actual_market_observations",
        )
        adapted = adapt_published_x13_factor_observations(factor, market)
        if set(adapted.treated_observations["game_id"]) != {game_id}:
            raise MatchedReactionPublicationError(
                f"{game_id} adapted identity mismatch"
            )
        treated_frames.append(adapted.treated_observations)
        control_frames.append(adapted.routine_control_observations)
        verified_games.add(game_id)
        coverage_games.append(
            {
                "game_id": game_id,
                "factor_observation_rows": len(factor),
                "actual_market_observation_rows": len(market),
                "treated_rows": len(adapted.treated_observations),
                "routine_control_rows": len(adapted.routine_control_observations),
                "factor_object_sha256": row["factor_observations"]["object_sha256"],
                "market_object_sha256": row["actual_market_observations"]["object_sha256"],
            }
        )
    if verified_games != {str(row["game_id"]) for row in manifests}:
        raise MatchedReactionPublicationError("full runner did not verify every game")
    for input_hash in input_hashes:
        if input_hash["game_id"] in verified_games:
            input_hash["byte_verified"] = True
    load_seconds = time.perf_counter() - load_started
    treated = pd.concat(treated_frames, ignore_index=True)
    controls = pd.concat(control_frames, ignore_index=True)
    spec = MatchedReactionSpecV1(bootstrap_samples=bootstrap_samples)
    output_root_path.mkdir(parents=True, exist_ok=True)
    object_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix=".exact153-out-of-core-",
        dir=output_root_path,
    ) as temporary_name:
        analysis_started = time.perf_counter()
        result = _build_partitioned_out_of_core(
            treated,
            controls,
            spec=spec,
            temporary_root=Path(temporary_name),
        )
        analysis_seconds = time.perf_counter() - analysis_started
        object_records.extend(
            [
                _publish_file_object(
                    output_root_path,
                    logical_name="matched_controls",
                    source_path=result.matched_controls_path,
                    suffix="parquet",
                    row_count=result.matched_pair_rows,
                ),
                _publish_file_object(
                    output_root_path,
                    logical_name="match_audit",
                    source_path=result.match_audit_path,
                    suffix="parquet",
                    row_count=result.match_audit_rows,
                ),
            ]
        )
        table_frames = {
            "match_balance": result.match_balance,
            "source_time_paths": result.source_time_paths,
            "single_game_results": result.single_game_results,
            "cross_game_results": result.cross_game_results,
            "venue_consistency": result.venue_consistency,
        }
        for logical_name, frame in table_frames.items():
            object_records.append(
                _publish_object(
                    output_root_path,
                    logical_name=logical_name,
                    payload=_parquet_bytes(frame),
                    suffix="parquet",
                    row_count=len(frame),
                )
            )
        input_source_hashes = tuple(
            sorted(
                {
                    source_hash
                    for frame in (treated, controls)
                    for value in frame["source_hashes"]
                    for source_hash in _hash_sequence(
                        value,
                        field="input source_hashes",
                    )
                }
            )
        )
        stage_rows: list[dict[str, object]] = []
        by_logical_name = {
            str(record["logical_name"]): record for record in object_records
        }
        for stage, descriptor in sorted(by_logical_name.items()):
            if descriptor["format"] != "parquet":
                continue
            if stage in table_frames:
                frame = table_frames[stage]
                schema_material = [
                    (column, str(frame[column].dtype))
                    for column in frame.columns
                ]
                semantic_hash = _semantic_rows_sha256(frame)
                semantic_basis = "CANONICAL_ROWS"
                column_count = len(frame.columns)
            else:
                source_path = (
                    result.matched_controls_path
                    if stage == "matched_controls"
                    else result.match_audit_path
                )
                arrow_schema = pq.read_schema(source_path)
                schema_material = [
                    (field.name, str(field.type)) for field in arrow_schema
                ]
                semantic_hash = str(descriptor["object_sha256"])
                semantic_basis = "PARQUET_OBJECT_BYTES"
                column_count = len(arrow_schema)
            stage_rows.append(
                {
                    "experiment_id": "X-13",
                    "builder_version": BUILDER_VERSION,
                    "stage": stage,
                    "row_count": descriptor["row_count"],
                    "column_count": column_count,
                    "schema_fingerprint": _sha256_bytes(
                        _canonical_bytes(schema_material)
                    ),
                    "semantic_rows_sha256": semantic_hash,
                    "semantic_hash_basis": semantic_basis,
                    "input_source_hash_count": len(input_source_hashes),
                    "input_source_hashes_sha256": _sha256_bytes(
                        _canonical_bytes(input_source_hashes)
                    ),
                    "claim_boundary": spec.claim_boundary,
                    "holdout_reaction_accessed": False,
                }
            )
        stage_manifest = pd.DataFrame(stage_rows)
        object_records.append(
            _publish_object(
                output_root_path,
                logical_name="stage_manifest",
                payload=_parquet_bytes(stage_manifest),
                suffix="parquet",
                row_count=len(stage_manifest),
            )
        )
    runtime = {
        "schema": "nfl_x13_matched_reaction_full_runtime_v1",
        "load_and_adapter_seconds": load_seconds,
        "analysis_seconds": analysis_seconds,
        "publication_seconds_excluded": True,
        "selected_game_count": EXPECTED_EXACT153_GAMES,
        "bootstrap_samples": bootstrap_samples,
        "execution_mode": "EXACT_KEY_PARTITIONED_OUT_OF_CORE",
    }
    coverage = {
        "schema": "nfl_x13_matched_reaction_full_coverage_v1",
        "batch_game_count": len(manifests),
        "selected_game_count": EXPECTED_EXACT153_GAMES,
        "selected_games": coverage_games,
        "treated_rows": len(treated),
        "routine_control_rows": len(controls),
        "matched_pair_rows": result.matched_pair_rows,
        "match_audit_rows": result.match_audit_rows,
        "source_time_path_rows": len(result.source_time_paths),
        "single_game_result_rows": len(result.single_game_results),
        "cross_game_result_rows": len(result.cross_game_results),
        "holdout_reaction_accessed": False,
    }
    for logical_name, value in (
        ("runtime", runtime),
        ("coverage", coverage),
        ("preflight", preflight),
        ("input_hashes", {"inputs": input_hashes}),
    ):
        object_records.append(
            _publish_object(
                output_root_path,
                logical_name=logical_name,
                payload=_canonical_bytes(value),
                suffix="json",
                row_count=1,
            )
        )
    selected = tuple(sorted(verified_games))
    material = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "cohort": "development",
        "publication_mode": "EXACT153_FULL",
        "selected_game_ids": list(selected),
        "batch_index_sha256": batch_index_sha,
        "batch_sha256": batch["batch_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
        "spec": asdict(spec),
        "objects": sorted(object_records, key=lambda value: value["logical_name"]),
        "full_run_status": "COMPLETED",
        "full_run_started": True,
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
    }
    bundle_sha = _sha256_bytes(_canonical_bytes(material))
    manifest = {**material, "bundle_sha256": bundle_sha}
    manifest_payload = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        output_root_path
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    # This is deliberately the first batch-pointer write on the full path.
    _publish_immutable(manifest_path, manifest_payload)
    runtime["total_seconds_including_publication"] = time.perf_counter() - started
    return MatchedReactionPublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        bundle_sha256=bundle_sha,
        selected_game_count=EXPECTED_EXACT153_GAMES,
        selected_game_ids=selected,
        full_run_status="COMPLETED",
    )


__all__ = [
    "BUILDER_VERSION",
    "DEFAULT_MAX_PROJECTED_PEAK_BYTES",
    "DEFAULT_MAX_PROJECTED_SECONDS",
    "EXPECTED_EXACT153_GAMES",
    "MatchedReactionPublication",
    "MatchedReactionPublicationError",
    "build_partitioned_matched_reaction_tables",
    "publish_matched_reaction_full",
    "publish_matched_reaction_smoke",
]
