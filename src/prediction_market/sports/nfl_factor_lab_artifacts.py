"""Immutable publication helpers for NFL Factor Lab feature and result bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_factor_lab_types import SportsFeatureViewV1


class FactorLabArtifactError(ValueError):
    """An immutable Factor Lab artifact failed verification."""


@dataclass(frozen=True, slots=True)
class PublishedFeatureViewV1:
    manifest_path: Path
    data_path: Path
    rows_sha256: str
    row_count: int
    code_lock_sha256: str
    factor_registry_sha256: str


@dataclass(frozen=True, slots=True)
class PublishedDiscoveryBundleV1:
    manifest_path: Path
    result_path: Path
    review_queue_path: Path
    bundle_sha256: str
    result_row_count: int
    code_lock_sha256: str
    factor_registry_sha256: str


_BUILDER_VERSION = "nfl-factor-lab-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _publish_directory(stage: Path, target: Path) -> None:
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


def _read_manifest(path: Path, schema: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabArtifactError("manifest is unreadable") from error
    if type(value) is not dict or raw != _canonical(value):
        raise FactorLabArtifactError("manifest is not canonical")
    if value.get("schema") != schema:
        raise FactorLabArtifactError("manifest schema differs")
    return value


def _verify_file(
    root: Path, descriptor: object
) -> tuple[Path, int]:
    if type(descriptor) is not dict:
        raise FactorLabArtifactError("file descriptor is invalid")
    path = root / str(descriptor.get("path"))
    expected_sha = descriptor.get("sha256")
    expected_bytes = descriptor.get("byte_length")
    if (
        type(expected_sha) is not str
        or type(expected_bytes) is not int
        or not path.is_file()
        or path.stat().st_size != expected_bytes
    ):
        raise FactorLabArtifactError("file length differs")
    if _file_sha(path) != expected_sha:
        raise FactorLabArtifactError("file hash differs")
    return path, expected_bytes


def publish_sports_feature_view(
    view: SportsFeatureViewV1,
    *,
    code_lock_sha256: str,
    factor_registry_sha256: str,
    output_root: str | Path,
) -> PublishedFeatureViewV1:
    if not isinstance(view, SportsFeatureViewV1):
        raise FactorLabArtifactError("view must be SportsFeatureViewV1")
    root = Path(output_root)
    target = root / view.rows_sha256.replace(":", "-")
    if target.exists():
        return verify_sports_feature_view(target / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".feature-view-", dir=root))
    try:
        data_path = stage / "sports_feature_view_v1.parquet"
        table = pa.Table.from_pylist(view.to_records())
        pq.write_table(
            table,
            data_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        with data_path.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest = {
            "schema": "SportsFeatureViewPublicationV1",
            "experiment_id": "X-13",
            "builder_version": _BUILDER_VERSION,
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "code_lock_sha256": code_lock_sha256,
            "factor_registry_sha256": factor_registry_sha256,
            "rows_sha256": view.rows_sha256,
            "row_count": len(view.rows),
            "source_hashes": [
                {"name": name, "sha256": digest}
                for name, digest in view.source_hashes
            ],
            "data": {
                "path": data_path.name,
                "byte_length": data_path.stat().st_size,
                "sha256": _file_sha(data_path),
            },
            "known_data_gaps": [
                "negative_provider_timeout_sentinel_encoded_as_null",
                "nfl4th_offline_outputs_not_built",
                "exact_injury_and_substitution_timestamps_unavailable",
            ],
        }
        (stage / "manifest.json").write_bytes(_canonical(manifest))
        _publish_directory(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_sports_feature_view(target / "manifest.json")


def verify_sports_feature_view(
    manifest_path: str | Path,
) -> PublishedFeatureViewV1:
    path = Path(manifest_path)
    value = _read_manifest(path, "SportsFeatureViewPublicationV1")
    rows_sha = value.get("rows_sha256")
    row_count = value.get("row_count")
    code_lock_sha = value.get("code_lock_sha256")
    factor_registry_sha = value.get("factor_registry_sha256")
    if (
        value.get("builder_version") != _BUILDER_VERSION
        or type(rows_sha) is not str
        or path.parent.name != rows_sha.replace(":", "-")
        or type(row_count) is not int
        or row_count <= 0
        or type(code_lock_sha) is not str
        or type(factor_registry_sha) is not str
    ):
        raise FactorLabArtifactError("feature-view identity differs")
    data_path, _ = _verify_file(path.parent, value.get("data"))
    if pq.ParquetFile(data_path).metadata.num_rows != row_count:
        raise FactorLabArtifactError("feature-view row count differs")
    return PublishedFeatureViewV1(
        manifest_path=path,
        data_path=data_path,
        rows_sha256=rows_sha,
        row_count=row_count,
        code_lock_sha256=code_lock_sha,
        factor_registry_sha256=factor_registry_sha,
    )


def publish_factor_discovery_bundle(
    results: pd.DataFrame,
    *,
    factor_registry_sha256: str,
    feature_view_sha256: str,
    code_lock_sha256: str,
    source_hashes: Sequence[str],
    output_root: str | Path,
) -> PublishedDiscoveryBundleV1:
    required = {
        "factor_id",
        "factor_version",
        "market_family",
        "horizon_seconds",
        "discovery_status",
        "point_estimate",
    }
    if not isinstance(results, pd.DataFrame) or not required <= set(results):
        raise FactorLabArtifactError("discovery results schema is incomplete")
    ordered = results.sort_values(
        ["factor_id", "market_family", "horizon_seconds"],
        kind="mergesort",
    ).reset_index(drop=True)
    result_records = json.loads(
        ordered.to_json(orient="records", date_format="iso")
    )
    identity = {
        "schema": "FactorDiscoveryIdentityV1",
        "experiment_id": "X-13",
        "builder_version": _BUILDER_VERSION,
        "code_lock_sha256": code_lock_sha256,
        "factor_registry_sha256": factor_registry_sha256,
        "feature_view_sha256": feature_view_sha256,
        "source_hashes": sorted(source_hashes),
        "results": result_records,
    }
    bundle_sha = _sha(_canonical(identity))
    root = Path(output_root)
    target = root / bundle_sha.replace(":", "-")
    if target.exists():
        return verify_factor_discovery_bundle(target / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".factor-discovery-", dir=root))
    try:
        result_path = stage / "factor_results_v1.parquet"
        ordered.to_parquet(result_path, index=False, compression="zstd")
        candidates = ordered[
            ordered["discovery_status"].eq("AWAITING_EXPERT_REVIEW")
        ]
        queue = {
            "schema": "HumanFactorReviewQueueV1",
            "experiment_id": "X-13",
            "holdout_reaction_accessed": False,
            "reviews": [
                {
                    "factor_id": row.factor_id,
                    "factor_version": row.factor_version,
                    "market_family": row.market_family,
                    "horizon_seconds": int(row.horizon_seconds),
                    "decision": "PENDING",
                    "sports_rationale": "",
                    "anomaly_case_ids": [],
                    "priority": "",
                }
                for row in candidates.itertuples(index=False)
            ],
        }
        queue_path = stage / "human_review_queue_v1.json"
        queue_path.write_bytes(_canonical(queue))
        manifest = {
            "schema": "FactorDiscoveryBundleV1",
            "experiment_id": "X-13",
            "builder_version": _BUILDER_VERSION,
            "bundle_sha256": bundle_sha,
            "claim_boundary": (
                "PRELIMINARY_SOURCE_TIME_ONLY; discovery associations only; "
                "no causality, latency, execution, tradeability, or alpha claim"
            ),
            "factor_registry_sha256": factor_registry_sha256,
            "code_lock_sha256": code_lock_sha256,
            "feature_view_sha256": feature_view_sha256,
            "source_hashes": sorted(source_hashes),
            "holdout_reaction_accessed": False,
            "result_row_count": len(ordered),
            "files": {
                "results": {
                    "path": result_path.name,
                    "byte_length": result_path.stat().st_size,
                    "sha256": _file_sha(result_path),
                },
                "review_queue": {
                    "path": queue_path.name,
                    "byte_length": queue_path.stat().st_size,
                    "sha256": _file_sha(queue_path),
                },
            },
        }
        (stage / "manifest.json").write_bytes(_canonical(manifest))
        _publish_directory(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_factor_discovery_bundle(target / "manifest.json")


def verify_factor_discovery_bundle(
    manifest_path: str | Path,
) -> PublishedDiscoveryBundleV1:
    path = Path(manifest_path)
    value = _read_manifest(path, "FactorDiscoveryBundleV1")
    bundle_sha = value.get("bundle_sha256")
    row_count = value.get("result_row_count")
    code_lock_sha = value.get("code_lock_sha256")
    factor_registry_sha = value.get("factor_registry_sha256")
    if (
        value.get("builder_version") != _BUILDER_VERSION
        or type(bundle_sha) is not str
        or path.parent.name != bundle_sha.replace(":", "-")
        or type(row_count) is not int
        or row_count < 0
        or value.get("holdout_reaction_accessed") is not False
        or type(code_lock_sha) is not str
        or type(factor_registry_sha) is not str
    ):
        raise FactorLabArtifactError("discovery identity differs")
    files = value.get("files")
    if type(files) is not dict:
        raise FactorLabArtifactError("discovery files are missing")
    result_path, _ = _verify_file(path.parent, files.get("results"))
    queue_path, _ = _verify_file(path.parent, files.get("review_queue"))
    if pq.ParquetFile(result_path).metadata.num_rows != row_count:
        raise FactorLabArtifactError("discovery row count differs")
    return PublishedDiscoveryBundleV1(
        manifest_path=path,
        result_path=result_path,
        review_queue_path=queue_path,
        bundle_sha256=bundle_sha,
        result_row_count=row_count,
        code_lock_sha256=code_lock_sha,
        factor_registry_sha256=factor_registry_sha,
    )


def load_sports_feature_frame(manifest_path: str | Path) -> pd.DataFrame:
    """Verify the immutable publication before exposing its Parquet rows."""

    published = verify_sports_feature_view(manifest_path)
    return pd.read_parquet(published.data_path)


def load_factor_discovery_results(manifest_path: str | Path) -> pd.DataFrame:
    """Verify the immutable discovery bundle before exposing result rows."""

    published = verify_factor_discovery_bundle(manifest_path)
    return pd.read_parquet(published.result_path)


def load_human_review_queue(manifest_path: str | Path) -> dict[str, Any]:
    """Verify the bundle and return the still-unreviewed queue."""

    published = verify_factor_discovery_bundle(manifest_path)
    value = _read_manifest(
        published.review_queue_path, "HumanFactorReviewQueueV1"
    )
    if value.get("holdout_reaction_accessed") is not False:
        raise FactorLabArtifactError("review queue accessed holdout reactions")
    return value


__all__ = [
    "FactorLabArtifactError",
    "PublishedDiscoveryBundleV1",
    "PublishedFeatureViewV1",
    "load_factor_discovery_results",
    "load_human_review_queue",
    "load_sports_feature_frame",
    "publish_factor_discovery_bundle",
    "publish_sports_feature_view",
    "verify_factor_discovery_bundle",
    "verify_sports_feature_view",
]
