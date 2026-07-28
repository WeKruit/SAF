"""Publish the verified X-13/X-14/PMXT tick-data capability inventory.

The builder is deliberately read-only with respect to its source evidence.  It
does not contain a holdout reader, network client, PMXT hydrate path, or venue
API client.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


class TickDataInventoryError(ValueError):
    """Source evidence cannot support the requested inventory claim."""


_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRADE_ALLOWED = [
    "actual_trade_price_path",
    "direction_probability",
    "empirical_distribution",
    "calibration",
    "reference_gap",
    "same_episode_venue_pair",
    "matched_control",
]
_TRADE_FORBIDDEN = [
    "ofi",
    "depth",
    "bid_ask_lead",
    "executable_spread",
    "queue_position",
    "fill_probability",
    "real_reaction_latency",
]
_L2_METHOD_ALLOWED = [
    "spread_method",
    "top_n_depth_method",
    "book_imbalance_method",
    "microprice_method",
    "ofi_method",
    "depth_withdrawal_method",
    "cancel_intensity_method",
    "resilience_method",
    "markout_method",
    "book_walk_method",
]


@dataclass(frozen=True, slots=True)
class TickDataInventoryPublication:
    rows_path: Path
    manifest_path: Path
    rows_sha256: str
    manifest_sha256: str


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TickDataInventoryError("inventory is not canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise TickDataInventoryError(f"{name} is missing: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TickDataInventoryError(f"{name} is invalid JSON") from exc
    if not isinstance(document, dict):
        raise TickDataInventoryError(f"{name} must be a JSON object")
    return document, _sha_bytes(raw)


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise TickDataInventoryError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _exact_root(batch_manifest_path: Path) -> Path:
    for candidate in (batch_manifest_path, *batch_manifest_path.parents):
        if candidate.name == "exact-153":
            return candidate
    return batch_manifest_path.parent


def _fraction_digits(value: str | None) -> int:
    if not value or "." not in value:
        return 0
    tail = value.split(".", 1)[1]
    fraction = tail.rstrip("Z").split("+", 1)[0]
    if not fraction or set(fraction) == {"0"}:
        return 0
    return len(fraction)


def _scan_exact_trade_batch(
    *,
    batch_manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    batch, batch_file_sha = _load_json(batch_manifest_path, name="exact batch manifest")
    games = batch.get("games")
    if not isinstance(games, list) or not games:
        raise TickDataInventoryError("exact batch games must be a nonempty list")
    if batch.get("game_count") != len(games):
        raise TickDataInventoryError("exact batch game_count does not match games")
    if batch.get("final_holdout_access") != "CLOSED":
        raise TickDataInventoryError("exact batch must keep final holdout CLOSED")

    exact_root = _exact_root(batch_manifest_path)
    stats: dict[str, dict[str, Any]] = {
        "polymarket": {
            "trade_count": 0,
            "games": set(),
            "coverage_start": None,
            "coverage_end": None,
            "max_fraction_digits": 0,
        },
        "kalshi": {
            "trade_count": 0,
            "games": set(),
            "coverage_start": None,
            "coverage_end": None,
            "max_fraction_digits": 0,
        },
    }
    manifest_hashes: list[str] = []
    object_hashes: list[str] = []

    for game in games:
        if not isinstance(game, dict):
            raise TickDataInventoryError("exact batch game entry must be an object")
        game_id = game.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise TickDataInventoryError("exact batch game_id is invalid")
        manifest_path_value = game.get("manifest_path")
        if not isinstance(manifest_path_value, str):
            raise TickDataInventoryError(f"{game_id}: manifest_path is invalid")
        manifest_path = exact_root / manifest_path_value
        manifest, actual_manifest_sha = _load_json(
            manifest_path, name=f"{game_id} manifest"
        )
        expected_manifest_sha = _require_sha(
            game.get("manifest_sha256"), field=f"{game_id} manifest_sha256"
        )
        if actual_manifest_sha != expected_manifest_sha:
            raise TickDataInventoryError(f"{game_id}: manifest SHA-256 mismatch")
        manifest_hashes.append(actual_manifest_sha)

        stages = manifest.get("stages")
        stage = (
            stages.get("actual_market_observations")
            if isinstance(stages, dict)
            else None
        )
        if not isinstance(stage, dict):
            raise TickDataInventoryError(
                f"{game_id}: actual_market_observations stage is missing"
            )
        object_path_value = stage.get("object_path")
        if not isinstance(object_path_value, str):
            raise TickDataInventoryError(
                f"{game_id}: actual observation object_path is invalid"
            )
        object_path = exact_root / object_path_value
        expected_object_sha = _require_sha(
            stage.get("object_sha256"),
            field=f"{game_id} actual observation object_sha256",
        )
        if not object_path.is_file() or _sha_file(object_path) != expected_object_sha:
            raise TickDataInventoryError(
                f"{game_id}: actual observation object SHA-256 mismatch"
            )
        object_hashes.append(expected_object_sha)

        try:
            table = pq.read_table(
                object_path,
                columns=[
                    "game_id",
                    "venue",
                    "kind",
                    "native_source_time_utc",
                ],
            )
        except Exception as exc:
            raise TickDataInventoryError(
                f"{game_id}: cannot read actual observation object"
            ) from exc
        if table.num_rows != stage.get("row_count"):
            raise TickDataInventoryError(
                f"{game_id}: actual observation row_count mismatch"
            )
        trade_table = table.filter(pc.equal(table["kind"], "trade"))
        for venue in stats:
            venue_table = trade_table.filter(pc.equal(trade_table["venue"], venue))
            count = venue_table.num_rows
            if count == 0:
                continue
            game_values = set(venue_table["game_id"].to_pylist())
            if game_values != {game_id}:
                raise TickDataInventoryError(
                    f"{game_id}: actual observation game identity mismatch"
                )
            times = venue_table["native_source_time_utc"]
            min_time = pc.min(times).as_py()
            max_time = pc.max(times).as_py()
            sample_times = times.slice(0, min(64, count)).to_pylist()
            stat = stats[venue]
            stat["trade_count"] += count
            stat["games"].add(game_id)
            if stat["coverage_start"] is None or min_time < stat["coverage_start"]:
                stat["coverage_start"] = min_time
            if stat["coverage_end"] is None or max_time > stat["coverage_end"]:
                stat["coverage_end"] = max_time
            stat["max_fraction_digits"] = max(
                stat["max_fraction_digits"],
                *(_fraction_digits(value) for value in (min_time, max_time, *sample_times)),
            )

    for venue, stat in stats.items():
        if not stat["trade_count"]:
            raise TickDataInventoryError(f"exact batch has no {venue} actual trades")
        stat["game_count"] = len(stat.pop("games"))

    lineage = {
        "batch_manifest_path": str(batch_manifest_path),
        "batch_manifest_file_sha256": batch_file_sha,
        "game_manifest_count": len(manifest_hashes),
        "game_manifest_bundle_sha256": _sha_bytes(
            _canonical_bytes(sorted(manifest_hashes))
        ),
        "actual_object_count": len(object_hashes),
        "actual_object_bundle_sha256": _sha_bytes(
            _canonical_bytes(sorted(object_hashes))
        ),
    }
    return stats, lineage


def _evidence_ref(path: Path, repo_root: Path, digest: str) -> dict[str, str]:
    return {"path": _relative(path, repo_root), "sha256": digest}


def build_tick_data_inventory(
    *,
    repo_root: Path,
    batch_manifest_path: Path,
    pmxt_verify: Path,
    pmxt_audit: Path,
    x14: Path,
    kalshi_candle: Path,
    timeseventeen: Path,
) -> dict[str, Any]:
    """Build one exact-batch inventory strictly from local verified evidence."""

    stats, exact_lineage = _scan_exact_trade_batch(
        batch_manifest_path=batch_manifest_path
    )
    exact_lineage["batch_manifest_path"] = _relative(
        batch_manifest_path, repo_root
    )
    pmxt_verify_doc, pmxt_verify_sha = _load_json(
        pmxt_verify, name="PMXT remote verification"
    )
    pmxt_audit_doc, pmxt_audit_sha = _load_json(
        pmxt_audit, name="PMXT L2 method audit"
    )
    x14_doc, x14_sha = _load_json(x14, name="X-14 session")
    candle_doc, candle_sha = _load_json(
        kalshi_candle, name="Kalshi candle evidence"
    )
    ts_doc, ts_sha = _load_json(timeseventeen, name="TimeSeventeen evidence")

    remote_objects = pmxt_verify_doc.get("objects")
    if not isinstance(remote_objects, list) or not remote_objects:
        raise TickDataInventoryError("PMXT verification objects are missing")
    for item in remote_objects:
        if not isinstance(item, dict) or item.get("status") != "verified":
            raise TickDataInventoryError("PMXT verification contains a failed object")
        _require_sha(item.get("sha256"), field="PMXT object sha256")
        if item.get("actual_sha256") not in (None, item["sha256"]):
            raise TickDataInventoryError("PMXT actual SHA-256 does not match")
        if not isinstance(item.get("byte_size"), int) or item["byte_size"] < 0:
            raise TickDataInventoryError("PMXT byte_size is invalid")
    remote_count = len(remote_objects)
    remote_bytes = sum(item["byte_size"] for item in remote_objects)
    crosswalk = pmxt_audit_doc.get("crosswalk")
    if (
        pmxt_audit_doc.get("audit_kind") != "pmxt_l2_method_audit_v1"
        or not isinstance(crosswalk, dict)
        or crosswalk.get("attested_remote_object_count") != remote_count
        or crosswalk.get("unattested_remote_object_count") != 0
    ):
        raise TickDataInventoryError("PMXT method audit crosswalk is not complete")
    raw_parquet_scanned = bool(pmxt_audit_doc.get("raw_parquet_scanned", False))
    if pmxt_audit_doc.get("data_access") != "local_fixture_and_metadata_only":
        raise TickDataInventoryError("PMXT method audit access boundary changed")
    if raw_parquet_scanned:
        raise TickDataInventoryError("PMXT method audit cannot claim a raw scan")

    raw_capture = candle_doc.get("raw_capture")
    kalshi_capture = (
        raw_capture.get("kalshi") if isinstance(raw_capture, dict) else None
    )
    candle_captures = (
        kalshi_capture.get("candle_captures")
        if isinstance(kalshi_capture, dict)
        else None
    )
    if not isinstance(candle_captures, dict) or not candle_captures:
        raise TickDataInventoryError("Kalshi NFL candle capability is unproven")
    candle_count = 0
    for ticker, capture in candle_captures.items():
        if not isinstance(ticker, str) or not isinstance(capture, dict):
            raise TickDataInventoryError("Kalshi NFL candle capture is invalid")
        if capture.get("interval_minutes") != 1:
            raise TickDataInventoryError("Kalshi candle interval is not one minute")
        count = capture.get("candles")
        if not isinstance(count, int) or count <= 0:
            raise TickDataInventoryError("Kalshi NFL candle count is invalid")
        _require_sha(
            capture.get("manifest_sha256"),
            field=f"Kalshi candle {ticker} manifest_sha256",
        )
        candle_count += count
    candle_game_id = candle_doc.get("canonical_game_id")
    if not isinstance(candle_game_id, str) or not candle_game_id:
        raise TickDataInventoryError("Kalshi NFL candle game identity is missing")

    x14_capture = x14_doc.get("capture")
    x14_health = x14_capture.get("health") if isinstance(x14_capture, dict) else None
    x14_coverage = x14_doc.get("coverage")
    if not isinstance(x14_health, dict) or not isinstance(x14_coverage, dict):
        raise TickDataInventoryError("X-14 capture evidence is incomplete")

    ts_dataset = ts_doc.get("dataset")
    ts_extract = ts_doc.get("derived_extract")
    if not isinstance(ts_dataset, dict) or not isinstance(ts_extract, dict):
        raise TickDataInventoryError("TimeSeventeen bounded POC evidence is incomplete")
    _require_sha(
        ts_extract.get("object_sha256"), field="TimeSeventeen extract object_sha256"
    )
    _require_sha(
        ts_extract.get("manifest_sha256"),
        field="TimeSeventeen extract manifest_sha256",
    )

    batch_ref = _evidence_ref(
        batch_manifest_path,
        repo_root,
        exact_lineage["batch_manifest_file_sha256"],
    )
    pmxt_verify_ref = _evidence_ref(pmxt_verify, repo_root, pmxt_verify_sha)
    pmxt_audit_ref = _evidence_ref(pmxt_audit, repo_root, pmxt_audit_sha)
    x14_ref = _evidence_ref(x14, repo_root, x14_sha)
    candle_ref = _evidence_ref(kalshi_candle, repo_root, candle_sha)
    ts_ref = _evidence_ref(timeseventeen, repo_root, ts_sha)

    poly = stats["polymarket"]
    kalshi = stats["kalshi"]
    rows = [
        {
            "schema_version": "TickDataInventoryV1",
            "source_id": "x13-2025-polymarket-trades",
            "venue": "polymarket",
            "data_grade": "TRADE_TICK",
            "availability": "LOCAL_VERIFIED_EXACT_BATCH",
            "capabilities": ["trade_tick"],
            "coverage": {
                "cohort": "development",
                "game_count": poly["game_count"],
                "trade_count": poly["trade_count"],
                "start_utc": poly["coverage_start"],
                "end_utc": poly["coverage_end"],
            },
            "timestamp_domains": ["venue_source_time"],
            "timestamp_precision": (
                "microsecond" if poly["max_fraction_digits"] >= 6 else "second"
            ),
            "direction_authority": "actual_public_trade_outcome_orientation",
            "candle_capability": {"status": "NOT_APPLICABLE"},
            "allowed_metrics": _TRADE_ALLOWED,
            "forbidden_metrics": _TRADE_FORBIDDEN,
            "license": {"review_id": "O-001", "status": "research_only"},
            "gap": {
                "status": "KNOWN_BOUNDARY",
                "details": "No historical BBO/L2 or local receive timestamp.",
            },
            "claim_boundary": "HISTORICAL_SOURCE_TIME_CORRELATION_ONLY",
            "evidence_refs": [batch_ref],
        },
        {
            "schema_version": "TickDataInventoryV1",
            "source_id": "x13-2025-kalshi-trades",
            "venue": "kalshi",
            "data_grade": "TRADE_TICK_PLUS_BOUNDED_CANDLE_CAPABILITY",
            "availability": "LOCAL_VERIFIED_EXACT_BATCH",
            "capabilities": ["trade_tick", "candle_l1"],
            "coverage": {
                "cohort": "development",
                "game_count": kalshi["game_count"],
                "trade_count": kalshi["trade_count"],
                "start_utc": kalshi["coverage_start"],
                "end_utc": kalshi["coverage_end"],
            },
            "timestamp_domains": ["venue_native_trade_time"],
            "timestamp_precision": (
                "microsecond" if kalshi["max_fraction_digits"] >= 6 else "second"
            ),
            "direction_authority": "actual_public_trade_outcome_orientation",
            "candle_capability": {
                "status": "BOUNDED_NFL_ONE_GAME_NOT_EXACT_BATCH_COVERAGE",
                "interval_minutes": 1,
                "observed_candles": candle_count,
                "observed_game_count": 1,
                "observed_ticker_count": len(candle_captures),
            },
            "allowed_metrics": _TRADE_ALLOWED + ["minute_candle_background"],
            "forbidden_metrics": _TRADE_FORBIDDEN,
            "license": {"review_id": "O-003", "status": "research_only"},
            "gap": {
                "status": "KNOWN_BOUNDARY",
                "details": "No historical L2; candle evidence does not prove exact-153 candle coverage.",
            },
            "claim_boundary": "HISTORICAL_SOURCE_TIME_CORRELATION_ONLY",
            "evidence_refs": [batch_ref, candle_ref],
        },
        {
            "schema_version": "TickDataInventoryV1",
            "source_id": "pmxt-v2-s3-mirror",
            "venue": "polymarket",
            "data_grade": "L2_METHOD_ONLY",
            "availability": "REMOTE_OBJECTS_SHA256_VERIFIED",
            "capabilities": ["l2"],
            "coverage": {
                "object_count": remote_count,
                "byte_count": remote_bytes,
                "sha256_verified_count": remote_count,
            },
            "timestamp_domains": ["exchange_source_time", "exporter_receive_time"],
            "timestamp_precision": "source_schema_dependent",
            "direction_authority": "unavailable_without_authoritative_fills",
            "raw_parquet_scanned": False,
            "allowed_metrics": _L2_METHOD_ALLOWED,
            "forbidden_metrics": [
                "nfl_2025_alpha",
                "aggressor_side_formal",
                "l3_order_identity",
                "queue_position",
                "fill_probability",
            ],
            "license": {"review_id": "O-006", "status": "approved"},
            "gap": {
                "status": "METHOD_ONLY_NO_RAW_SCAN",
                "details": "97 remote objects are hash-attested; this publication did not scan their Parquet payloads.",
            },
            "claim_boundary": "L2_METHOD_VALIDATION_NOT_NFL_ALPHA",
            "evidence_refs": [pmxt_verify_ref, pmxt_audit_ref],
        },
        {
            "schema_version": "TickDataInventoryV1",
            "source_id": "x14-polymarket-smoke",
            "venue": "polymarket",
            "data_grade": "BOUNDED_L2_SMOKE",
            "availability": "PARTIAL_CAPTURE",
            "capabilities": ["l2"],
            "coverage": {
                "frame_count": x14_health.get("frames"),
                "sealed_segment_count": x14_health.get("sealed_segments"),
                "status": x14_coverage.get("status"),
            },
            "timestamp_domains": ["exchange_source_time", "local_receive_time"],
            "timestamp_precision": "capture_envelope",
            "direction_authority": "unavailable_without_authoritative_fills",
            "allowed_metrics": ["collector_continuity_smoke"],
            "forbidden_metrics": [
                "reaction_latency",
                "nfl_alpha",
                "execution",
                "creation_to_resolution_claim",
            ],
            "license": {"review_id": "O-001", "status": "research_only"},
            "gap": {
                "status": "PARTIAL_LATE_START",
                "details": "One-frame bounded smoke; not a complete NFL session.",
            },
            "claim_boundary": "OPERATIONAL_SMOKE_ONLY",
            "evidence_refs": [x14_ref],
        },
        {
            "schema_version": "TickDataInventoryV1",
            "source_id": "kalshi-live-l2",
            "venue": "kalshi",
            "data_grade": "NO_CAPTURE",
            "availability": "ABSENT_RSA_KEY_REQUIRED",
            "capabilities": [],
            "coverage": {"object_count": 0, "frame_count": 0},
            "timestamp_domains": [],
            "timestamp_precision": "unavailable",
            "direction_authority": "unavailable",
            "allowed_metrics": [],
            "forbidden_metrics": [
                "ofi",
                "depth",
                "bid_ask_lead",
                "reaction_latency",
                "execution",
            ],
            "license": {"review_id": "O-003", "status": "pending"},
            "gap": {
                "status": "CREDENTIAL_BLOCKED",
                "details": "Authenticated live L2 requires a Kalshi RSA API key.",
            },
            "claim_boundary": "NO_DATA_NO_METRIC",
            "evidence_refs": [],
        },
        {
            "schema_version": "TickDataInventoryV1",
            "source_id": "timeseventeen-polymarket-v1",
            "venue": "polymarket",
            "data_grade": "BOUNDED_POC_ONLY",
            "availability": "LOCAL_BOUNDED_EXTRACT",
            "capabilities": ["fills", "lifecycle"],
            "coverage": {
                "bounded_extract_row_count": ts_extract.get("row_count"),
                "full_dataset_mirrored": False,
            },
            "timestamp_domains": ["dataset_source_time_requires_schema_audit"],
            "timestamp_precision": "source_schema_dependent",
            "direction_authority": "authoritative_fill_fields_require_full_schema_audit",
            "allowed_metrics": ["bounded_schema_and_identity_validation"],
            "forbidden_metrics": [
                "historical_l2",
                "queue_position",
                "exact_153_coverage",
                "formal_aggressor_statistics",
            ],
            "license": {
                "review_id": "R-039",
                "license_id": ts_dataset.get("license_id"),
                "status": ts_dataset.get("license_status"),
            },
            "gap": {
                "status": "FULL_DATASET_NOT_MIRRORED",
                "details": "Only the bounded POC is local; this is not L2.",
            },
            "claim_boundary": "BOUNDED_SCHEMA_POC_ONLY",
            "evidence_refs": [ts_ref],
        },
    ]
    return {
        "schema_version": "TickDataInventoryV1",
        "experiment_id": "X-13/X-14",
        "cohort": "exact-153-development",
        "holdout_access": "CLOSED_UNREAD",
        "source_evidence": {
            "exact_batch": exact_lineage,
            "inputs": [
                batch_ref,
                pmxt_verify_ref,
                pmxt_audit_ref,
                x14_ref,
                candle_ref,
                ts_ref,
            ],
        },
        "rows": rows,
        "claim_boundary": (
            "Inventory of locally evidenced data capabilities; no causality, "
            "execution, real-money, maker, hedge, or tradeability claim."
        ),
    }


def _atomic_publish(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise TickDataInventoryError(f"content-address collision: {path}")
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        temp = Path(temp_name)
        if temp.exists():
            temp.unlink()


def publish_tick_data_inventory(
    *,
    output_root: Path,
    repo_root: Path,
    batch_manifest_path: Path,
    pmxt_verify: Path,
    pmxt_audit: Path,
    x14: Path,
    kalshi_candle: Path,
    timeseventeen: Path,
) -> TickDataInventoryPublication:
    document = build_tick_data_inventory(
        repo_root=repo_root,
        batch_manifest_path=batch_manifest_path,
        pmxt_verify=pmxt_verify,
        pmxt_audit=pmxt_audit,
        x14=x14,
        kalshi_candle=kalshi_candle,
        timeseventeen=timeseventeen,
    )
    rows_raw = _canonical_bytes(document)
    rows_sha = _sha_bytes(rows_raw)
    rows_hex = rows_sha.removeprefix("sha256:")
    rows_path = output_root / "objects" / "sha256" / rows_hex[:2] / f"{rows_hex}.json"
    _atomic_publish(rows_path, rows_raw)

    manifest = {
        "schema_version": "TickDataInventoryManifestV1",
        "artifact_id": "NFL_X13_X14_TICK_DATA_INVENTORY",
        "artifact_version": "v1",
        "rows": {
            "path": str(rows_path.relative_to(output_root)),
            "sha256": rows_sha,
            "row_count": len(document["rows"]),
        },
        "holdout_access": document["holdout_access"],
        "source_inputs": document["source_evidence"]["inputs"],
        "claim_boundary": document["claim_boundary"],
    }
    manifest_raw = _canonical_bytes(manifest)
    manifest_sha = _sha_bytes(manifest_raw)
    manifest_hex = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        output_root
        / "manifests"
        / "sha256"
        / manifest_hex[:2]
        / f"{manifest_hex}.manifest.json"
    )
    _atomic_publish(manifest_path, manifest_raw)
    return verify_tick_data_inventory_publication(manifest_path)


def verify_tick_data_inventory_publication(
    manifest_path: Path,
) -> TickDataInventoryPublication:
    manifest, manifest_sha = _load_json(
        manifest_path, name="tick inventory manifest"
    )
    expected_manifest_name = (
        manifest_sha.removeprefix("sha256:") + ".manifest.json"
    )
    if manifest_path.name != expected_manifest_name:
        raise TickDataInventoryError("manifest path is not content addressed")
    rows = manifest.get("rows")
    if not isinstance(rows, dict):
        raise TickDataInventoryError("inventory manifest rows binding is missing")
    rows_sha = _require_sha(rows.get("sha256"), field="rows sha256")
    output_root = manifest_path.parents[3]
    rows_path_value = rows.get("path")
    if not isinstance(rows_path_value, str):
        raise TickDataInventoryError("inventory rows path is invalid")
    rows_path = output_root / rows_path_value
    if not rows_path.is_file() or _sha_file(rows_path) != rows_sha:
        raise TickDataInventoryError("rows SHA-256 mismatch")
    expected_rows_name = rows_sha.removeprefix("sha256:") + ".json"
    if rows_path.name != expected_rows_name:
        raise TickDataInventoryError("rows path is not content addressed")
    document, _ = _load_json(rows_path, name="tick inventory rows")
    document_rows = document.get("rows")
    if not isinstance(document_rows, list) or len(document_rows) != rows.get("row_count"):
        raise TickDataInventoryError("inventory row_count mismatch")
    if manifest.get("holdout_access") != "CLOSED_UNREAD":
        raise TickDataInventoryError("inventory publication opened the holdout")
    return TickDataInventoryPublication(
        rows_path=rows_path,
        manifest_path=manifest_path,
        rows_sha256=rows_sha,
        manifest_sha256=manifest_sha,
    )


__all__ = [
    "TickDataInventoryError",
    "TickDataInventoryPublication",
    "build_tick_data_inventory",
    "publish_tick_data_inventory",
    "verify_tick_data_inventory_publication",
]
