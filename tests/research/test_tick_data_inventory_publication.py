from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.research.tick_data_inventory_publication import (
    TickDataInventoryError,
    build_tick_data_inventory,
    publish_tick_data_inventory,
    verify_tick_data_inventory_publication,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


def _write_exact_batch(root: Path) -> Path:
    exact_root = root / "exact"
    games = []
    venue_rows = {
        "game-a": [
            {
                "game_id": "game-a",
                "venue": "kalshi",
                "kind": "trade",
                "native_source_time_utc": "2025-09-05T00:18:01.538827Z",
                "source_start_utc": "2025-09-05T00:18:01.538827Z",
            },
            {
                "game_id": "game-a",
                "venue": "polymarket",
                "kind": "trade",
                "native_source_time_utc": "2025-09-05T00:18:13.000000Z",
                "source_start_utc": "2025-09-05T00:18:13.000000Z",
            },
        ],
        "game-b": [
            {
                "game_id": "game-b",
                "venue": "kalshi",
                "kind": "trade",
                "native_source_time_utc": "2025-09-06T00:18:01.123456Z",
                "source_start_utc": "2025-09-06T00:18:01.123456Z",
            },
            {
                "game_id": "game-b",
                "venue": "polymarket",
                "kind": "trade",
                "native_source_time_utc": "2025-09-06T00:18:13.000000Z",
                "source_start_utc": "2025-09-06T00:18:13.000000Z",
            },
        ],
    }
    for game_id, rows in venue_rows.items():
        game_root = exact_root / "single-game" / game_id
        object_path = game_root / "objects" / "actual.parquet"
        object_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), object_path)
        object_sha = _sha(object_path.read_bytes())
        manifest = {
            "game_id": game_id,
            "stages": {
                "actual_market_observations": {
                    "object_path": str(object_path.relative_to(exact_root)),
                    "object_sha256": object_sha,
                    "row_count": len(rows),
                }
            },
        }
        manifest_path = game_root / "manifest.json"
        manifest_sha = _write_json(manifest_path, manifest)
        games.append(
            {
                "game_id": game_id,
                "manifest_path": str(manifest_path.relative_to(exact_root)),
                "manifest_sha256": manifest_sha,
            }
        )
    batch = {
        "schema": "fixture",
        "game_count": 2,
        "games": games,
        "final_holdout_access": "CLOSED",
    }
    batch_path = exact_root / "batch.json"
    _write_json(batch_path, batch)
    return batch_path


def _write_supporting_evidence(root: Path) -> dict[str, Path]:
    paths = {
        "pmxt_verify": root / "pmxt-verify.json",
        "pmxt_audit": root / "pmxt-audit.json",
        "x14": root / "x14.json",
        "kalshi_candle": root / "kalshi-candle.json",
        "timeseventeen": root / "timeseventeen.json",
    }
    _write_json(
        paths["pmxt_verify"],
        {
            "audit_kind": "pmxt_s3_verify_v1",
            "objects": [
                {"sha256": "sha256:" + "a" * 64, "byte_size": 10, "status": "verified"},
                {"sha256": "sha256:" + "b" * 64, "byte_size": 20, "status": "verified"},
            ],
        },
    )
    _write_json(
        paths["pmxt_audit"],
        {
            "audit_kind": "pmxt_l2_method_audit_v1",
            "data_access": "local_fixture_and_metadata_only",
            "raw_parquet_scanned": False,
            "crosswalk": {
                "attested_remote_object_count": 2,
                "unattested_remote_object_count": 0,
            },
        },
    )
    _write_json(
        paths["x14"],
        {
            "experiment_id": "X-14",
            "capture": {"health": {"frames": 1, "sealed_segments": 1}},
            "coverage": {"status": "PARTIAL_LATE_START"},
            "claims": {"creation_to_resolution_complete": False},
            "report_sha256": "sha256:" + "c" * 64,
        },
    )
    _write_json(
        paths["kalshi_candle"],
        {
            "canonical_game_id": "game_nflverse_fixture",
            "raw_capture": {
                "kalshi": {
                    "candle_captures": {
                        "KXNFLGAME-A": {
                            "interval_minutes": 1,
                            "candles": 207,
                            "manifest_sha256": "sha256:" + "1" * 64,
                        },
                        "KXNFLGAME-B": {
                            "interval_minutes": 1,
                            "candles": 207,
                            "manifest_sha256": "sha256:" + "2" * 64,
                        },
                    },
                    "license_status": "pending",
                }
            },
        },
    )
    _write_json(
        paths["timeseventeen"],
        {
            "artifact_id": "POLYMARKET_V1_BOUNDED_SPORTS_EXTRACT",
            "status": "PRELIMINARY_RESEARCH_ONLY",
            "dataset": {
                "dataset_id": "DS-POLYMARKET-V1",
                "license_id": "CC-BY-4.0",
                "license_status": "approved",
            },
            "derived_extract": {
                "row_count": 17,
                "object_sha256": "sha256:" + "d" * 64,
                "manifest_sha256": "sha256:" + "e" * 64,
            },
        },
    )
    return paths


def test_build_inventory_derives_trade_counts_and_fail_closed_capabilities(
    tmp_path: Path,
) -> None:
    batch_path = _write_exact_batch(tmp_path)
    evidence = _write_supporting_evidence(tmp_path)

    document = build_tick_data_inventory(
        repo_root=tmp_path,
        batch_manifest_path=batch_path,
        **evidence,
    )

    rows = {row["source_id"]: row for row in document["rows"]}
    assert document["holdout_access"] == "CLOSED_UNREAD"
    assert document["source_evidence"]["exact_batch"]["batch_manifest_path"] == (
        "exact/batch.json"
    )
    assert rows["x13-2025-polymarket-trades"]["coverage"]["trade_count"] == 2
    assert rows["x13-2025-polymarket-trades"]["coverage"]["game_count"] == 2
    assert rows["x13-2025-polymarket-trades"]["timestamp_precision"] == "second"
    assert rows["x13-2025-kalshi-trades"]["coverage"]["trade_count"] == 2
    assert rows["x13-2025-kalshi-trades"]["timestamp_precision"] == "microsecond"
    assert rows["x13-2025-kalshi-trades"]["candle_capability"] == {
        "status": "BOUNDED_NFL_ONE_GAME_NOT_EXACT_BATCH_COVERAGE",
        "interval_minutes": 1,
        "observed_candles": 414,
        "observed_game_count": 1,
        "observed_ticker_count": 2,
    }
    assert "ofi" not in rows["x13-2025-kalshi-trades"]["allowed_metrics"]
    assert {"ofi", "depth", "bid_ask_lead", "executable_spread"} <= set(
        rows["x13-2025-kalshi-trades"]["forbidden_metrics"]
    )
    assert rows["pmxt-v2-s3-mirror"]["coverage"] == {
        "object_count": 2,
        "byte_count": 30,
        "sha256_verified_count": 2,
    }
    assert rows["pmxt-v2-s3-mirror"]["raw_parquet_scanned"] is False
    assert rows["pmxt-v2-s3-mirror"]["data_grade"] == "L2_METHOD_ONLY"
    assert rows["x14-polymarket-smoke"]["coverage"]["frame_count"] == 1
    assert rows["kalshi-live-l2"]["availability"] == "ABSENT_RSA_KEY_REQUIRED"
    assert rows["timeseventeen-polymarket-v1"]["data_grade"] == "BOUNDED_POC_ONLY"


def test_publication_is_content_addressed_and_detects_tampering(tmp_path: Path) -> None:
    batch_path = _write_exact_batch(tmp_path)
    evidence = _write_supporting_evidence(tmp_path)
    out = tmp_path / "artifacts" / "data-audit" / "tick-data-inventory-v1"

    published = publish_tick_data_inventory(
        output_root=out,
        repo_root=tmp_path,
        batch_manifest_path=batch_path,
        **evidence,
    )
    verified = verify_tick_data_inventory_publication(published.manifest_path)
    assert verified.rows_sha256 == published.rows_sha256
    assert verified.manifest_sha256 == published.manifest_sha256
    assert published.rows_path.name == published.rows_sha256.removeprefix("sha256:") + ".json"
    assert (
        published.manifest_path.name
        == published.manifest_sha256.removeprefix("sha256:") + ".manifest.json"
    )

    published.rows_path.write_bytes(published.rows_path.read_bytes() + b" ")
    with pytest.raises(TickDataInventoryError, match="rows SHA-256 mismatch"):
        verify_tick_data_inventory_publication(published.manifest_path)


def test_build_inventory_rejects_modified_exact_game_object(tmp_path: Path) -> None:
    batch_path = _write_exact_batch(tmp_path)
    evidence = _write_supporting_evidence(tmp_path)
    batch = json.loads(batch_path.read_text())
    game_manifest_path = batch_path.parent / batch["games"][0]["manifest_path"]
    game_manifest = json.loads(game_manifest_path.read_text())
    object_path = (
        batch_path.parent
        / game_manifest["stages"]["actual_market_observations"]["object_path"]
    )
    object_path.write_bytes(object_path.read_bytes() + b"tamper")

    with pytest.raises(TickDataInventoryError, match="actual observation object SHA-256"):
        build_tick_data_inventory(
            repo_root=tmp_path,
            batch_manifest_path=batch_path,
            **evidence,
        )
