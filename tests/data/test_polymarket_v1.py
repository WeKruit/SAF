from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.polymarket_v1 import (
    BOUNDED_SHARD_PATH,
    FROZEN_NFL_CONDITIONS,
    POLYMARKET_V1_REVISION,
    PolymarketV1SourceError,
    build_exact_condition_extract,
    capture_polymarket_v1_poc,
    inspect_daily_aligned_shard,
    inspect_first_rows_response,
    inspect_gamma_market,
    inspect_parquet_index,
    inspect_repo_metadata,
    inspect_repo_tree,
)
from prediction_market.static_store import read_verified_static_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCHED_AT = datetime(2026, 7, 22, 18, 30, tzinfo=timezone.utc)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _repo_metadata() -> bytes:
    return _canonical(
        {
            "id": "TimeSeventeen/Polymarket-v1",
            "sha": POLYMARKET_V1_REVISION,
            "private": False,
            "gated": False,
            "tags": [
                "license:cc-by-4.0",
                "size_categories:1B<n<10B",
                "polymarket",
            ],
        }
    )


def _daily_rows() -> list[dict[str, object]]:
    conditions = list(FROZEN_NFL_CONDITIONS)
    rows: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        rows.append(
            {
                "asset_id": str(1000 + index),
                "block_timestamp": 1672531200 + index,
                "price": 0.40 + index / 100,
                "maker": f"0xmaker{index}",
                "taker": f"0xtaker{index}",
                "taker_direction": "BUY",
                "usdc_amount": 10.0 + index,
                "fee_usdc": 0.0,
                "condition_id": condition.condition_id,
                "outcome_seq": index % 2,
                "neg_risk": "f",
                "category": "deliberately-not-used-by-join",
                "category_refined": "deliberately-not-used-by-join",
                "outcome_label": "Yes",
                "winning_outcome_label": "Yes",
                "resolution_status": "resolved",
                "taker_base_fee": 0.0,
                "maker_base_fee": 0.0,
                "opens_at": "2022-12-27T00:00:00",
                "close_at": "2023-01-02T00:00:00",
                "resolved_at": None,
                "market_slug": condition.slug,
                "p_event": 0.40 + index / 100,
                "D": -1,
            }
        )
    rows.append(
        {
            **rows[0],
            "asset_id": "9999",
            "condition_id": "0x" + "9" * 64,
            "category_refined": "Sports",
            "market_slug": "nfl-looking-slug-must-not-be-used",
        }
    )
    return rows


def _parquet_bytes() -> bytes:
    table = pa.Table.from_pylist(_daily_rows())
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def _repo_tree(shard: bytes) -> bytes:
    return _canonical(
        [
            {
                "type": "directory",
                "oid": "d" * 40,
                "size": 0,
                "path": "daily_aligned",
            },
            {
                "type": "file",
                "oid": "f" * 40,
                "size": len(shard),
                "lfs": {
                    "oid": hashlib.sha256(shard).hexdigest(),
                    "size": len(shard),
                    "pointerSize": 132,
                },
                "path": BOUNDED_SHARD_PATH,
            },
        ]
    )


def _parquet_index() -> bytes:
    return _canonical(
        {
            "parquet_files": [
                {
                    "dataset": "TimeSeventeen/Polymarket-v1",
                    "config": "daily_aligned",
                    "split": "train",
                    "url": (
                        "https://huggingface.co/datasets/TimeSeventeen/"
                        "Polymarket-v1/resolve/refs%2Fconvert%2Fparquet/"
                        "daily_aligned/train/0000.parquet"
                    ),
                    "filename": "0000.parquet",
                    "size": len(_parquet_bytes()),
                },
                {
                    "dataset": "TimeSeventeen/Polymarket-v1",
                    "config": "orderfilled",
                    "split": "train",
                    "url": (
                        "https://huggingface.co/datasets/TimeSeventeen/"
                        "Polymarket-v1/resolve/refs%2Fconvert%2Fparquet/"
                        "orderfilled/train/0000.parquet"
                    ),
                    "filename": "0000.parquet",
                    "size": 123,
                },
                {
                    "dataset": "TimeSeventeen/Polymarket-v1",
                    "config": "ctf",
                    "split": "train",
                    "url": (
                        "https://huggingface.co/datasets/TimeSeventeen/"
                        "Polymarket-v1/resolve/refs%2Fconvert%2Fparquet/"
                        "ctf/train/0000.parquet"
                    ),
                    "filename": "0000.parquet",
                    "size": 456,
                },
            ],
            "pending": [],
            "failed": [],
        }
    )


def _first_rows() -> bytes:
    rows = _daily_rows()[:2]
    features = [
        {"feature_idx": index, "name": name, "type": {"dtype": "string"}}
        for index, name in enumerate(rows[0])
    ]
    return _canonical(
        {
            "dataset": "TimeSeventeen/Polymarket-v1",
            "config": "daily_aligned",
            "split": "train",
            "features": features,
            "rows": [
                {"row_idx": index, "row": row, "truncated_cells": []}
                for index, row in enumerate(rows)
            ],
            "truncated": False,
        }
    )


def _gamma_payload(condition_index: int) -> bytes:
    condition = FROZEN_NFL_CONDITIONS[condition_index]
    return _canonical(
        {
            "id": str(248292 + condition_index),
            "conditionId": condition.condition_id,
            "slug": condition.slug,
            "question": f"NFL fixture {condition_index}",
            "active": True,
            "closed": True,
            "events": [
                {
                    "id": str(886411 + condition_index),
                    "slug": condition.slug,
                    "seriesSlug": "nfl",
                }
            ],
        }
    )


def test_repo_metadata_and_tree_lock_revision_license_and_exact_lfs_hash() -> None:
    shard = _parquet_bytes()
    metadata = inspect_repo_metadata(_repo_metadata())
    tree = inspect_repo_tree(_repo_tree(shard))

    assert metadata.revision == POLYMARKET_V1_REVISION
    assert metadata.license_id == "CC-BY-4.0"
    assert tree.shard_path == BOUNDED_SHARD_PATH
    assert tree.shard_size == len(shard)
    assert tree.shard_sha256 == "sha256:" + hashlib.sha256(shard).hexdigest()

    wrong = json.loads(_repo_metadata())
    wrong["sha"] = "0" * 40
    with pytest.raises(PolymarketV1SourceError, match="revision"):
        inspect_repo_metadata(_canonical(wrong))

    wrong = json.loads(_repo_metadata())
    wrong["tags"].remove("license:cc-by-4.0")
    with pytest.raises(PolymarketV1SourceError, match="CC-BY-4.0"):
        inspect_repo_metadata(_canonical(wrong))


def test_dataset_server_endpoints_require_exact_response_revision_and_schema() -> None:
    parquet = inspect_parquet_index(
        _parquet_index(), response_revision=POLYMARKET_V1_REVISION
    )
    first_rows = inspect_first_rows_response(
        _first_rows(), response_revision=POLYMARKET_V1_REVISION
    )

    assert parquet.configs == ("ctf", "daily_aligned", "orderfilled")
    assert parquet.file_count == 3
    assert first_rows.row_count == 2
    assert first_rows.sample_truncated is False
    assert "condition_id" in first_rows.feature_names
    assert "block_timestamp" in first_rows.feature_names

    with pytest.raises(PolymarketV1SourceError, match="X-Revision"):
        inspect_parquet_index(_parquet_index(), response_revision="main")

    malformed = json.loads(_first_rows())
    malformed["features"] = [
        feature
        for feature in malformed["features"]
        if feature["name"] != "condition_id"
    ]
    for index, feature in enumerate(malformed["features"]):
        feature["feature_idx"] = index
    with pytest.raises(PolymarketV1SourceError, match="condition_id"):
        inspect_first_rows_response(
            _canonical(malformed), response_revision=POLYMARKET_V1_REVISION
        )


def test_daily_shard_audit_and_extract_use_condition_id_only() -> None:
    shard = _parquet_bytes()
    audit = inspect_daily_aligned_shard(
        shard,
        expected_size=len(shard),
        expected_sha256="sha256:" + hashlib.sha256(shard).hexdigest(),
    )
    extract = build_exact_condition_extract(
        shard,
        source_object_ref=audit.object_sha256,
        gamma_object_refs=tuple(
            "sha256:" + str(index) * 64 for index in range(1, 5)
        ),
    )

    assert audit.row_count == len(FROZEN_NFL_CONDITIONS) + 1
    assert audit.condition_count == len(FROZEN_NFL_CONDITIONS) + 1
    assert extract.row_count == len(FROZEN_NFL_CONDITIONS)
    assert extract.condition_count == len(FROZEN_NFL_CONDITIONS)
    assert b"nfl-looking-slug-must-not-be-used" not in extract.payload
    assert extract.query["join_key"] == "condition_id"
    assert "category" not in extract.query["predicate_fields"]
    assert "market_slug" not in extract.query["predicate_fields"]
    assert extract.lineage_refs[0] == audit.object_sha256
    assert extract.query_sha256.startswith("sha256:")
    assert extract.pit_safe_for_model_features is False
    assert extract.contains_l2 is False


def test_gamma_validation_requires_exact_condition_and_nfl_series() -> None:
    expected = FROZEN_NFL_CONDITIONS[0]
    audit = inspect_gamma_market(_gamma_payload(0), expected=expected)
    assert audit.condition_id == expected.condition_id
    assert audit.sport == "NFL"

    mismatch = json.loads(_gamma_payload(0))
    mismatch["conditionId"] = FROZEN_NFL_CONDITIONS[1].condition_id
    with pytest.raises(PolymarketV1SourceError, match="conditionId"):
        inspect_gamma_market(_canonical(mismatch), expected=expected)

    wrong_sport = json.loads(_gamma_payload(0))
    wrong_sport["events"][0]["seriesSlug"] = "politics"
    with pytest.raises(PolymarketV1SourceError, match="NFL"):
        inspect_gamma_market(_canonical(wrong_sport), expected=expected)


def test_capture_preserves_every_raw_response_and_derived_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import polymarket_v1 as source

    shard = _parquet_bytes()
    monkeypatch.setattr(source, "BOUNDED_SHARD_SIZE", len(shard))
    monkeypatch.setattr(
        source,
        "BOUNDED_SHARD_SHA256",
        "sha256:" + hashlib.sha256(shard).hexdigest(),
    )
    monkeypatch.setattr(
        source, "BOUNDED_SHARD_EXPECTED_ROWS", len(_daily_rows())
    )
    monkeypatch.setattr(
        source,
        "BOUNDED_EXTRACT_EXPECTED_ROWS",
        len(FROZEN_NFL_CONDITIONS),
    )
    raw_by_path = {
        "repo": _repo_metadata(),
        "tree": _repo_tree(shard),
        "parquet": _parquet_index(),
        "first-rows": _first_rows(),
        "shard": shard,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        if request.url.host == "huggingface.co":
            if "/revision/" in request.url.path:
                payload = raw_by_path["repo"]
            elif "/tree/" in request.url.path:
                payload = raw_by_path["tree"]
            elif "/resolve/" in request.url.path:
                payload = raw_by_path["shard"]
            else:
                return httpx.Response(404, request=request)
            return httpx.Response(
                200,
                content=payload,
                headers={
                    "Content-Length": str(len(payload)),
                    "Content-Type": (
                        "application/octet-stream"
                        if payload is shard
                        else "application/json"
                    ),
                    "ETag": '"fixture"',
                },
                request=request,
            )
        if request.url.host == "datasets-server.huggingface.co":
            if request.url.path == "/parquet":
                payload = raw_by_path["parquet"]
            elif request.url.path == "/first-rows":
                payload = raw_by_path["first-rows"]
            else:
                return httpx.Response(404, request=request)
            return httpx.Response(
                200,
                content=payload,
                headers={
                    "Content-Length": str(len(payload)),
                    "Content-Type": "application/json",
                    "X-Revision": POLYMARKET_V1_REVISION,
                },
                request=request,
            )
        if request.url.host == "gamma-api.polymarket.com":
            slug = request.url.path.rsplit("/", 1)[-1]
            index = next(
                index
                for index, condition in enumerate(FROZEN_NFL_CONDITIONS)
                if condition.slug == slug
            )
            payload = _gamma_payload(index)
            return httpx.Response(
                200,
                content=payload,
                headers={
                    "Content-Length": str(len(payload)),
                    "Content-Type": "application/json",
                },
                request=request,
            )
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        capture = capture_polymarket_v1_poc(
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert capture.source_shard.audit.row_count == len(_daily_rows())
    assert capture.sports_extract.audit.row_count == len(
        FROZEN_NFL_CONDITIONS
    )
    assert len(capture.gamma_markets) == len(FROZEN_NFL_CONDITIONS)
    assert capture.sports_extract.record.manifest.object_kind == (
        "source_derived_extract"
    )
    assert capture.sports_extract.record.manifest.license_ref == "R-039"
    assert capture.sports_extract.record.manifest.license_status == "approved"
    assert len(
        capture.sports_extract.record.manifest.lineage.source_object_refs
    ) == 1 + len(FROZEN_NFL_CONDITIONS)

    originals = (
        capture.repo_metadata,
        capture.repo_tree,
        capture.parquet_index,
        capture.first_rows,
        capture.source_shard.record,
        *(market.record for market in capture.gamma_markets),
    )
    for record in originals:
        assert record.manifest.object_kind == "byte_exact_original"
        verified = read_verified_static_object(
            record.manifest_path,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )
        assert hashlib.sha256(verified.object_bytes).hexdigest() in (
            record.manifest.object_sha256
        )

    assert all(
        market.record.manifest.dataset_id == "DS-POLYMARKET-PUBLIC"
        and market.record.manifest.license_ref == "O-001"
        and market.record.manifest.license_status == "pending"
        for market in capture.gamma_markets
    )
