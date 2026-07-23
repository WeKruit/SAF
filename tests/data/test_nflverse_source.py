from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.sports.nflverse import (
    NFLVERSE_RELEASE_VERSION,
    NFLVerseSourceError,
    fetch_and_preserve_nflverse_year,
    inspect_nflverse_partition,
)
from prediction_market.static_store import read_verified_static_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCHED_AT = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)


def _parquet_bytes(*, include_game_id: bool = True) -> bytes:
    values: dict[str, pa.Array] = {
        "play_id": pa.array([1.0, 2.0]),
        "game_id": pa.array(["2025_01_A_B", "2025_01_A_B"]),
        "season": pa.array([2025, 2025], type=pa.int32()),
        "season_type": pa.array(["REG", "REG"]),
        "game_date": pa.array(["2025-09-01", "2025-09-01"]),
        "home_team": pa.array(["B", "B"]),
        "away_team": pa.array(["A", "A"]),
        "posteam": pa.array(["A", "B"]),
        "score_differential": pa.array([0.0, 7.0]),
        "game_seconds_remaining": pa.array([3600.0, 3500.0]),
        "home_timeouts_remaining": pa.array([3.0, 3.0]),
        "away_timeouts_remaining": pa.array([3.0, 3.0]),
        "spread_line": pa.array([2.5, 2.5]),
        "home_wp": pa.array([0.45, 0.60]),
        "fixed_drive": pa.array([1.0, 2.0]),
        "fixed_drive_result": pa.array(["Punt", "Touchdown"]),
        "home_score": pa.array([24, 24]),
        "away_score": pa.array([20, 20]),
    }
    if not include_game_id:
        values.pop("game_id")
    sink = BytesIO()
    pq.write_table(pa.table(values), sink)
    return sink.getvalue()


def _client(payload: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == (
            "https://github.com/nflverse/nflverse-data/releases/download/"
            "pbp/play_by_play_2025.parquet"
        )
        assert request.headers["accept"] == "application/vnd.apache.parquet"
        return httpx.Response(
            200,
            content=payload,
            headers={
                "ETag": '"fixture-etag"',
                "Last-Modified": "Thu, 12 Feb 2026 09:58:23 GMT",
                "Content-Length": str(len(payload)),
                "Content-Type": "application/octet-stream",
            },
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_nflverse_partition_inspection_locks_native_schema_and_coverage() -> None:
    payload = _parquet_bytes()

    audit = inspect_nflverse_partition(payload, expected_year=2025)

    assert audit.row_count == 2
    assert audit.game_count == 1
    assert audit.season_types == ("REG",)
    assert audit.schema_fingerprint.startswith("sha256:")
    assert audit.object_sha256 == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_fetch_preserves_byte_exact_partition_with_governed_manifest(
    tmp_path: Path,
) -> None:
    payload = _parquet_bytes()
    with _client(payload) as client:
        result = fetch_and_preserve_nflverse_year(
            2025,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert result.audit.row_count == 2
    assert result.record.version == NFLVERSE_RELEASE_VERSION
    assert result.record.partition == "season-2025"
    assert result.record.manifest.dataset_id == "DS-NFLVERSE"
    assert result.record.manifest.license_ref == "I-018"
    assert result.record.manifest.license_status == "approved"
    assert result.record.manifest.object_sha256 == result.audit.object_sha256
    verified = read_verified_static_object(
        result.record.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert verified.object_bytes == payload


def test_fetch_retry_reuses_object_and_manifest_without_duplicates(
    tmp_path: Path,
) -> None:
    payload = _parquet_bytes()
    with _client(payload) as client:
        first = fetch_and_preserve_nflverse_year(
            2025,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )
        second = fetch_and_preserve_nflverse_year(
            2025,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert second == first
    assert len(list((tmp_path / "raw").rglob("*.parquet"))) == 1
    assert len(list((tmp_path / "manifests").rglob("*.manifest.json"))) == 1


def test_nflverse_rejects_unregistered_year_or_wrong_native_schema(
    tmp_path: Path,
) -> None:
    with pytest.raises(NFLVerseSourceError, match="2015-2025"):
        fetch_and_preserve_nflverse_year(
            2014,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=_client(_parquet_bytes()),
        )

    with _client(_parquet_bytes(include_game_id=False)) as client:
        with pytest.raises(NFLVerseSourceError, match="missing required columns.*game_id"):
            fetch_and_preserve_nflverse_year(
                2025,
                store_root=tmp_path,
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
            )
    assert list(tmp_path.rglob("*.parquet")) == []
