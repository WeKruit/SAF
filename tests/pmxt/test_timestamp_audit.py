from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.pmxt.archive import ArchiveEntry
from prediction_market.pmxt.full_day import (
    HourlyObjectRef,
    build_full_day_manifest,
    select_complete_utc_day,
)
from prediction_market.pmxt.timestamp_audit import (
    TimestampAuditError,
    audit_full_day_timestamps,
    select_timestamp_audit_days,
)


X01_DAY = date(2026, 5, 28)
MARKET = "0x" + "7" * 64


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _inventory(days: list[date], *, incomplete: date | None = None):
    entries: list[ArchiveEntry] = []
    for day in days:
        limit = 23 if day == incomplete else 24
        for hour in range(limit):
            observed = datetime(
                day.year, day.month, day.day, hour, tzinfo=timezone.utc
            )
            filename = f"polymarket_orderbook_{day.isoformat()}T{hour:02d}.parquet"
            entries.append(
                ArchiveEntry(
                    filename=filename,
                    url=f"https://r2v2.pmxt.dev/{filename}",
                    hour=observed,
                    size_bytes=1,
                )
            )
    return entries


def _write_hour(path: Path, hour: int) -> None:
    received = datetime(2026, 5, 28, hour, 0, 10, tzinfo=timezone.utc)
    if hour == 5:
        source = datetime(2026, 5, 28, 0, 0, tzinfo=timezone.utc)
    elif hour == 10:
        source = received + timedelta(milliseconds=50)
    else:
        source = received - timedelta(milliseconds=100)
    schema = pa.schema(
        [
            pa.field("timestamp_received", pa.timestamp("ms", tz="UTC"), False),
            pa.field("timestamp", pa.timestamp("ms", tz="UTC"), False),
            pa.field("market", pa.binary(66), False),
            pa.field("event_type", pa.string(), False),
            pa.field("asset_id", pa.string(), False),
            pa.field("bids", pa.string()),
            pa.field("asks", pa.string()),
            pa.field("price", pa.decimal128(9, 4)),
            pa.field("size", pa.decimal128(18, 6)),
            pa.field("side", pa.string()),
        ]
    )
    values: dict[str, list[object]] = {
        "timestamp_received": [received],
        "timestamp": [source],
        "market": [MARKET.encode("ascii")],
        "event_type": ["price_change"],
        "asset_id": ["asset-a"],
        "bids": [None],
        "asks": [None],
        "price": [Decimal("0.5000")],
        "size": [Decimal("1.000000")],
        "side": ["BUY"],
    }
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(values[field.name], type=field.type) for field in schema],
            schema=schema,
        ),
        path,
    )


def _manifest(tmp_path: Path):
    entries: list[ArchiveEntry] = []
    objects: list[HourlyObjectRef] = []
    for hour in range(24):
        filename = f"polymarket_orderbook_2026-05-28T{hour:02d}.parquet"
        path = tmp_path / filename
        _write_hour(path, hour)
        payload = path.read_bytes()
        observed = datetime(2026, 5, 28, hour, tzinfo=timezone.utc)
        url = f"https://r2v2.pmxt.dev/{filename}"
        entries.append(ArchiveEntry(filename, url, observed, len(payload)))
        objects.append(
            HourlyObjectRef(
                hour=observed,
                source_url=url,
                object_path=filename,
                object_sha256=_digest(payload),
                static_manifest_sha256=_digest(f"manifest-{hour}".encode()),
            )
        )
    return build_full_day_manifest(
        day=X01_DAY,
        entries=select_complete_utc_day(entries, day=X01_DAY),
        objects=objects,
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )


def test_x02_day_selection_is_seeded_and_complete() -> None:
    days = [date(2026, 5, day) for day in range(24, 30)]
    incomplete = days[0]
    entries = _inventory(days, incomplete=incomplete)

    first = select_timestamp_audit_days(
        entries,
        x01_day=X01_DAY,
        additional_days=3,
        seed=20260722,
    )
    second = select_timestamp_audit_days(
        reversed(entries),
        x01_day=X01_DAY,
        additional_days=3,
        seed=20260722,
    )

    assert first == second
    assert first[0] == X01_DAY
    assert len(first) == 4
    assert len(set(first)) == 4
    assert incomplete not in first


def test_x02_selection_rejects_incomplete_x01_day() -> None:
    entries = _inventory([X01_DAY], incomplete=X01_DAY)

    with pytest.raises(TimestampAuditError, match="X-01 day.*complete"):
        select_timestamp_audit_days(
            entries,
            x01_day=X01_DAY,
            additional_days=0,
            seed=20260722,
        )


def test_full_day_timestamp_audit_reports_locked_metrics(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    report = audit_full_day_timestamps(tmp_path, manifest)

    assert report.day == "2026-05-28"
    assert report.input_manifest_sha256 == manifest.manifest_sha256
    assert report.row_count == 24
    assert report.delta_count == 24
    assert report.quantiles_ms["p50"] == pytest.approx(100.0)
    assert report.quantiles_ms["p95"] >= 100.0
    assert report.quantiles_ms["p99"] > report.quantiles_ms["p95"]
    assert report.negative_delta_count == 1
    assert report.negative_delta_rate == pytest.approx(1 / 24)
    assert report.ordered_comparison_count == 23
    assert report.out_of_order_count == 1
    assert report.out_of_order_rate == pytest.approx(1 / 23)
    assert report.hourly_median_drift_ms == pytest.approx(0.0)
    assert report.timestamp_semantics == (
        "receive_at_primary;source_at_secondary_audit_only"
    )
