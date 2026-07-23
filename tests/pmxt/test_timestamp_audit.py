from __future__ import annotations

import hashlib
from collections import Counter
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
    _canonical_disorder_counts,
    _exact_frequency_quantile,
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


def _write_hour(
    path: Path,
    hour: int,
    *,
    timestamp_rows: list[tuple[datetime, datetime]] | None = None,
) -> None:
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
    rows = timestamp_rows or [(received, source)]
    row_count = len(rows)
    values: dict[str, list[object]] = {
        "timestamp_received": [row[0] for row in rows],
        "timestamp": [row[1] for row in rows],
        "market": [MARKET.encode("ascii")] * row_count,
        "event_type": ["price_change"] * row_count,
        "asset_id": ["asset-a"] * row_count,
        "bids": [None] * row_count,
        "asks": [None] * row_count,
        "price": [Decimal("0.5000")] * row_count,
        "size": [Decimal("1.000000")] * row_count,
        "side": ["BUY"] * row_count,
    }
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(values[field.name], type=field.type) for field in schema],
            schema=schema,
        ),
        path,
    )


def _manifest(
    tmp_path: Path,
    *,
    timestamp_rows_by_hour: dict[int, list[tuple[datetime, datetime]]] | None = None,
):
    entries: list[ArchiveEntry] = []
    objects: list[HourlyObjectRef] = []
    for hour in range(24):
        filename = f"polymarket_orderbook_2026-05-28T{hour:02d}.parquet"
        path = tmp_path / filename
        _write_hour(
            path,
            hour,
            timestamp_rows=(timestamp_rows_by_hour or {}).get(hour),
        )
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
    assert report.absolute_p99_ms > 5_000.0
    assert report.quantile_method == "exact_frequency_quantile_cont_ms_v1"
    assert report.disorder_definition == (
        "per_market_asset_canonical_receive_source_adjacent_source_regression_v1"
    )
    assert report.millisecond_research_eligible is False
    assert report.downgrade_triggers == (
        "negative_delta_rate_ge_0.001",
        "absolute_p99_ms_gt_5000",
    )
    assert report.timestamp_semantics == (
        "receive_at_primary;source_at_secondary_audit_only"
    )


def test_timestamp_audit_fails_closed_when_native_file_order_is_broken(
    tmp_path: Path,
) -> None:
    later = datetime(2026, 5, 28, 0, 0, 20, tzinfo=timezone.utc)
    earlier = datetime(2026, 5, 28, 0, 0, 10, tzinfo=timezone.utc)
    manifest = _manifest(
        tmp_path,
        timestamp_rows_by_hour={
            0: [
                (later, later - timedelta(milliseconds=100)),
                (earlier, earlier - timedelta(milliseconds=100)),
            ]
        },
    )

    with pytest.raises(TimestampAuditError, match="native PMXT order"):
        audit_full_day_timestamps(tmp_path, manifest)


def test_timestamp_audit_applies_canonical_source_tie_break_without_sorting_rows(
    tmp_path: Path,
) -> None:
    receive_1 = datetime(2026, 5, 28, 0, 0, 10, tzinfo=timezone.utc)
    receive_2 = datetime(2026, 5, 28, 0, 0, 20, tzinfo=timezone.utc)
    manifest = _manifest(
        tmp_path,
        timestamp_rows_by_hour={
            0: [
                (receive_1, receive_1 + timedelta(milliseconds=100)),
                (receive_1, receive_1 - timedelta(milliseconds=100)),
                (receive_2, receive_1 - timedelta(milliseconds=50)),
                (receive_2, receive_1 - timedelta(milliseconds=20)),
            ]
        },
    )

    report = audit_full_day_timestamps(tmp_path, manifest)

    # Canonical source ordering turns each receive-time tie into an ascending
    # run.  Only the boundary from max(run 1) to min(run 2) is a regression.
    assert report.ordered_comparison_count == 26
    assert report.out_of_order_count == 2


def test_streaming_disorder_carries_tied_receive_run_across_batches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-batch.parquet"
    receive_1 = datetime(2026, 5, 28, 0, 0, 10, tzinfo=timezone.utc)
    receive_2 = datetime(2026, 5, 28, 0, 0, 20, tzinfo=timezone.utc)
    _write_hour(
        path,
        0,
        timestamp_rows=[
            (receive_1, receive_1 + timedelta(milliseconds=100)),
            (receive_1, receive_1 + timedelta(milliseconds=200)),
            (receive_1, receive_1 + timedelta(milliseconds=50)),
            (receive_2, receive_1 + timedelta(milliseconds=150)),
        ],
    )

    rows, comparisons, regressions = _canonical_disorder_counts(
        [str(path)], batch_size=2
    )

    assert rows == 4
    assert comparisons == 3
    assert regressions == 1


def test_exact_frequency_quantile_matches_continuous_rank_interpolation() -> None:
    counts = Counter({0: 1, 10: 3})

    assert _exact_frequency_quantile(counts, 0.0) == 0.0
    assert _exact_frequency_quantile(counts, 0.25) == pytest.approx(7.5)
    assert _exact_frequency_quantile(counts, 0.5) == 10.0
    assert _exact_frequency_quantile(counts, 1.0) == 10.0
