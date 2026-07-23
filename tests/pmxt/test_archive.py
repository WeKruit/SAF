from __future__ import annotations

import hashlib
import os
import stat
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pmxt"


def _official_table(receive_milliseconds: list[int]) -> pa.Table:
    rows = len(receive_milliseconds)
    source_milliseconds = [
        value - (10 + index * 5) for index, value in enumerate(receive_milliseconds)
    ]
    received = [
        datetime.fromtimestamp(value / 1_000, tz=timezone.utc)
        for value in receive_milliseconds
    ]
    sourced = [
        datetime.fromtimestamp(value / 1_000, tz=timezone.utc)
        for value in source_milliseconds
    ]
    event_types = ["book" if index == 0 else "price_change" for index in range(rows)]

    schema = pa.schema(
        [
            pa.field("timestamp_received", pa.timestamp("ms", tz="UTC")),
            pa.field("timestamp", pa.timestamp("ms", tz="UTC")),
            pa.field("market", pa.binary(66)),
            pa.field("event_type", pa.string()),
            pa.field("asset_id", pa.string()),
            pa.field("bids", pa.string()),
            pa.field("asks", pa.string()),
            pa.field("price", pa.decimal128(9, 4)),
            pa.field("size", pa.decimal128(18, 6)),
            pa.field("side", pa.string()),
            pa.field("best_bid", pa.decimal128(9, 4)),
            pa.field("best_ask", pa.decimal128(9, 4)),
            pa.field("fee_rate_bps", pa.uint16()),
            pa.field("transaction_hash", pa.string()),
            pa.field("old_tick_size", pa.decimal128(9, 4)),
            pa.field("new_tick_size", pa.decimal128(9, 4)),
        ]
    )
    columns = {
        "timestamp_received": received,
        "timestamp": sourced,
        "market": [
            b"0x" + bytes(str(index % 10), "ascii") * 64 for index in range(rows)
        ],
        "event_type": event_types,
        "asset_id": [f"asset-{index % 2}" for index in range(rows)],
        "bids": ["[]" if kind == "book" else None for kind in event_types],
        "asks": ["[]" if kind == "book" else None for kind in event_types],
        "price": [
            None if kind == "book" else Decimal("0.5000") for kind in event_types
        ],
        "size": [
            None if kind == "book" else Decimal("2.000000") for kind in event_types
        ],
        "side": [None if kind == "book" else "BUY" for kind in event_types],
        "best_bid": [None] * rows,
        "best_ask": [None] * rows,
        "fee_rate_bps": [None] * rows,
        "transaction_hash": [None] * rows,
        "old_tick_size": [None] * rows,
        "new_tick_size": [None] * rows,
    }
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def test_preserve_parquet_is_content_addressed_and_append_only(tmp_path):
    from prediction_market.pmxt.archive import ArchiveIntegrityError, preserve_parquet

    source = tmp_path / "source.parquet"
    source.write_bytes(b"original parquet bytes")
    raw_root = tmp_path / "raw"
    expected = f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"

    first = preserve_parquet(source, raw_root=raw_root)
    second = preserve_parquet(source, raw_root=raw_root)

    assert first.sha256 == expected
    assert first.object_path == second.object_path
    assert first.object_path.read_bytes() == source.read_bytes()
    assert first.object_path.name == f"{expected.removeprefix('sha256:')}.parquet"
    assert stat.S_IMODE(first.object_path.stat().st_mode) == 0o444

    first.object_path.chmod(0o644)
    first.object_path.write_bytes(b"tampered")
    with pytest.raises(ArchiveIntegrityError, match="content address"):
        preserve_parquet(source, raw_root=raw_root)
    assert first.object_path.read_bytes() == b"tampered"


@pytest.mark.parametrize("link_level", ["raw_root", "pmxt_v2"])
def test_preserve_parquet_rejects_symlink_at_every_raw_store_level(
    tmp_path, link_level
):
    from prediction_market.pmxt.archive import ArchiveIntegrityError, preserve_parquet

    source = tmp_path / "source.parquet"
    source.write_bytes(b"original parquet bytes")
    external = tmp_path / "external"
    external.mkdir()
    raw_root = tmp_path / "raw"
    if link_level == "raw_root":
        raw_root.symlink_to(external, target_is_directory=True)
    else:
        raw_root.mkdir()
        (raw_root / "pmxt-v2").symlink_to(external, target_is_directory=True)

    with pytest.raises(ArchiveIntegrityError, match="symbolic link"):
        preserve_parquet(source, raw_root=raw_root)

    assert list(external.iterdir()) == []


def test_preserve_parquet_rejects_parent_traversal_in_raw_root(tmp_path):
    from prediction_market.pmxt.archive import ArchiveIntegrityError, preserve_parquet

    source = tmp_path / "source.parquet"
    source.write_bytes(b"original parquet bytes")
    escaped_root = tmp_path / "raw" / ".." / "escaped"

    with pytest.raises(ArchiveIntegrityError, match="parent traversal"):
        preserve_parquet(source, raw_root=escaped_root)

    assert not (tmp_path / "escaped").exists()


def test_preserve_parquet_does_not_expose_final_object_before_file_fsync(
    tmp_path, monkeypatch
):
    from prediction_market.pmxt import archive

    source = tmp_path / "source.parquet"
    source.write_bytes(b"original parquet bytes")
    digest_hex = hashlib.sha256(source.read_bytes()).hexdigest()
    target = (
        tmp_path
        / "raw"
        / "pmxt-v2"
        / "sha256"
        / digest_hex[:2]
        / f"{digest_hex}.parquet"
    )
    original_fsync = archive.os.fsync
    final_visibility_at_file_fsync: list[bool] = []

    def observe_fsync(file_descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            final_visibility_at_file_fsync.append(target.exists())
        original_fsync(file_descriptor)

    monkeypatch.setattr(archive.os, "fsync", observe_fsync)

    archive.preserve_parquet(source, raw_root=tmp_path / "raw")

    assert final_visibility_at_file_fsync
    assert final_visibility_at_file_fsync == [False]


def test_archived_content_address_is_verified_before_every_read(tmp_path):
    from prediction_market.pmxt.archive import (
        ArchiveIntegrityError,
        audit_parquet,
        preserve_parquet,
        read_parquet_events,
    )

    source = tmp_path / "source.parquet"
    table = _official_table([1_780_272_000_100])
    pq.write_table(table, source)
    archived = preserve_parquet(source, raw_root=tmp_path / "raw")
    archived.object_path.chmod(0o644)
    pq.write_table(
        table.replace_schema_metadata({b"tampered": b"true"}),
        archived.object_path,
    )

    with pytest.raises(ArchiveIntegrityError, match="content address"):
        audit_parquet(archived.object_path)
    with pytest.raises(ArchiveIntegrityError, match="content address"):
        read_parquet_events([archived.object_path])


def test_audit_parquet_reports_schema_precision_and_timestamp_lag(tmp_path):
    from prediction_market.pmxt.archive import audit_parquet

    path = tmp_path / "sample.parquet"
    pq.write_table(_official_table([1_780_272_000_100, 1_780_272_000_200]), path)

    audit = audit_parquet(path)

    assert audit.original_sha256.startswith("sha256:")
    assert audit.row_count == 2
    assert audit.column_count == 16
    assert audit.schema["timestamp_received"] == "timestamp[ms, tz=UTC]"
    assert audit.schema["price"] == "decimal128(9, 4)"
    assert audit.schema["size"] == "decimal128(18, 6)"
    assert audit.event_type_counts == {"book": 1, "price_change": 1}
    assert audit.receive_minus_source_ms == {
        "count": 2,
        "negative_count": 0,
        "min": 10,
        "median": 12.5,
        "p95": 14.75,
        "max": 15,
    }


def test_read_parquet_events_merges_files_in_explicit_global_order(tmp_path):
    from prediction_market.pmxt.archive import read_parquet_events

    late_path = tmp_path / "late.parquet"
    early_path = tmp_path / "early.parquet"
    pq.write_table(_official_table([1_780_272_000_300]), late_path)
    pq.write_table(_official_table([1_780_272_000_100, 1_780_272_000_200]), early_path)

    events = read_parquet_events([late_path, early_path])

    assert [event["timestamp_received"].microsecond for event in events] == [
        100_000,
        200_000,
        300_000,
    ]
    assert all(not isinstance(event.get("price"), float) for event in events)


def test_read_parquet_events_pushes_down_one_market_scope(tmp_path):
    from prediction_market.pmxt.archive import read_parquet_events

    path = tmp_path / "two-markets.parquet"
    pq.write_table(_official_table([1_780_272_000_100, 1_780_272_000_200]), path)
    selected_market = "0x" + "1" * 64

    events = read_parquet_events([path], market=selected_market)

    assert len(events) == 1
    assert events[0]["market"] == selected_market.encode("utf-8")


def test_inventory_parser_deduplicates_page_metadata_and_quantifies_coverage():
    from prediction_market.pmxt.archive import parse_inventory_page, summarize_inventory

    entries = parse_inventory_page(
        (FIXTURE_ROOT / "archive_index.html").read_text(encoding="utf-8"),
        index_url="https://archive.pmxt.dev/Polymarket/v2",
    )
    summary = summarize_inventory(entries)

    assert [entry.filename for entry in entries] == [
        "polymarket_orderbook_2026-06-01T00.parquet",
        "polymarket_orderbook_2026-06-01T02.parquet",
    ]
    assert [entry.size_bytes for entry in entries] == [104_857_600, 314_572_800]
    assert summary["covered_hours"] == 2
    assert summary["expected_hours_inclusive"] == 3
    assert summary["missing_hours"] == ["2026-06-01T01:00:00.000Z"]
    assert summary["mean_file_bytes"] == 209_715_200
    assert summary["projected_mean_gb_per_day"] == 5.033165


def test_inventory_fetch_follows_advertised_total_page_count(monkeypatch):
    from prediction_market.pmxt import archive

    def page(hour: int, page_number: int) -> str:
        next_link = (
            f'<a href="/Polymarket/v2?page={page_number + 1}">Next</a>'
            if page_number < 3
            else ""
        )
        return (
            '<a href="https://r2v2.pmxt.dev/'
            f'polymarket_orderbook_2026-06-01T{hour:02d}.parquet">file</a>'
            "Mon, 01 Jun 2026 00:00 UTC    100.0 MB\n"
            f'<span class="page-info">Page <!-- -->{page_number}<!-- --> '
            f"of <!-- -->3</span>{next_link}"
        )

    pages = {1: page(0, 1), 2: page(1, 2), 3: page(2, 3)}
    calls: list[int] = []

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            page_number = int(url.rsplit("page=", 1)[1]) if "page=" in url else 1
            calls.append(page_number)
            return FakeResponse(pages[page_number])

    monkeypatch.setattr(archive.httpx, "Client", FakeClient)

    entries = archive.fetch_archive_inventory()

    assert calls == [1, 2, 3]
    assert [entry.hour.hour for entry in entries] == [0, 1, 2]
