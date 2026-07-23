from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.pmxt.archive import ArchiveEntry
from prediction_market.pmxt.full_day import (
    FullDayInputError,
    HourlyObjectRef,
    build_full_day_manifest,
    preflight_full_day_inputs,
    run_full_day_reconstruction,
    select_complete_utc_day,
    validate_full_day_manifest,
)


DAY = date(2026, 5, 28)
MARKET = "0x" + "3" * 64
ASSET = "asset-home"


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _schema() -> pa.Schema:
    return pa.schema(
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
        ]
    )


def _write_hour(path: Path, hour: int, *, market: str | None = MARKET) -> None:
    observed = datetime(2026, 5, 28, hour, 0, 0, 100_000, tzinfo=timezone.utc)
    snapshot = hour == 0
    values: dict[str, list[object]] = {
        "timestamp_received": [observed],
        "timestamp": [observed - timedelta(milliseconds=20)],
        "market": [market.encode("ascii") if market is not None else None],
        "event_type": ["book" if snapshot else "price_change"],
        "asset_id": [ASSET],
        "bids": ['[["0.45", "100"]]' if snapshot else None],
        "asks": ['[["0.55", "100"]]' if snapshot else None],
        "price": [None if snapshot else Decimal("0.4500")],
        "size": [None if snapshot else Decimal(str(100 + hour))],
        "side": [None if snapshot else "BUY"],
    }
    schema = _schema()
    table = pa.Table.from_arrays(
        [pa.array(values[field.name], type=field.type) for field in schema],
        schema=schema,
    )
    pq.write_table(table, path)


def _day_inputs(
    tmp_path: Path, *, market: str | None = MARKET
) -> tuple[list[ArchiveEntry], list[HourlyObjectRef]]:
    entries: list[ArchiveEntry] = []
    objects: list[HourlyObjectRef] = []
    for hour in range(24):
        filename = f"polymarket_orderbook_2026-05-28T{hour:02d}.parquet"
        path = tmp_path / filename
        _write_hour(path, hour, market=market)
        payload = path.read_bytes()
        observed_at = datetime(2026, 5, 28, hour, tzinfo=timezone.utc)
        url = f"https://r2v2.pmxt.dev/{filename}"
        entries.append(
            ArchiveEntry(
                filename=filename,
                url=url,
                hour=observed_at,
                size_bytes=len(payload),
            )
        )
        objects.append(
            HourlyObjectRef(
                hour=observed_at,
                source_url=url,
                object_path=filename,
                object_sha256=_digest(payload),
                static_manifest_sha256=_digest(f"manifest-{hour}".encode()),
            )
        )
    return entries, objects


def _manifest(tmp_path: Path):
    entries, objects = _day_inputs(tmp_path)
    selected = select_complete_utc_day(entries, day=DAY)
    return build_full_day_manifest(
        day=DAY,
        entries=selected,
        objects=reversed(objects),
        inventory_sha256=_digest(b"frozen-pmxt-inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )


def test_complete_day_requires_exactly_one_object_for_each_utc_hour(
    tmp_path: Path,
) -> None:
    entries, _ = _day_inputs(tmp_path)

    selected = select_complete_utc_day(reversed(entries), day=DAY)

    assert len(selected) == 24
    assert [entry.hour.hour for entry in selected] == list(range(24))

    with pytest.raises(FullDayInputError, match="missing.*23"):
        select_complete_utc_day(entries[:-1], day=DAY)

    duplicate = replace(
        entries[0],
        filename="polymarket_orderbook_2026-05-28T00-duplicate.parquet",
        url="https://r2v2.pmxt.dev/duplicate.parquet",
    )
    with pytest.raises(FullDayInputError, match="duplicate.*00"):
        select_complete_utc_day([*entries, duplicate], day=DAY)


def test_day_manifest_locks_ordered_hourly_objects_and_self_hash(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    assert manifest.day == "2026-05-28"
    assert [item.hour for item in manifest.objects] == [
        f"2026-05-28T{hour:02d}:00:00Z" for hour in range(24)
    ]
    assert manifest.manifest_sha256.startswith("sha256:")
    assert validate_full_day_manifest(manifest) == manifest

    with pytest.raises(FullDayInputError, match="manifest_sha256"):
        validate_full_day_manifest(
            replace(manifest, manifest_sha256="sha256:" + "0" * 64)
        )


def test_cross_hour_book_is_reconstructed_once_for_every_market(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    report = run_full_day_reconstruction(tmp_path, manifest)

    assert report.market_count == 1
    assert report.input_event_count == 24
    assert report.semantic_event_count == 24
    assert report.market_results[0].native_market_id == MARKET
    assert report.market_results[0].counts["missing_initial_snapshots"] == 0
    assert report.market_results[0].stream_sha256.startswith("sha256:")
    assert report.day_stream_sha256.startswith("sha256:")
    assert report.independent_comparison_performed is False
    assert report.x01_formal_gate_passed is False
    assert report.open_gate == "independent_price_and_size_comparison"


def test_full_day_preflight_verifies_partition_schema_and_counts(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    first = preflight_full_day_inputs(tmp_path, manifest)
    second = preflight_full_day_inputs(tmp_path, manifest)

    assert first == second
    assert first.object_count == 24
    assert first.row_count == 24
    assert first.market_count == 1
    assert first.asset_count == 1
    assert first.event_type_counts == {"book": 1, "price_change": 23}
    assert first.timestamp_received_min == "2026-05-28T00:00:00.100Z"
    assert first.timestamp_received_max == "2026-05-28T23:00:00.100Z"
    assert first.schema_fingerprint.startswith("sha256:")
    assert first.report_sha256.startswith("sha256:")
    assert first.reconstruction_executed is False
    assert first.x01_formal_gate_passed is False
    assert first.open_gates == (
        "all_contract_full_day_semantic_reconstruction",
        "independent_price_and_size_comparison",
    )


def test_full_day_preflight_rejects_object_outside_locked_receive_hour(
    tmp_path: Path,
) -> None:
    entries, objects = _day_inputs(tmp_path)
    wrong_hour_path = tmp_path / objects[1].object_path
    _write_hour(wrong_hour_path, 0)
    payload = wrong_hour_path.read_bytes()
    objects[1] = replace(objects[1], object_sha256=_digest(payload))
    manifest = build_full_day_manifest(
        day=DAY,
        entries=select_complete_utc_day(entries, day=DAY),
        objects=objects,
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )

    with pytest.raises(FullDayInputError, match="receive-time partition"):
        preflight_full_day_inputs(tmp_path, manifest)


def test_full_day_result_is_independent_of_manifest_input_order(
    tmp_path: Path,
) -> None:
    entries, objects = _day_inputs(tmp_path)
    first_manifest = build_full_day_manifest(
        day=DAY,
        entries=select_complete_utc_day(entries, day=DAY),
        objects=objects,
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )
    second_manifest = build_full_day_manifest(
        day=DAY,
        entries=select_complete_utc_day(reversed(entries), day=DAY),
        objects=reversed(objects),
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )

    first = run_full_day_reconstruction(tmp_path, first_manifest)
    second = run_full_day_reconstruction(tmp_path, second_manifest)

    assert first == second
    assert first_manifest == second_manifest


def test_full_day_fails_closed_when_locked_object_is_tampered(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    first_path = tmp_path / manifest.objects[0].object_path
    first_path.write_bytes(first_path.read_bytes() + b"tampered")

    with pytest.raises(FullDayInputError, match="SHA-256"):
        run_full_day_reconstruction(tmp_path, manifest)


def test_native_market_case_is_preserved_for_exact_parquet_filtering(
    tmp_path: Path,
) -> None:
    uppercase_market = "0x" + "A" * 64
    entries, objects = _day_inputs(tmp_path, market=uppercase_market)
    manifest = build_full_day_manifest(
        day=DAY,
        entries=select_complete_utc_day(entries, day=DAY),
        objects=objects,
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )

    report = run_full_day_reconstruction(tmp_path, manifest)

    assert report.market_count == 1
    assert report.input_event_count == 24
    assert report.market_results[0].native_market_id == uppercase_market


@pytest.mark.parametrize("alias_prefix", ["./", "nested//"])
def test_manifest_rejects_noncanonical_path_aliases(
    tmp_path: Path, alias_prefix: str
) -> None:
    entries, objects = _day_inputs(tmp_path)
    objects[1] = replace(
        objects[1], object_path=alias_prefix + objects[1].object_path
    )

    with pytest.raises(FullDayInputError, match="object_path.*canonical"):
        build_full_day_manifest(
            day=DAY,
            entries=select_complete_utc_day(entries, day=DAY),
            objects=objects,
            inventory_sha256=_digest(b"inventory"),
            canonicalization_version="pmxt-reconstructor-v1",
        )


def test_full_day_rejects_null_native_market_rows_instead_of_dropping_them(
    tmp_path: Path,
) -> None:
    entries, objects = _day_inputs(tmp_path, market=None)
    manifest = build_full_day_manifest(
        day=DAY,
        entries=select_complete_utc_day(entries, day=DAY),
        objects=objects,
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )

    with pytest.raises(FullDayInputError, match="NULL market"):
        run_full_day_reconstruction(tmp_path, manifest)
