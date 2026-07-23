from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import prediction_market.experiments as experiments_module
import prediction_market.pmxt.timestamp_audit as timestamp_audit_module
from prediction_market.pmxt.archive import ArchiveEntry
from prediction_market.pmxt.full_day import (
    FullDayManifest,
    HourlyObjectRef,
    build_full_day_manifest,
    select_complete_utc_day,
)
from prediction_market.pmxt.timestamp_audit import (
    TimestampAuditError,
    _canonical_disorder_counts,
    _exact_frequency_quantile,
    audit_full_day_timestamps,
    audit_timestamp_sample,
    select_timestamp_audit_days,
)


X01_DAY = date(2026, 5, 28)
MARKET = "0x" + "7" * 64


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
    day: date = X01_DAY,
    timestamp_rows: list[tuple[datetime, datetime]] | None = None,
) -> None:
    received = datetime(
        day.year, day.month, day.day, hour, 0, 10, tzinfo=timezone.utc
    )
    if hour == 5:
        source = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
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
    day: date = X01_DAY,
    timestamp_rows_by_hour: dict[int, list[tuple[datetime, datetime]]] | None = None,
):
    entries: list[ArchiveEntry] = []
    objects: list[HourlyObjectRef] = []
    for hour in range(24):
        filename = f"polymarket_orderbook_{day.isoformat()}T{hour:02d}.parquet"
        path = tmp_path / filename
        _write_hour(
            path,
            hour,
            day=day,
            timestamp_rows=(timestamp_rows_by_hour or {}).get(hour),
        )
        payload = path.read_bytes()
        observed = datetime(
            day.year, day.month, day.day, hour, tzinfo=timezone.utc
        )
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
        day=day,
        entries=select_complete_utc_day(entries, day=day),
        objects=objects,
        inventory_sha256=_digest(b"inventory"),
        canonicalization_version="pmxt-reconstructor-v1",
    )


def _write_timestamp_bundle(
    program_root: Path,
    manifests: tuple[FullDayManifest, ...],
) -> tuple[str, str, str]:
    audit_root = program_root / "artifacts" / "data-audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for manifest in manifests:
        relative = (
            "artifacts/data-audit/"
            f"x02_full_day_input_manifest_{manifest.day}_v1.json"
        )
        payload = (
            json.dumps(
                asdict(manifest),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        (program_root / relative).write_bytes(payload)
        entries.append(
            {
                "artifact_file_sha256": _digest(payload),
                "day": manifest.day,
                "full_day_manifest_sha256": manifest.manifest_sha256,
                "object_count": len(manifest.objects),
                "path": relative,
            }
        )
    days = [manifest.day for manifest in manifests]
    material = {
        "additional_days": [day for day in days if day != "2026-05-28"],
        "day_count": len(manifests),
        "day_manifests": entries,
        "formal_result": False,
        "inventory_path": "artifacts/data-audit/phase0_inventory.json",
        "inventory_sha256": _digest(b"inventory"),
        "object_count": sum(len(manifest.objects) for manifest in manifests),
        "purpose": "frozen_input_only_before_X02_evaluation",
        "selection_procedure": "fixture_exact_four_day_selection",
        "selection_seed": 20260722,
        "version": "x02-timestamp-input-bundle-v1",
        "x01_day": "2026-05-28",
    }
    bundle_sha256 = _digest(_canonical(material))
    bundle = {**material, "bundle_sha256": bundle_sha256}
    relative = "artifacts/data-audit/x02_timestamp_input_bundle_v1.json"
    bundle_payload = (
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (program_root / relative).write_bytes(bundle_payload)
    return relative, bundle_sha256, _digest(bundle_payload)


@pytest.fixture
def governed_x02(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_registry(program_root: str | Path) -> dict[str, dict[str, object]]:
        root = Path(program_root)
        relative = "artifacts/data-audit/x02_timestamp_input_bundle_v1.json"
        payload = (root / relative).read_bytes()
        document = json.loads(payload)
        bundle_sha256 = document["bundle_sha256"]
        return {
            "X-02": {
                "timestamp_input_manifest_binding": {
                    "bundle_path": relative,
                    "bundle_file_sha256": _digest(payload),
                    "bundle_sha256": bundle_sha256,
                },
                "preregistered_inputs": {
                    "formal_result": {
                        "code_sha256": _digest(
                            Path(timestamp_audit_module.__file__).read_bytes()
                        ),
                        "data_sha256": bundle_sha256,
                        "dataset_ids": ["DS-PMXT-V2"],
                        "model_ids": [],
                        "registered_at": "2026-07-23T05:30:00Z",
                    }
                },
            }
        }

    monkeypatch.setattr(experiments_module, "load_experiment_registry", load_registry)


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


def test_four_day_timestamp_sample_aggregates_the_preregistered_unit(
    tmp_path: Path,
    governed_x02: None,
) -> None:
    days = (
        date(2026, 4, 22),
        date(2026, 5, 28),
        date(2026, 6, 5),
        date(2026, 6, 25),
    )
    manifests = tuple(_manifest(tmp_path, day=day) for day in days)
    bundle_path, bundle_sha256, bundle_file_sha256 = _write_timestamp_bundle(
        tmp_path, manifests
    )

    report = audit_timestamp_sample(
        tmp_path,
        program_root=tmp_path,
        input_bundle_path=bundle_path,
    )

    assert report.version == "pmxt-timestamp-sample-audit-v1"
    assert report.days == tuple(day.isoformat() for day in days)
    assert report.input_bundle_path == bundle_path
    assert report.input_bundle_file_sha256 == bundle_file_sha256
    assert report.input_bundle_sha256 == bundle_sha256
    assert report.input_manifest_sha256s == tuple(
        manifest.manifest_sha256 for manifest in manifests
    )
    assert report.day_count == 4
    assert report.object_count == 96
    assert report.row_count == 96
    assert report.delta_count == 96
    assert report.quantiles_ms["p50"] == pytest.approx(100.0)
    assert report.negative_delta_count == 4
    assert report.negative_delta_rate == pytest.approx(4 / 96)
    assert report.ordered_comparison_count == 95
    assert report.out_of_order_count == 4
    assert report.hourly_median_drift_ms == pytest.approx(0.0)
    assert report.millisecond_research_eligible is False
    assert report.report_sha256.startswith("sha256:")


def test_timestamp_sample_rejects_noncanonical_four_day_membership(
    tmp_path: Path,
    governed_x02: None,
) -> None:
    days = (
        date(2026, 4, 22),
        date(2026, 5, 28),
        date(2026, 6, 5),
        date(2026, 6, 25),
    )
    manifests = tuple(_manifest(tmp_path, day=day) for day in days)

    three_day_bundle, _, _ = _write_timestamp_bundle(tmp_path, manifests[:3])
    with pytest.raises(TimestampAuditError, match="exactly four"):
        audit_timestamp_sample(
            tmp_path,
            program_root=tmp_path,
            input_bundle_path=three_day_bundle,
        )

    reversed_root = tmp_path / "reversed"
    reversed_root.mkdir()
    reversed_manifests = tuple(
        _manifest(reversed_root, day=day) for day in reversed(days)
    )
    reversed_bundle, _, _ = _write_timestamp_bundle(
        reversed_root, reversed_manifests
    )
    with pytest.raises(TimestampAuditError, match="strictly increasing"):
        audit_timestamp_sample(
            reversed_root,
            program_root=reversed_root,
            input_bundle_path=reversed_bundle,
        )


def test_timestamp_sample_rejects_tampered_bound_day_manifest(
    tmp_path: Path,
    governed_x02: None,
) -> None:
    days = (
        date(2026, 4, 22),
        date(2026, 5, 28),
        date(2026, 6, 5),
        date(2026, 6, 25),
    )
    manifests = tuple(_manifest(tmp_path, day=day) for day in days)
    bundle_path, _, _ = _write_timestamp_bundle(tmp_path, manifests)
    bound_path = (
        tmp_path
        / "artifacts/data-audit/x02_full_day_input_manifest_2026-04-22_v1.json"
    )
    bound_path.write_bytes(bound_path.read_bytes() + b" ")

    with pytest.raises(TimestampAuditError, match="artifact file SHA-256"):
        audit_timestamp_sample(
            tmp_path,
            program_root=tmp_path,
            input_bundle_path=bundle_path,
        )


def test_timestamp_sample_rejects_missing_governance_before_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = (
        date(2026, 4, 22),
        date(2026, 5, 28),
        date(2026, 6, 5),
        date(2026, 6, 25),
    )
    manifests = tuple(_manifest(tmp_path, day=day) for day in days)
    bundle_path, _, _ = _write_timestamp_bundle(tmp_path, manifests)
    metric_calls = 0

    def reject_registry(program_root: str | Path) -> dict[str, dict[str, object]]:
        raise experiments_module.ExperimentRegistryError("missing sidecar")

    def forbidden_metrics(paths: list[str]):
        nonlocal metric_calls
        metric_calls += 1
        raise AssertionError("metrics must not run before governance")

    monkeypatch.setattr(
        experiments_module, "load_experiment_registry", reject_registry
    )
    monkeypatch.setattr(
        timestamp_audit_module, "_compute_timestamp_metrics", forbidden_metrics
    )

    with pytest.raises(TimestampAuditError, match="governance"):
        audit_timestamp_sample(
            tmp_path,
            program_root=tmp_path,
            input_bundle_path=bundle_path,
        )
    assert metric_calls == 0


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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_streaming_disorder_carries_tied_receive_run_across_batches(
    tmp_path: Path, batch_size: int
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
        [str(path)], batch_size=batch_size
    )

    assert rows == 4
    assert comparisons == 3
    assert regressions == 1


def test_streaming_disorder_defers_boundary_until_tied_target_run_is_complete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-batch-target-min.parquet"
    receive_0 = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)
    receive_1 = datetime(2026, 5, 28, 0, 0, 10, tzinfo=timezone.utc)
    receive_2 = datetime(2026, 5, 28, 0, 0, 20, tzinfo=timezone.utc)
    _write_hour(
        path,
        0,
        timestamp_rows=[
            (receive_0, receive_0 + timedelta(milliseconds=100)),
            (receive_1, receive_0 + timedelta(milliseconds=200)),
            (receive_1, receive_0 + timedelta(milliseconds=50)),
            (receive_2, receive_0 + timedelta(milliseconds=300)),
        ],
    )

    rows, comparisons, regressions = _canonical_disorder_counts(
        [str(path)], batch_size=2
    )

    assert rows == 4
    assert comparisons == 3
    # Canonical source ordering makes the target receive-time run [50, 200].
    # Its minimum is only observed in the next batch, so the 100 -> 50
    # regression must be evaluated after the whole tied run is complete.
    assert regressions == 1


def test_exact_frequency_quantile_matches_continuous_rank_interpolation() -> None:
    counts = Counter({0: 1, 10: 3})

    assert _exact_frequency_quantile(counts, 0.0) == 0.0
    assert _exact_frequency_quantile(counts, 0.25) == pytest.approx(7.5)
    assert _exact_frequency_quantile(counts, 0.5) == 10.0
    assert _exact_frequency_quantile(counts, 1.0) == 10.0
