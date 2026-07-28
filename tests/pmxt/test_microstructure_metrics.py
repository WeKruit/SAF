from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

import pytest

from prediction_market.pmxt.microstructure_metrics import (
    L2MethodError,
    MetricBasis,
    PHASE1_SAMPLE_MEASUREMENT_SHA256,
    build_l2_metric_ticks,
    build_method_audit_payload,
    build_tick_data_inventory_v1_rows,
    cancel_intensity,
    method_audit_fixture_events,
    parity_deviation,
    resilience_fraction,
    signed_markout,
    walk_book,
)


def _event(event_type: str, receive_at: str, **fields: object) -> dict[str, object]:
    return {
        "timestamp_received": receive_at,
        "timestamp": receive_at,
        "market": "market-1",
        "asset_id": "yes-1",
        "event_type": event_type,
        **fields,
    }


def _l2_events() -> list[dict[str, object]]:
    return [dict(event) for event in method_audit_fixture_events()]


def test_l2_ticks_compute_bbo_depth_microprice_and_no_implied_aggressor() -> None:
    ticks = build_l2_metric_ticks(_l2_events(), top_n=2)

    first, second, third = ticks[:3]
    assert first.best_bid == Decimal("0.48")
    assert first.best_ask == Decimal("0.52")
    assert first.midpoint == Decimal("0.50")
    assert first.spread == Decimal("0.04")
    assert first.bid_depth_top_n == Decimal("120")
    assert first.ask_depth_top_n == Decimal("80")
    assert first.imbalance == Decimal("0.2")
    assert first.microprice == Decimal("0.5066666666666666666666666667")

    assert second.ofi == Decimal("-40")
    assert second.bid_depth_withdrawal == Decimal("40")
    assert second.ask_depth_withdrawal == Decimal("0")
    assert third.ask_depth_withdrawal == Decimal("50")
    assert second.cancel_intensity_by_basis == {
        "bid": None,
        "ask": None,
    }
    assert second.markout_by_basis == {
        "bid": None,
        "ask": None,
        "midpoint": None,
    }
    assert "aggressor_side" not in second.to_dict()
    assert second.to_dict()["formal_aggressor_status"] == "unavailable_no_authoritative_fill"


def test_delta_requires_snapshot_and_gap_or_reconnect_require_new_epoch_snapshot() -> None:
    delta = _event(
        "price_change",
        "2026-07-01T00:00:01.000Z",
        price="0.48",
        size="60",
        side="BUY",
    )
    with pytest.raises(L2MethodError, match="snapshot"):
        build_l2_metric_ticks([delta])

    events = [
        _l2_events()[0],
        _event("gap", "2026-07-01T00:00:01.000Z"),
        _event(
            "book",
            "2026-07-01T00:00:02.000Z",
            bids=[["0.49", "10"]],
            asks=[["0.51", "10"]],
        ),
    ]
    ticks = build_l2_metric_ticks(events)
    assert [tick.epoch for tick in ticks] == [1, 2]

    post_gap_delta = _event(
        "price_change",
        "2026-07-01T00:00:01.001Z",
        price="0.48",
        size="60",
        side="BUY",
    )
    with pytest.raises(L2MethodError, match="snapshot"):
        build_l2_metric_ticks(
            [_l2_events()[0], _event("gap", "2026-07-01T00:00:01.000Z"), post_gap_delta]
        )

    reconnect_events = [
        _l2_events()[0],
        _event("reconnect", "2026-07-01T00:00:01.000Z"),
        _event(
            "book",
            "2026-07-01T00:00:02.000Z",
            bids=[["0.49", "10"]],
            asks=[["0.51", "10"]],
        ),
    ]
    reconnect_ticks = build_l2_metric_ticks(reconnect_events)
    assert [tick.epoch for tick in reconnect_ticks] == [1, 2]
    with pytest.raises(L2MethodError, match="snapshot"):
        build_l2_metric_ticks(
            [
                _l2_events()[0],
                _event("reconnect", "2026-07-01T00:00:01.000Z"),
                post_gap_delta,
            ]
        )


def test_l2_tick_rebuild_is_canonical_and_deterministic() -> None:
    expected = [tick.to_dict() for tick in build_l2_metric_ticks(_l2_events(), top_n=2)]
    rebuilt = [tick.to_dict() for tick in build_l2_metric_ticks(list(reversed(_l2_events())), top_n=2)]

    assert rebuilt == expected


def test_pure_execution_adjacent_calculations_require_explicit_inputs() -> None:
    walked = walk_book(
        side="BUY",
        levels=[("0.52", "2"), ("0.53", "3")],
        requested_size="4",
    )
    assert walked.average_price == Decimal("0.525")
    assert walked.filled_size == Decimal("4")
    assert walked.unfilled_size == Decimal("0")
    cancel = cancel_intensity(
        basis=MetricBasis.BID,
        cancelled_size="4",
        observed_depth="20",
    )
    assert cancel.value == Decimal("0.2")
    assert cancel.to_dict() == {
        "method": "cancel_intensity",
        "basis": "bid",
        "inputs": {"cancelled_size": "4", "observed_depth": "20"},
        "value": "0.2",
    }
    resilience = resilience_fraction(
        basis=MetricBasis.MIDPOINT,
        baseline_reference="0.50",
        shocked_reference="0.45",
        recovered_reference="0.49",
    )
    assert resilience.value == Decimal("0.8")
    buy_markout = signed_markout(
        side="BUY",
        basis=MetricBasis.MIDPOINT,
        execution_price="0.48",
        future_reference="0.50",
    )
    sell_markout = signed_markout(
        side="SELL",
        basis=MetricBasis.MIDPOINT,
        execution_price="0.52",
        future_reference="0.50",
    )
    assert buy_markout.value == Decimal("0.02")
    assert sell_markout.value == Decimal("0.02")
    assert parity_deviation("0.51", "0.47") == Decimal("-0.02")

    with pytest.raises(TypeError):
        cancel_intensity(cancelled_size="4", observed_depth="20")


def test_tick_inventory_rows_and_audit_are_metadata_only_and_deterministic() -> None:
    remote_inventory = {
        "audit_kind": "pmxt_s3_verify_v1",
        "objects": [
            {
                "sha256": "sha256:bbb",
                "byte_size": 200,
                "status": "verified",
            },
            {
                "sha256": "sha256:aaa",
                "byte_size": 100,
                "status": "verified",
            },
        ],
    }
    phase0_inventory = {
        "audit_kind": "pmxt_phase0_inventory_v1",
        "objects": [
            {
                "filename": "polymarket_orderbook_2026-07-01T01.parquet",
                "hour": "2026-07-01T01:00:00.000Z",
                "size_bytes": 200,
            },
            {
                "filename": "polymarket_orderbook_2026-07-01T00.parquet",
                "hour": "2026-07-01T00:00:00.000Z",
                "size_bytes": 101,
            },
        ],
    }

    manifests = [
        {
            "object_sha256": "sha256:aaa",
            "byte_length": 100,
            "source_cursor": "filename:polymarket_orderbook_2026-07-01T00.parquet",
            "upstream_partition": "hour-2026-07-01T00",
        }
    ]
    rows = build_tick_data_inventory_v1_rows(
        remote_inventory, phase0_inventory, authoritative_manifests=manifests
    )
    assert [row.row_kind for row in rows] == [
        "REMOTE_VERIFIED_OBJECT",
        "REMOTE_VERIFIED_OBJECT",
        "PHASE0_DISCOVERY_ENTRY",
    ]
    assert [row.object_sha256 for row in rows] == ["sha256:aaa", "sha256:bbb", None]
    assert rows[0].to_dict() == {
        "schema_version": "TickDataInventoryV1",
        "source": "pmxt-v2",
        "row_kind": "REMOTE_VERIFIED_OBJECT",
        "link_status": "ATTESTED_MANIFEST_CROSSWALK",
        "object_sha256": "sha256:aaa",
        "filename": "polymarket_orderbook_2026-07-01T00.parquet",
        "hour": "2026-07-01T00:00:00.000Z",
        "size_bytes": 100,
        "phase0_reported_size_bytes": 101,
        "phase0_size_matches_verified_bytes": False,
        "remote_verification_status": "verified",
        "access_scope": "metadata_only_no_raw_scan",
    }

    audit = build_method_audit_payload(
        fixture_events=_l2_events(),
        remote_inventory=remote_inventory,
        phase0_inventory=phase0_inventory,
        authoritative_manifests=manifests,
    )
    assert audit["audit_kind"] == "pmxt_l2_method_audit_v1"
    assert audit["data_access"] == "local_fixture_and_metadata_only"
    assert audit["formal_aggressor_metrics"] == {
        "emitted": False,
        "reason": "no_authoritative_fill_in_l2_input",
    }
    assert audit["method_gates"]["snapshot_before_delta"]["status"] == "PASS"
    assert audit["method_gates"]["gap_requires_new_snapshot_epoch"] == {
        "status": "PASS",
        "invalidation_count": 1,
        "resnapshot_count": 1,
        "epoch_increment_count": 1,
    }
    assert audit["method_gates"]["reconnect_requires_new_snapshot_epoch"] == {
        "status": "PASS",
        "invalidation_count": 1,
        "resnapshot_count": 1,
        "epoch_increment_count": 1,
    }
    assert audit["fixture_tick_metrics"] == [
        tick.to_dict() for tick in build_l2_metric_ticks(_l2_events())
    ]
    assert {
        "bbo",
        "spread",
        "top_n_depth",
        "imbalance",
        "microprice",
        "ofi",
        "depth_withdrawal",
        "cancel_intensity",
        "resilience",
        "markout",
        "book_walk",
        "parity_deviation",
    } <= audit["method_verification"].keys()
    assert {item["basis"] for item in audit["method_verification"]["cancel_intensity"]} == {
        "bid",
        "ask",
    }
    assert {
        item["basis"] for item in audit["method_verification"]["resilience"]
    } == {"bid", "ask", "midpoint"}
    assert {item["basis"] for item in audit["method_verification"]["markout"]} == {
        "bid",
        "ask",
        "midpoint",
    }
    assert audit["crosswalk"] == {
        "attested_remote_object_count": 1,
        "unattested_remote_object_count": 1,
        "phase0_size_mismatch_count": 1,
        "phase0_discovery_entry_count": 1,
        "hourly_l2_scan_status": "BLOCKED_UNATTESTED_REMOTE_OBJECT",
    }
    assert audit["tick_inventory_rows"] == [row.to_dict() for row in rows]


def test_phase1_sample_evidence_attests_the_97th_object_and_makes_scan_ready() -> None:
    project_root = Path(__file__).resolve().parents[2]
    phase1 = json.loads(
        (
            project_root / "artifacts/data-audit/phase1_sample_measurement.json"
        ).read_text(encoding="utf-8")
    )
    phase1_archive = phase1["files"][0]["archive"]
    phase1_inventory = phase1["files"][0]["inventory_object"]
    remote_inventory = {
        "objects": [
            {"sha256": "sha256:aaa", "byte_size": 100, "status": "verified"},
            {
                "sha256": phase1_archive["sha256"],
                "byte_size": phase1_archive["byte_size"],
                "status": "verified",
            },
        ]
    }
    phase0_inventory = {
        "objects": [
            {
                "filename": "polymarket_orderbook_2026-07-01T00.parquet",
                "hour": "2026-07-01T00:00:00.000Z",
                "size_bytes": 101,
            },
            {
                "filename": phase1_inventory["filename"],
                "hour": phase1_inventory["hour"],
                "size_bytes": phase1_inventory["size_bytes"],
                "url": phase1_inventory["url"],
            },
        ]
    }
    manifests = [
        {
            "object_sha256": "sha256:aaa",
            "byte_length": 100,
            "source_cursor": "filename:polymarket_orderbook_2026-07-01T00.parquet",
        }
    ]
    rows = build_tick_data_inventory_v1_rows(
        remote_inventory,
        phase0_inventory,
        authoritative_manifests=manifests,
        phase1_sample_measurement=phase1,
        phase1_measurement_sha256=PHASE1_SAMPLE_MEASUREMENT_SHA256,
    )
    assert [row.link_status for row in rows] == [
        "ATTESTED_MANIFEST_CROSSWALK",
        "ATTESTED_PHASE1_SAMPLE_EVIDENCE",
    ]
    audit = build_method_audit_payload(
        fixture_events=_l2_events(),
        remote_inventory=remote_inventory,
        phase0_inventory=phase0_inventory,
        authoritative_manifests=manifests,
        phase1_sample_measurement=phase1,
        phase1_measurement_sha256=PHASE1_SAMPLE_MEASUREMENT_SHA256,
    )
    assert audit["crosswalk"]["attested_remote_object_count"] == 2
    assert audit["crosswalk"]["unattested_remote_object_count"] == 0
    assert audit["crosswalk"]["hourly_l2_scan_status"] == "READY_ALL_REMOTE_ATTESTED"
    with pytest.raises(L2MethodError, match="frozen hash"):
        build_tick_data_inventory_v1_rows(
            remote_inventory,
            phase0_inventory,
            authoritative_manifests=manifests,
            phase1_sample_measurement=phase1,
            phase1_measurement_sha256="sha256:" + "0" * 64,
        )


def test_inventory_rejects_duplicate_identity_and_verified_size_mismatch() -> None:
    phase0 = {
        "objects": [
            {
                "filename": "one.parquet",
                "hour": "2026-07-01T00:00:00.000Z",
                "size_bytes": 100,
            }
        ]
    }
    manifest = {
        "object_sha256": "sha256:aaa",
        "byte_length": 100,
        "source_cursor": "filename:one.parquet",
    }
    duplicate_remote = {
        "objects": [
            {"sha256": "sha256:aaa", "byte_size": 100, "status": "verified"},
            {"sha256": "sha256:aaa", "byte_size": 100, "status": "verified"},
        ]
    }
    with pytest.raises(L2MethodError, match="duplicate verified remote SHA"):
        build_tick_data_inventory_v1_rows(
            duplicate_remote, phase0, authoritative_manifests=[manifest]
        )

    duplicate_phase0 = {
        "objects": [phase0["objects"][0], phase0["objects"][0]]
    }
    with pytest.raises(L2MethodError, match="duplicate Phase-0 filename"):
        build_tick_data_inventory_v1_rows(
            {"objects": [duplicate_remote["objects"][0]]},
            duplicate_phase0,
            authoritative_manifests=[manifest],
        )

    bad_size_manifest = {**manifest, "byte_length": 99}
    with pytest.raises(L2MethodError, match="byte size"):
        build_tick_data_inventory_v1_rows(
            {"objects": [duplicate_remote["objects"][0]]},
            phase0,
            authoritative_manifests=[bad_size_manifest],
        )


@pytest.mark.parametrize(
    "forbidden_claim",
    [
        {"nfl_alpha": "0.1"},
        {"l3_order_id": "order-1"},
        {"queue_fill_reconstructed": True},
        {"fill_reconstructed": True},
        {"authoritative_fill": True},
        {"aggressor_side": "BUY"},
    ],
)
def test_l2_schema_rejects_nfl_alpha_l3_and_positive_fill_claims(
    forbidden_claim: dict[str, object],
) -> None:
    event = {**_l2_events()[0], **forbidden_claim}
    with pytest.raises(L2MethodError, match="schema boundary"):
        build_l2_metric_ticks([event])


def test_method_audit_script_writes_only_local_fixture_and_metadata(tmp_path: Path) -> None:
    fixture_path = tmp_path / "l2-events.jsonl"
    fixture_path.write_text(
        "\n".join(json.dumps(event) for event in _l2_events()) + "\n", encoding="utf-8"
    )
    remote_path = tmp_path / "remote.json"
    remote_path.write_text(
        json.dumps(
            {
                "objects": [
                    {"sha256": "sha256:aaa", "byte_size": 100, "status": "verified"}
                ]
            }
        ),
        encoding="utf-8",
    )
    phase0_path = tmp_path / "phase0.json"
    phase0_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "filename": "polymarket_orderbook_2026-07-01T00.parquet",
                        "hour": "2026-07-01T00:00:00.000Z",
                        "size_bytes": 100,
                    }
                ],
                "summary": {"covered_hours": 1, "missing_hours": []},
            }
        ),
        encoding="utf-8",
    )
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    (manifest_root / "one.manifest.json").write_text(
        json.dumps(
            {
                "object_sha256": "sha256:aaa",
                "byte_length": 100,
                "source_cursor": "filename:polymarket_orderbook_2026-07-01T00.parquet",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "method-audit.json"
    project_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "scripts/research/build_pmxt_l2_method_audit.py",
            "--fixture",
            str(fixture_path),
            "--remote-inventory",
            str(remote_path),
            "--phase0-inventory",
            str(phase0_path),
            "--manifest-root",
            str(manifest_root),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    audit = json.loads(output_path.read_text(encoding="utf-8"))
    assert audit["data_access"] == "local_fixture_and_metadata_only"
    assert audit["raw_parquet_scanned"] is False
    assert audit["crosswalk"]["attested_remote_object_count"] == 1
    assert audit["crosswalk"]["hourly_l2_scan_status"] == "READY_ALL_REMOTE_ATTESTED"
    assert audit["fixture_tick_metrics"]
    assert audit["method_verification"]["parity_deviation"]
