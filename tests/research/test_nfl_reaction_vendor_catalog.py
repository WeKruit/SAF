from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_market.research.vendor_sample_audit import (
    VendorAuditError,
    canonical_sha256,
    load_reaction_method_catalog,
    load_source_catalog,
    validate_vendor_sample_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = (
    PROJECT_ROOT
    / "registries"
    / "vendors"
    / "nfl_reaction_source_catalog_v1.json"
)
SCHEMA_PATH = (
    PROJECT_ROOT
    / "registries"
    / "vendors"
    / "vendor_sample_audit_v1.schema.json"
)
METHOD_CATALOG_PATH = (
    PROJECT_ROOT
    / "registries"
    / "methods"
    / "nfl_reaction_method_catalog_v1.json"
)

EXPECTED_SOURCE_IDS = {
    "SRC-BETFAIR-HISTORICAL",
    "SRC-JON-BECKER-PM-ANALYSIS",
    "SRC-KALSHI-OFFICIAL-L2",
    "SRC-ODDPOOL",
    "SRC-PMXT-KALSHI",
    "SRC-PMXT-POLYMARKET-V2",
    "SRC-POLYMARKET-OFFICIAL-L2",
    "SRC-POLYMARKETDATA",
    "SRC-PROBALYTICS",
    "SRC-TELONEX",
}


def test_source_catalog_is_content_addressed_and_capability_explicit() -> None:
    catalog = load_source_catalog(CATALOG_PATH)

    assert catalog["schema_version"] == "NFLReactionSourceCatalogV1"
    assert [card["source_id"] for card in catalog["cards"]] == sorted(
        EXPECTED_SOURCE_IDS
    )
    assert catalog["catalog_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in catalog.items()
            if key != "catalog_sha256"
        }
    )

    for card in catalog["cards"]:
        assert card["content_sha256"] == canonical_sha256(
            {
                key: value
                for key, value in card.items()
                if key != "content_sha256"
            }
        )
        assert card["claim_boundary"]
        assert card["timestamps"]
        assert set(card["capabilities"]) == {
            "historical_l2",
            "prospective_l2",
            "l3_order_identity",
            "exact_queue_position",
            "authoritative_trade_direction",
            "taker_book_walk",
            "exact_maker_fill",
        }
        assert card["capabilities"]["l3_order_identity"] is False
        assert card["capabilities"]["exact_queue_position"] is False
        assert card["capabilities"]["exact_maker_fill"] is False


def test_official_sources_are_canonical_and_vendors_require_sample() -> None:
    cards = {
        card["source_id"]: card for card in load_source_catalog(CATALOG_PATH)["cards"]
    }

    assert cards["SRC-POLYMARKET-OFFICIAL-L2"]["status"] == "CANONICAL"
    assert cards["SRC-KALSHI-OFFICIAL-L2"]["status"] == "CANONICAL"
    assert cards["SRC-PMXT-POLYMARKET-V2"]["status"] == "APPROVED_ARCHIVE"
    assert cards["SRC-PMXT-KALSHI"]["status"] == "STALE_ARCHIVE"
    assert cards["SRC-JON-BECKER-PM-ANALYSIS"]["capabilities"][
        "historical_l2"
    ] is False

    for source_id in {
        "SRC-ODDPOOL",
        "SRC-POLYMARKETDATA",
        "SRC-PROBALYTICS",
        "SRC-TELONEX",
    }:
        assert cards[source_id]["status"] == "SAMPLE_REQUIRED"
        assert cards[source_id]["license_status"] == "PENDING"

    assert cards["SRC-POLYMARKETDATA"]["granularity"] == "ONE_MINUTE_SNAPSHOT"


def test_vendor_sample_audit_requires_reconstruction_and_license_evidence() -> None:
    audit = {
        "schema_version": "VendorSampleAuditV1",
        "sample_id": "SAMPLE-POLY-001",
        "source_id": "SRC-TELONEX",
        "market_ids": ["condition:0xabc"],
        "window_start": "2026-07-26T00:00:00Z",
        "window_end": "2026-07-26T02:00:00Z",
        "raw_manifest_sha256": "sha256:" + "1" * 64,
        "normalized_manifest_sha256": "sha256:" + "2" * 64,
        "grain": "EVENT_DRIVEN",
        "timestamp_evidence": {
            "source_clock": "POLYMARKET_EXCHANGE_MS",
            "vendor_receive_clock": "VENDOR_UTC_US",
            "resolution_claim": "source milliseconds; vendor microseconds",
        },
        "continuity": {
            "initial_snapshot": True,
            "gap_count": 0,
            "duplicate_count": 0,
            "out_of_order_count": 0,
        },
        "reconstruction": {
            "run_one_sha256": "sha256:" + "3" * 64,
            "run_two_sha256": "sha256:" + "3" * 64,
            "official_reconciliation_status": "PASS",
        },
        "trade_direction": {
            "provenance": "AUTHORITATIVE_FILL",
            "inference_used": False,
        },
        "license": {
            "status": "APPROVED",
            "evidence_ref": "I-APPROVAL-SAMPLE-001",
        },
        "verdict": "PASS",
    }

    validated = validate_vendor_sample_audit(audit)
    assert validated["verdict"] == "PASS"

    mismatched = json.loads(json.dumps(audit))
    mismatched["reconstruction"]["run_two_sha256"] = "sha256:" + "4" * 64
    with pytest.raises(VendorAuditError, match="reconstruction hashes"):
        validate_vendor_sample_audit(mismatched)

    unlicensed = json.loads(json.dumps(audit))
    unlicensed["license"]["status"] = "PENDING"
    with pytest.raises(VendorAuditError, match="PASS requires approved license"):
        validate_vendor_sample_audit(unlicensed)

    inferred = json.loads(json.dumps(audit))
    inferred["trade_direction"]["provenance"] = "QUOTE_INFERENCE"
    inferred["trade_direction"]["inference_used"] = True
    inferred["verdict"] = "PASS"
    with pytest.raises(VendorAuditError, match="authoritative trade direction"):
        validate_vendor_sample_audit(inferred)


def test_vendor_sample_schema_is_closed_and_matches_runtime_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$id"] == "urn:saf:vendor-sample-audit:v1"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "sample_id",
        "source_id",
        "market_ids",
        "window_start",
        "window_end",
        "raw_manifest_sha256",
        "normalized_manifest_sha256",
        "grain",
        "timestamp_evidence",
        "continuity",
        "reconstruction",
        "trade_direction",
        "license",
        "verdict",
    }


def test_reaction_method_catalog_separates_method_from_empirical_evidence() -> None:
    catalog = load_reaction_method_catalog(METHOD_CATALOG_PATH)
    assert [card["method_id"] for card in catalog["cards"]] == [
        "METHOD-BBE-SYNTHETIC-EXCHANGE",
        "METHOD-CONT-ORDER-FLOW-IMBALANCE",
        "METHOD-POLYMARKET-MICROSTRUCTURE-2026",
    ]
    assert catalog["catalog_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in catalog.items()
            if key != "catalog_sha256"
        }
    )
    by_id = {card["method_id"]: card for card in catalog["cards"]}
    assert by_id["METHOD-BBE-SYNTHETIC-EXCHANGE"]["use_status"] == (
        "SYNTHETIC_STRESS_TEST_ONLY"
    )
    assert by_id["METHOD-CONT-ORDER-FLOW-IMBALANCE"]["data_requirement"] == (
        "CONTINUOUS_L2"
    )
    assert by_id["METHOD-POLYMARKET-MICROSTRUCTURE-2026"][
        "authoritative_trade_direction_required"
    ] is True
    for card in catalog["cards"]:
        assert card["content_sha256"] == canonical_sha256(
            {
                key: value
                for key, value in card.items()
                if key != "content_sha256"
            }
        )
        assert card["published_result_is_project_evidence"] is False
