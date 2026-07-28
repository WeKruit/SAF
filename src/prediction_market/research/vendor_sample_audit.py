"""Fail-closed source catalog and vendor sample audit for NFL reaction data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


_SHA256_PREFIX = "sha256:"
_CARD_KEYS = {
    "source_id",
    "name",
    "source_kind",
    "status",
    "observed_at",
    "urls",
    "coverage",
    "granularity",
    "auth",
    "license_status",
    "timestamps",
    "capabilities",
    "audit_requirements",
    "claim_boundary",
    "content_sha256",
}
_CAPABILITY_KEYS = {
    "historical_l2",
    "prospective_l2",
    "l3_order_identity",
    "exact_queue_position",
    "authoritative_trade_direction",
    "taker_book_walk",
    "exact_maker_fill",
}
_CATALOG_KEYS = {
    "schema_version",
    "observed_at",
    "cards",
    "catalog_sha256",
}
_AUDIT_KEYS = {
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
_METHOD_CATALOG_KEYS = {
    "schema_version",
    "observed_at",
    "cards",
    "catalog_sha256",
}
_METHOD_CARD_KEYS = {
    "method_id",
    "title",
    "source_url",
    "source_version",
    "use_status",
    "data_requirement",
    "authoritative_trade_direction_required",
    "published_result_is_project_evidence",
    "reusable_method",
    "prohibited_claims",
    "content_sha256",
}


class VendorAuditError(ValueError):
    """A source claim or sample cannot pass the governed audit."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VendorAuditError(f"{context} must be an object")
    return value


def _closed_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise VendorAuditError(
            f"{context} fields differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VendorAuditError(f"{field} must be nonempty text")
    return value


def _utc(value: object, *, field: str) -> datetime:
    text = _nonempty_text(value, field=field)
    if not text.endswith("Z"):
        raise VendorAuditError(f"{field} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise VendorAuditError(f"{field} must be canonical UTC") from error
    return parsed


def _sha(value: object, *, field: str) -> str:
    text = _nonempty_text(value, field=field)
    if (
        not text.startswith(_SHA256_PREFIX)
        or len(text) != len(_SHA256_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise VendorAuditError(f"{field} must be sha256:<64 lowercase hex>")
    return text


def _validate_card(value: object) -> dict[str, Any]:
    card = dict(_mapping(value, context="source card"))
    _closed_keys(card, _CARD_KEYS, context="source card")
    source_id = _nonempty_text(card["source_id"], field="source_id")
    if not source_id.startswith("SRC-"):
        raise VendorAuditError("source_id must start with SRC-")
    _nonempty_text(card["name"], field=f"{source_id}.name")
    _nonempty_text(card["source_kind"], field=f"{source_id}.source_kind")
    if card["status"] not in {
        "CANONICAL",
        "APPROVED_ARCHIVE",
        "STALE_ARCHIVE",
        "SAMPLE_REQUIRED",
        "BENCHMARK_ONLY",
        "DATASET_ONLY",
    }:
        raise VendorAuditError(f"{source_id}.status is invalid")
    _utc(card["observed_at"], field=f"{source_id}.observed_at")
    urls = card["urls"]
    if (
        not isinstance(urls, list)
        or not urls
        or any(not isinstance(url, str) or not url.startswith("https://") for url in urls)
    ):
        raise VendorAuditError(f"{source_id}.urls must be nonempty HTTPS URLs")
    _mapping(card["coverage"], context=f"{source_id}.coverage")
    _nonempty_text(card["granularity"], field=f"{source_id}.granularity")
    _nonempty_text(card["auth"], field=f"{source_id}.auth")
    if card["license_status"] not in {"APPROVED", "RESEARCH_ONLY", "PENDING"}:
        raise VendorAuditError(f"{source_id}.license_status is invalid")
    timestamps = card["timestamps"]
    if not isinstance(timestamps, list) or not timestamps:
        raise VendorAuditError(f"{source_id}.timestamps must be nonempty")
    for index, timestamp in enumerate(timestamps):
        item = _mapping(
            timestamp, context=f"{source_id}.timestamps[{index}]"
        )
        if set(item) != {"field", "clock_domain", "semantics"}:
            raise VendorAuditError(
                f"{source_id}.timestamps[{index}] fields differ"
            )
        for field, entry in item.items():
            _nonempty_text(
                entry, field=f"{source_id}.timestamps[{index}].{field}"
            )
    capabilities = _mapping(
        card["capabilities"], context=f"{source_id}.capabilities"
    )
    _closed_keys(
        capabilities, _CAPABILITY_KEYS, context=f"{source_id}.capabilities"
    )
    if any(type(flag) is not bool for flag in capabilities.values()):
        raise VendorAuditError(f"{source_id}.capabilities must be boolean")
    if (
        capabilities["l3_order_identity"]
        or capabilities["exact_queue_position"]
        or capabilities["exact_maker_fill"]
    ):
        raise VendorAuditError(
            f"{source_id} cannot claim L3, exact queue, or exact maker fill"
        )
    requirements = card["audit_requirements"]
    if (
        not isinstance(requirements, list)
        or not requirements
        or any(not isinstance(item, str) or not item for item in requirements)
    ):
        raise VendorAuditError(
            f"{source_id}.audit_requirements must be nonempty"
        )
    _nonempty_text(
        card["claim_boundary"], field=f"{source_id}.claim_boundary"
    )
    expected_hash = canonical_sha256(
        {key: entry for key, entry in card.items() if key != "content_sha256"}
    )
    if card["content_sha256"] != expected_hash:
        raise VendorAuditError(f"{source_id}.content_sha256 mismatch")
    return card


def load_source_catalog(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                VendorAuditError(f"non-finite JSON value: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VendorAuditError("source catalog is not strict JSON") from error
    catalog = dict(_mapping(value, context="source catalog"))
    _closed_keys(catalog, _CATALOG_KEYS, context="source catalog")
    if catalog["schema_version"] != "NFLReactionSourceCatalogV1":
        raise VendorAuditError("source catalog schema_version is invalid")
    _utc(catalog["observed_at"], field="source catalog observed_at")
    cards_value = catalog["cards"]
    if not isinstance(cards_value, list) or not cards_value:
        raise VendorAuditError("source catalog cards must be nonempty")
    cards = [_validate_card(card) for card in cards_value]
    ids = [card["source_id"] for card in cards]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise VendorAuditError("source catalog IDs must be unique and sorted")
    expected_hash = canonical_sha256(
        {
            key: entry
            for key, entry in catalog.items()
            if key != "catalog_sha256"
        }
    )
    if catalog["catalog_sha256"] != expected_hash:
        raise VendorAuditError("source catalog catalog_sha256 mismatch")
    catalog["cards"] = cards
    return catalog


def validate_vendor_sample_audit(value: object) -> dict[str, Any]:
    audit = dict(_mapping(value, context="vendor sample audit"))
    _closed_keys(audit, _AUDIT_KEYS, context="vendor sample audit")
    if audit["schema_version"] != "VendorSampleAuditV1":
        raise VendorAuditError("vendor sample schema_version is invalid")
    _nonempty_text(audit["sample_id"], field="sample_id")
    _nonempty_text(audit["source_id"], field="source_id")
    market_ids = audit["market_ids"]
    if (
        not isinstance(market_ids, list)
        or not market_ids
        or any(not isinstance(item, str) or not item for item in market_ids)
    ):
        raise VendorAuditError("market_ids must be nonempty")
    if _utc(audit["window_start"], field="window_start") >= _utc(
        audit["window_end"], field="window_end"
    ):
        raise VendorAuditError("sample window must be increasing")
    _sha(audit["raw_manifest_sha256"], field="raw_manifest_sha256")
    _sha(
        audit["normalized_manifest_sha256"],
        field="normalized_manifest_sha256",
    )
    if audit["grain"] not in {"EVENT_DRIVEN", "INTERVAL_SNAPSHOT"}:
        raise VendorAuditError("grain is invalid")

    timestamp = _mapping(
        audit["timestamp_evidence"], context="timestamp_evidence"
    )
    _closed_keys(
        timestamp,
        {"source_clock", "vendor_receive_clock", "resolution_claim"},
        context="timestamp_evidence",
    )
    for field, entry in timestamp.items():
        _nonempty_text(entry, field=f"timestamp_evidence.{field}")

    continuity = _mapping(audit["continuity"], context="continuity")
    _closed_keys(
        continuity,
        {
            "initial_snapshot",
            "gap_count",
            "duplicate_count",
            "out_of_order_count",
        },
        context="continuity",
    )
    if type(continuity["initial_snapshot"]) is not bool:
        raise VendorAuditError("initial_snapshot must be boolean")
    for field in {"gap_count", "duplicate_count", "out_of_order_count"}:
        if type(continuity[field]) is not int or continuity[field] < 0:
            raise VendorAuditError(f"{field} must be a non-negative integer")

    reconstruction = _mapping(
        audit["reconstruction"], context="reconstruction"
    )
    _closed_keys(
        reconstruction,
        {
            "run_one_sha256",
            "run_two_sha256",
            "official_reconciliation_status",
        },
        context="reconstruction",
    )
    first_hash = _sha(
        reconstruction["run_one_sha256"], field="run_one_sha256"
    )
    second_hash = _sha(
        reconstruction["run_two_sha256"], field="run_two_sha256"
    )
    if first_hash != second_hash:
        raise VendorAuditError("reconstruction hashes do not match")
    if reconstruction["official_reconciliation_status"] not in {
        "PASS",
        "FAIL",
        "NOT_AVAILABLE",
    }:
        raise VendorAuditError("official reconciliation status is invalid")

    trade_direction = _mapping(
        audit["trade_direction"], context="trade_direction"
    )
    _closed_keys(
        trade_direction,
        {"provenance", "inference_used"},
        context="trade_direction",
    )
    if type(trade_direction["inference_used"]) is not bool:
        raise VendorAuditError("trade_direction.inference_used must be boolean")

    license_value = _mapping(audit["license"], context="license")
    _closed_keys(
        license_value, {"status", "evidence_ref"}, context="license"
    )
    if license_value["status"] not in {"APPROVED", "PENDING", "BLOCKED"}:
        raise VendorAuditError("license.status is invalid")
    _nonempty_text(license_value["evidence_ref"], field="license.evidence_ref")

    if audit["verdict"] not in {"PASS", "FAIL", "PENDING"}:
        raise VendorAuditError("verdict is invalid")
    if audit["verdict"] == "PASS":
        if license_value["status"] != "APPROVED":
            raise VendorAuditError("PASS requires approved license")
        if (
            trade_direction["provenance"] != "AUTHORITATIVE_FILL"
            or trade_direction["inference_used"]
        ):
            raise VendorAuditError(
                "PASS requires authoritative trade direction"
            )
        if (
            not continuity["initial_snapshot"]
            or continuity["gap_count"] != 0
            or reconstruction["official_reconciliation_status"] != "PASS"
        ):
            raise VendorAuditError(
                "PASS requires continuous reconciled reconstruction"
            )
    return audit


def load_reaction_method_catalog(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VendorAuditError(
            "reaction method catalog is not strict JSON"
        ) from error
    catalog = dict(_mapping(value, context="reaction method catalog"))
    _closed_keys(
        catalog, _METHOD_CATALOG_KEYS, context="reaction method catalog"
    )
    if catalog["schema_version"] != "NFLReactionMethodCatalogV1":
        raise VendorAuditError("reaction method schema_version is invalid")
    _utc(catalog["observed_at"], field="reaction method observed_at")
    cards_value = catalog["cards"]
    if not isinstance(cards_value, list) or not cards_value:
        raise VendorAuditError("reaction method cards must be nonempty")
    cards: list[dict[str, Any]] = []
    for raw_card in cards_value:
        card = dict(_mapping(raw_card, context="reaction method card"))
        _closed_keys(
            card, _METHOD_CARD_KEYS, context="reaction method card"
        )
        method_id = _nonempty_text(card["method_id"], field="method_id")
        if not method_id.startswith("METHOD-"):
            raise VendorAuditError("method_id must start with METHOD-")
        for field in {
            "title",
            "source_version",
            "use_status",
            "data_requirement",
            "reusable_method",
        }:
            _nonempty_text(card[field], field=f"{method_id}.{field}")
        source_url = _nonempty_text(
            card["source_url"], field=f"{method_id}.source_url"
        )
        if not source_url.startswith("https://"):
            raise VendorAuditError(f"{method_id}.source_url must be HTTPS")
        for field in {
            "authoritative_trade_direction_required",
            "published_result_is_project_evidence",
        }:
            if type(card[field]) is not bool:
                raise VendorAuditError(f"{method_id}.{field} must be boolean")
        if card["published_result_is_project_evidence"]:
            raise VendorAuditError(
                f"{method_id} cannot treat a published result as project evidence"
            )
        prohibited = card["prohibited_claims"]
        if (
            not isinstance(prohibited, list)
            or not prohibited
            or any(not isinstance(item, str) or not item for item in prohibited)
        ):
            raise VendorAuditError(
                f"{method_id}.prohibited_claims must be nonempty"
            )
        expected_card_hash = canonical_sha256(
            {
                key: entry
                for key, entry in card.items()
                if key != "content_sha256"
            }
        )
        if card["content_sha256"] != expected_card_hash:
            raise VendorAuditError(f"{method_id}.content_sha256 mismatch")
        cards.append(card)
    method_ids = [card["method_id"] for card in cards]
    if method_ids != sorted(method_ids) or len(method_ids) != len(
        set(method_ids)
    ):
        raise VendorAuditError("reaction method IDs must be unique and sorted")
    expected_catalog_hash = canonical_sha256(
        {
            key: entry
            for key, entry in catalog.items()
            if key != "catalog_sha256"
        }
    )
    if catalog["catalog_sha256"] != expected_catalog_hash:
        raise VendorAuditError("reaction method catalog_sha256 mismatch")
    catalog["cards"] = cards
    return catalog


__all__ = [
    "VendorAuditError",
    "canonical_sha256",
    "load_reaction_method_catalog",
    "load_source_catalog",
    "validate_vendor_sample_audit",
]
