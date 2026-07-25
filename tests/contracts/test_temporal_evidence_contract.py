from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction_market import contracts


def _digest(fill: str) -> str:
    return f"sha256:{fill * 64}"


def _temporal(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "temporal_version": "v0",
        "role": "market_exchange",
        "clock_domain": "polymarket.exchange",
        "utc_lower_ns": 100,
        "utc_upper_ns": 101,
        "precision_ns": 1,
        "semantics_status": "verified",
        "application_point": "trade_match",
        "clock_quality": "UTC_BOUNDED_SOFTWARE_RX",
        "clock_error_ns": 10,
        "source_sequence": "123",
        "source_clock_owner": "polymarket",
    }
    value.update(overrides)
    return value


def _root(**overrides: object) -> dict[str, object]:
    token = b"test-rfc3161-token"
    hashes = [_digest("a"), _digest("b")]
    root_sha256 = contracts.canonical_sha256(
        {
            "domain": "saf.daily-capture-root.v0",
            "capture_date": "2026-07-24",
            "manifest_hashes": hashes,
        }
    )
    value: dict[str, object] = {
        "root_version": "v0",
        "capture_date": "2026-07-24",
        "manifest_hashes": hashes,
        "root_sha256": root_sha256,
        "signing_key_id": "key_saf_capture_v0",
        "signature_b64": base64.b64encode(b"signature").decode("ascii"),
        "anchor_status": "anchored",
        "external_timestamp_at": "2026-07-24T12:00:00Z",
        "external_timestamp_token_b64": base64.b64encode(token).decode("ascii"),
        "external_timestamp_token_sha256": (
            "sha256:" + hashlib.sha256(token).hexdigest()
        ),
    }
    value.update(overrides)
    return value


def test_temporal_evidence_rejects_empty_half_open_interval() -> None:
    with pytest.raises(ValidationError, match="half-open"):
        contracts.TemporalEvidenceV0.model_validate(
            _temporal(utc_upper_ns=100)
        )


def test_temporal_evidence_requires_bound_for_utc_bounded_clock() -> None:
    with pytest.raises(ValidationError, match="clock_error_ns"):
        contracts.TemporalEvidenceV0.model_validate(
            _temporal(clock_error_ns=None)
        )


def test_temporal_evidence_allows_local_monotonic_without_utc_error() -> None:
    evidence = contracts.TemporalEvidenceV0.model_validate(
        _temporal(
            clock_quality="LOCAL_MONOTONIC_ONLY",
            clock_error_ns=None,
        )
    )
    assert evidence.clock_error_ns is None


def test_daily_capture_root_rejects_timestamp_token_hash_mismatch() -> None:
    with pytest.raises(ValidationError, match="token SHA-256"):
        contracts.DailyCaptureRootV0.model_validate(
            _root(external_timestamp_token_sha256=_digest("c"))
        )


def test_daily_capture_root_rejects_unsorted_or_duplicate_manifest_hashes() -> None:
    with pytest.raises(ValidationError, match="strictly sorted"):
        contracts.DailyCaptureRootV0.model_validate(
            _root(manifest_hashes=[_digest("b"), _digest("a")])
        )
    with pytest.raises(ValidationError, match="strictly sorted"):
        contracts.DailyCaptureRootV0.model_validate(
            _root(manifest_hashes=[_digest("a"), _digest("a")])
        )


def test_audit_verdict_blocks_unknown_reason_code() -> None:
    with pytest.raises(ValidationError, match="reason_code"):
        contracts.AuditVerdictV0.model_validate(
            {
                "audit_version": "v0",
                "scope": "EVENT_RESULT",
                "verdict": "BLOCK",
                "reason_codes": ["NOT_A_REASON"],
                "evidence_refs": [_digest("d")],
            }
        )
