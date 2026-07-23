from __future__ import annotations

import json
from pathlib import Path

from prediction_market.contracts import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "game-state"
    / "nfl_season_state_validation_v2.json"
)


def _artifact() -> dict[str, object]:
    document = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_nfl_season_state_artifact_is_self_hashed() -> None:
    document = _artifact()
    artifact_sha256 = document.pop("artifact_sha256")

    assert artifact_sha256 == canonical_sha256(document)


def test_nfl_season_state_artifact_reports_complete_deterministic_census() -> None:
    document = _artifact()
    validation = document["validation"]
    assert isinstance(validation, dict)

    assert validation["reducer_version"] == "v3"
    assert validation["games_total"] == 285
    assert validation["completed_games"] == 285
    assert validation["fail_closed_games"] == 0
    assert validation["transitions"] == 48_486
    assert validation["scan_runs"] == 2
    assert validation["deterministic"] is True
    assert validation["failures"] == []
    assert validation["canonical_state_sha256"] == (
        "sha256:"
        "63fc2732726e204fcf49500ae39ff40a4ec3b3e1dd9825459ad3369dc53697ed"
    )
    assert all(
        audit["mismatches"] == 0
        for audit in validation["field_audits"]
    )


def test_nfl_season_latency_is_reducer_only_and_has_exact_sample_count() -> None:
    document = _artifact()
    validation = document["validation"]
    boundary = document["evidence_boundary"]
    assert isinstance(validation, dict)
    assert isinstance(boundary, dict)
    latency = validation["reducer_latency"]
    assert isinstance(latency, dict)

    assert latency["samples"] == 96_972
    assert 0 < latency["p50_ns"] <= latency["p95_ns"]
    assert latency["p95_ns"] <= latency["p99_ns"] <= latency["max_ns"]
    assert latency["operations_per_second"] > 0
    assert boundary["reducer_latency_scope"] == (
        "state_plus_normalized_event_to_next_state_only; excludes source "
        "parsing, I/O, envelopes, model, market join, and network"
    )


def test_nfl_season_artifact_does_not_claim_model_or_market_evidence() -> None:
    boundary = _artifact()["evidence_boundary"]
    assert isinstance(boundary, dict)

    assert boundary["full_season_event_envelope_integration"] is False
    assert boundary["probability_model_evaluated"] is False
    assert boundary["model_accuracy_claimed"] is False
    assert boundary["prediction_market_aligned"] is False
    assert boundary["matched_as_of_rows"] == 0
    assert boundary["alpha_status"] == "NOT_VERIFIED"
    assert boundary["symmetry_status"] == "NOT_VERIFIED"
    assert boundary["live_sla_claimed"] is False
