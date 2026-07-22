from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from prediction_market.clusters import (
    ClusterInputError,
    ClusterPairV0,
    MissingRecallDenominatorError,
    X10AuthorizationError,
    build_review_queue,
    cluster_gate,
    confidence_calibration,
    compute_precision,
    compute_recall,
    load_cluster_pairs_csv,
    run_x10_precision_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pair(
    pair_id: str,
    confidence: str,
    relation: str = "identity",
) -> ClusterPairV0:
    return ClusterPairV0(
        pair_id=pair_id,
        left_market_id=f"market_left-{pair_id}",
        right_market_id=f"market_right-{pair_id}",
        relation_type=relation,
        confidence=Decimal(confidence),
        router_version="pmxt-router@unresolved-test",
        observed_at="2026-07-22T18:00:00Z",
    )


def test_precision_gate_requires_at_least_45_of_exactly_50() -> None:
    assert cluster_gate(correct=45, reviewed=50).may_advance is True
    assert cluster_gate(correct=44, reviewed=50).may_advance is False
    assert cluster_gate(correct=45, reviewed=49).may_advance is False
    assert cluster_gate(correct=45, reviewed=50).live_arbitrage_authorized is False


def test_recall_is_refused_without_candidate_universe() -> None:
    with pytest.raises(MissingRecallDenominatorError, match="candidate universe"):
        compute_recall({"pair-1"}, candidate_universe=None)


def test_recall_requires_reviewed_matches_to_belong_to_universe() -> None:
    with pytest.raises(ClusterInputError, match="outside candidate universe"):
        compute_recall({"pair-2"}, candidate_universe={"pair-1"})


def test_review_queue_is_deterministic_and_prioritizes_low_confidence() -> None:
    queue = build_review_queue(
        [_pair("pair-2", "0.90"), _pair("pair-1", "0.40")]
    )

    assert [item.pair_id for item in queue] == ["pair-1", "pair-2"]
    assert all(item.status == "PENDING_MANUAL_REVIEW" for item in queue)


def test_relation_taxonomy_is_closed_and_confidence_rejects_float() -> None:
    with pytest.raises(ClusterInputError, match="relation_type"):
        _pair("pair-1", "0.50", relation="same-event")
    with pytest.raises(ClusterInputError, match="Decimal"):
        ClusterPairV0(
            pair_id="pair-1",
            left_market_id="market_left",
            right_market_id="market_right",
            relation_type="identity",
            confidence=0.5,  # type: ignore[arg-type]
            router_version="router@v0",
            observed_at="2026-07-22T18:00:00Z",
        )


def test_precision_refuses_partial_or_duplicate_adjudication() -> None:
    pairs = [_pair("pair-1", "0.50"), _pair("pair-2", "0.60")]
    with pytest.raises(ClusterInputError, match="exactly once"):
        compute_precision(pairs, {"pair-1": True})


def test_confidence_calibration_uses_explicit_locked_bins() -> None:
    pairs = [
        _pair("pair-1", "0.20"),
        _pair("pair-2", "0.40"),
        _pair("pair-3", "0.80"),
        _pair("pair-4", "0.90"),
    ]
    report = confidence_calibration(
        pairs,
        {"pair-1": False, "pair-2": True, "pair-3": True, "pair-4": True},
        bin_edges=(Decimal("0"), Decimal("0.5"), Decimal("1")),
    )

    assert [item.count for item in report.bins] == [2, 2]
    assert report.bins[0].observed_precision == Decimal("0.5")
    assert report.bins[1].observed_precision == Decimal("1.0")
    assert report.expected_calibration_error == Decimal("0.175")


def test_confidence_bins_cannot_be_inferred_or_malformed() -> None:
    pairs = [_pair("pair-1", "0.50")]
    with pytest.raises(ClusterInputError, match="bin_edges"):
        confidence_calibration(
            pairs,
            {"pair-1": True},
            bin_edges=(Decimal("0.1"), Decimal("1")),
        )


def test_cluster_csv_import_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(
        "pair_id,left_market_id,right_market_id,relation_type,confidence,router_version,observed_at\n"
        "pair-1,market_left-pair-1,market_right-pair-1,identity,0.80,pmxt-router@unresolved-test,2026-07-22T18:00:00Z\n",
        encoding="utf-8",
    )
    assert load_cluster_pairs_csv(path) == (_pair("pair-1", "0.80"),)

    path.write_text(path.read_text(encoding="utf-8").replace("observed_at", "hidden"), encoding="utf-8")
    with pytest.raises(ClusterInputError, match="columns"):
        load_cluster_pairs_csv(path)

    path.write_text(
        "pair_id,left_market_id,right_market_id,relation_type,confidence,router_version,observed_at\n"
        "pair-1,market_left-pair-1,market_right-pair-1,identity,8e-1,pmxt-router@unresolved-test,2026-07-22T18:00:00Z\n",
        encoding="utf-8",
    )
    with pytest.raises(ClusterInputError, match="canonical Decimal"):
        load_cluster_pairs_csv(path)


def test_x10_formal_audit_is_blocked_by_current_registry() -> None:
    pairs = tuple(_pair(f"pair-{index:02d}", "0.50") for index in range(50))
    adjudications = {pair.pair_id: True for pair in pairs}

    with pytest.raises(X10AuthorizationError, match="unresolved|preregistered"):
        run_x10_precision_audit(
            PROJECT_ROOT,
            pairs=pairs,
            adjudications=adjudications,
        )


def test_all_eight_module_reports_have_required_sections_and_no_demo_scope() -> None:
    reports = sorted(
        (PROJECT_ROOT / "artifacts" / "strategy-reports" / "modules").glob("*.md")
    )
    assert len(reports) == 8
    for report in reports:
        text = report.read_text(encoding="utf-8")
        assert "## Evidence" in text
        assert "## Real execution difficulty" in text
        assert "## Honest backtest design" in text
        assert "NO_DEMO_NO_LIVE" in text

    llm = (PROJECT_ROOT / "artifacts" / "strategy-reports" / "modules" / "llm.md").read_text(
        encoding="utf-8"
    )
    assert "slow path only" in llm
