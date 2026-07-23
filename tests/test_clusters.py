from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

import prediction_market.clusters as clusters_module
from prediction_market.clusters import (
    ClusterInputError,
    ClusterPairV0,
    SemanticAdjudicationV0,
    X10AuthorizationError,
    build_review_queue,
    build_x10_precision_preregistration,
    build_x10_recall_preregistration,
    cluster_gate,
    confidence_calibration,
    compute_precision,
    load_cluster_pairs_csv,
    run_x10_precision_audit,
    run_x10_recall_audit,
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


def _candidate_universe(count: int = 55) -> tuple[ClusterPairV0, ...]:
    return tuple(
        _pair(f"pair-{index:02d}", f"0.{index % 9 + 1}0")
        for index in range(count)
    )


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _authorized_card(
    *,
    scope_name: str,
    code_sha256: str,
    data_sha256: str,
    evidence: dict[str, str],
) -> dict[str, object]:
    required_locks = {
        "precision_audit": [
            "matched_sample_registered",
            "router_and_taxonomy_available",
            "gold_standard_protocol",
            "h_split_approval",
        ],
        "recall": [
            "recall_candidate_universe",
            "gold_standard_protocol",
            "router_and_taxonomy_available",
            "h_split_approval",
        ],
    }[scope_name]
    return {
        "id": "X-10",
        "execution_authorized": True,
        "result_acceptance_not_before": "2026-07-23T00:00:00Z",
        "registration_head_sha256": "sha256:" + "a" * 64,
        "authorization_scopes": {
            scope_name: {
                "authorized": True,
                "required_result_label": "FORMAL",
                "required_lock_ids": required_locks,
            },
            "live_arbitrage": {
                "authorized": False,
                "permanent_no_go": True,
                "required_result_label": "FORMAL",
                "required_lock_ids": [],
            },
        },
        "registration_locks": [
            {"id": lock_id, "status": "resolved", "evidence_ref": evidence[lock_id]}
            for lock_id in required_locks
        ],
        "preregistered_inputs": {
            scope_name: {
                "code_sha256": code_sha256,
                "data_sha256": data_sha256,
                "registered_at": "2026-07-23T00:00:01Z",
            }
        },
    }


def test_x10_child_scope_cannot_bypass_top_level_execution_gate() -> None:
    card = _authorized_card(
        scope_name="precision_audit",
        code_sha256="sha256:" + "1" * 64,
        data_sha256="sha256:" + "2" * 64,
        evidence={
            "matched_sample_registered": "sha256:" + "3" * 64,
            "router_and_taxonomy_available": "sha256:" + "4" * 64,
            "gold_standard_protocol": "sha256:" + "5" * 64,
            "h_split_approval": "sha256:" + "6" * 64,
        },
    )
    card["execution_authorized"] = False

    with pytest.raises(X10AuthorizationError, match="execution"):
        clusters_module._authorized_scope(card, "precision_audit")


def test_precision_gate_is_descriptive_and_never_authorizes_research() -> None:
    passed = cluster_gate(correct=45, reviewed=50)
    failed = cluster_gate(correct=44, reviewed=50)
    wrong_size = cluster_gate(correct=45, reviewed=49)

    assert passed.numeric_threshold_met is True
    assert failed.numeric_threshold_met is False
    assert wrong_size.numeric_threshold_met is False
    assert passed.may_advance is False
    assert failed.may_advance is False
    assert wrong_size.may_advance is False
    assert passed.live_arbitrage_authorized is False

    pairs = tuple(_pair(f"pure-{index:02d}", "0.50") for index in range(50))
    report = compute_precision(pairs, {pair.pair_id: True for pair in pairs})
    assert report.gate.numeric_threshold_met is True
    assert report.gate.may_advance is False


def test_recall_has_no_public_numeric_compute_entrypoint() -> None:
    assert "compute_recall" not in clusters_module.__all__
    assert not hasattr(clusters_module, "compute_recall")


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
    pairs = _candidate_universe()

    with pytest.raises(X10AuthorizationError, match="execution"):
        run_x10_precision_audit(
            PROJECT_ROOT,
            candidate_universe=pairs,
            selection_method="sha256_seeded_rank_v0",
            selection_seed=7,
            router_query="sports/open-markets",
            router_version="pmxt-router@unresolved-test",
            adjudications={},
            h_approval_path=PROJECT_ROOT / "does-not-exist",
            evaluation_started_at="2026-07-24T00:00:00Z",
            registration_head_sha256="sha256:" + "0" * 64,
        )

    with pytest.raises(X10AuthorizationError, match="execution"):
        run_x10_recall_audit(
            PROJECT_ROOT,
            candidate_universe=pairs,
            reviewed_match_ids=set(),
            router_query="sports/open-markets",
            router_version="pmxt-router@unresolved-test",
            h_approval_path=PROJECT_ROOT / "does-not-exist",
            evaluation_started_at="2026-07-24T00:00:00Z",
            registration_head_sha256="sha256:" + "0" * 64,
        )


def test_x10_preregistration_binds_full_universe_selection_router_and_contract_bytes() -> None:
    preregistration = build_x10_precision_preregistration(
        PROJECT_ROOT,
        candidate_universe=_candidate_universe(),
        selection_method="sha256_seeded_rank_v0",
        selection_seed=23,
        router_query="sports/open-markets",
        router_version="pmxt-router@unresolved-test",
    )

    sample_manifest = json.loads(preregistration.sample_manifest_bytes)
    assert len(sample_manifest["candidate_universe"]) == 55
    assert sample_manifest["selection"]["method"] == "sha256_seeded_rank_v0"
    assert sample_manifest["selection"]["seed"] == 23
    assert sample_manifest["router_query"] == "sports/open-markets"
    assert sample_manifest["router_version"] == "pmxt-router@unresolved-test"
    assert len(sample_manifest["selection"]["selected_pair_ids"]) == 50
    assert len(sample_manifest["selected_pairs"]) == 50
    assert preregistration.data_sha256 == (
        "sha256:" + hashlib.sha256(preregistration.sample_manifest_bytes).hexdigest()
    )

    code_manifest = json.loads(preregistration.code_manifest_bytes)
    taxonomy = PROJECT_ROOT / "contracts" / "market-relations" / "v0.yaml"
    protocol = PROJECT_ROOT / "contracts" / "x10-gold-adjudication" / "v0.yaml"
    assert code_manifest["contracts/market-relations/v0.yaml"] == _file_sha256(taxonomy)
    assert code_manifest["contracts/x10-gold-adjudication/v0.yaml"] == _file_sha256(protocol)
    assert preregistration.code_sha256 == (
        "sha256:" + hashlib.sha256(preregistration.code_manifest_bytes).hexdigest()
    )
    router_evidence = json.loads(preregistration.router_taxonomy_evidence_bytes)
    assert router_evidence == {
        "router_query": "sports/open-markets",
        "router_version": "pmxt-router@unresolved-test",
        "taxonomy_sha256": _file_sha256(taxonomy),
    }


def test_formal_precision_report_atomically_binds_all_outputs_and_is_only_g1_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_universe = _candidate_universe()
    preregistration = build_x10_precision_preregistration(
        PROJECT_ROOT,
        candidate_universe=candidate_universe,
        selection_method="sha256_seeded_rank_v0",
        selection_seed=23,
        router_query="sports/open-markets",
        router_version="pmxt-router@unresolved-test",
    )
    h_approval = tmp_path / "h-approval.txt"
    h_approval.write_bytes(b"H approves X-10 split exemption\n")
    card = _authorized_card(
        scope_name="precision_audit",
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        evidence={
            "matched_sample_registered": preregistration.data_sha256,
            "router_and_taxonomy_available": (
                preregistration.router_taxonomy_evidence_sha256
            ),
            "gold_standard_protocol": preregistration.gold_protocol_sha256,
            "h_split_approval": _file_sha256(h_approval),
        },
    )
    monkeypatch.setattr(
        clusters_module, "load_experiment_registry", lambda _root: {"X-10": card}
    )
    call_order: list[str] = []
    validated_refs: list[dict[str, str]] = []

    def validate_result_ref(
        _root: str | Path, _experiment_id: str, result_ref: dict[str, str]
    ) -> dict[str, str]:
        call_order.append("validate")
        validated_refs.append(dict(result_ref))
        return dict(result_ref)

    monkeypatch.setattr(
        clusters_module.experiments_module, "validate_result_ref", validate_result_ref
    )
    original_compute = clusters_module.compute_precision

    def ordered_compute(*args: object, **kwargs: object):
        call_order.append("compute")
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(clusters_module, "compute_precision", ordered_compute)
    selected = preregistration.selected_pairs
    adjudications = {
        pair.pair_id: SemanticAdjudicationV0(
            is_correct=index < 45,
            semantic_difference_category=(
                "none" if index < 45 else "resolution_source"
            ),
        )
        for index, pair in enumerate(selected)
    }

    report = run_x10_precision_audit(
        PROJECT_ROOT,
        candidate_universe=candidate_universe,
        selection_method="sha256_seeded_rank_v0",
        selection_seed=23,
        router_query="sports/open-markets",
        router_version="pmxt-router@unresolved-test",
        adjudications=adjudications,
        h_approval_path=h_approval,
        evaluation_started_at="2026-07-23T00:00:02Z",
        registration_head_sha256="sha256:" + "a" * 64,
    )

    assert call_order == ["validate", "compute", "validate"]
    assert report.precision.correct == 45
    assert report.precision.precision == Decimal("0.9")
    assert report.precision.gate.may_advance is False
    assert report.g1_research_decision == "G1_RESEARCH_ADVANCE"
    assert report.live_arbitrage_authorized is False
    assert report.registration_head_sha256 == "sha256:" + "a" * 64
    assert tuple(
        (item.lower, item.upper) for item in report.calibration.bins
    ) == tuple(
        zip(
            preregistration.confidence_bin_edges,
            preregistration.confidence_bin_edges[1:],
        )
    )
    semantic_counts = {
        item.category: item.count for item in report.semantic_differences
    }
    assert semantic_counts["none"] == 45
    assert semantic_counts["resolution_source"] == 5
    assert tuple(item.category for item in report.semantic_differences) == (
        preregistration.semantic_difference_categories
    )
    assert [item.priority for item in report.review_queue] == list(range(1, 51))
    assert validated_refs[0]["result_sha256"] == "sha256:" + "0" * 64
    assert validated_refs[-1]["result_sha256"] == report.result_sha256
    assert all(
        ref["registration_head_sha256"] == "sha256:" + "a" * 64
        and ref["evaluation_started_at"] == "2026-07-23T00:00:02Z"
        for ref in validated_refs
    )


def test_formal_runner_hashes_h_approval_bytes_instead_of_trusting_a_hash_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_universe = _candidate_universe()
    preregistration = build_x10_precision_preregistration(
        PROJECT_ROOT,
        candidate_universe=candidate_universe,
        selection_method="sha256_seeded_rank_v0",
        selection_seed=23,
        router_query="sports/open-markets",
        router_version="pmxt-router@unresolved-test",
    )
    h_approval = tmp_path / "h-approval.txt"
    h_approval.write_bytes(b"approved bytes\n")
    registered_approval_sha256 = _file_sha256(h_approval)
    card = _authorized_card(
        scope_name="precision_audit",
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        evidence={
            "matched_sample_registered": preregistration.data_sha256,
            "router_and_taxonomy_available": (
                preregistration.router_taxonomy_evidence_sha256
            ),
            "gold_standard_protocol": preregistration.gold_protocol_sha256,
            "h_split_approval": registered_approval_sha256,
        },
    )
    monkeypatch.setattr(
        clusters_module, "load_experiment_registry", lambda _root: {"X-10": card}
    )
    h_approval.write_bytes(b"changed after registration\n")

    with pytest.raises(X10AuthorizationError, match="h_split_approval evidence mismatch"):
        run_x10_precision_audit(
            PROJECT_ROOT,
            candidate_universe=candidate_universe,
            selection_method="sha256_seeded_rank_v0",
            selection_seed=23,
            router_query="sports/open-markets",
            router_version="pmxt-router@unresolved-test",
            adjudications={},
            h_approval_path=h_approval,
            evaluation_started_at="2026-07-23T00:00:02Z",
            registration_head_sha256="sha256:" + "a" * 64,
        )


def test_formal_recall_requires_preregistered_denominator_and_registry_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_universe = _candidate_universe()
    preregistration = build_x10_recall_preregistration(
        PROJECT_ROOT,
        candidate_universe=candidate_universe,
        router_query="sports/open-markets",
        router_version="pmxt-router@unresolved-test",
    )
    h_approval = tmp_path / "h-approval.txt"
    h_approval.write_bytes(b"H approves X-10 split exemption\n")
    card = _authorized_card(
        scope_name="recall",
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        evidence={
            "recall_candidate_universe": preregistration.data_sha256,
            "router_and_taxonomy_available": (
                preregistration.router_taxonomy_evidence_sha256
            ),
            "gold_standard_protocol": preregistration.gold_protocol_sha256,
            "h_split_approval": _file_sha256(h_approval),
        },
    )
    monkeypatch.setattr(
        clusters_module, "load_experiment_registry", lambda _root: {"X-10": card}
    )
    validated: list[dict[str, str]] = []

    def validate_result_ref(
        _root: str | Path, _experiment_id: str, result_ref: dict[str, str]
    ) -> dict[str, str]:
        validated.append(dict(result_ref))
        return dict(result_ref)

    monkeypatch.setattr(
        clusters_module.experiments_module, "validate_result_ref", validate_result_ref
    )
    reviewed = {pair.pair_id for pair in candidate_universe[:11]}

    report = run_x10_recall_audit(
        PROJECT_ROOT,
        candidate_universe=candidate_universe,
        reviewed_match_ids=reviewed,
        router_query="sports/open-markets",
        router_version="pmxt-router@unresolved-test",
        h_approval_path=h_approval,
        evaluation_started_at="2026-07-23T00:00:02Z",
        registration_head_sha256="sha256:" + "a" * 64,
    )

    assert report.recalled == 11
    assert report.denominator == 55
    assert report.recall == Decimal(11) / Decimal(55)
    assert report.live_arbitrage_authorized is False
    assert validated[-1]["data_sha256"] == preregistration.data_sha256


def test_x10_card_remains_registered_not_run() -> None:
    text = (PROJECT_ROOT / "artifacts" / "strategy-reports" / "x10_matched_cluster_audit_v0.md").read_text(
        encoding="utf-8"
    )
    assert "**Status:** `REGISTERED_NOT_RUN`" in text
    assert "G1_RESEARCH_ADVANCE" not in text


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
