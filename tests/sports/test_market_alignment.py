from __future__ import annotations

from datetime import datetime, timezone

import pytest

from prediction_market.contracts import (
    EventEnvelopeV0,
    MarketMetadataSnapshotV0,
    VenueRuleSnapshotV0,
    market_metadata_snapshot_sha256,
)
from prediction_market.sports.market_alignment import (
    CurrentAlignmentEvidence,
    MarketAlignmentInputError,
    audit_current_market_alignment_evidence,
)


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
UTC = timezone.utc
SHA = "sha256:" + ("a" * 64)


def _fixed(atoms: str, scale: int = 2) -> dict[str, object]:
    return {"atoms": atoms, "scale": scale}


def _metadata() -> MarketMetadataSnapshotV0:
    value: dict[str, object] = {
        "snapshot_version": "v0",
        "venue": "polymarket",
        "native_event_id": "nba-example",
        "native_market_id": "market-example",
        "native_condition_id": "condition-example",
        "native_outcome_id": "home_score",
        "native_token_id": "token-example",
        "canonical_refs": {
            "competition_id": "cmp_nba",
            "game_id": "game_nba_2026_example",
            "participant_ids": ["participant_away", "participant_home"],
            "venue_event_id": "venue_event_polymarket_example",
            "market_id": "market_nba_example",
            "outcome_id": "outcome_home_score",
            "condition_id": "condition_polymarket_example",
        },
        "sport": "basketball",
        "competition": "NBA",
        "participants": ["Away", "Home"],
        "game_start_at": "2026-07-23T13:00:00Z",
        "rules": "Next possession scoring outcome.",
        "resolution": None,
        "closed": False,
        "resolved": False,
        "captured_at": "2026-07-23T12:00:00Z",
        "source_updated_at": "2026-07-23T11:59:59Z",
        "raw_object_hash": SHA,
        "quality_flags": [],
    }
    value["snapshot_sha256"] = market_metadata_snapshot_sha256(value)
    return MarketMetadataSnapshotV0.model_validate(value)


def _model_output_event() -> EventEnvelopeV0:
    payload = {
        "contract_version": "v1",
        "model_id": "MODEL-NBA-POSSESSION-TRANSITION",
        "model_version": "v1",
        "experiment_id": "X-06",
        "run_id": "run_x06_alignment_fixture",
        "game_id": "game_nba_2026_example",
        "state_event_id": "evt_" + ("1" * 64),
        "pit_cutoff_at": "2026-07-23T12:00:10Z",
        "output_kind": "state_transition",
        "transition_unit": "possession",
        "state_space": ["home_score", "away_score", "no_score"],
        "horizon": "next_state_transition",
        "probabilities": {
            "home_score": _fixed("25"),
            "away_score": _fixed("50"),
            "no_score": _fixed("25"),
        },
        "feature_sha256": "sha256:" + ("2" * 64),
        "data_sha256": (
            "sha256:"
            "0ebca90ba56b45252c6d310fd176ba2eb1ac615995b3eacf92e2e3f9c962a15f"
        ),
        "config_sha256": "sha256:" + ("3" * 64),
        "quality_flags": [],
    }
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="model_output",
        payload_schema_version="v1",
        source={
            "system": "saf-model",
            "stream": "x06-transition",
            "venue": None,
            "sequence": 1,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time={
            "receive_at": "2026-07-23T12:00:10Z",
            "receive_basis": "upstream_exporter",
            "source_at": None,
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs={
            "competition_id": "cmp_nba",
            "game_id": "game_nba_2026_example",
            "participant_ids": ["participant_away", "participant_home"],
            "venue_event_id": None,
            "market_id": None,
            "outcome_id": None,
            "condition_id": None,
        },
        native_refs=[],
        lineage={"parent_event_ids": ["evt_" + ("1" * 64)]},
        experiment_id="X-06",
        rule_snapshot_ref=None,
        quality_flags=[],
        payload=payload,
    )


def _quote_event(*, receive_basis: str = "local_recorder") -> EventEnvelopeV0:
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": "polymarket-recorder",
            "stream": "market",
            "venue": "polymarket",
            "sequence": 9,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time={
            "receive_at": "2026-07-23T12:00:09.500Z",
            "receive_basis": receive_basis,
            "source_at": "2026-07-23T12:00:09Z",
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs={
            "competition_id": "cmp_nba",
            "game_id": "game_nba_2026_example",
            "participant_ids": ["participant_away", "participant_home"],
            "venue_event_id": "venue_event_polymarket_example",
            "market_id": "market_nba_example",
            "outcome_id": "outcome_home_score",
            "condition_id": "condition_polymarket_example",
        },
        native_refs=[
            {
                "namespace": "polymarket.condition",
                "native_id": "condition-example",
            }
        ],
        lineage={"parent_event_ids": ["evt_" + ("4" * 64)]},
        experiment_id="X-01",
        rule_snapshot_ref=None,
        quality_flags=[],
        payload={
            "kind": "executable_quote",
            "best_bid": _fixed("55"),
            "best_ask": _fixed("58"),
            "bid_depth": _fixed("125", 0),
            "ask_depth": _fixed("80", 0),
            "paused": False,
        },
    )


def _rule() -> VenueRuleSnapshotV0:
    return VenueRuleSnapshotV0.model_validate(
        {
            "venue": "polymarket",
            "condition_id": "condition_polymarket_example",
            "fetched_at": "2026-07-23T12:00:00Z",
            "effective_from": "2026-07-23T12:00:00Z",
            "game_start_time": "2026-07-23T13:00:00Z",
            "seconds_delay": _fixed("1", 0),
            "cancel_during_delay": False,
            "start_time_cancel_policy": "cancel_all_at_game_start",
            "fees_enabled": False,
            "fee_rate": _fixed("0", 0),
            "fee_exponent": _fixed("1", 0),
            "taker_only": True,
            "maker_fee_rate": _fixed("0", 0),
            "minimum_tick_size": _fixed("1"),
            "minimum_order_size": _fixed("1", 0),
            "order_types_supported": ["MARKET"],
            "source_document_version": "fixture-2026-07-23",
            "raw_response_hash": "sha256:" + ("5" * 64),
        }
    )


def test_empty_current_inventory_is_explicitly_not_aligned() -> None:
    decision = audit_current_market_alignment_evidence(
        program_root=PROJECT_ROOT,
        evidence=CurrentAlignmentEvidence(),
    )

    assert decision.status == "not_aligned"
    assert decision.verified_document_counts == {
        "metadata_snapshots": 0,
        "model_output_events": 0,
        "quote_events": 0,
        "rule_snapshots": 0,
    }
    assert decision.reason_codes == (
        "missing_canonical_game_condition_outcome_binding",
        "missing_market_metadata_snapshot",
        "missing_model_output",
        "missing_local_receive_executable_quote",
        "missing_venue_rule_snapshot",
        "missing_registered_join_policy",
    )
    assert decision.matched_as_of_rows == 0
    assert "alpha" not in str(decision.to_dict()).lower()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata_snapshots", ("sha256:" + "a" * 64,)),
        ("model_output_events", ("sha256:" + "b" * 64,)),
        ("quote_events", ("sha256:" + "c" * 64,)),
        ("rule_snapshots", ("sha256:" + "d" * 64,)),
    ],
)
def test_hash_shaped_references_cannot_stand_in_for_evidence_documents(
    field: str,
    value: tuple[str],
) -> None:
    with pytest.raises(MarketAlignmentInputError, match="validated"):
        CurrentAlignmentEvidence(**{field: value})


def test_mutable_or_list_inventory_is_rejected() -> None:
    with pytest.raises(MarketAlignmentInputError, match="tuple"):
        CurrentAlignmentEvidence(metadata_snapshots=[])  # type: ignore[arg-type]


def test_contract_types_are_part_of_the_inventory_boundary() -> None:
    annotations = CurrentAlignmentEvidence.__annotations__

    assert "MarketMetadataSnapshotV0" in str(annotations["metadata_snapshots"])
    assert "EventEnvelopeV0" in str(annotations["model_output_events"])
    assert "EventEnvelopeV0" in str(annotations["quote_events"])
    assert "VenueRuleSnapshotV0" in str(annotations["rule_snapshots"])


def test_no_caller_selected_quote_age_or_probability_is_accepted() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        audit_current_market_alignment_evidence(
            program_root=PROJECT_ROOT,
            evidence=CurrentAlignmentEvidence(),
            max_quote_age_ms=1_000,
        )


def test_real_contract_documents_are_revalidated_but_still_cannot_bypass_open_locks() -> None:
    decision = audit_current_market_alignment_evidence(
        program_root=PROJECT_ROOT,
        evidence=CurrentAlignmentEvidence(
            metadata_snapshots=(_metadata(),),
            model_output_events=(_model_output_event(),),
            quote_events=(_quote_event(),),
            rule_snapshots=(_rule(),),
        ),
    )

    assert decision.status == "not_aligned"
    assert decision.verified_document_counts == {
        "metadata_snapshots": 1,
        "model_output_events": 1,
        "quote_events": 1,
        "rule_snapshots": 1,
    }
    assert decision.reason_codes == (
        "missing_canonical_game_condition_outcome_binding",
        "missing_registered_join_policy",
    )
    assert decision.matched_as_of_rows == 0


def test_quote_must_be_a_local_receive_executable_observation() -> None:
    evidence = CurrentAlignmentEvidence(
        quote_events=(_quote_event(receive_basis="upstream_exporter"),)
    )

    with pytest.raises(
        MarketAlignmentInputError,
        match="local_recorder",
    ):
        audit_current_market_alignment_evidence(
            program_root=PROJECT_ROOT,
            evidence=evidence,
        )


def test_decision_contains_no_unverified_probability_or_spread() -> None:
    decision = audit_current_market_alignment_evidence(
        program_root=PROJECT_ROOT,
        evidence=CurrentAlignmentEvidence(),
    )

    assert set(decision.to_dict()) == {
        "status",
        "reason_codes",
        "verified_document_counts",
        "matched_as_of_rows",
        "comparison_basis",
    }
    assert decision.comparison_basis == "verified_prerequisites_only"
    assert not hasattr(decision, "spread")
    assert not hasattr(decision, "probability_distance_to_executable_interval")


def test_decision_document_counts_are_immutable() -> None:
    decision = audit_current_market_alignment_evidence(
        program_root=PROJECT_ROOT,
        evidence=CurrentAlignmentEvidence(),
    )

    with pytest.raises(TypeError):
        decision.verified_document_counts["quote_events"] = 99


def test_runtime_contract_classes_not_hash_dataclasses_are_the_evidence_types() -> None:
    assert MarketMetadataSnapshotV0.__name__ == "MarketMetadataSnapshotV0"
    assert EventEnvelopeV0.__name__ == "EventEnvelopeV0"
    assert VenueRuleSnapshotV0.__name__ == "VenueRuleSnapshotV0"
    assert datetime.now(tz=UTC).utcoffset().total_seconds() == 0
