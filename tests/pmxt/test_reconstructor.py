from __future__ import annotations

import json
from decimal import Decimal

import pytest


def _fixed_decimal(value: object) -> Decimal:
    from prediction_market.contracts import FixedPointV0

    return FixedPointV0.model_validate(dict(value)).to_decimal()


def test_reconstruction_is_level_2_deterministic(pmxt_fixture):
    from prediction_market.contracts import (
        EventEnvelopeV0,
        level2_stream_sha256,
        validate_contract_v0,
    )
    from prediction_market.pmxt.reconstructor import reconstruct

    first = reconstruct(pmxt_fixture)
    second = reconstruct(pmxt_fixture)

    assert first.semantic_events == second.semantic_events
    assert first.stream_sha256 == second.stream_sha256
    assert first.stream_sha256 == (
        "sha256:6f2874def23b05d9d450f1d7858a51c8e2c9d1a8a8c6fea89a92ecb9a8289d4e"
    )
    assert all(isinstance(event, EventEnvelopeV0) for event in first.semantic_events)
    assert first.stream_sha256 == level2_stream_sha256(first.semantic_events)
    assert first.counts["input_events"] == 6
    assert first.counts["duplicate_events"] == 1
    assert first.counts["out_of_order_events"] == 1
    assert first.counts["semantic_events"] == 5
    assert {event.payload["source_event_type"] for event in first.semantic_events} == {
        "book",
        "price_change",
    }
    assert {event.event_type for event in first.semantic_events} == {
        "normalized_observation"
    }
    assert all(event.lineage.parent_event_ids for event in first.semantic_events)
    assert all(
        EventEnvelopeV0.model_validate(event.model_dump(mode="python", round_trip=True))
        == event
        for event in first.semantic_events
    )
    assert all(
        validate_contract_v0("quality-flags/v0.yaml", flag) == flag
        for flag in first.quality_flags
    )
    assert first.quality_flags == ("duplicate_event", "out_of_order")
    assert any(
        "duplicate_event" in event.quality_flags for event in first.semantic_events
    )
    assert first.queue_fill_reconstructed is False


def test_reconstructor_calls_the_accepted_level_2_framer(pmxt_fixture, monkeypatch):
    from prediction_market.pmxt import reconstructor

    framed: list[tuple[object, ...]] = []
    framed_hash = "sha256:" + "f" * 64

    def fake_level2(events):
        framed.append(tuple(events))
        return framed_hash

    monkeypatch.setattr(
        reconstructor,
        "level2_stream_sha256",
        fake_level2,
        raising=False,
    )

    result = reconstructor.reconstruct(pmxt_fixture)

    assert framed == [result.semantic_events]
    assert result.stream_sha256 == framed_hash


def test_reconstruction_does_not_depend_on_input_enumeration_order(pmxt_fixture):
    from prediction_market.pmxt.reconstructor import reconstruct

    events = [
        json.loads(line)
        for line in pmxt_fixture.read_text(encoding="utf-8").splitlines()
        if line
    ]

    forward = reconstruct(events)
    reverse = reconstruct(reversed(events))

    assert forward.semantic_events == reverse.semantic_events
    assert forward.stream_sha256 == reverse.stream_sha256
    assert forward.quality_flags == reverse.quality_flags
    assert forward.counts == reverse.counts


def test_reconstructor_applies_fixed_point_book_updates(pmxt_fixture):
    from prediction_market.pmxt.reconstructor import reconstruct

    result = reconstruct(pmxt_fixture)
    final_asset_a = [
        event
        for event in result.semantic_events
        if event.payload["native_asset_id"] == "asset-a"
    ][-1]

    assert [
        (_fixed_decimal(level["price"]), _fixed_decimal(level["size"]))
        for level in final_asset_a.payload["bids"]
    ] == [(Decimal("0.46"), Decimal("25")), (Decimal("0.45"), Decimal("100"))]
    assert [
        (_fixed_decimal(level["price"]), _fixed_decimal(level["size"]))
        for level in final_asset_a.payload["asks"]
    ] == [(Decimal("0.54"), Decimal("30"))]
    assert "non_positive_size" not in result.quality_flags
    assert all(
        isinstance(level["price"], dict)
        for level in final_asset_a.model_dump(mode="python")["payload"]["bids"]
    )


def test_reconstructor_flags_crossed_and_nonpositive_books(anomalous_fixture):
    from prediction_market.pmxt.reconstructor import reconstruct

    result = reconstruct(anomalous_fixture)

    assert {
        "crossed_book",
        "missing_initial_snapshot",
        "non_positive_size",
    } <= set(result.quality_flags)
    assert result.counts["crossed_books"] == 1
    assert result.counts["missing_initial_snapshots"] == 1
    assert result.counts["nonpositive_sizes"] == 2


def test_reconstructor_marks_receive_gap_as_candidate_not_confirmed_loss():
    from prediction_market.pmxt.reconstructor import reconstruct

    events = [
        {
            "timestamp_received": "2026-06-01T00:00:00.000Z",
            "timestamp": "2026-06-01T00:00:00.000Z",
            "market": "0xgap",
            "event_type": "book",
            "asset_id": "asset-gap",
            "bids": "[]",
            "asks": "[]",
        },
        {
            "timestamp_received": "2026-06-01T00:00:05.000Z",
            "timestamp": "2026-06-01T00:00:04.900Z",
            "market": "0xgap",
            "event_type": "price_change",
            "asset_id": "asset-gap",
            "price": "0.5000",
            "size": "2.000000",
            "side": "BUY",
        },
    ]

    result = reconstruct(events, gap_threshold_ms=1_000)

    assert "gap_detected" not in result.quality_flags
    assert result.counts["gap_candidates"] == 1


def test_reconstructor_uses_payload_hash_as_final_global_tie_breaker():
    from prediction_market.pmxt.reconstructor import reconstruct

    snapshot = {
        "timestamp_received": "2026-06-01T00:00:00.000Z",
        "timestamp": "2026-06-01T00:00:00.000Z",
        "market": "0xtie",
        "event_type": "book",
        "asset_id": "asset-tie",
        "bids": "[]",
        "asks": "[]",
    }
    first_delta = {
        "timestamp_received": "2026-06-01T00:00:00.100Z",
        "timestamp": "2026-06-01T00:00:00.090Z",
        "market": "0xtie",
        "event_type": "price_change",
        "asset_id": "asset-tie",
        "price": "0.5000",
        "size": "1.000000",
        "side": "BUY",
    }
    second_delta = {**first_delta, "size": "2.000000"}

    forward = reconstruct([snapshot, first_delta, second_delta])
    reversed_ties = reconstruct([snapshot, second_delta, first_delta])

    assert forward.semantic_events == reversed_ties.semantic_events
    assert forward.stream_sha256 == reversed_ties.stream_sha256


def test_reconstructor_rejects_binary_floats():
    from prediction_market.pmxt.reconstructor import PMXTValidationError, reconstruct

    event = {
        "timestamp_received": "2026-06-01T00:00:00.000Z",
        "timestamp": "2026-06-01T00:00:00.000Z",
        "market": "0xfloat",
        "event_type": "price_change",
        "asset_id": "asset-float",
        "price": 0.5,
        "size": Decimal("1.0"),
        "side": "BUY",
    }

    with pytest.raises(PMXTValidationError, match="binary float"):
        reconstruct([event])
