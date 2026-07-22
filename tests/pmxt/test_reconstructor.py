from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import pytest


def _canonical_stream_hash(events: tuple[dict[str, object], ...]) -> str:
    lines = [
        json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for event in events
    ]
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_reconstruction_is_level_2_deterministic(pmxt_fixture):
    from prediction_market.pmxt.reconstructor import reconstruct

    first = reconstruct(pmxt_fixture)
    second = reconstruct(pmxt_fixture)

    assert first.semantic_events == second.semantic_events
    assert first.stream_sha256 == second.stream_sha256
    assert first.stream_sha256 == _canonical_stream_hash(first.semantic_events)
    assert first.counts["input_events"] == 6
    assert first.counts["duplicate_events"] == 1
    assert first.counts["out_of_order_events"] == 1
    assert first.counts["semantic_events"] == 5
    assert {event["event_type"] for event in first.semantic_events} == {
        "book",
        "price_change",
    }
    assert first.queue_fill_reconstructed is False


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
        event for event in result.semantic_events if event["asset_id"] == "asset-a"
    ][-1]

    assert final_asset_a["bids"] == [
        {"price": "0.46", "size": "25"},
        {"price": "0.45", "size": "100"},
    ]
    assert final_asset_a["asks"] == [{"price": "0.54", "size": "30"}]
    assert "NONPOSITIVE_SIZE" not in result.quality_flags
    assert all(not isinstance(level["price"], float) for level in final_asset_a["bids"])


def test_reconstructor_flags_crossed_and_nonpositive_books(anomalous_fixture):
    from prediction_market.pmxt.reconstructor import reconstruct

    result = reconstruct(anomalous_fixture)

    assert {
        "CROSSED_BOOK",
        "MISSING_INITIAL_SNAPSHOT",
        "NONPOSITIVE_SIZE",
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

    assert "RECEIVE_GAP_CANDIDATE" in result.quality_flags
    assert "GAP_DETECTED" not in result.quality_flags
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
