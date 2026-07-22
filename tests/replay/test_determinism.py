from __future__ import annotations

from dataclasses import replace

import pytest

from prediction_market.contracts import (
    EventEnvelopeV0,
    canonical_json,
    canonical_json_bytes,
)
from prediction_market.replay import (
    ReplayDeterminismError,
    ReplayInputError,
    assert_replay_deterministic,
    build_replay,
)


def _event_id(fill: str) -> str:
    return "evt_" + fill * 64


def _sha256(fill: str) -> str:
    return "sha256:" + fill * 64


def _event(
    event_type: str,
    receive_at: str,
    payload: dict[str, object],
) -> EventEnvelopeV0:
    simulated = event_type.startswith("simulated_")
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type=event_type,
        payload_schema_version="v0",
        source={
            "system": "x09-test",
            "stream": event_type,
            "venue": "polymarket",
            "sequence": None,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time={
            "receive_at": receive_at,
            "receive_basis": "local_recorder",
            "source_at": "2026-07-22T12:00:00Z",
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs={
            "competition_id": "cmp_nba",
            "game_id": "game_nba_x09",
            "participant_ids": ["participant_away", "participant_home"],
            "venue_event_id": "venue_event_x09",
            "market_id": "market_x09",
            "outcome_id": "outcome_home",
            "condition_id": "condition_x09",
        },
        native_refs=[],
        lineage={"parent_event_ids": [_event_id("a")]},
        experiment_id="X-09",
        rule_snapshot_ref=_sha256("b") if simulated else None,
        quality_flags=[],
        payload=payload,
    )


def _stream() -> tuple[EventEnvelopeV0, ...]:
    return (
        _event("signal", "2026-07-22T12:00:01Z", {"action": "BUY"}),
        _event(
            "simulated_order",
            "2026-07-22T12:00:02Z",
            {"order_id": "order-x09", "quantity": {"atoms": "5", "scale": 0}},
        ),
        _event(
            "simulated_fill",
            "2026-07-22T12:00:03Z",
            {"order_id": "order-x09", "vwap": {"atoms": "52", "scale": 2}},
        ),
        _event(
            "simulated_pnl",
            "2026-07-22T12:00:04Z",
            {"order_id": "order-x09", "pnl": {"atoms": "1", "scale": 2}},
        ),
    )


def test_replay_sorts_validated_events_and_canonicalizes_the_complete_log() -> None:
    events = _stream()

    forward = build_replay(events)
    reverse = build_replay(reversed(events))

    assert forward.events == events
    assert reverse.events == events
    assert forward.semantic_summary == reverse.semantic_summary
    assert forward.stream_sha256 == reverse.stream_sha256
    assert forward.canonical_log == reverse.canonical_log
    assert forward.canonical_log == (
        b"\n".join(canonical_json_bytes(event) for event in events) + b"\n"
    )


def test_level1_summary_covers_order_fill_pnl_and_terminal_state() -> None:
    run = build_replay(_stream())

    assert run.semantic_summary.event_count == 4
    assert run.semantic_summary.event_ids == tuple(
        event.event_id for event in run.events
    )
    assert run.semantic_summary.event_types == (
        "signal",
        "simulated_order",
        "simulated_fill",
        "simulated_pnl",
    )
    assert run.semantic_summary.orders == (canonical_json(run.orders[0].payload),)
    assert run.semantic_summary.fills == (canonical_json(run.fills[0].payload),)
    assert run.semantic_summary.pnl == (canonical_json(run.pnl_events[0].payload),)
    assert run.semantic_summary.terminal_pnl == canonical_json(
        run.pnl_events[-1].payload
    )


def test_replay_rejects_values_that_are_not_validated_envelopes() -> None:
    with pytest.raises(ReplayInputError, match="EventEnvelopeV0"):
        build_replay([{"event_type": "signal"}])  # type: ignore[list-item]


def test_replay_comparison_fails_closed_at_each_required_level() -> None:
    first = build_replay(_stream())
    changed_pnl = _event(
        "simulated_pnl",
        "2026-07-22T12:00:04Z",
        {"pnl": {"atoms": "2", "scale": 2}},
    )
    changed = build_replay(
        (*_stream()[:-1], changed_pnl)
    )

    with pytest.raises(ReplayDeterminismError, match="Level 1"):
        assert_replay_deterministic(first, changed)

    hash_only_change = replace(first, stream_sha256=_sha256("f"))
    with pytest.raises(ReplayDeterminismError, match="Level 2"):
        assert_replay_deterministic(first, hash_only_change)
