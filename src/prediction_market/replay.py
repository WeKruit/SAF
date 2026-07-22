"""Deterministic replay summaries and canonical event logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prediction_market.contracts import (
    EventEnvelopeV0,
    canonical_json,
    canonical_json_bytes,
    level2_stream_sha256,
    replay_order_key,
)


class ReplayInputError(ValueError):
    """The supplied stream cannot be replayed as validated v0 envelopes."""


class ReplayDeterminismError(RuntimeError):
    """Two replay runs diverged at a required determinism level."""


@dataclass(frozen=True, slots=True)
class Level1SemanticSummaryV0:
    event_count: int
    event_ids: tuple[str, ...]
    event_types: tuple[str, ...]
    orders: tuple[str, ...]
    fills: tuple[str, ...]
    pnl: tuple[str, ...]
    terminal_pnl: str | None


@dataclass(frozen=True, slots=True)
class ReplayRunV0:
    events: tuple[EventEnvelopeV0, ...]
    orders: tuple[EventEnvelopeV0, ...]
    fills: tuple[EventEnvelopeV0, ...]
    pnl_events: tuple[EventEnvelopeV0, ...]
    semantic_summary: Level1SemanticSummaryV0
    stream_sha256: str
    canonical_log: bytes


def _validated_snapshot(
    events: Iterable[EventEnvelopeV0],
) -> tuple[EventEnvelopeV0, ...]:
    snapshot = tuple(events)
    validated: list[EventEnvelopeV0] = []
    for event in snapshot:
        if not isinstance(event, EventEnvelopeV0):
            raise ReplayInputError("replay events must be validated EventEnvelopeV0 values")
        try:
            validated.append(
                EventEnvelopeV0.model_validate(
                    event.model_dump(mode="python", round_trip=True)
                )
            )
        except (TypeError, ValueError) as exc:
            raise ReplayInputError("invalid EventEnvelopeV0 in replay stream") from exc
    return tuple(validated)


def build_replay(events: Iterable[EventEnvelopeV0]) -> ReplayRunV0:
    """Validate, order, summarize, and canonically serialize one event stream."""

    ordered = tuple(sorted(_validated_snapshot(events), key=replay_order_key))
    orders = tuple(event for event in ordered if event.event_type == "simulated_order")
    fills = tuple(event for event in ordered if event.event_type == "simulated_fill")
    pnl_events = tuple(event for event in ordered if event.event_type == "simulated_pnl")
    order_payloads = tuple(canonical_json(event.payload) for event in orders)
    fill_payloads = tuple(canonical_json(event.payload) for event in fills)
    pnl_payloads = tuple(canonical_json(event.payload) for event in pnl_events)
    summary = Level1SemanticSummaryV0(
        event_count=len(ordered),
        event_ids=tuple(event.event_id for event in ordered),
        event_types=tuple(event.event_type for event in ordered),
        orders=order_payloads,
        fills=fill_payloads,
        pnl=pnl_payloads,
        terminal_pnl=pnl_payloads[-1] if pnl_payloads else None,
    )
    canonical_log = (
        b"\n".join(canonical_json_bytes(event) for event in ordered) + b"\n"
        if ordered
        else b""
    )
    return ReplayRunV0(
        events=ordered,
        orders=orders,
        fills=fills,
        pnl_events=pnl_events,
        semantic_summary=summary,
        stream_sha256=level2_stream_sha256(ordered),
        canonical_log=canonical_log,
    )


def assert_replay_deterministic(first: ReplayRunV0, second: ReplayRunV0) -> None:
    """Fail closed unless both mandatory replay levels are identical."""

    if not isinstance(first, ReplayRunV0) or not isinstance(second, ReplayRunV0):
        raise TypeError("replay comparison requires ReplayRunV0 values")
    if first.semantic_summary != second.semantic_summary:
        raise ReplayDeterminismError("Level 1 semantic replay mismatch")
    if first.stream_sha256 != second.stream_sha256:
        raise ReplayDeterminismError("Level 2 canonical stream hash mismatch")


__all__ = [
    "Level1SemanticSummaryV0",
    "ReplayDeterminismError",
    "ReplayInputError",
    "ReplayRunV0",
    "assert_replay_deterministic",
    "build_replay",
]
