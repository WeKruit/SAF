"""Deterministic PMXT v2 Level-2 order-book reconstruction.

The reconstruction boundary is deliberately narrow: it rebuilds visible
price levels from ``book`` snapshots and ``price_change`` deltas.  PMXT has no
exchange sequence or queue-priority data, so this module never reconstructs or
claims fills, queue position, or exact packet-loss recovery.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from prediction_market.contracts import (
    EventEnvelopeV0,
    FixedPointV0,
    level2_stream_sha256,
    replay_order_key,
)
from prediction_market.pmxt.quality import (
    CROSSED_BOOK,
    DUPLICATE_EVENT,
    MISSING_INITIAL_SNAPSHOT,
    NONPOSITIVE_SIZE,
    OUT_OF_ORDER,
    QualityTracker,
)


_DECIMAL_FIELDS = frozenset(
    {
        "price",
        "size",
        "best_bid",
        "best_ask",
        "old_tick_size",
        "new_tick_size",
    }
)
_TIMESTAMP_FIELDS = frozenset({"timestamp", "timestamp_received"})


class PMXTValidationError(ValueError):
    """Raised when an event cannot be reconstructed without guessing."""


@dataclass(frozen=True)
class ReconstructionResult:
    """Deterministic semantic stream plus auditable quality observations."""

    semantic_events: tuple[EventEnvelopeV0, ...]
    stream_sha256: str
    quality_flags: tuple[str, ...]
    counts: dict[str, int]
    queue_fill_reconstructed: bool = False


@dataclass
class _BookState:
    bids: dict[Decimal, Decimal]
    asks: dict[Decimal, Decimal]
    has_snapshot: bool = False


@dataclass(frozen=True)
class _PreparedEvent:
    canonical: dict[str, Any]
    canonical_bytes: bytes
    sha256: str
    sort_key: tuple[str, str, str, str, str]
    receive_at: datetime
    source_at: datetime
    source_event_id: str


def _plain_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise PMXTValidationError("decimal values must be finite")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"", "-0"}:
        return "0"
    return rendered


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise PMXTValidationError(f"{field} must be a fixed-point decimal")
    if isinstance(value, float):
        raise PMXTValidationError(
            f"{field} contains a binary float; use Decimal or a decimal string"
        )
    if not isinstance(value, (str, int, Decimal)):
        raise PMXTValidationError(f"{field} must be a fixed-point decimal")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise PMXTValidationError(f"{field} is not a valid decimal") from exc
    if not parsed.is_finite():
        raise PMXTValidationError(f"{field} must be finite")
    return parsed


def _utc_datetime(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise PMXTValidationError(f"{field} is not an ISO-8601 timestamp") from exc
    else:
        raise PMXTValidationError(f"{field} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PMXTValidationError(f"{field} must include a timezone")
    utc = parsed.astimezone(timezone.utc)
    if utc.microsecond % 1_000:
        raise PMXTValidationError(f"{field} must have millisecond precision")
    return utc


def _utc_text(value: object, *, field: str) -> str:
    return (
        _utc_datetime(value, field=field)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _text_identifier(value: object, *, field: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PMXTValidationError(f"{field} bytes are not UTF-8") from exc
    if not isinstance(value, str) or not value:
        raise PMXTValidationError(f"{field} must be a non-empty string")
    return value


def _reject_binary_floats(value: object, *, field: str = "event") -> None:
    if isinstance(value, float):
        raise PMXTValidationError(
            f"{field} contains a binary float; use Decimal or a decimal string"
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_binary_floats(child, field=f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_binary_floats(child, field=f"{field}[{index}]")


def _book_levels(value: object, *, field: str) -> list[dict[str, str]]:
    if value is None:
        raise PMXTValidationError(f"{field} is required for book snapshots")
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise PMXTValidationError(f"{field} is not valid JSON") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PMXTValidationError(f"{field} must be a list of price levels")
    levels: list[dict[str, str]] = []
    for index, level in enumerate(value):
        if (
            not isinstance(level, Sequence)
            or isinstance(level, (str, bytes, bytearray))
            or len(level) != 2
        ):
            raise PMXTValidationError(
                f"{field}[{index}] must be PMXT's [price, size] pair"
            )
        price = _decimal(level[0], field=f"{field}[{index}].price")
        size = _decimal(level[1], field=f"{field}[{index}].size")
        levels.append({"price": _plain_decimal(price), "size": _plain_decimal(size)})
    return levels


def _normalize_generic(value: object, *, field: str) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise PMXTValidationError(
            f"{field} contains a binary float; use Decimal or a decimal string"
        )
    if isinstance(value, Decimal):
        return _plain_decimal(value)
    if isinstance(value, datetime):
        return _utc_text(value, field=field)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise PMXTValidationError(f"{field} has a non-string key")
            normalized[key] = _normalize_generic(value[key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_generic(child, field=f"{field}[{index}]")
            for index, child in enumerate(value)
        ]
    raise PMXTValidationError(f"{field} has unsupported type {type(value).__name__}")


def canonicalize_event(event: Mapping[str, object]) -> dict[str, Any]:
    """Return the canonical, JSON-safe form used for hashing and ordering."""

    _reject_binary_floats(event)
    required = {"timestamp_received", "timestamp", "market", "event_type", "asset_id"}
    missing = required - event.keys()
    if missing:
        raise PMXTValidationError(
            f"event is missing required fields: {', '.join(sorted(missing))}"
        )

    normalized: dict[str, Any] = {}
    for key in sorted(event):
        value = event[key]
        if key in _TIMESTAMP_FIELDS:
            normalized[key] = _utc_text(value, field=key)
        elif key in {"market", "asset_id", "event_type"}:
            normalized[key] = _text_identifier(value, field=key)
        elif key in {"bids", "asks"} and value is not None:
            normalized[key] = _book_levels(value, field=key)
        elif key in _DECIMAL_FIELDS and value is not None:
            normalized[key] = _plain_decimal(_decimal(value, field=key))
        else:
            normalized[key] = _normalize_generic(value, field=key)
    normalized["event_type"] = normalized["event_type"].lower()
    if "side" in normalized and normalized["side"] is not None:
        if not isinstance(normalized["side"], str):
            raise PMXTValidationError("side must be a string")
        normalized["side"] = normalized["side"].upper()
    return normalized


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_event_sha256(event: Mapping[str, object]) -> str:
    """Hash one PMXT event after schema-aware canonicalization."""

    return f"sha256:{hashlib.sha256(_canonical_bytes(canonicalize_event(event))).hexdigest()}"


def canonical_event_sort_key(
    event: Mapping[str, object],
) -> tuple[str, str, str, str, str]:
    """Return Charter §8.2's complete global replay key."""

    canonical = canonicalize_event(event)
    payload_hash = f"sha256:{hashlib.sha256(_canonical_bytes(canonical)).hexdigest()}"
    return (
        canonical["timestamp_received"],
        canonical["timestamp"],
        canonical["market"],
        canonical["asset_id"],
        payload_hash,
    )


def _native_references(event: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "namespace": "polymarket.asset_id",
            "native_id": event["asset_id"],
        },
        {
            "namespace": "polymarket.condition_id",
            "native_id": event["market"],
        },
    ]


def _empty_canonical_references() -> dict[str, object]:
    return {
        "competition_id": None,
        "game_id": None,
        "participant_ids": [],
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }


def _source_event_envelope(
    canonical: Mapping[str, Any],
    *,
    source_event_sha256: str,
) -> EventEnvelopeV0:
    digest_hex = source_event_sha256.removeprefix("sha256:")
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="raw_observation",
        payload_schema_version="v0",
        source={
            "system": "pmxt-v2",
            "stream": "canonical-orderbook-row",
            "venue": "polymarket",
            "sequence": source_event_sha256,
            "capture_session_id": f"pmxt-canonical-row-{digest_hex}",
            "record_ordinal": 0,
        },
        time={
            "receive_at": canonical["timestamp_received"],
            "receive_basis": "upstream_exporter",
            "source_at": canonical["timestamp"],
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs=_empty_canonical_references(),
        native_refs=_native_references(canonical),
        lineage={
            "raw_object_hash": source_event_sha256,
            "raw_record_ordinal": 0,
        },
        experiment_id=None,
        rule_snapshot_ref=None,
        quality_flags=[],
        payload={"canonical_pmxt_event": canonical},
    )


def _prepare(event: Mapping[str, object]) -> _PreparedEvent:
    canonical = canonicalize_event(event)
    payload = _canonical_bytes(canonical)
    sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    sort_key = (
        canonical["timestamp_received"],
        canonical["timestamp"],
        canonical["market"],
        canonical["asset_id"],
        sha256,
    )
    return _PreparedEvent(
        canonical=canonical,
        canonical_bytes=payload,
        sha256=sha256,
        sort_key=sort_key,
        receive_at=_utc_datetime(
            canonical["timestamp_received"], field="timestamp_received"
        ),
        source_at=_utc_datetime(canonical["timestamp"], field="timestamp"),
        source_event_id=_source_event_envelope(
            canonical,
            source_event_sha256=sha256,
        ).event_id,
    )


def _load_events(
    source: str | Path | Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".jsonl":
            events: list[Mapping[str, object]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line, parse_float=Decimal)
                    except json.JSONDecodeError as exc:
                        raise PMXTValidationError(
                            f"{path}:{line_number} is not valid JSON"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise PMXTValidationError(
                            f"{path}:{line_number} must contain a JSON object"
                        )
                    events.append(value)
            return events
        if path.suffix.lower() == ".parquet":
            from prediction_market.pmxt.archive import read_parquet_events

            return read_parquet_events([path])
        raise PMXTValidationError(f"unsupported PMXT source extension: {path.suffix}")

    events = list(source)
    if not all(isinstance(event, Mapping) for event in events):
        raise PMXTValidationError("PMXT event iterables may contain mappings only")
    return events


def _apply_snapshot(
    event: Mapping[str, Any],
    state: _BookState,
    tracker: QualityTracker,
    local_flags: set[str],
) -> None:
    state.bids.clear()
    state.asks.clear()
    state.has_snapshot = True
    for field, levels in (("bids", event.get("bids")), ("asks", event.get("asks"))):
        if not isinstance(levels, list):
            raise PMXTValidationError(f"{field} is required for book snapshots")
        target = state.bids if field == "bids" else state.asks
        for level in levels:
            price = _decimal(level["price"], field=f"{field}.price")
            size = _decimal(level["size"], field=f"{field}.size")
            if size <= 0:
                tracker.mark(NONPOSITIVE_SIZE, "nonpositive_sizes")
                local_flags.add(NONPOSITIVE_SIZE)
                continue
            target[price] = size


def _apply_delta(
    event: Mapping[str, Any],
    state: _BookState,
    tracker: QualityTracker,
    local_flags: set[str],
) -> None:
    if not state.has_snapshot:
        tracker.mark(MISSING_INITIAL_SNAPSHOT, "missing_initial_snapshots")
        local_flags.add(MISSING_INITIAL_SNAPSHOT)
    if event.get("price") is None or event.get("size") is None:
        raise PMXTValidationError("price_change requires price and size")
    price = _decimal(event["price"], field="price")
    size = _decimal(event["size"], field="size")
    side = event.get("side")
    if side == "BUY":
        target = state.bids
    elif side == "SELL":
        target = state.asks
    else:
        raise PMXTValidationError("price_change side must be BUY or SELL")
    if size < 0:
        tracker.mark(NONPOSITIVE_SIZE, "nonpositive_sizes")
        local_flags.add(NONPOSITIVE_SIZE)
        return
    if size == 0:
        target.pop(price, None)
        return
    target[price] = size


def _levels_for_output(
    levels: Mapping[Decimal, Decimal], *, reverse: bool
) -> list[dict[str, dict[str, object]]]:
    return [
        {
            "price": FixedPointV0.from_value(price).model_dump(mode="python"),
            "size": FixedPointV0.from_value(levels[price]).model_dump(mode="python"),
        }
        for price in sorted(levels, reverse=reverse)
    ]


def _semantic_event(
    event: _PreparedEvent,
    state: _BookState,
    local_flags: set[str],
) -> EventEnvelopeV0:
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": "pmxt-l2-reconstructor",
            "stream": "visible-orderbook-state",
            "venue": "polymarket",
            "sequence": event.sha256,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time={
            "receive_at": event.canonical["timestamp_received"],
            "receive_basis": "upstream_exporter",
            "source_at": event.canonical["timestamp"],
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs=_empty_canonical_references(),
        native_refs=_native_references(event.canonical),
        lineage={"parent_event_ids": [event.source_event_id]},
        experiment_id="X-01",
        rule_snapshot_ref=None,
        quality_flags=sorted(local_flags),
        payload={
            "native_market": event.canonical["market"],
            "native_asset_id": event.canonical["asset_id"],
            "source_event_type": event.canonical["event_type"],
            "source_event_sha256": event.sha256,
            "source_event_id": event.source_event_id,
            "bids": _levels_for_output(state.bids, reverse=True),
            "asks": _levels_for_output(state.asks, reverse=False),
        },
    )


def reconstruct(
    source: str | Path | Iterable[Mapping[str, object]],
    *,
    gap_threshold_ms: int = 60_000,
) -> ReconstructionResult:
    """Reconstruct a canonical Level-2 semantic event stream.

    Events are globally ordered by receive time, source time, market, asset,
    then canonical payload hash.  Exact canonical duplicates are removed.
    ``gap_threshold_ms`` is a receive-time diagnostic only because PMXT has no
    source sequence number.
    """

    if isinstance(gap_threshold_ms, bool) or gap_threshold_ms <= 0:
        raise PMXTValidationError("gap_threshold_ms must be a positive integer")

    raw_events = _load_events(source)
    prepared = [_prepare(event) for event in raw_events]
    tracker = QualityTracker()
    counts = tracker.counts
    counts.update(
        {
            "input_events": len(prepared),
            "unique_events": 0,
            "duplicate_events": 0,
            "out_of_order_events": 0,
            "gap_candidates": 0,
            "book_events": 0,
            "price_change_events": 0,
            "semantic_events": 0,
            "crossed_books": 0,
            "nonpositive_sizes": 0,
            "missing_initial_snapshots": 0,
        }
    )

    unique_by_hash: dict[str, _PreparedEvent] = {}
    duplicate_hashes: set[str] = set()
    for event in prepared:
        existing = unique_by_hash.get(event.sha256)
        if existing is not None:
            if existing.canonical_bytes != event.canonical_bytes:
                raise PMXTValidationError("canonical SHA-256 collision detected")
            tracker.mark(DUPLICATE_EVENT, "duplicate_events")
            duplicate_hashes.add(event.sha256)
            continue
        unique_by_hash[event.sha256] = event

    ordered = sorted(unique_by_hash.values(), key=lambda event: event.sort_key)
    counts["unique_events"] = len(ordered)
    states: dict[tuple[str, str], _BookState] = {}
    previous_receive: dict[tuple[str, str], datetime] = {}
    previous_source: dict[tuple[str, str], datetime] = {}
    semantic_events: list[EventEnvelopeV0] = []

    for event in ordered:
        canonical = event.canonical
        stream = (canonical["market"], canonical["asset_id"])
        state = states.setdefault(stream, _BookState(bids={}, asks={}))
        local_flags: set[str] = set()
        if event.sha256 in duplicate_hashes:
            local_flags.add(DUPLICATE_EVENT)

        prior_source = previous_source.get(stream)
        if prior_source is not None and event.source_at < prior_source:
            tracker.mark(OUT_OF_ORDER, "out_of_order_events")
            local_flags.add(OUT_OF_ORDER)
        previous_source[stream] = event.source_at

        prior = previous_receive.get(stream)
        if prior is not None:
            delta_ms = int((event.receive_at - prior).total_seconds() * 1_000)
            if delta_ms > gap_threshold_ms:
                tracker.counts["gap_candidates"] += 1
        previous_receive[stream] = event.receive_at

        event_type = canonical["event_type"]
        if event_type == "book":
            counts["book_events"] += 1
            _apply_snapshot(canonical, state, tracker, local_flags)
        elif event_type == "price_change":
            counts["price_change_events"] += 1
            _apply_delta(canonical, state, tracker, local_flags)
        else:
            continue

        if state.bids and state.asks and max(state.bids) >= min(state.asks):
            tracker.mark(CROSSED_BOOK, "crossed_books")
            local_flags.add(CROSSED_BOOK)
        semantic_events.append(_semantic_event(event, state, local_flags))

    ordered_semantic_events = tuple(sorted(semantic_events, key=replay_order_key))
    counts["semantic_events"] = len(ordered_semantic_events)
    stream_sha256 = level2_stream_sha256(ordered_semantic_events)
    return ReconstructionResult(
        semantic_events=ordered_semantic_events,
        stream_sha256=stream_sha256,
        quality_flags=tracker.sorted_flags(),
        counts=dict(sorted(counts.items())),
    )
