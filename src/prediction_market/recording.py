"""Raw-first venue recording and fail-closed venue-rule snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from prediction_market.adapters.base import (
    CaptureCounters,
    CaptureResult,
    ProtocolError,
    SequenceTracker,
    WebSocketTransport,
    exact_frame_bytes,
    utc_now_text,
)
from prediction_market.adapters.kalshi import (
    build_orderbook_subscription,
    parse_orderbook_frame,
)
from prediction_market.adapters.polymarket import (
    build_market_subscription,
    parse_market_frame,
)
from prediction_market.raw_store import (
    PartitionBoundaryError,
    RawSegmentWriter,
    SegmentManifest,
)


Clock = Callable[[], str]
Sleep = Callable[[float], Awaitable[None]]
RuleParser = Callable[[bytes, Path], Mapping[str, Any]]
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)


def _positive_frame_limit(max_frames: int) -> int:
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    return max_frames


async def _receive(
    websocket: WebSocketTransport,
    timeout_seconds: float | None,
) -> str | bytes:
    if timeout_seconds is None:
        return await websocket.recv()
    if timeout_seconds <= 0:
        raise ValueError("receive_timeout_seconds must be positive")
    return await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)


def _terminal_reason(exc: Exception) -> str:
    detail = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


async def _record_connection(
    websocket: WebSocketTransport,
    *,
    subscription: str,
    raw_root: str | Path,
    source: str,
    stream: str,
    max_frames: int,
    parser: Callable[[bytes], Any],
    sequence_tracker: SequenceTracker | None,
    receive_timeout_seconds: float | None,
    clock: Clock,
) -> CaptureResult:
    limit = _positive_frame_limit(max_frames)
    writer: RawSegmentWriter | None = None
    sealed_manifests: list[SegmentManifest] = []
    starting_gaps = sequence_tracker.gaps if sequence_tracker is not None else 0
    starting_out_of_order = (
        sequence_tracker.out_of_order if sequence_tracker is not None else 0
    )
    frames = 0
    parse_errors = 0
    complete = False
    reason = "capture did not start"
    try:
        try:
            await websocket.send(subscription)
        except Exception as exc:
            reason = _terminal_reason(exc)
        else:
            while frames < limit:
                try:
                    frame = await _receive(websocket, receive_timeout_seconds)
                except Exception as exc:
                    reason = _terminal_reason(exc)
                    break

                payload = exact_frame_bytes(frame)
                receive_at = clock()
                if writer is None:
                    writer = RawSegmentWriter(raw_root, source=source, stream=stream)
                try:
                    writer.append(payload, receive_at=receive_at)
                except PartitionBoundaryError:
                    sealed_manifests.append(writer.seal())
                    writer = RawSegmentWriter(raw_root, source=source, stream=stream)
                    writer.append(payload, receive_at=receive_at)
                frames += 1
                try:
                    parsed = parser(payload)
                    if sequence_tracker is not None and parsed is not None:
                        sequence_tracker.observe(parsed["sid"], parsed["seq"])
                except (KeyError, ProtocolError, TypeError, ValueError):
                    parse_errors += 1
            else:
                complete = True
                reason = "max_frames reached"
    finally:
        if writer is not None:
            sealed_manifests.append(writer.seal())
        manifests = tuple(sealed_manifests)

    return CaptureResult(
        manifests=manifests,
        counters=CaptureCounters(
            frames=frames,
            parse_errors=parse_errors,
            gaps=(
                sequence_tracker.gaps - starting_gaps
                if sequence_tracker is not None
                else 0
            ),
            out_of_order=(
                sequence_tracker.out_of_order - starting_out_of_order
                if sequence_tracker is not None
                else 0
            ),
        ),
        complete=complete,
        terminal_reason=reason,
    )


async def record_polymarket(
    websocket: WebSocketTransport,
    asset_ids: Sequence[str],
    raw_root: str | Path,
    *,
    max_frames: int,
    receive_timeout_seconds: float | None = None,
    clock: Clock = utc_now_text,
) -> CaptureResult:
    """Capture one public connection, persisting every frame before parsing."""

    return await _record_connection(
        websocket,
        subscription=build_market_subscription(asset_ids),
        raw_root=raw_root,
        source="polymarket",
        stream="market",
        max_frames=max_frames,
        parser=parse_market_frame,
        sequence_tracker=None,
        receive_timeout_seconds=receive_timeout_seconds,
        clock=clock,
    )


async def record_kalshi(
    websocket: WebSocketTransport,
    market_tickers: Sequence[str],
    raw_root: str | Path,
    *,
    max_frames: int,
    receive_timeout_seconds: float | None = None,
    clock: Clock = utc_now_text,
) -> CaptureResult:
    """Capture an already-authenticated Kalshi orderbook connection raw-first."""

    return await _record_kalshi_connection(
        websocket,
        market_tickers,
        raw_root,
        max_frames=max_frames,
        receive_timeout_seconds=receive_timeout_seconds,
        clock=clock,
        sequence_tracker=SequenceTracker(),
    )


async def _record_kalshi_connection(
    websocket: WebSocketTransport,
    market_tickers: Sequence[str],
    raw_root: str | Path,
    *,
    max_frames: int,
    receive_timeout_seconds: float | None,
    clock: Clock,
    sequence_tracker: SequenceTracker,
) -> CaptureResult:
    return await _record_connection(
        websocket,
        subscription=build_orderbook_subscription(market_tickers),
        raw_root=raw_root,
        source="kalshi",
        stream="orderbook",
        max_frames=max_frames,
        parser=_parse_kalshi_capture_frame,
        sequence_tracker=sequence_tracker,
        receive_timeout_seconds=receive_timeout_seconds,
        clock=clock,
    )


def _parse_kalshi_capture_frame(payload: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return parse_orderbook_frame(payload)
    if type(value) is dict and value.get("type") == "subscribed":
        if type(value.get("id")) is not int or type(value.get("msg")) is not dict:
            raise ProtocolError("Kalshi subscribed acknowledgement is malformed")
        return None
    return parse_orderbook_frame(payload)


def _validated_backoff(
    max_reconnects: int, backoff_seconds: Sequence[float]
) -> tuple[float, ...]:
    if type(max_reconnects) is not int or max_reconnects < 0:
        raise ValueError("max_reconnects must be a non-negative integer")
    delays = tuple(backoff_seconds)
    if len(delays) < max_reconnects:
        raise ValueError("backoff_seconds must bound every reconnect")
    if any(type(delay) not in {int, float} or delay < 0 for delay in delays):
        raise ValueError("backoff_seconds must be non-negative numbers")
    return delays


async def record_polymarket_with_reconnect(
    connect: Callable[[], Any],
    asset_ids: Sequence[str],
    raw_root: str | Path,
    *,
    max_frames: int,
    max_reconnects: int = 3,
    backoff_seconds: Sequence[float] = (0.25, 0.5, 1.0),
    receive_timeout_seconds: float | None = 15.0,
    clock: Clock = utc_now_text,
    sleep: Sleep = asyncio.sleep,
) -> CaptureResult:
    """Reconnect a bounded number of times and expose each continuity risk."""

    async def record_session(
        websocket: WebSocketTransport, remaining: int
    ) -> CaptureResult:
        return await record_polymarket(
            websocket,
            asset_ids,
            raw_root,
            max_frames=remaining,
            receive_timeout_seconds=receive_timeout_seconds,
            clock=clock,
        )

    return await _record_with_reconnect(
        connect,
        record_session,
        max_frames=max_frames,
        max_reconnects=max_reconnects,
        backoff_seconds=backoff_seconds,
        reconnect_is_gap=True,
        sleep=sleep,
    )


async def record_kalshi_with_reconnect(
    connect: Callable[[], Any],
    market_tickers: Sequence[str],
    raw_root: str | Path,
    *,
    max_frames: int,
    max_reconnects: int = 3,
    backoff_seconds: Sequence[float] = (0.25, 0.5, 1.0),
    receive_timeout_seconds: float | None = 15.0,
    clock: Clock = utc_now_text,
    sleep: Sleep = asyncio.sleep,
) -> CaptureResult:
    """Reconnect Kalshi while retaining native sid/seq continuity state."""

    tracker = SequenceTracker()

    async def record_session(
        websocket: WebSocketTransport, remaining: int
    ) -> CaptureResult:
        return await _record_kalshi_connection(
            websocket,
            market_tickers,
            raw_root,
            max_frames=remaining,
            receive_timeout_seconds=receive_timeout_seconds,
            clock=clock,
            sequence_tracker=tracker,
        )

    return await _record_with_reconnect(
        connect,
        record_session,
        max_frames=max_frames,
        max_reconnects=max_reconnects,
        backoff_seconds=backoff_seconds,
        reconnect_is_gap=False,
        sleep=sleep,
    )


async def _record_with_reconnect(
    connect: Callable[[], Any],
    record_session: Callable[[WebSocketTransport, int], Awaitable[CaptureResult]],
    *,
    max_frames: int,
    max_reconnects: int,
    backoff_seconds: Sequence[float],
    reconnect_is_gap: bool,
    sleep: Sleep,
) -> CaptureResult:
    """Aggregate bounded connection attempts without inventing missing frames."""

    remaining = _positive_frame_limit(max_frames)
    delays = _validated_backoff(max_reconnects, backoff_seconds)
    manifests: list[SegmentManifest] = []
    frames = 0
    parse_errors = 0
    reconnects = 0
    gaps = 0
    out_of_order = 0
    reason = "capture did not start"

    while True:
        try:
            async with connect() as websocket:
                session = await record_session(websocket, remaining)
        except Exception as exc:
            session = None
            reason = _terminal_reason(exc)
        else:
            manifests.extend(session.manifests)
            frames += session.counters.frames
            parse_errors += session.counters.parse_errors
            gaps += session.counters.gaps
            out_of_order += session.counters.out_of_order
            remaining -= session.counters.frames
            reason = session.terminal_reason
            if remaining == 0:
                return CaptureResult(
                    manifests=tuple(manifests),
                    counters=CaptureCounters(
                        frames=frames,
                        parse_errors=parse_errors,
                        reconnects=reconnects,
                        gaps=gaps,
                        out_of_order=out_of_order,
                    ),
                    complete=True,
                    terminal_reason="max_frames reached",
                )

        if reconnects == max_reconnects:
            return CaptureResult(
                manifests=tuple(manifests),
                counters=CaptureCounters(
                    frames=frames,
                    parse_errors=parse_errors,
                    reconnects=reconnects,
                    gaps=gaps,
                    out_of_order=out_of_order,
                ),
                complete=False,
                terminal_reason=reason,
            )
        await sleep(delays[reconnects])
        reconnects += 1
        if reconnect_is_gap:
            # Polymarket documents no native sequence number; every reconnect is
            # an observed interval whose continuity cannot be proven.
            gaps += 1


class FormalReplayRejected(RuntimeError):
    """A venue-rule snapshot is incomplete or non-canonical."""


_RULE_FIELDS = (
    "effective_from",
    "game_start_time",
    "seconds_delay",
    "cancel_during_delay",
    "start_time_cancel_policy",
    "fees_enabled",
    "fee_rate",
    "fee_exponent",
    "taker_only",
    "maker_fee_rate",
    "minimum_tick_size",
    "minimum_order_size",
    "order_types_supported",
)
_DECIMAL_RULE_FIELDS = (
    "fee_rate",
    "fee_exponent",
    "maker_fee_rate",
    "minimum_tick_size",
    "minimum_order_size",
)


@dataclass(frozen=True, slots=True)
class VenueRuleSnapshot:
    venue: str
    condition_id: str
    fetched_at: str
    effective_from: str | None
    game_start_time: str | None
    seconds_delay: int | None
    cancel_during_delay: bool | None
    start_time_cancel_policy: str | None
    fees_enabled: bool | None
    fee_rate: str | None
    fee_exponent: str | None
    taker_only: bool | None
    maker_fee_rate: str | None
    minimum_tick_size: str | None
    minimum_order_size: str | None
    order_types_supported: tuple[str, ...]
    source_document_version: str
    raw_response_hash: str
    valid: bool
    quality_flags: tuple[str, ...]

    def require_formal_replay(self) -> VenueRuleSnapshot:
        if not self.valid:
            raise FormalReplayRejected(
                "venue-rule snapshot rejected: " + ", ".join(self.quality_flags)
            )
        return self


@dataclass(frozen=True, slots=True)
class VenueRuleCapture:
    raw_manifest: SegmentManifest
    snapshot: VenueRuleSnapshot


def _strict_rule_json(payload: bytes, _manifest_path: Path) -> Mapping[str, Any]:
    def object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        parse_float=Decimal,
        object_pairs_hook=object_no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite number: {value}")
        ),
    )
    if type(value) is not dict:
        raise ValueError("rule response must be a JSON object")
    return value


def _is_utc_timestamp(value: Any) -> bool:
    if type(value) is not str or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _decimal_text(value: Any) -> str | None:
    if type(value) is float or type(value) is bool:
        return None
    if type(value) not in {str, int, Decimal}:
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    return str(value)


def _normalize_rule_document(
    document: Mapping[str, Any],
    *,
    venue: str,
    condition_id: str,
    fetched_at: str,
    source_document_version: str,
    raw_response_hash: str,
) -> VenueRuleSnapshot:
    flags: list[str] = []
    for field in _RULE_FIELDS:
        if field not in document:
            flags.append(f"MISSING_RULE_FIELD:{field}")
    for field in document:
        if field not in _RULE_FIELDS:
            flags.append(f"UNEXPECTED_RULE_FIELD:{field}")

    timestamps: dict[str, str | None] = {}
    for field in ("effective_from", "game_start_time"):
        value = document.get(field)
        if not _is_utc_timestamp(value):
            if field in document:
                flags.append(f"INVALID_RULE_FIELD:{field}")
            timestamps[field] = None
        else:
            timestamps[field] = value

    seconds_value = document.get("seconds_delay")
    seconds_delay = (
        seconds_value
        if type(seconds_value) is int and seconds_value >= 0
        else None
    )
    if "seconds_delay" in document and seconds_delay is None:
        flags.append("INVALID_RULE_FIELD:seconds_delay")

    booleans: dict[str, bool | None] = {}
    for field in ("cancel_during_delay", "fees_enabled", "taker_only"):
        value = document.get(field)
        if type(value) is not bool:
            if field in document:
                flags.append(f"INVALID_RULE_FIELD:{field}")
            booleans[field] = None
        else:
            booleans[field] = value

    policy_value = document.get("start_time_cancel_policy")
    policy = (
        policy_value
        if type(policy_value) is str and bool(policy_value.strip())
        else None
    )
    if "start_time_cancel_policy" in document and policy is None:
        flags.append("INVALID_RULE_FIELD:start_time_cancel_policy")

    decimals: dict[str, str | None] = {}
    for field in _DECIMAL_RULE_FIELDS:
        normalized = _decimal_text(document.get(field))
        if field in document and normalized is None:
            flags.append(f"INVALID_RULE_FIELD:{field}")
        decimals[field] = normalized

    orders_value = document.get("order_types_supported")
    orders: tuple[str, ...] = ()
    if (
        type(orders_value) is list
        and orders_value
        and all(type(value) is str and value for value in orders_value)
        and len(set(orders_value)) == len(orders_value)
    ):
        orders = tuple(orders_value)
    elif "order_types_supported" in document:
        flags.append("INVALID_RULE_FIELD:order_types_supported")

    quality_flags = tuple(sorted(set(flags)))
    return VenueRuleSnapshot(
        venue=venue,
        condition_id=condition_id,
        fetched_at=fetched_at,
        effective_from=timestamps["effective_from"],
        game_start_time=timestamps["game_start_time"],
        seconds_delay=seconds_delay,
        cancel_during_delay=booleans["cancel_during_delay"],
        start_time_cancel_policy=policy,
        fees_enabled=booleans["fees_enabled"],
        fee_rate=decimals["fee_rate"],
        fee_exponent=decimals["fee_exponent"],
        taker_only=booleans["taker_only"],
        maker_fee_rate=decimals["maker_fee_rate"],
        minimum_tick_size=decimals["minimum_tick_size"],
        minimum_order_size=decimals["minimum_order_size"],
        order_types_supported=orders,
        source_document_version=source_document_version,
        raw_response_hash=raw_response_hash,
        valid=not quality_flags,
        quality_flags=quality_flags,
    )


def capture_venue_rule_snapshot(
    raw_response: bytes,
    *,
    raw_root: str | Path,
    venue: str,
    condition_id: str,
    fetched_at: str,
    source_document_version: str,
    parser: RuleParser = _strict_rule_json,
) -> VenueRuleCapture:
    """Seal one exact response before any parser or normalizer sees it."""

    if type(raw_response) is not bytes:
        raise TypeError("raw_response must be bytes")
    if type(condition_id) is not str or not condition_id:
        raise ValueError("condition_id must be non-empty")
    if type(source_document_version) is not str or not source_document_version:
        raise ValueError("source_document_version must be non-empty")

    writer = RawSegmentWriter(raw_root, source=venue, stream="venue-rules")
    writer.append(raw_response, receive_at=fetched_at)
    raw_manifest = writer.seal()
    raw_response_hash = "sha256:" + hashlib.sha256(raw_response).hexdigest()

    try:
        document = parser(raw_response, raw_manifest.path)
        if not isinstance(document, Mapping):
            raise ValueError("rule parser must return a mapping")
    except Exception as exc:
        snapshot = VenueRuleSnapshot(
            venue=venue,
            condition_id=condition_id,
            fetched_at=fetched_at,
            effective_from=None,
            game_start_time=None,
            seconds_delay=None,
            cancel_during_delay=None,
            start_time_cancel_policy=None,
            fees_enabled=None,
            fee_rate=None,
            fee_exponent=None,
            taker_only=None,
            maker_fee_rate=None,
            minimum_tick_size=None,
            minimum_order_size=None,
            order_types_supported=(),
            source_document_version=source_document_version,
            raw_response_hash=raw_response_hash,
            valid=False,
            quality_flags=(f"RULE_RESPONSE_PARSE_ERROR:{type(exc).__name__}",),
        )
    else:
        snapshot = _normalize_rule_document(
            document,
            venue=venue,
            condition_id=condition_id,
            fetched_at=fetched_at,
            source_document_version=source_document_version,
            raw_response_hash=raw_response_hash,
        )
    return VenueRuleCapture(raw_manifest=raw_manifest, snapshot=snapshot)


__all__ = [
    "CaptureCounters",
    "CaptureResult",
    "FormalReplayRejected",
    "VenueRuleCapture",
    "VenueRuleSnapshot",
    "capture_venue_rule_snapshot",
    "record_kalshi",
    "record_kalshi_with_reconnect",
    "record_polymarket",
    "record_polymarket_with_reconnect",
]
