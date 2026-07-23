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
from prediction_market.contracts import FixedPointV0, VenueRuleSnapshotV0
from prediction_market.raw_store import (
    PartitionBoundaryError,
    RawSegmentWriter,
    RawStoreError,
    SegmentManifest,
    read_verified_segment,
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
    """Reconnect Kalshi with one native sequence epoch per subscription."""

    async def record_session(
        websocket: WebSocketTransport, remaining: int
    ) -> CaptureResult:
        tracker = SequenceTracker()
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
        sleep=sleep,
    )


async def _record_with_reconnect(
    connect: Callable[[], Any],
    record_session: Callable[[WebSocketTransport, int], Awaitable[CaptureResult]],
    *,
    max_frames: int,
    max_reconnects: int,
    backoff_seconds: Sequence[float],
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
    continuity_unknown = 0
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
                        continuity_unknown=continuity_unknown,
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
                    continuity_unknown=continuity_unknown,
                    out_of_order=out_of_order,
                ),
                complete=False,
                terminal_reason=reason,
            )
        await sleep(delays[reconnects])
        reconnects += 1
        # A reconnect starts a new WebSocket/subscription epoch.  Native
        # sequence numbers cannot prove continuity across that boundary, even
        # when the next epoch happens to reuse the same sid or seq value.
        gaps += 1
        continuity_unknown += 1


class FormalReplayRejected(RuntimeError):
    """No valid, current canonical venue-rule snapshot is available."""


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
class VenueRuleCapture:
    raw_manifest: SegmentManifest
    canonical_manifest: SegmentManifest
    snapshot: VenueRuleSnapshotV0 | None
    valid: bool
    quality_flags: tuple[str, ...]
    validation_errors: tuple[str, ...]


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_rule_json(payload: bytes, _manifest_path: Path) -> Mapping[str, Any]:

    value = json.loads(
        payload.decode("utf-8"),
        parse_float=Decimal,
        object_pairs_hook=_object_no_duplicates,
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


def _utc_instant(value: Any, *, field: str) -> datetime:
    if not _is_utc_timestamp(value):
        raise ValueError(f"{field} must be a canonical UTC timestamp")
    assert isinstance(value, str)
    return datetime.fromisoformat(value[:-1] + "+00:00")


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
) -> tuple[VenueRuleSnapshotV0 | None, tuple[str, ...]]:
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
    policy = policy_value if policy_value in {
        "cancel_all_at_game_start",
        "cancel_all_with_schedule_change_exception",
        "preserve_orders_at_game_start",
    } else None
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

    validation_errors = tuple(sorted(set(flags)))
    if validation_errors:
        return None, validation_errors

    try:
        snapshot = VenueRuleSnapshotV0.model_validate(
            {
                "venue": venue,
                "condition_id": condition_id,
                "fetched_at": fetched_at,
                "effective_from": timestamps["effective_from"],
                "game_start_time": timestamps["game_start_time"],
                "seconds_delay": FixedPointV0.from_value(seconds_delay),
                "cancel_during_delay": booleans["cancel_during_delay"],
                "start_time_cancel_policy": policy,
                "fees_enabled": booleans["fees_enabled"],
                "fee_rate": FixedPointV0.from_value(decimals["fee_rate"]),
                "fee_exponent": FixedPointV0.from_value(decimals["fee_exponent"]),
                "taker_only": booleans["taker_only"],
                "maker_fee_rate": FixedPointV0.from_value(
                    decimals["maker_fee_rate"]
                ),
                "minimum_tick_size": FixedPointV0.from_value(
                    decimals["minimum_tick_size"]
                ),
                "minimum_order_size": FixedPointV0.from_value(
                    decimals["minimum_order_size"]
                ),
                "order_types_supported": orders,
                "source_document_version": source_document_version,
                "raw_response_hash": raw_response_hash,
            }
        )
    except (TypeError, ValueError) as exc:
        return None, (f"INVALID_CANONICAL_RULE:{type(exc).__name__}",)
    return snapshot, ()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _relative_to_store(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("venue-rule artifact escapes canonical store root") from exc


def _rule_record(
    *,
    root: Path,
    raw_manifest: SegmentManifest,
    venue: str,
    market_id: str,
    condition_id: str,
    fetched_at: str,
    source_document_version: str,
    raw_response_hash: str,
    snapshot: VenueRuleSnapshotV0 | None,
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    valid = snapshot is not None and not validation_errors
    return {
        "record_version": "v0",
        "venue": venue,
        "market_id": market_id,
        "condition_id": condition_id,
        "fetched_at": fetched_at,
        "source_document_version": source_document_version,
        "raw_response_hash": raw_response_hash,
        "raw_manifest_path": _relative_to_store(raw_manifest.path, root),
        "raw_object_sha256": raw_manifest.object_sha256,
        "raw_record_ordinal": 0,
        "valid": valid,
        "quality_flags": [] if valid else ["preliminary_rules"],
        "validation_errors": list(validation_errors),
        "snapshot": snapshot.model_dump(mode="json") if snapshot is not None else None,
    }


def capture_venue_rule_snapshot(
    raw_response: bytes,
    *,
    raw_root: str | Path,
    venue: str,
    market_id: str,
    condition_id: str,
    fetched_at: str,
    source_document_version: str,
    parser: RuleParser = _strict_rule_json,
) -> VenueRuleCapture:
    """Seal one exact response before any parser or normalizer sees it."""

    if type(raw_response) is not bytes:
        raise TypeError("raw_response must be bytes")
    if type(market_id) is not str or re.fullmatch(
        r"market_[A-Za-z0-9][A-Za-z0-9._:-]*", market_id
    ) is None:
        raise ValueError("market_id must be a canonical market ID")
    if type(condition_id) is not str or re.fullmatch(
        r"condition_[A-Za-z0-9][A-Za-z0-9._:-]*", condition_id
    ) is None:
        raise ValueError("condition_id must be a canonical condition ID")
    if type(source_document_version) is not str or not source_document_version:
        raise ValueError("source_document_version must be non-empty")
    _utc_instant(fetched_at, field="fetched_at")

    writer = RawSegmentWriter(raw_root, source=venue, stream="venue-rules")
    writer.append(raw_response, receive_at=fetched_at)
    raw_manifest = writer.seal()
    raw_response_hash = "sha256:" + hashlib.sha256(raw_response).hexdigest()

    try:
        document = parser(raw_response, raw_manifest.path)
        if not isinstance(document, Mapping):
            raise ValueError("rule parser must return a mapping")
    except Exception as exc:
        snapshot = None
        validation_errors = (f"RULE_RESPONSE_PARSE_ERROR:{type(exc).__name__}",)
    else:
        snapshot, validation_errors = _normalize_rule_document(
            document,
            venue=venue,
            condition_id=condition_id,
            fetched_at=fetched_at,
            source_document_version=source_document_version,
            raw_response_hash=raw_response_hash,
        )

    root = Path(raw_root).resolve(strict=True)
    record = _rule_record(
        root=root,
        raw_manifest=raw_manifest,
        venue=venue,
        market_id=market_id,
        condition_id=condition_id,
        fetched_at=fetched_at,
        source_document_version=source_document_version,
        raw_response_hash=raw_response_hash,
        snapshot=snapshot,
        validation_errors=validation_errors,
    )
    canonical_writer = RawSegmentWriter(
        root, source=venue, stream="venue-rule-snapshots"
    )
    canonical_writer.append(_canonical_json_bytes(record), receive_at=fetched_at)
    canonical_manifest = canonical_writer.seal()
    valid = snapshot is not None and not validation_errors
    return VenueRuleCapture(
        raw_manifest=raw_manifest,
        canonical_manifest=canonical_manifest,
        snapshot=snapshot,
        valid=valid,
        quality_flags=() if valid else ("preliminary_rules",),
        validation_errors=validation_errors,
    )


_RULE_RECORD_KEYS = frozenset(
    {
        "record_version",
        "venue",
        "market_id",
        "condition_id",
        "fetched_at",
        "source_document_version",
        "raw_response_hash",
        "raw_manifest_path",
        "raw_object_sha256",
        "raw_record_ordinal",
        "valid",
        "quality_flags",
        "validation_errors",
        "snapshot",
    }
)


def _load_rule_record(root: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        canonical = read_verified_segment(manifest_path, root=root)
    except RawStoreError as exc:
        raise FormalReplayRejected(
            "canonical venue-rule segment failed integrity verification"
        ) from exc
    if len(canonical.payloads) != 1:
        raise FormalReplayRejected(
            "canonical venue-rule segment must contain exactly one record"
        )
    payload = canonical.payloads[0]
    try:
        record = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_object_no_duplicates
        )
    except (UnicodeError, ValueError) as exc:
        raise FormalReplayRejected("canonical venue-rule record is invalid JSON") from exc
    if type(record) is not dict or set(record) != _RULE_RECORD_KEYS:
        raise FormalReplayRejected("canonical venue-rule record fields are invalid")
    if payload != _canonical_json_bytes(record):
        raise FormalReplayRejected("canonical venue-rule record is not canonical JSON")

    raw_manifest_value = record["raw_manifest_path"]
    if type(raw_manifest_value) is not str:
        raise FormalReplayRejected("venue-rule raw manifest reference is invalid")
    try:
        raw = read_verified_segment(root / raw_manifest_value, root=root)
    except RawStoreError as exc:
        raise FormalReplayRejected(
            "venue-rule raw response failed integrity verification"
        ) from exc
    if raw.manifest.object_sha256 != record["raw_object_sha256"]:
        raise FormalReplayRejected("venue-rule raw object reference does not match")
    if len(raw.payloads) != 1:
        raise FormalReplayRejected(
            "venue-rule raw response segment must contain exactly one record"
        )
    raw_response = raw.payloads[0]
    if "sha256:" + hashlib.sha256(raw_response).hexdigest() != record["raw_response_hash"]:
        raise FormalReplayRejected("venue-rule raw response hash does not match")
    return record


class VenueRuleStore:
    """Append-only canonical venue-rule observations with strict PIT lookup."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve(strict=True)

    def _records(self) -> tuple[dict[str, Any], ...]:
        pattern = (
            "manifests/source=*/stream=venue-rule-snapshots/"
            "date=*/hour=*/*.manifest.json"
        )
        return tuple(
            _load_rule_record(self._root, path)
            for path in sorted(self._root.glob(pattern))
        )

    def require_as_of(
        self,
        *,
        venue: str,
        market_id: str,
        condition_id: str,
        at: str,
        max_age_seconds: int,
    ) -> VenueRuleSnapshotV0:
        """Return the latest valid observation available at ``at`` or fail closed."""

        instant = _utc_instant(at, field="at")
        if (
            isinstance(max_age_seconds, bool)
            or type(max_age_seconds) is not int
            or max_age_seconds <= 0
        ):
            raise ValueError("max_age_seconds must be a positive integer")
        candidates = [
            record
            for record in self._records()
            if record.get("venue") == venue
            and record.get("market_id") == market_id
            and record.get("condition_id") == condition_id
            and _utc_instant(record.get("fetched_at"), field="fetched_at") <= instant
        ]
        if not candidates:
            raise FormalReplayRejected("venue-rule snapshot is missing as of replay time")
        active: list[tuple[datetime, datetime, dict[str, Any], VenueRuleSnapshotV0]] = []
        invalid: list[tuple[datetime, str]] = []
        for record in candidates:
            fetched = _utc_instant(record["fetched_at"], field="fetched_at")
            validation_errors = record.get("validation_errors")
            if record.get("valid") is not True or record.get("snapshot") is None:
                detail = (
                    ", ".join(validation_errors)
                    if type(validation_errors) is list
                    else ""
                )
                invalid.append((fetched, detail))
                continue
            try:
                snapshot = VenueRuleSnapshotV0.model_validate(record["snapshot"])
            except (TypeError, ValueError):
                invalid.append((fetched, "stored snapshot is non-canonical"))
                continue
            if (
                snapshot.venue != venue
                or snapshot.condition_id != condition_id
                or snapshot.source_document_version
                != record.get("source_document_version")
                or snapshot.raw_response_hash != record.get("raw_response_hash")
            ):
                invalid.append((fetched, "stored snapshot lineage does not match"))
                continue
            effective = _utc_instant(snapshot.effective_from, field="effective_from")
            if effective <= instant:
                active.append((effective, fetched, record, snapshot))

        if not active:
            if invalid:
                detail = max(invalid, key=lambda value: value[0])[1]
                raise FormalReplayRejected(
                    "venue-rule snapshot is invalid" + (f": {detail}" if detail else "")
                )
            raise FormalReplayRejected(
                "venue-rule snapshot is not yet effective as of replay time"
            )

        _, fetched, selected, snapshot = max(
            active,
            key=lambda value: (
                value[0],
                value[1],
                value[2]["raw_response_hash"],
            ),
        )
        blocking_invalid = [value for value in invalid if value[0] >= fetched]
        if blocking_invalid:
            detail = max(blocking_invalid, key=lambda value: value[0])[1]
            raise FormalReplayRejected(
                "venue-rule snapshot is invalid" + (f": {detail}" if detail else "")
            )
        if (instant - fetched).total_seconds() > max_age_seconds:
            raise FormalReplayRejected("venue-rule snapshot is stale as of replay time")
        return snapshot


__all__ = [
    "CaptureCounters",
    "CaptureResult",
    "FormalReplayRejected",
    "VenueRuleCapture",
    "VenueRuleStore",
    "capture_venue_rule_snapshot",
    "record_kalshi",
    "record_kalshi_with_reconnect",
    "record_polymarket",
    "record_polymarket_with_reconnect",
]
