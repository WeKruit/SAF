"""Long-running, fail-closed Polymarket capture readiness for NFL X-14.

This module deliberately separates three things:

* byte-exact native WebSocket frames;
* canonical, timestamped prospective market records;
* Gamma universe/resolution refreshes.

It does not claim that Polymarket exposes a sequence number.  Every reconnect
therefore opens a new explicit epoch and requires a new book snapshot for each
token before a depth delta can be accepted.
"""

from __future__ import annotations

import asyncio
import argparse
import calendar
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

import httpx
from websockets.asyncio.client import connect as websocket_connect

from prediction_market.adapters.base import exact_frame_bytes
from prediction_market.adapters.polymarket import (
    MARKET_WS_URL,
    build_market_subscription,
    parse_market_frame,
)
from prediction_market.pmxt.data_store import (
    DataStoreError,
    configured_object_store,
)
from prediction_market.raw_store import (
    PartitionBoundaryError,
    RawSegmentWriter,
    SegmentManifest,
    verify_segment,
)
from prediction_market.contracts import CaptureTimingV0
from prediction_market.sports.nfl_x14_polymarket_recorder import (
    GammaHttpResponse,
    TargetUniverse,
    X14PolymarketRecorderError,
    _iso_instant,
    _strict_json,
    fetch_target_event,
    parse_target_universe,
    preserve_target_metadata,
)
from prediction_market.sports.nfl_x14_prospective import (
    CaptureContinuityError,
    CaptureMode,
    ProspectiveCaptureRecord,
    ProspectiveCaptureValidator,
)
from prediction_market.sports.nfl_x14_reaction import (
    BookLevel,
    CaptureContinuityGate,
    CaptureClockContextV1,
    MarketUpdateV1,
    ProspectiveGameBindingV1,
    RawCaptureEnvelopeV1,
    load_prospective_game_binding,
)
from prediction_market.static_store import StaticObjectRecord


_EVENT_KIND = {
    "book": "market_snapshot",
    "price_change": "market_delta",
    "last_trade_price": "trade",
    "best_bid_ask": "bbo",
}
_RAW_CONTROL_EVENT_TYPES = frozenset(
    {"tick_size_change", "new_market", "market_resolved"}
)


class LiveCaptureError(ValueError):
    """A live X-14 capture invariant could not be proven."""


@dataclass(frozen=True, slots=True)
class UnknownTargetMarket:
    market_id: str
    condition_id: str | None
    market_type: str | None
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "market_type": self.market_type,
            "reason": self.reason,
        }


class UnknownMarketUniverseError(LiveCaptureError):
    """Gamma exposed a market whose single-game dependency is not supported."""

    def __init__(
        self,
        inventory: Sequence[UnknownTargetMarket],
        *,
        inventory_path: Path | None = None,
    ) -> None:
        self.inventory = tuple(inventory)
        self.inventory_path = inventory_path
        ids = ", ".join(item.market_id for item in self.inventory)
        super().__init__(
            "unknown Gamma markets require an explicit parser before "
            f"subscription: {ids}"
        )


@dataclass(frozen=True, slots=True)
class TargetRefreshInspection:
    universe: TargetUniverse | None
    unknown_markets: tuple[UnknownTargetMarket, ...]
    resolution_observed: bool
    fetched_at: str

    def require_supported_universe(self) -> TargetUniverse:
        if self.unknown_markets:
            raise UnknownMarketUniverseError(self.unknown_markets)
        if self.universe is None:
            raise LiveCaptureError("supported target universe is absent")
        return self.universe


@dataclass(frozen=True, slots=True)
class LiveCaptureResult:
    terminal_reason: str
    resolution_observed: bool
    manifests: tuple[SegmentManifest, ...]
    metadata_records: tuple[StaticObjectRecord, ...]
    report_path: Path
    report_sha256: str
    s3_upload: Mapping[str, Any]
    audit: Mapping[str, Any]


class LiveObjectStore(Protocol):
    def head_object(self, key: str) -> dict[str, object] | None: ...

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None: ...

    def read_object_chunks(
        self, key: str, *, chunk_size: int = 1024 * 1024
    ) -> Any: ...


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_size += len(block)
    return "sha256:" + digest.hexdigest(), byte_size


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _canonical_utc(value: str, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise LiveCaptureError(f"{field} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise LiveCaptureError(
            f"{field} must be a canonical UTC timestamp"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LiveCaptureError(f"{field} must be a canonical UTC timestamp")
    if (
        parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
        != value
    ):
        raise LiveCaptureError(f"{field} must be a canonical UTC timestamp")
    return value


def _source_time(value: object) -> str:
    if type(value) is str and value.endswith("Z"):
        return _canonical_utc(value, "market source timestamp")
    if type(value) is str and value.isdigit():
        milliseconds = int(value)
    elif type(value) is int and not isinstance(value, bool):
        milliseconds = value
    else:
        raise LiveCaptureError(
            "market frame requires a documented millisecond source timestamp"
        )
    # Polymarket market-channel timestamps are Unix milliseconds.  Refuse to
    # infer seconds or microseconds from their magnitude.
    if milliseconds < 1_000_000_000_000 or milliseconds > 9_999_999_999_999:
        raise LiveCaptureError(
            "market frame timestamp must be Unix milliseconds"
        )
    try:
        parsed = datetime.fromtimestamp(
            milliseconds / 1000, tz=timezone.utc
        )
    except (OverflowError, OSError, ValueError) as error:
        raise LiveCaptureError(
            "market frame timestamp is outside supported UTC range"
        ) from error
    return (
        parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _resolution_observed(event: Mapping[str, Any]) -> bool:
    if event.get("closed") is not True:
        return False
    markets = event.get("markets")
    if type(markets) is not list or not markets:
        return False
    for market in markets:
        if type(market) is not dict or market.get("closed") is not True:
            return False
        prices = market.get("outcomePrices")
        if type(prices) is not str:
            return False
        try:
            values = json.loads(prices)
        except json.JSONDecodeError:
            return False
        if type(values) is not list or len(values) != 2:
            return False
        try:
            normalized = sorted(Decimal(str(item)) for item in values)
        except (InvalidOperation, ValueError):
            return False
        if normalized != [Decimal("0"), Decimal("1")]:
            return False
    return True


def _unknown_reason(
    market: Mapping[str, Any],
    *,
    binding: ProspectiveGameBindingV1,
) -> str | None:
    market_type = market.get("sportsMarketType")
    if type(market_type) is not str or not market_type:
        return "unsupported_sports_market_type"
    dependencies = (
        market.get("mveCollection"),
        market.get("mveLegs"),
        market.get("dependentEvents"),
        market.get("dependencyEventIds"),
    )
    if market.get("comboStatus") != "disabled" or any(
        value is not None and value != () and value != []
        for value in dependencies
    ):
        return "composite_or_cross_game_dependency"
    try:
        market_start = _iso_instant(
            market.get("gameStartTime"), "market.gameStartTime"
        )
    except X14PolymarketRecorderError:
        return "cross_game_start_time"
    expected_start = datetime.fromisoformat(
        binding.scheduled_start_utc[:-1] + "+00:00"
    )
    if market_start != expected_start:
        return "cross_game_start_time"
    return None


def inspect_target_refresh(
    response: GammaHttpResponse,
    *,
    binding: ProspectiveGameBindingV1,
) -> TargetRefreshInspection:
    """Inventory unsupported markets, then validate the exact supported set."""

    event = _strict_json(response.payload)
    if type(event) is not dict:
        raise LiveCaptureError("Gamma target refresh must be an object")
    markets = event.get("markets")
    if type(markets) is not list or not markets:
        # Reuse the exact identity/parser diagnostics for malformed payloads.
        parse_target_universe(response, binding=binding)
        raise AssertionError("unreachable")

    unknown: list[UnknownTargetMarket] = []
    supported: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(markets):
        if type(value) is not dict:
            raise LiveCaptureError(f"Gamma market[{index}] is not an object")
        market_id = value.get("id")
        if (
            type(market_id) is not str
            or not market_id
            or market_id in seen_ids
        ):
            raise LiveCaptureError("Gamma market identity is invalid")
        seen_ids.add(market_id)
        reason = _unknown_reason(value, binding=binding)
        if reason is None:
            supported.append(value)
        else:
            condition = value.get("conditionId")
            unknown.append(
                UnknownTargetMarket(
                    market_id=market_id,
                    condition_id=condition
                    if type(condition) is str
                    else None,
                    market_type=value.get("sportsMarketType")
                    if type(value.get("sportsMarketType")) is str
                    else None,
                    reason=reason,
                )
            )

    # Validate event/team/game identity and every supported market using the
    # existing exact parser.  The filtered copy is validation-only; the exact
    # response bytes are always preserved separately.
    supported_event = dict(event)
    supported_event["markets"] = supported
    supported_response = GammaHttpResponse(
        payload=_canonical_bytes(supported_event),
        fetched_at=response.fetched_at,
        source_url=response.source_url,
        request_params=response.request_params,
        headers=response.headers,
    )
    universe = parse_target_universe(
        supported_response, binding=binding
    )
    return TargetRefreshInspection(
        universe=None if unknown else universe,
        unknown_markets=tuple(unknown),
        resolution_observed=_resolution_observed(event),
        fetched_at=response.fetched_at,
    )


def _token_events(event: Mapping[str, Any]) -> tuple[str, ...]:
    event_type = event.get("event_type")
    if event_type in _RAW_CONTROL_EVENT_TYPES:
        return ()
    if event_type not in _EVENT_KIND:
        raise LiveCaptureError(
            f"unsupported Polymarket market event_type: {event_type}"
        )
    if event_type == "price_change" and "price_changes" in event:
        changes = event.get("price_changes")
        if type(changes) is not list or not changes:
            raise LiveCaptureError(
                "price_change.price_changes must be a non-empty array"
            )
        tokens: list[str] = []
        for change in changes:
            if type(change) is not dict or type(change.get("asset_id")) is not str:
                raise LiveCaptureError(
                    "price_change item requires asset_id"
                )
            tokens.append(change["asset_id"])
        return tuple(tokens)
    token = event.get("asset_id")
    if type(token) is not str or not token:
        raise LiveCaptureError("market event requires asset_id")
    return (token,)


def parse_live_market_frame(
    payload: bytes,
    *,
    universe: TargetUniverse,
    epoch: int,
    local_received_time: str,
) -> tuple[ProspectiveCaptureRecord, ...]:
    """Convert one exact native frame without inventing time or order."""

    local_time = _canonical_utc(
        local_received_time, "local_received_time"
    )
    if payload == b"PONG":
        return ()
    events = parse_market_frame(payload)
    by_token = {
        token: market
        for market in universe.markets
        for token in market.token_ids
    }
    records: list[ProspectiveCaptureRecord] = []
    for event in events:
        event_type = event["event_type"]
        tokens = _token_events(event)
        if not tokens:
            continue
        source_time = _source_time(event.get("timestamp"))
        supplied_condition = event.get("market")
        for token in tokens:
            market = by_token.get(token)
            if market is None:
                raise LiveCaptureError(
                    f"market frame token is outside frozen universe: {token}"
                )
            if (
                supplied_condition is not None
                and supplied_condition != market.condition_id
            ):
                raise LiveCaptureError(
                    "market frame condition/token orientation mismatch"
                )
            records.append(
                ProspectiveCaptureRecord(
                    experiment_id="X-14",
                    session_id=universe.binding.capture_session_id,
                    game_id=universe.binding.game_id,
                    epoch=epoch,
                    capture_mode=CaptureMode.PROSPECTIVE_L2,
                    record_class="market",
                    event_kind=_EVENT_KIND[event_type],
                    source="polymarket",
                    source_received_time=source_time,
                    local_received_time=local_time,
                    source_sequence=None,
                    market_id=market.market_id,
                    condition_id=market.condition_id,
                    token_id=token,
                    play_id=None,
                    play_revision=None,
                    native_payload=payload,
                )
            )
    return tuple(records)


def _utc_ns(value: str, field: str) -> int:
    canonical = _canonical_utc(value, field)
    parsed = datetime.fromisoformat(canonical[:-1] + "+00:00")
    return (
        calendar.timegm(parsed.utctimetuple()) * 1_000_000_000
        + parsed.microsecond * 1_000
    )


def _decimal(value: object, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise LiveCaptureError(f"{field} is not decimal") from error
    if not parsed.is_finite():
        raise LiveCaptureError(f"{field} is not finite")
    return parsed


def _book_levels(
    value: object,
    *,
    side: str,
) -> tuple[BookLevel, ...]:
    if type(value) is not list:
        raise LiveCaptureError(f"{side} book levels must be an array")
    levels: list[BookLevel] = []
    for index, row in enumerate(value):
        if type(row) is not dict:
            raise LiveCaptureError(
                f"{side}[{index}] must be an object"
            )
        levels.append(
            BookLevel(
                price=_decimal(row.get("price"), f"{side}.price"),
                size=_decimal(row.get("size"), f"{side}.size"),
            )
        )
    return tuple(
        sorted(
            levels,
            key=lambda row: row.price,
            reverse=side == "bids",
        )
    )


@dataclass(frozen=True, slots=True)
class _PreparedMarketUpdate:
    market_id: str
    token_id: str
    outcome: str
    update_kind: str
    source_time_utc_ns: int
    book_hash: str | None
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    tick_size: Decimal
    continuity_status: str


@dataclass(frozen=True, slots=True)
class _PendingTimedFrame:
    envelope_id: str
    native_payload: bytes
    local_received_time: str
    timing: CaptureTimingV0
    message_kind: str
    source_time_utc_ns: int | None
    source_time_precision_ns: int | None
    book_hash: str | None
    market_id: str | None
    token_id: str | None
    updates: tuple[MarketUpdateV1, ...]


class LiveCaptureState:
    """Durable native/canonical journals plus explicit epoch audit state."""

    def __init__(
        self, *, raw_root: str | Path, universe: TargetUniverse
    ) -> None:
        self.raw_root = Path(raw_root)
        self.universe = universe
        self.validator = ProspectiveCaptureValidator(
            session_id=universe.binding.capture_session_id,
            game_id=universe.binding.game_id,
        )
        self._native: RawSegmentWriter | None = None
        self._canonical: RawSegmentWriter | None = None
        self._manifests: list[SegmentManifest] = []
        self._drain_index = 0
        self._snapshots: set[str] = set()
        self._required_tokens_by_epoch: dict[int, set[str]] = {
            0: set(universe.token_ids)
        }
        self._snapshots_by_epoch: dict[int, set[str]] = {0: set()}
        self._closed = False
        self._epoch_rows: list[dict[str, Any]] = [
            {"epoch": 0, "reason": "initial_connect"}
        ]
        self._gap_count = 0
        self._snapshot_requirements = 0
        self._delta_before_snapshot_count = 0
        self._native_frame_count = 0
        self._canonical_record_count = 0
        self._control_frame_count = 0
        self._record_ordinal = 0
        self._pending_timed_frames: list[_PendingTimedFrame] = []
        self._finalized_envelopes: list[RawCaptureEnvelopeV1] = []
        self._market_updates: list[MarketUpdateV1] = []
        self._books: dict[
            str, tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]
        ] = {}
        self._tick_sizes: dict[str, Decimal] = {}
        self._timed_gate = CaptureContinuityGate()

    @property
    def finalized_envelopes(self) -> tuple[RawCaptureEnvelopeV1, ...]:
        return tuple(self._finalized_envelopes)

    @property
    def market_updates(self) -> tuple[MarketUpdateV1, ...]:
        return tuple(self._market_updates)

    @property
    def epoch(self) -> int:
        return self.validator.epoch

    @property
    def audit(self) -> dict[str, Any]:
        missing = {
            str(epoch): sorted(
                required - self._snapshots_by_epoch.get(epoch, set())
            )
            for epoch, required in sorted(
                self._required_tokens_by_epoch.items()
            )
        }
        snapshot_complete = all(not values for values in missing.values())
        return {
            "audit_version": "nfl-x14-live-continuity/v1",
            "epoch_count": len(self._epoch_rows),
            "epochs": list(self._epoch_rows),
            "gap_count": self._gap_count,
            "new_snapshot_required_count": self._snapshot_requirements,
            "delta_before_snapshot_count": (
                self._delta_before_snapshot_count
            ),
            "native_frame_count": self._native_frame_count,
            "canonical_record_count": self._canonical_record_count,
            "control_frame_count": self._control_frame_count,
            "source_sequence_available": False,
            "gap_repair_fabricated": False,
            "required_tokens_by_epoch": {
                str(epoch): sorted(tokens)
                for epoch, tokens in sorted(
                    self._required_tokens_by_epoch.items()
                )
            },
            "missing_snapshot_tokens_by_epoch": missing,
            "snapshot_coverage_complete": snapshot_complete,
            "continuity_complete": (
                self._gap_count == 0
                and self._delta_before_snapshot_count == 0
                and snapshot_complete
            ),
        }

    def _new_native(self) -> RawSegmentWriter:
        return RawSegmentWriter(
            self.raw_root,
            source="polymarket",
            stream="nfl-x14-native",
            capture_session_id=self.universe.binding.capture_session_id,
        )

    def _new_canonical(self) -> RawSegmentWriter:
        return RawSegmentWriter(
            self.raw_root,
            source="polymarket",
            stream="nfl-x14-canonical",
            capture_session_id=self.universe.binding.capture_session_id,
        )

    def _seal_native(self) -> None:
        if self._native is None:
            if self._pending_timed_frames:
                raise LiveCaptureError(
                    "timed frames exist without a native segment"
                )
            return
        manifest = self._native.seal()
        self._manifests.append(manifest)
        self._native = None
        for pending in self._pending_timed_frames:
            envelope = RawCaptureEnvelopeV1(
                envelope_id=pending.envelope_id,
                binding_id=self.universe.binding.binding_id,
                game_id=self.universe.binding.game_id,
                source="polymarket",
                stream="market",
                message_kind=pending.message_kind,
                timing=pending.timing,
                source_time_utc_ns=pending.source_time_utc_ns,
                source_time_precision_ns=(
                    pending.source_time_precision_ns
                ),
                source_clock_domain=(
                    "polymarket_exchange"
                    if pending.source_time_utc_ns is not None
                    else None
                ),
                source_sequence=None,
                book_hash=pending.book_hash,
                market_id=pending.market_id,
                token_or_ticker=pending.token_id,
                provider_event_id=None,
                revision=None,
                segment_sha256=manifest.object_sha256,
                raw_payload=pending.native_payload,
            )
            self._append_canonical(
                _canonical_bytes(envelope.to_dict()),
                pending.local_received_time,
            )
            for update in pending.updates:
                self._append_canonical(
                    _canonical_bytes(update.to_dict()),
                    pending.local_received_time,
                )
            self._finalized_envelopes.append(envelope)
        self._pending_timed_frames.clear()

    def _append_native(self, payload: bytes, receive_at: str) -> None:
        if self._native is None:
            self._native = self._new_native()
        try:
            self._native.append(payload, receive_at=receive_at)
        except PartitionBoundaryError:
            self._seal_native()
            self._native = self._new_native()
            self._native.append(payload, receive_at=receive_at)
        self._native_frame_count += 1

    def _append_canonical(
        self, payload: bytes, receive_at: str
    ) -> None:
        if self._canonical is None:
            self._canonical = self._new_canonical()
        try:
            self._canonical.append(payload, receive_at=receive_at)
        except PartitionBoundaryError:
            self._manifests.append(self._canonical.seal())
            self._canonical = self._new_canonical()
            self._canonical.append(payload, receive_at=receive_at)

    def append_native_frame(
        self, *, native_payload: bytes, local_received_time: str
    ) -> None:
        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        self._append_native(native_payload, local_received_time)

    def _prepare_updates(
        self, events: Sequence[Mapping[str, Any]]
    ) -> tuple[_PreparedMarketUpdate, ...]:
        by_token = {
            token: (market, outcome)
            for market in self.universe.markets
            for token, outcome in zip(
                market.token_ids, market.outcomes
            )
        }
        prepared: list[_PreparedMarketUpdate] = []
        for event in events:
            event_type = event.get("event_type")
            if event_type == "new_market":
                continue
            if event_type == "price_change" and type(
                event.get("price_changes")
            ) is list:
                native_rows = event["price_changes"]
            elif event_type == "market_resolved":
                condition_id = event.get("market")
                resolved_market = next(
                    (
                        market
                        for market in self.universe.markets
                        if market.condition_id == condition_id
                    ),
                    None,
                )
                if resolved_market is None:
                    raise LiveCaptureError(
                        "resolved market is outside the bound universe"
                    )
                native_rows = [
                    {**event, "asset_id": token}
                    for token in resolved_market.token_ids
                ]
            else:
                native_rows = [event]
            for native in native_rows:
                token = native.get("asset_id")
                if type(token) is not str or token not in by_token:
                    raise LiveCaptureError(
                        "timed market frame token is outside universe"
                    )
                market, outcome = by_token[token]
                source_time = _source_time(event.get("timestamp"))
                source_ns = _utc_ns(
                    source_time, "market source timestamp"
                )
                current_bids, current_asks = self._books.get(
                    token, ((), ())
                )
                tick = self._tick_sizes.get(token)
                if event_type == "book":
                    current_bids = _book_levels(
                        event.get("bids"), side="bids"
                    )
                    current_asks = _book_levels(
                        event.get("asks"), side="asks"
                    )
                    tick = _decimal(
                        event.get("tick_size"), "book.tick_size"
                    )
                    kind = "snapshot"
                    status = "CONTIGUOUS"
                elif event_type == "tick_size_change":
                    tick = _decimal(
                        event.get("new_tick_size"),
                        "tick_size_change.new_tick_size",
                    )
                    kind = "tick_size"
                    status = "CONTIGUOUS"
                elif event_type == "market_resolved":
                    if tick is None:
                        raise LiveCaptureError(
                            "resolution arrived before tick-size state"
                        )
                    kind = "suspension"
                    status = "SUSPENDED"
                elif event_type == "price_change":
                    if tick is None or not current_bids or not current_asks:
                        raise LiveCaptureError(
                            "price delta arrived before snapshot state"
                        )
                    side = str(native.get("side", "")).upper()
                    price = _decimal(
                        native.get("price"), "price_change.price"
                    )
                    size = _decimal(
                        native.get("size"), "price_change.size"
                    )
                    target = {
                        row.price: row.size
                        for row in (
                            current_bids
                            if side in {"BUY", "BID"}
                            else current_asks
                            if side in {"SELL", "ASK"}
                            else ()
                        )
                    }
                    if not target and side not in {
                        "BUY",
                        "BID",
                        "SELL",
                        "ASK",
                    }:
                        raise LiveCaptureError(
                            "price_change side is unsupported"
                        )
                    if size == 0:
                        target.pop(price, None)
                    else:
                        target[price] = size
                    levels = tuple(
                        BookLevel(level_price, level_size)
                        for level_price, level_size in sorted(
                            target.items(),
                            reverse=side in {"BUY", "BID"},
                        )
                    )
                    if side in {"BUY", "BID"}:
                        current_bids = levels
                    else:
                        current_asks = levels
                    kind = "delta"
                    status = "CONTIGUOUS"
                elif event_type in {"best_bid_ask", "last_trade_price"}:
                    if tick is None or not current_bids or not current_asks:
                        raise LiveCaptureError(
                            "market update arrived before snapshot state"
                        )
                    if event_type == "best_bid_ask":
                        observed_bid = _decimal(
                            event.get("best_bid"),
                            "best_bid_ask.best_bid",
                        )
                        observed_ask = _decimal(
                            event.get("best_ask"),
                            "best_bid_ask.best_ask",
                        )
                        if (
                            observed_bid != current_bids[0].price
                            or observed_ask != current_asks[0].price
                        ):
                            raise LiveCaptureError(
                                "BBO disagrees with reconstructed depth"
                            )
                    kind = (
                        "bbo"
                        if event_type == "best_bid_ask"
                        else "trade"
                    )
                    status = "CONTIGUOUS"
                else:
                    raise LiveCaptureError(
                        f"unsupported timed event_type: {event_type}"
                    )
                assert tick is not None
                self._books[token] = (current_bids, current_asks)
                self._tick_sizes[token] = tick
                prepared.append(
                    _PreparedMarketUpdate(
                        market_id=market.market_id,
                        token_id=token,
                        outcome=outcome,
                        update_kind=kind,
                        source_time_utc_ns=source_ns,
                        book_hash=(
                            event.get("hash")
                            if type(event.get("hash")) is str
                            else None
                        ),
                        bids=current_bids,
                        asks=current_asks,
                        tick_size=tick,
                        continuity_status=status,
                    )
                )
        return tuple(prepared)

    def ingest_timed_market_frame(
        self,
        *,
        native_payload: bytes,
        local_received_time: str,
        clock_context: CaptureClockContextV1,
        monotonic_clock: Callable[[], int],
    ) -> tuple[MarketUpdateV1, ...]:
        """Commit raw bytes, then emit linked timing and normalized book rows."""

        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        receive_ns = monotonic_clock()
        self._append_native(native_payload, local_received_time)
        append_done_ns = monotonic_clock()
        events = parse_market_frame(native_payload)
        parse_done_ns = monotonic_clock()
        prepared = self._prepare_updates(events)
        normalize_done_ns = monotonic_clock()
        ready_ns = monotonic_clock()
        if not (
            receive_ns
            <= append_done_ns
            <= parse_done_ns
            <= normalize_done_ns
            <= ready_ns
        ):
            raise LiveCaptureError(
                "monotonic capture boundaries moved backward"
            )
        ordinal = self._record_ordinal
        self._record_ordinal += 1
        payload_sha = "sha256:" + hashlib.sha256(
            native_payload
        ).hexdigest()
        timing = CaptureTimingV0(
            timing_version="v0",
            capture_session_id=(
                self.universe.binding.capture_session_id
            ),
            record_ordinal=ordinal,
            payload_sha256=payload_sha,
            receive_boundary="application_callback",
            local_receive_utc_ns=_utc_ns(
                local_received_time, "local_received_time"
            ),
            local_receive_monotonic_ns=receive_ns,
            pairing_uncertainty_ns=(
                clock_context.pairing_uncertainty_ns
            ),
            host_id=clock_context.host_id,
            boot_id=clock_context.boot_id,
            monotonic_clock_id=clock_context.monotonic_clock_id,
            clock_epoch_id=clock_context.clock_epoch_id,
            connection_epoch=self.epoch,
            clock_sync_sample_id=clock_context.clock_sync_sample_id,
            raw_append_done_monotonic_ns=append_done_ns,
            parse_done_monotonic_ns=parse_done_ns,
            normalize_done_monotonic_ns=normalize_done_ns,
            state_or_book_ready_monotonic_ns=ready_ns,
        )
        envelope_id = (
            f"{self.universe.binding.binding_id}:"
            f"{self.epoch}:{ordinal}"
        )
        updates = tuple(
            MarketUpdateV1(
                update_id=f"{envelope_id}:{index}",
                raw_envelope_id=envelope_id,
                game_id=self.universe.binding.game_id,
                venue="polymarket",
                logical_market_id=row.market_id,
                outcome=row.outcome,
                token_or_ticker=row.token_id,
                update_kind=row.update_kind,
                timing=timing,
                exchange_time_utc_ns=row.source_time_utc_ns,
                source_sequence=None,
                book_hash=row.book_hash,
                bids=row.bids,
                asks=row.asks,
                observed=True,
                derived=False,
                continuity_status=row.continuity_status,
                price_representation="TOKEN_OUTCOME",
                tick_size=row.tick_size,
            )
            for index, row in enumerate(prepared)
        )
        for update in updates:
            self._timed_gate.observe(update)
            if update.update_kind == "snapshot":
                self._snapshots.add(update.token_or_ticker)
                self._snapshots_by_epoch.setdefault(
                    self.epoch, set()
                ).add(update.token_or_ticker)
        source_times = {
            row.source_time_utc_ns for row in prepared
        }
        market_ids = {row.market_id for row in prepared}
        token_ids = {row.token_id for row in prepared}
        book_hashes = {
            row.book_hash
            for row in prepared
            if row.book_hash is not None
        }
        self._pending_timed_frames.append(
            _PendingTimedFrame(
                envelope_id=envelope_id,
                native_payload=native_payload,
                local_received_time=local_received_time,
                timing=timing,
                message_kind=(
                    "control"
                    if not prepared
                    else prepared[0].update_kind
                    if len({row.update_kind for row in prepared}) == 1
                    else "batch"
                ),
                source_time_utc_ns=(
                    next(iter(source_times))
                    if len(source_times) == 1
                    else None
                ),
                source_time_precision_ns=(
                    1_000_000 if len(source_times) == 1 else None
                ),
                book_hash=(
                    next(iter(book_hashes))
                    if len(book_hashes) == 1
                    else None
                ),
                market_id=(
                    next(iter(market_ids))
                    if len(market_ids) == 1
                    else None
                ),
                token_id=(
                    next(iter(token_ids))
                    if len(token_ids) == 1
                    else None
                ),
                updates=updates,
            )
        )
        self._market_updates.extend(updates)
        self._canonical_record_count += len(updates) + 1
        return updates

    def append_canonical_records(
        self, records: Sequence[ProspectiveCaptureRecord]
    ) -> None:
        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        if not records:
            self._control_frame_count += 1
            return
        for record in records:
            if record.epoch != self.epoch:
                raise CaptureContinuityError("record epoch mismatch")
            assert record.token_id is not None
            if (
                record.event_kind == "market_delta"
                and record.token_id not in self._snapshots
            ):
                self._delta_before_snapshot_count += 1
                raise CaptureContinuityError(
                    "market delta requires a new snapshot for its token "
                    f"in epoch {self.epoch}"
                )
            self.validator.observe(record)
            self._append_canonical(
                record.to_bytes(), record.local_received_time
            )
            self._canonical_record_count += 1
            if record.event_kind == "market_snapshot":
                self._snapshots.add(record.token_id)
                self._snapshots_by_epoch.setdefault(
                    self.epoch, set()
                ).add(record.token_id)

    def append_native_and_canonical(
        self,
        *,
        native_payload: bytes,
        local_received_time: str,
        records: Sequence[ProspectiveCaptureRecord],
    ) -> None:
        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        self.append_native_frame(
            native_payload=native_payload,
            local_received_time=local_received_time,
        )
        self.append_canonical_records(records)

    def expand_universe(self, universe: TargetUniverse) -> None:
        """Accept only an append-only expansion of the bound game universe."""

        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        if universe.binding != self.universe.binding:
            raise LiveCaptureError(
                "universe expansion changed the prospective game binding"
            )
        prior = {
            token: (
                market.market_id,
                market.condition_id,
                outcome,
            )
            for market in self.universe.markets
            for token, outcome in zip(
                market.token_ids, market.outcomes
            )
        }
        updated = {
            token: (
                market.market_id,
                market.condition_id,
                outcome,
            )
            for market in universe.markets
            for token, outcome in zip(
                market.token_ids, market.outcomes
            )
        }
        if (
            not set(prior).issubset(updated)
            or any(updated[token] != identity for token, identity in prior.items())
        ):
            raise LiveCaptureError(
                "universe expansion removed or reoriented an existing token"
            )
        if len(updated) <= len(prior):
            raise LiveCaptureError(
                "universe expansion must add at least one token"
            )
        self.universe = universe

    def start_new_epoch(
        self,
        *,
        reason: str,
        required_tokens: Sequence[str],
        audit_reason: str | None = None,
    ) -> int:
        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        required = tuple(required_tokens)
        if (
            not required
            or any(type(token) is not str or not token for token in required)
            or len(required) != len(set(required))
        ):
            raise ValueError(
                "required_tokens must be unique non-empty token IDs"
            )
        self._seal_native()
        if self._canonical is not None:
            self._manifests.append(self._canonical.seal())
            self._canonical = None
        epoch = self.validator.start_new_epoch(reason=reason)
        self._timed_gate.start_epoch(
            venue="polymarket", epoch=epoch
        )
        self._snapshots.clear()
        self._books.clear()
        self._tick_sizes.clear()
        self._required_tokens_by_epoch[epoch] = set(required)
        self._snapshots_by_epoch[epoch] = set()
        self._gap_count += 1
        self._snapshot_requirements += 1
        self._epoch_rows.append(
            {"epoch": epoch, "reason": audit_reason or reason}
        )
        return epoch

    def drain_sealed_manifests(self) -> tuple[SegmentManifest, ...]:
        rows = tuple(self._manifests[self._drain_index :])
        self._drain_index = len(self._manifests)
        return rows

    def close(self) -> tuple[SegmentManifest, ...]:
        if self._closed:
            raise CaptureContinuityError("capture state is closed")
        self._seal_native()
        if self._canonical is not None:
            self._manifests.append(self._canonical.seal())
            self._canonical = None
        self._closed = True
        return tuple(
            sorted(
                self._manifests,
                key=lambda item: (
                    item.partition_date,
                    item.partition_hour,
                    item.stream,
                    item.object_sha256,
                ),
            )
        )


def _artifact_key(
    *,
    prefix: str,
    kind: str,
    sha256: str,
    suffix: str,
) -> str:
    normalized = prefix.strip("/")
    if not normalized:
        raise LiveCaptureError("S3 prefix must be non-empty")
    digest = sha256.removeprefix("sha256:")
    if (
        len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise LiveCaptureError("artifact SHA-256 is invalid")
    return (
        f"{normalized}/{kind}/polymarket-x14/sha256/"
        f"{digest[:2]}/{digest}{suffix}"
    )


def _upload_verified(
    *,
    path: Path,
    key: str,
    sha256: str,
    byte_size: int,
    object_store: LiveObjectStore,
) -> dict[str, Any]:
    head = object_store.head_object(key)
    if head is not None and (
        head.get("byte_size") != byte_size
        or type(head.get("metadata")) is not dict
        or head["metadata"].get("sha256") != sha256
    ):
        raise LiveCaptureError("existing S3 artifact HEAD mismatch")
    status = "ALREADY_PRESENT"
    if head is None:
        object_store.put_file(
            key, path, metadata={"sha256": sha256}
        )
        status = "UPLOADED"
    verified = object_store.head_object(key)
    if (
        verified is None
        or verified.get("byte_size") != byte_size
        or type(verified.get("metadata")) is not dict
        or verified["metadata"].get("sha256") != sha256
    ):
        raise LiveCaptureError("post-upload S3 artifact HEAD mismatch")
    digest = hashlib.sha256()
    observed_size = 0
    for block in object_store.read_object_chunks(key):
        if type(block) is not bytes:
            raise LiveCaptureError("S3 readback yielded non-bytes")
        digest.update(block)
        observed_size += len(block)
    readback = "sha256:" + digest.hexdigest()
    if observed_size != byte_size or readback != sha256:
        raise LiveCaptureError("S3 artifact readback mismatch")
    return {
        "key": key,
        "sha256": sha256,
        "byte_size": byte_size,
        "status": status,
        "readback_verified": True,
    }


def upload_live_artifacts(
    *,
    manifests: Sequence[SegmentManifest],
    metadata_paths: Sequence[Path],
    report_paths: Sequence[Path],
    object_store: LiveObjectStore,
    prefix: str,
) -> dict[str, Any]:
    """Upload raw objects and every commit/audit sidecar with readback proof."""

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for manifest in manifests:
        verification = verify_segment(manifest.path, root=manifest.path.parents[5])
        if not verification.valid:
            raise LiveCaptureError(
                "local segment verification failed before S3 upload"
            )
        raw_key = _artifact_key(
            prefix=prefix,
            kind="raw",
            sha256=manifest.object_sha256,
            suffix=".jsonl.zst",
        )
        rows.append(
            {
                "artifact_kind": "raw",
                **_upload_verified(
                    path=manifest.object_path,
                    key=raw_key,
                    sha256=manifest.object_sha256,
                    byte_size=manifest.object_size_bytes,
                    object_store=object_store,
                ),
            }
        )
        counts["raw"] += 1
        manifest_sha, manifest_size = _sha256_path(manifest.path)
        manifest_key = _artifact_key(
            prefix=prefix,
            kind="manifests",
            sha256=manifest_sha,
            suffix=".manifest.json",
        )
        rows.append(
            {
                "artifact_kind": "sidecar_manifest",
                **_upload_verified(
                    path=manifest.path,
                    key=manifest_key,
                    sha256=manifest_sha,
                    byte_size=manifest_size,
                    object_store=object_store,
                ),
            }
        )
        counts["sidecar_manifest"] += 1
    for kind, paths in (
        ("metadata", metadata_paths),
        ("report", report_paths),
    ):
        for path in paths:
            sha256, byte_size = _sha256_path(path)
            key = _artifact_key(
                prefix=prefix,
                kind="metadata" if kind == "metadata" else "reports",
                sha256=sha256,
                suffix=path.suffix or ".bin",
            )
            rows.append(
                {
                    "artifact_kind": kind,
                    **_upload_verified(
                        path=path,
                        key=key,
                        sha256=sha256,
                        byte_size=byte_size,
                        object_store=object_store,
                    ),
                }
            )
            counts[kind] += 1
    return {
        "status": "VERIFIED",
        "counts": {
            "raw": counts["raw"],
            "sidecar_manifest": counts["sidecar_manifest"],
            "metadata": counts["metadata"],
            "report": counts["report"],
        },
        "objects": rows,
    }


def _publish_document(
    report_root: str | Path,
    document: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, str]:
    material = dict(document)
    digest = "sha256:" + hashlib.sha256(
        _canonical_bytes(material)
    ).hexdigest()
    payload = _canonical_bytes({**material, "document_sha256": digest}) + b"\n"
    hex_digest = digest.removeprefix("sha256:")
    target = (
        Path(report_root)
        / label
        / "sha256"
        / hex_digest[:2]
        / f"{hex_digest}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{label}-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise LiveCaptureError(
                    "content-addressed live document collision"
                )
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return target, digest


GammaFetcher = Callable[[], Awaitable[GammaHttpResponse]]
Connector = Callable[[Sequence[str]], Any]
Clock = Callable[[], str]
MonotonicClock = Callable[[], int]


async def run_live_capture(
    *,
    binding: ProspectiveGameBindingV1,
    gamma_fetch: GammaFetcher,
    connect: Connector,
    raw_root: str | Path,
    report_root: str | Path,
    program_root: str | Path,
    clock_context: CaptureClockContextV1,
    gamma_refresh_seconds: float = 300.0,
    receive_timeout_seconds: float = 30.0,
    max_runtime_seconds: float | None = None,
    reconnect_backoff_seconds: float = 1.0,
    receive_clock: Clock = _utc_now,
    monotonic_clock: MonotonicClock = time.monotonic_ns,
    object_store: LiveObjectStore | None = None,
    s3_prefix: str | None = None,
) -> LiveCaptureResult:
    """Capture until Gamma proves resolution or an explicit runtime cap.

    ``max_runtime_seconds`` exists for operational drills; omitting it is the
    actual market-creation-to-resolution mode.
    """

    if gamma_refresh_seconds <= 0 or receive_timeout_seconds <= 0:
        raise ValueError("refresh and receive timeout must be positive")
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive or null")
    if object_store is not None and s3_prefix is None:
        raise LiveCaptureError("configured S3 store requires a prefix")
    loop = asyncio.get_running_loop()
    deadline = (
        None
        if max_runtime_seconds is None
        else loop.time() + max_runtime_seconds
    )
    metadata_records: list[StaticObjectRecord] = []

    async def refresh() -> tuple[TargetRefreshInspection, StaticObjectRecord]:
        response = await gamma_fetch()
        record = preserve_target_metadata(
            response,
            binding=binding,
            raw_root=raw_root,
            program_root=program_root,
        )
        metadata_records.append(record)
        if object_store is not None:
            assert s3_prefix is not None
            upload_live_artifacts(
                manifests=(),
                metadata_paths=(
                    record.object_path,
                    record.manifest_path,
                ),
                report_paths=(),
                object_store=object_store,
                prefix=s3_prefix,
            )
        inspection = inspect_target_refresh(
            response, binding=binding
        )
        if inspection.unknown_markets:
            inventory_path, _ = _publish_document(
                report_root,
                {
                    "audit_version": "nfl-x14-unknown-market-inventory/v1",
                    "experiment_id": "X-14",
                    "fetched_at": response.fetched_at,
                    "markets": [
                        item.to_dict()
                        for item in inspection.unknown_markets
                    ],
                    "capture_started": False,
                },
                label="unknown-market-inventory",
            )
            if object_store is not None:
                assert s3_prefix is not None
                upload_live_artifacts(
                    manifests=(),
                    metadata_paths=(inventory_path,),
                    report_paths=(),
                    object_store=object_store,
                    prefix=s3_prefix,
                )
            raise UnknownMarketUniverseError(
                inspection.unknown_markets,
                inventory_path=inventory_path,
            )
        return inspection, record

    initial, _ = await refresh()
    universe = initial.require_supported_universe()
    capture_started_at = _utc_now()
    late_start = datetime.fromisoformat(
        capture_started_at[:-1] + "+00:00"
    ) > datetime.fromisoformat(
        universe.creation_at[:-1] + "+00:00"
    )
    state = LiveCaptureState(raw_root=raw_root, universe=universe)
    tokens = universe.token_ids
    resolution = initial.resolution_observed
    terminal_reason = (
        "resolution observed before WebSocket connection"
        if resolution
        else "capture not started"
    )
    next_refresh = loop.time() + gamma_refresh_seconds
    reconnect_reason = "initial_connect"

    def upload_sealed() -> None:
        sealed = state.drain_sealed_manifests()
        if not sealed or object_store is None:
            return
        assert s3_prefix is not None
        upload_live_artifacts(
            manifests=sealed,
            metadata_paths=(),
            report_paths=(),
            object_store=object_store,
            prefix=s3_prefix,
        )

    try:
        while not resolution:
            if deadline is not None and loop.time() >= deadline:
                terminal_reason = "max_runtime_seconds elapsed"
                break
            try:
                async with connect(tokens) as websocket:
                    await websocket.send(build_market_subscription(tokens))
                    while True:
                        now = loop.time()
                        if deadline is not None and now >= deadline:
                            terminal_reason = "max_runtime_seconds elapsed"
                            break
                        refresh_wait = max(0.0, next_refresh - now)
                        timeout = min(receive_timeout_seconds, refresh_wait)
                        if deadline is not None:
                            timeout = min(timeout, max(0.0, deadline - now))
                        if timeout <= 0:
                            inspection, _ = await refresh()
                            updated = inspection.require_supported_universe()
                            if inspection.resolution_observed:
                                resolution = True
                                terminal_reason = "Gamma resolution observed"
                                break
                            updated_tokens = updated.token_ids
                            if not set(tokens).issubset(updated_tokens):
                                raise LiveCaptureError(
                                    "Gamma removed a subscribed token before "
                                    "resolution"
                                )
                            next_refresh = (
                                loop.time() + gamma_refresh_seconds
                            )
                            if updated_tokens != tokens:
                                tokens = updated_tokens
                                state.expand_universe(updated)
                                universe = updated
                                reconnect_reason = "universe_expansion"
                                break
                            continue
                        try:
                            frame = await asyncio.wait_for(
                                websocket.recv(), timeout=timeout
                            )
                        except TimeoutError:
                            if loop.time() >= next_refresh:
                                continue
                            reconnect_reason = "receive_idle_timeout"
                            break
                        payload = exact_frame_bytes(frame)
                        local_time = receive_clock()
                        state.ingest_timed_market_frame(
                            native_payload=payload,
                            local_received_time=local_time,
                            clock_context=clock_context,
                            monotonic_clock=monotonic_clock,
                        )
                        upload_sealed()
                    if resolution or terminal_reason == (
                        "max_runtime_seconds elapsed"
                    ):
                        break
            except UnknownMarketUniverseError:
                raise
            except LiveCaptureError:
                # Identity, timestamp, token, and protocol violations are data
                # failures, not transient transport errors.
                raise
            except CaptureContinuityError:
                reconnect_reason = "sequence_gap"
                state.start_new_epoch(
                    reason="sequence_gap",
                    required_tokens=tokens,
                    audit_reason="delta_before_new_snapshot",
                )
                upload_sealed()
                await asyncio.sleep(reconnect_backoff_seconds)
                continue
            except Exception as error:
                reconnect_reason = (
                    f"reconnect_after_{type(error).__name__}"
                )

            if resolution or terminal_reason == "max_runtime_seconds elapsed":
                break
            state.start_new_epoch(
                reason="reconnect",
                required_tokens=tokens,
                audit_reason=reconnect_reason,
            )
            upload_sealed()
            await asyncio.sleep(reconnect_backoff_seconds)
    finally:
        manifests = state.close()
        upload_sealed()

    metadata_paths = tuple(
        path
        for record in metadata_records
        for path in (record.object_path, record.manifest_path)
    )
    preliminary_upload: Mapping[str, Any]
    if object_store is None:
        preliminary_upload = {
            "status": "NOT_CONFIGURED",
            "counts": {},
            "objects": [],
        }
    else:
        assert s3_prefix is not None
        preliminary_upload = upload_live_artifacts(
            manifests=manifests,
            metadata_paths=metadata_paths,
            report_paths=(),
            object_store=object_store,
            prefix=s3_prefix,
        )
    report = {
        "report_version": "nfl-x14-live-capture/v1",
        "experiment_id": "X-14",
        "session_id": universe.binding.capture_session_id,
        "binding_id": universe.binding.binding_id,
        "game_id": universe.binding.game_id,
        "generated_at": _utc_now(),
        "terminal_reason": terminal_reason,
        "resolution_observed": resolution,
        "coverage": {
            "required_window": "market_creation_to_resolution",
            "market_creation_at": universe.creation_at,
            "local_recorder_started_at": capture_started_at,
            "status": (
                "PARTIAL_LATE_START"
                if late_start
                else "STARTED_BY_CREATION"
            ),
            "complete_from_creation": not late_start,
            "unobserved_pre_recorder_l2": late_start,
        },
        "subscribed_token_ids": list(tokens),
        "metadata_refresh_count": len(metadata_records),
        "segment_count": len(manifests),
        "continuity": state.audit,
        "s3_capture_upload": preliminary_upload,
        "claims": {
            "creation_to_resolution_complete": (
                resolution
                and not late_start
                and state.audit["continuity_complete"]
            ),
            "local_start_to_resolution_complete": (
                resolution and state.audit["continuity_complete"]
            ),
            "sports_feed_complete": False,
            "feed_to_market_latency_measured": False,
            "alpha_or_execution": False,
        },
    }
    report_path, report_sha = _publish_document(
        report_root, report, label="sessions"
    )
    if object_store is None:
        final_upload = preliminary_upload
    else:
        assert s3_prefix is not None
        final_upload = upload_live_artifacts(
            manifests=(),
            metadata_paths=(),
            report_paths=(report_path,),
            object_store=object_store,
            prefix=s3_prefix,
        )
    return LiveCaptureResult(
        terminal_reason=terminal_reason,
        resolution_observed=resolution,
        manifests=manifests,
        metadata_records=tuple(metadata_records),
        report_path=report_path,
        report_sha256=report_sha,
        s3_upload={
            "capture": preliminary_upload,
            "report": final_upload,
        },
        audit=state.audit,
    )


async def _heartbeat(websocket: Any) -> None:
    while True:
        await asyncio.sleep(10)
        await websocket.send("PING")


def public_connector(*, open_timeout_seconds: float = 10.0) -> Connector:
    """Build the unauthenticated market-channel connector used by the CLI."""

    if open_timeout_seconds <= 0:
        raise ValueError("open_timeout_seconds must be positive")

    @asynccontextmanager
    async def connection(_tokens: Sequence[str]) -> Any:
        async with websocket_connect(
            MARKET_WS_URL,
            open_timeout=open_timeout_seconds,
            close_timeout=5,
            ping_interval=None,
            max_size=16 * 1024 * 1024,
        ) as websocket:
            task = asyncio.create_task(_heartbeat(websocket))
            try:
                yield websocket
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return connection


def build_live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m prediction_market.sports."
            "nfl_x14_live_recorder"
        ),
        description=(
            "Capture registry-bound X-14 NFL Polymarket from the Gamma "
            "universe until resolution."
        ),
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--program-root", type=Path, default=Path("."))
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--monotonic-clock-id", required=True)
    parser.add_argument("--clock-epoch-id", required=True)
    parser.add_argument("--clock-sync-sample-id", required=True)
    parser.add_argument(
        "--pairing-uncertainty-ns", type=int, required=True
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--gamma-refresh-seconds", type=float, default=300.0
    )
    parser.add_argument(
        "--receive-timeout-seconds", type=float, default=30.0
    )
    parser.add_argument(
        "--reconnect-backoff-seconds", type=float, default=1.0
    )
    parser.add_argument("--http-timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        help=(
            "Operational drill cap; omit for the formal until-resolution mode."
        ),
    )
    return parser


async def _run_network(args: argparse.Namespace) -> LiveCaptureResult:
    binding = load_prospective_game_binding(args.binding)
    clock_context = CaptureClockContextV1(
        host_id=args.host_id,
        boot_id=args.boot_id,
        monotonic_clock_id=args.monotonic_clock_id,
        clock_epoch_id=args.clock_epoch_id,
        clock_sync_sample_id=args.clock_sync_sample_id,
        pairing_uncertainty_ns=args.pairing_uncertainty_ns,
    )

    async def gamma_fetcher() -> GammaHttpResponse:
        def fetch() -> GammaHttpResponse:
            with httpx.Client(timeout=args.http_timeout_seconds) as client:
                return fetch_target_event(
                    client=client, binding=binding
                )

        return await asyncio.to_thread(fetch)

    try:
        object_store, prefix = configured_object_store(
            env_file=args.env_file
        )
    except DataStoreError as error:
        if args.env_file is not None:
            raise LiveCaptureError(
                f"explicit S3 configuration is invalid: {error}"
            ) from error
        object_store = None
        prefix = None
    return await run_live_capture(
        binding=binding,
        gamma_fetch=gamma_fetcher,
        connect=public_connector(),
        raw_root=args.raw_root,
        report_root=args.report_root,
        program_root=args.program_root,
        clock_context=clock_context,
        gamma_refresh_seconds=args.gamma_refresh_seconds,
        receive_timeout_seconds=args.receive_timeout_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
        reconnect_backoff_seconds=args.reconnect_backoff_seconds,
        object_store=object_store,
        s3_prefix=prefix,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_live_parser().parse_args(argv)
    try:
        result = asyncio.run(_run_network(args))
    except (
        DataStoreError,
        LiveCaptureError,
        OSError,
        TimeoutError,
        X14PolymarketRecorderError,
    ) as error:
        print(f"X-14 live recorder failed: {error}", file=sys.stderr)
        return 2
    print(result.report_path)
    return 0 if result.resolution_observed else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiveCaptureError",
    "LiveCaptureResult",
    "LiveCaptureState",
    "TargetRefreshInspection",
    "UnknownMarketUniverseError",
    "UnknownTargetMarket",
    "inspect_target_refresh",
    "build_live_parser",
    "main",
    "parse_live_market_frame",
    "public_connector",
    "run_live_capture",
    "upload_live_artifacts",
]
