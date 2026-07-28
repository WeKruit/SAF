"""Fail-closed PMXT Level-2 microstructure calculations.

PMXT Level-2 events describe visible book states.  They do not authoritatively
identify executions, order owners, cancellations, or aggressors.  This module
therefore computes book-derived measures only and keeps formal aggressor,
cancel-intensity, and markout outputs unavailable unless a caller separately
supplies the required authoritative inputs to the corresponding pure function.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from prediction_market.pmxt.reconstructor import PMXTValidationError, canonicalize_event


_FORMAL_AGGRESSOR_UNAVAILABLE = "unavailable_no_authoritative_fill"
_METADATA_ONLY_SCOPE = "metadata_only_no_raw_scan"
PHASE1_SAMPLE_MEASUREMENT_SHA256 = (
    "sha256:95b2e0426dc631f53f7ac80c02e2c1e91dbc62ae91d29097ab5aa56def77230b"
)


class L2MethodError(ValueError):
    """Raised when Level-2 input cannot support a non-guessing calculation."""


def method_audit_fixture_events() -> tuple[Mapping[str, object], ...]:
    """Return the frozen method fixture with real gap and reconnect markers."""

    common = {"market": "market-1", "asset_id": "yes-1"}

    def event(
        event_type: str, receive_at: str, **fields: object
    ) -> Mapping[str, object]:
        return {
            "timestamp_received": receive_at,
            "timestamp": receive_at,
            **common,
            "event_type": event_type,
            **fields,
        }

    return (
        event(
            "book",
            "2026-07-01T00:00:00.000Z",
            bids=[["0.48", "100"], ["0.47", "20"]],
            asks=[["0.52", "50"], ["0.53", "30"]],
        ),
        event(
            "price_change",
            "2026-07-01T00:00:01.000Z",
            price="0.48",
            size="60",
            side="BUY",
        ),
        event(
            "price_change",
            "2026-07-01T00:00:02.000Z",
            price="0.52",
            size="0",
            side="SELL",
        ),
        event("gap", "2026-07-01T00:00:03.000Z"),
        event(
            "book",
            "2026-07-01T00:00:04.000Z",
            bids=[["0.46", "40"], ["0.45", "10"]],
            asks=[["0.54", "30"], ["0.55", "20"]],
        ),
        event(
            "price_change",
            "2026-07-01T00:00:05.000Z",
            price="0.46",
            size="35",
            side="BUY",
        ),
        event("reconnect", "2026-07-01T00:00:06.000Z"),
        event(
            "book",
            "2026-07-01T00:00:07.000Z",
            bids=[["0.475", "50"]],
            asks=[["0.525", "50"]],
        ),
    )


class MetricBasis(str, Enum):
    """Explicit price/depth basis for every basis-sensitive calculation."""

    BID = "bid"
    ASK = "ask"
    MIDPOINT = "midpoint"


@dataclass(frozen=True)
class AuditedMetricV1:
    """Pure calculation result carrying its basis and exact inputs."""

    method: str
    basis: MetricBasis
    inputs: Mapping[str, Decimal | str]
    value: Decimal | None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "basis": self.basis.value,
            "inputs": {
                key: _decimal_text(value) if isinstance(value, Decimal) else value
                for key, value in self.inputs.items()
            },
            "value": _decimal_text(self.value),
        }


@dataclass(frozen=True)
class BookWalkV1:
    """Visible-depth walk, not an assertion that an execution occurred."""

    average_price: Decimal | None
    filled_size: Decimal
    unfilled_size: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            "average_price": _decimal_text(self.average_price),
            "filled_size": _decimal_text(self.filled_size),
            "unfilled_size": _decimal_text(self.unfilled_size),
        }


@dataclass(frozen=True)
class TickMetricV1:
    """One reconstructed visible-book observation within a snapshot epoch."""

    epoch: int
    market: str
    asset_id: str
    receive_at: str
    source_at: str
    event_type: str
    best_bid: Decimal | None
    best_bid_size: Decimal | None
    best_ask: Decimal | None
    best_ask_size: Decimal | None
    midpoint: Decimal | None
    spread: Decimal | None
    bid_depth_top_n: Decimal
    ask_depth_top_n: Decimal
    imbalance: Decimal | None
    microprice: Decimal | None
    ofi: Decimal | None
    bid_depth_withdrawal: Decimal | None
    ask_depth_withdrawal: Decimal | None
    cancel_intensity_by_basis: Mapping[str, Decimal | None]
    resilience_by_basis: Mapping[str, Decimal | None]
    markout_by_basis: Mapping[str, Decimal | None]
    formal_aggressor_status: str = _FORMAL_AGGRESSOR_UNAVAILABLE

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready record without an inferred aggressor field."""

        return {
            "schema_version": "PMXTL2TickMetricV1",
            "epoch": self.epoch,
            "market": self.market,
            "asset_id": self.asset_id,
            "receive_at": self.receive_at,
            "source_at": self.source_at,
            "event_type": self.event_type,
            "best_bid": _decimal_text(self.best_bid),
            "best_bid_size": _decimal_text(self.best_bid_size),
            "best_ask": _decimal_text(self.best_ask),
            "best_ask_size": _decimal_text(self.best_ask_size),
            "midpoint": _decimal_text(self.midpoint),
            "spread": _decimal_text(self.spread),
            "bid_depth_top_n": _decimal_text(self.bid_depth_top_n),
            "ask_depth_top_n": _decimal_text(self.ask_depth_top_n),
            "imbalance": _decimal_text(self.imbalance),
            "microprice": _decimal_text(self.microprice),
            "ofi": _decimal_text(self.ofi),
            "bid_depth_withdrawal": _decimal_text(self.bid_depth_withdrawal),
            "ask_depth_withdrawal": _decimal_text(self.ask_depth_withdrawal),
            "cancel_intensity_by_basis": _basis_values(self.cancel_intensity_by_basis),
            "resilience_by_basis": _basis_values(self.resilience_by_basis),
            "markout_by_basis": _basis_values(self.markout_by_basis),
            "formal_aggressor_status": self.formal_aggressor_status,
        }


@dataclass(frozen=True)
class TickDataInventoryRowV1:
    """Metadata-only PMXT tick-input inventory row."""

    row_kind: str
    link_status: str
    object_sha256: str | None
    filename: str | None
    hour: str | None
    size_bytes: int
    phase0_reported_size_bytes: int | None
    phase0_size_matches_verified_bytes: bool | None
    remote_verification_status: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "TickDataInventoryV1",
            "source": "pmxt-v2",
            "row_kind": self.row_kind,
            "link_status": self.link_status,
            "object_sha256": self.object_sha256,
            "filename": self.filename,
            "hour": self.hour,
            "size_bytes": self.size_bytes,
            "phase0_reported_size_bytes": self.phase0_reported_size_bytes,
            "phase0_size_matches_verified_bytes": (
                self.phase0_size_matches_verified_bytes
            ),
            "remote_verification_status": self.remote_verification_status,
            "access_scope": _METADATA_ONLY_SCOPE,
        }


@dataclass
class _BookState:
    epoch: int
    bids: dict[Decimal, Decimal]
    asks: dict[Decimal, Decimal]


@dataclass(frozen=True)
class _BookView:
    best_bid: Decimal | None
    best_bid_size: Decimal | None
    best_ask: Decimal | None
    best_ask_size: Decimal | None
    midpoint: Decimal | None
    spread: Decimal | None
    bid_depth: Decimal
    ask_depth: Decimal
    imbalance: Decimal | None
    microprice: Decimal | None


@dataclass
class _GateEvidence:
    snapshot_count: int = 0
    accepted_delta_count: int = 0
    gap_invalidation_count: int = 0
    gap_resnapshot_count: int = 0
    gap_epoch_increment_count: int = 0
    reconnect_invalidation_count: int = 0
    reconnect_resnapshot_count: int = 0
    reconnect_epoch_increment_count: int = 0


@dataclass(frozen=True)
class _BuildResult:
    ticks: tuple[TickMetricV1, ...]
    gates: _GateEvidence


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise L2MethodError(f"{field} must be a decimal string, integer, or Decimal")
    if not isinstance(value, (str, int, Decimal)):
        raise L2MethodError(f"{field} must be a decimal string, integer, or Decimal")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise L2MethodError(f"{field} is not a valid decimal") from exc
    if not parsed.is_finite():
        raise L2MethodError(f"{field} must be finite")
    return parsed


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _basis_values(values: Mapping[str, Decimal | None]) -> dict[str, str | None]:
    return {basis: _decimal_text(value) for basis, value in values.items()}


def _metric_basis(value: object) -> MetricBasis:
    if isinstance(value, MetricBasis):
        return value
    try:
        return MetricBasis(value)
    except (TypeError, ValueError) as exc:
        raise L2MethodError("basis must be bid, ask, or midpoint") from exc


def _require_positive(value: object, *, field: str) -> Decimal:
    parsed = _decimal(value, field=field)
    if parsed <= 0:
        raise L2MethodError(f"{field} must be positive")
    return parsed


def _require_nonnegative(value: object, *, field: str) -> Decimal:
    parsed = _decimal(value, field=field)
    if parsed < 0:
        raise L2MethodError(f"{field} must be non-negative")
    return parsed


def _normalise_side(side: object) -> str:
    if not isinstance(side, str):
        raise L2MethodError("side must be BUY or SELL")
    normalized = side.upper()
    if normalized not in {"BUY", "SELL"}:
        raise L2MethodError("side must be BUY or SELL")
    return normalized


def _normalise_levels(
    levels: object,
    *,
    field: str,
    descending: bool,
    allow_zero: bool,
) -> dict[Decimal, Decimal]:
    if isinstance(levels, str):
        try:
            levels = json.loads(levels, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise L2MethodError(f"{field} is not valid JSON") from exc
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes, bytearray)):
        raise L2MethodError(f"{field} must be a sequence of [price, size] levels")

    normalized: dict[Decimal, Decimal] = {}
    for index, level in enumerate(levels):
        if isinstance(level, Mapping):
            price_value = level.get("price")
            size_value = level.get("size")
        elif (
            isinstance(level, Sequence)
            and not isinstance(level, (str, bytes, bytearray))
            and len(level) == 2
        ):
            price_value, size_value = level
        else:
            raise L2MethodError(f"{field}[{index}] must be a [price, size] pair")
        price = _require_positive(price_value, field=f"{field}[{index}].price")
        size = _require_nonnegative(size_value, field=f"{field}[{index}].size")
        if size == 0 and not allow_zero:
            raise L2MethodError(f"{field}[{index}].size must be positive")
        if price in normalized:
            raise L2MethodError(f"{field} contains duplicate price {price}")
        if size > 0:
            normalized[price] = size
    return dict(sorted(normalized.items(), reverse=descending))


def _view(state: _BookState, *, top_n: int) -> _BookView:
    bids = tuple(sorted(state.bids.items(), reverse=True))[:top_n]
    asks = tuple(sorted(state.asks.items()))[:top_n]
    best_bid, best_bid_size = (bids[0] if bids else (None, None))
    best_ask, best_ask_size = (asks[0] if asks else (None, None))
    if best_bid is not None and best_ask is not None and best_bid >= best_ask:
        raise L2MethodError("crossed or locked book cannot produce microstructure metrics")

    bid_depth = sum((size for _, size in bids), start=Decimal("0"))
    ask_depth = sum((size for _, size in asks), start=Decimal("0"))
    midpoint = (
        (best_bid + best_ask) / Decimal("2")
        if best_bid is not None and best_ask is not None
        else None
    )
    spread = (
        best_ask - best_bid
        if best_bid is not None and best_ask is not None
        else None
    )
    denominator = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / denominator if denominator else None
    microprice = (
        (best_ask * best_bid_size + best_bid * best_ask_size)
        / (best_bid_size + best_ask_size)
        if best_bid is not None
        and best_ask is not None
        and best_bid_size is not None
        and best_ask_size is not None
        and best_bid_size + best_ask_size
        else None
    )
    return _BookView(
        best_bid=best_bid,
        best_bid_size=best_bid_size,
        best_ask=best_ask,
        best_ask_size=best_ask_size,
        midpoint=midpoint,
        spread=spread,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        imbalance=imbalance,
        microprice=microprice,
    )


def _order_flow_imbalance(previous: _BookView | None, current: _BookView) -> Decimal | None:
    if (
        previous is None
        or previous.best_bid is None
        or previous.best_bid_size is None
        or previous.best_ask is None
        or previous.best_ask_size is None
        or current.best_bid is None
        or current.best_bid_size is None
        or current.best_ask is None
        or current.best_ask_size is None
    ):
        return None

    bid_change = (
        (current.best_bid_size if current.best_bid >= previous.best_bid else Decimal("0"))
        - (previous.best_bid_size if current.best_bid <= previous.best_bid else Decimal("0"))
    )
    ask_change = (
        (current.best_ask_size if current.best_ask <= previous.best_ask else Decimal("0"))
        - (previous.best_ask_size if current.best_ask >= previous.best_ask else Decimal("0"))
    )
    return bid_change - ask_change


def _withdrawal(previous_depth: Decimal | None, current_depth: Decimal) -> Decimal | None:
    if previous_depth is None:
        return None
    return max(previous_depth - current_depth, Decimal("0"))


def _prepare_events(events: Iterable[Mapping[str, object]]) -> list[dict[str, Any]]:
    prepared: list[tuple[tuple[str, str, str, str, str], dict[str, Any]]] = []
    for event in events:
        _validate_l2_schema_boundary(event)
        try:
            canonical = canonicalize_event(event)
        except PMXTValidationError as exc:
            raise L2MethodError(str(exc)) from exc
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        prepared.append(
            (
                (
                    canonical["timestamp_received"],
                    canonical["timestamp"],
                    canonical["market"],
                    canonical["asset_id"],
                    digest,
                ),
                canonical,
            )
        )
    return [event for _, event in sorted(prepared, key=lambda item: item[0])]


def _validate_l2_schema_boundary(event: Mapping[str, object]) -> None:
    lowered_keys = {str(key).lower() for key in event}
    if any("nfl" in key and "alpha" in key for key in lowered_keys):
        raise L2MethodError("schema boundary rejects NFL alpha fields")
    if any(key == "l3" or key.startswith("l3_") for key in lowered_keys):
        raise L2MethodError("schema boundary rejects Level-3 fields")
    if "aggressor_side" in lowered_keys:
        raise L2MethodError("schema boundary rejects inferred aggressor claims")
    positive_claims = {
        "authoritative_fill",
        "fill_reconstructed",
        "queue_fill_reconstructed",
    }
    if any(event.get(key) is True for key in positive_claims):
        raise L2MethodError("schema boundary rejects positive fill claims")
    event_type = event.get("event_type")
    if isinstance(event_type, str) and event_type.lower() in {
        "fill",
        "trade",
        "last_trade_price",
    }:
        raise L2MethodError("schema boundary rejects fill/trade events as Level-2 state")


def build_l2_metric_ticks(
    events: Iterable[Mapping[str, object]], *, top_n: int = 5
) -> tuple[TickMetricV1, ...]:
    """Rebuild visible-book ticks with snapshot and gap gates.

    A delta is valid only after a book snapshot for its ``(market, asset_id)``.
    A ``gap`` invalidates that state; only a later snapshot creates the next
    epoch.  The function never treats a ``price_change`` side as a trade side.
    """

    return _build_l2_metric_result(events, top_n=top_n).ticks


def _build_l2_metric_result(
    events: Iterable[Mapping[str, object]], *, top_n: int = 5
) -> _BuildResult:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise L2MethodError("top_n must be a positive integer")

    states: dict[tuple[str, str], _BookState] = {}
    invalid_after_gap: set[tuple[str, str]] = set()
    next_epoch: dict[tuple[str, str], int] = {}
    pending_invalidation: dict[tuple[str, str], tuple[str, int]] = {}
    gates = _GateEvidence()
    ticks: list[TickMetricV1] = []

    for event in _prepare_events(events):
        market = event["market"]
        asset_id = event["asset_id"]
        event_type = event["event_type"]
        key = (market, asset_id)

        if event_type in {"gap", "reconnect"}:
            previous_epoch = next_epoch.get(key, 0)
            states.pop(key, None)
            invalid_after_gap.add(key)
            pending_invalidation[key] = (event_type, previous_epoch)
            if event_type == "gap":
                gates.gap_invalidation_count += 1
            else:
                gates.reconnect_invalidation_count += 1
            continue

        state = states.get(key)
        prior_view = _view(state, top_n=top_n) if state is not None else None
        if event_type in {"book", "snapshot"}:
            if "bids" not in event or "asks" not in event:
                raise L2MethodError("book snapshot requires bids and asks")
            epoch = next_epoch.get(key, 0) + 1 if state is None else state.epoch
            next_epoch[key] = epoch
            gates.snapshot_count += 1
            state = _BookState(
                epoch=epoch,
                bids=_normalise_levels(
                    event["bids"], field="bids", descending=True, allow_zero=False
                ),
                asks=_normalise_levels(
                    event["asks"], field="asks", descending=False, allow_zero=False
                ),
            )
            states[key] = state
            invalid_after_gap.discard(key)
            invalidation = pending_invalidation.pop(key, None)
            if invalidation is not None:
                kind, previous_epoch = invalidation
                incremented_once = epoch == previous_epoch + 1
                if kind == "gap":
                    gates.gap_resnapshot_count += 1
                    gates.gap_epoch_increment_count += int(incremented_once)
                else:
                    gates.reconnect_resnapshot_count += 1
                    gates.reconnect_epoch_increment_count += int(incremented_once)
        elif event_type == "price_change":
            if state is None or key in invalid_after_gap:
                raise L2MethodError("price_change delta requires a snapshot")
            if "price" not in event or "size" not in event or "side" not in event:
                raise L2MethodError("price_change requires price, size, and side")
            price = _require_positive(event["price"], field="price")
            size = _require_nonnegative(event["size"], field="size")
            levels = state.bids if _normalise_side(event["side"]) == "BUY" else state.asks
            if size == 0:
                levels.pop(price, None)
            else:
                levels[price] = size
            gates.accepted_delta_count += 1
        else:
            raise L2MethodError(f"unsupported Level-2 event type: {event_type}")

        current_view = _view(state, top_n=top_n)
        ticks.append(
            TickMetricV1(
                epoch=state.epoch,
                market=market,
                asset_id=asset_id,
                receive_at=event["timestamp_received"],
                source_at=event["timestamp"],
                event_type=event_type,
                best_bid=current_view.best_bid,
                best_bid_size=current_view.best_bid_size,
                best_ask=current_view.best_ask,
                best_ask_size=current_view.best_ask_size,
                midpoint=current_view.midpoint,
                spread=current_view.spread,
                bid_depth_top_n=current_view.bid_depth,
                ask_depth_top_n=current_view.ask_depth,
                imbalance=current_view.imbalance,
                microprice=current_view.microprice,
                ofi=_order_flow_imbalance(prior_view, current_view),
                bid_depth_withdrawal=_withdrawal(
                    prior_view.bid_depth if prior_view else None, current_view.bid_depth
                ),
                ask_depth_withdrawal=_withdrawal(
                    prior_view.ask_depth if prior_view else None, current_view.ask_depth
                ),
                cancel_intensity_by_basis={"bid": None, "ask": None},
                resilience_by_basis={"bid": None, "ask": None, "midpoint": None},
                markout_by_basis={"bid": None, "ask": None, "midpoint": None},
            )
        )

    return _BuildResult(ticks=tuple(ticks), gates=gates)


def cancel_intensity(
    *,
    basis: MetricBasis | str,
    cancelled_size: object,
    observed_depth: object,
) -> AuditedMetricV1:
    """Return a cancellation ratio only from authoritative cancellation data."""

    normalized_basis = _metric_basis(basis)
    if normalized_basis is MetricBasis.MIDPOINT:
        raise L2MethodError("cancel intensity basis must be bid or ask")
    cancelled = _require_nonnegative(cancelled_size, field="cancelled_size")
    depth = _require_positive(observed_depth, field="observed_depth")
    return AuditedMetricV1(
        method="cancel_intensity",
        basis=normalized_basis,
        inputs={"cancelled_size": cancelled, "observed_depth": depth},
        value=cancelled / depth,
    )


def resilience_fraction(
    *,
    basis: MetricBasis | str,
    baseline_reference: object,
    shocked_reference: object,
    recovered_reference: object,
) -> AuditedMetricV1:
    """Measure recovery on one explicit bid/ask/midpoint basis."""

    normalized_basis = _metric_basis(basis)
    baseline = _decimal(baseline_reference, field="baseline_reference")
    shocked = _decimal(shocked_reference, field="shocked_reference")
    recovered = _decimal(recovered_reference, field="recovered_reference")
    shock_distance = abs(shocked - baseline)
    value = (
        None
        if shock_distance == 0
        else Decimal("1") - abs(recovered - baseline) / shock_distance
    )
    return AuditedMetricV1(
        method="resilience_fraction",
        basis=normalized_basis,
        inputs={
            "baseline_reference": baseline,
            "shocked_reference": shocked,
            "recovered_reference": recovered,
        },
        value=value,
    )


def signed_markout(
    *,
    side: object,
    basis: MetricBasis | str,
    execution_price: object,
    future_reference: object,
) -> AuditedMetricV1:
    """Return per-unit markout for an explicitly authoritative execution."""

    normalized_basis = _metric_basis(basis)
    normalized_side = _normalise_side(side)
    execution = _decimal(execution_price, field="execution_price")
    future = _decimal(future_reference, field="future_reference")
    value = future - execution if normalized_side == "BUY" else execution - future
    return AuditedMetricV1(
        method="signed_markout",
        basis=normalized_basis,
        inputs={
            "side": normalized_side,
            "execution_price": execution,
            "future_reference": future,
        },
        value=value,
    )


def walk_book(
    *, side: object, levels: object, requested_size: object
) -> BookWalkV1:
    """Walk visible depth for a hypothetical request without inferring a fill."""

    requested = _require_positive(requested_size, field="requested_size")
    normalized_side = _normalise_side(side)
    book = _normalise_levels(
        levels,
        field="levels",
        descending=normalized_side == "SELL",
        allow_zero=False,
    )
    remaining = requested
    filled = Decimal("0")
    notional = Decimal("0")
    for price, available in book.items():
        used = min(remaining, available)
        notional += price * used
        filled += used
        remaining -= used
        if remaining == 0:
            break
    return BookWalkV1(
        average_price=notional / filled if filled else None,
        filled_size=filled,
        unfilled_size=remaining,
    )


def parity_deviation(yes_midpoint: object, no_midpoint: object) -> Decimal:
    """Return ``yes + no - 1`` for explicitly paired complementary contracts."""

    yes = _decimal(yes_midpoint, field="yes_midpoint")
    no = _decimal(no_midpoint, field="no_midpoint")
    return yes + no - Decimal("1")


def build_tick_data_inventory_v1_rows(
    remote_inventory: Mapping[str, object],
    phase0_inventory: Mapping[str, object],
    *,
    authoritative_manifests: Iterable[Mapping[str, object]] = (),
    phase1_sample_measurement: Mapping[str, object] | None = None,
    phase1_measurement_sha256: str | None = None,
) -> tuple[TickDataInventoryRowV1, ...]:
    """Build rows through an exact SHA-256 manifest crosswalk only.

    The remote inventory is content-addressed and Phase 0 is filename/hour
    addressed.  They are connected only when an authoritative local manifest
    supplies both identities.  Remote objects or Phase-0 entries without that
    chain remain explicitly unattested; neither byte size nor position is used
    as a substitute.  No raw parquet object is opened.
    """

    remote_objects = _inventory_objects(remote_inventory, field="remote_inventory")
    phase0_objects = _inventory_objects(phase0_inventory, field="phase0_inventory")
    phase0_by_filename: dict[str, Mapping[str, object]] = {}
    for item in phase0_objects:
        filename = _inventory_text(item.get("filename"), field="phase0 filename")
        if filename in phase0_by_filename:
            raise L2MethodError(f"duplicate Phase-0 filename: {filename}")
        phase0_by_filename[filename] = item

    manifests_by_sha = _manifest_index(authoritative_manifests)
    phase1_by_sha = _phase1_evidence_index(
        phase1_sample_measurement,
        measurement_sha256=phase1_measurement_sha256,
    )
    manifest_filenames = {
        _manifest_filename(manifest) for manifest in manifests_by_sha.values()
    }
    phase1_filenames = {
        _inventory_text(record["filename"], field="Phase-1 filename")
        for record in phase1_by_sha.values()
    }
    manifest_sha_by_filename = {
        _manifest_filename(manifest): object_sha
        for object_sha, manifest in manifests_by_sha.items()
    }
    for object_sha, record in phase1_by_sha.items():
        filename = _inventory_text(record["filename"], field="Phase-1 filename")
        manifest_sha = manifest_sha_by_filename.get(filename)
        if manifest_sha is not None and manifest_sha != object_sha:
            raise L2MethodError(
                "authoritative crosswalk sources conflict for the same filename"
            )
    verified_remote_shas = {
        _inventory_text(item.get("sha256"), field="remote sha256")
        for item in remote_objects
        if item.get("status") == "verified"
    }

    rows: list[TickDataInventoryRowV1] = []
    seen_shas: set[str] = set()
    for item in remote_objects:
        status = item.get("status")
        if status != "verified":
            continue
        object_sha = _inventory_text(item.get("sha256"), field="remote sha256")
        actual_sha = item.get("actual_sha256")
        if actual_sha is not None and actual_sha != object_sha:
            raise L2MethodError("verified remote object SHA-256 does not match actual SHA-256")
        if object_sha in seen_shas:
            raise L2MethodError(f"duplicate verified remote SHA-256: {object_sha}")
        seen_shas.add(object_sha)
        size = _inventory_size(item, field="remote_inventory.objects")
        manifest = manifests_by_sha.get(object_sha)
        if manifest is None:
            phase1_record = phase1_by_sha.get(object_sha)
            if phase1_record is None:
                rows.append(
                    TickDataInventoryRowV1(
                        row_kind="REMOTE_VERIFIED_OBJECT",
                        link_status="UNATTESTED_NO_CROSSWALK",
                        object_sha256=object_sha,
                        filename=None,
                        hour=None,
                        size_bytes=size,
                        phase0_reported_size_bytes=None,
                        phase0_size_matches_verified_bytes=None,
                        remote_verification_status="verified",
                    )
                )
                continue
            phase1_size = phase1_record["byte_size"]
            if not isinstance(phase1_size, int) or phase1_size != size:
                raise L2MethodError(
                    "remote verified byte size does not match Phase-1 evidence"
                )
            filename = _inventory_text(
                phase1_record["filename"], field="Phase-1 filename"
            )
            phase0 = phase0_by_filename.get(filename)
            if phase0 is None:
                raise L2MethodError(
                    "Phase-1 evidence filename is absent from Phase-0 inventory"
                )
            phase0_hour = _inventory_text(phase0.get("hour"), field="phase0 hour")
            evidence_hour = _inventory_text(
                phase1_record["hour"], field="Phase-1 hour"
            )
            if phase0_hour != evidence_hour:
                raise L2MethodError("Phase-1 evidence hour does not match Phase-0")
            phase0_size = _inventory_size(
                phase0, field="phase0_inventory.objects"
            )
            evidence_phase0_size = phase1_record["phase0_size_bytes"]
            if (
                not isinstance(evidence_phase0_size, int)
                or evidence_phase0_size != phase0_size
            ):
                raise L2MethodError(
                    "Phase-1 inventory object size does not match Phase-0"
                )
            phase0_url = phase0.get("url")
            if (
                phase0_url is not None
                and phase0_url
                != _inventory_text(
                    phase1_record["source_url"], field="Phase-1 source URL"
                )
            ):
                raise L2MethodError(
                    "Phase-1 evidence source URL does not match Phase-0"
                )
            rows.append(
                TickDataInventoryRowV1(
                    row_kind="REMOTE_VERIFIED_OBJECT",
                    link_status="ATTESTED_PHASE1_SAMPLE_EVIDENCE",
                    object_sha256=object_sha,
                    filename=filename,
                    hour=phase0_hour,
                    size_bytes=size,
                    phase0_reported_size_bytes=phase0_size,
                    phase0_size_matches_verified_bytes=phase0_size == size,
                    remote_verification_status="verified",
                )
            )
            continue
        manifest_size = _manifest_size(manifest)
        if manifest_size != size:
            raise L2MethodError("remote verified byte size does not match manifest byte length")
        filename = _manifest_filename(manifest)
        phase0 = phase0_by_filename.get(filename)
        if phase0 is None:
            raise L2MethodError("authoritative manifest filename is absent from Phase-0 inventory")
        phase0_size = _inventory_size(phase0, field="phase0_inventory.objects")
        rows.append(
            TickDataInventoryRowV1(
                row_kind="REMOTE_VERIFIED_OBJECT",
                link_status="ATTESTED_MANIFEST_CROSSWALK",
                object_sha256=object_sha,
                filename=filename,
                hour=_inventory_text(phase0.get("hour"), field="phase0 hour"),
                size_bytes=size,
                phase0_reported_size_bytes=phase0_size,
                phase0_size_matches_verified_bytes=phase0_size == manifest_size,
                remote_verification_status="verified",
            )
        )

    for filename, phase0 in phase0_by_filename.items():
        manifest = next(
            (
                candidate
                for candidate in manifests_by_sha.values()
                if _manifest_filename(candidate) == filename
            ),
            None,
        )
        phase1_record = next(
            (
                candidate
                for candidate in phase1_by_sha.values()
                if _inventory_text(
                    candidate["filename"], field="Phase-1 filename"
                )
                == filename
            ),
            None,
        )
        attested_sha = (
            _inventory_text(
                manifest.get("object_sha256"), field="manifest object_sha256"
            )
            if manifest is not None
            else (
                _inventory_text(
                    phase1_record["object_sha256"], field="Phase-1 object_sha256"
                )
                if phase1_record is not None
                else None
            )
        )
        if attested_sha is not None and attested_sha in verified_remote_shas:
            continue
        rows.append(
            TickDataInventoryRowV1(
                row_kind="PHASE0_DISCOVERY_ENTRY",
                link_status=(
                    "MANIFEST_CROSSWALK_REMOTE_UNVERIFIED"
                    if filename in manifest_filenames
                    else (
                        "PHASE1_CROSSWALK_REMOTE_UNVERIFIED"
                        if filename in phase1_filenames
                        else "UNATTESTED_NO_CROSSWALK"
                    )
                ),
                object_sha256=None,
                filename=filename,
                hour=_inventory_text(phase0.get("hour"), field="phase0 hour"),
                size_bytes=_inventory_size(phase0, field="phase0_inventory.objects"),
                phase0_reported_size_bytes=_inventory_size(
                    phase0, field="phase0_inventory.objects"
                ),
                phase0_size_matches_verified_bytes=None,
                remote_verification_status=None,
            )
        )
    if not rows:
        raise L2MethodError("remote inventory contains no verified objects")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                0 if row.row_kind == "REMOTE_VERIFIED_OBJECT" else 1,
                row.object_sha256 or "",
                row.hour or "",
                row.filename or "",
            ),
        )
    )


def _inventory_objects(inventory: Mapping[str, object], *, field: str) -> list[Mapping[str, object]]:
    objects = inventory.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes, bytearray)):
        raise L2MethodError(f"{field}.objects must be a sequence")
    normalized: list[Mapping[str, object]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise L2MethodError(f"{field}.objects[{index}] must be an object")
        normalized.append(item)
    return normalized


def _inventory_size(item: Mapping[str, object], *, field: str) -> int:
    value = item.get("byte_size", item.get("size_bytes"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise L2MethodError(f"{field} needs a positive integer byte size")
    return value


def _inventory_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise L2MethodError(f"{field} must be a non-empty string")
    return value


def _manifest_index(
    authoritative_manifests: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    filename_to_sha: dict[str, str] = {}
    for index, manifest in enumerate(authoritative_manifests):
        if not isinstance(manifest, Mapping):
            raise L2MethodError(f"authoritative_manifests[{index}] must be an object")
        object_sha = _inventory_text(
            manifest.get("object_sha256"), field="manifest object_sha256"
        )
        filename = _manifest_filename(manifest)
        _manifest_size(manifest)
        filename_sha = filename_to_sha.get(filename)
        if filename_sha is not None and filename_sha != object_sha:
            raise L2MethodError(
                "duplicate authoritative manifest filename has different SHA-256"
            )
        filename_to_sha[filename] = object_sha
        prior = indexed.get(object_sha)
        if prior is not None and (
            _manifest_filename(prior) != _manifest_filename(manifest)
            or _manifest_size(prior) != _manifest_size(manifest)
        ):
            raise L2MethodError("duplicate manifest SHA-256 has conflicting metadata")
        indexed[object_sha] = manifest
    return indexed


def _phase1_evidence_index(
    measurement: Mapping[str, object] | None,
    *,
    measurement_sha256: str | None,
) -> dict[str, Mapping[str, object]]:
    if measurement is None:
        if measurement_sha256 is not None:
            raise L2MethodError("Phase-1 hash supplied without measurement")
        return {}
    if measurement_sha256 != PHASE1_SAMPLE_MEASUREMENT_SHA256:
        raise L2MethodError("Phase-1 measurement file SHA-256 is not the frozen hash")
    if measurement.get("audit_kind") != "pmxt_phase1_public_sample":
        raise L2MethodError("Phase-1 measurement has the wrong audit_kind")
    if measurement.get("queue_fill_reconstructed") is not False:
        raise L2MethodError("Phase-1 measurement makes a positive fill claim")
    files = measurement.get("files")
    if not isinstance(files, Sequence) or isinstance(
        files, (str, bytes, bytearray)
    ):
        raise L2MethodError("Phase-1 measurement files must be a sequence")

    indexed: dict[str, Mapping[str, object]] = {}
    filename_to_sha: dict[str, str] = {}
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise L2MethodError(f"Phase-1 files[{index}] must be an object")
        archive = _required_mapping(item.get("archive"), field="Phase-1 archive")
        inventory = _required_mapping(
            item.get("inventory_object"), field="Phase-1 inventory_object"
        )
        parquet = _required_mapping(
            item.get("parquet_audit"), field="Phase-1 parquet_audit"
        )
        object_sha = _inventory_text(
            archive.get("sha256"), field="Phase-1 archive SHA-256"
        )
        if parquet.get("original_sha256") != object_sha:
            raise L2MethodError("Phase-1 archive and parquet SHA-256 differ")
        byte_size = _inventory_size(archive, field="Phase-1 archive")
        if _inventory_size(parquet, field="Phase-1 parquet_audit") != byte_size:
            raise L2MethodError("Phase-1 archive and parquet byte sizes differ")
        filename = _inventory_text(
            archive.get("source_filename"), field="Phase-1 source filename"
        )
        if inventory.get("filename") != filename:
            raise L2MethodError("Phase-1 archive and inventory filenames differ")
        source_url = _inventory_text(
            archive.get("source_url"), field="Phase-1 source URL"
        )
        if inventory.get("url") != source_url:
            raise L2MethodError("Phase-1 archive and inventory URLs differ")
        if object_sha in indexed:
            raise L2MethodError(f"duplicate Phase-1 object SHA-256: {object_sha}")
        filename_sha = filename_to_sha.get(filename)
        if filename_sha is not None and filename_sha != object_sha:
            raise L2MethodError(
                "duplicate Phase-1 filename has different SHA-256"
            )
        filename_to_sha[filename] = object_sha
        indexed[object_sha] = {
            "object_sha256": object_sha,
            "byte_size": byte_size,
            "filename": filename,
            "hour": _inventory_text(
                inventory.get("hour"), field="Phase-1 inventory hour"
            ),
            "phase0_size_bytes": _inventory_size(
                inventory, field="Phase-1 inventory_object"
            ),
            "source_url": source_url,
            "measurement_sha256": measurement_sha256,
        }
    return indexed


def _required_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise L2MethodError(f"{field} must be an object")
    return value


def _manifest_filename(manifest: Mapping[str, object]) -> str:
    cursor = _inventory_text(manifest.get("source_cursor"), field="manifest source_cursor")
    prefix = "filename:"
    if not cursor.startswith(prefix) or not cursor[len(prefix) :]:
        raise L2MethodError("manifest source_cursor must contain a filename")
    return cursor[len(prefix) :]


def _manifest_size(manifest: Mapping[str, object]) -> int:
    value = manifest.get("byte_length")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise L2MethodError("manifest byte_length must be a positive integer")
    return value


def build_method_audit_payload(
    *,
    fixture_events: Iterable[Mapping[str, object]],
    remote_inventory: Mapping[str, object],
    phase0_inventory: Mapping[str, object],
    authoritative_manifests: Iterable[Mapping[str, object]] = (),
    phase1_sample_measurement: Mapping[str, object] | None = None,
    phase1_measurement_sha256: str | None = None,
) -> dict[str, object]:
    """Build a deterministic audit payload from local fixture and metadata only."""

    events = tuple(fixture_events)
    reconstruction = _build_l2_metric_result(events)
    ticks = reconstruction.ticks
    rows = build_tick_data_inventory_v1_rows(
        remote_inventory,
        phase0_inventory,
        authoritative_manifests=authoritative_manifests,
        phase1_sample_measurement=phase1_sample_measurement,
        phase1_measurement_sha256=phase1_measurement_sha256,
    )
    remote_rows = [row for row in rows if row.row_kind == "REMOTE_VERIFIED_OBJECT"]
    phase0_rows = [row for row in rows if row.row_kind == "PHASE0_DISCOVERY_ENTRY"]
    attested_count = sum(
        row.link_status.startswith("ATTESTED_") for row in remote_rows
    )
    phase0_size_mismatch_count = sum(
        row.phase0_size_matches_verified_bytes is False for row in remote_rows
    )
    phase0_summary = phase0_inventory.get("summary")
    coverage = _phase0_coverage(
        phase0_summary,
        entry_count=len(
            _inventory_objects(phase0_inventory, field="phase0_inventory")
        ),
    )
    unattested_count = len(remote_rows) - attested_count
    return {
        "audit_kind": "pmxt_l2_method_audit_v1",
        "data_access": "local_fixture_and_metadata_only",
        "raw_parquet_scanned": False,
        "upstream_accessed": False,
        "fixture_event_count": len(events),
        "tick_count": len(ticks),
        "epoch_count": len({(tick.market, tick.asset_id, tick.epoch) for tick in ticks}),
        "method_gates": _gate_report(reconstruction.gates, ticks),
        "fixture_tick_metrics": [tick.to_dict() for tick in ticks],
        "method_verification": _method_verification(ticks),
        "formal_aggressor_metrics": {
            "emitted": False,
            "reason": "no_authoritative_fill_in_l2_input",
        },
        "schema_boundary": {
            "nfl_alpha_accepted": False,
            "level3_accepted": False,
            "positive_fill_claim_accepted": False,
        },
        "remote_verification": {
            "verified_object_count": len(remote_rows),
            "verified_total_bytes": sum(row.size_bytes for row in remote_rows),
        },
        "phase1_sample_measurement_sha256": phase1_measurement_sha256,
        "phase0_coverage": coverage,
        "crosswalk": {
            "attested_remote_object_count": attested_count,
            "unattested_remote_object_count": unattested_count,
            "phase0_size_mismatch_count": phase0_size_mismatch_count,
            "phase0_discovery_entry_count": len(phase0_rows),
            "hourly_l2_scan_status": (
                "READY_ALL_REMOTE_ATTESTED"
                if remote_rows and unattested_count == 0
                else "BLOCKED_UNATTESTED_REMOTE_OBJECT"
            ),
        },
        "tick_inventory_rows": [row.to_dict() for row in rows],
    }


def _gate_report(
    gates: _GateEvidence, ticks: Sequence[TickMetricV1]
) -> dict[str, object]:
    formal_aggressor_fields = sum(
        "aggressor_side" in tick.to_dict() for tick in ticks
    )
    return {
        "snapshot_before_delta": {
            "status": (
                "PASS"
                if gates.snapshot_count > 0 and gates.accepted_delta_count > 0
                else "INSUFFICIENT_FIXTURE_EVIDENCE"
            ),
            "snapshot_count": gates.snapshot_count,
            "accepted_delta_count": gates.accepted_delta_count,
        },
        "gap_requires_new_snapshot_epoch": _invalidation_gate(
            gates.gap_invalidation_count,
            gates.gap_resnapshot_count,
            gates.gap_epoch_increment_count,
        ),
        "reconnect_requires_new_snapshot_epoch": _invalidation_gate(
            gates.reconnect_invalidation_count,
            gates.reconnect_resnapshot_count,
            gates.reconnect_epoch_increment_count,
        ),
        "book_update_side_is_not_trade_aggressor": {
            "status": "PASS" if formal_aggressor_fields == 0 else "FAIL",
            "formal_aggressor_fields_emitted": formal_aggressor_fields,
        },
    }


def _invalidation_gate(
    invalidation_count: int,
    resnapshot_count: int,
    epoch_increment_count: int,
) -> dict[str, object]:
    return {
        "status": (
            "PASS"
            if invalidation_count > 0
            and resnapshot_count == invalidation_count
            and epoch_increment_count == invalidation_count
            else "FAIL"
        ),
        "invalidation_count": invalidation_count,
        "resnapshot_count": resnapshot_count,
        "epoch_increment_count": epoch_increment_count,
    }


def _method_verification(
    ticks: Sequence[TickMetricV1],
) -> dict[str, object]:
    tick_rows = [tick.to_dict() for tick in ticks]
    by_tick = lambda fields: [  # noqa: E731
        {"tick_index": index, **{field: row[field] for field in fields}}
        for index, row in enumerate(tick_rows)
    ]
    cancel_results = [
        cancel_intensity(
            basis=MetricBasis.BID, cancelled_size="4", observed_depth="20"
        ).to_dict(),
        cancel_intensity(
            basis=MetricBasis.ASK, cancelled_size="3", observed_depth="10"
        ).to_dict(),
    ]
    resilience_results = [
        resilience_fraction(
            basis=MetricBasis.BID,
            baseline_reference="0.48",
            shocked_reference="0.46",
            recovered_reference="0.475",
        ).to_dict(),
        resilience_fraction(
            basis=MetricBasis.ASK,
            baseline_reference="0.52",
            shocked_reference="0.54",
            recovered_reference="0.525",
        ).to_dict(),
        resilience_fraction(
            basis=MetricBasis.MIDPOINT,
            baseline_reference="0.50",
            shocked_reference="0.49",
            recovered_reference="0.498",
        ).to_dict(),
    ]
    markout_results = [
        signed_markout(
            side="BUY",
            basis=basis,
            execution_price="0.48",
            future_reference=future,
        ).to_dict()
        for basis, future in (
            (MetricBasis.BID, "0.49"),
            (MetricBasis.ASK, "0.51"),
            (MetricBasis.MIDPOINT, "0.50"),
        )
    ]
    walked = walk_book(
        side="BUY",
        levels=[("0.52", "2"), ("0.53", "3")],
        requested_size="4",
    )
    return {
        "calculation_fixture_scope": "method_only_not_empirical_fill_claim",
        "bbo": by_tick(("best_bid", "best_bid_size", "best_ask", "best_ask_size")),
        "spread": by_tick(("spread",)),
        "top_n_depth": by_tick(("bid_depth_top_n", "ask_depth_top_n")),
        "imbalance": by_tick(("imbalance",)),
        "microprice": by_tick(("microprice",)),
        "ofi": by_tick(("ofi",)),
        "depth_withdrawal": by_tick(
            ("bid_depth_withdrawal", "ask_depth_withdrawal")
        ),
        "cancel_intensity": cancel_results,
        "resilience": resilience_results,
        "markout": markout_results,
        "book_walk": {
            "inputs": {
                "side": "BUY",
                "levels": [["0.52", "2"], ["0.53", "3"]],
                "requested_size": "4",
            },
            "result": walked.to_dict(),
        },
        "parity_deviation": {
            "inputs": {"yes_midpoint": "0.51", "no_midpoint": "0.47"},
            "value": _decimal_text(parity_deviation("0.51", "0.47")),
        },
    }


def _phase0_coverage(summary: object, *, entry_count: int) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        return {"phase0_entry_count": entry_count}
    keys = ("first_hour", "last_hour", "covered_hours", "expected_hours_inclusive", "missing_hours")
    coverage = {key: summary[key] for key in keys if key in summary}
    coverage["phase0_entry_count"] = entry_count
    return coverage
