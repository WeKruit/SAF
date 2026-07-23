"""Taker-only PRELIMINARY execution simulation and TCA records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, Sequence

from prediction_market.contracts import (
    FeeFormulaV0,
    FixedPointV0,
    TcaMarkoutV0,
    TcaRecordV0,
    VenueRuleSnapshotV0,
    canonical_sha256,
    validate_contract_v0,
)
from prediction_market.recording import VenueRuleStore


_ORDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_VENUE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_MARKET_ID = re.compile(r"market_[A-Za-z0-9][A-Za-z0-9._:-]*")
_CONDITION_ID = re.compile(r"condition_[A-Za-z0-9][A-Za-z0-9._:-]*")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class SimulationInputError(ValueError):
    """A simulation input cannot support a deterministic taker claim."""


class InsufficientDepthError(SimulationInputError):
    """Executable depth cannot fill the requested quantity."""


class PreliminaryOnlyError(PermissionError):
    """The caller attempted to promote the current X-07 pipeline."""


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise SimulationInputError(f"{field} must be timezone-aware UTC")


def _fixed(value: Decimal) -> FixedPointV0:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return FixedPointV0.from_value(text)


def _utc_text(value: datetime) -> str:
    _require_utc(value, "timestamp")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class DepthLevelV0:
    price: FixedPointV0
    quantity: FixedPointV0

    def __post_init__(self) -> None:
        if not isinstance(self.price, FixedPointV0) or not isinstance(
            self.quantity, FixedPointV0
        ):
            raise SimulationInputError("depth values must be fixed-point")
        if not Decimal(0) <= self.price.to_decimal() <= Decimal(1):
            raise SimulationInputError("depth price must be in [0, 1]")
        if self.quantity.to_decimal() <= 0:
            raise SimulationInputError("depth quantity must be positive")


@dataclass(frozen=True, slots=True)
class BookSnapshotV0:
    venue: str
    condition_id: str
    local_receive_at: datetime
    bids: tuple[DepthLevelV0, ...]
    asks: tuple[DepthLevelV0, ...]
    suspended: bool
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.venue, str) or _VENUE.fullmatch(self.venue) is None:
            raise SimulationInputError("book venue must be canonical")
        if (
            not isinstance(self.condition_id, str)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
        ):
            raise SimulationInputError("book condition_id must be canonical")
        _require_utc(self.local_receive_at, "book local_receive_at")
        if type(self.bids) is not tuple or type(self.asks) is not tuple:
            raise SimulationInputError("book bids and asks must be immutable tuples")
        if any(not isinstance(level, DepthLevelV0) for level in (*self.bids, *self.asks)):
            raise SimulationInputError("book levels must be DepthLevelV0")
        if type(self.suspended) is not bool:
            raise SimulationInputError("book suspended must be boolean")
        bid_prices = [level.price.to_decimal() for level in self.bids]
        ask_prices = [level.price.to_decimal() for level in self.asks]
        if bid_prices != sorted(bid_prices, reverse=True) or len(bid_prices) != len(
            set(bid_prices)
        ):
            raise SimulationInputError("bid levels must be unique and descending")
        if ask_prices != sorted(ask_prices) or len(ask_prices) != len(set(ask_prices)):
            raise SimulationInputError("ask levels must be unique and ascending")
        if bid_prices and ask_prices and bid_prices[0] > ask_prices[0]:
            raise SimulationInputError("book cannot be crossed")
        if not isinstance(self.content_sha256, str) or _SHA256.fullmatch(
            self.content_sha256
        ) is None:
            raise SimulationInputError("book content_sha256 must be canonical")
        material = {
            "book_version": "v0",
            "venue": self.venue,
            "condition_id": self.condition_id,
            "local_receive_at": _utc_text(self.local_receive_at),
            "bids": [
                {
                    "price": level.price.model_dump(mode="json"),
                    "quantity": level.quantity.model_dump(mode="json"),
                }
                for level in self.bids
            ],
            "asks": [
                {
                    "price": level.price.model_dump(mode="json"),
                    "quantity": level.quantity.model_dump(mode="json"),
                }
                for level in self.asks
            ],
            "suspended": self.suspended,
        }
        if self.content_sha256 != canonical_sha256(material):
            raise SimulationInputError("book content_sha256 does not match content")


@dataclass(frozen=True, slots=True)
class TakerBuyOrderV0:
    order_id: str
    experiment_id: Literal["X-07"]
    venue: str
    market_id: str
    condition_id: str
    created_at: datetime
    quantity: FixedPointV0
    fee_formula: FeeFormulaV0

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not _ORDER_ID.fullmatch(self.order_id):
            raise SimulationInputError("order_id must be canonical and nonempty")
        if self.experiment_id != "X-07":
            raise SimulationInputError("taker simulator is scoped to registered X-07")
        if not isinstance(self.venue, str) or _VENUE.fullmatch(self.venue) is None:
            raise SimulationInputError("venue must be canonical")
        if (
            not isinstance(self.market_id, str)
            or _MARKET_ID.fullmatch(self.market_id) is None
        ):
            raise SimulationInputError("market_id must be canonical")
        if (
            not isinstance(self.condition_id, str)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
        ):
            raise SimulationInputError("condition_id must be canonical")
        _require_utc(self.created_at, "order created_at")
        if not isinstance(self.quantity, FixedPointV0) or self.quantity.to_decimal() <= 0:
            raise SimulationInputError("order quantity must be positive fixed-point")
        if self.fee_formula != "C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT":
            raise SimulationInputError("unsupported fee formula")


def _decimal_seconds(value: FixedPointV0, field: str) -> timedelta:
    microseconds = value.to_decimal() * Decimal(1_000_000)
    if microseconds != microseconds.to_integral_value():
        raise SimulationInputError(f"{field} is not exactly representable in microseconds")
    return timedelta(microseconds=int(microseconds))


def _consume(
    levels: tuple[DepthLevelV0, ...],
    quantity: Decimal,
    *,
    side: Literal["ask", "bid"],
) -> tuple[Decimal, tuple[tuple[Decimal, Decimal], ...]]:
    remaining = quantity
    notional = Decimal(0)
    fills: list[tuple[Decimal, Decimal]] = []
    for level in levels:
        take = min(remaining, level.quantity.to_decimal())
        if take <= 0:
            continue
        notional += take * level.price.to_decimal()
        remaining -= take
        fills.append((take, level.price.to_decimal()))
        if remaining == 0:
            break
    if remaining != 0:
        raise InsufficientDepthError(
            f"insufficient executable {side} depth for requested quantity"
        )
    return notional / quantity, tuple(fills)


def _fee_for_fills(
    fills: tuple[tuple[Decimal, Decimal], ...],
    rule: VenueRuleSnapshotV0,
) -> Decimal:
    if not rule.fees_enabled:
        return Decimal(0)
    exponent = rule.fee_exponent.to_decimal()
    if exponent != exponent.to_integral_value():
        raise SimulationInputError("fee_exponent must be an integer in TCA v0")
    rate = rule.fee_rate.to_decimal()
    return sum(
        (
            quantity
            * rate
            * (price * (Decimal(1) - price)) ** int(exponent)
        )
        for quantity, price in fills
    )


def _validate_stream(book_stream: Sequence[BookSnapshotV0]) -> tuple[BookSnapshotV0, ...]:
    if not book_stream:
        raise SimulationInputError("book_stream must not be empty")
    books = tuple(book_stream)
    if any(not isinstance(book, BookSnapshotV0) for book in books):
        raise SimulationInputError("book_stream contains an invalid snapshot")
    times = [book.local_receive_at for book in books]
    if any(left >= right for left, right in zip(times, times[1:], strict=False)):
        raise SimulationInputError("book_stream must be strictly chronological")
    return books


def _first_executable(
    books: tuple[BookSnapshotV0, ...],
    not_before: datetime,
) -> BookSnapshotV0:
    for book in books:
        if book.local_receive_at >= not_before and not book.suspended:
            return book
    raise SimulationInputError("no executable book at or after required time")


def _validate_horizons(values: tuple[timedelta, ...]) -> None:
    if type(values) is not tuple or not values:
        raise SimulationInputError("markout_horizons must be a nonempty tuple")
    if any(not isinstance(value, timedelta) or value <= timedelta(0) for value in values):
        raise SimulationInputError("markout_horizons must be positive timedeltas")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise SimulationInputError("markout_horizons must be unique and ascending")


def simulate_taker_buy(
    book_stream: Sequence[BookSnapshotV0],
    order: TakerBuyOrderV0,
    rule_store: VenueRuleStore,
    *,
    rule_max_age_seconds: int,
    own_delay: timedelta,
    markout_horizons: tuple[timedelta, ...],
    formal: bool,
) -> TcaRecordV0:
    """Consume ask depth and emit PRELIMINARY TCA with executable-bid markouts."""

    if formal:
        raise PreliminaryOnlyError(
            "X-07 is PRELIMINARY until snapshot replay and formal scope are authorized"
        )
    if not isinstance(order, TakerBuyOrderV0):
        raise SimulationInputError("order must be TakerBuyOrderV0")
    if not isinstance(rule_store, VenueRuleStore):
        raise SimulationInputError("rule_store must be VenueRuleStore")
    rule = rule_store.require_as_of(
        venue=order.venue,
        market_id=order.market_id,
        condition_id=order.condition_id,
        at=_utc_text(order.created_at),
        max_age_seconds=rule_max_age_seconds,
    )
    rule = validate_contract_v0("venue-rule-snapshot/v0.schema.yaml", rule)
    if rule.venue != order.venue or rule.condition_id != order.condition_id:
        raise SimulationInputError("rule snapshot lineage does not match order")
    if order.quantity.to_decimal() < rule.minimum_order_size.to_decimal():
        raise SimulationInputError("order is below snapshot minimum_order_size")
    if not isinstance(own_delay, timedelta) or own_delay < timedelta(0):
        raise SimulationInputError("own_delay must be a nonnegative timedelta")
    _validate_horizons(markout_horizons)
    books = _validate_stream(book_stream)
    if any(
        book.venue != order.venue or book.condition_id != order.condition_id
        for book in books
    ):
        raise SimulationInputError("book lineage does not match order")

    venue_delay = _decimal_seconds(rule.seconds_delay, "seconds_delay")
    execute_not_before = order.created_at + venue_delay + own_delay
    execution_book = _first_executable(books, execute_not_before)
    quantity = order.quantity.to_decimal()
    vwap, entry_fills = _consume(execution_book.asks, quantity, side="ask")
    gross_cost = quantity * vwap
    entry_fee = _fee_for_fills(entry_fills, rule)

    markouts: list[TcaMarkoutV0] = []
    for horizon in markout_horizons:
        markout_book = _first_executable(
            books, execution_book.local_receive_at + horizon
        )
        bid_vwap, exit_fills = _consume(markout_book.bids, quantity, side="bid")
        exit_rule = rule_store.require_as_of(
            venue=order.venue,
            market_id=order.market_id,
            condition_id=order.condition_id,
            at=_utc_text(markout_book.local_receive_at),
            max_age_seconds=rule_max_age_seconds,
        )
        exit_rule = validate_contract_v0(
            "venue-rule-snapshot/v0.schema.yaml", exit_rule
        )
        if quantity < exit_rule.minimum_order_size.to_decimal():
            raise SimulationInputError(
                "markout exit is below snapshot minimum_order_size"
            )
        exit_fee = _fee_for_fills(exit_fills, exit_rule)
        gross_markout = bid_vwap - vwap
        net_markout = (
            quantity * bid_vwap - exit_fee - gross_cost - entry_fee
        ) / quantity
        markouts.append(
            TcaMarkoutV0(
                horizon_microseconds=horizon // timedelta(microseconds=1),
                local_receive_at=_utc_text(markout_book.local_receive_at),
                executable_bid_vwap=_fixed(bid_vwap),
                gross_markout_per_unit=_fixed(gross_markout),
                exit_fee=_fixed(exit_fee),
                net_markout_per_unit=_fixed(net_markout),
                exit_levels_consumed=len(exit_fills),
                rule_snapshot_ref=canonical_sha256(
                    exit_rule.model_dump(mode="json", round_trip=True)
                ),
                book_snapshot_ref=markout_book.content_sha256,
            )
        )

    rule_ref = canonical_sha256(rule.model_dump(mode="json", round_trip=True))
    return TcaRecordV0(
        tca_version="v0",
        order_id=order.order_id,
        experiment_id="X-07",
        result_label="PRELIMINARY",
        fee_formula=order.fee_formula,
        venue=order.venue,
        market_id=order.market_id,
        condition_id=order.condition_id,
        created_at=_utc_text(order.created_at),
        executed_at=_utc_text(execution_book.local_receive_at),
        filled_quantity=_fixed(quantity),
        entry_vwap=_fixed(vwap),
        gross_entry_cost=_fixed(gross_cost),
        entry_fee=_fixed(entry_fee),
        total_entry_cost=_fixed(gross_cost + entry_fee),
        entry_levels_consumed=len(entry_fills),
        own_delay_microseconds=own_delay // timedelta(microseconds=1),
        venue_delay_microseconds=venue_delay // timedelta(microseconds=1),
        entry_rule_snapshot_ref=rule_ref,
        entry_book_snapshot_ref=execution_book.content_sha256,
        markouts=tuple(markouts),
    )


__all__ = [
    "BookSnapshotV0",
    "DepthLevelV0",
    "InsufficientDepthError",
    "PreliminaryOnlyError",
    "SimulationInputError",
    "TakerBuyOrderV0",
    "simulate_taker_buy",
]
