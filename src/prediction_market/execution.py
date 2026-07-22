"""Taker-only PRELIMINARY execution simulation and TCA records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, Sequence

from prediction_market.contracts import (
    FixedPointV0,
    VenueRuleSnapshotV0,
    canonical_sha256,
    validate_contract_v0,
)


_ORDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
FeeFormulaV0 = Literal["C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT"]


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
    observed_at: datetime
    bids: tuple[DepthLevelV0, ...]
    asks: tuple[DepthLevelV0, ...]
    suspended: bool

    def __post_init__(self) -> None:
        _require_utc(self.observed_at, "book observed_at")
        if type(self.bids) is not tuple or type(self.asks) is not tuple:
            raise SimulationInputError("book bids and asks must be immutable tuples")
        if any(not isinstance(level, DepthLevelV0) for level in (*self.bids, *self.asks)):
            raise SimulationInputError("book levels must be DepthLevelV0")
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


@dataclass(frozen=True, slots=True)
class TakerBuyOrderV0:
    order_id: str
    experiment_id: Literal["X-07"]
    condition_id: str
    created_at: datetime
    quantity: FixedPointV0
    fee_formula: FeeFormulaV0

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not _ORDER_ID.fullmatch(self.order_id):
            raise SimulationInputError("order_id must be canonical and nonempty")
        if self.experiment_id != "X-07":
            raise SimulationInputError("taker simulator is scoped to registered X-07")
        if not isinstance(self.condition_id, str) or not self.condition_id.startswith(
            "condition_"
        ):
            raise SimulationInputError("condition_id must be canonical")
        _require_utc(self.created_at, "order created_at")
        if not isinstance(self.quantity, FixedPointV0) or self.quantity.to_decimal() <= 0:
            raise SimulationInputError("order quantity must be positive fixed-point")
        if self.fee_formula != "C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT":
            raise SimulationInputError("unsupported fee formula")


@dataclass(frozen=True, slots=True)
class MarkoutV0:
    horizon_microseconds: int
    observed_at: datetime
    executable_bid_vwap: FixedPointV0
    gross_markout_per_unit: FixedPointV0


@dataclass(frozen=True, slots=True)
class TakerFillV0:
    tca_version: Literal["v0"]
    order_id: str
    experiment_id: Literal["X-07"]
    result_label: Literal["PRELIMINARY"]
    fee_formula: FeeFormulaV0
    condition_id: str
    created_at: datetime
    executed_at: datetime
    filled_quantity: FixedPointV0
    vwap: FixedPointV0
    gross_cost: FixedPointV0
    fee: FixedPointV0
    total_cost: FixedPointV0
    levels_consumed: int
    own_delay_microseconds: int
    venue_delay_microseconds: int
    rule_snapshot_ref: str
    markouts: tuple[MarkoutV0, ...]


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00").astimezone(UTC)


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
) -> tuple[Decimal, int]:
    remaining = quantity
    notional = Decimal(0)
    consumed = 0
    for level in levels:
        take = min(remaining, level.quantity.to_decimal())
        if take <= 0:
            continue
        notional += take * level.price.to_decimal()
        remaining -= take
        consumed += 1
        if remaining == 0:
            break
    if remaining != 0:
        raise InsufficientDepthError(
            f"insufficient executable {side} depth for requested quantity"
        )
    return notional / quantity, consumed


def _validate_stream(book_stream: Sequence[BookSnapshotV0]) -> tuple[BookSnapshotV0, ...]:
    if not book_stream:
        raise SimulationInputError("book_stream must not be empty")
    books = tuple(book_stream)
    if any(not isinstance(book, BookSnapshotV0) for book in books):
        raise SimulationInputError("book_stream contains an invalid snapshot")
    times = [book.observed_at for book in books]
    if any(left >= right for left, right in zip(times, times[1:], strict=False)):
        raise SimulationInputError("book_stream must be strictly chronological")
    return books


def _first_executable(
    books: tuple[BookSnapshotV0, ...],
    not_before: datetime,
) -> BookSnapshotV0:
    for book in books:
        if book.observed_at >= not_before and not book.suspended:
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
    rule_snapshot: VenueRuleSnapshotV0 | None,
    *,
    own_delay: timedelta,
    markout_horizons: tuple[timedelta, ...],
    formal: bool,
) -> TakerFillV0:
    """Consume ask depth and emit PRELIMINARY TCA with executable-bid markouts."""

    if formal and rule_snapshot is None:
        raise PreliminaryOnlyError(
            "formal result requires an exact venue_rule_snapshot"
        )
    if formal:
        raise PreliminaryOnlyError(
            "X-07 is PRELIMINARY until snapshot replay and formal scope are authorized"
        )
    if not isinstance(order, TakerBuyOrderV0):
        raise SimulationInputError("order must be TakerBuyOrderV0")
    if not isinstance(rule_snapshot, VenueRuleSnapshotV0):
        raise SimulationInputError("PRELIMINARY simulation still requires a rule snapshot")
    # Revalidate a detached representation so mutated/constructed model state is
    # not trusted merely because it carries the expected Python type.
    rule = validate_contract_v0(
        "venue-rule-snapshot/v0.schema.yaml", rule_snapshot
    )
    if rule.condition_id != order.condition_id:
        raise SimulationInputError("rule snapshot condition does not match order")
    if _parse_utc(rule.fetched_at) > order.created_at or _parse_utc(
        rule.effective_from
    ) > order.created_at:
        raise SimulationInputError("venue rule snapshot is not point-in-time available")
    if order.quantity.to_decimal() < rule.minimum_order_size.to_decimal():
        raise SimulationInputError("order is below snapshot minimum_order_size")
    if not isinstance(own_delay, timedelta) or own_delay < timedelta(0):
        raise SimulationInputError("own_delay must be a nonnegative timedelta")
    _validate_horizons(markout_horizons)
    books = _validate_stream(book_stream)

    venue_delay = _decimal_seconds(rule.seconds_delay, "seconds_delay")
    execute_not_before = order.created_at + venue_delay + own_delay
    execution_book = _first_executable(books, execute_not_before)
    quantity = order.quantity.to_decimal()
    vwap, levels_consumed = _consume(execution_book.asks, quantity, side="ask")
    gross_cost = quantity * vwap

    if rule.fees_enabled:
        exponent = rule.fee_exponent.to_decimal()
        if exponent != exponent.to_integral_value():
            raise SimulationInputError("fee_exponent must be an integer in TCA v0")
        fee = (
            quantity
            * rule.fee_rate.to_decimal()
            * (vwap * (Decimal(1) - vwap)) ** int(exponent)
        )
    else:
        fee = Decimal(0)

    markouts: list[MarkoutV0] = []
    for horizon in markout_horizons:
        markout_book = _first_executable(books, execution_book.observed_at + horizon)
        bid_vwap, _ = _consume(markout_book.bids, quantity, side="bid")
        markouts.append(
            MarkoutV0(
                horizon_microseconds=horizon // timedelta(microseconds=1),
                observed_at=markout_book.observed_at,
                executable_bid_vwap=_fixed(bid_vwap),
                gross_markout_per_unit=_fixed(bid_vwap - vwap),
            )
        )

    rule_ref = canonical_sha256(rule.model_dump(mode="json", round_trip=True))
    return TakerFillV0(
        tca_version="v0",
        order_id=order.order_id,
        experiment_id="X-07",
        result_label="PRELIMINARY",
        fee_formula=order.fee_formula,
        condition_id=order.condition_id,
        created_at=order.created_at,
        executed_at=execution_book.observed_at,
        filled_quantity=_fixed(quantity),
        vwap=_fixed(vwap),
        gross_cost=_fixed(gross_cost),
        fee=_fixed(fee),
        total_cost=_fixed(gross_cost + fee),
        levels_consumed=levels_consumed,
        own_delay_microseconds=own_delay // timedelta(microseconds=1),
        venue_delay_microseconds=venue_delay // timedelta(microseconds=1),
        rule_snapshot_ref=rule_ref,
        markouts=tuple(markouts),
    )


__all__ = [
    "BookSnapshotV0",
    "DepthLevelV0",
    "InsufficientDepthError",
    "MarkoutV0",
    "PreliminaryOnlyError",
    "SimulationInputError",
    "TakerBuyOrderV0",
    "TakerFillV0",
    "simulate_taker_buy",
]
