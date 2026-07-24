"""Fail-closed interval arithmetic for game and market timing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class TemporalEligibilityError(ValueError):
    """The supplied timing evidence cannot support the requested comparison."""


@dataclass(frozen=True, slots=True)
class UtcInterval:
    """A nonempty half-open interval measured in UTC nanoseconds."""

    lower_ns: int
    upper_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.lower_ns, int) or not isinstance(self.upper_ns, int):
            raise TypeError("UTC interval bounds must be integers")
        if self.upper_ns <= self.lower_ns:
            raise TemporalEligibilityError(
                "UTC interval must be nonempty and half-open"
            )


@dataclass(frozen=True, slots=True)
class LocalInstant:
    """One monotonic instant with the full identity needed for subtraction."""

    host_id: str
    boot_id: str
    monotonic_clock_id: str
    clock_epoch_id: str
    monotonic_ns: int

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.host_id,
                self.boot_id,
                self.monotonic_clock_id,
                self.clock_epoch_id,
            )
        ):
            raise ValueError("local instant clock identity fields must be nonempty")
        if not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be a non-negative integer")

    @property
    def clock_key(self) -> tuple[str, str, str, str]:
        return (
            self.host_id,
            self.boot_id,
            self.monotonic_clock_id,
            self.clock_epoch_id,
        )


@dataclass(frozen=True, slots=True)
class CrossClockDuration:
    """A cross-clock difference whose sign may be unresolved by uncertainty."""

    status: Literal["RESOLVED", "UNRESOLVED"]
    interval: UtcInterval
    duration: int | None


@dataclass(frozen=True, slots=True)
class EventMarketOrder:
    """Economic and local-availability ordering, kept deliberately separate."""

    economic_order: Literal["PRE", "POST", "BEFORE_FINALIZATION", "AMBIGUOUS"]
    availability_order: Literal["PRE", "POST", "AMBIGUOUS", "BLOCKED", "UNAVAILABLE"]
    clean_pre_baseline: bool
    clean_event_after: bool


def point_interval(
    timestamp_ns: int, *, precision_ns: int, clock_error_ns: int
) -> UtcInterval:
    """Expand one source timestamp by native precision and clock error once."""

    if not all(
        isinstance(value, int) and value >= 0
        for value in (precision_ns, clock_error_ns)
    ):
        raise ValueError("precision_ns and clock_error_ns must be non-negative integers")
    if not isinstance(timestamp_ns, int):
        raise TypeError("timestamp_ns must be an integer")
    return UtcInterval(
        lower_ns=timestamp_ns - clock_error_ns,
        upper_ns=timestamp_ns + precision_ns + clock_error_ns,
    )


def subtract_intervals(later: UtcInterval, earlier: UtcInterval) -> UtcInterval:
    """Return B-A without double-counting uncertainty already in B and A."""

    return UtcInterval(
        lower_ns=later.lower_ns - earlier.upper_ns,
        upper_ns=later.upper_ns - earlier.lower_ns,
    )


def same_host_elapsed(later: LocalInstant, earlier: LocalInstant) -> int:
    """Subtract monotonic times only when their complete clock identity matches."""

    if later.clock_key != earlier.clock_key:
        raise TemporalEligibilityError("monotonic clocks are not comparable")
    if later.monotonic_ns < earlier.monotonic_ns:
        raise TemporalEligibilityError("negative same-host monotonic duration")
    return later.monotonic_ns - earlier.monotonic_ns


def classify_cross_clock_duration(interval: UtcInterval) -> CrossClockDuration:
    """Classify uncertainty crossing zero without silently clamping it."""

    if interval.upper_ns <= 0:
        raise TemporalEligibilityError(
            "negative cross-clock duration outside uncertainty"
        )
    if interval.lower_ns < 0:
        return CrossClockDuration(
            status="UNRESOLVED",
            interval=interval,
            duration=None,
        )
    return CrossClockDuration(
        status="RESOLVED",
        interval=interval,
        duration=interval.lower_ns,
    )


def _availability_order(
    game_ready: LocalInstant | None, market_ready: LocalInstant | None
) -> Literal["PRE", "POST", "AMBIGUOUS", "BLOCKED", "UNAVAILABLE"]:
    if game_ready is None or market_ready is None:
        return "UNAVAILABLE"
    if game_ready.clock_key != market_ready.clock_key:
        return "BLOCKED"
    if market_ready.monotonic_ns < game_ready.monotonic_ns:
        return "PRE"
    if market_ready.monotonic_ns > game_ready.monotonic_ns:
        return "POST"
    return "AMBIGUOUS"


def classify_event_market_order(
    *,
    play_start: UtcInterval,
    finalization: UtcInterval,
    game_ready: LocalInstant | None,
    market_event: UtcInterval,
    market_ready: LocalInstant | None,
) -> EventMarketOrder:
    """Apply the approved M0/G0 and M3/G3 predicates without substitution."""

    if market_event.upper_ns <= play_start.lower_ns:
        economic_order: Literal[
            "PRE", "POST", "BEFORE_FINALIZATION", "AMBIGUOUS"
        ] = "PRE"
    elif market_event.lower_ns >= finalization.upper_ns:
        economic_order = "POST"
    elif market_event.upper_ns <= finalization.lower_ns:
        economic_order = "BEFORE_FINALIZATION"
    else:
        economic_order = "AMBIGUOUS"
    availability_order = _availability_order(game_ready, market_ready)
    return EventMarketOrder(
        economic_order=economic_order,
        availability_order=availability_order,
        clean_pre_baseline=(
            economic_order == "PRE" and availability_order == "PRE"
        ),
        clean_event_after=(
            economic_order == "POST" and availability_order == "POST"
        ),
    )


__all__ = [
    "CrossClockDuration",
    "EventMarketOrder",
    "LocalInstant",
    "TemporalEligibilityError",
    "UtcInterval",
    "classify_cross_clock_duration",
    "classify_event_market_order",
    "point_interval",
    "same_host_elapsed",
    "subtract_intervals",
]
