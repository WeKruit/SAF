from __future__ import annotations

import pytest

from prediction_market.temporal import (
    LocalInstant,
    TemporalEligibilityError,
    UtcInterval,
    classify_cross_clock_duration,
    classify_event_market_order,
    point_interval,
    same_host_elapsed,
    subtract_intervals,
)


def interval(lower: int, upper: int) -> UtcInterval:
    return UtcInterval(lower_ns=lower, upper_ns=upper)


def instant(value: int, *, host: str = "host", boot: str = "boot") -> LocalInstant:
    return LocalInstant(
        host_id=host,
        boot_id=boot,
        monotonic_clock_id="CLOCK_MONOTONIC",
        clock_epoch_id="epoch-1",
        monotonic_ns=value,
    )


def test_point_interval_includes_precision_and_clock_error_once() -> None:
    assert point_interval(1_000, precision_ns=10, clock_error_ns=3) == interval(
        997, 1_013
    )
    assert subtract_intervals(interval(1_020, 1_030), interval(997, 1_013)) == interval(
        7, 33
    )


def test_same_host_elapsed_rejects_different_clock_identity() -> None:
    with pytest.raises(TemporalEligibilityError, match="not comparable"):
        same_host_elapsed(instant(20, host="host-b"), instant(10))
    assert same_host_elapsed(instant(20), instant(10)) == 10


def test_cross_clock_duration_is_unresolved_when_zero_is_in_interval() -> None:
    assessment = classify_cross_clock_duration(interval(-2, 3))
    assert assessment.status == "UNRESOLVED"
    assert assessment.duration is None


def test_cross_clock_duration_rejects_negative_interval() -> None:
    with pytest.raises(TemporalEligibilityError, match="negative"):
        classify_cross_clock_duration(interval(-5, -1))


def test_clean_post_requires_both_economic_and_availability_order() -> None:
    result = classify_event_market_order(
        play_start=interval(100, 101),
        finalization=interval(110, 111),
        game_ready=instant(200),
        market_event=interval(109, 110),
        market_ready=instant(220),
    )
    assert result.economic_order == "BEFORE_FINALIZATION"
    assert result.availability_order == "POST"
    assert result.clean_event_after is False


def test_delayed_pre_event_update_is_not_available_pre_baseline() -> None:
    result = classify_event_market_order(
        play_start=interval(100, 101),
        finalization=interval(110, 111),
        game_ready=instant(200),
        market_event=interval(90, 91),
        market_ready=instant(210),
    )
    assert result.economic_order == "PRE"
    assert result.availability_order == "POST"
    assert result.clean_pre_baseline is False


def test_clean_orders_require_same_host_monotonic_clock() -> None:
    result = classify_event_market_order(
        play_start=interval(100, 101),
        finalization=interval(110, 111),
        game_ready=instant(200),
        market_event=interval(120, 121),
        market_ready=instant(220, boot="other-boot"),
    )
    assert result.availability_order == "BLOCKED"
    assert result.clean_event_after is False
