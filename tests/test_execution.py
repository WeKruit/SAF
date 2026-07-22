from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from prediction_market.contracts import FixedPointV0, VenueRuleSnapshotV0
from prediction_market.execution import (
    BookSnapshotV0,
    DepthLevelV0,
    InsufficientDepthError,
    PreliminaryOnlyError,
    SimulationInputError,
    TakerBuyOrderV0,
    simulate_taker_buy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 7, 22, 18, 0, tzinfo=UTC)


def _fp(value: str) -> FixedPointV0:
    return FixedPointV0.from_value(value)


def _rule(
    *,
    fetched_at: str = "2026-07-22T17:59:00Z",
    delay: str = "1.0",
    fee_rate: str = "0.05",
) -> VenueRuleSnapshotV0:
    return VenueRuleSnapshotV0.model_validate(
        {
            "venue": "polymarket",
            "condition_id": "condition_nba-game-1",
            "fetched_at": fetched_at,
            "effective_from": "2026-07-22T17:00:00Z",
            "game_start_time": "2026-07-22T18:00:00Z",
            "seconds_delay": _fp(delay).model_dump(mode="json"),
            "cancel_during_delay": False,
            "start_time_cancel_policy": "cancel_all_at_game_start",
            "fees_enabled": True,
            "fee_rate": _fp(fee_rate).model_dump(mode="json"),
            "fee_exponent": {"atoms": "1", "scale": 0},
            "taker_only": True,
            "maker_fee_rate": {"atoms": "0", "scale": 0},
            "minimum_tick_size": {"atoms": "1", "scale": 2},
            "minimum_order_size": {"atoms": "1", "scale": 0},
            "order_types_supported": ["FAK", "FOK", "GTC", "GTD"],
            "source_document_version": "polymarket-orders-create@2026-07-22",
            "raw_response_hash": "sha256:" + "1" * 64,
        }
    )


def _level(price: str, quantity: str) -> DepthLevelV0:
    return DepthLevelV0(price=_fp(price), quantity=_fp(quantity))


def _book(
    milliseconds: int,
    *,
    bids: tuple[DepthLevelV0, ...],
    asks: tuple[DepthLevelV0, ...],
    suspended: bool = False,
) -> BookSnapshotV0:
    return BookSnapshotV0(
        observed_at=T0 + timedelta(milliseconds=milliseconds),
        bids=bids,
        asks=asks,
        suspended=suspended,
    )


def _order(quantity: str = "4") -> TakerBuyOrderV0:
    return TakerBuyOrderV0(
        order_id="sim-order-1",
        experiment_id="X-07",
        condition_id="condition_nba-game-1",
        created_at=T0,
        quantity=_fp(quantity),
        fee_formula="C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT",
    )


def _stream() -> tuple[BookSnapshotV0, ...]:
    return (
        _book(
            1000,
            bids=(_level("0.49", "10"),),
            asks=(_level("0.50", "10"),),
        ),
        _book(
            1500,
            bids=(_level("0.50", "10"),),
            asks=(_level("0.51", "2"), _level("0.52", "4")),
        ),
        _book(
            2500,
            bids=(_level("0.53", "2"), _level("0.52", "4")),
            asks=(_level("0.54", "10"),),
        ),
    )


def test_buy_consumes_ask_depth_after_rule_and_system_delay() -> None:
    fill = simulate_taker_buy(
        _stream(),
        _order(),
        _rule(),
        own_delay=timedelta(milliseconds=500),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )

    assert fill.executed_at == T0 + timedelta(milliseconds=1500)
    assert fill.vwap == _fp("0.515")
    assert fill.filled_quantity == _fp("4")
    assert fill.levels_consumed == 2
    assert fill.result_label == "PRELIMINARY"
    assert fill.fee_formula == "C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT"
    assert fill.markouts[0].executable_bid_vwap == _fp("0.525")
    assert fill.markouts[0].gross_markout_per_unit == _fp("0.01")


def test_fee_is_computed_from_snapshot_not_a_constant() -> None:
    first = simulate_taker_buy(
        _stream(),
        _order(),
        _rule(fee_rate="0.05"),
        own_delay=timedelta(milliseconds=500),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )
    second = simulate_taker_buy(
        _stream(),
        _order(),
        _rule(fee_rate="0.10"),
        own_delay=timedelta(milliseconds=500),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )

    assert second.fee == _fp("0.09991")
    assert second.fee.to_decimal() == first.fee.to_decimal() * 2
    assert first.rule_snapshot_ref != second.rule_snapshot_ref


def test_formal_result_rejects_missing_rule_snapshot() -> None:
    with pytest.raises(PreliminaryOnlyError, match="venue_rule_snapshot"):
        simulate_taker_buy(
            _stream(),
            _order(),
            None,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=True,
        )


def test_formal_result_is_blocked_in_current_x07_phase() -> None:
    with pytest.raises(PreliminaryOnlyError, match="PRELIMINARY"):
        simulate_taker_buy(
            _stream(),
            _order(),
            _rule(),
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=True,
        )


def test_future_rule_snapshot_is_rejected_as_lookahead() -> None:
    with pytest.raises(SimulationInputError, match="point-in-time"):
        simulate_taker_buy(
            _stream(),
            _order(),
            _rule(fetched_at="2026-07-22T18:00:01Z"),
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def test_insufficient_ask_or_markout_bid_depth_fails_closed() -> None:
    with pytest.raises(InsufficientDepthError, match="ask"):
        simulate_taker_buy(
            _stream(),
            _order(quantity="100"),
            _rule(),
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def test_markout_horizons_must_be_explicit_unique_and_positive() -> None:
    with pytest.raises(SimulationInputError, match="markout_horizons"):
        simulate_taker_buy(
            _stream(),
            _order(),
            _rule(),
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(),
            formal=False,
        )


def test_tca_schema_and_maker_bounds_are_scope_limited() -> None:
    schema = yaml.safe_load(
        (PROJECT_ROOT / "contracts" / "tca" / "v0.schema.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert schema["$id"] == "prediction-market/contracts/tca/v0"
    assert schema["additionalProperties"] is False
    assert {"rule_snapshot_ref", "result_label", "fee_formula", "markouts"} <= set(
        schema["required"]
    )

    maker = (
        PROJECT_ROOT / "artifacts" / "execution-tca" / "maker_bounds_v0.md"
    ).read_text(encoding="utf-8")
    assert "optimistic" in maker
    assert "base" in maker
    assert "pessimistic" in maker
    assert "no trained queue model" in maker
