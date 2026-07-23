from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from prediction_market.contracts import (
    FixedPointV0,
    VenueRuleSnapshotV0,
    canonical_json,
    canonical_sha256,
    validate_contract_v0,
)
from prediction_market.execution import (
    BookSnapshotV0,
    DepthLevelV0,
    InsufficientDepthError,
    PreliminaryOnlyError,
    SimulationInputError,
    TakerBuyOrderV0,
    simulate_taker_buy,
)
from prediction_market.recording import (
    FormalReplayRejected,
    VenueRuleStore,
    capture_venue_rule_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 7, 22, 18, 0, tzinfo=UTC)
VENUE = "polymarket"
MARKET_ID = "market_nba-game-1"
CONDITION_ID = "condition_nba-game-1"


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
            "venue": VENUE,
            "condition_id": CONDITION_ID,
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


def _rule_document(
    *,
    effective_from: str = "2026-07-22T17:00:00Z",
    delay: str = "1.0",
    fee_rate: str = "0.05",
    fee_exponent: str = "1",
) -> dict[str, object]:
    return {
        "effective_from": effective_from,
        "game_start_time": "2026-07-22T19:00:00Z",
        "seconds_delay": int(delay) if delay.isdigit() else delay,
        "cancel_during_delay": False,
        "start_time_cancel_policy": "cancel_all_at_game_start",
        "fees_enabled": True,
        "fee_rate": fee_rate,
        "fee_exponent": fee_exponent,
        "taker_only": True,
        "maker_fee_rate": "0",
        "minimum_tick_size": "0.01",
        "minimum_order_size": "1",
        "order_types_supported": ["FAK", "FOK", "GTC", "GTD"],
    }


def _capture_rule(
    root: Path,
    *,
    fetched_at: str,
    effective_from: str = "2026-07-22T17:00:00Z",
    delay: str = "1",
    fee_rate: str = "0.05",
    fee_exponent: str = "1",
    condition_id: str = CONDITION_ID,
    venue: str = VENUE,
    missing_field: str | None = None,
) -> None:
    document = _rule_document(
        effective_from=effective_from,
        delay=delay,
        fee_rate=fee_rate,
        fee_exponent=fee_exponent,
    )
    if missing_field is not None:
        del document[missing_field]
    capture_venue_rule_snapshot(
        json.dumps(document, separators=(",", ":")).encode(),
        raw_root=root,
        venue=venue,
        market_id=MARKET_ID,
        condition_id=condition_id,
        fetched_at=fetched_at,
        source_document_version=f"official-response@{fetched_at}",
    )


def _level(price: str, quantity: str) -> DepthLevelV0:
    return DepthLevelV0(price=_fp(price), quantity=_fp(quantity))


def _book(
    milliseconds: int,
    *,
    bids: tuple[DepthLevelV0, ...],
    asks: tuple[DepthLevelV0, ...],
    suspended: bool = False,
    venue: str = VENUE,
    condition_id: str = CONDITION_ID,
) -> BookSnapshotV0:
    local_receive_at = T0 + timedelta(milliseconds=milliseconds)
    hash_material = {
        "book_version": "v0",
        "venue": venue,
        "condition_id": condition_id,
        "local_receive_at": local_receive_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "bids": [
            {
                "price": level.price.model_dump(mode="json"),
                "quantity": level.quantity.model_dump(mode="json"),
            }
            for level in bids
        ],
        "asks": [
            {
                "price": level.price.model_dump(mode="json"),
                "quantity": level.quantity.model_dump(mode="json"),
            }
            for level in asks
        ],
        "suspended": suspended,
    }
    return BookSnapshotV0(
        venue=venue,
        condition_id=condition_id,
        local_receive_at=local_receive_at,
        bids=bids,
        asks=asks,
        suspended=suspended,
        content_sha256=canonical_sha256(hash_material),
    )


def _order(quantity: str = "4") -> TakerBuyOrderV0:
    return TakerBuyOrderV0(
        order_id="sim-order-1",
        experiment_id="X-07",
        venue=VENUE,
        market_id=MARKET_ID,
        condition_id=CONDITION_ID,
        created_at=T0,
        quantity=_fp(quantity),
        fee_formula="C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT",
    )


def test_book_snapshot_rejects_unverified_content_hash() -> None:
    with pytest.raises(SimulationInputError, match="content_sha256"):
        BookSnapshotV0(
            venue=VENUE,
            condition_id=CONDITION_ID,
            local_receive_at=T0,
            bids=(_level("0.49", "10"),),
            asks=(_level("0.50", "10"),),
            suspended=False,
            content_sha256="sha256:" + "0" * 64,
        )


def test_simulator_selects_latest_strict_rule_snapshot_as_of_order(
    tmp_path: Path,
) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(rule_root, fetched_at="2026-07-22T17:00:00Z", delay="0")
    _capture_rule(rule_root, fetched_at="2026-07-22T17:59:00Z", delay="1")
    books = (
        _book(
            0,
            bids=(_level("0.49", "10"),),
            asks=(_level("0.50", "10"),),
        ),
        _book(
            1000,
            bids=(_level("0.50", "10"),),
            asks=(_level("0.51", "10"),),
        ),
        _book(
            2000,
            bids=(_level("0.51", "10"),),
            asks=(_level("0.52", "10"),),
        ),
    )

    fill = simulate_taker_buy(
        books,
        _order(),
        VenueRuleStore(rule_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(0),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )

    assert fill.executed_at == "2026-07-22T18:00:01.000000Z"
    assert fill.venue_delay_microseconds == 1_000_000


@pytest.mark.parametrize("state", ["missing", "invalid", "stale"])
def test_simulator_fails_closed_when_rule_store_is_not_usable(
    tmp_path: Path,
    state: str,
) -> None:
    rule_root = tmp_path / state
    rule_root.mkdir()
    if state == "invalid":
        _capture_rule(
            rule_root,
            fetched_at="2026-07-22T17:59:00Z",
            missing_field="fee_rate",
        )
    elif state == "stale":
        _capture_rule(
            rule_root,
            fetched_at="2026-07-22T16:00:00Z",
            effective_from="2026-07-22T16:00:00Z",
        )

    with pytest.raises(FormalReplayRejected, match=state):
        simulate_taker_buy(
            _stream(),
            _order(),
            VenueRuleStore(rule_root),
            rule_max_age_seconds=60,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def test_simulator_rejects_direct_caller_rule_snapshot() -> None:
    with pytest.raises(SimulationInputError, match="VenueRuleStore"):
        simulate_taker_buy(
            _stream(),
            _order(),
            _rule(),
            rule_max_age_seconds=3600,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def test_simulator_rejects_book_lineage_mismatch(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(rule_root, fetched_at="2026-07-22T17:59:00Z")
    books = (
        _book(
            1500,
            bids=(_level("0.50", "10"),),
            asks=(_level("0.51", "10"),),
            condition_id="condition_other",
        ),
        _book(
            2500,
            bids=(_level("0.50", "10"),),
            asks=(_level("0.51", "10"),),
            condition_id="condition_other",
        ),
    )

    with pytest.raises(SimulationInputError, match="lineage"):
        simulate_taker_buy(
            books,
            _order(),
            VenueRuleStore(rule_root),
            rule_max_age_seconds=3600,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def _multi_level_fee_stream() -> tuple[BookSnapshotV0, ...]:
    return (
        _book(
            0,
            bids=(_level("0.10", "2"),),
            asks=(_level("0.20", "1"), _level("0.60", "1")),
        ),
        _book(
            1000,
            bids=(_level("0.50", "1"), _level("0.30", "1")),
            asks=(_level("0.70", "2"),),
        ),
    )


def test_entry_and_exit_fees_sum_each_actual_depth_fill(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(
        rule_root,
        fetched_at="2026-07-22T17:59:00Z",
        delay="0",
        fee_exponent="2",
    )

    fill = simulate_taker_buy(
        _multi_level_fee_stream(),
        _order(quantity="2"),
        VenueRuleStore(rule_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(0),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )
    markout = fill.markouts[0]

    assert fill.entry_vwap == _fp("0.4")
    assert fill.entry_fee == _fp("0.00416")
    assert fill.entry_levels_consumed == 2
    assert markout.executable_bid_vwap == _fp("0.4")
    assert markout.exit_fee == _fp("0.00533")
    assert markout.exit_levels_consumed == 2
    assert markout.gross_markout_per_unit == _fp("0")
    assert markout.net_markout_per_unit == _fp("-0.004745")


def test_exit_fee_uses_latest_rule_as_of_markout_time(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(
        rule_root,
        fetched_at="2026-07-22T17:59:00Z",
        delay="0",
        fee_rate="0.05",
        fee_exponent="2",
    )
    _capture_rule(
        rule_root,
        fetched_at="2026-07-22T18:00:00.500000Z",
        effective_from="2026-07-22T18:00:00.500000Z",
        delay="0",
        fee_rate="0.10",
        fee_exponent="2",
    )

    fill = simulate_taker_buy(
        _multi_level_fee_stream(),
        _order(quantity="2"),
        VenueRuleStore(rule_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(0),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )
    markout = fill.markouts[0]

    assert fill.entry_fee == _fp("0.00416")
    assert markout.exit_fee == _fp("0.01066")
    assert markout.rule_snapshot_ref != fill.entry_rule_snapshot_ref


def test_tca_runtime_contract_and_schema_round_trip_exactly(tmp_path: Path) -> None:
    from prediction_market import contracts as contracts_module

    rule_root = tmp_path / "rules"
    _capture_rule(
        rule_root,
        fetched_at="2026-07-22T17:59:00Z",
        delay="0",
        fee_exponent="2",
    )
    record = simulate_taker_buy(
        _multi_level_fee_stream(),
        _order(quantity="2"),
        VenueRuleStore(rule_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(0),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )

    assert isinstance(record, contracts_module.TcaRecordV0)
    validated = validate_contract_v0("tca/v0.schema.yaml", record)
    encoded = canonical_json(validated)
    replayed = validate_contract_v0("tca/v0.schema.yaml", json.loads(encoded))
    assert replayed == record
    assert canonical_json(replayed) == encoded
    assert canonical_sha256(replayed) == canonical_sha256(record)

    schema = yaml.safe_load(
        (PROJECT_ROOT / "contracts" / "tca" / "v0.schema.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(schema["required"]) == set(schema["properties"]) == set(
        contracts_module.TcaRecordV0.model_fields
    )
    markout_schema = schema["properties"]["markouts"]["items"]
    assert set(markout_schema["required"]) == set(markout_schema["properties"]) == set(
        contracts_module.TcaMarkoutV0.model_fields
    )

    formal = json.loads(encoded)
    formal["result_label"] = "FORMAL"
    with pytest.raises(ValueError):
        validate_contract_v0("tca/v0.schema.yaml", formal)


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


def test_buy_consumes_ask_depth_after_rule_and_system_delay(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(rule_root, fetched_at="2026-07-22T17:59:00Z")
    fill = simulate_taker_buy(
        _stream(),
        _order(),
        VenueRuleStore(rule_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(milliseconds=500),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )

    assert fill.executed_at == "2026-07-22T18:00:01.500000Z"
    assert fill.entry_vwap == _fp("0.515")
    assert fill.filled_quantity == _fp("4")
    assert fill.entry_levels_consumed == 2
    assert fill.result_label == "PRELIMINARY"
    assert fill.fee_formula == "C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT"
    assert fill.markouts[0].executable_bid_vwap == _fp("0.525")
    assert fill.markouts[0].gross_markout_per_unit == _fp("0.01")


def test_fee_is_computed_from_snapshot_not_a_constant(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _capture_rule(
        first_root, fetched_at="2026-07-22T17:59:00Z", fee_rate="0.05"
    )
    _capture_rule(
        second_root, fetched_at="2026-07-22T17:59:00Z", fee_rate="0.10"
    )
    first = simulate_taker_buy(
        _stream(),
        _order(),
        VenueRuleStore(first_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(milliseconds=500),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )
    second = simulate_taker_buy(
        _stream(),
        _order(),
        VenueRuleStore(second_root),
        rule_max_age_seconds=3600,
        own_delay=timedelta(milliseconds=500),
        markout_horizons=(timedelta(seconds=1),),
        formal=False,
    )

    assert second.entry_fee == _fp("0.0999")
    assert second.entry_fee.to_decimal() == first.entry_fee.to_decimal() * 2
    assert first.entry_rule_snapshot_ref != second.entry_rule_snapshot_ref


def test_formal_result_is_blocked_in_current_x07_phase(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    rule_root.mkdir()
    with pytest.raises(PreliminaryOnlyError, match="PRELIMINARY"):
        simulate_taker_buy(
            _stream(),
            _order(),
            VenueRuleStore(rule_root),
            rule_max_age_seconds=3600,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=True,
        )


def test_future_rule_snapshot_is_rejected_as_lookahead(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(rule_root, fetched_at="2026-07-22T18:00:01Z")
    with pytest.raises(FormalReplayRejected, match="missing"):
        simulate_taker_buy(
            _stream(),
            _order(),
            VenueRuleStore(rule_root),
            rule_max_age_seconds=3600,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def test_insufficient_ask_depth_fails_closed(tmp_path: Path) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(rule_root, fetched_at="2026-07-22T17:59:00Z")
    with pytest.raises(InsufficientDepthError, match="ask"):
        simulate_taker_buy(
            _stream(),
            _order(quantity="100"),
            VenueRuleStore(rule_root),
            rule_max_age_seconds=3600,
            own_delay=timedelta(milliseconds=500),
            markout_horizons=(timedelta(seconds=1),),
            formal=False,
        )


def test_markout_horizons_must_be_explicit_unique_and_positive(
    tmp_path: Path,
) -> None:
    rule_root = tmp_path / "rules"
    _capture_rule(rule_root, fetched_at="2026-07-22T17:59:00Z")
    with pytest.raises(SimulationInputError, match="markout_horizons"):
        simulate_taker_buy(
            _stream(),
            _order(),
            VenueRuleStore(rule_root),
            rule_max_age_seconds=3600,
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
    assert {
        "entry_rule_snapshot_ref",
        "entry_book_snapshot_ref",
        "entry_fee",
        "result_label",
        "fee_formula",
        "markouts",
    } <= set(schema["required"])
    assert {
        "gross_markout_per_unit",
        "exit_fee",
        "net_markout_per_unit",
        "rule_snapshot_ref",
        "book_snapshot_ref",
    } <= set(schema["properties"]["markouts"]["items"]["required"])

    taker = (
        PROJECT_ROOT
        / "artifacts"
        / "execution-tca"
        / "taker_simulator_spec_v0.md"
    ).read_text(encoding="utf-8")
    assert "VenueRuleStore" in taker
    assert "does not accept a caller-supplied snapshot" in taker
    assert "each consumed depth level" in taker
    assert "net markout" in taker
    assert "TcaRecordV0" in taker
    assert "content_sha256" in taker

    maker = (
        PROJECT_ROOT / "artifacts" / "execution-tca" / "maker_bounds_v0.md"
    ).read_text(encoding="utf-8")
    assert "optimistic" in maker
    assert "base" in maker
    assert "pessimistic" in maker
    assert "no trained queue model" in maker
