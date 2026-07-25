from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from prediction_market.sports.nfl_x13_market import (
    CandleOHLC,
    HORIZONS_SECONDS,
    PaginationPage,
    PropositionLeg,
    SourceInterval,
    SubjectRef,
    X13MarketError,
    associate_intervals,
    audit_directional_anomaly,
    build_layer_g_interval,
    build_raw_manifest_receipt,
    deduplicate_observations,
    derive_complement,
    group_kalshi_winner_market,
    measure_reaction,
    normalize_contract,
    normalize_kalshi_candle_bbo,
    normalize_kalshi_trade,
    normalize_polymarket_trade,
    prove_capture_complete,
    verify_raw_manifest_receipt,
)


UTC = timezone.utc
GAME_ID = "2025_14_DAL_DET"
POLY_LOGICAL_MARKET_ID = f"polymarket:{GAME_ID}:winner"
RULE_HASH = "sha256:" + hashlib.sha256(b"frozen rules").hexdigest()


def _at(second: int, *, microsecond: int = 0) -> datetime:
    return datetime(2025, 12, 5, 1, 0, second, microsecond, tzinfo=UTC)


def _poly_trade(
    *,
    native_id: str,
    second: int,
    price: str,
    size: str = "1",
) -> object:
    return normalize_polymarket_trade(
        {
            "transactionHash": native_id,
            "timestamp": int(_at(second).timestamp()),
            "price": price,
            "size": size,
        },
        raw_market_id="0xcondition",
        logical_market_id=POLY_LOGICAL_MARKET_ID,
        outcome="DET",
    )


def test_two_outcome_contract_is_typed_and_analysis_eligible() -> None:
    contract = normalize_contract(
        venue="polymarket",
        venue_market_id="0xcondition",
        family="moneyline",
        period="full_game",
        subject=SubjectRef(kind="team", canonical_id="DET", name="Detroit"),
        measure="wins",
        comparator="eq",
        line=None,
        outcomes=("DET", "DAL"),
        rule_hash=RULE_HASH,
        dependency_game_ids=(GAME_ID,),
        kind="primitive",
    )

    assert contract.venue == "polymarket"
    assert contract.line is None
    assert contract.outcomes == ("DET", "DAL")
    assert contract.dependency_game_ids == (GAME_ID,)
    assert contract.analysis_eligible is True
    assert contract.ineligible_reasons == ()
    assert contract.canonical_hash.startswith("sha256:")


def test_unknown_family_and_unknown_composite_leg_fail_closed() -> None:
    with pytest.raises(X13MarketError, match="unknown proposition family"):
        normalize_contract(
            venue="kalshi",
            venue_market_id="KX-DET",
            family="mystery",
            period="full_game",
            subject=SubjectRef(kind="team", canonical_id="DET", name="Detroit"),
            measure="wins",
            comparator="eq",
            line=None,
            outcomes=("yes", "no"),
            rule_hash=RULE_HASH,
            dependency_game_ids=(GAME_ID,),
            kind="primitive",
        )

    bad_leg = PropositionLeg(
        family="mystery",
        period="full_game",
        subject=SubjectRef(kind="team", canonical_id="DET", name="Detroit"),
        measure="wins",
        comparator="eq",
        line=None,
        outcomes=("yes", "no"),
        dependency_game_ids=(GAME_ID,),
    )
    with pytest.raises(X13MarketError, match="unknown composite leg"):
        normalize_contract(
            venue="kalshi",
            venue_market_id="combo",
            family="same_game_composite",
            period="full_game",
            subject=SubjectRef(kind="game", canonical_id=GAME_ID, name="game"),
            measure="all_legs",
            comparator="eq",
            line=None,
            outcomes=("yes", "no"),
            rule_hash=RULE_HASH,
            dependency_game_ids=(GAME_ID,),
            kind="same_game_composite",
            legs=(bad_leg,),
        )


def test_player_ambiguity_and_cross_game_mve_inventory_are_ineligible() -> None:
    ambiguous = normalize_contract(
        venue="polymarket",
        venue_market_id="player-prop",
        family="player_receiving",
        period="full_game",
        subject=SubjectRef(
            kind="player",
            canonical_id=None,
            name="Chris Johnson",
            candidate_ids=("player-a", "player-b"),
        ),
        measure="receiving_yards",
        comparator="gt",
        line=Decimal("49.5"),
        outcomes=("over", "under"),
        rule_hash=RULE_HASH,
        dependency_game_ids=(GAME_ID,),
        kind="primitive",
    )
    cross_game = normalize_contract(
        venue="kalshi",
        venue_market_id="parlay-inventory",
        family="cross_game_mve_inventory",
        period="full_game",
        subject=SubjectRef(kind="inventory", canonical_id="slip-1", name="slip"),
        measure="all_legs",
        comparator="eq",
        line=None,
        outcomes=("yes", "no"),
        rule_hash=RULE_HASH,
        dependency_game_ids=(GAME_ID, "2025_14_PHI_LAC"),
        kind="cross_game_inventory",
    )

    assert ambiguous.analysis_eligible is False
    assert ambiguous.ineligible_reasons == ("ambiguous_player_subject",)
    assert cross_game.analysis_eligible is False
    assert cross_game.ineligible_reasons == (
        "cross_game_mve_inventory_excluded",
    )


@pytest.mark.parametrize(
    "retired_family",
    ("game_winner", "player_prop", "cross_game_inventory"),
)
def test_retired_family_aliases_fail_closed(retired_family: str) -> None:
    with pytest.raises(X13MarketError, match="unknown proposition family"):
        normalize_contract(
            venue="kalshi",
            venue_market_id="retired-alias",
            family=retired_family,
            period="full_game",
            subject=SubjectRef(
                kind="team",
                canonical_id="DET",
                name="Detroit",
            ),
            measure="wins",
            comparator="eq",
            line=None,
            outcomes=("yes", "no"),
            rule_hash=RULE_HASH,
            dependency_game_ids=(GAME_ID,),
            kind="primitive",
        )


def test_family_and_kind_vocabularies_remain_distinct() -> None:
    with pytest.raises(X13MarketError, match="unknown proposition kind"):
        normalize_contract(
            venue="kalshi",
            venue_market_id="inventory",
            family="cross_game_mve_inventory",
            period="full_game",
            subject=SubjectRef(
                kind="inventory",
                canonical_id="slip-1",
                name="slip",
            ),
            measure="all_legs",
            comparator="eq",
            line=None,
            outcomes=("yes", "no"),
            rule_hash=RULE_HASH,
            dependency_game_ids=(GAME_ID, "2025_14_PHI_LAC"),
            kind="cross_game_mve_inventory",
        )


def test_decimal_line_has_canonical_hash_independent_of_scale() -> None:
    common = {
        "venue": "kalshi",
        "venue_market_id": "KX-SPREAD",
        "family": "spread",
        "period": "full_game",
        "subject": SubjectRef(kind="team", canonical_id="DET", name="Detroit"),
        "measure": "margin",
        "comparator": "gt",
        "outcomes": ("yes", "no"),
        "rule_hash": RULE_HASH,
        "dependency_game_ids": (GAME_ID,),
        "kind": "primitive",
    }

    first = normalize_contract(line=Decimal("2.50"), **common)
    second = normalize_contract(line=Decimal("2.5"), **common)

    assert first.line == Decimal("2.50")
    assert first.canonical_hash == second.canonical_hash
    with pytest.raises(X13MarketError, match="line must be Decimal"):
        normalize_contract(line=2.5, **common)


def test_observed_and_derived_complement_semantics_are_distinct() -> None:
    observed = _poly_trade(
        native_id="0xtrade",
        second=10,
        price="0.62",
        size="12.50",
    )
    derived = derive_complement(observed, outcome="DAL")

    assert observed.price == Decimal("0.62")
    assert observed.size == Decimal("12.50")
    assert observed.provenance == "observed"
    assert observed.render_semantics == "solid"
    assert derived.price == Decimal("0.38")
    assert derived.size == Decimal("12.50")
    assert derived.provenance == "derived_complement"
    assert derived.render_semantics == "dashed"
    assert derived.derived_from_observation_id == observed.observation_id
    assert derived.bid is None and derived.ask is None
    assert derived.observation_id != observed.observation_id


def test_polymarket_trade_does_not_fabricate_bbo() -> None:
    observation = _poly_trade(
        native_id="0xtrade",
        second=10,
        price="0.50",
    )

    assert observation.kind == "trade"
    assert observation.raw_market_id == "0xcondition"
    assert observation.logical_market_id == POLY_LOGICAL_MARKET_ID
    assert observation.source_interval.start == _at(10)
    assert observation.source_interval.end == _at(11)
    assert observation.source_interval.end_inclusive is False
    assert observation.bid is None
    assert observation.ask is None
    assert observation.has_executable_bbo is False

    with pytest.raises(X13MarketError, match="price must be an exact decimal"):
        normalize_polymarket_trade(
            {
                "transactionHash": "0xbinary-float",
                "timestamp": int(_at(10).timestamp()),
                "price": 0.50,
                "size": "1",
            },
            raw_market_id="0xcondition",
            logical_market_id=POLY_LOGICAL_MARKET_ID,
            outcome="DET",
        )


def test_kalshi_trade_and_one_minute_candle_bbo_remain_separate() -> None:
    trade = normalize_kalshi_trade(
        {
            "trade_id": "kalshi-trade",
            "created_time": "2025-12-05T01:00:10.123456Z",
            "is_block_trade": False,
            "taker_side": "yes",
            "taker_outcome_side": "yes",
            "yes_price_dollars": "0.61",
            "no_price_dollars": "0.39",
            "count_fp": "4.00",
        },
        ticker="KXNFLGAME-25DEC04DALDET-DET",
        outcome="DET",
        logical_market_id=GAME_ID,
    )
    candle = normalize_kalshi_candle_bbo(
        {
            "end_period_ts": int(_at(0).timestamp()) + 60,
            "yes_bid": {
                "open": "0.55",
                "high": "0.59",
                "low": "0.54",
                "close": "0.58",
            },
            "yes_ask": {
                "open": "0.60",
                "high": "0.63",
                "low": "0.59",
                "close": "0.62",
            },
            "volume": "50.00",
            "open_interest": "125.00",
        },
        ticker="KXNFLGAME-25DEC04DALDET-DET",
        outcome="DET",
        logical_market_id=GAME_ID,
    )

    assert trade.kind == "trade"
    assert trade.price == Decimal("0.61")
    assert trade.bid is None and trade.ask is None
    assert trade.source_interval.start == _at(10)
    assert trade.source_interval.end == _at(11)
    assert trade.native_source_time == _at(10, microsecond=123456)
    assert trade.is_block_trade is False
    assert trade.taker_side == "yes"
    assert trade.taker_outcome_side == "yes"
    assert trade.yes_price_dollars == Decimal("0.61")
    assert trade.no_price_dollars == Decimal("0.39")
    assert trade.primary_path_eligible is True
    assert candle.kind == "kalshi_1m_candle_bbo"
    assert candle.price is None
    assert candle.bid == Decimal("0.58")
    assert candle.ask == Decimal("0.62")
    assert candle.size == Decimal("50.00")
    assert candle.source_interval.start == _at(0)
    assert candle.source_interval.end == _at(0) + timedelta(minutes=1)
    assert candle.has_executable_bbo is True
    assert candle.yes_bid_ohlc == CandleOHLC(
        open=Decimal("0.55"),
        high=Decimal("0.59"),
        low=Decimal("0.54"),
        close=Decimal("0.58"),
    )
    assert candle.yes_ask_ohlc == CandleOHLC(
        open=Decimal("0.60"),
        high=Decimal("0.63"),
        low=Decimal("0.59"),
        close=Decimal("0.62"),
    )
    assert candle.volume == Decimal("50.00")
    assert candle.open_interest == Decimal("125.00")


def test_kalshi_trade_microseconds_are_provenance_not_ordering_evidence() -> None:
    common = {
        "is_block_trade": False,
        "taker_side": "no",
        "taker_outcome_side": "no",
        "yes_price_dollars": "0.61",
        "no_price_dollars": "0.39",
        "count_fp": "1",
    }
    early_native = normalize_kalshi_trade(
        {
            **common,
            "trade_id": "kalshi-trade-a",
            "created_time": "2025-12-05T01:00:10.000001Z",
        },
        ticker="KXNFLGAME-25DEC04DALDET-DET",
        outcome="DET",
        logical_market_id=GAME_ID,
    )
    late_native = normalize_kalshi_trade(
        {
            **common,
            "trade_id": "kalshi-trade-b",
            "created_time": "2025-12-05T01:00:10.999999Z",
        },
        ticker="KXNFLGAME-25DEC04DALDET-DET",
        outcome="DET",
        logical_market_id=GAME_ID,
    )

    assert early_native.native_source_time == _at(10, microsecond=1)
    assert late_native.native_source_time == _at(10, microsecond=999999)
    assert early_native.source_interval == late_native.source_interval
    assert early_native.source_interval == SourceInterval(
        _at(10), _at(11), False
    )


def test_kalshi_trade_validates_binary_arithmetic_and_excludes_blocks() -> None:
    block = normalize_kalshi_trade(
        {
            "trade_id": "kalshi-block",
            "created_time": "2025-12-05T01:00:10.123456Z",
            "is_block_trade": True,
            "taker_side": "yes",
            "taker_outcome_side": "no",
            "yes_price_dollars": "0.61",
            "no_price_dollars": "0.39",
            "count_fp": "4.00",
        },
        ticker="KXNFLGAME-25DEC04DALDET-DET",
        outcome="DET",
        logical_market_id=GAME_ID,
    )

    assert block.is_block_trade is True
    assert block.primary_path_eligible is False
    with pytest.raises(X13MarketError, match="sum to one"):
        normalize_kalshi_trade(
            {
                "trade_id": "bad-arithmetic",
                "created_time": "2025-12-05T01:00:10Z",
                "is_block_trade": False,
                "taker_side": "yes",
                "taker_outcome_side": "yes",
                "yes_price_dollars": "0.61",
                "no_price_dollars": "0.38",
                "count_fp": "1",
            },
            ticker="KXNFLGAME-25DEC04DALDET-DET",
            outcome="DET",
            logical_market_id=GAME_ID,
        )


def test_block_trades_are_excluded_from_primary_reaction_path() -> None:
    def trade(
        native_id: str,
        second: int,
        price: str,
        *,
        is_block_trade: bool,
    ) -> object:
        return normalize_kalshi_trade(
            {
                "trade_id": native_id,
                "created_time": _at(second)
                .isoformat()
                .replace("+00:00", "Z"),
                "is_block_trade": is_block_trade,
                "taker_side": "yes",
                "taker_outcome_side": "yes",
                "yes_price_dollars": price,
                "no_price_dollars": str(Decimal("1") - Decimal(price)),
                "count_fp": "1",
            },
            ticker="KXNFLGAME-25DEC04DALDET-DET",
            outcome="DET",
            logical_market_id=GAME_ID,
        )

    reaction = measure_reaction(
        event_interval=build_layer_g_interval(
            _at(10), delay_seconds=Decimal("0")
        ),
        observations=(
            trade("pre", 8, "0.50", is_block_trade=False),
            trade("block", 12, "0.80", is_block_trade=True),
            trade("eligible", 13, "0.60", is_block_trade=False),
        ),
        venue="kalshi",
        market_id=GAME_ID,
        outcome="DET",
    )

    assert reaction.first_post_price == Decimal("0.60")
    assert reaction.horizons[2].status == "missing"
    assert reaction.horizons[5].last_actual_price == Decimal("0.60")


def test_kalshi_candle_validates_full_ohlc_and_nonnegative_inventory() -> None:
    common = {
        "end_period_ts": int(_at(0).timestamp()) + 60,
        "yes_bid": {
            "open": "0.55",
            "high": "0.59",
            "low": "0.54",
            "close": "0.58",
        },
        "yes_ask": {
            "open": "0.60",
            "high": "0.63",
            "low": "0.59",
            "close": "0.62",
        },
        "volume": "50.00",
        "open_interest": "125.00",
    }
    normalize = lambda raw: normalize_kalshi_candle_bbo(
        raw,
        ticker="KXNFLGAME-25DEC04DALDET-DET",
        outcome="DET",
        logical_market_id=GAME_ID,
    )

    with pytest.raises(X13MarketError, match="yes_bid.open"):
        normalize(
            {
                **common,
                "yes_bid": {
                    key: value
                    for key, value in common["yes_bid"].items()
                    if key != "open"
                },
            }
        )
    with pytest.raises(X13MarketError, match="crossed BBO"):
        normalize(
            {
                **common,
                "yes_bid": {**common["yes_bid"], "high": "0.64"},
            }
        )
    with pytest.raises(X13MarketError, match="open_interest"):
        normalize({**common, "open_interest": "-1"})


def test_two_kalshi_team_tickers_form_one_logical_market_without_losing_raw_ids(
) -> None:
    logical = group_kalshi_winner_market(
        game_id=GAME_ID,
        event_ticker="KXNFLGAME-25DEC04DALDET",
        team_tickers={
            "DET": "KXNFLGAME-25DEC04DALDET-DET",
            "DAL": "KXNFLGAME-25DEC04DALDET-DAL",
        },
    )

    assert logical.logical_market_id == f"kalshi:{GAME_ID}:winner"
    assert logical.outcomes == ("DAL", "DET")
    assert logical.raw_contracts == (
        ("DAL", "KXNFLGAME-25DEC04DALDET-DAL"),
        ("DET", "KXNFLGAME-25DEC04DALDET-DET"),
    )


def test_stable_observation_id_deduplicates_exact_repeats() -> None:
    first = _poly_trade(native_id="same", second=10, price="0.55")
    duplicate = _poly_trade(native_id="same", second=10, price="0.55")
    other = _poly_trade(native_id="other", second=11, price="0.56")

    assert first.observation_id == duplicate.observation_id
    assert deduplicate_observations(
        (first, duplicate, other)
    ) == (first, other)


def test_pagination_requires_unique_cursor_chain_and_unsaturated_terminal_leaf(
) -> None:
    proof = prove_capture_complete(
        (
            PaginationPage(
                cursor_in=None,
                cursor_out="next",
                item_count=2,
                terminal=False,
            ),
            PaginationPage(
                cursor_in="next",
                cursor_out=None,
                item_count=1,
                terminal=True,
            ),
        ),
        page_limit=2,
    )
    assert proof.complete is True
    assert proof.page_count == 2
    assert proof.item_count == 3
    assert proof.terminal_proof == "explicit_unsaturated_terminal_page"

    with pytest.raises(X13MarketError, match="repeated cursor"):
        prove_capture_complete(
            (
                PaginationPage(None, "same", 1, False),
                PaginationPage("same", "same", 1, False),
            ),
            page_limit=2,
        )
    with pytest.raises(X13MarketError, match="terminal proof"):
        prove_capture_complete(
            (PaginationPage(None, "next", 1, False),),
            page_limit=2,
        )
    with pytest.raises(X13MarketError, match="saturated terminal leaf"):
        prove_capture_complete(
            (PaginationPage(None, None, 2, True),),
            page_limit=2,
        )


def test_raw_manifest_receipt_detects_object_and_manifest_tamper() -> None:
    raw = b'{"trades":[1,2]}'
    object_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    manifest = {
        "dataset_id": "DS-KALSHI-HISTORICAL",
        "object_sha256": object_hash,
        "partition": "trades-window",
    }
    receipt = build_raw_manifest_receipt(raw, manifest)

    assert verify_raw_manifest_receipt(raw, manifest, receipt) is True
    with pytest.raises(X13MarketError, match="raw object tamper"):
        verify_raw_manifest_receipt(raw + b" ", manifest, receipt)
    with pytest.raises(X13MarketError, match="manifest tamper"):
        verify_raw_manifest_receipt(
            raw,
            {**manifest, "partition": "different"},
            receipt,
        )


def test_layer_g_and_m_same_second_overlap_is_order_ambiguous() -> None:
    event = build_layer_g_interval(_at(10), delay_seconds=Decimal("0"))
    trade = _poly_trade(native_id="same-second", second=10, price="0.55")

    association = associate_intervals(event, trade.source_interval)

    assert event.start == _at(10)
    assert event.end == _at(11)
    assert event.end_inclusive is True
    assert association.overlap is True
    assert association.order_ambiguous is True
    assert association.order_relation is None


def test_delay_expands_uncertainty_without_manufacturing_order() -> None:
    trade = _poly_trade(native_id="later", second=12, price="0.55")
    no_delay = associate_intervals(
        build_layer_g_interval(_at(10), delay_seconds=Decimal("0")),
        trade.source_interval,
    )
    delayed = associate_intervals(
        build_layer_g_interval(_at(10), delay_seconds=Decimal("2")),
        trade.source_interval,
    )

    assert no_delay.order_relation == "market_after_event"
    assert no_delay.order_ambiguous is False
    assert delayed.order_relation is None
    assert delayed.order_ambiguous is True


def test_sparse_horizons_require_new_actual_observation_without_forward_fill() -> None:
    event = build_layer_g_interval(_at(10), delay_seconds=Decimal("0"))
    observations = (
        _poly_trade(native_id="pre", second=8, price="0.50", size="3"),
        _poly_trade(native_id="h2", second=12, price="0.60", size="2"),
        _poly_trade(native_id="h10", second=16, price="0.70", size="1"),
        _poly_trade(native_id="h60", second=41, price="0.45", size="4"),
    )

    reaction = measure_reaction(
        event_interval=event,
        observations=observations,
        venue="polymarket",
        market_id="0xcondition",
        outcome="DET",
    )

    assert HORIZONS_SECONDS == (1, 2, 5, 10, 30, 60)
    assert reaction.pre_actual_price == Decimal("0.50")
    assert reaction.first_post_price == Decimal("0.60")
    assert reaction.horizons[1].status == "missing"
    assert reaction.horizons[2].status == "observed"
    assert reaction.horizons[2].last_actual_price == Decimal("0.60")
    assert reaction.horizons[2].vwap == Decimal("0.60")
    assert reaction.horizons[2].signed_delta == Decimal("0.10")
    assert reaction.horizons[2].observation_count == 1
    assert reaction.horizons[2].volume == Decimal("2")
    assert reaction.horizons[5].status == "missing"
    assert reaction.horizons[5].last_actual_price is None
    assert reaction.horizons[10].last_actual_price == Decimal("0.70")
    assert reaction.horizons[30].status == "missing"
    assert reaction.horizons[60].last_actual_price == Decimal("0.45")
    assert reaction.net_60s == Decimal("-0.05")
    assert reaction.max_excursion == Decimal("0.20")
    assert reaction.overshoot_candidate is True
    assert reaction.reversal_candidate is True
    assert reaction.venue_direction == "down"


def test_next_salient_episode_contaminates_crossing_horizons() -> None:
    event = build_layer_g_interval(_at(10), delay_seconds=Decimal("0"))
    observations = (
        _poly_trade(native_id="pre", second=8, price="0.50"),
        _poly_trade(native_id="h2", second=12, price="0.60"),
        _poly_trade(native_id="after-next", second=16, price="0.70"),
    )

    reaction = measure_reaction(
        event_interval=event,
        observations=observations,
        venue="polymarket",
        market_id="0xcondition",
        outcome="DET",
        next_salient_at=_at(15),
    )

    assert reaction.horizons[2].status == "observed"
    assert reaction.horizons[5].status == "contaminated"
    assert reaction.horizons[10].status == "contaminated"
    assert reaction.horizons[10].last_actual_price is None
    assert reaction.contaminated is True


def test_directional_anomaly_audit_is_fail_closed_in_required_order() -> None:
    blocked = audit_directional_anomaly(
        orientation_valid=False,
        adjudication_flags=frozenset({"review", "penalty"}),
        score_possession_valid=False,
        same_second_ambiguous=True,
        contaminated=True,
        stale=True,
        wrong_family=True,
        venue_signed_deltas={
            "polymarket": Decimal("0.10"),
            "kalshi": Decimal("-0.10"),
        },
    )

    assert blocked.status == "blocked"
    assert blocked.primary_reason == "orientation_unresolved"
    assert blocked.reason_codes == (
        "orientation_unresolved",
        "adjudication_unresolved",
        "score_or_possession_unresolved",
        "same_second_order_ambiguous",
        "next_salient_episode_contamination",
        "stale_or_wrong_market_family",
    )
    assert "diagnostic_fair_delta_disagreement" not in blocked.reason_codes

    diagnostic = audit_directional_anomaly(
        orientation_valid=True,
        adjudication_flags=frozenset(),
        score_possession_valid=True,
        same_second_ambiguous=False,
        contaminated=False,
        stale=False,
        wrong_family=False,
        venue_signed_deltas={
            "polymarket": Decimal("0.10"),
            "kalshi": Decimal("-0.10"),
        },
    )
    assert diagnostic.status == "diagnostic"
    assert diagnostic.primary_reason == "diagnostic_fair_delta_disagreement"
    assert diagnostic.reason_codes == ("diagnostic_fair_delta_disagreement",)
