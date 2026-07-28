from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from prediction_market.sports.nfl_x13_market import (
    SourceInterval,
    derive_complement,
    normalize_kalshi_candle_bbo,
    normalize_kalshi_trade,
    normalize_polymarket_trade,
)
from prediction_market.sports.nfl_x13_replay import (
    X13GameLedgerV1,
    build_x13_game_ledger,
)
from prediction_market.sports.nfl_x13_spec import MarketContractV1

GAME_ID = "2025_14_DAL_DET"
RULE_SHA256 = "sha256:" + hashlib.sha256(b"x13-rule").hexdigest()


def _at(second: int) -> datetime:
    return datetime(2025, 12, 5, 1, 0, tzinfo=UTC) + timedelta(seconds=second)


def _play(
    play_id: int,
    order_sequence: int,
    second: int | None,
    **changes: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": GAME_ID,
        "play_id": play_id,
        "order_sequence": order_sequence,
        "_raw_record_ordinal": order_sequence,
        "qtr": 1,
        "quarter_seconds_remaining": 900 - second if second is not None else 899,
        "game_seconds_remaining": 3600 - second if second is not None else 3599,
        "home_team": "DET",
        "away_team": "DAL",
        "posteam": "DET",
        "defteam": "DAL",
        "fixed_drive": 1,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 65,
        "total_home_score": 0,
        "total_away_score": 0,
        "posteam_score": 0,
        "defteam_score": 0,
        "posteam_score_post": 0,
        "defteam_score_post": 0,
        "play_type": "pass",
        "pass_attempt": 1,
    }
    if second is not None:
        row["time_of_day"] = _at(second).isoformat().replace("+00:00", "Z")
    row.update(changes)
    return row


def _ledger(*, missing_first_time: bool = False) -> X13GameLedgerV1:
    return build_x13_game_ledger(
        [
            _play(10, 1, None if missing_first_time else 10),
            _play(20, 2, 50, play_type="run", pass_attempt=0, rush_attempt=1),
        ]
    )


def _contract(
    *,
    venue: str = "polymarket",
    native_id: str = "0xcondition",
    logical_market_id: str | None = None,
    subject: str = "DET",
    outcomes: tuple[str, ...] = ("DET", "DAL"),
    dependency_game_ids: tuple[str, ...] = (GAME_ID,),
    analysis_eligible: bool = True,
) -> MarketContractV1:
    return MarketContractV1(
        contract_id=f"{venue}:{native_id}",
        logical_market_id=logical_market_id or f"{venue}:{native_id}:winner",
        venue=venue,
        family="moneyline",
        period="full_game",
        subject=subject,
        measure="wins",
        comparator="eq",
        line=None,
        outcomes=outcomes,
        rule_sha256=RULE_SHA256,
        dependency_game_ids=dependency_game_ids,
        kind="primitive",
        analysis_eligible=analysis_eligible,
    )


def _poly(
    second: int,
    price: str,
    *,
    outcome: str = "DET",
    native_id: str | None = None,
):
    return normalize_polymarket_trade(
        {
            "transactionHash": native_id or f"0xtrade-{second}-{price}",
            "timestamp": int(_at(second).timestamp()),
            "price": price,
            "size": "2",
        },
        raw_market_id="0xcondition",
        logical_market_id="polymarket:0xcondition:winner",
        outcome=outcome,
    )


def _kalshi(
    second: int,
    price: str,
    *,
    native_id: str,
    microsecond: int = 0,
    is_block_trade: bool = False,
):
    created_time = _at(second).replace(microsecond=microsecond)
    return normalize_kalshi_trade(
        {
            "trade_id": native_id,
            "created_time": created_time.isoformat().replace("+00:00", "Z"),
            "is_block_trade": is_block_trade,
            "taker_side": "yes",
            "taker_outcome_side": "yes",
            "yes_price_dollars": price,
            "no_price_dollars": str(Decimal(1) - Decimal(price)),
            "count_fp": "3",
        },
        ticker="KXNFLGAME-DET",
        outcome="yes",
        logical_market_id="kalshi:det-winner",
    )


def _metrics(row) -> dict[str, object]:
    return dict(row.candidate.metrics)


def test_layer_g_audit_builds_only_the_frozen_delay_grid() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_g

    audit = audit_layer_g(_ledger())

    assert audit.status == "PASS"
    first = [
        item
        for item in audit.episode_intervals
        if item.episode_id == _ledger().episodes[0].episode_id
    ]
    assert tuple(item.delay_scenario_seconds for item in first) == (
        0,
        1,
        2,
        3,
        5,
        10,
    )
    assert first[0].source_interval.start == _at(10)
    assert first[0].source_interval.end == _at(11)
    assert first[0].source_interval.end_inclusive is True
    assert first[-1].source_interval.end == _at(21)
    assert first[0].source_play_id == "10"


def test_layer_g_fails_closed_on_missing_source_time_or_future_endpoint() -> None:
    from prediction_market.sports.nfl_x13_association import (
        AssociationEvidenceError,
        audit_layer_g,
        run_x13_association,
    )

    missing = _ledger(missing_first_time=True)
    assert audit_layer_g(missing).status == "FAIL"
    with pytest.raises(AssociationEvidenceError, match="Layer G"):
        run_x13_association(missing, (_contract(),), ())

    good = _ledger()
    first = replace(
        good.episodes[0],
        post_state=replace(good.events[-1].post_state, home_score=99),
    )
    tampered = replace(good, episodes=(first, *good.episodes[1:]))
    audit = audit_layer_g(tampered)
    assert audit.status == "FAIL"
    assert "episode_post_state_not_bound_to_episode_evidence" in audit.errors


def test_layer_g_accepts_last_observed_state_inside_nonreversed_episode() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_g

    ledger = _ledger()
    first_event = ledger.events[0]
    trailing_without_state = replace(ledger.events[1], post_state=None)
    merged = replace(
        ledger.episodes[0],
        play_ids=(first_event.play_id, trailing_without_state.play_id),
        post_state=first_event.post_state,
    )
    bounded = replace(
        ledger,
        events=(first_event, trailing_without_state),
        episodes=(merged,),
    )

    audit = audit_layer_g(bounded)

    assert audit.status == "PASS"
    assert audit.eligible_episode_count == 1


def test_layer_m_proves_trade_and_candle_time_semantics_without_receive_claim() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_m

    trade = _poly(12, "0.60")
    candle = normalize_kalshi_candle_bbo(
        {
            "end_period_ts": int(_at(60).timestamp()),
            "yes_bid": {
                "open": "0.54",
                "high": "0.56",
                "low": "0.53",
                "close": "0.55",
            },
            "yes_ask": {
                "open": "0.57",
                "high": "0.59",
                "low": "0.56",
                "close": "0.58",
            },
            "volume": "10",
            "open_interest": "25",
        },
        ticker="KXNFLGAME-DET",
        outcome="yes",
        logical_market_id="kalshi:det-winner",
    )
    contracts = (
        _contract(),
        _contract(
            venue="kalshi",
            native_id="KXNFLGAME-DET",
            logical_market_id="kalshi:det-winner",
            outcomes=("yes", "no"),
        ),
    )

    audit = audit_layer_m(contracts, (trade, candle), game_id=GAME_ID)

    assert audit.status == "PASS"
    assert audit.trade_count == 1
    assert audit.candle_count == 1
    assert candle.source_interval.end - candle.source_interval.start == timedelta(
        minutes=1
    )
    assert audit.local_receive_time_present is False
    assert audit.real_latency_claim_permitted is False
    assert audit.point_order_within_source_second_permitted is False
    assert audit.candle_point_price_permitted is False


def test_layer_m_rejects_fake_width_and_executable_derived_complement() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_m

    trade = _poly(12, "0.60")
    wrong_width = replace(
        trade,
        source_interval=SourceInterval(_at(12), _at(14), False),
    )
    derived = replace(
        derive_complement(trade, outcome="DAL"),
        bid=Decimal("0.39"),
        ask=Decimal("0.41"),
    )

    first = audit_layer_m((_contract(),), (wrong_width,), game_id=GAME_ID)
    second = audit_layer_m((_contract(),), (trade, derived), game_id=GAME_ID)

    assert first.status == "FAIL"
    assert "trade_interval_not_one_second" in first.errors
    assert second.status == "FAIL"
    assert "derived_complement_marked_executable" in second.errors


def test_layer_m_does_not_double_count_derived_lines_as_actual_trades() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_m

    trade = _poly(12, "0.60")
    derived = derive_complement(trade, outcome="DAL")

    audit = audit_layer_m((_contract(),), (trade, derived), game_id=GAME_ID)

    assert audit.status == "PASS"
    assert audit.trade_count == 1
    assert audit.derived_count == 1


def test_layer_m_rejects_conflicting_economics_for_stable_native_id() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_m

    trade = _poly(12, "0.60", native_id="0xstable")
    conflict = replace(
        trade,
        observation_id="sha256:" + "e" * 64,
        price=Decimal("0.61"),
    )

    audit = audit_layer_m(
        (_contract(),),
        (trade, conflict),
        game_id=GAME_ID,
    )

    assert audit.status == "FAIL"
    assert audit.errors == (
        "observation_deduplication_failed:"
        "conflicting economics for stable observation identity",
    )


def test_layer_m_revalidates_normalized_price_size_and_bbo_bounds() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_m

    trade = _poly(12, "0.60")
    invalid_price = replace(trade, price=Decimal("1.01"))
    invalid_size = replace(trade, size=Decimal(-1))
    fake_trade_bbo = replace(
        trade,
        bid=Decimal("0.59"),
        ask=Decimal("0.61"),
    )

    for observation in (invalid_price, invalid_size, fake_trade_bbo):
        audit = audit_layer_m(
            (_contract(),),
            (observation,),
            game_id=GAME_ID,
        )
        assert audit.status == "FAIL"


def test_layer_m_rejects_outcome_outside_bound_contract() -> None:
    from prediction_market.sports.nfl_x13_association import audit_layer_m

    unknown_outcome = _poly(12, "0.60", outcome="UNKNOWN")

    audit = audit_layer_m(
        (_contract(),),
        (unknown_outcome,),
        game_id=GAME_ID,
    )

    assert audit.status == "FAIL"
    assert "observation_outcome_not_in_contract" in audit.errors


def test_both_layers_must_pass_before_timeline_or_association_exists() -> None:
    from prediction_market.sports.nfl_x13_association import (
        AssociationEvidenceError,
        run_x13_association,
    )

    unknown = replace(
        _poly(12, "0.60"),
        raw_market_id="not-the-contract",
        logical_market_id="not-the-logical-market",
    )

    with pytest.raises(AssociationEvidenceError, match="Layer M"):
        run_x13_association(_ledger(), (_contract(),), (unknown,))


def test_source_timeline_and_layer_a_keep_ambiguity_and_never_forward_fill() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    candle = normalize_kalshi_candle_bbo(
        {
            "end_period_ts": int(_at(60).timestamp()),
            "yes_bid": {
                "open": "0.54",
                "high": "0.56",
                "low": "0.53",
                "close": "0.55",
            },
            "yes_ask": {
                "open": "0.57",
                "high": "0.59",
                "low": "0.56",
                "close": "0.58",
            },
            "volume": "10",
            "open_interest": "25",
        },
        ticker="KXNFLGAME-DET",
        outcome="yes",
        logical_market_id="kalshi:det-winner",
    )
    contracts = (
        _contract(),
        _contract(
            venue="kalshi",
            native_id="KXNFLGAME-DET",
            logical_market_id="kalshi:det-winner",
            outcomes=("yes", "no"),
        ),
    )
    observations = (
        _poly(9, "0.50"),
        _poly(10, "0.55"),
        _poly(12, "0.60"),
        _poly(20, "0.65"),
        _poly(45, "0.58"),
        candle,
    )

    run = run_x13_association(_ledger(), contracts, observations)

    assert run.status == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert len(run.timelines) == 6
    zero = next(item for item in run.timelines if item.delay_scenario_seconds == 0)
    assert any(
        item.identity == observations[1].observation_id and item.order_ambiguous
        for item in zero.items
    )
    assert any(
        item.identity == candle.observation_id
        and item.interval_width_seconds == Decimal(60)
        for item in zero.items
    )

    episode_id = _ledger().episodes[0].episode_id
    rows = [
        item
        for item in run.associations
        if item.episode_id == episode_id
        and item.contract_id == "polymarket:0xcondition"
        and item.outcome == "DET"
        and item.delay_scenario_seconds == 0
    ]
    by_horizon = {item.horizon_seconds: item for item in rows}
    assert by_horizon[1].status == "ORDER_AMBIGUOUS"
    assert _metrics(by_horizon[1])["first_post_event_trade"] == Decimal("0.60")
    assert _metrics(by_horizon[1])["last_actual_price"] == Decimal("0.60")
    assert by_horizon[2].status == "ORDER_AMBIGUOUS"
    assert _metrics(by_horizon[2])["pre_event_actual_trade"] == Decimal("0.50")
    assert _metrics(by_horizon[2])["first_post_event_trade"] == Decimal("0.60")
    assert _metrics(by_horizon[2])["last_actual_price"] == Decimal("0.60")
    assert _metrics(by_horizon[2])["signed_price_change"] == Decimal("0.10")
    assert by_horizon[30].contaminated is False
    assert by_horizon[60].contaminated is True
    assert by_horizon[60].status == "CONTAMINATED"
    assert _metrics(by_horizon[60])["candle_bbo_point_metrics_prohibited"] is True


def test_persistable_interval_builders_preserve_widths_ambiguity_and_no_fill() -> None:
    from prediction_market.sports.nfl_x13_association import (
        build_game_time_intervals_v1,
        build_market_time_intervals_v1,
        build_source_timeline_v1,
        build_timeline_audit_rows_v1,
    )

    candle = normalize_kalshi_candle_bbo(
        {
            "end_period_ts": int(_at(60).timestamp()),
            "yes_bid": {
                "open": "0.54",
                "high": "0.56",
                "low": "0.53",
                "close": "0.55",
            },
            "yes_ask": {
                "open": "0.57",
                "high": "0.59",
                "low": "0.56",
                "close": "0.58",
            },
            "volume": "10",
            "open_interest": "25",
        },
        ticker="KXNFLGAME-DET",
        outcome="yes",
        logical_market_id="kalshi:det-winner",
    )
    observed = _poly(10, "0.55")
    derived = derive_complement(observed, outcome="DAL")
    contracts = (
        _contract(),
        _contract(
            venue="kalshi",
            native_id="KXNFLGAME-DET",
            logical_market_id="kalshi:det-winner",
            outcomes=("yes", "no"),
        ),
    )

    game = build_game_time_intervals_v1(_ledger())
    market = build_market_time_intervals_v1(
        contracts,
        (observed, _poly(10, "0.57", native_id="0xsecond"), derived, candle),
        game_id=GAME_ID,
    )
    timeline = build_source_timeline_v1(game, market)
    audit = build_timeline_audit_rows_v1(timeline)

    first_episode = _ledger().episodes[0].episode_id
    zero = next(
        row
        for row in game.rows
        if row.episode_id == first_episode and row.delay_seconds == 0
    )
    ten = next(
        row
        for row in game.rows
        if row.episode_id == first_episode and row.delay_seconds == 10
    )
    assert (zero.interval_start_utc, zero.interval_end_utc) == (
        _at(10).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        _at(11).isoformat(timespec="microseconds").replace("+00:00", "Z"),
    )
    assert zero.end_inclusive is True
    assert ten.interval_end_utc == _at(21).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    assert game.to_records() == [row.to_dict() for row in game.rows]

    actual = next(
        row
        for row in market.rows
        if row.observation_id == observed.observation_id
    )
    candle_row = next(
        row
        for row in market.rows
        if row.observation_id == candle.observation_id
    )
    derived_row = next(
        row
        for row in market.rows
        if row.observation_id == derived.observation_id
    )
    assert actual.observation_kind == "actual_trade"
    assert actual.interval_end_utc == _at(11).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    assert actual.end_inclusive is False
    assert actual.bbo_available is False
    assert candle_row.observation_kind == "kalshi_1m_candle"
    assert candle_row.interval_start_utc == _at(0).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    assert candle_row.interval_end_utc == _at(60).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    assert candle_row.actual_trade is False
    assert candle_row.bbo_tick is False
    assert derived_row.observation_kind == "derived_complement"
    assert derived_row.actual_observation is False
    assert derived_row.executable is False
    assert market.to_records() == [row.to_dict() for row in market.rows]

    records = timeline.to_records()
    required = {
        "timeline_id",
        "game_id",
        "layer",
        "identity",
        "interval_start_utc",
        "interval_end_utc",
        "delay_seconds",
        "provenance",
        "order_ambiguous",
        "forward_filled",
    }
    assert records
    assert all(required <= record.keys() for record in records)
    assert all(record["forward_filled"] is False for record in records)
    zero_rows = [row for row in timeline.rows if row.delay_seconds == 0]
    same_second = [
        row
        for row in zero_rows
        if row.identity
        in {
            observed.observation_id,
            next(
                item.observation_id
                for item in market.rows
                if item.observation_id not in {
                    observed.observation_id,
                    candle.observation_id,
                    derived.observation_id,
                }
            ),
        }
    ]
    assert len(same_second) == 2
    assert all(row.order_ambiguous for row in same_second)
    assert any(
        row.layer == "G"
        and row.identity == first_episode
        and row.order_ambiguous
        for row in zero_rows
    )
    assert audit.rows
    assert all(row.order_ambiguous for row in audit.rows)
    ambiguous_timeline_rows = [
        row for row in timeline.rows if row.order_ambiguous
    ]
    assert len(audit.rows) == len(ambiguous_timeline_rows)
    audit_by_timeline_id = {row.timeline_id: row for row in audit.rows}
    for row in same_second:
        audited = audit_by_timeline_id[row.timeline_id]
        assert audited.identity == row.identity
        assert "same_source_second" in audited.reason_codes
        assert "overlapping_source_intervals" in audited.reason_codes
        assert audited.overlap_count >= 2
    same_second_event = next(
        row
        for row in zero_rows
        if row.layer == "G" and row.identity == first_episode
    )
    assert "same_source_second" in audit_by_timeline_id[
        same_second_event.timeline_id
    ].reason_codes
    assert all(row.overlap_count > 0 for row in audit.rows)
    assert audit.to_records() == [row.to_dict() for row in audit.rows]
    assert [row.timeline_id for row in timeline.rows] == [
        row.timeline_id
        for row in build_source_timeline_v1(game, market).rows
    ]


def test_timeline_respects_half_open_trade_and_inclusive_game_boundaries() -> None:
    from prediction_market.sports.nfl_x13_association import (
        build_game_time_intervals_v1,
        build_market_time_intervals_v1,
        build_source_timeline_v1,
        build_timeline_audit_rows_v1,
    )

    ledger = build_x13_game_ledger([_play(10, 1, 10)])
    before = _poly(9, "0.50")
    boundary_after = _poly(11, "0.60")
    game = build_game_time_intervals_v1(ledger)
    market = build_market_time_intervals_v1(
        (_contract(),),
        (before, boundary_after),
        game_id=GAME_ID,
    )
    timeline = build_source_timeline_v1(game, market)
    zero = {
        row.identity: row
        for row in timeline.rows
        if row.delay_seconds == 0
    }

    assert zero[before.observation_id].order_ambiguous is False
    assert zero[boundary_after.observation_id].order_ambiguous is True
    episode_id = ledger.episodes[0].episode_id
    assert zero[episode_id].order_ambiguous is True
    audit = build_timeline_audit_rows_v1(timeline)
    audited_ids = {row.timeline_id for row in audit.rows}
    assert zero[before.observation_id].timeline_id not in audited_ids
    assert zero[boundary_after.observation_id].timeline_id in audited_ids
    assert zero[episode_id].timeline_id in audited_ids


def test_horizons_start_after_the_layer_g_upper_bound() -> None:
    from prediction_market.sports.nfl_x13_association import (
        run_x13_association,
    )

    ledger = build_x13_game_ledger([_play(10, 1, 10)])
    run = run_x13_association(
        ledger,
        (_contract(),),
        (
            _poly(9, "0.50"),
            _poly(22, "0.60"),
            _poly(80, "0.70"),
        ),
    )
    episode_id = ledger.episodes[0].episode_id
    rows = {
        row.horizon_seconds: row
        for row in run.associations
        if row.episode_id == episode_id
        and row.outcome == "DET"
        and row.delay_scenario_seconds == 10
    }

    # G=[10s,21s]. A one-second reaction horizon therefore closes at 22s,
    # not at source_time+1s (11s).
    one_second = _metrics(rows[1])
    assert rows[1].status == "OBSERVED"
    assert one_second["first_post_event_trade"] == Decimal("0.60")
    assert one_second["last_actual_price"] == Decimal("0.60")
    assert one_second["staleness_seconds"] == Decimal(0)

    # The 60-second path is also anchored at G.end and includes the 80s trade.
    minute = _metrics(rows[60])
    assert minute["net_60_second_change"] == Decimal("0.20")
    assert minute["last_actual_price"] == Decimal("0.70")


def test_short_horizon_does_not_leak_path_after_next_episode() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    ledger = build_x13_game_ledger(
        [
            _play(10, 1, 10),
            _play(20, 2, 25, play_type="run", pass_attempt=0, rush_attempt=1),
        ]
    )
    run = run_x13_association(
        ledger,
        (_contract(),),
        (
            _poly(9, "0.50"),
            _poly(12, "0.60"),
            _poly(30, "0.35"),
        ),
    )
    first_episode_id = ledger.episodes[0].episode_id
    rows = {
        row.horizon_seconds: row
        for row in run.associations
        if row.episode_id == first_episode_id
        and row.outcome == "DET"
        and row.delay_scenario_seconds == 0
    }

    short = _metrics(rows[1])
    assert rows[1].contaminated is False
    assert short["last_actual_price"] == Decimal("0.60")
    assert short["net_60_second_change"] is None
    assert short["net_60_second_maximum_excursion"] is None
    assert short["overshoot_candidate"] is False
    assert short["reversal_candidate"] is False
    assert short["path_order_ambiguous"] is False

    minute = _metrics(rows[60])
    assert rows[60].contaminated is True
    assert minute["net_60_second_change"] == Decimal("-0.15")
    assert minute["net_60_second_maximum_excursion"] == Decimal("-0.15")
    assert minute["reversal_candidate"] is True


def test_same_second_trades_are_aggregated_but_not_artificially_ordered() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    run = run_x13_association(
        _ledger(),
        (_contract(),),
        (
            _poly(9, "0.50"),
            _poly(12, "0.60", native_id="0xa"),
            _poly(12, "0.62", native_id="0xb"),
        ),
    )
    episode_id = _ledger().episodes[0].episode_id
    row = next(
        item
        for item in run.associations
        if item.episode_id == episode_id
        and item.outcome == "DET"
        and item.delay_scenario_seconds == 0
        and item.horizon_seconds == 2
    )
    metrics = _metrics(row)

    assert row.order_ambiguous is True
    assert row.status == "ORDER_AMBIGUOUS"
    assert metrics["first_post_event_trade"] is None
    assert metrics["last_actual_price"] is None
    assert metrics["trade_count"] == 2
    assert metrics["volume"] == Decimal(4)
    assert metrics["vwap"] == Decimal("0.61")


def test_kalshi_microseconds_floor_to_same_second_order_ambiguity() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    contract = _contract(
        venue="kalshi",
        native_id="KXNFLGAME-DET",
        logical_market_id="kalshi:det-winner",
        outcomes=("yes", "no"),
    )
    early = _kalshi(
        12,
        "0.60",
        native_id="k-early",
        microsecond=1,
    )
    late = _kalshi(
        12,
        "0.62",
        native_id="k-late",
        microsecond=999999,
    )
    run = run_x13_association(
        _ledger(),
        (contract,),
        (
            _kalshi(9, "0.50", native_id="k-pre"),
            early,
            late,
        ),
    )
    episode_id = _ledger().episodes[0].episode_id
    row = next(
        item
        for item in run.associations
        if item.episode_id == episode_id
        and item.outcome == "yes"
        and item.delay_scenario_seconds == 0
        and item.horizon_seconds == 2
    )
    metrics = _metrics(row)

    assert early.native_source_time == _at(12).replace(microsecond=1)
    assert late.native_source_time == _at(12).replace(microsecond=999999)
    assert early.source_interval == late.source_interval
    assert row.order_ambiguous is True
    assert row.status == "ORDER_AMBIGUOUS"
    assert metrics["first_post_event_trade"] is None
    assert metrics["last_actual_price"] is None
    assert metrics["trade_count"] == 2
    assert metrics["vwap"] == Decimal("0.61")


def test_kalshi_block_trade_is_excluded_from_primary_association_path() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    contract = _contract(
        venue="kalshi",
        native_id="KXNFLGAME-DET",
        logical_market_id="kalshi:det-winner",
        outcomes=("yes", "no"),
    )
    block = _kalshi(
        12,
        "0.90",
        native_id="k-block",
        is_block_trade=True,
    )
    run = run_x13_association(
        _ledger(),
        (contract,),
        (
            _kalshi(9, "0.50", native_id="k-pre"),
            block,
            _kalshi(13, "0.60", native_id="k-eligible"),
        ),
    )
    episode_id = _ledger().episodes[0].episode_id
    rows = {
        item.horizon_seconds: item
        for item in run.associations
        if item.episode_id == episode_id
        and item.outcome == "yes"
        and item.delay_scenario_seconds == 0
    }

    assert block.primary_path_eligible is False
    assert run.layer_m_audit.trade_count == 3
    assert rows[1].status == "MISSING_OBSERVATION"
    assert _metrics(rows[1])["trade_count"] == 0
    assert rows[2].status == "OBSERVED"
    assert _metrics(rows[2])["first_post_event_trade"] == Decimal("0.60")
    assert _metrics(rows[2])["last_actual_price"] == Decimal("0.60")
    assert _metrics(rows[2])["signed_price_change"] == Decimal("0.10")


def test_derived_complement_never_becomes_actual_or_executable() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    observed = _poly(12, "0.60")
    run = run_x13_association(
        _ledger(),
        (_contract(),),
        (_poly(9, "0.50"), observed, derive_complement(observed, outcome="DAL")),
    )
    episode_id = _ledger().episodes[0].episode_id
    dal = next(
        item
        for item in run.associations
        if item.episode_id == episode_id
        and item.outcome == "DAL"
        and item.delay_scenario_seconds == 0
        and item.horizon_seconds == 2
    )

    assert dal.status == "MISSING_OBSERVATION"
    assert _metrics(dal)["trade_count"] == 0
    assert _metrics(dal)["derived_observation_count"] == 1
    assert _metrics(dal)["executable_price_claim_permitted"] is False


def test_association_attrition_rows_expose_all_fail_closed_reasons() -> None:
    from prediction_market.sports.nfl_x13_association import (
        build_association_attrition_rows_v1,
        run_x13_association,
    )

    observed = _poly(12, "0.60")
    contracts = (_contract(),)
    run = run_x13_association(
        _ledger(),
        contracts,
        (
            _poly(9, "0.50"),
            observed,
            derive_complement(observed, outcome="DAL"),
        ),
    )
    episode_id = _ledger().episodes[0].episode_id
    selected = [
        row
        for row in run.associations
        if row.episode_id == episode_id
        and row.delay_scenario_seconds == 0
        and row.horizon_seconds == 2
    ]
    attrition = build_association_attrition_rows_v1(
        selected,
        contracts,
        game_id=GAME_ID,
        allowed_families=("moneyline",),
    )
    by_outcome = {row.outcome: row for row in attrition.rows}

    assert by_outcome["DET"].status == "OBSERVED"
    assert by_outcome["DET"].eligible is True
    assert by_outcome["DET"].exclusion_reasons == ()
    assert by_outcome["DAL"].status == "MISSING_OBSERVATION"
    assert by_outcome["DAL"].eligible is False
    assert by_outcome["DAL"].exclusion_reasons == (
        "missing",
        "derived_only",
        "no_actual_trade",
    )

    det = next(row for row in selected if row.outcome == "DET")
    stale_metrics = dict(det.candidate.metrics)
    stale_metrics["pre_trade_age_seconds"] = Decimal(61)
    stale = replace(
        det,
        candidate=replace(
            det.candidate,
            metrics=tuple(sorted(stale_metrics.items())),
        ),
    )
    invalid_orientation = replace(det, outcome="UNKNOWN")
    stale_and_invalid = build_association_attrition_rows_v1(
        (stale, invalid_orientation),
        contracts,
        game_id=GAME_ID,
        allowed_families=("spread",),
    )
    assert stale_and_invalid.rows[0].exclusion_reasons == (
        "stale",
        "wrong_family",
    )
    assert stale_and_invalid.rows[1].exclusion_reasons == (
        "wrong_family",
        "invalid_orientation",
    )

    ambiguous_run = run_x13_association(
        _ledger(),
        contracts,
        (
            _poly(9, "0.50"),
            _poly(12, "0.60", native_id="0xa"),
            _poly(12, "0.62", native_id="0xb"),
        ),
    )
    ambiguous = next(
        row
        for row in ambiguous_run.associations
        if row.episode_id == episode_id
        and row.outcome == "DET"
        and row.delay_scenario_seconds == 0
        and row.horizon_seconds == 2
    )
    contaminated = next(
        row
        for row in run.associations
        if row.episode_id == episode_id
        and row.outcome == "DET"
        and row.delay_scenario_seconds == 0
        and row.horizon_seconds == 60
    )
    flagged = build_association_attrition_rows_v1(
        (ambiguous, contaminated),
        contracts,
        game_id=GAME_ID,
        allowed_families=("moneyline",),
    )
    assert "ambiguous" in flagged.rows[0].exclusion_reasons
    assert "contaminated" in flagged.rows[1].exclusion_reasons
    assert attrition.to_records() == [row.to_dict() for row in attrition.rows]
    assert attrition.to_dict()["reason_counts"] == {
        "derived_only": 1,
        "missing": 1,
        "no_actual_trade": 1,
    }
    assert [row.association_id for row in attrition.rows] == [
        row.association_id
        for row in build_association_attrition_rows_v1(
            selected,
            contracts,
            game_id=GAME_ID,
            allowed_families=("moneyline",),
        ).rows
    ]


def test_two_venue_direction_is_compared_only_after_semantic_orientation() -> None:
    from prediction_market.sports.nfl_x13_association import run_x13_association

    contracts = (
        _contract(subject="DAL at DET", outcomes=("DET", "DAL")),
        _contract(
            venue="kalshi",
            native_id="KXNFLGAME-DET",
            logical_market_id="kalshi:det-winner",
            outcomes=("yes", "no"),
        ),
    )
    run = run_x13_association(
        _ledger(),
        contracts,
        (
            normalize_polymarket_trade(
                {
                    "transactionHash": "0xp-pre",
                    "timestamp": int(_at(9).timestamp()),
                    "price": "0.50",
                    "size": "1",
                },
                    raw_market_id="0xcondition",
                    logical_market_id="polymarket:0xcondition:winner",
                    outcome="DET",
            ),
            normalize_polymarket_trade(
                {
                    "transactionHash": "0xp-post",
                    "timestamp": int(_at(12).timestamp()),
                    "price": "0.55",
                    "size": "1",
                },
                    raw_market_id="0xcondition",
                    logical_market_id="polymarket:0xcondition:winner",
                    outcome="DET",
            ),
            _kalshi(9, "0.48", native_id="k-pre"),
            _kalshi(12, "0.54", native_id="k-post"),
        ),
    )
    episode_id = _ledger().episodes[0].episode_id
    rows = [
        item
        for item in run.associations
        if item.episode_id == episode_id
        and item.outcome in {"DET", "yes"}
        and item.delay_scenario_seconds == 0
        and item.horizon_seconds == 2
    ]

    assert {item.venue for item in rows} == {"polymarket", "kalshi"}
    assert {_metrics(item)["two_venue_direction_consistency"] for item in rows} == {
        "CONSISTENT"
    }


def test_streaming_association_matches_small_materialized_fixture() -> None:
    from prediction_market.sports.nfl_x13_association import (
        iter_x13_associations,
        prepare_x13_association_stream,
        run_x13_association,
    )

    ledger = _ledger()
    contracts = (_contract(),)
    observations = (_poly(9, "0.50"), _poly(12, "0.60"))

    stream = prepare_x13_association_stream(
        ledger,
        contracts,
        observations,
    )
    streamed = tuple(iter_x13_associations(stream))
    materialized = run_x13_association(
        ledger,
        contracts,
        observations,
    )

    assert not hasattr(stream, "associations")
    assert stream.associations_materialized is False
    assert stream.expected_association_count == 144
    assert len(streamed) == stream.expected_association_count
    assert streamed == materialized.associations


def test_streaming_association_bisects_dense_observation_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_association as association_module

    observations = tuple(
        _poly(
            second,
            "0.50",
            native_id=f"dense-{second}",
        )
        for second in range(-1000, 1001)
    )
    stream = association_module.prepare_x13_association_stream(
        _ledger(),
        (_contract(),),
        observations,
    )
    strict_relation_calls = 0
    original = association_module._strict_relation

    def counted_relation(event_interval, observation):
        nonlocal strict_relation_calls
        strict_relation_calls += 1
        return original(event_interval, observation)

    monkeypatch.setattr(
        association_module,
        "_strict_relation",
        counted_relation,
    )

    rows = tuple(association_module.iter_x13_associations(stream))

    assert len(rows) == stream.expected_association_count == 144
    # Only the narrow possible-overlap slices are rechecked.  A full scan for
    # every association row would invoke this hundreds of thousands of times.
    assert strict_relation_calls < 500


def test_streaming_association_keeps_every_missing_cardinality_row() -> None:
    from prediction_market.sports.nfl_x13_association import (
        iter_x13_associations,
        prepare_x13_association_stream,
    )

    stream = prepare_x13_association_stream(
        _ledger(),
        (_contract(),),
        (),
    )

    rows = tuple(iter_x13_associations(stream))

    assert len(rows) == stream.expected_association_count == 144
    assert {row.status for row in rows} <= {
        "MISSING_OBSERVATION",
        "CONTAMINATED",
    }
    assert {
        (
            row.episode_id,
            row.contract_id,
            row.outcome,
            row.delay_scenario_seconds,
            row.horizon_seconds,
        )
        for row in rows
    } == {
        (
            episode.episode_id,
            "polymarket:0xcondition",
            outcome,
            delay,
            horizon,
        )
        for episode in _ledger().episodes
        for outcome in ("DET", "DAL")
        for delay in (0, 1, 2, 3, 5, 10)
        for horizon in (1, 2, 5, 10, 30, 60)
    }


def _exploration_rows(
    *,
    count: int,
    games: int,
    groups: tuple[str, ...] = ("pooled",),
    venues: tuple[str, ...] = ("polymarket", "kalshi"),
):
    from prediction_market.sports.nfl_x13_association import (
        ExplorationObservationV1,
    )

    rows = []
    for index in range(count):
        rows.append(
            ExplorationObservationV1(
                observation_id=f"unit-{index}",
                hypothesis_id="h1",
                group=groups[index % len(groups)],
                game_id=f"game-{(index // len(groups)) % games}",
                episode_id=f"episode-{index}",
                venue=venues[index % len(venues)],
                effect=Decimal(index % 7) / Decimal(100),
                eligible=True,
            )
        )
    return tuple(rows)


def test_primary_pooled_gate_and_inference_inputs_use_games_as_clusters() -> None:
    from prediction_market.sports.nfl_x13_association import (
        evaluate_exploration_gate,
    )

    report = evaluate_exploration_gate(
        _exploration_rows(count=300, games=16),
        analysis_kind="primary_pooled",
        bootstrap_seed=20260722,
    )

    assert report.status == "PRELIMINARY_ELIGIBLE"
    assert report.eligible_episode_count == 300
    assert report.game_count == 16
    assert dict(report.venue_episode_counts) == {
        "kalshi": 150,
        "polymarket": 150,
    }
    assert report.inference.game_cluster_bootstrap is True
    assert len(report.inference.cluster_game_ids) == 16
    assert len(report.inference.observation_game_clusters) == 300
    assert report.inference.observation_game_clusters[0][0].startswith("unit-")
    assert report.inference.leave_one_game_out is True
    assert report.inference.logo_holdout_game_ids == report.inference.cluster_game_ids
    assert report.inference.multiple_testing_method == "benjamini_hochberg"
    assert report.inference.bootstrap_seed == 20260722
    assert "tradeable_alpha" in report.prohibited_claims


@pytest.mark.parametrize(
    ("kind", "count", "games", "groups"),
    (
        ("binary_moderator", 80, 8, ("low", "high")),
        ("named_subtype", 20, 6, ("touchdown",)),
        ("sequence", 15, 6, ("turnover_to_score",)),
    ),
)
def test_registered_secondary_report_gates(
    kind: str,
    count: int,
    games: int,
    groups: tuple[str, ...],
) -> None:
    from prediction_market.sports.nfl_x13_association import (
        evaluate_exploration_gate,
    )

    report = evaluate_exploration_gate(
        _exploration_rows(count=count, games=games, groups=groups),
        analysis_kind=kind,
    )

    assert report.status == "PRELIMINARY_ELIGIBLE"


def test_low_support_and_single_game_concentration_are_downgraded() -> None:
    from prediction_market.sports.nfl_x13_association import (
        evaluate_exploration_gate,
    )

    descriptive = evaluate_exploration_gate(
        _exploration_rows(count=6, games=3),
        analysis_kind="named_subtype",
    )
    case_only = evaluate_exploration_gate(
        _exploration_rows(count=4, games=2),
        analysis_kind="named_subtype",
    )
    concentrated_rows = list(_exploration_rows(count=300, games=16))
    for index in range(100):
        concentrated_rows[index] = replace(
            concentrated_rows[index],
            game_id="dominant-game",
        )
    concentrated = evaluate_exploration_gate(
        tuple(concentrated_rows),
        analysis_kind="primary_pooled",
    )

    assert descriptive.status == "DESCRIPTIVE_ONLY"
    assert case_only.status == "CASE_ONLY"
    assert concentrated.status == "DESCRIPTIVE_ONLY"
    assert "single_game_contribution_exceeds_25_percent" in concentrated.reason_codes


def test_single_game_concentration_is_checked_inside_each_registered_group() -> None:
    from prediction_market.sports.nfl_x13_association import (
        evaluate_exploration_gate,
    )

    rows = list(
        _exploration_rows(
            count=80,
            games=8,
            groups=("low", "high"),
        )
    )
    low_indexes = [index for index, row in enumerate(rows) if row.group == "low"]
    for index in low_indexes[:11]:
        rows[index] = replace(rows[index], game_id="dominant-low-game")

    report = evaluate_exploration_gate(
        tuple(rows),
        analysis_kind="binary_moderator",
    )

    assert report.status == "DESCRIPTIVE_ONLY"
    assert "single_game_contribution_exceeds_25_percent" in report.reason_codes


def test_two_venues_for_one_episode_are_not_counted_as_two_sports_events() -> None:
    from prediction_market.sports.nfl_x13_association import (
        ExplorationObservationV1,
        evaluate_exploration_gate,
    )

    duplicate_venues = (
        ExplorationObservationV1(
            observation_id="poly",
            hypothesis_id="h1",
            group="touchdown",
            game_id="game-1",
            episode_id="episode-1",
            venue="polymarket",
            effect=Decimal("0.1"),
            eligible=True,
        ),
        ExplorationObservationV1(
            observation_id="kalshi",
            hypothesis_id="h1",
            group="touchdown",
            game_id="game-1",
            episode_id="episode-1",
            venue="kalshi",
            effect=Decimal("0.1"),
            eligible=True,
        ),
    )

    report = evaluate_exploration_gate(
        duplicate_venues,
        analysis_kind="named_subtype",
    )

    assert report.eligible_episode_count == 1
    assert report.status == "CASE_ONLY"


def _registered_binary_rows(
    *,
    hypotheses: tuple[str, ...] = ("fair_delta_by_semantic_salience",),
) -> tuple:
    from prediction_market.sports.nfl_x13_association import (
        ExplorationObservationV1,
    )

    rows = []
    for hypothesis_index, hypothesis_id in enumerate(hypotheses):
        for index in range(80):
            group = "low" if index % 2 == 0 else "high"
            base_effect = Decimal("0.01") if group == "low" else Decimal("0.03")
            effect = base_effect + Decimal(hypothesis_index) / Decimal(100)
            game_id = f"game-{(index // 2) % 8}"
            episode_id = f"episode-{index}"
            for venue_index, venue in enumerate(("polymarket", "kalshi")):
                rows.append(
                    ExplorationObservationV1(
                        observation_id=(f"{hypothesis_id}-{episode_id}-{venue}"),
                        hypothesis_id=hypothesis_id,
                        group=group,
                        game_id=game_id,
                        episode_id=episode_id,
                        venue=venue,
                        effect=effect + Decimal(venue_index) / Decimal(1000),
                        eligible=True,
                    )
                )
    return tuple(rows)


def test_registered_analysis_lock_is_frozen_and_has_exact_whitelist() -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
    )

    lock = X13_REGISTERED_ANALYSIS_LOCK_V1
    hypotheses = {item.hypothesis_id: item for item in lock.hypotheses}

    assert lock.experiment_id == "X-13"
    assert lock.status == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert lock.bootstrap_seed == 20260722
    assert lock.lock_id.startswith("sha256:")
    assert hypotheses["fair_delta_by_semantic_salience"].approved_groups == (
        "low",
        "high",
    )
    assert hypotheses["event_superclass"].approved_groups == (
        "score",
        "turnover",
        "routine",
    )
    assert hypotheses["sequence"].approved_groups == (
        "turnover_to_score",
        "answer_score",
        "back_to_back_turnovers",
        "red_zone_empty",
        "two_minute_drive",
        "overtime",
    )
    with pytest.raises(FrozenInstanceError):
        lock.bootstrap_seed = 1  # type: ignore[misc]


def test_registered_inference_rejects_unknown_hypothesis_group_and_lock() -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        AssociationEvidenceError,
        evaluate_exploration_gate,
        run_registered_inference,
    )

    rows = _registered_binary_rows()
    gate = evaluate_exploration_gate(
        rows,
        analysis_kind="binary_moderator",
    )
    unknown_hypothesis = tuple(
        replace(row, hypothesis_id="post_hoc_search") for row in rows
    )
    unknown_group = tuple(replace(row, group="result_optimized_bucket") for row in rows)

    with pytest.raises(AssociationEvidenceError, match="hypothesis"):
        run_registered_inference(
            unknown_hypothesis,
            gate,
            X13_REGISTERED_ANALYSIS_LOCK_V1,
            bootstrap_samples=100,
            seed=20260722,
        )
    with pytest.raises(AssociationEvidenceError, match="group"):
        run_registered_inference(
            unknown_group,
            gate,
            X13_REGISTERED_ANALYSIS_LOCK_V1,
            bootstrap_samples=100,
            seed=20260722,
        )
    with pytest.raises(AssociationEvidenceError, match="trusted"):
        run_registered_inference(
            rows,
            gate,
            replace(
                X13_REGISTERED_ANALYSIS_LOCK_V1,
                version="post-hoc",
            ),
            bootstrap_samples=100,
            seed=20260722,
        )
    with pytest.raises(AssociationEvidenceError, match="seed"):
        run_registered_inference(
            rows,
            gate,
            X13_REGISTERED_ANALYSIS_LOCK_V1,
            bootstrap_samples=100,
            seed=7,
        )


def test_registered_inference_aggregates_venues_then_bootstraps_games() -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        evaluate_exploration_gate,
        run_registered_inference,
    )

    rows = _registered_binary_rows()
    gate = evaluate_exploration_gate(
        rows,
        analysis_kind="binary_moderator",
    )

    first = run_registered_inference(
        rows,
        gate,
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        bootstrap_samples=200,
        seed=20260722,
    )
    second = run_registered_inference(
        rows,
        gate,
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        bootstrap_samples=200,
        seed=20260722,
    )

    assert first == second
    assert len(first.aggregated_episode_effects) == 80
    assert all(unit.venue_count == 2 for unit in first.aggregated_episode_effects)
    estimate = first.estimates[0]
    assert estimate.estimand == "difference:high-minus-low"
    assert estimate.point_estimate == Decimal("0.02")
    assert estimate.ci_lower == Decimal("0.02")
    assert estimate.ci_upper == Decimal("0.02")
    assert estimate.bootstrap_requested_samples == 200
    assert estimate.bootstrap_valid_samples == 200
    assert estimate.empirical_two_sided_p == Decimal(2) / Decimal(201)
    assert len(estimate.logo_effects) == 8
    assert estimate.logo_sign_stability == Decimal(1)
    assert estimate.cluster_multiplicity_preserved is True
    assert estimate.support_status == "PRELIMINARY_ELIGIBLE"
    assert estimate.preliminary_inference_claim_permitted is True
    assert first.venue_aggregation_semantics == (
        "arithmetic_mean_within_(game_id,episode_id,hypothesis_id,group)"
    )
    assert "tradeable_alpha" in first.prohibited_claims


def test_cluster_bootstrap_repeats_whole_game_by_draw_multiplicity() -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        ExplorationObservationV1,
        evaluate_exploration_gate,
        run_registered_inference,
    )

    rows = (
        ExplorationObservationV1(
            observation_id="game-a-0",
            hypothesis_id="signed_diagnostic_fair_delta_pooled",
            group="pooled",
            game_id="game-a",
            episode_id="episode-a",
            venue="polymarket",
            effect=Decimal(0),
            eligible=True,
        ),
        *tuple(
            ExplorationObservationV1(
                observation_id=f"game-b-{index}",
                hypothesis_id="signed_diagnostic_fair_delta_pooled",
                group="pooled",
                game_id="game-b",
                episode_id=f"episode-b-{index}",
                venue="polymarket",
                effect=Decimal(1),
                eligible=True,
            )
            for index in range(3)
        ),
    )
    gate = evaluate_exploration_gate(
        rows,
        analysis_kind="primary_pooled",
    )

    report = run_registered_inference(
        rows,
        gate,
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        bootstrap_samples=1,
        seed=20260722,
    )
    estimate = report.estimates[0]

    assert estimate.point_estimate == Decimal("0.75")
    # The frozen first replicate draws game-a twice. Repeating its complete
    # one-row cluster gives zero; unit-level resampling would not prove this.
    assert estimate.ci_lower == Decimal(0)
    assert estimate.ci_upper == Decimal(0)
    assert estimate.bootstrap_valid_samples == 1


def test_registered_inference_bh_is_within_registered_hypothesis_family() -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        evaluate_exploration_gate,
        run_registered_inference,
    )

    rows = _registered_binary_rows(
        hypotheses=(
            "fair_delta_by_semantic_salience",
            "fair_delta_by_pre_event_liquidity",
        )
    )
    gate = evaluate_exploration_gate(
        rows,
        analysis_kind="binary_moderator",
    )

    report = run_registered_inference(
        rows,
        gate,
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        bootstrap_samples=100,
        seed=20260722,
    )

    assert len(report.estimates) == 2
    assert {estimate.hypothesis_family for estimate in report.estimates} == {
        "fair_delta_moderators"
    }
    assert all(
        estimate.bh_q_value is not None
        and estimate.bh_q_value >= estimate.empirical_two_sided_p
        for estimate in report.estimates
    )
    assert report.multiple_testing_method == (
        "benjamini_hochberg_within_registered_hypothesis_family"
    )


@pytest.mark.parametrize(
    ("count", "games", "expected_status"),
    (
        (6, 3, "DESCRIPTIVE_ONLY"),
        (4, 2, "CASE_ONLY"),
    ),
)
def test_registered_inference_never_promotes_low_support(
    count: int,
    games: int,
    expected_status: str,
) -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        ExplorationObservationV1,
        evaluate_exploration_gate,
        run_registered_inference,
    )

    rows = tuple(
        ExplorationObservationV1(
            observation_id=f"score-{index}",
            hypothesis_id="event_superclass",
            group="score",
            game_id=f"game-{index % games}",
            episode_id=f"episode-{index}",
            venue="polymarket",
            effect=Decimal("0.01"),
            eligible=True,
        )
        for index in range(count)
    )
    gate = evaluate_exploration_gate(
        rows,
        analysis_kind="named_subtype",
    )

    report = run_registered_inference(
        rows,
        gate,
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        bootstrap_samples=50,
        seed=20260722,
    )
    estimate = report.estimates[0]

    assert gate.status == expected_status
    assert estimate.support_status == expected_status
    assert estimate.preliminary_inference_claim_permitted is False
    assert estimate.promotion_claim_permitted is False


def test_registered_inference_rejects_duplicate_venue_unit_and_gate_mismatch() -> None:
    from prediction_market.sports.nfl_x13_association import (
        X13_REGISTERED_ANALYSIS_LOCK_V1,
        AssociationEvidenceError,
        evaluate_exploration_gate,
        run_registered_inference,
    )

    rows = _registered_binary_rows()
    gate = evaluate_exploration_gate(
        rows,
        analysis_kind="binary_moderator",
    )
    duplicate = replace(rows[0], observation_id="duplicate-same-venue")
    wrong_gate = evaluate_exploration_gate(
        rows[:20],
        analysis_kind="binary_moderator",
    )

    with pytest.raises(AssociationEvidenceError, match="venue"):
        run_registered_inference(
            (*rows, duplicate),
            gate,
            X13_REGISTERED_ANALYSIS_LOCK_V1,
            bootstrap_samples=50,
            seed=20260722,
        )
    with pytest.raises(AssociationEvidenceError, match="gate"):
        run_registered_inference(
            rows,
            wrong_gate,
            X13_REGISTERED_ANALYSIS_LOCK_V1,
            bootstrap_samples=50,
            seed=20260722,
        )
