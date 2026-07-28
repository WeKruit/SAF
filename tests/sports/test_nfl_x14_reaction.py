from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal

import pytest

from prediction_market.contracts import CaptureTimingV0
from prediction_market.sports.nfl_x14_reaction import (
    BookLevel,
    CaptureContinuityGate,
    InformationEventV1,
    MarketUpdateV1,
    ProspectiveGameBindingV1,
    RawCaptureEnvelopeV1,
    ReactionCensorReason,
    SportsFeedAdapter,
    build_shadow_taker_observation,
    measure_reaction,
)


NS = 1_000_000_000


def _timing(
    *,
    ordinal: int,
    payload: bytes,
    receive_ns: int,
    ready_ns: int,
    epoch: int = 0,
) -> CaptureTimingV0:
    return CaptureTimingV0(
        timing_version="v0",
        capture_session_id="session-2026-bal-buf",
        record_ordinal=ordinal,
        payload_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        receive_boundary="application_callback",
        local_receive_utc_ns=1_800_000_000_000_000_000 + receive_ns,
        local_receive_monotonic_ns=receive_ns,
        pairing_uncertainty_ns=100,
        host_id="capture-host-a",
        boot_id="boot-a",
        monotonic_clock_id="CLOCK_MONOTONIC_RAW",
        clock_epoch_id="clock-epoch-a",
        connection_epoch=epoch,
        clock_sync_sample_id="sync-a",
        raw_append_done_monotonic_ns=receive_ns + 10,
        parse_done_monotonic_ns=receive_ns + 20,
        normalize_done_monotonic_ns=receive_ns + 30,
        state_or_book_ready_monotonic_ns=ready_ns,
    )


def _binding() -> ProspectiveGameBindingV1:
    return ProspectiveGameBindingV1.from_registry_record(
        {
            "binding_id": "NFL-2026-01-BAL-BUF",
            "game_id": "2026_01_BAL_BUF",
            "capture_session_id": "session-2026-bal-buf",
            "scheduled_start_utc": "2026-09-10T17:00:00Z",
            "away_team": "BAL",
            "home_team": "BUF",
            "sports_provider": "licensed-feed",
            "sports_provider_game_id": "provider-8801",
            "polymarket": {
                "event_id": "pm-event-1",
                "event_slug": "nfl-bal-buf-2026-09-10",
                "native_game_id": "8801",
                "condition_ids": ["condition-1"],
                "token_outcomes": {
                    "token-bal": "BAL",
                    "token-buf": "BUF",
                },
            },
            "kalshi": {
                "event_id": "KXNFLGAME-26SEP10BALBUF",
                "ticker_outcomes": {
                    "KXNFLGAME-26SEP10BALBUF-BAL": "BAL",
                    "KXNFLGAME-26SEP10BALBUF-BUF": "BUF",
                },
            },
            "capture_start": "market_creation",
            "capture_end": "market_resolution",
            "venue_rule_snapshot_refs": [
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
            ],
            "license_refs": ["O-POLY", "O-KALSHI", "O-SPORTS"],
            "s3_namespace": "prediction-market/research/nfl/x14",
        }
    )


def _market_update(
    *,
    ordinal: int,
    receive_ns: int,
    bid: str,
    ask: str,
    kind: str = "bbo",
    epoch: int = 0,
    venue: str = "polymarket",
    sequence: int | None = None,
    continuity_status: str = "CONTIGUOUS",
    tick_size: str = ".01",
) -> MarketUpdateV1:
    payload = f"{venue}-{ordinal}".encode()
    timing = _timing(
        ordinal=ordinal,
        payload=payload,
        receive_ns=receive_ns,
        ready_ns=receive_ns + 100,
        epoch=epoch,
    )
    return MarketUpdateV1(
        update_id=f"update-{venue}-{ordinal}",
        raw_envelope_id=f"raw-{venue}-{ordinal}",
        game_id="2026_01_BAL_BUF",
        venue=venue,
        logical_market_id="winner",
        outcome="BUF",
        token_or_ticker="token-buf" if venue == "polymarket" else "ticker-buf",
        update_kind=kind,
        timing=timing,
        exchange_time_utc_ns=timing.local_receive_utc_ns - 1_000,
        source_sequence=sequence,
        book_hash=f"hash-{ordinal}" if venue == "polymarket" else None,
        bids=(BookLevel(price=Decimal(bid), size=Decimal("100")),),
        asks=(BookLevel(price=Decimal(ask), size=Decimal("100")),),
        observed=True,
        derived=False,
        continuity_status=continuity_status,
        price_representation=(
            "TOKEN_OUTCOME"
            if venue == "polymarket"
            else "YES_LEG"
        ),
        tick_size=Decimal(tick_size),
    )


def _event(*, ready_ns: int = 10 * NS) -> InformationEventV1:
    payload = b'{"event":"touchdown"}'
    timing = _timing(
        ordinal=1,
        payload=payload,
        receive_ns=ready_ns - 100,
        ready_ns=ready_ns,
    )
    return InformationEventV1(
        event_id="episode-101",
        game_id="2026_01_BAL_BUF",
        provider="licensed-feed",
        provider_event_id="play-101",
        revision=2,
        revision_status="FINALIZED",
        timing=timing,
        provider_publish_time_utc_ns=timing.local_receive_utc_ns - 1_000,
        finalized_monotonic_ns=ready_ns + 50,
        pre_state={"home_score": 20, "away_score": 17},
        post_state={"home_score": 27, "away_score": 17},
        episode_kind="touchdown",
        salient=True,
        review=False,
        penalty=False,
        no_play=False,
    )


def test_binding_is_registry_driven_and_has_no_car_ari_default() -> None:
    binding = _binding()

    assert binding.game_id == "2026_01_BAL_BUF"
    assert binding.home_team == "BUF"
    assert binding.polymarket_token_outcomes["token-bal"] == "BAL"
    assert binding.kalshi_ticker_outcomes[
        "KXNFLGAME-26SEP10BALBUF-BUF"
    ] == "BUF"
    assert binding.capture_window == (
        "market_creation",
        "market_resolution",
    )

    with pytest.raises(ValueError, match="complete binary outcomes"):
        ProspectiveGameBindingV1.from_registry_record(
            {
                **{
                    key: value
                    for key, value in {
                        "binding_id": "bad",
                        "game_id": "bad-game",
                        "capture_session_id": "bad-session",
                        "scheduled_start_utc": "2026-09-10T17:00:00Z",
                        "away_team": "BAL",
                        "home_team": "BUF",
                        "sports_provider": "feed",
                        "sports_provider_game_id": "game",
                        "capture_start": "market_creation",
                        "capture_end": "market_resolution",
                        "venue_rule_snapshot_refs": ["sha256:" + "1" * 64],
                        "license_refs": ["O-1"],
                        "s3_namespace": "x14",
                    }.items()
                },
                "polymarket": {
                    "event_id": "pm",
                    "event_slug": "nfl-bal-buf-2026-09-10",
                    "native_game_id": "8801",
                    "condition_ids": ["condition"],
                    "token_outcomes": {"one-token": "BAL"},
                },
                "kalshi": None,
            }
        )


def test_market_update_requires_book_ready_at_construction() -> None:
    update = _market_update(
        ordinal=9,
        receive_ns=9 * NS,
        bid=".45",
        ask=".47",
    )
    timing_without_book_ready = update.timing.model_copy(
        update={"state_or_book_ready_monotonic_ns": None},
    )

    with pytest.raises(ValueError, match="book-ready boundary"):
        replace(update, timing=timing_without_book_ready)


def test_raw_envelope_reuses_capture_timing_and_exact_payload_hash() -> None:
    payload = b'{"native":true}'
    timing = _timing(
        ordinal=7,
        payload=payload,
        receive_ns=10 * NS,
        ready_ns=10 * NS + 500,
        epoch=4,
    )
    envelope = RawCaptureEnvelopeV1(
        envelope_id="raw-7",
        binding_id=_binding().binding_id,
        game_id="2026_01_BAL_BUF",
        source="polymarket",
        stream="market",
        message_kind="book",
        timing=timing,
        source_time_utc_ns=timing.local_receive_utc_ns - 50_000,
        source_time_precision_ns=1_000_000,
        source_clock_domain="polymarket_exchange",
        source_sequence=None,
        book_hash="book-hash-7",
        market_id="winner",
        token_or_ticker="token-buf",
        provider_event_id=None,
        revision=None,
        segment_sha256="sha256:" + "3" * 64,
        raw_payload=payload,
    )

    assert envelope.local_receive_utc_ns == timing.local_receive_utc_ns
    assert envelope.local_receive_monotonic_ns == 10 * NS
    assert envelope.connection_epoch == 4
    assert envelope.record_ordinal == 7

    with pytest.raises(ValueError, match="payload SHA"):
        replace(envelope, raw_payload=b"tampered")


def test_sports_feed_adapter_protocol_outputs_information_event() -> None:
    class FixtureAdapter:
        provider = "licensed-feed"

        def parse(
            self, envelope: RawCaptureEnvelopeV1
        ) -> InformationEventV1:
            assert envelope.source == self.provider
            return _event()

    adapter: SportsFeedAdapter = FixtureAdapter()
    assert adapter.parse.__call__
    assert adapter.provider == "licensed-feed"


def test_polymarket_snapshot_gate_is_per_epoch_and_token() -> None:
    gate = CaptureContinuityGate()
    token_a = _market_update(
        ordinal=1,
        receive_ns=1 * NS,
        bid=".40",
        ask=".42",
        kind="snapshot",
    )
    token_b_delta = replace(
        _market_update(
            ordinal=2,
            receive_ns=2 * NS,
            bid=".41",
            ask=".43",
            kind="delta",
        ),
        token_or_ticker="token-bal",
    )

    gate.observe(token_a)
    with pytest.raises(ValueError, match="snapshot.*token"):
        gate.observe(token_b_delta)

    gate.start_epoch(venue="polymarket", epoch=1)
    with pytest.raises(ValueError, match="snapshot.*token"):
        gate.observe(replace(token_a, update_kind="delta", timing=_timing(
            ordinal=3,
            payload=b"polymarket-3",
            receive_ns=3 * NS,
            ready_ns=3 * NS + 100,
            epoch=1,
        )))


def test_kalshi_gap_blocks_deltas_until_new_snapshot() -> None:
    gate = CaptureContinuityGate()
    snapshot = _market_update(
        ordinal=1,
        receive_ns=1 * NS,
        bid=".40",
        ask=".42",
        kind="snapshot",
        venue="kalshi",
        sequence=10,
    )
    gate.observe(snapshot)
    gate.observe(replace(
        _market_update(
            ordinal=2,
            receive_ns=2 * NS,
            bid=".41",
            ask=".43",
            kind="delta",
            venue="kalshi",
            sequence=11,
        ),
    ))

    with pytest.raises(ValueError, match="sequence gap"):
        gate.observe(_market_update(
            ordinal=3,
            receive_ns=3 * NS,
            bid=".42",
            ask=".44",
            kind="delta",
            venue="kalshi",
            sequence=13,
        ))
    with pytest.raises(ValueError, match="blocked"):
        gate.observe(_market_update(
            ordinal=4,
            receive_ns=4 * NS,
            bid=".42",
            ask=".44",
            kind="delta",
            venue="kalshi",
            sequence=14,
        ))

    gate.start_epoch(venue="kalshi", epoch=1)
    new_snapshot = _market_update(
        ordinal=5,
        receive_ns=5 * NS,
        bid=".42",
        ask=".44",
        kind="snapshot",
        venue="kalshi",
        sequence=20,
        epoch=1,
    )
    gate.observe(new_snapshot)
    assert new_snapshot.price_representation == "YES_LEG"


def test_reaction_measurement_finds_start_plateau_t50_t90_and_settled_t90() -> None:
    event = _event()
    updates = (
        _market_update(
            ordinal=1, receive_ns=9 * NS, bid=".39", ask=".41"
        ),
        _market_update(
            ordinal=2,
            receive_ns=10 * NS + 200_000_000,
            bid=".40",
            ask=".42",
        ),
        _market_update(
            ordinal=3, receive_ns=11 * NS, bid=".45", ask=".47"
        ),
        _market_update(
            ordinal=4, receive_ns=12 * NS, bid=".47", ask=".49"
        ),
        _market_update(
            ordinal=5, receive_ns=13 * NS, bid=".49", ask=".51"
        ),
        _market_update(
            ordinal=6, receive_ns=55 * NS, bid=".49", ask=".51"
        ),
    )

    result = measure_reaction(
        event=event,
        updates=updates,
        venue_tick=Decimal(".01"),
        control_noise_95=Decimal(".02"),
        next_salient_event_monotonic_ns=None,
        reference_delta=Decimal(".20"),
    )

    assert result.t_seen_monotonic_ns == 10 * NS - 100
    assert result.t_ready_monotonic_ns == 10 * NS
    assert result.t_final_monotonic_ns == 10 * NS + 50
    assert result.first_any_market_update_monotonic_ns == (
        10 * NS + 200_000_100
    )
    assert result.first_bid_move_monotonic_ns == 10 * NS + 200_000_100
    assert result.first_ask_move_monotonic_ns == 10 * NS + 200_000_100
    assert result.first_material_move_monotonic_ns == 11 * NS + 100
    assert result.analysis_end_monotonic_ns == 70 * NS
    assert result.terminal_midpoint == Decimal(".50")
    assert result.first_passage_t50_ns == NS + 100
    assert result.first_passage_t90_ns == 3 * NS + 100
    assert result.settled_t90_ns == 3 * NS + 100
    assert result.bid_first_passage_t50_ns == NS + 100
    assert result.bid_first_passage_t90_ns == 3 * NS + 100
    assert result.ask_first_passage_t50_ns == NS + 100
    assert result.ask_first_passage_t90_ns == 3 * NS + 100
    assert result.order_flow_imbalance == Decimal("800")
    assert result.reference_delta == Decimal(".20")
    assert result.reference_completion_fraction == Decimal(".5")
    assert result.reference_first_passage_t50_ns == 3 * NS + 100
    assert result.reference_first_passage_t90_ns is None
    assert result.censor_reason is None
    assert result.midpoint_is_executable is False


@pytest.mark.parametrize(
    ("continuity_status", "expected"),
    [
        ("GAP", ReactionCensorReason.CONTINUITY_GAP),
        ("SUSPENDED", ReactionCensorReason.SUSPENSION),
        ("AMBIGUOUS", ReactionCensorReason.ORDER_AMBIGUOUS),
    ],
)
def test_gap_suspension_and_ambiguity_censor_completion(
    continuity_status: str,
    expected: ReactionCensorReason,
) -> None:
    updates = (
        _market_update(
            ordinal=1, receive_ns=9 * NS, bid=".39", ask=".41"
        ),
        _market_update(
            ordinal=2,
            receive_ns=11 * NS,
            bid=".49",
            ask=".51",
            continuity_status=continuity_status,
        ),
    )

    result = measure_reaction(
        event=_event(),
        updates=updates,
        venue_tick=Decimal(".01"),
        control_noise_95=Decimal(".01"),
        next_salient_event_monotonic_ns=None,
    )

    assert result.censor_reason is expected
    assert result.first_passage_t50_ns is None
    assert result.first_passage_t90_ns is None
    assert result.settled_t90_ns is None


def test_next_salient_event_censors_the_window_as_contaminated() -> None:
    result = measure_reaction(
        event=_event(),
        updates=(
            _market_update(
                ordinal=1, receive_ns=9 * NS, bid=".39", ask=".41"
            ),
            _market_update(
                ordinal=2, receive_ns=11 * NS, bid=".49", ask=".51"
            ),
        ),
        venue_tick=Decimal(".01"),
        control_noise_95=Decimal(".01"),
        next_salient_event_monotonic_ns=15 * NS,
    )

    assert result.analysis_end_monotonic_ns == 15 * NS
    assert result.censor_reason is ReactionCensorReason.CONTAMINATION


def test_same_local_timestamp_with_conflicting_prices_is_ambiguous() -> None:
    first = _market_update(
        ordinal=2, receive_ns=11 * NS, bid=".45", ask=".47"
    )
    second = _market_update(
        ordinal=3, receive_ns=11 * NS, bid=".49", ask=".51"
    )

    result = measure_reaction(
        event=_event(),
        updates=(
            _market_update(
                ordinal=1, receive_ns=9 * NS, bid=".39", ask=".41"
            ),
            first,
            second,
        ),
        venue_tick=Decimal(".01"),
        control_noise_95=Decimal(".01"),
        next_salient_event_monotonic_ns=None,
    )

    assert result.censor_reason is ReactionCensorReason.ORDER_AMBIGUOUS
    assert result.first_passage_t50_ns is None


def test_formal_reaction_requires_same_capture_clock_and_observed_book() -> None:
    baseline = _market_update(
        ordinal=1, receive_ns=9 * NS, bid=".39", ask=".41"
    )
    wrong_host = replace(
        _market_update(
            ordinal=2, receive_ns=11 * NS, bid=".49", ask=".51"
        ),
        timing=_market_update(
            ordinal=2, receive_ns=11 * NS, bid=".49", ask=".51"
        ).timing.model_copy(update={"host_id": "capture-host-b"}),
    )

    with pytest.raises(ValueError, match="host/boot/clock"):
        measure_reaction(
            event=_event(),
            updates=(baseline, wrong_host),
            venue_tick=Decimal(".01"),
            control_noise_95=Decimal(".01"),
            next_salient_event_monotonic_ns=None,
        )

    with pytest.raises(ValueError, match="derived"):
        measure_reaction(
            event=_event(),
            updates=(baseline, replace(
                wrong_host,
                timing=baseline.timing.model_copy(
                    update={
                        "record_ordinal": 2,
                        "local_receive_monotonic_ns": 11 * NS,
                        "raw_append_done_monotonic_ns": 11 * NS + 10,
                        "parse_done_monotonic_ns": 11 * NS + 20,
                        "normalize_done_monotonic_ns": 11 * NS + 30,
                        "state_or_book_ready_monotonic_ns": 11 * NS + 100,
                    }
                ),
                observed=False,
                derived=True,
            )),
            venue_tick=Decimal(".01"),
            control_noise_95=Decimal(".01"),
            next_salient_event_monotonic_ns=None,
        )


def test_tick_size_change_controls_material_move_threshold() -> None:
    result = measure_reaction(
        event=_event(),
        updates=(
            _market_update(
                ordinal=1,
                receive_ns=9 * NS,
                bid=".39",
                ask=".41",
                tick_size=".01",
            ),
            _market_update(
                ordinal=2,
                receive_ns=11 * NS,
                bid=".41",
                ask=".43",
                tick_size=".05",
            ),
            _market_update(
                ordinal=3,
                receive_ns=12 * NS,
                bid=".44",
                ask=".46",
                tick_size=".05",
            ),
        ),
        venue_tick=Decimal(".01"),
        control_noise_95=Decimal(".01"),
        next_salient_event_monotonic_ns=15 * NS,
    )

    assert result.first_tick_move_monotonic_ns == 12 * NS + 100
    assert result.first_material_move_monotonic_ns == 12 * NS + 100


def test_shadow_taker_walks_ask_then_bid_and_never_claims_a_fill() -> None:
    entry = replace(
        _market_update(
            ordinal=1, receive_ns=10 * NS, bid=".39", ask=".41"
        ),
        asks=(
            BookLevel(Decimal(".41"), Decimal("3")),
            BookLevel(Decimal(".42"), Decimal("4")),
        ),
    )
    exit_update = replace(
        _market_update(
            ordinal=2, receive_ns=15 * NS, bid=".44", ask=".46"
        ),
        bids=(
            BookLevel(Decimal(".44"), Decimal("2")),
            BookLevel(Decimal(".43"), Decimal("5")),
        ),
    )

    quote = build_shadow_taker_observation(
        decision_ready_monotonic_ns=10 * NS,
        entry_update=entry,
        exit_update=exit_update,
        requested_size=Decimal("5"),
        fee=Decimal(".01"),
        venue_rule_snapshot_ref="sha256:" + "4" * 64,
    )

    assert quote.entry_ask_vwap == Decimal(".414")
    assert quote.exit_bid_vwap == Decimal(".434")
    assert quote.available_entry_size == Decimal("7")
    assert quote.available_exit_size == Decimal("7")
    assert quote.quote_status == "COMPLETE_SHADOW_QUOTE"
    assert not hasattr(quote, "fill")
    assert not hasattr(quote, "order_id")

    with pytest.raises(ValueError, match="continuous"):
        build_shadow_taker_observation(
            decision_ready_monotonic_ns=10 * NS,
            entry_update=entry,
            exit_update=replace(
                exit_update, continuity_status="SUSPENDED"
            ),
            requested_size=Decimal("5"),
            fee=Decimal(".01"),
            venue_rule_snapshot_ref="sha256:" + "4" * 64,
        )
