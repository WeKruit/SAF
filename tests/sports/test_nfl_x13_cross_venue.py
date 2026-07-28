from __future__ import annotations

import pytest

from prediction_market.sports.nfl_x13_cross_venue import (
    CrossVenueStudyError,
    build_cross_venue_pairs,
    evaluate_shadow_locked_pair,
)


HASH = "sha256:" + "a" * 64


def _reaction(
    venue: str,
    *,
    effect: float,
    episode_id: str = "episode-1",
    actual_observation: bool = True,
    order_ambiguous: bool = False,
    contaminated: bool = False,
) -> dict[str, object]:
    return {
        "game_id": "2025_01_AAA_BBB",
        "episode_id": episode_id,
        "logical_market_id": "winner",
        "outcome": "AAA",
        "market_family": "moneyline",
        "outcome_orientation": "AAA",
        "horizon_seconds": 10,
        "venue": venue,
        "signed_price_change": effect,
        "pre_event_actual_trade": 0.44,
        "first_post_event_trade": 0.49,
        "trade_count": 3,
        "volume": 27.0,
        "staleness_seconds": 0.4,
        "actual_observation": actual_observation,
        "analysis_eligible": actual_observation,
        "order_ambiguous": order_ambiguous,
        "contaminated": contaminated,
        "source_hashes": (HASH,),
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
    }


def test_pairs_same_episode_and_exposes_unmatched_audit() -> None:
    result = build_cross_venue_pairs(
        [
            _reaction("polymarket", effect=0.08),
            _reaction("kalshi", effect=0.03),
            _reaction("polymarket", effect=0.02, episode_id="episode-2"),
        ]
    )

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.polymarket_change == pytest.approx(0.08)
    assert pair.kalshi_change == pytest.approx(0.03)
    assert pair.polymarket_minus_kalshi == pytest.approx(0.05)
    assert pair.claim_boundary == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert result.audit["status"].tolist() == ["PAIRED", "MISSING_KALSHI"]


def test_historical_pair_refuses_ambiguous_or_non_actual_observation() -> None:
    result = build_cross_venue_pairs(
        [
            _reaction("polymarket", effect=0.08, order_ambiguous=True),
            _reaction("kalshi", effect=0.03),
            _reaction("polymarket", effect=0.04, episode_id="episode-2"),
            _reaction(
                "kalshi",
                effect=0.02,
                episode_id="episode-2",
                actual_observation=False,
            ),
        ]
    )

    assert not result.pairs
    assert set(result.audit["status"]) == {
        "ORDER_AMBIGUOUS",
        "NON_ACTUAL_OR_INELIGIBLE",
    }


def test_duplicate_venue_identity_fails_closed() -> None:
    with pytest.raises(CrossVenueStudyError, match="duplicate venue"):
        build_cross_venue_pairs(
            [
                _reaction("polymarket", effect=0.08),
                _reaction("polymarket", effect=0.07),
                _reaction("kalshi", effect=0.03),
            ]
        )


def test_shadow_locked_pair_requires_identical_rules_fresh_quotes_and_cost_below_one() -> None:
    quote = evaluate_shadow_locked_pair(
        proposition_id="game-winner-aaa",
        settlement_rule_sha256="sha256:" + "b" * 64,
        polymarket_yes_ask=0.44,
        kalshi_no_ask=0.49,
        polymarket_fee=0.01,
        kalshi_fee=0.01,
        polymarket_slippage=0.0,
        kalshi_slippage=0.0,
        polymarket_quote_age_ns=100_000_000,
        kalshi_quote_age_ns=120_000_000,
        max_quote_age_ns=250_000_000,
        size_contracts=10,
    )

    assert quote.status == "LOCKED_PAYOUT_SHADOW"
    assert quote.total_cost_per_contract == pytest.approx(0.95)
    assert quote.executable is False

    stale = evaluate_shadow_locked_pair(
        proposition_id="game-winner-aaa",
        settlement_rule_sha256="sha256:" + "b" * 64,
        polymarket_yes_ask=0.44,
        kalshi_no_ask=0.49,
        polymarket_fee=0.01,
        kalshi_fee=0.01,
        polymarket_slippage=0.0,
        kalshi_slippage=0.0,
        polymarket_quote_age_ns=300_000_000,
        kalshi_quote_age_ns=120_000_000,
        max_quote_age_ns=250_000_000,
        size_contracts=10,
    )
    assert stale.status == "STALE_QUOTE"
