from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.sports.nfl_x13_historical_study import (
    HistoricalStudyError,
    build_historical_factor_study,
)


def _row(
    observation_id: str,
    *,
    game_id: str,
    episode_id: str,
    venue: str,
    horizon_seconds: int,
    effect: float,
    game_seconds_remaining: int,
    liquidity: float = 10.0,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "factor_id": "NFL.EVENT.TEST",
        "factor_version": "v1",
        "game_id": game_id,
        "episode_id": episode_id,
        "logical_market_id": f"{venue}:winner",
        "outcome": "AAA",
        "venue": venue,
        "market_family": "moneyline",
        "outcome_orientation": "AAA",
        "horizon_seconds": horizon_seconds,
        "effect": effect,
        "pre_probability": 0.50,
        "actual_trade_count": 2,
        "staleness_seconds": 1.0,
        "liquidity": liquidity,
        "game_time_seconds": game_seconds_remaining,
        "signed_margin": 0,
        "down": 2.0,
        "distance": 7,
        "field_position": 40,
        "shape_classification": "GRADUAL",
        "claim_boundary_dense": "PRELIMINARY_SOURCE_TIME_ONLY",
        "eligible": True,
    }


def _fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    effects = {
        "game-1": (0.02, 0.04),
        "game-2": (-0.01, -0.03),
        "game-3": (0.00, 0.02),
    }
    for game_index, (game_id, (effect_10, effect_60)) in enumerate(
        effects.items(), start=1
    ):
        episode_id = f"episode-{game_index}"
        for venue in ("polymarket", "kalshi"):
            venue_scale = 1.0 if venue == "polymarket" else 0.5
            rows.append(
                _row(
                    f"{game_id}:{venue}:10",
                    game_id=game_id,
                    episode_id=episode_id,
                    venue=venue,
                    horizon_seconds=10,
                    effect=effect_10 * venue_scale,
                    game_seconds_remaining=3000 - 800 * game_index,
                )
            )
            rows.append(
                _row(
                    f"{game_id}:{venue}:60",
                    game_id=game_id,
                    episode_id=episode_id,
                    venue=venue,
                    horizon_seconds=60,
                    effect=effect_60 * venue_scale,
                    game_seconds_remaining=3000 - 800 * game_index,
                )
            )
    return pd.DataFrame(rows)


def test_factor_study_publishes_empirical_probability_and_cluster_uncertainty() -> None:
    result = build_historical_factor_study(
        _fixture(),
        bootstrap_samples=200,
        bootstrap_seed=20260727,
    )

    row = result.summary[
        (result.summary["venue"] == "polymarket")
        & (result.summary["horizon_seconds"] == 10)
    ].iloc[0]
    assert row["observed_n"] == 3
    assert row["game_count"] == 3
    assert row["mean"] == pytest.approx(1 / 300)
    assert row["median"] == pytest.approx(0.0)
    assert row["mad"] == pytest.approx(0.01)
    assert row["p_positive"] == pytest.approx(1 / 3)
    assert row["p_negative"] == pytest.approx(1 / 3)
    assert row["p_zero"] == pytest.approx(1 / 3)
    assert row["p_abs_gte_1pp"] == pytest.approx(2 / 3)
    assert row["bootstrap_ci_low"] <= row["mean"] <= row["bootstrap_ci_high"]
    assert 0 <= row["bh_q_value"] <= 1
    assert 0 <= row["loo_same_sign_rate"] <= 1

    histogram = result.histogram[
        (result.histogram["venue"] == "polymarket")
        & (result.histogram["horizon_seconds"] == 10)
    ]
    assert histogram["count"].sum() == 3
    assert histogram["probability"].sum() == pytest.approx(1.0)

    game_effects = result.game_effects[
        (result.game_effects["venue"] == "polymarket")
        & (result.game_effects["horizon_seconds"] == 10)
    ]
    assert set(game_effects["game_id"]) == {"game-1", "game-2", "game-3"}


def test_factor_study_breaks_down_game_phase_and_path_shape() -> None:
    frame = _fixture()
    frame["quarter"] = [2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 5, 5]
    frame.loc[frame["quarter"].eq(5), "game_time_seconds"] = 0
    result = build_historical_factor_study(
        frame,
        bootstrap_samples=50,
        bootstrap_seed=7,
    )

    phases = set(result.state_breakdown["game_phase"])
    assert {"Q2", "Q3", "Q4", "OT"}.issubset(phases)
    assert {
        "game_phase",
        "time_bucket",
        "score_state",
        "down",
        "distance_bucket",
        "field_position_bucket",
        "pre_probability_bucket",
        "staleness_bucket",
        "trade_count_bucket",
        "trade_size_bucket",
    }.issubset(set(result.state_breakdown["breakdown_dimension"]))

    poly_path = result.path_probabilities[
        result.path_probabilities["venue"] == "polymarket"
    ].iloc[0]
    assert poly_path["paired_path_n"] == 3
    assert poly_path["continuation_rate"] == pytest.approx(1.0)
    assert poly_path["reversal_rate"] == pytest.approx(0.0)
    assert poly_path["overshoot_rate"] == pytest.approx(0.0)


def test_factor_study_retains_extreme_case_lineage() -> None:
    result = build_historical_factor_study(
        _fixture(),
        bootstrap_samples=25,
        bootstrap_seed=13,
    )

    assert not result.cases.empty
    assert {
        "game_id",
        "episode_id",
        "logical_market_id",
        "case_type",
        "effect",
    }.issubset(result.cases.columns)
    assert {"MAX_POSITIVE", "MAX_NEGATIVE"}.issubset(set(result.cases["case_type"]))


def test_factor_study_pairs_same_episode_moneyline_across_venues() -> None:
    result = build_historical_factor_study(
        _fixture(),
        bootstrap_samples=25,
        bootstrap_seed=11,
    )

    assert len(result.venue_pairs) == 6
    pair = result.venue_pairs.iloc[0]
    assert pair["polymarket_effect"] == pytest.approx(
        pair["kalshi_effect"] * 2
    )
    assert pair["polymarket_minus_kalshi"] == pytest.approx(
        pair["polymarket_effect"] - pair["kalshi_effect"]
    )

    diagnostics = result.venue_diagnostics
    assert set(diagnostics["comparison_mode"]) == {
        "RAW",
        "SAME_EPISODE_PAIRED",
    }


def test_factor_study_rejects_noneligible_or_non_source_time_rows() -> None:
    frame = _fixture()
    frame.loc[0, "eligible"] = False
    with pytest.raises(HistoricalStudyError, match="eligible"):
        build_historical_factor_study(frame, bootstrap_samples=10)

    frame = _fixture()
    frame.loc[0, "claim_boundary_dense"] = "LIVE"
    with pytest.raises(HistoricalStudyError, match="SOURCE_TIME_ONLY"):
        build_historical_factor_study(frame, bootstrap_samples=10)


def test_same_market_observation_may_belong_to_distinct_factors() -> None:
    frame = _fixture().iloc[[0]].copy()
    second = frame.copy()
    second["factor_id"] = "NFL.STATE.TEST"
    combined = pd.concat([frame, second], ignore_index=True)

    result = build_historical_factor_study(combined, bootstrap_samples=10)

    assert set(result.summary["factor_id"]) == {
        "NFL.EVENT.TEST",
        "NFL.STATE.TEST",
    }

    duplicate = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(HistoricalStudyError, match="factor observation grain"):
        build_historical_factor_study(duplicate, bootstrap_samples=10)
