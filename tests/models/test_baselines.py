from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_market.models.nba import BaselineInputError, fit_predict_nba_baselines
from prediction_market.models.nfl import nfl_logistic_features
from prediction_market.models.soccer import dixon_coles_outcome_probabilities


def _nba_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for index in range(40):
        prediction_at = start + pd.Timedelta(days=index)
        rows.append(
            {
                "game_id": f"game-{index}",
                "prediction_at": prediction_at,
                "score_diff": float((index % 11) - 5),
                "seconds_remaining": float(3000 - (index % 20) * 100),
                "market_prior": 0.25 + (index % 10) * 0.05,
                "target": index % 2,
                "score_diff__available_at": prediction_at,
                "seconds_remaining__available_at": prediction_at,
                "market_prior__available_at": prediction_at,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.iloc[:32].copy(), frame.iloc[32:].copy()


def test_nba_compares_market_prior_logistic_and_gbdt_deterministically() -> None:
    train, test = _nba_frames()

    first = fit_predict_nba_baselines(
        train,
        test,
        feature_columns=("score_diff", "seconds_remaining"),
        target_column="target",
        market_prior_column="market_prior",
        seed=7,
    )
    second = fit_predict_nba_baselines(
        train,
        test,
        feature_columns=("score_diff", "seconds_remaining"),
        target_column="target",
        market_prior_column="market_prior",
        seed=7,
    )

    assert set(first) == {"market_prior", "logistic", "gbdt"}
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert all(np.all((values >= 0) & (values <= 1)) for values in first.values())


def test_nba_point_in_time_feature_contract_rejects_future_availability() -> None:
    train, test = _nba_frames()
    test.loc[test.index[0], "score_diff__available_at"] = (
        test.loc[test.index[0], "prediction_at"] + pd.Timedelta(seconds=1)
    )

    with pytest.raises(BaselineInputError, match="point-in-time"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="market_prior",
            seed=7,
        )


def test_nfl_classic_features_are_point_in_time_state_only() -> None:
    features = nfl_logistic_features(
        score_differential=7,
        seconds_remaining=120,
        possession_is_home=True,
        home_timeouts=2,
        away_timeouts=1,
    )

    assert features == {
        "score_differential": 7.0,
        "seconds_remaining": 120.0,
        "possession_is_home": 1.0,
        "home_timeouts": 2.0,
        "away_timeouts": 1.0,
    }


def test_soccer_poisson_baseline_returns_normalized_outcomes() -> None:
    probabilities = dixon_coles_outcome_probabilities(
        home_goal_rate=1.4,
        away_goal_rate=1.0,
        home_goals=1,
        away_goals=0,
        minutes_remaining=30,
        max_additional_goals=10,
        rho=-0.05,
    )

    assert set(probabilities) == {"home_win", "draw", "away_win"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert probabilities["home_win"] > probabilities["away_win"]
