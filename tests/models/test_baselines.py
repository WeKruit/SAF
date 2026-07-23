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
        game_start_at = prediction_at - pd.Timedelta(hours=1)
        rows.append(
            {
                "game_id": f"game-{index}",
                "prediction_at": prediction_at,
                "game_start_at": game_start_at,
                "score_diff": float((index % 11) - 5),
                "seconds_remaining": float(3000 - (index % 20) * 100),
                "market_prior": 0.25 + (index % 10) * 0.05,
                "target": index % 2,
                "score_diff__available_at": prediction_at,
                "seconds_remaining__available_at": prediction_at,
                "market_prior__available_at": game_start_at
                - pd.Timedelta(seconds=1),
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


def test_nba_logistic_and_gbdt_use_frozen_market_prior_as_an_input() -> None:
    train, test = _nba_frames()
    for frame in (train, test):
        frame["score_diff"] = 0.0
        frame["seconds_remaining"] = 1000.0
        frame["market_prior"] = np.where(frame["target"] == 1, 0.9, 0.1)

    predictions = fit_predict_nba_baselines(
        train,
        test,
        feature_columns=("score_diff", "seconds_remaining"),
        target_column="target",
        market_prior_column="market_prior",
        seed=7,
    )

    assert np.ptp(predictions["logistic"]) > 0.1
    assert np.ptp(predictions["gbdt"]) > 0.1


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


def test_nba_rejects_game_overlap_between_train_and_test() -> None:
    train, test = _nba_frames()
    test.loc[test.index[0], "game_id"] = train.iloc[0]["game_id"]

    with pytest.raises(BaselineInputError, match="game.*overlap"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="market_prior",
            seed=7,
        )


def test_nba_rejects_training_time_not_strictly_before_test() -> None:
    train, test = _nba_frames()
    train.loc[train.index[-1], "prediction_at"] = test["prediction_at"].min()

    with pytest.raises(BaselineInputError, match="strictly earlier"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="market_prior",
            seed=7,
        )


def test_nba_rejects_prior_that_was_not_frozen_before_game_start() -> None:
    train, test = _nba_frames()
    test.loc[test.index[0], "market_prior__available_at"] = (
        test.loc[test.index[0], "game_start_at"] + pd.Timedelta(seconds=1)
    )

    with pytest.raises(BaselineInputError, match="pregame prior"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="market_prior",
            seed=7,
        )


def test_nba_rejects_prior_that_was_unavailable_at_prediction_time() -> None:
    train, test = _nba_frames()
    row = test.index[0]
    test.loc[row, "game_start_at"] = test.loc[row, "prediction_at"] + pd.Timedelta(
        hours=2
    )
    test.loc[row, "market_prior__available_at"] = test.loc[
        row, "prediction_at"
    ] + pd.Timedelta(hours=1)

    with pytest.raises(BaselineInputError, match="prediction_at"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="market_prior",
            seed=7,
        )


def test_nba_rejects_target_and_market_prior_role_collision() -> None:
    train, test = _nba_frames()
    for frame in (train, test):
        frame["target__available_at"] = frame["game_start_at"] - pd.Timedelta(
            seconds=1
        )

    with pytest.raises(BaselineInputError, match="distinct"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="target",
            seed=7,
        )


@pytest.mark.parametrize(
    "invalid_target",
    ["game_id", "prediction_at", "game_start_at", "score_diff__available_at"],
)
def test_nba_rejects_structural_or_availability_target_roles(
    invalid_target: str,
) -> None:
    train, test = _nba_frames()

    with pytest.raises(BaselineInputError, match="target column.*role"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column=invalid_target,
            market_prior_column="market_prior",
            seed=7,
        )


@pytest.mark.parametrize(
    "invalid_prior",
    ["game_id", "prediction_at", "game_start_at", "score_diff__available_at"],
)
def test_nba_rejects_structural_or_availability_market_prior_roles(
    invalid_prior: str,
) -> None:
    train, test = _nba_frames()

    with pytest.raises(BaselineInputError, match="market prior column.*role"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column=invalid_prior,
            seed=7,
        )


def test_nba_rejects_numeric_game_identity_disguised_as_market_prior() -> None:
    train, test = _nba_frames()
    for frame in (train, test):
        frame["game_id"] = frame.index.to_numpy(dtype=float) / 100
        frame["game_id__available_at"] = frame["game_start_at"] - pd.Timedelta(
            seconds=1
        )

    with pytest.raises(BaselineInputError, match="market prior column.*role"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=("score_diff", "seconds_remaining"),
            target_column="target",
            market_prior_column="game_id",
            seed=7,
        )


@pytest.mark.parametrize(
    "leaking_feature",
    [
        "target",
        "game_id",
        "prediction_at",
        "game_start_at",
        "market_prior",
        "score_diff__available_at",
    ],
)
def test_nba_rejects_target_identity_time_and_prior_features(
    leaking_feature: str,
) -> None:
    train, test = _nba_frames()

    with pytest.raises(BaselineInputError, match="reserved.*feature"):
        fit_predict_nba_baselines(
            train,
            test,
            feature_columns=(leaking_feature,),
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
