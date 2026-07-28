from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_market.research.nfl_x15_models import (
    MODEL_IDS,
    X15ModelRun,
    X15ModelInputError,
    build_x15_week_folds,
    direction_labels,
    run_x15_walk_forward,
    training_noise_threshold,
)


def _fold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": f"2025_{week:02d}_A_B",
                "nfl_week": week,
                "venue": venue,
            }
            for week in range(1, 13)
            for venue in ("polymarket", "kalshi")
        ]
    )


def test_builds_the_five_frozen_expanding_nfl_week_folds() -> None:
    folds = build_x15_week_folds(_fold_frame())

    assert [
        (fold.train_weeks, fold.validation_weeks)
        for fold in folds
    ] == [
        ((1, 2), (3, 4)),
        (tuple(range(1, 5)), (5, 6)),
        (tuple(range(1, 7)), (7, 8)),
        (tuple(range(1, 9)), (9, 10)),
        (tuple(range(1, 11)), (11, 12)),
    ]
    for fold in folds:
        assert set(fold.train_game_ids).isdisjoint(fold.validation_game_ids)


def test_rejects_a_game_assigned_to_multiple_nfl_weeks() -> None:
    frame = _fold_frame()
    frame.loc[len(frame)] = {
        "game_id": "2025_01_A_B",
        "nfl_week": 2,
        "venue": "polymarket",
    }

    with pytest.raises(X15ModelInputError, match="one NFL week"):
        build_x15_week_folds(frame)


def test_rejects_rows_outside_the_frozen_development_weeks() -> None:
    frame = _fold_frame()
    frame.loc[len(frame)] = {
        "game_id": "2025_13_HOLDOUT",
        "nfl_week": 13,
        "venue": "polymarket",
    }

    with pytest.raises(X15ModelInputError, match="weeks 1 through 12"):
        build_x15_week_folds(frame)


def test_direction_labels_use_the_fold_noise_threshold() -> None:
    labels = direction_labels(
        pd.Series([-0.011, -0.01, -0.009, 0.0, 0.009, 0.01, 0.011]),
        noise_threshold=0.01,
    )

    assert labels.tolist() == [
        "DOWN",
        "NO_MOVE",
        "NO_MOVE",
        "NO_MOVE",
        "NO_MOVE",
        "NO_MOVE",
        "UP",
    ]


def _landmarks() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_values = (-0.03, -0.02, -0.002, 0.002, 0.02, 0.03)
    reference_gaps = (0.025, 0.015, 0.001, -0.001, -0.015, -0.025)
    for week in range(1, 13):
        game_id = f"2025_{week:02d}_A_B"
        for venue_index, venue in enumerate(("polymarket", "kalshi")):
            for episode_index, (target, reference_gap) in enumerate(
                zip(target_values, reference_gaps, strict=True)
            ):
                rows.append(
                    {
                        "game_id": game_id,
                        "episode_id": f"{game_id}:e{episode_index}",
                        "venue": venue,
                        "nfl_week": week,
                        "landmark_seconds": 5,
                        "endpoint_seconds": 30,
                        "target_delta": target
                        + venue_index * 0.0005
                        + week * 0.00001,
                        "reference_gap_at_landmark": reference_gap,
                        "factor_id": (
                            "NFL.CONTROL.ROUTINE"
                            if episode_index in (2, 3)
                            else (
                                "NFL.EVENT.NEGATIVE"
                                if target < 0
                                else "NFL.EVENT.POSITIVE"
                            )
                        ),
                        "score_margin": episode_index - 3,
                        "game_seconds_remaining": 3_600 - week * 60,
                        "pit_strength": 0.4 + week * 0.01,
                        "decision_eligible": True,
                        "target_eligible": True,
                        "eligible": True,
                        "exclusion_reasons": [],
                        "is_matched_control": episode_index in (2, 3),
                    }
                )
    return pd.DataFrame(rows)


def test_training_noise_uses_only_training_matched_controls() -> None:
    train = pd.DataFrame(
        {
            "target_delta": [0.01, -0.02, 0.03, -0.04, 0.9],
            "is_matched_control": [True, True, True, True, False],
        }
    )

    assert training_noise_threshold(train) == pytest.approx(0.0385)


def test_training_noise_rejects_missing_control_membership() -> None:
    train = pd.DataFrame(
        {
            "target_delta": [0.01, -0.02],
            "is_matched_control": pd.Series([True, pd.NA], dtype="boolean"),
        }
    )

    with pytest.raises(X15ModelInputError, match="control membership"):
        training_noise_threshold(train)


def test_walk_forward_emits_separate_venue_oof_predictions_and_fold_metrics() -> None:
    frame = _landmarks()
    original = frame.copy(deep=True)

    result = run_x15_walk_forward(
        frame,
        feature_columns=(
            "reference_gap_at_landmark",
            "factor_id",
            "score_margin",
            "game_seconds_remaining",
            "pit_strength",
        ),
        random_state=20260728,
    )

    assert isinstance(result, X15ModelRun)
    assert set(result.oof_predictions["model_id"]) == set(MODEL_IDS)
    assert len(result.oof_predictions) == 10 * 2 * 6 * len(MODEL_IDS)
    assert len(result.fold_metrics) == 5 * 2 * len(MODEL_IDS)
    assert set(result.fold_metrics["venue"]) == {"polymarket", "kalshi"}
    assert set(result.fold_metrics.groupby(["fold_id", "model_id"])["venue"].nunique()) == {
        2
    }
    probability_rows = result.oof_predictions[
        ["prob_down", "prob_no_move", "prob_up"]
    ].dropna()
    assert len(probability_rows) == len(result.oof_predictions) // len(MODEL_IDS)
    assert np.allclose(
        probability_rows.sum(axis=1),
        np.ones(len(probability_rows)),
        rtol=0,
        atol=1e-12,
    )
    for row in result.fold_metrics.itertuples(index=False):
        assert set(row.train_game_ids).isdisjoint(row.validation_game_ids)
        assert row.training_venue == row.venue
        assert row.noise_threshold >= 0.01
        if row.model_id == "multinomial_logistic_v1":
            assert np.isfinite(row.multiclass_log_loss)
            assert row.probability_status == "CALIBRATED_CLASS_PROBABILITY"
        else:
            assert np.isnan(row.multiclass_log_loss)
            assert row.probability_status == "NOT_PROBABILISTIC"
    baseline = result.oof_predictions[
        result.oof_predictions["model_id"] == "reference_gap_baseline_v1"
    ]
    assert baseline["predicted_delta"].to_numpy() == pytest.approx(
        -baseline["reference_gap_at_landmark"].to_numpy()
    )
    pd.testing.assert_frame_equal(frame, original)


def test_walk_forward_predicts_decision_rows_without_using_future_target_availability() -> None:
    frame = _landmarks()
    mask = (
        frame["nfl_week"].eq(3)
        & frame["venue"].eq("kalshi")
        & frame["episode_id"].str.endswith(":e0")
    )
    frame.loc[mask, "target_eligible"] = False
    frame.loc[mask, "eligible"] = False
    frame.loc[mask, "target_delta"] = np.nan
    frame.loc[mask, "exclusion_reasons"] = [
        ["endpoint_no_actual_trade"] for _ in range(int(mask.sum()))
    ]

    result = run_x15_walk_forward(
        frame,
        feature_columns=(
            "reference_gap_at_landmark",
            "factor_id",
            "score_margin",
            "game_seconds_remaining",
            "pit_strength",
        ),
    )

    selected = result.oof_predictions.loc[
        result.oof_predictions["source_row_id"].isin(frame.index[mask])
    ]
    assert len(selected) == len(MODEL_IDS)
    assert selected["decision_eligible"].eq(True).all()
    assert selected["target_eligible"].eq(False).all()
    assert selected["target_delta"].isna().all()
    assert selected["direction_label"].isna().all()
    assert selected["predicted_delta"].notna().all()


def test_walk_forward_ignores_nondecision_rows_with_unavailable_landmark_features() -> None:
    frame = _landmarks()
    mask = (
        frame["nfl_week"].eq(3)
        & frame["venue"].eq("kalshi")
        & frame["episode_id"].str.endswith(":e0")
    )
    source_indexes = frame.index[mask]
    frame.loc[mask, ["decision_eligible", "target_eligible", "eligible"]] = False
    frame.loc[mask, ["reference_gap_at_landmark", "target_delta"]] = np.nan
    frame.loc[mask, "exclusion_reasons"] = [
        ["landmark_no_actual_trade"] for _ in range(int(mask.sum()))
    ]

    result = run_x15_walk_forward(
        frame,
        feature_columns=(
            "reference_gap_at_landmark",
            "factor_id",
            "score_margin",
            "game_seconds_remaining",
            "pit_strength",
        ),
    )

    assert not result.oof_predictions["source_row_id"].isin(
        source_indexes
    ).any()


def test_walk_forward_is_deterministic() -> None:
    frame = _landmarks()
    kwargs = {
        "feature_columns": (
            "reference_gap_at_landmark",
            "factor_id",
            "score_margin",
            "game_seconds_remaining",
            "pit_strength",
        ),
        "random_state": 20260728,
    }

    first = run_x15_walk_forward(frame, **kwargs)
    second = run_x15_walk_forward(frame, **kwargs)

    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)
    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)


def test_walk_forward_rejects_leakage_and_eligibility_contradictions() -> None:
    frame = _landmarks()
    frame.loc[0, "exclusion_reasons"] = ["contaminated"]

    with pytest.raises(
        X15ModelInputError,
        match="target-eligible rows.*exclusion",
    ):
        run_x15_walk_forward(
            frame,
            feature_columns=("target_delta", "score_margin"),
        )

    frame = _landmarks()
    frame.loc[0, "eligible"] = False
    with pytest.raises(X15ModelInputError, match="exact alias"):
        run_x15_walk_forward(
            frame,
            feature_columns=("target_delta", "score_margin"),
        )

    with pytest.raises(X15ModelInputError, match="leakage"):
        run_x15_walk_forward(
            _landmarks(),
            feature_columns=("target_delta", "score_margin"),
        )


def test_walk_forward_rejects_missing_eligibility() -> None:
    frame = _landmarks()
    frame["eligible"] = frame["eligible"].astype("boolean")
    frame.loc[0, "eligible"] = pd.NA

    with pytest.raises(X15ModelInputError, match="eligible must be present"):
        run_x15_walk_forward(
            frame,
            feature_columns=("score_margin", "pit_strength"),
        )
