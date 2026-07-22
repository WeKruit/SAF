from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_market.models.validation import (
    ValidationInputError,
    evaluate_model_vs_prior,
    evaluate_probabilities,
    game_grouped_walk_forward,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for game_index in range(8):
        for event_index in range(3):
            rows.append(
                {
                    "game_id": f"game-{game_index}",
                    "played_at": start
                    + pd.Timedelta(days=game_index, minutes=event_index),
                    "target": game_index % 2,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_never_splits_a_game_or_uses_future_games() -> None:
    folds = list(game_grouped_walk_forward(_frame(), min_train_games=3))

    assert folds
    for train, test in folds:
        assert set(train.game_id).isdisjoint(set(test.game_id))
        assert train.played_at.max() < test.played_at.min()
        assert test.game_id.nunique() == 1


def test_walk_forward_rejects_naive_time_and_overlapping_game_intervals() -> None:
    frame = _frame()
    frame["played_at"] = frame["played_at"].dt.tz_localize(None)
    with pytest.raises(ValidationInputError, match="timezone-aware UTC"):
        list(game_grouped_walk_forward(frame, min_train_games=3))


def test_metric_report_contains_required_calibration_and_cluster_ci() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 4, dtype=int)
    probabilities = np.array(
        [0.10, 0.25, 0.75, 0.90, 0.35, 0.65, 0.40, 0.60] * 4,
        dtype=float,
    )
    groups = np.repeat([f"game-{index}" for index in range(8)], 4)

    report = evaluate_probabilities(
        y,
        probabilities,
        groups=groups,
        bootstrap_samples=100,
        confidence_level=0.90,
        minimum_valid_samples=100,
        seed=17,
    )

    assert {
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
        "bootstrap_ci",
    } <= report.keys()
    assert set(report["bootstrap_ci"]) == {
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
    }
    assert report["clusters"] == 8
    assert report["confidence_level"] == 0.90
    assert report["bootstrap_samples_valid"] == 100


def test_metrics_reject_unclustered_or_invalid_predictions() -> None:
    with pytest.raises(ValidationInputError, match="length"):
        evaluate_probabilities(
            [0, 1],
            [0.2],
            groups=["a", "b"],
            bootstrap_samples=20,
            confidence_level=0.95,
            minimum_valid_samples=20,
            seed=1,
        )
    with pytest.raises(ValidationInputError, match=r"\[0, 1\]"):
        evaluate_probabilities(
            [0, 1],
            [0.2, 1.2],
            groups=["a", "b"],
            bootstrap_samples=20,
            confidence_level=0.95,
            minimum_valid_samples=20,
            seed=1,
        )


def test_metrics_require_registered_confidence_and_enough_valid_resamples() -> None:
    with pytest.raises(ValidationInputError, match="confidence_level"):
        evaluate_probabilities(
            [0, 1],
            [0.2, 0.8],
            groups=["a", "b"],
            bootstrap_samples=20,
            confidence_level=1.0,
            minimum_valid_samples=20,
            seed=1,
        )

    with pytest.raises(ValidationInputError, match="valid clustered bootstrap"):
        evaluate_probabilities(
            [0, 0, 1, 1],
            [0.2, 0.3, 0.7, 0.8],
            groups=["only-zero", "only-zero", "only-one", "only-one"],
            bootstrap_samples=20,
            confidence_level=0.95,
            minimum_valid_samples=20,
            seed=1,
        )


def test_model_vs_prior_reports_deterministic_paired_cluster_delta_ci() -> None:
    y = np.tile([0, 1], 10)
    prior_strength = np.array(
        [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.55, 0.60, 0.65, 0.70]
    )
    model_strength = 1 - np.sqrt((1 - prior_strength) ** 2 - 0.01)
    prior = np.column_stack((1 - prior_strength, prior_strength)).reshape(-1)
    model = np.column_stack((1 - model_strength, model_strength)).reshape(-1)
    groups = np.repeat([f"game-{index}" for index in range(10)], 2)

    expected = evaluate_model_vs_prior(
        y,
        model,
        prior,
        groups=groups,
        bootstrap_samples=100,
        confidence_level=0.90,
        minimum_valid_samples=100,
        seed=23,
    )
    repeated = evaluate_model_vs_prior(
        y,
        model,
        prior,
        groups=groups,
        bootstrap_samples=100,
        confidence_level=0.90,
        minimum_valid_samples=100,
        seed=23,
    )

    assert repeated == expected
    assert expected["delta_definition"] == "model_minus_prior"
    assert set(expected["delta"]) == {"brier", "log_loss"}
    assert set(expected["delta_bootstrap_ci"]) == {"brier", "log_loss"}
    for metric in ("brier", "log_loss"):
        assert expected["delta"][metric] == pytest.approx(
            expected["model_metrics"][metric]
            - expected["prior_metrics"][metric]
        )
        assert expected["delta_bootstrap_ci"][metric][1] < 0
    assert expected["delta_bootstrap_ci"]["brier"] == pytest.approx(
        (-0.01, -0.01)
    )
    assert expected["confidence_level"] == 0.90
    assert expected["minimum_valid_samples"] == 100
    assert expected["bootstrap_samples_requested"] == 100
    assert expected["bootstrap_samples_valid"] == 100
    assert expected["clusters"] == 10
    assert expected["observations"] == 20


def test_model_vs_prior_requires_enough_valid_paired_resamples() -> None:
    with pytest.raises(ValidationInputError, match="valid clustered bootstrap"):
        evaluate_model_vs_prior(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            [0.3, 0.4, 0.6, 0.7],
            groups=["only-zero", "only-zero", "only-one", "only-one"],
            bootstrap_samples=20,
            confidence_level=0.95,
            minimum_valid_samples=20,
            seed=1,
        )
