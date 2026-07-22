from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_market.models.validation import (
    ValidationInputError,
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


def test_metrics_reject_unclustered_or_invalid_predictions() -> None:
    with pytest.raises(ValidationInputError, match="length"):
        evaluate_probabilities(
            [0, 1],
            [0.2],
            groups=["a", "b"],
            bootstrap_samples=20,
            seed=1,
        )
    with pytest.raises(ValidationInputError, match=r"\[0, 1\]"):
        evaluate_probabilities(
            [0, 1],
            [0.2, 1.2],
            groups=["a", "b"],
            bootstrap_samples=20,
            seed=1,
        )
