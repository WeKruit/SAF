from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from prediction_market.models.validation import (
    ValidationInputError,
    evaluate_multiclass_probabilities,
)


CLASSES = ("home_win", "draw", "away_win")


def _balanced_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.array(list(CLASSES) * 8, dtype=object)
    probabilities = np.array(
        [
            [0.72, 0.18, 0.10],
            [0.18, 0.64, 0.18],
            [0.10, 0.18, 0.72],
        ]
        * 8,
        dtype=float,
    )
    groups = np.repeat([f"game-{index}" for index in range(8)], 3)
    return targets, probabilities, groups


def _pit_times(observations: int) -> tuple[pd.Series, pd.Series]:
    prediction_at = pd.Series(
        pd.date_range("2026-01-01", periods=observations, freq="min", tz="UTC")
    )
    feature_available_at = prediction_at - pd.Timedelta(seconds=1)
    return prediction_at, feature_available_at


def _evaluate(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    **kwargs: object,
) -> dict[str, object]:
    prediction_at, feature_available_at = _pit_times(len(targets))
    inputs: dict[str, object] = {
        "prediction_at": prediction_at,
        "feature_available_at": feature_available_at,
    }
    inputs.update(kwargs)
    return evaluate_multiclass_probabilities(
        targets,
        probabilities,
        classes=CLASSES,
        groups=groups,
        bootstrap_samples=100,
        confidence_level=0.90,
        minimum_valid_samples=80,
        seed=20260722,
        **inputs,
    )


def test_multiclass_metrics_are_joint_and_hand_computable() -> None:
    targets = np.array(["home_win", "draw", "away_win"], dtype=object)
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.50, 0.30],
            [0.10, 0.20, 0.70],
        ],
        dtype=float,
    )
    groups = np.array(["game-a", "game-b", "game-c"], dtype=object)
    prediction_at, feature_available_at = _pit_times(len(targets))

    report = evaluate_multiclass_probabilities(
        targets,
        probabilities,
        classes=CLASSES,
        groups=groups,
        bootstrap_samples=100,
        confidence_level=0.90,
        minimum_valid_samples=20,
        seed=7,
        prediction_at=prediction_at,
        feature_available_at=feature_available_at,
    )

    expected_brier = np.mean([0.14, 0.38, 0.14])
    expected_log_loss = -np.mean(np.log([0.70, 0.50, 0.70]))
    assert report["brier"] == pytest.approx(expected_brier)
    assert report["log_loss"] == pytest.approx(expected_log_loss)
    assert report["brier_definition"] == "mean_sum_squared_class_error"
    assert report["classes"] == CLASSES
    assert set(report["ovr_calibration"]) == set(CLASSES)
    assert all(
        set(values) == {"slope", "intercept"}
        for values in report["ovr_calibration"].values()
    )
    assert report["prior_comparison"] == {
        "available": False,
        "reason": "pit_prior_not_supplied",
    }


def test_multiclass_bootstrap_is_game_clustered_and_deterministic() -> None:
    targets, probabilities, groups = _balanced_inputs()

    first = _evaluate(targets, probabilities, groups)
    second = _evaluate(targets, probabilities, groups)

    assert first == second
    assert first["clusters"] == 8
    assert first["observations"] == 24
    assert first["bootstrap_samples_requested"] == 100
    assert 80 <= first["bootstrap_samples_valid"] <= 100
    assert set(first["bootstrap_ci"]) == {"brier", "log_loss"}
    assert set(first["ovr_calibration_bootstrap_ci"]) == set(CLASSES)
    assert all(
        set(intervals) == {"slope", "intercept"}
        for intervals in first["ovr_calibration_bootstrap_ci"].values()
    )


def test_multiclass_prior_delta_is_paired_within_the_same_game_resample() -> None:
    targets, model, groups = _balanced_inputs()
    prior = np.full_like(model, 1 / 3)
    prediction_at, _ = _pit_times(len(targets))

    report = _evaluate(
        targets,
        model,
        groups,
        prior_probabilities=prior,
        prior_available_at=prediction_at - pd.Timedelta(hours=1),
    )
    comparison = report["prior_comparison"]

    assert comparison["available"] is True
    assert comparison["delta_definition"] == "model_minus_prior"
    assert set(comparison["delta"]) == {"brier", "log_loss"}
    assert set(comparison["delta_bootstrap_ci"]) == {"brier", "log_loss"}
    assert comparison["delta"]["brier"] < 0
    assert comparison["delta"]["log_loss"] < 0
    assert comparison["delta_bootstrap_ci"]["brier"][1] < 0
    assert comparison["delta_bootstrap_ci"]["log_loss"][1] < 0
    assert comparison["bootstrap_samples_valid"] == report[
        "bootstrap_samples_valid"
    ]


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.7, 0.2, 0.2], [0.2, 0.6, 0.2]],
        [[0.7, np.nan, 0.3], [0.2, 0.6, 0.2]],
        [[0.7, -0.1, 0.4], [0.2, 0.6, 0.2]],
        [[0.7, 0.3], [0.2, 0.8]],
    ],
)
def test_multiclass_rejects_non_normalized_or_malformed_distributions(
    probabilities: list[list[float]],
) -> None:
    prediction_at, feature_available_at = _pit_times(2)

    with pytest.raises(ValidationInputError, match="probabilit|sum|class"):
        evaluate_multiclass_probabilities(
            ["home_win", "draw"],
            probabilities,
            classes=CLASSES,
            groups=["game-a", "game-b"],
            bootstrap_samples=20,
            confidence_level=0.90,
            minimum_valid_samples=20,
            seed=1,
            prediction_at=prediction_at,
            feature_available_at=feature_available_at,
        )


def test_multiclass_rejects_unknown_target_or_ambiguous_class_order() -> None:
    prediction_at, feature_available_at = _pit_times(2)

    with pytest.raises(ValidationInputError, match="target"):
        evaluate_multiclass_probabilities(
            ["home_win", "unknown"],
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]],
            classes=CLASSES,
            groups=["game-a", "game-b"],
            bootstrap_samples=20,
            confidence_level=0.90,
            minimum_valid_samples=20,
            seed=1,
            prediction_at=prediction_at,
            feature_available_at=feature_available_at,
        )

    with pytest.raises(ValidationInputError, match="classes"):
        evaluate_multiclass_probabilities(
            ["home_win", "draw"],
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]],
            classes=("home_win", "draw", "home_win"),
            groups=["game-a", "game-b"],
            bootstrap_samples=20,
            confidence_level=0.90,
            minimum_valid_samples=20,
            seed=1,
            prediction_at=prediction_at,
            feature_available_at=feature_available_at,
        )


def test_multiclass_rejects_future_feature_availability() -> None:
    targets, probabilities, groups = _balanced_inputs()
    prediction_at = pd.Series(
        pd.date_range("2026-01-01", periods=len(targets), freq="min", tz="UTC")
    )
    feature_available_at = prediction_at.copy()
    feature_available_at.iloc[7] = prediction_at.iloc[7] + pd.Timedelta(
        milliseconds=1
    )

    with pytest.raises(ValidationInputError, match="point-in-time"):
        _evaluate(
            targets,
            probabilities,
            groups,
            prediction_at=prediction_at,
            feature_available_at=feature_available_at,
        )


def test_multiclass_requires_prediction_and_feature_availability() -> None:
    targets, probabilities, groups = _balanced_inputs()

    with pytest.raises(TypeError, match="prediction_at.*feature_available_at"):
        evaluate_multiclass_probabilities(
            targets,
            probabilities,
            classes=CLASSES,
            groups=groups,
            bootstrap_samples=100,
            confidence_level=0.90,
            minimum_valid_samples=80,
            seed=20260722,
        )


def test_multiclass_requires_complete_utc_availability_pair() -> None:
    targets, probabilities, groups = _balanced_inputs()
    prediction_at = pd.Series(
        pd.date_range("2026-01-01", periods=len(targets), freq="min", tz="UTC")
    )

    with pytest.raises(ValidationInputError, match="provided together"):
        _evaluate(
            targets,
            probabilities,
            groups,
            prediction_at=prediction_at,
            feature_available_at=None,
        )


def test_multiclass_prior_requires_point_in_time_availability() -> None:
    targets, probabilities, groups = _balanced_inputs()
    prior = np.full_like(probabilities, 1 / 3)

    with pytest.raises(ValidationInputError, match="prior_available_at"):
        _evaluate(
            targets,
            probabilities,
            groups,
            prior_probabilities=prior,
        )


def test_multiclass_rejects_future_prior_availability() -> None:
    targets, probabilities, groups = _balanced_inputs()
    prior = np.full_like(probabilities, 1 / 3)
    prediction_at, _ = _pit_times(len(targets))
    prior_available_at = prediction_at.copy()
    prior_available_at.iloc[5] = prediction_at.iloc[5] + pd.Timedelta(
        milliseconds=1
    )

    with pytest.raises(ValidationInputError, match="prior.*point-in-time"):
        _evaluate(
            targets,
            probabilities,
            groups,
            prior_probabilities=prior,
            prior_available_at=prior_available_at,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prediction_at", lambda values: values.dt.tz_localize(None)),
        ("feature_available_at", lambda values: values.dt.tz_localize(None)),
        ("prediction_at", lambda values: values.mask(values.index == 3)),
        ("feature_available_at", lambda values: values.mask(values.index == 3)),
        ("prediction_at", lambda values: values.iloc[:-1]),
        ("feature_available_at", lambda values: values.iloc[:-1]),
    ],
)
def test_multiclass_rejects_invalid_pit_timestamp_series(
    field: str,
    replacement: Callable[[pd.Series], pd.Series],
) -> None:
    targets, probabilities, groups = _balanced_inputs()
    prediction_at, feature_available_at = _pit_times(len(targets))
    inputs = {
        "prediction_at": prediction_at,
        "feature_available_at": feature_available_at,
    }
    inputs[field] = replacement(inputs[field])

    with pytest.raises(
        ValidationInputError,
        match="availability length|timezone-aware UTC",
    ):
        _evaluate(
            targets,
            probabilities,
            groups,
            **inputs,
        )


@pytest.mark.parametrize(
    "replacement",
    [
        lambda values: values.dt.tz_localize(None),
        lambda values: values.mask(values.index == 3),
        lambda values: values.iloc[:-1],
    ],
)
def test_multiclass_rejects_invalid_prior_timestamp_series(
    replacement: Callable[[pd.Series], pd.Series],
) -> None:
    targets, probabilities, groups = _balanced_inputs()
    prior = np.full_like(probabilities, 1 / 3)
    prediction_at, _ = _pit_times(len(targets))
    prior_available_at = replacement(prediction_at - pd.Timedelta(hours=1))

    with pytest.raises(
        ValidationInputError,
        match="prior.*length|prior_available_at.*timezone-aware UTC",
    ):
        _evaluate(
            targets,
            probabilities,
            groups,
            prior_probabilities=prior,
            prior_available_at=prior_available_at,
        )


def test_multiclass_rejects_missing_or_single_game_clusters() -> None:
    targets, probabilities, groups = _balanced_inputs()
    groups[0] = ""
    with pytest.raises(ValidationInputError, match="groups must be present"):
        _evaluate(targets, probabilities, groups)

    with pytest.raises(ValidationInputError, match="at least two groups"):
        _evaluate(
            targets,
            probabilities,
            np.array(["only-game"] * len(targets), dtype=object),
        )
