from __future__ import annotations

import numpy as np
import pytest

from prediction_market.research.nfl_x15_calibration import (
    CALIBRATED,
    INSUFFICIENT_SUPPORT,
    RAW_UNCALIBRATED,
    fit_binary_platt,
    fit_direction_temperature,
)


def test_platt_uses_only_supplied_prequential_training_games() -> None:
    raw = np.array([0.08, 0.18, 0.35, 0.62, 0.78, 0.91])
    truth = np.array([0, 0, 0, 1, 1, 1])
    games = np.array([f"g{index}" for index in range(6)])

    first = fit_binary_platt(raw, truth, games)
    second = fit_binary_platt(raw, truth, games)

    assert first.status == CALIBRATED
    assert first.fit_game_ids == tuple(games)
    assert first.training_sha256 == second.training_sha256
    assert first.training_sha256.startswith("sha256:")
    transformed = first.transform(np.array([0.2, 0.5, 0.8]))
    assert np.all((transformed > 0) & (transformed < 1))


def test_binary_single_class_is_an_explicit_raw_uncalibrated_artifact() -> None:
    raw = np.array([0.2, 0.3, 0.4, 0.5])
    calibrator = fit_binary_platt(
        raw,
        np.ones(4, dtype=int),
        np.array(["g1", "g2", "g3", "g4"]),
    )

    assert calibrator.status == RAW_UNCALIBRATED
    assert calibrator.support_reason == INSUFFICIENT_SUPPORT
    assert calibrator.slope is None
    assert calibrator.intercept is None
    assert calibrator.transform(raw) == pytest.approx(raw)


def test_direction_temperature_is_deterministic_and_normalized() -> None:
    raw = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.65, 0.25, 0.10],
            [0.10, 0.80, 0.10],
            [0.15, 0.65, 0.20],
            [0.05, 0.15, 0.80],
            [0.10, 0.20, 0.70],
        ]
    )
    truth = np.array(["DOWN", "DOWN", "NO_MOVE", "NO_MOVE", "UP", "UP"])
    games = np.array([f"g{index}" for index in range(6)])

    first = fit_direction_temperature(raw, truth, games)
    second = fit_direction_temperature(raw, truth, games)

    assert first.status == CALIBRATED
    assert first.temperature == second.temperature
    assert first.training_sha256 == second.training_sha256
    calibrated = first.transform(raw)
    assert calibrated.sum(axis=1) == pytest.approx(np.ones(len(raw)))
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_direction_temperature_requires_all_three_training_classes() -> None:
    raw = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
    )
    calibrator = fit_direction_temperature(
        raw,
        np.array(["DOWN", "DOWN", "UP", "UP"]),
        np.array(["g1", "g2", "g3", "g4"]),
    )

    assert calibrator.status == RAW_UNCALIBRATED
    assert calibrator.support_reason == INSUFFICIENT_SUPPORT
    assert calibrator.temperature is None
    assert calibrator.transform(raw) == pytest.approx(raw)


def test_calibration_hashes_and_fit_games_do_not_depend_on_row_order() -> None:
    raw_binary = np.array([0.08, 0.18, 0.35, 0.62, 0.78, 0.91])
    binary_truth = np.array([0, 0, 0, 1, 1, 1])
    raw_direction = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.2, 0.7],
        ]
    )
    direction_truth = np.array(
        ["DOWN", "DOWN", "NO_MOVE", "NO_MOVE", "UP", "UP"]
    )
    games = np.array(["g3", "g1", "g6", "g2", "g5", "g4"])
    weights = np.array([1.0, 0.5, 0.7, 1.3, 0.8, 1.1])
    permutation = np.array([4, 1, 5, 0, 3, 2])

    binary_first = fit_binary_platt(
        raw_binary, binary_truth, games, sample_weight=weights
    )
    binary_second = fit_binary_platt(
        raw_binary[permutation],
        binary_truth[permutation],
        games[permutation],
        sample_weight=weights[permutation],
    )
    direction_first = fit_direction_temperature(
        raw_direction, direction_truth, games, sample_weight=weights
    )
    direction_second = fit_direction_temperature(
        raw_direction[permutation],
        direction_truth[permutation],
        games[permutation],
        sample_weight=weights[permutation],
    )

    assert binary_first.training_sha256 == binary_second.training_sha256
    assert binary_first.fit_game_ids == binary_second.fit_game_ids == tuple(
        sorted(games)
    )
    assert binary_first.slope == pytest.approx(binary_second.slope)
    assert direction_first.training_sha256 == direction_second.training_sha256
    assert direction_first.fit_game_ids == direction_second.fit_game_ids
    assert direction_first.temperature == direction_second.temperature
