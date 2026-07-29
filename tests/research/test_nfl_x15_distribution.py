from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_market.research.nfl_x15_distribution import (
    EXTREME_QUANTILES,
    INSUFFICIENT_SUPPORT,
    PRIMARY_QUANTILES,
    QuantileSupportContract,
    enforce_non_crossing,
    fit_directional_quantiles,
    magnitude_distribution_metrics,
)


def test_non_crossing_projection_is_monotone_and_nonnegative() -> None:
    crossed = np.array(
        [
            [0.04, 0.03, -0.01, 0.08, 0.07],
            [0.01, 0.02, 0.03, 0.04, 0.05],
        ]
    )

    projected = enforce_non_crossing(crossed)

    assert np.all(projected >= 0)
    assert np.all(np.diff(projected, axis=1) >= 0)


def test_quantiles_are_fit_only_for_up_and_down_with_a_frozen_extreme_gate() -> None:
    rows = 12
    features = pd.DataFrame(
        {
            "market": np.linspace(0.1, 0.9, rows * 2),
            "state": np.tile([0.0, 1.0], rows),
        }
    )
    directions = np.repeat(["DOWN", "UP"], rows)
    magnitude = np.concatenate(
        [np.linspace(0.01, 0.12, rows), np.linspace(0.02, 0.13, rows)]
    )
    games = np.array(
        [f"down-{index // 2}" for index in range(rows)]
        + [f"up-{index // 2}" for index in range(rows)]
    )
    contract = QuantileSupportContract(
        primary_min_rows=8,
        primary_min_games=4,
        extreme_min_rows=100,
        extreme_min_games=50,
    )

    fitted = fit_directional_quantiles(
        features,
        magnitude,
        directions,
        games,
        support_contract=contract,
        random_state=7,
    )
    predicted = fitted.predict(
        pd.DataFrame({"market": [0.2, 0.8], "state": [0.0, 1.0]}),
        direction="UP",
    )

    assert fitted.support["UP"].primary_status == "SUPPORTED"
    assert fitted.support["UP"].extreme_status == INSUFFICIENT_SUPPORT
    assert tuple(fitted.quantiles) == PRIMARY_QUANTILES
    assert tuple(EXTREME_QUANTILES) == (0.05, 0.95)
    assert predicted.columns.tolist() == [
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
    ]
    assert np.all(np.diff(predicted.to_numpy(), axis=1) >= 0)

    not_applicable = fitted.predict(features.iloc[:2], direction="NO_MOVE")
    assert not_applicable.isna().all(axis=None)


def test_unsupported_direction_returns_a_support_artifact_not_fake_quantiles() -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    fitted = fit_directional_quantiles(
        features,
        np.array([0.01, 0.02, 0.03, 0.04]),
        np.array(["UP", "UP", "UP", "UP"]),
        np.array(["g1", "g2", "g3", "g4"]),
        support_contract=QuantileSupportContract(
            primary_min_rows=4,
            primary_min_games=4,
            extreme_min_rows=8,
            extreme_min_games=8,
        ),
    )

    assert fitted.support["DOWN"].primary_status == INSUFFICIENT_SUPPORT
    assert fitted.predict(features, direction="DOWN").isna().all(axis=None)


def test_extreme_quantiles_publish_only_after_both_directions_pass_gate() -> None:
    rows = 4
    features = pd.DataFrame(
        {"x": np.linspace(0, 1, rows * 2), "z": np.tile([0.0, 1.0], rows)}
    )
    fitted = fit_directional_quantiles(
        features,
        np.linspace(0.01, 0.08, rows * 2),
        np.repeat(["DOWN", "UP"], rows),
        np.array(
            [f"down-{index}" for index in range(rows)]
            + [f"up-{index}" for index in range(rows)]
        ),
        support_contract=QuantileSupportContract(
            primary_min_rows=4,
            primary_min_games=4,
            extreme_min_rows=4,
            extreme_min_games=4,
        ),
    )

    assert fitted.support["DOWN"].extreme_status == "SUPPORTED"
    assert fitted.support["UP"].extreme_status == "SUPPORTED"
    assert fitted.predict(features.iloc[:2], direction="UP").columns.tolist() == [
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
    ]


def test_magnitude_metrics_include_pinball_crps_coverage_and_width() -> None:
    truth = np.array([0.02, 0.05])
    predictions = pd.DataFrame(
        {
            "q10": [0.01, 0.03],
            "q25": [0.015, 0.04],
            "q50": [0.02, 0.05],
            "q75": [0.025, 0.06],
            "q90": [0.03, 0.07],
        }
    )

    metrics = magnitude_distribution_metrics(truth, predictions)

    assert metrics["pinball_q50"] == pytest.approx(0)
    assert metrics["approximate_crps"] >= 0
    assert metrics["interval_80_coverage"] == pytest.approx(1)
    assert metrics["interval_80_width"] == pytest.approx(0.03)


def test_quantile_training_hash_is_shuffle_invariant() -> None:
    features = pd.DataFrame(
        {"x": np.linspace(0, 1, 8), "z": np.tile([0.0, 1.0], 4)}
    )
    magnitude = np.linspace(0.01, 0.08, 8)
    directions = np.array(["UP"] * 4 + ["DOWN"] * 4)
    games = np.array(["u4", "u2", "u1", "u3", "d3", "d1", "d4", "d2"])
    weights = np.linspace(0.5, 1.2, 8)
    contract = QuantileSupportContract(
        primary_min_rows=4,
        primary_min_games=4,
        extreme_min_rows=8,
        extreme_min_games=8,
    )
    permutation = np.array([6, 2, 7, 1, 5, 0, 4, 3])

    first = fit_directional_quantiles(
        features,
        magnitude,
        directions,
        games,
        sample_weight=weights,
        support_contract=contract,
    )
    second = fit_directional_quantiles(
        features.iloc[permutation].reset_index(drop=True),
        magnitude[permutation],
        directions[permutation],
        games[permutation],
        sample_weight=weights[permutation],
        support_contract=contract,
    )

    assert first.support["UP"].training_sha256 == second.support[
        "UP"
    ].training_sha256
    assert first.support["DOWN"].training_sha256 == second.support[
        "DOWN"
    ].training_sha256
