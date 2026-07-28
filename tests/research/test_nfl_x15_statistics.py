from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_statistics import (
    X15StatisticsInputError,
    summarize_development_signals,
)


def _signal_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "episode_id": "ep-1",
                "factor_id": "NFL.X15.PRIMARY.INTERCEPTION",
                "venue": "kalshi",
                "model_id": "model-a",
                "gross_markout": 0.10,
                "evaluation_status": "EVALUATED",
                "dataset_partition": "development",
            },
            {
                "game_id": "game-1",
                "episode_id": "ep-2",
                "factor_id": "NFL.X15.PRIMARY.INTERCEPTION",
                "venue": "kalshi",
                "model_id": "model-a",
                "gross_markout": 0.20,
                "evaluation_status": "EVALUATED",
                "dataset_partition": "development",
            },
            {
                "game_id": "game-2",
                "episode_id": "ep-3",
                "factor_id": "NFL.X15.PRIMARY.INTERCEPTION",
                "venue": "kalshi",
                "model_id": "model-a",
                "gross_markout": 0.30,
                "evaluation_status": "EVALUATED",
                "dataset_partition": "development",
            },
            {
                "game_id": "game-3",
                "episode_id": "ep-4",
                "factor_id": "NFL.X15.PRIMARY.INTERCEPTION",
                "venue": "kalshi",
                "model_id": "model-a",
                "gross_markout": -0.10,
                "evaluation_status": "EVALUATED",
                "dataset_partition": "development",
            },
        ]
    )


def test_reports_game_clustered_effect_stability_and_support() -> None:
    result = summarize_development_signals(
        _signal_rows(),
        seed=20260728,
        n_bootstrap=500,
        min_support_games=3,
        min_support_signals=4,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["factor_id"] == "NFL.X15.PRIMARY.INTERCEPTION"
    assert row["venue"] == "kalshi"
    assert row["model_id"] == "model-a"
    assert row["support_games"] == 3
    assert row["support_signals"] == 4
    assert row["mean_effect"] == pytest.approx(0.125)
    assert row["median_effect"] == pytest.approx(0.15)
    assert row["mean_ci_lower"] <= row["mean_ci_upper"]
    assert row["median_ci_lower"] <= row["median_ci_upper"]
    assert row["leave_one_game_out_same_sign_rate"] == pytest.approx(1.0)
    assert row["max_single_game_absolute_contribution_ratio"] == pytest.approx(
        3 / 7
    )
    assert 0 < row["cluster_sign_flip_p_value"] <= 1
    assert row["bootstrap_samples"] == 500
    assert row["bootstrap_seed"] == 20260728
    assert bool(row["recommended_for_gating"]) is True
    assert row["support_status"] == "SUPPORTED_FOR_GATING"
    assert bool(row["passes_development_gate"]) is False


def test_fixed_seed_makes_bootstrap_and_permutation_reproducible() -> None:
    first = summarize_development_signals(
        _signal_rows(), seed=17, n_bootstrap=200
    )
    second = summarize_development_signals(
        _signal_rows().sample(frac=1, random_state=9),
        seed=17,
        n_bootstrap=200,
    )

    pd.testing.assert_frame_equal(first, second)


def test_bh_q_values_are_adjusted_within_factor_family_only() -> None:
    base = _signal_rows()
    same_factor_second_model = base.assign(
        model_id="model-b",
        gross_markout=base["gross_markout"] * 0.4,
    )
    other_factor = base.assign(
        factor_id="NFL.X15.PRIMARY.LOST_FUMBLE",
        model_id="model-c",
    )

    result = summarize_development_signals(
        pd.concat(
            [base, same_factor_second_model, other_factor],
            ignore_index=True,
        ),
        seed=31,
        n_bootstrap=300,
    )

    interceptions = result.loc[
        result["factor_id"].eq("NFL.X15.PRIMARY.INTERCEPTION")
    ].sort_values("cluster_sign_flip_p_value")
    assert set(interceptions["factor_family_hypothesis_count"]) == {3}
    assert (interceptions["bh_q_value"] >= interceptions[
        "cluster_sign_flip_p_value"
    ]).all()

    fumble = result.loc[
        result["factor_id"].eq("NFL.X15.PRIMARY.LOST_FUMBLE")
    ].iloc[0]
    assert fumble["factor_family_hypothesis_count"] == 3


def test_low_support_is_not_recommended_for_gating() -> None:
    result = summarize_development_signals(
        _signal_rows().iloc[:2],
        seed=1,
        n_bootstrap=100,
        min_support_games=3,
        min_support_signals=3,
    )

    row = result.iloc[0]
    assert row["support_games"] == 1
    assert row["support_signals"] == 2
    assert bool(row["recommended_for_gating"]) is False
    assert row["support_status"] == "LOW_SUPPORT_NOT_RECOMMENDED"


def test_development_gate_requires_supported_same_sign_effect_on_both_venues() -> None:
    rows = []
    for venue in ("kalshi", "polymarket"):
        for game_number in range(1, 9):
            rows.append(
                {
                    "game_id": f"game-{game_number}",
                    "episode_id": f"{venue}-ep-{game_number}",
                    "factor_id": "NFL.X15.PRIMARY.TURNOVER_ON_DOWNS",
                    "venue": venue,
                    "model_id": "model-a",
                    "gross_markout": 0.08,
                    "evaluation_status": "EVALUATED",
                    "dataset_partition": "development",
                }
            )

    result = summarize_development_signals(
        pd.DataFrame(rows),
        seed=13,
        n_bootstrap=2_000,
        min_support_games=6,
        min_support_signals=6,
    )

    assert result["cross_venue_same_sign"].all()
    assert result["mean_ci_excludes_zero"].all()
    assert result["passes_development_gate"].all()


def test_cross_venue_gate_requires_each_venue_to_pass_individual_statistics() -> None:
    rows = []
    for game_number in range(1, 9):
        rows.append(
            {
                "game_id": f"game-{game_number}",
                "episode_id": f"kalshi-ep-{game_number}",
                "factor_id": "NFL.X15.PRIMARY.PASS",
                "venue": "kalshi",
                "model_id": "model-a",
                "gross_markout": 0.08,
                "evaluation_status": "EVALUATED",
                "dataset_partition": "development",
            }
        )
        rows.append(
            {
                "game_id": f"game-{game_number}",
                "episode_id": f"polymarket-ep-{game_number}",
                "factor_id": "NFL.X15.PRIMARY.PASS",
                "venue": "polymarket",
                "model_id": "model-a",
                "gross_markout": 0.0 if game_number % 2 else 0.01,
                "evaluation_status": "EVALUATED",
                "dataset_partition": "development",
            }
        )

    result = summarize_development_signals(
        pd.DataFrame(rows),
        seed=13,
        n_bootstrap=2_000,
        min_support_games=6,
        min_support_signals=6,
    )

    assert result["recommended_for_gating"].all()
    assert not result["individual_statistical_gate"].all()
    assert not result["cross_venue_same_sign"].any()
    assert not result["passes_development_gate"].any()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(dataset_partition="holdout"),
            "development",
        ),
        (
            lambda frame: frame.assign(
                evaluation_status="CENSORED_EVALUATION",
                gross_markout=float("nan"),
            ),
            "EVALUATED",
        ),
        (
            lambda frame: frame.assign(gross_markout=float("nan")),
            "finite",
        ),
        (
            lambda frame: pd.concat(
                [frame, frame.iloc[[0]]], ignore_index=True
            ),
            "duplicate",
        ),
    ],
)
def test_rejects_rows_that_cannot_enter_formal_effect_statistics(
    mutate,
    message: str,
) -> None:
    with pytest.raises(X15StatisticsInputError, match=message):
        summarize_development_signals(
            mutate(_signal_rows()),
            seed=5,
            n_bootstrap=20,
        )
