from __future__ import annotations

import math

import pandas as pd
import pytest

from prediction_market.sports.nfl_x13_dense_factor_summary import (
    DenseFactorSummaryError,
    build_dense_factor_summary,
)


LEGACY_HORIZONS = (1, 2, 5, 10, 20, 30, 60)


def _factor_rows(
    *,
    game_id: str,
    episode_id: str,
    predicate_nonexcluded_hit: bool = True,
    factor_eligible: bool = True,
) -> list[dict[str, object]]:
    return [
        {
            "factor_id": "NFL.TEST.EVENT",
            "factor_version": "v1",
            "game_id": game_id,
            "episode_id": episode_id,
            "logical_market_id": "kalshi:game:winner",
            "outcome": "AAA",
            "venue": "kalshi",
            "horizon_seconds": horizon,
            "predicate_nonexcluded_hit": predicate_nonexcluded_hit,
            "factor_eligible": factor_eligible,
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        }
        for horizon in LEGACY_HORIZONS
    ]


def _dense_row(
    *,
    game_id: str,
    episode_id: str,
    horizon_seconds: int,
    effect: float,
    shape: str,
    forward_filled: bool = False,
) -> dict[str, object]:
    return {
        "observation_id": f"{game_id}:{episode_id}:{horizon_seconds}",
        "game_id": game_id,
        "episode_id": episode_id,
        "logical_market_id": "kalshi:game:winner",
        "outcome": "AAA",
        "venue": "kalshi",
        "market_family": "winner",
        "horizon_seconds": horizon_seconds,
        "cumulative_change": effect,
        "observed": True,
        "forward_filled": forward_filled,
        "contaminated": False,
        "order_ambiguous": False,
        "censored": False,
        "shape_classification": shape,
        "attrition_reason": "OBSERVED",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
    }


def test_dense_factor_summary_has_per_game_counts_quantiles_and_shape_shares() -> None:
    factors = pd.DataFrame(
        [
            *_factor_rows(game_id="game-1", episode_id="episode-1"),
            *_factor_rows(game_id="game-2", episode_id="episode-2"),
        ]
    )
    paths = pd.DataFrame(
        [
            _dense_row(
                game_id="game-1",
                episode_id="episode-1",
                horizon_seconds=1,
                effect=0.10,
                shape="JUMP_DOMINANT",
            ),
            _dense_row(
                game_id="game-2",
                episode_id="episode-2",
                horizon_seconds=1,
                effect=0.30,
                shape="GRADUAL",
            ),
        ]
    )

    result = build_dense_factor_summary(factors, paths)

    summary = result.summary.iloc[0]
    assert summary["observed_n"] == 2
    assert summary["game_count"] == 2
    assert summary["mean"] == pytest.approx(0.20)
    assert summary["median"] == pytest.approx(0.20)
    assert summary["std"] == pytest.approx(math.sqrt(0.02))
    assert summary["p10"] == pytest.approx(0.12)
    assert summary["p25"] == pytest.approx(0.15)
    assert summary["p75"] == pytest.approx(0.25)
    assert summary["p90"] == pytest.approx(0.28)
    assert summary["positive_rate"] == 1.0
    assert summary["negative_rate"] == 0.0
    assert summary["abs_mean"] == pytest.approx(0.20)
    assert summary["jump_dominant_share"] == 0.5
    assert summary["gradual_share"] == 0.5
    assert summary["mixed_share"] == 0.0
    assert summary["unidentified_sparse_share"] == 0.0
    assert summary["claim_boundary"] == "PRELIMINARY_SOURCE_TIME_ONLY"


def test_dense_factor_summary_excludes_forward_filled_path_and_retains_attrition() -> None:
    factors = pd.DataFrame(_factor_rows(game_id="game-1", episode_id="episode-1"))
    paths = pd.DataFrame(
        [
            _dense_row(
                game_id="game-1",
                episode_id="episode-1",
                horizon_seconds=1,
                effect=0.10,
                shape="JUMP_DOMINANT",
                forward_filled=True,
            )
        ]
    )

    result = build_dense_factor_summary(factors, paths)

    assert result.summary.empty
    assert result.attrition.iloc[0]["attrition_reason"] == "FORWARD_FILLED"
    assert result.attrition.iloc[0]["eligible"] == False


def test_dense_factor_summary_rejects_membership_that_changes_across_legacy_horizons() -> None:
    rows = _factor_rows(game_id="game-1", episode_id="episode-1")
    rows[1]["predicate_nonexcluded_hit"] = False
    factors = pd.DataFrame(rows)
    paths = pd.DataFrame(
        [
            _dense_row(
                game_id="game-1",
                episode_id="episode-1",
                horizon_seconds=1,
                effect=0.10,
                shape="JUMP_DOMINANT",
            )
        ]
    )

    with pytest.raises(DenseFactorSummaryError, match="membership differs"):
        build_dense_factor_summary(factors, paths)


def test_dense_factor_summary_accepts_a_factor_with_its_registered_horizon_subset() -> None:
    """Definitions may register a subset of the shared legacy horizon grid."""

    factors = pd.DataFrame(
        [
            row
            for row in _factor_rows(game_id="game-1", episode_id="episode-1")
            if row["horizon_seconds"] in {5, 30}
        ]
    )
    paths = pd.DataFrame(
        [
            _dense_row(
                game_id="game-1",
                episode_id="episode-1",
                horizon_seconds=5,
                effect=0.10,
                shape="JUMP_DOMINANT",
            )
        ]
    )

    result = build_dense_factor_summary(factors, paths)

    assert result.summary.iloc[0]["horizon_seconds"] == 5
