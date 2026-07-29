from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_model_selection import (
    ModelSelectionError,
    paired_game_model_selection,
)


AUTHORITY = "sha256:" + "a" * 64


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game in range(4):
        for episode, outcome in (("e1", 1), ("e2", 0)):
            for model_id, probability in (("B0", 0.50), ("B1", 0.80 if outcome else 0.20)):
                rows.append(
                    {
                        "game_id": f"game-{game}",
                        "atomic_information_episode_id": episode,
                        "venue": "polymarket",
                        "actual_home_contract_id": f"home-{game}",
                        "landmark_seconds": 5,
                        "endpoint_seconds": 30,
                        "head": "survival",
                        "nfl_week": game + 1,
                        "cohort": "development",
                        "authority_sha256": AUTHORITY,
                        "model_id": model_id,
                        "probability": probability,
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows)


def test_same_exact_pairs_are_scored_per_game_then_bootstrapped() -> None:
    result = paired_game_model_selection(
        _predictions(),
        candidate_model_id="B1",
        authority_sha256=AUTHORITY,
        n_bootstrap=10_000,
        seed=7,
    )

    assert len(result.game_losses) == 4
    assert set(result.game_losses["game_id"]) == {f"game-{index}" for index in range(4)}
    assert result.bootstrap_samples == 10_000
    assert result.mean_log_loss_improvement > 0
    assert result.bootstrap_ci_low > 0
    assert result.point_gate_passed is True
    assert result.bootstrap_gate_passed is True
    assert result.selected is True


def test_selection_rejects_unpaired_or_non_development_rows() -> None:
    unpaired = _predictions().iloc[:-1]
    with pytest.raises(ModelSelectionError, match="exact pairs"):
        paired_game_model_selection(
            unpaired,
            candidate_model_id="B1",
            authority_sha256=AUTHORITY,
        )

    holdout = _predictions().copy()
    holdout.loc[0, ["cohort", "nfl_week"]] = ["holdout", 13]
    with pytest.raises(ModelSelectionError, match="development"):
        paired_game_model_selection(
            holdout,
            candidate_model_id="B1",
            authority_sha256=AUTHORITY,
        )


def test_factor_membership_cannot_enter_selection_score() -> None:
    predictions = _predictions().copy()
    predictions["factor_id"] = "NFL.UNRELATED"
    with pytest.raises(ModelSelectionError, match="factor membership"):
        paired_game_model_selection(
            predictions,
            candidate_model_id="B1",
            authority_sha256=AUTHORITY,
        )
