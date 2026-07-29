from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_model_selection import (
    FrozenSelectionSpec,
    ModelSelectionError,
    build_factor_claim_audit,
    select_candidate_against_b0,
)
from prediction_market.research.nfl_x15_models import X15ModelRun


AUTHORITY = "sha256:" + "a" * 64
TARGET_CONTRACT = "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
CLAIM_BOUNDARY = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)


def _run_config() -> dict[str, object]:
    return {
        "schema_version": "HistoricalTradesOnlyProbabilityPanelV1",
        "target_contract": TARGET_CONTRACT,
        "claim_boundary": CLAIM_BOUNDARY,
        "analysis_scope": (
            "HISTORICAL_TRADES_ONLY_SOURCE_TIME_DIAGNOSTIC"
        ),
        "direction_threshold_probability": 0.01,
        "direction_threshold_semantics": (
            "FIXED_CROSS_VENUE_RESEARCH_MATERIALITY_NOT_TICK"
        ),
        "venue_tick_support": "UNSUPPORTED",
        "market_continuity_support": "UNKNOWN",
        "claim_eligible": False,
        "cohort_authority_sha256": AUTHORITY,
        "feature_blocks": {
            "D0": (),
            "D1": (),
            "D2": (),
            "D3": (),
            "D4": (),
        },
        "transport_pairs": (),
    }


def _task4_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_row_id = 0
    for game_index in range(4):
        for episode_index in range(2):
            truth = bool((game_index + episode_index) % 2)
            direction = ("UP", "DOWN", "NO_MOVE")[
                (game_index + episode_index) % 3
            ]
            for landmark, endpoint in ((3, 30), (5, 60)):
                for model_id, block_id, good in (
                    ("b0_empirical_v1", "D0", False),
                    ("regularized_logistic_v1", "D4", True),
                ):
                    probability = 0.82 if truth else 0.18
                    if not good:
                        probability = 0.56 if truth else 0.44
                    direction_probabilities = {
                        "DOWN": [0.12, 0.16, 0.72],
                        "NO_MOVE": [0.16, 0.72, 0.12],
                        "UP": [0.72, 0.16, 0.12],
                    }[direction]
                    if good:
                        direction_probabilities = {
                            "DOWN": [0.84, 0.08, 0.08],
                            "NO_MOVE": [0.08, 0.84, 0.08],
                            "UP": [0.08, 0.08, 0.84],
                        }[direction]
                    rows.append(
                        {
                            "source_row_id": source_row_id,
                            "cohort_authority_sha256": AUTHORITY,
                            "game_id": f"game-{game_index}",
                            "nfl_week": game_index + 3,
                            "atomic_information_episode_id": (
                                f"game-{game_index}:episode-{episode_index}"
                            ),
                            "venue": "polymarket",
                            "training_venue": "polymarket",
                            "calibration_venue": "polymarket",
                            "transport_mode": "VENUE_SPECIFIC",
                            "actual_home_contract_id": f"poly-{game_index}",
                            "landmark_seconds": landmark,
                            "endpoint_seconds": endpoint,
                            "fold_id": "fold_01",
                            "model_id": model_id,
                            "feature_block_id": block_id,
                            "target_contract": TARGET_CONTRACT,
                            "claim_boundary": CLAIM_BOUNDARY,
                            "schema_version": (
                                "HistoricalTradesOnlyProbabilityPanelV1"
                            ),
                            "analysis_scope": (
                                "HISTORICAL_TRADES_ONLY_SOURCE_TIME_"
                                "DIAGNOSTIC"
                            ),
                            "claim_eligible": False,
                            "direction_threshold_probability": 0.01,
                            "direction_threshold_semantics": (
                                "FIXED_CROSS_VENUE_RESEARCH_"
                                "MATERIALITY_NOT_TICK"
                            ),
                            "venue_tick_support": "UNSUPPORTED",
                            "market_continuity_support": "UNKNOWN",
                            "s_h_truth": truth,
                            "o_h_given_s_truth": truth,
                            "direction_truth": direction,
                            "s_h_calibrated_probability": probability,
                            "o_h_given_s_calibrated_probability": probability,
                            "direction_calibrated_prob_down": (
                                direction_probabilities[0]
                            ),
                            "direction_calibrated_prob_no_move": (
                                direction_probabilities[1]
                            ),
                            "direction_calibrated_prob_up": (
                                direction_probabilities[2]
                            ),
                        }
                    )
                source_row_id += 1
    return pd.DataFrame(rows)


def _run() -> X15ModelRun:
    return X15ModelRun(
        oof_predictions=_task4_predictions(),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256="sha256:" + "c" * 64,
        run_config=_run_config(),
    )


def _spec() -> FrozenSelectionSpec:
    return FrozenSelectionSpec(
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D4",
    )


def test_actual_task4_wide_schema_runs_multihead_multihorizon_dual_gate() -> None:
    result = select_candidate_against_b0(_run(), spec=_spec())

    assert result.bootstrap_samples == 10_000
    assert result.integrated_mean_improvement > 0
    assert result.integrated_ci_low > 0
    assert result.integrated_gate_passed is True
    assert result.anchor_mean_improvement >= 0
    assert result.anchor_sign_reversed is False
    assert result.anchor_gate_passed is True
    assert result.selected is True
    assert result.cohort_authority_sha256 == AUTHORITY
    assert result.target_contract == TARGET_CONTRACT
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.schema_version == (
        "HistoricalTradesOnlyProbabilityPanelV1"
    )
    assert result.analysis_scope == (
        "HISTORICAL_TRADES_ONLY_SOURCE_TIME_DIAGNOSTIC"
    )
    assert result.diagnostic_status == "HISTORICAL_SIGNAL_CANDIDATE"
    assert result.execution_claim_eligible is False
    assert result.tick_claim_eligible is False
    assert result.continuity_claim_eligible is False
    assert result.game_losses["hierarchical_weight"].eq(1.0).all()
    assert set(result.paired_rows["landmark_seconds"]) == {3, 5}
    assert set(result.paired_rows["endpoint_seconds"]) == {30, 60}


def test_exact_candidate_b0_pairs_and_frozen_anchor_are_mandatory() -> None:
    predictions = _task4_predictions()
    missing = predictions.drop(
        predictions[
            predictions["model_id"].eq("regularized_logistic_v1")
        ].index[0]
    )
    run = _run()
    run = X15ModelRun(
        oof_predictions=missing,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(ModelSelectionError, match="exact candidate/B0 pairs"):
        select_candidate_against_b0(run, spec=_spec())

    without_anchor = _task4_predictions().loc[
        lambda frame: ~(
            frame["landmark_seconds"].eq(3)
            & frame["endpoint_seconds"].eq(30)
        )
    ]
    run = X15ModelRun(
        oof_predictions=without_anchor,
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256="sha256:" + "c" * 64,
        run_config=_run_config(),
    )
    with pytest.raises(ModelSelectionError, match="L=3.*H=30"):
        select_candidate_against_b0(run, spec=_spec())


def test_factor_membership_is_a_claim_only_bh_loo_support_surface() -> None:
    result = select_candidate_against_b0(_run(), spec=_spec())
    membership = result.paired_rows[
        ["game_id", "atomic_information_episode_id"]
    ].drop_duplicates()
    membership["factor_id"] = [
        "NFL.PASS.COMPLETE" if index % 2 else "NFL.PASS.INCOMPLETE"
        for index in range(len(membership))
    ]
    membership["factor_version"] = "v1"

    audit = build_factor_claim_audit(
        result,
        factor_membership=membership,
        min_support_games=2,
        min_support_episodes=2,
    )

    required = {
        "factor_family",
        "bh_q_value",
        "leave_one_game_out_same_sign_rate",
        "max_single_game_absolute_contribution_ratio",
        "support_games",
        "support_episodes",
        "support_status",
    }
    assert required.issubset(audit.columns)
    assert set(audit["factor_family"]) == {"NFL.PASS"}


def test_diagnostic_block_and_nonexecution_contract_fail_closed() -> None:
    with pytest.raises(ModelSelectionError, match="D1..D4"):
        FrozenSelectionSpec(
            candidate_model_id="regularized_logistic_v1",
            candidate_feature_block_id="B4",
        )

    predictions = _task4_predictions()
    predictions.loc[
        predictions["model_id"].eq("regularized_logistic_v1"),
        "claim_eligible",
    ] = True
    base = _run()
    run = X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=base.conditional_quantiles,
        fold_metrics=base.fold_metrics,
        support_audit=base.support_audit,
        weight_audit=base.weight_audit,
        run_config_sha256=base.run_config_sha256,
        run_config=base.run_config,
    )
    with pytest.raises(ModelSelectionError, match="claim_eligible=False"):
        select_candidate_against_b0(run, spec=_spec())
