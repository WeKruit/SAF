from __future__ import annotations

import math

import pandas as pd
import pytest

from prediction_market.research import nfl_x15_model_selection as selection_module
from prediction_market.research.nfl_x15_model_selection import (
    FrozenSelectionSpec,
    ModelSelectionError,
    bind_frozen_development_authority,
    build_factor_claim_audit,
    select_candidate_against_b0,
)
from prediction_market.research.nfl_x15_models import X15ModelRun


AUTHORITY = "sha256:" + "a" * 64
TARGET_CONTRACT = "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
CLAIM_BOUNDARY = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)
FOLDS = (
    ("fold_01", (1, 2), (3, 4)),
    ("fold_02", (1, 2, 3, 4), (5, 6)),
    ("fold_03", tuple(range(1, 7)), (7, 8)),
    ("fold_04", tuple(range(1, 9)), (9, 10)),
    ("fold_05", tuple(range(1, 11)), (11, 12)),
)


def _development_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": f"game-{index:03d}",
                "nfl_week": index % 12 + 1,
                "kickoff_utc": (
                    f"2025-08-{index % 28 + 1:02d}T12:00:00Z"
                ),
                "batch_sha256": "sha256:" + f"{index + 1:064x}",
                "cohort_authority_sha256": AUTHORITY,
            }
            for index in range(153)
        ]
    )


def _authority():
    return bind_frozen_development_authority(
        _development_metadata(),
        cohort_authority_sha256=AUTHORITY,
    )


def _run_config(*, fold_ids: tuple[str, ...]) -> dict[str, object]:
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
        "fold_ids": fold_ids,
        "transport_pairs": (),
    }


def _fold_for_week(week: int):
    return next(fold for fold in FOLDS if week in fold[2])


def _task4_predictions(
    *,
    complete: bool,
    unequal_episodes: bool = False,
) -> pd.DataFrame:
    metadata = _development_metadata()
    if complete:
        games = metadata.loc[metadata["nfl_week"].between(3, 12)]
    else:
        games = metadata.loc[metadata["nfl_week"].isin((3, 4))].head(4)
    week_by_game = metadata.set_index("game_id")["nfl_week"]
    rows: list[dict[str, object]] = []
    source_row_id = 0
    for game_position, game in enumerate(games.itertuples(index=False)):
        fold_id, train_weeks, validation_weeks = _fold_for_week(
            int(game.nfl_week)
        )
        training_game_ids = tuple(
            sorted(
                week_by_game[
                    week_by_game.isin(train_weeks)
                ].index.astype(str)
            )
        )
        validation_game_ids = tuple(
            sorted(
                week_by_game[
                    week_by_game.isin(validation_weeks)
                ].index.astype(str)
            )
        )
        episode_count = (
            1 + game_position % 3 if unequal_episodes else 2
        )
        for episode_index in range(episode_count):
            truth = bool((game_position + episode_index) % 2)
            direction = ("UP", "DOWN", "NO_MOVE")[
                (game_position + episode_index) % 3
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
                            "game_id": str(game.game_id),
                            "nfl_week": int(game.nfl_week),
                            "atomic_information_episode_id": (
                                f"{game.game_id}:episode-{episode_index}"
                            ),
                            "venue": "polymarket",
                            "training_venue": "polymarket",
                            "calibration_venue": "polymarket",
                            "transport_mode": "VENUE_SPECIFIC",
                            "actual_home_contract_id": (
                                f"poly-{game.game_id}"
                            ),
                            "landmark_seconds": landmark,
                            "endpoint_seconds": endpoint,
                            "fold_id": fold_id,
                            "train_weeks": train_weeks,
                            "validation_weeks": validation_weeks,
                            "training_game_ids": training_game_ids,
                            "validation_game_ids": validation_game_ids,
                            "preprocessor_fit_game_ids": training_game_ids,
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
                            "o_h_given_s_calibrated_probability": (
                                probability
                            ),
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


def _run(
    *,
    complete: bool = True,
    unequal_episodes: bool = False,
) -> X15ModelRun:
    fold_ids = tuple(fold[0] for fold in FOLDS) if complete else ("fold_01",)
    return X15ModelRun(
        oof_predictions=_task4_predictions(
            complete=complete,
            unequal_episodes=unequal_episodes,
        ),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256="sha256:" + "c" * 64,
        run_config=_run_config(fold_ids=fold_ids),
    )


def _spec() -> FrozenSelectionSpec:
    return FrozenSelectionSpec(
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D4",
    )


def test_multihead_selection_uses_proper_joint_log_score() -> None:
    pairs = pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "atomic_information_episode_id": "episode-1",
                "s_h_truth_b0": True,
                "s_h_truth_candidate": True,
                "o_h_given_s_truth_b0": True,
                "o_h_given_s_truth_candidate": True,
                "direction_truth_b0": "UP",
                "direction_truth_candidate": "UP",
                "s_h_calibrated_probability_b0": 0.6,
                "s_h_calibrated_probability_candidate": 0.8,
                "o_h_given_s_calibrated_probability_b0": 0.5,
                "o_h_given_s_calibrated_probability_candidate": 0.7,
                "direction_calibrated_prob_down_b0": 0.2,
                "direction_calibrated_prob_no_move_b0": 0.3,
                "direction_calibrated_prob_up_b0": 0.5,
                "direction_calibrated_prob_down_candidate": 0.1,
                "direction_calibrated_prob_no_move_candidate": 0.3,
                "direction_calibrated_prob_up_candidate": 0.6,
            }
        ]
    )

    scored = selection_module._attach_multihead_losses(pairs)

    assert scored.loc[0, "available_head_count"] == 3
    assert scored.loc[0, "baseline_integrated_row_loss"] == pytest.approx(
        -math.log(0.6) - math.log(0.5) - math.log(0.5)
    )
    assert scored.loc[0, "candidate_integrated_row_loss"] == pytest.approx(
        -math.log(0.8) - math.log(0.7) - math.log(0.6)
    )


def test_complete_153_all_five_folds_is_required_for_selection() -> None:
    result = select_candidate_against_b0(
        _run(), spec=_spec(), authority=_authority()
    )

    assert result.authority_gate_passed is True
    assert result.development_authority_game_count == 153
    assert result.observed_fold_ids == tuple(fold[0] for fold in FOLDS)
    assert result.bootstrap_samples == 10_000
    assert result.integrated_gate_passed is True
    assert result.anchor_gate_passed is True
    assert result.selected is True
    assert result.diagnostic_status == "HISTORICAL_SIGNAL_CANDIDATE"
    assert result.game_losses["hierarchical_weight"].eq(1.0).all()


def test_development_authority_requires_explicit_utc_kickoff() -> None:
    metadata = _development_metadata()
    metadata.loc[metadata.index[0], "kickoff_utc"] = (
        "2025-09-01T12:00:00"
    )

    with pytest.raises(ModelSelectionError, match="explicit UTC"):
        bind_frozen_development_authority(
            metadata,
            cohort_authority_sha256=AUTHORITY,
        )


def test_resumable_fold_slice_is_diagnostic_only_and_cannot_select() -> None:
    result = select_candidate_against_b0(
        _run(complete=False), spec=_spec(), authority=_authority()
    )

    assert result.integrated_gate_passed is True
    assert result.authority_gate_passed is False
    assert result.selected is False
    assert "ALL_FIVE_FOLDS_REQUIRED" in result.authority_gate_failures
    assert result.diagnostic_status == "PARTIAL_DEVELOPMENT_DIAGNOSTIC_ONLY"


def test_clean_anchor_requires_thirty_distinct_games() -> None:
    run = _run()
    anchor = (
        run.oof_predictions["landmark_seconds"].eq(3)
        & run.oof_predictions["endpoint_seconds"].eq(30)
    )
    retained_games = set(
        sorted(run.oof_predictions.loc[anchor, "game_id"].unique())[:29]
    )
    trimmed = run.oof_predictions.loc[
        ~anchor | run.oof_predictions["game_id"].isin(retained_games)
    ].copy()
    trimmed_run = X15ModelRun(
        oof_predictions=trimmed,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )

    result = select_candidate_against_b0(
        trimmed_run, spec=_spec(), authority=_authority()
    )

    assert result.authority_gate_passed is True
    assert result.anchor_game_count == 29
    assert result.anchor_support_status == "INSUFFICIENT_SUPPORT"
    assert result.anchor_gate_passed is False
    assert result.selected is False


def test_selection_authority_rejects_game_week_identity_drift() -> None:
    run = _run()
    drifted_predictions = run.oof_predictions.copy()
    game_id = str(
        drifted_predictions.loc[
            drifted_predictions["nfl_week"].eq(3), "game_id"
        ].iloc[0]
    )
    drifted_predictions.loc[
        drifted_predictions["game_id"].eq(game_id), "nfl_week"
    ] = 4
    drifted_run = X15ModelRun(
        oof_predictions=drifted_predictions,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )

    result = select_candidate_against_b0(
        drifted_run, spec=_spec(), authority=_authority()
    )

    assert result.authority_gate_passed is False
    assert "GAME_WEEK_AUTHORITY_MISMATCH" in result.authority_gate_failures
    assert result.selected is False


def test_exact_pairs_anchor_and_polymarket_source_are_frozen() -> None:
    run = _run()
    missing = run.oof_predictions.drop(
        run.oof_predictions[
            run.oof_predictions["model_id"].eq(
                "regularized_logistic_v1"
            )
        ].index[0]
    )
    missing_run = X15ModelRun(
        oof_predictions=missing,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(ModelSelectionError, match="exact candidate/B0 pairs"):
        select_candidate_against_b0(
            missing_run, spec=_spec(), authority=_authority()
        )

    without_anchor = run.oof_predictions.loc[
        lambda frame: ~(
            frame["landmark_seconds"].eq(3)
            & frame["endpoint_seconds"].eq(30)
        )
    ]
    anchor_run = X15ModelRun(
        oof_predictions=without_anchor,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(ModelSelectionError, match="L=3.*H=30"):
        select_candidate_against_b0(
            anchor_run, spec=_spec(), authority=_authority()
        )

    with pytest.raises(ModelSelectionError, match="fixed at polymarket"):
        FrozenSelectionSpec(
            candidate_model_id="regularized_logistic_v1",
            candidate_feature_block_id="D4",
            selection_venue="kalshi",
        )


def test_factor_audit_uses_equal_game_units_and_all_registered_gates() -> None:
    selection = select_candidate_against_b0(
        _run(unequal_episodes=True),
        spec=_spec(),
        authority=_authority(),
    )
    membership = selection.paired_rows[
        ["game_id", "atomic_information_episode_id"]
    ].drop_duplicates()
    membership["factor_id"] = "NFL.PASS.COMPLETE"
    membership["factor_version"] = "v1"

    audit = build_factor_claim_audit(
        selection,
        factor_membership=membership,
        min_support_games=30,
        min_support_episodes=30,
    )
    row = audit.iloc[0]
    anchor = selection.paired_rows.loc[
        selection.paired_rows["landmark_seconds"].eq(3)
        & selection.paired_rows["endpoint_seconds"].eq(30)
    ]
    expected_equal_game_mean = float(
        anchor.groupby("game_id")["loss_improvement"].mean().mean()
    )

    assert row["mean_effect"] == pytest.approx(expected_equal_game_mean)
    assert row["equal_game_effect_unit_count"] == row["support_games"]
    required_gates = {
        "recommended_for_gating",
        "mean_ci_excludes_zero",
        "bh_q_gate_passed",
        "loo_sign_gate_passed",
        "max_game_contribution_gate_passed",
        "individual_statistical_gate",
        "passes_development_gate",
        "registered_gate_passed",
    }
    assert required_gates.issubset(audit.columns)
    assert bool(row["passes_development_gate"]) is False
    assert bool(row["registered_gate_passed"]) is False
    assert row["diagnostic_status"] != "HISTORICAL_SIGNAL_CANDIDATE"


def test_diagnostic_block_and_nonexecution_contract_fail_closed() -> None:
    with pytest.raises(ModelSelectionError, match="D1..D4"):
        FrozenSelectionSpec(
            candidate_model_id="regularized_logistic_v1",
            candidate_feature_block_id="B4",
        )

    run = _run()
    predictions = run.oof_predictions.copy()
    predictions.loc[
        predictions["model_id"].eq("regularized_logistic_v1"),
        "claim_eligible",
    ] = True
    drifted = X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(ModelSelectionError, match="claim_eligible=False"):
        select_candidate_against_b0(
            drifted, spec=_spec(), authority=_authority()
        )
