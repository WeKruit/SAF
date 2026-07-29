from __future__ import annotations

from dataclasses import replace
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
    select_stage_b_v2_winner,
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
STAGE_B_CANDIDATE_SUITE = (
    ("regularized_logistic_v1", "D1"),
    ("regularized_logistic_v1", "D2"),
    ("regularized_logistic_v1", "D3"),
    ("regularized_logistic_v1", "D4"),
    ("shallow_xgboost_v1", "D1"),
    ("shallow_xgboost_v1", "D2"),
    ("shallow_xgboost_v1", "D3"),
    ("shallow_xgboost_v1", "D4"),
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


def _source_shards(
    *,
    fold_ids: tuple[str, ...],
    candidate_model_id: str,
    candidate_feature_block_id: str,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "fold_id": fold_id,
            "model_id": model_id,
            "feature_block_id": feature_block_id,
            "manifest_sha256": (
                "sha256:"
                + f"{position * 2 + cell_position + 1:064x}"
            ),
            "shard_bundle_sha256": (
                "sha256:"
                + f"{position * 2 + cell_position + 101:064x}"
            ),
            "oof_predictions_sha256": (
                "sha256:"
                + f"{position * 2 + cell_position + 201:064x}"
            ),
        }
        for position, fold_id in enumerate(fold_ids)
        for cell_position, (model_id, feature_block_id) in enumerate(
            (
                ("b0_empirical_v1", "D0"),
                (candidate_model_id, candidate_feature_block_id),
            )
        )
    )


def _run_config(
    *,
    fold_ids: tuple[str, ...],
    candidate_model_id: str = "regularized_logistic_v1",
    candidate_feature_block_id: str = "D4",
) -> dict[str, object]:
    return {
        "schema_version": "HistoricalTradesOnlyProbabilityPanelV2",
        "survival_probability_contract": (
            "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
        ),
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
        "model_ids": ("b0_empirical_v1", candidate_model_id),
        "feature_block_ids": ("D0", candidate_feature_block_id),
        "transport_pairs": (),
        "source_shards": _source_shards(
            fold_ids=fold_ids,
            candidate_model_id=candidate_model_id,
            candidate_feature_block_id=candidate_feature_block_id,
        ),
        "source_batch_manifest_path": (
            "batches/sha256-example.batch-manifest.json"
        ),
        "source_batch_manifest_sha256": "sha256:" + "b" * 64,
        "source_batch_sha256": "sha256:" + "d" * 64,
    }


def _fold_for_week(week: int):
    return next(fold for fold in FOLDS if week in fold[2])


def _task4_predictions(
    *,
    complete: bool,
    unequal_episodes: bool = False,
    candidate_model_id: str = "regularized_logistic_v1",
    candidate_feature_block_id: str = "D4",
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
                    (
                        candidate_model_id,
                        candidate_feature_block_id,
                        True,
                    ),
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
                                "HistoricalTradesOnlyProbabilityPanelV2"
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
    candidate_model_id: str = "regularized_logistic_v1",
    candidate_feature_block_id: str = "D4",
) -> X15ModelRun:
    fold_ids = tuple(fold[0] for fold in FOLDS) if complete else ("fold_01",)
    run_config = _run_config(
        fold_ids=fold_ids,
        candidate_model_id=candidate_model_id,
        candidate_feature_block_id=candidate_feature_block_id,
    )
    return X15ModelRun(
        oof_predictions=_task4_predictions(
            complete=complete,
            unequal_episodes=unequal_episodes,
            candidate_model_id=candidate_model_id,
            candidate_feature_block_id=candidate_feature_block_id,
        ),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256=selection_module._canonical_sha256(run_config),
        run_config=run_config,
    )


def _stage_b_v2_runs() -> tuple[X15ModelRun, ...]:
    return tuple(
        _run(
            candidate_model_id=model_id,
            candidate_feature_block_id=feature_block_id,
        )
        for model_id, feature_block_id in STAGE_B_CANDIDATE_SUITE
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


def test_stage_b_v2_winner_interface_is_exposed() -> None:
    assert callable(select_stage_b_v2_winner)
    assert callable(selection_module.verify_stage_b_v2_selection_result)
    assert (
        selection_module.STAGE_B_CANDIDATE_SUITE
        == STAGE_B_CANDIDATE_SUITE
    )


@pytest.mark.parametrize(
    ("schema_version", "survival_contract"),
    [
        ("HistoricalTradesOnlyProbabilityPanelV1", None),
        ("HistoricalTradesOnlyProbabilityPanelV2", None),
        (
            "HistoricalTradesOnlyProbabilityPanelV2",
            "CONDITIONAL_HEAD_PRODUCT_V0",
        ),
    ],
)
def test_stage_b_rejects_v1_or_missing_survival_contract(
    schema_version: str,
    survival_contract: str | None,
) -> None:
    runs = _stage_b_v2_runs()
    run = runs[0]
    run_config = dict(run.run_config)
    run_config["schema_version"] = schema_version
    if survival_contract is None:
        run_config.pop("survival_probability_contract", None)
    else:
        run_config["survival_probability_contract"] = survival_contract
    predictions = run.oof_predictions.copy()
    predictions["schema_version"] = schema_version
    drifted = replace(
        run,
        oof_predictions=predictions,
        run_config=run_config,
        run_config_sha256=selection_module._canonical_sha256(run_config),
    )

    with pytest.raises(
        ModelSelectionError,
        match=(
            "HistoricalTradesOnlyProbabilityPanelV2"
            if schema_version.endswith("V1")
            else "survival_probability_contract"
        ),
    ):
        select_stage_b_v2_winner(
            (drifted, *runs[1:]),
            authority=_authority(),
        )


def test_stage_b_requires_all_eight_frozen_candidates() -> None:
    runs = _stage_b_v2_runs()

    with pytest.raises(ModelSelectionError, match="all 8 frozen"):
        select_stage_b_v2_winner(
            runs[:-1],
            authority=_authority(),
        )


def test_stage_b_consumes_eight_verified_projection_runs_end_to_end() -> None:
    runs = _stage_b_v2_runs()
    for run, candidate in zip(
        runs, STAGE_B_CANDIDATE_SUITE, strict=True
    ):
        observed = set(
            zip(
                run.oof_predictions["model_id"],
                run.oof_predictions["feature_block_id"],
                strict=True,
            )
        )
        assert observed == {("b0_empirical_v1", "D0"), candidate}

    result = select_stage_b_v2_winner(
        runs,
        authority=_authority(),
    )

    assert len(result.candidate_results) == 8
    assert result.candidate_audit["gate_selected"].all()
    assert result.winner is not None
    assert (
        result.winner.spec.candidate_model_id,
        result.winner.spec.candidate_feature_block_id,
    ) == STAGE_B_CANDIDATE_SUITE[0]
    assert {
        "paired_rows_sha256",
        "episode_losses_sha256",
        "game_losses_sha256",
        "anchor_game_losses_sha256",
        "model_selection_result_sha256",
    }.issubset(result.candidate_audit.columns)
    assert (
        selection_module.verify_stage_b_v2_selection_result(result)
        is result
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "paired_rows",
        "paired_rows_index",
        "paired_rows_dtype",
        "promotion_metric",
        "winner_identity",
    ],
)
def test_stage_b_verifier_rejects_replaced_candidate_evidence(
    mutation: str,
) -> None:
    result = select_stage_b_v2_winner(
        _stage_b_v2_runs(),
        authority=_authority(),
    )
    winner = result.winner
    assert winner is not None

    if mutation == "winner_identity":
        replaced = replace(result, winner=result.candidate_results[1])
    else:
        if mutation == "paired_rows":
            mutated_winner = replace(
                winner,
                paired_rows=winner.paired_rows.iloc[:-1].copy(),
            )
        elif mutation == "paired_rows_index":
            mutated_rows = winner.paired_rows.copy()
            mutated_rows.index = pd.Index(
                range(len(mutated_rows)),
                dtype="uint64",
            )
            mutated_winner = replace(
                winner,
                paired_rows=mutated_rows,
            )
        elif mutation == "paired_rows_dtype":
            mutated_rows = winner.paired_rows.copy()
            dtype = mutated_rows["game_id"].dtype
            assert isinstance(dtype, pd.StringDtype)
            storage = "pyarrow" if dtype.storage == "python" else "python"
            mutated_rows["game_id"] = mutated_rows["game_id"].astype(
                pd.StringDtype(storage=storage)
            )
            mutated_winner = replace(
                winner,
                paired_rows=mutated_rows,
            )
        else:
            mutated_winner = replace(
                winner,
                integrated_mean_improvement=(
                    winner.integrated_mean_improvement + 1.0
                ),
            )
        candidate_results = tuple(
            mutated_winner if candidate is winner else candidate
            for candidate in result.candidate_results
        )
        replaced = replace(
            result,
            winner=mutated_winner,
            candidate_results=candidate_results,
        )

    with pytest.raises(ModelSelectionError, match="Stage-B V2"):
        selection_module.verify_stage_b_v2_selection_result(replaced)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(None, math.inf, id="missing-to-positive-infinity"),
        pytest.param(math.nan, None, id="nan-to-none"),
        pytest.param(math.inf, -math.inf, id="infinity-sign"),
    ],
)
def test_stage_b_verifier_rejects_typed_paired_row_cell_replacement(
    before: object,
    after: object,
) -> None:
    result = select_stage_b_v2_winner(
        _stage_b_v2_runs(),
        authority=_authority(),
    )
    winner = result.winner
    assert winner is not None
    paired_rows = winner.paired_rows.copy()
    paired_rows["loss_improvement"] = paired_rows[
        "loss_improvement"
    ].astype(object)
    paired_rows.at[paired_rows.index[0], "loss_improvement"] = before
    rebound_winner = replace(winner, paired_rows=paired_rows)
    rebound_candidates = tuple(
        rebound_winner if candidate is winner else candidate
        for candidate in result.candidate_results
    )
    (
        standard_errors,
        expected_winner,
        _,
        _,
        _,
        within_one_se,
    ) = selection_module._stage_b_decision(rebound_candidates)
    assert expected_winner is rebound_winner
    audit_rows = selection_module._stage_b_candidate_audit_rows(
        rebound_candidates,
        standard_errors=standard_errors,
        within_one_se=within_one_se,
        winner=expected_winner,
        shared_run_config_sha256=result.shared_run_config_sha256,
        suite_run_config_sha256=result.suite_run_config_sha256,
    )
    candidate_audit = pd.DataFrame(audit_rows)
    rebound = replace(
        result,
        winner=rebound_winner,
        candidate_results=rebound_candidates,
        candidate_audit=candidate_audit,
        candidate_audit_sha256=(
            selection_module._dataframe_evidence_sha256(
                candidate_audit,
                field="candidate_audit",
            )
        ),
    )
    assert (
        selection_module.verify_stage_b_v2_selection_result(rebound)
        is rebound
    )

    attacked_rows = paired_rows.copy()
    attacked_rows.at[
        attacked_rows.index[0], "loss_improvement"
    ] = after
    attacked_winner = replace(
        rebound_winner,
        paired_rows=attacked_rows,
    )
    attacked_candidates = tuple(
        attacked_winner if candidate is rebound_winner else candidate
        for candidate in rebound_candidates
    )
    attacked = replace(
        rebound,
        winner=attacked_winner,
        candidate_results=attacked_candidates,
    )

    with pytest.raises(ModelSelectionError, match="Stage-B V2"):
        selection_module.verify_stage_b_v2_selection_result(attacked)


@pytest.mark.parametrize(
    "field",
    (
        "paired_rows",
        "episode_losses",
        "game_losses",
        "anchor_game_losses",
    ),
)
@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(None, pd.NA, id="none-vs-pandas-na"),
        pytest.param(None, math.nan, id="none-vs-nan"),
        pytest.param(None, math.inf, id="none-vs-positive-infinity"),
        pytest.param(math.inf, -math.inf, id="infinity-sign"),
        pytest.param(True, 1, id="bool-vs-int"),
        pytest.param(1, 1.0, id="int-vs-float"),
        pytest.param(1, "1", id="int-vs-string"),
    ],
)
def test_dataframe_evidence_hash_distinguishes_typed_cells(
    field: str,
    left: object,
    right: object,
) -> None:
    left_frame = pd.DataFrame(
        {"value": pd.Series([left], dtype=object)}
    )
    right_frame = pd.DataFrame(
        {"value": pd.Series([right], dtype=object)}
    )

    assert selection_module._dataframe_evidence_sha256(
        left_frame,
        field=field,
    ) != selection_module._dataframe_evidence_sha256(
        right_frame,
        field=field,
    )


@pytest.mark.parametrize("dtype", ("uint64", object))
def test_dataframe_evidence_hash_binds_row_index_class_and_dtype(
    dtype: object,
) -> None:
    range_indexed = pd.DataFrame({"value": [1]})
    typed_indexed = pd.DataFrame(
        {"value": [1]},
        index=pd.Index([0], dtype=dtype),
    )

    assert selection_module._dataframe_evidence_sha256(
        range_indexed,
        field="paired_rows",
    ) != selection_module._dataframe_evidence_sha256(
        typed_indexed,
        field="paired_rows",
    )


@pytest.mark.parametrize("dtype", ("uint64", object))
def test_dataframe_evidence_hash_binds_column_index_class_and_dtype(
    dtype: object,
) -> None:
    range_columns = pd.DataFrame([[1]])
    typed_columns = pd.DataFrame(
        [[1]],
        columns=pd.Index([0], dtype=dtype),
    )

    assert selection_module._dataframe_evidence_sha256(
        range_columns,
        field="paired_rows",
    ) != selection_module._dataframe_evidence_sha256(
        typed_columns,
        field="paired_rows",
    )


@pytest.mark.parametrize("axis", ("index", "columns"))
@pytest.mark.parametrize("mutation", ("level_dtype", "levels_and_codes"))
def test_dataframe_evidence_hash_binds_multiindex_metadata(
    axis: str,
    mutation: str,
) -> None:
    if mutation == "level_dtype":
        left_axis = pd.MultiIndex(
            levels=[
                pd.Index([1], dtype="int64"),
                pd.Index(["a"]),
            ],
            codes=[[0], [0]],
        )
        right_axis = pd.MultiIndex(
            levels=[
                pd.Index([1], dtype="uint64"),
                pd.Index(["a"]),
            ],
            codes=[[0], [0]],
        )
    else:
        left_axis = pd.MultiIndex(
            levels=[pd.Index(["a", "b"]), pd.Index([1])],
            codes=[[0], [0]],
        )
        right_axis = pd.MultiIndex(
            levels=[pd.Index(["b", "a"]), pd.Index([1])],
            codes=[[1], [0]],
        )
    left = pd.DataFrame([[1]])
    right = pd.DataFrame([[1]])
    setattr(left, axis, left_axis)
    setattr(right, axis, right_axis)

    assert selection_module._dataframe_evidence_sha256(
        left,
        field="paired_rows",
    ) != selection_module._dataframe_evidence_sha256(
        right,
        field="paired_rows",
    )


@pytest.mark.parametrize(
    ("left_dtype", "right_dtype"),
    [
        pytest.param(
            pd.StringDtype(storage="python"),
            pd.StringDtype(storage="pyarrow"),
            id="string-storage",
        ),
        pytest.param(
            pd.CategoricalDtype(
                categories=["a", "b"],
                ordered=False,
            ),
            pd.CategoricalDtype(
                categories=["a", "c"],
                ordered=False,
            ),
            id="categorical-categories",
        ),
        pytest.param(
            pd.CategoricalDtype(
                categories=["a", "b"],
                ordered=False,
            ),
            pd.CategoricalDtype(
                categories=["a", "b"],
                ordered=True,
            ),
            id="categorical-order",
        ),
    ],
)
def test_dataframe_evidence_hash_binds_typed_dtype_parameters(
    left_dtype: object,
    right_dtype: object,
) -> None:
    left = pd.DataFrame(
        {"value": pd.Series(["a"], dtype=left_dtype)}
    )
    right = pd.DataFrame(
        {"value": pd.Series(["a"], dtype=right_dtype)}
    )

    assert selection_module._dataframe_evidence_sha256(
        left,
        field="paired_rows",
    ) != selection_module._dataframe_evidence_sha256(
        right,
        field="paired_rows",
    )


def test_typed_dtype_descriptor_binds_datetime_and_arrow_metadata() -> None:
    datetime_descriptor = selection_module._typed_dtype_descriptor(
        pd.DatetimeTZDtype(unit="ns", tz="UTC")
    )
    arrow_descriptor = selection_module._typed_dtype_descriptor(
        pd.Series([1], dtype="int64[pyarrow]").dtype
    )

    assert datetime_descriptor["parameters"] == {
        "unit": "ns",
        "timezone_type": "datetime.timezone",
        "timezone_str": "UTC",
        "timezone_repr": "datetime.timezone.utc",
    }
    assert arrow_descriptor["parameters"] == {
        "arrow_type": "pyarrow.lib.DataType",
        "arrow_str": "int64",
        "arrow_repr": "DataType(int64)",
    }


def test_stage_b_verifier_rejects_candidate_audit_typed_cell_replacement(
) -> None:
    result = select_stage_b_v2_winner(
        _stage_b_v2_runs(),
        authority=_authority(),
    )
    attacked_audit = result.candidate_audit.copy()
    attacked_audit["final_winner"] = attacked_audit[
        "final_winner"
    ].astype(object)
    winner_index = attacked_audit.index[
        attacked_audit["final_winner"].eq(True)
    ][0]
    attacked_audit.at[winner_index, "final_winner"] = 1
    attacked = replace(result, candidate_audit=attacked_audit)

    with pytest.raises(ModelSelectionError, match="candidate audit"):
        selection_module.verify_stage_b_v2_selection_result(attacked)


def _stub_suite_results(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_means: dict[tuple[str, str], tuple[bool, float, float]],
) -> None:
    base = select_candidate_against_b0(
        _run(),
        spec=_spec(),
        authority=_authority(),
    )

    def stub(
        model_run: X15ModelRun,
        *,
        spec: FrozenSelectionSpec,
        authority,
    ):
        del authority
        selected, mean, standard_error = selected_means[
            (spec.candidate_model_id, spec.candidate_feature_block_id)
        ]
        spread = standard_error
        game_losses = pd.DataFrame(
            {
                "game_id": ("stub-game-1", "stub-game-2"),
                "loss_improvement": (
                    mean - spread,
                    mean + spread,
                ),
            }
        )
        return replace(
            base,
            spec=spec,
            diagnostic_status=(
                "HISTORICAL_SIGNAL_CANDIDATE"
                if selected
                else "HISTORICAL_SIGNAL_REJECTED"
            ),
            integrated_mean_improvement=mean,
            integrated_ci_low=mean - 0.01,
            integrated_ci_high=mean + 0.01,
            integrated_gate_passed=selected,
            anchor_mean_improvement=max(mean, 0.0),
            anchor_sign_reversed=False,
            anchor_gate_passed=selected,
            game_losses=game_losses,
            run_config_sha256=model_run.run_config_sha256,
            selected=selected,
        )

    monkeypatch.setattr(
        selection_module,
        "select_candidate_against_b0",
        stub,
    )


def test_stage_b_no_gate_passes_means_no_model_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_suite_results(
        monkeypatch,
        selected_means={
            pair: (False, -0.01, 0.01)
            for pair in STAGE_B_CANDIDATE_SUITE
        },
    )

    result = select_stage_b_v2_winner(
        _stage_b_v2_runs(),
        authority=_authority(),
    )

    assert result.decision_status == "NO_MODEL_ADVANCE"
    assert result.winner is None
    assert len(result.candidate_audit) == 8
    assert not result.candidate_audit["within_one_se"].any()
    assert not result.candidate_audit["final_winner"].any()


def test_stage_b_one_se_rule_selects_simpler_eligible_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_means = {
        pair: (True, 0.10, 0.01)
        for pair in STAGE_B_CANDIDATE_SUITE
    }
    selected_means[("regularized_logistic_v1", "D1")] = (
        True,
        0.16,
        0.01,
    )
    selected_means[("shallow_xgboost_v1", "D4")] = (
        True,
        0.20,
        0.05,
    )
    _stub_suite_results(
        monkeypatch,
        selected_means=selected_means,
    )

    result = select_stage_b_v2_winner(
        _stage_b_v2_runs(),
        authority=_authority(),
    )

    assert result.decision_status == "MODEL_ADVANCE"
    assert result.winner is not None
    assert (
        result.winner.spec.candidate_model_id,
        result.winner.spec.candidate_feature_block_id,
    ) == ("regularized_logistic_v1", "D1")
    assert result.best_standard_error == pytest.approx(0.05)
    assert result.one_se_threshold == pytest.approx(0.15)
    assert result.candidate_audit["complexity_rank"].tolist() == list(
        range(1, 9)
    )
    assert result.candidate_audit["winner_rule_sha256"].nunique() == 1
    assert result.candidate_audit["cohort_authority_sha256"].nunique() == 1
    assert result.candidate_audit["run_config_sha256"].nunique() == 8
    assert (
        result.candidate_audit["shared_run_config_sha256"].nunique()
        == 1
    )
    assert result.candidate_audit_sha256 == (
        selection_module._dataframe_evidence_sha256(
            result.candidate_audit,
            field="candidate_audit",
        )
    )
    assert result.suite_run_config_sha256 == (
        selection_module._canonical_sha256(
            tuple(
                {
                    "candidate": candidate,
                    "run_config_sha256": run.run_config_sha256,
                }
                for candidate, run in zip(
                    STAGE_B_CANDIDATE_SUITE,
                    _stage_b_v2_runs(),
                    strict=True,
                )
            )
        )
    )
    assert not (
        result.candidate_audit["candidate_feature_block_id"] == "D0"
    ).any()


def test_stage_b_clearly_better_candidate_outside_one_se_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_means = {
        pair: (True, 0.10, 0.01)
        for pair in STAGE_B_CANDIDATE_SUITE
    }
    selected_means[("regularized_logistic_v1", "D1")] = (
        True,
        0.20,
        0.01,
    )
    selected_means[("shallow_xgboost_v1", "D4")] = (
        True,
        0.30,
        0.01,
    )
    _stub_suite_results(
        monkeypatch,
        selected_means=selected_means,
    )

    result = select_stage_b_v2_winner(
        _stage_b_v2_runs(),
        authority=_authority(),
    )

    assert result.winner is not None
    assert (
        result.winner.spec.candidate_model_id,
        result.winner.spec.candidate_feature_block_id,
    ) == ("shallow_xgboost_v1", "D4")
    simple = result.candidate_audit.loc[
        result.candidate_audit["complexity_rank"].eq(1)
    ].iloc[0]
    assert bool(simple["within_one_se"]) is False


def test_stage_b_kalshi_rows_cannot_satisfy_candidate_suite() -> None:
    runs = _stage_b_v2_runs()
    run = runs[0]
    missing_pair = STAGE_B_CANDIDATE_SUITE[0]
    predictions = run.oof_predictions.copy()
    kalshi_mask = (
        predictions["model_id"].eq(missing_pair[0])
        & predictions["feature_block_id"].eq(missing_pair[1])
    )
    predictions.loc[kalshi_mask, "venue"] = "kalshi"
    predictions.loc[kalshi_mask, "training_venue"] = "kalshi"
    predictions.loc[kalshi_mask, "calibration_venue"] = "kalshi"

    with pytest.raises(ModelSelectionError, match="all 8 frozen"):
        select_stage_b_v2_winner(
            (
                replace(run, oof_predictions=predictions),
                *runs[1:],
            ),
            authority=_authority(),
        )


def test_stage_b_rejects_duplicate_projection_candidate_identity() -> None:
    runs = _stage_b_v2_runs()

    with pytest.raises(ModelSelectionError, match="duplicate candidate"):
        select_stage_b_v2_winner(
            (*runs[:-1], runs[0]),
            authority=_authority(),
        )


def test_stage_b_rejects_wrong_projection_run_identity() -> None:
    runs = _stage_b_v2_runs()
    run = runs[0]
    run_config = dict(run.run_config)
    run_config["model_ids"] = (
        "b0_empirical_v1",
        "not_a_real_model",
    )
    drifted = replace(
        run,
        run_config=run_config,
        run_config_sha256=selection_module._canonical_sha256(run_config),
    )

    with pytest.raises(ModelSelectionError, match="run_config identity"):
        select_stage_b_v2_winner(
            (drifted, *runs[1:]),
            authority=_authority(),
        )


def test_stage_b_rejects_run_config_digest_mismatch() -> None:
    runs = _stage_b_v2_runs()
    run = runs[0]
    run_config = dict(run.run_config)
    run_config["source_batch_sha256"] = "sha256:" + "e" * 64

    with pytest.raises(ModelSelectionError, match="SHA-256 mismatch"):
        select_stage_b_v2_winner(
            (
                replace(run, run_config=run_config),
                *runs[1:],
            ),
            authority=_authority(),
        )


def test_stage_b_rejects_duplicate_source_shard_identity() -> None:
    runs = _stage_b_v2_runs()
    run = runs[0]
    run_config = dict(run.run_config)
    source_shards = list(run_config["source_shards"])
    source_shards[-1] = source_shards[0]
    run_config["source_shards"] = tuple(source_shards)
    drifted = replace(
        run,
        run_config=run_config,
        run_config_sha256=selection_module._canonical_sha256(run_config),
    )

    with pytest.raises(ModelSelectionError, match="source_shards"):
        select_stage_b_v2_winner(
            (drifted, *runs[1:]),
            authority=_authority(),
        )
