from __future__ import annotations

from dataclasses import replace
import inspect
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.research import nfl_x16_kalshi_validation as validation
from prediction_market.research.nfl_x15_model_selection import (
    FrozenSelectionSpec,
    bind_frozen_development_authority,
    select_candidate_against_b0,
    select_stage_b_v3_winner,
)
from prediction_market.research.nfl_x15_models import (
    X15ModelRun,
    run_x15_historical_trades_diagnostic_walk_forward,
)
from prediction_market.research.nfl_x16_kalshi_validation import (
    KalshiValidationError,
    bind_frozen_authority_metadata,
    begin_prelock_metadata_ledger,
    hash_game_id_evidence,
    lock_preholdout_metadata_audit,
    record_metadata_access,
)
from test_nfl_x15_models import (
    _diagnostic_panel,
    _sha256_text,
)
from test_nfl_x15_model_selection import (
    _authority as _complete_selection_authority,
    _run as _complete_selection_run,
    _spec as _complete_selection_spec,
    _stage_b_v3_runs,
)


AUTHORITY = "sha256:" + "a" * 64
TARGET_CONTRACT = "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
CLAIM_BOUNDARY = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)
X11_SPORTS_OUTCOME_EVIDENCE_SHA256 = (
    "sha256:1d0c033459c69778e265be3fca16ae2c87f650d5003a61ffdea4c020a4fd0b05"
)
STAGE_B_BATCH_MANIFEST_PATH = Path(
    "x15-selection-v3/verified-exact45.batch-index.json"
)
STAGE_B_ARTIFACT_ROOT = Path("/verified-test-artifacts")


def _validate_transport(
    model_run: X15ModelRun,
    *,
    authority_metadata,
    stage_b_selection,
):
    return validation.validate_development_venue_transport(
        model_run,
        authority_metadata=authority_metadata,
        stage_b_selection=stage_b_selection,
        stage_b_batch_manifest_path=STAGE_B_BATCH_MANIFEST_PATH,
        stage_b_artifact_root=STAGE_B_ARTIFACT_ROOT,
    )


def test_kalshi_validation_uses_proper_joint_log_score() -> None:
    grain = {
        "game_id": "game-1",
        "nfl_week": 3,
        "atomic_information_episode_id": "episode-1",
        "landmark_seconds": 3,
        "endpoint_seconds": 30,
        "fold_id": "fold_01",
        "cohort_authority_sha256": AUTHORITY,
        "target_contract": TARGET_CONTRACT,
        "claim_boundary": CLAIM_BOUNDARY,
        "direction_threshold_probability": 0.01,
        "direction_threshold_semantics": (
            "FIXED_CROSS_VENUE_RESEARCH_MATERIALITY_NOT_TICK"
        ),
        "venue_tick_support": "UNSUPPORTED",
        "market_continuity_support": "UNKNOWN",
        "schema_version": "HistoricalTradesOnlyProbabilityPanelV2",
        "analysis_scope": "HISTORICAL_TRADES_ONLY_SOURCE_TIME_DIAGNOSTIC",
        "claim_eligible": False,
    }
    pairs = pd.DataFrame(
        [
            {
                **grain,
                "model_id_candidate": "regularized_logistic_v1",
                "feature_block_id_candidate": "D4",
                "model_id_b0": "b0_empirical_v1",
                "feature_block_id_b0": "D0",
                "actual_home_contract_id_candidate": "kalshi:home",
                "actual_home_contract_id_b0": "kalshi:home",
                "s_h_truth_candidate": True,
                "s_h_truth_b0": True,
                "o_h_given_s_truth_candidate": True,
                "o_h_given_s_truth_b0": True,
                "direction_truth_candidate": "UP",
                "direction_truth_b0": "UP",
                "s_h_calibrated_probability_candidate": 0.8,
                "s_h_calibrated_probability_b0": 0.6,
                "o_h_given_s_calibrated_probability_candidate": 0.7,
                "o_h_given_s_calibrated_probability_b0": 0.5,
                "direction_calibrated_prob_down_candidate": 0.1,
                "direction_calibrated_prob_no_move_candidate": 0.3,
                "direction_calibrated_prob_up_candidate": 0.6,
                "direction_calibrated_prob_down_b0": 0.2,
                "direction_calibrated_prob_no_move_b0": 0.3,
                "direction_calibrated_prob_up_b0": 0.5,
            }
        ]
    )

    scored = validation._candidate_b0_integrated_log_loss(
        pairs, layer="TEST"
    )

    assert scored.loc[0, "available_head_count"] == 3
    assert scored.loc[0, "candidate_integrated_log_loss"] == pytest.approx(
        -math.log(0.8) - math.log(0.7) - math.log(0.6)
    )
    assert scored.loc[0, "b0_integrated_log_loss"] == pytest.approx(
        -math.log(0.6) - math.log(0.5) - math.log(0.5)
    )
X11_HOLDOUT_DRIVE_OUTCOME_COUNT = 1_683


def _selection_spec() -> FrozenSelectionSpec:
    return FrozenSelectionSpec(
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D1",
    )


def _stage_b_selection(*, authority=None):
    source_authority = _complete_selection_authority()
    if authority is None:
        authority = source_authority
    if authority is source_authority:
        runs = _stage_b_v3_runs()
    else:
        target_cohort = authority.cohort_authority_sha256
        authority = bind_frozen_development_authority(
            pd.DataFrame(
                [
                    {
                        "game_id": game_id,
                        "nfl_week": nfl_week,
                        "kickoff_utc": kickoff_utc,
                        "batch_sha256": batch_sha256,
                        "cohort_authority_sha256": target_cohort,
                    }
                    for (
                        game_id,
                        nfl_week,
                        kickoff_utc,
                        batch_sha256,
                    ) in source_authority.development_games
                ]
            ),
            cohort_authority_sha256=target_cohort,
        )
        remapped_runs: list[X15ModelRun] = []
        for run in _stage_b_v3_runs():
            predictions = run.oof_predictions.copy()
            predictions["cohort_authority_sha256"] = target_cohort
            run_config = dict(run.run_config)
            run_config["cohort_authority_sha256"] = target_cohort
            remapped_runs.append(
                replace(
                    run,
                    oof_predictions=predictions,
                    run_config=run_config,
                    run_config_sha256=validation._canonical_sha256(
                        run_config
                    ),
                )
            )
        runs = tuple(remapped_runs)
    return select_stage_b_v3_winner(runs, authority=authority)


def _run_config(
    *,
    source_provenance_sha256: str | None = None,
) -> dict[str, object]:
    stage_b = _stage_b_selection()
    winner = stage_b.winner
    assert winner is not None
    source_run = next(
        run
        for run in _stage_b_v3_runs()
        if winner.spec.candidate_model_id
        in run.run_config["model_ids"]
        and winner.spec.candidate_feature_block_id
        in run.run_config["feature_block_ids"]
    )
    winner_audit = stage_b.candidate_audit.loc[
        stage_b.candidate_audit["final_winner"].eq(True)
    ].iloc[0]
    config = dict(source_run.run_config)
    config["transport_pairs"] = (("polymarket", "kalshi"),)
    config["stage_b_source_run_config_sha256"] = (
        source_run.run_config_sha256
    )
    config["stage_b_winner_model_selection_result_sha256"] = (
        winner_audit["model_selection_result_sha256"]
    )
    config["stage_b_source_provenance_sha256"] = (
        source_provenance_sha256
        or "sha256:" + "0" * 64
    )
    return config


def _verified_stage_b_source_run(
    *,
    candidate_model_id: str,
    candidate_feature_block_id: str,
) -> X15ModelRun:
    source = next(
        run
        for run in _stage_b_v3_runs()
        if run.run_config["model_ids"]
        == ("b0_empirical_v1", candidate_model_id)
        and run.run_config["feature_block_ids"]
        == ("D0", candidate_feature_block_id)
    )
    predictions = source.oof_predictions.copy()
    training_ids = predictions["training_game_ids"].map(tuple)
    for column in (
        "calibrator_fit_game_ids_s_h",
        "calibrator_fit_game_ids_o_h_given_s",
        "calibrator_fit_game_ids_direction",
    ):
        predictions[column] = training_ids
    fold_character = predictions["fold_id"].astype(str).str[-1]
    predictions["training_data_sha256"] = fold_character.map(
        lambda value: "sha256:" + value * 64
    )
    predictions["preprocessor_training_sha256"] = (
        "sha256:" + "b" * 64
    )
    predictions["s_h_model_training_sha256"] = "sha256:" + "4" * 64
    predictions["o_h_given_s_model_training_sha256"] = (
        "sha256:" + "5" * 64
    )
    predictions["direction_model_training_sha256"] = (
        "sha256:" + "6" * 64
    )
    predictions["s_h_calibration_training_sha256"] = (
        "sha256:" + "c" * 64
    )
    predictions["o_h_given_s_calibration_training_sha256"] = (
        "sha256:" + "d" * 64
    )
    predictions["direction_calibration_training_sha256"] = (
        "sha256:" + "e" * 64
    )
    baseline = predictions["model_id"].eq("b0_empirical_v1")
    predictions["feature_block_sha256"] = "sha256:" + "7" * 64
    predictions["model_spec_sha256"] = "sha256:" + "8" * 64
    predictions.loc[baseline, "feature_block_sha256"] = (
        "sha256:" + "0" * 64
    )
    predictions.loc[baseline, "model_spec_sha256"] = (
        "sha256:" + "1" * 64
    )
    predictions["fold_sha256"] = fold_character.map(
        lambda value: "sha256:" + value * 64
    )
    return replace(source, oof_predictions=predictions)


@pytest.fixture(autouse=True)
def _verified_stage_b_source_loader(monkeypatch):
    def load(
        *,
        batch_manifest_path,
        artifact_root,
        candidate_model_id,
        candidate_feature_block_id,
    ):
        assert Path(batch_manifest_path) == STAGE_B_BATCH_MANIFEST_PATH
        assert Path(artifact_root) == STAGE_B_ARTIFACT_ROOT
        return _verified_stage_b_source_run(
            candidate_model_id=candidate_model_id,
            candidate_feature_block_id=candidate_feature_block_id,
        )

    monkeypatch.setattr(
        validation,
        "load_x15_selection_projection_v3",
        load,
    )


def _authority_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, count, weeks in (
        ("development", 153, tuple(range(1, 13))),
        ("holdout", 81, tuple(range(13, 19))),
    ):
        for index in range(count):
            development = cohort == "development"
            rows.append(
                {
                    "game_id": (
                        f"game-{index:03d}"
                        if development
                        else f"{cohort}-{index:03d}"
                    ),
                    "cohort": cohort,
                    "nfl_week": weeks[index % len(weeks)],
                    "kickoff_utc": (
                        f"2025-08-{index % 28 + 1:02d}T12:00:00Z"
                        if development
                        else (
                            f"2025-09-{index % 28 + 1:02d}"
                            "T12:00:00Z"
                        )
                    ),
                    "batch_sha256": "sha256:" + f"{index + 1:064x}",
                    "cohort_authority_sha256": AUTHORITY,
                    "market_reaction_exposure": (
                        "DEVELOPMENT_USED"
                        if cohort == "development"
                        else "SEALED_UNREAD"
                    ),
                    "sports_outcome_exposure": (
                        "DEVELOPMENT_USED"
                        if cohort == "development"
                        else "PRIOR_EXPOSED_X11"
                    ),
                    "reaction_read_count": 0,
                }
            )
    return pd.DataFrame(rows)


def _bind_authority(
    rows: pd.DataFrame | None = None,
    *,
    sports_outcome_exposed_game_ids: tuple[str, ...] | None = None,
    reaction_blind_metadata_game_ids: tuple[str, ...] | None = None,
    authority: str = AUTHORITY,
):
    authority_rows = _authority_rows() if rows is None else rows
    holdout_game_ids = tuple(
        authority_rows.loc[
            authority_rows["cohort"].eq("holdout"), "game_id"
        ].astype(str)
    )
    sports_ids = (
        holdout_game_ids
        if sports_outcome_exposed_game_ids is None
        else sports_outcome_exposed_game_ids
    )
    metadata_ids = (
        holdout_game_ids
        if reaction_blind_metadata_game_ids is None
        else reaction_blind_metadata_game_ids
    )
    return bind_frozen_authority_metadata(
        authority_rows,
        cohort_authority_sha256=authority,
        sports_outcome_exposed_game_ids=sports_ids,
        sports_outcome_exposed_game_ids_sha256=hash_game_id_evidence(
            sports_ids
        ),
        sports_outcome_source_evidence_sha256=(
            X11_SPORTS_OUTCOME_EVIDENCE_SHA256
        ),
        sports_outcome_observation_count=(
            X11_HOLDOUT_DRIVE_OUTCOME_COUNT
        ),
        reaction_blind_metadata_game_ids=metadata_ids,
        reaction_blind_metadata_game_ids_sha256=hash_game_id_evidence(
            metadata_ids
        ),
    )


def _verified_preholdout_lock(
    *,
    cohort_authority_sha256: str = AUTHORITY,
    authority_metadata=None,
) -> dict[str, object]:
    metadata = authority_metadata
    if metadata is None:
        rows = _authority_rows().copy()
        rows["cohort_authority_sha256"] = cohort_authority_sha256
        metadata = _bind_authority(
            rows,
            authority=cohort_authority_sha256,
        )
    elif (
        metadata.cohort_authority_sha256
        != cohort_authority_sha256
    ):
        raise AssertionError("test authority metadata mismatch")
    ledger = begin_prelock_metadata_ledger(metadata)
    return lock_preholdout_metadata_audit(
        ledger,
        authority_metadata=metadata,
    )


def _authority_rows_for_panel(
    panel: pd.DataFrame, *, authority: str
) -> pd.DataFrame:
    game_weeks = (
        panel[["game_id", "nfl_week"]]
        .drop_duplicates()
        .sort_values("game_id", kind="mergesort")
    )
    development = [
        (str(row.game_id), int(row.nfl_week))
        for row in game_weeks.itertuples(index=False)
    ]
    used_ids = {game_id for game_id, _ in development}
    padding_index = 0
    while len(development) < 153:
        game_id = f"development-padding-{padding_index:03d}"
        padding_index += 1
        if game_id not in used_ids:
            development.append(
                (game_id, (padding_index - 1) % 12 + 1)
            )
    rows: list[dict[str, object]] = []
    for index, (game_id, week) in enumerate(development):
        rows.append(
            {
                "game_id": game_id,
                "cohort": "development",
                "nfl_week": week,
                "kickoff_utc": (
                    f"2025-08-{index % 28 + 1:02d}T12:00:00Z"
                ),
                "batch_sha256": "sha256:" + f"{index + 1:064x}",
                "cohort_authority_sha256": authority,
                "market_reaction_exposure": "DEVELOPMENT_USED",
                "sports_outcome_exposure": "DEVELOPMENT_USED",
                "reaction_read_count": 0,
            }
        )
    for index in range(81):
        rows.append(
            {
                "game_id": f"holdout-genuine-{index:03d}",
                "cohort": "holdout",
                "nfl_week": index % 6 + 13,
                "kickoff_utc": (
                    f"2026-01-{index % 28 + 1:02d}T12:00:00Z"
                ),
                "batch_sha256": (
                    "sha256:" + f"{index + 10_000:064x}"
                ),
                "cohort_authority_sha256": authority,
                "market_reaction_exposure": "SEALED_UNREAD",
                "sports_outcome_exposure": "PRIOR_EXPOSED_X11",
                "reaction_read_count": 0,
            }
        )
    return pd.DataFrame(rows)


def _class_probabilities(
    direction: str, *, correct_probability: float
) -> tuple[float, float, float]:
    remainder = (1.0 - correct_probability) / 2.0
    values = {
        "DOWN": [correct_probability, remainder, remainder],
        "NO_MOVE": [remainder, correct_probability, remainder],
        "UP": [remainder, remainder, correct_probability],
    }
    return tuple(values[direction])


def _source_and_transport(
    *,
    catastrophic: bool = False,
    jointly_bad: bool = False,
    mixed_factor_bad_games: int = 0,
) -> X15ModelRun:
    metadata = _authority_rows().loc[
        lambda frame: frame["cohort"].eq("development")
    ]
    stage_b = _stage_b_selection()
    winner = stage_b.winner
    assert winner is not None
    frozen_anchor = (
        winner.paired_rows.loc[
            lambda frame: frame["landmark_seconds"].eq(3)
            & frame["endpoint_seconds"].eq(30)
            & frame["nfl_week"].isin((3, 4, 5, 6))
        ]
        .drop_duplicates("game_id")
        .head(36)
        .set_index("game_id")
    )
    evaluation_games = metadata.loc[
        metadata["game_id"].isin(frozen_anchor.index)
    ].sort_values(["nfl_week", "game_id"], kind="mergesort")
    assert len(evaluation_games) == 36
    source: list[dict[str, object]] = []
    transported: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    last_candidate_source: dict[str, object] | None = None
    for game_position, game in enumerate(
        evaluation_games.itertuples(index=False)
    ):
        frozen = frozen_anchor.loc[str(game.game_id)]
        train_weeks = tuple(frozen["train_weeks_candidate"])
        validation_weeks = tuple(
            frozen["validation_weeks_candidate"]
        )
        training_ids = tuple(frozen["training_game_ids_candidate"])
        validation_ids = tuple(
            frozen["validation_game_ids_candidate"]
        )
        fold_id = str(frozen["fold_id"])
        fold_character = fold_id[-1]
        s_truth = frozen["s_h_truth_candidate"]
        o_truth = frozen["o_h_given_s_truth_candidate"]
        direction = frozen["direction_truth_candidate"]
        native_binary = 0.82 if bool(s_truth) else 0.18
        transport_binary = 0.74 if bool(s_truth) else 0.26
        if catastrophic:
            transport_binary = 0.08 if bool(s_truth) else 0.92
        elif jointly_bad:
            transport_binary = 0.18 if bool(s_truth) else 0.82
            native_binary = 0.20 if bool(s_truth) else 0.80
        elif game_position < mixed_factor_bad_games:
            transport_binary = 0.45 if bool(s_truth) else 0.55
            native_binary = 0.47 if bool(s_truth) else 0.53
        native_observation = (
            0.82 if o_truth is not pd.NA and bool(o_truth) else 0.18
        )
        transport_observation = (
            0.74 if o_truth is not pd.NA and bool(o_truth) else 0.26
        )
        if catastrophic:
            transport_observation = (
                0.08 if o_truth is not pd.NA and bool(o_truth) else 0.92
            )
        elif jointly_bad:
            transport_observation = (
                0.18 if o_truth is not pd.NA and bool(o_truth) else 0.82
            )
            native_observation = (
                0.20 if o_truth is not pd.NA and bool(o_truth) else 0.80
            )
        elif game_position < mixed_factor_bad_games:
            transport_observation = (
                0.45 if o_truth is not pd.NA and bool(o_truth) else 0.55
            )
            native_observation = (
                0.47 if o_truth is not pd.NA and bool(o_truth) else 0.53
            )
        native_direction = (
            _class_probabilities(
                str(direction), correct_probability=0.82
            )
            if direction is not pd.NA
            else (0.2, 0.6, 0.2)
        )
        transport_direction = (
            _class_probabilities(
                str(direction),
                correct_probability=(
                    0.08
                    if catastrophic
                    else (
                        0.15
                        if jointly_bad
                        else (
                            0.45
                            if game_position < mixed_factor_bad_games
                            else 0.70
                        )
                    )
                ),
            )
            if direction is not pd.NA
            else (0.25, 0.5, 0.25)
        )
        if jointly_bad and direction is not pd.NA:
            native_direction = _class_probabilities(
                str(direction), correct_probability=0.17
            )
        elif (
            game_position < mixed_factor_bad_games
            and direction is not pd.NA
        ):
            native_direction = _class_probabilities(
                str(direction), correct_probability=0.47
            )
        identity = {
            "cohort_authority_sha256": AUTHORITY,
            "game_id": str(game.game_id),
            "nfl_week": int(game.nfl_week),
            "atomic_information_episode_id": (
                str(frozen["atomic_information_episode_id"])
            ),
            "landmark_seconds": 3,
            "endpoint_seconds": 30,
            "fold_id": fold_id,
            "train_weeks": train_weeks,
            "validation_weeks": validation_weeks,
            "training_game_ids": training_ids,
            "validation_game_ids": validation_ids,
            "preprocessor_fit_game_ids": tuple(
                frozen["preprocessor_fit_game_ids_candidate"]
            ),
            "calibrator_fit_game_ids_s_h": training_ids,
            "calibrator_fit_game_ids_o_h_given_s": training_ids,
            "calibrator_fit_game_ids_direction": training_ids,
            "model_id": "regularized_logistic_v1",
            "feature_block_id": "D1",
            "target_contract": TARGET_CONTRACT,
            "claim_boundary": CLAIM_BOUNDARY,
            "schema_version": "HistoricalTradesOnlyProbabilityPanelV2",
            "analysis_scope": (
                "HISTORICAL_TRADES_ONLY_SOURCE_TIME_DIAGNOSTIC"
            ),
            "claim_eligible": False,
            "direction_threshold_probability": 0.01,
            "direction_threshold_semantics": (
                "FIXED_CROSS_VENUE_RESEARCH_MATERIALITY_NOT_TICK"
            ),
            "venue_tick_support": "UNSUPPORTED",
            "market_continuity_support": "UNKNOWN",
            "s_h_truth": s_truth,
            "o_h_given_s_truth": o_truth,
            "direction_truth": direction,
            "training_data_sha256": (
                "sha256:" + fold_character * 64
            ),
            "preprocessor_training_sha256": "sha256:" + "b" * 64,
            "s_h_model_training_sha256": "sha256:" + "4" * 64,
            "o_h_given_s_model_training_sha256": (
                "sha256:" + "5" * 64
            ),
            "direction_model_training_sha256": (
                "sha256:" + "6" * 64
            ),
            "s_h_calibration_training_sha256": "sha256:" + "c" * 64,
            "o_h_given_s_calibration_training_sha256": (
                "sha256:" + "d" * 64
            ),
            "direction_calibration_training_sha256": (
                "sha256:" + "e" * 64
            ),
            "feature_block_sha256": "sha256:" + "7" * 64,
            "model_spec_sha256": "sha256:" + "8" * 64,
            "fold_sha256": "sha256:" + fold_character * 64,
        }
        candidate_source = {
            **identity,
            "source_row_id": frozen["source_row_id"],
            "actual_home_contract_id": frozen[
                "actual_home_contract_id"
            ],
            "venue": "polymarket",
            "training_venue": "polymarket",
            "calibration_venue": "polymarket",
            "transport_mode": "VENUE_SPECIFIC",
            "s_h_calibrated_probability": frozen[
                "s_h_calibrated_probability_candidate"
            ],
            "o_h_given_s_calibrated_probability": frozen[
                "o_h_given_s_calibrated_probability_candidate"
            ],
            "direction_calibrated_prob_down": frozen[
                "direction_calibrated_prob_down_candidate"
            ],
            "direction_calibrated_prob_no_move": frozen[
                "direction_calibrated_prob_no_move_candidate"
            ],
            "direction_calibrated_prob_up": frozen[
                "direction_calibrated_prob_up_candidate"
            ],
        }
        source.append(candidate_source)
        last_candidate_source = candidate_source
        transported.append(
            {
                **identity,
                "source_row_id": game_position + 1_000,
                "actual_home_contract_id": f"kalshi-{game.game_id}",
                "venue": "kalshi",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "NO_TARGET_RECALIBRATION",
                "s_h_calibrated_probability": transport_binary,
                "o_h_given_s_calibrated_probability": (
                    transport_observation
                ),
                "direction_calibrated_prob_down": transport_direction[0],
                "direction_calibrated_prob_no_move": (
                    transport_direction[1]
                ),
                "direction_calibrated_prob_up": transport_direction[2],
            }
        )
        native.append(
            {
                **identity,
                "source_row_id": game_position + 2_000,
                "actual_home_contract_id": f"kalshi-{game.game_id}",
                "venue": "kalshi",
                "training_venue": "kalshi",
                "calibration_venue": "kalshi",
                "transport_mode": "VENUE_SPECIFIC",
                "s_h_calibrated_probability": native_binary,
                "o_h_given_s_calibrated_probability": (
                    native_observation
                ),
                "direction_calibrated_prob_down": native_direction[0],
                "direction_calibrated_prob_no_move": native_direction[1],
                "direction_calibrated_prob_up": native_direction[2],
            }
        )
        b0_binary = 0.62 if bool(s_truth) else 0.38
        b0_observation = (
            0.62 if o_truth is not pd.NA and bool(o_truth) else 0.38
        )
        b0_direction = (
            _class_probabilities(
                str(direction), correct_probability=0.58
            )
            if direction is not pd.NA
            else (0.3, 0.4, 0.3)
        )
        b0_identity = {
            **identity,
            "model_id": "b0_empirical_v1",
            "feature_block_id": "D0",
            "feature_block_sha256": "sha256:" + "0" * 64,
            "model_spec_sha256": "sha256:" + "1" * 64,
        }
        source.append(
            {
                **b0_identity,
                "source_row_id": frozen["source_row_id"],
                "actual_home_contract_id": frozen[
                    "actual_home_contract_id"
                ],
                "venue": "polymarket",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "VENUE_SPECIFIC",
                "s_h_calibrated_probability": frozen[
                    "s_h_calibrated_probability_b0"
                ],
                "o_h_given_s_calibrated_probability": frozen[
                    "o_h_given_s_calibrated_probability_b0"
                ],
                "direction_calibrated_prob_down": frozen[
                    "direction_calibrated_prob_down_b0"
                ],
                "direction_calibrated_prob_no_move": frozen[
                    "direction_calibrated_prob_no_move_b0"
                ],
                "direction_calibrated_prob_up": frozen[
                    "direction_calibrated_prob_up_b0"
                ],
            }
        )
        transported.append(
            {
                **b0_identity,
                "source_row_id": game_position + 4_000,
                "actual_home_contract_id": f"kalshi-{game.game_id}",
                "venue": "kalshi",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "NO_TARGET_RECALIBRATION",
                "s_h_calibrated_probability": b0_binary,
                "o_h_given_s_calibrated_probability": b0_observation,
                "direction_calibrated_prob_down": b0_direction[0],
                "direction_calibrated_prob_no_move": b0_direction[1],
                "direction_calibrated_prob_up": b0_direction[2],
            }
        )
        native.append(
            {
                **b0_identity,
                "source_row_id": game_position + 5_000,
                "actual_home_contract_id": f"kalshi-{game.game_id}",
                "venue": "kalshi",
                "training_venue": "kalshi",
                "calibration_venue": "kalshi",
                "transport_mode": "VENUE_SPECIFIC",
                "s_h_calibrated_probability": b0_binary,
                "o_h_given_s_calibrated_probability": b0_observation,
                "direction_calibrated_prob_down": b0_direction[0],
                "direction_calibrated_prob_no_move": b0_direction[1],
                "direction_calibrated_prob_up": b0_direction[2],
            }
        )
    assert last_candidate_source is not None
    source_frame = pd.DataFrame(source)
    source_provenance_sha256 = (
        validation._transport_source_provenance_sha256(source_frame)
    )
    run_config = _run_config(
        source_provenance_sha256=source_provenance_sha256
    )
    return X15ModelRun(
        oof_predictions=pd.DataFrame(
            [*source, *transported, *native]
        ),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256=validation._canonical_sha256(run_config),
        run_config=run_config,
    )


def _replace_transported_candidate_probabilities(
    run: X15ModelRun,
    *,
    s_h_correct_probability: float | None = None,
    o_h_given_s_correct_probability: float | None = None,
    direction_correct_probability: float | None = None,
    native_correct_probability: float | None = None,
) -> X15ModelRun:
    """Return a run with one precisely targeted development score change."""

    predictions = run.oof_predictions.copy()
    candidate = (
        predictions["venue"].eq("kalshi")
        & predictions["training_venue"].eq("polymarket")
        & predictions["model_id"].eq("regularized_logistic_v1")
        & predictions["feature_block_id"].eq("D1")
    )
    native = (
        predictions["venue"].eq("kalshi")
        & predictions["training_venue"].eq("kalshi")
        & predictions["model_id"].eq("regularized_logistic_v1")
        & predictions["feature_block_id"].eq("D1")
    )

    def _binary_probability(
        truth: object, correct_probability: float
    ) -> float:
        return (
            correct_probability
            if bool(truth)
            else 1.0 - correct_probability
        )

    if s_h_correct_probability is not None:
        predictions.loc[
            candidate, "s_h_calibrated_probability"
        ] = predictions.loc[candidate, "s_h_truth"].map(
            lambda truth: _binary_probability(
                truth, s_h_correct_probability
            )
        )
    if o_h_given_s_correct_probability is not None:
        evaluable = candidate & predictions[
            "o_h_given_s_truth"
        ].notna()
        predictions.loc[
            evaluable, "o_h_given_s_calibrated_probability"
        ] = predictions.loc[evaluable, "o_h_given_s_truth"].map(
            lambda truth: _binary_probability(
                truth, o_h_given_s_correct_probability
            )
        )
    if direction_correct_probability is not None:
        evaluable = candidate & predictions["direction_truth"].notna()
        for row_index in predictions.index[evaluable]:
            down, no_move, up = _class_probabilities(
                str(predictions.at[row_index, "direction_truth"]),
                correct_probability=direction_correct_probability,
            )
            predictions.at[
                row_index, "direction_calibrated_prob_down"
            ] = down
            predictions.at[
                row_index, "direction_calibrated_prob_no_move"
            ] = no_move
            predictions.at[
                row_index, "direction_calibrated_prob_up"
            ] = up
    if native_correct_probability is not None:
        predictions.loc[
            native, "s_h_calibrated_probability"
        ] = predictions.loc[native, "s_h_truth"].map(
            lambda truth: _binary_probability(
                truth, native_correct_probability
            )
        )
        native_o = native & predictions["o_h_given_s_truth"].notna()
        predictions.loc[
            native_o, "o_h_given_s_calibrated_probability"
        ] = predictions.loc[native_o, "o_h_given_s_truth"].map(
            lambda truth: _binary_probability(
                truth, native_correct_probability
            )
        )
        native_direction = native & predictions["direction_truth"].notna()
        for row_index in predictions.index[native_direction]:
            down, no_move, up = _class_probabilities(
                str(predictions.at[row_index, "direction_truth"]),
                correct_probability=native_correct_probability,
            )
            predictions.at[
                row_index, "direction_calibrated_prob_down"
            ] = down
            predictions.at[
                row_index, "direction_calibrated_prob_no_move"
            ] = no_move
            predictions.at[
                row_index, "direction_calibrated_prob_up"
            ] = up
    return replace(run, oof_predictions=predictions)


def test_binds_authoritative_153_81_target_specific_exposure() -> None:
    metadata = _bind_authority()
    evidence = validation.load_frozen_historical_exposure_evidence()

    assert len(metadata.development) == 153
    assert len(metadata.holdout) == 81
    assert metadata.overlap_game_ids == ()
    assert metadata.holdout_reviewed_overlap_game_ids == ()
    assert metadata.holdout_reaction_overlap_game_ids == ()
    assert metadata.holdout_reaction_read_count == 0
    assert metadata.historical_reviewed_game_ids_sha256 == (
        evidence.reviewed_game_ids_sha256
    )
    assert metadata.historical_reaction_game_ids_sha256 == (
        evidence.reaction_game_ids_sha256
    )
    assert metadata.selection_authority.game_count == 153
    assert metadata.stage_a_outcome_validation_eligible is False
    assert metadata.sports_outcome_source_evidence_sha256 == (
        X11_SPORTS_OUTCOME_EVIDENCE_SHA256
    )
    assert (
        metadata.sports_outcome_observation_count
        == X11_HOLDOUT_DRIVE_OUTCOME_COUNT
    )
    assert metadata.holdout["market_reaction_exposure"].eq(
        "SEALED_UNREAD"
    ).all()
    assert metadata.holdout["sports_outcome_exposure"].eq(
        "PRIOR_EXPOSED_X11"
    ).all()
    assert {
        "HISTORICAL_REVIEWED_GAME_IDS",
        "HISTORICAL_REACTION_GAME_IDS",
        "SPORTS_OUTCOME_EXPOSED_GAME_IDS",
        "REACTION_BLIND_METADATA_GAME_IDS",
    }.issubset(set(metadata.prior_exposure_audit["evidence_kind"]))


def test_historical_exposure_is_derived_from_fixed_tracked_artifacts() -> None:
    evidence = validation.load_frozen_historical_exposure_evidence()

    assert evidence.reviewed_artifact_path.endswith(
        "6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
        ".holdout-selection.json"
    )
    assert evidence.reviewed_artifact_sha256 == (
        "sha256:"
        "1685ee5084d99670abeb623bba16c9dd310b972c1b7778c0a967d171dd679034"
    )
    assert len(evidence.reviewed_game_ids) == 20
    assert evidence.reaction_artifact_path.endswith(
        "a2aa20c764523d9c117e0087eb319cc50bbf736924009c9f7b8850fb471a3ea4"
        ".manifest.json"
    )
    assert evidence.reaction_artifact_sha256 == (
        "sha256:"
        "a2aa20c764523d9c117e0087eb319cc50bbf736924009c9f7b8850fb471a3ea4"
    )
    assert len(evidence.reaction_game_ids) == 153


def test_sealed_reaction_declaration_cannot_override_authoritative_overlap() -> None:
    evidence = validation.load_frozen_historical_exposure_evidence()
    rows = _authority_rows()
    rows.loc[
        rows.loc[rows["cohort"].eq("holdout")].index[0], "game_id"
    ] = evidence.reviewed_game_ids[0]
    with pytest.raises(
        KalshiValidationError,
        match="authoritative historical.*holdout.*overlap",
    ):
        _bind_authority(rows)

    parameters = inspect.signature(
        bind_frozen_authority_metadata
    ).parameters
    assert "historical_reviewed_game_ids" not in parameters
    assert "historical_reaction_game_ids" not in parameters


def test_sports_outcome_exposure_must_cover_exact_holdout_authority() -> None:
    with pytest.raises(
        KalshiValidationError,
        match="sports outcome.*exact 81-game holdout",
    ):
        _bind_authority(sports_outcome_exposed_game_ids=())


def test_transport_scores_kalshi_truth_against_native_kalshi_oof() -> None:
    result = _validate_transport(
        _source_and_transport(),
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    assert result.target_recalibration_applied is False
    assert result.unmatched_source_rows == 0
    assert result.unmatched_target_rows == 0
    assert result.native_comparator_paired_rows == 36
    assert result.paired_game_count == 36
    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is False
    assert result.transport_b0_improvement_gate_passed is True
    assert result.transport_b0_ci_low > 0.0
    assert result.transport_b0_bootstrap_samples == 10_000
    assert result.transport_b0_bootstrap_seed == 20260729
    assert not result.native_b0_score_summary.empty
    assert result.transport_gate_passed is True
    assert (
        result.diagnostic_status
        == "HISTORICAL_PREDICTIVE_UTILITY_CANDIDATE"
    )
    assert {"S_H", "O_H_GIVEN_S", "DIRECTION"} == set(
        result.score_summary["head"]
    )
    assert {"LOG_LOSS", "BRIER"}.issubset(
        set(result.score_summary["metric"])
    )
    assert {
        "transport_score",
        "native_kalshi_score",
        "score_degradation",
        "evaluated_game_count",
        "evaluated_row_count",
    }.issubset(result.score_summary.columns)
    assert not result.calibration_summary.empty


@pytest.mark.parametrize(
    ("degraded_head", "expected_failed_head"),
    (
        ("s_h", "S_H"),
        ("o_h_given_s", "O_H_GIVEN_S"),
    ),
)
def test_transport_requires_each_binary_nuisance_head_to_improve_b0(
    degraded_head: str,
    expected_failed_head: str,
) -> None:
    run = _replace_transported_candidate_probabilities(
        _source_and_transport(),
        s_h_correct_probability=(
            0.59 if degraded_head == "s_h" else None
        ),
        o_h_given_s_correct_probability=(
            0.59 if degraded_head == "o_h_given_s" else None
        ),
    )

    result = _validate_transport(
        run,
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    s_h = result.transport_b0_head_gate_summary.loc[
        lambda frame: frame["head"].eq("S_H")
        & frame["metric"].eq("LOG_LOSS")
    ].iloc[0]
    o_h = result.transport_b0_head_gate_summary.loc[
        lambda frame: frame["head"].eq("O_H_GIVEN_S")
        & frame["metric"].eq("LOG_LOSS")
    ].iloc[0]
    failed = s_h if expected_failed_head == "S_H" else o_h
    passed = o_h if expected_failed_head == "S_H" else s_h
    assert failed["overall_mean_improvement"] < 0.0
    assert bool(failed["gate_passed"]) is False
    assert passed["overall_mean_improvement"] > 0.0
    assert bool(passed["gate_passed"]) is True
    assert result.binary_nuisance_gate_passed is False
    assert result.transport_gate_passed is False


def test_transport_requires_direction_log_loss_and_brier_improvement() -> None:
    run = _replace_transported_candidate_probabilities(
        _source_and_transport(),
        direction_correct_probability=0.52,
    )

    result = _validate_transport(
        run,
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    direction = result.transport_b0_head_gate_summary.loc[
        lambda frame: frame["head"].eq("DIRECTION")
    ].set_index("metric")
    assert set(direction.index) == {"LOG_LOSS", "BRIER"}
    assert (
        direction["bootstrap_samples"].eq(10_000).all()
    )
    assert direction["bootstrap_seed"].eq(20260729).all()
    assert direction["overall_ci_low"].lt(0.0).all()
    assert not direction["gate_passed"].any()
    assert result.direction_gate_passed is False
    assert result.transport_gate_passed is False


def test_direction_log_loss_improvement_cannot_mask_brier_failure() -> None:
    run = _source_and_transport()
    predictions = run.oof_predictions.copy()
    candidate = (
        predictions["venue"].eq("kalshi")
        & predictions["training_venue"].eq("polymarket")
        & predictions["model_id"].eq("regularized_logistic_v1")
        & predictions["direction_truth"].notna()
    )
    columns = (
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    )
    class_index = {"DOWN": 0, "NO_MOVE": 1, "UP": 2}
    for row_index in predictions.index[candidate]:
        truth_index = class_index[
            str(predictions.at[row_index, "direction_truth"])
        ]
        values = [0.001, 0.001, 0.001]
        values[truth_index] = 0.59
        values[next(index for index in range(3) if index != truth_index)] = (
            0.409
        )
        predictions.loc[row_index, list(columns)] = values
    result = _validate_transport(
        replace(run, oof_predictions=predictions),
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    direction = result.transport_b0_head_gate_summary.loc[
        lambda frame: frame["head"].eq("DIRECTION")
    ].set_index("metric")
    assert direction.loc["LOG_LOSS", "overall_ci_low"] > 0.0
    assert bool(direction.loc["LOG_LOSS", "gate_passed"]) is True
    assert direction.loc["BRIER", "overall_ci_low"] < 0.0
    assert bool(direction.loc["BRIER", "gate_passed"]) is False
    assert result.direction_gate_passed is False


def test_direction_brier_improvement_cannot_mask_log_loss_failure() -> None:
    run = _source_and_transport()
    predictions = run.oof_predictions.copy()
    transported_b0 = (
        predictions["venue"].eq("kalshi")
        & predictions["training_venue"].eq("polymarket")
        & predictions["model_id"].eq("b0_empirical_v1")
        & predictions["direction_truth"].notna()
    )
    candidate = (
        predictions["venue"].eq("kalshi")
        & predictions["training_venue"].eq("polymarket")
        & predictions["model_id"].eq("regularized_logistic_v1")
        & predictions["direction_truth"].notna()
    )
    columns = (
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    )
    class_index = {"DOWN": 0, "NO_MOVE": 1, "UP": 2}
    for row_index in predictions.index[transported_b0]:
        truth_index = class_index[
            str(predictions.at[row_index, "direction_truth"])
        ]
        values = [0.001, 0.001, 0.001]
        values[truth_index] = 0.60
        values[next(index for index in range(3) if index != truth_index)] = (
            0.399
        )
        predictions.loc[row_index, list(columns)] = values
    for row_index in predictions.index[candidate]:
        truth_index = class_index[
            str(predictions.at[row_index, "direction_truth"])
        ]
        values = [0.205, 0.205, 0.205]
        values[truth_index] = 0.59
        predictions.loc[row_index, list(columns)] = values
    result = _validate_transport(
        replace(run, oof_predictions=predictions),
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    direction = result.transport_b0_head_gate_summary.loc[
        lambda frame: frame["head"].eq("DIRECTION")
    ].set_index("metric")
    assert direction.loc["LOG_LOSS", "overall_ci_low"] < 0.0
    assert bool(direction.loc["LOG_LOSS", "gate_passed"]) is False
    assert direction.loc["BRIER", "overall_ci_low"] > 0.0
    assert bool(direction.loc["BRIER", "gate_passed"]) is True
    assert result.direction_gate_passed is False


def test_transport_anchor_requires_thirty_evaluable_games() -> None:
    run = _source_and_transport()
    dropped_games = frozenset(
        sorted(run.oof_predictions["game_id"].unique())[29:36]
    )
    predictions = run.oof_predictions.loc[
        ~(
            run.oof_predictions["venue"].eq("kalshi")
            & run.oof_predictions["game_id"].isin(dropped_games)
        )
    ].copy()
    reduced = replace(run, oof_predictions=predictions)

    result = _validate_transport(
        reduced,
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    assert result.transport_b0_anchor_game_count == 29
    assert result.transport_b0_improvement_gate_passed is False
    assert result.transport_gate_passed is False


def test_native_kalshi_noninferiority_is_headwise_bootstrapped() -> None:
    passing = _validate_transport(
        _source_and_transport(),
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )
    assert passing.native_noninferiority_gate_passed is True
    assert len(passing.native_noninferiority_summary) == 6
    assert passing.native_noninferiority_summary[
        "bootstrap_samples"
    ].eq(10_000).all()
    assert passing.native_noninferiority_summary[
        "bootstrap_seed"
    ].eq(20260729).all()

    failing_run = _replace_transported_candidate_probabilities(
        _source_and_transport(),
        native_correct_probability=0.999,
    )
    failing = _validate_transport(
        failing_run,
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )
    assert failing.native_noninferiority_gate_passed is False
    assert failing.transport_gate_passed is False
    assert failing.diagnostic_status == (
        "HISTORICAL_PREDICTIVE_UTILITY_REJECTED"
    )
    assert (
        failing.native_noninferiority_summary[
            "upper95_within_margin"
        ]
        .eq(False)
        .any()
    )


def test_transport_rejects_joint_source_target_stage_b_provenance_drift() -> None:
    run = _source_and_transport()
    predictions = run.oof_predictions.copy()
    selected = (
        predictions["model_id"].eq("regularized_logistic_v1")
        & predictions["feature_block_id"].eq("D1")
        & predictions["training_venue"].eq("polymarket")
    )
    predictions.loc[
        selected, "preprocessor_training_sha256"
    ] = "sha256:" + "f" * 64
    drifted = replace(run, oof_predictions=predictions)

    with pytest.raises(
        KalshiValidationError,
        match="independently verified exact45 winner provenance",
    ):
        _validate_transport(
            drifted,
            authority_metadata=_bind_authority(),
            stage_b_selection=_stage_b_selection(),
        )


def test_transport_rejects_coherent_rewrite_with_recomputed_digests() -> None:
    run = _source_and_transport()
    predictions = run.oof_predictions.copy()
    selected = (
        predictions["model_id"].eq("regularized_logistic_v1")
        & predictions["feature_block_id"].eq("D1")
        & predictions["training_venue"].eq("polymarket")
    )
    predictions.loc[
        selected, "preprocessor_training_sha256"
    ] = "sha256:" + "f" * 64
    source = predictions.loc[
        predictions["venue"].eq("polymarket")
        & predictions["training_venue"].eq("polymarket")
    ]
    rewritten_config = dict(run.run_config)
    rewritten_config["stage_b_source_provenance_sha256"] = (
        validation._transport_source_provenance_sha256(source)
    )
    rewritten = replace(
        run,
        oof_predictions=predictions,
        run_config=rewritten_config,
        run_config_sha256=validation._canonical_sha256(
            rewritten_config
        ),
    )

    with pytest.raises(
        KalshiValidationError,
        match="independently verified exact45 winner provenance",
    ):
        _validate_transport(
            rewritten,
            authority_metadata=_bind_authority(),
            stage_b_selection=_stage_b_selection(),
        )


def test_transport_rejects_native_b0_population_attrition() -> None:
    run = _source_and_transport()
    predictions = run.oof_predictions.copy()
    native = (
        predictions["venue"].eq("kalshi")
        & predictions["training_venue"].eq("kalshi")
    )
    missing_game = str(predictions.loc[native, "game_id"].iloc[0])
    predictions = predictions.drop(
        predictions.index[
            native & predictions["game_id"].eq(missing_game)
        ]
    )
    drifted = replace(run, oof_predictions=predictions)

    with pytest.raises(
        KalshiValidationError,
        match="identical Kalshi truth/identity population",
    ):
        _validate_transport(
            drifted,
            authority_metadata=_bind_authority(),
            stage_b_selection=_stage_b_selection(),
        )


def test_transport_rejects_self_reported_noncanonical_run_config_sha() -> None:
    run = _source_and_transport()
    drifted_config = dict(run.run_config)
    drifted_config["unbound_extra_field"] = "MUTATED"
    drifted = replace(run, run_config=drifted_config)

    with pytest.raises(
        KalshiValidationError,
        match="run_config_sha256 does not bind",
    ):
        _validate_transport(
            drifted,
            authority_metadata=_bind_authority(),
            stage_b_selection=_stage_b_selection(),
        )


def test_transport_requires_frozen_survival_probability_contract() -> None:
    run = _source_and_transport()
    drifted_config = dict(run.run_config)
    drifted_config.pop("survival_probability_contract")
    drifted = replace(
        run,
        run_config=drifted_config,
        run_config_sha256=validation._canonical_sha256(drifted_config),
    )

    with pytest.raises(
        KalshiValidationError,
        match="survival_probability_contract",
    ):
        _validate_transport(
            drifted,
            authority_metadata=_bind_authority(),
            stage_b_selection=_stage_b_selection(),
        )


def test_transport_rejects_legacy_probability_panel_v1() -> None:
    run = _source_and_transport()
    legacy_predictions = run.oof_predictions.copy()
    legacy_predictions["schema_version"] = (
        "HistoricalTradesOnlyProbabilityPanelV1"
    )
    legacy_config = dict(run.run_config)
    legacy_config["schema_version"] = (
        "HistoricalTradesOnlyProbabilityPanelV1"
    )
    legacy_run = X15ModelRun(
        oof_predictions=legacy_predictions,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=validation._canonical_sha256(legacy_config),
        run_config=legacy_config,
    )

    with pytest.raises(
        KalshiValidationError,
        match="run_config drifted on schema_version",
    ):
        _validate_transport(
            legacy_run,
            authority_metadata=_bind_authority(),
            stage_b_selection=_stage_b_selection(),
        )


def test_catastrophic_transport_degradation_fails_candidate_gate() -> None:
    result = _validate_transport(
        _source_and_transport(catastrophic=True),
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is True
    assert result.transport_gate_passed is False
    assert (
        result.diagnostic_status
        == "HISTORICAL_PREDICTIVE_UTILITY_REJECTED"
    )


def test_jointly_bad_transport_and_native_cannot_beat_frozen_b0() -> None:
    result = _validate_transport(
        _source_and_transport(jointly_bad=True),
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )

    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is False
    assert result.transport_b0_mean_improvement < 0.0
    assert result.transport_b0_improvement_gate_passed is False
    assert result.transport_gate_passed is False
    assert (
        result.diagnostic_status
        == "HISTORICAL_PREDICTIVE_UTILITY_REJECTED"
    )


def _exact_factor_inputs(
    *,
    shared: bool,
    transport_run: X15ModelRun | None = None,
    pair_count: int = 36,
    kalshi_rows: pd.DataFrame | None = None,
) -> tuple[
    object,
    object,
    validation.FrozenFactorMembershipEvidence,
    validation.FrozenAuthorityMetadata,
]:
    evidence = validation.load_frozen_factor_membership_evidence()
    original_authority = _complete_selection_authority()
    original_game_ids = tuple(
        game[0] for game in original_authority.development_games
    )
    complete_pass = (
        evidence.membership_rows.loc[
            lambda frame: frame["factor_id"].eq(
                "NFL.EVENT.COMPLETE_PASS"
            )
            & frame["factor_version"].eq("v1"),
            ["game_id", "atomic_information_episode_id"],
        ]
        .sort_values(
            ["game_id", "atomic_information_episode_id"],
            kind="mergesort",
        )
        .drop_duplicates()
    )
    real_game_ids = tuple(
        sorted(complete_pass["game_id"].unique().tolist())
    )
    assert len(real_game_ids) == len(original_game_ids) == 153
    game_id_map = dict(
        zip(original_game_ids, real_game_ids, strict=True)
    )
    episodes_by_game = {
        str(game_id): tuple(
            frame["atomic_information_episode_id"].head(2).astype(str)
        )
        for game_id, frame in complete_pass.groupby(
            "game_id", sort=True
        )
    }
    assert all(
        len(episodes) == 2
        for episodes in episodes_by_game.values()
    )
    episode_id_map = {
        (original_game_id, f"{original_game_id}:episode-{position}"): (
            episodes_by_game[real_game_id][position]
        )
        for original_game_id, real_game_id in game_id_map.items()
        for position in (0, 1)
    }
    authority_frame = pd.DataFrame(
        [
            {
                "game_id": game_id_map[game_id],
                "nfl_week": nfl_week,
                "kickoff_utc": kickoff_utc,
                "batch_sha256": batch_sha256,
                "cohort_authority_sha256": (
                    evidence.cohort_authority_sha256
                ),
            }
            for (
                game_id,
                nfl_week,
                kickoff_utc,
                batch_sha256,
            ) in original_authority.development_games
        ]
    )
    membership_authority = bind_frozen_development_authority(
        authority_frame,
        cohort_authority_sha256=evidence.cohort_authority_sha256,
    )
    provenance_columns = (
        "training_game_ids",
        "validation_game_ids",
        "preprocessor_fit_game_ids",
    )
    membership_runs: list[X15ModelRun] = []
    for run in _stage_b_v3_runs():
        predictions = run.oof_predictions.copy()
        original_prediction_game_ids = predictions["game_id"].astype(
            str
        )
        original_episode_ids = predictions[
            "atomic_information_episode_id"
        ].astype(str)
        predictions["game_id"] = original_prediction_game_ids.map(
            game_id_map
        )
        predictions["atomic_information_episode_id"] = [
            episode_id_map[(game_id, episode_id)]
            for game_id, episode_id in zip(
                original_prediction_game_ids,
                original_episode_ids,
                strict=True,
            )
        ]
        predictions["cohort_authority_sha256"] = (
            evidence.cohort_authority_sha256
        )
        for column in provenance_columns:
            predictions[column] = predictions[column].map(
                lambda game_ids: tuple(
                    sorted(game_id_map[str(game_id)] for game_id in game_ids)
                )
            )
        run_config = dict(run.run_config)
        run_config["cohort_authority_sha256"] = (
            evidence.cohort_authority_sha256
        )
        membership_runs.append(
            replace(
                run,
                oof_predictions=predictions,
                run_config=run_config,
                run_config_sha256=validation._canonical_sha256(
                    run_config
                ),
            )
        )
    stage_b_selection = select_stage_b_v3_winner(
        tuple(membership_runs),
        authority=membership_authority,
    )
    development_metadata = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "cohort": "development",
                "nfl_week": nfl_week,
                "kickoff_utc": kickoff_utc,
                "batch_sha256": batch_sha256,
                "cohort_authority_sha256": (
                    evidence.cohort_authority_sha256
                ),
                "market_reaction_exposure": "DEVELOPMENT_USED",
                "sports_outcome_exposure": "DEVELOPMENT_USED",
                "reaction_read_count": 0,
            }
            for (
                game_id,
                nfl_week,
                kickoff_utc,
                batch_sha256,
            ) in membership_authority.development_games
        ]
    )
    holdout_metadata = _authority_rows().loc[
        lambda frame: frame["cohort"].eq("holdout")
    ].copy()
    holdout_metadata["cohort_authority_sha256"] = (
        evidence.cohort_authority_sha256
    )
    factor_authority_metadata = _bind_authority(
        pd.concat(
            [development_metadata, holdout_metadata],
            ignore_index=True,
        ),
        authority=evidence.cohort_authority_sha256,
    )
    selection = stage_b_selection.winner
    assert selection is not None
    raw_transport_run = transport_run or _source_and_transport()
    transport_predictions = raw_transport_run.oof_predictions.copy()
    candidate_mask = ~(
        transport_predictions["model_id"].eq("b0_empirical_v1")
        & transport_predictions["feature_block_id"].eq("D0")
    )
    transport_predictions.loc[candidate_mask, "model_id"] = (
        selection.spec.candidate_model_id
    )
    transport_predictions.loc[candidate_mask, "feature_block_id"] = (
        selection.spec.candidate_feature_block_id
    )
    stage_b_transport_run = X15ModelRun(
        oof_predictions=transport_predictions,
        conditional_quantiles=raw_transport_run.conditional_quantiles,
        fold_metrics=raw_transport_run.fold_metrics,
        support_audit=raw_transport_run.support_audit,
        weight_audit=raw_transport_run.weight_audit,
        run_config_sha256=raw_transport_run.run_config_sha256,
        run_config=raw_transport_run.run_config,
    )
    transport = _validate_transport(
        stage_b_transport_run,
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )
    poly_rows = (
        selection.paired_rows.loc[
            lambda frame: frame["landmark_seconds"].eq(3)
            & frame["endpoint_seconds"].eq(30)
        ]
        .sort_values(
            ["game_id", "atomic_information_episode_id"],
            kind="mergesort",
        )
        .drop_duplicates("game_id")
        .head(pair_count)
        .reset_index(drop=True)
    )
    poly_identities = poly_rows[
        ["game_id", "atomic_information_episode_id"]
    ].copy()
    available_kalshi_identities = (
        complete_pass.loc[
            ~complete_pass["game_id"].isin(poly_rows["game_id"]),
            ["game_id", "atomic_information_episode_id"],
        ]
        .drop_duplicates("game_id")
        .head(pair_count)
        .reset_index(drop=True)
    )
    kalshi_identities = (
        poly_identities
        if shared
        else available_kalshi_identities
    )
    target_rows = (
        transport.transport_b0_pair_diagnostics.head(pair_count)
        if kalshi_rows is None
        else kalshi_rows.copy()
    ).reset_index(drop=True)
    target_rows["candidate_model_id"] = (
        selection.spec.candidate_model_id
    )
    target_rows["candidate_feature_block_id"] = (
        selection.spec.candidate_feature_block_id
    )
    assert len(poly_rows) == pair_count
    assert len(target_rows) == pair_count
    poly_rows["game_id"] = poly_identities["game_id"]
    poly_rows["atomic_information_episode_id"] = poly_identities[
        "atomic_information_episode_id"
    ]
    target_rows["game_id"] = kalshi_identities["game_id"]
    target_rows["atomic_information_episode_id"] = kalshi_identities[
        "atomic_information_episode_id"
    ]
    if shared:
        target_rows["nfl_week"] = poly_rows["nfl_week"]
        target_rows["fold_id"] = poly_rows["fold_id"]
    transport = replace(
        transport,
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        transport_b0_pair_diagnostics=target_rows,
    )
    return (
        stage_b_selection,
        transport,
        evidence,
        factor_authority_metadata,
    )


def test_factor_membership_is_bound_to_fixed_registry_and_artifact() -> None:
    evidence = validation.load_frozen_factor_membership_evidence()

    assert evidence.authority_manifest_file_sha256 == (
        "sha256:"
        "39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3"
    )
    assert evidence.facts_authority_manifest_file_sha256 == (
        "sha256:"
        "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    )
    assert evidence.registry_file_sha256 == (
        "sha256:"
        "92e5001d92afa0748731b5310dae8289ff6930b26a141e981ed910d2c761575f"
    )
    assert evidence.registry_semantic_sha256 == (
        "sha256:"
        "527a084317ec4a728e5567feea756c1541b65bb814fcf96900b6cfbfd223ead8"
    )
    assert evidence.membership_rows_sha256 == (
        validation._factor_membership_rows_sha256(
            evidence.membership_rows
        )
    )
    assert evidence.membership_rows["game_id"].nunique() == 153
    assert (
        evidence.membership_rows[
            ["game_id", "atomic_information_episode_id"]
        ]
        .drop_duplicates()
        .shape[0]
        == 25_408
    )
    assert len(evidence.membership_rows) == 83_659
    assert evidence.membership_rows_sha256 == (
        "sha256:"
        "d2fc72c3d81720bcf2bf2a7550272f734544923abbfec72d2165e12cc634a874"
    )
    assert evidence.membership_artifact_bindings_sha256 == (
        "sha256:"
        "0725380c27e0353a0f6c92bef482b72b757970981cf6c31102f93fb6c64047c4"
    )
    parameters = inspect.signature(
        validation.build_cross_venue_predictive_utility_shortlist
    ).parameters
    assert "factor_membership" not in parameters
    assert "factor_membership_evidence" in parameters
    assert "preholdout_metadata_lock" in parameters
    assert "authority_metadata" in parameters

    selection, transport, _, authority_metadata = _exact_factor_inputs(
        shared=True
    )
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )
    tampered = replace(
        evidence,
        authority_batch_sha256="sha256:" + "0" * 64,
    )
    with pytest.raises(
        KalshiValidationError,
        match="does not bind the frozen artifacts",
    ):
        validation.build_cross_venue_predictive_utility_shortlist(
            selection,
            transport_validation=transport,
            factor_membership_evidence=tampered,
            authority_metadata=authority_metadata,
            preholdout_metadata_lock=metadata_lock,
        )

    dropped_rows = evidence.membership_rows.iloc[:-1].copy()
    self_reported = replace(
        evidence,
        membership_rows=dropped_rows,
        membership_rows_sha256=(
            validation._factor_membership_rows_sha256(dropped_rows)
        ),
    )
    with pytest.raises(
        KalshiValidationError,
        match="does not bind the frozen artifacts",
    ):
        validation.build_cross_venue_predictive_utility_shortlist(
            selection,
            transport_validation=transport,
            factor_membership_evidence=self_reported,
            authority_metadata=authority_metadata,
            preholdout_metadata_lock=metadata_lock,
        )


def test_factor_authority_contracts_reject_v3_registry_and_v1_panel() -> None:
    project_root = Path(__file__).resolve().parents[2]
    legacy_registry = json.loads(
        (
            project_root
            / "registries/factors/nfl_factor_registry_v3_draft.json"
        ).read_text(encoding="utf-8")
    )
    legacy_panel = json.loads(
        (
            project_root
            / "artifacts/market-observation/nfl/x15/"
            "historical-trades-only-development-panel-v1/batches/"
            "manifests/sha256/3d/"
            "3d2247c8b075748ccfa219daaf760e1681e85cb8fe0601bc2c9657381c7a969e"
            ".batch-index.json"
        ).read_text(encoding="utf-8")
    )

    with pytest.raises(
        KalshiValidationError,
        match="registry semantic contract drifted",
    ):
        validation._validate_factor_registry_contract(legacy_registry)
    with pytest.raises(
        KalshiValidationError,
        match="membership authority contract drifted",
    ):
        validation._validate_factor_membership_authority_contract(
            legacy_panel
        )


def test_factor_authority_contracts_bind_exact_v4_facts_and_v2_panel() -> None:
    project_root = Path(__file__).resolve().parents[2]
    registry = json.loads(
        (
            project_root / "registries/factors/nfl_factor_registry_v4.json"
        ).read_text(encoding="utf-8")
    )
    facts = json.loads(
        (
            project_root
            / "artifacts/market-observation/nfl/x13/exact-153-facts-v4/"
            "batches/manifests/sha256/5d/"
            "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
            ".batch-index.json"
        ).read_text(encoding="utf-8")
    )
    panel = json.loads(
        (
            project_root
            / "artifacts/market-observation/nfl/x15/"
            "historical-trades-only-development-panel-v2/batches/"
            "manifests/sha256/39/"
            "39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3"
            ".batch-index.json"
        ).read_text(encoding="utf-8")
    )

    validation._validate_factor_registry_contract(registry)
    validation._validate_factor_facts_authority_contract(facts)
    validation._validate_factor_membership_authority_contract(panel)
    assert panel["source_batch_file_sha256s"]["facts"] == (
        "sha256:"
        "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    )
    assert facts["factor_registry"]["semantic_sha256"] == (
        "sha256:"
        "527a084317ec4a728e5567feea756c1541b65bb814fcf96900b6cfbfd223ead8"
    )


def test_kalshi_factor_validation_accepts_only_stage_b_v3_winner() -> None:
    transport_parameters = inspect.signature(
        validation.validate_development_venue_transport
    ).parameters
    assert "stage_b_selection" in transport_parameters
    assert "stage_b_batch_manifest_path" in transport_parameters
    assert "stage_b_artifact_root" in transport_parameters
    assert "spec" not in transport_parameters

    legacy_single_candidate = select_candidate_against_b0(
        _complete_selection_run(),
        spec=_complete_selection_spec(),
        authority=_complete_selection_authority(),
    )
    with pytest.raises(
        KalshiValidationError,
        match="Stage-B V3",
    ):
        validation._require_frozen_stage_b_v3_winner(
            legacy_single_candidate,
            authority=_complete_selection_authority(),
        )

    stage_b = select_stage_b_v3_winner(
        _stage_b_v3_runs(),
        authority=_complete_selection_authority(),
    )
    assert stage_b.winner is not None
    assert (
        validation._require_frozen_stage_b_v3_winner(
            stage_b,
            authority=_complete_selection_authority(),
        )
        is stage_b.winner
    )
    assert stage_b.winner.spec.selection_venue == "polymarket"


@pytest.mark.parametrize(
    "mutation",
    ("paired_rows", "promotion_metric"),
)
def test_kalshi_rejects_stage_b_candidate_evidence_mutation(
    mutation: str,
) -> None:
    stage_b = select_stage_b_v3_winner(
        _stage_b_v3_runs(),
        authority=_complete_selection_authority(),
    )
    winner = stage_b.winner
    assert winner is not None
    mutated_winner = (
        replace(
            winner,
            paired_rows=winner.paired_rows.iloc[:-1].copy(),
        )
        if mutation == "paired_rows"
        else replace(
            winner,
            integrated_mean_improvement=(
                winner.integrated_mean_improvement + 1.0
            ),
        )
    )
    mutated_results = tuple(
        mutated_winner if result is winner else result
        for result in stage_b.candidate_results
    )
    mutated = replace(
        stage_b,
        winner=mutated_winner,
        candidate_results=mutated_results,
    )

    with pytest.raises(KalshiValidationError, match="Stage-B V3"):
        validation._require_frozen_stage_b_v3_winner(
            mutated,
            authority=_complete_selection_authority(),
        )


def test_exact_paired_factor_rows_pass_dual_venue_gate() -> None:
    (
        selection,
        transport,
        evidence,
        authority_metadata,
    ) = _exact_factor_inputs(shared=True)
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )

    shortlist = validation.build_cross_venue_predictive_utility_shortlist(
        selection,
        transport_validation=transport,
        factor_membership_evidence=evidence,
        authority_metadata=authority_metadata,
        preholdout_metadata_lock=metadata_lock,
        min_support_games=30,
        min_support_episodes=20,
    )

    row = shortlist.loc[
        shortlist["factor_id"].eq("NFL.EVENT.COMPLETE_PASS")
    ].iloc[0]
    assert row["shared_pair_row_count"] == 36
    assert row["shared_game_count"] == 36
    assert row["shared_episode_count"] == 36
    assert row["polymarket_only_pair_row_count"] == 218
    assert row["kalshi_only_pair_row_count"] == 0
    assert bool(
        row["polymarket_predictive_utility_gate_passed"]
    ) is True
    assert bool(row["kalshi_predictive_utility_gate_passed"]) is True
    assert bool(row["cross_venue_both_positive"]) is True
    assert bool(
        row["factor_conditioned_dual_venue_gate_passed"]
    ) is True
    assert row["shortlist_schema_version"] == (
        "NFLFactorPredictiveUtilityShortlistV1"
    )
    assert row["diagnostic_status"] == (
        "HISTORICAL_PREDICTIVE_UTILITY_CANDIDATE"
    )
    forbidden = (
        "gross_markout",
        "mean_effect",
        "factor_effect",
        "signed_reaction",
        "alpha",
        "signal",
    )
    assert not any(
        token in str(column).lower()
        for column in shortlist.columns
        for token in forbidden
    )


def test_disjoint_positive_factor_samples_cannot_enter_shortlist() -> None:
    (
        selection,
        transport,
        evidence,
        authority_metadata,
    ) = _exact_factor_inputs(shared=False)
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )

    shortlist = validation.build_cross_venue_predictive_utility_shortlist(
        selection,
        transport_validation=transport,
        factor_membership_evidence=evidence,
        authority_metadata=authority_metadata,
        preholdout_metadata_lock=metadata_lock,
        min_support_games=30,
        min_support_episodes=20,
    )

    row = shortlist.loc[
        shortlist["factor_id"].eq("NFL.EVENT.COMPLETE_PASS")
    ].iloc[0]
    assert bool(row["global_transport_gate_passed"]) is True
    assert row["diagnostic_status"] == (
        "HISTORICAL_PREDICTIVE_UTILITY_REJECTED"
    )
    assert row["shared_pair_row_count"] == 0
    assert row["shared_game_count"] == 0
    assert row["shared_episode_count"] == 0
    assert row["polymarket_only_pair_row_count"] == 254
    assert row["kalshi_only_pair_row_count"] == 36
    assert bool(
        row["factor_conditioned_dual_venue_gate_passed"]
    ) is False


@pytest.mark.parametrize(
    ("min_support_games", "min_support_episodes"),
    ((29, 20), (30, 19), (31, 20), (30, 21)),
)
def test_factor_conditioned_support_floors_are_not_overridable(
    min_support_games: int,
    min_support_episodes: int,
) -> None:
    (
        selection,
        transport,
        evidence,
        authority_metadata,
    ) = _exact_factor_inputs(shared=True)
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )

    with pytest.raises(
        KalshiValidationError,
        match="support contract is frozen",
    ):
        validation.build_cross_venue_predictive_utility_shortlist(
            selection,
            transport_validation=transport,
            factor_membership_evidence=evidence,
            authority_metadata=authority_metadata,
            preholdout_metadata_lock=metadata_lock,
            min_support_games=min_support_games,
            min_support_episodes=min_support_episodes,
        )


def test_factor_shortlist_requires_verified_current_metadata_lock() -> None:
    (
        selection,
        transport,
        evidence,
        authority_metadata,
    ) = _exact_factor_inputs(shared=True)
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )
    forged = dict(metadata_lock)
    forged["holdout_reaction_read_count"] = 1

    with pytest.raises(
        KalshiValidationError,
        match="verified pre-holdout metadata lock",
    ):
        validation.build_cross_venue_predictive_utility_shortlist(
            selection,
            transport_validation=transport,
            factor_membership_evidence=evidence,
            authority_metadata=authority_metadata,
            preholdout_metadata_lock=forged,
        )


def test_factor_shortlist_rejects_post_lock_reaction_read_mutation() -> None:
    (
        selection,
        transport,
        evidence,
        authority_metadata,
    ) = _exact_factor_inputs(shared=True)
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )
    first_holdout_index = authority_metadata.holdout.index[0]
    authority_metadata.holdout.loc[
        first_holdout_index, "reaction_read_count"
    ] = 1

    with pytest.raises(
        KalshiValidationError,
        match="verified pre-holdout metadata lock",
    ):
        validation.build_cross_venue_predictive_utility_shortlist(
            selection,
            transport_validation=transport,
            factor_membership_evidence=evidence,
            authority_metadata=authority_metadata,
            preholdout_metadata_lock=metadata_lock,
        )


def test_global_transport_gate_cannot_replace_bad_kalshi_factor_gate() -> None:
    mixed_run = _source_and_transport()
    initial_transport = _validate_transport(
        mixed_run,
        authority_metadata=_bind_authority(),
        stage_b_selection=_stage_b_selection(),
    )
    negative_rows = (
        initial_transport.transport_b0_pair_diagnostics.head(30).copy()
    )
    negative_rows["loss_improvement"] = -1.0
    (
        selection,
        transport,
        evidence,
        authority_metadata,
    ) = _exact_factor_inputs(
        shared=True,
        transport_run=mixed_run,
        pair_count=30,
        kalshi_rows=negative_rows,
    )
    assert transport.transport_gate_passed is True
    metadata_lock = _verified_preholdout_lock(
        cohort_authority_sha256=evidence.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )

    shortlist = validation.build_cross_venue_predictive_utility_shortlist(
        selection,
        transport_validation=transport,
        factor_membership_evidence=evidence,
        authority_metadata=authority_metadata,
        preholdout_metadata_lock=metadata_lock,
        min_support_games=30,
        min_support_episodes=20,
    )

    row = shortlist.loc[
        shortlist["factor_id"].eq("NFL.EVENT.COMPLETE_PASS")
    ].iloc[0]
    assert row["shared_pair_row_count"] == 30
    assert row["kalshi_mean_log_score_improvement"] < 0.0
    assert bool(row["global_transport_gate_passed"]) is True
    assert bool(
        row["factor_conditioned_dual_venue_gate_passed"]
    ) is False
    assert row["diagnostic_status"] != (
        "HISTORICAL_PREDICTIVE_UTILITY_CANDIDATE"
    )


def test_transport_rejects_target_recalibration_and_provenance_drift() -> None:
    run = _source_and_transport()
    metadata = _bind_authority()
    target_mask = run.oof_predictions["transport_mode"].eq(
        "NO_TARGET_RECALIBRATION"
    )

    recalibrated = run.oof_predictions.copy()
    recalibrated.loc[target_mask, "calibration_venue"] = "kalshi"
    recalibrated_run = X15ModelRun(
        oof_predictions=recalibrated,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(
        KalshiValidationError, match="genuine Polymarket source OOF"
    ):
        _validate_transport(
            recalibrated_run,
            authority_metadata=metadata,
            stage_b_selection=_stage_b_selection(),
        )

    drifted = run.oof_predictions.copy()
    drifted.loc[
        drifted[target_mask].index[0],
        "preprocessor_training_sha256",
    ] = "sha256:" + "0" * 64
    drifted_run = X15ModelRun(
        oof_predictions=drifted,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(
        KalshiValidationError,
        match="source-only preprocessor_training_sha256",
    ):
        _validate_transport(
            drifted_run,
            authority_metadata=metadata,
            stage_b_selection=_stage_b_selection(),
        )


def test_prelock_ledger_remains_metadata_only() -> None:
    metadata = _bind_authority()
    ledger = begin_prelock_metadata_ledger(metadata)
    ledger = record_metadata_access(
        ledger,
        access_kind="cohort_metadata_read",
        source_sha256="sha256:" + "9" * 64,
    )
    lock = lock_preholdout_metadata_audit(
        ledger, authority_metadata=metadata
    )

    assert lock["holdout_reaction_read_count"] == 0
    assert lock["market_reaction_exposure"] == "SEALED_UNREAD"
    assert lock["sports_outcome_exposure"] == "PRIOR_EXPOSED_X11"
    assert lock["stage_a_outcome_validation_eligible"] is False
    assert lock["stage_b_market_reaction_validation_eligible"] is True
    assert lock["sports_outcome_source_evidence_sha256"] == (
        X11_SPORTS_OUTCOME_EVIDENCE_SHA256
    )
    assert (
        lock["sports_outcome_observation_count"]
        == X11_HOLDOUT_DRIVE_OUTCOME_COUNT
    )
    with pytest.raises(KalshiValidationError, match="metadata-only"):
        record_metadata_access(
            ledger,
            access_kind="holdout_reaction_read",
            source_sha256="sha256:" + "8" * 64,
        )
    assert not hasattr(validation, "transport_to_kalshi_holdout")


def test_preholdout_lock_recomputes_mutable_authority_frames() -> None:
    metadata = _bind_authority()
    ledger = begin_prelock_metadata_ledger(metadata)
    metadata.development.loc[
        metadata.development.index[0], "batch_sha256"
    ] = "sha256:" + "f" * 64

    with pytest.raises(
        KalshiValidationError,
        match="current authority metadata",
    ):
        lock_preholdout_metadata_audit(
            ledger,
            authority_metadata=metadata,
        )


def test_genuine_fold_slice_scores_but_cannot_select() -> None:
    base_panel = _diagnostic_panel()
    panel = pd.concat(
        [
            base_panel.assign(endpoint_seconds=endpoint_seconds)
            for endpoint_seconds in range(5, 31, 5)
        ],
        ignore_index=True,
    )
    panel["landmark_seconds"] = 3
    for index in panel.index:
        decision = json.loads(panel.loc[index, "decision_features_json"])
        decision["landmark_seconds"] = 3
        decision_json = json.dumps(
            decision, sort_keys=True, separators=(",", ":")
        )
        panel.loc[index, "decision_features_json"] = decision_json
        panel.loc[index, "decision_feature_sha256"] = _sha256_text(
            decision_json
        )
    run = run_x15_historical_trades_diagnostic_walk_forward(
        panel,
        model_ids=("b0_empirical_v1", "regularized_logistic_v1"),
        feature_block_ids=("D0", "D1"),
        fold_ids=("fold_01",),
        transport_pairs=(("polymarket", "kalshi"),),
        include_magnitude=False,
    )
    authority = str(
        run.oof_predictions["cohort_authority_sha256"].iloc[0]
    )
    metadata = bind_frozen_authority_metadata(
        (
            authority_rows := _authority_rows_for_panel(
                panel, authority=authority
            )
        ),
        cohort_authority_sha256=authority,
        sports_outcome_exposed_game_ids=tuple(
            authority_rows.loc[
                authority_rows["cohort"].eq("holdout"), "game_id"
            ].astype(str)
        ),
        sports_outcome_exposed_game_ids_sha256=hash_game_id_evidence(
            tuple(
                authority_rows.loc[
                    authority_rows["cohort"].eq("holdout"), "game_id"
                ].astype(str)
            )
        ),
        sports_outcome_source_evidence_sha256=(
            X11_SPORTS_OUTCOME_EVIDENCE_SHA256
        ),
        sports_outcome_observation_count=(
            X11_HOLDOUT_DRIVE_OUTCOME_COUNT
        ),
        reaction_blind_metadata_game_ids=tuple(
            authority_rows.loc[
                authority_rows["cohort"].eq("holdout"), "game_id"
            ].astype(str)
        ),
        reaction_blind_metadata_game_ids_sha256=hash_game_id_evidence(
            tuple(
                authority_rows.loc[
                    authority_rows["cohort"].eq("holdout"), "game_id"
                ].astype(str)
            )
        ),
    )
    selection = select_candidate_against_b0(
        run,
        spec=FrozenSelectionSpec(
            candidate_model_id="regularized_logistic_v1",
            candidate_feature_block_id="D1",
        ),
        authority=metadata.selection_authority,
    )
    assert selection.authority_gate_passed is False
    assert selection.selected is False
    assert selection.diagnostic_status == (
        "PARTIAL_DEVELOPMENT_DIAGNOSTIC_ONLY"
    )
    with pytest.raises(
        KalshiValidationError,
        match="Stage-B V3 selection evidence",
    ):
        _validate_transport(
            run,
            authority_metadata=metadata,
            stage_b_selection=_stage_b_selection(
                authority=metadata.selection_authority,
            ),
        )
