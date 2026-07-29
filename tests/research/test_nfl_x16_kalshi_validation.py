from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from prediction_market.research import nfl_x16_kalshi_validation as validation
from prediction_market.research.nfl_x15_model_selection import (
    FrozenSelectionSpec,
    select_candidate_against_b0,
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
    validate_development_venue_transport,
)
from test_nfl_x15_models import (
    _diagnostic_panel,
    _sha256_text,
)
from test_nfl_x15_model_selection import (
    _authority as _complete_selection_authority,
    _run as _complete_selection_run,
    _spec as _complete_selection_spec,
)


AUTHORITY = "sha256:" + "a" * 64
TARGET_CONTRACT = "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
CLAIM_BOUNDARY = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)
X11_SPORTS_OUTCOME_EVIDENCE_SHA256 = (
    "sha256:1d0c033459c69778e265be3fca16ae2c87f650d5003a61ffdea4c020a4fd0b05"
)
X11_HOLDOUT_DRIVE_OUTCOME_COUNT = 1_683


def _selection_spec() -> FrozenSelectionSpec:
    return FrozenSelectionSpec(
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D4",
    )


def _run_config(
    *, fold_ids: tuple[str, ...] = ("fold_01", "fold_02")
) -> dict[str, object]:
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
        "transport_pairs": (("polymarket", "kalshi"),),
    }


def _authority_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, count, weeks in (
        ("development", 153, tuple(range(1, 13))),
        ("holdout", 81, tuple(range(13, 19))),
    ):
        for index in range(count):
            rows.append(
                {
                    "game_id": f"{cohort}-{index:03d}",
                    "cohort": cohort,
                    "nfl_week": weeks[index % len(weeks)],
                    "kickoff_utc": (
                        f"2025-09-{index % 28 + 1:02d}T12:00:00Z"
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
    evaluation_games = metadata.loc[
        metadata["nfl_week"].isin((3, 4, 5, 6))
    ].head(36)
    week_by_game = metadata.set_index("game_id")["nfl_week"]
    source: list[dict[str, object]] = []
    transported: list[dict[str, object]] = []
    native: list[dict[str, object]] = []
    last_candidate_source: dict[str, object] | None = None
    for game_position, game in enumerate(
        evaluation_games.itertuples(index=False)
    ):
        first_fold = int(game.nfl_week) in (3, 4)
        train_weeks = (
            (1, 2) if first_fold else (1, 2, 3, 4)
        )
        validation_weeks = (3, 4) if first_fold else (5, 6)
        training_ids = tuple(
            sorted(
                week_by_game[
                    week_by_game.isin(train_weeks)
                ].index.astype(str)
            )
        )
        validation_ids = tuple(
            sorted(
                week_by_game[
                    week_by_game.isin(validation_weeks)
                ].index.astype(str)
            )
        )
        fold_character = "1" if first_fold else "2"
        s_truth = game_position % 6 != 0
        o_truth = (
            game_position % 5 != 0 if s_truth else pd.NA
        )
        direction = (
            ("DOWN", "NO_MOVE", "UP")[game_position % 3]
            if s_truth and bool(o_truth)
            else pd.NA
        )
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
                f"{game.game_id}:episode"
            ),
            "landmark_seconds": 3,
            "endpoint_seconds": 30,
            "fold_id": f"fold_0{fold_character}",
            "train_weeks": train_weeks,
            "validation_weeks": validation_weeks,
            "training_game_ids": training_ids,
            "validation_game_ids": validation_ids,
            "preprocessor_fit_game_ids": training_ids,
            "calibrator_fit_game_ids_s_h": training_ids,
            "calibrator_fit_game_ids_o_h_given_s": training_ids,
            "calibrator_fit_game_ids_direction": training_ids,
            "model_id": "regularized_logistic_v1",
            "feature_block_id": "D4",
            "target_contract": TARGET_CONTRACT,
            "claim_boundary": CLAIM_BOUNDARY,
            "schema_version": "HistoricalTradesOnlyProbabilityPanelV1",
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
            "source_row_id": game_position,
            "actual_home_contract_id": f"poly-{game.game_id}",
            "venue": "polymarket",
            "training_venue": "polymarket",
            "calibration_venue": "polymarket",
            "transport_mode": "VENUE_SPECIFIC",
            "s_h_calibrated_probability": transport_binary,
            "o_h_given_s_calibrated_probability": transport_observation,
            "direction_calibrated_prob_down": transport_direction[0],
            "direction_calibrated_prob_no_move": transport_direction[1],
            "direction_calibrated_prob_up": transport_direction[2],
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
                "source_row_id": game_position + 3_000,
                "actual_home_contract_id": f"poly-{game.game_id}",
                "venue": "polymarket",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "VENUE_SPECIFIC",
                "s_h_calibrated_probability": b0_binary,
                "o_h_given_s_calibrated_probability": b0_observation,
                "direction_calibrated_prob_down": b0_direction[0],
                "direction_calibrated_prob_no_move": b0_direction[1],
                "direction_calibrated_prob_up": b0_direction[2],
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
    unmatched_source = last_candidate_source.copy()
    unmatched_source["source_row_id"] = 9_999
    unmatched_source["atomic_information_episode_id"] = (
        "source-only-episode"
    )
    unmatched_source["landmark_seconds"] = 5
    source.append(unmatched_source)
    return X15ModelRun(
        oof_predictions=pd.DataFrame(
            [*source, *transported, *native]
        ),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256="sha256:" + "f" * 64,
        run_config=_run_config(),
    )


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
    result = validate_development_venue_transport(
        _source_and_transport(),
        authority_metadata=_bind_authority(),
        spec=_selection_spec(),
    )

    assert result.target_recalibration_applied is False
    assert result.unmatched_source_rows == 1
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
    assert result.diagnostic_status == "HISTORICAL_SIGNAL_CANDIDATE"
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


def test_catastrophic_transport_degradation_fails_candidate_gate() -> None:
    result = validate_development_venue_transport(
        _source_and_transport(catastrophic=True),
        authority_metadata=_bind_authority(),
        spec=_selection_spec(),
    )

    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is True
    assert result.transport_gate_passed is False
    assert result.diagnostic_status == "HISTORICAL_SIGNAL_REJECTED"


def test_jointly_bad_transport_and_native_cannot_beat_frozen_b0() -> None:
    result = validate_development_venue_transport(
        _source_and_transport(jointly_bad=True),
        authority_metadata=_bind_authority(),
        spec=_selection_spec(),
    )

    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is False
    assert result.transport_b0_mean_improvement < 0.0
    assert result.transport_b0_improvement_gate_passed is False
    assert result.transport_gate_passed is False
    assert result.diagnostic_status == "HISTORICAL_SIGNAL_REJECTED"


def test_factor_shortlist_requires_factor_specific_dual_venue_gate() -> None:
    selection = select_candidate_against_b0(
        _complete_selection_run(),
        spec=_complete_selection_spec(),
        authority=_complete_selection_authority(),
    )
    transport = validate_development_venue_transport(
        _source_and_transport(),
        authority_metadata=_bind_authority(),
        spec=_selection_spec(),
    )
    polymarket_membership = selection.paired_rows[
        ["game_id", "atomic_information_episode_id"]
    ].drop_duplicates()
    kalshi_membership = transport.transport_b0_pair_diagnostics[
        ["game_id", "atomic_information_episode_id"]
    ].drop_duplicates()
    membership = pd.concat(
        [polymarket_membership, kalshi_membership],
        ignore_index=True,
    ).drop_duplicates()
    membership["factor_id"] = "NFL.PASS.COMPLETE"
    membership["factor_version"] = "v1"

    shortlist = validation.build_cross_venue_factor_shortlist(
        selection,
        transport_validation=transport,
        factor_membership=membership,
        min_support_games=30,
        min_support_episodes=30,
    )

    row = shortlist.iloc[0]
    assert bool(row["polymarket_individual_statistical_gate"]) is True
    assert bool(row["kalshi_individual_statistical_gate"]) is True
    assert bool(row["cross_venue_same_sign"]) is True
    assert bool(row["factor_specific_dual_venue_gate_passed"]) is True
    assert row["diagnostic_status"] == "HISTORICAL_SIGNAL_CANDIDATE"


def test_global_transport_gate_cannot_replace_bad_kalshi_factor_gate() -> None:
    selection = select_candidate_against_b0(
        _complete_selection_run(),
        spec=_complete_selection_spec(),
        authority=_complete_selection_authority(),
    )
    transport = validate_development_venue_transport(
        _source_and_transport(mixed_factor_bad_games=5),
        authority_metadata=_bind_authority(),
        spec=_selection_spec(),
    )
    assert transport.transport_gate_passed is True
    negative_kalshi_games = set(
        transport.transport_b0_pair_diagnostics.groupby(
            "game_id", sort=True
        )["loss_improvement"]
        .mean()
        .loc[lambda values: values.lt(0)]
        .index.astype(str)
    )
    assert len(negative_kalshi_games) == 5
    polymarket_membership = (
        selection.paired_rows[
            ["game_id", "atomic_information_episode_id"]
        ]
        .drop_duplicates()
        .loc[lambda frame: frame["game_id"].isin(
            sorted(frame["game_id"].unique())[:5]
        )]
    )
    kalshi_membership = transport.transport_b0_pair_diagnostics.loc[
        lambda frame: frame["game_id"].isin(negative_kalshi_games),
        ["game_id", "atomic_information_episode_id"],
    ].drop_duplicates()
    membership = pd.concat(
        [polymarket_membership, kalshi_membership],
        ignore_index=True,
    ).drop_duplicates()
    membership["factor_id"] = "NFL.PASS.COMPLETE"
    membership["factor_version"] = "v1"

    shortlist = validation.build_cross_venue_factor_shortlist(
        selection,
        transport_validation=transport,
        factor_membership=membership,
        min_support_games=5,
        min_support_episodes=5,
    )

    row = shortlist.iloc[0]
    assert row["kalshi_mean_effect"] < 0.0
    assert bool(row["global_transport_gate_passed"]) is True
    assert bool(row["factor_specific_dual_venue_gate_passed"]) is False
    assert row["diagnostic_status"] != "HISTORICAL_SIGNAL_CANDIDATE"


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
        validate_development_venue_transport(
            recalibrated_run,
            authority_metadata=metadata,
            spec=_selection_spec(),
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
        validate_development_venue_transport(
            drifted_run,
            authority_metadata=metadata,
            spec=_selection_spec(),
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


def test_genuine_fold_slice_scores_but_cannot_select() -> None:
    panel = _diagnostic_panel()
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
        feature_block_ids=("D0", "D4"),
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
            candidate_feature_block_id="D4",
        ),
        authority=metadata.selection_authority,
    )
    transport = validate_development_venue_transport(
        run,
        authority_metadata=metadata,
        spec=_selection_spec(),
    )

    assert selection.authority_gate_passed is False
    assert selection.selected is False
    assert selection.diagnostic_status == (
        "PARTIAL_DEVELOPMENT_DIAGNOSTIC_ONLY"
    )
    assert not transport.score_summary.empty
    expected_model_blocks = set(
        map(
            tuple,
            transport.native_comparison_diagnostics[
                ["model_id", "feature_block_id"]
            ]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    score_counts = transport.score_summary.groupby(
        ["model_id", "feature_block_id"]
    ).size()
    assert set(score_counts.index) == expected_model_blocks
    assert score_counts.eq(6).all()
    assert transport.score_gate_passed is False
    assert transport.transport_gate_passed is False
