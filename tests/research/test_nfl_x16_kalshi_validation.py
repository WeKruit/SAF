from __future__ import annotations

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


AUTHORITY = "sha256:" + "a" * 64
TARGET_CONTRACT = "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
CLAIM_BOUNDARY = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)
X11_SPORTS_OUTCOME_EVIDENCE_SHA256 = (
    "sha256:1d0c033459c69778e265be3fca16ae2c87f650d5003a61ffdea4c020a4fd0b05"
)
X11_HOLDOUT_DRIVE_OUTCOME_COUNT = 1_683


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
    reviewed_game_ids: tuple[str, ...] = (),
    reaction_game_ids: tuple[str, ...] = (),
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
        historical_reviewed_game_ids=reviewed_game_ids,
        historical_reviewed_game_ids_sha256=hash_game_id_evidence(
            reviewed_game_ids
        ),
        historical_reaction_game_ids=reaction_game_ids,
        historical_reaction_game_ids_sha256=hash_game_id_evidence(
            reaction_game_ids
        ),
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


def _source_and_transport(*, catastrophic: bool = False) -> X15ModelRun:
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
                correct_probability=(0.08 if catastrophic else 0.70),
            )
            if direction is not pd.NA
            else (0.25, 0.5, 0.25)
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
        source.append(
            {
                **identity,
                "source_row_id": game_position,
                "actual_home_contract_id": f"poly-{game.game_id}",
                "venue": "polymarket",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "VENUE_SPECIFIC",
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
    unmatched_source = source[-1].copy()
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
    metadata = _bind_authority(
        reviewed_game_ids=("development-000",),
        reaction_game_ids=("development-001",),
    )

    assert len(metadata.development) == 153
    assert len(metadata.holdout) == 81
    assert metadata.overlap_game_ids == ()
    assert metadata.holdout_reviewed_overlap_game_ids == ()
    assert metadata.holdout_reaction_overlap_game_ids == ()
    assert metadata.holdout_reaction_read_count == 0
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


def test_sealed_reaction_declaration_cannot_override_authoritative_overlap() -> None:
    holdout_game_id = str(
        _authority_rows()
        .loc[lambda frame: frame["cohort"].eq("holdout"), "game_id"]
        .iloc[0]
    )
    with pytest.raises(
        KalshiValidationError,
        match="authoritative historical.*holdout.*overlap",
    ):
        _bind_authority(reviewed_game_ids=(holdout_game_id,))

    with pytest.raises(KalshiValidationError, match="evidence hash"):
        bind_frozen_authority_metadata(
            _authority_rows(),
            cohort_authority_sha256=AUTHORITY,
            historical_reviewed_game_ids=(),
            historical_reviewed_game_ids_sha256="sha256:" + "0" * 64,
            historical_reaction_game_ids=(),
            historical_reaction_game_ids_sha256=hash_game_id_evidence(()),
            sports_outcome_exposed_game_ids=tuple(
                _authority_rows()
                .loc[
                    lambda frame: frame["cohort"].eq("holdout"),
                    "game_id",
                ]
                .astype(str)
            ),
            sports_outcome_exposed_game_ids_sha256=hash_game_id_evidence(
                tuple(
                    _authority_rows()
                    .loc[
                        lambda frame: frame["cohort"].eq("holdout"),
                        "game_id",
                    ]
                    .astype(str)
                )
            ),
            sports_outcome_source_evidence_sha256=(
                X11_SPORTS_OUTCOME_EVIDENCE_SHA256
            ),
            sports_outcome_observation_count=(
                X11_HOLDOUT_DRIVE_OUTCOME_COUNT
            ),
            reaction_blind_metadata_game_ids=tuple(
                _authority_rows()
                .loc[
                    lambda frame: frame["cohort"].eq("holdout"),
                    "game_id",
                ]
                .astype(str)
            ),
            reaction_blind_metadata_game_ids_sha256=hash_game_id_evidence(
                tuple(
                    _authority_rows()
                    .loc[
                        lambda frame: frame["cohort"].eq("holdout"),
                        "game_id",
                    ]
                    .astype(str)
                )
            ),
        )


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
    )

    assert result.target_recalibration_applied is False
    assert result.unmatched_source_rows == 1
    assert result.unmatched_target_rows == 0
    assert result.native_comparator_paired_rows == 36
    assert result.paired_game_count == 36
    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is False
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
    )

    assert result.score_gate_passed is True
    assert result.catastrophic_degradation is True
    assert result.transport_gate_passed is False
    assert result.diagnostic_status == "HISTORICAL_SIGNAL_REJECTED"


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
            recalibrated_run, authority_metadata=metadata
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
            drifted_run, authority_metadata=metadata
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
        historical_reviewed_game_ids=(),
        historical_reviewed_game_ids_sha256=hash_game_id_evidence(()),
        historical_reaction_game_ids=(),
        historical_reaction_game_ids_sha256=hash_game_id_evidence(()),
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
        run, authority_metadata=metadata
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
