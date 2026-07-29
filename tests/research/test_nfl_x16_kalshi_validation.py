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
                    "prior_exposure_status": (
                        "DEVELOPMENT_USED"
                        if cohort == "development"
                        else "PRISTINE_NO_PRIOR_REACTION_READ"
                    ),
                    "reaction_read_count": 0,
                }
            )
    return pd.DataFrame(rows)


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
        if game_id in used_ids:
            continue
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
                "prior_exposure_status": "DEVELOPMENT_USED",
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
                "prior_exposure_status": (
                    "PRISTINE_NO_PRIOR_REACTION_READ"
                ),
                "reaction_read_count": 0,
            }
        )
    return pd.DataFrame(rows)


def _source_and_transport() -> X15ModelRun:
    source: list[dict[str, object]] = []
    target: list[dict[str, object]] = []
    for game_index in range(4):
        first_fold = game_index < 2
        train_weeks = (1, 2) if first_fold else (1, 2, 3, 4)
        validation_weeks = (3, 4) if first_fold else (5, 6)
        training_ids = (
            ("development-000", "development-001")
            if first_fold
            else (
                "development-000",
                "development-001",
                "development-002",
                "development-003",
            )
        )
        validation_ids = (
            ("development-002", "development-003")
            if first_fold
            else ("development-004", "development-005")
        )
        fold_character = "1" if first_fold else "2"
        identity = {
            "cohort_authority_sha256": AUTHORITY,
            "game_id": f"development-{game_index + 2:03d}",
            "nfl_week": game_index + 3,
            "atomic_information_episode_id": f"episode-{game_index}",
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
            "s_h_calibrated_probability": 0.7,
            "o_h_given_s_calibrated_probability": 0.6,
            "direction_calibrated_prob_down": 0.1,
            "direction_calibrated_prob_no_move": 0.2,
            "direction_calibrated_prob_up": 0.7,
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
            "o_h_given_s_calibration_training_sha256": "sha256:" + "d" * 64,
            "direction_calibration_training_sha256": "sha256:" + "e" * 64,
            "feature_block_sha256": "sha256:" + "7" * 64,
            "model_spec_sha256": "sha256:" + "8" * 64,
            "fold_sha256": "sha256:" + fold_character * 64,
        }
        source.append(
            {
                **identity,
                "source_row_id": game_index,
                "actual_home_contract_id": f"poly-{game_index}",
                "venue": "polymarket",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "VENUE_SPECIFIC",
            }
        )
        target.append(
            {
                **identity,
                "source_row_id": game_index + 100,
                "actual_home_contract_id": f"kalshi-{game_index}",
                "venue": "kalshi",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "NO_TARGET_RECALIBRATION",
            }
        )
    unmatched_source = source[-1].copy()
    unmatched_source["source_row_id"] = 999
    unmatched_source["atomic_information_episode_id"] = (
        "source-only-episode"
    )
    unmatched_source["landmark_seconds"] = 5
    source.append(unmatched_source)
    return X15ModelRun(
        oof_predictions=pd.DataFrame([*source, *target]),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256="sha256:" + "f" * 64,
        run_config=_run_config(),
    )


def test_binds_153_81_authority_without_reaction_access() -> None:
    metadata = bind_frozen_authority_metadata(
        _authority_rows(), cohort_authority_sha256=AUTHORITY
    )

    assert len(metadata.development) == 153
    assert len(metadata.holdout) == 81
    assert set(metadata.development["nfl_week"]).issubset(range(1, 13))
    assert set(metadata.holdout["nfl_week"]).issubset(range(13, 19))
    assert metadata.overlap_game_ids == ()
    assert metadata.holdout_reaction_read_count == 0
    assert set(metadata.prior_exposure_audit["cohort"]) == {
        "development",
        "holdout",
    }


def test_transport_is_polymarket_source_only_to_kalshi_development_pairs() -> None:
    run = _source_and_transport()
    result = validate_development_venue_transport(
        run,
        authority_metadata=bind_frozen_authority_metadata(
            _authority_rows(), cohort_authority_sha256=AUTHORITY
        ),
    )

    assert result.source_venue == "polymarket"
    assert result.target_venue == "kalshi"
    assert result.cohort == "development"
    assert result.target_recalibration_applied is False
    assert len(result.exact_pair_diagnostics) == 4
    assert result.unmatched_source_rows == 1
    assert result.unmatched_target_rows == 0
    assert result.coverage_gate_passed is True
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

    invalid_rows = run.oof_predictions.copy()
    target_index = invalid_rows[
        invalid_rows["transport_mode"].eq("NO_TARGET_RECALIBRATION")
    ].index[0]
    invalid_rows.loc[target_index, "nfl_week"] = 13
    invalid_run = X15ModelRun(
        oof_predictions=invalid_rows,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )
    with pytest.raises(KalshiValidationError, match="development.*1..12"):
        validate_development_venue_transport(
            invalid_run,
            authority_metadata=bind_frozen_authority_metadata(
                _authority_rows(),
                cohort_authority_sha256=AUTHORITY,
            ),
        )


def test_prelock_ledger_is_metadata_only_and_cannot_hide_reaction_reads() -> None:
    metadata = bind_frozen_authority_metadata(
        _authority_rows(), cohort_authority_sha256=AUTHORITY
    )
    ledger = begin_prelock_metadata_ledger(metadata)
    ledger = record_metadata_access(
        ledger,
        access_kind="cohort_metadata_read",
        source_sha256="sha256:" + "9" * 64,
    )
    lock = lock_preholdout_metadata_audit(ledger, authority_metadata=metadata)

    assert lock["holdout_reaction_read_count"] == 0
    assert lock["chain_sha256"].startswith("sha256:")
    with pytest.raises(KalshiValidationError, match="metadata-only"):
        record_metadata_access(
            ledger,
            access_kind="holdout_reaction_read",
            source_sha256="sha256:" + "8" * 64,
        )
    assert not hasattr(validation, "transport_to_kalshi_holdout")


def test_transport_rejects_target_recalibration_and_provenance_drift() -> None:
    run = _source_and_transport()
    metadata = bind_frozen_authority_metadata(
        _authority_rows(), cohort_authority_sha256=AUTHORITY
    )
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


def test_authority_binding_rejects_any_holdout_reaction_counter() -> None:
    rows = _authority_rows()
    holdout_index = rows[rows["cohort"].eq("holdout")].index[0]
    rows.loc[holdout_index, "reaction_read_count"] = 1

    with pytest.raises(
        KalshiValidationError,
        match="holdout reaction access counter.*zero",
    ):
        bind_frozen_authority_metadata(
            rows, cohort_authority_sha256=AUTHORITY
        )


def test_genuine_task4_run_integrates_selection_and_transport() -> None:
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
    selection = select_candidate_against_b0(
        run,
        spec=FrozenSelectionSpec(
            candidate_model_id="regularized_logistic_v1",
            candidate_feature_block_id="D4",
        ),
    )
    authority = str(
        run.oof_predictions["cohort_authority_sha256"].iloc[0]
    )
    transport = validate_development_venue_transport(
        run,
        authority_metadata=bind_frozen_authority_metadata(
            _authority_rows_for_panel(panel, authority=authority),
            cohort_authority_sha256=authority,
        ),
    )

    assert set(selection.paired_rows["feature_block_id_b0"]) == {"D0"}
    assert set(
        selection.paired_rows["feature_block_id_candidate"]
    ) == {"D4"}
    assert selection.schema_version == (
        "HistoricalTradesOnlyProbabilityPanelV1"
    )
    assert transport.paired_identity_count > 0
    assert transport.target_recalibration_applied is False
    assert transport.execution_claim_eligible is False
