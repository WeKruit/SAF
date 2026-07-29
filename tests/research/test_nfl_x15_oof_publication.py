from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prediction_market.research.nfl_x15_oof_publication as oof_publication
from prediction_market.research.nfl_x15_model_selection import (
    EXPECTED_FOLDS,
    bind_frozen_development_authority,
)
from prediction_market.research.nfl_x15_models import (
    DIAGNOSTIC_ANALYSIS_SCOPE,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS,
    DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT,
    DIAGNOSTIC_SCHEMA_VERSION,
    DIAGNOSTIC_TARGET_CONTRACT,
    DIAGNOSTIC_VENUE_TICK_SUPPORT,
    X15ModelRun,
)
from prediction_market.research.nfl_x15_oof_publication import (
    NEGATIVE_CONTROL_DESIGN_MATRIX,
    OOF_TABLE_NAMES,
    PROBABILITY_DESIGN_MATRIX,
    SELECTION_PROJECTION_COLUMNS,
    SELECTABLE_CANDIDATE_DESIGN_MATRIX,
    X15OOFPublicationError,
    load_x15_oof_selection_projection,
    load_x15_oof_shard,
    publish_x15_oof_batch,
    publish_x15_oof_shard,
)


AUTHORITY_SHA = "sha256:" + "a" * 64
MAPPING_SHA = "sha256:" + "b" * 64


def _sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _authority():
    rows: list[dict[str, object]] = []
    game_index = 0
    week_counts = {
        1: 15,
        2: 15,
        3: 13,
        4: 13,
        5: 13,
        6: 12,
        7: 12,
        8: 12,
        9: 12,
        10: 12,
        11: 12,
        12: 12,
    }
    for week, count in week_counts.items():
        for _ in range(count):
            rows.append(
                {
                    "game_id": f"2025_{week:02d}_G{game_index:03d}",
                    "nfl_week": week,
                    "kickoff_utc": f"2025-09-{week:02d}T00:00:00Z",
                    "batch_sha256": "sha256:" + f"{game_index:064x}",
                    "cohort_authority_sha256": AUTHORITY_SHA,
                }
            )
            game_index += 1
    return bind_frozen_development_authority(
        pd.DataFrame(rows),
        cohort_authority_sha256=AUTHORITY_SHA,
    )


def _run_config(
    fold_id: str,
    *,
    model_id: str,
    feature_block_id: str,
) -> dict[str, object]:
    return {
        "model_ids": (model_id,),
        "feature_block_ids": (feature_block_id,),
        "fold_ids": (fold_id,),
        "transport_pairs": (),
        "include_magnitude": False,
        "random_state": 20260728,
        "feature_blocks": {
            f"D{index}": ("landmark_seconds", "endpoint_seconds")
            for index in range(5)
        },
        "xgb_params": {"max_depth": 2},
        "quantile_support_contract": "QuantileSupportContract()",
        "cohort_authority_sha256": AUTHORITY_SHA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "analysis_scope": DIAGNOSTIC_ANALYSIS_SCOPE,
        "direction_threshold_probability": (
            DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
        ),
        "direction_threshold_semantics": (
            DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
        ),
        "venue_tick_support": DIAGNOSTIC_VENUE_TICK_SUPPORT,
        "market_continuity_support": (
            DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT
        ),
        "claim_eligible": False,
    }


def _fold_run(
    fold_id: str,
    *,
    authority=None,
    model_id: str = "b0_empirical_v1",
    feature_block_id: str = "D0",
    config_override: dict[str, object] | None = None,
    omit_validation_game: bool = False,
) -> X15ModelRun:
    authority = authority or _authority()
    fold = next(value for value in EXPECTED_FOLDS if value[0] == fold_id)
    _, train_weeks, validation_weeks = fold
    week_by_game = authority.game_weeks
    training_game_ids = tuple(
        sorted(
            game_id
            for game_id, week in week_by_game.items()
            if week in train_weeks
        )
    )
    validation_game_ids = tuple(
        sorted(
            game_id
            for game_id, week in week_by_game.items()
            if week in validation_weeks
        )
    )
    observed_validation = validation_game_ids
    if omit_validation_game:
        observed_validation = observed_validation[:-1]
    config = _run_config(
        fold_id,
        model_id=model_id,
        feature_block_id=feature_block_id,
    )
    if config_override:
        config.update(config_override)
    config_sha = _sha(config)
    common = {
        "fold_id": fold_id,
        "cohort_authority_sha256": AUTHORITY_SHA,
        "run_config_sha256": config_sha,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "claim_eligible": False,
    }
    predictions = pd.DataFrame(
        [
            {
                **common,
                "source_row_id": position,
                "game_id": game_id,
                "nfl_week": week_by_game[game_id],
                "atomic_information_episode_id": f"{game_id}:ep",
                "venue": "polymarket",
                "actual_home_contract_id": f"{game_id}:home",
                "training_venue": "polymarket",
                "calibration_venue": "polymarket",
                "transport_mode": "VENUE_SPECIFIC",
                "landmark_seconds": 3,
                "endpoint_seconds": 30,
                "model_id": model_id,
                "feature_block_id": feature_block_id,
                "train_weeks": train_weeks,
                "validation_weeks": validation_weeks,
                "training_game_ids": training_game_ids,
                "validation_game_ids": validation_game_ids,
                "preprocessor_fit_game_ids": training_game_ids,
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "analysis_scope": DIAGNOSTIC_ANALYSIS_SCOPE,
                "direction_threshold_probability": (
                    DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
                ),
                "direction_threshold_semantics": (
                    DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
                ),
                "venue_tick_support": DIAGNOSTIC_VENUE_TICK_SUPPORT,
                "market_continuity_support": (
                    DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT
                ),
                "s_h_truth": True,
                "o_h_given_s_truth": True,
                "direction_truth": "UP",
                "s_h_calibrated_probability": 0.5,
                "o_h_given_s_calibrated_probability": 0.5,
                "direction_calibrated_prob_down": 0.2,
                "direction_calibrated_prob_no_move": 0.3,
                "direction_calibrated_prob_up": 0.5,
            }
            for position, game_id in enumerate(observed_validation)
        ]
    )
    quantiles = pd.DataFrame(
        columns=[
            *common,
            "game_id",
            "direction_condition",
            "q50",
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                **common,
                "venue": "polymarket",
                "model_id": model_id,
                "feature_block_id": feature_block_id,
                "head": "S_H",
                "metric_name": "brier",
                "metric_value": 0.2,
            }
        ]
    )
    support = pd.DataFrame(
        [
            {
                **common,
                "venue": "polymarket",
                "model_id": model_id,
                "feature_block_id": feature_block_id,
                "head": "S_H",
                "support_status": "SUPPORTED",
            }
        ]
    )
    weights = pd.DataFrame(
        [
            {
                **common,
                "venue": "polymarket",
                "partition": "VALIDATION",
                "head": "S_H",
                "game_id": observed_validation[0],
                "total_weight": 1.0,
            }
        ]
    )
    return X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=quantiles,
        fold_metrics=metrics,
        support_audit=support,
        weight_audit=weights,
        run_config_sha256=config_sha,
        run_config=config,
    )


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_predictions(
    run: X15ModelRun,
    predictions: pd.DataFrame,
) -> X15ModelRun:
    return X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )


def _with_nonselection_rows(run: X15ModelRun) -> X15ModelRun:
    native = run.oof_predictions
    kalshi_native = native.copy()
    kalshi_native["venue"] = "kalshi"
    kalshi_native["training_venue"] = "kalshi"
    kalshi_native["calibration_venue"] = "kalshi"
    kalshi_native["actual_home_contract_id"] = (
        kalshi_native["actual_home_contract_id"].astype(str) + ":kalshi"
    )
    transported = kalshi_native.copy()
    transported["training_venue"] = "polymarket"
    transported["calibration_venue"] = "polymarket"
    transported["transport_mode"] = "NO_TARGET_RECALIBRATION"
    return _replace_predictions(
        run,
        pd.concat(
            [native, kalshi_native, transported],
            ignore_index=True,
        ),
    )


def test_fold_publication_is_content_addressed_idempotent_and_recoverable(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)

    first = publish_x15_oof_shard(
        model_run=run,
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )
    second = publish_x15_oof_shard(
        model_run=run,
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )

    assert second == first
    document = _manifest(first.manifest_path)
    assert document["fold_id"] == "fold_01"
    assert document["model_id"] == "b0_empirical_v1"
    assert document["feature_block_id"] == "D0"
    assert document["selection_role"] == "BASELINE"
    assert document["cohort_authority_sha256"] == AUTHORITY_SHA
    assert document["cohort_mapping_sha256"] == MAPPING_SHA
    assert document["run_config_sha256"] == run.run_config_sha256
    assert document["effective_seed_contract"] == {
        "base_random_state": 20260728,
        "execution_unit_seed_stride": 100,
        "formula": (
            "base_random_state + fold_index*1000 + unit_index*100 "
            "+ feature_block_index*10 + model_index"
        ),
        "shard_fold_index": 0,
        "shard_feature_block_index": 0,
        "shard_model_index": 0,
    }
    assert document["effective_seed_contract_sha256"].startswith("sha256:")
    assert document["diagnostic_claim_boundary"] == DIAGNOSTIC_CLAIM_BOUNDARY
    assert document["holdout_reaction_accessed"] is False
    assert document["selection_ready"] is False
    assert {table["name"] for table in document["tables"]} == set(
        OOF_TABLE_NAMES
    )
    for table in document["tables"]:
        assert table["object_sha256"].startswith("sha256:")
        assert table["schema_fingerprint"].startswith("sha256:")
        assert table["semantic_rows_sha256"].startswith("sha256:")
        assert table["row_count"] >= 0

    recovered = load_x15_oof_shard(
        manifest_path=first.manifest_path,
        output_root=tmp_path,
    )
    assert recovered.run_config_sha256 == run.run_config_sha256
    for name in OOF_TABLE_NAMES:
        pd.testing.assert_frame_equal(
            getattr(recovered, name),
            getattr(run, name),
            check_dtype=False,
        )


def test_partial_fold_batch_is_diagnostic_only(tmp_path: Path) -> None:
    authority = _authority()
    fold = publish_x15_oof_shard(
        model_run=_fold_run("fold_01", authority=authority),
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )

    batch = publish_x15_oof_batch(
        shard_manifest_paths=(fold.manifest_path,),
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )

    document = _manifest(batch.manifest_path)
    assert batch.selection_ready is False
    assert document["selection_ready"] is False
    assert document["publication_status"] == "PARTIAL_DIAGNOSTIC_ONLY"
    assert document["expected_execution_cell_count"] == 55
    assert document["observed_execution_cell_count"] == 1
    assert document["missing_execution_cell_count"] == 54
    assert document["holdout_reaction_accessed"] is False


def test_fold_allows_proven_venue_specific_training_subset(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    predictions = run.oof_predictions.copy()
    training_subset = tuple(predictions.iloc[0]["training_game_ids"][:-1])
    predictions["training_game_ids"] = pd.Series(
        [training_subset] * len(predictions), dtype=object
    )
    predictions["preprocessor_fit_game_ids"] = pd.Series(
        [training_subset] * len(predictions), dtype=object
    )
    subset_run = X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config=run.run_config,
    )

    publication = publish_x15_oof_shard(
        model_run=subset_run,
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )

    assert publication.authority_coverage_complete is True
    document = _manifest(publication.manifest_path)
    assert document["selection_fold_arrays"]["training_game_ids"] == list(
        training_subset
    )
    assert document["selection_fold_arrays"][
        "preprocessor_fit_game_ids"
    ] == list(training_subset)


def test_exact_five_fold_batch_verifies_153_authority_before_selection_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    shards = tuple(
        publish_x15_oof_shard(
            model_run=_with_nonselection_rows(
                _fold_run(
                    fold_id,
                    authority=authority,
                    model_id=model_id,
                    feature_block_id=feature_block_id,
                )
            ),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )
        for fold_id, _, _ in EXPECTED_FOLDS
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX
    )

    batch = publish_x15_oof_batch(
        shard_manifest_paths=tuple(
            shard.manifest_path for shard in shards
        ),
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )

    document = _manifest(batch.manifest_path)
    assert batch.selection_ready is True
    assert document["selection_ready"] is True
    assert document["publication_status"] == "SELECTION_READY"
    assert document["authority_game_count"] == 153
    assert document["training_only_game_count"] == 30
    assert document["oof_validation_game_count"] == 123
    assert document["oof_validation_game_ids_sha256"].startswith("sha256:")
    assert document["observed_fold_ids"] == [
        "fold_01",
        "fold_02",
        "fold_03",
        "fold_04",
        "fold_05",
    ]
    assert document["missing_fold_ids"] == []
    assert document["expected_execution_cell_count"] == 55
    assert document["observed_execution_cell_count"] == 55
    assert document["missing_execution_cell_count"] == 0
    assert document["negative_control_design_matrix"] == [
        list(cell) for cell in NEGATIVE_CONTROL_DESIGN_MATRIX
    ]
    assert document["selectable_candidate_design_matrix"] == [
        list(cell) for cell in SELECTABLE_CANDIDATE_DESIGN_MATRIX
    ]
    assert document["holdout_reaction_accessed"] is False
    assert len(document["shards"]) == 55
    assert document["batch_run_config"]["fold_ids"] == [
        fold[0] for fold in EXPECTED_FOLDS
    ]
    assert len(document["batch_run_config"]["source_shards"]) == 55
    role_by_cell = {
        (reference["model_id"], reference["feature_block_id"]): reference[
            "selection_role"
        ]
        for reference in document["shards"]
        if reference["fold_id"] == "fold_01"
    }
    for cell in NEGATIVE_CONTROL_DESIGN_MATRIX:
        assert (
            role_by_cell[cell]
            == "NEGATIVE_CONTROL_TIME_GRID_ONLY_NOT_SELECTABLE"
        )
    for cell in SELECTABLE_CANDIDATE_DESIGN_MATRIX:
        assert role_by_cell[cell] == "SELECTABLE_CANDIDATE"
    for reference in document["shards"]:
        expected = next(
            fold for fold in EXPECTED_FOLDS if fold[0] == reference["fold_id"]
        )
        assert reference["train_weeks"] == list(expected[1])
        assert reference["validation_weeks"] == list(expected[2])
        assert reference["run_config_sha256"].startswith("sha256:")

    observed_reads: list[dict[str, object]] = []
    real_read_table = oof_publication.pq.read_table

    def audited_read_table(*args, **kwargs):
        if kwargs.get("columns") is not None:
            observed_reads.append(dict(kwargs))
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(
        oof_publication.pq,
        "read_table",
        audited_read_table,
    )
    projection = load_x15_oof_selection_projection(
        batch_manifest_path=batch.manifest_path,
        output_root=tmp_path,
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D1",
    )
    assert tuple(projection.oof_predictions.columns) == (
        SELECTION_PROJECTION_COLUMNS
    )
    assert len(projection.oof_predictions) == 246
    assert projection.oof_predictions["venue"].eq("polymarket").all()
    assert projection.oof_predictions["training_venue"].eq(
        "polymarket"
    ).all()
    assert projection.oof_predictions["transport_mode"].eq(
        "VENUE_SPECIFIC"
    ).all()
    reconstructed = projection.oof_predictions.iloc[0]
    expected = next(row for row in EXPECTED_FOLDS if row[0] == "fold_01")
    assert reconstructed["train_weeks"] == expected[1]
    assert reconstructed["validation_weeks"] == expected[2]
    expected_training = tuple(
        sorted(
            game_id
            for game_id, week in authority.game_weeks.items()
            if week in expected[1]
        )
    )
    expected_validation = tuple(
        sorted(
            game_id
            for game_id, week in authority.game_weeks.items()
            if week in expected[2]
        )
    )
    assert reconstructed["training_game_ids"] == expected_training
    assert reconstructed["validation_game_ids"] == expected_validation
    assert reconstructed["preprocessor_fit_game_ids"] == expected_training
    assert len(observed_reads) == 10
    for read in observed_reads:
        assert read["filters"] == [
            ("venue", "=", "polymarket"),
            ("training_venue", "=", "polymarket"),
            ("transport_mode", "=", "VENUE_SPECIFIC"),
        ]
        assert not {
            "train_weeks",
            "validation_weeks",
            "training_game_ids",
            "validation_game_ids",
            "preprocessor_fit_game_ids",
        }.intersection(read["columns"])
    assert projection.conditional_quantiles.empty
    assert projection.fold_metrics.empty
    assert projection.support_audit.empty
    assert projection.weight_audit.empty
    assert projection.run_config["fold_ids"] == tuple(
        fold[0] for fold in EXPECTED_FOLDS
    )
    assert projection.run_config["model_ids"] == (
        "b0_empirical_v1",
        "regularized_logistic_v1",
    )
    assert projection.run_config["feature_block_ids"] == ("D0", "D1")
    assert len(projection.run_config["source_shards"]) == 10
    assert projection.run_config_sha256.startswith("sha256:")
    with pytest.raises(
        X15OOFPublicationError, match="negative controls only"
    ):
        load_x15_oof_selection_projection(
            batch_manifest_path=batch.manifest_path,
            output_root=tmp_path,
            candidate_model_id="regularized_logistic_v1",
            candidate_feature_block_id="D0",
        )


def test_exact_batch_fails_closed_on_incomplete_fold_or_config_drift(
    tmp_path: Path,
) -> None:
    authority = _authority()
    shards = []
    for fold_id, _, _ in EXPECTED_FOLDS:
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX:
            shards.append(
                publish_x15_oof_shard(
                    model_run=_fold_run(
                        fold_id,
                        authority=authority,
                        model_id=model_id,
                        feature_block_id=feature_block_id,
                        omit_validation_game=(
                            fold_id == "fold_05"
                            and model_id == "shallow_xgboost_v1"
                            and feature_block_id == "D4"
                        ),
                    ),
                    authority=authority,
                    cohort_mapping_sha256=MAPPING_SHA,
                    output_root=tmp_path,
                )
            )
    with pytest.raises(
        X15OOFPublicationError,
        match="selection-ready.*authority coverage",
    ):
        publish_x15_oof_batch(
            shard_manifest_paths=tuple(
                shard.manifest_path for shard in shards
            ),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )

    drift_root = tmp_path / "drift"
    drifted = []
    for fold_id, _, _ in EXPECTED_FOLDS:
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX:
            drifted.append(
                publish_x15_oof_shard(
                    model_run=_fold_run(
                        fold_id,
                        authority=authority,
                        model_id=model_id,
                        feature_block_id=feature_block_id,
                        config_override=(
                            {"random_state": 7}
                            if (
                                fold_id == "fold_05"
                                and model_id == "shallow_xgboost_v1"
                                and feature_block_id == "D4"
                            )
                            else None
                        ),
                    ),
                    authority=authority,
                    cohort_mapping_sha256=MAPPING_SHA,
                    output_root=drift_root,
                )
            )
    with pytest.raises(X15OOFPublicationError, match="run config"):
        publish_x15_oof_batch(
            shard_manifest_paths=tuple(
                shard.manifest_path for shard in drifted
            ),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=drift_root,
        )


def test_recovery_rejects_tampered_table_bytes(tmp_path: Path) -> None:
    authority = _authority()
    publication = publish_x15_oof_shard(
        model_run=_fold_run("fold_01", authority=authority),
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )
    document = _manifest(publication.manifest_path)
    object_path = tmp_path / document["tables"][0]["object_path"]
    object_path.write_bytes(object_path.read_bytes() + b"tamper")

    with pytest.raises(X15OOFPublicationError, match="SHA-256|byte length"):
        load_x15_oof_shard(
            manifest_path=publication.manifest_path,
            output_root=tmp_path,
        )


def test_shard_rejects_game_week_different_from_authority(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    predictions = run.oof_predictions.copy()
    predictions.loc[predictions.index[0], "nfl_week"] = 12

    with pytest.raises(
        X15OOFPublicationError,
        match="game.*week.*authority",
    ):
        publish_x15_oof_shard(
            model_run=_replace_predictions(run, predictions),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )


def test_shard_rejects_duplicate_selection_pair_grain(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    predictions = pd.concat(
        [run.oof_predictions, run.oof_predictions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        X15OOFPublicationError,
        match="duplicate.*selection pair grain",
    ):
        publish_x15_oof_shard(
            model_run=_replace_predictions(run, predictions),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("s_h_truth", "yes", "binary truth"),
        ("o_h_given_s_truth", 2, "binary truth"),
        ("direction_truth", "SIDEWAYS", "direction truth"),
        ("s_h_calibrated_probability", -0.01, "probability"),
        ("o_h_given_s_calibrated_probability", 1.01, "probability"),
        ("direction_calibrated_prob_up", np.nan, "probability"),
    ],
)
def test_shard_rejects_invalid_truth_or_probability_semantics(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    predictions = run.oof_predictions.copy()
    predictions[column] = predictions[column].astype(object)
    predictions.loc[predictions.index[0], column] = value

    with pytest.raises(X15OOFPublicationError, match=message):
        publish_x15_oof_shard(
            model_run=_replace_predictions(run, predictions),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )


def test_shard_rejects_direction_probabilities_that_do_not_sum_to_one(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    predictions = run.oof_predictions.copy()
    predictions.loc[
        predictions.index[0],
        "direction_calibrated_prob_up",
    ] = 0.4

    with pytest.raises(
        X15OOFPublicationError,
        match="direction probabilities.*sum to one",
    ):
        publish_x15_oof_shard(
            model_run=_replace_predictions(run, predictions),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"s_h_truth": False, "o_h_given_s_truth": True},
            "O_H truth requires S_H",
        ),
        (
            {
                "s_h_truth": True,
                "o_h_given_s_truth": False,
                "direction_truth": "UP",
            },
            "direction truth requires O_H",
        ),
    ],
)
def test_shard_rejects_impossible_hurdle_truth_dependencies(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    predictions = run.oof_predictions.copy()
    for column, value in updates.items():
        predictions[column] = predictions[column].astype(object)
        predictions.loc[predictions.index[0], column] = value

    with pytest.raises(X15OOFPublicationError, match=message):
        publish_x15_oof_shard(
            model_run=_replace_predictions(run, predictions),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )


def test_shard_rejects_nonintegral_random_state_seed(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run = _fold_run(
        "fold_01",
        authority=authority,
        config_override={"random_state": "20260728"},
    )

    with pytest.raises(
        X15OOFPublicationError,
        match="random_state.*integer",
    ):
        publish_x15_oof_shard(
            model_run=run,
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )
