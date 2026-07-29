from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from prediction_market.research import (
    nfl_x15_development_panel as development_panel_module,
)
from prediction_market.research import nfl_x15_models as models_module
from prediction_market.research.nfl_x15_models import (
    DIAGNOSTIC_ANALYSIS_SCOPE,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS,
    DIAGNOSTIC_FEATURE_BLOCKS,
    DIAGNOSTIC_MODEL_INPUT_COLUMNS,
    DIAGNOSTIC_SCHEMA_VERSION,
    DIAGNOSTIC_TARGET_CONTRACT,
    FEATURE_BLOCKS,
    MODEL_IDS,
    X15ModelInputError,
    X15ModelRun,
    _magnitude_metric_rows,
    build_x15_week_folds,
    hierarchical_sample_weights,
    run_x15_historical_trades_diagnostic_walk_forward,
    run_x15_walk_forward,
)
from prediction_market.research.nfl_x15_distribution import (
    QuantileSupportContract,
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decision_json(
    *,
    game_id: str,
    episode_id: str,
    venue: str,
    game_index: int,
    week: int,
) -> str:
    price = 0.35 + game_index * 0.025 + (0.01 if venue == "kalshi" else 0)
    payload = {
        "schema_version": "VenueReactionPanelV3",
        "game_id": game_id,
        "atomic_information_episode_id": episode_id,
        "venue": venue,
        "actual_home_contract_id": f"{venue}:{game_id}:home",
        "landmark_seconds": 5,
        "endpoint_seconds": 10,
        "tick_rule_id": f"{venue}:tick",
        "tick_size": 0.01,
        "mark_l_trade_id": f"{venue}:{game_id}:l",
        "mark_l_source_time_utc": f"2025-09-{week + 1:02d}T00:00:05+00:00",
        "mark_l_price": price,
        "mark_l_staleness_seconds": 0.1,
        "prior_30s_actual_trade_count": game_index + 1,
        "prior_30s_actual_trade_size": 10.0 + game_index,
        "prior_60s_actual_trade_count": game_index + 2,
        "prior_60s_actual_trade_size": 20.0 + game_index,
        "stage_a_status": "AVAILABLE",
        "p_before_home": price - 0.02,
        "p_after_home": price - 0.01,
        "reference_delta_home": 0.01,
        "reference_gap_at_landmark": 0.01,
        "multi_hot_features": {
            "event_tag__touchdown": game_index % 3 == 0,
            "factor__nfl_information": game_index % 2 == 0,
        },
        "fact_features": {
            "source_resolution": "play",
            "game_seconds_remaining": 3_600 - week * 60,
            "score_margin_home": game_index - 3,
            "possession_is_home": game_index % 2 == 0,
            "down": game_index % 4 + 1,
            "distance": game_index + 1,
            "yardline_100": 80 - game_index,
            "primary_action": "pass" if game_index % 2 else "rush",
            "outcome_tags": ["touchdown"] if game_index % 3 == 0 else [],
            "yards_gained": game_index,
            "return_yards": 0,
            "actor_is_home": game_index % 2 == 0,
            "beneficiary_is_home": game_index % 2 == 0,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Six complete games per week gives every training-only prequential split
    # both binary classes and all three direction classes.
    outcomes = (
        (False, None, None),
        (True, False, None),
        (True, True, "DOWN"),
        (True, True, "NO_MOVE"),
        (True, True, "UP"),
        (True, True, "UP"),
    )
    for week in range(1, 13):
        for game_index, (survived, observed, direction) in enumerate(outcomes):
            game_id = f"2025_{week:02d}_G{game_index:02d}"
            episode_id = f"{game_id}:episode"
            for venue in ("polymarket", "kalshi"):
                decision_json = _decision_json(
                    game_id=game_id,
                    episode_id=episode_id,
                    venue=venue,
                    game_index=game_index,
                    week=week,
                )
                target_eligible = bool(survived and observed)
                magnitude = (
                    0.01 * (game_index + 1)
                    if direction in {"UP", "DOWN"}
                    else np.nan
                )
                rows.append(
                    {
                        "schema_version": "VenueReactionPanelV3",
                        "cohort_authority_sha256": _sha256_text(
                            "frozen-development-authority"
                        ),
                        "game_id": game_id,
                        "atomic_information_episode_id": episode_id,
                        "venue": venue,
                        "actual_home_contract_id": f"{venue}:{game_id}:home",
                        "nfl_week": week,
                        "landmark_seconds": 5,
                        "endpoint_seconds": 10,
                        "decision_eligible": True,
                        "target_eligible": target_eligible,
                        "s_h": survived,
                        "o_h_given_s": observed,
                        "direction": direction or "UNOBSERVED",
                        "conditional_magnitude": magnitude,
                        "delta_l_h": (
                            -magnitude
                            if direction == "DOWN"
                            else magnitude if direction == "UP" else 0.0
                            if direction == "NO_MOVE"
                            else np.nan
                        ),
                        "mark_h_price": 0.5 if target_eligible else np.nan,
                        "mark_h_trade_id": (
                            f"{venue}:{game_id}:h"
                            if target_eligible
                            else None
                        ),
                        "reference_status": "SUPPORTED",
                        "decision_features_json": decision_json,
                        "decision_feature_sha256": _sha256_text(decision_json),
                    }
                )
    return pd.DataFrame(rows)


def _fold_frame() -> pd.DataFrame:
    return _panel().loc[:, ["game_id", "nfl_week", "venue"]]


def _diagnostic_panel() -> pd.DataFrame:
    panel = _panel()
    panel["schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
    panel["target_contract"] = DIAGNOSTIC_TARGET_CONTRACT
    panel["claim_boundary"] = DIAGNOSTIC_CLAIM_BOUNDARY
    panel["sports_clean_l"] = True
    panel["sports_clean_l_reason"] = "SPORTS_WINDOW_CLEAN"
    panel["sports_clean_h"] = panel["s_h"]
    panel["sports_clean_reason"] = np.where(
        panel["s_h"].astype(bool), "CLEAN", "NEXT_SALIENT_EVENT"
    )
    panel["actual_trade_observed_h"] = (
        panel["o_h_given_s"].fillna(False).astype(bool)
    )
    panel["direction_threshold_probability"] = 0.01
    panel["direction_threshold_semantics"] = (
        DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
    )
    panel["venue_tick_support"] = "UNSUPPORTED"
    panel["market_continuity_support"] = "UNKNOWN"
    for index in panel.index:
        decoded = json.loads(panel.loc[index, "decision_features_json"])
        decoded["schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
        decoded["target_contract"] = DIAGNOSTIC_TARGET_CONTRACT
        decoded.pop("endpoint_seconds", None)
        decoded.pop("tick_rule_id", None)
        decoded.pop("tick_size", None)
        decoded["direction_threshold_probability"] = 0.01
        decoded["direction_threshold_semantics"] = (
            DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
        )
        decision_json = json.dumps(
            decoded, sort_keys=True, separators=(",", ":")
        )
        panel.loc[index, "decision_features_json"] = decision_json
        panel.loc[index, "decision_feature_sha256"] = _sha256_text(
            decision_json
        )
    return panel.drop(columns=["s_h", "o_h_given_s"])


def _two_endpoint_diagnostic_panel() -> pd.DataFrame:
    first = _diagnostic_panel()
    second = first.copy(deep=True)
    second["endpoint_seconds"] = 15
    loses_survival = second["game_id"].str.endswith("G02")
    second.loc[loses_survival, "sports_clean_h"] = False
    second.loc[loses_survival, "sports_clean_reason"] = (
        "NEXT_FINALIZED_INFORMATION_EVENT_AT_OR_BEFORE_H"
    )
    second.loc[loses_survival, "target_eligible"] = False
    second.loc[loses_survival, "delta_l_h"] = np.nan
    second.loc[loses_survival, "direction"] = "UNOBSERVED"
    second.loc[loses_survival, "conditional_magnitude"] = np.nan
    return pd.concat([first, second], ignore_index=True)


def _verified_diagnostic_partitions(
    panel: pd.DataFrame,
):
    groups = tuple(panel.groupby("game_id", sort=True, observed=True))
    batch_file_sha = _sha256_text("diagnostic-batch-file")
    batch_sha = _sha256_text("diagnostic-batch-semantic")
    mapping_sha = _sha256_text("diagnostic-cohort-mapping")
    return tuple(
        development_panel_module.VerifiedDiagnosticPanelPartition(
            game_id=str(game_id),
            panel=game.reset_index(drop=True),
            batch_game_count=len(groups),
            batch_manifest_file_sha256=batch_file_sha,
            batch_sha256=batch_sha,
            game_manifest_sha256=_sha256_text(f"{game_id}:manifest"),
            diagnostic_object_sha256=_sha256_text(f"{game_id}:panel"),
            cohort_authority_sha256=str(
                game["cohort_authority_sha256"].iloc[0]
            ),
            cohort_mapping_sha256=mapping_sha,
        )
        for game_id, game in groups
    )


def test_builds_five_complete_game_folds_shared_by_both_venues() -> None:
    folds = build_x15_week_folds(_fold_frame())

    assert [
        (fold.train_weeks, fold.validation_weeks) for fold in folds
    ] == [
        ((1, 2), (3, 4)),
        (tuple(range(1, 5)), (5, 6)),
        (tuple(range(1, 7)), (7, 8)),
        (tuple(range(1, 9)), (9, 10)),
        (tuple(range(1, 11)), (11, 12)),
    ]
    for fold in folds:
        assert set(fold.train_game_ids).isdisjoint(fold.validation_game_ids)
        selected = _fold_frame()[
            _fold_frame()["game_id"].isin(fold.validation_game_ids)
        ]
        assert selected.groupby("game_id")["venue"].nunique().eq(2).all()


def test_rejects_conflicting_game_week_and_non_development_week() -> None:
    conflicting = _fold_frame()
    conflicting.loc[len(conflicting)] = {
        "game_id": conflicting.iloc[0]["game_id"],
        "nfl_week": 2,
        "venue": "polymarket",
    }
    with pytest.raises(X15ModelInputError, match="one NFL week"):
        build_x15_week_folds(conflicting)

    outside = _fold_frame()
    outside.loc[len(outside)] = {
        "game_id": "holdout",
        "nfl_week": 13,
        "venue": "polymarket",
    }
    with pytest.raises(X15ModelInputError, match="weeks 1 through 12"):
        build_x15_week_folds(outside)


def test_feature_blocks_are_fixed_and_strictly_incremental() -> None:
    assert tuple(FEATURE_BLOCKS) == ("B0", "B1", "B2", "B3", "B4")
    for earlier, later in zip(
        FEATURE_BLOCKS.values(),
        tuple(FEATURE_BLOCKS.values())[1:],
        strict=False,
    ):
        assert set(earlier).issubset(later)
        assert len(later) > len(earlier)
    assert "mark_h_price" not in {
        feature for block in FEATURE_BLOCKS.values() for feature in block
    }
    assert "reference_status" not in {
        feature for block in FEATURE_BLOCKS.values() for feature in block
    }


def test_diagnostic_feature_blocks_are_independent_and_never_use_tick_or_continuity() -> None:
    assert tuple(DIAGNOSTIC_FEATURE_BLOCKS) == (
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
    )
    forbidden = {
        "tick_size",
        "tick_rule_id",
        "market_continuity_support",
        "venue_tick_support",
    }
    assert forbidden.isdisjoint(
        {
            feature
            for block in DIAGNOSTIC_FEATURE_BLOCKS.values()
            for feature in block
        }
    )
    for earlier, later in zip(
        DIAGNOSTIC_FEATURE_BLOCKS.values(),
        tuple(DIAGNOSTIC_FEATURE_BLOCKS.values())[1:],
        strict=False,
    ):
        assert set(earlier).issubset(later)


def test_diagnostic_model_input_projection_is_public_minimal_and_immutable() -> None:
    assert isinstance(DIAGNOSTIC_MODEL_INPUT_COLUMNS, tuple)
    assert {
        "schema_version",
        "target_contract",
        "claim_boundary",
        "cohort_authority_sha256",
        "sports_clean_l",
        "sports_clean_l_reason",
        "sports_clean_h",
        "sports_clean_reason",
        "actual_trade_observed_h",
        "target_eligible",
        "direction",
        "conditional_magnitude",
        "decision_features_json",
        "decision_feature_sha256",
        "venue_tick_support",
        "market_continuity_support",
    }.issubset(DIAGNOSTIC_MODEL_INPUT_COLUMNS)
    assert {
        "mark_l_trade_ids_json",
        "mark_l_trade_id_set_sha256",
        "mark_h_trade_ids_json",
        "mark_h_trade_id_set_sha256",
        "mark_h_price",
        "mark_h_semantics",
        "attrition_reason",
    }.isdisjoint(DIAGNOSTIC_MODEL_INPUT_COLUMNS)


def test_explicit_diagnostic_entry_stamps_nonconfirmatory_target_contract_everywhere() -> None:
    result = run_x15_historical_trades_diagnostic_walk_forward(
        _diagnostic_panel(),
        model_ids=("b0_empirical_v1", "regularized_logistic_v1"),
        feature_block_ids=tuple(DIAGNOSTIC_FEATURE_BLOCKS),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )

    assert result.run_config is not None
    assert result.run_config["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert result.run_config["target_contract"] == DIAGNOSTIC_TARGET_CONTRACT
    assert result.run_config["claim_boundary"] == DIAGNOSTIC_CLAIM_BOUNDARY
    assert result.run_config["analysis_scope"] == DIAGNOSTIC_ANALYSIS_SCOPE
    assert result.run_config["survival_probability_contract"] == (
        "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
    )
    assert (
        result.run_config["analysis_scope"]
        == "HISTORICAL_TRADES_ONLY_SOURCE_TIME_DIAGNOSTIC"
    )
    assert set(result.oof_predictions["feature_block_id"]) == {
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
    }
    for output in (
        result.oof_predictions,
        result.conditional_quantiles,
        result.fold_metrics,
        result.support_audit,
        result.weight_audit,
    ):
        assert output["schema_version"].eq(DIAGNOSTIC_SCHEMA_VERSION).all()
        assert output["target_contract"].eq(DIAGNOSTIC_TARGET_CONTRACT).all()
        assert output["claim_boundary"].eq(DIAGNOSTIC_CLAIM_BOUNDARY).all()
        assert output["analysis_scope"].eq(DIAGNOSTIC_ANALYSIS_SCOPE).all()
        assert output["direction_threshold_probability"].eq(0.01).all()
        assert output["direction_threshold_semantics"].eq(
            DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
        ).all()
        assert output["venue_tick_support"].eq("UNSUPPORTED").all()
        assert output["market_continuity_support"].eq("UNKNOWN").all()
        assert output["claim_eligible"].eq(False).all()
    assert set(result.oof_predictions["s_h_truth"].dropna().unique()) == {
        False,
        True,
    }


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "schema_version",
            "VenueReactionPanelV3",
            "HistoricalTradesOnlyProbabilityPanelV2",
        ),
        ("target_contract", "WRONG_TARGET", "target_contract"),
        ("claim_boundary", "WRONG_CLAIM", "claim_boundary"),
        ("direction_threshold_probability", 0.02, "frozen at 0.01"),
        (
            "direction_threshold_semantics",
            "VENUE_TICK",
            "research materiality",
        ),
        ("venue_tick_support", "SUPPORTED", "UNSUPPORTED"),
        ("market_continuity_support", "SUPPORTED", "UNKNOWN"),
    ],
)
def test_diagnostic_entry_rejects_contract_drift(
    column: str, value: object, message: str
) -> None:
    panel = _diagnostic_panel()
    panel[column] = value

    with pytest.raises(X15ModelInputError, match=message):
        run_x15_historical_trades_diagnostic_walk_forward(
            panel,
            model_ids=("b0_empirical_v1",),
            feature_block_ids=("D0",),
            fold_ids=("fold_01",),
            include_magnitude=False,
        )


def test_confirmatory_entry_still_rejects_diagnostic_schema() -> None:
    with pytest.raises(X15ModelInputError, match="VenueReactionPanelV3"):
        run_x15_walk_forward(
            _diagnostic_panel(),
            model_ids=("b0_empirical_v1",),
            feature_block_ids=("B0",),
            fold_ids=("fold_01",),
            include_magnitude=False,
        )


def test_diagnostic_entry_rejects_direction_inconsistent_with_fixed_materiality() -> None:
    panel = _diagnostic_panel()
    row = panel.index[panel["target_eligible"]].tolist()[0]
    panel.loc[row, "delta_l_h"] = 0.009
    panel.loc[row, "direction"] = "UP"
    panel.loc[row, "conditional_magnitude"] = 0.009

    with pytest.raises(X15ModelInputError, match="fixed 0.01 materiality"):
        run_x15_historical_trades_diagnostic_walk_forward(
            panel,
            model_ids=("b0_empirical_v1",),
            feature_block_ids=("D0",),
            fold_ids=("fold_01",),
            include_magnitude=False,
        )


def test_diagnostic_entry_rejects_target_eligibility_not_derived_from_sources() -> None:
    panel = _diagnostic_panel()
    row = panel.index[panel["target_eligible"]].tolist()[0]
    panel.loc[row, "target_eligible"] = False

    with pytest.raises(X15ModelInputError, match="target_eligible"):
        run_x15_historical_trades_diagnostic_walk_forward(
            panel,
            model_ids=("b0_empirical_v1",),
            feature_block_ids=("D0",),
            fold_ids=("fold_01",),
            include_magnitude=False,
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("target_contract", "WRONG_TARGET", "decision target_contract"),
        ("tick_size", 0.01, "cannot contain tick_size"),
    ],
)
def test_diagnostic_entry_rejects_decision_json_contract_drift(
    key: str, value: object, message: str
) -> None:
    panel = _diagnostic_panel()
    decoded = json.loads(panel.loc[0, "decision_features_json"])
    decoded[key] = value
    decision_json = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    panel.loc[0, "decision_features_json"] = decision_json
    panel.loc[0, "decision_feature_sha256"] = _sha256_text(decision_json)

    with pytest.raises(X15ModelInputError, match=message):
        run_x15_historical_trades_diagnostic_walk_forward(
            panel,
            model_ids=("b0_empirical_v1",),
            feature_block_ids=("D0",),
            fold_ids=("fold_01",),
            include_magnitude=False,
        )


def test_diagnostic_adapter_projects_to_core_columns_and_preserves_categories() -> None:
    panel = _diagnostic_panel()
    panel["publication_only_blob"] = "x" * 10_000

    adapted, _ = models_module._diagnostic_panel_frame(panel)

    assert "publication_only_blob" not in adapted
    assert "sports_clean_reason" not in adapted
    assert "direction_threshold_semantics" not in adapted
    assert set(adapted).issubset(
        models_module._PANEL_REQUIRED
        | {
            "_direction_truth",
            "_conditional_magnitude",
            "_s_interval_start_seconds",
            "_s_interval_at_risk",
            "_s_interval_truth",
            "_source_row_id",
        }
    )
    for column in (
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "decision_features_json",
        "decision_feature_sha256",
    ):
        assert isinstance(adapted[column].dtype, pd.CategoricalDtype)


def test_discrete_survival_intervals_use_frozen_grid_and_only_at_risk_rows() -> None:
    adapted, _ = models_module._diagnostic_panel_frame(
        _two_endpoint_diagnostic_panel()
    )
    selected = adapted[
        adapted["venue"].astype(str).eq("kalshi")
        & adapted["nfl_week"].eq(1)
    ]

    never_survives = selected[
        selected["game_id"].astype(str).eq("2025_01_G00")
    ].sort_values("endpoint_seconds")
    assert never_survives["_s_interval_start_seconds"].tolist() == [5, 10]
    assert never_survives["_s_interval_at_risk"].tolist() == [True, False]
    assert never_survives["_s_interval_truth"].tolist()[0] == False
    assert pd.isna(never_survives["_s_interval_truth"].tolist()[1])

    fails_second_interval = selected[
        selected["game_id"].astype(str).eq("2025_01_G02")
    ].sort_values("endpoint_seconds")
    assert fails_second_interval["_s_interval_start_seconds"].tolist() == [
        5,
        10,
    ]
    assert fails_second_interval["_s_interval_at_risk"].tolist() == [
        True,
        True,
    ]
    assert fails_second_interval["_s_interval_truth"].tolist() == [True, False]


def test_model_outputs_product_of_interval_survival_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_predict = models_module._predict_head

    def fixed_interval_predictions(
        fitted: object,
        frame: pd.DataFrame,
        matrix: np.ndarray,
    ) -> np.ndarray:
        if fitted.head == "S_H":
            return np.where(frame["endpoint_seconds"].eq(10), 0.8, 0.5)
        return original_predict(fitted, frame, matrix)

    monkeypatch.setattr(
        models_module,
        "_predict_head",
        fixed_interval_predictions,
    )
    monkeypatch.setattr(
        models_module,
        "_calibrate_binary",
        lambda _calibrator, raw: np.asarray(raw, dtype=float),
    )

    result = run_x15_historical_trades_diagnostic_walk_forward(
        _two_endpoint_diagnostic_panel(),
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("D0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )
    predictions = result.oof_predictions.sort_values(
        [
            "game_id",
            "venue",
            "atomic_information_episode_id",
            "landmark_seconds",
            "endpoint_seconds",
        ],
        kind="mergesort",
    )
    for probability_column in (
        "s_h_raw_probability",
        "s_h_calibrated_probability",
    ):
        paths = predictions.groupby(
            [
                "game_id",
                "venue",
                "atomic_information_episode_id",
                "landmark_seconds",
            ],
            observed=True,
            sort=True,
        )
        for _, path in paths:
            assert path[probability_column].tolist() == pytest.approx([0.8, 0.4])
            assert path[probability_column].is_monotonic_decreasing
    assert set(predictions["s_h_truth"].dropna().unique()) == {False, True}


def test_diagnostic_d0_expands_only_requested_features() -> None:
    adapted, parsed = models_module._diagnostic_panel_frame(
        _diagnostic_panel()
    )

    features = models_module._decision_feature_frame(
        adapted.loc[adapted["decision_eligible"]],
        feature_blocks=DIAGNOSTIC_FEATURE_BLOCKS,
        feature_block_ids=("D0",),
        parsed_by_sha256=parsed,
    )

    assert tuple(features.columns) == (
        "landmark_seconds",
        "endpoint_seconds",
    )


def test_diagnostic_multi_hot_expansion_uses_compact_nullable_booleans() -> None:
    adapted, parsed = models_module._diagnostic_panel_frame(
        _diagnostic_panel()
    )

    features = models_module._decision_feature_frame(
        adapted.loc[adapted["decision_eligible"]],
        feature_blocks=DIAGNOSTIC_FEATURE_BLOCKS,
        feature_block_ids=("D3",),
        parsed_by_sha256=parsed,
    )

    multi_hot = [
        column
        for column in features
        if column.startswith("multi_hot__")
    ]
    assert multi_hot
    assert all(str(features[column].dtype) == "boolean" for column in multi_hot)


def test_diagnostic_parses_each_unique_decision_payload_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _diagnostic_panel()
    second_endpoint = panel.copy(deep=True)
    second_endpoint["endpoint_seconds"] = 15
    panel = pd.concat([panel, second_endpoint], ignore_index=True)
    unique_payloads = panel["decision_feature_sha256"].nunique()
    original_loads = json.loads
    parse_calls = 0

    def counting_loads(value: str) -> object:
        nonlocal parse_calls
        parse_calls += 1
        return original_loads(value)

    monkeypatch.setattr(models_module.json, "loads", counting_loads)

    run_x15_historical_trades_diagnostic_walk_forward(
        panel,
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("D0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )

    assert parse_calls == unique_payloads


def test_prepared_diagnostic_partitions_filter_before_global_compaction() -> None:
    panel = _diagnostic_panel()
    ineligible = panel.index[::7]
    panel.loc[ineligible, "decision_eligible"] = False
    panel.loc[ineligible, "target_eligible"] = False
    panel.loc[ineligible, "delta_l_h"] = np.nan
    panel.loc[ineligible, "direction"] = "UNOBSERVED"
    panel.loc[ineligible, "conditional_magnitude"] = np.nan
    partitions = _verified_diagnostic_partitions(panel)

    prepared = (
        models_module.prepare_x15_historical_trades_diagnostic_partitions(
            iter(partitions)
        )
    )
    reversed_prepared = (
        models_module.prepare_x15_historical_trades_diagnostic_partitions(
            reversed(partitions)
        )
    )

    assert isinstance(prepared, models_module.X15PreparedDiagnosticPanel)
    assert len(prepared.frame) == int(panel["decision_eligible"].sum())
    assert prepared.frame["decision_eligible"].all()
    assert prepared.partition_count == panel["game_id"].nunique()
    assert prepared.game_ids == tuple(sorted(panel["game_id"].unique()))
    assert prepared.frame["_source_row_id"].tolist() == list(
        range(len(prepared.frame))
    )
    assert not prepared.frame.duplicated(
        list(models_module._PANEL_GRAIN)
    ).any()
    for column in (
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
        "decision_feature_sha256",
    ):
        assert isinstance(
            prepared.frame[column].dtype,
            pd.CategoricalDtype,
        )
    assert "decision_features_json" not in prepared.frame
    pd.testing.assert_frame_equal(
        prepared.frame,
        reversed_prepared.frame,
    )
    assert dict(prepared.parsed_by_sha256) == dict(
        reversed_prepared.parsed_by_sha256
    )


def test_prepared_diagnostic_adapter_never_receives_raw_pooled_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _diagnostic_panel()
    partitions = _verified_diagnostic_partitions(panel)
    observed_partition_sizes: list[int] = []
    adapter = models_module._diagnostic_panel_frame

    def recording_adapter(
        partition: object,
        *,
        decision_only: bool = False,
    ):
        observed_partition_sizes.append(len(partition))
        return adapter(partition, decision_only=decision_only)

    monkeypatch.setattr(
        models_module,
        "_diagnostic_panel_frame",
        recording_adapter,
    )

    models_module.prepare_x15_historical_trades_diagnostic_partitions(
        iter(partitions)
    )

    assert len(observed_partition_sizes) == len(partitions)
    assert max(observed_partition_sizes) < len(panel)
    assert observed_partition_sizes == [
        len(partition.panel) for partition in partitions
    ]


def test_prepared_diagnostic_cache_is_reused_across_execution_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _diagnostic_panel()
    partitions = _verified_diagnostic_partitions(panel)
    original_loads = json.loads
    parse_calls = 0

    def counting_loads(value: str) -> object:
        nonlocal parse_calls
        parse_calls += 1
        return original_loads(value)

    monkeypatch.setattr(models_module.json, "loads", counting_loads)
    prepared = (
        models_module.prepare_x15_historical_trades_diagnostic_partitions(
            iter(partitions)
        )
    )
    parses_after_preparation = parse_calls
    cache_before = dict(prepared.parsed_by_sha256)

    baseline = run_x15_historical_trades_diagnostic_walk_forward(
        prepared,
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("D0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )
    candidate = run_x15_historical_trades_diagnostic_walk_forward(
        prepared,
        model_ids=("regularized_logistic_v1",),
        feature_block_ids=("D1",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )

    assert parses_after_preparation == panel[
        "decision_feature_sha256"
    ].nunique()
    assert parse_calls == parses_after_preparation
    assert dict(prepared.parsed_by_sha256) == cache_before
    assert not baseline.oof_predictions.empty
    assert not candidate.oof_predictions.empty


def test_probability_outputs_are_identical_for_full_grid_and_one_cell_slice() -> None:
    panel = _diagnostic_panel()
    full = run_x15_historical_trades_diagnostic_walk_forward(
        panel,
        model_ids=("b0_empirical_v1", "shallow_xgboost_v1"),
        feature_block_ids=("D0", "D4"),
        fold_ids=("fold_01", "fold_02"),
        include_magnitude=False,
    )
    sliced = run_x15_historical_trades_diagnostic_walk_forward(
        panel,
        model_ids=("shallow_xgboost_v1",),
        feature_block_ids=("D4",),
        fold_ids=("fold_02",),
        include_magnitude=False,
    )

    identity = [
        "source_row_id",
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "training_venue",
        "transport_mode",
        "fold_id",
        "feature_block_id",
        "model_id",
    ]
    probabilities = [
        "s_h_raw_probability",
        "s_h_calibrated_probability",
        "o_h_given_s_raw_probability",
        "o_h_given_s_calibrated_probability",
        "direction_raw_prob_down",
        "direction_raw_prob_no_move",
        "direction_raw_prob_up",
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    ]
    full_cell = full.oof_predictions.loc[
        full.oof_predictions["fold_id"].eq("fold_02")
        & full.oof_predictions["feature_block_id"].eq("D4")
        & full.oof_predictions["model_id"].eq("shallow_xgboost_v1"),
        [*identity, *probabilities],
    ].sort_values(identity, kind="mergesort").reset_index(drop=True)
    sliced_cell = sliced.oof_predictions.loc[
        :,
        [*identity, *probabilities],
    ].sort_values(identity, kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(full_cell, sliced_cell)


def test_prepared_diagnostic_cache_keeps_only_governed_features() -> None:
    panel = _diagnostic_panel()
    for index in panel.index:
        decoded = json.loads(panel.loc[index, "decision_features_json"])
        decoded["unused_publication_metadata"] = {
            "blob": "not-a-model-feature" * 100
        }
        decision_json = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
        )
        panel.loc[index, "decision_features_json"] = decision_json
        panel.loc[index, "decision_feature_sha256"] = _sha256_text(
            decision_json
        )

    prepared = (
        models_module.prepare_x15_historical_trades_diagnostic_partitions(
            iter(_verified_diagnostic_partitions(panel))
        )
    )

    assert all(
        "unused_publication_metadata" not in payload
        for payload in prepared.parsed_by_sha256.values()
    )
    assert all(
        "mark_l_price" in payload
        for payload in prepared.parsed_by_sha256.values()
    )


def test_prepared_diagnostic_merge_fails_closed_on_digest_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = (
        _diagnostic_panel()
        .groupby("game_id", sort=True, observed=True)
        .head(1)
        .head(2)
        .copy()
    )
    collision_sha = "sha256:" + "f" * 64
    panel["decision_feature_sha256"] = collision_sha
    monkeypatch.setattr(
        models_module,
        "_sha256_text_exact",
        lambda _: collision_sha,
    )

    with pytest.raises(
        X15ModelInputError,
        match="digest maps to conflicting decision payloads",
    ):
        models_module.prepare_x15_historical_trades_diagnostic_partitions(
            iter(_verified_diagnostic_partitions(panel))
        )


def test_prepared_diagnostic_rejects_unverified_dataframe_partition() -> None:
    with pytest.raises(
        X15ModelInputError,
        match="VerifiedDiagnosticPanelPartition",
    ):
        models_module.prepare_x15_historical_trades_diagnostic_partitions(
            [_diagnostic_panel()]
        )


def test_hierarchical_weights_equalize_game_then_episode_then_rows() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["a", "a", "a", "b", "b"],
            "atomic_information_episode_id": ["a1", "a1", "a2", "b1", "b1"],
            "eligible": [True, True, True, True, False],
            # Multiple tags remain attributes on one row and do not multiply it.
            "event_tags": [["x", "y"], ["x"], ["z"], ["x", "z"], ["z"]],
        }
    )

    weights = hierarchical_sample_weights(frame, frame["eligible"])

    assert weights.groupby(frame["game_id"]).sum().to_dict() == pytest.approx(
        {"a": 1.0, "b": 1.0}
    )
    a_weights = weights[frame["game_id"] == "a"]
    assert a_weights.iloc[:2].sum() == pytest.approx(0.5)
    assert a_weights.iloc[2] == pytest.approx(0.5)
    assert weights.iloc[4] == 0


def _small_run(frame: pd.DataFrame, **kwargs: object) -> X15ModelRun:
    return run_x15_walk_forward(
        frame,
        model_ids=MODEL_IDS,
        feature_block_ids=("B0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
        random_state=20260728,
        **kwargs,
    )


def test_runner_emits_all_heads_separately_by_venue_with_train_only_calibration() -> None:
    frame = _panel()
    original = frame.copy(deep=True)

    result = _small_run(frame)

    assert isinstance(result, X15ModelRun)
    assert set(result.oof_predictions["model_id"]) == set(MODEL_IDS)
    assert set(result.oof_predictions["training_venue"]) == {
        "polymarket",
        "kalshi",
    }
    assert result.oof_predictions.groupby(
        ["source_row_id", "feature_block_id"]
    )["model_id"].nunique().eq(len(MODEL_IDS)).all()
    direction_probability = result.oof_predictions[
        [
            "direction_calibrated_prob_down",
            "direction_calibrated_prob_no_move",
            "direction_calibrated_prob_up",
        ]
    ].dropna()
    assert direction_probability.sum(axis=1).to_numpy() == pytest.approx(
        np.ones(len(direction_probability))
    )
    assert set(result.oof_predictions["s_h_calibration_status"]).issubset(
        {"CALIBRATED", "RAW_UNCALIBRATED"}
    )
    for row in result.oof_predictions.itertuples(index=False):
        assert set(row.preprocessor_fit_game_ids).isdisjoint(
            row.validation_game_ids
        )
        assert set(row.calibrator_fit_game_ids_s_h).isdisjoint(
            row.validation_game_ids
        )
        assert row.training_venue == row.venue
    pd.testing.assert_frame_equal(frame, original)


def test_validation_target_tamper_does_not_change_fold_predictions() -> None:
    frame = _panel()
    tampered = frame.copy(deep=True)
    mask = tampered["nfl_week"].eq(3)
    tampered.loc[mask, "s_h"] = ~tampered.loc[mask, "s_h"].astype(bool)
    tampered.loc[mask, "o_h_given_s"] = None
    tampered.loc[mask, "target_eligible"] = False
    tampered.loc[mask, "direction"] = "UNOBSERVED"
    tampered.loc[mask, ["delta_l_h", "conditional_magnitude"]] = np.nan

    first = _small_run(frame).oof_predictions
    second = _small_run(tampered).oof_predictions
    probability_columns = [
        column
        for column in first.columns
        if "prob" in column or column.endswith("_sha256")
    ]

    pd.testing.assert_frame_equal(
        first.loc[:, ["source_row_id", "model_id", *probability_columns]],
        second.loc[:, ["source_row_id", "model_id", *probability_columns]],
    )


def test_heldout_features_never_fit_outer_transformer_or_calibrator() -> None:
    frame = _panel()
    tampered = frame.copy(deep=True)
    mask = (
        tampered["nfl_week"].eq(3)
        & tampered["game_id"].eq("2025_03_G00")
        & tampered["venue"].eq("kalshi")
    )
    decoded = json.loads(tampered.loc[mask, "decision_features_json"].iloc[0])
    decoded["mark_l_price"] = 0.99
    decision_json = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    tampered.loc[mask, "decision_features_json"] = decision_json
    tampered.loc[mask, "decision_feature_sha256"] = _sha256_text(decision_json)

    first = _small_run(frame).oof_predictions
    second = _small_run(tampered).oof_predictions
    audit_columns = [
        "model_id",
        "venue",
        "training_data_sha256",
        "preprocessor_training_sha256",
        "s_h_calibration_training_sha256",
        "direction_calibration_training_sha256",
    ]

    pd.testing.assert_frame_equal(
        first.loc[:, audit_columns].drop_duplicates().reset_index(drop=True),
        second.loc[:, audit_columns].drop_duplicates().reset_index(drop=True),
    )


def test_future_only_multi_hot_tag_cannot_change_fold_schema_or_predictions() -> None:
    frame = _panel()
    tampered = frame.copy(deep=True)
    future = (
        tampered["nfl_week"].eq(3)
        & tampered["venue"].eq("kalshi")
        & tampered["game_id"].str.endswith("G00")
    )
    decoded = json.loads(tampered.loc[future, "decision_features_json"].iloc[0])
    decoded["multi_hot_features"]["event_tag__future_only"] = True
    decision_json = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    tampered.loc[future, "decision_features_json"] = decision_json
    tampered.loc[future, "decision_feature_sha256"] = _sha256_text(decision_json)

    kwargs = {
        "model_ids": ("regularized_logistic_v1",),
        "feature_block_ids": ("B3",),
        "fold_ids": ("fold_01",),
        "include_magnitude": False,
    }
    first = run_x15_walk_forward(frame, **kwargs).oof_predictions
    second = run_x15_walk_forward(tampered, **kwargs).oof_predictions
    compared = [
        "source_row_id",
        "model_id",
        "venue",
        "preprocessor_training_sha256",
        "s_h_raw_probability",
        "o_h_given_s_raw_probability",
        "direction_raw_prob_down",
        "direction_raw_prob_no_move",
        "direction_raw_prob_up",
    ]

    pd.testing.assert_frame_equal(first.loc[:, compared], second.loc[:, compared])


def test_missing_endpoint_is_retained_as_unavailable_and_never_no_move() -> None:
    panel = _panel()
    missing_mask = (
        panel["nfl_week"].isin([3, 4])
        & panel["game_id"].str.endswith("G00")
    )
    panel["s_h"] = panel["s_h"].astype("boolean")
    panel.loc[missing_mask, "s_h"] = pd.NA
    result = _small_run(panel)
    missing = result.oof_predictions[
        result.oof_predictions["game_id"].str.endswith("G00")
    ]

    assert not missing.empty
    assert missing["s_h_truth"].isna().all()
    assert missing["o_h_given_s_truth"].isna().all()
    assert missing["direction_truth"].isna().all()
    assert missing["direction_truth_status"].eq("UNAVAILABLE").all()
    assert not missing["direction_truth"].eq("NO_MOVE").any()


def test_learned_heads_execute_every_frozen_feature_block() -> None:
    result = run_x15_walk_forward(
        _panel(),
        model_ids=("regularized_logistic_v1", "shallow_xgboost_v1"),
        feature_block_ids=tuple(FEATURE_BLOCKS),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )

    assert set(result.oof_predictions["feature_block_id"]) == set(
        FEATURE_BLOCKS
    )
    assert set(result.oof_predictions["model_id"]) == {
        "regularized_logistic_v1",
        "shallow_xgboost_v1",
    }


def test_single_class_head_publishes_support_failure_without_fake_model() -> None:
    frame = _panel()
    mask = frame["nfl_week"].isin([1, 2]) & frame["venue"].eq("kalshi")
    frame.loc[mask, "s_h"] = True

    result = run_x15_walk_forward(
        frame,
        model_ids=("regularized_logistic_v1",),
        feature_block_ids=("B0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )
    selected = result.oof_predictions[
        result.oof_predictions["venue"].eq("kalshi")
    ]

    assert selected["s_h_support_status"].eq("INSUFFICIENT_SUPPORT").all()
    assert selected["s_h_raw_probability"].isna().all()
    support = result.support_audit[
        (result.support_audit["venue"] == "kalshi")
        & (result.support_audit["head"] == "S_H")
    ]
    assert support["support_status"].eq("INSUFFICIENT_SUPPORT").all()
    assert support["support_reason"].str.contains("single class").all()


def test_support_audit_training_classes_use_one_text_schema() -> None:
    result = run_x15_historical_trades_diagnostic_walk_forward(
        _diagnostic_panel(),
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("D0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )

    assert result.support_audit["training_classes"].map(
        lambda values: all(isinstance(value, str) for value in values)
    ).all()


def test_rejects_endpoint_leakage_inside_decision_features() -> None:
    frame = _panel()
    decoded = json.loads(frame.loc[0, "decision_features_json"])
    decoded["mark_h_price"] = 0.9
    decision_json = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    frame.loc[0, "decision_features_json"] = decision_json
    frame.loc[0, "decision_feature_sha256"] = _sha256_text(decision_json)

    with pytest.raises(X15ModelInputError, match="leakage.*mark_h_price"):
        _small_run(frame)


def test_runner_connects_oof_features_to_conditional_quantile_distribution() -> None:
    result = run_x15_walk_forward(
        _panel(),
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("B0",),
        fold_ids=("fold_01",),
        include_magnitude=True,
        quantile_support_contract=QuantileSupportContract(
            primary_min_rows=2,
            primary_min_games=2,
            extreme_min_rows=100,
            extreme_min_games=50,
        ),
    )

    quantiles = result.conditional_quantiles
    assert set(quantiles["direction_condition"]) == {"DOWN", "UP"}
    assert quantiles["extreme_quantile_status"].eq(
        "INSUFFICIENT_SUPPORT"
    ).all()
    supported = quantiles[
        quantiles["support_status"].eq("SUPPORTED")
    ].loc[:, ["q10", "q25", "q50", "q75", "q90"]]
    assert not supported.empty
    assert np.all(supported.to_numpy() >= 0)
    assert np.all(np.diff(supported.to_numpy(), axis=1) >= 0)


def test_fold_metrics_label_game_effect_quantiles_as_descriptive_not_ci() -> None:
    metrics = _small_run(_panel()).fold_metrics

    assert "cluster_interval_low" not in metrics
    assert "cluster_interval_high" not in metrics
    assert {"game_effect_p025", "game_effect_p975"}.issubset(metrics)


def test_magnitude_metrics_use_validation_hierarchical_weights() -> None:
    quantiles = pd.DataFrame(
        {
            "fold_id": ["fold_01", "fold_01"],
            "venue": ["kalshi", "kalshi"],
            "training_venue": ["kalshi", "kalshi"],
            "calibration_venue": ["kalshi", "kalshi"],
            "transport_mode": ["VENUE_SPECIFIC", "VENUE_SPECIFIC"],
            "model_id": ["directional_quantile_xgboost_v1"] * 2,
            "feature_block_id": ["B0", "B0"],
            "game_id": ["g1", "g1"],
            "conditional_magnitude_truth": [0.0, 1.0],
            "magnitude_weight": [0.9, 0.1],
            "q10": [0.0, 0.0],
            "q25": [0.0, 0.0],
            "q50": [0.0, 0.0],
            "q75": [0.0, 0.0],
            "q90": [0.0, 0.0],
        }
    )

    rows = pd.DataFrame(_magnitude_metric_rows(quantiles))
    q50 = rows[
        rows["metric_scope"].eq("GAME")
        & rows["metric_name"].eq("pinball_q50")
    ]

    assert q50["metric_value"].iloc[0] == pytest.approx(0.05)


def test_zero_moving_magnitude_rows_emit_two_explicit_support_artifacts() -> None:
    panel = _panel()
    training_moving = (
        panel["nfl_week"].isin([1, 2])
        & panel["direction"].isin(["UP", "DOWN"])
    )
    panel.loc[training_moving, "direction"] = "NO_MOVE"
    panel.loc[training_moving, "conditional_magnitude"] = np.nan
    panel.loc[training_moving, "delta_l_h"] = 0.0

    result = run_x15_walk_forward(
        panel,
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("B0",),
        fold_ids=("fold_01",),
        include_magnitude=True,
    )
    support = result.support_audit[
        result.support_audit["head"].isin(["MAGNITUDE_DOWN", "MAGNITUDE_UP"])
    ]

    assert set(support["head"]) == {"MAGNITUDE_DOWN", "MAGNITUDE_UP"}
    assert support["support_status"].eq("INSUFFICIENT_SUPPORT").all()
    assert set(result.conditional_quantiles["direction_condition"]) == {
        "DOWN",
        "UP",
    }
    assert result.conditional_quantiles[
        ["q10", "q25", "q50", "q75", "q90"]
    ].isna().all(axis=None)


def test_published_fit_games_are_actual_venue_specific_training_games() -> None:
    panel = _panel()
    missing_game = "2025_01_G00"
    panel = panel[
        ~(
            panel["venue"].eq("kalshi")
            & panel["game_id"].eq(missing_game)
        )
    ].reset_index(drop=True)

    result = run_x15_walk_forward(
        panel,
        model_ids=("b0_empirical_v1",),
        feature_block_ids=("B0",),
        fold_ids=("fold_01",),
        include_magnitude=False,
    )
    kalshi = result.oof_predictions[result.oof_predictions["venue"].eq("kalshi")]

    assert all(missing_game not in values for values in kalshi["training_game_ids"])
    assert all(
        missing_game not in values
        for values in kalshi["preprocessor_fit_game_ids"]
    )
    kalshi_support = result.support_audit[
        result.support_audit["venue"].eq("kalshi")
    ]
    assert all(
        missing_game not in values
        for values in kalshi_support["training_game_ids"]
    )


def test_requires_one_frozen_cohort_authority_and_carries_it_to_provenance() -> None:
    panel = _panel()
    authority = panel["cohort_authority_sha256"].iloc[0]

    result = _small_run(panel)

    assert result.oof_predictions["cohort_authority_sha256"].eq(authority).all()
    assert result.support_audit["cohort_authority_sha256"].eq(authority).all()

    missing = panel.drop(columns="cohort_authority_sha256")
    with pytest.raises(X15ModelInputError, match="cohort_authority_sha256"):
        _small_run(missing)

    inconsistent = panel.copy()
    inconsistent.loc[
        inconsistent["game_id"].eq("2025_01_G00"),
        "cohort_authority_sha256",
    ] = _sha256_text("different-authority")
    with pytest.raises(X15ModelInputError, match="one cohort authority"):
        _small_run(inconsistent)


def test_transport_fits_and_calibrates_only_source_then_scores_target_venue() -> None:
    panel = _panel()
    target_tampered = panel.copy(deep=True)
    target_training = (
        target_tampered["venue"].eq("kalshi")
        & target_tampered["nfl_week"].isin([1, 2])
    )
    target_tampered.loc[target_training, "s_h"] = True
    for index in target_tampered.index[target_training]:
        decoded = json.loads(
            target_tampered.loc[index, "decision_features_json"]
        )
        decoded["mark_l_price"] = 0.99
        decision_json = json.dumps(
            decoded, sort_keys=True, separators=(",", ":")
        )
        target_tampered.loc[index, "decision_features_json"] = decision_json
        target_tampered.loc[index, "decision_feature_sha256"] = _sha256_text(
            decision_json
        )
    kwargs = {
        "model_ids": ("regularized_logistic_v1",),
        "feature_block_ids": ("B3",),
        "fold_ids": ("fold_01",),
        "include_magnitude": False,
        "transport_pairs": (("polymarket", "kalshi"),),
    }

    first = run_x15_walk_forward(panel, **kwargs).oof_predictions
    second = run_x15_walk_forward(
        target_tampered, **kwargs
    ).oof_predictions
    first = first[first["transport_mode"].eq("NO_TARGET_RECALIBRATION")]
    second = second[second["transport_mode"].eq("NO_TARGET_RECALIBRATION")]
    compared = [
        "source_row_id",
        "training_venue",
        "calibration_venue",
        "venue",
        "transport_mode",
        "preprocessor_training_sha256",
        "s_h_calibration_training_sha256",
        "direction_calibration_training_sha256",
        "s_h_raw_probability",
        "o_h_given_s_raw_probability",
        "direction_raw_prob_down",
        "direction_raw_prob_no_move",
        "direction_raw_prob_up",
    ]

    assert not first.empty
    assert first["training_venue"].eq("polymarket").all()
    assert first["calibration_venue"].eq("polymarket").all()
    assert first["venue"].eq("kalshi").all()
    pd.testing.assert_frame_equal(
        first.loc[:, compared].reset_index(drop=True),
        second.loc[:, compared].reset_index(drop=True),
    )


def test_predictions_and_training_hashes_are_deterministic() -> None:
    first = _small_run(_panel())
    second = _small_run(_panel())

    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)
    pd.testing.assert_frame_equal(first.support_audit, second.support_audit)
    assert first.run_config_sha256 == second.run_config_sha256


def test_model_outputs_and_hashes_are_shuffle_invariant() -> None:
    panel = _panel()
    shuffled = panel.sample(frac=1, random_state=91).reset_index(drop=True)

    first = _small_run(panel)
    second = _small_run(shuffled)

    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)
    pd.testing.assert_frame_equal(first.support_audit, second.support_audit)
    assert first.run_config_sha256 == second.run_config_sha256
