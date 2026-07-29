from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from prediction_market.research.nfl_x15_models import (
    FEATURE_BLOCKS,
    MODEL_IDS,
    X15ModelInputError,
    X15ModelRun,
    build_x15_week_folds,
    hierarchical_sample_weights,
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
        "endpoint_seconds": 30,
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
                        "game_id": game_id,
                        "atomic_information_episode_id": episode_id,
                        "venue": venue,
                        "actual_home_contract_id": f"{venue}:{game_id}:home",
                        "nfl_week": week,
                        "landmark_seconds": 5,
                        "endpoint_seconds": 30,
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
