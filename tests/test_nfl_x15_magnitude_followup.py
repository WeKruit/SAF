from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from prediction_market.research.nfl_x15_magnitude_followup import (
    BOOTSTRAP_SAMPLES,
    MagnitudeFollowupError,
    VerifiedMagnitudeInputs,
    build_magnitude_metric_tables,
    compare_winner_to_d0,
    publish_not_applicable,
    validate_magnitude_cell,
)
from prediction_market.research.nfl_x15_model_selection import (
    FrozenDevelopmentAuthority,
)
from prediction_market.research.nfl_x15_models import X15ModelRun


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _authority() -> FrozenDevelopmentAuthority:
    return FrozenDevelopmentAuthority(
        cohort_authority_sha256=SHA_A,
        development_games=tuple(
            (
                f"game_{index:03d}",
                1 + (index % 12),
                f"2025-09-{1 + (index % 28):02d}T00:00:00.000000Z",
                SHA_B,
            )
            for index in range(153)
        ),
        metadata_sha256=SHA_C,
    )


def _verified_inputs(
    *,
    decision_status: str = "NO_MODEL_ADVANCE",
    winner_feature_block_id: str | None = None,
) -> VerifiedMagnitudeInputs:
    return VerifiedMagnitudeInputs(
        exact45_batch_document={
            "batch_sha256": SHA_B,
            "selection_contract_sha256": SHA_D,
            "panel_batch": {
                "manifest_path": "panel.json",
                "file_sha256": SHA_A,
                "batch_sha256": SHA_C,
            },
            "holdout_reaction_accessed": False,
        },
        exact45_batch_path=Path("/tmp/exact45.json"),
        exact45_batch_file_sha256=SHA_A,
        authority=_authority(),
        selection_document={
            "decision_status": decision_status,
            "holdout_reaction_accessed": False,
        },
        selection_path=Path("/tmp/selection.json"),
        selection_file_sha256=SHA_D,
        decision_status=decision_status,
        winner_model_id=(
            "regularized_logistic_v1"
            if winner_feature_block_id is not None
            else None
        ),
        winner_feature_block_id=winner_feature_block_id,
        source_transport_pairs=(("polymarket", "kalshi"),),
    )


def _quantile_rows(block_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_index, game_id in enumerate(("g1", "g2", "g3")):
        for direction in ("DOWN", "UP"):
            truth = 0.02 + 0.002 * game_index
            center = truth + (0.001 if block_id == "D0" else 0.0)
            for landmark, endpoint in ((3, 30), (5, 60)):
                rows.append(
                    {
                        "source_row_id": len(rows),
                        "cohort_authority_sha256": SHA_A,
                        "game_id": game_id,
                        "nfl_week": game_index + 1,
                        "atomic_information_episode_id": f"{game_id}-ep",
                        "venue": "polymarket",
                        "training_venue": "polymarket",
                        "calibration_venue": "polymarket",
                        "transport_mode": "VENUE_SPECIFIC",
                        "fold_id": "fold_01",
                        "feature_block_id": block_id,
                        "model_id": "directional_quantile_xgboost_v1",
                        "direction_condition": direction,
                        "direction_truth": direction,
                        "conditional_magnitude_truth": truth,
                        "magnitude_weight": 1.0,
                        "support_status": "SUPPORTED",
                        "extreme_quantile_status": "INSUFFICIENT_SUPPORT",
                        "landmark_seconds": landmark,
                        "endpoint_seconds": endpoint,
                        "q05": np.nan,
                        "q10": max(0.0, center - 0.004),
                        "q25": max(0.0, center - 0.002),
                        "q50": center,
                        "q75": center + 0.002,
                        "q90": center + 0.004,
                        "q95": np.nan,
                        "preprocessor_training_sha256": SHA_B,
                        "quantile_training_sha256": SHA_C,
                        "model_spec_sha256": SHA_D,
                        "preprocessor_fit_game_ids": ("train_1", "train_2"),
                    }
                )
    return pd.DataFrame(rows)


def _run(block_id: str) -> X15ModelRun:
    return X15ModelRun(
        oof_predictions=pd.DataFrame(),
        conditional_quantiles=_quantile_rows(block_id),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(
            {
                "fold_id": ["fold_01", "fold_01"],
                "venue": ["polymarket", "polymarket"],
                "training_venue": ["polymarket", "polymarket"],
                "calibration_venue": ["polymarket", "polymarket"],
                "transport_mode": ["VENUE_SPECIFIC", "VENUE_SPECIFIC"],
                "feature_block_id": [block_id, block_id],
                "model_id": [
                    "directional_quantile_xgboost_v1",
                    "directional_quantile_xgboost_v1",
                ],
                "head": ["MAGNITUDE_DOWN", "MAGNITUDE_UP"],
                "support_status": ["SUPPORTED", "SUPPORTED"],
            }
        ),
        weight_audit=pd.DataFrame(),
        run_config_sha256=SHA_A,
        run_config={
            "fold_ids": ("fold_01",),
            "feature_block_ids": (block_id,),
            "transport_pairs": (("polymarket", "kalshi"),),
            "include_magnitude": True,
            "cohort_authority_sha256": SHA_A,
            "holdout_reaction_accessed": False,
        },
    )


def test_metric_tables_publish_q10_to_q90_and_anchor_bootstrap() -> None:
    rows, games, summaries = build_magnitude_metric_tables(
        _quantile_rows("D1"),
        bootstrap_samples=200,
        bootstrap_seed=17,
    )

    assert set(("q10", "q25", "q50", "q75", "q90")).issubset(rows.columns)
    assert set(rows["direction_truth"]) == {"DOWN", "UP"}
    assert {
        "ALL_HORIZONS",
        "ANCHOR_L3_H30",
        "PAIR_L5_H60",
    }.issubset(set(games["scope"]))
    assert games["fold_id"].eq("fold_01").all()
    anchor = summaries.loc[
        summaries["scope"].eq("ANCHOR_L3_H30")
        & summaries["metric_name"].eq("approximate_crps")
    ]
    assert len(anchor) == 2
    assert anchor["game_count"].eq(3).all()
    assert anchor["bootstrap_samples"].eq(200).all()
    assert anchor["bootstrap_ci_low"].notna().all()
    assert anchor["bootstrap_ci_high"].notna().all()


def test_candidate_comparison_is_paired_by_game_and_positive_when_better() -> None:
    _, d0_games, _ = build_magnitude_metric_tables(
        _quantile_rows("D0"),
        bootstrap_samples=100,
        bootstrap_seed=11,
    )
    _, d1_games, _ = build_magnitude_metric_tables(
        _quantile_rows("D1"),
        bootstrap_samples=100,
        bootstrap_seed=11,
    )

    comparison = compare_winner_to_d0(
        pd.concat((d0_games, d1_games), ignore_index=True),
        winner_feature_block_id="D1",
        bootstrap_samples=200,
        bootstrap_seed=19,
    )

    losses = comparison[
        comparison["metric_name"].isin(
            ("pinball_q50", "approximate_crps")
        )
    ]
    assert not losses.empty
    assert losses["mean_improvement"].gt(0).all()
    assert losses["paired_game_count"].eq(3).all()
    assert losses["bootstrap_samples"].eq(200).all()


def test_validate_cell_rejects_wrong_fold_or_transport_contract() -> None:
    run = _run("D1")
    bad = X15ModelRun(
        oof_predictions=run.oof_predictions,
        conditional_quantiles=run.conditional_quantiles,
        fold_metrics=run.fold_metrics,
        support_audit=run.support_audit,
        weight_audit=run.weight_audit,
        run_config_sha256=run.run_config_sha256,
        run_config={
            **dict(run.run_config or {}),
            "transport_pairs": (),
        },
    )

    with pytest.raises(
        MagnitudeFollowupError, match="transport pair"
    ):
        validate_magnitude_cell(
            bad,
            fold_id="fold_01",
            feature_block_id="D1",
            cohort_authority_sha256=SHA_A,
            source_transport_pairs=(("polymarket", "kalshi"),),
        )


def test_no_model_advance_publishes_not_applicable_without_training(
    tmp_path: Path,
) -> None:
    publication = publish_not_applicable(
        verified=_verified_inputs(),
        output_root=tmp_path,
    )

    document = json.loads(publication.manifest_path.read_text())
    assert publication.status == "NOT_APPLICABLE"
    assert document["status"] == "NOT_APPLICABLE"
    assert document["reason"] == "NO_MODEL_ADVANCE"
    assert document["holdout_reaction_accessed"] is False
    assert document["input_exact45"]["manifest_file_sha256"] == SHA_A
    assert document["input_selection"]["manifest_file_sha256"] == SHA_D
    assert publication.manifest_sha256.endswith(
        publication.manifest_path.stem.split(".")[0]
    )


def test_publish_not_applicable_rejects_a_winner() -> None:
    with pytest.raises(
        MagnitudeFollowupError, match="NO_MODEL_ADVANCE"
    ):
        publish_not_applicable(
            verified=_verified_inputs(
                decision_status="MODEL_ADVANCE",
                winner_feature_block_id="D1",
            ),
            output_root=Path("/tmp"),
        )


def test_default_bootstrap_contract_is_frozen() -> None:
    assert BOOTSTRAP_SAMPLES == 10_000


def test_cli_exposes_only_governed_inputs() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/research/run_nfl_x15_magnitude_followup.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--exact45-batch-manifest" in completed.stdout
    assert "--exact45-batch-manifest-file-sha256" in completed.stdout
    assert "--selection-manifest" in completed.stdout
    assert "--selection-manifest-file-sha256" in completed.stdout
    assert "--holdout" not in completed.stdout
