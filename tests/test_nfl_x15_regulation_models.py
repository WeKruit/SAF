from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import prediction_market.research.nfl_x15_regulation_models as regulation_models
from prediction_market.research.nfl_x15_regulation_panel_v4 import (
    FORMAL_MODEL_FORBIDDEN_COLUMNS,
    RegulationAnchorGamePanel,
    publish_regulation_panel_v4_frames,
)
from prediction_market.research.nfl_x15_native_rule_audit import (
    NativeRuleInput,
    RuleAuditFrames,
    build_game_rule_audit,
    publish_rule_audit_frames,
)
from prediction_market.research.nfl_x15_regulation_models import (
    CORRECTED_MODEL_CELLS,
    DIRECTION_CLASSES,
    FEATURE_BLOCKS,
    FrozenPolymarketWinnerV2,
    MARKET_CONTINUITY_CONTRACT,
    PRESTATE_CONTEXT_CONTRACT,
    PRIMARY_FEATURE_CONTRACT,
    PublishedRegulationSelectionLock,
    RegulationModelInputError,
    REQUIRED_PANEL_COLUMNS,
    TRAINING_PROVENANCE_COLUMNS,
    VerifiedRegulationPanelBatch,
    VerifiedRegulationPanelManifest,
    VerifiedRegulationOOFShard,
    VerifiedRegulationSelectionLock,
    compute_equal_game_metrics,
    discover_verified_regulation_oof_shards,
    load_verified_regulation_panel_batch,
    prepare_regulation_panel,
    publish_regulation_oof_shard,
    publish_regulation_oof_batch_index,
    publish_regulation_selection_lock,
    regulation_oof_batch_status,
    regulation_row_losses,
    run_frozen_kalshi_transport,
    run_regulation_walk_forward,
    select_regulation_model,
    validate_kalshi_transport,
    verify_regulation_panel_batch_manifest,
    verify_regulation_oof_shard,
    verify_regulation_selection_lock,
)
from prediction_market.research.nfl_x15_regulation_models import (
    _aligned_direction_probability,
    _fit_estimator,
    _fit_two_heads,
    _parquet_bytes,
    _predict_two_heads,
)


SHA = "sha256:" + "a" * 64
PANEL_SINGLE_SHA_SET = "sha256:" + hashlib.sha256(
    (
        json.dumps([SHA], separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
).hexdigest()
_PROJECT_ROOT = Path(__file__).parents[1]
_RUNNER_PATH = (
    _PROJECT_ROOT / "scripts/research/run_nfl_x15_regulation_models.py"
)


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_row_id = 0
    for week in range(1, 13):
        for game_number in range(4):
            game_id = f"2025_{week:02d}_A{game_number}_B{game_number}"
            for episode_number in range(6):
                available = episode_number != 0
                direction = (
                    DIRECTION_CLASSES[
                        (game_number + episode_number) % 3
                    ]
                    if available
                    else pd.NA
                )
                source_row_id += 1
                source_start = pd.Timestamp(
                    "2025-09-01T00:00:00Z"
                ) + pd.Timedelta(
                    days=7 * (week - 1) + game_number,
                    seconds=10 * episode_number,
                )
                source_end = source_start + pd.Timedelta(seconds=1)
                rows.append(
                    {
                        "schema_version": "RegulationDecisionPanelV4",
                        "dataset_split": "DEVELOPMENT",
                        "holdout_reaction_accessed": False,
                        "cohort_authority_sha256": SHA,
                        "source_row_id": f"row-{source_row_id}",
                        "game_id": game_id,
                        "nfl_week": week,
                        "information_anchor_id": (
                            f"{game_id}:anchor:{episode_number}"
                        ),
                        "episode_id": (
                            f"{game_id}:episode:{episode_number}"
                        ),
                        "anchor_kind": "PROVISIONAL_FIRST_SEEN",
                        "analysis_role": "PRIMARY_SELECTION",
                        "source_interval_start": source_start,
                        "source_interval_end": source_end,
                        "information_anchor": source_end,
                        "primary_feature_known_at": source_end,
                        "primary_feature_support_status": "SUPPORTED",
                        "primary_feature_support_reason": (
                            "PIA_V2_PROVISIONAL_SNAPSHOT"
                        ),
                        "primary_feature_snapshot_sha256": SHA,
                        "prestate_known_at": source_end,
                        "prestate_context_sha256": SHA,
                        "p_before_support_status": "SUPPORTED",
                        "venue": "POLYMARKET",
                        "actual_home_contract_id": f"{game_id}:home",
                        "logical_market_id": f"{game_id}:winner",
                        "market_family": "moneyline",
                        "market_period": "full_game",
                        "proposition_kind": "primitive",
                        "home_team": f"B{game_number}",
                        "outcome_team": f"B{game_number}",
                        "actual_home_contract_identity_sha256": SHA,
                        "native_raw_manifest_sha256s_json": json.dumps(
                            [SHA], separators=(",", ":")
                        ),
                        "native_raw_object_sha256s_json": json.dumps(
                            [SHA], separators=(",", ":")
                        ),
                        "native_raw_manifest_set_sha256": (
                            PANEL_SINGLE_SHA_SET
                        ),
                        "native_raw_object_set_sha256": (
                            PANEL_SINGLE_SHA_SET
                        ),
                        "native_rule_evidence_sha256": pd.NA,
                        "native_rule_evidence_status": (
                            "RAW_METADATA_CAPTURED_RULE_SEMANTICS_"
                            "NOT_ADJUDICATED"
                        ),
                        "realized_outcome_comparability_evidence_sha256": (
                            pd.NA
                        ),
                        "realized_outcome_comparability_status": (
                            "UNPROVEN_NOT_ADJUDICATED"
                        ),
                        "strict_contract_rule_equivalence_evidence_sha256": (
                            pd.NA
                        ),
                        "strict_contract_rule_equivalence_status": (
                            "UNPROVEN"
                        ),
                        "landmark_seconds": 3,
                        "endpoint_seconds": 30,
                        "primary_selection_eligible": True,
                        "landmark_eligible": True,
                        "continuity_valid": True,
                        "order_ambiguous": False,
                        "censor_boundary_eligible": True,
                        "censored": False,
                        "censor_reason": "NOT_CENSORED",
                        "loss_eligible": True,
                        "availability_h": int(available),
                        "direction_h": direction,
                        "direction_eligible": available,
                        "direction_threshold_probability": 0.01,
                        "mark_l_price": 0.35 + 0.02 * episode_number,
                        "mark_l_staleness_seconds": float(episode_number),
                        "prior_30s_actual_trade_count": episode_number + 1,
                        "prior_30s_actual_trade_size": 10 + episode_number,
                        "prior_60s_actual_trade_count": episode_number + 2,
                        "prior_60s_actual_trade_size": 20 + episode_number,
                        "quarter": min(4, (episode_number // 2) + 1),
                        "is_overtime": False,
                        "game_seconds_remaining": (
                            3_600 - 250 * episode_number
                        ),
                        "score_margin_home": game_number - episode_number,
                        "possession_is_home": episode_number % 2 == 0,
                        "possession_is_home_missing": False,
                        "down": (episode_number % 4) + 1,
                        "down_missing": False,
                        "distance": 2 + episode_number,
                        "distance_missing": False,
                        "yardline_100": 80 - 8 * episode_number,
                        "yardline_100_missing": False,
                        "goal_to_go": False,
                        "pre_red_zone": episode_number >= 4,
                        "home_timeouts_remaining": 3,
                        "away_timeouts_remaining": 3,
                        "p_before_home": 0.40 + 0.02 * episode_number,
                        "p_before_home_missing": False,
                        "action_group": (
                            "PASS" if episode_number % 2 else "RUSH"
                        ),
                        "possession_result_group": (
                            "TURNOVER"
                            if episode_number == 5
                            else "RETAINED"
                        ),
                        "score_result_group": (
                            "TOUCHDOWN"
                            if episode_number == 4
                            else "NO_SCORE"
                        ),
                        "kick_result_group": "NOT_KICK",
                        "adjudication_group": "FINAL_NO_REVIEW",
                        "provisional_yards_gained": episode_number * 3,
                        "provisional_return_yards": 0,
                        "provisional_score_points_observed": (
                            6 if episode_number == 4 else 0
                        ),
                        "provisional_turnover_observed": (
                            episode_number == 5
                        ),
                        "actor_is_home": episode_number % 2 == 0,
                        "beneficiary_is_home": episode_number % 2 == 0,
                    }
                )
    frame = pd.DataFrame(rows)
    # A censored row carries no target. It must disappear from both heads.
    censored = frame.iloc[0].copy()
    censored["source_row_id"] = "row-censored"
    censored["censored"] = True
    censored["censor_reason"] = "NEXT_SALIENT_EPISODE"
    censored["loss_eligible"] = False
    censored["availability_h"] = pd.NA
    censored["direction_h"] = pd.NA
    censored["direction_eligible"] = False
    return pd.concat([frame, censored.to_frame().T], ignore_index=True)


def _verified_two_venue_panel_153(
    tmp_path: Path,
    *,
    include_kalshi: bool = True,
) -> VerifiedRegulationPanelBatch:
    source = _panel()
    poly = source.loc[source["loss_eligible"].astype(bool)].copy()
    template_game = str(poly.iloc[0]["game_id"])
    template = poly.loc[poly["game_id"].eq(template_game)].copy()
    additions: list[pd.DataFrame] = [poly]
    for index in range(105):
        game_id = f"2025_{(index % 12) + 1:02d}_X{index:03d}_Y{index:03d}"
        clone = template.copy()
        clone["game_id"] = game_id
        clone["nfl_week"] = (index % 12) + 1
        clone["source_row_id"] = [
            f"{game_id}:poly:row:{position}"
            for position in range(len(clone))
        ]
        clone["information_anchor_id"] = [
            f"{game_id}:anchor:{position}"
            for position in range(len(clone))
        ]
        clone["episode_id"] = [
            f"{game_id}:episode:{position}"
            for position in range(len(clone))
        ]
        clone["home_team"] = f"Y{index:03d}"
        clone["outcome_team"] = f"Y{index:03d}"
        clone["actual_home_contract_id"] = f"{game_id}:poly:home"
        clone["logical_market_id"] = f"{game_id}:winner"
        offset = pd.Timedelta(days=200 + index)
        for column in (
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "primary_feature_known_at",
            "prestate_known_at",
        ):
            clone[column] = pd.to_datetime(
                clone[column], utc=True
            ) + offset
        additions.append(clone)
    poly = pd.concat(additions, ignore_index=True)
    assert poly["game_id"].nunique() == 153
    kalshi = poly.copy()
    kalshi["venue"] = "KALSHI"
    kalshi["source_row_id"] = "kalshi:" + kalshi["source_row_id"].astype(str)
    kalshi["actual_home_contract_id"] = (
        "kalshi:" + kalshi["actual_home_contract_id"].astype(str)
    )
    for venue_rows in (poly, kalshi):
        venue_rows["native_rule_evidence_status"] = "VERIFIED_NATIVE_RULE"
        venue_rows["native_rule_evidence_sha256"] = SHA
        venue_rows["realized_outcome_comparability_status"] = (
            "COMPLETED_NON_TIE_REALIZATION_COMPARABLE"
        )
        venue_rows["realized_outcome_comparability_evidence_sha256"] = SHA
    prepared = prepare_regulation_panel(
        (
            pd.concat([poly, kalshi], ignore_index=True)
            if include_kalshi
            else poly
        )
    )
    manifest = VerifiedRegulationPanelManifest(
        output_root=tmp_path,
        batch_manifest_path=tmp_path / "panel.batch-index.json",
        batch_manifest_file_sha256=SHA,
        batch_sha256=SHA,
        game_count=153,
        panel_row_count=len(prepared),
        loss_eligible_count=len(prepared),
        cohort_authority_sha256=SHA,
        cohort_mapping_sha256=SHA,
        sources={"panel": SHA},
        games=(),
    )
    return VerifiedRegulationPanelBatch(
        frame=prepared,
        manifest=manifest,
        source_row_count=len(prepared),
        loaded_venue_scope=(
            ("POLYMARKET", "KALSHI")
            if include_kalshi
            else ("POLYMARKET",)
        ),
        loaded_row_count=len(prepared),
        loss_eligible_count=len(prepared),
    )


def _verified_test_selection_lock(
    panel: VerifiedRegulationPanelBatch,
) -> VerifiedRegulationSelectionLock:
    cells = tuple(
        (
            f"fold_{fold:02d}",
            model_id,
            feature_block_id,
        )
        for fold in range(1, 6)
        for model_id, feature_block_id in (
            ("b0_empirical_v1", "D0"),
            ("regularized_logistic_v1", "D2"),
        )
    )
    selection_frame = panel.frame.loc[
        panel.frame["venue"].eq("POLYMARKET")
    ].reset_index(drop=True)
    selection_panel = replace(
        panel,
        frame=selection_frame,
        loaded_venue_scope=("POLYMARKET",),
        loaded_row_count=len(selection_frame),
        loss_eligible_count=len(selection_frame),
    )
    native = run_regulation_walk_forward(selection_panel, cells=cells)
    hashes: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for model_id, feature_block_id in (
        ("b0_empirical_v1", "D0"),
        ("regularized_logistic_v1", "D2"),
    ):
        hashes[(model_id, feature_block_id)] = {
            str(row.fold_id): {
                column: str(getattr(row, column))
                for column in TRAINING_PROVENANCE_COLUMNS
            }
            for row in native.support_audit.loc[
                native.support_audit["model_id"].eq(model_id)
                & native.support_audit["feature_block_id"].eq(
                    feature_block_id
                )
            ].itertuples(index=False)
        }
    winner = FrozenPolymarketWinnerV2(
        model_id="regularized_logistic_v1",
        feature_block_id="D2",
        polymarket_selection_sha256=SHA,
        anchor_kind="PROVISIONAL_FIRST_SEEN",
        polymarket_fold_training_hashes=hashes[
            ("regularized_logistic_v1", "D2")
        ],
    )
    return VerifiedRegulationSelectionLock(
        manifest_path=panel.manifest.output_root / "selection-lock.json",
        manifest_file_sha256=SHA,
        selection_lock_sha256=SHA,
        selection_sha256=SHA,
        decision_status="WINNER_FROZEN",
        winner=winner,
        kalshi_transport_allowed=True,
        baseline_fold_training_hashes=hashes[
            ("b0_empirical_v1", "D0")
        ],
        fold_definitions={},
        panel_batch_manifest_file_sha256=(
            panel.manifest.batch_manifest_file_sha256
        ),
        panel_batch_sha256=panel.manifest.batch_sha256,
        cohort_authority_sha256=panel.manifest.cohort_authority_sha256,
        cohort_mapping_sha256=panel.manifest.cohort_mapping_sha256,
        landmark_seconds=(1, 2, 3, 5, 10),
        endpoint_seconds=tuple(range(5, 61, 5)),
        direction_threshold_probability=0.01,
        target_version="REGULATION_AVAILABILITY_AND_DIRECTION_V1",
    )


def _verified_test_native_rule_overlay(
    panel: VerifiedRegulationPanelBatch,
) -> object:
    contract_columns = [
        "game_id",
        "venue",
        "home_team",
        "actual_home_contract_id",
        "logical_market_id",
        "actual_home_contract_identity_sha256",
        "native_raw_manifest_set_sha256",
        "native_raw_object_set_sha256",
    ]
    contracts = (
        panel.frame.loc[:, contract_columns]
        .drop_duplicates()
        .sort_values(["game_id", "venue"], kind="mergesort")
        .reset_index(drop=True)
    )
    bindings: list[dict[str, object]] = []
    for contract in contracts.to_dict("records"):
        game_id = str(contract["game_id"])
        venue = str(contract["venue"])
        parts = game_id.split("_", 3)
        assert len(parts) == 4
        pair_evidence = _canonical_digest(
            {"game_id": game_id, "gate": "COMPARABLE"}
        )
        binding = {
            "game_id": game_id,
            "venue": venue,
            "panel_home_team": str(contract["home_team"]),
            "panel_away_team": parts[2],
            "panel_actual_home_contract_id": str(
                contract["actual_home_contract_id"]
            ),
            "panel_logical_market_id": str(
                contract["logical_market_id"]
            ),
            "panel_contract_identity_sha256": str(
                contract["actual_home_contract_identity_sha256"]
            ),
            "panel_raw_manifest_set_sha256": str(
                contract["native_raw_manifest_set_sha256"]
            ),
            "panel_raw_object_set_sha256": str(
                contract["native_raw_object_set_sha256"]
            ),
            "native_contract_ids_json": json.dumps(
                [str(contract["actual_home_contract_id"])],
                separators=(",", ":"),
            ),
            "native_contract_identity_sha256": _canonical_digest(
                {
                    "game_id": game_id,
                    "venue": venue,
                    "native_contract": str(
                        contract["actual_home_contract_id"]
                    ),
                }
            ),
            "native_rule_text_sha256": SHA,
            "native_metadata_manifest_set_sha256": SHA,
            "native_metadata_object_set_sha256": SHA,
            "native_request_set_sha256": SHA,
            "native_rule_evidence_sha256": _canonical_digest(
                {"game_id": game_id, "venue": venue, "rule": "verified"}
            ),
            "native_market_membership_status": "VERIFIED",
            "native_market_membership_evidence_sha256": _canonical_digest(
                {
                    "game_id": game_id,
                    "venue": venue,
                    "panel_contract": str(
                        contract["actual_home_contract_id"]
                    ),
                }
            ),
            "pair_completed_non_tie_comparability_status": "COMPARABLE",
            "pair_completed_non_tie_comparability_evidence_sha256": (
                pair_evidence
            ),
            "pair_strict_rule_equivalence_status": "NON_EQUIVALENT",
            "pair_strict_rule_equivalence_evidence_sha256": (
                _canonical_digest(
                    {"game_id": game_id, "gate": "NON_EQUIVALENT"}
                )
            ),
            "pair_evidence_row_sha256": _canonical_digest(
                {"game_id": game_id, "pair": "audited"}
            ),
        }
        binding["cross_binding_sha256"] = _canonical_digest(binding)
        bindings.append(binding)
    binding_hashes = [
        str(binding["cross_binding_sha256"]) for binding in bindings
    ]
    return regulation_models.VerifiedNativeRuleOverlay(
        manifest_path=(
            panel.manifest.output_root
            / "native-rule-overlay.manifest.json"
        ),
        manifest_file_sha256=SHA,
        batch_sha256=SHA,
        source_capture_batch_file_sha256=SHA,
        panel_batch_manifest_file_sha256=(
            panel.manifest.batch_manifest_file_sha256
        ),
        panel_batch_sha256=panel.manifest.batch_sha256,
        panel_cohort_authority_sha256=(
            panel.manifest.cohort_authority_sha256
        ),
        panel_contract_authority_sha256=(
            regulation_models._panel_transport_contract_authority_sha256(
                panel.frame
            )
        ),
        game_count=153,
        venue_evidence_row_count=306,
        contract_pair_row_count=153,
        completed_non_tie_comparable_count=153,
        strict_rule_equivalent_count=0,
        cross_binding_set_sha256=_canonical_digest(binding_hashes),
        bindings=tuple(bindings),
    )


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: object) -> str:
    return _digest(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _one_game_verified_bundle(
    tmp_path: Path,
    *,
    include_kalshi: bool = False,
) -> tuple[Path, str, Path]:
    frame = _panel()
    game_id = str(frame.iloc[0]["game_id"])
    frame = frame.loc[frame["game_id"].eq(game_id)].reset_index(drop=True)
    if include_kalshi:
        kalshi = frame.copy()
        kalshi["venue"] = "KALSHI"
        kalshi["source_row_id"] = (
            "kalshi:" + kalshi["source_row_id"].astype(str)
        )
        kalshi["actual_home_contract_id"] = (
            "kalshi:"
            + kalshi["actual_home_contract_id"].astype(str)
        )
        frame = pd.concat((frame, kalshi), ignore_index=True)
    source_hashes = {
        "information_anchor_manifest_sha256": SHA,
        "information_anchor_object_sha256": SHA,
        "context_manifest_sha256": SHA,
        "context_object_sha256": SHA,
        "market_game_manifest_sha256": SHA,
        "market_observations_object_sha256": SHA,
        "market_inventory_object_sha256": SHA,
        "market_capture_batch_sha256": SHA,
        "market_capture_index_sha256": SHA,
        "market_capture_checkpoint_sha256": SHA,
        "cohort_authority_sha256": SHA,
    }
    attrition = (
        frame.groupby(
            [
                "game_id",
                "landmark_seconds",
                "endpoint_seconds",
                "censor_reason",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("row_count")
        .reset_index()
        .rename(columns={"censor_reason": "attrition_reason"})
    )
    attrition.insert(
        0,
        "schema_version",
        "RegulationDecisionPanelAttritionV2",
    )
    published = publish_regulation_panel_v4_frames(
        project_root=tmp_path,
        output_root=Path("regulation-decision-panel-v4"),
        game_panels={
            game_id: RegulationAnchorGamePanel(
                panel=frame,
                attrition=attrition,
            )
        },
        game_source_hashes={game_id: source_hashes},
        batch_sources={
            "market_batch_file_sha256": SHA,
            "market_capture_batch_sha256": SHA,
            "information_anchor_manifest_file_sha256": SHA,
            "information_anchor_object_sha256": SHA,
            "context_manifest_file_sha256": SHA,
            "context_object_sha256": SHA,
            "cohort_authority_sha256": SHA,
            "cohort_mapping_sha256": SHA,
        },
        expected_game_count=1,
    )
    batch_document = json.loads(
        published.batch_manifest_path.read_text(encoding="utf-8")
    )
    game_entry = batch_document["games"][0]
    game_manifest_path = (
        published.output_root / game_entry["manifest_path"]
    )
    game_document = json.loads(
        game_manifest_path.read_text(encoding="utf-8")
    )
    panel_descriptor = next(
        descriptor
        for descriptor in game_document["tables"]
        if descriptor["name"] == "regulation_decision_panel"
    )
    object_path = (
        published.output_root / panel_descriptor["object_path"]
    )
    return (
        published.batch_manifest_path,
        published.batch_manifest_sha256,
        object_path,
    )


def _fake_complete_shards(
    tmp_path: Path,
    verified_panel: VerifiedRegulationPanelBatch | None = None,
    *,
    passing_candidates: bool = True,
) -> tuple[VerifiedRegulationOOFShard, ...]:
    shards: list[VerifiedRegulationOOFShard] = []
    passing = (
        {
            ("regularized_logistic_v1", "D1"): 0.10,
            ("regularized_logistic_v1", "D2"): 0.10,
        }
        if passing_candidates
        else {}
    )
    for fold_id, model_id, feature_block_id in CORRECTED_MODEL_CELLS:
        fold_number = int(fold_id[-2:])
        improvement = passing.get((model_id, feature_block_id), -0.05)
        if model_id == "b0_empirical_v1":
            improvement = 0.0
        rows = []
        for game_number in range(8):
            game_id = f"selection-{fold_id}-game-{game_number}"
            anchor_id = f"{game_id}:anchor"
            rows.append(
                {
                    "source_row_id": f"{game_id}:row",
                    "game_id": game_id,
                    "nfl_week": 2 * fold_number + 1,
                    "information_anchor_id": anchor_id,
                    "episode_id": f"{game_id}:episode",
                    "anchor_kind": "PROVISIONAL_FIRST_SEEN",
                    "analysis_role": "PRIMARY_SELECTION",
                    "venue": "POLYMARKET",
                    "actual_home_contract_id": f"{game_id}:home",
                    "landmark_seconds": 3,
                    "endpoint_seconds": 30,
                    "availability_truth": 1,
                    "direction_truth": "UP",
                    "cohort_authority_sha256": SHA,
                    "panel_batch_sha256": SHA,
                    "fold_id": fold_id,
                    "model_id": model_id,
                    "feature_block_id": feature_block_id,
                    "availability_log_loss": 0.40 - improvement,
                    "availability_brier": 0.20 - improvement,
                    "direction_log_loss": 0.80 - improvement,
                    "direction_multiclass_brier": 0.50 - improvement,
                    "joint_log_loss": 1.20 - improvement,
                    **{
                        column: SHA
                        for column in TRAINING_PROVENANCE_COLUMNS
                    },
                }
            )
        predictions = pd.DataFrame(rows)
        support = pd.DataFrame(
            [
                {
                    "fold_id": fold_id,
                    "model_id": model_id,
                    "feature_block_id": feature_block_id,
                    **{
                        column: SHA
                        for column in TRAINING_PROVENANCE_COLUMNS
                    },
                }
            ]
        )
        run_config: dict[str, object] = {}
        if verified_panel is not None:
            poly = verified_panel.frame.loc[
                verified_panel.frame["venue"].eq("POLYMARKET")
            ]
            from prediction_market.research.nfl_x15_regulation_models import (
                _fold_map,
                _sha256,
            )

            fold = _fold_map(poly)[fold_id]
            fold_sha = _sha256(
                {
                    "fold_id": fold.fold_id,
                    "train_weeks": fold.train_weeks,
                    "validation_weeks": fold.validation_weeks,
                    "train_game_ids": fold.train_game_ids,
                    "validation_game_ids": fold.validation_game_ids,
                }
            )
            support.loc[:, "fold_sha256"] = fold_sha
            run_config = {
                "formal_input_verified": True,
                "input_lineage": {
                    "panel_batch_manifest_file_sha256": (
                        verified_panel.manifest
                        .batch_manifest_file_sha256
                    ),
                    "panel_batch_sha256": (
                        verified_panel.manifest.batch_sha256
                    ),
                },
            }
        training_hashes = {
            column: str(support.iloc[0][column])
            for column in TRAINING_PROVENANCE_COLUMNS
        }
        for column, value in training_hashes.items():
            predictions.loc[:, column] = value
        cell_sha = _digest(
            f"{fold_id}:{model_id}:{feature_block_id}".encode()
        )
        output_root = tmp_path / "fake-selection-shards"
        payload, schema_fingerprint = regulation_models._parquet_bytes(
            predictions
        )
        prediction_descriptor = regulation_models._publish_model_object(
            output_root=output_root,
            cell=(fold_id, model_id, feature_block_id),
            logical_name="oof_predictions",
            payload=payload,
            suffix="parquet",
            row_count=len(predictions),
            schema_fingerprint=schema_fingerprint,
        )
        shards.append(
            VerifiedRegulationOOFShard(
                output_root=output_root,
                manifest_path=Path(
                    f"/private/tmp/{fold_id}-{model_id}-"
                    f"{feature_block_id}.json"
                ),
                manifest_file_sha256=cell_sha,
                shard_sha256=cell_sha,
                cell=(fold_id, model_id, feature_block_id),
                oof_prediction_row_count=len(predictions),
                object_descriptors={
                    "oof_predictions": prediction_descriptor
                },
                training_hashes=training_hashes,
                support_metadata=support.iloc[0].to_dict(),
                run_config=run_config,
            )
        )
    return tuple(shards)


def test_corrected_matrix_is_exactly_45_cells_and_b0_is_d0_only() -> None:
    assert len(CORRECTED_MODEL_CELLS) == 45
    assert len(set(CORRECTED_MODEL_CELLS)) == 45
    assert {
        (model_id, block_id)
        for _, model_id, block_id in CORRECTED_MODEL_CELLS
    } == {
        ("b0_empirical_v1", "D0"),
        ("regularized_logistic_v1", "D0"),
        ("regularized_logistic_v1", "D1"),
        ("regularized_logistic_v1", "D2"),
        ("regularized_logistic_v1", "D3"),
        ("shallow_xgboost_v1", "D0"),
        ("shallow_xgboost_v1", "D1"),
        ("shallow_xgboost_v1", "D2"),
        ("shallow_xgboost_v1", "D3"),
    }
    partial = regulation_oof_batch_status(CORRECTED_MODEL_CELLS[:7])
    complete = regulation_oof_batch_status(CORRECTED_MODEL_CELLS)
    assert partial["run_status"] == "PARTIAL"
    assert partial["publication_gate"] == "BLOCKED_INCOMPLETE"
    assert partial["completed_cell_count"] == 7
    assert complete["run_status"] == "COMPLETE"
    assert complete["publication_gate"] == "PASS"
    assert complete["missing_cells"] == ()


def test_selection_is_exact_paired_game_bootstrap_and_one_se(
    tmp_path: Path,
) -> None:
    result = select_regulation_model(_fake_complete_shards(tmp_path))

    assert result.decision_status == "WINNER_FROZEN"
    assert result.winner is not None
    assert result.winner.model_id == "regularized_logistic_v1"
    assert result.winner.feature_block_id == "D1"
    assert result.winner.anchor_kind == "PROVISIONAL_FIRST_SEEN"
    assert set(result.winner.polymarket_fold_training_hashes) == {
        f"fold_{fold:02d}" for fold in range(1, 6)
    }
    selected = next(
        candidate
        for candidate in result.candidate_audit
        if (
            candidate.model_id,
            candidate.feature_block_id,
        )
        == ("regularized_logistic_v1", "D1")
    )
    assert selected.game_count == 40
    assert selected.anchor_game_count == 40
    assert selected.availability_non_regression is True
    assert selected.candidate_gate_passed is True
    assert result.holdout_reaction_accessed is False


def test_selection_pairs_only_one_fold_baseline_and_candidate_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = _fake_complete_shards(tmp_path)
    original_load = (
        regulation_models
        .load_verified_regulation_oof_prediction_projection
    )
    original_pair = regulation_models._paired_candidate_rows
    live_cells: list[tuple[str, str, str]] = []
    peak = 0

    def tracked_load(
        shard: VerifiedRegulationOOFShard,
        *,
        columns: tuple[str, ...],
    ) -> pd.DataFrame:
        nonlocal peak
        live_cells.append(shard.cell)
        peak = max(peak, len(live_cells))
        assert len(live_cells) <= 2
        return original_load(shard, columns=columns)

    def tracked_pair(
        *,
        baseline: pd.DataFrame,
        candidate: pd.DataFrame,
    ) -> pd.DataFrame:
        assert len(live_cells) == 2
        assert live_cells[0][0] == live_cells[1][0]
        assert live_cells[0][1:] == ("b0_empirical_v1", "D0")
        paired = original_pair(
            baseline=baseline,
            candidate=candidate,
        )
        live_cells.clear()
        return paired

    monkeypatch.setattr(
        regulation_models,
        "load_verified_regulation_oof_prediction_projection",
        tracked_load,
    )
    monkeypatch.setattr(
        regulation_models,
        "_paired_candidate_rows",
        tracked_pair,
    )
    select_regulation_model(shards)
    assert peak == 2
    assert live_cells == []


def test_selection_lock_rederives_exact_45_and_binds_b0(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=False
    )
    shards = _fake_complete_shards(tmp_path, panel)
    published = publish_regulation_selection_lock(
        project_root=tmp_path,
        output_root=tmp_path / "oof",
        verified_panel=panel,
        shards=shards,
    )
    assert isinstance(published, PublishedRegulationSelectionLock)
    verified = verify_regulation_selection_lock(
        project_root=tmp_path,
        output_root=tmp_path / "oof",
        manifest_path=published.manifest_path,
        manifest_file_sha256=published.manifest_file_sha256,
        verified_panel=panel,
        shards=shards,
    )
    assert verified.winner.model_id == "regularized_logistic_v1"
    assert verified.winner.feature_block_id == "D1"
    assert set(verified.baseline_fold_training_hashes) == {
        f"fold_{fold:02d}" for fold in range(1, 6)
    }
    assert verified.landmark_seconds == (1, 2, 3, 5, 10)
    assert verified.endpoint_seconds == tuple(range(5, 61, 5))
    assert verified.direction_threshold_probability == 0.01
    assert verified.target_version == (
        "REGULATION_AVAILABILITY_AND_DIRECTION_V1"
    )


def test_no_model_advance_lock_is_terminal_and_blocks_kalshi(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=False
    )
    shards = _fake_complete_shards(
        tmp_path,
        panel,
        passing_candidates=False,
    )
    selected = select_regulation_model(shards)
    assert selected.decision_status == "NO_CANDIDATE_PASSED"
    assert selected.winner is None
    assert all(
        not candidate.candidate_gate_passed
        for candidate in selected.candidate_audit
    )

    published = publish_regulation_selection_lock(
        project_root=tmp_path,
        output_root=tmp_path / "oof-no-advance",
        verified_panel=panel,
        shards=shards,
    )
    assert published.decision_status == "NO_MODEL_ADVANCE"
    assert published.winner_frozen is False
    verified = verify_regulation_selection_lock(
        project_root=tmp_path,
        output_root=tmp_path / "oof-no-advance",
        manifest_path=published.manifest_path,
        manifest_file_sha256=published.manifest_file_sha256,
        verified_panel=panel,
        shards=shards,
    )
    assert verified.decision_status == "NO_MODEL_ADVANCE"
    assert verified.winner is None
    assert verified.kalshi_transport_allowed is False
    with pytest.raises(
        RegulationModelInputError,
        match="NO_MODEL_ADVANCE",
    ):
        validate_kalshi_transport(
            panel,
            selection_lock=verified,
            native_rule_overlay=None,
        )


def test_prepare_excludes_censoring_from_both_heads_and_forbids_old_s_h() -> None:
    source = _panel()
    prepared = prepare_regulation_panel(source)

    assert "row-censored" not in set(prepared["source_row_id"])
    assert len(prepared) == len(source) - 1
    assert prepared["availability_h"].notna().all()
    assert prepared.loc[
        ~prepared["availability_h"], "direction_h"
    ].isna().all()
    assert prepared.loc[
        prepared["availability_h"], "direction_h"
    ].isin(DIRECTION_CLASSES).all()

    contaminated = source.assign(s_h=True)
    with pytest.raises(RegulationModelInputError, match="forbidden legacy"):
        prepare_regulation_panel(contaminated)


def test_prepare_accepts_panel_native_int8_target_and_nullable_orientations() -> None:
    source = _panel()
    source["availability_h"] = pd.array(
        source["availability_h"], dtype="Int8"
    )
    source["direction_h"] = source["direction_h"].astype("string")
    source.loc[3, "actor_is_home"] = pd.NA
    source.loc[3, "beneficiary_is_home"] = pd.NA
    source.loc[3, "possession_is_home"] = pd.NA
    source.loc[3, "possession_is_home_missing"] = True

    prepared = prepare_regulation_panel(source)

    assert prepared["availability_h"].dtype == bool
    assert pd.isna(prepared.loc[
        prepared["source_row_id"].eq("row-4"), "actor_is_home"
    ]).all()


def test_prestate_missing_indicators_are_explicit_and_consistent() -> None:
    source = _panel()
    source.loc[3, "down"] = pd.NA
    with pytest.raises(RegulationModelInputError, match="down_missing"):
        prepare_regulation_panel(source)

    source.loc[3, "down_missing"] = True
    source.loc[4, "distance"] = pd.NA
    source.loc[4, "distance_missing"] = True
    source.loc[5, "yardline_100"] = pd.NA
    source.loc[5, "yardline_100_missing"] = True
    prepared = prepare_regulation_panel(source)
    for row_id, missing_column in (
        ("row-4", "down_missing"),
        ("row-5", "distance_missing"),
        ("row-6", "yardline_100_missing"),
    ):
        assert prepared.loc[
            prepared["source_row_id"].eq(row_id),
            missing_column,
        ].all()


def test_prepare_fails_closed_on_ot_or_holdout_rows() -> None:
    overtime = _panel()
    overtime.loc[1, "is_overtime"] = True
    overtime.loc[1, "quarter"] = 5
    with pytest.raises(RegulationModelInputError, match="regulation-only"):
        prepare_regulation_panel(overtime)

    holdout = _panel()
    holdout.loc[1, "dataset_split"] = "HOLDOUT"
    with pytest.raises(RegulationModelInputError, match="DEVELOPMENT"):
        prepare_regulation_panel(holdout)

    finalized = _panel().assign(final_score_points=7)
    with pytest.raises(
        RegulationModelInputError,
        match="finalized primary fields",
    ):
        prepare_regulation_panel(finalized)


def test_anchor_kind_and_analysis_role_cannot_be_mixed() -> None:
    invalid = _panel()
    invalid.loc[1, "anchor_kind"] = "FINALIZED"
    with pytest.raises(RegulationModelInputError, match="PROVISIONAL"):
        prepare_regulation_panel(invalid)

    diagnostic = _panel()
    diagnostic["anchor_kind"] = "FINALIZED"
    diagnostic["analysis_role"] = "RESIDUAL_CORRECTION_DIAGNOSTIC"
    with pytest.raises(RegulationModelInputError, match="PROVISIONAL"):
        prepare_regulation_panel(diagnostic)


def test_joint_loss_is_availability_plus_observed_direction_only() -> None:
    truth_availability = np.asarray([0, 1, 1], dtype=int)
    probability_availability = np.asarray([0.20, 0.70, 0.80])
    truth_direction = np.asarray([None, "UP", "DOWN"], dtype=object)
    probability_direction = np.asarray(
        [
            [0.10, 0.20, 0.70],
            [0.10, 0.20, 0.70],
            [0.60, 0.30, 0.10],
        ]
    )

    result = regulation_row_losses(
        availability_truth=truth_availability,
        availability_probability=probability_availability,
        direction_truth=truth_direction,
        direction_probability=probability_direction,
    )

    assert result["availability_log_loss"].tolist() == pytest.approx(
        [-math.log(0.8), -math.log(0.7), -math.log(0.8)]
    )
    assert math.isnan(result.loc[0, "direction_log_loss"])
    assert result.loc[1:, "direction_log_loss"].tolist() == pytest.approx(
        [-math.log(0.7), -math.log(0.6)]
    )
    assert result["joint_log_loss"].tolist() == pytest.approx(
        [
            -math.log(0.8),
            -math.log(0.7) - math.log(0.7),
            -math.log(0.8) - math.log(0.6),
        ]
    )
    assert not any("s_h" in column.lower() for column in result.columns)


def test_equal_game_metrics_do_not_allow_large_game_to_dominate() -> None:
    rows = pd.DataFrame(
        {
            "game_id": ["small", "large", "large", "large"],
            "information_anchor_id": [
                "small-a",
                "large-a",
                "large-a",
                "large-b",
            ],
            "availability_log_loss": [0.0, 1.0, 1.0, 1.0],
            "availability_brier": [0.0, 1.0, 1.0, 1.0],
            "direction_log_loss": [0.0, 1.0, 1.0, 1.0],
            "direction_multiclass_brier": [0.0, 1.0, 1.0, 1.0],
            "joint_log_loss": [0.0, 2.0, 2.0, 2.0],
        }
    )

    metrics = compute_equal_game_metrics(rows)

    assert metrics["availability_log_loss"] == pytest.approx(0.5)
    assert metrics["availability_brier"] == pytest.approx(0.5)
    assert metrics["direction_log_loss"] == pytest.approx(0.5)
    assert metrics["direction_multiclass_brier"] == pytest.approx(0.5)
    assert metrics["joint_log_loss"] == pytest.approx(1.0)


def test_one_cell_walk_forward_is_polymarket_only_and_has_two_heads() -> None:
    run = run_regulation_walk_forward(
        _panel(),
        cells=(
            ("fold_01", "regularized_logistic_v1", "D2"),
        ),
    )

    assert not run.oof_predictions.empty
    assert set(run.oof_predictions["venue"]) == {"POLYMARKET"}
    assert set(run.oof_predictions["fold_id"]) == {"fold_01"}
    assert set(run.oof_predictions["model_id"]) == {
        "regularized_logistic_v1"
    }
    assert set(run.oof_predictions["feature_block_id"]) == {"D2"}
    assert run.oof_predictions["availability_probability"].between(
        0, 1
    ).all()
    direction = run.oof_predictions[
        [
            "direction_probability_down",
            "direction_probability_no_move",
            "direction_probability_up",
        ]
    ].to_numpy(dtype=float)
    assert np.allclose(direction.sum(axis=1), 1.0)
    assert not any(
        "s_h" in column.lower() for column in run.oof_predictions.columns
    )
    assert {
        "availability_log_loss",
        "availability_brier",
        "direction_log_loss",
        "direction_multiclass_brier",
        "joint_log_loss",
    }.issubset(run.fold_metrics["metric_name"])
    assert {
        "fold_sha256",
        "availability_preprocessor_training_sha256",
        "direction_preprocessor_training_sha256",
        "availability_model_training_sha256",
        "direction_model_training_sha256",
        "availability_calibration_training_sha256",
        "direction_calibration_training_sha256",
    }.issubset(run.support_audit.columns)
    for column in (
        "fold_sha256",
        "availability_preprocessor_training_sha256",
        "direction_preprocessor_training_sha256",
        "availability_model_training_sha256",
        "direction_model_training_sha256",
        "availability_calibration_training_sha256",
        "direction_calibration_training_sha256",
    ):
        assert run.support_audit[column].str.fullmatch(
            r"sha256:[0-9a-f]{64}"
        ).all()
    row = run.support_audit.iloc[0]
    assert row["producer_max_week"] < row["calibration_week"]
    assert (
        row["availability_calibrator_base_preprocessor_sha256"]
        == row["availability_preprocessor_training_sha256"]
    )
    assert (
        row["direction_calibrator_base_preprocessor_sha256"]
        == row["direction_preprocessor_training_sha256"]
    )
    assert (
        row["availability_calibrator_base_model_sha256"]
        == row["availability_model_training_sha256"]
    )
    assert (
        row["direction_calibrator_base_model_sha256"]
        == row["direction_model_training_sha256"]
    )
    with pytest.raises(RegulationModelInputError, match="verified PanelV4"):
        publish_regulation_oof_shard(
            model_run=run,
            output_root=Path("/private/tmp/never-written"),
        )


def test_atomic_one_cell_shard_is_verified_and_resumable(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=False
    )
    output_root = tmp_path / "oof"
    run = run_regulation_walk_forward(
        panel,
        cells=(("fold_01", "b0_empirical_v1", "D0"),),
    )
    published = publish_regulation_oof_shard(
        model_run=run,
        output_root=output_root,
        verified_panel=panel,
    )
    verified = verify_regulation_oof_shard(
        project_root=tmp_path,
        output_root=output_root,
        manifest_path=published.manifest_path,
        manifest_file_sha256=published.manifest_file_sha256,
        verified_panel=panel,
    )
    assert verified.cell == ("fold_01", "b0_empirical_v1", "D0")
    assert not any(
        isinstance(getattr(verified, field), pd.DataFrame)
        for field in verified.__dataclass_fields__
    )

    discovered = discover_verified_regulation_oof_shards(
        project_root=tmp_path,
        output_root=output_root,
        verified_panel=panel,
    )
    assert tuple(shard.cell for shard in discovered) == (verified.cell,)
    batch = publish_regulation_oof_batch_index(
        project_root=tmp_path,
        output_root=output_root,
        verified_panel=panel,
        shards=discovered,
    )
    assert batch.run_status == "PARTIAL"
    assert batch.publication_gate == "BLOCKED_INCOMPLETE"
    assert batch.selection_ready is False
    assert tuple(
        shard.manifest_file_sha256
        for shard in discover_verified_regulation_oof_shards(
            project_root=tmp_path,
            output_root=output_root,
            verified_panel=panel,
        )
    ) == (published.manifest_file_sha256,)


def test_resume_does_not_materialize_completed_prediction_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=False
    )
    output_root = tmp_path / "oof-metadata-only"
    run = run_regulation_walk_forward(
        panel,
        cells=(("fold_01", "b0_empirical_v1", "D0"),),
    )
    publish_regulation_oof_shard(
        model_run=run,
        output_root=output_root,
        verified_panel=panel,
    )
    loaded_names: list[str] = []
    original = regulation_models._load_verified_shard_object

    def recording_loader(**kwargs: object) -> object:
        loaded_names.append(str(kwargs["logical_name"]))
        return original(**kwargs)

    monkeypatch.setattr(
        regulation_models,
        "_load_verified_shard_object",
        recording_loader,
    )
    discovered = discover_verified_regulation_oof_shards(
        project_root=tmp_path,
        output_root=output_root,
        verified_panel=panel,
    )

    assert len(discovered) == 1
    assert "oof_predictions" not in loaded_names
    assert "fold_metrics" not in loaded_names


def _republish_mutated_shard_object(
    *,
    output_root: Path,
    published: object,
    logical_name: str,
    mutate: object,
) -> tuple[Path, str]:
    document = json.loads(
        published.manifest_path.read_text(encoding="utf-8")
    )
    cell = (
        document["cell"]["fold_id"],
        document["cell"]["model_id"],
        document["cell"]["feature_block_id"],
    )
    descriptor = next(
        value
        for value in document["objects"]
        if value["logical_name"] == logical_name
    )
    frame = pq.read_table(
        output_root / descriptor["object_path"]
    ).to_pandas()
    mutate(frame)
    payload, schema_fingerprint = regulation_models._parquet_bytes(frame)
    replacement = regulation_models._publish_model_object(
        output_root=output_root,
        cell=cell,
        logical_name=logical_name,
        payload=payload,
        suffix="parquet",
        row_count=len(frame),
        schema_fingerprint=schema_fingerprint,
    )
    document["objects"] = [
        replacement if value["logical_name"] == logical_name else value
        for value in document["objects"]
    ]
    material = {
        key: value
        for key, value in document.items()
        if key != "shard_sha256"
    }
    document["shard_sha256"] = regulation_models._sha256(material)
    manifest_payload = regulation_models._canonical_json_bytes(document)
    file_sha256 = "sha256:" + hashlib.sha256(
        manifest_payload
    ).hexdigest()
    hexadecimal = file_sha256.removeprefix("sha256:")
    path = (
        output_root
        / "shards"
        / cell[0]
        / cell[1]
        / cell[2]
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.shard-manifest.json"
    )
    regulation_models._atomic_publish(path, manifest_payload)
    return path, file_sha256


@pytest.mark.parametrize(
    ("logical_name", "mutate", "message"),
    (
        (
            "oof_predictions",
            lambda frame: frame.__setitem__(
                "game_id",
                frame["game_id"].mask(
                    frame.index == frame.index[0],
                    "self-consistent-wrong-game",
                ),
            ),
            "identity/truth",
        ),
        (
            "oof_predictions",
            lambda frame: frame.__setitem__(
                "training_venue", "KALSHI"
            ),
            "one primary cell",
        ),
        (
            "support_audit",
            lambda frame: frame.__setitem__(
                "effective_random_state",
                frame["effective_random_state"] + 1,
            ),
            "membership/chronology",
        ),
        (
            "support_audit",
            lambda frame: frame.__setitem__(
                "producer_training_game_ids", "[]"
            ),
            "membership/chronology",
        ),
    ),
)
def test_resume_rejects_self_consistent_wrong_semantic_shards(
    tmp_path: Path,
    logical_name: str,
    mutate: object,
    message: str,
) -> None:
    panel = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=False
    )
    output_root = tmp_path / f"wrong-{logical_name}-{message}"
    run = run_regulation_walk_forward(
        panel,
        cells=(("fold_01", "b0_empirical_v1", "D0"),),
    )
    published = publish_regulation_oof_shard(
        model_run=run,
        output_root=output_root,
        verified_panel=panel,
    )
    manifest_path, manifest_sha256 = _republish_mutated_shard_object(
        output_root=output_root,
        published=published,
        logical_name=logical_name,
        mutate=mutate,
    )
    with pytest.raises(RegulationModelInputError, match=message):
        verify_regulation_oof_shard(
            project_root=tmp_path,
            output_root=output_root,
            manifest_path=manifest_path,
            manifest_file_sha256=manifest_sha256,
            verified_panel=panel,
        )


def test_parquet_fingerprint_binds_exact_persisted_schema() -> None:
    frame = pd.DataFrame(
        {
            "native_string": pd.Series(
                ["alpha", pd.NA, "gamma"],
                dtype="category",
            ),
            "nullable_flag": pd.Series(
                [True, pd.NA, False],
                dtype="boolean",
            ),
            "nullable_integer": pd.Series(
                [1, pd.NA, 3],
                dtype="Int8",
            ),
        }
    )
    payload, observed = _parquet_bytes(frame)
    persisted_schema = pq.read_schema(pa.BufferReader(payload))
    expected = "sha256:" + hashlib.sha256(
        persisted_schema.remove_metadata().serialize().to_pybytes()
    ).hexdigest()
    assert observed == expected


def test_b0_has_no_learned_calibration_and_uses_all_outer_train() -> None:
    run = run_regulation_walk_forward(
        _panel(),
        cells=(("fold_01", "b0_empirical_v1", "D0"),),
    )
    support = run.support_audit.iloc[0]
    assert support["availability_calibration_status"] == (
        "NOT_APPLICABLE_EMPIRICAL_BASELINE"
    )
    assert support["direction_calibration_status"] == (
        "NOT_APPLICABLE_EMPIRICAL_BASELINE"
    )
    assert support["calibration_game_count"] == 0
    assert (
        support["producer_training_game_count"]
        == support["outer_training_game_count"]
    )


def test_training_is_invariant_to_duplicate_rows_inside_one_anchor() -> None:
    original = _panel()
    duplicated = original.copy()
    source = duplicated.loc[
        duplicated["nfl_week"].eq(1)
        & duplicated["source_row_id"].ne("row-censored")
    ].iloc[1].copy()
    source["source_row_id"] = "row-duplicate-same-anchor"
    duplicated = pd.concat(
        [duplicated, source.to_frame().T], ignore_index=True
    )
    cell = (("fold_01", "regularized_logistic_v1", "D2"),)
    base = run_regulation_walk_forward(original, cells=cell)
    repeated = run_regulation_walk_forward(duplicated, cells=cell)
    columns = [
        "availability_probability",
        "direction_probability_down",
        "direction_probability_no_move",
        "direction_probability_up",
    ]
    left = base.oof_predictions.sort_values("source_row_id")
    right = repeated.oof_predictions.sort_values("source_row_id")
    assert left["source_row_id"].tolist() == right[
        "source_row_id"
    ].tolist()
    assert np.allclose(
        left[columns].to_numpy(dtype=float),
        right[columns].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_direction_preprocessor_uses_only_direction_risk_set_weights() -> None:
    prepared = prepare_regulation_panel(_panel())
    train = prepared.loc[
        prepared["venue"].eq("POLYMARKET")
        & prepared["nfl_week"].le(8)
    ].reset_index(drop=True)
    altered = train.copy()
    unavailable = ~altered["availability_h"].astype(bool)
    assert unavailable.any()
    altered.loc[unavailable, "mark_l_price"] = (
        pd.to_numeric(
            altered.loc[unavailable, "mark_l_price"],
            errors="raise",
        )
        + 10_000
    )
    altered.loc[unavailable, "mark_l_staleness_seconds"] = 999_999

    base = _fit_two_heads(
        train,
        model_id="regularized_logistic_v1",
        feature_block_id="D2",
        random_state=20260729,
    )
    perturbed = _fit_two_heads(
        altered,
        model_id="regularized_logistic_v1",
        feature_block_id="D2",
        random_state=20260729,
    )
    _, base_direction = _predict_two_heads(
        base,
        train,
        model_id="regularized_logistic_v1",
        feature_block_id="D2",
    )
    _, perturbed_direction = _predict_two_heads(
        perturbed,
        train,
        model_id="regularized_logistic_v1",
        feature_block_id="D2",
    )

    assert (
        base.availability_preprocessor_training_sha256
        != perturbed.availability_preprocessor_training_sha256
    )
    assert (
        base.direction_preprocessor_training_sha256
        == perturbed.direction_preprocessor_training_sha256
    )
    assert (
        base.direction_preprocessor_fitted_sha256
        == perturbed.direction_preprocessor_fitted_sha256
    )
    assert (
        base.direction_model_training_sha256
        == perturbed.direction_model_training_sha256
    )
    assert np.allclose(
        base_direction,
        perturbed_direction,
        rtol=1e-12,
        atol=1e-12,
    )
    assert base.availability_transformer is not base.direction_transformer


def test_missing_prestate_is_retained_and_multiplicity_invariant() -> None:
    original = _panel()
    row_index = int(
        original.index[
            original["nfl_week"].eq(1)
            & original["source_row_id"].ne("row-censored")
        ][1]
    )
    original.loc[row_index, "p_before_support_status"] = "MISSING"
    original.loc[row_index, "p_before_home"] = pd.NA
    original.loc[row_index, "p_before_home_missing"] = True
    original.loc[row_index, "prestate_known_at"] = pd.NaT
    prepared = prepare_regulation_panel(original)
    retained = prepared.loc[
        prepared["source_row_id"].eq(
            original.loc[row_index, "source_row_id"]
        )
    ]
    assert len(retained) == 1
    assert retained["p_before_home_missing"].all()
    assert retained["prestate_known_at"].isna().all()

    duplicate = original.loc[row_index].copy()
    duplicate["source_row_id"] = "missing-prestate-duplicate-anchor-row"
    repeated_source = pd.concat(
        [original, duplicate.to_frame().T], ignore_index=True
    )
    cell = (("fold_01", "regularized_logistic_v1", "D2"),)
    base = run_regulation_walk_forward(original, cells=cell)
    repeated = run_regulation_walk_forward(repeated_source, cells=cell)
    columns = [
        "availability_probability",
        "direction_probability_down",
        "direction_probability_no_move",
        "direction_probability_up",
    ]
    left = base.oof_predictions.sort_values("source_row_id")
    right = repeated.oof_predictions.sort_values("source_row_id")
    assert left["source_row_id"].tolist() == right[
        "source_row_id"
    ].tolist()
    assert np.allclose(
        left[columns].to_numpy(dtype=float),
        right[columns].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_xgboost_direction_encoding_preserves_frozen_class_order() -> None:
    run = run_regulation_walk_forward(
        _panel(),
        cells=(("fold_01", "shallow_xgboost_v1", "D2"),),
    )
    probabilities = run.oof_predictions[
        [
            "direction_probability_down",
            "direction_probability_no_move",
            "direction_probability_up",
        ]
    ].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "present_classes",
    (
        ("DOWN", "UP"),
        ("NO_MOVE", "UP"),
    ),
)
def test_xgboost_two_class_noncontiguous_labels_reexpand_semantically(
    present_classes: tuple[str, str],
) -> None:
    matrix = np.asarray(
        [[float(index), float(index % 3)] for index in range(12)]
    )
    truth = np.asarray(
        [present_classes[index % 2] for index in range(len(matrix))],
        dtype=object,
    )
    estimator = _fit_estimator(
        model_id="shallow_xgboost_v1",
        matrix=matrix,
        truth=truth,
        weights=np.ones(len(matrix), dtype=float),
        direction=True,
        random_state=20260729,
    )
    assert tuple(estimator.classes_) == present_classes
    probability = _aligned_direction_probability(estimator, matrix)
    assert probability.shape == (len(matrix), 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    missing_class = next(
        label
        for label in DIRECTION_CLASSES
        if label not in present_classes
    )
    assert np.all(
        probability[:, DIRECTION_CLASSES.index(missing_class)] < 1e-5
    )


def test_kalshi_transport_refits_exact_frozen_poly_producer(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    overlay = _verified_test_native_rule_overlay(panel)
    panel_before = panel.frame.copy(deep=True)
    transported = run_frozen_kalshi_transport(
        panel,
        selection_lock=lock,
        native_rule_overlay=overlay,
    )
    pd.testing.assert_frame_equal(panel.frame, panel_before)
    assert set(transported.oof_predictions["evaluation_venue"]) == {
        "KALSHI"
    }
    assert set(transported.oof_predictions["training_venue"]) == {
        "POLYMARKET"
    }
    assert set(transported.oof_predictions["transport_mode"]) == {
        "FROZEN_POLYMARKET_WINNER_TRANSPORT",
        "FROZEN_POLYMARKET_B0_TRANSPORT",
    }
    assert set(transported.oof_predictions["model_id"]) == {
        "regularized_logistic_v1",
        "b0_empirical_v1",
    }
    assert (
        transported.transport_risk_set.loc[
            ~transported.transport_risk_set["post_trade_observed"],
            "risk_set_status",
        ]
        .eq("BOUND_NO_NEW_ACTUAL_TRADE_OBSERVED")
        .all()
    )
    winner_pairs = set(
        transported.oof_predictions.loc[
            transported.oof_predictions["model_id"].eq(
                "regularized_logistic_v1"
            ),
            "descriptive_transport_pair_id",
        ]
    )
    b0_pairs = set(
        transported.oof_predictions.loc[
            transported.oof_predictions["model_id"].eq(
                "b0_empirical_v1"
            ),
            "descriptive_transport_pair_id",
        ]
    )
    assert winner_pairs == b0_pairs
    assert (
        transported.fold_metrics["metric_scope"]
        .eq(
            "DESCRIPTIVE_PAIRED_KALSHI_WINNER_MINUS_TRANSPORTED_B0"
        )
        .any()
    )
    assert {
        "BOTH_VENUES_OBSERVED_SETTLEMENT_EQUIVALENCE_UNPROVEN",
        "POLYMARKET_ONLY",
        "KALSHI_ONLY",
    }.issuperset(
        set(
            transported.descriptive_pair_coverage[
                "descriptive_pair_coverage_status"
            ]
        )
    )
    assert transported.run_config["selection_allowed"] is False
    assert transported.run_config["reranking_allowed"] is False
    assert transported.run_config["venue_speed_claim_allowed"] is False
    assert (
        transported.run_config["formal_contract_transport_allowed"]
        is False
    )
    assert (
        transported.run_config[
            "independent_sports_holdout_validation_allowed"
        ]
        is False
    )
    assert transported.run_config[
        "strict_contract_rule_equivalence_status"
    ] == "UNPROVEN"


@pytest.mark.parametrize(
    "provenance_column",
    (
        "availability_model_spec_sha256",
        "availability_model_fitted_sha256",
    ),
)
def test_kalshi_transport_rejects_frozen_producer_hash_mutation(
    tmp_path: Path,
    provenance_column: str,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    overlay = _verified_test_native_rule_overlay(panel)
    assert lock.winner is not None
    mutated_hashes = {
        fold_id: dict(values)
        for fold_id, values in (
            lock.winner.polymarket_fold_training_hashes.items()
        )
    }
    mutated_hashes["fold_01"][provenance_column] = (
        "sha256:" + "b" * 64
    )
    mutated_winner = replace(
        lock.winner,
        polymarket_fold_training_hashes=mutated_hashes,
    )
    mutated_lock = replace(lock, winner=mutated_winner)

    with pytest.raises(
        RegulationModelInputError,
        match="producer hashes differ from verified selection lock",
    ):
        run_frozen_kalshi_transport(
            panel,
            selection_lock=mutated_lock,
            native_rule_overlay=overlay,
        )


def test_kalshi_requires_verified_selection_lock_and_cannot_select(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    overlay = _verified_test_native_rule_overlay(panel)
    request = validate_kalshi_transport(
        panel,
        selection_lock=lock,
        native_rule_overlay=overlay,
    )
    assert request.selection_allowed is False
    assert request.training_venue == "POLYMARKET"
    assert request.evaluation_venue == "KALSHI"
    assert request.model_id == lock.winner.model_id
    assert request.feature_block_id == lock.winner.feature_block_id
    assert request.selection_lock_sha256 == lock.selection_lock_sha256

    with pytest.raises(RegulationModelInputError, match="selection lock"):
        validate_kalshi_transport(
            panel,
            selection_lock=None,
            native_rule_overlay=overlay,
        )


def test_native_rule_overlay_integrity_binds_exact_panel_contract_authority(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    overlay = replace(
        _verified_test_native_rule_overlay(panel),
        panel_contract_authority_sha256=("sha256:" + "b" * 64),
    )

    with pytest.raises(
        RegulationModelInputError,
        match="native-rule overlay does not bind",
    ):
        validate_kalshi_transport(
            panel,
            selection_lock=lock,
            native_rule_overlay=overlay,
        )


def test_native_rule_overlay_rejects_dual_venue_pair_gate_disagreement(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    original = _verified_test_native_rule_overlay(panel)
    bindings = [dict(binding) for binding in original.bindings]
    bindings[0][
        "pair_completed_non_tie_comparability_evidence_sha256"
    ] = _canonical_digest({"tampered": "single-venue-pair-evidence"})
    material = dict(bindings[0])
    material.pop("cross_binding_sha256")
    bindings[0]["cross_binding_sha256"] = _canonical_digest(material)
    overlay = replace(
        original,
        bindings=tuple(bindings),
        cross_binding_set_sha256=_canonical_digest(
            [
                str(binding["cross_binding_sha256"])
                for binding in bindings
            ]
        ),
    )

    with pytest.raises(
        RegulationModelInputError,
        match="dual-venue pair gate fields disagree",
    ):
        validate_kalshi_transport(
            panel,
            selection_lock=lock,
            native_rule_overlay=overlay,
        )


def test_native_rule_overlay_verifies_cross_layer_membership_and_raw_hashes(
    tmp_path: Path,
) -> None:
    panel_batch_path, panel_batch_sha, _ = _one_game_verified_bundle(
        tmp_path, include_kalshi=True
    )
    panel = load_verified_regulation_panel_batch(
        project_root=tmp_path,
        batch_manifest_path=panel_batch_path,
        batch_manifest_file_sha256=panel_batch_sha,
        venue_scope=("POLYMARKET", "KALSHI"),
        expected_game_count=1,
    )
    contracts = (
        panel.frame.loc[
            :,
            ["venue", "actual_home_contract_id"],
        ]
        .drop_duplicates()
        .set_index("venue")["actual_home_contract_id"]
        .astype(str)
        .to_dict()
    )
    raw_root = tmp_path / "raw-root"
    poly_path = raw_root / "raw" / "poly-metadata.json"
    kalshi_path = raw_root / "raw" / "kalshi-metadata.json"
    poly_path.parent.mkdir(parents=True)
    poly_payload = json.dumps(
        {
            "markets": [
                {
                    "id": "poly-native-market",
                    "clobTokenIds": json.dumps(
                        ["poly-away-token", contracts["POLYMARKET"]],
                        separators=(",", ":"),
                    ),
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    kalshi_payload = json.dumps(
        {"markets": [{"ticker": contracts["KALSHI"]}]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    poly_path.write_bytes(poly_payload)
    kalshi_path.write_bytes(kalshi_payload)
    poly_object_sha = _digest(poly_payload)
    kalshi_object_sha = _digest(kalshi_payload)
    poly_manifest_sha = _canonical_digest({"manifest": "poly"})
    kalshi_manifest_sha = _canonical_digest({"manifest": "kalshi"})
    poly_request_sha = _canonical_digest({"request": "poly"})
    kalshi_request_sha = _canonical_digest({"request": "kalshi"})

    poly_native = NativeRuleInput(
        venue="POLYMARKET",
        contract_ids=("poly-native-market",),
        identity_material={
            "game_id": "2025_01_A0_B0",
            "home_team": "B0",
            "away_team": "A0",
            "clob_token_ids": [
                "poly-away-token",
                contracts["POLYMARKET"],
            ],
        },
        rule_texts=(
            "If B0 wins, the market resolves to B0. "
            "If A0 wins, the market resolves to A0. "
            "If postponed, the market will remain open until completed. "
            "If canceled with no make-up game, the market resolves 50-50. "
            "If declared a tie, the market resolves 50-50.",
        ),
        resolution_source="https://www.nfl.com/",
        raw_manifest_sha256s=(poly_manifest_sha,),
        raw_object_sha256s=(poly_object_sha,),
        request_sha256s=(poly_request_sha,),
    )
    kalshi_native = NativeRuleInput(
        venue="KALSHI",
        contract_ids=(
            contracts["KALSHI"],
            "kalshi-away-contract",
        ),
        identity_material={
            "game_id": "2025_01_A0_B0",
            "home_team": "B0",
            "away_team": "A0",
            "home_ticker": contracts["KALSHI"],
        },
        rule_texts=(
            "If B0 wins the game, the market resolves to Yes.",
            "If A0 wins the game, the market resolves to Yes.",
            "If postponed or delayed, the market will remain open until "
            "the rescheduled game finishes. If cancelled and not played, "
            "rescheduled, or declared a tie, all teams resolve 50/50.",
        ),
        resolution_source=None,
        raw_manifest_sha256s=(kalshi_manifest_sha,),
        raw_object_sha256s=(kalshi_object_sha,),
        request_sha256s=(kalshi_request_sha,),
    )
    evidence, pair = build_game_rule_audit(
        game_id="2025_01_A0_B0",
        home_team="B0",
        away_team="A0",
        final_home_score=21,
        final_away_score=17,
        facts_manifest_sha256=SHA,
        facts_object_sha256=SHA,
        polymarket=poly_native,
        kalshi=kalshi_native,
    )
    assert pair["strict_rule_equivalence_status"] == "UNPROVEN"
    assert pair["strict_mismatch_clauses_json"] == "[]"
    raw_refs = [
        {
            "manifest_sha256": poly_manifest_sha,
            "object_sha256": poly_object_sha,
            "request_sha256": poly_request_sha,
            "object_path": "raw/poly-metadata.json",
            "byte_length": len(poly_payload),
        },
        {
            "manifest_sha256": kalshi_manifest_sha,
            "object_sha256": kalshi_object_sha,
            "request_sha256": kalshi_request_sha,
            "object_path": "raw/kalshi-metadata.json",
            "byte_length": len(kalshi_payload),
        },
    ]
    capture_root = tmp_path / "artifacts" / "capture"
    index_path = (
        capture_root / "games" / "2025_01_A0_B0" / "index.json"
    )
    index_path.parent.mkdir(parents=True)
    index_payload = json.dumps(
        {
            "schema": "nfl_moneyline_game_capture_index_v1",
            "game_id": "2025_01_A0_B0",
            "capture_scope": "moneyline-only",
            "reaction_access": "CLOSED",
            "raw_manifests": raw_refs,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    index_path.write_bytes(index_payload)
    index_sha = _digest(index_payload)
    capture_path = (
        capture_root / "batch" / "manifests" / "capture.json"
    )
    capture_path.parent.mkdir(parents=True)
    capture_payload = json.dumps(
        {
            "schema": "nfl_moneyline_expansion_batch_manifest_v1",
            "reaction_access": "CLOSED",
            "game_indexes": [
                {
                    "game_id": "2025_01_A0_B0",
                    "index_path": (
                        "games/2025_01_A0_B0/index.json"
                    ),
                    "index_sha256": index_sha,
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    capture_path.write_bytes(capture_payload)
    capture_sha = _digest(capture_payload)
    read_audit = {
        "metadata_object_reads": 2,
        "trade_object_reads": 0,
        "candle_object_reads": 0,
        "reaction_object_reads": 0,
        "holdout_game_index_reads": 0,
        "holdout_fact_object_reads": 0,
    }
    source_bindings = {
        "governance_object_sha256": SHA,
        "capture_batch_file_sha256": capture_sha,
        "facts_batch_file_sha256": SHA,
    }
    published = publish_rule_audit_frames(
        project_root=tmp_path,
        output_root=Path("artifacts/native-rule-audit-v1"),
        frames=RuleAuditFrames(
            venue_evidence=pd.DataFrame(evidence),
            contract_pairs=pd.DataFrame([pair]),
            read_audit=read_audit,
        ),
        source_bindings=source_bindings,
        expected_game_count=1,
    )
    overlay = regulation_models.verify_native_rule_transport_overlay(
        project_root=tmp_path,
        manifest_path=published.manifest_path,
        manifest_file_sha256=published.manifest_file_sha256,
        panel=panel,
        raw_root=raw_root,
        capture_batch_path=capture_path,
        expected_game_count=1,
    )
    assert overlay.game_count == 1
    assert overlay.venue_evidence_row_count == 2
    assert overlay.completed_non_tie_comparable_count == 1
    assert overlay.strict_rule_equivalent_count == 0
    assert {
        str(binding["pair_strict_rule_equivalence_status"])
        for binding in overlay.bindings
    } == {"UNPROVEN"}
    assert {
        str(binding["native_market_membership_status"])
        for binding in overlay.bindings
    } == {"VERIFIED"}
    assert any(
        binding["panel_contract_identity_sha256"]
        != binding["native_contract_identity_sha256"]
        for binding in overlay.bindings
    )

    tampered_pair = dict(pair)
    tampered_pair["realized_winner_team"] = "A0"
    tampered = publish_rule_audit_frames(
        project_root=tmp_path,
        output_root=Path("artifacts/native-rule-audit-tampered-v1"),
        frames=RuleAuditFrames(
            venue_evidence=pd.DataFrame(evidence),
            contract_pairs=pd.DataFrame([tampered_pair]),
            read_audit=read_audit,
        ),
        source_bindings=source_bindings,
        expected_game_count=1,
    )
    with pytest.raises(
        RegulationModelInputError,
        match="identity or realized winner mismatch",
    ):
        regulation_models.verify_native_rule_transport_overlay(
            project_root=tmp_path,
            manifest_path=tampered.manifest_path,
            manifest_file_sha256=tampered.manifest_file_sha256,
            panel=panel,
            raw_root=raw_root,
            capture_batch_path=capture_path,
            expected_game_count=1,
        )

    poly_path.write_bytes(poly_payload + b"tamper")
    with pytest.raises(RegulationModelInputError, match="hash mismatch"):
        regulation_models.verify_native_rule_transport_overlay(
            project_root=tmp_path,
            manifest_path=published.manifest_path,
            manifest_file_sha256=published.manifest_file_sha256,
            panel=panel,
            raw_root=raw_root,
            capture_batch_path=capture_path,
            expected_game_count=1,
        )


def test_native_rule_gate_one_allows_descriptive_transport_only(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    overlay = _verified_test_native_rule_overlay(panel)

    request = validate_kalshi_transport(
        panel,
        selection_lock=lock,
        native_rule_overlay=overlay,
    )
    assert request.transport_scope == (
        "DESCRIPTIVE_COMPLETED_NON_TIE_PRODUCER_TRANSPORT"
    )
    assert request.realized_outcome_comparability_status == (
        "COMPLETED_NON_TIE_REALIZATION_COMPARABLE"
    )
    assert request.formal_contract_transport_allowed is False
    assert (
        request.independent_sports_holdout_validation_allowed is False
    )
    assert request.venue_speed_claim_allowed is False


def test_native_rule_gate_two_blocks_formal_transport(
    tmp_path: Path,
) -> None:
    panel = _verified_two_venue_panel_153(tmp_path)
    lock = _verified_test_selection_lock(panel)
    overlay = _verified_test_native_rule_overlay(panel)
    request = validate_kalshi_transport(
        panel,
        selection_lock=lock,
        native_rule_overlay=overlay,
    )
    assert overlay.strict_rule_equivalent_count == 0
    assert request.strict_contract_rule_equivalence_status == "UNPROVEN"
    assert request.formal_contract_transport_allowed is False

    transported = run_frozen_kalshi_transport(
        panel,
        selection_lock=lock,
        native_rule_overlay=overlay,
    )
    assert (
        transported.run_config["formal_contract_transport_allowed"]
        is False
    )
    assert (
        transported.run_config[
            "independent_sports_holdout_validation_allowed"
        ]
        is False
    )
    assert transported.run_config["venue_speed_claim_allowed"] is False
    assert not transported.fold_metrics[
        "formal_contract_transport_allowed"
    ].any()
    assert (
        transported.fold_metrics["transport_inference_status"]
        .astype(str)
        .str.startswith("DESCRIPTIVE_PRODUCER_TRANSPORT_")
        .all()
    )


def test_verified_panel_loader_checks_every_manifest_and_object_hash(
    tmp_path: Path,
) -> None:
    batch_path, batch_sha, object_path = _one_game_verified_bundle(tmp_path)

    manifest = verify_regulation_panel_batch_manifest(
        project_root=tmp_path,
        batch_manifest_path=batch_path,
        batch_manifest_file_sha256=batch_sha,
        expected_game_count=1,
    )
    loaded = load_verified_regulation_panel_batch(
        project_root=tmp_path,
        batch_manifest_path=batch_path,
        batch_manifest_file_sha256=batch_sha,
        venue_scope=("POLYMARKET",),
        expected_game_count=1,
    )

    assert manifest.game_count == 1
    assert loaded.formal_input_verified is True
    assert loaded.loaded_venue_scope == ("POLYMARKET",)
    assert loaded.loaded_row_count == len(loaded.frame)
    assert set(loaded.frame["venue"]) == {"POLYMARKET"}
    assert loaded.batch_manifest_file_sha256 == batch_sha
    source = _panel()
    first_game = str(source.iloc[0]["game_id"])
    legacy = prepare_regulation_panel(
        pq.read_table(object_path).to_pandas()
    )
    legacy = legacy.sort_values(
        list(regulation_models._VERIFIED_PANEL_SORT_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        loaded.frame,
        legacy,
        check_dtype=True,
        check_like=False,
    )
    assert loaded.source_row_count == int(
        source["game_id"].eq(first_game).sum()
    )
    assert len(loaded.frame) == loaded.loss_eligible_count
    assert not loaded.frame["censored"].any()

    original = object_path.read_bytes()
    object_path.write_bytes(original + b"tamper")
    with pytest.raises(RegulationModelInputError, match="object hash"):
        load_verified_regulation_panel_batch(
            project_root=tmp_path,
            batch_manifest_path=batch_path,
            batch_manifest_file_sha256=batch_sha,
            venue_scope=("POLYMARKET",),
            expected_game_count=1,
        )


def test_verified_panel_loader_has_one_bounded_arrow_to_pandas_boundary() -> None:
    source = inspect.getsource(load_verified_regulation_panel_batch)
    assert "prepared_games" not in source
    assert "pd.concat" not in source
    assert source.count(".to_pandas(") == 1
    assert "columns=list(REQUIRED_PANEL_COLUMNS)" in source
    assert "filters=" in source


def test_verified_panel_loader_requires_explicit_supported_venue_scope(
    tmp_path: Path,
) -> None:
    batch_path, batch_sha, _ = _one_game_verified_bundle(
        tmp_path, include_kalshi=True
    )
    with pytest.raises(TypeError, match="venue_scope"):
        load_verified_regulation_panel_batch(
            project_root=tmp_path,
            batch_manifest_path=batch_path,
            batch_manifest_file_sha256=batch_sha,
            expected_game_count=1,
        )
    with pytest.raises(RegulationModelInputError, match="venue_scope"):
        load_verified_regulation_panel_batch(
            project_root=tmp_path,
            batch_manifest_path=batch_path,
            batch_manifest_file_sha256=batch_sha,
            venue_scope=("KALSHI",),
            expected_game_count=1,
        )

    selection = load_verified_regulation_panel_batch(
        project_root=tmp_path,
        batch_manifest_path=batch_path,
        batch_manifest_file_sha256=batch_sha,
        venue_scope=("POLYMARKET",),
        expected_game_count=1,
    )
    transport = load_verified_regulation_panel_batch(
        project_root=tmp_path,
        batch_manifest_path=batch_path,
        batch_manifest_file_sha256=batch_sha,
        venue_scope=("POLYMARKET", "KALSHI"),
        expected_game_count=1,
    )
    assert selection.loaded_venue_scope == ("POLYMARKET",)
    assert set(selection.frame["venue"]) == {"POLYMARKET"}
    assert transport.loaded_venue_scope == (
        "POLYMARKET",
        "KALSHI",
    )
    assert set(transport.frame["venue"]) == {
        "POLYMARKET",
        "KALSHI",
    }
    assert transport.loaded_row_count == 2 * selection.loaded_row_count


def test_selection_and_transport_reject_wrong_materialized_scope(
    tmp_path: Path,
) -> None:
    dual = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=True
    )
    with pytest.raises(
        RegulationModelInputError,
        match="POLYMARKET-only",
    ):
        run_regulation_walk_forward(
            dual,
            cells=(("fold_01", "b0_empirical_v1", "D0"),),
        )

    poly = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=False
    )
    lock = _verified_test_selection_lock(poly)
    with pytest.raises(
        RegulationModelInputError,
        match="dual-venue scope",
    ):
        validate_kalshi_transport(
            poly,
            selection_lock=lock,
            native_rule_overlay=None,
        )


def test_polymarket_scoped_panel_is_b0_output_hash_equivalent(
    tmp_path: Path,
) -> None:
    legacy_all = _verified_two_venue_panel_153(
        tmp_path, include_kalshi=True
    )
    poly_frame = legacy_all.frame.loc[
        legacy_all.frame["venue"].eq("POLYMARKET")
    ].reset_index(drop=True)
    scoped = replace(
        legacy_all,
        frame=poly_frame,
        loaded_venue_scope=("POLYMARKET",),
        loaded_row_count=len(poly_frame),
        loss_eligible_count=len(poly_frame),
    )
    cell = (("fold_01", "b0_empirical_v1", "D0"),)
    legacy_run = run_regulation_walk_forward(
        legacy_all.frame, cells=cell
    )
    scoped_run = run_regulation_walk_forward(
        scoped.frame, cells=cell
    )

    pd.testing.assert_frame_equal(
        scoped_run.oof_predictions,
        legacy_run.oof_predictions,
        check_dtype=True,
    )
    pd.testing.assert_frame_equal(
        scoped_run.fold_metrics,
        legacy_run.fold_metrics,
        check_dtype=True,
    )
    pd.testing.assert_frame_equal(
        scoped_run.support_audit,
        legacy_run.support_audit,
        check_dtype=True,
    )
    assert scoped_run.run_config_sha256 == legacy_run.run_config_sha256
    assert _parquet_bytes(scoped_run.oof_predictions) == _parquet_bytes(
        legacy_run.oof_predictions
    )


def test_runner_prints_contract_without_loading_or_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_nfl_x15_regulation_runner", _RUNNER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)

    assert module.main(["--print-contract"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "RegulationDecisionPanelV4"
    assert payload["execution_cell_count"] == 45
    assert payload["selection_venue"] == "POLYMARKET"
    assert payload["kalshi_selection_allowed"] is False
    assert payload["holdout_reaction_accessed"] is False


def test_runner_discovers_completed_shards_only_once() -> None:
    spec = importlib.util.spec_from_file_location(
        "_nfl_x15_regulation_runner_single_discovery", _RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = inspect.getsource(module.main)
    assert source.count("discover_verified_regulation_oof_shards(") == 1
    assert "verify_regulation_oof_shard(" in source


def test_runner_applies_venue_scope_only_when_materializing_panel() -> None:
    spec = importlib.util.spec_from_file_location(
        "_nfl_x15_regulation_runner_scope_boundary", _RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = inspect.getsource(module.main)
    manifest_call = source.split(
        "panel_manifest = verify_regulation_panel_batch_manifest(", 1
    )[1].split("requested =", 1)[0]
    materialize_call = source.split(
        "panel = load_verified_regulation_panel_batch(", 1
    )[1].split("existing =", 1)[0]

    assert "venue_scope" not in manifest_call
    assert "venue_scope=(POLYMARKET,)" in materialize_call


def test_runner_memory_gate_warns_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "_nfl_x15_regulation_runner_memory", _RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_peak_rss_mib", lambda: 6_500.0)
    monkeypatch.setattr(
        module, "_system_memory_free_percent", lambda: 50
    )
    warning = module._memory_gate_snapshot(
        stage="test-warning", started_at=0.0, now=1.0
    )
    assert warning["status"] == "RSS_WARNING"
    assert warning["peak_rss_mib"] == 6_500.0
    assert warning["system_memory_free_percent"] == 50

    monkeypatch.setattr(module, "_peak_rss_mib", lambda: 8_192.0)
    with pytest.raises(MemoryError, match="8 GiB"):
        module._memory_gate_snapshot(
            stage="test-rss-stop", started_at=0.0, now=2.0
        )

    monkeypatch.setattr(module, "_peak_rss_mib", lambda: 100.0)
    monkeypatch.setattr(
        module, "_system_memory_free_percent", lambda: 10
    )
    with pytest.raises(MemoryError, match="10%"):
        module._memory_gate_snapshot(
            stage="test-free-stop", started_at=0.0, now=3.0
        )
