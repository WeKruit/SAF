from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_decision_context import (
    DecisionContextError,
    build_decision_context_frame,
    publish_decision_context_tables,
)


FACT_SHA = "sha256:" + "1" * 64
STATE_SHA = "sha256:" + "2" * 64
FEATURE_SHA = "sha256:" + "3" * 64
REFERENCE_SHA = "sha256:" + "4" * 64
FACT_MANIFEST_SHA = "sha256:" + "5" * 64
STAGE_A_MANIFEST_SHA = "sha256:" + "6" * 64
MODEL_SHA = "sha256:" + "7" * 64
MODEL_MANIFEST_SHA = "sha256:" + "8" * 64
PBP_SHA = "sha256:" + "9" * 64
STATE_INPUT_SHA = "sha256:" + "a" * 64


def _facts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "event_id": "event-10",
                "raw_play_id": "10",
                "atomic_information_episode_id": "episode-10",
                "quarter": 4,
                "source_interval_end": "2025-09-01T01:02:03.004000Z",
                "stage_b_information_event_eligible": True,
                "pbp_source_sha256": PBP_SHA,
            },
            {
                "game_id": "2025_01_A_B",
                "event_id": "event-20",
                "raw_play_id": "20",
                "atomic_information_episode_id": "episode-20",
                "quarter": 5,
                "source_interval_end": "2025-09-01T01:03:04.005000Z",
                "stage_b_information_event_eligible": True,
                "pbp_source_sha256": PBP_SHA,
            },
        ]
    )


def _states() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "state_event_id": "event-10",
                "state_raw_play_id": "10",
                "state_atomic_information_episode_id": "episode-10",
                "state_known_at": "2025-09-01T01:02:02.999000Z",
                "p_home": 0.61,
                "reference_status": "SUPPORTED",
                "state_input_sha256": STATE_INPUT_SHA,
                "model_id": "MODEL-1",
                "model_version": "v1",
                "model_sha256": MODEL_SHA,
                "model_manifest_sha256": MODEL_MANIFEST_SHA,
                "pbp_source_sha256": PBP_SHA,
            },
            {
                "game_id": "2025_01_A_B",
                "state_event_id": "event-20",
                "state_raw_play_id": "20",
                "state_atomic_information_episode_id": "episode-20",
                "state_known_at": "2025-09-01T01:03:03.999000Z",
                "p_home": pd.NA,
                "reference_status": "MODEL_SUPPORT_UNPROVEN",
                "state_input_sha256": pd.NA,
                "model_id": "MODEL-1",
                "model_version": "v1",
                "model_sha256": MODEL_SHA,
                "model_manifest_sha256": MODEL_MANIFEST_SHA,
                "pbp_source_sha256": PBP_SHA,
            },
        ]
    )


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "event_id": "event-10",
                "raw_play_id": "10",
                "atomic_information_episode_id": "episode-10",
                "pre_state_event_id": "event-10",
                "pre_state_raw_play_id": "10",
                "pbp_source_sha256": PBP_SHA,
                # These values deliberately contradict the state table.  The
                # decision context must never use them as its authority.
                "pre_state_known_at": "2099-01-01T00:00:00Z",
                "p_before_home": 0.01,
                "reference_status": "POST_STATE_FAILURE",
                "p_after_home": 0.99,
                "reference_delta_home": 0.98,
            },
            {
                "game_id": "2025_01_A_B",
                "event_id": "event-20",
                "raw_play_id": "20",
                "atomic_information_episode_id": "episode-20",
                "pre_state_event_id": "event-20",
                "pre_state_raw_play_id": "20",
                "pbp_source_sha256": PBP_SHA,
                "pre_state_known_at": "2099-01-01T00:00:00Z",
                "p_before_home": 0.77,
                "reference_status": "SUPPORTED",
                "p_after_home": 0.88,
                "reference_delta_home": 0.11,
            },
        ]
    )


def _feature_provenance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "state_event_id": "event-10",
                "state_raw_play_id": "10",
                "state_known_at": "2025-09-01T01:02:02.999000Z",
                "feature_name": f"feature_{index:02d}",
                "feature_value": 1.0,
                "feature_known_at": "2025-09-01T01:02:02.999000Z",
                "source_event_id": "event-10",
                "source_raw_play_id": "10",
                "source_row_id": f"event-10:feature_{index:02d}",
                "source_hash": PBP_SHA,
                "PIT_status": "PIT_VERIFIED",
            }
            for index in range(11)
        ]
    )


def _build() -> tuple[pd.DataFrame, dict[str, int]]:
    return build_decision_context_frame(
        facts=_facts(),
        state_predictions=_states(),
        feature_provenance=_feature_provenance(),
        reference_observations=_references(),
        facts_object_sha256=FACT_SHA,
        state_predictions_object_sha256=STATE_SHA,
        feature_provenance_object_sha256=FEATURE_SHA,
        reference_observations_object_sha256=REFERENCE_SHA,
        facts_manifest_sha256=FACT_MANIFEST_SHA,
        stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
    )


def test_prestate_probability_status_and_time_come_only_from_state_prediction() -> None:
    context, audit = _build()

    regulation = context.loc[
        context["atomic_information_episode_id"].eq("episode-10")
    ].iloc[0]
    assert regulation["p_before_home"] == pytest.approx(0.61)
    assert regulation["prestate_support_status"] == "SUPPORTED"
    assert (
        regulation["pre_state_known_at"]
        == "2025-09-01T01:02:02.999000Z"
    )
    assert regulation["state_input_sha256"] == STATE_INPUT_SHA
    assert regulation["is_regulation"]
    assert audit["supported_prestate_rows"] == 1

    overtime = context.loc[
        context["atomic_information_episode_id"].eq("episode-20")
    ].iloc[0]
    assert pd.isna(overtime["p_before_home"])
    assert overtime["prestate_support_status"] == "MODEL_SUPPORT_UNPROVEN"
    assert not overtime["is_regulation"]
    assert audit["overtime_rows"] == 1
    assert audit["supported_overtime_rows"] == 0

    forbidden = {
        "p_after_home",
        "reference_delta_home",
        "reference_status",
        "reference_gap",
    }
    assert forbidden.isdisjoint(context.columns)


def test_duplicate_grain_or_missing_state_join_fails_closed() -> None:
    duplicated = pd.concat([_facts(), _facts().iloc[[0]]], ignore_index=True)
    with pytest.raises(DecisionContextError, match="duplicate"):
        build_decision_context_frame(
            facts=duplicated,
            state_predictions=_states(),
            feature_provenance=_feature_provenance(),
            reference_observations=_references(),
            facts_object_sha256=FACT_SHA,
            state_predictions_object_sha256=STATE_SHA,
            feature_provenance_object_sha256=FEATURE_SHA,
            reference_observations_object_sha256=REFERENCE_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )

    missing = _states().iloc[[1]].copy()
    with pytest.raises(DecisionContextError, match="pre-state join"):
        build_decision_context_frame(
            facts=_facts(),
            state_predictions=missing,
            feature_provenance=_feature_provenance(),
            reference_observations=_references(),
            facts_object_sha256=FACT_SHA,
            state_predictions_object_sha256=STATE_SHA,
            feature_provenance_object_sha256=FEATURE_SHA,
            reference_observations_object_sha256=REFERENCE_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )


def test_supported_state_requires_all_eleven_pit_feature_rows() -> None:
    with pytest.raises(DecisionContextError, match="exactly 11"):
        build_decision_context_frame(
            facts=_facts(),
            state_predictions=_states(),
            feature_provenance=_feature_provenance().iloc[:-1].copy(),
            reference_observations=_references(),
            facts_object_sha256=FACT_SHA,
            state_predictions_object_sha256=STATE_SHA,
            feature_provenance_object_sha256=FEATURE_SHA,
            reference_observations_object_sha256=REFERENCE_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )


def test_publication_is_content_addressed_and_carries_audit(tmp_path: Path) -> None:
    context, audit = _build()
    publication = publish_decision_context_tables(
        output_root=tmp_path,
        context=context,
        audit_counts=audit,
        facts_batch_file_sha256=FACT_MANIFEST_SHA,
        facts_batch_sha256=FACT_SHA,
        stage_a_batch_file_sha256=STAGE_A_MANIFEST_SHA,
        stage_a_batch_sha256=STATE_SHA,
        expected_game_count=1,
    )

    assert publication.context_path.is_file()
    assert publication.manifest_path.is_file()
    manifest = json.loads(publication.manifest_path.read_text())
    assert manifest["schema"] == "EventPrestateContextManifestV1"
    assert manifest["market_data_read"] is False
    assert manifest["holdout_reaction_accessed"] is False
    assert manifest["audit_counts"]["context_rows"] == 2
    assert manifest["context"]["object_sha256"] == publication.context_sha256
