from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_decision_context_v2 import (
    DecisionContextV2Error,
    build_decision_context_v2_frame,
    compute_prestate_context_sha256,
    publish_decision_context_v2_tables,
)


PIA_OBJECT_SHA = "sha256:" + "0" * 64
FACT_OBJECT_SHA = "sha256:" + "1" * 64
STATE_OBJECT_SHA = "sha256:" + "2" * 64
FEATURE_OBJECT_SHA = "sha256:" + "3" * 64
PIA_MANIFEST_SHA = "sha256:" + "4" * 64
FACT_MANIFEST_SHA = "sha256:" + "5" * 64
STAGE_A_MANIFEST_SHA = "sha256:" + "6" * 64
MODEL_SHA = "sha256:" + "7" * 64
MODEL_MANIFEST_SHA = "sha256:" + "8" * 64
PBP_SHA = "sha256:" + "9" * 64
STATE_INPUT_SHA = "sha256:" + "a" * 64


def _anchors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "schema_version": "ProvisionalInformationAnchorV2",
                "game_id": "2025_01_A_B",
                "information_anchor_id": "anchor-10",
                "episode_id": "parent-10",
                "event_id": "event-10",
                "raw_play_id": "10",
                "order_sequence": 10,
                "source_interval_start": "2025-09-01T01:02:02Z",
                "source_interval_end": "2025-09-01T01:02:03Z",
                "information_anchor": "2025-09-01T01:02:03Z",
                "provisional_known_at": "2025-09-01T01:02:03Z",
                "provisional_primary_event_id": "event-10",
                "provisional_feature_support_status": "SUPPORTED",
                "primary_selection_eligible": True,
                "censor_boundary_eligible": True,
                "pbp_source_sha256": PBP_SHA,
            },
            {
                "schema_version": "ProvisionalInformationAnchorV2",
                "game_id": "2025_01_A_B",
                "information_anchor_id": "anchor-20",
                "episode_id": "parent-10",
                "event_id": "event-20",
                "raw_play_id": "20",
                "order_sequence": 20,
                "source_interval_start": "2025-09-01T01:02:10Z",
                "source_interval_end": "2025-09-01T01:02:11Z",
                "information_anchor": "2025-09-01T01:02:11Z",
                "provisional_known_at": "2025-09-01T01:02:11Z",
                "provisional_primary_event_id": "event-20",
                "provisional_feature_support_status": (
                    "PRIMARY_FEATURE_SUPPORT_UNPROVEN"
                ),
                "primary_selection_eligible": False,
                "censor_boundary_eligible": True,
                "pbp_source_sha256": PBP_SHA,
            },
        ]
    )


def _facts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "event_id": "event-10",
                "raw_play_id": "10",
                "source_interval_start": "2025-09-01T01:02:02Z",
                "source_interval_end": "2025-09-01T01:02:03Z",
                "known_at": "2025-09-01T01:02:03Z",
                "order_sequence": 10,
                "quarter": 4,
                "game_clock": "02:00",
                "game_seconds_remaining": 120,
                "home_team": "B",
                "away_team": "A",
                "possession_before": "B",
                "possession_is_home": True,
                "pre_home_score": 20,
                "pre_away_score": 17,
                "score_margin_home": 3,
                "down": 3,
                "distance": 7,
                "yardline_100": 35,
                "goal_to_go": False,
                "pre_red_zone": False,
                "home_timeouts_remaining": 2,
                "away_timeouts_remaining": 1,
                "pbp_source_sha256": PBP_SHA,
            },
            {
                "game_id": "2025_01_A_B",
                "event_id": "event-20",
                "raw_play_id": "20",
                "source_interval_start": "2025-09-01T01:02:10Z",
                "source_interval_end": "2025-09-01T01:02:11Z",
                "known_at": pd.NA,
                "order_sequence": 20,
                "quarter": 5,
                "game_clock": "10:00",
                "game_seconds_remaining": 600,
                "home_team": "B",
                "away_team": "A",
                "possession_before": "A",
                "possession_is_home": False,
                "pre_home_score": 20,
                "pre_away_score": 20,
                "score_margin_home": 0,
                "down": 1,
                "distance": 10,
                "yardline_100": 75,
                "goal_to_go": False,
                "pre_red_zone": False,
                "home_timeouts_remaining": 2,
                "away_timeouts_remaining": 1,
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
                "state_known_at": "2025-09-01T01:02:02.900000Z",
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
                "state_known_at": "2025-09-01T01:02:10.900000Z",
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


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "state_event_id": "event-10",
                "state_raw_play_id": "10",
                "state_known_at": "2025-09-01T01:02:02.900000Z",
                "feature_name": f"feature_{index:02d}",
                "feature_value": 1.0,
                "feature_known_at": "2025-09-01T01:02:02.900000Z",
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
    return build_decision_context_v2_frame(
        information_anchors=_anchors(),
        facts=_facts(),
        state_predictions=_states(),
        feature_provenance=_features(),
        information_anchor_object_sha256=PIA_OBJECT_SHA,
        facts_object_sha256=FACT_OBJECT_SHA,
        state_predictions_object_sha256=STATE_OBJECT_SHA,
        feature_provenance_object_sha256=FEATURE_OBJECT_SHA,
        information_anchor_manifest_sha256=PIA_MANIFEST_SHA,
        facts_manifest_sha256=FACT_MANIFEST_SHA,
        stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
    )


def test_context_v2_is_one_row_per_pia_and_preserves_censor_only_anchor() -> None:
    context, audit = _build()

    assert list(context["information_anchor_id"]) == ["anchor-10", "anchor-20"]
    assert context["information_anchor_id"].is_unique
    assert audit["information_anchor_expected_rows"] == 2
    assert audit["information_anchor_matched_rows"] == 2
    assert audit["censor_boundary_rows"] == 2
    assert audit["primary_selection_rows"] == 1

    primary = context.loc[
        context["information_anchor_id"].eq("anchor-10")
    ].iloc[0]
    assert primary["quarter"] == 4
    assert primary["game_seconds_remaining"] == pytest.approx(120)
    assert primary["home_team"] == "B"
    assert primary["away_team"] == "A"
    assert primary["score_margin_home"] == pytest.approx(3)
    assert primary["possession_is_home"]
    assert primary["down"] == pytest.approx(3)
    assert primary["distance"] == pytest.approx(7)
    assert primary["yardline_100"] == pytest.approx(35)
    assert not primary["goal_to_go"]
    assert not primary["pre_red_zone"]
    assert primary["p_before_home"] == pytest.approx(0.61)
    assert not primary["p_before_home_missing"]

    censor_only = context.loc[
        context["information_anchor_id"].eq("anchor-20")
    ].iloc[0]
    assert censor_only["censor_boundary_eligible"]
    assert not censor_only["primary_selection_eligible"]
    assert censor_only["is_overtime"]
    assert pd.isna(censor_only["p_before_home"])
    assert censor_only["p_before_home_missing"]
    assert (
        censor_only["sports_prestate_known_at_basis"]
        == "SOURCE_INTERVAL_END_BOUND"
    )
    assert (
        censor_only["sports_prestate_known_at"]
        == "2025-09-01 01:02:11+00:00"
    )


def test_all_output_is_prestate_and_known_by_information_anchor() -> None:
    context, audit = _build()

    anchors = pd.to_datetime(context["information_anchor"], utc=True)
    sports_known = pd.to_datetime(
        context["sports_prestate_known_at"], utc=True
    )
    model_known = pd.to_datetime(
        context["pre_state_known_at"], utc=True
    )
    assert (sports_known <= anchors).all()
    assert (model_known <= anchors).all()
    assert audit["known_after_anchor_rows"] == 0
    assert not any(
        column.startswith(("post_", "final_"))
        or column
        in {
            "p_after_home",
            "reference_delta_home",
            "bridge_delta_home",
            "reference_gap",
        }
        for column in context.columns
    )
    assert context["prestate_context_sha256"].str.fullmatch(
        r"sha256:[0-9a-f]{64}"
    ).all()


def test_prestate_context_hash_changes_when_any_formal_feature_changes() -> None:
    context, _ = _build()
    row = context.iloc[0].to_dict()

    original = compute_prestate_context_sha256(row)
    assert original == row["prestate_context_sha256"]
    for column, changed_value in (
        ("game_seconds_remaining", 119.0),
        ("score_margin_home", 4.0),
        ("p_before_home", 0.62),
        ("pre_state_known_at", "2025-09-01T01:02:02.800000Z"),
        ("sports_prestate_known_at", "2025-09-01T01:02:02.800000Z"),
        ("source_binding_sha256", "sha256:" + "f" * 64),
    ):
        mutated = dict(row)
        mutated[column] = changed_value
        assert compute_prestate_context_sha256(mutated) != original


def test_future_known_fact_or_state_fails_closed() -> None:
    facts = _facts()
    facts.loc[0, "known_at"] = "2025-09-01T01:02:04Z"
    with pytest.raises(DecisionContextV2Error, match="known after"):
        build_decision_context_v2_frame(
            information_anchors=_anchors(),
            facts=facts,
            state_predictions=_states(),
            feature_provenance=_features(),
            information_anchor_object_sha256=PIA_OBJECT_SHA,
            facts_object_sha256=FACT_OBJECT_SHA,
            state_predictions_object_sha256=STATE_OBJECT_SHA,
            feature_provenance_object_sha256=FEATURE_OBJECT_SHA,
            information_anchor_manifest_sha256=PIA_MANIFEST_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )

    states = _states()
    states.loc[0, "state_known_at"] = "2025-09-01T01:02:04Z"
    with pytest.raises(DecisionContextV2Error, match="known after"):
        build_decision_context_v2_frame(
            information_anchors=_anchors(),
            facts=_facts(),
            state_predictions=states,
            feature_provenance=_features(),
            information_anchor_object_sha256=PIA_OBJECT_SHA,
            facts_object_sha256=FACT_OBJECT_SHA,
            state_predictions_object_sha256=STATE_OBJECT_SHA,
            feature_provenance_object_sha256=FEATURE_OBJECT_SHA,
            information_anchor_manifest_sha256=PIA_MANIFEST_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )


def test_supported_state_requires_eleven_pit_rows_and_matching_source() -> None:
    with pytest.raises(DecisionContextV2Error, match="exactly 11"):
        build_decision_context_v2_frame(
            information_anchors=_anchors(),
            facts=_facts(),
            state_predictions=_states(),
            feature_provenance=_features().iloc[:-1].copy(),
            information_anchor_object_sha256=PIA_OBJECT_SHA,
            facts_object_sha256=FACT_OBJECT_SHA,
            state_predictions_object_sha256=STATE_OBJECT_SHA,
            feature_provenance_object_sha256=FEATURE_OBJECT_SHA,
            information_anchor_manifest_sha256=PIA_MANIFEST_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )

    anchors = _anchors()
    anchors.loc[0, "pbp_source_sha256"] = "sha256:" + "b" * 64
    with pytest.raises(DecisionContextV2Error, match="PBP source"):
        build_decision_context_v2_frame(
            information_anchors=anchors,
            facts=_facts(),
            state_predictions=_states(),
            feature_provenance=_features(),
            information_anchor_object_sha256=PIA_OBJECT_SHA,
            facts_object_sha256=FACT_OBJECT_SHA,
            state_predictions_object_sha256=STATE_OBJECT_SHA,
            feature_provenance_object_sha256=FEATURE_OBJECT_SHA,
            information_anchor_manifest_sha256=PIA_MANIFEST_SHA,
            facts_manifest_sha256=FACT_MANIFEST_SHA,
            stage_a_manifest_sha256=STAGE_A_MANIFEST_SHA,
        )


def test_publication_is_content_addressed_and_records_all_three_sources(
    tmp_path: Path,
) -> None:
    context, audit = _build()
    publication = publish_decision_context_v2_tables(
        output_root=tmp_path,
        context=context,
        audit_counts=audit,
        information_anchor_manifest_file_sha256=PIA_MANIFEST_SHA,
        information_anchor_bundle_sha256=PIA_OBJECT_SHA,
        facts_batch_file_sha256=FACT_MANIFEST_SHA,
        facts_batch_sha256=FACT_OBJECT_SHA,
        stage_a_batch_file_sha256=STAGE_A_MANIFEST_SHA,
        stage_a_batch_sha256=STATE_OBJECT_SHA,
        expected_game_count=1,
        expected_anchor_count=2,
    )

    manifest = json.loads(publication.manifest_path.read_text())
    assert manifest["schema"] == "EventPrestateContextManifestV2"
    assert manifest["context"]["schema"] == "EventPrestateContextV2"
    assert manifest["context"]["grain"] == [
        "game_id",
        "information_anchor_id",
    ]
    assert manifest["market_data_read"] is False
    assert manifest["holdout_reaction_accessed"] is False
    assert manifest["sources"]["provisional_information_anchor_v2"] == {
        "manifest_file_sha256": PIA_MANIFEST_SHA,
        "bundle_sha256": PIA_OBJECT_SHA,
    }
    assert manifest["audit_counts"]["information_anchor_matched_rows"] == 2
    assert publication.context_path.is_file()
