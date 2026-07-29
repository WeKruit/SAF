from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from prediction_market.models.nfl_fastrmodels import (
    FEATURE_NAMES,
    NoSpreadModelInput,
)
from prediction_market.sports.nfl_reference_value import (
    ReferenceFeatureProvenanceV1,
    ReferenceModelBindingV1,
    ReferenceValueBuildError,
    ReferenceValueObservationV1,
    build_game_reference_tables,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _model() -> ReferenceModelBindingV1:
    return ReferenceModelBindingV1(
        model_id="MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
        model_version="9f2495fdb4943087ca663d96706eb5df7973aff4",
        model_sha256=_sha("a"),
        manifest_sha256=_sha("b"),
        feature_spec_commit="75c7b68bc49535370236c38c9826265da075bd71",
    )


def _event(
    *,
    order_sequence: int,
    raw_play_id: str,
    known_at: str | None,
    quarter: int = 1,
    game_seconds_remaining: int = 3500,
    possession_before: str | None = "AWY",
    actor_team: str | None = "AWY",
    primary_action: str = "PASS",
    down: int | None = 1,
    distance: int = 10,
    yardline_100: float | None = 70.0,
    pre_home_score: int = 0,
    pre_away_score: int = 0,
    information_status: str = "FINAL",
    stage_b_eligible: bool = True,
    final_eligible: bool = True,
    row_disposition: str = "LIVE_PLAY",
    home_wp_before_diagnostic: float = 0.99,
    home_wp_after_diagnostic: float = 0.01,
) -> dict[str, object]:
    event_id = f"2025_01_AWY_HME:event:{raw_play_id}"
    return {
        "schema_version": "EpisodeFactV3",
        "game_id": "2025_01_AWY_HME",
        "event_id": event_id,
        "raw_play_id": raw_play_id,
        "play_id": raw_play_id,
        "atomic_information_episode_id": (
            f"2025_01_AWY_HME:episode:{raw_play_id}"
        ),
        "information_status": information_status,
        "stage_b_information_event_eligible": stage_b_eligible,
        "final_sports_outcome_eligible": final_eligible,
        "source_interval_start": known_at,
        "source_interval_end": known_at,
        "known_at": known_at,
        "order_sequence": order_sequence,
        "quarter": quarter,
        "game_seconds_remaining": game_seconds_remaining,
        "home_team": "HME",
        "away_team": "AWY",
        "actor_team": actor_team,
        "possession_before": possession_before,
        "pre_home_score": pre_home_score,
        "pre_away_score": pre_away_score,
        "down": down,
        "distance": distance,
        "yardline_100": yardline_100,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "primary_action": primary_action,
        "row_disposition": row_disposition,
        "review_result": None,
        "pbp_source_sha256": _sha("c"),
        "source_hashes": json.dumps([_sha("c")]),
        # These are deliberately nonsense. Stage A must never consume them.
        "home_wp_before_diagnostic": home_wp_before_diagnostic,
        "home_wp_after_diagnostic": home_wp_after_diagnostic,
        "home_wp_delta_diagnostic": (
            home_wp_after_diagnostic - home_wp_before_diagnostic
        ),
    }


class _RecordingPredictor:
    def __init__(self) -> None:
        self.inputs: list[NoSpreadModelInput] = []

    def predict_home(self, model_input: NoSpreadModelInput) -> float:
        self.inputs.append(model_input)
        return {
            3500.0: 0.30,
            3450.0: 0.60,
            3400.0: 0.55,
            3350.0: 0.52,
        }[model_input.feature_values[3]]


def _base_events() -> pd.DataFrame:
    # Deliberately physical-order scrambled. The first chronological kickoff
    # says AWY received, so HME receives the second half.
    return pd.DataFrame(
        [
            _event(
                order_sequence=30,
                raw_play_id="30",
                known_at="2025-09-07T17:04:00Z",
                game_seconds_remaining=3450,
                possession_before="HME",
                actor_team="HME",
                pre_home_score=7,
                pre_away_score=0,
            ),
            _event(
                order_sequence=10,
                raw_play_id="10",
                known_at="2025-09-07T17:02:00Z",
                game_seconds_remaining=3600,
                possession_before="AWY",
                actor_team="HME",
                primary_action="KICKOFF",
                down=None,
                yardline_100=35.0,
            ),
            _event(
                order_sequence=20,
                raw_play_id="20",
                known_at="2025-09-07T17:03:00Z",
                game_seconds_remaining=3500,
                possession_before="AWY",
                actor_team="AWY",
            ),
        ]
    )


def test_builds_sorted_pit_states_and_ignores_precomputed_diagnostics() -> None:
    predictor = _RecordingPredictor()
    first = build_game_reference_tables(
        _base_events(),
        predictor=predictor,
        model=_model(),
        expected_pbp_source_sha256=_sha("c"),
    )
    tampered = _base_events()
    tampered["home_wp_before_diagnostic"] = [0.02, 0.88, 0.44]
    tampered["home_wp_after_diagnostic"] = [0.98, 0.12, 0.66]
    second = build_game_reference_tables(
        tampered,
        predictor=_RecordingPredictor(),
        model=_model(),
        expected_pbp_source_sha256=_sha("c"),
    )

    assert first.state_predictions["state_raw_play_id"].tolist() == [
        "10",
        "20",
        "30",
    ]
    supported = first.state_predictions.query(
        "reference_status == 'SUPPORTED'"
    )
    assert supported["state_raw_play_id"].tolist() == ["20", "30"]
    assert supported["p_home"].tolist() == pytest.approx([0.30, 0.60])
    assert (
        first.state_predictions["state_input_sha256"].tolist()
        == second.state_predictions["state_input_sha256"].tolist()
    )
    pd.testing.assert_series_equal(
        first.state_predictions["p_home"],
        second.state_predictions["p_home"],
    )

    input_by_clock = {
        item.feature_values[3]: item for item in predictor.inputs
    }
    assert input_by_clock[3500.0].feature_values[0] == 0.0
    assert input_by_clock[3450.0].feature_values[0] == 1.0
    assert input_by_clock[3450.0].feature_values[5] == 7.0
    assert input_by_clock[3450.0].feature_names == FEATURE_NAMES

    play_20 = first.reference_observations.loc[
        first.reference_observations["raw_play_id"].eq("20")
    ].iloc[0]
    assert play_20["reference_status"] == "SUPPORTED"
    assert play_20["pre_state_raw_play_id"] == "20"
    assert play_20["post_state_raw_play_id"] == "30"
    assert play_20["pre_state_known_at"] == "2025-09-07T17:03:00Z"
    assert play_20["post_state_known_at"] == "2025-09-07T17:04:00Z"
    assert play_20["p_before_home"] == pytest.approx(0.30)
    assert play_20["p_after_home"] == pytest.approx(0.60)
    assert play_20["reference_delta_home"] == pytest.approx(0.30)
    assert play_20["intervening_episode_ids"] == "[]"

    feature_rows = first.feature_provenance.loc[
        first.feature_provenance["state_raw_play_id"].eq("30")
    ]
    assert len(feature_rows) == 11
    kickoff_feature = feature_rows.loc[
        feature_rows["feature_name"].eq("receive_2h_ko")
    ].iloc[0]
    assert kickoff_feature["feature_known_at"] == "2025-09-07T17:02:00Z"
    assert kickoff_feature["source_raw_play_id"] == "10"
    assert set(feature_rows["pit_status"]) == {"PIT_VERIFIED"}


def test_composite_transition_keeps_bridge_separate_from_atomic_delta() -> None:
    events = _base_events()
    intermediate = _event(
        order_sequence=25,
        raw_play_id="25",
        known_at="2025-09-07T17:03:30Z",
        game_seconds_remaining=3480,
        primary_action="TRY",
        down=None,
        yardline_100=None,
    )
    events = pd.concat([events, pd.DataFrame([intermediate])], ignore_index=True)

    result = build_game_reference_tables(
        events,
        predictor=_RecordingPredictor(),
        model=_model(),
        expected_pbp_source_sha256=_sha("c"),
    )
    observation = result.reference_observations.loc[
        result.reference_observations["raw_play_id"].eq("20")
    ].iloc[0]

    assert observation["reference_status"] == "COMPOSITE_TRANSITION"
    assert observation["p_before_home"] == pytest.approx(0.30)
    assert pd.isna(observation["p_after_home"])
    assert pd.isna(observation["reference_delta_home"])
    assert observation["bridge_p_after_home"] == pytest.approx(0.60)
    assert observation["bridge_delta_home"] == pytest.approx(0.30)
    assert json.loads(observation["intervening_episode_ids"]) == [
        intermediate["atomic_information_episode_id"]
    ]


def test_event_ineligibility_ot_and_missing_post_are_explicit() -> None:
    events = _base_events()
    events.loc[events["raw_play_id"].eq("20"), "final_sports_outcome_eligible"] = (
        False
    )
    events.loc[events["raw_play_id"].eq("20"), "row_disposition"] = (
        "NULLIFIED_NO_PLAY"
    )
    ot = _event(
        order_sequence=40,
        raw_play_id="40",
        known_at="2025-09-07T17:05:00Z",
        quarter=5,
        game_seconds_remaining=600,
    )
    events = pd.concat([events, pd.DataFrame([ot])], ignore_index=True)

    result = build_game_reference_tables(
        events,
        predictor=_RecordingPredictor(),
        model=_model(),
        expected_pbp_source_sha256=_sha("c"),
    )
    statuses = dict(
        zip(
            result.reference_observations["raw_play_id"],
            result.reference_observations["reference_status"],
            strict=True,
        )
    )

    assert statuses["20"] == "EVENT_INELIGIBLE"
    assert statuses["30"] == "MISSING_POST_STATE"
    assert statuses["40"] == "MODEL_SUPPORT_UNPROVEN"
    ineligible = result.reference_observations.loc[
        result.reference_observations["raw_play_id"].eq("20")
    ].iloc[0]
    assert pd.isna(ineligible["p_before_home"])
    assert pd.isna(ineligible["p_after_home"])


def test_missing_or_future_opening_kickoff_evidence_fails_closed() -> None:
    no_kickoff = _base_events().loc[
        ~_base_events()["primary_action"].eq("KICKOFF")
    ]
    missing = build_game_reference_tables(
        no_kickoff,
        predictor=_RecordingPredictor(),
        model=_model(),
        expected_pbp_source_sha256=_sha("c"),
    )
    assert set(missing.state_predictions["reference_status"]) == {
        "MISSING_AS_OF_RECEIVER"
    }

    future = _base_events()
    future.loc[future["raw_play_id"].eq("10"), "known_at"] = (
        "2025-09-07T18:00:00Z"
    )
    rejected = build_game_reference_tables(
        future,
        predictor=_RecordingPredictor(),
        model=_model(),
        expected_pbp_source_sha256=_sha("c"),
    )
    by_play = rejected.state_predictions.set_index("state_raw_play_id")
    assert by_play.loc["20", "reference_status"] == "FUTURE_FEATURE_REJECTED"
    assert by_play.loc["30", "reference_status"] == "FUTURE_FEATURE_REJECTED"


def test_contracts_reject_hash_or_supported_null_probability() -> None:
    provenance = ReferenceFeatureProvenanceV1(
        game_id="2025_01_AWY_HME",
        state_event_id="event-1",
        state_raw_play_id="1",
        state_known_at="2025-09-07T17:00:00Z",
        feature_name="down",
        feature_value=1.0,
        feature_known_at="2025-09-07T17:00:00Z",
        source_event_id="event-1",
        source_raw_play_id="1",
        source_hash=_sha("c"),
        pit_status="PIT_VERIFIED",
    )
    assert provenance.schema_version == "ReferenceFeatureProvenanceV1"
    with pytest.raises(ReferenceValueBuildError, match="source_hash"):
        replace(provenance, source_hash="not-a-hash")

    with pytest.raises(
        ReferenceValueBuildError,
        match="supported reference requires",
    ):
        ReferenceValueObservationV1(
            game_id="2025_01_AWY_HME",
            event_id="event-1",
            raw_play_id="1",
            atomic_information_episode_id="episode-1",
            source_interval_start="2025-09-07T17:00:00Z",
            source_interval_end="2025-09-07T17:00:01Z",
            pre_state_event_id="event-1",
            pre_state_raw_play_id="1",
            pre_state_known_at="2025-09-07T17:00:00Z",
            post_state_event_id="event-2",
            post_state_raw_play_id="2",
            post_state_known_at="2025-09-07T17:00:02Z",
            p_before_home=0.4,
            p_after_home=None,
            reference_delta_home=None,
            bridge_p_after_home=None,
            bridge_delta_home=None,
            intervening_episode_ids=(),
            reference_status="SUPPORTED",
            exclusion_reasons=(),
            pre_state_input_sha256=_sha("d"),
            post_state_input_sha256=_sha("e"),
            model_id=_model().model_id,
            model_sha256=_model().model_sha256,
            model_manifest_sha256=_model().manifest_sha256,
            pbp_source_sha256=_sha("c"),
        )
