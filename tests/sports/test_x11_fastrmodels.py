from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.models.nfl_fastrmodels import (
    FEATURE_NAMES,
    NoSpreadModelInput,
)
from prediction_market.sports import x11_fastrmodels as reproduction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "game-state"
    / "nfl"
    / "fastrmodels_no_spread_reproduction_v2.json"
)
SUPERSEDED_ARTIFACT_PATH = ARTIFACT_PATH.with_name(
    "fastrmodels_no_spread_reproduction_v1.json"
)


def _native_row(
    *,
    game_id: str,
    ordinal: int,
    order_sequence: int,
    result: int,
    posteam: str = "HME",
    defteam: str = "AWY",
    qtr: int = 1,
    home_opening_kickoff: int = 0,
    home_wp: float = 0.6,
    ep: float | None = 1.0,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": 2021,
        "season_type": "REG",
        "home_team": "HME",
        "away_team": "AWY",
        "posteam": posteam,
        "defteam": defteam,
        "qtr": qtr,
        "order_sequence": order_sequence,
        "_raw_record_ordinal": ordinal,
        "ep": ep,
        "play_type": "run",
        "score_differential": 7.0 if posteam == "HME" else -7.0,
        "half_seconds_remaining": 1200.0,
        "game_seconds_remaining": 3000.0,
        "down": 2.0,
        "ydstogo": 5.0,
        "yardline_100": 45.0,
        "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 2.0,
        "home_opening_kickoff": home_opening_kickoff,
        "result": result,
        "home_wp": home_wp,
    }


def _prepared_frame() -> pd.DataFrame:
    native = pd.DataFrame(
        [
            _native_row(
                game_id="2021_01_AWY_HME",
                ordinal=0,
                order_sequence=10,
                result=3,
                home_wp=0.61,
            ),
            _native_row(
                game_id="2021_01_AWY_HME",
                ordinal=1,
                order_sequence=10,
                result=3,
                posteam="AWY",
                defteam="HME",
                home_wp=0.58,
            ),
            _native_row(
                game_id="2021_02_AWY_HME",
                ordinal=2,
                order_sequence=20,
                result=-7,
                home_wp=0.42,
            ),
            _native_row(
                game_id="2021_03_AWY_HME",
                ordinal=3,
                order_sequence=30,
                result=0,
                home_wp=0.50,
            ),
            _native_row(
                game_id="2021_04_AWY_HME",
                ordinal=4,
                order_sequence=40,
                result=10,
                ep=None,
            ),
        ]
    )
    return reproduction.prepare_partition_frame(
        native,
        season=2021,
        manifest_sha256="sha256:" + "1" * 64,
        object_sha256="sha256:" + "2" * 64,
    ).frame


def test_committed_reproduction_gate_is_exact_and_precedes_evaluation() -> None:
    evaluation_started_at = datetime.now(timezone.utc)

    gate = reproduction.verify_reproduction_gate(
        PROJECT_ROOT,
        evaluation_started_at=evaluation_started_at,
    )

    assert gate.registration_head_sha256 == (
        "sha256:"
        "0dcd4a1a62c7790967023b2383a2cb93eaf35b25e3e4d64baabe8decb8f45960"
    )
    assert gate.scope == "team_h_nfl_fastrmodels_reproduction_v2"
    assert gate.dataset_ids == ("DS-NFL-FASTRMODELS", "DS-NFLVERSE")
    assert gate.model_ids == (
        "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
    )
    assert gate.amended_at < evaluation_started_at


def test_result_timestamp_uses_registry_second_precision() -> None:
    observed = datetime(
        2026,
        7,
        24,
        0,
        52,
        31,
        987654,
        tzinfo=timezone.utc,
    )

    assert reproduction._canonical_utc(observed) == "2026-07-24T00:52:31Z"


def test_filter_retains_duplicate_order_with_raw_ordinal_and_excludes_missing() -> None:
    native = pd.DataFrame(
        [
            _native_row(
                game_id="2021_01_AWY_HME",
                ordinal=0,
                order_sequence=10,
                result=3,
            ),
            _native_row(
                game_id="2021_01_AWY_HME",
                ordinal=1,
                order_sequence=10,
                result=3,
                posteam="AWY",
                defteam="HME",
            ),
            _native_row(
                game_id="2021_02_AWY_HME",
                ordinal=2,
                order_sequence=20,
                result=-7,
            ),
            _native_row(
                game_id="2021_03_AWY_HME",
                ordinal=3,
                order_sequence=30,
                result=0,
            ),
            _native_row(
                game_id="2021_04_AWY_HME",
                ordinal=4,
                order_sequence=40,
                result=10,
                ep=None,
            ),
        ]
    )
    prepared = reproduction.prepare_partition_frame(
        native,
        season=2021,
        manifest_sha256="sha256:" + "1" * 64,
        object_sha256="sha256:" + "2" * 64,
    )
    frame = prepared.frame

    assert len(frame) == 4
    assert prepared.census["eligible_rows"] == 3
    assert prepared.census["eligible_non_tie_games"] == 2
    duplicate = frame.loc[frame["game_id"] == "2021_01_AWY_HME"]
    assert duplicate["order_sequence"].tolist() == [10, 10]
    assert duplicate["raw_record_ordinal"].tolist() == [0, 1]
    assert duplicate["row_key"].is_unique


def test_labels_use_only_final_result_and_ties_stay_out_of_binary_metrics() -> None:
    baseline = _prepared_frame()
    mutated = baseline.copy()
    mutated["home_wp"] = 1.0 - mutated["home_wp"]

    assert reproduction.binary_labels(baseline).tolist() == [1, 1, 0]
    assert reproduction.binary_labels(mutated).tolist() == [1, 1, 0]
    assert reproduction.binary_metric_frame(baseline)["game_id"].nunique() == 2
    assert "2021_03_AWY_HME" not in set(
        reproduction.binary_metric_frame(baseline)["game_id"]
    )


def test_receive_second_half_flag_follows_pinned_helper_definition() -> None:
    first_half_home_receiver = _native_row(
        game_id="2021_01_AWY_HME",
        ordinal=0,
        order_sequence=1,
        result=1,
        home_opening_kickoff=0,
    )
    first_half_away_receiver = dict(
        first_half_home_receiver,
        posteam="AWY",
        defteam="HME",
        home_opening_kickoff=1,
    )
    second_half = dict(
        first_half_home_receiver,
        qtr=3,
        half_seconds_remaining=1200.0,
        game_seconds_remaining=1200.0,
    )

    home_input = reproduction.project_no_spread_input(first_half_home_receiver)
    away_input = reproduction.project_no_spread_input(first_half_away_receiver)
    second_half_input = reproduction.project_no_spread_input(second_half)

    assert home_input.feature_names == FEATURE_NAMES
    assert home_input.feature_values[0] == 1.0
    assert away_input.feature_values[0] == 1.0
    assert second_half_input.feature_values[0] == 0.0


def test_projection_preserves_native_end_of_first_half_boundary() -> None:
    boundary = _native_row(
        game_id="2021_05_IND_BAL",
        ordinal=12451,
        order_sequence=1960,
        result=6,
        posteam="IND",
        defteam="BAL",
        qtr=2,
        home_opening_kickoff=1,
    )
    boundary.update(
        home_team="BAL",
        away_team="IND",
        half_seconds_remaining=0.0,
        game_seconds_remaining=1800.0,
        score_differential=4.0,
        down=1.0,
        ydstogo=5.0,
        yardline_100=19.0,
        posteam_timeouts_remaining=0.0,
        defteam_timeouts_remaining=3.0,
    )

    model_input = reproduction.project_no_spread_input(boundary)

    assert model_input.feature_values == pytest.approx(
        (
            1.0,
            0.0,
            0.0,
            1800.0,
            29.556224395722598,
            4.0,
            1.0,
            5.0,
            19.0,
            0.0,
            3.0,
        ),
        abs=1e-12,
    )


def test_metrics_are_game_clustered_deterministic_and_protocol_locked() -> None:
    frame = _prepared_frame()
    extra_rows = []
    for index, source_index in enumerate((0, 2, 0, 2), start=5):
        row = frame.iloc[source_index].copy()
        row["game_id"] = f"2021_{index:02d}_AWY_HME"
        row["row_key"] = f"synthetic-{index}"
        extra_rows.append(row)
    frame = pd.concat([frame, pd.DataFrame(extra_rows)], ignore_index=True)
    frame["home_probability"] = [
        0.5 if outcome == "tie" else 0.7 if outcome == "home_win" else 0.3
        for outcome in frame["final_outcome"]
    ]

    first = reproduction.evaluate_metrics(frame)
    second = reproduction.evaluate_metrics(frame)

    assert reproduction.metrics_bytes(first) == reproduction.metrics_bytes(second)
    assert first["aggregate"]["row_micro"]["clusters"] == 6
    assert first["aggregate"]["row_micro"]["observations"] == 7
    assert first["aggregate"]["row_micro"]["bootstrap_samples_requested"] == 200
    assert (
        reproduction.MINIMUM_VALID_BOOTSTRAP_SAMPLES
        <= first["aggregate"]["row_micro"]["bootstrap_samples_valid"]
        <= 200
    )
    assert first["bootstrap_seed"] == 20260723
    assert first["ties"]["games"] == 1
    assert first["ties"]["excluded_from_binary_metrics"] is True
    assert len(first["aggregate"]["reliability"]) == 10


def test_latency_contract_is_full_path_and_excludes_external_work() -> None:
    assert reproduction.FULL_PATH_INCLUDED_STAGES == (
        "normalized_event_construction",
        "state_event_transition",
        "official_feature_projection",
        "preloaded_official_booster",
        "probability_output_validation",
    )
    assert reproduction.LATENCY_EXCLUDED_STAGES == (
        "io",
        "market_join",
        "network",
        "registry_loading",
    )


def test_full_path_prediction_executes_all_included_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    prior_state = object()
    transitioned_state = object()
    model_input = object()
    fixture = reproduction.FullPathFixture(
        prior_state=prior_state,  # type: ignore[arg-type]
        source_row={"native": "source"},
        successor_rows=({"native": "post"},),
        sequence=1,
        raw_object_sha256="sha256:" + "a" * 64,
        source_version="frozen",
        second_half_receiver="HME",
    )

    def actual_event(*args: object, **kwargs: object) -> object:
        calls.append("normalized_event_construction")
        return object()

    def reduce_state(state: object, event: object) -> object:
        assert state is prior_state
        calls.append("state_event_transition")
        return transitioned_state

    def project_state(
        state: object,
        *,
        second_half_receiver: str,
    ) -> object:
        assert state is transitioned_state
        assert second_half_receiver == "HME"
        calls.append("official_feature_projection")
        return model_input

    class Predictor:
        def predict_home(self, value: object) -> float:
            assert value is model_input
            calls.append("preloaded_official_booster")
            return 0.75

    original_validate = reproduction._validate_probability

    def validate_probability(value: object) -> float:
        calls.append("probability_output_validation")
        return original_validate(value)

    monkeypatch.setattr(
        reproduction.season_census,
        "_actual_event",
        actual_event,
    )
    monkeypatch.setattr(reproduction.nfl_state, "reduce", reduce_state)
    monkeypatch.setattr(reproduction, "_input_from_state", project_state)
    monkeypatch.setattr(
        reproduction,
        "_validate_probability",
        validate_probability,
    )

    assert reproduction._full_path_prediction(  # type: ignore[arg-type]
        fixture,
        Predictor(),
    ) == 0.75
    assert tuple(calls) == reproduction.FULL_PATH_INCLUDED_STAGES


def test_checked_in_artifact_has_exact_census_boundaries_and_self_hash() -> None:
    document = reproduction.read_reproduction_artifact(ARTIFACT_PATH)
    material = copy.deepcopy(document)
    observed_hash = material.pop("artifact_sha256")

    assert observed_hash == reproduction.canonical_sha256(material)
    assert document["artifact_id"] == (
        "NFL-FASTRMODELS-NO-SPREAD-REPRODUCTION-V2"
    )
    assert document["artifact_version"] == "v2"
    assert not SUPERSEDED_ARTIFACT_PATH.exists()
    assert document["status"] == "PRELIMINARY"
    assert document["pit_status"] == "PIT_UNPROVEN"
    assert document["observation_mode"] == "offline_reconstruction_not_live_PIT"
    assert document["evidence_boundaries"] == {
        "alpha_evidence": "none",
        "calibrator": "none_fitted",
        "market_data_used": False,
        "prediction_market_alignment": "none",
        "prediction_market_symmetry": "not_evaluated",
        "training": "official_model_not_retrained",
    }
    assert document["metrics"]["census"]["total"] == {
        "eligible_non_tie_games": 1420,
        "eligible_rows": 205607,
        "games": 1424,
        "raw_rows": 247284,
    }
    assert document["metrics"]["evaluation"]["ties"] == {
        "excluded_from_binary_metrics": True,
        "game_ids": [
            "2021_10_DET_PIT",
            "2022_01_IND_HOU",
            "2022_13_WAS_NYG",
            "2025_04_GB_DAL",
        ],
        "games": 4,
        "rows": 595,
    }
    assert document["determinism"]["complete_runs"] == 2
    assert document["determinism"]["metrics_bytes_identical"] is True
    assert document["latency"]["measurement_scope"] == "full_path"
    assert document["latency"]["included_stages"] == list(
        reproduction.FULL_PATH_INCLUDED_STAGES
    )
    assert document["latency"]["fixture"]["raw_object_sha256"] == (
        "sha256:"
        "3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a"
    )
    assert document["latency"]["samples"] == 1000
    assert reproduction.metrics_bytes(document)
