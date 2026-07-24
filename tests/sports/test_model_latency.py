from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports import (
    model_latency,
    soccer_game_state,
    soccer_transition_model,
    x12,
)
from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_bundle,
)
from prediction_market.sports.model_latency import (
    LatencyBenchmarkError,
    _fixed_point_probabilities,
    benchmark_model_stages,
    measure_warm_inference,
    resolve_json_pointer,
    unavailable_sport_record,
    validate_model_latency_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_nfl_latency_builder_uses_canonical_source_order_not_index_or_play_id() -> None:
    raw = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AWY_HME",
                "play_id": 100,
                "order_sequence": 30,
                "_raw_record_ordinal": 3,
                "fixed_drive": 2,
                "posteam": "HME",
            },
            {
                "game_id": "2025_01_AWY_HME",
                "play_id": 200,
                "order_sequence": 20,
                "_raw_record_ordinal": 2,
                "fixed_drive": 2,
                "posteam": "HME",
            },
            {
                "game_id": "2025_01_AWY_HME",
                "play_id": 300,
                "order_sequence": 15,
                "_raw_record_ordinal": 1,
                "fixed_drive": 2,
                "posteam": "HME",
            },
            {
                "game_id": "2025_01_AWY_HME",
                "play_id": 400,
                "order_sequence": 10,
                "_raw_record_ordinal": 0,
                "fixed_drive": 1,
                "posteam": "AWY",
            },
        ],
        index=(90, 10, 70, 50),
    )
    candidate = pd.Series(
        {
            "game_id": "2025_01_AWY_HME",
            "drive_number": 2,
            "play_id": 200,
        }
    )

    pre_row, post_row = model_latency._nfl_latency_transition_rows(
        raw,
        candidate,
    )

    assert pre_row["order_sequence"] == 15
    assert post_row["order_sequence"] == 20
    assert pre_row["play_id"] > post_row["play_id"]


def test_nfl_latency_report_binds_the_selected_raw_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_ordinals: tuple[int, int] | None = None
    select_transition = model_latency._nfl_latency_transition_rows

    def capture_transition(
        raw: pd.DataFrame,
        candidate: pd.Series,
    ) -> tuple[dict[str, object], dict[str, object]] | None:
        nonlocal selected_ordinals
        rows = select_transition(raw, candidate)
        if rows is not None:
            selected_ordinals = (
                int(rows[0]["_raw_record_ordinal"]),
                int(rows[1]["_raw_record_ordinal"]),
            )
        return rows

    monkeypatch.setattr(
        model_latency,
        "_nfl_latency_transition_rows",
        capture_transition,
    )
    monkeypatch.setattr(
        model_latency,
        "benchmark_model_stages",
        lambda **_arguments: {},
    )
    monkeypatch.setattr(
        model_latency,
        "_strict_registered_model_output",
        lambda *_arguments, **_keywords: pytest.fail(
            "a source_at=None NFL sample must not materialize ModelOutputV1"
        ),
    )
    registry_row = next(
        row
        for row in model_latency.load_model_registry(PROJECT_ROOT)
        if row.model_id == "MODEL-NFL-DRIVE-TRANSITION"
    )

    report = model_latency._nfl_benchmark(
        program_root=PROJECT_ROOT,
        registry_row=registry_row,
        warmup=1,
        repeats=1,
    )

    assert selected_ordinals is not None
    assert report["representative_input_lineage"]["raw_record_ordinals"] == list(
        selected_ordinals
    )
    assert report["contract_output"] is None
    assert report["pit_status"] == "PIT_UNPROVEN"
    assert report["pit_cutoff_at"] is None
    assert report["output_validation_scope"] == (
        "probability_domain_and_sum_only"
    )


def _statsbomb_half_start_event() -> tuple[
    soccer_game_state.SoccerGameState,
    soccer_game_state.SoccerGameEvent,
]:
    game_id = "game_statsbomb_latency_fixture"
    raw_event = {
        "id": "statsbomb-latency-event-3",
        "index": 3,
        "period": 1,
        "timestamp": "00:00:00.000",
        "minute": 0,
        "second": 0,
        "type": {"id": 18, "name": "Half Start"},
        "team": {"id": 10, "name": "Home"},
        "possession": 1,
        "possession_team": {"id": 10, "name": "Home"},
    }
    payload = soccer_game_state.statsbomb_event_payload(
        raw_event,
        game_id=game_id,
    )
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-12",
        dataset_id="DS-STATSBOMB-OPEN",
        source_system="statsbomb",
        source_stream="events",
        raw_object_hash="sha256:" + "8" * 64,
        raw_record_ordinals=(2,),
        partition="synthetic-latency-fixture",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=None,
        competition_id="cmp_statsbomb_2",
        game_id=game_id,
        participant_ids=(
            "participant_statsbomb_10",
            "participant_statsbomb_20",
        ),
        native_namespace="statsbomb.event",
        native_ids=("statsbomb-latency-event-3",),
        normalized_source_sequence=3,
        normalized_payload=payload,
    )
    event = soccer_game_state.adapt_statsbomb_event(
        bundle.normalized,
        program_root=PROJECT_ROOT,
        raw_parents=bundle.raw,
    )
    return (
        soccer_game_state.SoccerGameState(
            game_id=game_id,
            home_team_id=10,
            away_team_id=20,
            lifecycle="not_started",
            sequence=2,
            period=1,
            possession_id=1,
            possession_team_id=10,
            last_action="Starting XI",
            last_event_id="evt_" + "2" * 64,
            active_players=(
                soccer_game_state.SoccerTeamPlayers(
                    team_id=10,
                    player_ids=tuple(range(1_001, 1_012)),
                ),
                soccer_game_state.SoccerTeamPlayers(
                    team_id=20,
                    player_ids=tuple(range(2_001, 2_012)),
                ),
            ),
        ),
        event,
    )


def _rehash_report(report: dict[str, object]) -> None:
    unhashed = deepcopy(report)
    unhashed.pop("report_sha256")
    payload = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    report["report_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()


def test_measure_warm_inference_counts_only_timed_single_observations() -> None:
    calls = 0

    def inference() -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return (0.25, 0.75)

    result = measure_warm_inference(
        inference,
        validator=lambda value: sum(value) == 1.0,
        warmup=7,
        repeats=1_001,
    )

    assert calls == 1_008
    assert result["measurement"] == "single_observation_warm_stage"
    assert result["timer"] == "time.perf_counter_ns"
    assert result["warmup_iterations"] == 7
    assert result["timed_iterations"] == 1_001
    assert result["unit"] == "nanoseconds"
    assert result["measured_operations"] == 1_001
    assert result["total_ns"] > 0
    assert result["operations_per_second"] == (
        1_001 * 1_000_000_000 // result["total_ns"]
    )
    assert 0 <= result["p50_ns"] <= result["p95_ns"]
    assert result["p95_ns"] <= result["p99_ns"] <= result["max_ns"]


@pytest.mark.parametrize(
    ("warmup", "repeats", "message"),
    [
        (0, 1_000, "warmup"),
        (1, 999, "at least 1000"),
    ],
)
def test_measure_warm_inference_rejects_invalid_protocol(
    warmup: int,
    repeats: int,
    message: str,
) -> None:
    with pytest.raises(LatencyBenchmarkError, match=message):
        measure_warm_inference(
            lambda: 1,
            validator=lambda value: value == 1,
            warmup=warmup,
            repeats=repeats,
        )


def test_measure_warm_inference_fails_closed_on_invalid_output() -> None:
    with pytest.raises(LatencyBenchmarkError, match="invalid output"):
        measure_warm_inference(
            lambda: (0.1, 0.1),
            validator=lambda value: sum(value) == 1.0,
            warmup=1,
            repeats=1_000,
        )


def test_benchmark_model_stages_requires_complete_stage_separation() -> None:
    valid = lambda value: value == 1
    report = benchmark_model_stages(
        model_id="MODEL-TEST",
        experiment_id="X-11",
        stages={
            "state_reducer": lambda: 1,
            "feature_extraction": lambda: 1,
            "model_inference": lambda: 1,
            "output_validation": lambda: 1,
            "full_path": lambda: 1,
        },
        validators={name: valid for name in (
            "state_reducer",
            "feature_extraction",
            "model_inference",
            "output_validation",
            "full_path",
        )},
        warmup=1,
        repeats=1_000,
    )

    assert report["model_id"] == "MODEL-TEST"
    assert report["experiment_id"] == "X-11"
    assert list(report["stages"]) == [
        "state_reducer",
        "feature_extraction",
        "model_inference",
        "output_validation",
        "full_path",
    ]
    assert report["measured_events"] == 1_000
    assert report["full_path_events_per_second"] == (
        report["stages"]["full_path"]["operations_per_second"]
    )
    assert all(
        stage["timed_iterations"] == 1_000
        for stage in report["stages"].values()
    )

    with pytest.raises(LatencyBenchmarkError, match="stage inventory"):
        benchmark_model_stages(
            model_id="MODEL-TEST",
            experiment_id="X-11",
            stages={"full_path": lambda: 1},
            validators={"full_path": valid},
            warmup=1,
            repeats=1_000,
        )


def test_resolve_json_pointer_implements_rfc6901_and_fails_closed() -> None:
    document = {
        "a/b": {"~key": [{"value": 7}]},
        "": "empty-key",
    }

    assert resolve_json_pointer(document, "/a~1b/~0key/0/value") == 7
    assert resolve_json_pointer(document, "/") == "empty-key"
    assert resolve_json_pointer(document, "") == document

    for pointer in (
        "a/b",
        "/a~2b",
        "/a~1b/~0key/01",
        "/a~1b/~0key/-",
        "/missing",
    ):
        with pytest.raises(LatencyBenchmarkError, match="JSON Pointer"):
            resolve_json_pointer(document, pointer)


def test_fixed_point_output_is_exact_for_fitted_model_float_precision() -> None:
    labels = ("touchdown", "field_goal", "punt", "turnover", "other")
    probabilities = (
        0.226023809957540,
        0.160838675237223,
        0.499917167471596,
        0.082806631680007,
        0.030413715653634,
    )

    encoded = _fixed_point_probabilities(labels, probabilities)

    scales = {item["scale"] for item in encoded.values()}
    assert scales == {15}
    assert sum(int(item["atoms"]) for item in encoded.values()) == 10**15


def test_unavailable_sports_never_receive_placeholder_latency() -> None:
    for sport in ("nba", "mlb", "f1"):
        record = unavailable_sport_record(sport)
        assert record == {
            "sport": sport,
            "status": "not_measured_no_eligible_model",
            "models": [],
            "latency": None,
        }


def _soccer_fixture_models() -> tuple[
    x12.DixonColesModel,
    soccer_transition_model.DynamicIntensityModel,
    soccer_transition_model.TemperatureCalibration,
]:
    return (
        x12.DixonColesModel(
            team_ids=(10, 20),
            reference_team_id=10,
            parameters=(0.0, 0.1, -0.1, 0.2),
            home_advantage=0.2,
            rho=-0.05,
            initial_objective=100.0,
            initial_projected_gradient_inf_norm=1.0,
            objective=90.0,
            objective_improvement=10.0,
            parameter_displacement=0.4,
            projected_gradient_inf_norm=1e-6,
            iterations=12,
            optimizer_status="synthetic_converged",
        ),
        soccer_transition_model.DynamicIntensityModel(
            coefficients=(-0.20, -0.45, -0.90),
            l2_penalty=1.0,
            objective=0.0,
            iterations=0,
            optimizer_status="engineering_fixture_not_empirical",
        ),
        soccer_transition_model.TemperatureCalibration(
            temperature=1.7,
            initial_objective=0.8,
            objective=0.7,
            iterations=4,
            optimizer_status="synthetic_converged",
            calibration_match_count=24,
            calibration_observation_count=432,
        ),
    )


def _calibrated_soccer_evidence() -> dict[str, object]:
    _, dynamic_model, calibration = _soccer_fixture_models()
    dixon_coles_sha256 = "sha256:" + "1" * 64
    raw_transition_sha256 = x12._sha256(
        {
            "dixon_coles_parameter_sha256": dixon_coles_sha256,
            "dynamic_parameter_sha256": dynamic_model.parameter_sha256,
            "model_id": x12.X12_MODEL_ID,
            "model_version": x12.X12_MODEL_VERSION,
            "probability_variant": "uncalibrated",
        }
    )
    calibrated_transition_sha256 = x12._sha256(
        {
            "model_id": x12.X12_MODEL_ID,
            "model_version": x12.X12_MODEL_VERSION,
            "raw_transition_parameter_sha256": raw_transition_sha256,
            "temperature_parameter_sha256": calibration.parameter_sha256,
            "probability_variant": "temperature_calibrated",
        }
    )
    return {
        "artifact_type": (
            "x12_real_data_dixon_coles_dynamic_transition_poc_v1"
        ),
        "experiment_id": "X-12",
        "model_id": x12.X12_MODEL_ID,
        "model_version": x12.X12_MODEL_VERSION,
        "authorization_scope": x12.X12_AUTHORIZATION_SCOPE,
        "result_label": "PRELIMINARY",
        "execution_mode": "full",
        "registration_preflight": {
            "experiment_id": "X-12",
            "scope": x12.X12_AUTHORIZATION_SCOPE,
            "result_label": "PRELIMINARY",
            "status": "resolved",
            "registration_head_sha256": "sha256:" + "9" * 64,
        },
        "model": {
            "optimizer_max_iterations": 250,
            "transition_model": {
                "methodology": (
                    "Maia-family dynamic-covariate adaptation with frozen "
                    "Dixon-Coles base-rate offset"
                ),
                "reproduction_scope": (
                    "not a complete Maia or Cox reproduction"
                ),
                "l2_penalty": 1.0,
                "optimizer_max_iterations": 250,
                "split": {
                    "method": (
                        "frozen_chronological_date_group_holdout_50_25_25"
                    ),
                    "base_fit_first_date": "2015-08-08T00:00:00+00:00",
                    "base_fit_last_date": "2016-01-01T00:00:00+00:00",
                    "calibration_first_date": "2016-01-02T00:00:00+00:00",
                    "calibration_last_date": "2016-03-01T00:00:00+00:00",
                    "final_test_first_date": "2016-03-02T00:00:00+00:00",
                    "final_test_last_date": "2016-05-15T00:00:00+00:00",
                    "final_test_evaluated_first_date": (
                        "2016-03-02T00:00:00+00:00"
                    ),
                    "final_test_evaluated_last_date": (
                        "2016-05-01T00:00:00+00:00"
                    ),
                    "base_fit_date_count": 50,
                    "calibration_date_count": 25,
                    "final_test_date_count": 25,
                    "final_test_evaluated_date_count": 22,
                    "base_fit_match_count": 190,
                    "calibration_match_count": 90,
                    "final_test_match_count": 100,
                    "final_test_evaluated_match_count": 90,
                    "dynamic_fit_evaluation_cutoff": (
                        "2016-01-02T12:00:00+00:00"
                    ),
                    "temperature_fit_evaluation_cutoff": (
                        "2016-03-02T12:00:00+00:00"
                    ),
                    "base_fit_max_outcome_available_at": (
                        "2016-01-01T18:00:00+00:00"
                    ),
                    "calibration_max_label_available_at": (
                        "2016-03-01T22:00:00+00:00"
                    ),
                    "dixon_coles_parameter_sha256": dixon_coles_sha256,
                    "dynamic_parameter_sha256": (
                        dynamic_model.parameter_sha256
                    ),
                    "raw_transition_parameter_sha256": (
                        raw_transition_sha256
                    ),
                    "temperature_parameter_sha256": (
                        calibration.parameter_sha256
                    ),
                    "calibrated_transition_parameter_sha256": (
                        calibrated_transition_sha256
                    ),
                },
                "temperature_calibration": {
                    "temperature": calibration.temperature,
                    "initial_objective": calibration.initial_objective,
                    "objective": calibration.objective,
                    "iterations": calibration.iterations,
                    "optimizer_status": calibration.optimizer_status,
                    "calibration_match_count": 90,
                    "calibration_observation_count": 90 * 18,
                    "parameter_sha256": calibration.parameter_sha256,
                },
                "output_probability_variants": {
                    "primary": "temperature_calibrated",
                    "diagnostic": "uncalibrated",
                },
            },
        },
        "transition_output": {
            "state_space": list(soccer_transition_model.TRANSITION_CLASSES),
            "horizon_seconds": 300,
            "evaluation_protocol": (
                "frozen_chronological_date_group_holdout_50_25_25"
            ),
            "primary_probability_variant": "temperature_calibrated",
            "diagnostic_probability_variant": "uncalibrated",
            "metrics": {
                "classes": list(
                    soccer_transition_model.TRANSITION_CLASSES
                ),
                "observations": 1_620,
                "brier": 0.42,
                "log_loss": 0.71,
                "probability_variant": "temperature_calibrated",
                "raw_model_metrics": {
                    "classes": list(
                        soccer_transition_model.TRANSITION_CLASSES
                    ),
                    "observations": 1_620,
                    "brier": 0.45,
                    "log_loss": 0.75,
                    "probability_variant": "uncalibrated",
                },
            },
        },
    }


def test_soccer_latency_applies_fitted_temperature_after_dynamic_prediction() -> None:
    previous_state, event = _statsbomb_half_start_event()
    _, fixture_model, calibration = _soccer_fixture_models()

    next_state, features, distribution, calibrated = (
        model_latency._soccer_dynamic_transition(
            previous_state,
            event,
            model=fixture_model,
            temperature_calibration=calibration,
            base_home_goals=1.6,
            base_away_goals=1.2,
        )
    )

    assert event.event_id.startswith("evt_")
    assert next_state.sequence == event.sequence
    assert features.source_state_sha256 == next_state.state_sha256
    assert distribution.source_feature_sha256 == features.feature_sha256
    assert distribution.model_parameter_sha256 == fixture_model.parameter_sha256
    assert sum(distribution.probabilities) == pytest.approx(1.0, abs=1e-15)
    expected = soccer_transition_model.apply_multiclass_temperature(
        np.asarray([distribution.probabilities], dtype=float),
        temperature=calibration.temperature,
    )[0]
    assert calibrated == pytest.approx(expected, abs=0.0)
    assert calibrated.tolist() != pytest.approx(distribution.probabilities)


def test_soccer_parameter_snapshot_requires_bound_temperature_calibrator() -> None:
    base_model, dynamic_model, calibration = _soccer_fixture_models()
    snapshot = model_latency._soccer_model_parameter_snapshot(
        base_model,
        dynamic_model,
        calibration,
    )

    assert snapshot["temperature_calibration"]["parameter_sha256"] == (
        calibration.parameter_sha256
    )
    assert (
        model_latency._validate_model_parameter_snapshot(
            model_latency.DYNAMIC_SOCCER_MODEL_ID,
            snapshot,
        )
        == snapshot
    )

    missing = deepcopy(snapshot)
    del missing["temperature_calibration"]
    with pytest.raises(LatencyBenchmarkError, match="snapshot structure"):
        model_latency._validate_model_parameter_snapshot(
            model_latency.DYNAMIC_SOCCER_MODEL_ID,
            missing,
        )

    forged = deepcopy(snapshot)
    forged["temperature_calibration"]["parameter_sha256"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(LatencyBenchmarkError, match="calibration.*identity"):
        model_latency._validate_model_parameter_snapshot(
            model_latency.DYNAMIC_SOCCER_MODEL_ID,
            forged,
        )


def test_soccer_evidence_parser_requires_complete_calibrated_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _calibrated_soccer_evidence()
    monkeypatch.setattr(
        x12,
        "_validate_x12_reproduction_preflight",
        lambda _: deepcopy(evidence["registration_preflight"]),
    )

    parsed = model_latency._require_dynamic_soccer_evidence(
        evidence,
        program_root=PROJECT_ROOT,
    )

    assert isinstance(parsed["split"], x12.X12TransitionSplitAudit)
    assert isinstance(
        parsed["temperature_calibration"],
        soccer_transition_model.TemperatureCalibration,
    )
    assert parsed["metrics"] == evidence["transition_output"]["metrics"]
    assert parsed["split"].temperature_parameter_sha256 == (
        parsed["temperature_calibration"].parameter_sha256
    )


def test_soccer_evidence_parser_rejects_old_static_or_uncalibrated_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_preflight = _calibrated_soccer_evidence()[
        "registration_preflight"
    ]
    monkeypatch.setattr(
        x12,
        "_validate_x12_reproduction_preflight",
        lambda _: deepcopy(current_preflight),
    )
    invalid_documents: list[dict[str, object]] = []
    wrong_model = _calibrated_soccer_evidence()
    wrong_model["model_id"] = "MODEL-SOCCER-FIVE-MINUTE-TRANSITION"
    invalid_documents.append(wrong_model)

    wrong_version = _calibrated_soccer_evidence()
    wrong_version["model_version"] = "v0"
    invalid_documents.append(wrong_version)

    wrong_scope = _calibrated_soccer_evidence()
    wrong_scope["authorization_scope"] = "poc_result"
    invalid_documents.append(wrong_scope)

    bounded = _calibrated_soccer_evidence()
    bounded["execution_mode"] = "bounded_smoke"
    invalid_documents.append(bounded)

    incomplete_split = _calibrated_soccer_evidence()
    del incomplete_split["model"]["transition_model"]["split"][
        "calibration_max_label_available_at"
    ]
    invalid_documents.append(incomplete_split)

    incomplete_calibration = _calibrated_soccer_evidence()
    del incomplete_calibration["model"]["transition_model"][
        "temperature_calibration"
    ]["parameter_sha256"]
    invalid_documents.append(incomplete_calibration)

    uncalibrated_metrics = _calibrated_soccer_evidence()
    uncalibrated_metrics["transition_output"]["metrics"][
        "probability_variant"
    ] = "uncalibrated"
    invalid_documents.append(uncalibrated_metrics)

    forged_hash_chain = _calibrated_soccer_evidence()
    forged_hash_chain["model"]["transition_model"]["split"][
        "calibrated_transition_parameter_sha256"
    ] = "sha256:" + "0" * 64
    invalid_documents.append(forged_hash_chain)

    for invalid in invalid_documents:
        with pytest.raises(
            LatencyBenchmarkError,
            match="registered calibrated dynamic empirical evidence",
        ):
            model_latency._require_dynamic_soccer_evidence(
                invalid,
                program_root=PROJECT_ROOT,
            )


def test_soccer_evidence_preflight_fails_closed_when_missing_stale_or_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _calibrated_soccer_evidence()
    current_preflight = deepcopy(evidence["registration_preflight"])
    monkeypatch.setattr(
        x12,
        "_validate_x12_reproduction_preflight",
        lambda _: deepcopy(current_preflight),
    )

    missing = deepcopy(evidence)
    del missing["registration_preflight"]
    stale = deepcopy(evidence)
    stale["registration_preflight"]["registration_head_sha256"] = (
        "sha256:" + "0" * 64
    )
    for invalid in (missing, stale):
        with pytest.raises(
            LatencyBenchmarkError,
            match="current X-12 reproduction preflight",
        ):
            model_latency._require_dynamic_soccer_evidence(
                invalid,
                program_root=PROJECT_ROOT,
            )

    def unresolved(_: object) -> dict[str, object]:
        raise x12.X12DataError(
            "X-12 reproduction registration has unresolved locks"
        )

    monkeypatch.setattr(
        x12,
        "_validate_x12_reproduction_preflight",
        unresolved,
    )
    with pytest.raises(
        LatencyBenchmarkError,
        match="current X-12 reproduction preflight",
    ):
        model_latency._require_dynamic_soccer_evidence(
            evidence,
            program_root=PROJECT_ROOT,
        )


def test_soccer_latency_rejects_legacy_static_transition_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_evidence = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "game-state"
            / "soccer"
            / "x12_real_data_poc_v0.json"
        ).read_text(encoding="utf-8")
    )
    current_preflight = _calibrated_soccer_evidence()[
        "registration_preflight"
    ]
    legacy_evidence["registration_preflight"] = deepcopy(
        current_preflight
    )
    monkeypatch.setattr(
        x12,
        "_validate_x12_reproduction_preflight",
        lambda _: deepcopy(current_preflight),
    )

    with pytest.raises(
        LatencyBenchmarkError,
        match="registered calibrated dynamic empirical evidence",
    ):
        model_latency._require_dynamic_soccer_evidence(
            legacy_evidence,
            program_root=PROJECT_ROOT,
        )


def test_latency_protocol_migrates_atomically_to_dynamic_soccer_identity() -> None:
    assert model_latency.REPORT_VERSION == "v1"
    assert model_latency.DYNAMIC_SOCCER_MODEL_ID == (
        "MODEL-SOCCER-DYNAMIC-INTENSITY"
    )
    assert model_latency.DYNAMIC_SOCCER_EVIDENCE_PATH == (
        "artifacts/game-state/soccer/x12_dynamic_transition_poc_v1.json"
    )
    assert "MODEL-SOCCER-FIVE-MINUTE-TRANSITION" not in (
        model_latency.MEASURED_MODELS
    )
    assert model_latency.MEASURED_MODELS[
        model_latency.DYNAMIC_SOCCER_MODEL_ID
    ] == ("X-12", "v1")


def test_nfl_latency_limitation_states_v3_census_boundary() -> None:
    assert model_latency.NFL_REDUCER_LATENCY_LIMITATION == (
        "the NFL reducer-v3 latency path uses season-complete 2025 state "
        "semantics validated by the separate 285-game census; timed work "
        "remains one representative reducer transition and excludes census "
        "execution"
    )


def test_committed_static_latency_artifact_is_retired_by_dynamic_migration() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "game-state" / "model_latency_v0.json"
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert report["artifact_type"] == "sport_model_full_path_latency"
    assert report["result_label"] == "PRELIMINARY_ENGINEERING_BENCHMARK"
    assert report["live_sla_claimed"] is False
    assert any(
        "not state-conditioned" in limitation
        for limitation in report["limitations"]
    )
    with pytest.raises(LatencyBenchmarkError, match="version is invalid"):
        validate_model_latency_report(report, program_root=PROJECT_ROOT)


def test_latency_report_rejects_detached_accuracy_pointer_and_snapshot() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "game-state" / "model_latency_v0.json"
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    missing_pointer = deepcopy(report)
    benchmark = missing_pointer["benchmarks"][0]
    accuracy = benchmark["accuracy_reference"]
    accuracy["metric_pointer"] = "/does/not/exist"
    with pytest.raises(LatencyBenchmarkError, match="JSON Pointer"):
        model_latency._validate_accuracy_reference(
            accuracy,
            benchmark=benchmark,
            program_root=PROJECT_ROOT,
        )

    changed_snapshot = deepcopy(report)
    benchmark = changed_snapshot["benchmarks"][0]
    accuracy = benchmark["accuracy_reference"]
    accuracy["reported_metrics"]["observations"] += 1
    with pytest.raises(LatencyBenchmarkError, match="reported_metrics"):
        model_latency._validate_accuracy_reference(
            accuracy,
            benchmark=benchmark,
            program_root=PROJECT_ROOT,
        )

    changed_binding = deepcopy(report)
    benchmark = changed_binding["benchmarks"][0]
    accuracy = benchmark["accuracy_reference"]
    accuracy["aggregate_walk_forward_model_family_binding"][
        "parameter_config_sha256"
    ] = "sha256:" + "0" * 64
    with pytest.raises(LatencyBenchmarkError, match="aggregate"):
        model_latency._validate_accuracy_reference(
            accuracy,
            benchmark=benchmark,
            program_root=PROJECT_ROOT,
        )


def test_accuracy_reference_rejects_single_latency_fit_binding() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "game-state" / "model_latency_v0.json"
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    reintroduced = deepcopy(report)
    accuracy = reintroduced["benchmarks"][0]["accuracy_reference"]
    accuracy["aggregate_walk_forward_model_family_binding"][
        "model_parameter_sha256"
    ] = reintroduced["benchmarks"][0]["governance"][
        "model_parameter_sha256"
    ]

    with pytest.raises(LatencyBenchmarkError, match="fitted parameter"):
        model_latency._validate_accuracy_reference(
            accuracy,
            benchmark=reintroduced["benchmarks"][0],
            program_root=PROJECT_ROOT,
        )


def test_rehashing_legacy_parameter_forgery_cannot_bypass_retirement() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "game-state" / "model_latency_v0.json"
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    forged = deepcopy(report)
    forged_hash = "sha256:" + "0" * 64
    forged["benchmarks"][0]["governance"][
        "model_parameter_sha256"
    ] = forged_hash
    _rehash_report(forged)

    with pytest.raises(LatencyBenchmarkError, match="version is invalid"):
        validate_model_latency_report(forged, program_root=PROJECT_ROOT)
