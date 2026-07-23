from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

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


def test_committed_latency_artifact_is_governed_and_complete() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "game-state" / "model_latency_v0.json"
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    validated = validate_model_latency_report(report, program_root=PROJECT_ROOT)

    assert validated == report
    assert report["artifact_type"] == "sport_model_full_path_latency"
    assert report["result_label"] == "PRELIMINARY_ENGINEERING_BENCHMARK"
    assert report["live_sla_claimed"] is False
    assert report["protocol"]["minimum_timed_iterations"] >= 1_000
    assert report["protocol"]["stages"] == [
        "state_reducer",
        "feature_extraction",
        "model_inference",
        "output_validation",
        "full_path",
    ]
    assert report["protocol"]["included_in_full_path"] == [
        "state_reducer",
        "feature_extraction",
        "registered_model_inference",
        "model_output_materialization",
        "strict_model_output_v1_validation_against_preloaded_registry",
    ]
    assert report["protocol"]["excluded_from_all_timed_regions"] == [
        "model_training",
        "raw_source_loading",
        "registry_snapshot_loading",
        "network_io",
    ]
    assert {item["sport"] for item in report["unavailable_sports"]} == {
        "nba",
        "mlb",
        "f1",
    }
    assert {item["model_id"] for item in report["benchmarks"]} == {
        "MODEL-NFL-DRIVE-TRANSITION",
        "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
    }
    assert {item["model_id"] for item in report["unavailable_models"]} == {
        "MODEL-NFL-LOGISTIC",
        "MODEL-NFL-GBDT",
        "MODEL-SOCCER-DIXON-COLES",
    }
    assert all(
        benchmark["accuracy_reference"]["status"]
        == "referenced_existing_evidence_not_recomputed"
        for benchmark in report["benchmarks"]
    )
    assert all(
        {
            "brier",
            "log_loss",
            "observations",
        }.issubset(benchmark["accuracy_reference"]["reported_metrics"])
        for benchmark in report["benchmarks"]
    )
    for benchmark in report["benchmarks"]:
        assert set(benchmark["stages"]) == set(report["protocol"]["stages"])
        assert benchmark["measured_events"] >= 1_000
        assert benchmark["full_path_events_per_second"] > 0
        governance = benchmark["governance"]
        assert set(governance["code_sha256"]) == {
            "latency_harness",
            "sport_reducer",
            "model_implementation",
        }
        for digest in (
            *governance["code_sha256"].values(),
            governance["data_manifest_sha256"],
            governance["training_manifest_sha256"],
            governance["parameter_config_sha256"],
            governance["model_parameter_sha256"],
            governance["model_registry_row_sha256"],
        ):
            assert digest.startswith("sha256:") and len(digest) == 71
        parameter_snapshot_payload = json.dumps(
            governance["model_parameter_snapshot"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        assert governance["model_parameter_sha256"] == (
            "sha256:"
            + hashlib.sha256(parameter_snapshot_payload).hexdigest()
        )
        lineage = benchmark["representative_input_lineage"]
        assert lineage["previous_state_sha256"].startswith("sha256:")
        assert lineage["event_sha256"].startswith("sha256:")
        assert lineage["next_state_sha256"].startswith("sha256:")
        assert lineage["source_object_sha256"].startswith("sha256:")

        accuracy = benchmark["accuracy_reference"]
        evidence = json.loads(
            (PROJECT_ROOT / accuracy["evidence_path"]).read_text(encoding="utf-8")
        )
        assert accuracy["reported_metrics"] == resolve_json_pointer(
            evidence,
            accuracy["metric_pointer"],
        )
        assert accuracy["metric_snapshot_sha256"].startswith("sha256:")
        assert (
            accuracy["scope"]
            == "aggregate_walk_forward_model_family_evidence"
        )
        assert "model_binding" not in accuracy
        assert (
            "model_parameter_sha256"
            not in accuracy["aggregate_walk_forward_model_family_binding"]
        )
        assert accuracy[
            "aggregate_walk_forward_model_family_binding"
        ] == {
            "model_id": benchmark["model_id"],
            "model_version": benchmark["model_version"],
            "data_manifest_sha256": governance["data_manifest_sha256"],
            "parameter_config_sha256": governance["parameter_config_sha256"],
            "evidence_sha256": accuracy["evidence_sha256"],
            "evidence_file_sha256": accuracy["evidence_file_sha256"],
            "metric_snapshot_sha256": accuracy["metric_snapshot_sha256"],
        }


def test_latency_report_rejects_detached_accuracy_pointer_and_snapshot() -> None:
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "game-state" / "model_latency_v0.json"
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    missing_pointer = deepcopy(report)
    missing_pointer["benchmarks"][0]["accuracy_reference"]["metric_pointer"] = (
        "/does/not/exist"
    )
    with pytest.raises(LatencyBenchmarkError, match="JSON Pointer"):
        validate_model_latency_report(missing_pointer, program_root=PROJECT_ROOT)

    changed_snapshot = deepcopy(report)
    changed_snapshot["benchmarks"][0]["accuracy_reference"]["reported_metrics"][
        "observations"
    ] += 1
    with pytest.raises(LatencyBenchmarkError, match="reported_metrics"):
        validate_model_latency_report(changed_snapshot, program_root=PROJECT_ROOT)

    changed_binding = deepcopy(report)
    changed_binding["benchmarks"][0]["accuracy_reference"][
        "aggregate_walk_forward_model_family_binding"
    ]["parameter_config_sha256"] = "sha256:" + "0" * 64
    _rehash_report(changed_binding)
    with pytest.raises(LatencyBenchmarkError, match="aggregate"):
        validate_model_latency_report(changed_binding, program_root=PROJECT_ROOT)


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
    _rehash_report(reintroduced)

    with pytest.raises(LatencyBenchmarkError, match="fitted parameter"):
        validate_model_latency_report(reintroduced, program_root=PROJECT_ROOT)


def test_latency_report_rejects_synchronized_parameter_hash_forgery() -> None:
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

    with pytest.raises(LatencyBenchmarkError, match="parameter snapshot"):
        validate_model_latency_report(forged, program_root=PROJECT_ROOT)
