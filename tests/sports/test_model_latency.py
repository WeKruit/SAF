from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_market.sports.model_latency import (
    LatencyBenchmarkError,
    benchmark_model_stages,
    measure_warm_inference,
    unavailable_sport_record,
    validate_model_latency_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    assert result["measurement"] == "single_observation_warm_inference"
    assert result["timer"] == "time.perf_counter_ns"
    assert result["warmup_iterations"] == 7
    assert result["timed_iterations"] == 1_001
    assert result["unit"] == "nanoseconds"
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
            "feature_binding": lambda: 1,
            "model_inference": lambda: 1,
            "output_validation": lambda: 1,
            "end_to_end": lambda: 1,
        },
        validators={name: valid for name in (
            "feature_binding",
            "model_inference",
            "output_validation",
            "end_to_end",
        )},
        warmup=1,
        repeats=1_000,
    )

    assert report["model_id"] == "MODEL-TEST"
    assert report["experiment_id"] == "X-11"
    assert list(report["stages"]) == [
        "feature_binding",
        "model_inference",
        "output_validation",
        "end_to_end",
    ]
    assert all(
        stage["timed_iterations"] == 1_000
        for stage in report["stages"].values()
    )

    with pytest.raises(LatencyBenchmarkError, match="stage inventory"):
        benchmark_model_stages(
            model_id="MODEL-TEST",
            experiment_id="X-11",
            stages={"end_to_end": lambda: 1},
            validators={"end_to_end": valid},
            warmup=1,
            repeats=1_000,
        )


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
    assert report["artifact_type"] == "model_inference_latency"
    assert report["result_label"] == "PRELIMINARY_ENGINEERING_BENCHMARK"
    assert report["live_sla_claimed"] is False
    assert report["protocol"]["minimum_timed_iterations"] >= 1_000
    assert {item["sport"] for item in report["unavailable_sports"]} == {
        "nba",
        "mlb",
        "f1",
    }
    assert {item["model_id"] for item in report["benchmarks"]} == {
        "MODEL-NFL-LOGISTIC",
        "MODEL-NFL-GBDT",
        "MODEL-NFL-DRIVE-TRANSITION",
        "MODEL-SOCCER-DIXON-COLES",
        "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
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
