"""Reproducible engineering latency benchmarks for fitted sports models.

Training is deliberately outside every timed region.  Accuracy is not
recomputed here; each benchmark points to an existing governed evidence
artifact instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prediction_market.sports import x11, x12


RESULT_LABEL = "PRELIMINARY_ENGINEERING_BENCHMARK"
REQUIRED_STAGES = (
    "feature_binding",
    "model_inference",
    "output_validation",
    "end_to_end",
)
MEASURED_MODELS = {
    "MODEL-NFL-LOGISTIC": ("X-11", "v0"),
    "MODEL-NFL-GBDT": ("X-11", "v0"),
    "MODEL-NFL-DRIVE-TRANSITION": ("X-11", "v1"),
    "MODEL-SOCCER-DIXON-COLES": ("X-12", "v0"),
    "MODEL-SOCCER-FIVE-MINUTE-TRANSITION": ("X-12", "v1"),
}
UNAVAILABLE_SPORTS = ("nba", "mlb", "f1")


class LatencyBenchmarkError(ValueError):
    """A benchmark input or result violates the frozen protocol."""


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise LatencyBenchmarkError(f"{field} must be a positive integer")
    return value


def _nearest_rank(sorted_samples: Sequence[int], percentile: float) -> int:
    index = max(0, math.ceil(percentile * len(sorted_samples)) - 1)
    return int(sorted_samples[index])


def measure_warm_inference(
    inference: Callable[[], object],
    *,
    validator: Callable[[object], bool],
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """Measure one already-fitted, single-observation callable.

    The validator runs after each invocation but outside its timed region.
    This makes invalid model output fail closed without mixing validation
    latency into the model-inference stage.
    """

    if not callable(inference) or not callable(validator):
        raise LatencyBenchmarkError("inference and validator must be callable")
    _require_positive_int(warmup, "warmup")
    _require_positive_int(repeats, "repeats")
    if repeats < 1_000:
        raise LatencyBenchmarkError("repeats must contain at least 1000 iterations")

    for _ in range(warmup):
        output = inference()
        if not bool(validator(output)):
            raise LatencyBenchmarkError("benchmark callable emitted invalid output")

    samples: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        output = inference()
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise LatencyBenchmarkError("perf_counter_ns moved backwards")
        if not bool(validator(output)):
            raise LatencyBenchmarkError("benchmark callable emitted invalid output")
        samples.append(elapsed)

    ordered = sorted(samples)
    return {
        "measurement": "single_observation_warm_inference",
        "timer": "time.perf_counter_ns",
        "unit": "nanoseconds",
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "p50_ns": _nearest_rank(ordered, 0.50),
        "p95_ns": _nearest_rank(ordered, 0.95),
        "p99_ns": _nearest_rank(ordered, 0.99),
        "max_ns": int(ordered[-1]),
    }


def benchmark_model_stages(
    *,
    model_id: str,
    experiment_id: str,
    stages: Mapping[str, Callable[[], object]],
    validators: Mapping[str, Callable[[object], bool]],
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """Benchmark the exact four stages in their governed order."""

    if tuple(stages) != REQUIRED_STAGES or tuple(validators) != REQUIRED_STAGES:
        raise LatencyBenchmarkError(
            "stage inventory must be feature_binding, model_inference, "
            "output_validation, end_to_end in that order"
        )
    if type(model_id) is not str or not model_id:
        raise LatencyBenchmarkError("model_id must be non-empty")
    if type(experiment_id) is not str or not experiment_id:
        raise LatencyBenchmarkError("experiment_id must be non-empty")
    return {
        "model_id": model_id,
        "experiment_id": experiment_id,
        "stages": {
            name: measure_warm_inference(
                stages[name],
                validator=validators[name],
                warmup=warmup,
                repeats=repeats,
            )
            for name in REQUIRED_STAGES
        },
    }


def unavailable_sport_record(sport: str) -> dict[str, object]:
    if sport not in UNAVAILABLE_SPORTS:
        raise LatencyBenchmarkError("unknown unavailable sport")
    return {
        "sport": sport,
        "status": "not_measured_no_eligible_model",
        "models": [],
        "latency": None,
    }


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _probability_vector_is_valid(
    value: object,
    *,
    size: int,
) -> bool:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        vector.shape == (size,)
        and bool(np.all(np.isfinite(vector)))
        and bool(np.all(vector >= 0))
        and bool(np.all(vector <= 1))
        and math.isclose(float(vector.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    )


def _binary_probability_is_valid(value: object) -> bool:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        vector.shape == (1,)
        and bool(np.all(np.isfinite(vector)))
        and bool(np.all(vector >= 0))
        and bool(np.all(vector <= 1))
    )


def _feature_vector_is_valid(value: object, *, width: int) -> bool:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return vector.shape == (1, width) and bool(np.all(np.isfinite(vector)))


def _validated_output(value: object, validator: Callable[[object], bool]) -> object:
    if not bool(validator(value)):
        raise LatencyBenchmarkError("model emitted invalid probabilities")
    return value


def _load_evidence_reference(
    *,
    program_root: Path,
    relative_path: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = program_root / relative_path
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise LatencyBenchmarkError(
            f"cannot read accuracy evidence {relative_path}"
        ) from error
    semantic_hash = document.get("evidence_sha256")
    if (
        type(semantic_hash) is not str
        or len(semantic_hash) != 71
        or not semantic_hash.startswith("sha256:")
    ):
        raise LatencyBenchmarkError("accuracy evidence has no governed hash")
    return document, {
        "evidence_path": relative_path,
        "evidence_sha256": semantic_hash,
        "evidence_file_sha256": _sha256_bytes(payload),
    }


def _metric_snapshot(metrics: Mapping[str, object]) -> dict[str, object]:
    selected: dict[str, object] = {}
    for name in (
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
        "observations",
    ):
        if name in metrics:
            selected[name] = metrics[name]
    return selected


def _accuracy_reference(
    common: Mapping[str, str],
    *,
    metric_pointer: str,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": "referenced_existing_evidence_not_recomputed",
        **common,
        "metric_pointer": metric_pointer,
        "reported_metrics": _metric_snapshot(metrics),
    }


def _nfl_benchmarks(
    *,
    program_root: Path,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    store_root = program_root / "var" / "raw"
    loaded = x11.load_x11_dataset(
        store_root=store_root,
        program_root=program_root,
    )
    frame = x11.attach_point_in_time_spread_prior(
        loaded.drive_starts,
        minimum_train_games=32,
    )
    eligible = frame.loc[
        frame["season"].between(2020, 2025) & frame["spread_prior"].notna()
    ]
    if eligible.empty:
        raise LatencyBenchmarkError("X-11 has no eligible real-data latency fold")
    cutoff = eligible["game_date"].max()
    train = frame.loc[
        (frame["game_date"] < cutoff) & frame["spread_prior"].notna()
    ].copy()
    test = eligible.loc[eligible["game_date"] == cutoff].copy()
    if train.empty or test.empty:
        raise LatencyBenchmarkError("X-11 last fold is incomplete")
    if train[list(x11.GAME_STATE_FEATURES)].isna().any().any():
        raise LatencyBenchmarkError("X-11 last-fold training features are incomplete")
    logistic, gbdt = x11._outcome_models(train, gbdt_max_iter=50)
    transition = x11._transition_model(train)
    observation = test.sort_values(
        ["game_id", "drive_number", "play_id"], kind="mergesort"
    ).iloc[0]
    raw_features = {
        name: float(observation[name]) for name in x11.GAME_STATE_FEATURES
    }

    def bind_features() -> np.ndarray:
        return np.asarray(
            [[raw_features[name] for name in x11.GAME_STATE_FEATURES]],
            dtype=float,
        )

    bound = bind_features()
    binary_validator = _binary_probability_is_valid
    transition_validator = lambda value: _probability_vector_is_valid(
        value,
        size=len(x11.TRANSITION_CLASSES),
    )
    logistic_output = x11._positive_class_probability(logistic, bound)
    gbdt_output = x11._positive_class_probability(gbdt, bound)
    transition_output = x11._fixed_transition_probabilities(transition, bound)[0]
    evidence, evidence_common = _load_evidence_reference(
        program_root=program_root,
        relative_path=(
            "artifacts/game-state/nfl/"
            "x11_real_data_pipeline_evidence_v0.json"
        ),
    )
    evidence_models = evidence["outcome_evaluation"]["models"]
    evidence_transition = evidence["transition_evaluation"]["metrics"]

    common_lineage = {
        "source_kind": "frozen_real_data_last_fold_fit",
        "dataset_id": x11.X11_DATASET_ID,
        "input_inventory_sha256": loaded.inventory.inventory_sha256,
        "training_cutoff_exclusive": pd.Timestamp(cutoff).isoformat(),
        "training_games": int(train["game_id"].nunique()),
        "training_observations": int(len(train)),
        "timed_observation": {
            "game_id": str(observation["game_id"]),
            "drive_number": int(observation["drive_number"]),
            "play_id": str(observation["play_id"]),
        },
        "feature_names": list(x11.GAME_STATE_FEATURES),
        "feature_vector_sha256": _sha256_json(raw_features),
        "training_time_included": False,
    }

    def binary_report(
        model_id: str,
        model: object,
        fixed_output: np.ndarray,
        evidence_key: str,
    ) -> dict[str, object]:
        infer = lambda: x11._positive_class_probability(model, bound)
        end_to_end = lambda: x11._positive_class_probability(
            model, bind_features()
        )
        report = benchmark_model_stages(
            model_id=model_id,
            experiment_id=x11.X11_EXPERIMENT_ID,
            stages={
                "feature_binding": bind_features,
                "model_inference": infer,
                "output_validation": lambda: _validated_output(
                    fixed_output, binary_validator
                ),
                "end_to_end": end_to_end,
            },
            validators={
                "feature_binding": lambda value: _feature_vector_is_valid(
                    value, width=len(x11.GAME_STATE_FEATURES)
                ),
                "model_inference": binary_validator,
                "output_validation": binary_validator,
                "end_to_end": binary_validator,
            },
            warmup=warmup,
            repeats=repeats,
        )
        report.update(
            {
                "model_version": MEASURED_MODELS[model_id][1],
                "sport": "nfl",
                "lineage": common_lineage,
                "accuracy_reference": _accuracy_reference(
                    evidence_common,
                    metric_pointer=f"/outcome_evaluation/models/{evidence_key}",
                    metrics=evidence_models[evidence_key],
                ),
            }
        )
        return report

    transition_infer = lambda: x11._fixed_transition_probabilities(
        transition, bound
    )[0]
    transition_end_to_end = lambda: x11._fixed_transition_probabilities(
        transition, bind_features()
    )[0]
    transition_report = benchmark_model_stages(
        model_id="MODEL-NFL-DRIVE-TRANSITION",
        experiment_id=x11.X11_EXPERIMENT_ID,
        stages={
            "feature_binding": bind_features,
            "model_inference": transition_infer,
            "output_validation": lambda: _validated_output(
                transition_output, transition_validator
            ),
            "end_to_end": transition_end_to_end,
        },
        validators={
            "feature_binding": lambda value: _feature_vector_is_valid(
                value, width=len(x11.GAME_STATE_FEATURES)
            ),
            "model_inference": transition_validator,
            "output_validation": transition_validator,
            "end_to_end": transition_validator,
        },
        warmup=warmup,
        repeats=repeats,
    )
    transition_report.update(
        {
            "model_version": "v1",
            "sport": "nfl",
            "lineage": common_lineage,
            "accuracy_reference": _accuracy_reference(
                evidence_common,
                metric_pointer="/transition_evaluation/metrics",
                metrics=evidence_transition,
            ),
        }
    )
    return [
        binary_report(
            "MODEL-NFL-LOGISTIC", logistic, logistic_output, "logistic"
        ),
        binary_report("MODEL-NFL-GBDT", gbdt, gbdt_output, "gbdt"),
        transition_report,
    ]


def _soccer_benchmarks(
    *,
    program_root: Path,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    store_root = program_root / "var" / "raw"
    loaded = x12.load_x12_dataset(
        store_root=store_root,
        program_root=program_root,
    )
    matches = loaded.matches.sort_values(
        ["played_at", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    cutoff = matches["match_date"].max()
    train = matches.loc[matches["match_date"] < cutoff].copy()
    test = matches.loc[matches["match_date"] == cutoff].copy()
    if train.empty or test.empty:
        raise LatencyBenchmarkError("X-12 last fold is incomplete")
    team_ids = tuple(
        sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    )
    model = x12._fit_dixon_coles(
        train,
        team_ids=team_ids,
        optimizer_max_iterations=250,
        initial_parameters=None,
    )
    observation = test.sort_values(["played_at", "match_id"], kind="mergesort").iloc[
        0
    ]
    raw_features = {
        "home_team_id": int(observation["home_team_id"]),
        "away_team_id": int(observation["away_team_id"]),
    }

    def bind_features() -> tuple[int, int]:
        return raw_features["home_team_id"], raw_features["away_team_id"]

    bound = bind_features()

    def outcome_inference() -> np.ndarray:
        return x12._outcome_probabilities(
            model,
            home_team_id=bound[0],
            away_team_id=bound[1],
            goal_grid_max=10,
        )[0]

    def outcome_end_to_end() -> np.ndarray:
        home_team_id, away_team_id = bind_features()
        return x12._outcome_probabilities(
            model,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            goal_grid_max=10,
        )[0]

    outcome_output = x12._outcome_probabilities(
        model,
        home_team_id=bound[0],
        away_team_id=bound[1],
        goal_grid_max=10,
    )[0]
    home_rate, away_rate = x12._expected_goals(
        model,
        home_team_id=bound[0],
        away_team_id=bound[1],
    )

    def transition_inference() -> np.ndarray:
        rates = x12._expected_goals(
            model,
            home_team_id=bound[0],
            away_team_id=bound[1],
        )
        return x12._competing_goal_probabilities(*rates)

    def transition_end_to_end() -> np.ndarray:
        home_team_id, away_team_id = bind_features()
        rates = x12._expected_goals(
            model,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        return x12._competing_goal_probabilities(*rates)

    transition_output = x12._competing_goal_probabilities(home_rate, away_rate)
    feature_validator = lambda value: (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is int and item in team_ids for item in value)
        and value[0] != value[1]
    )
    outcome_validator = lambda value: _probability_vector_is_valid(value, size=3)
    transition_validator = lambda value: _probability_vector_is_valid(
        value, size=3
    )
    evidence, evidence_common = _load_evidence_reference(
        program_root=program_root,
        relative_path=(
            "artifacts/game-state/soccer/x12_real_data_poc_v0.json"
        ),
    )
    common_lineage = {
        "source_kind": "frozen_real_data_last_fold_fit",
        "dataset_id": x12.X12_DATASET_ID,
        "input_inventory_sha256": loaded.inventory.inventory_sha256,
        "training_cutoff_exclusive": pd.Timestamp(cutoff).isoformat(),
        "training_matches": int(len(train)),
        "timed_observation": {
            "match_id": int(observation["match_id"]),
            **raw_features,
        },
        "feature_names": list(raw_features),
        "feature_vector_sha256": _sha256_json(raw_features),
        "training_time_included": False,
    }

    outcome_report = benchmark_model_stages(
        model_id="MODEL-SOCCER-DIXON-COLES",
        experiment_id=x12.X12_EXPERIMENT_ID,
        stages={
            "feature_binding": bind_features,
            "model_inference": outcome_inference,
            "output_validation": lambda: _validated_output(
                outcome_output, outcome_validator
            ),
            "end_to_end": outcome_end_to_end,
        },
        validators={
            "feature_binding": feature_validator,
            "model_inference": outcome_validator,
            "output_validation": outcome_validator,
            "end_to_end": outcome_validator,
        },
        warmup=warmup,
        repeats=repeats,
    )
    outcome_report.update(
        {
            "model_version": "v0",
            "sport": "soccer",
            "lineage": common_lineage,
            "accuracy_reference": _accuracy_reference(
                evidence_common,
                metric_pointer="/outcome_evaluation/metrics",
                metrics=evidence["outcome_evaluation"]["metrics"],
            ),
        }
    )

    transition_report = benchmark_model_stages(
        model_id="MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
        experiment_id=x12.X12_EXPERIMENT_ID,
        stages={
            "feature_binding": bind_features,
            "model_inference": transition_inference,
            "output_validation": lambda: _validated_output(
                transition_output, transition_validator
            ),
            "end_to_end": transition_end_to_end,
        },
        validators={
            "feature_binding": feature_validator,
            "model_inference": transition_validator,
            "output_validation": transition_validator,
            "end_to_end": transition_validator,
        },
        warmup=warmup,
        repeats=repeats,
    )
    transition_report.update(
        {
            "model_version": "v1",
            "sport": "soccer",
            "lineage": common_lineage,
            "accuracy_reference": _accuracy_reference(
                evidence_common,
                metric_pointer="/transition_output/metrics",
                metrics=evidence["transition_output"]["metrics"],
            ),
        }
    )
    return [outcome_report, transition_report]


def _runtime_record() -> dict[str, object]:
    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not_installed"

    clock = time.get_clock_info("perf_counter")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not_reported_by_runtime",
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "scipy": version("scipy"),
        },
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "clock": {
            "name": "perf_counter",
            "implementation": clock.implementation,
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
            "resolution_ns": int(round(clock.resolution * 1_000_000_000)),
        },
    }


def build_current_model_latency_report(
    *,
    program_root: str | Path,
    warmup: int = 100,
    repeats: int = 2_000,
) -> dict[str, object]:
    """Fit the last frozen real-data folds and benchmark warm inference."""

    root = Path(program_root).resolve()
    _require_positive_int(warmup, "warmup")
    _require_positive_int(repeats, "repeats")
    if repeats < 1_000:
        raise LatencyBenchmarkError("repeats must contain at least 1000 iterations")
    benchmarks = [
        *_nfl_benchmarks(program_root=root, warmup=warmup, repeats=repeats),
        *_soccer_benchmarks(program_root=root, warmup=warmup, repeats=repeats),
    ]
    report: dict[str, object] = {
        "artifact_type": "model_inference_latency",
        "version": "v0",
        "result_label": RESULT_LABEL,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "live_sla_claimed": False,
        "scope": (
            "warm single-observation inference on fitted last-fold models; "
            "training, data loading, network, and market joins are excluded"
        ),
        "protocol": {
            "timer": "time.perf_counter_ns",
            "percentile_method": "nearest_rank",
            "warmup_iterations": warmup,
            "timed_iterations": repeats,
            "minimum_timed_iterations": 1_000,
            "stages": list(REQUIRED_STAGES),
            "training_timed": False,
            "single_observation": True,
        },
        "runtime": _runtime_record(),
        "benchmarks": benchmarks,
        "unavailable_sports": [
            unavailable_sport_record(sport) for sport in UNAVAILABLE_SPORTS
        ],
        "limitations": [
            "engineering benchmark only; no live SLA or production claim",
            "accuracy is referenced from existing evidence and is not recomputed",
            "wall-clock results are machine- and load-specific",
            "prediction-market transport, joins, and execution are out of scope",
        ],
    }
    report["report_sha256"] = _sha256_json(report)
    return validate_model_latency_report(report, program_root=root)


def _validate_digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise LatencyBenchmarkError(f"{field} must be a SHA-256 digest")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise LatencyBenchmarkError(f"{field} must be a SHA-256 digest") from error
    return value


def validate_model_latency_report(
    report: Mapping[str, object],
    *,
    program_root: str | Path,
) -> dict[str, object]:
    """Fail closed on incomplete, placeholder, or detached latency evidence."""

    if not isinstance(report, Mapping):
        raise LatencyBenchmarkError("latency report must be an object")
    document = dict(report)
    if document.get("artifact_type") != "model_inference_latency":
        raise LatencyBenchmarkError("latency report artifact_type is invalid")
    if document.get("version") != "v0":
        raise LatencyBenchmarkError("latency report version is invalid")
    if document.get("result_label") != RESULT_LABEL:
        raise LatencyBenchmarkError("latency report result label is invalid")
    if document.get("live_sla_claimed") is not False:
        raise LatencyBenchmarkError("latency report cannot claim a live SLA")
    protocol = document.get("protocol")
    if not isinstance(protocol, Mapping):
        raise LatencyBenchmarkError("latency report protocol is missing")
    if (
        protocol.get("timer") != "time.perf_counter_ns"
        or protocol.get("percentile_method") != "nearest_rank"
        or protocol.get("training_timed") is not False
        or protocol.get("single_observation") is not True
        or tuple(protocol.get("stages", ())) != REQUIRED_STAGES
    ):
        raise LatencyBenchmarkError("latency report protocol changed")
    minimum = protocol.get("minimum_timed_iterations")
    if type(minimum) is not int or minimum < 1_000:
        raise LatencyBenchmarkError("latency report permits too few iterations")

    benchmarks = document.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise LatencyBenchmarkError("latency report benchmarks are missing")
    if {item.get("model_id") for item in benchmarks if isinstance(item, dict)} != set(
        MEASURED_MODELS
    ):
        raise LatencyBenchmarkError("latency report model inventory changed")
    root = Path(program_root).resolve()
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            raise LatencyBenchmarkError("latency benchmark must be an object")
        model_id = benchmark.get("model_id")
        expected_experiment, expected_version = MEASURED_MODELS[str(model_id)]
        if (
            benchmark.get("experiment_id") != expected_experiment
            or benchmark.get("model_version") != expected_version
        ):
            raise LatencyBenchmarkError("latency benchmark registry binding changed")
        stages = benchmark.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != set(REQUIRED_STAGES):
            raise LatencyBenchmarkError("latency benchmark stage inventory changed")
        for stage in stages.values():
            if not isinstance(stage, Mapping):
                raise LatencyBenchmarkError("latency stage must be an object")
            if stage.get("timer") != "time.perf_counter_ns":
                raise LatencyBenchmarkError("latency stage timer changed")
            count = stage.get("timed_iterations")
            values = [
                stage.get(name)
                for name in ("p50_ns", "p95_ns", "p99_ns", "max_ns")
            ]
            if (
                type(count) is not int
                or count < minimum
                or any(type(value) is not int or value < 0 for value in values)
                or values != sorted(values)
            ):
                raise LatencyBenchmarkError("latency stage statistics are invalid")
        accuracy = benchmark.get("accuracy_reference")
        if (
            not isinstance(accuracy, Mapping)
            or accuracy.get("status")
            != "referenced_existing_evidence_not_recomputed"
        ):
            raise LatencyBenchmarkError("accuracy reference is invalid")
        relative = accuracy.get("evidence_path")
        if type(relative) is not str or Path(relative).is_absolute():
            raise LatencyBenchmarkError("accuracy evidence path must be relative")
        evidence_path = (root / relative).resolve()
        if root not in evidence_path.parents:
            raise LatencyBenchmarkError("accuracy evidence escapes program root")
        try:
            payload = evidence_path.read_bytes()
            evidence = json.loads(payload)
        except (OSError, json.JSONDecodeError) as error:
            raise LatencyBenchmarkError("accuracy evidence cannot be verified") from error
        if (
            _validate_digest(
                accuracy.get("evidence_file_sha256"), "evidence_file_sha256"
            )
            != _sha256_bytes(payload)
            or _validate_digest(
                accuracy.get("evidence_sha256"), "evidence_sha256"
            )
            != evidence.get("evidence_sha256")
        ):
            raise LatencyBenchmarkError("accuracy evidence hash changed")

    unavailable = document.get("unavailable_sports")
    expected_unavailable = [
        unavailable_sport_record(sport) for sport in UNAVAILABLE_SPORTS
    ]
    if unavailable != expected_unavailable:
        raise LatencyBenchmarkError(
            "unavailable sports must be explicit and contain no placeholder latency"
        )
    recorded_hash = _validate_digest(document.get("report_sha256"), "report_sha256")
    unhashed = dict(document)
    del unhashed["report_sha256"]
    if recorded_hash != _sha256_json(unhashed):
        raise LatencyBenchmarkError("latency report hash is invalid")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2_000)
    args = parser.parse_args(argv)
    report = build_current_model_latency_report(
        program_root=args.program_root,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LatencyBenchmarkError",
    "benchmark_model_stages",
    "build_current_model_latency_report",
    "measure_warm_inference",
    "unavailable_sport_record",
    "validate_model_latency_report",
]
