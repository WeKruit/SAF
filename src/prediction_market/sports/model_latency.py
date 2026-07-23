"""Governed state/event transition latency benchmarks.

Only models that can execute the complete registered transition path are
measured.  Training, raw-source loading, registry loading, and network I/O are
completed before timing.  A timed path materializes ``ModelOutputV1`` only when
the representative source has a proven point-in-time cutoff; otherwise it
measures reducer, features, inference, and probability-distribution validation
while reporting a null contract output.
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
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prediction_market import contracts
from prediction_market.program_audit import ModelRegistryRow, load_model_registry
from prediction_market.sports import (
    nfl_game_state,
    soccer_game_state,
    soccer_transition_model,
    x11,
    x12,
)
from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_bundle,
)
from prediction_market.sports.game_state import canonical_state_sha256


RESULT_LABEL = "PRELIMINARY_ENGINEERING_BENCHMARK"
ARTIFACT_TYPE = "sport_model_full_path_latency"
REPORT_VERSION = "v1"
DYNAMIC_SOCCER_MODEL_ID = "MODEL-SOCCER-DYNAMIC-INTENSITY"
DYNAMIC_SOCCER_EVIDENCE_PATH = (
    "artifacts/game-state/soccer/x12_dynamic_transition_poc_v1.json"
)
ACCURACY_EVIDENCE_SCOPE = "aggregate_walk_forward_model_family_evidence"
NFL_REDUCER_LATENCY_LIMITATION = (
    "the NFL reducer-v3 latency path uses season-complete 2025 state "
    "semantics validated by the separate 285-game census; timed work "
    "remains one representative reducer transition and excludes census "
    "execution"
)
REQUIRED_STAGES = (
    "state_reducer",
    "feature_extraction",
    "model_inference",
    "output_validation",
    "full_path",
)
MEASURED_MODELS = {
    "MODEL-NFL-DRIVE-TRANSITION": ("X-11", "v1"),
    DYNAMIC_SOCCER_MODEL_ID: ("X-12", "v1"),
}
UNAVAILABLE_MODEL_REASONS = {
    "MODEL-NFL-LOGISTIC": (
        "nfl",
        "registered game-end model has no state-transition ModelOutputV1 contract",
    ),
    "MODEL-NFL-GBDT": (
        "nfl",
        "registered game-end model has no state-transition ModelOutputV1 contract",
    ),
    "MODEL-SOCCER-DIXON-COLES": (
        "soccer",
        "registered game-end model has no state-transition ModelOutputV1 contract",
    ),
}
UNAVAILABLE_SPORTS = ("nba", "mlb", "f1")
_HASH_PREFIX = "sha256:"


class LatencyBenchmarkError(ValueError):
    """A benchmark input or result violates the frozen protocol."""


def _json_pointer_token(token: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise LatencyBenchmarkError("JSON Pointer contains an invalid escape")
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def resolve_json_pointer(document: object, pointer: object) -> object:
    """Resolve one RFC 6901 JSON Pointer without permissive fallbacks."""

    if type(pointer) is not str or (pointer and not pointer.startswith("/")):
        raise LatencyBenchmarkError("JSON Pointer must be empty or start with /")
    current = document
    if pointer == "":
        return current
    for raw_token in pointer[1:].split("/"):
        token = _json_pointer_token(raw_token)
        if isinstance(current, dict):
            if token not in current:
                raise LatencyBenchmarkError(
                    f"JSON Pointer does not exist at token {token!r}"
                )
            current = current[token]
            continue
        if isinstance(current, list):
            if not token.isdigit() or (
                len(token) > 1 and token.startswith("0")
            ):
                raise LatencyBenchmarkError(
                    f"JSON Pointer array token is invalid: {token!r}"
                )
            item_index = int(token)
            if item_index >= len(current):
                raise LatencyBenchmarkError(
                    f"JSON Pointer array index is out of range: {token!r}"
                )
            current = current[item_index]
            continue
        raise LatencyBenchmarkError(
            f"JSON Pointer traverses a scalar at token {token!r}"
        )
    return current


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise LatencyBenchmarkError(f"{field} must be a positive integer")
    return value


def _nearest_rank(sorted_samples: Sequence[int], percentile: float) -> int:
    index = max(0, math.ceil(percentile * len(sorted_samples)) - 1)
    return int(sorted_samples[index])


def measure_warm_inference(
    operation: Callable[[], object],
    *,
    validator: Callable[[object], bool],
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """Measure one already-prepared, single-observation stage callable."""

    if not callable(operation) or not callable(validator):
        raise LatencyBenchmarkError("operation and validator must be callable")
    _require_positive_int(warmup, "warmup")
    _require_positive_int(repeats, "repeats")
    if repeats < 1_000:
        raise LatencyBenchmarkError("repeats must contain at least 1000 iterations")

    for _ in range(warmup):
        output = operation()
        if not bool(validator(output)):
            raise LatencyBenchmarkError("benchmark callable emitted invalid output")

    samples: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        output = operation()
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise LatencyBenchmarkError("perf_counter_ns moved backwards")
        if not bool(validator(output)):
            raise LatencyBenchmarkError("benchmark callable emitted invalid output")
        samples.append(elapsed)

    ordered = sorted(samples)
    total_ns = sum(samples)
    if total_ns <= 0:
        raise LatencyBenchmarkError("timed stage has a non-positive duration")
    return {
        "measurement": "single_observation_warm_stage",
        "timer": "time.perf_counter_ns",
        "unit": "nanoseconds",
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "measured_operations": repeats,
        "total_ns": total_ns,
        "operations_per_second": repeats * 1_000_000_000 // total_ns,
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
    """Benchmark the five required stages in their governed order."""

    if tuple(stages) != REQUIRED_STAGES or tuple(validators) != REQUIRED_STAGES:
        raise LatencyBenchmarkError(
            "stage inventory must be state_reducer, feature_extraction, "
            "model_inference, output_validation, full_path in that order"
        )
    if type(model_id) is not str or not model_id:
        raise LatencyBenchmarkError("model_id must be non-empty")
    if type(experiment_id) is not str or not experiment_id:
        raise LatencyBenchmarkError("experiment_id must be non-empty")
    measured = {
        name: measure_warm_inference(
            stages[name],
            validator=validators[name],
            warmup=warmup,
            repeats=repeats,
        )
        for name in REQUIRED_STAGES
    }
    return {
        "model_id": model_id,
        "experiment_id": experiment_id,
        "measured_events": repeats,
        "full_path_events_per_second": measured["full_path"][
            "operations_per_second"
        ],
        "stages": measured,
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


def _unavailable_model_records() -> list[dict[str, str]]:
    return [
        {
            "model_id": model_id,
            "sport": sport,
            "status": "not_measured_no_model_output_v1_full_path",
            "reason": reason,
        }
        for model_id, (sport, reason) in UNAVAILABLE_MODEL_REASONS.items()
    ]


def _sha256_bytes(payload: bytes) -> str:
    return _HASH_PREFIX + hashlib.sha256(payload).hexdigest()


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


def _validate_digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_HASH_PREFIX)
    ):
        raise LatencyBenchmarkError(f"{field} must be a SHA-256 digest")
    try:
        int(value.removeprefix(_HASH_PREFIX), 16)
    except ValueError as error:
        raise LatencyBenchmarkError(f"{field} must be a SHA-256 digest") from error
    return value


def _load_verified_evidence(
    *,
    program_root: Path,
    relative_path: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = (program_root / relative_path).resolve()
    if program_root not in path.parents:
        raise LatencyBenchmarkError("accuracy evidence escapes program root")
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise LatencyBenchmarkError(
            f"cannot read accuracy evidence {relative_path}"
        ) from error
    if not isinstance(document, dict):
        raise LatencyBenchmarkError("accuracy evidence must be a JSON object")
    semantic_hash = _validate_digest(
        document.get("evidence_sha256"),
        "accuracy evidence evidence_sha256",
    )
    unhashed = dict(document)
    del unhashed["evidence_sha256"]
    if semantic_hash != _sha256_json(unhashed):
        raise LatencyBenchmarkError("accuracy evidence semantic hash is invalid")
    return document, {
        "evidence_path": relative_path,
        "evidence_sha256": semantic_hash,
        "evidence_file_sha256": _sha256_bytes(payload),
    }


def _accuracy_reference(
    evidence_common: Mapping[str, str],
    *,
    evidence: Mapping[str, object],
    metric_pointer: str,
    row: ModelRegistryRow,
    governance: Mapping[str, object],
) -> dict[str, object]:
    snapshot = resolve_json_pointer(evidence, metric_pointer)
    if not isinstance(snapshot, dict):
        raise LatencyBenchmarkError(
            "accuracy JSON Pointer must resolve to a canonical metrics object"
        )
    metric_snapshot_sha256 = _sha256_json(snapshot)
    return {
        "status": "referenced_existing_evidence_not_recomputed",
        "scope": ACCURACY_EVIDENCE_SCOPE,
        **evidence_common,
        "metric_pointer": metric_pointer,
        "metric_snapshot_sha256": metric_snapshot_sha256,
        "reported_metrics": snapshot,
        "aggregate_walk_forward_model_family_binding": {
            "model_id": row.model_id,
            "model_version": row.model_version,
            "data_manifest_sha256": str(
                governance["data_manifest_sha256"]
            ),
            "parameter_config_sha256": str(
                governance["parameter_config_sha256"]
            ),
            "evidence_sha256": evidence_common["evidence_sha256"],
            "evidence_file_sha256": evidence_common["evidence_file_sha256"],
            "metric_snapshot_sha256": metric_snapshot_sha256,
        },
    }


def _registry_rows(program_root: Path) -> dict[str, ModelRegistryRow]:
    rows = {row.model_id: row for row in load_model_registry(program_root)}
    missing = set(MEASURED_MODELS) - set(rows)
    if missing:
        raise LatencyBenchmarkError(
            f"measured models are absent from registry: {sorted(missing)}"
        )
    return rows


def _registry_row_sha256(row: ModelRegistryRow) -> str:
    return _sha256_json(asdict(row))


def _source_code_hashes(
    *,
    reducer_path: Path,
    model_paths: tuple[Path, ...],
) -> dict[str, str]:
    if not model_paths:
        raise LatencyBenchmarkError(
            "benchmark model implementation inventory is empty"
        )
    try:
        model_hashes = {
            path.name: _sha256_bytes(path.read_bytes())
            for path in model_paths
        }
        return {
            "latency_harness": _sha256_bytes(Path(__file__).read_bytes()),
            "sport_reducer": _sha256_bytes(reducer_path.read_bytes()),
            "model_implementation": _sha256_json(model_hashes),
        }
    except OSError as error:
        raise LatencyBenchmarkError("benchmark source code cannot be hashed") from error


def _json_model_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_model_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_model_value(value.item())
    if isinstance(value, tuple):
        return [_json_model_value(item) for item in value]
    if isinstance(value, list):
        return [_json_model_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_model_value(item) for key, item in value.items()
        }
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise LatencyBenchmarkError(
        f"model parameters contain unsupported {type(value).__name__}"
    )


def _nfl_model_parameter_snapshot(model: object) -> dict[str, object]:
    try:
        scaler = model.named_steps["scale"]  # type: ignore[attr-defined]
        classifier = model.named_steps["model"]  # type: ignore[attr-defined]
        material = {
            "snapshot_version": "v0",
            "model_family": (
                "sklearn_standard_scaler_multinomial_logistic_regression"
            ),
            "pipeline_steps": ["scale", "model"],
            "scaler": {
                "mean": scaler.mean_,
                "scale": scaler.scale_,
                "var": scaler.var_,
                "n_samples_seen": scaler.n_samples_seen_,
            },
            "classifier": {
                "classes": classifier.classes_,
                "coef": classifier.coef_,
                "intercept": classifier.intercept_,
                "n_iter": classifier.n_iter_,
            },
        }
    except (AttributeError, KeyError) as error:
        raise LatencyBenchmarkError(
            "NFL transition model has no fitted parameter snapshot"
        ) from error
    snapshot = _json_model_value(material)
    if not isinstance(snapshot, dict):
        raise LatencyBenchmarkError("NFL parameter snapshot is not canonical")
    return snapshot


def _soccer_model_parameter_snapshot(
    base_rate_model: x12.DixonColesModel,
    dynamic_model: soccer_transition_model.DynamicIntensityModel,
    temperature_calibration: soccer_transition_model.TemperatureCalibration,
) -> dict[str, object]:
    if (
        not isinstance(base_rate_model, x12.DixonColesModel)
        or not isinstance(
            dynamic_model,
            soccer_transition_model.DynamicIntensityModel,
        )
        or not isinstance(
            temperature_calibration,
            soccer_transition_model.TemperatureCalibration,
        )
    ):
        raise LatencyBenchmarkError(
            "soccer transition model has no fitted parameter snapshot"
        )
    snapshot = _json_model_value(
        {
            "snapshot_version": "v1",
            "model_family": (
                "dixon_coles_base_rate_plus_symmetric_dynamic_intensity_"
                "plus_multiclass_temperature"
            ),
            "base_rate_model": asdict(base_rate_model),
            "dynamic_intensity_model": asdict(dynamic_model),
            "temperature_calibration": asdict(temperature_calibration),
        }
    )
    if not isinstance(snapshot, dict):
        raise LatencyBenchmarkError("soccer parameter snapshot is not canonical")
    return snapshot


def _finite_parameter_number(value: object, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise LatencyBenchmarkError(
            f"model parameter snapshot {field} must be finite"
        )
    return float(value)


def _validate_model_parameter_snapshot(
    model_id: str,
    snapshot: object,
) -> dict[str, object]:
    """Validate the canonical fitted payload hashed by the benchmark."""

    if not isinstance(snapshot, dict):
        raise LatencyBenchmarkError("model parameter snapshot must be an object")
    if model_id == "MODEL-NFL-DRIVE-TRANSITION":
        if set(snapshot) != {
            "snapshot_version",
            "model_family",
            "pipeline_steps",
            "scaler",
            "classifier",
        } or (
            snapshot.get("snapshot_version") != "v0"
            or snapshot.get("model_family")
            != "sklearn_standard_scaler_multinomial_logistic_regression"
            or snapshot.get("pipeline_steps") != ["scale", "model"]
        ):
            raise LatencyBenchmarkError(
                "NFL model parameter snapshot structure is invalid"
            )
        scaler = snapshot.get("scaler")
        classifier = snapshot.get("classifier")
        if not isinstance(scaler, dict) or set(scaler) != {
            "mean",
            "scale",
            "var",
            "n_samples_seen",
        }:
            raise LatencyBenchmarkError(
                "NFL scaler parameter snapshot is invalid"
            )
        if not isinstance(classifier, dict) or set(classifier) != {
            "classes",
            "coef",
            "intercept",
            "n_iter",
        }:
            raise LatencyBenchmarkError(
                "NFL classifier parameter snapshot is invalid"
            )
        width = len(x11.GAME_STATE_FEATURES)
        for name in ("mean", "scale", "var"):
            values = scaler.get(name)
            if not isinstance(values, list) or len(values) != width:
                raise LatencyBenchmarkError(
                    f"NFL scaler {name} parameter snapshot is invalid"
                )
            numbers = [
                _finite_parameter_number(value, f"scaler.{name}")
                for value in values
            ]
            if name == "scale" and any(value <= 0 for value in numbers):
                raise LatencyBenchmarkError(
                    "NFL scaler scale parameter snapshot must be positive"
                )
            if name == "var" and any(value < 0 for value in numbers):
                raise LatencyBenchmarkError(
                    "NFL scaler variance parameter snapshot must be non-negative"
                )
        if _finite_parameter_number(
            scaler.get("n_samples_seen"),
            "scaler.n_samples_seen",
        ) <= 0:
            raise LatencyBenchmarkError(
                "NFL scaler sample count parameter snapshot must be positive"
            )
        classes = classifier.get("classes")
        if classes != sorted(x11.TRANSITION_CLASSES):
            raise LatencyBenchmarkError(
                "NFL classifier classes parameter snapshot is invalid"
            )
        coefficients = classifier.get("coef")
        intercept = classifier.get("intercept")
        iterations = classifier.get("n_iter")
        if (
            not isinstance(coefficients, list)
            or len(coefficients) != len(classes)
            or any(
                not isinstance(row, list) or len(row) != width
                for row in coefficients
            )
            or not isinstance(intercept, list)
            or len(intercept) != len(classes)
            or not isinstance(iterations, list)
            or not iterations
        ):
            raise LatencyBenchmarkError(
                "NFL fitted classifier parameter snapshot dimensions are invalid"
            )
        for row in coefficients:
            for value in row:
                _finite_parameter_number(value, "classifier.coef")
        for value in intercept:
            _finite_parameter_number(value, "classifier.intercept")
        if any(
            type(value) is not int or value <= 0 for value in iterations
        ):
            raise LatencyBenchmarkError(
                "NFL classifier iteration parameter snapshot is invalid"
            )
        return snapshot

    if model_id == DYNAMIC_SOCCER_MODEL_ID:
        if set(snapshot) != {
            "snapshot_version",
            "model_family",
            "base_rate_model",
            "dynamic_intensity_model",
            "temperature_calibration",
        } or (
            snapshot.get("snapshot_version") != "v1"
            or snapshot.get("model_family")
            != (
                "dixon_coles_base_rate_plus_symmetric_dynamic_intensity_"
                "plus_multiclass_temperature"
            )
        ):
            raise LatencyBenchmarkError(
                "soccer model parameter snapshot structure is invalid"
            )
        base_rate_model = snapshot.get("base_rate_model")
        base_rate_fields = {item.name for item in fields(x12.DixonColesModel)}
        if (
            not isinstance(base_rate_model, dict)
            or set(base_rate_model) != base_rate_fields
        ):
            raise LatencyBenchmarkError(
                "soccer base-rate parameter snapshot fields are invalid"
            )
        team_ids = base_rate_model.get("team_ids")
        parameters = base_rate_model.get("parameters")
        if (
            not isinstance(team_ids, list)
            or len(team_ids) < 2
            or any(type(value) is not int or value <= 0 for value in team_ids)
            or len(team_ids) != len(set(team_ids))
            or team_ids != sorted(team_ids)
            or base_rate_model.get("reference_team_id") not in team_ids
            or not isinstance(parameters, list)
            or not parameters
        ):
            raise LatencyBenchmarkError(
                "soccer base-rate parameter snapshot identity is invalid"
            )
        for value in parameters:
            _finite_parameter_number(value, "base_rate_model.parameters")
        for name in base_rate_fields - {
            "team_ids",
            "reference_team_id",
            "parameters",
            "optimizer_status",
        }:
            _finite_parameter_number(
                base_rate_model.get(name),
                f"base_rate_model.{name}",
            )
        if (
            type(base_rate_model.get("optimizer_status")) is not str
            or not base_rate_model["optimizer_status"]
        ):
            raise LatencyBenchmarkError(
                "soccer base-rate optimizer status parameter snapshot is invalid"
            )

        dynamic_model = snapshot.get("dynamic_intensity_model")
        dynamic_fields = {
            item.name
            for item in fields(
                soccer_transition_model.DynamicIntensityModel
            )
        }
        if (
            not isinstance(dynamic_model, dict)
            or set(dynamic_model) != dynamic_fields
            or dynamic_model.get("coefficient_names")
            != list(soccer_transition_model.COEFFICIENT_NAMES)
        ):
            raise LatencyBenchmarkError(
                "soccer dynamic parameter snapshot fields are invalid"
            )
        coefficients = dynamic_model.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != len(
            soccer_transition_model.COEFFICIENT_NAMES
        ):
            raise LatencyBenchmarkError(
                "soccer dynamic coefficients are invalid"
            )
        for value in coefficients:
            _finite_parameter_number(
                value,
                "dynamic_intensity_model.coefficients",
            )
        for name in (
            "l2_penalty",
            "objective",
            "initial_objective",
            "projected_gradient_inf_norm",
        ):
            _finite_parameter_number(
                dynamic_model.get(name),
                f"dynamic_intensity_model.{name}",
            )
        if (
            type(dynamic_model.get("iterations")) is not int
            or int(dynamic_model["iterations"]) < 0
            or type(dynamic_model.get("optimizer_status")) is not str
            or not dynamic_model["optimizer_status"]
            or dynamic_model.get("parameter_sha256")
            != soccer_transition_model.DynamicIntensityModel(
                coefficients=tuple(float(value) for value in coefficients),
                l2_penalty=float(dynamic_model["l2_penalty"]),
                objective=float(dynamic_model["objective"]),
                iterations=int(dynamic_model["iterations"]),
                optimizer_status=str(dynamic_model["optimizer_status"]),
                initial_objective=float(dynamic_model["initial_objective"]),
                projected_gradient_inf_norm=float(
                    dynamic_model["projected_gradient_inf_norm"]
                ),
            ).parameter_sha256
        ):
            raise LatencyBenchmarkError(
                "soccer dynamic parameter snapshot identity is invalid"
            )

        temperature_calibration = snapshot.get("temperature_calibration")
        calibration_fields = {
            item.name
            for item in fields(
                soccer_transition_model.TemperatureCalibration
            )
        }
        if (
            not isinstance(temperature_calibration, dict)
            or set(temperature_calibration) != calibration_fields
        ):
            raise LatencyBenchmarkError(
                "soccer temperature calibration snapshot fields are invalid"
            )
        try:
            fitted_calibration = (
                soccer_transition_model.TemperatureCalibration(
                    temperature=float(
                        temperature_calibration["temperature"]
                    ),
                    initial_objective=float(
                        temperature_calibration["initial_objective"]
                    ),
                    objective=float(
                        temperature_calibration["objective"]
                    ),
                    iterations=temperature_calibration["iterations"],
                    optimizer_status=temperature_calibration[
                        "optimizer_status"
                    ],
                    calibration_match_count=temperature_calibration[
                        "calibration_match_count"
                    ],
                    calibration_observation_count=temperature_calibration[
                        "calibration_observation_count"
                    ],
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            soccer_transition_model.SoccerTransitionModelError,
        ) as error:
            raise LatencyBenchmarkError(
                "soccer temperature calibration snapshot identity is invalid"
            ) from error
        if (
            temperature_calibration.get("parameter_sha256")
            != fitted_calibration.parameter_sha256
        ):
            raise LatencyBenchmarkError(
                "soccer temperature calibration snapshot identity is invalid"
            )
        return snapshot

    raise LatencyBenchmarkError(
        f"unsupported model parameter snapshot for {model_id}"
    )


def _find_raw_object(
    *,
    program_root: Path,
    digest: str,
    suffix: str,
) -> Path:
    hexdigest = _validate_digest(digest, "source object hash").removeprefix(
        _HASH_PREFIX
    )
    matches = list((program_root / "var" / "raw" / "raw").rglob(
        f"{hexdigest}{suffix}"
    ))
    if len(matches) != 1:
        raise LatencyBenchmarkError(
            f"source object {digest} has {len(matches)} local matches"
        )
    return matches[0]


def _find_manifest(
    *,
    program_root: Path,
    digest: str,
) -> tuple[Path, dict[str, object]]:
    hexdigest = _validate_digest(digest, "manifest hash").removeprefix(
        _HASH_PREFIX
    )
    matches = list((program_root / "var" / "raw" / "manifests").rglob(
        f"{hexdigest}.manifest.json"
    ))
    if len(matches) != 1:
        raise LatencyBenchmarkError(
            f"manifest {digest} has {len(matches)} local matches"
        )
    try:
        document = json.loads(matches[0].read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise LatencyBenchmarkError("source manifest cannot be read") from error
    if document.get("manifest_sha256") != digest:
        raise LatencyBenchmarkError("source manifest hash binding changed")
    return matches[0], document


def _native_scalar(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
    if type(value) is str and value and value == value.strip():
        return value
    raise LatencyBenchmarkError("native identifier is not a stable scalar")


def _fixed_point_probabilities(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    scale: int = 15,
) -> dict[str, dict[str, object]]:
    probabilities = np.asarray(values, dtype=float)
    if (
        probabilities.shape != (len(labels),)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not math.isclose(
            float(probabilities.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise LatencyBenchmarkError("model probabilities are not a distribution")
    denominator = 10**scale
    scaled = probabilities * denominator
    atoms = np.floor(scaled).astype(object)
    remainder = denominator - sum(int(value) for value in atoms)
    order = sorted(
        range(len(labels)),
        key=lambda index: (-(scaled[index] - math.floor(scaled[index])), index),
    )
    for index in order[:remainder]:
        atoms[index] = int(atoms[index]) + 1
    if sum(int(value) for value in atoms) != denominator:
        raise LatencyBenchmarkError("fixed-point probabilities are not exact")
    return {
        label: {"atoms": str(int(atoms[index])), "scale": scale}
        for index, label in enumerate(labels)
    }


def _strict_registered_model_output(
    document: Mapping[str, object],
    *,
    registry_row: ModelRegistryRow,
) -> contracts.ModelOutputV1:
    try:
        output = contracts.ModelOutputV1.model_validate(document)
    except ValueError as error:
        raise LatencyBenchmarkError(
            "model output failed strict ModelOutputV1 validation"
        ) from error
    expected = {
        "model_id": registry_row.model_id,
        "model_version": registry_row.model_version,
        "experiment_id": registry_row.experiment_id,
        "horizon": registry_row.horizon,
        "state_space": tuple(sorted(registry_row.state_space)),
        "data_sha256": registry_row.data_manifest_sha256,
        "config_sha256": registry_row.parameter_config_sha256,
    }
    actual = {
        "model_id": output.model_id,
        "model_version": output.model_version,
        "experiment_id": output.experiment_id,
        "horizon": output.horizon,
        "state_space": output.state_space,
        "data_sha256": str(output.data_sha256),
        "config_sha256": str(output.config_sha256),
    }
    if actual != expected:
        raise LatencyBenchmarkError(
            "ModelOutputV1 does not match the preloaded model registry binding"
        )
    return output


def _output_is_bound(
    value: object,
    *,
    registry_row: ModelRegistryRow,
) -> bool:
    return (
        isinstance(value, contracts.ModelOutputV1)
        and value.model_id == registry_row.model_id
        and value.model_version == registry_row.model_version
        and value.experiment_id == registry_row.experiment_id
    )


def _common_governance(
    *,
    row: ModelRegistryRow,
    model_parameter_snapshot: Mapping[str, object],
    code_sha256: Mapping[str, str],
) -> dict[str, object]:
    canonical_snapshot = _validate_model_parameter_snapshot(
        row.model_id,
        dict(model_parameter_snapshot),
    )
    return {
        "registry_status": row.status,
        "data_manifest_sha256": _validate_digest(
            row.data_manifest_sha256,
            "registered data_manifest_sha256",
        ),
        "training_manifest_sha256": _validate_digest(
            row.training_manifest_sha256,
            "registered training_manifest_sha256",
        ),
        "parameter_config_sha256": _validate_digest(
            row.parameter_config_sha256,
            "registered parameter_config_sha256",
        ),
        "model_parameter_snapshot": canonical_snapshot,
        "model_parameter_sha256": _sha256_json(canonical_snapshot),
        "model_registry_row_sha256": _registry_row_sha256(row),
        "code_sha256": dict(code_sha256),
        "preflight_full_registry_contract_validation": True,
    }


def _require_dynamic_soccer_evidence(
    evidence: Mapping[str, object],
    *,
    program_root: Path,
) -> dict[str, object]:
    """Parse only the registered full calibrated X-12 v1 evidence."""

    error_message = (
        "soccer latency has no registered calibrated dynamic empirical evidence"
    )

    def invalid(cause: BaseException | None = None) -> None:
        if cause is None:
            raise LatencyBenchmarkError(error_message)
        raise LatencyBenchmarkError(error_message) from cause

    try:
        current_registration_preflight = (
            x12._validate_x12_reproduction_preflight(program_root)
        )
    except (OSError, TypeError, ValueError, x12.X12DataError) as error:
        raise LatencyBenchmarkError(
            "soccer latency cannot establish the current X-12 "
            "reproduction preflight"
        ) from error
    evidence_registration_preflight = evidence.get(
        "registration_preflight"
    )
    if (
        not isinstance(evidence_registration_preflight, Mapping)
        or dict(evidence_registration_preflight)
        != current_registration_preflight
    ):
        raise LatencyBenchmarkError(
            "soccer evidence differs from the current X-12 "
            "reproduction preflight"
        )

    if (
        evidence.get("artifact_type")
        != "x12_real_data_dixon_coles_dynamic_transition_poc_v1"
        or evidence.get("experiment_id") != x12.X12_EXPERIMENT_ID
        or evidence.get("model_id") != x12.X12_MODEL_ID
        or evidence.get("model_version") != x12.X12_MODEL_VERSION
        or evidence.get("authorization_scope")
        != x12.X12_AUTHORIZATION_SCOPE
        or evidence.get("result_label") != x12.X12_RESULT_LABEL
        or evidence.get("execution_mode") != "full"
    ):
        invalid()

    model = evidence.get("model")
    transition_output = evidence.get("transition_output")
    transition_model = (
        model.get("transition_model")
        if isinstance(model, Mapping)
        else None
    )
    if (
        not isinstance(model, Mapping)
        or not isinstance(transition_model, Mapping)
        or transition_model.get("methodology")
        != (
            "Maia-family dynamic-covariate adaptation with frozen "
            "Dixon-Coles base-rate offset"
        )
        or transition_model.get("reproduction_scope")
        != "not a complete Maia or Cox reproduction"
        or not isinstance(transition_output, Mapping)
        or transition_output.get("state_space")
        != list(soccer_transition_model.TRANSITION_CLASSES)
        or transition_output.get("horizon_seconds")
        != soccer_transition_model.TRANSITION_HORIZON_SECONDS
        or transition_output.get("evaluation_protocol")
        != "frozen_chronological_date_group_holdout_50_25_25"
        or transition_output.get("primary_probability_variant")
        != "temperature_calibrated"
        or transition_output.get("diagnostic_probability_variant")
        != "uncalibrated"
        or transition_model.get("output_probability_variants")
        != {
            "primary": "temperature_calibrated",
            "diagnostic": "uncalibrated",
        }
    ):
        invalid()

    optimizer_max_iterations = model.get("optimizer_max_iterations")
    transition_optimizer_max_iterations = transition_model.get(
        "optimizer_max_iterations"
    )
    l2_penalty = transition_model.get("l2_penalty")
    if (
        type(optimizer_max_iterations) is not int
        or optimizer_max_iterations < 1
        or transition_optimizer_max_iterations != optimizer_max_iterations
        or type(l2_penalty) not in {int, float}
        or not math.isfinite(float(l2_penalty))
        or float(l2_penalty) < 0.0
    ):
        invalid()

    split_document = transition_model.get("split")
    split_fields = {
        item.name for item in fields(x12.X12TransitionSplitAudit)
    }
    if (
        not isinstance(split_document, Mapping)
        or set(split_document) != split_fields
        or split_document.get("method")
        != "frozen_chronological_date_group_holdout_50_25_25"
    ):
        invalid()

    date_fields = (
        "base_fit_first_date",
        "base_fit_last_date",
        "calibration_first_date",
        "calibration_last_date",
        "final_test_first_date",
        "final_test_last_date",
        "final_test_evaluated_first_date",
        "final_test_evaluated_last_date",
    )
    timestamp_fields = (
        "dynamic_fit_evaluation_cutoff",
        "temperature_fit_evaluation_cutoff",
        "base_fit_max_outcome_available_at",
        "calibration_max_label_available_at",
    )
    count_fields = (
        "base_fit_date_count",
        "calibration_date_count",
        "final_test_date_count",
        "final_test_evaluated_date_count",
        "base_fit_match_count",
        "calibration_match_count",
        "final_test_match_count",
        "final_test_evaluated_match_count",
    )
    hash_fields = (
        "dixon_coles_parameter_sha256",
        "dynamic_parameter_sha256",
        "raw_transition_parameter_sha256",
        "temperature_parameter_sha256",
        "calibrated_transition_parameter_sha256",
    )
    try:
        parsed_dates: dict[str, pd.Timestamp] = {}
        for name in date_fields:
            value = split_document[name]
            if type(value) is not str:
                invalid()
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                invalid()
            parsed_dates[name] = timestamp.tz_convert("UTC")
        parsed_timestamps: dict[str, pd.Timestamp] = {}
        for name in timestamp_fields:
            value = split_document[name]
            if type(value) is not str:
                invalid()
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                invalid()
            parsed_timestamps[name] = timestamp.tz_convert("UTC")
        for name in count_fields:
            if (
                type(split_document[name]) is not int
                or int(split_document[name]) < 1
            ):
                invalid()
        for name in hash_fields:
            _validate_digest(split_document[name], f"soccer split {name}")
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        LatencyBenchmarkError,
    ) as error:
        invalid(error)

    if not (
        parsed_dates["base_fit_first_date"]
        <= parsed_dates["base_fit_last_date"]
        < parsed_dates["calibration_first_date"]
        <= parsed_dates["calibration_last_date"]
        < parsed_dates["final_test_first_date"]
        <= parsed_dates["final_test_evaluated_first_date"]
        <= parsed_dates["final_test_evaluated_last_date"]
        <= parsed_dates["final_test_last_date"]
        and int(split_document["final_test_evaluated_date_count"])
        <= int(split_document["final_test_date_count"])
        and int(split_document["final_test_evaluated_match_count"])
        <= int(split_document["final_test_match_count"])
        and parsed_timestamps["base_fit_max_outcome_available_at"]
        < parsed_timestamps["dynamic_fit_evaluation_cutoff"]
        < parsed_timestamps["temperature_fit_evaluation_cutoff"]
        and parsed_timestamps["calibration_max_label_available_at"]
        < parsed_timestamps["temperature_fit_evaluation_cutoff"]
    ):
        invalid()

    calibration_document = transition_model.get("temperature_calibration")
    calibration_fields = {
        item.name
        for item in fields(
            soccer_transition_model.TemperatureCalibration
        )
    }
    if (
        not isinstance(calibration_document, Mapping)
        or set(calibration_document) != calibration_fields
    ):
        invalid()
    try:
        temperature_calibration = (
            soccer_transition_model.TemperatureCalibration(
                temperature=calibration_document["temperature"],
                initial_objective=calibration_document[
                    "initial_objective"
                ],
                objective=calibration_document["objective"],
                iterations=calibration_document["iterations"],
                optimizer_status=calibration_document[
                    "optimizer_status"
                ],
                calibration_match_count=calibration_document[
                    "calibration_match_count"
                ],
                calibration_observation_count=calibration_document[
                    "calibration_observation_count"
                ],
            )
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        soccer_transition_model.SoccerTransitionModelError,
    ) as error:
        invalid(error)
    if (
        calibration_document.get("parameter_sha256")
        != temperature_calibration.parameter_sha256
        or split_document["temperature_parameter_sha256"]
        != temperature_calibration.parameter_sha256
        or temperature_calibration.calibration_match_count
        != split_document["calibration_match_count"]
        or temperature_calibration.calibration_observation_count
        != (
            temperature_calibration.calibration_match_count
            * len(x12.TRANSITION_SNAPSHOTS)
        )
    ):
        invalid()

    raw_transition_sha256 = _sha256_json(
        {
            "dixon_coles_parameter_sha256": split_document[
                "dixon_coles_parameter_sha256"
            ],
            "dynamic_parameter_sha256": split_document[
                "dynamic_parameter_sha256"
            ],
            "model_id": x12.X12_MODEL_ID,
            "model_version": x12.X12_MODEL_VERSION,
            "probability_variant": "uncalibrated",
        }
    )
    calibrated_transition_sha256 = _sha256_json(
        {
            "model_id": x12.X12_MODEL_ID,
            "model_version": x12.X12_MODEL_VERSION,
            "raw_transition_parameter_sha256": raw_transition_sha256,
            "temperature_parameter_sha256": (
                temperature_calibration.parameter_sha256
            ),
            "probability_variant": "temperature_calibrated",
        }
    )
    if (
        split_document["raw_transition_parameter_sha256"]
        != raw_transition_sha256
        or split_document["calibrated_transition_parameter_sha256"]
        != calibrated_transition_sha256
    ):
        invalid()

    metrics = transition_output.get("metrics")
    raw_metrics = (
        metrics.get("raw_model_metrics")
        if isinstance(metrics, Mapping)
        else None
    )
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("classes")
        != list(soccer_transition_model.TRANSITION_CLASSES)
        or type(metrics.get("observations")) is not int
        or int(metrics["observations"]) < 1
        or type(metrics.get("brier")) not in {int, float}
        or not math.isfinite(float(metrics["brier"]))
        or float(metrics["brier"]) < 0.0
        or type(metrics.get("log_loss")) not in {int, float}
        or not math.isfinite(float(metrics["log_loss"]))
        or float(metrics["log_loss"]) < 0.0
        or metrics.get("probability_variant")
        != "temperature_calibrated"
        or not isinstance(raw_metrics, Mapping)
        or raw_metrics.get("classes")
        != list(soccer_transition_model.TRANSITION_CLASSES)
        or raw_metrics.get("observations") != metrics["observations"]
        or type(raw_metrics.get("brier")) not in {int, float}
        or not math.isfinite(float(raw_metrics["brier"]))
        or float(raw_metrics["brier"]) < 0.0
        or type(raw_metrics.get("log_loss")) not in {int, float}
        or not math.isfinite(float(raw_metrics["log_loss"]))
        or float(raw_metrics["log_loss"]) < 0.0
        or raw_metrics.get("probability_variant") != "uncalibrated"
    ):
        invalid()

    try:
        split = x12.X12TransitionSplitAudit(
            **{
                name: (
                    parsed_timestamps[name]
                    if name in parsed_timestamps
                    else split_document[name]
                )
                for name in split_fields
            }
        )
    except (TypeError, ValueError) as error:
        invalid(error)
    return {
        "model": dict(model),
        "transition_model": dict(transition_model),
        "split": split,
        "temperature_calibration": temperature_calibration,
        "metrics": dict(metrics),
        "registration_preflight": current_registration_preflight,
        "optimizer_max_iterations": optimizer_max_iterations,
        "l2_penalty": float(l2_penalty),
    }


def _soccer_dynamic_transition(
    previous_state: soccer_game_state.SoccerGameState,
    event: soccer_game_state.SoccerGameEvent,
    *,
    model: soccer_transition_model.DynamicIntensityModel,
    temperature_calibration: soccer_transition_model.TemperatureCalibration,
    base_home_goals: float,
    base_away_goals: float,
) -> tuple[
    soccer_game_state.SoccerGameState,
    soccer_transition_model.SoccerTransitionFeatures,
    soccer_transition_model.SoccerTransitionDistribution,
    np.ndarray,
]:
    """Execute reducer -> feature -> dynamic predictor -> calibration."""

    next_state = soccer_game_state.SOCCER_GAME_STATE_REDUCER.reduce(
        previous_state,
        event,
    )
    features = soccer_transition_model.extract_transition_features(next_state)
    distribution = soccer_transition_model.predict_transition_distribution(
        model,
        base_home_goals=base_home_goals,
        base_away_goals=base_away_goals,
        features=features,
    )
    calibrated = soccer_transition_model.apply_multiclass_temperature(
        np.asarray([distribution.probabilities], dtype=float),
        temperature=temperature_calibration.temperature,
    )[0]
    return next_state, features, distribution, calibrated


def _nfl_latency_transition_rows(
    raw: pd.DataFrame,
    candidate: pd.Series,
) -> tuple[dict[str, object], dict[str, object]] | None:
    """Select one adjacent NFL transition by canonical native source order."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise LatencyBenchmarkError(
            "NFL latency raw rows must be a nonempty DataFrame"
        )
    required = {
        "game_id",
        "play_id",
        "order_sequence",
        "_raw_record_ordinal",
        "fixed_drive",
        "posteam",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise LatencyBenchmarkError(
            f"NFL latency raw rows are missing columns: {missing}"
        )
    game_id = str(candidate["game_id"])
    try:
        drive_number = int(candidate["drive_number"])
        candidate_play_id = _native_scalar(candidate["play_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LatencyBenchmarkError(
            "NFL latency candidate requires drive_number and play_id identity"
        ) from exc
    ordered = raw.sort_values(
        ["game_id", "order_sequence", "_raw_record_ordinal"],
        kind="mergesort",
    )
    game_rows = ordered.loc[ordered["game_id"] == game_id].reset_index(
        drop=True
    )
    matching_positions = [
        position
        for position, play_id in enumerate(game_rows["play_id"])
        if _native_scalar(play_id) == candidate_play_id
    ]
    if len(matching_positions) != 1:
        raise LatencyBenchmarkError(
            "NFL latency candidate play_id must identify exactly one native row"
        )
    post_position = matching_positions[0]
    if post_position == 0:
        return None
    post = game_rows.iloc[post_position]
    if (
        pd.isna(post["fixed_drive"])
        or int(post["fixed_drive"]) != drive_number
        or pd.isna(post["posteam"])
    ):
        return None
    pre = game_rows.iloc[post_position - 1]
    if pd.isna(pre["posteam"]):
        return None
    return pre.to_dict(), post.to_dict()


def _nfl_benchmark(
    *,
    program_root: Path,
    registry_row: ModelRegistryRow,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    loaded = x11.load_x11_dataset(
        store_root=program_root / "var" / "raw",
        program_root=program_root,
    )
    if loaded.inventory.inventory_sha256 != registry_row.data_manifest_sha256:
        raise LatencyBenchmarkError(
            "NFL loaded inventory differs from registered data manifest"
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
    model = x11._transition_model(train)
    ordered_test = test.sort_values(
        ["game_id", "drive_number"], kind="mergesort"
    )

    source_partition = next(
        item for item in loaded.inventory.partitions if item.year == 2025
    )
    source_path = _find_raw_object(
        program_root=program_root,
        digest=source_partition.object_sha256,
        suffix=".parquet",
    )
    manifest_path, manifest = _find_manifest(
        program_root=program_root,
        digest=source_partition.manifest_sha256,
    )
    required_columns = (
        "play_id",
        "order_sequence",
        "game_id",
        "season_type",
        "home_team",
        "away_team",
        "qtr",
        "quarter_seconds_remaining",
        "time",
        "game_seconds_remaining",
        "fixed_drive",
        "play_type",
        "play_type_nfl",
        "play_clock",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "goal_to_go",
        "posteam_score",
        "defteam_score",
        "posteam_score_post",
        "defteam_score_post",
        "total_home_score",
        "total_away_score",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
        "desc",
        "timeout",
        "timeout_team",
        "interception",
        "fumble_lost",
        "first_down",
        "quarter_end",
    )
    raw = pq.read_table(source_path, columns=list(required_columns)).to_pandas()
    raw["_raw_record_ordinal"] = np.arange(len(raw), dtype=int)
    observation: pd.Series | None = None
    pre_row: dict[str, object] | None = None
    post_row: dict[str, object] | None = None
    for _, candidate in ordered_test.iterrows():
        transition_rows = _nfl_latency_transition_rows(raw, candidate)
        if transition_rows is None:
            continue
        observation = candidate
        pre_row, post_row = transition_rows
        break
    if observation is None or pre_row is None or post_row is None:
        raise LatencyBenchmarkError(
            "NFL latency fold has no causally contextual adjacent state row"
        )
    state = nfl_game_state.state_from_nflverse_row(pre_row, sequence=0)
    payload = nfl_game_state.nflverse_transition_payload(
        state,
        pre_row,
        (post_row,),
        sequence=1,
    )
    native_game_id = str(observation["game_id"])
    raw_record_ordinals = (
        int(pre_row["_raw_record_ordinal"]),
        int(post_row["_raw_record_ordinal"]),
    )
    bundle = build_static_sport_observation_bundle(
        program_root=program_root,
        experiment_id=x11.X11_EXPERIMENT_ID,
        dataset_id=x11.X11_DATASET_ID,
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash=source_partition.object_sha256,
        raw_record_ordinals=raw_record_ordinals,
        partition=source_partition.partition,
        fetched_at=str(manifest["fetched_at"]),
        source_at=None,
        competition_id="cmp_nfl",
        game_id=f"game_nflverse_{native_game_id}",
        participant_ids=(
            f"participant_{pre_row['away_team']}",
            f"participant_{pre_row['home_team']}",
        ),
        native_namespace="nflverse.play",
        native_ids=(
            f"{native_game_id}:{_native_scalar(pre_row['play_id'])}",
            f"{native_game_id}:{_native_scalar(post_row['play_id'])}",
        ),
        normalized_source_sequence=1,
        normalized_payload=payload,
    )
    event = nfl_game_state.event_from_nflverse_envelope(
        bundle.normalized,
        program_root=program_root,
        raw_parents=bundle.raw,
    )
    next_state = nfl_game_state.NFL_GAME_STATE_REDUCER.reduce(state, event)
    spread_prior = float(observation["spread_prior"])

    def extract_features(current: nfl_game_state.NFLGameState) -> np.ndarray:
        if current.possession_team not in {
            current.home_team,
            current.away_team,
        }:
            raise LatencyBenchmarkError(
                "NFL representative state has no canonical possession"
            )
        return np.asarray(
            [[
                float(current.home_score - current.away_score),
                float(current.game_seconds_remaining),
                float(current.possession_team == current.home_team),
                float(current.home_timeouts_remaining),
                float(current.away_timeouts_remaining),
                spread_prior,
            ]],
            dtype=float,
        )

    fixed_features = extract_features(next_state)
    expected_features = observation.loc[list(x11.GAME_STATE_FEATURES)].to_numpy(
        dtype=float
    ).reshape(1, -1)
    if not np.array_equal(fixed_features, expected_features):
        raise LatencyBenchmarkError(
            "NFL reducer state does not reproduce the registered model feature row"
        )

    def infer(features: np.ndarray) -> np.ndarray:
        return x11._fixed_transition_probabilities(model, features)[0]

    fixed_probabilities = infer(fixed_features)
    model_parameter_snapshot = _nfl_model_parameter_snapshot(model)
    evidence, evidence_common = _load_verified_evidence(
        program_root=program_root,
        relative_path=(
            "artifacts/game-state/nfl/"
            "x11_real_data_pipeline_evidence_v0.json"
        ),
    )
    execution_config_sha256 = _sha256_json(
        evidence["model_configuration"]["drive_transition"]
    )
    if execution_config_sha256 != registry_row.parameter_config_sha256:
        raise LatencyBenchmarkError(
            "NFL executed configuration differs from the model registry"
        )
    governance = _common_governance(
        row=registry_row,
        model_parameter_snapshot=model_parameter_snapshot,
        code_sha256=_source_code_hashes(
            reducer_path=Path(nfl_game_state.__file__),
            model_paths=(Path(x11.__file__),),
        ),
    )
    feature_sha256 = _sha256_json(
        {
            "next_state_sha256": canonical_state_sha256(next_state),
            "feature_names": list(x11.GAME_STATE_FEATURES),
            "feature_values": fixed_features[0].tolist(),
        }
    )

    def validate_probabilities(
        probabilities: np.ndarray,
    ) -> tuple[float, ...]:
        values = np.asarray(probabilities, dtype=float)
        if (
            values.shape != (len(x11.TRANSITION_CLASSES),)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or not math.isclose(
                float(values.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise LatencyBenchmarkError(
                "NFL model probabilities are not a distribution"
            )
        return tuple(float(value) for value in values)

    fixed_probability_sample = validate_probabilities(fixed_probabilities)

    def full_path() -> tuple[float, ...]:
        current = nfl_game_state.NFL_GAME_STATE_REDUCER.reduce(state, event)
        features = extract_features(current)
        probabilities = infer(features)
        return validate_probabilities(probabilities)

    report = benchmark_model_stages(
        model_id=registry_row.model_id,
        experiment_id=registry_row.experiment_id,
        stages={
            "state_reducer": (
                lambda: nfl_game_state.NFL_GAME_STATE_REDUCER.reduce(state, event)
            ),
            "feature_extraction": lambda: extract_features(next_state),
            "model_inference": lambda: infer(fixed_features),
            "output_validation": (
                lambda: validate_probabilities(fixed_probabilities)
            ),
            "full_path": full_path,
        },
        validators={
            "state_reducer": lambda value: isinstance(
                value, nfl_game_state.NFLGameState
            )
            and value == next_state,
            "feature_extraction": lambda value: isinstance(value, np.ndarray)
            and value.shape == fixed_features.shape
            and bool(np.array_equal(value, fixed_features)),
            "model_inference": lambda value: isinstance(value, np.ndarray)
            and value.shape == fixed_probabilities.shape
            and bool(np.allclose(value, fixed_probabilities, rtol=0.0, atol=0.0)),
            "output_validation": lambda value: (
                value == fixed_probability_sample
            ),
            "full_path": lambda value: value == fixed_probability_sample,
        },
        warmup=warmup,
        repeats=repeats,
    )
    report.update(
        {
            "model_version": registry_row.model_version,
            "sport": "nfl",
            "contract_output": None,
            "pit_status": "PIT_UNPROVEN",
            "pit_cutoff_at": None,
            "output_validation_scope": (
                "probability_domain_and_sum_only"
            ),
            "governance": governance,
            "representative_input_lineage": {
                "dataset_id": x11.X11_DATASET_ID,
                "source_partition": source_partition.partition,
                "source_manifest_path": manifest_path.relative_to(
                    program_root
                ).as_posix(),
                "source_manifest_sha256": source_partition.manifest_sha256,
                "source_object_sha256": source_partition.object_sha256,
                "raw_record_ordinals": list(raw_record_ordinals),
                "game_id": state.game_id,
                "state_sequence": state.sequence,
                "event_sequence": event.sequence,
                "event_id": event.event_id,
                "previous_state_sha256": canonical_state_sha256(state),
                "event_sha256": canonical_state_sha256(event),
                "next_state_sha256": canonical_state_sha256(next_state),
                "feature_sha256": feature_sha256,
                "training_cutoff_exclusive": pd.Timestamp(cutoff).isoformat(),
                "training_games": int(train["game_id"].nunique()),
                "training_observations": int(len(train)),
            },
            "accuracy_reference": _accuracy_reference(
                evidence_common,
                evidence=evidence,
                metric_pointer="/transition_evaluation/metrics",
                row=registry_row,
                governance=governance,
            ),
        }
    )
    return report


def _soccer_benchmark(
    *,
    program_root: Path,
    registry_row: ModelRegistryRow,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    evidence, evidence_common = _load_verified_evidence(
        program_root=program_root,
        relative_path=DYNAMIC_SOCCER_EVIDENCE_PATH,
    )
    parsed_evidence = _require_dynamic_soccer_evidence(
        evidence,
        program_root=program_root,
    )
    split = parsed_evidence["split"]
    temperature_calibration = parsed_evidence["temperature_calibration"]
    if (
        not isinstance(split, x12.X12TransitionSplitAudit)
        or not isinstance(
            temperature_calibration,
            soccer_transition_model.TemperatureCalibration,
        )
    ):
        raise LatencyBenchmarkError(
            "soccer calibrated evidence parser emitted invalid fitted inputs"
        )
    loaded = x12.load_x12_dataset(
        store_root=program_root / "var" / "raw",
        program_root=program_root,
    )
    if loaded.inventory.inventory_sha256 != registry_row.data_manifest_sha256:
        raise LatencyBenchmarkError(
            "soccer loaded inventory differs from registered data manifest"
        )
    matches = loaded.matches.sort_values(
        ["played_at", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    try:
        (
            base_fit,
            calibration,
            final_test,
            final_test_evaluated,
        ) = x12._frozen_transition_split(
            matches,
            evaluation_match_limit=(
                split.final_test_evaluated_match_count
            ),
        )
    except (TypeError, ValueError, x12.X12DataError) as error:
        raise LatencyBenchmarkError(
            "soccer registered frozen transition split cannot be reproduced"
        ) from error

    def utc_timestamp(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    frame_audits = (
        (
            base_fit,
            split.base_fit_first_date,
            split.base_fit_last_date,
            split.base_fit_date_count,
            split.base_fit_match_count,
        ),
        (
            calibration,
            split.calibration_first_date,
            split.calibration_last_date,
            split.calibration_date_count,
            split.calibration_match_count,
        ),
        (
            final_test,
            split.final_test_first_date,
            split.final_test_last_date,
            split.final_test_date_count,
            split.final_test_match_count,
        ),
        (
            final_test_evaluated,
            split.final_test_evaluated_first_date,
            split.final_test_evaluated_last_date,
            split.final_test_evaluated_date_count,
            split.final_test_evaluated_match_count,
        ),
    )
    if any(
        frame.empty
        or utc_timestamp(frame["match_date"].min())
        != utc_timestamp(first_date)
        or utc_timestamp(frame["match_date"].max())
        != utc_timestamp(last_date)
        or int(frame["match_date"].nunique()) != date_count
        or int(frame["match_id"].nunique()) != match_count
        for (
            frame,
            first_date,
            last_date,
            date_count,
            match_count,
        ) in frame_audits
    ) or (
        utc_timestamp(calibration["played_at"].min())
        != split.dynamic_fit_evaluation_cutoff
        or utc_timestamp(final_test["played_at"].min())
        != split.temperature_fit_evaluation_cutoff
        or utc_timestamp(base_fit["outcome_available_at"].max())
        != split.base_fit_max_outcome_available_at
    ):
        raise LatencyBenchmarkError(
            "soccer registered frozen transition split differs from source data"
        )

    team_ids = tuple(
        sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    )
    optimizer_max_iterations = int(
        parsed_evidence["optimizer_max_iterations"]
    )
    try:
        model = x12._fit_dixon_coles(
            base_fit,
            team_ids=team_ids,
            optimizer_max_iterations=optimizer_max_iterations,
            initial_parameters=None,
        )
    except (TypeError, ValueError, x12.X12DataError) as error:
        raise LatencyBenchmarkError(
            "soccer registered base Dixon-Coles fit failed closed"
        ) from error
    dixon_coles_parameter_sha256 = _sha256_json(list(model.parameters))
    dynamic_training_rows = x12._dynamic_training_rows(
        base_fit,
        loaded.goals,
        loaded.dismissals,
        model=model,
    )
    held_out_match_ids = frozenset(
        set(calibration["match_id"].astype(int))
        | set(final_test["match_id"].astype(int))
    )
    try:
        dynamic_model = soccer_transition_model.fit_dynamic_intensity(
            dynamic_training_rows,
            evaluation_cutoff=split.dynamic_fit_evaluation_cutoff,
            held_out_match_ids=held_out_match_ids,
            l2_penalty=float(parsed_evidence["l2_penalty"]),
            optimizer_max_iterations=optimizer_max_iterations,
        )
    except soccer_transition_model.SoccerTransitionModelError as error:
        raise LatencyBenchmarkError(
            "soccer dynamic transition latency fit failed closed"
        ) from error
    raw_transition_parameter_sha256 = _sha256_json(
        {
            "dixon_coles_parameter_sha256": (
                dixon_coles_parameter_sha256
            ),
            "dynamic_parameter_sha256": dynamic_model.parameter_sha256,
            "model_id": x12.X12_MODEL_ID,
            "model_version": x12.X12_MODEL_VERSION,
            "probability_variant": "uncalibrated",
        }
    )
    calibration_rows = x12._transition_rows(
        calibration,
        loaded.goals,
        loaded.dismissals,
        model=model,
        dynamic_model=dynamic_model,
        inventory_sha256_value=loaded.inventory.inventory_sha256,
        dixon_coles_parameter_sha256=dixon_coles_parameter_sha256,
        raw_transition_parameter_sha256=(
            raw_transition_parameter_sha256
        ),
        calibrated_transition_parameter_sha256=None,
        temperature_calibration=None,
    )
    calibration_predictions = pd.DataFrame(calibration_rows).sort_values(
        ["prediction_at", "match_id", "period", "period_clock_ms"],
        kind="mergesort",
    )
    calibration_max_label_available_at = utc_timestamp(
        calibration_predictions["prediction_at"].max()
        + pd.Timedelta(
            seconds=soccer_transition_model.TRANSITION_HORIZON_SECONDS
        )
    )
    if (
        calibration_predictions.empty
        or calibration_max_label_available_at
        != split.calibration_max_label_available_at
    ):
        raise LatencyBenchmarkError(
            "soccer registered calibration interval differs from source data"
        )
    raw_columns = [
        f"raw_probability_{label}"
        for label in soccer_transition_model.TRANSITION_CLASSES
    ]
    try:
        fitted_temperature_calibration = (
            soccer_transition_model.fit_multiclass_temperature(
                calibration_predictions[
                    "observed_transition"
                ].to_numpy(dtype=object),
                calibration_predictions[raw_columns].to_numpy(dtype=float),
                groups=calibration_predictions[
                    "match_id"
                ].to_numpy(dtype=object),
                minimum_matches=20,
                optimizer_max_iterations=optimizer_max_iterations,
            )
        )
    except soccer_transition_model.SoccerTransitionModelError as error:
        raise LatencyBenchmarkError(
            "soccer transition calibration latency fit failed closed"
        ) from error
    calibrated_transition_parameter_sha256 = _sha256_json(
        {
            "model_id": x12.X12_MODEL_ID,
            "model_version": x12.X12_MODEL_VERSION,
            "raw_transition_parameter_sha256": (
                raw_transition_parameter_sha256
            ),
            "temperature_parameter_sha256": (
                fitted_temperature_calibration.parameter_sha256
            ),
            "probability_variant": "temperature_calibrated",
        }
    )
    if (
        split.dixon_coles_parameter_sha256
        != dixon_coles_parameter_sha256
        or split.dynamic_parameter_sha256
        != dynamic_model.parameter_sha256
        or split.raw_transition_parameter_sha256
        != raw_transition_parameter_sha256
        or split.temperature_parameter_sha256
        != fitted_temperature_calibration.parameter_sha256
        or split.calibrated_transition_parameter_sha256
        != calibrated_transition_parameter_sha256
        or fitted_temperature_calibration != temperature_calibration
    ):
        raise LatencyBenchmarkError(
            "soccer fitted calibrated parameters differ from registered evidence"
        )
    observation = final_test_evaluated.iloc[0]
    base_home_goals, base_away_goals = x12._expected_goals(
        model,
        home_team_id=int(observation["home_team_id"]),
        away_team_id=int(observation["away_team_id"]),
    )
    source_object_sha256 = str(observation["event_object_sha256"])
    source_manifest_sha256 = str(observation["event_manifest_sha256"])
    source_path = _find_raw_object(
        program_root=program_root,
        digest=source_object_sha256,
        suffix=".json",
    )
    manifest_path, manifest = _find_manifest(
        program_root=program_root,
        digest=source_manifest_sha256,
    )
    try:
        raw_events = json.loads(source_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise LatencyBenchmarkError(
            "soccer representative source events cannot be read"
        ) from error
    if not isinstance(raw_events, list) or not raw_events:
        raise LatencyBenchmarkError("soccer representative event file is empty")
    game_id = f"game_statsbomb_{int(observation['match_id'])}"
    current_state = soccer_game_state.initial_soccer_game_state(
        game_id,
        home_team_id=int(observation["home_team_id"]),
        away_team_id=int(observation["away_team_id"]),
    )
    representative: tuple[
        soccer_game_state.SoccerGameState,
        soccer_game_state.SoccerGameEvent,
        soccer_game_state.SoccerGameState,
        soccer_transition_model.SoccerTransitionFeatures,
        soccer_transition_model.SoccerTransitionDistribution,
        int,
    ] | None = None
    for raw_ordinal, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise LatencyBenchmarkError(
                "soccer representative event is not an object"
            )
        event_payload = soccer_game_state.statsbomb_event_payload(
            raw_event,
            game_id=game_id,
        )
        event_sequence = int(event_payload["sequence"])
        if event_sequence - 1 != raw_ordinal:
            raise LatencyBenchmarkError(
                "soccer raw event ordinal differs from its native sequence"
            )
        native_event_id = str(event_payload["native_event_id"])
        bundle = build_static_sport_observation_bundle(
            program_root=program_root,
            experiment_id=x12.X12_EXPERIMENT_ID,
            dataset_id=x12.X12_DATASET_ID,
            source_system="statsbomb",
            source_stream="events",
            raw_object_hash=source_object_sha256,
            raw_record_ordinals=(raw_ordinal,),
            partition=str(manifest["upstream_partition"]),
            fetched_at=str(manifest["fetched_at"]),
            source_at=None,
            competition_id="cmp_statsbomb_2",
            game_id=game_id,
            participant_ids=(
                f"participant_statsbomb_{int(observation['home_team_id'])}",
                f"participant_statsbomb_{int(observation['away_team_id'])}",
            ),
            native_namespace="statsbomb.event",
            native_ids=(native_event_id,),
            normalized_source_sequence=event_sequence,
            normalized_payload=event_payload,
        )
        candidate_event = soccer_game_state.adapt_statsbomb_event(
            bundle.normalized,
            program_root=program_root,
            raw_parents=bundle.raw,
        )
        candidate_next_state = (
            soccer_game_state.SOCCER_GAME_STATE_REDUCER.reduce(
                current_state,
                candidate_event,
            )
        )
        try:
            candidate_features = (
                soccer_transition_model.extract_transition_features(
                    candidate_next_state
                )
            )
        except soccer_transition_model.SoccerTransitionModelError:
            current_state = candidate_next_state
            continue
        candidate_distribution = (
            soccer_transition_model.predict_transition_distribution(
                dynamic_model,
                base_home_goals=base_home_goals,
                base_away_goals=base_away_goals,
                features=candidate_features,
            )
        )
        representative = (
            current_state,
            candidate_event,
            candidate_next_state,
            candidate_features,
            candidate_distribution,
            raw_ordinal,
        )
        break
    if representative is None:
        raise LatencyBenchmarkError(
            "soccer source has no event with an eligible five-minute horizon"
        )
    (
        state,
        event,
        next_state,
        fixed_features,
        fixed_distribution,
        raw_ordinal,
    ) = representative
    fixed_probabilities = (
        soccer_transition_model.apply_multiclass_temperature(
            np.asarray([fixed_distribution.probabilities], dtype=float),
            temperature=temperature_calibration.temperature,
        )[0]
    )
    model_parameter_snapshot = _soccer_model_parameter_snapshot(
        model,
        dynamic_model,
        temperature_calibration,
    )
    transition = evidence["transition_output"]
    execution_config_sha256 = _sha256_json(
        {
            "model": evidence["model"],
            "transition_definition": {
                key: transition[key]
                for key in (
                    "availability_status",
                    "boundary_rule",
                    "horizon_seconds",
                    "state_space",
                )
            },
        }
    )
    if execution_config_sha256 != registry_row.parameter_config_sha256:
        raise LatencyBenchmarkError(
            "soccer executed configuration differs from the model registry"
        )
    governance = _common_governance(
        row=registry_row,
        model_parameter_snapshot=model_parameter_snapshot,
        code_sha256=_source_code_hashes(
            reducer_path=Path(soccer_game_state.__file__),
            model_paths=(
                Path(x12.__file__),
                Path(soccer_transition_model.__file__),
            ),
        ),
    )
    model_parameter_sha256 = str(governance["model_parameter_sha256"])
    pit_cutoff = pd.Timestamp(observation["played_at"]) + pd.Timedelta(
        milliseconds=next_state.clock_ms
    )

    def extract_features(
        current: soccer_game_state.SoccerGameState,
    ) -> soccer_transition_model.SoccerTransitionFeatures:
        return soccer_transition_model.extract_transition_features(current)

    def infer(
        features: soccer_transition_model.SoccerTransitionFeatures,
    ) -> np.ndarray:
        distribution = (
            soccer_transition_model.predict_transition_distribution(
                dynamic_model,
                base_home_goals=base_home_goals,
                base_away_goals=base_away_goals,
                features=features,
            )
        )
        return soccer_transition_model.apply_multiclass_temperature(
            np.asarray([distribution.probabilities], dtype=float),
            temperature=temperature_calibration.temperature,
        )[0]

    def output_document(
        probabilities: np.ndarray,
        features: soccer_transition_model.SoccerTransitionFeatures,
    ) -> dict[str, object]:
        return {
            "contract_version": "v1",
            "model_id": registry_row.model_id,
            "model_version": registry_row.model_version,
            "experiment_id": registry_row.experiment_id,
            "run_id": (
                "run_latency_"
                + model_parameter_sha256.removeprefix(_HASH_PREFIX)[:24]
            ),
            "game_id": state.game_id,
            "state_event_id": event.event_id,
            "pit_cutoff_at": pit_cutoff.isoformat().replace("+00:00", "Z"),
            "output_kind": "state_transition",
            "transition_unit": "five_minute_interval",
            "state_space": list(soccer_transition_model.TRANSITION_CLASSES),
            "horizon": "next_state_transition",
            "probabilities": _fixed_point_probabilities(
                soccer_transition_model.TRANSITION_CLASSES,
                probabilities,
            ),
            "feature_sha256": features.feature_sha256,
            "data_sha256": registry_row.data_manifest_sha256,
            "config_sha256": registry_row.parameter_config_sha256,
            "quality_flags": [
                "preliminary_rules",
                "source_clock_unverified",
            ],
        }

    def validate_output(
        probabilities: np.ndarray,
        features: soccer_transition_model.SoccerTransitionFeatures,
    ) -> contracts.ModelOutputV1:
        return _strict_registered_model_output(
            output_document(probabilities, features),
            registry_row=registry_row,
        )

    fixed_output = validate_output(fixed_probabilities, fixed_features)
    contracts.validate_contract_v1(
        program_root,
        "model-output/v1.schema.yaml",
        fixed_output,
    )

    def full_path() -> contracts.ModelOutputV1:
        _, features, _, probabilities = _soccer_dynamic_transition(
            state,
            event,
            model=dynamic_model,
            temperature_calibration=temperature_calibration,
            base_home_goals=base_home_goals,
            base_away_goals=base_away_goals,
        )
        return validate_output(probabilities, features)

    report = benchmark_model_stages(
        model_id=registry_row.model_id,
        experiment_id=registry_row.experiment_id,
        stages={
            "state_reducer": (
                lambda: soccer_game_state.SOCCER_GAME_STATE_REDUCER.reduce(
                    state, event
                )
            ),
            "feature_extraction": lambda: extract_features(next_state),
            "model_inference": lambda: infer(fixed_features),
            "output_validation": (
                lambda: validate_output(fixed_probabilities, fixed_features)
            ),
            "full_path": full_path,
        },
        validators={
            "state_reducer": lambda value: isinstance(
                value, soccer_game_state.SoccerGameState
            )
            and value == next_state,
            "feature_extraction": lambda value: value == fixed_features,
            "model_inference": lambda value: isinstance(value, np.ndarray)
            and value.shape == fixed_probabilities.shape
            and bool(np.allclose(value, fixed_probabilities, rtol=0.0, atol=0.0)),
            "output_validation": lambda value: _output_is_bound(
                value,
                registry_row=registry_row,
            ),
            "full_path": lambda value: _output_is_bound(
                value,
                registry_row=registry_row,
            ),
        },
        warmup=warmup,
        repeats=repeats,
    )
    report.update(
        {
            "model_version": registry_row.model_version,
            "sport": "soccer",
            "governance": governance,
            "representative_input_lineage": {
                "dataset_id": x12.X12_DATASET_ID,
                "source_partition": str(manifest["upstream_partition"]),
                "source_manifest_path": manifest_path.relative_to(
                    program_root
                ).as_posix(),
                "source_manifest_sha256": source_manifest_sha256,
                "source_object_sha256": source_object_sha256,
                "raw_record_ordinals": [raw_ordinal],
                "game_id": state.game_id,
                "state_sequence": state.sequence,
                "event_sequence": event.sequence,
                "event_id": event.event_id,
                "previous_state_sha256": canonical_state_sha256(state),
                "event_sha256": canonical_state_sha256(event),
                "next_state_sha256": canonical_state_sha256(next_state),
                "feature_sha256": str(fixed_output.feature_sha256),
                "base_fit_cutoff_exclusive": (
                    split.dynamic_fit_evaluation_cutoff.isoformat()
                ),
                "base_fit_matches": int(len(base_fit)),
                "calibration_cutoff_exclusive": (
                    split.temperature_fit_evaluation_cutoff.isoformat()
                ),
                "calibration_matches": int(len(calibration)),
                "final_test_matches": int(len(final_test)),
                "final_test_evaluated_matches": int(
                    len(final_test_evaluated)
                ),
                "dixon_coles_parameter_sha256": (
                    dixon_coles_parameter_sha256
                ),
                "dynamic_parameter_sha256": (
                    dynamic_model.parameter_sha256
                ),
                "raw_transition_parameter_sha256": (
                    raw_transition_parameter_sha256
                ),
                "temperature_parameter_sha256": (
                    temperature_calibration.parameter_sha256
                ),
                "calibrated_transition_parameter_sha256": (
                    calibrated_transition_parameter_sha256
                ),
            },
            "accuracy_reference": _accuracy_reference(
                evidence_common,
                evidence=evidence,
                metric_pointer="/transition_output/metrics",
                row=registry_row,
                governance=governance,
            ),
        }
    )
    return report


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
            "pyarrow": version("pyarrow"),
            "pydantic": version("pydantic"),
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
    """Fit frozen last-fold models and benchmark eligible complete paths."""

    root = Path(program_root).resolve()
    _require_positive_int(warmup, "warmup")
    _require_positive_int(repeats, "repeats")
    if repeats < 1_000:
        raise LatencyBenchmarkError("repeats must contain at least 1000 iterations")
    registry = _registry_rows(root)
    benchmarks = [
        _nfl_benchmark(
            program_root=root,
            registry_row=registry["MODEL-NFL-DRIVE-TRANSITION"],
            warmup=warmup,
            repeats=repeats,
        ),
        _soccer_benchmark(
            program_root=root,
            registry_row=registry[DYNAMIC_SOCCER_MODEL_ID],
            warmup=warmup,
            repeats=repeats,
        ),
    ]
    report: dict[str, object] = {
        "artifact_type": ARTIFACT_TYPE,
        "version": REPORT_VERSION,
        "result_label": RESULT_LABEL,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "live_sla_claimed": False,
        "scope": (
            "warm batch-one state/event transition through reducer, feature "
            "extraction, fitted registered transition model, and output "
            "validation bounded by each source's point-in-time eligibility"
        ),
        "protocol": {
            "timer": "time.perf_counter_ns",
            "percentile_method": "nearest_rank",
            "warmup_iterations": warmup,
            "timed_iterations": repeats,
            "minimum_timed_iterations": 1_000,
            "stages": list(REQUIRED_STAGES),
            "single_observation": True,
            "included_in_full_path": [
                "state_reducer",
                "feature_extraction",
                "registered_model_inference",
                "probability_distribution_validation",
                "strict_model_output_v1_validation_only_when_pit_eligible",
            ],
            "excluded_from_all_timed_regions": [
                "model_training",
                "raw_source_loading",
                "registry_snapshot_loading",
                "network_io",
            ],
            "preflight_registry_validation": (
                "the immutable model registry is validated before timing; "
                "ModelOutputV1 is materialized only for a source with a proven "
                "point-in-time cutoff"
            ),
        },
        "runtime": _runtime_record(),
        "benchmarks": benchmarks,
        "unavailable_models": _unavailable_model_records(),
        "unavailable_sports": [
            unavailable_sport_record(sport) for sport in UNAVAILABLE_SPORTS
        ],
        "limitations": [
            "engineering benchmark only; no live SLA or production claim",
            "accuracy is referenced from verified evidence and is not recomputed",
            (
                "the soccer five-minute path uses the registered dynamic state model "
                "with Dixon-Coles pregame base-rate offsets"
            ),
            NFL_REDUCER_LATENCY_LIMITATION,
            "wall-clock results are machine- and load-specific",
            "source ingestion, network transport, market joins, and execution are out of scope",
            "game-end-only models are not mislabeled as ModelOutputV1 full paths",
            (
                "the real NFL source has no source_at cutoff, so its latency "
                "sample is PIT_UNPROVEN with a null contract output"
            ),
        ],
    }
    report["report_sha256"] = _sha256_json(report)
    return validate_model_latency_report(report, program_root=root)


def _validate_accuracy_reference(
    accuracy: object,
    *,
    benchmark: Mapping[str, object],
    program_root: Path,
) -> None:
    if (
        not isinstance(accuracy, Mapping)
        or accuracy.get("status")
        != "referenced_existing_evidence_not_recomputed"
        or accuracy.get("scope") != ACCURACY_EVIDENCE_SCOPE
    ):
        raise LatencyBenchmarkError("accuracy reference is invalid")
    binding = accuracy.get(
        "aggregate_walk_forward_model_family_binding"
    )
    if (
        "model_binding" in accuracy
        or "model_parameter_sha256" in accuracy
        or (
            isinstance(binding, Mapping)
            and "model_parameter_sha256" in binding
        )
    ):
        raise LatencyBenchmarkError(
            "aggregate accuracy evidence must not bind a single fitted parameter"
        )
    relative = accuracy.get("evidence_path")
    if type(relative) is not str or Path(relative).is_absolute():
        raise LatencyBenchmarkError("accuracy evidence path must be relative")
    evidence, common = _load_verified_evidence(
        program_root=program_root,
        relative_path=relative,
    )
    if (
        accuracy.get("evidence_sha256") != common["evidence_sha256"]
        or accuracy.get("evidence_file_sha256")
        != common["evidence_file_sha256"]
    ):
        raise LatencyBenchmarkError("accuracy evidence hash changed")
    pointer = accuracy.get("metric_pointer")
    snapshot = resolve_json_pointer(evidence, pointer)
    if not isinstance(snapshot, dict):
        raise LatencyBenchmarkError(
            "accuracy JSON Pointer must resolve to a canonical metrics object"
        )
    if accuracy.get("reported_metrics") != snapshot:
        raise LatencyBenchmarkError(
            "accuracy reported_metrics differ from the JSON Pointer snapshot"
        )
    if accuracy.get("metric_snapshot_sha256") != _sha256_json(snapshot):
        raise LatencyBenchmarkError("accuracy metric snapshot hash changed")
    governance = benchmark.get("governance")
    if not isinstance(governance, Mapping):
        raise LatencyBenchmarkError("benchmark governance is missing")
    expected_binding: dict[str, object] = {
        "model_id": benchmark.get("model_id"),
        "model_version": benchmark.get("model_version"),
        "data_manifest_sha256": governance.get("data_manifest_sha256"),
        "parameter_config_sha256": governance.get("parameter_config_sha256"),
        "evidence_sha256": common["evidence_sha256"],
        "evidence_file_sha256": common["evidence_file_sha256"],
        "metric_snapshot_sha256": _sha256_json(snapshot),
    }
    if binding != expected_binding:
        raise LatencyBenchmarkError(
            "aggregate walk-forward model-family evidence binding is detached"
        )


def validate_model_latency_report(
    report: Mapping[str, object],
    *,
    program_root: str | Path,
) -> dict[str, object]:
    """Fail closed on incomplete, placeholder, or detached latency evidence."""

    if not isinstance(report, Mapping):
        raise LatencyBenchmarkError("latency report must be an object")
    document = dict(report)
    if document.get("artifact_type") != ARTIFACT_TYPE:
        raise LatencyBenchmarkError("latency report artifact_type is invalid")
    if document.get("version") != REPORT_VERSION:
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
        or protocol.get("single_observation") is not True
        or tuple(protocol.get("stages", ())) != REQUIRED_STAGES
    ):
        raise LatencyBenchmarkError("latency report protocol changed")
    expected_included = [
        "state_reducer",
        "feature_extraction",
        "registered_model_inference",
        "probability_distribution_validation",
        "strict_model_output_v1_validation_only_when_pit_eligible",
    ]
    expected_excluded = [
        "model_training",
        "raw_source_loading",
        "registry_snapshot_loading",
        "network_io",
    ]
    if (
        protocol.get("included_in_full_path") != expected_included
        or protocol.get("excluded_from_all_timed_regions") != expected_excluded
    ):
        raise LatencyBenchmarkError("latency timing boundary changed")
    minimum = protocol.get("minimum_timed_iterations")
    if type(minimum) is not int or minimum < 1_000:
        raise LatencyBenchmarkError("latency report permits too few iterations")

    benchmarks = document.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise LatencyBenchmarkError("latency report benchmarks are missing")
    if {
        item.get("model_id") for item in benchmarks if isinstance(item, dict)
    } != set(MEASURED_MODELS):
        raise LatencyBenchmarkError("latency report model inventory changed")
    root = Path(program_root).resolve()
    registry = _registry_rows(root)
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
        if benchmark.get("sport") == "nfl" and (
            benchmark.get("contract_output", object()) is not None
            or benchmark.get("pit_status") != "PIT_UNPROVEN"
            or benchmark.get("pit_cutoff_at", object()) is not None
            or benchmark.get("output_validation_scope")
            != "probability_domain_and_sum_only"
        ):
            raise LatencyBenchmarkError(
                "NFL non-PIT latency sample contract changed"
            )
        row = registry[str(model_id)]
        governance = benchmark.get("governance")
        if not isinstance(governance, Mapping):
            raise LatencyBenchmarkError("benchmark governance is missing")
        for field in (
            "data_manifest_sha256",
            "training_manifest_sha256",
            "parameter_config_sha256",
            "model_parameter_sha256",
            "model_registry_row_sha256",
        ):
            _validate_digest(governance.get(field), field)
        parameter_snapshot = _validate_model_parameter_snapshot(
            str(model_id),
            governance.get("model_parameter_snapshot"),
        )
        if governance.get("model_parameter_sha256") != _sha256_json(
            parameter_snapshot
        ):
            raise LatencyBenchmarkError(
                "model parameter snapshot hash mismatch"
            )
        if (
            governance.get("data_manifest_sha256") != row.data_manifest_sha256
            or governance.get("training_manifest_sha256")
            != row.training_manifest_sha256
            or governance.get("parameter_config_sha256")
            != row.parameter_config_sha256
            or governance.get("model_registry_row_sha256")
            != _registry_row_sha256(row)
            or governance.get("preflight_full_registry_contract_validation")
            is not True
        ):
            raise LatencyBenchmarkError("benchmark governance registry binding changed")
        code_hashes = governance.get("code_sha256")
        if not isinstance(code_hashes, Mapping) or set(code_hashes) != {
            "latency_harness",
            "sport_reducer",
            "model_implementation",
        }:
            raise LatencyBenchmarkError("benchmark code hash inventory changed")
        for name, digest in code_hashes.items():
            _validate_digest(digest, f"code_sha256.{name}")
        expected_code_hashes = _source_code_hashes(
            reducer_path=Path(
                nfl_game_state.__file__
                if benchmark.get("sport") == "nfl"
                else soccer_game_state.__file__
            ),
            model_paths=(
                (Path(x11.__file__),)
                if benchmark.get("sport") == "nfl"
                else (
                    Path(x12.__file__),
                    Path(soccer_transition_model.__file__),
                )
            ),
        )
        if dict(code_hashes) != expected_code_hashes:
            raise LatencyBenchmarkError("benchmark code hashes are stale")

        lineage = benchmark.get("representative_input_lineage")
        if not isinstance(lineage, Mapping):
            raise LatencyBenchmarkError(
                "representative input lineage is missing"
            )
        for field in (
            "source_manifest_sha256",
            "source_object_sha256",
            "previous_state_sha256",
            "event_sha256",
            "next_state_sha256",
            "feature_sha256",
        ):
            _validate_digest(lineage.get(field), field)
        if (
            type(lineage.get("event_id")) is not str
            or not str(lineage["event_id"]).startswith("evt_")
            or len(str(lineage["event_id"])) != 68
        ):
            raise LatencyBenchmarkError("representative event_id is invalid")

        stages = benchmark.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != set(REQUIRED_STAGES):
            raise LatencyBenchmarkError("latency benchmark stage inventory changed")
        measured_events = benchmark.get("measured_events")
        if type(measured_events) is not int or measured_events < minimum:
            raise LatencyBenchmarkError("latency benchmark event count is invalid")
        for stage in stages.values():
            if not isinstance(stage, Mapping):
                raise LatencyBenchmarkError("latency stage must be an object")
            if (
                stage.get("measurement") != "single_observation_warm_stage"
                or stage.get("timer") != "time.perf_counter_ns"
            ):
                raise LatencyBenchmarkError("latency stage protocol changed")
            count = stage.get("timed_iterations")
            operation_count = stage.get("measured_operations")
            total_ns = stage.get("total_ns")
            throughput = stage.get("operations_per_second")
            values = [
                stage.get(name)
                for name in ("p50_ns", "p95_ns", "p99_ns", "max_ns")
            ]
            if (
                type(count) is not int
                or count < minimum
                or operation_count != count
                or type(total_ns) is not int
                or total_ns <= 0
                or type(throughput) is not int
                or throughput != count * 1_000_000_000 // total_ns
                or any(type(value) is not int or value < 0 for value in values)
                or values != sorted(values)
            ):
                raise LatencyBenchmarkError("latency stage statistics are invalid")
        if benchmark.get("full_path_events_per_second") != stages["full_path"].get(
            "operations_per_second"
        ):
            raise LatencyBenchmarkError("full-path event throughput is invalid")
        _validate_accuracy_reference(
            benchmark.get("accuracy_reference"),
            benchmark=benchmark,
            program_root=root,
        )

    if document.get("unavailable_models") != _unavailable_model_records():
        raise LatencyBenchmarkError(
            "ineligible models must remain explicit and unmeasured"
        )
    expected_unavailable = [
        unavailable_sport_record(sport) for sport in UNAVAILABLE_SPORTS
    ]
    if document.get("unavailable_sports") != expected_unavailable:
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
    "resolve_json_pointer",
    "unavailable_sport_record",
    "validate_model_latency_report",
]
