"""Strict, append-only experiment preregistration and result validation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from prediction_market.raw_store import RawStoreError, read_verified_segment


EXPERIMENT_IDS = tuple(f"X-{number:02d}" for number in range(1, 13))
_EXPERIMENT_ID_SET = frozenset(EXPERIMENT_IDS)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DATASET_ID_RE = re.compile(r"^DS-[A-Z0-9][A-Z0-9-]*$")
_MODEL_ID_RE = re.compile(r"^MODEL-[A-Z0-9][A-Z0-9-]*$")
_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_RESULT_ACCEPTANCE_NOT_BEFORE = "2026-07-23T00:00:00Z"
_X08_CAPTURE_STREAMS = {
    "DS-KALSHI-LIVE-L2": ("kalshi", "orderbook"),
    "DS-POLYMARKET-PUBLIC": ("polymarket", "market"),
}
_X08_GAP_POLICY = {
    "policy_id": "recorder-heartbeat/v0",
    "maximum_gap_seconds": 60,
}
_X02_PREREGISTRATION_LOCK_IDS = frozenset(
    {
        "sampling_and_seed",
        "diff_and_stability_definitions",
        "h_split_approval",
    }
)
_REPRODUCTION_CONTRACTS = {
    "X-11": {
        "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V1",
        "scope": "team_h_nfl_fastrmodels_reproduction_v1",
        "base_scope": "preregistered_pipeline",
        "dataset_ids": (
            "DS-NFL-FASTRMODELS",
            "DS-NFLVERSE",
        ),
        "model_ids": ("MODEL-NFL-FASTRMODELS-NO-SPREAD",),
        "code_paths": (
            "src/prediction_market/models/nfl.py",
            "src/prediction_market/models/nfl_fastrmodels.py",
        ),
        "protocol_path": (
            "registries/protocols/"
            "x11_fastrmodels_no_spread_v0.json"
        ),
        "protocol_content_sha256": (
            "sha256:"
            "3e29b0aa7d85a69329b59a74b31487e4736306c9fc80d448a4a43be0b719014a"
        ),
        "inherit_base_locks": False,
    },
    "X-12": {
        "reproduction_id": (
            "REPRO-X12-SOCCER-DYNAMIC-TRANSITION-V1"
        ),
        "scope": "team_h_soccer_dynamic_transition_reproduction_v1",
        "base_scope": "poc_result",
        "dataset_ids": ("DS-STATSBOMB-OPEN",),
        "model_ids": (
            "MODEL-SOCCER-DIXON-COLES",
            "MODEL-SOCCER-DYNAMIC-INTENSITY",
        ),
        "code_paths": (
            "src/prediction_market/sports/soccer_transition_model.py",
            "src/prediction_market/sports/x12.py",
        ),
        "protocol_path": None,
        "protocol_content_sha256": None,
        "inherit_base_locks": True,
    },
}
_X11_REPRODUCTION_V2_CONTRACT = {
    "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V2",
    "scope": "team_h_nfl_fastrmodels_reproduction_v2",
    "base_scope": "preregistered_pipeline",
    "dataset_ids": (
        "DS-NFL-FASTRMODELS",
        "DS-NFLVERSE",
    ),
    "model_ids": (
        "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
    ),
    "code_paths": (
        "src/prediction_market/models/nfl.py",
        "src/prediction_market/models/nfl_fastrmodels.py",
    ),
    "protocol_path": (
        "registries/protocols/"
        "x11_fastrmodels_no_spread_v1.json"
    ),
    "protocol_content_sha256": (
        "sha256:"
        "4c9d503b6577f9aad46998f36a87b02884b2be1a7eaca8bbcc15624fc45e6c70"
    ),
    "inherit_base_locks": False,
}
_X02_TIMESTAMP_AUDIT_PREREGISTRATION = {
    "sampling_and_seed": {
        "selection_seed": 20260722,
        "game_day": "2026-05-28",
        "random_days": [
            "2026-04-22",
            "2026-06-05",
            "2026-06-25",
        ],
        "selection_inventory_sha256": (
            "sha256:"
            "74d7e9f21003f595d2d505bc63c89c7eadcd5339d541032bc80704d4b14b3043"
        ),
        "x01_game_day_manifest_sha256": (
            "sha256:"
            "e7dfc9e7992f1eb085edc0c67f37100db10c6541533220833dcace2a1e244df3"
        ),
        "pending_archive_object_count": 72,
    },
    "diff_and_stability_definitions": {
        "diff_ms": "epoch_ms(timestamp_received)-epoch_ms(timestamp)",
        "signed_quantiles": {
            "estimator": "quantile_cont",
            "interpolation": "linear",
            "input_distribution": "integer_millisecond_frequency",
            "probabilities": ["0.50", "0.95", "0.99"],
        },
        "absolute_p99": {
            "transform": "abs(diff_ms)",
            "estimator": "quantile_cont",
            "interpolation": "linear",
            "input_distribution": "integer_millisecond_frequency",
            "probability": "0.99",
        },
        "hourly_drift_ms": (
            "median_utc_hour_23_diff_ms-median_utc_hour_00_diff_ms"
        ),
        "disorder": {
            "partition_by": ["market", "asset_id"],
            "canonical_sort": ["timestamp_received", "timestamp"],
            "numerator": "adjacent_source_timestamp_strict_descents",
            "ordered_comparisons": (
                "n_rows-n_unique_market_asset_streams"
            ),
        },
        "downgrade_gate": {
            "negative_rate_gte": "0.001",
            "absolute_p99_ms_gt": 5000,
            "decision": "downgrade_millisecond_research_to_seconds",
        },
    },
    "h_split_approval": {
        "split": "n.a.",
        "basis": "charter_section_9_measurement_audit_exemption",
        "approved_by": "H",
    },
}
_X02_TIMESTAMP_INPUT_MANIFEST_BINDING = {
    "bundle_path": (
        "artifacts/data-audit/x02_timestamp_input_bundle_v1.json"
    ),
    "bundle_file_sha256": (
        "sha256:"
        "46c3f23007929ad31b131f3618009810501b6ef06d0ac2645eb3dcfac217bd8d"
    ),
    "bundle_sha256": (
        "sha256:"
        "9477f8d9a224b47b6dda47dd691761d4ca8b5d88be6ad07416217a2bb44c89a4"
    ),
}
_X02_DAY_MANIFEST_BINDINGS = [
    {
        "artifact_file_sha256": (
            "sha256:"
            "36d20e073a41ed4cbe0a2e7f37e726c3334075a0ff4017d42b38d5e15d4a5e85"
        ),
        "day": "2026-04-22",
        "full_day_manifest_sha256": (
            "sha256:"
            "d01571e413a942c615489bc674560a0abc5a718507d990ad4106c1a90f3f61ed"
        ),
        "object_count": 24,
        "path": (
            "artifacts/data-audit/"
            "x02_full_day_input_manifest_2026-04-22_v1.json"
        ),
    },
    {
        "artifact_file_sha256": (
            "sha256:"
            "06f1c0aeadd0a735474756e272fd4f9cccb8d718ca65c316163ff08361ac22c5"
        ),
        "day": "2026-05-28",
        "full_day_manifest_sha256": (
            "sha256:"
            "e7dfc9e7992f1eb085edc0c67f37100db10c6541533220833dcace2a1e244df3"
        ),
        "object_count": 24,
        "path": "artifacts/data-audit/x01_full_day_input_manifest_v1.json",
    },
    {
        "artifact_file_sha256": (
            "sha256:"
            "ddb609d9bf15550e487caa3e785999340b4fa337ab82036eccec09965aa598f8"
        ),
        "day": "2026-06-05",
        "full_day_manifest_sha256": (
            "sha256:"
            "b4f688f1da57827426efd5b0b212b15457282f56a253ceffca74542d11f6419e"
        ),
        "object_count": 24,
        "path": (
            "artifacts/data-audit/"
            "x02_full_day_input_manifest_2026-06-05_v1.json"
        ),
    },
    {
        "artifact_file_sha256": (
            "sha256:"
            "94546f473149f31ef75179f8a1e8982fc31d99e712ba9d5c3b94cf603b458850"
        ),
        "day": "2026-06-25",
        "full_day_manifest_sha256": (
            "sha256:"
            "0b6e87e6b59035751405b5d87a8b9bde2902f0e5545163e0eb17c30a1f3f956e"
        ),
        "object_count": 24,
        "path": (
            "artifacts/data-audit/"
            "x02_full_day_input_manifest_2026-06-25_v1.json"
        ),
    },
]
_X02_TIMESTAMP_INPUT_BUNDLE = {
    "additional_days": [
        "2026-04-22",
        "2026-06-05",
        "2026-06-25",
    ],
    "bundle_sha256": _X02_TIMESTAMP_INPUT_MANIFEST_BINDING[
        "bundle_sha256"
    ],
    "day_count": 4,
    "day_manifests": _X02_DAY_MANIFEST_BINDINGS,
    "formal_result": False,
    "inventory_path": "artifacts/data-audit/phase0_inventory.json",
    "inventory_sha256": (
        "sha256:"
        "74d7e9f21003f595d2d505bc63c89c7eadcd5339d541032bc80704d4b14b3043"
    ),
    "object_count": 96,
    "purpose": "frozen_input_only_before_X02_evaluation",
    "selection_procedure": (
        "x01_day_plus_random.Random(seed).sample("
        "sorted_exact_complete_utc_days_excluding_x01,3);"
        "additional_days_sorted"
    ),
    "selection_seed": 20260722,
    "version": "x02-timestamp-input-bundle-v1",
    "x01_day": "2026-05-28",
}
_X02_REGISTERED_INPUT_ARTIFACTS = {
    _X02_TIMESTAMP_INPUT_MANIFEST_BINDING["bundle_path"],
    *(
        item["path"]
        for item in _X02_DAY_MANIFEST_BINDINGS
        if item["day"] != "2026-05-28"
    ),
}
_X02_STATIC_SIDECAR_ROOT = (
    "artifacts/data-audit/x02-static-store-v1"
)
_ALLOWED_STATUSES = frozenset(
    {"registered", "running", "done", "failed", "abandoned"}
)
_PROGRAM_NO_GOS = frozenset(
    {
        "real_money_execution",
        "maker_strategy_live",
        "exact_queue_fill_claim_from_pmxt_l2",
        "multi_venue_simultaneous_execution_arbitrage",
        "live_copy_trading",
        "llm_hot_path",
        "reinforcement_learning",
        "large_scale_microservices",
        "readme_return_strategy_selection",
        "unregistered_quick_backtest",
    }
)

# This mapping is the in-runtime trust anchor for the initial program registration.
# The append-only Git history is the external monotonic anchor for later ledger rows.
_TRUSTED_BASE_REGISTRATIONS = {
    "X-01": "sha256:da901afeac387a054c4e8f64cb928bb96df45919af5b5b696639b44ccbd860d3",
    "X-02": "sha256:1b6393ef8cca4bf482cc3a167844358c07fda9a97c45cbedbe4ceda3e2033ed1",
    "X-03": "sha256:7634916448695c08bf66826c217cce0b8d1f67f32bc85c163c5de433bcd07962",
    "X-04": "sha256:5e2265e8d9d992a27024a4af5914fd94bdb172efb97b2a9d819b9b4f7aafedb9",
    "X-05": "sha256:174e6422e4531aaff8a28daa9a7ff254b177f6685e895cf8fb286d21081f463b",
    "X-06": "sha256:8e487785dc877f70fbdb8b7bdb3825e70ebdd950af474512635429cc6b6c1c34",
    "X-07": "sha256:44acdd9db4be6dd50b375384fe419dcaef631f381d37a054152f85e26a3295d5",
    "X-08": "sha256:e4d9f0a72ac6dcb4ad0e78859bf0264eb9a79508987263428678bf4342b97e8e",
    "X-09": "sha256:b72b5395b37a189150c00f2b278c713942fe78b41da4b68dd5beef8d3cc0160b",
    "X-10": "sha256:bb1fc8aca25e10250bdef744b682788afab1b83357855b7c6a6231086623911a",
    "X-11": "sha256:1706e20201346560f38b4bf1ab3f040c8318f871d809eeff03666827b1b5ec4e",
    "X-12": "sha256:f1482d5268cbcb556ae8ac8fb37f15d31586cbc2e13565cb05f0341efcabdc96",
}

_COMMON_CARD_FIELDS = frozenset(
    {
        "id",
        "name",
        "owner_team",
        "status",
        "hypothesis",
        "data",
        "method",
        "leakage_checks",
        "split",
        "metrics",
        "pass_criteria",
        "fail_criteria",
        "cost_estimate",
        "dependencies",
        "registered_at",
        "result_acceptance_not_before",
        "due_gate",
        "results_ref",
        "promotion_decision",
        "execution_authorized",
        "completion_required_scopes",
        "authorization_scopes",
        "registration_locks",
        "measurement_exemption",
        "falsified_direction_is_valid_measurement",
        "linked_first_artifact_due_gates",
        "source_lineage",
        "program_no_go_restrictions",
        "amendments",
        "registration_record_sha256",
    }
)
_OPTIONAL_CARD_FIELDS = {
    "X-01": {"deterministic_replay_required_levels"},
    "X-02": set(),
    "X-03": set(),
    "X-04": set(),
    "X-05": {"artifact_dependencies", "midpoint_allowed"},
    "X-06": {
        "decision_gates",
        "dataset_ids",
        "output_contract",
        "synthetic_fixture",
    },
    "X-07": {"midpoint_allowed"},
    "X-08": {"prospective_observation", "unresolved_decision_band"},
    "X-09": {"artifact_dependencies", "deterministic_replay_required_levels", "signal"},
    "X-10": {"recall_denominator_registered"},
    "X-11": {"dataset_ids", "output_contract", "tie_policy"},
    "X-12": {"dataset_ids", "output_contract", "promotion_restriction"},
}
_EXPERIMENT_REGISTRY_FIELDS = [
    "experiment_id",
    "card_path",
    "owner_team",
    "status",
    "execution_authorized",
    "registered_at",
    "due_gate",
    "card_sha256",
]
_CATALOG_FIELDS = [
    "catalog_item_id",
    "source_catalog_id",
    "catalog",
    "title",
    "primary_team",
    "secondary_teams",
    "priority",
    "program_stage",
    "first_artifact",
    "linked_experiments",
    "status",
    "due_gate",
]
_LEDGER_FIELDS = [
    "experiment_id",
    "sequence",
    "record_sha256",
    "prior_sha256",
    "amended_at",
    "approved_by",
    "reason",
]
_ARTIFACT_REGISTRY_FIELDS = [
    "artifact_id",
    "path",
    "owner_team",
    "version",
    "due_gate",
    "status",
]
_REQUIRED_TASK3_ARTIFACTS = frozenset(
    {
        "registries/experiment_registry.csv",
        "registries/experiment_amendment_ledger.csv",
        "registries/artifact_registry.csv",
        "artifacts/validation/validation_standard_v0.md",
        "src/prediction_market/experiments.py",
        "tests/test_experiment_registry.py",
        "contracts/model-output/v1.schema.yaml",
        "registries/dataset_registry.csv",
        "registries/model_registry.csv",
        *(f"registries/experiments/X-{number:02d}.yaml" for number in range(1, 13)),
    }
)
_RESULT_FIELDS = frozenset(
    {
        "scope",
        "result_label",
        "evaluation_started_at",
        "code_sha256",
        "data_sha256",
        "result_sha256",
        "registration_head_sha256",
        "dataset_ids",
        "model_ids",
    }
)
_INPUT_BOUND_EXPERIMENT_IDS = frozenset(
    {"X-01", "X-02", "X-03", "X-06", "X-08", "X-11", "X-12"}
)


class ExperimentRegistryError(ValueError):
    """The registry is malformed, inconsistent, or has been altered."""


class UnregisteredExperimentError(ExperimentRegistryError):
    """A result refers to an experiment without a preregistration."""


class InvalidResultReferenceError(ExperimentRegistryError):
    """A result reference is incomplete or inconsistent with preregistration."""


class PreRegistrationEvaluationError(ExperimentRegistryError):
    """Evaluation began before its controlling registration."""


class UnauthorizedResultScopeError(ExperimentRegistryError):
    """The requested result scope is absent or not authorized."""


class UnresolvedDependencyError(ExperimentRegistryError):
    """A preregistered dependency lacks completed, validated evidence."""


class UnresolvedRegistrationLockError(ExperimentRegistryError):
    """A scope depends on a registration choice that remains unresolved."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise ExperimentRegistryError("YAML object keys must be canonical strings")
        if key in result:
            raise ExperimentRegistryError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _canonical_value(value: Any, path: str = "$") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is list:
        return [_canonical_value(item, f"{path}[]") for item in value]
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ExperimentRegistryError(
                    f"non-canonical object key at {path}: keys must be strings"
                )
            result[key] = _canonical_value(item, f"{path}.{key}")
        return result
    raise ExperimentRegistryError(
        f"non-canonical value at {path}: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    canonical = _canonical_value(value)
    rendered = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return rendered.encode("utf-8")


def compute_registration_record_sha256(card: dict[str, Any]) -> str:
    """Hash the immutable base card, excluding its digest and amendment suffix."""

    if type(card) is not dict:
        raise ExperimentRegistryError("registration must be a canonical plain object")
    base = {
        key: value
        for key, value in card.items()
        if key not in {"registration_record_sha256", "amendments"}
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(base)).hexdigest()


def compute_amendment_sha256(amendment: dict[str, Any]) -> str:
    if type(amendment) is not dict:
        raise ExperimentRegistryError("amendment must be a canonical plain object")
    content = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(content)).hexdigest()


def _strict_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_fields:
                raise ExperimentRegistryError(
                    f"unexpected CSV columns in {path}: {reader.fieldnames!r}"
                )
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or set(row) != set(expected_fields):
                    raise ExperimentRegistryError(
                        f"malformed CSV row {line_number} in {path}"
                    )
                if any(type(value) is not str for value in row.values()):
                    raise ExperimentRegistryError(
                        f"malformed CSV row {line_number} in {path}"
                    )
                rows.append(row)  # type: ignore[arg-type]
            return rows
    except ExperimentRegistryError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExperimentRegistryError(f"cannot read registry CSV: {path}") from exc


def _safe_file(root: Path, relative: str, purpose: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ExperimentRegistryError(f"invalid {purpose} path")
    lexical = root / relative
    try:
        root_resolved = root.resolve(strict=True)
        current = root_resolved
        relative_parts = Path(relative).parts
        if ".." in relative_parts:
            raise ExperimentRegistryError(f"{purpose} path escape")
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise ExperimentRegistryError(f"{purpose} path is a symlink")
        resolved = lexical.resolve(strict=True)
    except ExperimentRegistryError:
        raise
    except OSError as exc:
        raise ExperimentRegistryError(f"missing {purpose}: {relative}") from exc
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ExperimentRegistryError(f"{purpose} path escape or non-file")
    return resolved


def _safe_directory(root: Path, relative: str, purpose: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ExperimentRegistryError(f"invalid {purpose} path")
    lexical = root / relative
    try:
        root_resolved = root.resolve(strict=True)
        current = root_resolved
        relative_parts = Path(relative).parts
        if ".." in relative_parts:
            raise ExperimentRegistryError(f"{purpose} path escape")
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise ExperimentRegistryError(f"{purpose} path is a symlink")
        resolved = lexical.resolve(strict=True)
    except ExperimentRegistryError:
        raise
    except OSError as exc:
        raise ExperimentRegistryError(f"missing {purpose}: {relative}") from exc
    if not resolved.is_relative_to(root_resolved) or not resolved.is_dir():
        raise ExperimentRegistryError(f"{purpose} path escape or non-directory")
    return resolved


def _validate_card_inventory(root: Path) -> None:
    directory = root / "registries" / "experiments"
    try:
        if directory.is_symlink():
            raise ExperimentRegistryError("experiment card inventory directory is a symlink")
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_dir():
            raise ExperimentRegistryError("experiment card inventory path escape")
        names = {entry.name for entry in resolved.iterdir()}
    except ExperimentRegistryError:
        raise
    except OSError as exc:
        raise ExperimentRegistryError("cannot enumerate experiment card inventory") from exc
    expected = {f"{experiment_id}.yaml" for experiment_id in EXPERIMENT_IDS}
    if names != expected:
        raise ExperimentRegistryError("experiment card inventory must be exactly X-01 through X-12")


def _validate_artifact_registry(root: Path) -> None:
    path = _safe_file(root, "registries/artifact_registry.csv", "artifact registry")
    rows = _strict_csv(path, _ARTIFACT_REGISTRY_FIELDS)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in rows:
        artifact_id = _nonempty_string(row["artifact_id"], "artifact id")
        artifact_path = _nonempty_string(row["path"], "artifact path")
        if artifact_id in seen_ids or artifact_path in seen_paths:
            raise ExperimentRegistryError("duplicate artifact id or path")
        seen_ids.add(artifact_id)
        seen_paths.add(artifact_path)
        for field_name in ("owner_team", "version", "due_gate", "status"):
            _nonempty_string(row[field_name], f"artifact {field_name}")
    missing = sorted(_REQUIRED_TASK3_ARTIFACTS - seen_paths)
    if missing:
        raise ExperimentRegistryError(
            "artifact registry misses Task 3 coverage: " + ", ".join(missing)
        )


def _load_yaml_card(raw: bytes, experiment_id: str) -> dict[str, Any]:
    try:
        document = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except ExperimentRegistryError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ExperimentRegistryError(f"cannot read card {experiment_id}") from exc
    if type(document) is not dict or set(document) != {"experiment_card"}:
        raise ExperimentRegistryError(f"{experiment_id}: invalid card document")
    card = document["experiment_card"]
    if type(card) is not dict:
        raise ExperimentRegistryError(f"{experiment_id}: invalid experiment_card")
    _canonical_value(card)
    return card


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ExperimentRegistryError(f"{label} has unexpected or missing keys")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ExperimentRegistryError(f"{label} must be a non-empty canonical string")
    return value


def _sha256(value: Any, label: str, result_error: bool = False) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        if result_error:
            raise InvalidResultReferenceError(f"invalid {label}")
        raise ExperimentRegistryError(f"invalid {label}")
    return value


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def unique_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ExperimentRegistryError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            value[key] = item
        return value

    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
        )
    except ExperimentRegistryError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRegistryError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if type(document) is not dict:
        raise ExperimentRegistryError(f"{label} must be a JSON object")
    return document


def _verify_x02_static_manifest_sidecar(
    program_root: Path,
    hourly_object: Any,
) -> str:
    object_parts = PurePosixPath(hourly_object.object_path).parts
    object_digest = hourly_object.object_sha256.removeprefix("sha256:")
    expected_partition = f"partition=hour-{hourly_object.hour[:13]}"
    if (
        len(object_parts) != 6
        or object_parts[:4]
        != (
            "raw",
            "source=pmxt",
            "dataset=DS-PMXT-V2",
            "version=v2",
        )
        or object_parts[4] != expected_partition
        or object_parts[5] != f"{object_digest}.parquet"
    ):
        raise ExperimentRegistryError(
            "X-02: hourly object path/digest/partition cannot derive a "
            "canonical static manifest sidecar"
        )
    sidecar_digest = hourly_object.static_manifest_sha256.removeprefix(
        "sha256:"
    )
    sidecar_ref = (
        PurePosixPath(_X02_STATIC_SIDECAR_ROOT)
        / "manifests"
        / object_parts[1]
        / object_parts[2]
        / object_parts[3]
        / object_parts[4]
        / f"{sidecar_digest}.manifest.json"
    ).as_posix()
    sidecar_path = _safe_file(
        program_root,
        sidecar_ref,
        "X-02 static manifest sidecar",
    )
    sidecar_raw = sidecar_path.read_bytes()
    document = _decode_json_object(
        sidecar_raw,
        "X-02 static manifest sidecar",
    )
    from prediction_market.contracts import (
        StaticDatasetManifestV0,
        canonical_json_bytes,
        validate_static_dataset_manifest_v0,
    )

    try:
        canonical = canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError) as exc:
        raise ExperimentRegistryError(
            "X-02: static manifest sidecar is not canonical JSON"
        ) from exc
    if sidecar_raw != canonical:
        raise ExperimentRegistryError(
            "X-02: static manifest sidecar is not canonical JSON"
        )
    try:
        sidecar = validate_static_dataset_manifest_v0(
            program_root,
            document,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentRegistryError(
            f"X-02: static manifest sidecar contract is invalid: {exc}"
        ) from exc
    if not isinstance(sidecar, StaticDatasetManifestV0):
        raise ExperimentRegistryError(
            "X-02: static manifest sidecar validator returned invalid data"
        )
    if (
        sidecar.manifest_sha256
        != hourly_object.static_manifest_sha256
        or sidecar.native_object_path != hourly_object.object_path
        or sidecar.object_sha256 != hourly_object.object_sha256
        or sidecar.source_url != hourly_object.source_url
        or sidecar.dataset_id != "DS-PMXT-V2"
        or sidecar.license_ref != "O-006"
        or sidecar.license_status != "approved"
        or sidecar.upstream_partition
        != expected_partition.removeprefix("partition=")
        or sidecar.object_kind != "byte_exact_original"
    ):
        raise ExperimentRegistryError(
            "X-02: static manifest sidecar does not match its day-manifest "
            "object, dataset, or license binding"
        )
    return sidecar_ref


def _verify_x02_timestamp_input_bundle(
    program_root: str | Path,
    binding: Any,
) -> dict[str, Any]:
    """Verify X-02's frozen four-day bundle without evaluating its data."""

    binding = _exact_keys(
        binding,
        {
            "bundle_path",
            "bundle_file_sha256",
            "bundle_sha256",
        },
        "X-02 timestamp input manifest binding",
    )
    bundle_ref = _nonempty_string(
        binding["bundle_path"],
        "X-02 timestamp input bundle path",
    )
    _sha256(
        binding["bundle_file_sha256"],
        "X-02 timestamp input bundle file SHA-256",
    )
    _sha256(
        binding["bundle_sha256"],
        "X-02 timestamp input bundle self-hash",
    )
    root = Path(program_root)
    bundle_path = _safe_file(
        root,
        bundle_ref,
        "X-02 timestamp input bundle",
    )
    bundle_raw = bundle_path.read_bytes()
    actual_file_sha256 = (
        "sha256:" + hashlib.sha256(bundle_raw).hexdigest()
    )
    if actual_file_sha256 != binding["bundle_file_sha256"]:
        raise ExperimentRegistryError(
            "X-02: timestamp input bundle file SHA-256 mismatch"
        )
    document = _decode_json_object(
        bundle_raw,
        "X-02 timestamp input bundle",
    )
    document = _exact_keys(
        document,
        set(_X02_TIMESTAMP_INPUT_BUNDLE),
        "X-02 timestamp input bundle",
    )
    material = dict(document)
    material.pop("bundle_sha256")
    actual_bundle_sha256 = (
        "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()
    )
    if (
        document["bundle_sha256"] != actual_bundle_sha256
        or binding["bundle_sha256"] != actual_bundle_sha256
    ):
        raise ExperimentRegistryError(
            "X-02: timestamp input bundle self-hash mismatch"
        )

    for key, expected in _X02_TIMESTAMP_INPUT_BUNDLE.items():
        if key in {"bundle_sha256", "day_manifests"}:
            continue
        if document[key] != expected:
            raise ExperimentRegistryError(
                "X-02: timestamp input bundle selection/day/object/formal "
                "contract mismatch"
            )

    day_entries = document["day_manifests"]
    if type(day_entries) is not list or len(day_entries) != 4:
        raise ExperimentRegistryError(
            "X-02: timestamp input bundle must contain four day manifests"
        )
    expected_by_path = {
        item["path"]: item for item in _X02_DAY_MANIFEST_BINDINGS
    }
    if [
        item.get("path") if type(item) is dict else None
        for item in day_entries
    ] != [item["path"] for item in _X02_DAY_MANIFEST_BINDINGS]:
        raise ExperimentRegistryError(
            "X-02: timestamp input day manifest paths or order mismatch"
        )

    artifact_rows = _strict_csv(
        _safe_file(
            root,
            "registries/artifact_registry.csv",
            "artifact registry",
        ),
        _ARTIFACT_REGISTRY_FIELDS,
    )
    registered_by_path = {row["path"]: row for row in artifact_rows}
    for artifact_ref in _X02_REGISTERED_INPUT_ARTIFACTS:
        row = registered_by_path.get(artifact_ref)
        if row is None or {
            "owner_team": row["owner_team"],
            "version": row["version"],
            "due_gate": row["due_gate"],
            "status": row["status"],
        } != {
            "owner_team": "C+H",
            "version": "v1",
            "due_gate": "2026-08-05_W2_review",
            "status": "registered",
        }:
            raise ExperimentRegistryError(
                "X-02: input bundle and three new day manifests must be "
                "registered C+H v1 input artifacts"
            )

    from prediction_market.pmxt.full_day import (
        FullDayInputError,
        FullDayManifest,
        LockedHourlyObject,
        validate_full_day_manifest,
    )

    days: list[str] = []
    static_manifest_refs: list[str] = []
    verified_static_sidecar_refs: list[str] = []
    for entry_value in day_entries:
        entry = _exact_keys(
            entry_value,
            {
                "artifact_file_sha256",
                "day",
                "full_day_manifest_sha256",
                "object_count",
                "path",
            },
            "X-02 timestamp input day manifest binding",
        )
        day_ref = _nonempty_string(
            entry["path"],
            "X-02 timestamp input day manifest path",
        )
        expected_entry = expected_by_path[day_ref]
        day_path = _safe_file(
            root,
            day_ref,
            "X-02 timestamp input day manifest",
        )
        day_raw = day_path.read_bytes()
        actual_day_file_sha256 = (
            "sha256:" + hashlib.sha256(day_raw).hexdigest()
        )
        if actual_day_file_sha256 != entry["artifact_file_sha256"]:
            raise ExperimentRegistryError(
                "X-02: timestamp input day manifest file SHA-256 mismatch"
            )
        day_document = _decode_json_object(
            day_raw,
            "X-02 timestamp input day manifest",
        )
        day_document = _exact_keys(
            day_document,
            {
                "canonicalization_version",
                "day",
                "inventory_sha256",
                "manifest_sha256",
                "objects",
                "version",
            },
            "X-02 timestamp input day manifest",
        )
        object_values = day_document["objects"]
        if type(object_values) is not list:
            raise ExperimentRegistryError(
                "X-02: timestamp input day objects must be a list"
            )
        locked_objects: list[LockedHourlyObject] = []
        for object_value in object_values:
            item = _exact_keys(
                object_value,
                {
                    "hour",
                    "inventory_size_bytes",
                    "object_path",
                    "object_sha256",
                    "source_url",
                    "static_manifest_sha256",
                },
                "X-02 timestamp input hourly object",
            )
            locked_objects.append(LockedHourlyObject(**item))
        manifest = FullDayManifest(
            version=day_document["version"],
            day=day_document["day"],
            inventory_sha256=day_document["inventory_sha256"],
            canonicalization_version=day_document[
                "canonicalization_version"
            ],
            objects=tuple(locked_objects),
            manifest_sha256=day_document["manifest_sha256"],
        )
        try:
            validate_full_day_manifest(manifest)
        except (FullDayInputError, TypeError) as exc:
            raise ExperimentRegistryError(
                f"X-02: invalid timestamp input day manifest: {exc}"
            ) from exc
        verified_static_sidecar_refs.extend(
            _verify_x02_static_manifest_sidecar(root, item)
            for item in manifest.objects
        )
        if (
            manifest.day != entry["day"]
            or manifest.manifest_sha256
            != entry["full_day_manifest_sha256"]
            or len(manifest.objects) != entry["object_count"]
            or manifest.inventory_sha256
            != _X02_TIMESTAMP_INPUT_BUNDLE["inventory_sha256"]
        ):
            raise ExperimentRegistryError(
                "X-02: timestamp input day manifest binding mismatch"
            )
        if entry != expected_entry:
            raise ExperimentRegistryError(
                "X-02: timestamp input day manifest does not match the "
                "preregistered four-day bundle"
            )
        days.append(manifest.day)
        static_manifest_refs.extend(
            item.static_manifest_sha256 for item in manifest.objects
        )

    if (
        days
        != [
            "2026-04-22",
            "2026-05-28",
            "2026-06-05",
            "2026-06-25",
        ]
        or len(static_manifest_refs) != 96
        or len(set(static_manifest_refs)) != 96
        or len(verified_static_sidecar_refs) != 96
        or len(set(verified_static_sidecar_refs)) != 96
        or any(
            _SHA256_RE.fullmatch(reference) is None
            for reference in static_manifest_refs
        )
    ):
        raise ExperimentRegistryError(
            "X-02: timestamp input bundle must bind 96 distinct valid "
            "static manifest references across the exact four days"
        )
    return {
        "bundle_sha256": actual_bundle_sha256,
        "days": days,
        "object_count": sum(
            entry["object_count"] for entry in day_entries
        ),
        "static_manifest_reference_count": len(static_manifest_refs),
        "verified_static_sidecar_count": len(
            verified_static_sidecar_refs
        ),
    }


def _canonical_id_list(
    value: Any,
    *,
    pattern: re.Pattern[str],
    label: str,
    result_error: bool = False,
) -> list[str]:
    error_type = (
        InvalidResultReferenceError if result_error else ExperimentRegistryError
    )
    if (
        type(value) is not list
        or any(type(item) is not str or pattern.fullmatch(item) is None for item in value)
        or len(value) != len(set(value))
        or value != sorted(value)
    ):
        raise error_type(f"{label} must be a canonical sorted unique ID list")
    return list(value)


def _canonical_utc(
    value: Any, label: str, *, result_error: bool = False
) -> datetime:
    error_type = InvalidResultReferenceError if result_error else ExperimentRegistryError
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise error_type(f"{label} must be a canonical UTC timestamp ending in Z")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise error_type(f"{label} must be a canonical UTC timestamp") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _verified_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ExperimentRegistryError(f"{label} is not a valid UTC instant") from exc
    if not value.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ExperimentRegistryError(f"{label} is not a valid UTC instant")
    return parsed


def _lock_map(card: dict[str, Any], experiment_id: str) -> dict[str, dict[str, Any]]:
    locks = card["registration_locks"]
    if type(locks) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: registration_locks must be a list")
    result: dict[str, dict[str, Any]] = {}
    for lock in locks:
        lock = _exact_keys(lock, {"id", "status", "reason"}, f"{experiment_id} lock")
        lock_id = _nonempty_string(lock["id"], f"{experiment_id} lock id")
        if lock_id in result:
            raise ExperimentRegistryError(f"{experiment_id}: duplicate lock {lock_id}")
        if lock["status"] not in {"unresolved", "resolved"}:
            raise ExperimentRegistryError(f"{experiment_id}: invalid lock status")
        _nonempty_string(lock["reason"], f"{experiment_id} lock reason")
        result[lock_id] = lock
    return result


def _validate_scopes(
    card: dict[str, Any], experiment_id: str, locks: dict[str, dict[str, Any]]
) -> None:
    scopes = card["authorization_scopes"]
    if type(scopes) is not dict or not scopes:
        raise ExperimentRegistryError(f"{experiment_id}: invalid authorization_scopes")
    for scope_name, raw_scope in scopes.items():
        _nonempty_string(scope_name, f"{experiment_id} scope")
        if type(raw_scope) is not dict:
            raise ExperimentRegistryError(f"{experiment_id}: invalid scope {scope_name}")
        allowed = {
            "authorized",
            "required_result_label",
            "required_lock_ids",
        }
        if "input_binding" in raw_scope:
            allowed.add("input_binding")
        if "permanent_no_go" in raw_scope:
            allowed.add("permanent_no_go")
        scope = _exact_keys(raw_scope, allowed, f"{experiment_id} scope {scope_name}")
        if type(scope["authorized"]) is not bool:
            raise ExperimentRegistryError(f"{experiment_id}: scope authorization must be bool")
        if scope["required_result_label"] not in {"FORMAL", "PRELIMINARY"}:
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} has invalid required_result_label"
            )
        required = scope["required_lock_ids"]
        if type(required) is not list or any(type(item) is not str for item in required):
            raise ExperimentRegistryError(f"{experiment_id}: invalid scope locks")
        if len(required) != len(set(required)):
            raise ExperimentRegistryError(f"{experiment_id}: duplicate scope lock")
        unknown = sorted(set(required) - set(locks))
        if unknown:
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} references unknown registration lock: "
                + ", ".join(unknown)
            )
        permanent = scope.get("permanent_no_go", False)
        if type(permanent) is not bool:
            raise ExperimentRegistryError(
                f"{experiment_id}: permanent_no_go must be boolean"
            )
        if permanent and scope["authorized"]:
            raise ExperimentRegistryError(
                f"{experiment_id}: permanent NO-GO scope cannot be authorized"
            )
        binding = scope.get("input_binding")
        if experiment_id in _INPUT_BOUND_EXPERIMENT_IDS:
            binding = _exact_keys(
                binding,
                {
                    "result_class",
                    "dataset_ids",
                    "model_ids",
                    "synthetic_data_sha256",
                },
                f"{experiment_id} scope {scope_name} input_binding",
            )
            if binding["result_class"] not in {"formal", "poc", "synthetic"}:
                raise ExperimentRegistryError(
                    f"{experiment_id}: scope {scope_name} invalid result_class"
                )
            datasets = _canonical_id_list(
                binding["dataset_ids"],
                pattern=_DATASET_ID_RE,
                label=f"{experiment_id} scope {scope_name} dataset_ids",
            )
            models = _canonical_id_list(
                binding["model_ids"],
                pattern=_MODEL_ID_RE,
                label=f"{experiment_id} scope {scope_name} model_ids",
            )
            synthetic_hash = binding["synthetic_data_sha256"]
            if binding["result_class"] == "synthetic":
                if datasets:
                    raise ExperimentRegistryError(
                        f"{experiment_id}: synthetic scope cannot bind datasets"
                    )
                _sha256(
                    synthetic_hash,
                    f"{experiment_id} scope {scope_name} synthetic_data_sha256",
                )
            elif synthetic_hash is not None:
                raise ExperimentRegistryError(
                    f"{experiment_id}: non-synthetic scope cannot bind synthetic data"
                )
        elif binding is not None:
            raise ExperimentRegistryError(
                f"{experiment_id}: unexpected input_binding"
            )


def _validate_card_structure(card: dict[str, Any], experiment_id: str) -> None:
    expected = _COMMON_CARD_FIELDS | _OPTIONAL_CARD_FIELDS[experiment_id]
    if set(card) != expected:
        raise ExperimentRegistryError(
            f"{experiment_id}: card has unexpected or missing keys"
        )
    if card["id"] != experiment_id:
        raise ExperimentRegistryError(f"{experiment_id}: filename/card id mismatch")
    for field_name in (
        "name",
        "owner_team",
        "hypothesis",
        "method",
        "split",
        "pass_criteria",
        "fail_criteria",
    ):
        _nonempty_string(card[field_name], f"{experiment_id}.{field_name}")
    if card["status"] not in _ALLOWED_STATUSES:
        raise ExperimentRegistryError(f"{experiment_id}: invalid status")
    if type(card["registered_at"]) is not str or _DATE_RE.fullmatch(card["registered_at"]) is None:
        raise ExperimentRegistryError(f"{experiment_id}: invalid registered_at")
    if card["registered_at"] != "2026-07-22":
        raise ExperimentRegistryError(f"{experiment_id}: immutable registration date changed")
    if card["result_acceptance_not_before"] != _RESULT_ACCEPTANCE_NOT_BEFORE:
        raise ExperimentRegistryError(f"{experiment_id}: immutable preregistration boundary changed")
    if card["due_gate"] is not None:
        raise ExperimentRegistryError(f"{experiment_id}: due_gate must be null")
    if type(card["execution_authorized"]) is not bool:
        raise ExperimentRegistryError(f"{experiment_id}: execution_authorized must be bool")
    if type(card["measurement_exemption"]) is not bool or type(
        card["falsified_direction_is_valid_measurement"]
    ) is not bool:
        raise ExperimentRegistryError(f"{experiment_id}: invalid measurement flags")
    if type(card["promotion_decision"]) is not str:
        raise ExperimentRegistryError(f"{experiment_id}: invalid promotion_decision")
    if type(card["results_ref"]) is not list or card["results_ref"]:
        raise ExperimentRegistryError(f"{experiment_id}: immutable base results_ref must be []")
    if type(card["data"]) is not list or not card["data"]:
        raise ExperimentRegistryError(f"{experiment_id}: data must be non-empty")
    for index, item in enumerate(card["data"]):
        item = _exact_keys(
            item, {"source", "version", "pit_basis"}, f"{experiment_id} data[{index}]"
        )
        for field_name in ("source", "version", "pit_basis"):
            _nonempty_string(item[field_name], f"{experiment_id} data[{index}].{field_name}")
    for field_name in ("leakage_checks", "metrics", "dependencies"):
        value = card[field_name]
        if type(value) is not list or any(type(item) is not str or not item for item in value):
            raise ExperimentRegistryError(f"{experiment_id}: malformed {field_name}")
        if len(value) != len(set(value)):
            raise ExperimentRegistryError(f"{experiment_id}: duplicate {field_name}")
    cost = _exact_keys(card["cost_estimate"], {"compute", "human_days"}, f"{experiment_id} cost")
    _nonempty_string(cost["compute"], f"{experiment_id} cost compute")
    _nonempty_string(cost["human_days"], f"{experiment_id} cost human_days")
    locks = _lock_map(card, experiment_id)
    _validate_scopes(card, experiment_id, locks)
    completion_scopes = card["completion_required_scopes"]
    if (
        type(completion_scopes) is not list
        or not completion_scopes
        or any(type(scope) is not str for scope in completion_scopes)
        or len(completion_scopes) != len(set(completion_scopes))
        or not set(completion_scopes).issubset(card["authorization_scopes"])
    ):
        raise ExperimentRegistryError(
            f"{experiment_id}: invalid completion_required_scopes"
        )
    if any(
        card["authorization_scopes"][scope].get("permanent_no_go", False)
        for scope in completion_scopes
    ):
        raise ExperimentRegistryError(
            f"{experiment_id}: permanent NO-GO cannot be a completion scope"
        )
    lineage = _exact_keys(
        card["source_lineage"],
        {"charter_file", "charter_sections", "catalog_item_ids"},
        f"{experiment_id} source_lineage",
    )
    if lineage["charter_file"] != "charter/research_program_charter_v0.2.md":
        raise ExperimentRegistryError(f"{experiment_id}: wrong charter lineage")
    if type(lineage["charter_sections"]) is not list or not lineage["charter_sections"]:
        raise ExperimentRegistryError(f"{experiment_id}: missing charter sections")
    if type(lineage["catalog_item_ids"]) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: malformed catalog lineage")
    gates = card["linked_first_artifact_due_gates"]
    if type(gates) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: malformed catalog gates")
    for gate in gates:
        gate = _exact_keys(gate, {"catalog_item_id", "due_gate"}, f"{experiment_id} gate")
        _nonempty_string(gate["catalog_item_id"], f"{experiment_id} catalog id")
        _nonempty_string(gate["due_gate"], f"{experiment_id} catalog due gate")
    if set(card["program_no_go_restrictions"]) != _PROGRAM_NO_GOS or len(
        card["program_no_go_restrictions"]
    ) != len(_PROGRAM_NO_GOS):
        raise ExperimentRegistryError(f"{experiment_id}: program NO-GO drift")
    if "artifact_dependencies" in card:
        dependencies = card["artifact_dependencies"]
        if type(dependencies) is not list or not dependencies:
            raise ExperimentRegistryError(f"{experiment_id}: invalid artifact dependencies")
        for dependency in dependencies:
            dependency = _exact_keys(
                dependency, {"path", "version", "sha256"}, f"{experiment_id} artifact dependency"
            )
            _nonempty_string(dependency["path"], f"{experiment_id} artifact path")
            _nonempty_string(dependency["version"], f"{experiment_id} artifact version")
            _sha256(dependency["sha256"], f"{experiment_id} artifact SHA-256")
    if "prospective_observation" in card:
        observation = _exact_keys(
            card["prospective_observation"],
            {
                "required_elapsed_days",
                "observed_elapsed_days",
                "fixtures_can_satisfy_elapsed_time",
            },
            f"{experiment_id} prospective observation",
        )
        if (
            observation["required_elapsed_days"] != 7
            or observation["observed_elapsed_days"] != 0
            or type(observation["fixtures_can_satisfy_elapsed_time"]) is not bool
        ):
            raise ExperimentRegistryError(f"{experiment_id}: invalid prospective observation")
    if "synthetic_fixture" in card:
        fixture = _exact_keys(
            card["synthetic_fixture"],
            {"path", "sha256", "data_class"},
            f"{experiment_id} synthetic fixture",
        )
        _nonempty_string(fixture["path"], f"{experiment_id} synthetic fixture path")
        _sha256(fixture["sha256"], f"{experiment_id} synthetic fixture SHA-256")
        if fixture["data_class"] != "synthetic_contract_only":
            raise ExperimentRegistryError(
                f"{experiment_id}: invalid synthetic fixture data_class"
            )
    if "dataset_ids" in card:
        dataset_ids = card["dataset_ids"]
        if (
            type(dataset_ids) is not list
            or not dataset_ids
            or any(
                type(dataset_id) is not str
                or re.fullmatch(r"DS-[A-Z0-9][A-Z0-9-]*", dataset_id) is None
                for dataset_id in dataset_ids
            )
            or len(dataset_ids) != len(set(dataset_ids))
        ):
            raise ExperimentRegistryError(f"{experiment_id}: invalid dataset_ids")
    if "output_contract" in card:
        output = _exact_keys(
            card["output_contract"],
            {"contract", "output_kind", "transition_unit", "state_space"},
            f"{experiment_id} output contract",
        )
        if output["contract"] != "model-output/v1.schema.yaml":
            raise ExperimentRegistryError(f"{experiment_id}: wrong output contract")
        if output["output_kind"] != "state_transition":
            raise ExperimentRegistryError(f"{experiment_id}: output must be state_transition")
        expected_states = {
            "possession": ["home_score", "away_score", "no_score"],
            "drive": ["touchdown", "field_goal", "punt", "turnover", "other"],
            "five_minute_interval": ["home_goal", "away_goal", "no_goal"],
        }
        if output["transition_unit"] not in expected_states or output[
            "state_space"
        ] != expected_states[output["transition_unit"]]:
            raise ExperimentRegistryError(
                f"{experiment_id}: transition unit/state_space mismatch"
            )
    if "tie_policy" in card:
        _nonempty_string(card["tie_policy"], f"{experiment_id} tie_policy")
    if "promotion_restriction" in card:
        if card["promotion_restriction"] != (
            "POC_ONLY_FORMAL_PROMOTION_UNAUTHORIZED"
        ):
            raise ExperimentRegistryError(
                f"{experiment_id}: invalid promotion restriction"
            )
        formal_promotion = card["authorization_scopes"].get(
            "formal_promotion"
        )
        if (
            not isinstance(formal_promotion, dict)
            or formal_promotion["authorized"] is not False
            or formal_promotion.get("permanent_no_go") is not True
        ):
            raise ExperimentRegistryError(
                f"{experiment_id}: formal promotion restriction must be "
                "a permanent NO-GO"
            )
    if type(card["amendments"]) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: amendments must be a list")


def _catalog_gates(root: Path) -> dict[str, list[dict[str, str]]]:
    path = _safe_file(root, "charter/catalog_registry.csv", "catalog registry")
    rows = _strict_csv(path, _CATALOG_FIELDS)
    result = {experiment_id: [] for experiment_id in EXPERIMENT_IDS}
    seen_ids: set[str] = set()
    for row in rows:
        catalog_id = row["catalog_item_id"]
        if not catalog_id or catalog_id in seen_ids:
            raise ExperimentRegistryError("catalog registry has duplicate stable IDs")
        seen_ids.add(catalog_id)
        raw_links = row["linked_experiments"]
        if raw_links == "-" or raw_links == "":
            links: list[str] = []
        else:
            if raw_links != raw_links.strip():
                raise ExperimentRegistryError(f"catalog {catalog_id} link has whitespace")
            links = raw_links.split(";")
            if any(link != link.strip() for link in links):
                raise ExperimentRegistryError(f"catalog {catalog_id} link has whitespace")
            if len(links) != len(set(links)):
                raise ExperimentRegistryError(f"catalog {catalog_id} has duplicate experiment link")
        for experiment_id in links:
            if experiment_id not in _EXPERIMENT_ID_SET:
                raise ExperimentRegistryError(
                    f"catalog {catalog_id} links unknown experiment {experiment_id}"
                )
            result[experiment_id].append(
                {"catalog_item_id": catalog_id, "due_gate": row["due_gate"]}
            )
    return result


def _check_dependency_graph(cards: dict[str, dict[str, Any]]) -> None:
    for experiment_id, card in cards.items():
        unknown = sorted(set(card["dependencies"]) - set(cards))
        if unknown:
            raise ExperimentRegistryError(f"{experiment_id}: unknown dependencies")
        if experiment_id in card["dependencies"]:
            raise ExperimentRegistryError(f"{experiment_id}: self dependency")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in visiting:
            raise ExperimentRegistryError("experiment dependency cycle detected")
        if experiment_id in visited:
            return
        visiting.add(experiment_id)
        for dependency in cards[experiment_id]["dependencies"]:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in cards:
        visit(experiment_id)


def _validate_result_shape(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise InvalidResultReferenceError("result_ref must be a plain dict")
    if set(value) != _RESULT_FIELDS:
        raise InvalidResultReferenceError("result_ref has unexpected or missing fields")
    snapshot: dict[str, Any] = {}
    for field_name in _RESULT_FIELDS - {"dataset_ids", "model_ids"}:
        field_value = value[field_name]
        if type(field_value) is not str:
            raise InvalidResultReferenceError(f"{field_name} must be a string")
        snapshot[field_name] = field_value
    snapshot["dataset_ids"] = _canonical_id_list(
        value["dataset_ids"],
        pattern=_DATASET_ID_RE,
        label="result dataset_ids",
        result_error=True,
    )
    snapshot["model_ids"] = _canonical_id_list(
        value["model_ids"],
        pattern=_MODEL_ID_RE,
        label="result model_ids",
        result_error=True,
    )
    if not snapshot["scope"]:
        raise InvalidResultReferenceError("scope must be non-empty")
    if snapshot["result_label"] not in {"FORMAL", "PRELIMINARY"}:
        raise InvalidResultReferenceError("result_label must be FORMAL or PRELIMINARY")
    _canonical_utc(snapshot["evaluation_started_at"], "evaluation_started_at", result_error=True)
    for field_name in (
        "code_sha256",
        "data_sha256",
        "result_sha256",
        "registration_head_sha256",
    ):
        _sha256(snapshot[field_name], field_name, result_error=True)
    return snapshot


def _validate_reproduction_registration(
    value: Any,
    experiment_id: str,
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if experiment_id not in {"X-11", "X-12"}:
        raise ExperimentRegistryError(
            f"{experiment_id}: register_reproduction is only valid for "
            "X-11 or X-12"
        )
    expected_keys = {
        "reproduction_id",
        "scope",
        "result_class",
        "dataset_ids",
        "model_bindings",
        "code_paths",
        "code_sha256",
        "data_sha256",
        "reproduction_spec_sha256",
    }
    if experiment_id == "X-11":
        expected_keys.update({"protocol_path", "protocol_sha256"})
    registration = _exact_keys(
        value,
        expected_keys,
        f"{experiment_id} reproduction registration",
    )
    _nonempty_string(
        registration["reproduction_id"],
        f"{experiment_id} reproduction_id",
    )
    _nonempty_string(
        registration["scope"],
        f"{experiment_id} reproduction scope",
    )
    if registration["result_class"] != "poc":
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction result_class must be poc"
        )
    dataset_ids = _canonical_id_list(
        registration["dataset_ids"],
        pattern=_DATASET_ID_RE,
        label=f"{experiment_id} reproduction dataset_ids",
    )
    if not dataset_ids:
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction dataset_ids must be non-empty"
        )
    model_bindings = registration["model_bindings"]
    if type(model_bindings) is not list or not model_bindings:
        raise ExperimentRegistryError(
            f"{experiment_id}: model_bindings must be a non-empty list"
        )
    model_ids: list[str] = []
    for item in model_bindings:
        binding = _exact_keys(
            item,
            {
                "model_id",
                "model_version",
                "model_record_sha256",
            },
            f"{experiment_id} reproduction model binding",
        )
        model_id = _nonempty_string(
            binding["model_id"],
            f"{experiment_id} reproduction model_id",
        )
        if _MODEL_ID_RE.fullmatch(model_id) is None:
            raise ExperimentRegistryError(
                f"{experiment_id}: invalid reproduction model_id"
            )
        model_ids.append(model_id)
        _nonempty_string(
            binding["model_version"],
            f"{experiment_id} reproduction model_version",
        )
        _sha256(
            binding["model_record_sha256"],
            f"{experiment_id} model_record_sha256",
        )
    if len(model_ids) != len(set(model_ids)):
        raise ExperimentRegistryError(
            f"{experiment_id}: duplicate reproduction model_id"
        )
    if model_ids != sorted(model_ids):
        raise ExperimentRegistryError(
            f"{experiment_id}: model_bindings must be canonically sorted "
            "by model_id"
        )
    _sha256(
        registration["code_sha256"],
        f"{experiment_id} reproduction code_sha256",
    )
    code_paths = registration["code_paths"]
    if (
        type(code_paths) is not list
        or not code_paths
        or any(type(item) is not str for item in code_paths)
    ):
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction code_paths must be a non-empty "
            "list of strings"
        )
    for code_path in code_paths:
        _nonempty_string(
            code_path,
            f"{experiment_id} reproduction code path",
        )
    if len(code_paths) != len(set(code_paths)):
        raise ExperimentRegistryError(
            f"{experiment_id}: duplicate reproduction code path"
        )
    if code_paths != sorted(code_paths):
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction code_paths must be canonically "
            "sorted"
        )
    _sha256(
        registration["data_sha256"],
        f"{experiment_id} reproduction data_sha256",
    )
    contract = (
        _REPRODUCTION_CONTRACTS[experiment_id]
        if contract is None
        else contract
    )
    if (
        registration["reproduction_id"] != contract["reproduction_id"]
        or registration["scope"] != contract["scope"]
        or dataset_ids != list(contract["dataset_ids"])
        or model_ids != list(contract["model_ids"])
        or code_paths != list(contract["code_paths"])
    ):
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction contract mismatch"
        )
    if experiment_id == "X-11":
        protocol_path = _nonempty_string(
            registration["protocol_path"],
            "X-11 reproduction protocol_path",
        )
        if protocol_path != contract["protocol_path"]:
            raise ExperimentRegistryError(
                "X-11: reproduction contract mismatch"
            )
        _sha256(
            registration["protocol_sha256"],
            "X-11 reproduction protocol_sha256",
        )
    _sha256(
        registration["reproduction_spec_sha256"],
        f"{experiment_id} reproduction_spec_sha256",
    )
    specification = {
        key: item
        for key, item in registration.items()
        if key != "reproduction_spec_sha256"
    }
    expected_spec_sha256 = (
        "sha256:" + hashlib.sha256(_canonical_bytes(specification)).hexdigest()
    )
    if registration["reproduction_spec_sha256"] != expected_spec_sha256:
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction spec hash mismatch"
        )
    return registration


def _model_record_sha256(row: Any) -> str:
    record: dict[str, Any] = {}
    for item in fields(row):
        value = getattr(row, item.name)
        record[item.name] = list(value) if type(value) is tuple else value
    return "sha256:" + hashlib.sha256(_canonical_bytes(record)).hexdigest()


def _reproduction_code_sha256(
    program_root: Path,
    experiment_id: str,
    code_paths: list[str],
) -> str:
    code_objects: list[dict[str, str]] = []
    for code_path in code_paths:
        path = _safe_file(
            program_root,
            code_path,
            f"{experiment_id} reproduction code object",
        )
        code_objects.append(
            {
                "path": code_path,
                "sha256": (
                    "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                ),
            }
        )
    return "sha256:" + hashlib.sha256(
        _canonical_bytes(code_objects)
    ).hexdigest()


def _reproduction_data_sha256(datasets: tuple[Any, ...]) -> str:
    bindings = [
        {
            "dataset_id": dataset.dataset_id,
            "manifest_sha256": dataset.manifest_sha256,
        }
        for dataset in datasets
    ]
    return "sha256:" + hashlib.sha256(
        _canonical_bytes(bindings)
    ).hexdigest()


def _validate_reproduction_protocol(
    program_root: Path,
    experiment_id: str,
    registration: dict[str, Any],
    *,
    contract: dict[str, Any] | None = None,
) -> None:
    contract = (
        _REPRODUCTION_CONTRACTS[experiment_id]
        if contract is None
        else contract
    )
    protocol_path = contract["protocol_path"]
    expected_content_sha256 = contract["protocol_content_sha256"]
    if protocol_path is None:
        return
    path = _safe_file(
        program_root,
        registration["protocol_path"],
        f"{experiment_id} reproduction protocol",
    )
    raw = path.read_bytes()
    actual_file_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    if actual_file_sha256 != registration["protocol_sha256"]:
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction protocol hash mismatch"
        )
    document = _decode_json_object(
        raw,
        f"{experiment_id} reproduction protocol",
    )
    try:
        canonical_protocol = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction protocol is not canonical JSON"
        ) from exc
    actual_content_sha256 = (
        "sha256:" + hashlib.sha256(canonical_protocol).hexdigest()
    )
    if actual_content_sha256 != expected_content_sha256:
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction protocol contract mismatch"
        )


def _validate_reproduction_bindings(
    program_root: Path,
    experiment_id: str,
    registration: dict[str, Any],
    *,
    contract: dict[str, Any] | None = None,
    validate_code: bool = True,
) -> None:
    from prediction_market.program_audit import (
        ResearchRegistryError,
        validate_registered_research_bindings,
    )

    try:
        datasets, models = validate_registered_research_bindings(
            program_root,
            experiment_id=experiment_id,
            dataset_ids=registration["dataset_ids"],
            model_ids=[
                binding["model_id"]
                for binding in registration["model_bindings"]
            ],
            result_class="poc",
        )
    except ResearchRegistryError as exc:
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction bindings are invalid: {exc}"
        ) from exc
    models_by_id = {row.model_id: row for row in models}
    for binding in registration["model_bindings"]:
        model_id = binding["model_id"]
        model = models_by_id[model_id]
        if model.model_version != binding["model_version"]:
            raise ExperimentRegistryError(
                f"{experiment_id}: reproduction model {model_id} version "
                "mismatch"
            )
        if _model_record_sha256(model) != binding["model_record_sha256"]:
            raise ExperimentRegistryError(
                f"{experiment_id}: reproduction model {model_id} record "
                "hash mismatch"
            )
    if validate_code:
        actual_code_sha256 = _reproduction_code_sha256(
            program_root,
            experiment_id,
            registration["code_paths"],
        )
        if registration["code_sha256"] != actual_code_sha256:
            raise ExperimentRegistryError(
                f"{experiment_id}: reproduction code hash mismatch"
            )
    _validate_reproduction_protocol(
        program_root,
        experiment_id,
        registration,
        contract=contract,
    )
    if registration["data_sha256"] != _reproduction_data_sha256(datasets):
        raise ExperimentRegistryError(
            f"{experiment_id}: reproduction data hash mismatch"
        )


def _validate_changes(changes: Any, experiment_id: str) -> dict[str, Any]:
    if type(changes) is not dict or not changes:
        raise ExperimentRegistryError(f"{experiment_id}: amendment changes must be non-empty")
    allowed = {
        "status",
        "resolve_locks",
        "authorize_scopes",
        "preregistered_inputs",
        "results_ref",
        "observed_elapsed_evidence",
        "archive_audit_clarification",
        "timestamp_audit_preregistration",
        "timestamp_input_manifest_binding",
        "register_reproduction",
        "supersede_reproduction",
    }
    if not set(changes).issubset(allowed):
        raise ExperimentRegistryError(f"{experiment_id}: uncontrolled amendment changes")
    if "status" in changes and changes["status"] not in _ALLOWED_STATUSES:
        raise ExperimentRegistryError(f"{experiment_id}: invalid amended status")
    if "register_reproduction" in changes:
        if set(changes) != {"register_reproduction"}:
            raise ExperimentRegistryError(
                f"{experiment_id}: register_reproduction must be an atomic "
                "amendment"
            )
        _validate_reproduction_registration(
            changes["register_reproduction"],
            experiment_id,
        )
    if "supersede_reproduction" in changes:
        if set(changes) != {"supersede_reproduction"}:
            raise ExperimentRegistryError(
                f"{experiment_id}: supersede_reproduction must be an "
                "atomic amendment"
            )
        if experiment_id != "X-11":
            raise ExperimentRegistryError(
                f"{experiment_id}: supersede_reproduction is only valid "
                "for X-11"
            )
        supersession = _exact_keys(
            changes["supersede_reproduction"],
            {
                "supersedes_reproduction_id",
                "registration",
            },
            "X-11 reproduction supersession",
        )
        _nonempty_string(
            supersession["supersedes_reproduction_id"],
            "X-11 supersedes_reproduction_id",
        )
        _validate_reproduction_registration(
            supersession["registration"],
            experiment_id,
            contract=_X11_REPRODUCTION_V2_CONTRACT,
        )
    if "resolve_locks" in changes:
        items = changes["resolve_locks"]
        if type(items) is not list:
            raise ExperimentRegistryError(f"{experiment_id}: resolve_locks must be a list")
        seen: set[str] = set()
        for item in items:
            item = _exact_keys(item, {"lock_id", "evidence_ref"}, f"{experiment_id} resolved lock")
            lock_id = _nonempty_string(item["lock_id"], f"{experiment_id} resolved lock id")
            if lock_id in seen:
                raise ExperimentRegistryError(f"{experiment_id}: duplicate resolved lock")
            seen.add(lock_id)
            _sha256(item["evidence_ref"], f"{experiment_id} lock evidence")
    if "authorize_scopes" in changes:
        scopes = changes["authorize_scopes"]
        if type(scopes) is not list or any(type(item) is not str for item in scopes):
            raise ExperimentRegistryError(f"{experiment_id}: authorize_scopes must be strings")
        if len(scopes) != len(set(scopes)):
            raise ExperimentRegistryError(f"{experiment_id}: duplicate authorized scope")
    if "preregistered_inputs" in changes:
        inputs = changes["preregistered_inputs"]
        if type(inputs) is not list:
            raise ExperimentRegistryError(f"{experiment_id}: preregistered_inputs must be a list")
        seen_inputs: set[str] = set()
        for item in inputs:
            item = _exact_keys(
                item,
                {
                    "scope",
                    "code_sha256",
                    "data_sha256",
                    "dataset_ids",
                    "model_ids",
                },
                f"{experiment_id} preregistered input",
            )
            scope = _nonempty_string(item["scope"], f"{experiment_id} input scope")
            if scope in seen_inputs:
                raise ExperimentRegistryError(f"{experiment_id}: duplicate input scope")
            seen_inputs.add(scope)
            _sha256(item["code_sha256"], f"{experiment_id} code_sha256")
            _sha256(item["data_sha256"], f"{experiment_id} data_sha256")
            _canonical_id_list(
                item["dataset_ids"],
                pattern=_DATASET_ID_RE,
                label=f"{experiment_id} preregistered dataset_ids",
            )
            _canonical_id_list(
                item["model_ids"],
                pattern=_MODEL_ID_RE,
                label=f"{experiment_id} preregistered model_ids",
            )
    if "results_ref" in changes:
        try:
            _validate_result_shape(changes["results_ref"])
        except InvalidResultReferenceError as exc:
            raise ExperimentRegistryError(f"{experiment_id}: invalid appended results_ref: {exc}") from exc
    if "observed_elapsed_evidence" in changes:
        if experiment_id != "X-08":
            raise ExperimentRegistryError(
                f"{experiment_id}: observed elapsed evidence is only valid for X-08"
            )
        if "results_ref" in changes or "status" in changes:
            raise ExperimentRegistryError(
                "X-08: observed elapsed evidence must be appended before "
                "result or terminal status amendments"
            )
        evidence = _exact_keys(
            changes["observed_elapsed_evidence"],
            {
                "capture_manifest_path",
                "capture_manifest_sha256",
            },
            "X-08 observed elapsed evidence",
        )
        _nonempty_string(
            evidence["capture_manifest_path"],
            "X-08 capture manifest path",
        )
        _sha256(
            evidence["capture_manifest_sha256"],
            "X-08 capture manifest SHA-256",
        )
    if "archive_audit_clarification" in changes:
        if experiment_id != "X-08":
            raise ExperimentRegistryError(
                f"{experiment_id}: archive audit clarification is only "
                "valid for X-08"
            )
        clarification = _exact_keys(
            changes["archive_audit_clarification"],
            {
                "stopped_archive_role",
                "official_historical_rest_supplement",
                "historical_l2_status",
                "live_l2_dataset_id",
            },
            "X-08 archive audit clarification",
        )
        if clarification != {
            "stopped_archive_role": "reference audit only",
            "official_historical_rest_supplement": (
                "Kalshi official historical REST candles and trades at "
                "fixed fetch cutoff"
            ),
            "historical_l2_status": "no historical L2",
            "live_l2_dataset_id": "DS-KALSHI-LIVE-L2",
        }:
            raise ExperimentRegistryError(
                "X-08: archive audit clarification must preserve the "
                "stopped archive and prohibit historical L2 claims"
            )
    if "timestamp_audit_preregistration" in changes:
        if experiment_id != "X-02":
            raise ExperimentRegistryError(
                f"{experiment_id}: timestamp audit preregistration is only "
                "valid for X-02"
            )
        if (
            changes["timestamp_audit_preregistration"]
            != _X02_TIMESTAMP_AUDIT_PREREGISTRATION
        ):
            raise ExperimentRegistryError(
                "X-02: timestamp audit preregistration must match the exact "
                "H-approved sample and definitions"
            )
        if set(changes) != {
            "resolve_locks",
            "timestamp_audit_preregistration",
        }:
            raise ExperimentRegistryError(
                "X-02: timestamp audit preregistration amendment must "
                "contain only its exact structured change and lock evidence"
            )
    if "timestamp_input_manifest_binding" in changes:
        if experiment_id != "X-02":
            raise ExperimentRegistryError(
                f"{experiment_id}: timestamp input manifest binding is only "
                "valid for X-02"
            )
        if (
            changes["timestamp_input_manifest_binding"]
            != _X02_TIMESTAMP_INPUT_MANIFEST_BINDING
        ):
            raise ExperimentRegistryError(
                "X-02: timestamp input manifest binding must match the exact "
                "H-approved four-day bundle"
            )
        if set(changes) != {
            "resolve_locks",
            "timestamp_input_manifest_binding",
        }:
            raise ExperimentRegistryError(
                "X-02: timestamp input manifest binding amendment must "
                "contain only its exact structured change and lock evidence"
            )
    if experiment_id == "X-02":
        resolved_items = changes.get("resolve_locks", [])
        resolved_by_id = {
            item["lock_id"]: item["evidence_ref"]
            for item in resolved_items
        }
        targeted_lock_ids = (
            set(resolved_by_id) & _X02_PREREGISTRATION_LOCK_IDS
        )
        has_preregistration = "timestamp_audit_preregistration" in changes
        has_input_binding = "timestamp_input_manifest_binding" in changes
        if has_preregistration:
            if set(resolved_by_id) != _X02_PREREGISTRATION_LOCK_IDS:
                raise ExperimentRegistryError(
                    "X-02: timestamp audit preregistration lock resolution "
                    "must contain exactly the three preregistration locks"
                )
            preregistration = changes["timestamp_audit_preregistration"]
            expected_evidence = {
                lock_id: "sha256:"
                + hashlib.sha256(
                    _canonical_bytes(preregistration[lock_id])
                ).hexdigest()
                for lock_id in _X02_PREREGISTRATION_LOCK_IDS
            }
            if resolved_by_id != expected_evidence:
                raise ExperimentRegistryError(
                    "X-02: preregistration lock evidence must hash the exact "
                    "approved sections"
                )
        elif targeted_lock_ids and not has_input_binding:
            raise ExperimentRegistryError(
                "X-02: preregistration locks require the structured "
                "timestamp audit preregistration"
            )
        targets_input_manifest = (
            "timestamp_input_manifest" in resolved_by_id
        )
        if has_input_binding:
            if set(resolved_by_id) != {"timestamp_input_manifest"}:
                raise ExperimentRegistryError(
                    "X-02: timestamp input manifest binding must resolve "
                    "only timestamp_input_manifest"
                )
            if resolved_by_id["timestamp_input_manifest"] != (
                _X02_TIMESTAMP_INPUT_MANIFEST_BINDING["bundle_sha256"]
            ):
                raise ExperimentRegistryError(
                    "X-02: timestamp input manifest lock evidence must equal "
                    "the verified bundle self-hash"
                )
        elif targets_input_manifest:
            raise ExperimentRegistryError(
                "X-02: timestamp_input_manifest requires the structured "
                "input manifest binding"
            )
    return changes


@dataclass
class _RegistrationMeta:
    head: str
    head_at: str = _RESULT_ACCEPTANCE_NOT_BEFORE
    preregistered_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    registered_reproduction_id: str | None = None
    registered_reproduction_scope: str | None = None
    registered_reproduction_scopes: set[str] = field(default_factory=set)
    stored_results: list[dict[str, Any]] = field(default_factory=list)
    result_appended_at: list[str] = field(default_factory=list)
    lock_resolved_at: dict[str, str] = field(default_factory=dict)
    scope_authorized_at: dict[str, str] = field(default_factory=dict)
    ready_at: str | None = None
    completion_evaluated_at: str | None = None
    observed_elapsed_evidence: list[dict[str, Any]] = field(default_factory=list)
    archive_audit_clarified: bool = False
    timestamp_audit_preregistered: bool = False
    timestamp_input_manifest_bound: bool = False


def _observed_elapsed_days(meta: _RegistrationMeta) -> int:
    """Return whole days in the longest continuous immutable X-08 window."""

    windows = sorted(
        (
            _verified_utc(item["window_started_at"], "X-08 observation start"),
            _verified_utc(item["window_ended_at"], "X-08 observation end"),
        )
        for item in meta.observed_elapsed_evidence
    )
    if not windows:
        return 0
    merged: list[tuple[datetime, datetime]] = []
    for started_at, ended_at in windows:
        if not merged or started_at > merged[-1][1]:
            merged.append((started_at, ended_at))
            continue
        prior_started_at, prior_ended_at = merged[-1]
        if ended_at > prior_ended_at:
            merged[-1] = (prior_started_at, ended_at)
    elapsed_seconds = max(
        (ended_at - started_at).total_seconds()
        for started_at, ended_at in merged
    )
    return int(elapsed_seconds // 86_400)


def _x08_capture_window(
    program_root: Path,
    evidence: dict[str, str],
    *,
    amended_at: datetime,
) -> dict[str, str]:
    manifest_ref = evidence["capture_manifest_path"]
    artifact_rows = _strict_csv(
        _safe_file(
            program_root,
            "registries/artifact_registry.csv",
            "artifact registry",
        ),
        _ARTIFACT_REGISTRY_FIELDS,
    )
    artifact = next(
        (row for row in artifact_rows if row["path"] == manifest_ref),
        None,
    )
    if artifact is None or artifact["status"] != "registered":
        raise ExperimentRegistryError(
            "X-08: capture manifest must be a registered artifact"
        )
    manifest_path = _safe_file(
        program_root,
        manifest_ref,
        "X-08 capture manifest",
    )
    raw = manifest_path.read_bytes()
    actual_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    if actual_sha256 != evidence["capture_manifest_sha256"]:
        raise ExperimentRegistryError(
            "X-08: capture manifest SHA-256 mismatch"
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRegistryError(
            "X-08: capture manifest is not canonical JSON"
        ) from exc
    if type(document) is not dict or raw != _canonical_bytes(document) + b"\n":
        raise ExperimentRegistryError(
            "X-08: capture manifest is not canonical JSON"
        )
    document = _exact_keys(
        document,
        {
            "schema_version",
            "experiment_id",
            "capture_session_id",
            "fixtures_used",
            "gap_policy",
            "raw_store_root",
            "streams",
        },
        "X-08 capture manifest",
    )
    if (
        document["schema_version"] != "x08-capture-evidence/v0"
        or document["experiment_id"] != "X-08"
    ):
        raise ExperimentRegistryError(
            "X-08: capture manifest identity is invalid"
        )
    capture_session_id = _nonempty_string(
        document["capture_session_id"],
        "X-08 capture session id",
    )
    if document["fixtures_used"] is not False:
        raise ExperimentRegistryError(
            "X-08: fixtures cannot satisfy observed elapsed time"
        )
    if document["gap_policy"] != _X08_GAP_POLICY:
        raise ExperimentRegistryError(
            "X-08: capture manifest must use the registered gap policy"
        )
    raw_store_ref = _nonempty_string(
        document["raw_store_root"],
        "X-08 raw store root",
    )
    raw_store_root = _safe_directory(
        program_root,
        raw_store_ref,
        "X-08 raw store root",
    )
    streams = document["streams"]
    if type(streams) is not list:
        raise ExperimentRegistryError("X-08: capture streams must be a list")
    stream_ids = [
        item.get("dataset_id") if type(item) is dict else None
        for item in streams
    ]
    if stream_ids != sorted(_X08_CAPTURE_STREAMS):
        raise ExperimentRegistryError(
            "X-08: capture manifest must contain both registered live streams"
        )

    all_manifest_refs: set[str] = set()
    stream_windows: list[tuple[datetime, datetime]] = []
    for stream_document in streams:
        stream_document = _exact_keys(
            stream_document,
            {
                "dataset_id",
                "source",
                "stream",
                "segment_manifest_paths",
            },
            "X-08 capture stream",
        )
        dataset_id = stream_document["dataset_id"]
        expected_source, expected_stream = _X08_CAPTURE_STREAMS[dataset_id]
        if (
            stream_document["source"] != expected_source
            or stream_document["stream"] != expected_stream
        ):
            raise ExperimentRegistryError(
                f"X-08: capture stream identity mismatch for {dataset_id}"
            )
        manifest_refs = stream_document["segment_manifest_paths"]
        if (
            type(manifest_refs) is not list
            or not manifest_refs
            or any(type(item) is not str or not item for item in manifest_refs)
            or len(manifest_refs) != len(set(manifest_refs))
        ):
            raise ExperimentRegistryError(
                f"X-08: {dataset_id} segment manifest paths are invalid"
            )
        receive_times: list[datetime] = []
        for segment_ref in manifest_refs:
            if segment_ref in all_manifest_refs:
                raise ExperimentRegistryError(
                    "X-08: duplicate capture segment manifest"
                )
            all_manifest_refs.add(segment_ref)
            try:
                verified = read_verified_segment(
                    raw_store_root / segment_ref,
                    root=raw_store_root,
                )
            except RawStoreError as exc:
                raise ExperimentRegistryError(
                    f"X-08: capture segment verification failed: {exc}"
                ) from exc
            segment = verified.manifest
            if (
                segment.source != expected_source
                or segment.stream != expected_stream
                or segment.capture_session_id != capture_session_id
                or not verified.receive_times
            ):
                raise ExperimentRegistryError(
                    f"X-08: {dataset_id} segment binding is invalid"
                )
            segment_times = [
                _verified_utc(value, "X-08 segment receive_at")
                for value in verified.receive_times
            ]
            if any(
                later <= earlier
                for earlier, later in zip(segment_times, segment_times[1:])
            ):
                raise ExperimentRegistryError(
                    f"X-08: {dataset_id} segment records are not monotonic"
                )
            sealed_at = _verified_utc(
                segment.sealed_at,
                "X-08 segment sealed_at",
            )
            if sealed_at <= segment_times[-1] or sealed_at >= amended_at:
                raise ExperimentRegistryError(
                    "X-08: segment must be sealed after its last record "
                    "and before the evidence amendment"
                )
            receive_times.extend(segment_times)
        if any(
            later <= earlier
            for earlier, later in zip(receive_times, receive_times[1:])
        ):
            raise ExperimentRegistryError(
                f"X-08: {dataset_id} segment sequence is not monotonic"
            )
        if any(
            (later - earlier).total_seconds()
            > _X08_GAP_POLICY["maximum_gap_seconds"]
            for earlier, later in zip(receive_times, receive_times[1:])
        ):
            raise ExperimentRegistryError(
                f"X-08: {dataset_id} capture contains a gap"
            )
        stream_windows.append((receive_times[0], receive_times[-1]))

    started_at = max(window[0] for window in stream_windows)
    ended_at = min(window[1] for window in stream_windows)
    if ended_at <= started_at:
        raise ExperimentRegistryError(
            "X-08: live capture streams have no overlapping coverage"
        )
    return {
        "capture_manifest_path": manifest_ref,
        "capture_manifest_sha256": evidence["capture_manifest_sha256"],
        "window_started_at": (
            started_at.isoformat().replace("+00:00", "Z")
        ),
        "window_ended_at": ended_at.isoformat().replace("+00:00", "Z"),
    }


def require_execution_authorized(card: dict[str, Any]) -> None:
    """Enforce the experiment-level runtime kill switch before child scopes."""

    if card.get("execution_authorized") is not True:
        raise UnauthorizedResultScopeError(
            f"{card.get('id', 'experiment')}: execution is not authorized"
        )


def _scope_and_locks(
    card: dict[str, Any], scope_name: str
) -> tuple[dict[str, Any], list[str]]:
    require_execution_authorized(card)
    scopes = card["authorization_scopes"]
    if scope_name not in scopes:
        raise UnauthorizedResultScopeError(f"{card['id']}: unknown result scope {scope_name!r}")
    scope = scopes[scope_name]
    if scope["authorized"] is not True or scope.get("permanent_no_go", False) is True:
        raise UnauthorizedResultScopeError(f"{card['id']}: result scope {scope_name} is not authorized")
    lock_by_id = {lock["id"]: lock for lock in card["registration_locks"]}
    unresolved = [
        lock_id
        for lock_id in scope["required_lock_ids"]
        if lock_by_id[lock_id]["status"] != "resolved"
    ]
    return scope, unresolved


def _validate_bound_result_inputs(
    program_root: Path,
    card: dict[str, Any],
    scope: dict[str, Any],
    registered_inputs: dict[str, Any],
    result: dict[str, Any],
) -> None:
    experiment_id = card["id"]
    binding = scope.get("input_binding")
    if binding is None:
        if result["dataset_ids"] or result["model_ids"]:
            raise InvalidResultReferenceError(
                f"{experiment_id}: unbound scope cannot claim datasets or models"
            )
        if registered_inputs["dataset_ids"] or registered_inputs["model_ids"]:
            raise InvalidResultReferenceError(
                f"{experiment_id}: unbound scope cannot preregister datasets or models"
            )
        return

    for field_name in ("dataset_ids", "model_ids"):
        expected = binding[field_name]
        if registered_inputs[field_name] != expected:
            raise InvalidResultReferenceError(
                f"{experiment_id}: preregistered {field_name} do not match scope binding"
            )
        if result[field_name] != expected:
            raise InvalidResultReferenceError(
                f"{experiment_id}: result {field_name} do not match scope binding"
            )

    if binding["result_class"] == "synthetic":
        fixture_hash = binding["synthetic_data_sha256"]
        if (
            registered_inputs["data_sha256"] != fixture_hash
            or result["data_sha256"] != fixture_hash
        ):
            raise InvalidResultReferenceError(
                f"{experiment_id}: synthetic result is not bound to its registered fixture"
            )
        from prediction_market.program_audit import (
            ResearchRegistryError,
            load_model_registry,
        )

        try:
            models_by_id = {
                row.model_id: row
                for row in load_model_registry(program_root)
            }
        except ResearchRegistryError as exc:
            raise InvalidResultReferenceError(
                f"{experiment_id}: model registry is invalid: {exc}"
            ) from exc
        for model_id in result["model_ids"]:
            model = models_by_id.get(model_id)
            if model is None or model.experiment_id != experiment_id:
                raise InvalidResultReferenceError(
                    f"{experiment_id}: synthetic result model {model_id} "
                    "is not registered to the experiment"
                )
        return

    from prediction_market.program_audit import (
        FormalResearchInputError,
        ResearchRegistryError,
        validate_registered_research_bindings,
    )

    try:
        validate_registered_research_bindings(
            program_root,
            experiment_id=experiment_id,
            dataset_ids=result["dataset_ids"],
            model_ids=result["model_ids"],
            result_class=binding["result_class"],
        )
    except (FormalResearchInputError, ResearchRegistryError) as exc:
        raise InvalidResultReferenceError(
            f"{experiment_id}: research input eligibility failed: {exc}"
        ) from exc


def _validate_result_against_state(
    program_root: Path,
    card: dict[str, Any],
    meta: _RegistrationMeta,
    result: dict[str, Any],
) -> None:
    scope, unresolved = _scope_and_locks(card, result["scope"])
    if (
        card["id"] == "X-08"
        and result["scope"] in card["completion_required_scopes"]
    ):
        required_days = card["prospective_observation"]["required_elapsed_days"]
        observed_days = _observed_elapsed_days(meta)
        if observed_days < required_days:
            raise InvalidResultReferenceError(
                "X-08: completion result requires seven actual elapsed "
                f"prospective days; observed {observed_days} of {required_days}"
            )
    if result["result_label"] != scope["required_result_label"]:
        raise UnauthorizedResultScopeError(
            f"{card['id']}: scope {result['scope']} requires {scope['required_result_label']} label"
        )
    if result["registration_head_sha256"] != meta.head:
        raise InvalidResultReferenceError(f"{card['id']}: registration head mismatch")
    registered_inputs = meta.preregistered_inputs.get(result["scope"])
    if registered_inputs is None:
        raise InvalidResultReferenceError(
            f"{card['id']}: no preregistered inputs for scope {result['scope']}"
        )
    if result["code_sha256"] != registered_inputs["code_sha256"] or result[
        "data_sha256"
    ] != registered_inputs["data_sha256"]:
        raise InvalidResultReferenceError(
            f"{card['id']}: result inputs do not match preregistered hashes"
        )
    if unresolved:
        raise UnresolvedRegistrationLockError(
            f"{card['id']}: unresolved registration locks: {', '.join(unresolved)}"
        )
    _validate_bound_result_inputs(
        program_root,
        card,
        scope,
        registered_inputs,
        result,
    )
    evaluation_at = _canonical_utc(
        result["evaluation_started_at"], "evaluation_started_at", result_error=True
    )
    if (
        result["scope"] == meta.registered_reproduction_scope
        and evaluation_at > _utc_now()
    ):
        raise PreRegistrationEvaluationError(
            f"{card['id']}: reproduction evaluation cannot be future-dated"
        )
    boundary = _canonical_utc(
        card["result_acceptance_not_before"], "result_acceptance_not_before", result_error=True
    )
    input_registered_at = _canonical_utc(
        registered_inputs["registered_at"], "input preregistration", result_error=True
    )
    if evaluation_at < boundary:
        raise PreRegistrationEvaluationError(
            f"{card['id']}: evaluation predates effective preregistration"
        )
    if evaluation_at <= input_registered_at:
        raise PreRegistrationEvaluationError(
            f"{card['id']}: evaluation must follow input preregistration amendment"
        )
    head_at = _canonical_utc(meta.head_at, "registration head", result_error=True)
    if evaluation_at <= head_at:
        raise PreRegistrationEvaluationError(
            f"{card['id']}: evaluation must follow the claimed registration head"
        )
    scope_authorized_at = meta.scope_authorized_at.get(result["scope"])
    if scope_authorized_at is None:
        raise UnauthorizedResultScopeError(
            f"{card['id']}: scope {result['scope']} lacks authorization history"
        )
    if evaluation_at <= _canonical_utc(
        scope_authorized_at, "scope authorization", result_error=True
    ):
        raise PreRegistrationEvaluationError(
            f"{card['id']}: scope was not authorized before evaluation"
        )
    for lock_id in scope["required_lock_ids"]:
        resolved_at = meta.lock_resolved_at.get(lock_id)
        if resolved_at is None or evaluation_at <= _canonical_utc(
            resolved_at, "lock resolution", result_error=True
        ):
            raise PreRegistrationEvaluationError(
                f"{card['id']}: lock {lock_id} was not resolved before evaluation"
            )


def _apply_amendments(
    program_root: Path,
    base_card: dict[str, Any],
    ledger_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], _RegistrationMeta]:
    experiment_id = base_card["id"]
    x11_v1_has_supersession = (
        experiment_id == "X-11"
        and any(
            type(amendment) is dict
            and type(amendment.get("changes")) is dict
            and "supersede_reproduction" in amendment["changes"]
            for amendment in base_card["amendments"]
        )
    )
    for expected_sequence, amendment in enumerate(base_card["amendments"], start=1):
        if type(amendment) is not dict or amendment.get("sequence") != expected_sequence:
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment sequence must be contiguous from 1"
            )
    expected_rows = 1 + len(base_card["amendments"])
    if len(ledger_rows) != expected_rows:
        raise ExperimentRegistryError(f"{experiment_id}: ledger and card chain length mismatch")
    seed = ledger_rows[0]
    if seed != {
        "experiment_id": experiment_id,
        "sequence": "0",
        "record_sha256": base_card["registration_record_sha256"],
        "prior_sha256": "",
        "amended_at": "",
        "approved_by": "",
        "reason": "",
    }:
        raise ExperimentRegistryError(f"{experiment_id}: ledger base record mismatch")
    effective = copy.deepcopy(base_card)
    meta = _RegistrationMeta(head=base_card["registration_record_sha256"])
    meta.scope_authorized_at = {
        scope_name: _RESULT_ACCEPTANCE_NOT_BEFORE
        for scope_name, scope in effective["authorization_scopes"].items()
        if scope["authorized"] is True and not scope.get("permanent_no_go", False)
    }
    prior_time = _canonical_utc(_RESULT_ACCEPTANCE_NOT_BEFORE, "registration boundary")
    for expected_sequence, amendment in enumerate(base_card["amendments"], start=1):
        expected_keys = {
            "sequence",
            "amended_at",
            "prior_sha256",
            "approved_by",
            "reason",
            "changes",
            "amendment_sha256",
        }
        amendment = _exact_keys(amendment, expected_keys, f"{experiment_id} amendment")
        if amendment["sequence"] != expected_sequence:
            raise ExperimentRegistryError(f"{experiment_id}: amendment sequence must be contiguous from 1")
        if amendment["prior_sha256"] != meta.head:
            raise ExperimentRegistryError(f"{experiment_id}: amendment prior hash mismatch")
        if amendment["approved_by"] not in {"H", "Team H"}:
            raise ExperimentRegistryError(f"{experiment_id}: amendment approved_by must be H")
        _nonempty_string(amendment["reason"], f"{experiment_id} amendment reason")
        amended_time = _canonical_utc(amendment["amended_at"], "amended_at")
        if amended_time <= prior_time:
            raise ExperimentRegistryError(f"{experiment_id}: amendment times must be strictly monotonic")
        if amendment["amendment_sha256"] != compute_amendment_sha256(amendment):
            raise ExperimentRegistryError(f"{experiment_id}: amendment content hash mismatch")
        ledger = ledger_rows[expected_sequence]
        expected_ledger = {
            "experiment_id": experiment_id,
            "sequence": str(expected_sequence),
            "record_sha256": amendment["amendment_sha256"],
            "prior_sha256": amendment["prior_sha256"],
            "amended_at": amendment["amended_at"],
            "approved_by": amendment["approved_by"],
            "reason": amendment["reason"],
        }
        if ledger != expected_ledger:
            raise ExperimentRegistryError(f"{experiment_id}: ledger and card chain record mismatch")
        changes = _validate_changes(amendment["changes"], experiment_id)
        if (
            (
                "register_reproduction" in changes
                or "supersede_reproduction" in changes
            )
            and amended_time > _utc_now()
        ):
            raise ExperimentRegistryError(
                f"{experiment_id}: reproduction amendment cannot be "
                "future-dated"
            )
        if (
            "timestamp_audit_preregistration" in changes
            and meta.timestamp_audit_preregistered
        ):
            raise ExperimentRegistryError(
                "X-02: timestamp audit preregistration is append-once"
            )
        if (
            "timestamp_input_manifest_binding" in changes
            and meta.timestamp_input_manifest_bound
        ):
            raise ExperimentRegistryError(
                "X-02: timestamp input manifest binding is append-once"
            )
        if "timestamp_input_manifest_binding" in changes:
            _verify_x02_timestamp_input_bundle(
                program_root,
                changes["timestamp_input_manifest_binding"],
            )
        if "register_reproduction" in changes:
            registration = changes["register_reproduction"]
            reproduction_id = registration["reproduction_id"]
            scope_name = registration["scope"]
            model_ids = [
                item["model_id"]
                for item in registration["model_bindings"]
            ]
            if meta.registered_reproduction_id is not None:
                raise ExperimentRegistryError(
                    f"{experiment_id}: register_reproduction is append-once; "
                    f"duplicate reproduction_id {reproduction_id}"
                )
            if scope_name in effective["authorization_scopes"]:
                raise ExperimentRegistryError(
                    f"{experiment_id}: reproduction scope {scope_name} "
                    "already exists"
                )
            lock_id = f"reproduction:{reproduction_id}"
            if any(
                lock["id"] == lock_id
                for lock in effective["registration_locks"]
            ):
                raise ExperimentRegistryError(
                    f"{experiment_id}: reproduction lock {lock_id} "
                    "already exists"
                )
            _validate_reproduction_bindings(
                program_root,
                experiment_id,
                registration,
                validate_code=not (
                    x11_v1_has_supersession
                    and reproduction_id
                    == _REPRODUCTION_CONTRACTS["X-11"][
                        "reproduction_id"
                    ]
                ),
            )
            effective["registration_locks"].append(
                {
                    "id": lock_id,
                    "status": "resolved",
                    "reason": (
                        f"Team H registered reproduction {reproduction_id} "
                        "before evaluation."
                    ),
                    "evidence_ref": registration[
                        "reproduction_spec_sha256"
                    ],
                }
            )
            base_scope_name = _REPRODUCTION_CONTRACTS[
                experiment_id
            ]["base_scope"]
            base_scope = effective["authorization_scopes"][
                base_scope_name
            ]
            inherited_lock_ids = (
                list(base_scope["required_lock_ids"])
                if _REPRODUCTION_CONTRACTS[experiment_id][
                    "inherit_base_locks"
                ]
                else []
            )
            effective["authorization_scopes"][scope_name] = {
                "authorized": True,
                "required_result_label": "PRELIMINARY",
                "required_lock_ids": [
                    *inherited_lock_ids,
                    lock_id,
                ],
                "input_binding": {
                    "result_class": "poc",
                    "dataset_ids": list(registration["dataset_ids"]),
                    "model_ids": model_ids,
                    "synthetic_data_sha256": None,
                },
            }
            meta.preregistered_inputs[scope_name] = {
                "code_sha256": registration["code_sha256"],
                "data_sha256": registration["data_sha256"],
                "dataset_ids": list(registration["dataset_ids"]),
                "model_ids": model_ids,
                "registered_at": amendment["amended_at"],
            }
            meta.scope_authorized_at[scope_name] = amendment["amended_at"]
            meta.lock_resolved_at[lock_id] = amendment["amended_at"]
            meta.registered_reproduction_id = reproduction_id
            meta.registered_reproduction_scope = scope_name
            meta.registered_reproduction_scopes.add(scope_name)
        if "supersede_reproduction" in changes:
            supersession = changes["supersede_reproduction"]
            supersedes_id = supersession[
                "supersedes_reproduction_id"
            ]
            registration = supersession["registration"]
            old_scope_name = meta.registered_reproduction_scope
            if (
                meta.registered_reproduction_id != supersedes_id
                or supersedes_id
                != _REPRODUCTION_CONTRACTS["X-11"][
                    "reproduction_id"
                ]
                or old_scope_name
                != _REPRODUCTION_CONTRACTS["X-11"]["scope"]
            ):
                raise ExperimentRegistryError(
                    "X-11: supersession must supersede the current "
                    "reproduction"
                )
            if any(
                result["scope"] == old_scope_name
                for result in meta.stored_results
            ):
                raise ExperimentRegistryError(
                    "X-11: V1 reproduction cannot be superseded after "
                    "evaluation"
                )
            reproduction_id = registration["reproduction_id"]
            scope_name = registration["scope"]
            model_ids = [
                item["model_id"]
                for item in registration["model_bindings"]
            ]
            old_scope = effective["authorization_scopes"][
                old_scope_name
            ]
            old_model_ids = old_scope["input_binding"]["model_ids"]
            if (
                reproduction_id == supersedes_id
                or scope_name == old_scope_name
                or set(model_ids) & set(old_model_ids)
            ):
                raise ExperimentRegistryError(
                    "X-11: supersession cannot reuse the V1 identity, "
                    "scope, or model"
                )
            if scope_name in effective["authorization_scopes"]:
                raise ExperimentRegistryError(
                    f"X-11: reproduction scope {scope_name} already exists"
                )
            lock_id = f"reproduction:{reproduction_id}"
            if any(
                lock["id"] == lock_id
                for lock in effective["registration_locks"]
            ):
                raise ExperimentRegistryError(
                    f"X-11: reproduction lock {lock_id} already exists"
                )
            _validate_reproduction_bindings(
                program_root,
                experiment_id,
                registration,
                contract=_X11_REPRODUCTION_V2_CONTRACT,
            )
            old_scope["authorized"] = False
            meta.scope_authorized_at.pop(old_scope_name, None)
            effective["registration_locks"].append(
                {
                    "id": lock_id,
                    "status": "resolved",
                    "reason": (
                        f"Team H registered reproduction "
                        f"{reproduction_id} before evaluation."
                    ),
                    "evidence_ref": registration[
                        "reproduction_spec_sha256"
                    ],
                }
            )
            effective["authorization_scopes"][scope_name] = {
                "authorized": True,
                "required_result_label": "PRELIMINARY",
                "required_lock_ids": [lock_id],
                "input_binding": {
                    "result_class": "poc",
                    "dataset_ids": list(registration["dataset_ids"]),
                    "model_ids": model_ids,
                    "synthetic_data_sha256": None,
                },
            }
            meta.preregistered_inputs[scope_name] = {
                "code_sha256": registration["code_sha256"],
                "data_sha256": registration["data_sha256"],
                "dataset_ids": list(registration["dataset_ids"]),
                "model_ids": model_ids,
                "registered_at": amendment["amended_at"],
            }
            meta.scope_authorized_at[scope_name] = amendment[
                "amended_at"
            ]
            meta.lock_resolved_at[lock_id] = amendment["amended_at"]
            meta.registered_reproduction_id = reproduction_id
            meta.registered_reproduction_scope = scope_name
            meta.registered_reproduction_scopes.add(scope_name)
        if "resolve_locks" in changes:
            lock_by_id = {lock["id"]: lock for lock in effective["registration_locks"]}
            for item in changes["resolve_locks"]:
                if item["lock_id"] not in lock_by_id:
                    raise ExperimentRegistryError(f"{experiment_id}: unknown resolved lock")
                lock = lock_by_id[item["lock_id"]]
                if lock["status"] == "resolved":
                    raise ExperimentRegistryError(f"{experiment_id}: lock already resolved")
                lock["status"] = "resolved"
                lock["evidence_ref"] = item["evidence_ref"]
                meta.lock_resolved_at[item["lock_id"]] = amendment["amended_at"]
        if "authorize_scopes" in changes:
            for scope_name in changes["authorize_scopes"]:
                if scope_name not in effective["authorization_scopes"]:
                    raise ExperimentRegistryError(f"{experiment_id}: unknown authorized scope")
                if (
                    scope_name
                    in meta.registered_reproduction_scopes
                ):
                    raise ExperimentRegistryError(
                        f"{experiment_id}: reproduction scopes cannot be "
                        "generically authorized"
                    )
                scope = effective["authorization_scopes"][scope_name]
                if scope.get("permanent_no_go", False):
                    raise ExperimentRegistryError(f"{experiment_id}: cannot authorize permanent NO-GO")
                scope["authorized"] = True
                meta.scope_authorized_at[scope_name] = amendment["amended_at"]
        if "preregistered_inputs" in changes:
            for item in changes["preregistered_inputs"]:
                if (
                    item["scope"]
                    in meta.registered_reproduction_scopes
                ):
                    raise ExperimentRegistryError(
                        f"{experiment_id}: reproduction inputs are "
                        "append-once"
                    )
                if item["scope"] not in effective["authorization_scopes"]:
                    raise ExperimentRegistryError(f"{experiment_id}: unknown input scope")
                scope_binding = effective["authorization_scopes"][
                    item["scope"]
                ].get("input_binding")
                if (
                    isinstance(scope_binding, dict)
                    and scope_binding["result_class"] == "synthetic"
                    and (
                        item["dataset_ids"] != scope_binding["dataset_ids"]
                        or item["model_ids"] != scope_binding["model_ids"]
                        or item["data_sha256"]
                        != scope_binding["synthetic_data_sha256"]
                    )
                ):
                    raise ExperimentRegistryError(
                        f"{experiment_id}: synthetic preregistration must bind "
                        "the exact fixture and registered models"
                    )
                meta.preregistered_inputs[item["scope"]] = {
                    "code_sha256": item["code_sha256"],
                    "data_sha256": item["data_sha256"],
                    "dataset_ids": list(item["dataset_ids"]),
                    "model_ids": list(item["model_ids"]),
                    "registered_at": amendment["amended_at"],
                }
        if "observed_elapsed_evidence" in changes:
            if amended_time > _utc_now():
                raise ExperimentRegistryError(
                    "X-08: observed elapsed evidence amendment cannot be "
                    "future-dated"
                )
            evidence = _x08_capture_window(
                program_root,
                changes["observed_elapsed_evidence"],
                amended_at=amended_time,
            )
            boundary = _canonical_utc(
                effective["result_acceptance_not_before"],
                "X-08 result acceptance boundary",
            )
            started_at = _verified_utc(
                evidence["window_started_at"], "X-08 observation start"
            )
            ended_at = _verified_utc(
                evidence["window_ended_at"], "X-08 observation end"
            )
            if started_at < boundary:
                raise ExperimentRegistryError(
                    "X-08: observed elapsed evidence predates the "
                    "prospective registration boundary"
                )
            if ended_at >= amended_time:
                raise ExperimentRegistryError(
                    "X-08: observed elapsed evidence must end before its amendment"
                )
            if any(
                item["capture_manifest_sha256"]
                == evidence["capture_manifest_sha256"]
                for item in meta.observed_elapsed_evidence
            ):
                raise ExperimentRegistryError(
                    "X-08: duplicate observed elapsed evidence"
                )
            meta.observed_elapsed_evidence.append(evidence)
            effective["prospective_observation"]["observed_elapsed_days"] = (
                _observed_elapsed_days(meta)
            )
        if "archive_audit_clarification" in changes:
            if meta.archive_audit_clarified:
                raise ExperimentRegistryError(
                    "X-08: archive audit clarification is append-once"
                )
            meta.archive_audit_clarified = True
        if "timestamp_audit_preregistration" in changes:
            meta.timestamp_audit_preregistered = True
            effective["timestamp_audit_preregistration"] = copy.deepcopy(
                changes["timestamp_audit_preregistration"]
            )
        if "timestamp_input_manifest_binding" in changes:
            meta.timestamp_input_manifest_bound = True
            effective["timestamp_input_manifest_binding"] = copy.deepcopy(
                changes["timestamp_input_manifest_binding"]
            )
        if "results_ref" in changes:
            result = _validate_result_shape(changes["results_ref"])
            if (
                result["scope"] == meta.registered_reproduction_scope
                and amended_time > _utc_now()
            ):
                raise ExperimentRegistryError(
                    f"{experiment_id}: reproduction result amendment cannot "
                    "be future-dated"
                )
            try:
                _validate_result_against_state(program_root, effective, meta, result)
            except ExperimentRegistryError as exc:
                raise ExperimentRegistryError(
                    f"{experiment_id}: appended result does not match evaluation head: {exc}"
                ) from exc
            result_time = _canonical_utc(result["evaluation_started_at"], "evaluation_started_at")
            if amended_time <= result_time:
                raise ExperimentRegistryError(f"{experiment_id}: result amendment must follow evaluation")
            effective["results_ref"].append(result)
            meta.stored_results.append(result)
            meta.result_appended_at.append(amendment["amended_at"])
        if "status" in changes:
            prior_status = effective["status"]
            next_status = changes["status"]
            if prior_status in {"done", "failed", "abandoned"}:
                raise ExperimentRegistryError(
                    f"{experiment_id}: terminal status cannot transition"
                )
            allowed_transitions = {
                "registered": {"running", "done", "failed", "abandoned"},
                "running": {"done", "failed", "abandoned"},
            }
            if next_status not in allowed_transitions.get(prior_status, set()):
                raise ExperimentRegistryError(
                    f"{experiment_id}: invalid status transition"
                )
            if next_status == "done":
                completed_scopes = {result["scope"] for result in meta.stored_results}
                missing_scopes = set(effective["completion_required_scopes"]) - completed_scopes
                if missing_scopes:
                    raise ExperimentRegistryError(
                        f"{experiment_id}: terminal done status lacks completion scope evidence"
                    )
                if experiment_id == "X-08":
                    required_days = effective["prospective_observation"][
                        "required_elapsed_days"
                    ]
                    observed_days = _observed_elapsed_days(meta)
                    if observed_days < required_days:
                        raise ExperimentRegistryError(
                            "X-08: terminal done status requires seven actual "
                            f"elapsed prospective days; observed {observed_days} "
                            f"of {required_days}"
                        )
                meta.ready_at = amendment["amended_at"]
                completion_results = [
                    result
                    for result in meta.stored_results
                    if result["scope"] in effective["completion_required_scopes"]
                ]
                meta.completion_evaluated_at = min(
                    result["evaluation_started_at"] for result in completion_results
                )
            effective["status"] = next_status
        meta.head = amendment["amendment_sha256"]
        meta.head_at = amendment["amended_at"]
        prior_time = amended_time
    effective["registration_head_sha256"] = meta.head
    effective["preregistered_inputs"] = copy.deepcopy(meta.preregistered_inputs)
    return effective, meta


def _dependency_ready(
    experiment_id: str,
    registry: dict[str, dict[str, Any]],
    metadata: dict[str, _RegistrationMeta],
    visiting: set[str] | None = None,
    *,
    before: str | None = None,
) -> bool:
    visiting = set() if visiting is None else visiting
    if experiment_id in visiting:
        return False
    visiting.add(experiment_id)
    card = registry[experiment_id]
    meta = metadata[experiment_id]
    ready = card["status"] == "done" and meta.ready_at is not None
    if ready and before is not None:
        ready = _canonical_utc(meta.ready_at, "dependency ready_at") < _canonical_utc(
            before, "dependent evaluation"
        )
    if ready:
        dependency_deadline = meta.completion_evaluated_at
        ready = dependency_deadline is not None and all(
            _dependency_ready(
                dependency,
                registry,
                metadata,
                visiting,
                before=dependency_deadline,
            )
            for dependency in card["dependencies"]
        )
    visiting.remove(experiment_id)
    return ready


def _load_registry_internal(
    program_root: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, _RegistrationMeta]]:
    root = Path(program_root)
    _validate_card_inventory(root)
    _validate_artifact_registry(root)
    registry_csv = _safe_file(root, "registries/experiment_registry.csv", "experiment registry")
    rows = _strict_csv(registry_csv, _EXPERIMENT_REGISTRY_FIELDS)
    row_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        experiment_id = row["experiment_id"]
        if experiment_id in row_by_id:
            raise ExperimentRegistryError(f"duplicate experiment registry row: {experiment_id}")
        row_by_id[experiment_id] = row
    if set(row_by_id) != _EXPERIMENT_ID_SET:
        raise ExperimentRegistryError("registry must contain exactly X-01 through X-12")

    catalog_gates = _catalog_gates(root)
    base_cards: dict[str, dict[str, Any]] = {}
    for experiment_id in EXPERIMENT_IDS:
        row = row_by_id[experiment_id]
        relative = f"registries/experiments/{experiment_id}.yaml"
        if row["card_path"] != relative:
            raise ExperimentRegistryError(f"{experiment_id}: filename/card/CSV path mismatch")
        path = _safe_file(root, relative, "experiment card")
        raw = path.read_bytes()
        if row["card_sha256"] != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ExperimentRegistryError(f"{experiment_id}: card SHA-256 mismatch")
        card = _load_yaml_card(raw, experiment_id)
        _validate_card_structure(card, experiment_id)
        if "synthetic_fixture" in card:
            fixture = card["synthetic_fixture"]
            fixture_path = _safe_file(
                root,
                fixture["path"],
                f"{experiment_id} synthetic fixture",
            )
            fixture_digest = "sha256:" + hashlib.sha256(
                fixture_path.read_bytes()
            ).hexdigest()
            if fixture_digest != fixture["sha256"]:
                raise ExperimentRegistryError(
                    f"{experiment_id}: synthetic fixture SHA-256 mismatch"
                )
        if card["registration_record_sha256"] != compute_registration_record_sha256(card):
            raise ExperimentRegistryError(
                f"{experiment_id}: immutable registration record SHA-256 mismatch"
            )
        if row["owner_team"] != card["owner_team"] or row["status"] != card["status"]:
            raise ExperimentRegistryError(f"{experiment_id}: CSV/card metadata mismatch")
        if row["execution_authorized"] != str(card["execution_authorized"]).lower():
            raise ExperimentRegistryError(f"{experiment_id}: CSV/card authorization mismatch")
        if row["registered_at"] != card["registered_at"] or row["due_gate"]:
            raise ExperimentRegistryError(f"{experiment_id}: CSV/card registration mismatch")
        if card["linked_first_artifact_due_gates"] != catalog_gates[experiment_id]:
            raise ExperimentRegistryError(f"{experiment_id}: catalog first-artifact gate mismatch")
        if card["source_lineage"]["catalog_item_ids"] != [
            item["catalog_item_id"] for item in catalog_gates[experiment_id]
        ]:
            raise ExperimentRegistryError(f"{experiment_id}: stable catalog lineage mismatch")
        base_cards[experiment_id] = card

    # Semantic errors are surfaced before the trust-anchor error so malformed test
    # fixtures receive the most specific rejection.
    _check_dependency_graph(base_cards)
    for experiment_id, card in base_cards.items():
        if card["registration_record_sha256"] != _TRUSTED_BASE_REGISTRATIONS[experiment_id]:
            raise ExperimentRegistryError(
                f"{experiment_id}: trusted base registration SHA-256 mismatch"
            )
        for dependency in card.get("artifact_dependencies", []):
            artifact = _safe_file(root, dependency["path"], "artifact dependency")
            digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != dependency["sha256"]:
                raise ExperimentRegistryError(f"{experiment_id}: artifact dependency SHA-256 mismatch")

    ledger_path = _safe_file(
        root, "registries/experiment_amendment_ledger.csv", "amendment ledger"
    )
    ledger = _strict_csv(ledger_path, _LEDGER_FIELDS)
    ledger_by_id = {experiment_id: [] for experiment_id in EXPERIMENT_IDS}
    for row in ledger:
        if row["experiment_id"] not in _EXPERIMENT_ID_SET:
            raise ExperimentRegistryError("amendment ledger has unknown experiment")
        ledger_by_id[row["experiment_id"]].append(row)

    registry: dict[str, dict[str, Any]] = {}
    metadata: dict[str, _RegistrationMeta] = {}
    for experiment_id in EXPERIMENT_IDS:
        effective, meta = _apply_amendments(
            root,
            base_cards[experiment_id],
            ledger_by_id[experiment_id],
        )
        registry[experiment_id] = effective
        metadata[experiment_id] = meta
    for experiment_id, card in registry.items():
        if card["status"] == "done" and not _dependency_ready(
            experiment_id, registry, metadata
        ):
            if card["dependencies"]:
                raise ExperimentRegistryError(
                    f"{experiment_id}: dependency was not ready before evaluation"
                )
            raise ExperimentRegistryError(
                f"{experiment_id}: done status lacks validated result/dependency evidence"
            )
    return registry, metadata


def load_experiment_registry(program_root: str | Path) -> dict[str, dict[str, Any]]:
    registry, _ = _load_registry_internal(program_root)
    return registry


def validate_result_ref(
    program_root: str | Path,
    experiment_id: str,
    result_ref: Any,
) -> dict[str, Any]:
    root = Path(program_root)
    if experiment_id not in _EXPERIMENT_ID_SET or not (
        root / "registries" / "experiment_registry.csv"
    ).is_file():
        raise UnregisteredExperimentError(
            f"experiment {experiment_id} has no preexisting registration"
        )
    result = _validate_result_shape(result_ref)
    registry, metadata = _load_registry_internal(root)
    card = registry[experiment_id]
    meta = metadata[experiment_id]
    _validate_result_against_state(root, card, meta, result)
    incomplete = [
        dependency
        for dependency in card["dependencies"]
        if not _dependency_ready(
            dependency,
            registry,
            metadata,
            before=result["evaluation_started_at"],
        )
    ]
    if incomplete:
        raise UnresolvedDependencyError(
            f"{experiment_id}: unresolved dependencies: {', '.join(incomplete)}"
        )
    return dict(result)


__all__ = [
    "ExperimentRegistryError",
    "InvalidResultReferenceError",
    "PreRegistrationEvaluationError",
    "UnauthorizedResultScopeError",
    "UnregisteredExperimentError",
    "UnresolvedDependencyError",
    "UnresolvedRegistrationLockError",
    "compute_amendment_sha256",
    "compute_registration_record_sha256",
    "load_experiment_registry",
    "require_execution_authorized",
    "validate_result_ref",
]
