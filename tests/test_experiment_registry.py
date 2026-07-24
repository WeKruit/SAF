from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import patch

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction_market.experiments import (  # noqa: E402
    ExperimentRegistryError,
    InvalidResultReferenceError,
    PreRegistrationEvaluationError,
    UnauthorizedResultScopeError,
    UnregisteredExperimentError,
    UnresolvedDependencyError,
    UnresolvedRegistrationLockError,
    compute_amendment_sha256,
    compute_registration_record_sha256,
    load_experiment_registry,
    validate_result_ref,
)
import prediction_market.experiments as experiments_module  # noqa: E402
from prediction_market.program_audit import (  # noqa: E402
    load_dataset_registry,
    load_model_registry,
)
import prediction_market.raw_store as raw_store_module  # noqa: E402
from prediction_market.raw_store import RawSegmentWriter  # noqa: E402


EXPECTED_EXPERIMENT_IDS = {f"X-{number:02d}" for number in range(1, 13)}
EXPECTED_MEASUREMENT_EXEMPTIONS = {"X-02", "X-03", "X-07"}
EXPECTED_NO_GOS = {
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
EXPECTED_TASK_3_PATHS = {
    "registries/experiment_registry.csv",
    "registries/experiment_amendment_ledger.csv",
    "registries/artifact_registry.csv",
    "artifacts/validation/validation_standard_v0.md",
    "src/prediction_market/experiments.py",
    "tests/test_experiment_registry.py",
    *(f"registries/experiments/X-{number:02d}.yaml" for number in range(1, 13)),
}
REPRODUCTION_CONTRACTS = {
    "X-11": {
        "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V1",
        "scope": "team_h_nfl_fastrmodels_reproduction_v1",
        "base_scope": "preregistered_pipeline",
        "dataset_ids": ["DS-NFL-FASTRMODELS", "DS-NFLVERSE"],
        "model_ids": [
            "MODEL-NFL-FASTRMODELS-NO-SPREAD",
        ],
        "code_paths": [
            "src/prediction_market/models/nfl.py",
            "src/prediction_market/models/nfl_fastrmodels.py",
        ],
        "protocol_path": (
            "registries/protocols/"
            "x11_fastrmodels_no_spread_v0.json"
        ),
        "inherit_base_locks": False,
    },
    "X-12": {
        "reproduction_id": "REPRO-X12-SOCCER-DYNAMIC-TRANSITION-V1",
        "scope": "team_h_soccer_dynamic_transition_reproduction_v1",
        "base_scope": "poc_result",
        "dataset_ids": ["DS-STATSBOMB-OPEN"],
        "model_ids": [
            "MODEL-SOCCER-DIXON-COLES",
            "MODEL-SOCCER-DYNAMIC-INTENSITY",
        ],
        "code_paths": [
            "src/prediction_market/sports/soccer_transition_model.py",
            "src/prediction_market/sports/x12.py",
        ],
    },
}
X11_CURRENT_REPRODUCTION_CONTRACT = {
    "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V2",
    "scope": "team_h_nfl_fastrmodels_reproduction_v2",
    "base_scope": "preregistered_pipeline",
    "dataset_ids": ["DS-NFL-FASTRMODELS", "DS-NFLVERSE"],
    "model_ids": [
        "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
    ],
    "code_paths": [
        "src/prediction_market/models/nfl.py",
        "src/prediction_market/models/nfl_fastrmodels.py",
    ],
    "protocol_path": (
        "registries/protocols/"
        "x11_fastrmodels_no_spread_v1.json"
    ),
    "inherit_base_locks": False,
}
X11_V1_REPRODUCTION_ID = "REPRO-X11-NFL-FASTRMODELS-V1"
X11_V1_SCOPE = "team_h_nfl_fastrmodels_reproduction_v1"
X11_V1_AMENDMENT_SHA256 = (
    "sha256:"
    "ad594d2aa06ff7ecc99ba4389d53c2973f1aba8bc922bba45a7c0cedc3ed6177"
)
X11_V1_PROTOCOL_FILE_SHA256 = (
    "sha256:"
    "7957bb56a23ff4908fb70f3219cd2c6a6196e84fddab65ea54d693fb13c914bc"
)
EXPECTED_X02_PREREGISTRATION = {
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
EXPECTED_X02_INPUT_MANIFEST_BINDING = {
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
EXPECTED_X02_FORMAL_INPUT = {
    "scope": "formal_result",
    "code_sha256": (
        "sha256:"
        "2fca15f8c1cef4066887d08c13c4036266be9fa35bbe25c09b1c07089f4b0a86"
    ),
    "data_sha256": EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_sha256"],
    "dataset_ids": ["DS-PMXT-V2"],
    "model_ids": [],
}
EXPECTED_X02_DAY_MANIFESTS = [
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
X02_STATIC_SIDECAR_ROOT = (
    "artifacts/data-audit/x02-static-store-v1"
)


@pytest.fixture
def program_root() -> Path:
    return PROJECT_ROOT


def _copy_program_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "program"
    shutil.copytree(PROJECT_ROOT / "charter", root / "charter")
    shutil.copytree(PROJECT_ROOT / "registries", root / "registries")
    shutil.copytree(PROJECT_ROOT / "artifacts", root / "artifacts")
    shutil.copytree(PROJECT_ROOT / "contracts", root / "contracts")
    code_paths = {
        relative
        for contract in REPRODUCTION_CONTRACTS.values()
        for relative in contract["code_paths"]
    }
    for relative_value in code_paths:
        relative = Path(str(relative_value))
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)
    for contract in REPRODUCTION_CONTRACTS.values():
        protocol_path = contract.get("protocol_path")
        if protocol_path is None:
            continue
        relative = Path(str(protocol_path))
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)
    _remove_checked_in_x11_reproduction(root)
    return root


def _remove_checked_in_x11_reproduction(root: Path) -> None:
    """Keep mutation fixtures anchored to X-11's immutable base card."""

    card = _read_card(root, "X-11")
    registration_index = next(
        (
            index
            for index, amendment in enumerate(card["amendments"])
            if "register_reproduction" in amendment["changes"]
        ),
        None,
    )
    if registration_index is None:
        return
    card["amendments"] = card["amendments"][:registration_index]
    _write_card(root, "X-11", card)
    _update_registry_card_hash(root, "X-11")

    ledger_path = root / "registries" / "experiment_amendment_ledger.csv"
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows = [
        row
        for row in rows
        if row["experiment_id"] != "X-11"
        or int(row["sequence"]) <= registration_index
    ]
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_card(root: Path, experiment_id: str) -> dict:
    path = root / "registries" / "experiments" / f"{experiment_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment_card"]


def _write_card(root: Path, experiment_id: str, card: dict) -> None:
    path = root / "registries" / "experiments" / f"{experiment_id}.yaml"
    path.write_text(
        yaml.safe_dump(
            {"experiment_card": card},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _update_registry_card_hash(root: Path, experiment_id: str) -> None:
    registry_path = root / "registries" / "experiment_registry.csv"
    with registry_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    card_path = root / "registries" / "experiments" / f"{experiment_id}.yaml"
    digest = "sha256:" + hashlib.sha256(card_path.read_bytes()).hexdigest()
    for row in rows:
        if row["experiment_id"] == experiment_id:
            row["card_sha256"] = digest
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_registry_field(
    root: Path, experiment_id: str, field: str, value: str
) -> None:
    registry_path = root / "registries" / "experiment_registry.csv"
    with registry_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    for row in rows:
        if row["experiment_id"] == experiment_id:
            row[field] = value
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_registered_card(root: Path, experiment_id: str, card: dict) -> None:
    card["registration_record_sha256"] = compute_registration_record_sha256(card)
    _write_card(root, experiment_id, card)
    _update_registry_card_hash(root, experiment_id)


def _update_dataset_registry(
    root: Path, dataset_id: str, **updates: str
) -> None:
    path = root / "registries" / "dataset_registry.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    row = next(item for item in rows if item["dataset_id"] == dataset_id)
    row.update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_license_review(
    root: Path, review_id: str, **updates: str
) -> None:
    path = root / "registries" / "data_license_register.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    row = next(item for item in rows if item["catalog_item_id"] == review_id)
    row.update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _model_record_sha256(root: Path, model_id: str) -> str:
    row = next(
        item for item in load_model_registry(root) if item.model_id == model_id
    )
    record = {
        field_name: (
            list(value)
            if type(value := getattr(row, field_name)) is tuple
            else value
        )
        for field_name in row.__dataclass_fields__
    }
    return _canonical_sha256(record)


def _reproduction_code_sha256(
    root: Path,
    code_paths: list[str],
) -> str:
    return _canonical_sha256(
        [
            {
                "path": code_path,
                "sha256": (
                    "sha256:"
                    + hashlib.sha256(
                        (root / code_path).read_bytes()
                    ).hexdigest()
                ),
            }
            for code_path in code_paths
        ]
    )


def _seed_reproduction_model_rows(
    root: Path,
    experiment_id: str,
) -> None:
    path = root / "registries" / "model_registry.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    by_id = {row["model_id"]: row for row in rows}
    if experiment_id == "X-11":
        template = by_id["MODEL-NFL-NFLFASTR-COMPARATOR"]
        required = (
            "MODEL-NFL-FASTRMODELS-NO-SPREAD",
        )
    else:
        template = by_id["MODEL-SOCCER-FIVE-MINUTE-TRANSITION"]
        required = ("MODEL-SOCCER-DYNAMIC-INTENSITY",)
    changed = False
    for model_id in required:
        if model_id in by_id:
            continue
        row = dict(template)
        row["model_id"] = model_id
        row["model_version"] = "v1"
        row["parameter_config_sha256"] = _canonical_sha256(
            {
                "fixture_only": True,
                "model_id": model_id,
            }
        )
        rows.append(row)
        by_id[model_id] = row
        changed = True
    if changed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _valid_reproduction_registration(
    root: Path,
    experiment_id: str,
    *,
    scope: str | None = None,
    reproduction_id: str | None = None,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    contract = REPRODUCTION_CONTRACTS[experiment_id]
    _seed_reproduction_model_rows(root, experiment_id)
    dataset_ids = list(contract["dataset_ids"])
    default_model_ids = list(contract["model_ids"])
    selected_model_ids = (
        default_model_ids if model_ids is None else model_ids
    )
    models = {
        row.model_id: row for row in load_model_registry(root)
    }
    datasets = {
        row.dataset_id: row for row in load_dataset_registry(root)
    }
    code_paths = [str(item) for item in contract["code_paths"]]
    dataset_bindings = [
        {
            "dataset_id": dataset_id,
            "manifest_sha256": datasets[dataset_id].manifest_sha256,
        }
        for dataset_id in dataset_ids
    ]
    registration: dict[str, Any] = {
        "reproduction_id": (
            str(contract["reproduction_id"])
            if reproduction_id is None
            else reproduction_id
        ),
        "scope": (
            str(contract["scope"])
            if scope is None
            else scope
        ),
        "result_class": "poc",
        "dataset_ids": dataset_ids,
        "model_bindings": [
            {
                "model_id": model_id,
                "model_version": models[model_id].model_version,
                "model_record_sha256": _model_record_sha256(
                    root, model_id
                ),
            }
            for model_id in selected_model_ids
        ],
        "code_paths": code_paths,
        "code_sha256": _reproduction_code_sha256(root, code_paths),
        "data_sha256": _canonical_sha256(dataset_bindings),
    }
    protocol_path = contract.get("protocol_path")
    if protocol_path is not None:
        protocol_bytes = (root / str(protocol_path)).read_bytes()
        registration["protocol_path"] = str(protocol_path)
        registration["protocol_sha256"] = (
            "sha256:" + hashlib.sha256(protocol_bytes).hexdigest()
        )
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        registration
    )
    return registration


def _x11_v1_protocol_document(root: Path) -> dict[str, Any]:
    protocol = json.loads(
        (
            root
            / "registries/protocols/"
            "x11_fastrmodels_no_spread_v0.json"
        ).read_bytes()
    )
    protocol["identity"] = {
        "experiment_id": "X-11",
        "model_id": "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
        "protocol_version": "v1",
        "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V2",
        "result_class": "poc",
        "scope": "team_h_nfl_fastrmodels_reproduction_v2",
    }
    protocol["feature_contract"]["half_seconds_remaining_semantics"] = {
        "qtr_lte_2": (
            "half_seconds_remaining=game_seconds_remaining-1800"
        ),
        "qtr_gte_3": (
            "half_seconds_remaining=game_seconds_remaining"
        ),
    }
    return protocol


def _seed_x11_v2_reproduction_assets(root: Path) -> None:
    protocol_path = (
        root
        / "registries/protocols/"
        "x11_fastrmodels_no_spread_v1.json"
    )
    if not protocol_path.exists():
        _write_pretty_json(
            protocol_path,
            _x11_v1_protocol_document(root),
        )
    protocol_sha256 = (
        "sha256:" + hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )

    model_path = root / "registries/model_registry.csv"
    with model_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    new_model_id = str(
        X11_CURRENT_REPRODUCTION_CONTRACT["model_ids"][0]
    )
    if any(row["model_id"] == new_model_id for row in rows):
        return
    old = next(
        row
        for row in rows
        if row["model_id"] == "MODEL-NFL-FASTRMODELS-NO-SPREAD"
    )
    new = dict(old)
    new.update(
        {
            "model_id": new_model_id,
            "model_version": "v1",
            "pit_feature_contract": protocol_sha256,
            "parameter_config_sha256": protocol_sha256,
        }
    )
    rows.append(new)
    with model_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _valid_x11_v2_reproduction_registration(
    root: Path,
) -> dict[str, Any]:
    _seed_x11_v2_reproduction_assets(root)
    contract = X11_CURRENT_REPRODUCTION_CONTRACT
    models = {row.model_id: row for row in load_model_registry(root)}
    datasets = {
        row.dataset_id: row for row in load_dataset_registry(root)
    }
    code_paths = list(contract["code_paths"])
    protocol_path = str(contract["protocol_path"])
    protocol_bytes = (root / protocol_path).read_bytes()
    registration: dict[str, Any] = {
        "reproduction_id": str(contract["reproduction_id"]),
        "scope": str(contract["scope"]),
        "result_class": "poc",
        "dataset_ids": list(contract["dataset_ids"]),
        "model_bindings": [
            {
                "model_id": model_id,
                "model_version": models[model_id].model_version,
                "model_record_sha256": _model_record_sha256(
                    root, model_id
                ),
            }
            for model_id in contract["model_ids"]
        ],
        "code_paths": code_paths,
        "code_sha256": _reproduction_code_sha256(root, code_paths),
        "data_sha256": _canonical_sha256(
            [
                {
                    "dataset_id": dataset_id,
                    "manifest_sha256": datasets[
                        dataset_id
                    ].manifest_sha256,
                }
                for dataset_id in contract["dataset_ids"]
            ]
        ),
        "protocol_path": protocol_path,
        "protocol_sha256": (
            "sha256:" + hashlib.sha256(protocol_bytes).hexdigest()
        ),
    }
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        registration
    )
    return registration


def _append_x11_v1_then_v2(
    root: Path,
    *,
    supersession_at: str = "2026-07-23T00:00:03Z",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v1 = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": v1},
    )
    v2 = _valid_x11_v2_reproduction_registration(root)
    amendment = _append_amendment(
        root,
        "X-11",
        amended_at=supersession_at,
        changes={
            "supersede_reproduction": {
                "supersedes_reproduction_id": v1["reproduction_id"],
                "registration": v2,
            }
        },
    )
    return v1, v2, amendment


def _write_pretty_json(path: Path, value: dict[str, Any]) -> str:
    rendered = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(rendered)
    return "sha256:" + hashlib.sha256(rendered).hexdigest()


def _resign_x02_bundle(
    root: Path,
    *,
    bundle_mutation: tuple[tuple[str, ...], Any] | None = None,
    day_mutation: tuple[str, tuple[str, ...], Any] | None = None,
) -> dict[str, str]:
    bundle_path = root / EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if day_mutation is not None:
        day, path, replacement = day_mutation
        entry = next(
            item for item in bundle["day_manifests"] if item["day"] == day
        )
        manifest_path = root / entry["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _set_nested(manifest, path, replacement)
        material = dict(manifest)
        material.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = _canonical_sha256(material)
        entry["full_day_manifest_sha256"] = manifest["manifest_sha256"]
        entry["artifact_file_sha256"] = _write_pretty_json(
            manifest_path, manifest
        )
    if bundle_mutation is not None:
        path, replacement = bundle_mutation
        _set_nested(bundle, path, replacement)
    material = dict(bundle)
    material.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _canonical_sha256(material)
    return {
        "bundle_path": EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"],
        "bundle_file_sha256": _write_pretty_json(bundle_path, bundle),
        "bundle_sha256": bundle["bundle_sha256"],
    }


def _x02_static_sidecar_path(
    root: Path,
    *,
    day_manifest_index: int = 0,
    object_index: int = 0,
) -> Path:
    entry = EXPECTED_X02_DAY_MANIFESTS[day_manifest_index]
    day_manifest = json.loads(
        (root / entry["path"]).read_text(encoding="utf-8")
    )
    hourly_object = day_manifest["objects"][object_index]
    object_parts = PurePosixPath(hourly_object["object_path"]).parts
    assert object_parts[:4] == (
        "raw",
        "source=pmxt",
        "dataset=DS-PMXT-V2",
        "version=v2",
    )
    static_digest = hourly_object[
        "static_manifest_sha256"
    ].removeprefix("sha256:")
    return (
        root
        / X02_STATIC_SIDECAR_ROOT
        / "manifests"
        / object_parts[1]
        / object_parts[2]
        / object_parts[3]
        / object_parts[4]
        / f"{static_digest}.manifest.json"
    )


def _set_nested(
    value: dict[str, Any],
    path: tuple[str, ...],
    replacement: Any,
) -> None:
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _valid_result_ref(
    *,
    scope: str = "archive_audit",
    result_label: str = "FORMAL",
    evaluation_started_at: str = "2026-07-23T00:00:03Z",
    registration_head_sha256: str = "sha256:" + "4" * 64,
    code_sha256: str = "sha256:" + "1" * 64,
    data_sha256: str = "sha256:" + "2" * 64,
    dataset_ids: list[str] | None = None,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "result_label": result_label,
        "evaluation_started_at": evaluation_started_at,
        "code_sha256": code_sha256,
        "data_sha256": data_sha256,
        "result_sha256": "sha256:" + "3" * 64,
        "registration_head_sha256": registration_head_sha256,
        "dataset_ids": (
            ["DS-KALSHI-HISTORICAL"] if dataset_ids is None else dataset_ids
        ),
        "model_ids": [] if model_ids is None else model_ids,
    }


LEDGER_FIELDS = [
    "experiment_id",
    "sequence",
    "record_sha256",
    "prior_sha256",
    "amended_at",
    "approved_by",
    "reason",
]


def _ensure_seed_ledger(root: Path) -> None:
    path = root / "registries" / "experiment_amendment_ledger.csv"
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for experiment_id in sorted(EXPECTED_EXPERIMENT_IDS):
            card = _read_card(root, experiment_id)
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "sequence": "0",
                    "record_sha256": card["registration_record_sha256"],
                    "prior_sha256": "",
                    "amended_at": "",
                    "approved_by": "",
                    "reason": "",
                }
            )


def _append_amendment(
    root: Path,
    experiment_id: str,
    *,
    amended_at: str,
    changes: dict[str, Any],
    approved_by: str = "H",
    reason: str = "test fixture",
) -> dict[str, Any]:
    _ensure_seed_ledger(root)
    card = _read_card(root, experiment_id)
    sequence = len(card["amendments"]) + 1
    prior_sha256 = (
        card["registration_record_sha256"]
        if not card["amendments"]
        else card["amendments"][-1]["amendment_sha256"]
    )
    amendment: dict[str, Any] = {
        "sequence": sequence,
        "amended_at": amended_at,
        "prior_sha256": prior_sha256,
        "approved_by": approved_by,
        "reason": reason,
        "changes": changes,
    }
    amendment["amendment_sha256"] = compute_amendment_sha256(amendment)
    card["amendments"].append(amendment)
    _write_card(root, experiment_id, card)
    _update_registry_card_hash(root, experiment_id)

    ledger_path = root / "registries" / "experiment_amendment_ledger.csv"
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writerow(
            {
                "experiment_id": experiment_id,
                "sequence": str(sequence),
                "record_sha256": amendment["amendment_sha256"],
                "prior_sha256": prior_sha256,
                "amended_at": amended_at,
                "approved_by": approved_by,
                "reason": reason,
            }
        )
    return amendment


def _resolve_reproduction_base_locks(
    root: Path,
    experiment_id: str,
    *,
    amended_at: str,
) -> dict[str, Any]:
    contract = REPRODUCTION_CONTRACTS[experiment_id]
    card = _read_card(root, experiment_id)
    lock_ids = card["authorization_scopes"][
        contract["base_scope"]
    ]["required_lock_ids"]
    return _append_amendment(
        root,
        experiment_id,
        amended_at=amended_at,
        changes={
            "resolve_locks": [
                {
                    "lock_id": lock_id,
                    "evidence_ref": _canonical_sha256(
                        {
                            "experiment_id": experiment_id,
                            "lock_id": lock_id,
                            "fixture": "explicit_resolution",
                        }
                    ),
                }
                for lock_id in lock_ids
            ]
        },
    )


def _rewrite_seed_amendment(
    root: Path,
    experiment_id: str,
    amendment: dict[str, Any],
) -> dict[str, Any]:
    card = _read_card(root, experiment_id)
    amendment = copy.deepcopy(amendment)
    amendment["amendment_sha256"] = compute_amendment_sha256(amendment)
    card["amendments"][0] = amendment
    _write_card(root, experiment_id, card)
    _update_registry_card_hash(root, experiment_id)

    ledger_path = root / "registries" / "experiment_amendment_ledger.csv"
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    ledger = next(
        row
        for row in rows
        if row["experiment_id"] == experiment_id
        and row["sequence"] == str(amendment["sequence"])
    )
    ledger.update(
        {
            "record_sha256": amendment["amendment_sha256"],
            "prior_sha256": amendment["prior_sha256"],
            "amended_at": amendment["amended_at"],
            "approved_by": amendment["approved_by"],
            "reason": amendment["reason"],
        }
    )
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return amendment


def _rewrite_x02_binding_amendment(
    root: Path,
    amendment: dict[str, Any],
) -> dict[str, Any]:
    card = _read_card(root, "X-02")
    amendment = copy.deepcopy(amendment)
    amendment["amendment_sha256"] = compute_amendment_sha256(amendment)
    card["amendments"][1] = amendment
    _write_card(root, "X-02", card)
    _update_registry_card_hash(root, "X-02")

    ledger_path = root / "registries" / "experiment_amendment_ledger.csv"
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    ledger = next(
        row
        for row in rows
        if row["experiment_id"] == "X-02"
        and row["sequence"] == "2"
    )
    ledger.update(
        {
            "record_sha256": amendment["amendment_sha256"],
            "prior_sha256": amendment["prior_sha256"],
            "amended_at": amendment["amended_at"],
            "approved_by": amendment["approved_by"],
            "reason": amendment["reason"],
        }
    )
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return amendment


def _preregister_x08_inputs(root: Path) -> dict[str, Any]:
    _update_license_review(
        root,
        "O-003",
        status="GREEN",
        commercial_use="PERMITTED_WITH_CONDITIONS",
        redistribution="PERMITTED_WITH_CONDITIONS",
        attribution_required="YES",
        operational_use="APPROVED",
        open_blocker="",
        approval_ref="I-APPROVAL-O003-TEST",
    )
    _update_dataset_registry(
        root,
        "DS-KALSHI-HISTORICAL",
        license_status="approved",
        manifest_sha256="sha256:" + "7" * 64,
    )
    return _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:02Z",
        changes={
            "resolve_locks": [
                {
                    "lock_id": "archive_audit_input_manifest",
                    "evidence_ref": "sha256:" + "8" * 64,
                },
                {
                    "lock_id": "h_split_approval",
                    "evidence_ref": "sha256:" + "9" * 64,
                },
            ],
            "preregistered_inputs": [
                {
                    "scope": "archive_audit",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": ["DS-KALSHI-HISTORICAL"],
                    "model_ids": [],
                }
            ],
        },
    )


def _preregister_x08_dual_venue_inputs(root: Path) -> dict[str, Any]:
    for review_id in ("O-001", "O-003"):
        _update_license_review(
            root,
            review_id,
            status="GREEN",
            commercial_use="PERMITTED_WITH_CONDITIONS",
            redistribution="PERMITTED_WITH_CONDITIONS",
            attribution_required="YES",
            operational_use="APPROVED",
            open_blocker="",
            approval_ref=f"I-APPROVAL-{review_id[2:]}-TEST",
        )
    for dataset_id in (
        "DS-KALSHI-HISTORICAL",
        "DS-KALSHI-LIVE-L2",
        "DS-POLYMARKET-PUBLIC",
    ):
        _update_dataset_registry(
            root,
            dataset_id,
            use_class="canonical",
            license_status="approved",
            manifest_sha256="sha256:" + "7" * 64,
            status="registered",
        )
    card = _read_card(root, "X-08")
    return _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:02Z",
        changes={
            "resolve_locks": [
                {
                    "lock_id": lock["id"],
                    "evidence_ref": "sha256:" + "8" * 64,
                }
                for lock in card["registration_locks"]
            ],
            "authorize_scopes": ["dual_venue_result"],
            "preregistered_inputs": [
                {
                    "scope": "dual_venue_result",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": [
                        "DS-KALSHI-HISTORICAL",
                        "DS-KALSHI-LIVE-L2",
                        "DS-POLYMARKET-PUBLIC",
                    ],
                    "model_ids": [],
                }
            ],
        },
    )


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _set_x08_validation_clock(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    monkeypatch.setattr(experiments_module, "_utc_now", lambda: instant)


def _write_x08_capture_evidence(
    root: Path,
    *,
    capture_session_id: str,
    started_at: str,
    ended_at: str,
    fixtures_used: bool = False,
    drop_heartbeat_at: str | None = None,
    sealed_at: str | None = None,
    include_dataset_ids: tuple[str, ...] = (
        "DS-KALSHI-LIVE-L2",
        "DS-POLYMARKET-PUBLIC",
    ),
) -> dict[str, str]:
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    segment_sealed_at = (
        _utc_text(ended + timedelta(milliseconds=500))
        if sealed_at is None
        else sealed_at
    )
    receive_times: list[datetime] = []
    instant = started
    while instant <= ended:
        receive_times.append(instant)
        instant += timedelta(seconds=60)
    if receive_times[-1] != ended:
        receive_times.append(ended)

    stream_specs = {
        "DS-KALSHI-LIVE-L2": ("kalshi", "orderbook"),
        "DS-POLYMARKET-PUBLIC": ("polymarket", "market"),
    }
    capture_dir = (
        root / "artifacts" / "capture-evidence" / capture_session_id
    )
    raw_store_root = capture_dir / "raw-store"
    streams: list[dict[str, Any]] = []
    for dataset_id in include_dataset_ids:
        source, stream = stream_specs[dataset_id]
        manifests: list[str] = []
        writers: dict[tuple[str, str], RawSegmentWriter] = {}
        for received_at in receive_times:
            received_text = _utc_text(received_at)
            if received_text == drop_heartbeat_at:
                continue
            partition = (
                received_at.date().isoformat(),
                f"{received_at.hour:02d}",
            )
            writer = writers.get(partition)
            if writer is None:
                writer = RawSegmentWriter(
                    raw_store_root,
                    source=source,
                    stream=stream,
                    capture_session_id=capture_session_id,
                )
                writers[partition] = writer
            writer.append(
                json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "kind": "recorder_heartbeat",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                receive_at=received_text,
            )
        with patch.object(
            raw_store_module,
            "_utc_now_text",
            return_value=segment_sealed_at,
        ):
            for partition in sorted(writers):
                segment = writers[partition].seal()
                manifests.append(
                    segment.path.relative_to(raw_store_root).as_posix()
                )
        streams.append(
            {
                "dataset_id": dataset_id,
                "source": source,
                "stream": stream,
                "segment_manifest_paths": manifests,
            }
        )

    document = {
        "schema_version": "x08-capture-evidence/v0",
        "experiment_id": "X-08",
        "capture_session_id": capture_session_id,
        "fixtures_used": fixtures_used,
        "gap_policy": {
            "policy_id": "recorder-heartbeat/v0",
            "maximum_gap_seconds": 60,
        },
        "raw_store_root": raw_store_root.relative_to(root).as_posix(),
        "streams": streams,
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_manifest = capture_dir / "capture-manifest.json"
    manifest_bytes = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    capture_manifest.write_bytes(manifest_bytes)
    manifest_ref = capture_manifest.relative_to(root).as_posix()

    artifact_registry = root / "registries" / "artifact_registry.csv"
    with artifact_registry.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows.append(
        {
            "artifact_id": f"ART-X08-{capture_session_id.upper()}",
            "path": manifest_ref,
            "owner_team": "C+B",
            "version": "v0",
            "due_gate": "first_round_artifact_gate",
            "status": "registered",
        }
    )
    with artifact_registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "capture_manifest_path": manifest_ref,
        "capture_manifest_sha256": (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        ),
    }


def _complete_scope(
    root: Path,
    experiment_id: str,
    scope: str,
    *,
    preregistered_at: str,
    evaluated_at: str,
    completed_at: str,
) -> dict[str, Any]:
    card = _read_card(root, experiment_id)
    binding = card["authorization_scopes"][scope]["input_binding"]
    dataset_path = root / "registries" / "dataset_registry.csv"
    with dataset_path.open(encoding="utf-8", newline="") as handle:
        datasets_by_id = {
            row["dataset_id"]: row for row in csv.DictReader(handle)
        }
    for dataset_id in binding["dataset_ids"]:
        review_id = datasets_by_id[dataset_id]["license_review_id"]
        _update_license_review(
            root,
            review_id,
            status="GREEN",
            commercial_use="PERMITTED_WITH_CONDITIONS",
            redistribution="PERMITTED_WITH_CONDITIONS",
            attribution_required="YES",
            operational_use="APPROVED",
            open_blocker="",
            approval_ref=f"I-APPROVAL-{review_id[2:]}-TEST",
        )
        _update_dataset_registry(
            root,
            dataset_id,
            use_class="canonical",
            license_status="approved",
            manifest_sha256="sha256:" + "7" * 64,
            status="registered",
        )
    effective_card = load_experiment_registry(root)[experiment_id]
    lock_status_by_id = {
        lock["id"]: lock["status"]
        for lock in effective_card["registration_locks"]
    }
    preregistration = _append_amendment(
        root,
        experiment_id,
        amended_at=preregistered_at,
        changes={
            "resolve_locks": [
                {
                    "lock_id": lock_id,
                    "evidence_ref": "sha256:" + "8" * 64,
                }
                for lock_id in card["authorization_scopes"][scope][
                    "required_lock_ids"
                ]
                if lock_status_by_id[lock_id] == "unresolved"
            ],
            "preregistered_inputs": [
                {
                    "scope": scope,
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": binding["dataset_ids"],
                    "model_ids": binding["model_ids"],
                }
            ],
        },
    )
    result_ref = _valid_result_ref(
        scope=scope,
        result_label=card["authorization_scopes"][scope]["required_result_label"],
        evaluation_started_at=evaluated_at,
        registration_head_sha256=preregistration["amendment_sha256"],
        dataset_ids=binding["dataset_ids"],
        model_ids=binding["model_ids"],
    )
    return _append_amendment(
        root,
        experiment_id,
        amended_at=completed_at,
        changes={"results_ref": result_ref, "status": "done"},
    )


def test_all_seed_experiments_are_registered(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)

    assert set(registry) == EXPECTED_EXPERIMENT_IDS
    assert {card["id"] for card in registry.values()} == EXPECTED_EXPERIMENT_IDS


def test_x02_seed_preregistration_is_exact_and_does_not_resolve_input_manifest(
    program_root: Path,
) -> None:
    card = _read_card(program_root, "X-02")
    assert card["registration_record_sha256"] == (
        "sha256:"
        "1b6393ef8cca4bf482cc3a167844358c07fda9a97c45cbedbe4ceda3e2033ed1"
    )
    assert card["registration_record_sha256"] == (
        compute_registration_record_sha256(card)
    )
    assert len(card["amendments"]) == 3
    amendment = card["amendments"][0]
    assert amendment["sequence"] == 1
    assert amendment["prior_sha256"] == card["registration_record_sha256"]
    assert amendment["approved_by"] == "H"
    assert amendment["amendment_sha256"] == compute_amendment_sha256(
        amendment
    )
    assert amendment["changes"]["timestamp_audit_preregistration"] == (
        EXPECTED_X02_PREREGISTRATION
    )
    resolved = {
        item["lock_id"]: item["evidence_ref"]
        for item in amendment["changes"]["resolve_locks"]
    }
    assert resolved == {
        lock_id: _canonical_sha256(section)
        for lock_id, section in EXPECTED_X02_PREREGISTRATION.items()
    }

    assert {
        item["lock_id"] for item in amendment["changes"]["resolve_locks"]
    } == {
        "sampling_and_seed",
        "diff_and_stability_definitions",
        "h_split_approval",
    }
    assert "timestamp_input_manifest_binding" not in amendment["changes"]
    with (
        program_root / "registries" / "experiment_amendment_ledger.csv"
    ).open(encoding="utf-8", newline="") as handle:
        ledger = [
            row
            for row in csv.DictReader(handle)
            if row["experiment_id"] == "X-02"
        ]
    assert [row["sequence"] for row in ledger] == ["0", "1", "2", "3"]
    assert ledger[0]["record_sha256"] == card["registration_record_sha256"]
    assert ledger[1]["record_sha256"] == amendment["amendment_sha256"]


def test_x02_input_manifest_and_formal_inputs_are_exact(
    program_root: Path,
) -> None:
    card = _read_card(program_root, "X-02")
    seed, binding_amendment, input_amendment = card["amendments"]

    amendment_times = [
        datetime.fromisoformat(
            item["amended_at"].removesuffix("Z") + "+00:00"
        )
        for item in card["amendments"]
    ]
    assert amendment_times == sorted(amendment_times)
    assert all(item <= datetime.now(timezone.utc) for item in amendment_times)
    assert binding_amendment["sequence"] == 2
    assert binding_amendment["prior_sha256"] == seed["amendment_sha256"]
    assert binding_amendment["approved_by"] == "H"
    assert binding_amendment["amendment_sha256"] == (
        compute_amendment_sha256(binding_amendment)
    )
    assert set(binding_amendment["changes"]) == {
        "resolve_locks",
        "timestamp_input_manifest_binding",
    }
    assert binding_amendment["changes"][
        "timestamp_input_manifest_binding"
    ] == EXPECTED_X02_INPUT_MANIFEST_BINDING
    assert binding_amendment["changes"]["resolve_locks"] == [
        {
            "lock_id": "timestamp_input_manifest",
            "evidence_ref": EXPECTED_X02_INPUT_MANIFEST_BINDING[
                "bundle_sha256"
            ],
        }
    ]
    assert input_amendment["sequence"] == 3
    assert input_amendment["prior_sha256"] == binding_amendment[
        "amendment_sha256"
    ]
    assert input_amendment["approved_by"] == "H"
    assert input_amendment["amendment_sha256"] == (
        compute_amendment_sha256(input_amendment)
    )
    assert input_amendment["changes"] == {
        "preregistered_inputs": [EXPECTED_X02_FORMAL_INPUT]
    }

    effective = load_experiment_registry(program_root)["X-02"]
    assert effective["timestamp_audit_preregistration"] == (
        EXPECTED_X02_PREREGISTRATION
    )
    assert effective["timestamp_input_manifest_binding"] == (
        EXPECTED_X02_INPUT_MANIFEST_BINDING
    )
    assert {
        lock["id"]: lock["status"]
        for lock in effective["registration_locks"]
    } == {
        "timestamp_input_manifest": "resolved",
        "sampling_and_seed": "resolved",
        "diff_and_stability_definitions": "resolved",
        "h_split_approval": "resolved",
    }
    assert effective["preregistered_inputs"] == {
        "formal_result": {
            **{
                key: value
                for key, value in EXPECTED_X02_FORMAL_INPUT.items()
                if key != "scope"
            },
            "registered_at": input_amendment["amended_at"],
        }
    }
    assert effective["results_ref"] == []
    assert effective["status"] == "registered"


def test_x02_bundle_verifier_checks_all_four_days_and_static_references(
    program_root: Path,
) -> None:
    summary = experiments_module._verify_x02_timestamp_input_bundle(
        program_root,
        EXPECTED_X02_INPUT_MANIFEST_BINDING,
    )

    assert summary == {
        "bundle_sha256": EXPECTED_X02_INPUT_MANIFEST_BINDING[
            "bundle_sha256"
        ],
        "days": [
            "2026-04-22",
            "2026-05-28",
            "2026-06-05",
            "2026-06-25",
        ],
        "object_count": 96,
        "static_manifest_reference_count": 96,
        "verified_static_sidecar_count": 96,
    }


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        (
            "bundle_path",
            "../x02_timestamp_input_bundle_v1.json",
            "path escape|binding",
        ),
        (
            "bundle_file_sha256",
            "sha256:" + "a" * 64,
            "bundle file SHA-256",
        ),
        (
            "bundle_sha256",
            "sha256:" + "b" * 64,
            "bundle self-hash|binding",
        ),
    ],
)
def test_x02_bundle_verifier_rejects_binding_tampering(
    program_root: Path,
    field: str,
    replacement: str,
    match: str,
) -> None:
    binding = dict(EXPECTED_X02_INPUT_MANIFEST_BINDING)
    binding[field] = replacement

    with pytest.raises(ExperimentRegistryError, match=match):
        experiments_module._verify_x02_timestamp_input_bundle(
            program_root,
            binding,
        )


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    [
        (("selection_seed",), 20260723, "selection|bundle"),
        (("day_count",), 3, "day|bundle"),
        (("object_count",), 95, "object|bundle"),
        (("formal_result",), True, "formal|bundle"),
        (
            ("day_manifests", "0", "day"),
            "2026-04-23",
            "day|bundle",
        ),
    ],
)
def test_x02_bundle_verifier_rejects_resigned_semantic_tampering(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    match: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    normalized_path = tuple(
        int(item) if item.isdigit() else item for item in path
    )
    bundle_path = root / EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    current: Any = bundle
    for item in normalized_path[:-1]:
        current = current[item]
    current[normalized_path[-1]] = replacement
    material = dict(bundle)
    material.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _canonical_sha256(material)
    binding = {
        "bundle_path": EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"],
        "bundle_file_sha256": _write_pretty_json(bundle_path, bundle),
        "bundle_sha256": bundle["bundle_sha256"],
    }

    with pytest.raises(ExperimentRegistryError, match=match):
        experiments_module._verify_x02_timestamp_input_bundle(root, binding)


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    [
        (
            ("objects", "0", "static_manifest_sha256"),
            "sha256:" + "c" * 64,
            "manifest|day",
        ),
        (
            ("objects", "0", "hour"),
            "2026-04-22T01:00:00Z",
            "hour|ordered",
        ),
        (
            ("objects", "0", "static_manifest_sha256"),
            "not-a-sha256",
            "static_manifest_sha256",
        ),
    ],
)
def test_x02_bundle_verifier_rejects_resigned_day_manifest_tampering(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    match: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    normalized_path = tuple(
        int(item) if item.isdigit() else item for item in path
    )
    bundle_path = root / EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in bundle["day_manifests"]
        if item["day"] == "2026-04-22"
    )
    manifest_path = root / entry["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current: Any = manifest
    for item in normalized_path[:-1]:
        current = current[item]
    current[normalized_path[-1]] = replacement
    material = dict(manifest)
    material.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_sha256(material)
    entry["full_day_manifest_sha256"] = manifest["manifest_sha256"]
    entry["artifact_file_sha256"] = _write_pretty_json(
        manifest_path, manifest
    )
    bundle_material = dict(bundle)
    bundle_material.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _canonical_sha256(bundle_material)
    binding = {
        "bundle_path": EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"],
        "bundle_file_sha256": _write_pretty_json(bundle_path, bundle),
        "bundle_sha256": bundle["bundle_sha256"],
    }

    with pytest.raises(ExperimentRegistryError, match=match):
        experiments_module._verify_x02_timestamp_input_bundle(root, binding)


@pytest.mark.parametrize("failure", ["missing", "symlink"])
def test_x02_bundle_verifier_rejects_missing_or_symlinked_day_manifest(
    tmp_path: Path,
    failure: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    path = (
        root
        / "artifacts/data-audit/"
        "x02_full_day_input_manifest_2026-06-05_v1.json"
    )
    if failure == "missing":
        path.unlink()
    else:
        target = path.with_name("x02-day-target.json")
        path.rename(target)
        path.symlink_to(target)

    with pytest.raises(ExperimentRegistryError, match="missing|symlink"):
        experiments_module._verify_x02_timestamp_input_bundle(
            root,
            EXPECTED_X02_INPUT_MANIFEST_BINDING,
        )


@pytest.mark.parametrize(
    "failure",
    ["missing", "symlink", "tampered", "wrong_reference"],
)
def test_x02_bundle_verifier_opens_and_validates_every_static_sidecar(
    tmp_path: Path,
    failure: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    sidecar = _x02_static_sidecar_path(root)
    assert sidecar.is_file()
    if failure == "missing":
        sidecar.unlink()
    elif failure == "symlink":
        target = sidecar.with_name("sidecar-target.manifest.json")
        sidecar.rename(target)
        sidecar.symlink_to(target.name)
    elif failure == "tampered":
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        document["native_object_path"] = (
            document["native_object_path"] + ".tampered"
        )
        _write_pretty_json(sidecar, document)
    elif failure == "wrong_reference":
        wrong_sidecar = _x02_static_sidecar_path(
            root,
            object_index=1,
        )
        sidecar.write_bytes(wrong_sidecar.read_bytes())

    with pytest.raises(
        ExperimentRegistryError,
        match="static manifest|sidecar|symlink|missing",
    ):
        experiments_module._verify_x02_timestamp_input_bundle(
            root,
            EXPECTED_X02_INPUT_MANIFEST_BINDING,
        )


def test_x02_inventory_size_is_documented_as_listing_estimate(
    program_root: Path,
) -> None:
    design = (
        program_root
        / "docs/superpowers/specs/"
        "2026-07-22-x02-timestamp-preregistration-design.md"
    ).read_text(encoding="utf-8")

    assert (
        "`inventory_size_bytes` is an archive-listing estimate"
        in design
    )
    assert (
        "must not be reported as an exact object byte length"
        in design
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("sampling_and_seed", "selection_seed"), 20260723),
        (("sampling_and_seed", "game_day"), "2026-05-29"),
        (
            ("sampling_and_seed", "random_days"),
            ["2026-04-22", "2026-06-05", "2026-06-26"],
        ),
        (
            ("sampling_and_seed", "selection_inventory_sha256"),
            "sha256:" + "a" * 64,
        ),
        (
            ("sampling_and_seed", "x01_game_day_manifest_sha256"),
            "sha256:" + "b" * 64,
        ),
        (("sampling_and_seed", "pending_archive_object_count"), 71),
        (
            ("diff_and_stability_definitions", "diff_ms"),
            "epoch_ms(timestamp)-epoch_ms(timestamp_received)",
        ),
        (
            (
                "diff_and_stability_definitions",
                "signed_quantiles",
                "estimator",
            ),
            "quantile_disc",
        ),
        (
            (
                "diff_and_stability_definitions",
                "signed_quantiles",
                "interpolation",
            ),
            "nearest",
        ),
        (
            (
                "diff_and_stability_definitions",
                "signed_quantiles",
                "input_distribution",
            ),
            "raw_float_rows",
        ),
        (
            (
                "diff_and_stability_definitions",
                "signed_quantiles",
                "probabilities",
            ),
            ["0.50", "0.90", "0.99"],
        ),
        (
            (
                "diff_and_stability_definitions",
                "absolute_p99",
                "transform",
            ),
            "diff_ms",
        ),
        (
            (
                "diff_and_stability_definitions",
                "absolute_p99",
                "probability",
            ),
            "0.95",
        ),
        (
            ("diff_and_stability_definitions", "hourly_drift_ms"),
            "median_utc_hour_00_diff_ms-median_utc_hour_23_diff_ms",
        ),
        (
            (
                "diff_and_stability_definitions",
                "disorder",
                "partition_by",
            ),
            ["asset_id"],
        ),
        (
            (
                "diff_and_stability_definitions",
                "disorder",
                "canonical_sort",
            ),
            ["timestamp", "timestamp_received"],
        ),
        (
            (
                "diff_and_stability_definitions",
                "disorder",
                "numerator",
            ),
            "adjacent_source_timestamp_nonincreasing_pairs",
        ),
        (
            (
                "diff_and_stability_definitions",
                "disorder",
                "ordered_comparisons",
            ),
            "n_rows",
        ),
        (
            (
                "diff_and_stability_definitions",
                "downgrade_gate",
                "negative_rate_gte",
            ),
            "0.01",
        ),
        (
            (
                "diff_and_stability_definitions",
                "downgrade_gate",
                "absolute_p99_ms_gt",
            ),
            5001,
        ),
        (
            (
                "diff_and_stability_definitions",
                "downgrade_gate",
                "decision",
            ),
            "continue_millisecond_research",
        ),
        (("h_split_approval", "split"), "walk_forward"),
        (
            ("h_split_approval", "basis"),
            "unregistered_exception",
        ),
        (("h_split_approval", "approved_by"), "C"),
    ],
)
def test_x02_preregistration_rejects_any_definition_mutation(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-02")
    amendment = copy.deepcopy(card["amendments"][0])
    preregistration = amendment["changes"][
        "timestamp_audit_preregistration"
    ]
    _set_nested(preregistration, path, replacement)
    for resolved in amendment["changes"]["resolve_locks"]:
        resolved["evidence_ref"] = _canonical_sha256(
            preregistration[resolved["lock_id"]]
        )
    _rewrite_seed_amendment(root, "X-02", amendment)

    with pytest.raises(
        ExperimentRegistryError,
        match="timestamp audit preregistration.*exact",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_structured_change",
        "missing_lock_resolution",
        "extra_input_manifest_resolution",
        "wrong_section_evidence_hash",
        "status_co_mutation",
        "results_co_mutation",
        "preregistered_inputs_co_mutation",
    ],
)
def test_x02_preregistration_is_coupled_to_exact_lock_resolution(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-02")
    amendment = copy.deepcopy(card["amendments"][0])
    changes = amendment["changes"]
    if mutation == "missing_structured_change":
        changes.pop("timestamp_audit_preregistration")
    elif mutation == "missing_lock_resolution":
        changes["resolve_locks"].pop()
    elif mutation == "extra_input_manifest_resolution":
        changes["resolve_locks"].append(
            {
                "lock_id": "timestamp_input_manifest",
                "evidence_ref": "sha256:" + "c" * 64,
            }
        )
    elif mutation == "wrong_section_evidence_hash":
        changes["resolve_locks"][0]["evidence_ref"] = "sha256:" + "d" * 64
    elif mutation == "status_co_mutation":
        changes["status"] = "running"
    elif mutation == "results_co_mutation":
        changes["results_ref"] = _valid_result_ref(
            scope="formal_result",
            registration_head_sha256=amendment["prior_sha256"],
            dataset_ids=["DS-PMXT-V2"],
        )
    elif mutation == "preregistered_inputs_co_mutation":
        changes["preregistered_inputs"] = [
            {
                "scope": "formal_result",
                "code_sha256": "sha256:" + "1" * 64,
                "data_sha256": "sha256:" + "2" * 64,
                "dataset_ids": ["DS-PMXT-V2"],
                "model_ids": [],
            }
        ]
    _rewrite_seed_amendment(root, "X-02", amendment)

    with pytest.raises(
        ExperimentRegistryError,
        match="X-02.*preregistration|preregistration.*lock",
    ):
        load_experiment_registry(root)


def test_x02_timestamp_audit_preregistration_is_append_once(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-02",
        amended_at="2026-07-24T00:00:01Z",
        changes={
            "resolve_locks": [
                {
                    "lock_id": lock_id,
                    "evidence_ref": _canonical_sha256(section),
                }
                for lock_id, section in (
                    EXPECTED_X02_PREREGISTRATION.items()
                )
            ],
            "timestamp_audit_preregistration": copy.deepcopy(
                EXPECTED_X02_PREREGISTRATION
            )
        },
    )

    with pytest.raises(ExperimentRegistryError, match="append-once"):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_binding",
        "missing_lock_resolution",
        "wrong_evidence",
        "extra_lock_resolution",
        "status_co_mutation",
        "result_co_mutation",
        "preregistered_inputs_co_mutation",
    ],
)
def test_x02_input_manifest_binding_is_coupled_to_only_its_lock(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = copy.deepcopy(
        _read_card(root, "X-02")["amendments"][1]
    )
    changes = amendment["changes"]
    if mutation == "missing_binding":
        changes.pop("timestamp_input_manifest_binding")
    elif mutation == "missing_lock_resolution":
        changes["resolve_locks"] = []
    elif mutation == "wrong_evidence":
        changes["resolve_locks"][0]["evidence_ref"] = (
            "sha256:" + "d" * 64
        )
    elif mutation == "extra_lock_resolution":
        changes["resolve_locks"].append(
            {
                "lock_id": "sampling_and_seed",
                "evidence_ref": "sha256:" + "e" * 64,
            }
        )
    elif mutation == "status_co_mutation":
        changes["status"] = "running"
    elif mutation == "result_co_mutation":
        changes["results_ref"] = _valid_result_ref(
            scope="formal_result",
            registration_head_sha256=amendment["prior_sha256"],
            dataset_ids=["DS-PMXT-V2"],
        )
    elif mutation == "preregistered_inputs_co_mutation":
        changes["preregistered_inputs"] = [
            {
                "scope": "formal_result",
                "code_sha256": "sha256:" + "1" * 64,
                "data_sha256": "sha256:" + "2" * 64,
                "dataset_ids": ["DS-PMXT-V2"],
                "model_ids": [],
            }
        ]
    _rewrite_x02_binding_amendment(root, amendment)

    with pytest.raises(
        ExperimentRegistryError,
        match="X-02.*input manifest|input manifest.*binding|controlled",
    ):
        load_experiment_registry(root)


def test_x02_timestamp_input_manifest_binding_is_append_once(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-02",
        amended_at="2026-07-24T00:00:01Z",
        changes={
            "resolve_locks": [
                {
                    "lock_id": "timestamp_input_manifest",
                    "evidence_ref": EXPECTED_X02_INPUT_MANIFEST_BINDING[
                        "bundle_sha256"
                    ],
                }
            ],
            "timestamp_input_manifest_binding": copy.deepcopy(
                EXPECTED_X02_INPUT_MANIFEST_BINDING
            ),
        },
    )

    with pytest.raises(ExperimentRegistryError, match="append-once|already resolved"):
        load_experiment_registry(root)


def test_x02_formal_result_remains_blocked_by_x01_after_input_preregistration(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    effective = load_experiment_registry(root)["X-02"]
    assert next(
        lock for lock in effective["registration_locks"]
        if lock["id"] == "timestamp_input_manifest"
    )["status"] == "resolved"
    registered = effective["preregistered_inputs"]["formal_result"]
    result = _valid_result_ref(
        scope="formal_result",
        evaluation_started_at="2026-07-24T00:00:00Z",
        registration_head_sha256=effective[
            "registration_head_sha256"
        ],
        code_sha256=registered["code_sha256"],
        data_sha256=registered["data_sha256"],
        dataset_ids=["DS-PMXT-V2"],
    )

    with pytest.raises(
        UnresolvedDependencyError,
        match="dependenc",
    ):
        validate_result_ref(root, "X-02", result)


def test_registry_filename_card_and_csv_ids_are_identical(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)
    card_filenames = {
        path.stem
        for path in (program_root / "registries" / "experiments").glob("*.yaml")
    }
    with (program_root / "registries" / "experiment_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_ids = {row["experiment_id"] for row in csv.DictReader(handle)}

    assert set(registry) == card_filenames == csv_ids == EXPECTED_EXPERIMENT_IDS


def test_registration_has_no_experiment_deadlines(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)
    assert all(card["due_gate"] is None for card in registry.values())

    with (program_root / "registries" / "experiment_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert all(row["due_gate"] == "" for row in csv.DictReader(handle))


def test_catalog_first_artifact_gates_use_stable_catalog_ids(
    program_root: Path,
) -> None:
    registry = load_experiment_registry(program_root)
    with (program_root / "charter" / "catalog_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        catalog_rows = list(csv.DictReader(handle))

    for experiment_id, card in registry.items():
        expected = [
            {
                "catalog_item_id": row["catalog_item_id"],
                "due_gate": row["due_gate"],
            }
            for row in catalog_rows
            if experiment_id in row["linked_experiments"].split(";")
        ]
        assert card["linked_first_artifact_due_gates"] == expected
        assert card["source_lineage"]["catalog_item_ids"] == [
            item["catalog_item_id"] for item in expected
        ]

    assert registry["X-02"]["linked_first_artifact_due_gates"] == []


def test_all_point_in_time_inputs_have_explicit_lineage(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)

    for card in registry.values():
        assert card["source_lineage"]["charter_file"] == (
            "charter/research_program_charter_v0.2.md"
        )
        assert card["source_lineage"]["charter_sections"]
        for data_input in card["data"]:
            assert data_input["source"].strip()
            assert data_input["version"].strip()
            assert data_input["pit_basis"].strip()


def test_only_charter_measurement_exemptions_are_marked(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)
    exemptions = {
        experiment_id
        for experiment_id, card in registry.items()
        if card["measurement_exemption"]
    }

    assert exemptions == EXPECTED_MEASUREMENT_EXEMPTIONS
    for experiment_id in exemptions:
        assert registry[experiment_id]["falsified_direction_is_valid_measurement"] is True


def test_partial_authorization_scopes_fail_closed(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)

    x05 = registry["X-05"]
    assert x05["execution_authorized"] is False
    assert x05["authorization_scopes"]["spec_drafting"]["authorized"] is True
    assert x05["authorization_scopes"]["label_generation"]["authorized"] is False
    assert x05["authorization_scopes"]["formal_result"]["authorized"] is False

    x07 = registry["X-07"]
    assert x07["execution_authorized"] is False
    assert x07["authorization_scopes"]["preliminary_pipeline"]["authorized"] is True
    assert (
        x07["authorization_scopes"]["preliminary_pipeline"]["required_result_label"]
        == "PRELIMINARY"
    )
    assert x07["authorization_scopes"]["formal_result"]["authorized"] is False
    assert x07["authorization_scopes"]["go_no_go"]["authorized"] is False

    x08 = registry["X-08"]
    assert x08["execution_authorized"] is True
    assert x08["authorization_scopes"]["archive_audit"]["authorized"] is True
    assert x08["authorization_scopes"]["polymarket_capture"]["authorized"] is True
    assert x08["authorization_scopes"]["kalshi_capture"]["authorized"] is False
    assert x08["authorization_scopes"]["dual_venue_result"]["authorized"] is False

    x10 = registry["X-10"]
    assert x10["execution_authorized"] is False
    assert x10["authorization_scopes"]["precision_audit"]["authorized"] is True
    assert set(
        x10["authorization_scopes"]["precision_audit"]["required_lock_ids"]
    ) == {
        "matched_sample_registered",
        "router_and_taxonomy_available",
        "gold_standard_protocol",
        "h_split_approval",
    }
    assert x10["authorization_scopes"]["recall"]["authorized"] is False
    assert x10["authorization_scopes"]["live_arbitrage"]["authorized"] is False
    assert x10["authorization_scopes"]["live_arbitrage"]["permanent_no_go"] is True

    for experiment_id in EXPECTED_EXPERIMENT_IDS - {
        "X-05",
        "X-06",
        "X-07",
        "X-08",
        "X-04",
        "X-09",
        "X-10",
        "X-11",
        "X-12",
    }:
        assert registry[experiment_id]["execution_authorized"] is True


def test_x06_uses_prior_as_model_input_and_is_license_blocked(
    program_root: Path,
) -> None:
    x06 = load_experiment_registry(program_root)["X-06"]

    assert x06["execution_authorized"] is True
    assert x06["authorization_scopes"]["formal_result"]["authorized"] is False
    assert "nba_license_clearance" in {
        lock["id"] for lock in x06["registration_locks"]
    }
    assert all(lock["status"] == "unresolved" for lock in x06["registration_locks"])
    assert "prior as an input feature" in x06["method"].lower()
    assert x06["dataset_ids"] == ["DS-NBA-CANDIDATE"]
    assert x06["output_contract"] == {
        "contract": "model-output/v1.schema.yaml",
        "output_kind": "state_transition",
        "transition_unit": "possession",
        "state_space": ["home_score", "away_score", "no_score"],
    }


def test_x06_contract_harness_is_bound_to_registered_synthetic_fixture(
    program_root: Path,
) -> None:
    x06 = load_experiment_registry(program_root)["X-06"]
    fixture = x06["synthetic_fixture"]
    fixture_path = program_root / fixture["path"]

    assert fixture["data_class"] == "synthetic_contract_only"
    assert fixture_path.is_file()
    assert fixture["sha256"] == (
        "sha256:" + hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    )
    assert x06["authorization_scopes"]["contract_harness"]["input_binding"] == {
        "result_class": "synthetic",
        "dataset_ids": [],
        "model_ids": ["MODEL-NBA-POSSESSION-TRANSITION"],
        "synthetic_data_sha256": fixture["sha256"],
    }


def test_x06_synthetic_contract_harness_accepts_the_exact_fixture_end_to_end(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    fixture_hash = (
        "sha256:"
        "0ebca90ba56b45252c6d310fd176ba2eb1ac615995b3eacf92e2e3f9c962a15f"
    )
    model_ids = ["MODEL-NBA-POSSESSION-TRANSITION"]
    preregistration = _append_amendment(
        root,
        "X-06",
        amended_at="2026-07-23T00:00:01Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": "contract_harness",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": fixture_hash,
                    "dataset_ids": [],
                    "model_ids": model_ids,
                }
            ]
        },
    )
    result = _valid_result_ref(
        scope="contract_harness",
        result_label="PRELIMINARY",
        registration_head_sha256=preregistration["amendment_sha256"],
        data_sha256=fixture_hash,
        dataset_ids=[],
        model_ids=model_ids,
    )

    assert validate_result_ref(root, "X-06", result) == result


def test_x06_synthetic_contract_harness_rejects_arbitrary_data_hash(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-06",
        amended_at="2026-07-23T00:00:01Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": "contract_harness",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": [],
                    "model_ids": ["MODEL-NBA-POSSESSION-TRANSITION"],
                }
            ]
        },
    )

    with pytest.raises(ExperimentRegistryError, match="fixture|synthetic"):
        load_experiment_registry(root)


def test_x11_preregisters_nfl_walk_forward_ties_and_required_locks(
    program_root: Path,
) -> None:
    x11 = load_experiment_registry(program_root)["X-11"]
    combined = " ".join(
        [x11["method"], x11["split"], x11["pass_criteria"], x11["tie_policy"]]
    ).lower()

    assert "2015-2025" in combined
    assert "2015-2019" in combined
    assert "2020-2025" in combined
    assert "regular" in combined and "postseason" in combined
    assert "game-grouped" in combined and "chronological" in combined
    assert all(
        name in combined
        for name in ("spread", "logistic", "gbdt", "nflfastr")
    )
    assert "ties" in combined and "binary" in combined
    assert x11["execution_authorized"] is True
    assert x11["completion_required_scopes"] == [
        "preregistered_pipeline"
    ]
    assert x11["authorization_scopes"]["preregistered_pipeline"]["authorized"] is True
    assert x11["authorization_scopes"]["formal_result"]["authorized"] is False
    assert x11["authorization_scopes"]["formal_result"]["permanent_no_go"] is True
    assert x11["dataset_ids"] == ["DS-NFLVERSE"]
    assert x11["output_contract"]["transition_unit"] == "drive"
    lock_ids = {lock["id"] for lock in x11["registration_locks"]}
    assert {
        "nfl_data_manifest_and_version",
        "pit_feature_contract",
        "model_config_and_seed",
        "bootstrap_parameters",
    } <= lock_ids


def test_x11_completion_can_only_use_the_pipeline_scope(
    program_root: Path,
) -> None:
    x11 = _read_card(program_root, "X-11")

    assert x11["completion_required_scopes"] == [
        "preregistered_pipeline"
    ]
    assert x11["authorization_scopes"]["preregistered_pipeline"][
        "authorized"
    ] is True
    assert x11["authorization_scopes"]["formal_result"][
        "permanent_no_go"
    ] is True


@pytest.mark.parametrize("experiment_id", ["X-11", "X-12"])
def test_team_h_can_register_an_exact_preliminary_poc_reproduction(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    base = _read_card(root, experiment_id)
    registration = _valid_reproduction_registration(
        root, experiment_id
    )
    amendment = _append_amendment(
        root,
        experiment_id,
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    effective = load_experiment_registry(root)[experiment_id]
    scope = effective["authorization_scopes"][registration["scope"]]
    model_ids = [
        binding["model_id"]
        for binding in registration["model_bindings"]
    ]
    dedicated_locks = [
        lock
        for lock in effective["registration_locks"]
        if lock["id"]
        not in {item["id"] for item in base["registration_locks"]}
    ]
    base_lock_ids = (
        base["authorization_scopes"][
            REPRODUCTION_CONTRACTS[experiment_id]["base_scope"]
        ]["required_lock_ids"]
        if REPRODUCTION_CONTRACTS[experiment_id].get(
            "inherit_base_locks", True
        )
        else []
    )

    assert registration["code_paths"] == REPRODUCTION_CONTRACTS[
        experiment_id
    ]["code_paths"]
    assert registration["code_sha256"] == _reproduction_code_sha256(
        root,
        registration["code_paths"],
    )
    datasets = {
        row.dataset_id: row for row in load_dataset_registry(root)
    }
    assert registration["data_sha256"] == _canonical_sha256(
        [
            {
                "dataset_id": dataset_id,
                "manifest_sha256": datasets[dataset_id].manifest_sha256,
            }
            for dataset_id in registration["dataset_ids"]
        ]
    )
    assert (
        effective["registration_record_sha256"]
        == base["registration_record_sha256"]
    )
    assert all(
        effective["authorization_scopes"][scope_name] == base_scope
        for scope_name, base_scope in base["authorization_scopes"].items()
    )
    assert scope == {
        "authorized": True,
        "required_result_label": "PRELIMINARY",
        "required_lock_ids": [
            *base_lock_ids,
            dedicated_locks[0]["id"],
        ],
        "input_binding": {
            "result_class": "poc",
            "dataset_ids": registration["dataset_ids"],
            "model_ids": model_ids,
            "synthetic_data_sha256": None,
        },
    }
    assert dedicated_locks == [
        {
            "id": f"reproduction:{registration['reproduction_id']}",
            "status": "resolved",
            "reason": (
                f"Team H registered reproduction "
                f"{registration['reproduction_id']} before evaluation."
            ),
            "evidence_ref": registration[
                "reproduction_spec_sha256"
            ],
        }
    ]
    assert effective["preregistered_inputs"][registration["scope"]] == {
        "code_sha256": registration["code_sha256"],
        "data_sha256": registration["data_sha256"],
        "dataset_ids": registration["dataset_ids"],
        "model_ids": model_ids,
        "registered_at": "2026-07-23T00:00:02Z",
    }

    result = _valid_result_ref(
        scope=registration["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:03Z",
        registration_head_sha256=amendment["amendment_sha256"],
        code_sha256=registration["code_sha256"],
        data_sha256=registration["data_sha256"],
        dataset_ids=registration["dataset_ids"],
        model_ids=model_ids,
    )
    if base_lock_ids:
        with pytest.raises(
            UnresolvedRegistrationLockError,
            match="unresolved registration locks",
        ):
            validate_result_ref(root, experiment_id, result)

        resolution_amendment = _resolve_reproduction_base_locks(
            root,
            experiment_id,
            amended_at="2026-07-23T00:00:03Z",
        )
        result["evaluation_started_at"] = "2026-07-23T00:00:04Z"
        result["registration_head_sha256"] = resolution_amendment[
            "amendment_sha256"
        ]
    assert validate_result_ref(root, experiment_id, result) == result


def test_x11_reproduction_is_no_spread_only_and_does_not_inherit_prior_locks(
    program_root: Path,
) -> None:
    effective = load_experiment_registry(program_root)["X-11"]
    scope_name = str(X11_CURRENT_REPRODUCTION_CONTRACT["scope"])
    scope = effective["authorization_scopes"][scope_name]
    supersession = effective["amendments"][-1]["changes"][
        "supersede_reproduction"
    ]
    registration = supersession["registration"]
    reproduction_locks = [
        lock["id"]
        for lock in effective["registration_locks"]
        if lock["id"]
        == f"reproduction:{registration['reproduction_id']}"
    ]

    assert supersession["supersedes_reproduction_id"] == (
        X11_V1_REPRODUCTION_ID
    )
    assert scope["input_binding"]["dataset_ids"] == [
        "DS-NFL-FASTRMODELS",
        "DS-NFLVERSE",
    ]
    assert scope["input_binding"]["model_ids"] == [
        "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1"
    ]
    assert scope["required_lock_ids"] == reproduction_locks
    assert "spread_prior_manifest" not in scope["required_lock_ids"]
    assert registration["protocol_path"] == (
        "registries/protocols/x11_fastrmodels_no_spread_v1.json"
    )
    protocol_bytes = (
        program_root / registration["protocol_path"]
    ).read_bytes()
    assert registration["protocol_sha256"] == (
        "sha256:" + hashlib.sha256(protocol_bytes).hexdigest()
    )
    assert "MODEL-NFL-FASTRMODELS-SPREAD" not in json.dumps(
        scope,
        sort_keys=True,
    )


def test_checked_in_x11_v2_supersession_preserves_v1_history(
    program_root: Path,
) -> None:
    card = _read_card(program_root, "X-11")
    assert card["results_ref"] == []
    assert len(card["amendments"]) == 2

    v1 = card["amendments"][0]
    assert v1["amendment_sha256"] == X11_V1_AMENDMENT_SHA256
    assert v1["amended_at"] == "2026-07-23T23:49:01Z"
    assert v1["changes"]["register_reproduction"] == {
        "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V1",
        "scope": "team_h_nfl_fastrmodels_reproduction_v1",
        "result_class": "poc",
        "dataset_ids": ["DS-NFL-FASTRMODELS", "DS-NFLVERSE"],
        "model_bindings": [
            {
                "model_id": "MODEL-NFL-FASTRMODELS-NO-SPREAD",
                "model_version": "v0",
                "model_record_sha256": (
                    "sha256:"
                    "4939626af8975ad8cfddb60190af905c87812eb73f167eb4baa1721e9cde91ec"
                ),
            }
        ],
        "code_paths": [
            "src/prediction_market/models/nfl.py",
            "src/prediction_market/models/nfl_fastrmodels.py",
        ],
        "code_sha256": (
            "sha256:"
            "5cd141dc145b44a88e950ef297c9f418a8ad91efde503e10f07ace63311c85cd"
        ),
        "data_sha256": (
            "sha256:"
            "6342fceb33fba8b7f2f3b601f85e11a8e201898419013d0e1a46f9f66623fbc4"
        ),
        "protocol_path": (
            "registries/protocols/"
            "x11_fastrmodels_no_spread_v0.json"
        ),
        "protocol_sha256": X11_V1_PROTOCOL_FILE_SHA256,
        "reproduction_spec_sha256": (
            "sha256:"
            "bfe89b09cbefbfca701bdb2537c14ccb3d92acbd77af4bf37e785db78500371a"
        ),
    }
    assert (
        "sha256:"
        + hashlib.sha256(
            (
                program_root
                / "registries/protocols/"
                "x11_fastrmodels_no_spread_v0.json"
            ).read_bytes()
        ).hexdigest()
        == X11_V1_PROTOCOL_FILE_SHA256
    )

    v2 = card["amendments"][1]
    assert v2["prior_sha256"] == X11_V1_AMENDMENT_SHA256
    assert v2["changes"]["supersede_reproduction"][
        "supersedes_reproduction_id"
    ] == X11_V1_REPRODUCTION_ID
    registration = v2["changes"]["supersede_reproduction"][
        "registration"
    ]
    assert registration["reproduction_id"] == (
        X11_CURRENT_REPRODUCTION_CONTRACT["reproduction_id"]
    )
    assert registration["scope"] == (
        X11_CURRENT_REPRODUCTION_CONTRACT["scope"]
    )

    effective = load_experiment_registry(program_root)["X-11"]
    assert effective["authorization_scopes"][X11_V1_SCOPE][
        "authorized"
    ] is False
    assert effective["authorization_scopes"][registration["scope"]][
        "authorized"
    ] is True


def test_checked_in_x11_v1_scope_is_rejected_after_supersession(
    program_root: Path,
) -> None:
    effective = load_experiment_registry(program_root)["X-11"]
    v1 = effective["amendments"][0]["changes"][
        "register_reproduction"
    ]
    result = _valid_result_ref(
        scope=v1["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-24T00:59:00Z",
        registration_head_sha256=effective[
            "registration_head_sha256"
        ],
        code_sha256=v1["code_sha256"],
        data_sha256=v1["data_sha256"],
        dataset_ids=v1["dataset_ids"],
        model_ids=[
            binding["model_id"]
            for binding in v1["model_bindings"]
        ],
    )

    with (
        patch.object(
            experiments_module,
            "_utc_now",
            return_value=datetime(
                2026, 7, 24, 1, 0, 0, tzinfo=timezone.utc
            ),
        ),
        pytest.raises(
            UnauthorizedResultScopeError,
            match="result scope .* is not authorized",
        ),
    ):
        validate_result_ref(program_root, "X-11", result)


def test_checked_in_x11_v2_result_gate_opens_only_after_supersession(
    program_root: Path,
) -> None:
    effective = load_experiment_registry(program_root)["X-11"]
    amendment = effective["amendments"][1]
    registration = amendment["changes"]["supersede_reproduction"][
        "registration"
    ]
    amended_at = datetime.strptime(
        amendment["amended_at"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    result = _valid_result_ref(
        scope=registration["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at=(
            amended_at + timedelta(seconds=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        registration_head_sha256=amendment["amendment_sha256"],
        code_sha256=registration["code_sha256"],
        data_sha256=registration["data_sha256"],
        dataset_ids=registration["dataset_ids"],
        model_ids=[
            binding["model_id"]
            for binding in registration["model_bindings"]
        ],
    )
    too_early = dict(
        result,
        evaluation_started_at=amendment["amended_at"],
    )

    with pytest.raises(
        PreRegistrationEvaluationError,
        match=(
            "evaluation must follow input preregistration amendment|"
            "evaluation must follow the claimed registration head"
        ),
    ):
        validate_result_ref(program_root, "X-11", too_early)
    assert validate_result_ref(program_root, "X-11", result) == result


def test_checked_in_x11_v1_protocol_disambiguates_halftime_by_quarter(
    program_root: Path,
) -> None:
    path = (
        program_root
        / "registries/protocols/"
        "x11_fastrmodels_no_spread_v1.json"
    )
    protocol = json.loads(path.read_bytes())
    assert protocol["identity"] == {
        "experiment_id": "X-11",
        "model_id": "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
        "protocol_version": "v1",
        "reproduction_id": "REPRO-X11-NFL-FASTRMODELS-V2",
        "result_class": "poc",
        "scope": "team_h_nfl_fastrmodels_reproduction_v2",
    }
    assert protocol["feature_contract"][
        "half_seconds_remaining_semantics"
    ] == {
        "qtr_lte_2": (
            "half_seconds_remaining=game_seconds_remaining-1800"
        ),
        "qtr_gte_3": (
            "half_seconds_remaining=game_seconds_remaining"
        ),
    }
    assert protocol["census_lock"]["total"][
        "eligible_rows"
    ] == 205607
    assert protocol["census_lock"]["total"][
        "eligible_non_tie_games"
    ] == 1420
    assert protocol["ordering_and_labels"]["tie_policy"] == (
        "report_separately_and_exclude_from_binary_metrics"
    )

    protocol_sha256 = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    registration = _read_card(program_root, "X-11")[
        "amendments"
    ][1]["changes"]["supersede_reproduction"]["registration"]
    model = next(
        row
        for row in load_model_registry(program_root)
        if row.model_id
        == "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1"
    )
    assert registration["protocol_sha256"] == protocol_sha256
    assert model.pit_feature_contract == protocol_sha256
    assert model.parameter_config_sha256 == protocol_sha256


def test_team_h_can_atomically_supersede_x11_v1_with_exact_v2(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1, v2, amendment = _append_x11_v1_then_v2(root)

    effective = load_experiment_registry(root)["X-11"]
    assert effective["registration_head_sha256"] == amendment[
        "amendment_sha256"
    ]
    assert effective["authorization_scopes"][v1["scope"]][
        "authorized"
    ] is False
    v2_scope = effective["authorization_scopes"][v2["scope"]]
    assert v2_scope == {
        "authorized": True,
        "required_result_label": "PRELIMINARY",
        "required_lock_ids": [
            f"reproduction:{v2['reproduction_id']}"
        ],
        "input_binding": {
            "result_class": "poc",
            "dataset_ids": v2["dataset_ids"],
            "model_ids": [
                binding["model_id"]
                for binding in v2["model_bindings"]
            ],
            "synthetic_data_sha256": None,
        },
    }
    assert effective["preregistered_inputs"][v1["scope"]][
        "registered_at"
    ] == "2026-07-23T00:00:02Z"
    assert effective["preregistered_inputs"][v2["scope"]] == {
        "code_sha256": v2["code_sha256"],
        "data_sha256": v2["data_sha256"],
        "dataset_ids": v2["dataset_ids"],
        "model_ids": [
            binding["model_id"]
            for binding in v2["model_bindings"]
        ],
        "registered_at": "2026-07-23T00:00:03Z",
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("wrong_prior", "must supersede the current reproduction"),
        ("old_scope", "reproduction contract mismatch"),
        ("old_model", "reproduction contract mismatch"),
        ("atomicity", "must be an atomic amendment"),
    ],
)
def test_x11_supersession_rejects_wrong_transition_or_co_mutation(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1 = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": v1},
    )
    v2 = _valid_x11_v2_reproduction_registration(root)
    supersession = {
        "supersedes_reproduction_id": v1["reproduction_id"],
        "registration": v2,
    }
    changes: dict[str, Any] = {
        "supersede_reproduction": supersession
    }
    if mutation == "wrong_prior":
        supersession["supersedes_reproduction_id"] = (
            "REPRO-X11-NFL-FASTRMODELS-UNKNOWN"
        )
    elif mutation == "old_scope":
        v2["scope"] = v1["scope"]
        v2["reproduction_spec_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in v2.items()
                if key != "reproduction_spec_sha256"
            }
        )
    elif mutation == "old_model":
        v2["model_bindings"] = copy.deepcopy(
            v1["model_bindings"]
        )
        v2["reproduction_spec_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in v2.items()
                if key != "reproduction_spec_sha256"
            }
        )
    else:
        changes["status"] = "running"
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
        changes=changes,
    )

    with pytest.raises(ExperimentRegistryError, match=match):
        load_experiment_registry(root)


def test_x11_supersession_is_single_use(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1, v2, _ = _append_x11_v1_then_v2(root)
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:04Z",
        changes={
            "supersede_reproduction": {
                "supersedes_reproduction_id": v1[
                    "reproduction_id"
                ],
                "registration": v2,
            }
        },
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="must supersede the current reproduction",
    ):
        load_experiment_registry(root)


def test_x11_supersession_preserves_v1_input_history_append_only(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1, _, _ = _append_x11_v1_then_v2(root)
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:04Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": v1["scope"],
                    "code_sha256": "sha256:" + "c" * 64,
                    "data_sha256": v1["data_sha256"],
                    "dataset_ids": v1["dataset_ids"],
                    "model_ids": [
                        binding["model_id"]
                        for binding in v1["model_bindings"]
                    ],
                }
            ]
        },
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction inputs are append-once",
    ):
        load_experiment_registry(root)


def test_superseded_x11_scope_cannot_be_generically_reauthorized(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1, _, _ = _append_x11_v1_then_v2(root)
    attack = _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:04Z",
        changes={"authorize_scopes": [v1["scope"]]},
    )
    old_result = _valid_result_ref(
        scope=v1["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:05Z",
        registration_head_sha256=attack["amendment_sha256"],
        code_sha256=v1["code_sha256"],
        data_sha256=v1["data_sha256"],
        dataset_ids=v1["dataset_ids"],
        model_ids=[
            binding["model_id"]
            for binding in v1["model_bindings"]
        ],
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction scopes cannot be generically authorized",
    ):
        load_experiment_registry(root)
    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction scopes cannot be generically authorized",
    ):
        validate_result_ref(root, "X-11", old_result)


def test_generic_authorization_still_allows_non_reproduction_scope(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:02Z",
        changes={"authorize_scopes": ["kalshi_capture"]},
    )

    effective = load_experiment_registry(root)["X-08"]
    assert effective["authorization_scopes"]["kalshi_capture"][
        "authorized"
    ] is True


def test_x11_supersession_is_rejected_after_any_v1_result(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1 = _valid_reproduction_registration(root, "X-11")
    registration_amendment = _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": v1},
    )
    result = _valid_result_ref(
        scope=v1["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:03Z",
        registration_head_sha256=registration_amendment[
            "amendment_sha256"
        ],
        code_sha256=v1["code_sha256"],
        data_sha256=v1["data_sha256"],
        dataset_ids=v1["dataset_ids"],
        model_ids=[
            binding["model_id"]
            for binding in v1["model_bindings"]
        ],
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:04Z",
        changes={"results_ref": result},
    )
    v2 = _valid_x11_v2_reproduction_registration(root)
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:05Z",
        changes={
            "supersede_reproduction": {
                "supersedes_reproduction_id": v1[
                    "reproduction_id"
                ],
                "registration": v2,
            }
        },
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="cannot be superseded after evaluation",
    ):
        load_experiment_registry(root)


def test_x11_supersession_rejects_future_amendment(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_x11_v1_then_v2(
        root,
        supersession_at="2026-07-23T00:00:04Z",
    )

    with (
        patch.object(
            experiments_module,
            "_utc_now",
            return_value=datetime(
                2026, 7, 23, 0, 0, 3, tzinfo=timezone.utc
            ),
        ),
        pytest.raises(
            ExperimentRegistryError,
            match="reproduction amendment cannot be future-dated",
        ),
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("spec", "reproduction spec hash mismatch"),
        ("model", "record hash mismatch"),
        ("data", "reproduction data hash mismatch"),
        ("code", "reproduction code hash mismatch"),
        ("protocol", "reproduction protocol hash mismatch"),
        ("protocol_content", "reproduction protocol contract mismatch"),
    ],
)
def test_x11_supersession_recomputes_every_v2_binding(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    v1 = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": v1},
    )
    v2 = _valid_x11_v2_reproduction_registration(root)
    if mutation == "spec":
        v2["reproduction_spec_sha256"] = "sha256:" + "c" * 64
    elif mutation == "model":
        v2["model_bindings"][0][
            "model_record_sha256"
        ] = "sha256:" + "c" * 64
    elif mutation in {"data", "code", "protocol"}:
        v2[
            {
                "data": "data_sha256",
                "code": "code_sha256",
                "protocol": "protocol_sha256",
            }[mutation]
        ] = "sha256:" + "c" * 64
    else:
        protocol_path = root / v2["protocol_path"]
        protocol = json.loads(protocol_path.read_bytes())
        protocol["bootstrap"]["resamples"] = 201
        v2["protocol_sha256"] = _write_pretty_json(
            protocol_path, protocol
        )
    if mutation != "spec":
        v2["reproduction_spec_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in v2.items()
                if key != "reproduction_spec_sha256"
            }
        )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
        changes={
            "supersede_reproduction": {
                "supersedes_reproduction_id": v1[
                    "reproduction_id"
                ],
                "registration": v2,
            }
        },
    )

    with pytest.raises(ExperimentRegistryError, match=match):
        load_experiment_registry(root)


def test_x11_reproduction_contract_rejects_spread_model_id(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    registration["model_bindings"] = [
        {
            "model_id": "MODEL-NFL-FASTRMODELS-SPREAD",
            "model_version": "v0",
            "model_record_sha256": "sha256:" + "a" * 64,
        }
    ]
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in registration.items()
            if key != "reproduction_spec_sha256"
        }
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction contract mismatch",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("evaluation_filter", "regulation_quarters"), [1, 2, 3, 4, 5]),
        (("bootstrap", "resamples"), 199),
        (("asset", "github_release_asset_id"), 253928624),
    ],
)
def test_x11_protocol_semantics_cannot_be_rebound_after_mutation(
    tmp_path: Path,
    path: tuple[str, str],
    replacement: object,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    protocol_path = root / registration["protocol_path"]
    protocol = json.loads(protocol_path.read_bytes())
    protocol[path[0]][path[1]] = replacement
    rendered = (
        json.dumps(
            protocol,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    protocol_path.write_bytes(rendered)
    registration["protocol_sha256"] = (
        "sha256:" + hashlib.sha256(rendered).hexdigest()
    )
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in registration.items()
            if key != "reproduction_spec_sha256"
        }
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction protocol contract mismatch",
    ):
        load_experiment_registry(root)


def test_x11_asset_manifest_change_breaks_reproduction_data_binding(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _update_dataset_registry(
        root,
        "DS-NFL-FASTRMODELS",
        manifest_sha256="sha256:" + "a" * 64,
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction data hash mismatch",
    ):
        load_experiment_registry(root)


def test_register_reproduction_is_limited_to_x11_and_x12(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-06",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="register_reproduction.*only valid for X-11 or X-12",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "duplicate",
    ["existing_scope", "model_id"],
)
def test_register_reproduction_rejects_duplicate_scope_or_model_ids(
    tmp_path: Path,
    duplicate: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    if duplicate == "existing_scope":
        registration["scope"] = "preregistered_pipeline"
    else:
        registration["model_bindings"].append(
            copy.deepcopy(registration["model_bindings"][0])
        )
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in registration.items()
            if key != "reproduction_spec_sha256"
        }
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="contract mismatch|duplicate|already exists",
    ):
        load_experiment_registry(root)


def test_register_reproduction_is_append_once(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    first = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": first},
    )
    second = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
        changes={"register_reproduction": second},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="register_reproduction is append-once",
    ):
        load_experiment_registry(root)


def test_registered_reproduction_inputs_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": registration["scope"],
                    "code_sha256": "sha256:" + "c" * 64,
                    "data_sha256": registration["data_sha256"],
                    "dataset_ids": registration["dataset_ids"],
                    "model_ids": [
                        item["model_id"]
                        for item in registration["model_bindings"]
                    ],
                }
            ]
        },
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction inputs are append-once",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "mutation",
    [
        "model_version",
        "model_record_sha256",
        "reproduction_spec_sha256",
    ],
)
def test_register_reproduction_rejects_hash_or_version_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    if mutation == "model_version":
        binding = registration["model_bindings"][0]
        binding["model_version"] = (
            "v1" if binding["model_version"] == "v0" else "v0"
        )
    elif mutation == "model_record_sha256":
        registration["model_bindings"][0][
            "model_record_sha256"
        ] = "sha256:" + "c" * 64
    else:
        registration["reproduction_spec_sha256"] = (
            "sha256:" + "c" * 64
        )
    if mutation != "reproduction_spec_sha256":
        registration["reproduction_spec_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in registration.items()
                if key != "reproduction_spec_sha256"
            }
        )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="version mismatch|record hash mismatch|spec hash mismatch",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    ("experiment_id", "model_ids"),
    [
        ("X-11", ["MODEL-NFL-LOGISTIC"]),
        ("X-12", ["MODEL-SOCCER-DIXON-COLES"]),
    ],
)
def test_register_reproduction_rejects_noncanonical_model_set(
    tmp_path: Path,
    experiment_id: str,
    model_ids: list[str],
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(
        root,
        experiment_id,
        model_ids=model_ids,
    )
    _append_amendment(
        root,
        experiment_id,
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction contract mismatch",
    ):
        load_experiment_registry(root)


def test_register_reproduction_rejects_unknown_reproduction_id(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(
        root,
        "X-11",
        reproduction_id="REPRO-X11-LEGACY-V1",
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction contract mismatch",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize("binding", ["code", "data"])
def test_register_reproduction_recomputes_code_and_data_bindings(
    tmp_path: Path,
    binding: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    if binding == "code":
        registration["code_sha256"] = "sha256:" + "c" * 64
    else:
        registration["data_sha256"] = "sha256:" + "c" * 64
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in registration.items()
            if key != "reproduction_spec_sha256"
        }
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match=f"reproduction {binding} hash mismatch",
    ):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    ("experiment_id", "code_path"),
    [
        ("X-11", "src/prediction_market/models/nfl.py"),
        (
            "X-11",
            "src/prediction_market/models/nfl_fastrmodels.py",
        ),
        (
            "X-12",
            "src/prediction_market/sports/soccer_transition_model.py",
        ),
        ("X-12", "src/prediction_market/sports/x12.py"),
    ],
)
def test_registered_reproduction_fails_if_any_code_object_changes(
    tmp_path: Path,
    experiment_id: str,
    code_path: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, experiment_id)
    mutated_path = root / code_path
    mutated_path.write_bytes(mutated_path.read_bytes() + b"\n# mutation\n")
    _append_amendment(
        root,
        experiment_id,
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="reproduction code hash mismatch",
    ):
        load_experiment_registry(root)


def test_register_reproduction_rejects_legacy_singular_code_path(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    registration["code_path"] = registration.pop("code_paths")[0]
    registration["reproduction_spec_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in registration.items()
            if key != "reproduction_spec_sha256"
        }
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="unexpected or missing keys",
    ):
        load_experiment_registry(root)


def test_register_reproduction_rejects_future_amendment(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:05Z",
        changes={"register_reproduction": registration},
    )

    with (
        patch.object(
            experiments_module,
            "_utc_now",
            return_value=datetime(2026, 7, 23, 0, 0, 4, tzinfo=timezone.utc),
        ),
        pytest.raises(
            ExperimentRegistryError,
            match="reproduction amendment cannot be future-dated",
        ),
    ):
        load_experiment_registry(root)


def test_reproduction_rejects_future_evaluation(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )
    amendment = _resolve_reproduction_base_locks(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
    )
    result = _valid_result_ref(
        scope=registration["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:05Z",
        registration_head_sha256=amendment["amendment_sha256"],
        code_sha256=registration["code_sha256"],
        data_sha256=registration["data_sha256"],
        dataset_ids=registration["dataset_ids"],
        model_ids=[
            item["model_id"]
            for item in registration["model_bindings"]
        ],
    )

    with (
        patch.object(
            experiments_module,
            "_utc_now",
            return_value=datetime(2026, 7, 23, 0, 0, 4, tzinfo=timezone.utc),
        ),
        pytest.raises(
            PreRegistrationEvaluationError,
            match="reproduction evaluation cannot be future-dated",
        ),
    ):
        validate_result_ref(root, "X-11", result)


def test_reproduction_rejects_future_result_amendment(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )
    registration_amendment = _resolve_reproduction_base_locks(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
    )
    result = _valid_result_ref(
        scope=registration["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:04Z",
        registration_head_sha256=registration_amendment[
            "amendment_sha256"
        ],
        code_sha256=registration["code_sha256"],
        data_sha256=registration["data_sha256"],
        dataset_ids=registration["dataset_ids"],
        model_ids=[
            item["model_id"]
            for item in registration["model_bindings"]
        ],
    )
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:06Z",
        changes={"results_ref": result},
    )

    with (
        patch.object(
            experiments_module,
            "_utc_now",
            return_value=datetime(2026, 7, 23, 0, 0, 5, tzinfo=timezone.utc),
        ),
        pytest.raises(
            ExperimentRegistryError,
            match="reproduction result amendment cannot be future-dated",
        ),
    ):
        load_experiment_registry(root)


def test_reproduction_evaluation_must_follow_registration_amendment(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    registration = _valid_reproduction_registration(root, "X-11")
    _append_amendment(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:02Z",
        changes={"register_reproduction": registration},
    )
    amendment = _resolve_reproduction_base_locks(
        root,
        "X-11",
        amended_at="2026-07-23T00:00:03Z",
    )
    result = _valid_result_ref(
        scope=registration["scope"],
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:01Z",
        registration_head_sha256=amendment["amendment_sha256"],
        code_sha256=registration["code_sha256"],
        data_sha256=registration["data_sha256"],
        dataset_ids=registration["dataset_ids"],
        model_ids=[
            item["model_id"]
            for item in registration["model_bindings"]
        ],
    )

    with pytest.raises(
        PreRegistrationEvaluationError,
        match="evaluation must follow input preregistration amendment",
    ):
        validate_result_ref(root, "X-11", result)


def test_x12_is_statsbomb_poc_without_market_prior_or_formal_promotion(
    program_root: Path,
) -> None:
    x12 = load_experiment_registry(program_root)["X-12"]
    combined = " ".join(
        [x12["method"], x12["split"], *x12["metrics"]]
    ).lower()

    assert "statsbomb" in x12["data"][0]["source"].lower()
    assert "2015/16" in combined
    assert "expanding-window" in combined
    assert "dixon-coles" in combined
    assert "1x2" in combined and "multiclass" in combined
    assert "no point-in-time market prior" in combined
    assert "one-vs-rest calibration slope" in combined
    assert "one-vs-rest calibration intercept" in combined
    assert "game-cluster bootstrap confidence interval" in combined
    assert "paired" in combined and "point-in-time prior" in combined
    assert x12["execution_authorized"] is True
    assert x12["completion_required_scopes"] == ["poc_result"]
    assert x12["authorization_scopes"]["poc_result"]["authorized"] is True
    assert x12["authorization_scopes"]["formal_promotion"]["authorized"] is False
    assert x12["authorization_scopes"]["formal_promotion"]["permanent_no_go"] is True
    assert x12["promotion_restriction"] == "POC_ONLY_FORMAL_PROMOTION_UNAUTHORIZED"
    assert x12["dataset_ids"] == ["DS-STATSBOMB-OPEN"]
    assert x12["output_contract"]["transition_unit"] == "five_minute_interval"


def test_phase_execution_authorization_is_exact_and_fail_closed(
    program_root: Path,
) -> None:
    registry = load_experiment_registry(program_root)
    authorized = {
        experiment_id
        for experiment_id, card in registry.items()
        if card["execution_authorized"]
    }

    assert authorized == {
        "X-01",
        "X-02",
        "X-03",
        "X-06",
        "X-08",
        "X-11",
        "X-12",
    }
    assert registry["X-04"]["authorization_scopes"]["formal_result"][
        "authorized"
    ] is False
    assert registry["X-09"]["authorization_scopes"]["formal_result"][
        "authorized"
    ] is False


def test_research_scopes_declare_exact_input_bindings(
    program_root: Path,
) -> None:
    registry = load_experiment_registry(program_root)
    expected = {
        ("X-01", "formal_result"): (
            ["DS-PMXT-V2", "DS-POLYMARKET-PUBLIC"],
            [],
            "formal",
        ),
        ("X-02", "formal_result"): (["DS-PMXT-V2"], [], "formal"),
        ("X-03", "formal_result"): (
            [
                "DS-KALSHI-HISTORICAL",
                "DS-PMXT-V2",
                "DS-POLYMARKET-PUBLIC",
            ],
            [],
            "formal",
        ),
        ("X-06", "formal_result"): (
            ["DS-NBA-CANDIDATE"],
            [
                "MODEL-NBA-GBDT",
                "MODEL-NBA-LOGISTIC",
                "MODEL-NBA-POSSESSION-TRANSITION",
                "MODEL-NBA-PRIOR",
            ],
            "formal",
        ),
        ("X-06", "contract_harness"): (
            [],
            ["MODEL-NBA-POSSESSION-TRANSITION"],
            "synthetic",
        ),
        ("X-08", "archive_audit"): (
            ["DS-KALSHI-HISTORICAL"],
            [],
            "formal",
        ),
        ("X-08", "polymarket_capture"): (
            ["DS-POLYMARKET-PUBLIC"],
            [],
            "poc",
        ),
        ("X-08", "kalshi_capture"): (
            ["DS-KALSHI-LIVE-L2"],
            [],
            "poc",
        ),
        ("X-08", "dual_venue_result"): (
            [
                "DS-KALSHI-HISTORICAL",
                "DS-KALSHI-LIVE-L2",
                "DS-POLYMARKET-PUBLIC",
            ],
            [],
            "formal",
        ),
        ("X-11", "formal_result"): (
            ["DS-NFLVERSE"],
            [
                "MODEL-NFL-DRIVE-TRANSITION",
                "MODEL-NFL-GBDT",
                "MODEL-NFL-LOGISTIC",
                "MODEL-NFL-NFLFASTR-COMPARATOR",
                "MODEL-NFL-SPREAD-PRIOR",
            ],
            "formal",
        ),
        ("X-12", "poc_result"): (
            ["DS-STATSBOMB-OPEN"],
            [
                "MODEL-SOCCER-DIXON-COLES",
                "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
            ],
            "poc",
        ),
    }
    for (experiment_id, scope_name), (
        dataset_ids,
        model_ids,
        result_class,
    ) in expected.items():
        binding = registry[experiment_id]["authorization_scopes"][scope_name][
            "input_binding"
        ]
        assert binding["dataset_ids"] == dataset_ids
        assert binding["model_ids"] == model_ids
        assert binding["result_class"] == result_class


def test_execution_authorized_false_is_a_hard_result_gate(
    program_root: Path,
) -> None:
    x10 = load_experiment_registry(program_root)["X-10"]
    result = _valid_result_ref(
        scope="precision_audit",
        registration_head_sha256=x10["registration_head_sha256"],
        dataset_ids=[],
        model_ids=[],
    )

    with pytest.raises(UnauthorizedResultScopeError, match="execution"):
        validate_result_ref(program_root, "X-10", result)


def test_x08_is_prospective_and_preserves_the_decision_band(program_root: Path) -> None:
    base = _read_card(program_root, "X-08")
    assert base["data"][0]["source"] == "Kalshi pre-stop archive sample"
    assert "stopped archive" in base["method"].lower()
    assert (
        base["registration_record_sha256"]
        == "sha256:e4d9f0a72ac6dcb4ad0e78859bf0264eb9a79508987263428678bf4342b97e8e"
    )
    clarification = base["amendments"][0]
    x08 = load_experiment_registry(program_root)["X-08"]
    hypothesis = x08["hypothesis"].lower()
    historical = " ".join(
        str(value)
        for value in clarification["changes"][
            "archive_audit_clarification"
        ].values()
    ).lower()

    assert "prospective" in hypothesis
    assert "already" not in hypothesis
    assert clarification["sequence"] == 1
    assert (
        clarification["prior_sha256"]
        == base["registration_record_sha256"]
    )
    assert (
        clarification["amendment_sha256"]
        == compute_amendment_sha256(clarification)
    )
    with (
        program_root / "registries" / "experiment_amendment_ledger.csv"
    ).open(encoding="utf-8", newline="") as handle:
        x08_ledger = [
            row
            for row in csv.DictReader(handle)
            if row["experiment_id"] == "X-08"
        ]
    assert [row["sequence"] for row in x08_ledger] == ["0", "1"]
    assert x08_ledger[0]["record_sha256"] == base[
        "registration_record_sha256"
    ]
    assert x08_ledger[1]["prior_sha256"] == base[
        "registration_record_sha256"
    ]
    assert (
        x08_ledger[1]["record_sha256"]
        == clarification["amendment_sha256"]
        == x08["registration_head_sha256"]
    )
    assert "official historical rest" in historical
    assert "candle" in historical and "trade" in historical
    assert "cutoff" in historical
    assert "no historical l2" in historical
    assert "reference audit only" in historical
    assert (
        clarification["changes"]["archive_audit_clarification"][
            "live_l2_dataset_id"
        ]
        == "DS-KALSHI-LIVE-L2"
    )
    assert x08["prospective_observation"] == {
        "required_elapsed_days": 7,
        "observed_elapsed_days": 0,
        "fixtures_can_satisfy_elapsed_time": False,
    }
    assert x08["prospective_observation"]["fixtures_can_satisfy_elapsed_time"] is False
    assert "no gaps" in x08["pass_criteria"].lower()
    assert "<99%" in x08["fail_criteria"].replace(" ", "")
    assert "99% <= uptime < 100%" in x08["unresolved_decision_band"]


def test_x08_completion_rejects_result_without_seven_elapsed_days(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    preregistration = _preregister_x08_dual_venue_inputs(root)
    result = _valid_result_ref(
        scope="dual_venue_result",
        evaluation_started_at="2026-07-24T00:00:00Z",
        registration_head_sha256=preregistration["amendment_sha256"],
        dataset_ids=[
            "DS-KALSHI-HISTORICAL",
            "DS-KALSHI-LIVE-L2",
            "DS-POLYMARKET-PUBLIC",
        ],
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:01Z",
        changes={"results_ref": result, "status": "done"},
    )

    with pytest.raises(ExperimentRegistryError, match="elapsed|prospective"):
        load_experiment_registry(root)


def test_x08_elapsed_evidence_is_immutable_derived_time_and_unlocks_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-30T00:00:06Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-seven-day",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-30T00:00:02Z",
    )
    observation = _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-30T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )
    effective = load_experiment_registry(root)["X-08"]
    assert effective["prospective_observation"]["required_elapsed_days"] == 7
    assert effective["prospective_observation"]["observed_elapsed_days"] == 7

    result = _valid_result_ref(
        scope="dual_venue_result",
        evaluation_started_at="2026-07-30T00:00:04Z",
        registration_head_sha256=observation["amendment_sha256"],
        dataset_ids=[
            "DS-KALSHI-HISTORICAL",
            "DS-KALSHI-LIVE-L2",
            "DS-POLYMARKET-PUBLIC",
        ],
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-30T00:00:05Z",
        changes={"results_ref": result, "status": "done"},
    )

    assert load_experiment_registry(root)["X-08"]["status"] == "done"


def test_x08_elapsed_evidence_rejects_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-30T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-fixture",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-30T00:00:02Z",
        fixtures_used=True,
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-30T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="fixture"):
        load_experiment_registry(root)


def test_x08_disjoint_observation_windows_cannot_accumulate_to_seven_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-08-01T00:00:05Z")
    _preregister_x08_dual_venue_inputs(root)
    first_evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-first-window",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-27T00:00:02Z",
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-27T00:00:03Z",
        changes={"observed_elapsed_evidence": first_evidence},
    )
    second_evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-second-window",
        started_at="2026-07-28T00:00:02Z",
        ended_at="2026-08-01T00:00:02Z",
    )
    second = _append_amendment(
        root,
        "X-08",
        amended_at="2026-08-01T00:00:03Z",
        changes={"observed_elapsed_evidence": second_evidence},
    )
    result = _valid_result_ref(
        scope="dual_venue_result",
        evaluation_started_at="2026-08-01T00:00:04Z",
        registration_head_sha256=second["amendment_sha256"],
        dataset_ids=[
            "DS-KALSHI-HISTORICAL",
            "DS-KALSHI-LIVE-L2",
            "DS-POLYMARKET-PUBLIC",
        ],
    )

    with pytest.raises(InvalidResultReferenceError, match="elapsed|prospective"):
        validate_result_ref(root, "X-08", result)


def test_x08_capture_manifest_must_be_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-unregistered",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    artifact_registry = root / "registries" / "artifact_registry.csv"
    with artifact_registry.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [
            row
            for row in reader
            if row["path"] != evidence["capture_manifest_path"]
        ]
    assert fieldnames is not None
    with artifact_registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="registered artifact"):
        load_experiment_registry(root)


def test_x08_capture_manifest_hash_is_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-manifest-hash",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    evidence["capture_manifest_sha256"] = "sha256:" + "a" * 64
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="manifest SHA-256"):
        load_experiment_registry(root)


def test_x08_capture_manifest_must_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-missing-manifest",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    (root / evidence["capture_manifest_path"]).unlink()
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="missing X-08 capture manifest"):
        load_experiment_registry(root)


def test_x08_capture_manifest_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-symlink-manifest",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    manifest_path = root / evidence["capture_manifest_path"]
    target = manifest_path.with_name("capture-manifest-target.json")
    manifest_path.rename(target)
    manifest_path.symlink_to(target)
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="symlink"):
        load_experiment_registry(root)


def test_x08_capture_manifest_path_cannot_escape_program_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-path-escape",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    artifact_registry = root / "registries" / "artifact_registry.csv"
    with artifact_registry.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    row = next(
        item
        for item in rows
        if item["path"] == evidence["capture_manifest_path"]
    )
    escaped_path = "../escaped-x08-capture-manifest.json"
    row["path"] = escaped_path
    escaped_manifest = root.parent / "escaped-x08-capture-manifest.json"
    escaped_manifest.write_bytes(
        (root / evidence["capture_manifest_path"]).read_bytes()
    )
    evidence["capture_manifest_path"] = escaped_path
    with artifact_registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="path escape"):
        load_experiment_registry(root)


def test_x08_capture_manifest_cannot_be_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:06Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-reused",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:05Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="duplicate"):
        load_experiment_registry(root)


def test_x08_capture_segment_object_hash_is_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-24T00:00:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-object-hash",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-24T00:00:02Z",
    )
    capture_manifest = json.loads(
        (root / evidence["capture_manifest_path"]).read_text(encoding="utf-8")
    )
    raw_store_root = root / capture_manifest["raw_store_root"]
    first_segment = (
        raw_store_root
        / capture_manifest["streams"][0]["segment_manifest_paths"][0]
    )
    sidecar = json.loads(first_segment.read_text(encoding="utf-8"))
    object_path = raw_store_root / sidecar["object_path"]
    object_path.chmod(0o600)
    object_path.write_bytes(object_path.read_bytes() + b"tampered")
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-24T00:00:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="segment verification|SHA-256",
    ):
        load_experiment_registry(root)


def test_x08_capture_rejects_heartbeat_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-23T00:10:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-gap",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-23T00:10:02Z",
        drop_heartbeat_at="2026-07-23T00:05:02Z",
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:10:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="gap"):
        load_experiment_registry(root)


def test_x08_capture_requires_both_live_venue_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-23T00:10:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-one-leg",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-23T00:10:02Z",
        include_dataset_ids=("DS-KALSHI-LIVE-L2",),
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:10:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="both registered live streams"):
        load_experiment_registry(root)


def test_x08_capture_segment_must_be_sealed_after_last_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-23T00:10:04Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-early-seal",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-23T00:10:02Z",
        sealed_at="2026-07-23T00:00:02Z",
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:10:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="sealed after"):
        load_experiment_registry(root)


def test_x08_capture_evidence_amendment_cannot_be_future_dated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-23T00:10:02Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-future",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-23T00:10:02Z",
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:10:03Z",
        changes={"observed_elapsed_evidence": evidence},
    )

    with pytest.raises(ExperimentRegistryError, match="future-dated"):
        load_experiment_registry(root)


def test_x08_fifteen_second_smoke_capture_cannot_unlock_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _set_x08_validation_clock(monkeypatch, "2026-07-23T00:00:21Z")
    _preregister_x08_dual_venue_inputs(root)
    evidence = _write_x08_capture_evidence(
        root,
        capture_session_id="x08-smoke",
        started_at="2026-07-23T00:00:02Z",
        ended_at="2026-07-23T00:00:17Z",
    )
    observation = _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:18Z",
        changes={"observed_elapsed_evidence": evidence},
    )
    result = _valid_result_ref(
        scope="dual_venue_result",
        evaluation_started_at="2026-07-23T00:00:19Z",
        registration_head_sha256=observation["amendment_sha256"],
        dataset_ids=[
            "DS-KALSHI-HISTORICAL",
            "DS-KALSHI-LIVE-L2",
            "DS-POLYMARKET-PUBLIC",
        ],
    )

    with pytest.raises(InvalidResultReferenceError, match="elapsed|prospective"):
        validate_result_ref(root, "X-08", result)


def test_charter_specific_gates_are_preserved(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)

    assert registry["X-01"]["deterministic_replay_required_levels"] == [1, 2]
    assert registry["X-09"]["deterministic_replay_required_levels"] == [1, 2]
    assert set(registry["X-06"]["decision_gates"]) == {
        "gate_1_reaction_model",
        "gate_2_trading_relevance",
    }
    assert any("bootstrap" in metric.lower() for metric in registry["X-06"]["metrics"])
    assert "45 of 50" in registry["X-10"]["pass_criteria"]
    assert registry["X-10"]["recall_denominator_registered"] is False
    assert registry["X-05"]["midpoint_allowed"] is False
    assert registry["X-07"]["midpoint_allowed"] is False


def test_every_card_preserves_all_program_no_gos(program_root: Path) -> None:
    registry = load_experiment_registry(program_root)

    for card in registry.values():
        assert set(card["program_no_go_restrictions"]) == EXPECTED_NO_GOS


def test_artifact_registry_covers_task_3_without_completion_claims(
    program_root: Path,
) -> None:
    with (program_root / "registries" / "artifact_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert EXPECTED_TASK_3_PATHS <= {row["path"] for row in rows}
    assert all(row["owner_team"] for row in rows)
    assert all(row["version"] for row in rows)
    assert all(row["due_gate"] for row in rows)
    task_3_rows = [row for row in rows if row["path"] in EXPECTED_TASK_3_PATHS]
    assert {row["status"] for row in task_3_rows} == {"registered"}


def test_x02_input_artifacts_are_registered_without_formal_result_claims(
    program_root: Path,
) -> None:
    with (program_root / "registries" / "artifact_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    expected_paths = {
        item["path"]
        for item in EXPECTED_X02_DAY_MANIFESTS
        if item["day"] != "2026-05-28"
    } | {EXPECTED_X02_INPUT_MANIFEST_BINDING["bundle_path"]}
    selected = {
        row["path"]: row
        for row in rows
        if row["path"] in expected_paths
    }

    assert set(selected) == expected_paths
    assert {
        (
            row["owner_team"],
            row["version"],
            row["due_gate"],
            row["status"],
        )
        for row in selected.values()
    } == {
        ("C+H", "v1", "2026-08-05_W2_review", "registered")
    }
    assert all("result" not in row["status"].lower() for row in selected.values())


def test_polymarket_v1_poc_artifact_is_preliminary_and_license_bound(
    program_root: Path,
) -> None:
    artifact_path = (
        "artifacts/data-audit/"
        "polymarket_v1_bounded_sports_extract_v0.json"
    )
    with (program_root / "registries" / "artifact_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        artifact = next(
            row for row in csv.DictReader(handle)
            if row["path"] == artifact_path
        )
    assert artifact == {
        "artifact_id": "ART-C-015",
        "path": artifact_path,
        "owner_team": "C",
        "version": "v0",
        "due_gate": "2026-07-29_W1",
        "status": "PRELIMINARY_RESEARCH_ONLY",
    }
    raw = (program_root / artifact_path).read_bytes()
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == (
        "sha256:"
        "8bae8f1509ae852afcb7a0c8e1bf6d1a892aa96c34ba9d6e15f96b059bf8c759"
    )
    document = json.loads(raw)
    assert document["status"] == "PRELIMINARY_RESEARCH_ONLY"
    assert "formal X-01 comparison gate" in document["evidence_boundary"][
        "not_supported"
    ]

    with (program_root / "registries" / "dataset_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        datasets = {
            row["dataset_id"]: row for row in csv.DictReader(handle)
        }
    assert datasets["DS-POLYMARKET-V1"]["license_status"] == "approved"
    assert datasets["DS-POLYMARKET-V1"]["license_review_id"] == "R-039"
    assert datasets["DS-POLYMARKET-PUBLIC"]["license_status"] == "pending"
    assert datasets["DS-POLYMARKET-PUBLIC"]["license_review_id"] == "O-001"


def test_result_is_rejected_without_preexisting_registration(tmp_path: Path) -> None:
    with pytest.raises(UnregisteredExperimentError):
        validate_result_ref(tmp_path, "X-99", "sha256:" + "0" * 64)


def test_registered_result_requires_code_data_and_result_hashes(
    program_root: Path,
) -> None:
    with pytest.raises(InvalidResultReferenceError):
        validate_result_ref(
            program_root,
            "X-08",
            "sha256:" + "0" * 64,
        )

    malformed = _valid_result_ref()
    malformed["code_sha256"] = "sha256:not-a-hash"
    with pytest.raises(InvalidResultReferenceError):
        validate_result_ref(program_root, "X-08", malformed)


def test_result_scope_must_be_known_and_authorized(program_root: Path) -> None:
    unknown_scope = _valid_result_ref(scope="not_registered")
    with pytest.raises(UnauthorizedResultScopeError):
        validate_result_ref(program_root, "X-08", unknown_scope)

    blocked_scope = _valid_result_ref(scope="kalshi_capture")
    with pytest.raises(UnauthorizedResultScopeError):
        validate_result_ref(program_root, "X-08", blocked_scope)


def test_evaluation_must_start_after_registration(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _preregister_x08_inputs(root)
    too_early = _valid_result_ref(
        evaluation_started_at="2026-07-22T23:59:59Z",
        registration_head_sha256=amendment["amendment_sha256"],
    )
    with pytest.raises(PreRegistrationEvaluationError):
        validate_result_ref(root, "X-08", too_early)

    validated = validate_result_ref(
        root,
        "X-08",
        _valid_result_ref(
            registration_head_sha256=amendment["amendment_sha256"]
        ),
    )
    assert validated["scope"] == "archive_audit"


def test_result_rejects_unresolved_registration_locks(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": "formal_result",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": ["DS-PMXT-V2", "DS-POLYMARKET-PUBLIC"],
                    "model_ids": [],
                }
            ]
        },
    )
    result_ref = _valid_result_ref(
        scope="formal_result",
        registration_head_sha256=amendment["amendment_sha256"],
        dataset_ids=["DS-PMXT-V2", "DS-POLYMARKET-PUBLIC"],
    )
    with pytest.raises(UnresolvedRegistrationLockError):
        validate_result_ref(root, "X-01", result_ref)


def test_result_rejects_unfinished_dependencies(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    _update_dataset_registry(
        root,
        "DS-PMXT-V2",
        license_status="approved",
        manifest_sha256="sha256:" + "7" * 64,
    )
    amendment = _append_amendment(
        root,
        "X-02",
        amended_at="2026-07-24T00:00:01Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": "formal_result",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": ["DS-PMXT-V2"],
                    "model_ids": [],
                }
            ],
        },
    )
    result_ref = _valid_result_ref(
        scope="formal_result",
        evaluation_started_at="2026-07-24T00:00:02Z",
        registration_head_sha256=amendment["amendment_sha256"],
        dataset_ids=["DS-PMXT-V2"],
    )
    with pytest.raises(UnresolvedDependencyError):
        validate_result_ref(root, "X-02", result_ref)


def test_registry_rejects_card_hash_tampering(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / "experiments" / "X-01.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ExperimentRegistryError, match="SHA-256"):
        load_experiment_registry(root)


def test_registry_rejects_overwritten_base_registration(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-01")
    card["hypothesis"] = "overwritten after registration"
    _write_card(root, "X-01", card)
    _update_registry_card_hash(root, "X-01")

    with pytest.raises(ExperimentRegistryError, match="immutable registration"):
        load_experiment_registry(root)


def test_amendment_chain_requires_sequence_prior_hash_and_content_hash(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-01")
    card["amendments"] = [
        {
            "sequence": 2,
            "amended_at": "2026-07-23T00:00:00Z",
            "prior_sha256": "sha256:" + "0" * 64,
            "amendment_sha256": "sha256:" + "1" * 64,
            "approved_by": "H",
            "reason": "fixture",
            "changes": {"status": "running"},
        }
    ]
    _write_card(root, "X-01", card)
    _update_registry_card_hash(root, "X-01")

    with pytest.raises(ExperimentRegistryError, match="amendment sequence"):
        load_experiment_registry(root)


def test_dependency_graph_must_be_acyclic(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-01")
    card["dependencies"] = ["X-02"]
    _rewrite_registered_card(root, "X-01", card)

    with pytest.raises(ExperimentRegistryError, match="dependency cycle"):
        load_experiment_registry(root)


def test_loader_rejects_blank_pit_lineage_and_bad_scope_locks(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-03")
    card["data"][0]["pit_basis"] = ""
    _rewrite_registered_card(root, "X-03", card)

    with pytest.raises(ExperimentRegistryError, match="pit_basis"):
        load_experiment_registry(root)

    root = _copy_program_fixture(tmp_path / "scope")
    card = _read_card(root, "X-10")
    card["authorization_scopes"]["precision_audit"]["required_lock_ids"] = [
        "unknown_lock"
    ]
    _rewrite_registered_card(root, "X-10", card)

    with pytest.raises(ExperimentRegistryError, match="unknown registration lock"):
        load_experiment_registry(root)


def test_registry_rejects_base_rewrite_even_when_all_public_hashes_are_recomputed(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-01")
    card["hypothesis"] = "coherently rewritten base registration"
    _rewrite_registered_card(root, "X-01", card)

    with pytest.raises(ExperimentRegistryError, match="trusted base registration"):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "value",
    [1.5, {1: "integer-key"}, {True: "boolean-key"}, {None: "null-key"}],
    ids=["float", "integer-key", "boolean-key", "null-key"],
)
def test_registration_hash_rejects_noncanonical_values(value: Any) -> None:
    with pytest.raises(ExperimentRegistryError, match="canonical"):
        compute_registration_record_sha256({"value": value})


@pytest.mark.parametrize(
    "target,extra_key",
    [
        ("card", "unexpected"),
        ("data", "unexpected"),
        ("cost_estimate", "unexpected"),
        ("scope", "unexpected"),
        ("lock", "unexpected"),
        ("lineage", "unexpected"),
        ("gate", "unexpected"),
    ],
)
def test_registry_rejects_extra_keys_at_every_card_structure(
    tmp_path: Path, target: str, extra_key: str
) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-01")
    targets = {
        "card": card,
        "data": card["data"][0],
        "cost_estimate": card["cost_estimate"],
        "scope": card["authorization_scopes"]["formal_result"],
        "lock": card["registration_locks"][0],
        "lineage": card["source_lineage"],
        "gate": card["linked_first_artifact_due_gates"][0],
    }
    targets[target][extra_key] = "not registered"
    _rewrite_registered_card(root, "X-01", card)

    with pytest.raises(ExperimentRegistryError):
        load_experiment_registry(root)


def test_registry_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / "experiments" / "X-01.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "  status: registered\n",
        "  status: failed\n  status: registered\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    _update_registry_card_hash(root, "X-01")

    with pytest.raises(ExperimentRegistryError, match="duplicate YAML key"):
        load_experiment_registry(root)


@pytest.mark.parametrize("registry_name", ["experiment_registry.csv", "../charter/catalog_registry.csv"])
def test_registry_rejects_csv_extra_columns(
    tmp_path: Path, registry_name: str
) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / registry_name
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] += ",unexpected"
    lines[1] += ",value"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentRegistryError, match="CSV columns"):
        load_experiment_registry(root)


def test_registry_rejects_csv_row_overflow(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / "experiment_registry.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] += ",overflow"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentRegistryError, match="CSV row"):
        load_experiment_registry(root)


def test_registry_rejects_duplicate_catalog_experiment_links(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    catalog_path = root / "charter" / "catalog_registry.csv"
    with catalog_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    duplicated_row = next(row for row in rows if row["catalog_item_id"] == "R-038")
    duplicated_row["linked_experiments"] = "X-01;X-01"
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    card = _read_card(root, "X-01")
    duplicated_gate = copy.deepcopy(card["linked_first_artifact_due_gates"][0])
    card["linked_first_artifact_due_gates"].insert(1, duplicated_gate)
    card["source_lineage"]["catalog_item_ids"].insert(1, "R-038")
    _rewrite_registered_card(root, "X-01", card)

    with pytest.raises(ExperimentRegistryError, match="duplicate.*link"):
        load_experiment_registry(root)


def test_registry_rejects_whitespace_in_catalog_links(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "charter" / "catalog_registry.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(",X-06,assigned", ", X-06,assigned", 1), encoding="utf-8")

    with pytest.raises(ExperimentRegistryError, match="whitespace"):
        load_experiment_registry(root)


def test_registry_rejects_symlinked_card_even_when_content_hash_matches(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    card_path = root / "registries" / "experiments" / "X-01.yaml"
    outside_path = tmp_path / "outside-X-01.yaml"
    outside_path.write_bytes(card_path.read_bytes())
    card_path.unlink()
    os.symlink(outside_path, card_path)

    with pytest.raises(ExperimentRegistryError, match="symlink|escape"):
        load_experiment_registry(root)


def test_registry_rejects_symlinked_artifact_dependency(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    dependency_path = root / "contracts" / "event-envelope" / "v0.schema.yaml"
    outside_path = tmp_path / "outside-event-envelope.yaml"
    outside_path.write_bytes(dependency_path.read_bytes())
    dependency_path.unlink()
    os.symlink(outside_path, dependency_path)

    with pytest.raises(ExperimentRegistryError, match="symlink|escape"):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "amended_at",
    [
        "2026-07-23 00:00:01Z",
        "2026-07-23X00:00:01Z",
        "2026-07-23T00:00:01+00:00",
        "2026-W30-4T00:00:01Z",
    ],
)
def test_amendment_timestamp_must_be_canonical_utc(
    tmp_path: Path, amended_at: str
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-01",
        amended_at=amended_at,
        changes={"status": "running"},
    )

    with pytest.raises(ExperimentRegistryError, match="canonical UTC"):
        load_experiment_registry(root)


def test_amendment_timestamps_must_be_strictly_monotonic(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:02Z",
        changes={"status": "running"},
    )
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        changes={"status": "failed"},
    )

    with pytest.raises(ExperimentRegistryError, match="monotonic"):
        load_experiment_registry(root)


@pytest.mark.parametrize("approved_by", ["A", "H+A", "team h", ""])
def test_amendments_require_exact_team_h_approval(
    tmp_path: Path, approved_by: str
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        approved_by=approved_by,
        changes={"status": "running"},
    )

    with pytest.raises(ExperimentRegistryError, match="approved_by"):
        load_experiment_registry(root)


def test_amendment_rejects_uncontrolled_base_changes(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        changes={"hypothesis": "replace immutable scientific intent"},
    )

    with pytest.raises(ExperimentRegistryError, match="controlled"):
        load_experiment_registry(root)


def test_amendment_cannot_authorize_permanent_no_go(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-10",
        amended_at="2026-07-23T00:00:01Z",
        changes={"authorize_scopes": ["live_arbitrage"]},
    )

    with pytest.raises(ExperimentRegistryError, match="permanent NO-GO"):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    ("experiment_id", "scope_name"),
    [
        ("X-11", "formal_result"),
        ("X-12", "formal_promotion"),
    ],
)
def test_amendment_cannot_promote_pipeline_or_poc_to_formal_scope(
    tmp_path: Path,
    experiment_id: str,
    scope_name: str,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        experiment_id,
        amended_at="2026-07-23T00:00:01Z",
        changes={"authorize_scopes": [scope_name]},
    )

    with pytest.raises(
        ExperimentRegistryError,
        match="permanent NO-GO|formal promotion|formal scope",
    ):
        load_experiment_registry(root)


def test_amendment_ledger_and_card_chain_must_match_both_directions(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        changes={"status": "running"},
    )
    card = _read_card(root, "X-01")
    card["amendments"] = []
    _write_card(root, "X-01", card)
    _update_registry_card_hash(root, "X-01")

    with pytest.raises(ExperimentRegistryError, match="ledger.*chain|chain.*ledger"):
        load_experiment_registry(root)

    root = _copy_program_fixture(tmp_path / "reverse")
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        changes={"status": "running"},
    )
    ledger_path = root / "registries" / "experiment_amendment_ledger.csv"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentRegistryError, match="ledger.*chain|chain.*ledger"):
        load_experiment_registry(root)


def test_effective_amendment_applies_status_locks_and_inputs(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _preregister_x08_inputs(root)
    registry = load_experiment_registry(root)
    x08 = registry["X-08"]

    assert x08["registration_head_sha256"] == amendment["amendment_sha256"]
    assert x08["registration_locks"][0]["status"] == "resolved"
    assert x08["registration_locks"][0]["evidence_ref"] == "sha256:" + "8" * 64
    assert x08["preregistered_inputs"]["archive_audit"] == {
        "code_sha256": "sha256:" + "1" * 64,
        "data_sha256": "sha256:" + "2" * 64,
        "dataset_ids": ["DS-KALSHI-HISTORICAL"],
        "model_ids": [],
        "registered_at": "2026-07-23T00:00:02Z",
    }


class _ChangingResultMapping(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(
            {
                "scope": "archive_audit",
                "result_label": "FORMAL",
                "evaluation_started_at": "2026-07-23T00:00:02Z",
                "code_sha256": "sha256:" + "1" * 64,
                "data_sha256": "sha256:" + "2" * 64,
                "result_sha256": "sha256:" + "3" * 64,
            }
        )
        self._reads = 0

    def __getitem__(self, key: str) -> Any:
        if key == "result_sha256":
            self._reads += 1
            if self._reads >= 3:
                return "not-a-hash"
        return super().__getitem__(key)


def test_result_ref_rejects_non_plain_mapping_toctou(program_root: Path) -> None:
    with pytest.raises(InvalidResultReferenceError, match="plain dict"):
        validate_result_ref(program_root, "X-08", _ChangingResultMapping())


@pytest.mark.parametrize(
    "field,value",
    [
        ("result_label", None),
        ("scope", 7),
        ("evaluation_started_at", 1),
        ("registration_head_sha256", None),
        ("code_sha256", b"not-text"),
    ],
)
def test_result_ref_rejects_null_and_wrong_field_types(
    program_root: Path, field: str, value: Any
) -> None:
    result_ref: dict[str, Any] = _valid_result_ref()
    result_ref[field] = value
    with pytest.raises(InvalidResultReferenceError, match=field):
        validate_result_ref(program_root, "X-08", result_ref)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-23 00:00:02Z",
        "2026-07-23X00:00:02Z",
        "2026-07-23T00:00:02+00:00",
        "2026-W30-4T00:00:02Z",
        "2026-07-23T00:00:02-00:00",
    ],
)
def test_result_timestamp_requires_exact_utc_form(
    tmp_path: Path, timestamp: str
) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _preregister_x08_inputs(root)
    result_ref = _valid_result_ref(
        evaluation_started_at=timestamp,
        registration_head_sha256=amendment["amendment_sha256"],
    )

    with pytest.raises(InvalidResultReferenceError, match="canonical UTC"):
        validate_result_ref(root, "X-08", result_ref)


def test_seed_cards_reject_results_until_scope_inputs_are_preregistered(
    program_root: Path,
) -> None:
    registry = load_experiment_registry(program_root)
    for experiment_id, card in registry.items():
        authorized_scopes = [
            (name, value)
            for name, value in card["authorization_scopes"].items()
            if value["authorized"] and not value.get("permanent_no_go", False)
        ]
        if not authorized_scopes:
            assert card["execution_authorized"] is False
            continue
        scope_name, scope = authorized_scopes[0]
        result_ref = _valid_result_ref(
            scope=scope_name,
            result_label=scope["required_result_label"],
            registration_head_sha256=card["registration_head_sha256"],
        )
        if card["execution_authorized"] is False:
            with pytest.raises(UnauthorizedResultScopeError, match="execution"):
                validate_result_ref(program_root, experiment_id, result_ref)
        else:
            with pytest.raises(InvalidResultReferenceError, match="preregistered"):
                validate_result_ref(program_root, experiment_id, result_ref)


def test_valid_result_requires_matching_preregistered_inputs_and_head(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _preregister_x08_inputs(root)
    result_ref = _valid_result_ref(
        registration_head_sha256=amendment["amendment_sha256"]
    )

    assert validate_result_ref(root, "X-08", result_ref) == result_ref

    wrong_input = dict(result_ref, data_sha256="sha256:" + "a" * 64)
    with pytest.raises(InvalidResultReferenceError, match="preregistered"):
        validate_result_ref(root, "X-08", wrong_input)

    wrong_head = dict(result_ref, registration_head_sha256="sha256:" + "b" * 64)
    with pytest.raises(InvalidResultReferenceError, match="registration head"):
        validate_result_ref(root, "X-08", wrong_head)


@pytest.mark.parametrize(
    ("dataset_ids", "model_ids"),
    [
        ([], []),
        (["DS-KALSHI-HISTORICAL", "DS-PMXT-V2"], []),
        (["DS-KALSHI-HISTORICAL"], ["MODEL-NFL-LOGISTIC"]),
    ],
)
def test_result_requires_exact_scope_dataset_and_model_bindings(
    tmp_path: Path,
    dataset_ids: list[str],
    model_ids: list[str],
) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _preregister_x08_inputs(root)
    result = _valid_result_ref(
        registration_head_sha256=amendment["amendment_sha256"],
        dataset_ids=dataset_ids,
        model_ids=model_ids,
    )

    with pytest.raises(InvalidResultReferenceError, match="binding"):
        validate_result_ref(root, "X-08", result)


def test_appended_result_runs_dataset_license_eligibility_gate(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    preregistration = _preregister_x08_inputs(root)
    _update_dataset_registry(
        root,
        "DS-KALSHI-HISTORICAL",
        license_status="pending",
    )
    result = _valid_result_ref(
        registration_head_sha256=preregistration["amendment_sha256"]
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:04Z",
        changes={"results_ref": result},
    )

    with pytest.raises(ExperimentRegistryError, match="license"):
        load_experiment_registry(root)


def test_evaluation_must_follow_input_preregistration_amendment(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    amendment = _preregister_x08_inputs(root)
    result_ref = _valid_result_ref(
        evaluation_started_at="2026-07-23T00:00:01Z",
        registration_head_sha256=amendment["amendment_sha256"],
    )

    with pytest.raises(PreRegistrationEvaluationError, match="input preregistration"):
        validate_result_ref(root, "X-08", result_ref)


def test_generated_result_can_be_appended_only_against_its_evaluation_head(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    preregistration = _preregister_x08_inputs(root)
    result_ref = _valid_result_ref(
        registration_head_sha256=preregistration["amendment_sha256"]
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:04Z",
        changes={"results_ref": result_ref, "status": "running"},
    )

    effective = load_experiment_registry(root)["X-08"]
    assert effective["status"] == "running"
    assert effective["results_ref"] == [result_ref]

    root = _copy_program_fixture(tmp_path / "bad-head")
    _preregister_x08_inputs(root)
    wrong_head = _valid_result_ref(registration_head_sha256="sha256:" + "f" * 64)
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:04Z",
        changes={"results_ref": wrong_head},
    )
    with pytest.raises(ExperimentRegistryError, match="evaluation head"):
        load_experiment_registry(root)


def test_result_cannot_use_a_registration_head_created_after_evaluation(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _preregister_x08_inputs(root)
    later_head = _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:04Z",
        changes={"status": "running"},
    )
    result_ref = _valid_result_ref(
        evaluation_started_at="2026-07-23T00:00:03Z",
        registration_head_sha256=later_head["amendment_sha256"],
    )

    with pytest.raises(PreRegistrationEvaluationError, match="registration head"):
        validate_result_ref(root, "X-08", result_ref)


def test_result_cannot_retroactively_resolve_locks_or_authorize_scope(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _update_dataset_registry(
        root,
        "DS-KALSHI-LIVE-L2",
        use_class="canonical",
        license_status="approved",
        manifest_sha256="sha256:" + "7" * 64,
        status="registered",
    )
    _update_license_review(
        root,
        "O-003",
        status="GREEN",
        commercial_use="PERMITTED_WITH_CONDITIONS",
        redistribution="PERMITTED_WITH_CONDITIONS",
        attribution_required="YES",
        operational_use="APPROVED",
        open_blocker="",
        approval_ref="I-APPROVAL-O003-TEST",
    )
    preregistration = _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:02Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": "kalshi_capture",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                    "dataset_ids": ["DS-KALSHI-LIVE-L2"],
                    "model_ids": [],
                }
            ]
        },
    )
    card = _read_card(root, "X-08")
    result_ref = _valid_result_ref(
        scope="kalshi_capture",
        result_label="PRELIMINARY",
        evaluation_started_at="2026-07-23T00:00:03Z",
        registration_head_sha256=preregistration["amendment_sha256"],
        dataset_ids=["DS-KALSHI-LIVE-L2"],
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:04Z",
        changes={
            "resolve_locks": [
                {"lock_id": lock_id, "evidence_ref": "sha256:" + "8" * 64}
                for lock_id in card["authorization_scopes"]["kalshi_capture"][
                    "required_lock_ids"
                ]
            ],
            "authorize_scopes": ["kalshi_capture"],
            "results_ref": result_ref,
        },
    )

    with pytest.raises(ExperimentRegistryError, match="before evaluation|retroactive"):
        load_experiment_registry(root)


def test_partial_scope_result_cannot_complete_whole_experiment(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    preregistration = _preregister_x08_inputs(root)
    result_ref = _valid_result_ref(
        registration_head_sha256=preregistration["amendment_sha256"]
    )
    _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:04Z",
        changes={"results_ref": result_ref, "status": "done"},
    )

    with pytest.raises(ExperimentRegistryError, match="completion scope"):
        load_experiment_registry(root)


def test_dependency_must_be_ready_before_dependent_evaluation(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    _complete_scope(
        root,
        "X-01",
        "formal_result",
        preregistered_at="2026-07-24T00:00:03Z",
        evaluated_at="2026-07-24T00:00:05Z",
        completed_at="2026-07-24T00:00:06Z",
    )
    _complete_scope(
        root,
        "X-02",
        "formal_result",
        preregistered_at="2026-07-24T00:00:01Z",
        evaluated_at="2026-07-24T00:00:04Z",
        completed_at="2026-07-24T00:00:07Z",
    )

    with pytest.raises(ExperimentRegistryError, match="dependency.*before evaluation"):
        load_experiment_registry(root)


@pytest.mark.parametrize(
    "first_status,second_status",
    [("failed", "running"), ("abandoned", "registered"), ("done", "running")],
)
def test_terminal_status_cannot_regress(
    tmp_path: Path, first_status: str, second_status: str
) -> None:
    root = _copy_program_fixture(tmp_path)
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:01Z",
        changes={"status": first_status},
    )
    _append_amendment(
        root,
        "X-01",
        amended_at="2026-07-23T00:00:02Z",
        changes={"status": second_status},
    )

    with pytest.raises(ExperimentRegistryError, match="terminal|transition"):
        load_experiment_registry(root)


def test_unindexed_extra_experiment_card_is_rejected(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    shutil.copy2(
        root / "registries" / "experiments" / "X-01.yaml",
        root / "registries" / "experiments" / "X-99.yaml",
    )

    with pytest.raises(ExperimentRegistryError, match="card inventory"):
        load_experiment_registry(root)


def test_artifact_registry_is_strict_and_unique(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / "artifact_registry.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] += ",unexpected"
    lines[1] += ",value"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ExperimentRegistryError, match="artifact.*CSV columns|CSV columns"):
        load_experiment_registry(root)

    root = _copy_program_fixture(tmp_path / "duplicate")
    path = root / "registries" / "artifact_registry.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ExperimentRegistryError, match="duplicate artifact"):
        load_experiment_registry(root)


def test_dependency_status_done_without_evidence_is_not_ready(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    card = _read_card(root, "X-01")
    card["status"] = "done"
    _rewrite_registered_card(root, "X-01", card)
    _update_registry_field(root, "X-01", "status", "done")
    x02 = _read_card(root, "X-02")
    for lock in x02["registration_locks"]:
        lock["status"] = "resolved"
    x02["status"] = "done"
    _rewrite_registered_card(root, "X-02", x02)
    _update_registry_field(root, "X-02", "status", "done")

    with pytest.raises(ExperimentRegistryError, match="trusted base registration"):
        load_experiment_registry(root)


def test_spec_specific_locks_signals_outputs_and_artifact_dependencies(
    program_root: Path,
) -> None:
    registry = load_experiment_registry(program_root)

    assert set(
        registry["X-07"]["authorization_scopes"]["preliminary_pipeline"][
            "required_lock_ids"
        ]
    ) == {
        "x07_event_depth_manifest",
        "order_size_and_vwap_policy",
        "markout_horizons",
    }
    assert set(
        registry["X-08"]["authorization_scopes"]["archive_audit"][
            "required_lock_ids"
        ]
    ) == {"archive_audit_input_manifest", "h_split_approval"}
    assert "archive_audit_input_manifest" in registry["X-08"][
        "authorization_scopes"
    ]["dual_venue_result"]["required_lock_ids"]
    assert set(
        registry["X-10"]["authorization_scopes"]["precision_audit"][
            "required_lock_ids"
        ]
    ) == {
        "matched_sample_registered",
        "router_and_taxonomy_available",
        "gold_standard_protocol",
        "h_split_approval",
    }
    assert {
        "router_and_taxonomy_available",
        "h_split_approval",
    } <= set(
        registry["X-10"]["authorization_scopes"]["recall"][
            "required_lock_ids"
        ]
    )
    assert registry["X-08"]["completion_required_scopes"] == [
        "dual_venue_result"
    ]
    assert registry["X-07"]["completion_required_scopes"] == ["formal_result"]
    assert registry["X-09"]["signal"] == "buy five seconds after score"
    assert set(registry["X-09"]["artifact_dependencies"][0]) == {
        "path",
        "version",
        "sha256",
    }
    assert registry["X-09"]["artifact_dependencies"][0]["path"] == (
        "contracts/event-envelope/v0.schema.yaml"
    )
    assert registry["X-05"]["artifact_dependencies"][0]["path"] == (
        "artifacts/validation/validation_standard_v0.md"
    )
    x10_pass = registry["X-10"]["pass_criteria"].lower()
    assert "relation taxonomy" in x10_pass
    assert "review queue" in x10_pass
    assert "g1 go/no-go input" in x10_pass
