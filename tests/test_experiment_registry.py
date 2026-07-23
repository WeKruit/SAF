from __future__ import annotations

import csv
import copy
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

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


@pytest.fixture
def program_root() -> Path:
    return PROJECT_ROOT


def _copy_program_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "program"
    shutil.copytree(PROJECT_ROOT / "charter", root / "charter")
    shutil.copytree(PROJECT_ROOT / "registries", root / "registries")
    shutil.copytree(PROJECT_ROOT / "artifacts", root / "artifacts")
    shutil.copytree(PROJECT_ROOT / "contracts", root / "contracts")
    return root


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


def _valid_result_ref(
    *,
    scope: str = "archive_audit",
    result_label: str = "FORMAL",
    evaluation_started_at: str = "2026-07-23T00:00:02Z",
    registration_head_sha256: str = "sha256:" + "4" * 64,
) -> dict[str, str]:
    return {
        "scope": scope,
        "result_label": result_label,
        "evaluation_started_at": evaluation_started_at,
        "code_sha256": "sha256:" + "1" * 64,
        "data_sha256": "sha256:" + "2" * 64,
        "result_sha256": "sha256:" + "3" * 64,
        "registration_head_sha256": registration_head_sha256,
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


def _preregister_x08_inputs(root: Path) -> dict[str, Any]:
    return _append_amendment(
        root,
        "X-08",
        amended_at="2026-07-23T00:00:01Z",
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
                }
            ],
        },
    )


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
            ],
            "preregistered_inputs": [
                {
                    "scope": scope,
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                }
            ],
        },
    )
    result_ref = _valid_result_ref(
        scope=scope,
        result_label=card["authorization_scopes"][scope]["required_result_label"],
        evaluation_started_at=evaluated_at,
        registration_head_sha256=preregistration["amendment_sha256"],
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
        if experiment_id in {"X-11", "X-12"}:
            catalog_by_id = {
                row["catalog_item_id"]: row for row in catalog_rows
            }
            assert card["linked_first_artifact_due_gates"] == [
                {
                    "catalog_item_id": catalog_id,
                    "due_gate": catalog_by_id[catalog_id]["due_gate"],
                }
                for catalog_id in card["source_lineage"]["catalog_item_ids"]
            ]
            continue
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
    assert x11["dataset_ids"] == ["DS-NFLVERSE"]
    assert x11["output_contract"]["transition_unit"] == "drive"
    lock_ids = {lock["id"] for lock in x11["registration_locks"]}
    assert {
        "nfl_data_manifest_and_version",
        "pit_feature_contract",
        "model_config_and_seed",
        "bootstrap_parameters",
    } <= lock_ids


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
    assert x12["authorization_scopes"]["poc_result"]["authorized"] is True
    assert x12["authorization_scopes"]["formal_promotion"]["authorized"] is False
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

    assert authorized == {"X-01", "X-02", "X-03", "X-06", "X-08"}
    assert registry["X-04"]["authorization_scopes"]["formal_result"][
        "authorized"
    ] is False
    assert registry["X-09"]["authorization_scopes"]["formal_result"][
        "authorized"
    ] is False


def test_x08_is_prospective_and_preserves_the_decision_band(program_root: Path) -> None:
    x08 = load_experiment_registry(program_root)["X-08"]
    hypothesis = x08["hypothesis"].lower()

    assert "prospective" in hypothesis
    assert "already" not in hypothesis
    assert x08["prospective_observation"]["actual_elapsed_days"] == 7
    assert x08["prospective_observation"]["fixtures_can_satisfy_elapsed_time"] is False
    assert "no gaps" in x08["pass_criteria"].lower()
    assert "<99%" in x08["fail_criteria"].replace(" ", "")
    assert "99% <= uptime < 100%" in x08["unresolved_decision_band"]


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
                }
            ]
        },
    )
    result_ref = _valid_result_ref(
        scope="formal_result",
        registration_head_sha256=amendment["amendment_sha256"],
    )
    with pytest.raises(UnresolvedRegistrationLockError):
        validate_result_ref(root, "X-01", result_ref)


def test_result_rejects_unfinished_dependencies(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    x02 = _read_card(root, "X-02")
    amendment = _append_amendment(
        root,
        "X-02",
        amended_at="2026-07-23T00:00:01Z",
        changes={
            "resolve_locks": [
                {
                    "lock_id": lock["id"],
                    "evidence_ref": "sha256:" + "8" * 64,
                }
                for lock in x02["registration_locks"]
            ],
            "preregistered_inputs": [
                {
                    "scope": "formal_result",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                }
            ],
        },
    )
    result_ref = _valid_result_ref(
        scope="formal_result",
        registration_head_sha256=amendment["amendment_sha256"],
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
        "registered_at": "2026-07-23T00:00:01Z",
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
        amended_at="2026-07-23T00:00:03Z",
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
        amended_at="2026-07-23T00:00:03Z",
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
        amended_at="2026-07-23T00:00:03Z",
        changes={"status": "running"},
    )
    result_ref = _valid_result_ref(
        evaluation_started_at="2026-07-23T00:00:02Z",
        registration_head_sha256=later_head["amendment_sha256"],
    )

    with pytest.raises(PreRegistrationEvaluationError, match="registration head"):
        validate_result_ref(root, "X-08", result_ref)


def test_result_cannot_retroactively_resolve_locks_or_authorize_scope(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    preregistration = _append_amendment(
        root,
        "X-10",
        amended_at="2026-07-23T00:00:01Z",
        changes={
            "preregistered_inputs": [
                {
                    "scope": "precision_audit",
                    "code_sha256": "sha256:" + "1" * 64,
                    "data_sha256": "sha256:" + "2" * 64,
                }
            ]
        },
    )
    card = _read_card(root, "X-10")
    result_ref = _valid_result_ref(
        scope="precision_audit",
        evaluation_started_at="2026-07-23T00:00:02Z",
        registration_head_sha256=preregistration["amendment_sha256"],
    )
    _append_amendment(
        root,
        "X-10",
        amended_at="2026-07-23T00:00:03Z",
        changes={
            "resolve_locks": [
                {"lock_id": lock_id, "evidence_ref": "sha256:" + "8" * 64}
                for lock_id in card["authorization_scopes"]["precision_audit"][
                    "required_lock_ids"
                ]
            ],
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
        amended_at="2026-07-23T00:00:03Z",
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
        preregistered_at="2026-07-23T00:00:03Z",
        evaluated_at="2026-07-23T00:00:05Z",
        completed_at="2026-07-23T00:00:06Z",
    )
    _complete_scope(
        root,
        "X-02",
        "formal_result",
        preregistered_at="2026-07-23T00:00:01Z",
        evaluated_at="2026-07-23T00:00:02Z",
        completed_at="2026-07-23T00:00:07Z",
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
