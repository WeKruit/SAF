from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from pathlib import Path

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
    compute_registration_record_sha256,
    load_experiment_registry,
    validate_result_ref,
)


EXPECTED_EXPERIMENT_IDS = {f"X-{number:02d}" for number in range(1, 11)}
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
    "registries/artifact_registry.csv",
    "artifacts/validation/validation_standard_v0.md",
    "src/prediction_market/experiments.py",
    "tests/test_experiment_registry.py",
    *(f"registries/experiments/X-{number:02d}.yaml" for number in range(1, 11)),
}


@pytest.fixture
def program_root() -> Path:
    return PROJECT_ROOT


def _copy_program_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "program"
    shutil.copytree(PROJECT_ROOT / "charter", root / "charter")
    shutil.copytree(PROJECT_ROOT / "registries", root / "registries")
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


def _rewrite_registered_card(root: Path, experiment_id: str, card: dict) -> None:
    card["registration_record_sha256"] = compute_registration_record_sha256(card)
    _write_card(root, experiment_id, card)
    _update_registry_card_hash(root, experiment_id)


def _valid_result_ref(
    *, scope: str = "archive_audit", evaluation_started_at: str = "2026-07-23T00:00:00Z"
) -> dict[str, str]:
    return {
        "scope": scope,
        "result_label": "FORMAL",
        "evaluation_started_at": evaluation_started_at,
        "code_sha256": "sha256:" + "1" * 64,
        "data_sha256": "sha256:" + "2" * 64,
        "result_sha256": "sha256:" + "3" * 64,
    }


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
    assert x08["execution_authorized"] is False
    assert x08["authorization_scopes"]["archive_audit"]["authorized"] is True
    assert x08["authorization_scopes"]["polymarket_capture"]["authorized"] is True
    assert x08["authorization_scopes"]["kalshi_capture"]["authorized"] is False
    assert x08["authorization_scopes"]["dual_venue_result"]["authorized"] is False

    x10 = registry["X-10"]
    assert x10["execution_authorized"] is False
    assert x10["authorization_scopes"]["precision_audit"]["authorized"] is True
    assert set(
        x10["authorization_scopes"]["precision_audit"]["required_lock_ids"]
    ) == {"matched_sample_registered", "router_and_taxonomy_available"}
    assert x10["authorization_scopes"]["recall"]["authorized"] is False
    assert x10["authorization_scopes"]["live_arbitrage"]["authorized"] is False
    assert x10["authorization_scopes"]["live_arbitrage"]["permanent_no_go"] is True

    for experiment_id in EXPECTED_EXPERIMENT_IDS - {"X-05", "X-07", "X-08", "X-10"}:
        assert registry[experiment_id]["execution_authorized"] is True


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

    assert {row["path"] for row in rows} == EXPECTED_TASK_3_PATHS
    assert all(row["owner_team"] for row in rows)
    assert all(row["version"] for row in rows)
    assert all(row["due_gate"] for row in rows)
    assert {row["status"] for row in rows} == {"registered"}


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


def test_evaluation_must_start_after_registration(program_root: Path) -> None:
    too_early = _valid_result_ref(evaluation_started_at="2026-07-22T23:59:59Z")
    with pytest.raises(PreRegistrationEvaluationError):
        validate_result_ref(program_root, "X-08", too_early)

    validated = validate_result_ref(program_root, "X-08", _valid_result_ref())
    assert validated["scope"] == "archive_audit"


def test_result_rejects_unresolved_registration_locks(program_root: Path) -> None:
    result_ref = _valid_result_ref(scope="formal_result")
    with pytest.raises(UnresolvedRegistrationLockError):
        validate_result_ref(program_root, "X-01", result_ref)


def test_result_rejects_unfinished_dependencies(program_root: Path) -> None:
    result_ref = _valid_result_ref(scope="formal_result")
    with pytest.raises(UnresolvedDependencyError):
        validate_result_ref(program_root, "X-04", result_ref)


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
