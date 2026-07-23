from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from prediction_market import program_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_DATASET_IDS = {
    "DS-PMXT-V2",
    "DS-POLYMARKET-PUBLIC",
    "DS-POLYMARKET-V1",
    "DS-KALSHI-HISTORICAL",
    "DS-PREDICTION-MARKET-BENCH",
    "DS-POLYBENCH",
    "DS-NBA-CANDIDATE",
    "DS-NFLVERSE",
    "DS-STATSBOMB-OPEN",
    "DS-RETROSHEET",
    "DS-F1-JOLPICA",
    "DS-F1-FASTF1-TIMING",
}

EXPECTED_MODEL_IDS = {
    "MODEL-NBA-PRIOR",
    "MODEL-NBA-LOGISTIC",
    "MODEL-NBA-GBDT",
    "MODEL-NBA-POSSESSION-TRANSITION",
    "MODEL-NFL-SPREAD-PRIOR",
    "MODEL-NFL-LOGISTIC",
    "MODEL-NFL-GBDT",
    "MODEL-NFL-NFLFASTR-COMPARATOR",
    "MODEL-NFL-DRIVE-TRANSITION",
    "MODEL-SOCCER-DIXON-COLES",
    "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
}


def _copy_program_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "program"
    shutil.copytree(PROJECT_ROOT / "charter", root / "charter")
    shutil.copytree(PROJECT_ROOT / "registries", root / "registries")
    shutil.copytree(PROJECT_ROOT / "contracts", root / "contracts")
    shutil.copytree(PROJECT_ROOT / "artifacts", root / "artifacts")
    return root


def _rewrite_cell(
    path: Path, *, row_id: str, id_field: str, field: str, value: str
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    row = next(item for item in rows if item[id_field] == row_id)
    row[field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_registry_covers_canonical_secondary_and_blocked_sources() -> None:
    rows = program_audit.load_dataset_registry(PROJECT_ROOT)
    by_id = {row.dataset_id: row for row in rows}

    assert set(by_id) == EXPECTED_DATASET_IDS
    assert by_id["DS-PMXT-V2"].catalog_item_ids == ("I-013", "O-006")
    assert by_id["DS-POLYMARKET-PUBLIC"].catalog_item_ids == (
        "I-001",
        "I-003",
        "O-001",
    )
    assert by_id["DS-POLYMARKET-V1"].catalog_item_ids == ("R-039",)
    assert by_id["DS-POLYMARKET-V1"].use_class == "canonical"
    assert by_id["DS-POLYMARKET-V1"].license == "CC-BY-4.0"
    assert by_id["DS-KALSHI-HISTORICAL"].catalog_item_ids == ("I-005",)
    assert by_id["DS-PREDICTION-MARKET-BENCH"].use_class == "secondary"
    assert by_id["DS-PREDICTION-MARKET-BENCH"].license_status == "pending"
    assert by_id["DS-POLYBENCH"].use_class == "secondary"
    assert by_id["DS-POLYBENCH"].license_status == "pending"
    assert by_id["DS-NBA-CANDIDATE"].use_class == "blocked"
    assert by_id["DS-NBA-CANDIDATE"].status == "blocked"
    assert by_id["DS-KALSHI-HISTORICAL"].auth == "public"
    assert by_id["DS-NFLVERSE"].license == "CC-BY-4.0"
    assert by_id["DS-RETROSHEET"].allowed_experiments == ()
    assert by_id["DS-F1-JOLPICA"].catalog_item_ids == ("O-008",)
    assert by_id["DS-F1-JOLPICA"].license == "CC-BY-NC-SA-4.0"
    assert by_id["DS-F1-JOLPICA"].license_status == "research_only"
    assert by_id["DS-F1-JOLPICA"].allowed_experiments == ()
    assert by_id["DS-F1-FASTF1-TIMING"].catalog_item_ids == ("O-008",)
    assert by_id["DS-F1-FASTF1-TIMING"].license == "UPSTREAM-RIGHTS-UNRESOLVED"
    assert by_id["DS-F1-FASTF1-TIMING"].license_status == "blocked"
    assert by_id["DS-F1-FASTF1-TIMING"].allowed_experiments == ()


def test_dataset_registry_rows_have_fixed_governance_metadata() -> None:
    rows = program_audit.load_dataset_registry(PROJECT_ROOT)

    assert all(row.canonical_url.startswith("https://") for row in rows)
    assert all(row.source_version and row.coverage and row.grain for row in rows)
    assert all(row.auth and row.license and row.timestamp_semantics for row in rows)
    assert all(isinstance(row.allowed_experiments, tuple) for row in rows)
    assert all(row.manifest_sha256 == "UNRESOLVED" for row in rows)
    assert all(row.owner and row.version == "v0" and row.due_gate for row in rows)


def test_dataset_registry_catalog_foreign_keys_are_strict(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / "dataset_registry.csv"
    _rewrite_cell(
        path,
        row_id="DS-NFLVERSE",
        id_field="dataset_id",
        field="catalog_item_ids",
        value="I-999",
    )

    with pytest.raises(program_audit.ResearchRegistryError, match="catalog"):
        program_audit.load_dataset_registry(root)


def test_model_registry_covers_required_sport_models_and_transitions() -> None:
    rows = program_audit.load_model_registry(PROJECT_ROOT)
    by_id = {row.model_id: row for row in rows}

    assert set(by_id) == EXPECTED_MODEL_IDS
    assert {
        row.model_id
        for row in rows
        if row.experiment_id == "X-06"
    } == {
        "MODEL-NBA-PRIOR",
        "MODEL-NBA-LOGISTIC",
        "MODEL-NBA-GBDT",
        "MODEL-NBA-POSSESSION-TRANSITION",
    }
    assert by_id["MODEL-NBA-POSSESSION-TRANSITION"].horizon == (
        "next_state_transition"
    )
    assert by_id["MODEL-NBA-POSSESSION-TRANSITION"].state_space == (
        "home_score",
        "away_score",
        "no_score",
    )
    assert by_id["MODEL-NFL-DRIVE-TRANSITION"].experiment_id == "X-11"
    assert by_id["MODEL-NFL-DRIVE-TRANSITION"].horizon == (
        "next_state_transition"
    )
    assert by_id["MODEL-NFL-DRIVE-TRANSITION"].state_space == (
        "touchdown",
        "field_goal",
        "punt",
        "turnover",
        "other",
    )
    assert by_id["MODEL-SOCCER-DIXON-COLES"].experiment_id == "X-12"
    assert by_id["MODEL-SOCCER-FIVE-MINUTE-TRANSITION"].horizon == (
        "next_state_transition"
    )
    assert by_id["MODEL-SOCCER-FIVE-MINUTE-TRANSITION"].state_space == (
        "home_goal",
        "away_goal",
        "no_goal",
    )


def test_model_registry_has_no_fabricated_training_or_config_hashes() -> None:
    rows = program_audit.load_model_registry(PROJECT_ROOT)

    assert all(row.data_manifest_sha256 == "UNRESOLVED" for row in rows)
    assert all(row.training_manifest_sha256 == "UNRESOLVED" for row in rows)
    assert all(row.parameter_config_sha256 == "UNRESOLVED" for row in rows)
    assert all(row.seed == "UNRESOLVED" for row in rows)
    assert all(row.pit_feature_contract.startswith("unresolved:") for row in rows)
    assert all(row.metrics for row in rows)


def test_model_registry_rejects_unknown_experiment_foreign_key(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    path = root / "registries" / "model_registry.csv"
    _rewrite_cell(
        path,
        row_id="MODEL-NFL-LOGISTIC",
        id_field="model_id",
        field="experiment_id",
        value="X-99",
    )

    with pytest.raises(program_audit.ResearchRegistryError, match="experiment"):
        program_audit.load_model_registry(root)


@pytest.mark.parametrize(
    ("experiment_id", "dataset_id", "model_id", "message"),
    [
        ("X-06", "DS-NBA-CANDIDATE", "MODEL-NBA-POSSESSION-TRANSITION", "license|blocked"),
        ("X-11", "DS-NFLVERSE", "MODEL-NFL-DRIVE-TRANSITION", "manifest"),
        ("X-06", "DS-UNKNOWN", "MODEL-NBA-POSSESSION-TRANSITION", "dataset"),
    ],
)
def test_formal_source_validation_fails_closed(
    experiment_id: str, dataset_id: str, model_id: str, message: str
) -> None:
    with pytest.raises(program_audit.FormalResearchInputError, match=message):
        program_audit.validate_formal_research_inputs(
            PROJECT_ROOT,
            experiment_id=experiment_id,
            dataset_ids=[dataset_id],
            model_ids=[model_id],
        )


def test_formal_source_validation_rejects_unregistered_experiment() -> None:
    with pytest.raises(program_audit.FormalResearchInputError, match="experiment"):
        program_audit.validate_formal_research_inputs(
            PROJECT_ROOT,
            experiment_id="X-99",
            dataset_ids=["DS-NFLVERSE"],
            model_ids=["MODEL-NFL-DRIVE-TRANSITION"],
        )


def test_research_only_source_is_poc_eligible_but_never_formal() -> None:
    with pytest.raises(program_audit.FormalResearchInputError, match="manifest"):
        program_audit.validate_research_inputs(
            PROJECT_ROOT,
            experiment_id="X-12",
            dataset_ids=["DS-STATSBOMB-OPEN"],
            model_ids=["MODEL-SOCCER-DIXON-COLES"],
            result_class="poc",
        )

    with pytest.raises(program_audit.FormalResearchInputError, match="license"):
        program_audit.validate_formal_research_inputs(
            PROJECT_ROOT,
            experiment_id="X-12",
            dataset_ids=["DS-STATSBOMB-OPEN"],
            model_ids=["MODEL-SOCCER-DIXON-COLES"],
        )


def test_registry_csvs_fail_closed_on_extra_columns(tmp_path: Path) -> None:
    root = _copy_program_fixture(tmp_path)
    for filename, loader in (
        ("dataset_registry.csv", program_audit.load_dataset_registry),
        ("model_registry.csv", program_audit.load_model_registry),
    ):
        path = root / "registries" / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join([lines[0] + ",unexpected", *[line + ",value" for line in lines[1:]]])
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(program_audit.ResearchRegistryError, match="columns"):
            loader(root)
        shutil.copy2(PROJECT_ROOT / "registries" / filename, path)
