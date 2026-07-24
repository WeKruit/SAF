from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from prediction_market import program_audit
from prediction_market.compliance import load_data_license_register


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_DATASET_IDS = {
    "DS-PMXT-V2",
    "DS-POLYMARKET-PUBLIC",
    "DS-POLYMARKET-V1",
    "DS-KALSHI-HISTORICAL",
    "DS-KALSHI-LIVE-L2",
    "DS-PREDICTION-MARKET-BENCH",
    "DS-POLYBENCH",
    "DS-NBA-CANDIDATE",
    "DS-NFLVERSE",
    "DS-NFL-FASTRMODELS",
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
    "MODEL-NFL-FASTRMODELS-NO-SPREAD",
    "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
    "MODEL-NFL-DRIVE-TRANSITION",
    "MODEL-SOCCER-DIXON-COLES",
    "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
}

EXPECTED_LICENSE_REVIEW_IDS = {
    "DS-PMXT-V2": "O-006",
    "DS-POLYMARKET-PUBLIC": "O-001",
    "DS-POLYMARKET-V1": "R-039",
    "DS-KALSHI-HISTORICAL": "O-003",
    "DS-KALSHI-LIVE-L2": "O-003",
    "DS-PREDICTION-MARKET-BENCH": "R-042",
    "DS-POLYBENCH": "R-043",
    "DS-NBA-CANDIDATE": "O-005",
    "DS-NFLVERSE": "I-018",
    "DS-NFL-FASTRMODELS": "I-018",
    "DS-STATSBOMB-OPEN": "O-004",
    "DS-RETROSHEET": "O-007",
    "DS-F1-JOLPICA": "O-008",
    "DS-F1-FASTF1-TIMING": "O-008",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_evidence(relative: str) -> dict[str, object]:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


def _without(
    value: dict[str, object], *excluded: str
) -> dict[str, object]:
    return {key: item for key, item in value.items() if key not in excluded}


def _x11_registry_bindings() -> dict[str, dict[str, str]]:
    evidence = _json_evidence(
        "artifacts/game-state/nfl/x11_real_data_pipeline_evidence_v0.json"
    )
    inventory = evidence["input_inventory"]
    feature_contract = evidence["feature_contract"]
    pit_assessment = evidence["pit_assessment"]
    configurations = evidence["model_configuration"]
    walk_forward = evidence["walk_forward"]
    assert isinstance(inventory, dict)
    assert isinstance(feature_contract, dict)
    assert isinstance(pit_assessment, dict)
    assert isinstance(configurations, dict)
    assert isinstance(walk_forward, dict)
    assert evidence["result_label"] == "PRELIMINARY"
    assert evidence["formal_result_eligible"] is False
    assert pit_assessment["method_status"] == "PIT_UNPROVEN"
    seed = str(evidence["seed"])

    data_manifest = _canonical_sha256(
        _without(inventory, "inventory_sha256")
    )
    assert data_manifest == inventory["inventory_sha256"]
    training_manifest = _canonical_sha256(
        {
            "chronology_sha256": evidence["chronology_sha256"],
            "evaluation_input_sha256": evidence["evaluation_input_sha256"],
            "walk_forward": walk_forward,
        }
    )
    common_pit = {
        "pit_assessment": pit_assessment,
        "prohibited_as_features": feature_contract["prohibited_as_features"],
        "walk_forward_contract": {
            key: walk_forward[key]
            for key in (
                "evaluation_seasons",
                "minimum_prior_train_games",
                "same_date_training_excluded",
                "seed",
                "test_unit",
                "training_rule",
                "warmup_seasons",
            )
        },
    }
    specifications = {
        "MODEL-NFL-SPREAD-PRIOR": {
            "prior_method": pit_assessment["prior_method"],
            "spread_prior_is_model_input": feature_contract[
                "spread_prior_is_model_input"
            ],
        },
        "MODEL-NFL-LOGISTIC": {
            "features": feature_contract["logistic"],
        },
        "MODEL-NFL-GBDT": {
            "features": feature_contract["gbdt"],
        },
        "MODEL-NFL-NFLFASTR-COMPARATOR": {
            "evidence_model_key": next(
                key
                for key in evidence["outcome_evaluation"]["models"]
                if key.startswith("nflfastr_")
            ),
            "role": feature_contract["nflfastr_home_wp_role"],
        },
        "MODEL-NFL-DRIVE-TRANSITION": {
            "features": feature_contract["drive_transition"],
            "state_space": configurations["drive_transition"]["state_space"],
        },
    }
    parameter_material = {
        "MODEL-NFL-SPREAD-PRIOR": {
            **specifications["MODEL-NFL-SPREAD-PRIOR"],
            "seed": evidence["seed"],
        },
        "MODEL-NFL-LOGISTIC": configurations["logistic"],
        "MODEL-NFL-GBDT": configurations["gbdt"],
        "MODEL-NFL-NFLFASTR-COMPARATOR": {
            **specifications["MODEL-NFL-NFLFASTR-COMPARATOR"],
            "seed": evidence["seed"],
        },
        "MODEL-NFL-DRIVE-TRANSITION": configurations["drive_transition"],
    }
    return {
        model_id: {
            "pit_feature_contract": _canonical_sha256(
                {**common_pit, "model_specification": specification}
            ),
            "data_manifest_sha256": data_manifest,
            "training_manifest_sha256": training_manifest,
            "parameter_config_sha256": _canonical_sha256(
                parameter_material[model_id]
            ),
            "seed": seed,
            "status": "poc_only",
        }
        for model_id, specification in specifications.items()
    }


def _x12_registry_bindings() -> dict[str, dict[str, str]]:
    evidence = _json_evidence(
        "artifacts/game-state/soccer/x12_real_data_poc_v0.json"
    )
    inventory = evidence["input_inventory"]
    model = evidence["model"]
    walk_forward = evidence["walk_forward"]
    transition = evidence["transition_output"]
    assert isinstance(inventory, dict)
    assert isinstance(model, dict)
    assert isinstance(walk_forward, dict)
    assert isinstance(transition, dict)
    assert evidence["promotion_decision"] == "POC_ONLY"
    assert evidence["formal_result_eligible"] is False
    assert transition["availability_status"] == (
        "offline_reconstruction_not_live_PIT"
    )
    seed = str(model["seed"])
    assert seed == str(evidence["bootstrap"]["seed"]) == "20260722"

    data_manifest = _canonical_sha256(
        _without(inventory, "inventory_sha256", "manifest_count")
    )
    assert data_manifest == inventory["inventory_sha256"]
    training_manifest = _canonical_sha256(
        {
            "chronology_sha256": evidence["chronology_sha256"],
            "goal_timeline_sha256": evidence["goal_timeline_sha256"],
            "walk_forward": walk_forward,
        }
    )
    transition_definition = {
        key: transition[key]
        for key in (
            "availability_status",
            "boundary_rule",
            "horizon_seconds",
            "state_space",
        )
    }
    common_pit = {
        "kickoff_time_basis": evidence["kickoff_time_basis"],
        "outcome_availability_rule": model["outcome_availability_rule"],
        "source_time_audit": evidence["source_time_audit"],
        "training_rule": model["training_rule"],
    }
    specifications = {
        "MODEL-SOCCER-DIXON-COLES": {},
        "MODEL-SOCCER-FIVE-MINUTE-TRANSITION": {
            "transition_definition": transition_definition
        },
    }
    parameter_material = {
        "MODEL-SOCCER-DIXON-COLES": model,
        "MODEL-SOCCER-FIVE-MINUTE-TRANSITION": {
            "model": model,
            "transition_definition": transition_definition,
        },
    }
    return {
        model_id: {
            "pit_feature_contract": _canonical_sha256(
                {**common_pit, **specification}
            ),
            "data_manifest_sha256": data_manifest,
            "training_manifest_sha256": training_manifest,
            "parameter_config_sha256": _canonical_sha256(
                parameter_material[model_id]
            ),
            "seed": seed,
            "status": "poc_only",
        }
        for model_id, specification in specifications.items()
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
    assert by_id["DS-KALSHI-LIVE-L2"].catalog_item_ids == ("I-004", "O-003")
    assert by_id["DS-KALSHI-LIVE-L2"].auth == "rsa_key_required"
    assert by_id["DS-KALSHI-LIVE-L2"].status == "blocked"
    assert by_id["DS-KALSHI-LIVE-L2"].allowed_experiments == ("X-08",)
    assert by_id["DS-PREDICTION-MARKET-BENCH"].use_class == "secondary"
    assert by_id["DS-PREDICTION-MARKET-BENCH"].license_status == "pending"
    assert by_id["DS-POLYBENCH"].use_class == "secondary"
    assert by_id["DS-POLYBENCH"].license_status == "pending"
    assert by_id["DS-NBA-CANDIDATE"].use_class == "blocked"
    assert by_id["DS-NBA-CANDIDATE"].status == "blocked"
    assert by_id["DS-KALSHI-HISTORICAL"].auth == "public"
    assert by_id["DS-KALSHI-HISTORICAL"].license_status == "pending"
    assert by_id["DS-PMXT-V2"].license_status == "approved"
    assert by_id["DS-NFLVERSE"].license == "CC-BY-4.0"
    assert "release_id:58152862" in by_id["DS-NFLVERSE"].source_version
    assert "observed_at:2026-07-22" in by_id["DS-NFLVERSE"].source_version
    assert by_id["DS-NFL-FASTRMODELS"].catalog_item_ids == ("I-018",)
    assert by_id["DS-NFL-FASTRMODELS"].license == "MIT"
    assert by_id["DS-NFL-FASTRMODELS"].license_status == "approved"
    assert by_id["DS-NFL-FASTRMODELS"].allowed_experiments == ("X-11",)
    assert by_id["DS-RETROSHEET"].allowed_experiments == ()
    assert by_id["DS-F1-JOLPICA"].catalog_item_ids == ("O-008",)
    assert by_id["DS-F1-JOLPICA"].license == "CC-BY-NC-SA-4.0"
    assert by_id["DS-F1-JOLPICA"].license_status == "research_only"
    assert by_id["DS-F1-JOLPICA"].allowed_experiments == ()
    assert by_id["DS-F1-FASTF1-TIMING"].catalog_item_ids == ("O-008",)
    assert by_id["DS-F1-FASTF1-TIMING"].license == "UPSTREAM-RIGHTS-UNRESOLVED"
    assert by_id["DS-F1-FASTF1-TIMING"].license_status == "blocked"
    assert by_id["DS-F1-FASTF1-TIMING"].allowed_experiments == ()


def test_dataset_registry_license_reviews_are_exact_operational_foreign_keys() -> None:
    datasets = program_audit.load_dataset_registry(PROJECT_ROOT)
    reviews = {
        row.catalog_item_id: row
        for row in load_data_license_register(PROJECT_ROOT)
    }

    assert {
        row.dataset_id: row.license_review_id for row in datasets
    } == EXPECTED_LICENSE_REVIEW_IDS
    assert all(row.license_review_id in reviews for row in datasets)
    for dataset in datasets:
        if dataset.license_status == "approved":
            review = reviews[dataset.license_review_id]
            assert review.status == "GREEN"
            assert review.operational_use == "APPROVED"
            assert review.approval_ref


def test_approved_dataset_is_blocked_immediately_when_review_is_downgraded(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    review_path = root / "registries" / "data_license_register.csv"
    for field, value in {
        "status": "NOT_GREEN_OPEN",
        "commercial_use": "UNKNOWN",
        "redistribution": "UNKNOWN",
        "attribution_required": "UNKNOWN",
        "operational_use": "RESEARCH_ONLY",
        "open_blocker": "approval revoked by test",
        "approval_ref": "",
    }.items():
        _rewrite_cell(
            review_path,
            row_id="O-006",
            id_field="catalog_item_id",
            field=field,
            value=value,
        )

    with pytest.raises(program_audit.ResearchRegistryError, match="O-006|GREEN|approval"):
        program_audit.load_dataset_registry(root)


def test_dataset_registry_rejects_unknown_license_review_foreign_key(
    tmp_path: Path,
) -> None:
    root = _copy_program_fixture(tmp_path)
    _rewrite_cell(
        root / "registries" / "dataset_registry.csv",
        row_id="DS-NFLVERSE",
        id_field="dataset_id",
        field="license_review_id",
        value="O-999",
    )

    with pytest.raises(program_audit.ResearchRegistryError, match="license review|O-999"):
        program_audit.load_dataset_registry(root)


def test_dataset_registry_rows_have_fixed_governance_metadata() -> None:
    rows = program_audit.load_dataset_registry(PROJECT_ROOT)
    by_id = {row.dataset_id: row for row in rows}

    pmxt = _json_evidence(
        "artifacts/data-audit/x02_timestamp_input_bundle_v1.json"
    )
    polymarket = _json_evidence(
        "artifacts/data-audit/polymarket_v1_bounded_sports_extract_v0.json"
    )
    nfl = _json_evidence(
        "artifacts/game-state/nfl/x11_real_data_pipeline_evidence_v0.json"
    )
    soccer = _json_evidence(
        "artifacts/game-state/soccer/x12_real_data_poc_v0.json"
    )
    retrosheet = _json_evidence(
        "artifacts/game-state/mlb/source_inventory_v0.json"
    )
    expected_manifests = {
        "DS-PMXT-V2": _canonical_sha256(
            _without(pmxt, "bundle_sha256")
        ),
        "DS-POLYMARKET-V1": polymarket["derived_extract"]["manifest_sha256"],
            "DS-NFLVERSE": _canonical_sha256(
                _without(nfl["input_inventory"], "inventory_sha256")
            ),
            "DS-NFL-FASTRMODELS": (
                "sha256:"
                "080d98f34495fe59a532b7c24e17536f471700e92ac8415b"
                "682234d7241fe3cb"
            ),
            "DS-STATSBOMB-OPEN": _canonical_sha256(
            _without(
                soccer["input_inventory"],
                "inventory_sha256",
                "manifest_count",
            )
        ),
        "DS-RETROSHEET": retrosheet["dataset"]["manifest_sha256"],
    }

    assert all(row.canonical_url.startswith("https://") for row in rows)
    assert all(row.source_version and row.coverage and row.grain for row in rows)
    assert all(row.auth and row.license and row.timestamp_semantics for row in rows)
    assert all(isinstance(row.allowed_experiments, tuple) for row in rows)
    assert {
        dataset_id: by_id[dataset_id].manifest_sha256
        for dataset_id in expected_manifests
    } == expected_manifests
    assert {
        row.dataset_id
        for row in rows
        if row.manifest_sha256 == "UNRESOLVED"
    } == EXPECTED_DATASET_IDS - set(expected_manifests)
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
    assert (
        by_id["MODEL-NFL-FASTRMODELS-NO-SPREAD"].experiment_id
        == "X-11"
    )
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


def test_model_registry_hashes_are_recomputed_from_checked_in_evidence() -> None:
    rows = program_audit.load_model_registry(PROJECT_ROOT)
    by_id = {row.model_id: row for row in rows}
    expected = {**_x11_registry_bindings(), **_x12_registry_bindings()}

    assert {
        model_id: {
            "pit_feature_contract": by_id[model_id].pit_feature_contract,
            "data_manifest_sha256": by_id[model_id].data_manifest_sha256,
            "training_manifest_sha256": by_id[model_id].training_manifest_sha256,
            "parameter_config_sha256": by_id[
                model_id
            ].parameter_config_sha256,
            "seed": by_id[model_id].seed,
            "status": by_id[model_id].status,
        }
        for model_id in expected
    } == expected
    assert all(row.metrics for row in rows)

    nba = [row for row in rows if row.experiment_id == "X-06"]
    assert nba
    assert all(row.data_manifest_sha256 == "UNRESOLVED" for row in nba)
    assert all(row.training_manifest_sha256 == "UNRESOLVED" for row in nba)
    assert all(row.parameter_config_sha256 == "UNRESOLVED" for row in nba)
    assert all(row.seed == "UNRESOLVED" for row in nba)
    assert all(row.pit_feature_contract.startswith("unresolved:") for row in nba)
    assert all(row.status == "blocked" for row in nba)

    soccer_metrics = {
        row.model_id: row.metrics
        for row in rows
        if row.experiment_id == "X-12"
    }
    dixon_coles = soccer_metrics["MODEL-SOCCER-DIXON-COLES"]
    assert {
        "multiclass_Brier",
        "multiclass_log_loss",
        "ovr_calibration_slope",
        "ovr_calibration_intercept",
        "game_cluster_bootstrap_ci",
        "paired_prior_ci_only_if_pit_prior_registered",
    } <= set(dixon_coles)


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
        ("X-11", "DS-NFLVERSE", "MODEL-NFL-DRIVE-TRANSITION", "authorized"),
        ("X-06", "DS-UNKNOWN", "MODEL-NBA-POSSESSION-TRANSITION", "dataset"),
    ],
)
def test_formal_source_validation_fails_closed(
    experiment_id: str, dataset_id: str, model_id: str, message: str
) -> None:
    with pytest.raises(program_audit.FormalResearchInputError, match=message):
        program_audit.validate_registered_research_bindings(
            PROJECT_ROOT,
            experiment_id=experiment_id,
            dataset_ids=[dataset_id],
            model_ids=[model_id],
            result_class="formal",
        )


def test_formal_source_validation_rejects_unregistered_experiment() -> None:
    with pytest.raises(program_audit.FormalResearchInputError, match="experiment"):
        program_audit.validate_registered_research_bindings(
            PROJECT_ROOT,
            experiment_id="X-99",
            dataset_ids=["DS-NFLVERSE"],
            model_ids=["MODEL-NFL-DRIVE-TRANSITION"],
            result_class="formal",
        )


def test_research_only_source_is_poc_eligible_but_never_formal() -> None:
    datasets, models = program_audit.validate_registered_research_bindings(
        PROJECT_ROOT,
        experiment_id="X-12",
        dataset_ids=["DS-STATSBOMB-OPEN"],
        model_ids=["MODEL-SOCCER-DIXON-COLES"],
        result_class="poc",
    )
    assert [row.dataset_id for row in datasets] == ["DS-STATSBOMB-OPEN"]
    assert [row.model_id for row in models] == [
        "MODEL-SOCCER-DIXON-COLES"
    ]

    with pytest.raises(program_audit.FormalResearchInputError, match="license"):
        program_audit.validate_registered_research_bindings(
            PROJECT_ROOT,
            experiment_id="X-12",
            dataset_ids=["DS-STATSBOMB-OPEN"],
            model_ids=["MODEL-SOCCER-DIXON-COLES"],
            result_class="formal",
        )


def test_research_runner_obeys_experiment_execution_kill_switch() -> None:
    with pytest.raises(program_audit.FormalResearchInputError, match="execution"):
        program_audit.validate_research_inputs(
            PROJECT_ROOT,
            experiment_id="X-10",
            scope_name="precision_audit",
            dataset_ids=[],
            model_ids=[],
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


def test_registered_sport_experiment_ids_are_current_in_handoff_docs() -> None:
    nfl = (
        PROJECT_ROOT / "artifacts/game-state/nfl/evidence_pipeline_poc_v0.md"
    ).read_text(encoding="utf-8")
    soccer = (
        PROJECT_ROOT / "artifacts/game-state/soccer/evidence_pipeline_poc_v0.md"
    ).read_text(encoding="utf-8")
    validation = (
        PROJECT_ROOT / "artifacts/validation/validation_standard_v0.md"
    ).read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "has not issued a D2 experiment ID" not in nfl
    assert "has not issued a D3 experiment ID" not in soccer
    assert "X-11" in nfl
    assert "X-12" in soccer
    assert "`execution_authorized: true`" in nfl
    assert "`execution_authorized: true`" in soccer
    assert "formal" in nfl.lower() and "unauthorized" in nfl.lower()
    assert "formal" in soccer.lower() and "unauthorized" in soccer.lower()
    assert "X-01 through X-12" in validation
    assert "X-01 through X-12" in readme
