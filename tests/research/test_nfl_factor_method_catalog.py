from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS_ROOT = PROJECT_ROOT / "research" / "nfl-factor-lab" / "methods"
REGISTRY_ROOT = PROJECT_ROOT / "registries" / "methods"
CATALOG_PATH = REGISTRY_ROOT / "method_catalog_v1.json"
SCHEMA_PATH = REGISTRY_ROOT / "method_card_v1.schema.json"


def _load_catalog_module():
    spec = importlib.util.spec_from_file_location(
        "nfl_factor_method_catalog",
        METHODS_ROOT / "catalog.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPECTED_METHOD_IDS = {
    "METHOD-BDB-2025-PRESNAP",
    "METHOD-BDB-2026-ANALYTICS",
    "METHOD-FASTRMODELS-ASSETS",
    "METHOD-KAGGLE-HQIO",
    "METHOD-KAGGLE-POP",
    "METHOD-KAGGLE-SAFETY-ENTROPY",
    "METHOD-KAGGLE-XFP-PUNTS",
    "METHOD-NFL4TH-DECISION-WP",
    "METHOD-NFLFASTR-MODEL-OUTPUTS",
    "METHOD-NFLREADPY-LOADER-INTERFACE",
    "METHOD-QUARTO-PARAMETERIZED-JUPYTER",
    "METHOD-XPASS-PASS-OE",
}


def test_method_catalog_schema_and_hashes_are_valid() -> None:
    module = _load_catalog_module()
    catalog = module.load_method_catalog(CATALOG_PATH)

    assert catalog["schema_version"] == "MethodCatalogV1"
    assert {card["method_id"] for card in catalog["cards"]} == EXPECTED_METHOD_IDS
    assert [card["method_id"] for card in catalog["cards"]] == sorted(
        EXPECTED_METHOD_IDS
    )
    assert catalog["catalog_sha256"] == module.canonical_sha256(
        {
            key: value
            for key, value in catalog.items()
            if key != "catalog_sha256"
        }
    )

    for card in catalog["cards"]:
        assert card["content_sha256"] == module.canonical_sha256(
            {
                key: value
                for key, value in card.items()
                if key != "content_sha256"
            }
        )
        assert card["research"]["published_claims_not_evidence"] is True
        assert card["source"]["primary_source"] is True
        assert card["source"]["url"].startswith("https://")
        assert card["evidence_boundary"]["claim_boundary"]


def test_schema_requires_separate_code_and_data_license_evidence() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$id"] == "urn:saf:method-card:v1"
    assert set(schema["required"]) == {
        "method_id",
        "version",
        "title",
        "sport",
        "category",
        "status",
        "source",
        "licenses",
        "research",
        "reuse",
        "evidence_boundary",
        "content_sha256",
    }
    assert set(schema["properties"]["licenses"]["required"]) == {"code", "data"}


def test_verified_framework_versions_and_local_reuse_boundaries() -> None:
    module = _load_catalog_module()
    cards = {
        card["method_id"]: card
        for card in module.load_method_catalog(CATALOG_PATH)["cards"]
    }

    nflfastr = cards["METHOD-NFLFASTR-MODEL-OUTPUTS"]
    assert nflfastr["source"]["precise_version"] == (
        "v5.2.0@675a817a1563d9c4f18cbb127caecf6e9a33be25"
    )
    assert nflfastr["licenses"]["code"]["spdx_id"] == "MIT"
    assert nflfastr["licenses"]["data"]["spdx_id"] == "CC-BY-4.0"
    assert "DS-NFLVERSE" in nflfastr["reuse"]["local_dependencies"]
    assert nflfastr["reuse"]["implementation_state"] == (
        "INTEGRATED_FROZEN_OUTPUTS"
    )

    fastrmodels = cards["METHOD-FASTRMODELS-ASSETS"]
    assert fastrmodels["source"]["precise_version"] == (
        "model_archive@9f2495fdb4943087ca663d96706eb5df7973aff4"
    )
    assert "DS-NFL-FASTRMODELS" in fastrmodels["reuse"]["local_dependencies"]

    quarto = cards["METHOD-QUARTO-PARAMETERIZED-JUPYTER"]
    assert quarto["source"]["precise_version"] == (
        "v1.9.38@6ebb5db80eb542ac76189c3d7c33ae6f654b93d2"
    )
    assert (
        "47089a5020cfb41981ba0d4b46e110edfa608722aea45ef248e14efba6d6b18a"
        in quarto["source"]["provenance_notes"]
    )


def test_kaggle_ideas_fail_closed_on_unverified_license_and_missing_tracking() -> None:
    module = _load_catalog_module()
    cards = {
        card["method_id"]: card
        for card in module.load_method_catalog(CATALOG_PATH)["cards"]
    }

    kaggle_ids = {
        "METHOD-KAGGLE-XFP-PUNTS",
        "METHOD-KAGGLE-POP",
        "METHOD-KAGGLE-SAFETY-ENTROPY",
        "METHOD-KAGGLE-HQIO",
    }
    for method_id in kaggle_ids:
        card = cards[method_id]
        assert card["status"] in {"METHOD_REFERENCE_ONLY", "DATA_GAP"}
        assert "UNKNOWN" in {
            card["licenses"]["code"]["status"],
            card["licenses"]["data"]["status"],
        }
        assert card["reuse"]["implementation_state"] == "NOT_INTEGRATED"
        assert any(
            "tracking" in gap.lower()
            for gap in card["evidence_boundary"]["data_gaps"]
        )
        assert any(
            "copy" in prohibited.lower() or "download" in prohibited.lower()
            for prohibited in card["reuse"]["prohibited_uses"]
        )


def test_loader_rejects_tampering_duplicates_and_unlicensed_ready_card(
    tmp_path: Path,
) -> None:
    module = _load_catalog_module()
    original = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    tampered = copy.deepcopy(original)
    tampered["cards"][0]["title"] = "tampered"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(module.CatalogValidationError, match="content_sha256"):
        module.load_method_catalog(tampered_path)

    duplicate = copy.deepcopy(original)
    duplicate["cards"].append(copy.deepcopy(duplicate["cards"][0]))
    duplicate["catalog_sha256"] = module.canonical_sha256(
        {key: value for key, value in duplicate.items() if key != "catalog_sha256"}
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(module.CatalogValidationError, match="duplicate"):
        module.load_method_catalog(duplicate_path)

    unlicensed = copy.deepcopy(original)
    card = next(
        item
        for item in unlicensed["cards"]
        if item["method_id"] == "METHOD-KAGGLE-XFP-PUNTS"
    )
    card["status"] = "READY_FOR_LOCAL_REPRODUCTION"
    card["content_sha256"] = module.canonical_sha256(
        {key: value for key, value in card.items() if key != "content_sha256"}
    )
    unlicensed["catalog_sha256"] = module.canonical_sha256(
        {
            key: value
            for key, value in unlicensed.items()
            if key != "catalog_sha256"
        }
    )
    unlicensed_path = tmp_path / "unlicensed.json"
    unlicensed_path.write_text(json.dumps(unlicensed), encoding="utf-8")
    with pytest.raises(module.CatalogValidationError, match="license"):
        module.load_method_catalog(unlicensed_path)


def test_canonical_sha256_is_key_order_independent() -> None:
    module = _load_catalog_module()
    assert module.canonical_sha256({"b": 2, "a": 1}) == module.canonical_sha256(
        {"a": 1, "b": 2}
    )
