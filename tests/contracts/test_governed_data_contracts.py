from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from prediction_market import contracts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = PROJECT_ROOT / "contracts"


def _digest(fill: str = "0") -> str:
    return f"sha256:{fill * 64}"


def _static_manifest(*, object_kind: str = "byte_exact_original") -> dict:
    value = {
        "manifest_version": "v0",
        "dataset_id": "DS-PMXT-V2",
        "object_kind": object_kind,
        "source_url": "https://archive.pmxt.dev/polymarket/2026-07-01.parquet",
        "source_request": {
            "method": "GET",
            "path": "/polymarket/2026-07-01.parquet",
            "parameters": {},
        },
        "source_cursor": None,
        "fetched_at": "2026-07-22T12:00:00Z",
        "coverage": "polymarket:2026-07-01T00:00:00Z/2026-07-02T00:00:00Z",
        "etag": '"pmxt-20260701"',
        "last_modified": "2026-07-02T00:05:00Z",
        "byte_length": 1024,
        "object_sha256": _digest("1"),
        "native_object_path": "raw/DS-PMXT-V2/2026-07-01.parquet",
        "media_type": "application/vnd.apache.parquet",
        "schema_fingerprint": _digest("2"),
        "license_ref": "O-006",
        "license_status": "research_only",
        "upstream_partition": "venue=polymarket/date=2026-07-01",
        "lineage": {
            "source_object_refs": [],
            "query_sha256": None,
        },
    }
    value["manifest_sha256"] = contracts.static_dataset_manifest_sha256(value)
    return value


def _metadata_snapshot() -> dict:
    value = {
        "snapshot_version": "v0",
        "venue": "polymarket",
        "native_event_id": "nba-lal-bos-2026-07-22",
        "native_market_id": "0xmarket",
        "native_condition_id": "0xcondition",
        "native_outcome_id": "home",
        "native_token_id": "217426943",
        "canonical_refs": {
            "competition_id": "cmp_nba",
            "game_id": "game_nba_2026_001",
            "participant_ids": ["participant_bos", "participant_lal"],
            "venue_event_id": "venue_event_polymarket_001",
            "market_id": "market_moneyline_001",
            "outcome_id": "outcome_home_win",
            "condition_id": "condition_0xabc",
        },
        "sport": "basketball",
        "competition": "NBA",
        "participants": ["Boston Celtics", "Los Angeles Lakers"],
        "game_start_at": "2026-07-22T23:00:00Z",
        "rules": "Resolves to the official winner after final settlement.",
        "resolution": None,
        "closed": False,
        "resolved": False,
        "captured_at": "2026-07-22T12:00:00Z",
        "source_updated_at": "2026-07-22T11:59:59Z",
        "raw_object_hash": _digest("3"),
        "quality_flags": [],
    }
    value["snapshot_sha256"] = contracts.market_metadata_snapshot_sha256(value)
    return value


def _model_output() -> dict:
    return {
        "contract_version": "v1",
        "model_id": "MODEL-NBA-LOGISTIC",
        "model_version": "v1",
        "experiment_id": "X-06",
        "run_id": "run_x06_001",
        "game_id": "game_nba_2025_001",
        "state_event_id": "evt_" + "4" * 64,
        "pit_cutoff_at": "2025-12-01T20:00:00Z",
        "output_kind": "state_transition",
        "transition_unit": "possession",
        "state_space": ["away_drive", "home_drive", "game_over"],
        "horizon": "next_state_transition",
        "probabilities": {
            "away_drive": {"atoms": "25", "scale": 2},
            "home_drive": {"atoms": "50", "scale": 2},
            "game_over": {"atoms": "25", "scale": 2},
        },
        "feature_sha256": _digest("5"),
        "data_sha256": _digest("6"),
        "config_sha256": _digest("7"),
        "quality_flags": [],
    }


def test_static_dataset_manifest_validates_byte_exact_original() -> None:
    manifest = contracts.StaticDatasetManifestV0.model_validate(_static_manifest())

    assert manifest.object_kind == "byte_exact_original"
    assert manifest.lineage.source_object_refs == ()
    assert contracts.validate_contract_v0(
        "static-dataset-manifest/v0.schema.yaml", manifest
    ) == manifest


def test_static_dataset_manifest_requires_derived_lineage() -> None:
    derived = _static_manifest(object_kind="source_derived_extract")

    with pytest.raises(ValidationError, match="source_object_refs|query_sha256"):
        contracts.StaticDatasetManifestV0.model_validate(derived)

    derived["lineage"] = {
        "source_object_refs": [_digest("8")],
        "query_sha256": _digest("9"),
    }
    derived["manifest_sha256"] = contracts.static_dataset_manifest_sha256(derived)
    assert (
        contracts.StaticDatasetManifestV0.model_validate(derived).object_kind
        == "source_derived_extract"
    )


def test_byte_exact_original_cannot_claim_derived_lineage() -> None:
    original = _static_manifest()
    original["lineage"] = {
        "source_object_refs": [_digest("8")],
        "query_sha256": _digest("9"),
    }
    original["manifest_sha256"] = contracts.static_dataset_manifest_sha256(original)

    with pytest.raises(ValidationError, match="byte_exact_original"):
        contracts.StaticDatasetManifestV0.model_validate(original)


def test_static_dataset_manifest_hash_covers_all_other_fields() -> None:
    manifest = _static_manifest()
    manifest["coverage"] = "tampered"

    with pytest.raises(ValidationError, match="manifest_sha256 mismatch"):
        contracts.StaticDatasetManifestV0.model_validate(manifest)


def test_market_metadata_snapshot_is_point_in_time_and_content_hashed() -> None:
    snapshot = contracts.MarketMetadataSnapshotV0.model_validate(
        _metadata_snapshot()
    )

    assert snapshot.participants == ("Boston Celtics", "Los Angeles Lakers")
    assert snapshot.canonical_refs.condition_id == "condition_0xabc"
    assert contracts.validate_contract_v0(
        "market-metadata-snapshot/v0.schema.yaml", snapshot
    ) == snapshot


def test_market_metadata_snapshot_rejects_future_source_update() -> None:
    snapshot = _metadata_snapshot()
    snapshot["source_updated_at"] = "2026-07-22T12:00:01Z"
    snapshot["snapshot_sha256"] = contracts.market_metadata_snapshot_sha256(snapshot)

    with pytest.raises(ValidationError, match="source_updated_at.*captured_at"):
        contracts.MarketMetadataSnapshotV0.model_validate(snapshot)


def test_market_metadata_snapshot_requires_consistent_resolution_state() -> None:
    snapshot = _metadata_snapshot()
    snapshot["resolved"] = True
    snapshot["snapshot_sha256"] = contracts.market_metadata_snapshot_sha256(snapshot)

    with pytest.raises(ValidationError, match="resolved.*resolution|closed"):
        contracts.MarketMetadataSnapshotV0.model_validate(snapshot)


def test_model_output_v1_is_transition_only_and_registry_backed() -> None:
    output = contracts.validate_contract_v1(
        PROJECT_ROOT, "model-output/v1.schema.yaml", _model_output()
    )

    assert isinstance(output, contracts.ModelOutputV1)
    assert output.output_kind == "state_transition"
    assert output.transition_unit == "possession"

    unknown = dict(_model_output(), experiment_id="X-99")
    with pytest.raises(contracts.ContractValidationError, match="not registered"):
        contracts.validate_contract_v1(
            PROJECT_ROOT, "model-output/v1.schema.yaml", unknown
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_kind", "final_win_probability"),
        ("transition_unit", "game_end"),
        ("transition_unit", "final_win"),
        ("horizon", "game_end"),
    ],
)
def test_model_output_v1_rejects_final_win_only_outputs(
    field: str, value: str
) -> None:
    output = dict(_model_output())
    output[field] = value

    with pytest.raises(ValidationError):
        contracts.ModelOutputV1.model_validate(output)


def test_model_output_v1_probability_keys_and_sum_are_exact() -> None:
    output = copy.deepcopy(_model_output())
    output["probabilities"]["game_over"]["atoms"] = "24"

    with pytest.raises(ValidationError, match="sum exactly to 1"):
        contracts.ModelOutputV1.model_validate(output)

    output = copy.deepcopy(_model_output())
    output["probabilities"]["wrong_state"] = output["probabilities"].pop(
        "game_over"
    )
    with pytest.raises(ValidationError, match="exactly match state_space"):
        contracts.ModelOutputV1.model_validate(output)


def test_model_output_contract_is_atomically_v1_only() -> None:
    v1_path = CONTRACTS_ROOT / "model-output" / "v1.schema.yaml"
    assert v1_path.is_file()
    assert not (CONTRACTS_ROOT / "model-output" / "v0.schema.yaml").exists()
    assert not hasattr(contracts, "ModelOutputV0")

    document = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    assert document["contract_version"] == "v1"
    assert document["properties"]["experiment_id"]["pattern"] == "^X-[0-9]{2,}$"
    assert set(document["required"]) == set(_model_output())
