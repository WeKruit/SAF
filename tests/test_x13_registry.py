from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prediction_market.experiments import (
    ExperimentRegistryError,
    load_experiment_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_x13_is_preregistered_before_source_time_exploration() -> None:
    registry = load_experiment_registry(PROJECT_ROOT)

    assert set(registry) == {
        *(f"X-{number:02d}" for number in range(1, 15)),
    }
    x13 = registry["X-13"]
    assert x13["name"] == (
        "NFL 20-game source-time event and historical market exploration"
    )
    assert x13["owner_team"] == "C+D2+E+H"
    assert x13["status"] == "registered"
    assert x13["execution_authorized"] is True
    assert x13["completion_required_scopes"] == [
        "preliminary_source_time_only"
    ]
    assert x13["dataset_ids"] == [
        "DS-KALSHI-HISTORICAL",
        "DS-NFLVERSE",
        "DS-NFLVERSE-PARTICIPATION",
        "DS-POLYMARKET-PUBLIC",
    ]
    scope = x13["authorization_scopes"]["preliminary_source_time_only"]
    assert scope["authorized"] is True
    assert scope["required_result_label"] == "PRELIMINARY"
    assert scope["input_binding"]["model_ids"] == []
    assert set(scope["required_lock_ids"]) == {
        "analysis_whitelist",
        "contract_universe",
        "game_sample_and_bindings",
        "h_split_approval",
        "source_manifest_bundle",
        "source_time_interval_semantics",
    }
    assert x13["source_time_only"] is True
    assert x13["causal_or_execution_claims_authorized"] is False


def test_x13_resolved_source_bundle_and_unknown_registry_fail_closed(
    tmp_path: Path,
) -> None:
    registry = load_experiment_registry(PROJECT_ROOT)
    x13 = registry["X-13"]

    unresolved = {
        lock["id"]
        for lock in x13["registration_locks"]
        if lock["status"] == "unresolved"
    }
    assert unresolved == set()
    source_manifest_lock = next(
        lock
        for lock in x13["registration_locks"]
        if lock["id"] == "source_manifest_bundle"
    )
    assert source_manifest_lock["evidence_ref"] == (
        "sha256:"
        "aa273e66d8cbfda757cd0e63e4dc00d7087035d0c4bbd723506f7c8958e8f19e"
    )

    with pytest.raises(ExperimentRegistryError):
        load_experiment_registry(tmp_path)


def test_x13_holdout_is_appended_without_rewriting_registration_history() -> None:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "registries/experiments/X-13.yaml").read_text()
    )["experiment_card"]
    assert raw["registration_record_sha256"] == (
        "sha256:"
        "f637a647e881a048866e0dd4003b65b4397e6194cf6e58cc13c65be69920a697"
    )
    assert [
        amendment["amendment_sha256"]
        for amendment in raw["amendments"][:3]
    ] == [
        "sha256:f2c54c5df05a27d1d6500598bc81d4194f8875dfb82b4219a1ab1991dd9ff133",
        "sha256:b160b856da5c1f8115969bbbbda6418fe67fc84e0cffdb121f0ea9a677f1ac2f",
        "sha256:a652026456ddc2ea7ce1b2dc84891908e77af645d87589585976820905016838",
    ]
    assert "historical_holdout_selector" not in {
        key
        for key in raw
        if key != "amendments"
    }

    effective = load_experiment_registry(PROJECT_ROOT)["X-13"]
    selector = effective["historical_holdout_selector"]
    assert selector["frozen_before_reaction_access"] is True
    assert selector["selection_id"] == (
        "sha256:"
        "6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
    )
