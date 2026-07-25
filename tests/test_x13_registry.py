from __future__ import annotations

from pathlib import Path

import pytest

from prediction_market.experiments import (
    ExperimentRegistryError,
    load_experiment_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_x13_is_preregistered_before_source_time_exploration() -> None:
    registry = load_experiment_registry(PROJECT_ROOT)

    assert set(registry) == {
        *(f"X-{number:02d}" for number in range(1, 14)),
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
