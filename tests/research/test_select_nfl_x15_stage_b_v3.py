from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_selection_batch_v3 import (
    publish_x15_selection_batch_v3,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/select_nfl_x15_stage_b_v3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_nfl_x15_stage_b_v3",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

_BATCH_HELPER_PATH = Path(__file__).with_name(
    "test_nfl_x15_selection_batch_v3.py"
)
_BATCH_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_x15_selection_batch_test_helpers",
    _BATCH_HELPER_PATH,
)
assert (
    _BATCH_HELPER_SPEC is not None
    and _BATCH_HELPER_SPEC.loader is not None
)
_BATCH_HELPERS = importlib.util.module_from_spec(_BATCH_HELPER_SPEC)
sys.modules[_BATCH_HELPER_SPEC.name] = _BATCH_HELPERS
_BATCH_HELPER_SPEC.loader.exec_module(_BATCH_HELPERS)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _selection():
    spec = SimpleNamespace(
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D1",
    )
    winner = SimpleNamespace(
        spec=spec,
        integrated_mean_improvement=0.12,
        integrated_ci_low=0.04,
        integrated_ci_high=0.20,
        direction_log_loss_integrated_mean_improvement=0.08,
        direction_log_loss_integrated_ci_low=0.02,
        direction_log_loss_integrated_ci_high=0.14,
        direction_brier_integrated_mean_improvement=0.03,
        direction_brier_integrated_ci_low=0.01,
        direction_brier_integrated_ci_high=0.05,
        anchor_game_count=80,
        anchor_mean_improvement=0.09,
        direction_anchor_game_count=70,
        direction_anchor_log_loss_mean_improvement=0.06,
        direction_anchor_brier_mean_improvement=0.02,
    )
    return SimpleNamespace(
        decision_status="MODEL_ADVANCE",
        winner=winner,
        candidate_audit=pd.DataFrame(
            {"candidate_model_id": ["regularized_logistic_v1"]}
        ),
        best_integrated_mean_improvement=0.12,
        best_standard_error=0.02,
        one_se_threshold=0.10,
        cohort_authority_sha256=SHA_A,
        development_authority_metadata_sha256=SHA_A,
        shared_run_config_sha256=SHA_B,
        suite_run_config_sha256=SHA_C,
        candidate_audit_sha256=SHA_A,
        winner_rule_sha256=SHA_B,
        schema_version="HistoricalTradesOnlyProbabilityPanelV2",
        survival_probability_contract=(
            "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
        ),
        selection_contract_version=(
            "NFL_X15_STAGE_B_MODEL_SELECTION_V3"
        ),
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    authority = SimpleNamespace(metadata_sha256=SHA_C)
    batch_path = Path("/verified/selection.batch-index.json")
    monkeypatch.setattr(
        runner,
        "load_verified_x15_selection_batch_v3",
        lambda *, batch_manifest_path, artifact_root: (
            {
                "batch_sha256": SHA_B,
                "selection_ready": True,
                "holdout_reaction_accessed": False,
            },
            batch_path,
            SHA_A,
            authority,
        ),
    )
    observed: list[tuple[str, str]] = []

    def load_projection(
        *,
        batch_manifest_path,
        artifact_root,
        candidate_model_id,
        candidate_feature_block_id,
    ):
        assert batch_manifest_path == batch_path
        observed.append(
            (candidate_model_id, candidate_feature_block_id)
        )
        return SimpleNamespace(
            run_config_sha256=(
                "sha256:" + f"{len(observed):064x}"
            )
        )

    monkeypatch.setattr(
        runner,
        "load_x15_selection_projection_v3",
        load_projection,
    )
    monkeypatch.setattr(
        runner,
        "select_stage_b_v3_winner",
        lambda runs, *, authority: _selection(),
    )
    return observed


def test_runner_publishes_compact_v3_manifest_without_embedded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _install_fakes(monkeypatch)

    publication = runner.run_stage_b_v3_selection(
        batch_manifest_path=tmp_path / "batch.json",
        batch_manifest_file_sha256=SHA_A,
        artifact_root=tmp_path / "artifacts",
        selection_output_root=tmp_path / "selection",
    )

    document = runner.json.loads(
        publication.manifest_path.read_text()
    )
    assert document["schema"] == (
        "nfl_x15_stage_b_v3_selection_manifest_v1"
    )
    assert tuple(map(tuple, document["candidate_suite"])) == (
        runner.STAGE_B_CANDIDATE_SUITE
    )
    assert observed == list(runner.STAGE_B_CANDIDATE_SUITE)
    assert not any(block == "D4" for _, block in observed)
    assert document["decision_status"] == "MODEL_ADVANCE"
    assert document["candidate_audit_sha256"] == SHA_A
    assert document["authority_metadata_sha256"] == SHA_C
    assert "candidate_audit" not in document
    assert "candidate_results" not in document
    assert document["holdout_reaction_accessed"] is False
    assert publication.manifest_sha256 == runner._sha256_bytes(
        publication.manifest_path.read_bytes()
    )


def test_runner_rejects_unbound_batch_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)

    with pytest.raises(
        runner.StageBSelectionRunnerError,
        match="file SHA",
    ):
        runner.run_stage_b_v3_selection(
            batch_manifest_path=tmp_path / "batch.json",
            batch_manifest_file_sha256=SHA_B,
            artifact_root=tmp_path / "artifacts",
            selection_output_root=tmp_path / "selection",
        )


def test_exact45_batch_to_compact_selection_manifest_end_to_end(
    tmp_path: Path,
) -> None:
    paths = _BATCH_HELPERS._publish_shards(
        tmp_path,
        _BATCH_HELPERS.EXACT_SELECTION_CELLS_V3,
    )
    (
        panel_path,
        panel_sha,
        audit_path,
        audit_sha,
    ) = _BATCH_HELPERS._publish_panel_and_support_audit(tmp_path)
    batch = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_BATCH_HELPERS._authority(),
        cohort_mapping_sha256=_BATCH_HELPERS.MAPPING_SHA,
        output_root=tmp_path,
        project_root=tmp_path,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )

    publication = runner.run_stage_b_v3_selection(
        batch_manifest_path=batch.manifest_path,
        batch_manifest_file_sha256=batch.manifest_sha256,
        artifact_root=tmp_path,
        selection_output_root=tmp_path / "selection-result",
    )

    document = runner.json.loads(
        publication.manifest_path.read_text()
    )
    assert document["schema"] == (
        "nfl_x15_stage_b_v3_selection_manifest_v1"
    )
    assert len(document["candidate_run_config_sha256s"]) == 6
    assert document["holdout_reaction_accessed"] is False
