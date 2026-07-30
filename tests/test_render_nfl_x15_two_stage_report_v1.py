from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/research/render_nfl_x15_two_stage_report_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_nfl_x15_two_stage_report_v1",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _canonical_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode()


def _write_content_addressed_json(
    root: Path,
    document: dict[str, object],
    *,
    suffix: str,
) -> Path:
    payload = _canonical_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "manifests" / "sha256" / digest[:2] / f"{digest}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_content_addressed_parquet(
    root: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    temporary = root / "calibration.parquet"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    payload = temporary.read_bytes()
    temporary.unlink()
    digest = hashlib.sha256(payload).hexdigest()
    relative = (
        Path("batches")
        / "objects"
        / "sha256"
        / digest[:2]
        / f"{digest}.parquet"
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "name": "calibration_summary",
        "object_path": str(relative),
        "object_sha256": f"sha256:{digest}",
        "byte_length": len(payload),
        "row_count": len(rows),
        "schema_columns": list(rows[0]),
    }


def _factor_manifest() -> dict[str, object]:
    return {
        "schema": "nfl_x13_exact153_fact_batch_index_v4",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": 153,
        "publication_gate": "PASS",
        "holdout_reaction_accessed": False,
        "market_data_read": False,
        "factor_registry": {
            "version": "v4",
            "factor_count": 59,
            "file_sha256": f"sha256:{'a' * 64}",
        },
        "aggregate_counts": {
            "raw_pbp_rows": 26192,
            "canonical_factor_events_rows": 26192,
            "factor_hits_rows": 83659,
            "factor_coverage_audit_rows": 9027,
            "raw_rows_silently_dropped": 0,
        },
        "claim_boundary": "DEVELOPMENT_SPORTS_FACTS_ONLY",
    }


def _stage_a_manifest(stage_root: Path) -> dict[str, object]:
    table = _write_content_addressed_parquet(
        stage_root,
        [
            {
                "evaluated_games": 153,
                "evaluated_states": 21641,
                "brier": 0.158593,
                "log_loss": 0.470834,
                "calibration_slope": 1.091129,
                "calibration_intercept": -0.021436,
            }
        ],
    )
    return {
        "schema": "nfl_x15_stage_a_batch_index_v1",
        "experiment_id": "X-15",
        "cohort": "development",
        "game_count": 153,
        "publication_gate": "PASS",
        "holdout_reaction_accessed": False,
        "market_data_read": False,
        "model": {
            "model_id": "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1",
            "model_version": "fixture-v1",
            "model_sha256": f"sha256:{'b' * 64}",
        },
        "calibration_tables": [table],
        "claim_boundary": "RETROSPECTIVE_DIAGNOSTIC",
    }


def _exact45_manifest(*, complete: bool) -> dict[str, object]:
    designs = [
        ("b0_empirical_v1", "D0"),
        ("regularized_logistic_v1", "D0"),
        ("regularized_logistic_v1", "D1"),
        ("regularized_logistic_v1", "D2"),
        ("regularized_logistic_v1", "D3"),
        ("shallow_xgboost_v1", "D0"),
        ("shallow_xgboost_v1", "D1"),
        ("shallow_xgboost_v1", "D2"),
        ("shallow_xgboost_v1", "D3"),
    ]
    cells = [
        [f"fold_{fold:02d}", model, block]
        for fold in range(1, 6)
        for model, block in designs
    ]
    observed = cells if complete else cells[:2]
    missing = [] if complete else cells[2:]
    return {
        "schema": "nfl_x15_selection_batch_v3",
        "experiment_id": "X-15",
        "cohort": "development",
        "authority_game_count": 153,
        "authority_metadata_sha256": f"sha256:{'c' * 64}",
        "cohort_authority_sha256": f"sha256:{'d' * 64}",
        "expected_execution_cell_count": len(cells),
        "expected_execution_cells": cells,
        "observed_execution_cell_count": len(observed),
        "observed_execution_cells": observed,
        "missing_execution_cell_count": len(missing),
        "missing_execution_cells": missing,
        "selection_ready": complete,
        "publication_status": (
            "PASS" if complete else "PARTIAL_DIAGNOSTIC_ONLY"
        ),
        "holdout_reaction_accessed": False,
        "unsupported_feature_blocks": {
            "D4": "UNSUPPORTED_FEATURE_BLOCK"
        },
        "selectable_candidates": [
            ["regularized_logistic_v1", "D1"]
        ],
        "batch_sha256": f"sha256:{'e' * 64}",
    }


def _inputs(tmp_path: Path, *, exact_complete: bool = False) -> dict[str, Path]:
    factor = _write_content_addressed_json(
        tmp_path / "factor",
        _factor_manifest(),
        suffix="batch-index.json",
    )
    stage_root = tmp_path / "stage-a"
    stage_a = _write_content_addressed_json(
        stage_root / "batches",
        _stage_a_manifest(stage_root),
        suffix="batch-index.json",
    )
    exact45 = _write_content_addressed_json(
        tmp_path / "exact45",
        _exact45_manifest(complete=exact_complete),
        suffix="batch-index.json",
    )
    return {"factor": factor, "stage_a": stage_a, "exact45": exact45}


def test_pending_report_uses_only_real_completion_and_calibration_rows(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)

    publication = report.render_two_stage_report(
        factor_v4_manifest_path=paths["factor"],
        stage_a_manifest_path=paths["stage_a"],
        exact45_manifest_path=paths["exact45"],
        development_gate="OPEN",
        development_blockers=(),
        output_root=tmp_path / "reports",
    )

    html = publication.html_path.read_text()
    assert publication.html_sha256 == (
        "sha256:" + hashlib.sha256(publication.html_path.read_bytes()).hexdigest()
    )
    assert publication.html_path.name.startswith(
        publication.html_sha256.removeprefix("sha256:")
    )
    assert "NFL exact-153 两阶段研究状态报告" in html
    assert "153 场" in html
    assert "26,192" in html
    assert "83,659" in html
    assert "0.158593" in html
    assert "0.470834" in html
    assert "2 / 45" in html
    assert "PARTIAL_DIAGNOSTIC_ONLY" in html
    assert "selection result 尚未发布" in html
    assert "D4" in html and "UNSUPPORTED_FEATURE_BLOCK" in html
    assert "离线报告" in html
    assert "不存在的方向指标" not in html
    assert paths["factor"].as_posix() in html
    assert publication.manifest_path.exists()


def test_report_rejects_tampered_content_addressed_manifest(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    paths["factor"].write_text("{}")

    with pytest.raises(report.TwoStageReportError, match="SHA-256"):
        report.render_two_stage_report(
            factor_v4_manifest_path=paths["factor"],
            stage_a_manifest_path=paths["stage_a"],
            exact45_manifest_path=paths["exact45"],
            development_gate="OPEN",
            development_blockers=(),
            output_root=tmp_path / "reports",
        )


def test_report_rejects_non_exact45_execution_authority(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    invalid = _exact45_manifest(complete=True)
    invalid["expected_execution_cells"] = invalid[
        "expected_execution_cells"
    ][:4]
    invalid["observed_execution_cells"] = invalid[
        "observed_execution_cells"
    ][:4]
    invalid["expected_execution_cell_count"] = 4
    invalid["observed_execution_cell_count"] = 4
    invalid["missing_execution_cells"] = []
    invalid["missing_execution_cell_count"] = 0
    invalid_path = _write_content_addressed_json(
        tmp_path / "invalid-exact45",
        invalid,
        suffix="batch-index.json",
    )

    with pytest.raises(report.TwoStageReportError, match="45"):
        report.render_two_stage_report(
            factor_v4_manifest_path=paths["factor"],
            stage_a_manifest_path=paths["stage_a"],
            exact45_manifest_path=invalid_path,
            development_gate="OPEN",
            development_blockers=(),
            output_root=tmp_path / "reports",
        )


def test_completed_report_shows_real_selection_kalshi_and_magnitude_values(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path, exact_complete=True)
    exact45_sha = (
        "sha256:" + hashlib.sha256(paths["exact45"].read_bytes()).hexdigest()
    )
    selection = _write_content_addressed_json(
        tmp_path / "selection",
        {
            "schema": "nfl_x15_stage_b_v3_selection_manifest_v1",
            "experiment_id": "X-15",
            "cohort": "development",
            "selection_venue": "polymarket",
            "input_selection_batch": {
                "manifest_path": str(paths["exact45"].resolve()),
                "manifest_file_sha256": exact45_sha,
                "batch_sha256": f"sha256:{'e' * 64}",
            },
            "decision_status": "MODEL_ADVANCE",
            "winner": {
                "model_id": "regularized_logistic_v1",
                "feature_block_id": "D1",
                "joint_integrated_mean_improvement": 0.0123,
                "joint_integrated_ci": [0.004, 0.020],
                "direction_log_loss_integrated_mean_improvement": 0.0061,
                "direction_log_loss_integrated_ci": [0.001, 0.011],
                "direction_brier_integrated_mean_improvement": 0.0024,
                "direction_brier_integrated_ci": [0.0004, 0.004],
                "joint_anchor_game_count": 41,
                "joint_anchor_mean_improvement": 0.008,
            },
            "holdout_reaction_accessed": False,
            "claim_boundary": "HISTORICAL_TRADES_ONLY",
        },
        suffix="stage-b-v3-selection.json",
    )
    selection_sha = (
        "sha256:" + hashlib.sha256(selection.read_bytes()).hexdigest()
    )
    kalshi = _write_content_addressed_json(
        tmp_path / "kalshi",
        {
            "schema": "nfl_x16_kalshi_development_validation_v1",
            "experiment_id": "X-16",
            "cohort": "development",
            "selection_manifest_file_sha256": selection_sha,
            "candidate_model_id": "regularized_logistic_v1",
            "candidate_feature_block_id": "D1",
            "paired_game_count": 37,
            "paired_identity_count": 1200,
            "transport_b0_mean_improvement": 0.003,
            "transport_b0_ci_low": 0.0002,
            "transport_b0_ci_high": 0.0058,
            "transport_gate_passed": True,
            "holdout_reaction_accessed": False,
            "claim_boundary": "DEVELOPMENT_TRANSPORT_DIAGNOSTIC",
        },
        suffix="kalshi-validation.json",
    )
    magnitude = _write_content_addressed_json(
        tmp_path / "magnitude",
        {
            "schema": "nfl_x15_conditional_magnitude_result_v1",
            "experiment_id": "X-15",
            "cohort": "development",
            "selection_manifest_file_sha256": selection_sha,
            "status": "PASS",
            "metrics": [
                {
                    "direction": "UP",
                    "quantile": "q50",
                    "estimate": 0.021,
                    "pinball_loss": 0.008,
                    "support_games": 44,
                },
                {
                    "direction": "DOWN",
                    "quantile": "q50",
                    "estimate": 0.018,
                    "pinball_loss": 0.007,
                    "support_games": 42,
                },
            ],
            "holdout_reaction_accessed": False,
            "claim_boundary": "CONDITIONAL_MAGNITUDE_DIAGNOSTIC",
        },
        suffix="magnitude.json",
    )

    publication = report.render_two_stage_report(
        factor_v4_manifest_path=paths["factor"],
        stage_a_manifest_path=paths["stage_a"],
        exact45_manifest_path=paths["exact45"],
        selection_manifest_path=selection,
        kalshi_manifest_path=kalshi,
        magnitude_manifest_path=magnitude,
        development_gate="OPEN",
        development_blockers=(),
        output_root=tmp_path / "reports",
    )

    html = publication.html_path.read_text()
    assert "MODEL_ADVANCE" in html
    assert "regularized_logistic_v1" in html
    assert "D1" in html
    assert "0.012300" in html
    assert "0.006100" in html
    assert "37" in html
    assert "TRANSPORT PASS" in html
    assert "q50" in html
    assert "0.021000" in html
    assert "0.018000" in html


def test_no_model_advance_makes_downstream_validation_not_applicable(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path, exact_complete=True)
    exact45_sha = (
        "sha256:" + hashlib.sha256(paths["exact45"].read_bytes()).hexdigest()
    )
    selection = _write_content_addressed_json(
        tmp_path / "selection",
        {
            "schema": "nfl_x15_stage_b_v3_selection_manifest_v1",
            "experiment_id": "X-15",
            "cohort": "development",
            "selection_venue": "polymarket",
            "input_selection_batch": {
                "manifest_path": str(paths["exact45"].resolve()),
                "manifest_file_sha256": exact45_sha,
                "batch_sha256": f"sha256:{'e' * 64}",
            },
            "decision_status": "NO_MODEL_ADVANCE",
            "winner": None,
            "holdout_reaction_accessed": False,
        },
        suffix="stage-b-v3-selection.json",
    )

    publication = report.render_two_stage_report(
        factor_v4_manifest_path=paths["factor"],
        stage_a_manifest_path=paths["stage_a"],
        exact45_manifest_path=paths["exact45"],
        selection_manifest_path=selection,
        development_gate="OPEN",
        development_blockers=(),
        output_root=tmp_path / "reports",
    )

    html = publication.html_path.read_text()
    assert "NO_MODEL_ADVANCE" in html
    assert "Kalshi 验证：NOT_APPLICABLE" in html
    assert "Magnitude：NOT_APPLICABLE" in html


def test_development_blockers_override_complete_selection_and_kalshi_pass(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path, exact_complete=True)
    exact45_sha = (
        "sha256:" + hashlib.sha256(paths["exact45"].read_bytes()).hexdigest()
    )
    selection = _write_content_addressed_json(
        tmp_path / "selection",
        {
            "schema": "nfl_x15_stage_b_v3_selection_manifest_v1",
            "experiment_id": "X-15",
            "cohort": "development",
            "selection_venue": "polymarket",
            "input_selection_batch": {
                "manifest_path": str(paths["exact45"].resolve()),
                "manifest_file_sha256": exact45_sha,
                "batch_sha256": f"sha256:{'e' * 64}",
            },
            "decision_status": "MODEL_ADVANCE",
            "winner": {
                "model_id": "regularized_logistic_v1",
                "feature_block_id": "D1",
                "joint_integrated_mean_improvement": 0.0123,
                "joint_integrated_ci": [0.004, 0.020],
            },
            "holdout_reaction_accessed": False,
        },
        suffix="stage-b-v3-selection.json",
    )
    selection_sha = (
        "sha256:" + hashlib.sha256(selection.read_bytes()).hexdigest()
    )
    kalshi = _write_content_addressed_json(
        tmp_path / "kalshi",
        {
            "schema": "nfl_x16_kalshi_development_validation_v1",
            "experiment_id": "X-16",
            "cohort": "development",
            "selection_manifest_file_sha256": selection_sha,
            "transport_gate_passed": True,
            "paired_game_count": 37,
            "holdout_reaction_accessed": False,
        },
        suffix="kalshi-validation.json",
    )

    publication = report.render_two_stage_report(
        factor_v4_manifest_path=paths["factor"],
        stage_a_manifest_path=paths["stage_a"],
        exact45_manifest_path=paths["exact45"],
        selection_manifest_path=selection,
        kalshi_manifest_path=kalshi,
        development_gate="BLOCKED",
        development_blockers=(
            "OT_SCOPE_CONTAMINATION",
            "PIT_PRESTATE_REFERENCE_NOT_PROPAGATED",
        ),
        output_root=tmp_path / "reports",
    )

    html = publication.html_path.read_text()
    manifest = json.loads(publication.manifest_path.read_text())
    assert "Development Gate：BLOCKED" in html
    assert "OT_SCOPE_CONTAMINATION" in html
    assert "PIT_PRESTATE_REFERENCE_NOT_PROPAGATED" in html
    assert "Selection promotion gate：BLOCKED" in html
    assert "Kalshi gate：BLOCKED" in html
    assert "Holdout gate：BLOCKED" in html
    assert "Kalshi：TRANSPORT PASS" not in html
    assert manifest["development_gate"] == "BLOCKED"
    assert manifest["stage_status"]["selection"] == "BLOCKED"
    assert manifest["stage_status"]["kalshi"] == "BLOCKED"
    assert manifest["stage_status"]["holdout"] == "BLOCKED"
    assert manifest["diagnostic_status"]["selection"] == "MODEL_ADVANCE"
    assert manifest["diagnostic_status"]["kalshi"] == "PUBLISHED"


@pytest.mark.parametrize(
    ("gate", "blockers"),
    [
        ("OPEN", ("OT_SCOPE_CONTAMINATION",)),
        ("BLOCKED", ()),
        ("UNKNOWN", ()),
    ],
)
def test_development_gate_and_blockers_must_be_consistent(
    tmp_path: Path,
    gate: str,
    blockers: tuple[str, ...],
) -> None:
    paths = _inputs(tmp_path)

    with pytest.raises(report.TwoStageReportError, match="development"):
        report.render_two_stage_report(
            factor_v4_manifest_path=paths["factor"],
            stage_a_manifest_path=paths["stage_a"],
            exact45_manifest_path=paths["exact45"],
            development_gate=gate,
            development_blockers=blockers,
            output_root=tmp_path / "reports",
        )
