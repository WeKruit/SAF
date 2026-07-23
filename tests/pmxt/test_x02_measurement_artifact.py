from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prediction_market.experiments import load_experiment_registry
from prediction_market.program_audit import load_artifact_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts/data-audit/x02_timestamp_sample_measurement_v1.json"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


def test_x02_measurement_artifact_and_report_hashes_recompute() -> None:
    artifact = _artifact()
    artifact_hash = artifact.pop("artifact_sha256")
    assert artifact_hash == _canonical_sha256(artifact)

    report = dict(artifact["report"])
    report_hash = report.pop("report_sha256")
    assert report_hash == _canonical_sha256(report)


def test_x02_measurement_is_bound_to_seq3_without_formal_acceptance() -> None:
    artifact = _artifact()
    x02 = load_experiment_registry(PROJECT_ROOT)["X-02"]
    registered = x02["preregistered_inputs"]["formal_result"]

    assert artifact["registration"] == {
        "registration_head_sha256": x02["registration_head_sha256"],
        "preregistered_code_sha256": registered["code_sha256"],
        "preregistered_data_sha256": registered["data_sha256"],
        "dataset_ids": registered["dataset_ids"],
        "model_ids": registered["model_ids"],
    }
    assert artifact["report"]["code_sha256"] == registered["code_sha256"]
    assert artifact["report"]["data_sha256"] == registered["data_sha256"]
    assert x02["results_ref"] == []
    assert x02["status"] == "registered"
    assert artifact["is_formal_result"] is False
    assert artifact["formal_result_eligible"] is False
    assert artifact["acceptance"]["results_ref_appended"] is False
    assert artifact["acceptance"]["blocked_by_experiments"] == ["X-01"]


def test_x02_measurement_records_exact_scope_and_downgrade_decision() -> None:
    artifact = _artifact()
    report = artifact["report"]

    assert report["days"] == [
        "2026-04-22",
        "2026-05-28",
        "2026-06-05",
        "2026-06-25",
    ]
    assert report["object_count"] == 96
    assert report["row_count"] == report["delta_count"] == 6_549_816_634
    assert report["quantiles_ms"] == {
        "p50": 177.0,
        "p95": 159_161.0,
        "p99": 279_725.0,
    }
    assert report["absolute_p99_ms"] == 279_725.0
    assert report["negative_delta_count"] == 0
    assert report["out_of_order_count"] == 1_257_606
    assert report["millisecond_research_eligible"] is False
    assert report["downgrade_triggers"] == ["absolute_p99_ms_gt_5000"]
    assert artifact["interpretation"]["required_research_clock_granularity"] == (
        "seconds"
    )


def test_x02_measurement_artifact_registry_keeps_dependency_open() -> None:
    rows = load_artifact_registry(PROJECT_ROOT)
    row = next(item for item in rows if item.artifact_id == "ART-C-020")

    assert row.path == (
        "artifacts/data-audit/x02_timestamp_sample_measurement_v1.json"
    )
    assert row.owner_team == "C+H"
    assert row.version == "v1"
    assert row.status == "in_progress"
