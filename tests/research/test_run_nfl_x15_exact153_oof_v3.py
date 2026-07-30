from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from prediction_market.research.nfl_x15_oof_publication import (
    publish_x15_oof_shard,
)
from prediction_market.research.nfl_x15_selection_batch_v3 import (
    X15SelectionBatchV3Error,
)


_PROJECT_ROOT = Path(__file__).parents[2]
_RUNNER_PATH = (
    _PROJECT_ROOT / "scripts/research/run_nfl_x15_exact153_oof_v3.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "_x15_exact153_oof_v3_runner", _RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)
_discover_existing_shards = _RUNNER._discover_existing_shards
_expected_cells = _RUNNER._expected_cells

_HELPER_PATH = Path(__file__).with_name(
    "test_nfl_x15_oof_publication.py"
)
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_x15_oof_runner_test_helpers_v3", _HELPER_PATH
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)
MAPPING_SHA = _HELPERS.MAPPING_SHA
_authority = _HELPERS._authority
_fold_run = _HELPERS._fold_run


def test_exact45_runner_matrix_has_no_d4() -> None:
    cells = _expected_cells()
    assert len(cells) == 45
    assert all(feature_block_id != "D4" for _, _, feature_block_id in cells)
    assert len(set(cells)) == 45


def test_runner_rejects_external_output_before_prepare_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    project_root.mkdir()
    prepare_calls = 0

    def forbid_prepare(**_: object):
        nonlocal prepare_calls
        prepare_calls += 1
        raise AssertionError("panel prepare must not run")

    monkeypatch.setattr(_RUNNER, "_prepare_with_authority", forbid_prepare)
    dummy_sha = "sha256:" + "0" * 64
    with pytest.raises(X15SelectionBatchV3Error, match="output_root"):
        _RUNNER.main(
            [
                "--project-root",
                str(project_root),
                "--panel-batch",
                "missing-panel.json",
                "--panel-batch-file-sha256",
                dummy_sha,
                "--feature-support-audit",
                "missing-audit.json",
                "--feature-support-audit-file-sha256",
                dummy_sha,
                "--output-root",
                str(external_root),
            ]
        )

    assert prepare_calls == 0
    assert not external_root.exists()


def test_discovery_resumes_six_valid_shards_and_audits_d4(
    tmp_path: Path,
) -> None:
    authority = _authority()
    allowed = (
        ("fold_01", "b0_empirical_v1", "D0"),
        ("fold_01", "regularized_logistic_v1", "D0"),
        ("fold_01", "regularized_logistic_v1", "D1"),
        ("fold_01", "regularized_logistic_v1", "D2"),
        ("fold_01", "regularized_logistic_v1", "D3"),
        ("fold_01", "shallow_xgboost_v1", "D0"),
    )
    for fold_id, model_id, block_id in allowed:
        publish_x15_oof_shard(
            model_run=_fold_run(
                fold_id,
                authority=authority,
                model_id=model_id,
                feature_block_id=block_id,
            ),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )
    for random_state in (20260728, 20260729):
        publish_x15_oof_shard(
            model_run=_fold_run(
                "fold_01",
                authority=authority,
                model_id="regularized_logistic_v1",
                feature_block_id="D4",
                config_override={"random_state": random_state},
            ),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
        )
    events: list[tuple[str, dict[str, object]]] = []

    found = _discover_existing_shards(
        output_root=tmp_path,
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        emit=lambda event, **values: events.append((event, values)),
    )

    assert tuple(found) == allowed
    assert len(found) == 6
    assert events == [
        (
            "unsupported_shard_ignored",
            {
                "fold_id": "fold_01",
                "model_id": "regularized_logistic_v1",
                "feature_block_id": "D4",
                "reason": "UNSUPPORTED_FEATURE_BLOCK",
            },
        ),
        (
            "unsupported_shard_ignored",
            {
                "fold_id": "fold_01",
                "model_id": "regularized_logistic_v1",
                "feature_block_id": "D4",
                "reason": "UNSUPPORTED_FEATURE_BLOCK",
            },
        ),
    ]
