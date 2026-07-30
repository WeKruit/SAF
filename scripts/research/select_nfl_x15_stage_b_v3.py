#!/usr/bin/env python3
"""Select and publish the frozen NFL X-15 Stage-B V3 decision."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import NamedTuple

import numpy as np
import pandas as pd

from prediction_market.research.nfl_x15_model_selection import (
    STAGE_B_CANDIDATE_SUITE,
    StageBModelSelectionResult,
    select_stage_b_v3_winner,
)
from prediction_market.research.nfl_x15_models import X15ModelRun
from prediction_market.research.nfl_x15_selection_batch_v3 import (
    X15SelectionBatchV3Error,
    load_verified_x15_selection_batch_v3,
    load_x15_selection_projection_v3,
)


DEFAULT_SELECTION_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-b-model-selection-v3"
)


class StageBSelectionRunnerError(RuntimeError):
    """The verified exact-45 batch cannot support Stage-B V3 selection."""


class StageBSelectionPublication(NamedTuple):
    manifest_path: Path
    manifest_sha256: str
    decision_status: str


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(child)
            for key, child in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        children = list(value)
        if isinstance(value, set):
            children = sorted(children, key=repr)
        return [_canonical(child) for child in children]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return str(value)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in value[7:]
        )
    ):
        raise StageBSelectionRunnerError(
            f"{field} must be a sha256 digest"
        )
    return value


def _winner_document(
    selection: StageBModelSelectionResult,
) -> dict[str, object] | None:
    winner = selection.winner
    if winner is None:
        return None
    return {
        "model_id": winner.spec.candidate_model_id,
        "feature_block_id": winner.spec.candidate_feature_block_id,
        "joint_integrated_mean_improvement": (
            winner.integrated_mean_improvement
        ),
        "joint_integrated_ci": (
            winner.integrated_ci_low,
            winner.integrated_ci_high,
        ),
        "direction_log_loss_integrated_mean_improvement": (
            winner.direction_log_loss_integrated_mean_improvement
        ),
        "direction_log_loss_integrated_ci": (
            winner.direction_log_loss_integrated_ci_low,
            winner.direction_log_loss_integrated_ci_high,
        ),
        "direction_brier_integrated_mean_improvement": (
            winner.direction_brier_integrated_mean_improvement
        ),
        "direction_brier_integrated_ci": (
            winner.direction_brier_integrated_ci_low,
            winner.direction_brier_integrated_ci_high,
        ),
        "joint_anchor_game_count": winner.anchor_game_count,
        "joint_anchor_mean_improvement": (
            winner.anchor_mean_improvement
        ),
        "direction_anchor_game_count": (
            winner.direction_anchor_game_count
        ),
        "direction_anchor_log_loss_mean_improvement": (
            winner.direction_anchor_log_loss_mean_improvement
        ),
        "direction_anchor_brier_mean_improvement": (
            winner.direction_anchor_brier_mean_improvement
        ),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise StageBSelectionRunnerError(
                f"content-addressed collision: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_stage_b_v3_selection(
    *,
    batch_manifest_path: str | Path,
    batch_manifest_file_sha256: str,
    artifact_root: str | Path,
    selection_output_root: str | Path,
) -> StageBSelectionPublication:
    """Run reaction-blind V3 selection over one verified exact-45 batch."""

    expected_file_sha = _require_sha256(
        batch_manifest_file_sha256,
        field="batch manifest file SHA-256",
    )
    try:
        (
            batch_document,
            resolved_batch_path,
            observed_file_sha,
            authority,
        ) = load_verified_x15_selection_batch_v3(
            batch_manifest_path=batch_manifest_path,
            artifact_root=artifact_root,
        )
    except X15SelectionBatchV3Error as error:
        raise StageBSelectionRunnerError(str(error)) from error
    if observed_file_sha != expected_file_sha:
        raise StageBSelectionRunnerError(
            "batch manifest file SHA does not match requested input"
        )
    if batch_document.get("holdout_reaction_accessed") is not False:
        raise StageBSelectionRunnerError(
            "selection batch must remain holdout-reaction blind"
        )
    batch_sha256 = _require_sha256(
        batch_document.get("batch_sha256"),
        field="batch batch_sha256",
    )

    runs: list[X15ModelRun] = []
    for model_id, feature_block_id in STAGE_B_CANDIDATE_SUITE:
        try:
            run = load_x15_selection_projection_v3(
                batch_manifest_path=resolved_batch_path,
                artifact_root=artifact_root,
                candidate_model_id=model_id,
                candidate_feature_block_id=feature_block_id,
            )
        except X15SelectionBatchV3Error as error:
            raise StageBSelectionRunnerError(str(error)) from error
        runs.append(run)
    selection = select_stage_b_v3_winner(
        tuple(runs),
        authority=authority,
    )
    document = {
        "schema": "nfl_x15_stage_b_v3_selection_manifest_v1",
        "experiment_id": "X-15",
        "cohort": "development",
        "selection_venue": "polymarket",
        "input_selection_batch": {
            "manifest_path": str(resolved_batch_path),
            "manifest_file_sha256": expected_file_sha,
            "batch_sha256": batch_sha256,
        },
        "candidate_suite": STAGE_B_CANDIDATE_SUITE,
        "candidate_run_config_sha256s": tuple(
            {
                "model_id": model_id,
                "feature_block_id": feature_block_id,
                "run_config_sha256": run.run_config_sha256,
            }
            for (model_id, feature_block_id), run in zip(
                STAGE_B_CANDIDATE_SUITE,
                runs,
                strict=True,
            )
        ),
        "decision_status": selection.decision_status,
        "winner": _winner_document(selection),
        "best_integrated_mean_improvement": (
            selection.best_integrated_mean_improvement
        ),
        "best_standard_error": selection.best_standard_error,
        "one_se_threshold": selection.one_se_threshold,
        "cohort_authority_sha256": (
            selection.cohort_authority_sha256
        ),
        "authority_metadata_sha256": authority.metadata_sha256,
        "shared_run_config_sha256": (
            selection.shared_run_config_sha256
        ),
        "suite_run_config_sha256": (
            selection.suite_run_config_sha256
        ),
        "candidate_audit_sha256": (
            selection.candidate_audit_sha256
        ),
        "winner_rule_sha256": selection.winner_rule_sha256,
        "selection_contract_version": (
            selection.selection_contract_version
        ),
        "schema_version": selection.schema_version,
        "survival_probability_contract": (
            selection.survival_probability_contract
        ),
        "holdout_reaction_accessed": False,
    }
    payload = _canonical_bytes(document)
    manifest_sha256 = _sha256_bytes(payload)
    digest = manifest_sha256.removeprefix("sha256:")
    path = (
        Path(selection_output_root).resolve()
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.stage-b-v3-selection.json"
    )
    _atomic_write(path, payload)
    return StageBSelectionPublication(
        manifest_path=path,
        manifest_sha256=manifest_sha256,
        decision_status=selection.decision_status,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument(
        "--batch-manifest-file-sha256",
        required=True,
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--selection-output-root",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = run_stage_b_v3_selection(
        batch_manifest_path=args.batch_manifest,
        batch_manifest_file_sha256=(
            args.batch_manifest_file_sha256
        ),
        artifact_root=args.artifact_root,
        selection_output_root=args.selection_output_root,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(publication.manifest_path),
                "manifest_sha256": publication.manifest_sha256,
                "decision_status": publication.decision_status,
                "holdout_reaction_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
