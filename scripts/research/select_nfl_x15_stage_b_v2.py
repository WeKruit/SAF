#!/usr/bin/env python3
"""Select and publish the frozen NFL X-15 Stage-B V2 winner.

This runner consumes only a byte-bound, selection-ready exact-55 OOF batch.
It loads the eight native Polymarket baseline/candidate projections, applies
the frozen winner rule, and atomically publishes one content-addressed JSON
manifest.  It never reads holdout reaction data or fits a model.
"""

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
    ModelSelectionError,
    StageBModelSelectionResult,
    select_stage_b_v2_winner,
    verify_stage_b_v2_selection_result,
)
from prediction_market.research.nfl_x15_models import X15ModelRun
from prediction_market.research.nfl_x15_oof_publication import (
    X15OOFPublicationError,
    load_verified_x15_oof_selection_batch,
    load_x15_oof_selection_projection,
)


DEFAULT_SELECTION_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-b-model-selection-v2"
)


class StageBSelectionRunnerError(RuntimeError):
    """The verified exact-55 batch cannot support final Stage-B selection."""


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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    if selection.winner is None:
        return None
    winner = selection.winner
    return {
        "model_id": winner.spec.candidate_model_id,
        "feature_block_id": winner.spec.candidate_feature_block_id,
        "integrated_mean_improvement": (
            winner.integrated_mean_improvement
        ),
        "integrated_ci_low": winner.integrated_ci_low,
        "integrated_ci_high": winner.integrated_ci_high,
        "anchor_game_count": winner.anchor_game_count,
        "anchor_mean_improvement": winner.anchor_mean_improvement,
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


def run_stage_b_v2_selection(
    *,
    batch_manifest_path: str | Path,
    batch_manifest_file_sha256: str,
    oof_output_root: str | Path,
    selection_output_root: str | Path,
) -> StageBSelectionPublication:
    """Run the reaction-blind Stage-B V2 selection and publish its audit."""

    expected_batch_file_sha256 = _require_sha256(
        batch_manifest_file_sha256,
        field="batch manifest file SHA-256",
    )
    oof_root = Path(oof_output_root).resolve()
    try:
        (
            batch_document,
            resolved_batch_path,
            observed_batch_file_sha256,
            authority,
        ) = load_verified_x15_oof_selection_batch(
            batch_manifest_path=batch_manifest_path,
            output_root=oof_root,
        )
    except X15OOFPublicationError as error:
        raise StageBSelectionRunnerError(str(error)) from error
    if observed_batch_file_sha256 != expected_batch_file_sha256:
        raise StageBSelectionRunnerError(
            "batch manifest file SHA does not match requested input"
        )
    batch_sha256 = _require_sha256(
        batch_document.get("batch_sha256"),
        field="batch batch_sha256",
    )

    runs: list[X15ModelRun] = []
    for candidate in STAGE_B_CANDIDATE_SUITE:
        try:
            model_run = load_x15_oof_selection_projection(
                batch_manifest_path=resolved_batch_path,
                output_root=oof_root,
                candidate_model_id=candidate[0],
                candidate_feature_block_id=candidate[1],
            )
        except X15OOFPublicationError as error:
            raise StageBSelectionRunnerError(str(error)) from error
        runs.append(model_run)
    run_config_sha256s = tuple(
        model_run.run_config_sha256 for model_run in runs
    )

    selection = select_stage_b_v2_winner(
        tuple(runs),
        authority=authority,
    )
    try:
        verify_stage_b_v2_selection_result(selection)
    except ModelSelectionError as error:
        raise StageBSelectionRunnerError(str(error)) from error
    candidate_audit_records = selection.candidate_audit.to_dict(
        "records"
    )
    document = {
        "schema": "nfl_x15_stage_b_v2_selection_manifest_v1",
        "experiment_id": "X-15",
        "cohort": "development",
        "selection_venue": "polymarket",
        "input_oof_batch": {
            "manifest_path": str(resolved_batch_path),
            "manifest_file_sha256": expected_batch_file_sha256,
            "batch_sha256": batch_sha256,
        },
        "candidate_suite": STAGE_B_CANDIDATE_SUITE,
        "candidate_run_config_sha256s": tuple(
            {
                "model_id": candidate[0],
                "feature_block_id": candidate[1],
                "run_config_sha256": run_config_sha256,
            }
            for candidate, run_config_sha256 in zip(
                STAGE_B_CANDIDATE_SUITE,
                run_config_sha256s,
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
        "selection_result_audit_sha256": (
            selection.candidate_audit_sha256
        ),
        "winner_rule_sha256": selection.winner_rule_sha256,
        "schema_version": selection.schema_version,
        "survival_probability_contract": (
            selection.survival_probability_contract
        ),
        "candidate_audit": candidate_audit_records,
        "holdout_reaction_accessed": False,
    }
    payload = _canonical_bytes(document)
    manifest_sha256 = _sha256_bytes(payload)
    digest = manifest_sha256.removeprefix("sha256:")
    manifest_path = (
        Path(selection_output_root).resolve()
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.stage-b-v2-selection.json"
    )
    _atomic_write(manifest_path, payload)
    return StageBSelectionPublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        decision_status=selection.decision_status,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        required=True,
        help="Selection-ready exact-55 OOF batch-index path.",
    )
    parser.add_argument(
        "--batch-manifest-file-sha256",
        required=True,
        help="Byte SHA-256 of the exact batch-index file.",
    )
    parser.add_argument(
        "--oof-output-root",
        type=Path,
        required=True,
        help="Root containing the published exact-55 batch and shards.",
    )
    parser.add_argument(
        "--selection-output-root",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = run_stage_b_v2_selection(
        batch_manifest_path=args.batch_manifest,
        batch_manifest_file_sha256=(
            args.batch_manifest_file_sha256
        ),
        oof_output_root=args.oof_output_root,
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
