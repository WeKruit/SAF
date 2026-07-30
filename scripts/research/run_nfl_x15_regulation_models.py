#!/usr/bin/env python3
"""Verified, resumable NFL X-15 regulation-only Stage-B runner.

The default path verifies only the immutable PanelV4 manifest graph.  Model
fitting requires ``--execute`` and always publishes one atomic execution-cell
shard at a time.  Existing shards are byte- and semantics-verified before they
can be resumed; a partial 45-cell matrix remains selection-blocked.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import re
import resource
import subprocess
import time
from typing import Sequence

from prediction_market.research.nfl_x15_regulation_models import (
    CORRECTED_MODEL_CELLS,
    FEATURE_BLOCKS,
    POLYMARKET,
    PRESTATE_CONTEXT_CONTRACT,
    PRIMARY_ANALYSIS_ROLE,
    PRIMARY_ANCHOR_KIND,
    PRIMARY_FEATURE_CONTRACT,
    REQUIRED_PANEL_COLUMNS,
    SCHEMA_VERSION,
    discover_verified_regulation_oof_shards,
    load_verified_regulation_panel_batch,
    publish_regulation_oof_batch_index,
    publish_regulation_oof_shard,
    publish_regulation_selection_lock,
    regulation_oof_batch_status,
    run_regulation_walk_forward,
    verify_regulation_panel_batch_manifest,
    verify_regulation_oof_shard,
    verify_regulation_selection_lock,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-b-regulation-oof-v1"
)
EXPECTED_GAME_COUNT = 153
PEAK_RSS_WARNING_MIB = 6 * 1024
PEAK_RSS_STOP_MIB = 8 * 1024
SYSTEM_FREE_STOP_PERCENT = 10


def _peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return (
        rss / (1024 * 1024)
        if platform.system() == "Darwin"
        else rss / 1024
    )


def _system_memory_free_percent() -> int | None:
    try:
        completed = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%",
        completed.stdout,
    )
    return int(match.group(1)) if match else None


def _memory_gate_snapshot(
    *,
    stage: str,
    started_at: float,
    now: float | None = None,
) -> dict[str, object]:
    peak_rss = _peak_rss_mib()
    free_percent = _system_memory_free_percent()
    if peak_rss >= PEAK_RSS_STOP_MIB:
        raise MemoryError("model runner peak RSS reached 8 GiB stop gate")
    if (
        free_percent is not None
        and free_percent <= SYSTEM_FREE_STOP_PERCENT
    ):
        raise MemoryError("system free memory reached 10% stop gate")
    current = time.perf_counter() if now is None else now
    return {
        "stage": stage,
        "status": (
            "RSS_WARNING"
            if peak_rss >= PEAK_RSS_WARNING_MIB
            else "PASS"
        ),
        "elapsed_seconds": round(current - started_at, 6),
        "peak_rss_mib": round(peak_rss, 3),
        "system_memory_free_percent": free_percent,
    }


def _contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_columns": REQUIRED_PANEL_COLUMNS,
        "primary_feature_contract": dict(PRIMARY_FEATURE_CONTRACT),
        "prestate_context_contract": dict(PRESTATE_CONTEXT_CONTRACT),
        "feature_blocks": dict(FEATURE_BLOCKS),
        "execution_cell_count": len(CORRECTED_MODEL_CELLS),
        "execution_cells": CORRECTED_MODEL_CELLS,
        "selection_venue": POLYMARKET,
        "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
        "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
        "cross_anchor_pooling_allowed": False,
        "kalshi_selection_allowed": False,
        "holdout_reaction_accessed": False,
        "targets": (
            "availability_h",
            "direction_h_given_availability",
        ),
        "joint_loss": (
            "NLL(availability_h)+"
            "I(availability_h)*NLL(direction_h)"
        ),
        "censored_rows": "EXCLUDED_FROM_BOTH_HEADS",
        "weighting": "ROW_TO_INFORMATION_ANCHOR_TO_GAME",
        "b0_calibration": "NONE; FIT_ALL_OUTER_TRAIN_WEEKS",
        "learned_model_calibration": (
            "BASE_FIT_BEFORE_LAST_OUTER_TRAIN_WEEK; "
            "CALIBRATE_ON_LAST_OUTER_TRAIN_WEEK; "
            "NO_POST_CALIBRATION_REFIT"
        ),
        "publication": "ATOMIC_ONE_CELL_SHARDS_WITH_FAIL_CLOSED_RESUME",
    }


def _cell(value: str) -> tuple[str, str, str]:
    parts = tuple(part.strip() for part in value.split(":"))
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "cell must be FOLD_ID:MODEL_ID:FEATURE_BLOCK_ID"
        )
    cell = (parts[0], parts[1], parts[2])
    if cell not in CORRECTED_MODEL_CELLS:
        raise argparse.ArgumentTypeError(
            f"cell is outside the corrected 45-cell matrix: {value}"
        )
    return cell


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "max-new-cells must be an integer"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "max-new-cells must be positive"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify or resumably execute the corrected regulation-only "
            "NFL X-15 Stage-B matrix."
        )
    )
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--panel-batch",
        type=Path,
        help="Content-addressed RegulationDecisionPanelV4 batch manifest.",
    )
    parser.add_argument(
        "--panel-batch-file-sha256",
        help="Byte-exact SHA-256 of --panel-batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--cell",
        action="append",
        type=_cell,
        dest="cells",
        help=(
            "Requested FOLD:MODEL:BLOCK cell; repeat as needed. "
            "Omit to target the complete frozen matrix."
        ),
    )
    parser.add_argument(
        "--max-new-cells",
        type=_positive_integer,
        default=1,
        help=(
            "Maximum newly fitted cells this invocation (default: 1). "
            "Verified completed shards are resumed, never recomputed."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load PanelV4, fit missing cells, and publish verified shards.",
    )
    parser.add_argument(
        "--freeze-selection",
        action="store_true",
        help=(
            "After the exact 45-cell matrix is complete, recompute and "
            "publish the content-addressed Polymarket SelectionLock."
        ),
    )
    return parser


def _resolved(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started_at = time.perf_counter()
    memory_samples: list[dict[str, object]] = []
    if args.freeze_selection and not args.execute:
        raise ValueError("--freeze-selection requires --execute")
    if args.print_contract:
        print(
            json.dumps(
                _contract(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    memory_samples.append(
        _memory_gate_snapshot(stage="startup", started_at=started_at)
    )
    if (
        args.panel_batch is None
        or args.panel_batch_file_sha256 is None
    ):
        raise ValueError(
            "--panel-batch and --panel-batch-file-sha256 are required"
        )
    project_root = args.project_root.resolve()
    panel_batch = _resolved(project_root, args.panel_batch)
    output_root = _resolved(project_root, args.output_root)
    panel_manifest = verify_regulation_panel_batch_manifest(
        project_root=project_root,
        batch_manifest_path=panel_batch,
        batch_manifest_file_sha256=args.panel_batch_file_sha256,
        expected_game_count=EXPECTED_GAME_COUNT,
    )
    requested = tuple(args.cells or CORRECTED_MODEL_CELLS)
    if not args.execute:
        print(
            json.dumps(
                {
                    "event": "verified_prepare_complete",
                    "panel_schema_version": SCHEMA_VERSION,
                    "panel_batch_manifest_file_sha256": (
                        panel_manifest.batch_manifest_file_sha256
                    ),
                    "panel_batch_sha256": panel_manifest.batch_sha256,
                    "source_row_count": panel_manifest.panel_row_count,
                    "loss_eligible_count": panel_manifest.loss_eligible_count,
                    "game_count": panel_manifest.game_count,
                    "requested_cell_count": len(requested),
                    "models_fitted": 0,
                    "runtime_seconds": round(
                        time.perf_counter() - started_at, 6
                    ),
                    "memory_samples": memory_samples,
                    "holdout_reaction_accessed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    panel = load_verified_regulation_panel_batch(
        project_root=project_root,
        batch_manifest_path=panel_batch,
        batch_manifest_file_sha256=args.panel_batch_file_sha256,
        venue_scope=(POLYMARKET,),
        expected_game_count=EXPECTED_GAME_COUNT,
    )
    memory_samples.append(
        _memory_gate_snapshot(
            stage="after_panel_load",
            started_at=started_at,
        )
    )
    verified_shards = list(discover_verified_regulation_oof_shards(
        project_root=project_root,
        output_root=output_root,
        verified_panel=panel,
    ))
    completed = {shard.cell for shard in verified_shards}
    missing_requested = [
        cell for cell in requested if cell not in completed
    ]
    published: list[dict[str, object]] = []
    for cell in missing_requested[: args.max_new_cells]:
        memory_samples.append(
            _memory_gate_snapshot(
                stage=f"before_cell:{':'.join(cell)}",
                started_at=started_at,
            )
        )
        run = run_regulation_walk_forward(panel, cells=(cell,))
        shard = publish_regulation_oof_shard(
            model_run=run,
            output_root=output_root,
            verified_panel=panel,
        )
        verified_shards.append(
            verify_regulation_oof_shard(
                project_root=project_root,
                output_root=output_root,
                manifest_path=shard.manifest_path,
                manifest_file_sha256=shard.manifest_file_sha256,
                verified_panel=panel,
            )
        )
        published.append(
            {
                "cell": cell,
                "manifest_path": str(shard.manifest_path),
                "manifest_file_sha256": shard.manifest_file_sha256,
                "shard_sha256": shard.shard_sha256,
                "oof_prediction_row_count": (
                    shard.oof_prediction_row_count
                ),
            }
        )
        del run
        memory_samples.append(
            _memory_gate_snapshot(
                stage=f"after_cell:{':'.join(cell)}",
                started_at=started_at,
            )
        )
    batch = publish_regulation_oof_batch_index(
        project_root=project_root,
        output_root=output_root,
        verified_panel=panel,
        shards=verified_shards,
    )
    status = regulation_oof_batch_status(
        tuple(shard.cell for shard in verified_shards)
    )
    selection: dict[str, object] | None = None
    if args.freeze_selection:
        if status["run_status"] != "COMPLETE":
            raise ValueError(
                "--freeze-selection requires the verified exact-45 matrix"
            )
        published_selection = publish_regulation_selection_lock(
            project_root=project_root,
            output_root=output_root,
            verified_panel=panel,
            shards=verified_shards,
        )
        verified_selection = verify_regulation_selection_lock(
            project_root=project_root,
            output_root=output_root,
            manifest_path=published_selection.manifest_path,
            manifest_file_sha256=(
                published_selection.manifest_file_sha256
            ),
            verified_panel=panel,
            shards=verified_shards,
        )
        winner = verified_selection.winner
        selection = {
            "manifest_path": str(verified_selection.manifest_path),
            "manifest_file_sha256": (
                verified_selection.manifest_file_sha256
            ),
            "selection_lock_sha256": (
                verified_selection.selection_lock_sha256
            ),
            "selection_sha256": verified_selection.selection_sha256,
            "decision_status": verified_selection.decision_status,
            "kalshi_transport_allowed": (
                verified_selection.kalshi_transport_allowed
            ),
            "winner_model_id": (
                None if winner is None else winner.model_id
            ),
            "winner_feature_block_id": (
                None if winner is None else winner.feature_block_id
            ),
        }
    print(
        json.dumps(
            {
                "event": "resumable_execution_checkpoint",
                "newly_published": published,
                "new_cell_count": len(published),
                "completed_cell_count": status["completed_cell_count"],
                "missing_cell_count": len(status["missing_cells"]),
                "run_status": batch.run_status,
                "publication_gate": batch.publication_gate,
                "selection_ready": batch.selection_ready,
                "batch_manifest_path": str(batch.manifest_path),
                "batch_manifest_file_sha256": (
                    batch.manifest_file_sha256
                ),
                "batch_sha256": batch.batch_sha256,
                "selection_venue": POLYMARKET,
                "selection_lock": selection,
                "runtime_seconds": round(
                    time.perf_counter() - started_at, 6
                ),
                "memory_samples": memory_samples,
                "memory_warning_gate_peak_rss_mib": (
                    PEAK_RSS_WARNING_MIB
                ),
                "memory_stop_gate_peak_rss_mib": PEAK_RSS_STOP_MIB,
                "memory_stop_gate_system_free_percent": (
                    SYSTEM_FREE_STOP_PERCENT
                ),
                "kalshi_selection_allowed": False,
                "holdout_reaction_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
