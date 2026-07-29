#!/usr/bin/env python3
"""Run resumable exact-153 X-15 probability OOF shards.

The runner verifies the already-published diagnostic panel once, prepares one
compact in-memory decision cache, and then executes one fold/model/block cell
at a time.  Every completed cell and every partial/full index is published
atomically; no upstream capture or holdout reaction is read.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterator

import pandas as pd

from prediction_market.research.nfl_x15_development_panel import (
    VerifiedDiagnosticPanelPartition,
    iter_verified_diagnostic_panel_partitions,
)
from prediction_market.research.nfl_x15_model_selection import (
    EXPECTED_FOLDS,
    FrozenDevelopmentAuthority,
    bind_frozen_development_authority,
)
from prediction_market.research.nfl_x15_models import (
    X15PreparedDiagnosticPanel,
    prepare_x15_historical_trades_diagnostic_partitions,
    run_x15_historical_trades_diagnostic_walk_forward,
)
from prediction_market.research.nfl_x15_oof_publication import (
    PROBABILITY_DESIGN_MATRIX,
    X15OOFBatchPublication,
    publish_x15_oof_batch,
    publish_x15_oof_shard,
)


DEFAULT_PANEL_BATCH = Path(
    "artifacts/market-observation/nfl/x15/"
    "historical-trades-only-development-panel-v1/batches/manifests/"
    "sha256/3d/"
    "3d2247c8b075748ccfa219daaf760e1681e85cb8fe0601bc2c9657381c7a969e"
    ".batch-index.json"
)
DEFAULT_PANEL_BATCH_FILE_SHA256 = (
    "sha256:3d2247c8b075748ccfa219daaf760e1681e85cb8fe0601bc2c9657381c7a969e"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-b-probability-oof-v1"
)
EXPECTED_GAME_COUNT = 153


def _emit(event: str, **values: object) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _explicit_utc(value: pd.Timestamp) -> str:
    if value.tzinfo is None:
        raise ValueError("authority source time must be timezone-aware")
    # Keep one fixed ISO-8601 shape.  Pandas parses a mixed series of
    # fractional/non-fractional ISO strings with one inferred format, so
    # variable precision would turn otherwise valid rows into NaT.
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _prepare_with_authority(
    *,
    project_root: Path,
    panel_batch: Path,
    panel_batch_file_sha256: str,
) -> tuple[X15PreparedDiagnosticPanel, FrozenDevelopmentAuthority]:
    authority_rows: list[dict[str, object]] = []

    def observe() -> Iterator[VerifiedDiagnosticPanelPartition]:
        for partition in iter_verified_diagnostic_panel_partitions(
            project_root=project_root,
            batch_manifest_path=panel_batch,
            batch_manifest_file_sha256=panel_batch_file_sha256,
            expected_game_count=EXPECTED_GAME_COUNT,
        ):
            panel = partition.panel
            weeks = tuple(
                sorted(
                    set(
                        pd.to_numeric(
                            panel["nfl_week"], errors="raise"
                        ).astype(int)
                    )
                )
            )
            source_times = pd.to_datetime(
                panel["source_interval_start"],
                utc=True,
                errors="coerce",
            ).dropna()
            if len(weeks) != 1 or source_times.empty:
                raise ValueError(
                    f"{partition.game_id} lacks one week or source-time anchor"
                )
            authority_rows.append(
                {
                    "game_id": partition.game_id,
                    "nfl_week": weeks[0],
                    # This is the first verified canonical game-source time.
                    # It is used only as an immutable chronology identity.
                    "kickoff_utc": _explicit_utc(source_times.min()),
                    "batch_sha256": partition.game_manifest_sha256,
                    "cohort_authority_sha256": (
                        partition.cohort_authority_sha256
                    ),
                }
            )
            yield partition

    prepared = prepare_x15_historical_trades_diagnostic_partitions(observe())
    authority = bind_frozen_development_authority(
        pd.DataFrame(authority_rows),
        cohort_authority_sha256=prepared.cohort_authority_sha256,
    )
    if authority.game_count != EXPECTED_GAME_COUNT:
        raise ValueError("prepared authority is not exact-153")
    return prepared, authority


def _selected_cells(
    *,
    fold_ids: tuple[str, ...],
    model_ids: tuple[str, ...] | None,
    feature_block_ids: tuple[str, ...] | None,
) -> tuple[tuple[str, str, str], ...]:
    expected_folds = tuple(fold[0] for fold in EXPECTED_FOLDS)
    unknown_folds = set(fold_ids).difference(expected_folds)
    if unknown_folds:
        raise ValueError(f"unknown fold IDs: {sorted(unknown_folds)}")
    selected_models = set(model_ids or ())
    selected_blocks = set(feature_block_ids or ())
    cells = tuple(
        (fold_id, model_id, feature_block_id)
        for fold_id in expected_folds
        if fold_id in fold_ids
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX
        if not selected_models or model_id in selected_models
        if not selected_blocks or feature_block_id in selected_blocks
    )
    if not cells:
        raise ValueError("cell selection is empty")
    return cells


def _existing_shards(output_root: Path) -> dict[tuple[str, str, str], Path]:
    found: dict[tuple[str, str, str], Path] = {}
    pattern = (
        "shards/*/*/*/manifests/sha256/*/*.shard-manifest.json"
    )
    for path in sorted(output_root.glob(pattern)):
        relative = path.relative_to(output_root)
        parts = relative.parts
        cell = (parts[1], parts[2], parts[3])
        if cell in found:
            raise ValueError(f"duplicate published shard for {cell}")
        found[cell] = path
    return found


def _publish_index(
    *,
    manifests: dict[tuple[str, str, str], Path],
    authority: FrozenDevelopmentAuthority,
    prepared: X15PreparedDiagnosticPanel,
    output_root: Path,
) -> X15OOFBatchPublication:
    return publish_x15_oof_batch(
        shard_manifest_paths=tuple(
            manifests[cell] for cell in sorted(manifests)
        ),
        authority=authority,
        cohort_mapping_sha256=prepared.cohort_mapping_sha256,
        output_root=output_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact-153 Stage-B probability OOF cells sequentially with "
            "verified resumable publication."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--panel-batch", type=Path, default=DEFAULT_PANEL_BATCH
    )
    parser.add_argument(
        "--panel-batch-file-sha256",
        default=DEFAULT_PANEL_BATCH_FILE_SHA256,
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--fold-id",
        action="append",
        dest="fold_ids",
        choices=tuple(fold[0] for fold in EXPECTED_FOLDS),
    )
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument(
        "--feature-block-id",
        action="append",
        dest="feature_block_ids",
    )
    parser.add_argument(
        "--max-new-cells",
        type=int,
        default=None,
        help="Stop after this many newly computed cells; useful for a pilot.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify and prepare exact-153, then exit without fitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fold_ids = tuple(
        args.fold_ids
        or tuple(fold[0] for fold in EXPECTED_FOLDS)
    )
    cells = _selected_cells(
        fold_ids=fold_ids,
        model_ids=tuple(args.model_ids) if args.model_ids else None,
        feature_block_ids=(
            tuple(args.feature_block_ids)
            if args.feature_block_ids
            else None
        ),
    )
    if args.max_new_cells is not None and args.max_new_cells < 1:
        raise ValueError("--max-new-cells must be positive")

    _emit(
        "prepare_started",
        expected_game_count=EXPECTED_GAME_COUNT,
        selected_cell_count=len(cells),
        holdout_reaction_accessed=False,
    )
    prepared, authority = _prepare_with_authority(
        project_root=project_root,
        panel_batch=args.panel_batch,
        panel_batch_file_sha256=args.panel_batch_file_sha256,
    )
    _emit(
        "prepare_completed",
        source_row_count=prepared.source_row_count,
        decision_row_count=len(prepared.frame),
        parsed_cache_entries=len(prepared.parsed_by_sha256),
        game_count=len(prepared.game_ids),
        prepared_frame_bytes=int(
            prepared.frame.memory_usage(index=True, deep=True).sum()
        ),
        cohort_authority_sha256=prepared.cohort_authority_sha256,
        cohort_mapping_sha256=prepared.cohort_mapping_sha256,
        holdout_reaction_accessed=False,
    )
    if args.prepare_only:
        return 0

    manifests = _existing_shards(output_root)
    if manifests:
        index = _publish_index(
            manifests=manifests,
            authority=authority,
            prepared=prepared,
            output_root=output_root,
        )
        _emit(
            "resume_verified",
            existing_shard_count=len(manifests),
            batch_manifest_path=str(index.manifest_path),
            selection_ready=index.selection_ready,
        )

    completed = 0
    for fold_id, model_id, feature_block_id in cells:
        cell = (fold_id, model_id, feature_block_id)
        if cell in manifests:
            _emit(
                "cell_skipped_verified",
                fold_id=fold_id,
                model_id=model_id,
                feature_block_id=feature_block_id,
            )
            continue
        if (
            args.max_new_cells is not None
            and completed >= args.max_new_cells
        ):
            break
        _emit(
            "cell_started",
            fold_id=fold_id,
            model_id=model_id,
            feature_block_id=feature_block_id,
        )
        result = run_x15_historical_trades_diagnostic_walk_forward(
            prepared,
            model_ids=(model_id,),
            feature_block_ids=(feature_block_id,),
            fold_ids=(fold_id,),
            transport_pairs=(("polymarket", "kalshi"),),
            include_magnitude=False,
        )
        publication = publish_x15_oof_shard(
            model_run=result,
            authority=authority,
            cohort_mapping_sha256=prepared.cohort_mapping_sha256,
            output_root=output_root,
        )
        manifests[cell] = publication.manifest_path
        index = _publish_index(
            manifests=manifests,
            authority=authority,
            prepared=prepared,
            output_root=output_root,
        )
        completed += 1
        _emit(
            "cell_completed",
            fold_id=fold_id,
            model_id=model_id,
            feature_block_id=feature_block_id,
            shard_manifest_path=str(publication.manifest_path),
            shard_manifest_sha256=publication.manifest_sha256,
            completed_shard_count=len(manifests),
            batch_manifest_path=str(index.manifest_path),
            selection_ready=index.selection_ready,
        )
        del result
        gc.collect()

    final_index = _publish_index(
        manifests=manifests,
        authority=authority,
        prepared=prepared,
        output_root=output_root,
    )
    _emit(
        "run_completed",
        completed_this_process=completed,
        total_shard_count=len(manifests),
        selected_cell_count=len(cells),
        selection_ready=final_index.selection_ready,
        publication_status=(
            "SELECTION_READY"
            if final_index.selection_ready
            else "PARTIAL_DIAGNOSTIC_ONLY"
        ),
        batch_manifest_path=str(final_index.manifest_path),
        batch_manifest_sha256=final_index.manifest_sha256,
        batch_sha256=final_index.batch_sha256,
        holdout_reaction_accessed=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
