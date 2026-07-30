#!/usr/bin/env python3
"""Run the supported exact-45 X-15 OOF matrix.

PanelV2 is verified and prepared once.  Existing immutable V4 shards are
fully re-verified; valid D4 shards are explicitly audited and ignored.  Only
missing supported cells are computed.  V3 partial/final indexes are published
atomically and final-holdout reaction data is never read.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Callable, Iterator

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
    publish_x15_oof_shard,
)
from prediction_market.research.nfl_x15_selection_batch_v3 import (
    EXACT_SELECTION_CELLS_V3,
    X15SelectionBatchV3Publication,
    _require_output_root_contract,
    _read_v4_manifest,
    publish_x15_selection_batch_v3,
    verify_x15_selection_support_v3,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-b-probability-oof-v2"
)
EXPECTED_GAME_COUNT = 153
Emit = Callable[..., None]


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


def _expected_cells() -> tuple[tuple[str, str, str], ...]:
    return EXACT_SELECTION_CELLS_V3


def _discover_existing_shards(
    *,
    output_root: Path,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    emit: Emit = _emit,
) -> dict[tuple[str, str, str], Path]:
    """Fully verify supported V4 shards and explicitly ignore valid D4."""

    found: dict[tuple[str, str, str], Path] = {}
    pattern = "shards/*/*/*/manifests/sha256/*/*.shard-manifest.json"
    for path in sorted(output_root.glob(pattern)):
        cell, _, verified_path, _ = _read_v4_manifest(
            manifest_path=path,
            output_root=output_root,
            authority=authority,
            cohort_mapping_sha256=cohort_mapping_sha256,
            allow_unsupported_d4=True,
        )
        fold_id, model_id, feature_block_id = cell
        if feature_block_id == "D4":
            # _read_v4_manifest already invoked the public V4 loader, so this
            # is a verified unsupported shard rather than an untrusted stray.
            emit(
                "unsupported_shard_ignored",
                fold_id=fold_id,
                model_id=model_id,
                feature_block_id=feature_block_id,
                reason="UNSUPPORTED_FEATURE_BLOCK",
            )
            continue
        if cell in found:
            raise ValueError(
                f"conflicting duplicate published shard for {cell}: "
                f"{found[cell]} and {verified_path}"
            )
        found[cell] = verified_path
    return {
        cell: found[cell]
        for cell in EXACT_SELECTION_CELLS_V3
        if cell in found
    }


def _publish_index(
    *,
    manifests: dict[tuple[str, str, str], Path],
    authority: FrozenDevelopmentAuthority,
    prepared: X15PreparedDiagnosticPanel,
    project_root: Path,
    output_root: Path,
    panel_batch: Path,
    panel_batch_file_sha256: str,
    feature_support_audit: Path,
    feature_support_audit_file_sha256: str,
) -> X15SelectionBatchV3Publication:
    return publish_x15_selection_batch_v3(
        shard_manifest_paths=tuple(
            manifests[cell]
            for cell in EXACT_SELECTION_CELLS_V3
            if cell in manifests
        ),
        authority=authority,
        cohort_mapping_sha256=prepared.cohort_mapping_sha256,
        output_root=output_root,
        project_root=project_root,
        panel_batch_manifest_path=panel_batch,
        panel_batch_file_sha256=panel_batch_file_sha256,
        feature_support_audit_path=feature_support_audit,
        feature_support_audit_file_sha256=(
            feature_support_audit_file_sha256
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact-153 supported Stage-B probability OOF cells with "
            "verified V4 shard resume and exact-45 V3 publication."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--panel-batch", type=Path, required=True)
    parser.add_argument("--panel-batch-file-sha256", required=True)
    parser.add_argument(
        "--feature-support-audit", type=Path, required=True
    )
    parser.add_argument(
        "--feature-support-audit-file-sha256", required=True
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--max-new-cells", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    ).resolve()
    _require_output_root_contract(
        project_root=project_root,
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.max_new_cells is not None and args.max_new_cells < 1:
        raise ValueError("--max-new-cells must be positive")

    _emit(
        "prepare_started",
        expected_game_count=EXPECTED_GAME_COUNT,
        expected_execution_cell_count=45,
        holdout_reaction_accessed=False,
    )
    prepared, authority = _prepare_with_authority(
        project_root=project_root,
        panel_batch=args.panel_batch,
        panel_batch_file_sha256=args.panel_batch_file_sha256,
    )
    verify_x15_selection_support_v3(
        authority=authority,
        cohort_mapping_sha256=prepared.cohort_mapping_sha256,
        project_root=project_root,
        panel_batch_manifest_path=args.panel_batch,
        panel_batch_file_sha256=args.panel_batch_file_sha256,
        feature_support_audit_path=args.feature_support_audit,
        feature_support_audit_file_sha256=(
            args.feature_support_audit_file_sha256
        ),
    )
    _emit(
        "prepare_completed",
        source_row_count=prepared.source_row_count,
        decision_row_count=len(prepared.frame),
        parsed_cache_entries=len(prepared.parsed_by_sha256),
        game_count=len(prepared.game_ids),
        cohort_authority_sha256=prepared.cohort_authority_sha256,
        cohort_mapping_sha256=prepared.cohort_mapping_sha256,
        feature_support_audit_status="PASS",
        holdout_reaction_accessed=False,
    )
    if args.prepare_only:
        return 0

    manifests = _discover_existing_shards(
        output_root=output_root,
        authority=authority,
        cohort_mapping_sha256=prepared.cohort_mapping_sha256,
    )
    if manifests:
        partial = _publish_index(
            manifests=manifests,
            authority=authority,
            prepared=prepared,
            project_root=project_root,
            output_root=output_root,
            panel_batch=args.panel_batch,
            panel_batch_file_sha256=args.panel_batch_file_sha256,
            feature_support_audit=args.feature_support_audit,
            feature_support_audit_file_sha256=(
                args.feature_support_audit_file_sha256
            ),
        )
        _emit(
            "resume_verified",
            existing_supported_shard_count=len(manifests),
            missing_supported_shard_count=45 - len(manifests),
            batch_manifest_path=str(partial.manifest_path),
            selection_ready=partial.selection_ready,
        )

    completed = 0
    for fold_id, model_id, feature_block_id in EXACT_SELECTION_CELLS_V3:
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
        completed += 1
        _emit(
            "cell_completed",
            fold_id=fold_id,
            model_id=model_id,
            feature_block_id=feature_block_id,
            shard_manifest_path=str(publication.manifest_path),
            shard_manifest_sha256=publication.manifest_sha256,
            completed_supported_shard_count=len(manifests),
        )
        del result
        gc.collect()

    if not manifests:
        raise RuntimeError("runner produced no supported V4 shards")
    final_index = _publish_index(
        manifests=manifests,
        authority=authority,
        prepared=prepared,
        project_root=project_root,
        output_root=output_root,
        panel_batch=args.panel_batch,
        panel_batch_file_sha256=args.panel_batch_file_sha256,
        feature_support_audit=args.feature_support_audit,
        feature_support_audit_file_sha256=(
            args.feature_support_audit_file_sha256
        ),
    )
    _emit(
        "run_completed",
        completed_this_process=completed,
        total_supported_shard_count=len(manifests),
        expected_execution_cell_count=45,
        selection_ready=final_index.selection_ready,
        publication_status=(
            "SELECTION_READY"
            if final_index.selection_ready
            else "PARTIAL_DIAGNOSTIC_ONLY"
        ),
        batch_manifest_path=str(final_index.manifest_path),
        batch_manifest_sha256=final_index.manifest_sha256,
        batch_sha256=final_index.batch_sha256,
        selection_contract_sha256=(
            final_index.selection_contract_sha256
        ),
        holdout_reaction_accessed=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
