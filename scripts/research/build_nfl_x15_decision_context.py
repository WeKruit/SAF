#!/usr/bin/env python3
"""Build the verified exact-153 NFL event pre-state context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.research.nfl_x15_decision_context import (
    DEFAULT_OUTPUT_ROOT,
    DecisionContextSourceSpec,
    prepare_exact153_decision_context,
    publish_exact153_decision_context,
)
from prediction_market.research.nfl_x15_development_panel import (
    DEFAULT_FACTS_BATCH,
    DEFAULT_FACTS_BATCH_FILE_SHA256,
    DEFAULT_STAGE_A_BATCH,
    DEFAULT_STAGE_A_BATCH_FILE_SHA256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Facts V4 and Stage A, then build the event-level "
            "pre-state context without reading market or holdout data."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--facts-batch", type=Path, default=DEFAULT_FACTS_BATCH)
    parser.add_argument(
        "--facts-batch-file-sha256",
        default=DEFAULT_FACTS_BATCH_FILE_SHA256,
    )
    parser.add_argument(
        "--stage-a-batch", type=Path, default=DEFAULT_STAGE_A_BATCH
    )
    parser.add_argument(
        "--stage-a-batch-file-sha256",
        default=DEFAULT_STAGE_A_BATCH_FILE_SHA256,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify and build in memory but do not publish any artifact.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = DecisionContextSourceSpec(
        facts_batch_path=args.facts_batch,
        facts_batch_file_sha256=args.facts_batch_file_sha256,
        stage_a_batch_path=args.stage_a_batch,
        stage_a_batch_file_sha256=args.stage_a_batch_file_sha256,
    )
    project = args.project_root.resolve()
    if args.prepare_only:
        prepared = prepare_exact153_decision_context(
            project_root=project,
            source_spec=spec,
        )
        payload = {
            "status": "PREPARED_NOT_PUBLISHED",
            "schema": "EventPrestateContextV1",
            "stage_b_panel_status": "FINALIZED_EPISODE_PANEL_NOT_YET_BUILT",
            "game_count": prepared.game_count,
            "row_count": len(prepared.context),
            "audit_counts": dict(prepared.audit_counts),
            "facts_batch_file_sha256": prepared.facts_batch_file_sha256,
            "facts_batch_sha256": prepared.facts_batch_sha256,
            "stage_a_batch_file_sha256": (
                prepared.stage_a_batch_file_sha256
            ),
            "stage_a_batch_sha256": prepared.stage_a_batch_sha256,
            "market_data_read": False,
            "holdout_reaction_accessed": False,
        }
    else:
        publication = publish_exact153_decision_context(
            project_root=project,
            output_root=args.output_root,
            source_spec=spec,
        )
        payload = {
            "status": "PUBLISHED",
            "schema": "EventPrestateContextV1",
            "stage_b_panel_status": "FINALIZED_EPISODE_PANEL_NOT_YET_BUILT",
            "game_count": publication.game_count,
            "row_count": publication.row_count,
            "regulation_rows": publication.regulation_rows,
            "overtime_rows": publication.overtime_rows,
            "supported_prestate_rows": publication.supported_prestate_rows,
            "context_path": str(publication.context_path),
            "context_sha256": publication.context_sha256,
            "audit_path": str(publication.audit_path),
            "audit_sha256": publication.audit_sha256,
            "manifest_path": str(publication.manifest_path),
            "manifest_sha256": publication.manifest_sha256,
            "bundle_sha256": publication.bundle_sha256,
            "market_data_read": False,
            "holdout_reaction_accessed": False,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
