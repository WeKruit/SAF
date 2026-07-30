#!/usr/bin/env python3
"""Publish the exact-153 Stage-A decision-feature support audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.research.nfl_x15_stage_a_support_audit import (
    publish_stage_a_decision_support_audit,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-a-decision-support-audit-v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--panel-batch-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--panel-batch-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    )
    publication = publish_stage_a_decision_support_audit(
        project_root=project_root,
        panel_batch_manifest_path=args.panel_batch_manifest,
        panel_batch_file_sha256=args.panel_batch_file_sha256,
        output_root=output_root,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(publication.manifest_path),
                "manifest_file_sha256": (
                    publication.manifest_file_sha256
                ),
                "audit_sha256": publication.audit_sha256,
                "decision_eligible_rows": (
                    publication.decision_eligible_rows
                ),
                "feature_block_status": (
                    publication.feature_block_status
                ),
                "holdout_reaction_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
