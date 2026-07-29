#!/usr/bin/env python3
"""Publish verified historical trades-only Stage B development panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.research.nfl_x15_development_panel import (
    DEFAULT_OUTPUT_ROOT,
    publish_exact153_development_panel,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify frozen X13/StageA/market artifacts and publish the "
            "HistoricalTradesOnlyProbabilityPanelV2 one game at a time. "
            "No upstream I/O and no holdout reads."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Repo-relative content-addressed output root.",
    )
    parser.add_argument(
        "--game-id",
        action="append",
        default=None,
        help=(
            "Publish only this verified development game; repeat for multiple. "
            "Omit to stream all 153 games."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = publish_exact153_development_panel(
        project_root=args.project_root,
        output_root=args.output_root,
        game_ids=args.game_id,
        confirmatory_evidence={},
    )
    print(
        json.dumps(
            {
                "schema": "nfl_x15_development_panel_runner_result_v1",
                "game_count": publication.game_count,
                "batch_manifest_path": str(publication.batch_manifest_path),
                "batch_manifest_sha256": publication.batch_manifest_sha256,
                "batch_sha256": publication.batch_sha256,
                "cohort_authority_sha256": (
                    publication.cohort_authority_sha256
                ),
                "cohort_mapping_sha256": publication.cohort_mapping_sha256,
                "holdout_reaction_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
