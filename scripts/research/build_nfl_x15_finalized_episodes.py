#!/usr/bin/env python3
"""Publish regulation-only FinalizedEpisodeV2 from verified Exact-153 Facts V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.research.nfl_x15_finalized_episodes import (
    DEFAULT_FACTS_BATCH,
    DEFAULT_FACTS_BATCH_FILE_SHA256,
    DEFAULT_OUTPUT_ROOT,
    publish_finalized_episode_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build source-linked regulation-only NFL finalized episodes from "
            "the verified X-13 Exact-153 Facts V4 publication."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--facts-batch", type=Path, default=DEFAULT_FACTS_BATCH)
    parser.add_argument(
        "--facts-batch-file-sha256",
        default=DEFAULT_FACTS_BATCH_FILE_SHA256,
    )
    parser.add_argument("--expected-game-count", type=int, default=153)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    publication = publish_finalized_episode_batch(
        project_root=arguments.project_root,
        facts_batch_path=arguments.facts_batch,
        facts_batch_file_sha256=arguments.facts_batch_file_sha256,
        expected_game_count=arguments.expected_game_count,
        output_root=arguments.output_root,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_path": str(publication.manifest_path),
                "manifest_sha256": publication.manifest_sha256,
                "bundle_sha256": publication.bundle_sha256,
                "game_count": publication.game_count,
                "raw_row_count": publication.raw_row_count,
                "information_anchor_count": (
                    publication.information_anchor_count
                ),
                "episode_count": publication.episode_count,
                "stage_a_eligible_episode_count": (
                    publication.stage_a_eligible_episode_count
                ),
                "residual_diagnostic_eligible_episode_count": (
                    publication.residual_diagnostic_eligible_episode_count
                ),
                "primary_selection_eligible_anchor_count": (
                    publication.primary_selection_eligible_anchor_count
                ),
                "censor_boundary_eligible_anchor_count": (
                    publication.censor_boundary_eligible_anchor_count
                ),
                "unresolved_adjudication_episode_count": (
                    publication.unresolved_adjudication_episode_count
                ),
                "data_gap_episode_count": publication.data_gap_episode_count,
                "nullified_episode_count": publication.nullified_episode_count,
                "ot_excluded_row_count": publication.ot_excluded_row_count,
                "raw_rows_silently_dropped": (
                    publication.raw_rows_silently_dropped
                ),
                "runtime_seconds": round(publication.runtime_seconds, 6),
                "peak_rss_mib": round(publication.peak_rss_mib, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
