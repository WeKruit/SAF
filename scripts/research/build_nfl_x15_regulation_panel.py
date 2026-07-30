#!/usr/bin/env python3
"""Publish the exact-153 PIA-grained NFL X-15 PanelV4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from prediction_market.research.nfl_x15_regulation_panel_v4 import (
    DEFAULT_CONTEXT_MANIFEST,
    DEFAULT_CONTEXT_MANIFEST_FILE_SHA256,
    DEFAULT_INFORMATION_ANCHOR_MANIFEST,
    DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256,
    DEFAULT_OUTPUT_ROOT,
    RegulationPanelV4SourceSpec,
    publish_exact153_regulation_panel_v4,
    validate_exact153_regulation_panel_v4_config,
)
from prediction_market.research.nfl_x15_development_panel import (
    default_development_source_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen exact-153 ProvisionalInformationAnchorV2, "
            "EventPrestateContextV2, actual-trade capture pagination, and "
            "normalized market publications; then stream 153 regulation "
            "games into PanelV4. No network, FinalizedEpisode formal input, "
            "or holdout reaction is accessed."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify exact-153 authority/configuration without writing output.",
    )
    parser.add_argument(
        "--information-anchor-manifest",
        type=Path,
        default=DEFAULT_INFORMATION_ANCHOR_MANIFEST,
    )
    parser.add_argument(
        "--information-anchor-manifest-sha256",
        default=DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256,
    )
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=DEFAULT_CONTEXT_MANIFEST,
    )
    parser.add_argument(
        "--context-manifest-sha256",
        default=DEFAULT_CONTEXT_MANIFEST_FILE_SHA256,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_spec = RegulationPanelV4SourceSpec(
        development=default_development_source_spec(),
        information_anchor_manifest_path=(
            args.information_anchor_manifest
        ),
        information_anchor_manifest_file_sha256=(
            args.information_anchor_manifest_sha256
        ),
        context_manifest_path=args.context_manifest,
        context_manifest_file_sha256=args.context_manifest_sha256,
    )
    if args.validate_only:
        validation = validate_exact153_regulation_panel_v4_config(
            project_root=args.project_root.resolve(),
            output_root=args.output_root,
            source_spec=source_spec,
        )
        print(
            json.dumps(
                validation,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    publication = publish_exact153_regulation_panel_v4(
        project_root=args.project_root.resolve(),
        output_root=args.output_root,
        source_spec=source_spec,
        progress_callback=lambda event: print(
            json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        ),
    )
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "schema": "RegulationDecisionPanelV4",
                "batch_manifest_path": str(publication.batch_manifest_path),
                "batch_manifest_sha256": publication.batch_manifest_sha256,
                "batch_sha256": publication.batch_sha256,
                "game_count": publication.game_count,
                "primary_information_anchor_count": (
                    publication.primary_information_anchor_count
                ),
                "panel_row_count": publication.panel_row_count,
                "landmark_eligible_count": (
                    publication.landmark_eligible_count
                ),
                "loss_eligible_count": publication.loss_eligible_count,
                "direction_eligible_count": (
                    publication.direction_eligible_count
                ),
                "censored_count": publication.censored_count,
                "availability_zero_count": (
                    publication.availability_zero_count
                ),
                "availability_one_count": (
                    publication.availability_one_count
                ),
                "runtime_seconds": round(publication.runtime_seconds, 6),
                "peak_rss_mib": round(publication.peak_rss_mib, 3),
                "holdout_reaction_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
