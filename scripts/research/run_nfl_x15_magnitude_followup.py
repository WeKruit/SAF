#!/usr/bin/env python3
"""Run the governed NFL X-15 conditional-magnitude follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prediction_market.research.nfl_x15_magnitude_followup import (
    run_magnitude_followup,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/"
    "stage-b-magnitude-followup-v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact45-batch-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--exact45-batch-manifest-file-sha256", required=True
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest-file-sha256", required=True
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else artifact_root / args.output_root
    ).resolve()
    publication = run_magnitude_followup(
        exact45_batch_manifest_path=args.exact45_batch_manifest,
        exact45_batch_manifest_file_sha256=(
            args.exact45_batch_manifest_file_sha256
        ),
        selection_manifest_path=args.selection_manifest,
        selection_manifest_file_sha256=(
            args.selection_manifest_file_sha256
        ),
        artifact_root=artifact_root,
        output_root=output_root,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(publication.manifest_path),
                "manifest_sha256": publication.manifest_sha256,
                "status": publication.status,
                "winner_feature_block_id": (
                    publication.winner_feature_block_id
                ),
                "holdout_reaction_accessed": (
                    publication.holdout_reaction_accessed
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
