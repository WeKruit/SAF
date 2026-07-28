#!/usr/bin/env python3
"""Materialize reaction-closed NFL 2025 sports-clean bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.sports.nfl_historical_sports_clean import (
    DEFAULT_OUTPUT_ROOT,
    PublishedGameBundle,
    batch_result_record,
    materialize_historical_sports_clean,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable sports-only NFL 2025 clean/replay bundles. "
            "Market response and reaction access stay CLOSED."
        )
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--game-id",
        help="Run one bounded candidate game, e.g. 2025_01_ARI_NO.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Run the complete frozen 234-game candidate scope.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Compute workers for --all (1-4, default: 4).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_project_root(),
        help="Checkout that owns code, candidate coverage, and outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Main checkout containing frozen var/raw objects. "
            "Auto-detected from the linked worktree when omitted."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Publication root. Defaults to the new reaction-closed X-13 "
            "sports-clean expansion path."
        ),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        help="Override only for verifying the same frozen 234-game manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    project_root = args.project_root.resolve()
    output_root = (
        (project_root / DEFAULT_OUTPUT_ROOT).resolve()
        if args.output_root is None
        else args.output_root.resolve()
    )

    def progress(
        game: PublishedGameBundle,
        completed: int,
        total: int,
    ) -> None:
        status = "checkpoint" if game.reused_checkpoint else "built"
        print(
            f"[{completed:03d}/{total:03d}] {game.game_id} "
            f"{status} raw={game.counts['raw_pbp_rows']} "
            f"episodes={game.counts['finalized_episode_rows']} "
            f"seconds={game.elapsed_seconds:.3f}",
            flush=True,
        )

    batch = materialize_historical_sports_clean(
        project_root=project_root,
        data_root=args.data_root,
        output_root=output_root,
        candidate_manifest_path=args.candidate_manifest,
        game_ids=None if args.all else (args.game_id,),
        workers=args.workers if args.all else 1,
        progress=progress,
    )
    print(
        json.dumps(
            batch_result_record(batch, output_root=output_root),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
