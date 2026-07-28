#!/usr/bin/env python3
"""Materialize reaction-closed X-13 development sports features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.sports.nfl_expansion_development_features import (
    materialize_expansion_development_features,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable sports-only feature views for the frozen "
            "153-game X-13 development split."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help=(
            "Bounded development game. Repeat for multiple games. "
            "Omit to build the exact frozen 153-game split."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one JSON progress record per verified/published game.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    def progress(value: object) -> None:
        print(json.dumps(value, sort_keys=True), flush=True)

    result = materialize_expansion_development_features(
        project_root=args.project_root,
        data_root=args.data_root,
        output_root=args.output_root,
        game_ids=args.game_ids,
        workers=args.workers,
        progress=progress if args.progress else None,
    )
    print(
        json.dumps(
            {
                "aggregate_low_support_row_count": (
                    result.aggregate_low_support_row_count
                ),
                "aggregate_row_count": result.aggregate_row_count,
                "batch_sha256": result.batch_sha256,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "game_count": result.game_count,
                "index_path": str(result.index_path),
                "index_sha256": result.index_sha256,
                "reused_checkpoint_count": sum(
                    game.reused_checkpoint for game in result.games
                ),
                "scope": result.scope,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
