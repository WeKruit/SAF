#!/usr/bin/env python3
"""Run the frozen NFL expansion development market pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from prediction_market.sports.nfl_expansion_development_market import (
    COMPUTE_WORKERS,
    DELAY_SCENARIOS_SECONDS,
    DevelopmentMarketError,
    MAX_COMPUTE_WORKERS,
    run_bounded_fixture,
    run_development_market_pipeline,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bounded-fixture",
        action="store_true",
        help="Run the deterministic in-memory contract fixture only.",
    )
    parser.add_argument("--unlock-manifest", type=Path)
    parser.add_argument("--capture-batch", type=Path)
    parser.add_argument("--sports-feature-batch", type=Path)
    parser.add_argument("--factor-registry", type=Path)
    parser.add_argument("--expected-unlock-sha256")
    parser.add_argument("--expected-capture-batch-sha256")
    parser.add_argument("--expected-sports-feature-batch-sha256")
    parser.add_argument("--expected-factor-registry-sha256")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--game-id",
        action="append",
        help="Run only this development game; repeat for more than one.",
    )
    delay_group = parser.add_mutually_exclusive_group()
    delay_group.add_argument(
        "--all-delays",
        action="store_true",
        help="Run 0/1/2/3/5/10s; default is primary delay 0 only.",
    )
    delay_group.add_argument(
        "--delay",
        action="append",
        type=int,
        choices=DELAY_SCENARIOS_SECONDS,
        help=(
            "Run only this registered delay; repeat for a sensitivity set. "
            "Default is primary delay 0."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_COMPUTE_WORKERS + 1),
        default=COMPUTE_WORKERS,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.bounded_fixture:
        print(json.dumps(run_bounded_fixture(), sort_keys=True))
        return 0
    required = {
        name: getattr(args, name)
        for name in (
            "unlock_manifest",
            "capture_batch",
            "sports_feature_batch",
            "factor_registry",
            "expected_unlock_sha256",
            "expected_capture_batch_sha256",
            "expected_sports_feature_batch_sha256",
            "expected_factor_registry_sha256",
            "raw_root",
            "output_root",
        )
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise SystemExit(
            "missing required arguments: " + ", ".join(missing)
        )
    try:
        result = run_development_market_pipeline(
            **required,
            game_ids=args.game_id,
            delay_scenarios=(
                DELAY_SCENARIOS_SECONDS
                if args.all_delays
                else (
                    tuple(sorted(set(args.delay)))
                    if args.delay
                    else (0,)
                )
            ),
            compute_workers=args.workers,
        )
    except DevelopmentMarketError as error:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
