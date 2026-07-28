#!/usr/bin/env python3
"""Capture the reaction-closed 2025 NFL dual-venue moneyline cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.sports.nfl_historical_moneyline_capture import (
    DEFAULT_HOST_REQUESTS_PER_SECOND,
    capture_moneyline_expansion,
    dry_run_moneyline_expansion,
    load_moneyline_expansion_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Byte-exact, resumable moneyline-only capture. The capture index "
            "is reaction_access=CLOSED and never derives price direction."
        )
    )
    parser.add_argument("--candidate-scan", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--governance", type=Path, required=True)
    parser.add_argument("--program-root", type=Path, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Explicit main-repository var/raw path.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-response-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-kalshi-pages", type=int, default=100)
    parser.add_argument(
        "--gamma-rps",
        type=float,
        default=float(
            DEFAULT_HOST_REQUESTS_PER_SECOND["gamma-api.polymarket.com"]
        ),
    )
    parser.add_argument(
        "--polymarket-rps",
        type=float,
        default=float(
            DEFAULT_HOST_REQUESTS_PER_SECOND["data-api.polymarket.com"]
        ),
    )
    parser.add_argument(
        "--kalshi-rps",
        type=float,
        default=float(
            DEFAULT_HOST_REQUESTS_PER_SECOND["external-api.kalshi.com"]
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = load_moneyline_expansion_spec(
        args.candidate_scan,
        args.temporal_audit,
        args.governance,
    )
    if args.dry_run:
        result: object = dry_run_moneyline_expansion(
            spec,
            raw_root=args.raw_root,
            program_root=args.program_root,
            output_root=args.output_root,
        )
    else:
        capture = capture_moneyline_expansion(
            spec,
            raw_root=args.raw_root,
            program_root=args.program_root,
            output_root=args.output_root,
            host_requests_per_second={
                "gamma-api.polymarket.com": args.gamma_rps,
                "data-api.polymarket.com": args.polymarket_rps,
                "external-api.kalshi.com": args.kalshi_rps,
            },
            max_response_bytes=args.max_response_bytes,
            max_kalshi_pages=args.max_kalshi_pages,
        )
        result = {
            "captured_game_count": capture.captured_game_count,
            "game_count": capture.game_count,
            "manifest_path": str(capture.manifest_path),
            "manifest_sha256": capture.manifest_sha256,
            "pointer_path": str(capture.pointer_path),
            "resumed_game_count": capture.resumed_game_count,
        }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
