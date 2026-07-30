#!/usr/bin/env python3
"""Build or dry-validate the exact-153 native venue rule audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.research.nfl_x15_native_rule_audit import (
    DEFAULT_OUTPUT_ROOT,
    publish_exact153_native_rule_audit,
    validate_exact153_native_rule_audit_config,
    verify_rule_audit_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen NFL X-13 development cohort, native venue "
            "metadata, and FactsV4; publish a rules-only immutable overlay. "
            "No trade, candle, reaction, or holdout object is read."
        )
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Override the main-checkout var/raw root.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify frozen top-level authority without writing output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project = args.project_root.resolve()
    if args.validate_only:
        result = validate_exact153_native_rule_audit_config(
            project_root=project, raw_root=args.raw_root
        )
    else:
        published = publish_exact153_native_rule_audit(
            project_root=project,
            raw_root=args.raw_root,
            output_root=args.output_root,
        )
        verified = verify_rule_audit_batch(
            project_root=project,
            manifest_path=published.manifest_path,
            manifest_file_sha256=published.manifest_file_sha256,
            expected_game_count=153,
        )
        result = {
            "status": "PUBLISHED_AND_VERIFIED",
            "manifest_path": str(verified.manifest_path),
            "manifest_file_sha256": verified.manifest_file_sha256,
            "batch_sha256": verified.batch_sha256,
            "venue_evidence_row_count": verified.venue_evidence_row_count,
            "contract_pair_row_count": verified.contract_pair_row_count,
            "completed_non_tie_comparable_count": (
                verified.completed_non_tie_comparable_count
            ),
            "strict_rule_equivalent_count": (
                verified.strict_rule_equivalent_count
            ),
            "trade_object_reads": 0,
            "candle_object_reads": 0,
            "reaction_object_reads": 0,
            "holdout_game_index_reads": 0,
            "holdout_fact_object_reads": 0,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
