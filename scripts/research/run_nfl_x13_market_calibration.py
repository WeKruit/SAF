#!/usr/bin/env python3
"""Publish exact-153 development-only NFL winner-market calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.sports.nfl_x13_market_calibration import (
    build_calibration_tables,
    build_forecast_observations,
    load_exact_development_inputs,
    publish_calibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--market-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/factor-lab/v2/"
            "expansion-development-market/kalshi-native-time-v3/exact-153"
        ),
    )
    parser.add_argument(
        "--market-batch",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/factor-lab/v2/"
            "expansion-development-market/kalshi-native-time-v3/exact-153/"
            "batches/manifests/sha256/b2/"
            "b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341."
            "batch-index.json"
        ),
    )
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/dense-reaction-v3/"
            "kalshi-native-time-v3/exact-153"
        ),
    )
    parser.add_argument(
        "--dense-manifest",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/dense-reaction-v3/"
            "kalshi-native-time-v3/exact-153/manifests/sha256/a2/"
            "a2aa20c764523d9c117e0087eb319cc50bbf736924009c9f7b8850fb471a3ea4."
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/factor-lab/v2/"
            "expansion-development-sports-features"
        ),
    )
    parser.add_argument(
        "--feature-batch",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/factor-lab/v2/"
            "expansion-development-sports-features/batches/manifests/sha256/3a/"
            "3a09538aa3013d4ddab9aa5e24299c91914b98a41701854037c353326ccf4866."
            "batch-index.json"
        ),
    )
    parser.add_argument(
        "--sports-clean-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/sports-clean-expansion/"
            "reaction-closed-v1"
        ),
    )
    parser.add_argument(
        "--sports-clean-batch",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/sports-clean-expansion/"
            "reaction-closed-v1/batches/manifests/sha256/3f/"
            "3fee02261ae6aeeccbfaa03114ebdbdfd44e96eb7dad1501de5f39c2bce0271b."
            "batch-index.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/market-calibration-v1/"
            "kalshi-native-time-v3/exact-153"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--reliability-bins", type=int, default=10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths, features, hashes = load_exact_development_inputs(
        market_root=args.market_root,
        market_batch_index_path=args.market_batch,
        dense_root=args.dense_root,
        dense_manifest_path=args.dense_manifest,
        feature_root=args.feature_root,
        feature_batch_index_path=args.feature_batch,
        sports_clean_root=args.sports_clean_root,
        sports_clean_batch_index_path=args.sports_clean_batch,
    )
    feature_hashes = {
        key.split(":", 1)[1]: value
        for key, value in hashes.items()
        if key.startswith("feature_object:")
    }
    input_hashes = {
        key: value
        for key, value in hashes.items()
        if not key.startswith("feature_object:")
    }
    observations, exclusions = build_forecast_observations(
        reaction_paths=paths,
        feature_rows=features,
        feature_object_sha256s=feature_hashes,
    )
    tables = build_calibration_tables(
        observations,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        reliability_bin_count=args.reliability_bins,
    )
    publication = publish_calibration(
        output_root=args.output_root,
        observations=observations,
        exclusions=exclusions,
        tables=tables,
        input_hashes=input_hashes,
        expected_game_count=153,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "game_count": int(observations["game_id"].nunique()),
                "forecast_count": len(observations),
                "manifest_path": str(publication.manifest_path),
                "manifest_sha256": publication.manifest_sha256,
                "bundle_sha256": publication.bundle_sha256,
                "metrics": tables["metrics"].to_dict("records"),
                "holdout_reaction_rows_read": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
