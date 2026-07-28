#!/usr/bin/env python3
"""Publish the local-only NFL X-13/X-14 TickDataInventoryV1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_market.research.tick_data_inventory_publication import (
    publish_tick_data_inventory,
)


_BATCH = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-development-market/kalshi-native-time-v3/exact-153/"
    "batches/manifests/sha256/b2/"
    "b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341."
    "batch-index.json"
)
_PMXT_VERIFY = Path(
    "artifacts/data-audit/pmxt-s3-verify-v1/sha256/01/"
    "01d572225540916dd138598333e4398358d548615325cc7a0fa189c28e0e4e30.json"
)
_PMXT_AUDIT = Path(
    "artifacts/data-audit/pmxt-l2-method-audit-v1/sha256/92/"
    "92a80cd64f7cecd79990d2cd0e484743475afc86b867e711fbeb3c700aa29253.json"
)
_X14 = Path(
    "artifacts/market-observation/nfl/x14/sessions/sha256/56/"
    "560c5c80fd39c9cd53c1717a414109ce3c27f91bdc9497289be46cf0efe8c201.json"
)
_KALSHI_CANDLE = Path(
    "artifacts/market-observation/nfl/"
    "nfl_2025_14_dal_det_market_mapping_v0.json"
)
_TIMESEVENTEEN = Path(
    "artifacts/data-audit/polymarket_v1_bounded_sports_extract_v0.json"
)
_OUTPUT = Path("artifacts/data-audit/tick-data-inventory-v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    publication = publish_tick_data_inventory(
        output_root=root / _OUTPUT,
        repo_root=root,
        batch_manifest_path=root / _BATCH,
        pmxt_verify=root / _PMXT_VERIFY,
        pmxt_audit=root / _PMXT_AUDIT,
        x14=root / _X14,
        kalshi_candle=root / _KALSHI_CANDLE,
        timeseventeen=root / _TIMESEVENTEEN,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(publication.manifest_path),
                "manifest_sha256": publication.manifest_sha256,
                "rows_path": str(publication.rows_path),
                "rows_sha256": publication.rows_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
