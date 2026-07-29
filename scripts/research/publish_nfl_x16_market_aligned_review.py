#!/usr/bin/env python3
"""Publish the bilingual DAL–DET fact review with a source-time market overlay.

This consumes only already-published local objects.  It does not fetch market
data, rebuild X-13 reactions, or read holdout artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from prediction_market.reports.nfl_x16_fact_review import render_game_fact_review
from prediction_market.sports.nfl_x16_fact_extraction import GameFactTables


ROOT = Path(__file__).resolve().parents[2]
GAME_ID = "2025_14_DAL_DET"
X16_ROOT = ROOT / "artifacts/market-observation/nfl/x16-draft/fact-review/single-game" / GAME_ID
X16_MANIFEST = X16_ROOT / "manifests/sha256/5e/5e46a30559dbda817dceee12596bfd1de18d471159a2b6f459c14a1722bc30f8.manifest.json"
X13_ROOT = ROOT / "artifacts/market-observation/nfl/x13/factor-lab/v2/single-game" / GAME_ID / "sha256-cfe3c30bcc9710be9c9a95c995b7b6402e2f3dddc073cf2dd9da564f7d7696d1"
MARKET_OBJECT = X13_ROOT / "market_observations_v2.parquet"
MARKET_MANIFEST = X13_ROOT / "manifest.json"
OUTPUT_ROOT = ROOT / "artifacts/market-observation/nfl/x16-draft/fact-review-market/single-game" / GAME_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def object_path(manifest: dict[str, object], name: str) -> Path:
    for record in manifest.get("objects", []):
        if isinstance(record, dict) and record.get("name") == name:
            path = X16_ROOT / str(record["path"])
            expected = str(record["object_sha256"])
            if sha256_file(path) != expected:
                raise RuntimeError(f"hash mismatch for {name}")
            return path
    raise RuntimeError(f"object missing from fact manifest: {name}")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-addressed collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    fact_manifest = json.loads(X16_MANIFEST.read_text(encoding="utf-8"))
    tables = GameFactTables(
        events=pd.read_parquet(object_path(fact_manifest, "canonical_factor_events")),
        tags=pd.read_parquet(object_path(fact_manifest, "factor_event_tags")),
        players=pd.read_parquet(object_path(fact_manifest, "factor_event_players")),
        availability=pd.read_parquet(object_path(fact_manifest, "player_availability_events")),
        injury_evidence=pd.read_parquet(object_path(fact_manifest, "injury_evidence")),
        factor_hits=pd.read_parquet(object_path(fact_manifest, "factor_hits")),
        factor_coverage=pd.read_parquet(object_path(fact_manifest, "factor_coverage_audit")),
        reconciliation=pd.read_parquet(object_path(fact_manifest, "sports_row_reconciliation")),
        registry_sha256=str(fact_manifest["source_refs"]["factor_registry"]["semantic_sha256"]),
    )
    market_manifest = json.loads(MARKET_MANIFEST.read_text(encoding="utf-8"))
    market = pd.read_parquet(MARKET_OBJECT)
    market.attrs["source_bundle_sha256"] = str(market_manifest.get("bundle_sha256"))
    html = render_game_fact_review(tables, market_observations=market).encode("utf-8")
    report_sha = "sha256:" + hashlib.sha256(html).hexdigest()
    digest = report_sha.removeprefix("sha256:")
    report_path = OUTPUT_ROOT / "objects" / "sha256" / digest[:2] / f"{digest}.html"
    atomic_write(report_path, html)
    manifest = {
        "schema": "SingleGameFactMarketReviewBundleV1",
        "artifact_status": "DRAFT_EXPERT_REVIEW_SOURCE_TIME_ALIGNED",
        "game_id": GAME_ID,
        "claim_boundary": "SPORTS_FACT_AND_SOURCE_TIME_TREND_ONLY_NO_CAUSALITY_NO_EXECUTION",
        "market_data_read": True,
        "holdout_reaction_accessed": False,
        "objects": {
            "fact_manifest_sha256": sha256_file(X16_MANIFEST),
            "market_manifest_sha256": sha256_file(MARKET_MANIFEST),
            "market_object_sha256": sha256_file(MARKET_OBJECT),
            "report_sha256": report_sha,
            "report_path": str(report_path),
        },
        "market_semantics": "ACTUAL_WINNER_TRADES_ONLY_LAST_TRADE_PER_UTC_SECOND_NO_FORWARD_FILL_NO_HISTORICAL_BBO",
    }
    manifest_payload = (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    manifest_sha = "sha256:" + hashlib.sha256(manifest_payload).hexdigest()
    manifest_digest = manifest_sha.removeprefix("sha256:")
    manifest_path = OUTPUT_ROOT / "manifests" / "sha256" / manifest_digest[:2] / f"{manifest_digest}.manifest.json"
    atomic_write(manifest_path, manifest_payload)
    print(json.dumps({"report": str(report_path), "report_sha256": report_sha, "manifest": str(manifest_path), "manifest_sha256": manifest_sha}, indent=2))


if __name__ == "__main__":
    main()
