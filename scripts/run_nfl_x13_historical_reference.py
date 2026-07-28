#!/usr/bin/env python3
"""Publish the exact-153 retrospective nflfastR reference study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prediction_market.models.nfl_fastrmodels import (
    ASSET_SHA256,
    load_official_no_spread_predictor,
)
from prediction_market.sports.nfl_x13_historical_reference import (
    CLAIM_BOUNDARY,
    build_reference_summary,
    build_retrospective_reference,
    join_reference_to_market_paths,
)
from prediction_market.sports.nfl_x13_historical_study_publication import (
    _canonical_bytes,
    _file_sha,
    _publish_immutable,
    _publish_table,
    _sha,
    _verified_json,
    _verified_parquet,
    load_development_eligible_paths,
)


def load_verified_feature_views(
    *,
    feature_root: Path,
    batch_index_path: Path,
    expected_game_count: int,
) -> tuple[list[tuple[pd.DataFrame, str]], set[str]]:
    batch = _verified_json(
        batch_index_path,
        expected_sha256=None,
        field="sports feature batch",
    )
    if (
        batch.get("cohort") != "development"
        or batch.get("published_game_count") != expected_game_count
        or batch.get("failed_game_count") != 0
        or batch.get("reaction_access") != "CLOSED"
        or batch.get("market_data_read") is not False
    ):
        raise RuntimeError("sports feature batch does not prove sealed development")
    games = batch.get("games")
    if not isinstance(games, list) or len(games) != expected_game_count:
        raise RuntimeError("sports feature batch game set is invalid")
    frames: list[tuple[pd.DataFrame, str]] = []
    hashes = {_file_sha(batch_index_path)}
    for game in games:
        game_id = str(game["game_id"])
        manifest_path = feature_root / str(game["manifest_path"])
        manifest = _verified_json(
            manifest_path,
            expected_sha256=game["manifest_sha256"],
            field=f"{game_id}.feature_manifest",
        )
        if (
            manifest.get("cohort") != "development"
            or manifest.get("reaction_access") != "CLOSED"
            or manifest.get("market_data_read") is not False
        ):
            raise RuntimeError(f"{game_id} feature manifest is not sealed")
        reference = manifest.get("feature_view")
        frame = _verified_parquet(
            feature_root,
            reference,
            field=f"{game_id}.feature_view",
        )
        if frame["game_id"].nunique() != 1 or str(frame["game_id"].iloc[0]) != game_id:
            raise RuntimeError(f"{game_id} feature identity mismatch")
        object_sha = str(reference["object_sha256"])
        frames.append((frame, object_sha))
        hashes.update((str(game["manifest_sha256"]), object_sha))
    return frames, hashes


def publish_reference_tables(
    *,
    output_root: Path,
    reference: pd.DataFrame,
    reference_paths: pd.DataFrame,
    summary: pd.DataFrame,
    source_hashes: set[str],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    exclusions = (
        reference.groupby(["support_status", "exclusion_reason"], dropna=False)
        .agg(episode_count=("episode_id", "size"), game_count=("game_id", "nunique"))
        .reset_index()
    )
    objects = [
        _publish_table(output_root, logical_name="reference_observations", frame=reference),
        _publish_table(output_root, logical_name="reference_paths", frame=reference_paths),
        _publish_table(output_root, logical_name="reference_summary", frame=summary),
        _publish_table(output_root, logical_name="reference_exclusions", frame=exclusions),
    ]
    material = {
        "schema": "nfl_x13_historical_reference_publication_v1",
        "builder_version": "nfl-x13-historical-reference-v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "development_game_count": 153,
        "model_sha256": ASSET_SHA256,
        "source_manifest_sha256s": sorted(source_hashes),
        "materiality_threshold": 0.01,
        "pit_status": "RETROSPECTIVE_INFERRED_NOT_LIVE_PIT",
        "claim_boundary": CLAIM_BOUNDARY,
        "objects": sorted(objects, key=lambda row: str(row["logical_name"])),
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
    }
    bundle_sha = _sha(_canonical_bytes(material))
    payload = _canonical_bytes({**material, "bundle_sha256": bundle_sha})
    manifest_sha = _sha(payload).removeprefix("sha256:")
    path = (
        output_root
        / "manifests"
        / "sha256"
        / manifest_sha[:2]
        / f"{manifest_sha}.manifest.json"
    )
    _publish_immutable(path, payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-root", type=Path, default=Path("."))
    parser.add_argument(
        "--main-store-root",
        type=Path,
        default=Path("/Users/wekruitclaw1/Desktop/prediction-market/var/raw"),
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
        "--market-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/factor-lab/v2/"
            "expansion-development-market/primary-delay0-v1"
        ),
    )
    parser.add_argument(
        "--market-batch",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/factor-lab/v2/"
            "expansion-development-market/primary-delay0-v1/batches/manifests/"
            "sha256/23/23b4e8e9931dd809cdbcd261f37196a01098f172956d8cff063bf5a5788b449d."
            "batch-index.json"
        ),
    )
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=Path("artifacts/market-observation/nfl/x13/dense-reaction-v2/exact-153"),
    )
    parser.add_argument(
        "--dense-batch",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/dense-reaction-v2/exact-153/"
            "manifests/sha256/1a/"
            "1a27d6ed2a65fa8140bc96d4eaf8609ebe60bbfc69072f89868a5eab5a0ec230."
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path(
            "/Users/wekruitclaw1/Desktop/prediction-market/var/raw/manifests/"
            "source=nflverse/dataset=DS-NFL-FASTRMODELS/"
            "version=9f2495fdb4943087ca663d96706eb5df7973aff4/"
            "partition=asset-253928623/"
            "080d98f34495fe59a532b7c24e17536f471700e92ac8415b682234d7241fe3cb."
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/market-observation/nfl/x13/"
            "historical-reference-v1/exact-153"
        ),
    )
    args = parser.parse_args()

    feature_frames, feature_hashes = load_verified_feature_views(
        feature_root=args.feature_root.resolve(),
        batch_index_path=args.feature_batch.resolve(),
        expected_game_count=153,
    )
    predictor = load_official_no_spread_predictor(
        program_root=args.program_root.resolve(),
        store_root=args.main_store_root.resolve(),
        manifest_path=args.model_manifest.resolve(),
    )
    references = [
        build_retrospective_reference(
            frame,
            predictor=predictor,
            model_sha256=ASSET_SHA256,
            source_object_sha256=object_sha,
        )
        for frame, object_sha in feature_frames
    ]
    reference = pd.concat(references, ignore_index=True)
    market, market_hashes = load_development_eligible_paths(
        market_root=args.market_root,
        market_batch_index_path=args.market_batch,
        dense_root=args.dense_root,
        dense_batch_manifest_path=args.dense_batch,
        expected_game_count=153,
    )
    reference_paths = join_reference_to_market_paths(
        market,
        reference,
        materiality_threshold=0.01,
    )
    summary = build_reference_summary(reference_paths)
    source_hashes = {*feature_hashes, *market_hashes, ASSET_SHA256}
    manifest = publish_reference_tables(
        output_root=args.output_root.resolve(),
        reference=reference,
        reference_paths=reference_paths,
        summary=summary,
        source_hashes=source_hashes,
    )
    result = {
        "reference_rows": len(reference),
        "supported_reference_rows": int(
            reference["support_status"].eq("SUPPORTED_RETROSPECTIVE").sum()
        ),
        "reference_path_rows": len(reference_paths),
        "completion_eligible_rows": int(
            reference_paths["completion_status"].eq("ELIGIBLE").sum()
        ),
        "summary_rows": len(summary),
        "manifest_path": manifest.as_posix(),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
