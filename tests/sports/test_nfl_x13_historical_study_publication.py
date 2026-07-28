from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.sports.nfl_x13_historical_study import (
    build_historical_factor_study,
)
from prediction_market.sports.nfl_x13_historical_study_publication import (
    HistoricalStudyPublicationError,
    load_development_eligible_paths,
    publish_historical_study_tables,
    verify_historical_study_publication,
)


def _eligible() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "observation_id": f"obs-{game}-{venue}-{horizon}",
                "factor_id": "NFL.EVENT.TEST",
                "factor_version": "v1",
                "game_id": game,
                "episode_id": f"episode-{game}",
                "logical_market_id": f"{venue}:winner",
                "outcome": "AAA",
                "venue": venue,
                "market_family": "moneyline",
                "outcome_orientation": "AAA",
                "horizon_seconds": horizon,
                "effect": effect,
                "pre_probability": 0.5,
                "actual_trade_count": 2,
                "staleness_seconds": 1.0,
                "liquidity": 10.0,
                "game_time_seconds": 600,
                "signed_margin": 0,
                "down": 1.0,
                "distance": 10,
                "field_position": 75,
                "shape_classification": "GRADUAL",
                "claim_boundary_dense": "PRELIMINARY_SOURCE_TIME_ONLY",
                "eligible": True,
            }
            for game, effect in (("game-1", 0.01), ("game-2", -0.02))
            for venue in ("polymarket", "kalshi")
            for horizon in (10, 60)
        ]
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(root: Path, relative: str, value: object) -> tuple[str, str]:
    payload = _canonical(value)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return relative, _sha(payload)


def _write_parquet(
    root: Path, relative: str, frame: pd.DataFrame
) -> dict[str, object]:
    buffer = BytesIO()
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        buffer,
        compression="zstd",
    )
    payload = buffer.getvalue()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "object_path": relative,
        "object_sha256": _sha(payload),
        "byte_length": len(payload),
        "row_count": len(frame),
    }


def _sealed_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    market_root = tmp_path / "market"
    dense_root = tmp_path / "dense"
    factor = _eligible().iloc[[0]].copy()
    factor["predicate_nonexcluded_hit"] = True
    factor["factor_eligible"] = True
    factor["claim_boundary"] = "PRELIMINARY_SOURCE_TIME_ONLY"
    factor["pre_quarter"] = 5
    factor_ref = _write_parquet(
        market_root, "objects/factor.parquet", factor
    )
    market_game = {"stages": {"factor_observations": factor_ref}}
    market_game_path, market_game_sha = _write_json(
        market_root, "single-game/game-1/manifest.json", market_game
    )
    market_batch = {
        "game_count": 1,
        "final_holdout_access": "CLOSED",
        "games": [
            {
                "game_id": "game-1",
                "manifest_path": market_game_path,
                "manifest_sha256": market_game_sha,
            }
        ],
    }
    market_batch_path, _ = _write_json(
        market_root, "batch.json", market_batch
    )

    dense = factor[
        [
            "observation_id",
            "game_id",
            "episode_id",
            "logical_market_id",
            "outcome",
            "venue",
            "market_family",
            "horizon_seconds",
            "effect",
            "game_time_seconds",
            "signed_margin",
            "down",
            "distance",
            "field_position",
            "shape_classification",
        ]
    ].copy()
    dense["outcome_orientation"] = dense["outcome"]
    dense["pre_probability"] = 0.5
    dense["bin_last_trade_price"] = 0.51
    dense["cumulative_change"] = dense["effect"]
    dense["incremental_change"] = dense["effect"]
    dense["actual_trade_count"] = 1
    dense["source_time_first_trade"] = pd.Timestamp(
        "2025-01-01T00:00:01Z"
    )
    dense["last_trade_source_time"] = pd.Timestamp(
        "2025-01-01T00:00:01Z"
    )
    dense["staleness_seconds"] = 0.0
    dense["liquidity"] = 1.0
    dense["order_ambiguous"] = False
    dense["contaminated"] = False
    dense["censored"] = False
    dense["censoring_status"] = "NOT_CENSORED"
    dense["forward_filled"] = False
    dense["observed"] = True
    dense["attrition_reason"] = "OBSERVED"
    dense["jump"] = dense["effect"]
    dense["jump_share"] = 1.0
    dense["possession"] = "AAA"
    dense["actor_beneficiary"] = "AAA"
    dense["control_eligible"] = True
    dense["claim_boundary"] = "PRELIMINARY_SOURCE_TIME_ONLY"
    dense_ref = _write_parquet(dense_root, "objects/paths.parquet", dense)
    dense_game = {"objects": {"paths": dense_ref}}
    dense_game_path, dense_game_sha = _write_json(
        dense_root, "single-game/game-1/manifest.json", dense_game
    )
    dense_batch = {
        "game_count": 1,
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
        "games": [
            {
                "game_id": "game-1",
                "manifest_path": dense_game_path,
                "manifest_sha256": dense_game_sha,
            }
        ],
    }
    dense_batch_path, _ = _write_json(dense_root, "batch.json", dense_batch)
    return (
        market_root,
        market_root / market_batch_path,
        dense_root,
        dense_root / dense_batch_path,
    )


def test_loader_joins_only_verified_development_sources_and_retains_quarter(
    tmp_path: Path,
) -> None:
    market_root, market_batch, dense_root, dense_batch = _sealed_source_fixture(
        tmp_path
    )
    eligible, source_hashes = load_development_eligible_paths(
        market_root=market_root,
        market_batch_index_path=market_batch,
        dense_root=dense_root,
        dense_batch_manifest_path=dense_batch,
        expected_game_count=1,
    )

    assert len(eligible) == 1
    assert eligible.iloc[0]["quarter"] == 5
    assert source_hashes


def test_historical_study_publication_is_content_addressed_and_holdout_closed(
    tmp_path: Path,
) -> None:
    tables = build_historical_factor_study(
        _eligible(), bootstrap_samples=20, bootstrap_seed=3
    )
    publication = publish_historical_study_tables(
        tables,
        output_root=tmp_path,
        source_manifest_sha256s=("sha256:" + "a" * 64,),
        development_game_count=153,
        bootstrap_samples=20,
        bootstrap_seed=3,
    )

    manifest = json.loads(publication.manifest_path.read_text())
    assert manifest["schema"] == "nfl_x13_historical_study_publication_v1"
    assert manifest["development_game_count"] == 153
    assert manifest["final_holdout_access"] == "CLOSED"
    assert manifest["holdout_reaction_accessed"] is False
    assert manifest["objects"]

    verified = verify_historical_study_publication(
        publication.manifest_path,
        root=tmp_path,
    )
    assert set(verified) == {
        "bootstrap_histogram",
        "cases",
        "ecdf",
        "game_effects",
        "histogram",
        "path_probabilities",
        "state_breakdown",
        "summary",
        "venue_diagnostics",
        "venue_pairs",
    }


def test_historical_study_publication_fails_closed_after_object_tamper(
    tmp_path: Path,
) -> None:
    tables = build_historical_factor_study(
        _eligible(), bootstrap_samples=10, bootstrap_seed=4
    )
    publication = publish_historical_study_tables(
        tables,
        output_root=tmp_path,
        source_manifest_sha256s=("sha256:" + "b" * 64,),
        development_game_count=153,
        bootstrap_samples=10,
        bootstrap_seed=4,
    )
    manifest = json.loads(publication.manifest_path.read_text())
    object_path = tmp_path / manifest["objects"][0]["object_path"]
    object_path.chmod(0o644)
    object_path.write_bytes(object_path.read_bytes() + b"tamper")

    with pytest.raises(HistoricalStudyPublicationError, match="SHA-256"):
        verify_historical_study_publication(publication.manifest_path, root=tmp_path)
