"""Synthetic sealed-batch coverage for the dense-reaction publisher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.sports.nfl_x13_dense_reaction_publication import (
    DenseReactionPublicationError,
    _canonical_episode_features,
    publish_dense_reaction_batch,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_json(root: Path, relative: str, value: dict[str, object]) -> tuple[str, str]:
    payload = _canonical_bytes(value)
    digest = _sha(payload)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return relative, digest


def _write_content_addressed_json(
    root: Path,
    *,
    directory: str,
    suffix: str,
    value: dict[str, object],
) -> tuple[str, str]:
    payload = _canonical_bytes(value)
    digest = _sha(payload)
    hexadecimal = digest.removeprefix("sha256:")
    relative = f"{directory}/sha256/{hexadecimal[:2]}/{hexadecimal}.{suffix}"
    return _write_json(root, relative, value)


def _write_parquet(
    root: Path, relative: str, frame: pd.DataFrame) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return {
        "object_path": relative,
        "object_sha256": _sha_file(path),
        "byte_length": path.stat().st_size,
        "row_count": len(frame),
        "format": "parquet",
    }


def _market_frames(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contracts = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "logical_market_id": f"kalshi:{game_id}:winner",
                "outcome": "AAA",
                "venue": "kalshi",
                "family": "winner",
            }
        ]
    )
    actual = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "observation_id": f"{game_id}:pre",
                "logical_market_id": f"kalshi:{game_id}:winner",
                "outcome": "AAA",
                "venue": "kalshi",
                "kind": "trade",
                "price": 0.50,
                "size": 10.0,
                "source_start_utc": "2025-09-01T00:00:00Z",
                "source_end_utc": "2025-09-01T00:00:00Z",
                "primary_path_eligible": True,
                "provenance": "observed",
            },
            {
                "game_id": game_id,
                "observation_id": f"{game_id}:post",
                "logical_market_id": f"kalshi:{game_id}:winner",
                "outcome": "AAA",
                "venue": "kalshi",
                "kind": "trade",
                "price": 0.55,
                "size": 10.0,
                "source_start_utc": "2025-09-01T00:00:03Z",
                "source_end_utc": "2025-09-01T00:00:03Z",
                "primary_path_eligible": True,
                "provenance": "observed",
            },
        ]
    )
    timeline = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "episode_id": "episode-1",
                "delay_seconds": 0,
                "interval_start_utc": "2025-09-01T00:00:01Z",
                "source_time_utc": "2025-09-01T00:00:01Z",
                "interval_end_utc": "2025-09-01T00:00:02Z",
                "layer": "G",
            },
            {
                "game_id": game_id,
                "episode_id": "episode-2",
                "delay_seconds": 0,
                "source_time_utc": "2025-09-01T00:00:04Z",
                "interval_start_utc": "2025-09-01T00:00:04Z",
                "interval_end_utc": "2025-09-01T00:00:05Z",
                "layer": "G",
            },
            {
                "game_id": game_id,
                "episode_id": "episode-3",
                "delay_seconds": 0,
                "source_time_utc": "2025-09-01T00:00:07Z",
                "interval_start_utc": "2025-09-01T00:00:07Z",
                "interval_end_utc": "2025-09-01T00:00:08Z",
                "layer": "G",
            },
        ]
    )
    return contracts, actual, timeline


def _feature_frame(game_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (episode_id, source_time, is_score) in enumerate(
        (
            ("episode-1", "2025-09-01T00:00:01Z", False),
            ("episode-2", "2025-09-01T00:00:04Z", False),
            ("episode-3", "2025-09-01T00:00:07Z", True),
        ),
        start=1,
    ):
        rows.append(
            {
                "game_id": game_id,
                "play_id": f"{game_id}-play-{index}",
                "episode_id": episode_id,
                "order_sequence": index,
                "source_time_utc": source_time,
                "game_seconds_remaining": 3600 - index,
                "score_margin_focal": 0,
                "yardline_100": 50,
                "possession_team": "AAA",
                "focal_team": "AAA",
                "down": 1,
                "distance": 10,
                "event_eligible": True,
                "episode_audit_only": False,
                "episode_is_score": is_score,
                "episode_is_turnover": False,
                "penalty": False,
                "replay_or_challenge": False,
                "timeout": False,
                "safety": False,
                "punt_blocked": False,
                "return_touchdown": False,
                "fumble_lost": False,
                "interception": False,
                "fourth_down_failed": False,
            }
        )
    return pd.DataFrame(rows)


def test_episode_prestate_uses_first_row_and_salience_uses_all_rows() -> None:
    """A TD + try retains pre-state but joins at final episode source time."""

    features = _feature_frame("2025_01_AAA_BBB")
    first = features.iloc[0].copy()
    later = first.copy()
    later["play_id"] = "2025_01_AAA_BBB-play-1-try"
    later["order_sequence"] = 99
    later["source_time_utc"] = "2025-09-01T00:00:02Z"
    later["game_seconds_remaining"] = 3500
    later["score_margin_focal"] = 7
    later["episode_is_score"] = True
    source = pd.concat([features, pd.DataFrame([later])], ignore_index=True)

    canonical = _canonical_episode_features(source, game_id="2025_01_AAA_BBB")
    episode = canonical.loc[canonical["episode_id"].eq("episode-1")].iloc[0]

    assert episode["source_time_utc"] == "2025-09-01T00:00:02Z"
    assert episode["game_seconds_remaining"] == 3599
    assert episode["score_margin_focal"] == 0
    assert bool(episode["episode_is_score"]) is True


def _make_batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    market_root = tmp_path / "market"
    feature_root = tmp_path / "features"
    market_games: list[dict[str, object]] = []
    feature_games: list[dict[str, object]] = []
    for game_id in ("2025_01_AAA_BBB", "2025_02_CCC_DDD"):
        contracts, actual, timeline = _market_frames(game_id)
        stages = {
            "contract_inventory": _write_parquet(
                market_root, f"objects/{game_id}.contracts.parquet", contracts
            ),
            "actual_market_observations": _write_parquet(
                market_root, f"objects/{game_id}.actual.parquet", actual
            ),
            "layer_g_source_timeline": _write_parquet(
                market_root, f"objects/{game_id}.timeline.parquet", timeline
            ),
        }
        market_material = {
            "schema": "nfl_expansion_development_market_single_game_manifest_v1",
            "experiment_id": "X-13",
            "cohort": "development",
            "game_id": game_id,
            "publication_gate": "PASS",
            "stages": stages,
        }
        market_manifest = {
            **market_material,
            "bundle_sha256": _sha(_canonical_bytes(market_material)),
        }
        market_relative, market_sha = _write_content_addressed_json(
            market_root,
            directory=f"single-game/{game_id}/manifests",
            suffix="manifest.json",
            value=market_manifest,
        )
        market_games.append(
            {
                "game_id": game_id,
                "manifest_path": market_relative,
                "manifest_sha256": market_sha,
                "bundle_sha256": market_manifest["bundle_sha256"],
                "stage_count": len(stages),
            }
        )

        feature = _write_parquet(
            feature_root, f"objects/{game_id}.features.parquet", _feature_frame(game_id)
        )
        feature["rows_sha256"] = feature["object_sha256"]
        feature_material = {
            "schema": "nfl_expansion_development_single_game_feature_manifest_v1",
            "cohort": "development",
            "game_id": game_id,
            "publication_gate": "PASS",
            "reaction_access": "CLOSED",
            "market_data_read": False,
            "feature_view": feature,
        }
        feature_manifest = {
            **feature_material,
            "bundle_sha256": _sha(_canonical_bytes(feature_material)),
        }
        feature_relative, feature_sha = _write_content_addressed_json(
            feature_root,
            directory=f"single-game/{game_id}/manifests",
            suffix="manifest.json",
            value=feature_manifest,
        )
        feature_games.append(
            {
                "game_id": game_id,
                "manifest_path": feature_relative,
                "manifest_sha256": feature_sha,
                "bundle_sha256": feature_manifest["bundle_sha256"],
                "object_sha256": feature["object_sha256"],
                "row_count": feature["row_count"],
            }
        )

    market_material = {
        "schema": "nfl_expansion_development_market_batch_index_v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": 2,
        "publication_gate": "PASS",
        "final_holdout_access": "CLOSED",
        "games": market_games,
    }
    market_batch = {**market_material, "batch_sha256": _sha(_canonical_bytes(market_material))}
    market_batch_relative, _ = _write_content_addressed_json(
        market_root,
        directory="batches/manifests",
        suffix="batch-index.json",
        value=market_batch,
    )

    feature_material = {
        "schema": "nfl_expansion_development_feature_batch_index_v1",
        "scope": "development-2",
        "cohort": "development",
        "requested_game_count": 2,
        "published_game_count": 2,
        "failed_game_count": 0,
        "publication_gate": "PASS",
        "reaction_access": "CLOSED",
        "market_data_read": False,
        "games": feature_games,
    }
    feature_batch = {**feature_material, "batch_sha256": _sha(_canonical_bytes(feature_material))}
    feature_batch_relative, _ = _write_content_addressed_json(
        feature_root,
        directory="batches/manifests",
        suffix="batch-index.json",
        value=feature_batch,
    )
    return (
        market_root,
        market_root / market_batch_relative,
        feature_root,
        feature_root / feature_batch_relative,
    )


def test_publishes_verified_dense_paths_for_a_two_game_sealed_batch(tmp_path: Path) -> None:
    market_root, market_batch, feature_root, feature_batch = _make_batch_fixture(tmp_path)
    output_root = tmp_path / "publication"

    first = publish_dense_reaction_batch(
        market_root=market_root,
        market_batch_index_path=market_batch,
        feature_root=feature_root,
        feature_batch_index_path=feature_batch,
        output_root=output_root,
        expected_game_count=2,
    )
    second = publish_dense_reaction_batch(
        market_root=market_root,
        market_batch_index_path=market_batch,
        feature_root=feature_root,
        feature_batch_index_path=feature_batch,
        output_root=output_root,
        expected_game_count=2,
    )

    assert first.manifest_path == second.manifest_path
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.game_ids == ("2025_01_AAA_BBB", "2025_02_CCC_DDD")
    manifest = json.loads(first.manifest_path.read_text("utf-8"))
    assert manifest["final_holdout_access"] == "CLOSED"
    assert manifest["holdout_reaction_accessed"] is False
    assert len(manifest["games"]) == 2
    first_game = manifest["games"][0]
    lineage = json.loads(
        (output_root / first_game["objects"]["lineage"]["object_path"]).read_text("utf-8")
    )
    assert {row["kind"] for row in lineage["inputs"]} >= {
        "actual_market_observations",
        "contract_inventory",
        "layer_g_source_timeline",
        "sports_feature_view",
    }
    paths = pq.read_table(
        output_root / first_game["objects"]["paths"]["object_path"]
    ).to_pandas()
    assert set(paths["episode_id"]) == {"episode-1", "episode-2", "episode-3"}
    assert paths.loc[
        paths["episode_id"].eq("episode-1") & paths["horizon_seconds"].eq(2),
        "contaminated",
    ].eq(False).all()
    assert paths.loc[
        paths["episode_id"].eq("episode-1") & paths["horizon_seconds"].eq(5),
        "contaminated",
    ].eq(True).all()

    actual_path = market_root / "objects/2025_01_AAA_BBB.actual.parquet"
    actual_path.write_bytes(actual_path.read_bytes() + b"tampered")
    with pytest.raises(
        DenseReactionPublicationError, match="(byte length|SHA-256) mismatch"
    ):
        publish_dense_reaction_batch(
            market_root=market_root,
            market_batch_index_path=market_batch,
            feature_root=feature_root,
            feature_batch_index_path=feature_batch,
            output_root=output_root,
            expected_game_count=2,
        )
