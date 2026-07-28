from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.sports.nfl_x13_matched_reaction_publication import (
    EXPECTED_EXACT153_GAMES,
    MatchedReactionPublicationError,
    build_partitioned_matched_reaction_tables,
    publish_matched_reaction_full,
)
from prediction_market.sports.nfl_x13_matched_reaction import (
    MatchedReactionSpecV1,
    build_matched_reaction_tables,
)


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(root: Path, relative: str, value: dict[str, object]) -> tuple[str, str]:
    payload = _bytes(value)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return relative, _sha(payload)


def _write_parquet(root: Path, relative: str, frame: pd.DataFrame) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    payload = path.read_bytes()
    return {
        "object_path": relative,
        "object_sha256": _sha(payload),
        "byte_length": len(payload),
        "row_count": len(frame),
    }


def _factor_frame(game_id: str) -> pd.DataFrame:
    digest = "sha256:" + "a" * 64
    return pd.DataFrame(
        [
            {
                "factor_observation_id": f"{game_id}:factor",
                "factor_id": "NFL.TEST.EVENT",
                "factor_version": "v1",
                "game_id": game_id,
                "episode_id": "episode-1",
                "logical_market_id": "kalshi:winner",
                "outcome": "AAA",
                "venue": "kalshi",
                "market_family": "winner",
                "horizon_seconds": 1,
                "delay_seconds": 0,
                "signed_price_change": 0.01,
                "maximum_excursion": 0.01,
                "pre_event_actual_trade": 0.50,
                "event_source_time_utc": "2025-09-01T00:01:00Z",
                "game_seconds_remaining": 3600,
                "score_margin_focal": 0,
                "staleness_seconds": 1,
                "event_eligible": True,
                "episode_audit_only": False,
                "episode_is_score": False,
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
                "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
                "sports_source_hashes": [digest],
                "market_source_hashes": [digest],
            }
        ]
    )


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "venue": "kalshi",
                "logical_market_id": "kalshi:winner",
                "outcome": "AAA",
                "kind": "trade",
                "source_start_utc": "2025-09-01T00:00:00Z",
                "size": 10.0,
                "primary_path_eligible": True,
            }
        ]
    )


def _exact153_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "input"
    market = _write_parquet(root, "objects/market.parquet", _market_frame())
    games: list[dict[str, object]] = []
    for ordinal in range(EXPECTED_EXACT153_GAMES):
        game_id = f"game-{ordinal:03d}"
        factor = _write_parquet(
            root, f"objects/{game_id}.factor.parquet", _factor_frame(game_id)
        )
        manifest = {
            "game_id": game_id,
            "stages": {
                "factor_observations": factor,
                "actual_market_observations": market,
            },
        }
        manifest_payload = _bytes(manifest)
        manifest_sha = _sha(manifest_payload)
        hexadecimal = manifest_sha.removeprefix("sha256:")
        relative = f"single-game/{game_id}/manifests/sha256/{hexadecimal[:2]}/{hexadecimal}.manifest.json"
        _write_json(root, relative, manifest)
        games.append(
            {
                "game_id": game_id,
                "manifest_path": relative,
                "manifest_sha256": manifest_sha,
            }
        )
    material = {
        "schema": "nfl_expansion_development_market_batch_index_v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": EXPECTED_EXACT153_GAMES,
        "publication_gate": "PASS",
        "final_holdout_access": "CLOSED",
        "games": games,
    }
    batch = {**material, "batch_sha256": _sha(_bytes(material))}
    payload = _bytes(batch)
    sha = _sha(payload)
    hexadecimal = sha.removeprefix("sha256:")
    relative = f"batches/manifests/sha256/{hexadecimal[:2]}/{hexadecimal}.batch-index.json"
    _write_json(root, relative, batch)
    return root, root / relative


def test_full_publication_is_exact153_atomic_and_refuses_the_preflight_gate(tmp_path: Path) -> None:
    input_root, batch_index = _exact153_fixture(tmp_path)
    blocked_root = tmp_path / "blocked"

    with pytest.raises(MatchedReactionPublicationError, match="feasibility gate"):
        publish_matched_reaction_full(
            input_root=input_root,
            batch_index_path=batch_index,
            output_root=blocked_root,
            bootstrap_samples=1,
            max_projected_peak_bytes=1,
        )
    assert not (blocked_root / "manifests").exists()

    publication = publish_matched_reaction_full(
        input_root=input_root,
        batch_index_path=batch_index,
        output_root=tmp_path / "published",
        bootstrap_samples=1,
    )

    manifest = json.loads(publication.manifest_path.read_text("utf-8"))
    assert publication.selected_game_count == EXPECTED_EXACT153_GAMES
    assert len(publication.selected_game_ids) == EXPECTED_EXACT153_GAMES
    assert manifest["publication_mode"] == "EXACT153_FULL"
    assert manifest["full_run_status"] == "COMPLETED"
    assert manifest["full_run_started"] is True
    assert manifest["final_holdout_access"] == "CLOSED"
    assert manifest["holdout_reaction_accessed"] is False


def _matching_row(
    observation_id: str,
    *,
    factor_id: str = "NFL.TEST.EVENT",
    game_id: str,
    episode_id: str,
    market_family: str,
    routine: bool,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "factor_id": factor_id,
        "factor_version": "v1",
        "factor_family": "test",
        "game_id": game_id,
        "episode_id": episode_id,
        "logical_market_id": f"{market_family}:market",
        "venue": "kalshi",
        "market_family": market_family,
        "outcome_orientation": "AAA",
        "horizon_seconds": 1,
        "delay_seconds": 0,
        "effect": 0.02,
        "maximum_excursion": 0.02,
        "pre_probability": 0.5,
        "relative_time_seconds": 3000,
        "score_margin": 0,
        "staleness_seconds": 1.0,
        "liquidity": 10.0,
        "routine_control_eligible": routine,
        "hit_factor_ids": ("NFL.CONTROL.ROUTINE",) if routine else (factor_id,),
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "source_hashes": ("sha256:" + "b" * 64,),
    }


def test_partitioned_matcher_matches_v1_results_and_bootstrap_exactly() -> None:
    treated = pd.DataFrame(
        [
            _matching_row(
                "treated-winner",
                game_id="treated-game",
                episode_id="treated-winner",
                market_family="winner",
                routine=False,
            ),
            _matching_row(
                "treated-spread",
                game_id="treated-game",
                episode_id="treated-spread",
                market_family="spread",
                routine=False,
            ),
        ]
    )
    controls = pd.DataFrame(
        [
            _matching_row(
                f"{market_family}-control-{index}",
                game_id=f"control-game-{index}",
                episode_id=f"control-{index}",
                market_family=market_family,
                routine=True,
            )
            for market_family in ("winner", "spread")
            for index in range(10)
        ]
    )
    spec = MatchedReactionSpecV1(bootstrap_samples=5)

    direct = build_matched_reaction_tables(treated, controls, spec=spec)
    partitioned = build_partitioned_matched_reaction_tables(
        treated, controls, spec=spec
    )

    assert partitioned.matched_controls["match_scope"].eq(
        "CROSS_GAME_FALLBACK"
    ).all()
    assert set(partitioned.matched_controls["control_game_id"]) == {
        f"control-game-{index}" for index in range(10)
    }
    assert partitioned.matched_controls.groupby(
        ["treated_observation_id", "market_family"]
    ).size().eq(10).all()

    for field in (
        "matched_controls",
        "match_audit",
        "match_balance",
        "source_time_paths",
        "single_game_results",
        "cross_game_results",
        "venue_consistency",
        "manifest",
    ):
        expected = getattr(direct, field).sort_index(axis=1)
        actual = getattr(partitioned, field).sort_index(axis=1)
        if not expected.empty:
            expected = expected.sort_values(list(expected.columns), kind="mergesort").reset_index(drop=True)
            actual = actual.sort_values(list(actual.columns), kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
