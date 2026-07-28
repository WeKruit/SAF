from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from prediction_market.sports.nfl_factor_lab_bundles import (
    FactorLabBundleError,
    bundle_artifact_refs,
    build_cross_game_bundle,
    build_single_game_bundle,
    mirror_bundle,
    verify_cross_game_bundle,
    verify_single_game_bundle,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
GAME = "2025_14_DAL_DET"


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def head_object(self, key: str) -> dict[str, object] | None:
        if key not in self.objects:
            return None
        return {
            "byte_size": len(self.objects[key]),
            "metadata": self.metadata[key],
        }

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None:
        self.objects[key] = path.read_bytes()
        self.metadata[key] = dict(metadata)

    def read_object_chunks(self, key: str, *, chunk_size: int = 1024 * 1024):
        yield self.objects[key]


def _single_game_stages(game_id: str = GAME) -> dict[str, pd.DataFrame]:
    common = {"game_id": [game_id]}
    return {
        "sports_row_disposition": pd.DataFrame(
            {
                **common,
                "raw_play_id": ["10"],
                "canonical_play_id": ["10"],
                "episode_id": ["episode-10"],
                "factor_eligible": [True],
            }
        ),
        "canonical_events": pd.DataFrame(
            {**common, "play_id": ["10"], "event_type": ["pass"]}
        ),
        "finalized_episodes": pd.DataFrame(
            {**common, "episode_id": ["episode-10"], "episode_type": ["pass"]}
        ),
        "stat_ledger": pd.DataFrame(
            {**common, "through_play_id": ["10"], "home_score": [0]}
        ),
        "pit_features": pd.DataFrame(
            {**common, "play_id": ["10"], "history_game_count": [8]}
        ),
        "contract_inventory": pd.DataFrame(
            {
                **common,
                "logical_market_id": ["winner"],
                "outcome": ["DET"],
            }
        ),
        "market_observations": pd.DataFrame(
            {
                **common,
                "observation_id": ["trade-1"],
                "logical_market_id": ["winner"],
                "outcome": ["DET"],
                "price": [0.5],
            }
        ),
        "market_cleaning_audit": pd.DataFrame(
            {
                **common,
                "audit_row_id": ["market-audit-1"],
                "observation_id": ["trade-1"],
                "included": [True],
            }
        ),
        "game_time_intervals": pd.DataFrame(
            {
                **common,
                "interval_id": ["game-interval-1"],
                "episode_id": ["episode-10"],
                "delay_seconds": [0],
            }
        ),
        "market_time_intervals": pd.DataFrame(
            {
                **common,
                "interval_id": ["market-interval-1"],
                "observation_id": ["trade-1"],
            }
        ),
        "source_timeline": pd.DataFrame(
            {
                **common,
                "timeline_id": ["G:episode-10"],
                "interval_id": ["game-interval-1"],
                "layer": ["G"],
                "delay_seconds": [0],
            }
        ),
        "timeline_audit": pd.DataFrame(
            {
                **common,
                "timeline_audit_id": ["timeline-audit-1"],
                "timeline_id": ["G:episode-10"],
            }
        ),
        "association_attrition": pd.DataFrame(
            {
                **common,
                "association_id": ["assoc-1"],
                "episode_id": ["episode-10"],
                "logical_market_id": ["winner"],
                "outcome": ["DET"],
                "status": ["OBSERVED"],
            }
        ),
        "reaction_paths": pd.DataFrame(
            {
                **common,
                "episode_id": ["episode-10"],
                "logical_market_id": ["winner"],
                "outcome": ["DET"],
                "venue": ["polymarket"],
                "market_family": ["moneyline"],
                "horizon_seconds": [30],
            }
        ),
        "factor_observations": pd.DataFrame(
            {
                **common,
                "factor_id": ["NFL.EVENT.PASS"],
                "episode_id": ["episode-10"],
                "logical_market_id": ["winner"],
                "outcome": ["DET"],
                "venue": ["polymarket"],
                "market_family": ["moneyline"],
                "horizon_seconds": [30],
            }
        ),
        "factor_exclusion_audit": pd.DataFrame(
            {
                **common,
                "audit_id": ["factor-audit-1"],
                "decision": ["eligible"],
            }
        ),
        "single_game_factor_results": pd.DataFrame(
            {
                **common,
                "factor_id": ["NFL.EVENT.PASS"],
                "market_family": ["moneyline"],
                "horizon_seconds": [30],
                "point_estimate": [0.01],
            }
        ),
    }


def _cross_game_stages() -> dict[str, pd.DataFrame]:
    factor_key = {
        "factor_id": "NFL.EVENT.PASS",
        "market_family": "moneyline",
        "horizon_seconds": 30,
    }
    by_game = [
        {**factor_key, "game_id": game_id} for game_id in X13_GAME_IDS
    ]
    return {
        "cross_game_factor_results": pd.DataFrame([factor_key]),
        "game_contribution": pd.DataFrame(by_game),
        "leave_one_game_out": pd.DataFrame(by_game),
        "venue_consistency": pd.DataFrame(
            [{**factor_key, "venue": "polymarket"}]
        ),
        "expert_review_queue": pd.DataFrame(
            [{**factor_key, "decision": "PENDING"}]
        ),
    }


def test_single_game_bundle_is_content_addressed_and_semantically_verified(
    tmp_path: Path,
) -> None:
    first = build_single_game_bundle(
        game_id=GAME,
        stage_frames=_single_game_stages(),
        source_hashes=(HASH_A,),
        factor_registry_sha256=HASH_B,
        code_lock_sha256=HASH_A,
        output_root=tmp_path,
    )
    second = build_single_game_bundle(
        game_id=GAME,
        stage_frames=_single_game_stages(),
        source_hashes=(HASH_A,),
        factor_registry_sha256=HASH_B,
        code_lock_sha256=HASH_A,
        output_root=tmp_path,
    )

    assert first == second
    verified = verify_single_game_bundle(first.manifest_path)
    assert verified.game_id == GAME
    assert set(verified.stage_paths) == set(_single_game_stages())

    path = verified.stage_paths["canonical_events"]
    frame = pd.read_parquet(path)
    frame.loc[0, "event_type"] = "rush"
    frame.to_parquet(path, index=False)
    with pytest.raises(FactorLabBundleError, match="hash|semantic"):
        verify_single_game_bundle(first.manifest_path)


def test_bundle_verifier_preserves_nullable_arrow_identity(
    tmp_path: Path,
) -> None:
    stages = _single_game_stages()
    stages["source_timeline"]["nullable_source_order"] = pd.Series(
        [pd.NA],
        dtype="Int64",
    )

    published = build_single_game_bundle(
        game_id=GAME,
        stage_frames=stages,
        source_hashes=(HASH_A,),
        factor_registry_sha256=HASH_B,
        code_lock_sha256=HASH_A,
        output_root=tmp_path,
    )

    assert verify_single_game_bundle(published.manifest_path) == published


def test_cross_game_bundle_consumes_verified_single_game_manifests_only(
    tmp_path: Path,
) -> None:
    singles = [
        build_single_game_bundle(
            game_id=game_id,
            stage_frames=_single_game_stages(game_id),
            source_hashes=(HASH_A,),
            factor_registry_sha256=HASH_B,
            code_lock_sha256=HASH_A,
            output_root=tmp_path / "single",
        )
        for game_id in X13_GAME_IDS
    ]
    cross = build_cross_game_bundle(
        single_game_manifests=tuple(
            single.manifest_path for single in singles
        ),
        cross_game_frames={
            "cross_game_factor_results": pd.DataFrame(
                {
                    "factor_id": [
                        "NFL.EVENT.PASS",
                        "NFL.EVENT.REGISTERED_DATA_GAP",
                    ],
                    "market_family": ["moneyline", "moneyline"],
                    "horizon_seconds": [30, 30],
                    "point_estimate": [0.01, float("nan")],
                }
            ),
            "game_contribution": pd.DataFrame(
                [
                    {
                        "factor_id": "NFL.EVENT.PASS",
                        "market_family": "moneyline",
                        "horizon_seconds": 30,
                        "game_id": game_id,
                        "contribution": 1.0,
                    }
                    for game_id in X13_GAME_IDS
                ]
            ),
            "leave_one_game_out": pd.DataFrame(
                [
                    {
                        "factor_id": "NFL.EVENT.PASS",
                        "market_family": "moneyline",
                        "horizon_seconds": 30,
                        "game_id": game_id,
                        "estimate": float("nan"),
                    }
                    for game_id in X13_GAME_IDS
                ]
            ),
            "venue_consistency": pd.DataFrame(
                {
                    "factor_id": ["NFL.EVENT.PASS"],
                    "market_family": ["moneyline"],
                    "horizon_seconds": [30],
                    "venue": ["polymarket"],
                    "sign": [1],
                }
            ),
            "expert_review_queue": pd.DataFrame(
                {
                    "factor_id": ["NFL.EVENT.PASS"],
                    "market_family": ["moneyline"],
                    "horizon_seconds": [30],
                    "decision": ["PENDING"],
                }
            ),
        },
        batch_spec_sha256=HASH_A,
        factor_registry_sha256=HASH_B,
        code_lock_sha256=HASH_A,
        output_root=tmp_path / "cross",
    )

    verified = verify_cross_game_bundle(cross.manifest_path)
    assert verified.game_ids == X13_GAME_IDS
    assert verified.holdout_reaction_accessed is False

    singles[0].manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FactorLabBundleError):
        verify_cross_game_bundle(cross.manifest_path)


def test_cross_game_bundle_rejects_a_different_code_lock_than_singles(
    tmp_path: Path,
) -> None:
    singles = tuple(
        build_single_game_bundle(
            game_id=game_id,
            stage_frames=_single_game_stages(game_id),
            source_hashes=(HASH_A,),
            factor_registry_sha256=HASH_B,
            code_lock_sha256=HASH_A,
            output_root=tmp_path / "single",
        )
        for game_id in X13_GAME_IDS
    )

    with pytest.raises(FactorLabBundleError, match="code lock"):
        build_cross_game_bundle(
            single_game_manifests=tuple(
                single.manifest_path for single in singles
            ),
            cross_game_frames=_cross_game_stages(),
            batch_spec_sha256=HASH_A,
            factor_registry_sha256=HASH_B,
            code_lock_sha256=HASH_B,
            output_root=tmp_path / "cross",
        )


def test_single_game_bundle_rejects_cross_stage_foreign_key_drift(
    tmp_path: Path,
) -> None:
    stages = _single_game_stages()
    stages["reaction_paths"].loc[0, "episode_id"] = "missing-episode"

    with pytest.raises(FactorLabBundleError, match="reaction_paths.*episode"):
        build_single_game_bundle(
            game_id=GAME,
            stage_frames=stages,
            source_hashes=(HASH_A,),
            factor_registry_sha256=HASH_B,
            code_lock_sha256=HASH_A,
            output_root=tmp_path,
        )


def test_cross_game_bundle_requires_the_frozen_twenty_games(
    tmp_path: Path,
) -> None:
    single = build_single_game_bundle(
        game_id=GAME,
        stage_frames=_single_game_stages(),
        source_hashes=(HASH_A,),
        factor_registry_sha256=HASH_B,
        code_lock_sha256=HASH_A,
        output_root=tmp_path / "single",
    )

    with pytest.raises(FactorLabBundleError, match="frozen 20"):
        build_cross_game_bundle(
            single_game_manifests=(single.manifest_path,),
            cross_game_frames={
                "cross_game_factor_results": pd.DataFrame(
                    {
                        "factor_id": ["NFL.EVENT.PASS"],
                        "market_family": ["moneyline"],
                        "horizon_seconds": [30],
                    }
                ),
                "game_contribution": pd.DataFrame(
                    {
                        "factor_id": ["NFL.EVENT.PASS"],
                        "market_family": ["moneyline"],
                        "horizon_seconds": [30],
                        "game_id": [GAME],
                    }
                ),
                "leave_one_game_out": pd.DataFrame(
                    {
                        "factor_id": ["NFL.EVENT.PASS"],
                        "market_family": ["moneyline"],
                        "horizon_seconds": [30],
                        "game_id": [GAME],
                    }
                ),
                "venue_consistency": pd.DataFrame(
                    {
                        "factor_id": ["NFL.EVENT.PASS"],
                        "market_family": ["moneyline"],
                        "horizon_seconds": [30],
                        "venue": ["polymarket"],
                    }
                ),
                "expert_review_queue": pd.DataFrame(
                    {
                        "factor_id": ["NFL.EVENT.PASS"],
                        "market_family": ["moneyline"],
                        "horizon_seconds": [30],
                    }
                ),
            },
            batch_spec_sha256=HASH_A,
            factor_registry_sha256=HASH_B,
            code_lock_sha256=HASH_A,
            output_root=tmp_path / "cross",
        )


def test_single_game_bundle_maps_to_frozen_s3_layout_and_readback(
    tmp_path: Path,
) -> None:
    single = build_single_game_bundle(
        game_id=GAME,
        stage_frames=_single_game_stages(),
        source_hashes=(HASH_A,),
        factor_registry_sha256=HASH_B,
        code_lock_sha256=HASH_A,
        output_root=tmp_path / "single",
    )
    refs = bundle_artifact_refs(
        single,
        project_root=tmp_path,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
    )
    store = MemoryStore()

    mirrored = mirror_bundle(
        single,
        project_root=tmp_path,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
        object_store=store,
    )

    assert mirrored == refs
    assert len(refs) == len(_single_game_stages()) + 1
    assert any(
        f"research/nfl/x13/single-game/{GAME}/manifests/"
        in ref.s3_key
        for ref in refs
    )
    assert all(ref.s3_key in store.objects for ref in refs)
