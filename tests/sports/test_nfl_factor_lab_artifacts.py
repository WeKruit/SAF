from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.sports.nfl_factor_lab_artifacts import (
    FactorLabArtifactError,
    load_factor_discovery_results,
    load_human_review_queue,
    load_sports_feature_frame,
    publish_factor_discovery_bundle,
    publish_sports_feature_view,
    verify_factor_discovery_bundle,
    verify_sports_feature_view,
)
from prediction_market.sports.nfl_factor_lab_types import SportsFeatureViewV1
from prediction_market.sports.nfl_factor_lab_features import (
    build_sports_feature_view,
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _view() -> SportsFeatureViewV1:
    row = {
        "game_id": "2025_01_AAA_BBB",
        "play_id": 10,
        "order_sequence": 10,
        "season": 2025,
        "season_type": "REG",
        "week": 1,
        "time_of_day": "2025-09-10T17:10:00Z",
        "home_team": "BBB",
        "away_team": "AAA",
        "posteam": "AAA",
        "defteam": "BBB",
        "qtr": 1,
        "quarter_seconds_remaining": 600,
        "half_seconds_remaining": 1500,
        "game_seconds_remaining": 3300,
        "posteam_score": 0,
        "defteam_score": 0,
        "total_home_score": 0,
        "total_away_score": 0,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 75,
        "goal_to_go": 0,
        "fixed_drive": 1,
        "desc": "pass",
        "play_type": "pass",
        "pass_attempt": 1,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
    }
    history = {
        **row,
        "game_id": "2024_01_AAA_BBB",
        "play_id": 1,
        "order_sequence": 1,
        "season": 2024,
        "time_of_day": "2024-09-01T18:00:00Z",
    }
    return build_sports_feature_view(
        selected_pbp_rows=[row],
        finalized_episode_rows=[
            {
                "episode_id": "episode-10",
                "game_id": "2025_01_AAA_BBB",
                "play_ids": ["10"],
                "episode_type": "pass",
                "nullified": False,
                "audit_only": False,
            }
        ],
        historical_pbp_rows=[history],
        kickoff_times_utc={
            "2025_01_AAA_BBB": "2025-09-10T17:00:00Z"
        },
        source_hashes={"pbp": _hash("a")},
    )


def test_feature_view_publication_is_immutable_and_verified(
    tmp_path: Path,
) -> None:
    published = publish_sports_feature_view(
        _view(),
        code_lock_sha256=_hash("b"),
        factor_registry_sha256=_hash("c"),
        output_root=tmp_path,
    )
    repeated = publish_sports_feature_view(
        _view(),
        code_lock_sha256=_hash("b"),
        factor_registry_sha256=_hash("c"),
        output_root=tmp_path,
    )
    verified = verify_sports_feature_view(published.manifest_path)

    assert repeated == published
    assert verified.rows_sha256 == _view().rows_sha256
    assert verified.row_count == 1
    assert len(load_sports_feature_frame(published.manifest_path)) == 1

    published.data_path.write_bytes(b"tampered")
    with pytest.raises(FactorLabArtifactError, match="differs"):
        verify_sports_feature_view(published.manifest_path)


def test_discovery_bundle_contains_pending_human_queue_and_fails_closed(
    tmp_path: Path,
) -> None:
    result = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.PASS",
                "factor_version": "v1",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "discovery_status": "AWAITING_EXPERT_REVIEW",
                "point_estimate": 0.01,
            }
        ]
    )
    published = publish_factor_discovery_bundle(
        result,
        factor_registry_sha256=_hash("b"),
        feature_view_sha256=_hash("c"),
        code_lock_sha256=_hash("e"),
        source_hashes=(_hash("d"),),
        output_root=tmp_path,
    )
    verified = verify_factor_discovery_bundle(published.manifest_path)
    queue = json.loads(published.review_queue_path.read_bytes())

    assert verified.result_row_count == 1
    assert queue["reviews"][0]["decision"] == "PENDING"
    assert queue["holdout_reaction_accessed"] is False
    assert len(load_factor_discovery_results(published.manifest_path)) == 1
    assert load_human_review_queue(published.manifest_path) == queue

    published.review_queue_path.write_bytes(b"{}")
    with pytest.raises(FactorLabArtifactError, match="differs"):
        verify_factor_discovery_bundle(published.manifest_path)
