from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from prediction_market.sports.nfl_x16_fact_publication import (
    publish_single_game_fact_review,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, *, dataset_id: str, object_path: Path) -> Path:
    payload = {
        "manifest_version": "v0",
        "dataset_id": dataset_id,
        "license_status": "research_only",
        "object_sha256": _sha(object_path),
        "native_object_path": str(object_path),
    }
    target = path / f"{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_publish_single_game_fact_review_is_verified_and_content_addressed(
    tmp_path: Path,
) -> None:
    pbp_path = tmp_path / "pbp.parquet"
    participation_path = tmp_path / "participation.parquet"
    pd.DataFrame(
        [
            {
                "game_id": "2025_14_DAL_DET",
                "play_id": 1,
                "order_sequence": 1,
                "home_team": "DET",
                "away_team": "DAL",
                "qtr": 1,
                "time": "15:00",
                "time_of_day": "2025-12-05T01:15:00Z",
                "play_type": "run",
                "play_type_nfl": "RUSH",
                "posteam": "DAL",
                "defteam": "DET",
                "posteam_score": 0,
                "defteam_score": 0,
                "total_home_score": 0,
                "total_away_score": 0,
                "down": 1,
                "ydstogo": 10,
                "yrdln": "DAL 25",
                "end_yard_line": "DAL 29",
                "yardline_100": 75,
                "yards_gained": 4,
                "rush_attempt": 1,
                "rusher_player_id": "00-rb",
                "rusher_player_name": "R.Back",
                "desc": "R.Back right guard to DAL 29 for 4 yards.",
            }
        ]
    ).to_parquet(pbp_path, index=False)
    pd.DataFrame(
        [
            {
                "nflverse_game_id": "2025_14_DAL_DET",
                "play_id": 1,
                "offense_players": "00-rb",
                "offense_names": "Running Back",
                "offense_positions": "RB",
                "offense_numbers": "20",
                "defense_players": "00-lb",
                "defense_names": "Line Backer",
                "defense_positions": "LB",
                "defense_numbers": "50",
                "n_offense": 1,
                "n_defense": 1,
            }
        ]
    ).to_parquet(participation_path, index=False)
    manifest_root = tmp_path / "source-manifests"
    manifest_root.mkdir()
    pbp_manifest = _manifest(
        manifest_root, dataset_id="DS-NFLVERSE", object_path=pbp_path
    )
    participation_manifest = _manifest(
        manifest_root,
        dataset_id="DS-NFLVERSE-PARTICIPATION",
        object_path=participation_path,
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "NFLFactorRegistryV3",
                "version": "v3-draft",
                "status": "DRAFT_EXPERT_REVIEW",
                "factors": [
                    {
                        "factor_id": "NFL.ACTION.RUN",
                        "version": "v1",
                        "status": "DRAFT",
                        "predicate": {
                            "field": "primary_action",
                            "operator": "EQ",
                            "value": "RUN",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = publish_single_game_fact_review(
        game_id="2025_14_DAL_DET",
        pbp_object_path=pbp_path,
        pbp_manifest_path=pbp_manifest,
        participation_object_path=participation_path,
        participation_manifest_path=participation_manifest,
        factor_registry_path=registry_path,
        output_root=tmp_path / "published",
    )
    second = publish_single_game_fact_review(
        game_id="2025_14_DAL_DET",
        pbp_object_path=pbp_path,
        pbp_manifest_path=pbp_manifest,
        participation_object_path=participation_path,
        participation_manifest_path=participation_manifest,
        factor_registry_path=registry_path,
        output_root=tmp_path / "published",
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest_path == second.manifest_path
    assert first.report_path.is_file()
    assert "Market data deliberately excluded" in first.report_path.read_text()
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["formal_experiment_registered"] is False
    assert manifest["market_data_read"] is False
    assert manifest["holdout_reaction_accessed"] is False
    assert manifest["source_row_count"] == 1
    assert manifest["source_row_preserved_count"] == 1
    assert {item["name"] for item in manifest["objects"]} >= {
        "canonical_factor_events",
        "factor_event_tags",
        "factor_event_players",
        "player_availability_events",
        "injury_evidence",
        "factor_hits",
        "factor_coverage_audit",
        "sports_row_reconciliation",
        "game_fact_review",
    }
