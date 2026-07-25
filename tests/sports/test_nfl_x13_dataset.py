from __future__ import annotations

import os
from pathlib import Path

import pytest

from prediction_market.sports.nfl_x13_dataset import (
    X13DatasetError,
    canonical_team_field_orientation,
    load_x13_source_bundle,
    summarize_frozen_sample,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


def _row(game_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": game_id,
        "play_id": 1,
        "order_sequence": 1,
        "home_team": "BUF",
        "away_team": "BAL",
        "posteam": "BAL",
        "qtr": 1,
        "touchdown": 0,
        "field_goal_result": None,
        "interception": 0,
        "fumble_lost": 0,
        "fourth_down_failed": 0,
        "replay_or_challenge": 0,
        "replay_or_challenge_result": None,
        "safety": 0,
        "return_touchdown": 0,
        "total_home_score": 0,
        "total_away_score": 0,
    }
    row.update(overrides)
    return row


def test_team_field_orientation_changes_with_possession_without_claiming_tracking() -> None:
    away = canonical_team_field_orientation(
        home_team="BUF",
        away_team="BAL",
        possession_team="BAL",
        yardline_100=75,
    )
    home = canonical_team_field_orientation(
        home_team="BUF",
        away_team="BAL",
        possession_team="BUF",
        yardline_100=40,
    )

    assert away == {
        "direction": "right",
        "field_coordinate_0_100": 25,
        "semantics": "canonical_team_end_orientation_not_tracking_coordinate",
    }
    assert home == {
        "direction": "left",
        "field_coordinate_0_100": 40,
        "semantics": "canonical_team_end_orientation_not_tracking_coordinate",
    }

    with pytest.raises(X13DatasetError, match="possession"):
        canonical_team_field_orientation(
            home_team="BUF",
            away_team="BAL",
            possession_team="NYJ",
            yardline_100=50,
        )


def test_sample_summary_uses_made_field_goals_and_failed_fourth_down_indicator() -> None:
    rows = [
        _row(
            "2025_01_BAL_BUF",
            touchdown=1,
            field_goal_result="made",
            fourth_down_failed=1,
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
            total_away_score=7,
        ),
        _row(
            "2025_01_BAL_BUF",
            play_id=2,
            order_sequence=2,
            field_goal_result="missed",
            interception=1,
            fumble_lost=1,
            safety=1,
            return_touchdown=1,
            qtr=5,
            total_home_score=7,
            total_away_score=7,
        ),
    ]

    summary = summarize_frozen_sample(rows)

    assert summary["row_count"] == 2
    assert summary["touchdowns"] == 1
    assert summary["made_field_goals"] == 1
    assert summary["interceptions"] == 1
    assert summary["lost_fumbles"] == 1
    assert summary["turnover_on_downs"] == 1
    assert summary["reviews"] == 1
    assert summary["reversed_reviews"] == 1
    assert summary["safeties"] == 1
    assert summary["return_touchdowns"] == 1
    assert summary["overtime_game_count"] == 1
    assert summary["tie_game_count"] == 1


@pytest.mark.skipif(
    "X13_RAW_ROOT" not in os.environ,
    reason="set X13_RAW_ROOT to run the byte-verified 2025 integration",
)
def test_real_frozen_bundle_contains_exact_registered_twenty() -> None:
    raw_root = Path(os.environ["X13_RAW_ROOT"])
    program_root = Path(__file__).resolve().parents[2]
    bundle = load_x13_source_bundle(
        program_root=program_root,
        raw_store_root=raw_root,
    )

    assert tuple(bundle.pbp_rows_by_game) == X13_GAME_IDS
    assert sum(len(rows) for rows in bundle.pbp_rows_by_game.values()) == 3_699
    assert bundle.sample_audit["touchdowns"] == 124
    assert bundle.sample_audit["made_field_goals"] == 98
    assert bundle.sample_audit["interceptions"] == 38
    assert bundle.sample_audit["lost_fumbles"] == 28
    assert bundle.sample_audit["turnover_on_downs"] == 25
    assert bundle.sample_audit["reviews"] == 31
    assert bundle.sample_audit["reversed_reviews"] == 22
    assert bundle.sample_audit["safeties"] == 4
    assert bundle.sample_audit["return_touchdowns"] == 9
    assert bundle.sample_audit["overtime_game_count"] == 4
    assert bundle.sample_audit["tie_game_count"] == 1
    assert sum(len(rows) for rows in bundle.participation_rows_by_game.values()) == 3_425
    assert bundle.full_11v11_participation_rows == 3_413
