from __future__ import annotations

import pandas as pd

from prediction_market.reports.nfl_x16_fact_review import render_game_fact_review
from prediction_market.sports.nfl_x16_fact_extraction import GameFactTables


def _tables() -> GameFactTables:
    events = pd.DataFrame(
        [
            {
                "game_id": "2025_14_DAL_DET",
                "event_id": "2025_14_DAL_DET:4839",
                "play_id": "4839",
                "order_sequence": 4839,
                "quarter": 4,
                "game_clock": "01:18",
                "game_seconds_remaining": 78,
                "home_team": "DET",
                "away_team": "DAL",
                "possession_before": "DAL",
                "defense_team": "DET",
                "next_observed_possession": "DET",
                "offense_direction": "RIGHT",
                "pre_home_score": 44,
                "pre_away_score": 30,
                "post_home_score": 44,
                "post_away_score": 30,
                "down": 2,
                "distance": 10,
                "pre_yardline": "DET 24",
                "post_yardline": "DET 14",
                "pre_field_coordinate_0_100": 76,
                "post_field_coordinate_0_100": 86,
                "primary_action": "PASS",
                "outcome_tags": '["INTERCEPTION","POSSESSION_CHANGE","TURNOVER"]',
                "yards_gained": 0,
                "air_yards": None,
                "yards_after_catch": None,
                "return_yards": 0,
                "pass_length": "short",
                "pass_location": "middle",
                "run_location": None,
                "run_gap": None,
                "kick_distance": None,
                "field_goal_result": None,
                "extra_point_result": None,
                "two_point_result": None,
                "penalty_team": None,
                "penalty_type": None,
                "penalty_yards": None,
                "review_result": None,
                "timeout_team": None,
                "home_timeouts_remaining": 1,
                "away_timeouts_remaining": 2,
                "description": "D.Prescott pass INTERCEPTED by D.Reed at DET 14.",
                "participation_status": "PRESENT_COMPLETE_11V11",
                "factor_eligible": True,
                "row_disposition": "LIVE_PLAY",
                "source_time_utc": "2025-12-05T04:10:00Z",
                "source_time_semantics": "NFLVERSE_SOURCE_POINT_NOT_LOCAL_RECEIVE_TIME",
                "home_wp_before_diagnostic": 0.99,
                "home_wp_after_diagnostic": 0.999,
                "home_wp_delta_diagnostic": 0.009,
                "reference_semantics": "NFLVERSE_VEGAS_WP_DIAGNOSTIC_NOT_GROUND_TRUTH",
                "pbp_source_sha256": "sha256:" + "a" * 64,
                "participation_source_sha256": "sha256:" + "b" * 64,
            }
        ]
    )
    return GameFactTables(
        events=events,
        tags=pd.DataFrame(),
        players=pd.DataFrame(
            [
                {
                    "event_id": "2025_14_DAL_DET:4839",
                    "role": "INTERCEPTOR",
                    "player_id": "00-db",
                    "player_name": "D.J. Reed",
                    "team": "DET",
                    "position": "CB",
                    "jersey_number": "4",
                    "evidence_class": "EXPLICIT_PBP_ROLE",
                }
            ]
        ),
        availability=pd.DataFrame(),
        injury_evidence=pd.DataFrame(),
        factor_hits=pd.DataFrame(
            [
                {
                    "event_id": "2025_14_DAL_DET:4839",
                    "factor_id": "NFL.EVENT.INTERCEPTION",
                }
            ]
        ),
        factor_coverage=pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.INTERCEPTION",
                    "status": "ACTIVE",
                    "event_count": 1,
                    "availability_status": "AVAILABLE",
                }
            ]
        ),
        reconciliation=pd.DataFrame(
            [
                {
                    "raw_row_preserved": True,
                    "row_disposition": "LIVE_PLAY",
                }
            ]
        ),
        registry_sha256="sha256:" + "c" * 64,
    )


def test_review_is_offline_event_replay_with_field_players_and_audit() -> None:
    html = render_game_fact_review(_tables())

    assert "<svg" in html
    assert "football-field" in html
    assert "D.Prescott pass INTERCEPTED" in html
    assert "D.J. Reed" in html
    assert "NFL.EVENT.INTERCEPTION" in html
    assert "1 / 1 raw rows preserved" in html
    assert "Market data deliberately excluded" in html
    assert "https://" not in html
    assert "fetch(" not in html
    assert "event-search" in html
    assert "quarter-filter" in html
