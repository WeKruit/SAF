from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_publication import (
    X15PublicationError,
    build_episode_state_frame,
    normalize_actual_moneyline_trades,
)


def test_build_episode_state_uses_structured_state_and_reference_orientation() -> None:
    factor_rows = pd.DataFrame(
        [
            {
                "game_id": "2025_07_AAA_BBB",
                "episode_id": "ep-1",
                "event_interval_end_utc": "2025-10-19T18:00:00Z",
                "week": 7,
                "play_type": "pass",
                "interception": True,
                "fumble_lost": False,
                "fourth_down_failed": False,
                "pass_touchdown": False,
                "rush_touchdown": False,
                "return_touchdown": False,
                "field_goal_result": None,
                "safety": False,
                "punt": False,
                "home_team": "AAA",
                "away_team": "BBB",
                "home_score": 10,
                "away_score": 7,
                "game_seconds_remaining": 901,
                "rolling_offense_success_diff": 0.08,
                "factor_id": "NFL.TURNOVER.INTERCEPTION",
                "horizon_seconds": 5,
            }
        ]
    )
    references = pd.DataFrame(
        [
            {
                "game_id": "2025_07_AAA_BBB",
                "episode_id": "ep-1",
                "focal_team": "BBB",
                "p_after_ref": 0.31,
                "reference_delta": -0.12,
                "support_status": "SUPPORTED_RETROSPECTIVE",
            }
        ]
    )

    result = build_episode_state_frame(
        factor_rows,
        references,
        matched_control_episode_ids={("2025_07_AAA_BBB", "ep-1")},
    )

    row = result.iloc[0]
    assert row["beneficiary_team"] == "BBB"
    assert row["event_type"] == "interception"
    assert row["factor_id"] == "NFL.X15.PRIMARY.INTERCEPTION"
    assert row["score_margin"] == -3
    assert row["nfl_week"] == 7
    assert bool(row["is_matched_control"]) is True
    assert row["p_after_ref"] == pytest.approx(0.31)
    assert row["reference_delta"] == pytest.approx(-0.12)


def test_episode_state_rejects_unsupported_or_duplicate_reference() -> None:
    factor_rows = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "episode_id": "ep-1",
                "event_interval_end_utc": "2025-09-01T00:00:00Z",
                "week": 1,
                "play_type": "run",
                "interception": False,
                "fumble_lost": False,
                "fourth_down_failed": False,
                "pass_touchdown": False,
                "rush_touchdown": False,
                "return_touchdown": False,
                "field_goal_result": None,
                "safety": False,
                "punt": False,
                "home_team": "AAA",
                "away_team": "BBB",
                "home_score": 0,
                "away_score": 0,
                "game_seconds_remaining": 3500,
                "rolling_offense_success_diff": None,
                "factor_id": "NFL.PLAY.RUN",
                "horizon_seconds": 5,
            }
        ]
    )
    bad_reference = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "episode_id": "ep-1",
                "focal_team": "AAA",
                "p_after_ref": 0.5,
                "reference_delta": 0.0,
                "support_status": "MODEL_SUPPORT_UNPROVEN",
            }
        ]
    )
    with pytest.raises(X15PublicationError, match="supported reference"):
        build_episode_state_frame(
            factor_rows,
            bad_reference,
            matched_control_episode_ids=set(),
        )


def test_trade_normalization_keeps_native_time_and_actual_moneyline_only() -> None:
    observations = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "observation_id": "o-1",
                "venue": "kalshi",
                "logical_market_id": "kalshi:winner",
                "outcome": "AAA",
                "kind": "trade",
                "source_start_utc": "2025-09-01T00:00:01Z",
                "native_source_time_utc": "2025-09-01T00:00:01.123456Z",
                "price": 0.55,
                "provenance": "observed",
                "primary_path_eligible": True,
            },
            {
                "game_id": "2025_01_AAA_BBB",
                "observation_id": "o-2",
                "venue": "kalshi",
                "logical_market_id": "kalshi:winner",
                "outcome": "AAA",
                "kind": "candle",
                "source_start_utc": "2025-09-01T00:00:02Z",
                "native_source_time_utc": None,
                "price": 0.56,
                "provenance": "observed",
                "primary_path_eligible": True,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "logical_market_id": "kalshi:winner",
                "outcome": "AAA",
                "venue": "kalshi",
                "family": "moneyline",
                "analysis_eligible": True,
            }
        ]
    )

    result = normalize_actual_moneyline_trades(observations, inventory)

    assert result["trade_id"].tolist() == ["o-1"]
    assert result["source_time_utc"].iloc[0] == pd.Timestamp(
        "2025-09-01T00:00:01.123456Z"
    )
    assert result["market_family"].tolist() == ["moneyline"]
    assert result["outcome_team"].tolist() == ["AAA"]
