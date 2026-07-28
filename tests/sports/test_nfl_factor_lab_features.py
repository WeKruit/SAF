from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

import prediction_market.sports.nfl_factor_lab_features as feature_module
from prediction_market.sports.nfl_factor_lab_features import (
    MARKET_FACTOR_COLUMNS_V1,
    REGISTERED_FACTOR_COLUMNS_V1,
    SportsFeatureBuildError,
    _sequence_context,
    build_sports_feature_view,
)
from prediction_market.sports.nfl_factor_lab_cleaning import (
    build_sports_row_disposition_ledger,
)
from prediction_market.sports.nfl_factor_lab_types import PlayerRoleV1


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _pbp(
    game_id: str,
    play_id: int,
    *,
    time_of_day: str | None,
    posteam: str = "AAA",
    defteam: str = "BBB",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": game_id,
        "play_id": play_id,
        "order_sequence": play_id,
        "season": int(game_id[:4]),
        "season_type": "REG",
        "week": 1,
        "game_date": None if time_of_day is None else time_of_day[:10],
        "start_time": None,
        "time_of_day": time_of_day,
        "home_team": "BBB",
        "away_team": "AAA",
        "posteam": posteam,
        "defteam": defteam,
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
        "desc": "pass complete",
        "play_type": "pass",
        "pass_attempt": 1,
        "rush_attempt": 0,
        "qb_scramble": 0,
        "sack": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "yards_gained": 12,
        "first_down": 1,
        "success": 1,
        "touchdown": 0,
        "pass_touchdown": 0,
        "rush_touchdown": 0,
        "return_touchdown": 0,
        "field_goal_attempt": 0,
        "field_goal_result": None,
        "extra_point_attempt": 0,
        "extra_point_result": None,
        "two_point_attempt": 0,
        "two_point_conv_result": None,
        "defensive_two_point_attempt": 0,
        "defensive_two_point_conv": 0,
        "interception": 0,
        "fumble": 0,
        "fumble_lost": 0,
        "fumble_recovery_1_team": None,
        "fourth_down_converted": 0,
        "fourth_down_failed": 0,
        "punt_attempt": 0,
        "punt_blocked": 0,
        "punt_in_endzone": 0,
        "punt_out_of_bounds": 0,
        "punt_downed": 0,
        "punt_inside_twenty": 0,
        "punt_fair_catch": 0,
        "kickoff_attempt": 0,
        "kickoff_inside_twenty": 0,
        "kickoff_in_endzone": 0,
        "kickoff_out_of_bounds": 0,
        "kickoff_downed": 0,
        "kickoff_fair_catch": 0,
        "own_kickoff_recovery": 0,
        "own_kickoff_recovery_td": 0,
        "touchback": 0,
        "return_team": None,
        "return_yards": None,
        "fumble_recovery_1_yards": None,
        "safety": 0,
        "penalty": 0,
        "penalty_team": None,
        "penalty_yards": None,
        "penalty_type": None,
        "timeout": 0,
        "timeout_team": None,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "replay_or_challenge": 0,
        "replay_or_challenge_result": None,
        "play_deleted": 0,
        "shotgun": 1,
        "no_huddle": 0,
        "passer_player_id": "QB1",
        "passer_player_name": "Quarter Back",
        "rusher_player_id": None,
        "rusher_player_name": None,
        "receiver_player_id": "WR1",
        "receiver_player_name": "Wide Receiver",
        "kicker_player_id": None,
        "kicker_player_name": None,
        "punter_player_id": None,
        "punter_player_name": None,
        "punt_returner_player_id": None,
        "punt_returner_player_name": None,
        "interception_player_id": None,
        "interception_player_name": None,
        "fumbled_1_player_id": None,
        "fumbled_1_player_name": None,
        "fumble_recovery_1_player_id": None,
        "fumble_recovery_1_player_name": None,
        "solo_tackle_1_player_id": None,
        "solo_tackle_1_player_name": None,
        "wp": 0.45,
        "vegas_wp": 0.47,
        "wpa": 0.02,
        "vegas_home_wpa": -0.018,
        "epa": 0.8,
        "qb_epa": 0.8,
        "xpass": 0.7,
        "pass_oe": 0.3,
        "cp": 0.6,
        "cpoe": 0.4,
        "yac_epa": 0.2,
        "xyac_epa": 0.1,
    }
    row.update(overrides)
    return row


def _episode(game_id: str, play_id: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_id": f"episode-{play_id}",
        "game_id": game_id,
        "play_ids": [str(play_id)],
        "episode_type": "pass",
        "nullified": False,
        "audit_only": False,
    }
    row.update(overrides)
    return row


def _metric(row: object, metric_id: str) -> object:
    return next(
        value
        for value in row.rolling_features
        if value.metric_id == metric_id
    )


def test_build_feature_view_recovers_taxonomy_orientation_and_model_fields() -> None:
    target = _pbp(
        "2025_01_AAA_BBB",
        10,
        time_of_day="2025-09-10T17:10:00Z",
        touchdown=1,
        pass_touchdown=1,
        vegas_home_wpa=-0.018,
    )
    historical = [
        _pbp(
            "2024_01_AAA_BBB",
            1,
            time_of_day="2024-09-01T18:00:00Z",
            epa=1.0,
        ),
        _pbp(
            "2024_01_AAA_BBB",
            2,
            time_of_day="2024-09-01T18:01:00Z",
            epa=-0.5,
            success=0,
        ),
        # This game finishes after the target kickoff and must not leak.
        _pbp(
            "2025_02_AAA_BBB",
            1,
            time_of_day="2025-09-20T18:00:00Z",
            epa=100.0,
            interception=1,
        ),
    ]

    view = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode(target["game_id"], 10)],
        historical_pbp_rows=historical,
        kickoff_times_utc={"2025_01_AAA_BBB": "2025-09-10T17:00:00Z"},
        source_hashes={
            "nflverse_2022_2025": _hash("a"),
            "x13_episodes": _hash("b"),
        },
        low_support_observations=8,
    )

    assert len(view.rows) == 1
    row = view.rows[0]
    assert row.episode_id == "episode-10"
    assert row.focal_team == "AAA"
    assert row.realized_delta_wp_focal == pytest.approx(0.018)
    assert {"pass", "touchdown", "pass_touchdown"} <= set(row.event_flags)
    assert row.yac_surprise == pytest.approx(0.1)
    assert {role.role for role in row.player_roles} >= {"passer", "receiver"}

    passing = _metric(row, "team_pass_epa_per_play")
    assert passing.season_to_date_value is None
    assert passing.season_to_date_observations == 0
    assert passing.trailing_8_value == pytest.approx(0.25)
    assert passing.trailing_8_observations == 2
    assert passing.low_support is True
    assert row.to_dict()["rolling_pass_epa_diff"] is None
    assert all(
        metric.metric_id != "turnover_over_expected"
        for metric in row.rolling_features
    )
    assert set(REGISTERED_FACTOR_COLUMNS_V1) <= set(view.to_records()[0])


def test_sequence_rules_match_registered_sports_meaning() -> None:
    game_id = "2025_01_AAA_BBB"
    rows = [
        _pbp(
            game_id,
            10,
            time_of_day="2025-09-10T17:10:00Z",
            interception=1,
        ),
        _pbp(
            game_id,
            11,
            time_of_day="2025-09-10T17:11:00Z",
        ),
        _pbp(
            game_id,
            12,
            time_of_day="2025-09-10T17:12:00Z",
            fumble_lost=1,
        ),
        _pbp(
            game_id,
            13,
            time_of_day="2025-09-10T17:13:00Z",
            fixed_drive=2,
            yardline_100=15,
        ),
        _pbp(
            game_id,
            14,
            time_of_day="2025-09-10T17:14:00Z",
            fixed_drive=2,
            field_goal_attempt=1,
            field_goal_result="made",
        ),
        _pbp(
            game_id,
            15,
            time_of_day="2025-09-10T17:15:00Z",
            punt_attempt=1,
            punt_blocked=1,
            return_yards=5,
        ),
        _pbp(
            game_id,
            16,
            time_of_day="2025-09-10T17:16:00Z",
        ),
        _pbp(
            game_id,
            17,
            time_of_day="2025-09-10T17:17:00Z",
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
        ),
    ]
    episodes = {
        (game_id, str(row["play_id"])): {
            **_episode(game_id, int(row["play_id"])),
            "episode_id": (
                "episode-review"
                if int(row["play_id"]) in {16, 17}
                else f"episode-{int(row['play_id'])}"
            ),
        }
        for row in rows
    }
    sequence = _sequence_context(rows, episodes)

    assert sequence[(game_id, "12")][
        "sequence_back_to_back_turnovers"
    ]
    assert not sequence[(game_id, "14")][
        "sequence_red_zone_empty_drive"
    ]
    assert sequence[(game_id, "15")]["sequence_blocked_kick_return"]
    assert sequence[(game_id, "16")]["sequence_review_reversal"]


def test_native_special_teams_admin_fumble_and_roster_fields_are_propagated() -> None:
    game_id = "2025_01_AAA_BBB"
    target = _pbp(
        game_id,
        10,
        time_of_day="2025-09-10T17:10:00Z",
        play_type="punt",
        pass_attempt=0,
        punt_attempt=1,
        punt_blocked=1,
        punt_inside_twenty=1,
        punt_out_of_bounds=1,
        kickoff_attempt=1,
        kickoff_inside_twenty=1,
        kickoff_out_of_bounds=1,
        defensive_two_point_attempt=1,
        defensive_two_point_conv=1,
        penalty=1,
        penalty_team="BBB",
        penalty_type="Defensive Holding",
        penalty_yards=5,
        timeout=1,
        timeout_team="AAA",
        return_team="AAA",
        return_yards=12,
        fumble_recovery_1_team="AAA",
        fumble_recovery_1_yards=8,
        kicker_player_id="K1",
        kicker_player_name="Kicker One",
    )
    historical = _pbp(
        "2024_01_AAA_BBB",
        1,
        time_of_day="2024-09-01T18:00:00Z",
    )
    canonical = {
        "game_id": game_id,
        "play_id": "10",
        "penalty_status": "accepted",
        "between_play_roster_change": {
            "offense_entering_names": ["New Receiver"],
            "offense_leaving_names": ["Old Receiver"],
            "defense_entering_names": ["New Corner"],
            "defense_leaving_names": ["Old Corner"],
            "semantics": (
                "set difference between consecutive observed participation rows; "
                "timing is bounded between plays"
            ),
        },
    }

    view = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode(game_id, 10)],
        historical_pbp_rows=[historical],
        kickoff_times_utc={game_id: "2025-09-10T17:00:00Z"},
        source_hashes={"pbp": _hash("a")},
        canonical_event_rows=[canonical],
    )
    row = view.rows[0]

    assert row.defensive_two_point_attempt is True
    assert row.defensive_two_point_conversion is True
    assert row.kicker_player_id == "K1"
    assert row.kicker_player_name == "Kicker One"
    assert row.penalty_team == "BBB"
    assert row.penalty_type == "Defensive Holding"
    assert row.penalty_yards == 5
    assert row.penalty_status == "accepted"
    assert row.timeout_team == "AAA"
    assert row.punt_blocked is True
    assert row.punt_inside_twenty is True
    assert row.punt_out_of_bounds is True
    assert row.kickoff_inside_twenty is True
    assert row.kickoff_out_of_bounds is True
    assert row.return_team == "AAA"
    assert row.return_yards == 12
    assert row.fumble_recovery_team == "AAA"
    assert row.fumble_recovery_yards == 8
    # A supplied precomputed delta is not trusted without an adjacent,
    # same-team/same-unit live participation row.
    assert row.between_play_roster_change is None
    assert row.to_dict()["participation_transition"] is None


def test_negative_source_timeout_sentinel_is_preserved_as_missing() -> None:
    target = _pbp(
        "2025_20_AAA_BBB",
        10,
        time_of_day="2026-01-17T23:00:00Z",
        away_timeouts_remaining=-1,
    )
    historical = _pbp(
        "2024_01_AAA_BBB",
        1,
        time_of_day="2024-09-01T18:00:00Z",
    )
    view = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode("2025_20_AAA_BBB", 10)],
        historical_pbp_rows=[historical],
        kickoff_times_utc={
            "2025_20_AAA_BBB": "2026-01-17T22:59:00Z"
        },
        source_hashes={"pbp": _hash("a")},
    )

    assert view.rows[0].away_timeouts_remaining is None


def test_pre_event_score_does_not_use_post_play_total_score() -> None:
    target = _pbp(
        "2025_01_AAA_BBB",
        10,
        time_of_day="2025-09-10T17:10:00Z",
        posteam_score=3,
        defteam_score=0,
        total_home_score=9,
        total_away_score=3,
        touchdown=1,
    )
    historical = _pbp(
        "2024_01_AAA_BBB",
        1,
        time_of_day="2024-09-01T18:00:00Z",
    )
    view = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode("2025_01_AAA_BBB", 10)],
        historical_pbp_rows=[historical],
        kickoff_times_utc={
            "2025_01_AAA_BBB": "2025-09-10T17:00:00Z"
        },
        source_hashes={"pbp": _hash("a")},
    )

    assert view.rows[0].home_score == 0
    assert view.rows[0].away_score == 3
    assert view.rows[0].score_margin_focal == 3


def test_timeout_is_preserved_as_admin_without_entering_play_factor_view() -> None:
    target = _pbp(
        "2025_01_AAA_BBB",
        10,
        time_of_day="2025-09-10T17:10:00Z",
        posteam=None,
        defteam=None,
        posteam_score=None,
        defteam_score=None,
        total_home_score=10,
        total_away_score=17,
        play_type="no_play",
        timeout=1,
        timeout_team="AAA",
        pass_attempt=0,
        touchdown=1,
    )
    episode = _episode(
        "2025_01_AAA_BBB",
        10,
        episode_type="administrative",
        nullified=True,
        pre_state={
            "home_team": "BBB",
            "away_team": "AAA",
            "home_score": 10,
            "away_score": 17,
            "possession_team": None,
            "offense_team": None,
        },
    )
    historical = _pbp(
        "2024_01_AAA_BBB",
        1,
        time_of_day="2024-09-01T18:00:00Z",
    )

    ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=[target],
        finalized_episode_rows=[episode],
        source_hashes={"pbp": _hash("a")},
    )
    record = ledger.to_records()[0]
    assert record["row_type"] == "no_play"
    assert record["live_play_outcome_eligible"] is False
    assert record["administrative_event_eligible"] is True
    assert record["factor_eligible"] is False
    assert {"administrative", "episode_nullified", "no_play"} <= set(
        record["exclusion_reasons"]
    )
    with pytest.raises(
        SportsFeatureBuildError,
        match="no focal-team feature rows",
    ):
        build_sports_feature_view(
            selected_pbp_rows=[target],
            finalized_episode_rows=[episode],
            historical_pbp_rows=[historical],
            kickoff_times_utc={
                "2025_01_AAA_BBB": "2025-09-10T17:00:00Z"
            },
            source_hashes={"pbp": _hash("a")},
            row_disposition_ledger=ledger,
        )


def test_feature_contract_covers_every_registered_predicate_field() -> None:
    registry_path = (
        Path(__file__).resolve().parents[2]
        / "registries"
        / "factors"
        / "nfl_factor_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text("utf-8"))
    fields = {
        clause["field"]
        for definition in registry["factor_definitions"]
        for clause in definition["predicate"]
    }

    assert fields <= (
        set(REGISTERED_FACTOR_COLUMNS_V1)
        | set(MARKET_FACTOR_COLUMNS_V1)
    )


def test_build_is_input_order_invariant_and_preserves_missing_model_output() -> None:
    target_a = _pbp(
        "2025_01_AAA_BBB",
        10,
        time_of_day="2025-09-10T17:10:00Z",
        pass_oe=None,
    )
    target_b = _pbp(
        "2025_01_AAA_BBB",
        11,
        time_of_day="2025-09-10T17:11:00Z",
        play_type="run",
        pass_attempt=0,
        rush_attempt=1,
        passer_player_id=None,
        passer_player_name=None,
        receiver_player_id=None,
        receiver_player_name=None,
        rusher_player_id="RB1",
        rusher_player_name="Running Back",
        pass_oe=None,
    )
    kwargs = {
        "finalized_episode_rows": [
            _episode(target_a["game_id"], 10),
            _episode(target_b["game_id"], 11, episode_type="run"),
        ],
        "historical_pbp_rows": [
            _pbp(
                "2024_01_AAA_BBB",
                1,
                time_of_day="2024-09-01T18:00:00Z",
            )
        ],
        "kickoff_times_utc": {
            "2025_01_AAA_BBB": "2025-09-10T17:00:00Z"
        },
        "source_hashes": {
            "nflverse_2022_2025": _hash("a"),
            "x13_episodes": _hash("b"),
        },
    }

    first = build_sports_feature_view(
        selected_pbp_rows=[target_b, target_a],
        **deepcopy(kwargs),
    )
    second = build_sports_feature_view(
        selected_pbp_rows=[target_a, target_b],
        **deepcopy(kwargs),
    )

    assert first.rows_sha256 == second.rows_sha256
    assert [row.play_id for row in first.rows] == ["10", "11"]
    assert first.rows[0].pass_oe is None


def test_build_fails_closed_on_missing_episode_or_kickoff() -> None:
    target = _pbp(
        "2025_01_AAA_BBB",
        10,
        time_of_day="2025-09-10T17:10:00Z",
    )
    common = {
        "selected_pbp_rows": [target],
        "historical_pbp_rows": [],
        "source_hashes": {
            "nflverse_2022_2025": _hash("a"),
            "x13_episodes": _hash("b"),
        },
    }

    with pytest.raises(SportsFeatureBuildError, match="episode"):
        build_sports_feature_view(
            finalized_episode_rows=[],
            kickoff_times_utc={
                "2025_01_AAA_BBB": "2025-09-10T17:00:00Z"
            },
            **common,
        )
    with pytest.raises(SportsFeatureBuildError, match="kickoff"):
        build_sports_feature_view(
            finalized_episode_rows=[_episode(target["game_id"], 10)],
            kickoff_times_utc={},
            **common,
        )


def test_participation_ids_are_restored_as_aligned_player_id_name_pairs() -> None:
    game_id = "2025_01_AAA_BBB"
    target = _pbp(
        game_id,
        10,
        time_of_day="2025-09-10T17:10:00Z",
        participation={
            "offense_players": ["GSIS-Z", "GSIS-A"],
            "offense_names": ["Zed Receiver", "Alice Quarterback"],
            "defense_players": ["GSIS-D2", "GSIS-D1"],
            "defense_names": ["Zed Corner", "Alice Safety"],
        },
    )
    canonical = {
        "game_id": game_id,
        "play_id": "10",
        "offense_names": ["Zed Receiver", "Alice Quarterback"],
        "defense_names": ["Zed Corner", "Alice Safety"],
    }

    row = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode(game_id, 10)],
        historical_pbp_rows=[],
        kickoff_times_utc={game_id: "2025-09-10T17:00:00Z"},
        source_hashes={"pbp": _hash("a")},
        canonical_event_rows=[canonical],
    ).rows[0]

    assert row.offense_player_ids == ("GSIS-A", "GSIS-Z")
    assert row.defense_player_ids == ("GSIS-D1", "GSIS-D2")
    assert {
        (role.role, role.player_id, role.player_name)
        for role in row.player_roles
        if role.role in {"offense_participant", "defense_participant"}
    } == {
        ("offense_participant", "GSIS-A", "Alice Quarterback"),
        ("offense_participant", "GSIS-Z", "Zed Receiver"),
        ("defense_participant", "GSIS-D1", "Alice Safety"),
        ("defense_participant", "GSIS-D2", "Zed Corner"),
    }


def test_roster_delta_uses_adjacent_same_team_same_unit_live_snaps() -> None:
    game_id = "2025_01_AAA_BBB"
    rows = [
        _pbp(game_id, 10, time_of_day="2025-09-10T17:10:00Z"),
        _pbp(
            game_id,
            11,
            time_of_day="2025-09-10T17:10:30Z",
            play_type="no_play",
            pass_attempt=0,
            timeout=1,
        ),
        _pbp(game_id, 12, time_of_day="2025-09-10T17:11:00Z"),
        _pbp(
            game_id,
            13,
            time_of_day="2025-09-10T17:12:00Z",
            posteam="BBB",
            defteam="AAA",
            play_type="kickoff",
            pass_attempt=0,
            kickoff_attempt=1,
        ),
        _pbp(game_id, 14, time_of_day="2025-09-10T17:13:00Z"),
    ]
    canonical = [
        {
            "game_id": game_id,
            "play_id": "10",
            "offense_players": ["QB1", "WR1"],
            "offense_names": ["Quarter Back", "Old Receiver"],
            "defense_players": ["CB1"],
            "defense_names": ["Corner One"],
        },
        {
            "game_id": game_id,
            "play_id": "11",
            "offense_players": ["CONTAMINATED"],
            "offense_names": ["Contaminated Admin"],
            "defense_players": ["CONTAMINATED-D"],
            "defense_names": ["Contaminated Defense"],
            "between_play_roster_change": {
                "offense_entering_names": ["Contaminated Admin"],
                "offense_leaving_names": ["Old Receiver"],
                "defense_entering_names": ["Contaminated Defense"],
                "defense_leaving_names": ["Corner One"],
                "semantics": (
                    "set difference between consecutive observed participation rows; "
                    "timing is bounded between plays"
                ),
            },
        },
        {
            "game_id": game_id,
            "play_id": "12",
            "offense_players": ["QB1", "WR2"],
            "offense_names": ["Quarter Back", "New Receiver"],
            "defense_players": ["CB1"],
            "defense_names": ["Corner One"],
        },
        {
            "game_id": game_id,
            "play_id": "13",
            "offense_players": ["K1"],
            "offense_names": ["Kicker One"],
            "defense_players": ["RET1"],
            "defense_names": ["Returner One"],
            "between_play_roster_change": {
                "offense_entering_names": ["Kicker One"],
                "offense_leaving_names": ["Quarter Back", "New Receiver"],
                "defense_entering_names": ["Returner One"],
                "defense_leaving_names": ["Corner One"],
                "semantics": (
                    "set difference between consecutive observed participation rows; "
                    "timing is bounded between plays"
                ),
            },
        },
        {
            "game_id": game_id,
            "play_id": "14",
            "offense_players": ["QB1", "WR3"],
            "offense_names": ["Quarter Back", "Third Receiver"],
            "defense_players": ["CB1"],
            "defense_names": ["Corner One"],
        },
    ]
    episodes = [
        _episode(game_id, 10),
        _episode(game_id, 11, episode_type="administrative", nullified=True),
        _episode(game_id, 12),
        _episode(game_id, 13, episode_type="kickoff"),
        _episode(game_id, 14),
    ]

    view = build_sports_feature_view(
        selected_pbp_rows=rows,
        finalized_episode_rows=episodes,
        historical_pbp_rows=[],
        kickoff_times_utc={game_id: "2025-09-10T17:00:00Z"},
        source_hashes={"pbp": _hash("a")},
        canonical_event_rows=canonical,
    )
    by_play = {row.play_id: row for row in view.rows}

    assert "11" not in by_play
    change = by_play["12"].between_play_roster_change
    assert change is not None
    assert change.offense_entering_names == ("New Receiver",)
    assert change.offense_leaving_names == ("Old Receiver",)
    assert by_play["12"].to_dict()["participation_transition"] == "roster_change"
    for play_id in ("13", "14"):
        assert by_play[play_id].between_play_roster_change is None
        assert by_play[play_id].to_dict()["participation_transition"] == (
            "phase_transition"
        )


def test_history_is_adjudicated_before_rolling_metrics_and_exposes_pit_audit() -> None:
    game_id = "2025_01_AAA_BBB"
    historical_game_id = "2024_01_AAA_BBB"
    target = _pbp(
        game_id,
        10,
        time_of_day="2025-09-10T17:10:00Z",
    )
    historical = [
        _pbp(
            historical_game_id,
            1,
            time_of_day="2024-09-01T18:00:00Z",
            epa=1.0,
        ),
        _pbp(
            historical_game_id,
            2,
            time_of_day="2024-09-01T18:01:00Z",
            play_type="no_play",
            epa=100.0,
        ),
        _pbp(
            historical_game_id,
            3,
            time_of_day="2024-09-01T18:02:00Z",
            play_deleted=1,
            epa=100.0,
        ),
        _pbp(
            historical_game_id,
            4,
            time_of_day="2024-09-01T18:03:00Z",
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
            epa=100.0,
        ),
        _pbp(
            historical_game_id,
            5,
            time_of_day=None,
            epa=100.0,
        ),
    ]

    row = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode(game_id, 10)],
        historical_pbp_rows=historical,
        kickoff_times_utc={game_id: "2025-09-10T17:00:00Z"},
        source_hashes={"pbp": _hash("a")},
        low_support_observations=1,
    ).rows[0]

    passing = _metric(row, "team_pass_epa_per_play")
    assert passing.trailing_8_value == pytest.approx(1.0)
    assert passing.trailing_8_observations == 1
    record = row.to_dict()
    assert record["history_cutoff_utc"] == "2025-09-10T17:00:00Z"
    assert record["history_completed_game_count"] == 1
    assert record["history_eligible_row_count"] == 1
    assert record["history_excluded_row_count"] == 4
    assert str(record["history_support_sha256"]).startswith("sha256:")


def test_frozen_binding_kickoff_precedes_explicit_row_time_and_hashes_ledger_rows() -> (
    None
):
    game_id = "2025_01_AAA_BBB"
    target = _pbp(
        game_id,
        10,
        time_of_day="2025-09-10T17:10:00Z",
    )
    admin = _pbp(
        game_id,
        11,
        time_of_day=None,
        play_type=None,
        pass_attempt=0,
        posteam=None,
        defteam=None,
        desc="END Q2",
    )
    episodes = [
        _episode(game_id, 10),
        _episode(game_id, 11, episode_type="administrative"),
    ]
    kwargs = {
        "selected_pbp_rows": [target, admin],
        "finalized_episode_rows": episodes,
        "historical_pbp_rows": [],
        "kickoff_times_utc": {game_id: "2025-09-10T17:09:00Z"},
        "frozen_game_bindings": [
            {
                "native_game_id": game_id,
                "scheduled_at_utc": "2025-09-10T17:00:00Z",
            }
        ],
        "source_hashes": {"pbp": _hash("a")},
    }
    first_ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=[target, admin],
        finalized_episode_rows=episodes,
        source_hashes={"pbp": _hash("a")},
    )
    first = build_sports_feature_view(
        row_disposition_ledger=first_ledger,
        **deepcopy(kwargs),
    )
    changed_admin = {**admin, "desc": "OT END Q4"}
    changed_ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=[target, changed_admin],
        finalized_episode_rows=episodes,
        source_hashes={"pbp": _hash("a")},
    )
    second = build_sports_feature_view(
        selected_pbp_rows=[target, changed_admin],
        finalized_episode_rows=deepcopy(episodes),
        historical_pbp_rows=[],
        kickoff_times_utc={game_id: "2025-09-10T17:09:00Z"},
        frozen_game_bindings=[
            {
                "native_game_id": game_id,
                "scheduled_at_utc": "2025-09-10T17:00:00Z",
            }
        ],
        source_hashes={"pbp": _hash("a")},
        row_disposition_ledger=changed_ledger,
    )

    assert first.rows[0].kickoff_at_utc == "2025-09-10T17:00:00Z"
    assert len(first.rows) == len(second.rows) == 1
    assert first_ledger.rows_sha256 != changed_ledger.rows_sha256
    assert first.rows_sha256 != second.rows_sha256
    assert dict(first.source_hashes)["sports_row_disposition_ledger_v2"] == (
        first_ledger.rows_sha256
    )


def test_same_game_rows_are_never_admitted_to_pit_history_support() -> None:
    game_id = "2025_01_AAA_BBB"
    target = _pbp(
        game_id,
        10,
        time_of_day="2025-09-10T17:10:00Z",
    )
    history = [
        _pbp(
            "2024_01_AAA_BBB",
            1,
            time_of_day="2024-09-01T18:00:00Z",
            epa=1.0,
        ),
        _pbp(
            game_id,
            1,
            time_of_day="2025-09-10T16:59:00Z",
            epa=100.0,
        ),
    ]

    row = build_sports_feature_view(
        selected_pbp_rows=[target],
        finalized_episode_rows=[_episode(game_id, 10)],
        historical_pbp_rows=history,
        kickoff_times_utc={game_id: "2025-09-10T17:00:00Z"},
        source_hashes={"pbp": _hash("a")},
        low_support_observations=1,
    ).rows[0]

    passing = _metric(row, "team_pass_epa_per_play")
    assert passing.trailing_8_value == pytest.approx(1.0)
    assert passing.trailing_8_observations == 1
    assert row.to_dict()["history_same_game_excluded_row_count"] == 1


def test_each_target_excludes_only_itself_from_pit_history() -> None:
    early_game_id = "2025_01_AAA_BBB"
    later_game_id = "2025_02_AAA_BBB"
    future_game_id = "2025_03_AAA_BBB"
    selected = [
        _pbp(
            early_game_id,
            10,
            time_of_day="2025-09-01T18:10:00Z",
        ),
        _pbp(
            later_game_id,
            10,
            time_of_day="2025-09-08T18:10:00Z",
        ),
    ]
    history = [
        _pbp(
            early_game_id,
            1,
            time_of_day="2025-09-01T18:10:00Z",
            epa=2.0,
        ),
        # This anomalous pre-kickoff timestamp proves that the current target
        # game is excluded explicitly rather than merely by the time cutoff.
        _pbp(
            later_game_id,
            1,
            time_of_day="2025-09-08T17:59:00Z",
            epa=100.0,
        ),
        _pbp(
            future_game_id,
            1,
            time_of_day="2025-09-20T18:10:00Z",
            epa=1_000.0,
        ),
    ]

    view = build_sports_feature_view(
        selected_pbp_rows=selected,
        finalized_episode_rows=[
            _episode(early_game_id, 10),
            _episode(later_game_id, 10),
        ],
        historical_pbp_rows=history,
        kickoff_times_utc={
            early_game_id: "2025-09-01T18:00:00Z",
            later_game_id: "2025-09-08T18:00:00Z",
        },
        source_hashes={"pbp": _hash("a")},
        low_support_observations=1,
    )

    by_game = {row.game_id: row for row in view.rows}
    early_metric = _metric(by_game[early_game_id], "team_pass_epa_per_play")
    later_metric = _metric(by_game[later_game_id], "team_pass_epa_per_play")
    assert early_metric.trailing_8_value is None
    assert early_metric.trailing_8_observations == 0
    assert later_metric.trailing_8_value == pytest.approx(2.0)
    assert later_metric.trailing_8_observations == 1
    assert by_game[early_game_id].to_dict()["history_completed_game_count"] == 0
    assert by_game[later_game_id].to_dict()["history_completed_game_count"] == 1
    assert (
        by_game[later_game_id]
        .to_dict()["history_same_game_excluded_row_count"]
        == 1
    )
    assert (
        by_game[early_game_id].to_dict()["history_support_sha256"]
        != by_game[later_game_id].to_dict()["history_support_sha256"]
    )


def test_missing_frozen_schedule_is_labeled_first_live_source_time_data_gap() -> None:
    game_id = "2025_01_AAA_BBB"
    row = build_sports_feature_view(
        selected_pbp_rows=[
            _pbp(
                game_id,
                10,
                time_of_day="2025-09-10T17:10:00Z",
            )
        ],
        finalized_episode_rows=[_episode(game_id, 10)],
        historical_pbp_rows=[],
        kickoff_times_utc={game_id: "2025-09-10T17:05:00Z"},
        frozen_game_bindings=[
            {
                "native_game_id": game_id,
                "scheduled_at_utc": None,
            }
        ],
        source_hashes={"pbp": _hash("a")},
    ).rows[0]

    record = row.to_dict()
    assert row.kickoff_at_utc == "2025-09-10T17:05:00Z"
    assert record["history_cutoff_semantics"] == "first_live_source_time"
    assert record["scheduled_kickoff_status"] == (
        "scheduled_kickoff_unavailable"
    )
    assert record["sports_data_status"] == "DATA_GAP"


def test_pit_support_hash_is_computed_once_per_game_team_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_id = "2025_01_AAA_BBB"
    calls = 0
    original = feature_module.semantic_rows_sha256

    def counted(value: object) -> str:
        nonlocal calls
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "PITHistorySupportV2"
        ):
            calls += 1
        return original(value)

    monkeypatch.setattr(feature_module, "semantic_rows_sha256", counted)
    rows = [
        _pbp(game_id, 10, time_of_day="2025-09-10T17:10:00Z"),
        _pbp(game_id, 11, time_of_day="2025-09-10T17:11:00Z"),
    ]

    view = build_sports_feature_view(
        selected_pbp_rows=rows,
        finalized_episode_rows=[_episode(game_id, 10), _episode(game_id, 11)],
        historical_pbp_rows=[
            _pbp(
                "2024_01_AAA_BBB",
                1,
                time_of_day="2024-09-01T18:00:00Z",
            )
        ],
        kickoff_times_utc={game_id: "2025-09-10T17:00:00Z"},
        source_hashes={"pbp": _hash("a")},
    )

    assert len(view.rows) == 2
    assert calls == 1
    assert (
        view.rows[0].to_dict()["history_support_sha256"]
        == view.rows[1].to_dict()["history_support_sha256"]
    )


def test_player_role_without_registered_metrics_skips_history_scan() -> None:
    class HistoryScanMustNotRun:
        def completed_game_ids(self, **_: object) -> tuple[str, ...]:
            raise AssertionError("history scan must not run for an empty selector set")

        def rows_for(self, _: object) -> list[dict[str, object]]:
            raise AssertionError("history rows must not load for an empty selector set")

    result = feature_module._player_rolling_features(
        history=HistoryScanMustNotRun(),  # type: ignore[arg-type]
        before=datetime(2025, 9, 10, 17, tzinfo=timezone.utc),
        current_game_id="2025_01_AAA_BBB",
        season=2025,
        team="AAA",
        role=PlayerRoleV1(
            role="tackler",
            player_id="DEF1",
            player_name="Defender One",
        ),
        low_support_observations=8,
    )

    assert result == ()


def test_completed_game_query_cache_is_exact_and_scans_once() -> None:
    class CountingGames(dict[str, tuple[dict[str, object], ...]]):
        scans = 0

        def items(self):  # type: ignore[no-untyped-def]
            self.scans += 1
            return super().items()

    history = feature_module._HistoryIndex(
        [
            _pbp(
                "2025_01_AAA_BBB",
                1,
                time_of_day="2025-09-01T18:10:00Z",
            ),
            _pbp(
                "2025_02_AAA_BBB",
                1,
                time_of_day="2025-09-08T18:10:00Z",
            ),
        ]
    )
    counting_games = CountingGames(history.games)
    history.games = counting_games
    common = {
        "entity_id": "AAA",
        "entity_type": "team",
        "season": None,
        "last_games": 8,
    }
    cutoff = datetime(2025, 9, 15, tzinfo=timezone.utc)

    first = history.completed_game_ids(
        before=cutoff,
        current_game_id="2025_99_AAA_BBB",
        **common,
    )
    second = history.completed_game_ids(
        before=cutoff,
        current_game_id="2025_99_AAA_BBB",
        **common,
    )
    assert first == second == ("2025_01_AAA_BBB", "2025_02_AAA_BBB")
    assert isinstance(first, tuple)
    assert counting_games.scans == 1

    different_target = history.completed_game_ids(
        before=cutoff,
        current_game_id="2025_01_AAA_BBB",
        **common,
    )
    assert different_target == ("2025_02_AAA_BBB",)
    assert counting_games.scans == 2

    different_cutoff = history.completed_game_ids(
        before=datetime(2025, 9, 7, tzinfo=timezone.utc),
        current_game_id="2025_99_AAA_BBB",
        **common,
    )
    assert different_cutoff == ("2025_01_AAA_BBB",)
    assert counting_games.scans == 3
