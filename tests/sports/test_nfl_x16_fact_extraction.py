from __future__ import annotations

import json

import pandas as pd

from prediction_market.sports.nfl_x16_fact_extraction import (
    _outcome_tags,
    _primary_action,
    _transition_direction_semantics,
    build_game_fact_tables,
)


PBP_SHA = "sha256:" + "a" * 64
PARTICIPATION_SHA = "sha256:" + "b" * 64


def _pbp_rows() -> pd.DataFrame:
    common = {
        "game_id": "2025_14_DAL_DET",
        "home_team": "DET",
        "away_team": "DAL",
        "season": 2025,
        "season_type": "REG",
        "week": 14,
        "qtr": 1,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "play_deleted": 0,
        "goal_to_go": 0,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "play_id": 299,
                "order_sequence": 299,
                "time": "10:16",
                "time_of_day": "2025-12-05T01:20:00.000Z",
                "play_type": "pass",
                "play_type_nfl": "PASS",
                "posteam": "DAL",
                "defteam": "DET",
                "posteam_score": 0,
                "defteam_score": 3,
                "total_home_score": 3,
                "total_away_score": 0,
                "down": 1,
                "ydstogo": 10,
                "yrdln": "DAL 36",
                "yardline_100": 64,
                "end_yard_line": "DAL 47",
                "yards_gained": 11,
                "pass_attempt": 1,
                "complete_pass": 1,
                "first_down": 1,
                "pass_length": "short",
                "pass_location": "right",
                "air_yards": 8,
                "yards_after_catch": 3,
                "passer_player_id": "00-qb",
                "passer_player_name": "D.Prescott",
                "receiver_player_id": "00-wr",
                "receiver_player_name": "G.Pickens",
                "desc": (
                    "(10:16) 4-D.Prescott pass short right to 3-G.Pickens "
                    "to DAL 47 for 11 yards. DET-12-T.Harper was injured "
                    "during the play."
                ),
            },
            {
                **common,
                "play_id": 646,
                "order_sequence": 646,
                "time": "05:48",
                "time_of_day": "2025-12-05T01:30:00.000Z",
                "play_type": "pass",
                "play_type_nfl": "SACK",
                "posteam": "DAL",
                "defteam": "DET",
                "posteam_score": 3,
                "defteam_score": 3,
                "total_home_score": 3,
                "total_away_score": 3,
                "down": 3,
                "ydstogo": 9,
                "yrdln": "DAL 11",
                "yardline_100": 89,
                "end_yard_line": "DAL 1",
                "yards_gained": -10,
                "sack": 1,
                "safety": 0,
                "replay_or_challenge": 1,
                "replay_or_challenge_result": "reversed",
                "sack_player_id": "00-lb",
                "sack_player_name": "J.Campbell",
                "desc": (
                    "(5:48) D.Prescott sacked in End Zone, SAFETY. "
                    "The Replay Official reviewed the safety ruling, and "
                    "the play was REVERSED. D.Prescott sacked at DAL 1."
                ),
            },
            {
                **common,
                "play_id": 778,
                "order_sequence": 778,
                "time": "04:18",
                "time_of_day": "2025-12-05T01:35:00.000Z",
                "play_type": "no_play",
                "play_type_nfl": "PENALTY",
                "posteam": "DET",
                "defteam": "DAL",
                "posteam_score": 3,
                "defteam_score": 3,
                "total_home_score": 3,
                "total_away_score": 3,
                "down": 1,
                "ydstogo": 10,
                "yrdln": "DAL 31",
                "yardline_100": 31,
                "end_yard_line": "DAL 31",
                "penalty": 1,
                "penalty_team": "DET",
                "penalty_type": "Illegal Shift",
                "penalty_yards": 5,
                "desc": "PENALTY on DET, Illegal Shift - No Play.",
            },
            {
                **common,
                "play_id": 4839,
                "order_sequence": 4839,
                "qtr": 4,
                "time": "01:18",
                "game_seconds_remaining": 78,
                "time_of_day": "2025-12-05T04:10:00.000Z",
                "play_type": "pass",
                "play_type_nfl": "INTERCEPTION",
                "posteam": "DAL",
                "defteam": "DET",
                "posteam_score": 30,
                "defteam_score": 44,
                "total_home_score": 44,
                "total_away_score": 30,
                "down": 2,
                "ydstogo": 10,
                "yrdln": "DET 24",
                "yardline_100": 24,
                "end_yard_line": "DET 14",
                "yards_gained": 0,
                "pass_attempt": 1,
                "interception": 1,
                "return_yards": 0,
                "passer_player_id": "00-qb",
                "passer_player_name": "D.Prescott",
                "interception_player_id": "00-db",
                "interception_player_name": "D.Reed",
                "desc": (
                    "(1:18) D.Prescott pass INTERCEPTED by D.Reed at DET 14. "
                    "D.Reed to DET 14 for no gain."
                ),
            },
        ]
    )


def _participation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nflverse_game_id": "2025_14_DAL_DET",
                "play_id": 299,
                "possession_team": "DAL",
                "offense_players": "00-qb;00-wr",
                "offense_names": "Dak Prescott;George Pickens",
                "offense_positions": "QB;WR",
                "offense_numbers": "4;3",
                "defense_players": "00-th;00-db",
                "defense_names": "Thomas Harper;D.J. Reed",
                "defense_positions": "S;CB",
                "defense_numbers": "12;4",
                "n_offense": 2,
                "n_defense": 2,
            },
            {
                "nflverse_game_id": "2025_14_DAL_DET",
                "play_id": 646,
                "possession_team": "DAL",
                "offense_players": "00-qb",
                "offense_names": "Dak Prescott",
                "offense_positions": "QB",
                "offense_numbers": "4",
                "defense_players": "00-lb",
                "defense_names": "Jack Campbell",
                "defense_positions": "LB",
                "defense_numbers": "46",
                "n_offense": 1,
                "n_defense": 1,
            },
            {
                "nflverse_game_id": "2025_14_DAL_DET",
                "play_id": 4839,
                "possession_team": "DAL",
                "offense_players": "00-qb",
                "offense_names": "Dak Prescott",
                "offense_positions": "QB",
                "offense_numbers": "4",
                "defense_players": "00-db",
                "defense_names": "D.J. Reed",
                "defense_positions": "CB",
                "defense_numbers": "4",
                "n_offense": 1,
                "n_defense": 1,
            },
        ]
    )


def _registry() -> dict[str, object]:
    return {
        "schema": "NFLFactorRegistryV3",
        "version": "v3-pilot",
        "factors": [
            {
                "factor_id": "NFL.EVENT.INTERCEPTION",
                "version": "v1",
                "status": "ACTIVE",
                "predicate": {
                    "field": "outcome_tags",
                    "operator": "CONTAINS",
                    "value": "INTERCEPTION",
                },
            },
            {
                "factor_id": "NFL.STATE.RED_ZONE",
                "version": "v1",
                "status": "ACTIVE",
                "predicate": {
                    "field": "pre_red_zone",
                    "operator": "EQ",
                    "value": True,
                },
            },
        ],
    }


def test_extraction_preserves_every_raw_row_and_one_primary_action() -> None:
    tables = build_game_fact_tables(
        _pbp_rows(),
        _participation_rows(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )

    assert len(tables.events) == 4
    assert len(tables.reconciliation) == 4
    assert tables.reconciliation["raw_row_preserved"].all()
    assert tables.events["primary_action"].notna().all()
    assert tables.events["event_id"].is_unique
    pass_row = tables.events.set_index("play_id").loc["299"]
    assert pass_row["visual_end_field_coordinate_0_100"] == 47
    assert (
        pass_row["field_orientation_semantics"]
        == "TEAM_NORMALIZED_AWAY_END_LEFT_HOME_END_RIGHT_NOT_PHYSICAL_STADIUM_DIRECTION"
    )


def test_reversed_safety_keeps_review_fact_but_not_safety_outcome() -> None:
    tables = build_game_fact_tables(
        _pbp_rows(),
        _participation_rows(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )
    row = tables.events.set_index("play_id").loc["646"]
    tags = json.loads(row["outcome_tags"])

    assert row["primary_action"] == "SACK"
    assert {"REVIEWED", "REVERSED"}.issubset(tags)
    assert "SAFETY" not in tags


def test_no_play_is_visible_but_not_factor_eligible() -> None:
    tables = build_game_fact_tables(
        _pbp_rows(),
        _participation_rows(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )
    row = tables.events.set_index("play_id").loc["778"]

    assert row["primary_action"] == "PENALTY"
    assert row["factor_eligible"] is False or not row["factor_eligible"]
    assert "NO_PLAY" in json.loads(row["outcome_tags"])


def test_interception_keeps_return_and_actor_facts_without_event_duplication() -> None:
    tables = build_game_fact_tables(
        _pbp_rows(),
        _participation_rows(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )
    row = tables.events.set_index("play_id").loc["4839"]
    tags = json.loads(row["outcome_tags"])
    players = tables.players[
        tables.players["event_id"].eq(row["event_id"])
    ]

    assert row["primary_action"] == "PASS"
    assert row["return_yards"] == 0
    assert {"INTERCEPTION", "TURNOVER", "POSSESSION_CHANGE"}.issubset(tags)
    assert (
        players["role"].eq("INTERCEPTOR")
        & players["player_id"].eq("00-db")
    ).any()


def test_injury_text_is_evidence_with_player_identity_not_roster_inference() -> None:
    tables = build_game_fact_tables(
        _pbp_rows(),
        _participation_rows(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )

    assert len(tables.injury_evidence) == 1
    row = tables.injury_evidence.iloc[0]
    assert row["evidence_type"] == "CONFIRMED_IN_GAME_INJURY"
    assert row["evidence_grade"] == "SOURCE_TEXT_DECLARED"
    assert row["player_id"] == "00-th"
    assert row["player_name"] == "Thomas Harper"
    assert "never inferred from roster differences" in row[
        "evidence_semantics"
    ].lower()


def test_active_registry_predicates_generate_traceable_factor_hits() -> None:
    tables = build_game_fact_tables(
        _pbp_rows(),
        _participation_rows(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )

    hit = tables.factor_hits[
        tables.factor_hits["factor_id"].eq("NFL.EVENT.INTERCEPTION")
    ]
    assert len(hit) == 1
    assert hit.iloc[0]["play_id"] == "4839"
    assert hit.iloc[0]["pbp_source_sha256"] == PBP_SHA
    assert (
        tables.factor_coverage.set_index("factor_id")
        .loc["NFL.EVENT.INTERCEPTION", "event_count"]
        == 1
    )


def test_two_point_conversion_is_try_even_when_native_play_type_is_pass() -> None:
    assert (
        _primary_action(
            {
                "play_type": "pass",
                "play_type_nfl": "PAT2",
                "two_point_conv_result": "success",
            }
        )
        == "TRY"
    )


def test_special_teams_and_turnover_direction_are_explicitly_bidirectional() -> None:
    kickoff = _transition_direction_semantics(
        action="KICKOFF",
        tags=("KICKOFF_RETURN",),
        posteam="DET",
        home_team="DET",
        away_team="DAL",
    )
    interception = _transition_direction_semantics(
        action="PASS",
        tags=("INTERCEPTION", "TURNOVER"),
        posteam="DAL",
        home_team="DET",
        away_team="DAL",
    )

    assert kickoff == (
        "BIDIRECTIONAL",
        "KICK_AND_POSSIBLE_RETURN_NET_STATE_ONLY",
    )
    assert interception == (
        "BIDIRECTIONAL",
        "OFFENSE_THEN_POSSIBLE_RETURN_NET_STATE_ONLY",
    )


def test_fair_catch_and_touchback_are_not_mislabeled_as_returns() -> None:
    punt_tags = _outcome_tags(
        {
            "play_type": "punt",
            "play_type_nfl": "PUNT",
            "return_yards": 0,
            "punt_fair_catch": 1,
        },
        next_possession="DET",
    )
    kickoff_tags = _outcome_tags(
        {
            "play_type": "kickoff",
            "play_type_nfl": "KICK_OFF",
            "return_yards": 0,
            "touchback": 1,
        },
        next_possession="DAL",
    )

    assert "PUNT_FAIR_CATCH" in punt_tags
    assert "PUNT_RETURN" not in punt_tags
    assert "KICKOFF_TOUCHBACK" in kickoff_tags
    assert "KICKOFF_RETURN" not in kickoff_tags
