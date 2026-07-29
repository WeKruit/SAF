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


def _episode_rows(*rows: dict[str, object]) -> pd.DataFrame:
    common = {
        "game_id": "2025_14_DAL_DET",
        "home_team": "DET",
        "away_team": "DAL",
        "season": 2025,
        "season_type": "REG",
        "week": 14,
        "qtr": 1,
        "time": "10:00",
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "play_deleted": 0,
        "goal_to_go": 0,
        "posteam_score": 0,
        "defteam_score": 0,
        "total_home_score": 0,
        "total_away_score": 0,
        "down": 1,
        "ydstogo": 10,
        "yrdln": "DAL 25",
        "yardline_100": 75,
    }
    return pd.DataFrame([{**common, **row} for row in rows])


def _build_episode_rows(*rows: dict[str, object]):
    return build_game_fact_tables(
        _episode_rows(*rows),
        pd.DataFrame(),
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )


def test_td_and_pat_are_distinct_episode_facts_in_one_score_sequence() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 10,
            "order_sequence": 10,
            "time_of_day": "2025-12-05T01:20:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "posteam": "DET",
            "defteam": "DAL",
            "touchdown": 1,
            "pass_touchdown": 1,
            "td_team": "DET",
            "total_home_score": 6,
        },
        {
            "play_id": 11,
            "order_sequence": 11,
            "time_of_day": "2025-12-05T01:20:30.000Z",
            "play_type": "extra_point",
            "play_type_nfl": "XP_KICK",
            "posteam": "DET",
            "defteam": "DAL",
            "posteam_score": 6,
            "extra_point_result": "good",
            "total_home_score": 7,
        },
    )
    events = tables.events.set_index("raw_play_id")
    touchdown = events.loc["10"]
    pat = events.loc["11"]

    assert touchdown["schema_version"] == "EpisodeFactV3"
    assert (
        touchdown["atomic_information_episode_id"]
        != pat["atomic_information_episode_id"]
    )
    assert touchdown["score_sequence_id"] == pat["score_sequence_id"]
    assert pd.notna(touchdown["score_sequence_id"])
    assert tables.events["atomic_information_episode_id"].is_unique
    assert tables.reconciliation["atomic_information_episode_id"].is_unique


def test_td_and_two_point_try_are_distinct_facts_in_one_score_sequence() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 13,
            "order_sequence": 13,
            "time_of_day": "2025-12-05T01:22:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DET",
            "defteam": "DAL",
            "touchdown": 1,
            "rush_touchdown": 1,
            "td_team": "DET",
            "total_home_score": 6,
        },
        {
            "play_id": 14,
            "order_sequence": 14,
            "time_of_day": "2025-12-05T01:22:30.000Z",
            "play_type": "pass",
            "play_type_nfl": "PAT2",
            "posteam": "DET",
            "defteam": "DAL",
            "posteam_score": 6,
            "two_point_conv_result": "success",
            "total_home_score": 8,
        },
    )
    rows = tables.events.set_index("raw_play_id")

    assert rows.loc["14", "primary_action"] == "TRY"
    assert (
        rows.loc["13", "atomic_information_episode_id"]
        != rows.loc["14", "atomic_information_episode_id"]
    )
    assert rows.loc["13", "score_sequence_id"] == rows.loc[
        "14", "score_sequence_id"
    ]


def test_episode_and_sequence_ids_do_not_depend_on_input_row_order() -> None:
    pbp = _episode_rows(
        {
            "play_id": 15,
            "order_sequence": 15,
            "time_of_day": "2025-12-05T01:23:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "posteam": "DET",
            "defteam": "DAL",
            "touchdown": 1,
            "td_team": "DET",
        },
        {
            "play_id": 16,
            "order_sequence": 16,
            "time_of_day": "2025-12-05T01:23:30.000Z",
            "play_type": "extra_point",
            "play_type_nfl": "XP_KICK",
            "posteam": "DET",
            "defteam": "DAL",
            "extra_point_result": "good",
        },
    )

    def identities(frame: pd.DataFrame) -> list[dict[str, object]]:
        return (
            build_game_fact_tables(
                frame,
                pd.DataFrame(),
                factor_registry=_registry(),
                pbp_source_sha256=PBP_SHA,
                participation_source_sha256=PARTICIPATION_SHA,
            )
            .events[
                [
                    "raw_play_id",
                    "atomic_information_episode_id",
                    "score_sequence_id",
                ]
            ]
            .to_dict("records")
        )

    assert identities(pbp) == identities(pbp.iloc[::-1].reset_index(drop=True))


def test_episode_fact_v3_contract_preserves_required_identity_and_provenance() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 12,
            "order_sequence": 12,
            "time_of_day": "2025-12-05T01:21:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DAL",
            "defteam": "DET",
        }
    )
    required = {
        "atomic_information_episode_id",
        "score_sequence_id",
        "adjudication_sequence_id",
        "source_interval_start",
        "source_interval_end",
        "source_resolution",
        "information_status",
        "stage_b_information_event_eligible",
        "final_sports_outcome_eligible",
        "known_at",
        "home_team",
        "away_team",
        "actor_team",
        "beneficiary_team",
        "actor_is_home",
        "beneficiary_is_home",
        "possession_is_home",
        "beneficiary_resolution_status",
        "source_hashes",
        "pbp_source_sha256",
        "participation_source_sha256",
    }

    assert required.issubset(tables.events.columns)
    assert json.loads(tables.events.iloc[0]["source_hashes"]) == [
        PBP_SHA,
        PARTICIPATION_SHA,
    ]


def test_pick_six_is_one_multitag_episode_with_defense_as_beneficiary() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 20,
            "order_sequence": 20,
            "time_of_day": "2025-12-05T01:25:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "INTERCEPTION",
            "posteam": "DAL",
            "defteam": "DET",
            "interception": 1,
            "touchdown": 1,
            "return_touchdown": 1,
            "td_team": "DET",
            "total_home_score": 6,
        }
    )
    row = tables.events.iloc[0]
    tags = set(json.loads(row["outcome_tags"]))

    assert len(tables.events) == 1
    assert {
        "INTERCEPTION",
        "TURNOVER",
        "TOUCHDOWN",
        "DEFENSIVE_TOUCHDOWN",
    }.issubset(tags)
    assert row["actor_team"] == "DET"
    assert row["beneficiary_team"] == "DET"
    assert bool(row["actor_is_home"])
    assert bool(row["beneficiary_is_home"])
    assert not bool(row["possession_is_home"])
    assert row["beneficiary_resolution_status"] == "RESOLVED_FINAL_SPORTS_RULE"


def test_double_turnover_offensive_td_uses_structured_scoring_team() -> None:
    tables = _build_episode_rows(
        {
            "game_id": "2025_05_TEN_ARI",
            "home_team": "ARI",
            "away_team": "TEN",
            "play_id": 4224,
            "order_sequence": 4224,
            "time_of_day": "2025-10-05T22:51:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "INTERCEPTION",
            "posteam": "TEN",
            "defteam": "ARI",
            "interception": 1,
            "fumble": 1,
            "fumble_lost": 1,
            "fumble_recovery_1_team": "TEN",
            "touchdown": 1,
            "return_touchdown": 1,
            "td_team": "TEN",
            "total_away_score": 6,
        }
    )
    row = tables.events.iloc[0]
    tags = set(json.loads(row["outcome_tags"]))

    assert {
        "INTERCEPTION",
        "FUMBLE_LOST",
        "TURNOVER",
        "TOUCHDOWN",
    }.issubset(tags)
    assert "DEFENSIVE_TOUCHDOWN" not in tags
    assert row["beneficiary_team"] == "TEN"
    assert not bool(row["beneficiary_is_home"])
    assert row["beneficiary_resolution_status"] == "RESOLVED_FINAL_SPORTS_RULE"


def test_turnover_touchdown_without_td_team_fails_closed() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 4225,
            "order_sequence": 4225,
            "time_of_day": "2025-10-05T22:52:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "INTERCEPTION",
            "posteam": "DAL",
            "defteam": "DET",
            "interception": 1,
            "touchdown": 1,
            "return_touchdown": 1,
        }
    )
    row = tables.events.iloc[0]
    tags = set(json.loads(row["outcome_tags"]))

    assert "TURNOVER" in tags
    assert "TOUCHDOWN" in tags
    assert "DEFENSIVE_TOUCHDOWN" not in tags
    assert pd.isna(row["beneficiary_team"])
    assert pd.isna(row["beneficiary_is_home"])
    assert row["beneficiary_resolution_status"] == "UNRESOLVED"


def test_turnover_on_downs_benefits_defense_without_reorienting_actor() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 30,
            "order_sequence": 30,
            "time_of_day": "2025-12-05T01:30:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DAL",
            "defteam": "DET",
            "fourth_down_failed": 1,
            "down": 4,
            "ydstogo": 2,
            "yards_gained": 1,
        }
    )
    row = tables.events.iloc[0]

    assert "TURNOVER_ON_DOWNS" in json.loads(row["outcome_tags"])
    assert row["actor_team"] == "DAL"
    assert row["beneficiary_team"] == "DET"
    assert not bool(row["actor_is_home"])
    assert bool(row["beneficiary_is_home"])


def test_unsupported_columns_cannot_create_adjudication_evidence() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 40,
            "order_sequence": 40,
            "time_of_day": "2025-12-05T01:35:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DAL",
            "defteam": "DET",
            "information_status": "PROVISIONAL",
            "adjudication_sequence_key": "reviewed-snap-40",
        },
    )
    row = tables.events.iloc[0]

    assert row["information_status"] == "FINAL"
    assert pd.isna(row["adjudication_sequence_id"])
    assert bool(row["stage_b_information_event_eligible"])
    assert bool(row["final_sports_outcome_eligible"])


def test_nullified_final_ruling_is_not_a_final_sports_outcome() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 42,
            "order_sequence": 42,
            "time_of_day": "2025-12-05T01:36:00.000Z",
            "play_type": "no_play",
            "play_type_nfl": "PENALTY",
            "posteam": "DAL",
            "defteam": "DET",
            "penalty": 1,
            "desc": "Penalty, no play.",
        }
    )
    row = tables.events.iloc[0]

    assert row["information_status"] == "FINAL"
    assert not bool(row["final_sports_outcome_eligible"])
    assert not bool(row["factor_eligible"])
    assert pd.isna(row["beneficiary_team"])
    assert row["beneficiary_resolution_status"] == "UNRESOLVED"


def test_source_intervals_are_left_closed_right_open_at_source_resolution() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 51,
            "order_sequence": 51,
            "time_of_day": "2025-12-05T01:40:02.123Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DAL",
            "defteam": "DET",
        },
    )
    point = tables.events.iloc[0]

    assert (
        pd.Timestamp(point["source_interval_end"])
        - pd.Timestamp(point["source_interval_start"])
        == pd.Timedelta(milliseconds=1)
    )
    assert point["source_resolution"] == "MILLISECOND"
    assert point["known_at"] == point["source_interval_end"]
    assert point["source_interval_semantics"] == "[START,END)"


def test_retried_snap_is_new_episode_without_an_invented_sequence() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 60,
            "order_sequence": 60,
            "time_of_day": "2025-12-05T01:45:00.000Z",
            "play_type": "no_play",
            "play_type_nfl": "PENALTY",
            "posteam": "DAL",
            "defteam": "DET",
            "penalty": 1,
        },
        {
            "play_id": 61,
            "order_sequence": 61,
            "time_of_day": "2025-12-05T01:45:20.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DAL",
            "defteam": "DET",
        },
    )
    rows = tables.events.set_index("raw_play_id")
    nullified = rows.loc["60"]
    retry = rows.loc["61"]

    assert (
        nullified["atomic_information_episode_id"]
        != retry["atomic_information_episode_id"]
    )
    assert pd.isna(nullified["adjudication_sequence_id"])
    assert pd.isna(retry["adjudication_sequence_id"])
    assert not bool(nullified["final_sports_outcome_eligible"])
    assert bool(retry["final_sports_outcome_eligible"])


def test_administrative_episode_does_not_invent_a_beneficiary() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 62,
            "order_sequence": 62,
            "time_of_day": "2025-12-05T01:46:00.000Z",
            "play_type": None,
            "play_type_nfl": "END_QUARTER",
            "desc": "END QUARTER 1",
        }
    )
    row = tables.events.iloc[0]

    assert row["primary_action"] == "ADMIN"
    assert pd.isna(row["beneficiary_team"])
    assert pd.isna(row["beneficiary_is_home"])
    assert row["beneficiary_resolution_status"] == "UNRESOLVED"
    assert not bool(row["final_sports_outcome_eligible"])


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
    assert row["information_status"] == "FINAL"
    assert bool(row["final_sports_outcome_eligible"])
    assert not bool(row["stage_b_information_event_eligible"])
    assert pd.isna(row["adjudication_sequence_id"])
    assert pd.isna(row["known_at"])


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
