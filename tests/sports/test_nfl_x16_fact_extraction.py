from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prediction_market.sports.nfl_x16_fact_extraction import (
    _outcome_tags,
    _primary_action,
    _transition_direction_semantics,
    build_game_fact_tables,
)


PBP_SHA = "sha256:" + "a" * 64
PARTICIPATION_SHA = "sha256:" + "b" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_REGISTRY = (
    PROJECT_ROOT / "registries/factors/nfl_factor_registry_v4.json"
)


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
            "qtr": 1,
            "time": "00:01",
            "time_of_day": "2025-12-05T01:30:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DAL",
            "defteam": "DET",
            "fourth_down_failed": 1,
            "down": 4,
            "ydstogo": 2,
            "yards_gained": 1,
        },
        {
            "play_id": 31,
            "order_sequence": 31,
            "qtr": 2,
            "time": "15:00",
            "time_of_day": "2025-12-05T01:31:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DET",
            "defteam": "DAL",
        },
    )
    row = tables.events.set_index("play_id").loc["30"]

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


def test_replay_duplicate_injury_mentions_emit_one_semantic_fact() -> None:
    pbp = _episode_rows(
        {
            "game_id": "2025_04_CAR_NE",
            "home_team": "NE",
            "away_team": "CAR",
            "play_id": 3821,
            "order_sequence": 3821,
            "qtr": 4,
            "time": "02:20",
            "time_of_day": "2025-09-28T19:50:40.793Z",
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "posteam": "CAR",
            "defteam": "NE",
            "desc": (
                "(2:20) 14-A.Dalton pass short right to 82-T.Tremble for "
                "6 yards, TOUCHDOWN. CAR-82-T.Tremble was injured during "
                "the play. The Replay Official reviewed the runner broke "
                "the plane ruling, and the play was REVERSED. 14-A.Dalton "
                "pass short right to 82-T.Tremble to NE 1 for 5 yards. "
                "CAR-82-T.Tremble was injured during the play. "
                "CAR-85-J.Mitchell was injured during the play."
            ),
        },
        {
            "game_id": "2025_04_CAR_NE",
            "home_team": "NE",
            "away_team": "CAR",
            "play_id": 3900,
            "order_sequence": 3900,
            "qtr": 4,
            "time": "01:30",
            "time_of_day": "2025-09-28T19:52:00.000Z",
            "play_type": "no_play",
            "play_type_nfl": "GAME_ADVICE",
            "posteam": "CAR",
            "defteam": "NE",
            "desc": (
                "Injury Update: CAR-82-T.Tremble has returned to the game."
            ),
        },
        {
            "game_id": "2025_04_CAR_NE",
            "home_team": "NE",
            "away_team": "CAR",
            "play_id": 4000,
            "order_sequence": 4000,
            "qtr": 4,
            "time": "00:45",
            "time_of_day": "2025-09-28T19:53:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "CAR",
            "defteam": "NE",
            "desc": "CAR-82-T.Tremble was injured during the play.",
        },
    )
    participation = pd.DataFrame(
        [
            {
                "nflverse_game_id": "2025_04_CAR_NE",
                "play_id": play_id,
                "offense_players": "00-0037005;00-0036290",
                "offense_names": "Tommy Tremble;James Mitchell",
                "offense_positions": "TE;TE",
                "offense_numbers": "82;85",
                "defense_players": "",
                "defense_names": "",
                "defense_positions": "",
                "defense_numbers": "",
                "n_offense": 2,
                "n_defense": 0,
            }
            for play_id in (3821, 3900, 4000)
        ]
    )

    tables = build_game_fact_tables(
        pbp,
        participation,
        factor_registry=_registry(),
        pbp_source_sha256=PBP_SHA,
        participation_source_sha256=PARTICIPATION_SHA,
    )

    facts = tables.injury_evidence[
        [
            "event_id",
            "evidence_type",
            "status",
            "player_id",
            "player_name",
        ]
    ]
    assert facts.to_records(index=False).tolist() == [
        (
            "2025_04_CAR_NE:episode:3821",
            "CONFIRMED_IN_GAME_INJURY",
            "INJURED",
            "00-0037005",
            "Tommy Tremble",
        ),
        (
            "2025_04_CAR_NE:episode:3821",
            "CONFIRMED_IN_GAME_INJURY",
            "INJURED",
            "00-0036290",
            "James Mitchell",
        ),
        (
            "2025_04_CAR_NE:episode:3900",
            "CONFIRMED_RETURN_TO_GAME",
            "RETURNED",
            "00-0037005",
            "Tommy Tremble",
        ),
        (
            "2025_04_CAR_NE:episode:4000",
            "CONFIRMED_IN_GAME_INJURY",
            "INJURED",
            "00-0037005",
            "Tommy Tremble",
        ),
    ]


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


def _v4_registry() -> dict[str, object]:
    return json.loads(V4_REGISTRY.read_text(encoding="utf-8"))


def _v4_factor(factor_id: str) -> dict[str, object]:
    factors = _v4_registry()["factors"]
    assert isinstance(factors, list)
    matches = [
        factor
        for factor in factors
        if isinstance(factor, dict) and factor.get("factor_id") == factor_id
    ]
    assert len(matches) == 1
    return matches[0]


def _tag_predicate(tag: str) -> dict[str, object]:
    return {
        "field": "outcome_tags",
        "operator": "CONTAINS",
        "value": tag,
    }


def test_turnover_on_downs_requires_a_clean_fourth_down_possession_award() -> None:
    failed_fourth_down = {
        "play_type": "pass",
        "play_type_nfl": "PASS",
        "posteam": "DAL",
        "down": 4,
        "fourth_down_failed": 1,
    }

    strict_tags = _outcome_tags(
        failed_fourth_down,
        next_possession="DET",
    )
    interception_tags = _outcome_tags(
        {**failed_fourth_down, "interception": 1},
        next_possession="DET",
    )
    lost_fumble_tags = _outcome_tags(
        {
            **failed_fourth_down,
            "fumble": 1,
            "fumble_lost": 1,
        },
        next_possession="DET",
    )
    no_award_tags = _outcome_tags(
        failed_fourth_down,
        next_possession=None,
    )

    assert {"FOURTH_DOWN_FAILED", "TURNOVER_ON_DOWNS", "TURNOVER"}.issubset(
        strict_tags
    )
    assert "TURNOVER_ON_DOWNS" not in interception_tags
    assert "TURNOVER_ON_DOWNS" not in lost_fumble_tags
    assert "TURNOVER_ON_DOWNS" not in no_award_tags


def test_turnover_on_downs_does_not_borrow_next_half_kickoff_possession() -> None:
    tables = _build_episode_rows(
        {
            "game_id": "2025_07_NE_TEN",
            "home_team": "TEN",
            "away_team": "NE",
            "play_id": 1860,
            "order_sequence": 1860,
            "qtr": 2,
            "time": "00:03",
            "game_seconds_remaining": 1803,
            "time_of_day": "2025-10-19T19:05:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "posteam": "NE",
            "defteam": "TEN",
            "down": 4,
            "ydstogo": 15,
            "pass_attempt": 1,
            "incomplete_pass": 1,
            "fourth_down_failed": 1,
        },
        {
            "game_id": "2025_07_NE_TEN",
            "home_team": "TEN",
            "away_team": "NE",
            "play_id": 1861,
            "order_sequence": 1861,
            "qtr": 3,
            "time": "15:00",
            "game_seconds_remaining": 1800,
            "time_of_day": "2025-10-19T19:20:00.000Z",
            "play_type": "kickoff",
            "play_type_nfl": "KICK_OFF",
            "posteam": "TEN",
            "defteam": "NE",
            "touchback": 1,
        },
    )
    row = tables.events.set_index("play_id").loc["1860"]
    tags = set(json.loads(row["outcome_tags"]))

    assert row["primary_action"] == "PASS"
    assert "FOURTH_DOWN_FAILED" in tags
    assert "TURNOVER_ON_DOWNS" not in tags


def _terminal_end_game_case(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    play_id: int,
    quarter: int,
    posteam: str,
    play_fields: dict[str, object],
) -> pd.Series:
    defteam = home_team if posteam == away_team else away_team
    tables = _build_episode_rows(
        {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "play_id": play_id,
            "order_sequence": play_id,
            "qtr": quarter,
            "time": "00:02",
            "game_seconds_remaining": 2,
            "time_of_day": "2025-12-01T04:00:00.000Z",
            "posteam": posteam,
            "defteam": defteam,
            "down": 4,
            "ydstogo": 10,
            "fourth_down_failed": 1,
            **play_fields,
        },
        {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "play_id": play_id + 1,
            "order_sequence": play_id + 1,
            "qtr": quarter,
            "time": "00:00",
            "game_seconds_remaining": 0,
            "time_of_day": "2025-12-01T04:00:02.000Z",
            "play_type_nfl": "END_GAME",
            "posteam": defteam,
            "defteam": posteam,
        },
    )
    return tables.events.set_index("play_id").loc[str(play_id)]


def test_terminal_admin_rows_never_supply_downs_award_evidence() -> None:
    cases = {
        "2025_17_LA_ATL:3874:q4_incomplete": _terminal_end_game_case(
            game_id="2025_17_LA_ATL",
            home_team="ATL",
            away_team="LA",
            play_id=3874,
            quarter=4,
            posteam="LA",
            play_fields={
                "play_type": "pass",
                "play_type_nfl": "PASS",
                "pass_attempt": 1,
                "incomplete_pass": 1,
            },
        ),
        "2025_07_PHI_MIN:4065:q4_kneel": _terminal_end_game_case(
            game_id="2025_07_PHI_MIN",
            home_team="MIN",
            away_team="PHI",
            play_id=4065,
            quarter=4,
            posteam="PHI",
            play_fields={
                "play_type": "qb_kneel",
                "play_type_nfl": "RUSH",
                "qb_kneel": 1,
            },
        ),
        "2025_12_NYG_DET:4958:q5_incomplete": _terminal_end_game_case(
            game_id="2025_12_NYG_DET",
            home_team="DET",
            away_team="NYG",
            play_id=4958,
            quarter=5,
            posteam="NYG",
            play_fields={
                "play_type": "pass",
                "play_type_nfl": "PASS",
                "pass_attempt": 1,
                "incomplete_pass": 1,
            },
        ),
    }

    false_positives = {
        case_id
        for case_id, row in cases.items()
        if "TURNOVER_ON_DOWNS" in json.loads(row["outcome_tags"])
    }
    assert false_positives == set()


def test_terminal_posteam_remains_raw_audit_data_not_award_evidence() -> None:
    row = _terminal_end_game_case(
        game_id="2025_17_LA_ATL",
        home_team="ATL",
        away_team="LA",
        play_id=3874,
        quarter=4,
        posteam="LA",
        play_fields={
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "pass_attempt": 1,
            "incomplete_pass": 1,
        },
    )

    assert row["next_observed_possession"] == "ATL"
    assert "next_observed_possession_semantics" in row.index
    assert (
        row["next_observed_possession_semantics"]
        == "NEXT_LATER_SOURCE_ROW_WITH_NON_NULL_POSTEAM_AUDIT_ONLY_"
        "NOT_DOWNS_AWARD_EVIDENCE"
    )


def test_turnover_on_downs_accepts_q3_to_q4_actual_possession_award() -> None:
    tables = _build_episode_rows(
        {
            "play_id": 70,
            "order_sequence": 70,
            "qtr": 3,
            "time": "00:01",
            "time_of_day": "2025-12-05T03:00:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "posteam": "DAL",
            "defteam": "DET",
            "down": 4,
            "fourth_down_failed": 1,
            "incomplete_pass": 1,
        },
        {
            "play_id": 71,
            "order_sequence": 71,
            "qtr": 4,
            "time": "15:00",
            "time_of_day": "2025-12-05T03:01:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DET",
            "defteam": "DAL",
        },
    )
    row = tables.events.set_index("play_id").loc["70"]

    assert "TURNOVER_ON_DOWNS" in json.loads(row["outcome_tags"])


def _cross_period_fourth_down_row(
    start_quarter: int,
    next_quarter: int,
) -> pd.Series:
    tables = _build_episode_rows(
        {
            "play_id": 80,
            "order_sequence": 80,
            "qtr": start_quarter,
            "time": "00:01",
            "time_of_day": "2025-12-05T04:00:00.000Z",
            "play_type": "pass",
            "play_type_nfl": "PASS",
            "posteam": "DAL",
            "defteam": "DET",
            "down": 4,
            "fourth_down_failed": 1,
            "incomplete_pass": 1,
        },
        {
            "play_id": 81,
            "order_sequence": 81,
            "qtr": next_quarter,
            "time": "15:00",
            "time_of_day": "2025-12-05T04:01:00.000Z",
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "posteam": "DET",
            "defteam": "DAL",
        },
    )
    return tables.events.set_index("play_id").loc["80"]


def test_postseason_odd_to_even_extra_period_keeps_possession_continuity() -> None:
    missing_turnovers = {
        (start_quarter, next_quarter)
        for start_quarter, next_quarter in ((5, 6), (7, 8))
        if "TURNOVER_ON_DOWNS"
        not in json.loads(
            _cross_period_fourth_down_row(
                start_quarter,
                next_quarter,
            )["outcome_tags"]
        )
    }

    assert missing_turnovers == set()


def test_postseason_even_to_odd_extra_period_breaks_possession_continuity() -> None:
    row = _cross_period_fourth_down_row(6, 7)

    assert "TURNOVER_ON_DOWNS" not in json.loads(row["outcome_tags"])


def test_regulation_to_overtime_breaks_possession_continuity() -> None:
    row = _cross_period_fourth_down_row(4, 5)

    assert "TURNOVER_ON_DOWNS" not in json.loads(row["outcome_tags"])


def test_kickoff_return_requires_a_returner_and_excludes_terminal_endings() -> None:
    credited_return = {
        "play_type": "kickoff",
        "play_type_nfl": "KICK_OFF",
        "posteam": "DET",
        "return_yards": 12,
        "kickoff_returner_player_id": "00-returner",
    }

    assert "KICKOFF_RETURN" in _outcome_tags(
        credited_return,
        next_possession="DET",
    )
    assert "KICKOFF_RETURN" not in _outcome_tags(
        {
            "play_type": "kickoff",
            "play_type_nfl": "KICK_OFF",
            "posteam": "DET",
            "return_yards": 0,
        },
        next_possession="DET",
    )
    for terminal_field in (
        "touchback",
        "kickoff_fair_catch",
        "kickoff_out_of_bounds",
        "kickoff_downed",
        "own_kickoff_recovery",
    ):
        tags = _outcome_tags(
            {**credited_return, terminal_field: 1},
            next_possession="DET",
        )
        assert "KICKOFF_RETURN" not in tags


def test_kicking_team_recovery_is_not_labeled_as_onside_intent() -> None:
    tags = _outcome_tags(
        {
            "play_type": "kickoff",
            "play_type_nfl": "KICK_OFF",
            "posteam": "DET",
            "own_kickoff_recovery": 1,
        },
        next_possession="DAL",
    )

    assert "KICKING_TEAM_KICKOFF_RECOVERY" in tags
    assert "ONSIDE_RECOVERY" not in tags


def test_fumble_not_lost_is_an_explicit_atomic_tag() -> None:
    retained = _outcome_tags(
        {
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "fumble": 1,
            "fumble_lost": 0,
        },
        next_possession="DAL",
    )
    lost = _outcome_tags(
        {
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "fumble": 1,
            "fumble_lost": 1,
        },
        next_possession="DET",
    )

    assert "FUMBLE_NOT_LOST" in retained
    assert "FUMBLE_NOT_LOST" not in lost


def test_v4_registry_is_direct_authoritative_and_removes_rejected_aliases() -> None:
    registry = _v4_registry()
    factors = registry["factors"]
    assert isinstance(factors, list)
    factor_ids = {
        factor["factor_id"]
        for factor in factors
        if isinstance(factor, dict)
    }

    assert registry["schema"] == "NFLFactorRegistryV4"
    assert registry["version"] == "v4"
    assert registry["status"] == "AUTHORITATIVE"
    assert len(factors) == 59
    assert "NFL.COMBO.FOURTH_DOWN_FAILURE" not in factor_ids
    assert "NFL.SPECIAL.ONSIDE_RECOVERY" not in factor_ids
    assert "NFL.SPECIAL.KICKING_TEAM_KICKOFF_RECOVERY" in factor_ids


def test_v4_registry_encodes_the_reviewed_semantic_boundaries() -> None:
    assert _v4_factor("NFL.SCORE.EXTRA_POINT_FAILED")["predicate"] == {
        "any": [
            _tag_predicate("EXTRA_POINT_FAILED"),
            _tag_predicate("EXTRA_POINT_BLOCKED"),
        ]
    }
    assert _v4_factor("NFL.SPECIAL.FIELD_GOAL_MISSED")["predicate"] == {
        "all": [
            _tag_predicate("FIELD_GOAL_MISSED"),
            {"not": _tag_predicate("FIELD_GOAL_BLOCKED")},
        ]
    }
    assert _v4_factor("NFL.SPECIAL.TOUCHBACK")["predicate"] == {
        "all": [
            {
                "field": "primary_action",
                "operator": "IN",
                "value": ["KICKOFF", "PUNT"],
            },
            _tag_predicate("TOUCHBACK"),
        ]
    }
    assert _v4_factor("NFL.STATE.RED_ZONE")["predicate"] == {
        "all": [
            {
                "field": "pre_red_zone",
                "operator": "EQ",
                "value": True,
            },
            {
                "field": "down",
                "operator": "IN",
                "value": [1, 2, 3, 4],
            },
        ]
    }
    assert (
        _v4_factor("NFL.STATE.DISTANCE_GT_10")["meaning"]
        == "The source pre-play yards-to-go value exceeds ten; on "
        "goal-to-go plays it is distance to the goal line."
    )
    assert _v4_factor("NFL.COMBO.RED_ZONE_TURNOVER")["predicate"] == {
        "all": [
            {
                "field": "pre_red_zone",
                "operator": "EQ",
                "value": True,
            },
            {
                "field": "down",
                "operator": "IN",
                "value": [1, 2, 3, 4],
            },
            {
                "any": [
                    _tag_predicate("INTERCEPTION"),
                    _tag_predicate("FUMBLE_LOST"),
                    _tag_predicate("TURNOVER_ON_DOWNS"),
                ]
            },
        ]
    }


def test_v4_registry_adds_available_atomic_factors() -> None:
    expected_predicates = {
        "NFL.ACTION.QB_KNEEL": {
            "field": "primary_action",
            "operator": "EQ",
            "value": "KNEEL",
        },
        "NFL.ACTION.QB_SCRAMBLE": _tag_predicate("QB_SCRAMBLE"),
        "NFL.EVENT.THIRD_DOWN_CONVERTED": _tag_predicate(
            "THIRD_DOWN_CONVERTED"
        ),
        "NFL.EVENT.THIRD_DOWN_FAILED": _tag_predicate("THIRD_DOWN_FAILED"),
        "NFL.EVENT.FOURTH_DOWN_CONVERTED": _tag_predicate(
            "FOURTH_DOWN_CONVERTED"
        ),
        "NFL.EVENT.FUMBLE_NOT_LOST": _tag_predicate("FUMBLE_NOT_LOST"),
        "NFL.ADMIN.REVIEW_UPHELD": _tag_predicate("REVIEW_UPHELD"),
        "NFL.SPECIAL.BLOCKED_EXTRA_POINT": _tag_predicate(
            "EXTRA_POINT_BLOCKED"
        ),
    }

    assert {
        factor_id: _v4_factor(factor_id)["predicate"]
        for factor_id in expected_predicates
    } == expected_predicates
