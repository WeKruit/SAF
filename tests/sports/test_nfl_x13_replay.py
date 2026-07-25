from __future__ import annotations

import math
from types import SimpleNamespace


def _play(
    play_id: int,
    order_sequence: int,
    *,
    game_id: str = "2025_10_JAX_HOU",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": game_id,
        "play_id": play_id,
        "order_sequence": order_sequence,
        "_raw_record_ordinal": order_sequence,
        "time_of_day": f"2025-11-09T18:00:{order_sequence:02d}Z",
        "qtr": 1,
        "quarter_seconds_remaining": 900 - order_sequence,
        "game_seconds_remaining": 3600 - order_sequence,
        "home_team": "HOU",
        "away_team": "JAX",
        "posteam": "JAX",
        "defteam": "HOU",
        "fixed_drive": 1,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 75,
        "goal_to_go": 0,
        "total_home_score": 7,
        "total_away_score": 10,
        "posteam_score": 10,
        "defteam_score": 7,
        "posteam_score_post": 10,
        "defteam_score_post": 7,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
    }
    row.update(overrides)
    return row


def test_canonical_events_sort_and_preserve_missing_evidence() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        CanonicalGameEventV1,
        canonical_events_sha256,
        canonicalize_game_events,
    )

    rows = [
        {
            "game_id": "2025_10_JAX_HOU",
            "play_id": 20,
            "order_sequence": 2,
            "_raw_record_ordinal": 9,
            "qtr": 1,
            "quarter_seconds_remaining": 850,
            "home_team": "HOU",
            "away_team": "JAX",
            "posteam": "JAX",
            "defteam": "HOU",
            "play_type": "pass",
            "pass_attempt": 1,
        },
        {
            "game_id": "2025_10_JAX_HOU",
            "play_id": 10,
            "order_sequence": 1,
            "_raw_record_ordinal": 4,
            "time_of_day": "2025-11-09T18:00:00Z",
            "qtr": 1,
            "quarter_seconds_remaining": 900,
            "home_team": "HOU",
            "away_team": "JAX",
            "posteam": "JAX",
            "defteam": "HOU",
            "play_type": "run",
            "rush_attempt": 1,
        },
    ]
    participation = [
        {
            "nflverse_game_id": "2025_10_JAX_HOU",
            "play_id": 10,
            "offense_names": ";".join(f"JAX {index}" for index in range(11)),
            "defense_names": ";".join(f"HOU {index}" for index in range(11)),
            "offense_players": ";".join(f"jax-{index}" for index in range(11)),
            "defense_players": ";".join(f"hou-{index}" for index in range(11)),
            "offense_formation": "SHOTGUN",
            "offense_personnel": "11",
            "defense_personnel": "4-2-5",
        }
    ]

    first = canonicalize_game_events(rows, participation)
    second = canonicalize_game_events(rows, participation)

    assert all(isinstance(event, CanonicalGameEventV1) for event in first)
    assert [event.play_id for event in first] == ["10", "20"]
    assert first[0].source_time_status == "point"
    assert first[0].play_family == "run"
    assert first[0].participation_status == "present_complete"
    assert len(first[0].offense_names) == 11
    assert first[1].source_time_status == "missing"
    assert first[1].source_time_start_utc is None
    assert first[1].participation_status == "missing"
    assert first[1].offense_names == ()
    assert canonical_events_sha256(first) == canonical_events_sha256(second)


def test_canonical_event_carries_state_roles_personnel_and_roster_delta() -> None:
    from prediction_market.sports.nfl_x13_replay import canonicalize_game_events

    rows = [
        _play(10, 1, play_type="run", rush_attempt=1),
        _play(
            20,
            2,
            source_time_start_utc="2025-11-09T18:00:02Z",
            source_time_end_utc="2025-11-09T18:00:03Z",
            play_type="pass",
            pass_attempt=1,
            complete_pass=1,
            passer_player_name="Trevor Lawrence",
            receiver_player_name="Brian Thomas Jr.",
            passing_yards=17,
            receiving_yards=17,
            play_direction="right",
            field_coordinate_0_100=25,
            field_orientation_semantics=(
                "canonical_team_end_orientation_not_tracking_coordinate"
            ),
            shotgun=1,
            timeout=1,
            timeout_team="JAX",
            penalty=1,
            penalty_team="HOU",
            penalty_yards=5,
            penalty_type="Defensive Holding",
            desc="Pass complete. Defensive holding, accepted.",
            post_state={
                "qtr": 1,
                "quarter_seconds_remaining": 892,
                "game_seconds_remaining": 3592,
                "home_team": "HOU",
                "away_team": "JAX",
                "posteam": "JAX",
                "defteam": "HOU",
                "fixed_drive": 1,
                "down": 1,
                "ydstogo": 10,
                "yardline_100": 58,
                "goal_to_go": 0,
                "total_home_score": 7,
                "total_away_score": 10,
                "home_timeouts_remaining": 3,
                "away_timeouts_remaining": 2,
            },
        ),
    ]
    first_offense = [f"JAX {index}" for index in range(11)]
    second_offense = [*first_offense[:-1], "JAX replacement"]
    participation = [
        {
            "nflverse_game_id": "2025_10_JAX_HOU",
            "play_id": 10,
            "offense_names": first_offense,
            "defense_names": [f"HOU {index}" for index in range(11)],
            "offense_players": [f"jax-{index}" for index in range(11)],
            "defense_players": [f"hou-{index}" for index in range(11)],
            "offense_formation": "SINGLEBACK",
            "offense_personnel": "11",
            "defense_personnel": "4-2-5",
        },
        {
            "nflverse_game_id": "2025_10_JAX_HOU",
            "play_id": 20,
            "offense_names": second_offense,
            "defense_names": [f"HOU {index}" for index in range(11)],
            "offense_players": [
                *(f"jax-{index}" for index in range(10)),
                "jax-replacement",
            ],
            "defense_players": [f"hou-{index}" for index in range(11)],
            "offense_formation": "SHOTGUN",
            "offense_personnel": "11",
            "defense_personnel": "4-2-5",
        },
    ]

    event = canonicalize_game_events(rows, participation)[1]

    assert event.source_time_status == "interval"
    assert event.quarter == 1
    assert event.clock_seconds_remaining == 898
    assert event.home_score == 7
    assert event.away_score == 10
    assert event.score_margin == 3
    assert event.possession_team == event.offense_team == "JAX"
    assert event.defense_team == "HOU"
    assert event.drive_id == "1"
    assert (event.down, event.distance, event.yardline_100) == (1, 10, 75)
    assert event.goal_to_go is False
    assert event.direction == "right"
    assert event.description == "Pass complete. Defensive holding, accepted."
    assert event.field_coordinate_0_100 == 25
    assert (
        event.field_orientation_semantics
        == "canonical_team_end_orientation_not_tracking_coordinate"
    )
    assert event.player_roles == {
        "passer": "Trevor Lawrence",
        "receiver": "Brian Thomas Jr.",
    }
    assert event.formation == "SHOTGUN"
    assert event.offense_personnel == "11"
    assert event.defense_personnel == "4-2-5"
    assert event.timeout is True
    assert event.timeout_team == "JAX"
    assert event.home_timeouts_remaining == 3
    assert event.away_timeouts_remaining == 3
    assert event.penalty is True
    assert event.penalty_status == "accepted"
    assert event.penalty_team == "HOU"
    assert event.penalty_yards == 5
    assert event.penalty_type == "Defensive Holding"
    assert event.review is False
    assert event.review_result == "none"
    assert event.no_play is False
    assert event.deleted is False
    assert event.pre_state.home_score == 7
    assert event.post_state is not None
    assert event.post_state.yardline_100 == 58
    change = event.between_play_roster_change
    assert change is not None
    assert change["offense_entering_names"] == ("JAX replacement",)
    assert change["offense_leaving_names"] == ("JAX 10",)
    assert "substitution" not in change["semantics"]
    assert "injury" not in change["semantics"]


def test_touchdown_episode_absorbs_admin_try_and_retry_until_kickoff() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_finalized_episodes,
        canonicalize_game_events,
    )

    rows = [
        _play(
            100,
            1,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            total_home_score=0,
            total_away_score=0,
            posteam_score=0,
            defteam_score=0,
            posteam_score_post=6,
            defteam_score_post=0,
            play_type="pass",
            pass_attempt=1,
            complete_pass=1,
            touchdown=1,
            pass_touchdown=1,
            td_team="JAX",
            passer_player_name="Trevor Lawrence",
            receiver_player_name="Brian Thomas Jr.",
            post_state={
                "qtr": 1,
                "quarter_seconds_remaining": 820,
                "game_seconds_remaining": 3520,
                "home_team": "JAX",
                "away_team": "HOU",
                "posteam": "JAX",
                "defteam": "HOU",
                "total_home_score": 6,
                "total_away_score": 0,
            },
        ),
        _play(
            101,
            2,
            home_team="JAX",
            away_team="HOU",
            posteam=None,
            defteam=None,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            timeout=1,
            timeout_team="HOU",
        ),
        _play(
            102,
            3,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=6,
            defteam_score=0,
            posteam_score_post=6,
            defteam_score_post=0,
            play_type="no_play",
            two_point_attempt=1,
            penalty=1,
            desc="Two-point try. Penalty accepted, no play.",
        ),
        _play(
            103,
            4,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=6,
            defteam_score=0,
            posteam_score_post=7,
            defteam_score_post=0,
            play_type="extra_point",
            extra_point_attempt=1,
            extra_point_result="good",
            post_state={
                "qtr": 1,
                "quarter_seconds_remaining": 820,
                "game_seconds_remaining": 3520,
                "home_team": "JAX",
                "away_team": "HOU",
                "posteam": "JAX",
                "defteam": "HOU",
                "total_home_score": 7,
                "total_away_score": 0,
            },
        ),
        _play(
            104,
            5,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=7,
            defteam_score=0,
            posteam_score_post=7,
            defteam_score_post=0,
            play_type="kickoff",
            kickoff_attempt=1,
        ),
    ]

    episodes = build_finalized_episodes(canonicalize_game_events(rows))

    assert len(episodes) == 2
    touchdown, kickoff = episodes
    assert touchdown.episode_type == "touchdown"
    assert touchdown.play_ids == ("100", "101", "102", "103")
    assert touchdown.penalty_status == "accepted_no_play"
    assert touchdown.nullified is False
    assert touchdown.pre_state.home_score == 0
    assert touchdown.post_state is not None
    assert touchdown.post_state.home_score == 7
    assert kickoff.play_ids == ("104",)
    assert kickoff.episode_type == "kickoff"


def test_touchdown_episode_uses_last_observed_internal_post_state_when_admin_tail_is_missing() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_finalized_episodes,
        canonicalize_game_events,
    )

    rows = [
        _play(
            110,
            1,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            total_home_score=0,
            total_away_score=0,
            posteam_score=0,
            defteam_score=0,
            posteam_score_post=6,
            defteam_score_post=0,
            play_type="run",
            rush_attempt=1,
            touchdown=1,
            rush_touchdown=1,
            post_state={
                "qtr": 1,
                "quarter_seconds_remaining": 810,
                "game_seconds_remaining": 3510,
                "home_team": "JAX",
                "away_team": "HOU",
                "posteam": "JAX",
                "defteam": "HOU",
                "total_home_score": 6,
                "total_away_score": 0,
            },
        ),
        _play(
            111,
            2,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=6,
            defteam_score=0,
            posteam_score_post=7,
            defteam_score_post=0,
            play_type="extra_point",
            extra_point_attempt=1,
            extra_point_result="good",
            post_state={
                "qtr": 1,
                "quarter_seconds_remaining": 810,
                "game_seconds_remaining": 3510,
                "home_team": "JAX",
                "away_team": "HOU",
                "posteam": "JAX",
                "defteam": "HOU",
                "total_home_score": 7,
                "total_away_score": 0,
            },
        ),
        _play(
            112,
            3,
            home_team="JAX",
            away_team="HOU",
            posteam=None,
            defteam=None,
            posteam_score=None,
            defteam_score=None,
            posteam_score_post=None,
            defteam_score_post=None,
            total_home_score=None,
            total_away_score=None,
            qtr=None,
            quarter_seconds_remaining=None,
            game_seconds_remaining=None,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            timeout=1,
            timeout_team="HOU",
        ),
        _play(113, 4, play_type="kickoff", kickoff_attempt=1),
    ]

    touchdown = build_finalized_episodes(canonicalize_game_events(rows))[0]

    assert touchdown.play_ids == ("110", "111", "112")
    assert touchdown.post_state is not None
    assert touchdown.post_state.home_score == 7
    assert touchdown.post_state.game_seconds_remaining == 3510


def test_deleted_no_play_and_penalty_priority_are_explicit() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_finalized_episodes,
        canonicalize_game_events,
    )

    rows = [
        _play(
            200,
            1,
            play_type="pass",
            pass_attempt=1,
            touchdown=1,
            play_deleted=1,
        ),
        _play(
            201,
            2,
            play_type="no_play",
            pass_attempt=1,
            touchdown=1,
            penalty=1,
            desc="Touchdown nullified by an accepted penalty.",
        ),
        _play(202, 3, play_type="run", rush_attempt=1, touchdown=1),
        _play(
            203,
            4,
            play_type="extra_point",
            extra_point_attempt=1,
            penalty=1,
            desc="Penalty accepted.",
        ),
        _play(
            204,
            5,
            play_type="no_play",
            penalty=1,
            desc="Penalty declined.",
        ),
        _play(
            205,
            6,
            play_type="no_play",
            two_point_attempt=1,
            penalty=1,
            desc="Offsetting penalties. Retry.",
        ),
        _play(206, 7, play_type="kickoff", kickoff_attempt=1),
    ]

    events = canonicalize_game_events(rows)
    episodes = build_finalized_episodes(events)

    assert [event.penalty_status for event in events[3:6]] == [
        "accepted",
        "declined",
        "offsetting",
    ]
    deleted = episodes[0]
    assert deleted.play_ids == ("200",)
    assert deleted.episode_type == "touchdown"
    assert deleted.audit_only is True
    assert deleted.nullified is False
    no_play = episodes[1]
    assert no_play.play_ids == ("201",)
    assert no_play.episode_type == "touchdown"
    assert no_play.audit_only is False
    assert no_play.nullified is True
    touchdown = episodes[2]
    assert touchdown.play_ids == ("202", "203", "204", "205")
    assert touchdown.penalty_status == "offsetting"


def test_pick_six_is_a_score_turnover_episode() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_finalized_episodes,
        canonicalize_game_events,
    )

    row = _play(
        300,
        1,
        home_team="HOU",
        away_team="JAX",
        posteam="JAX",
        defteam="HOU",
        posteam_score=7,
        defteam_score=0,
        posteam_score_post=7,
        defteam_score_post=6,
        play_type="pass",
        pass_attempt=1,
        interception=1,
        touchdown=1,
        return_touchdown=1,
        td_team="HOU",
        td_player_name="Derek Stingley Jr.",
    )

    episode = build_finalized_episodes(canonicalize_game_events([row]))[0]

    assert episode.episode_type == "score_turnover"
    assert episode.turnover is True
    assert episode.scoring_team == "HOU"
    assert episode.score_points == 6


def test_review_admin_merges_previous_live_and_does_not_forward_fill_endpoint() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_finalized_episodes,
        canonicalize_game_events,
    )

    rows = [
        _play(
            400,
            1,
            play_type="pass",
            pass_attempt=1,
            complete_pass=1,
            passer_player_name="Trevor Lawrence",
            receiver_player_name="Brian Thomas Jr.",
            passing_yards=24,
            receiving_yards=24,
            post_state={
                "qtr": 1,
                "quarter_seconds_remaining": 860,
                "game_seconds_remaining": 3560,
                "home_team": "HOU",
                "away_team": "JAX",
                "posteam": "JAX",
                "defteam": "HOU",
                "total_home_score": 7,
                "total_away_score": 10,
                "yardline_100": 51,
            },
        ),
        _play(
            403,
            2,
            posteam=None,
            defteam=None,
            posteam_score_post=None,
            defteam_score_post=None,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            timeout=1,
            timeout_team="HOU",
        ),
        _play(
            401,
            3,
            posteam=None,
            defteam=None,
            posteam_score_post=None,
            defteam_score_post=None,
            play_type="no_play",
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
            desc="The previous play was reviewed and reversed.",
        ),
        _play(402, 4, play_type="run", rush_attempt=1),
    ]

    episodes = build_finalized_episodes(canonicalize_game_events(rows))

    assert len(episodes) == 2
    reviewed, following = episodes
    assert reviewed.play_ids == ("400", "403", "401")
    assert reviewed.episode_type == "pass"
    assert reviewed.review_result == "reversed"
    assert reviewed.nullified is True
    assert reviewed.pre_state.yardline_100 == 75
    assert reviewed.post_state is None
    assert following.play_ids == ("402",)


def test_cumulative_ledger_tracks_player_and_team_period_stats_and_ignores_no_play() -> (
    None
):
    from prediction_market.sports.nfl_x13_replay import (
        build_cumulative_stat_ledger,
        canonicalize_game_events,
    )

    rows = [
        _play(
            500,
            1,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=0,
            defteam_score=0,
            posteam_score_post=6,
            defteam_score_post=0,
            play_type="pass",
            pass_attempt=1,
            complete_pass=1,
            passing_yards=25,
            receiving_yards=25,
            passer_player_name="Trevor Lawrence",
            receiver_player_name="Brian Thomas Jr.",
            touchdown=1,
            pass_touchdown=1,
            td_team="JAX",
        ),
        _play(
            501,
            2,
            qtr=1,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=6,
            defteam_score=0,
            posteam_score_post=7,
            defteam_score_post=0,
            play_type="extra_point",
            extra_point_attempt=1,
        ),
        _play(
            502,
            3,
            qtr=2,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=7,
            defteam_score=0,
            posteam_score_post=7,
            defteam_score_post=0,
            play_type="run",
            rush_attempt=1,
            rushing_yards=8,
            rusher_player_name="Travis Etienne",
        ),
        _play(
            503,
            4,
            qtr=2,
            home_team="JAX",
            away_team="HOU",
            posteam="JAX",
            defteam="HOU",
            posteam_score=7,
            defteam_score=0,
            posteam_score_post=13,
            defteam_score_post=0,
            play_type="no_play",
            pass_attempt=1,
            complete_pass=1,
            passing_yards=40,
            receiving_yards=40,
            passer_player_name="Trevor Lawrence",
            receiver_player_name="Brian Thomas Jr.",
            touchdown=1,
            pass_touchdown=1,
            td_team="JAX",
            penalty=1,
            desc="Touchdown nullified by penalty.",
        ),
    ]

    ledger = build_cumulative_stat_ledger(canonicalize_game_events(rows))
    final = ledger[-1]

    assert len(ledger) == 4
    assert final.player_stats["Trevor Lawrence"] == {
        "passing_yards": 25,
        "passing_touchdowns": 1,
        "rushing_yards": 0,
        "rushing_touchdowns": 0,
        "receiving_yards": 0,
        "receptions": 0,
        "receiving_touchdowns": 0,
        "return_touchdowns": 0,
        "touchdowns": 0,
    }
    assert final.player_stats["Brian Thomas Jr."]["receiving_yards"] == 25
    assert final.player_stats["Brian Thomas Jr."]["receptions"] == 1
    assert final.player_stats["Brian Thomas Jr."]["receiving_touchdowns"] == 1
    assert final.player_stats["Brian Thomas Jr."]["touchdowns"] == 1
    assert final.player_stats["Travis Etienne"]["rushing_yards"] == 8
    assert final.team_scores == {"HOU": 0, "JAX": 7}
    assert final.period_scores == {
        1: {"HOU": 0, "JAX": 7},
        2: {"HOU": 0, "JAX": 0},
    }


def test_review_reversal_deterministically_corrects_the_previous_live_stats() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_cumulative_stat_ledger,
        canonicalize_game_events,
    )

    rows = [
        _play(
            600,
            1,
            play_type="pass",
            pass_attempt=1,
            complete_pass=1,
            passing_yards=24,
            receiving_yards=24,
            passer_player_name="Trevor Lawrence",
            receiver_player_name="Brian Thomas Jr.",
            posteam_score_post=16,
            defteam_score_post=7,
            touchdown=1,
            pass_touchdown=1,
            td_team="JAX",
        ),
        _play(
            601,
            2,
            posteam=None,
            defteam=None,
            posteam_score_post=None,
            defteam_score_post=None,
            play_type="no_play",
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
        ),
        _play(
            602,
            3,
            play_type="run",
            rush_attempt=1,
            rushing_yards=5,
            rusher_player_name="Travis Etienne",
        ),
    ]

    ledger = build_cumulative_stat_ledger(canonicalize_game_events(rows))

    assert ledger[0].player_stats["Trevor Lawrence"]["passing_yards"] == 24
    assert ledger[0].team_scores == {"HOU": 7, "JAX": 16}
    assert ledger[0].period_scores == {1: {"HOU": 0, "JAX": 6}}
    assert ledger[1].correction_of_play_id == "600"
    assert ledger[1].player_stats["Trevor Lawrence"]["passing_yards"] == 0
    assert ledger[1].player_stats["Trevor Lawrence"]["passing_touchdowns"] == 0
    assert ledger[1].player_stats["Brian Thomas Jr."]["receiving_yards"] == 0
    assert ledger[1].player_stats["Brian Thomas Jr."]["receptions"] == 0
    assert ledger[1].player_stats["Brian Thomas Jr."]["touchdowns"] == 0
    assert ledger[1].team_scores == {"HOU": 7, "JAX": 10}
    assert ledger[1].period_scores == {1: {"HOU": 0, "JAX": 0}}
    assert ledger[2].player_stats["Travis Etienne"]["rushing_yards"] == 5


def test_overtime_safety_and_terminal_tie_remain_source_observed() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_cumulative_stat_ledger,
        build_finalized_episodes,
        canonicalize_game_events,
    )

    rows = [
        _play(
            700,
            1,
            game_id="2025_13_ARI_SF",
            qtr=5,
            quarter_seconds_remaining=301,
            game_seconds_remaining=301,
            home_team="ARI",
            away_team="SF",
            posteam="ARI",
            defteam="SF",
            posteam_score=20,
            defteam_score=20,
            posteam_score_post=20,
            defteam_score_post=22,
            play_type="run",
            rush_attempt=1,
            safety=1,
            safety_player_name="SF Defender",
        ),
        _play(
            701,
            1,
            game_id="2025_04_GB_DAL",
            qtr=5,
            quarter_seconds_remaining=0,
            game_seconds_remaining=0,
            home_team="DAL",
            away_team="GB",
            posteam=None,
            defteam=None,
            posteam_score=None,
            defteam_score=None,
            posteam_score_post=None,
            defteam_score_post=None,
            total_home_score=40,
            total_away_score=40,
            play_type="no_play",
            game_end=1,
            post_state={
                "qtr": 5,
                "quarter_seconds_remaining": 0,
                "game_seconds_remaining": 0,
                "home_team": "DAL",
                "away_team": "GB",
                "total_home_score": 40,
                "total_away_score": 40,
                "terminal": 1,
            },
        ),
    ]

    events = canonicalize_game_events(rows)
    episodes = build_finalized_episodes(events)
    ledger = build_cumulative_stat_ledger(events)

    safety_event = next(event for event in events if event.game_id.endswith("ARI_SF"))
    assert safety_event.quarter == 5
    assert safety_event.safety is True
    safety_episode = next(
        episode for episode in episodes if episode.game_id.endswith("ARI_SF")
    )
    assert safety_episode.episode_type == "safety"
    assert safety_episode.scoring_team == "SF"
    assert safety_episode.score_points == 2
    safety_ledger = next(entry for entry in ledger if entry.game_id.endswith("ARI_SF"))
    assert safety_ledger.team_scores == {"ARI": 20, "SF": 22}
    assert safety_ledger.period_scores == {5: {"ARI": 0, "SF": 2}}

    tie_event = next(event for event in events if event.game_id.endswith("GB_DAL"))
    assert tie_event.quarter == 5
    assert tie_event.game_end is True
    assert tie_event.post_state is not None
    assert tie_event.post_state.terminal is True
    assert tie_event.post_state.home_score == tie_event.post_state.away_score == 40


def test_blocked_kick_retains_kick_family_without_a_score() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_finalized_episodes,
        canonicalize_game_events,
    )

    row = _play(
        800,
        1,
        play_type="field_goal",
        field_goal_attempt=1,
        field_goal_result="blocked",
        blocked_player_name="JAX Kicker",
        posteam_score=10,
        defteam_score=7,
        posteam_score_post=10,
        defteam_score_post=7,
    )

    event = canonicalize_game_events([row])[0]
    episode = build_finalized_episodes([event])[0]

    assert event.play_family == "field_goal"
    assert event.blocked_kick is True
    assert episode.episode_type == "field_goal"
    assert episode.blocked_kick is True
    assert episode.score_points == 0


def test_integration_accepts_existing_replay_and_participation_payloads() -> None:
    from prediction_market.sports.nfl_x13_replay import build_x13_game_ledger

    replay = SimpleNamespace(
        native_game_id="2025_10_JAX_HOU",
        home_team="HOU",
        away_team="JAX",
        steps=(
            {
                "source_play_id": "900",
                "source_order_sequence": 10,
                "source_time_utc": "2025-11-09T18:10:00Z",
                "source_time_status": "present",
                "next_state": {
                    "game_id": "game_nflverse_2025_10_JAX_HOU",
                    "period": 1,
                    "period_seconds_remaining": 870,
                    "game_seconds_remaining": 3570,
                    "home_team": "HOU",
                    "away_team": "JAX",
                    "home_score": 0,
                    "away_score": 0,
                    "possession_team": "JAX",
                    "drive_id": "1",
                    "down": 2,
                    "distance": 5,
                    "yardline_100": 70,
                    "goal_to_go": False,
                    "home_timeouts_remaining": 3,
                    "away_timeouts_remaining": 3,
                    "terminal": False,
                },
            },
        ),
    )
    participation = {
        "canonical_game_id": "game_nflverse_2025_10_JAX_HOU",
        "rows": [
            {
                "play_id": 900,
                "order_sequence": 10,
                "source_time_utc": "2025-11-09T18:10:00Z",
                "state": {
                    "qtr": 1,
                    "quarter_seconds_remaining": 875,
                    "game_seconds_remaining": 3575,
                    "home_team": "HOU",
                    "away_team": "JAX",
                    "posteam": "JAX",
                    "defteam": "HOU",
                    "fixed_drive": 1,
                    "down": 1,
                    "ydstogo": 10,
                    "yardline_100": 75,
                    "goal_to_go": 0,
                    "total_home_score": 0,
                    "total_away_score": 0,
                    "home_timeouts_remaining": 3,
                    "away_timeouts_remaining": 3,
                },
                "event": {
                    "play_type": "run",
                    "description": "Five-yard run.",
                    "flags": {"rush_attempt": True},
                    "actors": {"rusher_player_name": "Travis Etienne"},
                },
                "participation": {
                    "offense_names": [f"JAX {index}" for index in range(11)],
                    "defense_names": [f"HOU {index}" for index in range(11)],
                    "offense_formation": "SINGLEBACK",
                    "offense_personnel": "11",
                    "defense_personnel": "4-2-5",
                },
            }
        ],
    }

    first = build_x13_game_ledger(replay, participation)
    second = build_x13_game_ledger(replay, participation)

    assert len(first.events) == len(first.episodes) == len(first.stat_ledger) == 1
    event = first.events[0]
    assert event.game_id == "2025_10_JAX_HOU"
    assert event.play_family == "run"
    assert event.player_roles["rusher"] == "Travis Etienne"
    assert event.formation == "SINGLEBACK"
    assert event.pre_state.yardline_100 == 75
    assert event.post_state is not None
    assert event.post_state.yardline_100 == 70
    assert event.post_state.clock_seconds_remaining == 870
    assert event.post_state.offense_team == "JAX"
    assert event.post_state.defense_team == "HOU"
    assert first.events_sha256 == second.events_sha256
    assert first.artifact_sha256 == second.artifact_sha256


def test_integration_preserves_flat_nflverse_participation_rows() -> None:
    from prediction_market.sports.nfl_x13_replay import build_x13_game_ledger

    replay = [
        _play(
            905,
            1,
            game_id="2025_10_JAX_HOU",
            home_team="HOU",
            away_team="JAX",
            posteam="JAX",
            defteam="HOU",
            play_type="run",
            rush_attempt=1,
        )
    ]
    participation = [
        {
            "nflverse_game_id": "2025_10_JAX_HOU",
            "play_id": 905,
            "offense_names": [f"JAX {index}" for index in range(11)],
            "defense_names": [f"HOU {index}" for index in range(11)],
            "offense_formation": "SHOTGUN",
            "offense_personnel": "11",
            "defense_personnel": "4-2-5",
        }
    ]

    event = build_x13_game_ledger(replay, participation).events[0]

    assert event.participation_status == "present_complete"
    assert event.offense_names == tuple(f"JAX {index}" for index in range(11))
    assert event.defense_names == tuple(f"HOU {index}" for index in range(11))
    assert event.formation == "SHOTGUN"
    assert event.offense_personnel == "11"
    assert event.defense_personnel == "4-2-5"


def test_touchdown_bundle_stops_before_game_end_or_the_next_live_snap() -> None:
    from prediction_market.sports.nfl_x13_replay import (
        build_cumulative_stat_ledger,
        build_finalized_episodes,
        canonicalize_game_events,
    )

    rows = [
        _play(1000, 1, game_id="game-a", play_type="run", rush_attempt=1, touchdown=1),
        _play(
            1004,
            2,
            game_id="game-a",
            play_type="no_play",
            play_deleted=1,
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
        ),
        _play(
            1001,
            3,
            game_id="game-a",
            posteam=None,
            defteam=None,
            posteam_score_post=None,
            defteam_score_post=None,
            play_type="no_play",
            game_end=1,
        ),
        _play(1002, 1, game_id="game-b", play_type="pass", pass_attempt=1, touchdown=1),
        _play(1003, 2, game_id="game-b", play_type="run", rush_attempt=1),
    ]

    events = canonicalize_game_events(rows)
    episodes = build_finalized_episodes(events)
    ledger = build_cumulative_stat_ledger(events)

    assert [(episode.game_id, episode.play_ids) for episode in episodes] == [
        ("game-a", ("1000",)),
        ("game-a", ("1004",)),
        ("game-a", ("1001",)),
        ("game-b", ("1002",)),
        ("game-b", ("1003",)),
    ]
    deleted = episodes[1]
    assert deleted.audit_only is True
    assert deleted.nullified is False
    deleted_ledger = next(entry for entry in ledger if entry.through_play_id == "1004")
    assert deleted_ledger.correction_of_play_id is None


def test_dataframe_input_uses_game_order_raw_sort_and_preserves_nan_as_missing() -> (
    None
):
    import pandas as pd

    from prediction_market.sports.nfl_x13_replay import (
        canonical_events_sha256,
        canonicalize_game_events,
    )

    rows = [
        _play(
            1102,
            5,
            game_id="game-a",
            _raw_record_ordinal=2,
            time_of_day=math.nan,
            goal_to_go=math.nan,
            posteam_score_post=math.nan,
            defteam_score_post=math.nan,
        ),
        _play(
            1200,
            1,
            game_id="game-b",
            _raw_record_ordinal=0,
        ),
        _play(
            1101,
            5,
            game_id="game-a",
            _raw_record_ordinal=1,
        ),
    ]

    first = canonicalize_game_events(pd.DataFrame(rows))
    second = canonicalize_game_events(pd.DataFrame(list(reversed(rows))))

    assert [
        (event.game_id, event.order_sequence, event.raw_ordinal) for event in first
    ] == [
        ("game-a", 5, 1),
        ("game-a", 5, 2),
        ("game-b", 1, 0),
    ]
    assert first[1].source_time_status == "missing"
    assert first[1].goal_to_go is None
    assert first[1].post_state is None
    assert canonical_events_sha256(first) == canonical_events_sha256(second)


def test_canonical_source_and_participation_identities_must_be_unique() -> None:
    import pytest

    from prediction_market.sports.nfl_x13_replay import canonicalize_game_events

    first = _play(1300, 1, _raw_record_ordinal=7)
    duplicate_order = _play(1301, 1, _raw_record_ordinal=7)
    with pytest.raises(ValueError, match="raw order key must be unique"):
        canonicalize_game_events([first, duplicate_order])

    participation = {
        "nflverse_game_id": "2025_10_JAX_HOU",
        "play_id": 1300,
        "offense_names": ["JAX"] * 11,
        "defense_names": ["HOU"] * 11,
    }
    with pytest.raises(ValueError, match="participation key must be unique"):
        canonicalize_game_events([first], [participation, participation])


def test_field_coordinate_is_bounded_and_missing_orientation_is_explicit() -> None:
    import pytest

    from prediction_market.sports.nfl_x13_replay import canonicalize_game_events

    missing = _play(
        1400,
        1,
        posteam=None,
        defteam=None,
        yardline_100=None,
        field_coordinate_0_100=None,
    )
    event = canonicalize_game_events([missing])[0]
    assert event.field_coordinate_0_100 is None
    assert (
        event.field_orientation_semantics
        == "unavailable_without_possession_and_yardline"
    )

    invalid = _play(1401, 2, field_coordinate_0_100=101)
    with pytest.raises(ValueError, match="between 0 and 100"):
        canonicalize_game_events([invalid])
