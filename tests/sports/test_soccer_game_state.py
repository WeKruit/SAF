from __future__ import annotations

import re
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_bundle,
)
from prediction_market.sports.game_state import advance_state
from prediction_market.sports.soccer_game_state import (
    SOCCER_GAME_STATE_REDUCER,
    SoccerCard,
    SoccerGameEvent,
    SoccerGameStateReducer,
    SoccerGameState,
    SoccerGameStateError,
    SoccerSubstitution,
    SoccerTeamPlayers,
    adapt_statsbomb_event as _adapt_statsbomb_event,
    initial_soccer_game_state,
    reduce_soccer_game_state,
    statsbomb_event_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAME_ID = "game_statsbomb_100000"
HOME_TEAM_ID = 10
AWAY_TEAM_ID = 20


def adapt_statsbomb_event(
    raw_event: dict[str, Any],
    *,
    game_id: str,
    quality_flags: tuple[str, ...] = (),
) -> SoccerGameEvent:
    """Build the validated synthetic envelope used by unit tests."""

    sequence = raw_event["index"]
    assert type(sequence) is int
    native_id = raw_event["id"]
    assert type(native_id) is str
    payload = statsbomb_event_payload(
        raw_event,
        game_id=game_id,
        quality_flags=quality_flags,
    )
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-12",
        dataset_id="DS-STATSBOMB-OPEN",
        source_system="statsbomb",
        source_stream="events",
        raw_object_hash="sha256:" + "8" * 64,
        raw_record_ordinals=(sequence - 1,),
        partition="synthetic-fixture",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=None,
        competition_id="cmp_statsbomb_2",
        game_id=game_id,
        participant_ids=(
            f"participant_statsbomb_{HOME_TEAM_ID}",
            f"participant_statsbomb_{AWAY_TEAM_ID}",
        ),
        native_namespace="statsbomb.event",
        native_ids=(native_id,),
        normalized_source_sequence=sequence,
        normalized_payload=payload,
        quality_flags=quality_flags,
    )
    return _adapt_statsbomb_event(
        bundle.normalized,
        program_root=PROJECT_ROOT,
        raw_parents=bundle.raw,
    )


def _starting_xi_event(
    *,
    index: int,
    native_id: str,
    team_id: int,
) -> dict[str, Any]:
    return {
        "id": native_id,
        "index": index,
        "period": 1,
        "timestamp": "00:00:00.000",
        "minute": 0,
        "second": 0,
        "type": {"id": 35, "name": "Starting XI"},
        "team": {"id": team_id, "name": f"Team {team_id}"},
        "tactics": {
            "formation": 433,
            "lineup": [
                {
                    "player": {
                        "id": team_id * 100 + player,
                        "name": f"Player {team_id}-{player}",
                    },
                    "position": {"id": player, "name": f"Position {player}"},
                    "jersey_number": player,
                }
                for player in range(1, 12)
            ],
        },
    }


def _contains_binary_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_binary_float(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_binary_float(child) for child in value)
    return False


def _event(
    *,
    index: int,
    native_id: str,
    action: str,
    team_id: int,
    minute: int,
    second: int,
    timestamp: str,
    period: int = 1,
    player_id: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "id": native_id,
        "index": index,
        "period": period,
        "timestamp": timestamp,
        "minute": minute,
        "second": second,
        "type": {"id": 1, "name": action},
        "team": {"id": team_id, "name": f"Team {team_id}"},
    }
    if player_id is not None:
        event["player"] = {"id": player_id, "name": f"Player {player_id}"}
    event.update(extra)
    return event


def test_adapter_builds_immutable_canonical_event_without_future_data() -> None:
    raw = _starting_xi_event(
        index=1,
        native_id="statsbomb-native-1",
        team_id=HOME_TEAM_ID,
    )
    event = adapt_statsbomb_event(raw, game_id=GAME_ID)

    assert isinstance(event, SoccerGameEvent)
    assert event.sport == "soccer"
    assert event.game_id == GAME_ID
    assert event.sequence == 1
    assert event.period == 1
    assert event.clock_ms == 0
    assert event.period_clock_ms == 0
    assert event.action == "Starting XI"
    assert event.lineup_player_ids == tuple(
        HOME_TEAM_ID * 100 + player for player in range(1, 12)
    )
    assert re.fullmatch(r"evt_[0-9a-f]{64}", event.event_id)
    assert not _contains_binary_float(asdict(event))

    with pytest.raises(FrozenInstanceError):
        event.sequence = 2  # type: ignore[misc]

    raw_with_future = {
        **raw,
        "future_events": [
            {
                "type": {"name": "Shot"},
                "shot": {"outcome": {"name": "Goal"}},
            }
        ],
    }
    assert adapt_statsbomb_event(raw_with_future, game_id=GAME_ID) == event
    raw_with_misplaced_future_card = {
        **raw,
        "bad_behaviour": {
            "card": {"id": 5, "name": "Red Card"},
        },
    }
    assert (
        adapt_statsbomb_event(
            raw_with_misplaced_future_card,
            game_id=GAME_ID,
        )
        == event
    )


def test_adapter_requires_a_fully_bound_event_envelope() -> None:
    raw = _starting_xi_event(
        index=1,
        native_id="statsbomb-native-1",
        team_id=HOME_TEAM_ID,
    )
    event = adapt_statsbomb_event(raw, game_id=GAME_ID)
    assert re.fullmatch(r"evt_[0-9a-f]{64}", event.event_id)

    with pytest.raises(TypeError, match="EventEnvelopeV0"):
        _adapt_statsbomb_event(  # type: ignore[arg-type]
            raw,
            program_root=PROJECT_ROOT,
            raw_parents=(),
        )


def test_adapter_normalizes_pass_clock_possession_and_coordinates_to_integers() -> None:
    raw = _event(
        index=3,
        native_id="statsbomb-native-3",
        action="Pass",
        team_id=HOME_TEAM_ID,
        player_id=1001,
        minute=1,
        second=1,
        timestamp="00:01:01.250",
        possession=2,
        possession_team={"id": HOME_TEAM_ID, "name": "Home"},
        play_pattern={"id": 1, "name": "Regular Play"},
        location=[35.125, 40.0],
        **{
            "pass": {
                "recipient": {"id": 1002, "name": "Receiver"},
                "end_location": [60.5, 41],
            }
        },
    )

    event = adapt_statsbomb_event(raw, game_id=GAME_ID)

    assert event.clock_ms == 61_250
    assert event.period_clock_ms == 61_250
    assert event.player_id == 1001
    assert event.possession_id == 2
    assert event.possession_team_id == HOME_TEAM_ID
    assert event.play_pattern == "Regular Play"
    assert (event.ball_x_milli, event.ball_y_milli) == (35_125, 40_000)
    assert (event.end_ball_x_milli, event.end_ball_y_milli) == (60_500, 41_000)
    assert event.pass_outcome is None
    assert event.in_play is True
    assert not _contains_binary_float(asdict(event))


def test_adapter_extracts_goal_card_and_substitution_without_lookahead() -> None:
    goal = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Shot",
            team_id=HOME_TEAM_ID,
            player_id=1009,
            minute=12,
            second=5,
            timestamp="00:12:05.500",
            possession=8,
            possession_team={"id": HOME_TEAM_ID, "name": "Home"},
            play_pattern={"id": 1, "name": "Regular Play"},
            location=[108, 38.5],
            shot={
                "outcome": {"id": 97, "name": "Goal"},
                "end_location": [120, 40, 1.2],
            },
        ),
        game_id=GAME_ID,
    )
    red_card = adapt_statsbomb_event(
        _event(
            index=5,
            native_id="statsbomb-native-5",
            action="Bad Behaviour",
            team_id=AWAY_TEAM_ID,
            player_id=2004,
            minute=13,
            second=0,
            timestamp="00:13:00.000",
            bad_behaviour={"card": {"id": 5, "name": "Red Card"}},
        ),
        game_id=GAME_ID,
    )
    substitution = adapt_statsbomb_event(
        _event(
            index=6,
            native_id="statsbomb-native-6",
            action="Substitution",
            team_id=HOME_TEAM_ID,
            player_id=1001,
            minute=20,
            second=0,
            timestamp="00:20:00.000",
            substitution={
                "replacement": {"id": 1012, "name": "Replacement"},
                "outcome": {"id": 103, "name": "Tactical"},
            },
        ),
        game_id=GAME_ID,
    )

    assert goal.shot_outcome == "Goal"
    assert goal.score_for_team_id == HOME_TEAM_ID
    assert goal.in_play is False
    assert (goal.end_ball_x_milli, goal.end_ball_y_milli) == (120_000, 40_000)
    assert red_card.card == "Red Card"
    assert red_card.in_play is False
    assert substitution.replacement_player_id == 1012
    assert substitution.in_play is False


def _starting_states() -> tuple[SoccerGameState, SoccerGameState, SoccerGameState]:
    initial = initial_soccer_game_state(
        GAME_ID,
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
    )
    home_started = reduce_soccer_game_state(
        initial,
        adapt_statsbomb_event(
            _starting_xi_event(
                index=1,
                native_id="statsbomb-native-1",
                team_id=HOME_TEAM_ID,
            ),
            game_id=GAME_ID,
        ),
    )
    both_started = reduce_soccer_game_state(
        home_started,
        adapt_statsbomb_event(
            _starting_xi_event(
                index=2,
                native_id="statsbomb-native-2",
                team_id=AWAY_TEAM_ID,
            ),
            game_id=GAME_ID,
        ),
    )
    return initial, home_started, both_started


def test_reducer_initializes_team_lineups_without_mutating_prior_states() -> None:
    initial, home_started, both_started = _starting_states()

    assert initial.sport == "soccer"
    assert initial.game_id == GAME_ID
    assert initial.sequence == 0
    assert initial.period == 0
    assert initial.clock_ms == 0
    assert initial.home_score == initial.away_score == 0
    assert initial.terminal is False
    assert initial.active_players == (
        SoccerTeamPlayers(HOME_TEAM_ID, ()),
        SoccerTeamPlayers(AWAY_TEAM_ID, ()),
    )
    assert home_started.sequence == 1
    assert home_started.active_players[0].player_ids == tuple(
        HOME_TEAM_ID * 100 + player for player in range(1, 12)
    )
    assert home_started.active_players[1].player_ids == ()
    assert both_started.sequence == 2
    assert both_started.active_players[1].player_ids == tuple(
        AWAY_TEAM_ID * 100 + player for player in range(1, 12)
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", both_started.state_sha256)

    with pytest.raises(FrozenInstanceError):
        both_started.home_score = 1  # type: ignore[misc]


def test_reducer_updates_possession_ball_last_action_and_goal_score() -> None:
    _, _, state = _starting_states()
    pass_event = adapt_statsbomb_event(
        _event(
            index=3,
            native_id="statsbomb-native-3",
            action="Pass",
            team_id=HOME_TEAM_ID,
            player_id=1001,
            minute=1,
            second=1,
            timestamp="00:01:01.250",
            possession=2,
            possession_team={"id": HOME_TEAM_ID, "name": "Home"},
            play_pattern={"id": 1, "name": "Regular Play"},
            location=[35.125, 40.0],
            **{"pass": {"end_location": [60.5, 41]}},
        ),
        game_id=GAME_ID,
    )
    after_pass = reduce_soccer_game_state(state, pass_event)
    goal_event = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Shot",
            team_id=HOME_TEAM_ID,
            player_id=1009,
            minute=12,
            second=5,
            timestamp="00:12:05.500",
            possession=8,
            possession_team={"id": HOME_TEAM_ID, "name": "Home"},
            play_pattern={"id": 1, "name": "Regular Play"},
            location=[108, 38.5],
            shot={
                "outcome": {"id": 97, "name": "Goal"},
                "end_location": [120, 40, 1.2],
            },
        ),
        game_id=GAME_ID,
    )
    after_goal = reduce_soccer_game_state(after_pass, goal_event)

    assert after_pass.sequence == 3
    assert after_pass.possession_id == 2
    assert after_pass.possession_team_id == HOME_TEAM_ID
    assert after_pass.play_pattern == "Regular Play"
    assert (after_pass.ball_x_milli, after_pass.ball_y_milli) == (60_500, 41_000)
    assert after_pass.in_play is True
    assert after_pass.last_action == "Pass"
    assert after_pass.last_event_id == pass_event.event_id
    assert after_goal.sequence == 4
    assert (after_goal.home_score, after_goal.away_score) == (1, 0)
    assert (after_goal.ball_x_milli, after_goal.ball_y_milli) == (120_000, 40_000)
    assert after_goal.in_play is False
    assert after_goal.last_action == "Shot"


def test_reducer_records_cards_removes_red_and_applies_substitution() -> None:
    _, _, state = _starting_states()
    yellow_event = adapt_statsbomb_event(
        _event(
            index=3,
            native_id="statsbomb-native-3",
            action="Foul Committed",
            team_id=HOME_TEAM_ID,
            player_id=1003,
            minute=10,
            second=0,
            timestamp="00:10:00.000",
            foul_committed={"card": {"id": 7, "name": "Yellow Card"}},
        ),
        game_id=GAME_ID,
    )
    red_event = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Bad Behaviour",
            team_id=AWAY_TEAM_ID,
            player_id=2004,
            minute=11,
            second=0,
            timestamp="00:11:00.000",
            bad_behaviour={"card": {"id": 5, "name": "Red Card"}},
        ),
        game_id=GAME_ID,
    )
    substitution_event = adapt_statsbomb_event(
        _event(
            index=5,
            native_id="statsbomb-native-5",
            action="Substitution",
            team_id=HOME_TEAM_ID,
            player_id=1001,
            minute=20,
            second=0,
            timestamp="00:20:00.000",
            substitution={
                "replacement": {"id": 1012, "name": "Replacement"},
                "outcome": {"id": 103, "name": "Tactical"},
            },
        ),
        game_id=GAME_ID,
    )

    after_yellow = reduce_soccer_game_state(state, yellow_event)
    after_red = reduce_soccer_game_state(after_yellow, red_event)
    after_substitution = reduce_soccer_game_state(after_red, substitution_event)

    assert after_yellow.cards == (
        SoccerCard(
            sequence=3,
            team_id=HOME_TEAM_ID,
            player_id=1003,
            card="Yellow Card",
        ),
    )
    assert 1003 in after_yellow.active_players[0].player_ids
    assert after_red.cards[-1].card == "Red Card"
    assert 2004 not in after_red.active_players[1].player_ids
    assert after_substitution.substitutions == (
        SoccerSubstitution(
            sequence=5,
            team_id=HOME_TEAM_ID,
            player_out_id=1001,
            player_in_id=1012,
        ),
    )
    assert 1001 not in after_substitution.active_players[0].player_ids
    assert 1012 in after_substitution.active_players[0].player_ids


def test_period_change_and_explicit_match_end_are_fail_closed() -> None:
    _, _, state = _starting_states()
    events = [
        _event(
            index=3,
            native_id="statsbomb-native-3",
            action="Pass",
            team_id=HOME_TEAM_ID,
            minute=44,
            second=59,
            timestamp="00:44:59.000",
            **{"pass": {"end_location": [80, 40]}},
        ),
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Half End",
            team_id=HOME_TEAM_ID,
            minute=45,
            second=0,
            timestamp="00:45:00.000",
        ),
        _event(
            index=5,
            native_id="statsbomb-native-5",
            action="Half Start",
            team_id=AWAY_TEAM_ID,
            period=2,
            minute=45,
            second=0,
            timestamp="00:00:00.000",
        ),
        _event(
            index=6,
            native_id="statsbomb-native-6",
            action="Pass",
            team_id=AWAY_TEAM_ID,
            period=2,
            minute=45,
            second=1,
            timestamp="00:00:01.000",
            **{"pass": {"end_location": [40, 20]}},
        ),
        _event(
            index=7,
            native_id="statsbomb-native-7",
            action="Match End",
            team_id=HOME_TEAM_ID,
            period=2,
            minute=90,
            second=0,
            timestamp="00:45:00.000",
        ),
    ]
    for raw in events:
        state = reduce_soccer_game_state(
            state,
            adapt_statsbomb_event(raw, game_id=GAME_ID),
        )

    assert state.sequence == 7
    assert state.period == 2
    assert state.clock_ms == 5_400_000
    assert state.in_play is False
    assert state.terminal is True
    assert state.terminal_reason == "match_end"
    after_end = adapt_statsbomb_event(
        _event(
            index=8,
            native_id="statsbomb-native-8",
            action="Pass",
            team_id=HOME_TEAM_ID,
            period=2,
            minute=90,
            second=1,
            timestamp="00:45:01.000",
            **{"pass": {"end_location": [1, 1]}},
        ),
        game_id=GAME_ID,
    )
    with pytest.raises(SoccerGameStateError, match="terminal"):
        reduce_soccer_game_state(state, after_end)


def test_reducer_rejects_game_sequence_period_and_time_invariant_violations() -> None:
    _, _, state = _starting_states()
    valid_pass_raw = _event(
        index=3,
        native_id="statsbomb-native-3",
        action="Pass",
        team_id=HOME_TEAM_ID,
        minute=10,
        second=0,
        timestamp="00:10:00.000",
        **{"pass": {"end_location": [50, 40]}},
    )
    valid_pass = adapt_statsbomb_event(valid_pass_raw, game_id=GAME_ID)

    with pytest.raises(SoccerGameStateError, match="game_id"):
        reduce_soccer_game_state(
            state,
            replace(
                valid_pass,
                game_id="game_statsbomb_other",
            ),
        )

    skipped = adapt_statsbomb_event(
        {**valid_pass_raw, "index": 4, "id": "statsbomb-native-4"},
        game_id=GAME_ID,
    )
    with pytest.raises(SoccerGameStateError, match="sequence"):
        reduce_soccer_game_state(state, skipped)

    state = reduce_soccer_game_state(state, valid_pass)
    regressed_clock = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Pass",
            team_id=HOME_TEAM_ID,
            minute=9,
            second=59,
            timestamp="00:09:59.000",
            **{"pass": {"end_location": [50, 40]}},
        ),
        game_id=GAME_ID,
    )
    with pytest.raises(SoccerGameStateError, match="clock regression"):
        reduce_soccer_game_state(state, regressed_clock)

    authorized_regression = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Pass",
            team_id=HOME_TEAM_ID,
            minute=9,
            second=59,
            timestamp="00:09:59.000",
            **{"pass": {"end_location": [50, 40]}},
        ),
        game_id=GAME_ID,
        quality_flags=("clock_jump", "out_of_order"),
    )
    after_regressed_clock = reduce_soccer_game_state(
        state,
        authorized_regression,
    )
    assert regressed_clock.clock_ms == 599_000
    assert regressed_clock.period_clock_ms == 599_000
    assert after_regressed_clock.clock_ms == 599_000
    assert after_regressed_clock.period_clock_ms == 599_000
    assert after_regressed_clock.quality_flags == (
        "clock_jump",
        "out_of_order",
    )

    non_starting_next_period = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Pass",
            team_id=HOME_TEAM_ID,
            period=2,
            minute=45,
            second=0,
            timestamp="00:00:00.000",
            **{"pass": {"end_location": [50, 40]}},
        ),
        game_id=GAME_ID,
    )
    with pytest.raises(SoccerGameStateError, match="period start"):
        reduce_soccer_game_state(state, non_starting_next_period)

    malformed_period_start = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Half Start",
            team_id=HOME_TEAM_ID,
            period=2,
            minute=46,
            second=0,
            timestamp="00:01:00.000",
        ),
        game_id=GAME_ID,
    )
    with pytest.raises(SoccerGameStateError, match="period clock"):
        reduce_soccer_game_state(state, malformed_period_start)


def test_new_period_allows_statsbomb_nominal_minute_reset_after_stoppage() -> None:
    _, _, state = _starting_states()
    half_end = adapt_statsbomb_event(
        _event(
            index=3,
            native_id="statsbomb-native-3",
            action="Half End",
            team_id=HOME_TEAM_ID,
            minute=46,
            second=10,
            timestamp="00:46:10.114",
        ),
        game_id=GAME_ID,
    )
    half_start = adapt_statsbomb_event(
        _event(
            index=4,
            native_id="statsbomb-native-4",
            action="Half Start",
            team_id=AWAY_TEAM_ID,
            period=2,
            minute=45,
            second=0,
            timestamp="00:00:00.000",
        ),
        game_id=GAME_ID,
    )

    after_end = reduce_soccer_game_state(state, half_end)
    after_start = reduce_soccer_game_state(after_end, half_start)

    assert after_end.clock_ms == 2_770_114
    assert after_start.period == 2
    assert after_start.clock_ms == 2_700_000
    assert after_start.period_clock_ms == 0


def test_adapter_is_bounded_and_rejects_invalid_statsbomb_atoms() -> None:
    oversized = _event(
        index=1,
        native_id="statsbomb-native-1",
        action="Half Start",
        team_id=HOME_TEAM_ID,
        minute=0,
        second=0,
        timestamp="00:00:00.000",
    )
    oversized.update({f"unknown_{index}": index for index in range(64)})
    with pytest.raises(SoccerGameStateError, match="bounded"):
        adapt_statsbomb_event(oversized, game_id=GAME_ID)

    invalid_coordinate = _event(
        index=1,
        native_id="statsbomb-native-1",
        action="Pass",
        team_id=HOME_TEAM_ID,
        minute=0,
        second=0,
        timestamp="00:00:00.000",
        location=[120.001, 40],
        **{"pass": {"end_location": [121, 40]}},
    )
    with pytest.raises(SoccerGameStateError, match=r"<= 120000"):
        adapt_statsbomb_event(invalid_coordinate, game_id=GAME_ID)

    inconsistent_clock = _event(
        index=1,
        native_id="statsbomb-native-1",
        action="Half Start",
        team_id=HOME_TEAM_ID,
        minute=0,
        second=1,
        timestamp="00:00:02.000",
    )
    with pytest.raises(SoccerGameStateError, match="must match"):
        adapt_statsbomb_event(inconsistent_clock, game_id=GAME_ID)


def test_replay_and_canonical_hashes_are_deterministic() -> None:
    raw_pass = _event(
        index=3,
        native_id="statsbomb-native-3",
        action="Pass",
        team_id=HOME_TEAM_ID,
        player_id=1001,
        minute=1,
        second=1,
        timestamp="00:01:01.250",
        possession=2,
        possession_team={"id": HOME_TEAM_ID, "name": "Home"},
        play_pattern={"id": 1, "name": "Regular Play"},
        location=[35.125, 40.0],
        **{"pass": {"end_location": [60.5, 41]}},
    )
    event = adapt_statsbomb_event(raw_pass, game_id=GAME_ID)

    def replay() -> SoccerGameState:
        _, _, started = _starting_states()
        return reduce_soccer_game_state(
            started,
            adapt_statsbomb_event(
                dict(reversed(tuple(raw_pass.items()))),
                game_id=GAME_ID,
            ),
        )

    first = replay()
    second = replay()

    assert re.fullmatch(r"evt_[0-9a-f]{64}", event.event_id)
    assert first == second
    assert first.state_sha256 == second.state_sha256
    assert hash(first) == hash(second)
    assert not _contains_binary_float(asdict(first))


def test_reducer_object_integrates_with_common_hash_chain() -> None:
    initial = initial_soccer_game_state(
        GAME_ID,
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
    )
    event = adapt_statsbomb_event(
        _starting_xi_event(
            index=1,
            native_id="statsbomb-native-1",
            team_id=HOME_TEAM_ID,
        ),
        game_id=GAME_ID,
    )

    assert isinstance(SOCCER_GAME_STATE_REDUCER, SoccerGameStateReducer)
    trace = advance_state(SOCCER_GAME_STATE_REDUCER, initial, event)

    assert trace.sport == "soccer"
    assert trace.game_id == GAME_ID
    assert trace.sequence == 1
    assert trace.event_id == event.event_id
    assert trace.next_state.sequence == 1
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", trace.trace_sha256)
