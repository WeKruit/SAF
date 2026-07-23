from __future__ import annotations

import re
from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_bundle,
)


EVENT_ID = "evt_" + "a" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    return import_module("prediction_market.sports.nfl_game_state")


def _state(**changes: object) -> Any:
    nfl = _module()
    values: dict[str, object] = {
        "sport": "nfl",
        "game_id": "game_nflverse_2025_01_AWY_HME",
        "sequence": 0,
        "terminal": False,
        "home_team": "HME",
        "away_team": "AWY",
        "period": 1,
        "period_seconds_remaining": 600,
        "game_seconds_remaining": 3300,
        "source_play_id": "101",
        "drive_id": "1",
        "play_clock_seconds": 12,
        "possession_team": "HME",
        "down": 2,
        "distance": 4,
        "yardline_100": 45,
        "goal_to_go": False,
        "home_score": 0,
        "away_score": 0,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "last_event_id": None,
    }
    values.update(changes)
    return nfl.NFLGameState(**values)


def _event(state: Any, **changes: object) -> Any:
    nfl = _module()
    values: dict[str, object] = {
        "sport": "nfl",
        "game_id": state.game_id,
        "sequence": state.sequence + 1,
        "event_id": EVENT_ID,
        "source_play_id": "101",
        "observation_mode": "offline",
        "play_type": "run",
        "description": "HME runner gains ten yards.",
        "period": state.period,
        "period_seconds_remaining": state.period_seconds_remaining - 10,
        "game_seconds_remaining": state.game_seconds_remaining - 10,
        "next_source_play_id": "122",
        "next_drive_id": state.drive_id,
        "next_play_clock_seconds": 10,
        "possession_team": state.possession_team,
        "down": 1,
        "distance": 10,
        "yardline_100": 35,
        "goal_to_go": False,
        "home_score": state.home_score,
        "away_score": state.away_score,
        "home_timeouts_remaining": state.home_timeouts_remaining,
        "away_timeouts_remaining": state.away_timeouts_remaining,
        "first_down": True,
        "turnover": False,
        "possession_changed": False,
        "score": False,
        "timeout": False,
        "timeout_team": None,
        "period_changed": False,
        "terminal": False,
        "quality_flags": (),
    }
    values.update(changes)
    return nfl.NFLPlayEvent(**values)


def _nflverse_rows() -> tuple[dict[str, object], dict[str, object]]:
    pre = {
        "game_id": "2025_01_AWY_HME",
        "play_id": 101.0,
        "home_team": "HME",
        "away_team": "AWY",
        "qtr": 1.0,
        "quarter_seconds_remaining": 600.0,
        "game_seconds_remaining": 3300.0,
        "fixed_drive": 1.0,
        "goal_to_go": 0.0,
        "play_clock": "12",
        "posteam": "HME",
        "down": 2.0,
        "ydstogo": 4.0,
        "yardline_100": 45.0,
        "total_home_score": 0.0,
        "total_away_score": 0.0,
        "home_timeouts_remaining": 3.0,
        "away_timeouts_remaining": 3.0,
        "play_type": "run",
        "desc": "HME runner gains ten yards.",
        "first_down": 1.0,
        "interception": 0.0,
        "fumble_lost": 0.0,
        "timeout": 0.0,
        "timeout_team": None,
        # Present in a native nflverse row, but forbidden from the observation.
        "epa": 2.75,
        "wpa": 0.04,
        "home_wp": 0.64,
        "fixed_drive_result": "Touchdown",
        "home_score": 31,
        "away_score": 20,
    }
    post = {
        **pre,
        "play_id": 122.0,
        "quarter_seconds_remaining": 590.0,
        "game_seconds_remaining": 3290.0,
        "play_clock": "10",
        "down": 1.0,
        "ydstogo": 10.0,
        "yardline_100": 35.0,
        "first_down": 0.0,
        "epa": -999.0,
        "wpa": -999.0,
        "home_wp": -999.0,
        "fixed_drive_result": "Punt",
    }
    return pre, post


def _adapt_nflverse_observations(
    pre: dict[str, object],
    post: dict[str, object],
    *,
    state_sequence: int = 0,
    terminal: bool | None = None,
) -> tuple[Any, Any]:
    nfl = _module()
    payload = nfl.nflverse_transition_payload(
        pre,
        post,
        sequence=state_sequence + 1,
        terminal=terminal,
    )
    native_game_id = str(pre["game_id"])
    source_play_id = str(int(float(pre["play_id"])))
    next_source_play_id = str(int(float(post["play_id"])))
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "7" * 64,
        raw_record_ordinals=(state_sequence, state_sequence + 1),
        partition="season-2025",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=None,
        competition_id="cmp_nfl",
        game_id=f"game_nflverse_{native_game_id}",
        participant_ids=(
            f"participant_{pre['away_team']}",
            f"participant_{pre['home_team']}",
        ),
        native_namespace="nflverse.play",
        native_ids=(
            f"{native_game_id}:{source_play_id}",
            f"{native_game_id}:{next_source_play_id}",
        ),
        normalized_source_sequence=state_sequence + 1,
        normalized_payload=payload,
    )
    state = nfl.state_from_nflverse_row(pre, sequence=state_sequence)
    event = nfl.event_from_nflverse_envelope(
        bundle.normalized,
        program_root=PROJECT_ROOT,
        raw_parents=bundle.raw,
    )
    return state, event


def test_first_down_reduces_to_a_new_immutable_state() -> None:
    nfl = _module()
    state = _state()
    event = _event(state)

    next_state = nfl.reduce(state, event)

    assert next_state != state
    assert state.sequence == 0
    assert state.down == 2
    assert next_state.sequence == 1
    assert next_state.down == 1
    assert next_state.source_play_id == "122"
    assert next_state.drive_id == "1"
    assert next_state.play_clock_seconds == 10
    assert next_state.distance == 10
    assert next_state.yardline_100 == 35
    assert next_state.last_event_id == event.event_id
    with pytest.raises(FrozenInstanceError):
        next_state.down = 2


def test_score_and_possession_switch_are_explicit_play_level_transitions() -> None:
    nfl = _module()
    state = _state()

    scored = nfl.reduce(
        state,
        _event(
            state,
            home_score=6,
            down=None,
            distance=None,
            score=True,
            first_down=False,
            play_type="pass",
        ),
    )
    assert (scored.home_score, scored.away_score) == (6, 0)

    switched = nfl.reduce(
        state,
        _event(
            state,
            possession_team="AWY",
            down=1,
            distance=10,
            yardline_100=65,
            possession_changed=True,
            turnover=True,
            first_down=False,
            play_type="pass",
        ),
    )
    assert switched.possession_team == "AWY"
    assert switched.down == 1


def test_quarter_timeout_and_terminal_transitions_are_representable() -> None:
    nfl = _module()
    quarter_end = _state(
        period_seconds_remaining=2,
        game_seconds_remaining=2702,
    )
    next_quarter = nfl.reduce(
        quarter_end,
        _event(
            quarter_end,
            period=2,
            period_seconds_remaining=900,
            game_seconds_remaining=2700,
            down=3,
            distance=2,
            yardline_100=30,
            first_down=False,
            period_changed=True,
        ),
    )
    assert next_quarter.period == 2
    assert next_quarter.period_seconds_remaining == 900

    timeout_state = _state()
    after_timeout = nfl.reduce(
        timeout_state,
        _event(
            timeout_state,
            period_seconds_remaining=600,
            game_seconds_remaining=3300,
            down=2,
            distance=4,
            yardline_100=45,
            first_down=False,
            play_type="no_play",
            home_timeouts_remaining=2,
            timeout=True,
            timeout_team="HME",
        ),
    )
    assert after_timeout.home_timeouts_remaining == 2

    final_state = _state(
        period=4,
        period_seconds_remaining=5,
        game_seconds_remaining=5,
        down=4,
        distance=2,
        home_score=14,
        away_score=10,
    )
    terminal = nfl.reduce(
        final_state,
        _event(
            final_state,
            period_seconds_remaining=0,
            game_seconds_remaining=0,
            down=4,
            distance=2,
            yardline_100=45,
            first_down=False,
            terminal=True,
        ),
    )
    assert terminal.terminal is True
    with pytest.raises(nfl.NFLGameStateError, match="terminal"):
        nfl.reduce(
            terminal,
            _event(
                terminal,
                period_seconds_remaining=0,
                game_seconds_remaining=0,
                down=4,
                distance=2,
                yardline_100=45,
                first_down=False,
                terminal=True,
            ),
        )


def test_reduce_fails_closed_on_game_order_clock_score_and_flag_mismatch() -> None:
    nfl = _module()
    state = _state()

    with pytest.raises(nfl.NFLGameStateError, match="game_id"):
        nfl.reduce(
            state,
            _event(state, game_id="game_nflverse_2025_01_OTHER_GAME"),
        )
    with pytest.raises(nfl.NFLGameStateError, match="sequence"):
        nfl.reduce(state, _event(state, sequence=2))
    with pytest.raises(nfl.NFLGameStateError, match="clock"):
        nfl.reduce(
            state,
            _event(
                state,
                period_seconds_remaining=601,
                game_seconds_remaining=3301,
            ),
        )
    with pytest.raises(nfl.NFLGameStateError, match="score"):
        nfl.reduce(
            state,
            _event(
                state,
                home_score=1,
                away_score=2,
                score=True,
            ),
        )
    with pytest.raises(nfl.NFLGameStateError, match="possession_changed"):
        nfl.reduce(state, _event(state, possession_changed=True))


@pytest.mark.parametrize(
    "changes",
    [
        {"down": 5},
        {"distance": 0},
        {"yardline_100": 101},
        {"possession_team": None, "down": 1, "distance": 10},
        {"yardline_100": 5, "distance": 10},
        {"home_score": -1},
        {"home_timeouts_remaining": 4},
    ],
)
def test_state_rejects_illegal_football_values(changes: dict[str, object]) -> None:
    nfl = _module()
    with pytest.raises(nfl.NFLGameStateError):
        _state(**changes)


def test_nflverse_adapter_uses_only_offline_pre_post_observations() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()

    state, event = _adapt_nflverse_observations(
        pre,
        post,
        state_sequence=8,
    )
    next_state = nfl.reduce(state, event)

    assert state.sequence == 8
    assert state.game_id == "game_nflverse_2025_01_AWY_HME"
    assert event.sequence == 9
    assert event.observation_mode == "offline"
    assert event.source_play_id == "101"
    assert event.first_down is True
    assert next_state.down == 1
    assert next_state.game_seconds_remaining == 3290
    assert re.fullmatch(r"evt_[0-9a-f]{64}", event.event_id)

    mutated_pre = {
        **pre,
        "epa": -1_000_000,
        "wpa": -1_000_000,
        "home_wp": -1_000_000,
        "fixed_drive_result": "Opp touchdown",
        "home_score": 99,
        "away_score": 98,
    }
    mutated_post = {
        **post,
        "epa": 1_000_000,
        "wpa": 1_000_000,
        "home_wp": 1_000_000,
        "fixed_drive_result": "Field goal",
        "home_score": 1,
        "away_score": 0,
    }
    assert _adapt_nflverse_observations(
        mutated_pre,
        mutated_post,
        state_sequence=8,
    ) == (state, event)


def test_nflverse_adapter_never_falls_back_to_final_score_columns() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    pre.pop("total_home_score")
    post.pop("total_home_score")

    with pytest.raises(nfl.NFLGameStateError, match="total_home_score"):
        _adapt_nflverse_observations(pre, post)


def test_envelope_adapter_uses_complete_payload_and_ignores_next_play_outcome() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    post["play_type"] = "future-pass"
    post["series_result"] = "future touchdown"

    payload = nfl.nflverse_transition_payload(pre, post, sequence=9)
    mutated_payload = nfl.nflverse_transition_payload(
        pre,
        {
            **post,
            "play_type": "future-kneel",
            "series_result": "future punt",
        },
        sequence=9,
    )
    assert payload == mutated_payload
    assert "next_play_type" not in payload

    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "7" * 64,
        raw_record_ordinals=(8, 9),
        partition="season-2025",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=None,
        competition_id="cmp_nfl",
        game_id="game_nflverse_2025_01_AWY_HME",
        participant_ids=("participant_AWY", "participant_HME"),
        native_namespace="nflverse.play",
        native_ids=("2025_01_AWY_HME:101", "2025_01_AWY_HME:122"),
        normalized_source_sequence=9,
        normalized_payload=payload,
    )
    event = nfl.event_from_nflverse_envelope(
        bundle.normalized,
        program_root=PROJECT_ROOT,
        raw_parents=bundle.raw,
    )
    next_state = nfl.reduce(
        nfl.state_from_nflverse_row(pre, sequence=8),
        event,
    )

    assert event.sequence == 9
    assert not hasattr(next_state, "next_play_type")


def test_event_hash_and_reduction_are_canonical_and_deterministic() -> None:
    nfl = _module()
    from prediction_market.sports.game_state import canonical_state_sha256

    state = _state()
    first = _event(state)
    second = _event(state)

    assert first == second
    assert first.event_id == EVENT_ID
    assert canonical_state_sha256(first) == canonical_state_sha256(second)
    assert nfl.reduce(state, first) == nfl.reduce(state, second)
    with pytest.raises(nfl.NFLGameStateError, match="event_id"):
        replace(first, event_id="not-an-envelope-event")


def test_reducer_object_conforms_to_the_common_game_state_protocol() -> None:
    nfl = _module()
    from prediction_market.sports.game_state import (
        GameStateReducer,
        advance_state,
    )

    state = _state()
    event = _event(state)
    reducer = nfl.NFL_GAME_STATE_REDUCER

    assert isinstance(reducer, GameStateReducer)
    assert reducer.sport == "nfl"
    assert reducer.reducer_id == "REDUCER-NFL-PLAY-STATE"
    assert reducer.reducer_version == "v1"
    trace = advance_state(reducer, state, event)
    assert trace.next_state == nfl.reduce(state, event)
