from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_envelopes,
)
from prediction_market.sports.game_state import advance_state
from prediction_market.sports.state_validation import validate_state_replay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "game-state"
    / "nba"
    / "x06_contract_fixture_v0.json"
)
FIXTURE_SHA256 = (
    "sha256:0ebca90ba56b45252c6d310fd176ba2eb1ac615995b3eacf92e2e3f9c962a15f"
)
GAME_ID = "game_nba_synthetic_x06"
HOME_TEAM = "participant_NBA_HOME"
AWAY_TEAM = "participant_NBA_AWAY"


def _module():
    from prediction_market.sports import nba_game_state

    return nba_game_state


def _state(**changes: object):
    nba = _module()
    values: dict[str, object] = {
        "sport": "nba",
        "game_id": GAME_ID,
        "sequence": 0,
        "terminal": False,
        "state_kind": "source_observed",
        "observation_mode": "synthetic_fixture",
        "home_team": HOME_TEAM,
        "away_team": AWAY_TEAM,
        "period": 1,
        "clock_ms": 700_000,
        "home_score": 0,
        "away_score": 0,
        "possession_team": HOME_TEAM,
        "home_team_fouls": 0,
        "away_team_fouls": 0,
        "home_timeouts_remaining": 7,
        "away_timeouts_remaining": 7,
        "home_in_bonus": False,
        "away_in_bonus": False,
        "shot_clock_ms": 20_000,
        "live_ball": True,
        "last_event_id": None,
        "terminal_reason": None,
    }
    values.update(changes)
    return nba.NBAGameState(**values)


def _envelope_id(
    *,
    sequence: int,
    kind: str,
    payload: dict[str, object],
) -> str:
    pair = build_static_sport_observation_envelopes(
        program_root=PROJECT_ROOT,
        experiment_id="X-06",
        dataset_id="DS-NBA-CANDIDATE",
        source_system="saf-synthetic-fixture",
        source_stream="x06_contract_fixture",
        raw_object_hash=FIXTURE_SHA256,
        raw_record_ordinal=sequence,
        partition="x06_contract_fixture_v0",
        fetched_at="2026-07-23T00:00:00Z",
        source_at=None,
        competition_id="cmp_nba",
        game_id=GAME_ID,
        participant_ids=(HOME_TEAM, AWAY_TEAM),
        native_namespace="saf.x06.synthetic_event",
        native_id=f"{kind}:{sequence}",
        normalized_payload={
            "sport": "nba",
            "kind": kind,
            "sequence": sequence,
            **payload,
        },
        quality_flags=("source_clock_unverified",),
    )
    return pair.normalized.event_id


def _event(state, *, kind: str, **changes: object):
    nba = _module()
    sequence = state.sequence + 1
    values: dict[str, object] = {
        "sport": "nba",
        "game_id": state.game_id,
        "sequence": sequence,
        "event_id": "",
        "observation_mode": "synthetic_fixture",
        "kind": kind,
        "period": state.period,
        "clock_ms": max(0, state.clock_ms - 1_000),
        "shot_clock_ms": state.shot_clock_ms,
        "live_ball": state.live_ball,
        "home_in_bonus_after": state.home_in_bonus,
        "away_in_bonus_after": state.away_in_bonus,
        "team_id": None,
        "points": None,
        "possession_team_after": state.possession_team,
        "terminal_reason": None,
    }
    values.update(changes)
    payload = {
        key: value
        for key, value in values.items()
        if key not in {"event_id", "game_id", "observation_mode"}
    }
    values["event_id"] = _envelope_id(
        sequence=sequence,
        kind=kind,
        payload=payload,
    )
    return nba.NBAGameEvent(**values)


def test_nba_state_is_immutable_source_observation_without_prediction_fields() -> None:
    nba = _module()
    state = _state()

    assert state.state_kind == "source_observed"
    assert state.observation_mode == "synthetic_fixture"
    assert not any(
        "probability" in field or "prediction" in field
        for field in state.__dataclass_fields__
    )
    with pytest.raises(FrozenInstanceError):
        state.home_score = 1
    with pytest.raises(nba.NBAGameStateError, match="integer"):
        _state(clock_ms=700_000.0)
    with pytest.raises(nba.NBAGameStateError, match="integer"):
        replace(
            _event(state, kind="score", team_id=HOME_TEAM, points=2),
            points=2.0,
        )
    late_period_bonus = _state(
        clock_ms=90_000,
        away_team_fouls=2,
        home_in_bonus=True,
    )
    assert late_period_bonus.home_in_bonus is True


def test_registered_envelope_id_drives_common_hash_chain() -> None:
    state = _state()
    event = _event(
        state,
        kind="score",
        team_id=HOME_TEAM,
        points=2,
        clock_ms=695_000,
        live_ball=False,
        shot_clock_ms=None,
        possession_team_after=AWAY_TEAM,
    )

    trace = advance_state(_module().NBA_GAME_STATE_REDUCER, state, event)

    assert trace.event_id == event.event_id
    assert trace.next_state.home_score == 2
    assert trace.next_state.away_score == 0
    assert trace.next_state.possession_team == AWAY_TEAM
    assert trace.next_state.last_event_id == event.event_id
    assert trace.previous_state_sha256.startswith("sha256:")
    assert trace.next_state_sha256.startswith("sha256:")
    assert trace.trace_sha256.startswith("sha256:")


def test_all_required_transition_types_replay_deterministically() -> None:
    nba = _module()
    initial = _state()
    events = []
    state = initial

    score = _event(
        state,
        kind="score",
        team_id=HOME_TEAM,
        points=2,
        live_ball=False,
        shot_clock_ms=None,
        possession_team_after=AWAY_TEAM,
    )
    events.append(score)
    state = nba.reduce(state, score)

    foul_events = []
    for _ in range(5):
        foul = _event(
            state,
            kind="foul",
            team_id=AWAY_TEAM,
            live_ball=False,
            shot_clock_ms=None,
            home_in_bonus_after=state.away_team_fouls + 1 >= 5,
        )
        foul_events.append(foul)
        events.append(foul)
        state = nba.reduce(state, foul)
    assert state.away_team_fouls == 5
    assert state.home_in_bonus is True

    timeout = _event(
        state,
        kind="timeout",
        team_id=HOME_TEAM,
        live_ball=False,
        shot_clock_ms=None,
    )
    events.append(timeout)
    state = nba.reduce(state, timeout)
    assert state.home_timeouts_remaining == 6

    possession = _event(
        state,
        kind="possession",
        possession_team_after=HOME_TEAM,
        live_ball=True,
        shot_clock_ms=24_000,
    )
    events.append(possession)
    state = nba.reduce(state, possession)
    assert state.possession_team == HOME_TEAM

    period = _event(
        state,
        kind="period",
        period=2,
        clock_ms=720_000,
        possession_team_after=AWAY_TEAM,
        live_ball=False,
        shot_clock_ms=None,
        home_in_bonus_after=False,
        away_in_bonus_after=False,
    )
    events.append(period)
    state = nba.reduce(state, period)
    assert state.period == 2
    assert state.home_team_fouls == 0
    assert state.away_team_fouls == 0
    assert state.home_in_bonus is False

    for next_period in (3, 4):
        period = _event(
            state,
            kind="period",
            period=next_period,
            clock_ms=720_000,
            possession_team_after=HOME_TEAM,
            live_ball=False,
            shot_clock_ms=None,
            home_in_bonus_after=False,
            away_in_bonus_after=False,
        )
        events.append(period)
        state = nba.reduce(state, period)

    terminal = _event(
        state,
        kind="terminal",
        clock_ms=0,
        possession_team_after=None,
        live_ball=False,
        shot_clock_ms=None,
        terminal_reason="final_horn",
    )
    events.append(terminal)
    terminal_trace = advance_state(
        nba.NBA_GAME_STATE_REDUCER,
        state,
        terminal,
    )
    assert terminal_trace.next_state.terminal is True

    validation = validate_state_replay(
        reducer=nba.NBA_GAME_STATE_REDUCER,
        initial_state=initial,
        events=events,
    )
    assert validation.deterministic is True
    assert validation.first_replay_sha256 == validation.second_replay_sha256
    assert validation.events == len(events)


def test_transition_rules_fail_closed_on_semantic_mismatch() -> None:
    nba = _module()
    state = _state()

    with pytest.raises(nba.NBAGameStateError, match="score"):
        nba.reduce(
            state,
            _event(state, kind="score", team_id=HOME_TEAM, points=None),
        )
    with pytest.raises(nba.NBAGameStateError, match="clock"):
        nba.reduce(
            state,
            _event(state, kind="possession", clock_ms=701_000),
        )
    with pytest.raises(nba.NBAGameStateError, match="period"):
        nba.reduce(
            state,
            _event(state, kind="period", period=3, clock_ms=720_000),
        )
    with pytest.raises(nba.NBAGameStateError, match="timeout"):
        nba.reduce(
            _state(home_timeouts_remaining=0),
            _event(
                _state(home_timeouts_remaining=0),
                kind="timeout",
                team_id=HOME_TEAM,
            ),
        )
    with pytest.raises(nba.NBAGameStateError, match="terminal"):
        nba.reduce(
            state,
            _event(
                state,
                kind="terminal",
                clock_ms=0,
                live_ball=False,
                shot_clock_ms=None,
                possession_team_after=None,
                terminal_reason="final_horn",
            ),
        )


def test_x06_fixture_maps_only_to_synthetic_score_transitions() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["data_class"] == "synthetic_contract_only"
    assert fixture["schema"] == "x06_synthetic_contract_fixture_v0"

    state = _state(live_ball=False, shot_clock_ms=None)
    for transition in fixture["transitions"]:
        scoring_team = (
            HOME_TEAM
            if transition["to_state"] == "home_score"
            else AWAY_TEAM
        )
        event = _event(
            state,
            kind="score",
            team_id=scoring_team,
            points=2,
            live_ball=False,
            shot_clock_ms=None,
        )
        state = _module().reduce(state, event)

    assert (state.home_score, state.away_score) == (2, 2)


def test_committed_validation_is_reducer_only_and_keeps_license_gate_blocked() -> None:
    artifact = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "game-state"
            / "nba_state_engine_validation_v0.json"
        ).read_text(encoding="utf-8")
    )

    assert artifact["experiment_id"] == "X-06"
    assert artifact["data_class"] == "synthetic_contract_only"
    assert artifact["result_label"] == "PRELIMINARY_ENGINEERING_VALIDATION"
    assert artifact["replay"]["deterministic"] is True
    assert artifact["replay"]["first_replay_sha256"] == artifact["replay"][
        "second_replay_sha256"
    ]
    assert artifact["latency"]["measurement"] == "reducer_only"
    assert artifact["latency"]["timed_iterations"] >= 1_000
    assert artifact["latency"]["includes_model_inference"] is False
    assert artifact["accuracy"]["measured"] is False
    assert artifact["accuracy"]["real_nba_games"] == 0
    assert artifact["license_gate"]["id"] == "O-005"
    assert artifact["license_gate"]["status"] == "BLOCKED"
