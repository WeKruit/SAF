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
        "season_type": "REG",
        "home_team": "HME",
        "away_team": "AWY",
        "period": 1,
        "period_seconds_remaining": 600,
        "game_seconds_remaining": 3300,
        "source_play_id": "101",
        "source_order_sequence": 100,
        "context_source_play_id": "101",
        "context_source_order_sequence": 100,
        "suspended": False,
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
    default_next_play_id = (
        "123" if state.source_play_id == "122" else "122"
    )
    values: dict[str, object] = {
        "sport": "nfl",
        "game_id": state.game_id,
        "sequence": state.sequence + 1,
        "event_id": EVENT_ID,
        "season_type": state.season_type,
        "source_play_id": state.source_play_id,
        "source_order_sequence": state.source_order_sequence,
        "observation_mode": "offline",
        "play_type": "run",
        "play_type_nfl": "RUSH",
        "description": "HME runner gains ten yards.",
        "period": state.period,
        "period_seconds_remaining": state.period_seconds_remaining - 10,
        "game_seconds_remaining": state.game_seconds_remaining - 10,
        "next_source_play_id": default_next_play_id,
        "next_source_order_sequence": state.source_order_sequence + 1,
        "context_source_play_id": default_next_play_id,
        "context_source_order_sequence": state.source_order_sequence + 1,
        "source_window_play_ids": (
            state.source_play_id,
            default_next_play_id,
        ),
        "source_window_order_sequences": (
            state.source_order_sequence,
            state.source_order_sequence + 1,
        ),
        "lifecycle_action": "none",
        "clock_carry_forward": False,
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
        "timeout_observed": False,
        "timeout_observed_team": None,
        "timeout_kind": "none",
        "timeout_charge_team": None,
        "quarter_end": False,
        "clock_correction": False,
        "clock_correction_observed_period_seconds_remaining": None,
        "clock_correction_observed_game_seconds_remaining": None,
        "carry_forward_context": False,
        "period_changed": False,
        "terminal": False,
        "quality_flags": (),
    }
    values.update(changes)
    if "source_window_play_ids" not in changes:
        values["source_window_play_ids"] = (
            values["source_play_id"],
            values["next_source_play_id"],
        )
    if "source_window_order_sequences" not in changes:
        values["source_window_order_sequences"] = (
            values["source_order_sequence"],
            values["next_source_order_sequence"],
        )
    if bool(values["carry_forward_context"]):
        if "context_source_play_id" not in changes:
            values["context_source_play_id"] = state.context_source_play_id
        if "context_source_order_sequence" not in changes:
            values["context_source_order_sequence"] = (
                state.context_source_order_sequence
            )
    else:
        if "context_source_play_id" not in changes:
            values["context_source_play_id"] = values[
                "source_window_play_ids"
            ][-1]
        if "context_source_order_sequence" not in changes:
            values["context_source_order_sequence"] = values[
                "source_window_order_sequences"
            ][-1]
    return nfl.NFLPlayEvent(**values)


def _nflverse_rows() -> tuple[dict[str, object], dict[str, object]]:
    pre = {
        "game_id": "2025_01_AWY_HME",
        "play_id": 101.0,
        "order_sequence": 100.0,
        "season_type": "REG",
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
        "play_type_nfl": "RUSH",
        "desc": "HME runner gains ten yards.",
        "sp": 0.0,
        "posteam_score": 0.0,
        "defteam_score": 0.0,
        "posteam_score_post": 0.0,
        "defteam_score_post": 0.0,
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
        "order_sequence": 101.0,
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


def _nflverse_payload(
    source: dict[str, object],
    *successors: dict[str, object],
    state: Any | None = None,
    sequence: int = 1,
) -> dict[str, object]:
    nfl = _module()
    current = (
        nfl.state_from_nflverse_row(source)
        if state is None
        else state
    )
    return nfl.nflverse_transition_payload(
        current,
        source,
        tuple(successors),
        sequence=sequence,
    )


def _adapt_nflverse_observations(
    pre: dict[str, object],
    *successors: dict[str, object],
    state: Any | None = None,
    state_sequence: int = 0,
) -> tuple[Any, Any]:
    nfl = _module()
    if not successors:
        raise AssertionError("test adapter requires a successor window")
    current_state = (
        nfl.state_from_nflverse_row(pre, sequence=state_sequence)
        if state is None
        else state
    )
    payload = nfl.nflverse_transition_payload(
        current_state,
        pre,
        tuple(successors),
        sequence=current_state.sequence + 1,
    )
    native_game_id = str(pre["game_id"])
    window_rows = (pre, *successors)
    window_play_ids = tuple(
        str(int(float(row["play_id"]))) for row in window_rows
    )
    raw_record_ordinals = tuple(
        range(state_sequence, state_sequence + len(window_rows))
    )
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "7" * 64,
        raw_record_ordinals=raw_record_ordinals,
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
        native_ids=tuple(
            f"{native_game_id}:{play_id}"
            for play_id in window_play_ids
        ),
        normalized_source_sequence=current_state.sequence + 1,
        normalized_payload=payload,
    )
    event = nfl.event_from_nflverse_envelope(
        bundle.normalized,
        program_root=PROJECT_ROOT,
        raw_parents=bundle.raw,
    )
    return current_state, event


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
            quarter_end=True,
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
            play_type_nfl="TIMEOUT",
            home_timeouts_remaining=2,
            timeout_observed=True,
            timeout_observed_team="HME",
            timeout_kind="administrative",
            timeout_charge_team="HME",
            carry_forward_context=True,
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


def test_reducer_rejects_source_identity_or_order_discontinuity() -> None:
    nfl = _module()
    state = _state(source_play_id="100", source_order_sequence=100)

    with pytest.raises(nfl.NFLGameStateError, match="source_play_id"):
        nfl.reduce(state, _event(state, source_play_id="101"))
    with pytest.raises(nfl.NFLGameStateError, match="source_order_sequence"):
        nfl.reduce(
            state,
            _event(
                state,
                source_order_sequence=101,
                next_source_order_sequence=102,
            ),
        )
    with pytest.raises(nfl.NFLGameStateError, match="strictly increase"):
        nfl.reduce(
            state,
            _event(
                state,
                next_source_order_sequence=state.source_order_sequence,
            ),
        )


def test_reducer_rejects_season_type_discontinuity() -> None:
    nfl = _module()
    state = _state(season_type="REG")

    with pytest.raises(nfl.NFLGameStateError, match="season_type"):
        nfl.reduce(state, _event(state, season_type="POST"))


@pytest.mark.parametrize(
    ("season", "seconds"),
    [(2015, 900), (2016, 900), (2017, 600), (2025, 600)],
)
def test_regular_season_overtime_duration_uses_historical_rule_snapshot(
    season: int,
    seconds: int,
) -> None:
    nfl = _module()
    state = _state(
        game_id=f"game_nflverse_{season}_01_AWY_HME",
        season_type="REG",
        period=4,
        period_seconds_remaining=0,
        game_seconds_remaining=0,
        down=None,
        distance=None,
        yardline_100=None,
        home_score=20,
        away_score=20,
        home_timeouts_remaining=1,
        away_timeouts_remaining=0,
    )
    overtime = _event(
        state,
        period=5,
        period_seconds_remaining=seconds,
        game_seconds_remaining=seconds,
        home_timeouts_remaining=2,
        away_timeouts_remaining=2,
        down=1,
        distance=10,
        yardline_100=75,
        first_down=False,
        quarter_end=True,
        period_changed=True,
    )

    assert nfl.reduce(state, overtime).period_seconds_remaining == seconds


def test_postseason_overtime_uses_three_timeouts_per_half() -> None:
    nfl = _module()
    state = _state(
        season_type="POST",
        period=4,
        period_seconds_remaining=0,
        game_seconds_remaining=0,
        down=None,
        distance=None,
        yardline_100=None,
        home_score=20,
        away_score=20,
        home_timeouts_remaining=1,
        away_timeouts_remaining=0,
    )
    overtime = _event(
        state,
        period=5,
        period_seconds_remaining=900,
        game_seconds_remaining=900,
        home_timeouts_remaining=3,
        away_timeouts_remaining=3,
        down=1,
        distance=10,
        yardline_100=75,
        first_down=False,
        quarter_end=True,
        period_changed=True,
    )

    assert nfl.reduce(state, overtime).away_timeouts_remaining == 3


def test_overtime_rejects_wrong_timeout_allotment_at_period_five() -> None:
    nfl = _module()
    regulation_end = {
        "period": 4,
        "period_seconds_remaining": 0,
        "game_seconds_remaining": 0,
        "down": None,
        "distance": None,
        "yardline_100": None,
        "home_score": 20,
        "away_score": 20,
        "home_timeouts_remaining": 1,
        "away_timeouts_remaining": 0,
    }

    postseason = _state(season_type="POST", **regulation_end)
    with pytest.raises(nfl.NFLGameStateError, match="timeouts|consume"):
        nfl.reduce(
            postseason,
            _event(
                postseason,
                period=5,
                period_seconds_remaining=900,
                game_seconds_remaining=900,
                home_timeouts_remaining=2,
                away_timeouts_remaining=2,
                down=1,
                distance=10,
                yardline_100=75,
                first_down=False,
                quarter_end=True,
                period_changed=True,
            ),
        )

    regular = _state(season_type="REG", **regulation_end)
    with pytest.raises(nfl.NFLGameStateError, match="timeouts|allotment"):
        nfl.reduce(
            regular,
            _event(
                regular,
                period=5,
                period_seconds_remaining=600,
                game_seconds_remaining=600,
                home_timeouts_remaining=3,
                away_timeouts_remaining=3,
                down=1,
                distance=10,
                yardline_100=75,
                first_down=False,
                quarter_end=True,
                period_changed=True,
            ),
        )


def test_postseason_timeout_reset_occurs_every_two_overtime_periods() -> None:
    nfl = _module()
    first_half_end = _state(
        season_type="POST",
        period=5,
        period_seconds_remaining=0,
        game_seconds_remaining=0,
        home_score=20,
        away_score=20,
        home_timeouts_remaining=2,
        away_timeouts_remaining=1,
    )
    period_six = _event(
        first_half_end,
        period=6,
        period_seconds_remaining=900,
        game_seconds_remaining=900,
        home_timeouts_remaining=2,
        away_timeouts_remaining=1,
        first_down=False,
        quarter_end=True,
        period_changed=True,
    )
    assert nfl.reduce(first_half_end, period_six).home_timeouts_remaining == 2

    with pytest.raises(nfl.NFLGameStateError, match="timeouts|allotment"):
        nfl.reduce(
            first_half_end,
            _event(
                first_half_end,
                period=6,
                period_seconds_remaining=900,
                game_seconds_remaining=900,
                home_timeouts_remaining=3,
                away_timeouts_remaining=3,
                first_down=False,
                quarter_end=True,
                period_changed=True,
            ),
        )

    second_half_end = _state(
        season_type="POST",
        period=6,
        period_seconds_remaining=0,
        game_seconds_remaining=0,
        home_score=20,
        away_score=20,
        home_timeouts_remaining=2,
        away_timeouts_remaining=1,
    )
    period_seven = _event(
        second_half_end,
        period=7,
        period_seconds_remaining=900,
        game_seconds_remaining=900,
        home_timeouts_remaining=3,
        away_timeouts_remaining=3,
        first_down=False,
        quarter_end=True,
        period_changed=True,
    )
    assert nfl.reduce(second_half_end, period_seven).away_timeouts_remaining == 3
    with pytest.raises(nfl.NFLGameStateError, match="timeouts|multiple"):
        nfl.reduce(
            second_half_end,
            _event(
                second_half_end,
                period=7,
                period_seconds_remaining=900,
                game_seconds_remaining=900,
                home_timeouts_remaining=2,
                away_timeouts_remaining=1,
                first_down=False,
                quarter_end=True,
                period_changed=True,
            ),
        )


def test_native_postseason_overtime_timeout_counters_are_rules_normalized() -> None:
    nfl = _module()
    regulation_end, overtime = _nflverse_rows()
    regulation_end.update(
        {
            "season_type": "POST",
            "qtr": 4.0,
            "quarter_seconds_remaining": 0.0,
            "game_seconds_remaining": 0.0,
            "home_timeouts_remaining": 1.0,
            "away_timeouts_remaining": 0.0,
            "play_type": None,
            "play_type_nfl": "END_QUARTER",
            "quarter_end": 1.0,
        }
    )
    overtime.update(
        {
            "season_type": "POST",
            "qtr": 5.0,
            "quarter_seconds_remaining": 900.0,
            "game_seconds_remaining": 900.0,
            # nflverse's native POST OT counters are one below Rule 16.
            "home_timeouts_remaining": 2.0,
            "away_timeouts_remaining": 2.0,
        }
    )

    state, event = _adapt_nflverse_observations(regulation_end, overtime)
    after_reset = nfl.reduce(state, event)

    assert event.home_timeouts_remaining == 3
    assert event.away_timeouts_remaining == 3
    assert after_reset.home_timeouts_remaining == 3
    with pytest.raises(nfl.NFLGameStateError, match="normalized timeout"):
        nfl.state_from_nflverse_row(
            {**overtime, "home_timeouts_remaining": -2.0}
        )


def test_same_period_clock_increase_requires_inserted_admin_timeout() -> None:
    nfl = _module()
    state = _state(
        source_play_id="200",
        source_order_sequence=200,
        period_seconds_remaining=500,
        game_seconds_remaining=3200,
    )
    common = {
        "period_seconds_remaining": 510,
        "game_seconds_remaining": 3210,
        "first_down": False,
        "home_timeouts_remaining": 2,
        "timeout_observed": True,
        "timeout_observed_team": "HME",
        "timeout_kind": "administrative",
        "timeout_charge_team": "HME",
        "carry_forward_context": True,
        "play_type": "no_play",
        "play_type_nfl": "TIMEOUT",
        "next_source_play_id": "199",
        "next_source_order_sequence": 201,
    }

    with pytest.raises(nfl.NFLGameStateError, match="clock"):
        nfl.reduce(
            state,
            _event(
                state,
                **common,
                clock_correction=False,
                quality_flags=(),
            ),
        )
    with pytest.raises(nfl.NFLGameStateError, match="inserted"):
        nfl.reduce(
            state,
            _event(
                state,
                **{**common, "next_source_play_id": "201"},
                clock_correction=True,
                clock_correction_observed_period_seconds_remaining=510,
                clock_correction_observed_game_seconds_remaining=3210,
                quality_flags=("source_order_inserted_timeout",),
            ),
        )


def test_public_event_flags_cannot_authorize_clock_correction() -> None:
    nfl = _module()
    state = _state(
        source_play_id="200",
        source_order_sequence=200,
        period_seconds_remaining=500,
        game_seconds_remaining=3200,
    )

    forged = _event(
        state,
        period_seconds_remaining=510,
        game_seconds_remaining=3210,
        first_down=False,
        home_timeouts_remaining=2,
        timeout_observed=True,
        timeout_observed_team="HME",
        timeout_kind="administrative",
        timeout_charge_team="HME",
        carry_forward_context=True,
        play_type="no_play",
        play_type_nfl="TIMEOUT",
        next_source_play_id="199",
        next_source_order_sequence=201,
        clock_correction=True,
        clock_correction_observed_period_seconds_remaining=510,
        clock_correction_observed_game_seconds_remaining=3210,
        quality_flags=("source_order_inserted_timeout",),
    )

    with pytest.raises(nfl.NFLGameStateError, match="moved backwards"):
        nfl.reduce(state, forged)


def test_reducer_rejects_correction_without_observed_clock_increase() -> None:
    nfl = _module()
    state = _state(
        source_play_id="200",
        source_order_sequence=200,
        period_seconds_remaining=500,
        game_seconds_remaining=3200,
    )
    forged = _event(
        state,
        period_seconds_remaining=500,
        game_seconds_remaining=3200,
        first_down=False,
        home_timeouts_remaining=2,
        timeout_observed=True,
        timeout_observed_team="HME",
        timeout_kind="administrative",
        timeout_charge_team="HME",
        carry_forward_context=True,
        play_type="no_play",
        play_type_nfl="TIMEOUT",
        next_source_play_id="199",
        next_source_order_sequence=201,
        clock_correction=True,
        clock_correction_observed_period_seconds_remaining=500,
        clock_correction_observed_game_seconds_remaining=3200,
        quality_flags=("source_order_inserted_timeout",),
    )

    with pytest.raises(nfl.NFLGameStateError, match="did not increase"):
        nfl.reduce(state, forged)


def test_lifecycle_events_carry_clock_and_update_suspension() -> None:
    nfl = _module()
    state = _state()
    suspended = nfl.reduce(
        state,
        _event(
            state,
            play_type="comment",
            play_type_nfl="COMMENT",
            description="The game has been suspended.",
            lifecycle_action="suspend",
            clock_carry_forward=True,
            period=None,
            period_seconds_remaining=None,
            game_seconds_remaining=None,
            carry_forward_context=True,
            first_down=False,
        ),
    )

    assert suspended.suspended is True
    assert (
        suspended.period,
        suspended.period_seconds_remaining,
        suspended.game_seconds_remaining,
    ) == (
        state.period,
        state.period_seconds_remaining,
        state.game_seconds_remaining,
    )
    resumed = nfl.reduce(
        suspended,
        _event(
            suspended,
            play_type="comment",
            play_type_nfl="COMMENT",
            description="The game has resumed.",
            lifecycle_action="resume",
            clock_carry_forward=True,
            period=None,
            period_seconds_remaining=None,
            game_seconds_remaining=None,
            carry_forward_context=True,
            first_down=False,
        ),
    )
    assert resumed.suspended is False


def test_lifecycle_rejects_invalid_state_machine_transitions() -> None:
    nfl = _module()
    active = _state()
    resume = _event(
        active,
        play_type_nfl="COMMENT",
        description="The game has resumed.",
        lifecycle_action="resume",
        clock_carry_forward=True,
        period=None,
        period_seconds_remaining=None,
        game_seconds_remaining=None,
        carry_forward_context=True,
        first_down=False,
    )
    with pytest.raises(nfl.NFLGameStateError, match="not suspended"):
        nfl.reduce(active, resume)

    suspended = _state(suspended=True)
    suspend = _event(
        suspended,
        play_type_nfl="COMMENT",
        description="The game has been suspended.",
        lifecycle_action="suspend",
        clock_carry_forward=True,
        period=None,
        period_seconds_remaining=None,
        game_seconds_remaining=None,
        carry_forward_context=True,
        first_down=False,
    )
    with pytest.raises(nfl.NFLGameStateError, match="already suspended"):
        nfl.reduce(suspended, suspend)
    with pytest.raises(nfl.NFLGameStateError, match="resume"):
        nfl.reduce(suspended, _event(suspended))


def test_arbitrary_missing_clock_comment_fails_closed() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    pre.update(
        {
            "play_type": "no_play",
            "play_type_nfl": "COMMENT",
            "desc": "Weather update.",
            "quarter_seconds_remaining": None,
            "game_seconds_remaining": None,
            "time": None,
        }
    )

    with pytest.raises(nfl.NFLGameStateError, match="clock|qtr|time"):
        _nflverse_payload(pre, post)


def test_bounded_comment_descriptions_derive_lifecycle_actions() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    pre.update(
        {
            "play_type": "no_play",
            "play_type_nfl": "COMMENT",
            "desc": "The game has been suspended. Lightning nearby.",
            "quarter_seconds_remaining": None,
            "game_seconds_remaining": None,
            "time": None,
            "posteam": None,
            "down": None,
            "ydstogo": 0.0,
            "yardline_100": None,
            "posteam_score": None,
            "defteam_score": None,
            "posteam_score_post": None,
            "defteam_score_post": None,
        }
    )

    payload = _nflverse_payload(pre, post, state=_state())

    assert payload["lifecycle_action"] == "suspend"
    assert payload["clock_carry_forward"] is True
    assert payload["period"] is None
    assert payload["period_seconds_remaining"] is None
    assert payload["game_seconds_remaining"] is None


def test_timeout_kind_distinguishes_admin_from_play_attached_charge() -> None:
    nfl = _module()
    state = _state()
    admin = nfl.reduce(
        state,
        _event(
            state,
            first_down=False,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            home_timeouts_remaining=2,
            timeout_observed=True,
            timeout_observed_team="HME",
            timeout_kind="administrative",
            timeout_charge_team="HME",
            carry_forward_context=True,
        ),
    )
    assert admin.down == state.down

    attached_row, following_row = _nflverse_rows()
    attached_row.update(
        {
            "timeout": 1.0,
            "timeout_team": "HME",
            "home_timeouts_remaining": 2.0,
        }
    )
    attached_state = _state(
        source_play_id="101",
        source_order_sequence=100,
        home_timeouts_remaining=3,
    )
    _, attached = _adapt_nflverse_observations(
        attached_row,
        following_row,
        state=attached_state,
    )
    after_attached = nfl.reduce(attached_state, attached)

    assert attached.timeout_kind == "play_attached"
    assert attached.carry_forward_context is False
    assert after_attached.home_timeouts_remaining == 2
    assert after_attached.down == 1
    assert after_attached.yardline_100 == 35


def test_context_carry_rejects_forged_context_lineage() -> None:
    nfl = _module()
    state = _state(
        context_source_play_id="90",
        context_source_order_sequence=90,
    )
    forged = _event(
        state,
        first_down=False,
        carry_forward_context=True,
        context_source_play_id="999",
        context_source_order_sequence=999,
    )

    with pytest.raises(nfl.NFLGameStateError, match="context source"):
        nfl.reduce(state, forged)


def test_timeout_kind_and_charged_counter_must_agree() -> None:
    nfl = _module()
    state = _state()

    with pytest.raises(nfl.NFLGameStateError, match="timeout_kind"):
        _event(state, timeout_kind="administrative")
    with pytest.raises(nfl.NFLGameStateError, match="timeout_kind"):
        _event(
            state,
            home_timeouts_remaining=2,
            timeout_observed=True,
            timeout_observed_team="HME",
            timeout_kind="none",
            timeout_charge_team="HME",
        )


def test_event_source_types_prove_admin_and_lifecycle_classification() -> None:
    nfl = _module()
    state = _state()

    with pytest.raises(nfl.NFLGameStateError, match="administrative"):
        _event(
            state,
            play_type="run",
            play_type_nfl="TIMEOUT",
            first_down=False,
            home_timeouts_remaining=2,
            timeout_observed=True,
            timeout_observed_team="HME",
            timeout_kind="administrative",
            timeout_charge_team="HME",
            carry_forward_context=True,
        )
    with pytest.raises(nfl.NFLGameStateError, match="lifecycle"):
        _event(
            state,
            play_type="run",
            play_type_nfl="RUSH",
            description="The game has been suspended.",
            lifecycle_action="suspend",
            clock_carry_forward=True,
            period=None,
            period_seconds_remaining=None,
            game_seconds_remaining=None,
            carry_forward_context=True,
            first_down=False,
        )


def test_unobserved_timeout_source_is_not_classified_as_a_timeout() -> None:
    nfl = _module()
    timeout_row, post = _nflverse_rows()
    timeout_row.update(
        {
            "play_type": "no_play",
            "play_type_nfl": "TIMEOUT",
            "timeout": 0.0,
            "timeout_team": None,
        }
    )
    payload = _nflverse_payload(timeout_row, post)

    assert payload["timeout_observed"] is False
    assert payload["timeout_observed_team"] is None
    assert payload["timeout_kind"] == "none"
    assert payload["timeout_charge_team"] is None


def test_administrative_timeout_cannot_change_period_or_end_game() -> None:
    nfl = _module()
    state = _state(
        period=4,
        period_seconds_remaining=5,
        game_seconds_remaining=5,
        home_score=14,
        away_score=10,
    )
    event = _event(
        state,
        play_type="no_play",
        play_type_nfl="TIMEOUT",
        period_seconds_remaining=0,
        game_seconds_remaining=0,
        first_down=False,
        home_timeouts_remaining=2,
        timeout_observed=True,
        timeout_observed_team="HME",
        timeout_kind="administrative",
        timeout_charge_team="HME",
        carry_forward_context=True,
        terminal=True,
    )

    with pytest.raises(nfl.NFLGameStateError, match="administrative"):
        nfl.reduce(state, event)


def test_timeout_native_type_with_real_play_is_not_administrative() -> None:
    nfl = _module()
    attached, post = _nflverse_rows()
    attached.update(
        {
            "play_type": "run",
            "play_type_nfl": "TIMEOUT",
            "timeout": 1.0,
            "timeout_team": "HME",
            "home_timeouts_remaining": 2.0,
        }
    )

    payload = _nflverse_payload(attached, post)

    assert payload["timeout_kind"] == "play_attached"
    assert payload["carry_forward_context"] is False


def test_normal_play_uses_first_complete_context_across_timeout_window() -> None:
    nfl = _module()
    source, complete = _nflverse_rows()
    source["fumble_lost"] = 1.0
    timeout = {
        **source,
        "play_id": 150.0,
        "order_sequence": 101.0,
        "quarter_seconds_remaining": 590.0,
        "game_seconds_remaining": 3290.0,
        "posteam": None,
        "down": None,
        "yardline_100": None,
        "posteam_score": None,
        "defteam_score": None,
        "posteam_score_post": None,
        "defteam_score_post": None,
        "play_type": "no_play",
        "play_type_nfl": "TIMEOUT",
        "desc": "Timeout #1 by AWY at 09:50.",
        "timeout": 1.0,
        "timeout_team": "AWY",
        "away_timeouts_remaining": 2.0,
    }
    complete.update(
        {
            "play_id": 140.0,
            "order_sequence": 102.0,
            "posteam": "AWY",
            "down": 1.0,
            "ydstogo": 10.0,
            "yardline_100": 65.0,
            "away_timeouts_remaining": 2.0,
        }
    )
    state = nfl.state_from_nflverse_row(source)

    payload = nfl.nflverse_transition_payload(
        state,
        source,
        (timeout, complete),
    )

    assert payload["next_source_play_id"] == "150"
    assert payload["next_source_order_sequence"] == 101
    assert payload["context_source_play_id"] == "140"
    assert payload["context_source_order_sequence"] == 102
    assert payload["source_window_play_ids"] == ["101", "150", "140"]
    assert payload["source_window_order_sequences"] == [100, 101, 102]
    assert payload["carry_forward_context"] is False
    assert payload["possession_team"] == "AWY"
    assert payload["possession_changed"] is True
    assert payload["turnover"] is True
    assert payload["home_timeouts_remaining"] == 3
    assert payload["away_timeouts_remaining"] == 3


def test_observed_timeout_without_counter_charge_is_representable() -> None:
    nfl = _module()
    source, complete = _nflverse_rows()
    source.update(
        {
            "play_id": 150.0,
            "order_sequence": 101.0,
            "quarter_seconds_remaining": 590.0,
            "game_seconds_remaining": 3290.0,
            "posteam": None,
            "down": None,
            "yardline_100": None,
            "posteam_score": None,
            "defteam_score": None,
            "posteam_score_post": None,
            "defteam_score_post": None,
            "play_type": "no_play",
            "play_type_nfl": "TIMEOUT",
            "desc": "Timeout #4 by AWY at 09:50.",
            "timeout": 1.0,
            "timeout_team": "AWY",
            "away_timeouts_remaining": 0.0,
        }
    )
    complete.update(
        {
            "play_id": 140.0,
            "order_sequence": 102.0,
            "quarter_seconds_remaining": 590.0,
            "game_seconds_remaining": 3290.0,
            "away_timeouts_remaining": 0.0,
        }
    )
    state = _state(
        source_play_id="150",
        source_order_sequence=101,
        away_timeouts_remaining=0,
    )

    payload = nfl.nflverse_transition_payload(
        state,
        source,
        (complete,),
    )
    payload["quality_flags"] = tuple(payload["quality_flags"])
    payload["source_window_play_ids"] = tuple(
        payload["source_window_play_ids"]
    )
    payload["source_window_order_sequences"] = tuple(
        payload["source_window_order_sequences"]
    )
    event = nfl.NFLPlayEvent(event_id=EVENT_ID, **payload)
    next_state = nfl.reduce(state, event)

    assert event.timeout_observed is True
    assert event.timeout_kind == "administrative"
    assert event.timeout_charge_team is None
    assert next_state.away_timeouts_remaining == 0


def test_observed_timeout_team_must_be_a_game_participant() -> None:
    nfl = _module()
    state = _state(away_timeouts_remaining=0)
    event = _event(
        state,
        play_type="no_play",
        play_type_nfl="TIMEOUT",
        timeout_observed=True,
        timeout_observed_team="XXX",
        timeout_kind="administrative",
        timeout_charge_team=None,
        carry_forward_context=True,
        first_down=False,
    )

    with pytest.raises(nfl.NFLGameStateError, match="participant"):
        nfl.reduce(state, event)


def test_nflverse_adapter_owns_inserted_timeout_clock_correction() -> None:
    nfl = _module()
    source, successor = _nflverse_rows()
    source.update(
        {
            "play_id": 200.0,
            "order_sequence": 200.0,
            "quarter_seconds_remaining": 510.0,
            "game_seconds_remaining": 3210.0,
            "play_type": "no_play",
            "play_type_nfl": "TIMEOUT",
            "desc": "Timeout #1 by HME at 08:30.",
            "timeout": 1.0,
            "timeout_team": "HME",
            "home_timeouts_remaining": 2.0,
        }
    )
    successor.update(
        {
            "play_id": 199.0,
            "order_sequence": 201.0,
            "quarter_seconds_remaining": 500.0,
            "game_seconds_remaining": 3200.0,
            "home_timeouts_remaining": 2.0,
        }
    )
    state = _state(
        source_play_id="200",
        source_order_sequence=200,
        period_seconds_remaining=500,
        game_seconds_remaining=3200,
    )

    payload = nfl.nflverse_transition_payload(
        state,
        source,
        (successor,),
    )

    assert payload["clock_correction"] is True
    assert payload["quality_flags"] == [
        "source_order_inserted_timeout"
    ]
    assert payload["period_seconds_remaining"] == (
        state.period_seconds_remaining
    )
    assert payload["game_seconds_remaining"] == (
        state.game_seconds_remaining
    )
    assert payload.get(
        "clock_correction_observed_period_seconds_remaining"
    ) == 510
    assert payload.get(
        "clock_correction_observed_game_seconds_remaining"
    ) == 3210

    payload["quality_flags"] = tuple(payload["quality_flags"])
    payload["source_window_play_ids"] = tuple(
        payload["source_window_play_ids"]
    )
    payload["source_window_order_sequences"] = tuple(
        payload["source_window_order_sequences"]
    )
    event = nfl.NFLPlayEvent(event_id=EVENT_ID, **payload)
    corrected = nfl.reduce(state, event)
    assert corrected.period_seconds_remaining == 500
    assert corrected.game_seconds_remaining == 3200


def test_nflverse_adapter_rejects_caller_or_payload_clock_flags() -> None:
    nfl = _module()
    source, successor = _nflverse_rows()
    state = nfl.state_from_nflverse_row(source)

    with pytest.raises(TypeError, match="quality_flags"):
        nfl.nflverse_transition_payload(
            state,
            source,
            (successor,),
            quality_flags=("source_order_inserted_timeout",),
        )

    payload = nfl.nflverse_transition_payload(
        state,
        source,
        (successor,),
    )
    payload["quality_flags"] = ("source_order_inserted_timeout",)
    payload["source_window_play_ids"] = tuple(
        payload["source_window_play_ids"]
    )
    payload["source_window_order_sequences"] = tuple(
        payload["source_window_order_sequences"]
    )
    with pytest.raises(nfl.NFLGameStateError, match="quality flag"):
        nfl.NFLPlayEvent(event_id=EVENT_ID, **payload)


def test_penalty_row_does_not_generalize_to_context_carry() -> None:
    nfl = _module()
    penalty, post = _nflverse_rows()
    penalty.update(
        {
            "play_type": "no_play",
            "play_type_nfl": "PENALTY",
            "desc": "Penalty on HME.",
            "posteam": None,
            "down": None,
            "ydstogo": 0.0,
            "yardline_100": None,
            "posteam_score": None,
            "defteam_score": None,
            "posteam_score_post": None,
            "defteam_score_post": None,
        }
    )

    payload = _nflverse_payload(penalty, post)

    assert payload["carry_forward_context"] is False


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
    pre.pop("posteam_score_post")

    with pytest.raises(nfl.NFLGameStateError, match="posteam_score_post"):
        _adapt_nflverse_observations(pre, post)


def test_adapter_terminal_state_cannot_be_injected_by_caller() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    state = nfl.state_from_nflverse_row(pre)

    with pytest.raises(TypeError, match="terminal"):
        nfl.nflverse_transition_payload(
            state,
            pre,
            (post,),
            terminal=True,
        )


def test_nflverse_td_and_extra_point_scores_belong_to_their_source_rows() -> None:
    nfl = _module()
    td, extra_point = _nflverse_rows()
    td.update(
        {
            "play_id": 1101.0,
            "order_sequence": 200.0,
            "qtr": 2.0,
            "quarter_seconds_remaining": 866.0,
            "game_seconds_remaining": 2666.0,
            "posteam": "HME",
            "down": 2.0,
            "ydstogo": 3.0,
            "yardline_100": 18.0,
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "desc": "HME runner scores a touchdown.",
            "sp": 1.0,
            "posteam_score": 3.0,
            "defteam_score": 3.0,
            "posteam_score_post": 9.0,
            "defteam_score_post": 3.0,
            "total_home_score": 9.0,
            "total_away_score": 3.0,
            "first_down": 1.0,
        }
    )
    extra_point.update(
        {
            "play_id": 1124.0,
            "order_sequence": 201.0,
            "qtr": 2.0,
            "quarter_seconds_remaining": 860.0,
            "game_seconds_remaining": 2660.0,
            "posteam": "HME",
            "down": None,
            "ydstogo": 0.0,
            "yardline_100": 15.0,
            "play_type": "extra_point",
            "play_type_nfl": "XP_KICK",
            "desc": "HME extra point is good.",
            "sp": 1.0,
            "posteam_score": 9.0,
            "defteam_score": 3.0,
            "posteam_score_post": 10.0,
            "defteam_score_post": 3.0,
            "total_home_score": 10.0,
            "total_away_score": 3.0,
            "first_down": 0.0,
        }
    )
    kickoff = {
        **extra_point,
        "play_id": 1139.0,
        "order_sequence": 202.0,
        "posteam": "AWY",
        "yardline_100": 35.0,
        "play_type": "kickoff",
        "play_type_nfl": "KICK_OFF",
        "desc": "HME kicks to AWY.",
        "sp": 0.0,
        "posteam_score": 3.0,
        "defteam_score": 10.0,
        "posteam_score_post": 3.0,
        "defteam_score_post": 10.0,
        "first_down": 0.0,
    }

    state, touchdown = _adapt_nflverse_observations(td, extra_point)
    assert (state.home_score, state.away_score) == (3, 3)
    assert touchdown.source_play_id == "1101"
    assert touchdown.description == "HME runner scores a touchdown."
    assert (touchdown.home_score, touchdown.away_score) == (9, 3)
    assert touchdown.score is True

    after_touchdown = nfl.reduce(state, touchdown)
    _, point_after = _adapt_nflverse_observations(
        extra_point,
        kickoff,
        state_sequence=1,
    )
    assert point_after.source_play_id == "1124"
    assert point_after.description == "HME extra point is good."
    assert (point_after.home_score, point_after.away_score) == (10, 3)
    assert point_after.score is True

    after_point = nfl.reduce(after_touchdown, point_after)
    assert (after_touchdown.home_score, after_touchdown.away_score) == (9, 3)
    assert (after_point.home_score, after_point.away_score) == (10, 3)


def test_nflverse_timeout_is_same_row_and_context_carries_across_admin_row() -> None:
    nfl = _module()
    ordinary, timeout = _nflverse_rows()
    ordinary.update(
        {
            "play_id": 3881.0,
            "order_sequence": 300.0,
            "qtr": 4.0,
            "quarter_seconds_remaining": 288.0,
            "game_seconds_remaining": 288.0,
            "posteam": "AWY",
            "down": 3.0,
            "ydstogo": 17.0,
            "yardline_100": 35.0,
            "play_type": "run",
            "play_type_nfl": "RUSH",
            "desc": "AWY runner gains seven yards.",
            "posteam_score": 20.0,
            "defteam_score": 10.0,
            "posteam_score_post": 20.0,
            "defteam_score_post": 10.0,
            "total_home_score": 10.0,
            "total_away_score": 20.0,
            "first_down": 0.0,
        }
    )
    timeout.update(
        {
            "play_id": 3915.0,
            "order_sequence": 301.0,
            "qtr": 4.0,
            "quarter_seconds_remaining": 280.0,
            "game_seconds_remaining": 280.0,
            "posteam": None,
            "down": None,
            "ydstogo": 0.0,
            "yardline_100": None,
            "play_type": "no_play",
            "play_type_nfl": "TIMEOUT",
            "desc": "Timeout #1 by HME at 04:40.",
            "posteam_score": None,
            "defteam_score": None,
            "posteam_score_post": None,
            "defteam_score_post": None,
            "total_home_score": 10.0,
            "total_away_score": 20.0,
            "home_timeouts_remaining": 2.0,
            "away_timeouts_remaining": 3.0,
            "timeout": 1.0,
            "timeout_team": "HME",
            "first_down": 0.0,
        }
    )
    next_play = {
        **ordinary,
        "play_id": 3903.0,
        "order_sequence": 302.0,
        "quarter_seconds_remaining": 280.0,
        "game_seconds_remaining": 280.0,
        "down": 4.0,
        "ydstogo": 10.0,
        "yardline_100": 28.0,
        "play_type": "field_goal",
        "play_type_nfl": "FIELD_GOAL",
        "desc": "AWY field goal attempt.",
        "home_timeouts_remaining": 2.0,
        "away_timeouts_remaining": 3.0,
    }

    state, ordinary_event = _adapt_nflverse_observations(
        ordinary,
        timeout,
        next_play,
    )
    assert ordinary_event.source_play_id == "3881"
    assert ordinary_event.timeout_observed is False
    assert ordinary_event.timeout_observed_team is None
    assert ordinary_event.timeout_charge_team is None
    assert ordinary_event.carry_forward_context is False

    before_timeout = nfl.reduce(state, ordinary_event)
    assert before_timeout.home_timeouts_remaining == 3
    assert (
        before_timeout.possession_team,
        before_timeout.down,
        before_timeout.distance,
        before_timeout.yardline_100,
    ) == ("AWY", 4, 10, 28)

    _, timeout_event = _adapt_nflverse_observations(
        timeout,
        next_play,
        state=before_timeout,
        state_sequence=1,
    )
    assert timeout_event.source_play_id == "3915"
    assert timeout_event.timeout_observed is True
    assert timeout_event.timeout_observed_team == "HME"
    assert timeout_event.timeout_charge_team == "HME"
    assert timeout_event.carry_forward_context is True

    after_timeout = nfl.reduce(before_timeout, timeout_event)
    assert after_timeout.home_timeouts_remaining == 2
    assert (
        after_timeout.possession_team,
        after_timeout.down,
        after_timeout.distance,
        after_timeout.yardline_100,
    ) == ("AWY", 4, 10, 28)


def test_envelope_adapter_uses_complete_payload_and_ignores_next_play_outcome() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    post["play_type"] = "future-pass"
    post["series_result"] = "future touchdown"
    state = nfl.state_from_nflverse_row(pre, sequence=8)

    payload = _nflverse_payload(
        pre,
        post,
        state=state,
        sequence=9,
    )
    mutated_payload = _nflverse_payload(
        pre,
        {
            **post,
            "play_type": "future-kneel",
            "series_result": "future punt",
        },
        state=state,
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


def test_envelope_binding_does_not_reorder_transition_by_raw_ordinal() -> None:
    nfl = _module()
    pre, post = _nflverse_rows()
    payload = _nflverse_payload(
        pre,
        post,
        state=nfl.state_from_nflverse_row(pre, sequence=8),
        sequence=9,
    )
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "7" * 64,
        raw_record_ordinals=(20, 10),
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

    assert event.source_play_id == "101"
    assert event.next_source_play_id == "122"
    assert event.source_order_sequence < event.next_source_order_sequence


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
    assert reducer.reducer_version == "v3"
    trace = advance_state(reducer, state, event)
    assert trace.next_state == nfl.reduce(state, event)
