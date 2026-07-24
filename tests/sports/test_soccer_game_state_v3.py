from __future__ import annotations

import hashlib
from typing import Any

import pytest

from prediction_market.sports.soccer_game_state import (
    SoccerGameEvent,
    SoccerGameState,
    SoccerGameStateError,
    initial_soccer_game_state,
    reduce_soccer_game_state,
    statsbomb_event_payload,
)


GAME_ID = "game_statsbomb_v3_fixture"
HOME_TEAM_ID = 10
AWAY_TEAM_ID = 20
_PERIOD_BASE_MINUTES = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}


def _raw_event(
    *,
    index: int,
    action: str,
    team_id: int = HOME_TEAM_ID,
    period: int = 1,
    period_clock_ms: int = 0,
    possession_id: int = 1,
    possession_team_id: int = HOME_TEAM_ID,
    player_id: int | None = None,
    native_out: bool = False,
    off_camera: bool = False,
    shot_outcome: str | None = None,
    shot_type: str = "Open Play",
    **extra: Any,
) -> dict[str, Any]:
    seconds, millisecond = divmod(period_clock_ms, 1_000)
    timestamp_hour, seconds = divmod(seconds, 3_600)
    timestamp_minute, second = divmod(seconds, 60)
    absolute_minute = _PERIOD_BASE_MINUTES[period] + (
        period_clock_ms // 60_000
    )
    raw: dict[str, Any] = {
        "id": f"native-v3-{index}",
        "index": index,
        "period": period,
        "timestamp": (
            f"{timestamp_hour:02d}:{timestamp_minute:02d}:"
            f"{second:02d}.{millisecond:03d}"
        ),
        "minute": absolute_minute,
        "second": second,
        "type": {"id": index, "name": action},
        "team": {"id": team_id, "name": f"Team {team_id}"},
        "possession": possession_id,
        "possession_team": {
            "id": possession_team_id,
            "name": f"Team {possession_team_id}",
        },
    }
    if player_id is not None:
        raw["player"] = {
            "id": player_id,
            "name": f"Player {player_id}",
        }
    if native_out:
        raw["out"] = True
    if off_camera:
        raw["off_camera"] = True
    if action == "Starting XI":
        raw["tactics"] = {
            "formation": 433,
            "lineup": [
                {
                    "player": {
                        "id": team_id * 100 + player,
                        "name": f"Player {team_id}-{player}",
                    },
                    "position": {
                        "id": player,
                        "name": f"Position {player}",
                    },
                    "jersey_number": player,
                }
                for player in range(1, 12)
            ],
        }
    if action == "Pass":
        raw["pass"] = {"end_location": [60.0, 40.0]}
    if action == "Shot":
        raw["shot"] = {
            "end_location": [120.0, 40.0, 1.0],
            "outcome": {
                "id": 97,
                "name": shot_outcome or "Saved",
            },
            "type": {"id": 87, "name": shot_type},
        }
    raw.update(extra)
    return raw


def _domain_event(raw: dict[str, Any]) -> SoccerGameEvent:
    payload = statsbomb_event_payload(raw, game_id=GAME_ID)
    assert payload.pop("sport") == "soccer"
    payload["lineup_player_ids"] = tuple(payload["lineup_player_ids"])
    payload["quality_flags"] = tuple(payload["quality_flags"])
    native_id = str(payload["native_event_id"])
    event_id = "evt_" + hashlib.sha256(native_id.encode()).hexdigest()
    return SoccerGameEvent(event_id=event_id, **payload)  # type: ignore[arg-type]


def _reduce_raw(
    state: SoccerGameState,
    *,
    action: str,
    team_id: int = HOME_TEAM_ID,
    period: int | None = None,
    period_clock_ms: int = 0,
    possession_id: int | None = None,
    possession_team_id: int | None = None,
    **extra: Any,
) -> SoccerGameState:
    return reduce_soccer_game_state(
        state,
        _domain_event(
            _raw_event(
                index=state.sequence + 1,
                action=action,
                team_id=team_id,
                period=period if period is not None else max(state.period, 1),
                period_clock_ms=period_clock_ms,
                possession_id=(
                    possession_id
                    if possession_id is not None
                    else state.possession_id or 1
                ),
                possession_team_id=(
                    possession_team_id
                    if possession_team_id is not None
                    else state.possession_team_id or HOME_TEAM_ID
                ),
                **extra,
            )
        ),
    )


def _started_state() -> SoccerGameState:
    state = initial_soccer_game_state(
        GAME_ID,
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
    )
    state = _reduce_raw(state, action="Starting XI")
    state = _reduce_raw(
        state,
        action="Starting XI",
        team_id=AWAY_TEAM_ID,
    )
    state = _reduce_raw(state, action="Half Start")
    return _reduce_raw(
        state,
        action="Half Start",
        team_id=AWAY_TEAM_ID,
    )


def _finish_period(
    state: SoccerGameState,
    *,
    period_clock_ms: int,
) -> SoccerGameState:
    state = _reduce_raw(
        state,
        action="Half End",
        period_clock_ms=period_clock_ms,
    )
    return _reduce_raw(
        state,
        action="Half End",
        team_id=AWAY_TEAM_ID,
        period_clock_ms=period_clock_ms,
    )


def _start_next_period(
    state: SoccerGameState,
    *,
    period: int,
) -> SoccerGameState:
    state = _reduce_raw(
        state,
        action="Half Start",
        period=period,
    )
    return _reduce_raw(
        state,
        action="Half Start",
        team_id=AWAY_TEAM_ID,
        period=period,
    )


def test_lifecycle_is_explicit_and_native_out_and_off_camera_are_not_guessed() -> None:
    initial = initial_soccer_game_state(
        GAME_ID,
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
    )
    assert initial.lifecycle == "not_started"
    assert initial.in_play is False
    assert initial.terminal is False

    state = _started_state()
    assert state.lifecycle == "in_play"
    assert state.in_play is True

    out_event = _domain_event(
        _raw_event(
            index=state.sequence + 1,
            action="Clearance",
            period_clock_ms=1_000,
            native_out=True,
        )
    )
    after_out = reduce_soccer_game_state(state, out_event)
    assert out_event.native_out is True
    assert after_out.lifecycle == "paused"
    assert after_out.last_event_native_out is True

    off_camera_pass = _domain_event(
        _raw_event(
            index=after_out.sequence + 1,
            action="Pass",
            period_clock_ms=2_000,
            possession_id=2,
            off_camera=True,
        )
    )
    after_pass = reduce_soccer_game_state(after_out, off_camera_pass)
    assert off_camera_pass.off_camera is True
    assert after_pass.lifecycle == "in_play"
    assert after_pass.last_event_off_camera is True
    assert after_pass.last_event_native_out is False


def test_active_play_after_half_end_fails_closed_but_admin_row_remains_paused() -> None:
    ended = _finish_period(
        _started_state(),
        period_clock_ms=2_700_000,
    )
    assert ended.lifecycle == "paused"
    assert ended.period_complete is True
    assert ended.terminal is False

    administrative = _reduce_raw(
        ended,
        action="Bad Behaviour",
        period_clock_ms=2_701_000,
        off_camera=True,
    )
    assert administrative.lifecycle == "paused"
    assert administrative.period_complete is True

    pressure = _domain_event(
        _raw_event(
            index=administrative.sequence + 1,
            action="Pressure",
            period_clock_ms=2_702_000,
        )
    )
    with pytest.raises(
        SoccerGameStateError,
        match="completed period",
    ) as captured:
        reduce_soccer_game_state(administrative, pressure)
    assert captured.value.category == "post_period_end_active_event"

    second_half = _start_next_period(administrative, period=2)
    assert second_half.period == 2
    assert second_half.lifecycle == "in_play"
    assert second_half.period_complete is False


def test_clock_regression_cannot_be_authorized_by_quality_flags() -> None:
    state = _reduce_raw(
        _started_state(),
        action="Pass",
        period_clock_ms=10_000,
        possession_id=2,
    )
    regressed = _domain_event(
        _raw_event(
            index=state.sequence + 1,
            action="Pass",
            period_clock_ms=9_000,
            possession_id=2,
        )
    )

    with pytest.raises(SoccerGameStateError, match="clock regression") as captured:
        reduce_soccer_game_state(state, regressed)
    assert captured.value.category == "clock_regression"

    with pytest.raises(SoccerGameStateError, match="quality flag"):
        statsbomb_event_payload(
            _raw_event(
                index=state.sequence + 1,
                action="Pass",
                period_clock_ms=9_000,
                possession_id=2,
            ),
            game_id=GAME_ID,
            quality_flags=("clock_jump", "out_of_order"),
        )


def test_possession_id_and_team_transition_fail_closed() -> None:
    state = _reduce_raw(
        _started_state(),
        action="Pass",
        period_clock_ms=1_000,
        possession_id=2,
    )
    regressed = _domain_event(
        _raw_event(
            index=state.sequence + 1,
            action="Pass",
            period_clock_ms=2_000,
            possession_id=1,
        )
    )
    with pytest.raises(
        SoccerGameStateError,
        match="possession.*regress",
    ) as captured:
        reduce_soccer_game_state(state, regressed)
    assert captured.value.category == "possession_regression"

    changed_team = _domain_event(
        _raw_event(
            index=state.sequence + 1,
            action="Pass",
            team_id=AWAY_TEAM_ID,
            period_clock_ms=2_000,
            possession_id=2,
            possession_team_id=AWAY_TEAM_ID,
        )
    )
    with pytest.raises(
        SoccerGameStateError,
        match="same possession",
    ) as captured:
        reduce_soccer_game_state(state, changed_team)
    assert captured.value.category == "possession_team_mismatch"


def test_extra_time_and_shootout_have_explicit_period_and_score_semantics() -> None:
    state = _finish_period(_started_state(), period_clock_ms=2_700_000)
    state = _start_next_period(state, period=2)
    state = _finish_period(state, period_clock_ms=2_700_000)
    state = _start_next_period(state, period=3)
    state = _finish_period(state, period_clock_ms=900_000)
    state = _start_next_period(state, period=4)
    state = _finish_period(state, period_clock_ms=900_000)
    state = _start_next_period(state, period=5)

    home_goal = _reduce_raw(
        state,
        action="Shot",
        period=5,
        period_clock_ms=1_000,
        possession_id=2,
        player_id=1_001,
        shot_outcome="Goal",
        shot_type="Penalty",
    )
    away_miss = _reduce_raw(
        home_goal,
        action="Shot",
        team_id=AWAY_TEAM_ID,
        period=5,
        period_clock_ms=2_000,
        possession_id=3,
        possession_team_id=AWAY_TEAM_ID,
        player_id=2_001,
        shot_outcome="Off T",
        shot_type="Penalty",
    )

    assert (away_miss.home_score, away_miss.away_score) == (0, 0)
    assert (
        away_miss.home_shootout_score,
        away_miss.away_shootout_score,
    ) == (1, 0)
    assert (
        away_miss.home_shootout_attempts,
        away_miss.away_shootout_attempts,
    ) == (1, 1)
    assert away_miss.lifecycle == "paused"

    ended = _finish_period(away_miss, period_clock_ms=3_000)
    finished = _reduce_raw(
        ended,
        action="Match End",
        period=5,
        period_clock_ms=3_000,
    )
    assert finished.lifecycle == "finished"
    assert finished.terminal is True
    assert finished.terminal_reason == "match_end"


def test_half_end_never_infers_finished_and_period_skip_fails_closed() -> None:
    first_half = _finish_period(
        _started_state(),
        period_clock_ms=2_700_000,
    )
    second_half = _start_next_period(first_half, period=2)
    second_half = _finish_period(
        second_half,
        period_clock_ms=2_700_000,
    )
    assert second_half.lifecycle == "paused"
    assert second_half.terminal is False

    invalid_fourth_period = _domain_event(
        _raw_event(
            index=second_half.sequence + 1,
            action="Half Start",
            period=4,
        )
    )
    with pytest.raises(
        SoccerGameStateError,
        match="period transition",
    ) as captured:
        reduce_soccer_game_state(second_half, invalid_fourth_period)
    assert captured.value.category == "period_transition"


def test_unknown_native_action_and_non_penalty_shootout_fail_closed() -> None:
    with pytest.raises(
        SoccerGameStateError,
        match="unsupported StatsBomb action",
    ) as captured:
        statsbomb_event_payload(
            _raw_event(index=1, action="Unmapped Native Row"),
            game_id=GAME_ID,
        )
    assert captured.value.category == "unsupported_action"

    state = _finish_period(_started_state(), period_clock_ms=2_700_000)
    state = _start_next_period(state, period=2)
    state = _finish_period(state, period_clock_ms=2_700_000)
    state = _start_next_period(state, period=5)
    open_play_shot = _domain_event(
        _raw_event(
            index=state.sequence + 1,
            action="Shot",
            period=5,
            period_clock_ms=1_000,
            possession_id=2,
            player_id=1_001,
            shot_outcome="Goal",
            shot_type="Open Play",
        )
    )
    with pytest.raises(
        SoccerGameStateError,
        match="shootout",
    ) as captured:
        reduce_soccer_game_state(state, open_play_shot)
    assert captured.value.category == "unsupported_shootout_semantics"
