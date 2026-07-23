from __future__ import annotations

import hashlib
import re
from dataclasses import FrozenInstanceError, replace
from importlib.util import find_spec

import pytest

from prediction_market.sports import mlb_game_state as mlb
from prediction_market.sports.game_state import advance_state, canonical_state_sha256


_SAME_PITCHER = object()


def _initial() -> object:
    return mlb.initial_state(
        game_id="game_mlb_001",
        away_team="AWAY",
        home_team="HOME",
        batter_id="A1",
        pitcher_id="HP",
        lineup_slot=1,
    )


def _play(
    state: object,
    *,
    event_id: str,
    play_type: str,
    runner_destinations: tuple[object, ...],
    runs: tuple[str, ...] = (),
    outs: tuple[str, ...] = (),
    next_batter_id: str | None = "A2",
    next_pitcher_id: str | None | object = _SAME_PITCHER,
    next_balls: int | None = 0,
    next_strikes: int | None = 0,
    next_lineup_slot: int | None = 2,
    inning_transition: object | None = None,
    terminal: bool = False,
) -> object:
    assert isinstance(state, mlb.MLBGameState)
    canonical_event_id = (
        event_id
        if re.fullmatch(r"evt_[0-9a-f]{64}", event_id)
        else "evt_" + hashlib.sha256(event_id.encode()).hexdigest()
    )
    return mlb.MLBPlayEvent(
        sport="mlb",
        game_id=state.game_id,
        sequence=state.sequence + 1,
        event_id=canonical_event_id,
        inning=state.inning,
        half=state.half,
        outs_before=state.outs,
        bases_before=state.bases,
        score_before=state.score,
        batting_team=state.batting_team,
        fielding_team=state.fielding_team,
        batter_id=state.batter_id,
        pitcher_id=state.pitcher_id,
        balls_before=state.balls,
        strikes_before=state.strikes,
        lineup_slot_before=state.lineup_slot,
        play_type=play_type,
        runs=runs,
        outs=outs,
        runner_destinations=runner_destinations,
        next_batter_id=next_batter_id,
        next_pitcher_id=(
            state.pitcher_id
            if next_pitcher_id is _SAME_PITCHER
            else next_pitcher_id
        ),
        next_balls=next_balls,
        next_strikes=next_strikes,
        next_lineup_slot=next_lineup_slot,
        inning_transition=inning_transition,
        terminal=terminal,
    )


def test_mlb_game_state_module_exists() -> None:
    assert find_spec("prediction_market.sports.mlb_game_state") is not None


def test_initial_state_is_immutable_hashable_and_complete() -> None:
    state = mlb.initial_state(
        game_id="game_mlb_001",
        away_team="AWAY",
        home_team="HOME",
        batter_id="A1",
        pitcher_id="HP",
        lineup_slot=1,
    )

    assert state == mlb.MLBGameState(
        sport="mlb",
        game_id="game_mlb_001",
        sequence=0,
        inning=1,
        half="top",
        outs=0,
        bases=(None, None, None),
        score=mlb.MLBScore(away=0, home=0),
        away_team="AWAY",
        home_team="HOME",
        batting_team="AWAY",
        fielding_team="HOME",
        batter_id="A1",
        pitcher_id="HP",
        balls=0,
        strikes=0,
        lineup_slot=1,
        terminal=False,
    )
    assert hash(state) == hash(replace(state))
    with pytest.raises(FrozenInstanceError):
        state.outs = 1  # type: ignore[misc]


def test_state_rejects_duplicate_runner_ids() -> None:
    state = mlb.initial_state(
        game_id="game_mlb_001",
        away_team="AWAY",
        home_team="HOME",
        batter_id="A1",
        pitcher_id="HP",
    )

    with pytest.raises(mlb.MLBGameStateError, match="runner"):
        replace(state, bases=("A9", "A9", None))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sport": "baseball"}, "sport"),
        ({"sequence": -1}, "sequence"),
        ({"inning": 0}, "inning"),
        ({"half": "middle"}, "half"),
        ({"outs": 3}, "outs"),
        ({"bases": ("A1", None)}, "bases"),
        ({"home_team": "AWAY"}, "team"),
        ({"batting_team": "HOME"}, "batting"),
        ({"balls": 4}, "balls"),
        ({"strikes": 3}, "strikes"),
        ({"lineup_slot": 10}, "lineup"),
        ({"terminal": 1}, "terminal"),
    ],
)
def test_state_fails_closed_on_invalid_base_out_and_identity_values(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(mlb.MLBGameStateError, match=message):
        replace(_initial(), **changes)


def test_score_and_event_id_fail_closed_on_invalid_values() -> None:
    with pytest.raises(mlb.MLBGameStateError, match="score"):
        mlb.MLBScore(away=-1, home=0)

    state = _initial()
    event = _play(
        state,
        event_id="valid-source-id",
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=1),
        ),
    )
    with pytest.raises(mlb.MLBGameStateError, match="event_id"):
        replace(event, event_id="not-an-envelope-id")


def test_single_places_batter_on_first_without_mutating_prior_state() -> None:
    state = _initial()
    event = _play(
        state,
        event_id="play-1",
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance(runner_id="A1", start_base=0, destination=1),
        ),
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.sequence == 1
    assert reduced.bases == ("A1", None, None)
    assert reduced.score == mlb.MLBScore(away=0, home=0)
    assert reduced.batter_id == "A2"
    assert reduced.pitcher_id == "HP"
    assert reduced.lineup_slot == 2
    assert state.bases == (None, None, None)
    with pytest.raises(FrozenInstanceError):
        event.sequence = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("start_base", "destination"),
    [(-1, 1), (4, 1), (0, -1), (0, 5), (2, 1)],
)
def test_runner_advance_rejects_invalid_or_backward_destinations(
    start_base: int,
    destination: int,
) -> None:
    with pytest.raises(mlb.MLBGameStateError, match="base|destination"):
        mlb.RunnerAdvance(
            runner_id="A1",
            start_base=start_base,
            destination=destination,
        )


def test_event_requires_runs_and_outs_to_match_runner_destinations() -> None:
    state = _initial()

    with pytest.raises(mlb.MLBGameStateError, match="runs"):
        _play(
            state,
            event_id="bad-runs",
            play_type="home_run",
            runner_destinations=(
                mlb.RunnerAdvance("A1", start_base=0, destination=4),
            ),
            runs=(),
        )
    with pytest.raises(mlb.MLBGameStateError, match="outs"):
        _play(
            state,
            event_id="bad-outs",
            play_type="ground_out",
            runner_destinations=(
                mlb.RunnerAdvance("A1", start_base=0, destination=0),
            ),
            outs=(),
        )


def test_walk_forces_occupied_first_base_runner_to_second() -> None:
    state = replace(
        _initial(),
        bases=("A1", None, None),
        batter_id="A2",
        lineup_slot=2,
    )
    event = _play(
        state,
        event_id="walk-1",
        play_type="walk",
        runner_destinations=(
            mlb.RunnerAdvance("A2", start_base=0, destination=1),
            mlb.RunnerAdvance("A1", start_base=1, destination=2),
        ),
        next_batter_id="A3",
        next_lineup_slot=3,
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.bases == ("A2", "A1", None)
    assert reduced.outs == 0


def test_home_run_clears_bases_and_scores_every_explicit_runner() -> None:
    state = replace(
        _initial(),
        bases=("A1", "A2", "A3"),
        batter_id="A4",
        lineup_slot=4,
    )
    event = _play(
        state,
        event_id="home-run-1",
        play_type="home_run",
        runner_destinations=(
            mlb.RunnerAdvance("A3", start_base=3, destination=4),
            mlb.RunnerAdvance("A2", start_base=2, destination=4),
            mlb.RunnerAdvance("A1", start_base=1, destination=4),
            mlb.RunnerAdvance("A4", start_base=0, destination=4),
        ),
        runs=("A3", "A2", "A1", "A4"),
        next_batter_id="A5",
        next_lineup_slot=5,
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.bases == (None, None, None)
    assert reduced.score == mlb.MLBScore(away=4, home=0)


def test_double_play_records_two_outs_and_removes_both_runners() -> None:
    state = replace(
        _initial(),
        bases=("A1", None, None),
        batter_id="A2",
        lineup_slot=2,
    )
    event = _play(
        state,
        event_id="double-play-1",
        play_type="double_play",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=1, destination=0),
            mlb.RunnerAdvance("A2", start_base=0, destination=0),
        ),
        outs=("A1", "A2"),
        next_batter_id="A3",
        next_lineup_slot=3,
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.outs == 2
    assert reduced.bases == (None, None, None)


def test_third_out_transitions_to_the_next_half_inning() -> None:
    state = replace(_initial(), outs=2, bases=(None, "A9", None))
    transition = mlb.InningTransition(
        inning=1,
        half="bottom",
        batting_team="HOME",
        fielding_team="AWAY",
        batter_id="H1",
        pitcher_id="AP",
        lineup_slot=1,
        balls=2,
        strikes=1,
    )
    event = _play(
        state,
        event_id="third-out-1",
        play_type="ground_out",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=0),
        ),
        outs=("A1",),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        inning_transition=transition,
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.inning == 1
    assert reduced.half == "bottom"
    assert reduced.outs == 0
    assert reduced.bases == (None, None, None)
    assert reduced.batting_team == "HOME"
    assert reduced.fielding_team == "AWAY"
    assert reduced.batter_id == "H1"
    assert reduced.pitcher_id == "AP"
    assert reduced.lineup_slot == 1
    assert reduced.balls == 2
    assert reduced.strikes == 1


def test_walkoff_run_produces_a_terminal_state() -> None:
    state = replace(
        _initial(),
        inning=9,
        half="bottom",
        outs=1,
        bases=(None, None, "H3"),
        score=mlb.MLBScore(away=2, home=2),
        batting_team="HOME",
        fielding_team="AWAY",
        batter_id="H4",
        pitcher_id="AP",
        lineup_slot=4,
    )
    event = _play(
        state,
        event_id="walkoff-1",
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("H3", start_base=3, destination=4),
            mlb.RunnerAdvance("H4", start_base=0, destination=1),
        ),
        runs=("H3",),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        terminal=True,
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.sequence == 1
    assert reduced.score == mlb.MLBScore(away=2, home=3)
    assert reduced.bases == ("H4", None, None)
    assert reduced.outs == 1
    assert reduced.terminal is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"game_id": "OTHER-GAME"}, "game"),
        ({"sequence": 0}, "sequence"),
        ({"sequence": 2}, "sequence"),
        ({"inning": 2}, "inning"),
        ({"half": "bottom"}, "half"),
        ({"outs_before": 1}, "outs"),
        ({"bases_before": ("A9", None, None)}, "bases"),
        ({"score_before": mlb.MLBScore(away=1, home=0)}, "score"),
        ({"batting_team": "HOME"}, "batting"),
        ({"fielding_team": "AWAY"}, "fielding"),
        ({"batter_id": "A9"}, "batter"),
        ({"pitcher_id": "HP9"}, "pitcher"),
        ({"balls_before": 1}, "balls"),
        ({"strikes_before": 1}, "strikes"),
        ({"lineup_slot_before": 9}, "lineup"),
    ],
)
def test_reducer_rejects_cross_game_out_of_order_or_mismatched_observations(
    changes: dict[str, object],
    message: str,
) -> None:
    state = _initial()
    event = _play(
        state,
        event_id="play-1",
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=1),
        ),
    )

    with pytest.raises(mlb.MLBGameStateError, match=message):
        mlb.reduce_mlb_state(state, replace(event, **changes))


def test_reducer_rejects_runner_origin_and_destination_conflicts() -> None:
    state = replace(
        _initial(),
        bases=("A1", None, None),
        batter_id="A2",
        lineup_slot=2,
    )

    wrong_origin = _play(
        state,
        event_id="wrong-origin",
        play_type="walk",
        runner_destinations=(
            mlb.RunnerAdvance("A9", start_base=1, destination=2),
            mlb.RunnerAdvance("A2", start_base=0, destination=1),
        ),
        next_batter_id="A3",
        next_lineup_slot=3,
    )
    with pytest.raises(mlb.MLBGameStateError, match="runner|origin"):
        mlb.reduce_mlb_state(state, wrong_origin)

    occupied_destination = _play(
        state,
        event_id="occupied-destination",
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("A2", start_base=0, destination=1),
        ),
        next_batter_id="A3",
        next_lineup_slot=3,
    )
    with pytest.raises(mlb.MLBGameStateError, match="destination|occupied"):
        mlb.reduce_mlb_state(state, occupied_destination)

    duplicate_destination = _play(
        state,
        event_id="duplicate-destination",
        play_type="fielders_choice",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=1, destination=2),
            mlb.RunnerAdvance("A2", start_base=0, destination=2),
        ),
        next_batter_id="A3",
        next_lineup_slot=3,
    )
    with pytest.raises(mlb.MLBGameStateError, match="destination"):
        mlb.reduce_mlb_state(state, duplicate_destination)


def test_event_rejects_duplicate_runner_ids() -> None:
    state = replace(_initial(), bases=("A9", None, None))

    with pytest.raises(mlb.MLBGameStateError, match="unique runner"):
        _play(
            state,
            event_id="duplicate-runner",
            play_type="invalid",
            runner_destinations=(
                mlb.RunnerAdvance("A9", start_base=1, destination=2),
                mlb.RunnerAdvance("A9", start_base=2, destination=3),
            ),
        )


def test_reducer_rejects_illegal_out_totals_and_missing_or_early_transition() -> None:
    two_out_state = replace(
        _initial(),
        outs=2,
        bases=("A9", None, None),
    )
    too_many_outs = _play(
        two_out_state,
        event_id="fourth-out",
        play_type="double_play",
        runner_destinations=(
            mlb.RunnerAdvance("A9", start_base=1, destination=0),
            mlb.RunnerAdvance("A1", start_base=0, destination=0),
        ),
        outs=("A9", "A1"),
        next_batter_id="A2",
        next_lineup_slot=2,
    )
    with pytest.raises(mlb.MLBGameStateError, match="outs"):
        mlb.reduce_mlb_state(two_out_state, too_many_outs)

    third_out_without_transition = _play(
        two_out_state,
        event_id="missing-transition",
        play_type="ground_out",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=0),
        ),
        outs=("A1",),
        next_batter_id="A2",
        next_lineup_slot=2,
    )
    with pytest.raises(mlb.MLBGameStateError, match="transition"):
        mlb.reduce_mlb_state(two_out_state, third_out_without_transition)

    one_out_state = replace(_initial(), outs=1)
    early_transition = _play(
        one_out_state,
        event_id="early-transition",
        play_type="ground_out",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=0),
        ),
        outs=("A1",),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        inning_transition=mlb.InningTransition(
            inning=1,
            half="bottom",
            batting_team="HOME",
            fielding_team="AWAY",
            batter_id="H1",
            pitcher_id="AP",
            lineup_slot=1,
            balls=0,
            strikes=0,
        ),
    )
    with pytest.raises(mlb.MLBGameStateError, match="third out|transition"):
        mlb.reduce_mlb_state(one_out_state, early_transition)


def test_bottom_half_third_out_advances_to_the_next_inning_top() -> None:
    state = replace(
        _initial(),
        inning=4,
        half="bottom",
        outs=2,
        batting_team="HOME",
        fielding_team="AWAY",
        batter_id="H9",
        pitcher_id="AP",
        lineup_slot=9,
    )
    event = _play(
        state,
        event_id="end-fourth",
        play_type="fly_out",
        runner_destinations=(
            mlb.RunnerAdvance("H9", start_base=0, destination=0),
        ),
        outs=("H9",),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        inning_transition=mlb.InningTransition(
            inning=5,
            half="top",
            batting_team="AWAY",
            fielding_team="HOME",
            batter_id="A4",
            pitcher_id="HP2",
            lineup_slot=4,
            balls=0,
            strikes=0,
        ),
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert (reduced.inning, reduced.half) == (5, "top")
    assert (reduced.batting_team, reduced.fielding_team) == ("AWAY", "HOME")
    assert (reduced.batter_id, reduced.pitcher_id) == ("A4", "HP2")


@pytest.mark.parametrize(
    "transition",
    [
        mlb.InningTransition(
            inning=2,
            half="bottom",
            batting_team="HOME",
            fielding_team="AWAY",
            batter_id="H1",
            pitcher_id="AP",
            lineup_slot=1,
            balls=0,
            strikes=0,
        ),
        mlb.InningTransition(
            inning=1,
            half="top",
            batting_team="AWAY",
            fielding_team="HOME",
            batter_id="A2",
            pitcher_id="HP",
            lineup_slot=2,
            balls=0,
            strikes=0,
        ),
        mlb.InningTransition(
            inning=1,
            half="bottom",
            batting_team="AWAY",
            fielding_team="HOME",
            batter_id="H1",
            pitcher_id="AP",
            lineup_slot=1,
            balls=0,
            strikes=0,
        ),
    ],
)
def test_reducer_rejects_skipped_or_inconsistent_inning_transition(
    transition: object,
) -> None:
    state = replace(_initial(), outs=2)
    event = _play(
        state,
        event_id="bad-transition",
        play_type="ground_out",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=0),
        ),
        outs=("A1",),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        inning_transition=transition,
    )

    with pytest.raises(mlb.MLBGameStateError, match="transition"):
        mlb.reduce_mlb_state(state, event)


def test_terminal_state_rejects_all_later_events() -> None:
    state = replace(
        _initial(),
        inning=9,
        half="bottom",
        score=mlb.MLBScore(away=1, home=2),
        batting_team="HOME",
        fielding_team="AWAY",
        terminal=True,
    )
    event = _play(
        state,
        event_id="after-final",
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=1),
        ),
    )

    with pytest.raises(mlb.MLBGameStateError, match="terminal"):
        mlb.reduce_mlb_state(state, event)


@pytest.mark.parametrize(
    ("state_changes", "runner_id", "runs"),
    [
        (
            {
                "inning": 8,
                "half": "bottom",
                "score": mlb.MLBScore(away=0, home=0),
                "batting_team": "HOME",
                "fielding_team": "AWAY",
                "batter_id": "H1",
            },
            "H1",
            ("H1",),
        ),
        (
            {
                "inning": 9,
                "half": "bottom",
                "outs": 2,
                "score": mlb.MLBScore(away=0, home=0),
                "batting_team": "HOME",
                "fielding_team": "AWAY",
                "batter_id": "H1",
            },
            "H1",
            (),
        ),
        (
            {
                "inning": 9,
                "half": "top",
                "outs": 2,
                "score": mlb.MLBScore(away=1, home=0),
            },
            "A1",
            (),
        ),
    ],
)
def test_reducer_rejects_premature_tied_or_wrong_half_terminal_events(
    state_changes: dict[str, object],
    runner_id: str,
    runs: tuple[str, ...],
) -> None:
    state = replace(_initial(), **state_changes)
    destination = 4 if runs else 0
    event = _play(
        state,
        event_id="invalid-final",
        play_type="terminal_play",
        runner_destinations=(
            mlb.RunnerAdvance(runner_id, start_base=0, destination=destination),
        ),
        runs=runs,
        outs=() if runs else (runner_id,),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        terminal=True,
    )

    with pytest.raises(mlb.MLBGameStateError, match="terminal"):
        mlb.reduce_mlb_state(state, event)


def test_home_lead_after_top_ninth_third_out_is_terminal() -> None:
    state = replace(
        _initial(),
        inning=9,
        outs=2,
        score=mlb.MLBScore(away=2, home=3),
    )
    event = _play(
        state,
        event_id="top-nine-final",
        play_type="strikeout",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=0),
        ),
        outs=("A1",),
        next_batter_id=None,
        next_pitcher_id=None,
        next_balls=None,
        next_strikes=None,
        next_lineup_slot=None,
        terminal=True,
    )

    reduced = mlb.reduce_mlb_state(state, event)

    assert reduced.terminal is True
    assert reduced.outs == 3
    assert reduced.score == mlb.MLBScore(away=2, home=3)


def test_reducer_object_integrates_with_common_hash_chain() -> None:
    state = _initial()
    event = _play(
        state,
        event_id="evt_" + "a" * 64,
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=1),
        ),
    )

    assert isinstance(mlb.MLB_GAME_STATE_REDUCER, mlb.MLBGameStateReducer)
    trace = advance_state(mlb.MLB_GAME_STATE_REDUCER, state, event)

    assert trace.sport == "mlb"
    assert trace.game_id == "game_mlb_001"
    assert trace.sequence == 1
    assert trace.next_state.bases == ("A1", None, None)
    assert trace.trace_sha256.startswith("sha256:")


def test_reduction_and_canonical_hash_are_deterministic() -> None:
    state = _initial()
    first_event = _play(
        state,
        event_id="evt_" + "c" * 64,
        play_type="single",
        runner_destinations=(
            mlb.RunnerAdvance("A1", start_base=0, destination=1),
        ),
    )
    second_event = replace(first_event)

    first = mlb.reduce(state, first_event)
    second = mlb.MLB_GAME_STATE_REDUCER.reduce(state, second_event)

    assert first_event == second_event
    assert hash(first_event) == hash(second_event)
    assert first == second
    assert hash(first) == hash(second)
    assert canonical_state_sha256(first) == canonical_state_sha256(second)


def _cwevent_row(
    *,
    game_id: str,
    event_index: int,
    batter_id: str,
    lineup_slot: int,
    first_runner: str = "",
    batter_destination: int = 0,
) -> dict[str, object]:
    return {
        "GAME_ID": game_id,
        "INN_CT": "1",
        "BAT_HOME_ID": "0",
        "OUTS_CT": "0",
        "BALLS_CT": "0",
        "STRIKES_CT": "0",
        "AWAY_SCORE_CT": "0",
        "HOME_SCORE_CT": "0",
        "BAT_ID": batter_id,
        "PIT_ID": "HP",
        "BASE1_RUN_ID": first_runner,
        "BASE2_RUN_ID": "",
        "BASE3_RUN_ID": "",
        "EVENT_TX": "S7/G",
        "BAT_LINEUP_ID": str(lineup_slot),
        "EVENT_CD": "20",
        "BAT_EVENT_FL": "T",
        "EVENT_OUTS_CT": "0",
        "BAT_DEST_ID": str(batter_destination),
        "RUN1_DEST_ID": "1" if first_runner else "0",
        "RUN2_DEST_ID": "0",
        "RUN3_DEST_ID": "0",
        "GAME_END_FL": "F",
        "EVENT_ID": str(event_index),
    }


def test_cwevent_adapter_uses_explicit_v0100_fields_and_records_version() -> None:
    raw_game_id = "ANA202504040"
    state = mlb.initial_state(
        game_id=mlb.retrosheet_game_id(raw_game_id),
        away_team="ANA",
        home_team="TEX",
        batter_id="A1",
        pitcher_id="HP",
    )
    play_row = _cwevent_row(
        game_id=raw_game_id,
        event_index=1,
        batter_id="A1",
        lineup_slot=1,
        batter_destination=1,
    )
    next_row = _cwevent_row(
        game_id=raw_game_id,
        event_index=2,
        batter_id="A2",
        lineup_slot=2,
        first_runner="A1",
    )
    event = mlb.event_from_cwevent_rows(
        state,
        play_row,
        next_row,
        event_id="evt_" + "d" * 64,
        cwevent_version="0.10.0",
    )
    reduced = mlb.reduce(state, event)

    assert mlb.CWEVENT_FIELD_MAP["game_id"] == ("GAME_ID", 0)
    assert mlb.CWEVENT_FIELD_MAP["batter_destination"] == ("BAT_DEST_ID", 58)
    assert mlb.CWEVENT_FIELD_MAP["event_index"] == ("EVENT_ID", 96)
    assert event.source_parser == "chadwick.cwevent"
    assert event.source_parser_version == "0.10.0"
    assert event.source_event_index == 1
    assert reduced.bases == ("A1", None, None)
    assert reduced.batter_id == "A2"


def test_state_from_cwevent_row_preserves_observed_pre_play_count() -> None:
    row = _cwevent_row(
        game_id="ANA202504040",
        event_index=1,
        batter_id="A1",
        lineup_slot=1,
    )
    row["BALLS_CT"] = "2"
    row["STRIKES_CT"] = "1"

    state = mlb.state_from_cwevent_row(
        row,
        away_team="CLE",
        home_team="ANA",
        cwevent_version="0.10.0",
    )

    assert state.game_id == "game_retrosheet_ANA202504040"
    assert state.balls == 2
    assert state.strikes == 1
    assert state.batter_id == "A1"
    assert state.pitcher_id == "HP"


def test_cwevent_adapter_rejects_unverified_parser_version() -> None:
    raw_game_id = "ANA202504040"
    state = mlb.initial_state(
        game_id=mlb.retrosheet_game_id(raw_game_id),
        away_team="ANA",
        home_team="TEX",
        batter_id="A1",
        pitcher_id="HP",
    )
    play_row = _cwevent_row(
        game_id=raw_game_id,
        event_index=1,
        batter_id="A1",
        lineup_slot=1,
        batter_destination=1,
    )
    next_row = _cwevent_row(
        game_id=raw_game_id,
        event_index=2,
        batter_id="A2",
        lineup_slot=2,
        first_runner="A1",
    )

    with pytest.raises(mlb.MLBGameStateError, match="version"):
        mlb.event_from_cwevent_rows(
            state,
            play_row,
            next_row,
            event_id="evt_" + "d" * 64,
            cwevent_version="0.9.0",
        )


def test_cwevent_version_check_fails_closed_when_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlb.shutil, "which", lambda executable: None)

    with pytest.raises(mlb.MLBGameStateError, match="cwevent"):
        mlb.require_cwevent_version()
