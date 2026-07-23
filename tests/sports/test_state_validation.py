from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from prediction_market.sports.game_state import GameStateError
from prediction_market.sports.state_validation import validate_state_replay


@dataclass(frozen=True, slots=True)
class _State:
    sport: Literal["nfl"]
    game_id: str
    sequence: int
    terminal: bool
    score: int
    clock: int


@dataclass(frozen=True, slots=True)
class _Event:
    sport: Literal["nfl"]
    game_id: str
    sequence: int
    event_id: str
    points: int
    clock: int


class _Reducer:
    sport = "nfl"
    reducer_id = "REDUCER-VALIDATION"
    reducer_version = "v1"

    def reduce(self, state: _State, event: _Event) -> _State:
        return _State(
            sport="nfl",
            game_id=state.game_id,
            sequence=event.sequence,
            terminal=False,
            score=state.score + event.points,
            clock=event.clock,
        )


def _state(sequence: int, score: int, clock: int) -> _State:
    return _State(
        sport="nfl",
        game_id="game_validation_001",
        sequence=sequence,
        terminal=False,
        score=score,
        clock=clock,
    )


def _event(sequence: int, points: int, clock: int) -> _Event:
    return _Event(
        sport="nfl",
        game_id="game_validation_001",
        sequence=sequence,
        event_id="evt_" + f"{sequence:064x}",
        points=points,
        clock=clock,
    )


def test_replay_reports_exact_field_agreement_and_deterministic_hash() -> None:
    report = validate_state_replay(
        reducer=_Reducer(),
        initial_state=_state(0, 0, 100),
        events=(_event(1, 3, 90), _event(2, 0, 80)),
        expected_states=(_state(1, 3, 90), _state(2, 3, 80)),
    )

    assert report.events == 2
    assert report.expected_states == 2
    assert report.exact_matches == 2
    assert report.exact_match_ppm == 1_000_000
    assert report.field_mismatches == {}
    assert report.first_mismatch_sequence is None
    assert report.first_replay_sha256 == report.second_replay_sha256
    assert report.deterministic is True


def test_replay_reports_each_mismatched_field_without_hiding_it() -> None:
    report = validate_state_replay(
        reducer=_Reducer(),
        initial_state=_state(0, 0, 100),
        events=(_event(1, 3, 90), _event(2, 0, 80)),
        expected_states=(_state(1, 3, 89), _state(2, 4, 80)),
    )

    assert report.exact_matches == 0
    assert report.exact_match_ppm == 0
    assert report.field_mismatches == {"clock": 1, "score": 1}
    assert report.first_mismatch_sequence == 1


def test_replay_without_expected_states_is_replay_evidence_not_accuracy() -> None:
    report = validate_state_replay(
        reducer=_Reducer(),
        initial_state=_state(0, 0, 100),
        events=(_event(1, 3, 90),),
    )

    assert report.expected_states == 0
    assert report.exact_matches is None
    assert report.exact_match_ppm is None
    assert report.field_mismatches == {}


def test_replay_rejects_empty_or_misaligned_inputs() -> None:
    with pytest.raises(GameStateError, match="events"):
        validate_state_replay(
            reducer=_Reducer(),
            initial_state=_state(0, 0, 100),
            events=(),
        )

    with pytest.raises(GameStateError, match="expected_states"):
        validate_state_replay(
            reducer=_Reducer(),
            initial_state=_state(0, 0, 100),
            events=(_event(1, 3, 90),),
            expected_states=(),
        )
