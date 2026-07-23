from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from prediction_market.sports.game_state import (
    GameStateError,
    advance_state,
    benchmark_state_pipeline,
    canonical_state_sha256,
)


EVENT_ID = "evt_" + "a" * 64


@dataclass(frozen=True, slots=True)
class _State:
    sport: Literal["test"]
    game_id: str
    sequence: int
    terminal: bool
    score: int


@dataclass(frozen=True, slots=True)
class _Event:
    sport: Literal["test"]
    game_id: str
    sequence: int
    event_id: str
    points: int
    terminal: bool = False


class _Reducer:
    sport = "test"
    reducer_id = "REDUCER-TEST"
    reducer_version = "v1"

    def reduce(self, state: _State, event: _Event) -> _State:
        return _State(
            sport="test",
            game_id=state.game_id,
            sequence=event.sequence,
            terminal=event.terminal,
            score=state.score + event.points,
        )


def _state() -> _State:
    return _State(
        sport="test",
        game_id="game_test_001",
        sequence=0,
        terminal=False,
        score=0,
    )


def _event(**changes: object) -> _Event:
    values: dict[str, object] = {
        "sport": "test",
        "game_id": "game_test_001",
        "sequence": 1,
        "event_id": EVENT_ID,
        "points": 3,
        "terminal": False,
    }
    values.update(changes)
    return _Event(**values)  # type: ignore[arg-type]


def test_advance_is_deterministic_and_hashes_the_complete_step() -> None:
    first = advance_state(_Reducer(), _state(), _event())
    second = advance_state(_Reducer(), _state(), _event())

    assert first == second
    assert first.next_state.score == 3
    assert first.next_state.sequence == 1
    assert first.event_id == EVENT_ID
    assert first.previous_state_sha256 == canonical_state_sha256(_state())
    assert first.next_state_sha256 == canonical_state_sha256(first.next_state)
    assert first.trace_sha256.startswith("sha256:")


@pytest.mark.parametrize(
    ("state", "event", "message"),
    [
        (_state(), _event(sport="other"), "sport"),
        (_state(), _event(game_id="game_test_002"), "game"),
        (_state(), _event(sequence=0), "sequence"),
        (_state(), _event(sequence=2), "sequence"),
        (
            _State("test", "game_test_001", 0, True, 0),
            _event(),
            "terminal",
        ),
        (_state(), _event(event_id="not-an-event"), "event_id"),
    ],
)
def test_advance_fails_closed_on_identity_and_order(
    state: _State,
    event: _Event,
    message: str,
) -> None:
    with pytest.raises(GameStateError, match=message):
        advance_state(_Reducer(), state, event)


class _WrongReducer(_Reducer):
    def reduce(self, state: _State, event: _Event) -> _State:
        return _State(
            sport="test",
            game_id="game_test_wrong",
            sequence=event.sequence,
            terminal=False,
            score=state.score,
        )


def test_advance_validates_reducer_output() -> None:
    with pytest.raises(GameStateError, match="game"):
        advance_state(_WrongReducer(), _state(), _event())


def test_canonical_state_hash_rejects_binary_float() -> None:
    @dataclass(frozen=True)
    class Invalid:
        value: float

    with pytest.raises(ValueError, match="float"):
        canonical_state_sha256(Invalid(0.1))


def test_latency_benchmark_separates_all_pipeline_stages() -> None:
    report = benchmark_state_pipeline(
        reducer=_Reducer(),
        initial_state=_state(),
        events=(_event(),),
        feature_extractor=lambda state: (state.score,),
        predictor=lambda features: {"score": features[0]},
        warmups=2,
        repetitions=8,
    )

    assert report.batch_size == 1
    assert report.warmups == 2
    assert report.repetitions == 8
    assert report.measured_events == 8
    assert set(report.stages) == {
        "reducer",
        "feature_extraction",
        "model_inference",
        "end_to_end",
    }
    for distribution in report.stages.values():
        assert 0 <= distribution.p50_ns <= distribution.p95_ns
        assert distribution.p95_ns <= distribution.p99_ns
        assert distribution.p99_ns <= distribution.max_ns
        assert distribution.operations_per_second > 0


def test_latency_benchmark_validates_parameters_and_event_chain() -> None:
    with pytest.raises(GameStateError, match="warmups"):
        benchmark_state_pipeline(
            reducer=_Reducer(),
            initial_state=_state(),
            events=(_event(),),
            feature_extractor=lambda state: state.score,
            predictor=lambda value: value,
            warmups=-1,
            repetitions=1,
        )

    with pytest.raises(GameStateError, match="events"):
        benchmark_state_pipeline(
            reducer=_Reducer(),
            initial_state=_state(),
            events=(),
            feature_extractor=lambda state: state.score,
            predictor=lambda value: value,
            warmups=0,
            repetitions=1,
        )
