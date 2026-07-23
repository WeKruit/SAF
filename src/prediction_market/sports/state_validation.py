"""Deterministic replay and field-level validation for game-state reducers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from prediction_market.contracts import canonical_sha256
from prediction_market.sports.game_state import (
    EventT,
    GameState,
    GameStateError,
    GameStateReducer,
    StateT,
    advance_state,
)


@dataclass(frozen=True, slots=True)
class StateReplayValidation:
    validation_version: str
    sport: str
    reducer_id: str
    reducer_version: str
    events: int
    expected_states: int
    exact_matches: int | None
    exact_match_ppm: int | None
    field_mismatches: Mapping[str, int]
    first_mismatch_sequence: int | None
    first_replay_sha256: str
    second_replay_sha256: str
    deterministic: bool
    final_state_sha256: str


def _state_material(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        material = value.model_dump(mode="python")
    elif is_dataclass(value) and not isinstance(value, type):
        material = asdict(value)
    else:
        raise GameStateError(
            "state validation requires a dataclass or Pydantic state"
        )
    if type(material) is not dict or any(type(key) is not str for key in material):
        raise GameStateError("state validation requires canonical string fields")
    return material


def _replay(
    *,
    reducer: GameStateReducer[StateT, EventT],
    initial_state: StateT,
    events: Sequence[EventT],
) -> tuple[list[StateT], str, str]:
    states: list[StateT] = []
    trace_hashes: list[str] = []
    state = initial_state
    for event in events:
        trace = advance_state(reducer, state, event)
        states.append(trace.next_state)
        trace_hashes.append(trace.trace_sha256)
        state = trace.next_state
    return (
        states,
        canonical_sha256({"trace_sha256": trace_hashes}),
        canonical_sha256(_state_material(state)),
    )


def validate_state_replay(
    *,
    reducer: GameStateReducer[StateT, EventT],
    initial_state: StateT,
    events: Sequence[EventT],
    expected_states: Sequence[StateT] | None = None,
) -> StateReplayValidation:
    """Replay twice and, when supplied, compare every expected state field."""

    if not events:
        raise GameStateError("events must not be empty")
    if expected_states is not None and len(expected_states) != len(events):
        raise GameStateError(
            "expected_states must contain exactly one state per event"
        )

    first_states, first_hash, final_hash = _replay(
        reducer=reducer,
        initial_state=initial_state,
        events=events,
    )
    second_states, second_hash, second_final_hash = _replay(
        reducer=reducer,
        initial_state=initial_state,
        events=events,
    )
    if first_hash != second_hash or final_hash != second_final_hash:
        raise GameStateError("state replay is not deterministic")
    if first_states != second_states:
        raise GameStateError("state replay objects differ despite equal inputs")

    exact_matches: int | None = None
    exact_match_ppm: int | None = None
    mismatch_counts: Counter[str] = Counter()
    first_mismatch_sequence: int | None = None

    if expected_states is not None:
        exact_matches = 0
        for actual, expected in zip(
            first_states,
            expected_states,
            strict=True,
        ):
            actual_material = _state_material(actual)
            expected_material = _state_material(expected)
            fields = sorted(set(actual_material) | set(expected_material))
            mismatched = [
                field
                for field in fields
                if actual_material.get(field) != expected_material.get(field)
            ]
            if not mismatched:
                exact_matches += 1
            else:
                mismatch_counts.update(mismatched)
                if first_mismatch_sequence is None:
                    first_mismatch_sequence = actual.sequence
        exact_match_ppm = exact_matches * 1_000_000 // len(expected_states)

    return StateReplayValidation(
        validation_version="v0",
        sport=reducer.sport,
        reducer_id=reducer.reducer_id,
        reducer_version=reducer.reducer_version,
        events=len(events),
        expected_states=0 if expected_states is None else len(expected_states),
        exact_matches=exact_matches,
        exact_match_ppm=exact_match_ppm,
        field_mismatches=MappingProxyType(dict(sorted(mismatch_counts.items()))),
        first_mismatch_sequence=first_mismatch_sequence,
        first_replay_sha256=first_hash,
        second_replay_sha256=second_hash,
        deterministic=True,
        final_state_sha256=final_hash,
    )


__all__ = ["StateReplayValidation", "validate_state_replay"]
