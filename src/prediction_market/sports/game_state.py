"""Common lifecycle for immutable, sport-specific game states.

The common layer owns identity, ordering, hashing, and measurement.  It does
not define a cross-sport state vector; every sport supplies its own immutable
state, event, reducer, features, and predictor.
"""

from __future__ import annotations

import math
import platform
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from prediction_market.contracts import (
    GameStateStepV0,
    canonical_sha256,
    game_state_step_sha256,
)


_EVENT_ID_RE = re.compile(r"evt_[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"game_[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class GameStateError(ValueError):
    """A game-state step cannot be accepted without weakening its invariants."""


@runtime_checkable
class GameState(Protocol):
    """Structural identity shared by all sport-specific state objects."""

    sport: str
    game_id: str
    sequence: int
    terminal: bool


@runtime_checkable
class GameEvent(Protocol):
    """Structural identity shared by all normalized sport events."""

    sport: str
    game_id: str
    sequence: int
    event_id: str


StateT = TypeVar("StateT", bound=GameState)
EventT = TypeVar("EventT", bound=GameEvent)
FeatureT = TypeVar("FeatureT")
PredictionT = TypeVar("PredictionT")


@runtime_checkable
class GameStateReducer(Protocol[StateT, EventT]):
    """One deterministic reducer implementation."""

    sport: str
    reducer_id: str
    reducer_version: str

    def reduce(self, state: StateT, event: EventT) -> StateT:
        """Apply exactly one newly observed event."""


def _canonical_material(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            name: _canonical_material(getattr(value, name))
            for name in type(value).model_fields
        }
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_material(asdict(value))
    if isinstance(value, Enum):
        return _canonical_material(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime is forbidden in canonical game state")
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, float):
        raise ValueError("binary float is forbidden in canonical game state")
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Mapping):
        material: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("canonical game-state mapping keys must be strings")
            material[key] = _canonical_material(child)
        return material
    if isinstance(value, (list, tuple)):
        return [_canonical_material(child) for child in value]
    if isinstance(value, (set, frozenset)):
        children = [_canonical_material(child) for child in value]
        return sorted(children, key=repr)
    raise ValueError(
        f"unsupported canonical game-state value: {type(value).__name__}"
    )


def canonical_state_sha256(value: Any) -> str:
    """Hash a state or event after converting it to canonical contract atoms."""

    return canonical_sha256(_canonical_material(value))


def _identity(value: object, *, label: str) -> tuple[str, str, int]:
    sport = getattr(value, "sport", None)
    game_id = getattr(value, "game_id", None)
    sequence = getattr(value, "sequence", None)
    if type(sport) is not str or not sport or sport != sport.strip():
        raise GameStateError(f"{label} sport must be canonical")
    if type(game_id) is not str or _GAME_ID_RE.fullmatch(game_id) is None:
        raise GameStateError(f"{label} game_id must be canonical")
    if type(sequence) is not int or sequence < 0:
        raise GameStateError(f"{label} sequence must be a non-negative integer")
    return sport, game_id, sequence


def _validate_step_input(
    reducer: GameStateReducer[StateT, EventT],
    state: StateT,
    event: EventT,
) -> tuple[str, str, int]:
    reducer_sport = getattr(reducer, "sport", None)
    reducer_id = getattr(reducer, "reducer_id", None)
    reducer_version = getattr(reducer, "reducer_version", None)
    if type(reducer_sport) is not str or not reducer_sport:
        raise GameStateError("reducer sport must be canonical")
    if type(reducer_id) is not str or not reducer_id:
        raise GameStateError("reducer_id must be canonical")
    if type(reducer_version) is not str or not reducer_version:
        raise GameStateError("reducer_version must be canonical")

    state_sport, state_game, state_sequence = _identity(state, label="state")
    event_sport, event_game, event_sequence = _identity(event, label="event")
    event_id = getattr(event, "event_id", None)
    if type(event_id) is not str or _EVENT_ID_RE.fullmatch(event_id) is None:
        raise GameStateError("event_id must be evt_<lowercase sha256>")
    if type(getattr(state, "terminal", None)) is not bool:
        raise GameStateError("state terminal must be boolean")
    if state.terminal:
        raise GameStateError("terminal state cannot consume another event")
    if reducer_sport != state_sport or event_sport != state_sport:
        raise GameStateError("reducer, state, and event sport must match")
    if event_game != state_game:
        raise GameStateError("state and event game_id must match")
    if event_sequence != state_sequence + 1:
        raise GameStateError("event sequence must be exactly state sequence + 1")
    return state_sport, state_game, event_sequence


def _validate_step_output(
    next_state: StateT,
    *,
    sport: str,
    game_id: str,
    sequence: int,
) -> None:
    next_sport, next_game, next_sequence = _identity(
        next_state, label="next state"
    )
    if next_sport != sport:
        raise GameStateError("next state sport does not match the input")
    if next_game != game_id:
        raise GameStateError("next state game_id does not match the input")
    if next_sequence != sequence:
        raise GameStateError("next state sequence does not match the event")
    if type(getattr(next_state, "terminal", None)) is not bool:
        raise GameStateError("next state terminal must be boolean")


@dataclass(frozen=True, slots=True)
class StateTransitionTrace(Generic[StateT]):
    """Content-addressed evidence for one deterministic state transition."""

    reducer_id: str
    reducer_version: str
    sport: str
    game_id: str
    sequence: int
    event_id: str
    previous_state_sha256: str
    event_sha256: str
    next_state_sha256: str
    trace_sha256: str
    next_state: StateT

    def to_contract(
        self,
        *,
        state_schema_id: str,
        event_schema_id: str,
        observation_mode: Literal[
            "live_pit",
            "offline_reconstruction",
            "synthetic_fixture",
        ],
        quality_flags: tuple[str, ...],
    ) -> GameStateStepV0:
        """Bind the trace to explicit sport schemas and observation semantics."""

        material: dict[str, Any] = {
            "step_version": "v0",
            "sport": self.sport,
            "game_id": self.game_id,
            "sequence": self.sequence,
            "terminal": self.next_state.terminal,
            "reducer_id": self.reducer_id,
            "reducer_version": self.reducer_version,
            "state_schema_id": state_schema_id,
            "event_schema_id": event_schema_id,
            "event_id": self.event_id,
            "previous_state_sha256": self.previous_state_sha256,
            "event_sha256": self.event_sha256,
            "next_state_sha256": self.next_state_sha256,
            "observation_mode": observation_mode,
            "quality_flags": list(quality_flags),
            "step_sha256": "sha256:" + "0" * 64,
        }
        material["step_sha256"] = game_state_step_sha256(material)
        return GameStateStepV0.model_validate(material)


def advance_state(
    reducer: GameStateReducer[StateT, EventT],
    state: StateT,
    event: EventT,
) -> StateTransitionTrace[StateT]:
    """Validate and apply one event, returning a deterministic hash chain."""

    sport, game_id, sequence = _validate_step_input(reducer, state, event)
    previous_state_sha256 = canonical_state_sha256(state)
    event_sha256 = canonical_state_sha256(event)
    next_state = reducer.reduce(state, event)
    _validate_step_output(
        next_state,
        sport=sport,
        game_id=game_id,
        sequence=sequence,
    )
    next_state_sha256 = canonical_state_sha256(next_state)
    event_id = event.event_id
    trace_material = {
        "reducer_id": reducer.reducer_id,
        "reducer_version": reducer.reducer_version,
        "sport": sport,
        "game_id": game_id,
        "sequence": sequence,
        "event_id": event_id,
        "previous_state_sha256": previous_state_sha256,
        "event_sha256": event_sha256,
        "next_state_sha256": next_state_sha256,
    }
    return StateTransitionTrace(
        **trace_material,
        trace_sha256=canonical_sha256(trace_material),
        next_state=next_state,
    )


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    samples: int
    p50_ns: int
    p95_ns: int
    p99_ns: int
    max_ns: int
    mean_ns: int
    operations_per_second: int


@dataclass(frozen=True, slots=True)
class StatePipelineLatencyReport:
    benchmark_version: str
    sport: str
    reducer_id: str
    reducer_version: str
    clock: str
    python_version: str
    platform: str
    processor: str
    batch_size: int
    warmups: int
    repetitions: int
    events_per_repetition: int
    measured_events: int
    code_sha256: str | None
    data_sha256: str | None
    config_sha256: str | None
    formal_evidence_eligible: bool
    stages: Mapping[str, LatencyDistribution]


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def _distribution(values: Sequence[int]) -> LatencyDistribution:
    if not values:
        raise GameStateError("latency distribution requires samples")
    ordered = sorted(values)
    total = sum(ordered)
    mean = max(1, total // len(ordered))
    return LatencyDistribution(
        samples=len(ordered),
        p50_ns=_percentile(ordered, 0.50),
        p95_ns=_percentile(ordered, 0.95),
        p99_ns=_percentile(ordered, 0.99),
        max_ns=ordered[-1],
        mean_ns=mean,
        operations_per_second=max(1, 1_000_000_000 // mean),
    )


def _validate_optional_hash(value: str | None, label: str) -> None:
    if value is not None and _SHA256_RE.fullmatch(value) is None:
        raise GameStateError(f"{label} must be lowercase sha256")


def benchmark_state_pipeline(
    *,
    reducer: GameStateReducer[StateT, EventT],
    initial_state: StateT,
    events: Sequence[EventT],
    feature_extractor: Callable[[StateT], FeatureT],
    predictor: Callable[[FeatureT], PredictionT],
    warmups: int,
    repetitions: int,
    code_sha256: str | None = None,
    data_sha256: str | None = None,
    config_sha256: str | None = None,
) -> StatePipelineLatencyReport:
    """Measure batch-one reducer, feature, inference, and complete path latency."""

    if type(warmups) is not int or warmups < 0:
        raise GameStateError("warmups must be a non-negative integer")
    if type(repetitions) is not int or repetitions <= 0:
        raise GameStateError("repetitions must be a positive integer")
    if not events:
        raise GameStateError("events must not be empty")
    _validate_optional_hash(code_sha256, "code_sha256")
    _validate_optional_hash(data_sha256, "data_sha256")
    _validate_optional_hash(config_sha256, "config_sha256")

    reducer_times: list[int] = []
    feature_times: list[int] = []
    prediction_times: list[int] = []
    end_to_end_times: list[int] = []

    def run_once(*, record: bool) -> None:
        state = initial_state
        for event in events:
            end_to_end_started = time.perf_counter_ns()
            sport, game_id, sequence = _validate_step_input(reducer, state, event)

            reducer_started = time.perf_counter_ns()
            next_state = reducer.reduce(state, event)
            reducer_elapsed = time.perf_counter_ns() - reducer_started
            _validate_step_output(
                next_state,
                sport=sport,
                game_id=game_id,
                sequence=sequence,
            )

            feature_started = time.perf_counter_ns()
            features = feature_extractor(next_state)
            feature_elapsed = time.perf_counter_ns() - feature_started

            prediction_started = time.perf_counter_ns()
            predictor(features)
            prediction_elapsed = time.perf_counter_ns() - prediction_started
            end_to_end_elapsed = time.perf_counter_ns() - end_to_end_started

            if record:
                reducer_times.append(reducer_elapsed)
                feature_times.append(feature_elapsed)
                prediction_times.append(prediction_elapsed)
                end_to_end_times.append(end_to_end_elapsed)
            state = next_state

    for _ in range(warmups):
        run_once(record=False)
    for _ in range(repetitions):
        run_once(record=True)

    formal_eligible = all(
        value is not None for value in (code_sha256, data_sha256, config_sha256)
    )
    return StatePipelineLatencyReport(
        benchmark_version="v0",
        sport=reducer.sport,
        reducer_id=reducer.reducer_id,
        reducer_version=reducer.reducer_version,
        clock="perf_counter_ns",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.processor() or "unknown",
        batch_size=1,
        warmups=warmups,
        repetitions=repetitions,
        events_per_repetition=len(events),
        measured_events=len(events) * repetitions,
        code_sha256=code_sha256,
        data_sha256=data_sha256,
        config_sha256=config_sha256,
        formal_evidence_eligible=formal_eligible,
        stages=MappingProxyType(
            {
                "reducer": _distribution(reducer_times),
                "feature_extraction": _distribution(feature_times),
                "model_inference": _distribution(prediction_times),
                "end_to_end": _distribution(end_to_end_times),
            }
        ),
    )


__all__ = [
    "GameEvent",
    "GameState",
    "GameStateError",
    "GameStateReducer",
    "LatencyDistribution",
    "StatePipelineLatencyReport",
    "StateTransitionTrace",
    "advance_state",
    "benchmark_state_pipeline",
    "canonical_state_sha256",
]
