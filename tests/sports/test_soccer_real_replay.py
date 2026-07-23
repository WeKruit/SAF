from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.experiments import load_experiment_registry
from prediction_market.sports.soccer_game_state import (
    SoccerGameEvent,
    SoccerGameState,
    adapt_statsbomb_event,
    initial_soccer_game_state,
    reduce_soccer_game_state,
    statsbomb_event_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATSBOMB_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
MATCH_ID = 3_754_288
GAME_ID = f"game_statsbomb_{MATCH_ID}"
EVENT_OBJECT_SHA256 = (
    "740ff49347291252a573a8e56e08bf48e997edfe99eeaee5d73b705b226ade9b"
)
MATCH_OBJECT_SHA256 = (
    "e07d6d360b30e0cd17f9aeea0db1502d2a5666d298f1b44b83a2d2f21ba3d21b"
)
REGISTERED_CLOCK_ANOMALY = (
    1_578,
    "d7bf1db5-ab42-4ea6-8149-578725ad1508",
)


@lru_cache(maxsize=1)
def _assert_x12_registered() -> None:
    assert "X-12" in load_experiment_registry(PROJECT_ROOT)


@lru_cache(maxsize=1)
def _cached_registry() -> dict[str, dict[str, Any]]:
    return load_experiment_registry(PROJECT_ROOT)


def _raw_object(
    partition: str,
    object_sha256: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "var"
        / "raw"
        / "raw"
        / "source=statsbomb"
        / "dataset=DS-STATSBOMB-OPEN"
        / f"version={STATSBOMB_VERSION}"
        / f"partition={partition}"
        / f"{object_sha256}.json"
    )


def _load_frozen_json(path: Path, expected_sha256: str) -> Any:
    if not path.is_file():
        pytest.skip(f"frozen StatsBomb raw object is not available: {path}")
    raw_bytes = path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == expected_sha256
    return json.loads(raw_bytes)


def _adapt(
    raw_event: dict[str, Any],
) -> SoccerGameEvent:
    sequence = raw_event["index"]
    native_event_id = raw_event["id"]
    assert type(sequence) is int
    assert type(native_event_id) is str
    quality_flags = (
        ("clock_jump", "out_of_order")
        if (sequence, native_event_id) == REGISTERED_CLOCK_ANOMALY
        else ()
    )
    payload = statsbomb_event_payload(
        raw_event,
        game_id=GAME_ID,
        quality_flags=quality_flags,
    )
    _assert_x12_registered()
    canonical_refs = {
        "competition_id": "cmp_statsbomb_2",
        "game_id": GAME_ID,
        "participant_ids": (
            "participant_statsbomb_33",
            "participant_statsbomb_40",
        ),
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }
    native_refs = (
        {
            "namespace": "statsbomb.event",
            "native_id": native_event_id,
        },
    )
    event_time = {
        "receive_at": "2026-07-23T03:49:10.793545Z",
        "receive_basis": "upstream_exporter",
        "source_at": None,
        "publish_at": None,
        "exchange_at": None,
    }
    raw = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="raw_observation",
        payload_schema_version="v0",
        source={
            "system": "statsbomb",
            "stream": "events",
            "venue": None,
            "sequence": sequence - 1,
            "capture_session_id": f"static:sha256:{EVENT_OBJECT_SHA256}",
            "record_ordinal": sequence - 1,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=native_refs,
        lineage={
            "raw_object_hash": f"sha256:{EVENT_OBJECT_SHA256}",
            "raw_record_ordinal": sequence - 1,
            "parent_event_ids": (),
        },
        experiment_id=None,
        rule_snapshot_ref=None,
        quality_flags=quality_flags,
        payload={
            "dataset_id": "DS-STATSBOMB-OPEN",
            "partition": f"events-{MATCH_ID}",
            "raw_object_hash": f"sha256:{EVENT_OBJECT_SHA256}",
            "raw_record_ordinal": sequence - 1,
        },
    )
    normalized = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": "statsbomb",
            "stream": "events.normalized",
            "venue": None,
            "sequence": sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=native_refs,
        lineage={
            "raw_object_hash": None,
            "raw_record_ordinal": None,
            "parent_event_ids": (raw.event_id,),
        },
        experiment_id="X-12",
        rule_snapshot_ref=None,
        quality_flags=quality_flags,
        payload=payload,
    )
    with patch(
        "prediction_market.experiments.load_experiment_registry",
        return_value=_cached_registry(),
    ):
        return adapt_statsbomb_event(
            normalized,
            program_root=PROJECT_ROOT,
            raw_parents=(raw,),
        )


def _replay(
    events: list[SoccerGameEvent],
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[SoccerGameState, int]:
    state = initial_soccer_game_state(
        GAME_ID,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )
    clock_regressions = 0
    for event in events:
        if (
            event.period == state.period
            and (
                event.clock_ms < state.clock_ms
                or event.period_clock_ms < state.period_clock_ms
            )
        ):
            clock_regressions += 1
        state = reduce_soccer_game_state(state, event)
    return state, clock_regressions


def test_frozen_statsbomb_match_replays_twice_with_identical_state() -> None:
    raw_events = _load_frozen_json(
        _raw_object(f"events-{MATCH_ID}", EVENT_OBJECT_SHA256),
        EVENT_OBJECT_SHA256,
    )
    matches = _load_frozen_json(
        _raw_object("matches-2-27", MATCH_OBJECT_SHA256),
        MATCH_OBJECT_SHA256,
    )
    assert isinstance(raw_events, list)
    assert isinstance(matches, list)
    match = next(row for row in matches if row["match_id"] == MATCH_ID)
    home_team_id = match["home_team"]["home_team_id"]
    away_team_id = match["away_team"]["away_team_id"]

    events = [_adapt(raw_event) for raw_event in raw_events]
    assert _adapt(raw_events[0]) == events[0]

    first, first_clock_regressions = _replay(
        events,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )
    second, second_clock_regressions = _replay(
        events,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )

    assert len(raw_events) == 3_175
    assert first == second
    assert first.state_sha256 == second.state_sha256
    assert first.sequence == len(raw_events)
    assert first.period == 2
    assert (first.home_score, first.away_score) == (
        match["home_score"],
        match["away_score"],
    )
    assert len(first.cards) == 8
    assert len(first.substitutions) == 6
    assert tuple(len(team.player_ids) for team in first.active_players) == (
        11,
        10,
    )
    assert first.in_play is False
    assert first.last_action == "Half End"
    assert first.terminal is False
    assert first.terminal_reason is None
    assert first.quality_flags == ("clock_jump", "out_of_order")
    assert first_clock_regressions == second_clock_regressions == 1


def test_statsbomb_adapter_requires_a_fully_bound_event_envelope() -> None:
    raw_events = _load_frozen_json(
        _raw_object(f"events-{MATCH_ID}", EVENT_OBJECT_SHA256),
        EVENT_OBJECT_SHA256,
    )
    assert isinstance(raw_events, list)
    first = raw_events[0]
    event = _adapt(first)
    assert event.native_event_id == first["id"]

    with pytest.raises(TypeError, match="EventEnvelopeV0"):
        adapt_statsbomb_event(  # type: ignore[call-arg]
            raw_events[0],
            program_root=PROJECT_ROOT,
            raw_parents=(),
        )
