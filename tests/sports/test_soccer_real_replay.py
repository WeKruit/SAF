from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_envelopes,
)
from prediction_market.sports.soccer_game_state import (
    SoccerGameEvent,
    SoccerGameState,
    adapt_statsbomb_event,
    initial_soccer_game_state,
    reduce_soccer_game_state,
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


def _external_event_id(
    raw_event: dict[str, Any],
    *,
    raw_object_sha256: str,
) -> str:
    """Model the identity supplied by the EventEnvelope boundary."""

    identity = (
        f"statsbomb:{raw_object_sha256}:"
        f"{raw_event['index']}:{raw_event['id']}"
    )
    return "evt_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _adapt(
    raw_event: dict[str, Any],
) -> SoccerGameEvent:
    return adapt_statsbomb_event(
        raw_event,
        game_id=GAME_ID,
        event_id=_external_event_id(
            raw_event,
            raw_object_sha256=EVENT_OBJECT_SHA256,
        ),
    )


def _replay(
    raw_events: list[dict[str, Any]],
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
    for raw_event in raw_events:
        event = _adapt(raw_event)
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

    first, first_clock_regressions = _replay(
        raw_events,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )
    second, second_clock_regressions = _replay(
        raw_events,
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
    assert first_clock_regressions == second_clock_regressions == 1


def test_statsbomb_adapter_requires_external_event_identity() -> None:
    raw_events = _load_frozen_json(
        _raw_object(f"events-{MATCH_ID}", EVENT_OBJECT_SHA256),
        EVENT_OBJECT_SHA256,
    )
    assert isinstance(raw_events, list)
    first = raw_events[0]
    pair = build_static_sport_observation_envelopes(
        program_root=PROJECT_ROOT,
        experiment_id="X-12",
        dataset_id="DS-STATSBOMB-OPEN",
        source_system="statsbomb",
        source_stream="events",
        raw_object_hash=f"sha256:{EVENT_OBJECT_SHA256}",
        raw_record_ordinal=0,
        partition=f"events-{MATCH_ID}",
        fetched_at="2026-07-23T03:49:10.793545Z",
        source_at=None,
        competition_id="cmp_statsbomb_2",
        game_id=GAME_ID,
        participant_ids=(
            "participant_statsbomb_33",
            "participant_statsbomb_40",
        ),
        native_namespace="statsbomb.event",
        native_id=first["id"],
        normalized_payload={
            "sport": "soccer",
            "index": first["index"],
            "action": first["type"]["name"],
        },
    )
    event = adapt_statsbomb_event(
        first,
        game_id=GAME_ID,
        event_id=pair.normalized.event_id,
    )
    assert event.event_id == pair.normalized.event_id

    with pytest.raises(TypeError, match="event_id"):
        adapt_statsbomb_event(  # type: ignore[call-arg]
            raw_events[0],
            game_id=GAME_ID,
        )
