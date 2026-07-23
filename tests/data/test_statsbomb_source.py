from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from prediction_market.sports.statsbomb import (
    STATSBOMB_COMMIT,
    StatsBombSourceError,
    fetch_and_preserve_statsbomb_event,
    fetch_and_preserve_statsbomb_match_index,
    inspect_statsbomb_event,
    inspect_statsbomb_match_index,
)
from prediction_market.static_store import read_verified_static_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCHED_AT = datetime(2026, 7, 22, 13, 0, tzinfo=timezone.utc)


def _matches_bytes() -> bytes:
    matches = []
    for index in range(380):
        matches.append(
            {
                "match_id": 100000 + index,
                "match_date": f"2015-{8 + (index // 124):02d}-{1 + index % 28:02d}",
                "kick_off": "15:00:00.000",
                "competition": {
                    "competition_id": 2,
                    "competition_name": "Premier League",
                },
                "season": {"season_id": 27, "season_name": "2015/2016"},
                "home_team": {
                    "home_team_id": index * 2 + 1,
                    "home_team_name": f"Home {index}",
                },
                "away_team": {
                    "away_team_id": index * 2 + 2,
                    "away_team_name": f"Away {index}",
                },
                "home_score": index % 4,
                "away_score": (index + 1) % 3,
                "match_week": 1 + index // 10,
                "last_updated": "2020-07-29T05:00:00.000000",
            }
        )
    return json.dumps(matches, sort_keys=True, separators=(",", ":")).encode()


def _events_bytes(match_id: int = 100000) -> bytes:
    del match_id
    return json.dumps(
        [
            {
                "id": "event-1",
                "index": 1,
                "period": 1,
                "timestamp": "00:00:00.000",
                "minute": 0,
                "second": 0,
                "type": {"id": 35, "name": "Starting XI"},
                "possession": 1,
                "team": {"id": 1, "name": "Home 0"},
            },
            {
                "id": "event-2",
                "index": 2,
                "period": 1,
                "timestamp": "00:00:01.000",
                "minute": 0,
                "second": 1,
                "type": {"id": 30, "name": "Pass"},
                "possession": 1,
                "team": {"id": 1, "name": "Home 0"},
                "location": [50.0, 40.0],
            },
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _client(matches: bytes, events: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        if request.url.path.endswith("/data/matches/2/27.json"):
            payload = matches
        elif request.url.path.endswith("/data/events/100000.json"):
            payload = events
        else:
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            content=payload,
            headers={
                "ETag": '"statsbomb-fixture"',
                "Content-Length": str(len(payload)),
                "Content-Type": "application/json; charset=utf-8",
            },
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_statsbomb_match_index_locks_exact_competition_season_and_order() -> None:
    audit = inspect_statsbomb_match_index(_matches_bytes())

    assert audit.match_count == 380
    assert audit.match_ids[0] == 100000
    assert set(audit.match_ids) == set(audit.chronological_match_ids)
    assert audit.first_match_date == "2015-08-01"
    assert audit.schema_fingerprint.startswith("sha256:")
    assert audit.object_sha256.startswith("sha256:")


def test_statsbomb_match_index_is_preserved_with_research_only_manifest(
    tmp_path: Path,
) -> None:
    matches = _matches_bytes()
    with _client(matches, _events_bytes()) as client:
        preserved = fetch_and_preserve_statsbomb_match_index(
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert preserved.record.version == STATSBOMB_COMMIT
    assert preserved.record.partition == "matches-2-27"
    assert preserved.record.manifest.dataset_id == "DS-STATSBOMB-OPEN"
    assert preserved.record.manifest.license_ref == "O-004"
    assert preserved.record.manifest.license_status == "research_only"
    verified = read_verified_static_object(
        preserved.record.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert verified.object_bytes == matches


def test_statsbomb_event_requires_index_membership_and_preserves_exact_json(
    tmp_path: Path,
) -> None:
    matches = _matches_bytes()
    events = _events_bytes()
    with _client(matches, events) as client:
        index = fetch_and_preserve_statsbomb_match_index(
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )
        result = fetch_and_preserve_statsbomb_event(
            100000,
            match_index=index,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert result.audit.match_id == 100000
    assert result.audit.event_count == 2
    assert result.record.partition == "events-100000"
    assert result.record.manifest.object_sha256 == result.audit.object_sha256
    verified = read_verified_static_object(
        result.record.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert verified.object_bytes == events

    with _client(matches, events) as client:
        with pytest.raises(StatsBombSourceError, match="not present.*index"):
            fetch_and_preserve_statsbomb_event(
                999999,
                match_index=index,
                store_root=tmp_path,
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
            )


def test_statsbomb_rejects_wrong_season_duplicate_match_or_event() -> None:
    matches = json.loads(_matches_bytes())
    matches[0]["season"]["season_id"] = 99
    with pytest.raises(StatsBombSourceError, match="competition 2 season 27"):
        inspect_statsbomb_match_index(json.dumps(matches).encode())

    matches = json.loads(_matches_bytes())
    matches[1]["match_id"] = matches[0]["match_id"]
    with pytest.raises(StatsBombSourceError, match="duplicate match_id"):
        inspect_statsbomb_match_index(json.dumps(matches).encode())

    events = json.loads(_events_bytes())
    events[1]["id"] = events[0]["id"]
    with pytest.raises(StatsBombSourceError, match="duplicate event id"):
        inspect_statsbomb_event(json.dumps(events).encode(), match_id=100000)
