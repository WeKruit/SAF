"""Frozen StatsBomb Open Data acquisition for the X-12 research-only POC."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from prediction_market.static_store import StaticObjectRecord, preserve_static_object


STATSBOMB_COMMIT = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
STATSBOMB_COMPETITION_ID = 2
STATSBOMB_SEASON_ID = 27
STATSBOMB_EXPECTED_MATCHES = 380
STATSBOMB_RAW_ROOT = (
    "https://raw.githubusercontent.com/statsbomb/open-data/" + STATSBOMB_COMMIT
)


class StatsBombSourceError(ValueError):
    """A StatsBomb object cannot support the frozen X-12 POC."""


@dataclass(frozen=True, slots=True)
class StatsBombMatchIndexAudit:
    match_count: int
    match_ids: tuple[int, ...]
    chronological_match_ids: tuple[int, ...]
    first_match_date: str
    last_match_date: str
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class PreservedStatsBombMatchIndex:
    record: StaticObjectRecord
    audit: StatsBombMatchIndexAudit


@dataclass(frozen=True, slots=True)
class StatsBombEventAudit:
    match_id: int
    event_count: int
    first_event_index: int
    last_event_index: int
    periods: tuple[int, ...]
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class PreservedStatsBombEvent:
    record: StaticObjectRecord
    audit: StatsBombEventAudit


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise StatsBombSourceError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _strict_json_array(payload: bytes, *, context: str) -> list[Any]:
    if type(payload) is not bytes or not payload:
        raise StatsBombSourceError(f"{context} must contain exact nonempty bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                StatsBombSourceError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, StatsBombSourceError) as exc:
        raise StatsBombSourceError(f"{context} is not strict UTF-8 JSON") from exc
    if type(value) is not list:
        raise StatsBombSourceError(f"{context} must be a JSON array")
    return value


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    raise StatsBombSourceError(f"unsupported JSON value type: {type(value).__name__}")


def _schema_fingerprint(value: Any) -> str:
    observed: dict[str, set[str]] = defaultdict(set)

    def visit(child: Any, path: str) -> None:
        observed[path].add(_json_type(child))
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    material = {path: sorted(types) for path, types in sorted(observed.items())}
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise StatsBombSourceError(f"{field} must be an object")
    return value


def _required_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StatsBombSourceError(f"{field} must be an integer >= {minimum}")
    return value


def _required_text(value: Any, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise StatsBombSourceError(f"{field} must be a nonempty string")
    return value


def inspect_statsbomb_match_index(payload: bytes) -> StatsBombMatchIndexAudit:
    """Validate the fixed Premier League 2015/16 match index."""

    matches = _strict_json_array(payload, context="StatsBomb match index")
    if len(matches) != STATSBOMB_EXPECTED_MATCHES:
        raise StatsBombSourceError(
            f"StatsBomb match index must contain {STATSBOMB_EXPECTED_MATCHES} matches"
        )
    match_ids: list[int] = []
    chronology: list[tuple[date, str, int]] = []
    for position, raw_match in enumerate(matches):
        match = _required_mapping(raw_match, f"match[{position}]")
        match_id = _required_int(match.get("match_id"), "match_id", minimum=1)
        competition = _required_mapping(match.get("competition"), "competition")
        season = _required_mapping(match.get("season"), "season")
        if (
            competition.get("competition_id") != STATSBOMB_COMPETITION_ID
            or season.get("season_id") != STATSBOMB_SEASON_ID
        ):
            raise StatsBombSourceError(
                "StatsBomb matches must belong to competition 2 season 27"
            )
        try:
            match_date = date.fromisoformat(
                _required_text(match.get("match_date"), "match_date")
            )
        except ValueError as exc:
            raise StatsBombSourceError("match_date must be ISO-8601") from exc
        kick_off = _required_text(match.get("kick_off"), "kick_off")
        home = _required_mapping(match.get("home_team"), "home_team")
        away = _required_mapping(match.get("away_team"), "away_team")
        _required_int(home.get("home_team_id"), "home_team_id", minimum=1)
        _required_int(away.get("away_team_id"), "away_team_id", minimum=1)
        _required_text(home.get("home_team_name"), "home_team_name")
        _required_text(away.get("away_team_name"), "away_team_name")
        _required_int(match.get("home_score"), "home_score")
        _required_int(match.get("away_score"), "away_score")
        match_ids.append(match_id)
        chronology.append((match_date, kick_off, match_id))
    if len(set(match_ids)) != len(match_ids):
        raise StatsBombSourceError("StatsBomb match index contains duplicate match_id")
    ordered = tuple(match_id for _, _, match_id in sorted(chronology))
    return StatsBombMatchIndexAudit(
        match_count=len(match_ids),
        match_ids=tuple(match_ids),
        chronological_match_ids=ordered,
        first_match_date=min(item[0] for item in chronology).isoformat(),
        last_match_date=max(item[0] for item in chronology).isoformat(),
        schema_fingerprint=_schema_fingerprint(matches),
        object_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def inspect_statsbomb_event(
    payload: bytes, *, match_id: int
) -> StatsBombEventAudit:
    """Validate one native event array while retaining its exact bytes."""

    _required_int(match_id, "match_id", minimum=1)
    events = _strict_json_array(payload, context="StatsBomb event object")
    if not events:
        raise StatsBombSourceError("StatsBomb event object must not be empty")
    event_ids: list[str] = []
    indices: list[int] = []
    periods: set[int] = set()
    for position, raw_event in enumerate(events):
        event = _required_mapping(raw_event, f"event[{position}]")
        event_ids.append(_required_text(event.get("id"), "event id"))
        indices.append(_required_int(event.get("index"), "event index", minimum=1))
        periods.add(_required_int(event.get("period"), "event period", minimum=1))
        _required_text(event.get("timestamp"), "event timestamp")
        _required_int(event.get("minute"), "event minute")
        _required_int(event.get("second"), "event second")
        event_type = _required_mapping(event.get("type"), "event type")
        team = _required_mapping(event.get("team"), "event team")
        _required_int(event_type.get("id"), "event type id", minimum=1)
        _required_text(event_type.get("name"), "event type name")
        _required_int(team.get("id"), "event team id", minimum=1)
        _required_text(team.get("name"), "event team name")
    if len(set(event_ids)) != len(event_ids):
        raise StatsBombSourceError("StatsBomb event object contains duplicate event id")
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise StatsBombSourceError(
            "StatsBomb event indices must be unique and source ordered"
        )
    return StatsBombEventAudit(
        match_id=match_id,
        event_count=len(events),
        first_event_index=indices[0],
        last_event_index=indices[-1],
        periods=tuple(sorted(periods)),
        schema_fingerprint=_schema_fingerprint(events),
        object_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def statsbomb_match_index_url() -> str:
    return f"{STATSBOMB_RAW_ROOT}/data/matches/2/27.json"


def statsbomb_event_url(match_id: int) -> str:
    _required_int(match_id, "match_id", minimum=1)
    return f"{STATSBOMB_RAW_ROOT}/data/events/{match_id}.json"


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise StatsBombSourceError("fetched_at must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _download_json(
    url: str,
    *,
    client: httpx.Client,
    max_bytes: int,
) -> tuple[bytes, httpx.Headers]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise StatsBombSourceError("max_bytes must be a positive integer")
    try:
        with client.stream(
            "GET",
            url,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding not in {None, "identity"}:
                raise StatsBombSourceError(
                    "StatsBomb source ignored identity content encoding"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as exc:
                    raise StatsBombSourceError("invalid Content-Length") from exc
                if declared_bytes > max_bytes:
                    raise StatsBombSourceError("StatsBomb object exceeds max_bytes")
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise StatsBombSourceError("StatsBomb object exceeds max_bytes")
                chunks.append(chunk)
            if declared is not None and received != declared_bytes:
                raise StatsBombSourceError(
                    "StatsBomb Content-Length does not match received bytes"
                )
            payload = b"".join(chunks)
            headers = httpx.Headers(response.headers)
    except httpx.HTTPError as exc:
        raise StatsBombSourceError(f"StatsBomb download failed: {exc}") from exc
    if not payload:
        raise StatsBombSourceError("StatsBomb download returned empty bytes")
    return payload, headers


def _active_client(client: httpx.Client | None) -> tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return (
        httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=30.0),
        ),
        True,
    )


def fetch_and_preserve_statsbomb_match_index(
    *,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 10_000_000,
) -> PreservedStatsBombMatchIndex:
    """Preserve the fixed match index before any X-12 event reads."""

    fetched_at_text = _utc_text(fetched_at)
    active, owned = _active_client(client)
    try:
        payload, headers = _download_json(
            statsbomb_match_index_url(), client=active, max_bytes=max_bytes
        )
    finally:
        if owned:
            active.close()
    audit = inspect_statsbomb_match_index(payload)
    partition = "matches-2-27"
    record = preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="statsbomb",
        dataset="DS-STATSBOMB-OPEN",
        version=STATSBOMB_COMMIT,
        partition=partition,
        extension="json",
        source_url=statsbomb_match_index_url(),
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        },
        source_cursor=f"commit:{STATSBOMB_COMMIT};path:data/matches/2/27.json",
        fetched_at=fetched_at_text,
        coverage="competition_id=2;season_id=27;Premier League 2015/16",
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/json").split(";", 1)[0],
        schema_fingerprint=audit.schema_fingerprint,
        license_ref="O-004",
        license_status="research_only",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )
    return PreservedStatsBombMatchIndex(record=record, audit=audit)


def fetch_and_preserve_statsbomb_event(
    match_id: int,
    *,
    match_index: PreservedStatsBombMatchIndex,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 20_000_000,
) -> PreservedStatsBombEvent:
    """Preserve one exact event object only when its frozen index lists it."""

    match_id = _required_int(match_id, "match_id", minimum=1)
    if match_id not in set(match_index.audit.match_ids):
        raise StatsBombSourceError(f"match {match_id} is not present in frozen index")
    fetched_at_text = _utc_text(fetched_at)
    active, owned = _active_client(client)
    try:
        payload, headers = _download_json(
            statsbomb_event_url(match_id), client=active, max_bytes=max_bytes
        )
    finally:
        if owned:
            active.close()
    audit = inspect_statsbomb_event(payload, match_id=match_id)
    partition = f"events-{match_id}"
    record = preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="statsbomb",
        dataset="DS-STATSBOMB-OPEN",
        version=STATSBOMB_COMMIT,
        partition=partition,
        extension="json",
        source_url=statsbomb_event_url(match_id),
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            "match_index_manifest_sha256": (
                match_index.record.manifest.manifest_sha256
            ),
        },
        source_cursor=(
            f"commit:{STATSBOMB_COMMIT};path:data/events/{match_id}.json"
        ),
        fetched_at=fetched_at_text,
        coverage=f"match_id={match_id};competition_id=2;season_id=27",
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/json").split(";", 1)[0],
        schema_fingerprint=audit.schema_fingerprint,
        license_ref="O-004",
        license_status="research_only",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )
    if record.manifest.object_sha256 != audit.object_sha256:
        raise StatsBombSourceError("preserved event hash differs from audited bytes")
    return PreservedStatsBombEvent(record=record, audit=audit)


__all__ = [
    "STATSBOMB_COMMIT",
    "STATSBOMB_COMPETITION_ID",
    "STATSBOMB_EXPECTED_MATCHES",
    "STATSBOMB_SEASON_ID",
    "PreservedStatsBombEvent",
    "PreservedStatsBombMatchIndex",
    "StatsBombEventAudit",
    "StatsBombMatchIndexAudit",
    "StatsBombSourceError",
    "fetch_and_preserve_statsbomb_event",
    "fetch_and_preserve_statsbomb_match_index",
    "inspect_statsbomb_event",
    "inspect_statsbomb_match_index",
    "statsbomb_event_url",
    "statsbomb_match_index_url",
]
