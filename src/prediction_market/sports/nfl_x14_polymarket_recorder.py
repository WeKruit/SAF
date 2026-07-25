"""Targeted prospective Polymarket capture for X-14 CAR at ARI."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Protocol

import httpx

from prediction_market.pmxt.data_store import (
    DataStoreError,
    configured_object_store,
)
from prediction_market.recorder_supervisor import (
    SupervisorResult,
    build_polymarket_health_report,
)
from prediction_market.sports.nfl_x14_prospective import (
    ProspectiveSessionSpec,
)
from prediction_market.static_store import (
    StaticObjectRecord,
    VerifiedStaticObject,
    preserve_static_object,
    read_verified_static_object,
)


TARGET_EVENT_ID = "684046"
TARGET_EVENT_SLUG = "nfl-car-ari-2026-08-07"
GAMMA_EVENT_URL = (
    "https://gamma-api.polymarket.com/events/slug/"
    f"{TARGET_EVENT_SLUG}"
)
TARGET_NATIVE_GAME_ID = "19453"
TARGET_START_UTC = "2026-08-07T00:00:00Z"
TARGET_AWAY_TEAM = "CAR"
TARGET_HOME_TEAM = "ARI"
TARGET_MONEYLINE_MARKET_ID = "2858750"
TARGET_MONEYLINE_CONDITION_ID = (
    "0xd0733676277270d804b2c93701d476d85d62e1d0444369e62e9e38687fc8864f"
)
_ALLOWED_MARKET_TYPES = frozenset({"moneyline", "spreads", "totals"})
_CONDITION_RE = re.compile(r"^0x[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEPENDENCY_FIELDS = frozenset(
    {
        "mveCollection",
        "mveLegs",
        "dependentEvents",
        "dependencyEventIds",
    }
)


class X14PolymarketRecorderError(ValueError):
    """The targeted capture cannot prove identity, completeness, or integrity."""


@dataclass(frozen=True, slots=True)
class GammaHttpResponse:
    payload: bytes
    fetched_at: str
    source_url: str
    request_params: Mapping[str, str]
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TargetMarket:
    market_id: str
    condition_id: str
    outcomes: tuple[str, str]
    token_ids: tuple[str, str]
    market_type: str
    created_at: str


@dataclass(frozen=True, slots=True)
class TargetUniverse:
    event_id: str
    event_slug: str
    native_game_id: str
    start_utc: str
    creation_at: str
    away_team: str
    home_team: str
    markets: tuple[TargetMarket, ...]
    discovered_market_ids: tuple[str, ...]
    selected_universe_equals_discovered: bool

    @property
    def game_identity_key(self) -> str:
        return (
            f"{self.start_utc}|{self.away_team}|{self.home_team}"
        )

    @property
    def market_ids(self) -> tuple[str, ...]:
        return tuple(market.market_id for market in self.markets)

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(market.condition_id for market in self.markets)

    @property
    def token_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                token
                for market in self.markets
                for token in market.token_ids
            )
        )

    @property
    def session_spec(self) -> ProspectiveSessionSpec:
        conditions = tuple(sorted(self.condition_ids))
        tokens = self.token_ids
        return ProspectiveSessionSpec.freeze(
            session_id="nfl-2026-car-ari",
            game_id="nfl-2026-08-07-car-ari",
            scheduled_start_utc=self.start_utc,
            away_team=self.away_team,
            home_team=self.home_team,
            polymarket_event_id=self.event_id,
            polymarket_event_slug=self.event_slug,
            polymarket_native_game_id=self.native_game_id,
            kalshi_event_id="KXNFLGAME-26AUG06CARARI",
            polymarket_condition_ids=conditions,
            discovered_single_game_condition_ids=conditions,
            polymarket_token_ids=tokens,
            discovered_single_game_token_ids=tokens,
            kalshi_credentials_available=False,
            kalshi_market_ids=(),
        )


@dataclass(frozen=True, slots=True)
class SessionReportRecord:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class TargetedCaptureResult:
    universe: TargetUniverse
    metadata_record: StaticObjectRecord
    supervisor_result: SupervisorResult
    report: dict[str, Any]
    report_record: SessionReportRecord


class SegmentObjectStore(Protocol):
    def head_object(self, key: str) -> dict[str, object] | None: ...

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None: ...

    def read_object_chunks(
        self, key: str, *, chunk_size: int = 1024 * 1024
    ) -> Any: ...


Capture = Callable[..., Awaitable[SupervisorResult]]


def _utc_text(value: object, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise X14PolymarketRecorderError(
            f"{field} must be a UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise X14PolymarketRecorderError(
            f"{field} must be a UTC timestamp"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise X14PolymarketRecorderError(
            f"{field} must be a UTC timestamp"
        )
    return value


def _utc_instant(value: str, field: str) -> datetime:
    _utc_text(value, field)
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _iso_instant(value: object, field: str) -> datetime:
    if type(value) is not str or not value:
        raise X14PolymarketRecorderError(
            f"{field} must be an ISO timestamp with timezone"
        )
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as error:
        raise X14PolymarketRecorderError(
            f"{field} must be an ISO timestamp with timezone"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise X14PolymarketRecorderError(
            f"{field} must be an ISO timestamp with timezone"
        )
    return parsed.astimezone(timezone.utc)


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _strict_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise X14PolymarketRecorderError(
                f"Gamma response has duplicate key {key}"
            )
        result[key] = value
    return result


def _strict_json(payload: bytes) -> Any:
    if type(payload) is not bytes or not payload:
        raise X14PolymarketRecorderError(
            "Gamma response must be non-empty exact bytes"
        )
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                X14PolymarketRecorderError(
                    f"Gamma response has non-finite value {token}"
                )
            ),
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        X14PolymarketRecorderError,
    ) as error:
        raise X14PolymarketRecorderError(
            "Gamma response must be strict UTF-8 JSON"
        ) from error


def _json_pair(value: object, field: str) -> tuple[str, str]:
    if type(value) is not str:
        raise X14PolymarketRecorderError(
            f"{field} token/outcome array must be JSON text"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise X14PolymarketRecorderError(
            f"{field} token/outcome array is invalid"
        ) from error
    if (
        type(parsed) is not list
        or len(parsed) != 2
        or any(type(item) is not str or not item for item in parsed)
        or len(set(parsed)) != 2
    ):
        raise X14PolymarketRecorderError(
            f"{field} must contain two unique values"
        )
    return parsed[0], parsed[1]


def fetch_target_event(
    *,
    client: httpx.Client,
    fetched_at: Callable[[], str] = _now_utc,
    max_bytes: int = 8 * 1024 * 1024,
) -> GammaHttpResponse:
    """GET the exact Gamma event while retaining the response bytes."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    params: dict[str, str] = {}
    try:
        with client.stream(
            "GET",
            GAMMA_EVENT_URL,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        ) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding") not in {
                None,
                "identity",
            }:
                raise X14PolymarketRecorderError(
                    "Gamma response is content-encoded"
                )
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise X14PolymarketRecorderError(
                        "Gamma response exceeds max_bytes"
                    )
                chunks.append(chunk)
            payload = b"".join(chunks)
            headers = {
                key.lower(): value
                for key, value in response.headers.items()
            }
    except httpx.HTTPError as error:
        raise X14PolymarketRecorderError(
            f"Gamma GET failed: {error}"
        ) from error
    received_at = _utc_text(fetched_at(), "fetched_at")
    return GammaHttpResponse(
        payload=payload,
        fetched_at=received_at,
        source_url=GAMMA_EVENT_URL,
        request_params=params,
        headers=headers,
    )


def _event_teams(event: Mapping[str, Any]) -> tuple[str, str]:
    teams = event.get("teams")
    if type(teams) is not list or len(teams) != 2:
        raise X14PolymarketRecorderError(
            "Gamma teams do not prove CAR/ARI"
        )
    by_order: dict[str, str] = {}
    for team in teams:
        if type(team) is not dict:
            raise X14PolymarketRecorderError(
                "Gamma teams do not prove CAR/ARI"
            )
        order = team.get("ordering")
        abbreviation = team.get("abbreviation")
        if (
            order not in {"away", "home"}
            or type(abbreviation) is not str
            or order in by_order
        ):
            raise X14PolymarketRecorderError(
                "Gamma teams do not prove CAR/ARI"
            )
        by_order[order] = abbreviation.upper()
    if by_order != {
        "away": TARGET_AWAY_TEAM,
        "home": TARGET_HOME_TEAM,
    }:
        raise X14PolymarketRecorderError(
            "Gamma teams do not prove CAR/ARI"
        )
    return by_order["away"], by_order["home"]


def _market_dependency_is_single_game(
    market: Mapping[str, Any],
) -> None:
    has_dependency = any(
        market.get(field) is not None
        and market.get(field) != ()
        and market.get(field) != []
        for field in _DEPENDENCY_FIELDS
    )
    if market.get("comboStatus") != "disabled" or has_dependency:
        raise X14PolymarketRecorderError(
            "Gamma market has MVE or composite dependency"
        )
    market_type = market.get("sportsMarketType")
    if market_type not in _ALLOWED_MARKET_TYPES:
        raise X14PolymarketRecorderError(
            "Gamma market has unknown settlement dependency"
        )
    if _iso_instant(
        market.get("gameStartTime"), "market.gameStartTime"
    ) != _utc_instant(TARGET_START_UTC, "target start"):
        raise X14PolymarketRecorderError(
            "Gamma market has cross-game start dependency"
        )


def parse_target_universe(
    response: GammaHttpResponse,
) -> TargetUniverse:
    """Validate exact identity and select every embedded single-game market."""

    if (
        response.source_url != GAMMA_EVENT_URL
        or dict(response.request_params) != {}
    ):
        raise X14PolymarketRecorderError(
            "Gamma exact slug request identity is invalid"
        )
    value = _strict_json(response.payload)
    if type(value) is not dict:
        raise X14PolymarketRecorderError(
            "Gamma exact slug endpoint must return a single event object"
        )
    event = value
    if event.get("id") != TARGET_EVENT_ID:
        raise X14PolymarketRecorderError(
            "Gamma event.id does not match 684046"
        )
    if event.get("slug") != TARGET_EVENT_SLUG:
        raise X14PolymarketRecorderError(
            "Gamma event slug is not exact"
        )
    if (
        type(event.get("gameId")) is not int
        or event["gameId"] != int(TARGET_NATIVE_GAME_ID)
    ):
        raise X14PolymarketRecorderError(
            "Gamma gameId does not match 19453"
        )
    if event.get("startTime") != TARGET_START_UTC:
        raise X14PolymarketRecorderError(
            "Gamma UTC start does not match CAR/ARI"
        )
    if (
        event.get("seriesSlug") != "nfl-2026"
        or type(event.get("sport")) is not dict
        or event["sport"].get("sport") != "nfl"
    ):
        raise X14PolymarketRecorderError(
            "Gamma event is not structurally NFL 2026"
        )
    away, home = _event_teams(event)
    creation_at = _utc_text(event.get("createdAt"), "event.createdAt")
    raw_markets = event.get("markets")
    if type(raw_markets) is not list or not raw_markets:
        raise X14PolymarketRecorderError(
            "Gamma event has no discovered markets"
        )

    discovered_ids: list[str] = []
    selected: list[TargetMarket] = []
    seen_conditions: set[str] = set()
    seen_tokens: set[str] = set()
    for index, raw_market in enumerate(raw_markets):
        if type(raw_market) is not dict:
            raise X14PolymarketRecorderError(
                f"Gamma market[{index}] is not an object"
            )
        market_id = raw_market.get("id")
        if (
            type(market_id) is not str
            or not market_id
            or market_id in discovered_ids
        ):
            raise X14PolymarketRecorderError(
                "Gamma discovered market identity is invalid"
            )
        discovered_ids.append(market_id)
        _market_dependency_is_single_game(raw_market)
        condition_id = raw_market.get("conditionId")
        if (
            type(condition_id) is not str
            or _CONDITION_RE.fullmatch(condition_id) is None
            or condition_id in seen_conditions
        ):
            raise X14PolymarketRecorderError(
                "Gamma condition identity is invalid"
            )
        outcomes = _json_pair(
            raw_market.get("outcomes"),
            f"market {market_id} outcomes",
        )
        tokens = _json_pair(
            raw_market.get("clobTokenIds"),
            f"market {market_id} token IDs",
        )
        if any(token in seen_tokens for token in tokens):
            raise X14PolymarketRecorderError(
                "Gamma token identity is duplicated"
            )
        created_at = _utc_text(
            raw_market.get("createdAt"),
            f"market {market_id} createdAt",
        )
        seen_conditions.add(condition_id)
        seen_tokens.update(tokens)
        selected.append(
            TargetMarket(
                market_id=market_id,
                condition_id=condition_id,
                outcomes=outcomes,
                token_ids=tokens,
                market_type=raw_market["sportsMarketType"],
                created_at=created_at,
            )
        )

    moneyline = [
        market
        for market in selected
        if market.market_type == "moneyline"
    ]
    if (
        len(moneyline) != 1
        or moneyline[0].market_id != TARGET_MONEYLINE_MARKET_ID
        or moneyline[0].condition_id
        != TARGET_MONEYLINE_CONDITION_ID
    ):
        raise X14PolymarketRecorderError(
            "Gamma primary moneyline identity changed"
        )
    selected_ids = tuple(market.market_id for market in selected)
    discovered = tuple(discovered_ids)
    if selected_ids != discovered:
        raise X14PolymarketRecorderError(
            "selected universe is not exactly the discovered universe"
        )
    universe = TargetUniverse(
        event_id=TARGET_EVENT_ID,
        event_slug=TARGET_EVENT_SLUG,
        native_game_id=TARGET_NATIVE_GAME_ID,
        start_utc=TARGET_START_UTC,
        creation_at=creation_at,
        away_team=away,
        home_team=home,
        markets=tuple(selected),
        discovered_market_ids=discovered,
        selected_universe_equals_discovered=True,
    )
    _ = universe.session_spec
    return universe


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _schema_fingerprint(value: Any) -> str:
    observed: dict[str, set[str]] = {}

    def visit(child: Any, path: str) -> None:
        kind = (
            "null"
            if child is None
            else "boolean"
            if type(child) is bool
            else "integer"
            if type(child) is int
            else "number"
            if type(child) is float
            else "string"
            if type(child) is str
            else "array"
            if type(child) is list
            else "object"
            if type(child) is dict
            else "unsupported"
        )
        if kind == "unsupported":
            raise X14PolymarketRecorderError(
                "Gamma response has unsupported JSON type"
            )
        observed.setdefault(path, set()).add(kind)
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    material = {
        path: sorted(kinds) for path, kinds in sorted(observed.items())
    }
    return "sha256:" + hashlib.sha256(
        _canonical_bytes(material)
    ).hexdigest()


def preserve_target_metadata(
    response: GammaHttpResponse,
    *,
    raw_root: str | Path,
    program_root: str | Path,
) -> StaticObjectRecord:
    """Publish the exact Gamma bytes and governed sidecar commit point."""

    parsed = _strict_json(response.payload)
    return preserve_static_object(
        raw_root,
        response.payload,
        program_root=program_root,
        source="polymarket",
        dataset="DS-POLYMARKET-PUBLIC",
        version="gamma-x14-v0",
        partition="event-684046",
        extension="json",
        source_url=response.source_url,
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            "params": dict(response.request_params),
        },
        source_cursor=None,
        fetched_at=_utc_text(response.fetched_at, "fetched_at"),
        coverage=(
            "byte-exact response from Gamma event slug endpoint "
            f"{TARGET_EVENT_SLUG}; identity and market-universe "
            "validation is a downstream fail-closed gate"
        ),
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        media_type=response.headers.get(
            "content-type", "application/json"
        ).split(";", 1)[0],
        schema_fingerprint=_schema_fingerprint(parsed),
        license_ref="O-001",
        license_status="research_only",
        upstream_partition="event-684046",
        object_kind="byte_exact_original",
        lineage={
            "source_object_refs": [],
            "query_sha256": None,
        },
    )


def verify_target_metadata(
    manifest_path: str | Path,
    *,
    raw_root: str | Path,
    program_root: str | Path,
) -> VerifiedStaticObject:
    return read_verified_static_object(
        manifest_path,
        store_root=raw_root,
        program_root=program_root,
    )


def _s3_key(prefix: str, sha256: str) -> str:
    if type(prefix) is not str or not prefix.strip("/"):
        raise X14PolymarketRecorderError(
            "S3 prefix must be non-empty"
        )
    if _DIGEST_RE.fullmatch(sha256) is None:
        raise X14PolymarketRecorderError(
            "segment SHA-256 is invalid"
        )
    digest = sha256.removeprefix("sha256:")
    return (
        f"{prefix.strip('/')}/raw/polymarket-x14/sha256/"
        f"{digest[:2]}/{digest}.jsonl.zst"
    )


def _head_matches(
    head: Mapping[str, object] | None,
    *,
    byte_size: int,
    sha256: str,
) -> bool:
    return (
        head is not None
        and head.get("byte_size") == byte_size
        and type(head.get("metadata")) is dict
        and head["metadata"].get("sha256") == sha256
    )


def upload_capture_segments(
    result: SupervisorResult,
    *,
    object_store: SegmentObjectStore,
    prefix: str,
) -> dict[str, Any]:
    """Upload immutable segment objects and verify authoritative HEAD data."""

    objects: list[dict[str, Any]] = []
    every_readback_verified = True
    for segment in result.manifests:
        key = _s3_key(prefix, segment.object_sha256)
        head = object_store.head_object(key)
        if head is not None and not _head_matches(
            head,
            byte_size=segment.object_size_bytes,
            sha256=segment.object_sha256,
        ):
            raise X14PolymarketRecorderError(
                "existing S3 HEAD byte_size or sha256 metadata mismatch"
            )
        upload_status = "ALREADY_PRESENT"
        if head is None:
            object_store.put_file(
                key,
                segment.object_path,
                metadata={"sha256": segment.object_sha256},
            )
            upload_status = "UPLOADED"
        verified_head = object_store.head_object(key)
        if not _head_matches(
            verified_head,
            byte_size=segment.object_size_bytes,
            sha256=segment.object_sha256,
        ):
            raise X14PolymarketRecorderError(
                "post-upload S3 HEAD byte_size or sha256 metadata mismatch"
            )
        readback_digest = hashlib.sha256()
        readback_byte_size = 0
        for chunk in object_store.read_object_chunks(key):
            if type(chunk) is not bytes:
                raise X14PolymarketRecorderError(
                    "S3 GET readback yielded non-bytes"
                )
            readback_digest.update(chunk)
            readback_byte_size += len(chunk)
        readback_matches = (
            readback_byte_size == segment.object_size_bytes
            and "sha256:" + readback_digest.hexdigest()
            == segment.object_sha256
        )
        every_readback_verified = (
            every_readback_verified and readback_matches
        )
        objects.append(
            {
                "key": key,
                "sha256": segment.object_sha256,
                "byte_size": segment.object_size_bytes,
                "upload_status": upload_status,
                "head_confirmed": True,
                "readback_sha256_matches": readback_matches,
            }
        )
    return {
        "status": (
            "VERIFIED"
            if every_readback_verified
            else "HEAD_CONFIRMED"
        ),
        "object_count": len(objects),
        "objects": objects,
    }


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(
        root.resolve(strict=True)
    ).as_posix()


def _market_report(market: TargetMarket) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "market_type": market.market_type,
        "outcomes": list(market.outcomes),
        "token_ids": list(market.token_ids),
        "created_at": market.created_at,
    }


def _build_report(
    *,
    universe: TargetUniverse,
    metadata_record: StaticObjectRecord,
    supervisor_result: SupervisorResult,
    raw_root: Path,
    s3_upload: Mapping[str, Any],
) -> dict[str, Any]:
    started = _utc_instant(
        supervisor_result.started_at, "recorder started_at"
    )
    creation = _utc_instant(
        universe.creation_at, "market creation_at"
    )
    late_start = started > creation
    health = build_polymarket_health_report(
        supervisor_result,
        raw_root=raw_root,
    )
    resolution_observed = False
    continuous_gate = (
        s3_upload.get("status") == "VERIFIED"
        and not late_start
        and resolution_observed
        and supervisor_result.complete
        and supervisor_result.counters.gaps == 0
    )
    report: dict[str, Any] = {
        "report_version": "nfl-x14-polymarket-session/v0",
        "experiment_id": "X-14",
        "session_id": "nfl-2026-car-ari",
        "game_identity": {
            "identity_key": universe.game_identity_key,
            "scheduled_start_utc": universe.start_utc,
            "away_team": universe.away_team,
            "home_team": universe.home_team,
            "polymarket_event_id": universe.event_id,
            "polymarket_event_slug": universe.event_slug,
            "polymarket_native_game_id": universe.native_game_id,
        },
        "universe": {
            "market_count": len(universe.markets),
            "market_ids": list(universe.market_ids),
            "condition_ids": list(universe.condition_ids),
            "token_ids": list(universe.token_ids),
            "discovered_market_ids": list(
                universe.discovered_market_ids
            ),
            "selected_universe_equals_discovered": (
                universe.selected_universe_equals_discovered
            ),
            "markets": [
                _market_report(market)
                for market in universe.markets
            ],
        },
        "metadata": {
            "object_path": _relative(
                metadata_record.object_path, raw_root
            ),
            "object_sha256": (
                metadata_record.manifest.object_sha256
            ),
            "manifest_path": _relative(
                metadata_record.manifest_path, raw_root
            ),
            "manifest_sha256": (
                metadata_record.manifest.manifest_sha256
            ),
            "fetched_at": (
                metadata_record.manifest.fetched_at
            ),
        },
        "capture": health,
        "coverage": {
            "required_window": "market_creation_to_resolution",
            "market_creation_at": universe.creation_at,
            "local_recorder_started_at": (
                supervisor_result.started_at
            ),
            "resolution_observed": resolution_observed,
            "status": (
                "PARTIAL_LATE_START"
                if late_start
                else "STARTED_BY_CREATION"
            ),
            "complete_from_creation": not late_start,
            "unobserved_pre_recorder_l2": late_start,
        },
        "s3_upload": dict(s3_upload),
        "gates": {
            "metadata_identity": True,
            "universe_exact": True,
            "local_bounded_smoke": supervisor_result.complete,
            "continuous_capture": continuous_gate,
            "association_analysis": False,
        },
        "claims": {
            "creation_to_resolution_complete": False,
            "reaction_latency_measured": False,
            "alpha_or_execution": False,
        },
    }
    report["report_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


def _publish_session_report(
    report_root: str | Path,
    report: Mapping[str, Any],
) -> SessionReportRecord:
    sha256 = report.get("report_sha256")
    if type(sha256) is not str or _DIGEST_RE.fullmatch(sha256) is None:
        raise X14PolymarketRecorderError(
            "session report SHA-256 is invalid"
        )
    material = dict(report)
    material.pop("report_sha256")
    expected = "sha256:" + hashlib.sha256(
        _canonical_bytes(material)
    ).hexdigest()
    if sha256 != expected:
        raise X14PolymarketRecorderError(
            "session report SHA-256 mismatch"
        )
    payload = _canonical_bytes(dict(report)) + b"\n"
    digest = sha256.removeprefix("sha256:")
    root = Path(report_root)
    target = (
        root / "sha256" / digest[:2] / f"{digest}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".x14-report-",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise X14PolymarketRecorderError(
                    "content-addressed session report collision"
                )
        directory_fd = os.open(
            target.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return SessionReportRecord(path=target, sha256=sha256)


def verify_session_report(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise X14PolymarketRecorderError(
            "session report is unreadable"
        ) from error
    if not payload.endswith(b"\n"):
        raise X14PolymarketRecorderError(
            "session report is not canonical JSON"
        )
    document = _strict_json(payload[:-1])
    if type(document) is not dict:
        raise X14PolymarketRecorderError(
            "session report must be an object"
        )
    sha256 = document.get("report_sha256")
    if type(sha256) is not str or _DIGEST_RE.fullmatch(sha256) is None:
        raise X14PolymarketRecorderError(
            "session report SHA-256 is invalid"
        )
    digest = sha256.removeprefix("sha256:")
    if target.name != f"{digest}.json":
        raise X14PolymarketRecorderError(
            "session report path is not content-addressed"
        )
    material = dict(document)
    del material["report_sha256"]
    expected = "sha256:" + hashlib.sha256(
        _canonical_bytes(material)
    ).hexdigest()
    if sha256 != expected:
        raise X14PolymarketRecorderError(
            "session report SHA-256 mismatch"
        )
    if payload != _canonical_bytes(document) + b"\n":
        raise X14PolymarketRecorderError(
            "session report is not canonical JSON"
        )
    return document


async def run_targeted_capture(
    *,
    gamma_response: GammaHttpResponse,
    raw_root: str | Path,
    report_root: str | Path,
    program_root: str | Path,
    run_seconds: float,
    max_frames: int | None,
    receive_timeout_seconds: float,
    max_reconnects: int | None,
    capture: Capture | None = None,
    object_store: SegmentObjectStore | None,
    s3_prefix: str | None,
) -> TargetedCaptureResult:
    """Preserve metadata, run bounded WS capture, and publish one report."""

    raw_path = Path(raw_root)
    metadata_record = preserve_target_metadata(
        gamma_response,
        raw_root=raw_path,
        program_root=program_root,
    )
    universe = parse_target_universe(gamma_response)
    if capture is None:
        from prediction_market.cli.record_markets import (
            capture_public_polymarket,
        )

        capture = capture_public_polymarket
    supervisor_result = await capture(
        universe.session_spec.polymarket_token_ids,
        raw_path,
        run_seconds=run_seconds,
        max_frames=max_frames,
        receive_timeout_seconds=receive_timeout_seconds,
        max_reconnects=max_reconnects,
    )
    if object_store is None:
        s3_upload: dict[str, Any] = {
            "status": "NOT_CONFIGURED",
            "object_count": 0,
            "objects": [],
        }
    else:
        if s3_prefix is None:
            raise X14PolymarketRecorderError(
                "configured S3 store requires a prefix"
            )
        s3_upload = upload_capture_segments(
            supervisor_result,
            object_store=object_store,
            prefix=s3_prefix,
        )
    report = _build_report(
        universe=universe,
        metadata_record=metadata_record,
        supervisor_result=supervisor_result,
        raw_root=raw_path,
        s3_upload=s3_upload,
    )
    report_record = _publish_session_report(
        report_root, report
    )
    return TargetedCaptureResult(
        universe=universe,
        metadata_record=metadata_record,
        supervisor_result=supervisor_result,
        report=report,
        report_record=report_record,
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be non-negative"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prediction_market.sports."
        "nfl_x14_polymarket_recorder",
        description=(
            "Run a bounded public CAR-at-ARI Polymarket recorder."
        ),
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--program-root", type=Path, default=Path(".")
    )
    parser.add_argument(
        "--run-seconds", type=_positive_float, required=True
    )
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument(
        "--receive-timeout-seconds",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument(
        "--max-reconnects", type=_nonnegative_int, default=3
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=_positive_float,
        default=15.0,
    )
    parser.add_argument("--env-file", type=Path)
    return parser


async def _run_cli(args: argparse.Namespace) -> TargetedCaptureResult:
    with httpx.Client(timeout=args.http_timeout_seconds) as client:
        response = fetch_target_event(client=client)
    try:
        object_store, prefix = configured_object_store(
            env_file=args.env_file
        )
    except DataStoreError as error:
        if args.env_file is not None:
            raise X14PolymarketRecorderError(
                f"explicit S3 configuration is invalid: {error}"
            ) from error
        object_store = None
        prefix = None
    return await run_targeted_capture(
        gamma_response=response,
        raw_root=args.raw_root,
        report_root=args.report_root,
        program_root=args.program_root,
        run_seconds=args.run_seconds,
        max_frames=args.max_frames,
        receive_timeout_seconds=args.receive_timeout_seconds,
        max_reconnects=args.max_reconnects,
        object_store=object_store,
        s3_prefix=prefix,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(_run_cli(args))
    except (
        DataStoreError,
        OSError,
        TimeoutError,
        X14PolymarketRecorderError,
    ) as error:
        print(f"X-14 Polymarket recorder failed: {error}", file=sys.stderr)
        return 2
    print(result.report_record.path)
    return 0 if result.supervisor_result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GammaHttpResponse",
    "SessionReportRecord",
    "TargetMarket",
    "TargetUniverse",
    "TargetedCaptureResult",
    "X14PolymarketRecorderError",
    "build_parser",
    "fetch_target_event",
    "main",
    "parse_target_universe",
    "preserve_target_metadata",
    "run_targeted_capture",
    "upload_capture_segments",
    "verify_session_report",
    "verify_target_metadata",
]
