"""Raw-first bounded capture of public Polymarket historical observations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from prediction_market.static_store import StaticObjectRecord, preserve_static_object


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
DATA_API_TRADES_URL = "https://data-api.polymarket.com/trades"
POLYMARKET_PUBLIC_VERSION = "public-api-20260723"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_CONDITION_RE = re.compile(r"^0x[0-9a-f]{64}$")


class PolymarketHistoryError(ValueError):
    """A public historical response cannot prove the requested observation."""


@dataclass(frozen=True, slots=True)
class PolymarketHistoricalObject:
    record: StaticObjectRecord


@dataclass(frozen=True, slots=True)
class PolymarketHistoricalCapture:
    event: PolymarketHistoricalObject
    trades: PolymarketHistoricalObject
    event_id: str
    event_slug: str
    condition_id: str
    start_ts: int
    end_ts: int
    trade_count: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PolymarketHistoryError("fetched_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _json(payload: bytes, *, context: str) -> Any:
    if type(payload) is not bytes or not payload:
        raise PolymarketHistoryError(f"{context} must contain nonempty bytes")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PolymarketHistoryError(f"{context} contains non-finite {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, PolymarketHistoryError) as error:
        raise PolymarketHistoryError(
            f"{context} must be strict UTF-8 JSON"
        ) from error


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolymarketHistoryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _schema_fingerprint(value: Any) -> str:
    observed: dict[str, set[str]] = {}

    def kind(child: Any) -> str:
        if child is None:
            return "null"
        if type(child) is bool:
            return "boolean"
        if type(child) is int:
            return "integer"
        if type(child) is float:
            return "number"
        if type(child) is str:
            return "string"
        if type(child) is list:
            return "array"
        if type(child) is dict:
            return "object"
        raise PolymarketHistoryError("response contains unsupported JSON type")

    def visit(child: Any, path: str) -> None:
        observed.setdefault(path, set()).add(kind(child))
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    material = {path: sorted(types) for path, types in sorted(observed.items())}
    return "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _download(
    url: str,
    *,
    params: Mapping[str, object],
    client: httpx.Client,
    max_bytes: int,
) -> tuple[bytes, httpx.Headers]:
    try:
        with client.stream(
            "GET",
            url,
            params=params,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding") not in {None, "identity"}:
                raise PolymarketHistoryError("response is content-encoded")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise PolymarketHistoryError("response exceeds max_bytes")
                chunks.append(chunk)
            return b"".join(chunks), httpx.Headers(response.headers)
    except httpx.HTTPError as error:
        raise PolymarketHistoryError(f"public request failed: {error}") from error


def _preserve(
    payload: bytes,
    headers: httpx.Headers,
    *,
    store_root: str | Path,
    program_root: str | Path,
    partition: str,
    source_url: str,
    params: Mapping[str, object],
    source_cursor: str,
    fetched_at: str,
    coverage: str,
    parsed: Any,
) -> StaticObjectRecord:
    return preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="polymarket",
        dataset="DS-POLYMARKET-PUBLIC",
        version=POLYMARKET_PUBLIC_VERSION,
        partition=partition,
        extension="json",
        source_url=source_url,
        source_request={
            "method": "GET",
            "headers": {"Accept": "application/json", "Accept-Encoding": "identity"},
            "params": dict(params),
        },
        source_cursor=source_cursor,
        fetched_at=fetched_at,
        coverage=coverage,
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/json").split(";", 1)[0],
        schema_fingerprint=_schema_fingerprint(parsed),
        license_ref="O-001",
        license_status="pending",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )


def _validate_inputs(
    *, event_slug: str, condition_id: str, start_ts: int, end_ts: int
) -> None:
    if type(event_slug) is not str or _SLUG_RE.fullmatch(event_slug) is None:
        raise PolymarketHistoryError("event_slug is not canonical")
    if type(condition_id) is not str or _CONDITION_RE.fullmatch(condition_id) is None:
        raise PolymarketHistoryError("condition_id must be lowercase 0x SHA-256")
    if type(start_ts) is not int or type(end_ts) is not int or start_ts < 0 or end_ts < start_ts:
        raise PolymarketHistoryError("trade window must be non-negative and ordered")


def _validate_event(value: Any, *, event_slug: str, condition_id: str) -> str:
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise PolymarketHistoryError("Gamma event query must return exactly one object")
    event = value[0]
    event_id = event.get("id")
    if type(event_id) is not str or not event_id:
        raise PolymarketHistoryError("Gamma event lacks stable event id")
    if event.get("slug") != event_slug:
        raise PolymarketHistoryError("Gamma event slug does not match request")
    markets = event.get("markets")
    if type(markets) is not list or not any(
        type(market) is dict and market.get("conditionId") == condition_id
        for market in markets
    ):
        raise PolymarketHistoryError("Gamma event does not contain requested condition")
    return event_id


def _validate_trades(
    value: Any, *, condition_id: str, start_ts: int, end_ts: int
) -> int:
    if type(value) is not list:
        raise PolymarketHistoryError("Data API trades response must be an array")
    if len(value) == 10_000:
        raise PolymarketHistoryError("trade page reached 10000 rows; subdivide window")
    for index, trade in enumerate(value):
        if type(trade) is not dict:
            raise PolymarketHistoryError(f"trade[{index}] must be an object")
        if trade.get("conditionId") != condition_id:
            raise PolymarketHistoryError(f"trade[{index}] condition does not match")
        timestamp = trade.get("timestamp")
        if type(timestamp) is not int or not start_ts <= timestamp <= end_ts:
            raise PolymarketHistoryError(f"trade[{index}] timestamp outside window")
    return len(value)


def capture_polymarket_history(
    *,
    event_slug: str,
    condition_id: str,
    start_ts: int,
    end_ts: int,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 50_000_000,
) -> PolymarketHistoricalCapture:
    """Preserve one closed event and one exact condition's bounded trade window."""

    _validate_inputs(
        event_slug=event_slug,
        condition_id=condition_id,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    fetched_at_text = _utc_text(fetched_at)
    owned = client is None
    active = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=30.0),
    )
    try:
        event_params = {"slug": event_slug}
        event_payload, event_headers = _download(
            GAMMA_EVENTS_URL,
            params=event_params,
            client=active,
            max_bytes=max_bytes,
        )
        event_value = _json(event_payload, context="Gamma event")
        event_record = _preserve(
            event_payload,
            event_headers,
            store_root=store_root,
            program_root=program_root,
            partition=f"gamma-event-{event_slug}",
            source_url=GAMMA_EVENTS_URL,
            params=event_params,
            source_cursor=f"slug:{event_slug}",
            fetched_at=fetched_at_text,
            coverage=f"closed event metadata;slug={event_slug}",
            parsed=event_value,
        )
        event_id = _validate_event(
            event_value, event_slug=event_slug, condition_id=condition_id
        )

        trade_params = {
            "market": condition_id,
            "start": start_ts,
            "end": end_ts,
            "limit": 10_000,
            "offset": 0,
        }
        trade_payload, trade_headers = _download(
            DATA_API_TRADES_URL,
            params=trade_params,
            client=active,
            max_bytes=max_bytes,
        )
        trade_value = _json(trade_payload, context="Data API trades")
        query_digest = hashlib.sha256(_canonical_bytes(trade_params)).hexdigest()
        trade_record = _preserve(
            trade_payload,
            trade_headers,
            store_root=store_root,
            program_root=program_root,
            partition=f"trades-{query_digest}",
            source_url=DATA_API_TRADES_URL,
            params=trade_params,
            source_cursor=(
                f"condition:{condition_id};start:{start_ts};end:{end_ts};offset:0"
            ),
            fetched_at=fetched_at_text,
            coverage=(
                f"condition-scoped trades;condition={condition_id};"
                f"start={start_ts};end={end_ts};offset=0"
            ),
            parsed=trade_value,
        )
        trade_count = _validate_trades(
            trade_value,
            condition_id=condition_id,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    finally:
        if owned:
            active.close()

    return PolymarketHistoricalCapture(
        event=PolymarketHistoricalObject(record=event_record),
        trades=PolymarketHistoricalObject(record=trade_record),
        event_id=event_id,
        event_slug=event_slug,
        condition_id=condition_id,
        start_ts=start_ts,
        end_ts=end_ts,
        trade_count=trade_count,
    )


__all__ = [
    "DATA_API_TRADES_URL",
    "GAMMA_EVENTS_URL",
    "POLYMARKET_PUBLIC_VERSION",
    "PolymarketHistoricalCapture",
    "PolymarketHistoricalObject",
    "PolymarketHistoryError",
    "capture_polymarket_history",
]
