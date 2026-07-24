"""Raw-first public Kalshi historical REST capture and cutoff reconciliation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

from prediction_market.static_store import StaticObjectRecord, preserve_static_object


KALSHI_API_ROOT = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_HISTORICAL_VERSION = "official-rest-20260723"
HistoricalResource = Literal["markets", "trades"]
_MARKET_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,127}$")


class KalshiHistoricalError(ValueError):
    """A public historical response or live-boundary join is not auditable."""


@dataclass(frozen=True, slots=True)
class KalshiCutoffSnapshot:
    record: StaticObjectRecord
    market_settled_ts: str
    market_positions_last_updated_ts: str
    trades_created_ts: str
    orders_updated_ts: str


@dataclass(frozen=True, slots=True)
class KalshiHistoricalPage:
    record: StaticObjectRecord
    request_cursor: str | None
    response_cursor: str
    item_count: int


@dataclass(frozen=True, slots=True)
class KalshiHistoricalCandlesticks:
    record: StaticObjectRecord
    cutoff: KalshiCutoffSnapshot
    ticker: str
    start_ts: int
    end_ts: int
    period_interval: Literal[1, 60, 1440]
    candlestick_count: int
    first_end_period_ts: int | None
    last_end_period_ts: int | None


@dataclass(frozen=True, slots=True)
class KalshiHistoricalBackfill:
    resource: HistoricalResource
    cutoff: KalshiCutoffSnapshot
    pages: tuple[KalshiHistoricalPage, ...]
    page_count: int
    raw_item_count: int
    unique_item_count: int
    exact_duplicate_count: int
    stable_id_sha256: str
    terminal_cursor: str
    complete: bool


@dataclass(frozen=True, slots=True)
class KalshiBoundaryReport:
    stable_id_field: str
    historical_count: int
    live_count: int
    overlap_count: int
    union_count: int
    union_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_object(payload: bytes, *, context: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise KalshiHistoricalError(f"{context} must contain exact nonempty bytes")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise KalshiHistoricalError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                KalshiHistoricalError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, KalshiHistoricalError) as exc:
        raise KalshiHistoricalError(f"{context} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise KalshiHistoricalError(f"{context} must be a JSON object")
    return value


def _json_schema_fingerprint(value: Any) -> str:
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
        raise KalshiHistoricalError("response contains an unsupported JSON type")

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


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise KalshiHistoricalError("fetched_at must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_utc_text(value: Any, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise KalshiHistoricalError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise KalshiHistoricalError(f"{field} must be ISO-8601") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise KalshiHistoricalError(f"{field} must be UTC")
    return value


def _download_json(
    url: str,
    *,
    params: Mapping[str, object] | None,
    client: httpx.Client,
    max_bytes: int,
) -> tuple[bytes, httpx.Headers]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise KalshiHistoricalError("max_bytes must be a positive integer")
    try:
        with client.stream(
            "GET",
            url,
            params=params,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            encoding = response.headers.get("Content-Encoding")
            if encoding not in {None, "identity"}:
                raise KalshiHistoricalError("Kalshi response is content-encoded")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as exc:
                    raise KalshiHistoricalError("invalid Content-Length") from exc
                if declared_bytes > max_bytes:
                    raise KalshiHistoricalError("Kalshi response exceeds max_bytes")
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise KalshiHistoricalError("Kalshi response exceeds max_bytes")
                chunks.append(chunk)
            if declared is not None and received != declared_bytes:
                raise KalshiHistoricalError(
                    "Kalshi Content-Length does not match received bytes"
                )
            return b"".join(chunks), httpx.Headers(response.headers)
    except httpx.HTTPError as exc:
        raise KalshiHistoricalError(f"Kalshi historical request failed: {exc}") from exc


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


def _preserve_response(
    payload: bytes,
    headers: httpx.Headers,
    *,
    store_root: str | Path,
    program_root: str | Path,
    partition: str,
    source_url: str,
    source_request: Mapping[str, Any],
    source_cursor: str,
    fetched_at: str,
    coverage: str,
    parsed: Mapping[str, Any],
) -> StaticObjectRecord:
    return preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="kalshi",
        dataset="DS-KALSHI-HISTORICAL",
        version=KALSHI_HISTORICAL_VERSION,
        partition=partition,
        extension="json",
        source_url=source_url,
        source_request=source_request,
        source_cursor=source_cursor,
        fetched_at=fetched_at,
        coverage=coverage,
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/json").split(";", 1)[0],
        schema_fingerprint=_json_schema_fingerprint(parsed),
        license_ref="O-003",
        license_status="pending",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )


def capture_kalshi_cutoff(
    *,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 1_000_000,
) -> KalshiCutoffSnapshot:
    """Capture the moving live/historical cutoff as routing evidence."""

    fetched_at_text = _utc_text(fetched_at)
    active, owned = _active_client(client)
    url = f"{KALSHI_API_ROOT}/historical/cutoff"
    try:
        payload, headers = _download_json(
            url, params=None, client=active, max_bytes=max_bytes
        )
    finally:
        if owned:
            active.close()
    parsed = _strict_object(payload, context="Kalshi historical cutoff")
    record = _preserve_response(
        payload,
        headers,
        store_root=store_root,
        program_root=program_root,
        partition="historical-cutoff",
        source_url=url,
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            "params": {},
        },
        source_cursor="request_cursor:ROOT",
        fetched_at=fetched_at_text,
        coverage="dynamic live/historical cutoff",
        parsed=parsed,
    )
    expected = {
        "market_settled_ts",
        "market_positions_last_updated_ts",
        "trades_created_ts",
        "orders_updated_ts",
    }
    if set(parsed) != expected:
        raise KalshiHistoricalError("Kalshi cutoff fields do not match the contract")
    market_cutoff = _validate_utc_text(
        parsed["market_settled_ts"], "market_settled_ts"
    )
    positions_cutoff = _validate_utc_text(
        parsed["market_positions_last_updated_ts"],
        "market_positions_last_updated_ts",
    )
    trade_cutoff = _validate_utc_text(
        parsed["trades_created_ts"], "trades_created_ts"
    )
    order_cutoff = _validate_utc_text(
        parsed["orders_updated_ts"], "orders_updated_ts"
    )
    return KalshiCutoffSnapshot(
        record=record,
        market_settled_ts=market_cutoff,
        market_positions_last_updated_ts=positions_cutoff,
        trades_created_ts=trade_cutoff,
        orders_updated_ts=order_cutoff,
    )


def _resource_contract(resource: str) -> tuple[HistoricalResource, str]:
    if resource == "markets":
        return "markets", "ticker"
    if resource == "trades":
        return "trades", "trade_id"
    raise KalshiHistoricalError("resource must be markets or trades")


def _validated_query(
    resource: HistoricalResource,
    query: Mapping[str, object] | None,
) -> dict[str, object]:
    if query is None:
        return {}
    if not isinstance(query, Mapping):
        raise KalshiHistoricalError("query must be a mapping")
    allowed = {
        "trades": {"ticker", "min_ts", "max_ts", "is_block_trade"},
        "markets": {"tickers", "event_ticker", "series_ticker", "mve_filter"},
    }[resource]
    normalized = dict(query)
    if set(normalized) - allowed:
        raise KalshiHistoricalError("query contains unsupported fields")
    if resource == "trades":
        ticker = normalized.get("ticker")
        if ticker is not None and (
            type(ticker) is not str or _MARKET_TICKER_RE.fullmatch(ticker) is None
        ):
            raise KalshiHistoricalError("query ticker must be a canonical market ID")
        for field in ("min_ts", "max_ts"):
            value = normalized.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise KalshiHistoricalError(f"query {field} must be non-negative integer")
        if (
            normalized.get("min_ts") is not None
            and normalized.get("max_ts") is not None
            and int(normalized["min_ts"]) > int(normalized["max_ts"])
        ):
            raise KalshiHistoricalError("query min_ts must not exceed max_ts")
        block_trade = normalized.get("is_block_trade")
        if block_trade is not None and type(block_trade) is not bool:
            raise KalshiHistoricalError("query is_block_trade must be boolean")
        return normalized

    tickers = normalized.get("tickers")
    if tickers is not None:
        if type(tickers) is not str or not tickers:
            raise KalshiHistoricalError("query tickers must be comma-separated IDs")
        ticker_values = tickers.split(",")
        if any(_MARKET_TICKER_RE.fullmatch(value) is None for value in ticker_values):
            raise KalshiHistoricalError("query tickers must be canonical market IDs")
    for field in ("event_ticker", "series_ticker"):
        value = normalized.get(field)
        if value is not None and (
            type(value) is not str or _MARKET_TICKER_RE.fullmatch(value) is None
        ):
            raise KalshiHistoricalError(f"query {field} must be a canonical ID")
    if normalized.get("mve_filter") not in {None, "exclude"}:
        raise KalshiHistoricalError("query mve_filter must be exclude")
    return normalized


def capture_kalshi_historical_candlesticks(
    ticker: str,
    *,
    start_ts: int,
    end_ts: int,
    period_interval: int,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 50_000_000,
) -> KalshiHistoricalCandlesticks:
    """Capture one exact historical candlestick query with its cutoff."""

    if type(ticker) is not str or _MARKET_TICKER_RE.fullmatch(ticker) is None:
        raise KalshiHistoricalError("ticker must be a canonical Kalshi market ID")
    if type(start_ts) is not int or start_ts < 0:
        raise KalshiHistoricalError("start_ts must be a non-negative integer")
    if type(end_ts) is not int or end_ts < start_ts:
        raise KalshiHistoricalError("end_ts must be an integer at or after start_ts")
    if type(period_interval) is not int or period_interval not in {1, 60, 1440}:
        raise KalshiHistoricalError("period_interval must be 1, 60, or 1440")

    fetched_at_text = _utc_text(fetched_at)
    active, owned = _active_client(client)
    try:
        cutoff = capture_kalshi_cutoff(
            store_root=store_root,
            program_root=program_root,
            fetched_at=fetched_at,
            client=active,
        )
        url = f"{KALSHI_API_ROOT}/historical/markets/{ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        payload, headers = _download_json(
            url,
            params=params,
            client=active,
            max_bytes=max_bytes,
        )
    finally:
        if owned:
            active.close()

    parsed = _strict_object(payload, context="Kalshi historical candlesticks")
    query_material = _canonical_bytes(
        {
            "ticker": ticker,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
    )
    partition = "candlesticks-" + hashlib.sha256(query_material).hexdigest()
    record = _preserve_response(
        payload,
        headers,
        store_root=store_root,
        program_root=program_root,
        partition=partition,
        source_url=url,
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            "params": params,
        },
        source_cursor=(
            f"ticker:{ticker};start_ts:{start_ts};end_ts:{end_ts};"
            f"period_interval:{period_interval}"
        ),
        fetched_at=fetched_at_text,
        coverage=(
            f"historical candlesticks;ticker={ticker};start_ts={start_ts};"
            f"end_ts={end_ts};period_interval={period_interval};"
            f"cutoff_manifest={cutoff.record.manifest.manifest_sha256}"
        ),
        parsed=parsed,
    )
    if set(parsed) != {"ticker", "candlesticks"}:
        raise KalshiHistoricalError(
            "Kalshi candlestick fields do not match the contract"
        )
    if parsed["ticker"] != ticker:
        raise KalshiHistoricalError("Kalshi candlestick ticker mismatch")
    candles = parsed["candlesticks"]
    if type(candles) is not list:
        raise KalshiHistoricalError("Kalshi candlesticks must be an array")
    end_periods: list[int] = []
    for position, candle in enumerate(candles):
        if type(candle) is not dict:
            raise KalshiHistoricalError(
                f"Kalshi candlesticks[{position}] must be an object"
            )
        end_period_ts = candle.get("end_period_ts")
        if type(end_period_ts) is not int:
            raise KalshiHistoricalError(
                f"Kalshi candlesticks[{position}].end_period_ts must be an integer"
            )
        if not start_ts <= end_period_ts <= end_ts:
            raise KalshiHistoricalError(
                "Kalshi candlestick end_period_ts is outside the requested range"
            )
        if end_periods and end_period_ts <= end_periods[-1]:
            raise KalshiHistoricalError(
                "Kalshi candlestick end_period_ts must be strictly increasing"
            )
        end_periods.append(end_period_ts)

    return KalshiHistoricalCandlesticks(
        record=record,
        cutoff=cutoff,
        ticker=ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
        candlestick_count=len(candles),
        first_end_period_ts=end_periods[0] if end_periods else None,
        last_end_period_ts=end_periods[-1] if end_periods else None,
    )


def backfill_kalshi_historical(
    resource: str,
    *,
    query: Mapping[str, object] | None = None,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    page_limit: int = 1000,
    max_pages: int = 100,
    max_page_bytes: int = 50_000_000,
) -> KalshiHistoricalBackfill:
    """Capture cutoff and bounded cursor pages without claiming historical L2."""

    resource_name, stable_id_field = _resource_contract(resource)
    validated_query = _validated_query(resource_name, query)
    if type(page_limit) is not int or not 1 <= page_limit <= 1000:
        raise KalshiHistoricalError("page_limit must be between 1 and 1000")
    if type(max_pages) is not int or max_pages <= 0:
        raise KalshiHistoricalError("max_pages must be a positive integer")
    fetched_at_text = _utc_text(fetched_at)
    active, owned = _active_client(client)
    try:
        cutoff = capture_kalshi_cutoff(
            store_root=store_root,
            program_root=program_root,
            fetched_at=fetched_at,
            client=active,
        )
        url = f"{KALSHI_API_ROOT}/historical/{resource_name}"
        query_sha256 = (
            "sha256:" + hashlib.sha256(_canonical_bytes(validated_query)).hexdigest()
            if validated_query
            else None
        )
        request_cursor: str | None = None
        seen_cursors: set[str] = set()
        pages: list[KalshiHistoricalPage] = []
        canonical_by_id: dict[str, bytes] = {}
        exact_duplicates = 0
        raw_items = 0
        terminal_cursor = ""
        for _ in range(max_pages):
            params: dict[str, object] = {"limit": page_limit, **validated_query}
            if request_cursor is not None:
                params["cursor"] = request_cursor
            payload, headers = _download_json(
                url,
                params=params,
                client=active,
                max_bytes=max_page_bytes,
            )
            parsed = _strict_object(
                payload, context=f"Kalshi historical {resource_name} page"
            )
            cursor_key = request_cursor or "ROOT"
            partition = (
                f"{resource_name}-page-"
                + hashlib.sha256(cursor_key.encode("utf-8")).hexdigest()
            )
            record = _preserve_response(
                payload,
                headers,
                store_root=store_root,
                program_root=program_root,
                partition=partition,
                source_url=url,
                source_request={
                    "method": "GET",
                    "headers": {
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                    },
                    "params": params,
                },
                source_cursor=f"request_cursor:{cursor_key}",
                fetched_at=fetched_at_text,
                coverage=(
                    f"historical {resource_name};cutoff_manifest="
                    f"{cutoff.record.manifest.manifest_sha256};"
                    f"query_sha256={query_sha256 or 'none'}"
                ),
                parsed=parsed,
            )
            if set(parsed) != {resource_name, "cursor"}:
                raise KalshiHistoricalError(
                    f"Kalshi {resource_name} page fields do not match the contract"
                )
            items = parsed[resource_name]
            response_cursor = parsed["cursor"]
            if type(items) is not list or type(response_cursor) is not str:
                raise KalshiHistoricalError(
                    f"Kalshi {resource_name} page has invalid items or cursor"
                )
            pages.append(
                KalshiHistoricalPage(
                    record=record,
                    request_cursor=request_cursor,
                    response_cursor=response_cursor,
                    item_count=len(items),
                )
            )
            raw_items += len(items)
            for position, item in enumerate(items):
                if type(item) is not dict:
                    raise KalshiHistoricalError(
                        f"Kalshi {resource_name}[{position}] must be an object"
                    )
                stable_id = item.get(stable_id_field)
                if type(stable_id) is not str or not stable_id:
                    raise KalshiHistoricalError(
                        f"Kalshi {resource_name} item lacks {stable_id_field}"
                    )
                canonical = _canonical_bytes(item)
                previous = canonical_by_id.get(stable_id)
                if previous is None:
                    canonical_by_id[stable_id] = canonical
                elif previous == canonical:
                    exact_duplicates += 1
                else:
                    raise KalshiHistoricalError(
                        f"conflicting duplicate {stable_id_field}: {stable_id}"
                    )
            terminal_cursor = response_cursor
            if response_cursor == "":
                break
            if response_cursor in seen_cursors or response_cursor == request_cursor:
                raise KalshiHistoricalError(
                    f"Kalshi {resource_name} returned a repeated cursor"
                )
            seen_cursors.add(response_cursor)
            request_cursor = response_cursor
    finally:
        if owned:
            active.close()
    stable_material = [
        {"stable_id": stable_id, "record_sha256": hashlib.sha256(value).hexdigest()}
        for stable_id, value in sorted(canonical_by_id.items())
    ]
    return KalshiHistoricalBackfill(
        resource=resource_name,
        cutoff=cutoff,
        pages=tuple(pages),
        page_count=len(pages),
        raw_item_count=raw_items,
        unique_item_count=len(canonical_by_id),
        exact_duplicate_count=exact_duplicates,
        stable_id_sha256=(
            "sha256:" + hashlib.sha256(_canonical_bytes(stable_material)).hexdigest()
        ),
        terminal_cursor=terminal_cursor,
        complete=terminal_cursor == "",
    )


def _records_by_id(
    records: Sequence[Mapping[str, Any]],
    *,
    stable_id_field: str,
    side: str,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise KalshiHistoricalError(f"{side}[{index}] must be an object")
        stable_id = record.get(stable_id_field)
        if type(stable_id) is not str or not stable_id:
            raise KalshiHistoricalError(
                f"{side}[{index}] lacks stable ID {stable_id_field}"
            )
        canonical = _canonical_bytes(dict(record))
        previous = result.get(stable_id)
        if previous is not None and previous != canonical:
            raise KalshiHistoricalError(f"{side} contains conflicting duplicate IDs")
        result[stable_id] = canonical
    return result


def reconcile_live_historical_records(
    historical: Sequence[Mapping[str, Any]],
    live: Sequence[Mapping[str, Any]],
    *,
    stable_id_field: str,
) -> KalshiBoundaryReport:
    """Reconcile the moving boundary by stable ID, never by row position."""

    if type(stable_id_field) is not str or not stable_id_field:
        raise KalshiHistoricalError("stable_id_field must be nonempty")
    historical_by_id = _records_by_id(
        historical, stable_id_field=stable_id_field, side="historical"
    )
    live_by_id = _records_by_id(live, stable_id_field=stable_id_field, side="live")
    overlap = set(historical_by_id) & set(live_by_id)
    for stable_id in overlap:
        if historical_by_id[stable_id] != live_by_id[stable_id]:
            raise KalshiHistoricalError(
                f"live/historical boundary conflict for {stable_id}"
            )
    union = dict(historical_by_id)
    union.update(live_by_id)
    material = [
        {"stable_id": stable_id, "record_sha256": hashlib.sha256(value).hexdigest()}
        for stable_id, value in sorted(union.items())
    ]
    return KalshiBoundaryReport(
        stable_id_field=stable_id_field,
        historical_count=len(historical_by_id),
        live_count=len(live_by_id),
        overlap_count=len(overlap),
        union_count=len(union),
        union_sha256="sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest(),
    )


__all__ = [
    "KALSHI_API_ROOT",
    "KALSHI_HISTORICAL_VERSION",
    "KalshiBoundaryReport",
    "KalshiCutoffSnapshot",
    "KalshiHistoricalBackfill",
    "KalshiHistoricalCandlesticks",
    "KalshiHistoricalError",
    "KalshiHistoricalPage",
    "backfill_kalshi_historical",
    "capture_kalshi_cutoff",
    "capture_kalshi_historical_candlesticks",
    "reconcile_live_historical_records",
]
