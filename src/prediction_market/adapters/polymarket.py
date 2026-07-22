"""Public Polymarket discovery and market-channel wire protocol."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx

from prediction_market.adapters.base import ProtocolError


MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_SPORTS_URL = "https://gamma-api.polymarket.com/sports"


class DiscoveryError(RuntimeError):
    """Public market discovery did not yield a trustworthy response."""


def _validated_unique_strings(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of strings")
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if any(type(value) is not str or not value or value.strip() != value for value in normalized):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


def build_market_subscription(asset_ids: Sequence[str]) -> str:
    """Build the documented unauthenticated market-channel subscription."""

    assets = _validated_unique_strings(asset_ids, "asset_ids")
    return json.dumps(
        {
            "assets_ids": list(assets),
            "type": "market",
            "custom_feature_enabled": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_market_frame(payload: bytes) -> tuple[dict[str, Any], ...]:
    """Parse a market event only after the caller has persisted ``payload``."""

    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    if payload == b"PONG":
        return ()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Polymarket frame is not valid UTF-8 JSON") from exc
    events = value if type(value) is list else [value]
    parsed: list[dict[str, Any]] = []
    for event in events:
        if type(event) is not dict:
            raise ProtocolError("Polymarket frame must contain event objects")
        if type(event.get("event_type")) is not str or not event["event_type"]:
            raise ProtocolError("Polymarket event_type is required")
        parsed.append(event)
    return tuple(parsed)


def _market_assets(value: Any) -> tuple[str, ...]:
    if type(value) is str:
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()
    if type(value) is not list:
        return ()
    assets: list[str] = []
    for item in value:
        if type(item) is not str or not item or item.strip() != item:
            return ()
        assets.append(item)
    return tuple(assets)


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, object] | None,
    context: str,
) -> Any:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise DiscoveryError(f"Polymarket {context} failed: {exc}") from exc


async def _discover_sports_tag(client: httpx.AsyncClient) -> int:
    metadata = await _fetch_json(
        client,
        GAMMA_SPORTS_URL,
        params=None,
        context="sports metadata discovery",
    )
    if type(metadata) is not list:
        raise DiscoveryError("Polymarket sports metadata must be a JSON array")
    counts: Counter[int] = Counter()
    for sport in metadata:
        if type(sport) is not dict or type(sport.get("tags")) is not str:
            continue
        for raw_tag in sport["tags"].split(","):
            tag = raw_tag.strip()
            if tag.isdigit():
                counts[int(tag)] += 1
    if not counts:
        raise DiscoveryError("Polymarket sports metadata contains no numeric tag IDs")
    return min(counts, key=lambda tag: (-counts[tag], tag))


async def _fetch_markets(
    client: httpx.AsyncClient, market_limit: int, sports_tag: int
) -> Any:
    return await _fetch_json(
        client,
        GAMMA_MARKETS_URL,
        params={
            "active": "true",
            "closed": "false",
            "limit": market_limit,
            "tag_id": sports_tag,
            "related_tags": "true",
            "order": "liquidityNum",
            "ascending": "false",
        },
        context="sports market discovery",
    )


async def discover_active_sports_assets(
    *,
    client: httpx.AsyncClient | None = None,
    market_limit: int = 100,
    max_assets: int = 20,
    timeout_seconds: float = 15.0,
) -> tuple[str, ...]:
    """Return a bounded, de-duplicated asset list from active sports markets."""

    if not 1 <= market_limit <= 100:
        raise ValueError("market_limit must be between 1 and 100")
    if max_assets <= 0:
        raise ValueError("max_assets must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if client is None:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout_seconds,
            ) as owned_client:
                sports_tag = await _discover_sports_tag(owned_client)
                markets = await _fetch_markets(
                    owned_client, market_limit, sports_tag
                )
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"Polymarket sports discovery failed: {exc}") from exc
    else:
        sports_tag = await _discover_sports_tag(client)
        markets = await _fetch_markets(client, market_limit, sports_tag)

    if type(markets) is not list:
        raise DiscoveryError("Polymarket markets response must be a JSON array")

    assets: list[str] = []
    seen: set[str] = set()
    for market in markets:
        if type(market) is not dict:
            continue
        if market.get("active") is not True or market.get("closed") is not False:
            continue
        if market.get("acceptingOrders") is False:
            continue
        for asset in _market_assets(market.get("clobTokenIds")):
            if asset in seen:
                continue
            seen.add(asset)
            assets.append(asset)
            if len(assets) == max_assets:
                return tuple(assets)
    return tuple(assets)


__all__ = [
    "DiscoveryError",
    "GAMMA_MARKETS_URL",
    "GAMMA_SPORTS_URL",
    "MARKET_WS_URL",
    "build_market_subscription",
    "discover_active_sports_assets",
    "parse_market_frame",
]
