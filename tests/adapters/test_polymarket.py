from __future__ import annotations

import json

import httpx
import pytest

from prediction_market.adapters.polymarket import (
    MARKET_WS_URL,
    DiscoveryError,
    build_market_subscription,
    discover_active_sports_assets,
    parse_market_frame,
)


def test_market_subscription_matches_public_protocol() -> None:
    payload = json.loads(build_market_subscription(["asset-1", "asset-2"]))

    assert MARKET_WS_URL == "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    assert payload == {
        "assets_ids": ["asset-1", "asset-2"],
        "type": "market",
        "custom_feature_enabled": True,
    }


@pytest.mark.parametrize("asset_ids", [[], [""], ["asset-1", "asset-1"]])
def test_market_subscription_rejects_ambiguous_assets(asset_ids: list[str]) -> None:
    with pytest.raises(ValueError):
        build_market_subscription(asset_ids)


def test_market_frame_accepts_documented_object_and_batch() -> None:
    single = parse_market_frame(b'{"event_type":"price_change","asset_id":"1"}')
    batch = parse_market_frame(
        b'[{"event_type":"book","asset_id":"1"},'
        b'{"event_type":"book","asset_id":"2"}]'
    )

    assert [event["event_type"] for event in single] == ["price_change"]
    assert [event["asset_id"] for event in batch] == ["1", "2"]


def test_market_frame_rejects_non_event_json() -> None:
    with pytest.raises(ValueError, match="event_type"):
        parse_market_frame(b'{"asset_id":"1"}')


@pytest.mark.asyncio
async def test_sports_discovery_is_bounded_and_returns_only_active_clob_assets() -> None:
    observed_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(request)
        if request.url.path == "/sports":
            return httpx.Response(
                200,
                json=[
                    {"sport": "nba", "tags": "1,100639,100254"},
                    {"sport": "mlb", "tags": "1,100639,100381"},
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "active": True,
                    "closed": False,
                    "sportsMarketType": "moneyline",
                    "clobTokenIds": '["asset-1","asset-2"]',
                },
                {
                    "active": True,
                    "closed": False,
                    "sportsMarketType": "spread",
                    "clobTokenIds": ["asset-2", "asset-3"],
                },
                {
                    "active": True,
                    "closed": False,
                    "clobTokenIds": ["not-sports"],
                },
                {
                    "active": False,
                    "closed": False,
                    "sportsMarketType": "moneyline",
                    "clobTokenIds": ["inactive"],
                },
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assets = await discover_active_sports_assets(
            client=client,
            market_limit=4,
            max_assets=2,
        )

    assert assets == ("asset-1", "asset-2")
    assert [request.url.path for request in observed_requests] == [
        "/sports",
        "/markets",
    ]
    markets_request = observed_requests[1]
    assert markets_request.url.params["active"] == "true"
    assert markets_request.url.params["closed"] == "false"
    assert markets_request.url.params["limit"] == "4"
    assert markets_request.url.params["tag_id"] == "1"
    assert markets_request.url.params["related_tags"] == "true"


@pytest.mark.asyncio
async def test_sports_discovery_reports_invalid_public_response() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"markets": []})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(DiscoveryError, match="sports metadata.*JSON array"):
            await discover_active_sports_assets(client=client)
