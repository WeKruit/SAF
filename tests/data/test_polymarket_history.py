from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from prediction_market.polymarket_history import (
    PolymarketHistoryError,
    capture_polymarket_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCHED_AT = datetime(2026, 7, 23, 22, 0, tzinfo=timezone.utc)
EVENT_SLUG = "nfl-dal-det-2025-12-04"
CONDITION_ID = "0x" + "a" * 64


def _event() -> dict[str, object]:
    return {
        "id": "89365",
        "slug": EVENT_SLUG,
        "markets": [
            {"id": "700742", "conditionId": CONDITION_ID},
        ],
    }


def _trade(timestamp: int) -> dict[str, object]:
    return {
        "asset": "123",
        "conditionId": CONDITION_ID,
        "outcome": "Lions",
        "price": 0.56,
        "side": "BUY",
        "size": 10,
        "timestamp": timestamp,
        "transactionHash": "0x" + "b" * 64,
    }


def _client(*, trades: list[dict[str, object]] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/json"
        assert request.headers["accept-encoding"] == "identity"
        if request.url.host == "gamma-api.polymarket.com":
            assert dict(request.url.params) == {"slug": EVENT_SLUG}
            payload: object = [_event()]
        elif request.url.host == "data-api.polymarket.com":
            assert dict(request.url.params) == {
                "market": CONDITION_ID,
                "start": "100",
                "end": "200",
                "limit": "10000",
                "offset": "0",
            }
            payload = trades if trades is not None else [_trade(100), _trade(200)]
        else:  # pragma: no cover - mock route guard
            raise AssertionError(request.url)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return httpx.Response(
            200,
            content=encoded,
            headers={"Content-Type": "application/json", "ETag": '"fixture"'},
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_capture_preserves_gamma_and_condition_scoped_trades_raw_first(
    tmp_path: Path,
) -> None:
    with _client() as client:
        capture = capture_polymarket_history(
            event_slug=EVENT_SLUG,
            condition_id=CONDITION_ID,
            start_ts=100,
            end_ts=200,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert capture.trade_count == 2
    assert capture.event_id == "89365"
    assert capture.event.record.manifest.license_status == "pending"
    assert capture.trades.record.manifest.source_request["params"] == {
        "market": CONDITION_ID,
        "start": 100,
        "end": 200,
        "limit": 10_000,
        "offset": 0,
    }
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2


def test_capture_rejects_a_full_10000_row_trades_page_after_preserving_it(
    tmp_path: Path,
) -> None:
    with _client(trades=[_trade(100) for _ in range(10_000)]) as client:
        with pytest.raises(PolymarketHistoryError, match="subdivide"):
            capture_polymarket_history(
                event_slug=EVENT_SLUG,
                condition_id=CONDITION_ID,
                start_ts=100,
                end_ts=200,
                store_root=tmp_path,
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
            )

    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2
