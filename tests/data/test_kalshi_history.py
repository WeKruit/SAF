from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from prediction_market.kalshi_history import (
    KalshiHistoricalError,
    backfill_kalshi_historical,
    capture_kalshi_cutoff,
    capture_kalshi_historical_candlesticks,
    reconcile_live_historical_records,
)
from prediction_market.static_store import read_verified_static_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCHED_AT = datetime(2026, 7, 22, 14, 0, tzinfo=timezone.utc)


def _responses() -> dict[tuple[str, str], dict[str, object]]:
    return {
        ("/trade-api/v2/historical/cutoff", ""): {
            "market_settled_ts": "2026-04-01T00:00:00Z",
            "trades_created_ts": "2026-04-02T00:00:00Z",
            "orders_updated_ts": "2026-04-03T00:00:00Z",
        },
        ("/trade-api/v2/historical/markets", ""): {
            "markets": [
                {"ticker": "KXNBA-ONE", "status": "settled"},
                {"ticker": "KXNFL-TWO", "status": "settled"},
            ],
            "cursor": "next-page",
        },
        ("/trade-api/v2/historical/markets", "next-page"): {
            "markets": [
                {"ticker": "KXNFL-TWO", "status": "settled"},
                {"ticker": "KXEPL-THREE", "status": "settled"},
            ],
            "cursor": "",
        },
        ("/trade-api/v2/historical/markets/KXNBA-ONE/candlesticks", ""): {
            "ticker": "KXNBA-ONE",
            "candlesticks": [
                {
                    "end_period_ts": 1_700_000_060,
                    "yes_bid": {"open": "0.45", "close": "0.46"},
                    "yes_ask": {"open": "0.47", "close": "0.48"},
                    "price": {"open": "0.46", "close": "0.47"},
                    "volume": "10.00",
                    "open_interest": "25.00",
                },
                {
                    "end_period_ts": 1_700_000_120,
                    "yes_bid": {"open": "0.46", "close": "0.47"},
                    "yes_ask": {"open": "0.48", "close": "0.49"},
                    "price": {"open": "0.47", "close": "0.48"},
                    "volume": "12.00",
                    "open_interest": "27.00",
                },
            ],
        },
    }


def _client(
    responses: dict[tuple[str, str], dict[str, object]],
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/json"
        assert request.headers["accept-encoding"] == "identity"
        cursor = request.url.params.get("cursor", "")
        key = (request.url.path, cursor)
        if key not in responses:
            return httpx.Response(404, request=request)
        payload = json.dumps(
            responses[key], sort_keys=True, separators=(",", ":")
        ).encode()
        return httpx.Response(
            200,
            content=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
                "ETag": f'"{cursor or "root"}"',
            },
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_cutoff_is_preserved_before_historical_routing(tmp_path: Path) -> None:
    with _client(_responses()) as client:
        cutoff = capture_kalshi_cutoff(
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert cutoff.market_settled_ts == "2026-04-01T00:00:00Z"
    assert cutoff.record.partition == "historical-cutoff"
    assert cutoff.record.manifest.dataset_id == "DS-KALSHI-HISTORICAL"
    assert cutoff.record.manifest.license_ref == "O-003"
    verified = read_verified_static_object(
        cutoff.record.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert json.loads(verified.object_bytes)["trades_created_ts"] == (
        "2026-04-02T00:00:00Z"
    )


def test_semantically_changed_cutoff_is_still_preserved_raw_first(
    tmp_path: Path,
) -> None:
    changed = _responses()
    changed[("/trade-api/v2/historical/cutoff", "")]["new_cutoff"] = (
        "2026-04-04T00:00:00Z"
    )

    with _client(changed) as client:
        with pytest.raises(KalshiHistoricalError, match="fields"):
            capture_kalshi_cutoff(
                store_root=tmp_path,
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
            )

    assert len(list((tmp_path / "raw").rglob("*.json"))) == 1
    assert len(list((tmp_path / "manifests").rglob("*.manifest.json"))) == 1


def test_market_pages_preserve_request_cursor_and_dedupe_stable_ids(
    tmp_path: Path,
) -> None:
    with _client(_responses()) as client:
        result = backfill_kalshi_historical(
            "markets",
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
            page_limit=2,
            max_pages=3,
        )

    assert result.complete is True
    assert result.page_count == 2
    assert result.raw_item_count == 4
    assert result.unique_item_count == 3
    assert result.exact_duplicate_count == 1
    assert result.terminal_cursor == ""
    assert result.stable_id_sha256.startswith("sha256:")
    assert [page.request_cursor for page in result.pages] == [None, "next-page"]
    assert result.pages[0].record.manifest.source_cursor == "request_cursor:ROOT"
    assert result.pages[1].record.manifest.source_cursor == (
        "request_cursor:next-page"
    )
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 3


def test_semantically_changed_page_is_still_preserved_raw_first(
    tmp_path: Path,
) -> None:
    changed = _responses()
    changed[("/trade-api/v2/historical/markets", "")]["new_field"] = True

    with _client(changed) as client:
        with pytest.raises(KalshiHistoricalError, match="page fields"):
            backfill_kalshi_historical(
                "markets",
                store_root=tmp_path,
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
                page_limit=2,
                max_pages=1,
            )

    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2


def test_historical_candlesticks_preserve_exact_query_and_object(
    tmp_path: Path,
) -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        payload = json.dumps(
            _responses()[(request.url.path, "")],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "application/json"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = capture_kalshi_historical_candlesticks(
            "KXNBA-ONE",
            start_ts=1_700_000_000,
            end_ts=1_700_000_180,
            period_interval=1,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert result.ticker == "KXNBA-ONE"
    assert result.period_interval == 1
    assert result.candlestick_count == 2
    assert result.first_end_period_ts == 1_700_000_060
    assert result.last_end_period_ts == 1_700_000_120
    assert [request.url.path for request in observed] == [
        "/trade-api/v2/historical/cutoff",
        "/trade-api/v2/historical/markets/KXNBA-ONE/candlesticks",
    ]
    request = observed[-1]
    assert dict(request.url.params) == {
        "start_ts": "1700000000",
        "end_ts": "1700000180",
        "period_interval": "1",
    }
    assert result.record.manifest.source_request["params"] == {
        "start_ts": 1_700_000_000,
        "end_ts": 1_700_000_180,
        "period_interval": 1,
    }
    verified = read_verified_static_object(
        result.record.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert json.loads(verified.object_bytes)["ticker"] == "KXNBA-ONE"


@pytest.mark.parametrize(
    ("ticker", "start_ts", "end_ts", "period_interval"),
    [
        ("../escape", 1, 2, 1),
        ("KXNBA", -1, 2, 1),
        ("KXNBA", 2, 1, 1),
        ("KXNBA", 1, 2, 5),
    ],
)
def test_historical_candlesticks_reject_unsafe_or_invalid_query(
    tmp_path: Path,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> None:
    with pytest.raises(KalshiHistoricalError):
        capture_kalshi_historical_candlesticks(
            ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=period_interval,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=_client(_responses()),
        )


def test_historical_candlesticks_fail_closed_on_identity_or_time_anomaly(
    tmp_path: Path,
) -> None:
    wrong_identity = _responses()
    path = "/trade-api/v2/historical/markets/KXNBA-ONE/candlesticks"
    wrong_identity[(path, "")]["ticker"] = "KXNBA-TWO"
    with _client(wrong_identity) as client:
        with pytest.raises(KalshiHistoricalError, match="ticker mismatch"):
            capture_kalshi_historical_candlesticks(
                "KXNBA-ONE",
                start_ts=1_700_000_000,
                end_ts=1_700_000_180,
                period_interval=1,
                store_root=tmp_path / "identity",
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
            )
    assert len(list((tmp_path / "identity" / "raw").rglob("*.json"))) == 2

    bad_time = _responses()
    bad_time[(path, "")]["candlesticks"][1]["end_period_ts"] = 1_700_000_060  # type: ignore[index]
    with _client(bad_time) as client:
        with pytest.raises(KalshiHistoricalError, match="strictly increasing"):
            capture_kalshi_historical_candlesticks(
                "KXNBA-ONE",
                start_ts=1_700_000_000,
                end_ts=1_700_000_180,
                period_interval=1,
                store_root=tmp_path / "time",
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
            )


def test_nonterminal_page_budget_and_repeated_cursor_are_explicit(
    tmp_path: Path,
) -> None:
    with _client(_responses()) as client:
        bounded = backfill_kalshi_historical(
            "markets",
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
            page_limit=2,
            max_pages=1,
        )
    assert bounded.complete is False
    assert bounded.terminal_cursor == "next-page"

    repeated = _responses()
    repeated[("/trade-api/v2/historical/markets", "next-page")]["cursor"] = (
        "next-page"
    )
    with _client(repeated) as client:
        with pytest.raises(KalshiHistoricalError, match="repeated cursor"):
            backfill_kalshi_historical(
                "markets",
                store_root=tmp_path / "repeated",
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
                page_limit=2,
                max_pages=3,
            )


def test_conflicting_duplicate_id_fails_closed(tmp_path: Path) -> None:
    conflicting = _responses()
    second = conflicting[("/trade-api/v2/historical/markets", "next-page")]
    second["markets"][0] = {"ticker": "KXNFL-TWO", "status": "open"}  # type: ignore[index]

    with _client(conflicting) as client:
        with pytest.raises(KalshiHistoricalError, match="conflicting duplicate"):
            backfill_kalshi_historical(
                "markets",
                store_root=tmp_path,
                program_root=PROJECT_ROOT,
                fetched_at=FETCHED_AT,
                client=client,
                page_limit=2,
                max_pages=3,
            )


def test_live_historical_boundary_reconciles_by_stable_id_and_content() -> None:
    historical = [
        {"trade_id": "trade-1", "ticker": "KXNBA", "count_fp": "1.00"},
        {"trade_id": "trade-2", "ticker": "KXNFL", "count_fp": "2.00"},
    ]
    live = [
        {"trade_id": "trade-2", "ticker": "KXNFL", "count_fp": "2.00"},
        {"trade_id": "trade-3", "ticker": "KXEPL", "count_fp": "3.00"},
    ]

    report = reconcile_live_historical_records(
        historical,
        live,
        stable_id_field="trade_id",
    )

    assert report.historical_count == 2
    assert report.live_count == 2
    assert report.overlap_count == 1
    assert report.union_count == 3
    assert report.union_sha256.startswith("sha256:")

    with pytest.raises(KalshiHistoricalError, match="boundary conflict"):
        reconcile_live_historical_records(
            historical,
            [{"trade_id": "trade-2", "ticker": "KXNFL", "count_fp": "9.00"}],
            stable_id_field="trade_id",
        )
