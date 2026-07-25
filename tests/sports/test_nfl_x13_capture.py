from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import prediction_market.sports.nfl_x13_capture as capture_module
from prediction_market.sports.nfl_x13_capture import (
    GIB,
    CaptureHttpResponse,
    CaptureTarget,
    GameCapturePlan,
    HostRateLimiter,
    ImmutableRequestManifest,
    X13CaptureError,
    X13TransientCaptureError,
    capture_x13_batch,
    load_x13_capture,
    preflight_x13_capture,
    preserve_capture_response,
)
from prediction_market.sports.nfl_x13_capture import (
    _RunningDiskReservationGate,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
)


UTC = timezone.utc
FETCHED_AT = datetime(2026, 7, 24, 18, 0, tzinfo=UTC)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAST_HOST_RATES = {
    "gamma-api.polymarket.com": 1_000_000_000,
    "data-api.polymarket.com": 1_000_000_000,
    "external-api.kalshi.com": 1_000_000_000,
}
TEST_UNIVERSE_ID = f"sha256:{'7' * 64}"


def _poly_target(
    index: int,
    *,
    game_id: str,
    family: str = "moneyline",
    dependency_game_ids: tuple[str, ...] | None = None,
    analysis_eligible: bool = True,
    market_suffix: str = "",
) -> CaptureTarget:
    condition_id = "0x" + f"{index + 1:064x}"
    return CaptureTarget(
        contract_id=f"polymarket:{condition_id}{market_suffix}",
        venue="polymarket",
        venue_market_id=f"{700_000 + index}{market_suffix}",
        condition_id=condition_id,
        venue_title=f"poly-market-{index}{market_suffix}",
        kalshi_series_ticker=None,
        kalshi_event_ticker=None,
        family=family,
        dependency_game_ids=dependency_game_ids or (game_id,),
        kind="primitive",
        analysis_eligible=analysis_eligible,
    )


def _kalshi_target(
    index: int,
    *,
    game_id: str,
    family: str = "moneyline",
    dependency_game_ids: tuple[str, ...] | None = None,
    kind: str = "primitive",
    analysis_eligible: bool = True,
    ticker_suffix: str = "",
    series_ticker: str = "KXNFLGAME",
    event_ticker: str | None = None,
) -> CaptureTarget:
    binding = X13_GAME_BINDINGS[index]
    frozen_event = event_ticker or binding.kalshi_event_ticker
    team_code = binding.kalshi_away_ticker.rsplit("-", 1)[1]
    ticker = f"{frozen_event}-{team_code}{ticker_suffix}"
    return CaptureTarget(
        contract_id=f"kalshi:{ticker}",
        venue="kalshi",
        venue_market_id=ticker,
        condition_id=None,
        venue_title=f"{binding.away_team} at {binding.home_team}",
        kalshi_series_ticker=series_ticker,
        kalshi_event_ticker=frozen_event,
        family=family,
        dependency_game_ids=dependency_game_ids or (game_id,),
        kind=kind,
        analysis_eligible=analysis_eligible,
    )


def _plan(
    index: int = 0,
    *,
    targets: tuple[CaptureTarget, ...] | None = None,
    start_ts: int = 100,
    end_ts: int = 102,
) -> GameCapturePlan:
    binding = X13_GAME_BINDINGS[index]
    game_id = binding.native_game_id
    return GameCapturePlan(
        game_id=game_id,
        polymarket_event_id=binding.polymarket_event_id,
        polymarket_event_slug=(
            f"nfl-{binding.away_team.lower()}-"
            f"{binding.home_team.lower()}-{binding.game_date.isoformat()}"
        ),
        polymarket_event_title=(
            f"{binding.away_team} vs. {binding.home_team}"
        ),
        start_ts=start_ts,
        end_ts=end_ts,
        targets=targets
        or (
            _poly_target(index, game_id=game_id),
            _kalshi_target(index, game_id=game_id),
        ),
    )


def _json_response(value: object) -> CaptureHttpResponse:
    return CaptureHttpResponse(
        body=json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        headers=(("Content-Type", "application/json"),),
    )


def _gamma_event(plan: GameCapturePlan) -> list[dict[str, object]]:
    return [
        {
            "id": plan.polymarket_event_id,
            "slug": plan.polymarket_event_slug,
            "title": plan.polymarket_event_title,
            "markets": [
                {
                    "id": target.venue_market_id,
                    "conditionId": target.condition_id,
                    "question": target.venue_title,
                }
                for target in plan.targets
                if target.venue == "polymarket"
            ],
        }
    ]


def _kalshi_markets(plan: GameCapturePlan) -> list[dict[str, object]]:
    return [
        {
            "ticker": target.venue_market_id,
            "event_ticker": target.kalshi_event_ticker,
            "title": target.venue_title,
        }
        for target in plan.targets
        if target.venue == "kalshi"
    ]


def _default_responder(
    plans: tuple[GameCapturePlan, ...],
):
    by_game = {plan.game_id: plan for plan in plans}

    def respond(request):
        plan = by_game[request.game_id]
        params = dict(request.params)
        if request.resource == "gamma_event":
            return _json_response(_gamma_event(plan))
        if request.resource == "kalshi_markets":
            params = dict(request.params)
            return _json_response(
                {
                    "markets": [
                        market
                        for market in _kalshi_markets(plan)
                        if market["event_ticker"].startswith(
                            f"{params['series_ticker']}-"
                        )
                    ],
                    "cursor": "",
                }
            )
        if request.resource == "polymarket_trades":
            return _json_response([])
        if request.resource == "kalshi_trades":
            return _json_response({"trades": [], "cursor": ""})
        if request.resource == "kalshi_candlesticks":
            return _json_response(
                {
                    "ticker": request.scope_id,
                    "candlesticks": [],
                }
            )
        raise AssertionError(request)

    return respond


class RecordingManifestWriter:
    def __init__(self) -> None:
        self.requests = []
        self.response_received_at_utc = []
        self._lock = threading.Lock()

    def __call__(
        self,
        request,
        response,
        raw_store_root,
        program_root,
        response_received_at_utc,
    ) -> ImmutableRequestManifest:
        material = (
            request.request_sha256.encode()
            + b"\0"
            + response.body
            + b"\0"
            + response_received_at_utc.isoformat().encode()
        )
        object_sha = "sha256:" + hashlib.sha256(response.body).hexdigest()
        manifest_sha = "sha256:" + hashlib.sha256(material).hexdigest()
        with self._lock:
            self.requests.append(request)
            self.response_received_at_utc.append(
                response_received_at_utc
            )
        return ImmutableRequestManifest(
            request_sha256=request.request_sha256,
            object_sha256=object_sha,
            manifest_sha256=manifest_sha,
            byte_length=len(response.body),
            object_path=raw_store_root / f"{manifest_sha[7:]}.json",
            manifest_path=raw_store_root / f"{manifest_sha[7:]}.manifest.json",
            response_received_at_utc=response_received_at_utc,
        )


class IncrementingResponseClock:
    def __init__(self) -> None:
        self._next = FETCHED_AT
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            value = self._next
            self._next += timedelta(microseconds=1)
            return value


def _disk_usage_with_free(free: int):
    return lambda _path: SimpleNamespace(total=free, used=0, free=free)


def _capture(
    plans: tuple[GameCapturePlan, ...],
    *,
    tmp_path: Path,
    requester=None,
    writer=None,
    publisher=None,
    disk_free: int = 100 * GIB,
    kalshi_max_pages: int = 100,
    cutoff: dict[str, str] | None = None,
    response_clock=None,
    disk_usage=None,
    retry_delays_seconds: tuple[float, ...] = (),
):
    active_requester = requester or _default_responder(plans)
    cutoff_payload = cutoff or {
        "market_positions_last_updated_ts": "2026-05-25T00:00:00Z",
        "market_settled_ts": "2026-05-25T00:00:00Z",
        "orders_updated_ts": "2026-05-25T00:00:00Z",
        "trades_created_ts": "2026-05-25T00:00:00Z",
    }

    def requester_with_cutoff(request):
        if request.resource == "kalshi_cutoff":
            return _json_response(cutoff_payload)
        return active_requester(request)

    frozen_game_ids = tuple(sorted(plan.game_id for plan in plans))
    frozen_target_count = sum(len(plan.targets) for plan in plans)
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setattr(
            capture_module,
            "X13_FROZEN_CAPTURE_GAME_IDS",
            frozen_game_ids,
        )
        monkeypatch.setattr(
            capture_module,
            "X13_FROZEN_CAPTURE_TARGET_COUNT",
            frozen_target_count,
        )
        monkeypatch.setattr(
            capture_module,
            "X13_FROZEN_UNIVERSE_ID",
            TEST_UNIVERSE_ID,
        )
        return capture_x13_batch(
            plans,
            universe_id=TEST_UNIVERSE_ID,
            raw_store_root=tmp_path / "raw-store",
            program_root=PROJECT_ROOT,
            planned_raw_p95_bytes=2 * GIB,
            derived_p95_bytes=3 * GIB,
            requester=requester_with_cutoff,
            manifest_writer=writer,
            pointer_publisher=publisher,
            host_requests_per_second=FAST_HOST_RATES,
            disk_usage=disk_usage or _disk_usage_with_free(disk_free),
            kalshi_max_pages=kalshi_max_pages,
            response_clock=response_clock or IncrementingResponseClock(),
            retry_delays_seconds=retry_delays_seconds,
        )


def test_disk_gate_uses_exact_p95_temp_and_reserve_formula(
    tmp_path: Path,
) -> None:
    plan = _plan()
    required = 2 * GIB + 3 * GIB + GIB + 5 * GIB

    gate = preflight_x13_capture(
        (plan,),
        raw_store_root=tmp_path / "raw-store",
        program_root=PROJECT_ROOT,
        planned_raw_p95_bytes=2 * GIB,
        derived_p95_bytes=3 * GIB,
        disk_usage=_disk_usage_with_free(required),
    )

    assert gate.planned_raw_p95_bytes == 2 * GIB
    assert gate.derived_p95_bytes == 3 * GIB
    assert gate.temporary_headroom_bytes == GIB
    assert gate.reserve_bytes == 5 * GIB
    assert gate.required_free_bytes == required
    assert gate.available_free_bytes == required

    with pytest.raises(X13CaptureError, match="disk preflight"):
        preflight_x13_capture(
            (plan,),
            raw_store_root=tmp_path / "raw-store",
            program_root=PROJECT_ROOT,
            planned_raw_p95_bytes=2 * GIB,
            derived_p95_bytes=3 * GIB,
            disk_usage=_disk_usage_with_free(required - 1),
        )


def test_running_disk_gate_stops_without_publishing_before_reserve_is_consumed(
    tmp_path: Path,
) -> None:
    required = 2 * GIB + 3 * GIB + GIB + 5 * GIB
    calls = 0
    published = []

    def disk_usage(_path):
        nonlocal calls
        calls += 1
        free = required if calls == 1 else 3 * GIB + GIB + 5 * GIB - 1
        return SimpleNamespace(total=free, used=0, free=free)

    with pytest.raises(X13CaptureError, match="running disk gate"):
        _capture(
            (_plan(),),
            tmp_path=tmp_path,
            writer=RecordingManifestWriter(),
            publisher=lambda pointer, root: published.append(pointer),
            disk_usage=disk_usage,
        )

    assert calls >= 2
    assert published == []


def test_running_disk_gate_accounts_for_concurrent_active_writes(
    tmp_path: Path,
) -> None:
    minimum = 9 * GIB
    payload_bytes = 200
    one_reservation = (
        payload_bytes
        + _RunningDiskReservationGate._MANIFEST_HEADROOM_BYTES
    )
    gate = _RunningDiskReservationGate(
        raw_store_root=tmp_path,
        minimum_free_bytes=minimum,
        disk_usage=_disk_usage_with_free(minimum + one_reservation),
    )

    first = gate.acquire(payload_bytes)
    with pytest.raises(X13CaptureError, match="untouched reserve"):
        gate.acquire(payload_bytes)
    gate.release(first, verify_after_write=True)


@pytest.mark.parametrize(
    "bad_target",
    (
        _poly_target(
            0,
            game_id=X13_GAME_IDS[0],
            family="unknown_family",
        ),
        _poly_target(
            0,
            game_id=X13_GAME_IDS[0],
            dependency_game_ids=(X13_GAME_IDS[1],),
        ),
    ),
)
def test_unknown_family_or_wrong_game_dependency_fails_before_network(
    tmp_path: Path,
    bad_target: CaptureTarget,
) -> None:
    good_kalshi = _kalshi_target(0, game_id=X13_GAME_IDS[0])
    plan = _plan(targets=(bad_target, good_kalshi))
    network_requests = []
    published = []

    with pytest.raises(X13CaptureError):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=lambda request: network_requests.append(request),
            writer=RecordingManifestWriter(),
            publisher=lambda pointer, root: published.append(pointer),
        )

    assert network_requests == []
    assert published == []


def test_ambiguous_or_changed_source_title_fails_metadata_preflight_raw_first(
    tmp_path: Path,
) -> None:
    plan = _plan()
    writer = RecordingManifestWriter()
    published = []

    def responder(request):
        assert request.resource == "gamma_event"
        event = _gamma_event(plan)
        event[0]["title"] = "BAL or BUF"
        return _json_response(event)

    with pytest.raises(X13CaptureError, match="Gamma event title"):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=responder,
            writer=writer,
            publisher=lambda pointer, root: published.append(pointer),
        )

    assert [request.resource for request in writer.requests] == [
        "kalshi_cutoff",
        "gamma_event"
    ]
    assert published == []


def test_polymarket_captures_full_event_metadata_and_recursively_proves_windows(
    tmp_path: Path,
) -> None:
    game_id = X13_GAME_IDS[0]
    first = _poly_target(0, game_id=game_id)
    second = replace(
        _poly_target(
            1,
            game_id=game_id,
            market_suffix="-second",
        ),
        condition_id="0x" + "b" * 64,
    )
    kalshi = _kalshi_target(0, game_id=game_id)
    plan = _plan(targets=(first, second, kalshi))
    writer = RecordingManifestWriter()

    def responder(request):
        if request.resource == "gamma_event":
            return _json_response(_gamma_event(plan))
        if request.resource == "kalshi_markets":
            return _json_response(
                {"markets": _kalshi_markets(plan), "cursor": ""}
            )
        if request.resource == "polymarket_trades":
            params = dict(request.params)
            if (
                params["market"] == first.condition_id
                and params["start"] == 100
                and params["end"] == 102
            ):
                return _json_response(
                    [
                        {
                            "conditionId": first.condition_id,
                            "timestamp": 100,
                        }
                    ]
                    * 10_000
                )
            if (
                params["market"] == first.condition_id
                and params["start"] == 100
                and params["end"] == 101
            ):
                return _json_response(
                    [
                        {
                            "conditionId": first.condition_id,
                            "timestamp": 101,
                        }
                    ]
                )
            return _json_response([])
        if request.resource == "kalshi_trades":
            return _json_response({"trades": [], "cursor": ""})
        if request.resource == "kalshi_candlesticks":
            return _json_response(
                {
                    "ticker": request.scope_id,
                    "candlesticks": [],
                }
            )
        raise AssertionError(request)

    result = _capture(
        (plan,),
        tmp_path=tmp_path,
        requester=responder,
        writer=writer,
    )
    game = result.games[0]

    assert game.polymarket_market_count == 2
    assert game.polymarket_trade_count == 1
    assert [
        (
            proof.condition_id,
            proof.start_ts,
            proof.end_ts,
            proof.item_count,
        )
        for proof in game.polymarket_terminal_windows
    ] == [
        (first.condition_id, 100, 101, 1),
        (first.condition_id, 102, 102, 0),
        (second.condition_id, 100, 102, 0),
    ]
    assert len(writer.requests) == len(result.manifests)
    assert all(
        receipt.manifest_sha256.startswith("sha256:")
        for receipt in result.manifests
    )


def test_polymarket_10000_at_one_second_has_no_terminal_proof_or_pointer(
    tmp_path: Path,
) -> None:
    plan = _plan(start_ts=100, end_ts=100)
    condition_id = next(
        target.condition_id
        for target in plan.targets
        if target.venue == "polymarket"
    )
    base = _default_responder((plan,))
    writer = RecordingManifestWriter()

    def responder(request):
        if request.resource == "polymarket_trades":
            return _json_response(
                [
                    {
                        "conditionId": condition_id,
                        "timestamp": 100,
                    }
                ]
                * 10_000
            )
        return base(request)

    with pytest.raises(X13CaptureError, match="10,000.*terminal"):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=responder,
            writer=writer,
        )

    assert writer.requests[-1].resource == "polymarket_trades"
    assert not (
        tmp_path / "raw-store" / "capture-pointers"
    ).exists()


def test_gamma_market_not_bound_by_inventory_fails_before_trade_capture(
    tmp_path: Path,
) -> None:
    plan = _plan()
    writer = RecordingManifestWriter()
    base = _default_responder((plan,))

    def responder(request):
        if request.resource == "gamma_event":
            event = _gamma_event(plan)
            event[0]["markets"].append(
                {
                    "id": "unbound",
                    "conditionId": "0x" + "f" * 64,
                }
            )
            return _json_response(event)
        return base(request)

    with pytest.raises(X13CaptureError, match="inventory"):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=responder,
            writer=writer,
        )

    assert [request.resource for request in writer.requests] == [
        "kalshi_cutoff",
        "gamma_event",
    ]


def test_kalshi_cursor_stable_ids_and_only_eligible_official_1m_candles(
    tmp_path: Path,
) -> None:
    game_id = X13_GAME_IDS[0]
    poly = _poly_target(0, game_id=game_id)
    eligible = _kalshi_target(0, game_id=game_id)
    excluded = replace(
        _kalshi_target(
            0,
            game_id=game_id,
            family="cross_game_mve_inventory",
            dependency_game_ids=(game_id, X13_GAME_IDS[1]),
            kind="cross_game_inventory",
            analysis_eligible=False,
            ticker_suffix="-MVE",
            series_ticker="KXNFLMVE",
            event_ticker="KXNFLMVE-25SEP07BALBUF",
        ),
    )
    plan = _plan(targets=(poly, eligible, excluded))
    writer = RecordingManifestWriter()

    def responder(request):
        params = dict(request.params)
        if request.resource == "gamma_event":
            return _json_response(_gamma_event(plan))
        if request.resource == "kalshi_markets":
            markets = [
                market
                for market in _kalshi_markets(plan)
                if market["event_ticker"].startswith(
                    f"{params['series_ticker']}-"
                )
            ]
            if (
                params["series_ticker"]
                == eligible.kalshi_series_ticker
                and "cursor" not in params
            ):
                unrelated = {
                    "ticker": (
                        f"{eligible.kalshi_series_ticker}-"
                        "25SEP08OTHER-YES"
                    ),
                    "event_ticker": (
                        f"{eligible.kalshi_series_ticker}-"
                        "25SEP08OTHER"
                    ),
                    "title": "unrelated event in complete series scan",
                }
                return _json_response(
                    {
                        "markets": [markets[0], unrelated],
                        "cursor": "next",
                    }
                )
            if (
                params["series_ticker"]
                == eligible.kalshi_series_ticker
            ):
                assert params["cursor"] == "next"
                markets = []
            return _json_response(
                {"markets": markets, "cursor": ""}
            )
        if request.resource == "polymarket_trades":
            return _json_response([])
        if request.resource == "kalshi_trades":
            ticker = params["ticker"]
            if ticker == eligible.venue_market_id:
                if "cursor" not in params:
                    return _json_response(
                        {
                            "trades": [
                                {
                                    "trade_id": "trade-1",
                                    "ticker": ticker,
                                    "price": 51,
                                }
                            ],
                            "cursor": "trade-next",
                        }
                    )
                return _json_response(
                    {
                        "trades": [
                            {
                                "trade_id": "trade-1",
                                "ticker": ticker,
                                "price": 51,
                            },
                            {
                                "trade_id": "trade-2",
                                "ticker": ticker,
                                "price": 52,
                            },
                        ],
                        "cursor": "",
                    }
                )
            return _json_response(
                {
                    "trades": [
                        {
                            "trade_id": "excluded-raw",
                            "ticker": ticker,
                        }
                    ],
                    "cursor": "",
                }
            )
        if request.resource == "kalshi_candlesticks":
            assert params == {
                "start_ts": 100,
                "end_ts": 102,
                "period_interval": 1,
            }
            return _json_response(
                {
                    "ticker": request.scope_id,
                    "candlesticks": [
                        {"end_period_ts": 101},
                        {"end_period_ts": 102},
                    ],
                }
            )
        raise AssertionError(request)

    result = _capture(
        (plan,),
        tmp_path=tmp_path,
        requester=responder,
        writer=writer,
    )
    game = result.games[0]

    assert game.kalshi_market_count == 2
    market_requests = [
        request
        for request in writer.requests
        if request.resource == "kalshi_markets"
    ]
    assert {
        dict(request.params)["series_ticker"]
        for request in market_requests
    } == {
        eligible.kalshi_series_ticker,
        excluded.kalshi_series_ticker,
    }
    assert all(
        "event_ticker" not in dict(request.params)
        for request in market_requests
    )
    assert [
        (
            proof.ticker,
            proof.page_count,
            proof.raw_item_count,
            proof.unique_item_count,
            proof.exact_duplicate_count,
            proof.terminal_cursor,
        )
        for proof in game.kalshi_trade_proofs
    ] == [
        (eligible.venue_market_id, 2, 3, 2, 1, ""),
        (excluded.venue_market_id, 1, 1, 1, 0, ""),
    ]
    assert [
        candle.ticker for candle in game.kalshi_candlesticks
    ] == [eligible.venue_market_id]
    candle_requests = [
        request
        for request in writer.requests
        if request.resource == "kalshi_candlesticks"
    ]
    assert len(candle_requests) == 1
    assert dict(candle_requests[0].params)["period_interval"] == 1


@pytest.mark.parametrize("failure_mode", ("repeat", "missing_terminal"))
def test_kalshi_trade_pagination_fails_closed_without_terminal_proof(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    plan = _plan()
    base = _default_responder((plan,))
    writer = RecordingManifestWriter()
    published = []

    def responder(request):
        if request.resource != "kalshi_trades":
            return base(request)
        params = dict(request.params)
        cursor = params.get("cursor")
        if cursor is None:
            return _json_response({"trades": [], "cursor": "again"})
        if failure_mode == "repeat":
            return _json_response({"trades": [], "cursor": "again"})
        return _json_response({"trades": [], "cursor": "still-open"})

    expected = "repeated cursor" if failure_mode == "repeat" else "terminal"
    with pytest.raises(X13CaptureError, match=expected):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=responder,
            writer=writer,
            publisher=lambda pointer, root: published.append(pointer),
            kalshi_max_pages=2,
        )

    assert published == []


def test_conflicting_kalshi_trade_stable_id_fails_closed(
    tmp_path: Path,
) -> None:
    plan = _plan()
    base = _default_responder((plan,))
    published = []

    def responder(request):
        if request.resource != "kalshi_trades":
            return base(request)
        params = dict(request.params)
        trade = {
            "trade_id": "same-id",
            "ticker": params["ticker"],
            "price": 51 if "cursor" not in params else 52,
        }
        return _json_response(
            {
                "trades": [trade],
                "cursor": "next" if "cursor" not in params else "",
            }
        )

    with pytest.raises(X13CaptureError, match="conflicting duplicate"):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=responder,
            writer=RecordingManifestWriter(),
            publisher=lambda pointer, root: published.append(pointer),
        )

    assert published == []


def test_host_rate_limiter_is_configurable_per_host() -> None:
    now = [0.0]
    sleeps = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = HostRateLimiter(
        {"example.test": 2},
        monotonic=monotonic,
        sleep=sleep,
    )
    limiter.wait_for_url("https://example.test/one")
    limiter.wait_for_url("https://example.test/two")

    assert sleeps == [0.5]


def test_each_response_uses_its_actual_completion_clock_value(
    tmp_path: Path,
) -> None:
    plan = _plan()
    writer = RecordingManifestWriter()
    clock = IncrementingResponseClock()

    result = _capture(
        (plan,),
        tmp_path=tmp_path,
        writer=writer,
        response_clock=clock,
    )

    assert [request.resource for request in writer.requests] == [
        "kalshi_cutoff",
        "gamma_event",
        "kalshi_markets",
        "polymarket_trades",
        "kalshi_trades",
        "kalshi_candlesticks",
    ]
    assert writer.response_received_at_utc == [
        FETCHED_AT + timedelta(microseconds=index)
        for index in range(6)
    ]
    assert [
        receipt.response_received_at_utc
        for receipt in result.manifests
    ] == writer.response_received_at_utc


def test_non_utc_response_clock_fails_before_manifest_or_pointer(
    tmp_path: Path,
) -> None:
    writer = RecordingManifestWriter()
    published = []
    non_utc = datetime(
        2026,
        7,
        24,
        13,
        tzinfo=timezone(timedelta(hours=-5)),
    )

    with pytest.raises(X13CaptureError, match="response clock.*UTC"):
        _capture(
            (_plan(),),
            tmp_path=tmp_path,
            writer=writer,
            publisher=lambda pointer, root: published.append(pointer),
            response_clock=lambda: non_utc,
        )

    assert writer.requests == []
    assert published == []


def test_batch_uses_four_game_workers_and_publishes_only_after_success(
    tmp_path: Path,
) -> None:
    plans = tuple(_plan(index) for index in range(5))
    base = _default_responder(plans)
    active = 0
    maximum = 0
    lock = threading.Lock()
    four_started = threading.Event()
    published = []

    def responder(request):
        nonlocal active, maximum
        if request.resource == "gamma_event":
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == 4:
                    four_started.set()
            assert four_started.wait(timeout=3)
            with lock:
                active -= 1
        return base(request)

    result = _capture(
        plans,
        tmp_path=tmp_path,
        requester=responder,
        writer=RecordingManifestWriter(),
        publisher=lambda pointer, root: (
            published.append(pointer) or root / "pointer.json"
        ),
    )

    assert maximum == 4
    assert tuple(game.game_id for game in result.games) == tuple(
        sorted(plan.game_id for plan in plans)
    )
    assert len(published) == 1
    assert result.pointer.capture_sha256 == published[0].capture_sha256
    assert result.kalshi_cutoff.market_settled_ts.isoformat() == (
        "2026-05-25T00:00:00+00:00"
    )
    assert result.kalshi_cutoff.manifest_sha256 in (
        result.pointer.raw_manifest_sha256s
    )
    assert result.kalshi_cutoff.response_received_at_utc == (
        result.manifests[0].response_received_at_utc
    )
    assert result.pointer.raw_manifest_sha256s == tuple(
        sorted(
            receipt.manifest_sha256
            for receipt in result.manifests
        )
    )
    pointer_material = {
        "game_ids": list(result.pointer.game_ids),
        "plan_sha256": result.pointer.plan_sha256,
        "raw_manifest_sha256s": list(
            result.pointer.raw_manifest_sha256s
        ),
        "receipt_sha256": result.pointer.receipt_sha256,
        "universe_id": result.pointer.universe_id,
        "version": "nfl-x13-capture-pointer-v2",
    }
    expected_capture_sha256 = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                pointer_material,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert result.pointer.capture_sha256 == expected_capture_sha256


def test_kalshi_cutoff_is_frozen_once_and_window_must_be_historical(
    tmp_path: Path,
) -> None:
    cutoff_second = int(
        datetime(2026, 5, 25, tzinfo=UTC).timestamp()
    )
    plan = _plan(start_ts=cutoff_second - 10, end_ts=cutoff_second)
    writer = RecordingManifestWriter()
    published = []

    with pytest.raises(X13CaptureError, match="historical cutoff"):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            writer=writer,
            publisher=lambda pointer, root: published.append(pointer),
        )

    assert [request.resource for request in writer.requests] == [
        "kalshi_cutoff"
    ]
    assert published == []


def test_malformed_cutoff_is_preserved_raw_first_without_pointer(
    tmp_path: Path,
) -> None:
    writer = RecordingManifestWriter()
    published = []

    with pytest.raises(X13CaptureError, match="cutoff trades_created_ts"):
        _capture(
            (_plan(),),
            tmp_path=tmp_path,
            writer=writer,
            publisher=lambda pointer, root: published.append(pointer),
            cutoff={
                "market_positions_last_updated_ts": (
                    "2026-05-25T00:00:00Z"
                ),
                "market_settled_ts": "2026-05-25T00:00:00Z",
                "orders_updated_ts": "2026-05-25T00:00:00Z",
                "trades_created_ts": "not-a-timestamp",
            },
        )

    assert [request.resource for request in writer.requests] == [
        "kalshi_cutoff"
    ]
    assert published == []


def test_public_manifest_writer_preserves_every_request_and_pointer(
    tmp_path: Path,
) -> None:
    plan = _plan()

    result = _capture(
        (plan,),
        tmp_path=tmp_path,
        requester=_default_responder((plan,)),
        writer=preserve_capture_response,
    )

    manifests = list(
        (tmp_path / "raw-store" / "manifests").rglob(
            "*.manifest.json"
        )
    )
    assert len(manifests) == len(result.manifests)
    assert result.pointer_path.is_file()
    assert result.pointer_path.parent.name == "capture-pointers"
    assert result.receipt_path.is_file()
    assert result.receipt_path.parent.name == "capture-receipts"


def test_capture_receipt_reopens_all_typed_proofs_across_process_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    result = _capture(
        (plan,),
        tmp_path=tmp_path,
        requester=_default_responder((plan,)),
        writer=preserve_capture_response,
    )
    monkeypatch.setattr(
        capture_module,
        "X13_FROZEN_CAPTURE_GAME_IDS",
        (plan.game_id,),
    )
    monkeypatch.setattr(
        capture_module,
        "X13_FROZEN_CAPTURE_TARGET_COUNT",
        len(plan.targets),
    )
    monkeypatch.setattr(
        capture_module,
        "X13_FROZEN_UNIVERSE_ID",
        TEST_UNIVERSE_ID,
    )

    loaded = load_x13_capture(
        result.pointer_path,
        plans=(plan,),
        universe_id=TEST_UNIVERSE_ID,
        raw_store_root=tmp_path / "raw-store",
        program_root=PROJECT_ROOT,
    )

    assert loaded.pointer == result.pointer
    assert loaded.receipt.receipt_sha256 == result.receipt.receipt_sha256
    assert loaded.kalshi_cutoff == result.kalshi_cutoff
    assert loaded.games == result.games
    assert loaded.manifests == result.manifests

    receipt_path = loaded.receipt_path
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    with pytest.raises(X13CaptureError, match="canonical JSON"):
        load_x13_capture(
            result.pointer_path,
            plans=(plan,),
            universe_id=TEST_UNIVERSE_ID,
            raw_store_root=tmp_path / "raw-store",
            program_root=PROJECT_ROOT,
        )


def test_formal_capture_rejects_subset_before_any_network_request(
    tmp_path: Path,
) -> None:
    requested = []

    with pytest.raises(X13CaptureError, match="exact frozen 20"):
        capture_x13_batch(
            (_plan(),),
            universe_id=capture_module.X13_FROZEN_UNIVERSE_ID,
            raw_store_root=tmp_path / "raw-store",
            program_root=PROJECT_ROOT,
            planned_raw_p95_bytes=0,
            derived_p95_bytes=0,
            requester=lambda request: requested.append(request),
            host_requests_per_second=FAST_HOST_RATES,
            disk_usage=_disk_usage_with_free(100 * GIB),
        )

    assert requested == []


def test_transient_request_failure_retries_before_preserving_response(
    tmp_path: Path,
) -> None:
    plan = _plan()
    base = _default_responder((plan,))
    attempts: dict[str, int] = {}

    def flaky(request):
        attempts[request.request_sha256] = (
            attempts.get(request.request_sha256, 0) + 1
        )
        if (
            request.resource == "gamma_event"
            and attempts[request.request_sha256] == 1
        ):
            raise X13TransientCaptureError("temporary upstream reset")
        return base(request)

    writer = RecordingManifestWriter()
    result = _capture(
        (plan,),
        tmp_path=tmp_path,
        requester=flaky,
        writer=writer,
        retry_delays_seconds=(0,),
    )

    gamma_request = next(
        request
        for request in writer.requests
        if request.resource == "gamma_event"
    )
    assert attempts[gamma_request.request_sha256] == 2
    assert len(
        [
            request
            for request in writer.requests
            if request.resource == "gamma_event"
        ]
    ) == 1
    assert result.pointer_path.is_file()


def test_transient_request_exhaustion_never_publishes_pointer(
    tmp_path: Path,
) -> None:
    plan = _plan()
    published = []

    def unavailable(request):
        raise X13TransientCaptureError("temporary upstream unavailable")

    with pytest.raises(X13CaptureError, match="retry budget exhausted"):
        _capture(
            (plan,),
            tmp_path=tmp_path,
            requester=unavailable,
            publisher=lambda pointer, root: published.append(pointer),
            retry_delays_seconds=(0, 0),
        )

    assert published == []
