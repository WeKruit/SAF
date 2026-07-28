"""Scan reaction-blind NFL 2025 dual-venue metadata candidates.

The scanner reads only schedule identity columns from the frozen nflverse
Parquet and retains only identity fields from Polymarket Gamma and Kalshi
historical-market responses. It never ranks or filters on scores, prices,
volume, liquidity, or observed market reaction.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile

import httpx
import pandas as pd


POLY_EVENT_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"
KALSHI_MARKETS_URL = (
    "https://external-api.kalshi.com/trade-api/v2/historical/markets"
)
POLY_TRADES_URL = "https://data-api.polymarket.com/trades"
KALSHI_TRADES_URL = (
    "https://api.elections.kalshi.com/trade-api/v2/historical/trades"
)
KALSHI_TEAM_ALIAS = {"JAX": "JAC"}


def _load_used_games(run_index: Path, holdout_selection: Path) -> set[str]:
    discovery = json.loads(run_index.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_selection.read_text(encoding="utf-8"))
    return set(discovery["game_ids"]) | set(holdout["selected_game_ids"])


def _schedule_frame(pbp_path: Path, used_games: set[str]) -> pd.DataFrame:
    columns = [
        "game_id",
        "season_type",
        "week",
        "game_date",
        "away_team",
        "home_team",
    ]
    frame = pd.read_parquet(pbp_path, columns=columns).drop_duplicates("game_id")
    frame = frame[
        frame["season_type"].eq("REG") & ~frame["game_id"].isin(used_games)
    ].copy()
    frame["game_date"] = frame["game_date"].astype(str)
    frame["polymarket_event_slug"] = (
        "nfl-"
        + frame["away_team"].str.lower()
        + "-"
        + frame["home_team"].str.lower()
        + "-"
        + frame["game_date"]
    )
    kalshi_away = frame["away_team"].replace(KALSHI_TEAM_ALIAS)
    kalshi_home = frame["home_team"].replace(KALSHI_TEAM_ALIAS)
    date_code = pd.to_datetime(frame["game_date"]).dt.strftime("%y%b%d").str.upper()
    frame["kalshi_event_ticker"] = (
        "KXNFLGAME-" + date_code + kalshi_away + kalshi_home
    )
    return frame.sort_values("game_id").reset_index(drop=True)


async def _get_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    *,
    params: dict[str, object] | None = None,
) -> tuple[int, object | None]:
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.get(url, params=params)
                if response.status_code == 404:
                    return 404, None
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("retry-after")
                    delay = (
                        float(retry_after)
                        if retry_after is not None
                        else 0.75 * (2**attempt)
                    )
                    await asyncio.sleep(min(delay, 12.0))
                    continue
                response.raise_for_status()
                return response.status_code, response.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                if attempt == 4:
                    return 0, None
                await asyncio.sleep(min(0.75 * (2**attempt), 12.0))
        return 0, None
    raise AssertionError("unreachable")


def _poly_identity(payload: object, slug: str) -> dict[str, object] | None:
    if not isinstance(payload, dict) or payload.get("slug") != slug:
        return None
    event_id = payload.get("id")
    markets = payload.get("markets")
    if not isinstance(event_id, str) or not isinstance(markets, list):
        return None
    condition_ids = sorted(
        {
            market["conditionId"]
            for market in markets
            if isinstance(market, dict)
            and isinstance(market.get("conditionId"), str)
        }
    )
    if not condition_ids:
        return None
    moneylines = [
        market
        for market in markets
        if isinstance(market, dict)
        and market.get("sportsMarketType") == "moneyline"
        and isinstance(market.get("conditionId"), str)
        and isinstance(market.get("id"), str)
    ]
    if len(moneylines) != 1:
        return None
    return {
        "event_id": event_id,
        "event_slug": slug,
        "condition_count": len(condition_ids),
        "moneyline_condition_id": moneylines[0]["conditionId"],
        "moneyline_market_id": moneylines[0]["id"],
    }


def _kalshi_identity(
    payload: object,
    event_ticker: str,
    away_team: str,
    home_team: str,
) -> dict[str, object] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
        return None
    expected_suffixes = {
        KALSHI_TEAM_ALIAS.get(away_team, away_team),
        KALSHI_TEAM_ALIAS.get(home_team, home_team),
    }
    market_refs = sorted(
        market["ticker"]
        for market in payload["markets"]
        if isinstance(market, dict)
        and market.get("event_ticker") == event_ticker
        and isinstance(market.get("ticker"), str)
        and market["ticker"].removeprefix(event_ticker + "-") in expected_suffixes
    )
    if len(market_refs) != 2:
        return None
    return {
        "event_ticker": event_ticker,
        "market_refs": market_refs,
    }


async def _scan(frame: pd.DataFrame, concurrency: int) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "SAF-X13-metadata-candidate-scan/1.0",
    }
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=False,
    ) as client:
        async def inspect(row: object) -> dict[str, object]:
            game = row._asdict()
            poly_status, poly_payload = await _get_json(
                client,
                semaphore,
                POLY_EVENT_URL.format(slug=game["polymarket_event_slug"]),
            )
            kalshi_status, kalshi_payload = await _get_json(
                client,
                semaphore,
                KALSHI_MARKETS_URL,
                params={
                    "event_ticker": game["kalshi_event_ticker"],
                    "limit": 1000,
                },
            )
            poly = _poly_identity(
                poly_payload,
                game["polymarket_event_slug"],
            )
            kalshi = _kalshi_identity(
                kalshi_payload,
                game["kalshi_event_ticker"],
                game["away_team"],
                game["home_team"],
            )
            return {
                "game_id": game["game_id"],
                "week": int(game["week"]),
                "game_date": game["game_date"],
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "polymarket_event_slug": game["polymarket_event_slug"],
                "kalshi_event_ticker": game["kalshi_event_ticker"],
                "polymarket_http_status": poly_status,
                "kalshi_http_status": kalshi_status,
                "polymarket_identity": poly,
                "kalshi_identity": kalshi,
                "dual_venue_metadata": poly is not None and kalshi is not None,
            }

        return await asyncio.gather(
            *(inspect(row) for row in frame.itertuples(index=False))
        )


async def _trade_preflight(
    games: list[dict[str, object]],
    *,
    concurrency: int,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "SAF-X13-trade-coverage-preflight/1.0",
    }
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=False,
    ) as client:
        async def inspect(game: dict[str, object]) -> dict[str, object]:
            poly = game["polymarket_identity"]
            kalshi = game["kalshi_identity"]
            assert isinstance(poly, dict)
            assert isinstance(kalshi, dict)
            poly_status, poly_payload = await _get_json(
                client,
                semaphore,
                POLY_TRADES_URL,
                params={
                    "market": poly["moneyline_condition_id"],
                    "limit": 1,
                    "offset": 0,
                },
            )
            poly_has_trade = (
                poly_status == 200
                and isinstance(poly_payload, list)
                and len(poly_payload) > 0
            )
            ticker_results: dict[str, bool] = {}
            for ticker in kalshi["market_refs"]:
                status, payload = await _get_json(
                    client,
                    semaphore,
                    KALSHI_TRADES_URL,
                    params={"ticker": ticker, "limit": 1},
                )
                ticker_results[str(ticker)] = (
                    status == 200
                    and isinstance(payload, dict)
                    and isinstance(payload.get("trades"), list)
                    and len(payload["trades"]) > 0
                )
            return {
                "game_id": game["game_id"],
                "polymarket_has_trade": poly_has_trade,
                "kalshi_tickers_with_trade": sum(ticker_results.values()),
                "kalshi_ticker_count": len(ticker_results),
                "strict_dual_venue_trade_presence": (
                    poly_has_trade
                    and ticker_results
                    and all(ticker_results.values())
                ),
            }

        return await asyncio.gather(*(inspect(game) for game in games))


def _priority_cohort(
    eligible: list[dict[str, object]],
    *,
    seed: int,
    per_week: int,
) -> list[str]:
    by_week: dict[int, list[dict[str, object]]] = {}
    for game in eligible:
        by_week.setdefault(int(game["week"]), []).append(game)
    selected: list[str] = []
    for week in sorted(by_week):
        ordered = sorted(
            by_week[week],
            key=lambda game: hashlib.sha256(
                f"{seed}:{game['game_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(str(game["game_id"]) for game in ordered[:per_week])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", type=Path, required=True)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--holdout-selection", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--per-week", type=int, default=6)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--trade-preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    used_games = _load_used_games(args.run_index, args.holdout_selection)
    schedule = _schedule_frame(args.pbp, used_games)
    scan = asyncio.run(_scan(schedule, args.concurrency))
    eligible = sorted(
        (game for game in scan if game["dual_venue_metadata"]),
        key=lambda game: str(game["game_id"]),
    )
    priority_game_ids = _priority_cohort(
        eligible,
        seed=args.seed,
        per_week=args.per_week,
    )
    priority_games = [
        game for game in eligible if game["game_id"] in set(priority_game_ids)
    ]
    trade_preflight = (
        asyncio.run(
            _trade_preflight(
                priority_games,
                concurrency=args.concurrency,
            )
        )
        if args.trade_preflight
        else []
    )
    strict_trade_game_ids = sorted(
        str(game["game_id"])
        for game in trade_preflight
        if game["strict_dual_venue_trade_presence"]
    )
    output = {
        "schema": "nfl_2025_dual_venue_metadata_candidates_v0",
        "selection_basis": "metadata_only_no_score_price_volume_or_reaction",
        "seed": args.seed,
        "excluded_prior_game_count": len(used_games),
        "remaining_regular_season_game_count": len(schedule),
        "dual_venue_metadata_count": len(eligible),
        "polymarket_metadata_count": sum(
            game["polymarket_identity"] is not None for game in scan
        ),
        "kalshi_metadata_count": sum(
            game["kalshi_identity"] is not None for game in scan
        ),
        "priority_game_count": len(priority_game_ids),
        "priority_game_ids": priority_game_ids,
        "strict_dual_venue_trade_presence_count": len(strict_trade_game_ids),
        "strict_dual_venue_trade_game_ids": strict_trade_game_ids,
        "priority_trade_missing_game_ids": sorted(
            set(priority_game_ids) - set(strict_trade_game_ids)
        ),
        "trade_preflight": [] if args.summary_only else trade_preflight,
        "games": [] if args.summary_only else scan,
    }
    encoded = (json.dumps(output, sort_keys=True, indent=2) + "\n").encode()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(encoded.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
