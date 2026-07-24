"""Venue-local mapping evidence for the 2025 Dallas at Detroit NFL game."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prediction_market.contracts import canonical_json_bytes


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONDITION_ID = "0xe6f492eb15583ae324d43e987e3ad1b7bd1a86690b3b62a2843162278ff7af0e"
_EVENT_SLUG = "nfl-dal-det-2025-12-04"
_KALSHI_TICKERS = (
    "KXNFLGAME-25DEC04DALDET-DET",
    "KXNFLGAME-25DEC04DALDET-DAL",
)
_TICKER_TEAMS = {
    "KXNFLGAME-25DEC04DALDET-DET": "DET",
    "KXNFLGAME-25DEC04DALDET-DAL": "DAL",
}


class NFLMarketObservationError(ValueError):
    """Venue observations cannot prove the requested stable game mapping."""


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise NFLMarketObservationError(f"{field} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise NFLMarketObservationError(f"{field} must be a non-negative integer")
    return value


def _json_string_list(value: object, field: str) -> list[str]:
    if type(value) is not str:
        raise NFLMarketObservationError(f"{field} must be JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise NFLMarketObservationError(f"{field} must be JSON text") from error
    if (
        type(parsed) is not list
        or len(parsed) != 2
        or any(type(item) is not str or not item for item in parsed)
        or len(set(parsed)) != 2
    ):
        raise NFLMarketObservationError(f"{field} must contain two unique strings")
    return parsed


def _polymarket_mapping(event: Mapping[str, object]) -> dict[str, Any]:
    if event.get("slug") != _EVENT_SLUG:
        raise NFLMarketObservationError("Gamma event slug does not match game")
    event_id = event.get("id")
    if type(event_id) is not str or not event_id:
        raise NFLMarketObservationError("Gamma event must have stable id")
    markets = event.get("markets")
    if type(markets) is not list:
        raise NFLMarketObservationError("Gamma event markets must be a list")
    matching = [
        market
        for market in markets
        if type(market) is dict and market.get("conditionId") == _CONDITION_ID
    ]
    if len(matching) != 1:
        raise NFLMarketObservationError("Gamma condition mapping must be unique")
    market = matching[0]
    market_id = market.get("id")
    question = market.get("question")
    if type(market_id) is not str or not market_id or type(question) is not str:
        raise NFLMarketObservationError("Gamma market identity is incomplete")
    outcomes = _json_string_list(market.get("outcomes"), "Gamma outcomes")
    tokens = _json_string_list(market.get("clobTokenIds"), "Gamma tokens")
    return {
        "event_id": event_id,
        "event_slug": _EVENT_SLUG,
        "gamma_market_id": market_id,
        "condition_id": _CONDITION_ID,
        "question": question,
        "outcomes": [
            {"label": outcome, "token_id": token}
            for outcome, token in zip(outcomes, tokens)
        ],
        "observation_kind": "public_trade_records_not_l2",
    }


def _kalshi_mapping(
    markets: Sequence[Mapping[str, object]],
    trade_runs: Mapping[str, Mapping[str, object]],
    candle_runs: Mapping[str, Mapping[str, object]],
) -> list[dict[str, Any]]:
    by_ticker: dict[str, Mapping[str, object]] = {}
    for market in markets:
        ticker = market.get("ticker")
        if type(ticker) is not str or ticker in by_ticker:
            raise NFLMarketObservationError("Kalshi market tickers must be unique")
        by_ticker[ticker] = market
    if tuple(sorted(by_ticker)) != tuple(sorted(_KALSHI_TICKERS)):
        raise NFLMarketObservationError("Kalshi mapping requires the two winner tickers")
    if set(trade_runs) != set(_KALSHI_TICKERS) or set(candle_runs) != set(_KALSHI_TICKERS):
        raise NFLMarketObservationError("Kalshi runs must cover both winner tickers")

    result: list[dict[str, Any]] = []
    for ticker in _KALSHI_TICKERS:
        market = by_ticker[ticker]
        title = market.get("title")
        participant = market.get("yes_sub_title")
        if type(title) is not str or "Dallas at Detroit" not in title:
            raise NFLMarketObservationError("Kalshi title does not identify game")
        if type(participant) is not str or not participant:
            raise NFLMarketObservationError("Kalshi yes_sub_title is absent")
        run = trade_runs[ticker]
        pages = _nonnegative_int(run.get("pages"), f"{ticker}.pages")
        trades = _nonnegative_int(run.get("trades"), f"{ticker}.trades")
        if run.get("complete") is not True:
            raise NFLMarketObservationError(f"{ticker} historical trade chain is incomplete")
        candle = candle_runs[ticker]
        interval = candle.get("interval_minutes")
        if interval != 1:
            raise NFLMarketObservationError("Kalshi mapping requires one-minute candles")
        candles = _nonnegative_int(candle.get("candles"), f"{ticker}.candles")
        result.append(
            {
                "ticker": ticker,
                "team": _TICKER_TEAMS[ticker],
                "venue_yes_sub_title": participant,
                "historical_trade_pages": pages,
                "historical_trade_count": trades,
                "historical_trade_chain_complete": True,
                "candle_interval_minutes": interval,
                "candle_count": candles,
                "observation_kind": "public_trades_plus_interval_ending_candles_not_l2",
            }
        )
    return result


def build_nfl_market_mapping(
    *,
    polymarket_event: Mapping[str, object],
    polymarket_trade_count: int,
    polymarket_manifest_refs: Mapping[str, object],
    kalshi_markets: Sequence[Mapping[str, object]],
    kalshi_trade_runs: Mapping[str, Mapping[str, object]],
    kalshi_candle_runs: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Build mapping evidence only; price reaction belongs to a later phase."""

    if not isinstance(polymarket_event, Mapping):
        raise NFLMarketObservationError("polymarket_event must be an object")
    if not isinstance(polymarket_manifest_refs, Mapping):
        raise NFLMarketObservationError("polymarket_manifest_refs must be an object")
    if not isinstance(kalshi_markets, Sequence) or isinstance(kalshi_markets, (str, bytes)):
        raise NFLMarketObservationError("kalshi_markets must be a sequence")
    polymarket = _polymarket_mapping(polymarket_event)
    trade_count = _nonnegative_int(polymarket_trade_count, "polymarket_trade_count")
    event_manifest = _sha256(polymarket_manifest_refs.get("event"), "event manifest")
    trade_manifest = _sha256(polymarket_manifest_refs.get("trades"), "trades manifest")
    kalshi = _kalshi_mapping(kalshi_markets, kalshi_trade_runs, kalshi_candle_runs)
    return {
        "artifact_type": "nfl_market_mapping",
        "artifact_version": "v0",
        "canonical_game_id": "game_nflverse_2025_14_DAL_DET",
        "native_game_id": "2025_14_DAL_DET",
        "venue_time_semantics": {
            "polymarket": "trade.timestamp_epoch_seconds",
            "kalshi": "trade.created_time_and_candle.end_period_ts",
        },
        "venues": {
            "polymarket": {
                **polymarket,
                "trade_count": trade_count,
                "source_manifest_refs": {"event": event_manifest, "trades": trade_manifest},
            },
            "kalshi": {"outcome_contracts": kalshi},
        },
        "evidence_boundary": {
            "game_event_time_join_performed": False,
            "reaction_measured": False,
            "cross_venue_comparison": False,
            "pit_validated": False,
            "formal_result_eligible": False,
            "alpha_or_execution_claim": False,
        },
    }


def write_nfl_market_mapping(
    report: Mapping[str, object], *, output_path: str | Path
) -> None:
    """Write one derived mapping report atomically outside the raw store."""

    path = Path(output_path)
    if path.suffix != ".json":
        raise NFLMarketObservationError("mapping output must be a JSON file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(dict(report)) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "NFLMarketObservationError",
    "build_nfl_market_mapping",
    "write_nfl_market_mapping",
]
