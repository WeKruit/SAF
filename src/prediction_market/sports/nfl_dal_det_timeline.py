"""A deterministic source-time timeline for the DAL–DET evidence slice.

This is deliberately a visual/review timeline, not a latency or causality
measurement.  It merges native timestamps only and makes hypothetical latency
an explicit query parameter for a later study.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from prediction_market.contracts import canonical_json_bytes, canonical_sha256
from prediction_market.sports.nfl_dal_det_audit import (
    CANONICAL_GAME_ID,
    NATIVE_GAME_ID,
    DALDETAuditError,
    _require_list,
    _require_mapping,
    _require_sha256,
    _utc_ns,
    _verified_json,
)
from prediction_market.sports.nfl_game_replay import replay_frozen_nfl_game


_LANE_ORDER = {"game_state": 0, "polymarket_trade": 1, "kalshi_trade": 2, "kalshi_candle": 3}
_DEFAULT_LATENCY_SCENARIOS_SECONDS = (0, 1, 2, 5, 10, 30, 60)


class DALDETTimelineError(ValueError):
    """The one-game source-time timeline cannot be constructed safely."""


def _decimal_text(value: object, field: str) -> str:
    if isinstance(value, bool) or type(value) not in {int, float, str, Decimal}:
        raise DALDETTimelineError(f"{field} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise DALDETTimelineError(f"{field} must be a finite decimal") from error
    if not parsed.is_finite():
        raise DALDETTimelineError(f"{field} must be a finite decimal")
    return format(parsed, "f")


def _decimal_sum(values: Sequence[object], field: str) -> str:
    return format(
        sum((Decimal(_decimal_text(value, field)) for value in values), Decimal("0")),
        "f",
    )


def _event_label(previous: Mapping[str, object] | None, current: Mapping[str, object]) -> str:
    if previous is None:
        return "opening_state"
    home_changed = current["home_score"] != previous["home_score"]
    away_changed = current["away_score"] != previous["away_score"]
    if home_changed or away_changed:
        return "score_change"
    if current["possession_team"] != previous["possession_team"]:
        return "possession_change"
    return "play_state"


def _game_rows(program_root: Path) -> list[dict[str, Any]]:
    replay = replay_frozen_nfl_game(program_root=program_root, native_game_id=NATIVE_GAME_ID)
    rows: list[dict[str, Any]] = []
    previous_state: Mapping[str, object] | None = None
    for step in replay.steps:
        source_time = step["source_time_utc"]
        state = _require_mapping(step["next_state"], "game replay next_state")
        if source_time is not None:
            source_time_text = str(source_time)
            rows.append(
                {
                    "lane": "game_state",
                    "source_time_utc": source_time_text,
                    "source_time_ns": _utc_ns(source_time_text, "NFL source time"),
                    "source_precision_ns": 1_000_000,
                    "source_sequence": int(step["sequence"]),
                    "event_type": _event_label(previous_state, state),
                    "game_state": {
                        "period": state["period"],
                        "period_seconds_remaining": state["period_seconds_remaining"],
                        "home_team": state["home_team"],
                        "away_team": state["away_team"],
                        "home_score": state["home_score"],
                        "away_score": state["away_score"],
                        "possession_team": state["possession_team"],
                        "down": state["down"],
                        "distance": state["distance"],
                        "yardline_100": state["yardline_100"],
                    },
                    "time_semantics": "nflverse.source_time_of_day; historical source time only",
                }
            )
        previous_state = state
    if not rows:
        raise DALDETTimelineError("replay has no timestamped game-state rows")
    return rows


def _raw_identity(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _polymarket_rows(payload: object, *, condition_id: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for ordinal, value in enumerate(_require_list(payload, "Polymarket trades")):
        trade = _require_mapping(value, f"Polymarket trade {ordinal}")
        if trade.get("conditionId") != condition_id:
            raise DALDETTimelineError("Polymarket condition mapping changed")
        timestamp = trade.get("timestamp")
        outcome = trade.get("outcome")
        if type(timestamp) is not int or timestamp <= 0 or type(outcome) is not str:
            raise DALDETTimelineError("Polymarket trade has invalid timeline fields")
        buckets[(timestamp, outcome)].append(trade)
    rows: list[dict[str, Any]] = []
    for (timestamp, outcome), trades in sorted(buckets.items()):
        prices = [_decimal_text(trade.get("price"), "Polymarket price") for trade in trades]
        rows.append(
            {
                "lane": "polymarket_trade",
                "source_time_utc": datetime.fromtimestamp(timestamp, tz=timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "source_time_ns": timestamp * 1_000_000_000,
                "source_precision_ns": 1_000_000_000,
                "source_sequence": min(_raw_identity(trade) for trade in trades),
                "condition_id": condition_id,
                "outcome": outcome,
                "trade_count": len(trades),
                "price_min": format(min(Decimal(price) for price in prices), "f"),
                "price_max": format(max(Decimal(price) for price in prices), "f"),
                "size_sum": _decimal_sum([trade.get("size") for trade in trades], "Polymarket size"),
                "time_semantics": "Polymarket trade.timestamp; historical source time only",
            }
        )
    return rows


def _kalshi_trade_rows(payloads: Sequence[object], *, ticker: str) -> list[dict[str, Any]]:
    buckets: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for page_index, payload in enumerate(payloads):
        page = _require_mapping(payload, f"Kalshi trade page {page_index}")
        for trade_index, value in enumerate(_require_list(page.get("trades"), "Kalshi trades")):
            trade = _require_mapping(value, f"Kalshi trade {page_index}:{trade_index}")
            if trade.get("ticker") != ticker:
                raise DALDETTimelineError("Kalshi ticker mapping changed")
            trade_id = trade.get("trade_id")
            if type(trade_id) is not str or not trade_id or trade_id in seen_ids:
                raise DALDETTimelineError("Kalshi trade IDs must be unique")
            seen_ids.add(trade_id)
            buckets[_utc_ns(trade.get("created_time"), "Kalshi created_time") // 1_000_000_000].append(trade)
    rows: list[dict[str, Any]] = []
    for epoch_second, trades in sorted(buckets.items()):
        prices = [
            _decimal_text(trade.get("yes_price_dollars"), "Kalshi yes price")
            for trade in trades
        ]
        rows.append(
            {
                "lane": "kalshi_trade",
                "source_time_utc": datetime.fromtimestamp(epoch_second, tz=timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "source_time_ns": epoch_second * 1_000_000_000,
                "source_precision_ns": 1_000_000_000,
                "source_sequence": min(str(trade["trade_id"]) for trade in trades),
                "ticker": ticker,
                "trade_count": len(trades),
                "yes_price_min": format(min(Decimal(price) for price in prices), "f"),
                "yes_price_max": format(max(Decimal(price) for price in prices), "f"),
                "count_sum": _decimal_sum([trade.get("count_fp") for trade in trades], "Kalshi count"),
                "time_semantics": "Kalshi trade.created_time; historical source time only",
            }
        )
    return rows


def _kalshi_candle_rows(payload: object, *, ticker: str) -> list[dict[str, Any]]:
    data = _require_mapping(payload, "Kalshi candle payload")
    rows: list[dict[str, Any]] = []
    previous_endpoint: int | None = None
    for index, value in enumerate(_require_list(data.get("candlesticks"), "Kalshi candles")):
        candle = _require_mapping(value, f"Kalshi candle {index}")
        endpoint = candle.get("end_period_ts")
        price = _require_mapping(candle.get("price"), "Kalshi candle price")
        yes_bid = _require_mapping(candle.get("yes_bid"), "Kalshi candle yes_bid")
        yes_ask = _require_mapping(candle.get("yes_ask"), "Kalshi candle yes_ask")
        if type(endpoint) is not int or endpoint <= 0:
            raise DALDETTimelineError("Kalshi candle endpoint is invalid")
        if previous_endpoint is not None and endpoint - previous_endpoint != 60:
            raise DALDETTimelineError("Kalshi candle timeline is not contiguous")
        previous_endpoint = endpoint
        rows.append(
            {
                "lane": "kalshi_candle",
                "source_time_utc": datetime.fromtimestamp(endpoint, tz=timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "source_time_ns": endpoint * 1_000_000_000,
                "source_precision_ns": 60 * 1_000_000_000,
                "source_sequence": str(index),
                "ticker": ticker,
                "price_close": _decimal_text(price.get("close"), "Kalshi candle close"),
                "yes_bid_close": _decimal_text(yes_bid.get("close"), "Kalshi candle bid"),
                "yes_ask_close": _decimal_text(yes_ask.get("close"), "Kalshi candle ask"),
                "time_semantics": "Kalshi candle.end_period_ts; one-minute interval end, not event-time quote",
            }
        )
    return rows


def _ordered(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    material = [dict(row) for row in rows]
    try:
        material.sort(
            key=lambda row: (
                int(row["source_time_ns"]),
                _LANE_ORDER[str(row["lane"])],
                str(row["source_sequence"]),
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DALDETTimelineError("timeline row cannot be ordered") from error
    return material


def _latency_scenarios(values: Sequence[object]) -> list[int]:
    if not values:
        raise DALDETTimelineError("at least one latency scenario is required")
    normalized: list[int] = []
    for value in values:
        if type(value) is not int or value < 0:
            raise DALDETTimelineError("latency scenarios must be non-negative seconds")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise DALDETTimelineError("latency scenarios must be unique")
    return sorted(normalized)


def build_dal_det_source_timeline(
    *,
    program_root: str | Path,
    mapping: Mapping[str, object],
    latency_scenarios_seconds: Sequence[object] = _DEFAULT_LATENCY_SCENARIOS_SECONDS,
) -> dict[str, Any]:
    """Materialize the native-time review timeline for exactly one NFL game."""

    root = Path(program_root).resolve()
    store_root = root / "var" / "raw"
    scenarios = _latency_scenarios(latency_scenarios_seconds)
    if mapping.get("canonical_game_id") != CANONICAL_GAME_ID:
        raise DALDETTimelineError("market mapping does not belong to DAL–DET")
    capture = _require_mapping(mapping.get("raw_capture"), "raw_capture")
    venue_data = _require_mapping(mapping.get("venues"), "venues")
    polymarket_capture = _require_mapping(capture.get("polymarket"), "Polymarket capture")
    polymarket_mapping = _require_mapping(venue_data.get("polymarket"), "Polymarket mapping")
    condition_id = polymarket_mapping.get("condition_id")
    if type(condition_id) is not str or not condition_id:
        raise DALDETTimelineError("Polymarket condition id is absent")
    poly_payload, poly_evidence = _verified_json(
        program_root=root,
        store_root=store_root,
        manifest_sha256=_require_sha256(
            polymarket_capture.get("trades_manifest_sha256"), "Polymarket trade manifest"
        ),
    )

    capture_kalshi = _require_mapping(capture.get("kalshi"), "Kalshi capture")
    chains = _require_mapping(capture_kalshi.get("trade_chains"), "Kalshi trade chains")
    candles = _require_mapping(capture_kalshi.get("candle_captures"), "Kalshi candles")
    rows: list[dict[str, Any]] = _game_rows(root)
    rows.extend(_polymarket_rows(poly_payload, condition_id=condition_id))
    kalshi_evidence: dict[str, Any] = {}
    for ticker in sorted(chains):
        chain = _require_mapping(chains[ticker], f"Kalshi chain {ticker}")
        pages: list[object] = []
        evidence: list[dict[str, Any]] = []
        for manifest in _require_list(chain.get("manifest_sha256s"), f"{ticker} manifests"):
            payload, item_evidence = _verified_json(
                program_root=root,
                store_root=store_root,
                manifest_sha256=_require_sha256(manifest, f"{ticker} manifest"),
            )
            pages.append(payload)
            evidence.append(item_evidence)
        rows.extend(_kalshi_trade_rows(pages, ticker=ticker))
        candle = _require_mapping(candles.get(ticker), f"Kalshi candle {ticker}")
        candle_payload, candle_evidence = _verified_json(
            program_root=root,
            store_root=store_root,
            manifest_sha256=_require_sha256(candle.get("manifest_sha256"), f"{ticker} candle manifest"),
        )
        rows.extend(_kalshi_candle_rows(candle_payload, ticker=ticker))
        kalshi_evidence[ticker] = {
            "trade_page_manifests": [item["manifest_sha256"] for item in evidence],
            "candle_manifest": candle_evidence["manifest_sha256"],
        }
    ordered = _ordered(rows)
    lane_counts = {lane: sum(row["lane"] == lane for row in ordered) for lane in _LANE_ORDER}
    return {
        "artifact_type": "nfl_dal_det_source_timeline",
        "artifact_version": "v0",
        "canonical_game_id": CANONICAL_GAME_ID,
        "native_game_id": NATIVE_GAME_ID,
        "time_basis": "native_source_time_only",
        "ordering": {
            "sort_key": ["source_time_ns", "lane_order", "source_sequence"],
            "meaning": "deterministic visual order only; equal/near timestamps do not prove causality",
        },
        "input_evidence": {
            "game_state_replay": replay_frozen_nfl_game(
                program_root=root, native_game_id=NATIVE_GAME_ID
            ).source_manifest_sha256,
            "polymarket_trades": poly_evidence["manifest_sha256"],
            "kalshi": kalshi_evidence,
        },
        "latency_scenario_seconds": scenarios,
        "latency_scenario_semantics": "hypothetical additive delay to game source time; not measured p50/p95/p99",
        "rows_sha256": canonical_sha256(ordered),
        "row_count": len(ordered),
        "lane_counts": lane_counts,
        "rows": ordered,
        "research_boundary": {
            "timeline_visualization_allowed": True,
            "source_time_path_review_allowed": True,
            "measured_latency_allowed": False,
            "event_to_market_causality_allowed": False,
            "executable_quote_or_alpha_allowed": False,
        },
    }


def write_dal_det_source_timeline(
    report: Mapping[str, object], *, output_path: str | Path
) -> None:
    """Atomically write the derived timeline outside the raw store."""

    path = Path(output_path)
    if path.suffix != ".json":
        raise DALDETTimelineError("timeline output must be JSON")
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
    "DALDETTimelineError",
    "build_dal_det_source_timeline",
    "write_dal_det_source_timeline",
]
