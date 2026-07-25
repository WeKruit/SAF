"""First-layer source-time merge for one fully reconstructed NFL game.

The resulting order is a deterministic, historical-source-time view.  It is
intentionally *not* a reaction join: none of these archived sources contains
the local receipt time or a measured clock-error bound required for that
claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

from prediction_market.contracts import canonical_json_bytes, canonical_sha256
from prediction_market.sports.nfl_dal_det_audit import (
    CANONICAL_GAME_ID,
    DALDETAuditError,
    _require_list,
    _require_mapping,
    _require_sha256,
    _utc_ns,
    _verified_json,
)
from prediction_market.sports.nfl_dal_det_personnel import (
    build_dal_det_event_personnel_timeline,
)
from prediction_market.sports.nfl_dal_det_timeline import (
    _kalshi_candle_rows,
    _kalshi_trade_rows,
    _ordered,
    _polymarket_rows,
)


class DALDETEventMarketError(ValueError):
    """The event/personnel and market evidence cannot share a safe timeline."""


_EVENT_MARKET_LANE_ORDER = {
    "nfl_event_personnel": 0,
    "polymarket_trade": 1,
    "kalshi_trade": 2,
    "kalshi_candle": 3,
}


def _ordered_event_market(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    material = [dict(row) for row in rows]
    try:
        material.sort(
            key=lambda row: (
                int(row["source_time_ns"]),
                _EVENT_MARKET_LANE_ORDER[str(row["lane"])],
                str(row["source_sequence"]),
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DALDETEventMarketError("event/market row cannot be deterministically ordered") from error
    return material


def _market_rows(*, root: Path, mapping: Mapping[str, object]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    store_root = root / "var" / "raw"
    if mapping.get("canonical_game_id") != CANONICAL_GAME_ID:
        raise DALDETEventMarketError("market mapping does not belong to DAL–DET")
    capture = _require_mapping(mapping.get("raw_capture"), "raw_capture")
    venue_data = _require_mapping(mapping.get("venues"), "venues")
    polymarket_capture = _require_mapping(capture.get("polymarket"), "Polymarket capture")
    polymarket_mapping = _require_mapping(venue_data.get("polymarket"), "Polymarket mapping")
    condition_id = polymarket_mapping.get("condition_id")
    if type(condition_id) is not str or not condition_id:
        raise DALDETEventMarketError("Polymarket condition id is absent")
    poly_payload, poly_evidence = _verified_json(
        program_root=root,
        store_root=store_root,
        manifest_sha256=_require_sha256(polymarket_capture.get("trades_manifest_sha256"), "Polymarket trade manifest"),
    )
    rows = _polymarket_rows(poly_payload, condition_id=condition_id)
    capture_kalshi = _require_mapping(capture.get("kalshi"), "Kalshi capture")
    chains = _require_mapping(capture_kalshi.get("trade_chains"), "Kalshi trade chains")
    candles = _require_mapping(capture_kalshi.get("candle_captures"), "Kalshi candles")
    kalshi_evidence: dict[str, Any] = {}
    for ticker in sorted(chains):
        chain = _require_mapping(chains[ticker], f"Kalshi chain {ticker}")
        pages: list[object] = []
        page_evidence: list[dict[str, Any]] = []
        for manifest in _require_list(chain.get("manifest_sha256s"), f"{ticker} manifests"):
            payload, evidence = _verified_json(
                program_root=root,
                store_root=store_root,
                manifest_sha256=_require_sha256(manifest, f"{ticker} manifest"),
            )
            pages.append(payload)
            page_evidence.append(evidence)
        rows.extend(_kalshi_trade_rows(pages, ticker=ticker))
        candle = _require_mapping(candles.get(ticker), f"Kalshi candle {ticker}")
        candle_payload, candle_evidence = _verified_json(
            program_root=root,
            store_root=store_root,
            manifest_sha256=_require_sha256(candle.get("manifest_sha256"), f"{ticker} candle manifest"),
        )
        rows.extend(_kalshi_candle_rows(candle_payload, ticker=ticker))
        kalshi_evidence[ticker] = {
            "trade_page_manifests": [item["manifest_sha256"] for item in page_evidence],
            "candle_manifest": candle_evidence["manifest_sha256"],
        }
    return rows, {
        "polymarket_trade_manifest": poly_evidence["manifest_sha256"],
        "kalshi": kalshi_evidence,
    }


def build_dal_det_event_market_source_time_timeline(
    *, program_root: str | Path, mapping: Mapping[str, object]
) -> dict[str, Any]:
    """Build the first-layer chronological merge of full NFL state and markets."""

    root = Path(program_root).resolve()
    personnel = build_dal_det_event_personnel_timeline(program_root=root)
    event_rows: list[dict[str, Any]] = []
    for row in personnel["rows"]:
        source_time = row["source_time_utc"]
        if type(source_time) is not str:
            raise DALDETEventMarketError("event/personnel row is missing source time")
        state = _require_mapping(row["state"], "event/personnel state")
        event = _require_mapping(row["event"], "event/personnel event")
        event_rows.append({
            "lane": "nfl_event_personnel",
            "source_time_utc": source_time,
            "source_time_ns": _utc_ns(source_time, "NFL event source time"),
            "source_sequence": row["order_sequence"],
            "play_id": row["play_id"],
            "play_type": event["play_type"],
            "flags": event["flags"],
            "state": {
                key: state[key]
                for key in (
                    "qtr", "quarter_seconds_remaining", "posteam", "defteam", "down",
                    "yardline_100", "total_home_score", "total_away_score",
                )
            },
            "time_semantics": "nflverse.time_of_day; historical source event timestamp only",
        })
    market_rows, market_evidence = _market_rows(root=root, mapping=mapping)
    ordered = _ordered_event_market([*event_rows, *market_rows])
    lane_counts = {
        lane: sum(row["lane"] == lane for row in ordered)
        for lane in _EVENT_MARKET_LANE_ORDER
    }
    if lane_counts["nfl_event_personnel"] != personnel["join"]["joined_rows"]:
        raise DALDETEventMarketError("not every full event/personnel row entered the merge")
    report: dict[str, Any] = {
        "artifact_type": "nfl_dal_det_event_market_source_time_timeline",
        "artifact_version": "v1",
        "canonical_game_id": CANONICAL_GAME_ID,
        "inputs": {
            "event_personnel_artifact_sha256": personnel["artifact_sha256"],
            **market_evidence,
        },
        "counts": {
            "event_personnel_rows": lane_counts["nfl_event_personnel"],
            "market_rows": len(ordered) - lane_counts["nfl_event_personnel"],
            "total_rows": len(ordered),
            "by_lane": lane_counts,
        },
        "time_semantics": {
            "nfl_event": "nflverse.time_of_day historical source event timestamp",
            "polymarket_trade": "Polymarket trade.timestamp historical source time",
            "kalshi_trade": "Kalshi trade.created_time historical source time",
            "kalshi_candle": "Kalshi candle.end_period_ts one-minute interval end",
        },
        "merge_validation": {
            "status": "PASS_SOURCE_TIME_CHRONOLOGY_ONLY",
            "sort_key": ["source_time_ns", "lane_order", "source_sequence"],
            "event_order_preserved": True,
            "all_joined_events_timed": True,
            "local_received_time_present": False,
            "clock_error_bound_present": False,
        },
        "evidence_boundary": {
            "event_to_market_causal_join": False,
            "reaction_latency_measured": False,
            "executable_quote_or_alpha_claim": False,
            "reason": "Historical source timestamps lack local receive time and a measured cross-source clock-error bound.",
        },
        "rows": ordered,
    }
    report["rows_sha256"] = canonical_sha256(ordered)
    return report


def write_dal_det_event_market_source_time_timeline(
    report: Mapping[str, object], *, output_path: str | Path
) -> None:
    """Atomically publish the derived chronology outside immutable raw storage."""

    path = Path(output_path)
    if path.suffix != ".json":
        raise DALDETEventMarketError("event/market timeline output must be JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    "DALDETEventMarketError",
    "build_dal_det_event_market_source_time_timeline",
    "write_dal_det_event_market_source_time_timeline",
]
