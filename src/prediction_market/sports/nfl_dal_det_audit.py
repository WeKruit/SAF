"""Fail-closed evidence audit for the 2025 Dallas at Detroit vertical slice.

The module intentionally audits *what the historical source objects prove*.
It does not infer latency, quote executability, a reaction, or an alpha from
timestamps that lack an availability clock and an error bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prediction_market.contracts import canonical_json_bytes
from prediction_market.sports.nfl_game_replay import replay_frozen_nfl_game
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)


NATIVE_GAME_ID = "2025_14_DAL_DET"
CANONICAL_GAME_ID = "game_nflverse_2025_14_DAL_DET"


class DALDETAuditError(ValueError):
    """One-game evidence is malformed, incomplete, or not byte-verifiable."""


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DALDETAuditError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[object]:
    if type(value) is not list:
        raise DALDETAuditError(f"{field} must be a list")
    return value


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 71 or not value.startswith("sha256:"):
        raise DALDETAuditError(f"{field} must be a SHA-256 reference")
    hex_value = value.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in hex_value):
        raise DALDETAuditError(f"{field} must be a SHA-256 reference")
    return value


def _manifest_for_sha256(store_root: Path, manifest_sha256: str) -> Path:
    digest = _require_sha256(manifest_sha256, "manifest_sha256").removeprefix(
        "sha256:"
    )
    matches = tuple((store_root / "manifests").glob(f"**/{digest}.manifest.json"))
    if len(matches) != 1:
        raise DALDETAuditError(
            f"manifest {manifest_sha256} must resolve to exactly one object"
        )
    return matches[0]


def _verified_json(
    *, program_root: Path, store_root: Path, manifest_sha256: str
) -> tuple[Mapping[str, object] | list[object], dict[str, Any]]:
    manifest_path = _manifest_for_sha256(store_root, manifest_sha256)
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=program_root,
        )
    except StaticStoreError as error:
        raise DALDETAuditError(
            f"manifest {manifest_sha256} failed immutable-object verification"
        ) from error
    try:
        decoded = json.loads(verified.object_bytes)
    except json.JSONDecodeError as error:
        raise DALDETAuditError(f"manifest {manifest_sha256} is not JSON") from error
    if type(decoded) not in {dict, list}:
        raise DALDETAuditError(f"manifest {manifest_sha256} JSON root is unsupported")
    return decoded, {
        "manifest_sha256": verified.record.manifest.manifest_sha256,
        "object_sha256": verified.record.manifest.object_sha256,
        "dataset_id": verified.record.manifest.dataset_id,
        "license_status": verified.record.manifest.license_status,
        "source_url": verified.record.manifest.source_url,
        "coverage": verified.record.manifest.coverage,
    }


def _utc_ns(value: object, field: str) -> int:
    if type(value) is not str or not value.endswith("Z"):
        raise DALDETAuditError(f"{field} must be a UTC RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise DALDETAuditError(f"{field} is not a UTC RFC3339 timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise DALDETAuditError(f"{field} must be UTC")
    return int(parsed.timestamp() * 1_000_000_000)


def _nondecreasing_breaks(values: Sequence[int]) -> int:
    return sum(later < earlier for earlier, later in zip(values, values[1:]))


def _polymarket_summary(
    payload: object, *, expected_condition_id: str
) -> dict[str, Any]:
    trades = _require_list(payload, "Polymarket trades")
    if not trades:
        raise DALDETAuditError("Polymarket trades must be nonempty")
    timestamps: list[int] = []
    identities: set[str] = set()
    assets: set[str] = set()
    for ordinal, untrusted_trade in enumerate(trades):
        trade = _require_mapping(untrusted_trade, f"Polymarket trade {ordinal}")
        if trade.get("conditionId") != expected_condition_id:
            raise DALDETAuditError("Polymarket trade condition mapping changed")
        timestamp = trade.get("timestamp")
        if type(timestamp) is not int or timestamp <= 0:
            raise DALDETAuditError("Polymarket trade timestamp must be positive seconds")
        asset = trade.get("asset")
        if type(asset) is not str or not asset:
            raise DALDETAuditError("Polymarket trade asset is absent")
        # Raw API JSON can legitimately contain binary floats.  This identity
        # check is deliberately a raw-record check, not a contract hash.
        identity = "sha256:" + hashlib.sha256(
            json.dumps(
                dict(trade),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if identity in identities:
            raise DALDETAuditError("Polymarket historical trade response is duplicated")
        identities.add(identity)
        assets.add(asset)
        timestamps.append(timestamp)
    return {
        "record_count": len(trades),
        "unique_record_count": len(identities),
        "asset_count": len(assets),
        "source_timestamp": {
            "field": "timestamp",
            "unit": "epoch_seconds",
            "minimum_utc_ns": min(timestamps) * 1_000_000_000,
            "maximum_utc_ns": max(timestamps) * 1_000_000_000,
            "native_order_nonmonotonic_pairs": _nondecreasing_breaks(timestamps),
            "clock_error_bound_ns": None,
            "availability_time_present": False,
        },
        "observation_boundary": "trade_records_only_not_l1_or_l2",
    }


def _kalshi_trade_summary(
    *, payloads: Sequence[object], ticker: str
) -> dict[str, Any]:
    timestamps: list[int] = []
    identities: set[str] = set()
    record_count = 0
    for page_ordinal, payload in enumerate(payloads):
        page = _require_mapping(payload, f"Kalshi page {page_ordinal}")
        for trade_ordinal, untrusted_trade in enumerate(
            _require_list(page.get("trades"), "Kalshi trades")
        ):
            trade = _require_mapping(
                untrusted_trade, f"Kalshi trade {page_ordinal}:{trade_ordinal}"
            )
            if trade.get("ticker") != ticker:
                raise DALDETAuditError("Kalshi trade ticker mapping changed")
            trade_id = trade.get("trade_id")
            if type(trade_id) is not str or not trade_id:
                raise DALDETAuditError("Kalshi trade id is absent")
            if trade_id in identities:
                raise DALDETAuditError("Kalshi historical trade chain has duplicate ID")
            identities.add(trade_id)
            timestamps.append(_utc_ns(trade.get("created_time"), "created_time"))
            record_count += 1
    if not record_count:
        raise DALDETAuditError("Kalshi trade chain must be nonempty")
    return {
        "ticker": ticker,
        "record_count": record_count,
        "unique_trade_id_count": len(identities),
        "source_timestamp": {
            "field": "created_time",
            "unit": "UTC RFC3339 microsecond text",
            "minimum_utc_ns": min(timestamps),
            "maximum_utc_ns": max(timestamps),
            "clock_error_bound_ns": None,
            "availability_time_present": False,
        },
        "observation_boundary": "trade_records_only_not_l2",
    }


def _kalshi_candle_summary(payload: object, *, ticker: str) -> dict[str, Any]:
    decoded = _require_mapping(payload, "Kalshi candle payload")
    candles = _require_list(decoded.get("candlesticks"), "Kalshi candlesticks")
    if not candles:
        raise DALDETAuditError("Kalshi candlesticks must be nonempty")
    endpoints: list[int] = []
    for ordinal, untrusted_candle in enumerate(candles):
        candle = _require_mapping(untrusted_candle, f"Kalshi candle {ordinal}")
        endpoint = candle.get("end_period_ts")
        if type(endpoint) is not int or endpoint <= 0:
            raise DALDETAuditError("Kalshi candle end_period_ts must be positive")
        endpoints.append(endpoint)
    if any(later - earlier != 60 for earlier, later in zip(endpoints, endpoints[1:])):
        raise DALDETAuditError("Kalshi candles must be contiguous one-minute periods")
    return {
        "ticker": ticker,
        "record_count": len(candles),
        "source_timestamp": {
            "field": "end_period_ts",
            "unit": "epoch_seconds_interval_end",
            "minimum_utc_ns": min(endpoints) * 1_000_000_000,
            "maximum_utc_ns": max(endpoints) * 1_000_000_000,
            "interval_ns": 60 * 1_000_000_000,
            "clock_error_bound_ns": None,
            "availability_time_present": False,
        },
        "observation_boundary": "minute_interval_end_bid_ask_not_event_time_l1_or_l2",
    }


def build_dal_det_evidence_audit(
    *, program_root: str | Path, mapping: Mapping[str, object]
) -> dict[str, Any]:
    """Verify one game's immutable inputs and state the reachable conclusion.

    A missing clock-error bound deliberately blocks source-time reaction
    measurement.  This makes the report useful for a future approved timing
    source without reinterpreting this historical archive as local capture.
    """

    root = Path(program_root).resolve()
    store_root = root / "var" / "raw"
    if mapping.get("canonical_game_id") != CANONICAL_GAME_ID:
        raise DALDETAuditError("market mapping has the wrong canonical game id")
    raw_capture = _require_mapping(mapping.get("raw_capture"), "raw_capture")
    polymarket_capture = _require_mapping(raw_capture.get("polymarket"), "polymarket")
    event_manifest = _require_sha256(
        polymarket_capture.get("event_manifest_sha256"), "Polymarket event manifest"
    )
    trades_manifest = _require_sha256(
        polymarket_capture.get("trades_manifest_sha256"), "Polymarket trades manifest"
    )
    event_payload, event_evidence = _verified_json(
        program_root=root, store_root=store_root, manifest_sha256=event_manifest
    )
    if type(event_payload) is not list or len(event_payload) != 1:
        raise DALDETAuditError("Polymarket event response must contain one event")
    event = _require_mapping(event_payload[0], "Polymarket event")
    venues = _require_mapping(mapping.get("venues"), "venues")
    polymarket_venue = _require_mapping(venues.get("polymarket"), "venues.polymarket")
    if event.get("slug") != polymarket_venue.get("event_slug"):
        raise DALDETAuditError("Polymarket event slug does not match mapping")
    condition_id = polymarket_venue.get("condition_id")
    if type(condition_id) is not str or not condition_id:
        raise DALDETAuditError("Polymarket condition id is absent")
    trade_payload, trades_evidence = _verified_json(
        program_root=root, store_root=store_root, manifest_sha256=trades_manifest
    )
    polymarket = {
        "event_evidence": event_evidence,
        "trades_evidence": trades_evidence,
        "trade_inventory": _polymarket_summary(
            trade_payload, expected_condition_id=condition_id
        ),
    }

    kalshi_capture = _require_mapping(raw_capture.get("kalshi"), "kalshi")
    trade_chains = _require_mapping(kalshi_capture.get("trade_chains"), "trade_chains")
    candle_captures = _require_mapping(
        kalshi_capture.get("candle_captures"), "candle_captures"
    )
    kalshi: dict[str, Any] = {
        "markets_manifest_sha256": _require_sha256(
            kalshi_capture.get("markets_manifest_sha256"), "Kalshi markets manifest"
        ),
        "contracts": {},
    }
    market_payload, markets_evidence = _verified_json(
        program_root=root,
        store_root=store_root,
        manifest_sha256=kalshi["markets_manifest_sha256"],
    )
    kalshi["markets_evidence"] = markets_evidence
    _require_mapping(market_payload, "Kalshi markets payload")
    for ticker in sorted(trade_chains):
        chain = _require_mapping(trade_chains[ticker], f"trade chain {ticker}")
        manifests = _require_list(chain.get("manifest_sha256s"), f"{ticker} manifests")
        if chain.get("complete") is not True or not manifests:
            raise DALDETAuditError(f"Kalshi trade chain {ticker} is incomplete")
        pages: list[object] = []
        page_evidence: list[dict[str, Any]] = []
        for manifest in manifests:
            payload, evidence = _verified_json(
                program_root=root,
                store_root=store_root,
                manifest_sha256=_require_sha256(manifest, f"{ticker} manifest"),
            )
            pages.append(payload)
            page_evidence.append(evidence)
        candle = _require_mapping(candle_captures.get(ticker), f"candle {ticker}")
        candle_payload, candle_evidence = _verified_json(
            program_root=root,
            store_root=store_root,
            manifest_sha256=_require_sha256(
                candle.get("manifest_sha256"), f"{ticker} candle manifest"
            ),
        )
        trade_inventory = _kalshi_trade_summary(payloads=pages, ticker=ticker)
        expected_count = chain.get("trades")
        if expected_count != trade_inventory["record_count"]:
            raise DALDETAuditError(f"Kalshi trade count mismatch for {ticker}")
        kalshi["contracts"][ticker] = {
            "trade_page_evidence": page_evidence,
            "trade_inventory": trade_inventory,
            "candle_evidence": candle_evidence,
            "candle_inventory": _kalshi_candle_summary(
                candle_payload, ticker=ticker
            ),
        }

    replay = replay_frozen_nfl_game(program_root=root, native_game_id=NATIVE_GAME_ID)
    trace_times = [
        _utc_ns(step["source_time_utc"], "NFL source_time_utc")
        for step in replay.steps
        if step["source_time_utc"] is not None
    ]
    kalshi_manifest_count = 1 + sum(
        len(_require_list(_require_mapping(chain, f"trade chain {ticker}").get("manifest_sha256s"), f"{ticker} manifests"))
        for ticker, chain in trade_chains.items()
    ) + len(candle_captures)
    return {
        "artifact_type": "nfl_dal_det_evidence_audit",
        "artifact_version": "v0",
        "canonical_game_id": CANONICAL_GAME_ID,
        "native_game_id": NATIVE_GAME_ID,
        "game_state": {
            "dataset_id": replay.dataset_id,
            "source_manifest_sha256": replay.source_manifest_sha256,
            "source_object_sha256": replay.source_object_sha256,
            "trace_sha256": replay.trace_sha256,
            "final_state_sha256": replay.final_state_sha256,
            "source_row_count": replay.source_row_count,
            "transition_count": replay.transition_count,
            "source_timestamp": {
                "field": "nflverse.time_of_day",
                "present_transition_count": len(trace_times),
                "missing_transition_count": len(replay.steps) - len(trace_times),
                "minimum_utc_ns": min(trace_times),
                "maximum_utc_ns": max(trace_times),
                "native_order_nonmonotonic_pairs": _nondecreasing_breaks(trace_times),
                "clock_error_bound_ns": None,
                "availability_time_present": False,
            },
        },
        "immutable_input_verification": {
            "verified_manifest_count": 1 + 2 + kalshi_manifest_count,
            "by_source": {
                "nflverse": 1,
                "polymarket": 2,
                "kalshi": kalshi_manifest_count,
            },
        },
        "market_data": {"polymarket": polymarket, "kalshi": kalshi},
        "time_join_gate": {
            "status": "BLOCKED",
            "reason_codes": [
                "GAME_SOURCE_CLOCK_ERROR_BOUND_UNRESOLVED",
                "MARKET_SOURCE_CLOCK_ERROR_BOUND_UNRESOLVED",
                "HISTORICAL_LOCAL_RECEIVE_TIME_ABSENT",
                "GAME_SOURCE_TIMESTAMP_SEMANTICS_NOT_PIT_PROVEN",
            ],
            "allowed_claim": "immutable source objects are bound to one game and their native timestamp coverage is audited",
            "forbidden_claims": [
                "reaction_latency",
                "event_to_market_causality",
                "overreaction_or_correction",
                "bid_ask_executability",
                "alpha_or_execution",
            ],
        },
        "official_gamebook": {
            "status": "BLOCKED_NOT_INGESTED",
            "reason": "official PDF states non-media use requires written NFL permission; no registered permission or dataset license exists",
            "effect": "not stored in immutable raw, not supplied to an LLM, and not used as a formal validation source",
        },
        "formal_result_eligible": False,
    }


def write_dal_det_evidence_audit(
    report: Mapping[str, object], *, output_path: str | Path
) -> None:
    """Atomically write a derived audit outside immutable raw storage."""

    path = Path(output_path)
    if path.suffix != ".json":
        raise DALDETAuditError("audit output must be JSON")
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
    "CANONICAL_GAME_ID",
    "DALDETAuditError",
    "NATIVE_GAME_ID",
    "build_dal_det_evidence_audit",
    "write_dal_det_evidence_audit",
]
