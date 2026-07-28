"""Paired historical venue diagnostics and prospective shadow locked pairs.

The historical side consumes actual-trade reaction rows only.  It measures
same-episode Polymarket/Kalshi differences without treating a trade price as a
fillable quote.  The prospective helper only calculates a non-executable
shadow locked-payout quote; it never submits, simulates, or implies an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import pandas as pd


CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
_VENUES = frozenset({"polymarket", "kalshi"})
_PAIR_KEYS = (
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "market_family",
    "outcome_orientation",
    "horizon_seconds",
)
_REQUIRED_REACTION_COLUMNS = frozenset(
    {
        *_PAIR_KEYS,
        "venue",
        "signed_price_change",
        "pre_event_actual_trade",
        "first_post_event_trade",
        "trade_count",
        "volume",
        "staleness_seconds",
        "actual_observation",
        "analysis_eligible",
        "order_ambiguous",
        "contaminated",
        "source_hashes",
        "claim_boundary",
    }
)


class CrossVenueStudyError(ValueError):
    """Cross-venue pairing cannot produce governed evidence."""


@dataclass(frozen=True, slots=True)
class CrossVenuePairV1:
    game_id: str
    episode_id: str
    logical_market_id: str
    outcome: str
    market_family: str
    outcome_orientation: str
    horizon_seconds: int
    polymarket_change: float
    kalshi_change: float
    polymarket_minus_kalshi: float
    polymarket_pre_trade: float
    kalshi_pre_trade: float
    polymarket_trade_count: int
    kalshi_trade_count: int
    polymarket_volume: float
    kalshi_volume: float
    polymarket_staleness_seconds: float
    kalshi_staleness_seconds: float
    source_hashes: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True, slots=True)
class CrossVenuePairResultV1:
    pairs: tuple[CrossVenuePairV1, ...]
    audit: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ShadowLockedPairQuoteV1:
    proposition_id: str
    settlement_rule_sha256: str
    size_contracts: int
    polymarket_yes_ask: float
    kalshi_no_ask: float
    total_cost_per_contract: float | None
    max_quote_age_ns: int
    polymarket_quote_age_ns: int
    kalshi_quote_age_ns: int
    status: str
    executable: bool = False
    claim_boundary: str = "SHADOW_NON_EXECUTABLE"


def _rows(value: Sequence[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    missing = sorted(_REQUIRED_REACTION_COLUMNS - set(frame.columns))
    if missing:
        raise CrossVenueStudyError(
            "reaction rows missing required columns: " + ", ".join(missing)
        )
    if frame.empty:
        return frame
    invalid_venues = sorted(set(frame["venue"]) - _VENUES)
    if invalid_venues:
        raise CrossVenueStudyError(
            "cross-venue rows contain unsupported venue: "
            + ", ".join(str(value) for value in invalid_venues)
        )
    if not frame["claim_boundary"].eq(CLAIM_BOUNDARY).all():
        raise CrossVenueStudyError("historical claim boundary must be SOURCE_TIME_ONLY")
    if frame.duplicated([*_PAIR_KEYS, "venue"]).any():
        raise CrossVenueStudyError("duplicate venue identity in cross-venue rows")
    return frame.sort_values([*_PAIR_KEYS, "venue"], kind="stable").reset_index(drop=True)


def _row_status(row: Mapping[str, Any]) -> str | None:
    if not bool(row["actual_observation"]) or not bool(row["analysis_eligible"]):
        return "NON_ACTUAL_OR_INELIGIBLE"
    if bool(row["order_ambiguous"]):
        return "ORDER_AMBIGUOUS"
    if bool(row["contaminated"]):
        return "CONTAMINATED"
    return None


def build_cross_venue_pairs(
    reactions: Sequence[Mapping[str, object]] | pd.DataFrame,
) -> CrossVenuePairResultV1:
    """Pair actual, eligible historical reactions from the same event.

    A missing or invalid leg is preserved in the audit table.  No historical
    price is converted into a quote, midpoint, fill, lead-lag, or hedge claim.
    """

    frame = _rows(reactions)
    pair_rows: list[CrossVenuePairV1] = []
    audit_rows: list[dict[str, object]] = []
    if frame.empty:
        return CrossVenuePairResultV1(tuple(), pd.DataFrame())
    for pair_key, group in frame.groupby(list(_PAIR_KEYS), sort=True, dropna=False):
        by_venue = {str(row["venue"]): row for _, row in group.iterrows()}
        statuses = {
            venue: _row_status(row)
            for venue, row in by_venue.items()
        }
        identity = dict(zip(_PAIR_KEYS, pair_key, strict=True))
        if "polymarket" not in by_venue:
            audit_rows.append({**identity, "status": "MISSING_POLYMARKET"})
            continue
        if "kalshi" not in by_venue:
            audit_rows.append({**identity, "status": "MISSING_KALSHI"})
            continue
        if statuses["polymarket"] is not None:
            audit_rows.append({**identity, "status": statuses["polymarket"]})
            continue
        if statuses["kalshi"] is not None:
            audit_rows.append({**identity, "status": statuses["kalshi"]})
            continue
        poly = by_venue["polymarket"]
        kalshi = by_venue["kalshi"]
        hashes = tuple(
            sorted(
                {
                    *tuple(poly["source_hashes"]),
                    *tuple(kalshi["source_hashes"]),
                }
            )
        )
        pair_rows.append(
            CrossVenuePairV1(
                **{key: str(value) if key != "horizon_seconds" else int(value) for key, value in identity.items()},
                polymarket_change=float(poly["signed_price_change"]),
                kalshi_change=float(kalshi["signed_price_change"]),
                polymarket_minus_kalshi=(
                    float(poly["signed_price_change"])
                    - float(kalshi["signed_price_change"])
                ),
                polymarket_pre_trade=float(poly["pre_event_actual_trade"]),
                kalshi_pre_trade=float(kalshi["pre_event_actual_trade"]),
                polymarket_trade_count=int(poly["trade_count"]),
                kalshi_trade_count=int(kalshi["trade_count"]),
                polymarket_volume=float(poly["volume"]),
                kalshi_volume=float(kalshi["volume"]),
                polymarket_staleness_seconds=float(poly["staleness_seconds"]),
                kalshi_staleness_seconds=float(kalshi["staleness_seconds"]),
                source_hashes=hashes,
            )
        )
        audit_rows.append({**identity, "status": "PAIRED"})
    return CrossVenuePairResultV1(
        pairs=tuple(pair_rows),
        audit=pd.DataFrame(audit_rows),
    )


def _price(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossVenueStudyError(f"{field} must be a finite probability")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise CrossVenueStudyError(f"{field} must be between zero and one")
    return number


def evaluate_shadow_locked_pair(
    *,
    proposition_id: str,
    settlement_rule_sha256: str,
    polymarket_yes_ask: float,
    kalshi_no_ask: float,
    polymarket_fee: float,
    kalshi_fee: float,
    polymarket_slippage: float,
    kalshi_slippage: float,
    polymarket_quote_age_ns: int,
    kalshi_quote_age_ns: int,
    max_quote_age_ns: int,
    size_contracts: int,
) -> ShadowLockedPairQuoteV1:
    """Evaluate one non-executable same-proposition shadow locked pair."""

    if type(proposition_id) is not str or not proposition_id:
        raise CrossVenueStudyError("proposition_id is required")
    if type(settlement_rule_sha256) is not str or not settlement_rule_sha256.startswith("sha256:"):
        raise CrossVenueStudyError("settlement_rule_sha256 is required")
    if type(size_contracts) is not int or size_contracts < 1:
        raise CrossVenueStudyError("size_contracts must be a positive integer")
    for value, field in (
        (polymarket_yes_ask, "polymarket_yes_ask"),
        (kalshi_no_ask, "kalshi_no_ask"),
        (polymarket_fee, "polymarket_fee"),
        (kalshi_fee, "kalshi_fee"),
        (polymarket_slippage, "polymarket_slippage"),
        (kalshi_slippage, "kalshi_slippage"),
    ):
        _price(value, field)
    for value, field in (
        (polymarket_quote_age_ns, "polymarket_quote_age_ns"),
        (kalshi_quote_age_ns, "kalshi_quote_age_ns"),
        (max_quote_age_ns, "max_quote_age_ns"),
    ):
        if type(value) is not int or value < 0:
            raise CrossVenueStudyError(f"{field} must be a non-negative integer")
    if max(polymarket_quote_age_ns, kalshi_quote_age_ns) > max_quote_age_ns:
        return ShadowLockedPairQuoteV1(
            proposition_id=proposition_id,
            settlement_rule_sha256=settlement_rule_sha256,
            size_contracts=size_contracts,
            polymarket_yes_ask=float(polymarket_yes_ask),
            kalshi_no_ask=float(kalshi_no_ask),
            total_cost_per_contract=None,
            max_quote_age_ns=max_quote_age_ns,
            polymarket_quote_age_ns=polymarket_quote_age_ns,
            kalshi_quote_age_ns=kalshi_quote_age_ns,
            status="STALE_QUOTE",
        )
    total = sum(
        (
            float(polymarket_yes_ask),
            float(kalshi_no_ask),
            float(polymarket_fee),
            float(kalshi_fee),
            float(polymarket_slippage),
            float(kalshi_slippage),
        )
    )
    return ShadowLockedPairQuoteV1(
        proposition_id=proposition_id,
        settlement_rule_sha256=settlement_rule_sha256,
        size_contracts=size_contracts,
        polymarket_yes_ask=float(polymarket_yes_ask),
        kalshi_no_ask=float(kalshi_no_ask),
        total_cost_per_contract=total,
        max_quote_age_ns=max_quote_age_ns,
        polymarket_quote_age_ns=polymarket_quote_age_ns,
        kalshi_quote_age_ns=kalshi_quote_age_ns,
        status="LOCKED_PAYOUT_SHADOW" if total < 1.0 else "NO_LOCKED_PAYOUT",
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "CrossVenuePairResultV1",
    "CrossVenuePairV1",
    "CrossVenueStudyError",
    "ShadowLockedPairQuoteV1",
    "build_cross_venue_pairs",
    "evaluate_shadow_locked_pair",
]
