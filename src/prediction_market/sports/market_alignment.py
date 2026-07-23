"""Fail-closed game-state probability to executable-market alignment.

This module performs one narrow comparison: an exact canonical sport outcome
probability at a model point-in-time versus an executable bid/ask observation
known at that same point-in-time.  It does not estimate profitability,
execution, or causality.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import re
from typing import Literal


_CANONICAL_PATTERNS = {
    "game_id": re.compile(r"^game_[A-Za-z0-9][A-Za-z0-9._:-]*$"),
    "condition_id": re.compile(
        r"^condition_[A-Za-z0-9][A-Za-z0-9._:-]*$"
    ),
    "outcome_id": re.compile(r"^outcome_[A-Za-z0-9][A-Za-z0-9._:-]*$"),
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ZERO = Decimal("0")
_ONE = Decimal("1")


class MarketAlignmentInputError(ValueError):
    """Raised when an alignment input cannot be represented exactly."""


def _require_canonical_id(value: object, field: str) -> None:
    if type(value) is not str or _CANONICAL_PATTERNS[field].fullmatch(value) is None:
        raise MarketAlignmentInputError(f"{field} must be a canonical {field}")


def _require_sha256(value: object, field: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MarketAlignmentInputError(f"{field} must be sha256:<64 lowercase hex>")


def _require_utc(value: object, field: str) -> None:
    if type(value) is not datetime:
        raise MarketAlignmentInputError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise MarketAlignmentInputError(f"{field} must be timezone-aware UTC")


def _require_probability(value: object, field: str) -> None:
    if type(value) is float:
        raise MarketAlignmentInputError(
            f"{field} uses a binary float; use Decimal for exact comparison"
        )
    if type(value) is not Decimal:
        raise MarketAlignmentInputError(f"{field} must be Decimal")
    if not value.is_finite() or value < _ZERO or value > _ONE:
        raise MarketAlignmentInputError(f"{field} must be finite and in [0, 1]")


def _require_depth(value: object | None, field: str) -> None:
    if value is None:
        return
    if type(value) is float:
        raise MarketAlignmentInputError(
            f"{field} uses a binary float; use Decimal for exact comparison"
        )
    if type(value) is not Decimal:
        raise MarketAlignmentInputError(f"{field} must be Decimal or None")
    if not value.is_finite() or value < _ZERO:
        raise MarketAlignmentInputError(f"{field} must be finite and nonnegative")


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


@dataclass(frozen=True, slots=True)
class CanonicalGameConditionBinding:
    """Point-in-time proof that one venue condition represents one sport outcome."""

    game_id: str
    condition_id: str
    outcome_id: str
    metadata_snapshot_ref: str
    metadata_observed_at: datetime

    def __post_init__(self) -> None:
        _require_canonical_id(self.game_id, "game_id")
        _require_canonical_id(self.condition_id, "condition_id")
        _require_canonical_id(self.outcome_id, "outcome_id")
        _require_sha256(self.metadata_snapshot_ref, "metadata_snapshot_ref")
        _require_utc(self.metadata_observed_at, "metadata_observed_at")


@dataclass(frozen=True, slots=True)
class ModelProbabilityObservation:
    """One exact outcome probability emitted from information available by cutoff."""

    game_id: str
    outcome_id: str
    cutoff_at: datetime
    probability: Decimal
    model_output_ref: str

    def __post_init__(self) -> None:
        _require_canonical_id(self.game_id, "game_id")
        _require_canonical_id(self.outcome_id, "outcome_id")
        _require_utc(self.cutoff_at, "cutoff_at")
        _require_probability(self.probability, "probability")
        _require_sha256(self.model_output_ref, "model_output_ref")


@dataclass(frozen=True, slots=True)
class ExecutableQuoteObservation:
    """Top-of-book quote and the PIT evidence required to interpret it."""

    game_id: str
    condition_id: str
    outcome_id: str
    received_at: datetime
    best_bid: Decimal | None
    best_ask: Decimal | None
    bid_depth: Decimal | None
    ask_depth: Decimal | None
    midpoint: Decimal | None
    paused: bool
    metadata_snapshot_ref: str
    rule_snapshot_ref: str | None
    rule_snapshot_observed_at: datetime | None

    def __post_init__(self) -> None:
        _require_canonical_id(self.game_id, "game_id")
        _require_canonical_id(self.condition_id, "condition_id")
        _require_canonical_id(self.outcome_id, "outcome_id")
        _require_utc(self.received_at, "received_at")
        for field in ("best_bid", "best_ask", "midpoint"):
            value = getattr(self, field)
            if value is not None:
                _require_probability(value, field)
        _require_depth(self.bid_depth, "bid_depth")
        _require_depth(self.ask_depth, "ask_depth")
        if type(self.paused) is not bool:
            raise MarketAlignmentInputError("paused must be bool")
        _require_sha256(self.metadata_snapshot_ref, "metadata_snapshot_ref")
        if self.rule_snapshot_ref is not None:
            _require_sha256(self.rule_snapshot_ref, "rule_snapshot_ref")
        if self.rule_snapshot_observed_at is not None:
            _require_utc(
                self.rule_snapshot_observed_at,
                "rule_snapshot_observed_at",
            )


AlignmentStatus = Literal[
    "alignment_ready",
    "not_aligned",
    "predictive_disagreement",
]


@dataclass(frozen=True, slots=True)
class AlignmentDecision:
    """Machine-readable result with intentionally non-trading semantics."""

    status: AlignmentStatus
    reason_codes: tuple[str, ...]
    as_of_age_ms: int | None
    spread: Decimal | None
    probability_distance_to_executable_interval: Decimal | None
    comparison_basis: Literal["executable_bid_ask"] = "executable_bid_ask"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "comparison_basis": self.comparison_basis,
            "as_of_age_ms": self.as_of_age_ms,
            "spread": _decimal_text(self.spread),
            "probability_distance_to_executable_interval": _decimal_text(
                self.probability_distance_to_executable_interval
            ),
        }


def _elapsed_microseconds(later: datetime, earlier: datetime) -> int:
    delta = later - earlier
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def evaluate_market_alignment(
    *,
    binding: CanonicalGameConditionBinding,
    model: ModelProbabilityObservation,
    quote: ExecutableQuoteObservation,
    max_quote_age_ms: int,
) -> AlignmentDecision:
    """Compare only when canonical identity, PIT time, and execution facts all hold."""

    if (
        type(max_quote_age_ms) is not int
        or max_quote_age_ms < 0
    ):
        raise MarketAlignmentInputError(
            "max_quote_age_ms must be a nonnegative integer"
        )

    reasons: list[str] = []
    if model.game_id != binding.game_id:
        reasons.append("model_game_mismatch")
    if model.outcome_id != binding.outcome_id:
        reasons.append("model_outcome_mismatch")
    if quote.game_id != binding.game_id:
        reasons.append("quote_game_mismatch")
    if quote.condition_id != binding.condition_id:
        reasons.append("quote_condition_mismatch")
    if quote.outcome_id != binding.outcome_id:
        reasons.append("quote_outcome_mismatch")
    if quote.metadata_snapshot_ref != binding.metadata_snapshot_ref:
        reasons.append("metadata_snapshot_mismatch")
    if binding.metadata_observed_at > model.cutoff_at:
        reasons.append("future_metadata_snapshot")

    age_microseconds: int | None
    if quote.received_at > model.cutoff_at:
        reasons.append("future_quote")
        age_microseconds = None
    else:
        age_microseconds = _elapsed_microseconds(
            model.cutoff_at,
            quote.received_at,
        )
        if age_microseconds > max_quote_age_ms * 1_000:
            reasons.append("stale_quote")

    if quote.rule_snapshot_ref is None:
        reasons.append("missing_rule_snapshot")
    if quote.rule_snapshot_observed_at is None:
        if quote.rule_snapshot_ref is not None:
            reasons.append("missing_rule_snapshot_observed_at")
    elif quote.rule_snapshot_observed_at > model.cutoff_at:
        reasons.append("future_rule_snapshot")
    if quote.paused:
        reasons.append("market_paused")

    midpoint_only = (
        quote.best_bid is None
        and quote.best_ask is None
        and quote.midpoint is not None
    )
    if midpoint_only:
        reasons.append("midpoint_only_forbidden")
    else:
        if quote.best_bid is None:
            reasons.append("missing_executable_bid")
        elif quote.bid_depth is None or quote.bid_depth <= _ZERO:
            reasons.append("non_executable_bid_depth")
        if quote.best_ask is None:
            reasons.append("missing_executable_ask")
        elif quote.ask_depth is None or quote.ask_depth <= _ZERO:
            reasons.append("non_executable_ask_depth")

    if (
        quote.best_bid is not None
        and quote.best_ask is not None
        and quote.best_bid > quote.best_ask
    ):
        reasons.append("crossed_book")

    age_ms = (
        None
        if age_microseconds is None
        else age_microseconds // 1_000
    )
    if reasons:
        return AlignmentDecision(
            status="not_aligned",
            reason_codes=tuple(reasons),
            as_of_age_ms=age_ms,
            spread=None,
            probability_distance_to_executable_interval=None,
        )

    assert quote.best_bid is not None
    assert quote.best_ask is not None
    spread = quote.best_ask - quote.best_bid
    if model.probability < quote.best_bid:
        distance = quote.best_bid - model.probability
    elif model.probability > quote.best_ask:
        distance = model.probability - quote.best_ask
    else:
        distance = _ZERO
    return AlignmentDecision(
        status=(
            "predictive_disagreement"
            if distance > _ZERO
            else "alignment_ready"
        ),
        reason_codes=(),
        as_of_age_ms=age_ms,
        spread=spread,
        probability_distance_to_executable_interval=distance,
    )


__all__ = [
    "AlignmentDecision",
    "AlignmentStatus",
    "CanonicalGameConditionBinding",
    "ExecutableQuoteObservation",
    "MarketAlignmentInputError",
    "ModelProbabilityObservation",
    "evaluate_market_alignment",
]
