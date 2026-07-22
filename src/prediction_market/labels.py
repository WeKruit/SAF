"""Executable-quote barrier labels for the Team E X-05 specification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import groupby
from pathlib import Path
from typing import Iterable, Literal, Sequence

from prediction_market.contracts import FixedPointV0
from prediction_market.experiments import load_experiment_registry


QuoteState = Literal["ACTIVE", "SUSPENDED"]
TouchRule = Literal["UPPER_FIRST", "LOWER_FIRST", "AMBIGUOUS"]
OverlapRule = Literal["KEEP_GROUPED", "DROP_LATER"]
LabelOutcome = Literal["UPPER", "LOWER", "HORIZON", "AMBIGUOUS"]
_QUOTE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class LabelInputError(ValueError):
    """Quote data or locked label parameters are invalid."""


class LabelAuthorizationError(PermissionError):
    """The experiment registry does not authorize X-05 label generation."""


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise LabelInputError(f"{field} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class QuoteV0:
    """One point-in-time quote or explicit market-suspension observation."""

    quote_id: str
    source_at: datetime
    received_at: datetime
    state: QuoteState
    bid: FixedPointV0 | None
    ask: FixedPointV0 | None

    def __post_init__(self) -> None:
        if not isinstance(self.quote_id, str) or not _QUOTE_ID.fullmatch(self.quote_id):
            raise LabelInputError("quote_id must be canonical and nonempty")
        _require_utc(self.source_at, "source_at")
        _require_utc(self.received_at, "received_at")
        if self.source_at > self.received_at:
            raise LabelInputError("source_at cannot follow received_at")
        if self.state not in {"ACTIVE", "SUSPENDED"}:
            raise LabelInputError("state must be ACTIVE or SUSPENDED")
        if self.state == "SUSPENDED":
            if self.bid is not None or self.ask is not None:
                raise LabelInputError("SUSPENDED quotes cannot be executable")
            return
        if not isinstance(self.bid, FixedPointV0) or not isinstance(
            self.ask, FixedPointV0
        ):
            raise LabelInputError("ACTIVE quote requires fixed-point bid and ask")
        bid = self.bid.to_decimal()
        ask = self.ask.to_decimal()
        if not (Decimal(0) <= bid <= ask <= Decimal(1)):
            raise LabelInputError("ACTIVE quote requires 0 <= bid <= ask <= 1")


@dataclass(frozen=True, slots=True)
class BarrierLabelParameters:
    """All X-05 choices are required; v0 intentionally has no defaults."""

    upper_return: Decimal
    lower_return: Decimal
    horizon: timedelta
    max_quote_age: timedelta
    same_time_touch_rule: TouchRule
    overlap_rule: OverlapRule
    purge: timedelta
    embargo: timedelta

    def __post_init__(self) -> None:
        for field, value in (
            ("upper_return", self.upper_return),
            ("lower_return", self.lower_return),
        ):
            if not isinstance(value, Decimal) or not value.is_finite():
                raise LabelInputError(f"{field} must be a finite Decimal")
            if not Decimal(0) < value < Decimal(1):
                raise LabelInputError(f"{field} must be between zero and one")
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise LabelInputError("horizon must be a positive timedelta")
        for field, value in (
            ("max_quote_age", self.max_quote_age),
            ("purge", self.purge),
            ("embargo", self.embargo),
        ):
            if not isinstance(value, timedelta) or value < timedelta(0):
                raise LabelInputError(f"{field} must be a nonnegative timedelta")
        if self.same_time_touch_rule not in {
            "UPPER_FIRST",
            "LOWER_FIRST",
            "AMBIGUOUS",
        }:
            raise LabelInputError("invalid same_time_touch_rule")
        if self.overlap_rule not in {"KEEP_GROUPED", "DROP_LATER"}:
            raise LabelInputError("invalid overlap_rule")


@dataclass(frozen=True, slots=True)
class BarrierLabel:
    anchor_at: datetime
    entry_at: datetime
    exit_at: datetime
    entry_quote_id: str
    exit_quote_id: str | None
    entry_price: FixedPointV0
    exit_price: FixedPointV0 | None
    upper_price: FixedPointV0
    lower_price: FixedPointV0
    outcome: LabelOutcome
    resume_quote_ids: tuple[str, ...]
    window_end_at: datetime


def _eligible(quote: QuoteV0, parameters: BarrierLabelParameters) -> bool:
    return (
        quote.state == "ACTIVE"
        and quote.received_at - quote.source_at <= parameters.max_quote_age
    )


def _from_decimal(value: Decimal) -> FixedPointV0:
    # Multiplication of finite fixed-point decimals stays finite and never uses a
    # binary float.  FixedPointV0 preserves the resulting decimal exponent.
    return FixedPointV0.from_value(value)


def _ordered_quotes(quotes: Iterable[QuoteV0]) -> list[QuoteV0]:
    ordered = list(quotes)
    if any(not isinstance(quote, QuoteV0) for quote in ordered):
        raise LabelInputError("quotes must contain QuoteV0 values")
    ids = [quote.quote_id for quote in ordered]
    if len(set(ids)) != len(ids):
        raise LabelInputError("duplicate quote_id")
    return sorted(ordered, key=lambda quote: (quote.received_at, quote.quote_id))


def label_long(
    quotes: Iterable[QuoteV0],
    *,
    anchor_at: datetime,
    parameters: BarrierLabelParameters,
) -> BarrierLabel:
    """Label one long entry using ask-in and bid-out executable prices.

    The horizon is measured from ``anchor_at``.  Barrier observations at the
    exact deadline are evaluated before the horizon exit.  A suspension makes
    all quotes in that timestamp group ineligible; the first later non-stale,
    two-sided quote is the executable resume quote.
    """

    _require_utc(anchor_at, "anchor_at")
    if not isinstance(parameters, BarrierLabelParameters):
        raise LabelInputError("parameters must be BarrierLabelParameters")
    ordered = _ordered_quotes(quotes)
    deadline = anchor_at + parameters.horizon

    resume_pending = False
    resume_ids: list[str] = []
    entry_index: int | None = None
    entry: QuoteV0 | None = None
    for index, quote in enumerate(ordered):
        if quote.state == "SUSPENDED":
            resume_pending = True
            continue
        if not _eligible(quote, parameters):
            continue
        if quote.received_at < anchor_at:
            resume_pending = False
            continue
        if quote.received_at > deadline:
            break
        entry_index = index
        entry = quote
        if resume_pending:
            resume_ids.append(quote.quote_id)
            resume_pending = False
        break
    if entry is None or entry_index is None or entry.ask is None:
        raise LabelInputError("no executable ask entry within horizon")

    entry_decimal = entry.ask.to_decimal()
    upper_decimal = entry_decimal * (Decimal(1) + parameters.upper_return)
    lower_decimal = entry_decimal * (Decimal(1) - parameters.lower_return)
    upper = _from_decimal(upper_decimal)
    lower = _from_decimal(lower_decimal)

    post_entry = ordered[entry_index + 1 :]
    for received_at, group_iter in groupby(
        post_entry, key=lambda quote: quote.received_at
    ):
        group = list(group_iter)
        if any(quote.state == "SUSPENDED" for quote in group):
            resume_pending = True
            continue
        eligible = [quote for quote in group if _eligible(quote, parameters)]
        if not eligible:
            continue
        if resume_pending:
            resume_ids.append(eligible[0].quote_id)
            resume_pending = False

        if received_at <= deadline:
            upper_hits = [
                quote
                for quote in eligible
                if quote.bid is not None and quote.bid.to_decimal() >= upper_decimal
            ]
            lower_hits = [
                quote
                for quote in eligible
                if quote.bid is not None and quote.bid.to_decimal() <= lower_decimal
            ]
            if upper_hits and lower_hits:
                if parameters.same_time_touch_rule == "AMBIGUOUS":
                    return BarrierLabel(
                        anchor_at=anchor_at,
                        entry_at=entry.received_at,
                        exit_at=received_at,
                        entry_quote_id=entry.quote_id,
                        exit_quote_id=None,
                        entry_price=entry.ask,
                        exit_price=None,
                        upper_price=upper,
                        lower_price=lower,
                        outcome="AMBIGUOUS",
                        resume_quote_ids=tuple(resume_ids),
                        window_end_at=received_at,
                    )
                chosen = (
                    upper_hits[0]
                    if parameters.same_time_touch_rule == "UPPER_FIRST"
                    else lower_hits[0]
                )
                outcome: LabelOutcome = (
                    "UPPER"
                    if parameters.same_time_touch_rule == "UPPER_FIRST"
                    else "LOWER"
                )
                return BarrierLabel(
                    anchor_at=anchor_at,
                    entry_at=entry.received_at,
                    exit_at=received_at,
                    entry_quote_id=entry.quote_id,
                    exit_quote_id=chosen.quote_id,
                    entry_price=entry.ask,
                    exit_price=chosen.bid,
                    upper_price=upper,
                    lower_price=lower,
                    outcome=outcome,
                    resume_quote_ids=tuple(resume_ids),
                    window_end_at=received_at,
                )
            hits = (upper_hits, "UPPER") if upper_hits else (lower_hits, "LOWER")
            if hits[0]:
                chosen = hits[0][0]
                return BarrierLabel(
                    anchor_at=anchor_at,
                    entry_at=entry.received_at,
                    exit_at=received_at,
                    entry_quote_id=entry.quote_id,
                    exit_quote_id=chosen.quote_id,
                    entry_price=entry.ask,
                    exit_price=chosen.bid,
                    upper_price=upper,
                    lower_price=lower,
                    outcome=hits[1],  # type: ignore[arg-type]
                    resume_quote_ids=tuple(resume_ids),
                    window_end_at=received_at,
                )
        if received_at >= deadline:
            chosen = eligible[0]
            return BarrierLabel(
                anchor_at=anchor_at,
                entry_at=entry.received_at,
                exit_at=received_at,
                entry_quote_id=entry.quote_id,
                exit_quote_id=chosen.quote_id,
                entry_price=entry.ask,
                exit_price=chosen.bid,
                upper_price=upper,
                lower_price=lower,
                outcome="HORIZON",
                resume_quote_ids=tuple(resume_ids),
                window_end_at=received_at,
            )
    raise LabelInputError("no executable bid exit at or after horizon")


def _assert_x05_authorized(program_root: str | Path) -> None:
    registry = load_experiment_registry(program_root)
    card = registry["X-05"]
    scope = card["authorization_scopes"]["label_generation"]
    if not scope["authorized"]:
        raise LabelAuthorizationError("X-05 label_generation is not authorized")
    locks = {lock["id"]: lock for lock in card["registration_locks"]}
    unresolved = [
        lock_id
        for lock_id in scope["required_lock_ids"]
        if locks[lock_id]["status"] != "resolved"
    ]
    if unresolved:
        raise LabelAuthorizationError(
            "X-05 label_generation has unresolved locks: " + ", ".join(unresolved)
        )
    if "label_generation" not in card["preregistered_inputs"]:
        raise LabelAuthorizationError(
            "X-05 label_generation code/data hashes are not preregistered"
        )
    if any(registry[dependency]["status"] != "done" for dependency in card["dependencies"]):
        raise LabelAuthorizationError("X-05 dependency is not complete")


def generate_x05_long_labels(
    program_root: str | Path,
    *,
    quotes: Sequence[QuoteV0],
    anchors: Sequence[datetime],
    parameters: BarrierLabelParameters,
) -> tuple[BarrierLabel, ...]:
    """Generate X-05 labels only after registry authorization and preregistration."""

    _assert_x05_authorized(program_root)
    ordered_anchors = sorted(anchors)
    for anchor in ordered_anchors:
        _require_utc(anchor, "anchor")
    labels: list[BarrierLabel] = []
    excluded_until: datetime | None = None
    for anchor in ordered_anchors:
        if (
            parameters.overlap_rule == "DROP_LATER"
            and excluded_until is not None
            and anchor <= excluded_until
        ):
            continue
        label = label_long(quotes, anchor_at=anchor, parameters=parameters)
        labels.append(label)
        if parameters.overlap_rule == "DROP_LATER":
            excluded_until = label.window_end_at + parameters.purge + parameters.embargo
    return tuple(labels)


__all__ = [
    "BarrierLabel",
    "BarrierLabelParameters",
    "LabelAuthorizationError",
    "LabelInputError",
    "QuoteV0",
    "generate_x05_long_labels",
    "label_long",
]
