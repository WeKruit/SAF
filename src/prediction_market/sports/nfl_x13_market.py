"""Pure normalization and Layer G/M/A association core for NFL X-13.

The module deliberately performs no network or filesystem writes.  It keeps
observed trades, interval-ending BBO candles, and derived complements distinct,
and it fails closed whenever proposition identity or temporal evidence is not
strong enough for directional analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Literal


HORIZONS_SECONDS = (1, 2, 5, 10, 30, 60)

_VENUES = frozenset({"polymarket", "kalshi"})
_KINDS = frozenset(
    {"primitive", "same_game_composite", "cross_game_inventory"}
)
_PRIMITIVE_FAMILIES = frozenset(
    {
        "moneyline",
        "spread",
        "total",
        "period",
        "team_total",
        "player_passing",
        "player_rushing",
        "player_receiving",
        "player_receptions",
        "player_td",
        "first_td",
        "any_td",
    }
)
_FAMILIES = _PRIMITIVE_FAMILIES | frozenset(
    {"same_game_composite", "cross_game_mve_inventory"}
)
_SUBJECT_KINDS = frozenset({"game", "team", "player", "inventory"})
_COMPARATORS = frozenset({"eq", "gt", "gte", "lt", "lte"})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ZERO = Decimal("0")
_ONE = Decimal("1")


class X13MarketError(ValueError):
    """Evidence cannot support a normalized X-13 market conclusion."""


def _canonical_decimal(value: Decimal) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise X13MarketError("canonical decimal must be finite Decimal")
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise X13MarketError("value is not canonical JSON") from error


def _canonical_hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_nonempty(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise X13MarketError(f"{field} must be a nonempty string")
    return value


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise X13MarketError(f"{field} must be a lowercase SHA-256")
    return value


def _require_decimal(
    value: object,
    field: str,
    *,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    strictly_positive: bool = False,
) -> Decimal:
    if type(value) is Decimal:
        result = value
    elif type(value) in (str, int):
        try:
            result = Decimal(value)
        except (InvalidOperation, ValueError) as error:
            raise X13MarketError(f"{field} must be an exact decimal") from error
    else:
        raise X13MarketError(f"{field} must be an exact decimal")
    if not result.is_finite():
        raise X13MarketError(f"{field} must be finite")
    if strictly_positive and result <= 0:
        raise X13MarketError(f"{field} must be positive")
    if minimum is not None and result < minimum:
        raise X13MarketError(f"{field} is below its minimum")
    if maximum is not None and result > maximum:
        raise X13MarketError(f"{field} is above its maximum")
    return result


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise X13MarketError(f"{field} must be a UTC datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise X13MarketError(f"{field} must be a UTC datetime")
    return value.astimezone(timezone.utc)


def _parse_utc(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise X13MarketError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise X13MarketError(f"{field} must be an RFC3339 UTC timestamp") from error
    return _utc(parsed, field)


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise X13MarketError(f"{field} must be bool")
    return value


def _require_binary_side(
    value: object, field: str
) -> Literal["yes", "no"]:
    if value not in ("yes", "no") or type(value) is not str:
        raise X13MarketError(f"{field} must be yes or no")
    return value


def _epoch_second(value: object, field: str) -> datetime:
    if type(value) is str and value.isdigit():
        seconds = int(value)
    elif type(value) is int:
        seconds = value
    else:
        raise X13MarketError(f"{field} must be an integer epoch second")
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise X13MarketError(f"{field} is outside the datetime range") from error


@dataclass(frozen=True, slots=True)
class SubjectRef:
    kind: str
    canonical_id: str | None
    name: str
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PropositionLeg:
    family: str
    period: str
    subject: SubjectRef
    measure: str
    comparator: str
    line: Decimal | None
    outcomes: tuple[str, ...]
    dependency_game_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NormalizedContract:
    venue: str
    venue_market_id: str
    family: str
    period: str
    subject: SubjectRef
    measure: str
    comparator: str
    line: Decimal | None
    outcomes: tuple[str, ...]
    rule_hash: str
    dependency_game_ids: tuple[str, ...]
    kind: str
    legs: tuple[PropositionLeg, ...]
    analysis_eligible: bool
    ineligible_reasons: tuple[str, ...]
    canonical_hash: str


def _validate_subject(subject: object) -> SubjectRef:
    if not isinstance(subject, SubjectRef):
        raise X13MarketError("subject must be SubjectRef")
    if subject.kind not in _SUBJECT_KINDS:
        raise X13MarketError("unknown subject kind")
    _require_nonempty(subject.name, "subject.name")
    if subject.canonical_id is not None:
        _require_nonempty(subject.canonical_id, "subject.canonical_id")
    if (
        type(subject.candidate_ids) is not tuple
        or any(type(item) is not str or not item for item in subject.candidate_ids)
        or len(set(subject.candidate_ids)) != len(subject.candidate_ids)
    ):
        raise X13MarketError("subject.candidate_ids must be unique strings")
    return subject


def _validate_outcomes(outcomes: object) -> tuple[str, ...]:
    if (
        type(outcomes) is not tuple
        or len(outcomes) < 2
        or any(type(item) is not str or not item for item in outcomes)
        or len(set(outcomes)) != len(outcomes)
    ):
        raise X13MarketError("outcomes must contain unique nonempty strings")
    return outcomes


def _validate_dependencies(value: object) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise X13MarketError(
            "dependency_game_ids must contain unique nonempty strings"
        )
    return value


def _line(value: object) -> Decimal | None:
    if value is None:
        return None
    if type(value) is not Decimal:
        raise X13MarketError("line must be Decimal or None")
    if not value.is_finite():
        raise X13MarketError("line must be finite")
    return value


def _leg_material(leg: PropositionLeg) -> dict[str, object]:
    return {
        "family": leg.family,
        "period": leg.period,
        "subject": {
            "kind": leg.subject.kind,
            "canonical_id": leg.subject.canonical_id,
            "name": leg.subject.name,
            "candidate_ids": list(leg.subject.candidate_ids),
        },
        "measure": leg.measure,
        "comparator": leg.comparator,
        "line": None if leg.line is None else _canonical_decimal(leg.line),
        "outcomes": list(leg.outcomes),
        "dependency_game_ids": list(leg.dependency_game_ids),
    }


def normalize_contract(
    *,
    venue: str,
    venue_market_id: str,
    family: str,
    period: str,
    subject: SubjectRef,
    measure: str,
    comparator: str,
    line: Decimal | None,
    outcomes: tuple[str, ...],
    rule_hash: str,
    dependency_game_ids: tuple[str, ...],
    kind: str,
    legs: tuple[PropositionLeg, ...] = (),
) -> NormalizedContract:
    """Normalize a typed proposition and derive eligibility fail-closed."""

    if venue not in _VENUES:
        raise X13MarketError("unknown venue")
    if family not in _FAMILIES:
        raise X13MarketError("unknown proposition family")
    if kind not in _KINDS:
        raise X13MarketError("unknown proposition kind")
    _require_nonempty(venue_market_id, "venue_market_id")
    _require_nonempty(period, "period")
    subject = _validate_subject(subject)
    _require_nonempty(measure, "measure")
    if comparator not in _COMPARATORS:
        raise X13MarketError("unknown comparator")
    normalized_line = _line(line)
    outcomes = _validate_outcomes(outcomes)
    rule_hash = _require_sha256(rule_hash, "rule_hash")
    dependencies = _validate_dependencies(dependency_game_ids)
    if type(legs) is not tuple:
        raise X13MarketError("legs must be a tuple")

    if kind == "primitive" and family not in _PRIMITIVE_FAMILIES:
        raise X13MarketError("primitive kind requires a primitive family")
    if kind == "same_game_composite" and family != "same_game_composite":
        raise X13MarketError(
            "same_game_composite kind requires same_game_composite family"
        )
    if kind == "cross_game_inventory" and family != "cross_game_mve_inventory":
        raise X13MarketError(
            "cross_game_inventory kind requires "
            "cross_game_mve_inventory family"
        )
    if kind != "same_game_composite" and legs:
        raise X13MarketError("only same_game_composite may contain legs")

    for leg in legs:
        if not isinstance(leg, PropositionLeg) or leg.family not in _PRIMITIVE_FAMILIES:
            raise X13MarketError("unknown composite leg")
        _require_nonempty(leg.period, "leg.period")
        _validate_subject(leg.subject)
        _require_nonempty(leg.measure, "leg.measure")
        if leg.comparator not in _COMPARATORS:
            raise X13MarketError("unknown composite leg comparator")
        _line(leg.line)
        _validate_outcomes(leg.outcomes)
        _validate_dependencies(leg.dependency_game_ids)

    reasons: list[str] = []
    if subject.kind == "player" and subject.canonical_id is None:
        reasons.append("ambiguous_player_subject")
    if kind == "cross_game_inventory":
        reasons.append("cross_game_mve_inventory_excluded")
    elif kind == "primitive" and len(dependencies) != 1:
        reasons.append("primitive_dependency_cardinality")
    elif kind == "same_game_composite":
        leg_dependencies = {
            game_id for leg in legs for game_id in leg.dependency_game_ids
        }
        if not legs or len(dependencies) != 1 or leg_dependencies != set(dependencies):
            reasons.append("same_game_dependency_mismatch")
        if any(
            leg.subject.kind == "player" and leg.subject.canonical_id is None
            for leg in legs
        ):
            reasons.append("ambiguous_player_leg")

    material = {
        "venue": venue,
        "venue_market_id": venue_market_id,
        "family": family,
        "period": period,
        "subject": {
            "kind": subject.kind,
            "canonical_id": subject.canonical_id,
            "name": subject.name,
            "candidate_ids": list(subject.candidate_ids),
        },
        "measure": measure,
        "comparator": comparator,
        "line": (
            None
            if normalized_line is None
            else _canonical_decimal(normalized_line)
        ),
        "outcomes": list(outcomes),
        "rule_hash": rule_hash,
        "dependency_game_ids": list(dependencies),
        "kind": kind,
        "legs": [_leg_material(leg) for leg in legs],
        "analysis_eligible": not reasons,
        "ineligible_reasons": reasons,
    }
    return NormalizedContract(
        venue=venue,
        venue_market_id=venue_market_id,
        family=family,
        period=period,
        subject=subject,
        measure=measure,
        comparator=comparator,
        line=normalized_line,
        outcomes=outcomes,
        rule_hash=rule_hash,
        dependency_game_ids=dependencies,
        kind=kind,
        legs=legs,
        analysis_eligible=not reasons,
        ineligible_reasons=tuple(reasons),
        canonical_hash=_canonical_hash(material),
    )


@dataclass(frozen=True, slots=True)
class SourceInterval:
    start: datetime
    end: datetime
    end_inclusive: bool

    def __post_init__(self) -> None:
        start = _utc(self.start, "interval.start")
        end = _utc(self.end, "interval.end")
        if end <= start:
            raise X13MarketError("source interval must have positive width")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if type(self.end_inclusive) is not bool:
            raise X13MarketError("end_inclusive must be bool")


@dataclass(frozen=True, slots=True)
class CandleOHLC:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        values = {
            name: _require_decimal(
                value,
                name,
                minimum=_ZERO,
                maximum=_ONE,
            )
            for name, value in (
                ("open", self.open),
                ("high", self.high),
                ("low", self.low),
                ("close", self.close),
            )
        }
        if values["low"] > values["high"]:
            raise X13MarketError("candle low exceeds high")
        if not (
            values["low"] <= values["open"] <= values["high"]
            and values["low"] <= values["close"] <= values["high"]
        ):
            raise X13MarketError("candle open/close is outside low/high")
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class MarketObservation:
    observation_id: str
    venue: str
    raw_market_id: str
    logical_market_id: str | None
    outcome: str
    kind: str
    source_interval: SourceInterval
    price: Decimal | None
    size: Decimal
    bid: Decimal | None
    ask: Decimal | None
    provenance: Literal["observed", "derived_complement"]
    render_semantics: Literal["solid", "dashed"]
    native_id: str
    derived_from_observation_id: str | None = None
    native_source_time: datetime | None = None
    is_block_trade: bool | None = None
    taker_side: Literal["yes", "no"] | None = None
    taker_outcome_side: Literal["yes", "no"] | None = None
    yes_price_dollars: Decimal | None = None
    no_price_dollars: Decimal | None = None
    primary_path_eligible: bool = True
    yes_bid_ohlc: CandleOHLC | None = None
    yes_ask_ohlc: CandleOHLC | None = None
    volume: Decimal | None = None
    open_interest: Decimal | None = None

    @property
    def has_executable_bbo(self) -> bool:
        return self.bid is not None and self.ask is not None


def _normalize_candle_ohlc(value: object, field: str) -> CandleOHLC:
    if not isinstance(value, Mapping):
        raise X13MarketError(f"{field} must be an OHLC mapping")
    return CandleOHLC(
        open=_require_decimal(
            value.get("open"),
            f"{field}.open",
            minimum=_ZERO,
            maximum=_ONE,
        ),
        high=_require_decimal(
            value.get("high"),
            f"{field}.high",
            minimum=_ZERO,
            maximum=_ONE,
        ),
        low=_require_decimal(
            value.get("low"),
            f"{field}.low",
            minimum=_ZERO,
            maximum=_ONE,
        ),
        close=_require_decimal(
            value.get("close"),
            f"{field}.close",
            minimum=_ZERO,
            maximum=_ONE,
        ),
    )


def _timestamp_text(value: datetime) -> str:
    value = _utc(value, "timestamp")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _observation_material(
    *,
    venue: str,
    raw_market_id: str,
    logical_market_id: str | None,
    outcome: str,
    kind: str,
    source_interval: SourceInterval,
    price: Decimal | None,
    size: Decimal,
    bid: Decimal | None,
    ask: Decimal | None,
    provenance: str,
    native_id: str,
    derived_from_observation_id: str | None,
    native_source_time: datetime | None,
    is_block_trade: bool | None,
    taker_side: str | None,
    taker_outcome_side: str | None,
    yes_price_dollars: Decimal | None,
    no_price_dollars: Decimal | None,
    primary_path_eligible: bool,
    yes_bid_ohlc: CandleOHLC | None,
    yes_ask_ohlc: CandleOHLC | None,
    volume: Decimal | None,
    open_interest: Decimal | None,
) -> dict[str, object]:
    def ohlc_material(value: CandleOHLC | None) -> object:
        if value is None:
            return None
        return {
            "open": _canonical_decimal(value.open),
            "high": _canonical_decimal(value.high),
            "low": _canonical_decimal(value.low),
            "close": _canonical_decimal(value.close),
        }

    return {
        "venue": venue,
        "raw_market_id": raw_market_id,
        "logical_market_id": logical_market_id,
        "outcome": outcome,
        "kind": kind,
        "source_start": _timestamp_text(source_interval.start),
        "source_end": _timestamp_text(source_interval.end),
        "source_end_inclusive": source_interval.end_inclusive,
        "price": None if price is None else _canonical_decimal(price),
        "size": _canonical_decimal(size),
        "bid": None if bid is None else _canonical_decimal(bid),
        "ask": None if ask is None else _canonical_decimal(ask),
        "provenance": provenance,
        "native_id": native_id,
        "derived_from_observation_id": derived_from_observation_id,
        "native_source_time": (
            None
            if native_source_time is None
            else _timestamp_text(native_source_time)
        ),
        "is_block_trade": is_block_trade,
        "taker_side": taker_side,
        "taker_outcome_side": taker_outcome_side,
        "yes_price_dollars": (
            None
            if yes_price_dollars is None
            else _canonical_decimal(yes_price_dollars)
        ),
        "no_price_dollars": (
            None
            if no_price_dollars is None
            else _canonical_decimal(no_price_dollars)
        ),
        "primary_path_eligible": primary_path_eligible,
        "yes_bid_ohlc": ohlc_material(yes_bid_ohlc),
        "yes_ask_ohlc": ohlc_material(yes_ask_ohlc),
        "volume": None if volume is None else _canonical_decimal(volume),
        "open_interest": (
            None
            if open_interest is None
            else _canonical_decimal(open_interest)
        ),
    }


def _make_observation(
    *,
    venue: str,
    raw_market_id: str,
    logical_market_id: str | None,
    outcome: str,
    kind: str,
    source_interval: SourceInterval,
    price: Decimal | None,
    size: Decimal,
    bid: Decimal | None,
    ask: Decimal | None,
    provenance: Literal["observed", "derived_complement"],
    render_semantics: Literal["solid", "dashed"],
    native_id: str,
    derived_from_observation_id: str | None = None,
    native_source_time: datetime | None = None,
    is_block_trade: bool | None = None,
    taker_side: Literal["yes", "no"] | None = None,
    taker_outcome_side: Literal["yes", "no"] | None = None,
    yes_price_dollars: Decimal | None = None,
    no_price_dollars: Decimal | None = None,
    primary_path_eligible: bool = True,
    yes_bid_ohlc: CandleOHLC | None = None,
    yes_ask_ohlc: CandleOHLC | None = None,
    volume: Decimal | None = None,
    open_interest: Decimal | None = None,
) -> MarketObservation:
    material = _observation_material(
        venue=venue,
        raw_market_id=raw_market_id,
        logical_market_id=logical_market_id,
        outcome=outcome,
        kind=kind,
        source_interval=source_interval,
        price=price,
        size=size,
        bid=bid,
        ask=ask,
        provenance=provenance,
        native_id=native_id,
        derived_from_observation_id=derived_from_observation_id,
        native_source_time=native_source_time,
        is_block_trade=is_block_trade,
        taker_side=taker_side,
        taker_outcome_side=taker_outcome_side,
        yes_price_dollars=yes_price_dollars,
        no_price_dollars=no_price_dollars,
        primary_path_eligible=primary_path_eligible,
        yes_bid_ohlc=yes_bid_ohlc,
        yes_ask_ohlc=yes_ask_ohlc,
        volume=volume,
        open_interest=open_interest,
    )
    return MarketObservation(
        observation_id=_canonical_hash(material),
        venue=venue,
        raw_market_id=raw_market_id,
        logical_market_id=logical_market_id,
        outcome=outcome,
        kind=kind,
        source_interval=source_interval,
        price=price,
        size=size,
        bid=bid,
        ask=ask,
        provenance=provenance,
        render_semantics=render_semantics,
        native_id=native_id,
        derived_from_observation_id=derived_from_observation_id,
        native_source_time=native_source_time,
        is_block_trade=is_block_trade,
        taker_side=taker_side,
        taker_outcome_side=taker_outcome_side,
        yes_price_dollars=yes_price_dollars,
        no_price_dollars=no_price_dollars,
        primary_path_eligible=primary_path_eligible,
        yes_bid_ohlc=yes_bid_ohlc,
        yes_ask_ohlc=yes_ask_ohlc,
        volume=volume,
        open_interest=open_interest,
    )


def normalize_polymarket_trade(
    raw: Mapping[str, object],
    *,
    raw_market_id: str,
    logical_market_id: str,
    outcome: str,
) -> MarketObservation:
    """Normalize one observed Polymarket trade; no BBO is inferred."""

    if not isinstance(raw, Mapping):
        raise X13MarketError("Polymarket trade must be a mapping")
    raw_market_id = _require_nonempty(raw_market_id, "raw_market_id")
    logical_market_id = _require_nonempty(
        logical_market_id, "logical_market_id"
    )
    outcome = _require_nonempty(outcome, "outcome")
    native_id = _require_nonempty(raw.get("transactionHash"), "transactionHash")
    start = _epoch_second(raw.get("timestamp"), "timestamp")
    price = _require_decimal(
        raw.get("price"), "price", minimum=_ZERO, maximum=_ONE
    )
    size = _require_decimal(raw.get("size"), "size", strictly_positive=True)
    interval = SourceInterval(start, start + timedelta(seconds=1), False)
    return _make_observation(
        venue="polymarket",
        raw_market_id=raw_market_id,
        logical_market_id=logical_market_id,
        outcome=outcome,
        kind="trade",
        source_interval=interval,
        price=price,
        size=size,
        bid=None,
        ask=None,
        provenance="observed",
        render_semantics="solid",
        native_id=native_id,
        native_source_time=start,
    )


def normalize_kalshi_trade(
    raw: Mapping[str, object],
    *,
    ticker: str,
    outcome: str,
    logical_market_id: str,
) -> MarketObservation:
    """Normalize one observed Kalshi team-YES trade."""

    if not isinstance(raw, Mapping):
        raise X13MarketError("Kalshi trade must be a mapping")
    ticker = _require_nonempty(ticker, "ticker")
    outcome = _require_nonempty(outcome, "outcome")
    logical_market_id = _require_nonempty(
        logical_market_id, "logical_market_id"
    )
    native_id = _require_nonempty(raw.get("trade_id"), "trade_id")
    native_source_time = _parse_utc(
        raw.get("created_time"), "created_time"
    )
    yes_price = _require_decimal(
        raw.get("yes_price_dollars"),
        "yes_price_dollars",
        minimum=_ZERO,
        maximum=_ONE,
    )
    no_price = _require_decimal(
        raw.get("no_price_dollars"),
        "no_price_dollars",
        minimum=_ZERO,
        maximum=_ONE,
    )
    if yes_price + no_price != _ONE:
        raise X13MarketError(
            "yes_price_dollars and no_price_dollars must sum to one"
        )
    is_block_trade = _require_bool(
        raw.get("is_block_trade"), "is_block_trade"
    )
    taker_side = _require_binary_side(raw.get("taker_side"), "taker_side")
    taker_outcome_side = _require_binary_side(
        raw.get("taker_outcome_side"), "taker_outcome_side"
    )
    size = _require_decimal(
        raw.get("count_fp"), "count_fp", strictly_positive=True
    )
    return _make_observation(
        venue="kalshi",
        raw_market_id=ticker,
        logical_market_id=logical_market_id,
        outcome=outcome,
        kind="trade",
        # Kalshi supplies a per-trade RFC3339 timestamp with microsecond
        # precision.  Preserve it for historical ordering instead of
        # collapsing every trade in the same second into one fabricated
        # one-second interval.  The one-microsecond half-open interval is
        # merely the smallest representable positive SourceInterval; it does
        # not claim that a trade lasts a microsecond.
        source_interval=SourceInterval(
            native_source_time,
            native_source_time + timedelta(microseconds=1),
            False,
        ),
        price=yes_price,
        size=size,
        bid=None,
        ask=None,
        provenance="observed",
        render_semantics="solid",
        native_id=native_id,
        native_source_time=native_source_time,
        is_block_trade=is_block_trade,
        taker_side=taker_side,
        taker_outcome_side=taker_outcome_side,
        yes_price_dollars=yes_price,
        no_price_dollars=no_price,
        primary_path_eligible=not is_block_trade,
    )


def normalize_kalshi_candle_bbo(
    raw: Mapping[str, object],
    *,
    ticker: str,
    outcome: str,
    logical_market_id: str,
) -> MarketObservation:
    """Normalize one interval-ending Kalshi one-minute BBO candle."""

    if not isinstance(raw, Mapping):
        raise X13MarketError("Kalshi candle must be a mapping")
    ticker = _require_nonempty(ticker, "ticker")
    outcome = _require_nonempty(outcome, "outcome")
    logical_market_id = _require_nonempty(
        logical_market_id, "logical_market_id"
    )
    end = _epoch_second(raw.get("end_period_ts"), "end_period_ts")
    yes_bid_ohlc = _normalize_candle_ohlc(raw.get("yes_bid"), "yes_bid")
    yes_ask_ohlc = _normalize_candle_ohlc(raw.get("yes_ask"), "yes_ask")
    for basis in ("open", "high", "low", "close"):
        if getattr(yes_bid_ohlc, basis) > getattr(yes_ask_ohlc, basis):
            raise X13MarketError(
                f"Kalshi candle contains crossed BBO at {basis}"
            )
    volume = _require_decimal(raw.get("volume"), "volume", minimum=_ZERO)
    open_interest = _require_decimal(
        raw.get("open_interest"), "open_interest", minimum=_ZERO
    )
    interval = SourceInterval(end - timedelta(minutes=1), end, False)
    native_id = f"{ticker}:{int(end.timestamp())}:1m"
    return _make_observation(
        venue="kalshi",
        raw_market_id=ticker,
        logical_market_id=logical_market_id,
        outcome=outcome,
        kind="kalshi_1m_candle_bbo",
        source_interval=interval,
        price=None,
        size=volume,
        bid=yes_bid_ohlc.close,
        ask=yes_ask_ohlc.close,
        provenance="observed",
        render_semantics="solid",
        native_id=native_id,
        native_source_time=end,
        yes_bid_ohlc=yes_bid_ohlc,
        yes_ask_ohlc=yes_ask_ohlc,
        volume=volume,
        open_interest=open_interest,
    )


def derive_complement(
    observation: MarketObservation, *, outcome: str
) -> MarketObservation:
    """Create a visibly derived complement without inventing a book."""

    if not isinstance(observation, MarketObservation):
        raise X13MarketError("observation must be MarketObservation")
    outcome = _require_nonempty(outcome, "outcome")
    if observation.price is None:
        raise X13MarketError("complement requires an observed price")
    if outcome == observation.outcome:
        raise X13MarketError("complement outcome must differ from observed outcome")
    return _make_observation(
        venue=observation.venue,
        raw_market_id=observation.raw_market_id,
        logical_market_id=observation.logical_market_id,
        outcome=outcome,
        kind=observation.kind,
        source_interval=observation.source_interval,
        price=_ONE - observation.price,
        size=observation.size,
        bid=None,
        ask=None,
        provenance="derived_complement",
        render_semantics="dashed",
        native_id=observation.native_id,
        derived_from_observation_id=observation.observation_id,
        native_source_time=observation.native_source_time,
        is_block_trade=observation.is_block_trade,
        taker_side=observation.taker_side,
        taker_outcome_side=observation.taker_outcome_side,
        yes_price_dollars=observation.yes_price_dollars,
        no_price_dollars=observation.no_price_dollars,
        primary_path_eligible=False,
        volume=observation.volume,
        open_interest=observation.open_interest,
    )


@dataclass(frozen=True, slots=True)
class KalshiWinnerMarket:
    logical_market_id: str
    game_id: str
    event_ticker: str
    outcomes: tuple[str, str]
    raw_contracts: tuple[tuple[str, str], tuple[str, str]]


def group_kalshi_winner_market(
    *,
    game_id: str,
    event_ticker: str,
    team_tickers: Mapping[str, str],
) -> KalshiWinnerMarket:
    """Group two team-YES contracts while preserving both raw tickers."""

    game_id = _require_nonempty(game_id, "game_id")
    event_ticker = _require_nonempty(event_ticker, "event_ticker")
    if (
        not isinstance(team_tickers, Mapping)
        or len(team_tickers) != 2
        or any(
            type(team) is not str
            or not team
            or type(ticker) is not str
            or not ticker
            for team, ticker in team_tickers.items()
        )
        or len(set(team_tickers.values())) != 2
    ):
        raise X13MarketError(
            "Kalshi winner grouping requires two unique team tickers"
        )
    raw = tuple(sorted(team_tickers.items()))
    return KalshiWinnerMarket(
        logical_market_id=f"kalshi:{game_id}:winner",
        game_id=game_id,
        event_ticker=event_ticker,
        outcomes=(raw[0][0], raw[1][0]),
        raw_contracts=(raw[0], raw[1]),
    )


def deduplicate_observations(
    observations: Sequence[MarketObservation],
) -> tuple[MarketObservation, ...]:
    """Deduplicate stable observation IDs without silently accepting collision."""

    if not isinstance(observations, Sequence) or isinstance(
        observations, (str, bytes)
    ):
        raise X13MarketError("observations must be a sequence")
    by_id: dict[str, MarketObservation] = {}
    ordered: list[MarketObservation] = []
    for observation in observations:
        if not isinstance(observation, MarketObservation):
            raise X13MarketError("observations must contain MarketObservation")
        previous = by_id.get(observation.observation_id)
        if previous is None:
            by_id[observation.observation_id] = observation
            ordered.append(observation)
        elif previous != observation:
            raise X13MarketError("stable observation ID collision")
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class PaginationPage:
    cursor_in: str | None
    cursor_out: str | None
    item_count: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class CaptureCompleteness:
    complete: Literal[True]
    page_count: int
    item_count: int
    terminal_proof: Literal["explicit_unsaturated_terminal_page"]


def prove_capture_complete(
    pages: Sequence[PaginationPage], *, page_limit: int
) -> CaptureCompleteness:
    """Require an acyclic cursor chain and explicit unsaturated terminal leaf."""

    if type(page_limit) is not int or page_limit <= 0:
        raise X13MarketError("page_limit must be a positive integer")
    if (
        not isinstance(pages, Sequence)
        or isinstance(pages, (str, bytes))
        or not pages
    ):
        raise X13MarketError("pagination requires at least one page")
    seen_out: set[str] = set()
    previous_out: str | None = None
    total = 0
    for index, page in enumerate(pages):
        if not isinstance(page, PaginationPage):
            raise X13MarketError("pages must contain PaginationPage")
        if (
            type(page.item_count) is not int
            or page.item_count < 0
            or page.item_count > page_limit
        ):
            raise X13MarketError("page item_count is outside page limit")
        if type(page.terminal) is not bool:
            raise X13MarketError("page terminal flag must be bool")
        if index == 0:
            if page.cursor_in is not None:
                raise X13MarketError("first page cursor_in must be absent")
        elif page.cursor_in != previous_out:
            raise X13MarketError("cursor chain is discontinuous")
        if page.cursor_out is not None:
            _require_nonempty(page.cursor_out, "cursor_out")
            if page.cursor_out in seen_out:
                raise X13MarketError("repeated cursor in pagination chain")
            seen_out.add(page.cursor_out)
        if page.terminal:
            if index != len(pages) - 1 or page.cursor_out is not None:
                raise X13MarketError("terminal page must be the final cursor leaf")
        elif page.cursor_out is None:
            raise X13MarketError("nonterminal page lacks continuation cursor")
        previous_out = page.cursor_out
        total += page.item_count
    leaf = pages[-1]
    if not leaf.terminal:
        raise X13MarketError("pagination lacks terminal proof")
    if leaf.item_count == page_limit:
        raise X13MarketError("saturated terminal leaf cannot prove completeness")
    return CaptureCompleteness(
        complete=True,
        page_count=len(pages),
        item_count=total,
        terminal_proof="explicit_unsaturated_terminal_page",
    )


@dataclass(frozen=True, slots=True)
class RawManifestReceipt:
    object_sha256: str
    manifest_sha256: str
    receipt_sha256: str


def _object_hash(raw: bytes) -> str:
    if type(raw) is not bytes:
        raise X13MarketError("raw object must be bytes")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_raw_manifest_receipt(
    raw: bytes, manifest: Mapping[str, object]
) -> RawManifestReceipt:
    """Build an in-memory content receipt; no raw data is written."""

    if not isinstance(manifest, Mapping):
        raise X13MarketError("manifest must be a mapping")
    object_sha256 = _object_hash(raw)
    if manifest.get("object_sha256") != object_sha256:
        raise X13MarketError("manifest object hash does not match raw object")
    manifest_sha256 = _canonical_hash(dict(manifest))
    receipt_sha256 = _canonical_hash(
        {
            "object_sha256": object_sha256,
            "manifest_sha256": manifest_sha256,
        }
    )
    return RawManifestReceipt(
        object_sha256=object_sha256,
        manifest_sha256=manifest_sha256,
        receipt_sha256=receipt_sha256,
    )


def verify_raw_manifest_receipt(
    raw: bytes,
    manifest: Mapping[str, object],
    receipt: RawManifestReceipt,
) -> bool:
    """Fail closed on raw-object, manifest, or receipt tampering."""

    if not isinstance(receipt, RawManifestReceipt):
        raise X13MarketError("receipt must be RawManifestReceipt")
    if _object_hash(raw) != receipt.object_sha256:
        raise X13MarketError("raw object tamper detected")
    if not isinstance(manifest, Mapping):
        raise X13MarketError("manifest must be a mapping")
    manifest_sha256 = _canonical_hash(dict(manifest))
    if manifest_sha256 != receipt.manifest_sha256:
        raise X13MarketError("manifest tamper detected")
    if manifest.get("object_sha256") != receipt.object_sha256:
        raise X13MarketError("manifest object binding is invalid")
    expected_receipt = _canonical_hash(
        {
            "object_sha256": receipt.object_sha256,
            "manifest_sha256": receipt.manifest_sha256,
        }
    )
    if expected_receipt != receipt.receipt_sha256:
        raise X13MarketError("receipt tamper detected")
    return True


def build_layer_g_interval(
    source_time: datetime, *, delay_seconds: Decimal
) -> SourceInterval:
    """Build the half-open Layer G uncertainty interval [source, end)."""

    source_time = _utc(source_time, "source_time")
    delay = _require_decimal(delay_seconds, "delay_seconds", minimum=_ZERO)
    microseconds = delay * Decimal("1000000")
    if microseconds != microseconds.to_integral_value():
        raise X13MarketError("delay_seconds exceeds microsecond precision")
    return SourceInterval(
        source_time,
        source_time
        + timedelta(seconds=1, microseconds=int(microseconds)),
        False,
    )


@dataclass(frozen=True, slots=True)
class IntervalAssociation:
    overlap: bool
    order_ambiguous: bool
    order_relation: Literal["market_before_event", "market_after_event"] | None


def _ends_before(first: SourceInterval, second: SourceInterval) -> bool:
    return first.end < second.start or (
        first.end == second.start and not first.end_inclusive
    )


def associate_intervals(
    event_interval: SourceInterval,
    market_interval: SourceInterval,
) -> IntervalAssociation:
    """Associate Layer G and M without imposing an order on overlap."""

    if not isinstance(event_interval, SourceInterval) or not isinstance(
        market_interval, SourceInterval
    ):
        raise X13MarketError("association requires SourceInterval values")
    if _ends_before(event_interval, market_interval):
        return IntervalAssociation(False, False, "market_after_event")
    if _ends_before(market_interval, event_interval):
        return IntervalAssociation(False, False, "market_before_event")
    return IntervalAssociation(True, True, None)


@dataclass(frozen=True, slots=True)
class HorizonMeasurement:
    horizon_seconds: int
    status: Literal["observed", "missing", "contaminated", "order_ambiguous"]
    first_actual_price: Decimal | None
    last_actual_price: Decimal | None
    vwap: Decimal | None
    signed_delta: Decimal | None
    max_excursion: Decimal | None
    observation_count: int
    volume: Decimal
    staleness_seconds: Decimal | None
    venue_direction: Literal["up", "down", "flat", "missing"]


@dataclass(frozen=True, slots=True)
class ReactionSummary:
    venue: str
    market_id: str
    outcome: str
    pre_actual_price: Decimal | None
    first_post_price: Decimal | None
    horizons: Mapping[int, HorizonMeasurement]
    net_60s: Decimal | None
    max_excursion: Decimal | None
    overshoot_candidate: bool
    reversal_candidate: bool
    contaminated: bool
    venue_direction: Literal["up", "down", "flat", "missing"]


def _seconds(value: timedelta) -> Decimal:
    return (
        Decimal(value.days) * Decimal("86400")
        + Decimal(value.seconds)
        + Decimal(value.microseconds) / Decimal("1000000")
    )


def _direction(delta: Decimal | None) -> Literal["up", "down", "flat", "missing"]:
    if delta is None:
        return "missing"
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def _empty_horizon(
    horizon: int, status: Literal["missing", "contaminated", "order_ambiguous"]
) -> HorizonMeasurement:
    return HorizonMeasurement(
        horizon_seconds=horizon,
        status=status,
        first_actual_price=None,
        last_actual_price=None,
        vwap=None,
        signed_delta=None,
        max_excursion=None,
        observation_count=0,
        volume=_ZERO,
        staleness_seconds=None,
        venue_direction="missing",
    )


def measure_reaction(
    *,
    event_interval: SourceInterval,
    observations: Sequence[MarketObservation],
    venue: str,
    market_id: str,
    outcome: str,
    next_salient_at: datetime | None = None,
) -> ReactionSummary:
    """Measure actual-only horizon buckets with no carry-forward."""

    if not isinstance(event_interval, SourceInterval):
        raise X13MarketError("event_interval must be SourceInterval")
    if venue not in _VENUES:
        raise X13MarketError("unknown venue")
    market_id = _require_nonempty(market_id, "market_id")
    outcome = _require_nonempty(outcome, "outcome")
    if next_salient_at is not None:
        next_salient_at = _utc(next_salient_at, "next_salient_at")
        if next_salient_at <= event_interval.start:
            raise X13MarketError("next salient episode must follow event")

    material = [
        observation
        for observation in deduplicate_observations(observations)
        if observation.venue == venue
        and (
            observation.raw_market_id == market_id
            or observation.logical_market_id == market_id
        )
        and observation.outcome == outcome
        and observation.price is not None
        and observation.primary_path_eligible
    ]
    pre = [
        observation
        for observation in material
        if _ends_before(observation.source_interval, event_interval)
    ]
    pre.sort(key=lambda item: item.source_interval.end)
    pre_observation = pre[-1] if pre else None
    pre_price = None if pre_observation is None else pre_observation.price

    post = [
        observation
        for observation in material
        if associate_intervals(
            event_interval, observation.source_interval
        ).order_relation
        == "market_after_event"
        and observation.source_interval.start
        <= event_interval.start + timedelta(seconds=60)
    ]
    post.sort(key=lambda item: item.source_interval.start)
    first_post_price = None if not post else post[0].price

    measurements: dict[int, HorizonMeasurement] = {}
    previous_deadline = event_interval.start
    for horizon in HORIZONS_SECONDS:
        deadline = event_interval.start + timedelta(seconds=horizon)
        if next_salient_at is not None and next_salient_at <= deadline:
            measurements[horizon] = _empty_horizon(horizon, "contaminated")
            previous_deadline = deadline
            continue
        bucket = [
            observation
            for observation in post
            if previous_deadline < observation.source_interval.start <= deadline
        ]
        if pre_price is None or not bucket:
            measurements[horizon] = _empty_horizon(horizon, "missing")
            previous_deadline = deadline
            continue
        starts = [item.source_interval.start for item in bucket]
        if len(starts) != len(set(starts)):
            measurements[horizon] = _empty_horizon(
                horizon, "order_ambiguous"
            )
            previous_deadline = deadline
            continue
        bucket.sort(key=lambda item: item.source_interval.start)
        first_price = bucket[0].price
        last_price = bucket[-1].price
        assert first_price is not None and last_price is not None
        volume = sum((item.size for item in bucket), _ZERO)
        vwap = sum(
            (item.price * item.size for item in bucket if item.price is not None),
            _ZERO,
        ) / volume
        deltas = [
            item.price - pre_price
            for item in bucket
            if item.price is not None
        ]
        max_excursion = max(deltas, key=abs)
        signed_delta = last_price - pre_price
        measurements[horizon] = HorizonMeasurement(
            horizon_seconds=horizon,
            status="observed",
            first_actual_price=first_price,
            last_actual_price=last_price,
            vwap=vwap,
            signed_delta=signed_delta,
            max_excursion=max_excursion,
            observation_count=len(bucket),
            volume=volume,
            staleness_seconds=_seconds(
                deadline - bucket[-1].source_interval.start
            ),
            venue_direction=_direction(signed_delta),
        )
        previous_deadline = deadline

    minute = measurements[60]
    net_60s = (
        minute.signed_delta if minute.status == "observed" else None
    )
    deltas_60 = (
        []
        if pre_price is None
        else [
            observation.price - pre_price
            for observation in post
            if observation.price is not None
            and (
                next_salient_at is None
                or observation.source_interval.start < next_salient_at
            )
        ]
    )
    max_excursion = max(deltas_60, key=abs) if deltas_60 else None
    first_delta = deltas_60[0] if deltas_60 else None
    reversal = bool(
        first_delta is not None
        and net_60s is not None
        and first_delta * net_60s < 0
    )
    overshoot = bool(
        max_excursion is not None
        and net_60s is not None
        and abs(max_excursion) > abs(net_60s)
    )
    contaminated = any(
        item.status == "contaminated" for item in measurements.values()
    )
    return ReactionSummary(
        venue=venue,
        market_id=market_id,
        outcome=outcome,
        pre_actual_price=pre_price,
        first_post_price=first_post_price,
        horizons=MappingProxyType(measurements),
        net_60s=net_60s,
        max_excursion=max_excursion,
        overshoot_candidate=overshoot,
        reversal_candidate=reversal,
        contaminated=contaminated,
        venue_direction=_direction(net_60s),
    )


_AUDIT_ORDER = (
    "orientation",
    "adjudication",
    "score_possession",
    "same_second",
    "contamination",
    "stale_or_wrong_family",
    "diagnostic_fair_delta",
)


@dataclass(frozen=True, slots=True)
class DirectionalAudit:
    status: Literal["pass", "blocked", "diagnostic"]
    primary_reason: str | None
    reason_codes: tuple[str, ...]
    audit_order: tuple[str, ...] = _AUDIT_ORDER


def audit_directional_anomaly(
    *,
    orientation_valid: bool,
    adjudication_flags: frozenset[str],
    score_possession_valid: bool,
    same_second_ambiguous: bool,
    contaminated: bool,
    stale: bool,
    wrong_family: bool,
    venue_signed_deltas: Mapping[str, Decimal | None],
) -> DirectionalAudit:
    """Run directional anomaly gates in the fixed X-13 audit order."""

    boolean_values = (
        orientation_valid,
        score_possession_valid,
        same_second_ambiguous,
        contaminated,
        stale,
        wrong_family,
    )
    if any(type(value) is not bool for value in boolean_values):
        raise X13MarketError("directional audit gates must be bool")
    if type(adjudication_flags) is not frozenset or any(
        type(value) is not str or not value for value in adjudication_flags
    ):
        raise X13MarketError("adjudication_flags must be frozenset[str]")
    if not isinstance(venue_signed_deltas, Mapping):
        raise X13MarketError("venue_signed_deltas must be a mapping")
    for delta in venue_signed_deltas.values():
        if delta is not None and (
            type(delta) is not Decimal or not delta.is_finite()
        ):
            raise X13MarketError("venue signed deltas must be finite Decimal")

    reasons: list[str] = []
    if not orientation_valid:
        reasons.append("orientation_unresolved")
    if adjudication_flags & frozenset(
        {"nullification", "review", "penalty", "try"}
    ):
        reasons.append("adjudication_unresolved")
    if not score_possession_valid:
        reasons.append("score_or_possession_unresolved")
    if same_second_ambiguous:
        reasons.append("same_second_order_ambiguous")
    if contaminated:
        reasons.append("next_salient_episode_contamination")
    if stale or wrong_family:
        reasons.append("stale_or_wrong_market_family")
    if reasons:
        return DirectionalAudit(
            status="blocked",
            primary_reason=reasons[0],
            reason_codes=tuple(reasons),
        )

    signs = {
        1 if delta > 0 else -1
        for delta in venue_signed_deltas.values()
        if delta is not None and delta != 0
    }
    if signs == {-1, 1}:
        reason = "diagnostic_fair_delta_disagreement"
        return DirectionalAudit(
            status="diagnostic",
            primary_reason=reason,
            reason_codes=(reason,),
        )
    return DirectionalAudit(status="pass", primary_reason=None, reason_codes=())


__all__ = [
    "HORIZONS_SECONDS",
    "CandleOHLC",
    "CaptureCompleteness",
    "DirectionalAudit",
    "HorizonMeasurement",
    "IntervalAssociation",
    "KalshiWinnerMarket",
    "MarketObservation",
    "NormalizedContract",
    "PaginationPage",
    "PropositionLeg",
    "RawManifestReceipt",
    "ReactionSummary",
    "SourceInterval",
    "SubjectRef",
    "X13MarketError",
    "associate_intervals",
    "audit_directional_anomaly",
    "build_layer_g_interval",
    "build_raw_manifest_receipt",
    "deduplicate_observations",
    "derive_complement",
    "group_kalshi_winner_market",
    "measure_reaction",
    "normalize_contract",
    "normalize_kalshi_candle_bbo",
    "normalize_kalshi_trade",
    "normalize_polymarket_trade",
    "prove_capture_complete",
    "verify_raw_manifest_receipt",
]
