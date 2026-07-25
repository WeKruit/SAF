"""Auditable market-universe discovery for the frozen NFL X-13 batch.

Venue-native identifiers and structured metadata establish identity.  Titles
are retained as evidence but are never the primary contract classifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from prediction_market.kalshi_history import KALSHI_API_ROOT
from prediction_market.polymarket_history import GAMMA_EVENTS_URL
from prediction_market.sports.nfl_x13_capture import (
    KALSHI_PAGE_LIMIT,
    CaptureHttpResponse,
    CaptureRequest,
    CaptureTarget,
    GameCapturePlan,
    ImmutableRequestManifest,
    preserve_capture_response,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
    NativeGameBindingV1,
)


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONDITION_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,255}\Z")
_CANONICAL_COMPARATORS = frozenset({"eq", "gt", "gte", "lt", "lte"})


class X13UniverseError(ValueError):
    """Metadata cannot prove a complete, unambiguous X-13 universe."""


@dataclass(frozen=True, slots=True)
class SourceTimeWindowV1:
    start_ts: int
    end_ts: int

    def __post_init__(self) -> None:
        if type(self.start_ts) is not int or self.start_ts < 0:
            raise X13UniverseError("source window start_ts must be non-negative")
        if type(self.end_ts) is not int or self.end_ts < self.start_ts:
            raise X13UniverseError("source window end_ts must follow start_ts")


@dataclass(frozen=True, slots=True)
class KalshiSeriesSpecV1:
    series_ticker: str
    classification: str
    family: str | None
    expected_scope: str | None
    analysis_eligible: bool
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if _TICKER_RE.fullmatch(self.series_ticker) is None:
            raise X13UniverseError("Kalshi series ticker is malformed")
        if self.classification not in {
            "primitive",
            "composite",
            "excluded",
        }:
            raise X13UniverseError("unknown Kalshi series classification")
        if self.classification == "primitive":
            if self.family is None or not self.analysis_eligible:
                raise X13UniverseError(
                    "primitive series requires an eligible family"
                )
        elif self.classification == "composite":
            if self.family != "same_game_composite":
                raise X13UniverseError(
                    "composite series requires same_game_composite family"
                )
        elif (
            self.family is not None
            or self.analysis_eligible
            or not self.exclusion_reason
        ):
            raise X13UniverseError(
                "excluded series requires only an exclusion reason"
            )


@dataclass(frozen=True, slots=True)
class KalshiSeriesCatalogV1:
    version: str
    series: tuple[KalshiSeriesSpecV1, ...]

    def __post_init__(self) -> None:
        if self.version != "v1":
            raise X13UniverseError("Kalshi series catalog version must be v1")
        tickers = tuple(item.series_ticker for item in self.series)
        if tickers != tuple(sorted(tickers)) or len(set(tickers)) != len(
            tickers
        ):
            raise X13UniverseError(
                "Kalshi series catalog must be unique and sorted"
            )


def _series(
    ticker: str,
    classification: str,
    family: str | None,
    scope: str | None,
    *,
    exclusion_reason: str | None = None,
) -> KalshiSeriesSpecV1:
    return KalshiSeriesSpecV1(
        series_ticker=ticker,
        classification=classification,
        family=family,
        expected_scope=scope,
        analysis_eligible=classification in {"primitive", "composite"},
        exclusion_reason=exclusion_reason,
    )


_SERIES = (
    _series(
        "KXNFL1YDPASS",
        "excluded",
        None,
        "Seattle Passing Attempt from 1-Yard Line",
        exclusion_reason="one-off novelty prop is outside approved families",
    ),
    _series("KXNFL1HSPREAD", "primitive", "period", "1st Half Spread"),
    _series("KXNFL1HTOTAL", "primitive", "period", "1st Half Total"),
    _series("KXNFL1HWINNER", "primitive", "period", "1st Half Winner"),
    _series("KXNFL1QSPREAD", "primitive", "period", "1st Quarter Spread"),
    _series("KXNFL1QTOTAL", "primitive", "period", "1st Quarter Total"),
    _series("KXNFL1QWINNER", "primitive", "period", "1st Quarter Winner"),
    _series("KXNFL2HSPREAD", "primitive", "period", "2nd Half Spread"),
    _series("KXNFL2HTOTAL", "primitive", "period", "2nd Half Total"),
    _series("KXNFL2HWINNER", "primitive", "period", "2nd Half Winner"),
    _series(
        "KXNFL2PTCONV",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="outside approved X-13 proposition families",
    ),
    _series("KXNFL2QSPREAD", "primitive", "period", "2nd Quarter Spread"),
    _series("KXNFL2QTOTAL", "primitive", "period", "2nd Quarter Total"),
    _series("KXNFL2QWINNER", "primitive", "period", "2nd Quarter Winner"),
    _series("KXNFL2TD", "primitive", "player_td", "Touchdowns"),
    _series("KXNFL3QSPREAD", "primitive", "period", "3rd Quarter Spread"),
    _series("KXNFL3QTOTAL", "primitive", "period", "3rd Quarter Total"),
    _series("KXNFL3QWINNER", "primitive", "period", "3rd Quarter Winner"),
    _series(
        "KXNFL4DCONV",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="outside approved X-13 proposition families",
    ),
    _series(
        "KXNFL4DOWNCONV",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="outside approved X-13 proposition families",
    ),
    _series("KXNFL4QSPREAD", "primitive", "period", "4th Quarter Spread"),
    _series("KXNFL4QTOTAL", "primitive", "period", "4th Quarter Total"),
    _series("KXNFL4QWINNER", "primitive", "period", "4th Quarter Winner"),
    _series("KXNFLANYTD", "primitive", "any_td", "Touchdowns"),
    _series(
        "KXNFLCOMEBACK",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="comeback prop is outside approved families",
    ),
    _series(
        "KXNFLCOMBO",
        "excluded",
        None,
        "Combos",
        exclusion_reason=(
            "native rows do not expose structured leg identities"
        ),
    ),
    _series(
        "KXNFLDSTTD",
        "excluded",
        None,
        "Touchdown Specials",
        exclusion_reason="team/DST prop is outside approved player families",
    ),
    _series("KXNFLFIRSTTD", "primitive", "first_td", "Touchdowns"),
    _series(
        "KXNFLFIRSTTDTIME",
        "excluded",
        None,
        "Touchdown Specials",
        exclusion_reason="event-time prop is outside approved families",
    ),
    _series("KXNFLGAME", "primitive", "moneyline", "Game"),
    _series(
        "KXNFLGAMEFG",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="game field-goal prop is outside approved families",
    ),
    _series(
        "KXNFLGAMESACK",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="game sack prop is outside approved families",
    ),
    _series(
        "KXNFLGAMETD",
        "excluded",
        None,
        "Touchdown Specials",
        exclusion_reason="game touchdown count is outside approved families",
    ),
    _series(
        "KXNFLGAMETO",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="game turnover prop is outside approved families",
    ),
    _series(
        "KXNFLHIGHSCOREQ",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="game quarter prop is outside approved period markets",
    ),
    _series(
        "KXNFLLARGELEAD",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="largest-lead prop is outside approved families",
    ),
    _series(
        "KXNFLLARGESTLEAD",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="largest-lead prop is outside approved families",
    ),
    _series(
        "KXNFLLEADCHANGE",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="lead-change prop is outside approved families",
    ),
    _series(
        "KXNFLLONGESTTD",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="longest-touchdown prop is outside approved families",
    ),
    _series(
        "KXNFLNEXTINT",
        "excluded",
        None,
        "Season Specials",
        exclusion_reason="not proven to depend on one frozen game",
    ),
    _series(
        "KXNFLNEXTTD",
        "excluded",
        None,
        "Next Touchdown",
        exclusion_reason="next-touchdown prop is outside approved families",
    ),
    _series(
        "KXNFLNONQBPASS",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="non-QB-pass prop is outside approved families",
    ),
    _series(
        "KXNFLNOSCOREQ",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="scoreless-quarter prop is outside approved families",
    ),
    _series(
        "KXNFLOT",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="overtime prop is outside approved families",
    ),
    _series("KXNFLPASSTDS", "primitive", "player_td", "Passing"),
    _series("KXNFLPASSYDS", "primitive", "player_passing", "Passing"),
    _series("KXNFLPREPACK", "composite", "same_game_composite", None),
    _series("KXNFLPREPACK1HFT", "composite", "same_game_composite", None),
    _series("KXNFLPREPACK1Q1H", "composite", "same_game_composite", None),
    _series("KXNFLPREPACK2ML", "composite", "same_game_composite", None),
    _series("KXNFLPREPACK3ML", "composite", "same_game_composite", None),
    _series("KXNFLPREPACKSGP", "composite", "same_game_composite", None),
    _series(
        "KXNFLPREPACKSGPSPREAD",
        "composite",
        "same_game_composite",
        None,
    ),
    _series(
        "KXNFLPROBOWL",
        "excluded",
        None,
        "Game",
        exclusion_reason="not an X-13 frozen NFL team game",
    ),
    _series("KXNFLREC", "primitive", "player_receptions", "Receptions"),
    _series(
        "KXNFLRECYDS",
        "primitive",
        "player_receiving",
        "Receiving Yards",
    ),
    _series("KXNFLRSHYDS", "primitive", "player_rushing", "Rushing Yards"),
    _series(
        "KXNFLSAFETY",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="safety prop is outside approved families",
    ),
    _series(
        "KXNFLSHORTESTTD",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="shortest-touchdown prop is outside approved families",
    ),
    _series("KXNFLSPREAD", "primitive", "spread", "Spread"),
    _series(
        "KXNFLTEAMFIRSTTD",
        "primitive",
        "first_td",
        "Game Props",
    ),
    _series("KXNFLTEAMTOTAL", "primitive", "team_total", "Team Total"),
    _series("KXNFLTOTAL", "primitive", "total", "Point Total"),
    _series(
        "KXNFLWINMARGIN",
        "excluded",
        None,
        "Game Props",
        exclusion_reason="win-margin prop is outside approved families",
    ),
)

KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1 = KalshiSeriesCatalogV1(
    version="v1",
    series=tuple(sorted(_SERIES, key=lambda item: item.series_ticker)),
)


POLYMARKET_SPORTS_MARKET_TYPE_FAMILIES_V1: Mapping[str, str] = {
    "anytime_touchdowns": "any_td",
    "first_half_moneyline": "period",
    "first_half_spreads": "period",
    "first_half_totals": "period",
    "first_touchdowns": "first_td",
    "game_team_totals": "team_total",
    "h2_team_totals": "team_total",
    "moneyline": "moneyline",
    "parlays": "same_game_composite",
    "passing_touchdowns": "player_td",
    "passing_yards": "player_passing",
    "q1_moneyline": "period",
    "q1_spreads": "period",
    "q1_team_totals": "team_total",
    "q1_totals": "period",
    "q2_moneyline": "period",
    "q2_spreads": "period",
    "q2_team_totals": "team_total",
    "q2_totals": "period",
    "q3_moneyline": "period",
    "q3_spreads": "period",
    "q3_team_totals": "team_total",
    "q3_totals": "period",
    "q4_moneyline": "period",
    "q4_spreads": "period",
    "q4_team_totals": "team_total",
    "q4_totals": "period",
    "receiving_yards": "player_receiving",
    "receptions": "player_receptions",
    "rushing_yards": "player_rushing",
    "second_half_totals": "period",
    "spreads": "spread",
    "team_totals": "team_total",
    "team_totals_away": "team_total",
    "team_totals_home": "team_total",
    "totals": "total",
    "two_plus_touchdowns": "player_td",
}


@dataclass(frozen=True, slots=True)
class PolymarketTokenBindingV1:
    condition_id: str
    outcomes: tuple[str, ...]
    token_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UniverseContractV1:
    contract_id: str
    venue_market_id: str
    raw_contract_ids: tuple[str, ...]
    condition_id: str | None
    logical_market_id: str
    venue: str
    family: str
    period: str
    subject: str
    measure: str
    comparator: str
    line: Decimal | None
    outcomes: tuple[str, ...]
    raw_outcome_labels: tuple[tuple[str, str, str], ...]
    rule_sha256: str
    dependency_game_ids: tuple[str, ...]
    kind: str
    analysis_eligible: bool
    outcome_coverage: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.contract_id) is not str
            or not self.contract_id.startswith(f"{self.venue}:")
            or type(self.venue_market_id) is not str
            or not self.venue_market_id
            or type(self.raw_contract_ids) is not tuple
            or not self.raw_contract_ids
            or len(set(self.raw_contract_ids)) != len(self.raw_contract_ids)
            or type(self.logical_market_id) is not str
            or not self.logical_market_id
            or type(self.outcomes) is not tuple
            or len(self.outcomes) < 2
            or len(set(self.outcomes)) != len(self.outcomes)
            or self.comparator not in _CANONICAL_COMPARATORS
            or tuple(item[0] for item in self.outcome_coverage)
            != self.outcomes
        ):
            raise X13UniverseError("logical contract identity is malformed")
        _require_sha256(self.rule_sha256, "rule_sha256")


@dataclass(frozen=True, slots=True)
class PolymarketInventoryV1:
    game_id: str
    event_id: str
    event_slug: str
    event_title: str
    capture_targets: tuple[CaptureTarget, ...]
    contracts: tuple[UniverseContractV1, ...]
    token_bindings: tuple[PolymarketTokenBindingV1, ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class MetadataExchangeResult:
    payload: object
    response: CaptureHttpResponse
    manifest: ImmutableRequestManifest


def _strict_json(payload: bytes, *, context: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise X13UniverseError(
                    f"{context} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode(),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                X13UniverseError(
                    f"{context} contains non-finite JSON {value}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise X13UniverseError(f"{context} is not strict JSON") from error


def exchange_and_preserve_metadata(
    request: CaptureRequest,
    *,
    requester: Callable[[CaptureRequest], CaptureHttpResponse],
    raw_store_root: str | Path,
    program_root: str | Path,
    response_clock: Callable[[], datetime],
    manifest_writer: Callable[
        [
            CaptureRequest,
            CaptureHttpResponse,
            str | Path,
            str | Path,
            datetime,
        ],
        ImmutableRequestManifest,
    ] = preserve_capture_response,
) -> MetadataExchangeResult:
    """Fetch, timestamp, and preserve one metadata response before parsing."""

    response = requester(request)
    if not isinstance(response, CaptureHttpResponse):
        raise X13UniverseError("metadata requester returned no raw response")
    received_at = response_clock()
    if (
        not isinstance(received_at, datetime)
        or received_at.tzinfo is None
        or received_at.utcoffset() is None
        or received_at.utcoffset().total_seconds() != 0
    ):
        raise X13UniverseError("metadata response clock must return UTC")
    manifest = manifest_writer(
        request,
        response,
        raw_store_root,
        program_root,
        received_at,
    )
    if not isinstance(manifest, ImmutableRequestManifest):
        raise X13UniverseError("metadata writer returned no manifest")
    expected_object_sha = (
        "sha256:" + hashlib.sha256(response.body).hexdigest()
    )
    if (
        manifest.request_sha256 != request.request_sha256
        or manifest.object_sha256 != expected_object_sha
        or manifest.byte_length != len(response.body)
        or manifest.response_received_at_utc != received_at
    ):
        raise X13UniverseError(
            "metadata manifest does not bind exact request/response/clock"
        )
    payload = _strict_json(response.body, context=request.resource)
    return MetadataExchangeResult(
        payload=payload,
        response=response,
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class KalshiSeriesRegistryProofV1:
    series_tickers: tuple[str, ...]
    manifest_sha256: str
    terminal_proof: str = "single_complete_series_response"


@dataclass(frozen=True, slots=True)
class ExcludedContractV1:
    contract_id: str
    venue_market_id: str
    series_ticker: str
    event_ticker: str
    title: str
    reason: str


@dataclass(frozen=True, slots=True)
class CompositeLegV1:
    event_ticker: str
    market_ticker: str
    side: str
    game_id: str


@dataclass(frozen=True, slots=True)
class CompositeClassificationV1:
    venue_market_id: str
    venue_title: str
    series_ticker: str
    event_ticker: str
    legs: tuple[CompositeLegV1, ...]
    dependency_game_ids: tuple[str, ...]
    kind: str
    analysis_eligible: bool


@dataclass(frozen=True, slots=True)
class KalshiSeriesDiscoveryV1:
    game_id: str
    series_ticker: str
    capture_targets: tuple[CaptureTarget, ...]
    contracts: tuple[UniverseContractV1, ...]
    excluded_contracts: tuple[ExcludedContractV1, ...]
    cross_game_inventory: tuple[CompositeClassificationV1, ...]
    manifest_sha256s: tuple[str, ...]
    page_count: int
    terminal_proof: str = "explicit_empty_cursor"


@dataclass(frozen=True, slots=True)
class CapturedKalshiSeriesV1:
    series_spec: KalshiSeriesSpecV1
    markets: tuple[Mapping[str, object], ...]
    request_sha256s: tuple[str, ...]
    manifest_sha256s: tuple[str, ...]
    page_count: int
    terminal_proof: str = "explicit_empty_cursor"


@dataclass(frozen=True, slots=True)
class UniverseGameV1:
    game_id: str
    source_window: SourceTimeWindowV1
    polymarket_event_id: str
    polymarket_event_slug: str
    polymarket_event_title: str
    primitive_targets: tuple[CaptureTarget, ...]
    same_game_composite_targets: tuple[CaptureTarget, ...]
    cross_game_inventory: tuple[CompositeClassificationV1, ...]
    excluded_contracts: tuple[ExcludedContractV1, ...]
    unknown_contract_ids: tuple[str, ...]
    contracts: tuple[UniverseContractV1, ...]
    polymarket_token_bindings: tuple[PolymarketTokenBindingV1, ...]
    kalshi_series_tickers: tuple[str, ...]
    metadata_manifest_sha256s: tuple[str, ...]

    @property
    def capture_targets(self) -> tuple[CaptureTarget, ...]:
        return tuple(
            sorted(
                (
                    *self.primitive_targets,
                    *self.same_game_composite_targets,
                ),
                key=lambda target: target.contract_id,
            )
        )

    @property
    def capture_plan(self) -> GameCapturePlan:
        return GameCapturePlan(
            game_id=self.game_id,
            polymarket_event_id=self.polymarket_event_id,
            polymarket_event_slug=self.polymarket_event_slug,
            polymarket_event_title=self.polymarket_event_title,
            start_ts=self.source_window.start_ts,
            end_ts=self.source_window.end_ts,
            targets=self.capture_targets,
        )


@dataclass(frozen=True, slots=True)
class X13UniverseBatchV1:
    games: tuple[UniverseGameV1, ...]
    capture_plans: tuple[GameCapturePlan, ...]
    series_registry_proof: KalshiSeriesRegistryProofV1
    schema_version: str = "nfl-x13-universe-v1"

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(game.game_id for game in self.games)

    @property
    def universe_id(self) -> str:
        return _sha256(_batch_payload(self, include_universe_id=False))

    @property
    def canonical_payload(self) -> dict[str, object]:
        return _batch_payload(self, include_universe_id=True)


def _require_sha256(value: str, field: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise X13UniverseError(f"{field} must be a canonical SHA-256")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise X13UniverseError("metadata is not canonical JSON") from error


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _optional_decimal(value: object, field: str) -> Decimal | None:
    if value is None:
        return None
    if type(value) not in {int, str, float}:
        raise X13UniverseError(f"{field} is not a decimal scalar")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise X13UniverseError(f"{field} is not decimal") from error
    if not result.is_finite():
        raise X13UniverseError(f"{field} must be finite")
    return result


_FAMILY_MEASURE = {
    "moneyline": "winner",
    "spread": "score_margin",
    "total": "combined_points",
    "period": "period_result",
    "team_total": "team_points",
    "player_passing": "passing_yards",
    "player_rushing": "rushing_yards",
    "player_receiving": "receiving_yards",
    "player_receptions": "receptions",
    "player_td": "touchdowns",
    "first_td": "first_touchdown",
    "any_td": "any_touchdown",
    "same_game_composite": "structured_legs",
}


def _polymarket_period(market_type: str) -> str:
    if market_type.startswith("first_half_"):
        return "first_half"
    if market_type.startswith("second_half_"):
        return "second_half"
    for prefix in ("q1_", "q2_", "q3_", "q4_"):
        if market_type.startswith(prefix):
            return prefix[:2]
    return "game"


def _kalshi_period(series_ticker: str) -> str:
    for token, period in (
        ("1Q", "q1"),
        ("2Q", "q2"),
        ("3Q", "q3"),
        ("4Q", "q4"),
        ("1H", "first_half"),
        ("2H", "second_half"),
    ):
        if series_ticker.startswith(f"KXNFL{token}"):
            return period
    return "game"


def _kalshi_line(market: Mapping[str, object]) -> Decimal | None:
    floor = _optional_decimal(market.get("floor_strike"), "floor_strike")
    cap = _optional_decimal(market.get("cap_strike"), "cap_strike")
    if floor is not None and cap is not None and floor != cap:
        raise X13UniverseError(
            "Kalshi interval strike cannot fit proposition AST v1"
        )
    return floor if floor is not None else cap


def _polymarket_comparator(
    *,
    family: str,
    line: Decimal | None,
) -> str:
    if line is None:
        return "eq"
    if family not in {
        "spread",
        "total",
        "period",
        "team_total",
        "player_passing",
        "player_rushing",
        "player_receiving",
        "player_receptions",
        "player_td",
    }:
        raise X13UniverseError(
            "Polymarket line is invalid for proposition family"
        )
    return "gt"


def _kalshi_comparator(
    market: Mapping[str, object],
    *,
    line: Decimal | None,
) -> str:
    strike_type = market.get("strike_type")
    if strike_type == "greater":
        if line is None:
            raise X13UniverseError(
                "Kalshi greater strike_type requires a line"
            )
        return "gt"
    if strike_type == "less":
        if line is None:
            raise X13UniverseError(
                "Kalshi less strike_type requires a line"
            )
        return "lt"
    if strike_type == "structured":
        return "gt" if line is not None else "eq"
    if strike_type == "custom":
        if line is not None:
            raise X13UniverseError(
                "Kalshi custom strike_type cannot carry one scalar line"
            )
        return "eq"
    raise X13UniverseError(
        f"unknown Kalshi strike_type: {strike_type!r}"
    )


def _validate_exchange_result(
    value: object,
    *,
    context: str,
    request: CaptureRequest,
) -> MetadataExchangeResult:
    if not isinstance(value, MetadataExchangeResult):
        raise X13UniverseError(f"{context} was not immutably captured")
    if not isinstance(value.response, CaptureHttpResponse):
        raise X13UniverseError(f"{context} response evidence is malformed")
    if not isinstance(value.manifest, ImmutableRequestManifest):
        raise X13UniverseError(f"{context} manifest evidence is malformed")
    _require_sha256(value.manifest.request_sha256, "request_sha256")
    _require_sha256(value.manifest.object_sha256, "object_sha256")
    _require_sha256(value.manifest.manifest_sha256, "manifest_sha256")
    try:
        response_payload = json.loads(value.response.body)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise X13UniverseError(f"{context} response is not JSON") from error
    if _canonical_bytes(response_payload) != _canonical_bytes(value.payload):
        raise X13UniverseError(
            f"{context} parsed payload differs from sealed response"
        )
    expected_object_sha = (
        "sha256:" + hashlib.sha256(value.response.body).hexdigest()
    )
    received_at = value.manifest.response_received_at_utc
    if (
        value.manifest.request_sha256 != request.request_sha256
        or value.manifest.object_sha256 != expected_object_sha
        or value.manifest.byte_length != len(value.response.body)
        or received_at.tzinfo is None
        or received_at.utcoffset() is None
        or received_at.utcoffset().total_seconds() != 0
    ):
        raise X13UniverseError(
            f"{context} manifest is not bound to exact request/response"
        )
    return value


_GAME_LEVEL_SCOPES = frozenset(
    {
        "1st Half Spread",
        "1st Half Total",
        "1st Half Winner",
        "1st Quarter Spread",
        "1st Quarter Total",
        "1st Quarter Winner",
        "2nd Half Spread",
        "2nd Half Total",
        "2nd Half Winner",
        "2nd Quarter Spread",
        "2nd Quarter Total",
        "2nd Quarter Winner",
        "3rd Quarter Spread",
        "3rd Quarter Total",
        "3rd Quarter Winner",
        "4th Quarter Spread",
        "4th Quarter Total",
        "4th Quarter Winner",
        "Combo",
        "Combos",
        "Game",
        "Game Props",
        "Next Touchdown",
        "Passing",
        "Point Total",
        "Receiving Yards",
        "Receptions",
        "Rushing Yards",
        "Spread",
        "Team Total",
        "Touchdown Specials",
        "Touchdowns",
    }
)


def validate_kalshi_series_registry(
    payload: object,
    *,
    manifest_sha256: str,
) -> KalshiSeriesRegistryProofV1:
    """Prove every frozen series against Kalshi's structured series registry."""

    _require_sha256(manifest_sha256, "manifest_sha256")
    if type(payload) is not dict or set(payload) != {"series"}:
        raise X13UniverseError("Kalshi series registry payload is malformed")
    rows = payload["series"]
    if type(rows) is not list:
        raise X13UniverseError("Kalshi series registry rows must be a list")
    observed: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise X13UniverseError(f"Kalshi series[{index}] is not an object")
        ticker = row.get("ticker")
        if type(ticker) is not str or _TICKER_RE.fullmatch(ticker) is None:
            raise X13UniverseError("Kalshi series lacks a stable ticker")
        if ticker in observed:
            raise X13UniverseError("Kalshi series registry duplicates ticker")
        observed[ticker] = row
    catalog = {
        spec.series_ticker: spec
        for spec in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
    }
    missing = set(catalog) - set(observed)
    if missing:
        raise X13UniverseError(
            f"Kalshi series registry is missing catalog series: {sorted(missing)!r}"
        )
    for ticker, spec in catalog.items():
        row = observed[ticker]
        if row.get("category") != "Sports":
            raise X13UniverseError(
                f"Kalshi series {ticker} escaped Sports category"
            )
        if spec.expected_scope is not None:
            metadata = row.get("product_metadata")
            if (
                type(metadata) is not dict
                or metadata.get("scope") != spec.expected_scope
            ):
                raise X13UniverseError(
                    f"Kalshi series {ticker} scope changed"
                )
        sources = row.get("settlement_sources")
        if (
            type(sources) is not list
            or not any(
                type(source) is dict
                and type(source.get("url")) is str
                and "nfl.com" in source["url"].lower()
                for source in sources
            )
        ):
            raise X13UniverseError(
                f"Kalshi series {ticker} lacks NFL settlement provenance"
            )
    for ticker, row in observed.items():
        if ticker in catalog or not ticker.startswith("KXNFL"):
            continue
        metadata = row.get("product_metadata")
        scope = metadata.get("scope") if type(metadata) is dict else None
        if scope in _GAME_LEVEL_SCOPES:
            raise X13UniverseError(
                f"unknown NFL game-level series: {ticker}"
            )
    return KalshiSeriesRegistryProofV1(
        series_tickers=tuple(sorted(catalog)),
        manifest_sha256=manifest_sha256,
    )


def _json_string_tuple(
    value: object,
    field: str,
    *,
    minimum_length: int,
) -> tuple[str, ...]:
    if type(value) is not str:
        raise X13UniverseError(f"{field} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise X13UniverseError(f"{field} is invalid JSON") from error
    if (
        type(parsed) is not list
        or len(parsed) < minimum_length
        or any(type(item) is not str or not item for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise X13UniverseError(f"{field} must contain unique strings")
    return tuple(parsed)


def classify_polymarket_event(
    binding: NativeGameBindingV1,
    payload: object,
    *,
    manifest_sha256: str,
) -> PolymarketInventoryV1:
    """Classify an exact Gamma event only from frozen and structured fields."""

    _require_sha256(manifest_sha256, "manifest_sha256")
    if type(payload) is not dict:
        raise X13UniverseError("Gamma event payload must be an object")
    if payload.get("id") != binding.polymarket_event_id:
        raise X13UniverseError("Gamma event ID does not match frozen binding")
    slug = payload.get("slug")
    title = payload.get("title")
    sport = payload.get("sport")
    event_series = payload.get("series")
    nfl_structured_identity = (
        type(sport) is dict
        and sport.get("sport") == "nfl"
    ) or payload.get("seriesSlug") == "nfl-2025" or (
        type(event_series) is list
        and any(
            type(series) is dict
            and series.get("slug") == "nfl-2025"
            for series in event_series
        )
    )
    markets = payload.get("markets")
    if (
        type(slug) is not str
        or not slug
        or type(title) is not str
        or not title
        or not nfl_structured_identity
        or type(markets) is not list
        or not markets
    ):
        raise X13UniverseError("Gamma event identity is incomplete")

    targets_with_tokens: list[
        tuple[
            int,
            CaptureTarget,
            UniverseContractV1,
            PolymarketTokenBindingV1,
        ]
    ] = []
    seen_market_ids: set[str] = set()
    seen_conditions: set[str] = set()
    seen_tokens: set[str] = set()
    primary_found = False
    for index, market in enumerate(markets):
        if type(market) is not dict:
            raise X13UniverseError(f"Gamma market[{index}] is not an object")
        market_id = market.get("id")
        question = market.get("question")
        condition_id = market.get("conditionId")
        market_type = market.get("sportsMarketType")
        if (
            type(market_id) is not str
            or not market_id.isdigit()
            or type(question) is not str
            or not question
            or type(condition_id) is not str
            or _CONDITION_RE.fullmatch(condition_id) is None
            or type(market_type) is not str
        ):
            raise X13UniverseError("Gamma market identity is incomplete")
        if market_id in seen_market_ids or condition_id in seen_conditions:
            raise X13UniverseError("Gamma market identity is duplicated")
        seen_market_ids.add(market_id)
        seen_conditions.add(condition_id)
        family = POLYMARKET_SPORTS_MARKET_TYPE_FAMILIES_V1.get(market_type)
        if family is None:
            raise X13UniverseError(
                f"unknown Polymarket sportsMarketType: {market_type!r}"
            )
        outcomes = _json_string_tuple(
            market.get("outcomes"),
            "outcomes",
            minimum_length=2,
        )
        token_ids = _json_string_tuple(
            market.get("clobTokenIds"),
            "clobTokenIds",
            minimum_length=1,
        )
        if len(outcomes) != len(token_ids):
            raise X13UniverseError("token/outcome cardinality does not match")
        if set(token_ids) & seen_tokens:
            raise X13UniverseError("Gamma token identity is duplicated")
        seen_tokens.update(token_ids)
        kind = (
            "same_game_composite"
            if family == "same_game_composite"
            else "primitive"
        )
        target = CaptureTarget(
            contract_id=f"polymarket:{condition_id}",
            venue="polymarket",
            venue_market_id=market_id,
            condition_id=condition_id,
            venue_title=question,
            kalshi_series_ticker=None,
            kalshi_event_ticker=None,
            family=family,
            dependency_game_ids=(binding.native_game_id,),
            kind=kind,
            analysis_eligible=True,
        )
        token_binding = PolymarketTokenBindingV1(
            condition_id=condition_id,
            outcomes=outcomes,
            token_ids=token_ids,
        )
        subject_value = market.get("groupItemTitle")
        subject = (
            subject_value
            if type(subject_value) is str and subject_value
            else question
        )
        line = _optional_decimal(market.get("line"), "line")
        comparator = _polymarket_comparator(
            family=family,
            line=line,
        )
        logical_market_id = (
            f"polymarket:{binding.native_game_id}:winner"
            if market_id == binding.polymarket_market_id
            else f"polymarket:{condition_id}"
        )
        contract = UniverseContractV1(
            contract_id=target.contract_id,
            venue_market_id=market_id,
            raw_contract_ids=(target.contract_id,),
            condition_id=condition_id,
            logical_market_id=logical_market_id,
            venue="polymarket",
            family=family,
            period=_polymarket_period(market_type),
            subject=subject,
            measure=_FAMILY_MEASURE[family],
            comparator=comparator,
            line=line,
            outcomes=outcomes,
            raw_outcome_labels=(),
            rule_sha256=_sha256(
                {
                    "description": market.get("description"),
                    "resolutionSource": market.get("resolutionSource"),
                    "sportsMarketType": market_type,
                }
            ),
            dependency_game_ids=(binding.native_game_id,),
            kind=kind,
            analysis_eligible=True,
            outcome_coverage=tuple(
                (outcome, "unavailable_no_trades")
                for outcome in outcomes
            ),
        )
        targets_with_tokens.append(
            (int(market_id), target, contract, token_binding)
        )
        if market_id == binding.polymarket_market_id:
            if (
                condition_id != binding.polymarket_condition_id
                or market_type != "moneyline"
            ):
                raise X13UniverseError(
                    "primary moneyline does not match frozen binding"
                )
            primary_found = True
    if not primary_found:
        raise X13UniverseError("primary moneyline is absent from Gamma event")
    ordered = sorted(targets_with_tokens, key=lambda item: item[0])
    return PolymarketInventoryV1(
        game_id=binding.native_game_id,
        event_id=binding.polymarket_event_id,
        event_slug=slug,
        event_title=title,
        capture_targets=tuple(item[1] for item in ordered),
        contracts=tuple(item[2] for item in ordered),
        token_bindings=tuple(item[3] for item in ordered),
        manifest_sha256=manifest_sha256,
    )


def derive_kalshi_event_ticker(
    series_ticker: str,
    binding: NativeGameBindingV1,
) -> str:
    """Return the explicit series/game binding, preserving the JAX/JAC alias."""

    prefix = "KXNFLGAME-"
    if not binding.kalshi_event_ticker.startswith(prefix):
        raise X13UniverseError("frozen Kalshi game binding is malformed")
    if _TICKER_RE.fullmatch(series_ticker) is None:
        raise X13UniverseError("Kalshi series ticker is malformed")
    return f"{series_ticker}-{binding.kalshi_event_ticker.removeprefix(prefix)}"


def _composite_raw_legs(
    market: Mapping[str, object],
) -> tuple[tuple[str, str, str], ...]:
    selected = market.get("mve_selected_legs")
    if selected is not None:
        if type(selected) is not list or not selected:
            raise X13UniverseError("MVE selected legs are malformed")
        result: list[tuple[str, str, str]] = []
        for leg in selected:
            if type(leg) is not dict:
                raise X13UniverseError("MVE selected leg is not an object")
            event = leg.get("event_ticker")
            ticker = leg.get("market_ticker")
            side = leg.get("side")
            if (
                type(event) is not str
                or _TICKER_RE.fullmatch(event) is None
                or type(ticker) is not str
                or _TICKER_RE.fullmatch(ticker) is None
                or type(side) is not str
                or side.lower() not in {"yes", "no"}
            ):
                raise X13UniverseError("MVE selected leg identity is malformed")
            result.append((event, ticker, side.lower()))
        return tuple(result)

    custom = market.get("custom_strike")
    if type(custom) is not dict:
        raise X13UniverseError("composite lacks structured leg metadata")
    raw_events = custom.get("Associated Events")
    raw_markets = custom.get("Associated Markets")
    raw_sides = custom.get("Associated Market Sides")
    if not all(
        type(item) is str and item.strip()
        for item in (raw_events, raw_markets, raw_sides)
    ):
        raise X13UniverseError("composite lacks structured Associated legs")
    events = tuple(item.strip() for item in raw_events.split(","))
    markets = tuple(item.strip() for item in raw_markets.split(","))
    sides = tuple(item.strip().lower() for item in raw_sides.split(","))
    if (
        not events
        or len(markets) != len(sides)
        or any(side not in {"yes", "no"} for side in sides)
    ):
        raise X13UniverseError("composite Associated leg cardinality is invalid")
    result: list[tuple[str, str, str]] = []
    for market_ticker, side in zip(markets, sides, strict=True):
        matching_events = tuple(
            event
            for event in events
            if market_ticker.startswith(f"{event}-")
        )
        if not matching_events:
            raise X13UniverseError(
                f"unknown MVE leg: {market_ticker}"
            )
        if len(matching_events) != 1:
            raise X13UniverseError(
                "composite market leg has ambiguous Associated Event"
            )
        result.append((matching_events[0], market_ticker, side))
    return tuple(result)


def _composite_candidate_events(
    market: Mapping[str, object],
) -> tuple[str, ...]:
    selected = market.get("mve_selected_legs")
    if selected is not None:
        if type(selected) is not list or not selected:
            raise X13UniverseError("MVE selected legs are malformed")
        events = tuple(
            leg.get("event_ticker")
            for leg in selected
            if type(leg) is dict
        )
    else:
        custom = market.get("custom_strike")
        raw_events = (
            custom.get("Associated Events")
            if type(custom) is dict
            else None
        )
        if type(raw_events) is not str or not raw_events.strip():
            raise X13UniverseError(
                "composite lacks structured Associated Events"
            )
        events = tuple(item.strip() for item in raw_events.split(","))
    if (
        not events
        or any(
            type(event) is not str
            or _TICKER_RE.fullmatch(event) is None
            for event in events
        )
    ):
        raise X13UniverseError("composite candidate event identity is malformed")
    return events


def classify_cross_game_mve_market(
    market: object,
    *,
    series_ticker: str,
    event_to_game_id: Mapping[str, str],
) -> CompositeClassificationV1:
    """Classify a structured Kalshi composite without reading its title."""

    if type(market) is not dict:
        raise X13UniverseError("composite market must be an object")
    ticker = market.get("ticker")
    event_ticker = market.get("event_ticker")
    title = market.get("title")
    if (
        type(ticker) is not str
        or _TICKER_RE.fullmatch(ticker) is None
        or type(event_ticker) is not str
        or not event_ticker.startswith(f"{series_ticker}-")
        or not ticker.startswith(f"{event_ticker}-")
        or type(title) is not str
        or not title
    ):
        raise X13UniverseError("composite native identity is malformed")
    raw_legs = _composite_raw_legs(market)
    legs: list[CompositeLegV1] = []
    for event, market_ticker, side in raw_legs:
        game_id = event_to_game_id.get(event)
        if game_id is None:
            raise X13UniverseError(f"unknown MVE leg: {event}")
        if (
            _TICKER_RE.fullmatch(market_ticker) is None
            or not market_ticker.startswith(f"{event}-")
        ):
            raise X13UniverseError("MVE market leg does not match its event")
        legs.append(
            CompositeLegV1(
                event_ticker=event,
                market_ticker=market_ticker,
                side=side,
                game_id=game_id,
            )
        )
    dependencies = tuple(sorted({leg.game_id for leg in legs}))
    if not dependencies:
        raise X13UniverseError("composite has no game dependency")
    same_game = len(dependencies) == 1
    return CompositeClassificationV1(
        venue_market_id=ticker,
        venue_title=title,
        series_ticker=series_ticker,
        event_ticker=event_ticker,
        legs=tuple(legs),
        dependency_game_ids=dependencies,
        kind="same_game_composite" if same_game else "cross_game_inventory",
        analysis_eligible=same_game,
    )


def _event_to_game_registry() -> dict[str, str]:
    result: dict[str, str] = {}
    for binding in X13_GAME_BINDINGS:
        for spec in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series:
            if spec.classification == "composite":
                continue
            event_ticker = derive_kalshi_event_ticker(
                spec.series_ticker,
                binding,
            )
            prior = result.setdefault(event_ticker, binding.native_game_id)
            if prior != binding.native_game_id:
                raise X13UniverseError("Kalshi event registry is ambiguous")
    return result


def build_kalshi_event_dependency_registry(
    captured_series: Sequence[CapturedKalshiSeriesV1],
) -> dict[str, str]:
    """Bind venue event IDs to games using captured KXNFLGAME evidence."""

    by_series: dict[str, CapturedKalshiSeriesV1] = {}
    for captured in captured_series:
        if not isinstance(captured, CapturedKalshiSeriesV1):
            raise X13UniverseError(
                "Kalshi dependency registry requires captured series"
            )
        ticker = captured.series_spec.series_ticker
        if ticker in by_series:
            raise X13UniverseError(
                "Kalshi dependency registry duplicates a series capture"
            )
        if (
            captured.page_count < 1
            or captured.page_count != len(captured.request_sha256s)
            or captured.page_count != len(captured.manifest_sha256s)
            or captured.terminal_proof != "explicit_empty_cursor"
        ):
            raise X13UniverseError(
                "Kalshi dependency registry lacks terminal capture proof"
            )
        by_series[ticker] = captured
    game_capture = by_series.get("KXNFLGAME")
    if game_capture is None:
        raise X13UniverseError(
            "Kalshi dependency registry lacks KXNFLGAME proof"
        )

    frozen_games = {
        binding.kalshi_event_ticker: binding.native_game_id
        for binding in X13_GAME_BINDINGS
    }
    game_rows: dict[str, set[str]] = {}
    for market in game_capture.markets:
        market_ticker, event_ticker, _ = _market_identity(
            market,
            series_ticker="KXNFLGAME",
        )
        game_rows.setdefault(event_ticker, set()).add(market_ticker)
    suffix_dependencies: dict[str, str] = {}
    for event_ticker, market_tickers in game_rows.items():
        if len(market_tickers) != 2:
            raise X13UniverseError(
                "KXNFLGAME event must prove exactly two outcome tickers"
            )
        suffix = event_ticker.removeprefix("KXNFLGAME-")
        if not suffix or suffix == event_ticker:
            raise X13UniverseError("KXNFLGAME event suffix is malformed")
        dependency_id = frozen_games.get(
            event_ticker,
            f"kalshi_game:{event_ticker}",
        )
        prior = suffix_dependencies.setdefault(suffix, dependency_id)
        if prior != dependency_id:
            raise X13UniverseError(
                "Kalshi game event suffix is ambiguous"
            )

    result = _event_to_game_registry()
    for captured in captured_series:
        spec = captured.series_spec
        if spec.classification == "composite":
            continue
        prefix = f"{spec.series_ticker}-"
        for market in captured.markets:
            _, event_ticker, _ = _market_identity(
                market,
                series_ticker=spec.series_ticker,
            )
            if event_ticker in result:
                continue
            if not event_ticker.startswith(prefix):
                raise X13UniverseError(
                    "Kalshi event escaped captured series identity"
                )
            dependency_id = suffix_dependencies.get(
                event_ticker[len(prefix) :]
            )
            if dependency_id is not None:
                result[event_ticker] = dependency_id
    return result


def _market_identity(
    market: object,
    *,
    series_ticker: str,
) -> tuple[str, str, str]:
    if type(market) is not dict:
        raise X13UniverseError("Kalshi market must be an object")
    ticker = market.get("ticker")
    event_ticker = market.get("event_ticker")
    title = market.get("title")
    response_series = market.get("series_ticker")
    if (
        type(ticker) is not str
        or _TICKER_RE.fullmatch(ticker) is None
        or type(event_ticker) is not str
        or _TICKER_RE.fullmatch(event_ticker) is None
        or not event_ticker.startswith(f"{series_ticker}-")
        or not ticker.startswith(f"{event_ticker}-")
        or type(title) is not str
        or not title
        or market.get("market_type") != "binary"
        or (
            response_series is not None
            and response_series != series_ticker
        )
    ):
        raise X13UniverseError(
            "Kalshi market escaped requested series identity"
        )
    return ticker, event_ticker, title


def _series_request(
    *,
    game_id: str,
    series_ticker: str,
    cursor: str | None,
) -> CaptureRequest:
    params: dict[str, object] = {
        "limit": KALSHI_PAGE_LIMIT,
        "series_ticker": series_ticker,
    }
    if cursor is not None:
        params["cursor"] = cursor
    return CaptureRequest(
        game_id=game_id,
        source="kalshi",
        resource="x13_universe_series_markets",
        scope_id=series_ticker,
        url=f"{KALSHI_API_ROOT}/historical/markets",
        params=tuple(sorted(params.items())),
        source_cursor=f"request_cursor:{cursor or 'ROOT'}",
        coverage=(
            "complete historical series cursor chain;"
            f"series={series_ticker}"
        ),
    )


def capture_kalshi_series_metadata(
    series_spec: KalshiSeriesSpecV1,
    *,
    exchange: Callable[[CaptureRequest], MetadataExchangeResult],
    max_pages: int = 100,
) -> CapturedKalshiSeriesV1:
    """Capture one complete series once for reuse across all frozen games."""

    if max_pages < 1:
        raise X13UniverseError("max_pages must be positive")
    current_cursor: str | None = None
    seen_cursors: set[str] = set()
    observed: dict[str, tuple[bytes, Mapping[str, object]]] = {}
    manifests: list[str] = []
    request_hashes: list[str] = []
    for page_number in range(1, max_pages + 1):
        request = _series_request(
            game_id=X13_GAME_IDS[0],
            series_ticker=series_spec.series_ticker,
            cursor=current_cursor,
        )
        exchange_result = _validate_exchange_result(
            exchange(request),
            context=f"{series_spec.series_ticker} page {page_number}",
            request=request,
        )
        manifests.append(exchange_result.manifest.manifest_sha256)
        request_hashes.append(request.request_sha256)
        payload = exchange_result.payload
        if type(payload) is not dict or set(payload) != {
            "markets",
            "cursor",
        }:
            raise X13UniverseError("Kalshi market page fields are malformed")
        markets = payload["markets"]
        response_cursor = payload["cursor"]
        if type(markets) is not list or type(response_cursor) is not str:
            raise X13UniverseError("Kalshi market page is malformed")
        for market in markets:
            ticker, _, _ = _market_identity(
                market,
                series_ticker=series_spec.series_ticker,
            )
            canonical = _canonical_bytes(market)
            prior = observed.get(ticker)
            if prior is not None:
                if prior[0] != canonical:
                    raise X13UniverseError(
                        "Kalshi series has conflicting duplicate ticker"
                    )
                continue
            observed[ticker] = (canonical, market)
        if response_cursor == "":
            break
        if (
            response_cursor == current_cursor
            or response_cursor in seen_cursors
        ):
            raise X13UniverseError("Kalshi series returned a repeated cursor")
        seen_cursors.add(response_cursor)
        current_cursor = response_cursor
    else:
        raise X13UniverseError("Kalshi series lacks terminal cursor proof")
    return CapturedKalshiSeriesV1(
        series_spec=series_spec,
        markets=tuple(
            item[1] for _, item in sorted(observed.items())
        ),
        request_sha256s=tuple(request_hashes),
        manifest_sha256s=tuple(manifests),
        page_count=len(manifests),
    )


def classify_captured_kalshi_series_for_game(
    binding: NativeGameBindingV1,
    captured: CapturedKalshiSeriesV1,
    *,
    event_to_game_id: Mapping[str, str] | None = None,
) -> KalshiSeriesDiscoveryV1:
    """Select one frozen game's contracts from a sealed complete series."""

    if not isinstance(captured, CapturedKalshiSeriesV1):
        raise X13UniverseError("captured series evidence is missing")
    series_spec = captured.series_spec
    registry = dict(event_to_game_id or _event_to_game_registry())
    expected_event = (
        None
        if series_spec.classification == "composite"
        else derive_kalshi_event_ticker(series_spec.series_ticker, binding)
    )
    selected: dict[str, Mapping[str, object]] = {}
    excluded: list[ExcludedContractV1] = []
    cross_game: list[CompositeClassificationV1] = []
    for market in captured.markets:
        ticker, event_ticker, title = _market_identity(
            market,
            series_ticker=series_spec.series_ticker,
        )
        if series_spec.classification == "composite":
            raw_events = _composite_candidate_events(market)
            if not any(
                registry.get(event) == binding.native_game_id
                for event in raw_events
            ):
                continue
            classification = classify_cross_game_mve_market(
                market,
                series_ticker=series_spec.series_ticker,
                event_to_game_id=registry,
            )
            if classification.kind == "cross_game_inventory":
                cross_game.append(classification)
            else:
                selected[ticker] = market
            continue
        if event_ticker != expected_event:
            continue
        if series_spec.classification == "excluded":
            excluded.append(
                ExcludedContractV1(
                    contract_id=f"kalshi:{ticker}",
                    venue_market_id=ticker,
                    series_ticker=series_spec.series_ticker,
                    event_ticker=event_ticker,
                    title=title,
                    reason=series_spec.exclusion_reason or "excluded",
                )
            )
        else:
            selected[ticker] = market

    if series_spec.series_ticker == "KXNFLGAME" and set(selected) != {
        binding.kalshi_away_ticker,
        binding.kalshi_home_ticker,
    }:
        raise X13UniverseError(
            "Kalshi winner market does not match frozen team tickers"
        )

    targets: list[CaptureTarget] = []
    contracts: list[UniverseContractV1] = []
    winner_market_rows: dict[str, Mapping[str, object]] = {}
    for ticker, market in sorted(selected.items()):
        _, event_ticker, title = _market_identity(
            market,
            series_ticker=series_spec.series_ticker,
        )
        if series_spec.classification == "composite":
            classification = classify_cross_game_mve_market(
                market,
                series_ticker=series_spec.series_ticker,
                event_to_game_id=registry,
            )
            family = "same_game_composite"
            kind = "same_game_composite"
            dependencies = classification.dependency_game_ids
        else:
            if series_spec.family is None:
                raise X13UniverseError("eligible series lacks a family")
            family = series_spec.family
            kind = "primitive"
            dependencies = (binding.native_game_id,)
        target = CaptureTarget(
            contract_id=f"kalshi:{ticker}",
            venue="kalshi",
            venue_market_id=ticker,
            condition_id=None,
            venue_title=title,
            kalshi_series_ticker=series_spec.series_ticker,
            kalshi_event_ticker=event_ticker,
            family=family,
            dependency_game_ids=dependencies,
            kind=kind,
            analysis_eligible=True,
        )
        targets.append(target)
        yes_label = market.get("yes_sub_title")
        no_label = market.get("no_sub_title")
        if (
            type(yes_label) is not str
            or not yes_label
            or type(no_label) is not str
            or not no_label
        ):
            raise X13UniverseError(
                "Kalshi market lacks explicit yes/no outcome labels"
            )
        if series_spec.series_ticker == "KXNFLGAME":
            winner_market_rows[ticker] = market
            continue
        line = _kalshi_line(market)
        comparator = _kalshi_comparator(market, line=line)
        contracts.append(
            UniverseContractV1(
                contract_id=target.contract_id,
                venue_market_id=ticker,
                raw_contract_ids=(ticker,),
                condition_id=None,
                logical_market_id=f"kalshi:{ticker}",
                venue="kalshi",
                family=family,
                period=_kalshi_period(series_spec.series_ticker),
                subject=title,
                measure=_FAMILY_MEASURE[family],
                comparator=comparator,
                line=line,
                outcomes=("Yes", "No"),
                raw_outcome_labels=((ticker, yes_label, no_label),),
                rule_sha256=_sha256(
                    {
                        "custom_strike": market.get("custom_strike"),
                        "rules_primary": market.get("rules_primary"),
                        "rules_secondary": market.get("rules_secondary"),
                    }
                ),
                dependency_game_ids=dependencies,
                kind=kind,
                analysis_eligible=True,
                outcome_coverage=(
                    ("Yes", "unavailable_no_trades"),
                    ("No", "unavailable_no_trades"),
                ),
            )
        )
    if series_spec.series_ticker == "KXNFLGAME":
        raw_ids = (
            binding.kalshi_away_ticker,
            binding.kalshi_home_ticker,
        )
        raw_labels: list[tuple[str, str, str]] = []
        rule_material: list[object] = []
        for ticker in raw_ids:
            market = winner_market_rows[ticker]
            yes_label = market.get("yes_sub_title")
            no_label = market.get("no_sub_title")
            if (
                type(yes_label) is not str
                or not yes_label
                or type(no_label) is not str
                or not no_label
            ):
                raise X13UniverseError(
                    "Kalshi winner lacks explicit raw outcome labels"
                )
            raw_labels.append((ticker, yes_label, no_label))
            rule_material.append(
                {
                    "custom_strike": market.get("custom_strike"),
                    "rules_primary": market.get("rules_primary"),
                    "rules_secondary": market.get("rules_secondary"),
                    "ticker": ticker,
                }
            )
        outcomes = (binding.away_team, binding.home_team)
        contracts.append(
            UniverseContractV1(
                contract_id=f"kalshi:{binding.kalshi_event_ticker}",
                venue_market_id=binding.kalshi_event_ticker,
                raw_contract_ids=raw_ids,
                condition_id=None,
                logical_market_id=(
                    f"kalshi:{binding.native_game_id}:winner"
                ),
                venue="kalshi",
                family="moneyline",
                period="game",
                subject=f"{binding.away_team} at {binding.home_team}",
                measure="winner",
                comparator="eq",
                line=None,
                outcomes=outcomes,
                raw_outcome_labels=tuple(raw_labels),
                rule_sha256=_sha256(rule_material),
                dependency_game_ids=(binding.native_game_id,),
                kind="primitive",
                analysis_eligible=True,
                outcome_coverage=tuple(
                    (outcome, "unavailable_no_trades")
                    for outcome in outcomes
                ),
            )
        )
    return KalshiSeriesDiscoveryV1(
        game_id=binding.native_game_id,
        series_ticker=series_spec.series_ticker,
        capture_targets=tuple(targets),
        contracts=tuple(contracts),
        excluded_contracts=tuple(
            sorted(excluded, key=lambda item: item.contract_id)
        ),
        cross_game_inventory=tuple(
            sorted(cross_game, key=lambda item: item.venue_market_id)
        ),
        manifest_sha256s=captured.manifest_sha256s,
        page_count=captured.page_count,
    )


def discover_kalshi_series_for_game(
    binding: NativeGameBindingV1,
    series_spec: KalshiSeriesSpecV1,
    *,
    exchange: Callable[[CaptureRequest], MetadataExchangeResult],
    max_pages: int = 100,
    event_to_game_id: Mapping[str, str] | None = None,
) -> KalshiSeriesDiscoveryV1:
    captured = capture_kalshi_series_metadata(
        series_spec,
        exchange=exchange,
        max_pages=max_pages,
    )
    return classify_captured_kalshi_series_for_game(
        binding,
        captured,
        event_to_game_id=event_to_game_id,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or type(value) in {str, int, bool}:
        return value
    raise X13UniverseError(
        f"universe payload contains unsupported {type(value).__name__}"
    )


def _contract_payload(contract: UniverseContractV1) -> dict[str, object]:
    payload = _jsonable(contract)
    if type(payload) is not dict:  # pragma: no cover - dataclass invariant
        raise X13UniverseError("contract payload is malformed")
    payload["outcome_coverage"] = {
        outcome: coverage for outcome, coverage in contract.outcome_coverage
    }
    return payload


def _game_payload(game: UniverseGameV1) -> dict[str, object]:
    return {
        "capture_plan": _jsonable(game.capture_plan),
        "contracts": [
            _contract_payload(contract) for contract in game.contracts
        ],
        "cross_game_inventory": _jsonable(game.cross_game_inventory),
        "excluded_contracts": _jsonable(game.excluded_contracts),
        "game_id": game.game_id,
        "inventory": {
            "primitive_contract_ids": [
                target.contract_id for target in game.primitive_targets
            ],
            "same_game_composite_contract_ids": [
                target.contract_id
                for target in game.same_game_composite_targets
            ],
            "cross_game_contract_ids": [
                f"kalshi:{item.venue_market_id}"
                for item in game.cross_game_inventory
            ],
            "excluded_contract_ids": [
                item.contract_id for item in game.excluded_contracts
            ],
            "unknown_contract_ids": list(game.unknown_contract_ids),
        },
        "kalshi_series_tickers": list(game.kalshi_series_tickers),
        "metadata_manifest_sha256s": list(
            game.metadata_manifest_sha256s
        ),
        "polymarket_event": {
            "id": game.polymarket_event_id,
            "slug": game.polymarket_event_slug,
            "title": game.polymarket_event_title,
        },
        "polymarket_token_bindings": _jsonable(
            game.polymarket_token_bindings
        ),
        "source_window": _jsonable(game.source_window),
    }


def _batch_payload(
    batch: X13UniverseBatchV1,
    *,
    include_universe_id: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "capture_plans": _jsonable(batch.capture_plans),
        "game_ids": list(batch.game_ids),
        "games": [_game_payload(game) for game in batch.games],
        "kalshi_series_catalog_version": (
            KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.version
        ),
        "schema_version": batch.schema_version,
        "series_registry_proof": _jsonable(batch.series_registry_proof),
    }
    if include_universe_id:
        payload["universe_id"] = batch.universe_id
    return payload


def build_x13_universe(
    *,
    polymarket_inventories: Sequence[PolymarketInventoryV1],
    kalshi_discoveries: Sequence[KalshiSeriesDiscoveryV1],
    source_windows: Mapping[str, SourceTimeWindowV1],
    series_registry_proof: KalshiSeriesRegistryProofV1,
) -> X13UniverseBatchV1:
    """Assemble exact 20-game inventories into capture-ready plans."""

    if not isinstance(series_registry_proof, KalshiSeriesRegistryProofV1):
        raise X13UniverseError("series registry proof is missing")
    expected_series = tuple(
        spec.series_ticker
        for spec in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
    )
    if series_registry_proof.series_tickers != expected_series:
        raise X13UniverseError("series registry proof does not match catalog")
    if set(source_windows) != set(X13_GAME_IDS) or any(
        not isinstance(window, SourceTimeWindowV1)
        for window in source_windows.values()
    ):
        raise X13UniverseError("source windows must exactly cover 20 games")

    poly_by_game: dict[str, PolymarketInventoryV1] = {}
    for inventory in polymarket_inventories:
        if (
            not isinstance(inventory, PolymarketInventoryV1)
            or inventory.game_id not in X13_GAME_IDS
            or inventory.game_id in poly_by_game
        ):
            raise X13UniverseError(
                "Polymarket inventory must uniquely cover frozen games"
            )
        poly_by_game[inventory.game_id] = inventory
    if tuple(sorted(poly_by_game)) != X13_GAME_IDS:
        raise X13UniverseError(
            "Polymarket inventory does not exactly cover 20 games"
        )

    kalshi_by_game_series: dict[
        tuple[str, str], KalshiSeriesDiscoveryV1
    ] = {}
    for discovery in kalshi_discoveries:
        if (
            not isinstance(discovery, KalshiSeriesDiscoveryV1)
            or discovery.game_id not in X13_GAME_IDS
            or discovery.series_ticker not in expected_series
        ):
            raise X13UniverseError("Kalshi discovery identity is unknown")
        key = (discovery.game_id, discovery.series_ticker)
        if key in kalshi_by_game_series:
            raise X13UniverseError("Kalshi series discovery is duplicated")
        if (
            discovery.page_count != len(discovery.manifest_sha256s)
            or discovery.page_count < 1
            or discovery.terminal_proof != "explicit_empty_cursor"
        ):
            raise X13UniverseError(
                "Kalshi discovery lacks terminal pagination proof"
            )
        for manifest in discovery.manifest_sha256s:
            _require_sha256(manifest, "metadata manifest")
        kalshi_by_game_series[key] = discovery
    expected_keys = {
        (game_id, series_ticker)
        for game_id in X13_GAME_IDS
        for series_ticker in expected_series
    }
    if set(kalshi_by_game_series) != expected_keys:
        raise X13UniverseError(
            "Kalshi series coverage must be exact for all 20 games"
        )

    games: list[UniverseGameV1] = []
    for game_id in X13_GAME_IDS:
        poly = poly_by_game[game_id]
        discoveries = tuple(
            kalshi_by_game_series[(game_id, series_ticker)]
            for series_ticker in expected_series
        )
        targets = (
            *poly.capture_targets,
            *(
                target
                for discovery in discoveries
                for target in discovery.capture_targets
            ),
        )
        target_ids = tuple(target.contract_id for target in targets)
        if len(set(target_ids)) != len(target_ids):
            raise X13UniverseError("capture target identity is duplicated")
        contracts = (
            *poly.contracts,
            *(
                contract
                for discovery in discoveries
                for contract in discovery.contracts
            ),
        )
        logical_ids = tuple(
            contract.logical_market_id for contract in contracts
        )
        if len(set(logical_ids)) != len(logical_ids):
            raise X13UniverseError(
                "logical market identity is duplicated"
            )
        expected_winners = {
            f"polymarket:{game_id}:winner",
            f"kalshi:{game_id}:winner",
        }
        if {
            contract.logical_market_id
            for contract in contracts
            if contract.family == "moneyline"
        } != expected_winners:
            raise X13UniverseError(
                "game does not have exact two-venue winner contracts"
            )
        primitive = tuple(
            sorted(
                (
                    target
                    for target in targets
                    if target.kind == "primitive"
                ),
                key=lambda target: target.contract_id,
            )
        )
        same_game = tuple(
            sorted(
                (
                    target
                    for target in targets
                    if target.kind == "same_game_composite"
                ),
                key=lambda target: target.contract_id,
            )
        )
        cross = tuple(
            sorted(
                (
                    item
                    for discovery in discoveries
                    for item in discovery.cross_game_inventory
                ),
                key=lambda item: item.venue_market_id,
            )
        )
        excluded = tuple(
            sorted(
                (
                    item
                    for discovery in discoveries
                    for item in discovery.excluded_contracts
                ),
                key=lambda item: item.contract_id,
            )
        )
        manifests = tuple(
            sorted(
                {
                    poly.manifest_sha256,
                    *(
                        manifest
                        for discovery in discoveries
                        for manifest in discovery.manifest_sha256s
                    ),
                }
            )
        )
        games.append(
            UniverseGameV1(
                game_id=game_id,
                source_window=source_windows[game_id],
                polymarket_event_id=poly.event_id,
                polymarket_event_slug=poly.event_slug,
                polymarket_event_title=poly.event_title,
                primitive_targets=primitive,
                same_game_composite_targets=same_game,
                cross_game_inventory=cross,
                excluded_contracts=excluded,
                unknown_contract_ids=(),
                contracts=tuple(
                    sorted(
                        contracts,
                        key=lambda contract: contract.logical_market_id,
                    )
                ),
                polymarket_token_bindings=poly.token_bindings,
                kalshi_series_tickers=expected_series,
                metadata_manifest_sha256s=manifests,
            )
        )
    games_tuple = tuple(games)
    plans = tuple(game.capture_plan for game in games_tuple)
    if tuple(plan.game_id for plan in plans) != X13_GAME_IDS:
        raise X13UniverseError("capture plans do not preserve frozen order")
    return X13UniverseBatchV1(
        games=games_tuple,
        capture_plans=plans,
        series_registry_proof=series_registry_proof,
    )


def publish_x13_universe(
    batch: X13UniverseBatchV1,
    output_path: str | Path,
) -> Path:
    """Atomically publish one immutable, self-contained universe plan."""

    if not isinstance(batch, X13UniverseBatchV1):
        raise X13UniverseError("batch must be X13UniverseBatchV1")
    path = Path(output_path)
    if not path.is_absolute():
        raise X13UniverseError("universe output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise X13UniverseError("universe artifact already exists")
    body = (
        json.dumps(
            batch.canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise X13UniverseError(
                "universe artifact already exists"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1",
    "POLYMARKET_SPORTS_MARKET_TYPE_FAMILIES_V1",
    "CapturedKalshiSeriesV1",
    "KalshiSeriesCatalogV1",
    "KalshiSeriesDiscoveryV1",
    "KalshiSeriesRegistryProofV1",
    "KalshiSeriesSpecV1",
    "MetadataExchangeResult",
    "PolymarketInventoryV1",
    "PolymarketTokenBindingV1",
    "SourceTimeWindowV1",
    "X13UniverseError",
    "build_kalshi_event_dependency_registry",
    "build_x13_universe",
    "capture_kalshi_series_metadata",
    "classify_captured_kalshi_series_for_game",
    "classify_cross_game_mve_market",
    "classify_polymarket_event",
    "derive_kalshi_event_ticker",
    "discover_kalshi_series_for_game",
    "exchange_and_preserve_metadata",
    "publish_x13_universe",
    "validate_kalshi_series_registry",
]
