"""Development-only NFL moneyline market and reaction materialization.

The module is deliberately separate from the frozen 20-game X-13 association
types.  Expansion games use the same actual-trade normalizers, but their
contract inventory, source intervals, and registered seven-horizon reactions
are materialized here without widening the old X-13 whitelist.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from prediction_market.sports.nfl_factor_lab_factor_observations import (
    FactorObservationBuildV2,
    build_factor_observations_v2,
)
from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
    SubjectRef,
    build_layer_g_interval,
    deduplicate_observations,
    normalize_contract,
    normalize_kalshi_trade,
    normalize_polymarket_trade,
)
from prediction_market.sports.nfl_x13_spec import kalshi_team_code


HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60)
DELAY_SCENARIOS_SECONDS = (0, 1, 2, 3, 5, 10)
COMPUTE_WORKERS = 4
MAX_COMPUTE_WORKERS = 8
DISCOVERY_BOOTSTRAP_SAMPLES = 10_000
DISCOVERY_BOOTSTRAP_SEED = 20260725
BUILDER_VERSION = "nfl-expansion-development-market-v3"
CLAIM_BOUNDARY = (
    "PRELIMINARY historical source-time association only; no causality, "
    "alpha, live latency, execution, or tradeability claim."
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TEAM_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ARI": ("ARI", "Arizona", "Arizona Cardinals", "Cardinals"),
    "ATL": ("ATL", "Atlanta", "Atlanta Falcons", "Falcons"),
    "BAL": ("BAL", "Baltimore", "Baltimore Ravens", "Ravens"),
    "BUF": ("BUF", "Buffalo", "Buffalo Bills", "Bills"),
    "CAR": ("CAR", "Carolina", "Carolina Panthers", "Panthers"),
    "CHI": ("CHI", "Chicago", "Chicago Bears", "Bears"),
    "CIN": ("CIN", "Cincinnati", "Cincinnati Bengals", "Bengals"),
    "CLE": ("CLE", "Cleveland", "Cleveland Browns", "Browns"),
    "DAL": ("DAL", "Dallas", "Dallas Cowboys", "Cowboys"),
    "DEN": ("DEN", "Denver", "Denver Broncos", "Broncos"),
    "DET": ("DET", "Detroit", "Detroit Lions", "Lions"),
    "GB": ("GB", "Green Bay", "Green Bay Packers", "Packers"),
    "HOU": ("HOU", "Houston", "Houston Texans", "Texans"),
    "IND": ("IND", "Indianapolis", "Indianapolis Colts", "Colts"),
    "JAX": (
        "JAC",
        "JAX",
        "Jacksonville",
        "Jacksonville Jaguars",
        "Jaguars",
    ),
    "KC": ("KC", "Kansas City", "Kansas City Chiefs", "Chiefs"),
    "LA": ("LA", "Los Angeles Rams", "Rams"),
    "LAC": ("LAC", "Los Angeles Chargers", "Chargers"),
    "LV": ("LV", "Las Vegas", "Las Vegas Raiders", "Raiders"),
    "MIA": ("MIA", "Miami", "Miami Dolphins", "Dolphins"),
    "MIN": ("MIN", "Minnesota", "Minnesota Vikings", "Vikings"),
    "NE": ("NE", "New England", "New England Patriots", "Patriots"),
    "NO": ("NO", "New Orleans", "New Orleans Saints", "Saints"),
    "NYG": ("NYG", "New York Giants", "Giants"),
    "NYJ": ("NYJ", "New York Jets", "Jets"),
    "PHI": ("PHI", "Philadelphia", "Philadelphia Eagles", "Eagles"),
    "PIT": ("PIT", "Pittsburgh", "Pittsburgh Steelers", "Steelers"),
    "SEA": ("SEA", "Seattle", "Seattle Seahawks", "Seahawks"),
    "SF": ("SF", "San Francisco", "San Francisco 49ers", "49ers"),
    "TB": (
        "TB",
        "Tampa Bay",
        "Tampa Bay Buccaneers",
        "Buccaneers",
        "Bucs",
    ),
    "TEN": ("TEN", "Tennessee", "Tennessee Titans", "Titans"),
    "WAS": (
        "WAS",
        "Washington",
        "Washington Commanders",
        "Commanders",
    ),
}


class DevelopmentMarketError(RuntimeError):
    """An immutable input or a development-only analysis contract failed."""


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, pd.Timestamp)):
        return _utc_text(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not canonical JSON")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _semantic_sha(value: object) -> str:
    return _sha256(_canonical_bytes(value, newline=False))


def _utc(value: object, *, field: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DevelopmentMarketError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_utc(value: object, *, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise DevelopmentMarketError(f"{field} must be RFC3339 UTC")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise DevelopmentMarketError(f"{field} must be RFC3339 UTC") from error
    return _utc(result, field=field)


def _utc_text(value: object) -> str:
    return (
        _utc(value, field="timestamp")
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise DevelopmentMarketError(f"{field} must be numeric")
    if isinstance(value, Decimal):
        result = value
    elif type(value) in {str, int}:
        try:
            result = Decimal(value)
        except Exception as error:
            raise DevelopmentMarketError(f"{field} must be decimal") from error
    else:
        raise DevelopmentMarketError(f"{field} must preserve exact decimal")
    if not result.is_finite():
        raise DevelopmentMarketError(f"{field} must be finite")
    return result


def _missing(value: object) -> bool:
    return value is None or (
        not isinstance(value, (list, tuple, dict))
        and bool(pd.isna(value))
    )


def _strict_json(raw: bytes, *, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentMarketError(
                    f"{label} contains duplicate JSON key {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise DevelopmentMarketError(
            f"{label} contains non-JSON number {value}"
        )

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_float=Decimal,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentMarketError(f"{label} is invalid JSON") from error


def _json_string_tuple(
    value: object, *, field: str, length: int
) -> tuple[str, ...]:
    if type(value) is not str:
        raise DevelopmentMarketError(f"{field} must be a JSON string array")
    parsed = _strict_json(value.encode("utf-8"), label=field)
    if (
        type(parsed) is not list
        or len(parsed) != length
        or any(type(item) is not str or not item for item in parsed)
        or len(set(parsed)) != length
    ):
        raise DevelopmentMarketError(f"{field} has invalid string values")
    return tuple(parsed)


def _alias_key(value: str) -> str:
    return "".join(
        character for character in value.casefold()
        if character.isalnum()
    )


def _canonical_team_outcome(
    value: object, teams: tuple[str, str]
) -> str:
    if type(value) is not str:
        raise DevelopmentMarketError("moneyline outcome is not text")
    key = _alias_key(value)
    matches = [
        team
        for team in teams
        if key in {
            _alias_key(alias)
            for alias in _TEAM_ALIASES.get(team, (team,))
        }
    ]
    if len(matches) != 1:
        raise DevelopmentMarketError(
            f"moneyline outcome {value!r} is not uniquely team-oriented"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class DevelopmentAccess:
    """Path-open guard derived only from the verified unlock artifact."""

    development_game_ids: frozenset[str]
    final_holdout_game_ids: frozenset[str]

    def __post_init__(self) -> None:
        if (
            len(self.development_game_ids) != 153
            or len(self.final_holdout_game_ids) != 81
            or self.development_game_ids & self.final_holdout_game_ids
        ):
            raise DevelopmentMarketError("unlock split is not exact 153/81")

    def require(self, game_id: str) -> None:
        if game_id in self.final_holdout_game_ids:
            raise DevelopmentMarketError(
                f"final holdout raw access prohibited: {game_id}"
            )
        if game_id not in self.development_game_ids:
            raise DevelopmentMarketError(
                f"game is outside development unlock: {game_id}"
            )

    def read_bytes(
        self,
        game_id: str,
        path: Path,
        *,
        reader: Callable[[Path], bytes] = Path.read_bytes,
    ) -> bytes:
        # This check intentionally precedes both resolve() and reader().
        self.require(game_id)
        return reader(path)


@dataclass(frozen=True, slots=True)
class ExpansionMoneylineBinding:
    game_id: str
    away_team: str
    home_team: str
    polymarket_market_id: str
    polymarket_condition_id: str
    polymarket_tokens: tuple[tuple[str, str, str], ...]
    kalshi_event_ticker: str
    kalshi_team_tickers: tuple[tuple[str, str], ...]
    rule_sha256: str

    @property
    def teams(self) -> tuple[str, str]:
        return (self.away_team, self.home_team)


@dataclass(frozen=True, slots=True)
class VerifiedMarketDocument:
    """One byte-verified capture response and its direct lineage."""

    source: str
    source_url: str
    source_request: Mapping[str, object]
    object_bytes: bytes
    object_sha256: str
    manifest_sha256: str
    request_sha256: str

    def __post_init__(self) -> None:
        if self.source not in {"polymarket", "kalshi"}:
            raise DevelopmentMarketError("raw document source is invalid")
        if (
            type(self.source_url) is not str
            or not self.source_url.startswith("https://")
            or type(self.object_bytes) is not bytes
            or _sha256(self.object_bytes) != self.object_sha256
            or any(
                _SHA256_RE.fullmatch(value) is None
                for value in (
                    self.object_sha256,
                    self.manifest_sha256,
                    self.request_sha256,
                )
            )
        ):
            raise DevelopmentMarketError(
                "raw document bytes or lineage is invalid"
            )


def build_moneyline_binding(
    *,
    identity: Mapping[str, object],
    documents: Sequence[VerifiedMarketDocument],
) -> ExpansionMoneylineBinding:
    """Prove Gamma outcome and Kalshi ticker orientation for one game."""

    required = {
        "game_id",
        "away_team",
        "home_team",
        "polymarket_event_id",
        "polymarket_event_slug",
        "polymarket_condition_id",
        "polymarket_market_id",
        "kalshi_event_ticker",
        "kalshi_team_tickers",
        "start_ts",
        "end_ts",
    }
    if set(identity) != required:
        raise DevelopmentMarketError("capture identity fields differ")
    game_id = identity["game_id"]
    away_team = identity["away_team"]
    home_team = identity["home_team"]
    if (
        type(game_id) is not str
        or type(away_team) is not str
        or type(home_team) is not str
        or away_team == home_team
        or tuple(game_id.split("_")[-2:]) != (away_team, home_team)
    ):
        raise DevelopmentMarketError("capture game/team identity differs")
    teams = (away_team, home_team)
    gamma_documents = [
        document
        for document in documents
        if document.source == "polymarket"
        and "gamma-api.polymarket.com/events/slug/" in document.source_url
    ]
    if len(gamma_documents) != 1:
        raise DevelopmentMarketError("Gamma event document is not unique")
    gamma = _strict_json(
        gamma_documents[0].object_bytes,
        label=f"{game_id} Gamma event",
    )
    if (
        type(gamma) is not dict
        or gamma.get("id") != identity["polymarket_event_id"]
        or gamma.get("slug") != identity["polymarket_event_slug"]
        or type(gamma.get("markets")) is not list
    ):
        raise DevelopmentMarketError("Gamma event identity differs")
    moneylines = [
        market
        for market in gamma["markets"]
        if type(market) is dict
        and market.get("id") == identity["polymarket_market_id"]
        and market.get("conditionId")
        == identity["polymarket_condition_id"]
        and market.get("sportsMarketType") == "moneyline"
    ]
    if len(moneylines) != 1:
        raise DevelopmentMarketError("Gamma moneyline is not unique")
    moneyline = moneylines[0]
    native_outcomes = _json_string_tuple(
        moneyline.get("outcomes"), field="Gamma outcomes", length=2
    )
    token_ids = _json_string_tuple(
        moneyline.get("clobTokenIds"),
        field="Gamma clobTokenIds",
        length=2,
    )
    oriented_tokens = tuple(
        sorted(
            (
                native,
                _canonical_team_outcome(native, teams),
                token,
            )
            for native, token in zip(
                native_outcomes, token_ids, strict=True
            )
        )
    )
    if {team for _native, team, _token in oriented_tokens} != set(teams):
        raise DevelopmentMarketError(
            "Gamma moneyline does not cover both teams"
        )
    event_ticker = identity["kalshi_event_ticker"]
    raw_tickers = identity["kalshi_team_tickers"]
    if (
        type(event_ticker) is not str
        or type(raw_tickers) is not list
        or len(raw_tickers) != 2
        or any(type(ticker) is not str for ticker in raw_tickers)
    ):
        raise DevelopmentMarketError("Kalshi identity is malformed")
    expected_by_team = {
        team: f"{event_ticker}-{kalshi_team_code(team)}"
        for team in teams
    }
    if set(raw_tickers) != set(expected_by_team.values()):
        raise DevelopmentMarketError(
            "Kalshi team ticker suffix orientation differs"
        )
    market_documents = [
        document
        for document in documents
        if document.source == "kalshi"
        and document.source_url.endswith("/historical/markets")
    ]
    observed_tickers: set[str] = set()
    for document in market_documents:
        payload = _strict_json(
            document.object_bytes,
            label=f"{game_id} Kalshi markets",
        )
        if (
            type(payload) is not dict
            or set(payload) != {"markets", "cursor"}
            or type(payload["markets"]) is not list
        ):
            raise DevelopmentMarketError(
                "Kalshi markets response is malformed"
            )
        for market in payload["markets"]:
            if (
                type(market) is not dict
                or market.get("event_ticker") != event_ticker
                or type(market.get("ticker")) is not str
            ):
                raise DevelopmentMarketError(
                    "Kalshi market escaped event identity"
                )
            observed_tickers.add(market["ticker"])
    if observed_tickers != set(raw_tickers):
        raise DevelopmentMarketError(
            "Kalshi verified market inventory differs"
        )
    rule_sha = _semantic_sha(
        {
            "schema": "nfl_expansion_moneyline_binding_v1",
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "polymarket_event_id": identity["polymarket_event_id"],
            "polymarket_market_id": identity["polymarket_market_id"],
            "polymarket_condition_id": identity[
                "polymarket_condition_id"
            ],
            "polymarket_tokens": oriented_tokens,
            "kalshi_event_ticker": event_ticker,
            "kalshi_team_tickers": sorted(expected_by_team.items()),
            "metadata_manifest_sha256s": sorted(
                {
                    gamma_documents[0].manifest_sha256,
                    *(
                        document.manifest_sha256
                        for document in market_documents
                    ),
                }
            ),
        }
    )
    return ExpansionMoneylineBinding(
        game_id=game_id,
        away_team=away_team,
        home_team=home_team,
        polymarket_market_id=str(identity["polymarket_market_id"]),
        polymarket_condition_id=str(
            identity["polymarket_condition_id"]
        ),
        polymarket_tokens=oriented_tokens,
        kalshi_event_ticker=event_ticker,
        kalshi_team_tickers=tuple(sorted(expected_by_team.items())),
        rule_sha256=rule_sha,
    )


def normalize_verified_market_documents(
    *,
    binding: ExpansionMoneylineBinding,
    documents: Sequence[VerifiedMarketDocument],
) -> tuple[
    tuple[MarketObservation, ...],
    Mapping[str, tuple[str, ...]],
]:
    """Parse verified raw bytes exactly, normalize, dedupe, and bind lineage."""

    token_map = {
        token: (native, team)
        for native, team, token in binding.polymarket_tokens
    }
    ticker_to_team = {
        ticker: team for team, ticker in binding.kalshi_team_tickers
    }
    observed_ticker_pages: set[str] = set()
    observations: list[MarketObservation] = []
    lineage: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        if document.source == "polymarket" and document.source_url.endswith(
            "data-api.polymarket.com/trades"
        ):
            payload = _strict_json(
                document.object_bytes,
                label=f"{binding.game_id} Polymarket trades",
            )
            if type(payload) is not list:
                raise DevelopmentMarketError(
                    "Polymarket trades response is not a list"
                )
            for raw in payload:
                if type(raw) is not dict:
                    raise DevelopmentMarketError(
                        "Polymarket trade row is malformed"
                    )
                token = raw.get("asset")
                oriented = (
                    token_map.get(token) if type(token) is str else None
                )
                if (
                    oriented is None
                    or raw.get("conditionId")
                    != binding.polymarket_condition_id
                    or raw.get("outcome") != oriented[0]
                ):
                    raise DevelopmentMarketError(
                        "Polymarket trade orientation differs"
                    )
                observation = normalize_polymarket_trade(
                    raw,
                    raw_market_id=binding.polymarket_condition_id,
                    logical_market_id=(
                        f"polymarket:{binding.game_id}:winner"
                    ),
                    outcome=oriented[1],
                )
                observations.append(observation)
                lineage[observation.observation_id].update(
                    (
                        document.object_sha256,
                        document.manifest_sha256,
                        document.request_sha256,
                    )
                )
        elif document.source == "kalshi" and document.source_url.endswith(
            "/historical/trades"
        ):
            params = document.source_request.get("params")
            ticker = (
                params.get("ticker")
                if isinstance(params, Mapping)
                else None
            )
            if type(ticker) is not str or ticker not in ticker_to_team:
                raise DevelopmentMarketError(
                    "Kalshi trade document ticker is unresolved"
                )
            observed_ticker_pages.add(ticker)
            payload = _strict_json(
                document.object_bytes,
                label=f"{binding.game_id} Kalshi trades",
            )
            if (
                type(payload) is not dict
                or set(payload) != {"trades", "cursor"}
                or type(payload["trades"]) is not list
            ):
                raise DevelopmentMarketError(
                    "Kalshi trades response is malformed"
                )
            for raw in payload["trades"]:
                if type(raw) is not dict or raw.get("ticker") != ticker:
                    raise DevelopmentMarketError(
                        "Kalshi trade escaped ticker identity"
                    )
                observation = normalize_kalshi_trade(
                    raw,
                    ticker=ticker,
                    outcome=ticker_to_team[ticker],
                    logical_market_id=f"kalshi:{binding.game_id}:winner",
                )
                observations.append(observation)
                lineage[observation.observation_id].update(
                    (
                        document.object_sha256,
                        document.manifest_sha256,
                        document.request_sha256,
                    )
                )
    if observed_ticker_pages != set(ticker_to_team):
        raise DevelopmentMarketError(
            "Kalshi trade cursor chains do not cover both team tickers"
        )
    deduped = deduplicate_observations(observations)
    if set(lineage) != {
        observation.observation_id for observation in deduped
    }:
        raise DevelopmentMarketError(
            "normalized trade lineage does not reconcile"
        )
    return (
        deduped,
        {
            key: tuple(sorted(values))
            for key, values in sorted(lineage.items())
        },
    )


def _moneyline_contract_inventory(
    binding: ExpansionMoneylineBinding,
) -> pd.DataFrame:
    outcomes = tuple(sorted(binding.teams))
    rows: list[dict[str, object]] = []
    venue_inputs = (
        (
            "polymarket",
            binding.polymarket_market_id,
            f"polymarket:{binding.game_id}:winner",
            {
                team: token
                for _native, team, token in binding.polymarket_tokens
            },
        ),
        (
            "kalshi",
            binding.kalshi_event_ticker,
            f"kalshi:{binding.game_id}:winner",
            dict(binding.kalshi_team_tickers),
        ),
    )
    for venue, venue_market_id, logical_market_id, raw_by_team in venue_inputs:
        contract = normalize_contract(
            venue=venue,
            venue_market_id=venue_market_id,
            family="moneyline",
            period="full_game",
            subject=SubjectRef(
                kind="game",
                canonical_id=binding.game_id,
                name=binding.game_id,
            ),
            measure="winner",
            comparator="eq",
            line=None,
            outcomes=outcomes,
            rule_hash=binding.rule_sha256,
            dependency_game_ids=(binding.game_id,),
            kind="primitive",
        )
        for outcome in outcomes:
            rows.append(
                {
                    "game_id": binding.game_id,
                    "logical_market_id": logical_market_id,
                    "outcome": outcome,
                    "venue": venue,
                    "contract_id": contract.canonical_hash,
                    "venue_market_id": venue_market_id,
                    "raw_contract_id": raw_by_team[outcome],
                    "family": "moneyline",
                    "period": "full_game",
                    "subject": binding.game_id,
                    "measure": "winner",
                    "kind": "primitive",
                    "analysis_eligible": contract.analysis_eligible,
                    "rule_sha256": contract.canonical_hash,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["venue", "logical_market_id", "outcome"], kind="mergesort"
    ).reset_index(drop=True)


def normalize_actual_moneyline_trades(
    *,
    binding: ExpansionMoneylineBinding,
    polymarket_rows: Sequence[Mapping[str, object]],
    kalshi_rows_by_ticker: Mapping[
        str, Sequence[Mapping[str, object]]
    ],
) -> tuple[MarketObservation, ...]:
    """Normalize only native actual trades; complements are never fabricated."""

    token_map = {
        token: (native, team)
        for native, team, token in binding.polymarket_tokens
    }
    observations: list[MarketObservation] = []
    for raw in polymarket_rows:
        token = raw.get("asset")
        oriented = token_map.get(token) if type(token) is str else None
        if (
            oriented is None
            or raw.get("conditionId") != binding.polymarket_condition_id
            or raw.get("outcome") != oriented[0]
        ):
            raise DevelopmentMarketError(
                "Polymarket token/outcome orientation is unresolved"
            )
        observations.append(
            normalize_polymarket_trade(
                raw,
                raw_market_id=binding.polymarket_condition_id,
                logical_market_id=(
                    f"polymarket:{binding.game_id}:winner"
                ),
                outcome=oriented[1],
            )
        )
    ticker_to_team = {
        ticker: team for team, ticker in binding.kalshi_team_tickers
    }
    if set(kalshi_rows_by_ticker) != set(ticker_to_team):
        raise DevelopmentMarketError("Kalshi ticker inventory is incomplete")
    for ticker in sorted(kalshi_rows_by_ticker):
        for raw in kalshi_rows_by_ticker[ticker]:
            if raw.get("ticker") != ticker:
                raise DevelopmentMarketError(
                    "Kalshi trade escaped ticker identity"
                )
            observations.append(
                normalize_kalshi_trade(
                    raw,
                    ticker=ticker,
                    outcome=ticker_to_team[ticker],
                    logical_market_id=f"kalshi:{binding.game_id}:winner",
                )
            )
    return deduplicate_observations(observations)


def actual_market_observation_frame(
    game_id: str,
    observations: Sequence[MarketObservation],
) -> pd.DataFrame:
    rows = []
    for row in observations:
        rows.append(
            {
                "game_id": game_id,
                "observation_id": row.observation_id,
                "venue": row.venue,
                "raw_market_id": row.raw_market_id,
                "logical_market_id": row.logical_market_id,
                "outcome": row.outcome,
                "kind": row.kind,
                "source_start_utc": _utc_text(row.source_interval.start),
                "source_end_utc": _utc_text(row.source_interval.end),
                "source_end_inclusive": row.source_interval.end_inclusive,
                "native_source_time_utc": (
                    None
                    if row.native_source_time is None
                    else _utc_text(row.native_source_time)
                ),
                "price": row.price,
                "size": row.size,
                "provenance": row.provenance,
                "native_id": row.native_id,
                "is_block_trade": row.is_block_trade,
                "primary_path_eligible": row.primary_path_eligible,
                "bid": None,
                "ask": None,
                "has_executable_bbo": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["venue", "logical_market_id", "outcome", "source_start_utc",
         "observation_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_layer_m_intervals(
    market_observations: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "game_id",
        "observation_id",
        "venue",
        "logical_market_id",
        "outcome",
        "source_start_utc",
        "source_end_utc",
        "source_end_inclusive",
        "provenance",
        "primary_path_eligible",
    ]
    if "market_source_hashes" in market_observations:
        columns.append("market_source_hashes")
    result = market_observations[columns].copy()
    result["layer"] = "M"
    result["interval_semantics"] = "ACTUAL_TRADE_SECOND_HALF_OPEN"
    return result


def build_layer_g_timeline(
    sports_feature_view: pd.DataFrame,
    *,
    delay_scenarios: Sequence[int] = DELAY_SCENARIOS_SECONDS,
) -> pd.DataFrame:
    required = {
        "game_id",
        "play_id",
        "episode_id",
        "order_sequence",
        "source_time_utc",
    }
    missing = sorted(required.difference(sports_feature_view.columns))
    if missing:
        raise DevelopmentMarketError(
            "sports feature view missing Layer G columns: "
            + ", ".join(missing)
        )
    if tuple(delay_scenarios) != tuple(sorted(set(delay_scenarios))):
        raise DevelopmentMarketError(
            "delay scenarios must be sorted and unique"
        )
    ordered = sports_feature_view.sort_values(
        ["game_id", "episode_id", "order_sequence", "play_id"],
        kind="mergesort",
    )
    episodes: list[dict[str, object]] = []
    for (game_id, episode_id), frame in ordered.groupby(
        ["game_id", "episode_id"], sort=True, dropna=False
    ):
        timed = frame[frame["source_time_utc"].notna()].copy()
        if timed.empty:
            source_time = None
            source_play_id = None
            order_sequence = int(frame["order_sequence"].max())
        else:
            timed["_source"] = pd.to_datetime(
                timed["source_time_utc"],
                utc=True,
                errors="raise",
                format="mixed",
            )
            final = timed.sort_values(
                ["_source", "order_sequence", "play_id"],
                kind="mergesort",
            ).iloc[-1]
            source_time = final["_source"].to_pydatetime()
            source_play_id = str(final["play_id"])
            order_sequence = int(final["order_sequence"])
        episodes.append(
            {
                "game_id": str(game_id),
                "episode_id": str(episode_id),
                "source_play_id": source_play_id,
                "order_sequence": order_sequence,
                "_source_time": source_time,
                "sports_source_hashes": tuple(
                    sorted(
                        {
                            digest
                            for value in frame.get(
                                "sports_source_hashes",
                                pd.Series(dtype=object),
                            )
                            if isinstance(value, (list, tuple))
                            for digest in value
                            if type(digest) is str
                            and _SHA256_RE.fullmatch(digest)
                        }
                    )
                ),
            }
        )
    by_game: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        by_game[str(row["game_id"])].append(row)
    rows: list[dict[str, object]] = []
    for game_id in sorted(by_game):
        timed = sorted(
            by_game[game_id],
            key=lambda row: (
                row["_source_time"] is None,
                row["_source_time"] or datetime.max.replace(
                    tzinfo=timezone.utc
                ),
                int(row["order_sequence"]),
                str(row["episode_id"]),
            ),
        )
        for index, episode in enumerate(timed):
            source = episode["_source_time"]
            next_source = next(
                (
                    child["_source_time"]
                    for child in timed[index + 1 :]
                    if child["_source_time"] is not None
                ),
                None,
            )
            for delay in delay_scenarios:
                interval = (
                    None
                    if source is None
                    else build_layer_g_interval(
                        source, delay_seconds=Decimal(delay)
                    )
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "episode_id": episode["episode_id"],
                        "source_play_id": episode["source_play_id"],
                        "order_sequence": episode["order_sequence"],
                        "delay_seconds": int(delay),
                        "source_time_utc": (
                            None if source is None else _utc_text(source)
                        ),
                        "interval_start_utc": (
                            None
                            if interval is None
                            else _utc_text(interval.start)
                        ),
                        "interval_end_utc": (
                            None
                            if interval is None
                            else _utc_text(interval.end)
                        ),
                        "end_inclusive": (
                            None
                            if interval is None
                            else interval.end_inclusive
                        ),
                        "next_episode_source_time_utc": (
                            None
                            if next_source is None
                            else _utc_text(next_source)
                        ),
                        "layer": "G",
                        "provenance": (
                            "SPORTS_FEATURE_SOURCE_TIME_PLUS_REGISTERED_DELAY"
                            if source is not None
                            else "SOURCE_TIME_MISSING"
                        ),
                        "sports_source_hashes": episode[
                            "sports_source_hashes"
                        ],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["game_id", "order_sequence", "episode_id", "delay_seconds"],
        kind="mergesort",
    ).reset_index(drop=True)


def _unique_signed_maximum_excursion(
    prices: Sequence[Decimal], pre_price: Decimal | None
) -> Decimal | None:
    if pre_price is None or not prices:
        return None
    deltas = tuple(price - pre_price for price in prices)
    magnitude = max(abs(delta) for delta in deltas)
    candidates = {delta for delta in deltas if abs(delta) == magnitude}
    return next(iter(candidates)) if len(candidates) == 1 else None


def build_reaction_paths(
    *,
    layer_g: pd.DataFrame,
    contracts: pd.DataFrame,
    observations: Sequence[MarketObservation],
    market_lineage: Mapping[str, Sequence[str]],
    horizons: Sequence[int] = HORIZONS_SECONDS,
) -> pd.DataFrame:
    """Build explicit cumulative actual-trade paths, including every gap cell."""

    if tuple(horizons) != HORIZONS_SECONDS:
        raise DevelopmentMarketError("reaction horizon grid is not registered")
    series: dict[
        tuple[str, str, str], list[MarketObservation]
    ] = defaultdict(list)
    for observation in observations:
        if (
            observation.kind == "trade"
            and observation.provenance == "observed"
            and observation.primary_path_eligible
            and observation.price is not None
        ):
            series[
                (
                    observation.venue,
                    str(observation.logical_market_id),
                    observation.outcome,
                )
            ].append(observation)
    for values in series.values():
        values.sort(
            key=lambda row: (
                row.source_interval.start,
                row.observation_id,
            )
        )
    rows: list[dict[str, object]] = []
    contract_rows = contracts.sort_values(
        ["venue", "logical_market_id", "outcome"], kind="mergesort"
    ).to_dict("records")
    for event in layer_g.to_dict("records"):
        event_start = (
            None
            if _missing(event["interval_start_utc"])
            else _parse_utc(
                event["interval_start_utc"], field="event interval start"
            )
        )
        event_end = (
            None
            if _missing(event["interval_end_utc"])
            else _parse_utc(
                event["interval_end_utc"], field="event interval end"
            )
        )
        next_source = (
            None
            if _missing(event["next_episode_source_time_utc"])
            else _parse_utc(
                event["next_episode_source_time_utc"],
                field="next episode source time",
            )
        )
        for contract in contract_rows:
            key = (
                str(contract["venue"]),
                str(contract["logical_market_id"]),
                str(contract["outcome"]),
            )
            trades = series.get(key, [])
            for horizon in horizons:
                pre: MarketObservation | None = None
                post: list[MarketObservation] = []
                overlaps: list[MarketObservation] = []
                pre_ambiguous = False
                same_second_ambiguous = False
                deadline = (
                    None
                    if event_end is None
                    else event_end + timedelta(seconds=horizon)
                )
                if event_start is not None and event_end is not None:
                    pre_candidates = [
                        row
                        for row in trades
                        if row.source_interval.end <= event_start
                    ]
                    if pre_candidates:
                        latest = max(
                            row.source_interval.start
                            for row in pre_candidates
                        )
                        latest_rows = [
                            row
                            for row in pre_candidates
                            if row.source_interval.start == latest
                        ]
                        if len(latest_rows) == 1:
                            pre = latest_rows[0]
                        else:
                            pre_ambiguous = True
                    overlaps = [
                        row
                        for row in trades
                        if row.source_interval.end > event_start
                        and row.source_interval.start <= event_end
                    ]
                    assert deadline is not None
                    post = [
                        row
                        for row in trades
                        if row.source_interval.start > event_end
                        and row.source_interval.start <= deadline
                    ]
                    counts: dict[datetime, int] = defaultdict(int)
                    for row in post:
                        counts[row.source_interval.start] += 1
                    same_second_ambiguous = any(
                        count > 1 for count in counts.values()
                    )
                order_ambiguous = bool(
                    pre_ambiguous or overlaps or same_second_ambiguous
                )
                contaminated = bool(
                    deadline is not None
                    and next_source is not None
                    and next_source <= deadline
                )
                pre_price = None if pre is None else pre.price
                first = post[0] if post else None
                last = post[-1] if post else None
                first_price = None if first is None else first.price
                last_price = None if last is None else last.price
                volume = sum(
                    (row.size for row in post), Decimal("0")
                )
                vwap = (
                    None
                    if not post or volume == 0
                    else sum(
                        (row.price * row.size for row in post),  # type: ignore[operator]
                        Decimal("0"),
                    )
                    / volume
                )
                signed_change = (
                    None
                    if pre_price is None or last_price is None
                    else last_price - pre_price
                )
                excursion = _unique_signed_maximum_excursion(
                    tuple(
                        row.price
                        for row in post
                        if row.price is not None
                    ),
                    pre_price,
                )
                staleness = (
                    None
                    if deadline is None or last is None
                    else Decimal(
                        str(
                            (
                                deadline
                                - last.source_interval.start
                            ).total_seconds()
                        )
                    )
                )
                actual = bool(
                    pre_price is not None
                    and first_price is not None
                    and last_price is not None
                    and vwap is not None
                    and excursion is not None
                    and not order_ambiguous
                    and not contaminated
                )
                if event_start is None:
                    reason = "MISSING_EVENT_SOURCE_TIME"
                elif contaminated:
                    reason = "CONTAMINATED_BY_NEXT_EPISODE"
                elif order_ambiguous:
                    reason = "ORDER_AMBIGUOUS"
                elif pre_price is None:
                    reason = "NO_UNIQUE_PRE_EVENT_ACTUAL_TRADE"
                elif not post:
                    reason = "NO_POST_EVENT_ACTUAL_TRADE"
                elif not actual:
                    reason = "ACTUAL_TRADE_METRICS_UNDEFINED"
                else:
                    reason = "ELIGIBLE"
                metric_rows = (
                    ([pre] if pre is not None else []) + post + overlaps
                )
                lineage = tuple(
                    sorted(
                        {
                            digest
                            for child in metric_rows
                            for digest in market_lineage.get(
                                child.observation_id, ()
                            )
                        }
                    )
                )
                if not lineage:
                    lineage = (str(contract["rule_sha256"]),)
                rows.append(
                    {
                        "game_id": event["game_id"],
                        "episode_id": event["episode_id"],
                        "logical_market_id": contract[
                            "logical_market_id"
                        ],
                        "outcome": contract["outcome"],
                        "venue": contract["venue"],
                        "delay_seconds": event["delay_seconds"],
                        "horizon_seconds": int(horizon),
                        "event_source_time_utc": event[
                            "interval_start_utc"
                        ],
                        "event_interval_end_utc": event[
                            "interval_end_utc"
                        ],
                        "horizon_deadline_utc": (
                            None
                            if deadline is None
                            else _utc_text(deadline)
                        ),
                        "first_post_trade_time_utc": (
                            None
                            if first is None
                            else _utc_text(
                                first.source_interval.start
                            )
                        ),
                        "pre_event_actual_trade": pre_price,
                        "first_post_event_trade": first_price,
                        "horizon_vwap": vwap,
                        "horizon_last_trade": last_price,
                        "signed_price_change": signed_change,
                        "maximum_excursion": excursion,
                        "trade_count": len(post),
                        "volume": volume,
                        "staleness_seconds": staleness,
                        "actual_observation": actual,
                        "analysis_eligible": actual,
                        "analysis_exclusion_reason": reason,
                        "forward_filled": False,
                        "order_ambiguous": order_ambiguous,
                        "contaminated": contaminated,
                        "pre_event_observation_id": (
                            None if pre is None else pre.observation_id
                        ),
                        "first_post_observation_id": (
                            None if first is None else first.observation_id
                        ),
                        "last_post_observation_id": (
                            None if last is None else last.observation_id
                        ),
                        "horizon_trade_observation_ids": tuple(
                            row.observation_id for row in post
                        ),
                        "overlap_observation_ids": tuple(
                            row.observation_id for row in overlaps
                        ),
                        "market_source_hashes": lineage,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        [
            "game_id",
            "episode_id",
            "logical_market_id",
            "outcome",
            "venue",
            "delay_seconds",
            "horizon_seconds",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    grain = [
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "delay_seconds",
        "horizon_seconds",
    ]
    if result.duplicated(grain).any():
        raise DevelopmentMarketError("reaction path grain is duplicated")
    return result


def build_factor_observations_by_delay(
    *,
    sports_feature_view: pd.DataFrame,
    reaction_paths: pd.DataFrame,
    contract_inventory: pd.DataFrame,
    formal_factor_definitions: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Call the frozen V2 builder once per delay to preserve its old grain."""

    observation_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    contract_columns = {
        "contract_id",
        "family",
        "period",
        "subject",
        "measure",
        "kind",
        "rule_sha256",
    }
    for delay in DELAY_SCENARIOS_SECONDS:
        child = reaction_paths[
            reaction_paths["delay_seconds"].eq(delay)
        ].drop(
            columns=[
                column
                for column in contract_columns
                if column in reaction_paths.columns
            ]
        )
        built: FactorObservationBuildV2 = build_factor_observations_v2(
            sports_feature_view,
            child,
            contract_inventory,
            formal_factor_definitions,
        )
        observations = built.factor_observations.copy()
        audit = built.factor_exclusion_audit.copy()
        observations["delay_seconds"] = delay
        audit["delay_seconds"] = delay
        observation_frames.append(observations)
        audit_frames.append(audit)
    return (
        pd.concat(observation_frames, ignore_index=True),
        pd.concat(audit_frames, ignore_index=True),
    )


_FACTOR_DEFINITION_FIELDS = frozenset(
    {
        "factor_id",
        "version",
        "sports_meaning",
        "predicate",
        "pit_fields",
        "focal_entity",
        "exclusions",
        "target",
        "market_families",
        "horizons_seconds",
        "support_kind",
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
        "claim_boundary",
    }
)


@dataclass(frozen=True, slots=True)
class CompactFactorPartition:
    factor_id: str
    delay_seconds: int
    factor_observations: pd.DataFrame
    factor_exclusion_audit: pd.DataFrame
    predicate_attrition: pd.DataFrame


def _builder_definition(
    definition: Mapping[str, object],
) -> dict[str, object]:
    if not _FACTOR_DEFINITION_FIELDS <= set(definition):
        raise DevelopmentMarketError(
            "formal factor definition fields are incomplete"
        )
    result = {
        field: definition[field]
        for field in _FACTOR_DEFINITION_FIELDS
    }
    for key in ("predicate", "exclusions"):
        clauses = result[key]
        if not isinstance(clauses, Sequence) or isinstance(
            clauses, (str, bytes, bytearray)
        ):
            raise DevelopmentMarketError(
                f"factor {key} must be a sequence"
            )
        result[key] = [
            {
                **dict(clause),
                "value": _factor_predicate_value(clause.get("value")),
            }
            for clause in clauses
            if isinstance(clause, Mapping)
        ]
        if len(result[key]) != len(clauses):  # type: ignore[arg-type]
            raise DevelopmentMarketError(
                f"factor {key} contains a non-mapping clause"
            )
    return result


def _factor_predicate_value(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DevelopmentMarketError(
                "factor predicate Decimal is not finite"
            )
        return float(value)
    if isinstance(value, (list, tuple)):
        return [
            _factor_predicate_value(child) for child in value
        ]
    return value


def _predicate_mask(
    frame: pd.DataFrame,
    clauses: Sequence[Mapping[str, object]],
) -> pd.Series:
    result = pd.Series(True, index=frame.index, dtype=bool)
    for clause in clauses:
        if set(clause) != {"field", "operator", "value"}:
            raise DevelopmentMarketError(
                "factor predicate clause fields differ"
            )
        field = clause["field"]
        operator = clause["operator"]
        if type(field) is not str or field not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        values = frame[field]
        target = clause["value"]
        if operator == "eq":
            observed = values.eq(target)
        elif operator == "ne":
            observed = values.ne(target)
        elif operator == "lt":
            observed = values.lt(target)
        elif operator == "le":
            observed = values.le(target)
        elif operator == "gt":
            observed = values.gt(target)
        elif operator == "ge":
            observed = values.ge(target)
        elif operator == "in":
            observed = values.isin(target)
        elif operator == "not_in":
            observed = ~values.isin(target)
        elif operator == "is_null":
            observed = values.isna()
        elif operator == "not_null":
            observed = values.notna()
        else:
            raise DevelopmentMarketError(
                f"unsupported factor predicate operator: {operator}"
            )
        result &= observed.fillna(False)
    return result


def _factor_hit_paths(
    *,
    sports: pd.DataFrame,
    paths: pd.DataFrame,
    definition: Mapping[str, object],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    predicate = definition["predicate"]
    if not isinstance(predicate, Sequence) or isinstance(
        predicate, (str, bytes, bytearray)
    ):
        raise DevelopmentMarketError("factor predicate is not a sequence")
    fields = tuple(
        sorted(
            {
                str(clause.get("field"))
                for clause in predicate
                if isinstance(clause, Mapping)
            }
        )
    )
    missing = tuple(
        field
        for field in fields
        if field not in sports.columns and field not in paths.columns
    )
    if missing:
        return paths.iloc[0:0].copy(), missing
    if set(fields) <= set(sports.columns):
        matched = sports.loc[
            _predicate_mask(sports, predicate),  # type: ignore[arg-type]
            ["game_id", "episode_id"],
        ].drop_duplicates()
        hit = paths.merge(
            matched,
            on=["game_id", "episode_id"],
            how="inner",
            validate="many_to_one",
        )
        return hit, ()
    if set(fields) <= set(paths.columns):
        return (
            paths.loc[
                _predicate_mask(paths, predicate)  # type: ignore[arg-type]
            ].copy(),
            (),
        )
    sports_fields = [
        field for field in fields if field in sports.columns
    ]
    joined = paths.merge(
        sports[
            ["game_id", "episode_id", *sports_fields]
        ],
        on=["game_id", "episode_id"],
        how="left",
        validate="many_to_many",
    )
    matched = joined.loc[
        _predicate_mask(joined, predicate),  # type: ignore[arg-type]
        [
            "game_id",
            "episode_id",
            "logical_market_id",
            "outcome",
            "venue",
            "horizon_seconds",
        ],
    ].drop_duplicates()
    hit = paths.merge(
        matched,
        on=[
            "game_id",
            "episode_id",
            "logical_market_id",
            "outcome",
            "venue",
            "horizon_seconds",
        ],
        how="inner",
        validate="one_to_one",
    )
    return hit, ()


def iter_compact_factor_partitions(
    *,
    sports_feature_view: pd.DataFrame,
    reaction_paths: pd.DataFrame,
    contract_inventory: pd.DataFrame,
    formal_factor_definitions: Sequence[Mapping[str, object]],
) -> Sequence[CompactFactorPartition]:
    """Build only predicate-hit row audits; aggregate every predicate miss."""

    definitions = tuple(
        sorted(
            (
                _builder_definition(definition)
                for definition in formal_factor_definitions
            ),
            key=lambda definition: str(definition["factor_id"]),
        )
    )
    if len(definitions) != 101:
        raise DevelopmentMarketError(
            "compact formal factor build requires exact 101 definitions"
        )
    delays = tuple(
        sorted(
            int(value)
            for value in reaction_paths["delay_seconds"].unique()
        )
    )
    if not set(delays) <= set(DELAY_SCENARIOS_SECONDS):
        raise DevelopmentMarketError(
            "reaction paths contain an unregistered delay"
        )
    partitions: list[CompactFactorPartition] = []
    for delay in delays:
        delayed = reaction_paths[
            reaction_paths["delay_seconds"].eq(delay)
        ].copy()
        for definition in definitions:
            registered_horizons = {
                int(value)
                for value in definition["horizons_seconds"]  # type: ignore[union-attr]
            }
            candidates = delayed[
                delayed["horizon_seconds"].isin(registered_horizons)
            ].copy()
            hit, missing_fields = _factor_hit_paths(
                sports=sports_feature_view,
                paths=candidates,
                definition=definition,
            )
            attrition_rows = []
            for horizon in sorted(registered_horizons):
                candidate_count = int(
                    candidates["horizon_seconds"].eq(horizon).sum()
                )
                hit_count = int(
                    hit["horizon_seconds"].eq(horizon).sum()
                )
                attrition_rows.append(
                    {
                        "factor_id": definition["factor_id"],
                        "factor_version": definition["version"],
                        "game_id": (
                            str(candidates["game_id"].iloc[0])
                            if not candidates.empty
                            else str(
                                sports_feature_view[
                                    "game_id"
                                ].iloc[0]
                            )
                        ),
                        "market_family": "moneyline",
                        "delay_seconds": delay,
                        "horizon_seconds": horizon,
                        "candidate_path_count": candidate_count,
                        "predicate_hit_path_count": hit_count,
                        "predicate_miss_path_count": (
                            candidate_count - hit_count
                        ),
                        "missing_predicate_fields": missing_fields,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                )
            if hit.empty:
                observations = pd.DataFrame()
                audit = pd.DataFrame()
            else:
                built = build_factor_observations_v2(
                    sports_feature_view,
                    hit,
                    contract_inventory,
                    (definition,),
                )
                observations = built.factor_observations.copy()
                audit = built.factor_exclusion_audit.copy()
                observations["delay_seconds"] = delay
                audit["delay_seconds"] = delay
                if (
                    not audit.empty
                    and audit["predicate_hit"].eq(False).any()
                ):
                    raise DevelopmentMarketError(
                        "compact audit retained a predicate miss row"
                    )
            partitions.append(
                CompactFactorPartition(
                    factor_id=str(definition["factor_id"]),
                    delay_seconds=delay,
                    factor_observations=observations,
                    factor_exclusion_audit=audit,
                    predicate_attrition=pd.DataFrame(attrition_rows),
                )
            )
    return tuple(partitions)


def build_compact_factor_observations(
    *,
    sports_feature_view: pd.DataFrame,
    reaction_paths: pd.DataFrame,
    contract_inventory: pd.DataFrame,
    formal_factor_definitions: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """In-memory convenience wrapper for bounded equivalence tests."""

    partitions = iter_compact_factor_partitions(
        sports_feature_view=sports_feature_view,
        reaction_paths=reaction_paths,
        contract_inventory=contract_inventory,
        formal_factor_definitions=formal_factor_definitions,
    )
    observations = [
        partition.factor_observations
        for partition in partitions
        if not partition.factor_observations.empty
    ]
    audits = [
        partition.factor_exclusion_audit
        for partition in partitions
        if not partition.factor_exclusion_audit.empty
    ]
    attrition = [
        partition.predicate_attrition for partition in partitions
    ]
    return (
        (
            pd.concat(observations, ignore_index=True)
            if observations
            else pd.DataFrame()
        ),
        pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
        pd.concat(attrition, ignore_index=True),
    )


def build_compact_exclusion_stage(
    row_exclusions: pd.DataFrame,
    predicate_attrition: pd.DataFrame,
) -> pd.DataFrame:
    """Publish hit-level exclusions and miss aggregates in one typed stage."""

    row_child = row_exclusions.copy()
    aggregate_child = predicate_attrition.copy()
    row_child["row_kind"] = "PREDICATE_HIT_ROW_AUDIT"
    aggregate_child["row_kind"] = "PREDICATE_MISS_AGGREGATE"
    columns = sorted(
        set(row_child.columns) | set(aggregate_child.columns)
    )
    for frame in (row_child, aggregate_child):
        for column in columns:
            if column not in frame:
                frame[column] = pd.Series(
                    [None] * len(frame), dtype=object
                )
    result = pd.concat(
        (row_child[columns], aggregate_child[columns]),
        ignore_index=True,
    )
    leading = [
        "row_kind",
        "factor_id",
        "factor_version",
        "game_id",
        "delay_seconds",
        "horizon_seconds",
    ]
    return result[
        [*leading, *sorted(set(columns).difference(leading))]
    ].sort_values(
        [
            "factor_id",
            "delay_seconds",
            "horizon_seconds",
            "row_kind",
            *(
                ["episode_id"]
                if "episode_id" in result.columns
                else []
            ),
            *(["venue"] if "venue" in result.columns else []),
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _finite_decimal_or_none(value: object) -> Decimal | None:
    if value is None or _missing(value):
        return None
    try:
        return _decimal(value, field="factor result metric")
    except DevelopmentMarketError:
        return None


def build_single_game_results(
    factor_observations: pd.DataFrame,
    *,
    formal_factor_definitions: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Aggregate eligible effects inside one game before cross-game analysis."""

    columns = [
        "factor_id",
        "factor_version",
        "game_id",
        "market_family",
        "horizon_seconds",
        "delay_seconds",
        "venue",
        "observation_count",
        "episode_count",
        "sum_effect",
        "sum_squared_effect",
        "point_estimate",
        "maximum_absolute_episode_effect",
        "trade_count",
        "volume",
        "mean_staleness_seconds",
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
        "claim_boundary",
    ]
    if factor_observations.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "factor_id",
        "factor_version",
        "game_id",
        "market_family",
        "horizon_seconds",
        "delay_seconds",
        "venue",
        "episode_id",
        "signed_price_change",
        "trade_count",
        "volume",
        "staleness_seconds",
    }
    missing = sorted(required.difference(factor_observations.columns))
    if missing:
        raise DevelopmentMarketError(
            "factor observations missing result columns: "
            + ", ".join(missing)
        )
    definitions = {
        str(definition["factor_id"]): definition
        for definition in formal_factor_definitions
    }
    frame = factor_observations.copy()
    frame["_effect"] = frame["signed_price_change"].map(
        _finite_decimal_or_none
    )
    if frame["_effect"].isna().any():
        raise DevelopmentMarketError(
            "formal factor observation has a non-finite effect"
        )
    frame["_volume"] = frame["volume"].map(
        _finite_decimal_or_none
    )
    frame["_staleness"] = frame["staleness_seconds"].map(
        _finite_decimal_or_none
    )
    rows: list[dict[str, object]] = []
    base_keys = [
        "factor_id",
        "factor_version",
        "game_id",
        "market_family",
        "horizon_seconds",
        "delay_seconds",
    ]

    def append_result(
        key: tuple[object, ...],
        child: pd.DataFrame,
        *,
        venue: str,
        effect_column: str,
    ) -> None:
        effects = tuple(child[effect_column])
        total = sum(effects, Decimal("0"))
        squared = sum(
            (effect * effect for effect in effects), Decimal("0")
        )
        factor_id = str(key[0])
        definition = definitions.get(factor_id)
        if definition is None:
            raise DevelopmentMarketError(
                f"single-game result factor is unregistered: {factor_id}"
            )
        staleness = tuple(
            value
            for value in child.get(
                "_staleness", pd.Series(dtype=object)
            )
            if isinstance(value, Decimal)
        )
        volumes = tuple(
            value
            for value in child.get(
                "_volume", pd.Series(dtype=object)
            )
            if isinstance(value, Decimal)
        )
        rows.append(
            {
                **dict(zip(base_keys, key, strict=True)),
                "venue": venue,
                "observation_count": len(effects),
                "episode_count": int(child["episode_id"].nunique()),
                "sum_effect": total,
                "sum_squared_effect": squared,
                "point_estimate": total / Decimal(len(effects)),
                "maximum_absolute_episode_effect": max(
                    abs(effect) for effect in effects
                ),
                "trade_count": int(
                    pd.to_numeric(
                        child.get(
                            "trade_count",
                            pd.Series(0, index=child.index),
                        ),
                        errors="raise",
                    ).sum()
                ),
                "volume": sum(volumes, Decimal("0")),
                "mean_staleness_seconds": (
                    None
                    if not staleness
                    else sum(staleness, Decimal("0"))
                    / Decimal(len(staleness))
                ),
                "minimum_episodes": int(
                    definition["minimum_episodes"]
                ),
                "minimum_games": int(definition["minimum_games"]),
                "minimum_episodes_per_venue": int(
                    definition["minimum_episodes_per_venue"]
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    for key, child in frame.groupby(
        [*base_keys, "venue"], sort=True, dropna=False
    ):
        append_result(
            tuple(key[:-1]),
            child,
            venue=str(key[-1]),
            effect_column="_effect",
        )
    episode = (
        frame.groupby(
            [*base_keys, "episode_id"],
            sort=True,
            dropna=False,
        )
        .agg(
            _effect=("_effect", lambda values: (
                sum(values, Decimal("0")) / Decimal(len(values))
            )),
            _volume=("_volume", lambda values: sum(
                (
                    value
                    for value in values
                    if isinstance(value, Decimal)
                ),
                Decimal("0"),
            )),
            _staleness=("_staleness", lambda values: (
                None
                if not (
                    present := [
                        value
                        for value in values
                        if isinstance(value, Decimal)
                    ]
                )
                else sum(present, Decimal("0"))
                / Decimal(len(present))
            )),
            trade_count=("trade_count", "sum"),
        )
        .reset_index()
    )
    for key, child in episode.groupby(
        base_keys, sort=True, dropna=False
    ):
        append_result(
            tuple(key),
            child,
            venue="ALL_EPISODE_MEAN",
            effect_column="_effect",
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        [
            "factor_id",
            "delay_seconds",
            "horizon_seconds",
            "venue",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, Decimal):
        result = float(value)
    elif isinstance(value, bool):
        raise DevelopmentMarketError(f"{field} must be numeric")
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise DevelopmentMarketError(
                f"{field} must be numeric"
            ) from error
    if not math.isfinite(result):
        raise DevelopmentMarketError(f"{field} must be finite")
    return result


def _benjamini_hochberg(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    count = len(raw)
    if count == 0 or np.any(~np.isfinite(raw)) or np.any(
        (raw < 0) | (raw > 1)
    ):
        raise DevelopmentMarketError(
            "BH input must contain finite probabilities"
        )
    order = np.argsort(raw, kind="mergesort")
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        source_index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, raw[source_index] * count / rank)
        adjusted[source_index] = min(1.0, running)
    return adjusted


def _factor_horizon_grid(
    formal_factor_definitions: Sequence[Mapping[str, object]],
) -> tuple[
    tuple[Mapping[str, object], int],
    ...,
]:
    definitions = tuple(
        sorted(
            (
                _builder_definition(definition)
                for definition in formal_factor_definitions
            ),
            key=lambda definition: str(definition["factor_id"]),
        )
    )
    if (
        len(definitions) != 101
        or len({str(row["factor_id"]) for row in definitions}) != 101
    ):
        raise DevelopmentMarketError(
            "cross-game discovery requires exact 101 formal factors"
        )
    grid = tuple(
        (definition, int(horizon))
        for definition in definitions
        for horizon in sorted(
            int(value)
            for value in definition["horizons_seconds"]  # type: ignore[union-attr]
        )
    )
    if (
        len(grid) != 101 * len(HORIZONS_SECONDS)
        or {
            horizon for _, horizon in grid
        } != set(HORIZONS_SECONDS)
        or len(
            {
                (str(definition["factor_id"]), horizon)
                for definition, horizon in grid
            }
        )
        != len(grid)
    ):
        raise DevelopmentMarketError(
            "formal factor/horizon grid differs from registered 707 tests"
        )
    return grid


def build_cross_game_discovery(
    single_game_results: pd.DataFrame,
    *,
    formal_factor_definitions: Sequence[Mapping[str, object]],
    expected_game_ids: Sequence[str] | frozenset[str],
    bootstrap_samples: int = DISCOVERY_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DISCOVERY_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Run delay-zero game-cluster discovery from episode-mean game results."""

    if (
        bootstrap_samples != DISCOVERY_BOOTSTRAP_SAMPLES
        or bootstrap_seed != DISCOVERY_BOOTSTRAP_SEED
    ):
        raise DevelopmentMarketError(
            "cross-game discovery bootstrap parameters differ from registry"
        )
    game_ids = frozenset(str(value) for value in expected_game_ids)
    if len(game_ids) != 153:
        raise DevelopmentMarketError(
            "cross-game discovery requires exact 153 development games"
        )
    required = {
        "factor_id",
        "factor_version",
        "game_id",
        "market_family",
        "horizon_seconds",
        "delay_seconds",
        "venue",
        "observation_count",
        "episode_count",
        "sum_effect",
        "maximum_absolute_episode_effect",
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
    }
    missing = sorted(required.difference(single_game_results.columns))
    if missing:
        raise DevelopmentMarketError(
            "single-game results missing discovery columns: "
            + ", ".join(missing)
        )
    frame = single_game_results.copy()
    if (
        set(frame["game_id"].dropna()) != game_ids
        or set(frame["delay_seconds"].dropna()) != {0}
        or set(frame["market_family"].dropna()) != {"moneyline"}
        or not set(frame["venue"].dropna()) <= {
            "ALL_EPISODE_MEAN",
            "kalshi",
            "polymarket",
        }
    ):
        raise DevelopmentMarketError(
            "single-game results differ from delay-zero development scope"
        )
    grain = [
        "factor_id",
        "factor_version",
        "game_id",
        "market_family",
        "horizon_seconds",
        "delay_seconds",
        "venue",
    ]
    if frame.duplicated(grain).any():
        raise DevelopmentMarketError(
            "single-game discovery result grain is duplicated"
        )
    grid = _factor_horizon_grid(formal_factor_definitions)
    definitions = {
        str(definition["factor_id"]): definition
        for definition, _ in grid
    }
    unknown = set(frame["factor_id"].dropna()).difference(definitions)
    if unknown:
        raise DevelopmentMarketError(
            "single-game results contain unregistered factors"
        )
    expected_versions = frame["factor_id"].map(
        {
            factor_id: definition["version"]
            for factor_id, definition in definitions.items()
        }
    )
    if not frame["factor_version"].eq(expected_versions).all():
        raise DevelopmentMarketError(
            "single-game factor version differs from registry"
        )
    registered_horizons = {
        factor_id: {
            int(value)
            for value in definition["horizons_seconds"]  # type: ignore[union-attr]
        }
        for factor_id, definition in definitions.items()
    }
    if any(
        int(row.horizon_seconds)
        not in registered_horizons[str(row.factor_id)]
        for row in frame[["factor_id", "horizon_seconds"]].itertuples(
            index=False
        )
    ):
        raise DevelopmentMarketError(
            "single-game result horizon differs from registry"
        )
    for field in (
        "observation_count",
        "episode_count",
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
    ):
        numeric = pd.to_numeric(frame[field], errors="raise")
        if numeric.isna().any() or (numeric < 0).any():
            raise DevelopmentMarketError(
                f"single-game result {field} is invalid"
            )
        frame[field] = numeric.astype("int64")
    for field in (
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
    ):
        expected = frame["factor_id"].map(
            {
                factor_id: int(definition[field])
                for factor_id, definition in definitions.items()
            }
        )
        if not frame[field].eq(expected).all():
            raise DevelopmentMarketError(
                f"single-game result {field} differs from registry"
            )
    frame["_sum_effect_float"] = frame["sum_effect"].map(
        lambda value: _finite_float(
            value, field="single-game sum_effect"
        )
    )
    frame["_maximum_absolute_episode_effect_float"] = frame[
        "maximum_absolute_episode_effect"
    ].map(
        lambda value: _finite_float(
            value,
            field="single-game maximum_absolute_episode_effect",
        )
    )
    all_episode = frame[
        frame["venue"].eq("ALL_EPISODE_MEAN")
    ].copy()
    if not all_episode["observation_count"].eq(
        all_episode["episode_count"]
    ).all():
        raise DevelopmentMarketError(
            "primary game result is not one effect per episode"
        )
    venues = frame[
        frame["venue"].isin(("kalshi", "polymarket"))
    ].copy()
    rows: list[dict[str, object]] = []
    for stream_index, (definition, horizon) in enumerate(grid):
        factor_id = str(definition["factor_id"])
        child = all_episode[
            all_episode["factor_id"].eq(factor_id)
            & all_episode["horizon_seconds"].eq(horizon)
        ].sort_values("game_id", kind="mergesort")
        venue_child = venues[
            venues["factor_id"].eq(factor_id)
            & venues["horizon_seconds"].eq(horizon)
        ]
        sums = child["_sum_effect_float"].to_numpy(dtype=float)
        counts = child["observation_count"].to_numpy(dtype=float)
        total_count = int(counts.sum())
        total_sum = float(sums.sum())
        game_count = len(child)
        point_estimate = (
            total_sum / total_count if total_count else math.nan
        )
        if game_count:
            draws = np.random.default_rng(
                bootstrap_seed + stream_index
            ).integers(
                0,
                game_count,
                size=(bootstrap_samples, game_count),
                endpoint=False,
            )
            bootstrap = (
                sums[draws].sum(axis=1)
                / counts[draws].sum(axis=1)
            )
            ci_low, ci_high = np.quantile(
                bootstrap, (0.025, 0.975)
            )
            raw_p_value = min(
                1.0,
                2.0
                * min(
                    (
                        float(np.count_nonzero(bootstrap <= 0))
                        + 1.0
                    )
                    / (bootstrap_samples + 1.0),
                    (
                        float(np.count_nonzero(bootstrap >= 0))
                        + 1.0
                    )
                    / (bootstrap_samples + 1.0),
                ),
            )
            loo = np.asarray(
                [
                    (total_sum - game_sum)
                    / (total_count - game_observations)
                    for game_sum, game_observations in zip(
                        sums, counts, strict=True
                    )
                    if total_count > game_observations
                ],
                dtype=float,
            )
            loo_sign_stability = (
                float(
                    np.mean(
                        np.sign(loo) == np.sign(point_estimate)
                    )
                )
                if len(loo) and point_estimate != 0
                else 0.0
            )
            maximum_loo_shift = (
                float(
                    np.max(np.abs(loo - point_estimate))
                )
                if len(loo)
                else math.nan
            )
            contribution_denominator = float(
                np.abs(sums).sum()
            )
            maximum_game_contribution = (
                float(
                    np.abs(sums).max()
                    / contribution_denominator
                )
                if contribution_denominator > 0
                else 1.0
            )
            bootstrap_samples_valid = bootstrap_samples
        else:
            ci_low = ci_high = math.nan
            raw_p_value = 1.0
            loo_sign_stability = 0.0
            maximum_loo_shift = math.nan
            maximum_game_contribution = 1.0
            bootstrap_samples_valid = 0
        venue_statistics: dict[str, dict[str, float | int]] = {}
        for venue in ("kalshi", "polymarket"):
            child_venue = venue_child[
                venue_child["venue"].eq(venue)
            ]
            venue_observations = int(
                child_venue["observation_count"].sum()
            )
            venue_sum = float(
                child_venue["_sum_effect_float"].sum()
            )
            venue_statistics[venue] = {
                "episode_count": int(
                    child_venue["episode_count"].sum()
                ),
                "game_count": int(
                    child_venue["game_id"].nunique()
                ),
                "point_estimate": (
                    venue_sum / venue_observations
                    if venue_observations
                    else math.nan
                ),
            }
        kalshi_point = float(
            venue_statistics["kalshi"]["point_estimate"]
        )
        polymarket_point = float(
            venue_statistics["polymarket"]["point_estimate"]
        )
        venue_sign_consistent = (
            math.isfinite(kalshi_point)
            and math.isfinite(polymarket_point)
            and kalshi_point != 0
            and polymarket_point != 0
            and np.sign(kalshi_point)
            == np.sign(polymarket_point)
        )
        episode_count = int(child["episode_count"].sum())
        minimum_per_venue = int(
            definition["minimum_episodes_per_venue"]
        )
        support_gate_pass = (
            episode_count >= int(definition["minimum_episodes"])
            and game_count >= int(definition["minimum_games"])
            and (
                minimum_per_venue == 0
                or all(
                    int(value["episode_count"])
                    >= minimum_per_venue
                    for value in venue_statistics.values()
                )
            )
        )
        maximum_episode_effect = (
            float(
                child[
                    "_maximum_absolute_episode_effect_float"
                ].max()
            )
            if not child.empty
            else math.nan
        )
        rows.append(
            {
                "factor_id": factor_id,
                "factor_version": definition["version"],
                "market_family": "moneyline",
                "horizon_seconds": horizon,
                "delay_seconds": 0,
                "target": definition["target"],
                "support_kind": definition["support_kind"],
                "episode_count": episode_count,
                "game_count": game_count,
                "observation_count": total_count,
                "per_game_result_count": game_count,
                "kalshi_episode_count": int(
                    venue_statistics["kalshi"]["episode_count"]
                ),
                "polymarket_episode_count": int(
                    venue_statistics["polymarket"][
                        "episode_count"
                    ]
                ),
                "kalshi_game_count": int(
                    venue_statistics["kalshi"]["game_count"]
                ),
                "polymarket_game_count": int(
                    venue_statistics["polymarket"]["game_count"]
                ),
                "kalshi_point_estimate": kalshi_point,
                "polymarket_point_estimate": polymarket_point,
                "venue_sign_consistent": bool(
                    venue_sign_consistent
                ),
                "minimum_episodes": int(
                    definition["minimum_episodes"]
                ),
                "minimum_games": int(definition["minimum_games"]),
                "minimum_episodes_per_venue": minimum_per_venue,
                "support_gate_pass": bool(support_gate_pass),
                "point_estimate": point_estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "raw_p_value": float(raw_p_value),
                "loo_sign_stability_rate": loo_sign_stability,
                "maximum_leave_one_game_out_shift": (
                    maximum_loo_shift
                ),
                "maximum_single_game_contribution": (
                    maximum_game_contribution
                ),
                "maximum_absolute_episode_effect": (
                    maximum_episode_effect
                ),
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_stream_index": stream_index,
                "bootstrap_samples_requested": bootstrap_samples,
                "bootstrap_samples_valid": bootstrap_samples_valid,
            }
        )
    result = pd.DataFrame(rows)
    result["bh_q_value"] = _benjamini_hochberg(
        result["raw_p_value"]
    )
    result["multiple_testing_method"] = "benjamini_hochberg"
    result["multiple_testing_family_size"] = len(result)
    result["support_status"] = np.select(
        [
            result["observation_count"].eq(0),
            result["support_gate_pass"].eq(True),
        ],
        ["DATA_GAP", "PASS"],
        default="INSUFFICIENT_SUPPORT",
    )
    result["analysis_status"] = np.select(
        [
            result["observation_count"].eq(0),
            ~result["support_gate_pass"].eq(True),
        ],
        ["DATA_GAP", "INSUFFICIENT_SUPPORT"],
        default="DESCRIPTIVE",
    )
    result["estimand_type"] = np.where(
        result["support_kind"].eq("binary"),
        "CONDITIONAL_MEAN_NO_COMPLEMENT",
        "CONDITIONAL_MEAN_SIGNED_CHANGE",
    )
    result["promotion_status"] = (
        "NOT_PROMOTED_DEVELOPMENT_DESCRIPTIVE_ONLY"
    )
    result["claim_boundary"] = CLAIM_BOUNDARY
    return result.sort_values(
        ["factor_id", "horizon_seconds"],
        kind="mergesort",
    ).reset_index(drop=True)


def publish_verified_cross_game_discovery(
    inputs: object,
    *,
    output_root: Path,
    batch: object,
) -> object:
    """Build and publish discovery only from a verified exact-153 batch."""

    from prediction_market.sports.nfl_expansion_development_market_artifacts import (
        publish_cross_game_discovery,
    )

    try:
        single_game_results = pd.read_parquet(
            batch.merged_single_game_results.object_path
        )
    except Exception as error:
        raise DevelopmentMarketError(
            "verified merged single-game results are unreadable"
        ) from error
    discovery = build_cross_game_discovery(
        single_game_results,
        formal_factor_definitions=inputs.formal_factor_definitions,
        expected_game_ids=inputs.unlock.development_game_ids,
    )
    grid = tuple(
        (str(definition["factor_id"]), horizon)
        for definition, horizon in _factor_horizon_grid(
            inputs.formal_factor_definitions
        )
    )
    return publish_cross_game_discovery(
        inputs,
        output_root=Path(output_root),
        batch=batch,
        discovery=discovery,
        expected_factor_horizon_grid=grid,
        builder_version=BUILDER_VERSION,
        source_bindings={
            "unlock_object_sha256": inputs.unlock.object_sha256,
            "capture_batch_sha256": inputs.capture_batch_sha256,
            "feature_batch_sha256": inputs.feature_batch_sha256,
            "factor_registry_sha256": inputs.factor_registry_sha256,
            "development_market_batch_sha256": batch.batch_sha256,
            "merged_single_game_results_sha256": (
                batch.merged_single_game_results.object_sha256
            ),
        },
        claim_boundary=CLAIM_BOUNDARY,
    )


def _game_input_fingerprint(
    inputs: object,
    game_id: str,
    *,
    delay_scenarios: Sequence[int],
) -> str:
    capture = inputs.development_capture_games[game_id]
    feature = inputs.development_feature_games[game_id]
    return _semantic_sha(
        {
            "schema": "nfl_expansion_development_market_input_v1",
            "builder_code_sha256": _sha256(
                Path(__file__).read_bytes()
            ),
            "builder_version": BUILDER_VERSION,
            "game_id": game_id,
            "delay_scenarios_seconds": list(delay_scenarios),
            "horizons_seconds": list(HORIZONS_SECONDS),
            "unlock_object_sha256": inputs.unlock.object_sha256,
            "capture_batch_sha256": inputs.capture_batch_sha256,
            "capture_index_sha256": capture.index_sha256,
            "feature_batch_sha256": inputs.feature_batch_sha256,
            "feature_manifest_sha256": feature.manifest_sha256,
            "feature_object_sha256": feature.object_sha256,
            "feature_rows_sha256": feature.rows_sha256,
            "factor_registry_sha256": inputs.factor_registry_sha256,
            "formal_factor_count": len(
                inputs.formal_factor_definitions
            ),
        }
    )


def _game_source_bindings(
    inputs: object,
    game_id: str,
) -> dict[str, object]:
    capture = inputs.development_capture_games[game_id]
    feature = inputs.development_feature_games[game_id]
    return {
        "unlock_object_sha256": inputs.unlock.object_sha256,
        "unlock_manifest_sha256": inputs.unlock.manifest_sha256,
        "capture_batch_sha256": inputs.capture_batch_sha256,
        "capture_index_sha256": capture.index_sha256,
        "capture_checkpoint_sha256": capture.checkpoint_sha256,
        "feature_batch_file_sha256": (
            inputs.feature_batch_file_sha256
        ),
        "feature_batch_sha256": inputs.feature_batch_sha256,
        "feature_manifest_sha256": feature.manifest_sha256,
        "feature_object_sha256": feature.object_sha256,
        "feature_rows_sha256": feature.rows_sha256,
        "factor_registry_sha256": inputs.factor_registry_sha256,
    }


def build_verified_single_game_stages(
    inputs: object,
    game_id: str,
    *,
    delay_scenarios: Sequence[int],
) -> Mapping[str, pd.DataFrame]:
    """Materialize the exact eight registered stages for one verified game."""

    from prediction_market.sports.nfl_expansion_development_market_artifacts import (
        load_development_capture_game,
        load_development_capture_raw_documents,
        load_development_feature_game,
    )

    delays = tuple(int(value) for value in delay_scenarios)
    if (
        delays != tuple(sorted(set(delays)))
        or not set(delays) <= set(DELAY_SCENARIOS_SECONDS)
        or not delays
    ):
        raise DevelopmentMarketError(
            "single-game delay scenarios are not registered"
        )
    inputs.require_development(game_id)
    capture = load_development_capture_game(inputs, game_id)
    raw_documents = load_development_capture_raw_documents(
        inputs, game_id
    )
    documents = tuple(
        VerifiedMarketDocument(
            source=(
                "polymarket"
                if "polymarket.com" in document.source_url
                else "kalshi"
            ),
            source_url=document.source_url,
            source_request=document.manifest_document[
                "source_request"
            ],
            object_bytes=document.object_bytes,
            object_sha256=document.object_sha256,
            manifest_sha256=document.manifest_sha256,
            request_sha256=document.request_sha256,
        )
        for document in raw_documents
    )
    binding = build_moneyline_binding(
        identity=capture.index_document["identity"],
        documents=documents,
    )
    observations, market_lineage = (
        normalize_verified_market_documents(
            binding=binding,
            documents=documents,
        )
    )
    contracts = _moneyline_contract_inventory(binding)
    contract_lineage = tuple(
        sorted(
            {
                binding.rule_sha256,
                capture.index_sha256,
                *(
                    document.manifest_sha256
                    for document in documents
                    if (
                        "gamma-api.polymarket.com" in document.source_url
                        or document.source_url.endswith(
                            "/historical/markets"
                        )
                    )
                ),
            }
        )
    )
    contracts["source_hashes"] = [
        contract_lineage
    ] * len(contracts)
    market = actual_market_observation_frame(
        game_id, observations
    )
    market["market_source_hashes"] = market[
        "observation_id"
    ].map(market_lineage)
    if market["market_source_hashes"].isna().any():
        raise DevelopmentMarketError(
            "actual market observation lineage is incomplete"
        )
    layer_m = build_layer_m_intervals(market)
    feature, sports = load_development_feature_game(inputs, game_id)
    sports_hashes = (
        feature.manifest_sha256,
        feature.object_sha256,
        feature.rows_sha256,
    )
    sports["sports_source_hashes"] = [
        sports_hashes
    ] * len(sports)
    layer_g = build_layer_g_timeline(
        sports, delay_scenarios=delays
    )
    reaction_paths = build_reaction_paths(
        layer_g=layer_g,
        contracts=contracts,
        observations=observations,
        market_lineage=market_lineage,
    )
    (
        factor_observations,
        factor_row_audit,
        predicate_attrition,
    ) = build_compact_factor_observations(
        sports_feature_view=sports,
        reaction_paths=reaction_paths,
        contract_inventory=contracts,
        formal_factor_definitions=inputs.formal_factor_definitions,
    )
    factor_exclusion_audit = build_compact_exclusion_stage(
        factor_row_audit, predicate_attrition
    )
    single_game_results = build_single_game_results(
        factor_observations,
        formal_factor_definitions=inputs.formal_factor_definitions,
    )
    return {
        "contract_inventory": contracts,
        "actual_market_observations": market,
        "layer_m_intervals": layer_m,
        "layer_g_source_timeline": layer_g,
        "reaction_paths": reaction_paths,
        "factor_observations": factor_observations,
        "factor_exclusion_audit": factor_exclusion_audit,
        "single_game_results": single_game_results,
    }


def publish_verified_single_game(
    inputs: object,
    *,
    output_root: Path,
    game_id: str,
    delay_scenarios: Sequence[int],
) -> object:
    """Resume or publish one complete content-addressed game bundle."""

    from prediction_market.sports.nfl_expansion_development_market_artifacts import (
        load_single_game_checkpoint,
        publish_single_game_stages,
    )

    fingerprint = _game_input_fingerprint(
        inputs, game_id, delay_scenarios=delay_scenarios
    )
    resumed = load_single_game_checkpoint(
        inputs,
        output_root=output_root,
        game_id=game_id,
        expected_input_fingerprint=fingerprint,
    )
    if resumed is not None:
        return resumed
    stages = build_verified_single_game_stages(
        inputs, game_id, delay_scenarios=delay_scenarios
    )
    return publish_single_game_stages(
        inputs,
        output_root=output_root,
        game_id=game_id,
        stages=stages,
        input_fingerprint=fingerprint,
        builder_version=BUILDER_VERSION,
        source_bindings=_game_source_bindings(inputs, game_id),
        claim_boundary=CLAIM_BOUNDARY,
    )


def _fixture_sha(label: str) -> str:
    return _sha256(label.encode("ascii"))


def run_bounded_fixture() -> dict[str, object]:
    """Exercise the development-only contracts without filesystem evidence."""

    development = frozenset(
        {"2025_01_AAA_BBB"}
        | {f"2025_{index:02d}_D{index:03d}_E{index:03d}"
           for index in range(2, 154)}
    )
    final = frozenset(
        {f"2025_{index:02d}_F{index:03d}_H{index:03d}"
         for index in range(1, 82)}
    )
    access = DevelopmentAccess(development, final)
    reads: list[Path] = []
    try:
        access.read_bytes(
            next(iter(final)),
            Path("/must-not-resolve/final.raw"),
            reader=lambda path: reads.append(path) or b"",
        )
    except DevelopmentMarketError:
        pass
    else:  # pragma: no cover - fixture contract
        raise AssertionError("final holdout access did not fail")
    if reads:
        raise AssertionError("final holdout reader was invoked")

    binding = ExpansionMoneylineBinding(
        game_id="2025_01_AAA_BBB",
        away_team="AAA",
        home_team="BBB",
        polymarket_market_id="100",
        polymarket_condition_id="0x" + "1" * 64,
        polymarket_tokens=(
            ("Alpha", "AAA", "token-aaa"),
            ("Beta", "BBB", "token-bbb"),
        ),
        kalshi_event_ticker="KXNFL-TEST",
        kalshi_team_tickers=(
            ("AAA", "KXNFL-TEST-AAA"),
            ("BBB", "KXNFL-TEST-BBB"),
        ),
        rule_sha256=_fixture_sha("fixture-rule"),
    )
    poly_rows = [
        {
            "transactionHash": "poly-pre",
            "timestamp": 1735689609,
            "price": Decimal("0.40"),
            "size": Decimal("2"),
            "conditionId": binding.polymarket_condition_id,
            "asset": "token-aaa",
            "outcome": "Alpha",
        },
        {
            "transactionHash": "poly-post-1",
            "timestamp": 1735689612,
            "price": Decimal("0.50"),
            "size": Decimal("2"),
            "conditionId": binding.polymarket_condition_id,
            "asset": "token-aaa",
            "outcome": "Alpha",
        },
        {
            "transactionHash": "poly-post-2",
            "timestamp": 1735689614,
            "price": Decimal("0.55"),
            "size": Decimal("1"),
            "conditionId": binding.polymarket_condition_id,
            "asset": "token-aaa",
            "outcome": "Alpha",
        },
        {
            "transactionHash": "poly-bbb-pre",
            "timestamp": 1735689609,
            "price": Decimal("0.60"),
            "size": Decimal("1"),
            "conditionId": binding.polymarket_condition_id,
            "asset": "token-bbb",
            "outcome": "Beta",
        },
    ]
    kalshi_rows = {
        "KXNFL-TEST-AAA": [
            {
                "trade_id": "kalshi-pre",
                "ticker": "KXNFL-TEST-AAA",
                "created_time": "2025-01-01T00:00:09.100000Z",
                "yes_price_dollars": "0.40",
                "no_price_dollars": "0.60",
                "is_block_trade": False,
                "taker_side": "yes",
                "taker_outcome_side": "yes",
                "count_fp": "1",
            },
            {
                "trade_id": "kalshi-post-a",
                "ticker": "KXNFL-TEST-AAA",
                "created_time": "2025-01-01T00:00:12.100000Z",
                "yes_price_dollars": "0.51",
                "no_price_dollars": "0.49",
                "is_block_trade": False,
                "taker_side": "yes",
                "taker_outcome_side": "yes",
                "count_fp": "1",
            },
            {
                "trade_id": "kalshi-post-b",
                "ticker": "KXNFL-TEST-AAA",
                "created_time": "2025-01-01T00:00:12.800000Z",
                "yes_price_dollars": "0.52",
                "no_price_dollars": "0.48",
                "is_block_trade": False,
                "taker_side": "no",
                "taker_outcome_side": "no",
                "count_fp": "1",
            },
        ],
        "KXNFL-TEST-BBB": [],
    }
    observations = normalize_actual_moneyline_trades(
        binding=binding,
        polymarket_rows=poly_rows,
        kalshi_rows_by_ticker=kalshi_rows,
    )
    market = actual_market_observation_frame(binding.game_id, observations)
    layer_m = build_layer_m_intervals(market)
    sports_hash = _fixture_sha("sports")
    sports = pd.DataFrame(
        [
            {
                "game_id": binding.game_id,
                "play_id": "1",
                "episode_id": "episode-1",
                "order_sequence": 1,
                "source_time_utc": "2025-01-01T00:00:10.250000Z",
                "event_eligible": True,
                "focal_team": "AAA",
                "quarter": 1,
                "penalty": True,
                "sports_source_hashes": (sports_hash,),
            },
            {
                "game_id": binding.game_id,
                "play_id": "2",
                "episode_id": "episode-2",
                "order_sequence": 2,
                "source_time_utc": "2025-01-01T00:00:17.000000Z",
                "event_eligible": True,
                "focal_team": "AAA",
                "quarter": 1,
                "penalty": False,
                "sports_source_hashes": (sports_hash,),
            },
        ]
    )
    layer_g = build_layer_g_timeline(sports)
    contracts = _moneyline_contract_inventory(binding)
    lineage = {
        row.observation_id: (_fixture_sha("market-" + row.native_id),)
        for row in observations
    }
    paths = build_reaction_paths(
        layer_g=layer_g,
        contracts=contracts,
        observations=observations,
        market_lineage=lineage,
    )
    delay_zero = paths[
        paths["delay_seconds"].eq(0)
        & paths["episode_id"].eq("episode-1")
    ]
    observed = delay_zero[
        delay_zero["venue"].eq("polymarket")
        & delay_zero["outcome"].eq("AAA")
        & delay_zero["horizon_seconds"].eq(1)
    ].iloc[0]
    if (
        observed["signed_price_change"] != Decimal("0.10")
        or observed["trade_count"] != 1
        or not bool(observed["actual_observation"])
    ):
        raise AssertionError(
            "observed cumulative reaction differs: "
            + repr(observed.to_dict())
        )
    missing = delay_zero[
        delay_zero["venue"].eq("polymarket")
        & delay_zero["outcome"].eq("BBB")
        & delay_zero["horizon_seconds"].eq(1)
    ].iloc[0]
    if (
        bool(missing["actual_observation"])
        or missing["first_post_event_trade"] is not None
        or bool(missing["forward_filled"])
    ):
        raise AssertionError("missing trade was forward-filled")
    ambiguous = delay_zero[
        delay_zero["venue"].eq("kalshi")
        & delay_zero["outcome"].eq("AAA")
        & delay_zero["horizon_seconds"].eq(1)
    ].iloc[0]
    if (
        not bool(ambiguous["order_ambiguous"])
        or bool(ambiguous["actual_observation"])
    ):
        raise AssertionError("same-second ambiguity was not preserved")
    contaminated = delay_zero[
        delay_zero["venue"].eq("polymarket")
        & delay_zero["outcome"].eq("AAA")
        & delay_zero["horizon_seconds"].eq(10)
    ].iloc[0]
    if not bool(contaminated["contaminated"]):
        raise AssertionError("next episode contamination was not detected")
    widths = {
        int(row.delay_seconds): (
            _parse_utc(row.interval_end_utc, field="fixture G end")
            - _parse_utc(row.interval_start_utc, field="fixture G start")
        ).total_seconds()
        for row in layer_g[
            layer_g["episode_id"].eq("episode-1")
        ].itertuples()
    }
    if widths != {delay: delay + 1 for delay in DELAY_SCENARIOS_SECONDS}:
        raise AssertionError("Layer G delay widths differ")
    formal_definition = {
        "factor_id": "NFL.ADMIN.PENALTY",
        "version": "v2",
        "sports_meaning": "Fixture penalty.",
        "predicate": [
            {"field": "penalty", "operator": "eq", "value": True}
        ],
        "pit_fields": ["penalty"],
        "focal_entity": "team",
        "exclusions": [],
        "target": "signed_price_change",
        "market_families": ["moneyline"],
        "horizons_seconds": list(HORIZONS_SECONDS),
        "support_kind": "named",
        "minimum_episodes": 1,
        "minimum_games": 1,
        "minimum_episodes_per_venue": 0,
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
    }
    factor_observations, factor_audit = (
        build_factor_observations_by_delay(
            sports_feature_view=sports,
            reaction_paths=paths,
            contract_inventory=contracts,
            formal_factor_definitions=(formal_definition,),
        )
    )
    if (
        factor_observations.empty
        or set(factor_observations["delay_seconds"])
        - set(DELAY_SCENARIOS_SECONDS)
        or set(factor_audit["delay_seconds"])
        != set(DELAY_SCENARIOS_SECONDS)
        or factor_observations[
            factor_observations["delay_seconds"].eq(0)
            & factor_observations["venue"].eq("polymarket")
            & factor_observations["outcome"].eq("AAA")
            & factor_observations["horizon_seconds"].eq(1)
        ].empty
    ):
        raise AssertionError("per-delay formal factor build differs")
    return {
        "status": "PASS",
        "development_guard_final_reader_calls": len(reads),
        "actual_market_observation_rows": len(market),
        "layer_m_rows": len(layer_m),
        "layer_g_rows": len(layer_g),
        "reaction_path_rows": len(paths),
        "delay_zero_reaction_rows": len(
            paths[paths["delay_seconds"].eq(0)]
        ),
        "factor_observation_rows": len(factor_observations),
        "factor_exclusion_audit_rows": len(factor_audit),
        "registered_horizons": list(HORIZONS_SECONDS),
        "delay_scenarios": list(DELAY_SCENARIOS_SECONDS),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def run_development_market_pipeline(
    *,
    unlock_manifest: Path,
    capture_batch: Path,
    sports_feature_batch: Path,
    factor_registry: Path,
    expected_unlock_sha256: str,
    expected_capture_batch_sha256: str,
    expected_sports_feature_batch_sha256: str,
    expected_factor_registry_sha256: str,
    raw_root: Path,
    output_root: Path,
    game_ids: Sequence[str] | None = None,
    delay_scenarios: Sequence[int] = (0,),
    compute_workers: int = COMPUTE_WORKERS,
) -> dict[str, object]:
    """Verify frozen inputs, run four workers, and publish exact checkpoints."""

    from prediction_market.sports.nfl_expansion_development_market_artifacts import (
        EXPECTED_CAPTURE_BATCH_SHA256,
        EXPECTED_FACTOR_REGISTRY_SHA256,
        EXPECTED_FEATURE_BATCH_SHA256,
        EXPECTED_UNLOCK_OBJECT_SHA256,
        publish_development_batch_index,
        verify_development_inputs,
    )

    expected = {
        "unlock": (
            expected_unlock_sha256,
            EXPECTED_UNLOCK_OBJECT_SHA256,
        ),
        "capture batch": (
            expected_capture_batch_sha256,
            EXPECTED_CAPTURE_BATCH_SHA256,
        ),
        "sports feature batch": (
            expected_sports_feature_batch_sha256,
            EXPECTED_FEATURE_BATCH_SHA256,
        ),
        "factor registry": (
            expected_factor_registry_sha256,
            EXPECTED_FACTOR_REGISTRY_SHA256,
        ),
    }
    for label, (observed, frozen) in expected.items():
        if observed != frozen:
            raise DevelopmentMarketError(
                f"{label} expected SHA-256 differs from frozen input"
            )
    if (
        type(compute_workers) is not int
        or not 1 <= compute_workers <= MAX_COMPUTE_WORKERS
    ):
        raise DevelopmentMarketError(
            "development market compute_workers must be between one and eight"
        )
    delays = tuple(int(value) for value in delay_scenarios)
    if (
        not delays
        or delays != tuple(sorted(set(delays)))
        or not set(delays) <= set(DELAY_SCENARIOS_SECONDS)
    ):
        raise DevelopmentMarketError("pipeline delays are not registered")
    project_root = Path(__file__).resolve().parents[3]
    unlock_object = (
        Path(unlock_manifest).resolve().parent.parent
        / "objects"
        / "sha256"
        / "bf"
        / (
            EXPECTED_UNLOCK_OBJECT_SHA256.removeprefix("sha256:")
            + ".json"
        )
    )
    inputs = verify_development_inputs(
        project_root=project_root,
        raw_root=Path(raw_root),
        unlock_object_path=unlock_object,
        unlock_manifest_path=Path(unlock_manifest),
        capture_batch_path=Path(capture_batch),
        feature_batch_path=Path(sports_feature_batch),
        factor_registry_path=Path(factor_registry),
    )
    selected = (
        tuple(sorted(inputs.unlock.development_game_ids))
        if game_ids is None
        else tuple(sorted(set(game_ids)))
    )
    if not selected:
        raise DevelopmentMarketError("no development games selected")
    for game_id in selected:
        inputs.require_development(game_id)
    published: list[object] = []
    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(
        max_workers=compute_workers,
        thread_name_prefix="nfl-dev-market",
    ) as executor:
        future_by_game = {
            executor.submit(
                publish_verified_single_game,
                inputs,
                output_root=Path(output_root),
                game_id=game_id,
                delay_scenarios=delays,
            ): game_id
            for game_id in selected
        }
        for future in as_completed(future_by_game):
            game_id = future_by_game[future]
            try:
                published.append(future.result())
            except BaseException as error:
                failures.append((game_id, error))
    if failures:
        game_id, error = sorted(failures, key=lambda item: item[0])[0]
        raise DevelopmentMarketError(
            f"{len(failures)} development games failed; "
            f"first={game_id}:{type(error).__name__}:{error}"
        ) from error
    ordered = tuple(sorted(published, key=lambda row: row.game_id))
    batch = None
    discovery = None
    if frozenset(selected) == inputs.unlock.development_game_ids:
        batch_fingerprint = _semantic_sha(
            {
                "schema": (
                    "nfl_expansion_development_market_batch_input_v1"
                ),
                "builder_version": BUILDER_VERSION,
                "delay_scenarios_seconds": list(delays),
                "game_bundles": [
                    {
                        "game_id": game.game_id,
                        "bundle_sha256": game.bundle_sha256,
                    }
                    for game in ordered
                ],
                "unlock_object_sha256": inputs.unlock.object_sha256,
                "capture_batch_sha256": inputs.capture_batch_sha256,
                "feature_batch_sha256": inputs.feature_batch_sha256,
                "factor_registry_sha256": inputs.factor_registry_sha256,
            }
        )
        batch = publish_development_batch_index(
            inputs,
            output_root=Path(output_root),
            games=ordered,
            input_fingerprint=batch_fingerprint,
            builder_version=BUILDER_VERSION,
            source_bindings={
                "unlock_object_sha256": inputs.unlock.object_sha256,
                "capture_batch_sha256": inputs.capture_batch_sha256,
                "feature_batch_sha256": inputs.feature_batch_sha256,
                "factor_registry_sha256": inputs.factor_registry_sha256,
            },
            claim_boundary=CLAIM_BOUNDARY,
        )
        if delays == (0,):
            discovery = publish_verified_cross_game_discovery(
                inputs,
                output_root=Path(output_root),
                batch=batch,
            )
    return {
        "status": "PASS",
        "builder_version": BUILDER_VERSION,
        "compute_workers": compute_workers,
        "delay_scenarios_seconds": list(delays),
        "selected_game_count": len(selected),
        "published_game_count": len(ordered),
        "game_manifests": [
            {
                "game_id": game.game_id,
                "bundle_sha256": game.bundle_sha256,
                "manifest_path": str(game.manifest_path),
                "manifest_sha256": game.manifest_sha256,
                "checkpoint_path": str(game.checkpoint_path),
            }
            for game in ordered
        ],
        "batch": (
            None
            if batch is None
            else {
                "batch_sha256": batch.batch_sha256,
                "index_path": str(batch.index_path),
                "index_sha256": batch.index_sha256,
                "game_count": batch.game_count,
                "merged_single_game_results_path": str(
                    batch.merged_single_game_results.object_path
                ),
                "merged_single_game_results_sha256": (
                    batch.merged_single_game_results.object_sha256
                ),
            }
        ),
        "cross_game_discovery": (
            None
            if discovery is None
            else {
                "object_path": str(discovery.object_path),
                "object_sha256": discovery.object_sha256,
                "row_count": discovery.row_count,
                "manifest_path": str(discovery.manifest_path),
                "manifest_sha256": discovery.manifest_sha256,
                "batch_sha256": discovery.batch_sha256,
            }
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "COMPUTE_WORKERS",
    "DELAY_SCENARIOS_SECONDS",
    "DISCOVERY_BOOTSTRAP_SAMPLES",
    "DISCOVERY_BOOTSTRAP_SEED",
    "DevelopmentAccess",
    "DevelopmentMarketError",
    "ExpansionMoneylineBinding",
    "HORIZONS_SECONDS",
    "MAX_COMPUTE_WORKERS",
    "actual_market_observation_frame",
    "build_cross_game_discovery",
    "build_factor_observations_by_delay",
    "build_layer_g_timeline",
    "build_layer_m_intervals",
    "build_reaction_paths",
    "normalize_actual_moneyline_trades",
    "publish_verified_cross_game_discovery",
    "run_bounded_fixture",
    "run_development_market_pipeline",
]
