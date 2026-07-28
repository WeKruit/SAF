"""Reaction-closed raw capture for the 2025 NFL moneyline expansion.

The capture layer deliberately stops before market-reaction research.  Exact
upstream response bytes are preserved, but response validation and published
indexes inspect only stable identity, source timestamp, cursor termination,
and row counts.  Price, size, side, direction, and derived reaction values are
not read into the capture index.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import httpx

from prediction_market.kalshi_history import KALSHI_API_ROOT
from prediction_market.polymarket_history import DATA_API_TRADES_URL
from prediction_market.sports.nfl_x13_capture import HostRateLimiter
from prediction_market.static_store import (
    StaticStoreError,
    preserve_static_object,
    read_verified_static_object,
)


CAPTURE_SCOPE = "moneyline-only"
REACTION_ACCESS = "CLOSED"
BUILDER_VERSION = "nfl-historical-moneyline-capture-v1"
EXPECTED_GAME_COUNT = 234
NETWORK_WORKERS = 2
POLYMARKET_TRADE_LIMIT = 10_000
KALSHI_PAGE_LIMIT = 1_000
DEFAULT_MAX_KALSHI_PAGES = 100
DEFAULT_MAX_RESPONSE_BYTES = 50_000_000
DEFAULT_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_HOST_REQUESTS_PER_SECOND: Mapping[str, int | float] = {
    "gamma-api.polymarket.com": 4,
    "data-api.polymarket.com": 4,
    "external-api.kalshi.com": 4,
}
PINNED_CANDIDATE_SCAN_SHA256 = (
    "sha256:9fb96bfa10f641f410eda001803aa28dc3908ee5f957f196800b9e8adcddeaef"
)
PINNED_TEMPORAL_AUDIT_SHA256 = (
    "sha256:432277924ccc8249333b48b96060662cf630bf1a72616e79d9e186bdb2de1588"
)
PINNED_GOVERNANCE_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)

GAMMA_EVENT_SLUG_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"
KALSHI_HISTORICAL_MARKETS_URL = f"{KALSHI_API_ROOT}/historical/markets"
KALSHI_HISTORICAL_TRADES_URL = f"{KALSHI_API_ROOT}/historical/trades"

_RAW_VERSION = "nfl2025-moneyline-expansion-v1"
_SCHEMA_FINGERPRINT = (
    "sha256:"
    + hashlib.sha256(b"nfl-moneyline-opaque-json-response-v1").hexdigest()
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"2025_[0-9]{2}_[A-Z0-9]+_[A-Z0-9]+\Z")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,127}\Z")
_CONDITION_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,127}\Z")
_REQUEST_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "identity",
}


class HistoricalMoneylineCaptureError(RuntimeError):
    """A capture input, response, cache, or publication failed closed."""


class TransientMoneylineCaptureError(HistoricalMoneylineCaptureError):
    """A retryable public HTTP failure."""


@dataclass(frozen=True, slots=True)
class CaptureHttpResponse:
    """Exact HTTP response bytes plus non-authoritative response headers."""

    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """Canonical request identity for one byte-exact raw response."""

    game_id: str
    source: str
    resource: str
    scope_id: str
    url: str
    params: tuple[tuple[str, object], ...]
    source_cursor: str
    coverage: str

    @property
    def request_sha256(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "coverage": self.coverage,
                    "game_id": self.game_id,
                    "params": dict(self.params),
                    "resource": self.resource,
                    "scope_id": self.scope_id,
                    "source": self.source,
                    "source_cursor": self.source_cursor,
                    "url": self.url,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class MoneylineGamePlan:
    """Exact cross-source identities and inclusive source-time window."""

    game_id: str
    away_team: str
    home_team: str
    polymarket_event_id: str
    polymarket_event_slug: str
    polymarket_condition_id: str
    polymarket_market_id: str
    kalshi_event_ticker: str
    kalshi_team_tickers: tuple[str, str]
    start_ts: int
    end_ts: int

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "away_team": self.away_team,
            "end_ts": self.end_ts,
            "game_id": self.game_id,
            "home_team": self.home_team,
            "kalshi_event_ticker": self.kalshi_event_ticker,
            "kalshi_team_tickers": list(self.kalshi_team_tickers),
            "polymarket_condition_id": self.polymarket_condition_id,
            "polymarket_event_id": self.polymarket_event_id,
            "polymarket_event_slug": self.polymarket_event_slug,
            "polymarket_market_id": self.polymarket_market_id,
            "start_ts": self.start_ts,
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.canonical_payload))


@dataclass(frozen=True, slots=True)
class MoneylineExpansionSpec:
    """Verified reaction-blind inputs for the exact expansion cohort."""

    candidate_scan_path: Path
    temporal_audit_path: Path
    governance_path: Path
    candidate_scan_sha256: str
    temporal_audit_sha256: str
    governance_sha256: str
    game_assignments: tuple[tuple[str, str, int], ...]
    plans: tuple[MoneylineGamePlan, ...]

    @property
    def batch_plan_sha256(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "candidate_scan_sha256": self.candidate_scan_sha256,
                    "capture_scope": CAPTURE_SCOPE,
                    "game_assignments": [
                        {
                            "cohort": cohort,
                            "game_id": game_id,
                            "week": week,
                        }
                        for game_id, cohort, week in self.game_assignments
                    ],
                    "game_plans": [
                        plan.canonical_payload for plan in self.plans
                    ],
                    "governance_sha256": self.governance_sha256,
                    "reaction_access": REACTION_ACCESS,
                    "temporal_audit_sha256": self.temporal_audit_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class GameCaptureIndexRef:
    """Verified content-addressed index for one completed game."""

    game_id: str
    index_path: Path
    index_sha256: str
    checkpoint_path: Path
    reaction_access: str
    polymarket_trade_count: int
    kalshi_trade_count: int
    raw_manifest_count: int


@dataclass(frozen=True, slots=True)
class BatchCaptureManifestRef:
    """Final batch publication, created only when every game is complete."""

    manifest_path: Path
    manifest_sha256: str
    pointer_path: Path
    game_count: int
    resumed_game_count: int
    captured_game_count: int


@dataclass(frozen=True, slots=True)
class _RawResponseRef:
    request_sha256: str
    object_sha256: str
    manifest_sha256: str
    byte_length: int
    object_path: str
    manifest_path: str

    def payload(self) -> dict[str, object]:
        return {
            "byte_length": self.byte_length,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "object_path": self.object_path,
            "object_sha256": self.object_sha256,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True, slots=True)
class _VerifiedRawResponse:
    ref: _RawResponseRef
    object_bytes: bytes
    record: Any


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HistoricalMoneylineCaptureError(
            "value is not canonical JSON"
        ) from error
    return encoded + (b"\n" if newline else b"")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _strict_json(payload: bytes, *, context: str) -> Any:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise HistoricalMoneylineCaptureError(
                    f"{context} contains duplicate JSON key {key!r}"
                )
            output[key] = value
        return output

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HistoricalMoneylineCaptureError(
                    f"{context} contains non-finite JSON value {token}"
                )
            ),
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        HistoricalMoneylineCaptureError,
    ) as error:
        if isinstance(error, HistoricalMoneylineCaptureError):
            raise
        raise HistoricalMoneylineCaptureError(
            f"{context} is not strict UTF-8 JSON"
        ) from error


def _read_json_file(path: Path, *, context: str) -> tuple[bytes, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalMoneylineCaptureError(
            f"{context} must be a regular file"
        )
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise HistoricalMoneylineCaptureError(
            f"{context} is unreadable"
        ) from error
    return payload, _strict_json(payload, context=context)


def _parse_utc(value: object, *, field: str) -> datetime:
    if type(value) is not str or not value:
        raise HistoricalMoneylineCaptureError(
            f"{field} must be a nonempty UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HistoricalMoneylineCaptureError(
            f"{field} is malformed"
        ) from error
    if parsed.tzinfo is None:
        raise HistoricalMoneylineCaptureError(
            f"{field} must carry a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _validate_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise HistoricalMoneylineCaptureError(
            f"{field} must be a canonical SHA-256"
        )
    return value


def _validate_plan(plan: MoneylineGamePlan) -> None:
    if not isinstance(plan, MoneylineGamePlan):
        raise HistoricalMoneylineCaptureError(
            "game plan has the wrong type"
        )
    if _GAME_ID_RE.fullmatch(plan.game_id) is None:
        raise HistoricalMoneylineCaptureError("game_id is not canonical")
    for field, value in (
        ("away_team", plan.away_team),
        ("home_team", plan.home_team),
    ):
        if (
            type(value) is not str
            or not value
            or value != value.upper()
            or not value.isalnum()
        ):
            raise HistoricalMoneylineCaptureError(
                f"{field} is not canonical"
            )
    if (
        type(plan.polymarket_event_id) is not str
        or not plan.polymarket_event_id.isdigit()
        or _SLUG_RE.fullmatch(plan.polymarket_event_slug) is None
        or _CONDITION_RE.fullmatch(plan.polymarket_condition_id) is None
        or type(plan.polymarket_market_id) is not str
        or not plan.polymarket_market_id.isdigit()
    ):
        raise HistoricalMoneylineCaptureError(
            "Polymarket identity is malformed"
        )
    if (
        _TICKER_RE.fullmatch(plan.kalshi_event_ticker) is None
        or type(plan.kalshi_team_tickers) is not tuple
        or len(plan.kalshi_team_tickers) != 2
        or len(set(plan.kalshi_team_tickers)) != 2
        or any(
            _TICKER_RE.fullmatch(ticker) is None
            or not ticker.startswith(plan.kalshi_event_ticker + "-")
            for ticker in plan.kalshi_team_tickers
        )
    ):
        raise HistoricalMoneylineCaptureError(
            "Kalshi identity is malformed"
        )
    if (
        type(plan.start_ts) is not int
        or type(plan.end_ts) is not int
        or plan.start_ts < 0
        or plan.end_ts < plan.start_ts
    ):
        raise HistoricalMoneylineCaptureError(
            "capture window is malformed"
        )


def _governance_assignments(
    value: object,
    *,
    candidate_sha256: str,
    temporal_sha256: str,
    candidate_weeks: Mapping[str, int],
) -> tuple[tuple[str, str, int], ...]:
    if (
        type(value) is not dict
        or value.get("schema") != "nfl_factor_expansion_registry_v2"
        or value.get("experiment_id") != "X-13"
        or value.get("status") != "FROZEN"
    ):
        raise HistoricalMoneylineCaptureError(
            "governance object is not the frozen X-13 expansion registry"
        )
    scope = value.get("analysis_scope")
    boundary = value.get("claim_boundary")
    sources = value.get("source_bindings")
    split = value.get("split_lock")
    legacy = value.get("legacy_cohort")
    if (
        type(scope) is not dict
        or scope.get("authorized_market_families") != ["moneyline"]
        or scope.get("market_policy") != "MONEYLINE_FIRST"
        or scope.get("market_value_or_direction_accessed") is not False
        or scope.get("market_value_or_direction_emitted") is not False
        or type(boundary) is not dict
        or boundary.get("candidate_selection_reaction_blind") is not True
        or boundary.get("development_reaction_access") is not False
        or boundary.get("final_holdout_reaction_access") is not False
        or boundary.get("market_value_or_direction_output") is not False
        or boundary.get("market_value_or_direction_read") is not False
        or type(sources) is not dict
        or type(sources.get("candidate_scan")) is not dict
        or sources["candidate_scan"].get("sha256") != candidate_sha256
        or type(sources.get("temporal_audit_object")) is not dict
        or sources["temporal_audit_object"].get("sha256") != temporal_sha256
        or type(split) is not dict
        or split.get("method") != "OUT_OF_TIME_BY_WEEK"
        or type(legacy) is not dict
        or legacy.get("game_count") != 40
        or legacy.get("mutually_exclusive_with_expansion") is not True
        or legacy.get("status") != "LEGACY_CONSUMED"
    ):
        raise HistoricalMoneylineCaptureError(
            "governance scope or reaction-closed boundary differs"
        )
    development = split.get("development")
    final_holdout = split.get("final_holdout")
    if (
        type(development) is not dict
        or development.get("access_state") != REACTION_ACCESS
        or development.get("reaction_access") is not False
        or development.get("game_count") != 153
        or development.get("weeks") != list(range(1, 13))
        or type(final_holdout) is not dict
        or final_holdout.get("access_state") != REACTION_ACCESS
        or final_holdout.get("reaction_access") is not False
        or final_holdout.get("game_count") != 81
        or final_holdout.get("weeks") != list(range(13, 19))
        or type(split.get("game_assignments")) is not list
    ):
        raise HistoricalMoneylineCaptureError(
            "governance cohort scope or access state differs"
        )
    assignments: dict[str, tuple[str, int]] = {}
    for row in split["game_assignments"]:
        if (
            type(row) is not dict
            or type(row.get("game_id")) is not str
            or row.get("cohort") not in {"development", "final_holdout"}
            or type(row.get("week")) is not int
        ):
            raise HistoricalMoneylineCaptureError(
                "governance game assignment is malformed"
            )
        game_id = str(row["game_id"])
        if game_id in assignments:
            raise HistoricalMoneylineCaptureError(
                "governance game assignment is duplicated"
            )
        week = int(row["week"])
        expected_cohort = "development" if week <= 12 else "final_holdout"
        if (
            candidate_weeks.get(game_id) != week
            or row["cohort"] != expected_cohort
        ):
            raise HistoricalMoneylineCaptureError(
                "governance game assignment differs from frozen candidate"
            )
        assignments[game_id] = (str(row["cohort"]), week)
    legacy_ids = {
        str(game_id)
        for key in ("discovery_game_ids", "holdout_game_ids")
        for game_id in legacy.get(key, [])
    }
    if (
        len(assignments) != EXPECTED_GAME_COUNT
        or set(assignments) != set(candidate_weeks)
        or len(legacy_ids) != 40
        or legacy_ids.intersection(assignments)
    ):
        raise HistoricalMoneylineCaptureError(
            "governance does not bind exact 234 mutually exclusive games"
        )
    return tuple(
        (game_id, assignments[game_id][0], assignments[game_id][1])
        for game_id in sorted(assignments)
    )


def load_moneyline_expansion_spec(
    candidate_scan_path: str | Path,
    temporal_audit_path: str | Path,
    governance_path: str | Path,
) -> MoneylineExpansionSpec:
    """Load and cross-check the reaction-blind identity and time inputs."""

    candidate_path = Path(candidate_scan_path).resolve()
    temporal_path = Path(temporal_audit_path).resolve()
    governance_file = Path(governance_path).resolve()
    candidate_bytes, candidate_value = _read_json_file(
        candidate_path,
        context="candidate scan",
    )
    temporal_bytes, temporal_value = _read_json_file(
        temporal_path,
        context="temporal audit",
    )
    governance_bytes, governance_value = _read_json_file(
        governance_file,
        context="expansion governance",
    )
    candidate_sha = _sha256(candidate_bytes)
    temporal_sha = _sha256(temporal_bytes)
    governance_sha = _sha256(governance_bytes)
    if (
        candidate_sha != PINNED_CANDIDATE_SCAN_SHA256
        or temporal_sha != PINNED_TEMPORAL_AUDIT_SHA256
        or governance_sha != PINNED_GOVERNANCE_SHA256
    ):
        raise HistoricalMoneylineCaptureError(
            "formal expansion input hash differs from the frozen pins"
        )
    if (
        type(candidate_value) is not dict
        or candidate_value.get("schema")
        != "nfl_2025_dual_venue_metadata_candidates_v0"
        or candidate_value.get("selection_basis")
        != "metadata_only_no_score_price_volume_or_reaction"
        or type(candidate_value.get("games")) is not list
    ):
        raise HistoricalMoneylineCaptureError(
            "candidate scan is not the reaction-blind identity artifact"
        )
    if (
        type(temporal_value) is not dict
        or temporal_value.get("schema")
        != "nfl_2025_candidate_temporal_coverage_v1"
        or temporal_value.get("scope") != "all-dual"
        or temporal_value.get("selection_basis")
        != (
            "source_timestamp_and_trade_presence_only;"
            "no_score_price_size_volume_or_direction_retained"
        )
        or temporal_value.get("candidate_scan_sha256") != candidate_sha
        or type(temporal_value.get("games")) is not list
    ):
        raise HistoricalMoneylineCaptureError(
            "temporal audit is not the linked reaction-blind all-dual artifact"
        )

    candidate_games: dict[str, Mapping[str, object]] = {}
    for row in candidate_value["games"]:
        if (
            type(row) is not dict
            or row.get("dual_venue_metadata") is not True
        ):
            continue
        game_id = row.get("game_id")
        if type(game_id) is not str or game_id in candidate_games:
            raise HistoricalMoneylineCaptureError(
                "candidate scan contains malformed or duplicate game identity"
            )
        candidate_games[game_id] = row
    temporal_games: dict[str, Mapping[str, object]] = {}
    for row in temporal_value["games"]:
        if type(row) is not dict or type(row.get("game_id")) is not str:
            raise HistoricalMoneylineCaptureError(
                "temporal audit contains malformed game identity"
            )
        game_id = str(row["game_id"])
        if game_id in temporal_games:
            raise HistoricalMoneylineCaptureError(
                "temporal audit contains duplicate game identity"
            )
        temporal_games[game_id] = row
    if (
        len(candidate_games) != EXPECTED_GAME_COUNT
        or len(temporal_games) != EXPECTED_GAME_COUNT
        or candidate_value.get("dual_venue_metadata_count")
        != EXPECTED_GAME_COUNT
        or temporal_value.get("candidate_game_count")
        != EXPECTED_GAME_COUNT
        or set(candidate_games) != set(temporal_games)
    ):
        raise HistoricalMoneylineCaptureError(
            "candidate scan and temporal audit do not bind the exact cohort"
        )

    candidate_weeks: dict[str, int] = {}
    for game_id, row in candidate_games.items():
        if type(row.get("week")) is not int:
            raise HistoricalMoneylineCaptureError(
                f"{game_id} candidate week is malformed"
            )
        candidate_weeks[game_id] = int(row["week"])
    game_assignments = _governance_assignments(
        governance_value,
        candidate_sha256=candidate_sha,
        temporal_sha256=temporal_sha,
        candidate_weeks=candidate_weeks,
    )
    plans: list[MoneylineGamePlan] = []
    for game_id in sorted(candidate_games):
        candidate = candidate_games[game_id]
        temporal = temporal_games[game_id]
        poly = candidate.get("polymarket_identity")
        kalshi = candidate.get("kalshi_identity")
        if type(poly) is not dict or type(kalshi) is not dict:
            raise HistoricalMoneylineCaptureError(
                f"{game_id} lacks exact venue identity"
            )
        event_ticker = candidate.get("kalshi_event_ticker")
        market_refs = kalshi.get("market_refs")
        if (
            kalshi.get("event_ticker") != event_ticker
            or type(market_refs) is not list
            or len(market_refs) != 2
        ):
            raise HistoricalMoneylineCaptureError(
                f"{game_id} Kalshi identity is inconsistent"
            )
        first = _parse_utc(
            temporal.get("first_event_utc"),
            field=f"{game_id}.first_event_utc",
        )
        last = _parse_utc(
            temporal.get("last_event_utc"),
            field=f"{game_id}.last_event_utc",
        )
        plan = MoneylineGamePlan(
            game_id=game_id,
            away_team=str(candidate.get("away_team")),
            home_team=str(candidate.get("home_team")),
            polymarket_event_id=str(poly.get("event_id")),
            polymarket_event_slug=str(candidate.get("polymarket_event_slug")),
            polymarket_condition_id=str(
                poly.get("moneyline_condition_id")
            ),
            polymarket_market_id=str(poly.get("moneyline_market_id")),
            kalshi_event_ticker=str(event_ticker),
            kalshi_team_tickers=tuple(sorted(str(item) for item in market_refs)),
            start_ts=int(first.timestamp()) - 300,
            end_ts=int(last.timestamp()) + 300,
        )
        _validate_plan(plan)
        plans.append(plan)
    return MoneylineExpansionSpec(
        candidate_scan_path=candidate_path,
        temporal_audit_path=temporal_path,
        governance_path=governance_file,
        candidate_scan_sha256=candidate_sha,
        temporal_audit_sha256=temporal_sha,
        governance_sha256=governance_sha,
        game_assignments=game_assignments,
        plans=tuple(plans),
    )


def _request(
    *,
    plan: MoneylineGamePlan,
    source: str,
    resource: str,
    scope_id: str,
    url: str,
    params: Mapping[str, object],
    source_cursor: str,
    coverage: str,
) -> CaptureRequest:
    return CaptureRequest(
        game_id=plan.game_id,
        source=source,
        resource=resource,
        scope_id=scope_id,
        url=url,
        params=tuple(sorted(params.items())),
        source_cursor=source_cursor,
        coverage=coverage,
    )


def _utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise HistoricalMoneylineCaptureError(
                "content-addressed publication path is unsafe"
            )
        if path.read_bytes() != payload:
            raise HistoricalMoneylineCaptureError(
                "content-addressed publication collision"
            )
        return
    _atomic_replace(path, payload)


def _default_http_requester(
    client: httpx.Client,
    *,
    max_response_bytes: int,
) -> Callable[[CaptureRequest], CaptureHttpResponse]:
    def fetch(request: CaptureRequest) -> CaptureHttpResponse:
        try:
            with client.stream(
                "GET",
                request.url,
                params=dict(request.params),
                headers=_REQUEST_HEADERS,
            ) as response:
                if (
                    response.status_code in {408, 425, 429}
                    or 500 <= response.status_code <= 599
                ):
                    raise TransientMoneylineCaptureError(
                        f"retryable HTTP status {response.status_code}"
                    )
                response.raise_for_status()
                if response.headers.get("Content-Encoding") not in {
                    None,
                    "identity",
                }:
                    raise HistoricalMoneylineCaptureError(
                        "capture response is content-encoded"
                    )
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_length = int(declared)
                    except ValueError as error:
                        raise HistoricalMoneylineCaptureError(
                            "invalid Content-Length"
                        ) from error
                    if declared_length > max_response_bytes:
                        raise HistoricalMoneylineCaptureError(
                            "response exceeds max_response_bytes"
                        )
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_bytes():
                    received += len(chunk)
                    if received > max_response_bytes:
                        raise HistoricalMoneylineCaptureError(
                            "response exceeds max_response_bytes"
                        )
                    chunks.append(chunk)
                if declared is not None and received != int(declared):
                    raise TransientMoneylineCaptureError(
                        "response Content-Length mismatch"
                    )
                return CaptureHttpResponse(
                    body=b"".join(chunks),
                    headers=tuple(response.headers.multi_items()),
                )
        except httpx.HTTPStatusError as error:
            raise HistoricalMoneylineCaptureError(
                f"public request failed: {error}"
            ) from error
        except httpx.RequestError as error:
            raise TransientMoneylineCaptureError(
                f"public request temporarily failed: {error}"
            ) from error

    return fetch


def _validate_runtime_options(
    *,
    max_response_bytes: int,
    max_kalshi_pages: int,
    retry_delays_seconds: Sequence[int | float],
) -> tuple[float, ...]:
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise HistoricalMoneylineCaptureError(
            "max_response_bytes must be positive"
        )
    if type(max_kalshi_pages) is not int or max_kalshi_pages <= 0:
        raise HistoricalMoneylineCaptureError(
            "max_kalshi_pages must be positive"
        )
    if isinstance(retry_delays_seconds, (str, bytes, bytearray)):
        raise HistoricalMoneylineCaptureError(
            "retry_delays_seconds must be numeric"
        )
    delays = tuple(float(delay) for delay in retry_delays_seconds)
    if any(
        not math.isfinite(delay) or delay < 0 or delay > 60
        for delay in delays
    ):
        raise HistoricalMoneylineCaptureError(
            "retry delay must be between zero and 60 seconds"
        )
    return delays


def _raw_partition(request: CaptureRequest) -> str:
    return "nfl2025ml-" + request.request_sha256.removeprefix("sha256:")


def _raw_manifest_directory(
    raw_root: Path,
    request: CaptureRequest,
) -> Path:
    dataset = (
        "DS-POLYMARKET-PUBLIC"
        if request.source == "polymarket"
        else "DS-KALSHI-HISTORICAL"
    )
    return (
        raw_root
        / "manifests"
        / f"source={request.source}"
        / f"dataset={dataset}"
        / f"version={_RAW_VERSION}"
        / f"partition={_raw_partition(request)}"
    )


def _raw_ref(
    request: CaptureRequest,
    *,
    raw_root: Path,
    record: Any,
) -> _RawResponseRef:
    try:
        object_path = record.object_path.relative_to(raw_root).as_posix()
        manifest_path = record.manifest_path.relative_to(raw_root).as_posix()
    except ValueError as error:
        raise HistoricalMoneylineCaptureError(
            "raw record escapes raw_root"
        ) from error
    return _RawResponseRef(
        request_sha256=request.request_sha256,
        object_sha256=record.manifest.object_sha256,
        manifest_sha256=record.manifest.manifest_sha256,
        byte_length=record.manifest.byte_length,
        object_path=object_path,
        manifest_path=manifest_path,
    )


def _load_cached_response(
    request: CaptureRequest,
    *,
    raw_root: Path,
    program_root: Path,
) -> tuple[bytes, _RawResponseRef] | None:
    directory = _raw_manifest_directory(raw_root, request)
    manifests = sorted(directory.glob("*.manifest.json"))
    if not manifests:
        return None
    if len(manifests) != 1:
        raise HistoricalMoneylineCaptureError(
            "verified cache contains multiple manifests for one request"
        )
    try:
        verified = read_verified_static_object(
            manifests[0],
            store_root=raw_root,
            program_root=program_root,
        )
    except StaticStoreError as error:
        raise HistoricalMoneylineCaptureError(
            "verified cache failed byte/hash validation"
        ) from error
    manifest = verified.record.manifest
    source_request = manifest.source_request
    if (
        manifest.source_url != request.url
        or source_request.get("method") != "GET"
        or source_request.get("params") != dict(request.params)
        or source_request.get("headers") != _REQUEST_HEADERS
        or manifest.source_cursor != request.source_cursor
        or manifest.coverage != request.coverage
    ):
        raise HistoricalMoneylineCaptureError(
            "verified cache request identity differs"
        )
    return (
        verified.object_bytes,
        _raw_ref(request, raw_root=raw_root, record=verified.record),
    )


def _preserve_response(
    request: CaptureRequest,
    response: CaptureHttpResponse,
    *,
    raw_root: Path,
    program_root: Path,
) -> _RawResponseRef:
    if type(response.body) is not bytes or type(response.headers) is not tuple:
        raise HistoricalMoneylineCaptureError(
            "requester must return exact bytes and header tuples"
        )
    headers = httpx.Headers(response.headers)
    dataset = (
        "DS-POLYMARKET-PUBLIC"
        if request.source == "polymarket"
        else "DS-KALSHI-HISTORICAL"
    )
    license_ref = "O-001" if request.source == "polymarket" else "O-003"
    try:
        record = preserve_static_object(
            raw_root,
            response.body,
            program_root=program_root,
            source=request.source,
            dataset=dataset,
            version=_RAW_VERSION,
            partition=_raw_partition(request),
            extension="json",
            source_url=request.url,
            source_request={
                "method": "GET",
                "headers": dict(_REQUEST_HEADERS),
                "params": dict(request.params),
            },
            source_cursor=request.source_cursor,
            fetched_at=_utc_now_text(),
            coverage=request.coverage,
            etag=headers.get("ETag"),
            last_modified=headers.get("Last-Modified"),
            media_type=headers.get("Content-Type", "application/json").split(
                ";", 1
            )[0],
            schema_fingerprint=_SCHEMA_FINGERPRINT,
            license_ref=license_ref,
            license_status="research_only",
            upstream_partition=_raw_partition(request),
            object_kind="byte_exact_original",
            lineage={"query_sha256": None, "source_object_refs": []},
        )
    except StaticStoreError as error:
        raise HistoricalMoneylineCaptureError(
            "byte-exact raw preservation failed"
        ) from error
    return _raw_ref(request, raw_root=raw_root, record=record)


def _exchange(
    request: CaptureRequest,
    *,
    raw_root: Path,
    program_root: Path,
    requester: Callable[[CaptureRequest], CaptureHttpResponse],
    limiter: HostRateLimiter,
    retry_delays: tuple[float, ...],
) -> tuple[Any, _RawResponseRef]:
    cached = _load_cached_response(
        request,
        raw_root=raw_root,
        program_root=program_root,
    )
    if cached is None:
        response: CaptureHttpResponse | None = None
        for attempt in range(len(retry_delays) + 1):
            limiter.wait_for_url(request.url)
            try:
                response = requester(request)
                break
            except TransientMoneylineCaptureError as error:
                if attempt == len(retry_delays):
                    raise HistoricalMoneylineCaptureError(
                        "capture retry budget exhausted for "
                        f"{request.game_id}:{request.resource}"
                    ) from error
                time.sleep(retry_delays[attempt])
        if response is None:  # pragma: no cover - loop always returns or raises
            raise AssertionError("unreachable retry state")
        raw_ref = _preserve_response(
            request,
            response,
            raw_root=raw_root,
            program_root=program_root,
        )
        payload = response.body
    else:
        payload, raw_ref = cached
    return (
        _strict_json(
            payload,
            context=f"{request.game_id}:{request.resource}",
        ),
        raw_ref,
    )


def _capture_gamma_metadata(
    plan: MoneylineGamePlan,
    *,
    exchange: Callable[[CaptureRequest], tuple[Any, _RawResponseRef]],
) -> tuple[dict[str, object], _RawResponseRef]:
    request = _request(
        plan=plan,
        source="polymarket",
        resource="gamma_event",
        scope_id=plan.polymarket_event_id,
        url=GAMMA_EVENT_SLUG_URL.format(
            slug=plan.polymarket_event_slug
        ),
        params={},
        source_cursor=f"event_slug:{plan.polymarket_event_slug}",
        coverage=(
            "exact Gamma event metadata and frozen moneyline identity;"
            f"game={plan.game_id}"
        ),
    )
    value, raw_ref = exchange(request)
    if (
        type(value) is not dict
        or value.get("id") != plan.polymarket_event_id
        or value.get("slug") != plan.polymarket_event_slug
        or type(value.get("markets")) is not list
    ):
        raise HistoricalMoneylineCaptureError(
            "Gamma event identity does not match the frozen candidate"
        )
    matches = 0
    for market in value["markets"]:
        if type(market) is not dict:
            raise HistoricalMoneylineCaptureError(
                "Gamma market metadata row is malformed"
            )
        if (
            market.get("id") == plan.polymarket_market_id
            and market.get("conditionId") == plan.polymarket_condition_id
            and market.get("sportsMarketType") == "moneyline"
        ):
            matches += 1
    if matches != 1:
        raise HistoricalMoneylineCaptureError(
            "Gamma moneyline identity is missing or duplicated"
        )
    return (
        {
            "event_count": 1,
            "moneyline_market_count": 1,
            "request_sha256": request.request_sha256,
            "raw_manifest_sha256": raw_ref.manifest_sha256,
        },
        raw_ref,
    )


def _capture_kalshi_event_markets(
    plan: MoneylineGamePlan,
    *,
    exchange: Callable[[CaptureRequest], tuple[Any, _RawResponseRef]],
    max_pages: int,
) -> tuple[dict[str, object], list[_RawResponseRef]]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    observed_tickers: set[str] = set()
    raw_refs: list[_RawResponseRef] = []
    for page_index in range(max_pages):
        params: dict[str, object] = {
            "event_ticker": plan.kalshi_event_ticker,
            "limit": KALSHI_PAGE_LIMIT,
        }
        if cursor is not None:
            params["cursor"] = cursor
        request = _request(
            plan=plan,
            source="kalshi",
            resource="kalshi_event_markets",
            scope_id=plan.kalshi_event_ticker,
            url=KALSHI_HISTORICAL_MARKETS_URL,
            params=params,
            source_cursor=f"request_cursor:{cursor or 'ROOT'}",
            coverage=(
                "exact Kalshi historical event markets cursor chain;"
                f"event={plan.kalshi_event_ticker}"
            ),
        )
        value, raw_ref = exchange(request)
        raw_refs.append(raw_ref)
        if type(value) is not dict or set(value) != {"markets", "cursor"}:
            raise HistoricalMoneylineCaptureError(
                "Kalshi event markets page is malformed"
            )
        rows = value["markets"]
        next_cursor = value["cursor"]
        if type(rows) is not list or type(next_cursor) is not str:
            raise HistoricalMoneylineCaptureError(
                "Kalshi event markets cursor types are malformed"
            )
        for row in rows:
            if (
                type(row) is not dict
                or row.get("event_ticker") != plan.kalshi_event_ticker
                or type(row.get("ticker")) is not str
            ):
                raise HistoricalMoneylineCaptureError(
                    "Kalshi market escaped exact event identity"
                )
            observed_tickers.add(str(row["ticker"]))
        if next_cursor == "":
            if observed_tickers != set(plan.kalshi_team_tickers):
                raise HistoricalMoneylineCaptureError(
                    "Kalshi moneyline ticker inventory does not match"
                )
            return (
                {
                    "market_count": len(observed_tickers),
                    "page_count": page_index + 1,
                    "terminal_cursor": "",
                    "terminal_proof": "explicit_empty_cursor",
                },
                raw_refs,
            )
        if next_cursor == cursor or next_cursor in seen_cursors:
            raise HistoricalMoneylineCaptureError(
                "Kalshi event markets cursor repeated"
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise HistoricalMoneylineCaptureError(
        "Kalshi event markets terminal cursor proof is missing"
    )


def _capture_polymarket_window(
    plan: MoneylineGamePlan,
    left: int,
    right: int,
    *,
    exchange: Callable[[CaptureRequest], tuple[Any, _RawResponseRef]],
) -> tuple[list[dict[str, object]], list[_RawResponseRef], int]:
    request = _request(
        plan=plan,
        source="polymarket",
        resource="polymarket_moneyline_trades",
        scope_id=plan.polymarket_condition_id,
        url=DATA_API_TRADES_URL,
        params={
            "market": plan.polymarket_condition_id,
            "start": left,
            "end": right,
            "limit": POLYMARKET_TRADE_LIMIT,
            "offset": 0,
        },
        source_cursor=f"start:{left};end:{right};offset:0",
        coverage=(
            "moneyline inclusive trade window;"
            f"condition={plan.polymarket_condition_id};"
            f"start={left};end={right}"
        ),
    )
    value, raw_ref = exchange(request)
    if type(value) is not list or len(value) > POLYMARKET_TRADE_LIMIT:
        raise HistoricalMoneylineCaptureError(
            "Polymarket trades response is malformed"
        )
    for row in value:
        if (
            type(row) is not dict
            or row.get("conditionId") != plan.polymarket_condition_id
            or type(row.get("timestamp")) is not int
            or not left <= int(row["timestamp"]) <= right
        ):
            raise HistoricalMoneylineCaptureError(
                "Polymarket trade escaped identity or time scope"
            )
    if len(value) < POLYMARKET_TRADE_LIMIT:
        return (
            [
                {
                    "end_ts": right,
                    "item_count": len(value),
                    "raw_manifest_sha256": raw_ref.manifest_sha256,
                    "start_ts": left,
                    "terminal_proof": "unsaturated_time_window",
                }
            ],
            [raw_ref],
            len(value),
        )
    if left == right:
        raise HistoricalMoneylineCaptureError(
            "Polymarket saturated a one-second leaf"
        )
    midpoint = (left + right) // 2
    left_windows, left_refs, left_count = _capture_polymarket_window(
        plan,
        left,
        midpoint,
        exchange=exchange,
    )
    right_windows, right_refs, right_count = _capture_polymarket_window(
        plan,
        midpoint + 1,
        right,
        exchange=exchange,
    )
    return (
        left_windows + right_windows,
        left_refs + right_refs,
        left_count + right_count,
    )


def _capture_kalshi_trades(
    plan: MoneylineGamePlan,
    ticker: str,
    *,
    exchange: Callable[[CaptureRequest], tuple[Any, _RawResponseRef]],
    max_pages: int,
) -> tuple[dict[str, object], list[_RawResponseRef]]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    trade_timestamps: dict[str, str] = {}
    raw_count = 0
    exact_identity_time_duplicates = 0
    raw_refs: list[_RawResponseRef] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    for page_index in range(max_pages):
        params: dict[str, object] = {
            "limit": KALSHI_PAGE_LIMIT,
            "ticker": ticker,
            "min_ts": plan.start_ts,
            "max_ts": plan.end_ts,
        }
        if cursor is not None:
            params["cursor"] = cursor
        request = _request(
            plan=plan,
            source="kalshi",
            resource="kalshi_moneyline_trades",
            scope_id=ticker,
            url=KALSHI_HISTORICAL_TRADES_URL,
            params=params,
            source_cursor=(
                f"ticker:{ticker};request_cursor:{cursor or 'ROOT'}"
            ),
            coverage=(
                "stable-ID moneyline historical trade cursor chain;"
                f"ticker={ticker};start={plan.start_ts};end={plan.end_ts}"
            ),
        )
        value, raw_ref = exchange(request)
        raw_refs.append(raw_ref)
        if type(value) is not dict or set(value) != {"trades", "cursor"}:
            raise HistoricalMoneylineCaptureError(
                "Kalshi trades page is malformed"
            )
        rows = value["trades"]
        next_cursor = value["cursor"]
        if type(rows) is not list or type(next_cursor) is not str:
            raise HistoricalMoneylineCaptureError(
                "Kalshi trades cursor types are malformed"
            )
        raw_count += len(rows)
        for row in rows:
            if (
                type(row) is not dict
                or row.get("ticker") != ticker
                or type(row.get("trade_id")) is not str
                or not row["trade_id"]
                or type(row.get("created_time")) is not str
            ):
                raise HistoricalMoneylineCaptureError(
                    "Kalshi trade escaped identity scope"
                )
            timestamp_text = str(row["created_time"])
            timestamp = _parse_utc(
                timestamp_text,
                field=f"{ticker}.created_time",
            ).timestamp()
            if not plan.start_ts <= timestamp <= plan.end_ts:
                raise HistoricalMoneylineCaptureError(
                    "Kalshi trade escaped time scope"
                )
            trade_id = str(row["trade_id"])
            prior_timestamp = trade_timestamps.get(trade_id)
            if prior_timestamp is not None:
                if prior_timestamp != timestamp_text:
                    raise HistoricalMoneylineCaptureError(
                        "Kalshi stable trade ID conflicts on timestamp"
                    )
                exact_identity_time_duplicates += 1
            else:
                trade_timestamps[trade_id] = timestamp_text
            if first_timestamp is None or timestamp_text < first_timestamp:
                first_timestamp = timestamp_text
            if last_timestamp is None or timestamp_text > last_timestamp:
                last_timestamp = timestamp_text
        if next_cursor == "":
            return (
                {
                    "exact_identity_time_duplicate_count": (
                        exact_identity_time_duplicates
                    ),
                    "first_trade_timestamp_utc": first_timestamp,
                    "last_trade_timestamp_utc": last_timestamp,
                    "page_count": page_index + 1,
                    "raw_item_count": raw_count,
                    "terminal_cursor": "",
                    "terminal_proof": "explicit_empty_cursor",
                    "ticker": ticker,
                    "unique_trade_id_count": len(trade_timestamps),
                },
                raw_refs,
            )
        if next_cursor == cursor or next_cursor in seen_cursors:
            raise HistoricalMoneylineCaptureError(
                "Kalshi trades cursor repeated"
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise HistoricalMoneylineCaptureError(
        "Kalshi trades terminal cursor proof is missing"
    )


def _index_path(
    output_root: Path,
    game_id: str,
    index_sha256: str,
) -> Path:
    digest = index_sha256.removeprefix("sha256:")
    return (
        output_root
        / "games"
        / game_id
        / "objects"
        / "sha256"
        / digest[:2]
        / f"{digest}.json"
    )


def _checkpoint_path(output_root: Path, game_id: str) -> Path:
    return output_root / "checkpoints" / f"{game_id}.json"


def _verified_index_payload(
    *,
    checkpoint_path: Path,
    plan: MoneylineGamePlan,
    candidate_scan_sha256: str,
    temporal_audit_sha256: str,
    output_root: Path,
) -> tuple[dict[str, object], Path, str]:
    checkpoint_bytes, checkpoint = _read_json_file(
        checkpoint_path,
        context=f"{plan.game_id} checkpoint",
    )
    del checkpoint_bytes
    if (
        type(checkpoint) is not dict
        or checkpoint.get("schema")
        != "nfl_moneyline_game_capture_checkpoint_v1"
        or checkpoint.get("builder_version") != BUILDER_VERSION
        or checkpoint.get("game_id") != plan.game_id
        or checkpoint.get("plan_sha256") != plan.plan_sha256
        or checkpoint.get("candidate_scan_sha256")
        != candidate_scan_sha256
        or checkpoint.get("temporal_audit_sha256")
        != temporal_audit_sha256
    ):
        raise HistoricalMoneylineCaptureError(
            f"{plan.game_id} checkpoint is stale or malformed"
        )
    index_sha = _validate_sha256(
        checkpoint.get("index_sha256"),
        field="checkpoint.index_sha256",
    )
    relative = checkpoint.get("index_path")
    if type(relative) is not str:
        raise HistoricalMoneylineCaptureError(
            "checkpoint index_path is malformed"
        )
    index_path = (output_root / relative).resolve()
    try:
        index_path.relative_to(output_root.resolve())
    except ValueError as error:
        raise HistoricalMoneylineCaptureError(
            "checkpoint index path escapes output_root"
        ) from error
    index_bytes, index = _read_json_file(
        index_path,
        context=f"{plan.game_id} capture index",
    )
    if _sha256(index_bytes) != index_sha:
        raise HistoricalMoneylineCaptureError(
            "capture index content hash differs"
        )
    if (
        type(index) is not dict
        or index.get("schema") != "nfl_moneyline_game_capture_index_v1"
        or index.get("builder_version") != BUILDER_VERSION
        or index.get("capture_scope") != CAPTURE_SCOPE
        or index.get("reaction_access") != REACTION_ACCESS
        or index.get("game_id") != plan.game_id
        or index.get("plan_sha256") != plan.plan_sha256
        or index.get("candidate_scan_sha256") != candidate_scan_sha256
        or index.get("temporal_audit_sha256") != temporal_audit_sha256
        or type(index.get("raw_manifests")) is not list
    ):
        raise HistoricalMoneylineCaptureError(
            "capture index identity differs"
        )
    return index, index_path, index_sha


def _verify_raw_refs(
    index: Mapping[str, object],
    *,
    raw_root: Path,
    program_root: Path,
) -> dict[str, _VerifiedRawResponse]:
    rows = index.get("raw_manifests")
    if type(rows) is not list or not rows:
        raise HistoricalMoneylineCaptureError(
            "capture index has no raw manifests"
        )
    verified_by_request: dict[str, _VerifiedRawResponse] = {}
    for row in rows:
        if type(row) is not dict:
            raise HistoricalMoneylineCaptureError(
                "capture index raw reference is malformed"
            )
        request_sha = _validate_sha256(
            row.get("request_sha256"),
            field="raw request_sha256",
        )
        if request_sha in verified_by_request:
            raise HistoricalMoneylineCaptureError(
                "capture index duplicates a request"
            )
        manifest_relative = row.get("manifest_path")
        object_relative = row.get("object_path")
        if type(manifest_relative) is not str or type(object_relative) is not str:
            raise HistoricalMoneylineCaptureError(
                "capture index raw paths are malformed"
            )
        manifest_path = (raw_root / manifest_relative).resolve()
        object_path = (raw_root / object_relative).resolve()
        try:
            manifest_path.relative_to(raw_root.resolve())
            object_path.relative_to(raw_root.resolve())
        except ValueError as error:
            raise HistoricalMoneylineCaptureError(
                "capture index raw path escapes raw_root"
            ) from error
        try:
            verified = read_verified_static_object(
                manifest_path,
                store_root=raw_root,
                program_root=program_root,
            )
        except StaticStoreError as error:
            raise HistoricalMoneylineCaptureError(
                "capture index raw object verification failed"
            ) from error
        if (
            verified.record.object_path != object_path
            or verified.record.manifest.manifest_sha256
            != row.get("manifest_sha256")
            or verified.record.manifest.object_sha256
            != row.get("object_sha256")
            or verified.record.manifest.byte_length != row.get("byte_length")
        ):
            raise HistoricalMoneylineCaptureError(
                "capture index raw reference differs from verified bytes"
            )
        ref = _RawResponseRef(
            request_sha256=request_sha,
            object_sha256=str(row["object_sha256"]),
            manifest_sha256=str(row["manifest_sha256"]),
            byte_length=int(row["byte_length"]),
            object_path=object_relative,
            manifest_path=manifest_relative,
        )
        if verified.record.partition != _raw_partition_from_sha(request_sha):
            raise HistoricalMoneylineCaptureError(
                "raw manifest partition does not bind request SHA-256"
            )
        verified_by_request[request_sha] = _VerifiedRawResponse(
            ref=ref,
            object_bytes=verified.object_bytes,
            record=verified.record,
        )
    return verified_by_request


def _raw_partition_from_sha(request_sha256: str) -> str:
    return "nfl2025ml-" + request_sha256.removeprefix("sha256:")


def _verify_manifest_request_identity(
    request: CaptureRequest,
    verified: _VerifiedRawResponse,
) -> None:
    record = verified.record
    manifest = record.manifest
    expected_dataset = (
        "DS-POLYMARKET-PUBLIC"
        if request.source == "polymarket"
        else "DS-KALSHI-HISTORICAL"
    )
    if (
        verified.ref.request_sha256 != request.request_sha256
        or record.source != request.source
        or record.dataset != expected_dataset
        or record.version != _RAW_VERSION
        or record.partition != _raw_partition(request)
        or manifest.source_url != request.url
        or manifest.source_cursor != request.source_cursor
        or manifest.coverage != request.coverage
        or manifest.source_request.get("method") != "GET"
        or manifest.source_request.get("headers") != _REQUEST_HEADERS
        or manifest.source_request.get("params") != dict(request.params)
    ):
        raise HistoricalMoneylineCaptureError(
            "raw manifest request identity differs from required request"
        )


def _verify_checkpoint_capture_proof(
    plan: MoneylineGamePlan,
    index: Mapping[str, object],
    *,
    verified_by_request: Mapping[str, _VerifiedRawResponse],
    max_kalshi_pages: int = DEFAULT_MAX_KALSHI_PAGES,
) -> None:
    used_request_ids: set[str] = set()

    def exchange(
        request: CaptureRequest,
    ) -> tuple[Any, _RawResponseRef]:
        verified = verified_by_request.get(request.request_sha256)
        if verified is None:
            raise HistoricalMoneylineCaptureError(
                "checkpoint is missing a required raw request"
            )
        if request.request_sha256 in used_request_ids:
            raise HistoricalMoneylineCaptureError(
                "checkpoint proof reused one raw request"
            )
        _verify_manifest_request_identity(request, verified)
        used_request_ids.add(request.request_sha256)
        return (
            _strict_json(
                verified.object_bytes,
                context=f"resume:{plan.game_id}:{request.resource}",
            ),
            verified.ref,
        )

    gamma_summary, _ = _capture_gamma_metadata(plan, exchange=exchange)
    kalshi_market_summary, _ = _capture_kalshi_event_markets(
        plan,
        exchange=exchange,
        max_pages=max_kalshi_pages,
    )
    terminal_windows, _, polymarket_trade_count = (
        _capture_polymarket_window(
            plan,
            plan.start_ts,
            plan.end_ts,
            exchange=exchange,
        )
    )
    kalshi_trade_summaries = []
    for ticker in plan.kalshi_team_tickers:
        summary, _ = _capture_kalshi_trades(
            plan,
            ticker,
            exchange=exchange,
            max_pages=max_kalshi_pages,
        )
        kalshi_trade_summaries.append(summary)
    expected_polymarket_summary = {
        "condition_id": plan.polymarket_condition_id,
        "raw_item_count": polymarket_trade_count,
        "terminal_window_count": len(terminal_windows),
        "terminal_windows": terminal_windows,
    }
    if (
        index.get("identity") != plan.canonical_payload
        or index.get("polymarket_event_metadata") != gamma_summary
        or index.get("kalshi_event_markets") != kalshi_market_summary
        or index.get("polymarket_trades") != expected_polymarket_summary
        or index.get("kalshi_trades") != kalshi_trade_summaries
        or index.get("validation_fields")
        != ["identity", "timestamp", "count", "cursor"]
    ):
        raise HistoricalMoneylineCaptureError(
            "checkpoint terminal proof or capture summary differs"
        )
    if used_request_ids != set(verified_by_request):
        raise HistoricalMoneylineCaptureError(
            "checkpoint contains unexpected raw requests"
        )


def _result_from_index(
    index: Mapping[str, object],
    *,
    index_path: Path,
    index_sha256: str,
    checkpoint_path: Path,
) -> GameCaptureIndexRef:
    poly = index.get("polymarket_trades")
    kalshi = index.get("kalshi_trades")
    raw = index.get("raw_manifests")
    if type(poly) is not dict or type(kalshi) is not list or type(raw) is not list:
        raise HistoricalMoneylineCaptureError(
            "capture index summary is malformed"
        )
    return GameCaptureIndexRef(
        game_id=str(index["game_id"]),
        index_path=index_path,
        index_sha256=index_sha256,
        checkpoint_path=checkpoint_path,
        reaction_access=str(index["reaction_access"]),
        polymarket_trade_count=int(poly["raw_item_count"]),
        kalshi_trade_count=sum(
            int(item["raw_item_count"])
            for item in kalshi
            if type(item) is dict
        ),
        raw_manifest_count=len(raw),
    )


def _load_checkpoint(
    plan: MoneylineGamePlan,
    *,
    candidate_scan_sha256: str,
    temporal_audit_sha256: str,
    raw_root: Path,
    program_root: Path,
    output_root: Path,
) -> GameCaptureIndexRef | None:
    checkpoint = _checkpoint_path(output_root, plan.game_id)
    if not checkpoint.exists():
        return None
    index, index_path, index_sha = _verified_index_payload(
        checkpoint_path=checkpoint,
        plan=plan,
        candidate_scan_sha256=candidate_scan_sha256,
        temporal_audit_sha256=temporal_audit_sha256,
        output_root=output_root,
    )
    verified_by_request = _verify_raw_refs(
        index,
        raw_root=raw_root,
        program_root=program_root,
    )
    _verify_checkpoint_capture_proof(
        plan,
        index,
        verified_by_request=verified_by_request,
    )
    return _result_from_index(
        index,
        index_path=index_path,
        index_sha256=index_sha,
        checkpoint_path=checkpoint,
    )


def capture_moneyline_game(
    plan: MoneylineGamePlan,
    *,
    candidate_scan_sha256: str,
    temporal_audit_sha256: str,
    raw_root: str | Path,
    program_root: str | Path,
    output_root: str | Path,
    requester: Callable[[CaptureRequest], CaptureHttpResponse] | None = None,
    host_requests_per_second: Mapping[str, int | float] = (
        DEFAULT_HOST_REQUESTS_PER_SECOND
    ),
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_kalshi_pages: int = DEFAULT_MAX_KALSHI_PAGES,
    retry_delays_seconds: Sequence[int | float] = (
        DEFAULT_RETRY_DELAYS_SECONDS
    ),
    limiter: HostRateLimiter | None = None,
) -> GameCaptureIndexRef:
    """Capture or resume one game, publishing only a reaction-closed index."""

    _validate_plan(plan)
    candidate_sha = _validate_sha256(
        candidate_scan_sha256,
        field="candidate_scan_sha256",
    )
    temporal_sha = _validate_sha256(
        temporal_audit_sha256,
        field="temporal_audit_sha256",
    )
    raw_path = Path(raw_root).resolve()
    program_path = Path(program_root).resolve()
    output_path = Path(output_root).resolve()
    if not program_path.is_dir():
        raise HistoricalMoneylineCaptureError(
            "program_root must be an existing directory"
        )
    delays = _validate_runtime_options(
        max_response_bytes=max_response_bytes,
        max_kalshi_pages=max_kalshi_pages,
        retry_delays_seconds=retry_delays_seconds,
    )
    resumed = _load_checkpoint(
        plan,
        candidate_scan_sha256=candidate_sha,
        temporal_audit_sha256=temporal_sha,
        raw_root=raw_path,
        program_root=program_path,
        output_root=output_path,
    )
    if resumed is not None:
        return resumed

    active_limiter = limiter or HostRateLimiter(host_requests_per_second)
    owned_client: httpx.Client | None = None
    active_requester = requester
    if active_requester is None:
        owned_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=False,
        )
        active_requester = _default_http_requester(
            owned_client,
            max_response_bytes=max_response_bytes,
        )
    raw_refs: list[_RawResponseRef] = []
    try:
        def exchange(
            request: CaptureRequest,
        ) -> tuple[Any, _RawResponseRef]:
            value, raw_ref = _exchange(
                request,
                raw_root=raw_path,
                program_root=program_path,
                requester=active_requester,
                limiter=active_limiter,
                retry_delays=delays,
            )
            raw_refs.append(raw_ref)
            return value, raw_ref

        gamma_summary, _ = _capture_gamma_metadata(
            plan,
            exchange=exchange,
        )
        kalshi_market_summary, _ = _capture_kalshi_event_markets(
            plan,
            exchange=exchange,
            max_pages=max_kalshi_pages,
        )
        terminal_windows, _, polymarket_trade_count = (
            _capture_polymarket_window(
                plan,
                plan.start_ts,
                plan.end_ts,
                exchange=exchange,
            )
        )
        kalshi_trade_summaries = []
        for ticker in plan.kalshi_team_tickers:
            summary, _ = _capture_kalshi_trades(
                plan,
                ticker,
                exchange=exchange,
                max_pages=max_kalshi_pages,
            )
            kalshi_trade_summaries.append(summary)
    finally:
        if owned_client is not None:
            owned_client.close()

    raw_by_request: dict[str, _RawResponseRef] = {}
    for raw_ref in raw_refs:
        prior = raw_by_request.get(raw_ref.request_sha256)
        if prior is not None and prior != raw_ref:
            raise HistoricalMoneylineCaptureError(
                "one request resolved to conflicting immutable raw evidence"
            )
        raw_by_request[raw_ref.request_sha256] = raw_ref
    index = {
        "builder_version": BUILDER_VERSION,
        "candidate_scan_sha256": candidate_sha,
        "capture_scope": CAPTURE_SCOPE,
        "captured_at_utc": _utc_now_text(),
        "game_id": plan.game_id,
        "identity": plan.canonical_payload,
        "kalshi_event_markets": kalshi_market_summary,
        "kalshi_trades": kalshi_trade_summaries,
        "plan_sha256": plan.plan_sha256,
        "polymarket_event_metadata": gamma_summary,
        "polymarket_trades": {
            "condition_id": plan.polymarket_condition_id,
            "raw_item_count": polymarket_trade_count,
            "terminal_window_count": len(terminal_windows),
            "terminal_windows": terminal_windows,
        },
        "raw_manifests": [
            raw_by_request[key].payload()
            for key in sorted(raw_by_request)
        ],
        "reaction_access": REACTION_ACCESS,
        "schema": "nfl_moneyline_game_capture_index_v1",
        "temporal_audit_sha256": temporal_sha,
        "validation_fields": ["identity", "timestamp", "count", "cursor"],
    }
    index_bytes = _canonical_bytes(index, newline=True)
    index_sha = _sha256(index_bytes)
    index_path = _index_path(output_path, plan.game_id, index_sha)
    _publish_immutable(index_path, index_bytes)
    checkpoint = _checkpoint_path(output_path, plan.game_id)
    checkpoint_value = {
        "builder_version": BUILDER_VERSION,
        "candidate_scan_sha256": candidate_sha,
        "game_id": plan.game_id,
        "index_path": index_path.relative_to(output_path).as_posix(),
        "index_sha256": index_sha,
        "plan_sha256": plan.plan_sha256,
        "schema": "nfl_moneyline_game_capture_checkpoint_v1",
        "temporal_audit_sha256": temporal_sha,
    }
    _atomic_replace(
        checkpoint,
        _canonical_bytes(checkpoint_value, newline=True),
    )
    verified = _load_checkpoint(
        plan,
        candidate_scan_sha256=candidate_sha,
        temporal_audit_sha256=temporal_sha,
        raw_root=raw_path,
        program_root=program_path,
        output_root=output_path,
    )
    if verified is None:  # pragma: no cover - just published
        raise AssertionError("published checkpoint is missing")
    return verified


def _validate_formal_spec(spec: MoneylineExpansionSpec) -> None:
    if (
        not isinstance(spec, MoneylineExpansionSpec)
        or spec.candidate_scan_sha256 != PINNED_CANDIDATE_SCAN_SHA256
        or spec.temporal_audit_sha256 != PINNED_TEMPORAL_AUDIT_SHA256
        or spec.governance_sha256 != PINNED_GOVERNANCE_SHA256
        or len(spec.plans) != EXPECTED_GAME_COUNT
        or len(spec.game_assignments) != EXPECTED_GAME_COUNT
        or tuple(plan.game_id for plan in spec.plans)
        != tuple(row[0] for row in spec.game_assignments)
        or sum(
            cohort == "development"
            for _, cohort, _ in spec.game_assignments
        )
        != 153
        or sum(
            cohort == "final_holdout"
            for _, cohort, _ in spec.game_assignments
        )
        != 81
    ):
        raise HistoricalMoneylineCaptureError(
            "formal expansion spec differs from frozen governance"
        )


def dry_run_moneyline_expansion(
    spec: MoneylineExpansionSpec,
    *,
    raw_root: str | Path,
    program_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Verify inputs and existing checkpoints without making network calls."""

    _validate_formal_spec(spec)
    raw_path = Path(raw_root).resolve()
    program_path = Path(program_root).resolve()
    output_path = Path(output_root).resolve()
    resumed_ids: list[str] = []
    pending_ids: list[str] = []
    for plan in spec.plans:
        result = _load_checkpoint(
            plan,
            candidate_scan_sha256=spec.candidate_scan_sha256,
            temporal_audit_sha256=spec.temporal_audit_sha256,
            raw_root=raw_path,
            program_root=program_path,
            output_root=output_path,
        )
        (resumed_ids if result is not None else pending_ids).append(
            plan.game_id
        )
    return {
        "batch_plan_sha256": spec.batch_plan_sha256,
        "builder_version": BUILDER_VERSION,
        "candidate_scan_sha256": spec.candidate_scan_sha256,
        "capture_scope": CAPTURE_SCOPE,
        "game_count": len(spec.plans),
        "governance_sha256": spec.governance_sha256,
        "minimum_request_count_per_pending_game": 5,
        "network_accessed": False,
        "network_workers": NETWORK_WORKERS,
        "pending_game_count": len(pending_ids),
        "pending_game_ids": pending_ids,
        "reaction_access": REACTION_ACCESS,
        "resumed_game_count": len(resumed_ids),
        "resumed_game_ids": resumed_ids,
        "schema": "nfl_moneyline_expansion_dry_run_v1",
        "temporal_audit_sha256": spec.temporal_audit_sha256,
    }


def _publish_batch_manifest(
    spec: MoneylineExpansionSpec,
    results: Sequence[GameCaptureIndexRef],
    *,
    output_root: Path,
    resumed_game_count: int,
) -> BatchCaptureManifestRef:
    ordered = tuple(sorted(results, key=lambda row: row.game_id))
    expected_ids = tuple(plan.game_id for plan in spec.plans)
    if tuple(item.game_id for item in ordered) != expected_ids:
        raise HistoricalMoneylineCaptureError(
            "batch publication requires every exact game index"
        )
    manifest = {
        "batch_plan_sha256": spec.batch_plan_sha256,
        "builder_version": BUILDER_VERSION,
        "candidate_scan_sha256": spec.candidate_scan_sha256,
        "capture_scope": CAPTURE_SCOPE,
        "game_count": len(ordered),
        "governance_sha256": spec.governance_sha256,
        "game_indexes": [
            {
                "game_id": item.game_id,
                "index_path": item.index_path.relative_to(
                    output_root
                ).as_posix(),
                "index_sha256": item.index_sha256,
                "kalshi_trade_count": item.kalshi_trade_count,
                "polymarket_trade_count": item.polymarket_trade_count,
                "raw_manifest_count": item.raw_manifest_count,
            }
            for item in ordered
        ],
        "network_workers": NETWORK_WORKERS,
        "reaction_access": REACTION_ACCESS,
        "schema": "nfl_moneyline_expansion_batch_manifest_v1",
        "temporal_audit_sha256": spec.temporal_audit_sha256,
        "validation_fields": ["identity", "timestamp", "count", "cursor"],
    }
    manifest_bytes = _canonical_bytes(manifest, newline=True)
    manifest_sha = _sha256(manifest_bytes)
    digest = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        output_root
        / "batch"
        / "manifests"
        / f"{digest}.manifest.json"
    )
    _publish_immutable(manifest_path, manifest_bytes)
    pointer_path = output_root / "batch" / "latest.json"
    pointer = {
        "batch_plan_sha256": spec.batch_plan_sha256,
        "manifest_path": manifest_path.relative_to(output_root).as_posix(),
        "manifest_sha256": manifest_sha,
        "schema": "nfl_moneyline_expansion_batch_pointer_v1",
    }
    _atomic_replace(pointer_path, _canonical_bytes(pointer, newline=True))
    return BatchCaptureManifestRef(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        pointer_path=pointer_path,
        game_count=len(ordered),
        resumed_game_count=resumed_game_count,
        captured_game_count=len(ordered) - resumed_game_count,
    )


def capture_moneyline_expansion(
    spec: MoneylineExpansionSpec,
    *,
    raw_root: str | Path,
    program_root: str | Path,
    output_root: str | Path,
    host_requests_per_second: Mapping[str, int | float] = (
        DEFAULT_HOST_REQUESTS_PER_SECOND
    ),
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    max_kalshi_pages: int = DEFAULT_MAX_KALSHI_PAGES,
    retry_delays_seconds: Sequence[int | float] = (
        DEFAULT_RETRY_DELAYS_SECONDS
    ),
) -> BatchCaptureManifestRef:
    """Capture the exact 234-game cohort with two workers and checkpoints."""

    _validate_formal_spec(spec)
    raw_path = Path(raw_root).resolve()
    program_path = Path(program_root).resolve()
    output_path = Path(output_root).resolve()
    delays = _validate_runtime_options(
        max_response_bytes=max_response_bytes,
        max_kalshi_pages=max_kalshi_pages,
        retry_delays_seconds=retry_delays_seconds,
    )
    limiter = HostRateLimiter(host_requests_per_second)
    resumed: list[GameCaptureIndexRef] = []
    pending: list[MoneylineGamePlan] = []
    for plan in spec.plans:
        existing = _load_checkpoint(
            plan,
            candidate_scan_sha256=spec.candidate_scan_sha256,
            temporal_audit_sha256=spec.temporal_audit_sha256,
            raw_root=raw_path,
            program_root=program_path,
            output_root=output_path,
        )
        if existing is None:
            pending.append(plan)
        else:
            resumed.append(existing)

    captured: list[GameCaptureIndexRef] = []
    failures: list[tuple[str, BaseException]] = []
    with httpx.Client(
        timeout=httpx.Timeout(60.0, connect=20.0),
        follow_redirects=False,
    ) as client:
        requester = _default_http_requester(
            client,
            max_response_bytes=max_response_bytes,
        )
        with ThreadPoolExecutor(
            max_workers=NETWORK_WORKERS,
            thread_name_prefix="nfl-moneyline-capture",
        ) as executor:
            futures = {
                executor.submit(
                    capture_moneyline_game,
                    plan,
                    candidate_scan_sha256=spec.candidate_scan_sha256,
                    temporal_audit_sha256=spec.temporal_audit_sha256,
                    raw_root=raw_path,
                    program_root=program_path,
                    output_root=output_path,
                    requester=requester,
                    host_requests_per_second=host_requests_per_second,
                    max_response_bytes=max_response_bytes,
                    max_kalshi_pages=max_kalshi_pages,
                    retry_delays_seconds=delays,
                    limiter=limiter,
                ): plan.game_id
                for plan in pending
            }
            for future in as_completed(futures):
                try:
                    captured.append(future.result())
                except BaseException as error:
                    failures.append((futures[future], error))
    if failures:
        examples = "; ".join(
            f"{game_id}: {error}"
            for game_id, error in sorted(failures)[:5]
        )
        raise HistoricalMoneylineCaptureError(
            f"{len(failures)} game captures failed after checkpointing "
            f"successful games; no batch manifest published: {examples}"
        )
    results = tuple(resumed + captured)
    return _publish_batch_manifest(
        spec,
        results,
        output_root=output_path,
        resumed_game_count=len(resumed),
    )


__all__ = [
    "BUILDER_VERSION",
    "CAPTURE_SCOPE",
    "DEFAULT_HOST_REQUESTS_PER_SECOND",
    "EXPECTED_GAME_COUNT",
    "NETWORK_WORKERS",
    "PINNED_CANDIDATE_SCAN_SHA256",
    "PINNED_GOVERNANCE_SHA256",
    "PINNED_TEMPORAL_AUDIT_SHA256",
    "REACTION_ACCESS",
    "BatchCaptureManifestRef",
    "CaptureHttpResponse",
    "CaptureRequest",
    "GameCaptureIndexRef",
    "HistoricalMoneylineCaptureError",
    "MoneylineExpansionSpec",
    "MoneylineGamePlan",
    "TransientMoneylineCaptureError",
    "capture_moneyline_expansion",
    "capture_moneyline_game",
    "dry_run_moneyline_expansion",
    "load_moneyline_expansion_spec",
]
