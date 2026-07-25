"""Fail-closed raw capture and preflight for the frozen NFL X-13 batch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import httpx

from prediction_market.kalshi_history import (
    KALSHI_API_ROOT,
    KALSHI_HISTORICAL_VERSION,
)
from prediction_market.polymarket_history import (
    DATA_API_TRADES_URL,
    GAMMA_EVENTS_URL,
    POLYMARKET_PUBLIC_VERSION,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS
from prediction_market.static_store import (
    StaticStoreError,
    preserve_static_object,
    verify_static_object,
)


GIB = 1 << 30
POLYMARKET_TRADE_LIMIT = 10_000
KALSHI_PAGE_LIMIT = 1_000
GAME_NETWORK_WORKERS = 4
X13_FROZEN_CAPTURE_TARGET_COUNT = 4_315
X13_FROZEN_CAPTURE_GAME_IDS = tuple(X13_GAME_IDS)
X13_FROZEN_UNIVERSE_ID = (
    "sha256:01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
)

DEFAULT_HOST_REQUESTS_PER_SECOND = {
    "gamma-api.polymarket.com": 5,
    "data-api.polymarket.com": 5,
    "external-api.kalshi.com": 5,
}

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
_KINDS = frozenset(
    {"primitive", "same_game_composite", "cross_game_inventory"}
)
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,127}\Z")
_CONDITION_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SCHEMA_FINGERPRINT = (
    "sha256:" + hashlib.sha256(b"x13-opaque-json-response-v1").hexdigest()
)


class X13CaptureError(ValueError):
    """The raw evidence cannot prove a complete frozen X-13 capture."""


class X13TransientCaptureError(X13CaptureError):
    """A retryable transport or upstream availability failure."""


@dataclass(frozen=True, slots=True)
class CaptureTarget:
    """One normalized contract bound to its exact venue-native identifiers."""

    contract_id: str
    venue: str
    venue_market_id: str
    condition_id: str | None
    venue_title: str
    kalshi_series_ticker: str | None
    kalshi_event_ticker: str | None
    family: str
    dependency_game_ids: tuple[str, ...]
    kind: str
    analysis_eligible: bool


@dataclass(frozen=True, slots=True)
class GameCapturePlan:
    """Frozen venue identities and inclusive historical window for one game."""

    game_id: str
    polymarket_event_id: str
    polymarket_event_slug: str
    polymarket_event_title: str
    start_ts: int
    end_ts: int
    targets: tuple[CaptureTarget, ...]


@dataclass(frozen=True, slots=True)
class DiskPreflight:
    raw_store_root: Path
    program_root: Path
    planned_raw_p95_bytes: int
    derived_p95_bytes: int
    temporary_headroom_bytes: int
    reserve_bytes: int
    required_free_bytes: int
    available_free_bytes: int


@dataclass(frozen=True, slots=True)
class CaptureRequest:
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
        material = {
            "coverage": self.coverage,
            "game_id": self.game_id,
            "params": dict(self.params),
            "resource": self.resource,
            "scope_id": self.scope_id,
            "source": self.source,
            "source_cursor": self.source_cursor,
            "url": self.url,
        }
        return _sha256(_canonical_bytes(material))


@dataclass(frozen=True, slots=True)
class CaptureHttpResponse:
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ImmutableRequestManifest:
    request_sha256: str
    object_sha256: str
    manifest_sha256: str
    byte_length: int
    object_path: Path
    manifest_path: Path
    response_received_at_utc: datetime


@dataclass(frozen=True, slots=True)
class PolymarketTerminalWindow:
    condition_id: str
    start_ts: int
    end_ts: int
    item_count: int
    manifest_sha256: str
    terminal_proof: str = "unsaturated_time_window"


@dataclass(frozen=True, slots=True)
class KalshiTradeProof:
    ticker: str
    page_count: int
    raw_item_count: int
    unique_item_count: int
    exact_duplicate_count: int
    terminal_cursor: str
    terminal_proof: str = "explicit_empty_cursor"


@dataclass(frozen=True, slots=True)
class KalshiCandlestickCapture:
    ticker: str
    candlestick_count: int
    first_end_period_ts: int | None
    last_end_period_ts: int | None
    manifest_sha256: str
    period_interval: int = 1


@dataclass(frozen=True, slots=True)
class KalshiHistoricalCutoff:
    market_positions_last_updated_ts: datetime
    market_settled_ts: datetime
    orders_updated_ts: datetime
    trades_created_ts: datetime
    manifest_sha256: str
    response_received_at_utc: datetime


@dataclass(frozen=True, slots=True)
class GameCaptureResult:
    game_id: str
    manifests: tuple[ImmutableRequestManifest, ...]
    polymarket_market_count: int
    polymarket_trade_count: int
    polymarket_terminal_windows: tuple[PolymarketTerminalWindow, ...]
    kalshi_market_count: int
    kalshi_trade_proofs: tuple[KalshiTradeProof, ...]
    kalshi_candlesticks: tuple[KalshiCandlestickCapture, ...]


@dataclass(frozen=True, slots=True)
class CaptureReceipt:
    universe_id: str
    plan_sha256: str
    preflight: DiskPreflight
    kalshi_cutoff: KalshiHistoricalCutoff
    games: tuple[GameCaptureResult, ...]
    manifests: tuple[ImmutableRequestManifest, ...]

    @property
    def canonical_payload(self) -> dict[str, object]:
        return _capture_receipt_payload(self)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.canonical_payload))


@dataclass(frozen=True, slots=True)
class CapturePointer:
    capture_sha256: str
    universe_id: str
    plan_sha256: str
    receipt_sha256: str
    game_ids: tuple[str, ...]
    raw_manifest_sha256s: tuple[str, ...]

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "capture_sha256": self.capture_sha256,
            "universe_id": self.universe_id,
            "plan_sha256": self.plan_sha256,
            "receipt_sha256": self.receipt_sha256,
            "game_ids": list(self.game_ids),
            "raw_manifest_sha256s": list(self.raw_manifest_sha256s),
            "version": "nfl-x13-capture-pointer-v2",
        }


@dataclass(frozen=True, slots=True)
class BatchCaptureResult:
    preflight: DiskPreflight
    kalshi_cutoff: KalshiHistoricalCutoff
    games: tuple[GameCaptureResult, ...]
    manifests: tuple[ImmutableRequestManifest, ...]
    receipt: CaptureReceipt
    receipt_path: Path
    pointer: CapturePointer
    pointer_path: Path


class HostRateLimiter:
    """Thread-safe fixed-interval limiter shared by all game workers."""

    def __init__(
        self,
        requests_per_second: Mapping[str, int | float],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(requests_per_second, Mapping):
            raise X13CaptureError(
                "host_requests_per_second must be a mapping"
            )
        intervals: dict[str, float] = {}
        for host, rate in requests_per_second.items():
            if (
                type(host) is not str
                or not host
                or host != host.lower()
                or "/" in host
            ):
                raise X13CaptureError("rate-limit host must be canonical")
            if (
                type(rate) not in {int, float}
                or not math.isfinite(float(rate))
                or rate <= 0
            ):
                raise X13CaptureError(
                    "host request rate must be finite and positive"
                )
            intervals[host] = 1.0 / float(rate)
        if not intervals:
            raise X13CaptureError(
                "at least one host request rate is required"
            )
        self._intervals = intervals
        self._monotonic = monotonic
        self._sleep = sleep
        self._locks = {
            host: threading.Lock() for host in self._intervals
        }
        self._next_allowed = {
            host: 0.0 for host in self._intervals
        }

    def wait_for_url(self, url: str) -> None:
        host = urlsplit(url).hostname
        if host is None or host not in self._intervals:
            raise X13CaptureError(
                f"host rate limit is not configured for {host!r}"
            )
        with self._locks[host]:
            now = self._monotonic()
            deadline = self._next_allowed[host]
            if now < deadline:
                self._sleep(deadline - now)
                now = max(deadline, self._monotonic())
            self._next_allowed[host] = (
                max(now, deadline) + self._intervals[host]
            )


class _RunningDiskReservationGate:
    """Reserve concurrent raw-write headroom before a writer mutates disk."""

    _MANIFEST_HEADROOM_BYTES = 1 << 20

    def __init__(
        self,
        *,
        raw_store_root: Path,
        minimum_free_bytes: int,
        disk_usage: Callable[[Path], Any],
    ) -> None:
        self._probe = _disk_probe(raw_store_root)
        self._minimum_free_bytes = minimum_free_bytes
        self._disk_usage = disk_usage
        self._active_reserved_bytes = 0
        self._lock = threading.Lock()

    def _available(self) -> int:
        try:
            available = self._disk_usage(self._probe).free
        except (OSError, AttributeError) as error:
            raise X13CaptureError(
                "running disk gate could not read free bytes"
            ) from error
        if type(available) is not int or available < 0:
            raise X13CaptureError(
                "running disk gate returned invalid free bytes"
            )
        return available

    def acquire(self, response_byte_length: int) -> int:
        if (
            type(response_byte_length) is not int
            or response_byte_length < 0
        ):
            raise X13CaptureError(
                "running disk reservation byte length is invalid"
            )
        reservation = (
            response_byte_length + self._MANIFEST_HEADROOM_BYTES
        )
        with self._lock:
            available = self._available()
            required = (
                self._minimum_free_bytes
                + self._active_reserved_bytes
                + reservation
            )
            if available < required:
                raise X13CaptureError(
                    "running disk gate failed before the untouched reserve"
                )
            self._active_reserved_bytes += reservation
        return reservation

    def release(self, reservation: int, *, verify_after_write: bool) -> None:
        with self._lock:
            if (
                type(reservation) is not int
                or reservation <= 0
                or reservation > self._active_reserved_bytes
            ):
                raise X13CaptureError(
                    "running disk reservation state is invalid"
                )
            self._active_reserved_bytes -= reservation
            if verify_after_write:
                available = self._available()
                required = (
                    self._minimum_free_bytes
                    + self._active_reserved_bytes
                )
                if available < required:
                    raise X13CaptureError(
                        "running disk gate failed before the untouched reserve"
                    )


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
        raise X13CaptureError("capture identity is not canonical JSON") from error


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise X13CaptureError(f"{field} must be an exact nonempty string")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise X13CaptureError(f"{field} must be a non-negative integer")
    return value


def _absolute_path(value: str | Path, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise X13CaptureError(f"{field} must be a path")
    path = Path(value)
    if not path.is_absolute():
        raise X13CaptureError(f"{field} must be explicit and absolute")
    return path


def _disk_probe(path: Path) -> Path:
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise X13CaptureError(
                "raw_store_root has no existing filesystem ancestor"
            )
        probe = parent
    if not probe.is_dir():
        raise X13CaptureError(
            "raw_store_root filesystem ancestor is not a directory"
        )
    return probe


def _validate_target(
    target: CaptureTarget,
    *,
    plan: GameCapturePlan,
    known_game_ids: frozenset[str],
) -> None:
    if not isinstance(target, CaptureTarget):
        raise X13CaptureError(
            "targets must contain CaptureTarget values"
        )
    _nonempty_string(target.contract_id, "contract_id")
    if target.venue not in {"polymarket", "kalshi"}:
        raise X13CaptureError("unknown venue")
    if not target.contract_id.startswith(f"{target.venue}:"):
        raise X13CaptureError(
            "contract_id must be namespaced by its venue"
        )
    _nonempty_string(target.venue_market_id, "venue_market_id")
    _nonempty_string(target.venue_title, "venue_title")
    if target.family not in _FAMILIES:
        raise X13CaptureError("unknown proposition family")
    if target.kind not in _KINDS:
        raise X13CaptureError("unknown proposition kind")
    if (
        type(target.dependency_game_ids) is not tuple
        or not target.dependency_game_ids
        or any(
            type(game_id) is not str or not game_id
            for game_id in target.dependency_game_ids
        )
        or len(set(target.dependency_game_ids))
        != len(target.dependency_game_ids)
    ):
        raise X13CaptureError(
            "dependency_game_ids must be unique nonempty strings"
        )
    unknown_dependencies = set(target.dependency_game_ids) - known_game_ids
    if unknown_dependencies:
        raise X13CaptureError("unknown game dependency")
    if plan.game_id not in target.dependency_game_ids:
        raise X13CaptureError("target does not depend on its capture game")
    if type(target.analysis_eligible) is not bool:
        raise X13CaptureError("analysis_eligible must be bool")

    if target.kind == "primitive":
        if target.family not in _PRIMITIVE_FAMILIES:
            raise X13CaptureError(
                "primitive kind requires a primitive family"
            )
        if len(target.dependency_game_ids) != 1:
            raise X13CaptureError(
                "primitive target has wrong game dependency"
            )
    elif target.kind == "same_game_composite":
        if target.family != "same_game_composite":
            raise X13CaptureError(
                "same-game kind/family identity mismatch"
            )
        if target.dependency_game_ids != (plan.game_id,):
            raise X13CaptureError(
                "same-game target has wrong game dependency"
            )
    else:
        if target.family != "cross_game_mve_inventory":
            raise X13CaptureError(
                "cross-game kind/family identity mismatch"
            )
        if len(target.dependency_game_ids) < 2:
            raise X13CaptureError(
                "cross-game inventory requires multiple dependencies"
            )
        if target.analysis_eligible:
            raise X13CaptureError(
                "cross-game inventory cannot be analysis eligible"
            )

    if target.analysis_eligible and target.dependency_game_ids != (
        plan.game_id,
    ):
        raise X13CaptureError(
            "analysis-eligible target has wrong game dependency"
        )
    if target.venue == "polymarket":
        if (
            type(target.condition_id) is not str
            or _CONDITION_RE.fullmatch(target.condition_id) is None
            or target.kalshi_series_ticker is not None
            or target.kalshi_event_ticker is not None
        ):
            raise X13CaptureError(
                "Polymarket target requires only its canonical condition ID"
            )
    else:
        if (
            target.condition_id is not None
            or _TICKER_RE.fullmatch(target.venue_market_id) is None
            or type(target.kalshi_series_ticker) is not str
            or _TICKER_RE.fullmatch(
                target.kalshi_series_ticker
            )
            is None
            or type(target.kalshi_event_ticker) is not str
            or _TICKER_RE.fullmatch(
                target.kalshi_event_ticker
            )
            is None
            or not target.kalshi_event_ticker.startswith(
                f"{target.kalshi_series_ticker}-"
            )
            or not target.venue_market_id.startswith(
                f"{target.kalshi_event_ticker}-"
            )
        ):
            raise X13CaptureError(
                "Kalshi target requires frozen series/event/ticker identity"
            )


def _validate_plan(
    plan: GameCapturePlan,
    *,
    known_game_ids: frozenset[str],
) -> None:
    if not isinstance(plan, GameCapturePlan):
        raise X13CaptureError(
            "plans must contain GameCapturePlan values"
        )
    if plan.game_id not in known_game_ids:
        raise X13CaptureError("unknown capture game")
    if (
        type(plan.polymarket_event_id) is not str
        or not plan.polymarket_event_id.isdigit()
    ):
        raise X13CaptureError(
            "polymarket_event_id must contain digits"
        )
    if (
        type(plan.polymarket_event_slug) is not str
        or _SLUG_RE.fullmatch(plan.polymarket_event_slug) is None
    ):
        raise X13CaptureError(
            "polymarket_event_slug is not canonical"
        )
    _nonempty_string(
        plan.polymarket_event_title,
        "polymarket_event_title",
    )
    start_ts = _nonnegative_integer(plan.start_ts, "start_ts")
    end_ts = _nonnegative_integer(plan.end_ts, "end_ts")
    if end_ts < start_ts:
        raise X13CaptureError("capture window is reversed")
    if type(plan.targets) is not tuple or not plan.targets:
        raise X13CaptureError("game capture targets must be nonempty")

    contract_ids: set[str] = set()
    venue_ids: dict[str, set[str]] = {
        "polymarket": set(),
        "kalshi": set(),
    }
    condition_ids: set[str] = set()
    for target in plan.targets:
        _validate_target(
            target,
            plan=plan,
            known_game_ids=known_game_ids,
        )
        if target.contract_id in contract_ids:
            raise X13CaptureError("duplicate contract target")
        contract_ids.add(target.contract_id)
        if target.venue_market_id in venue_ids[target.venue]:
            raise X13CaptureError("duplicate venue market target")
        venue_ids[target.venue].add(target.venue_market_id)
        if target.condition_id is not None:
            if target.condition_id in condition_ids:
                raise X13CaptureError("duplicate condition target")
            condition_ids.add(target.condition_id)
    if not all(venue_ids.values()):
        raise X13CaptureError(
            "each game requires both venue inventories"
        )


def preflight_x13_capture(
    plans: Sequence[GameCapturePlan],
    *,
    raw_store_root: str | Path,
    program_root: str | Path,
    planned_raw_p95_bytes: int,
    derived_p95_bytes: int,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> DiskPreflight:
    """Validate the frozen plan and exact disk headroom before any request."""

    if (
        isinstance(plans, (str, bytes))
        or not isinstance(plans, Sequence)
        or not plans
    ):
        raise X13CaptureError("plans must be a nonempty sequence")
    raw_root = _absolute_path(raw_store_root, "raw_store_root")
    program = _absolute_path(program_root, "program_root")
    if not program.is_dir():
        raise X13CaptureError(
            "program_root must be an existing directory"
        )
    if raw_root.exists() and not raw_root.is_dir():
        raise X13CaptureError(
            "raw_store_root must be a directory path"
        )
    planned = _nonnegative_integer(
        planned_raw_p95_bytes,
        "planned_raw_p95_bytes",
    )
    derived = _nonnegative_integer(
        derived_p95_bytes,
        "derived_p95_bytes",
    )

    known_game_ids = frozenset(X13_GAME_IDS)
    seen_games: set[str] = set()
    seen_contracts: set[str] = set()
    for plan in plans:
        _validate_plan(plan, known_game_ids=known_game_ids)
        if plan.game_id in seen_games:
            raise X13CaptureError("duplicate game capture plan")
        seen_games.add(plan.game_id)
        for target in plan.targets:
            if target.contract_id in seen_contracts:
                raise X13CaptureError(
                    "contract target appears in multiple games"
                )
            seen_contracts.add(target.contract_id)

    required = planned + derived + GIB + 5 * GIB
    try:
        usage = disk_usage(_disk_probe(raw_root))
        available = usage.free
    except (OSError, AttributeError) as error:
        raise X13CaptureError("disk preflight could not read free bytes") from error
    if type(available) is not int or available < 0:
        raise X13CaptureError(
            "disk preflight returned invalid free bytes"
        )
    if available < required:
        raise X13CaptureError(
            "disk preflight failed: available bytes are below "
            "planned_raw_p95 + derived_p95 + 1GiB temp + 5GiB reserve"
        )
    return DiskPreflight(
        raw_store_root=raw_root,
        program_root=program,
        planned_raw_p95_bytes=planned,
        derived_p95_bytes=derived,
        temporary_headroom_bytes=GIB,
        reserve_bytes=5 * GIB,
        required_free_bytes=required,
        available_free_bytes=available,
    )


def _strict_json(payload: bytes, *, context: str) -> Any:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise X13CaptureError(
                    f"{context} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode(),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                X13CaptureError(
                    f"{context} contains non-finite JSON value {value}"
                )
            ),
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        X13CaptureError,
    ) as error:
        if isinstance(error, X13CaptureError):
            raise
        raise X13CaptureError(
            f"{context} is not strict UTF-8 JSON"
        ) from error


def _request(
    *,
    game_id: str,
    source: str,
    resource: str,
    scope_id: str,
    url: str,
    params: Mapping[str, object],
    source_cursor: str,
    coverage: str,
) -> CaptureRequest:
    return CaptureRequest(
        game_id=game_id,
        source=source,
        resource=resource,
        scope_id=scope_id,
        url=url,
        params=tuple(sorted(params.items())),
        source_cursor=source_cursor,
        coverage=coverage,
    )


def _http_response(
    client: httpx.Client,
    max_response_bytes: int,
) -> Callable[[CaptureRequest], CaptureHttpResponse]:
    def fetch(request: CaptureRequest) -> CaptureHttpResponse:
        try:
            with client.stream(
                "GET",
                request.url,
                params=dict(request.params),
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            ) as response:
                response.raise_for_status()
                if response.headers.get("Content-Encoding") not in {
                    None,
                    "identity",
                }:
                    raise X13CaptureError(
                        "capture response is content-encoded"
                    )
                declared = response.headers.get("Content-Length")
                declared_length: int | None = None
                if declared is not None:
                    try:
                        declared_length = int(declared)
                    except ValueError as error:
                        raise X13CaptureError(
                            "capture response has invalid Content-Length"
                        ) from error
                    if declared_length > max_response_bytes:
                        raise X13CaptureError(
                            "capture response exceeds max_response_bytes"
                        )
                body_parts: list[bytes] = []
                received = 0
                for part in response.iter_bytes():
                    received += len(part)
                    if received > max_response_bytes:
                        raise X13CaptureError(
                            "capture response exceeds max_response_bytes"
                        )
                    body_parts.append(part)
                if (
                    declared_length is not None
                    and received != declared_length
                ):
                    raise X13TransientCaptureError(
                        "capture response Content-Length mismatch"
                    )
                return CaptureHttpResponse(
                    body=b"".join(body_parts),
                    headers=tuple(response.headers.multi_items()),
                )
        except httpx.HTTPStatusError as error:
            if (
                error.response.status_code in {408, 425, 429}
                or 500 <= error.response.status_code <= 599
            ):
                raise X13TransientCaptureError(
                    f"capture request temporarily failed: {error}"
                ) from error
            raise X13CaptureError(
                f"capture request failed: {error}"
            ) from error
        except httpx.RequestError as error:
            raise X13TransientCaptureError(
                f"capture request temporarily failed: {error}"
            ) from error
        except httpx.HTTPError as error:
            raise X13CaptureError(
                f"capture request failed: {error}"
            ) from error

    return fetch


def _system_response_clock() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc_response_time(value: object) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise X13CaptureError(
            "response clock must return a UTC datetime"
        )
    return value.astimezone(timezone.utc)


def _utc_response_text(value: object) -> str:
    return (
        _require_utc_response_time(value)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def preserve_capture_response(
    request: CaptureRequest,
    response: CaptureHttpResponse,
    raw_store_root: str | Path,
    program_root: str | Path,
    response_received_at_utc: datetime,
) -> ImmutableRequestManifest:
    """Preserve one exact response with its request-bound static manifest."""

    raw_root = _absolute_path(raw_store_root, "raw_store_root")
    program = _absolute_path(program_root, "program_root")
    headers = httpx.Headers(response.headers)
    request_digest = request.request_sha256.removeprefix("sha256:")
    if request.source == "polymarket":
        dataset = "DS-POLYMARKET-PUBLIC"
        version = POLYMARKET_PUBLIC_VERSION
        license_ref = "O-001"
    elif request.source == "kalshi":
        dataset = "DS-KALSHI-HISTORICAL"
        version = KALSHI_HISTORICAL_VERSION
        license_ref = "O-003"
    else:  # pragma: no cover - requests are built internally
        raise X13CaptureError("unknown manifest source")
    record = preserve_static_object(
        raw_root,
        response.body,
        program_root=program,
        source=request.source,
        dataset=dataset,
        version=version,
        partition=f"x13-{request_digest}",
        extension="json",
        source_url=request.url,
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            "params": dict(request.params),
        },
        source_cursor=request.source_cursor,
        fetched_at=_utc_response_text(response_received_at_utc),
        coverage=request.coverage,
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/json").split(
            ";", 1
        )[0],
        schema_fingerprint=_SCHEMA_FINGERPRINT,
        license_ref=license_ref,
        license_status="research_only",
        upstream_partition=f"x13-{request_digest}",
        object_kind="byte_exact_original",
        lineage={
            "source_object_refs": [],
            "query_sha256": None,
        },
    )
    return ImmutableRequestManifest(
        request_sha256=request.request_sha256,
        object_sha256=record.manifest.object_sha256,
        manifest_sha256=record.manifest.manifest_sha256,
        byte_length=record.manifest.byte_length,
        object_path=record.object_path,
        manifest_path=record.manifest_path,
        response_received_at_utc=_require_utc_response_time(
            response_received_at_utc
        ),
    )


def _exchange(
    request: CaptureRequest,
    *,
    requester: Callable[[CaptureRequest], CaptureHttpResponse],
    manifest_writer: Callable[
        [
            CaptureRequest,
            CaptureHttpResponse,
            Path,
            Path,
            datetime,
        ],
        ImmutableRequestManifest,
    ],
    raw_store_root: Path,
    program_root: Path,
    response_clock: Callable[[], datetime],
    rate_limiter: HostRateLimiter,
    manifests: list[ImmutableRequestManifest],
) -> Any:
    rate_limiter.wait_for_url(request.url)
    response = requester(request)
    response_received_at_utc = _require_utc_response_time(
        response_clock()
    )
    if (
        not isinstance(response, CaptureHttpResponse)
        or type(response.body) is not bytes
        or type(response.headers) is not tuple
    ):
        raise X13CaptureError(
            "requester must return CaptureHttpResponse with exact bytes"
        )
    receipt = manifest_writer(
        request,
        response,
        raw_store_root,
        program_root,
        response_received_at_utc,
    )
    if not isinstance(receipt, ImmutableRequestManifest):
        raise X13CaptureError(
            "manifest_writer must return ImmutableRequestManifest"
        )
    if (
        receipt.request_sha256 != request.request_sha256
        or _SHA256_RE.fullmatch(receipt.object_sha256) is None
        or _SHA256_RE.fullmatch(receipt.manifest_sha256) is None
        or receipt.object_sha256 != _sha256(response.body)
        or receipt.byte_length != len(response.body)
        or receipt.response_received_at_utc
        != response_received_at_utc
    ):
        raise X13CaptureError(
            "immutable request manifest does not bind exact response bytes"
        )
    manifests.append(receipt)
    return _strict_json(
        response.body,
        context=f"{request.source} {request.resource}",
    )


def _capture_gamma_metadata(
    plan: GameCapturePlan,
    *,
    exchange: Callable[[CaptureRequest], Any],
) -> int:
    request = _request(
        game_id=plan.game_id,
        source="polymarket",
        resource="gamma_event",
        scope_id=plan.polymarket_event_id,
        url=GAMMA_EVENTS_URL,
        params={"id": plan.polymarket_event_id},
        source_cursor=f"event_id:{plan.polymarket_event_id}",
        coverage=(
            "exact Gamma event and complete embedded market metadata;"
            f"event_id={plan.polymarket_event_id}"
        ),
    )
    value = exchange(request)
    if (
        type(value) is not list
        or len(value) != 1
        or type(value[0]) is not dict
    ):
        raise X13CaptureError(
            "Gamma exact event query must return one event"
        )
    event = value[0]
    if event.get("id") != plan.polymarket_event_id:
        raise X13CaptureError("Gamma event ID is not exact")
    if event.get("slug") != plan.polymarket_event_slug:
        raise X13CaptureError("Gamma event slug is not exact")
    if event.get("title") != plan.polymarket_event_title:
        raise X13CaptureError(
            "Gamma event title is ambiguous or changed"
        )
    markets = event.get("markets")
    if type(markets) is not list:
        raise X13CaptureError(
            "Gamma event markets metadata must be an array"
        )
    expected_targets = {
        (target.venue_market_id, target.condition_id): target
        for target in plan.targets
        if target.venue == "polymarket"
    }
    observed: dict[tuple[str, str], bytes] = {}
    for index, market in enumerate(markets):
        if type(market) is not dict:
            raise X13CaptureError(
                f"Gamma market[{index}] must be an object"
            )
        market_id = market.get("id")
        condition_id = market.get("conditionId")
        if (
            type(market_id) is not str
            or not market_id
            or type(condition_id) is not str
            or _CONDITION_RE.fullmatch(condition_id) is None
        ):
            raise X13CaptureError(
                "Gamma market metadata identity is incomplete"
            )
        identity = (market_id, condition_id)
        target = expected_targets.get(identity)
        if (
            target is not None
            and market.get("question") != target.venue_title
        ):
            raise X13CaptureError(
                "Gamma market title is ambiguous or changed"
            )
        canonical = _canonical_bytes(market)
        prior = observed.get(identity)
        if prior is not None and prior != canonical:
            raise X13CaptureError(
                "Gamma market has conflicting duplicate metadata"
            )
        if prior is not None:
            raise X13CaptureError(
                "Gamma market metadata identity is duplicated"
            )
        observed[identity] = canonical
    if set(observed) != set(expected_targets):
        raise X13CaptureError(
            "Gamma all-market metadata does not match inventory"
        )
    return len(observed)


def _parse_utc_cutoff(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise X13CaptureError(f"Kalshi cutoff {field} must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise X13CaptureError(
            f"Kalshi cutoff {field} is malformed"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise X13CaptureError(f"Kalshi cutoff {field} must be UTC")
    return parsed


def _capture_kalshi_cutoff(
    plans: Sequence[GameCapturePlan],
    *,
    exchange: Callable[
        [CaptureRequest],
        tuple[Any, ImmutableRequestManifest],
    ],
) -> KalshiHistoricalCutoff:
    request = _request(
        game_id="X-13",
        source="kalshi",
        resource="kalshi_cutoff",
        scope_id="X-13",
        url=f"{KALSHI_API_ROOT}/historical/cutoff",
        params={},
        source_cursor="frozen_once_before_batch",
        coverage=(
            "historical/live cutoff frozen before X-13 batch capture"
        ),
    )
    value, manifest = exchange(request)
    required = {
        "market_positions_last_updated_ts",
        "market_settled_ts",
        "orders_updated_ts",
        "trades_created_ts",
    }
    if type(value) is not dict or set(value) != required:
        raise X13CaptureError(
            "Kalshi historical cutoff fields do not match the contract"
        )
    cutoff = KalshiHistoricalCutoff(
        market_positions_last_updated_ts=_parse_utc_cutoff(
            value["market_positions_last_updated_ts"],
            "market_positions_last_updated_ts",
        ),
        market_settled_ts=_parse_utc_cutoff(
            value["market_settled_ts"],
            "market_settled_ts",
        ),
        orders_updated_ts=_parse_utc_cutoff(
            value["orders_updated_ts"],
            "orders_updated_ts",
        ),
        trades_created_ts=_parse_utc_cutoff(
            value["trades_created_ts"],
            "trades_created_ts",
        ),
        manifest_sha256=manifest.manifest_sha256,
        response_received_at_utc=manifest.response_received_at_utc,
    )
    historical_end = min(
        cutoff.market_settled_ts,
        cutoff.trades_created_ts,
    ).timestamp()
    if any(plan.end_ts >= historical_end for plan in plans):
        raise X13CaptureError(
            "capture window is not wholly before the frozen historical cutoff"
        )
    return cutoff


def _cursor_page(
    value: Any,
    *,
    resource: str,
) -> tuple[list[Any], str]:
    if type(value) is not dict or set(value) != {resource, "cursor"}:
        raise X13CaptureError(
            f"Kalshi {resource} page fields do not match the contract"
        )
    items = value[resource]
    cursor = value["cursor"]
    if type(items) is not list or type(cursor) is not str:
        raise X13CaptureError(
            f"Kalshi {resource} page is malformed"
        )
    return items, cursor


def _capture_kalshi_markets(
    plan: GameCapturePlan,
    *,
    exchange: Callable[[CaptureRequest], Any],
    max_pages: int,
) -> int:
    groups: dict[str, list[CaptureTarget]] = {}
    for target in plan.targets:
        if target.venue != "kalshi":
            continue
        series_ticker = _nonempty_string(
            target.kalshi_series_ticker,
            "kalshi_series_ticker",
        )
        _nonempty_string(
            target.kalshi_event_ticker,
            "kalshi_event_ticker",
        )
        groups.setdefault(series_ticker, []).append(target)
    return sum(
        _capture_kalshi_market_group(
            plan,
            series_ticker=series_ticker,
            targets=tuple(groups[series_ticker]),
            exchange=exchange,
            max_pages=max_pages,
        )
        for series_ticker in sorted(groups)
    )


def _capture_kalshi_market_group(
    plan: GameCapturePlan,
    *,
    series_ticker: str,
    targets: tuple[CaptureTarget, ...],
    exchange: Callable[[CaptureRequest], Any],
    max_pages: int,
) -> int:
    request_cursor: str | None = None
    seen_cursors: set[str] = set()
    observed: dict[str, bytes] = {}
    all_observed: dict[str, bytes] = {}
    expected = {
        target.venue_market_id: target for target in targets
    }
    expected_events = {
        target.kalshi_event_ticker for target in targets
    }
    for _ in range(max_pages):
        params: dict[str, object] = {
            "limit": KALSHI_PAGE_LIMIT,
            "series_ticker": series_ticker,
        }
        if request_cursor is not None:
            params["cursor"] = request_cursor
        request = _request(
            game_id=plan.game_id,
            source="kalshi",
            resource="kalshi_markets",
            scope_id=series_ticker,
            url=f"{KALSHI_API_ROOT}/historical/markets",
            params=params,
            source_cursor=(
                f"request_cursor:{request_cursor or 'ROOT'}"
            ),
            coverage=(
                "complete historical market cursor chain;"
                f"series={series_ticker};"
                f"target_events={','.join(sorted(expected_events))}"
            ),
        )
        items, response_cursor = _cursor_page(
            exchange(request),
            resource="markets",
        )
        for index, market in enumerate(items):
            if type(market) is not dict:
                raise X13CaptureError(
                    f"Kalshi market[{index}] must be an object"
                )
            ticker = market.get("ticker")
            if (
                type(ticker) is not str
                or _TICKER_RE.fullmatch(ticker) is None
            ):
                raise X13CaptureError(
                    "Kalshi market lacks a stable ticker"
                )
            reported_series = market.get("series_ticker")
            if reported_series not in {None, series_ticker}:
                raise X13CaptureError(
                    "Kalshi market escaped requested series identity"
                )
            market_event = market.get("event_ticker")
            if (
                type(market_event) is not str
                or _TICKER_RE.fullmatch(market_event) is None
                or not market_event.startswith(f"{series_ticker}-")
            ):
                raise X13CaptureError(
                    "Kalshi market event identity is malformed"
                )
            canonical = _canonical_bytes(market)
            prior = all_observed.get(ticker)
            if prior is not None and prior != canonical:
                raise X13CaptureError(
                    "Kalshi markets contain conflicting duplicate ticker"
                )
            all_observed[ticker] = canonical
            if market_event in expected_events:
                target = expected.get(ticker)
                if (
                    target is not None
                    and market.get("title") != target.venue_title
                ):
                    raise X13CaptureError(
                        "Kalshi market title is ambiguous or changed"
                    )
                observed[ticker] = canonical
        if response_cursor == "":
            if set(observed) != set(expected):
                raise X13CaptureError(
                    "Kalshi complete market metadata does not match inventory"
                )
            return len(observed)
        if (
            response_cursor == request_cursor
            or response_cursor in seen_cursors
        ):
            raise X13CaptureError(
                "Kalshi markets returned a repeated cursor"
            )
        seen_cursors.add(response_cursor)
        request_cursor = response_cursor
    raise X13CaptureError(
        "Kalshi markets terminal cursor proof is missing"
    )


def _validate_polymarket_trade(
    trade: Any,
    *,
    condition_id: str,
    start_ts: int,
    end_ts: int,
    index: int,
) -> None:
    if type(trade) is not dict:
        raise X13CaptureError(
            f"Polymarket trade[{index}] must be an object"
        )
    if trade.get("conditionId") != condition_id:
        raise X13CaptureError(
            f"Polymarket trade[{index}] condition mismatch"
        )
    timestamp = trade.get("timestamp")
    if (
        type(timestamp) is not int
        or not start_ts <= timestamp <= end_ts
    ):
        raise X13CaptureError(
            f"Polymarket trade[{index}] is outside its window"
        )


def _capture_polymarket_window(
    plan: GameCapturePlan,
    condition_id: str,
    start_ts: int,
    end_ts: int,
    *,
    exchange: Callable[[CaptureRequest], Any],
) -> tuple[list[PolymarketTerminalWindow], int]:
    params = {
        "market": condition_id,
        "start": start_ts,
        "end": end_ts,
        "limit": POLYMARKET_TRADE_LIMIT,
        "offset": 0,
    }
    request = _request(
        game_id=plan.game_id,
        source="polymarket",
        resource="polymarket_trades",
        scope_id=condition_id,
        url=DATA_API_TRADES_URL,
        params=params,
        source_cursor=(
            f"condition:{condition_id};start:{start_ts};"
            f"end:{end_ts};offset:0"
        ),
        coverage=(
            "condition-scoped inclusive trade time window;"
            f"condition={condition_id};start={start_ts};end={end_ts}"
        ),
    )
    value, manifest_sha256 = exchange(request)
    if type(value) is not list:
        raise X13CaptureError(
            "Polymarket trades response must be an array"
        )
    if len(value) > POLYMARKET_TRADE_LIMIT:
        raise X13CaptureError(
            "Polymarket trades response exceeded the request limit"
        )
    for index, trade in enumerate(value):
        _validate_polymarket_trade(
            trade,
            condition_id=condition_id,
            start_ts=start_ts,
            end_ts=end_ts,
            index=index,
        )
    if len(value) < POLYMARKET_TRADE_LIMIT:
        return (
            [
                PolymarketTerminalWindow(
                    condition_id=condition_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    item_count=len(value),
                    manifest_sha256=manifest_sha256,
                )
            ],
            len(value),
        )
    if start_ts == end_ts:
        raise X13CaptureError(
            "Polymarket 10,000-row one-second leaf has no terminal proof"
        )
    midpoint = (start_ts + end_ts) // 2
    left, left_count = _capture_polymarket_window(
        plan,
        condition_id,
        start_ts,
        midpoint,
        exchange=exchange,
    )
    right, right_count = _capture_polymarket_window(
        plan,
        condition_id,
        midpoint + 1,
        end_ts,
        exchange=exchange,
    )
    return left + right, left_count + right_count


def _capture_kalshi_trades(
    plan: GameCapturePlan,
    target: CaptureTarget,
    *,
    exchange: Callable[[CaptureRequest], Any],
    max_pages: int,
) -> KalshiTradeProof:
    request_cursor: str | None = None
    seen_cursors: set[str] = set()
    canonical_by_id: dict[str, bytes] = {}
    raw_count = 0
    exact_duplicates = 0
    for page_index in range(max_pages):
        params: dict[str, object] = {
            "limit": KALSHI_PAGE_LIMIT,
            "ticker": target.venue_market_id,
            "min_ts": plan.start_ts,
            "max_ts": plan.end_ts,
        }
        if request_cursor is not None:
            params["cursor"] = request_cursor
        request = _request(
            game_id=plan.game_id,
            source="kalshi",
            resource="kalshi_trades",
            scope_id=target.venue_market_id,
            url=f"{KALSHI_API_ROOT}/historical/trades",
            params=params,
            source_cursor=(
                f"ticker:{target.venue_market_id};"
                f"request_cursor:{request_cursor or 'ROOT'}"
            ),
            coverage=(
                "complete stable-ID historical trade cursor chain;"
                f"ticker={target.venue_market_id};"
                f"start={plan.start_ts};end={plan.end_ts}"
            ),
        )
        items, response_cursor = _cursor_page(
            exchange(request)[0],
            resource="trades",
        )
        raw_count += len(items)
        for index, trade in enumerate(items):
            if type(trade) is not dict:
                raise X13CaptureError(
                    f"Kalshi trade[{index}] must be an object"
                )
            trade_id = trade.get("trade_id")
            if type(trade_id) is not str or not trade_id:
                raise X13CaptureError(
                    "Kalshi trade lacks stable trade_id"
                )
            if trade.get("ticker") != target.venue_market_id:
                raise X13CaptureError(
                    "Kalshi trade escaped requested ticker"
                )
            canonical = _canonical_bytes(trade)
            prior = canonical_by_id.get(trade_id)
            if prior is not None:
                if prior != canonical:
                    raise X13CaptureError(
                        "Kalshi trade has conflicting duplicate stable ID"
                    )
                exact_duplicates += 1
            else:
                canonical_by_id[trade_id] = canonical
        if response_cursor == "":
            return KalshiTradeProof(
                ticker=target.venue_market_id,
                page_count=page_index + 1,
                raw_item_count=raw_count,
                unique_item_count=len(canonical_by_id),
                exact_duplicate_count=exact_duplicates,
                terminal_cursor="",
            )
        if (
            response_cursor == request_cursor
            or response_cursor in seen_cursors
        ):
            raise X13CaptureError(
                "Kalshi trades returned a repeated cursor"
            )
        seen_cursors.add(response_cursor)
        request_cursor = response_cursor
    raise X13CaptureError(
        "Kalshi trades terminal cursor proof is missing"
    )


def _capture_kalshi_candlesticks(
    plan: GameCapturePlan,
    target: CaptureTarget,
    *,
    exchange: Callable[[CaptureRequest], Any],
) -> KalshiCandlestickCapture:
    params = {
        "start_ts": plan.start_ts,
        "end_ts": plan.end_ts,
        "period_interval": 1,
    }
    request = _request(
        game_id=plan.game_id,
        source="kalshi",
        resource="kalshi_candlesticks",
        scope_id=target.venue_market_id,
        url=(
            f"{KALSHI_API_ROOT}/historical/markets/"
            f"{target.venue_market_id}/candlesticks"
        ),
        params=params,
        source_cursor=(
            f"ticker:{target.venue_market_id};"
            f"start:{plan.start_ts};end:{plan.end_ts};period:1"
        ),
        coverage=(
            "official historical one-minute candlesticks;"
            f"ticker={target.venue_market_id};"
            f"start={plan.start_ts};end={plan.end_ts}"
        ),
    )
    value, manifest_sha256 = exchange(request)
    if type(value) is not dict or set(value) != {
        "ticker",
        "candlesticks",
    }:
        raise X13CaptureError(
            "Kalshi candlestick response fields do not match"
        )
    if value["ticker"] != target.venue_market_id:
        raise X13CaptureError(
            "Kalshi candlestick ticker is not exact"
        )
    candles = value["candlesticks"]
    if type(candles) is not list:
        raise X13CaptureError(
            "Kalshi candlesticks must be an array"
        )
    end_periods: list[int] = []
    for index, candle in enumerate(candles):
        if type(candle) is not dict:
            raise X13CaptureError(
                f"Kalshi candlestick[{index}] must be an object"
            )
        end_period_ts = candle.get("end_period_ts")
        if (
            type(end_period_ts) is not int
            or not plan.start_ts
            <= end_period_ts
            <= plan.end_ts
        ):
            raise X13CaptureError(
                "Kalshi candlestick end time escaped request window"
            )
        if end_periods and end_period_ts <= end_periods[-1]:
            raise X13CaptureError(
                "Kalshi candlestick end times are not increasing"
            )
        end_periods.append(end_period_ts)
    return KalshiCandlestickCapture(
        ticker=target.venue_market_id,
        candlestick_count=len(candles),
        first_end_period_ts=(
            end_periods[0] if end_periods else None
        ),
        last_end_period_ts=(
            end_periods[-1] if end_periods else None
        ),
        manifest_sha256=manifest_sha256,
    )


def _capture_game(
    plan: GameCapturePlan,
    *,
    requester: Callable[[CaptureRequest], CaptureHttpResponse],
    manifest_writer: Callable[
        [
            CaptureRequest,
            CaptureHttpResponse,
            Path,
            Path,
            datetime,
        ],
        ImmutableRequestManifest,
    ],
    raw_store_root: Path,
    program_root: Path,
    response_clock: Callable[[], datetime],
    rate_limiter: HostRateLimiter,
    kalshi_max_pages: int,
) -> GameCaptureResult:
    manifests: list[ImmutableRequestManifest] = []

    def exchange(request: CaptureRequest) -> tuple[Any, str]:
        value = _exchange(
            request,
            requester=requester,
            manifest_writer=manifest_writer,
            raw_store_root=raw_store_root,
            program_root=program_root,
            response_clock=response_clock,
            rate_limiter=rate_limiter,
            manifests=manifests,
        )
        return value, manifests[-1].manifest_sha256

    polymarket_market_count = _capture_gamma_metadata(
        plan,
        exchange=lambda request: exchange(request)[0],
    )
    kalshi_market_count = _capture_kalshi_markets(
        plan,
        exchange=lambda request: exchange(request)[0],
        max_pages=kalshi_max_pages,
    )

    terminal_windows: list[PolymarketTerminalWindow] = []
    polymarket_trade_count = 0
    for target in sorted(
        (
            target
            for target in plan.targets
            if target.venue == "polymarket"
        ),
        key=lambda value: value.condition_id or "",
    ):
        windows, count = _capture_polymarket_window(
            plan,
            target.condition_id or "",
            plan.start_ts,
            plan.end_ts,
            exchange=exchange,
        )
        terminal_windows.extend(windows)
        polymarket_trade_count += count

    trade_proofs = tuple(
        _capture_kalshi_trades(
            plan,
            target,
            exchange=exchange,
            max_pages=kalshi_max_pages,
        )
        for target in sorted(
            (
                target
                for target in plan.targets
                if target.venue == "kalshi"
            ),
            key=lambda value: value.venue_market_id,
        )
    )
    candlesticks = tuple(
        _capture_kalshi_candlesticks(
            plan,
            target,
            exchange=exchange,
        )
        for target in sorted(
            (
                target
                for target in plan.targets
                if target.venue == "kalshi"
                and target.analysis_eligible
            ),
            key=lambda value: value.venue_market_id,
        )
    )
    return GameCaptureResult(
        game_id=plan.game_id,
        manifests=tuple(manifests),
        polymarket_market_count=polymarket_market_count,
        polymarket_trade_count=polymarket_trade_count,
        polymarket_terminal_windows=tuple(terminal_windows),
        kalshi_market_count=kalshi_market_count,
        kalshi_trade_proofs=trade_proofs,
        kalshi_candlesticks=candlesticks,
    )


def _target_payload(target: CaptureTarget) -> dict[str, object]:
    return {
        "analysis_eligible": target.analysis_eligible,
        "condition_id": target.condition_id,
        "contract_id": target.contract_id,
        "dependency_game_ids": list(target.dependency_game_ids),
        "family": target.family,
        "kalshi_event_ticker": target.kalshi_event_ticker,
        "kalshi_series_ticker": target.kalshi_series_ticker,
        "kind": target.kind,
        "venue": target.venue,
        "venue_market_id": target.venue_market_id,
        "venue_title": target.venue_title,
    }


def _plan_payload(plan: GameCapturePlan) -> dict[str, object]:
    return {
        "end_ts": plan.end_ts,
        "game_id": plan.game_id,
        "polymarket_event_id": plan.polymarket_event_id,
        "polymarket_event_slug": plan.polymarket_event_slug,
        "polymarket_event_title": plan.polymarket_event_title,
        "start_ts": plan.start_ts,
        "targets": [_target_payload(target) for target in plan.targets],
    }


def _plans_sha256(plans: Sequence[GameCapturePlan]) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "capture_plans": [_plan_payload(plan) for plan in plans],
                "version": "nfl-x13-capture-plan-v1",
            }
        )
    )


def _raw_relative_path(path: Path, raw_store_root: Path) -> str:
    try:
        relative = path.relative_to(raw_store_root)
    except ValueError as error:
        raise X13CaptureError(
            "capture manifest path escapes raw_store_root"
        ) from error
    pure = PurePosixPath(relative.as_posix())
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise X13CaptureError("capture manifest path is unsafe")
    return pure.as_posix()


def _manifest_receipt_payload(
    manifest: ImmutableRequestManifest,
    *,
    raw_store_root: Path,
) -> dict[str, object]:
    return {
        "byte_length": manifest.byte_length,
        "manifest_path": _raw_relative_path(
            manifest.manifest_path,
            raw_store_root,
        ),
        "manifest_sha256": manifest.manifest_sha256,
        "object_path": _raw_relative_path(
            manifest.object_path,
            raw_store_root,
        ),
        "object_sha256": manifest.object_sha256,
        "request_sha256": manifest.request_sha256,
        "response_received_at_utc": _utc_response_text(
            manifest.response_received_at_utc
        ),
    }


def _cutoff_payload(
    cutoff: KalshiHistoricalCutoff,
) -> dict[str, object]:
    return {
        "manifest_sha256": cutoff.manifest_sha256,
        "market_positions_last_updated_ts": _utc_response_text(
            cutoff.market_positions_last_updated_ts
        ),
        "market_settled_ts": _utc_response_text(
            cutoff.market_settled_ts
        ),
        "orders_updated_ts": _utc_response_text(
            cutoff.orders_updated_ts
        ),
        "response_received_at_utc": _utc_response_text(
            cutoff.response_received_at_utc
        ),
        "trades_created_ts": _utc_response_text(
            cutoff.trades_created_ts
        ),
    }


def _game_result_payload(
    game: GameCaptureResult,
) -> dict[str, object]:
    return {
        "game_id": game.game_id,
        "kalshi_candlesticks": [
            {
                "candlestick_count": value.candlestick_count,
                "first_end_period_ts": value.first_end_period_ts,
                "last_end_period_ts": value.last_end_period_ts,
                "manifest_sha256": value.manifest_sha256,
                "period_interval": value.period_interval,
                "ticker": value.ticker,
            }
            for value in game.kalshi_candlesticks
        ],
        "kalshi_market_count": game.kalshi_market_count,
        "kalshi_trade_proofs": [
            {
                "exact_duplicate_count": value.exact_duplicate_count,
                "page_count": value.page_count,
                "raw_item_count": value.raw_item_count,
                "terminal_cursor": value.terminal_cursor,
                "terminal_proof": value.terminal_proof,
                "ticker": value.ticker,
                "unique_item_count": value.unique_item_count,
            }
            for value in game.kalshi_trade_proofs
        ],
        "manifest_sha256s": [
            value.manifest_sha256 for value in game.manifests
        ],
        "polymarket_market_count": game.polymarket_market_count,
        "polymarket_terminal_windows": [
            {
                "condition_id": value.condition_id,
                "end_ts": value.end_ts,
                "item_count": value.item_count,
                "manifest_sha256": value.manifest_sha256,
                "start_ts": value.start_ts,
                "terminal_proof": value.terminal_proof,
            }
            for value in game.polymarket_terminal_windows
        ],
        "polymarket_trade_count": game.polymarket_trade_count,
    }


def _capture_receipt_payload(
    receipt: CaptureReceipt,
) -> dict[str, object]:
    preflight = receipt.preflight
    return {
        "game_ids": [game.game_id for game in receipt.games],
        "games": [
            _game_result_payload(game) for game in receipt.games
        ],
        "kalshi_cutoff": _cutoff_payload(receipt.kalshi_cutoff),
        "manifests": [
            _manifest_receipt_payload(
                manifest,
                raw_store_root=preflight.raw_store_root,
            )
            for manifest in receipt.manifests
        ],
        "plan_sha256": receipt.plan_sha256,
        "preflight": {
            "available_free_bytes": preflight.available_free_bytes,
            "derived_p95_bytes": preflight.derived_p95_bytes,
            "planned_raw_p95_bytes": preflight.planned_raw_p95_bytes,
            "required_free_bytes": preflight.required_free_bytes,
            "reserve_bytes": preflight.reserve_bytes,
            "temporary_headroom_bytes": (
                preflight.temporary_headroom_bytes
            ),
        },
        "raw_manifest_sha256s": sorted(
            manifest.manifest_sha256
            for manifest in receipt.manifests
        ),
        "universe_id": receipt.universe_id,
        "version": "nfl-x13-capture-receipt-v1",
    }


def _capture_pointer(receipt: CaptureReceipt) -> CapturePointer:
    game_ids = tuple(game.game_id for game in receipt.games)
    manifests = tuple(
        sorted(
            manifest.manifest_sha256
            for manifest in receipt.manifests
        )
    )
    material = {
        "game_ids": list(game_ids),
        "plan_sha256": receipt.plan_sha256,
        "raw_manifest_sha256s": list(manifests),
        "receipt_sha256": receipt.receipt_sha256,
        "universe_id": receipt.universe_id,
        "version": "nfl-x13-capture-pointer-v2",
    }
    return CapturePointer(
        capture_sha256=_sha256(_canonical_bytes(material)),
        universe_id=receipt.universe_id,
        plan_sha256=receipt.plan_sha256,
        receipt_sha256=receipt.receipt_sha256,
        game_ids=game_ids,
        raw_manifest_sha256s=manifests,
    )


def _publish_content_addressed_json(
    *,
    directory: Path,
    identity_sha256: str,
    payload: bytes,
    temporary_prefix: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = (
        directory
        / f"{identity_sha256.removeprefix('sha256:')}.json"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=temporary_prefix,
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_name, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise X13CaptureError(
                    "content-addressed capture metadata hash collision"
                )
        directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return target
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _publish_receipt(
    receipt: CaptureReceipt,
    raw_store_root: Path,
) -> Path:
    return _publish_content_addressed_json(
        directory=raw_store_root / "capture-receipts",
        identity_sha256=receipt.receipt_sha256,
        payload=_canonical_bytes(receipt.canonical_payload) + b"\n",
        temporary_prefix=".capture-receipt.",
    )


def _publish_pointer(
    pointer: CapturePointer,
    raw_store_root: Path,
) -> Path:
    return _publish_content_addressed_json(
        directory=raw_store_root / "capture-pointers",
        identity_sha256=pointer.capture_sha256,
        payload=_canonical_bytes(pointer.canonical_payload) + b"\n",
        temporary_prefix=".capture-pointer.",
    )


def _receipt_mapping(
    value: object,
    *,
    fields: set[str],
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise X13CaptureError(
            f"{context} fields do not match the frozen receipt schema"
        )
    return value


def _receipt_sequence(value: object, context: str) -> list[Any]:
    if type(value) is not list:
        raise X13CaptureError(f"{context} must be an array")
    return value


def _receipt_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise X13CaptureError(f"{context} must be a canonical SHA-256")
    return value


def _receipt_integer(
    value: object,
    context: str,
    *,
    positive: bool = False,
) -> int:
    if (
        type(value) is not int
        or value < (1 if positive else 0)
    ):
        raise X13CaptureError(f"{context} has an invalid integer")
    return value


def _receipt_optional_integer(
    value: object,
    context: str,
) -> int | None:
    if value is None:
        return None
    return _receipt_integer(value, context)


def _receipt_time(value: object, context: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise X13CaptureError(f"{context} must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise X13CaptureError(f"{context} is malformed") from error
    return _require_utc_response_time(parsed)


def _receipt_raw_path(
    raw_store_root: Path,
    value: object,
    context: str,
) -> Path:
    if type(value) is not str:
        raise X13CaptureError(f"{context} must be a relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise X13CaptureError(f"{context} is unsafe")
    return raw_store_root.joinpath(*relative.parts)


def _read_canonical_capture_json(
    path: Path,
    *,
    context: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise X13CaptureError(f"{context} is not a regular file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise X13CaptureError(f"{context} is unreadable") from error
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise X13CaptureError(f"{context} is not canonical JSON")
    value = _strict_json(raw[:-1], context=context)
    if type(value) is not dict or raw != _canonical_bytes(value) + b"\n":
        raise X13CaptureError(f"{context} is not canonical JSON")
    return value


def _load_manifest_receipts(
    rows: object,
    *,
    raw_store_root: Path,
    program_root: Path,
) -> tuple[ImmutableRequestManifest, ...]:
    values: list[ImmutableRequestManifest] = []
    manifest_ids: set[str] = set()
    request_ids: set[str] = set()
    for index, item in enumerate(
        _receipt_sequence(rows, "capture receipt manifests")
    ):
        row = _receipt_mapping(
            item,
            fields={
                "byte_length",
                "manifest_path",
                "manifest_sha256",
                "object_path",
                "object_sha256",
                "request_sha256",
                "response_received_at_utc",
            },
            context=f"capture receipt manifest[{index}]",
        )
        manifest_sha256 = _receipt_sha256(
            row["manifest_sha256"],
            "manifest_sha256",
        )
        request_sha256 = _receipt_sha256(
            row["request_sha256"],
            "request_sha256",
        )
        if (
            manifest_sha256 in manifest_ids
            or request_sha256 in request_ids
        ):
            raise X13CaptureError(
                "capture receipt contains duplicate manifest identity"
            )
        manifest_ids.add(manifest_sha256)
        request_ids.add(request_sha256)
        object_sha256 = _receipt_sha256(
            row["object_sha256"],
            "object_sha256",
        )
        manifest_path = _receipt_raw_path(
            raw_store_root,
            row["manifest_path"],
            "manifest_path",
        )
        object_path = _receipt_raw_path(
            raw_store_root,
            row["object_path"],
            "object_path",
        )
        byte_length = _receipt_integer(
            row["byte_length"],
            "byte_length",
        )
        try:
            verified = verify_static_object(
                manifest_path,
                store_root=raw_store_root,
                program_root=program_root,
            )
        except StaticStoreError as error:
            raise X13CaptureError(
                "capture receipt raw manifest verification failed"
            ) from error
        if (
            verified.manifest_path != manifest_path
            or verified.object_path != object_path
            or verified.manifest.manifest_sha256 != manifest_sha256
            or verified.manifest.object_sha256 != object_sha256
            or verified.manifest.byte_length != byte_length
        ):
            raise X13CaptureError(
                "capture receipt does not match verified raw evidence"
            )
        values.append(
            ImmutableRequestManifest(
                request_sha256=request_sha256,
                object_sha256=object_sha256,
                manifest_sha256=manifest_sha256,
                byte_length=byte_length,
                object_path=object_path,
                manifest_path=manifest_path,
                response_received_at_utc=_receipt_time(
                    row["response_received_at_utc"],
                    "response_received_at_utc",
                ),
            )
        )
    if not values:
        raise X13CaptureError("capture receipt has no raw manifests")
    return tuple(values)


def _load_game_result(
    value: object,
    *,
    manifest_by_sha: Mapping[str, ImmutableRequestManifest],
) -> GameCaptureResult:
    row = _receipt_mapping(
        value,
        fields={
            "game_id",
            "kalshi_candlesticks",
            "kalshi_market_count",
            "kalshi_trade_proofs",
            "manifest_sha256s",
            "polymarket_market_count",
            "polymarket_terminal_windows",
            "polymarket_trade_count",
        },
        context="capture receipt game",
    )
    game_id = _nonempty_string(row["game_id"], "receipt game_id")
    manifest_ids = _receipt_sequence(
        row["manifest_sha256s"],
        f"{game_id}.manifest_sha256s",
    )
    if len(manifest_ids) != len(set(manifest_ids)):
        raise X13CaptureError(
            f"{game_id} receipt contains duplicate manifest references"
        )
    try:
        manifests = tuple(
            manifest_by_sha[
                _receipt_sha256(value, f"{game_id}.manifest_sha256")
            ]
            for value in manifest_ids
        )
    except KeyError as error:
        raise X13CaptureError(
            f"{game_id} receipt references an unknown raw manifest"
        ) from error

    windows: list[PolymarketTerminalWindow] = []
    for item in _receipt_sequence(
        row["polymarket_terminal_windows"],
        f"{game_id}.polymarket_terminal_windows",
    ):
        proof = _receipt_mapping(
            item,
            fields={
                "condition_id",
                "end_ts",
                "item_count",
                "manifest_sha256",
                "start_ts",
                "terminal_proof",
            },
            context=f"{game_id}.polymarket_terminal_window",
        )
        manifest_sha = _receipt_sha256(
            proof["manifest_sha256"],
            "terminal window manifest_sha256",
        )
        if manifest_sha not in manifest_by_sha:
            raise X13CaptureError(
                f"{game_id} terminal window has no raw manifest"
            )
        start_ts = _receipt_integer(proof["start_ts"], "start_ts")
        end_ts = _receipt_integer(proof["end_ts"], "end_ts")
        item_count = _receipt_integer(
            proof["item_count"],
            "item_count",
        )
        if (
            end_ts < start_ts
            or item_count >= POLYMARKET_TRADE_LIMIT
            or proof["terminal_proof"] != "unsaturated_time_window"
        ):
            raise X13CaptureError(
                f"{game_id} Polymarket terminal proof is not unsaturated"
            )
        windows.append(
            PolymarketTerminalWindow(
                condition_id=_nonempty_string(
                    proof["condition_id"],
                    "condition_id",
                ),
                start_ts=start_ts,
                end_ts=end_ts,
                item_count=item_count,
                manifest_sha256=manifest_sha,
            )
        )

    trade_proofs: list[KalshiTradeProof] = []
    for item in _receipt_sequence(
        row["kalshi_trade_proofs"],
        f"{game_id}.kalshi_trade_proofs",
    ):
        proof = _receipt_mapping(
            item,
            fields={
                "exact_duplicate_count",
                "page_count",
                "raw_item_count",
                "terminal_cursor",
                "terminal_proof",
                "ticker",
                "unique_item_count",
            },
            context=f"{game_id}.kalshi_trade_proof",
        )
        raw_count = _receipt_integer(
            proof["raw_item_count"],
            "raw_item_count",
        )
        unique_count = _receipt_integer(
            proof["unique_item_count"],
            "unique_item_count",
        )
        duplicate_count = _receipt_integer(
            proof["exact_duplicate_count"],
            "exact_duplicate_count",
        )
        if (
            raw_count != unique_count + duplicate_count
            or proof["terminal_cursor"] != ""
            or proof["terminal_proof"] != "explicit_empty_cursor"
        ):
            raise X13CaptureError(
                f"{game_id} Kalshi terminal cursor proof is invalid"
            )
        trade_proofs.append(
            KalshiTradeProof(
                ticker=_nonempty_string(proof["ticker"], "ticker"),
                page_count=_receipt_integer(
                    proof["page_count"],
                    "page_count",
                    positive=True,
                ),
                raw_item_count=raw_count,
                unique_item_count=unique_count,
                exact_duplicate_count=duplicate_count,
                terminal_cursor="",
            )
        )

    candlesticks: list[KalshiCandlestickCapture] = []
    for item in _receipt_sequence(
        row["kalshi_candlesticks"],
        f"{game_id}.kalshi_candlesticks",
    ):
        proof = _receipt_mapping(
            item,
            fields={
                "candlestick_count",
                "first_end_period_ts",
                "last_end_period_ts",
                "manifest_sha256",
                "period_interval",
                "ticker",
            },
            context=f"{game_id}.kalshi_candlestick",
        )
        manifest_sha = _receipt_sha256(
            proof["manifest_sha256"],
            "candlestick manifest_sha256",
        )
        if manifest_sha not in manifest_by_sha:
            raise X13CaptureError(
                f"{game_id} candlestick proof has no raw manifest"
            )
        count = _receipt_integer(
            proof["candlestick_count"],
            "candlestick_count",
        )
        first = _receipt_optional_integer(
            proof["first_end_period_ts"],
            "first_end_period_ts",
        )
        last = _receipt_optional_integer(
            proof["last_end_period_ts"],
            "last_end_period_ts",
        )
        if (
            proof["period_interval"] != 1
            or (count == 0 and (first is not None or last is not None))
            or (
                count > 0
                and (
                    first is None
                    or last is None
                    or first > last
                )
            )
        ):
            raise X13CaptureError(
                f"{game_id} candlestick proof is invalid"
            )
        candlesticks.append(
            KalshiCandlestickCapture(
                ticker=_nonempty_string(proof["ticker"], "ticker"),
                candlestick_count=count,
                first_end_period_ts=first,
                last_end_period_ts=last,
                manifest_sha256=manifest_sha,
            )
        )

    polymarket_trade_count = _receipt_integer(
        row["polymarket_trade_count"],
        "polymarket_trade_count",
    )
    if sum(value.item_count for value in windows) != polymarket_trade_count:
        raise X13CaptureError(
            f"{game_id} Polymarket terminal counts do not reconcile"
        )
    return GameCaptureResult(
        game_id=game_id,
        manifests=manifests,
        polymarket_market_count=_receipt_integer(
            row["polymarket_market_count"],
            "polymarket_market_count",
        ),
        polymarket_trade_count=polymarket_trade_count,
        polymarket_terminal_windows=tuple(windows),
        kalshi_market_count=_receipt_integer(
            row["kalshi_market_count"],
            "kalshi_market_count",
        ),
        kalshi_trade_proofs=tuple(trade_proofs),
        kalshi_candlesticks=tuple(candlesticks),
    )


def load_x13_capture(
    pointer_path: str | Path,
    *,
    plans: Sequence[GameCapturePlan],
    universe_id: str,
    raw_store_root: str | Path,
    program_root: str | Path,
) -> BatchCaptureResult:
    """Reopen one complete capture from immutable raw evidence and proofs."""

    raw_root = _absolute_path(raw_store_root, "raw_store_root")
    program = _absolute_path(program_root, "program_root")
    pointer_file = _absolute_path(pointer_path, "pointer_path")
    plans = tuple(plans)
    for plan in plans:
        _validate_plan(
            plan,
            known_game_ids=frozenset(X13_GAME_IDS),
        )
    if (
        universe_id != X13_FROZEN_UNIVERSE_ID
        or tuple(plan.game_id for plan in plans)
        != X13_FROZEN_CAPTURE_GAME_IDS
        or len(plans) != len(X13_FROZEN_CAPTURE_GAME_IDS)
        or sum(len(plan.targets) for plan in plans)
        != X13_FROZEN_CAPTURE_TARGET_COUNT
    ):
        raise X13CaptureError(
            "capture loader requires the exact frozen universe"
        )
    plan_sha256 = _plans_sha256(plans)

    pointer_value = _receipt_mapping(
        _read_canonical_capture_json(
            pointer_file,
            context="capture pointer",
        ),
        fields={
            "capture_sha256",
            "game_ids",
            "plan_sha256",
            "raw_manifest_sha256s",
            "receipt_sha256",
            "universe_id",
            "version",
        },
        context="capture pointer",
    )
    if pointer_value["version"] != "nfl-x13-capture-pointer-v2":
        raise X13CaptureError("capture pointer version is not canonical")
    capture_sha256 = _receipt_sha256(
        pointer_value["capture_sha256"],
        "capture_sha256",
    )
    receipt_sha256 = _receipt_sha256(
        pointer_value["receipt_sha256"],
        "receipt_sha256",
    )
    raw_manifest_ids = tuple(
        _receipt_sha256(value, "raw_manifest_sha256")
        for value in _receipt_sequence(
            pointer_value["raw_manifest_sha256s"],
            "raw_manifest_sha256s",
        )
    )
    game_ids = tuple(
        _nonempty_string(value, "pointer game_id")
        for value in _receipt_sequence(
            pointer_value["game_ids"],
            "pointer game_ids",
        )
    )
    pointer_material = {
        key: value
        for key, value in pointer_value.items()
        if key != "capture_sha256"
    }
    if (
        _sha256(_canonical_bytes(pointer_material)) != capture_sha256
        or pointer_file.name
        != f"{capture_sha256.removeprefix('sha256:')}.json"
        or pointer_value["universe_id"] != universe_id
        or pointer_value["plan_sha256"] != plan_sha256
        or game_ids != X13_FROZEN_CAPTURE_GAME_IDS
        or raw_manifest_ids != tuple(sorted(set(raw_manifest_ids)))
    ):
        raise X13CaptureError("capture pointer identity is invalid")

    receipt_path = (
        raw_root
        / "capture-receipts"
        / f"{receipt_sha256.removeprefix('sha256:')}.json"
    )
    receipt_value = _receipt_mapping(
        _read_canonical_capture_json(
            receipt_path,
            context="capture receipt",
        ),
        fields={
            "game_ids",
            "games",
            "kalshi_cutoff",
            "manifests",
            "plan_sha256",
            "preflight",
            "raw_manifest_sha256s",
            "universe_id",
            "version",
        },
        context="capture receipt",
    )
    if (
        receipt_value["version"] != "nfl-x13-capture-receipt-v1"
        or _sha256(_canonical_bytes(receipt_value)) != receipt_sha256
        or receipt_value["universe_id"] != universe_id
        or receipt_value["plan_sha256"] != plan_sha256
        or receipt_value["game_ids"]
        != list(X13_FROZEN_CAPTURE_GAME_IDS)
        or receipt_value["raw_manifest_sha256s"]
        != list(raw_manifest_ids)
    ):
        raise X13CaptureError("capture receipt identity is invalid")

    preflight_row = _receipt_mapping(
        receipt_value["preflight"],
        fields={
            "available_free_bytes",
            "derived_p95_bytes",
            "planned_raw_p95_bytes",
            "required_free_bytes",
            "reserve_bytes",
            "temporary_headroom_bytes",
        },
        context="capture receipt preflight",
    )
    planned = _receipt_integer(
        preflight_row["planned_raw_p95_bytes"],
        "planned_raw_p95_bytes",
    )
    derived = _receipt_integer(
        preflight_row["derived_p95_bytes"],
        "derived_p95_bytes",
    )
    temporary = _receipt_integer(
        preflight_row["temporary_headroom_bytes"],
        "temporary_headroom_bytes",
    )
    reserve = _receipt_integer(
        preflight_row["reserve_bytes"],
        "reserve_bytes",
    )
    required = _receipt_integer(
        preflight_row["required_free_bytes"],
        "required_free_bytes",
    )
    if (
        temporary != GIB
        or reserve != 5 * GIB
        or required != planned + derived + temporary + reserve
    ):
        raise X13CaptureError("capture receipt preflight formula is invalid")
    preflight = DiskPreflight(
        raw_store_root=raw_root,
        program_root=program,
        planned_raw_p95_bytes=planned,
        derived_p95_bytes=derived,
        temporary_headroom_bytes=temporary,
        reserve_bytes=reserve,
        required_free_bytes=required,
        available_free_bytes=_receipt_integer(
            preflight_row["available_free_bytes"],
            "available_free_bytes",
        ),
    )

    manifests = _load_manifest_receipts(
        receipt_value["manifests"],
        raw_store_root=raw_root,
        program_root=program,
    )
    manifest_by_sha = {
        value.manifest_sha256: value for value in manifests
    }
    if tuple(sorted(manifest_by_sha)) != raw_manifest_ids:
        raise X13CaptureError(
            "capture receipt manifest set differs from pointer"
        )
    games = tuple(
        _load_game_result(
            value,
            manifest_by_sha=manifest_by_sha,
        )
        for value in _receipt_sequence(
            receipt_value["games"],
            "capture receipt games",
        )
    )
    if (
        tuple(game.game_id for game in games)
        != X13_FROZEN_CAPTURE_GAME_IDS
    ):
        raise X13CaptureError("capture receipt game order is invalid")

    cutoff_row = _receipt_mapping(
        receipt_value["kalshi_cutoff"],
        fields={
            "manifest_sha256",
            "market_positions_last_updated_ts",
            "market_settled_ts",
            "orders_updated_ts",
            "response_received_at_utc",
            "trades_created_ts",
        },
        context="capture receipt Kalshi cutoff",
    )
    cutoff_manifest = _receipt_sha256(
        cutoff_row["manifest_sha256"],
        "cutoff manifest_sha256",
    )
    if cutoff_manifest not in manifest_by_sha:
        raise X13CaptureError("Kalshi cutoff has no raw manifest")
    cutoff = KalshiHistoricalCutoff(
        market_positions_last_updated_ts=_receipt_time(
            cutoff_row["market_positions_last_updated_ts"],
            "market_positions_last_updated_ts",
        ),
        market_settled_ts=_receipt_time(
            cutoff_row["market_settled_ts"],
            "market_settled_ts",
        ),
        orders_updated_ts=_receipt_time(
            cutoff_row["orders_updated_ts"],
            "orders_updated_ts",
        ),
        trades_created_ts=_receipt_time(
            cutoff_row["trades_created_ts"],
            "trades_created_ts",
        ),
        manifest_sha256=cutoff_manifest,
        response_received_at_utc=_receipt_time(
            cutoff_row["response_received_at_utc"],
            "response_received_at_utc",
        ),
    )
    referenced = {
        cutoff_manifest,
        *(
            manifest.manifest_sha256
            for game in games
            for manifest in game.manifests
        ),
    }
    if referenced != set(manifest_by_sha):
        raise X13CaptureError(
            "capture receipt has unreferenced or missing raw manifests"
        )

    receipt = CaptureReceipt(
        universe_id=universe_id,
        plan_sha256=plan_sha256,
        preflight=preflight,
        kalshi_cutoff=cutoff,
        games=games,
        manifests=manifests,
    )
    if (
        receipt.receipt_sha256 != receipt_sha256
        or receipt.canonical_payload != receipt_value
    ):
        raise X13CaptureError(
            "capture receipt does not reconstruct canonically"
        )
    pointer = _capture_pointer(receipt)
    if pointer.canonical_payload != pointer_value:
        raise X13CaptureError(
            "capture pointer does not match reconstructed receipt"
        )
    return BatchCaptureResult(
        preflight=preflight,
        kalshi_cutoff=cutoff,
        games=games,
        manifests=manifests,
        receipt=receipt,
        receipt_path=receipt_path,
        pointer=pointer,
        pointer_path=pointer_file,
    )


def capture_x13_batch(
    plans: Sequence[GameCapturePlan],
    *,
    universe_id: str,
    raw_store_root: str | Path,
    program_root: str | Path,
    planned_raw_p95_bytes: int,
    derived_p95_bytes: int,
    response_clock: Callable[[], datetime] = _system_response_clock,
    requester: (
        Callable[[CaptureRequest], CaptureHttpResponse] | None
    ) = None,
    manifest_writer: (
        Callable[
            [
                CaptureRequest,
                CaptureHttpResponse,
                Path,
                Path,
                datetime,
            ],
            ImmutableRequestManifest,
        ]
        | None
    ) = None,
    pointer_publisher: (
        Callable[[CapturePointer, Path], Path] | None
    ) = None,
    host_requests_per_second: Mapping[str, int | float] = (
        DEFAULT_HOST_REQUESTS_PER_SECOND
    ),
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    kalshi_max_pages: int = 100,
    max_response_bytes: int = 50_000_000,
    retry_delays_seconds: Sequence[int | float] = (
        0.5,
        1,
        2,
        4,
        8,
    ),
) -> BatchCaptureResult:
    """Capture games with four workers; publish only after every proof passes."""

    plans = tuple(plans)
    preflight = preflight_x13_capture(
        plans,
        raw_store_root=raw_store_root,
        program_root=program_root,
        planned_raw_p95_bytes=planned_raw_p95_bytes,
        derived_p95_bytes=derived_p95_bytes,
        disk_usage=disk_usage,
    )
    if (
        _SHA256_RE.fullmatch(universe_id) is None
        or universe_id != X13_FROZEN_UNIVERSE_ID
    ):
        raise X13CaptureError(
            "universe_id does not match the frozen X-13 universe"
        )
    if (
        tuple(plan.game_id for plan in plans)
        != X13_FROZEN_CAPTURE_GAME_IDS
        or len(plans) != len(X13_FROZEN_CAPTURE_GAME_IDS)
        or sum(len(plan.targets) for plan in plans)
        != X13_FROZEN_CAPTURE_TARGET_COUNT
    ):
        raise X13CaptureError(
            "formal capture requires the exact frozen 20 games and targets"
        )
    plan_sha256 = _plans_sha256(plans)
    if not callable(response_clock):
        raise X13CaptureError("response_clock must be callable")
    if type(kalshi_max_pages) is not int or kalshi_max_pages <= 0:
        raise X13CaptureError(
            "kalshi_max_pages must be a positive integer"
        )
    if (
        type(max_response_bytes) is not int
        or max_response_bytes <= 0
    ):
        raise X13CaptureError(
            "max_response_bytes must be a positive integer"
        )
    if (
        not isinstance(retry_delays_seconds, Sequence)
        or isinstance(retry_delays_seconds, (str, bytes, bytearray))
    ):
        raise X13CaptureError("retry_delays_seconds must be a sequence")
    retry_delays = tuple(retry_delays_seconds)
    if any(
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(delay)
        or delay < 0
        or delay > 60
        for delay in retry_delays
    ):
        raise X13CaptureError(
            "retry delays must be finite numbers between zero and 60"
        )
    limiter = HostRateLimiter(host_requests_per_second)
    base_writer = manifest_writer or preserve_capture_response
    minimum_running_free_bytes = (
        preflight.derived_p95_bytes
        + preflight.temporary_headroom_bytes
        + preflight.reserve_bytes
    )
    disk_gate = _RunningDiskReservationGate(
        raw_store_root=preflight.raw_store_root,
        minimum_free_bytes=minimum_running_free_bytes,
        disk_usage=disk_usage,
    )

    def writer(
        request: CaptureRequest,
        response: CaptureHttpResponse,
        raw_root: Path,
        program: Path,
        response_received_at_utc: datetime,
    ) -> ImmutableRequestManifest:
        reservation = disk_gate.acquire(len(response.body))
        try:
            receipt = base_writer(
                request,
                response,
                raw_root,
                program,
                response_received_at_utc,
            )
        except BaseException:
            disk_gate.release(reservation, verify_after_write=False)
            raise
        disk_gate.release(reservation, verify_after_write=True)
        return receipt

    publisher = pointer_publisher or _publish_pointer

    owned_client: httpx.Client | None = None
    active_requester = requester
    if active_requester is None:
        owned_client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=30.0),
        )
        active_requester = _http_response(
            owned_client,
            max_response_bytes,
        )
    base_requester = active_requester
    assert base_requester is not None

    def retrying_requester(
        request: CaptureRequest,
    ) -> CaptureHttpResponse:
        for attempt in range(len(retry_delays) + 1):
            try:
                return base_requester(request)
            except X13TransientCaptureError as error:
                if attempt == len(retry_delays):
                    raise X13CaptureError(
                        "capture request retry budget exhausted: "
                        f"{request.source} {request.resource} "
                        f"{request.scope_id}"
                    ) from error
                time.sleep(float(retry_delays[attempt]))
                limiter.wait_for_url(request.url)
        raise AssertionError("unreachable retry state")

    active_requester = retrying_requester

    results: list[GameCaptureResult] = []
    batch_manifests: list[ImmutableRequestManifest] = []
    try:
        def cutoff_exchange(
            request: CaptureRequest,
        ) -> tuple[Any, ImmutableRequestManifest]:
            value = _exchange(
                request,
                requester=active_requester,
                manifest_writer=writer,
                raw_store_root=preflight.raw_store_root,
                program_root=preflight.program_root,
                response_clock=response_clock,
                rate_limiter=limiter,
                manifests=batch_manifests,
            )
            return value, batch_manifests[-1]

        kalshi_cutoff = _capture_kalshi_cutoff(
            plans,
            exchange=cutoff_exchange,
        )
        with ThreadPoolExecutor(
            max_workers=GAME_NETWORK_WORKERS,
            thread_name_prefix="x13-game-capture",
        ) as executor:
            futures = {
                executor.submit(
                    _capture_game,
                    plan,
                    requester=active_requester,
                    manifest_writer=writer,
                    raw_store_root=preflight.raw_store_root,
                    program_root=preflight.program_root,
                    response_clock=response_clock,
                    rate_limiter=limiter,
                    kalshi_max_pages=kalshi_max_pages,
                ): plan.game_id
                for plan in plans
            }
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if owned_client is not None:
            owned_client.close()

    games = tuple(sorted(results, key=lambda result: result.game_id))
    manifests = tuple(
        batch_manifests
    ) + tuple(
        receipt for game in games for receipt in game.manifests
    )
    receipt = CaptureReceipt(
        universe_id=universe_id,
        plan_sha256=plan_sha256,
        preflight=preflight,
        kalshi_cutoff=kalshi_cutoff,
        games=games,
        manifests=manifests,
    )
    receipt_path = _publish_receipt(receipt, preflight.raw_store_root)
    pointer = _capture_pointer(receipt)
    pointer_path = publisher(pointer, preflight.raw_store_root)
    if not isinstance(pointer_path, Path):
        raise X13CaptureError(
            "pointer_publisher must return a Path"
        )
    return BatchCaptureResult(
        preflight=preflight,
        kalshi_cutoff=kalshi_cutoff,
        games=games,
        manifests=manifests,
        receipt=receipt,
        receipt_path=receipt_path,
        pointer=pointer,
        pointer_path=pointer_path,
    )


__all__ = [
    "DEFAULT_HOST_REQUESTS_PER_SECOND",
    "GAME_NETWORK_WORKERS",
    "GIB",
    "KALSHI_PAGE_LIMIT",
    "POLYMARKET_TRADE_LIMIT",
    "BatchCaptureResult",
    "CaptureHttpResponse",
    "CapturePointer",
    "CaptureReceipt",
    "CaptureRequest",
    "CaptureTarget",
    "DiskPreflight",
    "GameCapturePlan",
    "GameCaptureResult",
    "HostRateLimiter",
    "ImmutableRequestManifest",
    "KalshiCandlestickCapture",
    "KalshiHistoricalCutoff",
    "KalshiTradeProof",
    "PolymarketTerminalWindow",
    "X13CaptureError",
    "X13TransientCaptureError",
    "capture_x13_batch",
    "load_x13_capture",
    "preflight_x13_capture",
    "preserve_capture_response",
]
