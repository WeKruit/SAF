"""Pure, fail-closed contracts and measurements for prospective NFL X-14.

The module has no network or order-entry behavior.  It turns governed capture
records into source-local timing evidence, continuity-audited L2 updates,
reaction measurements, and explicitly non-fill shadow taker quotes.
"""

from __future__ import annotations

import hashlib
import json
import re
import base64
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from prediction_market.contracts import CaptureTimingV0


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NS_PER_SECOND = 1_000_000_000


def _identifier(value: object, field: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a stable non-empty identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical SHA-256 reference")
    return value


def _utc(value: object, field: str) -> str:
    if type(value) is not str or not value.endswith("Z") or "T" not in value:
        raise ValueError(f"{field} must be a canonical UTC timestamp")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    result = tuple(_identifier(item, field) for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique identifiers")
    return tuple(sorted(result))


def _outcomes(value: object, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must map native IDs to outcomes")
    normalized: dict[str, str] = {}
    for native_id, outcome in value.items():
        key = _identifier(native_id, f"{field}.native_id")
        normalized[key] = _identifier(outcome, f"{field}.outcome")
    if len(normalized) < 2 or len(set(normalized.values())) < 2:
        raise ValueError(f"{field} must contain complete binary outcomes")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class ProspectiveGameBindingV1:
    """Registry-loaded game/provider/venue identity for one capture session."""

    binding_id: str
    game_id: str
    capture_session_id: str
    scheduled_start_utc: str
    away_team: str
    home_team: str
    sports_provider: str
    sports_provider_game_id: str
    polymarket_event_id: str
    polymarket_event_slug: str
    polymarket_native_game_id: str
    polymarket_condition_ids: tuple[str, ...]
    polymarket_token_outcomes: Mapping[str, str]
    kalshi_event_id: str | None
    kalshi_ticker_outcomes: Mapping[str, str]
    capture_start: Literal["market_creation"]
    capture_end: Literal["market_resolution"]
    venue_rule_snapshot_refs: tuple[str, ...]
    license_refs: tuple[str, ...]
    s3_namespace: str

    @classmethod
    def from_registry_record(
        cls, record: Mapping[str, object]
    ) -> "ProspectiveGameBindingV1":
        """Build from governed registry data, with no game-specific defaults."""

        if not isinstance(record, Mapping):
            raise ValueError("binding registry record must be a mapping")
        polymarket = record.get("polymarket")
        if not isinstance(polymarket, Mapping):
            raise ValueError("polymarket binding must be present")
        kalshi = record.get("kalshi")
        if kalshi is not None and not isinstance(kalshi, Mapping):
            raise ValueError("kalshi binding must be a mapping or null")
        away = _identifier(record.get("away_team"), "away_team")
        home = _identifier(record.get("home_team"), "home_team")
        if away == home:
            raise ValueError("home_team and away_team must differ")
        start = record.get("capture_start")
        end = record.get("capture_end")
        if start != "market_creation" or end != "market_resolution":
            raise ValueError(
                "capture window must be market_creation through "
                "market_resolution"
            )
        rule_refs = _string_tuple(
            record.get("venue_rule_snapshot_refs"),
            "venue_rule_snapshot_refs",
        )
        for reference in rule_refs:
            _sha256(reference, "venue_rule_snapshot_refs")
        kalshi_outcomes: Mapping[str, str] = MappingProxyType({})
        kalshi_event_id: str | None = None
        if kalshi is not None:
            kalshi_event_id = _identifier(
                kalshi.get("event_id"), "kalshi.event_id"
            )
            kalshi_outcomes = _outcomes(
                kalshi.get("ticker_outcomes"),
                "kalshi.ticker_outcomes",
            )
        return cls(
            binding_id=_identifier(
                record.get("binding_id"), "binding_id"
            ),
            game_id=_identifier(record.get("game_id"), "game_id"),
            capture_session_id=_identifier(
                record.get("capture_session_id"), "capture_session_id"
            ),
            scheduled_start_utc=_utc(
                record.get("scheduled_start_utc"),
                "scheduled_start_utc",
            ),
            away_team=away,
            home_team=home,
            sports_provider=_identifier(
                record.get("sports_provider"), "sports_provider"
            ),
            sports_provider_game_id=_identifier(
                record.get("sports_provider_game_id"),
                "sports_provider_game_id",
            ),
            polymarket_event_id=_identifier(
                polymarket.get("event_id"), "polymarket.event_id"
            ),
            polymarket_event_slug=_identifier(
                polymarket.get("event_slug"), "polymarket.event_slug"
            ),
            polymarket_native_game_id=_identifier(
                polymarket.get("native_game_id"),
                "polymarket.native_game_id",
            ),
            polymarket_condition_ids=_string_tuple(
                polymarket.get("condition_ids"),
                "polymarket.condition_ids",
            ),
            polymarket_token_outcomes=_outcomes(
                polymarket.get("token_outcomes"),
                "polymarket.token_outcomes",
            ),
            kalshi_event_id=kalshi_event_id,
            kalshi_ticker_outcomes=kalshi_outcomes,
            capture_start="market_creation",
            capture_end="market_resolution",
            venue_rule_snapshot_refs=rule_refs,
            license_refs=_string_tuple(
                record.get("license_refs"), "license_refs"
            ),
            s3_namespace=_identifier(
                record.get("s3_namespace"), "s3_namespace"
            ),
        )

    @property
    def capture_window(self) -> tuple[str, str]:
        return self.capture_start, self.capture_end


@dataclass(frozen=True, slots=True)
class CaptureClockContextV1:
    """Stable same-host clock identity shared by sports and market capture."""

    host_id: str
    boot_id: str
    monotonic_clock_id: str
    clock_epoch_id: str
    clock_sync_sample_id: str
    pairing_uncertainty_ns: int

    def __post_init__(self) -> None:
        for field in (
            "host_id",
            "boot_id",
            "monotonic_clock_id",
            "clock_epoch_id",
            "clock_sync_sample_id",
        ):
            _identifier(getattr(self, field), field)
        if (
            type(self.pairing_uncertainty_ns) is not int
            or self.pairing_uncertainty_ns < 0
        ):
            raise ValueError(
                "pairing_uncertainty_ns must be non-negative"
            )


def load_prospective_game_binding(
    path: str | Path,
) -> ProspectiveGameBindingV1:
    """Load one direct JSON registry record; no embedded/default fallback."""

    target = Path(path)
    try:
        payload = target.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("prospective game binding is unreadable JSON") from error
    if not isinstance(value, dict):
        raise ValueError("prospective game binding JSON must be an object")
    return ProspectiveGameBindingV1.from_registry_record(value)


@dataclass(frozen=True, slots=True)
class RawCaptureEnvelopeV1:
    """One exact raw payload paired with the canonical capture timing sidecar."""

    envelope_id: str
    binding_id: str
    game_id: str
    source: str
    stream: str
    message_kind: str
    timing: CaptureTimingV0
    source_time_utc_ns: int | None
    source_time_precision_ns: int | None
    source_clock_domain: str | None
    source_sequence: int | str | None
    book_hash: str | None
    market_id: str | None
    token_or_ticker: str | None
    provider_event_id: str | None
    revision: int | None
    segment_sha256: str
    raw_payload: bytes

    def __post_init__(self) -> None:
        for field in (
            "envelope_id",
            "binding_id",
            "game_id",
            "source",
            "stream",
            "message_kind",
        ):
            _identifier(getattr(self, field), field)
        if not isinstance(self.timing, CaptureTimingV0):
            raise ValueError("timing must be CaptureTimingV0")
        _sha256(self.segment_sha256, "segment_sha256")
        if type(self.raw_payload) is not bytes:
            raise ValueError("raw_payload must be exact bytes")
        payload_sha = "sha256:" + hashlib.sha256(self.raw_payload).hexdigest()
        if payload_sha != self.timing.payload_sha256:
            raise ValueError("raw payload SHA does not match CaptureTimingV0")
        if self.source_time_utc_ns is None:
            if (
                self.source_time_precision_ns is not None
                or self.source_clock_domain is not None
            ):
                raise ValueError(
                    "source time metadata requires source_time_utc_ns"
                )
        else:
            if type(self.source_time_utc_ns) is not int:
                raise ValueError("source_time_utc_ns must be integer or null")
            if (
                type(self.source_time_precision_ns) is not int
                or self.source_time_precision_ns < 0
            ):
                raise ValueError(
                    "source_time_precision_ns must be non-negative"
                )
            _identifier(
                self.source_clock_domain, "source_clock_domain"
            )
        if isinstance(self.source_sequence, int) and self.source_sequence < 0:
            raise ValueError("source_sequence cannot be negative")
        if self.revision is not None and (
            type(self.revision) is not int or self.revision < 0
        ):
            raise ValueError("revision must be non-negative or null")

    @property
    def local_receive_utc_ns(self) -> int:
        return self.timing.local_receive_utc_ns

    @property
    def local_receive_monotonic_ns(self) -> int:
        return self.timing.local_receive_monotonic_ns

    @property
    def connection_epoch(self) -> int:
        return self.timing.connection_epoch

    @property
    def record_ordinal(self) -> int:
        return self.timing.record_ordinal

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "raw_capture_envelope_v1",
            "envelope_id": self.envelope_id,
            "binding_id": self.binding_id,
            "game_id": self.game_id,
            "source": self.source,
            "stream": self.stream,
            "message_kind": self.message_kind,
            "timing": self.timing.model_dump(mode="json"),
            "source_time_utc_ns": self.source_time_utc_ns,
            "source_time_precision_ns": self.source_time_precision_ns,
            "source_clock_domain": self.source_clock_domain,
            "source_sequence": self.source_sequence,
            "book_hash": self.book_hash,
            "market_id": self.market_id,
            "token_or_ticker": self.token_or_ticker,
            "provider_event_id": self.provider_event_id,
            "revision": self.revision,
            "segment_sha256": self.segment_sha256,
            "raw_payload_base64": base64.b64encode(
                self.raw_payload
            ).decode("ascii"),
        }


RevisionStatus = Literal["PROVISIONAL", "FINALIZED", "CORRECTED"]


@dataclass(frozen=True, slots=True)
class InformationEventV1:
    """One provider event revision with explicit seen/ready/final boundaries."""

    event_id: str
    game_id: str
    provider: str
    provider_event_id: str
    revision: int
    revision_status: RevisionStatus
    timing: CaptureTimingV0
    provider_publish_time_utc_ns: int | None
    finalized_monotonic_ns: int | None
    pre_state: Mapping[str, Any]
    post_state: Mapping[str, Any]
    episode_kind: str
    salient: bool
    review: bool
    penalty: bool
    no_play: bool

    def __post_init__(self) -> None:
        for field in (
            "event_id",
            "game_id",
            "provider",
            "provider_event_id",
            "episode_kind",
        ):
            _identifier(getattr(self, field), field)
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.revision_status not in {
            "PROVISIONAL",
            "FINALIZED",
            "CORRECTED",
        }:
            raise ValueError("revision_status is invalid")
        if self.timing.state_or_book_ready_monotonic_ns is None:
            raise ValueError("information event requires state-ready time")
        if self.finalized_monotonic_ns is not None:
            if self.finalized_monotonic_ns < self.t_ready_monotonic_ns:
                raise ValueError("T_final cannot precede T_ready")
        elif self.revision_status != "PROVISIONAL":
            raise ValueError("finalized/corrected event requires T_final")
        if self.no_play and self.salient:
            raise ValueError("no-play event cannot be salient")
        object.__setattr__(
            self, "pre_state", MappingProxyType(dict(self.pre_state))
        )
        object.__setattr__(
            self, "post_state", MappingProxyType(dict(self.post_state))
        )

    @property
    def t_seen_monotonic_ns(self) -> int:
        return self.timing.local_receive_monotonic_ns

    @property
    def t_ready_monotonic_ns(self) -> int:
        ready = self.timing.state_or_book_ready_monotonic_ns
        assert ready is not None
        return ready

    @property
    def t_final_monotonic_ns(self) -> int | None:
        return self.finalized_monotonic_ns


@runtime_checkable
class SportsFeedAdapter(Protocol):
    """Provider adapter boundary; parsing is deterministic and network-free."""

    provider: str

    def parse(self, envelope: RawCaptureEnvelopeV1) -> InformationEventV1:
        """Parse one already-preserved raw envelope."""


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: Decimal
    size: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.price, Decimal) or not (
            Decimal("0") <= self.price <= Decimal("1")
        ):
            raise ValueError("book price must be Decimal in [0, 1]")
        if not isinstance(self.size, Decimal) or self.size < 0:
            raise ValueError("book size must be non-negative Decimal")


UpdateKind = Literal[
    "snapshot",
    "delta",
    "bbo",
    "trade",
    "tick_size",
    "suspension",
    "resume",
]
ContinuityStatus = Literal[
    "CONTIGUOUS",
    "GAP",
    "SUSPENDED",
    "AMBIGUOUS",
]


@dataclass(frozen=True, slots=True)
class MarketUpdateV1:
    """Normalized observed market state/update, never an inferred execution."""

    update_id: str
    raw_envelope_id: str
    game_id: str
    venue: Literal["polymarket", "kalshi"]
    logical_market_id: str
    outcome: str
    token_or_ticker: str
    update_kind: UpdateKind
    timing: CaptureTimingV0
    exchange_time_utc_ns: int | None
    source_sequence: int | None
    book_hash: str | None
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    observed: bool
    derived: bool
    continuity_status: ContinuityStatus
    price_representation: Literal["TOKEN_OUTCOME", "YES_LEG"]
    tick_size: Decimal

    def __post_init__(self) -> None:
        for field in (
            "update_id",
            "raw_envelope_id",
            "game_id",
            "logical_market_id",
            "outcome",
            "token_or_ticker",
        ):
            _identifier(getattr(self, field), field)
        if self.venue not in {"polymarket", "kalshi"}:
            raise ValueError("venue must be polymarket or kalshi")
        expected_representation = (
            "TOKEN_OUTCOME"
            if self.venue == "polymarket"
            else "YES_LEG"
        )
        if self.price_representation != expected_representation:
            raise ValueError(
                f"{self.venue} requires {expected_representation} prices"
            )
        if (
            not isinstance(self.tick_size, Decimal)
            or self.tick_size <= 0
        ):
            raise ValueError("tick_size must be a positive Decimal")
        if self.update_kind not in {
            "snapshot",
            "delta",
            "bbo",
            "trade",
            "tick_size",
            "suspension",
            "resume",
        }:
            raise ValueError("update_kind is invalid")
        if not isinstance(self.timing, CaptureTimingV0):
            raise ValueError("timing must be CaptureTimingV0")
        if self.timing.state_or_book_ready_monotonic_ns is None:
            raise ValueError("market update requires book-ready boundary")
        if self.source_sequence is not None and (
            type(self.source_sequence) is not int
            or self.source_sequence < 0
        ):
            raise ValueError("source_sequence must be non-negative or null")
        if self.observed == self.derived:
            raise ValueError(
                "market update must be exactly one of observed or derived"
            )
        if self.continuity_status not in {
            "CONTIGUOUS",
            "GAP",
            "SUSPENDED",
            "AMBIGUOUS",
        }:
            raise ValueError("continuity_status is invalid")
        if any(
            earlier.price < later.price
            for earlier, later in zip(self.bids, self.bids[1:])
        ):
            raise ValueError("bids must be highest price first")
        if any(
            earlier.price > later.price
            for earlier, later in zip(self.asks, self.asks[1:])
        ):
            raise ValueError("asks must be lowest price first")
        if self.bids and self.asks and self.best_bid >= self.best_ask:
            raise ValueError("observed book must have positive spread")

    @property
    def connection_epoch(self) -> int:
        return self.timing.connection_epoch

    @property
    def local_receive_monotonic_ns(self) -> int:
        return self.timing.local_receive_monotonic_ns

    @property
    def book_ready_monotonic_ns(self) -> int:
        ready = self.timing.state_or_book_ready_monotonic_ns
        if ready is None:
            raise ValueError("market update requires book-ready boundary")
        return ready

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / Decimal("2")

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def to_dict(self) -> dict[str, Any]:
        def levels(rows: Sequence[BookLevel]) -> list[dict[str, str]]:
            return [
                {"price": str(row.price), "size": str(row.size)}
                for row in rows
            ]

        return {
            "record_type": "market_update_v1",
            "update_id": self.update_id,
            "raw_envelope_id": self.raw_envelope_id,
            "game_id": self.game_id,
            "venue": self.venue,
            "logical_market_id": self.logical_market_id,
            "outcome": self.outcome,
            "token_or_ticker": self.token_or_ticker,
            "update_kind": self.update_kind,
            "timing": self.timing.model_dump(mode="json"),
            "exchange_time_utc_ns": self.exchange_time_utc_ns,
            "source_sequence": self.source_sequence,
            "book_hash": self.book_hash,
            "bids": levels(self.bids),
            "asks": levels(self.asks),
            "observed": self.observed,
            "derived": self.derived,
            "continuity_status": self.continuity_status,
            "price_representation": self.price_representation,
            "tick_size": str(self.tick_size),
        }


class CaptureContinuityGate:
    """Per-venue epoch/snapshot gate for normalized prospective L2."""

    def __init__(self) -> None:
        self._active_epoch = {"polymarket": 0, "kalshi": 0}
        self._snapshots: set[tuple[str, int, str]] = set()
        self._last_kalshi_sequence: dict[int, int] = {}
        self._kalshi_blocked: set[int] = set()

    def start_epoch(
        self, *, venue: Literal["polymarket", "kalshi"], epoch: int
    ) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch <= self._active_epoch[venue]:
            raise ValueError("new epoch must increase monotonically")
        self._active_epoch[venue] = epoch
        self._snapshots = {
            item for item in self._snapshots if item[0] != venue
        }
        if venue == "kalshi":
            self._kalshi_blocked.discard(epoch)

    def observe(self, update: MarketUpdateV1) -> None:
        venue = update.venue
        epoch = update.connection_epoch
        if epoch != self._active_epoch[venue]:
            raise ValueError("market update does not match active epoch")
        key = (venue, epoch, update.token_or_ticker)
        if venue == "kalshi" and epoch in self._kalshi_blocked:
            raise ValueError(
                "Kalshi epoch is blocked until a new epoch and snapshot"
            )
        if update.update_kind == "snapshot":
            self._snapshots.add(key)
        elif update.update_kind == "delta" and key not in self._snapshots:
            raise ValueError(
                "delta requires snapshot for this token in this epoch"
            )
        if venue != "kalshi" or update.source_sequence is None:
            return
        prior = self._last_kalshi_sequence.get(epoch)
        if prior is not None and update.source_sequence != prior + 1:
            self._kalshi_blocked.add(epoch)
            raise ValueError(
                "Kalshi sequence gap blocks the epoch until new snapshot"
            )
        self._last_kalshi_sequence[epoch] = update.source_sequence


class ReactionCensorReason(str, Enum):
    CONTINUITY_GAP = "CONTINUITY_GAP"
    SUSPENSION = "SUSPENSION"
    ORDER_AMBIGUOUS = "ORDER_AMBIGUOUS"
    CONTAMINATION = "CONTAMINATION"
    MISSING_BASELINE = "MISSING_BASELINE"
    MISSING_OBSERVATION = "MISSING_OBSERVATION"
    UNSTABLE_TERMINAL = "UNSTABLE_TERMINAL"
    IMMATERIAL_RESPONSE = "IMMATERIAL_RESPONSE"


@dataclass(frozen=True, slots=True)
class ReactionMeasurementV1:
    event_id: str
    game_id: str
    venue: str | None
    logical_market_id: str | None
    outcome: str | None
    t_seen_monotonic_ns: int
    t_ready_monotonic_ns: int
    t_final_monotonic_ns: int | None
    first_any_market_update_monotonic_ns: int | None
    first_bid_move_monotonic_ns: int | None
    first_ask_move_monotonic_ns: int | None
    first_tick_move_monotonic_ns: int | None
    first_material_move_monotonic_ns: int | None
    analysis_end_monotonic_ns: int
    terminal_bid: Decimal | None
    terminal_ask: Decimal | None
    terminal_midpoint: Decimal | None
    baseline_spread: Decimal | None
    terminal_spread: Decimal | None
    baseline_top_depth: Decimal | None
    terminal_top_depth: Decimal | None
    depth_withdrawal: Decimal | None
    order_flow_imbalance: Decimal | None
    maximum_favorable_excursion: Decimal | None
    maximum_adverse_excursion: Decimal | None
    overshoot: bool | None
    reversal: bool | None
    first_passage_t50_ns: int | None
    first_passage_t90_ns: int | None
    settled_t90_ns: int | None
    bid_first_passage_t50_ns: int | None
    bid_first_passage_t90_ns: int | None
    ask_first_passage_t50_ns: int | None
    ask_first_passage_t90_ns: int | None
    reference_delta: Decimal | None
    reference_completion_fraction: Decimal | None
    reference_first_passage_t50_ns: int | None
    reference_first_passage_t90_ns: int | None
    censor_reason: ReactionCensorReason | None
    midpoint_is_executable: Literal[False] = False


def _bbo(update: MarketUpdateV1) -> bool:
    return update.best_bid is not None and update.best_ask is not None


def _weighted_plateau(
    updates: Sequence[MarketUpdateV1],
    *,
    start_ns: int,
    end_ns: int,
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    active = [
        row for row in updates if row.book_ready_monotonic_ns <= start_ns
    ]
    if not active:
        return None
    current = active[-1]
    if not _bbo(current):
        return None
    changes = [
        row
        for row in updates
        if start_ns < row.book_ready_monotonic_ns < end_ns and _bbo(row)
    ]
    timeline = [current, *changes]
    bid_weighted = Decimal("0")
    ask_weighted = Decimal("0")
    total_ns = 0
    observed_midpoints: list[Decimal] = []
    for index, row in enumerate(timeline):
        left = max(start_ns, row.book_ready_monotonic_ns)
        right = (
            changes[index].book_ready_monotonic_ns
            if index < len(changes)
            else end_ns
        )
        duration = max(0, right - left)
        if duration == 0:
            continue
        assert row.best_bid is not None
        assert row.best_ask is not None
        assert row.midpoint is not None
        bid_weighted += row.best_bid * duration
        ask_weighted += row.best_ask * duration
        total_ns += duration
        observed_midpoints.append(row.midpoint)
    if total_ns != end_ns - start_ns or not observed_midpoints:
        return None
    bid = bid_weighted / total_ns
    ask = ask_weighted / total_ns
    return bid, ask, min(observed_midpoints), max(observed_midpoints)


def _first_passage(
    *,
    baseline: Decimal,
    terminal: Decimal,
    fraction: Decimal,
    ready_ns: int,
    updates: Sequence[MarketUpdateV1],
) -> int | None:
    target = (terminal - baseline) * fraction
    for row in updates:
        midpoint = row.midpoint
        if midpoint is None:
            continue
        delta = midpoint - baseline
        if (target >= 0 and delta >= target) or (
            target < 0 and delta <= target
        ):
            return row.book_ready_monotonic_ns - ready_ns
    return None


def _first_side_passage(
    *,
    baseline: Decimal,
    terminal: Decimal,
    fraction: Decimal,
    ready_ns: int,
    updates: Sequence[MarketUpdateV1],
    side: Literal["bid", "ask"],
) -> int | None:
    target = (terminal - baseline) * fraction
    for row in updates:
        price = row.best_bid if side == "bid" else row.best_ask
        if price is None:
            continue
        delta = price - baseline
        if (target >= 0 and delta >= target) or (
            target < 0 and delta <= target
        ):
            return row.book_ready_monotonic_ns - ready_ns
    return None


def _ofi_step(
    prior: MarketUpdateV1, current: MarketUpdateV1
) -> Decimal | None:
    if not (
        prior.bids
        and prior.asks
        and current.bids
        and current.asks
    ):
        return None
    old_bid, old_ask = prior.bids[0], prior.asks[0]
    new_bid, new_ask = current.bids[0], current.asks[0]
    bid_flow = (
        new_bid.size
        if new_bid.price > old_bid.price
        else -old_bid.size
        if new_bid.price < old_bid.price
        else new_bid.size - old_bid.size
    )
    ask_flow = (
        -new_ask.size
        if new_ask.price < old_ask.price
        else old_ask.size
        if new_ask.price > old_ask.price
        else -(new_ask.size - old_ask.size)
    )
    return bid_flow + ask_flow


def _settled_time(
    *,
    baseline: Decimal,
    terminal: Decimal,
    ready_ns: int,
    end_ns: int,
    updates: Sequence[MarketUpdateV1],
    tolerance: Decimal,
) -> int | None:
    target_delta = terminal - baseline
    for index, row in enumerate(updates):
        midpoint = row.midpoint
        if midpoint is None or abs((midpoint - baseline) - target_delta) > tolerance:
            continue
        settle_end = row.book_ready_monotonic_ns + 2 * _NS_PER_SECOND
        if settle_end > end_ns:
            continue
        later = [
            child
            for child in updates[index + 1 :]
            if child.book_ready_monotonic_ns < settle_end
        ]
        if all(
            child.midpoint is not None
            and abs((child.midpoint - baseline) - target_delta) <= tolerance
            for child in later
        ):
            return row.book_ready_monotonic_ns - ready_ns
    return None


def measure_reaction(
    *,
    event: InformationEventV1,
    updates: Sequence[MarketUpdateV1],
    venue_tick: Decimal,
    control_noise_95: Decimal,
    next_salient_event_monotonic_ns: int | None,
    reference_delta: Decimal | None = None,
) -> ReactionMeasurementV1:
    """Measure one event path using local monotonic time only."""

    if venue_tick <= 0 or control_noise_95 < 0:
        raise ValueError("tick must be positive and control noise non-negative")
    ready = event.t_ready_monotonic_ns
    hard_end = ready + 60 * _NS_PER_SECOND
    contaminated = (
        next_salient_event_monotonic_ns is not None
        and ready < next_salient_event_monotonic_ns < hard_end
    )
    analysis_end = (
        next_salient_event_monotonic_ns if contaminated else hard_end
    )
    ordered = sorted(
        updates, key=lambda row: row.book_ready_monotonic_ns
    )
    if any(not row.observed or row.derived for row in ordered):
        raise ValueError("formal reaction cannot use derived book updates")
    expected_clock = (
        event.timing.capture_session_id,
        event.timing.host_id,
        event.timing.boot_id,
        event.timing.monotonic_clock_id,
        event.timing.clock_epoch_id,
    )
    for row in ordered:
        observed_clock = (
            row.timing.capture_session_id,
            row.timing.host_id,
            row.timing.boot_id,
            row.timing.monotonic_clock_id,
            row.timing.clock_epoch_id,
        )
        if observed_clock != expected_clock:
            raise ValueError(
                "event and market updates must share session and "
                "host/boot/clock epoch"
            )
    connection_epochs = {
        row.connection_epoch for row in ordered
    }
    if len(connection_epochs) > 1:
        raise ValueError(
            "formal reaction cannot cross a market connection epoch"
        )
    identities = {
        (row.game_id, row.venue, row.logical_market_id, row.outcome)
        for row in ordered
    }
    if any(row.game_id != event.game_id for row in ordered):
        raise ValueError("event and market updates must share one game")
    if len(identities) > 1:
        raise ValueError("reaction input must contain one logical outcome")
    identity = next(iter(identities), None)
    baseline_candidates = [
        row
        for row in ordered
        if row.book_ready_monotonic_ns <= ready and _bbo(row)
        and row.continuity_status == "CONTIGUOUS"
    ]
    post = [
        row
        for row in ordered
        if ready < row.book_ready_monotonic_ns <= analysis_end
    ]
    censor: ReactionCensorReason | None = (
        ReactionCensorReason.CONTAMINATION if contaminated else None
    )
    seen_post_times: set[int] = set()
    for row in post:
        timestamp = row.book_ready_monotonic_ns
        if timestamp in seen_post_times:
            analysis_end = min(analysis_end, timestamp)
            censor = ReactionCensorReason.ORDER_AMBIGUOUS
            post = [
                child
                for child in post
                if child.book_ready_monotonic_ns < analysis_end
            ]
            break
        seen_post_times.add(timestamp)
    status_reason = {
        "GAP": ReactionCensorReason.CONTINUITY_GAP,
        "SUSPENDED": ReactionCensorReason.SUSPENSION,
        "AMBIGUOUS": ReactionCensorReason.ORDER_AMBIGUOUS,
    }
    for row in post:
        reason = status_reason.get(row.continuity_status)
        if reason is not None:
            analysis_end = min(
                analysis_end, row.book_ready_monotonic_ns
            )
            censor = reason
            post = [
                child
                for child in post
                if child.book_ready_monotonic_ns < analysis_end
            ]
            break

    baseline = baseline_candidates[-1] if baseline_candidates else None
    first_any = post[0].book_ready_monotonic_ns if post else None
    first_bid = None
    first_ask = None
    first_tick = None
    first_material = None
    materiality = max(venue_tick, control_noise_95)
    if baseline is not None and _bbo(baseline):
        assert baseline.best_bid is not None
        assert baseline.best_ask is not None
        assert baseline.midpoint is not None
        for row in post:
            if not _bbo(row):
                continue
            assert row.best_bid is not None
            assert row.best_ask is not None
            assert row.midpoint is not None
            if first_bid is None and abs(
                row.best_bid - baseline.best_bid
            ) >= row.tick_size:
                first_bid = row.book_ready_monotonic_ns
            if first_ask is None and abs(
                row.best_ask - baseline.best_ask
            ) >= row.tick_size:
                first_ask = row.book_ready_monotonic_ns
            if first_tick is None and abs(
                row.midpoint - baseline.midpoint
            ) >= row.tick_size:
                first_tick = row.book_ready_monotonic_ns
            if first_material is None and abs(
                row.midpoint - baseline.midpoint
            ) >= max(row.tick_size, control_noise_95):
                first_material = row.book_ready_monotonic_ns

    terminal_bid = None
    terminal_ask = None
    terminal_midpoint = None
    first_t50 = None
    first_t90 = None
    settled_t90 = None
    max_favorable = None
    max_adverse = None
    overshoot = None
    reversal = None
    ofi = None
    reference_completion = None
    reference_t50 = None
    reference_t90 = None
    bid_t50 = None
    bid_t90 = None
    ask_t50 = None
    ask_t90 = None
    terminal_depth = None
    depth_withdrawal = None
    baseline_depth = (
        baseline.bids[0].size + baseline.asks[0].size
        if baseline is not None and baseline.bids and baseline.asks
        else None
    )

    if censor is None and baseline is None:
        censor = ReactionCensorReason.MISSING_BASELINE
    elif censor is None and not post:
        censor = ReactionCensorReason.MISSING_OBSERVATION
    elif censor is None:
        plateau = _weighted_plateau(
            ordered,
            start_ns=analysis_end - 5 * _NS_PER_SECOND,
            end_ns=analysis_end,
        )
        if plateau is None:
            censor = ReactionCensorReason.UNSTABLE_TERMINAL
        else:
            terminal_bid, terminal_ask, plateau_min, plateau_max = plateau
            terminal_midpoint = (
                terminal_bid + terminal_ask
            ) / Decimal("2")
            assert baseline is not None
            assert baseline.midpoint is not None
            response = terminal_midpoint - baseline.midpoint
            terminal_tick = next(
                (
                    row.tick_size
                    for row in reversed(ordered)
                    if row.book_ready_monotonic_ns <= analysis_end
                ),
                venue_tick,
            )
            materiality = max(terminal_tick, control_noise_95)
            tolerance = max(
                terminal_tick, abs(response) * Decimal(".10")
            )
            if plateau_max - plateau_min > tolerance:
                censor = ReactionCensorReason.UNSTABLE_TERMINAL
                terminal_bid = terminal_ask = terminal_midpoint = None
            elif abs(response) < materiality:
                censor = ReactionCensorReason.IMMATERIAL_RESPONSE
            else:
                eligible_post = [row for row in post if _bbo(row)]
                first_t50 = _first_passage(
                    baseline=baseline.midpoint,
                    terminal=terminal_midpoint,
                    fraction=Decimal(".50"),
                    ready_ns=ready,
                    updates=eligible_post,
                )
                first_t90 = _first_passage(
                    baseline=baseline.midpoint,
                    terminal=terminal_midpoint,
                    fraction=Decimal(".90"),
                    ready_ns=ready,
                    updates=eligible_post,
                )
                settled_t90 = _settled_time(
                    baseline=baseline.midpoint,
                    terminal=terminal_midpoint,
                    ready_ns=ready,
                    end_ns=analysis_end,
                    updates=eligible_post,
                    tolerance=tolerance,
                )
                assert baseline.best_bid is not None
                assert baseline.best_ask is not None
                assert terminal_bid is not None
                assert terminal_ask is not None
                bid_t50 = _first_side_passage(
                    baseline=baseline.best_bid,
                    terminal=terminal_bid,
                    fraction=Decimal(".50"),
                    ready_ns=ready,
                    updates=eligible_post,
                    side="bid",
                )
                bid_t90 = _first_side_passage(
                    baseline=baseline.best_bid,
                    terminal=terminal_bid,
                    fraction=Decimal(".90"),
                    ready_ns=ready,
                    updates=eligible_post,
                    side="bid",
                )
                ask_t50 = _first_side_passage(
                    baseline=baseline.best_ask,
                    terminal=terminal_ask,
                    fraction=Decimal(".50"),
                    ready_ns=ready,
                    updates=eligible_post,
                    side="ask",
                )
                ask_t90 = _first_side_passage(
                    baseline=baseline.best_ask,
                    terminal=terminal_ask,
                    fraction=Decimal(".90"),
                    ready_ns=ready,
                    updates=eligible_post,
                    side="ask",
                )
                deltas = [
                    row.midpoint - baseline.midpoint
                    for row in eligible_post
                    if row.midpoint is not None
                ]
                oriented = [
                    delta if response > 0 else -delta
                    for delta in deltas
                ]
                if oriented:
                    max_favorable = max(oriented)
                    max_adverse = min(oriented)
                    overshoot = max_favorable > abs(response) + tolerance
                    reversal = max_adverse < -materiality
                ofi_values = [
                    value
                    for value in (
                        _ofi_step(prior, current)
                        for prior, current in zip(
                            [baseline, *eligible_post[:-1]],
                            eligible_post,
                        )
                    )
                    if value is not None
                ]
                if ofi_values:
                    ofi = sum(ofi_values, Decimal("0"))
                if (
                    reference_delta is not None
                    and abs(reference_delta) >= materiality
                ):
                    reference_completion = response / reference_delta
                    reference_terminal = (
                        baseline.midpoint + reference_delta
                    )
                    reference_t50 = _first_passage(
                        baseline=baseline.midpoint,
                        terminal=reference_terminal,
                        fraction=Decimal(".50"),
                        ready_ns=ready,
                        updates=eligible_post,
                    )
                    reference_t90 = _first_passage(
                        baseline=baseline.midpoint,
                        terminal=reference_terminal,
                        fraction=Decimal(".90"),
                        ready_ns=ready,
                        updates=eligible_post,
                    )
                last_bbo = next(
                    (
                        row
                        for row in reversed(ordered)
                        if row.book_ready_monotonic_ns <= analysis_end
                        and row.bids
                        and row.asks
                    ),
                    None,
                )
                if last_bbo is not None:
                    terminal_depth = (
                        last_bbo.bids[0].size + last_bbo.asks[0].size
                    )
                if baseline_depth is not None and terminal_depth is not None:
                    depth_withdrawal = baseline_depth - terminal_depth

    baseline_spread = baseline.spread if baseline is not None else None
    terminal_spread = (
        terminal_ask - terminal_bid
        if terminal_ask is not None and terminal_bid is not None
        else None
    )
    return ReactionMeasurementV1(
        event_id=event.event_id,
        game_id=event.game_id,
        venue=identity[1] if identity else None,
        logical_market_id=identity[2] if identity else None,
        outcome=identity[3] if identity else None,
        t_seen_monotonic_ns=event.t_seen_monotonic_ns,
        t_ready_monotonic_ns=ready,
        t_final_monotonic_ns=event.t_final_monotonic_ns,
        first_any_market_update_monotonic_ns=first_any,
        first_bid_move_monotonic_ns=first_bid,
        first_ask_move_monotonic_ns=first_ask,
        first_tick_move_monotonic_ns=first_tick,
        first_material_move_monotonic_ns=first_material,
        analysis_end_monotonic_ns=analysis_end,
        terminal_bid=terminal_bid,
        terminal_ask=terminal_ask,
        terminal_midpoint=terminal_midpoint,
        baseline_spread=baseline_spread,
        terminal_spread=terminal_spread,
        baseline_top_depth=baseline_depth,
        terminal_top_depth=terminal_depth,
        depth_withdrawal=depth_withdrawal,
        order_flow_imbalance=ofi,
        maximum_favorable_excursion=max_favorable,
        maximum_adverse_excursion=max_adverse,
        overshoot=overshoot,
        reversal=reversal,
        first_passage_t50_ns=first_t50,
        first_passage_t90_ns=first_t90,
        settled_t90_ns=settled_t90,
        bid_first_passage_t50_ns=bid_t50,
        bid_first_passage_t90_ns=bid_t90,
        ask_first_passage_t50_ns=ask_t50,
        ask_first_passage_t90_ns=ask_t90,
        reference_delta=reference_delta,
        reference_completion_fraction=reference_completion,
        reference_first_passage_t50_ns=reference_t50,
        reference_first_passage_t90_ns=reference_t90,
        censor_reason=censor,
    )


@dataclass(frozen=True, slots=True)
class ShadowTakerObservationV1:
    decision_ready_monotonic_ns: int
    venue: str
    game_id: str
    logical_market_id: str
    outcome: str
    requested_size: Decimal
    available_entry_size: Decimal
    available_exit_size: Decimal
    entry_ask_vwap: Decimal | None
    exit_bid_vwap: Decimal | None
    fee: Decimal
    venue_rule_snapshot_ref: str
    quote_status: Literal[
        "COMPLETE_SHADOW_QUOTE", "INSUFFICIENT_VISIBLE_DEPTH"
    ]
    execution_claim: Literal["NONE_SHADOW_ONLY"] = "NONE_SHADOW_ONLY"


def _walk(
    levels: Sequence[BookLevel], requested_size: Decimal
) -> tuple[Decimal, Decimal | None]:
    visible = sum((level.size for level in levels), Decimal("0"))
    if visible < requested_size:
        return visible, None
    remaining = requested_size
    notional = Decimal("0")
    for level in levels:
        taken = min(remaining, level.size)
        notional += taken * level.price
        remaining -= taken
        if remaining == 0:
            break
    return visible, notional / requested_size


def build_shadow_taker_observation(
    *,
    decision_ready_monotonic_ns: int,
    entry_update: MarketUpdateV1,
    exit_update: MarketUpdateV1,
    requested_size: Decimal,
    fee: Decimal,
    venue_rule_snapshot_ref: str,
) -> ShadowTakerObservationV1:
    """Book-walk visible L2 without representing an order or a fill."""

    if requested_size <= 0 or fee < 0:
        raise ValueError("requested_size must be positive and fee non-negative")
    if (
        entry_update.derived
        or exit_update.derived
        or not entry_update.observed
        or not exit_update.observed
    ):
        raise ValueError("shadow taker requires observed book states")
    entry_identity = (
        entry_update.venue,
        entry_update.game_id,
        entry_update.logical_market_id,
        entry_update.outcome,
    )
    exit_identity = (
        exit_update.venue,
        exit_update.game_id,
        exit_update.logical_market_id,
        exit_update.outcome,
    )
    if entry_identity != exit_identity:
        raise ValueError("entry and exit books must share one outcome")
    if (
        entry_update.continuity_status != "CONTIGUOUS"
        or exit_update.continuity_status != "CONTIGUOUS"
    ):
        raise ValueError("shadow taker requires continuous book states")
    if entry_update.connection_epoch != exit_update.connection_epoch:
        raise ValueError("shadow taker cannot cross a connection epoch")
    entry_clock = (
        entry_update.timing.capture_session_id,
        entry_update.timing.host_id,
        entry_update.timing.boot_id,
        entry_update.timing.monotonic_clock_id,
        entry_update.timing.clock_epoch_id,
    )
    exit_clock = (
        exit_update.timing.capture_session_id,
        exit_update.timing.host_id,
        exit_update.timing.boot_id,
        exit_update.timing.monotonic_clock_id,
        exit_update.timing.clock_epoch_id,
    )
    if entry_clock != exit_clock:
        raise ValueError("shadow taker requires one capture clock")
    if exit_update.book_ready_monotonic_ns < decision_ready_monotonic_ns:
        raise ValueError("exit book cannot precede the decision")
    available_entry, entry_vwap = _walk(
        entry_update.asks, requested_size
    )
    available_exit, exit_vwap = _walk(
        exit_update.bids, requested_size
    )
    complete = entry_vwap is not None and exit_vwap is not None
    return ShadowTakerObservationV1(
        decision_ready_monotonic_ns=decision_ready_monotonic_ns,
        venue=entry_update.venue,
        game_id=entry_update.game_id,
        logical_market_id=entry_update.logical_market_id,
        outcome=entry_update.outcome,
        requested_size=requested_size,
        available_entry_size=available_entry,
        available_exit_size=available_exit,
        entry_ask_vwap=entry_vwap,
        exit_bid_vwap=exit_vwap,
        fee=fee,
        venue_rule_snapshot_ref=_sha256(
            venue_rule_snapshot_ref, "venue_rule_snapshot_ref"
        ),
        quote_status=(
            "COMPLETE_SHADOW_QUOTE"
            if complete
            else "INSUFFICIENT_VISIBLE_DEPTH"
        ),
    )


__all__ = [
    "BookLevel",
    "CaptureClockContextV1",
    "CaptureContinuityGate",
    "InformationEventV1",
    "MarketUpdateV1",
    "ProspectiveGameBindingV1",
    "RawCaptureEnvelopeV1",
    "ReactionCensorReason",
    "ReactionMeasurementV1",
    "ShadowTakerObservationV1",
    "SportsFeedAdapter",
    "build_shadow_taker_observation",
    "load_prospective_game_binding",
    "measure_reaction",
]
