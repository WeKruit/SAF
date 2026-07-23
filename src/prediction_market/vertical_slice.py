"""X-09 deterministic vertical-slice harness and formal execution gate."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import prediction_market.contracts as contracts_module
import prediction_market.execution as execution_module
import prediction_market.experiments as experiments_module
import prediction_market.replay as replay_module
from prediction_market.contracts import (
    EventEnvelopeV0,
    FixedPointV0,
    VenueRuleSnapshotV0,
    canonical_sha256,
)
from prediction_market.execution import BookSnapshotV0, DepthLevelV0
from prediction_market.experiments import load_experiment_registry
from prediction_market.replay import (
    Level1SemanticSummaryV0,
    ReplayRunV0,
    assert_replay_deterministic,
    build_replay,
)


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNAL_DELAY = timedelta(seconds=5)


class X09VerticalSliceInputError(ValueError):
    """The frozen fixture cannot support the deterministic X-09 path."""


class X09ExperimentBlocked(PermissionError):
    """The formal X-09 registry gate is not satisfied."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__("X-09 formal execution blocked: " + "; ".join(blockers))


def _require_fixed_positive(value: object, field: str) -> FixedPointV0:
    if not isinstance(value, FixedPointV0) or value.to_decimal() <= 0:
        raise X09VerticalSliceInputError(f"{field} must be positive fixed-point")
    return value


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise X09VerticalSliceInputError(f"{field} must be lowercase sha256")
    return value


@dataclass(frozen=True, slots=True)
class X09ConfigV0:
    order_quantity: FixedPointV0
    maximum_order_quantity: FixedPointV0
    own_delay: timedelta
    pnl_horizon: timedelta
    random_seed: int
    canonical_id_snapshot_sha256: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        _require_fixed_positive(self.order_quantity, "order_quantity")
        _require_fixed_positive(
            self.maximum_order_quantity, "maximum_order_quantity"
        )
        if not isinstance(self.own_delay, timedelta) or self.own_delay < timedelta(0):
            raise X09VerticalSliceInputError("own_delay must be nonnegative timedelta")
        if (
            not isinstance(self.pnl_horizon, timedelta)
            or self.pnl_horizon <= timedelta(0)
        ):
            raise X09VerticalSliceInputError("pnl_horizon must be positive timedelta")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise X09VerticalSliceInputError("random_seed must be a nonnegative integer")
        _require_sha256(
            self.canonical_id_snapshot_sha256,
            "canonical_id_snapshot_sha256",
        )
        _require_sha256(self.dependency_lock_sha256, "dependency_lock_sha256")


@dataclass(frozen=True, slots=True)
class X09FixtureV0:
    events: tuple[EventEnvelopeV0, ...]
    rule_snapshot: VenueRuleSnapshotV0
    config: X09ConfigV0

    def __post_init__(self) -> None:
        if type(self.events) is not tuple or not self.events:
            raise X09VerticalSliceInputError("events must be a nonempty frozen tuple")
        if any(not isinstance(event, EventEnvelopeV0) for event in self.events):
            raise X09VerticalSliceInputError(
                "events must contain validated EventEnvelopeV0 values"
            )
        if not isinstance(self.rule_snapshot, VenueRuleSnapshotV0):
            raise X09VerticalSliceInputError(
                "rule_snapshot must be a validated VenueRuleSnapshotV0"
            )
        if not isinstance(self.config, X09ConfigV0):
            raise X09VerticalSliceInputError("config must be X09ConfigV0")


@dataclass(frozen=True, slots=True)
class X09RuntimeHashesV0:
    code_sha256: str
    data_sha256: str


@dataclass(frozen=True, slots=True)
class X09HarnessResultV0:
    harness_status: Literal["HARNESS_PASS"]
    experiment_status: Literal["EXPERIMENT_BLOCKED"]
    first_run: ReplayRunV0
    second_run: ReplayRunV0

    @property
    def events(self) -> tuple[EventEnvelopeV0, ...]:
        return self.first_run.events

    @property
    def orders(self) -> tuple[EventEnvelopeV0, ...]:
        return self.first_run.orders

    @property
    def fills(self) -> tuple[EventEnvelopeV0, ...]:
        return self.first_run.fills

    @property
    def pnl_events(self) -> tuple[EventEnvelopeV0, ...]:
        return self.first_run.pnl_events

    @property
    def semantic_summary(self) -> Level1SemanticSummaryV0:
        return self.first_run.semantic_summary

    @property
    def stream_sha256(self) -> str:
        return self.first_run.stream_sha256

    @property
    def canonical_log(self) -> bytes:
        return self.first_run.canonical_log


@dataclass(frozen=True, slots=True)
class _ObservedBook:
    event: EventEnvelopeV0
    book: BookSnapshotV0


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00").astimezone(UTC)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise X09VerticalSliceInputError("derived event time must be UTC")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _fixed(value: Decimal) -> FixedPointV0:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return FixedPointV0.from_value(text)


def _fixed_dump(value: FixedPointV0) -> dict[str, object]:
    return value.model_dump(mode="json")


def _timedelta_microseconds(value: timedelta) -> int:
    return value // timedelta(microseconds=1)


def _fixed_seconds(value: FixedPointV0, field: str) -> timedelta:
    microseconds = value.to_decimal() * Decimal(1_000_000)
    if microseconds != microseconds.to_integral_value():
        raise X09VerticalSliceInputError(
            f"{field} must be exactly representable in microseconds"
        )
    return timedelta(microseconds=int(microseconds))


def _payload_kind(event: EventEnvelopeV0) -> object:
    return event.payload.get("kind")


def _fixed_from_payload(value: object, field: str) -> FixedPointV0:
    if not isinstance(value, Mapping):
        raise X09VerticalSliceInputError(f"{field} must be fixed-point")
    try:
        return FixedPointV0.model_validate(dict(value))
    except (TypeError, ValueError) as exc:
        raise X09VerticalSliceInputError(f"invalid {field} fixed-point") from exc


def _parse_levels(value: object, field: str) -> tuple[DepthLevelV0, ...]:
    if not isinstance(value, tuple):
        raise X09VerticalSliceInputError(f"book {field} must be an immutable array")
    levels: list[DepthLevelV0] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"price", "quantity"}:
            raise X09VerticalSliceInputError(
                f"book {field} levels require price and quantity"
            )
        try:
            levels.append(
                DepthLevelV0(
                    price=_fixed_from_payload(item["price"], f"book {field} price"),
                    quantity=_fixed_from_payload(
                        item["quantity"], f"book {field} quantity"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise X09VerticalSliceInputError(f"invalid book {field} level") from exc
    return tuple(levels)


def _parse_book(event: EventEnvelopeV0) -> _ObservedBook:
    payload = event.payload
    if set(payload) != {"kind", "suspended", "bids", "asks"}:
        raise X09VerticalSliceInputError("book payload has unexpected fields")
    suspended = payload["suspended"]
    if type(suspended) is not bool:
        raise X09VerticalSliceInputError("book suspended must be boolean")
    try:
        local_receive_at = _parse_timestamp(event.time.receive_at)
        bids = _parse_levels(payload["bids"], "bids")
        asks = _parse_levels(payload["asks"], "asks")
        venue = event.source.venue
        condition_id = event.canonical_refs.condition_id
        content_sha256 = canonical_sha256(
            {
                "book_version": "v0",
                "venue": venue,
                "condition_id": condition_id,
                "local_receive_at": _timestamp(local_receive_at),
                "bids": [
                    {
                        "price": level.price.model_dump(mode="json"),
                        "quantity": level.quantity.model_dump(mode="json"),
                    }
                    for level in bids
                ],
                "asks": [
                    {
                        "price": level.price.model_dump(mode="json"),
                        "quantity": level.quantity.model_dump(mode="json"),
                    }
                    for level in asks
                ],
                "suspended": suspended,
            }
        )
        book = BookSnapshotV0(
            venue=venue,
            condition_id=condition_id,
            local_receive_at=local_receive_at,
            bids=bids,
            asks=asks,
            suspended=suspended,
            content_sha256=content_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise X09VerticalSliceInputError("invalid executable book") from exc
    return _ObservedBook(event=event, book=book)


def _validated_rule(snapshot: VenueRuleSnapshotV0) -> VenueRuleSnapshotV0:
    try:
        return VenueRuleSnapshotV0.model_validate(
            snapshot.model_dump(mode="python", round_trip=True)
        )
    except (TypeError, ValueError) as exc:
        raise X09VerticalSliceInputError("invalid venue rule snapshot") from exc


def _same_market(left: EventEnvelopeV0, right: EventEnvelopeV0) -> bool:
    return (
        left.canonical_refs.market_id == right.canonical_refs.market_id
        and left.canonical_refs.outcome_id == right.canonical_refs.outcome_id
        and left.canonical_refs.condition_id == right.canonical_refs.condition_id
    )


def _prepare_fixture(
    fixture: X09FixtureV0,
) -> tuple[
    tuple[EventEnvelopeV0, ...],
    EventEnvelopeV0,
    tuple[_ObservedBook, ...],
    VenueRuleSnapshotV0,
]:
    if not isinstance(fixture, X09FixtureV0):
        raise X09VerticalSliceInputError("fixture must be X09FixtureV0")
    source_events = build_replay(fixture.events).events
    if any(event.event_type != "normalized_observation" for event in source_events):
        raise X09VerticalSliceInputError(
            "frozen fixture may contain only normalized observations"
        )
    if any(event.experiment_id != "X-01" for event in source_events):
        raise X09VerticalSliceInputError(
            "frozen fixture may contain only X-01 reconstructed events"
        )
    scores = tuple(event for event in source_events if _payload_kind(event) == "score")
    if len(scores) != 1:
        raise X09VerticalSliceInputError("fixture requires exactly one score event")
    score = scores[0]
    if score.canonical_refs.condition_id is None:
        raise X09VerticalSliceInputError("score event requires a condition_id")
    books = tuple(
        _parse_book(event)
        for event in source_events
        if _payload_kind(event) == "book" and _same_market(event, score)
    )
    if not books:
        raise X09VerticalSliceInputError("fixture requires matching book observations")
    rule = _validated_rule(fixture.rule_snapshot)
    if rule.condition_id != score.canonical_refs.condition_id:
        raise X09VerticalSliceInputError("rule condition does not match score market")
    if any(book.event.source.venue != rule.venue for book in books):
        raise X09VerticalSliceInputError("book venue does not match rule snapshot")
    return source_events, score, books, rule


def _derived_event(
    *,
    event_type: str,
    stream: str,
    sequence: int,
    receive_at: datetime,
    source_at: datetime,
    score: EventEnvelopeV0,
    parents: tuple[str, ...],
    payload: dict[str, object],
    rule_snapshot_ref: str | None,
) -> EventEnvelopeV0:
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type=event_type,
        payload_schema_version="v0",
        source={
            "system": "x09-vertical-slice",
            "stream": stream,
            "venue": None,
            "sequence": sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time={
            "receive_at": _timestamp(receive_at),
            "receive_basis": "local_recorder",
            "source_at": _timestamp(source_at),
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs=score.canonical_refs.model_dump(
            mode="python", round_trip=True
        ),
        native_refs=[],
        lineage={"parent_event_ids": parents},
        experiment_id="X-09",
        rule_snapshot_ref=rule_snapshot_ref,
        quality_flags=[],
        payload=payload,
    )


def _first_executable(
    books: tuple[_ObservedBook, ...], not_before: datetime
) -> _ObservedBook:
    for observed in books:
        if (
            observed.book.local_receive_at >= not_before
            and not observed.book.suspended
        ):
            return observed
    raise X09VerticalSliceInputError(
        "no executable book at or after the required simulated time"
    )


def _consume(
    levels: tuple[DepthLevelV0, ...], quantity: Decimal, field: str
) -> tuple[Decimal, Decimal, int]:
    remaining = quantity
    notional = Decimal(0)
    consumed = 0
    for level in levels:
        take = min(remaining, level.quantity.to_decimal())
        if take <= 0:
            continue
        notional += take * level.price.to_decimal()
        remaining -= take
        consumed += 1
        if remaining == 0:
            break
    if remaining != 0:
        raise X09VerticalSliceInputError(f"insufficient executable {field} depth")
    return notional / quantity, notional, consumed


def _check_tick_size(
    books: tuple[_ObservedBook, ...], rule: VenueRuleSnapshotV0
) -> None:
    tick = rule.minimum_tick_size.to_decimal()
    for observed in books:
        for level in (*observed.book.bids, *observed.book.asks):
            if level.price.to_decimal() % tick != 0:
                raise X09VerticalSliceInputError(
                    "book price is not aligned to snapshot tick size"
                )


def _execute_once(fixture: X09FixtureV0) -> ReplayRunV0:
    source_events, score, books, rule = _prepare_fixture(fixture)
    config = fixture.config
    quantity = config.order_quantity.to_decimal()
    if quantity > config.maximum_order_quantity.to_decimal():
        raise X09VerticalSliceInputError("order rejected by frozen risk limit")
    if quantity < rule.minimum_order_size.to_decimal():
        raise X09VerticalSliceInputError("order is below snapshot minimum_order_size")
    if "MARKET" not in rule.order_types_supported:
        raise X09VerticalSliceInputError("snapshot does not authorize MARKET taker orders")
    _check_tick_size(books, rule)

    score_at = _parse_timestamp(score.time.receive_at)
    signal_at = score_at + _SIGNAL_DELAY
    signal = _derived_event(
        event_type="signal",
        stream="score-trigger-signal",
        sequence=1,
        receive_at=signal_at,
        source_at=score_at,
        score=score,
        parents=(score.event_id,),
        payload={
            "action": "BUY",
            "delay_microseconds": _timedelta_microseconds(_SIGNAL_DELAY),
            "trigger_event_id": score.event_id,
        },
        rule_snapshot_ref=None,
    )

    rule_ref = canonical_sha256(rule.model_dump(mode="json", round_trip=True))
    order_id = "order_x09_" + canonical_sha256(
        {
            "signal_event_id": signal.event_id,
            "quantity": _fixed_dump(config.order_quantity),
        }
    ).removeprefix("sha256:")[:24]
    order = _derived_event(
        event_type="simulated_order",
        stream="simulated-orders",
        sequence=2,
        receive_at=signal_at,
        source_at=signal_at,
        score=score,
        parents=(score.event_id, signal.event_id),
        payload={
            "order_id": order_id,
            "order_type": "MARKET",
            "quantity": _fixed_dump(config.order_quantity),
            "risk_decision": "APPROVED",
            "risk_limit": _fixed_dump(config.maximum_order_quantity),
            "side": "BUY",
            "trigger_event_id": score.event_id,
        },
        rule_snapshot_ref=rule_ref,
    )

    if _parse_timestamp(rule.fetched_at) > signal_at or _parse_timestamp(
        rule.effective_from
    ) > signal_at:
        raise X09VerticalSliceInputError(
            "venue rule snapshot is not point-in-time available"
        )
    venue_delay = _fixed_seconds(rule.seconds_delay, "seconds_delay")
    execution = _first_executable(
        books, signal_at + venue_delay + config.own_delay
    )
    vwap, gross_cost, levels_consumed = _consume(
        execution.book.asks, quantity, "ask"
    )
    if rule.fees_enabled:
        exponent = rule.fee_exponent.to_decimal()
        if exponent != exponent.to_integral_value():
            raise X09VerticalSliceInputError(
                "fee_exponent must be an integer for X-09 v0"
            )
        fee = (
            quantity
            * rule.fee_rate.to_decimal()
            * (vwap * (Decimal(1) - vwap)) ** int(exponent)
        )
    else:
        fee = Decimal(0)
    total_cost = gross_cost + fee
    fill = _derived_event(
        event_type="simulated_fill",
        stream="simulated-fills",
        sequence=3,
        receive_at=execution.book.local_receive_at,
        source_at=_parse_timestamp(
            execution.event.time.source_at or execution.event.time.receive_at
        ),
        score=score,
        parents=(order.event_id, execution.event.event_id),
        payload={
            "execution_book_event_id": execution.event.event_id,
            "fee": _fixed_dump(_fixed(fee)),
            "filled_quantity": _fixed_dump(config.order_quantity),
            "gross_cost": _fixed_dump(_fixed(gross_cost)),
            "levels_consumed": levels_consumed,
            "order_id": order_id,
            "total_cost": _fixed_dump(_fixed(total_cost)),
            "vwap": _fixed_dump(_fixed(vwap)),
        },
        rule_snapshot_ref=rule_ref,
    )

    mark = _first_executable(
        books, execution.book.local_receive_at + config.pnl_horizon
    )
    exit_vwap, gross_proceeds, exit_levels_consumed = _consume(
        mark.book.bids, quantity, "bid"
    )
    pnl_value = gross_proceeds - total_cost
    pnl = _derived_event(
        event_type="simulated_pnl",
        stream="simulated-pnl",
        sequence=4,
        receive_at=mark.book.local_receive_at,
        source_at=_parse_timestamp(mark.event.time.source_at or mark.event.time.receive_at),
        score=score,
        parents=(fill.event_id, mark.event.event_id),
        payload={
            "exit_book_event_id": mark.event.event_id,
            "exit_levels_consumed": exit_levels_consumed,
            "exit_vwap": _fixed_dump(_fixed(exit_vwap)),
            "filled_quantity": _fixed_dump(config.order_quantity),
            "gross_proceeds": _fixed_dump(_fixed(gross_proceeds)),
            "order_id": order_id,
            "pnl": _fixed_dump(_fixed(pnl_value)),
            "total_cost": _fixed_dump(_fixed(total_cost)),
        },
        rule_snapshot_ref=rule_ref,
    )
    return build_replay((*source_events, signal, order, fill, pnl))


def run_vertical_slice(fixture: X09FixtureV0) -> X09HarnessResultV0:
    """Run the frozen engineering harness twice and require Levels 1 and 2."""

    first = _execute_once(fixture)
    second = _execute_once(fixture)
    assert_replay_deterministic(first, second)
    return X09HarnessResultV0(
        harness_status="HARNESS_PASS",
        experiment_status="EXPERIMENT_BLOCKED",
        first_run=first,
        second_run=second,
    )


def _configuration_material(config: X09ConfigV0) -> dict[str, object]:
    return {
        "canonical_id_snapshot_sha256": config.canonical_id_snapshot_sha256,
        "dependency_lock_sha256": config.dependency_lock_sha256,
        "maximum_order_quantity": _fixed_dump(config.maximum_order_quantity),
        "order_quantity": _fixed_dump(config.order_quantity),
        "own_delay_microseconds": _timedelta_microseconds(config.own_delay),
        "pnl_horizon_microseconds": _timedelta_microseconds(config.pnl_horizon),
        "random_seed": config.random_seed,
        "signal": "buy five seconds after score",
    }


def compute_x09_runtime_hashes(fixture: X09FixtureV0) -> X09RuntimeHashesV0:
    """Bind formal execution to code and the exact semantic fixture snapshot."""

    if not isinstance(fixture, X09FixtureV0):
        raise X09VerticalSliceInputError("fixture must be X09FixtureV0")
    ordered_events, _, _, rule = _prepare_fixture(fixture)
    code_paths = {
        "contracts.py": Path(contracts_module.__file__),
        "execution.py": Path(execution_module.__file__),
        "experiments.py": Path(experiments_module.__file__),
        "replay.py": Path(replay_module.__file__),
        "vertical_slice.py": Path(__file__),
    }
    code_manifest = {
        name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(code_paths.items())
    }
    data_material = {
        "configuration": _configuration_material(fixture.config),
        "events": [
            event.model_dump(mode="json", round_trip=True) for event in ordered_events
        ],
        "rule_snapshot": rule.model_dump(mode="json", round_trip=True),
    }
    return X09RuntimeHashesV0(
        code_sha256=canonical_sha256(code_manifest),
        data_sha256=canonical_sha256(data_material),
    )


def _formal_blockers(
    registry: dict[str, dict[str, object]], hashes: X09RuntimeHashesV0
) -> tuple[str, ...]:
    card = registry.get("X-09")
    if not isinstance(card, dict):
        return ("X-09 registration is missing",)
    blockers: list[str] = []
    scopes = card.get("authorization_scopes")
    scope = scopes.get("formal_result") if isinstance(scopes, dict) else None
    if not isinstance(scope, dict):
        blockers.append("X-09 formal_result scope is not authorized")
        required_lock_ids: tuple[str, ...] = ()
    else:
        if scope.get("authorized") is not True:
            blockers.append("X-09 formal_result scope is not authorized")
        raw_required = scope.get("required_lock_ids")
        required_lock_ids = (
            tuple(raw_required) if isinstance(raw_required, list) else ()
        )
    raw_locks = card.get("registration_locks")
    locks = (
        {
            lock.get("id"): lock
            for lock in raw_locks
            if isinstance(lock, dict) and isinstance(lock.get("id"), str)
        }
        if isinstance(raw_locks, list)
        else {}
    )
    for lock_id in required_lock_ids:
        lock = locks.get(lock_id)
        if not isinstance(lock, dict) or lock.get("status") != "resolved":
            blockers.append(f"X-09 unresolved registration lock: {lock_id}")

    dependencies = card.get("dependencies")
    if isinstance(dependencies, list):
        for dependency in dependencies:
            dependent_card = registry.get(dependency)
            if not isinstance(dependent_card, dict) or dependent_card.get("status") != "done":
                blockers.append(f"X-09 dependency is incomplete: {dependency}")
    else:
        blockers.append("X-09 dependency list is invalid")

    preregistered_inputs = card.get("preregistered_inputs")
    preregistered = (
        preregistered_inputs.get("formal_result")
        if isinstance(preregistered_inputs, dict)
        else None
    )
    if not isinstance(preregistered, dict):
        blockers.append("X-09 formal inputs are not preregistered")
    else:
        if preregistered.get("code_sha256") != hashes.code_sha256:
            blockers.append("X-09 preregistered code hash does not match runtime")
        if preregistered.get("data_sha256") != hashes.data_sha256:
            blockers.append("X-09 preregistered data hash does not match runtime")
    return tuple(blockers)


def run_x09_formal(
    program_root: str | Path, fixture: X09FixtureV0
) -> X09HarnessResultV0:
    """Read the registry and fail closed before producing formal-run evidence."""

    registry = load_experiment_registry(program_root)
    hashes = compute_x09_runtime_hashes(fixture)
    blockers = _formal_blockers(registry, hashes)
    if blockers:
        raise X09ExperimentBlocked(blockers)
    # A successful call produces deterministic candidate evidence only. Registry
    # result acceptance remains outside this harness, so the conservative result
    # status continues to be EXPERIMENT_BLOCKED.
    return run_vertical_slice(fixture)


__all__ = [
    "X09ConfigV0",
    "X09ExperimentBlocked",
    "X09FixtureV0",
    "X09HarnessResultV0",
    "X09RuntimeHashesV0",
    "X09VerticalSliceInputError",
    "compute_x09_runtime_hashes",
    "run_vertical_slice",
    "run_x09_formal",
]
