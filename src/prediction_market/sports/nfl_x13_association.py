"""Fail-closed source-time association executor for NFL X-13.

This module is intentionally pure and in-memory.  It does not fetch data,
persist artifacts, estimate receive latency, infer an order inside a source
second, or turn association candidates into execution claims.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
    SourceInterval,
    associate_intervals,
    build_layer_g_interval,
)
from prediction_market.sports.nfl_x13_market_cleaning import (
    MarketCleaningError,
    deduplicate_cleaned_observations_v1,
)
from prediction_market.sports.nfl_x13_replay import (
    CanonicalGameEventV1,
    FinalizedEpisodeV1,
    X13GameLedgerV1,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_DELAY_SCENARIOS_SECONDS,
    X13_HORIZONS_SECONDS,
    X13_STATUS,
    AssociationCandidateV1,
    MarketContractV1,
    canonical_sha256,
)

_ZERO = Decimal(0)
_DIRECTIONAL_AUDIT_ORDER = (
    "orientation",
    "adjudication",
    "score_possession",
    "same_second",
    "contamination",
    "stale_or_wrong_family",
    "diagnostic_fair_delta",
)
_PROHIBITED_CLAIMS = (
    "causality",
    "execution",
    "lead_lag",
    "real_latency",
    "tradeable_alpha",
)
_ANALYSIS_KINDS = frozenset(
    {"primary_pooled", "binary_moderator", "named_subtype", "sequence"}
)
_ESTIMANDS = frozenset({"pooled_mean", "binary_difference", "group_mean"})


@dataclass(frozen=True, slots=True)
class RegisteredHypothesisV1:
    hypothesis_id: str
    hypothesis_family: str
    analysis_kind: str
    approved_groups: tuple[str, ...]
    estimand: str

    def __post_init__(self) -> None:
        for field in ("hypothesis_id", "hypothesis_family"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise TypeError(f"{field} must be nonempty text")
        if self.analysis_kind not in _ANALYSIS_KINDS:
            raise ValueError("registered analysis_kind is unknown")
        if (
            type(self.approved_groups) is not tuple
            or not self.approved_groups
            or any(
                type(group) is not str or not group for group in self.approved_groups
            )
            or len(set(self.approved_groups)) != len(self.approved_groups)
        ):
            raise ValueError("approved_groups must contain unique nonempty strings")
        if self.estimand not in _ESTIMANDS:
            raise ValueError("registered estimand is unknown")
        expected_estimand = (
            "pooled_mean"
            if self.analysis_kind == "primary_pooled"
            else "binary_difference"
            if self.analysis_kind == "binary_moderator"
            else "group_mean"
        )
        if self.estimand != expected_estimand:
            raise ValueError("estimand does not match the registered analysis kind")
        if self.analysis_kind == "binary_moderator" and len(self.approved_groups) != 2:
            raise ValueError("binary moderator requires exactly two ordered groups")


@dataclass(frozen=True, slots=True)
class RegisteredAnalysisLockV1:
    experiment_id: str
    status: str
    version: str
    bootstrap_seed: int
    hypotheses: tuple[RegisteredHypothesisV1, ...]

    def __post_init__(self) -> None:
        if self.experiment_id != "X-13":
            raise ValueError("registered lock experiment_id must be X-13")
        if self.status != X13_STATUS:
            raise ValueError("registered lock status is not X-13 status")
        if type(self.version) is not str or not self.version:
            raise TypeError("registered lock version must be nonempty")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise TypeError("registered bootstrap_seed must be nonnegative")
        if (
            type(self.hypotheses) is not tuple
            or not self.hypotheses
            or any(
                not isinstance(item, RegisteredHypothesisV1) for item in self.hypotheses
            )
        ):
            raise TypeError("hypotheses must contain RegisteredHypothesisV1 values")
        hypothesis_ids = tuple(item.hypothesis_id for item in self.hypotheses)
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("registered lock contains duplicate hypothesis")
        if hypothesis_ids != tuple(sorted(hypothesis_ids)):
            raise ValueError("registered hypotheses must be sorted by hypothesis_id")

    @property
    def approved_whitelist(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (item.hypothesis_id, item.approved_groups) for item in self.hypotheses
        )

    @property
    def lock_id(self) -> str:
        return canonical_sha256(
            {
                "bootstrap_seed": self.bootstrap_seed,
                "experiment_id": self.experiment_id,
                "hypotheses": [
                    {
                        "analysis_kind": item.analysis_kind,
                        "approved_groups": item.approved_groups,
                        "estimand": item.estimand,
                        "hypothesis_family": item.hypothesis_family,
                        "hypothesis_id": item.hypothesis_id,
                    }
                    for item in self.hypotheses
                ],
                "status": self.status,
                "version": self.version,
            }
        )


X13_REGISTERED_ANALYSIS_LOCK_V1 = RegisteredAnalysisLockV1(
    experiment_id="X-13",
    status=X13_STATUS,
    version="v1",
    bootstrap_seed=20260722,
    hypotheses=(
        RegisteredHypothesisV1(
            hypothesis_id="event_superclass",
            hypothesis_family="event_superclass",
            analysis_kind="named_subtype",
            approved_groups=("score", "turnover", "routine"),
            estimand="group_mean",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="fair_delta_by_both_venues_active",
            hypothesis_family="fair_delta_moderators",
            analysis_kind="binary_moderator",
            approved_groups=(
                "single_venue_active",
                "both_venues_active",
            ),
            estimand="binary_difference",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="fair_delta_by_late_close_state",
            hypothesis_family="fair_delta_moderators",
            analysis_kind="binary_moderator",
            approved_groups=("not_late_close", "late_close"),
            estimand="binary_difference",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="fair_delta_by_pre_event_liquidity",
            hypothesis_family="fair_delta_moderators",
            analysis_kind="binary_moderator",
            approved_groups=("low", "high"),
            estimand="binary_difference",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="fair_delta_by_semantic_salience",
            hypothesis_family="fair_delta_moderators",
            analysis_kind="binary_moderator",
            approved_groups=("low", "high"),
            estimand="binary_difference",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="sequence",
            hypothesis_family="sequence",
            analysis_kind="sequence",
            approved_groups=(
                "turnover_to_score",
                "answer_score",
                "back_to_back_turnovers",
                "red_zone_empty",
                "two_minute_drive",
                "overtime",
            ),
            estimand="group_mean",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="signed_diagnostic_fair_delta_by_venue",
            hypothesis_family="fair_delta_primary",
            analysis_kind="binary_moderator",
            approved_groups=("polymarket", "kalshi"),
            estimand="binary_difference",
        ),
        RegisteredHypothesisV1(
            hypothesis_id="signed_diagnostic_fair_delta_pooled",
            hypothesis_family="fair_delta_primary",
            analysis_kind="primary_pooled",
            approved_groups=("pooled",),
            estimand="pooled_mean",
        ),
    ),
)
X13_APPROVED_ANALYSIS_WHITELIST = X13_REGISTERED_ANALYSIS_LOCK_V1.approved_whitelist


class AssociationEvidenceError(ValueError):
    """The supplied evidence cannot support a Layer A association."""


def _utc_timestamp(value: str | None, *, field: str) -> datetime:
    if type(value) is not str or not value:
        raise AssociationEvidenceError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AssociationEvidenceError(f"{field} is not RFC3339") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AssociationEvidenceError(f"{field} is not UTC")
    return parsed.astimezone(UTC)


def _seconds(value: timedelta) -> Decimal:
    return (
        Decimal(value.days) * Decimal(86400)
        + Decimal(value.seconds)
        + Decimal(value.microseconds) / Decimal(1000000)
    )


def _contract_native_id(contract: MarketContractV1) -> str:
    return contract.contract_id.removeprefix(f"{contract.venue}:")


def _event_superclass(episode: FinalizedEpisodeV1) -> str:
    if episode.score_points > 0 and episode.turnover:
        return "score_turnover"
    if episode.score_points > 0 or episode.episode_type in {
        "touchdown",
        "field_goal",
        "safety",
    }:
        return "score"
    if episode.turnover:
        return "turnover"
    return "routine"


def _analysis_episode(episode: FinalizedEpisodeV1) -> bool:
    return (
        not episode.audit_only
        and not episode.nullified
        and episode.episode_type != "administrative"
    )


@dataclass(frozen=True, slots=True)
class LayerGEpisodeIntervalV1:
    game_id: str
    episode_id: str
    source_play_id: str
    delay_scenario_seconds: int
    source_interval: SourceInterval
    event_superclass: str
    event_subtype: str


@dataclass(frozen=True, slots=True)
class LayerGAuditV1:
    status: Literal["PASS", "FAIL"]
    game_id: str | None
    eligible_episode_count: int
    excluded_episode_count: int
    episode_intervals: tuple[LayerGEpisodeIntervalV1, ...]
    errors: tuple[str, ...]
    source_time_semantics: str = "G=[source_time,source_time+1_second+delay_scenario]"
    delay_scenarios_seconds: tuple[int, ...] = X13_DELAY_SCENARIOS_SECONDS
    real_receive_latency_claim_permitted: Literal[False] = False


def audit_layer_g(ledger: X13GameLedgerV1) -> LayerGAuditV1:
    """Audit episode source time and bind it only to its own first source row."""

    if not isinstance(ledger, X13GameLedgerV1):
        raise TypeError("ledger must be X13GameLedgerV1")
    errors: list[str] = []
    game_ids = {event.game_id for event in ledger.events} | {
        episode.game_id for episode in ledger.episodes
    }
    game_id = next(iter(game_ids)) if len(game_ids) == 1 else None
    if game_id is None:
        errors.append("ledger_must_contain_exactly_one_game")

    by_key: dict[tuple[str, str], CanonicalGameEventV1] = {}
    for event in ledger.events:
        key = (event.game_id, event.play_id)
        if key in by_key:
            errors.append("duplicate_event_play_id")
        else:
            by_key[key] = event

    seen_episode_ids: set[str] = set()
    intervals: list[LayerGEpisodeIntervalV1] = []
    eligible_count = 0
    excluded_count = 0
    for episode in ledger.episodes:
        if episode.episode_id in seen_episode_ids:
            errors.append("duplicate_episode_id")
        seen_episode_ids.add(episode.episode_id)
        if not _analysis_episode(episode):
            excluded_count += 1
            continue
        eligible_count += 1
        if not episode.play_ids or len(set(episode.play_ids)) != len(episode.play_ids):
            errors.append("episode_play_identity_invalid")
            continue
        members = [
            by_key.get((episode.game_id, play_id)) for play_id in episode.play_ids
        ]
        if any(member is None for member in members):
            errors.append("episode_member_not_found")
            continue
        typed_members = tuple(member for member in members if member is not None)
        first = typed_members[0]
        last = typed_members[-1]
        if episode.pre_state != first.pre_state:
            errors.append("episode_pre_state_not_bound_to_first_member")
        expected_post_state = (
            last.post_state
            if episode.review_result == "reversed"
            else next(
                (
                    member.post_state
                    for member in reversed(typed_members)
                    if member.post_state is not None
                ),
                None,
            )
        )
        if (
            episode.post_state is None
            or expected_post_state is None
            or episode.post_state != expected_post_state
        ):
            errors.append("episode_post_state_not_bound_to_episode_evidence")
        try:
            source_time = _utc_timestamp(
                first.source_time_start_utc,
                field="episode source_time_start_utc",
            )
            if first.source_time_end_utc is not None:
                source_end = _utc_timestamp(
                    first.source_time_end_utc,
                    field="episode source_time_end_utc",
                )
                if source_end < source_time:
                    errors.append("source_time_interval_reversed")
                    continue
        except AssociationEvidenceError:
            errors.append("eligible_episode_missing_source_time")
            continue
        for delay in X13_DELAY_SCENARIOS_SECONDS:
            intervals.append(
                LayerGEpisodeIntervalV1(
                    game_id=episode.game_id,
                    episode_id=episode.episode_id,
                    source_play_id=first.play_id,
                    delay_scenario_seconds=delay,
                    source_interval=build_layer_g_interval(
                        source_time,
                        delay_seconds=Decimal(delay),
                    ),
                    event_superclass=_event_superclass(episode),
                    event_subtype=episode.episode_type,
                )
            )

    unique_errors = tuple(sorted(set(errors)))
    expected_interval_count = eligible_count * len(X13_DELAY_SCENARIOS_SECONDS)
    if (
        len(intervals) != expected_interval_count
        and eligible_count
        and not unique_errors
    ):
        unique_errors = ("layer_g_interval_cardinality_mismatch",)
    return LayerGAuditV1(
        status="FAIL" if unique_errors else "PASS",
        game_id=game_id,
        eligible_episode_count=eligible_count,
        excluded_episode_count=excluded_count,
        episode_intervals=tuple(
            sorted(
                intervals,
                key=lambda item: (
                    item.source_interval.start,
                    item.episode_id,
                    item.delay_scenario_seconds,
                ),
            )
        ),
        errors=unique_errors,
    )


@dataclass(frozen=True, slots=True)
class LayerMAuditV1:
    status: Literal["PASS", "FAIL"]
    game_id: str
    trade_count: int
    candle_count: int
    derived_count: int
    excluded_contract_ids: tuple[str, ...]
    observation_contract_bindings: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]
    trade_source_time_semantics: str = "M_trade=[trade_second,trade_second+1s)"
    candle_source_time_semantics: str = "M_kalshi_candle=complete_one_minute_interval"
    local_receive_time_present: Literal[False] = False
    real_latency_claim_permitted: Literal[False] = False
    point_order_within_source_second_permitted: Literal[False] = False
    candle_point_price_permitted: Literal[False] = False


def _bind_observation_contract(
    observation: MarketObservation,
    contracts: Sequence[MarketContractV1],
) -> MarketContractV1 | None:
    exact = [
        contract
        for contract in contracts
        if contract.venue == observation.venue
        and observation.raw_market_id
        in {contract.contract_id, _contract_native_id(contract)}
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    logical = [
        contract
        for contract in contracts
        if contract.venue == observation.venue
        and observation.logical_market_id is not None
        and contract.logical_market_id == observation.logical_market_id
    ]
    return logical[0] if len(logical) == 1 else None


def audit_layer_m(
    contracts: Sequence[MarketContractV1],
    observations: Sequence[MarketObservation],
    *,
    game_id: str,
) -> LayerMAuditV1:
    """Audit market intervals without manufacturing point time or BBO."""

    if not isinstance(contracts, Sequence) or isinstance(contracts, (str, bytes)):
        raise TypeError("contracts must be a sequence")
    if not contracts or any(
        not isinstance(contract, MarketContractV1) for contract in contracts
    ):
        raise TypeError("contracts must contain MarketContractV1")
    if type(game_id) is not str or not game_id:
        raise TypeError("game_id must be nonempty text")
    if len({contract.contract_id for contract in contracts}) != len(contracts):
        raise AssociationEvidenceError("duplicate contract_id")
    errors: list[str] = []
    excluded = tuple(
        sorted(
            contract.contract_id
            for contract in contracts
            if not contract.analysis_eligible
            or contract.dependency_game_ids != (game_id,)
        )
    )
    try:
        deduplicated = deduplicate_cleaned_observations_v1(observations)
    except (MarketCleaningError, TypeError) as error:
        return LayerMAuditV1(
            status="FAIL",
            game_id=game_id,
            trade_count=0,
            candle_count=0,
            derived_count=0,
            excluded_contract_ids=excluded,
            observation_contract_bindings=(),
            errors=(f"observation_deduplication_failed:{error}",),
        )

    bindings: list[tuple[str, str]] = []
    trade_count = 0
    candle_count = 0
    derived_count = 0
    for observation in deduplicated:
        contract = _bind_observation_contract(observation, contracts)
        if contract is None:
            errors.append("observation_contract_identity_unresolved")
        else:
            bindings.append((observation.observation_id, contract.contract_id))
            if observation.outcome not in contract.outcomes:
                errors.append("observation_outcome_not_in_contract")
        for field, value in (
            ("price", observation.price),
            ("bid", observation.bid),
            ("ask", observation.ask),
        ):
            if value is not None and (
                type(value) is not Decimal
                or not value.is_finite()
                or value < 0
                or value > 1
            ):
                errors.append(f"{field}_outside_probability_bounds")
        if (
            type(observation.size) is not Decimal
            or not observation.size.is_finite()
            or observation.size < 0
        ):
            errors.append("observation_size_invalid")
        if (
            type(observation.bid) is Decimal
            and type(observation.ask) is Decimal
            and observation.bid > observation.ask
        ):
            errors.append("crossed_bbo")
        width = observation.source_interval.end - observation.source_interval.start
        if observation.kind == "trade":
            if observation.provenance == "observed":
                trade_count += 1
            if (
                width != timedelta(seconds=1)
                or observation.source_interval.end_inclusive
            ):
                errors.append("trade_interval_not_one_second")
            if (
                observation.price is None
                or type(observation.size) is not Decimal
                or observation.size <= 0
            ):
                errors.append("trade_missing_actual_price_or_size")
            if observation.bid is not None or observation.ask is not None:
                errors.append("trade_contains_fabricated_bbo")
        elif observation.kind == "kalshi_1m_candle_bbo":
            candle_count += 1
            if (
                width != timedelta(minutes=1)
                or observation.source_interval.end_inclusive
            ):
                errors.append("candle_interval_not_one_minute")
            if (
                observation.price is not None
                or observation.bid is None
                or observation.ask is None
            ):
                errors.append("candle_contains_fake_point_or_missing_bbo")
        else:
            errors.append("unknown_observation_kind")
        if observation.provenance == "derived_complement":
            derived_count += 1
            if (
                observation.render_semantics != "dashed"
                or observation.bid is not None
                or observation.ask is not None
                or observation.derived_from_observation_id is None
            ):
                errors.append("derived_complement_marked_executable")
        elif (
            observation.provenance != "observed"
            or observation.render_semantics != "solid"
            or observation.derived_from_observation_id is not None
        ):
            errors.append("observation_provenance_invalid")

    unique_errors = tuple(sorted(set(errors)))
    return LayerMAuditV1(
        status="FAIL" if unique_errors else "PASS",
        game_id=game_id,
        trade_count=trade_count,
        candle_count=candle_count,
        derived_count=derived_count,
        excluded_contract_ids=excluded,
        observation_contract_bindings=tuple(sorted(bindings)),
        errors=unique_errors,
    )


@dataclass(frozen=True, slots=True)
class SourceTimelineItemV1:
    layer: Literal["G", "M"]
    identity: str
    kind: str
    venue: str | None
    source_interval: SourceInterval
    interval_width_seconds: Decimal
    order_ambiguous: bool
    provenance: str
    within_overlap_order_permitted: Literal[False] = False

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "identity": self.identity,
            "kind": self.kind,
            "venue": self.venue,
            "interval_start_utc": _rfc3339(self.source_interval.start),
            "interval_end_utc": _rfc3339(self.source_interval.end),
            "end_inclusive": self.source_interval.end_inclusive,
            "interval_width_seconds": str(self.interval_width_seconds),
            "order_ambiguous": self.order_ambiguous,
            "provenance": self.provenance,
            "within_overlap_order_permitted": (
                self.within_overlap_order_permitted
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceTimelineV1:
    delay_scenario_seconds: int
    items: tuple[SourceTimelineItemV1, ...]
    ordering_semantics: str = (
        "sorted_for_render_only; overlapping_intervals_have_no_asserted_order"
    )

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                **item.to_dict(),
                "delay_seconds": self.delay_scenario_seconds,
            }
            for item in self.items
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "delay_scenario_seconds": self.delay_scenario_seconds,
            "ordering_semantics": self.ordering_semantics,
            "rows": self.to_records(),
        }


def _rfc3339(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _overlap_counts(
    intervals: Sequence[SourceInterval],
) -> tuple[int, ...]:
    starts = tuple(sorted(interval.start for interval in intervals))
    ends = tuple(
        sorted(
            (interval.end, interval.end_inclusive)
            for interval in intervals
        )
    )
    counts: list[int] = []
    for interval in intervals:
        started_by_end = (
            bisect_right(starts, interval.end)
            if interval.end_inclusive
            else bisect_left(starts, interval.end)
        )
        ended_before_start = bisect_right(
            ends,
            (interval.start, False),
        )
        count = started_by_end - ended_before_start - 1
        if count < 0:
            raise AssociationEvidenceError(
                "interval overlap count became negative"
            )
        counts.append(count)
    return tuple(counts)


def _overlap_flags(
    intervals: Sequence[SourceInterval],
) -> tuple[bool, ...]:
    return tuple(count > 0 for count in _overlap_counts(intervals))


def _timeline(
    *,
    delay: int,
    layer_g_intervals: Sequence[LayerGEpisodeIntervalV1],
    observations: Sequence[MarketObservation],
) -> SourceTimelineV1:
    raw: list[SourceTimelineItemV1] = [
        SourceTimelineItemV1(
            layer="G",
            identity=item.episode_id,
            kind=item.event_subtype,
            venue=None,
            source_interval=item.source_interval,
            interval_width_seconds=_seconds(
                item.source_interval.end - item.source_interval.start
            ),
            order_ambiguous=False,
            provenance="observed_game_source_time_plus_scenario",
        )
        for item in layer_g_intervals
        if item.delay_scenario_seconds == delay
    ]
    raw.extend(
        SourceTimelineItemV1(
            layer="M",
            identity=observation.observation_id,
            kind=observation.kind,
            venue=observation.venue,
            source_interval=observation.source_interval,
            interval_width_seconds=_seconds(
                observation.source_interval.end - observation.source_interval.start
            ),
            order_ambiguous=False,
            provenance=observation.provenance,
        )
        for observation in observations
    )
    raw.sort(
        key=lambda item: (
            item.source_interval.start,
            item.source_interval.end,
            item.layer,
            item.identity,
        )
    )
    ambiguous = _overlap_flags(
        tuple(item.source_interval for item in raw)
    )
    return SourceTimelineV1(
        delay_scenario_seconds=delay,
        items=tuple(
            replace(item, order_ambiguous=ambiguous[index])
            for index, item in enumerate(raw)
        ),
    )


@dataclass(frozen=True, slots=True)
class GameTimeIntervalRowV1:
    interval_id: str
    game_id: str
    episode_id: str
    source_play_id: str
    delay_seconds: int
    interval_start_utc: str
    interval_end_utc: str
    end_inclusive: Literal[True]
    provenance: str
    order_ambiguous: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "interval_id": self.interval_id,
            "game_id": self.game_id,
            "episode_id": self.episode_id,
            "source_play_id": self.source_play_id,
            "delay_seconds": self.delay_seconds,
            "interval_start_utc": self.interval_start_utc,
            "interval_end_utc": self.interval_end_utc,
            "end_inclusive": self.end_inclusive,
            "provenance": self.provenance,
            "order_ambiguous": self.order_ambiguous,
        }


@dataclass(frozen=True, slots=True)
class GameTimeIntervalsV1:
    game_id: str
    rows: tuple[GameTimeIntervalRowV1, ...]
    status: Literal["PASS"] = "PASS"
    schema_version: str = "game_time_intervals_v1"

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "game_id": self.game_id,
            "row_count": len(self.rows),
            "rows": self.to_records(),
        }


@dataclass(frozen=True, slots=True)
class MarketTimeIntervalRowV1:
    interval_id: str
    game_id: str
    contract_id: str
    observation_id: str
    logical_market_id: str | None
    venue: str
    outcome: str
    observation_kind: Literal[
        "actual_trade",
        "kalshi_1m_candle",
        "derived_complement",
    ]
    interval_start_utc: str
    interval_end_utc: str
    end_inclusive: Literal[False]
    provenance: str
    actual_observation: bool
    actual_trade: bool
    bbo_available: bool
    bbo_tick: Literal[False]
    executable: Literal[False]
    order_ambiguous: bool
    forward_filled: Literal[False] = False

    def to_dict(self) -> dict[str, object]:
        return {
            "interval_id": self.interval_id,
            "game_id": self.game_id,
            "contract_id": self.contract_id,
            "observation_id": self.observation_id,
            "logical_market_id": self.logical_market_id,
            "venue": self.venue,
            "outcome": self.outcome,
            "observation_kind": self.observation_kind,
            "interval_start_utc": self.interval_start_utc,
            "interval_end_utc": self.interval_end_utc,
            "end_inclusive": self.end_inclusive,
            "provenance": self.provenance,
            "actual_observation": self.actual_observation,
            "actual_trade": self.actual_trade,
            "bbo_available": self.bbo_available,
            "bbo_tick": self.bbo_tick,
            "executable": self.executable,
            "order_ambiguous": self.order_ambiguous,
            "forward_filled": self.forward_filled,
        }


@dataclass(frozen=True, slots=True)
class MarketTimeIntervalsV1:
    game_id: str
    rows: tuple[MarketTimeIntervalRowV1, ...]
    status: Literal["PASS"] = "PASS"
    schema_version: str = "market_time_intervals_v1"

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "game_id": self.game_id,
            "row_count": len(self.rows),
            "rows": self.to_records(),
        }


@dataclass(frozen=True, slots=True)
class SourceTimelineRowV1:
    timeline_id: str
    game_id: str
    layer: Literal["G", "M"]
    identity: str
    interval_id: str
    interval_start_utc: str
    interval_end_utc: str
    end_inclusive: bool
    delay_seconds: int
    kind: str
    venue: str | None
    provenance: str
    order_ambiguous: bool
    forward_filled: Literal[False] = False
    asserted_order: None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline_id": self.timeline_id,
            "game_id": self.game_id,
            "layer": self.layer,
            "identity": self.identity,
            "interval_id": self.interval_id,
            "interval_start_utc": self.interval_start_utc,
            "interval_end_utc": self.interval_end_utc,
            "end_inclusive": self.end_inclusive,
            "delay_seconds": self.delay_seconds,
            "kind": self.kind,
            "venue": self.venue,
            "provenance": self.provenance,
            "order_ambiguous": self.order_ambiguous,
            "forward_filled": self.forward_filled,
            "asserted_order": self.asserted_order,
        }


@dataclass(frozen=True, slots=True)
class SourceTimelineTableV1:
    game_id: str
    rows: tuple[SourceTimelineRowV1, ...]
    schema_version: str = "source_timeline_v1"
    ordering_semantics: str = (
        "render_sort_only; overlap_and_same_second_have_no_asserted_order"
    )

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "row_count": len(self.rows),
            "ordering_semantics": self.ordering_semantics,
            "rows": self.to_records(),
        }


@dataclass(frozen=True, slots=True)
class TimelineAuditRowV1:
    timeline_audit_id: str
    timeline_id: str
    game_id: str
    delay_seconds: int
    layer: Literal["G", "M"]
    identity: str
    reason_codes: tuple[
        Literal["same_source_second", "overlapping_source_intervals"],
        ...,
    ]
    overlap_count: int
    order_ambiguous: Literal[True] = True
    asserted_order: None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline_audit_id": self.timeline_audit_id,
            "timeline_id": self.timeline_id,
            "game_id": self.game_id,
            "delay_seconds": self.delay_seconds,
            "layer": self.layer,
            "identity": self.identity,
            "reason_codes": list(self.reason_codes),
            "overlap_count": self.overlap_count,
            "order_ambiguous": self.order_ambiguous,
            "asserted_order": self.asserted_order,
        }


@dataclass(frozen=True, slots=True)
class TimelineAuditV1:
    game_id: str
    rows: tuple[TimelineAuditRowV1, ...]
    status: Literal["PASS"] = "PASS"
    schema_version: str = "timeline_audit_v1"

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "game_id": self.game_id,
            "row_count": len(self.rows),
            "rows": self.to_records(),
        }


def _persisted_interval(
    *,
    start: str,
    end: str,
    end_inclusive: bool,
) -> SourceInterval:
    return SourceInterval(
        _utc_timestamp(start, field="interval_start_utc"),
        _utc_timestamp(end, field="interval_end_utc"),
        end_inclusive,
    )


def build_game_time_intervals_v1(
    ledger: X13GameLedgerV1,
) -> GameTimeIntervalsV1:
    """Build persistable Layer G rows from the audited frozen delay grid."""

    audit = audit_layer_g(ledger)
    if audit.status != "PASS" or audit.game_id is None:
        raise AssociationEvidenceError(
            "Layer G audit failed: " + ",".join(audit.errors)
        )
    rows: list[GameTimeIntervalRowV1] = []
    for item in audit.episode_intervals:
        start = _rfc3339(item.source_interval.start)
        end = _rfc3339(item.source_interval.end)
        interval_id = canonical_sha256(
            {
                "delay_seconds": item.delay_scenario_seconds,
                "end_inclusive": item.source_interval.end_inclusive,
                "episode_id": item.episode_id,
                "game_id": item.game_id,
                "interval_end_utc": end,
                "interval_start_utc": start,
                "schema_version": "game_time_intervals_v1",
                "source_play_id": item.source_play_id,
            }
        )
        rows.append(
            GameTimeIntervalRowV1(
                interval_id=interval_id,
                game_id=item.game_id,
                episode_id=item.episode_id,
                source_play_id=item.source_play_id,
                delay_seconds=item.delay_scenario_seconds,
                interval_start_utc=start,
                interval_end_utc=end,
                end_inclusive=True,
                provenance="observed_game_source_time_plus_delay_scenario",
            )
        )
    indexes_by_delay: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes_by_delay[row.delay_seconds].append(index)
    for indexes in indexes_by_delay.values():
        flags = _overlap_flags(
            tuple(
                audit.episode_intervals[index].source_interval
                for index in indexes
            )
        )
        for index, ambiguous in zip(indexes, flags, strict=True):
            rows[index] = replace(
                rows[index],
                order_ambiguous=ambiguous,
            )
    return GameTimeIntervalsV1(game_id=audit.game_id, rows=tuple(rows))


def _persisted_market_kind(
    observation: MarketObservation,
) -> Literal[
    "actual_trade",
    "kalshi_1m_candle",
    "derived_complement",
]:
    if observation.provenance == "derived_complement":
        return "derived_complement"
    if observation.kind == "trade" and observation.provenance == "observed":
        return "actual_trade"
    if (
        observation.kind == "kalshi_1m_candle_bbo"
        and observation.venue == "kalshi"
        and observation.provenance == "observed"
    ):
        return "kalshi_1m_candle"
    raise AssociationEvidenceError("market observation kind is not persistable")


def build_market_time_intervals_v1(
    contracts: Sequence[MarketContractV1],
    observations: Sequence[MarketObservation],
    *,
    game_id: str,
) -> MarketTimeIntervalsV1:
    """Build persistable Layer M rows without manufacturing point evidence."""

    audit = audit_layer_m(contracts, observations, game_id=game_id)
    if audit.status != "PASS":
        raise AssociationEvidenceError(
            "Layer M audit failed: " + ",".join(audit.errors)
        )
    deduplicated = deduplicate_cleaned_observations_v1(observations)
    binding_map = dict(audit.observation_contract_bindings)
    intervals = tuple(item.source_interval for item in deduplicated)
    ambiguity = _overlap_flags(intervals)
    rows: list[MarketTimeIntervalRowV1] = []
    for index, observation in enumerate(deduplicated):
        contract_id = binding_map.get(observation.observation_id)
        if contract_id is None:
            raise AssociationEvidenceError(
                "market observation contract binding disappeared"
            )
        kind = _persisted_market_kind(observation)
        start = _rfc3339(observation.source_interval.start)
        end = _rfc3339(observation.source_interval.end)
        interval_id = canonical_sha256(
            {
                "contract_id": contract_id,
                "end_inclusive": observation.source_interval.end_inclusive,
                "game_id": game_id,
                "interval_end_utc": end,
                "interval_start_utc": start,
                "observation_id": observation.observation_id,
                "schema_version": "market_time_intervals_v1",
            }
        )
        rows.append(
            MarketTimeIntervalRowV1(
                interval_id=interval_id,
                game_id=game_id,
                contract_id=contract_id,
                observation_id=observation.observation_id,
                logical_market_id=observation.logical_market_id,
                venue=observation.venue,
                outcome=observation.outcome,
                observation_kind=kind,
                interval_start_utc=start,
                interval_end_utc=end,
                end_inclusive=False,
                provenance=observation.provenance,
                actual_observation=observation.provenance == "observed",
                actual_trade=kind == "actual_trade",
                bbo_available=kind == "kalshi_1m_candle"
                and observation.bid is not None
                and observation.ask is not None,
                bbo_tick=False,
                executable=False,
                order_ambiguous=ambiguity[index],
            )
        )
    return MarketTimeIntervalsV1(
        game_id=game_id,
        rows=tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.interval_start_utc,
                    row.interval_end_utc,
                    row.contract_id,
                    row.observation_id,
                ),
            )
        ),
    )


def build_source_timeline_v1(
    game_intervals: GameTimeIntervalsV1,
    market_intervals: MarketTimeIntervalsV1,
) -> SourceTimelineTableV1:
    """Join G/M interval rows for audit only; never forward-fill or order overlap."""

    if not isinstance(game_intervals, GameTimeIntervalsV1):
        raise TypeError("game_intervals must be GameTimeIntervalsV1")
    if not isinstance(market_intervals, MarketTimeIntervalsV1):
        raise TypeError("market_intervals must be MarketTimeIntervalsV1")
    if game_intervals.game_id != market_intervals.game_id:
        raise AssociationEvidenceError("game and market interval game_id differ")
    delays = tuple(sorted({row.delay_seconds for row in game_intervals.rows}))
    rows: list[SourceTimelineRowV1] = []
    for delay in delays:
        layer_rows: list[
            tuple[
                Literal["G", "M"],
                str,
                str,
                str,
                str,
                bool,
                str,
                str | None,
                str,
            ]
        ] = []
        layer_rows.extend(
            (
                "G",
                row.episode_id,
                row.interval_id,
                row.interval_start_utc,
                row.interval_end_utc,
                row.end_inclusive,
                "game_event",
                None,
                row.provenance,
            )
            for row in game_intervals.rows
            if row.delay_seconds == delay
        )
        layer_rows.extend(
            (
                "M",
                row.observation_id,
                row.interval_id,
                row.interval_start_utc,
                row.interval_end_utc,
                row.end_inclusive,
                row.observation_kind,
                row.venue,
                row.provenance,
            )
            for row in market_intervals.rows
        )
        layer_rows.sort(
            key=lambda row: (
                row[3],
                row[4],
                row[0],
                row[1],
            )
        )
        local_intervals = tuple(
            _persisted_interval(
                start=row[3],
                end=row[4],
                end_inclusive=row[5],
            )
            for row in layer_rows
        )
        flags = _overlap_flags(local_intervals)
        for index, (
            layer,
            identity,
            interval_id,
            start,
            end,
            end_inclusive,
            kind,
            venue,
            provenance,
        ) in enumerate(layer_rows):
            timeline_id = canonical_sha256(
                {
                    "delay_seconds": delay,
                    "game_id": game_intervals.game_id,
                    "identity": identity,
                    "interval_id": interval_id,
                    "layer": layer,
                    "schema_version": "source_timeline_v1",
                }
            )
            rows.append(
                SourceTimelineRowV1(
                    timeline_id=timeline_id,
                    game_id=game_intervals.game_id,
                    layer=layer,
                    identity=identity,
                    interval_id=interval_id,
                    interval_start_utc=start,
                    interval_end_utc=end,
                    end_inclusive=end_inclusive,
                    delay_seconds=delay,
                    kind=kind,
                    venue=venue,
                    provenance=provenance,
                    order_ambiguous=flags[index],
                )
            )
    return SourceTimelineTableV1(
        game_id=game_intervals.game_id,
        rows=tuple(rows),
    )


def build_timeline_audit_rows_v1(
    timeline: SourceTimelineTableV1,
) -> TimelineAuditV1:
    """Persist every overlap pair that prevents a source-time order claim."""

    if not isinstance(timeline, SourceTimelineTableV1):
        raise TypeError("timeline must be SourceTimelineTableV1")
    result: list[TimelineAuditRowV1] = []
    by_delay: dict[int, list[SourceTimelineRowV1]] = defaultdict(list)
    for row in timeline.rows:
        by_delay[row.delay_seconds].append(row)
    for delay, rows in sorted(by_delay.items()):
        intervals = tuple(
            _persisted_interval(
                start=row.interval_start_utc,
                end=row.interval_end_utc,
                end_inclusive=row.end_inclusive,
            )
            for row in rows
        )
        overlap_counts = _overlap_counts(intervals)
        flags = tuple(count > 0 for count in overlap_counts)
        if tuple(row.order_ambiguous for row in rows) != flags:
            raise AssociationEvidenceError(
                "timeline ambiguity flags do not match interval evidence"
            )
        start_counts = Counter(row.interval_start_utc for row in rows)
        for row, overlap_count in zip(
            rows,
            overlap_counts,
            strict=True,
        ):
            if overlap_count == 0:
                continue
            reason_codes: list[str] = []
            if start_counts[row.interval_start_utc] > 1:
                reason_codes.append("same_source_second")
            reason_codes.append("overlapping_source_intervals")
            result.append(
                TimelineAuditRowV1(
                    timeline_audit_id=canonical_sha256(
                        {
                            "schema_version": "timeline_audit_v1",
                            "timeline_id": row.timeline_id,
                        }
                    ),
                    timeline_id=row.timeline_id,
                    game_id=timeline.game_id,
                    delay_seconds=delay,
                    layer=row.layer,
                    identity=row.identity,
                    reason_codes=tuple(reason_codes),  # type: ignore[arg-type]
                    overlap_count=overlap_count,
                )
            )
    return TimelineAuditV1(game_id=timeline.game_id, rows=tuple(result))


def _direction(delta: Decimal | None) -> str:
    if delta is None:
        return "MISSING"
    if delta > 0:
        return "UP"
    if delta < 0:
        return "DOWN"
    return "FLAT"


def _same_source_start_ambiguous(
    observations: Sequence[MarketObservation],
) -> bool:
    starts = [item.source_interval.start for item in observations]
    return len(starts) != len(set(starts))


def _unique_edge_price(
    observations: Sequence[MarketObservation],
    *,
    edge: Literal["first", "last"],
) -> Decimal | None:
    if not observations:
        return None
    target = (
        min(item.source_interval.start for item in observations)
        if edge == "first"
        else max(item.source_interval.start for item in observations)
    )
    edge_items = [item for item in observations if item.source_interval.start == target]
    if len(edge_items) != 1:
        return None
    return edge_items[0].price


def _max_excursion(
    observations: Sequence[MarketObservation],
    *,
    pre_price: Decimal | None,
) -> Decimal | None:
    if pre_price is None:
        return None
    deltas = [
        observation.price - pre_price
        for observation in observations
        if observation.price is not None
    ]
    if not deltas:
        return None
    magnitude = max(abs(delta) for delta in deltas)
    candidates = {delta for delta in deltas if abs(delta) == magnitude}
    return next(iter(candidates)) if len(candidates) == 1 else None


def _strict_relation(
    event_interval: SourceInterval,
    observation: MarketObservation,
) -> str:
    relation = associate_intervals(
        event_interval,
        observation.source_interval,
    )
    if relation.overlap:
        return "overlap"
    assert relation.order_relation is not None
    return relation.order_relation


@dataclass(frozen=True, slots=True)
class DirectionalAnomalyAuditV1:
    status: Literal[
        "BLOCKED",
        "NOT_EVALUATED_NO_DIAGNOSTIC_FAIR_DELTA",
    ]
    reason_codes: tuple[str, ...]
    audit_order: tuple[str, ...] = _DIRECTIONAL_AUDIT_ORDER


@dataclass(frozen=True, slots=True)
class AssociationResultV1:
    episode_id: str
    contract_id: str
    venue: str
    outcome: str
    delay_scenario_seconds: int
    horizon_seconds: int
    status: str
    order_ambiguous: bool
    contaminated: bool
    candidate: AssociationCandidateV1
    directional_audit: DirectionalAnomalyAuditV1


def _metrics_pairs(metrics: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(metrics.items()))


def _directional_audit(
    *,
    episode: FinalizedEpisodeV1,
    contract: MarketContractV1,
    outcome: str,
    order_ambiguous: bool,
    contaminated: bool,
    has_pre: bool,
    has_post: bool,
) -> DirectionalAnomalyAuditV1:
    reasons: list[str] = []
    if outcome not in contract.outcomes:
        reasons.append("orientation_unresolved")
    if episode.review_result == "unknown" or episode.penalty_status == "unknown":
        reasons.append("adjudication_unresolved")
    if episode.post_state is None:
        reasons.append("score_or_possession_unresolved")
    else:
        scores = (
            (
                episode.pre_state.home_score,
                episode.post_state.home_score,
            ),
            (
                episode.pre_state.away_score,
                episode.post_state.away_score,
            ),
        )
        if any(
            before is not None and after is not None and after < before
            for before, after in scores
        ):
            reasons.append("score_or_possession_unresolved")
    if order_ambiguous:
        reasons.append("same_second_order_ambiguous")
    if contaminated:
        reasons.append("next_salient_episode_contamination")
    if not has_pre or not has_post:
        reasons.append("stale_or_wrong_market_family")
    if reasons:
        return DirectionalAnomalyAuditV1(
            status="BLOCKED",
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
    return DirectionalAnomalyAuditV1(
        status="NOT_EVALUATED_NO_DIAGNOSTIC_FAIR_DELTA",
        reason_codes=("diagnostic_fair_delta_unavailable",),
    )


def _minute_path_metrics(
    *,
    post: Sequence[MarketObservation],
    pre_price: Decimal | None,
) -> tuple[Decimal | None, Decimal | None, bool, bool, bool]:
    path_ambiguous = _same_source_start_ambiguous(post)
    last = _unique_edge_price(post, edge="last")
    net = None if pre_price is None or last is None else last - pre_price
    excursion = _max_excursion(post, pre_price=pre_price)
    first = _unique_edge_price(post, edge="first")
    first_delta = None if pre_price is None or first is None else first - pre_price
    overshoot = bool(
        not path_ambiguous
        and excursion is not None
        and net is not None
        and abs(excursion) > abs(net)
    )
    reversal = bool(
        not path_ambiguous
        and first_delta is not None
        and net is not None
        and first_delta * net < 0
    )
    return net, excursion, overshoot, reversal, path_ambiguous


@dataclass(frozen=True, slots=True)
class _ObservationSeriesV1:
    items: tuple[MarketObservation, ...]
    starts: tuple[datetime, ...]
    ends: tuple[datetime, ...]


@dataclass(frozen=True, slots=True)
class _OutcomeObservationIndexV1:
    actual: _ObservationSeriesV1
    derived: _ObservationSeriesV1
    candles: _ObservationSeriesV1


def _observation_series(
    observations: Sequence[MarketObservation],
) -> _ObservationSeriesV1:
    items = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.source_interval.start,
                item.observation_id,
            ),
        )
    )
    return _ObservationSeriesV1(
        items=items,
        starts=tuple(item.source_interval.start for item in items),
        ends=tuple(item.source_interval.end for item in items),
    )


def _empty_observation_index() -> _OutcomeObservationIndexV1:
    empty = _ObservationSeriesV1((), (), ())
    return _OutcomeObservationIndexV1(empty, empty, empty)


def _build_contract_outcome_indexes(
    contracts: Sequence[MarketContractV1],
    observations_by_contract: Mapping[str, Sequence[MarketObservation]],
) -> dict[tuple[str, str], _OutcomeObservationIndexV1]:
    indexes: dict[tuple[str, str], _OutcomeObservationIndexV1] = {}
    for contract in contracts:
        buckets: dict[str, tuple[list[MarketObservation], ...]] = {
            outcome: ([], [], []) for outcome in contract.outcomes
        }
        for observation in observations_by_contract.get(contract.contract_id, ()):
            bucket = buckets.get(observation.outcome)
            if bucket is None:
                continue
            if (
                observation.kind == "trade"
                and observation.provenance == "observed"
                and observation.price is not None
                and observation.primary_path_eligible
            ):
                bucket[0].append(observation)
            elif (
                observation.kind == "trade"
                and observation.provenance == "derived_complement"
            ):
                bucket[1].append(observation)
            elif observation.kind == "kalshi_1m_candle_bbo":
                bucket[2].append(observation)
        for outcome, (actual, derived, candles) in buckets.items():
            indexes[(contract.contract_id, outcome)] = _OutcomeObservationIndexV1(
                actual=_observation_series(actual),
                derived=_observation_series(derived),
                candles=_observation_series(candles),
            )
    return indexes


def _latest_pre_observations(
    series: _ObservationSeriesV1,
    event_interval: SourceInterval,
) -> tuple[MarketObservation, ...]:
    # Audited trades are fixed one-second half-open intervals, so both their
    # starts and ends are monotonic in the same pre-built ordering.
    stop = bisect_right(series.ends, event_interval.start)
    if stop == 0:
        return ()
    latest_start = series.starts[stop - 1]
    start = bisect_left(series.starts, latest_start, 0, stop)
    return series.items[start:stop]


def _overlapping_actual_observations(
    series: _ObservationSeriesV1,
    event_interval: SourceInterval,
) -> tuple[MarketObservation, ...]:
    # A one-second half-open trade can overlap G only when its start lies in
    # (G.start-1s, G.end].  Keep the exact interval predicate as a fail-closed
    # check over this bounded slice.
    start = bisect_right(
        series.starts,
        event_interval.start - timedelta(seconds=1),
    )
    stop = bisect_right(series.starts, event_interval.end)
    return tuple(
        item
        for item in series.items[start:stop]
        if _strict_relation(event_interval, item) == "overlap"
    )


def _post_observations(
    series: _ObservationSeriesV1,
    event_interval: SourceInterval,
    deadline: datetime,
) -> tuple[MarketObservation, ...]:
    start = bisect_right(series.starts, event_interval.end)
    stop = bisect_right(series.starts, deadline)
    return series.items[start:stop]


def _derived_observation_count(
    series: _ObservationSeriesV1,
    event_interval: SourceInterval,
    deadline: datetime,
) -> int:
    start = bisect_right(
        series.starts,
        event_interval.start - timedelta(seconds=1),
    )
    stop = bisect_right(series.starts, deadline)
    return stop - start


def _candle_overlap_count(
    series: _ObservationSeriesV1,
    event_interval: SourceInterval,
    deadline: datetime,
) -> int:
    # Audited Kalshi candles are fixed one-minute half-open intervals.
    start = bisect_right(
        series.starts,
        event_interval.start - timedelta(minutes=1),
    )
    stop = bisect_right(series.starts, deadline)
    return stop - start


def _build_association(
    *,
    episode: FinalizedEpisodeV1,
    event_interval: SourceInterval,
    contract: MarketContractV1,
    outcome: str,
    delay: int,
    horizon: int,
    observation_index: _OutcomeObservationIndexV1,
    next_salient_at: datetime | None,
) -> AssociationResultV1:
    latest_pre = _latest_pre_observations(
        observation_index.actual,
        event_interval,
    )
    pre_price = latest_pre[0].price if len(latest_pre) == 1 else None
    pre_ambiguous = len(latest_pre) > 1
    pre_age = (
        None
        if len(latest_pre) != 1
        else _seconds(event_interval.start - latest_pre[0].source_interval.end)
    )

    deadline = event_interval.end + timedelta(seconds=horizon)
    overlap = _overlapping_actual_observations(
        observation_index.actual,
        event_interval,
    )
    post = _post_observations(
        observation_index.actual,
        event_interval,
        deadline,
    )
    same_second = _same_source_start_ambiguous(post)
    first_post = _unique_edge_price(post, edge="first")
    last = _unique_edge_price(post, edge="last")
    volume = sum((item.size for item in post), _ZERO)
    vwap = (
        None
        if not post or volume == 0
        else sum(
            (item.price * item.size for item in post if item.price is not None),
            _ZERO,
        )
        / volume
    )
    signed_delta = None if pre_price is None or last is None else last - pre_price
    excursion = _max_excursion(post, pre_price=pre_price)
    last_start = None if not post else max(item.source_interval.start for item in post)
    staleness = None if last_start is None else _seconds(deadline - last_start)
    order_ambiguous = bool(overlap or pre_ambiguous or same_second)
    contaminated = next_salient_at is not None and next_salient_at <= deadline
    if contaminated:
        status = "CONTAMINATED"
    elif order_ambiguous:
        status = "ORDER_AMBIGUOUS"
    elif pre_price is None or not post:
        status = "MISSING_OBSERVATION"
    else:
        status = "OBSERVED"

    bbo_overlap_count = _candle_overlap_count(
        observation_index.candles,
        event_interval,
        deadline,
    )
    derived_count = _derived_observation_count(
        observation_index.derived,
        event_interval,
        deadline,
    )
    if horizon == 60:
        minute_net, minute_excursion, overshoot, reversal, minute_ambiguous = (
            _minute_path_metrics(
                post=post,
                pre_price=pre_price,
            )
        )
    else:
        (
            minute_net,
            minute_excursion,
            overshoot,
            reversal,
            minute_ambiguous,
        ) = (
            None,
            None,
            False,
            False,
            False,
        )
    metrics: dict[str, object] = {
        "bbo_interval_overlap_count": bbo_overlap_count,
        "candle_bbo_point_metrics_prohibited": True,
        "derived_observation_count": derived_count,
        "diagnostic_fair_delta_status": "PIT_UNPROVEN_NOT_SUPPLIED",
        "directional_audit_order": _DIRECTIONAL_AUDIT_ORDER,
        "executable_price_claim_permitted": False,
        "first_post_event_trade": first_post,
        "last_actual_price": last,
        "maximum_excursion": excursion,
        "net_60_second_change": minute_net,
        "net_60_second_maximum_excursion": minute_excursion,
        "overshoot_candidate": overshoot,
        "path_order_ambiguous": minute_ambiguous,
        "pre_event_actual_trade": pre_price,
        "pre_trade_age_seconds": pre_age,
        "real_latency_claim_permitted": False,
        "reversal_candidate": reversal,
        "signed_price_change": signed_delta,
        "staleness_seconds": staleness,
        "trade_count": len(post),
        "two_venue_direction_consistency": "PENDING",
        "venue_direction": _direction(signed_delta),
        "volume": volume,
        "vwap": vwap,
    }
    candidate = AssociationCandidateV1(
        episode_id=episode.episode_id,
        contract_id=contract.contract_id,
        delay_scenario_seconds=delay,
        horizon_seconds=horizon,
        status=status,
        order_ambiguous=order_ambiguous,
        contaminated=contaminated,
        metrics=_metrics_pairs(metrics),  # type: ignore[arg-type]
    )
    directional = _directional_audit(
        episode=episode,
        contract=contract,
        outcome=outcome,
        order_ambiguous=order_ambiguous,
        contaminated=contaminated,
        has_pre=pre_price is not None,
        has_post=bool(post),
    )
    return AssociationResultV1(
        episode_id=episode.episode_id,
        contract_id=contract.contract_id,
        venue=contract.venue,
        outcome=outcome,
        delay_scenario_seconds=delay,
        horizon_seconds=horizon,
        status=status,
        order_ambiguous=order_ambiguous,
        contaminated=contaminated,
        candidate=candidate,
        directional_audit=directional,
    )


def _semantic_orientation_key(
    contract: MarketContractV1,
    outcome: str,
) -> tuple[object, ...]:
    lowered = outcome.casefold()
    orientation = (
        contract.subject
        if lowered in {"yes", "over"}
        else f"NOT:{contract.subject}"
        if lowered in {"no", "under"}
        else outcome
    )
    return (
        contract.family,
        contract.period,
        contract.measure,
        contract.comparator,
        contract.line,
        orientation,
    )


def _attach_two_venue_consistency(
    rows: Sequence[AssociationResultV1],
    contract_by_id: Mapping[str, MarketContractV1],
) -> tuple[AssociationResultV1, ...]:
    grouped: dict[tuple[object, ...], list[AssociationResultV1]] = defaultdict(list)
    for row in rows:
        contract = contract_by_id[row.contract_id]
        grouped[
            (
                row.episode_id,
                row.delay_scenario_seconds,
                row.horizon_seconds,
                *_semantic_orientation_key(contract, row.outcome),
            )
        ].append(row)
    result: list[AssociationResultV1] = []
    for group in grouped.values():
        venue_directions: dict[str, set[str]] = defaultdict(set)
        for row in group:
            venue_directions[row.venue].add(
                str(dict(row.candidate.metrics)["venue_direction"])
            )
        usable = {
            venue: next(iter(directions))
            for venue, directions in venue_directions.items()
            if len(directions) == 1 and next(iter(directions)) != "MISSING"
        }
        if len(venue_directions) < 2:
            consistency = "SINGLE_VENUE"
        elif len(usable) < 2:
            consistency = "INSUFFICIENT"
        elif len(set(usable.values())) == 1:
            consistency = "CONSISTENT"
        else:
            consistency = "DIVERGENT"
        for row in group:
            metrics = dict(row.candidate.metrics)
            metrics["two_venue_direction_consistency"] = consistency
            candidate = replace(
                row.candidate,
                metrics=_metrics_pairs(metrics),  # type: ignore[arg-type]
            )
            result.append(replace(row, candidate=candidate))
    return tuple(
        sorted(
            result,
            key=lambda row: (
                row.episode_id,
                row.contract_id,
                row.outcome,
                row.delay_scenario_seconds,
                row.horizon_seconds,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class X13AssociationRunV1:
    status: Literal["PRELIMINARY_SOURCE_TIME_ONLY"]
    game_id: str
    layer_g_audit: LayerGAuditV1
    layer_m_audit: LayerMAuditV1
    timelines: tuple[SourceTimelineV1, ...]
    associations: tuple[AssociationResultV1, ...]
    claim_scope: str = "association_and_signal_candidates_only"
    prohibited_claims: tuple[str, ...] = _PROHIBITED_CLAIMS


@dataclass(frozen=True, slots=True)
class X13AssociationStreamV1:
    """Audited per-game context whose association rows remain lazy."""

    status: Literal["PRELIMINARY_SOURCE_TIME_ONLY"]
    game_id: str
    layer_g_audit: LayerGAuditV1
    layer_m_audit: LayerMAuditV1
    timelines: tuple[SourceTimelineV1, ...]
    expected_association_count: int
    _ledger: X13GameLedgerV1 = dataclass_field(repr=False, compare=False)
    _eligible_contracts: tuple[MarketContractV1, ...] = dataclass_field(
        repr=False,
        compare=False,
    )
    _observations: tuple[MarketObservation, ...] = dataclass_field(
        repr=False,
        compare=False,
    )
    associations_materialized: Literal[False] = False


def prepare_x13_association_stream(
    ledger: X13GameLedgerV1,
    contracts: Sequence[MarketContractV1],
    observations: Sequence[MarketObservation],
) -> X13AssociationStreamV1:
    """Audit a game and prepare a bounded lazy Layer A stream."""

    layer_g = audit_layer_g(ledger)
    if layer_g.status != "PASS" or layer_g.game_id is None:
        raise AssociationEvidenceError(
            "Layer G audit failed: " + ",".join(layer_g.errors)
        )
    layer_m = audit_layer_m(
        contracts,
        observations,
        game_id=layer_g.game_id,
    )
    if layer_m.status != "PASS":
        raise AssociationEvidenceError(
            "Layer M audit failed: " + ",".join(layer_m.errors)
        )
    deduplicated = deduplicate_cleaned_observations_v1(observations)
    timelines = tuple(
        _timeline(
            delay=delay,
            layer_g_intervals=layer_g.episode_intervals,
            observations=deduplicated,
        )
        for delay in X13_DELAY_SCENARIOS_SECONDS
    )
    eligible_contracts = tuple(
        sorted(
            (
                contract
                for contract in contracts
                if contract.analysis_eligible
                and contract.dependency_game_ids == (layer_g.game_id,)
            ),
            key=lambda contract: contract.contract_id,
        )
    )
    expected_count = (
        len(layer_g.episode_intervals)
        * len(X13_HORIZONS_SECONDS)
        * sum(len(contract.outcomes) for contract in eligible_contracts)
    )
    return X13AssociationStreamV1(
        status=X13_STATUS,
        game_id=layer_g.game_id,
        layer_g_audit=layer_g,
        layer_m_audit=layer_m,
        timelines=timelines,
        expected_association_count=expected_count,
        _ledger=ledger,
        _eligible_contracts=eligible_contracts,
        _observations=deduplicated,
    )


def _contract_outcome_groups(
    contracts: Sequence[MarketContractV1],
) -> tuple[tuple[tuple[MarketContractV1, str], ...], ...]:
    grouped: dict[
        tuple[object, ...],
        list[tuple[MarketContractV1, str]],
    ] = defaultdict(list)
    for contract in contracts:
        for outcome in contract.outcomes:
            grouped[_semantic_orientation_key(contract, outcome)].append(
                (contract, outcome)
            )
    return tuple(
        tuple(
            sorted(
                grouped[key],
                key=lambda item: (item[0].contract_id, item[1]),
            )
        )
        for key in sorted(grouped, key=canonical_sha256)
    )


def iter_x13_associations(
    stream: X13AssociationStreamV1,
) -> Iterator[AssociationResultV1]:
    """Yield complete association rows without batch materialization."""

    if not isinstance(stream, X13AssociationStreamV1):
        raise TypeError("stream must be X13AssociationStreamV1")
    ledger = stream._ledger
    eligible_contracts = stream._eligible_contracts
    observations = stream._observations
    layer_g = stream.layer_g_audit
    layer_m = stream.layer_m_audit
    episode_by_id = {episode.episode_id: episode for episode in ledger.episodes}
    event_times = {
        item.episode_id: item.source_interval.start
        for item in layer_g.episode_intervals
        if item.delay_scenario_seconds == 0
    }
    next_salient: dict[str, datetime | None] = {}
    ordered_episode_times = sorted(
        event_times.items(),
        key=lambda item: (item[1], item[0]),
    )
    for index, (episode_id, _source_time) in enumerate(ordered_episode_times):
        next_salient[episode_id] = (
            None
            if index + 1 == len(ordered_episode_times)
            else ordered_episode_times[index + 1][1]
        )
    binding_map = dict(layer_m.observation_contract_bindings)
    by_contract: dict[str, list[MarketObservation]] = defaultdict(list)
    for observation in observations:
        contract_id = binding_map.get(observation.observation_id)
        if contract_id is not None:
            by_contract[contract_id].append(observation)
    observation_indexes = _build_contract_outcome_indexes(
        eligible_contracts,
        by_contract,
    )
    empty_observation_index = _empty_observation_index()
    contract_by_id = {contract.contract_id: contract for contract in eligible_contracts}
    outcome_groups = _contract_outcome_groups(eligible_contracts)
    yielded = 0
    for event in layer_g.episode_intervals:
        episode = episode_by_id[event.episode_id]
        for horizon in X13_HORIZONS_SECONDS:
            for group in outcome_groups:
                local_rows = tuple(
                    _build_association(
                        episode=episode,
                        event_interval=event.source_interval,
                        contract=contract,
                        outcome=outcome,
                        delay=event.delay_scenario_seconds,
                        horizon=horizon,
                        observation_index=observation_indexes.get(
                            (contract.contract_id, outcome),
                            empty_observation_index,
                        ),
                        next_salient_at=next_salient[event.episode_id],
                    )
                    for contract, outcome in group
                )
                attached = _attach_two_venue_consistency(
                    local_rows,
                    contract_by_id,
                )
                for row in attached:
                    yielded += 1
                    yield row
    if yielded != stream.expected_association_count:
        raise AssociationEvidenceError(
            "streamed association cardinality differs from frozen plan"
        )


def run_x13_association(
    ledger: X13GameLedgerV1,
    contracts: Sequence[MarketContractV1],
    observations: Sequence[MarketObservation],
) -> X13AssociationRunV1:
    """Materialize the lazy stream for bounded fixtures and small games."""

    stream = prepare_x13_association_stream(
        ledger,
        contracts,
        observations,
    )
    return X13AssociationRunV1(
        status=stream.status,
        game_id=stream.game_id,
        layer_g_audit=stream.layer_g_audit,
        layer_m_audit=stream.layer_m_audit,
        timelines=stream.timelines,
        associations=tuple(iter_x13_associations(stream)),
    )


@dataclass(frozen=True, slots=True)
class AssociationAttritionRowV1:
    association_id: str
    game_id: str
    episode_id: str
    contract_id: str
    logical_market_id: str
    venue: str
    family: str
    outcome: str
    delay_seconds: int
    horizon_seconds: int
    status: str
    eligible: bool
    exclusion_reasons: tuple[
        Literal[
            "missing",
            "contaminated",
            "ambiguous",
            "stale",
            "wrong_family",
            "derived_only",
            "invalid_orientation",
            "no_actual_trade",
        ],
        ...,
    ]

    def to_dict(self) -> dict[str, object]:
        return {
            "association_id": self.association_id,
            "game_id": self.game_id,
            "episode_id": self.episode_id,
            "contract_id": self.contract_id,
            "logical_market_id": self.logical_market_id,
            "venue": self.venue,
            "family": self.family,
            "outcome": self.outcome,
            "delay_seconds": self.delay_seconds,
            "horizon_seconds": self.horizon_seconds,
            "status": self.status,
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True, slots=True)
class AssociationAttritionV1:
    game_id: str
    rows: tuple[AssociationAttritionRowV1, ...]
    schema_version: str = "association_attrition_v1"

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict[str, object]:
        counts = Counter(
            reason for row in self.rows for reason in row.exclusion_reasons
        )
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "row_count": len(self.rows),
            "eligible_row_count": sum(row.eligible for row in self.rows),
            "reason_counts": dict(sorted(counts.items())),
            "rows": self.to_records(),
        }


def _metric_decimal(
    metrics: Mapping[str, object],
    field: str,
) -> Decimal | None:
    value = metrics.get(field)
    if value is None:
        return None
    if type(value) is not Decimal or not value.is_finite():
        raise AssociationEvidenceError(
            f"association metric {field} is not a finite Decimal"
        )
    return value


def build_association_attrition_rows_v1(
    associations: Sequence[AssociationResultV1],
    contracts: Sequence[MarketContractV1],
    *,
    game_id: str,
    allowed_families: Sequence[str],
    max_staleness_seconds: Decimal = Decimal(60),
) -> AssociationAttritionV1:
    """Classify existing association results without rescanning market history."""

    if not isinstance(associations, Sequence) or isinstance(
        associations, (str, bytes)
    ):
        raise TypeError("associations must be a sequence")
    if any(not isinstance(row, AssociationResultV1) for row in associations):
        raise TypeError("associations must contain AssociationResultV1")
    if not isinstance(contracts, Sequence) or isinstance(contracts, (str, bytes)):
        raise TypeError("contracts must be a sequence")
    if any(not isinstance(contract, MarketContractV1) for contract in contracts):
        raise TypeError("contracts must contain MarketContractV1")
    if type(game_id) is not str or not game_id:
        raise TypeError("game_id must be nonempty text")
    if (
        not isinstance(allowed_families, Sequence)
        or isinstance(allowed_families, (str, bytes))
        or not allowed_families
        or any(type(family) is not str or not family for family in allowed_families)
        or len(set(allowed_families)) != len(allowed_families)
    ):
        raise TypeError("allowed_families must be unique nonempty text")
    if (
        type(max_staleness_seconds) is not Decimal
        or not max_staleness_seconds.is_finite()
        or max_staleness_seconds < 0
    ):
        raise TypeError("max_staleness_seconds must be a nonnegative Decimal")

    contract_by_id = {contract.contract_id: contract for contract in contracts}
    if len(contract_by_id) != len(contracts):
        raise AssociationEvidenceError("duplicate contract_id")
    allowed = set(allowed_families)
    result: list[AssociationAttritionRowV1] = []
    for association in associations:
        contract = contract_by_id.get(association.contract_id)
        if contract is None:
            raise AssociationEvidenceError(
                "association contract_id is not in the supplied contract map"
            )
        metrics = dict(association.candidate.metrics)
        trade_count = metrics.get("trade_count")
        if type(trade_count) is not int or trade_count < 0:
            raise AssociationEvidenceError(
                "association trade_count is not a nonnegative integer"
            )
        derived_count = metrics.get("derived_observation_count")
        if type(derived_count) is not int or derived_count < 0:
            raise AssociationEvidenceError(
                "association derived_observation_count is invalid"
            )
        pre_trade = metrics.get("pre_event_actual_trade")
        first_trade = metrics.get("first_post_event_trade")
        pre_age = _metric_decimal(metrics, "pre_trade_age_seconds")
        post_staleness = _metric_decimal(metrics, "staleness_seconds")
        reasons: list[str] = []
        if (
            association.status == "MISSING_OBSERVATION"
            or pre_trade is None
            or first_trade is None
        ):
            reasons.append("missing")
        if association.contaminated:
            reasons.append("contaminated")
        if association.order_ambiguous:
            reasons.append("ambiguous")
        if (
            pre_age is not None and pre_age > max_staleness_seconds
        ) or (
            post_staleness is not None
            and post_staleness > max_staleness_seconds
        ):
            reasons.append("stale")
        if contract.family not in allowed:
            reasons.append("wrong_family")
        if trade_count == 0 and derived_count > 0:
            reasons.append("derived_only")
        if association.outcome not in contract.outcomes:
            reasons.append("invalid_orientation")
        if trade_count == 0:
            reasons.append("no_actual_trade")
        typed_reasons = tuple(reasons)
        association_id = canonical_sha256(
            {
                "contract_id": association.contract_id,
                "delay_seconds": association.delay_scenario_seconds,
                "episode_id": association.episode_id,
                "game_id": game_id,
                "horizon_seconds": association.horizon_seconds,
                "outcome": association.outcome,
                "schema_version": "association_attrition_v1",
            }
        )
        result.append(
            AssociationAttritionRowV1(
                association_id=association_id,
                game_id=game_id,
                episode_id=association.episode_id,
                contract_id=association.contract_id,
                logical_market_id=contract.logical_market_id,
                venue=association.venue,
                family=contract.family,
                outcome=association.outcome,
                delay_seconds=association.delay_scenario_seconds,
                horizon_seconds=association.horizon_seconds,
                status=association.status,
                eligible=association.status == "OBSERVED" and not typed_reasons,
                exclusion_reasons=typed_reasons,  # type: ignore[arg-type]
            )
        )
    ids = [row.association_id for row in result]
    if len(ids) != len(set(ids)):
        raise AssociationEvidenceError("duplicate association attrition grain")
    return AssociationAttritionV1(game_id=game_id, rows=tuple(result))


@dataclass(frozen=True, slots=True)
class ExplorationObservationV1:
    observation_id: str
    hypothesis_id: str
    group: str
    game_id: str
    episode_id: str
    venue: Literal["polymarket", "kalshi"]
    effect: Decimal | None
    eligible: bool

    def __post_init__(self) -> None:
        for field in (
            "observation_id",
            "hypothesis_id",
            "group",
            "game_id",
            "episode_id",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise TypeError(f"{field} must be nonempty text")
        if self.venue not in {"polymarket", "kalshi"}:
            raise ValueError("unknown venue")
        if self.effect is not None and (
            type(self.effect) is not Decimal or not self.effect.is_finite()
        ):
            raise TypeError("effect must be a finite Decimal or None")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be bool")


@dataclass(frozen=True, slots=True)
class InferenceInputV1:
    eligible_observation_ids: tuple[str, ...]
    observation_game_clusters: tuple[tuple[str, str], ...]
    cluster_game_ids: tuple[str, ...]
    logo_holdout_game_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    bootstrap_seed: int
    game_cluster_bootstrap: Literal[True] = True
    leave_one_game_out: Literal[True] = True
    multiple_testing_method: Literal["benjamini_hochberg"] = "benjamini_hochberg"
    confidence_interval_label: str = "game_cluster_bootstrap_CI"
    stability_label: str = "leave_one_game_out_stability"


@dataclass(frozen=True, slots=True)
class ExplorationGroupGateV1:
    group: str
    eligible_episode_count: int
    game_count: int
    status: Literal[
        "PRELIMINARY_ELIGIBLE",
        "DESCRIPTIVE_ONLY",
        "CASE_ONLY",
    ]


@dataclass(frozen=True, slots=True)
class ExplorationGateReportV1:
    analysis_kind: str
    status: Literal[
        "PRELIMINARY_ELIGIBLE",
        "DESCRIPTIVE_ONLY",
        "CASE_ONLY",
    ]
    eligible_episode_count: int
    game_count: int
    venue_episode_counts: tuple[tuple[str, int], ...]
    group_gates: tuple[ExplorationGroupGateV1, ...]
    reason_codes: tuple[str, ...]
    inference: InferenceInputV1
    claim_scope: str = "preliminary_association_only"
    prohibited_claims: tuple[str, ...] = _PROHIBITED_CLAIMS


def _event_keys(
    observations: Sequence[ExplorationObservationV1],
) -> set[tuple[str, str]]:
    return {(row.game_id, row.episode_id) for row in observations}


def _support_status(
    *,
    episode_count: int,
    game_count: int,
    required_episodes: int,
    required_games: int,
) -> str:
    if episode_count >= required_episodes and game_count >= required_games:
        return "PRELIMINARY_ELIGIBLE"
    if episode_count >= 5 and game_count >= 3:
        return "DESCRIPTIVE_ONLY"
    return "CASE_ONLY"


def evaluate_exploration_gate(
    observations: Sequence[ExplorationObservationV1],
    *,
    analysis_kind: str,
    bootstrap_seed: int = 20260722,
) -> ExplorationGateReportV1:
    """Apply the frozen report support gates without fitting or searching."""

    if analysis_kind not in _ANALYSIS_KINDS:
        raise AssociationEvidenceError("analysis_kind is not registered")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise TypeError("bootstrap_seed must be a nonnegative integer")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a sequence")
    if any(not isinstance(row, ExplorationObservationV1) for row in observations):
        raise TypeError("observations must contain ExplorationObservationV1")
    ids = [row.observation_id for row in observations]
    if len(ids) != len(set(ids)):
        raise AssociationEvidenceError("duplicate exploration observation_id")
    eligible = tuple(
        row for row in observations if row.eligible and row.effect is not None
    )
    event_keys = _event_keys(eligible)
    games = tuple(sorted({game_id for game_id, _ in event_keys}))
    venue_counts = tuple(
        sorted(
            (
                venue,
                len(
                    {
                        (row.game_id, row.episode_id)
                        for row in eligible
                        if row.venue == venue
                    }
                ),
            )
            for venue in ("kalshi", "polymarket")
        )
    )
    grouped: dict[str, tuple[ExplorationObservationV1, ...]] = {}
    for group in sorted({row.group for row in eligible}):
        grouped[group] = tuple(row for row in eligible if row.group == group)

    if analysis_kind == "primary_pooled":
        requirements = {group: (300, 16) for group in grouped}
    elif analysis_kind == "binary_moderator":
        if len(grouped) != 2:
            raise AssociationEvidenceError(
                "binary_moderator requires exactly two registered sides"
            )
        requirements = {group: (40, 8) for group in grouped}
    elif analysis_kind == "named_subtype":
        requirements = {group: (20, 6) for group in grouped}
    else:
        requirements = {group: (15, 6) for group in grouped}

    group_gates: list[ExplorationGroupGateV1] = []
    for group, rows in grouped.items():
        keys = _event_keys(rows)
        group_games = {game_id for game_id, _ in keys}
        required_episodes, required_games = requirements[group]
        group_gates.append(
            ExplorationGroupGateV1(
                group=group,
                eligible_episode_count=len(keys),
                game_count=len(group_games),
                status=_support_status(
                    episode_count=len(keys),
                    game_count=len(group_games),
                    required_episodes=required_episodes,
                    required_games=required_games,
                ),  # type: ignore[arg-type]
            )
        )
    if not group_gates:
        status = "CASE_ONLY"
    elif all(group.status == "PRELIMINARY_ELIGIBLE" for group in group_gates):
        status = "PRELIMINARY_ELIGIBLE"
    elif all(group.status != "CASE_ONLY" for group in group_gates):
        status = "DESCRIPTIVE_ONLY"
    else:
        status = "CASE_ONLY"

    reasons: list[str] = []
    if analysis_kind == "primary_pooled":
        counts = dict(venue_counts)
        if any(counts[venue] < 100 for venue in ("kalshi", "polymarket")):
            status = (
                "DESCRIPTIVE_ONLY"
                if len(event_keys) >= 5 and len(games) >= 3
                else "CASE_ONLY"
            )
            reasons.append("per_venue_support_below_100")
    game_contributions = Counter(game_id for game_id, _ in event_keys)
    concentrated = bool(
        event_keys and max(game_contributions.values()) * 4 > len(event_keys)
    )
    for rows in grouped.values():
        group_keys = _event_keys(rows)
        group_contributions = Counter(game_id for game_id, _ in group_keys)
        if group_keys and max(group_contributions.values()) * 4 > len(group_keys):
            concentrated = True
    if concentrated:
        if status == "PRELIMINARY_ELIGIBLE":
            status = "DESCRIPTIVE_ONLY"
        reasons.append("single_game_contribution_exceeds_25_percent")
    if status != "PRELIMINARY_ELIGIBLE" and not reasons:
        reasons.append("registered_support_gate_not_met")

    inference = InferenceInputV1(
        eligible_observation_ids=tuple(sorted(row.observation_id for row in eligible)),
        observation_game_clusters=tuple(
            sorted((row.observation_id, row.game_id) for row in eligible)
        ),
        cluster_game_ids=games,
        logo_holdout_game_ids=games,
        hypothesis_ids=tuple(sorted({row.hypothesis_id for row in eligible})),
        bootstrap_seed=bootstrap_seed,
    )
    return ExplorationGateReportV1(
        analysis_kind=analysis_kind,
        status=status,  # type: ignore[arg-type]
        eligible_episode_count=len(event_keys),
        game_count=len(games),
        venue_episode_counts=venue_counts,
        group_gates=tuple(group_gates),
        reason_codes=tuple(reasons),
        inference=inference,
    )


@dataclass(frozen=True, slots=True)
class AggregatedEpisodeEffectV1:
    game_id: str
    episode_id: str
    hypothesis_id: str
    group: str
    effect: Decimal
    venues: tuple[str, ...]
    source_observation_ids: tuple[str, ...]

    @property
    def venue_count(self) -> int:
        return len(self.venues)


@dataclass(frozen=True, slots=True)
class LogoEffectV1:
    held_out_game_id: str
    effect: Decimal | None
    sign: Literal["POSITIVE", "NEGATIVE", "ZERO", "MISSING"]


@dataclass(frozen=True, slots=True)
class InferenceEstimateV1:
    estimate_id: str
    hypothesis_id: str
    hypothesis_family: str
    estimand: str
    point_estimate: Decimal | None
    ci_lower: Decimal | None
    ci_upper: Decimal | None
    bootstrap_requested_samples: int
    bootstrap_valid_samples: int
    empirical_two_sided_p: Decimal | None
    bh_q_value: Decimal | None
    logo_effects: tuple[LogoEffectV1, ...]
    logo_sign_stability: Decimal | None
    episode_count: int
    game_count: int
    support_status: str
    preliminary_inference_claim_permitted: bool
    bootstrap_distribution_sha256: str
    cluster_multiplicity_preserved: Literal[True] = True
    promotion_claim_permitted: Literal[False] = False
    interval_semantics: str = "game_cluster_percentile_2.5_97.5"
    p_value_semantics: str = "smoothed_empirical_two_sided_zero_effect_tail"


@dataclass(frozen=True, slots=True)
class InferenceReportV1:
    experiment_id: Literal["X-13"]
    status: Literal["PRELIMINARY_SOURCE_TIME_ONLY"]
    lock_id: str
    gate_status: str
    bootstrap_seed: int
    bootstrap_requested_samples: int
    aggregated_episode_effects: tuple[AggregatedEpisodeEffectV1, ...]
    estimates: tuple[InferenceEstimateV1, ...]
    venue_aggregation_semantics: str = (
        "arithmetic_mean_within_(game_id,episode_id,hypothesis_id,group)"
    )
    cluster_resampling_semantics: str = (
        "sample_games_with_replacement_and_repeat_all_cluster_rows_by_draw_multiplicity"
    )
    multiple_testing_method: str = (
        "benjamini_hochberg_within_registered_hypothesis_family"
    )
    claim_scope: str = "preliminary_registered_association_inference_only"
    prohibited_claims: tuple[str, ...] = _PROHIBITED_CLAIMS


def _aggregate_episode_effects(
    observations: Sequence[ExplorationObservationV1],
) -> tuple[AggregatedEpisodeEffectV1, ...]:
    grouped: dict[
        tuple[str, str, str, str],
        list[ExplorationObservationV1],
    ] = defaultdict(list)
    for row in observations:
        if row.eligible and row.effect is not None:
            grouped[
                (
                    row.game_id,
                    row.episode_id,
                    row.hypothesis_id,
                    row.group,
                )
            ].append(row)
    aggregated: list[AggregatedEpisodeEffectV1] = []
    for (game_id, episode_id, hypothesis_id, group), rows in sorted(grouped.items()):
        venues = tuple(sorted(row.venue for row in rows))
        if len(venues) != len(set(venues)):
            raise AssociationEvidenceError(
                "duplicate venue inside registered episode unit"
            )
        effects = tuple(row.effect for row in rows if row.effect is not None)
        if not effects:
            continue
        aggregated.append(
            AggregatedEpisodeEffectV1(
                game_id=game_id,
                episode_id=episode_id,
                hypothesis_id=hypothesis_id,
                group=group,
                effect=sum(effects, _ZERO) / Decimal(len(effects)),
                venues=venues,
                source_observation_ids=tuple(
                    sorted(row.observation_id for row in rows)
                ),
            )
        )
    return tuple(aggregated)


def _mean(
    units: Sequence[AggregatedEpisodeEffectV1],
    multiplicities: Mapping[str, int],
) -> Decimal | None:
    weighted_count = sum(multiplicities.get(unit.game_id, 0) for unit in units)
    if weighted_count == 0:
        return None
    return sum(
        (unit.effect * multiplicities.get(unit.game_id, 0) for unit in units),
        _ZERO,
    ) / Decimal(weighted_count)


def _estimate_statistic(
    units: Sequence[AggregatedEpisodeEffectV1],
    hypothesis: RegisteredHypothesisV1,
    *,
    target_group: str | None,
    multiplicities: Mapping[str, int],
) -> Decimal | None:
    if hypothesis.estimand == "pooled_mean":
        return _mean(units, multiplicities)
    if hypothesis.estimand == "group_mean":
        return _mean(
            tuple(unit for unit in units if unit.group == target_group),
            multiplicities,
        )
    lower_group, upper_group = hypothesis.approved_groups
    lower = _mean(
        tuple(unit for unit in units if unit.group == lower_group),
        multiplicities,
    )
    upper = _mean(
        tuple(unit for unit in units if unit.group == upper_group),
        multiplicities,
    )
    return None if lower is None or upper is None else upper - lower


def _deterministic_cluster_index(
    *,
    seed: int,
    estimate_id: str,
    sample_index: int,
    draw_index: int,
    cluster_count: int,
) -> int:
    material = (f"{seed}|{estimate_id}|{sample_index}|{draw_index}").encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % cluster_count


def _bootstrap_distribution(
    units: Sequence[AggregatedEpisodeEffectV1],
    hypothesis: RegisteredHypothesisV1,
    *,
    target_group: str | None,
    estimate_id: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[Decimal, ...]:
    games = tuple(sorted({unit.game_id for unit in units}))
    if not games:
        return ()
    distribution: list[Decimal] = []
    for sample_index in range(bootstrap_samples):
        multiplicities: Counter[str] = Counter()
        for draw_index in range(len(games)):
            selected = games[
                _deterministic_cluster_index(
                    seed=seed,
                    estimate_id=estimate_id,
                    sample_index=sample_index,
                    draw_index=draw_index,
                    cluster_count=len(games),
                )
            ]
            multiplicities[selected] += 1
        statistic = _estimate_statistic(
            units,
            hypothesis,
            target_group=target_group,
            multiplicities=multiplicities,
        )
        if statistic is not None:
            distribution.append(statistic)
    return tuple(distribution)


def _nearest_rank(
    ordered: Sequence[Decimal],
    *,
    numerator: int,
    denominator: int,
) -> Decimal | None:
    if not ordered:
        return None
    rank = (len(ordered) * numerator + denominator - 1) // denominator
    return ordered[max(1, rank) - 1]


def _empirical_two_sided_p(
    distribution: Sequence[Decimal],
) -> Decimal | None:
    if not distribution:
        return None
    nonpositive = sum(value <= 0 for value in distribution)
    nonnegative = sum(value >= 0 for value in distribution)
    denominator = Decimal(len(distribution) + 1)
    lower = Decimal(nonpositive + 1) / denominator
    upper = Decimal(nonnegative + 1) / denominator
    return min(Decimal(1), Decimal(2) * min(lower, upper))


def _sign(value: Decimal | None) -> str:
    if value is None:
        return "MISSING"
    if value > 0:
        return "POSITIVE"
    if value < 0:
        return "NEGATIVE"
    return "ZERO"


def _logo(
    units: Sequence[AggregatedEpisodeEffectV1],
    hypothesis: RegisteredHypothesisV1,
    *,
    target_group: str | None,
    point_estimate: Decimal | None,
) -> tuple[tuple[LogoEffectV1, ...], Decimal | None]:
    games = tuple(sorted({unit.game_id for unit in units}))
    rows: list[LogoEffectV1] = []
    for held_out in games:
        multiplicities = {game_id: int(game_id != held_out) for game_id in games}
        effect = _estimate_statistic(
            units,
            hypothesis,
            target_group=target_group,
            multiplicities=multiplicities,
        )
        rows.append(
            LogoEffectV1(
                held_out_game_id=held_out,
                effect=effect,
                sign=_sign(effect),  # type: ignore[arg-type]
            )
        )
    valid = tuple(row for row in rows if row.effect is not None)
    stability = (
        None
        if point_estimate is None or not valid
        else Decimal(sum(row.sign == _sign(point_estimate) for row in valid))
        / Decimal(len(valid))
    )
    return tuple(rows), stability


def _concentration_downgrade(
    units: Sequence[AggregatedEpisodeEffectV1],
    status: str,
) -> str:
    if not units:
        return status
    counts = Counter(unit.game_id for unit in units)
    if max(counts.values()) * 4 > len(units) and status == "PRELIMINARY_ELIGIBLE":
        return "DESCRIPTIVE_ONLY"
    return status


def _estimate_support(
    *,
    hypothesis: RegisteredHypothesisV1,
    target_group: str | None,
    units: Sequence[AggregatedEpisodeEffectV1],
    gate_status: str,
) -> str:
    if hypothesis.analysis_kind == "primary_pooled":
        status = _support_status(
            episode_count=len(units),
            game_count=len({unit.game_id for unit in units}),
            required_episodes=300,
            required_games=16,
        )
    elif hypothesis.analysis_kind == "binary_moderator":
        statuses = []
        for group in hypothesis.approved_groups:
            group_units = tuple(unit for unit in units if unit.group == group)
            statuses.append(
                _support_status(
                    episode_count=len(group_units),
                    game_count=len({unit.game_id for unit in group_units}),
                    required_episodes=40,
                    required_games=8,
                )
            )
        status = min(
            statuses,
            key={
                "CASE_ONLY": 0,
                "DESCRIPTIVE_ONLY": 1,
                "PRELIMINARY_ELIGIBLE": 2,
            }.__getitem__,
        )
    else:
        selected = tuple(unit for unit in units if unit.group == target_group)
        status = _support_status(
            episode_count=len(selected),
            game_count=len({unit.game_id for unit in selected}),
            required_episodes=(
                20 if hypothesis.analysis_kind == "named_subtype" else 15
            ),
            required_games=6,
        )
        units = selected
    status = _concentration_downgrade(units, status)
    severity = {
        "CASE_ONLY": 0,
        "DESCRIPTIVE_ONLY": 1,
        "PRELIMINARY_ELIGIBLE": 2,
    }
    return min(
        (status, gate_status),
        key=severity.__getitem__,
    )


def _distribution_sha256(distribution: Sequence[Decimal]) -> str:
    return canonical_sha256(tuple(format(value, "f") for value in distribution))


def _estimate_specs(
    hypotheses: Sequence[RegisteredHypothesisV1],
) -> tuple[tuple[RegisteredHypothesisV1, str | None, str, str], ...]:
    specs: list[tuple[RegisteredHypothesisV1, str | None, str, str]] = []
    for hypothesis in hypotheses:
        if hypothesis.estimand == "group_mean":
            specs.extend(
                (
                    hypothesis,
                    group,
                    f"{hypothesis.hypothesis_id}:{group}",
                    f"mean:{group}",
                )
                for group in hypothesis.approved_groups
            )
        elif hypothesis.estimand == "binary_difference":
            lower, upper = hypothesis.approved_groups
            specs.append(
                (
                    hypothesis,
                    None,
                    hypothesis.hypothesis_id,
                    f"difference:{upper}-minus-{lower}",
                )
            )
        else:
            specs.append(
                (
                    hypothesis,
                    None,
                    hypothesis.hypothesis_id,
                    "pooled_mean",
                )
            )
    return tuple(specs)


def _apply_bh(
    estimates: Sequence[InferenceEstimateV1],
) -> tuple[InferenceEstimateV1, ...]:
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, estimate in enumerate(estimates):
        if estimate.empirical_two_sided_p is not None:
            by_family[estimate.hypothesis_family].append(index)
    result = list(estimates)
    for indexes in by_family.values():
        ranked = sorted(
            indexes,
            key=lambda index: (
                result[index].empirical_two_sided_p,
                result[index].estimate_id,
            ),
        )
        family_size = len(ranked)
        adjusted: dict[int, Decimal] = {}
        running = Decimal(1)
        for reverse_index in range(family_size - 1, -1, -1):
            index = ranked[reverse_index]
            p_value = result[index].empirical_two_sided_p
            assert p_value is not None
            rank = reverse_index + 1
            raw = min(
                Decimal(1),
                p_value * Decimal(family_size) / Decimal(rank),
            )
            running = min(running, raw)
            adjusted[index] = running
        for index, q_value in adjusted.items():
            result[index] = replace(
                result[index],
                bh_q_value=q_value,
            )
    return tuple(result)


def run_registered_inference(
    observations: Sequence[ExplorationObservationV1],
    gate: ExplorationGateReportV1,
    lock: RegisteredAnalysisLockV1,
    bootstrap_samples: int,
    seed: int,
) -> InferenceReportV1:
    """Run only preregistered, game-clustered preliminary inference."""

    if lock != X13_REGISTERED_ANALYSIS_LOCK_V1:
        raise AssociationEvidenceError(
            "analysis lock is not the trusted registered X-13 lock"
        )
    if seed != lock.bootstrap_seed:
        raise AssociationEvidenceError(
            "seed differs from the registered bootstrap seed"
        )
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise TypeError("bootstrap_samples must be a positive integer")
    if not isinstance(gate, ExplorationGateReportV1):
        raise TypeError("gate must be ExplorationGateReportV1")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a sequence")
    if any(not isinstance(row, ExplorationObservationV1) for row in observations):
        raise TypeError("observations must contain ExplorationObservationV1")
    ids = [row.observation_id for row in observations]
    if len(ids) != len(set(ids)):
        raise AssociationEvidenceError("duplicate registered inference observation_id")

    registered = {item.hypothesis_id: item for item in lock.hypotheses}
    selected_hypothesis_ids: set[str] = set()
    for row in observations:
        hypothesis = registered.get(row.hypothesis_id)
        if hypothesis is None:
            raise AssociationEvidenceError(
                f"hypothesis is not registered: {row.hypothesis_id}"
            )
        selected_hypothesis_ids.add(hypothesis.hypothesis_id)
        if row.group not in hypothesis.approved_groups:
            raise AssociationEvidenceError(f"group is not registered: {row.group}")
        if hypothesis.analysis_kind != gate.analysis_kind:
            raise AssociationEvidenceError("hypothesis analysis kind differs from gate")
    aggregated = _aggregate_episode_effects(observations)
    recomputed_gate = evaluate_exploration_gate(
        observations,
        analysis_kind=gate.analysis_kind,
        bootstrap_seed=seed,
    )
    if recomputed_gate != gate:
        raise AssociationEvidenceError(
            "gate does not bind the supplied registered observations"
        )

    selected_hypotheses = tuple(
        registered[hypothesis_id] for hypothesis_id in sorted(selected_hypothesis_ids)
    )
    estimates: list[InferenceEstimateV1] = []
    for hypothesis, target_group, estimate_id, estimand_label in _estimate_specs(
        selected_hypotheses
    ):
        units = tuple(
            unit
            for unit in aggregated
            if unit.hypothesis_id == hypothesis.hypothesis_id
            and (hypothesis.estimand != "group_mean" or unit.group == target_group)
        )
        games = tuple(sorted({unit.game_id for unit in units}))
        point = _estimate_statistic(
            units,
            hypothesis,
            target_group=target_group,
            multiplicities={game_id: 1 for game_id in games},
        )
        distribution = _bootstrap_distribution(
            units,
            hypothesis,
            target_group=target_group,
            estimate_id=estimate_id,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        ordered_distribution = tuple(sorted(distribution))
        ci_lower = _nearest_rank(
            ordered_distribution,
            numerator=25,
            denominator=1000,
        )
        ci_upper = _nearest_rank(
            ordered_distribution,
            numerator=975,
            denominator=1000,
        )
        logo_effects, logo_stability = _logo(
            units,
            hypothesis,
            target_group=target_group,
            point_estimate=point,
        )
        support_status = _estimate_support(
            hypothesis=hypothesis,
            target_group=target_group,
            units=units,
            gate_status=gate.status,
        )
        estimates.append(
            InferenceEstimateV1(
                estimate_id=estimate_id,
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_family=hypothesis.hypothesis_family,
                estimand=estimand_label,
                point_estimate=point,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                bootstrap_requested_samples=bootstrap_samples,
                bootstrap_valid_samples=len(distribution),
                empirical_two_sided_p=_empirical_two_sided_p(distribution),
                bh_q_value=None,
                logo_effects=logo_effects,
                logo_sign_stability=logo_stability,
                episode_count=len(units),
                game_count=len(games),
                support_status=support_status,
                preliminary_inference_claim_permitted=(
                    support_status == "PRELIMINARY_ELIGIBLE"
                ),
                bootstrap_distribution_sha256=_distribution_sha256(distribution),
            )
        )
    return InferenceReportV1(
        experiment_id="X-13",
        status=X13_STATUS,
        lock_id=lock.lock_id,
        gate_status=gate.status,
        bootstrap_seed=seed,
        bootstrap_requested_samples=bootstrap_samples,
        aggregated_episode_effects=aggregated,
        estimates=_apply_bh(estimates),
    )


__all__ = [
    "X13_APPROVED_ANALYSIS_WHITELIST",
    "X13_REGISTERED_ANALYSIS_LOCK_V1",
    "AggregatedEpisodeEffectV1",
    "AssociationEvidenceError",
    "AssociationAttritionRowV1",
    "AssociationAttritionV1",
    "AssociationResultV1",
    "DirectionalAnomalyAuditV1",
    "ExplorationGateReportV1",
    "ExplorationGroupGateV1",
    "ExplorationObservationV1",
    "InferenceEstimateV1",
    "InferenceInputV1",
    "InferenceReportV1",
    "GameTimeIntervalRowV1",
    "GameTimeIntervalsV1",
    "LayerGAuditV1",
    "LayerGEpisodeIntervalV1",
    "LayerMAuditV1",
    "LogoEffectV1",
    "RegisteredAnalysisLockV1",
    "RegisteredHypothesisV1",
    "MarketTimeIntervalRowV1",
    "MarketTimeIntervalsV1",
    "SourceTimelineItemV1",
    "SourceTimelineRowV1",
    "SourceTimelineTableV1",
    "SourceTimelineV1",
    "TimelineAuditRowV1",
    "TimelineAuditV1",
    "X13AssociationRunV1",
    "X13AssociationStreamV1",
    "audit_layer_g",
    "audit_layer_m",
    "build_association_attrition_rows_v1",
    "build_game_time_intervals_v1",
    "build_market_time_intervals_v1",
    "build_source_timeline_v1",
    "build_timeline_audit_rows_v1",
    "evaluate_exploration_gate",
    "iter_x13_associations",
    "prepare_x13_association_stream",
    "run_registered_inference",
    "run_x13_association",
]
