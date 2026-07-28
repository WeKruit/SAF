"""Governed historical prediction-market alpha metric contracts.

The module consumes source-time historical observations only.  It has no
holdout reader, venue client, order model, or production-order output.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Literal, TypeAlias

from prediction_market.sports.nfl_factor_lab_analysis import (
    FactorLabError,
    verify_shortlist_lock,
)


class MetricContractError(ValueError):
    """An input cannot support the governed historical metric claim."""


DataCapability: TypeAlias = Literal[
    "trade_tick", "candle_l1", "l2", "execution"
]
MetricKind: TypeAlias = Literal[
    "probability_delta",
    "threshold_probability",
    "continuation",
    "reversal",
    "ofi",
    "depth",
    "shadow_nav",
    "maximum_drawdown",
    "sharpe_ratio",
]


_CAPABILITIES = frozenset({"trade_tick", "candle_l1", "l2", "execution"})
_METRIC_KINDS = frozenset(
    {
        "probability_delta",
        "threshold_probability",
        "continuation",
        "reversal",
        "ofi",
        "depth",
        "shadow_nav",
        "maximum_drawdown",
        "sharpe_ratio",
    }
)
_PROBABILITY_METRICS = frozenset(
    {"probability_delta", "threshold_probability", "continuation", "reversal"}
)
_NAV_METRICS = frozenset(
    {"shadow_nav", "maximum_drawdown", "sharpe_ratio"}
)
_TRADE_MARK_PRICE_BASIS = "ACTUAL_SUBSEQUENT_TRADE_MARK"
_TRADE_MARK_FILL_STATUS = "NO_FILL_OBSERVED_NON_EXECUTABLE"
_EXECUTION_PRICE_BASIS = "EXECUTION_REPLAY_NO_ORDER"
_EXECUTION_FILL_STATUS = "NO_STRATEGY_FILL_OBSERVED"
_PRICE_BASES = frozenset(
    {_TRADE_MARK_PRICE_BASIS, _EXECUTION_PRICE_BASIS}
)
_FILL_STATUSES = frozenset(
    {_TRADE_MARK_FILL_STATUS, _EXECUTION_FILL_STATUS}
)
_CLASSIFICATION_LABELS = {
    "NON_EXECUTABLE_TRADE_MARKED_SHADOW": (
        _TRADE_MARK_PRICE_BASIS,
        _TRADE_MARK_FILL_STATUS,
    ),
    "EXECUTION_GRADE_SHADOW": (
        _EXECUTION_PRICE_BASIS,
        _EXECUTION_FILL_STATUS,
    ),
}
_FAVORITE_UNDERDOG_PROXIES = frozenset(
    {"favorite", "underdog", "unclassified"}
)
_HOME_AWAY_PROXIES = frozenset(
    {"home", "away", "neutral", "unclassified"}
)
_POPULARITY_PROXIES = frozenset(
    {"popular_team", "other_team", "unclassified"}
)
_LICENSE_STATUSES = frozenset(
    {"approved", "research_only", "pending", "blocked"}
)
_RESEARCH_LICENSE_STATUSES = frozenset({"approved", "research_only"})
_LINK_STATUSES = frozenset({"linked", "partial", "unlinked"})
_GAP_STATUSES = frozenset({"complete", "known_gaps", "unknown"})
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")


def _identifier(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise MetricContractError(f"{field} must be a stable identifier")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise MetricContractError(f"{field} must be nonempty trimmed text")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricContractError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MetricContractError(f"{field} must be finite")
    return number


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricContractError(f"{field} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    number = _nonnegative_integer(value, field=field)
    if number == 0:
        raise MetricContractError(f"{field} must be a positive integer")
    return number


def _probability_delta(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if not -1 <= number <= 1:
        raise MetricContractError(f"{field} must be in [-1, 1]")
    return number


def _probability(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if not 0 <= number <= 1:
        raise MetricContractError(f"{field} must be in [0, 1]")
    return number


def _utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise MetricContractError(f"{field} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise MetricContractError(f"{field} must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MetricContractError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _text_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple or (not value and not allow_empty):
        requirement = "a tuple" if allow_empty else "a nonempty tuple"
        raise MetricContractError(f"{field} must be {requirement}")
    if len(set(value)) != len(value):
        raise MetricContractError(f"{field} must be unique")
    validator = _identifier if identifiers else _text
    for item in value:
        validator(item, field=field)
    return value


def _hash_tuple(value: object, *, field: str) -> tuple[str, ...]:
    values = _text_tuple(value, field=field)
    for item in values:
        _sha256(item, field=field)
    return values


def _capability_set(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> frozenset[DataCapability]:
    if type(value) is not frozenset or (not value and not allow_empty):
        requirement = "a frozenset" if allow_empty else "a nonempty frozenset"
        raise MetricContractError(f"{field} must be {requirement}")
    invalid = set(value) - _CAPABILITIES
    if invalid:
        raise MetricContractError(f"{field} contains invalid capabilities: {sorted(invalid)}")
    return value  # type: ignore[return-value]


def _metric_set(value: object, *, field: str) -> frozenset[MetricKind]:
    if type(value) is not frozenset:
        raise MetricContractError(f"{field} must be a frozenset")
    invalid = set(value) - _METRIC_KINDS
    if invalid:
        raise MetricContractError(f"{field} contains invalid metrics: {sorted(invalid)}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TickDataInventoryV1:
    """Immutable evidence inventory for one venue market."""

    market_id: str
    venue: str
    source_id: str
    capabilities: frozenset[DataCapability]
    coverage_start: datetime
    coverage_end: datetime
    timestamp_domains: tuple[str, ...]
    direction_authority: str
    storage_refs: tuple[str, ...]
    object_hashes: tuple[str, ...]
    gap_status: str
    gap_count: int
    license_id: str
    license_status: str
    allowed_metrics: frozenset[MetricKind]
    link_status: str
    claim_boundary: str = "HISTORICAL_SOURCE_TIME_ONLY"

    def __post_init__(self) -> None:
        _identifier(self.market_id, field="market_id")
        _identifier(self.venue, field="venue")
        _identifier(self.source_id, field="source_id")
        _capability_set(self.capabilities, field="capabilities")
        coverage_start = _utc(self.coverage_start, field="coverage_start")
        coverage_end = _utc(self.coverage_end, field="coverage_end")
        if coverage_end <= coverage_start:
            raise MetricContractError("coverage_end must be after coverage_start")
        _text_tuple(
            self.timestamp_domains,
            field="timestamp_domains",
            identifiers=True,
        )
        _text(self.direction_authority, field="direction_authority")
        storage_refs = _text_tuple(self.storage_refs, field="storage_refs")
        object_hashes = _hash_tuple(self.object_hashes, field="object_hashes")
        if len(storage_refs) != len(object_hashes):
            raise MetricContractError(
                "storage_refs and object_hashes must have equal lengths"
            )
        if self.gap_status not in _GAP_STATUSES:
            raise MetricContractError(f"gap_status must be one of {sorted(_GAP_STATUSES)}")
        _nonnegative_integer(self.gap_count, field="gap_count")
        if self.gap_status == "complete" and self.gap_count != 0:
            raise MetricContractError("gap_count must be zero when gap_status is complete")
        if self.gap_status == "known_gaps" and self.gap_count == 0:
            raise MetricContractError(
                "gap_count must be positive when gap_status is known_gaps"
            )
        _identifier(self.license_id, field="license_id")
        if self.license_status not in _LICENSE_STATUSES:
            raise MetricContractError(
                f"license_status must be one of {sorted(_LICENSE_STATUSES)}"
            )
        _metric_set(self.allowed_metrics, field="allowed_metrics")
        if self.link_status not in _LINK_STATUSES:
            raise MetricContractError(
                f"link_status must be one of {sorted(_LINK_STATUSES)}"
            )
        if self.claim_boundary != "HISTORICAL_SOURCE_TIME_ONLY":
            raise MetricContractError(
                "claim_boundary must be HISTORICAL_SOURCE_TIME_ONLY"
            )


@dataclass(frozen=True, slots=True)
class MarketMetricDefinitionV1:
    """Frozen formula, eligibility, grouping, and capability contract."""

    metric_id: str
    version: str
    layer: str
    metric_kind: MetricKind
    formula: str
    orientation: str
    window_seconds: int
    exclusions: tuple[str, ...]
    grouping: tuple[str, ...]
    claim_boundary: str
    required_capabilities: frozenset[DataCapability]
    threshold: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.metric_id, field="metric_id")
        _identifier(self.version, field="version")
        _identifier(self.layer, field="layer")
        if self.metric_kind not in _METRIC_KINDS:
            raise MetricContractError(f"metric_kind is invalid: {self.metric_kind!r}")
        _text(self.formula, field="formula")
        _text(self.orientation, field="orientation")
        _positive_integer(self.window_seconds, field="window_seconds")
        _text_tuple(self.exclusions, field="exclusions", allow_empty=True)
        _text_tuple(self.grouping, field="grouping", identifiers=True)
        if self.claim_boundary != "HISTORICAL_SOURCE_TIME_ONLY":
            raise MetricContractError(
                "claim_boundary must be HISTORICAL_SOURCE_TIME_ONLY"
            )
        capabilities = _capability_set(
            self.required_capabilities,
            field="required_capabilities",
        )
        if self.metric_kind in {"ofi", "depth"} and "l2" not in capabilities:
            raise MetricContractError(f"{self.metric_kind} requires l2 capability")
        if self.threshold is not None:
            threshold = _finite(self.threshold, field="threshold")
            if not 0 <= threshold <= 1:
                raise MetricContractError("threshold must be in [0, 1]")

    @property
    def logical_contract_id(self) -> str:
        return f"{self.metric_id}:{self.version}"


@dataclass(frozen=True, slots=True)
class MarketMetricObservationV1:
    """One governed metric value with metric-specific value semantics."""

    metric_id: str
    logical_contract_id: str
    metric_kind: MetricKind
    market_id: str
    venue: str
    game_id: str
    episode_id: str
    logical_proposition_id: str
    market_family: str
    period: str
    line: float | None
    settlement_rule: str
    outcome: str
    reference_id: str
    source_time: datetime
    time_anchor: datetime
    horizon_seconds: int
    value: float
    probability_delta: float | None
    continuation_delta: float | None
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    provenance_hashes: tuple[str, ...]
    dataset_partition: str = "historical"

    def __post_init__(self) -> None:
        _identifier(self.metric_id, field="metric_id")
        _identifier(self.logical_contract_id, field="logical_contract_id")
        if self.metric_kind not in _METRIC_KINDS:
            raise MetricContractError(
                f"metric_kind is invalid: {self.metric_kind!r}"
            )
        _identifier(self.market_id, field="market_id")
        _identifier(self.venue, field="venue")
        _identifier(self.game_id, field="game_id")
        _identifier(self.episode_id, field="episode_id")
        _identifier(
            self.logical_proposition_id,
            field="logical_proposition_id",
        )
        _identifier(self.market_family, field="market_family")
        _identifier(self.period, field="period")
        if self.line is not None:
            _finite(self.line, field="line")
        _identifier(self.settlement_rule, field="settlement_rule")
        _identifier(self.outcome, field="outcome")
        _identifier(self.reference_id, field="reference_id")
        source_time = _utc(self.source_time, field="source_time")
        time_anchor = _utc(self.time_anchor, field="time_anchor")
        if time_anchor != source_time:
            raise MetricContractError("time_anchor must equal source_time")
        _positive_integer(self.horizon_seconds, field="horizon_seconds")
        if self.metric_kind in _PROBABILITY_METRICS:
            value = _probability_delta(
                self.value,
                field="probability metric value",
            )
            if self.probability_delta is None:
                raise MetricContractError(
                    "probability metrics require probability_delta"
                )
            probability_delta = _probability_delta(
                self.probability_delta,
                field="probability_delta",
            )
            if value != probability_delta:
                raise MetricContractError(
                    "value must equal probability_delta for probability metrics"
                )
            if self.continuation_delta is not None:
                _probability_delta(
                    self.continuation_delta,
                    field="continuation_delta",
                )
        else:
            value = _finite(self.value, field=f"{self.metric_kind} value")
            if self.metric_kind == "depth" and value < 0:
                raise MetricContractError(
                    "depth value must be nonnegative"
                )
            if self.probability_delta is not None:
                raise MetricContractError(
                    f"{self.metric_kind} probability_delta must be null"
                )
            if self.continuation_delta is not None:
                raise MetricContractError(
                    f"{self.metric_kind} continuation_delta must be null"
                )
        if type(self.eligible) is not bool:
            raise MetricContractError("eligible must be boolean")
        reasons = _text_tuple(
            self.exclusion_reasons,
            field="exclusion_reasons",
            allow_empty=True,
        )
        if self.eligible and reasons:
            raise MetricContractError(
                "eligible observations cannot have exclusion_reasons"
            )
        if not self.eligible and not reasons:
            raise MetricContractError(
                "ineligible observations require exclusion_reasons"
            )
        _hash_tuple(self.provenance_hashes, field="provenance_hashes")
        if self.dataset_partition != "historical":
            raise MetricContractError(
                "dataset_partition must be historical; holdout is not an input"
            )


@dataclass(frozen=True, slots=True)
class CalibrationForecastObservationV1:
    """One deduplicated, contract-bound, PIT-safe forecast/label pair."""

    forecast_id: str
    logical_contract_id: str
    realized_label_contract_id: str
    game_id: str
    reference_id: str
    outcome: str
    forecast_time: datetime
    label_time: datetime
    forecast_probability: float
    realized_label: int
    favorite_underdog_proxy: str
    home_away_proxy: str
    popularity_proxy: str
    provenance_hashes: tuple[str, ...]
    dataset_partition: str = "historical"

    def __post_init__(self) -> None:
        _identifier(self.forecast_id, field="forecast_id")
        _identifier(
            self.logical_contract_id,
            field="logical_contract_id",
        )
        _identifier(
            self.realized_label_contract_id,
            field="realized_label_contract_id",
        )
        if self.realized_label_contract_id != self.logical_contract_id:
            raise MetricContractError(
                "realized label contract must match forecast logical contract"
            )
        _identifier(self.game_id, field="game_id")
        _identifier(self.reference_id, field="reference_id")
        _identifier(self.outcome, field="outcome")
        forecast_time = _utc(self.forecast_time, field="forecast_time")
        label_time = _utc(self.label_time, field="label_time")
        if label_time <= forecast_time:
            raise MetricContractError(
                "PIT-safe calibration requires label_time after forecast_time"
            )
        _probability(
            self.forecast_probability,
            field="forecast_probability",
        )
        if type(self.realized_label) is not int or self.realized_label not in {
            0,
            1,
        }:
            raise MetricContractError("realized_label must be binary 0/1")
        if self.favorite_underdog_proxy not in _FAVORITE_UNDERDOG_PROXIES:
            raise MetricContractError(
                "favorite_underdog_proxy is invalid"
            )
        if self.home_away_proxy not in _HOME_AWAY_PROXIES:
            raise MetricContractError("home_away_proxy is invalid")
        if self.popularity_proxy not in _POPULARITY_PROXIES:
            raise MetricContractError("popularity_proxy is invalid")
        _hash_tuple(self.provenance_hashes, field="provenance_hashes")
        if self.dataset_partition != "historical":
            raise MetricContractError(
                "dataset_partition must be historical; holdout is not an input"
            )


@dataclass(frozen=True, slots=True)
class ShadowStrategySpecV1:
    """Fully frozen strategy bound to one formally verified locked candidate."""

    strategy_id: str
    factor_id: str
    factor_version: str
    entry_rule: str
    exit_rule: str
    horizon_seconds: int
    fixed_size: float
    venue: str
    market_family: str
    max_staleness_seconds: int
    fee_bps: float
    max_concurrent_positions: int
    shortlist_manifest_path: Path
    shortlist_manifest_sha256: str
    initial_nav: float

    def __post_init__(self) -> None:
        _identifier(self.strategy_id, field="strategy_id")
        _identifier(self.factor_id, field="factor_id")
        _identifier(self.factor_version, field="factor_version")
        _text(self.entry_rule, field="entry_rule")
        _text(self.exit_rule, field="exit_rule")
        _positive_integer(self.horizon_seconds, field="horizon_seconds")
        if _finite(self.fixed_size, field="fixed_size") <= 0:
            raise MetricContractError("fixed_size must be positive")
        _identifier(self.venue, field="venue")
        _identifier(self.market_family, field="market_family")
        _nonnegative_integer(
            self.max_staleness_seconds,
            field="max_staleness_seconds",
        )
        if _finite(self.fee_bps, field="fee_bps") < 0:
            raise MetricContractError("fee_bps must be nonnegative")
        _positive_integer(
            self.max_concurrent_positions,
            field="max_concurrent_positions",
        )
        if not isinstance(self.shortlist_manifest_path, Path):
            raise MetricContractError("shortlist_manifest_path must be a Path")
        expected_hash = _sha256(
            self.shortlist_manifest_sha256,
            field="shortlist_manifest_sha256",
        )
        try:
            lock = verify_shortlist_lock(self.shortlist_manifest_path)
            payload = json.loads(self.shortlist_manifest_path.read_bytes())
        except (
            FactorLabError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise MetricContractError(
                "shortlist_manifest_path must reference a formal shortlist lock"
            ) from error
        if lock.shortlist_sha256 != expected_hash:
            raise MetricContractError(
                "formal shortlist lock hash does not match "
                "shortlist_manifest_sha256"
            )
        if payload.get("holdout_reaction_accessed") is not False:
            raise MetricContractError(
                "shortlist manifest must prove holdout_reaction_accessed=false"
            )
        candidate_key = (
            f"{self.factor_id}|{self.market_family}|h={self.horizon_seconds}"
        )
        if candidate_key not in lock.candidate_keys:
            raise MetricContractError(
                f"strategy does not match a locked candidate: {candidate_key}"
            )
        definitions = payload.get("factor_definitions")
        matching_definitions = (
            [
                definition
                for definition in definitions
                if isinstance(definition, dict)
                and definition.get("factor_id") == self.factor_id
            ]
            if isinstance(definitions, list)
            else []
        )
        if len(matching_definitions) != 1:
            raise MetricContractError(
                "locked candidate must have one factor definition"
            )
        definition = matching_definitions[0]
        if definition.get("version") != self.factor_version:
            raise MetricContractError(
                "strategy factor_version does not match locked definition"
            )
        if (
            self.market_family not in definition.get("market_families", ())
            or self.horizon_seconds
            not in definition.get("horizons_seconds", ())
        ):
            raise MetricContractError(
                "locked definition does not support strategy family/horizon"
            )
        if _finite(self.initial_nav, field="initial_nav") <= 0:
            raise MetricContractError("initial_nav must be positive")


@dataclass(frozen=True, slots=True)
class ShadowNAVObservationV1:
    """Historical NAV mark with explicit price, fee, and fill claim boundary."""

    strategy_id: str
    game_id: str
    episode_id: str
    source_time: datetime
    nav: float
    price_basis: str
    fees: float
    fill_status: str
    executable: bool
    dataset_partition: str = "historical"

    def __post_init__(self) -> None:
        _identifier(self.strategy_id, field="strategy_id")
        _identifier(self.game_id, field="game_id")
        _identifier(self.episode_id, field="episode_id")
        _utc(self.source_time, field="source_time")
        if _finite(self.nav, field="nav") <= 0:
            raise MetricContractError("nav must be positive")
        if self.price_basis not in _PRICE_BASES:
            raise MetricContractError(
                f"price_basis must be one of {sorted(_PRICE_BASES)}"
            )
        if _finite(self.fees, field="fees") < 0:
            raise MetricContractError("fees must be nonnegative")
        if self.fill_status not in _FILL_STATUSES:
            raise MetricContractError(
                f"fill_status must be one of {sorted(_FILL_STATUSES)}"
            )
        if self.executable is not False:
            raise MetricContractError(
                "historical shadow NAV executable must be false"
            )
        if self.dataset_partition != "historical":
            raise MetricContractError(
                "dataset_partition must be historical; holdout is not an input"
            )


def validate_metric_inventory(
    *,
    definition: MarketMetricDefinitionV1,
    inventory: TickDataInventoryV1,
) -> None:
    """Require exact declared capabilities and governed research-use evidence."""

    if not isinstance(definition, MarketMetricDefinitionV1):
        raise MetricContractError("definition must be MarketMetricDefinitionV1")
    if not isinstance(inventory, TickDataInventoryV1):
        raise MetricContractError("inventory must be TickDataInventoryV1")
    missing = definition.required_capabilities - inventory.capabilities
    if missing:
        raise MetricContractError(
            "inventory is missing required capabilities: "
            + ", ".join(sorted(missing))
        )
    if definition.metric_kind not in inventory.allowed_metrics:
        raise MetricContractError(
            f"{definition.metric_kind} is absent from allowed_metrics"
        )
    if inventory.license_status not in _RESEARCH_LICENSE_STATUSES:
        raise MetricContractError(
            f"license_status {inventory.license_status} is not research-usable"
        )
    if inventory.link_status != "linked":
        raise MetricContractError("link_status must be linked")
    if "source_time" not in inventory.timestamp_domains:
        raise MetricContractError(
            "timestamp_domains must include source_time"
        )
    if definition.claim_boundary != inventory.claim_boundary:
        raise MetricContractError(
            "definition and inventory claim_boundary must match"
        )


def _all_probability_observations(
    observations: Sequence[MarketMetricObservationV1],
) -> tuple[MarketMetricObservationV1, ...]:
    rows = tuple(observations)
    if not rows:
        raise MetricContractError(
            "probability metrics require at least one observation"
        )
    if any(not isinstance(row, MarketMetricObservationV1) for row in rows):
        raise MetricContractError(
            "observations must be MarketMetricObservationV1"
        )
    if any(row.metric_kind not in _PROBABILITY_METRICS for row in rows):
        raise MetricContractError(
            "probability calculations require probability metric observations"
        )
    return rows


def _eligible_probability_observations(
    observations: Sequence[MarketMetricObservationV1],
) -> tuple[MarketMetricObservationV1, ...]:
    rows = tuple(row for row in _all_probability_observations(observations) if row.eligible)
    if not rows:
        raise MetricContractError(
            "probability metrics require at least one eligible observation"
        )
    return rows


def _validated_probability_observations(
    observations: Sequence[MarketMetricObservationV1],
    *,
    definition: MarketMetricDefinitionV1,
    inventory: TickDataInventoryV1,
) -> tuple[
    tuple[MarketMetricObservationV1, ...],
    tuple[MarketMetricObservationV1, ...],
]:
    validate_metric_inventory(definition=definition, inventory=inventory)
    if definition.metric_kind not in _PROBABILITY_METRICS:
        raise MetricContractError(
            "probability-delta summary only accepts probability metric definitions"
        )
    all_rows = _all_probability_observations(observations)
    for row in all_rows:
        if row.metric_id != definition.metric_id:
            raise MetricContractError(
                "all observations must use definition.metric_id"
            )
        if row.metric_kind != definition.metric_kind:
            raise MetricContractError(
                "observation metric_kind must match definition.metric_kind"
            )
        if row.logical_contract_id != definition.logical_contract_id:
            raise MetricContractError(
                "observation logical_contract_id must match definition"
            )
        if row.horizon_seconds != definition.window_seconds:
            raise MetricContractError(
                "observation horizon_seconds must match definition.window_seconds"
            )
        if row.market_id != inventory.market_id:
            raise MetricContractError(
                "all observations must use inventory.market_id"
            )
        if row.venue != inventory.venue:
            raise MetricContractError(
                "all observations must use inventory.venue"
            )
        if not inventory.coverage_start <= row.source_time <= inventory.coverage_end:
            raise MetricContractError(
                "observation source_time is outside inventory coverage"
            )
        unknown_exclusions = set(row.exclusion_reasons) - set(
            definition.exclusions
        )
        if unknown_exclusions:
            raise MetricContractError(
                "observation exclusion_reasons are absent from definition.exclusions"
            )
    eligible = tuple(row for row in all_rows if row.eligible)
    if not eligible:
        raise MetricContractError(
            "probability metrics require at least one eligible observation"
        )
    return all_rows, eligible


def summarize_historical_probability_delta(
    *,
    observations: Sequence[MarketMetricObservationV1],
    definition: MarketMetricDefinitionV1,
    inventory: TickDataInventoryV1,
) -> dict[str, object]:
    """Report governed historical Δp distribution and post-move path rates."""

    all_rows, rows = _validated_probability_observations(
        observations,
        definition=definition,
        inventory=inventory,
    )
    deltas = tuple(row.probability_delta for row in rows)
    follow_ups = tuple(
        row
        for row in rows
        if row.probability_delta != 0
        and row.continuation_delta is not None
        and row.continuation_delta != 0
    )
    continuations = sum(
        row.probability_delta * row.continuation_delta > 0  # type: ignore[operator]
        for row in follow_ups
    )
    reversals = sum(
        row.probability_delta * row.continuation_delta < 0  # type: ignore[operator]
        for row in follow_ups
    )
    threshold_probability = (
        None
        if definition.threshold is None
        else sum(
            abs(delta) >= definition.threshold for delta in deltas
        )
        / len(deltas)
    )
    return {
        "metric_id": definition.metric_id,
        "logical_contract_id": definition.logical_contract_id,
        "market_id": inventory.market_id,
        "venue": inventory.venue,
        "claim_boundary": definition.claim_boundary,
        "capabilities": tuple(sorted(inventory.capabilities)),
        "gap_status": inventory.gap_status,
        "gap_count": inventory.gap_count,
        "observation_count": len(rows),
        "excluded_observation_count": len(all_rows) - len(rows),
        "probability_delta_distribution": tuple(sorted(deltas)),
        "mean_probability_delta": mean(deltas),
        "threshold": definition.threshold,
        "threshold_probability": threshold_probability,
        "follow_up_count": len(follow_ups),
        "continuation_probability": (
            continuations / len(follow_ups) if follow_ups else None
        ),
        "reversal_probability": (
            reversals / len(follow_ups) if follow_ups else None
        ),
    }


def _per_game_means(
    observations: Sequence[MarketMetricObservationV1],
) -> tuple[tuple[str, float], ...]:
    rows = _eligible_probability_observations(observations)
    by_game: dict[str, list[float]] = {}
    for row in rows:
        by_game.setdefault(row.game_id, []).append(row.probability_delta)
    if len(by_game) < 2:
        raise MetricContractError(
            "game-cluster inference requires at least two games"
        )
    return tuple(
        (game_id, mean(values))
        for game_id, values in sorted(by_game.items())
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise MetricContractError(
            "cannot calculate a quantile without values"
        )
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def game_cluster_bootstrap_probability_delta(
    observations: Sequence[MarketMetricObservationV1],
    *,
    draw_count: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Bootstrap the equal-weighted per-game Δp mean by whole game."""

    _positive_integer(draw_count, field="draw_count")
    if type(seed) is not int:
        raise MetricContractError("seed must be an integer")
    confidence = _finite(confidence_level, field="confidence_level")
    if not 0 < confidence < 1:
        raise MetricContractError("confidence_level must be in (0, 1)")
    game_means = _per_game_means(observations)
    values = tuple(value for _, value in game_means)
    rng = random.Random(seed)
    draws = tuple(
        mean(rng.choices(values, k=len(values))) for _ in range(draw_count)
    )
    alpha = (1 - confidence) / 2
    return {
        "cluster_unit": "game_id",
        "game_count": len(game_means),
        "point_estimate": mean(values),
        "confidence_level": confidence,
        "confidence_interval": (
            _quantile(draws, alpha),
            _quantile(draws, 1 - alpha),
        ),
        "bootstrap_draws": draws,
    }


def leave_one_game_out_probability_delta(
    observations: Sequence[MarketMetricObservationV1],
) -> dict[str, object]:
    """Re-estimate the equal-weighted Δp mean after omitting each game."""

    game_means = _per_game_means(observations)
    estimates = tuple(
        {
            "omitted_game_id": omitted_game_id,
            "point_estimate": mean(
                value
                for game_id, value in game_means
                if game_id != omitted_game_id
            ),
        }
        for omitted_game_id, _ in game_means
    )
    return {
        "cluster_unit": "game_id",
        "full_point_estimate": mean(value for _, value in game_means),
        "estimates": estimates,
    }


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return BH q-values for one explicit hypothesis family."""

    if not p_values:
        raise MetricContractError("p_values must be nonempty")
    checked: list[tuple[str, float]] = []
    for hypothesis_id, p_value in p_values.items():
        _identifier(hypothesis_id, field="hypothesis_id")
        probability = _finite(
            p_value,
            field=f"p_values[{hypothesis_id!r}]",
        )
        if not 0 <= probability <= 1:
            raise MetricContractError("p-values must be in [0, 1]")
        checked.append((hypothesis_id, probability))
    ranked = sorted(checked, key=lambda item: (item[1], item[0]))
    count = len(ranked)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(count, 0, -1):
        hypothesis_id, p_value = ranked[rank - 1]
        running = min(running, p_value * count / rank)
        adjusted[hypothesis_id] = running
    return {
        hypothesis_id: adjusted[hypothesis_id]
        for hypothesis_id, _ in checked
    }


def _calibration_parameters(
    observations: Sequence[CalibrationForecastObservationV1],
) -> tuple[float, float]:
    labels = tuple(row.realized_label for row in observations)
    if len(set(labels)) != 2:
        raise MetricContractError(
            "calibration slope/intercept require both realized classes"
        )
    logits = tuple(
        math.log(
            min(max(row.forecast_probability, 1e-6), 1 - 1e-6)
            / (
                1
                - min(
                    max(row.forecast_probability, 1e-6),
                    1 - 1e-6,
                )
            )
        )
        for row in observations
    )
    intercept = 0.0
    slope = 0.0
    ridge = 1e-9
    for _ in range(100):
        fitted: list[float] = []
        for logit in logits:
            linear = intercept + slope * logit
            if linear >= 0:
                fitted.append(1 / (1 + math.exp(-linear)))
            else:
                exponential = math.exp(linear)
                fitted.append(exponential / (1 + exponential))
        weights = tuple(
            max(probability * (1 - probability), 1e-12)
            for probability in fitted
        )
        gradient_intercept = sum(
            label - probability
            for label, probability in zip(labels, fitted)
        )
        gradient_slope = sum(
            (label - probability) * logit
            for label, probability, logit in zip(
                labels,
                fitted,
                logits,
            )
        )
        h00 = sum(weights) + ridge
        h01 = sum(
            weight * logit
            for weight, logit in zip(weights, logits)
        )
        h11 = (
            sum(
                weight * logit * logit
                for weight, logit in zip(weights, logits)
            )
            + ridge
        )
        determinant = h00 * h11 - h01 * h01
        if determinant <= 0 or not math.isfinite(determinant):
            raise MetricContractError(
                "calibration slope/intercept are not identifiable"
            )
        intercept_step = (
            gradient_intercept * h11 - gradient_slope * h01
        ) / determinant
        slope_step = (
            h00 * gradient_slope - h01 * gradient_intercept
        ) / determinant
        intercept += intercept_step
        slope += slope_step
        if max(abs(intercept_step), abs(slope_step)) < 1e-10:
            break
    if not math.isfinite(intercept) or not math.isfinite(slope):
        raise MetricContractError(
            "calibration slope/intercept are not finite"
        )
    return slope, intercept


def _calibration_point_metrics(
    observations: Sequence[CalibrationForecastObservationV1],
) -> dict[str, float]:
    probabilities = tuple(row.forecast_probability for row in observations)
    labels = tuple(row.realized_label for row in observations)
    clipped = tuple(
        min(max(probability, 1e-12), 1 - 1e-12)
        for probability in probabilities
    )
    slope, intercept = _calibration_parameters(observations)
    return {
        "brier_score": mean(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        ),
        "log_loss": -mean(
            label * math.log(probability)
            + (1 - label) * math.log(1 - probability)
            for probability, label in zip(clipped, labels)
        ),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _reliability_bins(
    observations: Sequence[CalibrationForecastObservationV1],
    *,
    bin_count: int,
) -> tuple[dict[str, object], ...]:
    by_bin: dict[int, list[CalibrationForecastObservationV1]] = {}
    for row in observations:
        index = min(
            int(row.forecast_probability * bin_count),
            bin_count - 1,
        )
        by_bin.setdefault(index, []).append(row)
    return tuple(
        {
            "bin_index": index,
            "bin_lower": index / bin_count,
            "bin_upper": (index + 1) / bin_count,
            "forecast_count": len(rows),
            "game_count": len({row.game_id for row in rows}),
            "mean_forecast": mean(
                row.forecast_probability for row in rows
            ),
            "realized_rate": mean(row.realized_label for row in rows),
        }
        for index, rows in sorted(by_bin.items())
    )


def _proxy_breakdown(
    observations: Sequence[CalibrationForecastObservationV1],
    *,
    field: str,
) -> tuple[dict[str, object], ...]:
    by_proxy: dict[str, list[CalibrationForecastObservationV1]] = {}
    for row in observations:
        proxy_value = getattr(row, field)
        by_proxy.setdefault(proxy_value, []).append(row)
    result: list[dict[str, object]] = []
    for proxy_value, rows in sorted(by_proxy.items()):
        probabilities = tuple(row.forecast_probability for row in rows)
        labels = tuple(row.realized_label for row in rows)
        clipped = tuple(
            min(max(probability, 1e-12), 1 - 1e-12)
            for probability in probabilities
        )
        result.append(
            {
                field: proxy_value,
                "claim_boundary": "DESCRIPTIVE_PROXY_ONLY",
                "forecast_count": len(rows),
                "game_count": len({row.game_id for row in rows}),
                "brier_score": mean(
                    (probability - label) ** 2
                    for probability, label in zip(
                        probabilities,
                        labels,
                    )
                ),
                "log_loss": -mean(
                    label * math.log(probability)
                    + (1 - label) * math.log(1 - probability)
                    for probability, label in zip(clipped, labels)
                ),
            }
        )
    return tuple(result)


def evaluate_game_clustered_calibration(
    observations: Sequence[CalibrationForecastObservationV1],
    *,
    bin_count: int,
    bootstrap_draw_count: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Evaluate deduplicated forecasts with whole-game bootstrap uncertainty."""

    rows = tuple(observations)
    if not rows:
        raise MetricContractError(
            "calibration requires at least one forecast"
        )
    if any(
        not isinstance(row, CalibrationForecastObservationV1)
        for row in rows
    ):
        raise MetricContractError(
            "observations must be CalibrationForecastObservationV1"
        )
    forecast_ids = tuple(row.forecast_id for row in rows)
    # Calibration is a game-level forecast study, not a tick-weighted study.
    # `reference_id` identifies lineage and may legitimately differ between
    # market ticks, so it must not make repeated forecasts for the same game
    # appear independent.  The frozen sampling unit is exactly one forecast
    # per logical contract, game and oriented outcome.
    deduplication_keys = tuple(
        (
            row.logical_contract_id,
            row.game_id,
            row.outcome,
        )
        for row in rows
    )
    if len(set(forecast_ids)) != len(forecast_ids) or len(
        set(deduplication_keys)
    ) != len(deduplication_keys):
        raise MetricContractError(
            "duplicate forecast observations are not allowed"
        )
    if len({row.logical_contract_id for row in rows}) != 1:
        raise MetricContractError(
            "calibration requires one logical forecast contract"
        )
    if len({row.outcome for row in rows}) != 1:
        raise MetricContractError(
            "calibration requires one contract-bound outcome"
        )
    by_game: dict[str, tuple[CalibrationForecastObservationV1, ...]] = {}
    for game_id in sorted({row.game_id for row in rows}):
        by_game[game_id] = tuple(
            row for row in rows if row.game_id == game_id
        )
    if len(by_game) < 2:
        raise MetricContractError(
            "game-clustered calibration requires at least two games"
        )
    _positive_integer(bin_count, field="bin_count")
    _positive_integer(
        bootstrap_draw_count,
        field="bootstrap_draw_count",
    )
    if bootstrap_draw_count < 20:
        raise MetricContractError(
            "bootstrap_draw_count must be at least 20"
        )
    if type(seed) is not int:
        raise MetricContractError("seed must be an integer")
    confidence = _finite(confidence_level, field="confidence_level")
    if not 0 < confidence < 1:
        raise MetricContractError(
            "confidence_level must be in (0, 1)"
        )
    point = _calibration_point_metrics(rows)
    game_ids = tuple(by_game)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        name: [] for name in point
    }
    for _ in range(bootstrap_draw_count):
        selected_games = rng.choices(game_ids, k=len(game_ids))
        sample = tuple(
            row
            for game_id in selected_games
            for row in by_game[game_id]
        )
        try:
            metrics = _calibration_point_metrics(sample)
        except MetricContractError:
            continue
        for name, value in metrics.items():
            samples[name].append(value)
    valid_draw_count = min(len(values) for values in samples.values())
    minimum_valid_draws = max(20, bootstrap_draw_count // 2)
    if valid_draw_count < minimum_valid_draws:
        raise MetricContractError(
            "too few valid game-cluster bootstrap calibration draws: "
            f"{valid_draw_count} < {minimum_valid_draws}"
        )
    alpha = (1 - confidence) / 2
    confidence_intervals = {
        name: (
            _quantile(values, alpha),
            _quantile(values, 1 - alpha),
        )
        for name, values in samples.items()
    }
    proxy_fields = (
        "favorite_underdog_proxy",
        "home_away_proxy",
        "popularity_proxy",
    )
    return {
        "logical_contract_id": rows[0].logical_contract_id,
        "outcome": rows[0].outcome,
        "claim_boundary": "HISTORICAL_CONTRACT_BOUND_PIT_SAFE",
        "sampling_unit": "DEDUPLICATED_CONTRACT_BOUND_FORECAST",
        "cluster_unit": "game_id",
        "forecast_count": len(rows),
        "game_count": len(game_ids),
        **point,
        "reliability_bins": _reliability_bins(
            rows,
            bin_count=bin_count,
        ),
        "game_cluster_bootstrap_ci": confidence_intervals,
        "bootstrap_draw_count_requested": bootstrap_draw_count,
        "bootstrap_valid_draw_count": valid_draw_count,
        "confidence_level": confidence,
        "seed": seed,
        "proxy_breakdowns": {
            field: _proxy_breakdown(rows, field=field)
            for field in proxy_fields
        },
    }


def compare_same_episode_venues(
    *,
    observations: Sequence[MarketMetricObservationV1],
    left_venue: str,
    right_venue: str,
) -> dict[str, object]:
    """Pair only the same contract, game, episode, outcome, and horizon."""

    left = _identifier(left_venue, field="left_venue")
    right = _identifier(right_venue, field="right_venue")
    if left == right:
        raise MetricContractError("left_venue and right_venue must differ")
    rows = _eligible_probability_observations(observations)
    contracts = {row.logical_contract_id for row in rows}
    if len(contracts) != 1:
        raise MetricContractError(
            "paired comparison requires one logical_contract_id"
        )
    values: dict[
        tuple[
            str,
            str,
            str,
            str,
            str,
            float | None,
            str,
            str,
            int,
        ],
        dict[str, float],
    ] = {}
    for row in rows:
        if row.venue not in {left, right}:
            continue
        key = (
            row.game_id,
            row.episode_id,
            row.logical_proposition_id,
            row.market_family,
            row.period,
            row.line,
            row.settlement_rule,
            row.outcome,
            row.horizon_seconds,
        )
        per_venue = values.setdefault(key, {})
        if row.venue in per_venue:
            raise MetricContractError(
                "paired comparison requires at most one observation "
                "per venue episode outcome"
            )
        per_venue[row.venue] = row.probability_delta
    pairs = tuple(
        {
            "game_id": game_id,
            "episode_id": episode_id,
            "logical_proposition_id": logical_proposition_id,
            "market_family": market_family,
            "period": period,
            "line": line,
            "settlement_rule": settlement_rule,
            "outcome": outcome,
            "horizon_seconds": horizon_seconds,
            "left_probability_delta": per_venue[left],
            "right_probability_delta": per_venue[right],
            "right_minus_left": per_venue[right] - per_venue[left],
        }
        for (
            game_id,
            episode_id,
            logical_proposition_id,
            market_family,
            period,
            line,
            settlement_rule,
            outcome,
            horizon_seconds,
        ), per_venue in sorted(
            values.items(),
            key=lambda item: repr(item[0]),
        )
        if left in per_venue and right in per_venue
    )
    differences = tuple(
        float(pair["right_minus_left"]) for pair in pairs
    )
    return {
        "left_venue": left,
        "right_venue": right,
        "pair_count": len(pairs),
        "right_minus_left_mean": (
            mean(differences) if differences else None
        ),
        "pairs": pairs,
    }


def _shadow_inventory_classification(
    inventories: Sequence[TickDataInventoryV1],
    *,
    strategy: ShadowStrategySpecV1,
) -> str:
    items = tuple(inventories)
    if not items:
        raise MetricContractError(
            "shadow NAV requires at least one inventory"
        )
    if any(not isinstance(item, TickDataInventoryV1) for item in items):
        raise MetricContractError(
            "inventories must be TickDataInventoryV1"
        )
    for item in items:
        if item.venue != strategy.venue:
            raise MetricContractError(
                "all shadow NAV inventories must use strategy.venue"
            )
        if item.license_status not in _RESEARCH_LICENSE_STATUSES:
            raise MetricContractError(
                f"license_status {item.license_status} is not research-usable"
            )
        if item.link_status != "linked":
            raise MetricContractError("link_status must be linked")
        missing_metrics = _NAV_METRICS - item.allowed_metrics
        if missing_metrics:
            raise MetricContractError(
                "shadow NAV inventories allowed_metrics are missing: "
                + ", ".join(sorted(missing_metrics))
            )
    if all("execution" in item.capabilities for item in items):
        return "EXECUTION_GRADE_SHADOW"
    if all("trade_tick" in item.capabilities for item in items):
        return "NON_EXECUTABLE_TRADE_MARKED_SHADOW"
    raise MetricContractError(
        "shadow NAV requires explicit trade_tick or execution capability"
    )


def summarize_shadow_nav(
    observations: Sequence[ShadowNAVObservationV1],
    *,
    strategy: ShadowStrategySpecV1,
    inventories: Sequence[TickDataInventoryV1],
) -> dict[str, object]:
    """Summarize frozen shadow NAV without claiming live orders or fills."""

    if not isinstance(strategy, ShadowStrategySpecV1):
        raise MetricContractError(
            "shadow NAV requires a frozen ShadowStrategySpecV1"
        )
    classification = _shadow_inventory_classification(
        inventories,
        strategy=strategy,
    )
    rows = tuple(observations)
    if not rows:
        raise MetricContractError(
            "no NAV observations; MDD and Sharpe are unavailable"
        )
    if any(not isinstance(row, ShadowNAVObservationV1) for row in rows):
        raise MetricContractError(
            "observations must be ShadowNAVObservationV1"
        )
    if any(row.strategy_id != strategy.strategy_id for row in rows):
        raise MetricContractError(
            "all NAV observations must use strategy.strategy_id"
        )
    price_bases = {row.price_basis for row in rows}
    fill_statuses = {row.fill_status for row in rows}
    if len(price_bases) != 1:
        raise MetricContractError(
            "shadow NAV observations must use one price_basis"
        )
    if len(fill_statuses) != 1:
        raise MetricContractError(
            "shadow NAV observations must use one fill_status"
        )
    price_basis = next(iter(price_bases))
    fill_status = next(iter(fill_statuses))
    expected_price_basis, expected_fill_status = _CLASSIFICATION_LABELS[
        classification
    ]
    if (
        price_basis != expected_price_basis
        or fill_status != expected_fill_status
    ):
        raise MetricContractError(
            f"{classification} capability-bound labels require "
            f"price_basis={expected_price_basis} and "
            f"fill_status={expected_fill_status}"
        )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.source_time,
                row.game_id,
                row.episode_id,
            ),
        )
    )
    navs = tuple(row.nav for row in ordered)
    nav_path = (strategy.initial_nav, *navs)
    peak = strategy.initial_nav
    maximum_drawdown = 0.0
    for nav in nav_path:
        peak = max(peak, nav)
        maximum_drawdown = max(
            maximum_drawdown,
            1 - nav / peak,
        )
    returns = tuple(
        current / prior - 1
        for prior, current in zip(nav_path, nav_path[1:])
    )
    if len(returns) < 2:
        sharpe_ratio: float | None = None
    else:
        volatility = stdev(returns)
        sharpe_ratio = (
            None if volatility == 0 else mean(returns) / volatility
        )
    return {
        "strategy_id": strategy.strategy_id,
        "shortlist_manifest_path": str(strategy.shortlist_manifest_path),
        "shortlist_manifest_sha256": strategy.shortlist_manifest_sha256,
        "evidence_classification": classification,
        "execution_status": classification,
        "order_status": "NO_ORDERS_CREATED",
        "executable": False,
        "price_basis": price_basis,
        "fees": sum(row.fees for row in ordered),
        "fee_bps": strategy.fee_bps,
        "fill_status": fill_status,
        "observation_count": len(ordered),
        "initial_nav": strategy.initial_nav,
        "terminal_nav": navs[-1],
        "maximum_drawdown": maximum_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }
