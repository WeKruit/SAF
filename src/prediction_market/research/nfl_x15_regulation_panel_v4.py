"""PIA-grained regulation decision panel for NFL X-15.

Panel V4 has exactly three semantic sources: ProvisionalInformationAnchorV2,
EventPrestateContextV2, and verified actual-trade market data.  It does not
accept FinalizedEpisode tables or Facts V4 directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import time
from typing import Callable, Final, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.research.nfl_x15_decision_context_v2 import (
    DEFAULT_INFORMATION_ANCHOR_MANIFEST,
    DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256,
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_GAME_COUNT,
    _read_information_anchor_publication,
    compute_prestate_context_sha256,
)
from prediction_market.research.nfl_x15_development_panel import (
    DevelopmentPanelError,
    DevelopmentSourceSpec,
    VerifiedDevelopmentSources,
    _adapt_market,
    _read_verified_table,
    _resolve_under,
    _sha256_file,
    _table_descriptor,
    _verify_game_manifest,
    default_development_source_spec,
    verify_development_sources,
)


SCHEMA_VERSION: Final[str] = "RegulationDecisionPanelV4"
ATTRITION_SCHEMA: Final[str] = "RegulationDecisionPanelAttritionV2"
GAME_MANIFEST_SCHEMA: Final[str] = "RegulationDecisionPanelGameManifestV2"
BATCH_MANIFEST_SCHEMA: Final[str] = "RegulationDecisionPanelBatchManifestV2"
BUILDER_VERSION: Final[str] = "nfl-x15-regulation-decision-panel-v4"
EXPERIMENT_ID: Final[str] = "X-15"
DATASET_SPLIT: Final[str] = "DEVELOPMENT"
CLAIM_BOUNDARY: Final[str] = (
    "HISTORICAL_ACTUAL_TRADES_ONLY;PIA_PRIMARY_SELECTION;"
    "REGULATION_SOURCE_TIME_PROBABILITY;NO_LIVE_LATENCY;NO_CAUSALITY;"
    "NO_EXECUTION;NO_ALPHA_CLAIM"
)
ANCHOR_KIND: Final[str] = "PROVISIONAL_FIRST_SEEN"
ANALYSIS_ROLE: Final[str] = "PRIMARY_SELECTION"
DIRECTION_THRESHOLD: Final[float] = 0.01
LANDMARK_SECONDS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
ENDPOINT_SECONDS: Final[tuple[int, ...]] = tuple(range(5, 61, 5))
DEFAULT_CONTEXT_MANIFEST: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/"
    "event-prestate-context-v2/manifests/sha256/61/"
    "61afabaf9cbec7eef5491501ae39fd717bf0e73e5a17e828ac7c4fc8e59a4638"
    ".manifest.json"
)
DEFAULT_CONTEXT_MANIFEST_FILE_SHA256: Final[str] = (
    "sha256:61afabaf9cbec7eef5491501ae39fd717bf0e73e5a17e828ac7c4fc8e59a4638"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/regulation-decision-panel-v4"
)
PEAK_RSS_WARNING_MIB: Final[int] = 6 * 1024
PEAK_RSS_STOP_MIB: Final[int] = 8 * 1024
SYSTEM_FREE_STOP_PERCENT: Final[int] = 10
REALIZED_OUTCOME_COMPARABILITY_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "COMPLETED_NON_TIE_REALIZATION_COMPARABLE",
        "UNPROVEN_NOT_ADJUDICATED",
    }
)
STRICT_CONTRACT_RULE_EQUIVALENCE_STATUSES: Final[frozenset[str]] = frozenset(
    {"CONTRACT_RULE_EQUIVALENT", "UNPROVEN"}
)
DEFAULT_CAPTURE_ROOT: Final[Path] = Path(
    "artifacts/market-observation/nfl/x13/expansion-capture/"
    "moneyline-reaction-closed-v1"
)

PRIMARY_FEATURE_CONTRACT: Final[Mapping[str, object]] = {
    "source_schema": "ProvisionalInformationAnchorV2",
    "join_key": "information_anchor_id",
    "cardinality": "ONE_TO_ONE",
    "source_interval_start_column": "source_interval_start",
    "source_interval_end_column": "source_interval_end",
    "anchor_column": "information_anchor",
    "anchor_semantics": "INFORMATION_ANCHOR_EQUALS_SOURCE_INTERVAL_END",
    "eligibility_column": "primary_selection_eligible",
    "known_at_column": "primary_feature_known_at",
    "support_status_column": "primary_feature_support_status",
    "support_reason_column": "primary_feature_support_reason",
    "snapshot_sha256_column": "primary_feature_snapshot_sha256",
    "allowed_statuses": ("SUPPORTED", "UNPROVEN"),
    "forbidden_feature_prefixes": ("final_", "finalized_"),
}
PRESTATE_CONTEXT_CONTRACT: Final[Mapping[str, object]] = {
    "source_schema": "EventPrestateContextV2",
    "join_key": "information_anchor_id",
    "cardinality": "ONE_TO_ONE",
    "known_at_column": "prestate_known_at",
    "context_sha256_column": "prestate_context_sha256",
    "p_before_support_status_column": "p_before_support_status",
    "known_at_semantics": (
        "SUPPORTED_IMPLIES_NONNULL_LTE_INFORMATION_ANCHOR;"
        "MISSING_OR_UNPROVEN_IMPLIES_NULL"
    ),
    "post_or_final_fields_allowed": False,
}
MARKET_CONTINUITY_CONTRACT: Final[Mapping[str, object]] = {
    "authority": (
        "SHA256_VERIFIED_CAPTURE_BATCH_AND_PER_GAME_CAPTURE_INDEX"
    ),
    "kalshi_terminal_proof": "EXPLICIT_EMPTY_CURSOR_PER_TICKER",
    "polymarket_terminal_proof": "UNSATURATED_TIME_WINDOW",
    "coverage": "ANCHOR_AND_ENDPOINT_WITHIN_CAPTURE_START_END",
    "meaning": (
        "HISTORICAL_TRADE_PAGINATION_COMPLETE_WITHIN_CAPTURE_WINDOW;"
        "NOT_L2_CONTINUITY"
    ),
}

_D1: Final[tuple[str, ...]] = (
    "landmark_seconds",
    "endpoint_seconds",
    "mark_l_price",
    "mark_l_staleness_seconds",
    "prior_30s_actual_trade_count",
    "prior_30s_actual_trade_size",
    "prior_60s_actual_trade_count",
    "prior_60s_actual_trade_size",
)
_D2: Final[tuple[str, ...]] = (
    *_D1,
    "quarter",
    "is_overtime",
    "game_seconds_remaining",
    "score_margin_home",
    "possession_is_home",
    "possession_is_home_missing",
    "down",
    "down_missing",
    "distance",
    "distance_missing",
    "yardline_100",
    "yardline_100_missing",
    "goal_to_go",
    "pre_red_zone",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "p_before_home",
    "p_before_home_missing",
)
FEATURE_BLOCKS: Final[Mapping[str, tuple[str, ...]]] = {
    "D0": ("landmark_seconds", "endpoint_seconds"),
    "D1": _D1,
    "D2": _D2,
    "D3": (
        *_D2,
        "action_group",
        "possession_result_group",
        "score_result_group",
        "kick_result_group",
        "adjudication_group",
        "provisional_yards_gained",
        "provisional_return_yards",
        "provisional_score_points_observed",
        "provisional_turnover_observed",
        "actor_is_home",
        "beneficiary_is_home",
    ),
}
FORMAL_MODEL_REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        (
            "schema_version",
            "dataset_split",
            "holdout_reaction_accessed",
            "cohort_authority_sha256",
            "source_row_id",
            "game_id",
            "nfl_week",
            "information_anchor_id",
            "episode_id",
            "anchor_kind",
            "analysis_role",
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "primary_feature_known_at",
            "primary_feature_support_status",
            "primary_feature_support_reason",
            "primary_feature_snapshot_sha256",
            "prestate_known_at",
            "prestate_context_sha256",
            "p_before_support_status",
            "venue",
            "actual_home_contract_id",
            "logical_market_id",
            "market_family",
            "market_period",
            "proposition_kind",
            "outcome_team",
            "home_team",
            "actual_home_contract_identity_sha256",
            "native_raw_manifest_sha256s_json",
            "native_raw_object_sha256s_json",
            "native_raw_manifest_set_sha256",
            "native_raw_object_set_sha256",
            "native_rule_evidence_sha256",
            "native_rule_evidence_status",
            "realized_outcome_comparability_status",
            "realized_outcome_comparability_evidence_sha256",
            "strict_contract_rule_equivalence_status",
            "strict_contract_rule_equivalence_evidence_sha256",
            "landmark_seconds",
            "endpoint_seconds",
            "primary_selection_eligible",
            "landmark_eligible",
            "continuity_valid",
            "order_ambiguous",
            "censor_boundary_eligible",
            "censored",
            "censor_reason",
            "loss_eligible",
            "availability_h",
            "direction_h",
            "direction_eligible",
            "direction_threshold_probability",
            "is_overtime",
            *FEATURE_BLOCKS["D3"],
        )
    )
)
FORMAL_MODEL_FORBIDDEN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "decision_eligible",
        "finalized_episode_id",
        "episode_type",
        "episode_status",
        "status",
        "stage_a_eligible",
        "residual_diagnostic_eligible",
        "s_h",
        "s_h_truth",
        "s_h_probability",
        "o_h_given_s",
        "o_h_given_s_truth",
        "o_h_given_s_probability",
    }
)
_SHA_PREFIX: Final[str] = "sha256:"
_REQUIRED_SOURCE_HASH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "information_anchor_manifest_sha256",
        "information_anchor_object_sha256",
        "context_manifest_sha256",
        "context_object_sha256",
        "market_game_manifest_sha256",
        "market_observations_object_sha256",
        "market_inventory_object_sha256",
        "market_capture_batch_sha256",
        "market_capture_index_sha256",
        "market_capture_checkpoint_sha256",
        "cohort_authority_sha256",
    }
)
_CONTEXT_FORBIDDEN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "p_after_home",
        "reference_delta_home",
        "bridge_delta_home",
        "reference_gap",
    }
)


class RegulationPanelV4Error(RuntimeError):
    """A PIA, context, time, target, or source invariant failed."""


@dataclass(frozen=True, slots=True)
class RegulationAnchorGamePanel:
    panel: pd.DataFrame
    attrition: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RegulationPanelV4SourceSpec:
    development: DevelopmentSourceSpec
    information_anchor_manifest_path: Path = (
        DEFAULT_INFORMATION_ANCHOR_MANIFEST
    )
    information_anchor_manifest_file_sha256: str = (
        DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256
    )
    context_manifest_path: Path = DEFAULT_CONTEXT_MANIFEST
    context_manifest_file_sha256: str = (
        DEFAULT_CONTEXT_MANIFEST_FILE_SHA256
    )
    expected_game_count: int = EXPECTED_GAME_COUNT
    expected_anchor_count: int = EXPECTED_ANCHOR_COUNT


@dataclass(frozen=True, slots=True)
class VerifiedRegulationPanelV4Sources:
    development: VerifiedDevelopmentSources
    information_anchors: pd.DataFrame
    context: pd.DataFrame
    information_anchor_manifest: Mapping[str, object]
    context_manifest: Mapping[str, object]
    information_anchor_manifest_file_sha256: str
    information_anchor_object_sha256: str
    context_manifest_file_sha256: str
    context_object_sha256: str


@dataclass(frozen=True, slots=True)
class PublishedRegulationPanelV4:
    output_root: Path
    batch_manifest_path: Path
    batch_manifest_sha256: str
    batch_sha256: str
    game_count: int
    primary_information_anchor_count: int
    panel_row_count: int
    landmark_eligible_count: int
    loss_eligible_count: int
    direction_eligible_count: int
    censored_count: int
    order_ambiguous_count: int
    availability_zero_count: int
    availability_one_count: int
    runtime_seconds: float
    peak_rss_mib: float


@dataclass(frozen=True, slots=True)
class _TradeIndex:
    times_ns: np.ndarray
    prices: np.ndarray
    sizes: np.ndarray
    trade_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class _Mark:
    status: str
    source_time: pd.Timestamp | None
    price: float
    trade_ids: tuple[str, ...]
    trade_id_set_sha256: str | None
    observation_count: int
    observed_size: float
    staleness_seconds: float

    @property
    def observed(self) -> bool:
        return self.status == "OBSERVED"


def _sha256_bytes(payload: bytes) -> str:
    return _SHA_PREFIX + hashlib.sha256(payload).hexdigest()


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return [
            _canonical_value(child)
            for child in sorted(value, key=repr)
        ]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise RegulationPanelV4Error(
        f"unsupported canonical value {type(value).__name__}"
    )


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _manifest_semantic_sha256(value: object) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_SHA_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RegulationPanelV4Error(f"{label} is not a canonical SHA-256")
    return value


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RegulationPanelV4Error(
            f"{label} missing columns: {', '.join(missing)}"
        )


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _nullable_bool(value: object) -> bool | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _tags(value: object) -> set[str]:
    if value is None or value is pd.NA:
        return set()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RegulationPanelV4Error(
                "provisional outcome tags are invalid JSON"
            ) from exc
    elif isinstance(value, (list, tuple, set)):
        decoded = list(value)
    else:
        raise RegulationPanelV4Error(
            "provisional outcome tags have unsupported type"
        )
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise RegulationPanelV4Error(
            "provisional outcome tags must be a string list"
        )
    return {item.strip().upper() for item in decoded if item.strip()}


def _trade_index(frame: pd.DataFrame) -> _TradeIndex:
    if frame.empty:
        return _TradeIndex(
            times_ns=np.array([], dtype=np.int64),
            prices=np.array([], dtype=float),
            sizes=np.array([], dtype=float),
            trade_ids=np.array([], dtype=object),
        )
    work = frame.copy()
    work["source_time_utc"] = pd.to_datetime(
        work["source_time_utc"], utc=True, errors="raise", format="mixed"
    )
    work["price"] = pd.to_numeric(work["price"], errors="raise")
    work["size"] = pd.to_numeric(work["size"], errors="raise")
    if (
        (~np.isfinite(work["price"])).any()
        or (~np.isfinite(work["size"])).any()
        or work["price"].lt(0).any()
        or work["price"].gt(1).any()
        or work["size"].le(0).any()
        or work["trade_id"].astype(str).duplicated().any()
    ):
        raise RegulationPanelV4Error(
            "market actual trades violate price/size/ID contract"
        )
    work = work.sort_values(
        ["source_time_utc", "trade_id"], kind="mergesort"
    ).reset_index(drop=True)
    return _TradeIndex(
        times_ns=np.fromiter(
            (pd.Timestamp(value).value for value in work["source_time_utc"]),
            dtype=np.int64,
            count=len(work),
        ),
        prices=work["price"].astype(float).to_numpy(),
        sizes=work["size"].astype(float).to_numpy(),
        trade_ids=work["trade_id"].astype(str).to_numpy(dtype=object),
    )


def _bucket_mark(
    trades: _TradeIndex,
    *,
    lower_exclusive_ns: int,
    upper_inclusive_ns: int,
    staleness_origin_ns: int,
) -> _Mark:
    lower = int(
        np.searchsorted(trades.times_ns, lower_exclusive_ns, side="right")
    )
    upper = int(
        np.searchsorted(trades.times_ns, upper_inclusive_ns, side="right")
    )
    if lower >= upper:
        return _Mark(
            status="NO_ACTUAL_TRADE",
            source_time=None,
            price=math.nan,
            trade_ids=(),
            trade_id_set_sha256=None,
            observation_count=0,
            observed_size=0.0,
            staleness_seconds=math.nan,
        )
    latest_ns = int(trades.times_ns[upper - 1])
    bucket_start = max(
        lower,
        int(np.searchsorted(trades.times_ns, latest_ns, side="left")),
    )
    bucket_end = int(
        np.searchsorted(trades.times_ns, latest_ns, side="right")
    )
    prices = trades.prices[bucket_start:bucket_end]
    sizes = trades.sizes[bucket_start:bucket_end]
    ids = tuple(sorted(map(str, trades.trade_ids[bucket_start:bucket_end])))
    total_size = float(sizes.sum())
    if (
        not ids
        or total_size <= 0
        or not np.isfinite(prices).all()
        or not np.isfinite(sizes).all()
    ):
        raise RegulationPanelV4Error(
            "same-time actual-trade bucket is invalid"
        )
    return _Mark(
        status="OBSERVED",
        source_time=pd.Timestamp(latest_ns, tz="UTC"),
        price=float(np.dot(prices, sizes) / total_size),
        trade_ids=ids,
        trade_id_set_sha256=_canonical_sha256(list(ids)),
        observation_count=len(ids),
        observed_size=total_size,
        staleness_seconds=float(
            (staleness_origin_ns - latest_ns) / 1_000_000_000
        ),
    )


def _activity(
    trades: _TradeIndex,
    *,
    decision_ns: int,
    seconds: int,
) -> tuple[int, float]:
    lower = decision_ns - seconds * 1_000_000_000
    first = int(np.searchsorted(trades.times_ns, lower, side="right"))
    after = int(np.searchsorted(trades.times_ns, decision_ns, side="right"))
    return after - first, float(trades.sizes[first:after].sum())


def _semantic_axes(anchor: Mapping[str, object]) -> dict[str, str]:
    tags = _tags(anchor["provisional_outcome_tags"])
    action = str(
        anchor.get("provisional_primary_action") or "UNKNOWN"
    ).upper()
    action_group = (
        action
        if action
        in {
            "PASS",
            "RUN",
            "SACK",
            "PUNT",
            "KICKOFF",
            "FIELD_GOAL",
            "TRY",
            "KNEEL",
            "TIMEOUT",
        }
        else "OTHER"
    )
    if "INTERCEPTION" in tags:
        possession = "INTERCEPTION"
    elif "LOST_FUMBLE" in tags:
        possession = "LOST_FUMBLE"
    elif "TURNOVER_ON_DOWNS" in tags:
        possession = "TURNOVER_ON_DOWNS"
    elif "MUFFED_PUNT" in tags:
        possession = "MUFFED_PUNT"
    elif "MUFFED_KICKOFF" in tags:
        possession = "MUFFED_KICKOFF"
    elif "POSSESSION_CHANGE" in tags:
        possession = "OTHER_POSSESSION_CHANGE"
    else:
        possession = "POSSESSION_RETAINED"

    if "PASS_TOUCHDOWN" in tags:
        score = "PASS_TOUCHDOWN"
    elif "RUSH_TOUCHDOWN" in tags:
        score = "RUSH_TOUCHDOWN"
    elif "RETURN_TOUCHDOWN" in tags:
        score = "RETURN_TOUCHDOWN"
    elif "DEFENSIVE_TWO_POINT" in tags:
        score = "DEFENSIVE_TWO_POINT"
    elif "SAFETY" in tags:
        score = "SAFETY"
    elif "FIELD_GOAL_MADE" in tags:
        score = "FIELD_GOAL"
    elif "EXTRA_POINT_GOOD" in tags:
        score = "EXTRA_POINT"
    elif "TWO_POINT_SUCCESS" in tags:
        score = "TWO_POINT"
    elif int(anchor.get("provisional_score_points_observed") or 0) > 0:
        score = "OTHER_SCORE"
    else:
        score = "NO_SCORE"

    kick_related = action in {"PUNT", "KICKOFF", "FIELD_GOAL", "TRY"}
    if "BLOCKED_KICK" in tags or any(tag.startswith("BLOCKED_") for tag in tags):
        kick = "BLOCKED"
    elif "ONSIDE_RECOVERY" in tags or "ONSIDE_KICK_RECOVERED" in tags:
        kick = "ONSIDE_RECOVERED"
    elif "KICKOFF_TOUCHBACK" in tags or "PUNT_TOUCHBACK" in tags:
        kick = "TOUCHBACK"
    elif "PUNT_FAIR_CATCH" in tags:
        kick = "FAIR_CATCH"
    elif "PUNT_INSIDE_20" in tags:
        kick = "INSIDE_20"
    elif "RETURN_TOUCHDOWN" in tags and kick_related:
        kick = "RETURN_TOUCHDOWN"
    elif any("RETURN" in tag for tag in tags) and kick_related:
        kick = "RETURN"
    elif "FIELD_GOAL_MADE" in tags or "EXTRA_POINT_GOOD" in tags:
        kick = "MADE"
    elif {
        "FIELD_GOAL_MISSED",
        "EXTRA_POINT_MISSED",
        "TWO_POINT_FAILED",
    }.intersection(tags):
        kick = "MISSED_OR_FAILED"
    elif kick_related:
        kick = "OTHER_KICK"
    else:
        kick = "NO_KICK"

    if "REVERSAL" in tags:
        adjudication = "REVIEW_REVERSED"
    elif "REVIEW" in tags:
        adjudication = "REVIEW_UNPROVEN_OR_PENDING"
    elif "PENALTY_ACCEPTED_NO_PLAY" in tags:
        adjudication = "PENALTY_ACCEPTED_NO_PLAY"
    elif "PENALTY_ACCEPTED" in tags:
        adjudication = "PENALTY_ACCEPTED"
    elif "PENALTY_OFFSETTING" in tags:
        adjudication = "PENALTY_OFFSETTING"
    elif "PENALTY_DECLINED" in tags:
        adjudication = "PENALTY_DECLINED"
    elif "PENALTY" in tags:
        adjudication = "PENALTY_OTHER"
    else:
        adjudication = "NO_ADJUDICATION"
    return {
        "action_group": action_group,
        "possession_result_group": possession,
        "score_result_group": score,
        "kick_result_group": kick,
        "adjudication_group": adjudication,
    }


def _direction(delta: float) -> str:
    if delta >= DIRECTION_THRESHOLD:
        return "UP"
    if delta <= -DIRECTION_THRESHOLD:
        return "DOWN"
    return "NO_MOVE"


def _team_orientation(team: object, home_team: str) -> bool | None:
    if team is None or team is pd.NA:
        return None
    value = str(team).strip()
    if not value:
        return None
    return value == home_team


def _primary_support_status(value: object) -> str:
    status = str(value)
    if status == "SUPPORTED":
        return "SUPPORTED"
    if status in {
        "PRIMARY_FEATURE_SUPPORT_UNPROVEN",
        "MODEL_SUPPORT_UNPROVEN",
        "UNPROVEN",
    }:
        return "UNPROVEN"
    raise RegulationPanelV4Error(
        f"unknown primary feature support status {status!r}"
    )


def _p_before_support_status(value: object) -> str:
    status = str(value)
    if status == "SUPPORTED":
        return "SUPPORTED"
    if status in {"MISSING_PRE_STATE", "MISSING"}:
        return "MISSING"
    if status in {"MODEL_SUPPORT_UNPROVEN", "UNPROVEN"}:
        return "UNPROVEN"
    raise RegulationPanelV4Error(
        f"unknown p_before support status {status!r}"
    )


def build_regulation_anchor_game_panel(
    *,
    information_anchors: pd.DataFrame,
    context: pd.DataFrame,
    market_rows: pd.DataFrame,
    contracts: pd.DataFrame,
    cohort_row: Mapping[str, object],
    source_hashes: Mapping[str, str],
    landmark_seconds: Sequence[int] = LANDMARK_SECONDS,
    endpoint_seconds: Sequence[int] = ENDPOINT_SECONDS,
) -> RegulationAnchorGamePanel:
    """Build one game's Panel V4 rows without publication or network access."""

    _require_columns(
        information_anchors,
        {
            "schema_version",
            "game_id",
            "information_anchor_id",
            "episode_id",
            "event_id",
            "raw_play_id",
            "order_sequence",
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "provisional_known_at",
            "provisional_primary_event_id",
            "provisional_primary_action",
            "provisional_outcome_tags",
            "provisional_actor_team",
            "provisional_beneficiary_team",
            "provisional_score_points_observed",
            "provisional_turnover_observed",
            "provisional_yards_gained",
            "provisional_return_yards",
            "provisional_feature_support_status",
            "provisional_feature_support_reason",
            "provisional_snapshot_sha256",
            "canonical_stage_b_information_event_eligible",
            "primary_selection_eligible",
            "censor_boundary_eligible",
            "pbp_source_sha256",
        },
        label="ProvisionalInformationAnchorV2",
    )
    _require_columns(
        context,
        {
            "schema_version",
            "game_id",
            "information_anchor_id",
            "parent_episode_id_audit_only",
            "event_id",
            "raw_play_id",
            "order_sequence",
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "primary_selection_eligible",
            "censor_boundary_eligible",
            "quarter",
            "is_overtime",
            "game_clock",
            "game_seconds_remaining",
            "home_team",
            "away_team",
            "possession_is_home",
            "pre_home_score",
            "pre_away_score",
            "score_margin_home",
            "down",
            "distance",
            "yardline_100",
            "goal_to_go",
            "pre_red_zone",
            "home_timeouts_remaining",
            "away_timeouts_remaining",
            "p_before_home",
            "p_before_home_missing",
            "prestate_support_status",
            "pre_state_known_at",
            "prestate_context_sha256",
            "source_binding_sha256",
        },
        label="EventPrestateContextV2",
    )
    _require_columns(
        market_rows,
        {
            "trade_id",
            "game_id",
            "venue",
            "contract_id",
            "source_time_utc",
            "price",
            "size",
        },
        label="actual market trades",
    )
    _require_columns(
        contracts,
        {
            "game_id",
            "venue",
            "contract_id",
            "contract_role",
            "logical_market_id",
            "market_family",
            "market_period",
            "proposition_kind",
            "outcome_team",
            "actual_home_contract_identity_sha256",
            "native_raw_manifest_sha256s_json",
            "native_raw_object_sha256s_json",
            "native_raw_manifest_set_sha256",
            "native_raw_object_set_sha256",
            "native_rule_evidence_sha256",
            "native_rule_evidence_status",
            "realized_outcome_comparability_status",
            "realized_outcome_comparability_evidence_sha256",
            "strict_contract_rule_equivalence_status",
            "strict_contract_rule_equivalence_evidence_sha256",
            "market_continuity_start_utc",
            "market_continuity_end_utc",
            "market_continuity_support_status",
            "market_continuity_evidence_sha256",
        },
        label="market contract inventory",
    )
    forbidden_context = set(
        _CONTEXT_FORBIDDEN_COLUMNS.intersection(context.columns)
    )
    forbidden_context.update(
        column
        for column in context.columns
        if column.startswith(("post_", "final_"))
    )
    if forbidden_context:
        raise RegulationPanelV4Error(
            f"context contains forbidden future/final columns: "
            f"{sorted(forbidden_context)}"
        )
    for row in context.to_dict("records"):
        if str(row["prestate_context_sha256"]) != (
            compute_prestate_context_sha256(row)
        ):
            raise RegulationPanelV4Error(
                "ContextV2 row feature hash mismatch"
            )
    if information_anchors["schema_version"].astype(str).ne(
        "ProvisionalInformationAnchorV2"
    ).any() or context["schema_version"].astype(str).ne(
        "EventPrestateContextV2"
    ).any():
        raise RegulationPanelV4Error("formal source schema mismatch")
    missing_hashes = _REQUIRED_SOURCE_HASH_KEYS.difference(source_hashes)
    if missing_hashes:
        raise RegulationPanelV4Error(
            f"source hashes missing keys: {sorted(missing_hashes)}"
        )
    normalized_hashes = {
        key: _require_sha256(value, label=f"source_hashes.{key}")
        for key, value in sorted(source_hashes.items())
    }

    game_sets = (
        set(information_anchors["game_id"].astype(str)),
        set(context["game_id"].astype(str)),
        set(contracts["game_id"].astype(str)),
    )
    if any(len(values) != 1 for values in game_sets) or not (
        game_sets[0] == game_sets[1] == game_sets[2]
    ):
        raise RegulationPanelV4Error("single-game inputs crossed game boundary")
    game_id = next(iter(game_sets[0]))
    if str(cohort_row["game_id"]) != game_id:
        raise RegulationPanelV4Error("cohort row game mismatch")
    if (
        information_anchors["information_anchor_id"].astype(str).duplicated().any()
        or information_anchors["event_id"].astype(str).duplicated().any()
        or context["information_anchor_id"].astype(str).duplicated().any()
    ):
        raise RegulationPanelV4Error("PIA/context identity is not unique")

    l_values = tuple(sorted(set(map(int, landmark_seconds))))
    h_values = tuple(sorted(set(map(int, endpoint_seconds))))
    if (
        not l_values
        or not h_values
        or any(value <= 0 for value in (*l_values, *h_values))
    ):
        raise RegulationPanelV4Error(
            "time grid must contain positive integer seconds"
        )

    anchors = information_anchors.copy()
    for column in (
        "source_interval_start",
        "source_interval_end",
        "information_anchor",
        "provisional_known_at",
    ):
        anchors[column] = pd.to_datetime(
            anchors[column], utc=True, errors="coerce", format="mixed"
        )
    if (
        anchors[
            ["source_interval_start", "source_interval_end", "information_anchor"]
        ]
        .isna()
        .any()
        .any()
        or not anchors["information_anchor"].eq(
            anchors["source_interval_end"]
        ).all()
        or (anchors["source_interval_start"] > anchors["source_interval_end"]).any()
    ):
        raise RegulationPanelV4Error("PIA source interval contract is invalid")
    if anchors["order_sequence"].astype("Int64").duplicated().any():
        raise RegulationPanelV4Error("PIA order sequence is not unique")

    context_work = context.copy()
    for column in (
        "source_interval_start",
        "source_interval_end",
        "information_anchor",
        "pre_state_known_at",
    ):
        context_work[column] = pd.to_datetime(
            context_work[column],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    anchor_identity = anchors[
        [
            "game_id",
            "information_anchor_id",
            "event_id",
            "raw_play_id",
            "order_sequence",
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "primary_selection_eligible",
            "censor_boundary_eligible",
        ]
    ].merge(
        context_work[
            [
                "game_id",
                "information_anchor_id",
                "event_id",
                "raw_play_id",
                "order_sequence",
                "source_interval_start",
                "source_interval_end",
                "information_anchor",
                "primary_selection_eligible",
                "censor_boundary_eligible",
            ]
        ],
        on=["game_id", "information_anchor_id"],
        how="outer",
        suffixes=("_pia", "_context"),
        validate="one_to_one",
        indicator=True,
    )
    compared = (
        "event_id",
        "raw_play_id",
        "order_sequence",
        "source_interval_start",
        "source_interval_end",
        "information_anchor",
        "primary_selection_eligible",
        "censor_boundary_eligible",
    )
    if not anchor_identity["_merge"].eq("both").all() or any(
        not anchor_identity[f"{column}_pia"].astype("string").eq(
            anchor_identity[f"{column}_context"].astype("string")
        ).all()
        for column in compared
    ):
        raise RegulationPanelV4Error("PIA and ContextV2 identity mismatch")

    context_index = {
        str(row["information_anchor_id"]): row
        for row in context_work.to_dict("records")
    }
    boundaries = anchors.loc[
        anchors["censor_boundary_eligible"].eq(True)  # noqa: E712
    ].sort_values(["order_sequence", "information_anchor_id"], kind="mergesort")
    boundary_records = boundaries.to_dict("records")
    next_boundary: dict[str, Mapping[str, object] | None] = {}
    for index, boundary in enumerate(boundary_records):
        next_boundary[str(boundary["information_anchor_id"])] = (
            boundary_records[index + 1]
            if index + 1 < len(boundary_records)
            else None
        )

    focal = anchors.loc[
        anchors["primary_selection_eligible"].eq(True)  # noqa: E712
    ].sort_values(["order_sequence", "information_anchor_id"], kind="mergesort")
    if focal.empty:
        raise RegulationPanelV4Error("game has no primary-selection PIA")
    for row in focal.to_dict("records"):
        known_at = pd.Timestamp(row["provisional_known_at"])
        if (
            str(row["provisional_feature_support_status"]) != "SUPPORTED"
            or pd.isna(known_at)
            or known_at > pd.Timestamp(row["information_anchor"])
            or not bool(
                row["canonical_stage_b_information_event_eligible"]
            )
        ):
            raise RegulationPanelV4Error(
                "primary PIA violates provisional feature contract"
            )

    home_contracts = contracts.loc[
        contracts["contract_role"].astype(str).eq("ACTUAL_HOME_OUTCOME")
    ].copy()
    if (
        home_contracts.empty
        or home_contracts.duplicated(["venue", "contract_id"]).any()
        or home_contracts["market_family"].astype(str).ne("moneyline").any()
        or home_contracts["market_period"].astype(str).ne("full_game").any()
        or home_contracts["proposition_kind"].astype(str).ne("primitive").any()
    ):
        raise RegulationPanelV4Error(
            "actual home contract identity is invalid"
        )
    for contract in home_contracts.to_dict("records"):
        for column in (
            "actual_home_contract_identity_sha256",
            "native_raw_manifest_set_sha256",
            "native_raw_object_set_sha256",
            "market_continuity_evidence_sha256",
        ):
            _require_sha256(contract[column], label=f"contract.{column}")
        try:
            raw_manifests = json.loads(
                str(contract["native_raw_manifest_sha256s_json"])
            )
            raw_objects = json.loads(
                str(contract["native_raw_object_sha256s_json"])
            )
        except json.JSONDecodeError as exc:
            raise RegulationPanelV4Error(
                "native raw lineage JSON is invalid"
            ) from exc
        if (
            not isinstance(raw_manifests, list)
            or not isinstance(raw_objects, list)
            or not raw_manifests
            or not raw_objects
            or any(
                _require_sha256(value, label="native raw lineage")
                != value
                for value in (*raw_manifests, *raw_objects)
            )
            or _canonical_sha256(sorted(raw_manifests))
            != contract["native_raw_manifest_set_sha256"]
            or _canonical_sha256(sorted(raw_objects))
            != contract["native_raw_object_set_sha256"]
        ):
            raise RegulationPanelV4Error(
                "native raw lineage set/hash mismatch"
            )
        native_rule_sha = contract["native_rule_evidence_sha256"]
        if pd.notna(native_rule_sha):
            _require_sha256(
                native_rule_sha,
                label="contract.native_rule_evidence_sha256",
            )
        if (
            str(contract["native_rule_evidence_status"])
            == "VERIFIED_NATIVE_RULE"
        ) != pd.notna(native_rule_sha):
            raise RegulationPanelV4Error(
                "native rule evidence status/hash mismatch"
            )
        for (
            status_column,
            evidence_column,
            positive_status,
            allowed_statuses,
        ) in (
            (
                "realized_outcome_comparability_status",
                "realized_outcome_comparability_evidence_sha256",
                "COMPLETED_NON_TIE_REALIZATION_COMPARABLE",
                REALIZED_OUTCOME_COMPARABILITY_STATUSES,
            ),
            (
                "strict_contract_rule_equivalence_status",
                "strict_contract_rule_equivalence_evidence_sha256",
                "CONTRACT_RULE_EQUIVALENT",
                STRICT_CONTRACT_RULE_EQUIVALENCE_STATUSES,
            ),
        ):
            status = str(contract[status_column])
            evidence = contract[evidence_column]
            if pd.notna(evidence):
                _require_sha256(evidence, label=f"contract.{evidence_column}")
            if (
                status not in allowed_statuses
                or (status == positive_status) != pd.notna(evidence)
            ):
                raise RegulationPanelV4Error(
                    f"{status_column} status/hash mismatch"
                )
        continuity_start = pd.to_datetime(
            contract["market_continuity_start_utc"],
            utc=True,
            errors="coerce",
        )
        continuity_end = pd.to_datetime(
            contract["market_continuity_end_utc"],
            utc=True,
            errors="coerce",
        )
        if (
            str(contract["market_continuity_support_status"])
            != "VERIFIED_CAPTURE_PAGINATION"
            or pd.isna(continuity_start)
            or pd.isna(continuity_end)
            or continuity_start >= continuity_end
        ):
            raise RegulationPanelV4Error(
                "market capture/pagination continuity is unverified"
            )
    trade_indices = {
        (str(venue), str(contract_id)): _trade_index(group)
        for (venue, contract_id), group in market_rows.groupby(
            ["venue", "contract_id"], sort=False
        )
    }
    empty_index = _trade_index(market_rows.iloc[0:0])
    source_hashes_json = json.dumps(
        normalized_hashes, sort_keys=True, separators=(",", ":")
    )
    source_binding_sha256 = _canonical_sha256(normalized_hashes)

    rows: list[dict[str, object]] = []
    for anchor in focal.to_dict("records"):
        anchor_id = str(anchor["information_anchor_id"])
        prestate = context_index[anchor_id]
        anchor_time = pd.Timestamp(anchor["information_anchor"])
        primary_support_status = _primary_support_status(
            anchor["provisional_feature_support_status"]
        )
        p_before_support_status = _p_before_support_status(
            prestate["prestate_support_status"]
        )
        raw_prestate_known = pd.to_datetime(
            prestate["pre_state_known_at"],
            utc=True,
            errors="coerce",
        )
        prestate_known = (
            pd.Timestamp(raw_prestate_known)
            if p_before_support_status == "SUPPORTED"
            else pd.NaT
        )
        if (
            bool(prestate["is_overtime"])
            or int(prestate["quarter"]) > 4
            or (
                p_before_support_status == "SUPPORTED"
                and (
                    pd.isna(prestate_known)
                    or prestate_known > anchor_time
                )
            )
            or (
                p_before_support_status != "SUPPORTED"
                and pd.notna(prestate_known)
            )
        ):
            raise RegulationPanelV4Error(
                "overtime/future prestate leaked into Panel V4"
            )
        p_before_missing = bool(prestate["p_before_home_missing"])
        p_before = _finite_or_nan(prestate["p_before_home"])
        if not p_before_missing and (
            not math.isfinite(p_before) or not 0 <= p_before <= 1
        ):
            raise RegulationPanelV4Error("supported p_before_home is invalid")
        if p_before_missing:
            p_before = math.nan
        if (p_before_support_status == "SUPPORTED") == p_before_missing:
            raise RegulationPanelV4Error(
                "p_before support status/missing indicator mismatch"
            )
        axes = _semantic_axes(anchor)
        next_row = next_boundary.get(anchor_id)
        next_start = (
            pd.Timestamp(next_row["source_interval_start"])
            if next_row is not None
            else None
        )
        order_ambiguous = bool(
            next_start is not None and next_start <= anchor_time
        )
        home_team = str(prestate["home_team"])

        for contract in home_contracts.to_dict("records"):
            native_venue = str(contract["venue"])
            venue = native_venue.upper()
            contract_id = str(contract["contract_id"])
            if str(contract["outcome_team"]) != home_team:
                raise RegulationPanelV4Error(
                    "actual home contract outcome differs from home team"
                )
            trades = trade_indices.get(
                (native_venue, contract_id), empty_index
            )
            continuity_start = pd.Timestamp(
                contract["market_continuity_start_utc"]
            )
            continuity_end = pd.Timestamp(
                contract["market_continuity_end_utc"]
            )
            anchor_ns = anchor_time.value
            for l_seconds in l_values:
                landmark = anchor_time + pd.Timedelta(seconds=l_seconds)
                landmark_ns = landmark.value
                mark_l = _bucket_mark(
                    trades,
                    lower_exclusive_ns=anchor_ns,
                    upper_inclusive_ns=landmark_ns,
                    staleness_origin_ns=landmark_ns,
                )
                count_30, size_30 = _activity(
                    trades, decision_ns=landmark_ns, seconds=30
                )
                count_60, size_60 = _activity(
                    trades, decision_ns=landmark_ns, seconds=60
                )
                for h_seconds in h_values:
                    if h_seconds <= l_seconds:
                        continue
                    endpoint = anchor_time + pd.Timedelta(seconds=h_seconds)
                    endpoint_ns = endpoint.value
                    if order_ambiguous:
                        censored = True
                        censor_reason = (
                            "NEXT_INFORMATION_INTERVAL_OVERLAPS_FOCAL"
                        )
                        censor_time = next_start
                    elif next_start is None:
                        censored = True
                        censor_reason = (
                            "CONTINUITY_BOUND_UNKNOWN_AFTER_FINAL_ANCHOR"
                        )
                        censor_time = pd.NaT
                    elif next_start <= endpoint:
                        censored = True
                        censor_time = next_start
                        censor_reason = (
                            "NEXT_INFORMATION_INTERVAL_AT_OR_BEFORE_L"
                            if next_start <= landmark
                            else "NEXT_INFORMATION_INTERVAL_IN_L_H_WINDOW"
                        )
                    else:
                        censored = False
                        censor_time = next_start
                        censor_reason = "NOT_CENSORED"
                    mark_h = _bucket_mark(
                        trades,
                        lower_exclusive_ns=landmark_ns,
                        upper_inclusive_ns=endpoint_ns,
                        staleness_origin_ns=endpoint_ns,
                    )
                    landmark_eligible = bool(mark_l.observed)
                    continuity_valid = bool(
                        continuity_start <= anchor_time
                        and endpoint <= continuity_end
                    )
                    loss_eligible = bool(
                        bool(anchor["primary_selection_eligible"])
                        and landmark_eligible
                        and not censored
                        and continuity_valid
                        and not order_ambiguous
                        and primary_support_status == "SUPPORTED"
                    )
                    availability: int | None = (
                        int(mark_h.observed) if loss_eligible else None
                    )
                    delta = math.nan
                    direction: str | None = None
                    if loss_eligible and mark_h.observed:
                        delta = mark_h.price - mark_l.price
                        direction = _direction(delta)
                    if censored:
                        attrition_reason = censor_reason
                    elif not continuity_valid:
                        attrition_reason = (
                            "CAPTURE_PAGINATION_CONTINUITY_INVALID"
                        )
                    elif primary_support_status != "SUPPORTED":
                        attrition_reason = (
                            "PRIMARY_FEATURE_SUPPORT_UNPROVEN"
                        )
                    elif not mark_l.observed:
                        attrition_reason = "LANDMARK_NO_ACTUAL_TRADE"
                    elif mark_h.observed:
                        attrition_reason = "ELIGIBLE_AVAILABILITY_ONE"
                    else:
                        attrition_reason = "ELIGIBLE_AVAILABILITY_ZERO"
                    identity = {
                        "schema_version": SCHEMA_VERSION,
                        "game_id": game_id,
                        "information_anchor_id": anchor_id,
                        "venue": venue,
                        "actual_home_contract_id": contract_id,
                        "landmark_seconds": l_seconds,
                        "endpoint_seconds": h_seconds,
                        "source_binding_sha256": source_binding_sha256,
                        "primary_feature_snapshot_sha256": anchor[
                            "provisional_snapshot_sha256"
                        ],
                        "prestate_context_sha256": prestate[
                            "prestate_context_sha256"
                        ],
                        "actual_home_contract_identity_sha256": contract[
                            "actual_home_contract_identity_sha256"
                        ],
                    }
                    possession = _nullable_bool(
                        prestate["possession_is_home"]
                    )
                    down = _finite_or_nan(prestate["down"])
                    distance = _finite_or_nan(prestate["distance"])
                    yardline = _finite_or_nan(prestate["yardline_100"])
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "claim_boundary": CLAIM_BOUNDARY,
                            "experiment_id": EXPERIMENT_ID,
                            "dataset_split": DATASET_SPLIT,
                            "holdout_reaction_accessed": False,
                            "anchor_kind": ANCHOR_KIND,
                            "analysis_role": ANALYSIS_ROLE,
                            "source_row_id": _canonical_sha256(identity),
                            "source_binding_sha256": source_binding_sha256,
                            "source_hashes_json": source_hashes_json,
                            "cohort_authority_sha256": str(
                                cohort_row["authority_sha256"]
                            ),
                            "game_id": game_id,
                            "nfl_week": int(cohort_row["nfl_week"]),
                            "information_anchor_id": anchor_id,
                            "episode_id": str(anchor["episode_id"]),
                            "primary_event_id": str(
                                anchor["provisional_primary_event_id"]
                            ),
                            "primary_feature_known_at": pd.Timestamp(
                                anchor["provisional_known_at"]
                            ),
                            "primary_feature_support_status": (
                                primary_support_status
                            ),
                            "primary_feature_support_reason": str(
                                anchor[
                                    "provisional_feature_support_reason"
                                ]
                            ),
                            "primary_feature_snapshot_sha256": str(
                                anchor["provisional_snapshot_sha256"]
                            ),
                            "prestate_context_sha256": str(
                                prestate["prestate_context_sha256"]
                            ),
                            "venue": venue,
                            "actual_home_contract_id": contract_id,
                            "logical_market_id": str(
                                contract["logical_market_id"]
                            ),
                            "market_family": str(
                                contract["market_family"]
                            ),
                            "market_period": str(
                                contract["market_period"]
                            ),
                            "proposition_kind": str(
                                contract["proposition_kind"]
                            ),
                            "outcome_team": str(
                                contract["outcome_team"]
                            ),
                            "actual_home_contract_identity_sha256": str(
                                contract[
                                    "actual_home_contract_identity_sha256"
                                ]
                            ),
                            "native_raw_manifest_sha256s_json": str(
                                contract[
                                    "native_raw_manifest_sha256s_json"
                                ]
                            ),
                            "native_raw_object_sha256s_json": str(
                                contract[
                                    "native_raw_object_sha256s_json"
                                ]
                            ),
                            "native_raw_manifest_set_sha256": str(
                                contract[
                                    "native_raw_manifest_set_sha256"
                                ]
                            ),
                            "native_raw_object_set_sha256": str(
                                contract[
                                    "native_raw_object_set_sha256"
                                ]
                            ),
                            "native_rule_evidence_sha256": (
                                str(
                                    contract[
                                        "native_rule_evidence_sha256"
                                    ]
                                )
                                if pd.notna(
                                    contract[
                                        "native_rule_evidence_sha256"
                                    ]
                                )
                                else pd.NA
                            ),
                            "native_rule_evidence_status": str(
                                contract[
                                    "native_rule_evidence_status"
                                ]
                            ),
                            "realized_outcome_comparability_status": str(
                                contract[
                                    "realized_outcome_comparability_status"
                                ]
                            ),
                            "realized_outcome_comparability_evidence_sha256": (
                                str(
                                    contract[
                                        "realized_outcome_comparability_evidence_sha256"
                                    ]
                                )
                                if pd.notna(
                                    contract[
                                        "realized_outcome_comparability_evidence_sha256"
                                    ]
                                )
                                else pd.NA
                            ),
                            "strict_contract_rule_equivalence_status": str(
                                contract[
                                    "strict_contract_rule_equivalence_status"
                                ]
                            ),
                            "strict_contract_rule_equivalence_evidence_sha256": (
                                str(
                                    contract[
                                        "strict_contract_rule_equivalence_evidence_sha256"
                                    ]
                                )
                                if pd.notna(
                                    contract[
                                        "strict_contract_rule_equivalence_evidence_sha256"
                                    ]
                                )
                                else pd.NA
                            ),
                            "market_continuity_evidence_sha256": str(
                                contract[
                                    "market_continuity_evidence_sha256"
                                ]
                            ),
                            "market_continuity_start_utc": (
                                continuity_start
                            ),
                            "market_continuity_end_utc": continuity_end,
                            "market_continuity_support_status": str(
                                contract[
                                    "market_continuity_support_status"
                                ]
                            ),
                            "target_orientation": "ACTUAL_HOME_OUTCOME",
                            "home_team": home_team,
                            "away_team": str(prestate["away_team"]),
                            "source_interval_start": pd.Timestamp(
                                anchor["source_interval_start"]
                            ),
                            "source_interval_end": pd.Timestamp(
                                anchor["source_interval_end"]
                            ),
                            "information_anchor": anchor_time,
                            "information_anchor_semantics": (
                                "SOURCE_INTERVAL_END_OF_THIS_CONSTITUENT_"
                                "FIRST_SEEN"
                            ),
                            "next_information_anchor_id": (
                                str(next_row["information_anchor_id"])
                                if next_row is not None
                                else pd.NA
                            ),
                            "next_parent_episode_id_audit_only": (
                                str(next_row["episode_id"])
                                if next_row is not None
                                else pd.NA
                            ),
                            "next_censor_source_interval_start": next_start,
                            "order_ambiguous": order_ambiguous,
                            "censor_time_utc": censor_time,
                            "landmark_seconds": l_seconds,
                            "endpoint_seconds": h_seconds,
                            "landmark_utc": landmark,
                            "endpoint_utc": endpoint,
                            "primary_selection_eligible": bool(
                                anchor["primary_selection_eligible"]
                            ),
                            "landmark_eligible": landmark_eligible,
                            "continuity_valid": continuity_valid,
                            "censor_boundary_eligible": bool(
                                anchor["censor_boundary_eligible"]
                            ),
                            "censored": censored,
                            "censor_reason": censor_reason,
                            "loss_eligible": loss_eligible,
                            "availability_h": availability,
                            "direction_h": direction,
                            "direction_eligible": direction is not None,
                            "delta_l_h": delta,
                            "direction_threshold_probability": (
                                DIRECTION_THRESHOLD
                            ),
                            "mark_l_status": mark_l.status,
                            "mark_l_source_time_utc": mark_l.source_time,
                            "mark_l_price": (
                                mark_l.price if mark_l.observed else math.nan
                            ),
                            "mark_l_staleness_seconds": (
                                mark_l.staleness_seconds
                            ),
                            "mark_l_trade_ids_json": json.dumps(
                                list(mark_l.trade_ids), separators=(",", ":")
                            ),
                            "mark_l_trade_id_set_sha256": (
                                mark_l.trade_id_set_sha256
                            ),
                            "mark_l_observation_count": (
                                mark_l.observation_count
                            ),
                            "mark_l_observed_size": mark_l.observed_size,
                            "mark_h_status": mark_h.status,
                            "mark_h_source_time_utc": mark_h.source_time,
                            "mark_h_price": (
                                mark_h.price if mark_h.observed else math.nan
                            ),
                            "mark_h_staleness_seconds": (
                                mark_h.staleness_seconds
                            ),
                            "mark_h_trade_ids_json": json.dumps(
                                list(mark_h.trade_ids), separators=(",", ":")
                            ),
                            "mark_h_trade_id_set_sha256": (
                                mark_h.trade_id_set_sha256
                            ),
                            "mark_h_observation_count": (
                                mark_h.observation_count
                            ),
                            "mark_h_observed_size": mark_h.observed_size,
                            "prior_30s_actual_trade_count": count_30,
                            "prior_30s_actual_trade_size": size_30,
                            "prior_60s_actual_trade_count": count_60,
                            "prior_60s_actual_trade_size": size_60,
                            "quarter": int(prestate["quarter"]),
                            "is_overtime": False,
                            "game_clock": prestate["game_clock"],
                            "game_seconds_remaining": _finite_or_nan(
                                prestate["game_seconds_remaining"]
                            ),
                            "pre_home_score": _finite_or_nan(
                                prestate["pre_home_score"]
                            ),
                            "pre_away_score": _finite_or_nan(
                                prestate["pre_away_score"]
                            ),
                            "score_margin_home": _finite_or_nan(
                                prestate["score_margin_home"]
                            ),
                            "possession_is_home": possession,
                            "possession_is_home_missing": possession is None,
                            "down": down,
                            "down_missing": not math.isfinite(down),
                            "distance": distance,
                            "distance_missing": not math.isfinite(distance),
                            "yardline_100": yardline,
                            "yardline_100_missing": not math.isfinite(
                                yardline
                            ),
                            "goal_to_go": _nullable_bool(
                                prestate["goal_to_go"]
                            ),
                            "pre_red_zone": _nullable_bool(
                                prestate["pre_red_zone"]
                            ),
                            "home_timeouts_remaining": _finite_or_nan(
                                prestate["home_timeouts_remaining"]
                            ),
                            "away_timeouts_remaining": _finite_or_nan(
                                prestate["away_timeouts_remaining"]
                            ),
                            "p_before_home": p_before,
                            "p_before_home_missing": p_before_missing,
                            "p_before_support_status": (
                                p_before_support_status
                            ),
                            "prestate_known_at": prestate_known,
                            **axes,
                            "provisional_yards_gained": _finite_or_nan(
                                anchor["provisional_yards_gained"]
                            ),
                            "provisional_return_yards": _finite_or_nan(
                                anchor["provisional_return_yards"]
                            ),
                            "provisional_score_points_observed": int(
                                anchor[
                                    "provisional_score_points_observed"
                                ]
                                or 0
                            ),
                            "provisional_turnover_observed": bool(
                                anchor[
                                    "provisional_turnover_observed"
                                ]
                            ),
                            "actor_is_home": _team_orientation(
                                anchor["provisional_actor_team"],
                                home_team,
                            ),
                            "beneficiary_is_home": _team_orientation(
                                anchor[
                                    "provisional_beneficiary_team"
                                ],
                                home_team,
                            ),
                            "attrition_reason": attrition_reason,
                        }
                    )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RegulationPanelV4Error(
            "single-game Panel V4 unexpectedly empty"
        )
    panel["availability_h"] = pd.array(
        panel["availability_h"], dtype="Int8"
    )
    panel["direction_h"] = panel["direction_h"].astype("string")
    grain = [
        "game_id",
        "information_anchor_id",
        "venue",
        "actual_home_contract_id",
        "landmark_seconds",
        "endpoint_seconds",
    ]
    if panel.duplicated(grain).any() or panel["source_row_id"].duplicated().any():
        raise RegulationPanelV4Error(
            "Panel V4 grain/source-row identity is not unique"
        )
    if (
        panel["anchor_kind"].ne(ANCHOR_KIND).any()
        or panel["analysis_role"].ne(ANALYSIS_ROLE).any()
        or panel["is_overtime"].any()
        or not set(FORMAL_MODEL_REQUIRED_COLUMNS).issubset(panel.columns)
        or bool(FORMAL_MODEL_FORBIDDEN_COLUMNS.intersection(panel.columns))
        or not panel.loc[
            panel["p_before_support_status"].eq("SUPPORTED"),
            "prestate_known_at",
        ].notna().all()
        or not panel.loc[
            panel["p_before_support_status"].ne("SUPPORTED"),
            "prestate_known_at",
        ].isna().all()
        or any(
            column.startswith(("post_", "final_", "factor__", "event_tag__"))
            for column in panel.columns
        )
    ):
        raise RegulationPanelV4Error(
            "Panel V4 formal feature/role boundary is invalid"
        )
    attrition = (
        panel.groupby(
            [
                "game_id",
                "landmark_seconds",
                "endpoint_seconds",
                "attrition_reason",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("row_count")
        .reset_index()
    )
    attrition.insert(0, "schema_version", ATTRITION_SCHEMA)
    return RegulationAnchorGamePanel(panel=panel, attrition=attrition)


def default_source_spec() -> RegulationPanelV4SourceSpec:
    return RegulationPanelV4SourceSpec(
        development=default_development_source_spec()
    )


def _read_context_v2_publication(
    *,
    project_root: Path,
    path: Path,
    expected_file_sha256: str,
    expected_game_count: int,
    expected_anchor_count: int,
) -> tuple[pd.DataFrame, Mapping[str, object], str]:
    expected = _require_sha256(
        expected_file_sha256,
        label="context_manifest_file_sha256",
    )
    try:
        resolved = _resolve_under(
            project_root, path, label="EventPrestateContextV2 manifest"
        )
    except DevelopmentPanelError as exc:
        raise RegulationPanelV4Error(str(exc)) from exc
    if not resolved.is_file() or _sha256_file(resolved) != expected:
        raise RegulationPanelV4Error("ContextV2 manifest hash mismatch")
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegulationPanelV4Error("ContextV2 manifest unreadable") from exc
    material = dict(manifest) if isinstance(manifest, dict) else {}
    declared_bundle = material.pop("bundle_sha256", None)
    descriptor = manifest.get("context")
    if (
        manifest.get("schema") != "EventPrestateContextManifestV2"
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("market_data_read") is not False
        or manifest.get("holdout_reaction_accessed") is not False
        or manifest.get("game_count") != expected_game_count
        or manifest.get("information_anchor_count") != expected_anchor_count
        or declared_bundle != _canonical_sha256(material)
        or not isinstance(descriptor, dict)
        or descriptor.get("schema") != "EventPrestateContextV2"
        or descriptor.get("row_count") != expected_anchor_count
        or descriptor.get("grain") != ["game_id", "information_anchor_id"]
    ):
        raise RegulationPanelV4Error("ContextV2 manifest contract mismatch")
    object_sha = _require_sha256(
        descriptor.get("object_sha256"),
        label="ContextV2.object_sha256",
    )
    dataset_root = resolved.parents[3]
    object_path = (
        dataset_root / str(descriptor.get("object_path", ""))
    ).resolve()
    if (
        project_root.resolve() not in object_path.parents
        or not object_path.is_file()
        or object_path.stat().st_size != descriptor.get("byte_length")
        or _sha256_file(object_path) != object_sha
    ):
        raise RegulationPanelV4Error("ContextV2 object verification failed")
    try:
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != expected_anchor_count:
            raise RegulationPanelV4Error("ContextV2 row-count mismatch")
        context = parquet.read().to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise RegulationPanelV4Error("ContextV2 parquet unreadable") from exc
    if (
        list(context.columns) != descriptor.get("schema_columns")
        or context["game_id"].nunique() != expected_game_count
        or context["information_anchor_id"].duplicated().any()
    ):
        raise RegulationPanelV4Error("ContextV2 schema/cohort mismatch")
    for row in context.to_dict("records"):
        if str(row["prestate_context_sha256"]) != (
            compute_prestate_context_sha256(row)
        ):
            raise RegulationPanelV4Error(
                "ContextV2 row feature hash mismatch"
            )
    return context, manifest, object_sha


def verify_regulation_panel_v4_sources(
    *,
    project_root: Path,
    source_spec: RegulationPanelV4SourceSpec | None = None,
) -> VerifiedRegulationPanelV4Sources:
    """Verify PIA, ContextV2, market, and cohort without opening holdout."""

    project = Path(project_root).resolve()
    spec = source_spec or default_source_spec()
    if (
        spec.expected_game_count != EXPECTED_GAME_COUNT
        or spec.expected_anchor_count != EXPECTED_ANCHOR_COUNT
    ):
        raise RegulationPanelV4Error(
            "Panel V4 requires the frozen exact-153/25,250-anchor cohort"
        )
    try:
        development = verify_development_sources(
            project_root=project,
            source_spec=spec.development,
        )
        anchors, anchor_manifest, anchor_object_sha = (
            _read_information_anchor_publication(
                project_root=project,
                path=spec.information_anchor_manifest_path,
                expected_file_sha256=(
                    spec.information_anchor_manifest_file_sha256
                ),
                expected_anchor_count=spec.expected_anchor_count,
                expected_game_count=spec.expected_game_count,
            )
        )
    except DevelopmentPanelError as exc:
        raise RegulationPanelV4Error(str(exc)) from exc
    context, context_manifest, context_object_sha = (
        _read_context_v2_publication(
            project_root=project,
            path=spec.context_manifest_path,
            expected_file_sha256=spec.context_manifest_file_sha256,
            expected_game_count=spec.expected_game_count,
            expected_anchor_count=spec.expected_anchor_count,
        )
    )
    game_sets = (
        set(development.market.games),
        set(development.cohort_metadata["game_id"].astype(str)),
        set(anchors["game_id"].astype(str)),
        set(context["game_id"].astype(str)),
    )
    if not all(values == game_sets[0] for values in game_sets[1:]):
        raise RegulationPanelV4Error("Panel V4 source game sets differ")
    if set(anchors["information_anchor_id"].astype(str)) != set(
        context["information_anchor_id"].astype(str)
    ):
        raise RegulationPanelV4Error("PIA and ContextV2 global FK mismatch")
    if (
        int(anchors["primary_selection_eligible"].sum()) != 25_070
        or int(anchors["censor_boundary_eligible"].sum())
        != spec.expected_anchor_count
        or context["is_overtime"].any()
        or context_manifest.get("sources", {})
        .get("provisional_information_anchor_v2", {})
        .get("manifest_file_sha256")
        != spec.information_anchor_manifest_file_sha256
    ):
        raise RegulationPanelV4Error(
            "PIA/ContextV2 eligibility or lineage contract mismatch"
        )
    return VerifiedRegulationPanelV4Sources(
        development=development,
        information_anchors=anchors,
        context=context,
        information_anchor_manifest=anchor_manifest,
        context_manifest=context_manifest,
        information_anchor_manifest_file_sha256=(
            spec.information_anchor_manifest_file_sha256
        ),
        information_anchor_object_sha256=anchor_object_sha,
        context_manifest_file_sha256=spec.context_manifest_file_sha256,
        context_object_sha256=context_object_sha,
    )


def _safe_path(root: Path, value: str | Path, *, label: str) -> Path:
    try:
        return _resolve_under(root.resolve(), value, label=label)
    except DevelopmentPanelError as exc:
        raise RegulationPanelV4Error(str(exc)) from exc


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise RegulationPanelV4Error("Panel V4 parquet encoding failed") from exc
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    payload = sink.getvalue().to_pybytes()
    try:
        persisted_schema = pq.read_schema(pa.BufferReader(payload))
    except pa.ArrowException as exc:
        raise RegulationPanelV4Error(
            "Panel V4 persisted parquet schema verification failed"
        ) from exc
    return (
        payload,
        _sha256_bytes(
            persisted_schema.remove_metadata().serialize().to_pybytes()
        ),
    )


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RegulationPanelV4Error(
                f"content-addressed collision: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise RegulationPanelV4Error(
                    f"content-addressed collision: {path}"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise RegulationPanelV4Error(f"atomic publication failed: {path}")


def _publish_frame(
    *,
    output_root: Path,
    game_id: str,
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload, schema_sha = _parquet_bytes(frame)
    object_sha = _sha256_bytes(payload)
    hexadecimal = object_sha.removeprefix(_SHA_PREFIX)
    relative = (
        Path("single-game")
        / game_id
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.parquet"
    )
    path = _safe_path(output_root, relative, label=f"{game_id}.{name}")
    _atomic_publish(path, payload)
    return {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": schema_sha,
    }


def _peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def _parse_memory_free_percent(output: str) -> int | None:
    match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%",
        output,
    )
    return int(match.group(1)) if match else None


def _system_memory_free_percent() -> int | None:
    try:
        completed = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_memory_free_percent(completed.stdout)


def _memory_gate() -> tuple[float, int | None]:
    peak_rss = _peak_rss_mib()
    if peak_rss >= PEAK_RSS_STOP_MIB:
        raise RegulationPanelV4Error("Panel V4 peak RSS reached 8 GiB stop gate")
    free_percent = _system_memory_free_percent()
    if (
        free_percent is not None
        and free_percent <= SYSTEM_FREE_STOP_PERCENT
    ):
        raise RegulationPanelV4Error(
            "system free memory reached 10% stop gate"
        )
    return peak_rss, free_percent


def _verified_capture_continuity(
    *,
    project_root: Path,
    market_batch: Mapping[str, object],
    market_game_manifest: Mapping[str, object],
    game_id: str,
) -> tuple[Mapping[str, Mapping[str, object]], Mapping[str, str]]:
    """Verify complete historical trade pagination for one capture window.

    This proves only completeness of the captured historical trade API pages
    inside the declared game window.  It is not an order-book, suspension, or
    continuous-tradability claim.
    """

    batch_bindings = market_batch.get("source_bindings")
    game_bindings = market_game_manifest.get("source_bindings")
    if not isinstance(batch_bindings, Mapping) or not isinstance(
        game_bindings, Mapping
    ):
        raise RegulationPanelV4Error(
            f"{game_id} market capture source bindings are missing"
        )
    capture_batch_sha = _require_sha256(
        batch_bindings.get("capture_batch_sha256"),
        label=f"{game_id}.capture_batch_sha256",
    )
    capture_index_sha = _require_sha256(
        game_bindings.get("capture_index_sha256"),
        label=f"{game_id}.capture_index_sha256",
    )
    capture_checkpoint_sha = _require_sha256(
        game_bindings.get("capture_checkpoint_sha256"),
        label=f"{game_id}.capture_checkpoint_sha256",
    )
    capture_root = _safe_path(
        project_root,
        DEFAULT_CAPTURE_ROOT,
        label="historical capture root",
    )
    batch_path = _safe_path(
        capture_root,
        Path("batch")
        / "manifests"
        / f"{capture_batch_sha.removeprefix(_SHA_PREFIX)}.manifest.json",
        label="historical capture batch manifest",
    )
    if (
        not batch_path.is_file()
        or _sha256_file(batch_path) != capture_batch_sha
    ):
        raise RegulationPanelV4Error(
            "historical capture batch manifest hash mismatch"
        )
    try:
        capture_batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegulationPanelV4Error(
            "historical capture batch manifest unreadable"
        ) from exc
    game_entries = [
        entry
        for entry in capture_batch.get("game_indexes", [])
        if isinstance(entry, Mapping)
        and str(entry.get("game_id")) == game_id
    ]
    if (
        capture_batch.get("schema")
        != "nfl_moneyline_expansion_batch_manifest_v1"
        or capture_batch.get("capture_scope") != "moneyline-only"
        or len(game_entries) != 1
        or game_entries[0].get("index_sha256") != capture_index_sha
    ):
        raise RegulationPanelV4Error(
            f"{game_id} capture batch/index contract mismatch"
        )
    index_path = _safe_path(
        capture_root,
        str(game_entries[0].get("index_path", "")),
        label=f"{game_id}.capture_index",
    )
    if (
        not index_path.is_file()
        or _sha256_file(index_path) != capture_index_sha
    ):
        raise RegulationPanelV4Error(
            f"{game_id} capture index hash mismatch"
        )
    try:
        capture_index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegulationPanelV4Error(
            f"{game_id} capture index unreadable"
        ) from exc
    identity = capture_index.get("identity")
    kalshi_event = capture_index.get("kalshi_event_markets")
    kalshi_trades = capture_index.get("kalshi_trades")
    polymarket = capture_index.get("polymarket_trades")
    raw_manifests = capture_index.get("raw_manifests")
    if (
        capture_index.get("schema")
        != "nfl_moneyline_game_capture_index_v1"
        or capture_index.get("reaction_access") != "CLOSED"
        or capture_index.get("game_id") != game_id
        or not isinstance(identity, Mapping)
        or not isinstance(kalshi_event, Mapping)
        or not isinstance(kalshi_trades, list)
        or not isinstance(polymarket, Mapping)
        or not isinstance(raw_manifests, list)
        or kalshi_event.get("terminal_proof") != "explicit_empty_cursor"
        or len(kalshi_trades) != 2
        or any(
            not isinstance(entry, Mapping)
            or entry.get("terminal_proof") != "explicit_empty_cursor"
            or int(entry.get("page_count", 0)) <= 0
            for entry in kalshi_trades
        )
    ):
        raise RegulationPanelV4Error(
            f"{game_id} Kalshi pagination proof is incomplete"
        )
    windows = polymarket.get("terminal_windows")
    if (
        not isinstance(windows, list)
        or not windows
        or int(polymarket.get("terminal_window_count", 0)) != len(windows)
        or any(
            not isinstance(window, Mapping)
            or window.get("terminal_proof") != "unsaturated_time_window"
            for window in windows
        )
    ):
        raise RegulationPanelV4Error(
            f"{game_id} Polymarket pagination proof is incomplete"
        )
    try:
        start = pd.Timestamp(
            int(identity["start_ts"]), unit="s", tz="UTC"
        )
        end = pd.Timestamp(int(identity["end_ts"]), unit="s", tz="UTC")
    except (KeyError, TypeError, ValueError) as exc:
        raise RegulationPanelV4Error(
            f"{game_id} capture window is invalid"
        ) from exc
    if (
        start >= end
        or min(int(window["start_ts"]) for window in windows)
        > int(identity["start_ts"])
        or max(int(window["end_ts"]) for window in windows)
        < int(identity["end_ts"])
    ):
        raise RegulationPanelV4Error(
            f"{game_id} Polymarket terminal windows do not cover capture"
        )

    result: dict[str, Mapping[str, object]] = {}
    for venue in ("kalshi", "polymarket"):
        entries = [
            entry
            for entry in raw_manifests
            if isinstance(entry, Mapping)
            and f"source={venue}/" in str(entry.get("manifest_path", ""))
        ]
        manifest_shas = sorted(
            {
                _require_sha256(
                    entry.get("manifest_sha256"),
                    label=f"{game_id}.{venue}.raw_manifest_sha256",
                )
                for entry in entries
            }
        )
        object_shas = sorted(
            {
                _require_sha256(
                    entry.get("object_sha256"),
                    label=f"{game_id}.{venue}.raw_object_sha256",
                )
                for entry in entries
            }
        )
        if not manifest_shas or not object_shas:
            raise RegulationPanelV4Error(
                f"{game_id} {venue} raw capture lineage is empty"
            )
        evidence = {
            "capture_batch_sha256": capture_batch_sha,
            "capture_index_sha256": capture_index_sha,
            "capture_checkpoint_sha256": capture_checkpoint_sha,
            "venue": venue,
            "start_utc": start,
            "end_utc": end,
            "raw_manifest_sha256s": manifest_shas,
            "raw_object_sha256s": object_shas,
            "continuity_contract": MARKET_CONTINUITY_CONTRACT,
        }
        result[venue] = {
            "market_continuity_start_utc": start,
            "market_continuity_end_utc": end,
            "market_continuity_support_status": (
                "VERIFIED_CAPTURE_PAGINATION"
            ),
            "market_continuity_evidence_sha256": _canonical_sha256(
                evidence
            ),
            "native_raw_manifest_sha256s_json": json.dumps(
                manifest_shas, separators=(",", ":")
            ),
            "native_raw_object_sha256s_json": json.dumps(
                object_shas, separators=(",", ":")
            ),
            "native_raw_manifest_set_sha256": _canonical_sha256(
                manifest_shas
            ),
            "native_raw_object_set_sha256": _canonical_sha256(
                object_shas
            ),
        }
    return result, {
        "market_capture_batch_sha256": capture_batch_sha,
        "market_capture_index_sha256": capture_index_sha,
        "market_capture_checkpoint_sha256": capture_checkpoint_sha,
    }


def _augment_contract_authority(
    *,
    contracts: pd.DataFrame,
    inventory: pd.DataFrame,
    home_team: str,
    market_game_manifest_sha256: str,
    market_inventory_object_sha256: str,
    continuity: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Bind adapted home contracts to the exact normalized and raw bytes."""

    manifest_sha = _require_sha256(
        market_game_manifest_sha256,
        label="market_game_manifest_sha256",
    )
    inventory_sha = _require_sha256(
        market_inventory_object_sha256,
        label="market_inventory_object_sha256",
    )
    rows: list[dict[str, object]] = []
    for contract in contracts.to_dict("records"):
        venue = str(contract["venue"])
        raw_contract_id = str(contract["contract_id"])
        matches = inventory.loc[
            inventory["venue"].astype(str).eq(venue)
            & inventory["raw_contract_id"].astype(str).eq(raw_contract_id)
            & inventory["outcome"].astype(str).eq(home_team)
            & inventory["family"].astype(str).eq("moneyline")
            & inventory["period"].astype(str).eq("full_game")
            & inventory["kind"].astype(str).eq("primitive")
            & inventory["analysis_eligible"].eq(True)  # noqa: E712
        ]
        if str(contract["contract_role"]) != "ACTUAL_HOME_OUTCOME":
            rows.append(contract)
            continue
        if len(matches) != 1 or venue not in continuity:
            raise RegulationPanelV4Error(
                "home contract cannot be bound to exact inventory/raw lineage"
            )
        inventory_row = matches.iloc[0].to_dict()
        continuity_row = dict(continuity[venue])
        identity_material = {
            "market_game_manifest_sha256": manifest_sha,
            "market_inventory_object_sha256": inventory_sha,
            "exact_inventory_row": inventory_row,
            "native_raw_manifest_sha256s_json": continuity_row[
                "native_raw_manifest_sha256s_json"
            ],
            "native_raw_object_sha256s_json": continuity_row[
                "native_raw_object_sha256s_json"
            ],
        }
        rows.append(
            {
                **contract,
                "logical_market_id": str(
                    inventory_row["logical_market_id"]
                ),
                "market_family": str(inventory_row["family"]),
                "market_period": str(inventory_row["period"]),
                "proposition_kind": str(inventory_row["kind"]),
                "outcome_team": str(inventory_row["outcome"]),
                "actual_home_contract_identity_sha256": (
                    _canonical_sha256(identity_material)
                ),
                **continuity_row,
                "native_rule_evidence_sha256": pd.NA,
                "native_rule_evidence_status": (
                    "RAW_METADATA_CAPTURED_RULE_SEMANTICS_NOT_ADJUDICATED"
                ),
                "realized_outcome_comparability_status": (
                    "UNPROVEN_NOT_ADJUDICATED"
                ),
                "realized_outcome_comparability_evidence_sha256": pd.NA,
                "strict_contract_rule_equivalence_status": "UNPROVEN",
                "strict_contract_rule_equivalence_evidence_sha256": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _panel_counts(built: RegulationAnchorGamePanel) -> dict[str, int]:
    return {
        "primary_information_anchor_count": int(
            built.panel["information_anchor_id"].nunique()
        ),
        "panel_row_count": len(built.panel),
        "landmark_eligible_count": int(
            built.panel["landmark_eligible"].sum()
        ),
        "continuity_valid_count": int(
            built.panel["continuity_valid"].sum()
        ),
        "loss_eligible_count": int(built.panel["loss_eligible"].sum()),
        "direction_eligible_count": int(
            built.panel["direction_eligible"].sum()
        ),
        "censored_count": int(built.panel["censored"].sum()),
        "order_ambiguous_count": int(
            built.panel["order_ambiguous"].sum()
        ),
        "availability_zero_count": int(
            built.panel["availability_h"].eq(0).sum()
        ),
        "availability_one_count": int(
            built.panel["availability_h"].eq(1).sum()
        ),
        "possession_is_home_missing_rows": int(
            built.panel["possession_is_home_missing"].sum()
        ),
        "down_missing_rows": int(built.panel["down_missing"].sum()),
        "distance_missing_rows": int(
            built.panel["distance_missing"].sum()
        ),
        "yardline_100_missing_rows": int(
            built.panel["yardline_100_missing"].sum()
        ),
    }


def _game_manifest_material(
    *,
    game_id: str,
    source_hashes: Mapping[str, str],
    counts: Mapping[str, int],
    descriptors: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": GAME_MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "experiment_id": EXPERIMENT_ID,
        "dataset_split": DATASET_SPLIT,
        "game_id": game_id,
        "holdout_reaction_accessed": False,
        "regulation_only": True,
        "formal_inputs": [
            "ProvisionalInformationAnchorV2",
            "EventPrestateContextV2",
            "VERIFIED_ACTUAL_MARKET_TRADES",
        ],
        "forbidden_formal_inputs": [
            "FinalizedEpisodeV2",
            "FactsV4",
            "POST_EVENT_REFERENCE",
        ],
        "anchor_kind": ANCHOR_KIND,
        "analysis_role": ANALYSIS_ROLE,
        "source_hashes": dict(source_hashes),
        "feature_blocks": {
            key: list(value) for key, value in FEATURE_BLOCKS.items()
        },
        "primary_feature_contract": PRIMARY_FEATURE_CONTRACT,
        "prestate_context_contract": PRESTATE_CONTEXT_CONTRACT,
        "market_continuity_contract": MARKET_CONTINUITY_CONTRACT,
        "formal_model_required_columns": list(
            FORMAL_MODEL_REQUIRED_COLUMNS
        ),
        "formal_model_forbidden_columns": sorted(
            FORMAL_MODEL_FORBIDDEN_COLUMNS
        ),
        "missing_indicators": [
            "possession_is_home_missing",
            "down_missing",
            "distance_missing",
            "yardline_100_missing",
        ],
        "censor_contract": {
            "boundary_table": "ProvisionalInformationAnchorV2",
            "next_boundary_time": "source_interval_start",
            "same_parent_constituents_included": True,
            "overlap": "ORDER_AMBIGUOUS_EXCLUDE",
            "unknown_after_last_anchor": "CENSORED",
        },
        "counts": dict(counts),
        "tables": [dict(descriptor) for descriptor in descriptors],
        "publication_gate": "PASS",
    }


def _publish_built_game(
    *,
    output_root: Path,
    game_id: str,
    built: RegulationAnchorGamePanel,
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, int]]:
    missing_hashes = _REQUIRED_SOURCE_HASH_KEYS.difference(source_hashes)
    if missing_hashes:
        raise RegulationPanelV4Error(
            f"{game_id} publication source hashes missing "
            f"{sorted(missing_hashes)}"
        )
    normalized_hashes = {
        key: _require_sha256(
            value, label=f"{game_id}.source_hashes.{key}"
        )
        for key, value in sorted(source_hashes.items())
    }
    descriptors = [
        _publish_frame(
            output_root=output_root,
            game_id=game_id,
            name="regulation_decision_panel",
            frame=built.panel,
        ),
        _publish_frame(
            output_root=output_root,
            game_id=game_id,
            name="attrition",
            frame=built.attrition,
        ),
    ]
    counts = _panel_counts(built)
    material = _game_manifest_material(
        game_id=game_id,
        source_hashes=normalized_hashes,
        counts=counts,
        descriptors=descriptors,
    )
    game_bundle = _manifest_semantic_sha256(material)
    payload = _canonical_bytes(
        {**material, "bundle_sha256": game_bundle}
    )
    manifest_sha = _sha256_bytes(payload)
    hexadecimal = manifest_sha.removeprefix(_SHA_PREFIX)
    relative = (
        Path("single-game")
        / game_id
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    _atomic_publish(
        _safe_path(
            output_root,
            relative,
            label=f"{game_id}.manifest",
        ),
        payload,
    )
    return (
        {
            "game_id": game_id,
            "manifest_path": relative.as_posix(),
            "manifest_sha256": manifest_sha,
            "bundle_sha256": game_bundle,
            "panel_object_sha256": descriptors[0]["object_sha256"],
            "panel_row_count": len(built.panel),
        },
        counts,
    )


def _batch_manifest_material(
    *,
    game_entries: Sequence[Mapping[str, object]],
    counts: Mapping[str, int],
    sources: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema": BATCH_MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "experiment_id": EXPERIMENT_ID,
        "dataset_split": DATASET_SPLIT,
        "holdout_reaction_accessed": False,
        "regulation_only": True,
        "game_count": len(game_entries),
        "formal_unit": (
            "information_anchor_id+venue+actual_home_contract_id+L+H"
        ),
        "anchor_kind": ANCHOR_KIND,
        "analysis_role": ANALYSIS_ROLE,
        "formal_inputs": [
            "ProvisionalInformationAnchorV2",
            "EventPrestateContextV2",
            "VERIFIED_ACTUAL_MARKET_TRADES",
        ],
        "forbidden_formal_inputs": [
            "FinalizedEpisodeV2",
            "FactsV4",
            "POST_EVENT_REFERENCE",
        ],
        "time_grid": {
            "landmark_seconds": list(LANDMARK_SECONDS),
            "endpoint_seconds": list(ENDPOINT_SECONDS),
            "require_h_gt_l": True,
        },
        "actual_trade_contract": {
            "mark_l": (
                "LATEST_SIZE_WEIGHTED_SAME_TIMESTAMP_BUCKET_IN_(T,T+L]"
            ),
            "availability_h": "NEW_ACTUAL_TRADE_IN_(T+L,T+H]",
            "availability_zero_semantics": (
                "NO_NEW_ACTUAL_TRADE_OBSERVED_IN_(T+L,T+H];"
                "NOT_OPEN_BOOK_TRADABILITY_OR_LIQUIDITY"
            ),
            "forward_fill": False,
            "authoritative_suspension_channel": False,
        },
        "censor_contract": {
            "next_boundary_time": (
                "NEXT_PIA_CENSOR_BOUNDARY_SOURCE_INTERVAL_START"
            ),
            "same_parent_constituents_included": True,
            "overlap": "ORDER_AMBIGUOUS_EXCLUDE",
            "censored_rows_in_loss": False,
        },
        "direction_threshold_probability": DIRECTION_THRESHOLD,
        "feature_blocks": {
            key: list(value) for key, value in FEATURE_BLOCKS.items()
        },
        "primary_feature_contract": PRIMARY_FEATURE_CONTRACT,
        "prestate_context_contract": PRESTATE_CONTEXT_CONTRACT,
        "market_continuity_contract": MARKET_CONTINUITY_CONTRACT,
        "formal_model_required_columns": list(
            FORMAL_MODEL_REQUIRED_COLUMNS
        ),
        "formal_model_forbidden_columns": sorted(
            FORMAL_MODEL_FORBIDDEN_COLUMNS
        ),
        "missing_indicators": [
            "possession_is_home_missing",
            "down_missing",
            "distance_missing",
            "yardline_100_missing",
        ],
        "sources": dict(sources),
        "counts": dict(counts),
        "games": [dict(entry) for entry in game_entries],
        "publication_gate": "PASS",
        "model_training_started": False,
    }


def _publish_batch_manifest(
    *,
    output_root: Path,
    game_entries: Sequence[Mapping[str, object]],
    counts: Mapping[str, int],
    sources: Mapping[str, str],
) -> tuple[Path, str, str]:
    normalized_sources = {
        key: _require_sha256(value, label=f"batch.sources.{key}")
        for key, value in sorted(sources.items())
    }
    material = _batch_manifest_material(
        game_entries=game_entries,
        counts=counts,
        sources=normalized_sources,
    )
    batch_sha = _manifest_semantic_sha256(material)
    payload = _canonical_bytes(
        {**material, "batch_sha256": batch_sha}
    )
    manifest_sha = _sha256_bytes(payload)
    hexadecimal = manifest_sha.removeprefix(_SHA_PREFIX)
    relative = (
        Path("batches")
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.batch-index.json"
    )
    path = _safe_path(
        output_root, relative, label="Panel V4 batch manifest"
    )
    _atomic_publish(path, payload)
    return path, manifest_sha, batch_sha


def publish_regulation_panel_v4_frames(
    *,
    project_root: Path,
    output_root: Path,
    game_panels: Mapping[str, RegulationAnchorGamePanel],
    game_source_hashes: Mapping[str, Mapping[str, str]],
    batch_sources: Mapping[str, str],
    expected_game_count: int,
) -> PublishedRegulationPanelV4:
    """Publish already-built frames through the canonical PanelV4 graph.

    This is the small, shared primitive used by the producer/model integration
    fixture.  The exact-153 publisher uses the same game and batch publishers.
    """

    started = time.perf_counter()
    if (
        isinstance(expected_game_count, bool)
        or expected_game_count < 1
        or len(game_panels) != expected_game_count
        or set(game_panels) != set(game_source_hashes)
    ):
        raise RegulationPanelV4Error(
            "frame publication game/source cohort mismatch"
        )
    project = Path(project_root).resolve()
    output = _safe_path(project, output_root, label="Panel V4 output")
    aggregate: dict[str, int] = {}
    entries: list[dict[str, object]] = []
    for game_id in sorted(game_panels):
        entry, counts = _publish_built_game(
            output_root=output,
            game_id=game_id,
            built=game_panels[game_id],
            source_hashes=game_source_hashes[game_id],
        )
        entries.append(entry)
        for key, value in counts.items():
            aggregate[key] = aggregate.get(key, 0) + value
    path, manifest_sha, batch_sha = _publish_batch_manifest(
        output_root=output,
        game_entries=entries,
        counts=aggregate,
        sources=batch_sources,
    )
    return PublishedRegulationPanelV4(
        output_root=output,
        batch_manifest_path=path,
        batch_manifest_sha256=manifest_sha,
        batch_sha256=batch_sha,
        game_count=len(entries),
        primary_information_anchor_count=aggregate[
            "primary_information_anchor_count"
        ],
        panel_row_count=aggregate["panel_row_count"],
        landmark_eligible_count=aggregate["landmark_eligible_count"],
        loss_eligible_count=aggregate["loss_eligible_count"],
        direction_eligible_count=aggregate[
            "direction_eligible_count"
        ],
        censored_count=aggregate["censored_count"],
        order_ambiguous_count=aggregate["order_ambiguous_count"],
        availability_zero_count=aggregate["availability_zero_count"],
        availability_one_count=aggregate["availability_one_count"],
        runtime_seconds=time.perf_counter() - started,
        peak_rss_mib=_peak_rss_mib(),
    )


def validate_exact153_regulation_panel_v4_config(
    *,
    project_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_spec: RegulationPanelV4SourceSpec | None = None,
) -> Mapping[str, object]:
    """Validate exact-153 authority and publication configuration, no writes."""

    project = Path(project_root).resolve()
    output = _safe_path(project, output_root, label="Panel V4 output")
    _memory_gate()
    sources = verify_regulation_panel_v4_sources(
        project_root=project,
        source_spec=source_spec,
    )
    batch_bindings = sources.development.market.document.get(
        "source_bindings"
    )
    if not isinstance(batch_bindings, Mapping):
        raise RegulationPanelV4Error(
            "market batch capture binding is missing"
        )
    capture_batch_sha = _require_sha256(
        batch_bindings.get("capture_batch_sha256"),
        label="market_capture_batch_sha256",
    )
    capture_root = _safe_path(
        project, DEFAULT_CAPTURE_ROOT, label="historical capture root"
    )
    capture_batch_path = _safe_path(
        capture_root,
        Path("batch")
        / "manifests"
        / f"{capture_batch_sha.removeprefix(_SHA_PREFIX)}.manifest.json",
        label="historical capture batch manifest",
    )
    if (
        not capture_batch_path.is_file()
        or _sha256_file(capture_batch_path) != capture_batch_sha
    ):
        raise RegulationPanelV4Error(
            "historical capture batch authority failed"
        )
    try:
        capture_batch = json.loads(
            capture_batch_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RegulationPanelV4Error(
            "historical capture batch is unreadable"
        ) from exc
    capture_games = {
        str(entry.get("game_id"))
        for entry in capture_batch.get("game_indexes", [])
        if isinstance(entry, Mapping)
        and isinstance(entry.get("game_id"), str)
        and _require_sha256(
            entry.get("index_sha256"),
            label="capture_batch.index_sha256",
        )
    }
    panel_games = set(sources.development.market.games)
    if (
        capture_batch.get("schema")
        != "nfl_moneyline_expansion_batch_manifest_v1"
        or capture_batch.get("capture_scope") != "moneyline-only"
        or len(panel_games) != EXPECTED_GAME_COUNT
        or not panel_games.issubset(capture_games)
    ):
        raise RegulationPanelV4Error(
            "capture and PanelV4 exact-153 cohorts do not reconcile"
        )
    for game_id, descriptor in sources.development.market.games.items():
        manifest, _ = _verify_game_manifest(
            batch=sources.development.market,
            descriptor=descriptor,
            label=f"market-config.{game_id}",
        )
        bindings = manifest.get("source_bindings")
        if not isinstance(bindings, Mapping):
            raise RegulationPanelV4Error(
                f"{game_id} capture bindings are missing"
            )
        _require_sha256(
            bindings.get("capture_index_sha256"),
            label=f"{game_id}.capture_index_sha256",
        )
        _require_sha256(
            bindings.get("capture_checkpoint_sha256"),
            label=f"{game_id}.capture_checkpoint_sha256",
        )
    return {
        "status": "VALIDATED_NO_WRITE",
        "schema": SCHEMA_VERSION,
        "game_count": len(panel_games),
        "information_anchor_count": len(sources.information_anchors),
        "primary_information_anchor_count": int(
            sources.information_anchors[
                "primary_selection_eligible"
            ].sum()
        ),
        "output_root": str(output),
        "output_namespace": (
            "<output>/single-game/<game_id>/objects|manifests/sha256/"
            "<aa>/<sha> + <output>/batches/manifests/sha256/<aa>/<sha>"
        ),
        "manifest_authority": {
            "information_anchor_manifest_file_sha256": (
                sources.information_anchor_manifest_file_sha256
            ),
            "context_manifest_file_sha256": (
                sources.context_manifest_file_sha256
            ),
            "market_batch_file_sha256": (
                sources.development.spec.market_batch_file_sha256
            ),
            "market_capture_batch_sha256": capture_batch_sha,
            "cohort_authority_sha256": (
                sources.development.cohort_authority_sha256
            ),
            "cohort_mapping_sha256": (
                sources.development.cohort_mapping_sha256
            ),
        },
        "execution": {
            "streaming": "ONE_GAME_AT_A_TIME",
            "batch_pointer_published_last": True,
            "atomic_publish": (
                "FSYNC_TEMP_THEN_HARDLINK_CREATE_ONLY;"
                "EXISTING_BYTES_MUST_MATCH"
            ),
            "memory_warning_gate_peak_rss_mib": PEAK_RSS_WARNING_MIB,
            "memory_stop_gate_peak_rss_mib": PEAK_RSS_STOP_MIB,
            "memory_stop_gate_system_free_percent": (
                SYSTEM_FREE_STOP_PERCENT
            ),
        },
        "formal_model_required_column_count": len(
            FORMAL_MODEL_REQUIRED_COLUMNS
        ),
        "holdout_reaction_accessed": False,
        "write_performed": False,
    }


def publish_exact153_regulation_panel_v4(
    *,
    project_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_spec: RegulationPanelV4SourceSpec | None = None,
    progress_callback: (
        Callable[[Mapping[str, object]], None] | None
    ) = None,
) -> PublishedRegulationPanelV4:
    """Stream and atomically publish the exact-153 Panel V4 once."""

    started = time.perf_counter()
    project = Path(project_root).resolve()
    output = _safe_path(project, output_root, label="Panel V4 output")
    _memory_gate()
    sources = verify_regulation_panel_v4_sources(
        project_root=project,
        source_spec=source_spec,
    )
    game_entries: list[dict[str, object]] = []
    aggregate = {
        "primary_information_anchor_count": 0,
        "panel_row_count": 0,
        "landmark_eligible_count": 0,
        "continuity_valid_count": 0,
        "loss_eligible_count": 0,
        "direction_eligible_count": 0,
        "censored_count": 0,
        "order_ambiguous_count": 0,
        "availability_zero_count": 0,
        "availability_one_count": 0,
        "possession_is_home_missing_rows": 0,
        "down_missing_rows": 0,
        "distance_missing_rows": 0,
        "yardline_100_missing_rows": 0,
    }
    for game_index, game_id in enumerate(
        sorted(sources.development.market.games), start=1
    ):
        peak_rss, free_percent = _memory_gate()
        try:
            market_manifest, _ = _verify_game_manifest(
                batch=sources.development.market,
                descriptor=sources.development.market.games[game_id],
                label="market",
            )
            observations = _read_verified_table(
                batch=sources.development.market,
                manifest=market_manifest,
                table_name="actual_market_observations",
                market_style=True,
                label=f"market.{game_id}",
            )
            inventory = _read_verified_table(
                batch=sources.development.market,
                manifest=market_manifest,
                table_name="contract_inventory",
                market_style=True,
                label=f"market.{game_id}",
            )
        except DevelopmentPanelError as exc:
            raise RegulationPanelV4Error(str(exc)) from exc
        game_context = sources.context.loc[
            sources.context["game_id"].astype(str).eq(game_id)
        ].reset_index(drop=True)
        teams = game_context.loc[:, ["home_team", "away_team"]].drop_duplicates()
        if len(teams) != 1:
            raise RegulationPanelV4Error(
                f"{game_id} ContextV2 team identity inconsistent"
            )
        market_rows, contracts, _ = _adapt_market(
            game_id=game_id,
            home_team=str(teams.iloc[0]["home_team"]),
            observations=observations,
            inventory=inventory,
        )
        observation_descriptor = _table_descriptor(
            market_manifest,
            table_name="actual_market_observations",
            market_style=True,
            label=f"market.{game_id}",
        )
        inventory_descriptor = _table_descriptor(
            market_manifest,
            table_name="contract_inventory",
            market_style=True,
            label=f"market.{game_id}",
        )
        continuity, capture_hashes = _verified_capture_continuity(
            project_root=project,
            market_batch=sources.development.market.document,
            market_game_manifest=market_manifest,
            game_id=game_id,
        )
        market_game_manifest_sha = str(
            sources.development.market.games[game_id]["manifest_sha256"]
        )
        contracts = _augment_contract_authority(
            contracts=contracts,
            inventory=inventory,
            home_team=str(teams.iloc[0]["home_team"]),
            market_game_manifest_sha256=market_game_manifest_sha,
            market_inventory_object_sha256=str(
                inventory_descriptor["object_sha256"]
            ),
            continuity=continuity,
        )
        source_hashes = {
            "information_anchor_manifest_sha256": (
                sources.information_anchor_manifest_file_sha256
            ),
            "information_anchor_object_sha256": (
                sources.information_anchor_object_sha256
            ),
            "context_manifest_sha256": (
                sources.context_manifest_file_sha256
            ),
            "context_object_sha256": sources.context_object_sha256,
            "market_game_manifest_sha256": market_game_manifest_sha,
            "market_observations_object_sha256": str(
                observation_descriptor["object_sha256"]
            ),
            "market_inventory_object_sha256": str(
                inventory_descriptor["object_sha256"]
            ),
            **capture_hashes,
            "cohort_authority_sha256": (
                sources.development.cohort_authority_sha256
            ),
        }
        cohort = sources.development.cohort_metadata.loc[
            sources.development.cohort_metadata["game_id"].eq(game_id)
        ]
        built = build_regulation_anchor_game_panel(
            information_anchors=sources.information_anchors.loc[
                sources.information_anchors["game_id"].astype(str).eq(game_id)
            ].reset_index(drop=True),
            context=game_context,
            market_rows=market_rows,
            contracts=contracts,
            cohort_row=cohort.iloc[0].to_dict(),
            source_hashes=source_hashes,
        )
        entry, counts = _publish_built_game(
            output_root=output,
            game_id=game_id,
            built=built,
            source_hashes=source_hashes,
        )
        for key, value in counts.items():
            aggregate[key] += value
        game_entries.append(entry)
        del observations, inventory, market_rows, contracts, built
        peak_rss, free_percent = _memory_gate()
        if (
            progress_callback is not None
            and (
                game_index % 25 == 0
                or game_index == EXPECTED_GAME_COUNT
                or peak_rss > PEAK_RSS_WARNING_MIB
            )
        ):
            progress_callback(
                {
                    "status": (
                        "RSS_WARNING"
                        if peak_rss > PEAK_RSS_WARNING_MIB
                        else "RUNNING"
                    ),
                    "games_completed": game_index,
                    "game_count": EXPECTED_GAME_COUNT,
                    "panel_rows_published": aggregate[
                        "panel_row_count"
                    ],
                    "loss_eligible_rows": aggregate[
                        "loss_eligible_count"
                    ],
                    "peak_rss_mib": round(peak_rss, 3),
                    "system_memory_free_percent": free_percent,
                    "failures": 0,
                }
            )

    if (
        len(game_entries) != EXPECTED_GAME_COUNT
        or aggregate["primary_information_anchor_count"] != 25_070
    ):
        raise RegulationPanelV4Error(
            "exact-153 Panel V4 publication reconciliation failed"
        )
    batch_sources = {
        "market_batch_file_sha256": (
            sources.development.spec.market_batch_file_sha256
        ),
        "market_capture_batch_sha256": _require_sha256(
            sources.development.market.document.get(
                "source_bindings", {}
            ).get("capture_batch_sha256"),
            label="market_capture_batch_sha256",
        ),
        "information_anchor_manifest_file_sha256": (
            sources.information_anchor_manifest_file_sha256
        ),
        "information_anchor_object_sha256": (
            sources.information_anchor_object_sha256
        ),
        "context_manifest_file_sha256": (
            sources.context_manifest_file_sha256
        ),
        "context_object_sha256": sources.context_object_sha256,
        "cohort_authority_sha256": (
            sources.development.cohort_authority_sha256
        ),
        "cohort_mapping_sha256": (
            sources.development.cohort_mapping_sha256
        ),
    }
    batch_path, batch_manifest_sha, batch_sha = _publish_batch_manifest(
        output_root=output,
        game_entries=game_entries,
        counts=aggregate,
        sources=batch_sources,
    )
    _memory_gate()
    return PublishedRegulationPanelV4(
        output_root=output,
        batch_manifest_path=batch_path,
        batch_manifest_sha256=batch_manifest_sha,
        batch_sha256=batch_sha,
        game_count=len(game_entries),
        primary_information_anchor_count=aggregate[
            "primary_information_anchor_count"
        ],
        panel_row_count=aggregate["panel_row_count"],
        landmark_eligible_count=aggregate["landmark_eligible_count"],
        loss_eligible_count=aggregate["loss_eligible_count"],
        direction_eligible_count=aggregate["direction_eligible_count"],
        censored_count=aggregate["censored_count"],
        order_ambiguous_count=aggregate["order_ambiguous_count"],
        availability_zero_count=aggregate["availability_zero_count"],
        availability_one_count=aggregate["availability_one_count"],
        runtime_seconds=time.perf_counter() - started,
        peak_rss_mib=_peak_rss_mib(),
    )


__all__ = [
    "ANALYSIS_ROLE",
    "ANCHOR_KIND",
    "ATTRITION_SCHEMA",
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "DIRECTION_THRESHOLD",
    "ENDPOINT_SECONDS",
    "FEATURE_BLOCKS",
    "FORMAL_MODEL_FORBIDDEN_COLUMNS",
    "FORMAL_MODEL_REQUIRED_COLUMNS",
    "LANDMARK_SECONDS",
    "MARKET_CONTINUITY_CONTRACT",
    "PRESTATE_CONTEXT_CONTRACT",
    "PRIMARY_FEATURE_CONTRACT",
    "REALIZED_OUTCOME_COMPARABILITY_STATUSES",
    "RegulationAnchorGamePanel",
    "RegulationPanelV4SourceSpec",
    "VerifiedRegulationPanelV4Sources",
    "PublishedRegulationPanelV4",
    "RegulationPanelV4Error",
    "SCHEMA_VERSION",
    "STRICT_CONTRACT_RULE_EQUIVALENCE_STATUSES",
    "build_regulation_anchor_game_panel",
    "default_source_spec",
    "publish_exact153_regulation_panel_v4",
    "publish_regulation_panel_v4_frames",
    "validate_exact153_regulation_panel_v4_config",
    "verify_regulation_panel_v4_sources",
]
