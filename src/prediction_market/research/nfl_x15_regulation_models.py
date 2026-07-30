"""Corrected regulation-only Stage-B probability models for NFL X-15.

This module consumes ``RegulationDecisionPanelV4``.  It deliberately has two
heads only:

``availability_H``
    Whether at least one *new* actual trade is observed in ``(T_L, T_H]``.

``direction_H | availability_H``
    The direction of the last new actual trade relative to ``mark_L``.

Censored windows are removed from both risk sets.  In particular, this module
does not contain or predict the legacy sports-survival target ``S_H``.
Polymarket is the sole development/model-selection venue.  Kalshi may only
receive a separately frozen Polymarket winner as a transport request.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from prediction_market.research.nfl_x15_regulation_panel_v4 import (
    ANALYSIS_ROLE as PANEL_ANALYSIS_ROLE,
    ANCHOR_KIND as PANEL_ANCHOR_KIND,
    BATCH_MANIFEST_SCHEMA as PANEL_BATCH_SCHEMA,
    ENDPOINT_SECONDS as PANEL_ENDPOINT_SECONDS,
    FEATURE_BLOCKS as PANEL_FEATURE_BLOCKS,
    FORMAL_MODEL_FORBIDDEN_COLUMNS as PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS,
    FORMAL_MODEL_REQUIRED_COLUMNS as PANEL_FORMAL_MODEL_REQUIRED_COLUMNS,
    GAME_MANIFEST_SCHEMA as PANEL_GAME_SCHEMA,
    LANDMARK_SECONDS as PANEL_LANDMARK_SECONDS,
    MARKET_CONTINUITY_CONTRACT as PANEL_MARKET_CONTINUITY_CONTRACT,
    PRESTATE_CONTEXT_CONTRACT as PANEL_PRESTATE_CONTEXT_CONTRACT,
    PRIMARY_FEATURE_CONTRACT as PANEL_PRIMARY_FEATURE_CONTRACT,
    REALIZED_OUTCOME_COMPARABILITY_STATUSES,
    SCHEMA_VERSION as PANEL_SCHEMA_VERSION,
    STRICT_CONTRACT_RULE_EQUIVALENCE_STATUSES,
)
from prediction_market.research.nfl_x15_calibration import (
    CALIBRATION_MIN_GAMES,
    CALIBRATION_MIN_ROWS,
    CALIBRATED,
    DIRECTION_CLASSES,
    PROBABILITY_EPSILON as CALIBRATION_PROBABILITY_EPSILON,
    TEMPERATURE_GRID,
    BinaryPlattCalibrator,
    DirectionTemperatureCalibrator,
    fit_binary_platt,
    fit_direction_temperature,
)
from prediction_market.research.nfl_x15_models import (
    RANDOM_STATE,
    X15WeekFold,
    _sha256_row_stream,
    build_x15_week_folds,
    effective_random_state,
    hierarchical_sample_weights,
)
from prediction_market.research.nfl_x15_native_rule_audit import (
    DEFAULT_CAPTURE_BATCH as NATIVE_RULE_CAPTURE_BATCH,
    RuleAuditError,
    verify_rule_audit_batch,
)
from prediction_market.sports.nfl_historical_sports_clean import (
    discover_main_checkout,
)


SCHEMA_VERSION: Final[str] = PANEL_SCHEMA_VERSION
PANEL_GAME_MANIFEST_SCHEMA: Final[str] = PANEL_GAME_SCHEMA
PANEL_BATCH_MANIFEST_SCHEMA: Final[str] = PANEL_BATCH_SCHEMA
OOF_SHARD_MANIFEST_SCHEMA: Final[str] = (
    "RegulationStageBOOFShardManifestV1"
)
OOF_BATCH_MANIFEST_SCHEMA: Final[str] = (
    "RegulationStageBOOFBatchManifestV1"
)
SELECTION_SCHEMA: Final[str] = "RegulationStageBSelectionV1"
SELECTION_LOCK_SCHEMA: Final[str] = "RegulationStageBSelectionLockV2"
TRANSPORT_TARGET_VERSION: Final[str] = (
    "REGULATION_AVAILABILITY_AND_DIRECTION_V1"
)
TRANSPORT_MARKET_SCOPE: Final[str] = "NFL_FULL_GAME_WINNER"
TRANSPORT_OUTCOME_SCOPE: Final[str] = "HOME_TEAM_OUTCOME"
DESCRIPTIVE_TRANSPORT_SCOPE: Final[str] = (
    "DESCRIPTIVE_COMPLETED_NON_TIE_PRODUCER_TRANSPORT"
)
FORMAL_CONTRACT_TRANSPORT_SCOPE: Final[str] = (
    "FORMAL_CONTRACT_EQUIVALENT_PRODUCER_TRANSPORT"
)
REALIZED_OUTCOME_COMPARABLE: Final[str] = (
    "COMPLETED_NON_TIE_REALIZATION_COMPARABLE"
)
REALIZED_OUTCOME_UNPROVEN: Final[str] = "UNPROVEN_NOT_ADJUDICATED"
STRICT_CONTRACT_RULE_EQUIVALENT: Final[str] = (
    "CONTRACT_RULE_EQUIVALENT"
)
STRICT_CONTRACT_RULE_UNPROVEN: Final[str] = "UNPROVEN"
NATIVE_RULE_OVERLAY_MEMBERSHIP_VERIFIED: Final[str] = "VERIFIED"
NATIVE_RULE_OVERLAY_GATE_ONE_COMPARABLE: Final[str] = "COMPARABLE"
NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT: Final[str] = "EQUIVALENT"
NATIVE_RULE_OVERLAY_GATE_TWO_NON_EQUIVALENT: Final[str] = (
    "NON_EQUIVALENT"
)
NATIVE_RULE_OVERLAY_GATE_TWO_UNPROVEN: Final[str] = "UNPROVEN"
DEVELOPMENT_SPLIT: Final[str] = "DEVELOPMENT"
POLYMARKET: Final[str] = "POLYMARKET"
KALSHI: Final[str] = "KALSHI"
PRIMARY_ANCHOR_KIND: Final[str] = PANEL_ANCHOR_KIND
PRIMARY_ANALYSIS_ROLE: Final[str] = PANEL_ANALYSIS_ROLE
_FORBIDDEN_PRIMARY_FEATURE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "final_outcome_tags",
        "score_points",
        "turnover",
        "beneficiary_team",
        "review_result",
        "episode_type",
        "episode_status",
        "stage_a_eligible",
        "residual_diagnostic_eligible",
    }
)
PRIMARY_FEATURE_CONTRACT: Final[Mapping[str, object]] = MappingProxyType(
    dict(PANEL_PRIMARY_FEATURE_CONTRACT)
)
PRESTATE_CONTEXT_CONTRACT: Final[Mapping[str, object]] = MappingProxyType(
    dict(PANEL_PRESTATE_CONTEXT_CONTRACT)
)
MARKET_CONTINUITY_CONTRACT: Final[Mapping[str, object]] = MappingProxyType(
    dict(PANEL_MARKET_CONTINUITY_CONTRACT)
)
LANDMARK_SECONDS: Final[tuple[int, ...]] = PANEL_LANDMARK_SECONDS
ENDPOINT_SECONDS: Final[tuple[int, ...]] = PANEL_ENDPOINT_SECONDS
MODEL_IDS: Final[tuple[str, ...]] = (
    "b0_empirical_v1",
    "regularized_logistic_v1",
    "shallow_xgboost_v1",
)
SELECTABLE_CANDIDATES: Final[
    tuple[tuple[str, str], ...]
] = (
    ("regularized_logistic_v1", "D1"),
    ("regularized_logistic_v1", "D2"),
    ("regularized_logistic_v1", "D3"),
    ("shallow_xgboost_v1", "D1"),
    ("shallow_xgboost_v1", "D2"),
    ("shallow_xgboost_v1", "D3"),
)
SELECTION_BOOTSTRAP_SAMPLES: Final[int] = 10_000
SELECTION_BOOTSTRAP_SEED: Final[int] = 20260729
SELECTION_CONFIDENCE_LEVEL: Final[float] = 0.95
SELECTION_ANCHOR_LANDMARK_SECONDS: Final[int] = 3
SELECTION_ANCHOR_ENDPOINT_SECONDS: Final[int] = 30
SELECTION_MIN_ANCHOR_GAMES: Final[int] = 30
KALSHI_TRANSPORT_MIN_CLEAN_ANCHOR_GAMES: Final[int] = 30
TRAINING_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "fold_sha256",
    "availability_preprocessor_spec_sha256",
    "availability_preprocessor_training_sha256",
    "availability_preprocessor_fitted_sha256",
    "direction_preprocessor_spec_sha256",
    "direction_preprocessor_training_sha256",
    "direction_preprocessor_fitted_sha256",
    "availability_model_spec_sha256",
    "availability_model_training_sha256",
    "availability_model_fitted_sha256",
    "direction_model_spec_sha256",
    "direction_model_training_sha256",
    "direction_model_fitted_sha256",
    "availability_calibration_spec_sha256",
    "availability_calibration_training_sha256",
    "availability_calibrator_fitted_sha256",
    "direction_calibration_spec_sha256",
    "direction_calibration_training_sha256",
    "direction_calibrator_fitted_sha256",
    "availability_calibration_raw_training_sha256",
    "direction_calibration_raw_training_sha256",
    "availability_calibrator_base_preprocessor_sha256",
    "direction_calibrator_base_preprocessor_sha256",
    "availability_calibrator_base_model_sha256",
    "direction_calibrator_base_model_sha256",
    "producer_primary_snapshot_set_sha256",
    "calibration_primary_snapshot_set_sha256",
)

FEATURE_BLOCKS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        str(block_id): tuple(columns)
        for block_id, columns in PANEL_FEATURE_BLOCKS.items()
    }
)

_FOLD_IDS: Final[tuple[str, ...]] = tuple(
    f"fold_{index:02d}" for index in range(1, 6)
)
_MODEL_BLOCKS: Final[tuple[tuple[str, str], ...]] = (
    ("b0_empirical_v1", "D0"),
    *tuple(
        ("regularized_logistic_v1", block_id)
        for block_id in FEATURE_BLOCKS
    ),
    *tuple(
        ("shallow_xgboost_v1", block_id)
        for block_id in FEATURE_BLOCKS
    ),
)
CORRECTED_MODEL_CELLS: Final[tuple[tuple[str, str, str], ...]] = tuple(
    (fold_id, model_id, feature_block_id)
    for fold_id in _FOLD_IDS
    for model_id, feature_block_id in _MODEL_BLOCKS
)

REQUIRED_PANEL_COLUMNS: Final[tuple[str, ...]] = tuple(
    PANEL_FORMAL_MODEL_REQUIRED_COLUMNS
)
_FORBIDDEN_LEGACY_COLUMNS: Final[frozenset[str]] = frozenset(
    set(PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS)
    | {
        "s_h",
        "s_h_truth",
        "s_h_probability",
        "s_h_calibrated_probability",
        "o_h_given_s",
        "o_h_given_s_truth",
        "o_h_given_s_probability",
        "o_h_given_s_calibrated_probability",
        "decision_eligible",
        "finalized_episode_id",
        "episode_type",
        "episode_status",
        "status",
        "stage_a_eligible",
        "residual_diagnostic_eligible",
    }
)
_FORBIDDEN_RUN_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "s_h",
        "s_h_truth",
        "s_h_probability",
        "s_h_calibrated_probability",
        "o_h_given_s",
        "o_h_given_s_truth",
        "o_h_given_s_probability",
        "o_h_given_s_calibrated_probability",
    }
)
_CATEGORICAL_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "action_group",
        "possession_result_group",
        "score_result_group",
        "kick_result_group",
        "adjudication_group",
    }
)
_BOOLEAN_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "possession_is_home",
        "possession_is_home_missing",
        "down_missing",
        "distance_missing",
        "yardline_100_missing",
        "goal_to_go",
        "pre_red_zone",
        "p_before_home_missing",
        "provisional_turnover_observed",
        "actor_is_home",
        "beneficiary_is_home",
        "is_overtime",
    }
)
_PROBABILITY_EPSILON: Final[float] = 1e-6
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RegulationModelInputError(ValueError):
    """The corrected regulation-only Stage-B contract was violated."""


@dataclass(frozen=True, slots=True)
class RegulationModelRun:
    """One or more corrected OOF execution cells."""

    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    support_audit: pd.DataFrame
    run_config: Mapping[str, object]
    run_config_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedRegulationPanelManifest:
    """Fully verified PanelV4 batch and game-manifest authority."""

    output_root: Path
    batch_manifest_path: Path
    batch_manifest_file_sha256: str
    batch_sha256: str
    game_count: int
    panel_row_count: int
    loss_eligible_count: int
    cohort_authority_sha256: str
    cohort_mapping_sha256: str
    sources: Mapping[str, str]
    games: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class VerifiedRegulationPanelBatch:
    """One venue-scoped uncensored risk set from verified PanelV4 bytes."""

    frame: pd.DataFrame
    manifest: VerifiedRegulationPanelManifest
    source_row_count: int
    loaded_venue_scope: tuple[str, ...]
    loaded_row_count: int
    loss_eligible_count: int
    formal_input_verified: bool = True

    @property
    def batch_manifest_file_sha256(self) -> str:
        return self.manifest.batch_manifest_file_sha256


@dataclass(frozen=True, slots=True)
class FrozenPolymarketWinnerV2:
    """Selection result that may be transported to Kalshi without reranking."""

    model_id: str
    feature_block_id: str
    polymarket_selection_sha256: str
    anchor_kind: str
    polymarket_fold_training_hashes: Mapping[
        str, Mapping[str, str]
    ]


@dataclass(frozen=True, slots=True)
class PublishedRegulationSelectionLock:
    """Content-addressed terminal Polymarket selection decision."""

    manifest_path: Path
    manifest_file_sha256: str
    selection_lock_sha256: str
    decision_status: str
    winner_frozen: bool


@dataclass(frozen=True, slots=True)
class VerifiedRegulationSelectionLock:
    """Verified terminal selection lock; winner may intentionally be absent."""

    manifest_path: Path
    manifest_file_sha256: str
    selection_lock_sha256: str
    selection_sha256: str
    decision_status: str
    winner: FrozenPolymarketWinnerV2 | None
    kalshi_transport_allowed: bool
    baseline_fold_training_hashes: Mapping[
        str, Mapping[str, str]
    ]
    fold_definitions: Mapping[str, Mapping[str, object]]
    panel_batch_manifest_file_sha256: str
    panel_batch_sha256: str
    cohort_authority_sha256: str
    cohort_mapping_sha256: str
    landmark_seconds: tuple[int, ...]
    endpoint_seconds: tuple[int, ...]
    direction_threshold_probability: float
    target_version: str
    formal_lock_verified: bool = True


@dataclass(frozen=True, slots=True)
class VerifiedNativeRuleOverlay:
    """Immutable native-rule audit cross-binding for transport."""

    manifest_path: Path
    manifest_file_sha256: str
    batch_sha256: str
    source_capture_batch_file_sha256: str
    panel_batch_manifest_file_sha256: str
    panel_batch_sha256: str
    panel_cohort_authority_sha256: str
    panel_contract_authority_sha256: str
    game_count: int
    venue_evidence_row_count: int
    contract_pair_row_count: int
    completed_non_tie_comparable_count: int
    strict_rule_equivalent_count: int
    cross_binding_set_sha256: str
    bindings: tuple[Mapping[str, object], ...]
    formal_overlay_verified: bool = True


@dataclass(frozen=True, slots=True)
class KalshiTransportRequestV2:
    """Governance-only descriptive transport; never a venue validation."""

    training_venue: str
    evaluation_venue: str
    model_id: str
    feature_block_id: str
    polymarket_selection_sha256: str
    selection_lock_sha256: str
    anchor_kind: str
    target_version: str
    transport_scope: str
    realized_outcome_comparability_status: str
    realized_outcome_comparability_evidence_set_sha256: str
    strict_contract_rule_equivalence_status: str
    strict_contract_rule_equivalence_evidence_set_sha256: str | None
    native_rule_evidence_status_by_venue: Mapping[str, tuple[str, ...]]
    native_rule_evidence_set_sha256_by_venue: Mapping[str, str | None]
    formal_contract_transport_allowed: bool
    independent_sports_holdout_validation_allowed: bool = False
    venue_speed_claim_allowed: bool = False
    selection_allowed: bool = False


@dataclass(frozen=True, slots=True)
class PublishedRegulationOOFShard:
    """One complete, immutable corrected Stage-B execution cell."""

    manifest_path: Path
    manifest_file_sha256: str
    shard_sha256: str
    fold_id: str
    model_id: str
    feature_block_id: str
    oof_prediction_row_count: int


@dataclass(frozen=True, slots=True)
class VerifiedRegulationOOFShard:
    """Lightweight verified descriptor for one corrected OOF cell."""

    output_root: Path
    manifest_path: Path
    manifest_file_sha256: str
    shard_sha256: str
    cell: tuple[str, str, str]
    oof_prediction_row_count: int
    object_descriptors: Mapping[str, Mapping[str, object]]
    training_hashes: Mapping[str, str]
    support_metadata: Mapping[str, object]
    run_config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PublishedRegulationOOFBatch:
    """Immutable partial or complete index over verified OOF shards."""

    manifest_path: Path
    manifest_file_sha256: str
    batch_sha256: str
    run_status: str
    publication_gate: str
    completed_cell_count: int
    selection_ready: bool


@dataclass(frozen=True, slots=True)
class RegulationSelectionCandidate:
    """Paired candidate-versus-B0 evidence across the five OOF folds."""

    model_id: str
    feature_block_id: str
    game_count: int
    integrated_joint_mean_improvement: float
    integrated_joint_standard_error: float
    integrated_joint_ci_low: float
    integrated_joint_ci_high: float
    availability_log_loss_ci_low: float
    availability_brier_ci_low: float
    direction_log_loss_ci_low: float
    direction_brier_ci_low: float
    anchor_game_count: int
    anchor_joint_mean_improvement: float
    anchor_direction_mean_improvement: float
    anchor_direction_brier_mean_improvement: float
    availability_non_regression: bool
    anchor_gate_passed: bool
    candidate_gate_passed: bool


@dataclass(frozen=True, slots=True)
class RegulationSelectionResult:
    """Frozen Polymarket-only one-standard-error decision."""

    decision_status: str
    winner: FrozenPolymarketWinnerV2 | None
    candidate_audit: tuple[RegulationSelectionCandidate, ...]
    best_integrated_mean_improvement: float | None
    best_standard_error: float | None
    one_se_threshold: float | None
    selection_sha256: str
    holdout_reaction_accessed: bool = False


@dataclass(frozen=True, slots=True)
class RegulationKalshiTransportRun:
    """Descriptive frozen-producer transport to one Kalshi risk set."""

    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    support_audit: pd.DataFrame
    transport_risk_set: pd.DataFrame
    descriptive_pair_coverage: pd.DataFrame
    run_config: Mapping[str, object]
    run_config_sha256: str


@dataclass(slots=True)
class _TwoHeadPredictor:
    availability_transformer: "_WeightedFeatureTransformer | None"
    direction_transformer: "_WeightedFeatureTransformer | None"
    availability_model: object
    direction_model: object
    availability_preprocessor_spec_sha256: str
    availability_preprocessor_training_sha256: str
    availability_preprocessor_fitted_sha256: str
    direction_preprocessor_spec_sha256: str
    direction_preprocessor_training_sha256: str
    direction_preprocessor_fitted_sha256: str
    availability_model_spec_sha256: str
    availability_model_training_sha256: str
    availability_model_fitted_sha256: str
    direction_model_spec_sha256: str
    direction_model_training_sha256: str
    direction_model_fitted_sha256: str


class _EmpiricalBinary:
    def __init__(self) -> None:
        self.global_probability = 0.5
        self.by_cell: dict[tuple[int, int], float] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        truth: np.ndarray,
        weights: np.ndarray,
    ) -> "_EmpiricalBinary":
        self.global_probability = _weighted_binary_mean(truth, weights)
        for key, positions in frame.groupby(
            ["landmark_seconds", "endpoint_seconds"],
            sort=True,
        ).indices.items():
            indices = np.asarray(positions, dtype=int)
            self.by_cell[(int(key[0]), int(key[1]))] = (
                _weighted_binary_mean(truth[indices], weights[indices])
            )
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                self.by_cell.get(
                    (int(landmark), int(endpoint)),
                    self.global_probability,
                )
                for landmark, endpoint in frame[
                    ["landmark_seconds", "endpoint_seconds"]
                ].itertuples(index=False, name=None)
            ],
            dtype=float,
        )


class _EmpiricalDirection:
    def __init__(self) -> None:
        self.global_probability = np.full(3, 1 / 3, dtype=float)
        self.by_cell: dict[tuple[int, int], np.ndarray] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        truth: np.ndarray,
        weights: np.ndarray,
    ) -> "_EmpiricalDirection":
        self.global_probability = _weighted_direction_mean(truth, weights)
        for key, positions in frame.groupby(
            ["landmark_seconds", "endpoint_seconds"],
            sort=True,
        ).indices.items():
            indices = np.asarray(positions, dtype=int)
            self.by_cell[(int(key[0]), int(key[1]))] = (
                _weighted_direction_mean(truth[indices], weights[indices])
            )
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return np.vstack(
            [
                self.by_cell.get(
                    (int(landmark), int(endpoint)),
                    self.global_probability,
                )
                for landmark, endpoint in frame[
                    ["landmark_seconds", "endpoint_seconds"]
                ].itertuples(index=False, name=None)
            ]
        )


class _ConstantBinary:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probability = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack((1 - probability, probability))


class _ConstantDirection:
    def __init__(self, probability: np.ndarray) -> None:
        self.probability = np.asarray(probability, dtype=float)
        self.classes_ = np.asarray(DIRECTION_CLASSES, dtype=object)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.tile(self.probability, (len(matrix), 1))


@dataclass(frozen=True, slots=True)
class _EncodedDirectionEstimator:
    """Expose XGBoost's encoded classes as the frozen semantic labels."""

    estimator: object
    classes_: np.ndarray

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(matrix), dtype=float)


@dataclass(frozen=True, slots=True)
class _WeightedFeatureTransformer:
    """Deterministic hierarchical-weighted imputation and scaling."""

    columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_impute: Mapping[str, float]
    numeric_mean: Mapping[str, float]
    numeric_scale: Mapping[str, float]
    categorical_levels: Mapping[str, tuple[str, ...]]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        coerced = _coerce_feature_frame(frame, self.columns)
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            numeric = np.column_stack(
                [
                    pd.to_numeric(coerced[column], errors="coerce")
                    .fillna(self.numeric_impute[column])
                    .to_numpy(dtype=float)
                    for column in self.numeric_columns
                ]
            )
            mean = np.asarray(
                [self.numeric_mean[column] for column in self.numeric_columns],
                dtype=float,
            )
            scale = np.asarray(
                [self.numeric_scale[column] for column in self.numeric_columns],
                dtype=float,
            )
            blocks.append((numeric - mean) / scale)
        for column in self.categorical_columns:
            values = (
                coerced[column]
                .astype("object")
                .where(coerced[column].notna(), "MISSING")
                .astype(str)
                .to_numpy(dtype=object)
            )
            levels = self.categorical_levels[column]
            blocks.append(
                np.column_stack(
                    [values == level for level in levels]
                ).astype(float)
            )
        if not blocks:
            return np.empty((len(frame), 0), dtype=float)
        return np.concatenate(blocks, axis=1)


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list, set, np.ndarray)):
        children = list(value)
        if isinstance(value, set):
            children = sorted(children, key=repr)
        return [_canonical(child) for child in children]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return str(value)


def _sha256(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_package_version(distribution: str) -> str:
    try:
        value = package_version(distribution)
    except PackageNotFoundError as error:
        raise RegulationModelInputError(
            f"required producer package is not installed: {distribution}"
        ) from error
    if not value:
        raise RegulationModelInputError(
            f"required producer package has no version: {distribution}"
        )
    return value


def _preprocessor_spec_sha256(
    columns: Sequence[str],
    *,
    head: str,
) -> str:
    """Bind the exact preprocessing policy, feature order, and runtime."""

    if head not in {"availability", "direction"}:
        raise RegulationModelInputError(
            f"unsupported preprocessing head: {head}"
        )
    ordered = tuple(str(column) for column in columns)
    return _sha256(
        {
            "schema": "RegulationStageBPreprocessorSpecV1",
            "head": head,
            "risk_set": (
                "ALL_UNCENSORED_PRIMARY_ROWS"
                if head == "availability"
                else "AVAILABILITY_H_EQUALS_ONE_ROWS_ONLY"
            ),
            "feature_columns": ordered,
            "numeric_columns": tuple(
                column
                for column in ordered
                if column not in _CATEGORICAL_FEATURES
            ),
            "categorical_columns": tuple(
                column
                for column in ordered
                if column in _CATEGORICAL_FEATURES
            ),
            "numeric_imputation": "HIERARCHICAL_WEIGHTED_MEDIAN",
            "numeric_center": "HIERARCHICAL_WEIGHTED_MEAN",
            "numeric_scale": "HIERARCHICAL_WEIGHTED_STANDARD_DEVIATION",
            "zero_or_nonfinite_scale": 1.0,
            "categorical_missing_level": "MISSING",
            "categorical_encoding": "SORTED_FIXED_ONE_HOT",
            "duplicate_collapse": (
                "GAME+INFORMATION_ANCHOR+FEATURES+TRUTH;"
                "SUM_HIERARCHICAL_WEIGHTS"
            ),
            "weighting": (
                "RECOMPUTED_WITHIN_HEAD_RISK_SET;"
                "ROW_TO_INFORMATION_ANCHOR_TO_GAME"
            ),
            "runtime_version_policy": "EXACT_DISTRIBUTION_VERSION",
            "runtime_versions": {
                "numpy": _required_package_version("numpy"),
                "pandas": _required_package_version("pandas"),
            },
        }
    )


def _model_spec_sha256(
    *,
    model_id: str,
    direction: bool,
    random_state: int,
) -> str:
    """Bind exact model parameters and package versions for one head."""

    head = "direction" if direction else "availability"
    if model_id == "b0_empirical_v1":
        implementation = {
            "estimator": (
                "WEIGHTED_EMPIRICAL_DIRECTION_BY_L_H"
                if direction
                else "WEIGHTED_EMPIRICAL_BINARY_BY_L_H"
            ),
            "probability_clip_epsilon": _PROBABILITY_EPSILON,
        }
        versions = {
            "numpy": _required_package_version("numpy"),
            "pandas": _required_package_version("pandas"),
        }
    elif model_id == "regularized_logistic_v1":
        implementation = {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "params": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1_000,
                "random_state": int(random_state),
            },
            "direction_classes": (
                list(DIRECTION_CLASSES) if direction else None
            ),
        }
        versions = {
            "scikit-learn": _required_package_version("scikit-learn"),
            "numpy": _required_package_version("numpy"),
        }
    elif model_id == "shallow_xgboost_v1":
        implementation = {
            "estimator": "xgboost.XGBClassifier",
            "params": {
                "n_estimators": 16,
                "max_depth": 2,
                "learning_rate": 0.08,
                "min_child_weight": 3,
                "subsample": 1.0,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": int(random_state),
                "eval_metric": "mlogloss" if direction else "logloss",
            },
            "direction_encoding": (
                "CONTIGUOUS_PRESENT_CLASSES_IN_FROZEN_SEMANTIC_ORDER"
                if direction
                else None
            ),
            "direction_classes": (
                list(DIRECTION_CLASSES) if direction else None
            ),
        }
        versions = {
            "xgboost": _required_package_version("xgboost"),
            "scikit-learn": _required_package_version("scikit-learn"),
            "numpy": _required_package_version("numpy"),
        }
    else:
        raise RegulationModelInputError(
            f"unsupported model spec: {model_id}"
        )
    return _sha256(
        {
            "schema": "RegulationStageBModelSpecV1",
            "model_id": model_id,
            "head": head,
            "implementation": implementation,
            "random_state_contract": (
                "EFFECTIVE_RANDOM_STATE_SHA256_BOUND_TO_BASE_SEED+FOLD+"
                "TRAINING_VENUE+EVALUATION_VENUE+TRANSPORT_MODE+"
                "FEATURE_BLOCK+MODEL+PURPOSE"
            ),
            "runtime_version_policy": "EXACT_DISTRIBUTION_VERSION",
            "runtime_versions": versions,
        }
    )


def _calibration_spec_sha256(*, head: str, model_id: str) -> str:
    if model_id == "b0_empirical_v1":
        return _sha256(
            {
                "schema": "RegulationStageBCalibrationSpecV1",
                "head": head,
                "model_id": model_id,
                "calibration": "NONE_EMPIRICAL_BASELINE",
            }
        )
    if head == "availability":
        implementation: Mapping[str, object] = {
            "calibrator": "BINARY_PLATT_LOGISTIC",
            "params": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1_000,
                "random_state": 20260728,
            },
        }
    elif head == "direction":
        implementation = {
            "calibrator": "MULTICLASS_TEMPERATURE_FIXED_GRID",
            "temperature_grid": tuple(float(value) for value in TEMPERATURE_GRID),
            "direction_classes": tuple(DIRECTION_CLASSES),
        }
    else:
        raise RegulationModelInputError(
            f"unsupported calibration head: {head}"
        )
    return _sha256(
        {
            "schema": "RegulationStageBCalibrationSpecV1",
            "head": head,
            "model_id": model_id,
            "support_min_rows": CALIBRATION_MIN_ROWS,
            "support_min_games": CALIBRATION_MIN_GAMES,
            "probability_epsilon": CALIBRATION_PROBABILITY_EPSILON,
            "implementation": implementation,
            "split": (
                "BASE_FIT_BEFORE_LAST_OUTER_TRAIN_WEEK;"
                "CALIBRATE_ON_LAST_OUTER_TRAIN_WEEK;"
                "NO_POST_CALIBRATION_REFIT"
            ),
            "weighting": "ROW_TO_INFORMATION_ANCHOR_TO_GAME",
            "runtime_version_policy": "EXACT_DISTRIBUTION_VERSION",
            "runtime_versions": {
                "scikit-learn": _required_package_version("scikit-learn"),
                "numpy": _required_package_version("numpy"),
            },
        }
    )


def _preprocessor_fitted_sha256(
    transformer: "_WeightedFeatureTransformer | None",
    *,
    model_id: str,
    head: str,
) -> str:
    if head not in {"availability", "direction"}:
        raise RegulationModelInputError(
            f"unsupported fitted preprocessing head: {head}"
        )
    if transformer is None:
        if model_id != "b0_empirical_v1":
            raise RegulationModelInputError(
                "non-empirical producer lacks a fitted transformer"
            )
        return _sha256(
            {
                "schema": "RegulationStageBPreprocessorArtifactV1",
                "model_id": model_id,
                "head": head,
                "transformer": None,
            }
        )
    return _sha256(
        {
            "schema": "RegulationStageBPreprocessorArtifactV1",
            "model_id": model_id,
            "head": head,
            "columns": transformer.columns,
            "numeric_columns": transformer.numeric_columns,
            "categorical_columns": transformer.categorical_columns,
            "numeric_impute": transformer.numeric_impute,
            "numeric_mean": transformer.numeric_mean,
            "numeric_scale": transformer.numeric_scale,
            "categorical_levels": transformer.categorical_levels,
        }
    )


def _fitted_estimator_sha256(
    estimator: object,
    *,
    model_id: str,
    head: str,
) -> str:
    material: dict[str, object] = {
        "schema": "RegulationStageBModelArtifactV1",
        "model_id": model_id,
        "head": head,
    }
    if isinstance(estimator, _EmpiricalBinary):
        material["artifact"] = {
            "kind": "EMPIRICAL_BINARY",
            "global_probability": estimator.global_probability,
            "by_cell": {
                f"{key[0]}:{key[1]}": value
                for key, value in sorted(estimator.by_cell.items())
            },
        }
    elif isinstance(estimator, _EmpiricalDirection):
        material["artifact"] = {
            "kind": "EMPIRICAL_DIRECTION",
            "global_probability": estimator.global_probability,
            "by_cell": {
                f"{key[0]}:{key[1]}": value
                for key, value in sorted(estimator.by_cell.items())
            },
        }
    elif isinstance(estimator, _ConstantBinary):
        material["artifact"] = {
            "kind": "CONSTANT_BINARY",
            "probability": estimator.probability,
        }
    elif isinstance(estimator, _ConstantDirection):
        material["artifact"] = {
            "kind": "CONSTANT_DIRECTION",
            "probability": estimator.probability,
            "classes": estimator.classes_,
        }
    else:
        wrapped = estimator
        semantic_classes: object | None = None
        if isinstance(estimator, _EncodedDirectionEstimator):
            wrapped = estimator.estimator
            semantic_classes = estimator.classes_
        if isinstance(wrapped, LogisticRegression):
            material["artifact"] = {
                "kind": "SKLEARN_LOGISTIC",
                "coef": wrapped.coef_,
                "intercept": wrapped.intercept_,
                "classes": (
                    semantic_classes
                    if semantic_classes is not None
                    else wrapped.classes_
                ),
                "native_classes": wrapped.classes_,
                "n_iter": wrapped.n_iter_,
            }
        elif hasattr(wrapped, "get_booster"):
            booster = wrapped.get_booster()
            raw = booster.save_raw(raw_format="json")
            material["artifact"] = {
                "kind": "XGBOOST_JSON",
                "booster_sha256": (
                    "sha256:" + hashlib.sha256(bytes(raw)).hexdigest()
                ),
                "semantic_classes": semantic_classes,
                "native_classes": getattr(wrapped, "classes_", None),
                "n_features_in": getattr(wrapped, "n_features_in_", None),
            }
        else:
            raise RegulationModelInputError(
                f"unsupported fitted estimator artifact: {type(estimator)!r}"
            )
    return _sha256(material)


def _fitted_calibrator_sha256(
    calibrator: BinaryPlattCalibrator
    | DirectionTemperatureCalibrator
    | None,
    *,
    head: str,
) -> str:
    if calibrator is None:
        return _sha256(
            {
                "schema": "RegulationStageBCalibratorArtifactV1",
                "head": head,
                "status": "NOT_APPLICABLE_EMPIRICAL_BASELINE",
                "artifact": None,
            }
        )
    return _sha256(
        {
            "schema": "RegulationStageBCalibratorArtifactV1",
            "head": head,
            "artifact": asdict(calibrator),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_project_path(
    project_root: Path,
    value: Path,
    *,
    label: str,
) -> Path:
    project = project_root.resolve()
    candidate = value if value.is_absolute() else project / value
    if candidate.is_symlink():
        raise RegulationModelInputError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project)
    except ValueError as error:
        raise RegulationModelInputError(
            f"{label} escapes project root"
        ) from error
    return resolved


def _verified_json_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Mapping[str, object]:
    if _SHA256_RE.fullmatch(str(expected_sha256)) is None:
        raise RegulationModelInputError(
            f"{label} expected SHA-256 is invalid"
        )
    if not path.is_file() or path.is_symlink():
        raise RegulationModelInputError(
            f"{label} is not a regular file"
        )
    if _sha256_file(path) != expected_sha256:
        raise RegulationModelInputError(f"{label} file hash mismatch")
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, child in pairs:
            if key in result:
                raise RegulationModelInputError(
                    f"{label} contains duplicate JSON key {key}"
                )
            result[key] = child
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegulationModelInputError(
            f"{label} is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise RegulationModelInputError(
            f"{label} must contain a JSON object"
        )
    return value


def _manifest_output_root(batch_path: Path) -> Path:
    # <root>/batches/manifests/sha256/<aa>/<digest>.batch-index.json
    parents = batch_path.parents
    if (
        len(parents) < 5
        or parents[0].name not in {
            batch_path.name[:2],
            batch_path.name.split(".", 1)[0][:2],
        }
        or parents[1].name != "sha256"
        or parents[2].name != "manifests"
        or parents[3].name != "batches"
    ):
        raise RegulationModelInputError(
            "PanelV4 batch manifest path is not content-addressed"
        )
    return parents[4]


def _require_sha_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise RegulationModelInputError(
            f"{label} must be a nonempty hash mapping"
        )
    result = {str(key): str(child) for key, child in value.items()}
    if any(_SHA256_RE.fullmatch(child) is None for child in result.values()):
        raise RegulationModelInputError(
            f"{label} contains an invalid SHA-256"
        )
    return result


def _require_nested_training_hashes(
    value: object,
    *,
    label: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(_FOLD_IDS):
        raise RegulationModelInputError(
            f"{label} must contain the exact five folds"
        )
    result: dict[str, dict[str, str]] = {}
    for fold_id in _FOLD_IDS:
        child = value.get(fold_id)
        if (
            not isinstance(child, dict)
            or set(child) != set(TRAINING_PROVENANCE_COLUMNS)
        ):
            raise RegulationModelInputError(
                f"{label} {fold_id} has incomplete producer provenance"
            )
        hashes = {str(key): str(item) for key, item in child.items()}
        if any(
            _SHA256_RE.fullmatch(item) is None
            for item in hashes.values()
        ):
            raise RegulationModelInputError(
                f"{label} {fold_id} contains invalid SHA-256"
            )
        result[fold_id] = hashes
    return result


def verify_regulation_panel_batch_manifest(
    *,
    project_root: Path,
    batch_manifest_path: Path,
    batch_manifest_file_sha256: str,
    expected_game_count: int = 153,
) -> VerifiedRegulationPanelManifest:
    """Verify the batch and all game manifests without reading table bytes."""

    if (
        isinstance(expected_game_count, bool)
        or not isinstance(expected_game_count, int)
        or expected_game_count < 1
    ):
        raise RegulationModelInputError(
            "expected_game_count must be positive"
        )
    project = Path(project_root).resolve()
    path = _safe_project_path(
        project,
        Path(batch_manifest_path),
        label="PanelV4 batch manifest",
    )
    document = _verified_json_file(
        path,
        expected_sha256=batch_manifest_file_sha256,
        label="PanelV4 batch manifest",
    )
    if (
        document.get("schema") != PANEL_BATCH_MANIFEST_SCHEMA
        or document.get("dataset_split") != DEVELOPMENT_SPLIT
        or document.get("holdout_reaction_accessed") is not False
        or document.get("regulation_only") is not True
        or document.get("publication_gate") != "PASS"
        or _canonical(document.get("primary_feature_contract"))
        != _canonical(PRIMARY_FEATURE_CONTRACT)
        or _canonical(document.get("prestate_context_contract"))
        != _canonical(PRESTATE_CONTEXT_CONTRACT)
        or _canonical(document.get("market_continuity_contract"))
        != _canonical(MARKET_CONTINUITY_CONTRACT)
        or document.get("formal_model_required_columns")
        != list(REQUIRED_PANEL_COLUMNS)
        or document.get("formal_model_forbidden_columns")
        != sorted(PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS)
    ):
        raise RegulationModelInputError(
            "PanelV4 batch governance contract failed"
        )
    anchor_kind = document.get("anchor_kind")
    analysis_role = document.get("analysis_role")
    if (
        anchor_kind != PRIMARY_ANCHOR_KIND
        or analysis_role != PRIMARY_ANALYSIS_ROLE
    ):
        raise RegulationModelInputError(
            "PanelV4 formal batch must be primary provisional-only"
        )
    batch_sha256 = str(document.get("batch_sha256", ""))
    material = {
        key: value
        for key, value in document.items()
        if key != "batch_sha256"
    }
    if (
        _SHA256_RE.fullmatch(batch_sha256) is None
        or _sha256(material) != batch_sha256
    ):
        raise RegulationModelInputError(
            "PanelV4 batch semantic hash mismatch"
        )
    if document.get("game_count") != expected_game_count:
        raise RegulationModelInputError(
            "PanelV4 batch game_count mismatch"
        )
    expected_time_grid = {
        "landmark_seconds": list(LANDMARK_SECONDS),
        "endpoint_seconds": list(ENDPOINT_SECONDS),
        "require_h_gt_l": True,
    }
    if (
        document.get("time_grid") != expected_time_grid
        or document.get("direction_threshold_probability") != 0.01
        or document.get("feature_blocks")
        != {key: list(value) for key, value in FEATURE_BLOCKS.items()}
    ):
        raise RegulationModelInputError(
            "PanelV4 time, threshold, or feature contract drifted"
        )
    sources = _require_sha_mapping(
        document.get("sources"), label="PanelV4 batch sources"
    )
    for field in ("cohort_authority_sha256", "cohort_mapping_sha256"):
        if field not in sources:
            raise RegulationModelInputError(
                f"PanelV4 batch sources missing {field}"
            )
    counts = document.get("counts")
    if not isinstance(counts, dict):
        raise RegulationModelInputError("PanelV4 batch counts are missing")
    try:
        panel_row_count = int(counts["panel_row_count"])
        loss_eligible_count = int(counts["loss_eligible_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise RegulationModelInputError(
            "PanelV4 batch row counts are invalid"
        ) from error
    if panel_row_count < 1 or not 0 < loss_eligible_count <= panel_row_count:
        raise RegulationModelInputError(
            "PanelV4 batch row counts are inconsistent"
        )
    game_entries = document.get("games")
    if (
        not isinstance(game_entries, list)
        or len(game_entries) != expected_game_count
    ):
        raise RegulationModelInputError(
            "PanelV4 game manifest index is incomplete"
        )
    output_root = _manifest_output_root(path)
    verified_games: list[Mapping[str, object]] = []
    seen_games: set[str] = set()
    summed_rows = 0
    for entry in game_entries:
        if not isinstance(entry, dict):
            raise RegulationModelInputError(
                "PanelV4 game index entry is malformed"
            )
        game_id = str(entry.get("game_id", ""))
        if not game_id or game_id in seen_games:
            raise RegulationModelInputError(
                "PanelV4 game index duplicates identity"
            )
        manifest_relative = Path(str(entry.get("manifest_path", "")))
        manifest_path = _safe_project_path(
            project,
            output_root / manifest_relative,
            label=f"PanelV4 {game_id} manifest",
        )
        manifest_sha256 = str(entry.get("manifest_sha256", ""))
        game_document = _verified_json_file(
            manifest_path,
            expected_sha256=manifest_sha256,
            label=f"PanelV4 {game_id} manifest",
        )
        if (
            game_document.get("schema") != PANEL_GAME_MANIFEST_SCHEMA
            or game_document.get("game_id") != game_id
            or game_document.get("dataset_split") != DEVELOPMENT_SPLIT
            or game_document.get("holdout_reaction_accessed") is not False
            or game_document.get("regulation_only") is not True
            or game_document.get("publication_gate") != "PASS"
            or _canonical(
                game_document.get("primary_feature_contract")
            )
            != _canonical(PRIMARY_FEATURE_CONTRACT)
            or _canonical(
                game_document.get("prestate_context_contract")
            )
            != _canonical(PRESTATE_CONTEXT_CONTRACT)
            or _canonical(
                game_document.get("market_continuity_contract")
            )
            != _canonical(MARKET_CONTINUITY_CONTRACT)
            or game_document.get("formal_model_required_columns")
            != list(REQUIRED_PANEL_COLUMNS)
            or game_document.get("formal_model_forbidden_columns")
            != sorted(PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS)
            or game_document.get("feature_blocks")
            != {
                key: list(value)
                for key, value in FEATURE_BLOCKS.items()
            }
        ):
            raise RegulationModelInputError(
                f"PanelV4 {game_id} governance contract failed"
            )
        if (
            game_document.get("anchor_kind") != anchor_kind
            or game_document.get("analysis_role") != analysis_role
        ):
            raise RegulationModelInputError(
                f"PanelV4 {game_id} anchor/analysis-role contract drifted"
            )
        game_bundle_sha256 = str(
            game_document.get("bundle_sha256", "")
        )
        game_material = {
            key: value
            for key, value in game_document.items()
            if key != "bundle_sha256"
        }
        if (
            _sha256(game_material) != game_bundle_sha256
            or entry.get("bundle_sha256") != game_bundle_sha256
        ):
            raise RegulationModelInputError(
                f"PanelV4 {game_id} semantic hash mismatch"
            )
        game_source_hashes = _require_sha_mapping(
            game_document.get("source_hashes"),
            label=f"PanelV4 {game_id} sources",
        )
        for game_field, batch_field in (
            (
                "information_anchor_manifest_sha256",
                "information_anchor_manifest_file_sha256",
            ),
            (
                "information_anchor_object_sha256",
                "information_anchor_object_sha256",
            ),
            (
                "context_manifest_sha256",
                "context_manifest_file_sha256",
            ),
            ("context_object_sha256", "context_object_sha256"),
            ("cohort_authority_sha256", "cohort_authority_sha256"),
        ):
            if game_source_hashes.get(game_field) != sources.get(
                batch_field
            ):
                raise RegulationModelInputError(
                    f"PanelV4 {game_id} source lineage mismatch"
                )
        tables = game_document.get("tables")
        if not isinstance(tables, list):
            raise RegulationModelInputError(
                f"PanelV4 {game_id} tables are missing"
            )
        panel_descriptors = [
            descriptor
            for descriptor in tables
            if isinstance(descriptor, dict)
            and descriptor.get("name") == "regulation_decision_panel"
        ]
        if len(panel_descriptors) != 1:
            raise RegulationModelInputError(
                f"PanelV4 {game_id} panel descriptor is not unique"
            )
        descriptor = panel_descriptors[0]
        if (
            descriptor.get("object_sha256")
            != entry.get("panel_object_sha256")
            or descriptor.get("row_count") != entry.get("panel_row_count")
            or not isinstance(descriptor.get("schema_columns"), list)
            or not set(REQUIRED_PANEL_COLUMNS).issubset(
                descriptor["schema_columns"]
            )
            or any(
                str(column).startswith(("final_", "finalized_"))
                or column in _FORBIDDEN_PRIMARY_FEATURE_NAMES
                or column in _FORBIDDEN_LEGACY_COLUMNS
                for column in descriptor["schema_columns"]
            )
        ):
            raise RegulationModelInputError(
                f"PanelV4 {game_id} descriptor conflicts with batch"
            )
        try:
            rows = int(descriptor["row_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise RegulationModelInputError(
                f"PanelV4 {game_id} row count is invalid"
            ) from error
        summed_rows += rows
        verified_games.append(
            MappingProxyType(
                {
                    **entry,
                    "manifest_path_absolute": str(manifest_path),
                    "panel_descriptor": MappingProxyType(dict(descriptor)),
                    "source_hashes": MappingProxyType(game_source_hashes),
                }
            )
        )
        seen_games.add(game_id)
    if summed_rows != panel_row_count:
        raise RegulationModelInputError(
            "PanelV4 game row counts do not reconcile to batch"
        )
    return VerifiedRegulationPanelManifest(
        output_root=output_root,
        batch_manifest_path=path,
        batch_manifest_file_sha256=batch_manifest_file_sha256,
        batch_sha256=batch_sha256,
        game_count=expected_game_count,
        panel_row_count=panel_row_count,
        loss_eligible_count=loss_eligible_count,
        cohort_authority_sha256=sources["cohort_authority_sha256"],
        cohort_mapping_sha256=sources["cohort_mapping_sha256"],
        sources=MappingProxyType(sources),
        games=tuple(verified_games),
    )


_VERIFIED_PANEL_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "nfl_week",
    "game_id",
    "information_anchor",
    "information_anchor_id",
    "episode_id",
    "anchor_kind",
    "venue",
    "actual_home_contract_id",
    "logical_market_id",
    "market_family",
    "market_period",
    "proposition_kind",
    "home_team",
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
    "landmark_seconds",
    "endpoint_seconds",
    "source_row_id",
)


def load_verified_regulation_panel_batch(
    *,
    project_root: Path,
    batch_manifest_path: Path,
    batch_manifest_file_sha256: str,
    venue_scope: Sequence[str],
    expected_game_count: int = 153,
) -> VerifiedRegulationPanelBatch:
    """Load one explicit formal venue scope through one Arrow→pandas boundary."""

    project = Path(project_root).resolve()
    scope = tuple(str(venue).upper() for venue in venue_scope)
    if scope not in {
        (POLYMARKET,),
        (POLYMARKET, KALSHI),
    }:
        raise RegulationModelInputError(
            "venue_scope must be exactly (POLYMARKET,) for selection or "
            "(POLYMARKET, KALSHI) for transport"
        )
    manifest = verify_regulation_panel_batch_manifest(
        project_root=project,
        batch_manifest_path=batch_manifest_path,
        batch_manifest_file_sha256=batch_manifest_file_sha256,
        expected_game_count=expected_game_count,
    )
    projected_tables: list[pa.Table] = []
    source_row_count = 0
    loaded_row_count = 0
    for game in manifest.games:
        descriptor = game["panel_descriptor"]
        if not isinstance(descriptor, Mapping):
            raise AssertionError("verified panel descriptor disappeared")
        object_path = _safe_project_path(
            project,
            manifest.output_root
            / Path(str(descriptor["object_path"])),
            label=f"PanelV4 {game['game_id']} object",
        )
        object_sha256 = str(descriptor["object_sha256"])
        if (
            not object_path.is_file()
            or object_path.is_symlink()
            or _sha256_file(object_path) != object_sha256
        ):
            raise RegulationModelInputError(
                f"PanelV4 {game['game_id']} object hash mismatch"
            )
        if object_path.stat().st_size != int(descriptor["byte_length"]):
            raise RegulationModelInputError(
                f"PanelV4 {game['game_id']} object byte length mismatch"
            )
        try:
            persisted_schema = pq.read_schema(object_path)
            persisted_row_count = pq.ParquetFile(
                object_path
            ).metadata.num_rows
        except Exception as error:
            raise RegulationModelInputError(
                f"PanelV4 {game['game_id']} Parquet is unreadable"
            ) from error
        schema_fingerprint = "sha256:" + hashlib.sha256(
            persisted_schema.remove_metadata().serialize().to_pybytes()
        ).hexdigest()
        if (
            persisted_row_count != int(descriptor["row_count"])
            or persisted_schema.names != descriptor["schema_columns"]
            or schema_fingerprint != descriptor["schema_fingerprint"]
        ):
            raise RegulationModelInputError(
                f"PanelV4 {game['game_id']} table contract mismatch"
            )
        try:
            table = pq.read_table(
                object_path,
                columns=list(REQUIRED_PANEL_COLUMNS),
                filters=[
                    ("loss_eligible", "=", True),
                    ("venue", "in", list(scope)),
                ],
            )
        except Exception as error:
            raise RegulationModelInputError(
                f"PanelV4 {game['game_id']} projection is unreadable"
            ) from error
        if set(table.column("game_id").unique().to_pylist()) != {
            str(game["game_id"])
        }:
            raise RegulationModelInputError(
                f"PanelV4 {game['game_id']} object crosses games"
            )
        source_row_count += persisted_row_count
        loaded_row_count += table.num_rows
        projected_tables.append(table)
    if source_row_count != manifest.panel_row_count:
        raise RegulationModelInputError(
            "PanelV4 loaded rows do not match batch count"
        )
    if not projected_tables:
        raise RegulationModelInputError(
            "PanelV4 has no loss-eligible rows in the requested venue scope"
        )
    combined = pa.concat_tables(projected_tables)
    projected_tables.clear()
    source = combined.to_pandas()
    del combined
    frame = prepare_regulation_panel(source)
    del source
    if (
        len(frame) != loaded_row_count
        or frame["game_id"].nunique() != manifest.game_count
        or set(frame["venue"].astype(str)) != set(scope)
        or (
            scope == (POLYMARKET, KALSHI)
            and len(frame) != manifest.loss_eligible_count
        )
    ):
        raise RegulationModelInputError(
            "PanelV4 scoped risk set does not reconcile to batch"
        )
    frame.sort_values(
        list(_VERIFIED_PANEL_SORT_COLUMNS),
        kind="mergesort",
        inplace=True,
    )
    frame.reset_index(drop=True, inplace=True)
    return VerifiedRegulationPanelBatch(
        frame=frame,
        manifest=manifest,
        source_row_count=source_row_count,
        loaded_venue_scope=scope,
        loaded_row_count=loaded_row_count,
        loss_eligible_count=len(frame),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    payload = sink.getvalue().to_pybytes()
    persisted_schema = pq.read_schema(pa.BufferReader(payload))
    return (
        payload,
        "sha256:"
        + hashlib.sha256(
            persisted_schema.remove_metadata().serialize().to_pybytes()
        ).hexdigest(),
    )


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise RegulationModelInputError(
                f"content-addressed collision: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RegulationModelInputError(
                    f"content-addressed publish race: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _publish_model_object(
    *,
    output_root: Path,
    cell: tuple[str, str, str],
    logical_name: str,
    payload: bytes,
    suffix: str,
    row_count: int,
    schema_fingerprint: str | None = None,
) -> dict[str, object]:
    object_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    hexadecimal = object_sha256.removeprefix("sha256:")
    fold_id, model_id, feature_block_id = cell
    path = (
        output_root
        / "shards"
        / fold_id
        / model_id
        / feature_block_id
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.{suffix}"
    )
    _atomic_publish(path, payload)
    descriptor: dict[str, object] = {
        "logical_name": logical_name,
        "object_path": str(path.relative_to(output_root)),
        "object_sha256": object_sha256,
        "byte_length": len(payload),
        "row_count": int(row_count),
        "format": suffix,
    }
    if schema_fingerprint is not None:
        descriptor["schema_fingerprint"] = schema_fingerprint
    return descriptor


def publish_regulation_oof_shard(
    *,
    model_run: RegulationModelRun,
    output_root: Path,
    verified_panel: VerifiedRegulationPanelBatch | None = None,
) -> PublishedRegulationOOFShard:
    """Publish exactly one fully verified execution cell atomically."""

    if (
        verified_panel is None
        or not verified_panel.formal_input_verified
        or verified_panel.manifest.game_count != 153
        or verified_panel.frame["game_id"].nunique() != 153
        or verified_panel.loaded_venue_scope != (POLYMARKET,)
        or len(verified_panel.frame) != verified_panel.loaded_row_count
        or len(verified_panel.frame) != verified_panel.loss_eligible_count
        or not verified_panel.frame["venue"].eq(POLYMARKET).all()
    ):
        raise RegulationModelInputError(
            "OOF shard publication requires a verified PanelV4 exact-153 batch"
        )
    if model_run.run_config.get("formal_input_verified") is not True:
        raise RegulationModelInputError(
            "OOF shard run was not fitted from verified PanelV4"
        )
    lineage = model_run.run_config.get("input_lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("panel_batch_manifest_file_sha256")
        != verified_panel.manifest.batch_manifest_file_sha256
        or lineage.get("panel_batch_sha256")
        != verified_panel.manifest.batch_sha256
    ):
        raise RegulationModelInputError(
            "OOF shard run lineage differs from verified PanelV4"
        )
    support = model_run.support_audit
    if len(support) != 1:
        raise RegulationModelInputError(
            "OOF shard publication requires exactly one execution cell"
        )
    support_row = support.iloc[0].to_dict()
    cell = (
        str(support_row["fold_id"]),
        str(support_row["model_id"]),
        str(support_row["feature_block_id"]),
    )
    if cell not in CORRECTED_MODEL_CELLS:
        raise RegulationModelInputError(
            "OOF shard cell is outside the corrected matrix"
        )
    predictions = model_run.oof_predictions
    metrics = model_run.fold_metrics
    for frame, label in (
        (predictions, "predictions"),
        (metrics, "metrics"),
        (support, "support"),
    ):
        if frame.empty:
            raise RegulationModelInputError(
                f"OOF shard {label} is empty"
            )
        observed_cells = set(
            zip(
                frame["fold_id"].astype(str),
                frame["model_id"].astype(str),
                frame["feature_block_id"].astype(str),
                strict=True,
            )
        )
        if observed_cells != {cell}:
            raise RegulationModelInputError(
                f"OOF shard {label} crosses execution cells"
            )
    if (
        not predictions["venue"].eq(POLYMARKET).all()
        or not predictions["anchor_kind"].eq(PRIMARY_ANCHOR_KIND).all()
        or not predictions["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE).all()
    ):
        raise RegulationModelInputError(
            "OOF shard contains non-primary or non-Polymarket rows"
        )
    fold = _fold_map(
        verified_panel.frame.loc[
            verified_panel.frame["venue"].eq(POLYMARKET)
            & verified_panel.frame["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
            & verified_panel.frame["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
        ]
    )[cell[0]]
    if set(predictions["game_id"].astype(str)) != set(
        fold.validation_game_ids
    ):
        raise RegulationModelInputError(
            "OOF shard validation games differ from frozen fold"
        )
    expected_fold_sha256 = _sha256(
        {
            "fold_id": fold.fold_id,
            "train_weeks": fold.train_weeks,
            "validation_weeks": fold.validation_weeks,
            "train_game_ids": fold.train_game_ids,
            "validation_game_ids": fold.validation_game_ids,
        }
    )
    if (
        support_row.get("fold_sha256") != expected_fold_sha256
        or not predictions["fold_sha256"].eq(expected_fold_sha256).all()
    ):
        raise RegulationModelInputError(
            "OOF shard fold hash differs from frozen fold"
        )
    output = Path(output_root).resolve()
    if output.is_symlink():
        raise RegulationModelInputError(
            "OOF output root must not be a symlink"
        )
    descriptors: list[dict[str, object]] = []
    for logical_name, frame in (
        ("oof_predictions", predictions),
        ("fold_metrics", metrics),
        ("support_audit", support),
    ):
        payload, schema_fingerprint = _parquet_bytes(frame)
        descriptors.append(
            _publish_model_object(
                output_root=output,
                cell=cell,
                logical_name=logical_name,
                payload=payload,
                suffix="parquet",
                row_count=len(frame),
                schema_fingerprint=schema_fingerprint,
            )
        )
    run_config_payload = _canonical_json_bytes(dict(model_run.run_config))
    descriptors.append(
        _publish_model_object(
            output_root=output,
            cell=cell,
            logical_name="run_config",
            payload=run_config_payload,
            suffix="json",
            row_count=1,
        )
    )
    training_hashes = {
        key: str(support_row[key])
        for key in TRAINING_PROVENANCE_COLUMNS
    }
    if any(
        _SHA256_RE.fullmatch(value) is None
        for value in training_hashes.values()
    ):
        raise RegulationModelInputError(
            "OOF shard training provenance hash is invalid"
        )
    shard_material = {
        "schema": OOF_SHARD_MANIFEST_SCHEMA,
        "cell": {
            "fold_id": cell[0],
            "model_id": cell[1],
            "feature_block_id": cell[2],
        },
        "input": {
            "panel_batch_manifest_path": str(
                verified_panel.manifest.batch_manifest_path
            ),
            "panel_batch_manifest_file_sha256": (
                verified_panel.manifest.batch_manifest_file_sha256
            ),
            "panel_batch_sha256": verified_panel.manifest.batch_sha256,
            "cohort_authority_sha256": (
                verified_panel.manifest.cohort_authority_sha256
            ),
            "cohort_mapping_sha256": (
                verified_panel.manifest.cohort_mapping_sha256
            ),
            "source_hashes": dict(verified_panel.manifest.sources),
        },
        "selection_venue": POLYMARKET,
        "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
        "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
        "primary_feature_contract": dict(PRIMARY_FEATURE_CONTRACT),
        "prestate_context_contract": dict(PRESTATE_CONTEXT_CONTRACT),
        "market_continuity_contract": dict(MARKET_CONTINUITY_CONTRACT),
        "formal_model_required_columns": list(REQUIRED_PANEL_COLUMNS),
        "formal_model_forbidden_columns": sorted(
            PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS
        ),
        "cross_anchor_pooling_allowed": False,
        "kalshi_selection_allowed": False,
        "holdout_reaction_accessed": False,
        "run_config_sha256": model_run.run_config_sha256,
        "training_hashes": training_hashes,
        "objects": descriptors,
        "status": "COMPLETE_CELL",
        "publication_gate": "PASS",
    }
    shard_sha256 = _sha256(shard_material)
    manifest = {**shard_material, "shard_sha256": shard_sha256}
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_file_sha256 = (
        "sha256:" + hashlib.sha256(manifest_payload).hexdigest()
    )
    hexadecimal = manifest_file_sha256.removeprefix("sha256:")
    manifest_path = (
        output
        / "shards"
        / cell[0]
        / cell[1]
        / cell[2]
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.shard-manifest.json"
    )
    _atomic_publish(manifest_path, manifest_payload)
    return PublishedRegulationOOFShard(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
        shard_sha256=shard_sha256,
        fold_id=cell[0],
        model_id=cell[1],
        feature_block_id=cell[2],
        oof_prediction_row_count=len(predictions),
    )


def _safe_under(root: Path, value: Path, *, label: str) -> Path:
    base = root.resolve()
    candidate = value if value.is_absolute() else base / value
    if candidate.is_symlink():
        raise RegulationModelInputError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise RegulationModelInputError(
            f"{label} escapes its governed root"
        ) from error
    return resolved


def _verify_content_addressed_name(
    path: Path,
    *,
    file_sha256: str,
    suffix: str,
    label: str,
) -> None:
    hexadecimal = file_sha256.removeprefix("sha256:")
    if (
        path.name != f"{hexadecimal}{suffix}"
        or path.parent.name != hexadecimal[:2]
        or path.parent.parent.name != "sha256"
    ):
        raise RegulationModelInputError(
            f"{label} path is not content-addressed"
        )


def _load_verified_shard_object(
    *,
    output_root: Path,
    descriptor: Mapping[str, object],
    logical_name: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame | Mapping[str, object]:
    path, suffix = _verify_shard_object_descriptor(
        output_root=output_root,
        descriptor=descriptor,
        logical_name=logical_name,
    )
    if suffix == "json":
        if columns is not None:
            raise RegulationModelInputError(
                f"OOF shard JSON object {logical_name} cannot be projected"
            )
        return _verified_json_file(
            path,
            expected_sha256=str(descriptor["object_sha256"]),
            label=f"OOF shard object {logical_name}",
        )
    try:
        table = pq.read_table(path, columns=columns)
    except Exception as error:
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} Parquet is unreadable"
        ) from error
    return table.to_pandas()


def _verify_shard_object_descriptor(
    *,
    output_root: Path,
    descriptor: Mapping[str, object],
    logical_name: str,
) -> tuple[Path, str]:
    """Verify governed bytes and persisted metadata without materializing rows."""

    if descriptor.get("logical_name") != logical_name:
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} identity mismatch"
        )
    object_sha256 = str(descriptor.get("object_sha256", ""))
    if _SHA256_RE.fullmatch(object_sha256) is None:
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} SHA-256 is invalid"
        )
    suffix = str(descriptor.get("format", ""))
    if suffix not in {"parquet", "json"}:
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} format is invalid"
        )
    path = _safe_under(
        output_root,
        Path(str(descriptor.get("object_path", ""))),
        label=f"OOF shard object {logical_name}",
    )
    _verify_content_addressed_name(
        path,
        file_sha256=object_sha256,
        suffix=f".{suffix}",
        label=f"OOF shard object {logical_name}",
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != object_sha256
        or path.stat().st_size != int(descriptor.get("byte_length", -1))
    ):
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} byte contract failed"
        )
    if suffix == "json":
        if int(descriptor.get("row_count", -1)) != 1:
            raise RegulationModelInputError(
                f"OOF shard object {logical_name} JSON row count is invalid"
            )
        return path, suffix
    try:
        schema = pq.read_schema(path)
        row_count = pq.ParquetFile(path).metadata.num_rows
    except Exception as error:
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} Parquet is unreadable"
        ) from error
    schema_fingerprint = "sha256:" + hashlib.sha256(
        schema.remove_metadata().serialize().to_pybytes()
    ).hexdigest()
    if (
        row_count != int(descriptor.get("row_count", -1))
        or descriptor.get("schema_fingerprint") != schema_fingerprint
    ):
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} table contract failed"
        )
    return path, suffix


_PREDICTION_PANEL_IDENTITY_MAP: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            "source_row_id": "source_row_id",
            "game_id": "game_id",
            "nfl_week": "nfl_week",
            "information_anchor_id": "information_anchor_id",
            "episode_id": "episode_id",
            "anchor_kind": "anchor_kind",
            "analysis_role": "analysis_role",
            "venue": "venue",
            "actual_home_contract_id": "actual_home_contract_id",
            "logical_market_id": "logical_market_id",
            "market_family": "market_family",
            "market_period": "market_period",
            "proposition_kind": "proposition_kind",
            "home_team": "home_team",
            "outcome_team": "outcome_team",
            "actual_home_contract_identity_sha256": (
                "actual_home_contract_identity_sha256"
            ),
            "native_raw_manifest_sha256s_json": (
                "native_raw_manifest_sha256s_json"
            ),
            "native_raw_object_sha256s_json": (
                "native_raw_object_sha256s_json"
            ),
            "native_raw_manifest_set_sha256": (
                "native_raw_manifest_set_sha256"
            ),
            "native_raw_object_set_sha256": (
                "native_raw_object_set_sha256"
            ),
            "native_rule_evidence_sha256": "native_rule_evidence_sha256",
            "native_rule_evidence_status": "native_rule_evidence_status",
            "realized_outcome_comparability_status": (
                "realized_outcome_comparability_status"
            ),
            "realized_outcome_comparability_evidence_sha256": (
                "realized_outcome_comparability_evidence_sha256"
            ),
            "strict_contract_rule_equivalence_status": (
                "strict_contract_rule_equivalence_status"
            ),
            "strict_contract_rule_equivalence_evidence_sha256": (
                "strict_contract_rule_equivalence_evidence_sha256"
            ),
            "landmark_seconds": "landmark_seconds",
            "endpoint_seconds": "endpoint_seconds",
            "availability_truth": "availability_h",
            "direction_truth": "direction_h",
            "cohort_authority_sha256": "cohort_authority_sha256",
            "source_interval_start": "source_interval_start",
            "source_interval_end": "source_interval_end",
            "information_anchor": "information_anchor",
            "primary_feature_known_at": "primary_feature_known_at",
            "primary_feature_snapshot_sha256": (
                "primary_feature_snapshot_sha256"
            ),
            "prestate_known_at": "prestate_known_at",
            "prestate_context_sha256": "prestate_context_sha256",
            "p_before_support_status": "p_before_support_status",
        }
    )
)
_PREDICTION_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "availability_probability",
    "direction_probability_down",
    "direction_probability_no_move",
    "direction_probability_up",
)
_PREDICTION_LOSS_COLUMNS: Final[tuple[str, ...]] = (
    "availability_log_loss",
    "availability_brier",
    "direction_log_loss",
    "direction_multiclass_brier",
    "joint_log_loss",
)


def _load_verified_parquet_projection(
    *,
    output_root: Path,
    descriptor: Mapping[str, object],
    logical_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Load only a governed projection; callers must discard it immediately."""

    path, suffix = _verify_shard_object_descriptor(
        output_root=output_root,
        descriptor=descriptor,
        logical_name=logical_name,
    )
    if suffix != "parquet":
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} must be Parquet"
        )
    try:
        return pq.read_table(path, columns=list(dict.fromkeys(columns))).to_pandas()
    except Exception as error:
        raise RegulationModelInputError(
            f"OOF shard object {logical_name} projection is unreadable"
        ) from error


def _semantic_series_equal(
    observed: pd.Series,
    expected: pd.Series,
) -> bool:
    """Compare persisted identity/truth values without dtype coercion drift."""

    if pd.api.types.is_datetime64_any_dtype(observed.dtype) or (
        pd.api.types.is_datetime64_any_dtype(expected.dtype)
    ):
        left = pd.to_datetime(observed, utc=True, errors="coerce")
        right = pd.to_datetime(expected, utc=True, errors="coerce")
        return bool(
            np.array_equal(
                left.astype("int64").to_numpy(),
                right.astype("int64").to_numpy(),
            )
        )
    if pd.api.types.is_numeric_dtype(observed.dtype) and (
        pd.api.types.is_numeric_dtype(expected.dtype)
    ):
        return bool(
            np.array_equal(
                pd.to_numeric(observed, errors="coerce").to_numpy(),
                pd.to_numeric(expected, errors="coerce").to_numpy(),
                equal_nan=True,
            )
        )
    missing = "<X15_NULL>"
    left = observed.astype("string").fillna(missing).to_numpy(dtype=str)
    right = expected.astype("string").fillna(missing).to_numpy(dtype=str)
    return bool(np.array_equal(left, right))


def _expected_fold_support(
    *,
    primary_panel: pd.DataFrame,
    fold: X15WeekFold,
    model_id: str,
    feature_block_id: str,
) -> Mapping[str, object]:
    outer_train = primary_panel.loc[
        primary_panel["game_id"].isin(fold.train_game_ids)
    ]
    validation = primary_panel.loc[
        primary_panel["game_id"].isin(fold.validation_game_ids)
    ]
    if model_id == "b0_empirical_v1":
        producer = outer_train
        calibration = outer_train.iloc[0:0]
    else:
        final_train_week = int(outer_train["nfl_week"].max())
        producer = outer_train.loc[
            outer_train["nfl_week"].lt(final_train_week)
        ]
        calibration = outer_train.loc[
            outer_train["nfl_week"].eq(final_train_week)
        ]
    random_state = effective_random_state(
        base_random_state=RANDOM_STATE,
        fold_id=fold.fold_id,
        training_venue=POLYMARKET,
        evaluation_venue=POLYMARKET,
        transport_mode="NATIVE_POLYMARKET_SELECTION",
        feature_block_id=feature_block_id,
        model_id=model_id,
        purpose="REGULATION_TWO_HEAD_FIT",
    )
    columns = FEATURE_BLOCKS[feature_block_id]
    calibration_empty_sha = _sha256(
        {"scheme": "NONE_EMPIRICAL_BASELINE", "rows": 0}
    )
    return MappingProxyType(
        {
            "outer_training_game_count": len(fold.train_game_ids),
            "producer_training_game_count": int(
                producer["game_id"].nunique()
            ),
            "calibration_game_count": int(
                calibration["game_id"].nunique()
            ),
            "validation_game_count": len(fold.validation_game_ids),
            "outer_training_row_count": len(outer_train),
            "producer_training_row_count": len(producer),
            "calibration_row_count": len(calibration),
            "validation_row_count": len(validation),
            "availability_training_positive_count": int(
                producer["availability_h"].sum()
            ),
            "direction_training_row_count": int(
                producer["availability_h"].sum()
            ),
            "producer_training_game_ids": json.dumps(
                sorted(producer["game_id"].astype(str).unique()),
                separators=(",", ":"),
            ),
            "calibration_game_ids": json.dumps(
                sorted(calibration["game_id"].astype(str).unique()),
                separators=(",", ":"),
            ),
            "producer_max_week": int(producer["nfl_week"].max()),
            "calibration_week": (
                None
                if calibration.empty
                else int(calibration["nfl_week"].min())
            ),
            "effective_random_state": random_state,
            "availability_preprocessor_spec_sha256": (
                _preprocessor_spec_sha256(columns, head="availability")
            ),
            "direction_preprocessor_spec_sha256": (
                _preprocessor_spec_sha256(columns, head="direction")
            ),
            "availability_model_spec_sha256": _model_spec_sha256(
                model_id=model_id,
                direction=False,
                random_state=random_state,
            ),
            "direction_model_spec_sha256": _model_spec_sha256(
                model_id=model_id,
                direction=True,
                random_state=random_state,
            ),
            "availability_calibration_spec_sha256": (
                _calibration_spec_sha256(
                    head="availability", model_id=model_id
                )
            ),
            "direction_calibration_spec_sha256": (
                _calibration_spec_sha256(
                    head="direction", model_id=model_id
                )
            ),
            "producer_primary_snapshot_set_sha256": (
                _primary_snapshot_set_sha256(producer)
            ),
            "calibration_primary_snapshot_set_sha256": (
                calibration_empty_sha
                if calibration.empty
                else _primary_snapshot_set_sha256(calibration)
            ),
            "availability_calibration_status": (
                "NOT_APPLICABLE_EMPIRICAL_BASELINE"
                if model_id == "b0_empirical_v1"
                else CALIBRATED
            ),
            "direction_calibration_status": (
                "NOT_APPLICABLE_EMPIRICAL_BASELINE"
                if model_id == "b0_empirical_v1"
                else CALIBRATED
            ),
        }
    )


def _verify_prediction_projection(
    *,
    output_root: Path,
    descriptor: Mapping[str, object],
    verified_panel: VerifiedRegulationPanelBatch,
    cell: tuple[str, str, str],
    training_hashes: Mapping[str, str],
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Rejoin persisted predictions to PanelV4 and rederive truth/provenance."""

    columns = (
        *tuple(_PREDICTION_PANEL_IDENTITY_MAP),
        "fold_id",
        "model_id",
        "feature_block_id",
        "training_venue",
        "evaluation_venue",
        "transport_mode",
        "availability_calibration_status",
        "direction_calibration_status",
        "panel_batch_sha256",
        *_PREDICTION_PROBABILITY_COLUMNS,
        *_PREDICTION_LOSS_COLUMNS,
        *TRAINING_PROVENANCE_COLUMNS,
        *extra_columns,
    )
    predictions = _load_verified_parquet_projection(
        output_root=output_root,
        descriptor=descriptor,
        logical_name="oof_predictions",
        columns=columns,
    )
    if (
        predictions.empty
        or predictions["source_row_id"].duplicated().any()
        or set(
            zip(
                predictions["fold_id"].astype(str),
                predictions["model_id"].astype(str),
                predictions["feature_block_id"].astype(str),
                strict=True,
            )
        )
        != {cell}
        or not predictions["venue"].eq(POLYMARKET).all()
        or not predictions["anchor_kind"].eq(PRIMARY_ANCHOR_KIND).all()
        or not predictions["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE).all()
        or not predictions["training_venue"].eq(POLYMARKET).all()
        or not predictions["evaluation_venue"].eq(POLYMARKET).all()
        or not predictions["transport_mode"].eq(
            "NATIVE_POLYMARKET_SELECTION"
        ).all()
        or not predictions["availability_calibration_status"].eq(
            (
                "NOT_APPLICABLE_EMPIRICAL_BASELINE"
                if cell[1] == "b0_empirical_v1"
                else CALIBRATED
            )
        ).all()
        or not predictions["direction_calibration_status"].eq(
            (
                "NOT_APPLICABLE_EMPIRICAL_BASELINE"
                if cell[1] == "b0_empirical_v1"
                else CALIBRATED
            )
        ).all()
    ):
        raise RegulationModelInputError(
            "OOF shard prediction population is not one primary cell"
        )
    primary_panel = verified_panel.frame.loc[
        verified_panel.frame["venue"].eq(POLYMARKET)
        & verified_panel.frame["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & verified_panel.frame["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ].reset_index(drop=True)
    fold = _fold_map(primary_panel)[cell[0]]
    expected_fold_sha256 = _sha256(
        {
            "fold_id": fold.fold_id,
            "train_weeks": fold.train_weeks,
            "validation_weeks": fold.validation_weeks,
            "train_game_ids": fold.train_game_ids,
            "validation_game_ids": fold.validation_game_ids,
        }
    )
    expected = primary_panel.loc[
        primary_panel["game_id"].isin(fold.validation_game_ids),
        list(dict.fromkeys(_PREDICTION_PANEL_IDENTITY_MAP.values())),
    ].copy()
    if len(predictions) != len(expected):
        raise RegulationModelInputError(
            "OOF shard prediction row count differs from frozen validation"
        )
    observed_ordered = predictions.sort_values(
        "source_row_id", kind="mergesort"
    ).reset_index(drop=True)
    expected_ordered = expected.sort_values(
        "source_row_id", kind="mergesort"
    ).reset_index(drop=True)
    for observed_column, panel_column in _PREDICTION_PANEL_IDENTITY_MAP.items():
        if not _semantic_series_equal(
            observed_ordered[observed_column],
            expected_ordered[panel_column],
        ):
            raise RegulationModelInputError(
                "OOF shard prediction identity/truth differs from PanelV4: "
                f"{observed_column}"
            )
    if (
        not observed_ordered["panel_batch_sha256"].eq(
            verified_panel.manifest.batch_sha256
        ).all()
        or training_hashes.get("fold_sha256") != expected_fold_sha256
        or not observed_ordered["fold_sha256"].eq(
            expected_fold_sha256
        ).all()
        or any(
            not observed_ordered[column].astype(str).eq(str(value)).all()
            for column, value in training_hashes.items()
        )
    ):
        raise RegulationModelInputError(
            "OOF shard prediction lineage/provenance differs from authority"
        )
    direction_probability = observed_ordered[
        list(_PREDICTION_PROBABILITY_COLUMNS[1:])
    ].to_numpy(dtype=float)
    availability_probability = observed_ordered[
        "availability_probability"
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(availability_probability).all()
        or not np.isfinite(direction_probability).all()
        or np.any((availability_probability < 0) | (availability_probability > 1))
        or np.any((direction_probability < 0) | (direction_probability > 1))
        or not np.allclose(
            direction_probability.sum(axis=1),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise RegulationModelInputError(
            "OOF shard probabilities violate the persisted simplex"
        )
    rederived_losses = regulation_row_losses(
        availability_truth=observed_ordered[
            "availability_truth"
        ].to_numpy(dtype=int),
        availability_probability=availability_probability,
        direction_truth=observed_ordered[
            "direction_truth"
        ].to_numpy(dtype=object),
        direction_probability=direction_probability,
    )
    for column in _PREDICTION_LOSS_COLUMNS:
        if not np.allclose(
            pd.to_numeric(
                observed_ordered[column], errors="coerce"
            ).to_numpy(dtype=float),
            rederived_losses[column].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RegulationModelInputError(
                f"OOF shard {column} does not rederive"
            )
    return observed_ordered


def verify_regulation_oof_shard(
    *,
    project_root: Path,
    output_root: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
    verified_panel: VerifiedRegulationPanelBatch,
) -> VerifiedRegulationOOFShard:
    """Verify one resumable shard against the exact PanelV4 authority."""

    if (
        not verified_panel.formal_input_verified
        or verified_panel.manifest.game_count != 153
        or verified_panel.frame["game_id"].nunique() != 153
        or verified_panel.loaded_venue_scope != (POLYMARKET,)
        or len(verified_panel.frame) != verified_panel.loaded_row_count
        or len(verified_panel.frame) != verified_panel.loss_eligible_count
        or not verified_panel.frame["venue"].eq(POLYMARKET).all()
    ):
        raise RegulationModelInputError(
            "OOF shard verification requires verified PanelV4 exact-153"
        )
    project = Path(project_root).resolve()
    output = _safe_project_path(
        project, Path(output_root), label="OOF output root"
    )
    path = _safe_under(
        output, Path(manifest_path), label="OOF shard manifest"
    )
    _verify_content_addressed_name(
        path,
        file_sha256=manifest_file_sha256,
        suffix=".shard-manifest.json",
        label="OOF shard manifest",
    )
    document = _verified_json_file(
        path,
        expected_sha256=manifest_file_sha256,
        label="OOF shard manifest",
    )
    if (
        document.get("schema") != OOF_SHARD_MANIFEST_SCHEMA
        or document.get("status") != "COMPLETE_CELL"
        or document.get("publication_gate") != "PASS"
        or document.get("selection_venue") != POLYMARKET
        or document.get("selection_anchor_kind") != PRIMARY_ANCHOR_KIND
        or document.get("selection_analysis_role")
        != PRIMARY_ANALYSIS_ROLE
        or _canonical(document.get("primary_feature_contract"))
        != _canonical(PRIMARY_FEATURE_CONTRACT)
        or _canonical(document.get("prestate_context_contract"))
        != _canonical(PRESTATE_CONTEXT_CONTRACT)
        or _canonical(document.get("market_continuity_contract"))
        != _canonical(MARKET_CONTINUITY_CONTRACT)
        or document.get("formal_model_required_columns")
        != list(REQUIRED_PANEL_COLUMNS)
        or document.get("formal_model_forbidden_columns")
        != sorted(PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS)
        or document.get("cross_anchor_pooling_allowed") is not False
        or document.get("kalshi_selection_allowed") is not False
        or document.get("holdout_reaction_accessed") is not False
    ):
        raise RegulationModelInputError(
            "OOF shard governance contract failed"
        )
    shard_sha256 = str(document.get("shard_sha256", ""))
    shard_material = {
        key: value
        for key, value in document.items()
        if key != "shard_sha256"
    }
    if (
        _SHA256_RE.fullmatch(shard_sha256) is None
        or _sha256(shard_material) != shard_sha256
    ):
        raise RegulationModelInputError(
            "OOF shard semantic hash mismatch"
        )
    cell_document = document.get("cell")
    if not isinstance(cell_document, dict):
        raise RegulationModelInputError("OOF shard cell is malformed")
    cell = (
        str(cell_document.get("fold_id", "")),
        str(cell_document.get("model_id", "")),
        str(cell_document.get("feature_block_id", "")),
    )
    if cell not in CORRECTED_MODEL_CELLS:
        raise RegulationModelInputError(
            "OOF shard cell is outside corrected matrix"
        )
    input_document = document.get("input")
    if (
        not isinstance(input_document, dict)
        or input_document.get("panel_batch_manifest_file_sha256")
        != verified_panel.manifest.batch_manifest_file_sha256
        or input_document.get("panel_batch_sha256")
        != verified_panel.manifest.batch_sha256
        or input_document.get("cohort_authority_sha256")
        != verified_panel.manifest.cohort_authority_sha256
        or input_document.get("cohort_mapping_sha256")
        != verified_panel.manifest.cohort_mapping_sha256
        or input_document.get("source_hashes")
        != dict(verified_panel.manifest.sources)
    ):
        raise RegulationModelInputError(
            "OOF shard PanelV4 lineage mismatch"
        )
    descriptors = document.get("objects")
    if not isinstance(descriptors, list) or len(descriptors) != 4:
        raise RegulationModelInputError(
            "OOF shard must contain exactly four governed objects"
        )
    by_name = {
        str(descriptor.get("logical_name")): descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    required_names = {
        "oof_predictions",
        "fold_metrics",
        "support_audit",
        "run_config",
    }
    if set(by_name) != required_names:
        raise RegulationModelInputError(
            "OOF shard object inventory is incomplete or duplicated"
        )
    for logical_name in ("oof_predictions", "fold_metrics"):
        _verify_shard_object_descriptor(
            output_root=output,
            descriptor=by_name[logical_name],
            logical_name=logical_name,
        )
    support = _load_verified_shard_object(
        output_root=output,
        descriptor=by_name["support_audit"],
        logical_name="support_audit",
    )
    run_config = _load_verified_shard_object(
        output_root=output,
        descriptor=by_name["run_config"],
        logical_name="run_config",
    )
    if (
        not isinstance(support, pd.DataFrame)
        or not isinstance(run_config, Mapping)
    ):
        raise AssertionError("verified OOF object type changed")
    if (
        _sha256(run_config) != document.get("run_config_sha256")
        or run_config.get("formal_input_verified") is not True
        or run_config.get("selection_venue") != POLYMARKET
        or run_config.get("selection_anchor_kind") != PRIMARY_ANCHOR_KIND
        or run_config.get("selection_analysis_role")
        != PRIMARY_ANALYSIS_ROLE
        or _canonical(run_config.get("primary_feature_contract"))
        != _canonical(PRIMARY_FEATURE_CONTRACT)
        or _canonical(run_config.get("prestate_context_contract"))
        != _canonical(PRESTATE_CONTEXT_CONTRACT)
        or _canonical(run_config.get("market_continuity_contract"))
        != _canonical(MARKET_CONTINUITY_CONTRACT)
        or run_config.get("formal_model_required_columns")
        != list(REQUIRED_PANEL_COLUMNS)
        or run_config.get("formal_model_forbidden_columns")
        != sorted(PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS)
        or run_config.get("cross_anchor_pooling_allowed") is not False
        or run_config.get("kalshi_selection_allowed") is not False
        or run_config.get("holdout_reaction_accessed") is not False
        or tuple(tuple(value) for value in run_config.get("cells", ()))
        != (cell,)
        or run_config.get("schema_version") != SCHEMA_VERSION
        or run_config.get("dataset_split") != DEVELOPMENT_SPLIT
        or tuple(run_config.get("targets", ()))
        != (
            "availability_h",
            "direction_h_given_availability",
        )
        or run_config.get("joint_loss")
        != (
            "NLL(availability_h)+"
            "I(availability_h)*NLL(direction_h)"
        )
        or run_config.get("censored_rows")
        != "EXCLUDED_FROM_BOTH_HEADS"
        or run_config.get("weighting_contract")
        != "ROWS_TO_PRIMARY_INFORMATION_ANCHOR_TO_GAME"
        or run_config.get("preprocessing_contract")
        != (
            "HIERARCHICAL_WEIGHTED_MEDIAN_IMPUTE_AND_SCALE; "
            "FIXED_MISSING_ONEHOT"
        )
        or _canonical(run_config.get("feature_blocks"))
        != _canonical(dict(FEATURE_BLOCKS))
        or int(run_config.get("random_state", -1)) != RANDOM_STATE
        or _canonical(run_config.get("calibration_contract"))
        != _canonical(
            {
                "b0_empirical_v1": (
                    "NO_LEARNED_CALIBRATION; FIT_ALL_OUTER_TRAIN_WEEKS"
                ),
                "regularized_logistic_v1": (
                    "BASE_FIT_BEFORE_LAST_OUTER_TRAIN_WEEK; "
                    "CALIBRATE_ON_LAST_OUTER_TRAIN_WEEK; "
                    "NO_POST_CALIBRATION_REFIT"
                ),
                "shallow_xgboost_v1": (
                    "BASE_FIT_BEFORE_LAST_OUTER_TRAIN_WEEK; "
                    "CALIBRATE_ON_LAST_OUTER_TRAIN_WEEK; "
                    "NO_POST_CALIBRATION_REFIT"
                ),
            }
        )
    ):
        raise RegulationModelInputError(
            "OOF shard run config contract failed"
        )
    run_lineage = run_config.get("input_lineage")
    if (
        not isinstance(run_lineage, Mapping)
        or run_lineage.get("panel_batch_manifest_file_sha256")
        != verified_panel.manifest.batch_manifest_file_sha256
        or run_lineage.get("panel_batch_sha256")
        != verified_panel.manifest.batch_sha256
        or run_lineage.get("panel_cohort_authority_sha256")
        != verified_panel.manifest.cohort_authority_sha256
        or run_lineage.get("panel_cohort_mapping_sha256")
        != verified_panel.manifest.cohort_mapping_sha256
        or run_lineage.get("panel_source_hashes")
        != dict(verified_panel.manifest.sources)
    ):
        raise RegulationModelInputError(
            "OOF shard run config lineage differs from PanelV4"
        )
    required_cell_columns = {
        "fold_id",
        "model_id",
        "feature_block_id",
    }
    if (
        support.empty
        or not required_cell_columns.issubset(support.columns)
        or set(
            zip(
                support["fold_id"].astype(str),
                support["model_id"].astype(str),
                support["feature_block_id"].astype(str),
                strict=True,
            )
        )
        != {cell}
        or len(support) != 1
    ):
        raise RegulationModelInputError(
            "OOF shard support does not represent exactly one cell"
        )
    training_hashes = document.get("training_hashes")
    if (
        not isinstance(training_hashes, dict)
        or set(training_hashes) != set(TRAINING_PROVENANCE_COLUMNS)
        or any(
            _SHA256_RE.fullmatch(str(value)) is None
            for value in training_hashes.values()
        )
        or any(
            str(support.iloc[0].get(key)) != str(value)
            for key, value in training_hashes.items()
        )
    ):
        raise RegulationModelInputError(
            "OOF shard training hashes disagree with support"
        )
    primary_panel = verified_panel.frame.loc[
        verified_panel.frame["venue"].eq(POLYMARKET)
        & verified_panel.frame["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & verified_panel.frame["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ].reset_index(drop=True)
    fold = _fold_map(primary_panel)[cell[0]]
    expected_fold_sha256 = _sha256(
        {
            "fold_id": fold.fold_id,
            "train_weeks": fold.train_weeks,
            "validation_weeks": fold.validation_weeks,
            "train_game_ids": fold.train_game_ids,
            "validation_game_ids": fold.validation_game_ids,
        }
    )
    expected_support = _expected_fold_support(
        primary_panel=primary_panel,
        fold=fold,
        model_id=cell[1],
        feature_block_id=cell[2],
    )
    support_row = support.iloc[0]
    if (
        str(support_row.get("fold_sha256")) != expected_fold_sha256
        or any(
            (
                not pd.isna(support_row.get(key))
                if expected is None
                else support_row.get(key) != expected
            )
            for key, expected in expected_support.items()
        )
    ):
        raise RegulationModelInputError(
            "OOF shard support membership/chronology differs from fold"
        )
    predictions = _verify_prediction_projection(
        output_root=output,
        descriptor=by_name["oof_predictions"],
        verified_panel=verified_panel,
        cell=cell,
        training_hashes=training_hashes,
    )
    metrics = _load_verified_parquet_projection(
        output_root=output,
        descriptor=by_name["fold_metrics"],
        logical_name="fold_metrics",
        columns=(
            "fold_id",
            "model_id",
            "feature_block_id",
            "metric_name",
            "metric_value",
        ),
    )
    if (
        metrics.empty
        or metrics["metric_name"].duplicated().any()
        or set(
            zip(
                metrics["fold_id"].astype(str),
                metrics["model_id"].astype(str),
                metrics["feature_block_id"].astype(str),
                strict=True,
            )
        )
        != {cell}
    ):
        raise RegulationModelInputError(
            "OOF shard metrics do not represent exactly one cell"
        )
    expected_metrics = compute_equal_game_metrics(predictions)
    observed_metrics = {
        str(row.metric_name): float(row.metric_value)
        for row in metrics.itertuples(index=False)
    }
    if set(observed_metrics) != set(expected_metrics) or any(
        not np.isclose(
            observed_metrics[key],
            expected,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        for key, expected in expected_metrics.items()
    ):
        raise RegulationModelInputError(
            "OOF shard equal-game metrics do not rederive"
        )
    if (
        support_row["availability_calibrator_base_preprocessor_sha256"]
        != support_row[
            "availability_preprocessor_training_sha256"
        ]
        or support_row["direction_calibrator_base_preprocessor_sha256"]
        != support_row["direction_preprocessor_training_sha256"]
        or support_row["availability_calibrator_base_model_sha256"]
        != support_row["availability_model_training_sha256"]
        or support_row["direction_calibrator_base_model_sha256"]
        != support_row["direction_model_training_sha256"]
    ):
        raise RegulationModelInputError(
            "OOF shard calibrator is not bound to its prediction producer"
        )
    return VerifiedRegulationOOFShard(
        output_root=output,
        manifest_path=path,
        manifest_file_sha256=manifest_file_sha256,
        shard_sha256=shard_sha256,
        cell=cell,
        oof_prediction_row_count=int(
            by_name["oof_predictions"]["row_count"]
        ),
        object_descriptors=MappingProxyType(
            {
                name: MappingProxyType(dict(descriptor))
                for name, descriptor in by_name.items()
            }
        ),
        training_hashes=MappingProxyType(
            {key: str(value) for key, value in training_hashes.items()}
        ),
        support_metadata=MappingProxyType(support_row.to_dict()),
        run_config=MappingProxyType(dict(run_config)),
    )


def discover_verified_regulation_oof_shards(
    *,
    project_root: Path,
    output_root: Path,
    verified_panel: VerifiedRegulationPanelBatch,
) -> tuple[VerifiedRegulationOOFShard, ...]:
    """Discover only hash-verified shards and reject duplicate cells."""

    project = Path(project_root).resolve()
    output = _safe_project_path(
        project, Path(output_root), label="OOF output root"
    )
    if not output.exists():
        return ()
    if not output.is_dir() or output.is_symlink():
        raise RegulationModelInputError(
            "OOF output root must be a regular directory"
        )
    manifests = sorted(
        output.glob(
            "shards/*/*/*/manifests/sha256/*/"
            "*.shard-manifest.json"
        )
    )
    verified: list[VerifiedRegulationOOFShard] = []
    by_cell: dict[tuple[str, str, str], Path] = {}
    for path in manifests:
        file_sha256 = _sha256_file(path)
        shard = verify_regulation_oof_shard(
            project_root=project,
            output_root=output,
            manifest_path=path,
            manifest_file_sha256=file_sha256,
            verified_panel=verified_panel,
        )
        previous = by_cell.get(shard.cell)
        if previous is not None and previous != shard.manifest_path:
            raise RegulationModelInputError(
                f"multiple verified OOF shards exist for cell {shard.cell}"
            )
        by_cell[shard.cell] = shard.manifest_path
        verified.append(shard)
    return tuple(
        sorted(
            verified,
            key=lambda shard: CORRECTED_MODEL_CELLS.index(shard.cell),
        )
    )


def publish_regulation_oof_batch_index(
    *,
    project_root: Path,
    output_root: Path,
    verified_panel: VerifiedRegulationPanelBatch,
    shards: Sequence[VerifiedRegulationOOFShard],
) -> PublishedRegulationOOFBatch:
    """Publish a resumable PARTIAL index or selection-ready COMPLETE index."""

    project = Path(project_root).resolve()
    output = _safe_project_path(
        project, Path(output_root), label="OOF output root"
    )
    cells = tuple(shard.cell for shard in shards)
    status = regulation_oof_batch_status(cells)
    if any(
        shard.run_config.get("formal_input_verified") is not True
        for shard in shards
    ):
        raise RegulationModelInputError(
            "OOF batch index accepts verified formal shards only"
        )
    shard_entries = [
        {
            "fold_id": shard.cell[0],
            "model_id": shard.cell[1],
            "feature_block_id": shard.cell[2],
            "manifest_path": str(
                shard.manifest_path.relative_to(output)
            ),
            "manifest_file_sha256": shard.manifest_file_sha256,
            "shard_sha256": shard.shard_sha256,
            "oof_prediction_row_count": shard.oof_prediction_row_count,
        }
        for shard in sorted(
            shards,
            key=lambda value: CORRECTED_MODEL_CELLS.index(value.cell),
        )
    ]
    material = {
        "schema": OOF_BATCH_MANIFEST_SCHEMA,
        "input": {
            "panel_batch_manifest_path": str(
                verified_panel.manifest.batch_manifest_path
            ),
            "panel_batch_manifest_file_sha256": (
                verified_panel.manifest.batch_manifest_file_sha256
            ),
            "panel_batch_sha256": verified_panel.manifest.batch_sha256,
            "cohort_authority_sha256": (
                verified_panel.manifest.cohort_authority_sha256
            ),
            "cohort_mapping_sha256": (
                verified_panel.manifest.cohort_mapping_sha256
            ),
        },
        "selection_venue": POLYMARKET,
        "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
        "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
        "primary_feature_contract": dict(PRIMARY_FEATURE_CONTRACT),
        "prestate_context_contract": dict(PRESTATE_CONTEXT_CONTRACT),
        "market_continuity_contract": dict(MARKET_CONTINUITY_CONTRACT),
        "formal_model_required_columns": list(REQUIRED_PANEL_COLUMNS),
        "formal_model_forbidden_columns": sorted(
            PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS
        ),
        "cross_anchor_pooling_allowed": False,
        "kalshi_selection_allowed": False,
        "holdout_reaction_accessed": False,
        "expected_cells": [list(cell) for cell in CORRECTED_MODEL_CELLS],
        "completed_cells": [
            list(cell) for cell in status["completed_cells"]
        ],
        "missing_cells": [list(cell) for cell in status["missing_cells"]],
        "completed_cell_count": status["completed_cell_count"],
        "expected_cell_count": status["expected_cell_count"],
        "run_status": status["run_status"],
        "selection_ready": status["run_status"] == "COMPLETE",
        "publication_gate": status["publication_gate"],
        "shards": shard_entries,
    }
    batch_sha256 = _sha256(material)
    document = {**material, "batch_sha256": batch_sha256}
    payload = _canonical_json_bytes(document)
    file_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    hexadecimal = file_sha256.removeprefix("sha256:")
    path = (
        output
        / "batches"
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.batch-index.json"
    )
    _atomic_publish(path, payload)
    return PublishedRegulationOOFBatch(
        manifest_path=path,
        manifest_file_sha256=file_sha256,
        batch_sha256=batch_sha256,
        run_status=str(status["run_status"]),
        publication_gate=str(status["publication_gate"]),
        completed_cell_count=int(status["completed_cell_count"]),
        selection_ready=status["run_status"] == "COMPLETE",
    )


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(
                child_key
                for child in value.values()
                for child_key in _mapping_keys(child)
            ),
        }
    if isinstance(value, (tuple, list)):
        return {
            child_key
            for child in value
            for child_key in _mapping_keys(child)
        }
    return set()


def _strict_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.isna().any():
        raise RegulationModelInputError(
            f"{column} must contain nonmissing booleans"
        )
    if not all(
        isinstance(value, (bool, np.bool_))
        for value in values.drop_duplicates()
    ):
        raise RegulationModelInputError(
            f"{column} must contain nonmissing booleans"
        )
    return values.astype(bool)


def _nullable_bool_numeric(
    frame: pd.DataFrame, column: str
) -> pd.Series:
    values = frame[column]
    supplied = values.notna()
    distinct = values.loc[supplied].drop_duplicates().tolist()
    valid = all(
        isinstance(value, (bool, np.bool_))
        or (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value))
            and float(value) in {0.0, 1.0}
        )
        for value in distinct
    )
    if not valid:
        raise RegulationModelInputError(
            f"{column} must contain booleans, exact 0/1, or null"
        )
    encoded = pd.to_numeric(values, errors="coerce")
    return encoded.astype("Float64")


def _validate_integer(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if (
        values.isna().any()
        or not np.isfinite(values.to_numpy(dtype=float)).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise RegulationModelInputError(f"{column} must contain integers")
    integer = values.astype(int)
    if minimum is not None and integer.lt(minimum).any():
        raise RegulationModelInputError(f"{column} is below {minimum}")
    if maximum is not None and integer.gt(maximum).any():
        raise RegulationModelInputError(f"{column} is above {maximum}")
    return integer


def prepare_regulation_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate PanelV4 and return only uncensored development risk rows."""

    if not isinstance(panel, pd.DataFrame):
        raise RegulationModelInputError("panel must be a DataFrame")
    missing = set(REQUIRED_PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise RegulationModelInputError(
            f"{SCHEMA_VERSION} missing columns: {sorted(missing)}"
        )
    forbidden = (
        _FORBIDDEN_LEGACY_COLUMNS
        | _FORBIDDEN_PRIMARY_FEATURE_NAMES
        | {
            column
            for column in panel.columns
            if str(column).startswith(("final_", "finalized_"))
        }
    ) & set(panel.columns)
    if forbidden:
        raise RegulationModelInputError(
            "forbidden legacy/finalized primary fields: "
            f"{sorted(forbidden)}"
        )
    if panel.empty:
        raise RegulationModelInputError("panel must be nonempty")
    frame = panel.loc[:, list(REQUIRED_PANEL_COLUMNS)].copy(deep=True)

    if not frame["schema_version"].eq(SCHEMA_VERSION).all():
        raise RegulationModelInputError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if not frame["dataset_split"].eq(DEVELOPMENT_SPLIT).all():
        raise RegulationModelInputError(
            "corrected model runner accepts DEVELOPMENT rows only"
        )
    holdout = _strict_bool(frame, "holdout_reaction_accessed")
    if holdout.any():
        raise RegulationModelInputError(
            "holdout reaction data must remain sealed"
        )
    for column in (
        "source_row_id",
        "game_id",
        "information_anchor_id",
        "episode_id",
    ):
        if frame[column].isna().any() or any(
            not isinstance(value, str) or not value.strip()
            for value in frame[column].drop_duplicates()
        ):
            raise RegulationModelInputError(f"{column} must be nonempty")
    if frame["source_row_id"].duplicated().any():
        raise RegulationModelInputError("source_row_id must be unique")
    if not frame["cohort_authority_sha256"].map(
        lambda value: isinstance(value, str)
        and _SHA256_RE.fullmatch(value) is not None
    ).all():
        raise RegulationModelInputError(
            "cohort_authority_sha256 must be a sha256 digest"
        )
    if frame["cohort_authority_sha256"].nunique() != 1:
        raise RegulationModelInputError(
            "panel must bind one development cohort authority"
        )
    if not frame["anchor_kind"].eq(PRIMARY_ANCHOR_KIND).all():
        raise RegulationModelInputError(
            "formal model input accepts PROVISIONAL_FIRST_SEEN only"
        )
    if not frame["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE).all():
        raise RegulationModelInputError(
            "formal model input accepts PRIMARY_SELECTION only"
        )
    source_interval_start = pd.to_datetime(
        frame["source_interval_start"], utc=True, errors="coerce"
    )
    source_interval_end = pd.to_datetime(
        frame["source_interval_end"], utc=True, errors="coerce"
    )
    information_anchor = pd.to_datetime(
        frame["information_anchor"], utc=True, errors="coerce"
    )
    feature_known_at = pd.to_datetime(
        frame["primary_feature_known_at"], utc=True, errors="coerce"
    )
    prestate_known_at = pd.to_datetime(
        frame["prestate_known_at"], utc=True, errors="coerce"
    )
    if (
        frame["p_before_support_status"].isna().any()
        or not frame["p_before_support_status"].isin(
            ("SUPPORTED", "MISSING", "UNPROVEN")
        ).all()
    ):
        raise RegulationModelInputError(
            "p_before_support_status is invalid"
        )
    supported_prestate = frame["p_before_support_status"].eq("SUPPORTED")
    if (
        source_interval_start.isna().any()
        or source_interval_end.isna().any()
        or information_anchor.isna().any()
        or source_interval_start.gt(source_interval_end).any()
        or not information_anchor.eq(source_interval_end).all()
        or prestate_known_at.loc[supported_prestate].isna().any()
        or prestate_known_at.loc[supported_prestate].gt(
            information_anchor.loc[supported_prestate]
        ).any()
        or prestate_known_at.loc[~supported_prestate].notna().any()
    ):
        raise RegulationModelInputError(
            "source interval must be ordered and information_anchor must "
            "equal source_interval_end; prestate_known_at is required and "
            "bounded only for SUPPORTED prestate, otherwise null"
        )
    supported_features = frame[
        "primary_feature_support_status"
    ].eq("SUPPORTED")
    if (
        frame["primary_feature_support_status"].isna().any()
        or not frame["primary_feature_support_status"].isin(
            ("SUPPORTED", "UNPROVEN")
        ).all()
        or frame["primary_feature_support_reason"].isna().any()
        or any(
            not isinstance(value, str) or not value.strip()
            for value in frame[
                "primary_feature_support_reason"
            ].drop_duplicates()
        )
    ):
        raise RegulationModelInputError(
            "primary feature support status/reason is invalid"
        )
    if (
        feature_known_at.loc[supported_features].isna().any()
        or feature_known_at.loc[supported_features].gt(
            information_anchor.loc[supported_features]
        ).any()
        or feature_known_at.loc[~supported_features].notna().any()
    ):
        raise RegulationModelInputError(
            "primary features must be known by provisional anchor; "
            "UNPROVEN rows must not invent known_at"
        )
    if not frame["primary_feature_snapshot_sha256"].map(
        lambda value: isinstance(value, str)
        and _SHA256_RE.fullmatch(value) is not None
    ).all():
        raise RegulationModelInputError(
            "primary_feature_snapshot_sha256 must be a SHA-256 digest"
        )
    if not frame["prestate_context_sha256"].map(
        lambda value: isinstance(value, str)
        and _SHA256_RE.fullmatch(value) is not None
    ).all():
        raise RegulationModelInputError(
            "prestate_context_sha256 must be a SHA-256 digest"
        )
    frame["source_interval_start"] = source_interval_start
    frame["source_interval_end"] = source_interval_end
    frame["information_anchor"] = information_anchor
    frame["primary_feature_known_at"] = feature_known_at
    frame["prestate_known_at"] = prestate_known_at
    venue = frame["venue"].astype(str).str.upper()
    if not venue.isin((POLYMARKET, KALSHI)).all():
        raise RegulationModelInputError(
            "venue must be POLYMARKET or KALSHI"
        )
    frame["venue"] = venue
    frame["nfl_week"] = _validate_integer(
        frame, "nfl_week", minimum=1, maximum=12
    )
    frame["landmark_seconds"] = _validate_integer(
        frame, "landmark_seconds", minimum=1
    )
    frame["endpoint_seconds"] = _validate_integer(
        frame, "endpoint_seconds", minimum=1
    )
    if (
        not frame["landmark_seconds"].isin(LANDMARK_SECONDS).all()
        or not frame["endpoint_seconds"].isin(ENDPOINT_SECONDS).all()
        or not frame["endpoint_seconds"].gt(
            frame["landmark_seconds"]
        ).all()
    ):
        raise RegulationModelInputError(
            "L/H must use the frozen grid and satisfy H > L"
        )
    frame["is_overtime"] = _strict_bool(frame, "is_overtime")
    quarter = _validate_integer(frame, "quarter", minimum=1)
    if frame["is_overtime"].any() or quarter.gt(4).any():
        raise RegulationModelInputError(
            "RegulationDecisionPanelV4 is regulation-only"
        )
    frame["quarter"] = quarter

    frame["primary_selection_eligible"] = _strict_bool(
        frame, "primary_selection_eligible"
    )
    frame["landmark_eligible"] = _strict_bool(
        frame, "landmark_eligible"
    )
    frame["continuity_valid"] = _strict_bool(
        frame, "continuity_valid"
    )
    frame["order_ambiguous"] = _strict_bool(
        frame, "order_ambiguous"
    )
    frame["censor_boundary_eligible"] = _strict_bool(
        frame, "censor_boundary_eligible"
    )
    frame["censored"] = _strict_bool(frame, "censored")
    frame["loss_eligible"] = _strict_bool(frame, "loss_eligible")
    frame["direction_eligible"] = _strict_bool(
        frame, "direction_eligible"
    )
    if frame["censor_reason"].isna().any() or any(
        not isinstance(value, str) or not value.strip()
        for value in frame["censor_reason"].drop_duplicates()
    ):
        raise RegulationModelInputError("censor_reason must be nonempty")
    if (
        frame.loc[~supported_features, "primary_selection_eligible"].any()
        or not frame.loc[~supported_features, "censored"].all()
    ):
        raise RegulationModelInputError(
            "UNPROVEN primary snapshots must be ineligible and retained "
            "as censored rows"
        )
    risk = (
        frame["primary_selection_eligible"]
        & frame["landmark_eligible"]
        & ~frame["censored"]
        & frame["continuity_valid"]
        & ~frame["order_ambiguous"]
        & supported_features
    )
    if not frame["loss_eligible"].eq(risk).all():
        raise RegulationModelInputError(
            "loss_eligible must equal primary_selection_eligible AND "
            "landmark_eligible AND NOT censored AND continuity_valid AND "
            "NOT order_ambiguous AND "
            "primary_feature_support_status=SUPPORTED"
        )
    availability = frame["availability_h"]
    if availability.loc[risk].isna().any():
        raise RegulationModelInputError(
            "availability_h must be observed on every uncensored risk row"
        )
    observed_values = pd.to_numeric(
        availability.loc[risk], errors="coerce"
    )
    if observed_values.isna().any() or not observed_values.isin((0, 1)).all():
        raise RegulationModelInputError(
            "availability_h must contain 0/1 on risk rows"
        )
    if availability.loc[~risk].notna().any():
        raise RegulationModelInputError(
            "censored/ineligible rows must not carry availability_h"
        )
    frame["availability_h"] = pd.to_numeric(
        availability, errors="coerce"
    ).astype("Int8")
    available = risk & frame["availability_h"].fillna(0).eq(1)
    direction = frame["direction_h"]
    if (
        direction.loc[available].isna().any()
        or not direction.loc[available].isin(DIRECTION_CLASSES).all()
    ):
        raise RegulationModelInputError(
            "direction_h must be DOWN/NO_MOVE/UP when availability_h is true"
        )
    if direction.loc[~available].notna().any():
        raise RegulationModelInputError(
            "direction_h must be null when availability_h is not true"
        )
    if not frame["direction_eligible"].eq(available).all():
        raise RegulationModelInputError(
            "direction_eligible must equal availability_h on loss rows"
        )
    direction_threshold = pd.to_numeric(
        frame["direction_threshold_probability"], errors="coerce"
    )
    if (
        direction_threshold.isna().any()
        or not np.isclose(
            direction_threshold.to_numpy(dtype=float),
            0.01,
            rtol=0,
            atol=0,
        ).all()
    ):
        raise RegulationModelInputError(
            "direction_threshold_probability must equal 0.01"
        )

    frame["p_before_home_missing"] = _strict_bool(
        frame, "p_before_home_missing"
    )
    p_before = pd.to_numeric(frame["p_before_home"], errors="coerce")
    missing_reference = frame["p_before_home_missing"]
    expected_reference_missing = ~frame[
        "p_before_support_status"
    ].eq("SUPPORTED")
    if not missing_reference.eq(expected_reference_missing).all():
        raise RegulationModelInputError(
            "p_before missing indicator disagrees with support status"
        )
    if (
        p_before.loc[~missing_reference].isna().any()
        or not p_before.loc[~missing_reference].between(0, 1).all()
        or p_before.loc[missing_reference].notna().any()
    ):
        raise RegulationModelInputError(
            "p_before_home and its missing indicator disagree"
        )
    frame["p_before_home"] = p_before
    for value_column, missing_column in (
        ("possession_is_home", "possession_is_home_missing"),
        ("down", "down_missing"),
        ("distance", "distance_missing"),
        ("yardline_100", "yardline_100_missing"),
    ):
        missing = _strict_bool(frame, missing_column)
        if not missing.eq(frame[value_column].isna()).all():
            raise RegulationModelInputError(
                f"{missing_column} disagrees with {value_column}"
            )
        frame[missing_column] = missing
    for column in _BOOLEAN_FEATURES - {
        "p_before_home_missing",
        "is_overtime",
        "possession_is_home_missing",
        "down_missing",
        "distance_missing",
        "yardline_100_missing",
    }:
        frame[column] = _nullable_bool_numeric(frame, column)

    prepared = frame.loc[risk].copy()
    if prepared.empty:
        raise RegulationModelInputError(
            "panel has no uncensored decision rows"
        )
    prepared["availability_h"] = prepared["availability_h"].astype(bool)
    prepared = prepared.sort_values(
        [
            "nfl_week",
            "game_id",
            "information_anchor",
            "information_anchor_id",
            "episode_id",
            "anchor_kind",
            "venue",
            "actual_home_contract_id",
            "landmark_seconds",
            "endpoint_seconds",
            "source_row_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    build_x15_week_folds(prepared)
    return prepared


def _weighted_binary_mean(
    truth: np.ndarray, weights: np.ndarray
) -> float:
    if len(truth) == 0:
        return 0.5
    probability = float(np.average(truth.astype(float), weights=weights))
    return float(
        np.clip(probability, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
    )


def _weighted_direction_mean(
    truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if len(truth) == 0:
        return np.full(3, 1 / 3, dtype=float)
    probability = np.asarray(
        [
            np.average(truth == label, weights=weights)
            for label in DIRECTION_CLASSES
        ],
        dtype=float,
    )
    probability = np.clip(probability, _PROBABILITY_EPSILON, 1)
    return probability / probability.sum()


def _equal_game_episode_weights(
    frame: pd.DataFrame, eligibility: np.ndarray
) -> np.ndarray:
    """Use the existing row-to-episode-to-game hierarchy utility."""

    identity = frame.loc[:, ["game_id", "information_anchor_id"]].rename(
        columns={
            "information_anchor_id": "atomic_information_episode_id"
        }
    )
    return hierarchical_sample_weights(
        identity, eligibility
    ).to_numpy(dtype=float)


def _weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return 0.0
    observed = values[finite]
    observed_weights = weights[finite]
    order = np.argsort(observed, kind="mergesort")
    observed = observed[order]
    observed_weights = observed_weights[order]
    cutoff = 0.5 * float(observed_weights.sum())
    position = int(
        np.searchsorted(
            np.cumsum(observed_weights), cutoff, side="left"
        )
    )
    return float(observed[min(position, len(observed) - 1)])


def _fit_weighted_feature_transformer(
    frame: pd.DataFrame,
    columns: Sequence[str],
    weights: np.ndarray,
) -> tuple[_WeightedFeatureTransformer, np.ndarray]:
    if len(frame) != len(weights):
        raise RegulationModelInputError(
            "preprocessor rows and hierarchical weights are unaligned"
        )
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise RegulationModelInputError(
            "preprocessor weights must be finite and positive"
        )
    normalized_weights = weights / float(weights.sum())
    categorical = tuple(
        column
        for column in columns
        if column in _CATEGORICAL_FEATURES
    )
    numeric = tuple(
        column for column in columns if column not in categorical
    )
    coerced = _coerce_feature_frame(frame, columns)
    impute: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in numeric:
        values = pd.to_numeric(
            coerced[column], errors="coerce"
        ).to_numpy(dtype=float)
        impute[column] = _weighted_median(values, normalized_weights)
        filled = np.where(np.isfinite(values), values, impute[column])
        means[column] = float(
            np.average(filled, weights=normalized_weights)
        )
        variance = float(
            np.average(
                (filled - means[column]) ** 2,
                weights=normalized_weights,
            )
        )
        scales[column] = math.sqrt(variance) if variance > 0 else 1.0
    levels = {
        column: tuple(
            sorted(
                set(
                    coerced[column]
                    .astype("object")
                    .where(coerced[column].notna(), "MISSING")
                    .astype(str)
                )
            )
        )
        for column in categorical
    }
    transformer = _WeightedFeatureTransformer(
        columns=tuple(columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        numeric_impute=MappingProxyType(impute),
        numeric_mean=MappingProxyType(means),
        numeric_scale=MappingProxyType(scales),
        categorical_levels=MappingProxyType(levels),
    )
    return transformer, transformer.transform(frame)


def _coerce_feature_frame(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in _CATEGORICAL_FEATURES:
            values = frame[column].astype("object")
            result[column] = values.where(pd.notna(values), np.nan)
        else:
            result[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype(float)
    return result


def _feature_training_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
    weights: np.ndarray,
    *,
    preprocessor_spec_sha256: str,
    head: str,
) -> str:
    if len(frame) != len(weights):
        raise RegulationModelInputError(
            "preprocessor training hash rows are unaligned"
        )
    ordered = frame.assign(_weight=weights).sort_values(
        "source_row_id", kind="mergesort"
    )
    records = (
        {
            "source_row_id": str(values[0]),
            "hierarchical_weight": float(values[-1]),
            **{
                column: values[position + 1]
                for position, column in enumerate(columns)
            },
        }
        for values in ordered.loc[
            :, ["source_row_id", *columns, "_weight"]
        ].itertuples(index=False, name=None)
    )
    return _sha256_row_stream(
        records,
        envelope={
            "head": head,
            "risk_set": (
                "ALL_UNCENSORED_PRIMARY_ROWS"
                if head == "availability"
                else "AVAILABILITY_H_EQUALS_ONE_ROWS_ONLY"
            ),
            "preprocessor_spec_sha256": preprocessor_spec_sha256,
            "feature_columns": tuple(columns),
            "purpose": "PREPROCESSOR_FIT",
            "hierarchical_weighting": (
                "ROW_TO_INFORMATION_ANCHOR_TO_GAME"
            ),
            "numeric_imputation": "WEIGHTED_MEDIAN",
            "numeric_scaling": "WEIGHTED_MEAN_AND_STANDARD_DEVIATION",
        },
    )


def _head_training_sha256(
    frame: pd.DataFrame,
    *,
    truth_column: str,
    weights: np.ndarray,
    preprocessor_training_sha256: str,
    model_spec_sha256: str,
) -> str:
    if len(frame) != len(weights):
        raise RegulationModelInputError("training hash rows are unaligned")
    ordered = frame.assign(_weight=weights).sort_values(
        "source_row_id", kind="mergesort"
    )
    records = (
        {
            "source_row_id": str(source_row_id),
            "truth": truth,
            "weight": float(weight),
        }
        for source_row_id, truth, weight in ordered[
            ["source_row_id", truth_column, "_weight"]
        ].itertuples(index=False, name=None)
    )
    return _sha256_row_stream(
        records,
        envelope={
            "model_spec_sha256": model_spec_sha256,
            "preprocessor_training_sha256": (
                preprocessor_training_sha256
            ),
            "label_column": truth_column,
        },
    )


def _primary_snapshot_set_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("source_row_id", kind="mergesort")
    return _sha256_row_stream(
        (
            {
                "source_row_id": str(source_row_id),
                "primary_feature_known_at": pd.Timestamp(known_at).isoformat(),
                "primary_feature_snapshot_sha256": str(snapshot_sha256),
                "prestate_known_at": (
                    None
                    if pd.isna(prestate_known_at)
                    else pd.Timestamp(prestate_known_at).isoformat()
                ),
                "prestate_context_sha256": str(prestate_sha256),
                "p_before_support_status": str(p_before_status),
            }
            for (
                source_row_id,
                known_at,
                snapshot_sha256,
                prestate_known_at,
                prestate_sha256,
                p_before_status,
            ) in ordered[
                [
                    "source_row_id",
                    "primary_feature_known_at",
                    "primary_feature_snapshot_sha256",
                    "prestate_known_at",
                    "prestate_context_sha256",
                    "p_before_support_status",
                ]
            ].itertuples(index=False, name=None)
        ),
        envelope={
            "purpose": "PRIMARY_PROVISIONAL_FEATURE_SUPPORT",
            "anchor_kind": PRIMARY_ANCHOR_KIND,
            "analysis_role": PRIMARY_ANALYSIS_ROLE,
        },
    )


def _fit_estimator(
    *,
    model_id: str,
    matrix: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    direction: bool,
    random_state: int,
) -> object:
    if direction:
        unique = tuple(sorted(set(str(value) for value in truth)))
        if len(unique) < 2:
            return _ConstantDirection(
                _weighted_direction_mean(truth, weights)
            )
    elif len(set(int(value) for value in truth)) < 2:
        return _ConstantBinary(_weighted_binary_mean(truth, weights))

    if model_id == "regularized_logistic_v1":
        estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1_000,
            random_state=random_state,
        )
    elif model_id == "shallow_xgboost_v1":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=16,
            max_depth=2,
            learning_rate=0.08,
            min_child_weight=3,
            subsample=1.0,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
            eval_metric="mlogloss" if direction else "logloss",
        )
    else:
        raise RegulationModelInputError(f"unsupported model_id: {model_id}")
    if model_id == "shallow_xgboost_v1" and direction:
        present_classes = tuple(
            label
            for label in DIRECTION_CLASSES
            if label in set(str(value) for value in truth)
        )
        class_index = {
            label: index
            for index, label in enumerate(present_classes)
        }
        encoded = np.asarray(
            [class_index[str(value)] for value in truth],
            dtype=int,
        )
        estimator.fit(matrix, encoded, sample_weight=weights)
        return _EncodedDirectionEstimator(
            estimator=estimator,
            classes_=np.asarray(present_classes, dtype=object),
        )
    estimator.fit(matrix, truth, sample_weight=weights)
    return estimator


def _collapse_duplicate_training_rows(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    feature_columns: Sequence[str],
    truth_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse exact semantic duplicates before numerical optimization."""

    if len(frame) != len(weights):
        raise RegulationModelInputError(
            "duplicate-collapse rows and weights are unaligned"
        )
    ordered = frame.assign(_weight=weights).sort_values(
        "source_row_id", kind="mergesort"
    )
    keys: dict[str, tuple[int, float]] = {}
    columns = (
        "game_id",
        "information_anchor_id",
        *feature_columns,
        truth_column,
    )
    for position, values in enumerate(
        ordered.loc[:, [*columns, "_weight"]].itertuples(
            index=False, name=None
        )
    ):
        key = json.dumps(
            _canonical(values[:-1]),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if key in keys:
            first, total = keys[key]
            keys[key] = (first, total + float(values[-1]))
        else:
            keys[key] = (position, float(values[-1]))
    selected = sorted(keys.values(), key=lambda value: value[0])
    collapsed = ordered.iloc[
        [position for position, _ in selected]
    ].drop(columns="_weight").reset_index(drop=True)
    collapsed_weights = np.asarray(
        [weight for _, weight in selected], dtype=float
    )
    return collapsed, collapsed_weights


def _fit_two_heads(
    train: pd.DataFrame,
    *,
    model_id: str,
    feature_block_id: str,
    random_state: int,
) -> _TwoHeadPredictor:
    columns = FEATURE_BLOCKS[feature_block_id]
    availability_weights = _equal_game_episode_weights(
        train, np.ones(len(train), dtype=bool)
    )
    direction_mask = train["availability_h"].to_numpy(dtype=bool)
    direction_weights = _equal_game_episode_weights(
        train, direction_mask
    )[direction_mask]
    availability_training_frame, availability_training_weights = (
        _collapse_duplicate_training_rows(
            train,
            availability_weights,
            feature_columns=columns,
            truth_column="availability_h",
        )
    )
    direction_training_frame = train.loc[
        direction_mask
    ].reset_index(drop=True)
    direction_training_frame, direction_weights = (
        _collapse_duplicate_training_rows(
            direction_training_frame,
            direction_weights,
            feature_columns=columns,
            truth_column="direction_h",
        )
    )
    availability_preprocessor_spec_sha256 = (
        _preprocessor_spec_sha256(columns, head="availability")
    )
    direction_preprocessor_spec_sha256 = (
        _preprocessor_spec_sha256(columns, head="direction")
    )
    availability_model_spec_sha256 = _model_spec_sha256(
        model_id=model_id,
        direction=False,
        random_state=random_state,
    )
    direction_model_spec_sha256 = _model_spec_sha256(
        model_id=model_id,
        direction=True,
        random_state=random_state,
    )
    availability_preprocessor_training_sha256 = (
        _feature_training_sha256(
            availability_training_frame,
            columns,
            availability_training_weights,
            preprocessor_spec_sha256=(
                availability_preprocessor_spec_sha256
            ),
            head="availability",
        )
    )
    direction_preprocessor_training_sha256 = (
        _feature_training_sha256(
            direction_training_frame,
            columns,
            direction_weights,
            preprocessor_spec_sha256=(
                direction_preprocessor_spec_sha256
            ),
            head="direction",
        )
    )
    availability_training_sha256 = _head_training_sha256(
        availability_training_frame,
        truth_column="availability_h",
        weights=availability_training_weights,
        preprocessor_training_sha256=(
            availability_preprocessor_training_sha256
        ),
        model_spec_sha256=availability_model_spec_sha256,
    )
    direction_training_sha256 = _head_training_sha256(
        direction_training_frame,
        truth_column="direction_h",
        weights=direction_weights,
        preprocessor_training_sha256=(
            direction_preprocessor_training_sha256
        ),
        model_spec_sha256=direction_model_spec_sha256,
    )

    if model_id == "b0_empirical_v1":
        availability_model = _EmpiricalBinary().fit(
            availability_training_frame,
            availability_training_frame["availability_h"].to_numpy(
                dtype=int
            ),
            availability_training_weights,
        )
        direction_model = _EmpiricalDirection().fit(
            direction_training_frame,
            direction_training_frame["direction_h"].to_numpy(dtype=object),
            direction_weights,
        )
        return _TwoHeadPredictor(
            availability_transformer=None,
            direction_transformer=None,
            availability_model=availability_model,
            direction_model=direction_model,
            availability_preprocessor_spec_sha256=(
                availability_preprocessor_spec_sha256
            ),
            availability_preprocessor_training_sha256=(
                availability_preprocessor_training_sha256
            ),
            availability_preprocessor_fitted_sha256=(
                _preprocessor_fitted_sha256(
                    None,
                    model_id=model_id,
                    head="availability",
                )
            ),
            direction_preprocessor_spec_sha256=(
                direction_preprocessor_spec_sha256
            ),
            direction_preprocessor_training_sha256=(
                direction_preprocessor_training_sha256
            ),
            direction_preprocessor_fitted_sha256=(
                _preprocessor_fitted_sha256(
                    None,
                    model_id=model_id,
                    head="direction",
                )
            ),
            availability_model_spec_sha256=(
                availability_model_spec_sha256
            ),
            availability_model_training_sha256=(
                availability_training_sha256
            ),
            availability_model_fitted_sha256=(
                _fitted_estimator_sha256(
                    availability_model,
                    model_id=model_id,
                    head="availability",
                )
            ),
            direction_model_spec_sha256=direction_model_spec_sha256,
            direction_model_training_sha256=direction_training_sha256,
            direction_model_fitted_sha256=(
                _fitted_estimator_sha256(
                    direction_model,
                    model_id=model_id,
                    head="direction",
                )
            ),
        )

    (
        availability_transformer,
        availability_matrix,
    ) = _fit_weighted_feature_transformer(
        availability_training_frame, columns, availability_training_weights
    )
    (
        direction_transformer,
        direction_matrix,
    ) = _fit_weighted_feature_transformer(
        direction_training_frame, columns, direction_weights
    )
    availability_model = _fit_estimator(
        model_id=model_id,
        matrix=availability_matrix,
        truth=availability_training_frame[
            "availability_h"
        ].to_numpy(dtype=int),
        weights=availability_training_weights,
        direction=False,
        random_state=random_state,
    )
    direction_model = _fit_estimator(
        model_id=model_id,
        matrix=direction_matrix,
        truth=direction_training_frame["direction_h"].to_numpy(
            dtype=object
        ),
        weights=direction_weights,
        direction=True,
        random_state=random_state,
    )
    return _TwoHeadPredictor(
        availability_transformer=availability_transformer,
        direction_transformer=direction_transformer,
        availability_model=availability_model,
        direction_model=direction_model,
        availability_preprocessor_spec_sha256=(
            availability_preprocessor_spec_sha256
        ),
        availability_preprocessor_training_sha256=(
            availability_preprocessor_training_sha256
        ),
        availability_preprocessor_fitted_sha256=(
            _preprocessor_fitted_sha256(
                availability_transformer,
                model_id=model_id,
                head="availability",
            )
        ),
        direction_preprocessor_spec_sha256=(
            direction_preprocessor_spec_sha256
        ),
        direction_preprocessor_training_sha256=(
            direction_preprocessor_training_sha256
        ),
        direction_preprocessor_fitted_sha256=_preprocessor_fitted_sha256(
            direction_transformer,
            model_id=model_id,
            head="direction",
        ),
        availability_model_spec_sha256=availability_model_spec_sha256,
        availability_model_training_sha256=(
            availability_training_sha256
        ),
        availability_model_fitted_sha256=_fitted_estimator_sha256(
            availability_model,
            model_id=model_id,
            head="availability",
        ),
        direction_model_spec_sha256=direction_model_spec_sha256,
        direction_model_training_sha256=direction_training_sha256,
        direction_model_fitted_sha256=_fitted_estimator_sha256(
            direction_model,
            model_id=model_id,
            head="direction",
        ),
    )


def _aligned_direction_probability(
    estimator: object, matrix: np.ndarray
) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(matrix), dtype=float)
    classes = tuple(str(value) for value in estimator.classes_)
    result = np.full(
        (len(matrix), len(DIRECTION_CLASSES)),
        _PROBABILITY_EPSILON,
        dtype=float,
    )
    for source_index, label in enumerate(classes):
        if label not in DIRECTION_CLASSES:
            raise RegulationModelInputError(
                f"unexpected direction class: {label}"
            )
        result[:, DIRECTION_CLASSES.index(label)] = raw[:, source_index]
    return result / result.sum(axis=1, keepdims=True)


def _predict_two_heads(
    fitted: _TwoHeadPredictor,
    frame: pd.DataFrame,
    *,
    model_id: str,
    feature_block_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    if model_id == "b0_empirical_v1":
        availability = fitted.availability_model.predict_probability(frame)
        direction = fitted.direction_model.predict_probability(frame)
    else:
        if (
            fitted.availability_transformer is None
            or fitted.direction_transformer is None
        ):
            raise AssertionError("fitted head transformer is missing")
        availability_matrix = np.asarray(
            fitted.availability_transformer.transform(frame),
            dtype=float,
        )
        direction_matrix = np.asarray(
            fitted.direction_transformer.transform(frame),
            dtype=float,
        )
        availability = np.asarray(
            fitted.availability_model.predict_proba(
                availability_matrix
            ),
            dtype=float,
        )[:, 1]
        direction = _aligned_direction_probability(
            fitted.direction_model, direction_matrix
        )
    return (
        np.clip(
            availability,
            _PROBABILITY_EPSILON,
            1 - _PROBABILITY_EPSILON,
        ),
        direction,
    )


def _raw_calibrators(
    *,
    outer_train: pd.DataFrame,
    model_id: str,
    feature_block_id: str,
    random_state: int,
) -> tuple[
    _TwoHeadPredictor,
    BinaryPlattCalibrator,
    DirectionTemperatureCalibrator,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit one base producer, then calibrate it on a strict-later week."""

    final_train_week = int(outer_train["nfl_week"].max())
    inner_train = outer_train.loc[
        outer_train["nfl_week"].lt(final_train_week)
    ].reset_index(drop=True)
    calibration = outer_train.loc[
        outer_train["nfl_week"].eq(final_train_week)
    ].reset_index(drop=True)
    if inner_train.empty or calibration.empty:
        raise RegulationModelInputError(
            "outer fold cannot produce an internal temporal calibration split"
        )
    if (
        set(inner_train["game_id"]) & set(calibration["game_id"])
        or int(inner_train["nfl_week"].max())
        >= int(calibration["nfl_week"].min())
    ):
        raise RegulationModelInputError(
            "base-producer and calibration games/weeks must be disjoint "
            "and chronological"
        )
    fitted = _fit_two_heads(
        inner_train,
        model_id=model_id,
        feature_block_id=feature_block_id,
        random_state=random_state,
    )
    availability_raw, direction_raw = _predict_two_heads(
        fitted,
        calibration,
        model_id=model_id,
        feature_block_id=feature_block_id,
    )
    availability_weights = _equal_game_episode_weights(
        calibration, np.ones(len(calibration), dtype=bool)
    )
    availability_calibrator = fit_binary_platt(
        availability_raw,
        calibration["availability_h"].to_numpy(dtype=int),
        calibration["game_id"].to_numpy(dtype=object),
        sample_weight=availability_weights,
        row_ids=calibration["source_row_id"].to_numpy(dtype=object),
    )
    direction_mask = calibration["availability_h"].to_numpy(dtype=bool)
    direction_weights = _equal_game_episode_weights(
        calibration, direction_mask
    )[direction_mask]
    direction_calibrator = fit_direction_temperature(
        direction_raw[direction_mask],
        calibration.loc[direction_mask, "direction_h"].to_numpy(
            dtype=object
        ),
        calibration.loc[direction_mask, "game_id"].to_numpy(dtype=object),
        sample_weight=direction_weights,
        row_ids=calibration.loc[
            direction_mask, "source_row_id"
        ].to_numpy(dtype=object),
    )
    return (
        fitted,
        availability_calibrator,
        direction_calibrator,
        inner_train,
        calibration,
    )


def regulation_row_losses(
    *,
    availability_truth: object,
    availability_probability: object,
    direction_truth: object,
    direction_probability: object,
) -> pd.DataFrame:
    """Return the frozen two-head row loss; no censoring head exists."""

    availability_y = np.asarray(availability_truth)
    availability_p = np.asarray(availability_probability, dtype=float)
    direction_y = np.asarray(direction_truth, dtype=object)
    direction_p = np.asarray(direction_probability, dtype=float)
    rows = len(availability_y)
    if (
        availability_y.ndim != 1
        or availability_p.ndim != 1
        or direction_y.ndim != 1
        or direction_p.shape != (rows, len(DIRECTION_CLASSES))
        or len(availability_p) != rows
        or len(direction_y) != rows
    ):
        raise RegulationModelInputError("loss inputs must be aligned")
    if not np.isin(availability_y, [0, 1, False, True]).all():
        raise RegulationModelInputError(
            "availability truth must be binary"
        )
    if (
        not np.isfinite(availability_p).all()
        or np.any((availability_p < 0) | (availability_p > 1))
        or not np.isfinite(direction_p).all()
        or np.any(direction_p < 0)
        or np.any(direction_p.sum(axis=1) <= 0)
    ):
        raise RegulationModelInputError("probabilities are invalid")
    availability_y = availability_y.astype(int)
    availability_p = np.clip(
        availability_p,
        _PROBABILITY_EPSILON,
        1 - _PROBABILITY_EPSILON,
    )
    direction_p = np.clip(direction_p, _PROBABILITY_EPSILON, 1)
    direction_p /= direction_p.sum(axis=1, keepdims=True)
    observed_direction = availability_y == 1
    if any(
        direction_y[index] not in DIRECTION_CLASSES
        for index in np.flatnonzero(observed_direction)
    ):
        raise RegulationModelInputError(
            "observed direction truth must use the frozen classes"
        )
    availability_log_loss = -(
        availability_y * np.log(availability_p)
        + (1 - availability_y) * np.log(1 - availability_p)
    )
    availability_brier = (
        availability_p - availability_y.astype(float)
    ) ** 2
    direction_log_loss = np.full(rows, math.nan, dtype=float)
    direction_brier = np.full(rows, math.nan, dtype=float)
    for index in np.flatnonzero(observed_direction):
        truth_index = DIRECTION_CLASSES.index(str(direction_y[index]))
        one_hot = np.eye(len(DIRECTION_CLASSES))[truth_index]
        direction_log_loss[index] = -math.log(
            direction_p[index, truth_index]
        )
        direction_brier[index] = float(
            np.sum((direction_p[index] - one_hot) ** 2)
        )
    return pd.DataFrame(
        {
            "availability_log_loss": availability_log_loss,
            "availability_brier": availability_brier,
            "direction_log_loss": direction_log_loss,
            "direction_multiclass_brier": direction_brier,
            "joint_log_loss": (
                availability_log_loss
                + np.nan_to_num(direction_log_loss, nan=0.0)
            ),
        }
    )


def compute_equal_game_metrics(
    row_losses: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate rows→information unit→game, then weight games equally."""

    metric_columns = (
        "availability_log_loss",
        "availability_brier",
        "direction_log_loss",
        "direction_multiclass_brier",
        "joint_log_loss",
    )
    missing = {"game_id", "information_anchor_id", *metric_columns} - set(
        row_losses.columns
    )
    if missing:
        raise RegulationModelInputError(
            f"row losses missing columns: {sorted(missing)}"
        )
    by_anchor = row_losses.groupby(
        ["game_id", "information_anchor_id"], sort=True
    )[
        list(metric_columns)
    ].mean()
    by_game = by_anchor.groupby(level="game_id", sort=True).mean()
    return {
        column: (
            float(by_game[column].dropna().mean())
            if by_game[column].notna().any()
            else math.nan
        )
        for column in metric_columns
    }


def _validate_cells(
    cells: Sequence[tuple[str, str, str]],
) -> tuple[tuple[str, str, str], ...]:
    requested = tuple(cells)
    if not requested:
        raise RegulationModelInputError("at least one cell is required")
    if len(set(requested)) != len(requested):
        raise RegulationModelInputError("execution cells must be unique")
    unsupported = set(requested) - set(CORRECTED_MODEL_CELLS)
    if unsupported:
        raise RegulationModelInputError(
            f"unsupported corrected cells: {sorted(unsupported)}"
        )
    return requested


def regulation_oof_batch_status(
    completed_cells: Sequence[tuple[str, str, str]],
) -> dict[str, object]:
    """Return the fail-closed partial/final status of the corrected matrix."""

    completed = tuple(completed_cells)
    if len(completed) != len(set(completed)):
        raise RegulationModelInputError(
            "completed OOF cells must be unique"
        )
    unsupported = set(completed) - set(CORRECTED_MODEL_CELLS)
    if unsupported:
        raise RegulationModelInputError(
            f"completed OOF cells are unsupported: {sorted(unsupported)}"
        )
    completed_set = set(completed)
    missing = tuple(
        cell
        for cell in CORRECTED_MODEL_CELLS
        if cell not in completed_set
    )
    complete = not missing
    return {
        "run_status": "COMPLETE" if complete else "PARTIAL",
        "publication_gate": "PASS" if complete else "BLOCKED_INCOMPLETE",
        "completed_cell_count": len(completed),
        "expected_cell_count": len(CORRECTED_MODEL_CELLS),
        "completed_cells": tuple(
            cell for cell in CORRECTED_MODEL_CELLS if cell in completed_set
        ),
        "missing_cells": missing,
        "holdout_reaction_accessed": False,
    }


def _game_level_improvement(
    paired: pd.DataFrame,
    *,
    metric: str,
) -> pd.Series:
    improvement = (
        pd.to_numeric(
            paired[f"{metric}_baseline"], errors="coerce"
        )
        - pd.to_numeric(
            paired[f"{metric}_candidate"], errors="coerce"
        )
    )
    work = paired.loc[
        :, ["game_id", "information_anchor_id"]
    ].assign(improvement=improvement)
    by_anchor = work.groupby(
        ["game_id", "information_anchor_id"], sort=True
    )["improvement"].mean()
    return by_anchor.groupby(level="game_id", sort=True).mean().dropna()


def _game_bootstrap_summary(
    values: pd.Series,
    *,
    seed: int,
) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise RegulationModelInputError(
            "paired selection metric needs at least two finite games"
        )
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    rng = np.random.default_rng(seed)
    selections = rng.integers(
        0,
        len(array),
        size=(SELECTION_BOOTSTRAP_SAMPLES, len(array)),
    )
    bootstrapped = array[selections].mean(axis=1)
    tail = (1 - SELECTION_CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(bootstrapped, [tail, 1 - tail])
    return (
        float(array.mean()),
        standard_error,
        float(low),
        float(high),
    )


def _paired_candidate_rows(
    *,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    pair_columns = (
        "source_row_id",
        "game_id",
        "nfl_week",
        "information_anchor_id",
        "episode_id",
        "anchor_kind",
        "analysis_role",
        "venue",
        "actual_home_contract_id",
        "landmark_seconds",
        "endpoint_seconds",
        "availability_truth",
        "direction_truth",
        "cohort_authority_sha256",
        "panel_batch_sha256",
        "fold_id",
    )
    loss_columns = (
        "availability_log_loss",
        "availability_brier",
        "direction_log_loss",
        "direction_multiclass_brier",
        "joint_log_loss",
    )
    missing = (
        set((*pair_columns, *loss_columns))
        - set(baseline.columns)
    ) | (
        set((*pair_columns, *loss_columns))
        - set(candidate.columns)
    )
    if missing:
        raise RegulationModelInputError(
            f"paired selection inputs miss columns: {sorted(missing)}"
        )
    if (
        baseline["source_row_id"].duplicated().any()
        or candidate["source_row_id"].duplicated().any()
    ):
        raise RegulationModelInputError(
            "paired selection source_row_id must be unique per model"
        )
    paired = baseline.loc[:, [*pair_columns, *loss_columns]].merge(
        candidate.loc[:, [*pair_columns, *loss_columns]],
        on=list(pair_columns),
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if (
        len(paired) != len(baseline)
        or len(paired) != len(candidate)
        or set(paired["source_row_id"].astype(str))
        != set(baseline["source_row_id"].astype(str))
    ):
        raise RegulationModelInputError(
            "candidate and B0 are not exact source-row pairs"
        )
    return paired


def load_verified_regulation_oof_prediction_projection(
    shard: VerifiedRegulationOOFShard,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Load a bounded selection projection from one verified descriptor."""

    required = (
        "source_row_id",
        "fold_id",
        "model_id",
        "feature_block_id",
        "venue",
        "anchor_kind",
        "analysis_role",
        *TRAINING_PROVENANCE_COLUMNS,
    )
    projected_columns = tuple(dict.fromkeys((*columns, *required)))
    descriptor = shard.object_descriptors.get("oof_predictions")
    if descriptor is None:
        raise RegulationModelInputError(
            "verified OOF descriptor lacks prediction object"
        )
    frame = _load_verified_parquet_projection(
        output_root=shard.output_root,
        descriptor=descriptor,
        logical_name="oof_predictions",
        columns=projected_columns,
    )
    if (
        len(frame) != shard.oof_prediction_row_count
        or frame["source_row_id"].duplicated().any()
        or set(
            zip(
                frame["fold_id"].astype(str),
                frame["model_id"].astype(str),
                frame["feature_block_id"].astype(str),
                strict=True,
            )
        )
        != {shard.cell}
        or not frame["venue"].eq(POLYMARKET).all()
        or not frame["anchor_kind"].eq(PRIMARY_ANCHOR_KIND).all()
        or not frame["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE).all()
        or any(
            not frame[column].astype(str).eq(str(value)).all()
            for column, value in shard.training_hashes.items()
        )
    ):
        raise RegulationModelInputError(
            "bounded OOF projection differs from verified descriptor"
        )
    return frame.loc[:, list(dict.fromkeys(columns))].copy()


def select_regulation_model(
    shards: Sequence[VerifiedRegulationOOFShard],
) -> RegulationSelectionResult:
    """Select once on exact-45 Polymarket OOF using paired game bootstrap."""

    ordered = tuple(
        sorted(
            shards,
            key=lambda shard: CORRECTED_MODEL_CELLS.index(shard.cell),
        )
    )
    status = regulation_oof_batch_status(
        tuple(shard.cell for shard in ordered)
    )
    if (
        status["run_status"] != "COMPLETE"
        or status["publication_gate"] != "PASS"
        or len(ordered) != len(CORRECTED_MODEL_CELLS)
    ):
        raise RegulationModelInputError(
            "selection requires one verified shard for every exact-45 cell"
        )
    by_cell = {shard.cell: shard for shard in ordered}
    audit: list[RegulationSelectionCandidate] = []
    metric_names = (
        "joint_log_loss",
        "availability_log_loss",
        "availability_brier",
        "direction_log_loss",
        "direction_multiclass_brier",
    )
    for candidate_index, (model_id, feature_block_id) in enumerate(
        SELECTABLE_CANDIDATES
    ):
        metric_chunks: dict[str, list[pd.Series]] = {
            metric: [] for metric in metric_names
        }
        anchor_chunks: dict[str, list[pd.Series]] = {
            "joint_log_loss": [],
            "direction_log_loss": [],
            "direction_multiclass_brier": [],
        }
        projection_columns = (
            "source_row_id",
            "game_id",
            "nfl_week",
            "information_anchor_id",
            "episode_id",
            "anchor_kind",
            "analysis_role",
            "venue",
            "actual_home_contract_id",
            "landmark_seconds",
            "endpoint_seconds",
            "availability_truth",
            "direction_truth",
            "cohort_authority_sha256",
            "panel_batch_sha256",
            "fold_id",
            *_PREDICTION_LOSS_COLUMNS,
        )
        for fold_id in _FOLD_IDS:
            baseline_frame = load_verified_regulation_oof_prediction_projection(
                by_cell[(fold_id, "b0_empirical_v1", "D0")],
                columns=projection_columns,
            )
            candidate_frame = (
                load_verified_regulation_oof_prediction_projection(
                    by_cell[(fold_id, model_id, feature_block_id)],
                    columns=projection_columns,
                )
            )
            paired = _paired_candidate_rows(
                baseline=baseline_frame,
                candidate=candidate_frame,
            )
            for metric in metric_names:
                metric_chunks[metric].append(
                    _game_level_improvement(paired, metric=metric)
                )
            anchor = paired.loc[
                paired["landmark_seconds"].eq(
                    SELECTION_ANCHOR_LANDMARK_SECONDS
                )
                & paired["endpoint_seconds"].eq(
                    SELECTION_ANCHOR_ENDPOINT_SECONDS
                )
            ]
            for metric in anchor_chunks:
                anchor_chunks[metric].append(
                    _game_level_improvement(anchor, metric=metric)
                )
            del baseline_frame, candidate_frame, paired, anchor
        game_values = {
            metric: pd.concat(chunks).sort_index()
            for metric, chunks in metric_chunks.items()
        }
        anchor_values = {
            metric: pd.concat(chunks).sort_index()
            for metric, chunks in anchor_chunks.items()
        }
        if any(values.index.duplicated().any() for values in game_values.values()):
            raise RegulationModelInputError(
                "selection fold projections contain duplicate games"
            )
        summaries: dict[str, tuple[float, float, float, float]] = {}
        for metric_index, metric in enumerate(metric_names):
            values = game_values[metric]
            summaries[metric] = _game_bootstrap_summary(
                values,
                seed=(
                    SELECTION_BOOTSTRAP_SEED
                    + 100 * candidate_index
                    + metric_index
                ),
            )
        anchor_joint = anchor_values["joint_log_loss"]
        anchor_direction = anchor_values["direction_log_loss"]
        anchor_direction_brier = anchor_values[
            "direction_multiclass_brier"
        ]
        anchor_games = len(
            set(anchor_joint.index)
            & set(anchor_direction.index)
            & set(anchor_direction_brier.index)
        )
        anchor_joint_mean = (
            float(anchor_joint.mean()) if len(anchor_joint) else math.nan
        )
        anchor_direction_mean = (
            float(anchor_direction.mean())
            if len(anchor_direction)
            else math.nan
        )
        anchor_direction_brier_mean = (
            float(anchor_direction_brier.mean())
            if len(anchor_direction_brier)
            else math.nan
        )
        availability_non_regression = (
            summaries["availability_log_loss"][2] >= 0
            and summaries["availability_brier"][2] >= 0
        )
        anchor_gate = (
            anchor_games >= SELECTION_MIN_ANCHOR_GAMES
            and anchor_joint_mean >= 0
            and anchor_direction_mean >= 0
            and anchor_direction_brier_mean >= 0
        )
        candidate_gate = (
            summaries["joint_log_loss"][2] > 0
            and summaries["direction_log_loss"][2] > 0
            and summaries["direction_multiclass_brier"][2] > 0
            and availability_non_regression
            and anchor_gate
        )
        audit.append(
            RegulationSelectionCandidate(
                model_id=model_id,
                feature_block_id=feature_block_id,
                game_count=len(game_values["joint_log_loss"]),
                integrated_joint_mean_improvement=(
                    summaries["joint_log_loss"][0]
                ),
                integrated_joint_standard_error=(
                    summaries["joint_log_loss"][1]
                ),
                integrated_joint_ci_low=summaries[
                    "joint_log_loss"
                ][2],
                integrated_joint_ci_high=summaries[
                    "joint_log_loss"
                ][3],
                availability_log_loss_ci_low=summaries[
                    "availability_log_loss"
                ][2],
                availability_brier_ci_low=summaries[
                    "availability_brier"
                ][2],
                direction_log_loss_ci_low=summaries[
                    "direction_log_loss"
                ][2],
                direction_brier_ci_low=summaries[
                    "direction_multiclass_brier"
                ][2],
                anchor_game_count=anchor_games,
                anchor_joint_mean_improvement=anchor_joint_mean,
                anchor_direction_mean_improvement=anchor_direction_mean,
                anchor_direction_brier_mean_improvement=(
                    anchor_direction_brier_mean
                ),
                availability_non_regression=availability_non_regression,
                anchor_gate_passed=anchor_gate,
                candidate_gate_passed=candidate_gate,
            )
        )
    passing = [candidate for candidate in audit if candidate.candidate_gate_passed]
    best_mean: float | None = None
    best_se: float | None = None
    threshold: float | None = None
    selected: RegulationSelectionCandidate | None = None
    if passing:
        best = max(
            passing,
            key=lambda candidate: (
                candidate.integrated_joint_mean_improvement,
                -SELECTABLE_CANDIDATES.index(
                    (candidate.model_id, candidate.feature_block_id)
                ),
            ),
        )
        best_mean = best.integrated_joint_mean_improvement
        best_se = best.integrated_joint_standard_error
        threshold = best_mean - best_se
        eligible = {
            (candidate.model_id, candidate.feature_block_id): candidate
            for candidate in passing
            if candidate.integrated_joint_mean_improvement >= threshold
        }
        selected = next(
            eligible[key]
            for key in SELECTABLE_CANDIDATES
            if key in eligible
        )
    selection_material = {
        "schema": SELECTION_SCHEMA,
        "selection_venue": POLYMARKET,
        "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
        "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
        "cross_anchor_pooling_allowed": False,
        "kalshi_selection_allowed": False,
        "holdout_reaction_accessed": False,
        "bootstrap_samples": SELECTION_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": SELECTION_BOOTSTRAP_SEED,
        "bootstrap_cluster": "game_id",
        "weighting": "ROW_TO_INFORMATION_ANCHOR_TO_GAME",
        "anchor": {
            "landmark_seconds": SELECTION_ANCHOR_LANDMARK_SECONDS,
            "endpoint_seconds": SELECTION_ANCHOR_ENDPOINT_SECONDS,
            "minimum_games": SELECTION_MIN_ANCHOR_GAMES,
        },
        "availability_non_regression": (
            "BOTH_LOG_LOSS_AND_BRIER_BOOTSTRAP_CI_LOW_GTE_ZERO"
        ),
        "candidate_suite": SELECTABLE_CANDIDATES,
        "candidate_audit": [asdict(candidate) for candidate in audit],
        "winner_identity": (
            None
            if selected is None
            else (selected.model_id, selected.feature_block_id)
        ),
        "one_se_threshold": threshold,
        "input_shards": [
            {
                "cell": shard.cell,
                "manifest_file_sha256": shard.manifest_file_sha256,
                "shard_sha256": shard.shard_sha256,
            }
            for shard in ordered
        ],
    }
    selection_sha256 = _sha256(selection_material)
    winner: FrozenPolymarketWinnerV2 | None = None
    if selected is not None:
        fold_hashes: dict[str, Mapping[str, str]] = {}
        for shard in ordered:
            if shard.cell[1:] != (
                selected.model_id,
                selected.feature_block_id,
            ):
                continue
            fold_hashes[shard.cell[0]] = MappingProxyType(
                {
                    column: str(shard.training_hashes[column])
                    for column in TRAINING_PROVENANCE_COLUMNS
                }
            )
        if set(fold_hashes) != set(_FOLD_IDS):
            raise RegulationModelInputError(
                "selected winner lacks five-fold training provenance"
            )
        winner = FrozenPolymarketWinnerV2(
            model_id=selected.model_id,
            feature_block_id=selected.feature_block_id,
            polymarket_selection_sha256=selection_sha256,
            anchor_kind=PRIMARY_ANCHOR_KIND,
            polymarket_fold_training_hashes=MappingProxyType(fold_hashes),
        )
    return RegulationSelectionResult(
        decision_status=(
            "WINNER_FROZEN" if winner is not None else "NO_CANDIDATE_PASSED"
        ),
        winner=winner,
        candidate_audit=tuple(audit),
        best_integrated_mean_improvement=best_mean,
        best_standard_error=best_se,
        one_se_threshold=threshold,
        selection_sha256=selection_sha256,
    )


def _fold_training_hashes_from_shards(
    shards: Sequence[VerifiedRegulationOOFShard],
    *,
    model_id: str,
    feature_block_id: str,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for shard in shards:
        if shard.cell[1:] != (model_id, feature_block_id):
            continue
        hashes = {
            column: str(shard.training_hashes[column])
            for column in TRAINING_PROVENANCE_COLUMNS
        }
        if any(
            _SHA256_RE.fullmatch(value) is None
            for value in hashes.values()
        ):
            raise RegulationModelInputError(
                "selection lock producer provenance is not SHA-256 bound"
            )
        result[shard.cell[0]] = hashes
    if set(result) != set(_FOLD_IDS):
        raise RegulationModelInputError(
            "selection lock producer lacks the exact five folds"
        )
    return result


def _selection_lock_material(
    *,
    result: RegulationSelectionResult,
    verified_panel: VerifiedRegulationPanelBatch,
    shards: Sequence[VerifiedRegulationOOFShard],
) -> dict[str, object]:
    if result.decision_status not in {
        "WINNER_FROZEN",
        "NO_CANDIDATE_PASSED",
    } or (result.winner is None) != (
        result.decision_status == "NO_CANDIDATE_PASSED"
    ):
        raise RegulationModelInputError(
            "selection result has an invalid terminal decision"
        )
    ordered = tuple(
        sorted(
            shards,
            key=lambda shard: CORRECTED_MODEL_CELLS.index(shard.cell),
        )
    )
    status = regulation_oof_batch_status(
        tuple(shard.cell for shard in ordered)
    )
    if (
        len(ordered) != len(CORRECTED_MODEL_CELLS)
        or status["run_status"] != "COMPLETE"
        or status["publication_gate"] != "PASS"
    ):
        raise RegulationModelInputError(
            "selection lock requires the verified exact-45 matrix"
        )
    recomputed = select_regulation_model(ordered)
    if (
        recomputed.selection_sha256 != result.selection_sha256
        or recomputed.decision_status != result.decision_status
        or (recomputed.winner is None) != (result.winner is None)
        or (
            recomputed.winner is not None
            and result.winner is not None
            and (
                recomputed.winner.model_id,
                recomputed.winner.feature_block_id,
            )
            != (
                result.winner.model_id,
                result.winner.feature_block_id,
            )
        )
    ):
        raise RegulationModelInputError(
            "selection result does not rederive from verified shards"
        )
    lineage = {
        "panel_batch_manifest_file_sha256": (
            verified_panel.manifest.batch_manifest_file_sha256
        ),
        "panel_batch_sha256": verified_panel.manifest.batch_sha256,
        "cohort_authority_sha256": (
            verified_panel.manifest.cohort_authority_sha256
        ),
        "cohort_mapping_sha256": (
            verified_panel.manifest.cohort_mapping_sha256
        ),
        "source_hashes": dict(verified_panel.manifest.sources),
    }
    for shard in ordered:
        shard_lineage = shard.run_config.get("input_lineage")
        if (
            shard.run_config.get("formal_input_verified") is not True
            or not isinstance(shard_lineage, Mapping)
            or shard_lineage.get("panel_batch_manifest_file_sha256")
            != lineage["panel_batch_manifest_file_sha256"]
            or shard_lineage.get("panel_batch_sha256")
            != lineage["panel_batch_sha256"]
        ):
            raise RegulationModelInputError(
                "selection lock shard lineage differs from PanelV4"
            )
    baseline_hashes = _fold_training_hashes_from_shards(
        ordered,
        model_id="b0_empirical_v1",
        feature_block_id="D0",
    )
    winner_hashes: dict[str, dict[str, str]] | None = None
    if result.winner is not None:
        winner_hashes = _fold_training_hashes_from_shards(
            ordered,
            model_id=result.winner.model_id,
            feature_block_id=result.winner.feature_block_id,
        )
        if _canonical(winner_hashes) != _canonical(
            result.winner.polymarket_fold_training_hashes
        ):
            raise RegulationModelInputError(
                "selected winner hashes differ from exact-45 source shards"
            )
    primary_poly = verified_panel.frame.loc[
        verified_panel.frame["venue"].eq(POLYMARKET)
        & verified_panel.frame["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & verified_panel.frame["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ].reset_index(drop=True)
    folds = _fold_map(primary_poly)
    fold_definitions: dict[str, dict[str, object]] = {}
    for fold_id in _FOLD_IDS:
        fold = folds[fold_id]
        fold_sha256 = _sha256(
            {
                "fold_id": fold.fold_id,
                "train_weeks": fold.train_weeks,
                "validation_weeks": fold.validation_weeks,
                "train_game_ids": fold.train_game_ids,
                "validation_game_ids": fold.validation_game_ids,
            }
        )
        if baseline_hashes[fold_id]["fold_sha256"] != fold_sha256 or (
            winner_hashes is not None
            and winner_hashes[fold_id]["fold_sha256"] != fold_sha256
        ):
            raise RegulationModelInputError(
                "selection lock fold definition differs from producer"
            )
        fold_definitions[fold_id] = {
            "fold_sha256": fold_sha256,
            "train_weeks": list(fold.train_weeks),
            "validation_weeks": list(fold.validation_weeks),
            "train_game_ids": list(fold.train_game_ids),
            "validation_game_ids": list(fold.validation_game_ids),
        }
    terminal_status = (
        "WINNER_FROZEN"
        if result.winner is not None
        else "NO_MODEL_ADVANCE"
    )
    winner_document: dict[str, object] | None = None
    if result.winner is not None and winner_hashes is not None:
        winner_document = {
            "model_id": result.winner.model_id,
            "feature_block_id": result.winner.feature_block_id,
            "polymarket_fold_training_hashes": winner_hashes,
        }
    return {
        "schema": SELECTION_LOCK_SCHEMA,
        "decision_status": terminal_status,
        "selection_result_status": result.decision_status,
        "selection_sha256": result.selection_sha256,
        "selection_venue": POLYMARKET,
        "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
        "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
        "holdout_reaction_accessed": False,
        "kalshi_selection_allowed": False,
        "kalshi_transport_allowed": result.winner is not None,
        "cross_anchor_pooling_allowed": False,
        "target": {
            "target_version": TRANSPORT_TARGET_VERSION,
            "availability": "availability_h",
            "direction": "direction_h_given_availability",
            "direction_threshold_probability": 0.01,
            "landmark_seconds": list(LANDMARK_SECONDS),
            "endpoint_seconds": list(ENDPOINT_SECONDS),
        },
        "selection_rule": {
            "bootstrap_samples": SELECTION_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": SELECTION_BOOTSTRAP_SEED,
            "bootstrap_cluster": "game_id",
            "weighting": "ROW_TO_INFORMATION_ANCHOR_TO_GAME",
            "anchor_landmark_seconds": (
                SELECTION_ANCHOR_LANDMARK_SECONDS
            ),
            "anchor_endpoint_seconds": (
                SELECTION_ANCHOR_ENDPOINT_SECONDS
            ),
            "minimum_anchor_games": SELECTION_MIN_ANCHOR_GAMES,
            "one_se_threshold": result.one_se_threshold,
            "candidate_suite": [
                list(value) for value in SELECTABLE_CANDIDATES
            ],
        },
        "winner": winner_document,
        "baseline": {
            "model_id": "b0_empirical_v1",
            "feature_block_id": "D0",
            "polymarket_fold_training_hashes": baseline_hashes,
        },
        "panel_lineage": lineage,
        "fold_definitions": fold_definitions,
        "candidate_audit": [
            asdict(candidate) for candidate in result.candidate_audit
        ],
        "input_shards": [
            {
                "cell": list(shard.cell),
                "manifest_path": str(shard.manifest_path),
                "manifest_file_sha256": shard.manifest_file_sha256,
                "shard_sha256": shard.shard_sha256,
            }
            for shard in ordered
        ],
        "publication_gate": "PASS",
    }


def publish_regulation_selection_lock(
    *,
    project_root: Path,
    output_root: Path,
    verified_panel: VerifiedRegulationPanelBatch,
    shards: Sequence[VerifiedRegulationOOFShard],
) -> PublishedRegulationSelectionLock:
    """Recompute and atomically freeze the complete Polymarket decision."""

    project = Path(project_root).resolve()
    output = _safe_project_path(
        project, Path(output_root), label="selection output root"
    )
    result = select_regulation_model(shards)
    material = _selection_lock_material(
        result=result,
        verified_panel=verified_panel,
        shards=shards,
    )
    selection_lock_sha256 = _sha256(material)
    document = {
        **material,
        "selection_lock_sha256": selection_lock_sha256,
    }
    payload = _canonical_json_bytes(document)
    file_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    hexadecimal = file_sha256.removeprefix("sha256:")
    path = (
        output
        / "selections"
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.selection-lock.json"
    )
    _atomic_publish(path, payload)
    return PublishedRegulationSelectionLock(
        manifest_path=path,
        manifest_file_sha256=file_sha256,
        selection_lock_sha256=selection_lock_sha256,
        decision_status=str(material["decision_status"]),
        winner_frozen=result.winner is not None,
    )


def verify_regulation_selection_lock(
    *,
    project_root: Path,
    output_root: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
    verified_panel: VerifiedRegulationPanelBatch,
    shards: Sequence[VerifiedRegulationOOFShard],
) -> VerifiedRegulationSelectionLock:
    """Verify bytes, PanelV4 lineage, folds, producers, and selection rules."""

    project = Path(project_root).resolve()
    output = _safe_project_path(
        project, Path(output_root), label="selection output root"
    )
    path = _safe_under(
        output, Path(manifest_path), label="selection lock manifest"
    )
    _verify_content_addressed_name(
        path,
        file_sha256=manifest_file_sha256,
        suffix=".selection-lock.json",
        label="selection lock manifest",
    )
    document = _verified_json_file(
        path,
        expected_sha256=manifest_file_sha256,
        label="selection lock manifest",
    )
    selection_lock_sha256 = str(
        document.get("selection_lock_sha256", "")
    )
    material = {
        key: value
        for key, value in document.items()
        if key != "selection_lock_sha256"
    }
    decision_status = str(document.get("decision_status", ""))
    winner_expected = decision_status == "WINNER_FROZEN"
    if (
        document.get("schema") != SELECTION_LOCK_SCHEMA
        or decision_status not in {"WINNER_FROZEN", "NO_MODEL_ADVANCE"}
        or document.get("publication_gate") != "PASS"
        or document.get("selection_venue") != POLYMARKET
        or document.get("selection_anchor_kind") != PRIMARY_ANCHOR_KIND
        or document.get("selection_analysis_role")
        != PRIMARY_ANALYSIS_ROLE
        or document.get("holdout_reaction_accessed") is not False
        or document.get("kalshi_selection_allowed") is not False
        or document.get("kalshi_transport_allowed") is not winner_expected
        or document.get("cross_anchor_pooling_allowed") is not False
        or _SHA256_RE.fullmatch(selection_lock_sha256) is None
        or _sha256(material) != selection_lock_sha256
    ):
        raise RegulationModelInputError(
            "selection lock governance or semantic hash failed"
        )
    recomputed = select_regulation_model(shards)
    expected_material = _selection_lock_material(
        result=recomputed,
        verified_panel=verified_panel,
        shards=shards,
    )
    if _canonical(material) != _canonical(expected_material):
        raise RegulationModelInputError(
            "selection lock does not rederive from PanelV4 and exact-45"
        )
    target = document.get("target")
    winner_document = document.get("winner")
    baseline_document = document.get("baseline")
    fold_definitions = document.get("fold_definitions")
    panel_lineage = document.get("panel_lineage")
    if (
        not isinstance(target, dict)
        or target.get("target_version") != TRANSPORT_TARGET_VERSION
        or tuple(target.get("landmark_seconds", ()))
        != LANDMARK_SECONDS
        or tuple(target.get("endpoint_seconds", ()))
        != ENDPOINT_SECONDS
        or float(target.get("direction_threshold_probability", math.nan))
        != 0.01
        or not isinstance(baseline_document, dict)
        or not isinstance(fold_definitions, dict)
        or set(fold_definitions) != set(_FOLD_IDS)
        or not isinstance(panel_lineage, dict)
    ):
        raise RegulationModelInputError(
            "selection lock target/fold identity is malformed"
        )
    baseline_hashes = _require_nested_training_hashes(
        baseline_document.get("polymarket_fold_training_hashes"),
        label="selection B0 producer hashes",
    )
    selection_sha256 = str(document.get("selection_sha256", ""))
    if (
        baseline_document.get("model_id") != "b0_empirical_v1"
        or baseline_document.get("feature_block_id") != "D0"
        or _SHA256_RE.fullmatch(selection_sha256) is None
        or selection_sha256 != recomputed.selection_sha256
    ):
        raise RegulationModelInputError(
            "selection lock frozen model identity is invalid"
        )
    winner: FrozenPolymarketWinnerV2 | None = None
    if winner_expected:
        if (
            not isinstance(winner_document, dict)
            or document.get("selection_result_status") != "WINNER_FROZEN"
        ):
            raise RegulationModelInputError(
                "winner lock must contain one frozen winner"
            )
        winner_identity = (
            str(winner_document.get("model_id")),
            str(winner_document.get("feature_block_id")),
        )
        if winner_identity not in SELECTABLE_CANDIDATES:
            raise RegulationModelInputError(
                "selection lock winner identity is invalid"
            )
        winner_hashes = _require_nested_training_hashes(
            winner_document.get("polymarket_fold_training_hashes"),
            label="selection winner producer hashes",
        )
        winner = FrozenPolymarketWinnerV2(
            model_id=winner_identity[0],
            feature_block_id=winner_identity[1],
            polymarket_selection_sha256=selection_sha256,
            anchor_kind=PRIMARY_ANCHOR_KIND,
            polymarket_fold_training_hashes=MappingProxyType(
                {
                    key: MappingProxyType(value)
                    for key, value in winner_hashes.items()
                }
            ),
        )
    elif (
        winner_document is not None
        or recomputed.winner is not None
        or document.get("selection_result_status")
        != "NO_CANDIDATE_PASSED"
    ):
        raise RegulationModelInputError(
            "NO_MODEL_ADVANCE lock must have a null winner"
        )
    return VerifiedRegulationSelectionLock(
        manifest_path=path,
        manifest_file_sha256=manifest_file_sha256,
        selection_lock_sha256=selection_lock_sha256,
        selection_sha256=selection_sha256,
        decision_status=decision_status,
        winner=winner,
        kalshi_transport_allowed=winner_expected,
        baseline_fold_training_hashes=MappingProxyType(
            {
                key: MappingProxyType(value)
                for key, value in baseline_hashes.items()
            }
        ),
        fold_definitions=MappingProxyType(
            {
                str(key): MappingProxyType(dict(value))
                for key, value in fold_definitions.items()
                if isinstance(value, dict)
            }
        ),
        panel_batch_manifest_file_sha256=str(
            panel_lineage["panel_batch_manifest_file_sha256"]
        ),
        panel_batch_sha256=str(panel_lineage["panel_batch_sha256"]),
        cohort_authority_sha256=str(
            panel_lineage["cohort_authority_sha256"]
        ),
        cohort_mapping_sha256=str(
            panel_lineage["cohort_mapping_sha256"]
        ),
        landmark_seconds=LANDMARK_SECONDS,
        endpoint_seconds=ENDPOINT_SECONDS,
        direction_threshold_probability=0.01,
        target_version=TRANSPORT_TARGET_VERSION,
    )


def _fold_map(frame: pd.DataFrame) -> dict[str, X15WeekFold]:
    return {
        fold.fold_id: fold
        for fold in build_x15_week_folds(frame)
    }


def _bound_calibration_sha256(
    *,
    head: str,
    calibration_status: str,
    raw_calibration_training_sha256: str,
    fitted: _TwoHeadPredictor,
    calibration_game_ids: Sequence[str],
) -> str:
    model_sha256 = (
        fitted.availability_model_training_sha256
        if head == "availability"
        else fitted.direction_model_training_sha256
    )
    preprocessor_sha256 = (
        fitted.availability_preprocessor_training_sha256
        if head == "availability"
        else fitted.direction_preprocessor_training_sha256
    )
    return _sha256(
        {
            "scheme": (
                "NONE_EMPIRICAL_BASELINE"
                if calibration_status
                == "NOT_APPLICABLE_EMPIRICAL_BASELINE"
                else "STRICT_LATER_WEEK_SAME_BASE_PRODUCER"
            ),
            "head": head,
            "calibration_status": calibration_status,
            "raw_calibration_training_sha256": (
                raw_calibration_training_sha256
            ),
            "base_preprocessor_training_sha256": (
                preprocessor_sha256
            ),
            "base_model_training_sha256": model_sha256,
            "calibration_game_ids": tuple(sorted(calibration_game_ids)),
        }
    )


def _producer_provenance_hashes(
    *,
    fitted: _TwoHeadPredictor,
    model_id: str,
    producer_train: pd.DataFrame,
    calibration: pd.DataFrame,
    availability_calibrator: BinaryPlattCalibrator | None,
    direction_calibrator: DirectionTemperatureCalibrator | None,
    availability_calibration_status: str,
    direction_calibration_status: str,
    availability_raw_calibration_sha256: str,
    direction_raw_calibration_sha256: str,
    availability_calibration_training_sha256: str,
    direction_calibration_training_sha256: str,
) -> dict[str, str]:
    calibration_empty_sha = _sha256(
        {"scheme": "NONE_EMPIRICAL_BASELINE", "rows": 0}
    )
    return {
        "availability_preprocessor_spec_sha256": (
            fitted.availability_preprocessor_spec_sha256
        ),
        "availability_preprocessor_training_sha256": (
            fitted.availability_preprocessor_training_sha256
        ),
        "availability_preprocessor_fitted_sha256": (
            fitted.availability_preprocessor_fitted_sha256
        ),
        "direction_preprocessor_spec_sha256": (
            fitted.direction_preprocessor_spec_sha256
        ),
        "direction_preprocessor_training_sha256": (
            fitted.direction_preprocessor_training_sha256
        ),
        "direction_preprocessor_fitted_sha256": (
            fitted.direction_preprocessor_fitted_sha256
        ),
        "availability_model_spec_sha256": (
            fitted.availability_model_spec_sha256
        ),
        "availability_model_training_sha256": (
            fitted.availability_model_training_sha256
        ),
        "availability_model_fitted_sha256": (
            fitted.availability_model_fitted_sha256
        ),
        "direction_model_spec_sha256": (
            fitted.direction_model_spec_sha256
        ),
        "direction_model_training_sha256": (
            fitted.direction_model_training_sha256
        ),
        "direction_model_fitted_sha256": (
            fitted.direction_model_fitted_sha256
        ),
        "availability_calibration_spec_sha256": (
            _calibration_spec_sha256(
                head="availability",
                model_id=model_id,
            )
        ),
        "availability_calibration_training_sha256": (
            availability_calibration_training_sha256
        ),
        "availability_calibrator_fitted_sha256": (
            _fitted_calibrator_sha256(
                availability_calibrator,
                head="availability",
            )
        ),
        "direction_calibration_spec_sha256": (
            _calibration_spec_sha256(
                head="direction",
                model_id=model_id,
            )
        ),
        "direction_calibration_training_sha256": (
            direction_calibration_training_sha256
        ),
        "direction_calibrator_fitted_sha256": (
            _fitted_calibrator_sha256(
                direction_calibrator,
                head="direction",
            )
        ),
        "availability_calibration_raw_training_sha256": (
            availability_raw_calibration_sha256
        ),
        "direction_calibration_raw_training_sha256": (
            direction_raw_calibration_sha256
        ),
        "availability_calibrator_base_preprocessor_sha256": (
            fitted.availability_preprocessor_training_sha256
        ),
        "direction_calibrator_base_preprocessor_sha256": (
            fitted.direction_preprocessor_training_sha256
        ),
        "availability_calibrator_base_model_sha256": (
            fitted.availability_model_training_sha256
        ),
        "direction_calibrator_base_model_sha256": (
            fitted.direction_model_training_sha256
        ),
        "producer_primary_snapshot_set_sha256": (
            _primary_snapshot_set_sha256(producer_train)
        ),
        "calibration_primary_snapshot_set_sha256": (
            calibration_empty_sha
            if calibration.empty
            else _primary_snapshot_set_sha256(calibration)
        ),
    }


def _prediction_rows(
    validation: pd.DataFrame,
    *,
    fold_id: str,
    model_id: str,
    feature_block_id: str,
    availability_probability: np.ndarray,
    direction_probability: np.ndarray,
    availability_calibration_status: str,
    direction_calibration_status: str,
    availability_calibration_training_sha256: str,
    direction_calibration_training_sha256: str,
    fitted: _TwoHeadPredictor,
    producer_provenance: Mapping[str, str],
    fold_sha256: str,
    input_lineage: Mapping[str, object],
    training_venue: str,
    evaluation_venue: str,
    transport_mode: str,
) -> pd.DataFrame:
    identity = validation.loc[
        :,
        [
            "source_row_id",
            "game_id",
            "nfl_week",
            "information_anchor_id",
            "episode_id",
            "anchor_kind",
            "analysis_role",
            "venue",
            "actual_home_contract_id",
            "logical_market_id",
            "market_family",
            "market_period",
            "proposition_kind",
            "home_team",
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
            "landmark_seconds",
            "endpoint_seconds",
            "availability_h",
            "direction_h",
            "cohort_authority_sha256",
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "primary_feature_known_at",
            "primary_feature_snapshot_sha256",
            "prestate_known_at",
            "prestate_context_sha256",
            "p_before_support_status",
        ],
    ].reset_index(drop=True)
    identity = identity.rename(
        columns={
            "availability_h": "availability_truth",
            "direction_h": "direction_truth",
        }
    )
    identity["fold_id"] = fold_id
    identity["model_id"] = model_id
    identity["feature_block_id"] = feature_block_id
    identity["training_venue"] = training_venue
    identity["evaluation_venue"] = evaluation_venue
    identity["transport_mode"] = transport_mode
    identity["availability_probability"] = availability_probability
    identity["direction_probability_down"] = direction_probability[:, 0]
    identity["direction_probability_no_move"] = direction_probability[:, 1]
    identity["direction_probability_up"] = direction_probability[:, 2]
    identity["availability_calibration_status"] = (
        availability_calibration_status
    )
    identity["direction_calibration_status"] = (
        direction_calibration_status
    )
    identity["fold_sha256"] = fold_sha256
    if set(producer_provenance) != (
        set(TRAINING_PROVENANCE_COLUMNS) - {"fold_sha256"}
    ):
        raise RegulationModelInputError(
            "prediction producer provenance is incomplete"
        )
    for key, value in producer_provenance.items():
        if _SHA256_RE.fullmatch(str(value)) is None:
            raise RegulationModelInputError(
                f"prediction producer provenance {key} is invalid"
            )
        identity[key] = str(value)
    if (
        identity[
            "availability_preprocessor_training_sha256"
        ].iloc[0]
        != fitted.availability_preprocessor_training_sha256
        or identity[
            "direction_preprocessor_training_sha256"
        ].iloc[0]
        != fitted.direction_preprocessor_training_sha256
        or identity["availability_model_training_sha256"].iloc[0]
        != fitted.availability_model_training_sha256
        or identity["direction_model_training_sha256"].iloc[0]
        != fitted.direction_model_training_sha256
        or identity[
            "availability_calibration_training_sha256"
        ].iloc[0]
        != availability_calibration_training_sha256
        or identity[
            "direction_calibration_training_sha256"
        ].iloc[0]
        != direction_calibration_training_sha256
    ):
        raise RegulationModelInputError(
            "prediction producer provenance disagrees with fitted objects"
        )
    identity["panel_batch_sha256"] = input_lineage.get(
        "panel_batch_sha256", "UNVERIFIED_SYNTHETIC_API"
    )
    losses = regulation_row_losses(
        availability_truth=identity["availability_truth"].to_numpy(dtype=int),
        availability_probability=availability_probability,
        direction_truth=identity["direction_truth"].to_numpy(dtype=object),
        direction_probability=direction_probability,
    )
    return pd.concat((identity, losses), axis=1)


def run_regulation_walk_forward(
    panel: pd.DataFrame | VerifiedRegulationPanelBatch,
    *,
    cells: Sequence[tuple[str, str, str]] = CORRECTED_MODEL_CELLS,
    random_state: int = RANDOM_STATE,
) -> RegulationModelRun:
    """Run corrected OOF cells; development selection is Polymarket-only."""

    requested = _validate_cells(cells)
    if isinstance(panel, VerifiedRegulationPanelBatch):
        if (
            not panel.formal_input_verified
            or panel.loaded_venue_scope != (POLYMARKET,)
            or panel.loaded_row_count != len(panel.frame)
            or not panel.frame["venue"].eq(POLYMARKET).all()
        ):
            raise RegulationModelInputError(
                "formal selection requires verified POLYMARKET-only PanelV4"
            )
        prepared = panel.frame
        formal_input_verified = True
        input_lineage: Mapping[str, object] = {
            "panel_batch_manifest_file_sha256": (
                panel.manifest.batch_manifest_file_sha256
            ),
            "panel_batch_sha256": panel.manifest.batch_sha256,
            "panel_cohort_authority_sha256": (
                panel.manifest.cohort_authority_sha256
            ),
            "panel_cohort_mapping_sha256": (
                panel.manifest.cohort_mapping_sha256
            ),
            "panel_source_hashes": dict(panel.manifest.sources),
        }
    else:
        prepared = prepare_regulation_panel(panel)
        formal_input_verified = False
        input_lineage = {"status": "UNVERIFIED_SYNTHETIC_API"}
    polymarket = prepared.loc[
        prepared["venue"].eq(POLYMARKET)
        & prepared["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & prepared["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ].reset_index(drop=True)
    if polymarket.empty:
        raise RegulationModelInputError(
            "development selection requires Polymarket "
            "PROVISIONAL_FIRST_SEEN / PRIMARY_SELECTION rows"
        )
    folds = _fold_map(polymarket)
    predictions: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    for fold_id, model_id, feature_block_id in requested:
        fold = folds[fold_id]
        train = polymarket.loc[
            polymarket["game_id"].isin(fold.train_game_ids)
        ].reset_index(drop=True)
        validation = polymarket.loc[
            polymarket["game_id"].isin(fold.validation_game_ids)
        ].reset_index(drop=True)
        if train.empty or validation.empty:
            raise RegulationModelInputError(
                f"{fold_id} has empty train or validation rows"
            )
        fold_sha256 = _sha256(
            {
                "fold_id": fold.fold_id,
                "train_weeks": fold.train_weeks,
                "validation_weeks": fold.validation_weeks,
                "train_game_ids": fold.train_game_ids,
                "validation_game_ids": fold.validation_game_ids,
            }
        )
        seed = effective_random_state(
            base_random_state=random_state,
            fold_id=fold_id,
            training_venue=POLYMARKET,
            evaluation_venue=POLYMARKET,
            transport_mode="NATIVE_POLYMARKET_SELECTION",
            feature_block_id=feature_block_id,
            model_id=model_id,
            purpose="REGULATION_TWO_HEAD_FIT",
        )
        if model_id == "b0_empirical_v1":
            producer_train = train
            calibration = train.iloc[0:0].copy()
            fitted = _fit_two_heads(
                producer_train,
                model_id=model_id,
                feature_block_id=feature_block_id,
                random_state=seed,
            )
            availability_calibrator = None
            direction_calibrator = None
            availability_calibration_status = (
                "NOT_APPLICABLE_EMPIRICAL_BASELINE"
            )
            direction_calibration_status = (
                "NOT_APPLICABLE_EMPIRICAL_BASELINE"
            )
            availability_raw_calibration_sha256 = _sha256(
                {
                    "scheme": "NONE_EMPIRICAL_BASELINE",
                    "head": "availability",
                }
            )
            direction_raw_calibration_sha256 = _sha256(
                {
                    "scheme": "NONE_EMPIRICAL_BASELINE",
                    "head": "direction",
                }
            )
        else:
            (
                fitted,
                availability_calibrator,
                direction_calibrator,
                producer_train,
                calibration,
            ) = _raw_calibrators(
                outer_train=train,
                model_id=model_id,
                feature_block_id=feature_block_id,
                random_state=seed,
            )
            if (
                availability_calibrator.status != CALIBRATED
                or direction_calibrator.status != CALIBRATED
            ):
                raise RegulationModelInputError(
                    f"{fold_id}/{model_id}/{feature_block_id} "
                    "strict-later calibration is unsupported"
                )
            availability_calibration_status = (
                availability_calibrator.status
            )
            direction_calibration_status = direction_calibrator.status
            availability_raw_calibration_sha256 = (
                availability_calibrator.training_sha256
            )
            direction_raw_calibration_sha256 = (
                direction_calibrator.training_sha256
            )
        calibration_game_ids = tuple(
            sorted(calibration["game_id"].astype(str).unique())
        )
        availability_calibration_training_sha256 = (
            _bound_calibration_sha256(
                head="availability",
                calibration_status=availability_calibration_status,
                raw_calibration_training_sha256=(
                    availability_raw_calibration_sha256
                ),
                fitted=fitted,
                calibration_game_ids=calibration_game_ids,
            )
        )
        direction_calibration_training_sha256 = (
            _bound_calibration_sha256(
                head="direction",
                calibration_status=direction_calibration_status,
                raw_calibration_training_sha256=(
                    direction_raw_calibration_sha256
                ),
                fitted=fitted,
                calibration_game_ids=calibration_game_ids,
            )
        )
        availability_raw, direction_raw = _predict_two_heads(
            fitted,
            validation,
            model_id=model_id,
            feature_block_id=feature_block_id,
        )
        availability_probability = (
            availability_raw
            if availability_calibrator is None
            else availability_calibrator.transform(availability_raw)
        )
        direction_probability = (
            direction_raw
            if direction_calibrator is None
            else direction_calibrator.transform(direction_raw)
        )
        producer_provenance = _producer_provenance_hashes(
            fitted=fitted,
            model_id=model_id,
            producer_train=producer_train,
            calibration=calibration,
            availability_calibrator=availability_calibrator,
            direction_calibrator=direction_calibrator,
            availability_calibration_status=(
                availability_calibration_status
            ),
            direction_calibration_status=direction_calibration_status,
            availability_raw_calibration_sha256=(
                availability_raw_calibration_sha256
            ),
            direction_raw_calibration_sha256=(
                direction_raw_calibration_sha256
            ),
            availability_calibration_training_sha256=(
                availability_calibration_training_sha256
            ),
            direction_calibration_training_sha256=(
                direction_calibration_training_sha256
            ),
        )
        cell_predictions = _prediction_rows(
            validation,
            fold_id=fold_id,
            model_id=model_id,
            feature_block_id=feature_block_id,
            availability_probability=availability_probability,
            direction_probability=direction_probability,
            availability_calibration_status=(
                availability_calibration_status
            ),
            direction_calibration_status=direction_calibration_status,
            availability_calibration_training_sha256=(
                availability_calibration_training_sha256
            ),
            direction_calibration_training_sha256=(
                direction_calibration_training_sha256
            ),
            fitted=fitted,
            producer_provenance=producer_provenance,
            fold_sha256=fold_sha256,
            input_lineage=input_lineage,
            training_venue=POLYMARKET,
            evaluation_venue=POLYMARKET,
            transport_mode="NATIVE_POLYMARKET_SELECTION",
        )
        predictions.append(cell_predictions)
        support_rows.append(
            {
                "fold_id": fold_id,
                "model_id": model_id,
                "feature_block_id": feature_block_id,
                "outer_training_game_count": len(fold.train_game_ids),
                "producer_training_game_count": int(
                    producer_train["game_id"].nunique()
                ),
                "calibration_game_count": int(
                    calibration["game_id"].nunique()
                ),
                "validation_game_count": len(fold.validation_game_ids),
                "outer_training_row_count": len(train),
                "producer_training_row_count": len(producer_train),
                "calibration_row_count": len(calibration),
                "validation_row_count": len(validation),
                "availability_training_positive_count": int(
                    producer_train["availability_h"].sum()
                ),
                "direction_training_row_count": int(
                    producer_train["availability_h"].sum()
                ),
                "producer_training_game_ids": json.dumps(
                    sorted(producer_train["game_id"].astype(str).unique()),
                    separators=(",", ":"),
                ),
                "calibration_game_ids": json.dumps(
                    list(calibration_game_ids), separators=(",", ":")
                ),
                "producer_max_week": int(
                    producer_train["nfl_week"].max()
                ),
                "calibration_week": (
                    None
                    if calibration.empty
                    else int(calibration["nfl_week"].min())
                ),
                "availability_calibration_status": (
                    availability_calibration_status
                ),
                "direction_calibration_status": (
                    direction_calibration_status
                ),
                "fold_sha256": fold_sha256,
                **producer_provenance,
                "effective_random_state": seed,
            }
        )

    oof = pd.concat(predictions, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for keys, candidate in oof.groupby(
        ["fold_id", "model_id", "feature_block_id"],
        sort=True,
    ):
        metrics = compute_equal_game_metrics(candidate)
        for metric_name, metric_value in metrics.items():
            metric_rows.append(
                {
                    "fold_id": keys[0],
                    "model_id": keys[1],
                    "feature_block_id": keys[2],
                    "metric_scope": "EQUAL_GAME",
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "game_count": int(candidate["game_id"].nunique()),
                }
            )
    run_config: Mapping[str, object] = MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_split": DEVELOPMENT_SPLIT,
            "selection_venue": POLYMARKET,
            "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
            "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
            "primary_feature_contract": dict(PRIMARY_FEATURE_CONTRACT),
            "prestate_context_contract": dict(PRESTATE_CONTEXT_CONTRACT),
            "market_continuity_contract": dict(
                MARKET_CONTINUITY_CONTRACT
            ),
            "formal_model_required_columns": list(
                REQUIRED_PANEL_COLUMNS
            ),
            "formal_model_forbidden_columns": sorted(
                PANEL_FORMAL_MODEL_FORBIDDEN_COLUMNS
            ),
            "cross_anchor_pooling_allowed": False,
            "kalshi_selection_allowed": False,
            "holdout_reaction_accessed": False,
            "formal_input_verified": formal_input_verified,
            "input_lineage": input_lineage,
            "targets": (
                "availability_h",
                "direction_h_given_availability",
            ),
            "joint_loss": (
                "NLL(availability_h)+"
                "I(availability_h)*NLL(direction_h)"
            ),
            "censored_rows": "EXCLUDED_FROM_BOTH_HEADS",
            "calibration_contract": {
                "b0_empirical_v1": (
                    "NO_LEARNED_CALIBRATION; FIT_ALL_OUTER_TRAIN_WEEKS"
                ),
                "regularized_logistic_v1": (
                    "BASE_FIT_BEFORE_LAST_OUTER_TRAIN_WEEK; "
                    "CALIBRATE_ON_LAST_OUTER_TRAIN_WEEK; "
                    "NO_POST_CALIBRATION_REFIT"
                ),
                "shallow_xgboost_v1": (
                    "BASE_FIT_BEFORE_LAST_OUTER_TRAIN_WEEK; "
                    "CALIBRATE_ON_LAST_OUTER_TRAIN_WEEK; "
                    "NO_POST_CALIBRATION_REFIT"
                ),
            },
            "weighting_contract": (
                "ROWS_TO_PRIMARY_INFORMATION_ANCHOR_TO_GAME"
            ),
            "preprocessing_contract": (
                "HIERARCHICAL_WEIGHTED_MEDIAN_IMPUTE_AND_SCALE; "
                "FIXED_MISSING_ONEHOT"
            ),
            "feature_blocks": dict(FEATURE_BLOCKS),
            "cells": requested,
            "random_state": int(random_state),
        }
    )
    if _FORBIDDEN_RUN_CONFIG_KEYS & _mapping_keys(run_config):
        raise AssertionError("legacy sports-survival target leaked into config")
    return RegulationModelRun(
        oof_predictions=oof,
        fold_metrics=pd.DataFrame(metric_rows),
        support_audit=pd.DataFrame(support_rows),
        run_config=run_config,
        run_config_sha256=_sha256(run_config),
    )


def _panel_lineage_set_sha256(values: Sequence[str]) -> str:
    payload = (
        json.dumps(
            _canonical(sorted(values)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_transport_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RegulationModelInputError(
            f"{label} must be a canonical SHA-256"
        )
    return value


def _optional_transport_sha(value: object, *, label: str) -> str | None:
    if value is None or value is pd.NA or (
        isinstance(pd.isna(value), (bool, np.bool_))
        and bool(pd.isna(value))
    ):
        return None
    return _require_transport_sha(value, label=label)


_PANEL_TRANSPORT_CONTRACT_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "venue",
    "home_team",
    "actual_home_contract_id",
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
)


def _transport_game_teams(game_id: str) -> tuple[str, str]:
    parts = str(game_id).split("_", 3)
    if len(parts) != 4 or not parts[2] or not parts[3]:
        raise RegulationModelInputError(
            "transport game_id does not encode away/home teams"
        )
    return parts[2], parts[3]


def _panel_transport_contract_rows(
    frame: pd.DataFrame,
) -> tuple[Mapping[str, object], ...]:
    missing = set(_PANEL_TRANSPORT_CONTRACT_COLUMNS) - set(frame.columns)
    if missing:
        raise RegulationModelInputError(
            "transport Panel contract identity is incomplete: "
            f"{sorted(missing)}"
        )
    contracts = (
        frame.loc[:, list(_PANEL_TRANSPORT_CONTRACT_COLUMNS)]
        .drop_duplicates()
        .sort_values(["game_id", "venue"], kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        contracts.empty
        or contracts.duplicated(["game_id", "venue"]).any()
    ):
        raise RegulationModelInputError(
            "transport Panel contract identity is not one-to-one by "
            "game_id+venue"
        )
    rows: list[Mapping[str, object]] = []
    for row in contracts.to_dict("records"):
        game_id = str(row["game_id"])
        venue = str(row["venue"])
        _, encoded_home = _transport_game_teams(game_id)
        if (
            venue not in {POLYMARKET, KALSHI}
            or str(row["market_family"]) != "moneyline"
            or str(row["market_period"]) != "full_game"
            or str(row["proposition_kind"]) != "primitive"
            or str(row["home_team"]) != encoded_home
            or str(row["outcome_team"]) != encoded_home
        ):
            raise RegulationModelInputError(
                "transport accepts one native primitive full-game "
                "home-team moneyline contract per game+venue"
            )
        for label in ("actual_home_contract_id", "logical_market_id"):
            if not str(row[label]).strip():
                raise RegulationModelInputError(
                    f"transport {label} must be nonempty"
                )
        _require_transport_sha(
            row["actual_home_contract_identity_sha256"],
            label="actual_home_contract_identity_sha256",
        )
        for kind in ("manifest", "object"):
            json_column = f"native_raw_{kind}_sha256s_json"
            set_column = f"native_raw_{kind}_set_sha256"
            try:
                values = json.loads(str(row[json_column]))
            except json.JSONDecodeError as error:
                raise RegulationModelInputError(
                    f"Panel native raw {kind} lineage JSON is invalid"
                ) from error
            if (
                not isinstance(values, list)
                or not values
                or values != sorted(set(values))
            ):
                raise RegulationModelInputError(
                    f"Panel native raw {kind} lineage must be sorted "
                    "and unique"
                )
            canonical_values = [
                _require_transport_sha(
                    value, label=f"Panel native raw {kind} lineage"
                )
                for value in values
            ]
            if _panel_lineage_set_sha256(
                canonical_values
            ) != _require_transport_sha(
                row[set_column],
                label=f"Panel native raw {kind} set SHA-256",
            ):
                raise RegulationModelInputError(
                    f"Panel native raw {kind} lineage set/hash mismatch"
                )
        rows.append(
            MappingProxyType(
                {
                    key: (
                        str(value)
                        if not isinstance(value, (bool, int, float))
                        else value
                    )
                    for key, value in row.items()
                }
            )
        )
    return tuple(rows)


def _panel_transport_contract_authority_sha256(
    frame: pd.DataFrame,
) -> str:
    rows = _panel_transport_contract_rows(frame)
    return _sha256(
        {
            "schema": "PanelTransportContractAuthorityV1",
            "rows": [dict(row) for row in rows],
        }
    )


def _require_audit_hash_list(
    value: object,
    *,
    expected_set_sha256: object,
    label: str,
) -> tuple[str, ...]:
    try:
        values = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RegulationModelInputError(
            f"{label} JSON is invalid"
        ) from error
    if (
        not isinstance(values, list)
        or not values
        or values != sorted(set(values))
    ):
        raise RegulationModelInputError(
            f"{label} must be a nonempty sorted unique list"
        )
    result = tuple(
        _require_transport_sha(child, label=label) for child in values
    )
    if _sha256(list(result)) != _require_transport_sha(
        expected_set_sha256,
        label=f"{label} set SHA-256",
    ):
        raise RegulationModelInputError(f"{label} set/hash mismatch")
    return result


def _read_native_rule_table(
    *,
    project_root: Path,
    output_root: Path,
    descriptor: Mapping[str, object],
    name: str,
) -> pd.DataFrame:
    if descriptor.get("name") != name:
        raise RegulationModelInputError(
            f"native-rule table {name} descriptor mismatch"
        )
    path = _safe_project_path(
        project_root,
        output_root / Path(str(descriptor.get("object_path", ""))),
        label=f"native-rule {name} object",
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != descriptor.get("object_sha256")
        or path.stat().st_size != descriptor.get("byte_length")
    ):
        raise RegulationModelInputError(
            f"native-rule {name} object integrity failed"
        )
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise RegulationModelInputError(
            f"native-rule {name} object is unreadable"
        ) from error
    if (
        table.num_rows != descriptor.get("row_count")
        or table.column_names != descriptor.get("schema_columns")
    ):
        raise RegulationModelInputError(
            f"native-rule {name} table contract mismatch"
        )
    return table.to_pandas()


def _native_market_membership_evidence(
    *,
    raw_root: Path,
    game_id: str,
    venue: str,
    panel_contract_id: str,
    native_contract_ids: tuple[str, ...],
    audit_manifest_hashes: tuple[str, ...],
    audit_object_hashes: tuple[str, ...],
    audit_request_hashes: tuple[str, ...],
    capture_index: Mapping[str, object],
) -> str:
    refs = capture_index.get("raw_manifests")
    if not isinstance(refs, list):
        raise RegulationModelInputError(
            f"{game_id} capture index lacks raw manifests"
        )
    matched = [
        ref
        for ref in refs
        if isinstance(ref, dict)
        and ref.get("object_sha256") in audit_object_hashes
    ]
    if (
        len(matched) != len(audit_object_hashes)
        or {
            str(ref.get("manifest_sha256", "")) for ref in matched
        }
        != set(audit_manifest_hashes)
        or {
            str(ref.get("request_sha256", "")) for ref in matched
        }
        != set(audit_request_hashes)
    ):
        raise RegulationModelInputError(
            f"{game_id} {venue} metadata lineage is not bound to the "
            "native-rule audit"
        )
    documents: list[Mapping[str, object]] = []
    for ref in matched:
        object_path = _safe_project_path(
            raw_root,
            raw_root / Path(str(ref.get("object_path", ""))),
            label=f"{game_id} {venue} metadata object",
        )
        expected_sha = _require_transport_sha(
            ref.get("object_sha256"),
            label=f"{game_id} {venue} metadata object SHA-256",
        )
        document = _verified_json_file(
            object_path,
            expected_sha256=expected_sha,
            label=f"{game_id} {venue} metadata object",
        )
        if object_path.stat().st_size != int(ref.get("byte_length", -1)):
            raise RegulationModelInputError(
                f"{game_id} {venue} metadata byte length mismatch"
            )
        documents.append(document)
    membership_material: dict[str, object]
    if venue == KALSHI:
        if panel_contract_id not in native_contract_ids:
            raise RegulationModelInputError(
                f"{game_id} Kalshi Panel ticker membership is unproven"
            )
        membership_material = {
            "venue": venue,
            "panel_home_contract_id": panel_contract_id,
            "native_contract_ids": list(native_contract_ids),
            "metadata_object_sha256s": list(audit_object_hashes),
        }
    elif venue == POLYMARKET:
        selected_markets: list[Mapping[str, object]] = []
        for document in documents:
            markets = document.get("markets")
            if not isinstance(markets, list):
                continue
            selected_markets.extend(
                market
                for market in markets
                if isinstance(market, dict)
                and str(market.get("id", "")) in native_contract_ids
            )
        if len(selected_markets) != 1:
            raise RegulationModelInputError(
                f"{game_id} Polymarket native market identity is unproven"
            )
        try:
            tokens = json.loads(
                str(selected_markets[0].get("clobTokenIds", ""))
            )
        except json.JSONDecodeError as error:
            raise RegulationModelInputError(
                f"{game_id} Polymarket token membership is malformed"
            ) from error
        if (
            not isinstance(tokens, list)
            or panel_contract_id not in tokens
        ):
            raise RegulationModelInputError(
                f"{game_id} Polymarket Panel token membership is unproven"
            )
        membership_material = {
            "venue": venue,
            "panel_home_contract_id": panel_contract_id,
            "native_market_id": str(selected_markets[0]["id"]),
            "native_clob_token_ids": sorted(str(token) for token in tokens),
            "metadata_object_sha256s": list(audit_object_hashes),
        }
    else:
        raise RegulationModelInputError(
            "native-rule overlay venue is invalid"
        )
    return _sha256(
        {
            "schema": "NativeMarketMembershipEvidenceV1",
            **membership_material,
        }
    )


def _native_clause_map(
    row: Mapping[str, object],
    *,
    game_id: str,
    venue: str,
) -> Mapping[str, str]:
    try:
        value = json.loads(str(row.get("normalized_clauses_json", "")))
    except json.JSONDecodeError as error:
        raise RegulationModelInputError(
            f"{game_id} {venue} normalized clauses are malformed"
        ) from error
    required = (
        "winner_clause",
        "postponement_clause",
        "cancellation_clause",
        "tie_clause",
        "resolution_source",
    )
    if not isinstance(value, dict) or set(value) != set(required):
        raise RegulationModelInputError(
            f"{game_id} {venue} normalized clause inventory is not exact"
        )
    clauses = {key: str(value[key]) for key in required}
    if any(
        str(row.get(f"{key}_status", "")) != clauses[key]
        for key in required
    ):
        raise RegulationModelInputError(
            f"{game_id} {venue} normalized clause columns disagree"
        )
    return MappingProxyType(clauses)


def _rederive_native_rule_pair_gate(
    *,
    game_id: str,
    evidence_by_venue: Mapping[str, Mapping[str, object]],
    pair: Mapping[str, object],
) -> tuple[str, str, str, str]:
    """Recompute both governed pair gates from the two venue evidence rows."""

    if set(evidence_by_venue) != {POLYMARKET, KALSHI}:
        raise RegulationModelInputError(
            f"{game_id} native-rule pair lacks exact dual-venue evidence"
        )
    poly = evidence_by_venue[POLYMARKET]
    kalshi = evidence_by_venue[KALSHI]
    home_team = str(pair.get("home_team", ""))
    away_team = str(pair.get("away_team", ""))
    try:
        home_score = int(pair["final_home_score"])
        away_score = int(pair["final_away_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise RegulationModelInputError(
            f"{game_id} native-rule pair final score is invalid"
        ) from error
    completed_non_tie = home_score != away_score
    winner = home_team if home_score > away_score else away_team
    facts_manifest_sha = _require_transport_sha(
        pair.get("facts_manifest_sha256"),
        label=f"{game_id} native-rule facts manifest",
    )
    facts_object_sha = _require_transport_sha(
        pair.get("facts_object_sha256"),
        label=f"{game_id} native-rule facts object",
    )
    for venue, row in evidence_by_venue.items():
        if (
            str(row.get("game_id", "")) != game_id
            or str(row.get("venue", "")) != venue
            or str(row.get("home_team", "")) != home_team
            or str(row.get("away_team", "")) != away_team
            or int(row.get("final_home_score", -1)) != home_score
            or int(row.get("final_away_score", -1)) != away_score
            or bool(row.get("completed_non_tie")) != completed_non_tie
            or row.get("facts_manifest_sha256") != facts_manifest_sha
            or row.get("facts_object_sha256") != facts_object_sha
        ):
            raise RegulationModelInputError(
                f"{game_id} {venue} native evidence disagrees with pair facts"
            )
    if (
        bool(pair.get("completed_non_tie")) != completed_non_tie
        or pair.get("realized_winner_team")
        != (winner if completed_non_tie else None)
        or pair.get("polymarket_contract_identity_sha256")
        != poly.get("contract_identity_sha256")
        or pair.get("kalshi_contract_identity_sha256")
        != kalshi.get("contract_identity_sha256")
        or pair.get("polymarket_rule_text_sha256")
        != poly.get("rule_text_sha256")
        or pair.get("kalshi_rule_text_sha256")
        != kalshi.get("rule_text_sha256")
    ):
        raise RegulationModelInputError(
            f"{game_id} native-rule pair identity or realized winner mismatch"
        )
    clauses = {
        POLYMARKET: _native_clause_map(
            poly, game_id=game_id, venue=POLYMARKET
        ),
        KALSHI: _native_clause_map(
            kalshi, game_id=game_id, venue=KALSHI
        ),
    }
    winner_clauses = [
        str(clauses[POLYMARKET]["winner_clause"]),
        str(clauses[KALSHI]["winner_clause"]),
    ]
    comparable_status = (
        NATIVE_RULE_OVERLAY_GATE_ONE_COMPARABLE
        if completed_non_tie
        and all(value == "SUPPORTED" for value in winner_clauses)
        else "UNPROVEN"
    )
    comparable_sha = _sha256(
        {
            "game_id": game_id,
            "facts": [facts_manifest_sha, facts_object_sha],
            "winner": winner if completed_non_tie else None,
            # Exact raw metadata membership is verified before this helper
            # is called, so identity support is independently rederived.
            "identity_ok": True,
            "winner_clauses": winner_clauses,
        }
    )
    if (
        pair.get("completed_non_tie_comparability_status")
        != comparable_status
        or pair.get(
            "completed_non_tie_comparability_evidence_sha256"
        )
        != comparable_sha
    ):
        raise RegulationModelInputError(
            f"{game_id} native-rule Gate 1 evidence does not rederive"
        )
    compared = (
        "winner_clause",
        "postponement_clause",
        "cancellation_clause",
        "tie_clause",
    )
    mismatches = [
        key
        for key in compared
        if clauses[POLYMARKET][key] != clauses[KALSHI][key]
    ]
    strict_status = str(pair.get("strict_rule_equivalence_status", ""))
    allowed_strict = (
        {NATIVE_RULE_OVERLAY_GATE_TWO_NON_EQUIVALENT}
        if mismatches
        else {
            NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT,
            NATIVE_RULE_OVERLAY_GATE_TWO_UNPROVEN,
        }
    )
    try:
        declared_mismatches = json.loads(
            str(pair.get("strict_mismatch_clauses_json", ""))
        )
    except json.JSONDecodeError as error:
        raise RegulationModelInputError(
            f"{game_id} strict mismatch clauses are malformed"
        ) from error
    strict_sha = _sha256(
        {
            "polymarket": dict(clauses[POLYMARKET]),
            "kalshi": dict(clauses[KALSHI]),
            "mismatches": mismatches,
            "status": strict_status,
        }
    )
    if (
        strict_status not in allowed_strict
        or declared_mismatches != mismatches
        or pair.get("strict_rule_equivalence_evidence_sha256")
        != strict_sha
    ):
        raise RegulationModelInputError(
            f"{game_id} native-rule Gate 2 evidence does not rederive"
        )
    return comparable_status, comparable_sha, strict_status, strict_sha


def verify_native_rule_transport_overlay(
    *,
    project_root: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
    panel: VerifiedRegulationPanelBatch,
    raw_root: Path | None = None,
    capture_batch_path: Path = NATIVE_RULE_CAPTURE_BATCH,
    expected_game_count: int = 153,
) -> VerifiedNativeRuleOverlay:
    """Verify and bind native rule evidence to exact Panel home contracts."""

    if (
        not panel.formal_input_verified
        or panel.loaded_venue_scope != (POLYMARKET, KALSHI)
        or panel.manifest.game_count != expected_game_count
        or panel.frame["game_id"].nunique() != expected_game_count
    ):
        raise RegulationModelInputError(
            "native-rule overlay requires the exact verified dual-venue "
            "Panel"
        )
    project = Path(project_root).resolve()
    path = _safe_project_path(
        project,
        Path(manifest_path),
        label="native-rule audit manifest",
    )
    try:
        verified_audit = verify_rule_audit_batch(
            project_root=project,
            manifest_path=path,
            manifest_file_sha256=manifest_file_sha256,
            expected_game_count=expected_game_count,
        )
    except RuleAuditError as error:
        raise RegulationModelInputError(str(error)) from error
    document = _verified_json_file(
        path,
        expected_sha256=manifest_file_sha256,
        label="native-rule audit manifest",
    )
    descriptors = document.get("tables")
    if not isinstance(descriptors, list):
        raise RegulationModelInputError(
            "native-rule audit table inventory is missing"
        )
    by_name = {
        str(descriptor.get("name")): descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    if set(by_name) != {
        "venue_native_rule_evidence",
        "venue_contract_pair_audit",
    }:
        raise RegulationModelInputError(
            "native-rule audit table inventory is not exact"
        )
    venue_evidence = _read_native_rule_table(
        project_root=project,
        output_root=verified_audit.output_root,
        descriptor=by_name["venue_native_rule_evidence"],
        name="venue_native_rule_evidence",
    )
    pairs = _read_native_rule_table(
        project_root=project,
        output_root=verified_audit.output_root,
        descriptor=by_name["venue_contract_pair_audit"],
        name="venue_contract_pair_audit",
    )
    if (
        len(venue_evidence) != expected_game_count * 2
        or venue_evidence.duplicated(["game_id", "venue"]).any()
        or set(venue_evidence["venue"].astype(str))
        != {POLYMARKET, KALSHI}
        or len(pairs) != expected_game_count
        or pairs["game_id"].duplicated().any()
    ):
        raise RegulationModelInputError(
            "native-rule audit cardinality is not 153x2 + 153 pairs"
        )
    panel_rows = _panel_transport_contract_rows(panel.frame)
    panel_by_key = {
        (str(row["game_id"]), str(row["venue"])): row
        for row in panel_rows
    }
    evidence_by_key = {
        (str(row["game_id"]), str(row["venue"])): row
        for row in venue_evidence.to_dict("records")
    }
    pair_by_game = {
        str(row["game_id"]): row for row in pairs.to_dict("records")
    }
    if (
        set(panel_by_key) != set(evidence_by_key)
        or {key[0] for key in panel_by_key} != set(pair_by_game)
    ):
        raise RegulationModelInputError(
            "native-rule overlay game+venue keys do not match Panel"
        )

    source_bindings = document.get("source_bindings")
    if not isinstance(source_bindings, dict):
        raise RegulationModelInputError(
            "native-rule source bindings are missing"
        )
    capture_file_sha = _require_transport_sha(
        source_bindings.get("capture_batch_file_sha256"),
        label="native-rule capture batch file SHA-256",
    )
    capture_path = _safe_project_path(
        project,
        Path(capture_batch_path),
        label="native-rule capture batch",
    )
    capture = _verified_json_file(
        capture_path,
        expected_sha256=capture_file_sha,
        label="native-rule capture batch",
    )
    capture_entries = capture.get("game_indexes")
    if (
        capture.get("schema")
        != "nfl_moneyline_expansion_batch_manifest_v1"
        or capture.get("reaction_access") != "CLOSED"
        or not isinstance(capture_entries, list)
    ):
        raise RegulationModelInputError(
            "native-rule capture batch contract mismatch"
        )
    selected_entries = [
        entry
        for entry in capture_entries
        if isinstance(entry, dict)
        and str(entry.get("game_id", "")) in pair_by_game
    ]
    if (
        len(selected_entries) != expected_game_count
        or len(
            {str(entry.get("game_id", "")) for entry in selected_entries}
        )
        != expected_game_count
    ):
        raise RegulationModelInputError(
            "native-rule capture indexes do not cover exact Panel games"
        )
    capture_root = capture_path.parents[2]
    indexes: dict[str, Mapping[str, object]] = {}
    for entry in selected_entries:
        game_id = str(entry["game_id"])
        index_path = _safe_project_path(
            project,
            capture_root / Path(str(entry.get("index_path", ""))),
            label=f"{game_id} native-rule capture index",
        )
        index = _verified_json_file(
            index_path,
            expected_sha256=_require_transport_sha(
                entry.get("index_sha256"),
                label=f"{game_id} capture index SHA-256",
            ),
            label=f"{game_id} native-rule capture index",
        )
        if (
            index.get("schema")
            != "nfl_moneyline_game_capture_index_v1"
            or index.get("game_id") != game_id
            or index.get("capture_scope") != "moneyline-only"
            or index.get("reaction_access") != "CLOSED"
        ):
            raise RegulationModelInputError(
                f"{game_id} native-rule capture index contract mismatch"
            )
        indexes[game_id] = index
    raw = (
        discover_main_checkout(project) / "var" / "raw"
        if raw_root is None
        else Path(raw_root).resolve()
    )
    if not raw.is_dir():
        raise RegulationModelInputError(
            "native-rule raw root is absent"
        )

    bindings: list[Mapping[str, object]] = []
    for key in sorted(panel_by_key):
        game_id, venue = key
        panel_row = panel_by_key[key]
        native_row = evidence_by_key[key]
        pair = pair_by_game[game_id]
        away_team, home_team = _transport_game_teams(game_id)
        if (
            str(panel_row["home_team"]) != home_team
            or str(native_row.get("home_team", "")) != home_team
            or str(native_row.get("away_team", "")) != away_team
            or str(pair.get("home_team", "")) != home_team
            or str(pair.get("away_team", "")) != away_team
            or native_row.get("completed_non_tie") is not True
            or pair.get("completed_non_tie") is not True
        ):
            raise RegulationModelInputError(
                f"{game_id} native-rule home/away or completion identity "
                "mismatch"
            )
        try:
            native_contract_ids_value = json.loads(
                str(native_row.get("contract_ids_json", ""))
            )
        except json.JSONDecodeError as error:
            raise RegulationModelInputError(
                f"{game_id} {venue} native contract IDs are malformed"
            ) from error
        if (
            not isinstance(native_contract_ids_value, list)
            or not native_contract_ids_value
            or native_contract_ids_value
            != sorted(set(native_contract_ids_value))
        ):
            raise RegulationModelInputError(
                f"{game_id} {venue} native contract IDs are invalid"
            )
        native_contract_ids = tuple(
            str(value) for value in native_contract_ids_value
        )
        native_identity_sha = _require_transport_sha(
            native_row.get("contract_identity_sha256"),
            label=f"{game_id} {venue} native contract identity",
        )
        pair_identity_column = (
            "polymarket_contract_identity_sha256"
            if venue == POLYMARKET
            else "kalshi_contract_identity_sha256"
        )
        if pair.get(pair_identity_column) != native_identity_sha:
            raise RegulationModelInputError(
                f"{game_id} {venue} native pair identity mismatch"
            )
        audit_manifests = _require_audit_hash_list(
            native_row.get("raw_manifest_sha256s_json"),
            expected_set_sha256=native_row.get(
                "raw_manifest_set_sha256"
            ),
            label=f"{game_id} {venue} audit metadata manifests",
        )
        audit_objects = _require_audit_hash_list(
            native_row.get("raw_object_sha256s_json"),
            expected_set_sha256=native_row.get("raw_object_set_sha256"),
            label=f"{game_id} {venue} audit metadata objects",
        )
        audit_requests = _require_audit_hash_list(
            native_row.get("request_sha256s_json"),
            expected_set_sha256=native_row.get("request_set_sha256"),
            label=f"{game_id} {venue} audit metadata requests",
        )
        membership_sha = _native_market_membership_evidence(
            raw_root=raw,
            game_id=game_id,
            venue=venue,
            panel_contract_id=str(
                panel_row["actual_home_contract_id"]
            ),
            native_contract_ids=native_contract_ids,
            audit_manifest_hashes=audit_manifests,
            audit_object_hashes=audit_objects,
            audit_request_hashes=audit_requests,
            capture_index=indexes[game_id],
        )
        pair_comparable_status = str(
            pair.get("completed_non_tie_comparability_status", "")
        )
        pair_strict_status = str(
            pair.get("strict_rule_equivalence_status", "")
        )
        pair_comparable_sha = _require_transport_sha(
            pair.get(
                "completed_non_tie_comparability_evidence_sha256"
            ),
            label=f"{game_id} realized comparability evidence",
        )
        pair_strict_sha = _require_transport_sha(
            pair.get("strict_rule_equivalence_evidence_sha256"),
            label=f"{game_id} strict rule audit evidence",
        )
        if (
            pair_comparable_status
            != NATIVE_RULE_OVERLAY_GATE_ONE_COMPARABLE
            or pair_strict_status
            not in {
                NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT,
                NATIVE_RULE_OVERLAY_GATE_TWO_NON_EQUIVALENT,
                NATIVE_RULE_OVERLAY_GATE_TWO_UNPROVEN,
            }
        ):
            raise RegulationModelInputError(
                f"{game_id} native-rule pair gate status is invalid"
            )
        material: dict[str, object] = {
            "game_id": game_id,
            "venue": venue,
            "panel_home_team": home_team,
            "panel_away_team": away_team,
            "panel_actual_home_contract_id": str(
                panel_row["actual_home_contract_id"]
            ),
            "panel_logical_market_id": str(
                panel_row["logical_market_id"]
            ),
            "panel_contract_identity_sha256": str(
                panel_row["actual_home_contract_identity_sha256"]
            ),
            "panel_raw_manifest_set_sha256": str(
                panel_row["native_raw_manifest_set_sha256"]
            ),
            "panel_raw_object_set_sha256": str(
                panel_row["native_raw_object_set_sha256"]
            ),
            "native_contract_ids_json": str(
                native_row["contract_ids_json"]
            ),
            "native_contract_identity_sha256": native_identity_sha,
            "native_rule_text_sha256": _require_transport_sha(
                native_row.get("rule_text_sha256"),
                label=f"{game_id} {venue} native rule text",
            ),
            "native_metadata_manifest_set_sha256": str(
                native_row["raw_manifest_set_sha256"]
            ),
            "native_metadata_object_set_sha256": str(
                native_row["raw_object_set_sha256"]
            ),
            "native_request_set_sha256": str(
                native_row["request_set_sha256"]
            ),
            "native_rule_evidence_sha256": _sha256(
                dict(native_row)
            ),
            "native_market_membership_status": (
                NATIVE_RULE_OVERLAY_MEMBERSHIP_VERIFIED
            ),
            "native_market_membership_evidence_sha256": membership_sha,
            "pair_completed_non_tie_comparability_status": (
                pair_comparable_status
            ),
            "pair_completed_non_tie_comparability_evidence_sha256": (
                pair_comparable_sha
            ),
            "pair_strict_rule_equivalence_status": pair_strict_status,
            "pair_strict_rule_equivalence_evidence_sha256": (
                pair_strict_sha
            ),
            "pair_evidence_row_sha256": _sha256(dict(pair)),
        }
        material["cross_binding_sha256"] = _sha256(material)
        bindings.append(MappingProxyType(material))
    derived_pair_gates: dict[str, tuple[str, str, str, str]] = {}
    bindings_by_game: dict[str, list[Mapping[str, object]]] = {}
    for binding in bindings:
        bindings_by_game.setdefault(str(binding["game_id"]), []).append(
            binding
        )
    for game_id in sorted(pair_by_game):
        game_bindings = bindings_by_game.get(game_id, [])
        if (
            len(game_bindings) != 2
            or {str(binding["venue"]) for binding in game_bindings}
            != {POLYMARKET, KALSHI}
            or any(
                binding.get("native_market_membership_status")
                != NATIVE_RULE_OVERLAY_MEMBERSHIP_VERIFIED
                for binding in game_bindings
            )
        ):
            raise RegulationModelInputError(
                f"{game_id} native-rule pair lacks exact verified bindings"
            )
        derived = _rederive_native_rule_pair_gate(
            game_id=game_id,
            evidence_by_venue={
                venue: evidence_by_key[(game_id, venue)]
                for venue in (POLYMARKET, KALSHI)
            },
            pair=pair_by_game[game_id],
        )
        comparable_status, comparable_sha, strict_status, strict_sha = (
            derived
        )
        if any(
            (
                binding[
                    "pair_completed_non_tie_comparability_status"
                ]
                != comparable_status
                or binding[
                    "pair_completed_non_tie_comparability_evidence_sha256"
                ]
                != comparable_sha
                or binding["pair_strict_rule_equivalence_status"]
                != strict_status
                or binding[
                    "pair_strict_rule_equivalence_evidence_sha256"
                ]
                != strict_sha
            )
            for binding in game_bindings
        ):
            raise RegulationModelInputError(
                f"{game_id} native-rule dual-venue gate binding mismatch"
            )
        derived_pair_gates[game_id] = derived
    binding_hashes = [
        str(binding["cross_binding_sha256"]) for binding in bindings
    ]
    comparable_games = {
        game_id
        for game_id, gate in derived_pair_gates.items()
        if gate[0] == NATIVE_RULE_OVERLAY_GATE_ONE_COMPARABLE
    }
    equivalent_games = {
        game_id
        for game_id, gate in derived_pair_gates.items()
        if gate[2] == NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT
    }
    if (
        len(comparable_games)
        != verified_audit.completed_non_tie_comparable_count
        or len(equivalent_games)
        != verified_audit.strict_rule_equivalent_count
    ):
        raise RegulationModelInputError(
            "native-rule overlay gate counts do not reconcile"
        )
    return VerifiedNativeRuleOverlay(
        manifest_path=path,
        manifest_file_sha256=manifest_file_sha256,
        batch_sha256=verified_audit.batch_sha256,
        source_capture_batch_file_sha256=capture_file_sha,
        panel_batch_manifest_file_sha256=(
            panel.manifest.batch_manifest_file_sha256
        ),
        panel_batch_sha256=panel.manifest.batch_sha256,
        panel_cohort_authority_sha256=(
            panel.manifest.cohort_authority_sha256
        ),
        panel_contract_authority_sha256=(
            _panel_transport_contract_authority_sha256(panel.frame)
        ),
        game_count=expected_game_count,
        venue_evidence_row_count=len(venue_evidence),
        contract_pair_row_count=len(pairs),
        completed_non_tie_comparable_count=len(comparable_games),
        strict_rule_equivalent_count=len(equivalent_games),
        cross_binding_set_sha256=_sha256(binding_hashes),
        bindings=tuple(bindings),
    )


def _transport_native_rule_overlay_summary(
    overlay: VerifiedNativeRuleOverlay,
    *,
    panel: VerifiedRegulationPanelBatch,
) -> Mapping[str, object]:
    if (
        not overlay.formal_overlay_verified
        or overlay.panel_batch_manifest_file_sha256
        != panel.manifest.batch_manifest_file_sha256
        or overlay.panel_batch_sha256 != panel.manifest.batch_sha256
        or overlay.panel_cohort_authority_sha256
        != panel.manifest.cohort_authority_sha256
        or overlay.panel_contract_authority_sha256
        != _panel_transport_contract_authority_sha256(panel.frame)
        or overlay.game_count != 153
        or overlay.venue_evidence_row_count != 306
        or overlay.contract_pair_row_count != 153
        or len(overlay.bindings) != 306
    ):
        raise RegulationModelInputError(
            "native-rule overlay does not bind this exact Panel contract "
            "authority"
        )
    keys: set[tuple[str, str]] = set()
    binding_hashes: list[str] = []
    comparable_evidence: set[str] = set()
    strict_evidence: set[str] = set()
    native_evidence_by_venue: dict[str, set[str]] = {}
    pair_gates: dict[str, tuple[str, str, str, str]] = {}
    for binding in overlay.bindings:
        material = dict(binding)
        cross_sha = _require_transport_sha(
            material.pop("cross_binding_sha256", None),
            label="native-rule cross binding SHA-256",
        )
        if _sha256(material) != cross_sha:
            raise RegulationModelInputError(
                "native-rule overlay cross binding was modified"
            )
        game_id = str(binding.get("game_id", ""))
        venue = str(binding.get("venue", ""))
        key = (game_id, venue)
        if (
            key in keys
            or venue not in {POLYMARKET, KALSHI}
            or binding.get("native_market_membership_status")
            != NATIVE_RULE_OVERLAY_MEMBERSHIP_VERIFIED
        ):
            raise RegulationModelInputError(
                "native-rule overlay key or market membership is invalid"
            )
        _require_transport_sha(
            binding.get("native_market_membership_evidence_sha256"),
            label="native market membership evidence",
        )
        native_sha = _require_transport_sha(
            binding.get("native_rule_evidence_sha256"),
            label="native rule evidence",
        )
        comparable_sha = _require_transport_sha(
            binding.get(
                "pair_completed_non_tie_comparability_evidence_sha256"
            ),
            label="realized comparability evidence",
        )
        strict_sha = _require_transport_sha(
            binding.get(
                "pair_strict_rule_equivalence_evidence_sha256"
            ),
            label="strict rule audit evidence",
        )
        comparable_status = str(
            binding.get(
                "pair_completed_non_tie_comparability_status", ""
            )
        )
        strict_status = str(
            binding.get("pair_strict_rule_equivalence_status", "")
        )
        if (
            comparable_status
            != NATIVE_RULE_OVERLAY_GATE_ONE_COMPARABLE
            or strict_status
            not in {
                NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT,
                NATIVE_RULE_OVERLAY_GATE_TWO_NON_EQUIVALENT,
                NATIVE_RULE_OVERLAY_GATE_TWO_UNPROVEN,
            }
        ):
            raise RegulationModelInputError(
                "native-rule pair gate status is invalid"
            )
        gate = (
            comparable_status,
            comparable_sha,
            strict_status,
            strict_sha,
        )
        prior_gate = pair_gates.setdefault(game_id, gate)
        if prior_gate != gate:
            raise RegulationModelInputError(
                "native-rule dual-venue pair gate fields disagree"
            )
        strict_evidence.add(strict_sha)
        native_evidence_by_venue.setdefault(venue, set()).add(native_sha)
        keys.add(key)
        binding_hashes.append(cross_sha)
    comparable_games = {
        game_id
        for game_id, gate in pair_gates.items()
        if gate[0] == NATIVE_RULE_OVERLAY_GATE_ONE_COMPARABLE
    }
    equivalent_games = {
        game_id
        for game_id, gate in pair_gates.items()
        if gate[2] == NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT
    }
    comparable_evidence.update(
        gate[1] for gate in pair_gates.values()
    )
    strict_evidence.update(gate[3] for gate in pair_gates.values())
    if (
        _sha256(binding_hashes) != overlay.cross_binding_set_sha256
        or len(keys) != 306
        or len({game_id for game_id, _ in keys}) != 153
        or len(pair_gates) != 153
        or len(comparable_games)
        != overlay.completed_non_tie_comparable_count
        or len(equivalent_games) != overlay.strict_rule_equivalent_count
        or overlay.completed_non_tie_comparable_count != 153
    ):
        raise RegulationModelInputError(
            "native-rule overlay integrity or Gate 1 coverage failed"
        )
    strict_equivalent = overlay.strict_rule_equivalent_count == 153
    return MappingProxyType(
        {
            "realized_outcome_comparability_status": (
                REALIZED_OUTCOME_COMPARABLE
            ),
            "realized_outcome_comparability_evidence_set_sha256": (
                _sha256(sorted(comparable_evidence))
            ),
            "strict_contract_rule_equivalence_status": (
                STRICT_CONTRACT_RULE_EQUIVALENT
                if strict_equivalent
                else STRICT_CONTRACT_RULE_UNPROVEN
            ),
            "strict_contract_rule_equivalence_evidence_set_sha256": (
                _sha256(sorted(strict_evidence))
            ),
            "native_rule_evidence_status_by_venue": MappingProxyType(
                {
                    venue: ("VERIFIED_NATIVE_RULE",)
                    for venue in sorted(native_evidence_by_venue)
                }
            ),
            "native_rule_evidence_set_sha256_by_venue": MappingProxyType(
                {
                    venue: _sha256(sorted(values))
                    for venue, values in sorted(
                        native_evidence_by_venue.items()
                    )
                }
            ),
        }
    )


def _attach_native_rule_overlay(
    frame: pd.DataFrame,
    overlay: VerifiedNativeRuleOverlay,
) -> pd.DataFrame:
    authority = pd.DataFrame(
        [
            {
                "game_id": str(binding["game_id"]),
                "venue": str(binding["venue"]),
                "native_rule_evidence_sha256": str(
                    binding["native_rule_evidence_sha256"]
                ),
                "native_rule_evidence_status": "VERIFIED_NATIVE_RULE",
                "realized_outcome_comparability_status": (
                    REALIZED_OUTCOME_COMPARABLE
                ),
                "realized_outcome_comparability_evidence_sha256": str(
                    binding[
                        "pair_completed_non_tie_comparability_"
                        "evidence_sha256"
                    ]
                ),
                "strict_contract_rule_equivalence_status": (
                    STRICT_CONTRACT_RULE_EQUIVALENT
                    if binding["pair_strict_rule_equivalence_status"]
                    == NATIVE_RULE_OVERLAY_GATE_TWO_EQUIVALENT
                    else STRICT_CONTRACT_RULE_UNPROVEN
                ),
                "strict_contract_rule_equivalence_evidence_sha256": str(
                    binding[
                        "pair_strict_rule_equivalence_evidence_sha256"
                    ]
                ),
                "native_rule_overlay_cross_binding_sha256": str(
                    binding["cross_binding_sha256"]
                ),
            }
            for binding in overlay.bindings
        ]
    )
    overlay_columns = [
        "native_rule_evidence_sha256",
        "native_rule_evidence_status",
        "realized_outcome_comparability_status",
        "realized_outcome_comparability_evidence_sha256",
        "strict_contract_rule_equivalence_status",
        "strict_contract_rule_equivalence_evidence_sha256",
    ]
    result = frame.drop(columns=overlay_columns).merge(
        authority,
        on=["game_id", "venue"],
        how="left",
        validate="many_to_one",
    )
    if result["native_rule_overlay_cross_binding_sha256"].isna().any():
        raise RegulationModelInputError(
            "native-rule overlay does not cover every transport row"
        )
    return result


def _transport_market_gate_summary(
    frame: pd.DataFrame,
    *,
    require_realized_comparability: bool,
) -> Mapping[str, object]:
    """Verify native market identity and summarize the two transport gates."""

    required = {
        "venue",
        "actual_home_contract_id",
        "logical_market_id",
        "market_family",
        "market_period",
        "proposition_kind",
        "home_team",
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
    }
    missing = required - set(frame.columns)
    if missing:
        raise RegulationModelInputError(
            "transport market identity is incomplete: "
            f"{sorted(missing)}"
        )
    if frame.empty:
        raise RegulationModelInputError(
            "transport market identity has no rows"
        )
    if (
        frame["market_family"].astype(str).ne("moneyline").any()
        or frame["market_period"].astype(str).ne("full_game").any()
        or frame["proposition_kind"].astype(str).ne("primitive").any()
        or frame["home_team"].isna().any()
        or frame["outcome_team"].isna().any()
        or frame["home_team"].astype(str).str.strip().eq("").any()
        or frame["outcome_team"].astype(str).str.strip().eq("").any()
        or not frame["home_team"].astype(str).eq(
            frame["outcome_team"].astype(str)
        ).all()
    ):
        raise RegulationModelInputError(
            "transport accepts only native primitive full-game moneyline "
            "contracts oriented to the home-team outcome"
        )
    for column in (
        "actual_home_contract_id",
        "logical_market_id",
    ):
        if frame[column].isna().any() or frame[
            column
        ].astype(str).str.strip().eq("").any():
            raise RegulationModelInputError(
                f"transport {column} must be nonempty"
            )

    contract_columns = [
        "venue",
        "actual_home_contract_id",
        "logical_market_id",
        "market_family",
        "market_period",
        "proposition_kind",
        "home_team",
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
    ]
    contracts = frame.loc[:, contract_columns].drop_duplicates()
    realized_evidence: list[str] = []
    strict_evidence: list[str] = []
    native_status_by_venue: dict[str, set[str]] = {}
    native_evidence_by_venue: dict[str, list[str]] = {}
    strict_statuses: set[str] = set()
    for contract in contracts.to_dict("records"):
        venue = str(contract["venue"])
        _require_transport_sha(
            contract["actual_home_contract_identity_sha256"],
            label="actual_home_contract_identity_sha256",
        )
        raw_sets: list[tuple[str, str, str]] = [
            (
                "manifest",
                str(contract["native_raw_manifest_sha256s_json"]),
                str(contract["native_raw_manifest_set_sha256"]),
            ),
            (
                "object",
                str(contract["native_raw_object_sha256s_json"]),
                str(contract["native_raw_object_set_sha256"]),
            ),
        ]
        for kind, encoded, expected_set_sha in raw_sets:
            try:
                values = json.loads(encoded)
            except json.JSONDecodeError as error:
                raise RegulationModelInputError(
                    f"native raw {kind} lineage JSON is invalid"
                ) from error
            if (
                not isinstance(values, list)
                or not values
                or values != sorted(set(values))
            ):
                raise RegulationModelInputError(
                    f"native raw {kind} lineage must be sorted and unique"
                )
            canonical_values = [
                _require_transport_sha(
                    value, label=f"native raw {kind} lineage"
                )
                for value in values
            ]
            if (
                _panel_lineage_set_sha256(canonical_values)
                != _require_transport_sha(
                    expected_set_sha,
                    label=f"native raw {kind} set SHA-256",
                )
            ):
                raise RegulationModelInputError(
                    f"native raw {kind} lineage set/hash mismatch"
                )

        native_status = str(contract["native_rule_evidence_status"])
        native_sha = _optional_transport_sha(
            contract["native_rule_evidence_sha256"],
            label="native_rule_evidence_sha256",
        )
        if (
            native_status == "VERIFIED_NATIVE_RULE"
        ) != (native_sha is not None):
            raise RegulationModelInputError(
                "native rule evidence status/hash mismatch"
            )
        if native_status not in {
            "VERIFIED_NATIVE_RULE",
            "RAW_METADATA_CAPTURED_RULE_SEMANTICS_NOT_ADJUDICATED",
        }:
            raise RegulationModelInputError(
                "native rule evidence status is invalid"
            )
        native_status_by_venue.setdefault(venue, set()).add(native_status)
        if native_sha is not None:
            native_evidence_by_venue.setdefault(venue, []).append(native_sha)

        realized_status = str(
            contract["realized_outcome_comparability_status"]
        )
        realized_sha = _optional_transport_sha(
            contract["realized_outcome_comparability_evidence_sha256"],
            label="realized_outcome_comparability_evidence_sha256",
        )
        if (
            realized_status not in REALIZED_OUTCOME_COMPARABILITY_STATUSES
            or (
                realized_status == REALIZED_OUTCOME_COMPARABLE
            ) != (realized_sha is not None)
        ):
            raise RegulationModelInputError(
                "realized outcome comparability status/hash mismatch"
            )
        if require_realized_comparability and (
            realized_status != REALIZED_OUTCOME_COMPARABLE
            or realized_sha is None
        ):
            raise RegulationModelInputError(
                "realized completed-game outcome comparability is "
                "unproven; descriptive Kalshi transport is blocked"
            )
        if realized_sha is not None:
            realized_evidence.append(realized_sha)

        strict_status = str(
            contract["strict_contract_rule_equivalence_status"]
        )
        strict_sha = _optional_transport_sha(
            contract["strict_contract_rule_equivalence_evidence_sha256"],
            label="strict_contract_rule_equivalence_evidence_sha256",
        )
        if (
            strict_status not in STRICT_CONTRACT_RULE_EQUIVALENCE_STATUSES
            or (
                strict_status == STRICT_CONTRACT_RULE_EQUIVALENT
            ) != (strict_sha is not None)
        ):
            raise RegulationModelInputError(
                "strict contract rule equivalence status/hash mismatch"
            )
        strict_statuses.add(strict_status)
        if strict_sha is not None:
            strict_evidence.append(strict_sha)

    if require_realized_comparability and not realized_evidence:
        raise RegulationModelInputError(
            "descriptive Kalshi transport lacks realized comparability "
            "evidence"
        )
    strict_global = (
        STRICT_CONTRACT_RULE_EQUIVALENT
        if strict_statuses == {STRICT_CONTRACT_RULE_EQUIVALENT}
        else STRICT_CONTRACT_RULE_UNPROVEN
    )
    strict_set_sha = (
        _sha256(sorted(set(strict_evidence)))
        if strict_global == STRICT_CONTRACT_RULE_EQUIVALENT
        else None
    )
    return MappingProxyType(
        {
            "realized_outcome_comparability_status": (
                REALIZED_OUTCOME_COMPARABLE
                if require_realized_comparability
                else (
                    REALIZED_OUTCOME_COMPARABLE
                    if realized_evidence
                    else REALIZED_OUTCOME_UNPROVEN
                )
            ),
            "realized_outcome_comparability_evidence_set_sha256": (
                _sha256(sorted(set(realized_evidence)))
                if realized_evidence
                else None
            ),
            "strict_contract_rule_equivalence_status": strict_global,
            "strict_contract_rule_equivalence_evidence_set_sha256": (
                strict_set_sha
            ),
            "native_rule_evidence_status_by_venue": MappingProxyType(
                {
                    venue: tuple(sorted(statuses))
                    for venue, statuses in sorted(
                        native_status_by_venue.items()
                    )
                }
            ),
            "native_rule_evidence_set_sha256_by_venue": MappingProxyType(
                {
                    venue: (
                        _sha256(
                            sorted(
                                set(
                                    native_evidence_by_venue.get(
                                        venue, ()
                                    )
                                )
                            )
                        )
                        if native_evidence_by_venue.get(venue)
                        else None
                    )
                    for venue in sorted(native_status_by_venue)
                }
            ),
        }
    )


def validate_kalshi_transport(
    panel: VerifiedRegulationPanelBatch,
    *,
    selection_lock: VerifiedRegulationSelectionLock | None,
    native_rule_overlay: VerifiedNativeRuleOverlay | None,
) -> KalshiTransportRequestV2:
    """Bind exact PanelV4 Kalshi rows to one verified Polymarket lock."""

    if (
        selection_lock is None
        or not selection_lock.formal_lock_verified
    ):
        raise RegulationModelInputError(
            "Kalshi transport requires a verified selection lock"
        )
    if (
        not panel.formal_input_verified
        or panel.manifest.game_count != 153
        or panel.frame["game_id"].nunique() != 153
        or selection_lock.panel_batch_manifest_file_sha256
        != panel.manifest.batch_manifest_file_sha256
        or selection_lock.panel_batch_sha256
        != panel.manifest.batch_sha256
        or selection_lock.cohort_authority_sha256
        != panel.manifest.cohort_authority_sha256
        or selection_lock.cohort_mapping_sha256
        != panel.manifest.cohort_mapping_sha256
    ):
        raise RegulationModelInputError(
            "selection lock does not bind this verified exact-153 PanelV4"
        )
    winner = selection_lock.winner
    if (
        selection_lock.decision_status != "WINNER_FROZEN"
        or not selection_lock.kalshi_transport_allowed
        or winner is None
    ):
        raise RegulationModelInputError(
            "NO_MODEL_ADVANCE selection lock forbids Kalshi transport"
        )
    if (
        panel.loaded_venue_scope != (POLYMARKET, KALSHI)
        or panel.loaded_row_count != len(panel.frame)
        or set(panel.frame["venue"].astype(str))
        != {POLYMARKET, KALSHI}
    ):
        raise RegulationModelInputError(
            "Kalshi transport requires verified POLYMARKET+KALSHI "
            "dual-venue scope"
        )
    if native_rule_overlay is None:
        raise RegulationModelInputError(
            "Kalshi transport requires a verified native-rule overlay"
        )
    if (
        (winner.model_id, winner.feature_block_id)
        not in SELECTABLE_CANDIDATES
        or winner.anchor_kind != PRIMARY_ANCHOR_KIND
        or winner.polymarket_selection_sha256
        != selection_lock.selection_sha256
        or set(winner.polymarket_fold_training_hashes) != set(_FOLD_IDS)
        or set(selection_lock.baseline_fold_training_hashes)
        != set(_FOLD_IDS)
        or selection_lock.landmark_seconds != LANDMARK_SECONDS
        or selection_lock.endpoint_seconds != ENDPOINT_SECONDS
        or selection_lock.direction_threshold_probability != 0.01
        or selection_lock.target_version != TRANSPORT_TARGET_VERSION
    ):
        raise RegulationModelInputError(
            "verified selection lock frozen identity is invalid"
        )
    prepared = panel.frame
    primary = prepared.loc[
        prepared["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & prepared["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ]
    if (
        primary.empty
        or set(primary["venue"].astype(str)) != {POLYMARKET, KALSHI}
    ):
        raise RegulationModelInputError(
            "verified PanelV4 lacks both bound Poly and Kalshi primary "
            "risk rows"
        )
    market_gates = _transport_native_rule_overlay_summary(
        native_rule_overlay,
        panel=panel,
    )
    formal_contract_transport_allowed = (
        market_gates["strict_contract_rule_equivalence_status"]
        == STRICT_CONTRACT_RULE_EQUIVALENT
    )
    return KalshiTransportRequestV2(
        training_venue=POLYMARKET,
        evaluation_venue=KALSHI,
        model_id=winner.model_id,
        feature_block_id=winner.feature_block_id,
        polymarket_selection_sha256=selection_lock.selection_sha256,
        selection_lock_sha256=selection_lock.selection_lock_sha256,
        anchor_kind=PRIMARY_ANCHOR_KIND,
        target_version=TRANSPORT_TARGET_VERSION,
        transport_scope=(
            FORMAL_CONTRACT_TRANSPORT_SCOPE
            if formal_contract_transport_allowed
            else DESCRIPTIVE_TRANSPORT_SCOPE
        ),
        realized_outcome_comparability_status=str(
            market_gates["realized_outcome_comparability_status"]
        ),
        realized_outcome_comparability_evidence_set_sha256=str(
            market_gates[
                "realized_outcome_comparability_evidence_set_sha256"
            ]
        ),
        strict_contract_rule_equivalence_status=str(
            market_gates["strict_contract_rule_equivalence_status"]
        ),
        strict_contract_rule_equivalence_evidence_set_sha256=(
            None
            if market_gates[
                "strict_contract_rule_equivalence_evidence_set_sha256"
            ]
            is None
            else str(
                market_gates[
                    "strict_contract_rule_equivalence_evidence_set_sha256"
                ]
            )
        ),
        native_rule_evidence_status_by_venue=market_gates[
            "native_rule_evidence_status_by_venue"
        ],
        native_rule_evidence_set_sha256_by_venue=market_gates[
            "native_rule_evidence_set_sha256_by_venue"
        ],
        formal_contract_transport_allowed=(
            formal_contract_transport_allowed
        ),
        independent_sports_holdout_validation_allowed=False,
        venue_speed_claim_allowed=False,
        selection_allowed=False,
    )


def _with_transport_pair_identity(
    frame: pd.DataFrame,
    *,
    transport_scope: str,
) -> pd.DataFrame:
    result = frame.copy()
    if transport_scope not in {
        DESCRIPTIVE_TRANSPORT_SCOPE,
        FORMAL_CONTRACT_TRANSPORT_SCOPE,
    }:
        raise RegulationModelInputError("transport scope is invalid")
    if (
        result["market_family"].astype(str).ne("moneyline").any()
        or result["market_period"].astype(str).ne("full_game").any()
        or result["proposition_kind"].astype(str).ne("primitive").any()
        or not result["outcome_team"].astype(str).eq(
            result["home_team"].astype(str)
        ).all()
    ):
        raise RegulationModelInputError(
            "descriptive pair identity requires native full-game "
            "home-team moneyline rows"
        )
    result["transport_market_scope"] = TRANSPORT_MARKET_SCOPE
    result["transport_outcome_scope"] = TRANSPORT_OUTCOME_SCOPE
    result["transport_scope"] = transport_scope
    result["transport_target_version"] = TRANSPORT_TARGET_VERSION
    result["descriptive_transport_pair_id"] = [
        _sha256(
            {
                "game_id": str(game_id),
                "information_anchor_id": str(information_anchor_id),
                "transport_market_scope": TRANSPORT_MARKET_SCOPE,
                "transport_outcome_scope": TRANSPORT_OUTCOME_SCOPE,
                "landmark_seconds": int(landmark_seconds),
                "endpoint_seconds": int(endpoint_seconds),
                "target_version": TRANSPORT_TARGET_VERSION,
                "transport_scope": transport_scope,
            }
        )
        for (
            game_id,
            information_anchor_id,
            landmark_seconds,
            endpoint_seconds,
        ) in result[
            [
                "game_id",
                "information_anchor_id",
                "landmark_seconds",
                "endpoint_seconds",
            ]
        ].itertuples(index=False, name=None)
    ]
    return result


def _fit_transport_producer(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fold: X15WeekFold,
    model_id: str,
    feature_block_id: str,
    frozen_hashes: Mapping[str, str],
    input_lineage: Mapping[str, object],
    transport_scope: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fold_id = fold.fold_id
    fold_sha256 = _sha256(
        {
            "fold_id": fold.fold_id,
            "train_weeks": fold.train_weeks,
            "validation_weeks": fold.validation_weeks,
            "train_game_ids": fold.train_game_ids,
            "validation_game_ids": fold.validation_game_ids,
        }
    )
    seed = effective_random_state(
        base_random_state=random_state,
        fold_id=fold_id,
        training_venue=POLYMARKET,
        evaluation_venue=POLYMARKET,
        transport_mode="NATIVE_POLYMARKET_SELECTION",
        feature_block_id=feature_block_id,
        model_id=model_id,
        purpose="REGULATION_TWO_HEAD_FIT",
    )
    if model_id == "b0_empirical_v1":
        producer_train = train
        calibration = train.iloc[0:0].copy()
        fitted = _fit_two_heads(
            producer_train,
            model_id=model_id,
            feature_block_id=feature_block_id,
            random_state=seed,
        )
        availability_calibrator = None
        direction_calibrator = None
        availability_status = "NOT_APPLICABLE_EMPIRICAL_BASELINE"
        direction_status = "NOT_APPLICABLE_EMPIRICAL_BASELINE"
        availability_raw_sha = _sha256(
            {"scheme": "NONE_EMPIRICAL_BASELINE", "head": "availability"}
        )
        direction_raw_sha = _sha256(
            {"scheme": "NONE_EMPIRICAL_BASELINE", "head": "direction"}
        )
    else:
        (
            fitted,
            availability_calibrator,
            direction_calibrator,
            producer_train,
            calibration,
        ) = _raw_calibrators(
            outer_train=train,
            model_id=model_id,
            feature_block_id=feature_block_id,
            random_state=seed,
        )
        if (
            availability_calibrator.status != CALIBRATED
            or direction_calibrator.status != CALIBRATED
        ):
            raise RegulationModelInputError(
                f"Kalshi transport {fold_id} calibration unsupported"
            )
        availability_status = availability_calibrator.status
        direction_status = direction_calibrator.status
        availability_raw_sha = availability_calibrator.training_sha256
        direction_raw_sha = direction_calibrator.training_sha256
    calibration_game_ids = tuple(
        sorted(calibration["game_id"].astype(str).unique())
    )
    availability_calibration_sha = _bound_calibration_sha256(
        head="availability",
        calibration_status=availability_status,
        raw_calibration_training_sha256=availability_raw_sha,
        fitted=fitted,
        calibration_game_ids=calibration_game_ids,
    )
    direction_calibration_sha = _bound_calibration_sha256(
        head="direction",
        calibration_status=direction_status,
        raw_calibration_training_sha256=direction_raw_sha,
        fitted=fitted,
        calibration_game_ids=calibration_game_ids,
    )
    producer_provenance = _producer_provenance_hashes(
        fitted=fitted,
        model_id=model_id,
        producer_train=producer_train,
        calibration=calibration,
        availability_calibrator=availability_calibrator,
        direction_calibrator=direction_calibrator,
        availability_calibration_status=availability_status,
        direction_calibration_status=direction_status,
        availability_raw_calibration_sha256=availability_raw_sha,
        direction_raw_calibration_sha256=direction_raw_sha,
        availability_calibration_training_sha256=(
            availability_calibration_sha
        ),
        direction_calibration_training_sha256=(
            direction_calibration_sha
        ),
    )
    observed_hashes = {
        "fold_sha256": fold_sha256,
        **producer_provenance,
    }
    if {
        key: str(value) for key, value in frozen_hashes.items()
    } != observed_hashes:
        raise RegulationModelInputError(
            f"Kalshi transport {fold_id}/{model_id}/{feature_block_id} "
            "producer hashes differ from verified selection lock"
        )
    availability_raw, direction_raw = _predict_two_heads(
        fitted,
        validation,
        model_id=model_id,
        feature_block_id=feature_block_id,
    )
    availability_probability = (
        availability_raw
        if availability_calibrator is None
        else availability_calibrator.transform(availability_raw)
    )
    direction_probability = (
        direction_raw
        if direction_calibrator is None
        else direction_calibrator.transform(direction_raw)
    )
    predictions = _prediction_rows(
        validation,
        fold_id=fold_id,
        model_id=model_id,
        feature_block_id=feature_block_id,
        availability_probability=availability_probability,
        direction_probability=direction_probability,
        availability_calibration_status=availability_status,
        direction_calibration_status=direction_status,
        availability_calibration_training_sha256=(
            availability_calibration_sha
        ),
        direction_calibration_training_sha256=direction_calibration_sha,
        fitted=fitted,
        producer_provenance=producer_provenance,
        fold_sha256=fold_sha256,
        input_lineage=input_lineage,
        training_venue=POLYMARKET,
        evaluation_venue=KALSHI,
        transport_mode=(
            "FROZEN_POLYMARKET_B0_TRANSPORT"
            if model_id == "b0_empirical_v1"
            else "FROZEN_POLYMARKET_WINNER_TRANSPORT"
        ),
    )
    predictions = _with_transport_pair_identity(
        predictions,
        transport_scope=transport_scope,
    )
    support: dict[str, object] = {
        "fold_id": fold_id,
        "model_id": model_id,
        "feature_block_id": feature_block_id,
        "training_venue": POLYMARKET,
        "evaluation_venue": KALSHI,
        "selection_allowed": False,
        "validation_row_count": len(validation),
        "validation_game_count": int(validation["game_id"].nunique()),
        "validation_availability_positive_count": int(
            validation["availability_h"].sum()
        ),
        "validation_direction_classes": json.dumps(
            sorted(
                validation.loc[
                    validation["availability_h"], "direction_h"
                ].astype(str).unique()
            ),
            separators=(",", ":"),
        ),
        **observed_hashes,
    }
    return predictions, support


def run_frozen_kalshi_transport(
    panel: VerifiedRegulationPanelBatch,
    *,
    selection_lock: VerifiedRegulationSelectionLock,
    native_rule_overlay: VerifiedNativeRuleOverlay | None,
    random_state: int = RANDOM_STATE,
) -> RegulationKalshiTransportRun:
    """Transport frozen winner and B0 to exact paired Kalshi risk rows."""

    request = validate_kalshi_transport(
        panel,
        selection_lock=selection_lock,
        native_rule_overlay=native_rule_overlay,
    )
    assert native_rule_overlay is not None
    prepared = _attach_native_rule_overlay(
        panel.frame, native_rule_overlay
    )
    input_lineage: Mapping[str, object] = {
        "panel_batch_manifest_file_sha256": (
            panel.manifest.batch_manifest_file_sha256
        ),
        "panel_batch_sha256": panel.manifest.batch_sha256,
        "panel_cohort_authority_sha256": (
            panel.manifest.cohort_authority_sha256
        ),
        "panel_cohort_mapping_sha256": (
            panel.manifest.cohort_mapping_sha256
        ),
        "panel_source_hashes": dict(panel.manifest.sources),
        "selection_lock_manifest_file_sha256": (
            selection_lock.manifest_file_sha256
        ),
        "selection_lock_sha256": (
            selection_lock.selection_lock_sha256
        ),
        "native_rule_overlay_manifest_file_sha256": (
            native_rule_overlay.manifest_file_sha256
        ),
        "native_rule_overlay_batch_sha256": (
            native_rule_overlay.batch_sha256
        ),
        "native_rule_overlay_cross_binding_set_sha256": (
            native_rule_overlay.cross_binding_set_sha256
        ),
        "transport_scope": request.transport_scope,
        "realized_outcome_comparability_evidence_set_sha256": (
            request.realized_outcome_comparability_evidence_set_sha256
        ),
        "strict_contract_rule_equivalence_status": (
            request.strict_contract_rule_equivalence_status
        ),
        "strict_contract_rule_equivalence_evidence_set_sha256": (
            request.strict_contract_rule_equivalence_evidence_set_sha256
        ),
    }
    polymarket = prepared.loc[
        prepared["venue"].eq(POLYMARKET)
        & prepared["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & prepared["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ].reset_index(drop=True)
    kalshi = prepared.loc[
        prepared["venue"].eq(KALSHI)
        & prepared["anchor_kind"].eq(PRIMARY_ANCHOR_KIND)
        & prepared["analysis_role"].eq(PRIMARY_ANALYSIS_ROLE)
    ].reset_index(drop=True)
    if polymarket.empty or kalshi.empty:
        raise RegulationModelInputError(
            "Kalshi transport needs both Poly training and Kalshi evaluation"
        )
    folds = _fold_map(polymarket)
    predictions: list[pd.DataFrame] = []
    supports: list[dict[str, object]] = []
    risk_sets: list[pd.DataFrame] = []
    poly_common_parts: list[pd.DataFrame] = []
    kalshi_common_parts: list[pd.DataFrame] = []
    for fold_id in _FOLD_IDS:
        fold = folds[fold_id]
        train = polymarket.loc[
            polymarket["game_id"].isin(fold.train_game_ids)
        ].reset_index(drop=True)
        validation = kalshi.loc[
            kalshi["game_id"].isin(fold.validation_game_ids)
        ].reset_index(drop=True)
        if train.empty or validation.empty:
            raise RegulationModelInputError(
                f"Kalshi transport {fold_id} lacks train/evaluation rows"
            )
        candidate_predictions, candidate_support = (
            _fit_transport_producer(
                train=train,
                validation=validation,
                fold=fold,
                model_id=request.model_id,
                feature_block_id=request.feature_block_id,
                frozen_hashes=(
                    selection_lock.winner.polymarket_fold_training_hashes[
                        fold_id
                    ]
                ),
                input_lineage=input_lineage,
                transport_scope=request.transport_scope,
                random_state=random_state,
            )
        )
        baseline_predictions, baseline_support = (
            _fit_transport_producer(
                train=train,
                validation=validation,
                fold=fold,
                model_id="b0_empirical_v1",
                feature_block_id="D0",
                frozen_hashes=(
                    selection_lock.baseline_fold_training_hashes[
                        fold_id
                    ]
                ),
                input_lineage=input_lineage,
                transport_scope=request.transport_scope,
                random_state=random_state,
            )
        )
        if (
            set(candidate_predictions["descriptive_transport_pair_id"])
            != set(baseline_predictions["descriptive_transport_pair_id"])
            or len(candidate_predictions) != len(baseline_predictions)
        ):
            raise RegulationModelInputError(
                f"Kalshi transport {fold_id} winner and B0 risk sets differ"
            )
        predictions.extend(
            (baseline_predictions, candidate_predictions)
        )
        supports.extend((baseline_support, candidate_support))
        risk = _with_transport_pair_identity(
            validation,
            transport_scope=request.transport_scope,
        )
        if risk["descriptive_transport_pair_id"].duplicated().any():
            raise RegulationModelInputError(
                f"Kalshi transport {fold_id} has duplicate pair identity"
            )
        risk_sets.append(
            risk.loc[
                :,
                [
                    "descriptive_transport_pair_id",
                    "game_id",
                    "information_anchor_id",
                    "episode_id",
                    "landmark_seconds",
                    "endpoint_seconds",
                    "availability_h",
                    "direction_h",
                    "direction_eligible",
                    "transport_market_scope",
                    "transport_outcome_scope",
                    "transport_scope",
                    "transport_target_version",
                    "actual_home_contract_id",
                    "logical_market_id",
                    "actual_home_contract_identity_sha256",
                    "native_raw_manifest_set_sha256",
                    "native_raw_object_set_sha256",
                    "native_rule_evidence_sha256",
                    "native_rule_evidence_status",
                    "realized_outcome_comparability_status",
                    "realized_outcome_comparability_evidence_sha256",
                    "strict_contract_rule_equivalence_status",
                    "strict_contract_rule_equivalence_evidence_sha256",
                    "source_row_id",
                ],
            ]
            .assign(
                fold_id=fold_id,
                venue=KALSHI,
                bound_risk_row=True,
            )
            .rename(
                columns={
                    "availability_h": "availability_truth",
                    "direction_h": "direction_truth",
                }
            )
        )
        for source, destination in (
            (
                polymarket.loc[
                    polymarket["game_id"].isin(
                        fold.validation_game_ids
                    )
                ].reset_index(drop=True),
                poly_common_parts,
            ),
            (validation, kalshi_common_parts),
        ):
            common = _with_transport_pair_identity(
                source,
                transport_scope=request.transport_scope,
            )
            if common["descriptive_transport_pair_id"].duplicated().any():
                raise RegulationModelInputError(
                    f"{fold_id} venue has duplicate transport pair identity"
                )
            destination.append(
                common.loc[
                    :,
                    [
                        "descriptive_transport_pair_id",
                        "source_row_id",
                        "availability_h",
                        "direction_h",
                        "actual_home_contract_id",
                        "logical_market_id",
                        "actual_home_contract_identity_sha256",
                        "native_rule_evidence_sha256",
                        "native_rule_evidence_status",
                        "realized_outcome_comparability_status",
                        (
                            "realized_outcome_comparability_"
                            "evidence_sha256"
                        ),
                        "strict_contract_rule_equivalence_status",
                        (
                            "strict_contract_rule_equivalence_"
                            "evidence_sha256"
                        ),
                    ],
                ].assign(fold_id=fold_id)
            )
    oof = pd.concat(predictions, ignore_index=True)
    baseline_oof = oof.loc[
        oof["model_id"].eq("b0_empirical_v1")
        & oof["feature_block_id"].eq("D0")
    ].reset_index(drop=True)
    winner_oof = oof.loc[
        oof["model_id"].eq(request.model_id)
        & oof["feature_block_id"].eq(request.feature_block_id)
    ].reset_index(drop=True)
    paired = _paired_candidate_rows(
        baseline=baseline_oof,
        candidate=winner_oof,
    )
    evaluated_risk = pd.concat(risk_sets, ignore_index=True)
    complete_risk = _with_transport_pair_identity(
        kalshi,
        transport_scope=request.transport_scope,
    )
    if complete_risk["descriptive_transport_pair_id"].duplicated().any():
        raise RegulationModelInputError(
            "complete Kalshi bound risk set has duplicate pair identity"
        )
    fold_by_pair = evaluated_risk.set_index(
        "descriptive_transport_pair_id"
    )[
        "fold_id"
    ].to_dict()
    complete_risk["evaluation_fold_id"] = complete_risk[
        "descriptive_transport_pair_id"
    ].map(fold_by_pair)
    complete_risk["oof_evaluation_eligible"] = complete_risk[
        "evaluation_fold_id"
    ].notna()
    complete_risk["bound_risk_row"] = True
    complete_risk["post_trade_observed"] = complete_risk[
        "availability_h"
    ].astype(bool)
    complete_risk["risk_set_status"] = np.where(
        complete_risk["post_trade_observed"],
        "BOUND_WITH_NEW_ACTUAL_TRADE_OBSERVED",
        "BOUND_NO_NEW_ACTUAL_TRADE_OBSERVED",
    )
    complete_risk = complete_risk.loc[
        :,
        [
            "descriptive_transport_pair_id",
            "game_id",
            "information_anchor_id",
            "episode_id",
            "landmark_seconds",
            "endpoint_seconds",
            "availability_h",
            "direction_h",
            "direction_eligible",
            "transport_market_scope",
            "transport_outcome_scope",
            "transport_scope",
            "transport_target_version",
            "actual_home_contract_id",
            "logical_market_id",
            "actual_home_contract_identity_sha256",
            "native_raw_manifest_set_sha256",
            "native_raw_object_set_sha256",
            "native_rule_evidence_sha256",
            "native_rule_evidence_status",
            "realized_outcome_comparability_status",
            "realized_outcome_comparability_evidence_sha256",
            "strict_contract_rule_equivalence_status",
            "strict_contract_rule_equivalence_evidence_sha256",
            "source_row_id",
            "evaluation_fold_id",
            "oof_evaluation_eligible",
            "bound_risk_row",
            "post_trade_observed",
            "risk_set_status",
        ],
    ].rename(
        columns={
            "availability_h": "availability_truth",
            "direction_h": "direction_truth",
        }
    )
    poly_common = pd.concat(poly_common_parts, ignore_index=True).rename(
        columns={
            "source_row_id": "polymarket_source_row_id",
            "availability_h": "polymarket_availability_truth",
            "direction_h": "polymarket_direction_truth",
            "actual_home_contract_id": (
                "polymarket_actual_home_contract_id"
            ),
            "logical_market_id": "polymarket_logical_market_id",
            "actual_home_contract_identity_sha256": (
                "polymarket_contract_identity_sha256"
            ),
            "native_rule_evidence_sha256": (
                "polymarket_native_rule_evidence_sha256"
            ),
            "native_rule_evidence_status": (
                "polymarket_native_rule_evidence_status"
            ),
            "realized_outcome_comparability_status": (
                "polymarket_realized_outcome_comparability_status"
            ),
            "realized_outcome_comparability_evidence_sha256": (
                "polymarket_realized_outcome_comparability_"
                "evidence_sha256"
            ),
            "strict_contract_rule_equivalence_status": (
                "polymarket_strict_contract_rule_equivalence_status"
            ),
            "strict_contract_rule_equivalence_evidence_sha256": (
                "polymarket_strict_contract_rule_equivalence_"
                "evidence_sha256"
            ),
        }
    )
    kalshi_common = pd.concat(kalshi_common_parts, ignore_index=True).rename(
        columns={
            "source_row_id": "kalshi_source_row_id",
            "availability_h": "kalshi_availability_truth",
            "direction_h": "kalshi_direction_truth",
            "actual_home_contract_id": "kalshi_actual_home_contract_id",
            "logical_market_id": "kalshi_logical_market_id",
            "actual_home_contract_identity_sha256": (
                "kalshi_contract_identity_sha256"
            ),
            "native_rule_evidence_sha256": (
                "kalshi_native_rule_evidence_sha256"
            ),
            "native_rule_evidence_status": (
                "kalshi_native_rule_evidence_status"
            ),
            "realized_outcome_comparability_status": (
                "kalshi_realized_outcome_comparability_status"
            ),
            "realized_outcome_comparability_evidence_sha256": (
                "kalshi_realized_outcome_comparability_evidence_sha256"
            ),
            "strict_contract_rule_equivalence_status": (
                "kalshi_strict_contract_rule_equivalence_status"
            ),
            "strict_contract_rule_equivalence_evidence_sha256": (
                "kalshi_strict_contract_rule_equivalence_evidence_sha256"
            ),
        }
    )
    descriptive_pair_coverage = poly_common.merge(
        kalshi_common,
        on=["fold_id", "descriptive_transport_pair_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    descriptive_pair_coverage[
        "descriptive_pair_observed_on_both_venues"
    ] = descriptive_pair_coverage["_merge"].eq("both")
    both_status = (
        "BOTH_VENUES_OBSERVED_CONTRACT_RULE_EQUIVALENT"
        if request.strict_contract_rule_equivalence_status
        == STRICT_CONTRACT_RULE_EQUIVALENT
        else "BOTH_VENUES_OBSERVED_SETTLEMENT_EQUIVALENCE_UNPROVEN"
    )
    descriptive_pair_coverage["descriptive_pair_coverage_status"] = (
        descriptive_pair_coverage["_merge"].map(
            {
                "both": both_status,
                "left_only": "POLYMARKET_ONLY",
                "right_only": "KALSHI_ONLY",
            }
        )
    )
    descriptive_pair_coverage = descriptive_pair_coverage.drop(
        columns="_merge"
    )
    clean_anchor_games = int(
        evaluated_risk["game_id"].astype(str).nunique()
    )
    fold_class_audit: dict[str, dict[str, object]] = {}
    critical_fold_support = True
    for fold_id, fold_risk in evaluated_risk.groupby(
        "fold_id", sort=True
    ):
        availability_classes = sorted(
            fold_risk["availability_truth"].astype(int).unique().tolist()
        )
        direction_classes = sorted(
            fold_risk.loc[
                fold_risk["availability_truth"].astype(bool),
                "direction_truth",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        fold_passed = (
            availability_classes == [0, 1]
            and direction_classes == sorted(DIRECTION_CLASSES)
        )
        critical_fold_support &= fold_passed
        fold_class_audit[str(fold_id)] = {
            "availability_classes": availability_classes,
            "direction_classes": direction_classes,
            "support_passed": fold_passed,
        }
    support_sufficient = (
        clean_anchor_games >= KALSHI_TRANSPORT_MIN_CLEAN_ANCHOR_GAMES
        and critical_fold_support
    )
    transport_status = (
        (
            "FORMAL_CONTRACT_TRANSPORT_SUPPORT_SUFFICIENT"
            if support_sufficient
            else "FORMAL_CONTRACT_TRANSPORT_INSUFFICIENT_SUPPORT"
        )
        if request.formal_contract_transport_allowed
        else (
            "DESCRIPTIVE_PRODUCER_TRANSPORT_SUPPORT_SUFFICIENT"
            if support_sufficient
            else "DESCRIPTIVE_PRODUCER_TRANSPORT_INSUFFICIENT_SUPPORT"
        )
    )
    metric_rows: list[dict[str, object]] = []
    for (model_id, feature_block_id), model_rows in oof.groupby(
        ["model_id", "feature_block_id"], sort=True
    ):
        for metric_name, metric_value in compute_equal_game_metrics(
            model_rows
        ).items():
            metric_rows.append(
                {
                    "model_id": model_id,
                    "feature_block_id": feature_block_id,
                    "metric_scope": (
                        "FORMAL_CONTRACT_EQUAL_GAME_PRODUCER_TRANSPORT"
                        if request.formal_contract_transport_allowed
                        else "DESCRIPTIVE_EQUAL_GAME_PRODUCER_TRANSPORT"
                    ),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "bootstrap_standard_error": math.nan,
                    "bootstrap_ci_low": math.nan,
                    "bootstrap_ci_high": math.nan,
                    "game_count": int(model_rows["game_id"].nunique()),
                    "training_venue": POLYMARKET,
                    "evaluation_venue": KALSHI,
                    "transport_inference_status": transport_status,
                }
            )
    for metric_index, metric_name in enumerate(
        (
            "joint_log_loss",
            "availability_log_loss",
            "availability_brier",
            "direction_log_loss",
            "direction_multiclass_brier",
        )
    ):
        values = _game_level_improvement(paired, metric=metric_name)
        mean, standard_error, ci_low, ci_high = _game_bootstrap_summary(
            values,
            seed=SELECTION_BOOTSTRAP_SEED + 10_000 + metric_index,
        )
        metric_rows.append(
            {
                "model_id": request.model_id,
                "feature_block_id": request.feature_block_id,
                "metric_scope": (
                    (
                        "FORMAL_CONTRACT_PAIRED_KALSHI_WINNER_MINUS_"
                        "TRANSPORTED_B0"
                    )
                    if request.formal_contract_transport_allowed
                    else (
                        "DESCRIPTIVE_PAIRED_KALSHI_WINNER_MINUS_"
                        "TRANSPORTED_B0"
                    )
                ),
                "metric_name": metric_name,
                "metric_value": mean,
                "bootstrap_standard_error": standard_error,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "game_count": len(values),
                "training_venue": POLYMARKET,
                "evaluation_venue": KALSHI,
                "transport_inference_status": transport_status,
            }
        )
    run_config: Mapping[str, object] = MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "formal_input_verified": True,
            "input_lineage": input_lineage,
            "frozen_polymarket_selection_sha256": (
                selection_lock.selection_sha256
            ),
            "selection_lock_sha256": selection_lock.selection_lock_sha256,
            "winner_model_id": request.model_id,
            "winner_feature_block_id": request.feature_block_id,
            "transported_baseline_model_id": "b0_empirical_v1",
            "transported_baseline_feature_block_id": "D0",
            "target_version": TRANSPORT_TARGET_VERSION,
            "landmark_seconds": LANDMARK_SECONDS,
            "endpoint_seconds": ENDPOINT_SECONDS,
            "direction_threshold_probability": 0.01,
            "training_venue": POLYMARKET,
            "evaluation_venue": KALSHI,
            "selection_allowed": False,
            "reranking_allowed": False,
            "selection_anchor_kind": PRIMARY_ANCHOR_KIND,
            "selection_analysis_role": PRIMARY_ANALYSIS_ROLE,
            "transport_scope": request.transport_scope,
            "transport_market_scope": TRANSPORT_MARKET_SCOPE,
            "transport_outcome_scope": TRANSPORT_OUTCOME_SCOPE,
            "descriptive_transport_pair_identity": (
                "game_id+information_anchor_id+NFL_FULL_GAME_WINNER+"
                "HOME_TEAM_OUTCOME+L+H+target_version+transport_scope"
            ),
            "realized_outcome_comparability_status": (
                request.realized_outcome_comparability_status
            ),
            "realized_outcome_comparability_evidence_set_sha256": (
                request.realized_outcome_comparability_evidence_set_sha256
            ),
            "strict_contract_rule_equivalence_status": (
                request.strict_contract_rule_equivalence_status
            ),
            "strict_contract_rule_equivalence_evidence_set_sha256": (
                request
                .strict_contract_rule_equivalence_evidence_set_sha256
            ),
            "native_rule_evidence_status_by_venue": dict(
                request.native_rule_evidence_status_by_venue
            ),
            "native_rule_evidence_set_sha256_by_venue": dict(
                request.native_rule_evidence_set_sha256_by_venue
            ),
            "formal_contract_transport_allowed": (
                request.formal_contract_transport_allowed
            ),
            "independent_sports_holdout_validation_allowed": False,
            "strict_venue_effect_claim_allowed": False,
            "complete_bound_risk_set_row_count": len(complete_risk),
            "oof_evaluation_risk_set_row_count": len(evaluated_risk),
            "availability_zero_rows_retained": int(
                (~complete_risk["post_trade_observed"]).sum()
            ),
            "availability_zero_semantics": (
                "NO_NEW_ACTUAL_TRADE_OBSERVED; NOT_OPEN_LIQUID_OR_TRADABLE"
            ),
            "clean_anchor_game_count": clean_anchor_games,
            "minimum_clean_anchor_games": (
                KALSHI_TRANSPORT_MIN_CLEAN_ANCHOR_GAMES
            ),
            "fold_class_audit": fold_class_audit,
            "transport_inference_status": transport_status,
            "descriptive_pair_coverage_is_separate": True,
            "venue_speed_claim_allowed": False,
            "holdout_reaction_accessed": False,
        }
    )
    metric_frame = pd.DataFrame(metric_rows).assign(
        transport_scope=request.transport_scope,
        realized_outcome_comparability_status=(
            request.realized_outcome_comparability_status
        ),
        strict_contract_rule_equivalence_status=(
            request.strict_contract_rule_equivalence_status
        ),
        formal_contract_transport_allowed=(
            request.formal_contract_transport_allowed
        ),
        independent_sports_holdout_validation_allowed=False,
        venue_speed_claim_allowed=False,
    )
    support_frame = pd.DataFrame(supports).assign(
        transport_scope=request.transport_scope,
        realized_outcome_comparability_status=(
            request.realized_outcome_comparability_status
        ),
        realized_outcome_comparability_evidence_set_sha256=(
            request.realized_outcome_comparability_evidence_set_sha256
        ),
        strict_contract_rule_equivalence_status=(
            request.strict_contract_rule_equivalence_status
        ),
        strict_contract_rule_equivalence_evidence_set_sha256=(
            request.strict_contract_rule_equivalence_evidence_set_sha256
        ),
        formal_contract_transport_allowed=(
            request.formal_contract_transport_allowed
        ),
        independent_sports_holdout_validation_allowed=False,
        venue_speed_claim_allowed=False,
    )
    return RegulationKalshiTransportRun(
        oof_predictions=oof,
        fold_metrics=metric_frame,
        support_audit=support_frame,
        transport_risk_set=complete_risk,
        descriptive_pair_coverage=descriptive_pair_coverage,
        run_config=run_config,
        run_config_sha256=_sha256(run_config),
    )


__all__ = [
    "CORRECTED_MODEL_CELLS",
    "DESCRIPTIVE_TRANSPORT_SCOPE",
    "DEVELOPMENT_SPLIT",
    "DIRECTION_CLASSES",
    "ENDPOINT_SECONDS",
    "FEATURE_BLOCKS",
    "FORMAL_CONTRACT_TRANSPORT_SCOPE",
    "FrozenPolymarketWinnerV2",
    "KalshiTransportRequestV2",
    "LANDMARK_SECONDS",
    "MODEL_IDS",
    "PRIMARY_ANALYSIS_ROLE",
    "PRIMARY_ANCHOR_KIND",
    "PRIMARY_FEATURE_CONTRACT",
    "PRESTATE_CONTEXT_CONTRACT",
    "POLYMARKET",
    "PublishedRegulationOOFBatch",
    "PublishedRegulationOOFShard",
    "PublishedRegulationSelectionLock",
    "REQUIRED_PANEL_COLUMNS",
    "RegulationModelInputError",
    "RegulationKalshiTransportRun",
    "RegulationModelRun",
    "RegulationSelectionCandidate",
    "RegulationSelectionResult",
    "SCHEMA_VERSION",
    "SELECTABLE_CANDIDATES",
    "TRAINING_PROVENANCE_COLUMNS",
    "TRANSPORT_MARKET_SCOPE",
    "TRANSPORT_OUTCOME_SCOPE",
    "TRANSPORT_TARGET_VERSION",
    "VerifiedRegulationOOFShard",
    "VerifiedNativeRuleOverlay",
    "VerifiedRegulationPanelBatch",
    "VerifiedRegulationPanelManifest",
    "VerifiedRegulationSelectionLock",
    "compute_equal_game_metrics",
    "discover_verified_regulation_oof_shards",
    "load_verified_regulation_panel_batch",
    "prepare_regulation_panel",
    "publish_regulation_oof_batch_index",
    "publish_regulation_oof_shard",
    "publish_regulation_selection_lock",
    "regulation_oof_batch_status",
    "regulation_row_losses",
    "run_frozen_kalshi_transport",
    "run_regulation_walk_forward",
    "select_regulation_model",
    "validate_kalshi_transport",
    "verify_regulation_oof_shard",
    "verify_regulation_panel_batch_manifest",
    "verify_regulation_selection_lock",
    "verify_native_rule_transport_overlay",
]
