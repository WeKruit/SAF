"""Frozen, paired model selection over the real NFL X-15 OOF schema.

The selection surface deliberately consumes :class:`X15ModelRun` rather than
an ad-hoc long-form probability table.  B0 and a candidate are compared on
identical wide OOF rows, with equal head weight inside a row, equal row weight
inside an episode, and equal episode weight inside a game.  Games are the
bootstrap clusters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd

from prediction_market.research.nfl_x15_models import (
    DIAGNOSTIC_ANALYSIS_SCOPE,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS,
    DIAGNOSTIC_FEATURE_BLOCKS,
    DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT,
    DIAGNOSTIC_SCHEMA_VERSION,
    DIAGNOSTIC_TARGET_CONTRACT,
    DIAGNOSTIC_VENUE_TICK_SUPPORT,
    SURVIVAL_PROBABILITY_CONTRACT,
    X15ModelRun,
)
from prediction_market.research.nfl_x15_statistics import (
    DEFAULT_MAX_GAME_CONTRIBUTION,
    DEFAULT_MAX_Q_VALUE,
    DEFAULT_MIN_LOO_SAME_SIGN,
    X15StatisticsInputError,
    summarize_development_signals,
)


BASELINE_MODEL_ID: Final[str] = "b0_empirical_v1"
BASELINE_FEATURE_BLOCK_ID: Final[str] = "D0"
STAGE_B_CANDIDATE_SUITE: Final[
    tuple[tuple[str, str], ...]
] = (
    ("regularized_logistic_v1", "D1"),
    ("regularized_logistic_v1", "D2"),
    ("regularized_logistic_v1", "D3"),
    ("regularized_logistic_v1", "D4"),
    ("shallow_xgboost_v1", "D1"),
    ("shallow_xgboost_v1", "D2"),
    ("shallow_xgboost_v1", "D3"),
    ("shallow_xgboost_v1", "D4"),
)
BOOTSTRAP_SAMPLES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 20260729
ANCHOR_LANDMARK_SECONDS: Final[int] = 3
ANCHOR_ENDPOINT_SECONDS: Final[int] = 30
MIN_ANCHOR_GAMES: Final[int] = 30
LOSS_IMPROVEMENT_SIGN_SEMANTICS: Final[str] = (
    "POSITIVE_MEANS_CANDIDATE_LOSS_IS_LOWER_THAN_B0"
)
DIRECTION_THRESHOLD_PROBABILITY: Final[float] = (
    DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
)
DIRECTION_THRESHOLD_SEMANTICS: Final[str] = (
    DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
)
VENUE_TICK_SUPPORT: Final[str] = DIAGNOSTIC_VENUE_TICK_SUPPORT
MARKET_CONTINUITY_SUPPORT: Final[str] = (
    DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT
)
HISTORICAL_TARGET_CONTRACT: Final[str] = DIAGNOSTIC_TARGET_CONTRACT
HISTORICAL_CLAIM_BOUNDARY: Final[str] = DIAGNOSTIC_CLAIM_BOUNDARY
HISTORICAL_SCHEMA_VERSION: Final[str] = DIAGNOSTIC_SCHEMA_VERSION
HISTORICAL_ANALYSIS_SCOPE: Final[str] = DIAGNOSTIC_ANALYSIS_SCOPE
EXPECTED_DEVELOPMENT_GAME_COUNT: Final[int] = 153
EXPECTED_FOLDS: Final[
    tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
] = (
    ("fold_01", (1, 2), (3, 4)),
    ("fold_02", (1, 2, 3, 4), (5, 6)),
    ("fold_03", tuple(range(1, 7)), (7, 8)),
    ("fold_04", tuple(range(1, 9)), (9, 10)),
    ("fold_05", tuple(range(1, 11)), (11, 12)),
)

_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "source_row_id",
    "game_id",
    "nfl_week",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
    "fold_id",
    "cohort_authority_sha256",
    "target_contract",
    "claim_boundary",
    "direction_threshold_probability",
    "direction_threshold_semantics",
    "venue_tick_support",
    "market_continuity_support",
    "schema_version",
    "analysis_scope",
    "claim_eligible",
    "calibration_venue",
    "transport_mode",
)
_IDENTITY_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "training_venue",
    "actual_home_contract_id",
    "fold_id",
    "model_id",
    "feature_block_id",
    "cohort_authority_sha256",
    "target_contract",
    "claim_boundary",
    "direction_threshold_semantics",
    "venue_tick_support",
    "market_continuity_support",
    "schema_version",
    "analysis_scope",
    "calibration_venue",
    "transport_mode",
)
_TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    "s_h_truth",
    "o_h_given_s_truth",
    "direction_truth",
)
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *_PAIR_COLUMNS,
        *_IDENTITY_TEXT_COLUMNS,
        *_TRUTH_COLUMNS,
        "model_id",
        "feature_block_id",
        "s_h_calibrated_probability",
        "o_h_given_s_calibrated_probability",
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
        "train_weeks",
        "validation_weeks",
        "training_game_ids",
        "validation_game_ids",
        "preprocessor_fit_game_ids",
    }
)
_DIRECTION_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "direction_calibrated_prob_down",
    "direction_calibrated_prob_no_move",
    "direction_calibrated_prob_up",
)
_DIRECTION_INDEX: Final[dict[str, int]] = {
    "DOWN": 0,
    "NO_MOVE": 1,
    "UP": 2,
}
_FORBIDDEN_SELECTION_COLUMNS: Final[frozenset[str]] = frozenset(
    {"factor_id", "factor_version", "predicate_evidence"}
)


class ModelSelectionError(ValueError):
    """The real OOF run cannot support the frozen paired selection claim."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise ModelSelectionError(f"{field} must be a sha256 digest")
    return str(value)


def _require_nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelSelectionError(f"{field} must be a nonempty string")
    return value


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(child)
            for key, child in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


STAGE_B_WINNER_RULE_SHA256: Final[str] = _canonical_sha256(
    {
        "rule": "ONE_STANDARD_ERROR_V1",
        "candidate_suite": STAGE_B_CANDIDATE_SUITE,
        "baseline": (BASELINE_MODEL_ID, BASELINE_FEATURE_BLOCK_ID),
        "candidate_gate": (
            "AUTHORITY_AND_INTEGRATED_PAIRED_GAME_CLUSTER_CI_LOW_GT_0"
            "_AND_L3_H30_ANCHOR_AT_LEAST_30_GAMES_NO_SIGN_REVERSAL"
        ),
        "best": "MAX_INTEGRATED_MEAN_IMPROVEMENT",
        "best_standard_error": (
            "SAMPLE_SD_PER_GAME_LOSS_IMPROVEMENT_DIV_SQRT_GAME_COUNT"
        ),
        "eligible": "MEAN_GTE_BEST_MEAN_MINUS_BEST_STANDARD_ERROR",
        "simplicity_order": STAGE_B_CANDIDATE_SUITE,
    }
)


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentAuthority:
    """Exact 153-game metadata binding used by the final selection gate."""

    cohort_authority_sha256: str
    development_games: tuple[
        tuple[str, int, str, str], ...
    ]
    metadata_sha256: str

    @property
    def game_count(self) -> int:
        return len(self.development_games)

    @property
    def game_weeks(self) -> dict[str, int]:
        return {
            game_id: week
            for game_id, week, _, _ in self.development_games
        }


def bind_frozen_development_authority(
    development_metadata: pd.DataFrame,
    *,
    cohort_authority_sha256: str,
) -> FrozenDevelopmentAuthority:
    """Bind exact governed IDs, weeks, kickoffs, and batch hashes."""

    authority = _require_sha256(
        cohort_authority_sha256,
        field="cohort_authority_sha256",
    )
    required = {
        "game_id",
        "nfl_week",
        "kickoff_utc",
        "batch_sha256",
        "cohort_authority_sha256",
    }
    if not isinstance(development_metadata, pd.DataFrame):
        raise ModelSelectionError(
            "development_metadata must be a DataFrame"
        )
    missing = sorted(required.difference(development_metadata.columns))
    if missing:
        raise ModelSelectionError(
            f"development_metadata missing required columns: {missing}"
        )
    frame = development_metadata.loc[:, sorted(required)].copy()
    if (
        len(frame) != EXPECTED_DEVELOPMENT_GAME_COUNT
        or frame["game_id"].nunique() != EXPECTED_DEVELOPMENT_GAME_COUNT
    ):
        raise ModelSelectionError(
            "selection authority requires exactly 153 unique "
            "development game IDs"
        )
    for column in ("game_id", "kickoff_utc"):
        if not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ModelSelectionError(
                f"development_metadata {column} must be nonempty text"
            )
    kickoff = pd.to_datetime(
        frame["kickoff_utc"], utc=True, errors="coerce"
    )
    explicitly_utc = frame["kickoff_utc"].str.endswith(
        ("Z", "+00:00")
    )
    if kickoff.isna().any() or not explicitly_utc.all():
        raise ModelSelectionError(
            "selection authority kickoff_utc must be an explicit UTC "
            "timestamp"
        )
    weeks = pd.to_numeric(frame["nfl_week"], errors="coerce")
    if (
        weeks.isna().any()
        or not np.equal(
            weeks.to_numpy(dtype=float),
            weeks.astype(int).to_numpy(dtype=float),
        ).all()
        or not weeks.between(1, 12).all()
        or set(weeks.astype(int)) != set(range(1, 13))
    ):
        raise ModelSelectionError(
            "selection authority nfl_week must cover integral weeks 1..12"
        )
    if not frame["batch_sha256"].map(_is_sha256).all():
        raise ModelSelectionError(
            "selection authority batch_sha256 values are invalid"
        )
    if not frame["cohort_authority_sha256"].eq(authority).all():
        raise ModelSelectionError(
            "selection authority rows disagree on "
            "cohort_authority_sha256"
        )
    games = tuple(
        (
            str(row.game_id),
            int(row.nfl_week),
            str(row.kickoff_utc),
            str(row.batch_sha256),
        )
        for row in frame.sort_values(
            "game_id", kind="mergesort"
        ).itertuples(index=False)
    )
    return FrozenDevelopmentAuthority(
        cohort_authority_sha256=authority,
        development_games=games,
        metadata_sha256=_canonical_sha256(games),
    )


@dataclass(frozen=True, slots=True)
class FrozenSelectionSpec:
    """Pre-locked identity, bootstrap, weighting, and anchor semantics."""

    candidate_model_id: str
    candidate_feature_block_id: str
    selection_venue: str = "polymarket"
    cohort_authority_sha256: str | None = None
    baseline_model_id: str = BASELINE_MODEL_ID
    baseline_feature_block_id: str = BASELINE_FEATURE_BLOCK_ID
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    confidence_level: float = 0.95
    anchor_landmark_seconds: int = ANCHOR_LANDMARK_SECONDS
    anchor_endpoint_seconds: int = ANCHOR_ENDPOINT_SECONDS
    loss_improvement_sign_semantics: str = (
        LOSS_IMPROVEMENT_SIGN_SEMANTICS
    )

    def __post_init__(self) -> None:
        _require_nonempty_text(
            self.candidate_model_id, field="candidate_model_id"
        )
        _require_nonempty_text(
            self.candidate_feature_block_id,
            field="candidate_feature_block_id",
        )
        _require_nonempty_text(self.selection_venue, field="selection_venue")
        if self.selection_venue != "polymarket":
            raise ModelSelectionError(
                "selection_venue is fixed at polymarket"
            )
        if self.cohort_authority_sha256 is not None:
            _require_sha256(
                self.cohort_authority_sha256,
                field="cohort_authority_sha256",
            )
        if (
            self.baseline_model_id != BASELINE_MODEL_ID
            or self.baseline_feature_block_id
            != BASELINE_FEATURE_BLOCK_ID
        ):
            raise ModelSelectionError(
                "the frozen diagnostic baseline pair is "
                "b0_empirical_v1/D0"
            )
        if (
            self.bootstrap_samples != BOOTSTRAP_SAMPLES
            or self.bootstrap_seed != BOOTSTRAP_SEED
        ):
            raise ModelSelectionError(
                "the paired game bootstrap is frozen at 10,000 draws "
                "with seed 20260729"
            )
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, Real)
            or float(self.confidence_level) != 0.95
        ):
            raise ModelSelectionError(
                "confidence_level is frozen at 0.95"
            )
        if (
            self.anchor_landmark_seconds != ANCHOR_LANDMARK_SECONDS
            or self.anchor_endpoint_seconds != ANCHOR_ENDPOINT_SECONDS
        ):
            raise ModelSelectionError(
                "the confirmatory anchor is frozen at L=3 -> H=30"
            )
        if (
            self.loss_improvement_sign_semantics
            != LOSS_IMPROVEMENT_SIGN_SEMANTICS
        ):
            raise ModelSelectionError(
                "loss-improvement sign semantics cannot be changed"
            )
        if (
            self.candidate_model_id == self.baseline_model_id
            and self.candidate_feature_block_id
            == self.baseline_feature_block_id
        ):
            raise ModelSelectionError("candidate and B0 must differ")
        if self.candidate_feature_block_id not in (
            set(DIAGNOSTIC_FEATURE_BLOCKS) - {"D0"}
        ):
            raise ModelSelectionError(
                "historical diagnostic candidates must use D1..D4"
            )


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Frozen dual-gate result and the exact audit surfaces behind it."""

    spec: FrozenSelectionSpec
    cohort_authority_sha256: str
    run_config_sha256: str
    schema_version: str
    analysis_scope: str
    target_contract: str
    claim_boundary: str
    diagnostic_status: str
    execution_claim_eligible: bool
    tick_claim_eligible: bool
    continuity_claim_eligible: bool
    authority_gate_passed: bool
    authority_gate_failures: tuple[str, ...]
    development_authority_game_count: int
    observed_fold_ids: tuple[str, ...]
    paired_rows: pd.DataFrame
    episode_losses: pd.DataFrame
    game_losses: pd.DataFrame
    integrated_mean_improvement: float
    integrated_ci_low: float
    integrated_ci_high: float
    bootstrap_probability_improved: float
    bootstrap_samples: int
    bootstrap_seed: int
    integrated_gate_passed: bool
    anchor_game_losses: pd.DataFrame
    anchor_game_count: int
    anchor_episode_count: int
    anchor_game_coverage: float
    anchor_episode_coverage: float
    anchor_support_status: str
    anchor_mean_improvement: float
    anchor_sign_reversed: bool
    anchor_gate_passed: bool
    loss_improvement_sign_semantics: str
    selected: bool


@dataclass(frozen=True, slots=True)
class StageBModelSelectionResult:
    """Frozen eight-candidate Stage-B V2 winner decision and audit."""

    decision_status: str
    winner: ModelSelectionResult | None
    candidate_results: tuple[ModelSelectionResult, ...]
    candidate_audit: pd.DataFrame
    best_integrated_mean_improvement: float | None
    best_standard_error: float | None
    one_se_threshold: float | None
    cohort_authority_sha256: str
    shared_run_config_sha256: str
    suite_run_config_sha256: str
    candidate_audit_sha256: str
    schema_version: str
    survival_probability_contract: str
    winner_rule_sha256: str


def _nullable_equal(left: pd.Series, right: pd.Series) -> bool:
    for left_value, right_value in zip(left, right, strict=True):
        left_missing = bool(pd.isna(left_value))
        right_missing = bool(pd.isna(right_value))
        if left_missing or right_missing:
            if left_missing and right_missing:
                continue
            return False
        if left_value != right_value:
            return False
    return True


def _validate_positive_integer_column(
    frame: pd.DataFrame, column: str
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if (
        values.isna().any()
        or not np.equal(
            values.to_numpy(dtype=float),
            values.astype(int).to_numpy(dtype=float),
        ).all()
        or (values <= 0).any()
    ):
        raise ModelSelectionError(f"{column} must be a positive integer")
    return values.astype(int)


def _validate_task4_rows(
    model_run: X15ModelRun, *, spec: FrozenSelectionSpec
) -> tuple[pd.DataFrame, str]:
    if not isinstance(model_run, X15ModelRun):
        raise ModelSelectionError(
            "selection requires an actual Task4 X15ModelRun"
        )
    _require_sha256(
        model_run.run_config_sha256, field="run_config_sha256"
    )
    if not isinstance(model_run.run_config, Mapping):
        raise ModelSelectionError(
            "selection requires the stamped Task4 diagnostic run_config"
        )
    if (
        _canonical_sha256(model_run.run_config)
        != model_run.run_config_sha256
    ):
        raise ModelSelectionError("Task4 run_config SHA-256 mismatch")
    expected_run_config = {
        "schema_version": HISTORICAL_SCHEMA_VERSION,
        "survival_probability_contract": (
            SURVIVAL_PROBABILITY_CONTRACT
        ),
        "target_contract": HISTORICAL_TARGET_CONTRACT,
        "claim_boundary": HISTORICAL_CLAIM_BOUNDARY,
        "analysis_scope": HISTORICAL_ANALYSIS_SCOPE,
        "direction_threshold_probability": (
            DIRECTION_THRESHOLD_PROBABILITY
        ),
        "direction_threshold_semantics": (
            DIRECTION_THRESHOLD_SEMANTICS
        ),
        "venue_tick_support": VENUE_TICK_SUPPORT,
        "market_continuity_support": MARKET_CONTINUITY_SUPPORT,
        "claim_eligible": False,
    }
    for field, expected in expected_run_config.items():
        if model_run.run_config.get(field) != expected:
            if field == "schema_version":
                raise ModelSelectionError(
                    "selection requires "
                    "HistoricalTradesOnlyProbabilityPanelV2 run_config"
                )
            raise ModelSelectionError(
                f"Task4 diagnostic run_config drifted on {field}"
            )
    registered_blocks = model_run.run_config.get("feature_blocks")
    if (
        not isinstance(registered_blocks, Mapping)
        or tuple(registered_blocks) != tuple(DIAGNOSTIC_FEATURE_BLOCKS)
        or spec.baseline_feature_block_id not in registered_blocks
        or spec.candidate_feature_block_id not in registered_blocks
    ):
        raise ModelSelectionError(
            "Task4 run_config must register immutable D0..D4 "
            "diagnostic feature blocks"
        )
    predictions = model_run.oof_predictions
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ModelSelectionError(
            "X15ModelRun.oof_predictions must be a nonempty DataFrame"
        )
    if predictions.columns.duplicated().any():
        raise ModelSelectionError("OOF predictions have duplicate columns")
    missing = sorted(_REQUIRED_COLUMNS.difference(predictions.columns))
    if missing:
        raise ModelSelectionError(
            f"Task4 OOF predictions missing required columns: {missing}"
        )
    forbidden = sorted(
        _FORBIDDEN_SELECTION_COLUMNS.intersection(predictions.columns)
    )
    if forbidden:
        raise ModelSelectionError(
            "factor membership is claim-only and cannot enter model "
            f"selection: {forbidden}"
        )

    baseline_mask = (
        predictions["model_id"].eq(spec.baseline_model_id)
        & predictions["feature_block_id"].eq(
            spec.baseline_feature_block_id
        )
        & predictions["venue"].eq(spec.selection_venue)
        & predictions["training_venue"].eq(spec.selection_venue)
    )
    candidate_mask = (
        predictions["model_id"].eq(spec.candidate_model_id)
        & predictions["feature_block_id"].eq(
            spec.candidate_feature_block_id
        )
        & predictions["venue"].eq(spec.selection_venue)
        & predictions["training_venue"].eq(spec.selection_venue)
    )
    work = predictions.loc[baseline_mask | candidate_mask].copy()
    if not baseline_mask.any() or not candidate_mask.any():
        raise ModelSelectionError(
            "the exact candidate/B0 model and feature-block pairs are "
            "required in the Task4 OOF run"
        )

    for column in _IDENTITY_TEXT_COLUMNS:
        valid = work[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            raise ModelSelectionError(
                f"Task4 OOF {column} must be a nonempty string"
            )
    work["nfl_week"] = _validate_positive_integer_column(
        work, "nfl_week"
    )
    if not work["nfl_week"].between(1, 12).all():
        raise ModelSelectionError(
            "development nfl_week must be integral in 1..12"
        )
    for column in ("landmark_seconds", "endpoint_seconds"):
        work[column] = _validate_positive_integer_column(work, column)
    threshold = pd.to_numeric(
        work["direction_threshold_probability"], errors="coerce"
    )
    if (
        threshold.isna().any()
        or not np.isfinite(threshold.to_numpy(dtype=float)).all()
        or not threshold.eq(DIRECTION_THRESHOLD_PROBABILITY).all()
    ):
        raise ModelSelectionError(
            "direction_threshold_probability is frozen at 0.01"
        )
    if not work["direction_threshold_semantics"].eq(
        DIRECTION_THRESHOLD_SEMANTICS
    ).all():
        raise ModelSelectionError(
            "direction threshold must remain fixed cross-venue research "
            "materiality, not a tick claim"
        )
    if not work["venue_tick_support"].eq(VENUE_TICK_SUPPORT).all():
        raise ModelSelectionError(
            "venue_tick_support must preserve raw status UNSUPPORTED"
        )
    if not work["market_continuity_support"].eq(
        MARKET_CONTINUITY_SUPPORT
    ).all():
        raise ModelSelectionError(
            "market_continuity_support must preserve raw status UNKNOWN"
        )
    if not work["schema_version"].eq(
        HISTORICAL_SCHEMA_VERSION
    ).all():
        raise ModelSelectionError(
            "selection requires HistoricalTradesOnlyProbabilityPanelV2"
        )
    if not work["analysis_scope"].eq(
        HISTORICAL_ANALYSIS_SCOPE
    ).all():
        raise ModelSelectionError(
            "selection requires the exact historical source-time "
            "diagnostic analysis_scope"
        )
    if not work["claim_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and not bool(value)
    ).all():
        raise ModelSelectionError(
            "historical diagnostic rows must preserve claim_eligible=False"
        )
    if (
        not work["calibration_venue"].eq(spec.selection_venue).all()
        or not work["transport_mode"].eq("VENUE_SPECIFIC").all()
    ):
        raise ModelSelectionError(
            "selection accepts native venue-specific OOF rows only"
        )
    if not work["target_contract"].eq(
        HISTORICAL_TARGET_CONTRACT
    ).all():
        raise ModelSelectionError(
            "selection requires the exact historical trades-only "
            "target_contract"
        )
    if not work["claim_boundary"].eq(
        HISTORICAL_CLAIM_BOUNDARY
    ).all():
        raise ModelSelectionError(
            "selection requires the exact historical source-time "
            "probability diagnostic claim_boundary"
        )

    authorities = tuple(
        sorted(set(work["cohort_authority_sha256"].astype(str)))
    )
    if len(authorities) != 1 or not _is_sha256(authorities[0]):
        raise ModelSelectionError(
            "Task4 OOF must bind one cohort_authority_sha256"
        )
    authority = authorities[0]
    if (
        spec.cohort_authority_sha256 is not None
        and authority != spec.cohort_authority_sha256
    ):
        raise ModelSelectionError(
            "Task4 OOF cohort_authority_sha256 does not match the "
            "frozen selection spec"
        )

    duplicate = work.duplicated(
        [*_PAIR_COLUMNS, "model_id", "feature_block_id"], keep=False
    )
    if duplicate.any():
        raise ModelSelectionError(
            "each model must have one OOF row per exact candidate/B0 pair"
        )
    return work, authority


def _tuple_value(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, (tuple, list)):
        return None
    return tuple(value)


def _development_authority_gate(
    model_run: X15ModelRun,
    work: pd.DataFrame,
    *,
    row_authority_sha256: str,
    authority: FrozenDevelopmentAuthority | None,
    spec: FrozenSelectionSpec,
) -> tuple[bool, tuple[str, ...], int, tuple[str, ...]]:
    failures: list[str] = []
    observed_fold_ids = tuple(
        sorted(set(work["fold_id"].astype(str)))
    )
    if authority is None:
        return (
            False,
            ("FROZEN_DEVELOPMENT_AUTHORITY_REQUIRED",),
            0,
            observed_fold_ids,
        )
    if not isinstance(authority, FrozenDevelopmentAuthority):
        raise ModelSelectionError(
            "authority must be FrozenDevelopmentAuthority"
        )
    if authority.game_count != EXPECTED_DEVELOPMENT_GAME_COUNT:
        failures.append("EXACT_153_GAME_AUTHORITY_REQUIRED")
    if (
        authority.cohort_authority_sha256 != row_authority_sha256
        or model_run.run_config.get("cohort_authority_sha256")
        != authority.cohort_authority_sha256
    ):
        failures.append("COHORT_AUTHORITY_MISMATCH")

    expected_fold_ids = tuple(fold[0] for fold in EXPECTED_FOLDS)
    configured_fold_ids = _tuple_value(
        model_run.run_config.get("fold_ids")
    )
    if (
        configured_fold_ids != expected_fold_ids
        or observed_fold_ids != expected_fold_ids
    ):
        failures.append("ALL_FIVE_FOLDS_REQUIRED")

    game_weeks = authority.game_weeks
    authority_weeks = work["game_id"].map(game_weeks)
    if (
        authority_weeks.isna().any()
        or not authority_weeks.astype("Int64").eq(
            work["nfl_week"].astype("Int64")
        ).all()
    ):
        failures.append("GAME_WEEK_AUTHORITY_MISMATCH")
    for fold_id, train_weeks, validation_weeks in EXPECTED_FOLDS:
        expected_training_ids = tuple(
            sorted(
                game_id
                for game_id, week in game_weeks.items()
                if week in train_weeks
            )
        )
        expected_validation_ids = tuple(
            sorted(
                game_id
                for game_id, week in game_weeks.items()
                if week in validation_weeks
            )
        )
        fold_rows = work.loc[work["fold_id"].eq(fold_id)]
        if fold_rows.empty:
            failures.append(f"{fold_id}:MISSING")
            continue
        for column, expected in (
            ("train_weeks", train_weeks),
            ("validation_weeks", validation_weeks),
            ("training_game_ids", expected_training_ids),
            ("validation_game_ids", expected_validation_ids),
            ("preprocessor_fit_game_ids", expected_training_ids),
        ):
            if not fold_rows[column].map(
                lambda value: _tuple_value(value) == tuple(expected)
            ).all():
                failures.append(f"{fold_id}:{column.upper()}_MISMATCH")
        for model_id, block_id in (
            (
                spec.baseline_model_id,
                spec.baseline_feature_block_id,
            ),
            (
                spec.candidate_model_id,
                spec.candidate_feature_block_id,
            ),
        ):
            observed_games = tuple(
                sorted(
                    set(
                        fold_rows.loc[
                            fold_rows["model_id"].eq(model_id)
                            & fold_rows["feature_block_id"].eq(block_id),
                            "game_id",
                        ].astype(str)
                    )
                )
            )
            if observed_games != expected_validation_ids:
                failures.append(
                    f"{fold_id}:{model_id}/{block_id}:"
                    "VALIDATION_GAME_COVERAGE_MISMATCH"
                )
    return (
        not failures,
        tuple(dict.fromkeys(failures)),
        authority.game_count,
        observed_fold_ids,
    )


def _merge_exact_pairs(
    work: pd.DataFrame, *, spec: FrozenSelectionSpec
) -> pd.DataFrame:
    baseline = work.loc[
        work["model_id"].eq(spec.baseline_model_id)
        & work["feature_block_id"].eq(
            spec.baseline_feature_block_id
        )
    ].copy()
    candidate = work.loc[
        work["model_id"].eq(spec.candidate_model_id)
        & work["feature_block_id"].eq(
            spec.candidate_feature_block_id
        )
    ].copy()
    pairs = baseline.merge(
        candidate,
        on=list(_PAIR_COLUMNS),
        suffixes=("_b0", "_candidate"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not pairs["_merge"].eq("both").all():
        raise ModelSelectionError(
            "exact candidate/B0 pairs must match on source row, game, "
            "episode, venue, fold, L, and H"
        )
    pairs = pairs.drop(columns="_merge")
    if not pairs["training_venue_b0"].eq(
        pairs["training_venue_candidate"]
    ).all():
        raise ModelSelectionError(
            "exact candidate/B0 pairs disagree on training_venue"
        )
    for column in _TRUTH_COLUMNS:
        if not _nullable_equal(
            pairs[f"{column}_b0"],
            pairs[f"{column}_candidate"],
        ):
            raise ModelSelectionError(
                f"exact candidate/B0 pairs disagree on {column}"
            )
    return pairs


def _binary_losses(
    pairs: pd.DataFrame,
    *,
    truth_column: str,
    probability_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_losses = np.full(len(pairs), np.nan, dtype=float)
    candidate_losses = np.full(len(pairs), np.nan, dtype=float)
    for position, (truth, baseline_probability, candidate_probability) in (
        enumerate(
            zip(
                pairs[f"{truth_column}_b0"],
                pairs[f"{probability_column}_b0"],
                pairs[f"{probability_column}_candidate"],
                strict=True,
            )
        )
    ):
        if pd.isna(truth):
            continue
        if isinstance(truth, (bool, np.bool_)):
            outcome = int(truth)
        elif (
            isinstance(truth, (Integral, np.integer))
            and not isinstance(truth, (bool, np.bool_))
            and int(truth) in (0, 1)
        ):
            outcome = int(truth)
        else:
            raise ModelSelectionError(
                f"{truth_column} must be binary or missing"
            )
        probabilities = np.asarray(
            [baseline_probability, candidate_probability], dtype=float
        )
        if (
            not np.isfinite(probabilities).all()
            or (probabilities <= 0).any()
            or (probabilities >= 1).any()
        ):
            raise ModelSelectionError(
                f"available {truth_column} rows require paired calibrated "
                "probabilities in (0, 1)"
            )
        losses = -(
            outcome * np.log(probabilities)
            + (1 - outcome) * np.log(1 - probabilities)
        )
        baseline_losses[position], candidate_losses[position] = losses
    return baseline_losses, candidate_losses


def _direction_losses(
    pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_losses = np.full(len(pairs), np.nan, dtype=float)
    candidate_losses = np.full(len(pairs), np.nan, dtype=float)
    baseline_columns = [
        f"{column}_b0" for column in _DIRECTION_PROBABILITY_COLUMNS
    ]
    candidate_columns = [
        f"{column}_candidate"
        for column in _DIRECTION_PROBABILITY_COLUMNS
    ]
    for position, truth in enumerate(pairs["direction_truth_b0"]):
        if pd.isna(truth):
            continue
        if str(truth) not in _DIRECTION_INDEX:
            raise ModelSelectionError(
                "direction_truth must be DOWN, NO_MOVE, UP, or missing"
            )
        baseline = pairs.loc[
            pairs.index[position], baseline_columns
        ].to_numpy(dtype=float)
        candidate = pairs.loc[
            pairs.index[position], candidate_columns
        ].to_numpy(dtype=float)
        probabilities = np.vstack([baseline, candidate])
        if (
            not np.isfinite(probabilities).all()
            or (probabilities <= 0).any()
            or (probabilities >= 1).any()
            or not np.allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0
            )
        ):
            raise ModelSelectionError(
                "available direction rows require paired calibrated "
                "three-class probabilities that sum to one"
            )
        truth_index = _DIRECTION_INDEX[str(truth)]
        baseline_losses[position] = -np.log(baseline[truth_index])
        candidate_losses[position] = -np.log(candidate[truth_index])
    return baseline_losses, candidate_losses


def _attach_multihead_losses(pairs: pd.DataFrame) -> pd.DataFrame:
    result = pairs.copy().reset_index(drop=True)
    s_h_b0, s_h_candidate = _binary_losses(
        result,
        truth_column="s_h_truth",
        probability_column="s_h_calibrated_probability",
    )
    observed_b0, observed_candidate = _binary_losses(
        result,
        truth_column="o_h_given_s_truth",
        probability_column="o_h_given_s_calibrated_probability",
    )
    direction_b0, direction_candidate = _direction_losses(result)
    baseline_heads = np.column_stack(
        [s_h_b0, observed_b0, direction_b0]
    )
    candidate_heads = np.column_stack(
        [s_h_candidate, observed_candidate, direction_candidate]
    )
    availability = np.isfinite(baseline_heads) & np.isfinite(
        candidate_heads
    )
    if not np.array_equal(
        np.isfinite(baseline_heads), np.isfinite(candidate_heads)
    ):
        raise ModelSelectionError(
            "candidate and B0 have asymmetric head availability"
        )
    available_count = availability.sum(axis=1)
    if (available_count == 0).any():
        raise ModelSelectionError(
            "every exact candidate/B0 row needs at least one available head"
        )
    result["available_head_count"] = available_count
    result["baseline_integrated_row_loss"] = np.nansum(
        baseline_heads, axis=1
    )
    result["candidate_integrated_row_loss"] = np.nansum(
        candidate_heads, axis=1
    )
    result["loss_improvement"] = (
        result["baseline_integrated_row_loss"]
        - result["candidate_integrated_row_loss"]
    )
    result["direction_truth"] = result["direction_truth_b0"]

    episode_columns = [
        "game_id",
        "atomic_information_episode_id",
    ]
    rows_per_episode = result.groupby(
        episode_columns, sort=True
    )["game_id"].transform("size")
    episodes_per_game = result.groupby("game_id", sort=True)[
        "atomic_information_episode_id"
    ].transform("nunique")
    result["row_weight_within_episode"] = 1.0 / rows_per_episode
    result["episode_weight_within_game"] = 1.0 / episodes_per_game
    result["game_weight"] = 1.0
    result["hierarchical_row_weight"] = (
        result["row_weight_within_episode"]
        * result["episode_weight_within_game"]
    )
    return result


def _hierarchical_losses(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_keys = [
        "game_id",
        "atomic_information_episode_id",
        "venue",
    ]
    episodes = (
        rows.groupby(episode_keys, sort=True, as_index=False)
        .agg(
            baseline_loss=("baseline_integrated_row_loss", "mean"),
            candidate_loss=("candidate_integrated_row_loss", "mean"),
            row_count=("loss_improvement", "size"),
        )
        .reset_index(drop=True)
    )
    episodes["loss_improvement"] = (
        episodes["baseline_loss"] - episodes["candidate_loss"]
    )
    games = (
        episodes.groupby("game_id", sort=True, as_index=False)
        .agg(
            baseline_loss=("baseline_loss", "mean"),
            candidate_loss=("candidate_loss", "mean"),
            episode_count=("atomic_information_episode_id", "nunique"),
            row_count=("row_count", "sum"),
        )
        .reset_index(drop=True)
    )
    games["loss_improvement"] = (
        games["baseline_loss"] - games["candidate_loss"]
    )
    games["hierarchical_weight"] = 1.0
    return episodes, games


def _bootstrap_game_improvements(
    game_losses: pd.DataFrame, *, spec: FrozenSelectionSpec
) -> tuple[float, float, float, float]:
    improvements = game_losses["loss_improvement"].to_numpy(dtype=float)
    if len(improvements) < 2:
        raise ModelSelectionError(
            "paired game bootstrap requires at least two games"
        )
    rng = np.random.default_rng(spec.bootstrap_seed)
    selections = rng.integers(
        0,
        len(improvements),
        size=(spec.bootstrap_samples, len(improvements)),
    )
    draws = improvements[selections].mean(axis=1)
    tail = (1.0 - float(spec.confidence_level)) / 2.0
    low, high = np.quantile(draws, (tail, 1.0 - tail))
    return (
        float(improvements.mean()),
        float(low),
        float(high),
        float(np.mean(draws > 0.0)),
    )


def select_candidate_against_b0(
    model_run: X15ModelRun,
    *,
    spec: FrozenSelectionSpec,
    authority: FrozenDevelopmentAuthority | None = None,
) -> ModelSelectionResult:
    """Apply the pre-locked integrated and clean-anchor intersection gate."""

    if not isinstance(spec, FrozenSelectionSpec):
        raise ModelSelectionError("spec must be a FrozenSelectionSpec")
    work, row_authority = _validate_task4_rows(model_run, spec=spec)
    (
        authority_gate,
        authority_failures,
        authority_game_count,
        observed_fold_ids,
    ) = _development_authority_gate(
        model_run,
        work,
        row_authority_sha256=row_authority,
        authority=authority,
        spec=spec,
    )
    paired = _attach_multihead_losses(
        _merge_exact_pairs(work, spec=spec)
    )
    episode_losses, game_losses = _hierarchical_losses(paired)
    (
        integrated_mean,
        integrated_low,
        integrated_high,
        probability_improved,
    ) = _bootstrap_game_improvements(game_losses, spec=spec)
    integrated_gate = integrated_mean > 0.0 and integrated_low > 0.0

    anchor = paired.loc[
        paired["landmark_seconds"].eq(spec.anchor_landmark_seconds)
        & paired["endpoint_seconds"].eq(spec.anchor_endpoint_seconds)
    ].copy()
    if anchor.empty:
        raise ModelSelectionError(
            "the frozen clean L=3 -> H=30 anchor is missing"
        )
    anchor_episode_losses, anchor_game_losses = _hierarchical_losses(
        anchor
    )
    anchor_game_count = len(anchor_game_losses)
    anchor_episode_count = len(anchor_episode_losses)
    anchor_game_coverage = anchor_game_count / len(game_losses)
    anchor_episode_coverage = anchor_episode_count / len(
        episode_losses
    )
    anchor_supported = anchor_game_count >= MIN_ANCHOR_GAMES
    anchor_mean = float(
        anchor_game_losses["loss_improvement"].mean()
    )
    anchor_sign_reversed = bool(
        np.sign(anchor_mean) * np.sign(integrated_mean) < 0
    )
    anchor_gate = (
        anchor_supported
        and anchor_mean >= 0.0
        and not anchor_sign_reversed
    )
    selected = authority_gate and integrated_gate and anchor_gate
    return ModelSelectionResult(
        spec=spec,
        cohort_authority_sha256=row_authority,
        run_config_sha256=model_run.run_config_sha256,
        schema_version=HISTORICAL_SCHEMA_VERSION,
        analysis_scope=HISTORICAL_ANALYSIS_SCOPE,
        target_contract=str(paired["target_contract"].iloc[0]),
        claim_boundary=str(paired["claim_boundary"].iloc[0]),
        diagnostic_status=(
            "HISTORICAL_SIGNAL_CANDIDATE"
            if selected
            else (
                "PARTIAL_DEVELOPMENT_DIAGNOSTIC_ONLY"
                if not authority_gate
                else "HISTORICAL_SIGNAL_REJECTED"
            )
        ),
        execution_claim_eligible=False,
        tick_claim_eligible=False,
        continuity_claim_eligible=False,
        authority_gate_passed=authority_gate,
        authority_gate_failures=authority_failures,
        development_authority_game_count=authority_game_count,
        observed_fold_ids=observed_fold_ids,
        paired_rows=paired.reset_index(drop=True),
        episode_losses=episode_losses.reset_index(drop=True),
        game_losses=game_losses.reset_index(drop=True),
        integrated_mean_improvement=integrated_mean,
        integrated_ci_low=integrated_low,
        integrated_ci_high=integrated_high,
        bootstrap_probability_improved=probability_improved,
        bootstrap_samples=spec.bootstrap_samples,
        bootstrap_seed=spec.bootstrap_seed,
        integrated_gate_passed=integrated_gate,
        anchor_game_losses=anchor_game_losses.reset_index(drop=True),
        anchor_game_count=anchor_game_count,
        anchor_episode_count=anchor_episode_count,
        anchor_game_coverage=float(anchor_game_coverage),
        anchor_episode_coverage=float(anchor_episode_coverage),
        anchor_support_status=(
            "SUPPORTED"
            if anchor_supported
            else "INSUFFICIENT_SUPPORT"
        ),
        anchor_mean_improvement=anchor_mean,
        anchor_sign_reversed=anchor_sign_reversed,
        anchor_gate_passed=anchor_gate,
        loss_improvement_sign_semantics=(
            spec.loss_improvement_sign_semantics
        ),
        selected=selected,
    )


def _stage_b_projection_identity(
    model_run: X15ModelRun,
) -> tuple[tuple[str, str], str]:
    if not isinstance(model_run, X15ModelRun):
        raise ModelSelectionError(
            "Stage-B selection requires X15ModelRun projections"
        )
    if not isinstance(model_run.run_config, Mapping):
        raise ModelSelectionError(
            "Stage-B selection requires stamped projection run_config"
        )
    _require_sha256(
        model_run.run_config_sha256, field="run_config_sha256"
    )
    if (
        _canonical_sha256(model_run.run_config)
        != model_run.run_config_sha256
    ):
        raise ModelSelectionError("Stage-B run_config SHA-256 mismatch")
    if (
        model_run.run_config.get("schema_version")
        != HISTORICAL_SCHEMA_VERSION
    ):
        raise ModelSelectionError(
            "Stage-B selection requires "
            "HistoricalTradesOnlyProbabilityPanelV2"
        )
    if (
        model_run.run_config.get("survival_probability_contract")
        != SURVIVAL_PROBABILITY_CONTRACT
    ):
        raise ModelSelectionError(
            "Stage-B selection requires "
            "survival_probability_contract="
            "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
        )

    model_ids = _tuple_value(model_run.run_config.get("model_ids"))
    feature_block_ids = _tuple_value(
        model_run.run_config.get("feature_block_ids")
    )
    if (
        model_ids is None
        or len(model_ids) != 2
        or model_ids[0] != BASELINE_MODEL_ID
        or feature_block_ids is None
        or len(feature_block_ids) != 2
        or feature_block_ids[0] != BASELINE_FEATURE_BLOCK_ID
    ):
        raise ModelSelectionError(
            "Stage-B projection run_config identity must be "
            "baseline plus one candidate"
        )
    candidate = (str(model_ids[1]), str(feature_block_ids[1]))
    if candidate not in STAGE_B_CANDIDATE_SUITE:
        raise ModelSelectionError(
            "Stage-B projection run_config identity is not selectable"
        )
    if model_ids != (BASELINE_MODEL_ID, candidate[0]) or (
        feature_block_ids
        != (BASELINE_FEATURE_BLOCK_ID, candidate[1])
    ):
        raise ModelSelectionError(
            "Stage-B projection run_config identity is malformed"
        )

    predictions = model_run.oof_predictions
    required = {
        "model_id",
        "feature_block_id",
        "venue",
        "training_venue",
    }
    if (
        not isinstance(predictions, pd.DataFrame)
        or predictions.empty
        or not required.issubset(predictions.columns)
    ):
        raise ModelSelectionError(
            "Stage-B selection requires identified OOF prediction rows"
        )
    if (
        not predictions["venue"].eq("polymarket").all()
        or not predictions["training_venue"].eq("polymarket").all()
    ):
        raise ModelSelectionError(
            "Stage-B selection requires all 8 frozen native Polymarket "
            "projection candidates"
        )
    observed_pairs = set(
        zip(
            predictions["model_id"].astype(str),
            predictions["feature_block_id"].astype(str),
            strict=True,
        )
    )
    expected_pairs = {
        (BASELINE_MODEL_ID, BASELINE_FEATURE_BLOCK_ID),
        candidate,
    }
    if observed_pairs != expected_pairs:
        raise ModelSelectionError(
            "Stage-B projection rows must contain exactly baseline "
            "plus the configured candidate"
        )

    fold_ids = _tuple_value(model_run.run_config.get("fold_ids"))
    expected_fold_ids = tuple(fold[0] for fold in EXPECTED_FOLDS)
    source_shards = model_run.run_config.get("source_shards")
    if (
        fold_ids != expected_fold_ids
        or not isinstance(source_shards, (tuple, list))
    ):
        raise ModelSelectionError(
            "Stage-B projection source_shards require all five folds"
        )
    observed_cells: list[tuple[str, str, str]] = []
    for shard in source_shards:
        if not isinstance(shard, Mapping):
            raise ModelSelectionError(
                "Stage-B projection source_shards are malformed"
            )
        cell = (
            str(shard.get("fold_id", "")),
            str(shard.get("model_id", "")),
            str(shard.get("feature_block_id", "")),
        )
        observed_cells.append(cell)
        for field in (
            "manifest_sha256",
            "shard_bundle_sha256",
            "oof_predictions_sha256",
        ):
            _require_sha256(
                shard.get(field),
                field=f"source_shards.{field}",
            )
    expected_cells = tuple(
        (fold_id, model_id, feature_block_id)
        for fold_id in expected_fold_ids
        for model_id, feature_block_id in (
            (BASELINE_MODEL_ID, BASELINE_FEATURE_BLOCK_ID),
            candidate,
        )
    )
    if (
        len(set(observed_cells)) != len(observed_cells)
        or tuple(observed_cells) != expected_cells
    ):
        raise ModelSelectionError(
            "Stage-B projection source_shards contain missing, "
            "duplicate, or wrong execution identities"
        )
    for field in (
        "source_batch_manifest_sha256",
        "source_batch_sha256",
    ):
        _require_sha256(
            model_run.run_config.get(field),
            field=f"run_config.{field}",
        )
    source_batch_path = model_run.run_config.get(
        "source_batch_manifest_path"
    )
    if not isinstance(source_batch_path, str) or not source_batch_path:
        raise ModelSelectionError(
            "Stage-B projection requires source_batch_manifest_path"
        )

    shared_config = dict(model_run.run_config)
    for field in ("model_ids", "feature_block_ids", "source_shards"):
        shared_config.pop(field, None)
    return candidate, _canonical_sha256(shared_config)


def _validate_stage_b_candidate_suite(
    model_runs: Sequence[X15ModelRun],
) -> tuple[
    tuple[X15ModelRun, ...],
    str,
    str,
]:
    if isinstance(model_runs, (str, bytes)) or not isinstance(
        model_runs, Sequence
    ):
        raise ModelSelectionError(
            "Stage-B selection requires 8 projection runs"
        )
    if len(model_runs) != len(STAGE_B_CANDIDATE_SUITE):
        raise ModelSelectionError(
            "Stage-B selection requires all 8 frozen Polymarket "
            "candidate projections"
        )
    by_candidate: dict[tuple[str, str], X15ModelRun] = {}
    shared_hashes: set[str] = set()
    for model_run in model_runs:
        candidate, shared_hash = _stage_b_projection_identity(model_run)
        if candidate in by_candidate:
            raise ModelSelectionError(
                f"Stage-B duplicate candidate projection: {candidate}"
            )
        by_candidate[candidate] = model_run
        shared_hashes.add(shared_hash)
    missing = [
        candidate
        for candidate in STAGE_B_CANDIDATE_SUITE
        if candidate not in by_candidate
    ]
    if missing:
        raise ModelSelectionError(
            "Stage-B selection requires all 8 frozen Polymarket "
            f"candidate projections; missing={missing}"
        )
    if len(shared_hashes) != 1:
        raise ModelSelectionError(
            "all Stage-B projection runs must share one run_config"
        )
    ordered_runs = tuple(
        by_candidate[candidate]
        for candidate in STAGE_B_CANDIDATE_SUITE
    )
    suite_run_config_sha256 = _canonical_sha256(
        tuple(
            {
                "candidate": candidate,
                "run_config_sha256": model_run.run_config_sha256,
            }
            for candidate, model_run in zip(
                STAGE_B_CANDIDATE_SUITE,
                ordered_runs,
                strict=True,
            )
        )
    )
    return (
        ordered_runs,
        next(iter(shared_hashes)),
        suite_run_config_sha256,
    )


def _game_standard_error(result: ModelSelectionResult) -> float:
    improvements = result.game_losses["loss_improvement"].to_numpy(
        dtype=float
    )
    if (
        len(improvements) < 2
        or not np.isfinite(improvements).all()
    ):
        raise ModelSelectionError(
            "one-standard-error selection requires at least two finite "
            "per-game loss improvements"
        )
    return float(
        np.std(improvements, ddof=1) / np.sqrt(len(improvements))
    )


def select_stage_b_v2_winner(
    model_runs: Sequence[X15ModelRun],
    *,
    authority: FrozenDevelopmentAuthority | None = None,
) -> StageBModelSelectionResult:
    """Select the frozen Stage-B V2 suite with the one-SE winner rule."""

    (
        ordered_runs,
        shared_run_config_sha256,
        suite_run_config_sha256,
    ) = _validate_stage_b_candidate_suite(model_runs)
    results = tuple(
        select_candidate_against_b0(
            model_run,
            spec=FrozenSelectionSpec(
                candidate_model_id=model_id,
                candidate_feature_block_id=feature_block_id,
            ),
            authority=authority,
        )
        for model_run, (model_id, feature_block_id) in zip(
            ordered_runs,
            STAGE_B_CANDIDATE_SUITE,
            strict=True,
        )
    )
    expected_authority = results[0].cohort_authority_sha256
    if any(
        result.cohort_authority_sha256 != expected_authority
        or result.schema_version != HISTORICAL_SCHEMA_VERSION
        or result.run_config_sha256 != model_run.run_config_sha256
        for result, model_run in zip(
            results, ordered_runs, strict=True
        )
    ):
        raise ModelSelectionError(
            "all Stage-B candidate results must share authority and "
            "run_config"
        )
    standard_errors = tuple(
        _game_standard_error(result) for result in results
    )
    passed = [
        (index, result)
        for index, result in enumerate(results)
        if result.selected
    ]
    winner: ModelSelectionResult | None = None
    best_mean: float | None = None
    best_standard_error: float | None = None
    threshold: float | None = None
    within_one_se = [False] * len(results)
    if passed:
        best_index, best = max(
            passed,
            key=lambda item: item[1].integrated_mean_improvement,
        )
        best_mean = float(best.integrated_mean_improvement)
        best_standard_error = standard_errors[best_index]
        threshold = best_mean - best_standard_error
        for index, result in passed:
            within_one_se[index] = (
                result.integrated_mean_improvement >= threshold
            )
        winner_index = next(
            index
            for index, eligible in enumerate(within_one_se)
            if eligible
        )
        winner = results[winner_index]

    audit_rows = []
    for index, (result, standard_error) in enumerate(
        zip(results, standard_errors, strict=True)
    ):
        audit_rows.append(
            {
                "candidate_model_id": result.spec.candidate_model_id,
                "candidate_feature_block_id": (
                    result.spec.candidate_feature_block_id
                ),
                "baseline_model_id": result.spec.baseline_model_id,
                "baseline_feature_block_id": (
                    result.spec.baseline_feature_block_id
                ),
                "gate_selected": result.selected,
                "diagnostic_status": result.diagnostic_status,
                "authority_gate_passed": (
                    result.authority_gate_passed
                ),
                "authority_gate_failures": (
                    result.authority_gate_failures
                ),
                "integrated_mean_improvement": (
                    result.integrated_mean_improvement
                ),
                "integrated_ci_low": result.integrated_ci_low,
                "integrated_ci_high": result.integrated_ci_high,
                "integrated_gate_passed": (
                    result.integrated_gate_passed
                ),
                "anchor_game_count": result.anchor_game_count,
                "anchor_mean_improvement": (
                    result.anchor_mean_improvement
                ),
                "anchor_sign_reversed": (
                    result.anchor_sign_reversed
                ),
                "anchor_gate_passed": result.anchor_gate_passed,
                "game_count": len(result.game_losses),
                "standard_error": standard_error,
                "within_one_se": within_one_se[index],
                "complexity_rank": index + 1,
                "final_winner": result is winner,
                "cohort_authority_sha256": (
                    result.cohort_authority_sha256
                ),
                "run_config_sha256": result.run_config_sha256,
                "shared_run_config_sha256": (
                    shared_run_config_sha256
                ),
                "suite_run_config_sha256": suite_run_config_sha256,
                "schema_version": result.schema_version,
                "survival_probability_contract": (
                    SURVIVAL_PROBABILITY_CONTRACT
                ),
                "winner_rule_sha256": STAGE_B_WINNER_RULE_SHA256,
            }
        )
    candidate_audit_sha256 = _canonical_sha256(audit_rows)
    return StageBModelSelectionResult(
        decision_status=(
            "MODEL_ADVANCE" if winner is not None else "NO_MODEL_ADVANCE"
        ),
        winner=winner,
        candidate_results=results,
        candidate_audit=pd.DataFrame(audit_rows),
        best_integrated_mean_improvement=best_mean,
        best_standard_error=best_standard_error,
        one_se_threshold=threshold,
        cohort_authority_sha256=expected_authority,
        shared_run_config_sha256=shared_run_config_sha256,
        suite_run_config_sha256=suite_run_config_sha256,
        candidate_audit_sha256=candidate_audit_sha256,
        schema_version=HISTORICAL_SCHEMA_VERSION,
        survival_probability_contract=SURVIVAL_PROBABILITY_CONTRACT,
        winner_rule_sha256=STAGE_B_WINNER_RULE_SHA256,
    )


def build_factor_claim_audit(
    selection: ModelSelectionResult,
    *,
    factor_membership: pd.DataFrame,
    min_support_games: int = 30,
    min_support_episodes: int = 20,
) -> pd.DataFrame:
    """Publish clean-anchor factor-family BH, LOO, direction, and support.

    Factor membership is joined only after the model has been selected.  It
    therefore cannot multiply training or selection weights.  The clean
    ``L=3 -> H=30`` row is the named-claim surface; full horizon paths remain
    secondary trajectories in ``selection.paired_rows``.
    """

    if not isinstance(selection, ModelSelectionResult):
        raise ModelSelectionError(
            "selection must be a ModelSelectionResult"
        )
    if (
        isinstance(min_support_games, bool)
        or not isinstance(min_support_games, Integral)
        or min_support_games <= 0
        or isinstance(min_support_episodes, bool)
        or not isinstance(min_support_episodes, Integral)
        or min_support_episodes <= 0
    ):
        raise ModelSelectionError(
            "factor support minima must be positive integers"
        )
    required = {
        "game_id",
        "atomic_information_episode_id",
        "factor_id",
        "factor_version",
    }
    if not isinstance(factor_membership, pd.DataFrame):
        raise ModelSelectionError(
            "factor_membership must be a DataFrame"
        )
    missing = sorted(required.difference(factor_membership.columns))
    if missing:
        raise ModelSelectionError(
            f"factor_membership missing required columns: {missing}"
        )
    membership = factor_membership.loc[:, sorted(required)].copy()
    for column in required:
        if not membership[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ModelSelectionError(
                f"factor_membership {column} must be a nonempty string"
            )
    if membership.duplicated(
        [
            "game_id",
            "atomic_information_episode_id",
            "factor_id",
            "factor_version",
        ],
        keep=False,
    ).any():
        raise ModelSelectionError(
            "factor_membership contains duplicate episode tags"
        )
    if (
        membership.groupby("factor_id")["factor_version"].nunique() > 1
    ).any():
        raise ModelSelectionError(
            "each factor_id must bind exactly one factor_version"
        )

    anchor = selection.paired_rows.loc[
        selection.paired_rows["landmark_seconds"].eq(
            selection.spec.anchor_landmark_seconds
        )
        & selection.paired_rows["endpoint_seconds"].eq(
            selection.spec.anchor_endpoint_seconds
        ),
        [
            "game_id",
            "atomic_information_episode_id",
            "venue",
            "loss_improvement",
            "direction_truth",
        ],
    ].copy()
    if anchor.duplicated(
        ["game_id", "atomic_information_episode_id"], keep=False
    ).any():
        raise ModelSelectionError(
            "clean-anchor factor claims require one row per episode"
        )
    joined = membership.merge(
        anchor,
        on=["game_id", "atomic_information_episode_id"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise ModelSelectionError(
            "factor_membership has no clean-anchor episodes"
        )
    membership_without_anchor = len(membership) - len(joined)
    equal_game_units = (
        joined.groupby(
            ["factor_id", "venue", "game_id"],
            sort=True,
            as_index=False,
        )
        .agg(gross_markout=("loss_improvement", "mean"))
        .reset_index(drop=True)
    )
    signals = pd.DataFrame(
        {
            "factor_id": equal_game_units["factor_id"],
            "venue": equal_game_units["venue"],
            "model_id": selection.spec.candidate_model_id,
            "game_id": equal_game_units["game_id"],
            "episode_id": equal_game_units["game_id"].map(
                lambda game_id: f"{game_id}:equal-game-effect-unit"
            ),
            "gross_markout": equal_game_units[
                "gross_markout"
            ].astype(float),
            "evaluation_status": "EVALUATED",
            "dataset_partition": "development",
        }
    )
    try:
        summary = summarize_development_signals(
            signals,
            seed=BOOTSTRAP_SEED,
            n_bootstrap=BOOTSTRAP_SAMPLES,
            confidence_level=0.95,
            min_support_games=int(min_support_games),
            min_support_signals=int(min_support_games),
        )
    except X15StatisticsInputError as exc:
        raise ModelSelectionError(
            f"factor claim audit is invalid: {exc}"
        ) from exc

    version = (
        membership[["factor_id", "factor_version"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    direction = (
        joined.assign(
            up_episode=joined["direction_truth"].eq("UP").astype(int),
            down_episode=joined["direction_truth"].eq("DOWN").astype(int),
            no_move_episode=joined["direction_truth"]
            .eq("NO_MOVE")
            .astype(int),
        )
        .groupby(["factor_id", "venue"], sort=True, as_index=False)
        .agg(
            distinct_games=("game_id", "nunique"),
            unique_episodes=(
                "atomic_information_episode_id",
                "nunique",
            ),
            up_episode_count=("up_episode", "sum"),
            down_episode_count=("down_episode", "sum"),
            no_move_episode_count=("no_move_episode", "sum"),
        )
    )
    audit = (
        summary.merge(version, on="factor_id", validate="many_to_one")
        .merge(
            direction,
            on=["factor_id", "venue"],
            validate="one_to_one",
        )
        .rename(
            columns={
                "support_signals": "equal_game_effect_unit_count"
            }
        )
    )
    audit["support_episodes"] = audit["unique_episodes"].astype(int)
    audit["support_status"] = np.where(
        audit["support_games"].ge(int(min_support_games))
        & audit["support_episodes"].ge(int(min_support_episodes)),
        "SUPPORTED",
        "INSUFFICIENT_SUPPORT",
    )
    audit["claim_landmark_seconds"] = ANCHOR_LANDMARK_SECONDS
    audit["claim_endpoint_seconds"] = ANCHOR_ENDPOINT_SECONDS
    audit["loss_improvement_sign_semantics"] = (
        LOSS_IMPROVEMENT_SIGN_SEMANTICS
    )
    audit["target_contract"] = selection.target_contract
    audit["claim_boundary"] = selection.claim_boundary
    audit["schema_version"] = selection.schema_version
    audit["analysis_scope"] = selection.analysis_scope
    audit["claim_eligible"] = False
    audit["bh_q_gate_passed"] = audit["bh_q_value"].le(
        DEFAULT_MAX_Q_VALUE
    )
    audit["loo_sign_gate_passed"] = audit[
        "leave_one_game_out_same_sign_rate"
    ].ge(DEFAULT_MIN_LOO_SAME_SIGN)
    audit["max_game_contribution_gate_passed"] = audit[
        "max_single_game_absolute_contribution_ratio"
    ].le(DEFAULT_MAX_GAME_CONTRIBUTION)
    audit["registered_gate_passed"] = (
        selection.selected
        & audit["support_status"].eq("SUPPORTED")
        & audit["recommended_for_gating"]
        & audit["mean_ci_excludes_zero"]
        & audit["bh_q_gate_passed"]
        & audit["loo_sign_gate_passed"]
        & audit["max_game_contribution_gate_passed"]
        & audit["individual_statistical_gate"]
        & audit["passes_development_gate"]
    )
    review_only = (
        selection.selected
        & audit["support_status"].eq("SUPPORTED")
        & audit["individual_statistical_gate"]
        & ~audit["passes_development_gate"]
    )
    audit["diagnostic_status"] = np.where(
        audit["registered_gate_passed"],
        "HISTORICAL_SIGNAL_CANDIDATE",
        np.where(
            review_only,
            "HISTORICAL_SIGNAL_REVIEW_ONLY",
            "HISTORICAL_SIGNAL_REJECTED",
        ),
    )
    audit["execution_claim_eligible"] = False
    audit["tick_claim_eligible"] = False
    audit["continuity_claim_eligible"] = False
    audit["membership_without_clean_anchor_count"] = (
        membership_without_anchor
    )
    return audit.sort_values(
        ["factor_family", "factor_id", "venue"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "ANCHOR_ENDPOINT_SECONDS",
    "ANCHOR_LANDMARK_SECONDS",
    "BASELINE_FEATURE_BLOCK_ID",
    "BASELINE_MODEL_ID",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "DIRECTION_THRESHOLD_PROBABILITY",
    "DIRECTION_THRESHOLD_SEMANTICS",
    "EXPECTED_DEVELOPMENT_GAME_COUNT",
    "EXPECTED_FOLDS",
    "FrozenDevelopmentAuthority",
    "MARKET_CONTINUITY_SUPPORT",
    "FrozenSelectionSpec",
    "HISTORICAL_CLAIM_BOUNDARY",
    "HISTORICAL_ANALYSIS_SCOPE",
    "HISTORICAL_SCHEMA_VERSION",
    "HISTORICAL_TARGET_CONTRACT",
    "LOSS_IMPROVEMENT_SIGN_SEMANTICS",
    "ModelSelectionError",
    "ModelSelectionResult",
    "STAGE_B_CANDIDATE_SUITE",
    "STAGE_B_WINNER_RULE_SHA256",
    "StageBModelSelectionResult",
    "bind_frozen_development_authority",
    "VENUE_TICK_SUPPORT",
    "build_factor_claim_audit",
    "select_candidate_against_b0",
    "select_stage_b_v2_winner",
]
