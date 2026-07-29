"""Governed Stage B OOF models for the NFL VenueReactionPanelV3.

All preprocessing, model fitting, and calibration happens inside complete-game
chronological folds.  The three probability heads use separate risk sets:
survival S_H, observation O_H conditional on S_H, and direction conditional on
S_H & O_H.  Missing endpoint trades remain unavailable outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from prediction_market.research.nfl_x15_calibration import (
    DIRECTION_CLASSES,
    RAW_UNCALIBRATED,
    BinaryPlattCalibrator,
    DirectionTemperatureCalibrator,
    fit_binary_platt,
    fit_direction_temperature,
)
from prediction_market.research.nfl_x15_distribution import (
    DIRECTION_CONDITIONS,
    INSUFFICIENT_SUPPORT,
    PRIMARY_QUANTILES,
    SUPPORTED,
    QuantileSupportContract,
    fit_directional_quantiles,
    magnitude_distribution_metrics,
)
from prediction_market.research.nfl_x15_development_panel import (
    VerifiedDiagnosticPanelPartition,
)
from prediction_market.research.nfl_x15_landmarks import ENDPOINT_SECONDS


RANDOM_STATE: Final[int] = 20260728
SURVIVAL_PROBABILITY_CONTRACT: Final[str] = (
    "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
)
EFFECTIVE_SEED_CONTRACT_ID: Final[str] = (
    "SHA256_COORDINATE_UINT31_V1"
)
EFFECTIVE_SEED_COORDINATE_FIELDS: Final[tuple[str, ...]] = (
    "base_random_state",
    "fold_id",
    "training_venue",
    "evaluation_venue",
    "transport_mode",
    "feature_block_id",
    "model_id",
    "purpose",
)
EFFECTIVE_SEED_MODULUS: Final[int] = 2**31 - 1
MODEL_IDS: Final[tuple[str, ...]] = (
    "b0_empirical_v1",
    "regularized_logistic_v1",
    "shallow_xgboost_v1",
)
QUANTILE_MODEL_ID: Final[str] = "directional_quantile_xgboost_v1"
FEATURE_BLOCKS: Final[Mapping[str, tuple[str, ...]]] = {
    "B0": ("landmark_seconds", "endpoint_seconds"),
    "B1": (
        "landmark_seconds",
        "endpoint_seconds",
        "tick_size",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
    ),
    "B2": (
        "landmark_seconds",
        "endpoint_seconds",
        "tick_size",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
        "fact__source_resolution",
        "fact__game_seconds_remaining",
        "fact__score_margin_home",
        "fact__possession_is_home",
        "fact__down",
        "fact__distance",
        "fact__yardline_100",
    ),
    "B3": (
        "landmark_seconds",
        "endpoint_seconds",
        "tick_size",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
        "fact__source_resolution",
        "fact__game_seconds_remaining",
        "fact__score_margin_home",
        "fact__possession_is_home",
        "fact__down",
        "fact__distance",
        "fact__yardline_100",
        "fact__primary_action",
        "fact__yards_gained",
        "fact__return_yards",
        "fact__actor_is_home",
        "fact__beneficiary_is_home",
        "multi_hot__*",
    ),
    "B4": (
        "landmark_seconds",
        "endpoint_seconds",
        "tick_size",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
        "fact__source_resolution",
        "fact__game_seconds_remaining",
        "fact__score_margin_home",
        "fact__possession_is_home",
        "fact__down",
        "fact__distance",
        "fact__yardline_100",
        "fact__primary_action",
        "fact__yards_gained",
        "fact__return_yards",
        "fact__actor_is_home",
        "fact__beneficiary_is_home",
        "multi_hot__*",
        "stage_a_status",
        "p_before_home",
        "p_after_home",
        "reference_delta_home",
        "reference_gap_at_landmark",
    ),
}
DIAGNOSTIC_SCHEMA_VERSION: Final[str] = (
    "HistoricalTradesOnlyProbabilityPanelV2"
)
DIAGNOSTIC_TARGET_CONTRACT: Final[str] = (
    "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
)
DIAGNOSTIC_CLAIM_BOUNDARY: Final[str] = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC"
)
DIAGNOSTIC_ANALYSIS_SCOPE: Final[str] = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_DIAGNOSTIC"
)
DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY: Final[float] = 0.01
DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS: Final[str] = (
    "FIXED_CROSS_VENUE_RESEARCH_MATERIALITY_NOT_TICK"
)
DIAGNOSTIC_VENUE_TICK_SUPPORT: Final[str] = "UNSUPPORTED"
DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT: Final[str] = "UNKNOWN"
DIAGNOSTIC_FEATURE_BLOCKS: Final[Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "D0": FEATURE_BLOCKS["B0"],
            "D1": tuple(
                feature
                for feature in FEATURE_BLOCKS["B1"]
                if feature != "tick_size"
            ),
            "D2": tuple(
                feature
                for feature in FEATURE_BLOCKS["B2"]
                if feature != "tick_size"
            ),
            "D3": tuple(
                feature
                for feature in FEATURE_BLOCKS["B3"]
                if feature != "tick_size"
            ),
            "D4": tuple(
                feature
                for feature in FEATURE_BLOCKS["B4"]
                if feature != "tick_size"
            ),
        }
    )
)

_FOLD_WEEK_WINDOWS: Final[
    tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
] = (
    ((1, 2), (3, 4)),
    (tuple(range(1, 5)), (5, 6)),
    (tuple(range(1, 7)), (7, 8)),
    (tuple(range(1, 9)), (9, 10)),
    (tuple(range(1, 11)), (11, 12)),
)
_PANEL_REQUIRED: Final[set[str]] = {
    "schema_version",
    "cohort_authority_sha256",
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "nfl_week",
    "landmark_seconds",
    "endpoint_seconds",
    "decision_eligible",
    "target_eligible",
    "s_h",
    "o_h_given_s",
    "direction",
    "conditional_magnitude",
    "decision_features_json",
    "decision_feature_sha256",
}
DIAGNOSTIC_MODEL_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "target_contract",
    "claim_boundary",
    "cohort_authority_sha256",
    "game_id",
    "nfl_week",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
    "decision_eligible",
    "target_eligible",
    "sports_clean_l",
    "sports_clean_l_reason",
    "sports_clean_h",
    "sports_clean_reason",
    "actual_trade_observed_h",
    "delta_l_h",
    "direction",
    "conditional_magnitude",
    "direction_threshold_probability",
    "direction_threshold_semantics",
    "venue_tick_support",
    "market_continuity_support",
    "decision_features_json",
    "decision_feature_sha256",
)
_PANEL_GRAIN: Final[tuple[str, ...]] = (
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
)
_DECISION_LEAKAGE_KEYS: Final[set[str]] = {
    "mark_h_trade_id",
    "mark_h_trade_ids_json",
    "mark_h_trade_id_set_sha256",
    "mark_h_observation_count",
    "mark_h_observed_size",
    "mark_h_semantics",
    "mark_h_source_time_utc",
    "mark_h_price",
    "mark_h_staleness_seconds",
    "s_h",
    "o_h_given_s",
    "target_eligible",
    "delta_l_h",
    "direction",
    "conditional_magnitude",
    "abnormal_move",
    "reference_status",
    "final_outcome",
    "final_home_win",
    "settlement_value",
}
_BINARY_MIN_GAMES: Final[int] = 2
_DIRECTION_MIN_GAMES: Final[int] = 3
_PROBABILITY_EPSILON: Final[float] = 1e-6
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")
_XGB_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 16,
    "max_depth": 2,
    "learning_rate": 0.08,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": 1,
}


class X15ModelInputError(ValueError):
    """The panel or execution slice violates the frozen Stage B contract."""


@dataclass(frozen=True, slots=True)
class X15WeekFold:
    fold_id: str
    train_weeks: tuple[int, ...]
    validation_weeks: tuple[int, ...]
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class X15ModelRun:
    oof_predictions: pd.DataFrame
    conditional_quantiles: pd.DataFrame
    fold_metrics: pd.DataFrame
    support_audit: pd.DataFrame
    weight_audit: pd.DataFrame
    run_config_sha256: str
    run_config: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class X15PreparedDiagnosticPanel:
    """Compact reusable decision rows prepared from a verified batch iterator."""

    frame: pd.DataFrame
    parsed_by_sha256: Mapping[str, Mapping[str, object]]
    game_ids: tuple[str, ...]
    partition_count: int
    source_row_count: int
    batch_manifest_file_sha256: str
    batch_sha256: str
    cohort_authority_sha256: str
    cohort_mapping_sha256: str


@dataclass(slots=True)
class _FittedHead:
    model_id: str
    head: str
    status: str
    reason: str
    estimator: object | None
    training_sha256: str
    classes: tuple[str, ...] | tuple[int, ...]


@dataclass(slots=True)
class _FeatureBundle:
    transformer: ColumnTransformer
    train_matrix: np.ndarray
    validation_matrix: np.ndarray
    columns: tuple[str, ...]
    training_sha256: str


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        children = list(value)
        if isinstance(value, set):
            children = sorted(children, key=repr)
        return [_canonical(child) for child in children]
    missing = pd.isna(value)
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


def effective_random_state(
    *,
    base_random_state: int,
    fold_id: str,
    training_venue: str,
    evaluation_venue: str,
    transport_mode: str,
    feature_block_id: str,
    model_id: str,
    purpose: str,
) -> int:
    """Derive a request-order-independent seed from one execution identity."""

    coordinate = {
        "contract_id": EFFECTIVE_SEED_CONTRACT_ID,
        "base_random_state": int(base_random_state),
        "fold_id": str(fold_id),
        "training_venue": str(training_venue),
        "evaluation_venue": str(evaluation_venue),
        "transport_mode": str(transport_mode),
        "feature_block_id": str(feature_block_id),
        "model_id": str(model_id),
        "purpose": str(purpose),
    }
    digest = hashlib.sha256(
        json.dumps(
            coordinate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % EFFECTIVE_SEED_MODULUS


def build_x15_week_folds(landmarks: pd.DataFrame) -> tuple[X15WeekFold, ...]:
    """Build the five frozen expanding complete-game folds."""

    if not isinstance(landmarks, pd.DataFrame):
        raise X15ModelInputError("landmarks must be a DataFrame")
    missing = {"game_id", "nfl_week"} - set(landmarks.columns)
    if missing:
        raise X15ModelInputError(f"fold input missing columns: {sorted(missing)}")
    if landmarks.empty:
        raise X15ModelInputError("fold input must be nonempty")
    if landmarks["game_id"].isna().any():
        raise X15ModelInputError("game_id must be nonempty")
    unique_games = landmarks["game_id"].drop_duplicates()
    if any(
        type(value) is not str or not value.strip()
        for value in unique_games
    ):
        raise X15ModelInputError("game_id must be nonempty")
    if landmarks["nfl_week"].isna().any():
        raise X15ModelInputError("nfl_week must contain integer weeks")
    unique_weeks = pd.to_numeric(
        landmarks["nfl_week"].drop_duplicates(), errors="coerce"
    )
    if (
        unique_weeks.isna().any()
        or not np.equal(unique_weeks, np.floor(unique_weeks)).all()
    ):
        raise X15ModelInputError("nfl_week must contain integer weeks")
    if not unique_weeks.astype(int).between(1, 12).all():
        raise X15ModelInputError(
            "development folds accept only NFL weeks 1 through 12"
        )
    game_weeks = (
        landmarks.loc[:, ["game_id", "nfl_week"]]
        .drop_duplicates()
    )
    game_weeks["nfl_week"] = pd.to_numeric(
        game_weeks["nfl_week"], errors="raise"
    ).astype(int)
    if game_weeks.groupby("game_id")["nfl_week"].nunique().gt(1).any():
        raise X15ModelInputError("each game_id must map to one NFL week")
    game_weeks = game_weeks.sort_values(
        ["nfl_week", "game_id"], kind="mergesort"
    )
    result: list[X15WeekFold] = []
    for index, (train_weeks, validation_weeks) in enumerate(
        _FOLD_WEEK_WINDOWS, start=1
    ):
        train_ids = tuple(
            game_weeks.loc[
                game_weeks["nfl_week"].isin(train_weeks), "game_id"
            ].sort_values(kind="mergesort")
        )
        validation_ids = tuple(
            game_weeks.loc[
                game_weeks["nfl_week"].isin(validation_weeks), "game_id"
            ].sort_values(kind="mergesort")
        )
        if set(train_ids) & set(validation_ids):
            raise X15ModelInputError("fold train and validation games overlap")
        result.append(
            X15WeekFold(
                fold_id=f"fold_{index:02d}",
                train_weeks=train_weeks,
                validation_weeks=validation_weeks,
                train_game_ids=train_ids,
                validation_game_ids=validation_ids,
            )
        )
    return tuple(result)


def hierarchical_sample_weights(
    frame: pd.DataFrame, eligibility: pd.Series | Sequence[bool]
) -> pd.Series:
    """Give each game one unit, then split by episode and valid L/H rows."""

    required = {"game_id", "atomic_information_episode_id"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise X15ModelInputError(
            "weight input requires game_id and atomic_information_episode_id"
        )
    if frame.index.has_duplicates:
        raise X15ModelInputError("weight input index must be unique")
    if len(eligibility) != len(frame):
        raise X15ModelInputError(
            "weight eligibility requires one value per row"
        )
    try:
        eligible = pd.Series(
            eligibility, index=frame.index, dtype="boolean"
        )
    except (TypeError, ValueError) as error:
        raise X15ModelInputError(
            "weight eligibility must contain booleans"
        ) from error
    if eligible.isna().any():
        raise X15ModelInputError("weight eligibility must not be missing")
    eligible_positions = np.flatnonzero(
        eligible.to_numpy(dtype=bool)
    )
    result = np.zeros(len(frame), dtype=float)
    if not len(eligible_positions):
        return pd.Series(result, index=frame.index, dtype=float)

    selected = frame.iloc[eligible_positions][
        ["game_id", "atomic_information_episode_id"]
    ].reset_index(drop=True)
    if selected.isna().any(axis=None):
        raise X15ModelInputError(
            "eligible weight identity must not be missing"
        )
    selected["_episode_id"] = selected[
        "atomic_information_episode_id"
    ].astype(str)
    episode_counts = selected.groupby(
        "game_id", sort=True, observed=True
    )["_episode_id"].transform("nunique")
    row_counts = selected.groupby(
        ["game_id", "_episode_id"],
        sort=True,
        observed=True,
    )["_episode_id"].transform("size")
    result[eligible_positions] = (
        1.0
        / episode_counts.to_numpy(dtype=float)
        / row_counts.to_numpy(dtype=float)
    )
    return pd.Series(result, index=frame.index, dtype=float)


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def _strict_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.isna().any():
        raise X15ModelInputError(f"{column} must contain nonmissing booleans")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values
    unique = values.drop_duplicates()
    if not all(isinstance(value, (bool, np.bool_)) for value in unique):
        raise X15ModelInputError(f"{column} must contain nonmissing booleans")
    return values.astype(bool)


def _nullable_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype) and not values.isna().any():
        return values
    supplied = values.notna()
    unique = values.loc[supplied].drop_duplicates()
    if not all(isinstance(value, (bool, np.bool_)) for value in unique):
        raise X15ModelInputError(f"{column} must contain booleans or null")
    return values.astype("boolean")


def _annotate_discrete_survival_intervals(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Derive conditional interval-survival targets from cumulative S(L, H)."""

    landmark = pd.to_numeric(frame["landmark_seconds"], errors="coerce")
    endpoint = pd.to_numeric(frame["endpoint_seconds"], errors="coerce")
    if (
        landmark.isna().any()
        or endpoint.isna().any()
        or not np.equal(landmark, np.floor(landmark)).all()
        or not np.equal(endpoint, np.floor(endpoint)).all()
    ):
        raise X15ModelInputError(
            "landmark_seconds and endpoint_seconds must be integers"
        )
    frame["landmark_seconds"] = landmark.astype(int)
    frame["endpoint_seconds"] = endpoint.astype(int)
    frozen_endpoints = tuple(int(value) for value in ENDPOINT_SECONDS)
    frozen_endpoint_set = set(frozen_endpoints)
    if (
        not frame["endpoint_seconds"].isin(frozen_endpoint_set).all()
        or not frame["endpoint_seconds"].gt(frame["landmark_seconds"]).all()
    ):
        raise X15ModelInputError(
            "survival endpoints must be frozen endpoints strictly after L"
        )

    interval_starts: list[int] = []
    for landmark_seconds, endpoint_seconds in frame.loc[
        :, ["landmark_seconds", "endpoint_seconds"]
    ].itertuples(index=False, name=None):
        previous = tuple(
            value for value in frozen_endpoints if value < endpoint_seconds
        )
        previous_endpoint = previous[-1] if previous else landmark_seconds
        interval_starts.append(max(landmark_seconds, previous_endpoint))
    frame["_s_interval_start_seconds"] = np.asarray(
        interval_starts, dtype=int
    )
    frame["_s_interval_at_risk"] = False
    interval_truth = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="boolean",
    )
    path_columns = list(_PANEL_GRAIN[:-1])
    for _, path in frame.groupby(
        path_columns,
        sort=False,
        observed=True,
        dropna=False,
    ):
        ordered = path.sort_values("endpoint_seconds", kind="mergesort")
        by_endpoint = {
            int(value): int(index)
            for index, value in ordered["endpoint_seconds"].items()
        }
        survival_failed = False
        for index, row in ordered.iterrows():
            cumulative_truth = row["s_h"]
            if pd.notna(cumulative_truth):
                if bool(cumulative_truth) and survival_failed:
                    raise X15ModelInputError(
                        "cumulative S_H cannot return true after failure"
                    )
                if not bool(cumulative_truth):
                    survival_failed = True
            interval_start = int(row["_s_interval_start_seconds"])
            landmark_seconds = int(row["landmark_seconds"])
            if interval_start == landmark_seconds:
                at_risk = bool(row["decision_eligible"])
            else:
                prior_index = by_endpoint.get(interval_start)
                if prior_index is None:
                    raise X15ModelInputError(
                        "survival endpoint grid is incomplete before H"
                    )
                prior_truth = frame.at[prior_index, "s_h"]
                at_risk = bool(
                    row["decision_eligible"]
                    and pd.notna(prior_truth)
                    and bool(prior_truth)
                )
            frame.at[index, "_s_interval_at_risk"] = at_risk
            if at_risk and pd.notna(cumulative_truth):
                interval_truth.at[index] = bool(cumulative_truth)
    frame["_s_interval_at_risk"] = frame[
        "_s_interval_at_risk"
    ].astype(bool)
    frame["_s_interval_truth"] = interval_truth
    return frame


def _cumulative_survival_product(
    frame: pd.DataFrame,
    interval_probabilities: np.ndarray,
) -> np.ndarray:
    """Multiply ordered interval-survival probabilities along each L path."""

    probabilities = np.asarray(interval_probabilities, dtype=float)
    if probabilities.shape != (len(frame),):
        raise X15ModelInputError(
            "interval survival probabilities must align with validation rows"
        )
    finite = np.isfinite(probabilities)
    if (
        (probabilities[finite] < 0).any()
        or (probabilities[finite] > 1).any()
    ):
        raise X15ModelInputError(
            "interval survival probabilities must lie in [0, 1]"
        )
    result = np.full(len(frame), np.nan, dtype=float)
    work = frame.loc[
        :, [*list(_PANEL_GRAIN[:-1]), "endpoint_seconds"]
    ].copy()
    work["_position"] = np.arange(len(work), dtype=int)
    for _, path in work.groupby(
        list(_PANEL_GRAIN[:-1]),
        sort=False,
        observed=True,
        dropna=False,
    ):
        running = 1.0
        available = True
        ordered = path.sort_values(
            "endpoint_seconds", kind="mergesort"
        )
        for position in ordered["_position"].to_numpy(dtype=int):
            probability = probabilities[position]
            if not np.isfinite(probability):
                available = False
            if available:
                running *= float(probability)
                result[position] = running
    return result


def _panel_frame(panel: object, *, copy_input: bool = True) -> pd.DataFrame:
    if isinstance(panel, pd.DataFrame):
        frame = panel.copy(deep=True) if copy_input else panel
    elif hasattr(panel, "panel") and isinstance(panel.panel, pd.DataFrame):
        frame = panel.panel.copy(deep=True)
    else:
        raise X15ModelInputError(
            "input must be VenueReactionPanelV3 or its panel DataFrame"
        )
    missing = _PANEL_REQUIRED - set(frame.columns)
    if missing:
        raise X15ModelInputError(f"VenueReactionPanelV3 missing: {sorted(missing)}")
    if frame.empty:
        raise X15ModelInputError("VenueReactionPanelV3 must be nonempty")
    if not frame["schema_version"].eq("VenueReactionPanelV3").all():
        raise X15ModelInputError("schema_version must be VenueReactionPanelV3")
    authorities = frame["cohort_authority_sha256"]
    unique_authorities = authorities.drop_duplicates()
    if authorities.isna().any() or not all(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None
        for value in unique_authorities
    ):
        raise X15ModelInputError(
            "cohort_authority_sha256 must contain sha256 digests"
        )
    if authorities.nunique() != 1:
        raise X15ModelInputError(
            "one cohort authority sha256 is required per Stage B run"
        )
    if frame.duplicated(list(_PANEL_GRAIN)).any():
        raise X15ModelInputError("VenueReactionPanelV3 grain is not unique")
    for column in (
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
    ):
        values = frame[column]
        unique_values = values.drop_duplicates()
        if values.isna().any() or any(
            type(value) is not str or not value.strip()
            for value in unique_values
        ):
            raise X15ModelInputError(f"{column} must be nonempty")
    frame["decision_eligible"] = _strict_bool(frame, "decision_eligible")
    frame["target_eligible"] = _strict_bool(frame, "target_eligible")
    frame["s_h"] = _nullable_bool(frame, "s_h")
    frame["o_h_given_s"] = _nullable_bool(frame, "o_h_given_s")
    frame["o_h_given_s"] = frame["o_h_given_s"].where(
        frame["s_h"].eq(True), pd.NA
    ).astype("boolean")
    frame["_direction_truth"] = frame["direction"].where(
        frame["direction"].isin(DIRECTION_CLASSES), pd.NA
    )
    target = frame["target_eligible"]
    if (
        (~frame.loc[target, "s_h"].fillna(False)).any()
        or (~frame.loc[target, "o_h_given_s"].fillna(False)).any()
        or frame.loc[target, "_direction_truth"].isna().any()
    ):
        raise X15ModelInputError(
            "target-eligible rows require S_H, O_H, and a direction class"
        )
    magnitude = pd.to_numeric(frame["conditional_magnitude"], errors="coerce")
    moving = target & frame["_direction_truth"].isin(DIRECTION_CONDITIONS)
    if magnitude.loc[moving].isna().any() or magnitude.loc[moving].lt(0).any():
        raise X15ModelInputError(
            "UP/DOWN rows require nonnegative conditional magnitude"
        )
    frame["_conditional_magnitude"] = magnitude
    frame = frame.sort_values(list(_PANEL_GRAIN), kind="mergesort").reset_index(
        drop=True
    )
    frame = _annotate_discrete_survival_intervals(frame)
    frame["_source_row_id"] = np.arange(len(frame), dtype=int)
    build_x15_week_folds(frame)
    return frame


def _parse_unique_decision_features(
    frame: pd.DataFrame,
    *,
    diagnostic_contract: bool = False,
) -> dict[str, dict[str, object]]:
    pairs = frame.loc[
        :, ["decision_feature_sha256", "decision_features_json"]
    ].drop_duplicates()
    if pairs["decision_feature_sha256"].duplicated().any():
        raise X15ModelInputError(
            "one decision feature sha256 maps to multiple JSON payloads"
        )
    parsed_by_sha256: dict[str, dict[str, object]] = {}
    for digest, decision_json in pairs.itertuples(index=False, name=None):
        actual_hash = _sha256_text_exact(decision_json)
        if digest != actual_hash:
            raise X15ModelInputError("decision feature sha256 mismatch")
        try:
            parsed = json.loads(decision_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise X15ModelInputError(
                "decision_features_json must be valid JSON objects"
            ) from exc
        if not isinstance(parsed, dict):
            raise X15ModelInputError(
                "decision_features_json must contain an object"
            )
        leaked = sorted(set(_walk_keys(parsed)) & _DECISION_LEAKAGE_KEYS)
        if leaked:
            raise X15ModelInputError(
                f"decision-feature leakage detected: {', '.join(leaked)}"
            )
        multi_hot = parsed.get("multi_hot_features", {})
        facts = parsed.get("fact_features", {})
        if not isinstance(multi_hot, dict) or not isinstance(facts, dict):
            raise X15ModelInputError(
                "decision feature fact/multi-hot blocks must be objects"
            )
        if not all(
            isinstance(value, (bool, np.bool_))
            for value in multi_hot.values()
        ):
            raise X15ModelInputError(
                "decision feature multi-hot values must be booleans"
            )
        if diagnostic_contract:
            _validate_diagnostic_decision_payload(parsed)
        parsed_by_sha256[str(digest)] = parsed
    return parsed_by_sha256


def _categorical_from_unique(
    unique_values: Sequence[object], row_codes: np.ndarray
) -> pd.Categorical:
    categories = tuple(
        sorted(
            {
                str(value)
                for value in unique_values
                if value is not None and not bool(pd.isna(value))
            }
        )
    )
    category_codes = {value: index for index, value in enumerate(categories)}
    unique_codes = np.array(
        [
            category_codes.get(str(value), -1)
            if value is not None and not bool(pd.isna(value))
            else -1
            for value in unique_values
        ],
        dtype=np.int32,
    )
    return pd.Categorical.from_codes(
        unique_codes[row_codes], categories=categories
    )


def _decision_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_blocks: Mapping[str, tuple[str, ...]],
    feature_block_ids: Sequence[str],
    parsed_by_sha256: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    governed_features = tuple(
        dict.fromkeys(
            feature
            for block_id in feature_block_ids
            for feature in feature_blocks[block_id]
        )
    )
    parsed = (
        _parse_unique_decision_features(frame)
        if parsed_by_sha256 is None
        else parsed_by_sha256
    )
    row_codes, unique_digests = pd.factorize(
        frame["decision_feature_sha256"], sort=False
    )
    if (row_codes < 0).any():
        raise X15ModelInputError("decision feature sha256 must be nonmissing")
    unique_payloads: list[Mapping[str, object]] = []
    for digest in unique_digests:
        payload = parsed.get(str(digest))
        if payload is None:
            raise X15ModelInputError(
                "decision feature sha256 missing from parsed cache"
            )
        unique_payloads.append(payload)

    feature_data: dict[str, object] = {}
    categorical_features = {
        "fact__source_resolution",
        "fact__primary_action",
        "stage_a_status",
    }
    for feature in governed_features:
        if feature == "multi_hot__*":
            continue
        if feature in {"landmark_seconds", "endpoint_seconds"}:
            panel_values = pd.to_numeric(
                frame[feature], errors="coerce"
            ).to_numpy(dtype=float)
            provided = np.array(
                [feature in payload for payload in unique_payloads],
                dtype=bool,
            )
            if provided.any():
                unique_values = np.array(
                    [
                        float(payload[feature])
                        if feature in payload
                        else math.nan
                        for payload in unique_payloads
                    ],
                    dtype=float,
                )
                row_provided = provided[row_codes]
                if not np.equal(
                    unique_values[row_codes][row_provided],
                    panel_values[row_provided],
                ).all():
                    raise X15ModelInputError(
                        f"decision {feature} does not match panel"
                    )
            feature_data[feature] = panel_values
            continue
        unique_values = [
            (
                payload.get("fact_features", {}).get(
                    feature.removeprefix("fact__")
                )
                if feature.startswith("fact__")
                else payload.get(feature)
            )
            for payload in unique_payloads
        ]
        if feature in categorical_features:
            feature_data[feature] = _categorical_from_unique(
                unique_values, row_codes
            )
        else:
            numeric = pd.to_numeric(
                pd.Series(unique_values, dtype=object), errors="coerce"
            ).to_numpy(dtype=float)
            feature_data[feature] = numeric[row_codes]

    if "multi_hot__*" in governed_features:
        all_multi_hot = sorted(
            {
                str(key)
                for payload in unique_payloads
                for key in payload.get("multi_hot_features", {})
            }
        )
        for raw_key in all_multi_hot:
            unique_states = np.full(
                len(unique_payloads), -1, dtype=np.int8
            )
            for index, payload in enumerate(unique_payloads):
                multi_hot = payload.get("multi_hot_features", {})
                if raw_key in multi_hot:
                    unique_states[index] = int(bool(multi_hot[raw_key]))
            row_states = unique_states[row_codes]
            feature_data[f"multi_hot__{raw_key}"] = pd.arrays.BooleanArray(
                row_states == 1, row_states < 0
            )
    return pd.DataFrame(feature_data, index=frame.index)


def _compact_parsed_decision_payload(
    payload: Mapping[str, object],
    *,
    feature_blocks: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    governed = {
        feature
        for features in feature_blocks.values()
        for feature in features
    }
    compact: dict[str, object] = {}
    fact_features = payload.get("fact_features", {})
    if not isinstance(fact_features, Mapping):
        raise X15ModelInputError(
            "decision feature fact_features must be an object"
        )
    selected_facts = {
        sys.intern(feature.removeprefix("fact__")): fact_features.get(
            feature.removeprefix("fact__")
        )
        for feature in governed
        if feature.startswith("fact__")
        and feature.removeprefix("fact__") in fact_features
    }
    if selected_facts:
        compact["fact_features"] = selected_facts
    if "multi_hot__*" in governed:
        multi_hot = payload.get("multi_hot_features", {})
        if not isinstance(multi_hot, Mapping):
            raise X15ModelInputError(
                "decision feature multi_hot_features must be an object"
            )
        compact["multi_hot_features"] = {
            sys.intern(str(key)): value
            for key, value in multi_hot.items()
        }
    for feature in governed:
        if (
            not feature.startswith("fact__")
            and feature != "multi_hot__*"
            and feature in payload
        ):
            compact[feature] = payload[feature]
    return compact


def _sha256_text_exact(value: object) -> str:
    if type(value) is not str:
        raise X15ModelInputError("decision_features_json must be text")
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolved_columns(
    feature_frame: pd.DataFrame,
    block_id: str,
    *,
    vocabulary_indexes: pd.Index,
    feature_blocks: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    requested = feature_blocks[block_id]
    resolved: list[str] = []
    for column in requested:
        if column == "multi_hot__*":
            resolved.extend(
                sorted(
                    value
                    for value in feature_frame.columns
                    if value.startswith("multi_hot__")
                    and feature_frame.loc[vocabulary_indexes, value]
                    .notna()
                    .any()
                )
            )
        else:
            resolved.append(column)
    missing = [column for column in resolved if column not in feature_frame]
    if missing:
        raise X15ModelInputError(
            f"decision feature block {block_id} missing: {missing}"
        )
    return tuple(dict.fromkeys(resolved))


def _feature_bundle(
    feature_frame: pd.DataFrame,
    train_indexes: pd.Index,
    validation_indexes: pd.Index,
    *,
    block_id: str,
    train_frame: pd.DataFrame,
    feature_blocks: Mapping[str, tuple[str, ...]],
) -> _FeatureBundle:
    columns = _resolved_columns(
        feature_frame,
        block_id,
        vocabulary_indexes=train_indexes,
        feature_blocks=feature_blocks,
    )
    train = feature_frame.loc[train_indexes, list(columns)].copy()
    validation = feature_frame.loc[validation_indexes, list(columns)].copy()
    categorical = [
        column
        for column in columns
        if column
        in {
            "fact__source_resolution",
            "fact__primary_action",
            "stage_a_status",
        }
    ]
    numeric = [column for column in columns if column not in categorical]
    for column in numeric:
        if column.startswith("multi_hot__"):
            train[column] = train[column].fillna(False).astype(np.uint8)
            validation[column] = (
                validation[column].fillna(False).astype(np.uint8)
            )
        else:
            train[column] = pd.to_numeric(train[column], errors="coerce")
            validation[column] = pd.to_numeric(
                validation[column], errors="coerce"
            )
    for column in categorical:
        train[column] = train[column].where(train[column].notna(), np.nan)
        validation[column] = validation[column].where(
            validation[column].notna(), np.nan
        )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )
    train_matrix = np.asarray(transformer.fit_transform(train), dtype=float)
    validation_matrix = np.asarray(transformer.transform(validation), dtype=float)
    digest_rows = sorted(
        [
            {
                "game_id": str(row["game_id"]),
                "source_row_id": int(row["_source_row_id"]),
                "decision_feature_sha256": str(
                    row["decision_feature_sha256"]
                ),
            }
            for _, row in train_frame.iterrows()
        ],
        key=lambda item: (
            item["game_id"],
            item["source_row_id"],
            item["decision_feature_sha256"],
        ),
    )
    return _FeatureBundle(
        transformer=transformer,
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        columns=columns,
        training_sha256=_sha256(
            {
                "feature_block_id": block_id,
                "columns": columns,
                "rows": digest_rows,
            }
        ),
    )


def _head_mask(frame: pd.DataFrame, head: str) -> pd.Series:
    if head == "S_H":
        return (
            frame["_s_interval_at_risk"]
            & frame["_s_interval_truth"].notna()
        )
    if head == "O_H_GIVEN_S":
        return frame["s_h"].eq(True) & frame["o_h_given_s"].notna()
    if head == "DIRECTION":
        return frame["target_eligible"] & frame["_direction_truth"].notna()
    raise AssertionError(head)


def _head_labels(frame: pd.DataFrame, head: str) -> np.ndarray:
    if head == "S_H":
        return (
            frame["_s_interval_truth"].astype(bool).astype(int).to_numpy()
        )
    if head == "O_H_GIVEN_S":
        return frame["o_h_given_s"].astype(bool).astype(int).to_numpy()
    return frame["_direction_truth"].astype(str).to_numpy()


def _evaluation_mask(frame: pd.DataFrame, head: str) -> pd.Series:
    if head == "S_H":
        return frame["s_h"].notna()
    return _head_mask(frame, head)


def _training_hash(
    frame: pd.DataFrame,
    *,
    model_id: str,
    block_id: str,
    head: str,
    weights: pd.Series,
) -> str:
    mask = _head_mask(frame, head)
    rows = sorted(
        [
            {
                "game_id": str(row.game_id),
                "episode_id": str(row.atomic_information_episode_id),
                "source_row_id": int(row._source_row_id),
                "decision_feature_sha256": str(row.decision_feature_sha256),
                "truth": _canonical(
                    row._s_interval_truth
                    if head == "S_H"
                    else row.o_h_given_s
                    if head == "O_H_GIVEN_S"
                    else row._direction_truth
                ),
                "weight": float(weights.loc[index]),
            }
            for index, row in frame.loc[mask].iterrows()
        ],
        key=lambda item: (
            item["game_id"],
            item["episode_id"],
            item["source_row_id"],
            item["decision_feature_sha256"],
            json.dumps(item["truth"], sort_keys=True),
            item["weight"],
        ),
    )
    return _sha256(
        {
            "model_id": model_id,
            "feature_block_id": block_id,
            "head": head,
            "rows": rows,
        }
    )


class _EmpiricalBinary:
    def __init__(
        self, frame: pd.DataFrame, labels: np.ndarray, weights: np.ndarray
    ) -> None:
        self.global_rate = float(np.average(labels, weights=weights))
        self.rates: dict[tuple[int, int], float] = {}
        work = frame.loc[:, ["landmark_seconds", "endpoint_seconds"]].copy()
        work["_label"] = labels
        work["_weight"] = weights
        for key, group in work.groupby(
            ["landmark_seconds", "endpoint_seconds"], sort=True
        ):
            self.rates[(int(key[0]), int(key[1]))] = float(
                np.average(group["_label"], weights=group["_weight"])
            )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                self.rates.get(
                    (int(row.landmark_seconds), int(row.endpoint_seconds)),
                    self.global_rate,
                )
                for row in frame.itertuples(index=False)
            ],
            dtype=float,
        )


class _EmpiricalDirection:
    def __init__(
        self, frame: pd.DataFrame, labels: np.ndarray, weights: np.ndarray
    ) -> None:
        self.global_rates = self._rates(labels, weights)
        self.rates: dict[tuple[int, int], np.ndarray] = {}
        work = frame.loc[:, ["landmark_seconds", "endpoint_seconds"]].copy()
        work["_label"] = labels
        work["_weight"] = weights
        for key, group in work.groupby(
            ["landmark_seconds", "endpoint_seconds"], sort=True
        ):
            self.rates[(int(key[0]), int(key[1]))] = self._rates(
                group["_label"].to_numpy(), group["_weight"].to_numpy()
            )

    @staticmethod
    def _rates(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
        totals = np.array(
            [
                weights[labels == direction].sum()
                for direction in DIRECTION_CLASSES
            ],
            dtype=float,
        )
        if totals.sum() <= 0:
            return np.full(3, 1 / 3)
        return totals / totals.sum()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.vstack(
            [
                self.rates.get(
                    (int(row.landmark_seconds), int(row.endpoint_seconds)),
                    self.global_rates,
                )
                for row in frame.itertuples(index=False)
            ]
        )


def _fit_head(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    model_id: str,
    block_id: str,
    head: str,
    random_state: int,
) -> tuple[_FittedHead, pd.Series]:
    mask = _head_mask(frame, head)
    weights = hierarchical_sample_weights(frame, mask)
    selected = frame.loc[mask]
    digest = _training_hash(
        frame,
        model_id=model_id,
        block_id=block_id,
        head=head,
        weights=weights,
    )
    if selected.empty:
        return (
            _FittedHead(
                model_id,
                head,
                INSUFFICIENT_SUPPORT,
                "no eligible training rows",
                None,
                digest,
                (),
            ),
            weights,
        )
    labels = _head_labels(selected, head)
    classes = tuple(sorted(set(labels)))
    game_count = selected["game_id"].nunique()
    required_classes = 3 if head == "DIRECTION" else 2
    minimum_games = (
        _DIRECTION_MIN_GAMES if head == "DIRECTION" else _BINARY_MIN_GAMES
    )
    learned = model_id != "b0_empirical_v1"
    if learned and (
        len(classes) < required_classes or game_count < minimum_games
    ):
        reason = (
            "single class training support"
            if len(classes) == 1
            else "missing required classes or games"
        )
        return (
            _FittedHead(
                model_id,
                head,
                INSUFFICIENT_SUPPORT,
                reason,
                None,
                digest,
                classes,
            ),
            weights,
        )
    selected_positions = np.flatnonzero(mask.to_numpy())
    selected_matrix = matrix[selected_positions]
    selected_weight = weights.loc[mask].to_numpy(dtype=float)
    if model_id == "b0_empirical_v1":
        estimator: object = (
            _EmpiricalDirection(selected, labels, selected_weight)
            if head == "DIRECTION"
            else _EmpiricalBinary(selected, labels.astype(int), selected_weight)
        )
    elif model_id == "regularized_logistic_v1":
        estimator = LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=1_000,
            random_state=random_state,
        )
        estimator.fit(selected_matrix, labels, sample_weight=selected_weight)
    elif model_id == "shallow_xgboost_v1":
        if head == "DIRECTION":
            encoded = np.array(
                [DIRECTION_CLASSES.index(str(label)) for label in labels]
            )
            estimator = XGBClassifier(
                **_XGB_PARAMS,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=random_state,
            )
            estimator.fit(
                selected_matrix, encoded, sample_weight=selected_weight, verbose=False
            )
        else:
            estimator = XGBClassifier(
                **_XGB_PARAMS,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
            )
            estimator.fit(
                selected_matrix,
                labels.astype(int),
                sample_weight=selected_weight,
                verbose=False,
            )
    else:
        raise X15ModelInputError(f"unknown model_id: {model_id}")
    return (
        _FittedHead(
            model_id,
            head,
            SUPPORTED,
            SUPPORTED,
            estimator,
            digest,
            classes,
        ),
        weights,
    )


def _predict_head(
    fitted: _FittedHead, frame: pd.DataFrame, matrix: np.ndarray
) -> np.ndarray:
    if fitted.estimator is None:
        columns = 3 if fitted.head == "DIRECTION" else 1
        result = np.full((len(frame), columns), np.nan)
        return result if columns == 3 else result[:, 0]
    if fitted.model_id == "b0_empirical_v1":
        return fitted.estimator.predict(frame)
    probabilities = np.asarray(fitted.estimator.predict_proba(matrix), dtype=float)
    if fitted.head != "DIRECTION":
        return probabilities[:, 1]
    result = np.zeros((len(frame), 3), dtype=float)
    estimator_classes = tuple(fitted.estimator.classes_)
    for source, value in enumerate(estimator_classes):
        class_index = (
            int(value)
            if isinstance(value, (int, np.integer))
            else DIRECTION_CLASSES.index(str(value))
        )
        result[:, class_index] = probabilities[:, source]
    return result


def _prequential_calibrator(
    train: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    model_id: str,
    block_id: str,
    head: str,
    random_state: int,
    feature_blocks: Mapping[str, tuple[str, ...]],
) -> BinaryPlattCalibrator | DirectionTemperatureCalibrator:
    raw_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    game_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    row_id_parts: list[np.ndarray] = []
    weeks = tuple(sorted(train["nfl_week"].unique()))
    for current_week in weeks[1:]:
        inner_train = train[train["nfl_week"] < current_week]
        inner_validation = train[train["nfl_week"] == current_week]
        if inner_train.empty or inner_validation.empty:
            continue
        bundle = _feature_bundle(
            feature_frame,
            inner_train.index,
            inner_validation.index,
            block_id=block_id,
            train_frame=inner_train,
            feature_blocks=feature_blocks,
        )
        fitted, _ = _fit_head(
            inner_train,
            bundle.train_matrix,
            model_id=model_id,
            block_id=block_id,
            head=head,
            random_state=random_state,
        )
        if fitted.status != SUPPORTED:
            continue
        evaluation_mask = _head_mask(inner_validation, head)
        if not evaluation_mask.any():
            continue
        raw = _predict_head(
            fitted, inner_validation, bundle.validation_matrix
        )[evaluation_mask.to_numpy()]
        labels = _head_labels(inner_validation.loc[evaluation_mask], head)
        weights = hierarchical_sample_weights(
            inner_validation, evaluation_mask
        ).loc[evaluation_mask]
        raw_parts.append(np.asarray(raw))
        label_parts.append(labels)
        game_parts.append(
            inner_validation.loc[evaluation_mask, "game_id"].to_numpy()
        )
        weight_parts.append(weights.to_numpy(dtype=float))
        row_id_parts.append(
            np.asarray(
                [
                    "|".join(
                        (
                            str(row["game_id"]),
                            str(row["atomic_information_episode_id"]),
                            str(row["venue"]),
                            str(row["landmark_seconds"]),
                            str(row["endpoint_seconds"]),
                        )
                    )
                    for _, row in inner_validation.loc[
                        evaluation_mask
                    ].iterrows()
                ]
            )
        )
    if head == "DIRECTION":
        raw = (
            np.vstack(raw_parts)
            if raw_parts
            else np.empty((0, len(DIRECTION_CLASSES)))
        )
        labels = (
            np.concatenate(label_parts)
            if label_parts
            else np.array([], dtype=object)
        )
        games = (
            np.concatenate(game_parts)
            if game_parts
            else np.array([], dtype=object)
        )
        weights = (
            np.concatenate(weight_parts)
            if weight_parts
            else np.array([], dtype=float)
        )
        row_ids = (
            np.concatenate(row_id_parts)
            if row_id_parts
            else np.array([], dtype=object)
        )
        return fit_direction_temperature(
            raw, labels, games, sample_weight=weights, row_ids=row_ids
        )
    raw_binary = (
        np.concatenate(raw_parts) if raw_parts else np.array([], dtype=float)
    )
    labels_binary = (
        np.concatenate(label_parts)
        if label_parts
        else np.array([], dtype=int)
    )
    games_binary = (
        np.concatenate(game_parts)
        if game_parts
        else np.array([], dtype=object)
    )
    weights_binary = (
        np.concatenate(weight_parts)
        if weight_parts
        else np.array([], dtype=float)
    )
    row_ids_binary = (
        np.concatenate(row_id_parts)
        if row_id_parts
        else np.array([], dtype=object)
    )
    return fit_binary_platt(
        raw_binary,
        labels_binary,
        games_binary,
        sample_weight=weights_binary,
        row_ids=row_ids_binary,
    )


def _calibrate_binary(
    calibrator: BinaryPlattCalibrator, raw: np.ndarray
) -> np.ndarray:
    result = np.full(len(raw), np.nan)
    valid = np.isfinite(raw)
    if valid.any():
        result[valid] = calibrator.transform(raw[valid])
    return result


def _calibrate_direction(
    calibrator: DirectionTemperatureCalibrator, raw: np.ndarray
) -> np.ndarray:
    result = np.full_like(raw, np.nan, dtype=float)
    valid = np.isfinite(raw).all(axis=1)
    if valid.any():
        result[valid] = calibrator.transform(raw[valid])
    return result


def _support_row(
    fold: X15WeekFold,
    *,
    training_venue: str,
    evaluation_venue: str,
    transport_mode: str,
    block_id: str,
    fitted: _FittedHead,
    train: pd.DataFrame,
    cohort_authority_sha256: str,
) -> dict[str, object]:
    mask = _head_mask(train, fitted.head)
    return {
        "fold_id": fold.fold_id,
        "cohort_authority_sha256": cohort_authority_sha256,
        "venue": evaluation_venue,
        "training_venue": training_venue,
        "calibration_venue": training_venue,
        "transport_mode": transport_mode,
        "feature_block_id": block_id,
        "model_id": fitted.model_id,
        "head": fitted.head,
        "support_status": fitted.status,
        "support_reason": fitted.reason,
        "training_row_count": int(mask.sum()),
        "training_game_count": int(train.loc[mask, "game_id"].nunique()),
        "training_game_ids": tuple(
            sorted(train["game_id"].astype(str).unique())
        ),
        "training_classes": tuple(str(value) for value in fitted.classes),
        "binary_min_games": _BINARY_MIN_GAMES,
        "direction_min_games": _DIRECTION_MIN_GAMES,
        "training_sha256": fitted.training_sha256,
    }


def _prediction_rows(
    validation: pd.DataFrame,
    *,
    fold: X15WeekFold,
    training_venue: str,
    evaluation_venue: str,
    transport_mode: str,
    block_id: str,
    model_id: str,
    feature_block_sha256: str,
    fold_sha256: str,
    training_data_sha256: str,
    training_game_ids: tuple[str, ...],
    validation_game_ids: tuple[str, ...],
    cohort_authority_sha256: str,
    bundle: _FeatureBundle,
    fits: Mapping[str, _FittedHead],
    calibrators: Mapping[
        str, BinaryPlattCalibrator | DirectionTemperatureCalibrator
    ],
    raw: Mapping[str, np.ndarray],
    calibrated: Mapping[str, np.ndarray],
    validation_weights: Mapping[str, pd.Series],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, (_, source) in enumerate(validation.iterrows()):
        direction_truth = source["_direction_truth"]
        survived = pd.notna(source["s_h"]) and bool(source["s_h"])
        rows.append(
            {
                "source_row_id": int(source["_source_row_id"]),
                "cohort_authority_sha256": cohort_authority_sha256,
                "game_id": source["game_id"],
                "nfl_week": int(source["nfl_week"]),
                "atomic_information_episode_id": source[
                    "atomic_information_episode_id"
                ],
                "venue": evaluation_venue,
                "training_venue": training_venue,
                "calibration_venue": training_venue,
                "transport_mode": transport_mode,
                "actual_home_contract_id": source["actual_home_contract_id"],
                "landmark_seconds": int(source["landmark_seconds"]),
                "endpoint_seconds": int(source["endpoint_seconds"]),
                "fold_id": fold.fold_id,
                "train_weeks": fold.train_weeks,
                "validation_weeks": fold.validation_weeks,
                "training_game_ids": training_game_ids,
                "validation_game_ids": validation_game_ids,
                "preprocessor_fit_game_ids": training_game_ids,
                "model_id": model_id,
                "feature_block_id": block_id,
                "decision_eligible": True,
                "target_eligible": bool(source["target_eligible"]),
                "s_h_truth": (
                    bool(source["s_h"]) if pd.notna(source["s_h"]) else pd.NA
                ),
                "o_h_given_s_truth": (
                    bool(source["o_h_given_s"])
                    if survived
                    and pd.notna(source["o_h_given_s"])
                    else pd.NA
                ),
                "direction_truth": (
                    str(direction_truth)
                    if pd.notna(direction_truth)
                    and bool(source["target_eligible"])
                    else pd.NA
                ),
                "direction_truth_status": (
                    "AVAILABLE" if bool(source["target_eligible"]) else "UNAVAILABLE"
                ),
                "conditional_magnitude_truth": (
                    float(source["_conditional_magnitude"])
                    if bool(source["target_eligible"])
                    and str(direction_truth) in DIRECTION_CONDITIONS
                    else math.nan
                ),
                "s_h_weight": float(validation_weights["S_H"].loc[source.name]),
                "o_h_given_s_weight": float(
                    validation_weights["O_H_GIVEN_S"].loc[source.name]
                ),
                "direction_weight": float(
                    validation_weights["DIRECTION"].loc[source.name]
                ),
                "s_h_support_status": fits["S_H"].status,
                "o_h_given_s_support_status": fits["O_H_GIVEN_S"].status,
                "direction_support_status": fits["DIRECTION"].status,
                "s_h_raw_probability": float(raw["S_H"][position]),
                "s_h_calibrated_probability": float(calibrated["S_H"][position]),
                "o_h_given_s_raw_probability": float(
                    raw["O_H_GIVEN_S"][position]
                ),
                "o_h_given_s_calibrated_probability": float(
                    calibrated["O_H_GIVEN_S"][position]
                ),
                "direction_raw_prob_down": float(raw["DIRECTION"][position, 0]),
                "direction_raw_prob_no_move": float(
                    raw["DIRECTION"][position, 1]
                ),
                "direction_raw_prob_up": float(raw["DIRECTION"][position, 2]),
                "direction_calibrated_prob_down": float(
                    calibrated["DIRECTION"][position, 0]
                ),
                "direction_calibrated_prob_no_move": float(
                    calibrated["DIRECTION"][position, 1]
                ),
                "direction_calibrated_prob_up": float(
                    calibrated["DIRECTION"][position, 2]
                ),
                "s_h_calibration_status": calibrators["S_H"].status,
                "o_h_given_s_calibration_status": calibrators[
                    "O_H_GIVEN_S"
                ].status,
                "direction_calibration_status": calibrators["DIRECTION"].status,
                "s_h_calibration_support_reason": calibrators[
                    "S_H"
                ].support_reason,
                "o_h_given_s_calibration_support_reason": calibrators[
                    "O_H_GIVEN_S"
                ].support_reason,
                "direction_calibration_support_reason": calibrators[
                    "DIRECTION"
                ].support_reason,
                "calibration_support_min_rows": calibrators[
                    "S_H"
                ].support_min_rows,
                "calibration_support_min_games": calibrators[
                    "S_H"
                ].support_min_games,
                "calibrator_fit_game_ids_s_h": tuple(
                    sorted(calibrators["S_H"].fit_game_ids)
                ),
                "calibrator_fit_game_ids_o_h_given_s": tuple(
                    sorted(calibrators["O_H_GIVEN_S"].fit_game_ids)
                ),
                "calibrator_fit_game_ids_direction": tuple(
                    sorted(calibrators["DIRECTION"].fit_game_ids)
                ),
                "training_data_sha256": training_data_sha256,
                "preprocessor_training_sha256": bundle.training_sha256,
                "s_h_model_training_sha256": fits["S_H"].training_sha256,
                "o_h_given_s_model_training_sha256": fits[
                    "O_H_GIVEN_S"
                ].training_sha256,
                "direction_model_training_sha256": fits[
                    "DIRECTION"
                ].training_sha256,
                "s_h_calibration_training_sha256": calibrators[
                    "S_H"
                ].training_sha256,
                "o_h_given_s_calibration_training_sha256": calibrators[
                    "O_H_GIVEN_S"
                ].training_sha256,
                "direction_calibration_training_sha256": calibrators[
                    "DIRECTION"
                ].training_sha256,
                "feature_block_sha256": feature_block_sha256,
                "model_spec_sha256": _sha256(
                    {
                        "model_id": model_id,
                        "logistic": {
                            "C": 0.5,
                            "solver": "lbfgs",
                            "max_iter": 1_000,
                        },
                        "xgboost": (
                            _XGB_PARAMS
                            if model_id == "shallow_xgboost_v1"
                            else None
                        ),
                        "b0_strata": (
                            ("landmark_seconds", "endpoint_seconds")
                            if model_id == "b0_empirical_v1"
                            else None
                        ),
                    }
                ),
                "fold_sha256": fold_sha256,
            }
        )
    return rows


def _binary_game_metrics(
    group: pd.DataFrame,
    *,
    truth_column: str,
    probability_column: str,
    weight_column: str,
) -> dict[str, float]:
    valid = group[truth_column].notna() & group[probability_column].notna()
    coverage = float(valid.mean())
    if not valid.any():
        return {
            "nll": math.nan,
            "brier": math.nan,
            "calibration_error": math.nan,
            "coverage": coverage,
        }
    truth = group.loc[valid, truth_column].astype(int).to_numpy()
    probability = np.clip(
        group.loc[valid, probability_column].to_numpy(dtype=float),
        _PROBABILITY_EPSILON,
        1 - _PROBABILITY_EPSILON,
    )
    weights = group.loc[valid, weight_column].to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(truth))
    nll = -(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))
    return {
        "nll": float(np.average(nll, weights=weights)),
        "brier": float(np.average((probability - truth) ** 2, weights=weights)),
        "calibration_error": float(
            abs(np.average(probability, weights=weights) - np.average(truth, weights=weights))
        ),
        "coverage": coverage,
    }


def _direction_game_metrics(group: pd.DataFrame) -> dict[str, float]:
    probability_columns = [
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    ]
    valid = group["direction_truth"].notna() & group[
        probability_columns
    ].notna().all(axis=1)
    coverage = float(valid.mean())
    if not valid.any():
        return {
            "multiclass_log_loss": math.nan,
            "multiclass_brier": math.nan,
            "brier_down": math.nan,
            "brier_no_move": math.nan,
            "brier_up": math.nan,
            "coverage": coverage,
        }
    probabilities = np.clip(
        group.loc[valid, probability_columns].to_numpy(dtype=float),
        _PROBABILITY_EPSILON,
        1,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    truth_index = np.array(
        [
            DIRECTION_CLASSES.index(str(value))
            for value in group.loc[valid, "direction_truth"]
        ]
    )
    one_hot = np.eye(3)[truth_index]
    weights = group.loc[valid, "direction_weight"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(truth_index))
    return {
        "multiclass_log_loss": float(
            np.average(
                -np.log(probabilities[np.arange(len(truth_index)), truth_index]),
                weights=weights,
            )
        ),
        "multiclass_brier": float(
            np.average(np.sum((probabilities - one_hot) ** 2, axis=1), weights=weights)
        ),
        "brier_down": float(
            np.average((probabilities[:, 0] - one_hot[:, 0]) ** 2, weights=weights)
        ),
        "brier_no_move": float(
            np.average((probabilities[:, 1] - one_hot[:, 1]) ** 2, weights=weights)
        ),
        "brier_up": float(
            np.average((probabilities[:, 2] - one_hot[:, 2]) ** 2, weights=weights)
        ),
        "coverage": coverage,
    }


def _probability_metric_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = [
        "fold_id",
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        "model_id",
        "feature_block_id",
    ]
    for key, candidate in predictions.groupby(keys, sort=True):
        game_rows: list[dict[str, object]] = []
        for game_id, game in candidate.groupby("game_id", sort=True):
            head_metrics = {
                "S_H": _binary_game_metrics(
                    game,
                    truth_column="s_h_truth",
                    probability_column="s_h_calibrated_probability",
                    weight_column="s_h_weight",
                ),
                "O_H_GIVEN_S": _binary_game_metrics(
                    game,
                    truth_column="o_h_given_s_truth",
                    probability_column="o_h_given_s_calibrated_probability",
                    weight_column="o_h_given_s_weight",
                ),
                "DIRECTION": _direction_game_metrics(game),
            }
            for head, metrics in head_metrics.items():
                for metric_name, metric_value in metrics.items():
                    row = {
                        **dict(zip(keys, key, strict=True)),
                        "metric_scope": "GAME",
                        "game_id": game_id,
                        "head": head,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "game_effect_p025": math.nan,
                        "game_effect_p975": math.nan,
                    }
                    rows.append(row)
                    game_rows.append(row)
        game_frame = pd.DataFrame(game_rows)
        if game_frame.empty:
            continue
        for (head, metric_name), values in game_frame.groupby(
            ["head", "metric_name"], sort=True
        ):
            finite = values["metric_value"].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "metric_scope": "GAME_CLUSTER_SUMMARY",
                    "game_id": pd.NA,
                    "head": head,
                    "metric_name": metric_name,
                    "metric_value": float(np.mean(finite)) if len(finite) else math.nan,
                    "game_effect_p025": (
                        float(np.quantile(finite, 0.025)) if len(finite) else math.nan
                    ),
                    "game_effect_p975": (
                        float(np.quantile(finite, 0.975)) if len(finite) else math.nan
                    ),
                }
            )
    return rows


def _training_data_hash(frame: pd.DataFrame) -> str:
    rows = sorted(
        [
            {
                "game_id": str(row["game_id"]),
                "episode_id": str(row["atomic_information_episode_id"]),
                "source_row_id": int(row["_source_row_id"]),
                "venue": str(row["venue"]),
                "decision_feature_sha256": str(
                    row["decision_feature_sha256"]
                ),
                "s_h": _canonical(row["s_h"]),
                "o_h_given_s": _canonical(row["o_h_given_s"]),
                "direction": _canonical(row["_direction_truth"]),
                "magnitude": _canonical(row["_conditional_magnitude"]),
            }
            for _, row in frame.iterrows()
        ],
        key=lambda item: (
            item["game_id"],
            item["episode_id"],
            item["source_row_id"],
            item["venue"],
            item["decision_feature_sha256"],
            json.dumps(item, sort_keys=True),
        ),
    )
    return _sha256(rows)


def _magnitude_outputs(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: _FeatureBundle,
    fold: X15WeekFold,
    training_venue: str,
    evaluation_venue: str,
    transport_mode: str,
    block_id: str,
    random_state: int,
    support_contract: QuantileSupportContract,
    validation_weights: pd.Series,
    training_game_ids: tuple[str, ...],
    cohort_authority_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mask = (
        train["target_eligible"]
        & train["_direction_truth"].isin(DIRECTION_CONDITIONS)
        & train["_conditional_magnitude"].notna()
    )
    weights = hierarchical_sample_weights(train, mask)
    positions = np.flatnonzero(mask.to_numpy())
    numeric_columns = tuple(
        f"feature_{index}" for index in range(bundle.train_matrix.shape[1])
    )
    training_features = pd.DataFrame(
        bundle.train_matrix[positions], columns=numeric_columns
    )
    fitted = fit_directional_quantiles(
        training_features,
        train.loc[mask, "_conditional_magnitude"].to_numpy(dtype=float),
        train.loc[mask, "_direction_truth"].to_numpy(),
        train.loc[mask, "game_id"].to_numpy(),
        sample_weight=weights.loc[mask].to_numpy(dtype=float),
        row_ids=np.asarray(
            [
                "|".join(
                    (
                        str(row["game_id"]),
                        str(row["atomic_information_episode_id"]),
                        str(row["venue"]),
                        str(row["landmark_seconds"]),
                        str(row["endpoint_seconds"]),
                    )
                )
                for _, row in train.loc[mask].iterrows()
            ]
        ),
        support_contract=support_contract,
        random_state=random_state,
    )
    validation_features = pd.DataFrame(
        bundle.validation_matrix, columns=numeric_columns
    )
    support_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    quantile_columns = [f"q{int(value * 100):02d}" for value in PRIMARY_QUANTILES]
    for direction in DIRECTION_CONDITIONS:
        support = fitted.support[direction]
        support_rows.append(
            {
                "fold_id": fold.fold_id,
                "cohort_authority_sha256": cohort_authority_sha256,
                "venue": evaluation_venue,
                "training_venue": training_venue,
                "calibration_venue": training_venue,
                "transport_mode": transport_mode,
                "feature_block_id": block_id,
                "model_id": QUANTILE_MODEL_ID,
                "head": f"MAGNITUDE_{direction}",
                "support_status": support.primary_status,
                "support_reason": (
                    SUPPORTED
                    if support.primary_status == SUPPORTED
                    else INSUFFICIENT_SUPPORT
                ),
                "training_row_count": support.row_count,
                "training_game_count": support.game_count,
                "training_game_ids": training_game_ids,
                "training_classes": (
                    (direction,) if support.row_count > 0 else ()
                ),
                "primary_min_rows": support.primary_min_rows,
                "primary_min_games": support.primary_min_games,
                "extreme_min_rows": support.extreme_min_rows,
                "extreme_min_games": support.extreme_min_games,
                "extreme_quantile_status": support.extreme_status,
                "training_sha256": support.training_sha256,
            }
        )
        predicted = fitted.predict(validation_features, direction=direction)
        for position, (_, source) in enumerate(validation.iterrows()):
            values = {
                column: (
                    float(predicted.iloc[position][column])
                    if column in predicted
                    and pd.notna(predicted.iloc[position][column])
                    else math.nan
                )
                for column in quantile_columns
            }
            output_rows.append(
                {
                    "source_row_id": int(source["_source_row_id"]),
                    "cohort_authority_sha256": cohort_authority_sha256,
                    "game_id": source["game_id"],
                    "nfl_week": int(source["nfl_week"]),
                    "atomic_information_episode_id": source[
                        "atomic_information_episode_id"
                    ],
                    "venue": evaluation_venue,
                    "training_venue": training_venue,
                    "calibration_venue": training_venue,
                    "transport_mode": transport_mode,
                    "fold_id": fold.fold_id,
                    "feature_block_id": block_id,
                    "model_id": QUANTILE_MODEL_ID,
                    "direction_condition": direction,
                    "direction_truth": (
                        source["_direction_truth"]
                        if bool(source["target_eligible"])
                        else pd.NA
                    ),
                    "conditional_magnitude_truth": (
                        float(source["_conditional_magnitude"])
                        if bool(source["target_eligible"])
                        and source["_direction_truth"] == direction
                        else math.nan
                    ),
                    "magnitude_weight": (
                        float(validation_weights.loc[source.name])
                        if bool(source["target_eligible"])
                        and source["_direction_truth"] == direction
                        else 0.0
                    ),
                    "support_status": support.primary_status,
                    "extreme_quantile_status": support.extreme_status,
                    "q05": (
                        float(predicted.iloc[position]["q05"])
                        if "q05" in predicted
                        and pd.notna(predicted.iloc[position]["q05"])
                        else math.nan
                    ),
                    **values,
                    "q95": (
                        float(predicted.iloc[position]["q95"])
                        if "q95" in predicted
                        and pd.notna(predicted.iloc[position]["q95"])
                        else math.nan
                    ),
                    "preprocessor_training_sha256": bundle.training_sha256,
                    "quantile_training_sha256": support.training_sha256,
                    "model_spec_sha256": _sha256(
                        {
                            "model_id": QUANTILE_MODEL_ID,
                            "primary_quantiles": PRIMARY_QUANTILES,
                            "support_contract": support_contract,
                        }
                    ),
                    "preprocessor_fit_game_ids": training_game_ids,
                }
            )
    return output_rows, support_rows


def _magnitude_metric_rows(
    quantiles: pd.DataFrame,
) -> list[dict[str, object]]:
    if quantiles.empty:
        return []
    rows: list[dict[str, object]] = []
    keys = [
        "fold_id",
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        "model_id",
        "feature_block_id",
    ]
    primary = [f"q{int(value * 100):02d}" for value in PRIMARY_QUANTILES]
    for key, candidate in quantiles.groupby(keys, sort=True):
        game_values: list[dict[str, object]] = []
        for game_id, game in candidate.groupby("game_id", sort=True):
            valid = game["conditional_magnitude_truth"].notna() & game[
                primary
            ].notna().all(axis=1)
            if not valid.any():
                continue
            metrics = magnitude_distribution_metrics(
                game.loc[valid, "conditional_magnitude_truth"].to_numpy(),
                game.loc[valid, primary],
                sample_weight=game.loc[valid, "magnitude_weight"].to_numpy(
                    dtype=float
                ),
            )
            for metric_name, value in metrics.items():
                row = {
                    **dict(zip(keys, key, strict=True)),
                    "metric_scope": "GAME",
                    "game_id": game_id,
                    "head": "MAGNITUDE",
                    "metric_name": metric_name,
                    "metric_value": value,
                    "game_effect_p025": math.nan,
                    "game_effect_p975": math.nan,
                }
                rows.append(row)
                game_values.append(row)
        game_frame = pd.DataFrame(game_values)
        if game_frame.empty:
            continue
        for metric_name, values in game_frame.groupby("metric_name", sort=True):
            finite = values["metric_value"].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "metric_scope": "GAME_CLUSTER_SUMMARY",
                    "game_id": pd.NA,
                    "head": "MAGNITUDE",
                    "metric_name": metric_name,
                    "metric_value": float(np.mean(finite)),
                    "game_effect_p025": float(np.quantile(finite, 0.025)),
                    "game_effect_p975": float(np.quantile(finite, 0.975)),
                }
            )
    return rows


def _run_x15_walk_forward_engine(
    frame: pd.DataFrame,
    *,
    feature_blocks: Mapping[str, tuple[str, ...]],
    baseline_block_id: str,
    parsed_by_sha256: Mapping[str, Mapping[str, object]] | None = None,
    model_ids: Sequence[str] = MODEL_IDS,
    feature_block_ids: Sequence[str],
    fold_ids: Sequence[str] | None = None,
    transport_pairs: Sequence[tuple[str, str]] = (),
    include_magnitude: bool = True,
    quantile_support_contract: QuantileSupportContract = QuantileSupportContract(),
    random_state: int = RANDOM_STATE,
) -> X15ModelRun:
    """Run fixed Stage B OOF folds, optionally sliced for resumable execution.

    Execution slicing never changes model parameters or support gates.  A
    confirmatory run uses all defaults.
    """

    unknown_models = sorted(set(model_ids) - set(MODEL_IDS))
    unknown_blocks = sorted(set(feature_block_ids) - set(feature_blocks))
    if unknown_models:
        raise X15ModelInputError(f"unknown model_ids: {unknown_models}")
    if unknown_blocks:
        raise X15ModelInputError(f"unknown feature blocks: {unknown_blocks}")
    if len(set(model_ids)) != len(tuple(model_ids)):
        raise X15ModelInputError("model_ids must be unique")
    if len(set(feature_block_ids)) != len(tuple(feature_block_ids)):
        raise X15ModelInputError("feature_block_ids must be unique")
    decision_mask = frame["decision_eligible"]
    decision = (
        frame if bool(decision_mask.all()) else frame.loc[decision_mask]
    )
    if decision.empty:
        raise X15ModelInputError("panel has no decision-eligible rows")
    cohort_authority_sha256 = str(
        decision["cohort_authority_sha256"].iloc[0]
    )
    venues = tuple(sorted(decision["venue"].unique()))
    normalized_transport_pairs: list[tuple[str, str]] = []
    for pair in transport_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise X15ModelInputError(
                "transport_pairs must contain (source, target) venue pairs"
            )
        source_venue, target_venue = (str(pair[0]), str(pair[1]))
        if source_venue == target_venue:
            raise X15ModelInputError(
                "transport source and target venues must differ"
            )
        if source_venue not in venues or target_venue not in venues:
            raise X15ModelInputError(
                "transport source and target must exist in the panel"
            )
        normalized_transport_pairs.append((source_venue, target_venue))
    if len(set(normalized_transport_pairs)) != len(normalized_transport_pairs):
        raise X15ModelInputError("transport_pairs must be unique")
    execution_units = [
        (venue, venue, "VENUE_SPECIFIC") for venue in venues
    ] + [
        (source, target, "NO_TARGET_RECALIBRATION")
        for source, target in normalized_transport_pairs
    ]
    folds = build_x15_week_folds(decision)
    available_fold_ids = {fold.fold_id for fold in folds}
    selected_fold_ids = (
        tuple(fold.fold_id for fold in folds)
        if fold_ids is None
        else tuple(fold_ids)
    )
    if not set(selected_fold_ids).issubset(available_fold_ids):
        raise X15ModelInputError("unknown fold_ids")
    selected_folds = tuple(
        fold for fold in folds if fold.fold_id in selected_fold_ids
    )
    selected_game_ids = {
        game_id
        for fold in selected_folds
        for game_id in (*fold.train_game_ids, *fold.validation_game_ids)
    }
    selected_game_mask = decision["game_id"].isin(selected_game_ids)
    if not bool(selected_game_mask.all()):
        decision = decision.loc[selected_game_mask]
    feature_frame = _decision_feature_frame(
        decision,
        feature_blocks=feature_blocks,
        feature_block_ids=feature_block_ids,
        parsed_by_sha256=parsed_by_sha256,
    )
    run_config = {
        "model_ids": tuple(model_ids),
        "feature_block_ids": tuple(feature_block_ids),
        "fold_ids": selected_fold_ids,
        "transport_pairs": tuple(normalized_transport_pairs),
        "include_magnitude": bool(include_magnitude),
        "survival_probability_contract": SURVIVAL_PROBABILITY_CONTRACT,
        "random_state": int(random_state),
        "effective_seed_contract": {
            "contract_id": EFFECTIVE_SEED_CONTRACT_ID,
            "coordinate_fields": EFFECTIVE_SEED_COORDINATE_FIELDS,
            "modulus": EFFECTIVE_SEED_MODULUS,
        },
        "feature_blocks": feature_blocks,
        "xgb_params": _XGB_PARAMS,
        "quantile_support_contract": quantile_support_contract,
        "cohort_authority_sha256": cohort_authority_sha256,
    }
    run_hash = _sha256(run_config)
    prediction_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for fold in selected_folds:
        fold_hash = _sha256(
            {
                "fold_id": fold.fold_id,
                "train_weeks": fold.train_weeks,
                "validation_weeks": fold.validation_weeks,
                "train_game_ids": fold.train_game_ids,
                "validation_game_ids": fold.validation_game_ids,
            }
        )
        for (
            training_venue,
            evaluation_venue,
            transport_mode,
        ) in execution_units:
            train = decision[
                decision["game_id"].isin(fold.train_game_ids)
                & decision["venue"].eq(training_venue)
            ]
            validation = decision[
                decision["game_id"].isin(fold.validation_game_ids)
                & decision["venue"].eq(evaluation_venue)
            ]
            if train.empty or validation.empty:
                continue
            actual_training_game_ids = tuple(
                sorted(train["game_id"].astype(str).unique())
            )
            actual_validation_game_ids = tuple(
                sorted(validation["game_id"].astype(str).unique())
            )
            training_data_sha256 = _training_data_hash(train)
            validation_weights = {
                head: hierarchical_sample_weights(
                    validation, _evaluation_mask(validation, head)
                )
                for head in ("S_H", "O_H_GIVEN_S", "DIRECTION")
            }
            magnitude_validation_mask = (
                validation["target_eligible"]
                & validation["_direction_truth"].isin(DIRECTION_CONDITIONS)
                & validation["_conditional_magnitude"].notna()
            )
            validation_weights["MAGNITUDE"] = hierarchical_sample_weights(
                validation, magnitude_validation_mask
            )
            for head, weights in validation_weights.items():
                for game_id, values in weights.groupby(
                    validation["game_id"], sort=True
                ):
                    weight_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "venue": evaluation_venue,
                            "training_venue": training_venue,
                            "calibration_venue": training_venue,
                            "transport_mode": transport_mode,
                            "partition": "VALIDATION",
                            "head": head,
                            "game_id": game_id,
                            "total_weight": float(values.sum()),
                        }
                    )
            for block_id in feature_block_ids:
                feature_block_hash = _sha256(
                    {
                        "feature_block_id": block_id,
                        "features": feature_blocks[block_id],
                    }
                )
                bundle = _feature_bundle(
                    feature_frame,
                    train.index,
                    validation.index,
                    block_id=block_id,
                    train_frame=train,
                    feature_blocks=feature_blocks,
                )
                for model_id in model_ids:
                    if (
                        model_id == "b0_empirical_v1"
                        and block_id != baseline_block_id
                    ):
                        continue
                    seed = effective_random_state(
                        base_random_state=int(random_state),
                        fold_id=fold.fold_id,
                        training_venue=training_venue,
                        evaluation_venue=evaluation_venue,
                        transport_mode=transport_mode,
                        feature_block_id=block_id,
                        model_id=model_id,
                        purpose="PROBABILITY_HEADS",
                    )
                    fits: dict[str, _FittedHead] = {}
                    calibrators: dict[
                        str,
                        BinaryPlattCalibrator
                        | DirectionTemperatureCalibrator,
                    ] = {}
                    raw: dict[str, np.ndarray] = {}
                    calibrated: dict[str, np.ndarray] = {}
                    for head in ("S_H", "O_H_GIVEN_S", "DIRECTION"):
                        fitted, training_weights = _fit_head(
                            train,
                            bundle.train_matrix,
                            model_id=model_id,
                            block_id=block_id,
                            head=head,
                            random_state=seed,
                        )
                        fits[head] = fitted
                        support_rows.append(
                            _support_row(
                                fold,
                                training_venue=training_venue,
                                evaluation_venue=evaluation_venue,
                                transport_mode=transport_mode,
                                block_id=block_id,
                                fitted=fitted,
                                train=train,
                                cohort_authority_sha256=cohort_authority_sha256,
                            )
                        )
                        for game_id, values in training_weights.groupby(
                            train["game_id"], sort=True
                        ):
                            weight_rows.append(
                                {
                                    "fold_id": fold.fold_id,
                                    "venue": evaluation_venue,
                                    "training_venue": training_venue,
                                    "calibration_venue": training_venue,
                                    "transport_mode": transport_mode,
                                    "partition": "TRAINING",
                                    "head": head,
                                    "game_id": game_id,
                                    "total_weight": float(values.sum()),
                                }
                            )
                        raw[head] = _predict_head(
                            fitted, validation, bundle.validation_matrix
                        )
                        calibrator = _prequential_calibrator(
                            train,
                            feature_frame,
                            model_id=model_id,
                            block_id=block_id,
                            head=head,
                            random_state=seed,
                            feature_blocks=feature_blocks,
                        )
                        calibrators[head] = calibrator
                        calibrated[head] = (
                            _calibrate_direction(calibrator, raw[head])
                            if head == "DIRECTION"
                            else _calibrate_binary(calibrator, raw[head])
                        )
                    raw["S_H"] = _cumulative_survival_product(
                        validation,
                        raw["S_H"],
                    )
                    calibrated["S_H"] = _cumulative_survival_product(
                        validation,
                        calibrated["S_H"],
                    )
                    prediction_rows.extend(
                        _prediction_rows(
                            validation,
                            fold=fold,
                            training_venue=training_venue,
                            evaluation_venue=evaluation_venue,
                            transport_mode=transport_mode,
                            block_id=block_id,
                            model_id=model_id,
                            feature_block_sha256=feature_block_hash,
                            fold_sha256=fold_hash,
                            training_data_sha256=training_data_sha256,
                            training_game_ids=actual_training_game_ids,
                            validation_game_ids=actual_validation_game_ids,
                            cohort_authority_sha256=cohort_authority_sha256,
                            bundle=bundle,
                            fits=fits,
                            calibrators=calibrators,
                            raw=raw,
                            calibrated=calibrated,
                            validation_weights=validation_weights,
                        )
                    )
                if include_magnitude:
                    output, support = _magnitude_outputs(
                        train=train,
                        validation=validation,
                        bundle=bundle,
                        fold=fold,
                        training_venue=training_venue,
                        evaluation_venue=evaluation_venue,
                        transport_mode=transport_mode,
                        block_id=block_id,
                        random_state=effective_random_state(
                            base_random_state=int(random_state),
                            fold_id=fold.fold_id,
                            training_venue=training_venue,
                            evaluation_venue=evaluation_venue,
                            transport_mode=transport_mode,
                            feature_block_id=block_id,
                            model_id=QUANTILE_MODEL_ID,
                            purpose="MAGNITUDE",
                        ),
                        support_contract=quantile_support_contract,
                        validation_weights=validation_weights["MAGNITUDE"],
                        training_game_ids=actual_training_game_ids,
                        cohort_authority_sha256=cohort_authority_sha256,
                    )
                    quantile_rows.extend(output)
                    support_rows.extend(support)
    predictions = pd.DataFrame(prediction_rows)
    quantiles = pd.DataFrame(quantile_rows)
    metrics = pd.DataFrame(
        [
            *_probability_metric_rows(predictions),
            *_magnitude_metric_rows(quantiles),
        ]
    )
    support = pd.DataFrame(support_rows)
    weights = pd.DataFrame(weight_rows).drop_duplicates().reset_index(drop=True)
    sort_predictions = [
        "fold_id",
        "venue",
        "training_venue",
        "transport_mode",
        "feature_block_id",
        "model_id",
        "game_id",
        "atomic_information_episode_id",
        "landmark_seconds",
        "endpoint_seconds",
    ]
    predictions = predictions.sort_values(
        sort_predictions, kind="mergesort"
    ).reset_index(drop=True)
    if not quantiles.empty:
        quantiles = quantiles.sort_values(
            [
                "fold_id",
                "venue",
                "training_venue",
                "transport_mode",
                "feature_block_id",
                "model_id",
                "game_id",
                "atomic_information_episode_id",
                "direction_condition",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
    if not metrics.empty:
        metrics = metrics.sort_values(
            [
                "fold_id",
                "venue",
                "training_venue",
                "transport_mode",
                "feature_block_id",
                "model_id",
                "head",
                "metric_scope",
                "metric_name",
                "game_id",
            ],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    if not support.empty:
        support = support.sort_values(
            [
                "fold_id",
                "venue",
                "training_venue",
                "transport_mode",
                "feature_block_id",
                "model_id",
                "head",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
    if not weights.empty:
        weights = weights.sort_values(
            [
                "fold_id",
                "venue",
                "training_venue",
                "transport_mode",
                "partition",
                "head",
                "game_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
    return X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=quantiles,
        fold_metrics=metrics,
        support_audit=support,
        weight_audit=weights,
        run_config_sha256=run_hash,
        run_config=run_config,
    )


def run_x15_walk_forward(
    panel: object,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    feature_block_ids: Sequence[str] = tuple(FEATURE_BLOCKS),
    fold_ids: Sequence[str] | None = None,
    transport_pairs: Sequence[tuple[str, str]] = (),
    include_magnitude: bool = True,
    quantile_support_contract: QuantileSupportContract = QuantileSupportContract(),
    random_state: int = RANDOM_STATE,
) -> X15ModelRun:
    """Run the confirmatory VenueReactionPanelV3 B0--B4 OOF contract."""

    return _run_x15_walk_forward_engine(
        _panel_frame(panel),
        feature_blocks=FEATURE_BLOCKS,
        baseline_block_id="B0",
        model_ids=model_ids,
        feature_block_ids=feature_block_ids,
        fold_ids=fold_ids,
        transport_pairs=transport_pairs,
        include_magnitude=include_magnitude,
        quantile_support_contract=quantile_support_contract,
        random_state=random_state,
    )


def _validate_diagnostic_decision_payload(
    decoded: Mapping[str, object],
) -> None:
    if decoded.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise X15ModelInputError(
            "diagnostic decision schema_version must be "
            f"{DIAGNOSTIC_SCHEMA_VERSION}"
        )
    if decoded.get("target_contract") != DIAGNOSTIC_TARGET_CONTRACT:
        raise X15ModelInputError(
            "diagnostic decision target_contract must be "
            f"{DIAGNOSTIC_TARGET_CONTRACT}"
        )
    decision_threshold = decoded.get("direction_threshold_probability")
    if (
        isinstance(decision_threshold, bool)
        or not isinstance(decision_threshold, (int, float))
        or not math.isfinite(float(decision_threshold))
        or float(decision_threshold)
        != DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
    ):
        raise X15ModelInputError(
            "diagnostic decision direction threshold is frozen at 0.01"
        )
    if decoded.get("direction_threshold_semantics") != (
        DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
    ):
        raise X15ModelInputError(
            "diagnostic decision direction threshold must remain fixed "
            "cross-venue research materiality, not a tick"
        )
    forbidden_tick_keys = {"tick_size", "tick_rule_id"} & set(
        _walk_keys(decoded)
    )
    if forbidden_tick_keys:
        raise X15ModelInputError(
            "diagnostic decision features cannot contain tick_size or "
            "tick_rule_id"
        )


def _diagnostic_panel_frame(
    panel: object,
    *,
    decision_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    if isinstance(panel, pd.DataFrame):
        source = panel
    elif hasattr(panel, "panel") and isinstance(panel.panel, pd.DataFrame):
        source = panel.panel
    else:
        raise X15ModelInputError(
            "diagnostic input must be HistoricalTradesOnlyProbabilityPanelV2 "
            "or its panel DataFrame"
        )
    required = set(DIAGNOSTIC_MODEL_INPUT_COLUMNS)
    missing = required - set(source.columns)
    if missing:
        raise X15ModelInputError(
            f"{DIAGNOSTIC_SCHEMA_VERSION} missing: {sorted(missing)}"
        )
    frame = source.loc[:, list(DIAGNOSTIC_MODEL_INPUT_COLUMNS)].copy(
        deep=True
    )
    if not frame["schema_version"].eq(DIAGNOSTIC_SCHEMA_VERSION).all():
        raise X15ModelInputError(
            f"diagnostic schema_version must be {DIAGNOSTIC_SCHEMA_VERSION}"
        )
    if not frame["target_contract"].eq(DIAGNOSTIC_TARGET_CONTRACT).all():
        raise X15ModelInputError(
            "diagnostic target_contract must be "
            f"{DIAGNOSTIC_TARGET_CONTRACT}"
        )
    if not frame["claim_boundary"].eq(DIAGNOSTIC_CLAIM_BOUNDARY).all():
        raise X15ModelInputError(
            f"diagnostic claim_boundary must be {DIAGNOSTIC_CLAIM_BOUNDARY}"
        )
    threshold = pd.to_numeric(
        frame["direction_threshold_probability"].drop_duplicates(),
        errors="coerce",
    )
    if (
        threshold.isna().any()
        or not np.isfinite(threshold.to_numpy(dtype=float)).all()
        or not threshold.eq(
            DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
        ).all()
    ):
        raise X15ModelInputError(
            "diagnostic direction threshold is frozen at 0.01"
        )
    if not frame["direction_threshold_semantics"].eq(
        DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
    ).all():
        raise X15ModelInputError(
            "diagnostic direction threshold must remain fixed cross-venue "
            "research materiality, not a tick"
        )
    if not frame["venue_tick_support"].eq(
        DIAGNOSTIC_VENUE_TICK_SUPPORT
    ).all():
        raise X15ModelInputError(
            "diagnostic venue_tick_support must be UNSUPPORTED"
        )
    if not frame["market_continuity_support"].eq(
        DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT
    ).all():
        raise X15ModelInputError(
            "diagnostic market_continuity_support must be UNKNOWN"
        )
    frame["decision_eligible"] = _strict_bool(frame, "decision_eligible")
    frame["target_eligible"] = _strict_bool(frame, "target_eligible")
    frame["sports_clean_l"] = _strict_bool(frame, "sports_clean_l")
    frame["sports_clean_h"] = _strict_bool(frame, "sports_clean_h")
    frame["actual_trade_observed_h"] = _strict_bool(
        frame, "actual_trade_observed_h"
    )
    landmark_reasons = frame["sports_clean_l_reason"]
    endpoint_reasons = frame["sports_clean_reason"]
    if landmark_reasons.isna().any() or any(
        type(value) is not str or not value.strip()
        for value in landmark_reasons.drop_duplicates()
    ):
        raise X15ModelInputError("sports_clean_l_reason must be nonempty")
    if endpoint_reasons.isna().any() or any(
        type(value) is not str or not value.strip()
        for value in endpoint_reasons.drop_duplicates()
    ):
        raise X15ModelInputError("sports_clean_reason must be nonempty")
    if (
        frame["decision_eligible"] & ~frame["sports_clean_l"]
    ).any():
        raise X15ModelInputError(
            "diagnostic decision_eligible requires sports_clean_l"
        )
    expected_target = (
        frame["decision_eligible"]
        & frame["sports_clean_h"]
        & frame["actual_trade_observed_h"]
    )
    if not frame["target_eligible"].eq(expected_target).all():
        raise X15ModelInputError(
            "diagnostic target_eligible must equal decision_eligible AND "
            "sports_clean_h AND actual_trade_observed_h"
        )
    delta = pd.to_numeric(frame["delta_l_h"], errors="coerce")
    if (
        delta.loc[expected_target].isna().any()
        or not np.isfinite(
            delta.loc[expected_target].to_numpy(dtype=float)
        ).all()
        or delta.loc[~expected_target].notna().any()
    ):
        raise X15ModelInputError(
            "diagnostic delta_l_h must be finite exactly when target_eligible"
        )
    up = expected_target & delta.ge(
        DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
    )
    down = expected_target & delta.le(
        -DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
    )
    no_move = expected_target & ~up & ~down
    if (
        not frame.loc[up, "direction"].eq("UP").all()
        or not frame.loc[down, "direction"].eq("DOWN").all()
        or not frame.loc[no_move, "direction"].eq("NO_MOVE").all()
        or not frame.loc[~expected_target, "direction"]
        .eq("UNOBSERVED")
        .all()
    ):
        raise X15ModelInputError(
            "diagnostic direction must follow fixed 0.01 materiality"
        )
    magnitude = pd.to_numeric(
        frame["conditional_magnitude"], errors="coerce"
    )
    moving = up | down
    if (
        magnitude.loc[moving].isna().any()
        or not np.isclose(
            magnitude.loc[moving].to_numpy(dtype=float),
            delta.loc[moving].abs().to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ).all()
        or magnitude.loc[~moving].notna().any()
    ):
        raise X15ModelInputError(
            "diagnostic conditional_magnitude must equal abs(delta_l_h) "
            "only for UP/DOWN"
        )
    frame["s_h"] = frame["sports_clean_h"]
    frame["o_h_given_s"] = frame["actual_trade_observed_h"].where(
        frame["sports_clean_h"], pd.NA
    )
    frame["schema_version"] = "VenueReactionPanelV3"
    frame.drop(
        columns=sorted(set(frame.columns) - _PANEL_REQUIRED), inplace=True
    )
    for column in (
        "schema_version",
        "cohort_authority_sha256",
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
        "direction",
        "decision_features_json",
        "decision_feature_sha256",
    ):
        if not isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("category")
    frame = _panel_frame(frame, copy_input=False)
    if decision_only:
        frame = frame.loc[frame["decision_eligible"]].copy()
        if frame.empty:
            raise X15ModelInputError(
                "diagnostic partitions have no decision-eligible rows"
            )
    parsed_by_sha256 = _parse_unique_decision_features(
        frame, diagnostic_contract=True
    )
    return frame, parsed_by_sha256


def prepare_x15_historical_trades_diagnostic_partitions(
    partitions: Iterable[VerifiedDiagnosticPanelPartition],
) -> X15PreparedDiagnosticPanel:
    """Adapt verified game partitions and retain only compact decision rows."""

    if isinstance(partitions, (pd.DataFrame, VerifiedDiagnosticPanelPartition)):
        raise X15ModelInputError(
            "preparation requires an iterator of "
            "VerifiedDiagnosticPanelPartition values"
        )
    prepared_frames: list[pd.DataFrame] = []
    parsed_by_sha256: dict[str, Mapping[str, object]] = {}
    decision_fingerprint_by_sha256: dict[str, bytes] = {}
    game_ids: set[str] = set()
    expected_lineage: tuple[object, ...] | None = None
    source_row_count = 0
    for partition in partitions:
        if not isinstance(partition, VerifiedDiagnosticPanelPartition):
            raise X15ModelInputError(
                "preparation requires VerifiedDiagnosticPanelPartition values"
            )
        if (
            type(partition.game_id) is not str
            or not partition.game_id.strip()
            or partition.game_id in game_ids
            or type(partition.batch_game_count) is not int
            or partition.batch_game_count <= 0
        ):
            raise X15ModelInputError(
                "verified diagnostic partition identity is invalid"
            )
        for label, value in (
            (
                "batch_manifest_file_sha256",
                partition.batch_manifest_file_sha256,
            ),
            ("batch_sha256", partition.batch_sha256),
            ("game_manifest_sha256", partition.game_manifest_sha256),
            (
                "diagnostic_object_sha256",
                partition.diagnostic_object_sha256,
            ),
            (
                "cohort_authority_sha256",
                partition.cohort_authority_sha256,
            ),
            ("cohort_mapping_sha256", partition.cohort_mapping_sha256),
        ):
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise X15ModelInputError(
                    f"verified diagnostic partition {label} is invalid"
                )
        lineage = (
            partition.batch_game_count,
            partition.batch_manifest_file_sha256,
            partition.batch_sha256,
            partition.cohort_authority_sha256,
            partition.cohort_mapping_sha256,
        )
        if expected_lineage is None:
            expected_lineage = lineage
        elif lineage != expected_lineage:
            raise X15ModelInputError(
                "verified diagnostic partitions have conflicting batch lineage"
            )
        partition_games = set(
            partition.panel["game_id"].astype(str)
        ) if (
            isinstance(partition.panel, pd.DataFrame)
            and "game_id" in partition.panel
        ) else set()
        if partition_games != {partition.game_id}:
            raise X15ModelInputError(
                "verified diagnostic partition crosses game boundaries"
            )
        source_row_count += len(partition.panel)
        adapted, parsed = _diagnostic_panel_frame(
            partition.panel,
            decision_only=True,
        )
        if not adapted["cohort_authority_sha256"].astype(str).eq(
            partition.cohort_authority_sha256
        ).all():
            raise X15ModelInputError(
                "verified diagnostic partition authority differs from panel"
            )
        for digest, decision_json in adapted.loc[
            :,
            ["decision_feature_sha256", "decision_features_json"],
        ].drop_duplicates().itertuples(index=False, name=None):
            digest = str(digest)
            decision_json = str(decision_json)
            fingerprint = hashlib.sha512(
                decision_json.encode("utf-8")
            ).digest()
            prior_fingerprint = decision_fingerprint_by_sha256.get(digest)
            if (
                prior_fingerprint is not None
                and prior_fingerprint != fingerprint
            ):
                raise X15ModelInputError(
                    "digest maps to conflicting decision payloads"
                )
            decision_fingerprint_by_sha256[digest] = fingerprint
        for digest, payload in parsed.items():
            compact_payload = _compact_parsed_decision_payload(
                payload,
                feature_blocks=DIAGNOSTIC_FEATURE_BLOCKS,
            )
            existing = parsed_by_sha256.get(digest)
            if (
                existing is not None
                and _canonical(existing) != _canonical(compact_payload)
            ):
                raise X15ModelInputError(
                    "digest maps to conflicting decision payloads"
                )
            parsed_by_sha256[digest] = compact_payload
        adapted.drop(columns=["decision_features_json"], inplace=True)
        prepared_frames.append(adapted)
        game_ids.add(partition.game_id)
        del partition
    if expected_lineage is None or not prepared_frames:
        raise X15ModelInputError(
            "verified diagnostic partition iterator must be nonempty"
        )
    expected_game_count = int(expected_lineage[0])
    if len(game_ids) != expected_game_count:
        raise X15ModelInputError(
            "verified diagnostic partition iterator is incomplete"
        )
    frame = pd.concat(
        prepared_frames,
        ignore_index=True,
    )
    prepared_frames.clear()
    if frame.duplicated(list(_PANEL_GRAIN)).any():
        raise X15ModelInputError(
            "verified diagnostic partitions duplicate panel grain"
        )
    for column in (
        "schema_version",
        "cohort_authority_sha256",
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
        "direction",
        "decision_feature_sha256",
    ):
        if not isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("category")
    frame = frame.sort_values(
        list(_PANEL_GRAIN),
        kind="mergesort",
    ).reset_index(drop=True)
    frame["_source_row_id"] = np.arange(len(frame), dtype=int)
    build_x15_week_folds(frame)
    observed_digests = set(
        frame["decision_feature_sha256"].astype(str)
    )
    if observed_digests != set(parsed_by_sha256):
        raise X15ModelInputError(
            "prepared decision cache does not match prepared rows"
        )
    return X15PreparedDiagnosticPanel(
        frame=frame,
        parsed_by_sha256=MappingProxyType(dict(parsed_by_sha256)),
        game_ids=tuple(sorted(game_ids)),
        partition_count=len(game_ids),
        source_row_count=source_row_count,
        batch_manifest_file_sha256=str(expected_lineage[1]),
        batch_sha256=str(expected_lineage[2]),
        cohort_authority_sha256=str(expected_lineage[3]),
        cohort_mapping_sha256=str(expected_lineage[4]),
    )


def _diagnostic_metadata() -> dict[str, object]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "analysis_scope": DIAGNOSTIC_ANALYSIS_SCOPE,
        "direction_threshold_probability": (
            DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
        ),
        "direction_threshold_semantics": (
            DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
        ),
        "venue_tick_support": DIAGNOSTIC_VENUE_TICK_SUPPORT,
        "market_continuity_support": (
            DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT
        ),
        "claim_eligible": False,
    }


def _stamp_diagnostic_run(result: X15ModelRun) -> X15ModelRun:
    metadata = _diagnostic_metadata()
    run_config = dict(result.run_config or {})
    run_config.update(metadata)
    run_config["feature_blocks"] = DIAGNOSTIC_FEATURE_BLOCKS
    run_config_sha256 = _sha256(run_config)
    lineage = {**metadata, "run_config_sha256": run_config_sha256}

    def stamp(frame: pd.DataFrame) -> pd.DataFrame:
        stamped = frame.copy(deep=True)
        for column, value in lineage.items():
            stamped[column] = value
        return stamped

    return X15ModelRun(
        oof_predictions=stamp(result.oof_predictions),
        conditional_quantiles=stamp(result.conditional_quantiles),
        fold_metrics=stamp(result.fold_metrics),
        support_audit=stamp(result.support_audit),
        weight_audit=stamp(result.weight_audit),
        run_config_sha256=run_config_sha256,
        run_config=MappingProxyType(run_config),
    )


def run_x15_historical_trades_diagnostic_walk_forward(
    panel: object,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    feature_block_ids: Sequence[str] = tuple(DIAGNOSTIC_FEATURE_BLOCKS),
    fold_ids: Sequence[str] | None = None,
    transport_pairs: Sequence[tuple[str, str]] = (),
    include_magnitude: bool = True,
    quantile_support_contract: QuantileSupportContract = QuantileSupportContract(),
    random_state: int = RANDOM_STATE,
) -> X15ModelRun:
    """Run an explicitly non-confirmatory historical-trades-only OOF study."""

    if isinstance(panel, X15PreparedDiagnosticPanel):
        adapted = panel.frame
        parsed_by_sha256 = panel.parsed_by_sha256
    else:
        adapted, parsed_by_sha256 = _diagnostic_panel_frame(panel)
    result = _run_x15_walk_forward_engine(
        adapted,
        feature_blocks=DIAGNOSTIC_FEATURE_BLOCKS,
        baseline_block_id="D0",
        parsed_by_sha256=parsed_by_sha256,
        model_ids=model_ids,
        feature_block_ids=feature_block_ids,
        fold_ids=fold_ids,
        transport_pairs=transport_pairs,
        include_magnitude=include_magnitude,
        quantile_support_contract=quantile_support_contract,
        random_state=random_state,
    )
    return _stamp_diagnostic_run(result)


__all__ = [
    "DIAGNOSTIC_ANALYSIS_SCOPE",
    "DIAGNOSTIC_CLAIM_BOUNDARY",
    "DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY",
    "DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS",
    "DIAGNOSTIC_FEATURE_BLOCKS",
    "DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT",
    "DIAGNOSTIC_MODEL_INPUT_COLUMNS",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DIAGNOSTIC_TARGET_CONTRACT",
    "DIAGNOSTIC_VENUE_TICK_SUPPORT",
    "EFFECTIVE_SEED_CONTRACT_ID",
    "EFFECTIVE_SEED_COORDINATE_FIELDS",
    "EFFECTIVE_SEED_MODULUS",
    "FEATURE_BLOCKS",
    "MODEL_IDS",
    "QUANTILE_MODEL_ID",
    "RANDOM_STATE",
    "SURVIVAL_PROBABILITY_CONTRACT",
    "X15ModelInputError",
    "X15ModelRun",
    "X15PreparedDiagnosticPanel",
    "X15WeekFold",
    "build_x15_week_folds",
    "effective_random_state",
    "hierarchical_sample_weights",
    "prepare_x15_historical_trades_diagnostic_partitions",
    "run_x15_historical_trades_diagnostic_walk_forward",
    "run_x15_walk_forward",
]
