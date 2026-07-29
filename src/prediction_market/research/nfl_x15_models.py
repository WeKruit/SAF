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
from dataclasses import dataclass
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


RANDOM_STATE: Final[int] = 20260728
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


def build_x15_week_folds(landmarks: pd.DataFrame) -> tuple[X15WeekFold, ...]:
    """Build the five frozen expanding complete-game folds."""

    if not isinstance(landmarks, pd.DataFrame):
        raise X15ModelInputError("landmarks must be a DataFrame")
    missing = {"game_id", "nfl_week"} - set(landmarks.columns)
    if missing:
        raise X15ModelInputError(f"fold input missing columns: {sorted(missing)}")
    if landmarks.empty:
        raise X15ModelInputError("fold input must be nonempty")
    games = landmarks["game_id"].astype(str)
    if games.str.strip().eq("").any():
        raise X15ModelInputError("game_id must be nonempty")
    numeric = pd.to_numeric(landmarks["nfl_week"], errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise X15ModelInputError("nfl_week must contain integer weeks")
    weeks = numeric.astype(int)
    if not weeks.between(1, 12).all():
        raise X15ModelInputError(
            "development folds accept only NFL weeks 1 through 12"
        )
    game_weeks = (
        pd.DataFrame({"game_id": games, "nfl_week": weeks})
        .drop_duplicates()
        .sort_values(["nfl_week", "game_id"], kind="mergesort")
    )
    if game_weeks.groupby("game_id")["nfl_week"].nunique().gt(1).any():
        raise X15ModelInputError("each game_id must map to one NFL week")
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
    eligible = pd.Series(eligibility, index=frame.index, dtype="boolean")
    if eligible.isna().any():
        raise X15ModelInputError("weight eligibility must not be missing")
    result = pd.Series(0.0, index=frame.index, dtype=float)
    selected = frame.loc[eligible.astype(bool)]
    for _, game in selected.groupby("game_id", sort=True):
        episodes = tuple(
            sorted(game["atomic_information_episode_id"].astype(str).unique())
        )
        if not episodes:
            continue
        episode_share = 1.0 / len(episodes)
        for episode in episodes:
            indexes = game.index[
                game["atomic_information_episode_id"].astype(str).eq(episode)
            ]
            result.loc[indexes] = episode_share / len(indexes)
    return result


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
    if values.isna().any() or not values.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise X15ModelInputError(f"{column} must contain nonmissing booleans")
    return values.astype(bool)


def _nullable_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    supplied = values.notna()
    if not values.loc[supplied].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise X15ModelInputError(f"{column} must contain booleans or null")
    return values.astype("boolean")


def _panel_frame(panel: object) -> pd.DataFrame:
    if isinstance(panel, pd.DataFrame):
        frame = panel.copy(deep=True)
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
    if frame.duplicated(list(_PANEL_GRAIN)).any():
        raise X15ModelInputError("VenueReactionPanelV3 grain is not unique")
    for column in (
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
    ):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq(
            ""
        ).any():
            raise X15ModelInputError(f"{column} must be nonempty")
        frame[column] = frame[column].astype(str)
    frame["decision_eligible"] = _strict_bool(frame, "decision_eligible")
    frame["target_eligible"] = _strict_bool(frame, "target_eligible")
    frame["s_h"] = _nullable_bool(frame, "s_h")
    frame["o_h_given_s"] = _nullable_bool(frame, "o_h_given_s")
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
    frame["_source_row_id"] = np.arange(len(frame), dtype=int)
    build_x15_week_folds(frame)
    return frame


def _decision_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    all_multi_hot: set[str] = set()
    parsed_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        try:
            parsed = json.loads(row.decision_features_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise X15ModelInputError(
                "decision_features_json must be valid JSON objects"
            ) from exc
        if not isinstance(parsed, dict):
            raise X15ModelInputError(
                "decision_features_json must contain an object"
            )
        actual_hash = _sha256_text_exact(row.decision_features_json)
        if row.decision_feature_sha256 != actual_hash:
            raise X15ModelInputError("decision feature sha256 mismatch")
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
        all_multi_hot.update(f"multi_hot__{key}" for key in multi_hot)
        parsed_rows.append(parsed)
    for parsed in parsed_rows:
        record: dict[str, object] = {}
        facts = parsed.get("fact_features", {})
        multi_hot = parsed.get("multi_hot_features", {})
        for feature in FEATURE_BLOCKS["B4"]:
            if feature == "multi_hot__*":
                continue
            if feature.startswith("fact__"):
                record[feature] = facts.get(feature.removeprefix("fact__"))
            else:
                record[feature] = parsed.get(feature)
        for column in sorted(all_multi_hot):
            record[column] = bool(
                multi_hot.get(column.removeprefix("multi_hot__"), False)
            )
        records.append(record)
    return pd.DataFrame(records, index=frame.index)


def _sha256_text_exact(value: object) -> str:
    if type(value) is not str:
        raise X15ModelInputError("decision_features_json must be text")
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolved_columns(
    feature_frame: pd.DataFrame, block_id: str
) -> tuple[str, ...]:
    requested = FEATURE_BLOCKS[block_id]
    resolved: list[str] = []
    for column in requested:
        if column == "multi_hot__*":
            resolved.extend(
                sorted(
                    value
                    for value in feature_frame.columns
                    if value.startswith("multi_hot__")
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
) -> _FeatureBundle:
    columns = _resolved_columns(feature_frame, block_id)
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
        train[column] = pd.to_numeric(train[column], errors="coerce")
        validation[column] = pd.to_numeric(validation[column], errors="coerce")
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
        return frame["s_h"].notna()
    if head == "O_H_GIVEN_S":
        return frame["s_h"].eq(True) & frame["o_h_given_s"].notna()
    if head == "DIRECTION":
        return frame["target_eligible"] & frame["_direction_truth"].notna()
    raise AssertionError(head)


def _head_labels(frame: pd.DataFrame, head: str) -> np.ndarray:
    if head == "S_H":
        return frame["s_h"].astype(bool).astype(int).to_numpy()
    if head == "O_H_GIVEN_S":
        return frame["o_h_given_s"].astype(bool).astype(int).to_numpy()
    return frame["_direction_truth"].astype(str).to_numpy()


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
                    row.s_h
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
    venue: str,
    block_id: str,
    fitted: _FittedHead,
    train: pd.DataFrame,
) -> dict[str, object]:
    mask = _head_mask(train, fitted.head)
    return {
        "fold_id": fold.fold_id,
        "venue": venue,
        "feature_block_id": block_id,
        "model_id": fitted.model_id,
        "head": fitted.head,
        "support_status": fitted.status,
        "support_reason": fitted.reason,
        "training_row_count": int(mask.sum()),
        "training_game_count": int(train.loc[mask, "game_id"].nunique()),
        "training_classes": fitted.classes,
        "binary_min_games": _BINARY_MIN_GAMES,
        "direction_min_games": _DIRECTION_MIN_GAMES,
        "training_sha256": fitted.training_sha256,
    }


def _prediction_rows(
    validation: pd.DataFrame,
    *,
    fold: X15WeekFold,
    venue: str,
    block_id: str,
    model_id: str,
    feature_block_sha256: str,
    fold_sha256: str,
    training_data_sha256: str,
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
                "game_id": source["game_id"],
                "nfl_week": int(source["nfl_week"]),
                "atomic_information_episode_id": source[
                    "atomic_information_episode_id"
                ],
                "venue": venue,
                "training_venue": venue,
                "actual_home_contract_id": source["actual_home_contract_id"],
                "landmark_seconds": int(source["landmark_seconds"]),
                "endpoint_seconds": int(source["endpoint_seconds"]),
                "fold_id": fold.fold_id,
                "train_weeks": fold.train_weeks,
                "validation_weeks": fold.validation_weeks,
                "training_game_ids": fold.train_game_ids,
                "validation_game_ids": fold.validation_game_ids,
                "preprocessor_fit_game_ids": tuple(
                    sorted(set(fold.train_game_ids))
                ),
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
    keys = ["fold_id", "venue", "model_id", "feature_block_id"]
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
                        "cluster_interval_low": math.nan,
                        "cluster_interval_high": math.nan,
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
                    "cluster_interval_low": (
                        float(np.quantile(finite, 0.025)) if len(finite) else math.nan
                    ),
                    "cluster_interval_high": (
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
    venue: str,
    block_id: str,
    random_state: int,
    support_contract: QuantileSupportContract,
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
    if training_features.empty:
        return [], []
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
                "venue": venue,
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
                "training_classes": (direction,),
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
                    "game_id": source["game_id"],
                    "nfl_week": int(source["nfl_week"]),
                    "atomic_information_episode_id": source[
                        "atomic_information_episode_id"
                    ],
                    "venue": venue,
                    "training_venue": venue,
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
                    "preprocessor_fit_game_ids": tuple(
                        sorted(set(fold.train_game_ids))
                    ),
                }
            )
    return output_rows, support_rows


def _magnitude_metric_rows(
    quantiles: pd.DataFrame,
) -> list[dict[str, object]]:
    if quantiles.empty:
        return []
    rows: list[dict[str, object]] = []
    keys = ["fold_id", "venue", "model_id", "feature_block_id"]
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
            )
            for metric_name, value in metrics.items():
                row = {
                    **dict(zip(keys, key, strict=True)),
                    "metric_scope": "GAME",
                    "game_id": game_id,
                    "head": "MAGNITUDE",
                    "metric_name": metric_name,
                    "metric_value": value,
                    "cluster_interval_low": math.nan,
                    "cluster_interval_high": math.nan,
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
                    "cluster_interval_low": float(np.quantile(finite, 0.025)),
                    "cluster_interval_high": float(np.quantile(finite, 0.975)),
                }
            )
    return rows


def run_x15_walk_forward(
    panel: object,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    feature_block_ids: Sequence[str] = tuple(FEATURE_BLOCKS),
    fold_ids: Sequence[str] | None = None,
    include_magnitude: bool = True,
    quantile_support_contract: QuantileSupportContract = QuantileSupportContract(),
    random_state: int = RANDOM_STATE,
) -> X15ModelRun:
    """Run fixed Stage B OOF folds, optionally sliced for resumable execution.

    Execution slicing never changes model parameters or support gates.  A
    confirmatory run uses all defaults.
    """

    frame = _panel_frame(panel)
    decision = frame.loc[frame["decision_eligible"]].copy()
    if decision.empty:
        raise X15ModelInputError("panel has no decision-eligible rows")
    feature_frame = _decision_feature_frame(decision)
    unknown_models = sorted(set(model_ids) - set(MODEL_IDS))
    unknown_blocks = sorted(set(feature_block_ids) - set(FEATURE_BLOCKS))
    if unknown_models:
        raise X15ModelInputError(f"unknown model_ids: {unknown_models}")
    if unknown_blocks:
        raise X15ModelInputError(f"unknown feature blocks: {unknown_blocks}")
    if len(set(model_ids)) != len(tuple(model_ids)):
        raise X15ModelInputError("model_ids must be unique")
    if len(set(feature_block_ids)) != len(tuple(feature_block_ids)):
        raise X15ModelInputError("feature_block_ids must be unique")
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
    run_config = {
        "model_ids": tuple(model_ids),
        "feature_block_ids": tuple(feature_block_ids),
        "fold_ids": selected_fold_ids,
        "include_magnitude": bool(include_magnitude),
        "random_state": int(random_state),
        "feature_blocks": FEATURE_BLOCKS,
        "xgb_params": _XGB_PARAMS,
        "quantile_support_contract": quantile_support_contract,
    }
    run_hash = _sha256(run_config)
    prediction_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    venues = tuple(sorted(decision["venue"].unique()))
    for fold_index, fold in enumerate(selected_folds):
        fold_hash = _sha256(
            {
                "fold_id": fold.fold_id,
                "train_weeks": fold.train_weeks,
                "validation_weeks": fold.validation_weeks,
                "train_game_ids": fold.train_game_ids,
                "validation_game_ids": fold.validation_game_ids,
            }
        )
        for venue_index, venue in enumerate(venues):
            train = decision[
                decision["game_id"].isin(fold.train_game_ids)
                & decision["venue"].eq(venue)
            ]
            validation = decision[
                decision["game_id"].isin(fold.validation_game_ids)
                & decision["venue"].eq(venue)
            ]
            if train.empty or validation.empty:
                continue
            training_data_sha256 = _training_data_hash(train)
            validation_weights = {
                head: hierarchical_sample_weights(
                    validation, _head_mask(validation, head)
                )
                for head in ("S_H", "O_H_GIVEN_S", "DIRECTION")
            }
            for head, weights in validation_weights.items():
                for game_id, values in weights.groupby(
                    validation["game_id"], sort=True
                ):
                    weight_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "venue": venue,
                            "partition": "VALIDATION",
                            "head": head,
                            "game_id": game_id,
                            "total_weight": float(values.sum()),
                        }
                    )
            for block_index, block_id in enumerate(feature_block_ids):
                feature_block_hash = _sha256(
                    {
                        "feature_block_id": block_id,
                        "features": FEATURE_BLOCKS[block_id],
                    }
                )
                bundle = _feature_bundle(
                    feature_frame,
                    train.index,
                    validation.index,
                    block_id=block_id,
                    train_frame=train,
                )
                for model_index, model_id in enumerate(model_ids):
                    if model_id == "b0_empirical_v1" and block_id != "B0":
                        continue
                    seed = (
                        int(random_state)
                        + fold_index * 1_000
                        + venue_index * 100
                        + block_index * 10
                        + model_index
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
                                venue=venue,
                                block_id=block_id,
                                fitted=fitted,
                                train=train,
                            )
                        )
                        for game_id, values in training_weights.groupby(
                            train["game_id"], sort=True
                        ):
                            weight_rows.append(
                                {
                                    "fold_id": fold.fold_id,
                                    "venue": venue,
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
                        )
                        calibrators[head] = calibrator
                        calibrated[head] = (
                            _calibrate_direction(calibrator, raw[head])
                            if head == "DIRECTION"
                            else _calibrate_binary(calibrator, raw[head])
                        )
                    prediction_rows.extend(
                        _prediction_rows(
                            validation,
                            fold=fold,
                            venue=venue,
                            block_id=block_id,
                            model_id=model_id,
                            feature_block_sha256=feature_block_hash,
                            fold_sha256=fold_hash,
                            training_data_sha256=training_data_sha256,
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
                        venue=venue,
                        block_id=block_id,
                        random_state=int(random_state) + fold_index + venue_index,
                        support_contract=quantile_support_contract,
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
                "feature_block_id",
                "model_id",
                "head",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
    if not weights.empty:
        weights = weights.sort_values(
            ["fold_id", "venue", "partition", "head", "game_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    return X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=quantiles,
        fold_metrics=metrics,
        support_audit=support,
        weight_audit=weights,
        run_config_sha256=run_hash,
    )


__all__ = [
    "FEATURE_BLOCKS",
    "MODEL_IDS",
    "QUANTILE_MODEL_ID",
    "RANDOM_STATE",
    "X15ModelInputError",
    "X15ModelRun",
    "X15WeekFold",
    "build_x15_week_folds",
    "hierarchical_sample_weights",
    "run_x15_walk_forward",
]
