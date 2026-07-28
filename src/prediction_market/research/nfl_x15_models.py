"""Landmark-only NFL X-15 walk-forward model core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


class X15ModelInputError(ValueError):
    """The supplied landmark rows cannot support the frozen X-15 evaluation."""


_FOLD_WEEK_WINDOWS: Final[
    tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
] = (
    ((1, 2), (3, 4)),
    (tuple(range(1, 5)), (5, 6)),
    (tuple(range(1, 7)), (7, 8)),
    (tuple(range(1, 9)), (9, 10)),
    (tuple(range(1, 11)), (11, 12)),
)
DIRECTION_CLASSES: Final[tuple[str, str, str]] = ("DOWN", "NO_MOVE", "UP")
MODEL_IDS: Final[tuple[str, str, str, str]] = (
    "reference_gap_baseline_v1",
    "multinomial_logistic_v1",
    "huber_regression_v1",
    "xgboost_regression_v1",
)
_SUPPORTED_VENUES: Final[frozenset[str]] = frozenset(
    {"polymarket", "kalshi"}
)
_REQUIRED_LANDMARK_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "game_id",
        "episode_id",
        "venue",
        "nfl_week",
        "landmark_seconds",
        "endpoint_seconds",
        "target_delta",
        "reference_gap_at_landmark",
        "factor_id",
        "decision_eligible",
        "target_eligible",
        "eligible",
        "exclusion_reasons",
        "is_matched_control",
        "decision_eligible",
        "target_eligible",
    }
)
_LEAKAGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "target_delta",
        "eligible",
        "exclusion_reasons",
        "game_id",
        "episode_id",
        "venue",
        "is_matched_control",
        "direction_label",
    }
)


@dataclass(frozen=True)
class X15WeekFold:
    """One frozen expanding NFL-week fold."""

    fold_id: int
    train_weeks: tuple[int, ...]
    validation_weeks: tuple[int, ...]
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]


@dataclass(frozen=True)
class X15ModelRun:
    """Development-only out-of-fold predictions and fold-level metrics."""

    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame


def build_x15_week_folds(landmarks: pd.DataFrame) -> tuple[X15WeekFold, ...]:
    """Build the five registered expanding folds without splitting a game."""

    if not isinstance(landmarks, pd.DataFrame) or landmarks.empty:
        raise X15ModelInputError("landmarks must be a nonempty DataFrame")
    missing = {"game_id", "nfl_week"} - set(landmarks.columns)
    if missing:
        raise X15ModelInputError(f"missing fold columns: {sorted(missing)}")
    if landmarks["game_id"].isna().any() or (
        landmarks["game_id"].astype(str).str.strip().eq("")
    ).any():
        raise X15ModelInputError("game_id must be nonempty")
    numeric_week = pd.to_numeric(landmarks["nfl_week"], errors="coerce")
    if numeric_week.isna().any() or not np.equal(
        numeric_week.to_numpy(dtype=float),
        numeric_week.to_numpy(dtype=int),
    ).all():
        raise X15ModelInputError("nfl_week must contain integer weeks")
    if not numeric_week.astype(int).between(1, 12).all():
        raise X15ModelInputError(
            "X-15 development folds only accept NFL weeks 1 through 12"
        )
    game_weeks = (
        pd.DataFrame(
            {
                "game_id": landmarks["game_id"].astype(str),
                "nfl_week": numeric_week.astype(int),
            }
        )
        .drop_duplicates()
        .sort_values(["nfl_week", "game_id"], kind="mergesort")
    )
    if game_weeks.groupby("game_id")["nfl_week"].nunique().gt(1).any():
        raise X15ModelInputError("each game_id must belong to one NFL week")

    folds: list[X15WeekFold] = []
    for fold_id, (train_weeks, validation_weeks) in enumerate(
        _FOLD_WEEK_WINDOWS,
        start=1,
    ):
        train_games = tuple(
            game_weeks.loc[
                game_weeks["nfl_week"].isin(train_weeks), "game_id"
            ].tolist()
        )
        validation_games = tuple(
            game_weeks.loc[
                game_weeks["nfl_week"].isin(validation_weeks), "game_id"
            ].tolist()
        )
        if not train_games or not validation_games:
            raise X15ModelInputError(
                f"fold {fold_id} has no train or validation games"
            )
        if set(train_games).intersection(validation_games):
            raise X15ModelInputError(f"fold {fold_id} splits a game")
        folds.append(
            X15WeekFold(
                fold_id=fold_id,
                train_weeks=train_weeks,
                validation_weeks=validation_weeks,
                train_game_ids=train_games,
                validation_game_ids=validation_games,
            )
        )
    return tuple(folds)


def direction_labels(
    target_delta: pd.Series,
    *,
    noise_threshold: float,
) -> pd.Series:
    """Classify signed probability change outside the fold noise band."""

    if not isinstance(target_delta, pd.Series):
        raise X15ModelInputError("target_delta must be a Series")
    if (
        not isinstance(noise_threshold, (float, int))
        or isinstance(noise_threshold, bool)
        or not np.isfinite(float(noise_threshold))
        or float(noise_threshold) < 0.01
    ):
        raise X15ModelInputError("noise_threshold must be finite and >= 0.01")
    values = pd.to_numeric(target_delta, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise X15ModelInputError("target_delta must be finite")
    threshold = float(noise_threshold)
    labels = np.full(len(values), "NO_MOVE", dtype=object)
    labels[values < -threshold] = "DOWN"
    labels[values > threshold] = "UP"
    return pd.Series(labels, index=target_delta.index, dtype="string")


def training_noise_threshold(
    training_landmarks: pd.DataFrame,
    *,
    matched_control_column: str = "is_matched_control",
) -> float:
    """Return max(1pp, training-control absolute target 95th percentile)."""

    if not isinstance(training_landmarks, pd.DataFrame) or training_landmarks.empty:
        raise X15ModelInputError(
            "training_landmarks must be a nonempty DataFrame"
        )
    missing = {"target_delta", matched_control_column} - set(
        training_landmarks.columns
    )
    if missing:
        raise X15ModelInputError(
            f"missing training-noise columns: {sorted(missing)}"
        )
    control_mask = training_landmarks[matched_control_column]
    if not pd.api.types.is_bool_dtype(control_mask.dtype):
        raise X15ModelInputError(
            f"{matched_control_column} must contain booleans"
        )
    if control_mask.isna().any():
        raise X15ModelInputError(
            f"{matched_control_column} control membership must be present"
        )
    controls = pd.to_numeric(
        training_landmarks.loc[control_mask, "target_delta"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if controls.size == 0 or not np.isfinite(controls).all():
        raise X15ModelInputError(
            "training fold requires finite matched-control targets"
        )
    return max(
        0.01,
        float(np.quantile(np.abs(controls), 0.95, method="linear")),
    )


def _has_exclusion_reasons(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "[]"}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return len(value) > 0
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return False
    raise X15ModelInputError(
        "exclusion_reasons must contain sequences, empty strings, or nulls"
    )


def _validate_landmarks(
    landmarks: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(landmarks, pd.DataFrame) or landmarks.empty:
        raise X15ModelInputError("landmarks must be a nonempty DataFrame")
    missing = _REQUIRED_LANDMARK_COLUMNS - set(landmarks.columns)
    if missing:
        raise X15ModelInputError(
            f"missing landmark columns: {sorted(missing)}"
        )
    if not isinstance(feature_columns, Sequence) or isinstance(
        feature_columns, (str, bytes)
    ):
        raise X15ModelInputError("feature_columns must be a sequence")
    features = tuple(str(column) for column in feature_columns)
    if not features or len(set(features)) != len(features):
        raise X15ModelInputError(
            "feature_columns must be nonempty and unique"
        )
    missing_features = set(features) - set(landmarks.columns)
    if missing_features:
        raise X15ModelInputError(
            f"missing feature columns: {sorted(missing_features)}"
        )
    leaked = sorted(set(features).intersection(_LEAKAGE_COLUMNS))
    reasons = landmarks["exclusion_reasons"].map(_has_exclusion_reasons)
    decision_eligible = landmarks["decision_eligible"]
    target_eligible = landmarks["target_eligible"]
    eligible = landmarks["eligible"]
    for label, values in (
        ("decision_eligible", decision_eligible),
        ("target_eligible", target_eligible),
        ("eligible", eligible),
    ):
        if not pd.api.types.is_bool_dtype(values.dtype):
            raise X15ModelInputError(f"{label} must contain booleans")
        if values.isna().any():
            raise X15ModelInputError(f"{label} must be present for every row")
    if not eligible.equals(target_eligible):
        raise X15ModelInputError(
            "eligible must be the exact alias of target_eligible"
        )
    if (target_eligible & ~decision_eligible).any():
        raise X15ModelInputError(
            "target_eligible rows must also be decision_eligible"
        )
    if (target_eligible & reasons).any():
        raise X15ModelInputError(
            "target-eligible rows cannot contain exclusion reasons"
        )
    if leaked:
        raise X15ModelInputError(
            f"feature_columns contain leakage fields: {leaked}"
        )
    venue = landmarks["venue"].astype(str)
    unknown_venues = sorted(set(venue).difference(_SUPPORTED_VENUES))
    if unknown_venues:
        raise X15ModelInputError(
            f"unsupported venues: {unknown_venues}"
        )
    if landmarks["game_id"].isna().any() or (
        landmarks["game_id"].astype(str).str.strip().eq("")
    ).any():
        raise X15ModelInputError("game_id must be nonempty")
    if landmarks["episode_id"].isna().any() or (
        landmarks["episode_id"].astype(str).str.strip().eq("")
    ).any():
        raise X15ModelInputError("episode_id must be nonempty")
    target = pd.to_numeric(landmarks["target_delta"], errors="coerce")
    reference_gap = pd.to_numeric(
        landmarks["reference_gap_at_landmark"],
        errors="coerce",
    )
    target_values = target.to_numpy(dtype=float)
    if not np.isfinite(target_values[target_eligible.to_numpy()]).all():
        raise X15ModelInputError(
            "target_delta must be finite for target-eligible rows"
        )
    reference_values = reference_gap.to_numpy(dtype=float)
    if not np.isfinite(
        reference_values[decision_eligible.to_numpy()]
    ).all():
        raise X15ModelInputError(
            "reference_gap_at_landmark must be finite for decision rows"
        )
    landmark = pd.to_numeric(
        landmarks["landmark_seconds"], errors="coerce"
    )
    endpoint = pd.to_numeric(
        landmarks["endpoint_seconds"], errors="coerce"
    )
    if (
        landmark.isna().any()
        or endpoint.isna().any()
        or (endpoint <= landmark).any()
    ):
        raise X15ModelInputError(
            "endpoint_seconds must be after landmark_seconds"
        )
    identity = [
        "game_id",
        "episode_id",
        "venue",
        "landmark_seconds",
        "endpoint_seconds",
    ]
    if landmarks.duplicated(identity).any():
        raise X15ModelInputError(
            "landmark identity rows must be unique"
        )
    build_x15_week_folds(landmarks)
    result = landmarks.copy(deep=True)
    result["_source_row_id"] = np.arange(len(result), dtype=int)
    result["_has_exclusion_reasons"] = reasons.to_numpy(dtype=bool)
    result["target_delta"] = target_values
    result["reference_gap_at_landmark"] = reference_values
    return result.loc[
        result["decision_eligible"]
    ].copy()


def _preprocessor(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> ColumnTransformer:
    numeric = tuple(
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(train[column].dtype)
        and not pd.api.types.is_bool_dtype(train[column].dtype)
    )
    categorical = tuple(
        column for column in feature_columns if column not in numeric
    )
    transformers: list[tuple[str, Pipeline, tuple[str, ...]]] = []
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
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _one_hot_probabilities(labels: pd.Series) -> np.ndarray:
    values = labels.astype(str).to_numpy()
    return np.column_stack(
        [values == class_name for class_name in DIRECTION_CLASSES]
    ).astype(float)


def _probability_frame(probabilities: np.ndarray) -> pd.DataFrame:
    if probabilities.shape[1] != len(DIRECTION_CLASSES):
        raise X15ModelInputError(
            "model probability output has the wrong class count"
        )
    if np.isnan(probabilities).all():
        return pd.DataFrame(
            probabilities,
            columns=("prob_down", "prob_no_move", "prob_up"),
        )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(
            probabilities.sum(axis=1),
            np.ones(len(probabilities)),
            rtol=0,
            atol=1e-12,
        )
    ):
        raise X15ModelInputError("model emitted invalid probabilities")
    return pd.DataFrame(
        probabilities,
        columns=("prob_down", "prob_no_move", "prob_up"),
    )


def _classification_metrics(
    truth: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(DIRECTION_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _regression_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
    }


def _fit_predict_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    noise_threshold: float,
    random_state: int,
) -> dict[str, tuple[np.ndarray, pd.Series, np.ndarray]]:
    train_labels = direction_labels(
        train["target_delta"],
        noise_threshold=noise_threshold,
    )
    if set(train_labels) != set(DIRECTION_CLASSES):
        raise X15ModelInputError(
            "each venue/fold training set requires DOWN, NO_MOVE, and UP"
        )
    output: dict[str, tuple[np.ndarray, pd.Series, np.ndarray]] = {}

    baseline_delta = -validation[
        "reference_gap_at_landmark"
    ].to_numpy(dtype=float)
    baseline_label = direction_labels(
        pd.Series(baseline_delta),
        noise_threshold=noise_threshold,
    )
    not_probabilistic = np.full(
        (len(validation), len(DIRECTION_CLASSES)),
        np.nan,
        dtype=float,
    )
    output[MODEL_IDS[0]] = (
        baseline_delta,
        baseline_label,
        not_probabilistic.copy(),
    )

    logistic = Pipeline(
        [
            (
                "features",
                _preprocessor(
                    train,
                    feature_columns=feature_columns,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    ).fit(train.loc[:, feature_columns], train_labels)
    raw_probability = np.asarray(
        logistic.predict_proba(validation.loc[:, feature_columns]),
        dtype=float,
    )
    observed_classes = tuple(
        str(value) for value in logistic.named_steps["model"].classes_
    )
    logistic_probability = np.column_stack(
        [
            raw_probability[:, observed_classes.index(class_name)]
            for class_name in DIRECTION_CLASSES
        ]
    )
    logistic_label = pd.Series(
        np.asarray(DIRECTION_CLASSES, dtype=object)[
            np.argmax(logistic_probability, axis=1)
        ],
        dtype="string",
    )
    centers = (
        pd.DataFrame(
            {
                "label": train_labels.to_numpy(dtype=object),
                "target": train["target_delta"].to_numpy(dtype=float),
            }
        )
        .groupby("label")["target"]
        .mean()
    )
    logistic_delta = logistic_probability @ np.asarray(
        [float(centers[class_name]) for class_name in DIRECTION_CLASSES]
    )
    output[MODEL_IDS[1]] = (
        logistic_delta,
        logistic_label,
        logistic_probability,
    )

    for model_id, estimator in (
        (
            MODEL_IDS[2],
            HuberRegressor(
                epsilon=1.35,
                alpha=0.0001,
                max_iter=2_000,
                tol=1e-7,
            ),
        ),
        (
            MODEL_IDS[3],
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=64,
                max_depth=3,
                learning_rate=0.05,
                min_child_weight=2,
                reg_lambda=1.0,
                subsample=1.0,
                colsample_bytree=1.0,
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            ),
        ),
    ):
        model = Pipeline(
            [
                (
                    "features",
                    _preprocessor(
                        train,
                        feature_columns=feature_columns,
                    ),
                ),
                ("model", estimator),
            ]
        ).fit(
            train.loc[:, feature_columns],
            train["target_delta"].to_numpy(dtype=float),
        )
        predicted_delta = np.asarray(
            model.predict(validation.loc[:, feature_columns]),
            dtype=float,
        )
        predicted_label = direction_labels(
            pd.Series(predicted_delta),
            noise_threshold=noise_threshold,
        )
        output[model_id] = (
            predicted_delta,
            predicted_label,
            not_probabilistic.copy(),
        )
    return output


def run_x15_walk_forward(
    landmarks: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    random_state: int = 20260728,
) -> X15ModelRun:
    """Run the five development folds independently for each venue."""

    if type(random_state) is not int:
        raise X15ModelInputError("random_state must be an integer")
    features = tuple(str(column) for column in feature_columns)
    decision_rows = _validate_landmarks(
        landmarks,
        feature_columns=features,
    )
    folds = build_x15_week_folds(decision_rows)
    oof_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for venue in sorted(decision_rows["venue"].astype(str).unique()):
        venue_rows = decision_rows.loc[
            decision_rows["venue"].eq(venue)
        ].copy()
        for fold in folds:
            train = venue_rows.loc[
                venue_rows["game_id"].isin(fold.train_game_ids)
                & venue_rows["target_eligible"]
            ].copy()
            validation_candidates = venue_rows.loc[
                venue_rows["game_id"].isin(fold.validation_game_ids)
            ].copy()
            validation_evaluation_mask = validation_candidates[
                "target_eligible"
            ].to_numpy(dtype=bool)
            validation_evaluation = validation_candidates.loc[
                validation_candidates["target_eligible"]
            ].copy()
            if (
                train.empty
                or validation_candidates.empty
                or validation_evaluation.empty
            ):
                raise X15ModelInputError(
                    f"{venue} fold {fold.fold_id} has no train or validation rows"
                )
            noise_threshold = training_noise_threshold(train)
            evaluation_labels = direction_labels(
                validation_evaluation["target_delta"],
                noise_threshold=noise_threshold,
            )
            predictions = _fit_predict_models(
                train,
                validation_candidates,
                feature_columns=features,
                noise_threshold=noise_threshold,
                random_state=random_state,
            )
            for model_id in MODEL_IDS:
                predicted_delta, predicted_label, probability = predictions[
                    model_id
                ]
                probability_frame = _probability_frame(probability)
                identity_columns = [
                    "game_id",
                    "episode_id",
                    "venue",
                    "nfl_week",
                    "landmark_seconds",
                    "endpoint_seconds",
                    "target_delta",
                    "decision_eligible",
                    "target_eligible",
                    "eligible",
                    "exclusion_reasons",
                    "reference_gap_at_landmark",
                    "factor_id",
                    "_source_row_id",
                ]
                model_oof = validation_candidates.loc[
                    :, identity_columns
                ].reset_index(drop=True)
                model_oof = model_oof.rename(
                    columns={"_source_row_id": "source_row_id"}
                )
                model_oof["fold_id"] = fold.fold_id
                model_oof["model_id"] = model_id
                model_oof["noise_threshold"] = noise_threshold
                model_oof["direction_label"] = pd.Series(
                    pd.NA,
                    index=model_oof.index,
                    dtype="string",
                )
                model_oof.loc[
                    validation_evaluation_mask, "direction_label"
                ] = evaluation_labels.to_numpy(dtype=object)
                model_oof["predicted_delta"] = predicted_delta
                model_oof["predicted_label"] = predicted_label.to_numpy(
                    dtype=object
                )
                model_oof = pd.concat(
                    [model_oof, probability_frame],
                    axis=1,
                )
                is_probability_model = (
                    model_id == "multinomial_logistic_v1"
                )
                model_oof["probability_status"] = (
                    "CALIBRATED_CLASS_PROBABILITY"
                    if is_probability_model
                    else "NOT_PROBABILISTIC"
                )
                oof_rows.append(model_oof)
                evaluation_prediction = predicted_label.loc[
                    validation_evaluation_mask
                ].reset_index(drop=True)
                evaluation_probability = probability[
                    validation_evaluation_mask
                ]
                classification = _classification_metrics(
                    evaluation_labels.reset_index(drop=True),
                    evaluation_prediction,
                )
                if is_probability_model:
                    classification["multiclass_log_loss"] = float(
                        log_loss(
                            evaluation_labels,
                            np.clip(
                                evaluation_probability,
                                1e-12,
                                1 - 1e-12,
                            ),
                            labels=list(DIRECTION_CLASSES),
                        )
                    )
                    probability_status = (
                        "CALIBRATED_CLASS_PROBABILITY"
                    )
                else:
                    classification["multiclass_log_loss"] = math.nan
                    probability_status = "NOT_PROBABILISTIC"
                metric_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "venue": venue,
                        "training_venue": venue,
                        "model_id": model_id,
                        "train_weeks": fold.train_weeks,
                        "validation_weeks": fold.validation_weeks,
                        "train_game_ids": tuple(
                            sorted(train["game_id"].astype(str).unique())
                        ),
                        "validation_game_ids": tuple(
                            sorted(
                                validation_candidates["game_id"]
                                .astype(str)
                                .unique()
                            )
                        ),
                        "training_row_count": len(train),
                        "validation_decision_row_count": len(
                            validation_candidates
                        ),
                        "validation_row_count": len(
                            validation_evaluation
                        ),
                        "training_game_count": train["game_id"].nunique(),
                        "validation_game_count": validation_evaluation[
                            "game_id"
                        ].nunique(),
                        "noise_threshold": noise_threshold,
                        "probability_status": probability_status,
                        **classification,
                        **_regression_metrics(
                            validation_evaluation[
                                "target_delta"
                            ].to_numpy(dtype=float),
                            predicted_delta[
                                validation_evaluation_mask
                            ],
                        ),
                    }
                )
    if not oof_rows:
        raise X15ModelInputError("no out-of-fold predictions were produced")
    oof = pd.concat(oof_rows, ignore_index=True).sort_values(
        [
            "venue",
            "fold_id",
            "model_id",
            "game_id",
            "episode_id",
            "landmark_seconds",
            "endpoint_seconds",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["venue", "fold_id", "model_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return X15ModelRun(oof_predictions=oof, fold_metrics=metrics)


__all__ = [
    "X15ModelInputError",
    "X15ModelRun",
    "X15WeekFold",
    "DIRECTION_CLASSES",
    "MODEL_IDS",
    "build_x15_week_folds",
    "direction_labels",
    "run_x15_walk_forward",
    "training_noise_threshold",
]
