"""Game-grouped chronological validation and calibration metrics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ValidationInputError(ValueError):
    """Inputs cannot support the registered grouped chronological evaluation."""


def _validate_utc_series(series: pd.Series, field: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(series.dtype):
        raise ValidationInputError(f"{field} must contain datetimes")
    for value in series:
        if pd.isna(value) or value.tzinfo is None or value.utcoffset().total_seconds() != 0:
            raise ValidationInputError(f"{field} must be timezone-aware UTC")


def game_grouped_walk_forward(
    frame: pd.DataFrame,
    *,
    min_train_games: int,
    game_column: str = "game_id",
    time_column: str = "played_at",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield one-game test folds with strictly earlier complete training games."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValidationInputError("frame must be a nonempty DataFrame")
    if type(min_train_games) is not int or min_train_games < 1:
        raise ValidationInputError("min_train_games must be a positive integer")
    missing = {game_column, time_column} - set(frame.columns)
    if missing:
        raise ValidationInputError(f"missing split columns: {sorted(missing)}")
    if frame[game_column].isna().any() or (frame[game_column].astype(str) == "").any():
        raise ValidationInputError("game IDs must be nonempty")
    _validate_utc_series(frame[time_column], time_column)

    intervals = (
        frame.groupby(game_column, sort=False)[time_column]
        .agg(start="min", end="max")
        .reset_index()
    )
    intervals["_game_sort"] = intervals[game_column].astype(str)
    intervals = intervals.sort_values(["start", "_game_sort"], kind="mergesort")
    for test_row in intervals.itertuples(index=False):
        test_start = test_row.start
        test_game = getattr(test_row, game_column)
        train_games = intervals.loc[intervals["end"] < test_start, game_column].tolist()
        if len(train_games) < min_train_games:
            continue
        train = frame.loc[frame[game_column].isin(train_games)].copy()
        test = frame.loc[frame[game_column] == test_game].copy()
        train = train.sort_values([time_column, game_column], kind="mergesort")
        test = test.sort_values([time_column, game_column], kind="mergesort")
        yield train, test


def _arrays(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).copy()
    p = np.asarray(probabilities, dtype=float).copy()
    cluster = np.asarray(groups, dtype=object).copy()
    if y.ndim != 1 or p.ndim != 1 or cluster.ndim != 1:
        raise ValidationInputError("metric inputs must be one-dimensional")
    if not (len(y) == len(p) == len(cluster)) or len(y) == 0:
        raise ValidationInputError("metric input length mismatch")
    if not np.all(np.isin(y, [0, 1])):
        raise ValidationInputError("targets must be binary 0/1")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValidationInputError("probabilities must be finite in [0, 1]")
    for value in cluster:
        if isinstance(value, str) and not value.strip():
            raise ValidationInputError(
                "groups must be present and finite for every row"
            )
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            raise ValidationInputError(
                "groups must be present and finite for every row"
            )
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValidationInputError(
                "groups must be present and finite for every row"
            )
    return y.astype(int), p, cluster


def _calibration_parameters(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise ValidationInputError("calibration requires both target classes")
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    ).fit(logit, y)
    return {
        "slope": float(calibration.coef_[0, 0]),
        "intercept": float(calibration.intercept_[0]),
    }


def _point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    calibration = _calibration_parameters(y, p)
    return {
        "brier": brier,
        "log_loss": log_loss,
        "calibration_slope": calibration["slope"],
        "calibration_intercept": calibration["intercept"],
    }


def _bootstrap_parameters(
    *,
    bootstrap_samples: int,
    confidence_level: float,
    minimum_valid_samples: int,
    seed: int,
) -> float:
    if type(bootstrap_samples) is not int or bootstrap_samples < 20:
        raise ValidationInputError("bootstrap_samples must be an integer >= 20")
    if type(confidence_level) is not float or not 0 < confidence_level < 1:
        raise ValidationInputError("confidence_level must be a float in (0, 1)")
    if (
        type(minimum_valid_samples) is not int
        or minimum_valid_samples < 20
        or minimum_valid_samples > bootstrap_samples
    ):
        raise ValidationInputError(
            "minimum_valid_samples must be an integer between 20 and bootstrap_samples"
        )
    if type(seed) is not int:
        raise ValidationInputError("seed must be an integer")
    return (1 - confidence_level) / 2


def evaluate_probabilities(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    groups: Sequence[object] | np.ndarray,
    bootstrap_samples: int,
    confidence_level: float,
    minimum_valid_samples: int,
    seed: int,
) -> dict[str, object]:
    """Report Brier/log loss/calibration and game-cluster bootstrap CIs."""

    alpha = _bootstrap_parameters(
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        minimum_valid_samples=minimum_valid_samples,
        seed=seed,
    )
    y, p, cluster = _arrays(y_true, probabilities, groups)
    point = _point_metrics(y, p)
    try:
        unique_groups = tuple(dict.fromkeys(cluster.tolist()))
    except TypeError as error:
        raise ValidationInputError("groups must be hashable") from error
    if len(unique_groups) < 2:
        raise ValidationInputError("cluster bootstrap requires at least two groups")
    index_by_group = {
        group: np.flatnonzero(cluster == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(bootstrap_samples):
        selected = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        indices = np.concatenate(
            [index_by_group[unique_groups[int(index)]] for index in selected]
        )
        try:
            metrics = _point_metrics(y[indices], p[indices])
        except ValidationInputError:
            continue
        for name, value in metrics.items():
            samples[name].append(value)
    valid_samples = min(len(values) for values in samples.values())
    if valid_samples < minimum_valid_samples:
        raise ValidationInputError(
            "too few valid clustered bootstrap samples: "
            f"{valid_samples} < {minimum_valid_samples}"
        )
    confidence_intervals = {
        name: (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )
        for name, values in samples.items()
    }
    return {
        **point,
        "bootstrap_ci": confidence_intervals,
        "confidence_level": confidence_level,
        "minimum_valid_samples": minimum_valid_samples,
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_valid": valid_samples,
        "clusters": len(unique_groups),
        "observations": len(y),
    }


def evaluate_model_vs_prior(
    y_true: Sequence[int] | np.ndarray,
    model_probabilities: Sequence[float] | np.ndarray,
    prior_probabilities: Sequence[float] | np.ndarray,
    *,
    groups: Sequence[object] | np.ndarray,
    bootstrap_samples: int,
    confidence_level: float,
    minimum_valid_samples: int,
    seed: int,
) -> dict[str, object]:
    """Report paired game-cluster CIs for model-minus-prior score deltas."""

    alpha = _bootstrap_parameters(
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        minimum_valid_samples=minimum_valid_samples,
        seed=seed,
    )
    y, model, cluster = _arrays(y_true, model_probabilities, groups)
    _, prior, _ = _arrays(y_true, prior_probabilities, groups)
    model_metrics = _point_metrics(y, model)
    prior_metrics = _point_metrics(y, prior)
    score_names = ("brier", "log_loss")
    delta = {
        name: model_metrics[name] - prior_metrics[name] for name in score_names
    }

    try:
        unique_groups = tuple(dict.fromkeys(cluster.tolist()))
    except TypeError as error:
        raise ValidationInputError("groups must be hashable") from error
    if len(unique_groups) < 2:
        raise ValidationInputError("cluster bootstrap requires at least two groups")
    index_by_group = {
        group: np.flatnonzero(cluster == group) for group in unique_groups
    }

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in score_names}
    for _ in range(bootstrap_samples):
        selected = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        indices = np.concatenate(
            [index_by_group[unique_groups[int(index)]] for index in selected]
        )
        try:
            sampled_model = _point_metrics(y[indices], model[indices])
            sampled_prior = _point_metrics(y[indices], prior[indices])
        except ValidationInputError:
            continue
        for name in score_names:
            samples[name].append(sampled_model[name] - sampled_prior[name])

    valid_samples = min(len(values) for values in samples.values())
    if valid_samples < minimum_valid_samples:
        raise ValidationInputError(
            "too few valid clustered bootstrap samples: "
            f"{valid_samples} < {minimum_valid_samples}"
        )
    confidence_intervals = {
        name: (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )
        for name, values in samples.items()
    }
    return {
        "model_metrics": model_metrics,
        "prior_metrics": prior_metrics,
        "delta": delta,
        "delta_definition": "model_minus_prior",
        "delta_bootstrap_ci": confidence_intervals,
        "confidence_level": confidence_level,
        "minimum_valid_samples": minimum_valid_samples,
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_valid": valid_samples,
        "clusters": len(unique_groups),
        "observations": len(y),
    }


def _multiclass_arrays(
    y_true: Sequence[object] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[str],
    groups: Sequence[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    class_order = tuple(classes)
    if (
        len(class_order) < 2
        or any(type(value) is not str or not value.strip() for value in class_order)
        or len(class_order) != len(set(class_order))
    ):
        raise ValidationInputError("classes must be a fixed unique nonempty order")
    y_values = np.asarray(y_true, dtype=object).copy()
    matrix = np.asarray(probabilities, dtype=float).copy()
    cluster = np.asarray(groups, dtype=object).copy()
    if y_values.ndim != 1 or cluster.ndim != 1:
        raise ValidationInputError("targets and groups must be one-dimensional")
    if matrix.ndim != 2 or matrix.shape[1] != len(class_order):
        raise ValidationInputError(
            "probability columns must exactly match the declared class order"
        )
    if len(y_values) == 0 or not (
        len(y_values) == matrix.shape[0] == len(cluster)
    ):
        raise ValidationInputError("multiclass metric input length mismatch")
    class_index = {value: index for index, value in enumerate(class_order)}
    if any(value not in class_index for value in y_values):
        raise ValidationInputError("every target must belong to the declared classes")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValidationInputError("probabilities must be finite in [0, 1]")
    if not np.allclose(
        matrix.sum(axis=1),
        np.ones(matrix.shape[0]),
        rtol=0,
        atol=1e-12,
    ):
        raise ValidationInputError("each probability row must sum to one")
    for value in cluster:
        if isinstance(value, str) and not value.strip():
            raise ValidationInputError(
                "groups must be present and finite for every row"
            )
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and missing:
            raise ValidationInputError(
                "groups must be present and finite for every row"
            )
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValidationInputError(
                "groups must be present and finite for every row"
            )
    encoded = np.asarray([class_index[value] for value in y_values], dtype=int)
    return encoded, matrix, cluster, class_order


def _multiclass_point_metrics(
    encoded_targets: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
) -> dict[str, object]:
    if set(np.unique(encoded_targets)) != set(range(len(classes))):
        raise ValidationInputError(
            "multiclass calibration requires every declared target class"
        )
    indicator = np.eye(len(classes), dtype=float)[encoded_targets]
    brier = float(np.mean(np.sum((probabilities - indicator) ** 2, axis=1)))
    selected = np.clip(
        probabilities[np.arange(len(encoded_targets)), encoded_targets],
        1e-15,
        1,
    )
    log_loss = float(-np.mean(np.log(selected)))
    calibration = {
        class_name: _calibration_parameters(
            (encoded_targets == class_index).astype(int),
            probabilities[:, class_index],
        )
        for class_index, class_name in enumerate(classes)
    }
    return {
        "brier": brier,
        "log_loss": log_loss,
        "ovr_calibration": calibration,
    }


def _validate_feature_availability(
    *,
    observations: int,
    prediction_at: Sequence[object] | pd.Series | None,
    feature_available_at: Sequence[object] | pd.Series | None,
) -> pd.Series:
    if prediction_at is None or feature_available_at is None:
        raise ValidationInputError(
            "prediction_at and feature_available_at must be provided together"
        )
    predictions = pd.Series(prediction_at).reset_index(drop=True)
    availability = pd.Series(feature_available_at).reset_index(drop=True)
    if len(predictions) != observations or len(availability) != observations:
        raise ValidationInputError("point-in-time availability length mismatch")
    _validate_utc_series(predictions, "prediction_at")
    _validate_utc_series(availability, "feature_available_at")
    if (availability > predictions).any():
        raise ValidationInputError(
            "point-in-time feature availability cannot follow prediction_at"
        )
    return predictions


def _validate_prior_availability(
    *,
    observations: int,
    prediction_at: pd.Series,
    prior_available_at: Sequence[object] | pd.Series | None,
) -> None:
    if prior_available_at is None:
        raise ValidationInputError(
            "prior_available_at is required when prior_probabilities are provided"
        )
    availability = pd.Series(prior_available_at).reset_index(drop=True)
    if len(availability) != observations:
        raise ValidationInputError("prior point-in-time availability length mismatch")
    _validate_utc_series(availability, "prior_available_at")
    if (availability > prediction_at).any():
        raise ValidationInputError(
            "prior point-in-time availability cannot follow prediction_at"
        )


def evaluate_multiclass_probabilities(
    y_true: Sequence[object] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[str],
    groups: Sequence[object] | np.ndarray,
    bootstrap_samples: int,
    confidence_level: float,
    minimum_valid_samples: int,
    seed: int,
    prediction_at: Sequence[object] | pd.Series,
    feature_available_at: Sequence[object] | pd.Series,
    prior_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    prior_available_at: Sequence[object] | pd.Series | None = None,
) -> dict[str, object]:
    """Evaluate one joint multiclass distribution with game-cluster inference.

    The class order is explicit and immutable.  Bootstrap resampling draws whole
    games and uses the same draw for model/prior deltas, so the reported paired
    intervals cannot silently degrade into row-level or unpaired inference.
    """

    alpha = _bootstrap_parameters(
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        minimum_valid_samples=minimum_valid_samples,
        seed=seed,
    )
    y, matrix, cluster, class_order = _multiclass_arrays(
        y_true,
        probabilities,
        classes=classes,
        groups=groups,
    )
    predictions = _validate_feature_availability(
        observations=len(y),
        prediction_at=prediction_at,
        feature_available_at=feature_available_at,
    )
    point = _multiclass_point_metrics(y, matrix, class_order)
    prior_matrix: np.ndarray | None = None
    prior_point: dict[str, object] | None = None
    if prior_probabilities is not None:
        prior_y, prior_matrix, prior_groups, prior_classes = _multiclass_arrays(
            y_true,
            prior_probabilities,
            classes=class_order,
            groups=groups,
        )
        if (
            not np.array_equal(prior_y, y)
            or not np.array_equal(prior_groups, cluster)
            or prior_classes != class_order
        ):
            raise ValidationInputError("prior inputs must align exactly with model inputs")
        _validate_prior_availability(
            observations=len(y),
            prediction_at=predictions,
            prior_available_at=prior_available_at,
        )
        prior_point = _multiclass_point_metrics(y, prior_matrix, class_order)

    try:
        unique_groups = tuple(dict.fromkeys(cluster.tolist()))
    except TypeError as error:
        raise ValidationInputError("groups must be hashable") from error
    if len(unique_groups) < 2:
        raise ValidationInputError("cluster bootstrap requires at least two groups")
    index_by_group = {
        group: np.flatnonzero(cluster == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    score_samples: dict[str, list[float]] = {"brier": [], "log_loss": []}
    calibration_samples: dict[str, dict[str, list[float]]] = {
        class_name: {"slope": [], "intercept": []}
        for class_name in class_order
    }
    delta_samples: dict[str, list[float]] = {"brier": [], "log_loss": []}
    valid_samples = 0
    for _ in range(bootstrap_samples):
        selected_groups = rng.choice(
            len(unique_groups), size=len(unique_groups), replace=True
        )
        indices = np.concatenate(
            [index_by_group[unique_groups[int(index)]] for index in selected_groups]
        )
        try:
            sampled_model = _multiclass_point_metrics(
                y[indices], matrix[indices], class_order
            )
            sampled_prior = (
                _multiclass_point_metrics(
                    y[indices], prior_matrix[indices], class_order
                )
                if prior_matrix is not None
                else None
            )
        except ValidationInputError:
            continue
        valid_samples += 1
        for metric in score_samples:
            score_samples[metric].append(float(sampled_model[metric]))
        sampled_calibration = sampled_model["ovr_calibration"]
        for class_name in class_order:
            for parameter in ("slope", "intercept"):
                calibration_samples[class_name][parameter].append(
                    float(sampled_calibration[class_name][parameter])
                )
        if sampled_prior is not None:
            for metric in delta_samples:
                delta_samples[metric].append(
                    float(sampled_model[metric]) - float(sampled_prior[metric])
                )
    if valid_samples < minimum_valid_samples:
        raise ValidationInputError(
            "too few valid clustered bootstrap samples: "
            f"{valid_samples} < {minimum_valid_samples}"
        )

    def interval(values: Sequence[float]) -> tuple[float, float]:
        return (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )

    bootstrap_ci = {
        metric: interval(values) for metric, values in score_samples.items()
    }
    calibration_ci = {
        class_name: {
            parameter: interval(values)
            for parameter, values in parameters.items()
        }
        for class_name, parameters in calibration_samples.items()
    }
    if prior_point is None:
        prior_comparison: dict[str, object] = {
            "available": False,
            "reason": "pit_prior_not_supplied",
        }
    else:
        delta = {
            metric: float(point[metric]) - float(prior_point[metric])
            for metric in delta_samples
        }
        prior_comparison = {
            "available": True,
            "prior_metrics": {
                "brier": prior_point["brier"],
                "log_loss": prior_point["log_loss"],
                "ovr_calibration": prior_point["ovr_calibration"],
            },
            "delta": delta,
            "delta_definition": "model_minus_prior",
            "delta_bootstrap_ci": {
                metric: interval(values)
                for metric, values in delta_samples.items()
            },
            "bootstrap_samples_valid": valid_samples,
        }
    return {
        "classes": class_order,
        "brier": point["brier"],
        "brier_definition": "mean_sum_squared_class_error",
        "log_loss": point["log_loss"],
        "ovr_calibration": point["ovr_calibration"],
        "bootstrap_ci": bootstrap_ci,
        "ovr_calibration_bootstrap_ci": calibration_ci,
        "prior_comparison": prior_comparison,
        "confidence_level": confidence_level,
        "minimum_valid_samples": minimum_valid_samples,
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_valid": valid_samples,
        "clusters": len(unique_groups),
        "observations": len(y),
    }


__all__ = [
    "ValidationInputError",
    "evaluate_multiclass_probabilities",
    "evaluate_model_vs_prior",
    "evaluate_probabilities",
    "game_grouped_walk_forward",
]
