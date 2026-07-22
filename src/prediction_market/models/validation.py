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
    if any(value is None for value in cluster):
        raise ValidationInputError("groups must be present for every row")
    return y.astype(int), p, cluster


def _point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise ValidationInputError("calibration requires both target classes")
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    ).fit(logit, y)
    return {
        "brier": brier,
        "log_loss": log_loss,
        "calibration_slope": float(calibration.coef_[0, 0]),
        "calibration_intercept": float(calibration.intercept_[0]),
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


__all__ = [
    "ValidationInputError",
    "evaluate_model_vs_prior",
    "evaluate_probabilities",
    "game_grouped_walk_forward",
]
