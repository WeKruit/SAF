"""NBA market-prior, logistic, and histogram-GBDT baseline POC."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaselineInputError(ValueError):
    """The baseline frame violates its point-in-time feature contract."""


def _validate_frame(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    prior: str,
) -> None:
    required = {
        "game_id",
        "prediction_at",
        "game_start_at",
        target,
        prior,
        *(features),
        *(f"{column}__available_at" for column in (*features, prior)),
    }
    missing = required - set(frame.columns)
    if missing:
        raise BaselineInputError(f"missing baseline columns: {sorted(missing)}")
    if frame.empty:
        raise BaselineInputError("baseline frame must not be empty")
    if frame["game_id"].isna().any() or (frame["game_id"].astype(str) == "").any():
        raise BaselineInputError("game IDs must be nonempty")
    prediction = frame["prediction_at"]
    if not pd.api.types.is_datetime64_any_dtype(prediction.dtype) or any(
        value.tzinfo is None or value.utcoffset().total_seconds() != 0
        for value in prediction
    ):
        raise BaselineInputError("prediction_at must be timezone-aware UTC")
    game_start = frame["game_start_at"]
    if not pd.api.types.is_datetime64_any_dtype(game_start.dtype) or any(
        value.tzinfo is None or value.utcoffset().total_seconds() != 0
        for value in game_start
    ):
        raise BaselineInputError("game_start_at must be timezone-aware UTC")
    for column in features:
        availability = frame[f"{column}__available_at"]
        if not pd.api.types.is_datetime64_any_dtype(availability.dtype) or any(
            value.tzinfo is None or value.utcoffset().total_seconds() != 0
            for value in availability
        ):
            raise BaselineInputError("feature availability must be timezone-aware UTC")
        if (availability > prediction).any():
            raise BaselineInputError(f"point-in-time violation for feature {column}")
    prior_availability = frame[f"{prior}__available_at"]
    if not pd.api.types.is_datetime64_any_dtype(prior_availability.dtype) or any(
        value.tzinfo is None or value.utcoffset().total_seconds() != 0
        for value in prior_availability
    ):
        raise BaselineInputError(
            "pregame prior availability must be timezone-aware UTC"
        )
    if (prior_availability > prediction).any():
        raise BaselineInputError("pregame prior must be available by prediction_at")
    if (prior_availability > game_start).any():
        raise BaselineInputError("pregame prior must be frozen by game_start_at")
    numeric = frame.loc[:, [*features, prior]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise BaselineInputError("baseline features must be finite numeric values")
    prior_values = frame[prior].to_numpy(dtype=float)
    if np.any((prior_values < 0) | (prior_values > 1)):
        raise BaselineInputError("market prior must be in [0, 1]")
    targets = frame[target].to_numpy()
    if not np.all(np.isin(targets, [0, 1])):
        raise BaselineInputError("target must be binary")


def fit_predict_nba_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    market_prior_column: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the registered three-model comparison on one pre-split PIT fold."""

    features = tuple(feature_columns)
    if not features or len(set(features)) != len(features):
        raise BaselineInputError("feature_columns must be nonempty and unique")
    if target_column == market_prior_column:
        raise BaselineInputError(
            "target and market prior columns must be distinct roles"
        )
    structural = {"game_id", "prediction_at", "game_start_at"}
    if target_column in structural or target_column.endswith("__available_at"):
        raise BaselineInputError(
            "target column cannot use a structural or availability role"
        )
    if market_prior_column in structural or market_prior_column.endswith(
        "__available_at"
    ):
        raise BaselineInputError(
            "market prior column cannot use a structural or availability role"
        )
    reserved = structural | {target_column, market_prior_column}
    leaking = tuple(
        feature
        for feature in features
        if feature in reserved or feature.endswith("__available_at")
    )
    if leaking:
        raise BaselineInputError(
            "reserved structural columns cannot be model features: "
            + ", ".join(leaking)
        )
    if type(seed) is not int:
        raise BaselineInputError("seed must be an integer")
    _validate_frame(
        train, features=features, target=target_column, prior=market_prior_column
    )
    _validate_frame(
        test, features=features, target=target_column, prior=market_prior_column
    )
    train_games = set(train["game_id"])
    test_games = set(test["game_id"])
    if train_games & test_games:
        raise BaselineInputError("train/test game overlap is forbidden")
    if train["prediction_at"].max() >= test["prediction_at"].min():
        raise BaselineInputError(
            "all training predictions must be strictly earlier than test predictions"
        )
    y_train = train[target_column].to_numpy(dtype=int)
    if len(np.unique(y_train)) != 2:
        raise BaselineInputError("training fold requires both target classes")
    x_train = train.loc[:, features].to_numpy(dtype=float)
    x_test = test.loc[:, features].to_numpy(dtype=float)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed),
    )
    gbdt = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        random_state=seed,
    )
    logistic.fit(x_train, y_train)
    gbdt.fit(x_train, y_train)
    return {
        "market_prior": test[market_prior_column].to_numpy(dtype=float).copy(),
        "logistic": logistic.predict_proba(x_test)[:, 1].copy(),
        "gbdt": gbdt.predict_proba(x_test)[:, 1].copy(),
    }


__all__ = ["BaselineInputError", "fit_predict_nba_baselines"]
