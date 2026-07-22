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
        target,
        prior,
        *(features),
        *(f"{column}__available_at" for column in (*features, prior)),
    }
    missing = required - set(frame.columns)
    if missing:
        raise BaselineInputError(f"missing baseline columns: {sorted(missing)}")
    prediction = frame["prediction_at"]
    if not pd.api.types.is_datetime64_any_dtype(prediction.dtype) or any(
        value.tzinfo is None or value.utcoffset().total_seconds() != 0
        for value in prediction
    ):
        raise BaselineInputError("prediction_at must be timezone-aware UTC")
    for column in (*features, prior):
        availability = frame[f"{column}__available_at"]
        if not pd.api.types.is_datetime64_any_dtype(availability.dtype) or any(
            value.tzinfo is None or value.utcoffset().total_seconds() != 0
            for value in availability
        ):
            raise BaselineInputError("feature availability must be timezone-aware UTC")
        if (availability > prediction).any():
            raise BaselineInputError(f"point-in-time violation for feature {column}")
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
    if type(seed) is not int:
        raise BaselineInputError("seed must be an integer")
    _validate_frame(
        train, features=features, target=target_column, prior=market_prior_column
    )
    _validate_frame(
        test, features=features, target=target_column, prior=market_prior_column
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
