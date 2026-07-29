"""Direction-conditional magnitude distributions for NFL X15 Stage B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


SUPPORTED: Final[str] = "SUPPORTED"
INSUFFICIENT_SUPPORT: Final[str] = "INSUFFICIENT_SUPPORT"
NOT_APPLICABLE: Final[str] = "NOT_APPLICABLE"
DIRECTION_CONDITIONS: Final[tuple[str, ...]] = ("DOWN", "UP")
PRIMARY_QUANTILES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50, 0.75, 0.90)
EXTREME_QUANTILES: Final[tuple[float, ...]] = (0.05, 0.95)

# Frozen before the first OOF run; the stricter q05/q95 gate is intentionally
# conservative and is always recorded in support output.
PRIMARY_MIN_ROWS: Final[int] = 24
PRIMARY_MIN_GAMES: Final[int] = 8
EXTREME_MIN_ROWS: Final[int] = 80
EXTREME_MIN_GAMES: Final[int] = 30

QUANTILE_XGB_PARAMS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "n_estimators": 24,
    "max_depth": 2,
    "learning_rate": 0.06,
    "min_child_weight": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": 1,
}


class X15DistributionError(ValueError):
    """Magnitude rows cannot support a governed conditional distribution."""


@dataclass(frozen=True, slots=True)
class QuantileSupportContract:
    """Immutable row/game thresholds fixed before fitting."""

    primary_min_rows: int = PRIMARY_MIN_ROWS
    primary_min_games: int = PRIMARY_MIN_GAMES
    extreme_min_rows: int = EXTREME_MIN_ROWS
    extreme_min_games: int = EXTREME_MIN_GAMES

    def __post_init__(self) -> None:
        values = (
            self.primary_min_rows,
            self.primary_min_games,
            self.extreme_min_rows,
            self.extreme_min_games,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise X15DistributionError(
                "quantile support thresholds must be positive integers"
            )
        if self.extreme_min_rows < self.primary_min_rows:
            raise X15DistributionError(
                "extreme row support cannot be weaker than primary support"
            )
        if self.extreme_min_games < self.primary_min_games:
            raise X15DistributionError(
                "extreme game support cannot be weaker than primary support"
            )


@dataclass(frozen=True, slots=True)
class DirectionQuantileSupport:
    direction: str
    row_count: int
    game_count: int
    primary_status: str
    extreme_status: str
    primary_min_rows: int
    primary_min_games: int
    extreme_min_rows: int
    extreme_min_games: int
    training_sha256: str


@dataclass(slots=True)
class DirectionalQuantileModels:
    """Fitted UP/DOWN quantile models plus immutable support evidence."""

    models: dict[str, XGBRegressor | None]
    support: dict[str, DirectionQuantileSupport]
    quantiles: tuple[float, ...]
    feature_columns: tuple[str, ...]

    def predict(self, features: pd.DataFrame, *, direction: str) -> pd.DataFrame:
        columns = [_quantile_column(value) for value in self.quantiles]
        if direction not in DIRECTION_CONDITIONS:
            return pd.DataFrame(np.nan, index=features.index, columns=columns)
        model = self.models.get(direction)
        if model is None:
            return pd.DataFrame(np.nan, index=features.index, columns=columns)
        matrix = _numeric_features(
            features, expected_columns=self.feature_columns
        )
        predicted = np.asarray(model.predict(matrix), dtype=float)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        projected = enforce_non_crossing(predicted)
        return pd.DataFrame(projected, index=features.index, columns=columns)


def _quantile_column(value: float) -> str:
    return f"q{int(round(value * 100)):02d}"


def _numeric_features(
    features: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    if not isinstance(features, pd.DataFrame) or (
        features.empty and not allow_empty
    ):
        raise X15DistributionError("features must be a nonempty DataFrame")
    columns = tuple(str(column) for column in features.columns)
    if expected_columns is not None and columns != expected_columns:
        raise X15DistributionError(
            "prediction feature columns differ from fitted columns"
        )
    matrix = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise X15DistributionError(
            "quantile features must be finite numeric values"
        )
    return matrix


def _direction_hash(
    matrix: np.ndarray,
    magnitude: np.ndarray,
    games: np.ndarray,
    weights: np.ndarray,
    row_ids: np.ndarray,
    *,
    direction: str,
    contract: QuantileSupportContract,
) -> str:
    rows = sorted(
        [
            {
                "game_id": str(game),
                "row_id": str(row_id),
                "features": [float(value) for value in feature_row],
                "magnitude": float(target),
                "weight": float(weight),
            }
            for feature_row, target, game, weight, row_id in zip(
                matrix, magnitude, games, weights, row_ids, strict=True
            )
        ],
        key=lambda row: (
            row["game_id"],
            row["row_id"],
            row["magnitude"],
            json.dumps(row["features"], separators=(",", ":")),
            row["weight"],
        ),
    )
    payload = json.dumps(
        {
            "direction": direction,
            "primary_quantiles": PRIMARY_QUANTILES,
            "extreme_quantiles": EXTREME_QUANTILES,
            "support_contract": {
                "primary_min_rows": contract.primary_min_rows,
                "primary_min_games": contract.primary_min_games,
                "extreme_min_rows": contract.extreme_min_rows,
                "extreme_min_games": contract.extreme_min_games,
            },
            "xgb_params": QUANTILE_XGB_PARAMS,
            "rows": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enforce_non_crossing(predictions: object) -> np.ndarray:
    """Project quantiles to nonnegative, monotonically increasing rows."""

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise X15DistributionError(
            "quantile predictions must be a two-dimensional matrix"
        )
    if not np.isfinite(values).all():
        raise X15DistributionError("quantile predictions must be finite")
    return np.maximum.accumulate(np.maximum(values, 0.0), axis=1)


def fit_directional_quantiles(
    features: pd.DataFrame,
    magnitude: object,
    directions: object,
    game_ids: object,
    *,
    sample_weight: object | None = None,
    row_ids: object | None = None,
    support_contract: QuantileSupportContract = QuantileSupportContract(),
    random_state: int = 20260728,
) -> DirectionalQuantileModels:
    """Fit direction-specific XGBoost quantiles on S&O, UP/DOWN rows only."""

    matrix = _numeric_features(features, allow_empty=True)
    target = np.asarray(magnitude, dtype=float)
    direction_values = np.asarray(directions, dtype=object)
    games = np.asarray(game_ids, dtype=object)
    if (
        target.ndim != 1
        or direction_values.ndim != 1
        or games.ndim != 1
        or not (len(matrix) == len(target) == len(direction_values) == len(games))
    ):
        raise X15DistributionError("quantile training inputs must be aligned")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise X15DistributionError(
            "conditional magnitude must be finite and nonnegative"
        )
    if not set(direction_values).issubset(
        {*DIRECTION_CONDITIONS, "NO_MOVE"}
    ):
        raise X15DistributionError("magnitude direction has an unknown class")
    if any(not str(game).strip() for game in games):
        raise X15DistributionError("game_ids must be nonempty")
    games = games.astype(str)
    if sample_weight is None:
        weights = np.ones(len(matrix), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) != len(matrix)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise X15DistributionError(
            "sample_weight must be aligned, finite, and positive"
        )
    if row_ids is None:
        identities = np.asarray(
            [
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        {
                            "game_id": str(game),
                            "features": [float(value) for value in feature_row],
                            "magnitude": float(value),
                            "direction": str(direction),
                            "weight": float(weight),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for feature_row, value, direction, game, weight in zip(
                    matrix,
                    target,
                    direction_values,
                    games,
                    weights,
                    strict=True,
                )
            ],
            dtype=str,
        )
    else:
        identities = np.asarray(row_ids, dtype=object)
        if identities.ndim != 1 or len(identities) != len(matrix):
            raise X15DistributionError(
                "row_ids must be one-dimensional and aligned"
            )
        if any(not str(value).strip() for value in identities):
            raise X15DistributionError("row_ids must be nonempty")
        identities = identities.astype(str)

    models: dict[str, XGBRegressor | None] = {}
    support: dict[str, DirectionQuantileSupport] = {}
    for offset, direction in enumerate(DIRECTION_CONDITIONS):
        mask = direction_values == direction
        direction_matrix = matrix[mask]
        direction_target = target[mask]
        direction_games = games[mask]
        direction_weights = weights[mask]
        direction_row_ids = identities[mask]
        row_count = int(mask.sum())
        game_count = len(set(direction_games))
        primary_supported = (
            row_count >= support_contract.primary_min_rows
            and game_count >= support_contract.primary_min_games
        )
        extreme_supported = (
            primary_supported
            and row_count >= support_contract.extreme_min_rows
            and game_count >= support_contract.extreme_min_games
        )
        quantiles = (
            tuple(sorted((*PRIMARY_QUANTILES, *EXTREME_QUANTILES)))
            if extreme_supported
            else PRIMARY_QUANTILES
        )
        digest = _direction_hash(
            direction_matrix,
            direction_target,
            direction_games,
            direction_weights,
            direction_row_ids,
            direction=direction,
            contract=support_contract,
        )
        support[direction] = DirectionQuantileSupport(
            direction=direction,
            row_count=row_count,
            game_count=game_count,
            primary_status=SUPPORTED if primary_supported else INSUFFICIENT_SUPPORT,
            extreme_status=SUPPORTED if extreme_supported else INSUFFICIENT_SUPPORT,
            primary_min_rows=support_contract.primary_min_rows,
            primary_min_games=support_contract.primary_min_games,
            extreme_min_rows=support_contract.extreme_min_rows,
            extreme_min_games=support_contract.extreme_min_games,
            training_sha256=digest,
        )
        if not primary_supported:
            models[direction] = None
            continue
        models[direction] = XGBRegressor(
            **QUANTILE_XGB_PARAMS,
            quantile_alpha=np.asarray(quantiles, dtype=float),
            random_state=int(random_state) + offset,
        )
        models[direction].fit(
            direction_matrix,
            direction_target,
            sample_weight=direction_weights,
            verbose=False,
        )

    # A single fit object must have one output schema.  Extreme quantiles are
    # published only when both conditional directions pass the stricter gate.
    publish_extremes = all(
        item.extreme_status == SUPPORTED for item in support.values()
    )
    published_quantiles = (
        tuple(sorted((*PRIMARY_QUANTILES, *EXTREME_QUANTILES)))
        if publish_extremes
        else PRIMARY_QUANTILES
    )
    if not publish_extremes:
        # Refit any direction that passed the extreme gate with the common
        # primary-only output schema.  This is still governed by fixed params.
        for offset, direction in enumerate(DIRECTION_CONDITIONS):
            if models[direction] is None:
                continue
            mask = direction_values == direction
            models[direction] = XGBRegressor(
                **QUANTILE_XGB_PARAMS,
                quantile_alpha=np.asarray(PRIMARY_QUANTILES, dtype=float),
                random_state=int(random_state) + offset,
            )
            models[direction].fit(
                matrix[mask],
                target[mask],
                sample_weight=weights[mask],
                verbose=False,
            )
    return DirectionalQuantileModels(
        models=models,
        support=support,
        quantiles=published_quantiles,
        feature_columns=tuple(str(column) for column in features.columns),
    )


def magnitude_distribution_metrics(
    truth: object,
    predictions: pd.DataFrame,
    *,
    sample_weight: object | None = None,
) -> dict[str, float]:
    """Return fixed pinball, approximate CRPS, and interval metrics."""

    target = np.asarray(truth, dtype=float)
    if target.ndim != 1 or len(target) != len(predictions):
        raise X15DistributionError("magnitude metric inputs must be aligned")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise X15DistributionError("magnitude truth must be nonnegative")
    required = [_quantile_column(value) for value in PRIMARY_QUANTILES]
    if not set(required).issubset(predictions.columns):
        raise X15DistributionError("primary magnitude quantiles are missing")
    predicted = predictions.loc[:, required].to_numpy(dtype=float)
    if not np.isfinite(predicted).all():
        raise X15DistributionError("magnitude predictions must be finite")
    weights = (
        np.ones(len(target), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        weights.ndim != 1
        or len(weights) != len(target)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise X15DistributionError(
            "magnitude metric weights must be finite and positive"
        )
    pinballs: dict[str, float] = {}
    losses: list[np.ndarray] = []
    for index, quantile in enumerate(PRIMARY_QUANTILES):
        residual = target - predicted[:, index]
        loss = np.maximum(quantile * residual, (quantile - 1) * residual)
        pinballs[f"pinball_{_quantile_column(quantile)}"] = float(
            np.average(loss, weights=weights)
        )
        losses.append(loss)
    interval_coverage = (
        (target >= predicted[:, 0]) & (target <= predicted[:, -1])
    ).astype(float)
    interval_width = predicted[:, -1] - predicted[:, 0]
    return {
        **pinballs,
        "approximate_crps": float(
            2 * np.mean([np.average(loss, weights=weights) for loss in losses])
        ),
        "interval_80_coverage": float(
            np.average(interval_coverage, weights=weights)
        ),
        "interval_80_width": float(np.average(interval_width, weights=weights)),
    }


__all__ = [
    "DIRECTION_CONDITIONS",
    "EXTREME_MIN_GAMES",
    "EXTREME_MIN_ROWS",
    "EXTREME_QUANTILES",
    "INSUFFICIENT_SUPPORT",
    "NOT_APPLICABLE",
    "PRIMARY_MIN_GAMES",
    "PRIMARY_MIN_ROWS",
    "PRIMARY_QUANTILES",
    "QUANTILE_XGB_PARAMS",
    "SUPPORTED",
    "DirectionQuantileSupport",
    "DirectionalQuantileModels",
    "QuantileSupportContract",
    "X15DistributionError",
    "enforce_non_crossing",
    "fit_directional_quantiles",
    "magnitude_distribution_metrics",
]
