"""Training-only probability calibration for the NFL X15 Stage B heads.

The fit functions consume prequential predictions produced exclusively from an
outer fold's training games.  Unsupported fits are represented explicitly and
apply the raw probabilities unchanged; they never synthesize a constant model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.linear_model import LogisticRegression


CALIBRATED: Final[str] = "CALIBRATED"
RAW_UNCALIBRATED: Final[str] = "RAW_UNCALIBRATED"
INSUFFICIENT_SUPPORT: Final[str] = "INSUFFICIENT_SUPPORT"
SUPPORTED: Final[str] = "SUPPORTED"

# Frozen before the first Stage B OOF run.  These are support requirements, not
# tunable hyperparameters.
CALIBRATION_MIN_ROWS: Final[int] = 6
CALIBRATION_MIN_GAMES: Final[int] = 4
PROBABILITY_EPSILON: Final[float] = 1e-6
TEMPERATURE_GRID: Final[tuple[float, ...]] = tuple(
    float(value) for value in np.geomspace(0.25, 4.0, 257)
)
DIRECTION_CLASSES: Final[tuple[str, ...]] = ("DOWN", "NO_MOVE", "UP")


class X15CalibrationError(ValueError):
    """Prequential inputs cannot support deterministic calibration."""


def _training_sha256(
    raw: np.ndarray,
    truth: np.ndarray,
    games: np.ndarray,
    weights: np.ndarray,
    row_ids: np.ndarray,
) -> str:
    rows = sorted(
        [
            {
                "game_id": str(game),
                "row_id": str(row_id),
                "raw": (
                    [float(value) for value in probability]
                    if np.ndim(probability)
                    else float(probability)
                ),
                "truth": str(label),
                "weight": float(weight),
            }
            for probability, label, game, weight, row_id in zip(
                raw, truth, games, weights, row_ids, strict=True
            )
        ],
        key=lambda row: (
            row["game_id"],
            row["row_id"],
            row["truth"],
            json.dumps(row["raw"], separators=(",", ":")),
            row["weight"],
        ),
    )
    payload = json.dumps(
        {
            "calibration_min_rows": CALIBRATION_MIN_ROWS,
            "calibration_min_games": CALIBRATION_MIN_GAMES,
            "rows": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_ids(
    values: object | None,
    *,
    raw: np.ndarray,
    truth: np.ndarray,
    games: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if values is not None:
        result = np.asarray(values, dtype=object)
        if result.ndim != 1 or len(result) != len(games):
            raise X15CalibrationError(
                "row_ids must be one-dimensional and aligned"
            )
        if any(not str(value).strip() for value in result):
            raise X15CalibrationError("row_ids must be nonempty")
        return result.astype(str)
    identities: list[str] = []
    for probability, label, game, weight in zip(
        raw, truth, games, weights, strict=True
    ):
        payload = json.dumps(
            {
                "game_id": str(game),
                "raw": _canonical_probability(probability),
                "truth": str(label),
                "weight": float(weight),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        identities.append(
            "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        )
    return np.asarray(identities, dtype=str)


def _canonical_probability(value: object) -> float | list[float]:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return [float(child) for child in array]


def _games(values: object, *, expected_rows: int) -> np.ndarray:
    games = np.asarray(values, dtype=object)
    if games.ndim != 1 or len(games) != expected_rows:
        raise X15CalibrationError("game_ids must be one-dimensional and aligned")
    if any(not str(value).strip() for value in games):
        raise X15CalibrationError("game_ids must be nonempty")
    return games.astype(str)


def _weights(values: object | None, *, expected_rows: int) -> np.ndarray:
    if values is None:
        result = np.ones(expected_rows, dtype=float)
    else:
        result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) != expected_rows:
        raise X15CalibrationError(
            "sample_weight must be one-dimensional and aligned"
        )
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise X15CalibrationError("sample_weight must be finite and positive")
    return result


def _unique_games_in_order(games: np.ndarray) -> tuple[str, ...]:
    return tuple(sorted(set(str(value) for value in games)))


@dataclass(frozen=True, slots=True)
class BinaryPlattCalibrator:
    """Frozen Platt parameters or an explicit raw-probability fallback."""

    status: str
    support_reason: str
    slope: float | None
    intercept: float | None
    training_sha256: str
    fit_game_ids: tuple[str, ...]
    support_min_rows: int = CALIBRATION_MIN_ROWS
    support_min_games: int = CALIBRATION_MIN_GAMES

    def transform(self, raw_probability: object) -> np.ndarray:
        raw = np.asarray(raw_probability, dtype=float)
        if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise X15CalibrationError("raw probabilities must be in [0, 1]")
        if self.status != CALIBRATED:
            return raw.copy()
        clipped = np.clip(raw, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        logits = np.log(clipped / (1 - clipped))
        calibrated_logits = float(self.intercept) + float(self.slope) * logits
        return 1 / (1 + np.exp(-calibrated_logits))


@dataclass(frozen=True, slots=True)
class DirectionTemperatureCalibrator:
    """Frozen multinomial temperature or explicit raw-probability fallback."""

    status: str
    support_reason: str
    temperature: float | None
    training_sha256: str
    fit_game_ids: tuple[str, ...]
    support_min_rows: int = CALIBRATION_MIN_ROWS
    support_min_games: int = CALIBRATION_MIN_GAMES

    def transform(self, raw_probabilities: object) -> np.ndarray:
        raw = np.asarray(raw_probabilities, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(DIRECTION_CLASSES):
            raise X15CalibrationError(
                "direction probabilities must have three columns"
            )
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise X15CalibrationError(
                "direction probabilities must be finite and nonnegative"
            )
        totals = raw.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise X15CalibrationError(
                "direction probability rows must have positive mass"
            )
        normalized = raw / totals
        if self.status != CALIBRATED:
            return normalized
        clipped = np.clip(
            normalized, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON
        )
        logits = np.log(clipped) / float(self.temperature)
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_binary_platt(
    raw_probability: object,
    truth: object,
    game_ids: object,
    *,
    sample_weight: object | None = None,
    row_ids: object | None = None,
) -> BinaryPlattCalibrator:
    """Fit Platt scaling to binary prequential training predictions."""

    raw = np.asarray(raw_probability, dtype=float)
    labels = np.asarray(truth)
    if raw.ndim != 1 or labels.ndim != 1 or len(raw) != len(labels):
        raise X15CalibrationError("binary calibration inputs must be aligned")
    if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise X15CalibrationError("raw probabilities must be in [0, 1]")
    if not np.isin(labels, [0, 1, False, True]).all():
        raise X15CalibrationError("binary truth must contain only zero or one")
    labels = labels.astype(int)
    games = _games(game_ids, expected_rows=len(raw))
    weights = _weights(sample_weight, expected_rows=len(raw))
    identities = _row_ids(
        row_ids, raw=raw, truth=labels, games=games, weights=weights
    )
    digest = _training_sha256(raw, labels, games, weights, identities)
    fit_games = _unique_games_in_order(games)
    supported = (
        len(raw) >= CALIBRATION_MIN_ROWS
        and len(set(games)) >= CALIBRATION_MIN_GAMES
        and len(set(labels)) == 2
    )
    if not supported:
        return BinaryPlattCalibrator(
            status=RAW_UNCALIBRATED,
            support_reason=INSUFFICIENT_SUPPORT,
            slope=None,
            intercept=None,
            training_sha256=digest,
            fit_game_ids=fit_games,
        )

    clipped = np.clip(raw, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=20260728,
    )
    model.fit(logits, labels, sample_weight=weights)
    return BinaryPlattCalibrator(
        status=CALIBRATED,
        support_reason=SUPPORTED,
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        training_sha256=digest,
        fit_game_ids=fit_games,
    )


def fit_direction_temperature(
    raw_probabilities: object,
    truth: object,
    game_ids: object,
    *,
    sample_weight: object | None = None,
    row_ids: object | None = None,
) -> DirectionTemperatureCalibrator:
    """Fit one fixed-grid multinomial temperature to prequential predictions."""

    raw = np.asarray(raw_probabilities, dtype=float)
    labels = np.asarray(truth, dtype=object)
    if (
        raw.ndim != 2
        or raw.shape[1] != len(DIRECTION_CLASSES)
        or labels.ndim != 1
        or len(raw) != len(labels)
    ):
        raise X15CalibrationError("direction calibration inputs must be aligned")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise X15CalibrationError(
            "direction probabilities must be finite and nonnegative"
        )
    totals = raw.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise X15CalibrationError(
            "direction probability rows must have positive mass"
        )
    if not set(labels).issubset(DIRECTION_CLASSES):
        raise X15CalibrationError("direction truth has an unknown class")
    raw = raw / totals
    games = _games(game_ids, expected_rows=len(raw))
    weights = _weights(sample_weight, expected_rows=len(raw))
    identities = _row_ids(
        row_ids, raw=raw, truth=labels, games=games, weights=weights
    )
    digest = _training_sha256(raw, labels, games, weights, identities)
    fit_games = _unique_games_in_order(games)
    supported = (
        len(raw) >= CALIBRATION_MIN_ROWS
        and len(set(games)) >= CALIBRATION_MIN_GAMES
        and set(labels) == set(DIRECTION_CLASSES)
    )
    if not supported:
        return DirectionTemperatureCalibrator(
            status=RAW_UNCALIBRATED,
            support_reason=INSUFFICIENT_SUPPORT,
            temperature=None,
            training_sha256=digest,
            fit_game_ids=fit_games,
        )

    class_indexes = np.array(
        [DIRECTION_CLASSES.index(str(label)) for label in labels],
        dtype=int,
    )
    clipped = np.clip(raw, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    log_probabilities = np.log(clipped)
    losses: list[float] = []
    for temperature in TEMPERATURE_GRID:
        logits = log_probabilities / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        row_loss = -np.log(
            np.clip(
                probabilities[np.arange(len(labels)), class_indexes],
                PROBABILITY_EPSILON,
                1,
            )
        )
        losses.append(float(np.average(row_loss, weights=weights)))
    temperature = TEMPERATURE_GRID[int(np.argmin(losses))]
    return DirectionTemperatureCalibrator(
        status=CALIBRATED,
        support_reason=SUPPORTED,
        temperature=temperature,
        training_sha256=digest,
        fit_game_ids=fit_games,
    )


__all__ = [
    "CALIBRATED",
    "CALIBRATION_MIN_GAMES",
    "CALIBRATION_MIN_ROWS",
    "DIRECTION_CLASSES",
    "INSUFFICIENT_SUPPORT",
    "RAW_UNCALIBRATED",
    "SUPPORTED",
    "BinaryPlattCalibrator",
    "DirectionTemperatureCalibrator",
    "X15CalibrationError",
    "fit_binary_platt",
    "fit_direction_temperature",
]
