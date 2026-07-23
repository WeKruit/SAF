"""State-conditioned five-minute soccer goal-transition intensities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from prediction_market.sports.soccer_game_state import SoccerGameState


TRANSITION_HORIZON_SECONDS = 300
TRANSITION_CLASSES = ("home_goal", "away_goal", "no_goal")
COEFFICIENT_NAMES = (
    "second_half",
    "score_difference",
    "dismissal_difference",
)

_TRAINING_COLUMNS = frozenset(
    {
        "match_id",
        "side",
        "base_goals_per_90",
        "exposure_seconds",
        "goal_count",
        "second_half",
        "score_difference",
        "dismissal_difference",
        "feature_available_at",
        "label_available_at",
    }
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class SoccerTransitionModelError(ValueError):
    """Input cannot support the frozen dynamic-intensity contract."""


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SoccerTransitionModelError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SoccerTransitionModelError(f"{field_name} must be finite")
    return number


def _integer(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SoccerTransitionModelError(f"{field_name} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise SoccerTransitionModelError(
            f"{field_name} must be >= {minimum}"
        )
    return number


def _utc_timestamp(value: object, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SoccerTransitionModelError(
            f"{field_name} must be a valid timestamp"
        ) from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise SoccerTransitionModelError(
            f"{field_name} must be timezone-aware"
        )
    return timestamp.tz_convert("UTC")


def _parameter_sha256(coefficients: tuple[float, ...]) -> str:
    material = {
        "coefficient_names": list(COEFFICIENT_NAMES),
        "coefficients": list(coefficients),
        "model": "symmetric_soccer_dynamic_intensity_v1",
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _temperature_parameter_sha256(temperature: float) -> str:
    material = {
        "classes": list(TRANSITION_CLASSES),
        "method": "multiclass_temperature_scaling_v1",
        "temperature": temperature,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _feature_sha256(features: SoccerTransitionFeatures) -> str:
    material = {
        "away_dismissals": features.away_dismissals,
        "away_score": features.away_score,
        "away_team_id": features.away_team_id,
        "elapsed_seconds": features.elapsed_seconds,
        "game_id": features.game_id,
        "home_dismissal_difference": features.home_dismissal_difference,
        "home_dismissals": features.home_dismissals,
        "home_score": features.home_score,
        "home_score_difference": features.home_score_difference,
        "home_team_id": features.home_team_id,
        "second_half": features.second_half,
        "source_state_sha256": features.source_state_sha256,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SoccerTransitionFeatures:
    """The model projection of one immutable reducer state."""

    game_id: str
    home_team_id: int
    away_team_id: int
    elapsed_seconds: float
    second_half: int
    home_score: int
    away_score: int
    home_dismissals: int
    away_dismissals: int
    home_score_difference: int
    home_dismissal_difference: int
    source_state_sha256: str
    feature_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.game_id) is not str or not self.game_id:
            raise SoccerTransitionModelError("game_id must be nonempty")
        home_team_id = _integer(
            self.home_team_id,
            "home_team_id",
            minimum=1,
        )
        away_team_id = _integer(
            self.away_team_id,
            "away_team_id",
            minimum=1,
        )
        if home_team_id == away_team_id:
            raise SoccerTransitionModelError("game teams must differ")
        if _finite_float(self.elapsed_seconds, "elapsed_seconds") < 0.0:
            raise SoccerTransitionModelError(
                "elapsed_seconds must be nonnegative"
            )
        if _integer(self.second_half, "second_half") not in (0, 1):
            raise SoccerTransitionModelError("second_half must be 0 or 1")
        home_score = _integer(self.home_score, "home_score", minimum=0)
        away_score = _integer(self.away_score, "away_score", minimum=0)
        home_dismissals = _integer(
            self.home_dismissals,
            "home_dismissals",
            minimum=0,
        )
        away_dismissals = _integer(
            self.away_dismissals,
            "away_dismissals",
            minimum=0,
        )
        if _integer(
            self.home_score_difference,
            "home_score_difference",
        ) != home_score - away_score:
            raise SoccerTransitionModelError(
                "home_score_difference does not match the scores"
            )
        if _integer(
            self.home_dismissal_difference,
            "home_dismissal_difference",
        ) != home_dismissals - away_dismissals:
            raise SoccerTransitionModelError(
                "home_dismissal_difference does not match dismissals"
            )
        if (
            type(self.source_state_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_state_sha256) is None
        ):
            raise SoccerTransitionModelError(
                "source_state_sha256 must be a SHA-256 digest"
            )
        object.__setattr__(self, "feature_sha256", _feature_sha256(self))


@dataclass(frozen=True, slots=True)
class DynamicIntensityModel:
    """A deterministic symmetric Poisson-offset intensity fit."""

    coefficients: tuple[float, ...]
    l2_penalty: float
    objective: float
    iterations: int
    optimizer_status: str
    initial_objective: float | None = None
    projected_gradient_inf_norm: float = 0.0
    coefficient_names: tuple[str, ...] = field(
        default=COEFFICIENT_NAMES,
        init=False,
    )
    parameter_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.coefficients) is not tuple or len(self.coefficients) != 3:
            raise SoccerTransitionModelError(
                "coefficients must contain the three frozen features"
            )
        coefficients = tuple(
            _finite_float(value, f"coefficients[{index}]")
            for index, value in enumerate(self.coefficients)
        )
        object.__setattr__(self, "coefficients", coefficients)
        l2_penalty = _finite_float(self.l2_penalty, "l2_penalty")
        if l2_penalty < 0.0:
            raise SoccerTransitionModelError(
                "l2_penalty must be nonnegative"
            )
        _finite_float(self.objective, "objective")
        initial_objective = self.initial_objective
        if initial_objective is None:
            initial_objective = float(self.objective)
            object.__setattr__(self, "initial_objective", initial_objective)
        _finite_float(initial_objective, "initial_objective")
        gradient_norm = _finite_float(
            self.projected_gradient_inf_norm,
            "projected_gradient_inf_norm",
        )
        if gradient_norm < 0.0:
            raise SoccerTransitionModelError(
                "projected_gradient_inf_norm must be nonnegative"
            )
        _integer(self.iterations, "iterations", minimum=0)
        if (
            type(self.optimizer_status) is not str
            or not self.optimizer_status
            or self.optimizer_status.strip() != self.optimizer_status
        ):
            raise SoccerTransitionModelError(
                "optimizer_status must be a nonempty string"
            )
        object.__setattr__(
            self,
            "parameter_sha256",
            _parameter_sha256(coefficients),
        )


@dataclass(frozen=True, slots=True)
class TemperatureCalibration:
    """One immutable multiclass temperature fitted on disjoint matches."""

    temperature: float
    initial_objective: float
    objective: float
    iterations: int
    optimizer_status: str
    calibration_match_count: int
    calibration_observation_count: int
    parameter_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        temperature = _finite_float(self.temperature, "temperature")
        if temperature <= 0.0:
            raise SoccerTransitionModelError(
                "temperature must be strictly positive"
            )
        object.__setattr__(self, "temperature", temperature)
        initial_objective = _finite_float(
            self.initial_objective,
            "initial_objective",
        )
        objective = _finite_float(self.objective, "objective")
        if objective > initial_objective + 1e-12:
            raise SoccerTransitionModelError(
                "temperature calibration must not worsen its objective"
            )
        _integer(self.iterations, "iterations", minimum=0)
        if (
            type(self.optimizer_status) is not str
            or not self.optimizer_status
            or self.optimizer_status.strip() != self.optimizer_status
        ):
            raise SoccerTransitionModelError(
                "optimizer_status must be a nonempty string"
            )
        match_count = _integer(
            self.calibration_match_count,
            "calibration_match_count",
            minimum=1,
        )
        observation_count = _integer(
            self.calibration_observation_count,
            "calibration_observation_count",
            minimum=match_count,
        )
        object.__setattr__(
            self,
            "parameter_sha256",
            _temperature_parameter_sha256(temperature),
        )


@dataclass(frozen=True, slots=True)
class SoccerTransitionDistribution:
    """The next-goal competing-risk distribution for one five-minute horizon."""

    probabilities: tuple[float, float, float]
    source_state_sha256: str
    source_feature_sha256: str
    model_parameter_sha256: str
    horizon_seconds: int = TRANSITION_HORIZON_SECONDS
    classes: tuple[str, str, str] = field(
        default=TRANSITION_CLASSES,
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self.probabilities) is not tuple or len(self.probabilities) != 3:
            raise SoccerTransitionModelError(
                "probabilities must contain all transition classes"
            )
        probabilities = tuple(
            _finite_float(value, f"probabilities[{index}]")
            for index, value in enumerate(self.probabilities)
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise SoccerTransitionModelError(
                "transition probabilities must be in [0, 1]"
            )
        if not math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise SoccerTransitionModelError(
                "transition probabilities must sum to one"
            )
        object.__setattr__(self, "probabilities", probabilities)
        if (
            type(self.source_state_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_state_sha256) is None
        ):
            raise SoccerTransitionModelError(
                "source_state_sha256 must be a SHA-256 digest"
            )
        if (
            type(self.source_feature_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_feature_sha256) is None
        ):
            raise SoccerTransitionModelError(
                "source_feature_sha256 must be a SHA-256 digest"
            )
        if (
            type(self.model_parameter_sha256) is not str
            or _SHA256_RE.fullmatch(self.model_parameter_sha256) is None
        ):
            raise SoccerTransitionModelError(
                "model_parameter_sha256 must be a SHA-256 digest"
            )
        if self.horizon_seconds != TRANSITION_HORIZON_SECONDS:
            raise SoccerTransitionModelError(
                "transition horizon must be exactly five minutes"
            )

    @property
    def probability_home_goal(self) -> float:
        return self.probabilities[0]

    @property
    def probability_away_goal(self) -> float:
        return self.probabilities[1]

    @property
    def probability_no_goal(self) -> float:
        return self.probabilities[2]


def extract_transition_features(
    state: SoccerGameState,
) -> SoccerTransitionFeatures:
    """Project only current, point-in-time reducer state into model features."""

    if not isinstance(state, SoccerGameState):
        raise SoccerTransitionModelError(
            "state must be an immutable SoccerGameState"
        )
    if state.terminal:
        raise SoccerTransitionModelError(
            "terminal state cannot produce a transition forecast"
        )
    if state.sequence == 0 or state.period == 0:
        raise SoccerTransitionModelError(
            "the match must have started before transition prediction"
        )
    if state.period not in {1, 2}:
        raise SoccerTransitionModelError(
            "transition prediction is limited to regulation periods"
        )
    if state.period_clock_ms > 40 * 60 * 1_000:
        raise SoccerTransitionModelError(
            "state does not have a full five-minute regulation horizon"
        )

    dismissal_keys: set[tuple[int, int]] = set()
    for card in state.cards:
        if card.sequence > state.sequence:
            raise SoccerTransitionModelError(
                "state contains a future card sequence"
            )
        if card.team_id not in state.team_ids:
            raise SoccerTransitionModelError(
                "card team is outside the current game"
            )
        if card.card in {"Second Yellow", "Red Card"}:
            dismissal_keys.add((card.team_id, card.player_id))
    home_dismissals = sum(
        team_id == state.home_team_id for team_id, _ in dismissal_keys
    )
    away_dismissals = sum(
        team_id == state.away_team_id for team_id, _ in dismissal_keys
    )
    return SoccerTransitionFeatures(
        game_id=state.game_id,
        home_team_id=state.home_team_id,
        away_team_id=state.away_team_id,
        elapsed_seconds=state.clock_ms / 1_000.0,
        second_half=int(state.period >= 2),
        home_score=state.home_score,
        away_score=state.away_score,
        home_dismissals=home_dismissals,
        away_dismissals=away_dismissals,
        home_score_difference=state.home_score - state.away_score,
        home_dismissal_difference=home_dismissals - away_dismissals,
        source_state_sha256=state.state_sha256,
    )


def predict_transition_distribution(
    model: DynamicIntensityModel,
    *,
    base_home_goals: float,
    base_away_goals: float,
    features: SoccerTransitionFeatures,
) -> SoccerTransitionDistribution:
    """Return symmetric state-conditioned competing goal intensities."""

    if not isinstance(model, DynamicIntensityModel):
        raise SoccerTransitionModelError(
            "model must be a DynamicIntensityModel"
        )
    if not isinstance(features, SoccerTransitionFeatures):
        raise SoccerTransitionModelError(
            "features must be SoccerTransitionFeatures"
        )
    base_home = _finite_float(base_home_goals, "base_home_goals")
    base_away = _finite_float(base_away_goals, "base_away_goals")
    if base_home <= 0.0 or base_away <= 0.0:
        raise SoccerTransitionModelError(
            "base goal rates must be strictly positive"
        )

    coefficients = np.asarray(model.coefficients, dtype=float)
    home_covariates = np.asarray(
        (
            features.second_half,
            features.home_score_difference,
            features.home_dismissal_difference,
        ),
        dtype=float,
    )
    away_covariates = np.asarray(
        (
            features.second_half,
            -features.home_score_difference,
            -features.home_dismissal_difference,
        ),
        dtype=float,
    )
    home_log_multiplier = float(home_covariates @ coefficients)
    away_log_multiplier = float(away_covariates @ coefficients)
    try:
        expected_home = (
            base_home
            * TRANSITION_HORIZON_SECONDS
            / (90 * 60)
            * math.exp(home_log_multiplier)
        )
        expected_away = (
            base_away
            * TRANSITION_HORIZON_SECONDS
            / (90 * 60)
            * math.exp(away_log_multiplier)
        )
    except OverflowError as error:
        raise SoccerTransitionModelError(
            "state-conditioned goal intensity overflowed"
        ) from error
    total_expected = expected_home + expected_away
    if (
        not math.isfinite(total_expected)
        or total_expected <= 0.0
        or expected_home <= 0.0
        or expected_away <= 0.0
    ):
        raise SoccerTransitionModelError(
            "state-conditioned goal intensities must be finite and positive"
        )

    probability_no_goal = math.exp(-total_expected)
    probability_any_goal = -math.expm1(-total_expected)
    probability_home_goal = (
        probability_any_goal * expected_home / total_expected
    )
    probability_away_goal = (
        1.0 - probability_home_goal - probability_no_goal
    )
    return SoccerTransitionDistribution(
        probabilities=(
            probability_home_goal,
            probability_away_goal,
            probability_no_goal,
        ),
        source_state_sha256=features.source_state_sha256,
        source_feature_sha256=features.feature_sha256,
        model_parameter_sha256=model.parameter_sha256,
    )


def _validated_temperature_inputs(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    target_array = np.asarray(targets, dtype=object)
    probability_array = np.asarray(probabilities, dtype=float)
    if target_array.ndim != 1 or target_array.size == 0:
        raise SoccerTransitionModelError(
            "calibration targets must be a nonempty one-dimensional array"
        )
    if probability_array.shape != (
        target_array.size,
        len(TRANSITION_CLASSES),
    ):
        raise SoccerTransitionModelError(
            "calibration probabilities must have one column per transition class"
        )
    if (
        not np.all(np.isfinite(probability_array))
        or np.any(probability_array <= 0.0)
        or np.any(probability_array >= 1.0)
        or not np.allclose(
            probability_array.sum(axis=1),
            np.ones(target_array.size),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise SoccerTransitionModelError(
            "calibration probabilities must be finite, strictly inside "
            "(0, 1), and normalized"
        )
    class_index = {label: index for index, label in enumerate(TRANSITION_CLASSES)}
    if any(
        type(target) is not str or target not in class_index
        for target in target_array
    ):
        raise SoccerTransitionModelError(
            "calibration targets must use the frozen transition classes"
        )
    observed_classes = {str(target) for target in target_array}
    if observed_classes != set(TRANSITION_CLASSES):
        raise SoccerTransitionModelError(
            "calibration targets must contain every transition class"
        )
    encoded_targets = np.asarray(
        [class_index[str(target)] for target in target_array],
        dtype=int,
    )
    if groups is None:
        return encoded_targets, probability_array, None
    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1 or group_array.shape != target_array.shape:
        raise SoccerTransitionModelError(
            "calibration groups must align one-to-one with targets"
        )
    canonical_groups: list[str] = []
    for group in group_array:
        if isinstance(group, bool) or not isinstance(group, (Integral, str)):
            raise SoccerTransitionModelError(
                "calibration groups must be integer or string match ids"
            )
        if isinstance(group, Integral):
            canonical_groups.append(f"int:{int(group)}")
        elif group:
            canonical_groups.append(f"str:{group}")
        else:
            raise SoccerTransitionModelError(
                "calibration group strings must be nonempty"
            )
    return (
        encoded_targets,
        probability_array,
        np.asarray(canonical_groups, dtype=object),
    )


def apply_multiclass_temperature(
    probabilities: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Apply the frozen log-probability temperature transform."""

    probability_array = np.asarray(probabilities, dtype=float)
    if (
        probability_array.ndim != 2
        or probability_array.shape[1] != len(TRANSITION_CLASSES)
        or probability_array.shape[0] == 0
    ):
        raise SoccerTransitionModelError(
            "probabilities must be a nonempty matrix with three columns"
        )
    if (
        not np.all(np.isfinite(probability_array))
        or np.any(probability_array <= 0.0)
        or np.any(probability_array >= 1.0)
        or not np.allclose(
            probability_array.sum(axis=1),
            np.ones(probability_array.shape[0]),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise SoccerTransitionModelError(
            "probabilities must be finite, strictly inside (0, 1), and normalized"
        )
    fitted_temperature = _finite_float(temperature, "temperature")
    if fitted_temperature <= 0.0:
        raise SoccerTransitionModelError(
            "temperature must be strictly positive"
        )
    scaled_logits = np.log(probability_array) / fitted_temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    scaled = np.exp(scaled_logits)
    calibrated = scaled / scaled.sum(axis=1, keepdims=True)
    if (
        not np.all(np.isfinite(calibrated))
        or not np.allclose(
            calibrated.sum(axis=1),
            np.ones(calibrated.shape[0]),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise SoccerTransitionModelError(
            "temperature transform failed exact normalization"
        )
    return calibrated


def fit_multiclass_temperature(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    groups: np.ndarray,
    minimum_matches: int,
    optimizer_max_iterations: int,
) -> TemperatureCalibration:
    """Fit one temperature by match-equalized multiclass log loss."""

    minimum = _integer(
        minimum_matches,
        "minimum_matches",
        minimum=1,
    )
    maximum_iterations = _integer(
        optimizer_max_iterations,
        "optimizer_max_iterations",
        minimum=1,
    )
    encoded_targets, raw_probabilities, canonical_groups = (
        _validated_temperature_inputs(
            targets,
            probabilities,
            groups=groups,
        )
    )
    assert canonical_groups is not None
    unique_groups, group_inverse, group_counts = np.unique(
        canonical_groups,
        return_inverse=True,
        return_counts=True,
    )
    if unique_groups.size < minimum:
        raise SoccerTransitionModelError(
            f"temperature calibration requires at least {minimum} "
            "calibration matches"
        )
    order = np.lexsort(
        (
            raw_probabilities[:, 2],
            raw_probabilities[:, 1],
            raw_probabilities[:, 0],
            encoded_targets,
            canonical_groups,
        )
    )
    encoded_targets = encoded_targets[order]
    raw_probabilities = raw_probabilities[order]
    canonical_groups = canonical_groups[order]
    unique_groups, group_inverse, group_counts = np.unique(
        canonical_groups,
        return_inverse=True,
        return_counts=True,
    )
    observation_weights = (
        1.0
        / float(unique_groups.size)
        / group_counts[group_inverse].astype(float)
    )
    log_probabilities = np.log(raw_probabilities)
    row_indices = np.arange(encoded_targets.size)

    def objective_and_gradient(log_temperature: np.ndarray) -> tuple[float, np.ndarray]:
        if log_temperature.shape != (1,) or not np.isfinite(log_temperature[0]):
            return 1e100, np.asarray((1e50,), dtype=float)
        temperature = math.exp(float(log_temperature[0]))
        scaled_logits = log_probabilities / temperature
        scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(scaled_logits)
        calibrated = exponentiated / exponentiated.sum(
            axis=1,
            keepdims=True,
        )
        selected = calibrated[row_indices, encoded_targets]
        if np.any(selected <= 0.0) or not np.all(np.isfinite(selected)):
            return 1e100, np.asarray((1e50,), dtype=float)
        objective = float(
            np.sum(observation_weights * -np.log(selected))
        )
        expected_log_probability = np.sum(
            calibrated * log_probabilities,
            axis=1,
        )
        selected_log_probability = log_probabilities[
            row_indices,
            encoded_targets,
        ]
        derivative = (
            selected_log_probability - expected_log_probability
        ) / temperature
        gradient = float(np.sum(observation_weights * derivative))
        if not math.isfinite(objective) or not math.isfinite(gradient):
            return 1e100, np.asarray((1e50,), dtype=float)
        return objective, np.asarray((gradient,), dtype=float)

    initial = np.asarray((0.0,), dtype=float)
    initial_objective, _ = objective_and_gradient(initial)
    lower = math.log(0.05)
    upper = math.log(20.0)
    result = minimize(
        lambda value: objective_and_gradient(value)[0],
        initial,
        method="L-BFGS-B",
        jac=lambda value: objective_and_gradient(value)[1],
        bounds=((lower, upper),),
        options={
            "maxiter": maximum_iterations,
            "ftol": 1e-15,
            "gtol": 1e-10,
            "maxls": 100,
        },
    )
    if (
        not result.success
        or result.x.shape != (1,)
        or not np.all(np.isfinite(result.x))
    ):
        raise SoccerTransitionModelError(
            "temperature optimizer failed closed: "
            f"status={result.status} message={result.message}"
        )
    fitted_log_temperature = float(result.x[0])
    if (
        math.isclose(fitted_log_temperature, lower, abs_tol=1e-8)
        or math.isclose(fitted_log_temperature, upper, abs_tol=1e-8)
    ):
        raise SoccerTransitionModelError(
            "temperature optimizer stopped on its frozen search boundary"
        )
    objective, gradient = objective_and_gradient(
        np.asarray(result.x, dtype=float)
    )
    if objective > initial_objective + 1e-12:
        raise SoccerTransitionModelError(
            "temperature optimizer worsened the calibration objective"
        )
    if abs(float(gradient[0])) > 1e-7:
        raise SoccerTransitionModelError(
            "temperature optimizer failed gradient convergence"
        )
    return TemperatureCalibration(
        temperature=math.exp(fitted_log_temperature),
        initial_objective=initial_objective,
        objective=objective,
        iterations=int(result.nit),
        optimizer_status=str(result.message),
        calibration_match_count=int(unique_groups.size),
        calibration_observation_count=int(encoded_targets.size),
    )


def _validated_training_arrays(
    rows: pd.DataFrame,
    *,
    evaluation_cutoff: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(rows, pd.DataFrame):
        raise SoccerTransitionModelError("rows must be a pandas DataFrame")
    actual_columns = frozenset(str(column) for column in rows.columns)
    if actual_columns != _TRAINING_COLUMNS or len(rows.columns) != len(
        _TRAINING_COLUMNS
    ):
        missing = sorted(_TRAINING_COLUMNS - actual_columns)
        extra = sorted(actual_columns - _TRAINING_COLUMNS)
        raise SoccerTransitionModelError(
            f"training schema mismatch: missing={missing} extra={extra}"
        )
    if rows.empty:
        raise SoccerTransitionModelError("training rows must not be empty")

    records: list[
        tuple[
            int,
            str,
            int,
            int,
            float,
            int,
            int,
            int,
            int,
        ]
    ] = []
    identities: set[tuple[int, str, int, int]] = set()
    for row in rows.itertuples(index=False):
        match_id = _integer(row.match_id, "match_id", minimum=1)
        if row.side not in {"home", "away"}:
            raise SoccerTransitionModelError("side must be home or away")
        side = str(row.side)
        base_goals = _finite_float(
            row.base_goals_per_90,
            "base_goals_per_90",
        )
        if base_goals <= 0.0:
            raise SoccerTransitionModelError(
                "base_goals_per_90 must be strictly positive"
            )
        exposure_seconds = _integer(
            row.exposure_seconds,
            "exposure_seconds",
            minimum=1,
        )
        if exposure_seconds != TRANSITION_HORIZON_SECONDS:
            raise SoccerTransitionModelError(
                "exposure_seconds must be exactly five minutes"
            )
        goal_count = _integer(row.goal_count, "goal_count", minimum=0)
        second_half = _integer(row.second_half, "second_half")
        if second_half not in (0, 1):
            raise SoccerTransitionModelError("second_half must be 0 or 1")
        score_difference = _integer(
            row.score_difference,
            "score_difference",
        )
        dismissal_difference = _integer(
            row.dismissal_difference,
            "dismissal_difference",
        )
        feature_at = _utc_timestamp(
            row.feature_available_at,
            "feature_available_at",
        )
        label_at = _utc_timestamp(
            row.label_available_at,
            "label_available_at",
        )
        if feature_at >= evaluation_cutoff or label_at >= evaluation_cutoff:
            raise SoccerTransitionModelError(
                "all features and labels must be strictly before evaluation_cutoff"
            )
        if label_at <= feature_at:
            raise SoccerTransitionModelError(
                "label_available_at must follow feature_available_at"
            )
        if (
            label_at - feature_at
            != pd.Timedelta(seconds=TRANSITION_HORIZON_SECONDS)
        ):
            raise SoccerTransitionModelError(
                "feature-to-label interval must be exactly five minutes"
            )
        feature_ns = int(feature_at.value)
        label_ns = int(label_at.value)
        identity = (match_id, side, feature_ns, label_ns)
        if identity in identities:
            raise SoccerTransitionModelError(
                "training rows contain a duplicate side interval"
            )
        identities.add(identity)
        records.append(
            (
                match_id,
                side,
                feature_ns,
                label_ns,
                base_goals,
                goal_count,
                second_half,
                score_difference,
                dismissal_difference,
            )
        )

    records.sort(key=lambda value: value[:4])
    paired: dict[tuple[int, int, int], dict[str, tuple[object, ...]]] = {}
    for record in records:
        key = (record[0], record[2], record[3])
        paired.setdefault(key, {})[record[1]] = record
    for pair in paired.values():
        if set(pair) != {"home", "away"}:
            raise SoccerTransitionModelError(
                "each interval must contain one home and one away row"
            )
        home = pair["home"]
        away = pair["away"]
        if (
            home[6] != away[6]
            or home[7] != -away[7]
            or home[8] != -away[8]
        ):
            raise SoccerTransitionModelError(
                "home and away interval covariates must be symmetric"
            )

    feature_matrix = np.asarray(
        [
            (
                record[6],
                record[7],
                record[8],
            )
            for record in records
        ],
        dtype=float,
    )
    outcomes = np.asarray([record[5] for record in records], dtype=float)
    log_offsets = np.log(
        np.asarray([record[4] for record in records], dtype=float)
        * TRANSITION_HORIZON_SECONDS
        / (90 * 60)
    )
    if (
        not np.all(np.isfinite(feature_matrix))
        or not np.all(np.isfinite(outcomes))
        or not np.all(np.isfinite(log_offsets))
    ):
        raise SoccerTransitionModelError(
            "validated training arrays must be finite"
        )
    return feature_matrix, outcomes, log_offsets


def fit_dynamic_intensity(
    rows: pd.DataFrame,
    *,
    evaluation_cutoff: object,
    held_out_match_ids: frozenset[int],
    l2_penalty: float,
    optimizer_max_iterations: int,
) -> DynamicIntensityModel:
    """Fit the frozen symmetric Poisson-offset model with strict PIT checks."""

    cutoff = _utc_timestamp(evaluation_cutoff, "evaluation_cutoff")
    penalty = _finite_float(l2_penalty, "l2_penalty")
    if penalty < 0.0:
        raise SoccerTransitionModelError(
            "l2_penalty must be nonnegative"
        )
    maximum_iterations = _integer(
        optimizer_max_iterations,
        "optimizer_max_iterations",
        minimum=1,
    )
    if (
        type(held_out_match_ids) is not frozenset
        or not held_out_match_ids
        or any(
            isinstance(match_id, bool)
            or not isinstance(match_id, Integral)
            or int(match_id) <= 0
            for match_id in held_out_match_ids
        )
    ):
        raise SoccerTransitionModelError(
            "held_out_match_ids must be a nonempty frozenset of positive integers"
        )
    features, outcomes, log_offsets = _validated_training_arrays(
        rows,
        evaluation_cutoff=cutoff,
    )
    if np.linalg.matrix_rank(features) != len(COEFFICIENT_NAMES):
        raise SoccerTransitionModelError(
            "dynamic-intensity feature design must have full column rank"
        )
    training_match_ids = {
        int(match_id) for match_id in rows["match_id"].tolist()
    }
    if training_match_ids & {int(value) for value in held_out_match_ids}:
        raise SoccerTransitionModelError(
            "training rows contain a held-out match"
        )

    def objective_and_gradient(
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        linear_predictor = log_offsets + features @ parameters
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.exp(linear_predictor)
        if not np.all(np.isfinite(expected)):
            return 1e100, np.full(3, 1e50, dtype=float)
        objective = float(
            np.sum(expected - outcomes * linear_predictor + gammaln(outcomes + 1.0))
            + 0.5 * penalty * float(parameters @ parameters)
        )
        gradient = (
            features.T @ (expected - outcomes) + penalty * parameters
        )
        if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
            return 1e100, np.full(3, 1e50, dtype=float)
        return objective, np.asarray(gradient, dtype=float)

    initial = np.zeros(3, dtype=float)
    initial_objective, _ = objective_and_gradient(initial)
    result = minimize(
        lambda parameters: objective_and_gradient(parameters)[0],
        initial,
        method="L-BFGS-B",
        jac=lambda parameters: objective_and_gradient(parameters)[1],
        options={
            "maxiter": maximum_iterations,
            "ftol": 1e-15,
            "gtol": 1e-10,
            "maxls": 100,
        },
    )
    if (
        not result.success
        or result.x.shape != (3,)
        or not np.all(np.isfinite(result.x))
    ):
        raise SoccerTransitionModelError(
            "dynamic-intensity optimizer failed closed: "
            f"status={result.status} message={result.message}"
        )
    final_objective, final_gradient = objective_and_gradient(
        np.asarray(result.x, dtype=float)
    )
    gradient_norm = float(np.max(np.abs(final_gradient)))
    if final_objective > initial_objective + 1e-10:
        raise SoccerTransitionModelError(
            "dynamic-intensity optimizer worsened the objective"
        )
    if gradient_norm > 1e-5:
        raise SoccerTransitionModelError(
            "dynamic-intensity optimizer failed gradient convergence: "
            f"projected_gradient_inf_norm={gradient_norm}"
        )
    return DynamicIntensityModel(
        coefficients=tuple(float(value) for value in result.x),
        l2_penalty=penalty,
        initial_objective=initial_objective,
        objective=final_objective,
        projected_gradient_inf_norm=gradient_norm,
        iterations=int(result.nit),
        optimizer_status=str(result.message),
    )


__all__ = [
    "COEFFICIENT_NAMES",
    "DynamicIntensityModel",
    "SoccerTransitionDistribution",
    "SoccerTransitionFeatures",
    "SoccerTransitionModelError",
    "TemperatureCalibration",
    "TRANSITION_CLASSES",
    "TRANSITION_HORIZON_SECONDS",
    "apply_multiclass_temperature",
    "extract_transition_features",
    "fit_dynamic_intensity",
    "fit_multiclass_temperature",
    "predict_transition_distribution",
]
