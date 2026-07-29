"""Paired, game-clustered development-only model selection for NFL X-15."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260729
_SHA256_PREFIX_LENGTH = len("sha256:") + 64
_PAIR_COLUMNS = (
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
    "head",
)
_REQUIRED = {
    *_PAIR_COLUMNS,
    "nfl_week",
    "cohort",
    "authority_sha256",
    "model_id",
    "probability",
    "outcome",
}


class ModelSelectionError(ValueError):
    """Development model rows cannot support a paired selection claim."""


@dataclass(frozen=True, slots=True)
class PairedModelSelection:
    baseline_model_id: str
    candidate_model_id: str
    game_losses: pd.DataFrame
    mean_log_loss_improvement: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_probability_improved: float
    bootstrap_samples: int
    seed: int
    point_gate_passed: bool
    bootstrap_gate_passed: bool
    selected: bool


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_PREFIX_LENGTH
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ModelSelectionError(f"{field} must be a sha256 digest")
    return value


def _finite_probability(series: pd.Series, *, field: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any() or (numeric >= 1).any():
        raise ModelSelectionError(f"{field} must contain finite probabilities in (0, 1)")
    return numeric


def _validate(
    predictions: pd.DataFrame,
    *,
    authority_sha256: str,
    baseline_model_id: str,
    candidate_model_id: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    expected_authority = _require_sha256(authority_sha256, field="authority_sha256")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ModelSelectionError("predictions must be a nonempty DataFrame")
    missing = sorted(_REQUIRED.difference(predictions.columns))
    if missing:
        raise ModelSelectionError(f"predictions missing required columns: {missing}")
    forbidden = {"factor_id", "factor_version", "predicate_evidence"}.intersection(predictions.columns)
    if forbidden:
        raise ModelSelectionError("factor membership is claim-only and cannot enter selection")
    if baseline_model_id == candidate_model_id:
        raise ModelSelectionError("baseline and candidate model IDs must differ")
    if not isinstance(n_bootstrap, Integral) or isinstance(n_bootstrap, bool) or n_bootstrap != BOOTSTRAP_SAMPLES:
        raise ModelSelectionError("n_bootstrap must equal the frozen 10,000 paired draws")
    work = predictions.copy()
    for column in (
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
        "head",
        "cohort",
        "authority_sha256",
        "model_id",
    ):
        if not work[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ModelSelectionError(f"{column} must be a nonempty string")
    for column in ("landmark_seconds", "endpoint_seconds"):
        values = pd.to_numeric(work[column], errors="coerce")
        if values.isna().any() or not np.equal(values.to_numpy(), values.astype(int).to_numpy()).all() or (values <= 0).any():
            raise ModelSelectionError(f"{column} must be a positive integer")
    if not work["cohort"].eq("development").all():
        raise ModelSelectionError("model selection accepts development rows only")
    weeks = pd.to_numeric(work["nfl_week"], errors="coerce")
    if weeks.isna().any() or not np.equal(weeks.to_numpy(), weeks.astype(int).to_numpy()).all() or not weeks.between(1, 12).all():
        raise ModelSelectionError("development nfl_week must be integral in 1..12")
    if not work["authority_sha256"].eq(expected_authority).all():
        raise ModelSelectionError("authority_sha256 does not match the frozen assignment")
    _finite_probability(work["probability"], field="probability")
    outcomes = pd.to_numeric(work["outcome"], errors="coerce")
    if outcomes.isna().any() or not outcomes.isin((0, 1)).all():
        raise ModelSelectionError("outcome must be binary")
    work["outcome"] = outcomes.astype(int)
    work = work.loc[work["model_id"].isin((baseline_model_id, candidate_model_id))].copy()
    if work.empty:
        raise ModelSelectionError("baseline and candidate rows are required")
    duplicate = work.duplicated([*_PAIR_COLUMNS, "model_id"], keep=False)
    if duplicate.any():
        raise ModelSelectionError("each model must have one row per exact pair")
    return work


def paired_game_model_selection(
    predictions: pd.DataFrame,
    *,
    candidate_model_id: str,
    authority_sha256: str,
    baseline_model_id: str = "B0",
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> PairedModelSelection:
    """Compare B0 and one candidate on identical rows, clustered by game."""

    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ModelSelectionError("seed must be a nonnegative integer")
    if not isinstance(confidence_level, Real) or isinstance(confidence_level, bool) or not 0 < float(confidence_level) < 1:
        raise ModelSelectionError("confidence_level must be strictly between zero and one")
    work = _validate(
        predictions,
        authority_sha256=authority_sha256,
        baseline_model_id=baseline_model_id,
        candidate_model_id=candidate_model_id,
        n_bootstrap=n_bootstrap,
    )
    baseline = work.loc[work["model_id"].eq(baseline_model_id)].copy()
    candidate = work.loc[work["model_id"].eq(candidate_model_id)].copy()
    pairs = baseline.merge(
        candidate,
        on=list(_PAIR_COLUMNS),
        suffixes=("_b0", "_candidate"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not pairs["_merge"].eq("both").all():
        raise ModelSelectionError("B0 and candidate must score the same exact pairs")
    if not pairs["game_id"].eq(pairs["game_id"]).all() or not pairs["outcome_b0"].eq(pairs["outcome_candidate"]).all():
        raise ModelSelectionError("exact pairs must share game identity and outcome")
    for column in ("nfl_week", "cohort", "authority_sha256"):
        if not pairs[f"{column}_b0"].eq(pairs[f"{column}_candidate"]).all():
            raise ModelSelectionError(f"exact pairs disagree on {column}")
    outcome = pairs["outcome_b0"].to_numpy(dtype=float)
    baseline_probability = pairs["probability_b0"].to_numpy(dtype=float)
    candidate_probability = pairs["probability_candidate"].to_numpy(dtype=float)
    baseline_loss = -outcome * np.log(baseline_probability) - (1 - outcome) * np.log(1 - baseline_probability)
    candidate_loss = -outcome * np.log(candidate_probability) - (1 - outcome) * np.log(1 - candidate_probability)
    games = pd.DataFrame(
        {
            "game_id": pairs["game_id"].astype(str),
            "baseline_log_loss": baseline_loss,
            "candidate_log_loss": candidate_loss,
        }
    ).groupby("game_id", sort=True, as_index=False).mean()
    if len(games) < 2:
        raise ModelSelectionError("paired game bootstrap requires at least two games")
    games["log_loss_improvement"] = games["baseline_log_loss"] - games["candidate_log_loss"]
    improvements = games["log_loss_improvement"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(improvements), size=(n_bootstrap, len(improvements)))
    draws = improvements[indices].mean(axis=1)
    tail = (1.0 - float(confidence_level)) / 2.0
    point_estimate = float(improvements.mean())
    ci_low, ci_high = (float(value) for value in np.quantile(draws, (tail, 1.0 - tail)))
    probability_improved = float(np.mean(draws > 0.0))
    point_gate = point_estimate > 0.0
    bootstrap_gate = ci_low > 0.0
    return PairedModelSelection(
        baseline_model_id=baseline_model_id,
        candidate_model_id=candidate_model_id,
        game_losses=games,
        mean_log_loss_improvement=point_estimate,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_probability_improved=probability_improved,
        bootstrap_samples=n_bootstrap,
        seed=int(seed),
        point_gate_passed=point_gate,
        bootstrap_gate_passed=bootstrap_gate,
        selected=point_gate and bootstrap_gate,
    )


__all__ = ["ModelSelectionError", "PairedModelSelection", "paired_game_model_selection"]
