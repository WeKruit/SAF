"""Game-clustered statistics for evaluated NFL X-15 development signals.

The public API accepts only finite, evaluated development trade-marked
diagnostics.  It does not accept sealed holdout rows or interpret repeated
model rows as independent sporting events.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd


BOOTSTRAP_SEED: Final[int] = 20260728
BOOTSTRAP_SAMPLES: Final[int] = 10_000
DEFAULT_MIN_SUPPORT_GAMES: Final[int] = 30
DEFAULT_MIN_SUPPORT_SIGNALS: Final[int] = 20
DEFAULT_MAX_Q_VALUE: Final[float] = 0.05
DEFAULT_MIN_LOO_SAME_SIGN: Final[float] = 0.80
DEFAULT_MAX_GAME_CONTRIBUTION: Final[float] = 0.25

_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "factor_id",
    "venue",
    "model_id",
)
_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    *_GROUP_COLUMNS,
    "game_id",
    "episode_id",
)
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *_EVENT_COLUMNS,
        "gross_markout",
        "evaluation_status",
        "dataset_partition",
    }
)


class X15StatisticsInputError(ValueError):
    """Rows cannot enter formal X-15 development effect statistics."""


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise X15StatisticsInputError(f"{field} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise X15StatisticsInputError(f"{field} must be a positive integer")
    return integer


def _validate_input(
    signals: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
    confidence_level: float,
    min_support_games: int,
    min_support_signals: int,
) -> tuple[int, int, float, int, int]:
    if not isinstance(signals, pd.DataFrame) or signals.empty:
        raise X15StatisticsInputError(
            "signals must be a nonempty pandas DataFrame"
        )
    missing = sorted(_REQUIRED_COLUMNS.difference(signals.columns))
    if missing:
        raise X15StatisticsInputError(f"missing required columns: {missing}")

    if not signals["dataset_partition"].eq("development").all():
        raise X15StatisticsInputError(
            "formal effect statistics accept development rows only"
        )
    if not signals["evaluation_status"].eq("EVALUATED").all():
        raise X15StatisticsInputError(
            "formal effect statistics accept EVALUATED rows only"
        )

    for field in (*_EVENT_COLUMNS,):
        valid = signals[field].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            raise X15StatisticsInputError(f"{field} must be a nonempty string")

    finite_markouts = signals["gross_markout"].map(
        lambda value: (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, Real)
            and math.isfinite(float(value))
        )
    )
    if not finite_markouts.all():
        raise X15StatisticsInputError(
            "gross_markout must contain only finite evaluated effects"
        )

    if signals.duplicated(list(_EVENT_COLUMNS), keep=False).any():
        raise X15StatisticsInputError(
            "duplicate sporting event within factor, venue, and model"
        )

    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise X15StatisticsInputError(
            "seed must be a nonnegative integer"
        )
    samples = _positive_integer(n_bootstrap, field="n_bootstrap")
    games = _positive_integer(
        min_support_games, field="min_support_games"
    )
    signal_count = _positive_integer(
        min_support_signals, field="min_support_signals"
    )
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(float(confidence_level))
        or not 0 < float(confidence_level) < 1
    ):
        raise X15StatisticsInputError(
            "confidence_level must be finite and strictly between zero and one"
        )
    return int(seed), samples, float(confidence_level), games, signal_count


def _rng_for_group(
    seed: int,
    group_key: tuple[str, str, str],
    *,
    stream: str,
) -> np.random.Generator:
    material = "\0".join((str(seed), stream, *group_key)).encode("utf-8")
    group_seed = int.from_bytes(
        hashlib.sha256(material).digest()[:8], byteorder="big"
    )
    return np.random.default_rng(group_seed)


def _cluster_arrays(group: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    ordered = group.sort_values(
        ["game_id", "episode_id"], kind="mergesort"
    )
    clusters = [
        rows["gross_markout"].to_numpy(dtype=float)
        for _, rows in ordered.groupby("game_id", sort=True)
    ]
    game_sums = np.asarray(
        [float(cluster.sum()) for cluster in clusters], dtype=float
    )
    return clusters, game_sums


def _bootstrap_intervals(
    clusters: list[np.ndarray],
    *,
    rng: np.random.Generator,
    samples: int,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    game_count = len(clusters)
    cluster_counts = np.asarray(
        [len(cluster) for cluster in clusters], dtype=int
    )
    cluster_sums = np.asarray(
        [float(cluster.sum()) for cluster in clusters], dtype=float
    )
    selections = rng.integers(
        0, game_count, size=(samples, game_count)
    )
    sampled_counts = cluster_counts[selections].sum(axis=1)
    sampled_sums = cluster_sums[selections].sum(axis=1)
    bootstrap_means = sampled_sums / sampled_counts
    bootstrap_medians = np.empty(samples, dtype=float)
    for index, selected_games in enumerate(selections):
        sampled_values = np.concatenate(
            [clusters[game_index] for game_index in selected_games]
        )
        bootstrap_medians[index] = float(np.median(sampled_values))

    tail = (1.0 - confidence_level) / 2.0
    mean_lower, mean_upper = np.quantile(
        bootstrap_means, [tail, 1.0 - tail]
    )
    median_lower, median_upper = np.quantile(
        bootstrap_medians, [tail, 1.0 - tail]
    )
    return (
        float(mean_lower),
        float(mean_upper),
        float(median_lower),
        float(median_upper),
    )


def _leave_one_game_out_same_sign_rate(
    clusters: list[np.ndarray],
    game_sums: np.ndarray,
) -> float:
    if len(clusters) < 2:
        return math.nan
    counts = np.asarray([len(cluster) for cluster in clusters], dtype=int)
    total_sum = float(game_sums.sum())
    total_count = int(counts.sum())
    full_sign = np.sign(total_sum / total_count)
    leave_one_out = (total_sum - game_sums) / (total_count - counts)
    return float(np.mean(np.sign(leave_one_out) == full_sign))


def _maximum_single_game_absolute_contribution(
    game_sums: np.ndarray,
) -> float:
    absolute = np.abs(game_sums)
    denominator = float(absolute.sum())
    if denominator == 0:
        return 0.0
    return float(absolute.max() / denominator)


def _cluster_sign_flip_p_value(
    game_sums: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> float:
    observed = abs(float(game_sums.sum()))
    signs = (
        rng.integers(
            0,
            2,
            size=(samples, len(game_sums)),
            dtype=np.int8,
        )
        * 2
        - 1
    )
    permuted = np.abs(signs @ game_sums)
    exceedances = int(np.count_nonzero(permuted >= observed))
    return float((exceedances + 1) / (samples + 1))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    hypothesis_count = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    ordered = p_values[order]
    adjusted = ordered * hypothesis_count / np.arange(
        1, hypothesis_count + 1
    )
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(hypothesis_count, dtype=float)
    result[order] = adjusted
    return result


def summarize_development_signals(
    signals: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    min_support_games: int = DEFAULT_MIN_SUPPORT_GAMES,
    min_support_signals: int = DEFAULT_MIN_SUPPORT_SIGNALS,
) -> pd.DataFrame:
    """Return factor-by-venue-by-model development effect diagnostics.

    Confidence intervals resample whole games with replacement.  The
    two-sided p-value randomly flips the sign of every signal in a game as
    one cluster, tests the absolute signal-weighted mean, and applies a
    plus-one Monte Carlo correction.  BH q-values are computed separately
    across the venue/model hypotheses belonging to each ``factor_id``.
    The leave-one-game-out rate compares each omitted-game mean's sign with
    the full-sample mean's sign.  Single-game concentration is the largest
    absolute game-level sum divided by the sum of absolute game-level sums.

    ``recommended_for_gating`` is a support recommendation only.  It does
    not assert statistical significance, executability, or holdout validity.
    """

    (
        checked_seed,
        checked_samples,
        checked_confidence,
        checked_min_games,
        checked_min_signals,
    ) = _validate_input(
        signals,
        seed=seed,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        min_support_games=min_support_games,
        min_support_signals=min_support_signals,
    )
    frame = signals.copy()
    frame["gross_markout"] = frame["gross_markout"].astype(float)
    frame["factor_family"] = frame["factor_id"].str.rsplit(
        ".", n=1
    ).str[0]

    records: list[dict[str, object]] = []
    for raw_key, group in frame.groupby(
        list(_GROUP_COLUMNS), sort=True, dropna=False
    ):
        group_key = tuple(str(value) for value in raw_key)
        clusters, game_sums = _cluster_arrays(group)
        (
            mean_lower,
            mean_upper,
            median_lower,
            median_upper,
        ) = _bootstrap_intervals(
            clusters,
            rng=_rng_for_group(
                checked_seed, group_key, stream="cluster-bootstrap"
            ),
            samples=checked_samples,
            confidence_level=checked_confidence,
        )
        support_games = len(clusters)
        support_signals = len(group)
        supported = (
            support_games >= checked_min_games
            and support_signals >= checked_min_signals
        )
        records.append(
            {
                "factor_id": group_key[0],
                "factor_family": str(
                    group["factor_family"].iloc[0]
                ),
                "venue": group_key[1],
                "model_id": group_key[2],
                "support_games": support_games,
                "support_signals": support_signals,
                "mean_effect": float(group["gross_markout"].mean()),
                "median_effect": float(group["gross_markout"].median()),
                "mean_ci_lower": mean_lower,
                "mean_ci_upper": mean_upper,
                "median_ci_lower": median_lower,
                "median_ci_upper": median_upper,
                "leave_one_game_out_same_sign_rate": (
                    _leave_one_game_out_same_sign_rate(
                        clusters, game_sums
                    )
                ),
                "max_single_game_absolute_contribution_ratio": (
                    _maximum_single_game_absolute_contribution(game_sums)
                ),
                "cluster_sign_flip_p_value": (
                    _cluster_sign_flip_p_value(
                        game_sums,
                        rng=_rng_for_group(
                            checked_seed,
                            group_key,
                            stream="cluster-sign-flip",
                        ),
                        samples=checked_samples,
                    )
                ),
                "bootstrap_samples": checked_samples,
                "bootstrap_seed": checked_seed,
                "bootstrap_confidence_level": checked_confidence,
                "min_support_games": checked_min_games,
                "min_support_signals": checked_min_signals,
                "recommended_for_gating": supported,
                "support_status": (
                    "SUPPORTED_FOR_GATING"
                    if supported
                    else "LOW_SUPPORT_NOT_RECOMMENDED"
                ),
            }
        )

    summary = pd.DataFrame(records).sort_values(
        list(_GROUP_COLUMNS), kind="mergesort"
    )
    summary["bh_q_value"] = math.nan
    summary["factor_family_hypothesis_count"] = 0
    for _, indexes in summary.groupby(
        "factor_family", sort=True
    ).groups.items():
        row_indexes = list(indexes)
        p_values = summary.loc[
            row_indexes, "cluster_sign_flip_p_value"
        ].to_numpy(dtype=float)
        summary.loc[row_indexes, "bh_q_value"] = _benjamini_hochberg(
            p_values
        )
        summary.loc[
            row_indexes, "factor_family_hypothesis_count"
        ] = len(row_indexes)
    summary["factor_family_hypothesis_count"] = summary[
        "factor_family_hypothesis_count"
    ].astype(int)
    summary["mean_ci_excludes_zero"] = (
        summary["mean_ci_lower"].gt(0)
        | summary["mean_ci_upper"].lt(0)
    )
    summary["individual_statistical_gate"] = (
        summary["recommended_for_gating"]
        & summary["mean_ci_excludes_zero"]
        & summary["bh_q_value"].le(DEFAULT_MAX_Q_VALUE)
        & summary["leave_one_game_out_same_sign_rate"].ge(
            DEFAULT_MIN_LOO_SAME_SIGN
        )
        & summary[
            "max_single_game_absolute_contribution_ratio"
        ].le(DEFAULT_MAX_GAME_CONTRIBUTION)
    )
    summary["cross_venue_same_sign"] = False
    for _, indexes in summary.groupby(
        ["factor_id", "model_id"], sort=True
    ).groups.items():
        row_indexes = list(indexes)
        venue_rows = summary.loc[row_indexes]
        if (
            set(venue_rows["venue"]) == {"kalshi", "polymarket"}
            and venue_rows["individual_statistical_gate"].all()
            and venue_rows["mean_effect"].map(np.sign).nunique() == 1
            and not venue_rows["mean_effect"].eq(0).any()
        ):
            summary.loc[
                row_indexes, "cross_venue_same_sign"
            ] = True
    summary["passes_development_gate"] = (
        summary["individual_statistical_gate"]
        & summary["cross_venue_same_sign"]
    )
    return summary.reset_index(drop=True)


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "DEFAULT_MIN_SUPPORT_GAMES",
    "DEFAULT_MIN_SUPPORT_SIGNALS",
    "DEFAULT_MAX_GAME_CONTRIBUTION",
    "DEFAULT_MAX_Q_VALUE",
    "DEFAULT_MIN_LOO_SAME_SIGN",
    "X15StatisticsInputError",
    "summarize_development_signals",
]
