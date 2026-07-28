"""Empirical distributions for the sealed X-13 exact-153 development cohort.

The builder is deliberately pure.  It accepts only already-eligible,
actual-trade source-time rows and returns descriptive and game-clustered
statistics.  It does not fetch data, read a holdout, infer L2, or fit a
parametric distribution to signed probability changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
_GROUP_KEYS = (
    "factor_id",
    "factor_version",
    "market_family",
    "venue",
    "horizon_seconds",
)
_PATH_KEYS = (
    "factor_id",
    "factor_version",
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
    "market_family",
)
_REQUIRED = {
    "observation_id",
    *_PATH_KEYS,
    "outcome_orientation",
    "horizon_seconds",
    "effect",
    "pre_probability",
    "actual_trade_count",
    "staleness_seconds",
    "liquidity",
    "game_time_seconds",
    "signed_margin",
    "down",
    "distance",
    "field_position",
    "shape_classification",
    "claim_boundary_dense",
    "eligible",
}
_MATERIAL_THRESHOLDS_PP = (0.5, 1.0, 2.0, 5.0)


class HistoricalStudyError(ValueError):
    """Eligible source-time rows cannot support the governed study."""


@dataclass(frozen=True, slots=True)
class HistoricalFactorStudy:
    summary: pd.DataFrame
    histogram: pd.DataFrame
    ecdf: pd.DataFrame
    bootstrap_histogram: pd.DataFrame
    game_effects: pd.DataFrame
    state_breakdown: pd.DataFrame
    path_probabilities: pd.DataFrame
    venue_pairs: pd.DataFrame
    venue_diagnostics: pd.DataFrame
    cases: pd.DataFrame


def _validate(frame: pd.DataFrame, *, bootstrap_samples: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise HistoricalStudyError("eligible factor paths must be non-empty")
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise HistoricalStudyError(
            "eligible factor paths missing columns: " + ", ".join(missing)
        )
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise HistoricalStudyError("bootstrap_samples must be positive")
    result = frame.copy()
    if not result["eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all() or not result["eligible"].astype(bool).all():
        raise HistoricalStudyError("all input rows must be eligible")
    if not result["claim_boundary_dense"].eq(CLAIM_BOUNDARY).all():
        raise HistoricalStudyError(
            "historical study inputs must remain SOURCE_TIME_ONLY"
        )
    for field in (
        "factor_id",
        "factor_version",
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "market_family",
        "outcome_orientation",
    ):
        if result[field].isna().any() or result[field].astype(str).str.strip().eq("").any():
            raise HistoricalStudyError(f"{field} must be non-empty")
        result[field] = result[field].astype(str)
    if not result["venue"].isin(("polymarket", "kalshi")).all():
        raise HistoricalStudyError("venue must be polymarket or kalshi")
    for field in (
        "effect",
        "pre_probability",
        "actual_trade_count",
        "staleness_seconds",
        "liquidity",
        "game_time_seconds",
        "signed_margin",
    ):
        result[field] = pd.to_numeric(result[field], errors="coerce")
        if result[field].isna().any() or not np.isfinite(result[field]).all():
            raise HistoricalStudyError(f"{field} must be finite")
    if not result["pre_probability"].between(0, 1).all():
        raise HistoricalStudyError("pre_probability must be in [0, 1]")
    if (result[["actual_trade_count", "staleness_seconds", "liquidity"]] < 0).any().any():
        raise HistoricalStudyError("trade activity fields must be nonnegative")
    horizons = pd.to_numeric(result["horizon_seconds"], errors="coerce")
    if horizons.isna().any() or not np.equal(horizons, np.floor(horizons)).all():
        raise HistoricalStudyError("horizon_seconds must be integers")
    result["horizon_seconds"] = horizons.astype("int64")
    factor_observation_grain = [
        "factor_id",
        "factor_version",
        "observation_id",
    ]
    if result.duplicated(factor_observation_grain).any():
        raise HistoricalStudyError("factor observation grain must be unique")
    result["factor_family"] = result["factor_id"].str.split(".").str[1].fillna("OTHER")
    if "quarter" in result.columns:
        quarter = pd.to_numeric(result["quarter"], errors="coerce")
        if quarter.isna().any() or not quarter.between(1, 5).all():
            raise HistoricalStudyError("quarter must be an integer from 1 through 5")
        result["quarter"] = quarter.astype("int64")
        result["game_phase"] = result["quarter"].map(
            {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "OT"}
        )
    else:
        result["game_phase"] = result["game_time_seconds"].map(_game_phase)
    result["time_bucket"] = result["game_time_seconds"].map(_time_bucket)
    result["score_state"] = result["signed_margin"].map(_score_state)
    result["down_bucket"] = result["down"].map(
        lambda value: "MISSING" if pd.isna(value) else str(int(value))
    )
    result["distance_bucket"] = result["distance"].map(_distance_bucket)
    result["field_position_bucket"] = result["field_position"].map(
        _field_position_bucket
    )
    result["pre_probability_bucket"] = result["pre_probability"].map(
        _probability_bucket
    )
    result["staleness_bucket"] = result["staleness_seconds"].map(
        _staleness_bucket
    )
    result["trade_count_bucket"] = result["actual_trade_count"].map(
        _trade_count_bucket
    )
    result["trade_size_bucket"] = result["liquidity"].map(_trade_size_bucket)
    return result


def _game_phase(seconds_remaining: object) -> str:
    value = float(seconds_remaining)
    if value > 2700:
        return "Q1"
    if value > 1800:
        return "Q2"
    if value > 900:
        return "Q3"
    if value >= 0:
        return "Q4"
    return "OT"


def _time_bucket(seconds_remaining: object) -> str:
    value = float(seconds_remaining)
    if value > 900:
        return ">15m"
    if value > 600:
        return "10–15m"
    if value > 300:
        return "5–10m"
    if value > 120:
        return "2–5m"
    return "≤2m"


def _score_state(margin: object) -> str:
    value = abs(float(margin))
    if value == 0:
        return "TIED"
    if value <= 8:
        return "ONE_SCORE"
    return "MULTI_SCORE"


def _distance_bucket(value: object) -> str:
    if pd.isna(value):
        return "MISSING"
    number = float(value)
    if number <= 3:
        return "≤3"
    if number <= 7:
        return "4–7"
    if number <= 10:
        return "8–10"
    return ">10"


def _field_position_bucket(value: object) -> str:
    if pd.isna(value):
        return "MISSING"
    number = float(value)
    if number <= 20:
        return "OPP_RED_ZONE"
    if number <= 50:
        return "OPP_TERRITORY"
    if number <= 80:
        return "OWN_TERRITORY"
    return "OWN_RED_ZONE"


def _probability_bucket(value: object) -> str:
    number = float(value)
    if number < 0.20:
        return "<20%"
    if number < 0.40:
        return "20–40%"
    if number < 0.60:
        return "40–60%"
    if number < 0.80:
        return "60–80%"
    return "≥80%"


def _staleness_bucket(value: object) -> str:
    number = float(value)
    if number <= 1:
        return "≤1s"
    if number <= 5:
        return "1–5s"
    if number <= 15:
        return "5–15s"
    return ">15s"


def _trade_count_bucket(value: object) -> str:
    number = float(value)
    if number <= 1:
        return "1"
    if number <= 3:
        return "2–3"
    if number <= 10:
        return "4–10"
    return ">10"


def _trade_size_bucket(value: object) -> str:
    number = float(value)
    if number <= 10:
        return "≤10"
    if number <= 100:
        return "10–100"
    if number <= 1_000:
        return "100–1k"
    return ">1k"


def _bh(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float, index=values.index)
    ordered = values.sort_values(kind="mergesort")
    count = len(ordered)
    adjusted = np.minimum.accumulate(
        (ordered.to_numpy(dtype=float) * count / np.arange(1, count + 1))[::-1]
    )[::-1]
    result = pd.Series(np.minimum(adjusted, 1.0), index=ordered.index)
    return result.reindex(values.index)


def _sign(value: float) -> int:
    return int(value > 0) - int(value < 0)


def _bootstrap(
    game_values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(game_values)
    indices = rng.integers(0, count, size=(samples, count))
    return game_values[indices].mean(axis=1)


def _histogram_rows(
    values: np.ndarray,
    identity: dict[str, object],
    *,
    value_name: str,
) -> list[dict[str, object]]:
    probability_points = values * 100.0
    clipped = np.clip(np.floor(probability_points), -50, 49).astype(int)
    rows: list[dict[str, object]] = []
    for left in sorted(set(clipped.tolist())):
        count = int((clipped == left).sum())
        rows.append(
            {
                **identity,
                "value_name": value_name,
                "bin_left_pp": float(left),
                "bin_right_pp": float(left + 1),
                "count": count,
                "probability": count / len(values),
            }
        )
    return rows


def _summary_tables(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    ecdf_rows: list[dict[str, object]] = []
    bootstrap_histogram_rows: list[dict[str, object]] = []
    game_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for key, child in frame.groupby(list(_GROUP_KEYS), sort=True, dropna=False):
        identity = dict(zip(_GROUP_KEYS, key, strict=True))
        values = child["effect"].to_numpy(dtype=float)
        median = float(np.median(values))
        by_game = (
            child.groupby("game_id", sort=True)["effect"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        for row in by_game.to_dict("records"):
            game_rows.append(
                {
                    **identity,
                    "game_id": row["game_id"],
                    "game_mean": float(row["mean"]),
                    "game_median": float(row["median"]),
                    "observed_n": int(row["count"]),
                }
            )
        game_values = by_game["mean"].to_numpy(dtype=float)
        draws = _bootstrap(
            game_values,
            samples=bootstrap_samples,
            rng=rng,
        )
        raw_mean = float(values.mean())
        overall_sign = _sign(raw_mean)
        loo_values = np.asarray(
            [
                float(by_game.loc[by_game["game_id"] != game_id, "mean"].mean())
                for game_id in by_game["game_id"]
            ],
            dtype=float,
        )
        loo_same = (
            1.0
            if len(game_values) == 1
            else float(np.mean([_sign(value) == overall_sign for value in loo_values]))
        )
        abs_game = np.abs(game_values)
        max_contribution = (
            0.0 if abs_game.sum() == 0 else float(abs_game.max() / abs_game.sum())
        )
        p_value = min(
            1.0,
            2.0
            * min(
                float(np.mean(draws <= 0)),
                float(np.mean(draws >= 0)),
            ),
        )
        row = {
            **identity,
            "factor_family": str(child["factor_family"].iloc[0]),
            "observed_n": int(len(values)),
            "game_count": int(len(game_values)),
            "mean": raw_mean,
            "median": median,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "mad": float(np.median(np.abs(values - median))),
            "p05": float(np.quantile(values, 0.05)),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p_positive": float(np.mean(values > 0)),
            "p_negative": float(np.mean(values < 0)),
            "p_zero": float(np.mean(values == 0)),
            "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
            "bootstrap_p_value": p_value,
            "loo_same_sign_rate": loo_same,
            "max_game_contribution": max_contribution,
            "jump_dominant_share": float(
                np.mean(child["shape_classification"].eq("JUMP_DOMINANT"))
            ),
            "gradual_share": float(
                np.mean(child["shape_classification"].eq("GRADUAL"))
            ),
            "mixed_share": float(
                np.mean(child["shape_classification"].eq("MIXED"))
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for threshold in _MATERIAL_THRESHOLDS_PP:
            delta = threshold / 100.0
            label = (
                str(int(threshold))
                if float(threshold).is_integer()
                else str(threshold).replace(".", "_")
            )
            row[f"p_gte_pos_{label}pp"] = float(np.mean(values >= delta))
            row[f"p_lte_neg_{label}pp"] = float(np.mean(values <= -delta))
            row[f"p_abs_gte_{label}pp"] = float(np.mean(np.abs(values) >= delta))
        summary_rows.append(row)
        histogram_rows.extend(
            _histogram_rows(values, identity, value_name="episode_effect")
        )
        bootstrap_histogram_rows.extend(
            _histogram_rows(draws, identity, value_name="cluster_bootstrap_mean")
        )
        sorted_values = np.sort(values)
        for index, value in enumerate(sorted_values, start=1):
            ecdf_rows.append(
                {
                    **identity,
                    "effect": float(value),
                    "cumulative_probability": index / len(sorted_values),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["bh_q_value"] = (
        summary.groupby(
            ["factor_family", "venue", "market_family", "horizon_seconds"],
            sort=True,
            group_keys=False,
        )["bootstrap_p_value"]
        .apply(_bh)
        .astype(float)
    )
    return (
        summary.sort_values(list(_GROUP_KEYS), kind="mergesort").reset_index(drop=True),
        pd.DataFrame(histogram_rows),
        pd.DataFrame(ecdf_rows),
        pd.DataFrame(bootstrap_histogram_rows),
        pd.DataFrame(game_rows),
    )


def _state_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = (
        ("game_phase",),
        ("time_bucket",),
        ("score_state",),
        ("down_bucket",),
        ("distance_bucket",),
        ("field_position_bucket",),
        ("pre_probability_bucket",),
        ("staleness_bucket",),
        ("trade_count_bucket",),
        ("trade_size_bucket",),
    )
    for fields in dimensions:
        keys = [*_GROUP_KEYS, *fields]
        for key, child in frame.groupby(keys, sort=True, dropna=False):
            identity = dict(zip(keys, key, strict=True))
            values = child["effect"].to_numpy(dtype=float)
            rows.append(
                {
                    **identity,
                    "breakdown_dimension": (
                        "down" if fields[0] == "down_bucket" else fields[0]
                    ),
                    "breakdown_value": str(identity[fields[0]]),
                    "observed_n": len(values),
                    "game_count": child["game_id"].nunique(),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p_positive": float(np.mean(values > 0)),
                    "p_abs_gte_1pp": float(np.mean(np.abs(values) >= 0.01)),
                }
            )
    return pd.DataFrame(rows)


def _case_rows(frame: pd.DataFrame, *, per_side: int = 3) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    lineage = [
        *_GROUP_KEYS,
        "observation_id",
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "outcome_orientation",
        "effect",
        "pre_probability",
        "game_phase",
        "time_bucket",
        "score_state",
        "actual_trade_count",
        "staleness_seconds",
        "liquidity",
    ]
    for _key, child in frame.groupby(list(_GROUP_KEYS), sort=True, dropna=False):
        ordered = child.sort_values(
            ["effect", "game_id", "episode_id"], kind="mergesort"
        )
        negative = ordered.head(per_side)[lineage].copy()
        negative["case_type"] = "MAX_NEGATIVE"
        positive = ordered.tail(per_side).sort_values(
            ["effect", "game_id", "episode_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )[lineage].copy()
        positive["case_type"] = "MAX_POSITIVE"
        rows.extend((negative, positive))
    if not rows:
        return pd.DataFrame(columns=[*lineage, "case_type"])
    return (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(
            ["factor_id", "factor_version", "observation_id", "case_type"]
        )
        .reset_index(drop=True)
    )


def _path_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["horizon_seconds"].isin((10, 60))].copy()
    pivot = subset.pivot_table(
        index=list(_PATH_KEYS),
        columns="horizon_seconds",
        values="effect",
        aggfunc="first",
    )
    if pivot.empty or 10 not in pivot.columns or 60 not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=[10, 60])
    if pivot.empty:
        return pd.DataFrame()
    pivot = pivot.reset_index()
    pivot["continuation"] = (
        (pivot[10] * pivot[60] >= 0)
        & (pivot[60].abs() >= pivot[10].abs())
    )
    pivot["reversal"] = pivot[10] * pivot[60] < 0
    pivot["overshoot"] = (
        (pivot[10] * pivot[60] >= 0)
        & (pivot[10].abs() > pivot[60].abs())
        & (pivot[10].abs() >= 0.01)
    )
    rows: list[dict[str, object]] = []
    keys = ("factor_id", "factor_version", "market_family", "venue")
    for key, child in pivot.groupby(list(keys), sort=True, dropna=False):
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "paired_path_n": len(child),
                "game_count": child["game_id"].nunique(),
                "continuation_rate": float(child["continuation"].mean()),
                "reversal_rate": float(child["reversal"].mean()),
                "overshoot_rate": float(child["overshoot"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _venue_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    moneyline = frame[frame["market_family"].eq("moneyline")].copy()
    pair_keys = (
        "factor_id",
        "factor_version",
        "game_id",
        "episode_id",
        "outcome_orientation",
        "market_family",
        "horizon_seconds",
    )
    pairs: list[dict[str, object]] = []
    for key, child in moneyline.groupby(list(pair_keys), sort=True, dropna=False):
        counts = child.groupby("venue").size()
        if set(counts.index) != {"polymarket", "kalshi"} or not counts.eq(1).all():
            continue
        by_venue = child.set_index("venue")
        poly = by_venue.loc["polymarket"]
        kalshi = by_venue.loc["kalshi"]
        pairs.append(
            {
                **dict(zip(pair_keys, key, strict=True)),
                "polymarket_effect": float(poly["effect"]),
                "kalshi_effect": float(kalshi["effect"]),
                "polymarket_minus_kalshi": float(poly["effect"] - kalshi["effect"]),
                "polymarket_pre_probability": float(poly["pre_probability"]),
                "kalshi_pre_probability": float(kalshi["pre_probability"]),
                "polymarket_trade_count": int(poly["actual_trade_count"]),
                "kalshi_trade_count": int(kalshi["actual_trade_count"]),
                "polymarket_trade_size": float(poly["liquidity"]),
                "kalshi_trade_size": float(kalshi["liquidity"]),
                "polymarket_staleness_seconds": float(poly["staleness_seconds"]),
                "kalshi_staleness_seconds": float(kalshi["staleness_seconds"]),
            }
        )
    venue_pairs = pd.DataFrame(pairs)
    diagnostics: list[dict[str, object]] = []
    raw_keys = ("factor_id", "factor_version", "market_family", "venue", "horizon_seconds")
    for key, child in frame.groupby(list(raw_keys), sort=True, dropna=False):
        diagnostics.append(
            {
                **dict(zip(raw_keys, key, strict=True)),
                "comparison_mode": "RAW",
                "observed_n": len(child),
                "game_count": child["game_id"].nunique(),
                "effect_mean": float(child["effect"].mean()),
                "effect_median": float(child["effect"].median()),
                "p_positive": float((child["effect"] > 0).mean()),
                "pre_probability_mean": float(child["pre_probability"].mean()),
                "trade_count_mean": float(child["actual_trade_count"].mean()),
                "trade_size_mean": float(child["liquidity"].mean()),
                "staleness_seconds_mean": float(child["staleness_seconds"].mean()),
            }
        )
    if not venue_pairs.empty:
        paired_keys = ("factor_id", "factor_version", "market_family", "horizon_seconds")
        for key, child in venue_pairs.groupby(
            list(paired_keys), sort=True, dropna=False
        ):
            diagnostics.append(
                {
                    **dict(zip(paired_keys, key, strict=True)),
                    "venue": "polymarket_minus_kalshi",
                    "comparison_mode": "SAME_EPISODE_PAIRED",
                    "observed_n": len(child),
                    "game_count": child["game_id"].nunique(),
                    "effect_mean": float(child["polymarket_minus_kalshi"].mean()),
                    "effect_median": float(child["polymarket_minus_kalshi"].median()),
                    "p_positive": float(
                        (child["polymarket_minus_kalshi"] > 0).mean()
                    ),
                    "pre_probability_mean": math.nan,
                    "trade_count_mean": math.nan,
                    "trade_size_mean": math.nan,
                    "staleness_seconds_mean": math.nan,
                }
            )
    return venue_pairs, pd.DataFrame(diagnostics)


def build_historical_factor_study(
    eligible_paths: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260727,
) -> HistoricalFactorStudy:
    """Build empirical distributions without assuming independent episodes."""

    frame = _validate(eligible_paths, bootstrap_samples=bootstrap_samples)
    summary, histogram, ecdf, bootstrap_histogram, game_effects = _summary_tables(
        frame,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    venue_pairs, venue_diagnostics = _venue_tables(frame)
    return HistoricalFactorStudy(
        summary=summary,
        histogram=histogram,
        ecdf=ecdf,
        bootstrap_histogram=bootstrap_histogram,
        game_effects=game_effects,
        state_breakdown=_state_breakdown(frame),
        path_probabilities=_path_probabilities(frame),
        venue_pairs=venue_pairs,
        venue_diagnostics=venue_diagnostics,
        cases=_case_rows(frame),
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HistoricalFactorStudy",
    "HistoricalStudyError",
    "build_historical_factor_study",
]
