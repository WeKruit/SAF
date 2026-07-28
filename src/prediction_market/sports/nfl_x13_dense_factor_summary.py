"""Pure descriptive summaries joining X-13 factors to dense actual-trade paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
LEGACY_HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60)
_MEMBERSHIP_KEYS = (
    "factor_id",
    "factor_version",
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
)
_DENSE_JOIN_KEYS = (
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
)
_SHAPES = (
    "JUMP_DOMINANT",
    "GRADUAL",
    "MIXED",
    "UNIDENTIFIED_SPARSE",
)
_SUMMARY_COLUMNS = (
    "factor_id",
    "factor_version",
    "market_family",
    "venue",
    "horizon_seconds",
    "observed_n",
    "game_count",
    "mean",
    "median",
    "std",
    "p10",
    "p25",
    "p75",
    "p90",
    "positive_rate",
    "negative_rate",
    "abs_mean",
    "jump_dominant_share",
    "gradual_share",
    "mixed_share",
    "unidentified_sparse_share",
    "claim_boundary",
)


class DenseFactorSummaryError(ValueError):
    """A factor membership or dense-path input violates the summary contract."""


@dataclass(frozen=True, slots=True)
class DenseFactorSummary:
    """Eligible descriptive results and complete ineligible-path attrition."""

    summary: pd.DataFrame
    eligible: pd.DataFrame
    attrition: pd.DataFrame
    memberships: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DenseFactorSummaryError(f"{label} must be a DataFrame")
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise DenseFactorSummaryError(
            f"{label} missing required columns: {', '.join(missing)}"
        )


def _identifiers(frame: pd.DataFrame, fields: tuple[str, ...], *, label: str) -> None:
    for field in fields:
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise DenseFactorSummaryError(f"{label} has a blank {field}")
        frame[field] = frame[field].astype(str)


def _booleans(frame: pd.DataFrame, fields: tuple[str, ...], *, label: str) -> None:
    for field in fields:
        if not frame[field].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise DenseFactorSummaryError(f"{label}.{field} must be boolean")
        frame[field] = frame[field].astype(bool)


def _factor_memberships(factor_observations: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        factor_observations,
        {
            *_MEMBERSHIP_KEYS,
            "horizon_seconds",
            "predicate_nonexcluded_hit",
            "factor_eligible",
            "claim_boundary",
        },
        label="factor observations",
    )
    factors = factor_observations.copy()
    _identifiers(factors, _MEMBERSHIP_KEYS, label="factor observations")
    _booleans(
        factors,
        ("predicate_nonexcluded_hit", "factor_eligible"),
        label="factor observations",
    )
    if not factors["claim_boundary"].eq(CLAIM_BOUNDARY).all():
        raise DenseFactorSummaryError("factor observations must remain SOURCE_TIME_ONLY")
    horizons = pd.to_numeric(factors["horizon_seconds"], errors="coerce")
    if horizons.isna().any() or not np.isfinite(horizons).all() or not np.equal(horizons, np.floor(horizons)).all():
        raise DenseFactorSummaryError("factor observation horizons must be finite integers")
    factors["horizon_seconds"] = horizons.astype("int64")
    if not factors["horizon_seconds"].isin(LEGACY_HORIZONS_SECONDS).all():
        raise DenseFactorSummaryError("factor observations contain a nonlegacy horizon")
    rows: list[dict[str, object]] = []
    for key, child in factors.groupby(list(_MEMBERSHIP_KEYS), sort=True, dropna=False):
        # The legacy artifact contains the horizons registered for each
        # factor, not necessarily all seven for every definition.  Membership
        # is still required to be invariant across every horizon that *is*
        # registered for this factor/event identity; demanding unregistered
        # horizons would incorrectly discard valid definitions.
        if child["horizon_seconds"].empty:
            raise DenseFactorSummaryError("factor membership has no registered horizon")
        states = child[["predicate_nonexcluded_hit", "factor_eligible"]].drop_duplicates()
        if len(states) != 1:
            raise DenseFactorSummaryError("factor membership differs across legacy horizons")
        state = states.iloc[0]
        rows.append(
            {
                **dict(zip(_MEMBERSHIP_KEYS, key, strict=True)),
                "predicate_nonexcluded_hit": bool(state["predicate_nonexcluded_hit"]),
                "factor_eligible": bool(state["factor_eligible"]),
                "factor_member": bool(
                    state["predicate_nonexcluded_hit"] and state["factor_eligible"]
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _dense_paths(dense_paths: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        dense_paths,
        {
            "observation_id",
            *_DENSE_JOIN_KEYS,
            "market_family",
            "horizon_seconds",
            "cumulative_change",
            "observed",
            "forward_filled",
            "contaminated",
            "order_ambiguous",
            "censored",
            "shape_classification",
            "attrition_reason",
            "claim_boundary",
        },
        label="dense paths",
    )
    paths = dense_paths.copy()
    _identifiers(
        paths,
        ("observation_id", *_DENSE_JOIN_KEYS, "market_family", "shape_classification"),
        label="dense paths",
    )
    _booleans(
        paths,
        ("observed", "forward_filled", "contaminated", "order_ambiguous", "censored"),
        label="dense paths",
    )
    if not paths["claim_boundary"].eq(CLAIM_BOUNDARY).all():
        raise DenseFactorSummaryError("dense paths must remain SOURCE_TIME_ONLY")
    horizons = pd.to_numeric(paths["horizon_seconds"], errors="coerce")
    if horizons.isna().any() or not np.isfinite(horizons).all() or not np.equal(horizons, np.floor(horizons)).all():
        raise DenseFactorSummaryError("dense path horizons must be finite integers")
    paths["horizon_seconds"] = horizons.astype("int64")
    if not paths["horizon_seconds"].isin((1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)).all():
        raise DenseFactorSummaryError("dense paths contain an unregistered horizon")
    if not paths["shape_classification"].isin(_SHAPES).all():
        raise DenseFactorSummaryError("dense paths contain an unknown shape classification")
    if paths.duplicated("observation_id").any():
        raise DenseFactorSummaryError("dense paths have duplicate observation IDs")
    if paths.duplicated([*_DENSE_JOIN_KEYS, "horizon_seconds"]).any():
        raise DenseFactorSummaryError("dense paths have duplicate path grain")
    paths["effect"] = pd.to_numeric(paths["cumulative_change"], errors="coerce").astype(float)
    return paths


def _attrition_reason(row: pd.Series) -> str:
    if not bool(row["factor_member"]):
        return "FACTOR_NOT_MEMBER"
    if bool(row["forward_filled"]):
        return "FORWARD_FILLED"
    if not bool(row["observed"]):
        return "NOT_OBSERVED"
    if bool(row["contaminated"]):
        return "CONTAMINATED"
    if bool(row["order_ambiguous"]):
        return "ORDER_AMBIGUOUS"
    if bool(row["censored"]):
        return "CENSORED"
    if not math.isfinite(float(row["effect"])):
        return "NONFINITE_CUMULATIVE_CHANGE"
    return "ELIGIBLE"


def _summary(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
    rows: list[dict[str, object]] = []
    keys = ("factor_id", "factor_version", "market_family", "venue", "horizon_seconds")
    for key, child in eligible.groupby(list(keys), sort=True, dropna=False):
        effects = child["effect"]
        shapes = child["shape_classification"]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "observed_n": int(len(child)),
                "game_count": int(child["game_id"].nunique()),
                "mean": float(effects.mean()),
                "median": float(effects.median()),
                "std": float(effects.std(ddof=1)) if len(effects) > 1 else math.nan,
                "p10": float(effects.quantile(0.10)),
                "p25": float(effects.quantile(0.25)),
                "p75": float(effects.quantile(0.75)),
                "p90": float(effects.quantile(0.90)),
                "positive_rate": float((effects > 0).mean()),
                "negative_rate": float((effects < 0).mean()),
                "abs_mean": float(effects.abs().mean()),
                "jump_dominant_share": float((shapes == "JUMP_DOMINANT").mean()),
                "gradual_share": float((shapes == "GRADUAL").mean()),
                "mixed_share": float((shapes == "MIXED").mean()),
                "unidentified_sparse_share": float((shapes == "UNIDENTIFIED_SPARSE").mean()),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


def build_dense_factor_summary(
    factor_observations: pd.DataFrame,
    dense_paths: pd.DataFrame,
) -> DenseFactorSummary:
    """Summarize factor-member dense paths without loading or writing artifacts.

    Membership is a factor/event identity, not a horizon-specific outcome.  It
    must therefore agree over every legacy factor horizon before any dense path
    is joined.  The returned attrition table retains all joined noneligible rows
    so that no forward-filled, nonobserved, contaminated, ambiguous, censored,
    or nonfinite path can silently enter the descriptive output.
    """

    memberships = _factor_memberships(factor_observations)
    paths = _dense_paths(dense_paths)
    joined = memberships.merge(
        paths,
        on=list(_DENSE_JOIN_KEYS),
        how="inner",
        # Different registered factors can intentionally hit the same
        # episode/market stream, while the dense table contains one row per
        # horizon.  Both sides are therefore many at this coarse join grain;
        # each side's finer grain has been validated above.
        validate="many_to_many",
        suffixes=("_factor", "_dense"),
    )
    if joined.empty:
        empty_attrition = pd.DataFrame(
            columns=[*memberships.columns, *paths.columns, "eligible", "dense_attrition_reason"]
        )
        return DenseFactorSummary(
            summary=pd.DataFrame(columns=list(_SUMMARY_COLUMNS)),
            eligible=pd.DataFrame(columns=[*memberships.columns, *paths.columns]),
            attrition=empty_attrition,
            memberships=memberships,
        )
    joined["dense_attrition_reason"] = joined["attrition_reason"]
    joined["attrition_reason"] = joined.apply(_attrition_reason, axis=1)
    joined["eligible"] = joined["attrition_reason"].eq("ELIGIBLE")
    eligible = joined[joined["eligible"]].copy()
    attrition = joined[~joined["eligible"]].copy()
    return DenseFactorSummary(
        summary=_summary(eligible),
        eligible=eligible.reset_index(drop=True),
        attrition=attrition.reset_index(drop=True),
        memberships=memberships.reset_index(drop=True),
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "DenseFactorSummary",
    "DenseFactorSummaryError",
    "LEGACY_HORIZONS_SECONDS",
    "build_dense_factor_summary",
]
