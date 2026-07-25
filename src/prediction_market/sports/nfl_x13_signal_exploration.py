"""Deterministic, source-time-only signal exploration for the X-13 NFL cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from itertools import combinations, product
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS

RESULT_LABEL = "PRELIMINARY_SOURCE_TIME_ONLY"
ANALYSIS_VERSION = "x13_signal_exploration_v2"
MAXIMUM_GAMES = 20
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUMMARY_GROUPS = (
    "episode_superclass",
    "venue",
    "market_family",
    "delay_seconds",
    "horizon_seconds",
)
_REQUIRED_SUPERCLASSES = ("score", "turnover", "routine")
_FROZEN_ACTUAL_HORIZONS_SECONDS = (1, 2, 5, 10, 30, 60)
_STRUCTURED_EVENT_FACTORS = (
    "interception",
    "blocked_kick",
    "penalty",
    "review",
    "timeout",
)
_CACHE_EPISODE_TYPE_FACTORS = (
    "touchdown",
    "field_goal",
    "punt",
    "safety",
    "score_turnover",
)
_UNAVAILABLE_STRUCTURED_EVENT_FACTORS = (
    "fumble",
    "turnover_on_downs",
)
_UNAVAILABLE_EVENT_FACTORS = (
    "pat",
    "two_point_conversion",
    "scoring_reversal",
    *_UNAVAILABLE_STRUCTURED_EVENT_FACTORS,
)
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_CHECKSUMS_BYTES = 2 * 1024 * 1024
_MAX_OBJECT_BYTES = 1024 * 1024 * 1024


class SignalExplorationError(ValueError):
    """The verified cache cannot support the frozen exploration."""


@dataclass(frozen=True, slots=True)
class SignalExplorationConfig:
    registered_delays_seconds: tuple[int, ...]
    registered_horizons_seconds: tuple[int, ...]
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95
    seed: int = 20260722
    minimum_episode_count: int = 20
    minimum_game_count: int = 6
    bh_q_threshold: float = 0.10

    def __post_init__(self) -> None:
        delays = self.registered_delays_seconds
        horizons = self.registered_horizons_seconds
        if (
            not delays
            or any(type(value) is not int or value < 0 for value in delays)
            or tuple(sorted(set(delays))) != delays
        ):
            raise SignalExplorationError(
                "registered_delays_seconds must be unique sorted non-negative integers"
            )
        if (
            not horizons
            or any(type(value) is not int or value <= 0 for value in horizons)
            or tuple(sorted(set(horizons))) != horizons
        ):
            raise SignalExplorationError(
                "registered_horizons_seconds must be unique sorted positive integers"
            )
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples != 10_000:
            raise SignalExplorationError("bootstrap_samples must equal 10000")
        if (
            type(self.confidence_level) is not float
            or not 0.5 < self.confidence_level < 1.0
        ):
            raise SignalExplorationError("confidence_level must be between 0.5 and 1")
        if type(self.seed) is not int or self.seed < 0:
            raise SignalExplorationError("seed must be a non-negative integer")
        if (
            type(self.minimum_episode_count) is not int
            or self.minimum_episode_count < 2
        ):
            raise SignalExplorationError("minimum_episode_count must be >= 2")
        if type(self.minimum_game_count) is not int or self.minimum_game_count < 2:
            raise SignalExplorationError("minimum_game_count must be >= 2")
        if (
            type(self.bh_q_threshold) is not float
            or not 0.0 < self.bh_q_threshold <= 1.0
        ):
            raise SignalExplorationError("bh_q_threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SignalExplorationBundle:
    manifest_path: Path
    bundle_sha256: str
    file_sha256: dict[str, str]


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    frame_name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise SignalExplorationError(f"{frame_name} must be a pandas DataFrame")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise SignalExplorationError(
            f"{frame_name} missing required columns: {', '.join(missing)}"
        )


def _strict_string_series(series: pd.Series, field: str) -> None:
    if series.isna().any() or not series.map(
        lambda value: type(value) is str and bool(value)
    ).all():
        raise SignalExplorationError(f"{field} must contain non-empty strings")


def _strict_numeric_series(
    series: pd.Series,
    field: str,
    *,
    nonnegative: bool = False,
    nullable: bool = False,
) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if (not nullable and converted.isna().any()) or not np.isfinite(
        converted.dropna().to_numpy(dtype=float)
    ).all():
        raise SignalExplorationError(f"{field} must contain finite numbers")
    if nonnegative and (converted < 0).any():
        raise SignalExplorationError(f"{field} must contain non-negative numbers")
    return converted


def _normalise_inputs(
    observed_cache: pd.DataFrame,
    episode_dim: pd.DataFrame,
    contract_orientation_dim: pd.DataFrame,
    config: SignalExplorationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    _require_columns(
        observed_cache,
        (
            "game_id",
            "episode_id",
            "contract_id",
            "outcome",
            "signal_focal_team",
            "scoring_team",
            "pre_offense_team",
            "pre_defense_team",
            "contract_measure",
            "source_time_start_utc",
            "source_time_end_utc",
            "pre_event_actual_trade",
            "first_post_event_trade",
            "vwap",
            "pre_game_seconds_remaining",
            "pre_quarter",
            "pre_score_margin_home",
            "pre_down",
            "pre_yardline_100",
            "venue",
            "market_family",
            "delay_seconds",
            "horizon_seconds",
            "signed_price_change",
            "maximum_excursion",
            "net_change_60s",
            "overshoot",
            "reversal",
            "trade_count",
            "volume",
            "staleness_seconds",
            "signal_event_eligible",
            "signal_candidate_eligible",
            "pre_trade_fresh_le_60s",
            "pre_trade_age_seconds",
            "pre_trade_age_status",
            "validity_status",
            "order_ambiguous",
            "contaminated",
        ),
        frame_name="observed_cache",
    )
    _require_columns(
        episode_dim,
        ("episode_id", "game_id", "episode_superclass"),
        frame_name="episode_dim",
    )
    _require_columns(
        contract_orientation_dim,
        (
            "contract_id",
            "outcome",
            "semantic_orientation_id",
            "orientation_audited",
        ),
        frame_name="contract_orientation_dim",
    )
    observed = observed_cache.copy(deep=True)
    episodes = episode_dim.copy(deep=True)
    orientations = contract_orientation_dim.copy(deep=True)
    for frame, columns, frame_name in (
        (
            observed,
                (
                    "game_id",
                    "episode_id",
                    "contract_id",
                    "outcome",
                    "venue",
                    "market_family",
                    "source_time_start_utc",
                    "source_time_end_utc",
                    "validity_status",
                ),
            "observed_cache",
        ),
        (
            episodes,
            ("episode_id", "game_id", "episode_superclass"),
            "episode_dim",
        ),
            (
                orientations,
                ("contract_id", "outcome"),
                "contract_orientation_dim",
            ),
    ):
        for column in columns:
            _strict_string_series(frame[column], f"{frame_name}.{column}")
    if episodes.duplicated(["game_id", "episode_id"]).any():
        raise SignalExplorationError(
            "episode_dim game_id + episode_id must be unique"
        )
    if orientations.duplicated(["contract_id", "outcome"]).any():
        raise SignalExplorationError(
            "contract_orientation_dim contract_id + outcome must be unique"
        )
    if orientations["orientation_audited"].isna().any() or not orientations[
        "orientation_audited"
    ].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise SignalExplorationError(
            "contract_orientation_dim.orientation_audited must contain booleans"
        )
    orientations["orientation_audited"] = orientations[
        "orientation_audited"
    ].astype(bool)
    orphan_episodes = set(
        observed[["game_id", "episode_id"]].itertuples(index=False, name=None)
    ).difference(
        episodes[["game_id", "episode_id"]].itertuples(index=False, name=None)
    )
    if orphan_episodes:
        raise SignalExplorationError(
            "observed_cache references unknown episode_id"
        )
    orphan_contracts = set(
        observed[["contract_id", "outcome"]].itertuples(index=False, name=None)
    ).difference(
        orientations[["contract_id", "outcome"]].itertuples(
            index=False,
            name=None,
        )
    )
    if orphan_contracts:
        raise SignalExplorationError(
            "observed_cache references unknown contract_id"
        )
    delays = pd.to_numeric(observed["delay_seconds"], errors="coerce")
    horizons = pd.to_numeric(observed["horizon_seconds"], errors="coerce")
    if (
        delays.isna().any()
        or not delays.map(lambda value: float(value).is_integer()).all()
    ):
        raise SignalExplorationError("delay_seconds must contain integers")
    if (
        horizons.isna().any()
        or not horizons.map(lambda value: float(value).is_integer()).all()
    ):
        raise SignalExplorationError("horizon_seconds must contain integers")
    observed["delay_seconds"] = delays.astype("int64")
    observed["horizon_seconds"] = horizons.astype("int64")
    if not set(observed["delay_seconds"]).issubset(
        config.registered_delays_seconds
    ):
        raise SignalExplorationError("observed_cache contains unregistered delay_seconds")
    if not set(observed["horizon_seconds"]).issubset(
        config.registered_horizons_seconds
    ):
        raise SignalExplorationError(
            "observed_cache contains unregistered horizon_seconds"
        )
    for field in (
        "signed_price_change",
        "maximum_excursion",
        "net_change_60s",
        "trade_count",
        "volume",
        "staleness_seconds",
    ):
        observed[field] = _strict_numeric_series(
            observed[field],
            f"observed_cache.{field}",
            nonnegative=field
            in {
                "trade_count",
                "volume",
                "staleness_seconds",
            },
            nullable=field
            in {
                "signed_price_change",
                "maximum_excursion",
                "net_change_60s",
                "staleness_seconds",
            },
        )
    observed_trade = (
        observed["validity_status"].eq("OBSERVED")
        & observed["trade_count"].gt(0)
    )
    if observed.loc[observed_trade, "staleness_seconds"].isna().any():
        raise SignalExplorationError(
            "observed trade rows require finite staleness_seconds"
        )
    for field in ("overshoot", "reversal"):
        if observed[field].isna().any() or not observed[field].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise SignalExplorationError(
                f"observed_cache.{field} must contain booleans"
            )
        observed[field] = observed[field].astype(bool)
    for field in (
        "signal_event_eligible",
        "signal_candidate_eligible",
        "order_ambiguous",
        "contaminated",
    ):
        if observed[field].isna().any() or not observed[field].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise SignalExplorationError(
                f"observed_cache.{field} must contain booleans"
            )
        observed[field] = observed[field].astype(bool)
    if not observed["pre_trade_fresh_le_60s"].map(
        lambda value: pd.isna(value) or isinstance(value, (bool, np.bool_))
    ).all():
        raise SignalExplorationError(
            "observed_cache.pre_trade_fresh_le_60s must contain booleans or null"
        )
    if (
        observed["signal_candidate_eligible"]
        & ~observed["pre_trade_fresh_le_60s"].eq(True)
    ).any():
        raise SignalExplorationError(
            "signal_candidate_eligible requires verified pre_trade_fresh_le_60s"
        )
    observed["pre_trade_age_seconds"] = pd.to_numeric(
        observed["pre_trade_age_seconds"], errors="coerce"
    )
    candidate_rows = observed["signal_candidate_eligible"]
    candidate_age = observed.loc[candidate_rows, "pre_trade_age_seconds"]
    if (
        candidate_age.isna().any()
        or (candidate_age < 0).any()
        or (candidate_age > 60).any()
        or not observed.loc[candidate_rows, "pre_trade_age_status"].eq(
            "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
        ).all()
    ):
        raise SignalExplorationError(
            "candidate rows require reconstructed numeric pre-trade age <=60s"
        )
    if not set(episodes["episode_superclass"]).issubset(_REQUIRED_SUPERCLASSES):
        raise SignalExplorationError(
            "episode_dim contains unsupported episode_superclass"
        )
    if set(episodes["episode_superclass"]) != set(_REQUIRED_SUPERCLASSES):
        raise SignalExplorationError(
            "episode_dim must cover score, turnover, and routine"
        )

    selected_games = tuple(sorted(episodes["game_id"].unique()))
    if selected_games != X13_GAME_IDS:
        raise SignalExplorationError(
            "episode_dim must contain the exact frozen 20-game set"
        )
    selected_set = set(selected_games)
    episodes = episodes[episodes["game_id"].isin(selected_set)].copy()
    observed = observed[observed["game_id"].isin(selected_set)].copy()
    if observed.empty:
        raise SignalExplorationError("selected games contain no observations")

    if not observed["signal_event_eligible"].any():
        raise SignalExplorationError("no signal_event_eligible observations")
    _strict_string_series(
        observed.loc[
            observed["signal_event_eligible"], "signal_focal_team"
        ],
        "observed_cache.signal_focal_team for eligible observations",
    )
    joined = observed.merge(
        episodes[["game_id", "episode_id", "episode_superclass"]],
        on=["game_id", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    joined = joined.merge(
        orientations[
            [
                "contract_id",
                "outcome",
                "semantic_orientation_id",
                "orientation_audited",
            ]
        ],
        on=["contract_id", "outcome"],
        how="left",
        validate="many_to_one",
    )
    sort_columns = [
        "game_id",
        "episode_id",
        "contract_id",
        "outcome",
        "venue",
        "market_family",
        "delay_seconds",
        "horizon_seconds",
    ]
    joined = joined.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    focal_source = np.select(
        [
            joined["episode_superclass"].eq("score"),
            joined["episode_superclass"].eq("turnover"),
            joined["episode_superclass"].eq("routine"),
        ],
        [
            joined["scoring_team"],
            joined["pre_defense_team"],
            joined["pre_offense_team"],
        ],
        default=None,
    )
    focal_mask = joined["signal_event_eligible"].eq(True).to_numpy()
    if (
        pd.isna(np.asarray(focal_source, dtype=object)[focal_mask]).any()
        or not np.array_equal(
            np.asarray(focal_source, dtype=object)[focal_mask],
            joined.loc[
                joined["signal_event_eligible"].eq(True),
                "signal_focal_team",
            ].to_numpy(dtype=object),
        )
    ):
        raise SignalExplorationError(
            "signal_focal_team does not match frozen event provenance"
        )
    return joined, episodes, orientations, selected_games


def _coverage(
    joined: pd.DataFrame,
    config: SignalExplorationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values in product(
        _REQUIRED_SUPERCLASSES,
        sorted(joined["venue"].unique()),
        sorted(joined["market_family"].unique()),
        config.registered_delays_seconds,
        config.registered_horizons_seconds,
    ):
        mask = np.ones(len(joined), dtype=bool)
        for column, value in zip(_SUMMARY_GROUPS, values):
            mask &= joined[column].to_numpy() == value
        subset = joined.loc[mask]
        rows.append(
            {
                **dict(zip(_SUMMARY_GROUPS, values)),
                "episode_count": int(subset["episode_id"].nunique()),
                "game_count": int(subset["game_id"].nunique()),
                "observation_count": int(len(subset)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        list(_SUMMARY_GROUPS), kind="mergesort"
    ).reset_index(drop=True)


def _bootstrap_summary(
    episode_level: pd.DataFrame,
    *,
    config: SignalExplorationConfig,
    group_index: int,
) -> tuple[
    float,
    float,
    float,
    bool | None,
    float,
    float,
    tuple[float, ...],
]:
    grouped_games = list(
        episode_level.groupby("game_id", sort=True)["signed_price_change"]
    )
    if len(grouped_games) < 2:
        return (
            math.nan,
            math.nan,
            math.nan,
            None,
            math.nan,
            math.nan,
            (),
        )
    rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, group_index])
    )
    game_sums = np.array(
        [float(values.sum()) for _, values in grouped_games],
        dtype=float,
    )
    game_counts = np.array(
        [len(values) for _, values in grouped_games],
        dtype=float,
    )
    multiplicities = rng.multinomial(
        len(grouped_games),
        np.full(len(grouped_games), 1.0 / len(grouped_games)),
        size=config.bootstrap_samples,
    )
    sampled = (multiplicities @ game_sums) / (
        multiplicities @ game_counts
    )
    alpha = 1.0 - config.confidence_level
    low, high = np.quantile(
        sampled,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method="linear",
    )
    nonpositive = int(np.count_nonzero(sampled <= 0.0))
    nonnegative = int(np.count_nonzero(sampled >= 0.0))
    raw_p = min(
        1.0,
        2.0
        * (min(nonpositive, nonnegative) + 1)
        / (config.bootstrap_samples + 1),
    )
    full_estimate = float(game_sums.sum() / game_counts.sum())
    full_sign = np.sign(full_estimate)
    loo = np.array(
        [
            (game_sums.sum() - game_sums[index])
            / (game_counts.sum() - game_counts[index])
            for index in range(len(grouped_games))
        ]
    )
    stable = bool(
        full_sign != 0
        and np.all(np.sign(loo) == full_sign)
    )
    stability_rate = (
        float(np.mean(np.sign(loo) == full_sign))
        if full_sign != 0
        else 0.0
    )
    absolute_total = float(
        np.abs(episode_level["signed_price_change"].to_numpy()).sum()
    )
    maximum_contribution = (
        float(np.abs(game_sums).max() / absolute_total)
        if absolute_total > 0
        else math.nan
    )
    return (
        float(low),
        float(high),
        float(raw_p),
        stable,
        stability_rate,
        maximum_contribution,
        tuple(float(value) for value in loo),
    )


def _bh_adjust(raw_p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(raw_p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if len(finite_positions) == 0:
        return result
    finite = values[finite_positions]
    order = np.argsort(finite, kind="mergesort")
    ordered = finite[order]
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    result[finite_positions] = adjusted[inverse]
    return result


def _reaction_summary(
    joined: pd.DataFrame,
    config: SignalExplorationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = joined.groupby(list(_SUMMARY_GROUPS), sort=True, dropna=False)
    for group_index, (key, group) in enumerate(grouped):
        episode_level = (
            group.groupby(["game_id", "episode_id"], sort=True)
            .agg(
                signed_price_change=("signed_price_change", "mean"),
                maximum_excursion=("maximum_excursion", "mean"),
                net_change_60s=("net_change_60s", "mean"),
                overshoot=("overshoot", "max"),
                reversal=("reversal", "max"),
                trade_count=("trade_count", "sum"),
                volume=("volume", "sum"),
                staleness_seconds=("staleness_seconds", "mean"),
            )
            .reset_index()
        )
        candidate_group = group[
            group["signal_candidate_eligible"].eq(True)
            & group["pre_trade_fresh_le_60s"].eq(True)
            & group["pre_trade_age_seconds"].le(60)
            & group["pre_trade_age_seconds"].ge(0)
            & group["pre_trade_age_status"].eq(
                "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
            )
            & group["outcome"].eq(group["signal_focal_team"])
        ]
        candidate_episode_level = (
            candidate_group.groupby(["game_id", "episode_id"], sort=True)
            .agg(signed_price_change=("signed_price_change", "mean"))
            .reset_index()
        )
        (
            low,
            high,
            raw_p,
            stable,
            stability_rate,
            maximum_contribution,
            loo_estimates,
        ) = _bootstrap_summary(
            episode_level,
            config=config,
            group_index=group_index,
        )
        (
            candidate_low,
            candidate_high,
            candidate_raw_p,
            candidate_loo_stable,
            candidate_stability_rate,
            candidate_maximum_contribution,
            candidate_loo_estimates,
        ) = _bootstrap_summary(
            candidate_episode_level,
            config=config,
            group_index=10_000 + group_index,
        )
        audited = bool(group["orientation_audited"].eq(True).all())
        semantic = bool(
            audited
            and group["semantic_orientation_id"].notna().all()
            and group["semantic_orientation_id"].map(
                lambda value: type(value) is str and bool(value)
            ).all()
        )
        rows.append(
            {
                **dict(zip(_SUMMARY_GROUPS, key)),
                "response_basis": (
                    "audited_semantic_orientation"
                    if semantic
                    else "raw_outcome_price_response"
                ),
                "orientation_available": semantic,
                "episode_count": int(group["episode_id"].nunique()),
                "game_count": int(group["game_id"].nunique()),
                "observation_count": int(len(group)),
                "signed_price_change_mean": float(
                    episode_level["signed_price_change"].mean()
                ),
                "maximum_excursion_mean": float(
                    episode_level["maximum_excursion"].mean()
                ),
                "net_change_60s_mean": float(
                    episode_level["net_change_60s"].mean()
                ),
                "overshoot_rate": float(episode_level["overshoot"].mean()),
                "reversal_rate": float(episode_level["reversal"].mean()),
                "trade_count_sum": float(episode_level["trade_count"].sum()),
                "volume_sum": float(episode_level["volume"].sum()),
                "staleness_seconds_mean": float(
                    episode_level["staleness_seconds"].mean()
                ),
                "game_cluster_ci_low": low,
                "game_cluster_ci_high": high,
                "loo_sign_stable": stable,
                "loo_sign_stability_rate": stability_rate,
                "maximum_single_game_contribution": maximum_contribution,
                "loo_estimates": loo_estimates,
                "raw_p_value": raw_p,
                "candidate_input_eligible": bool(
                    not candidate_group.empty
                    and key[_SUMMARY_GROUPS.index("market_family")] == "moneyline"
                    and key[_SUMMARY_GROUPS.index("delay_seconds")] == 0
                    and key[_SUMMARY_GROUPS.index("horizon_seconds")] == 10
                ),
                "candidate_episode_count": int(
                    candidate_episode_level["episode_id"].nunique()
                ),
                "candidate_game_count": int(
                    candidate_episode_level["game_id"].nunique()
                ),
                "candidate_effect": (
                    float(candidate_episode_level["signed_price_change"].mean())
                    if not candidate_episode_level.empty
                    else math.nan
                ),
                "candidate_game_cluster_ci_low": candidate_low,
                "candidate_game_cluster_ci_high": candidate_high,
                "candidate_raw_p_value": candidate_raw_p,
                "candidate_loo_sign_stable": candidate_loo_stable,
                "candidate_loo_sign_stability_rate": candidate_stability_rate,
                "candidate_maximum_single_game_contribution": (
                    candidate_maximum_contribution
                ),
                "candidate_loo_estimates": candidate_loo_estimates,
            }
        )
    result = pd.DataFrame(rows)
    result["frozen_comparison_id"] = result.apply(
        lambda row: (
            f"{row['episode_superclass']}|{row['venue']}|"
            f"{row['market_family']}|d={int(row['delay_seconds'])}|"
            f"h={int(row['horizon_seconds'])}"
        ),
        axis=1,
    )
    result["bh_q_value"] = _bh_adjust(result["raw_p_value"])
    primary_mask = (
        result["market_family"].eq("moneyline")
        & result["delay_seconds"].eq(0)
        & result["horizon_seconds"].eq(10)
    )
    candidate_p = np.where(
        primary_mask,
        result["candidate_raw_p_value"],
        np.nan,
    )
    result["candidate_bh_q_value"] = _bh_adjust(candidate_p)

    point_lookup = {
        (
            str(row["episode_superclass"]),
            str(row["venue"]),
            str(row["market_family"]),
            int(row["delay_seconds"]),
            int(row["horizon_seconds"]),
        ): float(row["candidate_effect"])
        for _, row in result.iterrows()
    }
    support_lookup = {
        (
            str(row["episode_superclass"]),
            str(row["venue"]),
            str(row["market_family"]),
            int(row["delay_seconds"]),
            int(row["horizon_seconds"]),
        ): (
            int(row["candidate_episode_count"]),
            int(row["candidate_game_count"]),
        )
        for _, row in result.iterrows()
    }

    def stability_gates(row: pd.Series) -> tuple[bool, bool]:
        event_class = str(row["episode_superclass"])
        venue = str(row["venue"])
        family = str(row["market_family"])
        full_sign = np.sign(float(row["candidate_effect"]))
        if full_sign == 0 or not math.isfinite(float(row["candidate_effect"])):
            return False, False
        required_delays = (0, 1, 2)
        required_horizons = (5, 10, 30)
        delay_points = [
            point_lookup.get((event_class, venue, family, delay, 10))
            for delay in required_delays
        ]
        horizon_points = [
            point_lookup.get((event_class, venue, family, 0, horizon))
            for horizon in required_horizons
        ]
        sensitivity_stable = bool(
            all(value is not None for value in delay_points + horizon_points)
            and all(
                np.sign(float(value)) == full_sign
                for value in delay_points + horizon_points
            )
        )
        venue_keys = [
            key
            for key in point_lookup
            if key[0] == event_class
            and key[2] == family
            and key[3] == 0
            and key[4] == 10
        ]
        peer_venues = sorted({key[1] for key in venue_keys})
        venue_stable = bool(
            len(peer_venues) >= 2
            and all(
                np.sign(
                    point_lookup[(event_class, peer, family, 0, 10)]
                )
                == full_sign
                and support_lookup[(event_class, peer, family, 0, 10)][0]
                >= 20
                and support_lookup[(event_class, peer, family, 0, 10)][1]
                >= 6
                for peer in peer_venues
            )
        )
        return sensitivity_stable, venue_stable

    gates = result.apply(stability_gates, axis=1)
    result["delay_horizon_sign_stable"] = [
        value[0] for value in gates
    ]
    result["cross_venue_sign_stable"] = [value[1] for value in gates]

    def candidate_label(row: pd.Series) -> str:
        if (
            int(row["candidate_episode_count"]) < 5
            or int(row["candidate_game_count"]) < 3
        ):
            return "CASE_ONLY"
        thresholds = (
            int(row["candidate_episode_count"]) >= config.minimum_episode_count
            and int(row["candidate_game_count"]) >= config.minimum_game_count
            and bool(row["candidate_input_eligible"])
            and bool(row["orientation_available"])
            and abs(float(row["candidate_effect"])) >= 0.005
            and bool(row["delay_horizon_sign_stable"])
            and bool(row["cross_venue_sign_stable"])
        )
        ci_excludes_zero = (
            math.isfinite(float(row["candidate_game_cluster_ci_low"]))
            and math.isfinite(float(row["candidate_game_cluster_ci_high"]))
            and (
                float(row["candidate_game_cluster_ci_low"]) > 0.0
                or float(row["candidate_game_cluster_ci_high"]) < 0.0
            )
        )
        stable = (
            row["candidate_loo_sign_stable"] is True
            or row["candidate_loo_sign_stable"] == True
        )
        contribution_stable = (
            math.isfinite(
                float(row["candidate_maximum_single_game_contribution"])
            )
            and float(row["candidate_maximum_single_game_contribution"])
            <= 0.25
            and float(row["candidate_loo_sign_stability_rate"]) >= 0.80
        )
        corrected = (
            math.isfinite(float(row["candidate_bh_q_value"]))
            and float(row["candidate_bh_q_value"]) <= config.bh_q_threshold
        )
        return (
            "SIGNAL_CANDIDATE"
            if (
                thresholds
                and ci_excludes_zero
                and stable
                and contribution_stable
                and corrected
            )
            else "DESCRIPTIVE"
        )

    result["candidate_label"] = result.apply(candidate_label, axis=1)
    return result.sort_values(list(_SUMMARY_GROUPS), kind="mergesort").reset_index(
        drop=True
    )


def _cross_venue_symmetry(joined: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    eligible = joined[
        joined["orientation_audited"].eq(True)
        & joined["semantic_orientation_id"].notna()
        & joined["semantic_orientation_id"].map(
            lambda value: type(value) is str and bool(value)
        )
    ].copy()
    columns = [
        "episode_superclass",
        "market_family",
        "delay_seconds",
        "horizon_seconds",
        "semantic_orientation_id",
        "venue_a",
        "venue_b",
        "venue_count",
        "shared_unit",
        "shared_episode_count",
        "shared_game_count",
        "mean_signed_price_change_a",
        "mean_signed_price_change_b",
        "mean_absolute_difference",
        "sign_agreement_rate",
    ]
    if eligible.empty:
        return (
            pd.DataFrame(columns=columns),
            "UNAVAILABLE_UNAUDITED_SEMANTIC_ORIENTATION",
        )
    pair_rows: list[dict[str, object]] = []
    keys = [
        "episode_superclass",
        "market_family",
        "delay_seconds",
        "horizon_seconds",
        "semantic_orientation_id",
    ]
    for key, group in eligible.groupby(keys, sort=True, dropna=False):
        per_episode = (
            group.groupby(["game_id", "episode_id", "venue"], sort=True)[
                "signed_price_change"
            ]
            .mean()
            .unstack("venue")
        )
        for venue_a, venue_b in combinations(sorted(per_episode.columns), 2):
            paired = per_episode[[venue_a, venue_b]].dropna()
            if paired.empty:
                continue
            signs = np.sign(paired[venue_a].to_numpy()) == np.sign(
                paired[venue_b].to_numpy()
            )
            pair_rows.append(
                {
                    **dict(zip(keys, key)),
                    "venue_a": venue_a,
                    "venue_b": venue_b,
                    "venue_count": 2,
                    "shared_unit": "episode_id",
                    "shared_episode_count": int(len(paired)),
                    "shared_game_count": int(
                        paired.index.get_level_values("game_id").nunique()
                    ),
                    "mean_signed_price_change_a": float(paired[venue_a].mean()),
                    "mean_signed_price_change_b": float(paired[venue_b].mean()),
                    "mean_absolute_difference": float(
                        (paired[venue_a] - paired[venue_b]).abs().mean()
                    ),
                    "sign_agreement_rate": float(np.mean(signs)),
                }
            )
    if not pair_rows:
        return (
            pd.DataFrame(columns=columns),
            "UNAVAILABLE_NO_SHARED_AUDITED_EPISODES",
        )
    return (
        pd.DataFrame(pair_rows)
        .sort_values(keys + ["venue_a", "venue_b"], kind="mergesort")
        .reset_index(drop=True),
        "AVAILABLE_AUDITED_SEMANTIC_ORIENTATION",
    )


def _cluster_inference(
    frame: pd.DataFrame,
    *,
    value: str,
    arm: str | None,
    config: SignalExplorationConfig,
    seed_offset: int,
) -> dict[str, object]:
    columns = [
        "point_estimate",
        "ci_low",
        "ci_high",
        "raw_p_value",
        "logo_sign_stability_rate",
        "maximum_single_game_contribution",
        "bootstrap_samples_requested",
        "bootstrap_samples_valid",
        "logo_samples_requested",
        "logo_samples_valid",
        "logo_estimates",
    ]
    if frame.empty or frame["game_id"].nunique() < 2:
        return {name: math.nan for name in columns}
    games = sorted(frame["game_id"].unique())
    if arm is None:
        sums = frame.groupby("game_id")[value].sum().reindex(games, fill_value=0).to_numpy()
        counts = frame.groupby("game_id")[value].count().reindex(games, fill_value=0).to_numpy()
        point = float(sums.sum() / counts.sum())

        def estimate(mult: np.ndarray) -> np.ndarray:
            return (mult @ sums) / (mult @ counts)

    else:
        positive = frame[frame[arm]]
        negative = frame[~frame[arm]]
        ps = positive.groupby("game_id")[value].sum().reindex(games, fill_value=0).to_numpy()
        pc = positive.groupby("game_id")[value].count().reindex(games, fill_value=0).to_numpy()
        ns = negative.groupby("game_id")[value].sum().reindex(games, fill_value=0).to_numpy()
        nc = negative.groupby("game_id")[value].count().reindex(games, fill_value=0).to_numpy()
        if pc.sum() == 0 or nc.sum() == 0:
            return {name: math.nan for name in columns}
        point = float(ps.sum() / pc.sum() - ns.sum() / nc.sum())

        def estimate(mult: np.ndarray) -> np.ndarray:
            pdn = mult @ pc
            ndn = mult @ nc
            with np.errstate(divide="ignore", invalid="ignore"):
                return (mult @ ps) / pdn - (mult @ ns) / ndn

    rng = np.random.default_rng(np.random.SeedSequence([config.seed, seed_offset]))
    mult = rng.multinomial(
        len(games),
        np.full(len(games), 1.0 / len(games)),
        size=config.bootstrap_samples,
    )
    samples = estimate(mult)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 20:
        return {name: math.nan for name in columns}
    alpha = 1 - config.confidence_level
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    p = min(
        1.0,
        2
        * (min(np.count_nonzero(samples <= 0), np.count_nonzero(samples >= 0)) + 1)
        / (len(samples) + 1),
    )
    logo: list[float] = []
    for index in range(len(games)):
        keep = np.ones((1, len(games)), dtype=int)
        keep[0, index] = 0
        result = estimate(keep)[0]
        if math.isfinite(float(result)):
            logo.append(float(result))
    sign = np.sign(point)
    stability = (
        float(np.mean(np.sign(logo) == sign)) if logo and sign != 0 else 0.0
    )
    if arm is not None:
        influences = [abs(point - estimate) for estimate in logo]
        influence_total = sum(influences)
        maximum = (
            max(influences) / influence_total
            if influence_total > 0
            else math.nan
        )
    else:
        contributions = [
            abs(float(frame[frame["game_id"] == game][value].sum()))
            for game in games
        ]
        denominator = float(np.abs(frame[value]).sum())
        maximum = (
            max(contributions) / denominator if denominator > 0 else math.nan
        )
    return {
        "point_estimate": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "raw_p_value": float(p),
        "logo_sign_stability_rate": stability,
        "maximum_single_game_contribution": maximum,
        "bootstrap_samples_requested": config.bootstrap_samples,
        "bootstrap_samples_valid": len(samples),
        "logo_samples_requested": len(games),
        "logo_samples_valid": len(logo),
        "logo_estimates": tuple(logo),
    }


def _label_frozen_table(
    table: pd.DataFrame,
    *,
    support_kind: str,
    config: SignalExplorationConfig,
) -> pd.DataFrame:
    if table.empty:
        table["bh_q_value"] = pd.Series(dtype=float)
        table["candidate_label"] = pd.Series(dtype=str)
        return table
    table = table.copy()
    table["bh_q_value"] = _bh_adjust(table["raw_p_value"])

    def label(row: pd.Series) -> str:
        episodes = int(row["episode_count"])
        games = int(row["game_count"])
        if episodes < 5 or games < 3:
            return "CASE_ONLY"
        if support_kind == "binary":
            support = (
                int(row["positive_episode_count"]) >= 40
                and int(row["negative_episode_count"]) >= 40
                and int(row["positive_game_count"]) >= 8
                and int(row["negative_game_count"]) >= 8
            )
        elif support_kind == "primary":
            support = (
                episodes >= 300
                and games >= 16
                and int(row["positive_episode_count"]) >= 20
                and int(row["negative_episode_count"]) >= 20
                and int(row["positive_game_count"]) >= 6
                and int(row["negative_game_count"]) >= 6
            )
        else:
            support = episodes >= 20 and games >= 6
        inference = (
            math.isfinite(float(row["ci_low"]))
            and (
                float(row["ci_low"]) > 0
                or float(row["ci_high"]) < 0
            )
            and float(row["bh_q_value"]) <= config.bh_q_threshold
            and abs(float(row["point_estimate"])) >= 0.005
            and float(row["logo_sign_stability_rate"]) >= 0.80
            and float(row["maximum_single_game_contribution"]) <= 0.25
            and int(row["bootstrap_samples_valid"])
            == int(row["bootstrap_samples_requested"])
            and int(row["logo_samples_valid"])
            == int(row["logo_samples_requested"])
        )
        return "SIGNAL_CANDIDATE" if support and inference else "DESCRIPTIVE"

    table["candidate_label"] = table.apply(label, axis=1)
    return table


def _frozen_tables(
    joined: pd.DataFrame,
    config: SignalExplorationConfig,
) -> dict[str, pd.DataFrame]:
    eligible = joined[
        joined["signal_candidate_eligible"].eq(True)
        & joined["pre_trade_age_seconds"].between(0, 60, inclusive="both")
        & joined["pre_trade_age_status"].eq(
            "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
        )
        & joined["outcome"].eq(joined["signal_focal_team"])
        & joined["market_family"].eq("moneyline")
        & joined["orientation_audited"].eq(True)
        & joined["contract_measure"].eq("winner")
    ].copy()
    primary_rows = eligible[
        eligible["delay_seconds"].eq(0) & eligible["horizon_seconds"].eq(10)
    ]
    dimensions = [
        "game_id",
        "episode_id",
        "episode_superclass",
        "pre_game_seconds_remaining",
        "pre_quarter",
        "pre_score_margin_home",
        "pre_down",
        "pre_yardline_100",
    ]
    venue_episode = (
        primary_rows.groupby(dimensions + ["venue"], dropna=False, sort=True)[
            "signed_price_change"
        ]
        .mean()
        .unstack("venue")
        .dropna()
        .reset_index()
    )
    if {"kalshi", "polymarket"}.issubset(venue_episode.columns):
        venue_episode["effect"] = venue_episode[["kalshi", "polymarket"]].mean(axis=1)
        venue_episode["venue_difference"] = (
            venue_episode["kalshi"] - venue_episode["polymarket"]
        )
    else:
        venue_episode = venue_episode.iloc[0:0].copy()
        venue_episode["effect"] = pd.Series(dtype=float)
        venue_episode["venue_difference"] = pd.Series(dtype=float)

    contrast_specs = (
        ("score_minus_routine", "score", "routine"),
        ("turnover_minus_routine", "turnover", "routine"),
        ("score_minus_turnover", "score", "turnover"),
    )
    primary: list[dict[str, object]] = []
    for index, (comparison, positive, negative) in enumerate(contrast_specs):
        subset = venue_episode[
            venue_episode["episode_superclass"].isin([positive, negative])
        ].copy()
        subset["arm"] = subset["episode_superclass"].eq(positive)
        stats = _cluster_inference(
            subset, value="effect", arm="arm", config=config, seed_offset=20_000 + index
        )
        primary.append(
            {
                "comparison_id": comparison,
                "episode_count": len(subset),
                "game_count": subset["game_id"].nunique(),
                "positive_episode_count": int(subset["arm"].sum()),
                "negative_episode_count": int((~subset["arm"]).sum()),
                "positive_game_count": subset.loc[
                    subset["arm"], "game_id"
                ].nunique(),
                "negative_game_count": subset.loc[
                    ~subset["arm"], "game_id"
                ].nunique(),
                **stats,
            }
        )

    path_rows = eligible[
        eligible["delay_seconds"].eq(0)
        & eligible["horizon_seconds"].isin([10, 60])
    ]
    path_pivot = (
        path_rows.groupby(
            ["game_id", "episode_id", "episode_superclass", "venue", "horizon_seconds"],
            sort=True,
        )
        .agg(effect=("signed_price_change", "mean"), trades=("trade_count", "sum"))
        .unstack("horizon_seconds")
    )
    path_pivot.columns = [f"{left}_{right}" for left, right in path_pivot.columns]
    path_pivot = path_pivot.reset_index()
    if {"effect_10", "effect_60", "trades_10", "trades_60"} <= set(path_pivot.columns):
        path_pivot = path_pivot[path_pivot["trades_60"] > path_pivot["trades_10"]]
        path_pivot["continuation"] = path_pivot["effect_60"] - path_pivot["effect_10"]
        path_episode = (
            path_pivot.groupby(
                ["game_id", "episode_id", "episode_superclass"], sort=True
            )
            .filter(lambda group: set(group["venue"]) >= {"kalshi", "polymarket"})
            .groupby(["game_id", "episode_id", "episode_superclass"], sort=True)[
                "continuation"
            ]
            .mean()
            .reset_index()
        )
    else:
        path_episode = pd.DataFrame(
            columns=["game_id", "episode_id", "episode_superclass", "continuation"]
        )
    path: list[dict[str, object]] = []
    for index, event_class in enumerate(_REQUIRED_SUPERCLASSES):
        subset = path_episode[path_episode["episode_superclass"].eq(event_class)]
        stats = _cluster_inference(
            subset,
            value="continuation",
            arm=None,
            config=config,
            seed_offset=30_000 + index,
        )
        path.append(
            {
                "comparison_id": f"continuation_10_60|{event_class}",
                "episode_count": len(subset),
                "game_count": subset["game_id"].nunique(),
                **stats,
            }
        )

    factors = {
        "second_half_or_ot": (
            ["pre_game_seconds_remaining", "pre_quarter"],
            lambda frame: (
                frame["pre_game_seconds_remaining"].le(1800)
                | frame["pre_quarter"].ge(5)
            ),
        ),
        "one_score_game": (
            ["pre_score_margin_home"],
            lambda frame: frame["pre_score_margin_home"].abs().le(8),
        ),
        "late_close": (
            ["pre_game_seconds_remaining", "pre_quarter", "pre_score_margin_home"],
            lambda frame: (
                frame["pre_quarter"].ge(5)
                | (
                    frame["pre_game_seconds_remaining"].le(300)
                    & frame["pre_score_margin_home"].abs().le(8)
                )
            ),
        ),
        "red_zone": (
            ["pre_yardline_100"],
            lambda frame: frame["pre_yardline_100"].le(20),
        ),
        "pressure_down": (
            ["pre_down"],
            lambda frame: frame["pre_down"].isin([3, 4]),
        ),
    }
    state: list[dict[str, object]] = []
    offset = 0
    for event_class in _REQUIRED_SUPERCLASSES:
        for factor, (required_state, build) in factors.items():
            subset = venue_episode[
                venue_episode["episode_superclass"].eq(event_class)
            ].dropna(subset=required_state).copy()
            subset["arm"] = build(subset)
            stats = _cluster_inference(
                subset,
                value="effect",
                arm="arm",
                config=config,
                seed_offset=40_000 + offset,
            )
            state.append(
                {
                    "comparison_id": f"{event_class}|{factor}",
                    "episode_count": len(subset),
                    "game_count": subset["game_id"].nunique(),
                    "positive_episode_count": int(subset["arm"].sum()),
                    "negative_episode_count": int((~subset["arm"]).sum()),
                    "positive_game_count": subset.loc[subset["arm"], "game_id"].nunique(),
                    "negative_game_count": subset.loc[~subset["arm"], "game_id"].nunique(),
                    **stats,
                }
            )
            offset += 1

    venue: list[dict[str, object]] = []
    for index, event_class in enumerate(_REQUIRED_SUPERCLASSES):
        subset = venue_episode[venue_episode["episode_superclass"].eq(event_class)]
        stats = _cluster_inference(
            subset,
            value="venue_difference",
            arm=None,
            config=config,
            seed_offset=50_000 + index,
        )
        venue.append(
            {
                "comparison_id": f"kalshi_minus_polymarket|{event_class}",
                "episode_count": len(subset),
                "game_count": subset["game_id"].nunique(),
                **stats,
            }
        )
    return {
        "primary_contrasts": _label_frozen_table(
            pd.DataFrame(primary), support_kind="primary", config=config
        ),
        "path_comparisons": _label_frozen_table(
            pd.DataFrame(path), support_kind="named", config=config
        ),
        "state_moderators": _label_frozen_table(
            pd.DataFrame(state), support_kind="binary", config=config
        ),
        "venue_comparisons": _label_frozen_table(
            pd.DataFrame(venue), support_kind="named", config=config
        ),
    }


def _factor_paths(actual: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "game_id",
        "episode_id",
        "contract_id",
        "outcome",
        "venue",
        "market_family",
    ]
    if actual.duplicated([*keys, "horizon_seconds"]).any():
        raise SignalExplorationError(
            "actual observations must be unique by path key and horizon"
        )
    path_source = actual.copy()
    if "actual_observation" in path_source:
        path_source["_path_change"] = path_source[
            "signed_price_change"
        ].where(path_source["actual_observation"].eq(True))
    else:
        path_source["_path_change"] = path_source["signed_price_change"]
    pivot = path_source.pivot(
        index=keys,
        columns="horizon_seconds",
        values="_path_change",
    ).reset_index()
    validity = path_source.pivot(
        index=keys,
        columns="horizon_seconds",
        values="validity_status",
    ).reset_index()
    for horizon in _FROZEN_ACTUAL_HORIZONS_SECONDS:
        if horizon not in pivot:
            pivot[horizon] = math.nan
        if horizon not in validity:
            validity[horizon] = pd.NA
        pivot[f"observed_{horizon}s"] = pivot[horizon].notna()
        pivot = pivot.rename(columns={horizon: f"change_{horizon}s"})
        pivot[f"validity_{horizon}s"] = validity[horizon].fillna(
            "MISSING_CELL"
        )
    observed_columns = [
        f"observed_{horizon}s" for horizon in _FROZEN_ACTUAL_HORIZONS_SECONDS
    ]
    change_columns = [
        f"change_{horizon}s" for horizon in _FROZEN_ACTUAL_HORIZONS_SECONDS
    ]
    validity_columns = [
        f"validity_{horizon}s" for horizon in _FROZEN_ACTUAL_HORIZONS_SECONDS
    ]
    pivot["path_complete"] = pivot[observed_columns].all(axis=1)

    def classify(row: pd.Series) -> str:
        if not bool(row["path_complete"]):
            return "INCOMPLETE_ACTUAL_PATH"
        signs = np.sign(row[change_columns].to_numpy(dtype=float))
        nonzero = signs[signs != 0]
        if len(nonzero) == 0:
            return "FLAT"
        if np.all(nonzero == 1):
            return "POSITIVE_PERSISTENT"
        if np.all(nonzero == -1):
            return "NEGATIVE_PERSISTENT"
        return "SIGN_REVERSAL"

    pivot["path_factor_value"] = pivot.apply(classify, axis=1)
    return pivot[
        [
            *keys,
            *change_columns,
            *observed_columns,
            *validity_columns,
            "path_complete",
            "path_factor_value",
        ]
    ].sort_values(keys, kind="mergesort").reset_index(drop=True)


def _normalise_event_factor_dim(
    event_factor_dim: pd.DataFrame | None,
    *,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    statuses = {
        factor: "UNAVAILABLE_SOURCE_FIELD"
        for factor in (
            *_STRUCTURED_EVENT_FACTORS,
            *_UNAVAILABLE_STRUCTURED_EVENT_FACTORS,
        )
    }
    if event_factor_dim is None:
        return None, statuses
    required = (
        "game_id",
        "episode_id",
        *_STRUCTURED_EVENT_FACTORS,
        *_UNAVAILABLE_STRUCTURED_EVENT_FACTORS,
        "lineage_build_id",
    )
    _require_columns(event_factor_dim, required, frame_name="event_factor_dim")
    factors = event_factor_dim.copy(deep=True)
    if factors.duplicated(["game_id", "episode_id"]).any():
        raise SignalExplorationError(
            "event_factor_dim game_id + episode_id must be unique"
        )
    expected = set(
        episodes[["game_id", "episode_id"]].itertuples(index=False, name=None)
    )
    observed = set(
        factors[["game_id", "episode_id"]].itertuples(index=False, name=None)
    )
    if not expected.issubset(observed):
        raise SignalExplorationError(
            "event_factor_dim must cover all analyzed frozen episodes"
        )
    factors = factors[
        factors.set_index(["game_id", "episode_id"]).index.isin(expected)
    ].copy()
    if not factors["lineage_build_id"].map(
        lambda value: type(value) is str and _SHA256_RE.fullmatch(value) is not None
    ).all():
        raise SignalExplorationError(
            "event_factor_dim lineage_build_id must be a SHA-256"
        )
    if factors["lineage_build_id"].nunique() != 1:
        raise SignalExplorationError(
            "event_factor_dim must bind one game-state build"
        )
    for factor in _STRUCTURED_EVENT_FACTORS:
        if factors[factor].isna().any() or not factors[factor].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise SignalExplorationError(
                f"event_factor_dim.{factor} must contain booleans"
            )
        factors[factor] = factors[factor].astype(bool)
        statuses[factor] = "AVAILABLE_BOUND_STRUCTURED_FIELD"
    for factor in _UNAVAILABLE_STRUCTURED_EVENT_FACTORS:
        if factors[factor].notna().any():
            raise SignalExplorationError(
                f"event_factor_dim.{factor} must remain unavailable"
            )
    return factors, statuses


def _load_bound_event_factor_dim(
    *,
    cache_root: str | Path,
    batch_id: str,
) -> pd.DataFrame:
    root = Path(cache_root)
    if root.parent.name != "correlation-cache":
        raise SignalExplorationError(
            "cache root is not inside the frozen X-13 artifact layout"
        )
    x13_root = root.parents[1]
    batch_root = x13_root / "batches" / batch_id.replace(":", "-")
    batch_manifest_path = batch_root / "batch_manifest.json"
    batch_manifest = json.loads(
        _read_bounded(
            batch_manifest_path,
            max_length=_MAX_MANIFEST_BYTES,
        )
    )
    if (
        type(batch_manifest) is not dict
        or batch_manifest.get("batch_id") != batch_id
        or batch_root.name != batch_id.replace(":", "-")
    ):
        raise SignalExplorationError("bound batch manifest identity changed")
    records = batch_manifest.get("auxiliary_artifacts")
    if type(records) is not list:
        raise SignalExplorationError("bound batch lacks auxiliary artifacts")
    lineage_records = [
        record
        for record in records
        if type(record) is dict
        and record.get("relative_path") == "reports/lineage-report-v1.json"
    ]
    if len(lineage_records) != 1:
        raise SignalExplorationError("bound batch lacks unique lineage report")
    lineage_record = lineage_records[0]
    lineage_length = lineage_record.get("byte_length")
    lineage_sha = lineage_record.get("sha256")
    if (
        type(lineage_length) is not int
        or lineage_length < 0
        or type(lineage_sha) is not str
        or _SHA256_RE.fullmatch(lineage_sha) is None
    ):
        raise SignalExplorationError("bound lineage record is invalid")
    lineage_payload = _read_bounded(
        batch_root / "reports" / "lineage-report-v1.json",
        max_length=_MAX_MANIFEST_BYTES,
        expected_length=lineage_length,
    )
    if _sha256_bytes(lineage_payload) != lineage_sha:
        raise SignalExplorationError("bound lineage report checksum mismatch")
    lineage = json.loads(lineage_payload)
    build_id = lineage.get("game_state_build_id") if type(lineage) is dict else None
    if type(build_id) is not str or _SHA256_RE.fullmatch(build_id) is None:
        raise SignalExplorationError("bound lineage lacks game-state build ID")
    state_root = x13_root / "game-state" / build_id.replace(":", "-")
    if len(X13_GAME_IDS) == MAXIMUM_GAMES:
        from prediction_market.sports.nfl_x13_game_build import (
            verify_x13_game_state_publication,
        )

        try:
            verified_manifest = verify_x13_game_state_publication(state_root)
        except (OSError, ValueError) as error:
            raise SignalExplorationError(
                "bound game-state publication verification failed"
            ) from error
        if verified_manifest.get("build_id") != build_id:
            raise SignalExplorationError(
                "bound game-state publication identity changed"
            )
    build_manifest = json.loads(
        _read_bounded(
            state_root / "build_manifest.json",
            max_length=_MAX_MANIFEST_BYTES,
        )
    )
    if (
        type(build_manifest) is not dict
        or build_manifest.get("build_id") != build_id
        or build_manifest.get("game_ids") != list(X13_GAME_IDS)
    ):
        raise SignalExplorationError("game-state build does not cover frozen games")

    factor_rows: list[dict[str, object]] = []
    for game_id in X13_GAME_IDS:
        game_root = state_root / "games" / game_id
        episodes = json.loads(
            _read_bounded(
                game_root / "finalized_episodes.json",
                max_length=_MAX_OBJECT_BYTES,
            )
        )
        events = json.loads(
            _read_bounded(
                game_root / "canonical_events.json",
                max_length=_MAX_OBJECT_BYTES,
            )
        )
        if type(episodes) is not list or type(events) is not list:
            raise SignalExplorationError(
                f"game-state lineage artifacts are malformed: {game_id}"
            )
        event_by_play: dict[str, dict[str, object]] = {}
        for event in events:
            if type(event) is not dict or event.get("game_id") != game_id:
                raise SignalExplorationError(
                    f"canonical event identity changed: {game_id}"
                )
            play_id = event.get("play_id")
            if type(play_id) is not str or not play_id or play_id in event_by_play:
                raise SignalExplorationError(
                    f"canonical play identity is invalid: {game_id}"
                )
            event_by_play[play_id] = event
        for episode in episodes:
            if type(episode) is not dict or episode.get("game_id") != game_id:
                raise SignalExplorationError(
                    f"finalized episode identity changed: {game_id}"
                )
            episode_id = episode.get("episode_id")
            play_ids = episode.get("play_ids")
            if (
                type(episode_id) is not str
                or not episode_id
                or type(play_ids) is not list
                or not play_ids
                or any(type(play_id) is not str for play_id in play_ids)
            ):
                raise SignalExplorationError(
                    f"finalized episode lineage is invalid: {game_id}"
                )
            try:
                lineage_events = [event_by_play[play_id] for play_id in play_ids]
            except KeyError as error:
                raise SignalExplorationError(
                    f"finalized episode references unknown play: {game_id}"
                ) from error
            for event in lineage_events:
                for field in ("blocked_kick", "penalty", "review", "timeout"):
                    if type(event.get(field)) is not bool:
                        raise SignalExplorationError(
                            f"canonical event structured field is invalid: {field}"
                        )
                if type(event.get("player_roles")) is not dict:
                    raise SignalExplorationError(
                        "canonical event player_roles is invalid"
                    )
            factor_rows.append(
                {
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "interception": any(
                        "interceptor" in event["player_roles"]
                        for event in lineage_events
                    ),
                    "blocked_kick": any(
                        bool(event["blocked_kick"]) for event in lineage_events
                    ),
                    "penalty": any(
                        bool(event["penalty"]) for event in lineage_events
                    ),
                    "review": any(
                        bool(event["review"]) for event in lineage_events
                    ),
                    "timeout": any(
                        bool(event["timeout"]) for event in lineage_events
                    ),
                    "fumble": pd.NA,
                    "turnover_on_downs": pd.NA,
                    "lineage_build_id": build_id,
                }
            )
    result = pd.DataFrame(factor_rows)
    if result.duplicated(["game_id", "episode_id"]).any():
        raise SignalExplorationError("bound event factor identity is duplicated")
    return result.sort_values(
        ["game_id", "episode_id"], kind="mergesort"
    ).reset_index(drop=True)


def _factor_contrast_ranking(
    long: pd.DataFrame,
    *,
    config: SignalExplorationConfig,
) -> pd.DataFrame:
    """Rank paired game-level factor contrasts, never raw bucket means."""

    source = long[
        long["market_family"].eq("moneyline")
        & long["horizon_seconds"].eq(10)
        & ~(
            long["factor_domain"].eq("MARKET_REGIME")
            & long["factor_key"].isin(["venue", "family"])
        )
    ].copy()
    rows: list[dict[str, object]] = []
    factor_groups = source[
        ["factor_domain", "factor_key"]
    ].drop_duplicates().sort_values(
        ["factor_domain", "factor_key"], kind="mergesort"
    )
    comparison_index = 0
    for domain, key in factor_groups.itertuples(index=False, name=None):
        factor = source[
            source["factor_domain"].eq(domain)
            & source["factor_key"].eq(key)
        ].copy()
        levels = sorted(
            set(factor["factor_value"]).difference(
                {"ABSENT", "INCOMPLETE_ACTUAL_PATH"}
            )
        )
        if key == "episode_superclass":
            levels = [
                level for level in levels if level in {"score", "turnover"}
            ]
        for level in levels:
            if key == "episode_superclass":
                reference_mask = factor["factor_value"].eq("routine")
                reference_value = "routine"
            elif "ABSENT" in set(factor["factor_value"]):
                reference_mask = factor["factor_value"].eq("ABSENT")
                reference_value = "ABSENT"
            else:
                reference_mask = (
                    factor["factor_value"].ne(level)
                    & factor["factor_value"].ne("INCOMPLETE_ACTUAL_PATH")
                )
                reference_value = "ALL_OTHER_LEVELS"
            target_mask = factor["factor_value"].eq(level)
            if not target_mask.any() or not reference_mask.any():
                continue
            episode_level = (
                factor.loc[
                    target_mask | reference_mask,
                    [
                        "game_id",
                        "episode_id",
                        "venue",
                        "factor_value",
                        "signed_price_change",
                    ],
                ]
                .assign(
                    contrast_arm=lambda frame: np.where(
                        frame["factor_value"].eq(level),
                        "TARGET",
                        "REFERENCE",
                    )
                )
                .groupby(
                    [
                        "game_id",
                        "episode_id",
                        "venue",
                        "contrast_arm",
                    ],
                    sort=True,
                )["signed_price_change"]
                .mean()
                .reset_index(name="effect")
            )
            target_episodes = episode_level[
                episode_level["contrast_arm"].eq("TARGET")
            ][["game_id", "episode_id"]].drop_duplicates()
            reference_episodes = episode_level[
                episode_level["contrast_arm"].eq("REFERENCE")
            ][["game_id", "episode_id"]].drop_duplicates()

            venue_effects: dict[str, float] = {}
            venue_game_counts: dict[str, int] = {}
            paired_venues: list[pd.DataFrame] = []
            for venue in ("kalshi", "polymarket"):
                by_game = (
                    episode_level[episode_level["venue"].eq(venue)]
                    .groupby(
                        ["game_id", "contrast_arm"], sort=True
                    )["effect"]
                    .mean()
                    .unstack("contrast_arm")
                )
                if {"TARGET", "REFERENCE"}.issubset(by_game.columns):
                    paired = by_game[["TARGET", "REFERENCE"]].dropna().copy()
                    paired["effect"] = (
                        paired["TARGET"] - paired["REFERENCE"]
                    )
                    paired = paired[["effect"]].reset_index()
                else:
                    paired = pd.DataFrame(columns=["game_id", "effect"])
                venue_effects[venue] = (
                    float(paired["effect"].mean())
                    if not paired.empty
                    else math.nan
                )
                venue_game_counts[venue] = int(len(paired))
                if not paired.empty:
                    paired["venue"] = venue
                    paired_venues.append(paired)
            if paired_venues:
                pooled = (
                    pd.concat(paired_venues, ignore_index=True)
                    .groupby("game_id", sort=True)["effect"]
                    .mean()
                    .reset_index()
                )
            else:
                pooled = pd.DataFrame(columns=["game_id", "effect"])
            stats = _cluster_inference(
                pooled,
                value="effect",
                arm=None,
                config=config,
                seed_offset=60_000 + comparison_index,
            )
            comparison_index += 1
            target_games = int(target_episodes["game_id"].nunique())
            reference_games = int(reference_episodes["game_id"].nunique())
            support = bool(
                len(target_episodes) >= config.minimum_episode_count
                and len(reference_episodes) >= config.minimum_episode_count
                and target_games >= config.minimum_game_count
                and reference_games >= config.minimum_game_count
                and len(pooled) >= config.minimum_game_count
                and all(
                    count >= config.minimum_game_count
                    for count in venue_game_counts.values()
                )
            )
            kalshi_effect = venue_effects["kalshi"]
            polymarket_effect = venue_effects["polymarket"]
            venue_sign_consistent = bool(
                math.isfinite(kalshi_effect)
                and math.isfinite(polymarket_effect)
                and np.sign(kalshi_effect) != 0
                and np.sign(kalshi_effect) == np.sign(polymarket_effect)
            )
            rows.append(
                {
                    "factor_domain": domain,
                    "factor_key": key,
                    "factor_value": level,
                    "reference_value": reference_value,
                    "family": "moneyline",
                    "horizon_seconds": 10,
                    "target_episode_count": int(len(target_episodes)),
                    "reference_episode_count": int(len(reference_episodes)),
                    "target_game_count": target_games,
                    "reference_game_count": reference_games,
                    "game_count": int(len(pooled)),
                    "kalshi_effect": kalshi_effect,
                    "polymarket_effect": polymarket_effect,
                    "kalshi_paired_game_count": venue_game_counts["kalshi"],
                    "polymarket_paired_game_count": venue_game_counts[
                        "polymarket"
                    ],
                    "cross_venue_sign_consistent": venue_sign_consistent,
                    "support_gate_pass": support,
                    **stats,
                }
            )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(
            columns=[
                "candidate_rank",
                "frozen_factor_id",
                "factor_domain",
                "factor_key",
                "factor_value",
                "reference_value",
                "candidate_label",
            ]
        )
    ranking["bh_q_value"] = _bh_adjust(ranking["raw_p_value"])
    ranking["frozen_factor_id"] = ranking.apply(
        lambda row: (
            f"{row['factor_domain']}|{row['factor_key']}|"
            f"{row['factor_value']}_VS_{row['reference_value']}|"
            "moneyline|h=10"
        ),
        axis=1,
    )
    ranking["candidate_label"] = "DISCOVERY_ONLY"
    ranking["_finite_q"] = ranking["bh_q_value"].fillna(math.inf)
    ranking["_absolute_effect"] = ranking["point_estimate"].abs().fillna(0.0)
    ranking = ranking.sort_values(
        [
            "support_gate_pass",
            "cross_venue_sign_consistent",
            "_finite_q",
            "_absolute_effect",
            "frozen_factor_id",
        ],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["candidate_rank"] = np.arange(1, len(ranking) + 1)
    return ranking.drop(columns=["_finite_q", "_absolute_effect"])


def _factor_tables(
    joined: pd.DataFrame,
    *,
    publication_rows: pd.DataFrame,
    episodes: pd.DataFrame,
    event_factor_dim: pd.DataFrame | None,
    config: SignalExplorationConfig,
) -> dict[str, pd.DataFrame]:
    actual = joined[
        joined["delay_seconds"].eq(0)
        & joined["horizon_seconds"].isin(_FROZEN_ACTUAL_HORIZONS_SECONDS)
    ].copy()
    if actual.empty:
        raise SignalExplorationError("no actual delay-zero observations")
    if (actual["trade_count"] <= 0).any():
        raise SignalExplorationError(
            "actual observations require at least one observed trade"
        )
    published_actual = publication_rows[
        publication_rows["delay_seconds"].eq(0)
        & publication_rows["horizon_seconds"].isin(
            _FROZEN_ACTUAL_HORIZONS_SECONDS
        )
    ].copy()
    published_actual["actual_observation"] = (
        published_actual["validity_status"].eq("OBSERVED")
        & published_actual["order_ambiguous"].eq(False)
        & published_actual["contaminated"].eq(False)
        & published_actual["trade_count"].gt(0)
        & published_actual["signed_price_change"].notna()
    )
    published_actual["forward_filled"] = False
    pre_price = pd.to_numeric(
        published_actual["pre_event_actual_trade"], errors="coerce"
    )
    published_actual["last_actual_trade"] = np.where(
        published_actual["actual_observation"],
        pre_price + published_actual["signed_price_change"],
        math.nan,
    )
    actual_output = published_actual[
        [
            "game_id",
            "episode_id",
            "contract_id",
            "outcome",
            "signal_focal_team",
            "venue",
            "market_family",
            "contract_measure",
            "semantic_orientation_id",
            "orientation_audited",
            "source_time_start_utc",
            "source_time_end_utc",
            "horizon_seconds",
            "pre_event_actual_trade",
            "first_post_event_trade",
            "vwap",
            "last_actual_trade",
            "signed_price_change",
            "maximum_excursion",
            "net_change_60s",
            "trade_count",
            "volume",
            "staleness_seconds",
            "validity_status",
            "order_ambiguous",
            "contaminated",
            "signal_candidate_eligible",
            "pre_trade_fresh_le_60s",
            "pre_trade_age_seconds",
            "pre_trade_age_status",
            "analysis_eligible",
            "analysis_exclusion_reason",
            "actual_observation",
            "forward_filled",
        ]
    ].rename(columns={"market_family": "family"})
    actual_output = actual_output.reset_index(drop=True)
    paths = _factor_paths(
        published_actual[published_actual["actual_observation"].eq(True)]
    )

    metrics = [
        "game_id",
        "episode_id",
        "venue",
        "market_family",
        "horizon_seconds",
        "signed_price_change",
        "trade_count",
        "volume",
    ]
    assignments: list[pd.DataFrame] = []
    availability: dict[tuple[str, str], tuple[str, str]] = {}

    def assign(
        frame: pd.DataFrame,
        *,
        domain: str,
        key: str,
        values: pd.Series,
        mask: pd.Series | None = None,
    ) -> None:
        selected = frame if mask is None else frame.loc[mask]
        value_series = values if mask is None else values.loc[mask]
        if selected.empty:
            return
        assignment = selected[metrics].copy()
        assignment["factor_domain"] = domain
        assignment["factor_key"] = key
        assignment["factor_value"] = value_series.astype(str).to_numpy()
        assignments.append(assignment)

    def register(
        domain: str,
        key: str,
        status: str,
        source: str,
    ) -> None:
        identity = (domain, key)
        if identity in availability:
            raise SignalExplorationError(f"duplicate frozen factor: {domain}|{key}")
        availability[identity] = (status, source)

    assign(
        actual,
        domain="EVENT",
        key="episode_superclass",
        values=actual["episode_superclass"],
    )
    register(
        "EVENT",
        "episode_superclass",
        "AVAILABLE_BOUND_CACHE_FIELD",
        "episode_superclass",
    )
    if "episode_type" in actual.columns:
        episode_type_valid = actual["episode_type"].map(
            lambda value: type(value) is str and bool(value)
        )
        episode_type_status = (
            "AVAILABLE_BOUND_CACHE_FIELD"
            if bool(episode_type_valid.all())
            else "PARTIAL_BOUND_CACHE_FIELD"
        )
        for factor in _CACHE_EPISODE_TYPE_FACTORS:
            assign(
                actual,
                domain="EVENT",
                key=factor,
                values=actual["episode_type"].eq(factor).map(
                    {True: "PRESENT", False: "ABSENT"}
                ),
                mask=episode_type_valid,
            )
            register(
                "EVENT",
                factor,
                episode_type_status,
                f"episode_type={factor}",
            )
    else:
        for factor in _CACHE_EPISODE_TYPE_FACTORS:
            register(
                "EVENT",
                factor,
                "UNAVAILABLE_SOURCE_FIELD",
                f"episode_type={factor}",
            )
    factor_dim, event_statuses = _normalise_event_factor_dim(
        event_factor_dim,
        episodes=episodes,
    )
    if factor_dim is not None:
        enriched = actual.merge(
            factor_dim[
                ["game_id", "episode_id", *_STRUCTURED_EVENT_FACTORS]
            ],
            on=["game_id", "episode_id"],
            how="left",
            validate="many_to_one",
        )
        for factor in _STRUCTURED_EVENT_FACTORS:
            assign(
                enriched,
                domain="EVENT",
                key=factor,
                values=enriched[factor].map(
                    {True: "PRESENT", False: "ABSENT"}
                ),
            )
            register(
                "EVENT",
                factor,
                event_statuses[factor],
                f"structured_event.{factor}",
            )
    else:
        for factor in _STRUCTURED_EVENT_FACTORS:
            register(
                "EVENT",
                factor,
                event_statuses[factor],
                f"structured_event.{factor}",
            )
    for factor in _UNAVAILABLE_EVENT_FACTORS:
        register(
            "EVENT",
            factor,
            "UNAVAILABLE_SOURCE_FIELD",
            (
                f"structured_event.{factor}"
                if factor in _UNAVAILABLE_STRUCTURED_EVENT_FACTORS
                else f"independent_event_field.{factor}"
            ),
        )

    quarter = actual["pre_quarter"].map(
        lambda value: f"Q{int(value)}" if int(value) <= 4 else "OVERTIME"
    )
    margin = actual["pre_score_margin_home"].abs()
    score_state = pd.Series(
        np.select(
            [margin.eq(0), margin.le(8)],
            ["TIED", "ONE_SCORE"],
            default="MULTI_SCORE",
        ),
        index=actual.index,
    )
    down = actual["pre_down"]
    down_state = pd.Series(
        np.select(
            [down.isna(), down.le(2)],
            ["NO_DOWN", "EARLY_DOWN"],
            default="LATE_DOWN",
        ),
        index=actual.index,
    )
    yardline = actual["pre_yardline_100"]
    field_state = pd.Series(
        np.select(
            [yardline.isna(), yardline.le(20), yardline.le(50)],
            ["UNKNOWN_FIELD_POSITION", "RED_ZONE", "OPPONENT_TERRITORY"],
            default="OWN_TERRITORY",
        ),
        index=actual.index,
    )
    time_remaining = pd.Series(
        np.select(
            [
                actual["pre_quarter"].ge(5),
                actual["pre_game_seconds_remaining"].le(120),
                actual["pre_game_seconds_remaining"].le(300),
                actual["pre_game_seconds_remaining"].le(900),
            ],
            ["OVERTIME", "FINAL_2_MINUTES", "FINAL_5_MINUTES", "FINAL_15_MINUTES"],
            default="EARLIER",
        ),
        index=actual.index,
    )
    red_zone = yardline.le(20).map({True: "PRESENT", False: "ABSENT"})
    staleness = actual["staleness_seconds"]
    staleness_band = pd.Series(
        np.select(
            [staleness.le(1), staleness.le(10), staleness.le(60)],
            ["LE_1S", "GT_1_TO_10S", "GT_10_TO_60S"],
            default="GT_60S",
        ),
        index=actual.index,
    )
    for key, values, source in (
        ("quarter", quarter, "pre_quarter"),
        ("time_remaining_band", time_remaining, "pre_game_seconds_remaining+pre_quarter"),
        ("score_margin_band", score_state, "pre_score_margin_home"),
        ("down_band", down_state, "pre_down"),
        ("field_position_band", field_state, "pre_yardline_100"),
        ("red_zone", red_zone, "pre_yardline_100"),
        ("staleness_band", staleness_band, "staleness_seconds"),
    ):
        assign(actual, domain="STATE", key=key, values=values)
        register("STATE", key, "AVAILABLE_BOUND_CACHE_FIELD", source)

    if "pre_possession_team" in actual.columns:
        possession_valid = actual["pre_possession_team"].map(
            lambda value: type(value) is str and bool(value)
        )
        assign(
            actual,
            domain="STATE",
            key="possession_side",
            values=actual["pre_possession_team"].eq(
                actual["signal_focal_team"]
            ).map({True: "FOCAL", False: "NON_FOCAL"}),
            mask=possession_valid,
        )
        register(
            "STATE",
            "possession_side",
            (
                "AVAILABLE_BOUND_CACHE_FIELD"
                if bool(possession_valid.all())
                else "PARTIAL_BOUND_CACHE_FIELD"
            ),
            "pre_possession_team+signal_focal_team",
        )
    else:
        register(
            "STATE",
            "possession_side",
            "UNAVAILABLE_SOURCE_FIELD",
            "pre_possession_team+signal_focal_team",
        )
    offense_valid = actual["pre_offense_team"].map(
        lambda value: type(value) is str and bool(value)
    )
    assign(
        actual,
        domain="STATE",
        key="offense_side",
        values=actual["pre_offense_team"].eq(
            actual["signal_focal_team"]
        ).map({True: "FOCAL", False: "NON_FOCAL"}),
        mask=offense_valid,
    )
    register(
        "STATE",
        "offense_side",
        (
            "AVAILABLE_BOUND_CACHE_FIELD"
            if bool(offense_valid.all())
            else "PARTIAL_BOUND_CACHE_FIELD"
        ),
        "pre_offense_team+signal_focal_team",
    )
    if "pre_distance" in actual.columns:
        distance = pd.to_numeric(actual["pre_distance"], errors="coerce")
        distance_valid = distance.notna() & np.isfinite(distance) & distance.ge(0)
        distance_band = pd.Series(
            np.select(
                [distance.le(3), distance.le(7), distance.le(10)],
                ["SHORT_LE_3", "MEDIUM_4_TO_7", "STANDARD_8_TO_10"],
                default="LONG_GT_10",
            ),
            index=actual.index,
        )
        assign(
            actual,
            domain="STATE",
            key="distance_band",
            values=distance_band,
            mask=distance_valid,
        )
        register(
            "STATE",
            "distance_band",
            (
                "AVAILABLE_BOUND_CACHE_FIELD"
                if bool(distance_valid.all())
                else "PARTIAL_BOUND_CACHE_FIELD"
            ),
            "pre_distance",
        )
    else:
        register(
            "STATE",
            "distance_band",
            "UNAVAILABLE_SOURCE_FIELD",
            "pre_distance",
        )
    if "pre_goal_to_go" in actual.columns:
        goal_valid = actual["pre_goal_to_go"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        assign(
            actual,
            domain="STATE",
            key="goal_to_go",
            values=actual["pre_goal_to_go"].map(
                {True: "PRESENT", False: "ABSENT"}
            ),
            mask=goal_valid,
        )
        register(
            "STATE",
            "goal_to_go",
            (
                "AVAILABLE_BOUND_CACHE_FIELD"
                if bool(goal_valid.all())
                else "PARTIAL_BOUND_CACHE_FIELD"
            ),
            "pre_goal_to_go",
        )
    else:
        register(
            "STATE",
            "goal_to_go",
            "UNAVAILABLE_SOURCE_FIELD",
            "pre_goal_to_go",
        )
    for key, source in (
        ("recent_activity_band", "pre_event_recent_activity"),
        ("timeouts_remaining", "pre_timeouts_remaining"),
        ("drive_progression", "drive_id+drive_play_index"),
        ("win_probability_band", "pre_event_win_probability"),
    ):
        register("STATE", key, "UNAVAILABLE_SOURCE_FIELD", source)

    assign(
        actual,
        domain="COMBINATION",
        key="event_x_score_margin",
        values=actual["episode_superclass"].astype(str) + "|" + score_state,
    )
    register(
        "COMBINATION",
        "event_x_score_margin",
        "AVAILABLE_BOUND_CACHE_FIELDS",
        "episode_superclass+pre_score_margin_home",
    )
    late_close_score = (
        actual["episode_superclass"].eq("score")
        & (
            actual["pre_quarter"].ge(5)
            | (
                actual["pre_game_seconds_remaining"].le(300)
                & actual["pre_score_margin_home"].abs().le(8)
            )
        )
    )
    red_zone_turnover = (
        actual["episode_superclass"].eq("turnover")
        & actual["pre_yardline_100"].le(20)
    )
    overtime = actual["pre_quarter"].ge(5)
    for key, values, source in (
        (
            "late_close_score",
            late_close_score,
            "episode_superclass+pre_game_seconds_remaining+pre_quarter+pre_score_margin_home",
        ),
        (
            "red_zone_turnover",
            red_zone_turnover,
            "episode_superclass+pre_yardline_100",
        ),
        ("overtime", overtime, "pre_quarter"),
    ):
        assign(
            actual,
            domain="COMBINATION",
            key=key,
            values=values.map({True: "PRESENT", False: "ABSENT"}),
        )
        register("COMBINATION", key, "AVAILABLE_BOUND_CACHE_FIELDS", source)

    sequence_required = {
        "game_id",
        "episode_id",
        "episode_superclass",
        "episode_type",
        "source_time_start_utc",
        "scoring_team",
    }
    if sequence_required.issubset(episodes.columns):
        sequence = episodes[list(sequence_required)].copy()
        sequence_valid = (
            sequence["episode_type"].map(
                lambda value: type(value) is str and bool(value)
            )
            & sequence["source_time_start_utc"].map(
                lambda value: type(value) is str and bool(value)
            )
        )
        if bool(sequence_valid.all()) and not sequence.duplicated(
            ["game_id", "source_time_start_utc"]
        ).any():
            sequence = sequence.sort_values(
                ["game_id", "source_time_start_utc", "episode_id"],
                kind="mergesort",
            )
            previous_superclass = sequence.groupby("game_id")[
                "episode_superclass"
            ].shift()
            sequence["back_to_back_turnover"] = (
                sequence["episode_superclass"].eq("turnover")
                & previous_superclass.eq("turnover")
            )
            sequence["answer_score"] = False
            scoring = sequence[
                sequence["episode_superclass"].eq("score")
                & sequence["scoring_team"].map(
                    lambda value: type(value) is str and bool(value)
                )
            ].copy()
            prior_scoring_team = scoring.groupby("game_id")[
                "scoring_team"
            ].shift()
            scoring["answer_score"] = (
                prior_scoring_team.notna()
                & scoring["scoring_team"].ne(prior_scoring_team)
            )
            sequence.loc[
                scoring.index, "answer_score"
            ] = scoring["answer_score"].astype(bool)
            sequence_flags = sequence[
                [
                    "game_id",
                    "episode_id",
                    "answer_score",
                    "back_to_back_turnover",
                ]
            ]
            sequenced_actual = actual.merge(
                sequence_flags,
                on=["game_id", "episode_id"],
                how="left",
                validate="many_to_one",
            )
            for key in ("answer_score", "back_to_back_turnover"):
                assign(
                    sequenced_actual,
                    domain="COMBINATION",
                    key=key,
                    values=sequenced_actual[key].map(
                        {True: "PRESENT", False: "ABSENT"}
                    ),
                )
                register(
                    "COMBINATION",
                    key,
                    "AVAILABLE_BOUND_EPISODE_SEQUENCE",
                    (
                        "episode_superclass+scoring_team+source_time_start_utc"
                        if key == "answer_score"
                        else "episode_superclass+source_time_start_utc"
                    ),
                )
        else:
            for key in ("answer_score", "back_to_back_turnover"):
                register(
                    "COMBINATION",
                    key,
                    "UNAVAILABLE_SOURCE_FIELD",
                    "verified_full_episode_sequence",
                )
    else:
        for key in ("answer_score", "back_to_back_turnover"):
            register(
                "COMBINATION",
                key,
                "UNAVAILABLE_SOURCE_FIELD",
                "verified_full_episode_sequence",
            )
    for key, source in (
        (
            "third_fourth_down_failure",
            "independent_turnover_on_downs+pre_down",
        ),
        ("two_minute_drive", "drive_id+drive_start_time_remaining"),
    ):
        register("COMBINATION", key, "UNAVAILABLE_SOURCE_FIELD", source)

    if "pre_event_actual_trade" in actual:
        pre_price = pd.to_numeric(
            actual["pre_event_actual_trade"], errors="coerce"
        )
        price_mask = pre_price.notna() & np.isfinite(pre_price)
        price_band = pd.Series(
            np.select(
                [pre_price.lt(0.25), pre_price.le(0.75)],
                ["LOW", "MID"],
                default="HIGH",
            ),
            index=actual.index,
        )
        assign(
            actual,
            domain="MARKET_REGIME",
            key="pre_event_probability_band",
            values=price_band,
            mask=price_mask,
        )
        market_status = (
            "AVAILABLE_BOUND_ACTUAL_PRICE"
            if bool(price_mask.all())
            else "PARTIAL_SOURCE_FIELD"
        )
    else:
        market_status = "UNAVAILABLE_SOURCE_FIELD"
    register(
        "MARKET_REGIME",
        "pre_event_probability_band",
        market_status,
        "pre_event_actual_trade",
    )
    assign(
        actual,
        domain="MARKET_REGIME",
        key="staleness_band",
        values=staleness_band,
    )
    register(
        "MARKET_REGIME",
        "staleness_band",
        "AVAILABLE_BOUND_CACHE_FIELD",
        "staleness_seconds",
    )
    assign(
        actual,
        domain="MARKET_REGIME",
        key="venue",
        values=actual["venue"],
    )
    register(
        "MARKET_REGIME",
        "venue",
        "AVAILABLE_BOUND_CACHE_FIELD",
        "venue",
    )
    assign(
        actual,
        domain="MARKET_REGIME",
        key="family",
        values=actual["market_family"],
    )
    register(
        "MARKET_REGIME",
        "family",
        "AVAILABLE_BOUND_CACHE_FIELD",
        "market_family",
    )
    register(
        "MARKET_REGIME",
        "recent_activity_band",
        "UNAVAILABLE_SOURCE_FIELD",
        "pre_event_recent_activity",
    )

    path_keys = [
        "game_id",
        "episode_id",
        "contract_id",
        "outcome",
        "venue",
        "market_family",
    ]
    path_enriched = actual.merge(
        paths[[*path_keys, "path_factor_value"]],
        on=path_keys,
        how="left",
        validate="many_to_one",
    )
    assign(
        path_enriched,
        domain="PATH",
        key="six_horizon_actual_path",
        values=path_enriched["path_factor_value"],
    )
    register(
        "PATH",
        "six_horizon_actual_path",
        "AVAILABLE_ACTUAL_OBSERVATIONS",
        "signed_price_change@1/2/5/10/30/60s",
    )

    long = pd.concat(assignments, ignore_index=True)
    group_columns = [
        "game_id",
        "factor_domain",
        "factor_key",
        "factor_value",
        "venue",
        "market_family",
        "horizon_seconds",
    ]
    game_factors = (
        long.groupby(group_columns, sort=True, dropna=False)
        .agg(
            episode_count=("episode_id", "nunique"),
            observation_count=("episode_id", "size"),
            signed_price_change_mean=("signed_price_change", "mean"),
            signed_price_change_median=("signed_price_change", "median"),
            positive_rate=("signed_price_change", lambda values: values.gt(0).mean()),
            trade_count_sum=("trade_count", "sum"),
            volume_sum=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"market_family": "family"})
    )
    game_factors["source_delay_seconds"] = 0
    game_factors["observation_policy"] = "ACTUAL_ONLY_NO_FORWARD_FILL"
    game_factors = game_factors.sort_values(
        [
            "game_id",
            "factor_domain",
            "factor_key",
            "factor_value",
            "venue",
            "family",
            "horizon_seconds",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    summary_groups = [
        "factor_domain",
        "factor_key",
        "factor_value",
        "venue",
        "family",
        "horizon_seconds",
    ]
    cross_game = (
        game_factors.groupby(summary_groups, sort=True, dropna=False)
        .agg(
            game_count=("game_id", "nunique"),
            episode_count=("episode_count", "sum"),
            observation_count=("observation_count", "sum"),
            cross_game_mean=("signed_price_change_mean", "mean"),
            cross_game_median=("signed_price_change_mean", "median"),
            positive_game_rate=(
                "signed_price_change_mean",
                lambda values: values.gt(0).mean(),
            ),
        )
        .reset_index()
    )
    cross_game["frozen_factor_id"] = cross_game.apply(
        lambda row: (
            f"{row['factor_domain']}|{row['factor_key']}|{row['factor_value']}|"
            f"{row['venue']}|{row['family']}|h={int(row['horizon_seconds'])}"
        ),
        axis=1,
    )
    cross_game["candidate_label"] = np.where(
        (cross_game["game_count"] >= config.minimum_game_count)
        & (cross_game["episode_count"] >= config.minimum_episode_count),
        "DESCRIPTIVE",
        "CASE_ONLY",
    )
    cross_game = cross_game.sort_values(
        summary_groups, kind="mergesort"
    ).reset_index(drop=True)
    ranking = _factor_contrast_ranking(long, config=config)

    coverage_rows: list[dict[str, object]] = []
    for (domain, key), (status, source) in sorted(availability.items()):
        subset = long[
            long["factor_domain"].eq(domain) & long["factor_key"].eq(key)
        ]
        coverage_rows.append(
            {
                "factor_domain": domain,
                "factor_key": key,
                "availability_status": status,
                "source_field": source,
                "game_count": int(subset["game_id"].nunique()),
                "episode_count": int(subset["episode_id"].nunique()),
                "observation_count": int(len(subset)),
            }
        )
    factor_coverage = pd.DataFrame(coverage_rows).sort_values(
        ["factor_domain", "factor_key"], kind="mergesort"
    ).reset_index(drop=True)

    market_coverage = (
        published_actual.groupby(
            [
                "game_id",
                "venue",
                "market_family",
                "validity_status",
                "order_ambiguous",
                "contaminated",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            episode_count=("episode_id", "nunique"),
            contract_count=("contract_id", "nunique"),
            cell_count=("episode_id", "size"),
            actual_observation_count=("actual_observation", "sum"),
            analysis_eligible_count=("analysis_eligible", "sum"),
            observed_horizon_count=("horizon_seconds", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "market_family": "market_kind",
                "validity_status": "cell_validity_status",
            }
        )
    )
    market_coverage["observation_kind"] = np.where(
        market_coverage["actual_observation_count"].gt(0),
        "actual_trade",
        "audit_cell_no_actual_trade",
    )
    market_coverage["provenance_status"] = "BOUND_OBSERVED_CACHE"
    candle_rows = pd.DataFrame(
        {
            "game_id": sorted(actual["game_id"].unique()),
            "venue": "kalshi",
            "market_kind": "ALL",
            "episode_count": 0,
            "contract_count": 0,
            "cell_count": 0,
            "actual_observation_count": 0,
            "analysis_eligible_count": 0,
            "observed_horizon_count": 0,
            "cell_validity_status": "NOT_CARRIED_TO_SIGNAL_CACHE",
            "order_ambiguous": False,
            "contaminated": False,
            "observation_kind": "kalshi_1m_candle_bbo",
            "provenance_status": "NOT_CARRIED_TO_SIGNAL_CACHE",
        }
    )
    market_coverage = pd.concat(
        [market_coverage, candle_rows], ignore_index=True
    ).sort_values(
        ["game_id", "venue", "market_kind", "observation_kind"],
        kind="mergesort",
    ).reset_index(drop=True)

    result_lookup = (
        ranking.sort_values("candidate_rank", kind="mergesort")
        .drop_duplicates(["factor_domain", "factor_key"])
        .set_index(["factor_domain", "factor_key"])
    )

    def registered_hypothesis_id(domain: str, key: str) -> str:
        if domain == "EVENT":
            return (
                "event_superclass"
                if key == "episode_superclass"
                else "fair_delta_by_semantic_salience"
            )
        if domain == "STATE":
            return (
                "fair_delta_by_pre_event_liquidity"
                if key in {"staleness_band", "recent_activity_band"}
                else "fair_delta_by_late_close_state"
            )
        if domain == "COMBINATION" and key in {
            "answer_score",
            "back_to_back_turnover",
            "red_zone_turnover",
            "third_fourth_down_failure",
            "two_minute_drive",
            "overtime",
        }:
            return "sequence"
        if domain == "MARKET_REGIME" and key == "venue":
            return "signed_diagnostic_fair_delta_by_venue"
        if domain == "MARKET_REGIME" and key == "family":
            return "fair_delta_by_both_venues_active"
        if domain == "MARKET_REGIME":
            return "fair_delta_by_pre_event_liquidity"
        return "signed_diagnostic_fair_delta_pooled"

    hypotheses: list[dict[str, object]] = []
    diagnostic_fair_hypotheses = {
        "fair_delta_by_both_venues_active",
        "fair_delta_by_late_close_state",
        "fair_delta_by_pre_event_liquidity",
        "fair_delta_by_semantic_salience",
        "signed_diagnostic_fair_delta_by_venue",
        "signed_diagnostic_fair_delta_pooled",
    }
    for row in factor_coverage.itertuples(index=False):
        identity = (str(row.factor_domain), str(row.factor_key))
        hypothesis_id = registered_hypothesis_id(*identity)
        available = not str(row.availability_status).startswith("UNAVAILABLE")
        selected = (
            result_lookup.loc[identity]
            if (
                hypothesis_id not in diagnostic_fair_hypotheses
                and available
                and identity in result_lookup.index
            )
            else None
        )
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "factor_domain": identity[0],
                "factor_key": identity[1],
                "run_status": "RESULT" if selected is not None else "NOT_RUN",
                "status_reason": (
                    "POST_HOC_DESCRIPTIVE_RESULT"
                    if selected is not None
                    else "NO_PIT_PROVEN_DIAGNOSTIC_FAIR_DELTA"
                    if hypothesis_id in diagnostic_fair_hypotheses
                    else str(row.availability_status)
                    if not available
                    else "NOT_RUN_NO_PRESENT_EPISODES"
                ),
                "game_count": int(selected["game_count"]) if selected is not None else 0,
                "point_estimate": (
                    float(selected["point_estimate"])
                    if selected is not None
                    else math.nan
                ),
                "candidate_label": (
                    str(selected["candidate_label"])
                    if selected is not None
                    else "CASE_ONLY"
                ),
            }
        )
    registered = pd.DataFrame(hypotheses).sort_values(
        "hypothesis_id", kind="mergesort"
    ).reset_index(drop=True)
    return {
        "actual_observations": actual_output,
        "actual_observation_paths": paths.rename(
            columns={"market_family": "family"}
        ),
        "game_factor_table": game_factors,
        "factor_coverage": factor_coverage,
        "cross_game_factor_summary": cross_game,
        "candidate_ranking": ranking,
        "per_game_market_coverage": market_coverage,
        "registered_hypothesis_results": registered,
    }


def _apply_analysis_gate(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen focal-outcome discovery gate without row-wise Python."""

    gated = rows.copy(deep=False)
    candidate_age = pd.to_numeric(
        gated["pre_trade_age_seconds"], errors="coerce"
    )
    eligible = (
        gated["signal_candidate_eligible"].eq(True)
        & gated["signal_event_eligible"].eq(True)
        & gated["outcome"].eq(gated["signal_focal_team"])
        & gated["orientation_audited"].eq(True)
        & gated["contract_measure"].eq("winner")
        & gated["market_family"].eq("moneyline")
        & gated["pre_trade_fresh_le_60s"].eq(True)
        & candidate_age.between(0, 60, inclusive="both")
        & gated["pre_trade_age_status"].eq(
            "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
        )
        & gated["validity_status"].eq("OBSERVED")
        & gated["order_ambiguous"].eq(False)
        & gated["contaminated"].eq(False)
        & pd.to_numeric(gated["trade_count"], errors="coerce").gt(0)
        & pd.to_numeric(
            gated["signed_price_change"], errors="coerce"
        ).notna()
    )
    gated["analysis_eligible"] = eligible
    reasons = np.select(
        [
            eligible,
            ~gated["validity_status"].eq("OBSERVED"),
            gated["order_ambiguous"].eq(True),
            gated["contaminated"].eq(True),
            pd.to_numeric(gated["trade_count"], errors="coerce").le(0),
            ~gated["signal_candidate_eligible"].eq(True),
            ~gated["signal_event_eligible"].eq(True),
            ~gated["outcome"].eq(gated["signal_focal_team"]),
            ~gated["orientation_audited"].eq(True),
            ~gated["contract_measure"].eq("winner"),
            ~gated["market_family"].eq("moneyline"),
            ~gated["pre_trade_fresh_le_60s"].eq(True),
            ~candidate_age.between(0, 60, inclusive="both"),
            ~gated["pre_trade_age_status"].eq(
                "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
            ),
        ],
        [
            "ELIGIBLE",
            gated["validity_status"].astype(str),
            "ORDER_AMBIGUOUS",
            "CONTAMINATED",
            "NO_ACTUAL_TRADE",
            "SIGNAL_CANDIDATE_INELIGIBLE",
            "SIGNAL_EVENT_INELIGIBLE",
            "NON_FOCAL_OUTCOME",
            "UNAUDITED_ORIENTATION",
            "NON_WINNER_MEASURE",
            "NON_MONEYLINE_FAMILY",
            "STALE_PRE_TRADE",
            "INVALID_PRE_TRADE_AGE",
            "UNVERIFIED_PRE_TRADE_AGE",
        ],
        default="INCOMPLETE_CANDIDATE_INPUT",
    )
    gated["analysis_exclusion_reason"] = reasons
    return gated


def explore_signal(
    *,
    observed_cache: pd.DataFrame,
    episode_dim: pd.DataFrame,
    contract_orientation_dim: pd.DataFrame,
    config: SignalExplorationConfig,
    event_factor_dim: pd.DataFrame | None = None,
    publication_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Explore event-factor × price changes without causal or execution claims."""

    if not isinstance(config, SignalExplorationConfig):
        raise SignalExplorationError("config must be SignalExplorationConfig")
    joined, episodes, _, selected_games = _normalise_inputs(
        observed_cache,
        episode_dim,
        contract_orientation_dim,
        config,
    )
    joined = _apply_analysis_gate(joined)
    analysis_rows = joined[joined["analysis_eligible"]].copy()
    if analysis_rows.empty:
        raise SignalExplorationError(
            "no rows satisfy the frozen signal candidate analysis gate"
        )
    publish = (
        joined
        if publication_rows is None
        else _apply_analysis_gate(publication_rows)
    )
    event_orientation_status = "AVAILABLE_EXPLICIT_FOCAL_TEAM_OUTCOME"
    coverage = _coverage(analysis_rows, config)
    summary = _reaction_summary(analysis_rows, config)
    symmetry, symmetry_status = _cross_venue_symmetry(analysis_rows)
    frozen = _frozen_tables(analysis_rows, config)
    factors = _factor_tables(
        analysis_rows,
        publication_rows=publish,
        episodes=episodes,
        event_factor_dim=event_factor_dim,
        config=config,
    )
    output_names = {
        "market_family": "family",
        "delay_seconds": "delay_scenario_seconds",
    }
    coverage = coverage.rename(columns=output_names)
    summary = summary.rename(columns=output_names)
    symmetry = symmetry.rename(columns=output_names)
    return {
        "artifact_type": "nfl_x13_signal_exploration",
        "analysis_version": ANALYSIS_VERSION,
        "result_label": RESULT_LABEL,
        "analysis_status": "POST_HOC_DISCOVERY_HOLDOUT_REQUIRED",
        "game_count": len(selected_games),
        "selected_game_ids": selected_games,
        "config": config,
        "coverage": coverage,
        "reaction_summary": summary,
        "cross_venue_symmetry": symmetry,
        **frozen,
        **factors,
        "orientation_audit": {
            "cross_venue_status": symmetry_status,
            "event_orientation_status": event_orientation_status,
            "shared_sports_unit": "episode_id",
            "unaudited_orientation_inferred": False,
        },
        "claim_boundary": {
            "causal": False,
            "latency": False,
            "execution": False,
            "tradeability": False,
            "alpha": False,
            "order_flow_imbalance": False,
            "depth": False,
        },
    }


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_ready(value: object) -> Any:
    if isinstance(value, SignalExplorationConfig):
        return {
            "registered_delays_seconds": list(value.registered_delays_seconds),
            "registered_horizons_seconds": list(value.registered_horizons_seconds),
            "bootstrap_samples": value.bootstrap_samples,
            "confidence_level": value.confidence_level,
            "seed": value.seed,
            "minimum_episode_count": value.minimum_episode_count,
            "minimum_game_count": value.minimum_game_count,
            "bh_q_threshold": value.bh_q_threshold,
        }
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise SignalExplorationError(
        f"unsupported canonical value type: {type(value).__name__}"
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="2.0",
    )
    return sink.getvalue().to_pybytes()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_bounded(
    path: Path,
    *,
    max_length: int,
    expected_length: int | None = None,
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SignalExplorationError(f"bounded input is not a regular file: {path}")
    size = path.stat().st_size
    if size > max_length or (
        expected_length is not None and size != expected_length
    ):
        raise SignalExplorationError(f"bounded input byte length mismatch: {path}")
    with path.open("rb") as source:
        payload = source.read(size + 1)
    if len(payload) != size:
        raise SignalExplorationError(f"bounded input changed while reading: {path}")
    return payload


def _runtime_record() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "network_used": False,
        "database_used": False,
        "ui_used": False,
    }


def _markdown_report(result: Mapping[str, object]) -> bytes:
    actual = result["actual_observations"]
    factors = result["game_factor_table"]
    factor_coverage = result["factor_coverage"]
    ranking = result["candidate_ranking"]
    assert isinstance(actual, pd.DataFrame)
    assert isinstance(factors, pd.DataFrame)
    assert isinstance(factor_coverage, pd.DataFrame)
    assert isinstance(ranking, pd.DataFrame)

    def number(value: object) -> str:
        return "NA" if pd.isna(value) else format(float(value), ".10g")

    unavailable = factor_coverage[
        factor_coverage["availability_status"].eq("UNAVAILABLE_SOURCE_FIELD")
    ]["factor_key"].tolist()
    observed_count = int(actual["actual_observation"].eq(True).sum())
    unavailable_count = len(actual) - observed_count
    lines = [
        "# X-13 NFL 信号探索报告",
        "",
        f"- 结果标签：`{result['result_label']}`",
        f"- 分析状态：`{result['analysis_status']}`",
        f"- 比赛数：{int(result['game_count'])}",
        "- 证据边界：仅 source-time 事后发现；需要独立 holdout 验证。",
        "- 不主张因果、实时延迟、可执行性或可交易性。",
        "- 不主张 OFI、depth 或 execution 证据。",
        "",
        "## 逐场冻结因子",
        "",
        f"- delay=0 审计单元：{len(actual)} 行；其中真实成交观测 "
        f"{observed_count} 行、缺失或无效单元 {unavailable_count} 行。",
        "- 观察窗口固定为 `1/2/5/10/30/60` 秒；"
        "缺失单元显式保留且不 forward-fill。",
        "- observation policy：`ACTUAL_ONLY_NO_FORWARD_FILL`。",
        f"- 逐场 factor table：{len(factors)} 行；"
        f"cross-game candidate ranking：{len(ranking)} 行。",
        "- 无独立结构字段：fumble、turnover_on_downs。",
        "- 结构字段不可用："
        + ("、".join(unavailable) if unavailable else "无")
        + "；未从 description 推断。",
        "",
        "## Top 10 discovery contrasts",
        "",
        "| rank | factor contrast | effect | 95% CI | BH q | Kalshi | Polymarket | venue sign | support |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in ranking.sort_values(
        "candidate_rank", kind="mergesort"
    ).head(10).iterrows():
        lines.append(
            f"| {int(row['candidate_rank'])} | "
            f"{row['factor_domain']} / {row['factor_key']} / "
            f"{row['factor_value']} vs {row['reference_value']} | "
            f"{number(row['point_estimate'])} | "
            f"[{number(row['ci_low'])}, {number(row['ci_high'])}] | "
            f"{number(row['bh_q_value'])} | "
            f"{number(row['kalshi_effect'])} | "
            f"{number(row['polymarket_effect'])} | "
            f"{bool(row['cross_venue_sign_consistent'])} | "
            f"{bool(row['support_gate_pass'])} |"
        )
    top = (
        ranking.sort_values("candidate_rank", kind="mergesort").iloc[0]
        if not ranking.empty
        else None
    )
    if (
        top is not None
        and str(top["factor_key"]) == "distance_band"
        and str(top["factor_value"]) == "LONG_GT_10"
    ):
        lines.append(
            "- 当前最强 discovery-only 对比为 distance > 10；"
            "独立 holdout reaction 尚未运行。"
        )
    else:
        lines.append("- 所有排序均为 discovery-only；独立 holdout reaction 尚未运行。")
    lines.extend(
        [
            "- touchdown、interception、field goal 与 sequence "
            "均未形成可晋升的跨 venue 稳健候选。",
            "- 冻结 holdout selection 不等于 holdout result；"
            "本报告未读取 holdout reaction。",
            "",
        "## 冻结推断表",
        "",
        ]
    )
    for name, title in (
        ("primary_contrasts", "Primary contrasts"),
        ("path_comparisons", "10–60 秒路径"),
        ("state_moderators", "PRE-state moderators"),
        ("venue_comparisons", "配对 venue differences"),
    ):
        frame = result[name]
        assert isinstance(frame, pd.DataFrame)
        labels = (
            frame["candidate_label"].value_counts().sort_index().to_dict()
            if "candidate_label" in frame
            else {}
        )
        rendered = "、".join(
            f"{label}={int(count)}" for label, count in labels.items()
        ) or "无"
        lines.append(f"- {title}：{len(frame)} 行；{rendered}")
    lines.extend(
        [
            "",
            "## Primary contrasts",
            "",
            "| comparison | estimate | 95% CI | BH q | label |",
            "|---|---:|---:|---:|---|",
        ]
    )
    primary = result["primary_contrasts"]
    assert isinstance(primary, pd.DataFrame)

    for _, row in primary.sort_values("comparison_id", kind="mergesort").iterrows():
        lines.append(
            f"| {row['comparison_id']} | {number(row['point_estimate'])} | "
            f"[{number(row['ci_low'])}, {number(row['ci_high'])}] | "
            f"{number(row['bh_q_value'])} | {row['candidate_label']} |"
        )
    lines.extend(
        [
            "",
            "本文件由 runner 确定性生成，并受 bundle manifest 与 checksums.sha256 约束。",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _manifest_material(manifest: Mapping[str, object]) -> dict[str, object]:
    material = dict(manifest)
    material.pop("bundle_sha256", None)
    return material


def _write_signal_exploration_stage(
    result: Mapping[str, object],
    *,
    output_root: str | Path,
    input_manifest_sha256: str,
) -> SignalExplorationBundle:
    """Write canonical content-addressed JSON/Parquet evidence and manifest."""

    if not isinstance(result, Mapping):
        raise SignalExplorationError("result must be a mapping")
    if _SHA256_RE.fullmatch(input_manifest_sha256) is None:
        raise SignalExplorationError(
            "input_manifest_sha256 must be a lowercase SHA-256"
        )
    required_frames = (
        "coverage",
        "reaction_summary",
        "cross_venue_symmetry",
        "primary_contrasts",
        "path_comparisons",
        "state_moderators",
        "venue_comparisons",
        "actual_observations",
        "actual_observation_paths",
        "game_factor_table",
        "factor_coverage",
        "cross_game_factor_summary",
        "candidate_ranking",
        "per_game_market_coverage",
        "registered_hypothesis_results",
    )
    for name in required_frames:
        if not isinstance(result.get(name), pd.DataFrame):
            raise SignalExplorationError(f"result.{name} must be a DataFrame")
    root = Path(output_root)
    objects = root / "objects"
    file_records: list[dict[str, object]] = []
    file_sha256: dict[str, str] = {}

    report = {
        key: value
        for key, value in result.items()
        if key not in required_frames
    }
    payloads: list[tuple[str, str, bytes, int]] = [
        (
            "report",
            "json",
            _canonical_json_bytes(_json_ready(report)) + b"\n",
            1,
        ),
        (
            "runtime_report",
            "json",
            _canonical_json_bytes(_runtime_record()) + b"\n",
            1,
        ),
    ]
    for name in required_frames:
        frame = result[name]
        assert isinstance(frame, pd.DataFrame)
        payloads.append((name, "parquet", _parquet_bytes(frame), len(frame)))
    payloads.append(
        ("user_report_zh_cn", "md", _markdown_report(result), 1)
    )
    for logical_name, suffix, payload, row_count in payloads:
        digest = _sha256_bytes(payload)
        relative = (
            PurePosixPath("X13_SIGNAL_REPORT.zh-CN.md")
            if logical_name == "user_report_zh_cn"
            else PurePosixPath(
                "objects",
                f"{digest.removeprefix('sha256:')}.{suffix}",
            )
        )
        target = root / Path(relative)
        if target.exists():
            if logical_name == "user_report_zh_cn":
                _atomic_write(target, payload)
            elif _read_bounded(
                target,
                max_length=len(payload),
                expected_length=len(payload),
            ) != payload:
                raise SignalExplorationError(
                    f"existing content-addressed object mismatch: {target}"
                )
        else:
            _atomic_write(target, payload)
        file_sha256[logical_name] = digest
        file_records.append(
            {
                "logical_name": logical_name,
                "path": str(relative),
                "media_type": (
                    "application/json"
                    if suffix == "json"
                    else "text/markdown; charset=utf-8"
                    if suffix == "md"
                    else "application/vnd.apache.parquet"
                ),
                "sha256": digest,
                "byte_length": len(payload),
                "row_count": row_count,
            }
        )
    manifest: dict[str, object] = {
        "artifact_type": "nfl_x13_signal_exploration_bundle",
        "analysis_version": ANALYSIS_VERSION,
        "result_label": RESULT_LABEL,
        "analysis_status": "POST_HOC_DISCOVERY_HOLDOUT_REQUIRED",
        "input_manifest_sha256": input_manifest_sha256,
        "files": sorted(file_records, key=lambda item: str(item["logical_name"])),
        "runtime": _runtime_record(),
    }
    holdout_reference = result.get("historical_holdout_selection")
    if holdout_reference is not None:
        if type(holdout_reference) is not dict:
            raise SignalExplorationError(
                "historical holdout selection reference must be an object"
            )
        manifest["historical_holdout_selection"] = holdout_reference
    bundle_sha256 = _sha256_bytes(
        _canonical_json_bytes(_json_ready(_manifest_material(manifest)))
    )
    manifest["bundle_sha256"] = bundle_sha256
    manifest_path = root / (
        f"{bundle_sha256.removeprefix('sha256:')}.manifest.json"
    )
    manifest_payload = _canonical_json_bytes(_json_ready(manifest)) + b"\n"
    if manifest_path.exists():
        if _read_bounded(
            manifest_path,
            max_length=_MAX_MANIFEST_BYTES,
            expected_length=len(manifest_payload),
        ) != manifest_payload:
            raise SignalExplorationError("existing bundle manifest mismatch")
    else:
        _atomic_write(manifest_path, manifest_payload)
    checksum_entries = [
        (
            str(record["path"]),
            str(record["sha256"]).removeprefix("sha256:"),
        )
        for record in file_records
    ]
    checksum_entries.append(
        (manifest_path.name, _sha256_bytes(manifest_payload).removeprefix("sha256:"))
    )
    checksums_payload = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(checksum_entries)
    ).encode("utf-8")
    checksums_path = root / "checksums.sha256"
    if checksums_path.exists():
        if _read_bounded(
            checksums_path,
            max_length=_MAX_CHECKSUMS_BYTES,
            expected_length=len(checksums_payload),
        ) != checksums_payload:
            raise SignalExplorationError("existing checksums file mismatch")
    else:
        _atomic_write(checksums_path, checksums_payload)
    return SignalExplorationBundle(
        manifest_path=manifest_path,
        bundle_sha256=bundle_sha256,
        file_sha256=file_sha256,
    )


def write_signal_exploration_bundle(
    result: Mapping[str, object],
    *,
    output_root: str | Path,
    input_manifest_sha256: str,
) -> SignalExplorationBundle:
    """Stage, verify, and atomically publish one immutable bundle directory."""

    publication_root = Path(output_root)
    publication_root.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".x13-signal-stage.", dir=publication_root)
    )
    try:
        staged = _write_signal_exploration_stage(
            result,
            output_root=stage,
            input_manifest_sha256=input_manifest_sha256,
        )
        staged_manifest = verify_signal_exploration_bundle(staged.manifest_path)
        target = publication_root / staged.bundle_sha256.replace(":", "-")
        target_manifest = target / staged.manifest_path.name
        if target.exists():
            existing_manifest = verify_signal_exploration_bundle(target_manifest)
            if existing_manifest != staged_manifest:
                raise SignalExplorationError(
                    "immutable bundle target conflicts with staged bundle"
                )
            shutil.rmtree(stage)
            return SignalExplorationBundle(
                manifest_path=target_manifest,
                bundle_sha256=staged.bundle_sha256,
                file_sha256=staged.file_sha256,
            )
        os.rename(stage, target)
        directory_fd = os.open(
            publication_root,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published_manifest = verify_signal_exploration_bundle(target_manifest)
        if published_manifest != staged_manifest:
            raise SignalExplorationError(
                "published bundle differs from verified staging bundle"
            )
        return SignalExplorationBundle(
            manifest_path=target_manifest,
            bundle_sha256=staged.bundle_sha256,
            file_sha256=staged.file_sha256,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _strict_manifest_json(path: Path) -> dict[str, Any]:
    try:
        payload = _read_bounded(path, max_length=_MAX_MANIFEST_BYTES)
    except OSError as error:
        raise SignalExplorationError(f"cannot read manifest: {path}") from error

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SignalExplorationError(
                    f"manifest contains duplicate key {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                SignalExplorationError(
                    f"manifest contains non-finite number {constant}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SignalExplorationError("manifest is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise SignalExplorationError("manifest must be a JSON object")
    if payload != _canonical_json_bytes(value) + b"\n":
        raise SignalExplorationError("manifest is not canonical JSON")
    return value


def verify_signal_exploration_bundle(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify manifest identity, path confinement, checksums, and row counts."""

    path = Path(manifest_path)
    manifest = _strict_manifest_json(path)
    observed_bundle = manifest.get("bundle_sha256")
    if type(observed_bundle) is not str or _SHA256_RE.fullmatch(observed_bundle) is None:
        raise SignalExplorationError("bundle_sha256 is invalid")
    expected_bundle = _sha256_bytes(
        _canonical_json_bytes(_json_ready(_manifest_material(manifest)))
    )
    if observed_bundle != expected_bundle:
        raise SignalExplorationError("bundle manifest checksum mismatch")
    if path.name != f"{observed_bundle.removeprefix('sha256:')}.manifest.json":
        raise SignalExplorationError("manifest path is not content addressed")
    holdout_reference = manifest.get("historical_holdout_selection")
    if holdout_reference is not None and (
        type(holdout_reference) is not dict
        or set(holdout_reference)
        != {
            "byte_length",
            "evidence_index_sha256",
            "selected_game_ids",
            "selection_id",
            "selection_sha256",
            "status",
        }
        or holdout_reference.get("status") != "FROZEN"
        or type(holdout_reference.get("selection_id")) is not str
        or _SHA256_RE.fullmatch(holdout_reference["selection_id"]) is None
        or type(holdout_reference.get("selection_sha256")) is not str
        or _SHA256_RE.fullmatch(holdout_reference["selection_sha256"]) is None
        or type(holdout_reference.get("evidence_index_sha256")) is not str
        or _SHA256_RE.fullmatch(
            holdout_reference["evidence_index_sha256"]
        )
        is None
        or type(holdout_reference.get("byte_length")) is not int
        or holdout_reference["byte_length"] <= 0
        or type(holdout_reference.get("selected_game_ids")) is not list
        or len(holdout_reference["selected_game_ids"]) != 20
    ):
        raise SignalExplorationError(
            "historical holdout selection reference is invalid"
        )
    files = manifest.get("files")
    if type(files) is not list or not files:
        raise SignalExplorationError("manifest files must be a non-empty array")
    logical_names: set[str] = set()
    for record in files:
        if type(record) is not dict:
            raise SignalExplorationError("manifest file record must be an object")
        logical_name = record.get("logical_name")
        relative_text = record.get("path")
        expected_sha = record.get("sha256")
        byte_length = record.get("byte_length")
        if (
            type(logical_name) is not str
            or not logical_name
            or logical_name in logical_names
        ):
            raise SignalExplorationError("manifest logical names must be unique")
        logical_names.add(logical_name)
        if type(relative_text) is not str:
            raise SignalExplorationError("manifest file path must be a string")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise SignalExplorationError("manifest file path escapes bundle root")
        if type(expected_sha) is not str or _SHA256_RE.fullmatch(expected_sha) is None:
            raise SignalExplorationError("manifest file SHA-256 is invalid")
        if (
            logical_name != "user_report_zh_cn"
            and relative.stem != expected_sha.removeprefix("sha256:")
        ):
            raise SignalExplorationError("manifest file path is not content addressed")
        object_path = path.parent / Path(relative)
        try:
            if type(byte_length) is not int or byte_length < 0:
                raise SignalExplorationError(
                    f"content byte length is invalid: {relative_text}"
                )
            payload = _read_bounded(
                object_path,
                max_length=_MAX_OBJECT_BYTES,
                expected_length=byte_length,
            )
        except OSError as error:
            raise SignalExplorationError(
                f"cannot read bundle object: {relative_text}"
            ) from error
        if _sha256_bytes(payload) != expected_sha:
            raise SignalExplorationError(
                f"content checksum mismatch: {relative_text}"
            )
        if type(byte_length) is not int or byte_length != len(payload):
            raise SignalExplorationError(
                f"content byte length mismatch: {relative_text}"
            )
        if relative.suffix == ".parquet":
            try:
                row_count = pq.ParquetFile(pa.BufferReader(payload)).metadata.num_rows
            except Exception as error:
                raise SignalExplorationError(
                    f"invalid Parquet object: {relative_text}"
                ) from error
            if record.get("row_count") != row_count:
                raise SignalExplorationError(
                    f"Parquet row count mismatch: {relative_text}"
                )
    if logical_names != {
        "report",
        "runtime_report",
        "coverage",
        "reaction_summary",
        "cross_venue_symmetry",
        "primary_contrasts",
        "path_comparisons",
        "state_moderators",
        "venue_comparisons",
        "actual_observations",
        "actual_observation_paths",
        "game_factor_table",
        "factor_coverage",
        "cross_game_factor_summary",
        "candidate_ranking",
        "per_game_market_coverage",
        "registered_hypothesis_results",
        "user_report_zh_cn",
    }:
        raise SignalExplorationError("bundle logical file set is incomplete")
    checksum_entries = [
        (
            str(record["path"]),
            str(record["sha256"]).removeprefix("sha256:"),
        )
        for record in files
    ]
    checksum_entries.append(
        (
            path.name,
            _sha256_bytes(
                _read_bounded(path, max_length=_MAX_MANIFEST_BYTES)
            ).removeprefix("sha256:"),
        )
    )
    expected_checksums = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(checksum_entries)
    ).encode("utf-8")
    checksums_path = path.parent / "checksums.sha256"
    try:
        observed_checksums = _read_bounded(
            checksums_path,
            max_length=_MAX_CHECKSUMS_BYTES,
            expected_length=len(expected_checksums),
        )
    except OSError as error:
        raise SignalExplorationError("cannot read checksums.sha256") from error
    if observed_checksums != expected_checksums:
        raise SignalExplorationError("checksums.sha256 mismatch")
    return manifest


def _verify_historical_holdout_selection(
    selection_path: str | Path,
) -> dict[str, object]:
    path = Path(selection_path)
    payload = _read_bounded(path, max_length=_MAX_MANIFEST_BYTES)
    try:
        selection = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SignalExplorationError(
            "historical holdout selection is invalid JSON"
        ) from error
    if (
        type(selection) is not dict
        or payload != _canonical_json_bytes(selection) + b"\n"
        or selection.get("artifact_type")
        != "nfl_x13_historical_holdout_selection"
        or selection.get("status") != "FROZEN"
        or selection.get("seed") != 20260725
    ):
        raise SignalExplorationError(
            "historical holdout selection is not a frozen canonical artifact"
        )
    selection_id = selection.get("selection_id")
    evidence_index_sha256 = selection.get("evidence_index_sha256")
    selected = selection.get("selected_game_ids")
    manifests = selection.get("evidence_manifest_sha256s")
    if (
        type(selection_id) is not str
        or _SHA256_RE.fullmatch(selection_id) is None
        or path.parent.name != selection_id.replace(":", "-")
        or type(evidence_index_sha256) is not str
        or _SHA256_RE.fullmatch(evidence_index_sha256) is None
        or type(selected) is not list
        or len(selected) != 20
        or len(set(selected)) != 20
        or any(
            type(game_id) is not str
            or game_id in X13_GAME_IDS
            or re.fullmatch(r"2025_[0-9]{2}_[A-Z]+_[A-Z]+", game_id) is None
            for game_id in selected
        )
        or type(manifests) is not list
        or not manifests
        or any(
            type(digest) is not str or _SHA256_RE.fullmatch(digest) is None
            for digest in manifests
        )
    ):
        raise SignalExplorationError(
            "historical holdout selection identity is invalid"
        )
    expected_selection_id = _sha256_bytes(
        _canonical_json_bytes(
            {
                "status": "FROZEN",
                "seed": 20260725,
                "selected_game_ids": selected,
                "evidence_index_sha256": evidence_index_sha256,
                "evidence_manifest_sha256s": manifests,
            }
        )
    )
    if selection_id != expected_selection_id:
        raise SignalExplorationError(
            "historical holdout selection semantic hash differs"
        )
    checksum_path = path.parent / "checksums.sha256"
    checksum_payload = (
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    ).encode("ascii")
    if _read_bounded(
        checksum_path,
        max_length=_MAX_CHECKSUMS_BYTES,
        expected_length=len(checksum_payload),
    ) != checksum_payload:
        raise SignalExplorationError(
            "historical holdout selection checksum differs"
        )
    return {
        "status": "FROZEN",
        "selection_id": selection_id,
        "selection_sha256": _sha256_bytes(payload),
        "byte_length": len(payload),
        "evidence_index_sha256": evidence_index_sha256,
        "selected_game_ids": selected,
    }


def _parse_cli(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic X-13 source-time signal exploration"
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--holdout-selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def run_signal_exploration(
    *,
    cache_path: str | Path,
    holdout_selection_path: str | Path,
    output_root: str | Path,
) -> SignalExplorationBundle:
    """Verify the X-13 cache, explore it offline, and write an evidence bundle."""

    from prediction_market.sports.nfl_x13_correlation_cache import (
        verify_x13_correlation_cache,
    )

    cache_root = Path(cache_path)
    manifest_path = cache_root / "manifest.json"
    manifest_snapshot = _read_bounded(
        manifest_path,
        max_length=_MAX_MANIFEST_BYTES,
    )
    manifest = verify_x13_correlation_cache(cache_root)
    try:
        snapshot_value = json.loads(manifest_snapshot)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SignalExplorationError("verified cache manifest snapshot is invalid") from error
    if snapshot_value != manifest:
        raise SignalExplorationError(
            "cache manifest changed across strict verification"
        )
    if manifest.get("batch_id") != (
        "sha256:3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
    ):
        raise SignalExplorationError("cache is not the frozen X-13 batch")
    observed_record = manifest.get("artifacts", {}).get("observed_cache")
    if not isinstance(observed_record, Mapping):
        raise SignalExplorationError("cache manifest lacks artifacts.observed_cache")
    relative = observed_record.get("relative_path")
    if type(relative) is not str:
        raise SignalExplorationError("observed cache relative_path is invalid")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SignalExplorationError("observed cache path escapes cache root")
    observed_path = cache_root / Path(relative_path)
    try:
        observed_length = observed_record.get("byte_length")
        if type(observed_length) is not int or observed_length < 0:
            raise SignalExplorationError("observed cache byte_length is invalid")
        observed_bytes = _read_bounded(
            observed_path,
            max_length=_MAX_OBJECT_BYTES,
            expected_length=observed_length,
        )
    except OSError as error:
        raise SignalExplorationError("cannot read verified observed cache") from error
    expected_observed_sha = observed_record.get("sha256")
    if (
        type(expected_observed_sha) is not str
        or _SHA256_RE.fullmatch(expected_observed_sha) is None
    ):
        raise SignalExplorationError("observed cache SHA-256 is invalid")
    if _sha256_bytes(observed_bytes) != expected_observed_sha:
        raise SignalExplorationError("observed cache checksum changed after verification")
    try:
        publication_observed = pq.read_table(
            BytesIO(observed_bytes),
            filters=[("delay_scenario_seconds", "=", 0)],
        ).to_pandas()
        analysis_observed = pq.read_table(
            BytesIO(observed_bytes),
            filters=[
                ("validity_status", "=", "OBSERVED"),
                ("order_ambiguous", "=", False),
                ("contaminated", "=", False),
            ],
        ).to_pandas()
    except Exception as error:
        raise SignalExplorationError("observed cache is not valid Parquet") from error

    required_joined = (
        "game_id",
        "episode_id",
        "episode_type",
        "episode_superclass",
        "contract_id",
        "venue",
        "family",
        "semantic_orientation_id",
        "cross_venue_eligible",
        "delay_scenario_seconds",
        "horizon_seconds",
        "signed_price_change",
        "maximum_excursion",
        "net_change_60s",
        "trade_count",
        "volume",
        "staleness_seconds",
        "overshoot_candidate",
        "reversal_candidate",
        "signal_event_eligible",
        "signal_candidate_eligible",
        "pre_trade_fresh_le_60s",
        "pre_trade_age_seconds",
        "pre_trade_age_status",
        "signal_focal_team",
        "scoring_team",
        "pre_offense_team",
        "pre_defense_team",
        "pre_possession_team",
        "contract_measure",
        "source_time_start_utc",
        "source_time_end_utc",
        "pre_event_actual_trade",
        "first_post_event_trade",
        "vwap",
        "pre_game_seconds_remaining",
        "pre_quarter",
        "pre_score_margin_home",
        "pre_down",
        "pre_distance",
        "pre_yardline_100",
        "pre_goal_to_go",
        "validity_status",
        "order_ambiguous",
        "contaminated",
    )
    _require_columns(
        publication_observed,
        required_joined,
        frame_name="publication_observed_cache",
    )
    _require_columns(
        analysis_observed,
        required_joined,
        frame_name="analysis_observed_cache",
    )
    if publication_observed["validity_status"].isna().any() or not publication_observed[
        "validity_status"
    ].map(lambda value: type(value) is str and bool(value)).all():
        raise SignalExplorationError(
            "observed cache validity_status must be explicit"
        )
    for field in ("order_ambiguous", "contaminated"):
        if publication_observed[field].isna().any() or not publication_observed[field].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise SignalExplorationError(
                f"observed cache {field} must be explicit booleans"
            )
    observed_games = tuple(
        sorted(publication_observed["game_id"].dropna().unique())
    )
    if observed_games != X13_GAME_IDS:
        raise SignalExplorationError(
            "observed cache must contain the exact frozen 20-game set"
        )
    candidate = publication_observed["signal_candidate_eligible"].eq(True)
    candidate_age = pd.to_numeric(
        publication_observed.loc[candidate, "pre_trade_age_seconds"],
        errors="coerce",
    )
    if (
        candidate_age.isna().any()
        or (candidate_age < 0).any()
        or (candidate_age > 60).any()
        or not publication_observed.loc[candidate, "pre_trade_age_status"].eq(
            "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
        ).all()
        or not publication_observed.loc[
            candidate, "pre_trade_fresh_le_60s"
        ].eq(True).all()
    ):
        raise SignalExplorationError(
            "candidate rows require reconstructed numeric pre-trade age <=60s"
        )
    episode_dim = publication_observed[
        [
            "game_id",
            "episode_id",
            "episode_superclass",
            "episode_type",
            "source_time_start_utc",
            "scoring_team",
        ]
    ].drop_duplicates()
    if episode_dim.duplicated(["game_id", "episode_id"]).any():
        raise SignalExplorationError(
            "episode sequence fields changed within an observed episode"
        )
    orientation_dim = publication_observed[
        [
            "contract_id",
            "outcome",
            "semantic_orientation_id",
            "cross_venue_eligible",
        ]
    ].drop_duplicates()
    def core_rows(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(
            columns=[
                "episode_superclass",
                "semantic_orientation_id",
                "cross_venue_eligible",
            ]
        ).rename(
            columns={
                "family": "market_family",
                "delay_scenario_seconds": "delay_seconds",
                "overshoot_candidate": "overshoot",
                "reversal_candidate": "reversal",
            }
        )

    core_observed = core_rows(analysis_observed)
    publication_rows = core_rows(publication_observed)
    for numeric_field in (
        "signed_price_change",
        "maximum_excursion",
        "net_change_60s",
        "trade_count",
        "volume",
        "staleness_seconds",
        "pre_event_actual_trade",
        "first_post_event_trade",
        "vwap",
        "pre_trade_age_seconds",
    ):
        publication_rows[numeric_field] = pd.to_numeric(
            publication_rows[numeric_field],
            errors="coerce",
        )
    publication_rows["semantic_orientation_id"] = publication_observed[
        "semantic_orientation_id"
    ].to_numpy()
    publication_rows["orientation_audited"] = publication_observed[
        "cross_venue_eligible"
    ].to_numpy()
    orientation_dim = orientation_dim.rename(
        columns={"cross_venue_eligible": "orientation_audited"}
    )
    delays = tuple(
        sorted(int(value) for value in core_observed["delay_seconds"].unique())
    )
    horizons = tuple(
        sorted(int(value) for value in core_observed["horizon_seconds"].unique())
    )
    config = SignalExplorationConfig(
        registered_delays_seconds=delays,
        registered_horizons_seconds=horizons,
        bootstrap_samples=10_000,
        minimum_episode_count=20,
        minimum_game_count=6,
    )
    event_factor_dim = (
        _load_bound_event_factor_dim(
            cache_root=cache_root,
            batch_id=str(manifest["batch_id"]),
        )
        if cache_root.parent.name == "correlation-cache"
        else None
    )
    result = explore_signal(
        observed_cache=core_observed,
        episode_dim=episode_dim,
        contract_orientation_dim=orientation_dim,
        config=config,
        event_factor_dim=event_factor_dim,
        publication_rows=publication_rows,
    )
    result["historical_holdout_selection"] = (
        _verify_historical_holdout_selection(holdout_selection_path)
    )
    input_sha = _sha256_bytes(manifest_snapshot)
    return write_signal_exploration_bundle(
        result,
        output_root=output_root,
        input_manifest_sha256=input_sha,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    try:
        bundle = run_signal_exploration(
            cache_path=args.cache,
            holdout_selection_path=args.holdout_selection,
            output_root=args.output_root,
        )
    except (SignalExplorationError, OSError, ValueError) as error:
        print(f"x13 signal exploration failed: {error}", file=sys.stderr)
        return 2
    print(bundle.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_VERSION",
    "RESULT_LABEL",
    "SignalExplorationBundle",
    "SignalExplorationConfig",
    "SignalExplorationError",
    "explore_signal",
    "main",
    "run_signal_exploration",
    "verify_signal_exploration_bundle",
    "write_signal_exploration_bundle",
]
