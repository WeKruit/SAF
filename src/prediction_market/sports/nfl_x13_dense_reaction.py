"""Dense source-time reaction paths and matching support for NFL X-13.

This is a pure V2 analysis domain.  It consumes only already-sealed episode,
timeline, and actual-market-observation tables; it neither fetches nor replays
an upstream source.  A dense bin is observed only when an actual trade occurs
inside that bin.  In particular, it never forward-fills a trade into a later
empty bin.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
BUILDER_VERSION = "nfl-x13-dense-reaction-v4"
DENSE_HORIZONS_SECONDS = (
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
)
DENSE_EXACT_MATCH_KEYS = (
    "venue",
    "market_family",
    "outcome_orientation",
    "horizon_seconds",
)
_NUMERIC_COVARIATES = (
    "pre_probability",
    "game_time_seconds",
    "signed_margin",
    "down",
    "distance",
    "field_position",
    "staleness_seconds",
    "liquidity",
)
_CATEGORICAL_COVARIATES = ("possession", "actor_beneficiary")
_PATH_COLUMNS = (
    "observation_id",
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
    "market_family",
    "outcome_orientation",
    "horizon_seconds",
    "pre_probability",
    "bin_last_trade_price",
    "cumulative_change",
    "incremental_change",
    "actual_trade_count",
    "source_time_first_trade",
    "last_trade_source_time",
    "staleness_seconds",
    "liquidity",
    "order_ambiguous",
    "contaminated",
    "censored",
    "censoring_status",
    "forward_filled",
    "observed",
    "attrition_reason",
    "jump",
    "jump_share",
    "shape_classification",
    "game_time_seconds",
    "signed_margin",
    "down",
    "distance",
    "field_position",
    "possession",
    "actor_beneficiary",
    "control_eligible",
    "claim_boundary",
)


class DenseReactionError(ValueError):
    """A dense source-time input violates a closed analysis invariant."""


@dataclass(frozen=True, slots=True)
class DenseReactionSpecV2:
    """Frozen V2 dense horizons and conservative shape thresholds."""

    horizons_seconds: tuple[int, ...] = DENSE_HORIZONS_SECONDS
    jump_dominant_share: float = 0.75
    gradual_max_increment_share: float = 0.50
    minimum_observed_bins_for_shape: int = 2
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.horizons_seconds != DENSE_HORIZONS_SECONDS:
            raise DenseReactionError("dense horizons must remain frozen")
        if not (0.5 <= self.jump_dominant_share <= 1.0):
            raise DenseReactionError("jump_dominant_share must be in [0.5, 1]")
        if not (0 < self.gradual_max_increment_share < 1):
            raise DenseReactionError(
                "gradual_max_increment_share must be in (0, 1)"
            )
        if self.minimum_observed_bins_for_shape < 2:
            raise DenseReactionError(
                "minimum_observed_bins_for_shape must be at least two"
            )
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise DenseReactionError(
                "historical dense analysis must remain SOURCE_TIME_ONLY"
            )


@dataclass(frozen=True, slots=True)
class DenseMatchedReactionSpecV2:
    """Frozen V2 matching limits for an observation-level effect table."""

    minimum_controls: int = 10
    maximum_controls: int = 20
    maximum_absolute_smd: float = 0.10
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.minimum_controls != 10 or self.maximum_controls != 20:
            raise DenseReactionError("V2 matching must retain min 10 and max 20")
        if not (0 < self.maximum_absolute_smd <= 0.10):
            raise DenseReactionError("V2 maximum_absolute_smd must be in (0, .10]")
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise DenseReactionError(
                "historical dense matching must remain SOURCE_TIME_ONLY"
            )


@dataclass(frozen=True, slots=True)
class DenseReactionTablesV2:
    """Actual-trade paths, shape summaries, and row-level attrition."""

    paths: pd.DataFrame
    path_summaries: pd.DataFrame
    attrition: pd.DataFrame


@dataclass(frozen=True, slots=True)
class DenseMatchedEffectsV2:
    """Selected controls, full audit, balance, and matched effects."""

    matched_controls: pd.DataFrame
    match_audit: pd.DataFrame
    match_balance: pd.DataFrame
    matched_effects: pd.DataFrame


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _require_frame(value: object, *, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise DenseReactionError(f"{label} must be a DataFrame")
    return value.copy()


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise DenseReactionError(
            f"{label} missing required columns: " + ", ".join(missing)
        )


def _required_identifier(
    frame: pd.DataFrame,
    field: str,
    *,
    label: str,
) -> pd.Series:
    values = frame[field]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise DenseReactionError(f"{label} has blank {field}")
    return values.astype(str)


def _finite_numeric(
    frame: pd.DataFrame,
    field: str,
    *,
    label: str,
    nonnegative: bool = False,
) -> pd.Series:
    values = pd.to_numeric(frame[field], errors="coerce").astype(float)
    invalid = ~np.isfinite(values)
    if nonnegative:
        invalid |= values < 0
    if invalid.any():
        description = "finite nonnegative" if nonnegative else "finite"
        raise DenseReactionError(f"{label} {field} must be {description}")
    return values


def _boolean(
    frame: pd.DataFrame,
    field: str,
    *,
    label: str,
    default: bool = False,
) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[field]
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DenseReactionError(f"{label} {field} must contain booleans")
    return values.astype(bool)


def _timestamps(
    frame: pd.DataFrame,
    field: str,
    *,
    label: str,
) -> pd.Series:
    try:
        values = pd.to_datetime(frame[field], utc=True, errors="raise", format="mixed")
    except (TypeError, ValueError) as error:
        raise DenseReactionError(f"{label} {field} must be UTC timestamps") from error
    if values.isna().any():
        raise DenseReactionError(f"{label} {field} must not be null")
    return values


def _normalize_timeline(
    episodes: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(episodes, {"game_id", "episode_id"}, label="episodes")
    episode_values = episodes.copy()
    for field in ("game_id", "episode_id"):
        episode_values[field] = _required_identifier(
            episode_values, field, label="episodes"
        )
    if episode_values.duplicated(["game_id", "episode_id"]).any():
        raise DenseReactionError("episodes contain a duplicate game/episode grain")

    timeline_values = _require_frame(timeline, label="timeline")
    direct_columns = {
        "game_id",
        "episode_id",
        "source_time_start_utc",
        "source_time_end_utc",
    }
    if direct_columns.issubset(timeline_values.columns):
        source = timeline_values.copy()
    else:
        _require_columns(
            timeline_values,
            {
                "game_id",
                "layer",
                "identity",
                "interval_start_utc",
                "interval_end_utc",
            },
            label="timeline",
        )
        source = timeline_values[timeline_values["layer"].eq("G")].copy()
        if "delay_seconds" in source.columns:
            source = source[source["delay_seconds"].eq(0)].copy()
        source = source.rename(
            columns={
                "identity": "episode_id",
                "interval_start_utc": "source_time_start_utc",
                "interval_end_utc": "source_time_end_utc",
            }
        )
    _require_columns(source, direct_columns, label="episode timeline")
    for field in ("game_id", "episode_id"):
        source[field] = _required_identifier(
            source, field, label="episode timeline"
        )
    source["source_time_start_utc"] = _timestamps(
        source, "source_time_start_utc", label="episode timeline"
    )
    source["source_time_end_utc"] = _timestamps(
        source, "source_time_end_utc", label="episode timeline"
    )
    if source["source_time_end_utc"].lt(source["source_time_start_utc"]).any():
        raise DenseReactionError("episode timeline end precedes start")
    if source.duplicated(["game_id", "episode_id"]).any():
        raise DenseReactionError("episode timeline has a duplicate game/episode grain")

    source["timeline_order_ambiguous"] = _boolean(
        source, "order_ambiguous", label="episode timeline"
    )
    source["timeline_contaminated"] = _boolean(
        source, "contaminated", label="episode timeline"
    )
    source_columns = [
        "game_id",
        "episode_id",
        "source_time_start_utc",
        "source_time_end_utc",
        "timeline_order_ambiguous",
        "timeline_contaminated",
    ]
    if "source_coverage_end_utc" in source.columns:
        source["timeline_source_coverage_end_utc"] = source[
            "source_coverage_end_utc"
        ]
        source_columns.append("timeline_source_coverage_end_utc")
    if "next_salient_source_time_utc" in source.columns:
        source["timeline_next_salient_source_time_utc"] = source[
            "next_salient_source_time_utc"
        ]
        source_columns.append("timeline_next_salient_source_time_utc")
    source = source[source_columns].copy()
    result = episode_values.merge(
        source,
        on=["game_id", "episode_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_episode"),
    )
    if len(result) != len(episode_values):
        raise DenseReactionError("episodes lack a source-time timeline row")
    result["order_ambiguous"] = (
        _boolean(result, "order_ambiguous", label="episodes")
        | _boolean(result, "timeline_order_ambiguous", label="episode timeline")
    )
    result["contaminated"] = (
        _boolean(result, "contaminated", label="episodes")
        | _boolean(result, "timeline_contaminated", label="episode timeline")
    )
    coverage_columns = [
        field
        for field in (
            "source_coverage_end_utc",
            "timeline_source_coverage_end_utc",
        )
        if field in result.columns
    ]
    if coverage_columns:
        coverage = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
        for field in coverage_columns:
            present = result[field].notna()
            parsed = pd.Series(pd.NaT, index=result.index, dtype=coverage.dtype)
            if present.any():
                try:
                    parsed.loc[present] = pd.to_datetime(
                        result.loc[present, field], utc=True, errors="raise", format="mixed"
                    )
                except (TypeError, ValueError) as error:
                    raise DenseReactionError(
                        "episode timeline source_coverage_end_utc must be "
                        "UTC timestamps"
                    ) from error
            conflict = coverage.notna() & parsed.notna() & coverage.ne(parsed)
            if conflict.any():
                raise DenseReactionError("episode source coverage end is inconsistent")
            coverage = coverage.fillna(parsed)
        result["source_coverage_end_utc"] = coverage
    if "timeline_next_salient_source_time_utc" in result.columns:
        next_salient = result["timeline_next_salient_source_time_utc"]
        parsed_next_salient = pd.Series(
            pd.NaT,
            index=result.index,
            dtype="datetime64[ns, UTC]",
        )
        present = next_salient.notna()
        if present.any():
            try:
                parsed_next_salient.loc[present] = pd.to_datetime(
                    next_salient.loc[present], utc=True, errors="raise", format="mixed"
                )
            except (TypeError, ValueError) as error:
                raise DenseReactionError(
                    "next_salient_source_time_utc must be UTC timestamps"
                ) from error
        if (
            parsed_next_salient.notna()
            & parsed_next_salient.le(result["source_time_end_utc"])
        ).any():
            raise DenseReactionError(
                "next salient episode must begin after the current episode"
            )
        result["next_salient_source_time_utc"] = parsed_next_salient
    return result


def _normalize_actual_market_observations(
    actual_market_observations: pd.DataFrame,
    *,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    actual = _require_frame(actual_market_observations, label="actual market")
    aliases = {
        "market_family": "family",
        "source_start_utc": "source_start",
        "source_end_utc": "source_end",
    }
    for canonical, source in aliases.items():
        if canonical not in actual.columns and source in actual.columns:
            actual[canonical] = actual[source]
    if "logical_market_id" not in actual.columns:
        timeline_values = _require_frame(timeline, label="timeline")
        _require_columns(
            timeline_values,
            {"game_id", "layer", "identity", "logical_market_id"},
            label="timeline M layer",
        )
        market_identity = timeline_values[
            timeline_values["layer"].eq("M")
            & timeline_values["logical_market_id"].notna()
        ][["game_id", "identity", "logical_market_id"]].copy()
        market_identity["game_id"] = _required_identifier(
            market_identity, "game_id", label="timeline M layer"
        )
        market_identity["observation_id"] = _required_identifier(
            market_identity, "identity", label="timeline M layer"
        )
        market_identity["logical_market_id"] = _required_identifier(
            market_identity, "logical_market_id", label="timeline M layer"
        )
        market_identity = market_identity.drop(columns="identity")
        distinct_identity = market_identity.drop_duplicates()
        if distinct_identity.duplicated(["game_id", "observation_id"]).any():
            raise DenseReactionError(
                "timeline M layer maps an observation to multiple logical markets"
            )
        actual = actual.merge(
            distinct_identity,
            on=["game_id", "observation_id"],
            how="left",
            validate="many_to_one",
        )
        if actual["logical_market_id"].isna().any():
            raise DenseReactionError(
                "actual market observations lack a timeline logical market identity"
            )
    _require_columns(
        actual,
        {
            "game_id",
            "observation_id",
            "logical_market_id",
            "outcome",
            "venue",
            "market_family",
            "kind",
            "price",
            "size",
            "source_start_utc",
            "primary_path_eligible",
        },
        label="actual market",
    )
    for field in (
        "game_id",
        "observation_id",
        "logical_market_id",
        "outcome",
        "venue",
        "market_family",
        "kind",
    ):
        actual[field] = _required_identifier(
            actual, field, label="actual market"
        )
    if "outcome_orientation" not in actual.columns:
        actual["outcome_orientation"] = actual["outcome"]
    outcome_orientation = _required_identifier(
        actual, "outcome_orientation", label="actual market"
    )
    actual["outcome_orientation"] = outcome_orientation
    actual["price"] = _finite_numeric(actual, "price", label="actual market")
    if not actual["price"].between(0, 1).all():
        raise DenseReactionError("actual market price must be in [0, 1]")
    actual["size"] = _finite_numeric(
        actual, "size", label="actual market", nonnegative=True
    )
    actual["source_start_utc"] = _timestamps(
        actual, "source_start_utc", label="actual market"
    )
    if "source_end_utc" not in actual.columns:
        actual["source_end_utc"] = actual["source_start_utc"]
    actual["source_end_utc"] = _timestamps(
        actual, "source_end_utc", label="actual market"
    )
    if actual["source_end_utc"].lt(actual["source_start_utc"]).any():
        raise DenseReactionError("actual market end precedes start")
    actual["primary_path_eligible"] = _boolean(
        actual, "primary_path_eligible", label="actual market"
    )
    actual["contaminated"] = _boolean(
        actual, "contaminated", label="actual market"
    )
    actual["order_ambiguous"] = _boolean(
        actual, "order_ambiguous", label="actual market"
    )
    if "forward_filled" in actual.columns and _boolean(
        actual, "forward_filled", label="actual market"
    ).any():
        raise DenseReactionError(
            "actual market observations must not be forward filled"
        )
    if actual.duplicated(["game_id", "observation_id"]).any():
        raise DenseReactionError("actual market has a duplicate game/observation")
    observed = (
        pd.Series(True, index=actual.index)
        if "provenance" not in actual.columns
        else actual["provenance"].eq("observed")
    )
    return actual[
        actual["kind"].eq("trade")
        & actual["primary_path_eligible"]
        & observed
    ].copy()


def _episode_covariates(episode: pd.Series) -> dict[str, object]:
    aliases = {
        "game_time_seconds": ("game_time_seconds", "game_seconds_remaining"),
        "signed_margin": ("signed_margin", "score_margin", "score_margin_focal"),
        "field_position": ("field_position", "yardline_100"),
        "possession": ("possession", "possession_team"),
    }
    values: dict[str, object] = {}
    for field in ("game_time_seconds", "signed_margin", "field_position", "possession"):
        selected = next(
            (candidate for candidate in aliases[field] if candidate in episode.index),
            None,
        )
        values[field] = episode[selected] if selected is not None else None
    for field in ("down", "distance", "actor_beneficiary"):
        values[field] = episode[field] if field in episode.index else None
    return values


def _coverage_end(
    trades: pd.DataFrame,
    episode: pd.Series,
) -> pd.Timestamp | None:
    values: list[pd.Timestamp] = []
    if "source_coverage_end_utc" in episode.index and pd.notna(
        episode["source_coverage_end_utc"]
    ):
        values.append(pd.Timestamp(episode["source_coverage_end_utc"]))
    if "source_coverage_end_utc" in trades.columns:
        raw = trades["source_coverage_end_utc"].dropna()
        if not raw.empty:
            try:
                values.extend(pd.to_datetime(raw, utc=True, errors="raise", format="mixed"))
            except (TypeError, ValueError) as error:
                raise DenseReactionError(
                    "actual market source_coverage_end_utc must be UTC timestamps"
                ) from error
    unique = {value for value in values}
    if len(unique) > 1:
        raise DenseReactionError("market source coverage end is inconsistent")
    return next(iter(unique), None)


def _same_timestamp(values: pd.Series) -> bool:
    return bool(values.duplicated(keep=False).any())


def _shape(
    rows: list[dict[str, object]],
    *,
    spec: DenseReactionSpecV2,
    globally_invalid: bool,
) -> tuple[float | None, float | None, str]:
    increments = [
        float(row["incremental_change"])
        for row in rows
        if row["observed"] and pd.notna(row["incremental_change"])
    ]
    if globally_invalid or len(increments) < spec.minimum_observed_bins_for_shape:
        return None, None, "UNIDENTIFIED_SPARSE"
    magnitudes = np.abs(np.asarray(increments, dtype=float))
    total = float(magnitudes.sum())
    if total <= 0:
        return None, None, "UNIDENTIFIED_SPARSE"
    jump = increments[0]
    jump_share = abs(jump) / total
    if jump_share >= spec.jump_dominant_share:
        classification = "JUMP_DOMINANT"
    elif float(magnitudes.max()) / total <= spec.gradual_max_increment_share:
        classification = "GRADUAL"
    else:
        classification = "MIXED"
    return jump, jump_share, classification


def build_dense_reaction_tables_v2(
    episodes: pd.DataFrame,
    actual_market_observations: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    spec: DenseReactionSpecV2 | None = None,
) -> DenseReactionTablesV2:
    """Build actual-trade-only dense source-time paths for every available series.

    The timeline supplies episode source-time boundaries.  No missing bin is
    filled from an earlier trade: its cumulative and incremental values remain
    missing and its attrition row records the absence explicitly.
    """

    active_spec = spec or DenseReactionSpecV2()
    episode_rows = _normalize_timeline(episodes, timeline)
    actual = _normalize_actual_market_observations(
        actual_market_observations, timeline=timeline
    )
    path_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for episode in episode_rows.sort_values(
        ["game_id", "episode_id"], kind="mergesort"
    ).to_dict("records"):
        episode_series = pd.Series(episode)
        game_trades = actual[actual["game_id"].eq(episode["game_id"])]
        for market_key, trades in game_trades.groupby(
            [
                "logical_market_id",
                "outcome",
                "venue",
                "market_family",
                "outcome_orientation",
            ],
            sort=True,
            dropna=False,
        ):
            trades = trades.sort_values(
                ["source_start_utc", "observation_id"], kind="mergesort"
            ).reset_index(drop=True)
            start = pd.Timestamp(episode["source_time_start_utc"])
            end = pd.Timestamp(episode["source_time_end_utc"])
            pre = trades[
                trades["source_end_utc"].le(start)
            ].copy()
            pre_ambiguous = False
            pre_row: pd.Series | None = None
            if not pre.empty:
                last_pre_start = pre["source_start_utc"].max()
                candidates = pre[pre["source_start_utc"].eq(last_pre_start)]
                if len(candidates) == 1:
                    pre_row = candidates.iloc[0]
                else:
                    pre_ambiguous = True
            post = trades[
                trades["source_start_utc"].ge(end)
                & trades["source_start_utc"].le(
                    end + pd.Timedelta(seconds=max(active_spec.horizons_seconds))
                )
            ].copy()
            post_ambiguous = _same_timestamp(post["source_start_utc"])
            trade_order_ambiguous = bool(post["order_ambiguous"].any())
            trade_contaminated = bool(post["contaminated"].any())
            order_ambiguous = bool(
                episode["order_ambiguous"]
                or pre_ambiguous
                or post_ambiguous
                or trade_order_ambiguous
            )
            contaminated = bool(episode["contaminated"] or trade_contaminated)
            coverage_end = _coverage_end(trades, episode_series)
            first_trade = (
                None
                if order_ambiguous or post.empty
                else pd.Timestamp(post["source_start_utc"].iloc[0])
            )
            pre_price = None if pre_row is None else float(pre_row["price"])
            previous_price = pre_price
            bin_rows: list[dict[str, object]] = []
            prior_horizon = 0
            logical_market_id, outcome, venue, market_family, orientation = market_key
            covariates = _episode_covariates(episode_series)
            next_salient = episode.get("next_salient_source_time_utc")
            next_salient_time = (
                None
                if next_salient is None or pd.isna(next_salient)
                else pd.Timestamp(next_salient)
            )
            for horizon in active_spec.horizons_seconds:
                bin_start = end + pd.Timedelta(seconds=prior_horizon)
                deadline = end + pd.Timedelta(seconds=horizon)
                bin_trades = post[
                    (
                        post["source_start_utc"].ge(bin_start)
                        if prior_horizon == 0
                        else post["source_start_utc"].gt(bin_start)
                    )
                    & post["source_start_utc"].le(deadline)
                ]
                censored = coverage_end is not None and coverage_end < deadline
                next_salient_contamination = bool(
                    next_salient_time is not None
                    and next_salient_time <= deadline
                )
                horizon_contaminated = bool(
                    contaminated or next_salient_contamination
                )
                if horizon_contaminated:
                    reason = "CONTAMINATED"
                elif order_ambiguous:
                    reason = "ORDER_AMBIGUOUS"
                elif censored:
                    reason = "CENSORED"
                elif pre_price is None:
                    reason = "NO_UNIQUE_PRE_EVENT_ACTUAL_TRADE"
                elif bin_trades.empty:
                    reason = "NO_ACTUAL_TRADE_IN_BIN"
                else:
                    reason = "OBSERVED"
                observed = reason == "OBSERVED"
                last_trade = bin_trades.iloc[-1] if observed else None
                bin_price = None if last_trade is None else float(last_trade["price"])
                incremental = (
                    None
                    if bin_price is None or previous_price is None
                    else bin_price - previous_price
                )
                cumulative = (
                    None
                    if bin_price is None or pre_price is None
                    else bin_price - pre_price
                )
                if observed and bin_price is not None:
                    previous_price = bin_price
                row = {
                    "observation_id": (
                        "dense-v2:"
                        + str(episode["game_id"])
                        + ":"
                        + str(episode["episode_id"])
                        + ":"
                        + str(logical_market_id)
                        + ":"
                        + str(outcome)
                        + ":"
                        + str(venue)
                        + ":"
                        + str(horizon)
                    ),
                    "game_id": episode["game_id"],
                    "episode_id": episode["episode_id"],
                    "logical_market_id": logical_market_id,
                    "outcome": outcome,
                    "venue": venue,
                    "market_family": market_family,
                    "outcome_orientation": orientation,
                    "horizon_seconds": horizon,
                    "pre_probability": pre_price,
                    "bin_last_trade_price": bin_price,
                    "cumulative_change": cumulative,
                    "incremental_change": incremental,
                    "actual_trade_count": int(len(bin_trades)),
                    "source_time_first_trade": first_trade,
                    "last_trade_source_time": (
                        None
                        if last_trade is None
                        else pd.Timestamp(last_trade["source_start_utc"])
                    ),
                    "staleness_seconds": (
                        None
                        if last_trade is None
                        else float(
                            (deadline - last_trade["source_start_utc"])
                            / pd.Timedelta(seconds=1)
                        )
                    ),
                    "liquidity": (
                        None
                        if bin_trades.empty
                        else float(bin_trades["size"].sum())
                    ),
                    "order_ambiguous": order_ambiguous,
                    "contaminated": horizon_contaminated,
                    "censored": censored,
                    "censoring_status": (
                        "CENSORED"
                        if censored
                        else (
                            "CENSORING_NOT_REPORTED"
                            if coverage_end is None
                            else "NOT_CENSORED"
                        )
                    ),
                    "forward_filled": False,
                    "observed": observed,
                    "attrition_reason": reason,
                    "jump": None,
                    "jump_share": None,
                    "shape_classification": "UNIDENTIFIED_SPARSE",
                    **covariates,
                    "control_eligible": observed
                    and not contaminated
                    and not order_ambiguous
                    and not censored,
                    "claim_boundary": active_spec.claim_boundary,
                }
                bin_rows.append(row)
                prior_horizon = horizon
            jump, jump_share, classification = _shape(
                bin_rows,
                spec=active_spec,
                globally_invalid=(
                    order_ambiguous
                    or any(bool(row["contaminated"]) for row in bin_rows)
                    or any(bool(row["censored"]) for row in bin_rows)
                ),
            )
            for row in bin_rows:
                row["jump"] = jump
                row["jump_share"] = jump_share
                row["shape_classification"] = classification
            path_rows.extend(bin_rows)
            summary_rows.append(
                {
                    "game_id": episode["game_id"],
                    "episode_id": episode["episode_id"],
                    "logical_market_id": logical_market_id,
                    "outcome": outcome,
                    "venue": venue,
                    "market_family": market_family,
                    "outcome_orientation": orientation,
                    "source_time_first_trade": first_trade,
                    "order_ambiguous": order_ambiguous,
                    "contaminated": any(
                        bool(row["contaminated"]) for row in bin_rows
                    ),
                    "censoring_status": (
                        "CENSORING_NOT_REPORTED"
                        if coverage_end is None
                        else "REPORTED"
                    ),
                    "jump": jump,
                    "jump_share": jump_share,
                    "shape_classification": classification,
                    "claim_boundary": active_spec.claim_boundary,
                }
            )
    paths = pd.DataFrame(path_rows, columns=list(_PATH_COLUMNS))
    paths = paths.sort_values(
        [
            "game_id",
            "episode_id",
            "logical_market_id",
            "outcome",
            "venue",
            "horizon_seconds",
        ],
        kind="mergesort",
    ).reset_index(drop=True) if not paths.empty else paths
    summaries = pd.DataFrame(summary_rows)
    attrition_columns = [
        "observation_id",
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "market_family",
        "outcome_orientation",
        "horizon_seconds",
        "observed",
        "attrition_reason",
        "order_ambiguous",
        "contaminated",
        "censored",
        "censoring_status",
        "forward_filled",
        "claim_boundary",
    ]
    attrition = (
        paths[attrition_columns].copy()
        if not paths.empty
        else _empty(attrition_columns)
    )
    return DenseReactionTablesV2(
        paths=paths,
        path_summaries=summaries,
        attrition=attrition,
    )


def _normalize_match_observations(
    observations: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    frame = _require_frame(observations, label=label)
    _require_columns(
        frame,
        {
            "observation_id",
            "game_id",
            "episode_id",
            *DENSE_EXACT_MATCH_KEYS,
            "effect",
            "control_eligible",
            "treated_eligible",
        },
        label=label,
    )
    for field in (
        "observation_id",
        "game_id",
        "episode_id",
        "venue",
        "market_family",
        "outcome_orientation",
    ):
        frame[field] = _required_identifier(frame, field, label=label)
    frame["horizon_seconds"] = _finite_numeric(
        frame, "horizon_seconds", label=label, nonnegative=True
    ).astype("int64")
    if not frame["horizon_seconds"].isin(DENSE_HORIZONS_SECONDS).all():
        raise DenseReactionError(f"{label} contains an unregistered dense horizon")
    frame["control_eligible"] = _boolean(
        frame, "control_eligible", label=label
    )
    frame["treated_eligible"] = _boolean(
        frame, "treated_eligible", label=label
    )
    effect = pd.to_numeric(frame["effect"], errors="coerce").astype(float)
    required_effect = (
        frame["treated_eligible"]
        if label == "treated"
        else frame["control_eligible"]
    )
    if (~np.isfinite(effect) & required_effect).any():
        raise DenseReactionError(
            f"{label} eligible observations must have finite effect"
        )
    frame["effect"] = effect
    if frame.duplicated("observation_id").any():
        raise DenseReactionError(f"{label} has duplicate observation_id values")
    return frame


def _active_covariates(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric: list[str] = []
    categorical: list[str] = []
    treated_active = treated["treated_eligible"]
    controls_active = controls["control_eligible"]
    for field in _NUMERIC_COVARIATES:
        supplied = field in treated.columns or field in controls.columns
        if not supplied:
            continue
        if field not in treated.columns or field not in controls.columns:
            raise DenseReactionError(
                f"matching covariate {field} must be supplied for both groups"
            )
        treated_values = pd.to_numeric(treated[field], errors="coerce").astype(
            float
        )
        control_values = pd.to_numeric(controls[field], errors="coerce").astype(
            float
        )
        if (
            (~np.isfinite(treated_values) & treated_active).any()
            or (~np.isfinite(control_values) & controls_active).any()
        ):
            raise DenseReactionError(
                f"matching covariate {field} must be finite for eligible rows"
            )
        treated[field] = treated_values
        controls[field] = control_values
        if not treated_active.any() or not controls_active.any():
            continue
        numeric.append(field)
    for field in _CATEGORICAL_COVARIATES:
        supplied = field in treated.columns or field in controls.columns
        if not supplied:
            continue
        if field not in treated.columns or field not in controls.columns:
            raise DenseReactionError(
                f"matching covariate {field} must be supplied for both groups"
            )
        treated_values = treated[field]
        control_values = controls[field]
        invalid_treated = (
            treated_values.isna()
            | treated_values.astype(str).str.strip().eq("")
        ) & treated_active
        invalid_controls = (
            control_values.isna()
            | control_values.astype(str).str.strip().eq("")
        ) & controls_active
        if invalid_treated.any() or invalid_controls.any():
            raise DenseReactionError(
                f"matching covariate {field} is blank for an eligible row"
            )
        treated[field] = treated_values.astype(str)
        controls[field] = control_values.astype(str)
        if not treated_active.any() or not controls_active.any():
            continue
        categorical.append(field)
    return tuple(numeric), tuple(categorical)


def _match_distance(
    treated: pd.Series,
    controls: pd.DataFrame,
    *,
    numeric_covariates: Sequence[str],
) -> np.ndarray:
    distance = np.zeros(len(controls), dtype=float)
    for field in numeric_covariates:
        values = controls[field].to_numpy(dtype=float)
        center = float(np.median(values))
        scale = float(np.std(values, ddof=0))
        if scale <= 0:
            scale = max(abs(float(treated[field]) - center), 1.0)
        distance += ((values - float(treated[field])) / scale) ** 2
    return np.sqrt(distance)


def _smd(treated_value: float, controls: pd.Series) -> float:
    control_values = controls.to_numpy(dtype=float)
    control_mean = float(np.mean(control_values))
    control_variance = float(np.var(control_values, ddof=0))
    difference = abs(treated_value - control_mean)
    if math.isclose(difference, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return 0.0
    if control_variance <= 0:
        return math.inf
    return difference / math.sqrt(control_variance / 2)


def build_dense_matched_effects_v2(
    treated_observations: pd.DataFrame,
    control_observations: pd.DataFrame,
    *,
    spec: DenseMatchedReactionSpecV2 | None = None,
) -> DenseMatchedEffectsV2:
    """Match each treated dense observation and return a complete row audit.

    Exact keys are venue, market family, orientation, and horizon.  Supplied
    possession and actor/beneficiary covariates are conditioned exactly; every
    supplied numeric covariate is included in distance and SMD balance checks.
    """

    active_spec = spec or DenseMatchedReactionSpecV2()
    treated = _normalize_match_observations(treated_observations, label="treated")
    controls = _normalize_match_observations(control_observations, label="controls")
    numeric_covariates, categorical_covariates = _active_covariates(
        treated, controls
    )
    match_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []

    for treated_row in treated.sort_values(
        ["game_id", "episode_id", "observation_id"], kind="mergesort"
    ).to_dict("records"):
        row = pd.Series(treated_row)
        if not bool(row["treated_eligible"]):
            audit_rows.append(
                {
                    "treated_observation_id": row["observation_id"],
                    "game_id": row["game_id"],
                    "episode_id": row["episode_id"],
                    **{key: row[key] for key in DENSE_EXACT_MATCH_KEYS},
                    "exact_match_keys": DENSE_EXACT_MATCH_KEYS,
                    "active_numeric_covariates": numeric_covariates,
                    "active_categorical_covariates": categorical_covariates,
                    "raw_exact_control_count": 0,
                    "eligible_exact_control_count": 0,
                    "exact_control_count": 0,
                    "same_game_control_count": 0,
                    "selected_control_count": 0,
                    "match_scope": "NONE",
                    "maximum_absolute_smd": math.nan,
                    "match_status": "UNMATCHED_INELIGIBLE_TREATED",
                    "unmatched_reason": "TREATED_NOT_ELIGIBLE",
                    "claim_boundary": active_spec.claim_boundary,
                }
            )
            continue
        exact_mask = pd.Series(True, index=controls.index)
        for key in DENSE_EXACT_MATCH_KEYS:
            exact_mask &= controls[key].eq(row[key])
        raw_exact = controls[exact_mask].copy()
        raw_exact = raw_exact[
            ~(
                raw_exact["game_id"].eq(row["game_id"])
                & raw_exact["episode_id"].eq(row["episode_id"])
            )
        ]
        eligible = raw_exact[raw_exact["control_eligible"]].copy()
        compatible = eligible.copy()
        for field in categorical_covariates:
            compatible = compatible[compatible[field].eq(row[field])]
        same_game = compatible[compatible["game_id"].eq(row["game_id"])]
        if raw_exact.empty:
            unmatched_reason: str | None = "NO_EXACT_KEY_CONTROLS"
            pool = compatible
        elif eligible.empty:
            unmatched_reason = "NO_ELIGIBLE_EXACT_CONTROLS"
            pool = compatible
        elif len(compatible) < active_spec.minimum_controls:
            unmatched_reason = "INSUFFICIENT_EXACT_CONTROLS"
            pool = compatible
        else:
            unmatched_reason = None
            pool = (
                same_game
                if len(same_game) >= active_spec.minimum_controls
                else compatible
            )
        selected = _empty(controls.columns)
        maximum_smd = math.nan
        balance_pass = False
        match_scope = "NONE"
        if unmatched_reason is None:
            selected = pool.copy()
            selected["match_distance"] = _match_distance(
                row, selected, numeric_covariates=numeric_covariates
            )
            selected = selected.sort_values(
                ["match_distance", "observation_id", "game_id", "episode_id"],
                kind="mergesort",
            ).head(active_spec.maximum_controls)
            match_scope = (
                "SAME_GAME"
                if len(same_game) >= active_spec.minimum_controls
                else "CROSS_GAME_FALLBACK"
            )
            feature_smd = {
                field: _smd(float(row[field]), selected[field])
                for field in numeric_covariates
            }
            maximum_smd = max(feature_smd.values(), default=0.0)
            balance_pass = maximum_smd <= active_spec.maximum_absolute_smd
            balance_rows.append(
                {
                    "treated_observation_id": row["observation_id"],
                    **{
                        f"smd_{field}": feature_smd[field]
                        for field in numeric_covariates
                    },
                    "maximum_absolute_smd": maximum_smd,
                    "balance_pass": balance_pass,
                    "claim_boundary": active_spec.claim_boundary,
                }
            )
            if not balance_pass:
                unmatched_reason = "SMD_EXCEEDS_0_10"
            else:
                control_effect = float(selected["effect"].mean())
                effect_rows.append(
                    {
                        "treated_observation_id": row["observation_id"],
                        "game_id": row["game_id"],
                        "episode_id": row["episode_id"],
                        **{key: row[key] for key in DENSE_EXACT_MATCH_KEYS},
                        "treated_effect": float(row["effect"]),
                        "matched_control_effect": control_effect,
                        "matched_effect": float(row["effect"] - control_effect),
                        "selected_control_count": len(selected),
                        "match_scope": match_scope,
                        "maximum_absolute_smd": maximum_smd,
                        "claim_boundary": active_spec.claim_boundary,
                    }
                )
            weight = 1.0 / len(selected)
            for control in selected.to_dict("records"):
                match_rows.append(
                    {
                        "treated_observation_id": row["observation_id"],
                        "control_observation_id": control["observation_id"],
                        "game_id": row["game_id"],
                        "control_game_id": control["game_id"],
                        "episode_id": row["episode_id"],
                        "control_episode_id": control["episode_id"],
                        **{key: row[key] for key in DENSE_EXACT_MATCH_KEYS},
                        "treated_effect": float(row["effect"]),
                        "control_effect": float(control["effect"]),
                        "pair_effect": float(row["effect"] - control["effect"]),
                        "match_distance": float(control["match_distance"]),
                        "match_weight": weight,
                        "match_scope": match_scope,
                        "balance_pass": balance_pass,
                        **{
                            f"treated_{field}": row[field]
                            for field in (*numeric_covariates, *categorical_covariates)
                        },
                        **{
                            f"control_{field}": control[field]
                            for field in (*numeric_covariates, *categorical_covariates)
                        },
                        "claim_boundary": active_spec.claim_boundary,
                    }
                )
        audit_rows.append(
            {
                "treated_observation_id": row["observation_id"],
                "game_id": row["game_id"],
                "episode_id": row["episode_id"],
                **{key: row[key] for key in DENSE_EXACT_MATCH_KEYS},
                "exact_match_keys": DENSE_EXACT_MATCH_KEYS,
                "active_numeric_covariates": numeric_covariates,
                "active_categorical_covariates": categorical_covariates,
                "raw_exact_control_count": len(raw_exact),
                "eligible_exact_control_count": len(eligible),
                "exact_control_count": len(compatible),
                "same_game_control_count": len(same_game),
                "selected_control_count": len(selected),
                "match_scope": match_scope,
                "maximum_absolute_smd": maximum_smd,
                "match_status": "MATCHED" if unmatched_reason is None else "UNMATCHED",
                "unmatched_reason": unmatched_reason,
                "claim_boundary": active_spec.claim_boundary,
            }
        )
    return DenseMatchedEffectsV2(
        matched_controls=pd.DataFrame(match_rows),
        match_audit=pd.DataFrame(audit_rows),
        match_balance=pd.DataFrame(balance_rows),
        matched_effects=pd.DataFrame(effect_rows),
    )


def adapt_dense_paths_to_matched_observations_v2(
    paths: pd.DataFrame,
    *,
    routine_control_episode_ids: Collection[str],
) -> pd.DataFrame:
    """Create matching inputs from sealed dense paths without accepting effects.

    ``effect`` is always copied from the eligible path's
    ``cumulative_change``.  The source path identity and source column remain
    attached so a publication can trace each matched observation to a sealed
    actual-trade path.
    """

    frame = _require_frame(paths, label="dense paths")
    _require_columns(
        frame,
        {
            "observation_id",
            "game_id",
            "episode_id",
            *DENSE_EXACT_MATCH_KEYS,
            "cumulative_change",
            "observed",
            "forward_filled",
            "order_ambiguous",
            "contaminated",
            "censored",
            "claim_boundary",
        },
        label="dense paths",
    )
    if not frame["claim_boundary"].eq(CLAIM_BOUNDARY).all():
        raise DenseReactionError(
            "dense paths must retain the SOURCE_TIME_ONLY claim boundary"
        )
    for field in (
        "observed",
        "forward_filled",
        "order_ambiguous",
        "contaminated",
        "censored",
    ):
        frame[field] = _boolean(frame, field, label="dense paths")
    if isinstance(routine_control_episode_ids, str):
        raise DenseReactionError("routine_control_episode_ids must be a collection")
    control_episode_ids = {
        str(value).strip() for value in routine_control_episode_ids
    }
    if any(not value for value in control_episode_ids):
        raise DenseReactionError("routine_control_episode_ids contains a blank value")
    effect = pd.to_numeric(frame["cumulative_change"], errors="coerce").astype(
        float
    )
    admissible = (
        frame["observed"]
        & ~frame["order_ambiguous"]
        & ~frame["contaminated"]
        & ~frame["censored"]
        & ~frame["forward_filled"]
        & np.isfinite(effect)
    )
    result = frame.copy()
    result["effect"] = effect
    result["treated_eligible"] = admissible
    result["source_path_observation_id"] = result["observation_id"]
    result["effect_source_column"] = "cumulative_change"
    result["source_claim_boundary"] = result["claim_boundary"]
    result["control_eligible"] = result["treated_eligible"] & result[
        "episode_id"
    ].isin(control_episode_ids)
    return result


__all__ = [
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "DENSE_EXACT_MATCH_KEYS",
    "DENSE_HORIZONS_SECONDS",
    "DenseMatchedEffectsV2",
    "DenseMatchedReactionSpecV2",
    "DenseReactionError",
    "DenseReactionSpecV2",
    "DenseReactionTablesV2",
    "adapt_dense_paths_to_matched_observations_v2",
    "build_dense_matched_effects_v2",
    "build_dense_reaction_tables_v2",
]
