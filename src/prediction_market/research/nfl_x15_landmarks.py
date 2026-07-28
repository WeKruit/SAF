"""Pure X-15 landmark-to-endpoint targets from observed moneyline trades.

The event anchor is the end of the finalized information-event interval.
Every mark is independently selected from an actual post-event trade no more
than three seconds old.  This module has no artifact, network, holdout, model,
or order-execution dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import pandas as pd


LANDMARK_SECONDS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
ENDPOINT_SECONDS: Final[tuple[int, ...]] = tuple(range(5, 61, 5))
MAX_STALENESS_SECONDS: Final[float] = 3.0
CLAIM_BOUNDARY: Final[str] = (
    "HISTORICAL_ACTUAL_TRADE_LANDMARK_TARGET_NOT_EXECUTION"
)
MODEL_PASSTHROUGH_COLUMNS: Final[tuple[str, ...]] = (
    "nfl_week",
    "is_matched_control",
    "factor_id",
    "reference_gap_at_landmark",
    "game_seconds_remaining",
    "score_margin",
    "pit_strength",
)

_TRADE_REQUIRED = {
    "trade_id",
    "game_id",
    "venue",
    "logical_market_id",
    "market_family",
    "outcome_team",
    "source_time_utc",
    "price",
    "kind",
    "provenance",
    "orientation_resolved",
}
_EPISODE_REQUIRED = {
    "game_id",
    "episode_id",
    "beneficiary_team",
    "finalized_interval_end_utc",
    "event_type",
    "game_seconds_remaining",
    "score_margin",
    "reference_delta",
    "p_after_ref",
    "nfl_week",
    "is_matched_control",
    "factor_id",
    "game_seconds_remaining",
    "score_margin",
    "pit_strength",
    "orientation_resolved",
    "contaminated",
    "contamination_time_utc",
    "censored",
    "censor_time_utc",
}
_GRAIN = [
    "game_id",
    "episode_id",
    "venue",
    "logical_market_id",
    "beneficiary_team",
    "landmark_seconds",
    "endpoint_seconds",
]
_OUTPUT_COLUMNS = [
    "game_id",
    "episode_id",
    "venue",
    "logical_market_id",
    "beneficiary_team",
    "market_family",
    "event_type",
    "event_anchor_semantics",
    "event_anchor_utc",
    "nfl_week",
    "is_matched_control",
    "factor_id",
    "reference_gap_at_landmark",
    "game_seconds_remaining",
    "score_margin",
    "pit_strength",
    "reference_delta",
    "landmark_seconds",
    "endpoint_seconds",
    "landmark_utc",
    "endpoint_utc",
    "mark_landmark_trade_id",
    "mark_landmark_source_time_utc",
    "mark_landmark_price",
    "landmark_staleness_seconds",
    "mark_endpoint_trade_id",
    "mark_endpoint_source_time_utc",
    "mark_endpoint_price",
    "endpoint_staleness_seconds",
    "target_delta",
    "target_orientation",
    "price_basis",
    "max_staleness_seconds",
    "decision_eligible",
    "target_observed",
    "target_eligible",
    "eligible",
    "exclusion_reasons",
    "claim_boundary",
]


class NFLX15LandmarkError(ValueError):
    """Inputs cannot support an auditable X-15 landmark target."""


@dataclass(frozen=True, slots=True)
class NFLX15LandmarkSpecV1:
    landmarks_seconds: tuple[int, ...] = LANDMARK_SECONDS
    endpoints_seconds: tuple[int, ...] = ENDPOINT_SECONDS
    max_staleness_seconds: float = MAX_STALENESS_SECONDS
    event_anchor: str = "FINALIZED_EVENT_INTERVAL_END"
    price_basis: str = "LATEST_UNIQUE_ACTUAL_TRADE"
    target_orientation: str = "BENEFICIARY_OUTCOME"
    claim_boundary: str = CLAIM_BOUNDARY


SPEC_V1: Final[NFLX15LandmarkSpecV1] = NFLX15LandmarkSpecV1()


@dataclass(frozen=True, slots=True)
class _Mark:
    trade_id: str | None
    source_time: pd.Timestamp | None
    price: float
    staleness_seconds: float
    orientation_resolved: bool | None
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class _MarketIndex:
    times_ns: np.ndarray
    trade_ids: np.ndarray
    prices: np.ndarray
    orientations_resolved: np.ndarray


def _missing_columns(frame: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(frame.columns))


def _nonempty_strings(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise NFLX15LandmarkError(f"{name}.{column} must be non-empty")
        frame[column] = values.astype(str)


def _strict_bool(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        if not frame[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise NFLX15LandmarkError(f"{name}.{column} must contain booleans")
        frame[column] = frame[column].astype(bool)


def _utc_series(
    values: pd.Series,
    *,
    field: str,
    nullable: bool,
) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError) as exc:
        raise NFLX15LandmarkError(f"{field} must contain UTC timestamps") from exc
    invalid = parsed.isna() & (values.notna() if nullable else True)
    if bool(invalid.any()) or (not nullable and bool(parsed.isna().any())):
        raise NFLX15LandmarkError(f"{field} must contain UTC timestamps")
    return parsed


def _validate_trades(actual_trades: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(actual_trades, pd.DataFrame) or actual_trades.empty:
        raise NFLX15LandmarkError("actual_trades must be a non-empty DataFrame")
    missing = _missing_columns(actual_trades, _TRADE_REQUIRED)
    if missing:
        raise NFLX15LandmarkError(
            "actual_trades missing columns: " + ", ".join(missing)
        )
    trades = actual_trades.copy(deep=True)
    _nonempty_strings(
        trades,
        (
            "trade_id",
            "game_id",
            "venue",
            "logical_market_id",
            "market_family",
            "outcome_team",
            "kind",
            "provenance",
        ),
        "actual_trades",
    )
    if trades["trade_id"].duplicated().any():
        raise NFLX15LandmarkError("actual_trades.trade_id must be unique")
    if not (
        trades["kind"].eq("trade") & trades["provenance"].eq("observed")
    ).all():
        raise NFLX15LandmarkError(
            "actual_trades must contain only actual observed trades"
        )
    _strict_bool(trades, ("orientation_resolved",), "actual_trades")
    trades["source_time_utc"] = _utc_series(
        trades["source_time_utc"],
        field="actual_trades.source_time_utc",
        nullable=False,
    )
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    if (
        trades["price"].isna().any()
        or not np.isfinite(trades["price"]).all()
        or not trades["price"].between(0, 1).all()
    ):
        raise NFLX15LandmarkError("actual_trades.price must be finite in [0, 1]")
    return trades


def _validate_episodes(finalized_episode_state: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(finalized_episode_state, pd.DataFrame)
        or finalized_episode_state.empty
    ):
        raise NFLX15LandmarkError(
            "finalized_episode_state must be a non-empty DataFrame"
        )
    missing = _missing_columns(finalized_episode_state, _EPISODE_REQUIRED)
    if missing:
        raise NFLX15LandmarkError(
            "finalized_episode_state missing columns: " + ", ".join(missing)
        )
    episodes = finalized_episode_state.copy(deep=True)
    _nonempty_strings(
        episodes,
        (
            "game_id",
            "episode_id",
            "beneficiary_team",
            "event_type",
            "factor_id",
        ),
        "finalized_episode_state",
    )
    if episodes.duplicated(["game_id", "episode_id"]).any():
        raise NFLX15LandmarkError(
            "finalized_episode_state episode grain must be unique"
        )
    _strict_bool(
        episodes,
        (
            "orientation_resolved",
            "contaminated",
            "censored",
            "is_matched_control",
        ),
        "finalized_episode_state",
    )
    episodes["finalized_interval_end_utc"] = _utc_series(
        episodes["finalized_interval_end_utc"],
        field="finalized_episode_state.finalized_interval_end_utc",
        nullable=False,
    )
    for field in ("contamination_time_utc", "censor_time_utc"):
        episodes[field] = _utc_series(
            episodes[field],
            field=f"finalized_episode_state.{field}",
            nullable=True,
        )
    for field in ("game_seconds_remaining", "score_margin"):
        episodes[field] = pd.to_numeric(episodes[field], errors="coerce")
        if episodes[field].isna().any() or not np.isfinite(episodes[field]).all():
            raise NFLX15LandmarkError(
                f"finalized_episode_state.{field} must be finite"
            )
    week = pd.to_numeric(episodes["nfl_week"], errors="coerce")
    if (
        week.isna().any()
        or not np.equal(
            week.to_numpy(dtype=float),
            week.to_numpy(dtype="int64"),
        ).all()
    ):
        raise NFLX15LandmarkError(
            "finalized_episode_state.nfl_week must contain integer weeks"
        )
    episodes["nfl_week"] = week.astype("int64")
    if not episodes["nfl_week"].between(1, 12).all():
        raise NFLX15LandmarkError(
            "finalized_episode_state only accepts NFL weeks 1 through 12"
        )
    episodes["p_after_ref"] = pd.to_numeric(
        episodes["p_after_ref"], errors="coerce"
    )
    if (
        episodes["p_after_ref"].isna().any()
        or not np.isfinite(episodes["p_after_ref"]).all()
        or not episodes["p_after_ref"].between(0, 1).all()
    ):
        raise NFLX15LandmarkError(
            "finalized_episode_state.p_after_ref must be finite in [0, 1]"
        )
    raw_pit_strength = episodes["pit_strength"]
    episodes["pit_strength"] = pd.to_numeric(
        raw_pit_strength, errors="coerce"
    )
    invalid_pit = raw_pit_strength.notna() & episodes["pit_strength"].isna()
    if bool(invalid_pit.any()):
        raise NFLX15LandmarkError(
            "finalized_episode_state.pit_strength must be numeric or null"
        )
    nonnull_pit = episodes["pit_strength"].dropna()
    if not np.isfinite(nonnull_pit).all():
        raise NFLX15LandmarkError(
            "finalized_episode_state.pit_strength must be numeric or null"
        )
    episodes["reference_delta"] = pd.to_numeric(
        episodes["reference_delta"], errors="coerce"
    )
    nonnull_reference = episodes["reference_delta"].dropna()
    if not np.isfinite(nonnull_reference).all():
        raise NFLX15LandmarkError(
            "finalized_episode_state.reference_delta must be finite or null"
        )
    return episodes


def _select_mark(
    trades: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    offset_seconds: int,
    role: str,
) -> _Mark:
    target_time = anchor + pd.Timedelta(seconds=offset_seconds)
    candidates = trades[
        trades["source_time_utc"].gt(anchor)
        & trades["source_time_utc"].le(target_time)
    ]
    if candidates.empty:
        return _Mark(
            trade_id=None,
            source_time=None,
            price=math.nan,
            staleness_seconds=math.nan,
            orientation_resolved=None,
            exclusion_reason=f"{role}_no_actual_trade",
        )
    latest = candidates["source_time_utc"].max()
    latest_rows = candidates[candidates["source_time_utc"].eq(latest)]
    staleness = float((target_time - latest).total_seconds())
    if len(latest_rows) != 1:
        return _Mark(
            trade_id=None,
            source_time=latest,
            price=math.nan,
            staleness_seconds=staleness,
            orientation_resolved=None,
            exclusion_reason=f"{role}_order_ambiguous",
        )
    row = latest_rows.iloc[0]
    if staleness > MAX_STALENESS_SECONDS:
        return _Mark(
            trade_id=None,
            source_time=latest,
            price=math.nan,
            staleness_seconds=staleness,
            orientation_resolved=bool(row["orientation_resolved"]),
            exclusion_reason=f"{role}_stale",
        )
    return _Mark(
        trade_id=str(row["trade_id"]),
        source_time=latest,
        price=float(row["price"]),
        staleness_seconds=staleness,
        orientation_resolved=bool(row["orientation_resolved"]),
        exclusion_reason=None,
    )


def _market_index(trades: pd.DataFrame) -> _MarketIndex:
    ordered = trades.sort_values(
        ["source_time_utc", "trade_id"], kind="mergesort"
    )
    return _MarketIndex(
        times_ns=ordered["source_time_utc"]
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy(),
        trade_ids=ordered["trade_id"].astype(str).to_numpy(),
        prices=ordered["price"].to_numpy(dtype=float),
        orientations_resolved=ordered["orientation_resolved"].to_numpy(
            dtype=bool
        ),
    )


def _select_mark_indexed(
    market: _MarketIndex,
    *,
    anchor: pd.Timestamp,
    offset_seconds: int,
    role: str,
) -> _Mark:
    anchor_ns = int(anchor.value)
    target_ns = anchor_ns + offset_seconds * 1_000_000_000
    position = int(np.searchsorted(market.times_ns, target_ns, side="right")) - 1
    if position < 0 or int(market.times_ns[position]) <= anchor_ns:
        return _Mark(
            trade_id=None,
            source_time=None,
            price=math.nan,
            staleness_seconds=math.nan,
            orientation_resolved=None,
            exclusion_reason=f"{role}_no_actual_trade",
        )
    latest_ns = int(market.times_ns[position])
    duplicate_before = (
        position > 0 and int(market.times_ns[position - 1]) == latest_ns
    )
    duplicate_after = (
        position + 1 < len(market.times_ns)
        and int(market.times_ns[position + 1]) == latest_ns
    )
    latest = pd.Timestamp(latest_ns, tz="UTC")
    staleness = float((target_ns - latest_ns) / 1_000_000_000)
    if duplicate_before or duplicate_after:
        return _Mark(
            trade_id=None,
            source_time=latest,
            price=math.nan,
            staleness_seconds=staleness,
            orientation_resolved=None,
            exclusion_reason=f"{role}_order_ambiguous",
        )
    if staleness > MAX_STALENESS_SECONDS:
        return _Mark(
            trade_id=None,
            source_time=latest,
            price=math.nan,
            staleness_seconds=staleness,
            orientation_resolved=bool(
                market.orientations_resolved[position]
            ),
            exclusion_reason=f"{role}_stale",
        )
    return _Mark(
        trade_id=str(market.trade_ids[position]),
        source_time=latest,
        price=float(market.prices[position]),
        staleness_seconds=staleness,
        orientation_resolved=bool(market.orientations_resolved[position]),
        exclusion_reason=None,
    )


def _append_once(reasons: list[str], reason: str | None) -> None:
    if reason is not None and reason not in reasons:
        reasons.append(reason)


def _decision_exclusions(
    episode: pd.Series,
    landmark_mark: _Mark,
    *,
    landmark_time: pd.Timestamp,
) -> list[str]:
    reasons: list[str] = []
    if (
        not bool(episode["orientation_resolved"])
        or (
            landmark_mark.exclusion_reason is None
            and landmark_mark.orientation_resolved is not True
        )
    ):
        reasons.append("orientation_unresolved")
    if bool(episode["contaminated"]):
        reasons.append("episode_contaminated")
    contamination_time = episode["contamination_time_utc"]
    if pd.notna(contamination_time) and contamination_time <= landmark_time:
        reasons.append("contaminated_before_landmark")
    if bool(episode["censored"]):
        reasons.append("episode_censored")
    censor_time = episode["censor_time_utc"]
    if pd.notna(censor_time) and censor_time <= landmark_time:
        reasons.append("censored_before_landmark")
    _append_once(reasons, landmark_mark.exclusion_reason)
    return reasons


def _target_exclusions(
    episode: pd.Series,
    endpoint_mark: _Mark,
    *,
    landmark_time: pd.Timestamp,
    endpoint_time: pd.Timestamp,
    decision_reasons: list[str],
) -> list[str]:
    reasons = list(decision_reasons)
    if (
        endpoint_mark.exclusion_reason is None
        and endpoint_mark.orientation_resolved is not True
    ):
        _append_once(reasons, "orientation_unresolved")
    contamination_time = episode["contamination_time_utc"]
    if (
        pd.notna(contamination_time)
        and landmark_time < contamination_time <= endpoint_time
    ):
        _append_once(reasons, "contaminated_before_endpoint")
    censor_time = episode["censor_time_utc"]
    if (
        pd.notna(censor_time)
        and landmark_time < censor_time <= endpoint_time
    ):
        _append_once(reasons, "censored_before_endpoint")
    _append_once(reasons, endpoint_mark.exclusion_reason)
    return reasons


def build_nfl_x15_landmark_panel(
    actual_trades: pd.DataFrame,
    finalized_episode_state: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen beneficiary-moneyline landmark target panel.

    Output grain:
    ``game × episode × venue × beneficiary moneyline × landmark × endpoint``.
    ``decision_eligible`` uses only information observable by the landmark.
    ``target_observed`` records whether both actual trade marks exist, while
    ``target_eligible`` also applies endpoint contamination/censoring rules.
    ``eligible`` is the internal alias for ``target_eligible``.  Combinations
    without an eligible future target remain in the audit panel with a null
    target and explicit ``exclusion_reasons``.
    """

    trades = _validate_trades(actual_trades)
    episodes = _validate_episodes(finalized_episode_state)
    moneyline = trades[trades["market_family"].eq("moneyline")].copy()
    markets_by_game: dict[
        str,
        list[
            tuple[
                str,
                str,
                pd.DataFrame,
                dict[str, _MarketIndex],
            ]
        ],
    ] = {}
    for (game_id, venue, logical_market_id), market in moneyline.groupby(
        ["game_id", "venue", "logical_market_id"],
        sort=True,
        dropna=False,
    ):
        indexes = {
            str(outcome): _market_index(group)
            for outcome, group in market.groupby(
                "outcome_team", sort=True, dropna=False
            )
        }
        markets_by_game.setdefault(str(game_id), []).append(
            (str(venue), str(logical_market_id), market, indexes)
        )
    rows: list[dict[str, object]] = []
    for episode in episodes.sort_values(
        ["game_id", "finalized_interval_end_utc", "episode_id"],
        kind="mergesort",
    ).itertuples(index=False):
        episode_row = pd.Series(episode._asdict())
        anchor = episode_row["finalized_interval_end_utc"]
        beneficiary = str(episode_row["beneficiary_team"])
        scoped_markets = markets_by_game.get(str(episode_row["game_id"]), [])
        for venue, logical_market_id, market, outcome_indexes in scoped_markets:
            market_index = outcome_indexes.get(
                beneficiary,
                _MarketIndex(
                    times_ns=np.asarray([], dtype=np.int64),
                    trade_ids=np.asarray([], dtype=object),
                    prices=np.asarray([], dtype=float),
                    orientations_resolved=np.asarray([], dtype=bool),
                ),
            )
            landmark_marks = {
                offset: _select_mark_indexed(
                    market_index,
                    anchor=anchor,
                    offset_seconds=offset,
                    role="landmark",
                )
                for offset in LANDMARK_SECONDS
            }
            endpoint_marks = {
                offset: _select_mark_indexed(
                    market_index,
                    anchor=anchor,
                    offset_seconds=offset,
                    role="endpoint",
                )
                for offset in ENDPOINT_SECONDS
            }
            for landmark in LANDMARK_SECONDS:
                landmark_mark = landmark_marks[landmark]
                for endpoint in ENDPOINT_SECONDS:
                    if endpoint <= landmark:
                        continue
                    landmark_time = anchor + pd.Timedelta(seconds=landmark)
                    endpoint_time = anchor + pd.Timedelta(seconds=endpoint)
                    endpoint_mark = endpoint_marks[endpoint]
                    decision_reasons = _decision_exclusions(
                        episode_row,
                        landmark_mark,
                        landmark_time=landmark_time,
                    )
                    reasons = _target_exclusions(
                        episode_row,
                        endpoint_mark,
                        landmark_time=landmark_time,
                        endpoint_time=endpoint_time,
                        decision_reasons=decision_reasons,
                    )
                    decision_eligible = not decision_reasons
                    target_observed = (
                        landmark_mark.exclusion_reason is None
                        and endpoint_mark.exclusion_reason is None
                    )
                    target_eligible = not reasons
                    rows.append(
                        {
                            "game_id": str(episode_row["game_id"]),
                            "episode_id": str(episode_row["episode_id"]),
                            "venue": str(venue),
                            "logical_market_id": str(logical_market_id),
                            "beneficiary_team": beneficiary,
                            "market_family": "moneyline",
                            "event_type": str(episode_row["event_type"]),
                            "event_anchor_semantics": (
                                "FINALIZED_EVENT_INTERVAL_END"
                            ),
                            "event_anchor_utc": anchor,
                            "nfl_week": int(episode_row["nfl_week"]),
                            "is_matched_control": bool(
                                episode_row["is_matched_control"]
                            ),
                            "factor_id": str(episode_row["factor_id"]),
                            "reference_gap_at_landmark": (
                                landmark_mark.price
                                - float(episode_row["p_after_ref"])
                                if landmark_mark.exclusion_reason is None
                                else math.nan
                            ),
                            "game_seconds_remaining": float(
                                episode_row["game_seconds_remaining"]
                            ),
                            "score_margin": float(episode_row["score_margin"]),
                            "pit_strength": (
                                float(episode_row["pit_strength"])
                                if pd.notna(episode_row["pit_strength"])
                                else math.nan
                            ),
                            "reference_delta": (
                                float(episode_row["reference_delta"])
                                if pd.notna(episode_row["reference_delta"])
                                else math.nan
                            ),
                            "landmark_seconds": landmark,
                            "endpoint_seconds": endpoint,
                            "landmark_utc": landmark_time,
                            "endpoint_utc": endpoint_time,
                            "mark_landmark_trade_id": landmark_mark.trade_id,
                            "mark_landmark_source_time_utc": (
                                landmark_mark.source_time
                            ),
                            "mark_landmark_price": landmark_mark.price,
                            "landmark_staleness_seconds": (
                                landmark_mark.staleness_seconds
                            ),
                            "mark_endpoint_trade_id": endpoint_mark.trade_id,
                            "mark_endpoint_source_time_utc": (
                                endpoint_mark.source_time
                            ),
                            "mark_endpoint_price": endpoint_mark.price,
                            "endpoint_staleness_seconds": (
                                endpoint_mark.staleness_seconds
                            ),
                            "target_delta": (
                                endpoint_mark.price - landmark_mark.price
                                if target_eligible
                                else math.nan
                            ),
                            "target_orientation": "BENEFICIARY_OUTCOME",
                            "price_basis": "LATEST_UNIQUE_ACTUAL_TRADE",
                            "max_staleness_seconds": MAX_STALENESS_SECONDS,
                            "decision_eligible": decision_eligible,
                            "target_observed": target_observed,
                            "target_eligible": target_eligible,
                            "eligible": target_eligible,
                            "exclusion_reasons": tuple(reasons),
                            "claim_boundary": CLAIM_BOUNDARY,
                        }
                    )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    if panel.duplicated(_GRAIN).any():
        raise NFLX15LandmarkError("landmark panel grain is not unique")
    return (
        panel.sort_values(_GRAIN, kind="mergesort")
        .reset_index(drop=True)
        .reindex(columns=_OUTPUT_COLUMNS)
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "ENDPOINT_SECONDS",
    "LANDMARK_SECONDS",
    "MAX_STALENESS_SECONDS",
    "MODEL_PASSTHROUGH_COLUMNS",
    "NFLX15LandmarkError",
    "NFLX15LandmarkSpecV1",
    "SPEC_V1",
    "build_nfl_x15_landmark_panel",
]
