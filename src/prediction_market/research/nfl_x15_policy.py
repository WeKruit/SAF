"""Deterministic, non-executable policy selection for NFL X-15.

The module consumes development out-of-fold predictions only.  It never reads
the sealed holdout, creates orders, or treats historical trades as fills.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


class X15PolicyError(ValueError):
    """An input cannot support the governed X-15 signal claim."""


_PREDICTION_COLUMNS = frozenset(
    {
        "game_id",
        "episode_id",
        "venue",
        "logical_market_id",
        "outcome",
        "source_time_utc",
        "model_id",
        "landmark_seconds",
        "endpoint_seconds",
        "predicted_delta",
        "predicted_class",
        "materiality_threshold",
        "target_delta",
        "mark_landmark",
        "mark_endpoint",
        "decision_eligible",
        "target_eligible",
    }
)
_GROUP_COLUMNS = (
    "game_id",
    "episode_id",
    "venue",
    "logical_market_id",
    "outcome",
    "model_id",
)
_DIRECTIONS = {-1: "DOWN", 1: "UP"}


def _require_columns(frame: pd.DataFrame, columns: frozenset[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise X15PolicyError(f"missing required columns: {missing}")


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise X15PolicyError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise X15PolicyError(f"{field} must be finite")
    return number


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _adjacent_direction_confirmed(rows: pd.DataFrame, direction: int) -> bool:
    ordered = rows.sort_values("endpoint_seconds", kind="mergesort")
    matches = [
        _direction(float(row["predicted_delta"])) == direction
        and str(row["predicted_class"]) == _DIRECTIONS[direction]
        for row in ordered.to_dict("records")
    ]
    return any(left and right for left, right in zip(matches, matches[1:]))


def _audit_row(
    group: Sequence[object],
    *,
    landmark: int,
    reason: str,
    selected: bool,
) -> dict[str, object]:
    return {
        **dict(zip(_GROUP_COLUMNS, group)),
        "landmark_seconds": landmark,
        "reason": reason,
        "selected": selected,
    }


def select_earliest_landmark_signals(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select at most one source-time signal for each episode and model.

    A landmark qualifies only when its predicted magnitude exceeds its
    training-fold materiality threshold and two adjacent future endpoints
    agree on direction.  The selected endpoint is the first one reaching
    ninety percent of the largest predicted movement in that direction.
    """

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise X15PolicyError("predictions must be a nonempty DataFrame")
    _require_columns(predictions, _PREDICTION_COLUMNS)
    if "dataset_partition" in predictions.columns and not predictions[
        "dataset_partition"
    ].eq("development").all():
        raise X15PolicyError(
            "X-15 policy selection accepts development rows only"
        )

    frame = predictions.copy()
    frame["source_time_utc"] = pd.to_datetime(
        frame["source_time_utc"], utc=True, errors="raise"
    )
    for field in ("decision_eligible", "target_eligible"):
        if not frame[field].map(
            lambda value: value is True or value is False
        ).all():
            raise X15PolicyError(f"{field} must contain booleans")
    signals: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for group, episode_rows in frame.groupby(
        list(_GROUP_COLUMNS), sort=True, dropna=False
    ):
        selected = False
        for landmark, rows in episode_rows.groupby(
            "landmark_seconds", sort=True, dropna=False
        ):
            landmark_seconds = int(landmark)
            if selected:
                audit.append(
                    _audit_row(
                        group,
                        landmark=landmark_seconds,
                        reason="LATER_LANDMARK_AFTER_SELECTION",
                        selected=False,
                    )
                )
                continue
            decision_rows = rows[rows["decision_eligible"].eq(True)].copy()
            if decision_rows.empty:
                audit.append(
                    _audit_row(
                        group,
                        landmark=landmark_seconds,
                        reason="NO_DECISION_ELIGIBLE_ENDPOINT",
                        selected=False,
                    )
                )
                continue
            thresholds = {
                _finite_number(value, field="materiality_threshold")
                for value in decision_rows["materiality_threshold"]
            }
            if len(thresholds) != 1:
                raise X15PolicyError(
                    "one landmark must use exactly one materiality threshold"
                )
            threshold = next(iter(thresholds))
            if threshold <= 0:
                raise X15PolicyError(
                    "materiality_threshold must be positive"
                )
            decision_rows["predicted_delta"] = decision_rows[
                "predicted_delta"
            ].map(
                lambda value: _finite_number(value, field="predicted_delta")
            )
            material = decision_rows[
                decision_rows["predicted_delta"].abs().ge(threshold)
            ].copy()
            if material.empty:
                audit.append(
                    _audit_row(
                        group,
                        landmark=landmark_seconds,
                        reason="BELOW_MATERIALITY",
                        selected=False,
                    )
                )
                continue

            best = max(
                material["predicted_delta"],
                key=lambda value: abs(float(value)),
            )
            direction = _direction(float(best))
            directional = decision_rows[
                decision_rows["predicted_delta"].map(_direction).eq(direction)
                & decision_rows["predicted_class"].eq(
                    _DIRECTIONS[direction]
                )
            ].copy()
            if not _adjacent_direction_confirmed(decision_rows, direction):
                audit.append(
                    _audit_row(
                        group,
                        landmark=landmark_seconds,
                        reason="NO_ADJACENT_DIRECTION_CONFIRMATION",
                        selected=False,
                    )
                )
                continue
            endpoint_threshold = 0.9 * max(
                directional["predicted_delta"].abs()
            )
            chosen = directional[
                directional["predicted_delta"].abs().ge(endpoint_threshold)
            ].sort_values("endpoint_seconds", kind="mergesort").iloc[0]
            target_eligible = bool(chosen["target_eligible"])
            if target_eligible:
                target_delta = _finite_number(
                    chosen["target_delta"], field="target_delta"
                )
                gross_markout = direction * target_delta
                evaluation_status = "EVALUATED"
                evidence_classification = (
                    "NON_EXECUTABLE_TRADE_MARKED_DIAGNOSTIC"
                )
            else:
                gross_markout = math.nan
                evaluation_status = "CENSORED_EVALUATION"
                evidence_classification = "CENSORED_EVALUATION"
            signal = chosen.to_dict()
            signal.update(
                {
                    "direction": direction,
                    "gross_markout": gross_markout,
                    "evaluation_status": evaluation_status,
                    "entry_time_utc": chosen["source_time_utc"]
                    + pd.Timedelta(seconds=landmark_seconds),
                    "exit_time_utc": chosen["source_time_utc"]
                    + pd.Timedelta(
                        seconds=int(chosen["endpoint_seconds"])
                    ),
                    "evidence_classification": evidence_classification,
                    "dataset_partition": "development",
                }
            )
            signals.append(signal)
            audit.append(
                _audit_row(
                    group,
                    landmark=landmark_seconds,
                    reason="SELECTED",
                    selected=True,
                )
            )
            selected = True

    return (
        pd.DataFrame(signals),
        pd.DataFrame(audit),
    )


def build_trade_marked_shadow_curve(
    signals: pd.DataFrame,
    *,
    fixed_contracts: int,
    initial_nav: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a gross, non-executable curve without overlapping positions."""

    if not isinstance(signals, pd.DataFrame) or signals.empty:
        raise X15PolicyError("signals must be a nonempty DataFrame")
    required = frozenset(
        {
            "game_id",
            "episode_id",
            "venue",
            "logical_market_id",
            "outcome",
            "source_time_utc",
            "landmark_seconds",
            "endpoint_seconds",
            "gross_markout",
            "evaluation_status",
        }
    )
    _require_columns(signals, required)
    if isinstance(fixed_contracts, bool) or not isinstance(
        fixed_contracts, int
    ) or fixed_contracts <= 0:
        raise X15PolicyError("fixed_contracts must be a positive integer")
    nav = _finite_number(initial_nav, field="initial_nav")
    if nav <= 0:
        raise X15PolicyError("initial_nav must be positive")

    frame = signals.copy()
    source_time = pd.to_datetime(
        frame["source_time_utc"], utc=True, errors="raise"
    )
    frame["entry_time_utc"] = source_time + pd.to_timedelta(
        frame["landmark_seconds"], unit="s"
    )
    frame["exit_time_utc"] = source_time + pd.to_timedelta(
        frame["endpoint_seconds"], unit="s"
    )
    if (frame["exit_time_utc"] <= frame["entry_time_utc"]).any():
        raise X15PolicyError("exit_time_utc must be after entry_time_utc")
    frame = frame.sort_values(
        ["entry_time_utc", "game_id", "venue", "episode_id"],
        kind="mergesort",
    )

    open_until: dict[tuple[object, ...], pd.Timestamp] = {}
    selected_rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        raw_markout = row["gross_markout"]
        if (
            str(row["evaluation_status"]) != "EVALUATED"
            or isinstance(raw_markout, bool)
            or not isinstance(raw_markout, (int, float))
            or not math.isfinite(float(raw_markout))
        ):
            audit.append(
                {
                    "game_id": row["game_id"],
                    "episode_id": row["episode_id"],
                    "venue": row["venue"],
                    "reason": "CENSORED_EVALUATION_SKIPPED",
                    "selected": False,
                }
            )
            continue
        key = (
            row["game_id"],
            row["venue"],
            row["logical_market_id"],
        )
        entry_time = pd.Timestamp(row["entry_time_utc"])
        exit_time = pd.Timestamp(row["exit_time_utc"])
        if key in open_until and entry_time < open_until[key]:
            audit.append(
                {
                    "game_id": row["game_id"],
                    "episode_id": row["episode_id"],
                    "venue": row["venue"],
                    "reason": "OVERLAPPING_POSITION_SKIPPED",
                    "selected": False,
                }
            )
            continue
        selected_rows.append(row)
        audit.append(
            {
                "game_id": row["game_id"],
                "episode_id": row["episode_id"],
                "venue": row["venue"],
                "reason": "SELECTED",
                "selected": True,
            }
        )
        open_until[key] = exit_time

    kept: list[dict[str, object]] = []
    peak_nav = nav
    for row in sorted(
        selected_rows,
        key=lambda value: (
            pd.Timestamp(value["exit_time_utc"]),
            pd.Timestamp(value["entry_time_utc"]),
            str(value["game_id"]),
            str(value["episode_id"]),
        ),
    ):
        pnl = fixed_contracts * _finite_number(
            row["gross_markout"], field="gross_markout"
        )
        nav += pnl
        peak_nav = max(peak_nav, nav)
        kept.append(
            {
                **row,
                "fixed_contracts": fixed_contracts,
                "pnl": pnl,
                "nav": nav,
                "peak_nav": peak_nav,
                "drawdown": nav / peak_nav - 1.0,
                "executable": False,
                "order_status": "NO_ORDERS_CREATED",
                "fill_status": "NO_FILL_OBSERVED_NON_EXECUTABLE",
                "evidence_classification": (
                    "NON_EXECUTABLE_TRADE_MARKED_DIAGNOSTIC"
                ),
            }
        )

    return pd.DataFrame(kept), pd.DataFrame(audit)
