"""Matched-control, historical source-time reaction analysis for NFL X-13.

This module is deliberately pure: it consumes already verified development
tables and returns publishable tables.  It never loads upstream data, opens the
sealed holdout, reconstructs a game, or represents historical source-time
intervals as measured latency.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
BUILDER_VERSION = "nfl-x13-matched-reaction-v1"
HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60)
MATCH_FEATURES = (
    "pre_probability",
    "relative_time_seconds",
    "score_margin",
    "staleness_seconds",
    "liquidity",
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EXACT_KEYS = (
    "venue",
    "market_family",
    "outcome_orientation",
    "delay_seconds",
    "horizon_seconds",
)
_RESULT_KEYS = (
    "factor_id",
    "factor_version",
    "factor_family",
    "market_family",
    "delay_seconds",
    "horizon_seconds",
    "venue",
)
_REQUIRED_COLUMNS = {
    "observation_id",
    "factor_id",
    "factor_version",
    "factor_family",
    "game_id",
    "episode_id",
    "logical_market_id",
    "venue",
    "market_family",
    "outcome_orientation",
    "horizon_seconds",
    "delay_seconds",
    "effect",
    "maximum_excursion",
    *MATCH_FEATURES,
    "routine_control_eligible",
    "hit_factor_ids",
    "claim_boundary",
    "source_hashes",
}


class MatchedReactionError(ValueError):
    """A matched-reaction input or invariant failed closed."""


@dataclass(frozen=True, slots=True)
class MatchedReactionSpecV1:
    """Frozen matching and inference choices for the X-13 development cohort."""

    horizons_seconds: tuple[int, ...] = HORIZONS_SECONDS
    nearest_controls: int = 20
    minimum_controls: int = 10
    maximum_absolute_smd: float = 0.10
    materiality_floor: float = 0.01
    control_noise_quantile: float = 0.95
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_726
    minimum_games_per_venue: int = 30
    bh_review_threshold: float = 0.10
    bh_formal_threshold: float = 0.05
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.horizons_seconds != HORIZONS_SECONDS:
            raise MatchedReactionError(
                "horizons_seconds must be frozen at 1/2/5/10/20/30/60"
            )
        if not (
            1 <= self.minimum_controls <= self.nearest_controls
        ):
            raise MatchedReactionError(
                "minimum_controls must be between 1 and nearest_controls"
            )
        if not (0 < self.maximum_absolute_smd <= 1):
            raise MatchedReactionError(
                "maximum_absolute_smd must be in (0, 1]"
            )
        if not (0 < self.control_noise_quantile < 1):
            raise MatchedReactionError(
                "control_noise_quantile must be in (0, 1)"
            )
        if self.materiality_floor <= 0:
            raise MatchedReactionError("materiality_floor must be positive")
        if self.bootstrap_samples < 1:
            raise MatchedReactionError("bootstrap_samples must be positive")
        if self.minimum_games_per_venue < 1:
            raise MatchedReactionError(
                "minimum_games_per_venue must be positive"
            )
        if not (
            0
            < self.bh_formal_threshold
            <= self.bh_review_threshold
            <= 1
        ):
            raise MatchedReactionError("BH thresholds are invalid")
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise MatchedReactionError(
                "historical matched analysis must remain SOURCE_TIME_ONLY"
            )


@dataclass(frozen=True, slots=True)
class MatchedReactionTablesV1:
    """All row-level and aggregate tables produced by the pure builder."""

    matched_controls: pd.DataFrame
    match_audit: pd.DataFrame
    match_balance: pd.DataFrame
    source_time_paths: pd.DataFrame
    single_game_results: pd.DataFrame
    cross_game_results: pd.DataFrame
    venue_consistency: pd.DataFrame
    manifest: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PublishedObservationInputsV1:
    """Canonical treated/control inputs adapted from published X-13 rows."""

    treated_observations: pd.DataFrame
    routine_control_observations: pd.DataFrame


_PUBLISHED_REQUIRED_COLUMNS = {
    "factor_observation_id",
    "factor_id",
    "factor_version",
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
    "market_family",
    "horizon_seconds",
    "delay_seconds",
    "signed_price_change",
    "maximum_excursion",
    "pre_event_actual_trade",
    "event_source_time_utc",
    "game_seconds_remaining",
    "score_margin_focal",
    "staleness_seconds",
    "event_eligible",
    "episode_audit_only",
    "episode_is_score",
    "episode_is_turnover",
    "penalty",
    "replay_or_challenge",
    "timeout",
    "safety",
    "punt_blocked",
    "return_touchdown",
    "fumble_lost",
    "interception",
    "fourth_down_failed",
    "claim_boundary",
    "sports_source_hashes",
    "market_source_hashes",
}
_ACTUAL_MARKET_REQUIRED_COLUMNS = {
    "venue",
    "logical_market_id",
    "outcome",
    "kind",
    "source_start_utc",
    "size",
    "primary_path_eligible",
}
_PRE_EVENT_LIQUIDITY_LOOKBACK_SECONDS = 60


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _finite_numeric(
    frame: pd.DataFrame,
    field: str,
    *,
    nonnegative: bool = False,
) -> pd.Series:
    values = pd.to_numeric(frame[field], errors="coerce").astype(float)
    invalid = ~np.isfinite(values)
    if nonnegative:
        invalid |= values < 0
    if invalid.any():
        raise MatchedReactionError(
            f"{field} must contain finite"
            + (" nonnegative" if nonnegative else "")
            + " values"
        )
    return values


def _valid_source_hashes(value: object) -> bool:
    if isinstance(value, np.ndarray):
        values = tuple(value.tolist())
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        return False
    return bool(values) and all(
        isinstance(item, str) and _SHA256_RE.fullmatch(item) is not None
        for item in values
    )


def _hash_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        values = tuple(value.tolist())
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        raise MatchedReactionError(f"{field} must be a SHA-256 sequence")
    if not _valid_source_hashes(values):
        raise MatchedReactionError(f"{field} must be a SHA-256 sequence")
    return tuple(str(item) for item in values)


def _factor_id_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        values = tuple(value.tolist())
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        raise MatchedReactionError(f"{field} must be a factor-ID sequence")
    normalized = tuple(sorted({str(item).strip() for item in values}))
    if not normalized or any(not item for item in normalized):
        raise MatchedReactionError(
            f"{field} must be a nonempty factor-ID sequence"
        )
    return normalized


def _published_factor_family(factor_id: object) -> str:
    parts = str(factor_id).split(".")
    if len(parts) < 3 or parts[0] != "NFL":
        raise MatchedReactionError(
            f"published factor_id has no registered family: {factor_id}"
        )
    return parts[1].lower()


def _published_canonical_row(
    row: pd.Series,
    *,
    observation_id: str,
    factor_id: str,
    factor_version: str,
    factor_family: str,
    routine_control_eligible: bool,
    hit_factor_ids: Sequence[str],
) -> dict[str, object]:
    hashes = tuple(
        sorted(
            set(
                _hash_sequence(
                    row["sports_source_hashes"],
                    field="sports_source_hashes",
                )
            )
            | set(
                _hash_sequence(
                    row["market_source_hashes"],
                    field="market_source_hashes",
                )
            )
        )
    )
    return {
        "observation_id": observation_id,
        "factor_id": factor_id,
        "factor_version": factor_version,
        "factor_family": factor_family,
        "game_id": str(row["game_id"]),
        "episode_id": str(row["episode_id"]),
        "logical_market_id": str(row["logical_market_id"]),
        "venue": str(row["venue"]),
        "market_family": str(row["market_family"]),
        "outcome_orientation": str(row["outcome"]),
        "horizon_seconds": row["horizon_seconds"],
        "delay_seconds": row["delay_seconds"],
        "effect": row["signed_price_change"],
        "maximum_excursion": row["maximum_excursion"],
        "pre_probability": row["pre_event_actual_trade"],
        "relative_time_seconds": row["game_seconds_remaining"],
        "score_margin": row["score_margin_focal"],
        "staleness_seconds": row["staleness_seconds"],
        "liquidity": row["_pre_event_liquidity"],
        "routine_control_eligible": routine_control_eligible,
        "hit_factor_ids": _factor_id_sequence(
            tuple(hit_factor_ids), field="hit_factor_ids"
        ),
        "claim_boundary": row["claim_boundary"],
        "source_hashes": hashes,
    }


def _attach_pre_event_liquidity(
    factor_rows: pd.DataFrame,
    actual_market_observations: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(actual_market_observations, pd.DataFrame):
        raise MatchedReactionError(
            "actual market observations must be a DataFrame"
        )
    missing = sorted(
        _ACTUAL_MARKET_REQUIRED_COLUMNS.difference(
            actual_market_observations.columns
        )
    )
    if missing:
        raise MatchedReactionError(
            "actual market observations missing required columns: "
            + ", ".join(missing)
        )
    market = actual_market_observations.copy()
    primary = market["primary_path_eligible"]
    if not primary.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise MatchedReactionError(
            "actual market primary_path_eligible must be boolean"
        )
    market = market[
        primary.astype(bool) & market["kind"].eq("trade")
    ].copy()
    market["_source_ns"] = pd.to_datetime(
        market["source_start_utc"], utc=True, errors="coerce"
    ).astype("int64")
    if market["_source_ns"].eq(np.iinfo(np.int64).min).any():
        raise MatchedReactionError(
            "actual market source_start_utc must be valid UTC"
        )
    market["_size"] = pd.to_numeric(
        market["size"], errors="coerce"
    ).astype(float)
    if (
        ~np.isfinite(market["_size"])
        | market["_size"].lt(0)
    ).any():
        raise MatchedReactionError(
            "actual market size must be finite and nonnegative"
        )
    result = factor_rows.copy()
    result["_event_ns"] = pd.to_datetime(
        result["event_source_time_utc"], utc=True, errors="coerce"
    ).astype("int64")
    if result["_event_ns"].eq(np.iinfo(np.int64).min).any():
        raise MatchedReactionError(
            "published event_source_time_utc must be valid UTC"
        )
    group_keys = [
        "venue",
        "logical_market_id",
        "outcome",
    ]
    market_groups: dict[
        tuple[str, str, str], tuple[np.ndarray, np.ndarray]
    ] = {}
    for key, child in market.groupby(
        group_keys, sort=False, dropna=False
    ):
        ordered = child.sort_values("_source_ns", kind="mergesort")
        times = ordered["_source_ns"].to_numpy(dtype=np.int64)
        prefix = np.concatenate(
            (
                np.asarray([0.0]),
                np.cumsum(ordered["_size"].to_numpy(dtype=float)),
            )
        )
        market_groups[tuple(str(value) for value in key)] = (
            times,
            prefix,
        )
    liquidity = np.zeros(len(result), dtype=float)
    lookback_ns = _PRE_EVENT_LIQUIDITY_LOOKBACK_SECONDS * 1_000_000_000
    for key, indexes in result.groupby(
        group_keys, sort=False, dropna=False
    ).groups.items():
        normalized_key = tuple(str(value) for value in key)
        group = market_groups.get(normalized_key)
        if group is None:
            raise MatchedReactionError(
                "published factor row has no exact actual market stream"
            )
        times, prefix = group
        event_times = result.loc[indexes, "_event_ns"].to_numpy(
            dtype=np.int64
        )
        left = np.searchsorted(
            times, event_times - lookback_ns, side="left"
        )
        # Same-source-second rows remain unordered, so equality is excluded.
        right = np.searchsorted(times, event_times, side="left")
        positions = result.index.get_indexer(indexes)
        liquidity[positions] = prefix[right] - prefix[left]
    result["_pre_event_liquidity"] = liquidity
    return result.drop(columns=["_event_ns"])


def adapt_published_x13_factor_observations(
    published_factor_observations: pd.DataFrame,
    actual_market_observations: pd.DataFrame,
) -> PublishedObservationInputsV1:
    """Adapt exact published factor rows without replay or upstream access.

    Controls are common, non-salient eligible episodes already present in the
    published factor table.  Multiple factor hits on the same market/episode
    are deduplicated only after their economic and state fields agree exactly.
    """

    if not isinstance(published_factor_observations, pd.DataFrame):
        raise MatchedReactionError(
            "published factor observations must be a DataFrame"
        )
    missing = sorted(
        _PUBLISHED_REQUIRED_COLUMNS.difference(
            published_factor_observations.columns
        )
    )
    if missing:
        raise MatchedReactionError(
            "published factor observations missing required columns: "
            + ", ".join(missing)
        )
    frame = _attach_pre_event_liquidity(
        published_factor_observations,
        actual_market_observations,
    )
    if not frame["claim_boundary"].eq(CLAIM_BOUNDARY).all():
        raise MatchedReactionError(
            "historical claim_boundary must be SOURCE_TIME_ONLY"
        )
    treated_rows = [
        _published_canonical_row(
            row,
            observation_id=str(row["factor_observation_id"]),
            factor_id=str(row["factor_id"]),
            factor_version=str(row["factor_version"]),
            factor_family=_published_factor_family(row["factor_id"]),
            routine_control_eligible=False,
            hit_factor_ids=(str(row["factor_id"]),),
        )
        for _, row in frame.iterrows()
    ]
    salient_fields = (
        "episode_is_score",
        "episode_is_turnover",
        "penalty",
        "replay_or_challenge",
        "timeout",
        "safety",
        "punt_blocked",
        "return_touchdown",
        "fumble_lost",
        "interception",
        "fourth_down_failed",
    )
    routine_mask = (
        frame["event_eligible"].fillna(False).astype(bool)
        & ~frame["episode_audit_only"].fillna(True).astype(bool)
    )
    for field in salient_fields:
        routine_mask &= ~frame[field].fillna(False).astype(bool)
    routine = frame[routine_mask].copy()
    duplicate_grain = [
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "market_family",
        "horizon_seconds",
        "delay_seconds",
    ]
    agreement_fields = [
        "signed_price_change",
        "maximum_excursion",
        "pre_event_actual_trade",
        "game_seconds_remaining",
        "score_margin_focal",
        "staleness_seconds",
        "_pre_event_liquidity",
        "claim_boundary",
    ]
    control_rows: list[dict[str, object]] = []
    for key, child in routine.groupby(
        duplicate_grain, sort=True, dropna=False
    ):
        if any(
            child[field].nunique(dropna=False) != 1
            for field in agreement_fields
        ):
            raise MatchedReactionError(
                "routine duplicate rows disagree on state or economic fields"
            )
        source_hashes = tuple(
            sorted(
                {
                    source_hash
                    for _, child_row in child.iterrows()
                    for source_hash in (
                        *_hash_sequence(
                            child_row["sports_source_hashes"],
                            field="sports_source_hashes",
                        ),
                        *_hash_sequence(
                            child_row["market_source_hashes"],
                            field="market_source_hashes",
                        ),
                    )
                }
            )
        )
        row = child.sort_values(
            "factor_observation_id", kind="mergesort"
        ).iloc[0].copy()
        row["sports_source_hashes"] = source_hashes
        row["market_source_hashes"] = source_hashes
        hit_factor_ids = tuple(
            sorted({str(value) for value in child["factor_id"]})
        )
        observation_id = "routine_" + _canonical_sha256(
            dict(zip(duplicate_grain, key, strict=True))
        ).removeprefix("sha256:")
        control_rows.append(
            _published_canonical_row(
                row,
                observation_id=observation_id,
                factor_id="NFL.CONTROL.ROUTINE",
                factor_version="v1",
                factor_family="control",
                routine_control_eligible=True,
                hit_factor_ids=hit_factor_ids,
            )
        )
    return PublishedObservationInputsV1(
        treated_observations=pd.DataFrame(treated_rows),
        routine_control_observations=pd.DataFrame(control_rows),
    )


def _validate_observations(
    frame: pd.DataFrame,
    *,
    label: str,
    spec: MatchedReactionSpecV1,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise MatchedReactionError(f"{label} observations must be a DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise MatchedReactionError(
            f"{label} observations missing required columns: "
            + ", ".join(missing)
        )
    result = frame.copy()
    for field in (
        "observation_id",
        "factor_id",
        "factor_version",
        "factor_family",
        "game_id",
        "episode_id",
        "logical_market_id",
        "venue",
        "market_family",
        "outcome_orientation",
    ):
        if (
            result[field].isna().any()
            or result[field].astype(str).str.strip().eq("").any()
        ):
            raise MatchedReactionError(
                f"{label} observations have blank {field}"
            )
        result[field] = result[field].astype(str)
    result["horizon_seconds"] = _finite_numeric(
        result, "horizon_seconds", nonnegative=True
    ).astype("int64")
    result["delay_seconds"] = _finite_numeric(
        result, "delay_seconds", nonnegative=True
    ).astype("int64")
    if not set(result["horizon_seconds"]).issubset(
        spec.horizons_seconds
    ):
        raise MatchedReactionError(
            f"{label} observations contain an unregistered horizon"
        )
    result["effect"] = _finite_numeric(result, "effect")
    result["maximum_excursion"] = _finite_numeric(
        result, "maximum_excursion"
    )
    result["pre_probability"] = _finite_numeric(
        result, "pre_probability"
    )
    if not result["pre_probability"].between(0, 1).all():
        raise MatchedReactionError("pre_probability must be in [0, 1]")
    for field in MATCH_FEATURES[1:]:
        result[field] = _finite_numeric(
            result,
            field,
            nonnegative=field in {"staleness_seconds", "liquidity"},
        )
    routine = result["routine_control_eligible"]
    if not routine.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise MatchedReactionError(
            "routine_control_eligible must contain booleans"
        )
    result["routine_control_eligible"] = routine.astype(bool)
    result["hit_factor_ids"] = result["hit_factor_ids"].map(
        lambda value: _factor_id_sequence(value, field="hit_factor_ids")
    )
    if not result["claim_boundary"].eq(spec.claim_boundary).all():
        raise MatchedReactionError(
            "historical claim_boundary must be SOURCE_TIME_ONLY"
        )
    if not result["source_hashes"].map(_valid_source_hashes).all():
        raise MatchedReactionError(
            f"{label} source_hashes must be nonempty SHA-256 sequences"
        )
    grain = [
        "observation_id",
        "game_id",
        "episode_id",
        "logical_market_id",
        "venue",
        "delay_seconds",
        "horizon_seconds",
    ]
    if result.duplicated(grain).any():
        raise MatchedReactionError(
            f"{label} observation grain is duplicated"
        )
    return result


def _candidate_distance(
    treated: pd.Series,
    controls: pd.DataFrame,
) -> np.ndarray:
    distance = np.zeros(len(controls), dtype=float)
    for feature in MATCH_FEATURES:
        control_values = controls[feature].to_numpy(dtype=float)
        center = float(np.median(control_values))
        scale = float(np.std(control_values, ddof=0))
        if scale <= 0:
            scale = max(abs(float(treated[feature]) - center), 1.0)
        distance += (
            (control_values - float(treated[feature])) / scale
        ) ** 2
    return np.sqrt(distance)


def _match_rows(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    routine_controls = controls[controls["routine_control_eligible"]]
    exact_indexes = {
        tuple(key): tuple(indexes)
        for key, indexes in routine_controls.groupby(
            list(_EXACT_KEYS), sort=False, dropna=False
        ).groups.items()
    }
    for row in treated.sort_values(
        [
            "factor_id",
            "game_id",
            "episode_id",
            "venue",
            "delay_seconds",
            "horizon_seconds",
            "observation_id",
        ],
        kind="mergesort",
    ).itertuples(index=False):
        exact_key = tuple(getattr(row, key) for key in _EXACT_KEYS)
        exact_pool = controls.loc[
            list(exact_indexes.get(exact_key, ()))
        ].copy()
        exact_pool = exact_pool[
            ~(
                exact_pool["game_id"].eq(row.game_id)
                & exact_pool["episode_id"].eq(row.episode_id)
            )
        ]
        exposed = exact_pool["hit_factor_ids"].map(
            lambda values: row.factor_id in values
        ).astype(bool)
        factor_exposed_control_count = int(exposed.sum())
        exact_pool = exact_pool[~exposed].copy()
        same_game = exact_pool[
            exact_pool["game_id"].eq(row.game_id)
        ].copy()
        if len(same_game) >= spec.minimum_controls:
            pool = same_game
            scope = "SAME_GAME"
        elif len(exact_pool) >= spec.minimum_controls:
            pool = exact_pool
            scope = "CROSS_GAME_FALLBACK"
        else:
            audit_rows.append(
                {
                    "treated_observation_id": row.observation_id,
                    "factor_id": row.factor_id,
                    "factor_version": row.factor_version,
                    "factor_family": row.factor_family,
                    "game_id": row.game_id,
                    "episode_id": row.episode_id,
                    "venue": row.venue,
                    "market_family": row.market_family,
                    "outcome_orientation": row.outcome_orientation,
                    "delay_seconds": row.delay_seconds,
                    "horizon_seconds": row.horizon_seconds,
                    "same_game_control_count": len(same_game),
                    "exact_control_count": len(exact_pool),
                    "factor_exposed_control_count": (
                        factor_exposed_control_count
                    ),
                    "selected_control_count": 0,
                    "match_scope": "NONE",
                    "maximum_absolute_smd": math.nan,
                    "match_status": "UNMATCHED_MIN_CONTROLS",
                    "claim_boundary": spec.claim_boundary,
                }
            )
            continue
        pool["match_distance"] = _candidate_distance(
            pd.Series(row._asdict()), pool
        )
        selected = (
            pool.sort_values(
                [
                    "match_distance",
                    "observation_id",
                    "game_id",
                    "episode_id",
                ],
                kind="mergesort",
            )
            .head(spec.nearest_controls)
            .copy()
        )
        count = len(selected)
        for control in selected.itertuples(index=False):
            match_rows.append(
                {
                    "treated_observation_id": row.observation_id,
                    "control_observation_id": control.observation_id,
                    "factor_id": row.factor_id,
                    "factor_version": row.factor_version,
                    "factor_family": row.factor_family,
                    "game_id": row.game_id,
                    "control_game_id": control.game_id,
                    "episode_id": row.episode_id,
                    "control_episode_id": control.episode_id,
                    "logical_market_id": row.logical_market_id,
                    "venue": row.venue,
                    "market_family": row.market_family,
                    "outcome_orientation": row.outcome_orientation,
                    "delay_seconds": row.delay_seconds,
                    "horizon_seconds": row.horizon_seconds,
                    "treated_effect": float(row.effect),
                    "control_effect": float(control.effect),
                    "pair_effect": float(row.effect - control.effect),
                    "treated_maximum_excursion": float(
                        row.maximum_excursion
                    ),
                    "control_maximum_excursion": float(
                        control.maximum_excursion
                    ),
                    "control_hit_factor_ids": tuple(
                        control.hit_factor_ids
                    ),
                    "match_distance": float(control.match_distance),
                    "match_weight": 1.0 / count,
                    "match_scope": scope,
                    "treated_source_hashes": _hash_sequence(
                        row.source_hashes,
                        field="treated source_hashes",
                    ),
                    "control_source_hashes": _hash_sequence(
                        control.source_hashes,
                        field="control source_hashes",
                    ),
                    **{
                        f"treated_{feature}": float(
                            getattr(row, feature)
                        )
                        for feature in MATCH_FEATURES
                    },
                    **{
                        f"control_{feature}": float(
                            getattr(control, feature)
                        )
                        for feature in MATCH_FEATURES
                    },
                    "balance_pass": True,
                    "claim_boundary": spec.claim_boundary,
                }
            )
        audit_rows.append(
            {
                "treated_observation_id": row.observation_id,
                "factor_id": row.factor_id,
                "factor_version": row.factor_version,
                "factor_family": row.factor_family,
                "game_id": row.game_id,
                "episode_id": row.episode_id,
                "venue": row.venue,
                "market_family": row.market_family,
                "outcome_orientation": row.outcome_orientation,
                "delay_seconds": row.delay_seconds,
                "horizon_seconds": row.horizon_seconds,
                "same_game_control_count": len(same_game),
                "exact_control_count": len(exact_pool),
                "factor_exposed_control_count": (
                    factor_exposed_control_count
                ),
                "selected_control_count": count,
                "match_scope": scope,
                "maximum_absolute_smd": math.nan,
                "match_status": "MATCHED",
                "claim_boundary": spec.claim_boundary,
            }
        )
    matches = pd.DataFrame(match_rows)
    audit = pd.DataFrame(audit_rows)
    return matches, audit


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def _balance_matches(
    matches: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return matches, audit, _empty(
            [
                *_RESULT_KEYS,
                "outcome_orientation",
                *(f"smd_{field}" for field in MATCH_FEATURES),
                "maximum_absolute_smd",
                "balance_pass",
                "claim_boundary",
            ]
        )
    group_keys = [*_RESULT_KEYS, "outcome_orientation"]
    balance_rows: list[dict[str, object]] = []
    for key, child in matches.groupby(
        group_keys, sort=True, dropna=False
    ):
        treated = child.drop_duplicates("treated_observation_id")
        weights = child["match_weight"].to_numpy(dtype=float)
        feature_smd: dict[str, float] = {}
        for feature in MATCH_FEATURES:
            treated_values = treated[
                f"treated_{feature}"
            ].to_numpy(dtype=float)
            control_values = child[
                f"control_{feature}"
            ].to_numpy(dtype=float)
            treated_mean = float(np.mean(treated_values))
            control_mean = _weighted_mean(control_values, weights)
            treated_variance = float(np.var(treated_values, ddof=0))
            control_variance = _weighted_variance(
                control_values, weights
            )
            pooled = math.sqrt(
                (treated_variance + control_variance) / 2
            )
            difference = abs(treated_mean - control_mean)
            if math.isclose(
                treated_mean,
                control_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                smd = 0.0
            else:
                smd = difference / pooled if pooled > 0 else math.inf
            feature_smd[feature] = smd
        maximum = max(feature_smd.values())
        balance_rows.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                **{
                    f"smd_{field}": feature_smd[field]
                    for field in MATCH_FEATURES
                },
                "maximum_absolute_smd": maximum,
                "balance_pass": maximum <= spec.maximum_absolute_smd,
                "claim_boundary": spec.claim_boundary,
            }
        )
    balance = pd.DataFrame(balance_rows)
    lookup = balance.set_index(group_keys)["balance_pass"]
    matches = matches.copy()
    matches["balance_pass"] = [
        bool(lookup.loc[tuple(row)])
        for row in matches[group_keys].itertuples(
            index=False, name=None
        )
    ]
    audit = audit.copy()
    matched_mask = audit["match_status"].eq("MATCHED")
    if matched_mask.any():
        audit_keys = [
            tuple(row)
            for row in audit.loc[
                matched_mask, group_keys
            ].itertuples(index=False, name=None)
        ]
        maximum_lookup = balance.set_index(group_keys)[
            "maximum_absolute_smd"
        ]
        maxima = [float(maximum_lookup.loc[key]) for key in audit_keys]
        audit.loc[matched_mask, "maximum_absolute_smd"] = maxima
        audit.loc[matched_mask, "match_status"] = [
            "MATCHED"
            if maximum <= spec.maximum_absolute_smd
            else "UNMATCHED_BALANCE"
            for maximum in maxima
        ]
    return matches, audit, balance


def _noise_floor(values: pd.Series, quantile: float) -> float:
    array = np.abs(values.to_numpy(dtype=float))
    if not len(array):
        return math.nan
    return float(np.quantile(array, quantile, method="higher"))


def _matched_observation_effects(
    matches: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> pd.DataFrame:
    if matches.empty:
        return _empty(
            [
                "treated_observation_id",
                "factor_id",
                "factor_version",
                "factor_family",
                "game_id",
                "episode_id",
                "logical_market_id",
                "venue",
                "market_family",
                "outcome_orientation",
                "delay_seconds",
                "horizon_seconds",
                "treated_effect",
                "matched_control_effect",
                "matched_effect",
                "treated_maximum_excursion",
                "matched_control_maximum_excursion",
                "matched_maximum_excursion",
                "control_noise_floor",
                "materiality_threshold",
                "material_source_time_candidate",
                "claim_boundary",
            ]
        )
    eligible = matches[matches["balance_pass"]].copy()
    if eligible.empty:
        return _empty(
            [
                "treated_observation_id",
                "factor_id",
                "factor_version",
                "factor_family",
                "game_id",
                "episode_id",
                "logical_market_id",
                "venue",
                "market_family",
                "outcome_orientation",
                "delay_seconds",
                "horizon_seconds",
                "treated_effect",
                "matched_control_effect",
                "matched_effect",
                "treated_maximum_excursion",
                "matched_control_maximum_excursion",
                "matched_maximum_excursion",
                "control_noise_floor",
                "materiality_threshold",
                "material_source_time_candidate",
                "claim_boundary",
            ]
        )
    identity = [
        "treated_observation_id",
        "factor_id",
        "factor_version",
        "factor_family",
        "game_id",
        "episode_id",
        "logical_market_id",
        "venue",
        "market_family",
        "outcome_orientation",
        "delay_seconds",
        "horizon_seconds",
    ]
    rows: list[dict[str, object]] = []
    for key, child in eligible.groupby(
        identity, sort=True, dropna=False
    ):
        weights = child["match_weight"].to_numpy(dtype=float)
        control_effect = _weighted_mean(
            child["control_effect"].to_numpy(dtype=float),
            weights,
        )
        treated_effect = float(child["treated_effect"].iloc[0])
        treated_excursion = float(
            child["treated_maximum_excursion"].iloc[0]
        )
        control_excursion = _weighted_mean(
            child["control_maximum_excursion"].to_numpy(dtype=float),
            weights,
        )
        floor = _noise_floor(
            child["control_effect"], spec.control_noise_quantile
        )
        threshold = max(spec.materiality_floor, floor)
        effect = treated_effect - control_effect
        rows.append(
            {
                **dict(zip(identity, key, strict=True)),
                "treated_effect": treated_effect,
                "matched_control_effect": control_effect,
                "matched_effect": effect,
                "treated_maximum_excursion": treated_excursion,
                "matched_control_maximum_excursion": control_excursion,
                "matched_maximum_excursion": (
                    treated_excursion - control_excursion
                ),
                "control_noise_floor": floor,
                "materiality_threshold": threshold,
                "material_source_time_candidate": abs(effect)
                >= threshold,
                "claim_boundary": spec.claim_boundary,
            }
        )
    return pd.DataFrame(rows)


def _build_source_time_paths(
    effects: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> pd.DataFrame:
    columns = [
        "factor_id",
        "factor_version",
        "factor_family",
        "game_id",
        "episode_id",
        "logical_market_id",
        "venue",
        "market_family",
        "outcome_orientation",
        "delay_seconds",
        "observed_horizons_seconds",
        "first_material_horizon_seconds",
        "first_material_source_time_candidate",
        "control_noise_floor",
        "matched_effect_10",
        "matched_effect_60",
        "maximum_excursion_10",
        "maximum_excursion_60",
        "continuation_10_to_60",
        "path_shape_10_to_60",
        "claim_boundary",
    ]
    if effects.empty:
        return _empty(columns)
    path_keys = [
        "factor_id",
        "factor_version",
        "factor_family",
        "game_id",
        "episode_id",
        "logical_market_id",
        "venue",
        "market_family",
        "outcome_orientation",
        "delay_seconds",
    ]
    rows: list[dict[str, object]] = []
    for key, child in effects.groupby(
        path_keys, sort=True, dropna=False
    ):
        child = child.sort_values("horizon_seconds", kind="mergesort")
        material = child[child["material_source_time_candidate"]]
        first_material = (
            None
            if material.empty
            else int(material["horizon_seconds"].iloc[0])
        )
        horizon_effects = dict(
            zip(
                child["horizon_seconds"].astype(int),
                child["matched_effect"].astype(float),
                strict=True,
            )
        )
        effect_10 = horizon_effects.get(10)
        effect_60 = horizon_effects.get(60)
        horizon_excursions = dict(
            zip(
                child["horizon_seconds"].astype(int),
                child["treated_maximum_excursion"].astype(float),
                strict=True,
            )
        )
        if effect_10 is None or effect_60 is None:
            continuation = math.nan
            shape = "INCOMPLETE"
        else:
            continuation = effect_60 - effect_10
            if (
                effect_10 != 0
                and np.sign(continuation) == -np.sign(effect_10)
                and abs(effect_60) < abs(effect_10)
            ):
                shape = "REVERSAL"
            elif (
                effect_10 != 0
                and continuation != 0
                and np.sign(continuation) == np.sign(effect_10)
            ):
                shape = "CONTINUATION"
            else:
                shape = "FLAT_OR_MIXED"
        rows.append(
            {
                **dict(zip(path_keys, key, strict=True)),
                "observed_horizons_seconds": tuple(
                    int(value) for value in child["horizon_seconds"]
                ),
                "first_material_horizon_seconds": first_material,
                "first_material_source_time_candidate": (
                    first_material is not None
                ),
                "control_noise_floor": float(
                    child["control_noise_floor"].max()
                ),
                "matched_effect_10": effect_10,
                "matched_effect_60": effect_60,
                "maximum_excursion_10": horizon_excursions.get(10),
                "maximum_excursion_60": horizon_excursions.get(60),
                "continuation_10_to_60": continuation,
                "path_shape_10_to_60": shape,
                "claim_boundary": spec.claim_boundary,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_single_game_results(
    effects: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> pd.DataFrame:
    columns = [
        *_RESULT_KEYS,
        "game_id",
        "observation_count",
        "episode_count",
        "point_estimate",
        "effect_sum",
        "control_noise_floor",
        "material_effect_count",
        "claim_boundary",
    ]
    if effects.empty:
        return _empty(columns)
    group_keys = [*_RESULT_KEYS, "game_id"]
    rows: list[dict[str, object]] = []
    for key, child in effects.groupby(
        group_keys, sort=True, dropna=False
    ):
        values = child["matched_effect"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "observation_count": len(child),
                "episode_count": int(child["episode_id"].nunique()),
                "point_estimate": float(np.mean(values)),
                "effect_sum": float(np.sum(values)),
                "control_noise_floor": float(
                    child["control_noise_floor"].max()
                ),
                "material_effect_count": int(
                    child["material_source_time_candidate"].sum()
                ),
                "claim_boundary": spec.claim_boundary,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _two_sided_bootstrap_p(samples: np.ndarray) -> float:
    denominator = len(samples) + 1.0
    negative = (float(np.count_nonzero(samples <= 0)) + 1.0) / denominator
    positive = (float(np.count_nonzero(samples >= 0)) + 1.0) / denominator
    return min(1.0, 2.0 * min(negative, positive))


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    if (
        not len(values)
        or np.any(~np.isfinite(values))
        or np.any((values < 0) | (values > 1))
    ):
        raise MatchedReactionError(
            "BH input must contain finite probabilities"
        )
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _build_cross_game_results(
    single_game: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> pd.DataFrame:
    columns = [
        *_RESULT_KEYS,
        "game_count",
        "observation_count",
        "episode_count",
        "point_estimate",
        "ci_low",
        "ci_high",
        "raw_p_value",
        "bh_q_value",
        "loo_sign_stability_rate",
        "maximum_leave_one_game_out_shift",
        "maximum_single_game_contribution",
        "control_noise_floor",
        "bootstrap_seed",
        "bootstrap_samples_requested",
        "bootstrap_samples_valid",
        "support_status",
        "analysis_status",
        "claim_boundary",
    ]
    if single_game.empty:
        return _empty(columns)
    rows: list[dict[str, object]] = []
    for stream_index, (key, child) in enumerate(
        single_game.groupby(
            list(_RESULT_KEYS), sort=True, dropna=False
        )
    ):
        child = child.sort_values("game_id", kind="mergesort")
        game_effects = child["point_estimate"].to_numpy(dtype=float)
        game_count = len(game_effects)
        point = float(np.mean(game_effects))
        generator = np.random.default_rng(
            spec.bootstrap_seed + stream_index
        )
        draw_indices = generator.integers(
            0,
            game_count,
            size=(spec.bootstrap_samples, game_count),
            endpoint=False,
        )
        bootstrap = game_effects[draw_indices].mean(axis=1)
        ci_low, ci_high = np.quantile(
            bootstrap, (0.025, 0.975)
        )
        loo = np.asarray(
            [
                float(np.mean(np.delete(game_effects, index)))
                for index in range(game_count)
            ],
            dtype=float,
        ) if game_count > 1 else np.asarray([], dtype=float)
        loo_stability = (
            float(np.mean(np.sign(loo) == np.sign(point)))
            if len(loo) and point != 0
            else 0.0
        )
        maximum_loo_shift = (
            float(np.max(np.abs(loo - point)))
            if len(loo)
            else math.nan
        )
        contribution_denominator = float(
            np.abs(game_effects).sum()
        )
        maximum_contribution = (
            float(np.abs(game_effects).max() / contribution_denominator)
            if contribution_denominator > 0
            else 1.0
        )
        rows.append(
            {
                **dict(zip(_RESULT_KEYS, key, strict=True)),
                "game_count": game_count,
                "observation_count": int(
                    child["observation_count"].sum()
                ),
                "episode_count": int(child["episode_count"].sum()),
                "point_estimate": point,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "raw_p_value": _two_sided_bootstrap_p(bootstrap),
                "bh_q_value": math.nan,
                "loo_sign_stability_rate": loo_stability,
                "maximum_leave_one_game_out_shift": maximum_loo_shift,
                "maximum_single_game_contribution": maximum_contribution,
                "control_noise_floor": float(
                    child["control_noise_floor"].max()
                ),
                "bootstrap_seed": spec.bootstrap_seed,
                "bootstrap_samples_requested": spec.bootstrap_samples,
                "bootstrap_samples_valid": spec.bootstrap_samples,
                "support_status": (
                    "PASS"
                    if game_count >= spec.minimum_games_per_venue
                    else "INSUFFICIENT_SUPPORT"
                ),
                "analysis_status": "SOURCE_TIME_ONLY_DESCRIPTIVE",
                "claim_boundary": spec.claim_boundary,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    for _, indexes in result.groupby(
        "factor_family", sort=True
    ).groups.items():
        positions = list(indexes)
        result.loc[positions, "bh_q_value"] = _benjamini_hochberg(
            result.loc[positions, "raw_p_value"].to_numpy(dtype=float)
        )
    formal_review = (
        result["support_status"].eq("PASS")
        & result["bh_q_value"].le(spec.bh_formal_threshold)
        & result["loo_sign_stability_rate"].ge(0.80)
        & result["maximum_single_game_contribution"].le(0.25)
    )
    review = (
        result["support_status"].eq("PASS")
        & result["bh_q_value"].le(spec.bh_review_threshold)
    )
    result.loc[review, "analysis_status"] = "MATCHED_SOURCE_TIME_REVIEW"
    result.loc[formal_review, "analysis_status"] = (
        "MATCHED_SOURCE_TIME_REVIEW"
    )
    return result


def _build_venue_consistency(
    cross_game: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1,
) -> pd.DataFrame:
    columns = [
        "factor_id",
        "factor_version",
        "factor_family",
        "market_family",
        "horizon_seconds",
        "kalshi_point_estimate",
        "polymarket_point_estimate",
        "venue_sign_consistent",
        "both_venues_support_pass",
        "claim_boundary",
    ]
    if cross_game.empty:
        return _empty(columns)
    keys = [
        "factor_id",
        "factor_version",
        "factor_family",
        "market_family",
        "horizon_seconds",
    ]
    rows: list[dict[str, object]] = []
    for key, child in cross_game.groupby(
        keys, sort=True, dropna=False
    ):
        by_venue = child.set_index("venue")
        kalshi = (
            float(by_venue.loc["kalshi", "point_estimate"])
            if "kalshi" in by_venue.index
            else math.nan
        )
        polymarket = (
            float(by_venue.loc["polymarket", "point_estimate"])
            if "polymarket" in by_venue.index
            else math.nan
        )
        consistent = (
            math.isfinite(kalshi)
            and math.isfinite(polymarket)
            and kalshi != 0
            and polymarket != 0
            and np.sign(kalshi) == np.sign(polymarket)
        )
        support = (
            {"kalshi", "polymarket"}.issubset(by_venue.index)
            and by_venue.loc[
                ["kalshi", "polymarket"], "support_status"
            ].eq("PASS").all()
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "kalshi_point_estimate": kalshi,
                "polymarket_point_estimate": polymarket,
                "venue_sign_consistent": bool(consistent),
                "both_venues_support_pass": bool(support),
                "claim_boundary": spec.claim_boundary,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _jsonable(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _semantic_rows_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    rows = [
        {column: _jsonable(row[column]) for column in ordered.columns}
        for _, row in ordered.iterrows()
    ]
    rows.sort(
        key=lambda row: json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return _canonical_sha256(rows)


def _build_manifest(
    tables: Mapping[str, pd.DataFrame],
    *,
    spec: MatchedReactionSpecV1,
    input_source_hashes: Sequence[str],
) -> pd.DataFrame:
    source_hashes = tuple(sorted(set(input_source_hashes)))
    if not source_hashes or any(
        _SHA256_RE.fullmatch(value) is None for value in source_hashes
    ):
        raise MatchedReactionError(
            "manifest input_source_hashes must be valid and nonempty"
        )
    input_source_hashes_sha256 = _canonical_sha256(source_hashes)
    rows: list[dict[str, object]] = []
    for stage, frame in sorted(tables.items()):
        rows.append(
            {
                "experiment_id": "X-13",
                "builder_version": BUILDER_VERSION,
                "stage": stage,
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "schema_fingerprint": _canonical_sha256(
                    [
                        (column, str(frame[column].dtype))
                        for column in frame.columns
                    ]
                ),
                "semantic_rows_sha256": _semantic_rows_sha256(frame),
                "input_source_hash_count": len(source_hashes),
                "input_source_hashes_sha256": (
                    input_source_hashes_sha256
                ),
                "claim_boundary": spec.claim_boundary,
                "holdout_reaction_accessed": False,
            }
        )
    return pd.DataFrame(rows)


def build_matched_reaction_tables(
    treated_observations: pd.DataFrame,
    routine_control_observations: pd.DataFrame,
    *,
    spec: MatchedReactionSpecV1 | None = None,
) -> MatchedReactionTablesV1:
    """Build matched, game-first, SOURCE_TIME_ONLY X-13 result tables.

    Callers must supply already published development observations.  Routine
    control eligibility is an explicit upstream adjudication field: this
    function never guesses it from descriptions or factor names.
    """

    active_spec = spec or MatchedReactionSpecV1()
    treated = _validate_observations(
        treated_observations,
        label="treated",
        spec=active_spec,
    )
    controls = _validate_observations(
        routine_control_observations,
        label="control",
        spec=active_spec,
    )
    if treated["routine_control_eligible"].any():
        raise MatchedReactionError(
            "treated observations cannot be routine controls"
        )
    if not controls["routine_control_eligible"].all():
        raise MatchedReactionError(
            "control observations must be explicitly routine eligible"
        )
    input_source_hashes = tuple(
        sorted(
            {
                source_hash
                for frame in (treated, controls)
                for value in frame["source_hashes"]
                for source_hash in _hash_sequence(
                    value, field="input source_hashes"
                )
            }
        )
    )
    matches, audit = _match_rows(
        treated, controls, spec=active_spec
    )
    matches, audit, balance = _balance_matches(
        matches, audit, spec=active_spec
    )
    effects = _matched_observation_effects(matches, spec=active_spec)
    source_time_paths = _build_source_time_paths(
        effects, spec=active_spec
    )
    single_game = _build_single_game_results(
        effects, spec=active_spec
    )
    cross_game = _build_cross_game_results(
        single_game, spec=active_spec
    )
    venue_consistency = _build_venue_consistency(
        cross_game, spec=active_spec
    )
    tables = {
        "matched_controls": matches,
        "match_audit": audit,
        "match_balance": balance,
        "source_time_paths": source_time_paths,
        "single_game_results": single_game,
        "cross_game_results": cross_game,
        "venue_consistency": venue_consistency,
    }
    return MatchedReactionTablesV1(
        matched_controls=matches,
        match_audit=audit,
        match_balance=balance,
        source_time_paths=source_time_paths,
        single_game_results=single_game,
        cross_game_results=cross_game,
        venue_consistency=venue_consistency,
        manifest=_build_manifest(
            tables,
            spec=active_spec,
            input_source_hashes=input_source_hashes,
        ),
    )


__all__ = [
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "HORIZONS_SECONDS",
    "MATCH_FEATURES",
    "MatchedReactionError",
    "MatchedReactionSpecV1",
    "MatchedReactionTablesV1",
    "PublishedObservationInputsV1",
    "adapt_published_x13_factor_observations",
    "build_matched_reaction_tables",
]
