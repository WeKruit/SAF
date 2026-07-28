"""Locked NFL factor discovery, human review, and historical holdout gates.

The module operates only on already-verified, source-time observations.  It
does not download data, infer an executable quote, or turn a preliminary
association into a trading claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HORIZONS = (1, 2, 5, 10, 30, 60)
_REVIEW_DECISIONS = frozenset({"ACCEPT", "REDEFINE", "DATA_GAP", "REJECT"})
_OPS = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "in",
        "not_in",
        "is_null",
        "not_null",
    }
)
_ACTUAL_REQUIRED = (
    "game_id",
    "episode_id",
    "contract_id",
    "outcome",
    "signal_focal_team",
    "venue",
    "family",
    "contract_measure",
    "horizon_seconds",
    "pre_event_actual_trade",
    "first_post_event_trade",
    "signed_price_change",
    "maximum_excursion",
    "net_change_60s",
    "trade_count",
    "volume",
    "staleness_seconds",
    "validity_status",
    "actual_observation",
    "forward_filled",
    "orientation_audited",
    "order_ambiguous",
    "contaminated",
    "signal_candidate_eligible",
)
_BASELINE_GRAIN = (
    "game_id",
    "episode_id",
    "contract_id",
    "outcome",
    "venue",
    "family",
    "horizon_seconds",
)


class FactorLabError(ValueError):
    """A factor input, lock, or validation gate failed closed."""


@dataclass(frozen=True, slots=True)
class HumanFactorReviewV1:
    factor_id: str
    factor_version: str
    market_family: str
    horizon_seconds: int
    decision: str
    sports_rationale: str
    anomaly_case_ids: tuple[str, ...]
    priority: str
    reviewer: str
    reviewed_at_utc: str

    def __post_init__(self) -> None:
        for field in (
            "factor_id",
            "factor_version",
            "market_family",
            "sports_rationale",
            "priority",
            "reviewer",
            "reviewed_at_utc",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise FactorLabError(f"{field} must be non-empty text")
        if self.decision not in _REVIEW_DECISIONS:
            raise FactorLabError(
                "review decision must be ACCEPT, REDEFINE, DATA_GAP, or REJECT"
            )
        if self.horizon_seconds not in _HORIZONS:
            raise FactorLabError("review horizon is not registered")
        if (
            type(self.anomaly_case_ids) is not tuple
            or any(type(value) is not str or not value for value in self.anomaly_case_ids)
            or tuple(sorted(set(self.anomaly_case_ids))) != self.anomaly_case_ids
        ):
            raise FactorLabError(
                "anomaly_case_ids must be a sorted unique tuple of strings"
            )
        if not self.reviewed_at_utc.endswith("Z"):
            raise FactorLabError("reviewed_at_utc must be a UTC timestamp")


@dataclass(frozen=True, slots=True)
class ShortlistLockV1:
    manifest_path: Path
    shortlist_sha256: str
    candidate_keys: tuple[str, ...]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FactorLabError(f"{field} must be a canonical SHA-256")
    return value


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, label: str
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise FactorLabError(f"{label} must be a pandas DataFrame")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise FactorLabError(f"{label} missing columns: {', '.join(missing)}")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _normalise_registry(
    factor_registry: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(factor_registry, Sequence) or not factor_registry:
        raise FactorLabError("factor registry must be a non-empty sequence")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {
        "factor_id",
        "version",
        "sports_meaning",
        "predicate",
        "pit_fields",
        "focal_entity",
        "exclusions",
        "target",
        "support_kind",
        "market_families",
        "horizons_seconds",
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
        "claim_boundary",
    }
    for raw in factor_registry:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise FactorLabError("factor definition has invalid fields")
        value = _plain(dict(raw))
        factor_id = value["factor_id"]
        if type(factor_id) is not str or not factor_id or factor_id in seen:
            raise FactorLabError("factor_id must be unique non-empty text")
        seen.add(factor_id)
        if (
            type(value["version"]) is not str
            or not value["version"]
            or type(value["sports_meaning"]) is not str
            or not value["sports_meaning"]
        ):
            raise FactorLabError("factor version and sports meaning are required")
        if value["support_kind"] not in {
            "primary",
            "binary",
            "named",
            "sequence",
            "descriptive",
        }:
            raise FactorLabError("factor support_kind is invalid")
        horizons = value["horizons_seconds"]
        if (
            type(horizons) is not list
            or not horizons
            or any(type(item) is not int or item not in _HORIZONS for item in horizons)
            or horizons != sorted(set(horizons))
        ):
            raise FactorLabError("factor horizons must be registered source horizons")
        families = value["market_families"]
        if (
            type(families) is not list
            or not families
            or any(type(item) is not str or not item for item in families)
        ):
            raise FactorLabError("factor market_families are invalid")
        pit_fields = value["pit_fields"]
        if (
            type(pit_fields) is not list
            or not pit_fields
            or any(type(item) is not str or not item for item in pit_fields)
        ):
            raise FactorLabError("factor pit_fields are invalid")
        if not set(pit_fields) >= _predicate_fields(value["predicate"]):
            raise FactorLabError("predicate uses a field not declared as PIT")
        if not set(pit_fields) >= _predicate_fields(
            value["exclusions"], allow_empty=True
        ):
            raise FactorLabError("exclusion uses a field not declared as PIT")
        for field in (
            "minimum_episodes",
            "minimum_games",
            "minimum_episodes_per_venue",
        ):
            minimum = 0 if field == "minimum_episodes_per_venue" else 1
            if (
                type(value[field]) is not int
                or value[field] < minimum
            ):
                raise FactorLabError(f"{field} is invalid")
        if (
            type(value["claim_boundary"]) is not str
            or value["claim_boundary"] != "PRELIMINARY_SOURCE_TIME_ONLY"
        ):
            raise FactorLabError("factor claim boundary is invalid")
        result.append(value)
    return tuple(sorted(result, key=lambda item: item["factor_id"]))


def _predicate_clauses(
    predicate: object, *, allow_empty: bool = False
) -> list[Mapping[str, object]]:
    if not isinstance(predicate, list) or (
        not predicate and not allow_empty
    ):
        raise FactorLabError("predicate must be a non-empty clause list")
    return predicate


def _predicate_fields(
    predicate: object, *, allow_empty: bool = False
) -> set[str]:
    clauses = _predicate_clauses(predicate, allow_empty=allow_empty)
    fields: set[str] = set()
    for clause in clauses:
        if (
            not isinstance(clause, Mapping)
            or set(clause) != {"field", "operator", "value"}
            or type(clause["field"]) is not str
            or not clause["field"]
            or clause["operator"] not in _OPS
        ):
            raise FactorLabError("predicate clause is invalid")
        if clause["operator"] in {"in", "not_in"} and not isinstance(
            clause["value"], list
        ):
            raise FactorLabError("in predicates require a list value")
        if (
            clause["operator"] in {"is_null", "not_null"}
            and clause["value"] is not None
        ):
            raise FactorLabError("null predicates require a null value")
        fields.add(clause["field"])
    return fields


def _predicate_mask(
    frame: pd.DataFrame,
    predicate: object,
    *,
    allow_empty: bool = False,
) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for clause in _predicate_clauses(predicate, allow_empty=allow_empty):
        field = clause["field"]
        if field not in frame:
            raise FactorLabError(f"factor PIT field is unavailable: {field}")
        operation = clause["operator"]
        value = clause["value"]
        series = frame[field]
        if operation == "eq":
            observed = series.eq(value)
        elif operation == "ne":
            observed = series.ne(value)
        elif operation == "gt":
            observed = series.gt(value)
        elif operation == "ge":
            observed = series.ge(value)
        elif operation == "lt":
            observed = series.lt(value)
        elif operation == "le":
            observed = series.le(value)
        elif operation == "in":
            observed = series.isin(value)
        elif operation == "not_in":
            observed = ~series.isin(value)
        elif operation == "is_null":
            observed = series.isna()
        elif operation == "not_null":
            observed = series.notna()
        else:  # guarded by _normalise_registry
            raise FactorLabError("unsupported predicate operation")
        mask &= observed.fillna(False)
    return mask


def _target_mask(frame: pd.DataFrame) -> pd.Series:
    family = frame["family"].astype(str).str.lower()
    outcome = frame["outcome"].astype(str)
    focal = frame["signal_focal_team"].astype(str)
    winner_or_spread = family.isin(
        {"moneyline", "winner", "spread", "quarter", "half", "period"}
    )
    total = family.eq("total")
    team_total = family.eq("team_total")
    player_prop = family.isin(
        {
            "player_passing",
            "player_rushing",
            "player_receiving",
            "first_td",
            "any_td",
            "field_goal",
            "touchdown",
        }
    )
    return (
        (winner_or_spread & outcome.eq(focal))
        | (total & outcome.str.casefold().eq("over"))
        | (team_total & outcome.str.casefold().eq("over"))
        | player_prop
    )


def _add_leave_one_game_out_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["diagnostic_baseline_prediction"] = np.nan
    moneyline = result["family"].astype(str).str.lower().isin(
        {"moneyline", "winner"}
    )
    if "realized_delta_wp_focal" not in result:
        result["market_residual"] = np.nan
        return result
    groups = result.loc[moneyline].groupby(
        ["venue", "horizon_seconds"], sort=True, dropna=False
    )
    for _, group in groups:
        games = sorted(group["game_id"].unique())
        for game_id in games:
            test = group[group["game_id"].eq(game_id)]
            train = group[~group["game_id"].eq(game_id)].dropna(
                subset=[
                    "realized_delta_wp_focal",
                    "pre_event_actual_trade",
                    "signed_price_change",
                ]
            )
            if train["game_id"].nunique() < 3:
                continue
            design = np.column_stack(
                (
                    np.ones(len(train)),
                    train["realized_delta_wp_focal"].to_numpy(dtype=float),
                    train["pre_event_actual_trade"].to_numpy(dtype=float),
                )
            )
            coefficients, *_ = np.linalg.lstsq(
                design,
                train["signed_price_change"].to_numpy(dtype=float),
                rcond=None,
            )
            test_design = np.column_stack(
                (
                    np.ones(len(test)),
                    test["realized_delta_wp_focal"].to_numpy(dtype=float),
                    test["pre_event_actual_trade"].to_numpy(dtype=float),
                )
            )
            result.loc[test.index, "diagnostic_baseline_prediction"] = (
                test_design @ coefficients
            )
    result["market_residual"] = (
        result["signed_price_change"]
        - result["diagnostic_baseline_prediction"]
    )
    return result


def build_factor_observations(
    sports_feature_view: pd.DataFrame,
    market_observations: pd.DataFrame,
    factor_registry: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Join registered PIT football features to actual source-time trades.

    Every returned row has a real pre/post trade and a registered market
    orientation.  Missing horizons are absent, never forward-filled.
    """

    registry = _normalise_registry(factor_registry)
    _require_columns(
        sports_feature_view,
        {"game_id", "play_id", "episode_id", "source_time_utc"},
        label="sports feature view",
    )
    _require_columns(
        market_observations, _ACTUAL_REQUIRED, label="market observations"
    )
    features = sports_feature_view.copy(deep=True)
    if "event_eligible" not in features:
        raise FactorLabError("sports feature view has no event eligibility gate")
    features = features[features["event_eligible"].eq(True)].copy()
    markets = market_observations.copy(deep=True)
    eligible_market = (
        markets["validity_status"].eq("OBSERVED")
        & markets["actual_observation"].eq(True)
        & markets["forward_filled"].eq(False)
        & markets["orientation_audited"].eq(True)
        & markets["order_ambiguous"].eq(False)
        & markets["contaminated"].eq(False)
        & markets["signal_candidate_eligible"].eq(True)
        & pd.to_numeric(markets["staleness_seconds"], errors="coerce").between(
            0, 60, inclusive="both"
        )
        & markets["pre_event_actual_trade"].notna()
        & markets["first_post_event_trade"].notna()
        & markets["signed_price_change"].notna()
    )
    markets = markets[eligible_market].copy()
    markets = markets[_target_mask(markets)].copy()
    baseline_features = features.sort_values(
        ["game_id", "episode_id", "source_time_utc", "play_id"],
        kind="mergesort",
    ).drop_duplicates(["game_id", "episode_id"], keep="first")
    baseline_columns = ["game_id", "episode_id"]
    if "realized_delta_wp_focal" in baseline_features:
        baseline_columns.append("realized_delta_wp_focal")
    baseline_input = markets.merge(
        baseline_features[baseline_columns],
        on=["game_id", "episode_id"],
        how="inner",
        validate="many_to_one",
    )
    if baseline_input.duplicated(list(_BASELINE_GRAIN)).any():
        raise FactorLabError(
            "moneyline baseline input is not unique at its observation grain"
        )
    baseline = _add_leave_one_game_out_baseline(baseline_input)
    baseline_metrics = baseline[
        [
            *_BASELINE_GRAIN,
            "diagnostic_baseline_prediction",
            "market_residual",
        ]
    ]

    rows: list[pd.DataFrame] = []
    for definition in registry:
        feature_fields = set(features.columns)
        feature_clauses = [
            clause
            for clause in definition["predicate"]
            if clause["field"] in feature_fields
        ]
        joined_clauses = [
            clause
            for clause in definition["predicate"]
            if clause["field"] not in feature_fields
        ]
        factor_features = features.copy()
        if feature_clauses:
            factor_features = factor_features[
                _predicate_mask(factor_features, feature_clauses)
            ].copy()
        if factor_features.empty:
            continue
        factor_features = factor_features.sort_values(
            ["game_id", "episode_id", "source_time_utc", "play_id"],
            kind="mergesort",
        ).drop_duplicates(["game_id", "episode_id"], keep="first")
        selected = markets[
            markets["family"].isin(definition["market_families"])
            & markets["horizon_seconds"].isin(definition["horizons_seconds"])
        ]
        joined = selected.merge(
            factor_features,
            on=["game_id", "episode_id"],
            how="inner",
            validate="many_to_one",
            suffixes=("", "_feature"),
        )
        if joined.empty:
            continue
        if joined_clauses:
            joined = joined[
                _predicate_mask(joined, joined_clauses)
            ].copy()
        if definition["exclusions"]:
            unavailable = {
                clause["field"]
                for clause in definition["exclusions"]
                if clause["field"] not in joined
            }
            if unavailable:
                raise FactorLabError(
                    "factor exclusion field is unavailable: "
                    + ", ".join(sorted(unavailable))
                )
            joined = joined[
                ~_predicate_mask(
                    joined,
                    definition["exclusions"],
                    allow_empty=True,
                )
            ].copy()
        if joined.empty:
            continue
        joined["factor_id"] = definition["factor_id"]
        joined["factor_version"] = definition["version"]
        joined["factor_target"] = definition["target"]
        joined["support_kind"] = definition["support_kind"]
        rows.append(joined)
    if not rows:
        return pd.DataFrame(
            columns=[
                *_ACTUAL_REQUIRED,
                "play_id",
                "source_time_utc",
                "factor_id",
                "factor_version",
                "factor_target",
                "support_kind",
                "diagnostic_baseline_prediction",
                "market_residual",
            ]
        )
    combined = pd.concat(rows, ignore_index=True)
    duplicates = [
        "factor_id",
        "game_id",
        "episode_id",
        "contract_id",
        "outcome",
        "venue",
        "horizon_seconds",
    ]
    if combined.duplicated(duplicates).any():
        raise FactorLabError("factor observations are not unique at their grain")
    combined = combined.merge(
        baseline_metrics,
        on=list(_BASELINE_GRAIN),
        how="left",
        validate="many_to_one",
    )
    return combined.sort_values(duplicates, kind="mergesort").reset_index(drop=True)


def _cluster_statistics(
    frame: pd.DataFrame,
    *,
    value: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    values = pd.to_numeric(frame[value], errors="coerce")
    usable = frame.loc[values.notna(), ["game_id"]].copy()
    usable["effect"] = values[values.notna()].to_numpy(dtype=float)
    games = tuple(sorted(usable["game_id"].unique()))
    if not games:
        return {
            "point_estimate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "raw_p_value": 1.0,
            "loo_sign_stability_rate": 0.0,
            "maximum_single_game_contribution": 1.0,
            "bootstrap_samples_requested": samples,
            "bootstrap_samples_valid": 0,
            "loo_estimates": (),
        }
    point = float(usable["effect"].mean())
    grouped = usable.groupby("game_id", sort=True)["effect"].agg(
        ["sum", "count"]
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(games),
        size=(samples, len(games)),
        endpoint=False,
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    raw_p = min(
        1.0,
        2.0
        * min(
            (float(np.count_nonzero(bootstrap <= 0)) + 1.0) / (samples + 1.0),
            (float(np.count_nonzero(bootstrap >= 0)) + 1.0) / (samples + 1.0),
        ),
    )
    loo: list[float] = []
    for game in games:
        rest = usable.loc[~usable["game_id"].eq(game), "effect"]
        if not rest.empty:
            loo.append(float(rest.mean()))
    sign = np.sign(point)
    stability = (
        float(np.mean(np.sign(np.asarray(loo)) == sign)) if loo and sign else 0.0
    )
    contributions = usable.groupby("game_id", sort=True)["effect"].sum().abs()
    denominator = float(contributions.sum())
    maximum_contribution = (
        float(contributions.max() / denominator) if denominator > 0 else 1.0
    )
    return {
        "point_estimate": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "raw_p_value": raw_p,
        "loo_sign_stability_rate": stability,
        "maximum_single_game_contribution": maximum_contribution,
        "bootstrap_samples_requested": samples,
        "bootstrap_samples_valid": samples,
        "loo_estimates": tuple(loo),
    }


def _bh(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    count = len(raw)
    order = np.argsort(raw, kind="mergesort")
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        source_index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, raw[source_index] * count / rank)
        adjusted[source_index] = min(1.0, running)
    return adjusted


def _support_pass(
    *,
    definition: Mapping[str, object],
    episodes: int,
    games: int,
    venue_counts: Mapping[str, int],
) -> bool:
    per_venue = int(definition["minimum_episodes_per_venue"])
    return (
        episodes >= int(definition["minimum_episodes"])
        and games >= int(definition["minimum_games"])
        and (
            per_venue == 0
            or all(
                venue_counts.get(venue, 0) >= per_venue
                for venue in ("kalshi", "polymarket")
            )
        )
    )


def _empty_factor_result(
    *,
    definition: Mapping[str, object],
    family: str,
    horizon: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    return {
        "factor_id": definition["factor_id"],
        "factor_version": definition["version"],
        "market_family": family,
        "horizon_seconds": horizon,
        "target": definition["target"],
        "support_kind": definition["support_kind"],
        "episode_count": 0,
        "game_count": 0,
        "observation_count": 0,
        "per_game_result_count": 0,
        "per_game_effects_json": "[]",
        "kalshi_episode_count": 0,
        "polymarket_episode_count": 0,
        "kalshi_effect": math.nan,
        "polymarket_effect": math.nan,
        "venue_sign_consistent": False,
        "support_gate_pass": False,
        "point_estimate": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "raw_p_value": 1.0,
        "loo_sign_stability_rate": 0.0,
        "maximum_single_game_contribution": 1.0,
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_valid": 0,
        "loo_estimates": (),
        "residual_point_estimate": math.nan,
        "residual_ci_low": math.nan,
        "residual_ci_high": math.nan,
        "residual_raw_p_value": math.nan,
    }


def run_factor_discovery(
    observations: pd.DataFrame,
    *,
    factor_registry: Sequence[Mapping[str, object]],
    bootstrap_samples: int = 10_000,
    seed: int = 20260725,
) -> pd.DataFrame:
    """Compute per-factor clustered discovery inference.

    The output can only become ``AWAITING_EXPERT_REVIEW``.  It is deliberately
    incapable of emitting ``SIGNAL_CANDIDATE`` before the human lock and
    independent holdout.
    """

    if bootstrap_samples != 10_000:
        raise FactorLabError("formal factor inference requires 10000 bootstraps")
    registry = {
        value["factor_id"]: value for value in _normalise_registry(factor_registry)
    }
    _require_columns(
        observations,
        {
            "factor_id",
            "factor_version",
            "support_kind",
            "game_id",
            "episode_id",
            "venue",
            "family",
            "horizon_seconds",
            "signed_price_change",
            "actual_observation",
            "forward_filled",
            "order_ambiguous",
            "contaminated",
        },
        label="factor observations",
    )
    if (
        (~observations["actual_observation"].eq(True)).any()
        or observations["forward_filled"].eq(True).any()
        or observations["order_ambiguous"].eq(True).any()
        or observations["contaminated"].eq(True).any()
    ):
        raise FactorLabError("discovery contains an ineligible market observation")
    unknown = set(observations["factor_id"]).difference(registry)
    if unknown:
        raise FactorLabError(f"unregistered factor observations: {sorted(unknown)}")
    rows: list[dict[str, object]] = []
    group_keys = ["factor_id", "factor_version", "family", "horizon_seconds"]
    for group_index, (key, frame) in enumerate(
        observations.groupby(group_keys, sort=True, dropna=False)
    ):
        factor_id, version, family, horizon = key
        definition = registry[factor_id]
        if version != definition["version"]:
            raise FactorLabError("factor observation version differs from registry")
        venue_effects = frame.groupby("venue", sort=True)[
            "signed_price_change"
        ].mean()
        nonzero = venue_effects[np.abs(venue_effects) > 0]
        venue_consistent = (
            {"kalshi", "polymarket"} <= set(venue_effects.index)
            and len(nonzero) >= 2
            and len(set(np.sign(nonzero))) == 1
        )
        venue_counts = {
            str(venue): int(
                child[["game_id", "episode_id"]].drop_duplicates().shape[0]
            )
            for venue, child in frame.groupby("venue", sort=True)
        }
        episode_effects = (
            frame.groupby(
                ["game_id", "episode_id"],
                sort=True,
                as_index=False,
            )["signed_price_change"]
            .mean()
        )
        stats = _cluster_statistics(
            episode_effects,
            value="signed_price_change",
            samples=bootstrap_samples,
            seed=seed + group_index,
        )
        per_game = (
            episode_effects.groupby("game_id", sort=True)[
                "signed_price_change"
            ]
            .agg(["mean", "count"])
            .reset_index()
        )
        per_game_effects = json.dumps(
            [
                {
                    "game_id": str(row.game_id),
                    "episode_count": int(row.count),
                    "point_estimate": float(row.mean),
                }
                for row in per_game.itertuples(index=False)
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        residual_stats: dict[str, object] = {}
        if "market_residual" in frame and frame["market_residual"].notna().any():
            residuals = (
                frame.dropna(subset=["market_residual"])
                .groupby(
                    ["game_id", "episode_id"],
                    sort=True,
                    as_index=False,
                )["market_residual"]
                .mean()
            )
            child = _cluster_statistics(
                residuals,
                value="market_residual",
                samples=bootstrap_samples,
                seed=seed + 100_000 + group_index,
            )
            residual_stats = {
                "residual_point_estimate": child["point_estimate"],
                "residual_ci_low": child["ci_low"],
                "residual_ci_high": child["ci_high"],
                "residual_raw_p_value": child["raw_p_value"],
            }
        else:
            residual_stats = {
                "residual_point_estimate": math.nan,
                "residual_ci_low": math.nan,
                "residual_ci_high": math.nan,
                "residual_raw_p_value": math.nan,
            }
        episodes = int(
            frame[["game_id", "episode_id"]].drop_duplicates().shape[0]
        )
        games = int(frame["game_id"].nunique())
        support = _support_pass(
            definition=definition,
            episodes=episodes,
            games=games,
            venue_counts=venue_counts,
        )
        rows.append(
            {
                "factor_id": factor_id,
                "factor_version": version,
                "market_family": family,
                "horizon_seconds": int(horizon),
                "target": definition["target"],
                "support_kind": definition["support_kind"],
                "episode_count": episodes,
                "game_count": games,
                "observation_count": len(frame),
                "per_game_result_count": len(per_game),
                "per_game_effects_json": per_game_effects,
                "kalshi_episode_count": venue_counts.get("kalshi", 0),
                "polymarket_episode_count": venue_counts.get("polymarket", 0),
                "kalshi_effect": float(venue_effects.get("kalshi", math.nan)),
                "polymarket_effect": float(
                    venue_effects.get("polymarket", math.nan)
                ),
                "venue_sign_consistent": bool(venue_consistent),
                "support_gate_pass": bool(support),
                **stats,
                **residual_stats,
            }
        )
    observed_by_key = {
        (
            row["factor_id"],
            row["market_family"],
            row["horizon_seconds"],
        ): row
        for row in rows
    }
    grid: list[dict[str, object]] = []
    for definition in registry.values():
        for family in definition["market_families"]:
            for horizon in definition["horizons_seconds"]:
                key = (definition["factor_id"], family, horizon)
                grid.append(
                    observed_by_key.get(key)
                    or _empty_factor_result(
                        definition=definition,
                        family=family,
                        horizon=horizon,
                        bootstrap_samples=bootstrap_samples,
                    )
                )
    result = pd.DataFrame(grid)
    result["bh_q_value"] = _bh(result["raw_p_value"])
    inferential = (
        result["support_gate_pass"].eq(True)
        & result["venue_sign_consistent"].eq(True)
        & (
            result["ci_low"].gt(0)
            | result["ci_high"].lt(0)
        )
        & result["bh_q_value"].le(0.10)
        & result["loo_sign_stability_rate"].ge(0.80)
        & result["maximum_single_game_contribution"].le(0.25)
        & result["bootstrap_samples_valid"].eq(
            result["bootstrap_samples_requested"]
        )
        & ~result["support_kind"].eq("binary")
    )
    data_gap = result["observation_count"].eq(0)
    conditional_binary = (
        result["support_kind"].eq("binary") & ~data_gap
    )
    insufficient = (
        ~result["support_gate_pass"].eq(True) & ~data_gap
    )
    result["estimand_type"] = np.where(
        result["support_kind"].eq("binary"),
        "CONDITIONAL_MEAN_NO_COMPLEMENT",
        "CONDITIONAL_MEAN_SIGNED_CHANGE",
    )
    result["support_status"] = np.select(
        [data_gap, result["support_gate_pass"].eq(True)],
        ["DATA_GAP", "PASS"],
        default="INSUFFICIENT_SUPPORT",
    )
    result["inferential_eligible"] = inferential
    result["discovery_status"] = np.select(
        [data_gap, conditional_binary, insufficient, inferential],
        [
            "DATA_GAP",
            "DESCRIPTIVE_CONDITIONAL_MEAN",
            "INSUFFICIENT_SUPPORT",
            "AWAITING_EXPERT_REVIEW",
        ],
        default="DESCRIPTIVE",
    )
    result["claim_boundary"] = (
        "PRELIMINARY_SOURCE_TIME_ONLY; no causality, latency, execution, "
        "tradeability, OFI, depth, or alpha claim"
    )
    return result.sort_values(
        ["factor_id", "market_family", "horizon_seconds"],
        kind="mergesort",
    ).reset_index(drop=True)


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FactorLabError("immutable shortlist artifact conflicts")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FactorLabError("immutable shortlist artifact conflicts")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def freeze_factor_shortlist(
    *,
    discovery_results: pd.DataFrame,
    factor_registry: Sequence[Mapping[str, object]],
    reviews: Sequence[HumanFactorReviewV1],
    source_hashes: Sequence[str],
    output_root: str | Path,
) -> ShortlistLockV1:
    registry = {
        value["factor_id"]: value for value in _normalise_registry(factor_registry)
    }
    _require_columns(
        discovery_results,
        {
            "factor_id",
            "factor_version",
            "market_family",
            "horizon_seconds",
            "target",
            "discovery_status",
            "point_estimate",
            "ci_low",
            "ci_high",
            "bh_q_value",
            "loo_sign_stability_rate",
            "maximum_single_game_contribution",
            "venue_sign_consistent",
            "support_gate_pass",
        },
        label="discovery results",
    )
    review_by_candidate: dict[
        tuple[str, str, int], HumanFactorReviewV1
    ] = {}
    for review in reviews:
        if not isinstance(review, HumanFactorReviewV1):
            raise FactorLabError("expert reviews must be HumanFactorReviewV1")
        key = (
            review.factor_id,
            review.market_family,
            review.horizon_seconds,
        )
        if key in review_by_candidate:
            raise FactorLabError("candidate has duplicate expert reviews")
        review_by_candidate[key] = review
    candidates = discovery_results[
        discovery_results["discovery_status"].eq("AWAITING_EXPERT_REVIEW")
    ].copy()
    if candidates.empty:
        raise FactorLabError("no discovery result is eligible for expert review")
    accepted: list[tuple[str, str, int]] = []
    candidate_rows = candidates.sort_values(
        ["factor_id", "market_family", "horizon_seconds"],
        kind="mergesort",
    )
    for row in candidate_rows.itertuples(index=False):
        key = (
            str(row.factor_id),
            str(row.market_family),
            int(row.horizon_seconds),
        )
        review = review_by_candidate.get(key)
        if review is None:
            raise FactorLabError(
                "expert review is missing for "
                + f"{key[0]}|{key[1]}|h={key[2]}"
            )
        if (
            review.factor_version != registry[key[0]]["version"]
            or review.decision != "ACCEPT"
        ):
            continue
        accepted.append(key)
    if not accepted:
        raise FactorLabError("expert review accepted no eligible factors")
    hashes = sorted({_require_sha256(value, "source_hash") for value in source_hashes})
    accepted_factor_ids = sorted({key[0] for key in accepted})
    definitions = [registry[factor_id] for factor_id in accepted_factor_ids]
    accepted_key_frame = pd.DataFrame(
        accepted,
        columns=["factor_id", "market_family", "horizon_seconds"],
    )
    accepted_results = discovery_results.merge(
        accepted_key_frame,
        on=["factor_id", "market_family", "horizon_seconds"],
        how="inner",
        validate="one_to_one",
    ).sort_values(
        ["factor_id", "market_family", "horizon_seconds"],
        kind="mergesort",
    )
    snapshot_columns = [
        "factor_id",
        "factor_version",
        "market_family",
        "horizon_seconds",
        "target",
        "point_estimate",
        "ci_low",
        "ci_high",
        "bh_q_value",
        "loo_sign_stability_rate",
        "maximum_single_game_contribution",
        "venue_sign_consistent",
        "support_gate_pass",
    ]
    material = {
        "schema": "nfl_factor_shortlist_lock_v1",
        "experiment_id": "X-13",
        "factor_definitions": definitions,
        "locked_candidates": [
            {
                "factor_id": factor_id,
                "market_family": family,
                "horizon_seconds": horizon,
            }
            for factor_id, family, horizon in accepted
        ],
        "discovery_results": [
            _plain(row)
            for row in accepted_results[snapshot_columns].to_dict("records")
        ],
        "human_reviews": [
            _plain(asdict(review_by_candidate[key]))
            for key in accepted
        ],
        "source_hashes": hashes,
        "holdout_reaction_accessed": False,
        "claim_boundary": (
            "Shortlist only; independent historical holdout has not been read."
        ),
    }
    digest = _sha256(_canonical_bytes(material))
    manifest = {**material, "shortlist_sha256": digest}
    payload = _canonical_bytes(manifest)
    root = Path(output_root) / f"sha256-{digest.removeprefix('sha256:')}"
    path = root / f"{digest.removeprefix('sha256:')}.shortlist-lock.json"
    _atomic_publish(path, payload)
    return ShortlistLockV1(
        manifest_path=path,
        shortlist_sha256=digest,
        candidate_keys=tuple(
            f"{factor_id}|{family}|h={horizon}"
            for factor_id, family, horizon in accepted
        ),
    )


def verify_shortlist_lock(path: str | Path) -> ShortlistLockV1:
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise FactorLabError("shortlist lock is unreadable") from error
    if not isinstance(value, dict) or value.get("schema") != (
        "nfl_factor_shortlist_lock_v1"
    ):
        raise FactorLabError("shortlist lock schema is invalid")
    observed = value.get("shortlist_sha256")
    _require_sha256(observed, "shortlist_sha256")
    material = {key: child for key, child in value.items() if key != "shortlist_sha256"}
    expected = _sha256(_canonical_bytes(material))
    if expected != observed:
        raise FactorLabError("shortlist lock hash differs from its content")
    expected_parent = f"sha256-{observed.removeprefix('sha256:')}"
    expected_name = f"{observed.removeprefix('sha256:')}.shortlist-lock.json"
    if (
        manifest_path.parent.name != expected_parent
        or manifest_path.name != expected_name
    ):
        raise FactorLabError("shortlist lock path does not match its hash")
    definitions = value.get("factor_definitions")
    if not isinstance(definitions, list) or not definitions:
        raise FactorLabError("shortlist lock has no factor definitions")
    factor_ids = tuple(item["factor_id"] for item in definitions)
    if factor_ids != tuple(sorted(set(factor_ids))):
        raise FactorLabError("shortlist factor IDs are not sorted and unique")
    candidates = value.get("locked_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FactorLabError("shortlist lock has no exact candidate keys")
    candidate_keys = tuple(
        f"{item['factor_id']}|{item['market_family']}|h={item['horizon_seconds']}"
        for item in candidates
    )
    if candidate_keys != tuple(sorted(set(candidate_keys))):
        raise FactorLabError(
            "shortlist candidate keys are not sorted and unique"
        )
    return ShortlistLockV1(
        manifest_path=manifest_path,
        shortlist_sha256=observed,
        candidate_keys=candidate_keys,
    )


def run_locked_historical_holdout(
    *,
    shortlist_manifest: str | Path,
    holdout_observations: pd.DataFrame,
    factor_registry: Sequence[Mapping[str, object]],
    holdout_selection_id: str,
    holdout_game_ids: Sequence[str],
    discovery_game_ids: Sequence[str],
    bootstrap_samples: int = 10_000,
    seed: int = 20260725,
) -> pd.DataFrame:
    lock = verify_shortlist_lock(shortlist_manifest)
    _require_sha256(holdout_selection_id, "holdout_selection_id")
    payload = json.loads(lock.manifest_path.read_bytes())
    locked_definitions = payload["factor_definitions"]
    registry = {
        value["factor_id"]: value for value in _normalise_registry(factor_registry)
    }
    locked_factor_ids = [
        item["factor_id"] for item in locked_definitions
    ]
    current = [registry.get(factor_id) for factor_id in locked_factor_ids]
    if current != locked_definitions:
        raise FactorLabError("factor registry differs from the shortlist lock")
    holdout_games = set(holdout_game_ids)
    discovery_games = set(discovery_game_ids)
    if (
        len(holdout_games) != 20
        or len(discovery_games) != 20
        or holdout_games & discovery_games
    ):
        raise FactorLabError("historical game split is not the frozen 20/20")
    observed_games = set(holdout_observations["game_id"])
    if not observed_games or not observed_games <= holdout_games:
        raise FactorLabError(
            "holdout observations are outside the frozen holdout"
        )
    if observed_games & discovery_games:
        raise FactorLabError("discovery game leaked into holdout")
    locked_candidates = payload["locked_candidates"]
    key_columns = ["factor_id", "family", "horizon_seconds"]
    allowed_keys = {
        (
            item["factor_id"],
            item["market_family"],
            item["horizon_seconds"],
        )
        for item in locked_candidates
    }
    observed_keys = set(
        holdout_observations[key_columns].itertuples(
            index=False, name=None
        )
    )
    if observed_keys != allowed_keys:
        raise FactorLabError(
            "holdout observations differ from exact shortlist candidates"
        )
    holdout = run_factor_discovery(
        holdout_observations,
        factor_registry=locked_definitions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    allowed_frame = pd.DataFrame(
        sorted(allowed_keys),
        columns=["factor_id", "market_family", "horizon_seconds"],
    )
    holdout = holdout.merge(
        allowed_frame,
        on=["factor_id", "market_family", "horizon_seconds"],
        how="inner",
        validate="one_to_one",
    )
    discovery_key = [
        "factor_id",
        "factor_version",
        "market_family",
        "horizon_seconds",
    ]
    discovery_snapshot = pd.DataFrame(payload["discovery_results"])[
        discovery_key + ["point_estimate"]
    ].rename(
        columns={"point_estimate": "discovery_point_estimate"}
    )
    merged = holdout.merge(
        discovery_snapshot,
        on=discovery_key,
        how="left",
        validate="one_to_one",
    )
    if merged["discovery_point_estimate"].isna().any():
        raise FactorLabError("shortlisted discovery result is missing")
    same_direction = np.sign(merged["point_estimate"]).eq(
        np.sign(merged["discovery_point_estimate"])
    )
    ci_excludes_zero = merged["ci_low"].gt(0) | merged["ci_high"].lt(0)
    retained = merged["point_estimate"].abs().ge(
        merged["discovery_point_estimate"].abs() * 0.50
    )
    replicated = (
        same_direction
        & ci_excludes_zero
        & retained
        & merged["venue_sign_consistent"].eq(True)
        & merged["support_gate_pass"].eq(True)
        & merged["maximum_single_game_contribution"].le(0.25)
    )
    insufficient = ~merged["support_gate_pass"].eq(True)
    merged["holdout_status"] = np.select(
        [replicated, insufficient],
        ["SIGNAL_CANDIDATE", "INSUFFICIENT_SUPPORT"],
        default="NOT_REPLICATED",
    )
    merged["shortlist_sha256"] = lock.shortlist_sha256
    merged["holdout_selection_id"] = holdout_selection_id
    merged["claim_boundary"] = (
        "Replicated historical association only; no causality, live latency, "
        "execution, or tradeability claim."
    )
    return merged


__all__ = [
    "FactorLabError",
    "HumanFactorReviewV1",
    "ShortlistLockV1",
    "build_factor_observations",
    "freeze_factor_shortlist",
    "run_factor_discovery",
    "run_locked_historical_holdout",
    "verify_shortlist_lock",
]
