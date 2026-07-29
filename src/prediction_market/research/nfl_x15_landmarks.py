"""Pure construction of the NFL VenueReactionPanelV3.

The builder consumes verified sports facts, Stage A observations, actual market
trades, contract/rule metadata, and explicit continuity evidence.  It does not
read artifacts, train models, infer clean windows, or treat a derived away
complement as an observed home-outcome trade.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


LANDMARK_SECONDS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
ENDPOINT_SECONDS: Final[tuple[int, ...]] = tuple(range(5, 61, 5))
MAX_STALENESS_SECONDS: Final[float] = 3.0
SCHEMA_VERSION: Final[str] = "VenueReactionPanelV3"
CLAIM_BOUNDARY: Final[str] = (
    "SOURCE_TIME_ASSOCIATION_ACTUAL_HOME_TRADES_ONLY_NOT_EXECUTION"
)

_FACT_IDENTITY = ("game_id", "atomic_information_episode_id")
_PANEL_GRAIN = (
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
)
_FACT_REQUIRED = {
    *_FACT_IDENTITY,
    "source_interval_start",
    "source_interval_end",
    "source_resolution",
    "stage_b_information_event_eligible",
    "home_team",
    "away_team",
}
_REFERENCE_REQUIRED = {
    *_FACT_IDENTITY,
    "support_status",
    "known_at",
    "p_before_home",
    "p_after_home",
    "reference_delta_home",
}
_MARKET_REQUIRED = {
    "trade_id",
    "game_id",
    "venue",
    "contract_id",
    "source_time_utc",
    "price",
    "size",
    "kind",
    "provenance",
}
_CONTRACT_REQUIRED = {
    "game_id",
    "venue",
    "contract_id",
    "outcome_team",
    "home_team",
    "market_family",
    "contract_role",
    "tick_rule_id",
}
_RULE_REQUIRED = {
    "venue",
    "tick_rule_id",
    "effective_start_utc",
    "effective_end_utc",
    "tick_size",
}
_CONTINUITY_REQUIRED = {
    *_FACT_IDENTITY,
    "next_salient_event_time_utc",
    "suspension_time_utc",
    "game_end_time_utc",
    "continuity_gap_time_utc",
    "continuity_verified_until_utc",
}
_NOISE_REQUIRED = {
    "venue",
    "landmark_seconds",
    "endpoint_seconds",
    "matched_control_p95",
}
_FACT_FEATURE_COLUMNS = (
    "source_resolution",
    "game_seconds_remaining",
    "score_margin_home",
    "possession_is_home",
    "down",
    "distance",
    "yardline_100",
    "primary_action",
    "outcome_tags",
    "yards_gained",
    "return_yards",
    "actor_is_home",
    "beneficiary_is_home",
)
_CONTINUITY_REASONS = (
    ("next_salient_event_time_utc", "NEXT_SALIENT_EVENT_BEFORE_H"),
    ("suspension_time_utc", "SUSPENSION_BEFORE_H"),
    ("game_end_time_utc", "GAME_END_BEFORE_H"),
    ("continuity_gap_time_utc", "CONTINUITY_GAP_BEFORE_H"),
)


class VenueReactionPanelError(ValueError):
    """Inputs cannot support a deterministic VenueReactionPanelV3."""


@dataclass(frozen=True, slots=True)
class VenueReactionPanelV3:
    """Primary home-outcome rows and their two audit tables."""

    panel: pd.DataFrame
    attrition: pd.DataFrame
    complement_diagnostics: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _Mark:
    status: str
    trade_id: str | None
    source_time: pd.Timestamp | None
    price: float
    staleness_seconds: float

    @property
    def observed(self) -> bool:
        return self.status == "OBSERVED"


def _require_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    required: set[str],
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise VenueReactionPanelError(f"{label} must be a DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise VenueReactionPanelError(f"{label} missing columns: {missing}")
    if frame.empty and not allow_empty:
        raise VenueReactionPanelError(f"{label} must be nonempty")
    return frame.copy(deep=True)


def _strings(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise VenueReactionPanelError(f"{label}.{column} must be nonempty")
        frame[column] = values.astype(str)


def _utc(
    values: pd.Series,
    *,
    label: str,
    nullable: bool,
) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    supplied = values.notna()
    if ((parsed.isna() & supplied).any()) or (not nullable and parsed.isna().any()):
        raise VenueReactionPanelError(f"{label} must contain UTC timestamps")
    return parsed


def _strict_bool(frame: pd.DataFrame, column: str, *, label: str) -> None:
    if not frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise VenueReactionPanelError(f"{label}.{column} must contain booleans")
    frame[column] = frame[column].astype(bool)


def _finite_numeric(
    values: pd.Series,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise VenueReactionPanelError(f"{label} must be finite")
    if lower is not None and numeric.lt(lower).any():
        raise VenueReactionPanelError(f"{label} is below its lower bound")
    if upper is not None and numeric.gt(upper).any():
        raise VenueReactionPanelError(f"{label} is above its upper bound")
    return numeric


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, set):
        return [_canonical_value(child) for child in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return str(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_facts(frame: pd.DataFrame) -> pd.DataFrame:
    facts = _require_frame(frame, label="episode_facts", required=_FACT_REQUIRED)
    _strings(
        facts,
        (*_FACT_IDENTITY, "source_resolution", "home_team", "away_team"),
        label="episode_facts",
    )
    if facts.duplicated(list(_FACT_IDENTITY)).any():
        raise VenueReactionPanelError("episode_facts identity is not unique")
    _strict_bool(
        facts,
        "stage_b_information_event_eligible",
        label="episode_facts",
    )
    facts["source_interval_start"] = _utc(
        facts["source_interval_start"],
        label="episode_facts.source_interval_start",
        nullable=False,
    )
    facts["source_interval_end"] = _utc(
        facts["source_interval_end"],
        label="episode_facts.source_interval_end",
        nullable=False,
    )
    if (facts["source_interval_end"] <= facts["source_interval_start"]).any():
        raise VenueReactionPanelError("episode source interval must be nonempty")
    if facts["home_team"].eq(facts["away_team"]).any():
        raise VenueReactionPanelError("episode home and away teams must differ")
    return facts


def _validate_references(frame: pd.DataFrame) -> pd.DataFrame:
    references = _require_frame(
        frame,
        label="stage_a_references",
        required=_REFERENCE_REQUIRED,
        allow_empty=True,
    )
    if references.empty:
        return references
    _strings(
        references,
        (*_FACT_IDENTITY, "support_status"),
        label="stage_a_references",
    )
    if references.duplicated(list(_FACT_IDENTITY)).any():
        raise VenueReactionPanelError("stage_a_references identity is not unique")
    references["known_at"] = _utc(
        references["known_at"],
        label="stage_a_references.known_at",
        nullable=False,
    )
    for column in ("p_before_home", "p_after_home"):
        references[column] = _finite_numeric(
            references[column],
            label=f"stage_a_references.{column}",
            lower=0,
            upper=1,
        )
    references["reference_delta_home"] = _finite_numeric(
        references["reference_delta_home"],
        label="stage_a_references.reference_delta_home",
    )
    return references


def _validate_market(frame: pd.DataFrame) -> pd.DataFrame:
    market = _require_frame(frame, label="market_rows", required=_MARKET_REQUIRED)
    _strings(
        market,
        (
            "trade_id",
            "game_id",
            "venue",
            "contract_id",
            "kind",
            "provenance",
        ),
        label="market_rows",
    )
    if market["trade_id"].duplicated().any():
        raise VenueReactionPanelError("market_rows.trade_id must be unique")
    market["source_time_utc"] = _utc(
        market["source_time_utc"],
        label="market_rows.source_time_utc",
        nullable=False,
    )
    market["price"] = _finite_numeric(
        market["price"], label="market_rows.price", lower=0, upper=1
    )
    market["size"] = _finite_numeric(
        market["size"], label="market_rows.size", lower=0
    )
    return market


def _validate_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    contracts = _require_frame(
        frame, label="contract_metadata", required=_CONTRACT_REQUIRED
    )
    _strings(
        contracts,
        tuple(sorted(_CONTRACT_REQUIRED)),
        label="contract_metadata",
    )
    identity = ["game_id", "venue", "contract_id"]
    if contracts.duplicated(identity).any():
        raise VenueReactionPanelError("contract_metadata identity is not unique")
    roles = {"ACTUAL_HOME_OUTCOME", "DERIVED_AWAY_COMPLEMENT"}
    if not set(contracts["contract_role"]).issubset(roles):
        raise VenueReactionPanelError("contract_metadata has an unknown contract_role")
    if not contracts["market_family"].eq("moneyline").all():
        raise VenueReactionPanelError("contract_metadata must be moneyline")
    home = contracts["contract_role"].eq("ACTUAL_HOME_OUTCOME")
    if not contracts.loc[home, "outcome_team"].eq(
        contracts.loc[home, "home_team"]
    ).all():
        raise VenueReactionPanelError("actual home contract orientation is invalid")
    return contracts


def _validate_rules(frame: pd.DataFrame) -> pd.DataFrame:
    rules = _require_frame(frame, label="tick_rules", required=_RULE_REQUIRED)
    _strings(rules, ("venue", "tick_rule_id"), label="tick_rules")
    rules["effective_start_utc"] = _utc(
        rules["effective_start_utc"],
        label="tick_rules.effective_start_utc",
        nullable=False,
    )
    rules["effective_end_utc"] = _utc(
        rules["effective_end_utc"],
        label="tick_rules.effective_end_utc",
        nullable=False,
    )
    if (rules["effective_end_utc"] <= rules["effective_start_utc"]).any():
        raise VenueReactionPanelError("tick rule interval must be nonempty")
    rules["tick_size"] = _finite_numeric(
        rules["tick_size"], label="tick_rules.tick_size", lower=np.nextafter(0, 1)
    )
    identity = ["venue", "tick_rule_id", "effective_start_utc"]
    if rules.duplicated(identity).any():
        raise VenueReactionPanelError("tick rule identity is not unique")
    return rules


def _validate_continuity(frame: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    continuity = _require_frame(
        frame, label="continuity", required=_CONTINUITY_REQUIRED
    )
    _strings(continuity, _FACT_IDENTITY, label="continuity")
    if continuity.duplicated(list(_FACT_IDENTITY)).any():
        raise VenueReactionPanelError("continuity identity is not unique")
    for column in (
        "next_salient_event_time_utc",
        "suspension_time_utc",
        "game_end_time_utc",
        "continuity_gap_time_utc",
    ):
        continuity[column] = _utc(
            continuity[column],
            label=f"continuity.{column}",
            nullable=True,
        )
    continuity["continuity_verified_until_utc"] = _utc(
        continuity["continuity_verified_until_utc"],
        label="continuity.continuity_verified_until_utc",
        nullable=False,
    )
    expected = set(map(tuple, facts.loc[:, list(_FACT_IDENTITY)].to_numpy()))
    observed = set(map(tuple, continuity.loc[:, list(_FACT_IDENTITY)].to_numpy()))
    if expected != observed:
        raise VenueReactionPanelError(
            "continuity must contain exactly one row per episode fact"
        )
    return continuity


def _validate_noise(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=sorted(_NOISE_REQUIRED))
    noise = _require_frame(
        frame,
        label="matched_control_noise",
        required=_NOISE_REQUIRED,
        allow_empty=True,
    )
    if noise.empty:
        return noise
    _strings(noise, ("venue",), label="matched_control_noise")
    for column in ("landmark_seconds", "endpoint_seconds"):
        numeric = _finite_numeric(
            noise[column], label=f"matched_control_noise.{column}"
        )
        if not np.equal(numeric.to_numpy(), numeric.astype(int).to_numpy()).all():
            raise VenueReactionPanelError(
                f"matched_control_noise.{column} must be integral"
            )
        noise[column] = numeric.astype(int)
    noise["matched_control_p95"] = _finite_numeric(
        noise["matched_control_p95"],
        label="matched_control_noise.matched_control_p95",
        lower=0,
    )
    key = ["venue", "landmark_seconds", "endpoint_seconds"]
    if noise.duplicated(key).any():
        raise VenueReactionPanelError("matched_control_noise identity is not unique")
    return noise


def _select_mark(
    trades: pd.DataFrame,
    *,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    target_time: pd.Timestamp,
) -> _Mark:
    candidates = trades.loc[
        trades["source_time_utc"].ge(interval_end)
        & trades["source_time_utc"].le(target_time)
    ]
    if candidates.empty:
        overlap = trades["source_time_utc"].ge(interval_start) & trades[
            "source_time_utc"
        ].lt(interval_end)
        status = "ORDER_AMBIGUOUS" if overlap.any() else "NO_ACTUAL_TRADE"
        return _Mark(status, None, None, math.nan, math.nan)
    latest_time = candidates["source_time_utc"].max()
    latest = candidates.loc[candidates["source_time_utc"].eq(latest_time)]
    staleness = float((target_time - latest_time).total_seconds())
    if len(latest) != 1:
        return _Mark("ORDER_AMBIGUOUS", None, latest_time, math.nan, staleness)
    row = latest.iloc[0]
    if staleness > MAX_STALENESS_SECONDS:
        return _Mark("STALE", None, latest_time, math.nan, staleness)
    return _Mark(
        "OBSERVED",
        str(row["trade_id"]),
        latest_time,
        float(row["price"]),
        staleness,
    )


def _survival_at_h(
    continuity: pd.Series,
    *,
    endpoint_time: pd.Timestamp,
) -> tuple[object, str | None]:
    if continuity["continuity_verified_until_utc"] < endpoint_time:
        return pd.NA, "CONTINUITY_UNVERIFIED_BEFORE_H"
    observed_censors = [
        (timestamp, reason)
        for column, reason in _CONTINUITY_REASONS
        if pd.notna(timestamp := continuity[column]) and timestamp <= endpoint_time
    ]
    if observed_censors:
        _, reason = min(observed_censors, key=lambda item: (item[0], item[1]))
        return False, reason
    return True, None


def _tick_at(
    rules: pd.DataFrame,
    contract: pd.Series,
    *,
    landmark_time: pd.Timestamp,
) -> tuple[float, str]:
    selected = rules.loc[
        rules["venue"].eq(contract["venue"])
        & rules["tick_rule_id"].eq(contract["tick_rule_id"])
        & rules["effective_start_utc"].le(landmark_time)
        & rules["effective_end_utc"].gt(landmark_time)
    ]
    if len(selected) != 1:
        raise VenueReactionPanelError(
            "exactly one applicable tick rule is required at each landmark"
        )
    row = selected.iloc[0]
    return float(row["tick_size"]), str(row["tick_rule_id"])


def _reference_at_l(
    reference: pd.Series | None,
    *,
    landmark_time: pd.Timestamp,
) -> tuple[str, dict[str, float | None]]:
    unavailable = {
        "p_before_home": None,
        "p_after_home": None,
        "reference_delta_home": None,
    }
    if reference is None:
        return "MISSING", unavailable
    if str(reference["support_status"]) != "SUPPORTED":
        return "UNSUPPORTED", unavailable
    if reference["known_at"] > landmark_time:
        return "NOT_KNOWN_AT_L", unavailable
    return (
        "AVAILABLE",
        {
            "p_before_home": float(reference["p_before_home"]),
            "p_after_home": float(reference["p_after_home"]),
            "reference_delta_home": float(reference["reference_delta_home"]),
        },
    )


def _activity(
    trades: pd.DataFrame,
    *,
    landmark_time: pd.Timestamp,
    seconds: int,
) -> tuple[int, float]:
    lower = landmark_time - pd.Timedelta(seconds=seconds)
    selected = trades.loc[
        trades["source_time_utc"].gt(lower)
        & trades["source_time_utc"].le(landmark_time)
    ]
    return len(selected), float(selected["size"].sum())


def _noise_threshold(
    noise: pd.DataFrame,
    *,
    venue: str,
    landmark: int,
    endpoint: int,
) -> float | None:
    selected = noise.loc[
        noise["venue"].eq(venue)
        & noise["landmark_seconds"].eq(landmark)
        & noise["endpoint_seconds"].eq(endpoint)
    ]
    if selected.empty:
        return None
    return float(selected.iloc[0]["matched_control_p95"])


def _direction(delta: float, tick: float) -> str:
    if delta > tick or math.isclose(delta, tick, rel_tol=0, abs_tol=1e-12):
        return "UP"
    if delta < -tick or math.isclose(delta, -tick, rel_tol=0, abs_tol=1e-12):
        return "DOWN"
    return "NO_MOVE"


def _complement_diagnostics(
    market: pd.DataFrame,
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    complements = contracts.loc[
        contracts["contract_role"].eq("DERIVED_AWAY_COMPLEMENT")
    ]
    if complements.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "venue",
                "contract_id",
                "trade_id",
                "source_time_utc",
                "price",
                "kind",
                "provenance",
                "primary_target_eligible",
                "diagnostic_reason",
            ]
        )
    diagnostics = market.merge(
        complements[["game_id", "venue", "contract_id"]],
        on=["game_id", "venue", "contract_id"],
        how="inner",
        validate="many_to_one",
    )
    diagnostics["primary_target_eligible"] = False
    diagnostics["diagnostic_reason"] = "DERIVED_AWAY_COMPLEMENT_DIAGNOSTIC_ONLY"
    columns = [
        "game_id",
        "venue",
        "contract_id",
        "trade_id",
        "source_time_utc",
        "price",
        "kind",
        "provenance",
        "primary_target_eligible",
        "diagnostic_reason",
    ]
    return diagnostics.loc[:, columns].sort_values(
        ["game_id", "venue", "contract_id", "source_time_utc", "trade_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_venue_reaction_panel_v3(
    *,
    episode_facts: pd.DataFrame,
    stage_a_references: pd.DataFrame,
    market_rows: pd.DataFrame,
    contract_metadata: pd.DataFrame,
    tick_rules: pd.DataFrame,
    continuity: pd.DataFrame,
    matched_control_noise: pd.DataFrame | None = None,
) -> VenueReactionPanelV3:
    """Build the complete actual-home VenueReactionPanelV3 and attrition audit."""

    facts = _validate_facts(episode_facts)
    references = _validate_references(stage_a_references)
    market = _validate_market(market_rows)
    contracts = _validate_contracts(contract_metadata)
    rules = _validate_rules(tick_rules)
    continuity_rows = _validate_continuity(continuity, facts)
    noise = _validate_noise(matched_control_noise)

    reference_index = {
        (str(row["game_id"]), str(row["atomic_information_episode_id"])): pd.Series(
            row
        )
        for row in references.to_dict("records")
    }
    continuity_index = {
        (str(row["game_id"]), str(row["atomic_information_episode_id"])): pd.Series(
            row
        )
        for row in continuity_rows.to_dict("records")
    }
    home_contracts = contracts.loc[
        contracts["contract_role"].eq("ACTUAL_HOME_OUTCOME")
    ].sort_values(["game_id", "venue", "contract_id"], kind="mergesort")
    observed_trades = market.loc[
        market["kind"].eq("trade") & market["provenance"].eq("observed")
    ].copy()
    rows: list[dict[str, object]] = []

    ordered_facts = facts.sort_values(
        ["game_id", "source_interval_start", "atomic_information_episode_id"],
        kind="mergesort",
    )
    for raw_fact in ordered_facts.to_dict("records"):
        fact = pd.Series(raw_fact)
        fact_key = (
            str(fact["game_id"]),
            str(fact["atomic_information_episode_id"]),
        )
        reference = reference_index.get(fact_key)
        continuity_row = continuity_index[fact_key]
        scoped_contracts = home_contracts.loc[
            home_contracts["game_id"].eq(fact["game_id"])
            & home_contracts["home_team"].eq(fact["home_team"])
        ]
        for raw_contract in scoped_contracts.to_dict("records"):
            contract = pd.Series(raw_contract)
            trades = observed_trades.loc[
                observed_trades["game_id"].eq(fact["game_id"])
                & observed_trades["venue"].eq(contract["venue"])
                & observed_trades["contract_id"].eq(contract["contract_id"])
            ].sort_values(["source_time_utc", "trade_id"], kind="mergesort")
            interval_start = fact["source_interval_start"]
            interval_end = fact["source_interval_end"]
            marks_l = {
                offset: _select_mark(
                    trades,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    target_time=interval_end + pd.Timedelta(seconds=offset),
                )
                for offset in LANDMARK_SECONDS
            }
            marks_h = {
                offset: _select_mark(
                    trades,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    target_time=interval_end + pd.Timedelta(seconds=offset),
                )
                for offset in ENDPOINT_SECONDS
            }
            for landmark in LANDMARK_SECONDS:
                landmark_time = interval_end + pd.Timedelta(seconds=landmark)
                mark_l = marks_l[landmark]
                tick, tick_rule_id = _tick_at(
                    rules, contract, landmark_time=landmark_time
                )
                stage_a_status, stage_a = _reference_at_l(
                    reference, landmark_time=landmark_time
                )
                count_30, size_30 = _activity(
                    trades, landmark_time=landmark_time, seconds=30
                )
                count_60, size_60 = _activity(
                    trades, landmark_time=landmark_time, seconds=60
                )
                for endpoint in ENDPOINT_SECONDS:
                    if endpoint <= landmark:
                        continue
                    endpoint_time = interval_end + pd.Timedelta(seconds=endpoint)
                    mark_h = marks_h[endpoint]
                    s_h, survival_reason = _survival_at_h(
                        continuity_row, endpoint_time=endpoint_time
                    )
                    decision_eligible = bool(
                        fact["stage_b_information_event_eligible"]
                    ) and mark_l.observed
                    if s_h is True:
                        if mark_h.status == "ORDER_AMBIGUOUS":
                            o_h: object = pd.NA
                        else:
                            o_h = mark_h.observed
                    else:
                        o_h = pd.NA
                    target_eligible = (
                        decision_eligible and s_h is True and o_h is True
                    )
                    delta = (
                        mark_h.price - mark_l.price
                        if target_eligible
                        else math.nan
                    )
                    direction = (
                        _direction(delta, tick)
                        if target_eligible
                        else "UNOBSERVED"
                    )
                    magnitude = (
                        abs(delta)
                        if direction in {"UP", "DOWN"}
                        else math.nan
                    )
                    p95 = _noise_threshold(
                        noise,
                        venue=str(contract["venue"]),
                        landmark=landmark,
                        endpoint=endpoint,
                    )
                    abnormal: object = pd.NA
                    if target_eligible and p95 is not None:
                        magnitude_at_h = abs(delta)
                        abnormal = magnitude_at_h > p95 or math.isclose(
                            magnitude_at_h,
                            p95,
                            rel_tol=0,
                            abs_tol=1e-12,
                        )
                    if not bool(fact["stage_b_information_event_eligible"]):
                        attrition_reason = "FACT_NOT_STAGE_B_ELIGIBLE"
                    elif not mark_l.observed:
                        attrition_reason = f"LANDMARK_{mark_l.status}"
                    elif survival_reason is not None:
                        attrition_reason = survival_reason
                    elif mark_h.status != "OBSERVED":
                        attrition_reason = f"ENDPOINT_{mark_h.status}"
                    else:
                        attrition_reason = "ELIGIBLE"

                    reference_gap = (
                        mark_l.price - float(stage_a["p_after_home"])
                        if mark_l.observed
                        and stage_a_status == "AVAILABLE"
                        and stage_a["p_after_home"] is not None
                        else math.nan
                    )
                    decision_features = {
                        "schema_version": SCHEMA_VERSION,
                        "game_id": str(fact["game_id"]),
                        "atomic_information_episode_id": str(
                            fact["atomic_information_episode_id"]
                        ),
                        "venue": str(contract["venue"]),
                        "actual_home_contract_id": str(contract["contract_id"]),
                        "landmark_seconds": landmark,
                        "endpoint_seconds": endpoint,
                        "event_interval_start": interval_start,
                        "event_interval_end": interval_end,
                        "tick_rule_id": tick_rule_id,
                        "tick_size": tick,
                        "mark_l_trade_id": mark_l.trade_id,
                        "mark_l_source_time_utc": mark_l.source_time,
                        "mark_l_price": (
                            mark_l.price if mark_l.observed else None
                        ),
                        "mark_l_staleness_seconds": (
                            mark_l.staleness_seconds if mark_l.observed else None
                        ),
                        "prior_30s_actual_trade_count": count_30,
                        "prior_30s_actual_trade_size": size_30,
                        "prior_60s_actual_trade_count": count_60,
                        "prior_60s_actual_trade_size": size_60,
                        "stage_a_status": stage_a_status,
                        **stage_a,
                        "reference_gap_at_landmark": reference_gap,
                        "fact_features": {
                            column: fact.get(column)
                            for column in _FACT_FEATURE_COLUMNS
                        },
                    }
                    decision_json = _canonical_json(decision_features)
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "claim_boundary": CLAIM_BOUNDARY,
                            "game_id": str(fact["game_id"]),
                            "atomic_information_episode_id": str(
                                fact["atomic_information_episode_id"]
                            ),
                            "venue": str(contract["venue"]),
                            "actual_home_contract_id": str(
                                contract["contract_id"]
                            ),
                            "home_team": str(fact["home_team"]),
                            "away_team": str(fact["away_team"]),
                            "target_orientation": "ACTUAL_HOME_OUTCOME",
                            "source_interval_start": interval_start,
                            "source_interval_end": interval_end,
                            "source_interval_semantics": "[START,END)",
                            "landmark_seconds": landmark,
                            "endpoint_seconds": endpoint,
                            "landmark_utc": landmark_time,
                            "endpoint_utc": endpoint_time,
                            "tick_rule_id": tick_rule_id,
                            "tick_size": tick,
                            "mark_l_trade_id": mark_l.trade_id,
                            "mark_l_source_time_utc": mark_l.source_time,
                            "mark_l_price": (
                                mark_l.price if mark_l.observed else math.nan
                            ),
                            "mark_l_staleness_seconds": (
                                mark_l.staleness_seconds
                                if mark_l.observed
                                else math.nan
                            ),
                            "mark_h_trade_id": mark_h.trade_id,
                            "mark_h_source_time_utc": mark_h.source_time,
                            "mark_h_price": (
                                mark_h.price if mark_h.observed else math.nan
                            ),
                            "mark_h_staleness_seconds": (
                                mark_h.staleness_seconds
                                if mark_h.observed
                                else math.nan
                            ),
                            "s_h": s_h,
                            "o_h_given_s": o_h,
                            "decision_eligible": decision_eligible,
                            "target_eligible": target_eligible,
                            "delta_l_h": delta,
                            "direction": direction,
                            "conditional_magnitude": magnitude,
                            "matched_control_p95": (
                                p95 if p95 is not None else math.nan
                            ),
                            "abnormal_move": abnormal,
                            "stage_a_status": stage_a_status,
                            "p_before_home": stage_a["p_before_home"],
                            "p_after_home": stage_a["p_after_home"],
                            "reference_delta_home": stage_a[
                                "reference_delta_home"
                            ],
                            "reference_gap_at_landmark": reference_gap,
                            "prior_30s_actual_trade_count": count_30,
                            "prior_30s_actual_trade_size": size_30,
                            "prior_60s_actual_trade_count": count_60,
                            "prior_60s_actual_trade_size": size_60,
                            "decision_features_json": decision_json,
                            "decision_feature_sha256": _sha256_text(decision_json),
                            "attrition_reason": attrition_reason,
                        }
                    )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise VenueReactionPanelError(
            "no actual home-outcome contracts joined the episode facts"
        )
    if panel.duplicated(list(_PANEL_GRAIN)).any():
        raise VenueReactionPanelError("VenueReactionPanelV3 grain is not unique")
    panel = panel.sort_values(list(_PANEL_GRAIN), kind="mergesort").reset_index(
        drop=True
    )
    attrition = (
        panel.groupby(
            [
                "venue",
                "landmark_seconds",
                "endpoint_seconds",
                "attrition_reason",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            row_count=("atomic_information_episode_id", "size"),
            game_count=("game_id", "nunique"),
            episode_count=("atomic_information_episode_id", "nunique"),
        )
        .sort_values(
            [
                "venue",
                "landmark_seconds",
                "endpoint_seconds",
                "attrition_reason",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return VenueReactionPanelV3(
        panel=panel,
        attrition=attrition,
        complement_diagnostics=_complement_diagnostics(market, contracts),
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "ENDPOINT_SECONDS",
    "LANDMARK_SECONDS",
    "MAX_STALENESS_SECONDS",
    "SCHEMA_VERSION",
    "VenueReactionPanelError",
    "VenueReactionPanelV3",
    "build_venue_reaction_panel_v3",
]
