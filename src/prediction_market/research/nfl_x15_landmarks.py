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
import re
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
    "event_id",
    "source_interval_start",
    "source_interval_end",
    "known_at",
    "source_resolution",
    "stage_b_information_event_eligible",
    "home_team",
    "away_team",
    "outcome_tags",
    "pbp_source_sha256",
}
_COHORT_METADATA_REQUIRED = {
    "game_id",
    "nfl_week",
    "cohort",
    "authority_sha256",
}
_REFERENCE_REQUIRED = {
    *_FACT_IDENTITY,
    "reference_status",
    "pre_state_known_at",
    "post_state_known_at",
    "p_before_home",
    "p_after_home",
    "reference_delta_home",
}
_FACTOR_HIT_REQUIRED = {
    "game_id",
    "event_id",
    "play_id",
    "factor_id",
    "factor_version",
    "registry_sha256",
    "pbp_source_sha256",
    "predicate_evidence",
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
    ("next_salient_event_time_utc", "NEXT_SALIENT_EVENT"),
    ("suspension_time_utc", "SUSPENSION"),
    ("game_end_time_utc", "GAME_END"),
    ("continuity_gap_time_utc", "CONTINUITY_GAP"),
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")


class VenueReactionPanelError(ValueError):
    """Inputs cannot support a deterministic VenueReactionPanelV3."""


@dataclass(frozen=True, slots=True)
class VenueReactionPanelV3:
    """Primary home-outcome rows and their audit tables."""

    panel: pd.DataFrame
    attrition: pd.DataFrame
    complement_diagnostics: pd.DataFrame
    fact_attrition: pd.DataFrame
    factor_membership: pd.DataFrame


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


@dataclass(frozen=True, slots=True)
class _TradeIndex:
    times_ns: np.ndarray
    trade_ids: np.ndarray
    prices: np.ndarray
    sizes: np.ndarray
    prefix_sizes: np.ndarray


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
    parsed_values: list[pd.Timestamp | pd.NaTType] = []
    for value in values:
        missing = value is None or value is pd.NA
        if not missing:
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
        if missing:
            if not nullable:
                raise VenueReactionPanelError(
                    f"{label} must contain UTC timestamps"
                )
            parsed_values.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise VenueReactionPanelError(
                f"{label} must contain UTC timestamps"
            ) from exc
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
            or timestamp.utcoffset().total_seconds() != 0
        ):
            raise VenueReactionPanelError(
                f"{label} must contain explicit UTC timestamps"
            )
        parsed_values.append(timestamp.tz_convert("UTC"))
    return pd.Series(
        parsed_values,
        index=values.index,
        dtype="datetime64[ns, UTC]",
    )


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


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise VenueReactionPanelError(f"{label} must be a sha256 digest")
    return value


def _parse_event_tags(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise VenueReactionPanelError(f"{label} must be a JSON array") from exc
    elif isinstance(value, (list, tuple)):
        decoded = value
    else:
        raise VenueReactionPanelError(f"{label} must be a JSON array")
    if not isinstance(decoded, (list, tuple)):
        raise VenueReactionPanelError(f"{label} must be a JSON array")
    tags: list[str] = []
    for tag in decoded:
        if type(tag) is not str or not tag.strip() or tag != tag.strip():
            raise VenueReactionPanelError(f"{label} has a noncanonical tag")
        tags.append(tag)
    if len(tags) != len(set(tags)):
        raise VenueReactionPanelError(f"{label} contains duplicate tags")
    return tuple(sorted(tags))


def _validate_facts(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    facts = _require_frame(frame, label="episode_facts", required=_FACT_REQUIRED)
    _strings(
        facts,
        (
            *_FACT_IDENTITY,
            "event_id",
            "source_resolution",
            "home_team",
            "away_team",
            "pbp_source_sha256",
        ),
        label="episode_facts",
    )
    if facts.duplicated(list(_FACT_IDENTITY)).any():
        raise VenueReactionPanelError("episode_facts identity is not unique")
    if facts.duplicated(["game_id", "event_id"]).any():
        raise VenueReactionPanelError("episode_facts event_id is not unique")
    _strict_bool(
        facts,
        "stage_b_information_event_eligible",
        label="episode_facts",
    )
    if facts["home_team"].eq(facts["away_team"]).any():
        raise VenueReactionPanelError("episode home and away teams must differ")
    if facts.groupby("game_id")[["home_team", "away_team"]].nunique().gt(1).any(
        axis=None
    ):
        raise VenueReactionPanelError("episode game team identity is inconsistent")
    for source_hash in facts["pbp_source_sha256"]:
        _sha256(source_hash, label="episode_facts.pbp_source_sha256")
    facts["_event_tags"] = [
        _parse_event_tags(value, label="episode_facts.outcome_tags")
        for value in facts["outcome_tags"]
    ]
    for column in ("source_interval_start", "source_interval_end", "known_at"):
        facts[column] = _utc(
            facts[column],
            label=f"episode_facts.{column}",
            nullable=True,
        )

    audit_rows: list[dict[str, object]] = []
    included: list[bool] = []
    for row in facts.to_dict("records"):
        if not bool(row["stage_b_information_event_eligible"]):
            reason = "NOT_STAGE_B_INFORMATION_EVENT"
        elif pd.isna(row["source_interval_start"]) or pd.isna(
            row["source_interval_end"]
        ):
            reason = "MISSING_SOURCE_INTERVAL"
        elif row["source_interval_end"] <= row["source_interval_start"]:
            reason = "INVALID_SOURCE_INTERVAL"
        elif pd.isna(row["known_at"]):
            reason = "MISSING_FACT_KNOWN_AT"
        elif row["known_at"] > row["source_interval_end"]:
            reason = "FACT_KNOWN_AFTER_INTERVAL_END"
        else:
            reason = "INCLUDED"
        is_included = reason == "INCLUDED"
        included.append(is_included)
        audit_rows.append(
            {
                "game_id": str(row["game_id"]),
                "event_id": str(row["event_id"]),
                "atomic_information_episode_id": str(
                    row["atomic_information_episode_id"]
                ),
                "stage_b_information_event_eligible": bool(
                    row["stage_b_information_event_eligible"]
                ),
                "included_in_panel": is_included,
                "fact_attrition_reason": reason,
            }
        )
    eligible = facts.loc[included].copy().reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values(
        ["game_id", "event_id", "atomic_information_episode_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return facts, eligible, audit


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
        (*_FACT_IDENTITY, "reference_status"),
        label="stage_a_references",
    )
    if references.duplicated(list(_FACT_IDENTITY)).any():
        raise VenueReactionPanelError("stage_a_references identity is not unique")
    for column in ("pre_state_known_at", "post_state_known_at"):
        references[column] = _utc(
            references[column],
            label=f"stage_a_references.{column}",
            nullable=True,
        )
    numeric_columns = (
        "p_before_home",
        "p_after_home",
        "reference_delta_home",
    )
    for column in numeric_columns:
        raw = references[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid_supplied = raw.notna() & (
            numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
        )
        if invalid_supplied.any():
            raise VenueReactionPanelError(
                f"stage_a_references.{column} must be finite or null"
            )
        references[column] = numeric
    for column in ("p_before_home", "p_after_home"):
        supplied = references[column].notna()
        if (
            references.loc[supplied, column].lt(0).any()
            or references.loc[supplied, column].gt(1).any()
        ):
            raise VenueReactionPanelError(
                f"stage_a_references.{column} must be in [0, 1]"
            )
    supported = references["reference_status"].eq("SUPPORTED")
    required_supported = [
        "pre_state_known_at",
        "post_state_known_at",
        *numeric_columns,
    ]
    if references.loc[supported, required_supported].isna().any(axis=None):
        raise VenueReactionPanelError(
            "SUPPORTED Stage A references require finite values and known-at timestamps"
        )
    supported_rows = references.loc[supported]
    if (
        supported_rows["post_state_known_at"]
        < supported_rows["pre_state_known_at"]
    ).any():
        raise VenueReactionPanelError(
            "SUPPORTED Stage A post_state_known_at precedes pre_state_known_at"
        )
    expected_delta = (
        supported_rows["p_after_home"] - supported_rows["p_before_home"]
    )
    if not np.allclose(
        expected_delta.to_numpy(dtype=float),
        supported_rows["reference_delta_home"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise VenueReactionPanelError(
            "SUPPORTED Stage A reference delta does not match probabilities"
        )
    return references


def _validate_factor_hits(
    frame: pd.DataFrame,
    facts: pd.DataFrame,
) -> pd.DataFrame:
    hits = _require_frame(
        frame,
        label="factor_hits",
        required=_FACTOR_HIT_REQUIRED,
        allow_empty=True,
    )
    if hits.empty:
        return hits
    _strings(
        hits,
        tuple(sorted(_FACTOR_HIT_REQUIRED)),
        label="factor_hits",
    )
    key = ["game_id", "event_id", "factor_id", "factor_version"]
    if hits.duplicated(key).any():
        raise VenueReactionPanelError("factor_hits identity is not unique")
    for column in ("registry_sha256", "pbp_source_sha256"):
        for value in hits[column]:
            _sha256(value, label=f"factor_hits.{column}")
    if hits["registry_sha256"].nunique() != 1:
        raise VenueReactionPanelError(
            "factor_hits must bind one factor registry sha256"
        )
    fact_sources = facts.loc[
        :,
        ["game_id", "event_id", "known_at", "pbp_source_sha256"],
    ].rename(columns={"pbp_source_sha256": "_fact_source_sha256"})
    joined = hits.merge(
        fact_sources,
        on=["game_id", "event_id"],
        how="left",
        indicator="_fact_merge",
        validate="many_to_one",
    )
    if not joined["_fact_merge"].eq("both").all():
        raise VenueReactionPanelError(
            "factor_hits.event_id must link to exactly one episode fact"
        )
    if not joined["pbp_source_sha256"].eq(
        joined["_fact_source_sha256"]
    ).all():
        raise VenueReactionPanelError(
            "factor_hits source hash differs from its episode fact"
        )
    for evidence in joined["predicate_evidence"]:
        try:
            json.loads(evidence)
        except json.JSONDecodeError as exc:
            raise VenueReactionPanelError(
                "factor_hits.predicate_evidence must be valid JSON"
            ) from exc
    return joined.drop(columns=["_fact_source_sha256", "_fact_merge"]).sort_values(
        key,
        kind="mergesort",
    ).reset_index(drop=True)


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
    home_counts = (
        contracts.assign(_is_actual_home=home)
        .groupby(["game_id", "venue"], sort=False)["_is_actual_home"]
        .sum()
    )
    if not home_counts.eq(1).all():
        raise VenueReactionPanelError(
            "exactly one actual home contract is required per game and venue"
        )
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


def _validate_cohort_metadata(
    frame: pd.DataFrame,
    *,
    facts: pd.DataFrame,
    expected_authority_sha256: str,
    expected_mapping_sha256: str,
) -> pd.DataFrame:
    expected_authority = _sha256(
        expected_authority_sha256,
        label="expected_cohort_authority_sha256",
    )
    expected_mapping = _sha256(
        expected_mapping_sha256,
        label="expected_cohort_mapping_sha256",
    )
    metadata = _require_frame(
        frame,
        label="cohort_metadata",
        required=_COHORT_METADATA_REQUIRED,
    )
    _strings(
        metadata,
        ("game_id", "cohort", "authority_sha256"),
        label="cohort_metadata",
    )
    if metadata.duplicated(["game_id"]).any():
        raise VenueReactionPanelError("cohort_metadata game_id is not unique")
    if not metadata["cohort"].eq("development").all():
        raise VenueReactionPanelError("cohort_metadata cohort must be development")
    weeks = _finite_numeric(
        metadata["nfl_week"], label="cohort_metadata.nfl_week", lower=1
    )
    if not np.equal(weeks.to_numpy(), weeks.astype(int).to_numpy()).all():
        raise VenueReactionPanelError("cohort_metadata.nfl_week must be integral")
    metadata["nfl_week"] = weeks.astype(int)
    if metadata["nfl_week"].gt(12).any():
        raise VenueReactionPanelError("cohort_metadata.nfl_week must be in 1..12")
    for authority in metadata["authority_sha256"]:
        _sha256(authority, label="cohort_metadata.authority_sha256")
    if metadata["authority_sha256"].nunique() != 1:
        raise VenueReactionPanelError("cohort_metadata authority_sha256 must be single")
    observed_authority = str(metadata["authority_sha256"].iloc[0])
    if observed_authority != expected_authority:
        raise VenueReactionPanelError(
            "cohort_metadata authority_sha256 does not match frozen authority"
        )
    expected = set(facts["game_id"].astype(str))
    observed = set(metadata["game_id"].astype(str))
    if expected != observed:
        raise VenueReactionPanelError(
            "cohort_metadata game set must exactly match episode_facts"
        )
    canonical_mapping = [
        {
            "cohort": str(row["cohort"]),
            "game_id": str(row["game_id"]),
            "nfl_week": int(row["nfl_week"]),
        }
        for row in metadata.sort_values("game_id", kind="mergesort").to_dict("records")
    ]
    observed_mapping = _sha256_text(_canonical_json(canonical_mapping))
    if observed_mapping != expected_mapping:
        raise VenueReactionPanelError(
            "cohort_metadata mapping does not match frozen authority"
        )
    return metadata


def _trade_index(trades: pd.DataFrame) -> _TradeIndex:
    ordered = trades.sort_values(["source_time_utc", "trade_id"], kind="mergesort")
    times_ns = ordered["source_time_utc"].astype("int64").to_numpy(copy=True)
    trade_ids = ordered["trade_id"].astype(str).to_numpy(copy=True)
    prices = ordered["price"].to_numpy(dtype=float, copy=True)
    sizes = ordered["size"].to_numpy(dtype=float, copy=True)
    prefix_sizes = np.empty(len(sizes) + 1, dtype=float)
    prefix_sizes[0] = 0.0
    np.cumsum(sizes, out=prefix_sizes[1:])
    for array in (times_ns, trade_ids, prices, sizes, prefix_sizes):
        array.setflags(write=False)
    return _TradeIndex(
        times_ns=times_ns,
        trade_ids=trade_ids,
        prices=prices,
        sizes=sizes,
        prefix_sizes=prefix_sizes,
    )


def _select_mark(
    trades: _TradeIndex,
    *,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    target_time: pd.Timestamp,
) -> _Mark:
    interval_start_ns = interval_start.value
    interval_end_ns = interval_end.value
    target_ns = target_time.value
    first_candidate = int(
        np.searchsorted(trades.times_ns, interval_end_ns, side="left")
    )
    after_target = int(
        np.searchsorted(trades.times_ns, target_ns, side="right")
    )
    if first_candidate == after_target:
        first_overlap = int(
            np.searchsorted(trades.times_ns, interval_start_ns, side="left")
        )
        after_overlap = int(
            np.searchsorted(trades.times_ns, interval_end_ns, side="left")
        )
        status = (
            "ORDER_AMBIGUOUS"
            if first_overlap < after_overlap
            else "NO_ACTUAL_TRADE"
        )
        return _Mark(status, None, None, math.nan, math.nan)
    latest_index = after_target - 1
    latest_ns = int(trades.times_ns[latest_index])
    first_latest = int(
        np.searchsorted(trades.times_ns, latest_ns, side="left")
    )
    after_latest = int(
        np.searchsorted(trades.times_ns, latest_ns, side="right")
    )
    latest_time = pd.Timestamp(latest_ns, tz="UTC")
    staleness = float((target_ns - latest_ns) / 1_000_000_000)
    if after_latest - first_latest != 1:
        return _Mark("ORDER_AMBIGUOUS", None, latest_time, math.nan, staleness)
    if staleness > MAX_STALENESS_SECONDS:
        return _Mark("STALE", None, latest_time, math.nan, staleness)
    return _Mark(
        "OBSERVED",
        str(trades.trade_ids[latest_index]),
        latest_time,
        float(trades.prices[latest_index]),
        staleness,
    )


def _survival_at_h(
    continuity: pd.Series,
    *,
    endpoint_time: pd.Timestamp,
) -> tuple[object, str | None]:
    if continuity["continuity_verified_until_utc"] < endpoint_time:
        return pd.NA, "CONTINUITY_UNVERIFIED_BEFORE_H"
    observed_censors: list[tuple[pd.Timestamp, str]] = []
    for column, reason_root in _CONTINUITY_REASONS:
        timestamp = continuity[column]
        if pd.isna(timestamp) or timestamp > endpoint_time:
            continue
        reason = (
            f"{reason_root}_AT_H_ORDER_AMBIGUOUS"
            if timestamp == endpoint_time
            else f"{reason_root}_BEFORE_H"
        )
        observed_censors.append((timestamp, reason))
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
) -> tuple[str, dict[str, object]]:
    unavailable = {
        "p_before_home": None,
        "p_after_home": None,
        "reference_delta_home": None,
        "pre_state_known_at": None,
        "post_state_known_at": None,
    }
    if reference is None:
        return "UNAVAILABLE", unavailable
    if (
        pd.notna(reference["post_state_known_at"])
        and reference["post_state_known_at"] > landmark_time
    ):
        return "NOT_KNOWN_AT_L", unavailable
    reference_status = str(reference["reference_status"])
    if reference_status != "SUPPORTED":
        return "UNAVAILABLE", unavailable
    return (
        "AVAILABLE",
        {
            "p_before_home": float(reference["p_before_home"]),
            "p_after_home": float(reference["p_after_home"]),
            "reference_delta_home": float(reference["reference_delta_home"]),
            "pre_state_known_at": reference["pre_state_known_at"],
            "post_state_known_at": reference["post_state_known_at"],
        },
    )


def _activity(
    trades: _TradeIndex,
    *,
    landmark_time: pd.Timestamp,
    seconds: int,
) -> tuple[int, float]:
    landmark_ns = landmark_time.value
    lower_ns = landmark_ns - seconds * 1_000_000_000
    first = int(np.searchsorted(trades.times_ns, lower_ns, side="right"))
    after = int(np.searchsorted(trades.times_ns, landmark_ns, side="right"))
    return after - first, float(trades.prefix_sizes[after] - trades.prefix_sizes[first])


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
    factor_hits: pd.DataFrame,
    market_rows: pd.DataFrame,
    contract_metadata: pd.DataFrame,
    tick_rules: pd.DataFrame,
    continuity: pd.DataFrame,
    cohort_metadata: pd.DataFrame,
    expected_cohort_authority_sha256: str,
    expected_cohort_mapping_sha256: str,
) -> VenueReactionPanelV3:
    """Build the complete actual-home VenueReactionPanelV3 and attrition audit."""

    all_facts, facts, fact_attrition = _validate_facts(episode_facts)
    cohorts = _validate_cohort_metadata(
        cohort_metadata,
        facts=all_facts,
        expected_authority_sha256=expected_cohort_authority_sha256,
        expected_mapping_sha256=expected_cohort_mapping_sha256,
    )
    references = _validate_references(stage_a_references)
    hits = _validate_factor_hits(factor_hits, all_facts)
    market = _validate_market(market_rows)
    contracts = _validate_contracts(contract_metadata)
    rules = _validate_rules(tick_rules)
    continuity_rows = _validate_continuity(continuity, facts)
    facts = facts.merge(
        cohorts.loc[:, ["game_id", "nfl_week", "authority_sha256"]],
        on="game_id",
        how="inner",
        validate="many_to_one",
    )

    eligible_event_keys = set(
        map(
            tuple,
            facts.loc[:, ["game_id", "event_id"]].astype(str).to_numpy(),
        )
    )
    relevant_hits = hits.loc[
        [
            (str(row["game_id"]), str(row["event_id"])) in eligible_event_keys
            for row in hits.to_dict("records")
        ]
    ].copy()
    factor_membership = relevant_hits.merge(
        facts.loc[
            :,
            ["game_id", "event_id", "atomic_information_episode_id"],
        ],
        on=["game_id", "event_id"],
        how="inner",
        validate="many_to_one",
    ).loc[
        :,
        [
            "game_id",
            "atomic_information_episode_id",
            "factor_id",
            "factor_version",
            "registry_sha256",
            "pbp_source_sha256",
            "predicate_evidence",
        ],
    ]
    membership_grain = [
        "game_id",
        "atomic_information_episode_id",
        "factor_id",
        "factor_version",
    ]
    if factor_membership.duplicated(membership_grain).any():
        raise VenueReactionPanelError("factor membership grain is not unique")
    factor_membership = factor_membership.sort_values(
        membership_grain,
        kind="mergesort",
    ).reset_index(drop=True)
    factor_ids = tuple(sorted(relevant_hits["factor_id"].astype(str).unique()))
    event_tags = tuple(
        sorted(
            {
                tag
                for tags in facts["_event_tags"]
                for tag in tags
            }
        )
    )
    factor_columns = {
        factor_id: f"factor__{factor_id}" for factor_id in factor_ids
    }
    event_tag_columns = {
        event_tag: f"event_tag__{event_tag}" for event_tag in event_tags
    }
    dynamic_columns = tuple(factor_columns.values()) + tuple(
        event_tag_columns.values()
    )
    if len(dynamic_columns) != len(set(dynamic_columns)):
        raise VenueReactionPanelError("factor/event multi-hot columns collide")

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
    factor_hits_by_event = {
        (str(game_id), str(event_id)): group.sort_values(
            ["factor_id", "factor_version"], kind="mergesort"
        ).reset_index(drop=True)
        for (game_id, event_id), group in relevant_hits.groupby(
            ["game_id", "event_id"],
            sort=False,
        )
    }
    empty_factor_hits = relevant_hits.iloc[0:0].copy()
    home_contracts_by_game_home = {
        (str(game_id), str(home_team)): group.reset_index(drop=True)
        for (game_id, home_team), group in home_contracts.groupby(
            ["game_id", "home_team"],
            sort=False,
        )
    }
    empty_home_contracts = home_contracts.iloc[0:0].copy()
    eligible_games = set(facts["game_id"].astype(str))
    contract_games = set(home_contracts["game_id"].astype(str))
    if not eligible_games.issubset(contract_games):
        raise VenueReactionPanelError(
            "every eligible fact game requires an actual home contract"
        )
    fact_home_by_game = (
        facts.loc[:, ["game_id", "home_team"]]
        .drop_duplicates()
        .set_index("game_id")["home_team"]
        .astype(str)
        .to_dict()
    )
    for contract in home_contracts.to_dict("records"):
        game_id = str(contract["game_id"])
        if (
            game_id in fact_home_by_game
            and str(contract["home_team"]) != fact_home_by_game[game_id]
        ):
            raise VenueReactionPanelError(
                "actual home contract disagrees with episode home team"
            )
    observed_trades = market.loc[
        market["kind"].eq("trade") & market["provenance"].eq("observed")
    ].copy()
    observed_trade_indices = {
        (str(game_id), str(venue), str(contract_id)): _trade_index(group)
        for (game_id, venue, contract_id), group in observed_trades.groupby(
            ["game_id", "venue", "contract_id"],
            sort=False,
        )
    }
    empty_observed_trades = _trade_index(observed_trades.iloc[0:0])
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
        reference_status = (
            "MISSING"
            if reference is None
            else str(reference["reference_status"])
        )
        reference_pre_state_known_at = (
            None if reference is None else reference["pre_state_known_at"]
        )
        reference_post_state_known_at = (
            None if reference is None else reference["post_state_known_at"]
        )
        continuity_row = continuity_index[fact_key]
        fact_hits = factor_hits_by_event.get(
            (str(fact["game_id"]), str(fact["event_id"])),
            empty_factor_hits,
        )
        hit_factor_ids = set(fact_hits["factor_id"].astype(str))
        fact_event_tags = set(fact["_event_tags"])
        multi_hot_features: dict[str, bool] = {
            **{
                column: factor_id in hit_factor_ids
                for factor_id, column in factor_columns.items()
            },
            **{
                column: event_tag in fact_event_tags
                for event_tag, column in event_tag_columns.items()
            },
        }
        factor_provenance = [
            {
                "feature_column": factor_columns[str(hit["factor_id"])],
                "factor_id": str(hit["factor_id"]),
                "factor_version": str(hit["factor_version"]),
                "feature_known_at": hit["known_at"],
                "source_event_id": str(hit["event_id"]),
                "source_play_id": str(hit["play_id"]),
                "source_hash": str(hit["pbp_source_sha256"]),
                "registry_sha256": str(hit["registry_sha256"]),
                "predicate_evidence": json.loads(
                    str(hit["predicate_evidence"])
                ),
            }
            for hit in fact_hits.to_dict("records")
        ]
        event_tag_provenance = [
            {
                "feature_column": event_tag_columns[event_tag],
                "event_tag": event_tag,
                "feature_known_at": fact["known_at"],
                "source_event_id": str(fact["event_id"]),
                "source_hash": str(fact["pbp_source_sha256"]),
            }
            for event_tag in sorted(fact_event_tags)
        ]
        scoped_contracts = home_contracts_by_game_home.get(
            (str(fact["game_id"]), str(fact["home_team"])),
            empty_home_contracts,
        )
        for raw_contract in scoped_contracts.to_dict("records"):
            contract = pd.Series(raw_contract)
            trades = observed_trade_indices.get(
                (
                    str(fact["game_id"]),
                    str(contract["venue"]),
                    str(contract["contract_id"]),
                ),
                empty_observed_trades,
            )
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
                        "multi_hot_features": multi_hot_features,
                        "factor_feature_provenance": factor_provenance,
                        "event_tag_feature_provenance": event_tag_provenance,
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
                            "nfl_week": int(fact["nfl_week"]),
                            "cohort_authority_sha256": str(
                                fact["authority_sha256"]
                            ),
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
                            "stage_a_status": stage_a_status,
                            "reference_status": reference_status,
                            "p_before_home": stage_a["p_before_home"],
                            "p_after_home": stage_a["p_after_home"],
                            "reference_delta_home": stage_a[
                                "reference_delta_home"
                            ],
                            "pre_state_known_at": (
                                reference_pre_state_known_at
                            ),
                            "post_state_known_at": (
                                reference_post_state_known_at
                            ),
                            "reference_gap_at_landmark": reference_gap,
                            "prior_30s_actual_trade_count": count_30,
                            "prior_30s_actual_trade_size": size_30,
                            "prior_60s_actual_trade_count": count_60,
                            "prior_60s_actual_trade_size": size_60,
                            "decision_features_json": decision_json,
                            "decision_feature_sha256": _sha256_text(decision_json),
                            "attrition_reason": attrition_reason,
                            **multi_hot_features,
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
        fact_attrition=fact_attrition,
        factor_membership=factor_membership,
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
