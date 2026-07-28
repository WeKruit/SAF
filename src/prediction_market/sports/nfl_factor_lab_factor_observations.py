"""Formal V2 construction of row-level NFL factor observations.

The builder is pure: callers provide already verified sports, reaction-path,
contract, and registry frames.  It neither reads artifacts nor falls back to
the older permissive target mask.  Every registered candidate is represented
in the exclusion audit; only candidates with a proven proposition target,
actual trades, unambiguous source time, and complete lineage enter the formal
observation frame.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from prediction_market.sports.nfl_factor_lab_cleaning import (
    semantic_rows_sha256,
)
from prediction_market.sports.nfl_factor_lab_types import (
    FactorDefinitionV1,
    FactorLabValidationError,
    PredicateClauseV1,
)


class FactorObservationBuildError(ValueError):
    """Verified inputs cannot produce a deterministic V2 factor table."""


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GRAIN = (
    "factor_id",
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
    "horizon_seconds",
)
_REACTION_GRAIN = _GRAIN[1:]
_CONTRACT_GRAIN = (
    "game_id",
    "logical_market_id",
    "outcome",
    "venue",
)
_PLAYER_FAMILIES = frozenset(
    {
        "player_passing",
        "player_rushing",
        "player_receiving",
        "player_receptions",
        "player_td",
        "first_td",
        "any_td",
    }
)
_TEAM_DIRECTIONAL_FAMILIES = frozenset(
    {"moneyline", "winner", "spread", "period"}
)
_SUPPORTED_FAMILIES = (
    _PLAYER_FAMILIES
    | _TEAM_DIRECTIONAL_FAMILIES
    | frozenset({"total", "team_total"})
)
_MEASURE_TO_PROGRESS_FIELD = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "touchdowns": "touchdowns",
    "any_touchdown": "touchdowns",
    "first_touchdown": "touchdowns",
}

_SPORTS_REQUIRED = frozenset(
    {
        "game_id",
        "play_id",
        "episode_id",
        "order_sequence",
        "source_time_utc",
        "event_eligible",
        "focal_team",
        "quarter",
        "sports_source_hashes",
    }
)
_REACTION_REQUIRED = frozenset(
    {
        *_REACTION_GRAIN,
        "event_source_time_utc",
        "first_post_trade_time_utc",
        "pre_event_actual_trade",
        "first_post_event_trade",
        "horizon_vwap",
        "horizon_last_trade",
        "signed_price_change",
        "maximum_excursion",
        "trade_count",
        "volume",
        "staleness_seconds",
        "actual_observation",
        "analysis_eligible",
        "analysis_exclusion_reason",
        "forward_filled",
        "order_ambiguous",
        "contaminated",
        "market_source_hashes",
    }
)
_CONTRACT_REQUIRED = frozenset(
    {
        *_CONTRACT_GRAIN,
        "contract_id",
        "family",
        "period",
        "subject",
        "measure",
        "kind",
        "analysis_eligible",
        "rule_sha256",
    }
)
_DEFINITION_FIELDS = frozenset(
    {
        "factor_id",
        "version",
        "sports_meaning",
        "predicate",
        "pit_fields",
        "focal_entity",
        "exclusions",
        "target",
        "market_families",
        "horizons_seconds",
        "support_kind",
        "minimum_episodes",
        "minimum_games",
        "minimum_episodes_per_venue",
        "claim_boundary",
    }
)


@dataclass(frozen=True, slots=True)
class FactorObservationBuildV2:
    """Deterministic in-memory result for content-addressed publication."""

    factor_observations: pd.DataFrame
    factor_exclusion_audit: pd.DataFrame
    factor_observations_sha256: str
    factor_exclusion_audit_sha256: str
    semantic_rows_sha256: str
    schema_version: str = "FactorObservationBuildV2"


def _require_columns(
    frame: pd.DataFrame,
    required: frozenset[str],
    *,
    label: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise FactorObservationBuildError(f"{label} must be a pandas DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FactorObservationBuildError(
            f"{label} missing columns: {', '.join(missing)}"
        )


def _require_nonmissing_identity(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    for column in columns:
        missing = frame[column].isna()
        if frame[column].dtype == object:
            missing |= frame[column].map(
                lambda value: type(value) is str and not value.strip()
            )
        if missing.any():
            raise FactorObservationBuildError(
                f"{label} has a missing {column}"
            )


def _clause(raw: object) -> PredicateClauseV1:
    if not isinstance(raw, Mapping) or set(raw) != {
        "field",
        "operator",
        "value",
    }:
        raise FactorObservationBuildError(
            "factor predicate clause has invalid fields"
        )
    value = raw["value"]
    if isinstance(value, list):
        value = tuple(value)
    try:
        return PredicateClauseV1(
            field=raw["field"],  # type: ignore[arg-type]
            operator=raw["operator"],  # type: ignore[arg-type]
            value=value,  # type: ignore[arg-type]
        )
    except FactorLabValidationError as error:
        raise FactorObservationBuildError(str(error)) from error


def _definition(raw: object) -> FactorDefinitionV1:
    if isinstance(raw, FactorDefinitionV1):
        return raw
    if not isinstance(raw, Mapping) or set(raw) != _DEFINITION_FIELDS:
        raise FactorObservationBuildError(
            "factor definition has invalid fields"
        )
    try:
        predicates = tuple(
            sorted(
                (
                    _clause(item)
                    for item in raw["predicate"]  # type: ignore[union-attr]
                ),
                key=lambda item: (
                    item.field,
                    item.operator,
                    repr(item.value),
                ),
            )
        )
        exclusions = tuple(
            sorted(
                (
                    _clause(item)
                    for item in raw["exclusions"]  # type: ignore[union-attr]
                ),
                key=lambda item: (
                    item.field,
                    item.operator,
                    repr(item.value),
                ),
            )
        )
        return FactorDefinitionV1(
            factor_id=raw["factor_id"],  # type: ignore[arg-type]
            version=raw["version"],  # type: ignore[arg-type]
            sports_meaning=raw["sports_meaning"],  # type: ignore[arg-type]
            predicate=predicates,
            pit_fields=tuple(
                sorted(raw["pit_fields"])  # type: ignore[arg-type]
            ),
            focal_entity=raw["focal_entity"],  # type: ignore[arg-type]
            exclusions=exclusions,
            target=raw["target"],  # type: ignore[arg-type]
            market_families=tuple(
                sorted(raw["market_families"])  # type: ignore[arg-type]
            ),
            horizons_seconds=tuple(
                sorted(raw["horizons_seconds"])  # type: ignore[arg-type]
            ),
            support_kind=raw["support_kind"],  # type: ignore[arg-type]
            minimum_episodes=raw["minimum_episodes"],  # type: ignore[arg-type]
            minimum_games=raw["minimum_games"],  # type: ignore[arg-type]
            minimum_episodes_per_venue=raw[
                "minimum_episodes_per_venue"
            ],  # type: ignore[arg-type]
            claim_boundary=raw["claim_boundary"],  # type: ignore[arg-type]
        )
    except (FactorLabValidationError, TypeError) as error:
        raise FactorObservationBuildError(str(error)) from error


def _definitions(
    registry: Sequence[Mapping[str, object] | FactorDefinitionV1],
) -> tuple[FactorDefinitionV1, ...]:
    if (
        not isinstance(registry, Sequence)
        or isinstance(registry, (str, bytes, bytearray))
        or not registry
    ):
        raise FactorObservationBuildError(
            "factor registry must be a nonempty sequence"
        )
    values = tuple(_definition(raw) for raw in registry)
    factor_ids = tuple(value.factor_id for value in values)
    if len(set(factor_ids)) != len(factor_ids):
        raise FactorObservationBuildError(
            "factor registry contains duplicate factor_id"
        )
    for value in values:
        used_fields = {
            clause.field
            for clause in (*value.predicate, *value.exclusions)
        }
        undeclared = sorted(used_fields.difference(value.pit_fields))
        if undeclared:
            raise FactorObservationBuildError(
                "predicate fields are not declared PIT inputs: "
                + ", ".join(undeclared)
            )
    return tuple(sorted(values, key=lambda value: value.factor_id))


def _predicate_mask(
    frame: pd.DataFrame,
    clauses: Sequence[PredicateClauseV1],
) -> pd.Series:
    result = pd.Series(True, index=frame.index, dtype=bool)
    for clause in clauses:
        values = frame[clause.field]
        if clause.operator == "eq":
            observed = values.eq(clause.value)
        elif clause.operator == "ne":
            observed = values.ne(clause.value)
        elif clause.operator == "lt":
            observed = values.lt(clause.value)
        elif clause.operator == "le":
            observed = values.le(clause.value)
        elif clause.operator == "gt":
            observed = values.gt(clause.value)
        elif clause.operator == "ge":
            observed = values.ge(clause.value)
        elif clause.operator == "in":
            observed = values.isin(clause.value)
        elif clause.operator == "not_in":
            observed = ~values.isin(clause.value)
        elif clause.operator == "is_null":
            observed = values.isna()
        elif clause.operator == "not_null":
            observed = values.notna()
        else:  # FactorDefinitionV1 has already rejected unknown operators.
            raise FactorObservationBuildError(
                f"unsupported predicate operator: {clause.operator}"
            )
        result &= observed.fillna(False)
    return result


def _hashes(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(
        type(item) is not str or _SHA256_RE.fullmatch(item) is None
        for item in value
    ):
        return None
    return tuple(sorted(set(value)))


def _is_true(value: object) -> bool:
    return type(value) is bool and value


def _finite(value: object) -> Decimal | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _present_text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _period_matches(period: object, quarter: object) -> bool:
    if type(period) is not str:
        return False
    if isinstance(quarter, bool) or not isinstance(quarter, Real):
        return False
    numeric_quarter = float(quarter)
    if not math.isfinite(numeric_quarter) or not numeric_quarter.is_integer():
        return False
    observed = int(numeric_quarter)
    canonical = period.casefold()
    if canonical in {"game", "full_game"}:
        return True
    if canonical == "first_half":
        return observed in {1, 2}
    if canonical == "second_half":
        return observed in {3, 4}
    if canonical in {"overtime", "ot"}:
        return observed >= 5
    quarter_aliases = {
        "q1": 1,
        "quarter_1": 1,
        "q2": 2,
        "quarter_2": 2,
        "q3": 3,
        "quarter_3": 3,
        "q4": 4,
        "quarter_4": 4,
    }
    return quarter_aliases.get(canonical) == observed


def _player_progress(
    row: Mapping[str, object],
    *,
    subject: object,
    measure: object,
) -> float | int | None:
    if type(subject) is not str or not subject:
        return None
    if type(measure) is not str:
        return None
    progress = row.get("player_stat_progress")
    if not isinstance(progress, Mapping):
        return None
    player = progress.get(subject)
    if not isinstance(player, Mapping):
        return None
    field = _MEASURE_TO_PROGRESS_FIELD.get(measure)
    if field is None:
        return None
    value = player.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _target_binding(
    row: Mapping[str, object],
) -> tuple[str | None, float | int | None, str | None]:
    family = row.get("family")
    if type(family) is not str or family not in _SUPPORTED_FAMILIES:
        return None, None, "unsupported_market_family"
    if not _period_matches(row.get("period"), row.get("quarter")):
        return None, None, "period_mismatch"
    focal_team = row.get("focal_team")
    outcome = row.get("outcome")
    subject = row.get("subject")
    if family in _TEAM_DIRECTIONAL_FAMILIES:
        if (
            type(focal_team) is not str
            or type(outcome) is not str
            or outcome != focal_team
        ):
            return None, None, "focal_team_outcome_mismatch"
        return "focal_team", None, None
    if family == "total":
        if type(outcome) is not str or outcome.casefold() != "over":
            return None, None, "total_outcome_not_over"
        return "over", None, None
    if family == "team_total":
        if type(subject) is not str or subject != focal_team:
            return None, None, "team_total_subject_mismatch"
        if type(outcome) is not str or outcome.casefold() != "over":
            return None, None, "team_total_outcome_not_over"
        return "subject_team_over", None, None
    if type(outcome) is not str or outcome.casefold() not in {"over", "yes"}:
        return None, None, "player_prop_outcome_not_affirmative"
    progress = _player_progress(
        row,
        subject=subject,
        measure=row.get("measure"),
    )
    if progress is None:
        return (
            None,
            None,
            "player_subject_or_stat_progress_unbound",
        )
    return "subject_player_affirmative", progress, None


def _market_evidence_reasons(row: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    actual_metrics = (
        "pre_event_actual_trade",
        "first_post_event_trade",
        "horizon_vwap",
        "horizon_last_trade",
        "signed_price_change",
        "maximum_excursion",
    )
    if not _is_true(row.get("actual_observation")) or any(
        _finite(row.get(field)) is None for field in actual_metrics
    ):
        reasons.append("no_actual_trade")
    if _is_true(row.get("order_ambiguous")):
        reasons.append("order_ambiguous")
    if _is_true(row.get("contaminated")):
        reasons.append("contaminated")
    if _is_true(row.get("forward_filled")):
        reasons.append("forward_filled")
    if not _present_text(row.get("event_source_time_utc")):
        reasons.append("missing_event_source_time")
    if not _present_text(row.get("first_post_trade_time_utc")):
        reasons.append("missing_first_post_trade_time")
    if _hashes(row.get("sports_source_hashes")) is None:
        reasons.append("sports_lineage_missing")
    if _hashes(row.get("market_source_hashes")) is None:
        reasons.append("market_lineage_missing")
    if (
        type(row.get("rule_sha256")) is not str
        or _SHA256_RE.fullmatch(row["rule_sha256"]) is None  # type: ignore[index]
    ):
        reasons.append("contract_rule_lineage_missing")
    trade_count = _finite(row.get("trade_count"))
    volume = _finite(row.get("volume"))
    staleness = _finite(row.get("staleness_seconds"))
    if (
        trade_count is None
        or trade_count <= 0
        or volume is None
        or volume < 0
        or staleness is None
        or staleness < 0
    ):
        reasons.append("invalid_market_metric")
    return reasons


def _plain(value: object) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise FactorObservationBuildError(
                "semantic timestamp must be timezone-aware"
            )
        return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise FactorObservationBuildError(
                "semantic datetime must be timezone-aware"
            )
        return value.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _plain(child)
            for key, child in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(child) for child in value]
    if pd.isna(value):
        return None
    raise FactorObservationBuildError(
        f"semantic row contains unsupported {type(value).__name__}"
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    return semantic_rows_sha256(
        [_plain(row) for row in frame.to_dict(orient="records")]
    )


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    observation_columns = [
        *_GRAIN,
        "factor_observation_id",
        "factor_version",
        "factor_definition_sha256",
        "factor_target",
        "support_kind",
        "play_id",
        "event_source_time_utc",
        "sports_play_source_time_utc",
        "first_post_trade_time_utc",
        "market_family",
        "reaction_analysis_eligible",
        "reaction_analysis_exclusion_reason",
        "target_binding",
        "subject_stat_progress",
        "predicate_hit",
        "exclusion_reasons",
        "sports_source_hashes",
        "market_source_hashes",
        "source_hashes",
        "claim_boundary",
    ]
    audit_columns = [
        "audit_id",
        *_GRAIN,
        "factor_version",
        "factor_definition_sha256",
        "play_id",
        "event_source_time_utc",
        "sports_play_source_time_utc",
        "first_post_trade_time_utc",
        "market_family",
        "reaction_analysis_eligible",
        "reaction_analysis_exclusion_reason",
        "target_binding",
        "subject_stat_progress",
        "predicate_hit",
        "decision",
        "exclusion_reasons",
        "sports_source_hashes",
        "market_source_hashes",
        "source_hashes",
        "claim_boundary",
    ]
    return (
        pd.DataFrame(columns=observation_columns),
        pd.DataFrame(columns=audit_columns),
    )


def _sort_frame(frame: pd.DataFrame, grain: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(list(grain), kind="mergesort").reset_index(
        drop=True
    )


def build_factor_observations_v2(
    sports_feature_view: pd.DataFrame,
    reaction_paths: pd.DataFrame,
    contract_inventory: pd.DataFrame,
    factor_registry: Sequence[
        Mapping[str, object] | FactorDefinitionV1
    ],
) -> FactorObservationBuildV2:
    """Construct formal observations and a complete candidate audit.

    The observation grain is
    ``factor × game × episode × logical market × outcome × venue × horizon``.
    A missing or unprovable target remains visible in the audit and never
    enters the formal observation frame.
    """

    _require_columns(
        sports_feature_view, _SPORTS_REQUIRED, label="sports feature view"
    )
    _require_columns(
        reaction_paths, _REACTION_REQUIRED, label="reaction paths"
    )
    _require_columns(
        contract_inventory,
        _CONTRACT_REQUIRED,
        label="contract inventory",
    )
    definitions = _definitions(factor_registry)
    sports = sports_feature_view.copy(deep=True)
    paths = reaction_paths.rename(
        columns={
            "analysis_eligible": "reaction_analysis_eligible",
            "analysis_exclusion_reason": (
                "reaction_analysis_exclusion_reason"
            ),
        }
    ).copy(deep=True)
    contracts = contract_inventory.copy(deep=True)

    _require_nonmissing_identity(
        sports,
        ("game_id", "play_id", "episode_id"),
        label="sports feature view",
    )
    _require_nonmissing_identity(
        paths, _REACTION_GRAIN, label="reaction paths"
    )
    _require_nonmissing_identity(
        contracts, _CONTRACT_GRAIN, label="contract inventory"
    )
    if sports.duplicated(["game_id", "play_id"]).any():
        raise FactorObservationBuildError(
            "sports feature view is not unique at game/play grain"
        )
    if paths.duplicated(list(_REACTION_GRAIN)).any():
        raise FactorObservationBuildError(
            "reaction paths are not unique at their formal grain"
        )
    if contracts.duplicated(list(_CONTRACT_GRAIN)).any():
        raise FactorObservationBuildError(
            "contract inventory is not unique at game/market/outcome/venue grain"
        )

    contract_columns = [
        *_CONTRACT_GRAIN,
        "contract_id",
        "family",
        "period",
        "subject",
        "measure",
        "kind",
        "analysis_eligible",
        "rule_sha256",
    ]
    path_contracts = paths.merge(
        contracts[contract_columns],
        on=list(_CONTRACT_GRAIN),
        how="left",
        validate="many_to_one",
        indicator="_contract_merge",
    )
    sports_columns = [
        column
        for column in sports.columns
        if column not in {"game_id", "episode_id"}
    ]

    observation_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for definition in definitions:
        candidate_paths = path_contracts[
            path_contracts["family"].isin(definition.market_families)
            & path_contracts["horizon_seconds"].isin(
                definition.horizons_seconds
            )
        ].copy()
        if candidate_paths.empty:
            continue
        joined = candidate_paths.merge(
            sports[["game_id", "episode_id", *sports_columns]],
            on=["game_id", "episode_id"],
            how="left",
            validate="many_to_many",
        )
        joined["factor_id"] = definition.factor_id
        candidate_key = list(_REACTION_GRAIN)

        missing_fields = sorted(
            {
                clause.field
                for clause in (*definition.predicate, *definition.exclusions)
                if clause.field not in joined.columns
            }
        )
        if missing_fields:
            joined["_predicate_hit"] = False
            joined["_registered_exclusion"] = False
        else:
            joined["_predicate_hit"] = _predicate_mask(
                joined, definition.predicate
            )
            joined["_registered_exclusion"] = (
                _predicate_mask(joined, definition.exclusions)
                if definition.exclusions
                else False
            )
        joined["_sports_present"] = joined["play_id"].notna()
        joined["_sports_eligible"] = joined["event_eligible"].map(_is_true)
        joined["_predicate_nonexcluded"] = (
            joined["_predicate_hit"] & ~joined["_registered_exclusion"]
        )
        joined["_eligible_factor_row"] = (
            joined["_sports_present"]
            & joined["_sports_eligible"]
            & joined["_predicate_nonexcluded"]
        )
        joined["_selection_priority"] = np.select(
            [
                joined["_eligible_factor_row"],
                joined["_predicate_hit"],
                joined["_sports_present"],
            ],
            [0, 1, 2],
            default=3,
        )
        joined["_order"] = pd.to_numeric(
            joined["order_sequence"], errors="coerce"
        ).fillna(float("inf"))
        joined["_play_sort"] = joined["play_id"].fillna("").astype(str)
        sort_columns = [
            *candidate_key,
            "_selection_priority",
            "_order",
            "_play_sort",
        ]
        ordered = joined.sort_values(sort_columns, kind="mergesort")
        selected = ordered.drop_duplicates(candidate_key, keep="first").copy()
        aggregate = joined.groupby(
            candidate_key, sort=False, dropna=False
        ).agg(
            predicate_hit=("_predicate_hit", "any"),
            predicate_nonexcluded_hit=("_predicate_nonexcluded", "any"),
            sports_present=("_sports_present", "any"),
            eligible_factor_row=("_eligible_factor_row", "any"),
        )
        selected = selected.merge(
            aggregate.reset_index(),
            on=candidate_key,
            how="left",
            validate="one_to_one",
        )
        for record in selected.to_dict(orient="records"):
            reasons: list[str] = []
            if record["_contract_merge"] != "both":
                reasons.append("contract_binding_missing")
            elif not _is_true(record.get("analysis_eligible")):
                reasons.append("contract_not_analysis_eligible")
            elif record.get("kind") != "primitive":
                reasons.append("contract_not_primitive")
            if not _is_true(record.get("reaction_analysis_eligible")):
                reasons.append("reaction_path_analysis_ineligible")
            if missing_fields:
                reasons.extend(
                    f"pit_field_unavailable:{field}"
                    for field in missing_fields
                )
            elif not record["sports_present"]:
                reasons.append("missing_sports_feature")
            elif not record["predicate_hit"]:
                reasons.append("predicate_not_matched")
            elif not record["eligible_factor_row"]:
                if not record["predicate_nonexcluded_hit"]:
                    reasons.append("registered_exclusion")
                else:
                    reasons.append("sports_event_ineligible")

            target_binding: str | None = None
            subject_progress: float | int | None = None
            if not missing_fields and record["sports_present"]:
                (
                    target_binding,
                    subject_progress,
                    target_reason,
                ) = _target_binding(record)
                if target_reason is not None:
                    reasons.append(target_reason)
            reasons.extend(_market_evidence_reasons(record))
            reasons = list(dict.fromkeys(reasons))

            sports_hashes = _hashes(record.get("sports_source_hashes"))
            market_hashes = _hashes(record.get("market_source_hashes"))
            rule_hash = (
                record.get("rule_sha256")
                if type(record.get("rule_sha256")) is str
                and _SHA256_RE.fullmatch(record["rule_sha256"]) is not None
                else None
            )
            source_hashes = tuple(
                sorted(
                    {
                        *(sports_hashes or ()),
                        *(market_hashes or ()),
                        *((rule_hash,) if rule_hash is not None else ()),
                    }
                )
            )
            identity = {
                "factor_id": definition.factor_id,
                "game_id": record["game_id"],
                "episode_id": record["episode_id"],
                "logical_market_id": record["logical_market_id"],
                "outcome": record["outcome"],
                "venue": record["venue"],
                "horizon_seconds": int(record["horizon_seconds"]),
            }
            identity_hash = semantic_rows_sha256(identity)
            common = {
                **identity,
                "factor_version": definition.version,
                "factor_definition_sha256": definition.definition_sha256,
                "factor_target": definition.target,
                "support_kind": definition.support_kind,
                "play_id": record.get("play_id"),
                "event_source_time_utc": record.get(
                    "event_source_time_utc"
                ),
                "sports_play_source_time_utc": record.get(
                    "source_time_utc"
                ),
                "first_post_trade_time_utc": record.get(
                    "first_post_trade_time_utc"
                ),
                "market_family": record.get("family"),
                "reaction_analysis_eligible": record.get(
                    "reaction_analysis_eligible"
                ),
                "reaction_analysis_exclusion_reason": record.get(
                    "reaction_analysis_exclusion_reason"
                ),
                "contract_id": record.get("contract_id"),
                "contract_period": record.get("period"),
                "contract_subject": record.get("subject"),
                "contract_measure": record.get("measure"),
                "target_binding": target_binding,
                "subject_stat_progress": subject_progress,
                "predicate_hit": bool(record["predicate_hit"]),
                "exclusion_reasons": tuple(reasons),
                "sports_source_hashes": sports_hashes or (),
                "market_source_hashes": market_hashes or (),
                "source_hashes": source_hashes,
                "claim_boundary": definition.claim_boundary,
            }
            audit_rows.append(
                {
                    "audit_id": identity_hash,
                    **common,
                    "decision": "INCLUDE" if not reasons else "EXCLUDE",
                }
            )
            if reasons:
                continue
            observation = {
                **{
                    key: value
                    for key, value in record.items()
                    if not key.startswith("_")
                    and key
                    not in {
                        "factor_id",
                        "source_time_utc",
                        "family",
                        "period",
                        "subject",
                        "measure",
                    }
                },
                "factor_observation_id": identity_hash,
                **common,
            }
            observation_rows.append(observation)

    if observation_rows:
        observations = pd.DataFrame(observation_rows)
        observations = _sort_frame(observations, _GRAIN)
        if observations.duplicated(list(_GRAIN)).any():
            raise FactorObservationBuildError(
                "factor observations are not unique at their formal grain"
            )
        leading = [
            *_GRAIN,
            "factor_observation_id",
            "factor_version",
            "factor_definition_sha256",
            "factor_target",
            "support_kind",
            "play_id",
            "event_source_time_utc",
            "sports_play_source_time_utc",
            "first_post_trade_time_utc",
            "market_family",
            "reaction_analysis_eligible",
            "reaction_analysis_exclusion_reason",
            "target_binding",
            "subject_stat_progress",
            "predicate_hit",
            "exclusion_reasons",
            "sports_source_hashes",
            "market_source_hashes",
            "source_hashes",
            "claim_boundary",
        ]
        observations = observations[
            [*leading, *sorted(set(observations.columns).difference(leading))]
        ]
    else:
        observations, _ = _empty_frames()
    if audit_rows:
        audit = _sort_frame(pd.DataFrame(audit_rows), _GRAIN)
        if audit.duplicated(list(_GRAIN)).any():
            raise FactorObservationBuildError(
                "factor exclusion audit is not unique at candidate grain"
            )
        leading_audit = [
            "audit_id",
            *_GRAIN,
            "factor_version",
            "factor_definition_sha256",
            "play_id",
            "event_source_time_utc",
            "sports_play_source_time_utc",
            "first_post_trade_time_utc",
            "market_family",
            "reaction_analysis_eligible",
            "reaction_analysis_exclusion_reason",
            "target_binding",
            "subject_stat_progress",
            "predicate_hit",
            "decision",
            "exclusion_reasons",
            "sports_source_hashes",
            "market_source_hashes",
            "source_hashes",
            "claim_boundary",
        ]
        audit = audit[
            [*leading_audit, *sorted(set(audit.columns).difference(leading_audit))]
        ]
    else:
        _, audit = _empty_frames()

    observations_hash = _frame_hash(observations)
    audit_hash = _frame_hash(audit)
    combined_hash = semantic_rows_sha256(
        {
            "schema_version": "FactorObservationBuildV2",
            "factor_observations_sha256": observations_hash,
            "factor_exclusion_audit_sha256": audit_hash,
        }
    )
    return FactorObservationBuildV2(
        factor_observations=observations,
        factor_exclusion_audit=audit,
        factor_observations_sha256=observations_hash,
        factor_exclusion_audit_sha256=audit_hash,
        semantic_rows_sha256=combined_hash,
    )


__all__ = [
    "FactorObservationBuildError",
    "FactorObservationBuildV2",
    "build_factor_observations_v2",
]
