"""Frozen value types for the NFL Sports Factor Lab."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence, TypeAlias


class FactorLabValidationError(ValueError):
    """A Factor Lab value does not satisfy the frozen V1 contract."""


PredicateOperator = Literal[
    "eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is_null", "not_null"
]
FocalEntity = Literal["team", "player", "possession", "game"]
SupportKind = Literal["primary", "binary", "named", "sequence", "descriptive"]
FactorResultSupportStatus = Literal[
    "eligible", "insufficient_support", "data_gap", "case_only"
]
PredicateScalar: TypeAlias = str | int | float | bool
PredicateValue: TypeAlias = PredicateScalar | tuple[PredicateScalar, ...] | None

_OPERATORS = frozenset(
    {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is_null", "not_null"}
)
_FOCAL_ENTITIES = frozenset({"team", "player", "possession", "game"})
_SUPPORT_KINDS = frozenset(
    {"primary", "binary", "named", "sequence", "descriptive"}
)
_FIELD_ZONES = frozenset({"own", "midfield", "opponent", "red_zone", "goal_line"})
_SEASON_TYPES = frozenset({"REG", "POST"})
_HORIZONS_SECONDS = frozenset({1, 2, 5, 10, 30, 60})
_DELAY_SECONDS = frozenset({0, 1, 2, 3, 5, 10})
_RESULT_SUPPORT_STATUSES = frozenset(
    {"eligible", "insufficient_support", "data_gap", "case_only"}
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")


def _hash_value(value: object) -> object:
    if type(value) is float:
        if not math.isfinite(value):
            raise FactorLabValidationError("canonical hash contains non-finite float")
        return {"$decimal": str(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _hash_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_hash_value(child) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise FactorLabValidationError(
        f"canonical hash contains unsupported {type(value).__name__}"
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _hash_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise FactorLabValidationError(f"{field} must be a stable identifier")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise FactorLabValidationError(f"{field} must be nonempty trimmed text")
    return value


def _finite_optional(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorLabValidationError(f"{field} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise FactorLabValidationError(f"{field} must be finite")
    return number


def _integer_optional(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FactorLabValidationError(f"{field} must be an integer or null")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


def _source_hashes(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise FactorLabValidationError("source_hashes must be a nonempty tuple")
    if value != tuple(sorted(value)) or len(set(value)) != len(value):
        raise FactorLabValidationError(
            "source_hashes must be sorted and unique"
        )
    for digest in value:
        if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
            raise FactorLabValidationError(
                "source_hashes must contain sha256 digests"
            )
    return value


def _probability(value: object, *, field: str) -> float:
    number = _finite_optional(value, field=field)
    if number is None or not 0 <= number <= 1:
        raise FactorLabValidationError(f"{field} must be between 0 and 1")
    return number


def _utc(value: object, *, field: str, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value.endswith("Z"):
        raise FactorLabValidationError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise FactorLabValidationError(
            f"{field} must be an RFC3339 UTC timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise FactorLabValidationError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _predicate_scalar(value: object, *, field: str) -> PredicateScalar:
    if type(value) not in {str, int, float, bool}:
        raise FactorLabValidationError(f"{field} contains an unsupported value")
    if type(value) is str and (not value or value != value.strip()):
        raise FactorLabValidationError(f"{field} string must be nonempty and trimmed")
    if type(value) is float and not math.isfinite(value):
        raise FactorLabValidationError(f"{field} must be finite")
    return value  # type: ignore[return-value]


def _predicate_sort_key(value: PredicateScalar) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


@dataclass(frozen=True, slots=True)
class PredicateClauseV1:
    """One typed conjunct in a registered factor predicate."""

    field: str
    operator: PredicateOperator
    value: PredicateValue

    def __post_init__(self) -> None:
        _identifier(self.field, field="predicate field")
        if self.operator not in _OPERATORS:
            raise FactorLabValidationError("predicate operator is unsupported")
        if self.operator in {"is_null", "not_null"}:
            if self.value is not None:
                raise FactorLabValidationError("null operator value must be null")
            return
        if self.operator in {"in", "not_in"}:
            if type(self.value) is not tuple or not self.value:
                raise FactorLabValidationError(
                    "membership predicate value must be a nonempty tuple"
                )
            canonical = tuple(
                _predicate_scalar(value, field="predicate value")
                for value in self.value
            )
            if canonical != tuple(sorted(canonical, key=_predicate_sort_key)):
                raise FactorLabValidationError(
                    "membership predicate values must use canonical order"
                )
            if len(set((type(value).__name__, value) for value in canonical)) != len(
                canonical
            ):
                raise FactorLabValidationError(
                    "membership predicate values must be unique"
                )
            return
        if isinstance(self.value, tuple) or self.value is None:
            raise FactorLabValidationError(
                "comparison predicate requires one scalar value"
            )
        _predicate_scalar(self.value, field="predicate value")

    def to_dict(self) -> dict[str, object]:
        value: object = (
            list(self.value) if isinstance(self.value, tuple) else self.value
        )
        return {"field": self.field, "operator": self.operator, "value": value}


def _clause_sort_key(clause: PredicateClauseV1) -> tuple[str, str, str]:
    return (clause.field, clause.operator, repr(clause.value))


@dataclass(frozen=True, slots=True)
class FactorDefinitionV1:
    """Immutable, hashable definition of one predeclared sports factor."""

    factor_id: str
    version: str
    sports_meaning: str
    predicate: tuple[PredicateClauseV1, ...]
    pit_fields: tuple[str, ...]
    focal_entity: FocalEntity
    exclusions: tuple[PredicateClauseV1, ...]
    target: str
    market_families: tuple[str, ...]
    horizons_seconds: tuple[int, ...]
    support_kind: SupportKind
    minimum_episodes: int
    minimum_games: int
    minimum_episodes_per_venue: int = 0
    claim_boundary: str = "PRELIMINARY_SOURCE_TIME_ONLY"

    def __post_init__(self) -> None:
        _identifier(self.factor_id, field="factor_id")
        _identifier(self.version, field="version")
        _text(self.sports_meaning, field="sports_meaning")
        _identifier(self.target, field="target")
        _identifier(self.claim_boundary, field="claim_boundary")
        if self.focal_entity not in _FOCAL_ENTITIES:
            raise FactorLabValidationError("focal_entity is unsupported")
        if self.support_kind not in _SUPPORT_KINDS:
            raise FactorLabValidationError("support_kind is unsupported")
        if not self.predicate:
            raise FactorLabValidationError("predicate must contain at least one clause")
        for collection, name in (
            (self.predicate, "predicate"),
            (self.exclusions, "exclusions"),
        ):
            if any(not isinstance(item, PredicateClauseV1) for item in collection):
                raise FactorLabValidationError(f"{name} must contain V1 clauses")
            if collection != tuple(sorted(collection, key=_clause_sort_key)):
                raise FactorLabValidationError(f"{name} must use canonical order")
            keys = [_clause_sort_key(item) for item in collection]
            if len(set(keys)) != len(keys):
                raise FactorLabValidationError(f"{name} clauses must be unique")
        for collection, name in (
            (self.pit_fields, "pit_fields"),
            (self.market_families, "market_families"),
        ):
            if not collection:
                raise FactorLabValidationError(f"{name} must not be empty")
            for value in collection:
                _identifier(value, field=name)
            if collection != tuple(sorted(collection)):
                raise FactorLabValidationError(f"{name} must use canonical order")
            if len(set(collection)) != len(collection):
                raise FactorLabValidationError(f"{name} must be unique")
        if (
            not self.horizons_seconds
            or self.horizons_seconds != tuple(sorted(self.horizons_seconds))
            or len(set(self.horizons_seconds)) != len(self.horizons_seconds)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.horizons_seconds
            )
        ):
            raise FactorLabValidationError(
                "horizons_seconds must be sorted unique positive integers"
            )
        for value, name, minimum in (
            (self.minimum_episodes, "minimum_episodes", 1),
            (self.minimum_games, "minimum_games", 1),
            (
                self.minimum_episodes_per_venue,
                "minimum_episodes_per_venue",
                0,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise FactorLabValidationError(f"{name} must be >= {minimum}")
        if "turnover_over_expected" in repr(self.to_dict()).lower():
            raise FactorLabValidationError(
                "turnover_over_expected requires an unavailable PIT probability"
            )

    @property
    def definition_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "FactorDefinitionV1",
            "factor_id": self.factor_id,
            "version": self.version,
            "sports_meaning": self.sports_meaning,
            "predicate": [item.to_dict() for item in self.predicate],
            "pit_fields": list(self.pit_fields),
            "focal_entity": self.focal_entity,
            "exclusions": [item.to_dict() for item in self.exclusions],
            "target": self.target,
            "market_families": list(self.market_families),
            "horizons_seconds": list(self.horizons_seconds),
            "support_kind": self.support_kind,
            "minimum_episodes": self.minimum_episodes,
            "minimum_games": self.minimum_games,
            "minimum_episodes_per_venue": self.minimum_episodes_per_venue,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PlayerRoleV1:
    role: str
    player_id: str | None
    player_name: str | None

    def __post_init__(self) -> None:
        _identifier(self.role, field="player role")
        if self.player_id is not None:
            _text(self.player_id, field="player_id")
        if self.player_name is not None:
            _text(self.player_name, field="player_name")
        if self.player_id is None and self.player_name is None:
            raise FactorLabValidationError(
                "player role requires a player ID or player name"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "player_id": self.player_id,
            "player_name": self.player_name,
        }


@dataclass(frozen=True, slots=True)
class RollingFeatureV1:
    """One PIT rolling feature with both requested lookback semantics."""

    metric_id: str
    entity_type: Literal["team", "player"]
    entity_id: str
    season_to_date_value: float | None
    season_to_date_observations: int
    trailing_8_value: float | None
    trailing_8_observations: int
    low_support: bool

    def __post_init__(self) -> None:
        _identifier(self.metric_id, field="metric_id")
        if self.entity_type not in {"team", "player"}:
            raise FactorLabValidationError("rolling entity_type is unsupported")
        _text(self.entity_id, field="rolling entity_id")
        season_value = _finite_optional(
            self.season_to_date_value, field="season_to_date_value"
        )
        trailing_value = _finite_optional(
            self.trailing_8_value, field="trailing_8_value"
        )
        for count, name, value in (
            (
                self.season_to_date_observations,
                "season_to_date_observations",
                season_value,
            ),
            (
                self.trailing_8_observations,
                "trailing_8_observations",
                trailing_value,
            ),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise FactorLabValidationError(f"{name} must be a nonnegative integer")
            if (count == 0) != (value is None):
                raise FactorLabValidationError(
                    f"{name} and its value have inconsistent missingness"
                )
        if type(self.low_support) is not bool:
            raise FactorLabValidationError("low_support must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "season_to_date_value": self.season_to_date_value,
            "season_to_date_observations": self.season_to_date_observations,
            "trailing_8_value": self.trailing_8_value,
            "trailing_8_observations": self.trailing_8_observations,
            "low_support": self.low_support,
        }


@dataclass(frozen=True, slots=True)
class BetweenPlayRosterChangeV1:
    """Observed roster-set difference bounded between adjacent plays.

    This is intentionally not called a substitution: the source does not
    prove the exact change time or the administrative reason for the change.
    """

    offense_entering_names: tuple[str, ...]
    offense_leaving_names: tuple[str, ...]
    defense_entering_names: tuple[str, ...]
    defense_leaving_names: tuple[str, ...]
    semantics: str = (
        "set difference between consecutive observed participation rows; "
        "timing is bounded between plays"
    )

    def __post_init__(self) -> None:
        for values, name in (
            (self.offense_entering_names, "offense_entering_names"),
            (self.offense_leaving_names, "offense_leaving_names"),
            (self.defense_entering_names, "defense_entering_names"),
            (self.defense_leaving_names, "defense_leaving_names"),
        ):
            if values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise FactorLabValidationError(
                    f"{name} must use canonical order and be unique"
                )
            for value in values:
                _text(value, field=name)
        if self.semantics != (
            "set difference between consecutive observed participation rows; "
            "timing is bounded between plays"
        ):
            raise FactorLabValidationError(
                "between-play roster-change semantics cannot be widened"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "offense_entering_names": list(self.offense_entering_names),
            "offense_leaving_names": list(self.offense_leaving_names),
            "defense_entering_names": list(self.defense_entering_names),
            "defense_leaving_names": list(self.defense_leaving_names),
            "semantics": self.semantics,
        }


@dataclass(frozen=True, slots=True)
class SportsFeatureRowV1:
    """One play joined to exactly one finalized episode."""

    game_id: str
    play_id: str
    episode_id: str
    order_sequence: int
    season: int
    season_type: str
    week: int
    kickoff_at_utc: str
    source_time_utc: str | None
    description: str | None
    home_team: str
    away_team: str
    focal_team: str
    possession_team: str | None
    offense_team: str | None
    defense_team: str | None
    quarter: int
    quarter_seconds_remaining: int | None
    half_seconds_remaining: int | None
    game_seconds_remaining: int | None
    home_score: int
    away_score: int
    score_margin_focal: int
    tied: bool
    one_score_game: bool
    multi_score_game: bool
    down: int | None
    distance: int | None
    yardline_100: int | None
    field_zone: str | None
    red_zone: bool | None
    goal_to_go: bool | None
    drive_id: str | None
    drive_play_index: int | None
    drive_yards_before: float | None
    drive_progression_yards_before: float | None
    home_timeouts_remaining: int | None
    away_timeouts_remaining: int | None
    play_type: str | None
    event_flags: tuple[str, ...]
    event_eligible: bool
    episode_nullified: bool
    episode_audit_only: bool
    player_roles: tuple[PlayerRoleV1, ...]
    formation: str | None
    offense_personnel: str | None
    defense_personnel: str | None
    offense_player_ids: tuple[str, ...]
    defense_player_ids: tuple[str, ...]
    offense_player_names: tuple[str, ...]
    defense_player_names: tuple[str, ...]
    shotgun: bool | None
    no_huddle: bool | None
    yards_gained: float | None
    wp_focal: float | None
    vegas_wp_focal: float | None
    wpa_focal: float | None
    realized_delta_wp_focal: float | None
    epa: float | None
    qb_epa: float | None
    xpass: float | None
    pass_oe: float | None
    cp: float | None
    cpoe: float | None
    yac_epa: float | None
    xyac_epa: float | None
    yac_surprise: float | None
    rolling_features: tuple[RollingFeatureV1, ...]
    factor_values: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    defensive_two_point_attempt: bool | None = None
    defensive_two_point_conversion: bool | None = None
    kicker_player_id: str | None = None
    kicker_player_name: str | None = None
    penalty_team: str | None = None
    penalty_type: str | None = None
    penalty_yards: float | None = None
    penalty_status: str | None = None
    timeout_team: str | None = None
    punt_blocked: bool | None = None
    punt_inside_twenty: bool | None = None
    punt_in_endzone: bool | None = None
    punt_out_of_bounds: bool | None = None
    punt_downed: bool | None = None
    punt_fair_catch: bool | None = None
    kickoff_inside_twenty: bool | None = None
    kickoff_in_endzone: bool | None = None
    kickoff_out_of_bounds: bool | None = None
    kickoff_downed: bool | None = None
    kickoff_fair_catch: bool | None = None
    own_kickoff_recovery: bool | None = None
    own_kickoff_recovery_touchdown: bool | None = None
    touchback: bool | None = None
    return_team: str | None = None
    return_yards: float | None = None
    fumble_recovery_team: str | None = None
    fumble_recovery_yards: float | None = None
    between_play_roster_change: BetweenPlayRosterChangeV1 | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.game_id, "game_id"),
            (self.play_id, "play_id"),
            (self.episode_id, "episode_id"),
        ):
            _identifier(value, field=name)
        for team, name in (
            (self.home_team, "home_team"),
            (self.away_team, "away_team"),
            (self.focal_team, "focal_team"),
        ):
            _text(team, field=name)
        if self.home_team == self.away_team:
            raise FactorLabValidationError("home_team and away_team must differ")
        if self.focal_team not in {self.home_team, self.away_team}:
            raise FactorLabValidationError("focal_team must be one of the game teams")
        for team, name in (
            (self.possession_team, "possession_team"),
            (self.offense_team, "offense_team"),
            (self.defense_team, "defense_team"),
        ):
            if team is not None and team not in {self.home_team, self.away_team}:
                raise FactorLabValidationError(f"{name} must be a game team or null")
        for value, name, minimum in (
            (self.order_sequence, "order_sequence", 0),
            (self.season, "season", 1999),
            (self.week, "week", 1),
            (self.quarter, "quarter", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise FactorLabValidationError(f"{name} is invalid")
        if self.season_type not in _SEASON_TYPES:
            raise FactorLabValidationError("season_type must be REG or POST")
        kickoff = _utc(self.kickoff_at_utc, field="kickoff_at_utc")
        source = _utc(self.source_time_utc, field="source_time_utc", optional=True)
        if source is not None and kickoff is not None and source < kickoff:
            raise FactorLabValidationError(
                "source_time_utc cannot precede kickoff_at_utc"
            )
        if self.description is not None:
            _text(self.description, field="description")
        for value, name in (
            (self.quarter_seconds_remaining, "quarter_seconds_remaining"),
            (self.half_seconds_remaining, "half_seconds_remaining"),
            (self.game_seconds_remaining, "game_seconds_remaining"),
            (self.down, "down"),
            (self.distance, "distance"),
            (self.yardline_100, "yardline_100"),
            (self.drive_play_index, "drive_play_index"),
            (self.home_timeouts_remaining, "home_timeouts_remaining"),
            (self.away_timeouts_remaining, "away_timeouts_remaining"),
        ):
            number = _integer_optional(value, field=name)
            if number is not None and number < 0:
                raise FactorLabValidationError(f"{name} must be nonnegative")
        if self.down is not None and self.down not in {0, 1, 2, 3, 4}:
            raise FactorLabValidationError("down is outside the NFL state space")
        if self.yardline_100 is not None and not 0 <= self.yardline_100 <= 100:
            raise FactorLabValidationError("yardline_100 must be between 0 and 100")
        if self.field_zone is not None and self.field_zone not in _FIELD_ZONES:
            raise FactorLabValidationError("field_zone is unsupported")
        for value, name in (
            (self.tied, "tied"),
            (self.one_score_game, "one_score_game"),
            (self.multi_score_game, "multi_score_game"),
            (self.event_eligible, "event_eligible"),
            (self.episode_nullified, "episode_nullified"),
            (self.episode_audit_only, "episode_audit_only"),
        ):
            if type(value) is not bool:
                raise FactorLabValidationError(f"{name} must be boolean")
        for value, name in (
            (
                self.defensive_two_point_attempt,
                "defensive_two_point_attempt",
            ),
            (
                self.defensive_two_point_conversion,
                "defensive_two_point_conversion",
            ),
            (self.punt_blocked, "punt_blocked"),
            (self.punt_inside_twenty, "punt_inside_twenty"),
            (self.punt_in_endzone, "punt_in_endzone"),
            (self.punt_out_of_bounds, "punt_out_of_bounds"),
            (self.punt_downed, "punt_downed"),
            (self.punt_fair_catch, "punt_fair_catch"),
            (self.kickoff_inside_twenty, "kickoff_inside_twenty"),
            (self.kickoff_in_endzone, "kickoff_in_endzone"),
            (self.kickoff_out_of_bounds, "kickoff_out_of_bounds"),
            (self.kickoff_downed, "kickoff_downed"),
            (self.kickoff_fair_catch, "kickoff_fair_catch"),
            (self.own_kickoff_recovery, "own_kickoff_recovery"),
            (
                self.own_kickoff_recovery_touchdown,
                "own_kickoff_recovery_touchdown",
            ),
            (self.touchback, "touchback"),
        ):
            if value is not None and type(value) is not bool:
                raise FactorLabValidationError(f"{name} must be boolean or null")
        if sum((self.one_score_game, self.multi_score_game)) != 1:
            raise FactorLabValidationError(
                "exactly one score-margin bucket must be active"
            )
        if self.tied != (self.score_margin_focal == 0):
            raise FactorLabValidationError("tied disagrees with score_margin_focal")
        if self.event_flags != tuple(sorted(self.event_flags)):
            raise FactorLabValidationError("event_flags must use canonical order")
        if len(set(self.event_flags)) != len(self.event_flags):
            raise FactorLabValidationError("event_flags must be unique")
        for flag in self.event_flags:
            _identifier(flag, field="event flag")
        if self.player_roles != tuple(
            sorted(
                self.player_roles,
                key=lambda role: (
                    role.role,
                    role.player_id or "",
                    role.player_name or "",
                ),
            )
        ):
            raise FactorLabValidationError("player_roles must use canonical order")
        for values, name in (
            (self.offense_player_ids, "offense_player_ids"),
            (self.defense_player_ids, "defense_player_ids"),
            (self.offense_player_names, "offense_player_names"),
            (self.defense_player_names, "defense_player_names"),
        ):
            if values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise FactorLabValidationError(f"{name} must be sorted and unique")
            for value in values:
                _text(value, field=name)
        for value, name in (
            (self.drive_yards_before, "drive_yards_before"),
            (self.drive_progression_yards_before, "drive_progression_yards_before"),
            (self.yards_gained, "yards_gained"),
            (self.wpa_focal, "wpa_focal"),
            (self.realized_delta_wp_focal, "realized_delta_wp_focal"),
            (self.epa, "epa"),
            (self.qb_epa, "qb_epa"),
            (self.xpass, "xpass"),
            (self.pass_oe, "pass_oe"),
            (self.cpoe, "cpoe"),
            (self.yac_epa, "yac_epa"),
            (self.xyac_epa, "xyac_epa"),
            (self.yac_surprise, "yac_surprise"),
            (self.penalty_yards, "penalty_yards"),
            (self.return_yards, "return_yards"),
            (self.fumble_recovery_yards, "fumble_recovery_yards"),
        ):
            _finite_optional(value, field=name)
        for value, name in (
            (self.kicker_player_id, "kicker_player_id"),
            (self.kicker_player_name, "kicker_player_name"),
            (self.penalty_type, "penalty_type"),
            (self.penalty_status, "penalty_status"),
        ):
            _optional_text(value, field=name)
        for team, name in (
            (self.penalty_team, "penalty_team"),
            (self.timeout_team, "timeout_team"),
            (self.return_team, "return_team"),
            (self.fumble_recovery_team, "fumble_recovery_team"),
        ):
            if team is not None and team not in {self.home_team, self.away_team}:
                raise FactorLabValidationError(
                    f"{name} must be one of the game teams or null"
                )
        if (
            self.between_play_roster_change is not None
            and not isinstance(
                self.between_play_roster_change, BetweenPlayRosterChangeV1
            )
        ):
            raise FactorLabValidationError(
                "between_play_roster_change must be V1 or null"
            )
        for value, name in (
            (self.wp_focal, "wp_focal"),
            (self.vegas_wp_focal, "vegas_wp_focal"),
            (self.cp, "cp"),
        ):
            number = _finite_optional(value, field=name)
            if number is not None and not 0 <= number <= 1:
                raise FactorLabValidationError(f"{name} must be between 0 and 1")
        if self.rolling_features != tuple(
            sorted(
                self.rolling_features,
                key=lambda item: (item.entity_type, item.entity_id, item.metric_id),
            )
        ):
            raise FactorLabValidationError(
                "rolling_features must use canonical order"
            )
        rolling_keys = [
            (item.entity_type, item.entity_id, item.metric_id)
            for item in self.rolling_features
        ]
        if len(set(rolling_keys)) != len(rolling_keys):
            raise FactorLabValidationError("rolling_features must be unique")
        if any(
            "turnover_over_expected" in item.metric_id
            for item in self.rolling_features
        ):
            raise FactorLabValidationError(
                "turnover_over_expected is unavailable without a PIT model"
            )
        if self.factor_values != tuple(
            sorted(self.factor_values, key=lambda item: item[0])
        ):
            raise FactorLabValidationError("factor_values must use canonical order")
        factor_names = [name for name, _value in self.factor_values]
        if len(set(factor_names)) != len(factor_names):
            raise FactorLabValidationError("factor_values must be unique")
        for name, value in self.factor_values:
            _identifier(name, field="factor value name")
            if value is not None and type(value) not in {str, int, float, bool}:
                raise FactorLabValidationError("factor value has unsupported type")
            if type(value) is float and not math.isfinite(value):
                raise FactorLabValidationError("factor value must be finite")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "player_roles":
                result[item.name] = [role.to_dict() for role in value]
            elif item.name == "rolling_features":
                result[item.name] = [feature.to_dict() for feature in value]
            elif item.name == "between_play_roster_change":
                result[item.name] = (
                    None if value is None else value.to_dict()
                )
            elif item.name == "factor_values":
                result[item.name] = {
                    name: child for name, child in value
                }
                result.update(result[item.name])
            elif isinstance(value, tuple):
                result[item.name] = list(value)
            else:
                result[item.name] = value
        return result


@dataclass(frozen=True, slots=True)
class FactorObservationV1:
    """One actual-trade observation at the registered analysis grain."""

    experiment_id: str
    factor_id: str
    factor_version: str
    factor_definition_sha256: str
    game_id: str
    episode_id: str
    logical_market_id: str
    outcome: str
    venue: str
    market_family: str
    horizon_seconds: int
    event_source_time_utc: str
    first_post_trade_time_utc: str
    pre_event_actual_trade: float
    first_post_event_trade: float
    horizon_vwap: float
    horizon_last_trade: float
    signed_price_change: float
    maximum_excursion: float
    continuation_10_to_60: float | None
    trade_count: int
    volume: float
    staleness_seconds: float
    order_ambiguous: bool
    contaminated: bool
    source_hashes: tuple[str, ...]
    claim_boundary: str = "PRELIMINARY_SOURCE_TIME_ONLY"

    def __post_init__(self) -> None:
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.factor_id, "factor_id"),
            (self.factor_version, "factor_version"),
            (self.game_id, "game_id"),
            (self.episode_id, "episode_id"),
            (self.venue, "venue"),
            (self.market_family, "market_family"),
            (self.claim_boundary, "claim_boundary"),
        ):
            _identifier(value, field=name)
        for value, name in (
            (self.logical_market_id, "logical_market_id"),
            (self.outcome, "outcome"),
        ):
            _text(value, field=name)
        if _SHA256_RE.fullmatch(self.factor_definition_sha256) is None:
            raise FactorLabValidationError(
                "factor_definition_sha256 must be a sha256 digest"
            )
        if self.horizon_seconds not in _HORIZONS_SECONDS:
            raise FactorLabValidationError("horizon_seconds is not registered")
        event_time = _utc(
            self.event_source_time_utc, field="event_source_time_utc"
        )
        trade_time = _utc(
            self.first_post_trade_time_utc,
            field="first_post_trade_time_utc",
        )
        if event_time is not None and trade_time is not None and trade_time < event_time:
            raise FactorLabValidationError(
                "first_post_trade_time_utc precedes the event source time"
            )
        for value, name in (
            (self.pre_event_actual_trade, "pre_event_actual_trade"),
            (self.first_post_event_trade, "first_post_event_trade"),
            (self.horizon_vwap, "horizon_vwap"),
            (self.horizon_last_trade, "horizon_last_trade"),
        ):
            _probability(value, field=name)
        for value, name in (
            (self.signed_price_change, "signed_price_change"),
            (self.maximum_excursion, "maximum_excursion"),
            (self.continuation_10_to_60, "continuation_10_to_60"),
            (self.volume, "volume"),
            (self.staleness_seconds, "staleness_seconds"),
        ):
            number = _finite_optional(value, field=name)
            if name in {"volume", "staleness_seconds"} and (
                number is None or number < 0
            ):
                raise FactorLabValidationError(f"{name} must be nonnegative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count <= 0
        ):
            raise FactorLabValidationError("trade_count must be positive")
        for value, name in (
            (self.order_ambiguous, "order_ambiguous"),
            (self.contaminated, "contaminated"),
        ):
            if type(value) is not bool:
                raise FactorLabValidationError(f"{name} must be boolean")
        _source_hashes(self.source_hashes)

    def to_dict(self) -> dict[str, object]:
        result = {
            item.name: getattr(self, item.name) for item in fields(self)
        }
        result["schema_version"] = "FactorObservationV1"
        result["source_hashes"] = list(self.source_hashes)
        return result


@dataclass(frozen=True, slots=True)
class FactorResultV1:
    """One governed cross-game factor result."""

    experiment_id: str
    factor_id: str
    factor_version: str
    factor_definition_sha256: str
    market_family: str
    horizon_seconds: int
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    game_count: int
    episode_count: int
    venue_episode_counts: tuple[tuple[str, int], ...]
    bootstrap_samples_requested: int
    bootstrap_samples_valid: int
    leave_one_game_out_same_sign_rate: float | None
    max_single_game_contribution: float | None
    bh_q_value: float | None
    venue_sign_consistent: bool | None
    support_status: FactorResultSupportStatus
    source_hashes: tuple[str, ...]
    claim_boundary: str = "PRELIMINARY_SOURCE_TIME_ONLY"

    def __post_init__(self) -> None:
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.factor_id, "factor_id"),
            (self.factor_version, "factor_version"),
            (self.market_family, "market_family"),
            (self.claim_boundary, "claim_boundary"),
        ):
            _identifier(value, field=name)
        if _SHA256_RE.fullmatch(self.factor_definition_sha256) is None:
            raise FactorLabValidationError(
                "factor_definition_sha256 must be a sha256 digest"
            )
        if self.horizon_seconds not in _HORIZONS_SECONDS:
            raise FactorLabValidationError("horizon_seconds is not registered")
        if self.support_status not in _RESULT_SUPPORT_STATUSES:
            raise FactorLabValidationError("support_status is unsupported")
        estimate = _finite_optional(self.estimate, field="estimate")
        lower = _finite_optional(self.ci_lower, field="ci_lower")
        upper = _finite_optional(self.ci_upper, field="ci_upper")
        if (estimate is None) != (lower is None) or (estimate is None) != (
            upper is None
        ):
            raise FactorLabValidationError(
                "estimate and CI must have consistent missingness"
            )
        if lower is not None and estimate is not None and upper is not None:
            if not lower <= estimate <= upper:
                raise FactorLabValidationError("CI must contain estimate")
        if self.support_status == "eligible" and estimate is None:
            raise FactorLabValidationError(
                "eligible result must contain estimate and CI"
            )
        for value, name in (
            (self.game_count, "game_count"),
            (self.episode_count, "episode_count"),
            (
                self.bootstrap_samples_requested,
                "bootstrap_samples_requested",
            ),
            (self.bootstrap_samples_valid, "bootstrap_samples_valid"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FactorLabValidationError(f"{name} must be nonnegative")
        if self.bootstrap_samples_valid > self.bootstrap_samples_requested:
            raise FactorLabValidationError(
                "bootstrap_samples_valid exceeds requested"
            )
        if (
            self.venue_episode_counts
            != tuple(sorted(self.venue_episode_counts))
            or len({venue for venue, _count in self.venue_episode_counts})
            != len(self.venue_episode_counts)
        ):
            raise FactorLabValidationError(
                "venue_episode_counts must be canonical and unique"
            )
        for venue, count in self.venue_episode_counts:
            _identifier(venue, field="venue")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise FactorLabValidationError(
                    "venue episode count must be nonnegative"
                )
        for value, name in (
            (
                self.leave_one_game_out_same_sign_rate,
                "leave_one_game_out_same_sign_rate",
            ),
            (
                self.max_single_game_contribution,
                "max_single_game_contribution",
            ),
            (self.bh_q_value, "bh_q_value"),
        ):
            number = _finite_optional(value, field=name)
            if number is not None and not 0 <= number <= 1:
                raise FactorLabValidationError(
                    f"{name} must be between 0 and 1"
                )
        if (
            self.venue_sign_consistent is not None
            and type(self.venue_sign_consistent) is not bool
        ):
            raise FactorLabValidationError(
                "venue_sign_consistent must be boolean or null"
            )
        _source_hashes(self.source_hashes)

    def to_dict(self) -> dict[str, object]:
        result = {
            item.name: getattr(self, item.name) for item in fields(self)
        }
        result["schema_version"] = "FactorResultV1"
        result["venue_episode_counts"] = {
            venue: count for venue, count in self.venue_episode_counts
        }
        result["source_hashes"] = list(self.source_hashes)
        return result


@dataclass(frozen=True, slots=True)
class FrontendReferenceV1:
    """Exact case locator for the existing SAF dashboard."""

    game_id: str
    episode_id: str
    logical_market_id: str
    delay_seconds: int
    horizon_seconds: int

    def __post_init__(self) -> None:
        _identifier(self.game_id, field="game_id")
        _identifier(self.episode_id, field="episode_id")
        _text(self.logical_market_id, field="logical_market_id")
        if self.delay_seconds not in _DELAY_SECONDS:
            raise FactorLabValidationError("delay_seconds is not registered")
        if self.horizon_seconds not in _HORIZONS_SECONDS:
            raise FactorLabValidationError("horizon_seconds is not registered")

    def to_query(self) -> dict[str, str]:
        return {
            "game": self.game_id,
            "episode": self.episode_id,
            "market": self.logical_market_id,
            "delay_s": str(self.delay_seconds),
            "horizon_s": str(self.horizon_seconds),
        }


@dataclass(frozen=True, slots=True)
class SportsFeatureViewV1:
    """Canonical, content-addressed play/episode sports feature view."""

    rows: tuple[SportsFeatureRowV1, ...]
    source_hashes: tuple[tuple[str, str], ...]
    rows_sha256: str
    schema_version: str = "SportsFeatureViewV1"

    def __post_init__(self) -> None:
        if self.schema_version != "SportsFeatureViewV1":
            raise FactorLabValidationError("SportsFeatureView schema version changed")
        if not self.rows:
            raise FactorLabValidationError("SportsFeatureView rows must not be empty")
        expected_order = tuple(
            sorted(
                self.rows,
                key=lambda row: (
                    row.game_id,
                    row.order_sequence,
                    row.play_id,
                    row.episode_id,
                ),
            )
        )
        if self.rows != expected_order:
            raise FactorLabValidationError("SportsFeatureView rows are not canonical")
        keys = [(row.game_id, row.play_id, row.episode_id) for row in self.rows]
        if len(set(keys)) != len(keys):
            raise FactorLabValidationError(
                "SportsFeatureView contains duplicate game/play/episode grain"
            )
        if not self.source_hashes:
            raise FactorLabValidationError("SportsFeatureView source_hashes are required")
        if self.source_hashes != tuple(sorted(self.source_hashes)):
            raise FactorLabValidationError("source_hashes must use canonical order")
        source_names = [name for name, _digest in self.source_hashes]
        if len(set(source_names)) != len(source_names):
            raise FactorLabValidationError("source_hash names must be unique")
        for name, digest in self.source_hashes:
            _identifier(name, field="source hash name")
            if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
                raise FactorLabValidationError("source hash must be a sha256 digest")
        if _SHA256_RE.fullmatch(self.rows_sha256) is None:
            raise FactorLabValidationError("rows_sha256 must be a sha256 digest")
        if self.rows_sha256 != self._expected_rows_sha256(
            self.rows, self.source_hashes
        ):
            raise FactorLabValidationError("rows_sha256 does not match canonical rows")

    @classmethod
    def build(
        cls,
        *,
        rows: Sequence[SportsFeatureRowV1],
        source_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
    ) -> SportsFeatureViewV1:
        canonical_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.game_id,
                    row.order_sequence,
                    row.play_id,
                    row.episode_id,
                ),
            )
        )
        if isinstance(source_hashes, Mapping):
            canonical_sources = tuple(
                sorted(
                    (str(name), str(digest))
                    for name, digest in source_hashes.items()
                )
            )
        else:
            canonical_sources = tuple(sorted(source_hashes))
        return cls(
            rows=canonical_rows,
            source_hashes=canonical_sources,
            rows_sha256=cls._expected_rows_sha256(
                canonical_rows, canonical_sources
            ),
        )

    @staticmethod
    def _expected_rows_sha256(
        rows: Sequence[SportsFeatureRowV1],
        source_hashes: Sequence[tuple[str, str]],
    ) -> str:
        return _canonical_sha256(
            {
                "schema_version": "SportsFeatureViewV1",
                "source_hashes": [
                    {"name": name, "sha256": digest}
                    for name, digest in source_hashes
                ],
                "rows": [row.to_dict() for row in rows],
            }
        )

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]


__all__ = [
    "BetweenPlayRosterChangeV1",
    "FactorDefinitionV1",
    "FactorLabValidationError",
    "FactorObservationV1",
    "FactorResultV1",
    "FrontendReferenceV1",
    "PlayerRoleV1",
    "PredicateClauseV1",
    "RollingFeatureV1",
    "SportsFeatureRowV1",
    "SportsFeatureViewV1",
]
