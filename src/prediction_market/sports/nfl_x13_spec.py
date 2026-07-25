"""Frozen X-13 NFL batch design, venue bindings, and immutable record types.

This module is deliberately a pure specification layer.  It does not fetch
venue data, infer kickoff times, or turn source timestamps into point-in-time
claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Any

type CanonicalValue = (
    None
    | bool
    | int
    | str
    | Decimal
    | date
    | datetime
    | tuple["CanonicalValue", ...]
)
type CanonicalPairs = tuple[tuple[str, CanonicalValue], ...]

X13_EXPERIMENT_ID = "X-13"
X13_STATUS = "PRELIMINARY_SOURCE_TIME_ONLY"
X13_HORIZONS_SECONDS = (1, 2, 5, 10, 30, 60)
X13_DELAY_SCENARIOS_SECONDS = (0, 1, 2, 3, 5, 10)
# Content address of the frozen universe artifact.  Its canonical payload
# commits all 20 capture-plan start/end values and the matching game
# source_window records, so this binds the real windows rather than a
# fabricated duplicate.
X13_SOURCE_WINDOW_UNIVERSE_ID = (
    "sha256:01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
)
X13_GAME_IDS = (
    "2025_01_BAL_BUF",
    "2025_03_ARI_SF",
    "2025_03_CIN_MIN",
    "2025_03_LA_PHI",
    "2025_03_NYJ_TB",
    "2025_04_GB_DAL",
    "2025_04_PHI_TB",
    "2025_08_CLE_NE",
    "2025_08_NYJ_CIN",
    "2025_09_CHI_CIN",
    "2025_10_JAX_HOU",
    "2025_13_NO_MIA",
    "2025_14_DAL_DET",
    "2025_14_PHI_LAC",
    "2025_14_SEA_ATL",
    "2025_16_LA_SEA",
    "2025_18_KC_LV",
    "2025_19_LA_CAR",
    "2025_20_BUF_DEN",
    "2025_20_HOU_NE",
)

# nflverse keeps Jacksonville as JAX.  Kalshi's native ticker uses JAC.
# Los Angeles Rams intentionally remain LA in both namespaces.
KALSHI_TEAM_ALIASES = (("JAX", "JAC"),)

_NFL_TEAM_CODES = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LA",
        "LAC",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    }
)
_VENUES = frozenset({"kalshi", "polymarket"})
_PRIMITIVE_FAMILIES = (
    "moneyline",
    "spread",
    "total",
    "period",
    "team_total",
    "player_passing",
    "player_rushing",
    "player_receiving",
    "player_receptions",
    "player_td",
    "first_td",
    "any_td",
)
_SAME_GAME_COMPOSITE_FAMILY = "same_game_composite"
_CROSS_GAME_INVENTORY_FAMILY = "cross_game_mve_inventory"
_ALL_FAMILIES = frozenset(
    (
        *_PRIMITIVE_FAMILIES,
        _SAME_GAME_COMPOSITE_FAMILY,
        _CROSS_GAME_INVENTORY_FAMILY,
    )
)
_CONTRACT_KINDS = frozenset(
    {
        "primitive",
        "same_game_composite",
        "cross_game_inventory",
    }
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONDITION_ID_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_NATIVE_GAME_ID_RE = re.compile(
    r"(?P<season>[0-9]{4})_(?P<week>[0-9]{2})_"
    r"(?P<away>[A-Z]+)_(?P<home>[A-Z]+)\Z"
)
_MONTH_ABBREVIATIONS = (
    "",
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)


def kalshi_team_code(nflverse_team: str) -> str:
    """Return the explicitly frozen Kalshi team code for an nflverse code."""

    if nflverse_team not in _NFL_TEAM_CODES:
        raise ValueError(f"unknown nflverse team: {nflverse_team!r}")
    aliases = dict(KALSHI_TEAM_ALIASES)
    return aliases.get(nflverse_team, nflverse_team)


def _require_nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _require_decimal(
    value: object,
    field_name: str,
    *,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal, not binary float")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value


def _require_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_string_tuple(
    value: object,
    field_name: str,
    *,
    nonempty: bool = False,
    sorted_values: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    for item in value:
        _require_nonempty_string(item, f"{field_name} item")
    if nonempty and not value:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} contains duplicate values")
    if sorted_values and value != tuple(sorted(value)):
        raise ValueError(f"{field_name} must be sorted")
    return value


def _require_int_tuple(
    value: object,
    field_name: str,
    *,
    expected: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    for item in value:
        _require_int(item, f"{field_name} item")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} contains duplicate values")
    if expected is not None and value != expected:
        raise ValueError(f"{field_name} must equal {expected!r}")
    return value


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("canonical Decimal must be finite")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _pairs_to_mapping(
    value: tuple[tuple[str, object], ...],
    *,
    path: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{path} must contain two-item tuples")
        key, child = item
        if type(key) is not str or not key:
            raise TypeError(f"{path} keys must be non-empty strings")
        if key in result:
            raise ValueError(f"{path} contains duplicate key {key!r}")
        result[key] = child
    return result


def _looks_like_pairs(value: tuple[object, ...]) -> bool:
    return bool(value) and all(
        type(item) is tuple
        and len(item) == 2
        and type(item[0]) is str
        for item in value
    )


def _canonical_value(value: object, *, path: str = "value") -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(
                getattr(value, field.name),
                path=f"{path}.{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, datetime):
        utc_value = _require_utc(value, path)
        return utc_value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, float):
        raise TypeError(f"{path} contains a binary float")
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            normalized[key] = _canonical_value(
                child,
                path=f"{path}.{key}",
            )
        return normalized
    if type(value) is tuple:
        if _looks_like_pairs(value):
            pair_mapping = _pairs_to_mapping(value, path=path)  # type: ignore[arg-type]
            return _canonical_value(pair_mapping, path=path)
        return [
            _canonical_value(child, path=f"{path}[]")
            for child in value
        ]
    if type(value) is list:
        return [
            _canonical_value(child, path=f"{path}[]")
            for child in value
        ]
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Serialize dataclasses, UTC datetimes, and Decimals deterministically."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def _validate_canonical_immutable(
    value: object,
    *,
    field_name: str,
) -> None:
    if isinstance(value, float):
        raise TypeError(f"{field_name} contains a binary float")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{field_name} contains a non-finite Decimal")
        return
    if isinstance(value, datetime):
        _require_utc(value, field_name)
        return
    if isinstance(value, date):
        return
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is tuple:
        for child in value:
            _validate_canonical_immutable(
                child,
                field_name=field_name,
            )
        return
    raise TypeError(
        f"{field_name} must contain only immutable canonical values"
    )


def _validate_pairs(
    value: object,
    field_name: str,
) -> tuple[tuple[str, object], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of key/value pairs")
    keys: list[str] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(
                f"{field_name} must contain two-item tuples"
            )
        key, child = item
        _require_nonempty_string(key, f"{field_name} key")
        keys.append(key)
        _validate_canonical_immutable(child, field_name=field_name)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{field_name} contains duplicate keys")
    if keys != sorted(keys):
        raise ValueError(f"{field_name} keys must be sorted")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BatchSpecV1:
    experiment_id: str
    status: str
    season: int
    allowed_season_types: tuple[str, ...]
    game_ids: tuple[str, ...]
    horizons_seconds: tuple[int, ...]
    delay_scenarios_seconds: tuple[int, ...]
    family_catalog_version: str
    schema_version: str
    native_game_bindings_sha256: str
    venue_family_catalog_sha256: str
    source_window_universe_id: str

    def __post_init__(self) -> None:
        if self.experiment_id != X13_EXPERIMENT_ID:
            raise ValueError(
                f"experiment_id must be {X13_EXPERIMENT_ID!r}"
            )
        if self.status != X13_STATUS:
            raise ValueError(f"status must be {X13_STATUS!r}")
        if type(self.season) is not int or self.season != 2025:
            raise ValueError("season must be 2025")
        _require_string_tuple(
            self.allowed_season_types,
            "allowed_season_types",
            nonempty=True,
        )
        if self.allowed_season_types != ("REG", "POST"):
            raise ValueError("allowed_season_types must be ('REG', 'POST')")
        _require_string_tuple(self.game_ids, "game_ids")
        if len(self.game_ids) != 20:
            raise ValueError("game_ids must contain exactly 20 games")
        if len(set(self.game_ids)) != len(self.game_ids):
            raise ValueError("game_ids contains a duplicate game")
        if self.game_ids != tuple(sorted(self.game_ids)):
            raise ValueError("game_ids must be sorted")
        if self.game_ids != X13_GAME_IDS:
            raise ValueError("game_ids must equal the frozen X-13 game set")
        _require_int_tuple(
            self.horizons_seconds,
            "horizons_seconds",
            expected=X13_HORIZONS_SECONDS,
        )
        _require_int_tuple(
            self.delay_scenarios_seconds,
            "delay_scenarios_seconds",
            expected=X13_DELAY_SCENARIOS_SECONDS,
        )
        if self.family_catalog_version != "v1":
            raise ValueError("family_catalog_version must be 'v1'")
        if self.schema_version != "v1":
            raise ValueError("schema_version must be 'v1'")
        _require_sha256(
            self.native_game_bindings_sha256,
            "native_game_bindings_sha256",
        )
        _require_sha256(
            self.venue_family_catalog_sha256,
            "venue_family_catalog_sha256",
        )
        _require_sha256(
            self.source_window_universe_id,
            "source_window_universe_id",
        )

    @property
    def canonical_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "allowed_season_types": self.allowed_season_types,
                "delay_scenarios_seconds": self.delay_scenarios_seconds,
                "experiment_id": self.experiment_id,
                "family_catalog_version": self.family_catalog_version,
                "game_ids": self.game_ids,
                "horizons_seconds": self.horizons_seconds,
                "native_game_bindings_sha256": (
                    self.native_game_bindings_sha256
                ),
                "schema_version": self.schema_version,
                "season": self.season,
                "source_window_universe_id": (
                    self.source_window_universe_id
                ),
                "status": self.status,
                "venue_family_catalog_sha256": (
                    self.venue_family_catalog_sha256
                ),
            }
        )

    @property
    def plan_id(self) -> str:
        return canonical_sha256(self.canonical_payload)


@dataclass(frozen=True, slots=True)
class NativeGameBindingV1:
    native_game_id: str
    stratum: str
    game_date: date
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    coverage_labels: tuple[str, ...]
    polymarket_event_id: str
    polymarket_market_id: str
    polymarket_condition_id: str
    kalshi_event_ticker: str
    kalshi_away_ticker: str
    kalshi_home_ticker: str
    scheduled_at_utc: datetime | None = None
    official_reference_status: str = "NOT_GREEN_BLOCKED"

    def __post_init__(self) -> None:
        if self.native_game_id not in X13_GAME_IDS:
            raise ValueError(
                "native_game_id is not in the frozen X-13 game set"
            )
        parsed = _NATIVE_GAME_ID_RE.fullmatch(self.native_game_id)
        if parsed is None:
            raise ValueError("native_game_id is malformed")
        if self.stratum not in {"Core", "Rare"}:
            raise ValueError("stratum must be 'Core' or 'Rare'")
        if type(self.game_date) is not date:
            raise TypeError("game_date must be a date")
        if self.away_team not in _NFL_TEAM_CODES:
            raise ValueError(f"unknown away_team: {self.away_team!r}")
        if self.home_team not in _NFL_TEAM_CODES:
            raise ValueError(f"unknown home_team: {self.home_team!r}")
        if (
            parsed.group("away") != self.away_team
            or parsed.group("home") != self.home_team
        ):
            raise ValueError(
                "native_game_id teams do not match away_team/home_team"
            )
        if self.away_team == self.home_team:
            raise ValueError("away_team and home_team must differ")
        _require_int(self.away_score, "away_score")
        _require_int(self.home_score, "home_score")
        _require_string_tuple(
            self.coverage_labels,
            "coverage_labels",
            nonempty=True,
        )
        if (
            type(self.polymarket_event_id) is not str
            or not self.polymarket_event_id.isdigit()
        ):
            raise ValueError("polymarket_event_id must contain digits")
        if (
            type(self.polymarket_market_id) is not str
            or not self.polymarket_market_id.isdigit()
        ):
            raise ValueError("polymarket_market_id must contain digits")
        if (
            type(self.polymarket_condition_id) is not str
            or _CONDITION_ID_RE.fullmatch(
                self.polymarket_condition_id
            )
            is None
        ):
            raise ValueError(
                "polymarket_condition_id must be a lowercase 32-byte hex ID"
            )
        expected_event_ticker = (
            "KXNFLGAME-"
            f"{self.game_date.year % 100:02d}"
            f"{_MONTH_ABBREVIATIONS[self.game_date.month]}"
            f"{self.game_date.day:02d}"
            f"{kalshi_team_code(self.away_team)}"
            f"{kalshi_team_code(self.home_team)}"
        )
        if self.kalshi_event_ticker != expected_event_ticker:
            raise ValueError(
                "kalshi event ticker must equal "
                f"{expected_event_ticker!r}"
            )
        expected_away_ticker = (
            f"{expected_event_ticker}-{kalshi_team_code(self.away_team)}"
        )
        if self.kalshi_away_ticker != expected_away_ticker:
            raise ValueError(
                "kalshi_away_ticker must preserve the native alias and "
                f"equal {expected_away_ticker!r}"
            )
        expected_home_ticker = (
            f"{expected_event_ticker}-{kalshi_team_code(self.home_team)}"
        )
        if self.kalshi_home_ticker != expected_home_ticker:
            raise ValueError(
                "kalshi_home_ticker must preserve the native alias and "
                f"equal {expected_home_ticker!r}"
            )
        if self.scheduled_at_utc is not None:
            _require_utc(self.scheduled_at_utc, "scheduled_at_utc")
        _require_nonempty_string(
            self.official_reference_status,
            "official_reference_status",
        )


@dataclass(frozen=True, slots=True)
class VenueFamilyCatalogV1:
    version: str
    primitive_families: tuple[str, ...]
    same_game_composite_family: str
    cross_game_inventory_family: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.version, "version")
        primitives = _require_string_tuple(
            self.primitive_families,
            "primitive_families",
            nonempty=True,
        )
        same_game = _require_nonempty_string(
            self.same_game_composite_family,
            "same_game_composite_family",
        )
        cross_game = _require_nonempty_string(
            self.cross_game_inventory_family,
            "cross_game_inventory_family",
        )
        if same_game == cross_game:
            raise ValueError(
                "same-game composite and cross-game inventory must differ"
            )
        if same_game in primitives or cross_game in primitives:
            raise ValueError(
                "composite families cannot be primitive families"
            )

    @property
    def all_families(self) -> tuple[str, ...]:
        return (
            *self.primitive_families,
            self.same_game_composite_family,
            self.cross_game_inventory_family,
        )


@dataclass(frozen=True, slots=True)
class MarketContractV1:
    contract_id: str
    logical_market_id: str
    venue: str
    family: str
    period: str
    subject: str
    measure: str
    comparator: str
    line: Decimal | None
    outcomes: tuple[str, ...]
    rule_sha256: str
    dependency_game_ids: tuple[str, ...]
    kind: str
    analysis_eligible: bool

    def __post_init__(self) -> None:
        _require_nonempty_string(self.contract_id, "contract_id")
        _require_nonempty_string(
            self.logical_market_id,
            "logical_market_id",
        )
        if self.venue not in _VENUES:
            raise ValueError(f"unknown venue: {self.venue!r}")
        expected_prefix = f"{self.venue}:"
        if (
            not self.contract_id.startswith(expected_prefix)
            or self.contract_id == expected_prefix
        ):
            raise ValueError(
                "contract_id must be a venue-namespaced native identity "
                f"starting with {expected_prefix!r}"
            )
        if self.family not in _ALL_FAMILIES:
            raise ValueError(f"unknown contract family: {self.family!r}")
        for name in ("period", "subject", "measure", "comparator"):
            _require_nonempty_string(getattr(self, name), name)
        if self.line is not None:
            _require_decimal(self.line, "line")
        _require_string_tuple(
            self.outcomes,
            "outcomes",
            nonempty=True,
        )
        if len(self.outcomes) < 2:
            raise ValueError("outcomes must contain at least two outcomes")
        _require_sha256(self.rule_sha256, "rule_sha256")
        dependencies = _require_string_tuple(
            self.dependency_game_ids,
            "dependency_game_ids",
            nonempty=True,
            sorted_values=True,
        )
        unknown_dependencies = tuple(
            game_id
            for game_id in dependencies
            if game_id not in X13_GAME_IDS
        )
        if unknown_dependencies:
            raise ValueError(
                "unknown dependency game IDs: "
                f"{unknown_dependencies!r}"
            )
        if self.kind not in _CONTRACT_KINDS:
            raise ValueError(f"unknown contract kind: {self.kind!r}")
        expected_kind = (
            "same_game_composite"
            if self.family == _SAME_GAME_COMPOSITE_FAMILY
            else "cross_game_inventory"
            if self.family == _CROSS_GAME_INVENTORY_FAMILY
            else "primitive"
        )
        if self.kind != expected_kind:
            raise ValueError(
                f"kind must be {expected_kind!r} for family {self.family!r}"
            )
        if type(self.analysis_eligible) is not bool:
            raise TypeError("analysis_eligible must be bool")
        if self.kind == "cross_game_inventory":
            if len(dependencies) < 2:
                raise ValueError(
                    "cross-game inventory requires at least two dependencies"
                )
            if self.analysis_eligible:
                raise ValueError(
                    "cross-game inventory cannot be analysis_eligible"
                )
        elif len(dependencies) != 1:
            raise ValueError(
                "primitive and same-game contracts require one dependency"
            )


@dataclass(frozen=True, slots=True)
class ContractInventoryV1:
    native_game_id: str
    contracts: tuple[MarketContractV1, ...]
    exclusions: tuple[str, ...]
    unknowns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.native_game_id not in X13_GAME_IDS:
            raise ValueError(
                "native_game_id is not in the frozen X-13 game set"
            )
        if type(self.contracts) is not tuple:
            raise TypeError("contracts must be a tuple")
        if not all(
            isinstance(contract, MarketContractV1)
            for contract in self.contracts
        ):
            raise TypeError("contracts must contain MarketContractV1 values")
        contract_ids = tuple(
            contract.contract_id for contract in self.contracts
        )
        if len(set(contract_ids)) != len(contract_ids):
            raise ValueError("inventory contains a duplicate contract")
        if contract_ids != tuple(sorted(contract_ids)):
            raise ValueError("contracts must be sorted by contract_id")
        for contract in self.contracts:
            if self.native_game_id not in contract.dependency_game_ids:
                raise ValueError(
                    "each contract must depend on the inventory game"
                )
        _require_string_tuple(
            self.exclusions,
            "exclusions",
            sorted_values=True,
        )
        _require_string_tuple(
            self.unknowns,
            "unknowns",
            sorted_values=True,
        )


@dataclass(frozen=True, slots=True)
class CapturePlanV1:
    plan_id: str
    native_game_ids: tuple[str, ...]
    contract_ids: tuple[str, ...]
    source_windows: CanonicalPairs
    required_free_bytes: int

    def __post_init__(self) -> None:
        _require_sha256(self.plan_id, "plan_id")
        game_ids = _require_string_tuple(
            self.native_game_ids,
            "native_game_ids",
            nonempty=True,
            sorted_values=True,
        )
        if len(game_ids) != 20:
            raise ValueError(
                "native_game_ids must contain exactly 20 games"
            )
        unknown = tuple(
            game_id for game_id in game_ids if game_id not in X13_GAME_IDS
        )
        if unknown:
            raise ValueError(f"unknown native_game_ids: {unknown!r}")
        _require_string_tuple(
            self.contract_ids,
            "contract_ids",
            nonempty=True,
            sorted_values=True,
        )
        _validate_pairs(self.source_windows, "source_windows")
        _require_int(
            self.required_free_bytes,
            "required_free_bytes",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class RawCaptureReceiptV1:
    source: str
    request_identity: CanonicalValue
    object_sha256: str
    manifest_sha256: str
    byte_length: int
    terminal_proof: CanonicalValue

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source, "source")
        _validate_canonical_immutable(
            self.request_identity,
            field_name="request_identity",
        )
        _require_sha256(self.object_sha256, "object_sha256")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_int(self.byte_length, "byte_length")
        _validate_canonical_immutable(
            self.terminal_proof,
            field_name="terminal_proof",
        )


@dataclass(frozen=True, slots=True)
class MarketObservationV1:
    observation_id: str
    contract_id: str
    venue: str
    observation_kind: str
    source_interval_start: datetime
    source_interval_end: datetime
    price: Decimal | None
    size: Decimal
    outcome: str
    observed: bool
    bid: Decimal | None = None
    ask: Decimal | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.observation_id, "observation_id")
        _require_nonempty_string(self.contract_id, "contract_id")
        if self.venue not in _VENUES:
            raise ValueError(f"unknown venue: {self.venue!r}")
        _require_nonempty_string(
            self.observation_kind,
            "observation_kind",
        )
        start = _require_utc(
            self.source_interval_start,
            "source_interval_start",
        )
        end = _require_utc(
            self.source_interval_end,
            "source_interval_end",
        )
        if start >= end:
            raise ValueError(
                "source interval must have positive width"
            )
        if self.price is not None:
            _require_decimal(
                self.price,
                "price",
                minimum=Decimal(0),
                maximum=Decimal(1),
            )
        _require_decimal(
            self.size,
            "size",
            minimum=Decimal(0),
        )
        _require_nonempty_string(self.outcome, "outcome")
        if type(self.observed) is not bool:
            raise TypeError("observed must be bool")
        if self.bid is not None:
            _require_decimal(
                self.bid,
                "bid",
                minimum=Decimal(0),
                maximum=Decimal(1),
            )
        if self.ask is not None:
            _require_decimal(
                self.ask,
                "ask",
                minimum=Decimal(0),
                maximum=Decimal(1),
            )
        if (
            self.bid is not None
            and self.ask is not None
            and self.bid > self.ask
        ):
            raise ValueError("bid cannot exceed ask")
        if self.observation_kind == "trade" and self.price is None:
            raise ValueError("trade observation requires a price")
        if self.observation_kind == "kalshi_1m_candle_bbo":
            if self.price is not None:
                raise ValueError(
                    "candle BBO cannot carry a fabricated point price"
                )
            if self.bid is None or self.ask is None:
                raise ValueError(
                    "candle BBO requires both bid and ask"
                )
        if self.price is None and self.bid is None and self.ask is None:
            raise ValueError(
                "at least one of price, bid, or ask must be present"
            )


@dataclass(frozen=True, slots=True)
class AssociationCandidateV1:
    episode_id: str
    contract_id: str
    delay_scenario_seconds: int
    horizon_seconds: int
    status: str
    order_ambiguous: bool
    contaminated: bool
    metrics: CanonicalPairs

    def __post_init__(self) -> None:
        _require_nonempty_string(self.episode_id, "episode_id")
        _require_nonempty_string(self.contract_id, "contract_id")
        _require_int(
            self.delay_scenario_seconds,
            "delay_scenario_seconds",
        )
        if self.delay_scenario_seconds not in X13_DELAY_SCENARIOS_SECONDS:
            raise ValueError(
                "delay_scenario_seconds is not in the frozen X-13 grid"
            )
        _require_int(self.horizon_seconds, "horizon_seconds", minimum=1)
        if self.horizon_seconds not in X13_HORIZONS_SECONDS:
            raise ValueError(
                "horizon_seconds is not in the frozen X-13 grid"
            )
        _require_nonempty_string(self.status, "status")
        if type(self.order_ambiguous) is not bool:
            raise TypeError("order_ambiguous must be bool")
        if type(self.contaminated) is not bool:
            raise TypeError("contaminated must be bool")
        _validate_pairs(self.metrics, "metrics")


@dataclass(frozen=True, slots=True)
class BatchManifestV1:
    plan_id: str
    raw_manifest_sha256s: tuple[str, ...]
    builder_versions: tuple[tuple[str, str], ...]
    game_artifact_sha256s: tuple[tuple[str, str], ...]
    published_at_utc: datetime | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.plan_id, "plan_id")
        _require_string_tuple(
            self.raw_manifest_sha256s,
            "raw_manifest_sha256s",
            nonempty=True,
            sorted_values=True,
        )
        for index, digest in enumerate(self.raw_manifest_sha256s):
            _require_sha256(digest, f"raw_manifest_sha256s[{index}]")
        _validate_pairs(self.builder_versions, "builder_versions")
        for builder, version in self.builder_versions:
            _require_nonempty_string(builder, "builder_versions key")
            _require_nonempty_string(version, "builder_versions value")
        _validate_pairs(
            self.game_artifact_sha256s,
            "game_artifact_sha256s",
        )
        artifact_game_ids = tuple(
            game_id for game_id, _ in self.game_artifact_sha256s
        )
        if len(artifact_game_ids) != 20:
            raise ValueError(
                "game_artifact_sha256s must contain exactly 20 games"
            )
        if artifact_game_ids != X13_GAME_IDS:
            raise ValueError(
                "game_artifact_sha256s must equal the frozen X-13 game set"
            )
        for game_id, digest in self.game_artifact_sha256s:
            if game_id not in X13_GAME_IDS:
                raise ValueError(
                    f"unknown game artifact dependency: {game_id!r}"
                )
            _require_sha256(digest, f"game artifact {game_id}")
        if self.published_at_utc is not None:
            _require_utc(self.published_at_utc, "published_at_utc")

    @property
    def canonical_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "builder_versions": self.builder_versions,
                "game_artifact_sha256s": self.game_artifact_sha256s,
                "plan_id": self.plan_id,
                "published_at_utc": self.published_at_utc,
                "raw_manifest_sha256s": self.raw_manifest_sha256s,
            }
        )

    @property
    def batch_id(self) -> str:
        return canonical_sha256(self.canonical_payload)


X13_VENUE_FAMILY_CATALOG = VenueFamilyCatalogV1(
    version="v1",
    primitive_families=_PRIMITIVE_FAMILIES,
    same_game_composite_family=_SAME_GAME_COMPOSITE_FAMILY,
    cross_game_inventory_family=_CROSS_GAME_INVENTORY_FAMILY,
)

def _binding(
    native_game_id: str,
    stratum: str,
    game_date_text: str,
    away_team: str,
    home_team: str,
    away_score: int,
    home_score: int,
    coverage_labels: tuple[str, ...],
    polymarket_event_id: str,
    polymarket_market_id: str,
    polymarket_condition_id: str,
    kalshi_event_ticker: str,
) -> NativeGameBindingV1:
    return NativeGameBindingV1(
        native_game_id=native_game_id,
        stratum=stratum,
        game_date=date.fromisoformat(game_date_text),
        away_team=away_team,
        home_team=home_team,
        away_score=away_score,
        home_score=home_score,
        coverage_labels=coverage_labels,
        polymarket_event_id=polymarket_event_id,
        polymarket_market_id=polymarket_market_id,
        polymarket_condition_id=polymarket_condition_id,
        kalshi_event_ticker=kalshi_event_ticker,
        kalshi_away_ticker=(
            f"{kalshi_event_ticker}-{kalshi_team_code(away_team)}"
        ),
        kalshi_home_ticker=(
            f"{kalshi_event_ticker}-{kalshi_team_code(home_team)}"
        ),
    )


X13_GAME_BINDINGS = (
    _binding(
        "2025_01_BAL_BUF",
        "Core",
        "2025-09-07",
        "BAL",
        "BUF",
        40,
        41,
        ("2PT", "HIGH"),
        "40828",
        "584347",
        "0x7f2e645fd40487882cc9e7a05d240f5b1c2220a361fbb345b21f921070411409",
        "KXNFLGAME-25SEP07BALBUF",
    ),
    _binding(
        "2025_03_ARI_SF",
        "Core",
        "2025-09-21",
        "ARI",
        "SF",
        15,
        16,
        ("SAF", "LOW"),
        "44348",
        "597273",
        "0x86114dd0c8cb0620543ac867b2c1c38715f6c71e62d2f6580c51c5223e6f3c3e",
        "KXNFLGAME-25SEP21ARISF",
    ),
    _binding(
        "2025_03_CIN_MIN",
        "Core",
        "2025-09-21",
        "CIN",
        "MIN",
        10,
        48,
        ("RTD2", "BLOW"),
        "44339",
        "597264",
        "0x00cfb1fe371a24b3a9d81f1bb9ba5a037a53f70cf5f2741f3e8d24ca0eb46591",
        "KXNFLGAME-25SEP21CINMIN",
    ),
    _binding(
        "2025_03_LA_PHI",
        "Rare",
        "2025-09-21",
        "LA",
        "PHI",
        26,
        33,
        ("BLK2",),
        "44341",
        "597266",
        "0xbaef857e53f69249525c725e35c4dafb4c9df13c915738351bf2d5ab9fef01b9",
        "KXNFLGAME-25SEP21LAPHI",
    ),
    _binding(
        "2025_03_NYJ_TB",
        "Core",
        "2025-09-21",
        "NYJ",
        "TB",
        27,
        29,
        ("RTD", "BLK"),
        "44342",
        "597267",
        "0x44d172447f68fcc304a574af3439776f798155fee1ea6ade6861ea62d269a856",
        "KXNFLGAME-25SEP21NYJTB",
    ),
    _binding(
        "2025_04_GB_DAL",
        "Rare",
        "2025-09-28",
        "GB",
        "DAL",
        40,
        40,
        ("def2PT", "OT", "TIE"),
        "47514",
        "606182",
        "0x0af48a189e649217f4c5af8184a22e3961500c63e97e9474d010a93350e0b4c7",
        "KXNFLGAME-25SEP28GBDAL",
    ),
    _binding(
        "2025_04_PHI_TB",
        "Rare",
        "2025-09-28",
        "PHI",
        "TB",
        31,
        25,
        ("SAF", "BLK"),
        "47509",
        "606177",
        "0xccc244fc8e622c7a3768672661e64b7e9bb53bbb4f1e53a3aa36bd0d33428548",
        "KXNFLGAME-25SEP28PHITB",
    ),
    _binding(
        "2025_08_CLE_NE",
        "Rare",
        "2025-10-26",
        "CLE",
        "NE",
        13,
        32,
        ("SAF", "ONS", "2PT"),
        "61641",
        "639540",
        "0xcfcbaa9a402806aa70910ddc64179d29af5bdbac676ec79ac600912f0db26c1b",
        "KXNFLGAME-25OCT26CLENE",
    ),
    _binding(
        "2025_08_NYJ_CIN",
        "Core",
        "2025-10-26",
        "NYJ",
        "CIN",
        39,
        38,
        ("HIGH",),
        "61639",
        "639538",
        "0x983948ed0de118d2567507ce97f3275ffd04edb1b94a427e7520e6b1161a23a1",
        "KXNFLGAME-25OCT26NYJCIN",
    ),
    _binding(
        "2025_09_CHI_CIN",
        "Rare",
        "2025-11-02",
        "CHI",
        "CIN",
        47,
        42,
        ("RTD", "2PT", "BLK", "ONS"),
        "65485",
        "650203",
        "0xb6059f90055603d5d41cddec29021d91614ac5398cb1a0aa4d6c641189578744",
        "KXNFLGAME-25NOV02CHICIN",
    ),
    _binding(
        "2025_10_JAX_HOU",
        "Rare",
        "2025-11-09",
        "JAX",
        "HOU",
        29,
        36,
        ("RTD2", "2PT", "NUL"),
        "70850",
        "660423",
        "0xf1afc8d6d150bb942af398a2e615ddfadf83517234e32ba67adc47feda90d803",
        "KXNFLGAME-25NOV09JACHOU",
    ),
    _binding(
        "2025_13_NO_MIA",
        "Rare",
        "2025-11-30",
        "NO",
        "MIA",
        17,
        21,
        ("def2PT", "ONS"),
        "89356",
        "700733",
        "0x04f68fed65c7436d84cd952dd334c5fe543e54a32b0d7f148d494e1eb8301a13",
        "KXNFLGAME-25NOV30NOMIA",
    ),
    _binding(
        "2025_14_DAL_DET",
        "Core",
        "2025-12-04",
        "DAL",
        "DET",
        30,
        44,
        ("HIGH", "BLK"),
        "89365",
        "700742",
        "0xe6f492eb15583ae324d43e987e3ad1b7bd1a86690b3b62a2843162278ff7af0e",
        "KXNFLGAME-25DEC04DALDET",
    ),
    _binding(
        "2025_14_PHI_LAC",
        "Core",
        "2025-12-08",
        "PHI",
        "LAC",
        19,
        22,
        ("OT", "MISS", "NUL"),
        "90035",
        "703065",
        "0x973a05cb855049840adb639aa0c927d4d0f5ee7cc383269a24256b39ec4cd2d9",
        "KXNFLGAME-25DEC08PHILAC",
    ),
    _binding(
        "2025_14_SEA_ATL",
        "Core",
        "2025-12-07",
        "SEA",
        "ATL",
        37,
        9,
        ("RTD", "BLK", "BLOW"),
        "89366",
        "700743",
        "0x07fdb434a81996e35e05aeb4d7e8d9f79866a6ed435282489b50c9a781c0734a",
        "KXNFLGAME-25DEC07SEAATL",
    ),
    _binding(
        "2025_16_LA_SEA",
        "Core",
        "2025-12-18",
        "LA",
        "SEA",
        37,
        38,
        ("OT", "RTD", "2PT"),
        "97878",
        "832734",
        "0x6749b632cb9df8883a4958aa5493e50f9bc71dd6597dcefaeb7ee401c2c790ae",
        "KXNFLGAME-25DEC18LASEA",
    ),
    _binding(
        "2025_18_KC_LV",
        "Rare",
        "2026-01-04",
        "KC",
        "LV",
        12,
        14,
        ("SAF", "LOW", "noTD"),
        "115293",
        "988312",
        "0x7b718a0402aa0bb771e441d60402fef788a965c3a4b79d2de5fc053dc705cf45",
        "KXNFLGAME-26JAN04KCLV",
    ),
    _binding(
        "2025_19_LA_CAR",
        "Core",
        "2026-01-10",
        "LA",
        "CAR",
        34,
        31,
        ("POST", "BLK"),
        "145825",
        "1116464",
        "0x32102e4ab8969621cab85949c3c571b5008a4aa30d31022fa659d0ed287686a9",
        "KXNFLGAME-26JAN10LACAR",
    ),
    _binding(
        "2025_20_BUF_DEN",
        "Core",
        "2026-01-17",
        "BUF",
        "DEN",
        30,
        33,
        ("POST", "OT"),
        "161553",
        "1174855",
        "0x746f66fb1859ce9f795b0bbe0b4fede7666bfe4de95b2aa280296b25f763deee",
        "KXNFLGAME-26JAN17BUFDEN",
    ),
    _binding(
        "2025_20_HOU_NE",
        "Core",
        "2026-01-18",
        "HOU",
        "NE",
        16,
        28,
        ("POST", "RTD"),
        "161554",
        "1174856",
        "0x4195016305dc0e1b849be8f3a39721c57c9a2c9717b9a050abfd10014de03078",
        "KXNFLGAME-26JAN18HOUNE",
    ),
)

if tuple(binding.native_game_id for binding in X13_GAME_BINDINGS) != X13_GAME_IDS:
    raise ValueError("X13_GAME_BINDINGS must exactly match sorted X13_GAME_IDS")


def build_x13_batch_spec(
    *,
    native_game_bindings: tuple[NativeGameBindingV1, ...],
    venue_family_catalog: VenueFamilyCatalogV1,
    source_window_universe_id: str,
) -> BatchSpecV1:
    """Build the X-13 plan root from every frozen identity commitment."""

    if type(native_game_bindings) is not tuple or not all(
        isinstance(binding, NativeGameBindingV1)
        for binding in native_game_bindings
    ):
        raise TypeError(
            "native_game_bindings must be a tuple of NativeGameBindingV1"
        )
    binding_game_ids = tuple(
        binding.native_game_id for binding in native_game_bindings
    )
    if binding_game_ids != X13_GAME_IDS:
        raise ValueError(
            "native_game_bindings must exactly match sorted X13_GAME_IDS"
        )
    if not isinstance(venue_family_catalog, VenueFamilyCatalogV1):
        raise TypeError(
            "venue_family_catalog must be VenueFamilyCatalogV1"
        )
    _require_sha256(
        source_window_universe_id,
        "source_window_universe_id",
    )
    return BatchSpecV1(
        experiment_id=X13_EXPERIMENT_ID,
        status=X13_STATUS,
        season=2025,
        allowed_season_types=("REG", "POST"),
        game_ids=X13_GAME_IDS,
        horizons_seconds=X13_HORIZONS_SECONDS,
        delay_scenarios_seconds=X13_DELAY_SCENARIOS_SECONDS,
        family_catalog_version=venue_family_catalog.version,
        schema_version="v1",
        native_game_bindings_sha256=canonical_sha256(
            native_game_bindings
        ),
        venue_family_catalog_sha256=canonical_sha256(
            venue_family_catalog
        ),
        source_window_universe_id=source_window_universe_id,
    )


X13_BATCH_SPEC = build_x13_batch_spec(
    native_game_bindings=X13_GAME_BINDINGS,
    venue_family_catalog=X13_VENUE_FAMILY_CATALOG,
    source_window_universe_id=X13_SOURCE_WINDOW_UNIVERSE_ID,
)


__all__ = [
    "KALSHI_TEAM_ALIASES",
    "X13_BATCH_SPEC",
    "X13_DELAY_SCENARIOS_SECONDS",
    "X13_EXPERIMENT_ID",
    "X13_GAME_BINDINGS",
    "X13_GAME_IDS",
    "X13_HORIZONS_SECONDS",
    "X13_SOURCE_WINDOW_UNIVERSE_ID",
    "X13_STATUS",
    "X13_VENUE_FAMILY_CATALOG",
    "AssociationCandidateV1",
    "BatchManifestV1",
    "BatchSpecV1",
    "CapturePlanV1",
    "ContractInventoryV1",
    "MarketContractV1",
    "MarketObservationV1",
    "NativeGameBindingV1",
    "RawCaptureReceiptV1",
    "VenueFamilyCatalogV1",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "build_x13_batch_spec",
    "kalshi_team_code",
]
