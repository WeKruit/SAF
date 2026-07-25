"""Atomic, offline publication for the frozen NFL X-13 twenty-game batch."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

from prediction_market.sports.nfl_x13_spec import (
    X13_BATCH_SPEC,
    X13_DELAY_SCENARIOS_SECONDS,
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
    X13_HORIZONS_SECONDS,
    canonical_json_bytes as x13_canonical_json_bytes,
    canonical_sha256,
)


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"2025_[0-9]{2}_[A-Z]+_[A-Z]+\Z")
_PASS_LAYER_VALUES = {
    "layer_g": {"PASS", "PASS_SOURCE_TIME_ONLY"},
    "layer_m": {"PASS", "PASS_SOURCE_TIME_ONLY", "PASS_SOURCE_OBSERVATIONS_ONLY"},
    "layer_a": {"PASS", "PASS_SOURCE_TIME_ONLY"},
}
_BINDING_BY_GAME_ID = {
    binding.native_game_id: binding for binding in X13_GAME_BINDINGS
}
_FIELD_ORIENTATION_SEMANTICS = (
    "canonical_team_end_orientation_not_tracking_coordinate"
)
_UNAVAILABLE_FIELD_SEMANTICS = "unavailable_without_possession_and_yardline"
_OUTCOME_COVERAGE_VALUES = frozenset(
    {"available", "unavailable_no_trades"}
)
_REQUIRED_NON_TABLE_AUXILIARY = {
    "association_cardinality_report": (
        "reports/association-cardinality-report-v1.json",
        "nfl_x13_association_cardinality_report_v1",
    ),
    "association_runtime_rss_report": (
        "reports/runtime-rss-v1.json",
        "nfl_x13_association_runtime_rss_report_v1",
    ),
    "robinhood_provenance_inventory": (
        "inventory/robinhood-provenance-inventory-v1.json",
        "nfl_x13_robinhood_provenance_inventory_v1",
    ),
    "contract_semantic_overlap_audit": (
        "reports/contract-semantic-overlap-audit-v1.json",
        "nfl_x13_contract_semantic_overlap_audit_v1",
    ),
    "cross_game_mve_inventory": (
        "inventory/cross-game-mve-inventory-v1.json",
        "nfl_x13_cross_game_mve_inventory_v1",
    ),
    "validation_report": (
        "reports/validation-report-v1.json",
        "nfl_x13_validation_report_v1",
    ),
    "lineage_report": (
        "reports/lineage-report-v1.json",
        "nfl_x13_lineage_report_v1",
    ),
}
X13_ASSOCIATION_SCHEMA_FIELDS = (
    ("game_id", "string", False),
    ("episode_id", "string", False),
    ("episode_type", "string", False),
    ("contract_id", "string", False),
    ("logical_market_id", "string", False),
    ("venue", "string", False),
    ("family", "string", False),
    ("outcome", "string", False),
    ("source_time_start_utc", "string", False),
    ("source_time_end_utc", "string", False),
    ("delay_scenario_seconds", "int32", False),
    ("horizon_seconds", "int32", False),
    ("pre_event_actual_trade", "string", True),
    ("first_post_event_trade", "string", True),
    ("vwap", "string", True),
    ("signed_price_change", "string", True),
    ("maximum_excursion", "string", True),
    ("net_change_60s", "string", True),
    ("trade_count", "int64", False),
    ("volume", "string", False),
    ("staleness_seconds", "string", True),
    ("overshoot_candidate", "bool", False),
    ("reversal_candidate", "bool", False),
    ("two_venue_direction_consistency", "string", False),
    ("order_ambiguous", "bool", False),
    ("contaminated", "bool", False),
    ("validity_status", "string", False),
)
X13_ASSOCIATION_SCHEMA_FINGERPRINT = canonical_sha256(
    {
        "schema": "nfl_x13_association_candidate_parquet_v1",
        "fields": X13_ASSOCIATION_SCHEMA_FIELDS,
    }
)
X13_FROZEN_UNIVERSE_ID = (
    "sha256:01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
)
X13_FROZEN_ASSOCIATION_ROWS_BY_GAME = {
    "2025_01_BAL_BUF": 2_749_824,
    "2025_03_ARI_SF": 836_784,
    "2025_03_CIN_MIN": 713_952,
    "2025_03_LA_PHI": 928_800,
    "2025_03_NYJ_TB": 842_688,
    "2025_04_GB_DAL": 895_104,
    "2025_04_PHI_TB": 831_600,
    "2025_08_CLE_NE": 1_679_184,
    "2025_08_NYJ_CIN": 1_459_008,
    "2025_09_CHI_CIN": 2_757_960,
    "2025_10_JAX_HOU": 2_505_600,
    "2025_13_NO_MIA": 2_801_664,
    "2025_14_DAL_DET": 4_624_992,
    "2025_14_PHI_LAC": 3_787_560,
    "2025_14_SEA_ATL": 3_655_080,
    "2025_16_LA_SEA": 4_422_384,
    "2025_18_KC_LV": 1_507_968,
    "2025_19_LA_CAR": 3_317_760,
    "2025_20_BUF_DEN": 4_284_576,
    "2025_20_HOU_NE": 4_022_424,
}
X13_FROZEN_ASSOCIATION_ROW_COUNT = 48_624_912
X13_FROZEN_CONTRACT_INVENTORY_SHA256_BY_GAME = {
    "2025_01_BAL_BUF": "sha256:d63e68cee8fb1ffbc51fb46cb85cc2dff07b66d7ca01443588f04ecf37644479",
    "2025_03_ARI_SF": "sha256:4a724efcbbcbee091cdc3d92699b2253e33915b4019628c118e1227a3eda8a3e",
    "2025_03_CIN_MIN": "sha256:7daefcf580e0deb5ef5078578e7456a096821d95f666f769ca82bb7de38b0f93",
    "2025_03_LA_PHI": "sha256:df25b7320dfbd338669ab1eb627040e19b4ef9c0f2b325ce406d98ecd032d4b2",
    "2025_03_NYJ_TB": "sha256:77fc9c1ae6e2945785dac6ab1e3c6a759f93ffcd547cf4d4f630ce4adfb9a73d",
    "2025_04_GB_DAL": "sha256:e656aabc1509ce156d9d9e8ffde9d9fd19990aa4ab363441aed3648f0ff017e9",
    "2025_04_PHI_TB": "sha256:a430c39e951bbd7ad3c7c6cb52511a500e35286addd1b5b7a67afb121f6b61ee",
    "2025_08_CLE_NE": "sha256:5d6200810e046794c637f689b53b529a409eac309959d13c621063c6aecf8ae4",
    "2025_08_NYJ_CIN": "sha256:2e65f2a0ce2a3ce2629e074434eb44180a35014c5116203f57046568963c913d",
    "2025_09_CHI_CIN": "sha256:78b49982f8b4b9672d1f32c3e48b043b33c1a9e22fc120d26ef687a8cf33c84a",
    "2025_10_JAX_HOU": "sha256:31f39ad0e8da0b2a6af23350a08572a1b482ee699fdb8ddd11a14155e529c396",
    "2025_13_NO_MIA": "sha256:91d7f8c4a9d3c6b5720204e3b875e56bf0f20c3f776bb189c8a64cda7935aeb5",
    "2025_14_DAL_DET": "sha256:e68c589548393ca41e9d8eaada785283bab2fee45fa6038bf60d42d8d1bbefe0",
    "2025_14_PHI_LAC": "sha256:01bfd07a42fbb463a041314ab5fdc4b700d2e75a8ec999cdab71b572ba9addca",
    "2025_14_SEA_ATL": "sha256:59e5751c5374f91d960aaeb3d4cf83bdb147c14c6405cb8fc732aa01af62bd3d",
    "2025_16_LA_SEA": "sha256:41d0e40fb915e5c849ddcdd7964ed49d93d271c9593f658e2094c38f30518563",
    "2025_18_KC_LV": "sha256:f417a6d21a2a0ee6265125a1ad2046d07ec88602eb4ebb64f086c8e14813dbf9",
    "2025_19_LA_CAR": "sha256:843f9fee76ef849b039a8293759a3ce983411c31f3d4edbb3ca0e1380890dc68",
    "2025_20_BUF_DEN": "sha256:066ac98d72e97a0fc77320b976a3501a609045d6c96fdb97ab3dc5306a92c5ff",
    "2025_20_HOU_NE": "sha256:2aa70d3a8a6a07461424358c3a2492a3b48061862674e8beb2d6432e5ffb0d4d",
}


class X13BatchPublishError(ValueError):
    """The batch cannot prove its frozen source-time publication gates."""


def association_arrow_schema() -> object:
    """Return the canonical Arrow schema for full X-13 association tables."""

    try:
        import pyarrow as pa
    except ImportError as error:  # pragma: no cover - direct dependency
        raise X13BatchPublishError(
            "pyarrow is required to verify association tables"
        ) from error
    types = {
        "string": pa.string(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
    }
    return pa.schema(
        [
            pa.field(name, types[type_name], nullable=nullable)
            for name, type_name, nullable in X13_ASSOCIATION_SCHEMA_FIELDS
        ],
        metadata={
            b"schema": b"nfl_x13_association_candidate_parquet_v1",
            b"schema_fingerprint": (
                X13_ASSOCIATION_SCHEMA_FINGERPRINT.encode()
            ),
            b"experiment_id": b"X-13",
            b"claim_scope": b"PRELIMINARY_SOURCE_TIME_ONLY",
        },
    )


@dataclass(frozen=True, slots=True)
class AuxiliaryArtifactV1:
    """One prebuilt derived artifact copied into the atomic X-13 publication."""

    relative_path: str
    source_path: Path
    role: str
    media_type: str
    schema_fingerprint: str
    row_count: int | None
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        _validate_auxiliary_relative_path(self.relative_path)
        if not isinstance(self.source_path, Path):
            raise X13BatchPublishError("auxiliary source_path must be a Path")
        if type(self.role) is not str or not self.role:
            raise X13BatchPublishError("auxiliary role must be nonempty")
        if type(self.media_type) is not str or not self.media_type:
            raise X13BatchPublishError("auxiliary media_type must be nonempty")
        _require_sha256(
            self.schema_fingerprint, "auxiliary schema_fingerprint"
        )
        _require_sha256(self.sha256, "auxiliary sha256")
        if (
            type(self.row_count) is not int
            and self.row_count is not None
        ) or (type(self.row_count) is int and self.row_count < 0):
            raise X13BatchPublishError(
                "auxiliary row_count must be a nonnegative integer or null"
            )
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise X13BatchPublishError(
                "auxiliary byte_length must be a nonnegative integer"
            )

    @property
    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "nfl_x13_auxiliary_artifact_v1",
            "relative_path": self.relative_path,
            "role": self.role,
            "media_type": self.media_type,
            "schema_fingerprint": self.schema_fingerprint,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return x13_canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise X13BatchPublishError(
            f"payload is not canonical JSON: {error}"
        ) from error


def canonical_payload_hash(value: object) -> str:
    """Return the content hash used for plans, games, and batch identity."""

    try:
        return canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise X13BatchPublishError(
            f"payload is not canonical JSON: {error}"
        ) from error


_OBSERVATION_PAYLOAD_FIELDS = {
    "ask",
    "bid",
    "derived_from_observation_id",
    "is_block_trade",
    "kind",
    "logical_market_id",
    "native_id",
    "native_source_time",
    "no_price_dollars",
    "observation_id",
    "open_interest",
    "outcome",
    "price",
    "primary_path_eligible",
    "provenance",
    "raw_market_id",
    "render_semantics",
    "size",
    "source_end_inclusive",
    "source_end",
    "source_start",
    "taker_outcome_side",
    "taker_side",
    "venue",
    "volume",
    "yes_ask_ohlc",
    "yes_bid_ohlc",
    "yes_price_dollars",
}


def observation_payload_sha256(value: Mapping[str, object]) -> str:
    """Recompute MarketObservation.observation_id from canonical payload."""

    row = dict(_require_mapping(value, "observation payload"))
    if set(row) != _OBSERVATION_PAYLOAD_FIELDS:
        raise X13BatchPublishError(
            "observation payload fields do not match MarketObservation v1"
        )
    material = {
        "ask": row["ask"],
        "bid": row["bid"],
        "derived_from_observation_id": (
            row["derived_from_observation_id"]
        ),
        "is_block_trade": row["is_block_trade"],
        "kind": row["kind"],
        "logical_market_id": row["logical_market_id"],
        "native_id": row["native_id"],
        "native_source_time": row["native_source_time"],
        "no_price_dollars": row["no_price_dollars"],
        "open_interest": row["open_interest"],
        "outcome": row["outcome"],
        "price": row["price"],
        "primary_path_eligible": row["primary_path_eligible"],
        "provenance": row["provenance"],
        "raw_market_id": row["raw_market_id"],
        "size": row["size"],
        "source_end": row["source_end"],
        "source_end_inclusive": row["source_end_inclusive"],
        "source_start": row["source_start"],
        "taker_outcome_side": row["taker_outcome_side"],
        "taker_side": row["taker_side"],
        "venue": row["venue"],
        "volume": row["volume"],
        "yes_ask_ohlc": row["yes_ask_ohlc"],
        "yes_bid_ohlc": row["yes_bid_ohlc"],
        "yes_price_dollars": row["yes_price_dollars"],
    }
    return canonical_payload_hash(material)


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise X13BatchPublishError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_auxiliary_relative_path(value: object) -> str:
    if type(value) is not str or not value:
        raise X13BatchPublishError(
            "auxiliary artifact must use a safe relative path"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.parts[0] not in {"data", "inventory", "reports"}
    ):
        raise X13BatchPublishError(
            "auxiliary artifact must use a safe relative path"
        )
    return value


def _normalize_auxiliary_descriptors(
    values: Sequence[AuxiliaryArtifactV1 | Mapping[str, object]],
) -> list[dict[str, object]]:
    descriptors: list[dict[str, object]] = []
    paths: set[str] = set()
    for value in _require_sequence(values, "auxiliary_artifacts"):
        raw = value.descriptor if isinstance(value, AuxiliaryArtifactV1) else value
        descriptor = dict(_require_mapping(raw, "auxiliary artifact"))
        expected_fields = {
            "schema",
            "relative_path",
            "role",
            "media_type",
            "schema_fingerprint",
            "row_count",
            "sha256",
            "byte_length",
        }
        if set(descriptor) != expected_fields:
            raise X13BatchPublishError(
                "auxiliary artifact descriptor fields are not canonical"
            )
        if descriptor.get("schema") != "nfl_x13_auxiliary_artifact_v1":
            raise X13BatchPublishError(
                "auxiliary artifact has an unknown schema"
            )
        relative_path = _validate_auxiliary_relative_path(
            descriptor.get("relative_path")
        )
        if relative_path in paths:
            raise X13BatchPublishError(
                "auxiliary artifacts contain a duplicate path"
            )
        paths.add(relative_path)
        for field in ("role", "media_type"):
            if type(descriptor.get(field)) is not str or not descriptor[field]:
                raise X13BatchPublishError(
                    f"auxiliary artifact {field} must be nonempty"
                )
        _require_sha256(
            descriptor.get("schema_fingerprint"),
            "auxiliary schema_fingerprint",
        )
        _require_sha256(descriptor.get("sha256"), "auxiliary sha256")
        row_count = descriptor.get("row_count")
        if (
            type(row_count) is not int
            and row_count is not None
        ) or (type(row_count) is int and row_count < 0):
            raise X13BatchPublishError(
                "auxiliary row_count must be a nonnegative integer or null"
            )
        byte_length = descriptor.get("byte_length")
        if type(byte_length) is not int or byte_length < 0:
            raise X13BatchPublishError(
                "auxiliary byte_length must be a nonnegative integer"
            )
        _canonical_json_bytes(descriptor)
        descriptors.append(descriptor)
    return sorted(descriptors, key=lambda item: str(item["relative_path"]))


def _validate_required_auxiliary_descriptors(
    descriptors: Sequence[Mapping[str, object]],
) -> None:
    roles: dict[str, list[Mapping[str, object]]] = {}
    for descriptor in descriptors:
        roles.setdefault(str(descriptor["role"]), []).append(descriptor)
    if set(roles) != {
        "association_candidate_table",
        *_REQUIRED_NON_TABLE_AUXILIARY,
    } or not roles["association_candidate_table"]:
        raise X13BatchPublishError(
            "auxiliary artifacts do not match the exact X-13 role set"
        )
    for role, (relative_path, schema_name) in (
        _REQUIRED_NON_TABLE_AUXILIARY.items()
    ):
        values = roles.get(role, [])
        expected_fingerprint = canonical_sha256(
            {
                "schema": schema_name,
                "version": "canonical-json-v1",
            }
        )
        if (
            len(values) != 1
            or values[0]["relative_path"] != relative_path
            or values[0]["media_type"] != "application/json"
            or values[0]["schema_fingerprint"]
            != expected_fingerprint
        ):
            raise X13BatchPublishError(
                f"auxiliary role {role} is missing or not canonical"
            )


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise X13BatchPublishError(f"{field} must be an object")
    return value


def _require_sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise X13BatchPublishError(f"{field} must be an array")
    return value


def _decimal_probability(value: object, field: str) -> Decimal:
    if isinstance(value, Decimal):
        number = value
    elif type(value) is str and value:
        try:
            number = Decimal(value)
        except InvalidOperation as error:
            raise X13BatchPublishError(
                f"{field} must be a canonical decimal string"
            ) from error
    else:
        raise X13BatchPublishError(f"{field} must be a canonical decimal string")
    if not number.is_finite() or not Decimal(0) <= number <= Decimal(1):
        raise X13BatchPublishError(f"{field} must be between zero and one")
    return number


def _decimal_nonnegative(value: object, field: str) -> Decimal:
    if isinstance(value, Decimal):
        number = value
    elif type(value) in {str, int}:
        try:
            number = Decimal(value)
        except (InvalidOperation, ValueError) as error:
            raise X13BatchPublishError(
                f"{field} must be an exact nonnegative decimal"
            ) from error
    else:
        raise X13BatchPublishError(
            f"{field} must be an exact nonnegative decimal"
        )
    if not number.is_finite() or number < 0:
        raise X13BatchPublishError(
            f"{field} must be an exact nonnegative decimal"
        )
    return number


def _observation_time(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise X13BatchPublishError(f"{field} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise X13BatchPublishError(f"{field} must be canonical UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise X13BatchPublishError(f"{field} must be canonical UTC")
    return parsed.astimezone(timezone.utc)


def _ohlc_probabilities(value: object, field: str) -> dict[str, Decimal]:
    row = _require_mapping(value, field)
    if set(row) != {"open", "high", "low", "close"}:
        raise X13BatchPublishError(f"{field} is not canonical OHLC")
    values = {
        key: _decimal_probability(row[key], f"{field}.{key}")
        for key in ("open", "high", "low", "close")
    }
    if (
        values["low"] > values["high"]
        or not values["low"] <= values["open"] <= values["high"]
        or not values["low"] <= values["close"] <= values["high"]
    ):
        raise X13BatchPublishError(f"{field} OHLC ordering is invalid")
    return values


def _validate_observation_semantics(
    row: Mapping[str, Any],
    *,
    game_id: str,
    observation_id: str,
) -> None:
    start = _observation_time(
        row.get("source_start"),
        f"{game_id}.{observation_id}.source_start",
    )
    end = _observation_time(
        row.get("source_end"),
        f"{game_id}.{observation_id}.source_end",
    )
    if end <= start or type(row.get("source_end_inclusive")) is not bool:
        raise X13BatchPublishError(
            f"{game_id} observation source interval is invalid"
        )
    native_source = row.get("native_source_time")
    native_time = (
        None
        if native_source is None
        else _observation_time(
            native_source,
            f"{game_id}.{observation_id}.native_source_time",
        )
    )
    _decimal_nonnegative(
        row.get("size"),
        f"{game_id}.{observation_id}.size",
    )
    if type(row.get("primary_path_eligible")) is not bool:
        raise X13BatchPublishError(
            f"{game_id} observation primary path eligibility is invalid"
        )

    kind = row.get("kind")
    venue = row.get("venue")
    provenance = row.get("provenance")
    if kind == "kalshi_1m_candle_bbo":
        if (
            venue != "kalshi"
            or provenance != "observed"
            or row.get("price") is not None
            or row.get("primary_path_eligible") is not True
            or end - start != timedelta(minutes=1)
            or native_time != end
            or row.get("is_block_trade") is not None
        ):
            raise X13BatchPublishError(
                f"{game_id} Kalshi candle semantics are invalid"
            )
        bids = _ohlc_probabilities(
            row.get("yes_bid_ohlc"),
            f"{game_id}.{observation_id}.yes_bid_ohlc",
        )
        asks = _ohlc_probabilities(
            row.get("yes_ask_ohlc"),
            f"{game_id}.{observation_id}.yes_ask_ohlc",
        )
        if any(bids[basis] > asks[basis] for basis in bids):
            raise X13BatchPublishError(
                f"{game_id} Kalshi candle contains crossed BBO"
            )
        if (
            _decimal_probability(
                row.get("bid"), f"{game_id}.{observation_id}.bid"
            )
            != bids["close"]
            or _decimal_probability(
                row.get("ask"), f"{game_id}.{observation_id}.ask"
            )
            != asks["close"]
            or _decimal_nonnegative(
                row.get("volume"),
                f"{game_id}.{observation_id}.volume",
            )
            != _decimal_nonnegative(
                row.get("size"),
                f"{game_id}.{observation_id}.size",
            )
        ):
            raise X13BatchPublishError(
                f"{game_id} Kalshi candle close/volume is inconsistent"
            )
        _decimal_nonnegative(
            row.get("open_interest"),
            f"{game_id}.{observation_id}.open_interest",
        )
        return

    if kind != "trade" or end - start != timedelta(seconds=1):
        raise X13BatchPublishError(
            f"{game_id} observation kind/time semantics are invalid"
        )
    if any(
        row.get(field) is not None
        for field in (
            "yes_bid_ohlc",
            "yes_ask_ohlc",
            "volume",
            "open_interest",
        )
    ):
        raise X13BatchPublishError(
            f"{game_id} trade observation contains candle fields"
        )
    if provenance == "derived_complement":
        if row.get("primary_path_eligible") is not False:
            raise X13BatchPublishError(
                f"{game_id} derived observation entered primary path"
            )
        return
    if provenance != "observed" or native_time is None:
        raise X13BatchPublishError(
            f"{game_id} observed trade lacks native source time"
        )
    if venue == "polymarket":
        if (
            native_time != start
            or row.get("is_block_trade") is not None
            or row.get("primary_path_eligible") is not True
        ):
            raise X13BatchPublishError(
                f"{game_id} Polymarket trade semantics are invalid"
            )
    elif venue == "kalshi":
        if (
            start.microsecond != 0
            or not start <= native_time < end
            or type(row.get("is_block_trade")) is not bool
            or row.get("primary_path_eligible")
            is not (not row["is_block_trade"])
        ):
            raise X13BatchPublishError(
                f"{game_id} Kalshi trade second/block semantics are invalid"
            )
        yes_price = _decimal_probability(
            row.get("yes_price_dollars"),
            f"{game_id}.{observation_id}.yes_price_dollars",
        )
        no_price = _decimal_probability(
            row.get("no_price_dollars"),
            f"{game_id}.{observation_id}.no_price_dollars",
        )
        if yes_price + no_price != Decimal(1):
            raise X13BatchPublishError(
                f"{game_id} Kalshi yes/no prices do not sum to one"
            )
    else:
        raise X13BatchPublishError(f"{game_id} observation venue is invalid")


_ASSOCIATION_PRESENTATION_FIELDS = {
    "episode_id",
    "episode_type",
    "logical_market_id",
    "outcome",
    "source_time_start_utc",
    "source_time_end_utc",
    "delay_scenario_seconds",
    "horizon_seconds",
    "pre_event_actual_trade",
    "first_post_event_trade",
    "vwap",
    "signed_price_change",
    "maximum_excursion",
    "net_change_60s",
    "trade_count",
    "volume",
    "staleness_seconds",
    "overshoot_candidate",
    "reversal_candidate",
    "two_venue_direction_consistency",
    "order_ambiguous",
    "contaminated",
    "validity_status",
}
_CONTRACT_INVENTORY_IDENTITY_FIELDS = (
    "contract_id",
    "condition_id",
    "venue_market_id",
    "raw_contract_ids",
    "logical_market_id",
    "family",
    "kind",
    "dependency_game_ids",
    "outcomes",
)


def contract_inventory_sha256(
    contracts: Sequence[Mapping[str, object]],
) -> str:
    """Hash the venue-native logical contract inventory, excluding observations."""

    material = [
        {
            field: contract.get(field)
            for field in _CONTRACT_INVENTORY_IDENTITY_FIELDS
        }
        for contract in sorted(
            contracts,
            key=lambda value: str(value.get("logical_market_id")),
        )
    ]
    return canonical_payload_hash(material)


def _association_partition_descriptors(
    game: Mapping[str, object],
) -> list[dict[str, object]]:
    coverage = _require_mapping(
        game.get("association_coverage"),
        f"{game.get('game_id')}.association_coverage",
    )
    return [
        dict(_require_mapping(item, "association partition"))
        for item in _require_sequence(
            coverage.get("full_table_partitions"),
            "association full_table_partitions",
        )
    ]


def _validate_game_payload(payload: object) -> dict[str, Any]:
    game = dict(_require_mapping(payload, "game payload"))
    _canonical_json_bytes(game)
    if game.get("schema") != "nfl_x13_game_bundle_v1":
        raise X13BatchPublishError("game payload has an unknown schema")
    if game.get("experiment_id") != "X-13":
        raise X13BatchPublishError("game payload is not bound to X-13")
    if game.get("status") != "PRELIMINARY_SOURCE_TIME_ONLY":
        raise X13BatchPublishError("game payload has an invalid evidence status")
    game_id = game.get("game_id")
    if type(game_id) is not str or _GAME_ID_RE.fullmatch(game_id) is None:
        raise X13BatchPublishError("game payload has an invalid game_id")
    binding = _BINDING_BY_GAME_ID.get(game_id)
    if binding is None:
        raise X13BatchPublishError("game payload is outside the frozen X-13 plan")

    teams = _require_mapping(game.get("teams"), f"{game_id}.teams")
    expected_teams = {
        "away": binding.away_team,
        "home": binding.home_team,
    }
    if any(teams.get(side) != team for side, team in expected_teams.items()):
        raise X13BatchPublishError(
            f"{game_id} teams do not match the frozen team binding"
        )
    final_score = _require_mapping(
        game.get("final_score"), f"{game_id}.final_score"
    )
    expected_final_score = {
        "away": binding.away_score,
        "home": binding.home_score,
    }
    if (
        set(final_score) != set(expected_final_score)
        or any(
            type(final_score.get(side)) is not int
            or final_score.get(side) != score
            for side, score in expected_final_score.items()
        )
    ):
        raise X13BatchPublishError(
            f"{game_id} does not match the frozen final score"
        )

    events = _require_sequence(game.get("events"), f"{game_id}.events")
    if not events:
        raise X13BatchPublishError(f"{game_id} has no canonical events")
    previous_order: int | None = None
    play_ids: set[str] = set()
    last_event: Mapping[str, Any] | None = None
    for event in events:
        row = _require_mapping(event, f"{game_id}.event")
        last_event = row
        play_id = row.get("play_id")
        order = row.get("order_sequence")
        if type(play_id) is not str or not play_id:
            raise X13BatchPublishError(f"{game_id} event lacks play_id")
        if play_id in play_ids:
            raise X13BatchPublishError(f"{game_id} contains duplicate play_id")
        play_ids.add(play_id)
        if type(order) is not int or order < 0:
            raise X13BatchPublishError(f"{game_id} event order is invalid")
        if previous_order is not None and order <= previous_order:
            raise X13BatchPublishError(f"{game_id} events are not strictly ordered")
        previous_order = order
        if (
            row.get("game_id") != game_id
            or row.get("away_team") != binding.away_team
            or row.get("home_team") != binding.home_team
        ):
            raise X13BatchPublishError(
                f"{game_id} contains a foreign event identity"
            )
        for field in ("possession_team", "offense_team", "defense_team"):
            team = row.get(field)
            if team is not None and team not in {
                binding.away_team,
                binding.home_team,
            }:
                raise X13BatchPublishError(
                    f"{game_id} event {field} is outside the game"
                )
        possession = row.get("possession_team")
        yardline = row.get("yardline_100")
        direction = row.get("direction")
        coordinate = row.get("field_coordinate_0_100")
        semantics = row.get("field_orientation_semantics")
        if possession is not None and yardline is not None:
            if type(yardline) is not int or not 0 <= yardline <= 100:
                raise X13BatchPublishError(
                    f"{game_id} event field orientation has an invalid yardline"
                )
            expected_direction = (
                "right" if possession == binding.away_team else "left"
            )
            expected_coordinate = (
                100 - yardline
                if possession == binding.away_team
                else yardline
            )
            if (
                direction != expected_direction
                or type(coordinate) is not int
                or coordinate != expected_coordinate
                or semantics != _FIELD_ORIENTATION_SEMANTICS
            ):
                raise X13BatchPublishError(
                    f"{game_id} event field orientation is inconsistent "
                    "with possession"
                )
        elif (
            direction is not None
            or coordinate is not None
            or semantics != _UNAVAILABLE_FIELD_SEMANTICS
        ):
            raise X13BatchPublishError(
                f"{game_id} event field orientation must remain unavailable"
            )
    assert last_event is not None
    if (
        last_event.get("game_end") is not True
        or type(last_event.get("away_score")) is not int
        or type(last_event.get("home_score")) is not int
        or last_event.get("away_score") != binding.away_score
        or last_event.get("home_score") != binding.home_score
    ):
        raise X13BatchPublishError(
            f"{game_id} terminal event does not prove the frozen final score"
        )

    episodes = _require_sequence(game.get("episodes"), f"{game_id}.episodes")
    episode_ids: set[str] = set()
    eligible_episode_count = 0
    for episode in episodes:
        row = _require_mapping(episode, f"{game_id}.episode")
        episode_id = row.get("episode_id")
        if type(episode_id) is not str or not episode_id:
            raise X13BatchPublishError(f"{game_id} episode lacks episode_id")
        if episode_id in episode_ids:
            raise X13BatchPublishError(f"{game_id} contains duplicate episode_id")
        episode_ids.add(episode_id)
        episode_play_ids = _require_sequence(
            row.get("play_ids"), f"{game_id}.{episode_id}.play_ids"
        )
        if any(play_id not in play_ids for play_id in episode_play_ids):
            raise X13BatchPublishError(
                f"{game_id} episode references an unknown play"
            )
        if (
            row.get("audit_only") is not True
            and row.get("nullified") is not True
            and row.get("episode_type") != "administrative"
        ):
            eligible_episode_count += 1

    contracts = _require_sequence(game.get("contracts"), f"{game_id}.contracts")
    logical_outcomes: dict[str, set[str]] = {}
    logical_venues: dict[str, str] = {}
    outcome_coverage: dict[str, dict[str, str]] = {}
    winner_venues: set[str] = set()
    eligible_outcome_count = 0
    for contract in contracts:
        row = _require_mapping(contract, f"{game_id}.contract")
        venue = row.get("venue")
        if venue not in {"polymarket", "kalshi"}:
            raise X13BatchPublishError(f"{game_id} contract has an unknown venue")
        contract_id = row.get("contract_id")
        if (
            type(contract_id) is not str
            or not contract_id.startswith(f"{venue}:")
            or contract_id == f"{venue}:"
        ):
            raise X13BatchPublishError(
                f"{game_id} contract lacks venue-native identity"
            )
        venue_market_id = row.get("venue_market_id")
        if type(venue_market_id) is not str or not venue_market_id:
            raise X13BatchPublishError(
                f"{game_id} contract lacks venue market identity"
            )
        dependencies = _require_sequence(
            row.get("dependency_game_ids"),
            f"{game_id}.contract.dependency_game_ids",
        )
        eligible = row.get("analysis_eligible")
        if type(eligible) is not bool:
            raise X13BatchPublishError(
                f"{game_id} contract eligibility is not explicit"
            )
        if eligible and (len(dependencies) != 1 or dependencies[0] != game_id):
            raise X13BatchPublishError(
                f"{game_id} cross-game contract entered single-game analysis"
            )
        logical_market_id = row.get("logical_market_id")
        if type(logical_market_id) is not str or not logical_market_id:
            raise X13BatchPublishError(f"{game_id} contract lacks stable identity")
        if logical_market_id in logical_outcomes:
            raise X13BatchPublishError(
                f"{game_id} contains duplicate logical market identity"
            )
        outcomes = _require_sequence(
            row.get("outcomes"), f"{game_id}.{logical_market_id}.outcomes"
        )
        if (
            len(outcomes) < 2
            or any(type(value) is not str or not value for value in outcomes)
            or len(set(outcomes)) != len(outcomes)
        ):
            raise X13BatchPublishError(
                f"{game_id} contract must declare at least two outcomes"
            )
        logical_outcomes[logical_market_id] = set(outcomes)
        logical_venues[logical_market_id] = venue
        if eligible:
            eligible_outcome_count += len(outcomes)
            coverage = _require_mapping(
                row.get("outcome_coverage"),
                f"{game_id}.{logical_market_id}.outcome_coverage",
            )
            if (
                set(coverage) != set(outcomes)
                or any(
                    type(value) is not str
                    or value not in _OUTCOME_COVERAGE_VALUES
                    for value in coverage.values()
                )
            ):
                raise X13BatchPublishError(
                    f"{game_id} contract outcome coverage is incomplete"
                )
            outcome_coverage[logical_market_id] = dict(coverage)

        expected_winner_id = f"{venue}:{game_id}:winner"
        if logical_market_id == expected_winner_id:
            if (
                row.get("family") != "moneyline"
                or row.get("kind") != "primitive"
                or eligible is not True
                or list(dependencies) != [game_id]
                or list(outcomes)
                != [binding.away_team, binding.home_team]
            ):
                raise X13BatchPublishError(
                    f"{game_id} has an invalid frozen winner inventory"
                )
            if venue == "polymarket":
                if (
                    contract_id
                    != f"polymarket:{binding.polymarket_condition_id}"
                    or venue_market_id != binding.polymarket_market_id
                    or row.get("condition_id")
                    != binding.polymarket_condition_id
                ):
                    raise X13BatchPublishError(
                        f"{game_id} has an invalid Polymarket winner binding"
                    )
            else:
                raw_contract_ids = _require_sequence(
                    row.get("raw_contract_ids"),
                    f"{game_id}.kalshi.raw_contract_ids",
                )
                if (
                    contract_id != f"kalshi:{binding.kalshi_event_ticker}"
                    or venue_market_id != binding.kalshi_event_ticker
                    or list(raw_contract_ids)
                    != [
                        binding.kalshi_away_ticker,
                        binding.kalshi_home_ticker,
                    ]
                ):
                    raise X13BatchPublishError(
                        f"{game_id} has an invalid Kalshi winner binding"
                    )
            winner_venues.add(venue)
    if winner_venues != {"polymarket", "kalshi"}:
        raise X13BatchPublishError(
            f"{game_id} lacks frozen Polymarket and Kalshi winner inventory"
        )
    if (
        contract_inventory_sha256(
            [dict(_require_mapping(value, f"{game_id}.contract")) for value in contracts]
        )
        != X13_FROZEN_CONTRACT_INVENTORY_SHA256_BY_GAME.get(game_id)
    ):
        raise X13BatchPublishError(
            f"{game_id} contract inventory differs from the frozen universe"
        )

    observations = _require_sequence(
        game.get("observations"), f"{game_id}.observations"
    )
    observation_ids: set[str] = set()
    observation_rows: dict[str, Mapping[str, Any]] = {}
    observed_priced_paths: set[tuple[str, str]] = set()
    for observation in observations:
        row = _require_mapping(observation, f"{game_id}.observation")
        observation_id = _require_sha256(
            row.get("observation_id"), f"{game_id}.observation_id"
        )
        if observation_payload_sha256(row) != observation_id:
            raise X13BatchPublishError(
                f"{game_id} observation_id does not match canonical material"
            )
        _validate_observation_semantics(
            row,
            game_id=game_id,
            observation_id=observation_id,
        )
        if observation_id in observation_ids:
            raise X13BatchPublishError(
                f"{game_id} contains duplicate observation_id"
            )
        observation_ids.add(observation_id)
        observation_rows[observation_id] = row
        logical_market_id = row.get("logical_market_id")
        if logical_market_id not in logical_outcomes:
            raise X13BatchPublishError(
                f"{game_id} observation references an unknown logical market"
            )
        if row.get("outcome") not in logical_outcomes[logical_market_id]:
            raise X13BatchPublishError(
                f"{game_id} observation references an unknown outcome"
            )
        if row.get("venue") != logical_venues[logical_market_id]:
            raise X13BatchPublishError(
                f"{game_id} observation venue does not match its contract"
            )
        outcome = str(row["outcome"])
        if row.get("price") is not None:
            _decimal_probability(
                row["price"], f"{game_id}.{observation_id}.price"
            )
        provenance = row.get("provenance")
        semantics = row.get("render_semantics")
        if provenance == "derived_complement":
            if semantics != "dashed":
                raise X13BatchPublishError(
                    f"{game_id} derived observation must render dashed"
                )
            if row.get("bid") is not None or row.get("ask") is not None:
                raise X13BatchPublishError(
                    f"{game_id} derived observation cannot invent bid/ask"
                )
            _require_sha256(
                row.get("derived_from_observation_id"),
                f"{game_id}.{observation_id}.derived_from_observation_id",
            )
        elif provenance == "observed":
            if semantics != "solid":
                raise X13BatchPublishError(
                    f"{game_id} observed data must render solid"
                )
            if row.get("price") is not None:
                observed_priced_paths.add((logical_market_id, outcome))
            if row.get("derived_from_observation_id") is not None:
                raise X13BatchPublishError(
                    f"{game_id} observed data cannot have a derived source"
                )
        else:
            raise X13BatchPublishError(
                f"{game_id} observation provenance is unknown"
            )
    for observation_id, row in observation_rows.items():
        if row.get("provenance") != "derived_complement":
            continue
        source_id = str(row["derived_from_observation_id"])
        source = observation_rows.get(source_id)
        if source is None or source.get("provenance") != "observed":
            raise X13BatchPublishError(
                f"{game_id} derived observation source is missing or non-observed"
            )
        invariant_fields = (
            "venue",
            "raw_market_id",
            "logical_market_id",
            "kind",
            "source_start",
            "source_end",
            "size",
            "native_id",
        )
        if any(row.get(field) != source.get(field) for field in invariant_fields):
            raise X13BatchPublishError(
                f"{game_id} derived observation does not preserve source identity"
            )
        if row.get("outcome") == source.get("outcome"):
            raise X13BatchPublishError(
                f"{game_id} derived observation does not change outcome"
            )
        if row.get("price") is None or source.get("price") is None:
            raise X13BatchPublishError(
                f"{game_id} derived observation lacks a priced source"
            )
        derived_price = _decimal_probability(
            row["price"], f"{game_id}.{observation_id}.price"
        )
        source_price = _decimal_probability(
            source["price"], f"{game_id}.{source_id}.price"
        )
        if derived_price != Decimal(1) - source_price:
            raise X13BatchPublishError(
                f"{game_id} derived observation is not the exact complement"
            )
    for logical_market_id, coverage in outcome_coverage.items():
        for outcome, status in coverage.items():
            path = (logical_market_id, outcome)
            if status == "available" and path not in observed_priced_paths:
                raise X13BatchPublishError(
                    f"{game_id} winner outcome path is missing"
                )
            if (
                status == "unavailable_no_trades"
                and path in observed_priced_paths
            ):
                raise X13BatchPublishError(
                    f"{game_id} unavailable outcome has a fabricated price path"
                )

    associations = _require_sequence(
        game.get("associations"), f"{game_id}.associations"
    )
    association_keys: set[tuple[object, ...]] = set()
    for association in associations:
        row = _require_mapping(association, f"{game_id}.association")
        if set(row) != _ASSOCIATION_PRESENTATION_FIELDS:
            raise X13BatchPublishError(
                f"{game_id} association presentation fields are not canonical"
            )
        if row.get("episode_id") not in episode_ids:
            raise X13BatchPublishError(
                f"{game_id} association references an unknown episode"
            )
        logical_market_id = row.get("logical_market_id")
        if logical_market_id not in logical_outcomes:
            raise X13BatchPublishError(
                f"{game_id} association references an unknown market"
            )
        if row.get("outcome") not in logical_outcomes[logical_market_id]:
            raise X13BatchPublishError(
                f"{game_id} association references an unknown outcome"
            )
        if row.get("delay_scenario_seconds") not in (
            X13_DELAY_SCENARIOS_SECONDS
        ):
            raise X13BatchPublishError(
                f"{game_id} association delay is outside the frozen grid"
            )
        if row.get("horizon_seconds") not in X13_HORIZONS_SECONDS:
            raise X13BatchPublishError(
                f"{game_id} association horizon is outside the frozen grid"
            )
        for field in (
            "overshoot_candidate",
            "reversal_candidate",
            "order_ambiguous",
            "contaminated",
        ):
            if type(row.get(field)) is not bool:
                raise X13BatchPublishError(
                    f"{game_id} association {field} must be boolean"
                )
        key = (
            row["episode_id"],
            logical_market_id,
            row["outcome"],
            row["delay_scenario_seconds"],
            row["horizon_seconds"],
        )
        if key in association_keys:
            raise X13BatchPublishError(
                f"{game_id} association presentation contains a duplicate"
            )
        association_keys.add(key)

    association_coverage = _require_mapping(
        game.get("association_coverage"),
        f"{game_id}.association_coverage",
    )
    if set(association_coverage) != {
        "expected_full_rows",
        "actual_full_rows",
        "presentation_row_count",
        "presentation_eligible_count",
        "presentation_omitted_count",
        "presentation_row_limit",
        "presentation_selection_order",
        "presentation_filter",
        "full_table_format",
        "full_table_partitions",
    }:
        raise X13BatchPublishError(
            f"{game_id} association coverage fields are not canonical"
        )
    expected_rows = association_coverage.get("expected_full_rows")
    actual_rows = association_coverage.get("actual_full_rows")
    presentation_rows = association_coverage.get("presentation_row_count")
    presentation_eligible = association_coverage.get(
        "presentation_eligible_count"
    )
    presentation_omitted = association_coverage.get(
        "presentation_omitted_count"
    )
    presentation_limit = association_coverage.get(
        "presentation_row_limit"
    )
    calculated_rows = (
        eligible_episode_count
        * len(X13_DELAY_SCENARIOS_SECONDS)
        * len(X13_HORIZONS_SECONDS)
        * eligible_outcome_count
    )
    if (
        type(expected_rows) is not int
        or expected_rows <= 0
        or expected_rows != calculated_rows
        or expected_rows
        != X13_FROZEN_ASSOCIATION_ROWS_BY_GAME.get(game_id)
        or actual_rows != expected_rows
        or presentation_rows != len(associations)
        or type(presentation_eligible) is not int
        or presentation_eligible < presentation_rows
        or presentation_limit != 5_000
        or presentation_rows != min(presentation_eligible, 5_000)
        or presentation_omitted
        != presentation_eligible - presentation_rows
    ):
        raise X13BatchPublishError(
            f"{game_id} association cardinality is incomplete"
        )
    if (
        association_coverage.get("presentation_filter")
        != "actual_market_evidence_and_ambiguous_or_contaminated_status"
        or association_coverage.get("presentation_selection_order")
        != "first_rows_in_canonical_association_stream_order"
        or association_coverage.get("full_table_format")
        != "partitioned_parquet"
    ):
        raise X13BatchPublishError(
            f"{game_id} association storage semantics are invalid"
        )
    partitions = _association_partition_descriptors(game)
    if not partitions:
        raise X13BatchPublishError(
            f"{game_id} association partitions are missing"
        )
    partition_paths: set[str] = set()
    partition_rows = 0
    for descriptor in partitions:
        if set(descriptor) != {
            "relative_path",
            "row_count",
            "sha256",
            "schema_fingerprint",
        }:
            raise X13BatchPublishError(
                f"{game_id} association partition fields are not canonical"
            )
        relative_path = _validate_auxiliary_relative_path(
            descriptor.get("relative_path")
        )
        if (
            not relative_path.startswith(
                f"data/associations/game_id={game_id}/"
            )
            or not relative_path.endswith(".parquet")
            or relative_path in partition_paths
        ):
            raise X13BatchPublishError(
                f"{game_id} association partition path is invalid"
            )
        partition_paths.add(relative_path)
        row_count = descriptor.get("row_count")
        if type(row_count) is not int or row_count <= 0:
            raise X13BatchPublishError(
                f"{game_id} association partition row_count is invalid"
            )
        partition_rows += row_count
        _require_sha256(
            descriptor.get("sha256"),
            f"{game_id} association partition sha256",
        )
        if (
            descriptor.get("schema_fingerprint")
            != X13_ASSOCIATION_SCHEMA_FINGERPRINT
        ):
            raise X13BatchPublishError(
                f"{game_id} association schema fingerprint is invalid"
            )
    if partition_rows != actual_rows:
        raise X13BatchPublishError(
            f"{game_id} association partition row sum is incomplete"
        )

    audit = _require_mapping(game.get("audit"), f"{game_id}.audit")
    for layer, accepted in _PASS_LAYER_VALUES.items():
        if audit.get(layer) not in accepted:
            raise X13BatchPublishError(
                f"{game_id} Layer G/M/A source audit is not green: {layer}"
            )
    if audit.get("manifest_hashes_verified") is not True:
        raise X13BatchPublishError(f"{game_id} raw manifests are not verified")

    lineage = _require_mapping(game.get("lineage"), f"{game_id}.lineage")
    raw_hashes = _require_sequence(
        lineage.get("raw_manifest_hashes"), f"{game_id}.raw_manifest_hashes"
    )
    if not raw_hashes:
        raise X13BatchPublishError(f"{game_id} has no raw manifest lineage")
    for index, value in enumerate(raw_hashes):
        _require_sha256(value, f"{game_id}.raw_manifest_hashes[{index}]")
    _require_sha256(lineage.get("analysis_lock"), f"{game_id}.analysis_lock")
    if type(lineage.get("builder_version")) is not str or not lineage["builder_version"]:
        raise X13BatchPublishError(f"{game_id} lacks builder version")

    return game


def build_batch_manifest(
    game_payloads: Sequence[Mapping[str, object]],
    *,
    plan_id: str,
    builder_version: str,
    auxiliary_artifacts: Sequence[
        AuxiliaryArtifactV1 | Mapping[str, object]
    ] = (),
) -> dict[str, Any]:
    """Validate all games and calculate the immutable X-13 batch identity."""

    _require_sha256(plan_id, "plan_id")
    if plan_id != X13_BATCH_SPEC.plan_id:
        raise X13BatchPublishError(
            "plan_id does not match the frozen X-13 plan"
        )
    if type(builder_version) is not str or not builder_version:
        raise X13BatchPublishError("builder_version must be nonempty")
    values = _require_sequence(game_payloads, "game_payloads")
    games = [_validate_game_payload(value) for value in values]
    by_id: dict[str, dict[str, Any]] = {}
    for game in games:
        game_id = game["game_id"]
        if game_id in by_id:
            raise X13BatchPublishError("batch contains duplicate game IDs")
        by_id[game_id] = game
    if set(by_id) != set(X13_GAME_IDS) or len(by_id) != 20:
        raise X13BatchPublishError("batch must contain the exact frozen 20 games")

    ordered_games = [by_id[game_id] for game_id in X13_GAME_IDS]
    game_hashes = {
        game["game_id"]: canonical_payload_hash(game) for game in ordered_games
    }
    raw_hashes = sorted(
        {
            value
            for game in ordered_games
            for value in game["lineage"]["raw_manifest_hashes"]
        }
    )
    auxiliary_descriptors = _normalize_auxiliary_descriptors(
        auxiliary_artifacts
    )
    _validate_required_auxiliary_descriptors(auxiliary_descriptors)
    auxiliary_by_path = {
        str(item["relative_path"]): item
        for item in auxiliary_descriptors
    }
    required_association_paths: set[str] = set()
    association_expected_row_count = 0
    for game in ordered_games:
        coverage = _require_mapping(
            game["association_coverage"],
            f"{game['game_id']}.association_coverage",
        )
        association_expected_row_count += int(
            coverage["expected_full_rows"]
        )
        for required in _association_partition_descriptors(game):
            relative_path = str(required["relative_path"])
            required_association_paths.add(relative_path)
            actual = auxiliary_by_path.get(relative_path)
            if (
                actual is None
                or actual["role"] != "association_candidate_table"
                or actual["media_type"]
                != "application/vnd.apache.parquet"
                or actual["row_count"] != required["row_count"]
                or actual["sha256"] != required["sha256"]
                or actual["schema_fingerprint"]
                != required["schema_fingerprint"]
            ):
                raise X13BatchPublishError(
                    f"{game['game_id']} association partition is not "
                    "bound to an auxiliary artifact"
                )
    declared_association_paths = {
        str(item["relative_path"])
        for item in auxiliary_descriptors
        if item["role"] == "association_candidate_table"
    }
    if declared_association_paths != required_association_paths:
        raise X13BatchPublishError(
            "association auxiliary partition set does not match all games"
        )
    if association_expected_row_count != X13_FROZEN_ASSOCIATION_ROW_COUNT:
        raise X13BatchPublishError(
            "batch association cardinality differs from the frozen universe"
        )
    identity_material = {
        "experiment_id": "X-13",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "plan_id": plan_id,
        "universe_id": X13_FROZEN_UNIVERSE_ID,
        "builder_version": builder_version,
        "game_ids": list(X13_GAME_IDS),
        "game_hashes": game_hashes,
        "raw_manifest_hashes": raw_hashes,
        "auxiliary_artifacts": auxiliary_descriptors,
        "association_expected_row_count": association_expected_row_count,
    }
    batch_id = canonical_payload_hash(identity_material)
    return {
        "schema": "nfl_x13_batch_manifest_v1",
        "experiment_id": "X-13",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "plan_id": plan_id,
        "universe_id": X13_FROZEN_UNIVERSE_ID,
        "batch_id": batch_id,
        "builder_version": builder_version,
        "game_count": 20,
        "game_ids": list(X13_GAME_IDS),
        "game_hashes": game_hashes,
        "raw_manifest_hashes": raw_hashes,
        "auxiliary_artifacts": auxiliary_descriptors,
        "association_expected_row_count": association_expected_row_count,
        "publication_gate": "PASS",
        "claim_boundary": {
            "source_time_association_only": True,
            "causality": False,
            "real_latency": False,
            "execution": False,
            "tradeable_alpha": False,
        },
    }


def _embedded_json(payload: Mapping[str, object]) -> str:
    text = _canonical_json_bytes(payload).decode("utf-8")
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_game_html(game_payload: Mapping[str, object]) -> str:
    """Render one generic, dependency-free replay and source-time market review."""

    game = _validate_game_payload(game_payload)
    game_id = html.escape(game["game_id"])
    away = html.escape(str(game["teams"]["away"]))
    home = html.escape(str(game["teams"]["home"]))
    embedded = _embedded_json(game)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{game_id} · X-13 source-time review</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2a;--panel2:#132337;--line:#28415d;--text:#eef6ff;--muted:#9bb0c6;--away:#2563eb;--home:#ef4444;--ok:#2dd4bf;--warn:#f59e0b}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(155deg,#06101d,#0b1828 55%,#08131f);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
button,select,input{{font:inherit}} .shell{{max-width:1520px;margin:0 auto;padding:22px}} header{{display:flex;gap:16px;justify-content:space-between;align-items:flex-start;margin-bottom:18px}}
h1{{font-size:clamp(22px,3vw,38px);line-height:1.05;margin:5px 0}} .eyebrow{{color:#7dd3fc;letter-spacing:.12em;text-transform:uppercase;font-size:11px}}
.badges{{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}} .badge{{border:1px solid var(--line);border-radius:999px;padding:4px 9px;color:var(--muted);background:#0a1625}}
.badge.ok{{border-color:#12685d;color:#5eead4}} .score{{font-size:24px;font-weight:800;white-space:nowrap}} .score .away{{color:var(--away)}} .score .home{{color:var(--home)}}
.grid{{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(360px,.85fr);gap:14px}} .panel{{background:rgba(13,27,42,.94);border:1px solid var(--line);border-radius:15px;padding:15px;box-shadow:0 18px 60px rgba(0,0,0,.2)}}
.panel h2{{font-size:16px;margin:0 0 10px}} .wide{{grid-column:1/-1}} .field-wrap{{background:#173f2b;border-radius:12px;padding:8px;overflow:hidden}} #field{{display:block;width:100%;height:auto}}
.controls{{display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:12px;align-items:center}} #play-slider{{width:100%}} .play-index{{font-variant-numeric:tabular-nums;color:#bae6fd}}
.teams{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}} .team-card{{border-radius:10px;padding:10px;background:#0a1726;border-left:4px solid}} .team-card.away{{border-color:var(--away)}} .team-card.home{{border-color:var(--home)}}
.state-grid{{display:grid;grid-template-columns:repeat(4,minmax(90px,1fr));gap:8px}} .metric{{background:#091624;border:1px solid #1e344d;border-radius:9px;padding:8px}} .metric b{{display:block;font-size:10px;text-transform:uppercase;color:var(--muted);letter-spacing:.06em}} .metric span{{font-size:15px;font-weight:700}}
.desc{{margin-top:10px;padding:10px;border-left:3px solid #38bdf8;background:#081521;border-radius:0 8px 8px 0;min-height:44px}} .flags{{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}} .flag{{padding:3px 7px;border-radius:5px;background:#20344b;color:#dbeafe;font-size:11px}} .flag.score{{background:#0f766e}} .flag.turnover{{background:#9f1239}} .flag.warn{{background:#92400e}}
.roster{{display:grid;grid-template-columns:1fr 1fr;gap:10px}} .roster ul{{margin:6px 0 0;padding-left:20px;columns:2;color:#cbd5e1}} .roster-change{{margin-top:10px;color:#fbbf24}}
.market-controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}} .market-controls label{{color:var(--muted);font-size:12px}} .market-controls select{{margin-left:6px;background:#102238;border:1px solid #2b4563;color:#fff;border-radius:7px;padding:5px 8px}}
.tabs{{display:flex;gap:6px;overflow:auto;flex:1;min-width:280px}} .tab{{background:#102238;border:1px solid #2b4563;color:#cbd5e1;border-radius:8px;padding:6px 9px;cursor:pointer;white-space:nowrap}} .tab.active{{color:white;border-color:#38bdf8;background:#153452}}
#market-chart{{width:100%;height:350px;background:#081521;border-radius:10px;display:block}} .legend{{display:flex;gap:13px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}} .line-sample{{display:inline-block;width:24px;border-top:3px solid;vertical-align:middle;margin-right:5px}} .line-sample.derived{{border-top-style:dashed}}
.market-detail{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:8px;margin-top:10px}} .market-detail div{{background:#081521;border:1px solid #1e344d;border-radius:8px;padding:8px;color:#b9d3ea;font-size:12px}} .market-detail b{{color:#fff}}
.coverage{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .coverage div{{padding:9px;background:#081521;border-radius:8px}} .coverage strong{{display:block;color:#fff}} table{{width:100%;border-collapse:collapse}} th,td{{padding:8px;text-align:left;border-bottom:1px solid #1e344d;vertical-align:top}} th{{color:#8fb0cd;font-size:11px;text-transform:uppercase;position:sticky;top:0;background:#0d1b2a}} .table-wrap{{max-height:480px;overflow:auto}}
pre{{white-space:pre-wrap;word-break:break-word;background:#07131f;padding:10px;border-radius:8px;color:#b9d3ea;max-height:300px;overflow:auto}} .empty{{fill:#6b8298;font-size:13px}} footer{{color:#7890a8;padding:18px 4px;text-align:center}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}.wide{{grid-column:auto}}.state-grid{{grid-template-columns:repeat(2,1fr)}}.coverage{{grid-template-columns:1fr}}header{{flex-direction:column}}}}
</style>
</head>
<body>
<main class="shell">
<header>
  <div><div class="eyebrow">X-13 · NFL 2025 · source-time only</div><h1>{away} @ {home}</h1>
  <div class="badges"><span class="badge ok">PRELIMINARY_SOURCE_TIME_ONLY</span><span class="badge">20 场统一标准</span><span class="badge">无真实 latency / 因果 / 执行结论</span></div></div>
  <div class="score"><span class="away">{away}</span> <span id="away-score">0</span> — <span id="home-score">0</span> <span class="home">{home}</span></div>
</header>
<section class="grid">
  <article class="panel">
	    <h2>橄榄球场 · 规范化球队端区方向与逐 play 位置（非 tracking 坐标）</h2>
    <div class="field-wrap"><svg id="field" viewBox="0 0 1100 420" role="img" aria-label="football field"></svg></div>
    <div class="controls"><input id="play-slider" type="range" min="0" step="1" value="0"><span class="play-index" id="play-index"></span></div>
    <div class="teams"><div class="team-card away"><b>进攻</b><div id="offense"></div></div><div class="team-card home"><b>防守</b><div id="defense"></div></div></div>
    <div class="state-grid" id="state-grid"></div><div class="desc" id="description"></div><div class="flags" id="flags"></div>
  </article>
  <aside class="panel">
    <h2>场上人员与人员包</h2>
    <div class="roster"><div><b id="offense-roster-title">Offense</b><ul id="offense-roster"></ul></div><div><b id="defense-roster-title">Defense</b><ul id="defense-roster"></ul></div></div>
    <div class="roster-change" id="roster-change"></div>
    <h2 style="margin-top:16px">当前行完整参数</h2><pre id="raw-state"></pre>
  </aside>
  <article class="panel wide">
    <h2>Prediction market source-time 趋势</h2><div class="market-controls"><div class="tabs" id="market-tabs"></div><label>Delay scenario <select id="delay-scenario"><option value="0">0s</option><option value="1">1s</option><option value="2">2s</option><option value="3">3s</option><option value="5">5s</option><option value="10">10s</option></select></label></div><svg id="market-chart" viewBox="0 0 1200 350" role="img" aria-label="market source-time chart"></svg>
    <div class="legend"><span><i class="line-sample" style="border-color:#38bdf8"></i>observed actual trade</span><span><i class="line-sample derived" style="border-color:#f59e0b"></i>derived / non-observed / non-executable</span><span>BBO bid / BBO ask = Kalshi 原始 1 分钟 candle OHLC</span><span>volume / open interest 为原始 candle 字段</span><span>Polymarket：历史 bid/ask unavailable</span></div>
    <div class="market-detail" id="market-candle-detail"></div>
  </article>
  <article class="panel wide"><h2>Finalized event episode × market path</h2><div class="table-wrap"><table><thead><tr><th>Episode / type</th><th>Source-time interval</th><th>Contract / outcome</th><th>Delay / Horizon</th><th>Pre / first / VWAP</th><th>Δ / excursion / 60s</th><th>Path / coverage</th><th>状态</th></tr></thead><tbody id="association-body"></tbody></table></div></article>
  <article class="panel wide"><h2>覆盖与审计门</h2><div class="coverage" id="coverage"></div><pre id="audit"></pre></article>
</section>
<footer>{game_id} · standalone offline artifact · raw hashes + builder + analysis lock embedded</footer>
</main>
<script id="x13-data" type="application/json">{embedded}</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("x13-data").textContent);
const events=D.events, slider=document.getElementById("play-slider");
slider.max=Math.max(0,events.length-1);
const esc=s=>String(s??"—").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const clock=s=>s==null?"—":`${{Math.floor(s/60)}}:${{String(s%60).padStart(2,"0")}}`;
function field(e){{
 const svg=document.getElementById("field"), away=D.teams.away_color||"#2563eb", home=D.teams.home_color||"#ef4444";
 let out=`<defs><pattern id="grass" width="20" height="20" patternUnits="userSpaceOnUse"><rect width="20" height="20" fill="#174c31"/><rect width="10" height="20" fill="#19472f"/></pattern><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#fde68a"/></marker></defs><rect x="20" y="20" width="1060" height="380" rx="8" fill="url(#grass)" stroke="#d1fae5" stroke-width="3"/>`;
 for(let i=0;i<=100;i+=10){{const x=50+i*10;out+=`<line x1="${{x}}" y1="20" x2="${{x}}" y2="400" stroke="#d1fae5" opacity=".65"/><text x="${{x}}" y="65" text-anchor="middle" fill="#d1fae5" font-size="22">${{i<=50?i:100-i}}</text><text x="${{x}}" y="380" text-anchor="middle" fill="#d1fae5" font-size="22">${{i<=50?i:100-i}}</text>`}}
	 const rawCoordinate=e.field_coordinate_0_100; const coordinate=rawCoordinate==null?NaN:Number(rawCoordinate); const hasField=Number.isFinite(coordinate)&&(e.direction==="left"||e.direction==="right");
	 if(hasField){{const x=50+coordinate*10,dir=e.direction==="left"?-1:1;out+=`<line x1="${{x}}" y1="105" x2="${{x}}" y2="345" stroke="#facc15" stroke-width="7" opacity=".9"/><line x1="${{x-dir*90}}" y1="210" x2="${{x+dir*90}}" y2="210" stroke="#fde68a" stroke-width="6" marker-end="url(#arrow)"/><circle cx="${{x}}" cy="210" r="23" fill="${{e.offense_team===D.teams.away?away:home}}" stroke="#fff" stroke-width="4"/><text x="${{x}}" y="217" fill="#fff" text-anchor="middle" font-weight="800">${{esc(e.offense_team||"?")}}</text>`}}else{{out+='<text x="550" y="220" text-anchor="middle" fill="#d1fae5">field coordinate / direction unavailable</text>'}}
 svg.innerHTML=out;
}}
const metrics=[["Source UTC",e=>e.source_time_start_utc||"—"],["Quarter","quarter"],["Clock",e=>clock(e.clock_seconds_remaining)],["Down","down"],["Distance","distance"],["Yardline 100","yardline_100"],["Drive","drive_id"],["Possession","possession_team"],["Direction","direction"],["Play family","play_family"],["Formation","formation"],["Offense pkg","offense_personnel"],["Defense pkg","defense_personnel"]];
function list(id,names){{document.getElementById(id).innerHTML=(names||[]).map(n=>`<li>${{esc(n)}}</li>`).join("")||"<li>未覆盖</li>"}}
function renderPlay(i){{
 const e=events[i]; if(!e)return; field(e); document.getElementById("play-index").textContent=`${{i+1}} / ${{events.length}} · play ${{e.play_id}}`;
 document.getElementById("away-score").textContent=e.away_score??"—";document.getElementById("home-score").textContent=e.home_score??"—";
 document.getElementById("offense").innerHTML=`<b style="color:${{e.offense_team===D.teams.away?D.teams.away_color:D.teams.home_color}}">${{esc(e.offense_team)}}</b> · ${{esc(e.possession_team)}} possession`;
 document.getElementById("defense").innerHTML=`<b style="color:${{e.defense_team===D.teams.away?D.teams.away_color:D.teams.home_color}}">${{esc(e.defense_team)}}</b>`;
 document.getElementById("state-grid").innerHTML=metrics.map(([label,key])=>{{const value=typeof key==="function"?key(e):e[key];return`<div class="metric"><b>${{label}}</b><span>${{esc(value)}}</span></div>`}}).join("");
 document.getElementById("description").textContent=e.description||"无文字描述";
 const flags=[]; for(const [key,label,cls] of [["touchdown","TD","score"],["turnover","Turnover","turnover"],["safety","Safety","score"],["blocked_kick","Blocked kick","warn"],["timeout","Timeout","warn"],["review","Review","warn"],["no_play","No play","warn"],["game_end","Game end","warn"]])if(e[key])flags.push(`<span class="flag ${{cls}}">${{label}}</span>`);if(e.play_family==="field_goal")flags.push('<span class="flag score">Field goal attempt</span>'); if(e.penalty_status&&e.penalty_status!=="none")flags.push(`<span class="flag warn">Penalty: ${{esc(e.penalty_status)}}</span>`);document.getElementById("flags").innerHTML=flags.join("")||'<span class="flag">routine</span>';
 document.getElementById("offense-roster-title").textContent=`${{e.offense_team||"Offense"}} · ${{e.formation||"formation unknown"}}`;document.getElementById("defense-roster-title").textContent=`${{e.defense_team||"Defense"}}`;list("offense-roster",e.offense_names);list("defense-roster",e.defense_names);
 document.getElementById("roster-change").textContent=e.between_play_roster_change?`between_play_roster_change: ${{JSON.stringify(e.between_play_roster_change)}}`:"相邻 play roster-set 无已记录变化";
 document.getElementById("raw-state").textContent=JSON.stringify(e,null,2);
}}
slider.addEventListener("input",()=>renderPlay(Number(slider.value))); renderPlay(0);
const contracts=D.contracts.filter(c=>c.analysis_eligible), tabs=document.getElementById("market-tabs"),delaySelect=document.getElementById("delay-scenario");
let active=contracts[0]?.logical_market_id||contracts[0]?.venue_market_id||null,activeDelay=0;
function marketId(c){{return c.logical_market_id||c.venue_market_id}}
function chart(){{
 const c=contracts.find(x=>marketId(x)===active), svg=document.getElementById("market-chart"), detail=document.getElementById("market-candle-detail"); if(!c){{svg.innerHTML='<text x="30" y="50" class="empty">该场没有通过 preflight 的可分析合约</text>';detail.textContent="";return}}
	 const all=D.observations.filter(o=>o.logical_market_id===active), obs=all.filter(o=>o.price!=null), bbo=all.filter(o=>o.bid!=null&&o.ask!=null), parse=t=>Date.parse(t); const times=all.flatMap(o=>[parse(o.source_start),parse(o.source_end)]).filter(Number.isFinite);
	 if(!times.length){{svg.innerHTML=(c.outcomes||[]).map((outcome,idx)=>`<text x="30" y="${{50+idx*24}}" class="empty">${{esc(outcome)}} · coverage unavailable · no fabricated path</text>`).join("");detail.textContent="无可展示的历史 candle";return}}
 const min=Math.min(...times),max=Math.max(...times),span=Math.max(1000,max-min),x=t=>60+(parse(t)-min)/span*1000,y=p=>315-Number(p)*280,utc=t=>new Date(t).toISOString().slice(11,19);
 let out='<line x1="60" y1="35" x2="60" y2="315" stroke="#47627f"/><line x1="60" y1="315" x2="1160" y2="315" stroke="#47627f"/>';
 for(const p of [0,.25,.5,.75,1])out+=`<line x1="55" y1="${{y(p)}}" x2="1160" y2="${{y(p)}}" stroke="#20364d"/><text x="48" y="${{y(p)+4}}" text-anchor="end" fill="#8ba2b8">${{p.toFixed(2)}}</text>`;
 for(let i=0;i<=4;i++){{const tick=min+span*i/4,tx=60+1000*i/4;out+=`<line x1="${{tx}}" y1="315" x2="${{tx}}" y2="321" stroke="#7890a8"/><text x="${{tx}}" y="338" text-anchor="middle" fill="#8ba2b8">${{utc(tick)}} UTC</text>`}}
 const overlays=(D.associations||[]).filter(a=>a.logical_market_id===active&&Number(a.delay_scenario_seconds)===activeDelay);const seenEpisodes=new Set();for(const a of overlays){{if(seenEpisodes.has(a.episode_id))continue;seenEpisodes.add(a.episode_id);const t=parse(a.source_time_start_utc);if(!Number.isFinite(t)||t<min||t>max)continue;const color=a.contaminated?"#ef4444":a.order_ambiguous?"#f59e0b":"#34d399";out+=`<line x1="${{x(a.source_time_start_utc)}}" y1="35" x2="${{x(a.source_time_start_utc)}}" y2="315" stroke="${{color}}" stroke-width="1.5" opacity=".65"/><text x="${{x(a.source_time_start_utc)+3}}" y="48" fill="${{color}}" font-size="10">${{esc(a.episode_type||"event")}}</text>`}}
 const palette=["#38bdf8","#f59e0b","#a78bfa","#34d399"];
 (c.outcomes||[]).forEach((outcome,idx)=>{{
  const color=palette[idx%palette.length];
  const outcomeRows=obs.filter(o=>o.outcome===outcome).sort((a,b)=>parse(a.source_start)-parse(b.source_start));
  const observedRows=outcomeRows.filter(o=>o.provenance==="observed");
  const derivedRows=outcomeRows.filter(o=>o.provenance==="derived_complement");
  const quotes=bbo.filter(o=>o.outcome===outcome).sort((a,b)=>parse(a.source_end)-parse(b.source_end));
  if(observedRows.length){{
   const points=observedRows.map(o=>`${{x(o.source_start)}},${{y(o.price)}}`).join(" ");
   out+=`<polyline data-provenance="observed" points="${{points}}" fill="none" stroke="${{color}}" stroke-width="3"/>${{observedRows.map(o=>`<circle cx="${{x(o.source_start)}}" cy="${{y(o.price)}}" r="3.5" fill="${{color}}"/>`).join("")}}<text x="${{x(observedRows.at(-1).source_start)-8}}" y="${{y(observedRows.at(-1).price)}}" text-anchor="end" fill="${{color}}">${{esc(outcome)}} observed trade</text>`;
  }}
  if(derivedRows.length){{
   const points=derivedRows.map(o=>`${{x(o.source_start)}},${{y(o.price)}}`).join(" ");
   out+=`<polyline data-provenance="derived_complement" points="${{points}}" fill="none" stroke="${{color}}" stroke-width="3" stroke-dasharray="10 8"/>${{derivedRows.map(o=>`<circle cx="${{x(o.source_start)}}" cy="${{y(o.price)}}" r="3.5" fill="none" stroke="${{color}}"/>`).join("")}}<text x="${{x(derivedRows.at(-1).source_start)-8}}" y="${{y(derivedRows.at(-1).price)}}" text-anchor="end" fill="${{color}}">${{esc(outcome)}} derived / non-observed</text>`;
  }}
  if(quotes.length){{
   const candleQuotes=quotes.filter(o=>o.yes_bid_ohlc&&o.yes_ask_ohlc);
   out+=candleQuotes.map(o=>{{const cx=x(o.source_end),bidO=o.yes_bid_ohlc,askO=o.yes_ask_ohlc,title=`${{outcome}} · ${{o.source_start}} → ${{o.source_end}} · bid O/H/L/C ${{bidO.open}}/${{bidO.high}}/${{bidO.low}}/${{bidO.close}} · ask O/H/L/C ${{askO.open}}/${{askO.high}}/${{askO.low}}/${{askO.close}} · volume ${{o.volume}} · open interest ${{o.open_interest}}`;return`<g class="candle-bbo"><title>${{esc(title)}}</title><line x1="${{cx-4}}" y1="${{y(bidO.high)}}" x2="${{cx-4}}" y2="${{y(bidO.low)}}" stroke="${{color}}" stroke-width="3"/><line x1="${{cx-9}}" y1="${{y(bidO.open)}}" x2="${{cx-4}}" y2="${{y(bidO.open)}}" stroke="${{color}}" stroke-width="2"/><line x1="${{cx-4}}" y1="${{y(bidO.close)}}" x2="${{cx+1}}" y2="${{y(bidO.close)}}" stroke="${{color}}" stroke-width="2"/><line x1="${{cx+4}}" y1="${{y(askO.high)}}" x2="${{cx+4}}" y2="${{y(askO.low)}}" stroke="${{color}}" stroke-width="3" opacity=".65"/><line x1="${{cx-1}}" y1="${{y(askO.open)}}" x2="${{cx+4}}" y2="${{y(askO.open)}}" stroke="${{color}}" stroke-width="2" opacity=".65"/><line x1="${{cx+4}}" y1="${{y(askO.close)}}" x2="${{cx+9}}" y2="${{y(askO.close)}}" stroke="${{color}}" stroke-width="2" opacity=".65"/></g>`}}).join("");
   out+=quotes.map(o=>`<line x1="${{x(o.source_end)}}" y1="${{y(o.ask)}}" x2="${{x(o.source_end)}}" y2="${{y(o.bid)}}" stroke="${{color}}" stroke-width="8" opacity=".28"/><circle class="bbo-point" cx="${{x(o.source_end)}}" cy="${{y(o.bid)}}" r="3" fill="${{color}}"/><circle class="bbo-point" cx="${{x(o.source_end)}}" cy="${{y(o.ask)}}" r="3" fill="${{color}}"/>`).join("");
   const bidPoints=quotes.map(o=>`${{x(o.source_end)}},${{y(o.bid)}}`).join(" "),askPoints=quotes.map(o=>`${{x(o.source_end)}},${{y(o.ask)}}`).join(" ");
   out+=`<polyline points="${{bidPoints}}" fill="none" stroke="${{color}}" stroke-width="2" stroke-dasharray="5 4"/><polyline points="${{askPoints}}" fill="none" stroke="${{color}}" stroke-width="2" stroke-dasharray="2 4"/><text x="${{x(quotes.at(-1).source_end)-8}}" y="${{y(quotes.at(-1).bid)+12}}" text-anchor="end" fill="${{color}}">${{esc(outcome)}} BBO bid / BBO ask</text>`;
  }}
  if(!outcomeRows.length&&!quotes.length)out+=`<text x="70" y="${{55+idx*22}}" class="empty">${{esc(outcome)}} · coverage unavailable · no fabricated path</text>`;
 }});
 svg.innerHTML=out;
 const ohlc=v=>["open","high","low","close"].map(key=>esc(v?.[key])).join(" / ");
 const candleRows=bbo.filter(o=>o.yes_bid_ohlc&&o.yes_ask_ohlc).sort((a,b)=>parse(a.source_end)-parse(b.source_end));
 detail.innerHTML=candleRows.length?candleRows.slice(-8).map(o=>`<div><b>${{esc(o.outcome)}} · ${{esc(o.source_start)}} → ${{esc(o.source_end)}}</b><br>Bid O/H/L/C: ${{ohlc(o.yes_bid_ohlc)}}<br>Ask O/H/L/C: ${{ohlc(o.yes_ask_ohlc)}}<br>volume / open interest: ${{esc(o.volume)}} / ${{esc(o.open_interest)}}</div>`).join(""):"<div>该逻辑 market 没有 Kalshi 历史 candle OHLC；不从成交价伪造 bid/ask。</div>";
}}
function tabsRender(){{tabs.innerHTML=contracts.map(c=>`<button class="tab ${{marketId(c)===active?"active":""}}" data-id="${{esc(marketId(c))}}">${{esc(c.family)}} · ${{esc(c.venue)}}</button>`).join("");tabs.querySelectorAll("button").forEach(b=>b.onclick=()=>{{active=b.dataset.id;tabsRender();chart()}})}}delaySelect.addEventListener("change",()=>{{activeDelay=Number(delaySelect.value);chart()}}); tabsRender();chart();
document.getElementById("association-body").innerHTML=(D.associations||[]).map(a=>`<tr><td>${{esc(a.episode_id)}}<br>${{esc(a.episode_type)}}</td><td>${{esc(a.source_time_start_utc)}} → ${{esc(a.source_time_end_utc)}}</td><td>${{esc(a.logical_market_id)}}<br>${{esc(a.outcome)}}</td><td>${{esc(a.delay_scenario_seconds)}}s / ${{esc(a.horizon_seconds)}}s</td><td>${{esc(a.pre_event_actual_trade)}} / ${{esc(a.first_post_event_trade)}} / ${{esc(a.vwap)}}</td><td>${{esc(a.signed_price_change)}} / ${{esc(a.maximum_excursion)}} / ${{esc(a.net_change_60s)}}</td><td>trades=${{esc(a.trade_count)}} · volume=${{esc(a.volume)}} · stale=${{esc(a.staleness_seconds)}}<br>overshoot_candidate=${{esc(a.overshoot_candidate)}} · reversal_candidate=${{esc(a.reversal_candidate)}} · venues=${{esc(a.two_venue_direction_consistency)}}</td><td>${{a.order_ambiguous?"order_ambiguous ":""}}${{a.contaminated?"contaminated ":""}}${{esc(a.validity_status||"eligible")}}</td></tr>`).join("")||'<tr><td colspan="8">暂无满足 actual-observation 与 isolation 门的 association candidate</td></tr>';
const cover=[["Polymarket BBO",D.market_coverage?.polymarket_historical_bbo||"unavailable"],["Kalshi BBO",D.market_coverage?.kalshi_historical_bbo||"1m only"],["Personnel",D.personnel?.coverage||"research_only"],["Canonical events",D.events.length],["Finalized episodes",D.episodes.length],["Analysis contracts",contracts.length]];
document.getElementById("coverage").innerHTML=cover.map(([k,v])=>`<div><strong>${{esc(k)}}</strong>${{esc(v)}}</div>`).join("");document.getElementById("audit").textContent=JSON.stringify({{audit:D.audit,lineage:D.lineage,status:D.status}},null,2);
</script>
</body>
</html>
"""


def _render_index(manifest: Mapping[str, object], games: Mapping[str, Mapping[str, object]]) -> str:
    rows: list[str] = []
    for game_id in X13_GAME_IDS:
        game = games[game_id]
        teams = game["teams"]
        score = game.get("final_score", {})
        rows.append(
            "<tr>"
            f"<td><a href=\"games/{html.escape(game_id)}.html\">{html.escape(game_id)}</a></td>"
            f"<td>{html.escape(str(teams['away']))} @ {html.escape(str(teams['home']))}</td>"
            f"<td>{html.escape(str(score.get('away', '—')))}–{html.escape(str(score.get('home', '—')))}</td>"
            f"<td>{len(game['events'])}</td><td>{len(game['episodes'])}</td>"
            f"<td>{len(game['contracts'])}</td><td>{len(game['observations'])}</td>"
            "<td>PASS_SOURCE_TIME_ONLY</td></tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL X-13 · 20 game batch</title><style>
body{{margin:0;background:#07111f;color:#e6f2ff;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1300px;margin:auto;padding:28px}}h1{{font-size:34px}}.note{{padding:12px;border:1px solid #31506e;border-radius:10px;color:#b7cde1}}table{{width:100%;border-collapse:collapse;margin-top:18px;background:#0d1b2a}}th,td{{padding:10px;border-bottom:1px solid #28415d;text-align:left}}th{{color:#8bb8dc}}a{{color:#67d4ff}}code{{color:#73e0c5}}</style></head><body><main><div>NFL 2025 · X-13</div><h1>20 场全深度 Game State × Prediction Market</h1><p class="note">状态：PRELIMINARY_SOURCE_TIME_ONLY。这里展示 source-time 关联与 signal candidates；不声称真实 latency、因果、执行性或可交易 alpha。</p><p>Batch <code>{html.escape(str(manifest["batch_id"]))}</code></p><table><thead><tr><th>Game ID</th><th>Matchup</th><th>Final</th><th>Plays</th><th>Episodes</th><th>Contracts</th><th>Observations</th><th>Gate</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main></body></html>"""


def _write_file(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(contents)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_auxiliary_file(
    artifact: AuxiliaryArtifactV1,
    destination: Path,
) -> None:
    source = artifact.source_path
    if source.is_symlink():
        raise X13BatchPublishError(
            f"auxiliary artifact source cannot be a symlink: {source}"
        )
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise X13BatchPublishError(
            f"auxiliary artifact source is unreadable: {source}"
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_descriptor: int | None = None
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise X13BatchPublishError(
                f"auxiliary artifact source is not a regular file: {source}"
            )
        if source_stat.st_size != artifact.byte_length:
            raise X13BatchPublishError(
                f"auxiliary artifact source byte length changed: {source}"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        digest = hashlib.sha256()
        byte_length = 0
        while True:
            chunk = os.read(source_descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_length += len(chunk)
            view = memoryview(chunk)
            written = 0
            while written < len(view):
                count = os.write(destination_descriptor, view[written:])
                if count <= 0:
                    raise OSError("short write")
                written += count
        observed = f"sha256:{digest.hexdigest()}"
        if observed != artifact.sha256:
            raise X13BatchPublishError(
                f"auxiliary artifact source hash changed: {source}"
            )
        if byte_length != artifact.byte_length:
            raise X13BatchPublishError(
                f"auxiliary artifact source byte length changed: {source}"
            )
        os.fsync(destination_descriptor)
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_association_parquet(
    path: Path,
    descriptor: Mapping[str, object],
) -> None:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise X13BatchPublishError(
            f"association auxiliary artifact is not valid Parquet: {path.name}"
        ) from error
    if (
        descriptor.get("schema_fingerprint")
        != X13_ASSOCIATION_SCHEMA_FINGERPRINT
        or parquet.metadata.num_rows != descriptor.get("row_count")
        or not parquet.schema_arrow.equals(
            association_arrow_schema(),
            check_metadata=True,
        )
    ):
        raise X13BatchPublishError(
            f"association auxiliary artifact footer mismatch: {path.name}"
        )


def _verify_required_json_artifact(
    path: Path,
    descriptor: Mapping[str, object],
) -> None:
    role = str(descriptor.get("role"))
    expected = _REQUIRED_NON_TABLE_AUXILIARY.get(role)
    if expected is None:
        raise X13BatchPublishError(
            f"unexpected JSON auxiliary artifact role: {role}"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X13BatchPublishError(
            f"required auxiliary JSON is unreadable: {path.name}"
        ) from error
    if (
        type(value) is not dict
        or raw != _canonical_json_bytes(value) + b"\n"
        or value.get("schema") != expected[1]
    ):
        raise X13BatchPublishError(
            f"required auxiliary JSON schema is invalid: {path.name}"
        )


def publish_x13_batch(
    game_payloads: Sequence[Mapping[str, object]],
    *,
    output_root: str | Path,
    plan_id: str,
    builder_version: str,
    auxiliary_artifacts: Sequence[AuxiliaryArtifactV1] = (),
) -> Path:
    """Atomically publish one complete immutable batch or publish nothing."""

    manifest = build_batch_manifest(
        game_payloads,
        plan_id=plan_id,
        builder_version=builder_version,
        auxiliary_artifacts=auxiliary_artifacts,
    )
    normalized_auxiliary = _normalize_auxiliary_descriptors(
        auxiliary_artifacts
    )
    if any(
        not isinstance(artifact, AuxiliaryArtifactV1)
        for artifact in auxiliary_artifacts
    ):
        raise X13BatchPublishError(
            "publishing auxiliary artifacts requires source-bound "
            "AuxiliaryArtifactV1 values"
        )
    auxiliary_by_path = {
        artifact.relative_path: artifact for artifact in auxiliary_artifacts
    }
    if set(auxiliary_by_path) != {
        str(item["relative_path"]) for item in normalized_auxiliary
    }:
        raise X13BatchPublishError(
            "auxiliary artifact source bindings are incomplete"
        )
    by_id = {
        str(game["game_id"]): _validate_game_payload(game) for game in game_payloads
    }
    root = Path(output_root).resolve()
    if root == Path(root.anchor):
        raise X13BatchPublishError("output_root cannot be a filesystem root")
    root.mkdir(parents=True, exist_ok=True)
    batch_name = manifest["batch_id"].replace(":", "-")
    target = root / batch_name
    if target.exists():
        verified = verify_published_batch(target)
        existing = json.loads((target / "batch_manifest.json").read_text("utf-8"))
        if existing != manifest:
            raise X13BatchPublishError("immutable batch target conflicts with content")
        if verified["game_count"] != 20:
            raise X13BatchPublishError("existing immutable batch is incomplete")
        return target

    stage = Path(
        tempfile.mkdtemp(prefix=f".{batch_name}.", suffix=".tmp", dir=root)
    )
    try:
        games_dir = stage / "games"
        games_dir.mkdir()
        for game_id in X13_GAME_IDS:
            payload = by_id[game_id]
            _write_file(
                games_dir / f"{game_id}.json",
                _canonical_json_bytes(payload) + b"\n",
            )
            _write_file(
                games_dir / f"{game_id}.html",
                render_game_html(payload).encode("utf-8"),
            )
        _write_file(
            stage / "batch_manifest.json",
            _canonical_json_bytes(manifest) + b"\n",
        )
        _write_file(
            stage / "index.html",
            _render_index(manifest, by_id).encode("utf-8"),
        )
        for descriptor in normalized_auxiliary:
            relative_path = str(descriptor["relative_path"])
            _copy_auxiliary_file(
                auxiliary_by_path[relative_path],
                stage.joinpath(*PurePosixPath(relative_path).parts),
            )
        for descriptor in normalized_auxiliary:
            if descriptor["role"] == "association_candidate_table":
                relative_path = str(descriptor["relative_path"])
                _verify_association_parquet(
                    stage.joinpath(*PurePosixPath(relative_path).parts),
                    descriptor,
                )
            else:
                relative_path = str(descriptor["relative_path"])
                _verify_required_json_artifact(
                    stage.joinpath(*PurePosixPath(relative_path).parts),
                    descriptor,
                )
        checksum_files = sorted(
            path
            for path in stage.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        lines = []
        for path in checksum_files:
            relative = path.relative_to(stage).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {relative}")
        _write_file(
            stage / "checksums.sha256", ("\n".join(lines) + "\n").encode("ascii")
        )
        _fsync_directory(games_dir)
        for directory in sorted(
            {
                path.parent
                for path in stage.rglob("*")
                if path.is_file() and path.parent != stage
            },
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(stage)
        os.replace(stage, target)
        _fsync_directory(root)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    verify_published_batch(target)
    return target


def verify_published_batch(batch_path: str | Path) -> dict[str, Any]:
    """Fail closed if a published file, manifest, or batch shape was changed."""

    unresolved_root = Path(batch_path)
    if unresolved_root.is_symlink():
        raise X13BatchPublishError("published batch path cannot be a symlink")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise X13BatchPublishError("published batch path is not a regular directory")
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise X13BatchPublishError("published batch has no checksum ledger")
    entries: dict[str, str] = {}
    for line in checksum_path.read_text("ascii").splitlines():
        if not line:
            continue
        if len(line) < 67 or line[64:66] != "  ":
            raise X13BatchPublishError("checksum ledger is malformed")
        digest, relative_text = line[:64], line[66:]
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise X13BatchPublishError("checksum ledger digest is malformed")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise X13BatchPublishError("checksum ledger path is unsafe")
        if relative_text in entries:
            raise X13BatchPublishError("checksum ledger contains duplicate paths")
        entries[relative_text] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    core_files = {
        "batch_manifest.json",
        "index.html",
        *(
            f"games/{game_id}.{suffix}"
            for game_id in X13_GAME_IDS
            for suffix in ("html", "json")
        ),
    }
    if not core_files.issubset(actual):
        raise X13BatchPublishError(
            "published batch is missing a frozen core artifact"
        )
    if set(entries) != actual:
        raise X13BatchPublishError("checksum ledger file set does not match batch")
    manifest_path = root / "batch_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_value = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X13BatchPublishError("batch manifest is unreadable") from error
    manifest = dict(_require_mapping(manifest_value, "batch manifest"))
    if manifest_bytes != _canonical_json_bytes(manifest) + b"\n":
        raise X13BatchPublishError("batch manifest is not canonical JSON")
    if (
        manifest.get("schema") != "nfl_x13_batch_manifest_v1"
        or manifest.get("game_ids") != list(X13_GAME_IDS)
        or manifest.get("game_count") != 20
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("plan_id") != X13_BATCH_SPEC.plan_id
        or manifest.get("universe_id") != X13_FROZEN_UNIVERSE_ID
        or manifest.get("association_expected_row_count")
        != X13_FROZEN_ASSOCIATION_ROW_COUNT
    ):
        raise X13BatchPublishError("batch manifest does not prove the frozen batch")
    auxiliary_descriptors = _normalize_auxiliary_descriptors(
        _require_sequence(
            manifest.get("auxiliary_artifacts"),
            "batch manifest auxiliary_artifacts",
        )
    )
    auxiliary_paths = {
        str(descriptor["relative_path"])
        for descriptor in auxiliary_descriptors
    }
    expected_files = core_files | auxiliary_paths
    if actual != expected_files:
        raise X13BatchPublishError(
            "published batch does not have the exact manifest-bound "
            "artifact shape"
        )
    for relative, expected in entries.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise X13BatchPublishError("checksum target is not a regular file")
        digest = hashlib.sha256()
        byte_length = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
                byte_length += len(chunk)
        observed = digest.hexdigest()
        if observed != expected:
            raise X13BatchPublishError(f"checksum mismatch: {relative}")
        if relative in auxiliary_paths:
            descriptor = next(
                item
                for item in auxiliary_descriptors
                if item["relative_path"] == relative
            )
            if (
                descriptor["sha256"] != f"sha256:{observed}"
                or descriptor["byte_length"] != byte_length
            ):
                raise X13BatchPublishError(
                    f"auxiliary artifact identity mismatch: {relative}"
                )
            if descriptor["role"] == "association_candidate_table":
                _verify_association_parquet(path, descriptor)
            else:
                _verify_required_json_artifact(path, descriptor)
    builder_version = manifest.get("builder_version")
    if type(builder_version) is not str or not builder_version:
        raise X13BatchPublishError("batch manifest lacks builder version")

    games: dict[str, dict[str, Any]] = {}
    for game_id in X13_GAME_IDS:
        game_path = root / "games" / f"{game_id}.json"
        try:
            game_bytes = game_path.read_bytes()
            game_value = json.loads(game_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise X13BatchPublishError(
                f"{game_id} JSON artifact is unreadable"
            ) from error
        game = _validate_game_payload(game_value)
        if game["game_id"] != game_id:
            raise X13BatchPublishError(
                f"{game_id} JSON artifact has a foreign identity"
            )
        if game_bytes != _canonical_json_bytes(game) + b"\n":
            raise X13BatchPublishError(
                f"{game_id} JSON artifact is not canonical"
            )
        games[game_id] = game

    expected_manifest = build_batch_manifest(
        [games[game_id] for game_id in X13_GAME_IDS],
        plan_id=X13_BATCH_SPEC.plan_id,
        builder_version=builder_version,
        auxiliary_artifacts=auxiliary_descriptors,
    )
    if manifest.get("batch_id") != expected_manifest["batch_id"]:
        raise X13BatchPublishError(
            "batch_id does not match the canonical manifest identity"
        )
    if manifest != expected_manifest:
        raise X13BatchPublishError(
            "batch manifest does not match the published game identities"
        )
    if root.name != str(expected_manifest["batch_id"]).replace(":", "-"):
        raise X13BatchPublishError("batch directory is not content-addressed")

    for game_id in X13_GAME_IDS:
        html_path = root / "games" / f"{game_id}.html"
        if html_path.read_bytes() != render_game_html(games[game_id]).encode(
            "utf-8"
        ):
            raise X13BatchPublishError(
                f"{game_id} HTML does not match its canonical game JSON"
            )
    if (root / "index.html").read_bytes() != _render_index(
        expected_manifest, games
    ).encode("utf-8"):
        raise X13BatchPublishError(
            "batch index HTML does not match manifest and game JSON"
        )
    return {
        "batch_id": expected_manifest["batch_id"],
        "game_count": 20,
        "verified_file_count": len(entries) + 1,
    }


__all__ = [
    "AuxiliaryArtifactV1",
    "X13BatchPublishError",
    "build_batch_manifest",
    "canonical_payload_hash",
    "observation_payload_sha256",
    "publish_x13_batch",
    "render_game_html",
    "verify_published_batch",
]
