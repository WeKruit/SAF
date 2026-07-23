"""Versioned canonical contracts for the prediction-market research program.

The models in this module deliberately fail closed.  They are the executable
counterpart of the v0 YAML contracts under ``contracts/`` and contain no venue
or engine defaults.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import PurePosixPath
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)


_CANONICAL_ATOMS_RE = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_PLAIN_DECIMAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z"
)


MARKET_RELATIONS = frozenset(
    {
        "identity",
        "subset",
        "superset",
        "overlap",
        "mutex",
        "exhaustive",
        "incompatible",
    }
)

QUALITY_FLAGS = frozenset(
    {
        "clock_jump",
        "crossed_book",
        "duplicate_event",
        "gap_detected",
        "missing_initial_snapshot",
        "missing_side",
        "negative_time_delta",
        "non_positive_size",
        "out_of_order",
        "pause_observed",
        "preliminary_rules",
        "reconstruction_divergence",
        "source_clock_unverified",
        "stale_as_of_join",
        "tick_size_change",
    }
)

REGISTERED_EXPERIMENT_IDS = frozenset(
    f"X-{number:02d}" for number in range(1, 13)
)

EVENT_TYPES = frozenset(
    {
        "raw_observation",
        "normalized_observation",
        "model_output",
        "label",
        "signal",
        "simulated_order",
        "simulated_fill",
        "simulated_pnl",
    }
)

DERIVED_EVENT_TYPES = EVENT_TYPES - {"raw_observation"}
SIMULATED_EVENT_TYPES = frozenset(
    {"simulated_order", "simulated_fill", "simulated_pnl"}
)

LEVEL2_STREAM_DOMAIN_TAG = b"prediction-market:event-stream:v0\x00"

START_TIME_CANCEL_POLICIES = frozenset(
    {
        "cancel_all_at_game_start",
        "cancel_all_with_schedule_change_exception",
        "preserve_orders_at_game_start",
    }
)

ORDER_TYPES = frozenset(
    {"DAY", "FAK", "FOK", "GTC", "GTD", "IOC", "LIMIT", "MARKET", "POST_ONLY"}
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ContractValidationError(ValueError):
    """Raised when an in-memory contract graph violates boundary invariants."""


_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class FrozenDict(Mapping[_Key, _Value], Generic[_Key, _Value]):
    """A mapping backed only by an immutable tuple of key/value pairs."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[_Key, _Value]) -> None:
        object.__setattr__(self, "_items", tuple(values.items()))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def __getitem__(self, key: _Key) -> _Value:
        for stored_key, stored_value in self._items:
            if stored_key == key:
                return stored_value
        raise KeyError(key)

    def __iter__(self) -> Iterator[_Key]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        contents = ", ".join(
            f"{key!r}: {value!r}" for key, value in self._items
        )
        return f"FrozenDict({{{contents}}})"


def _freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return FrozenDict(
            {
                field_name: _freeze(getattr(value, field_name))
                for field_name in type(value).model_fields
            }
        )
    if isinstance(value, Mapping):
        return FrozenDict(
            {key: _freeze(child_value) for key, child_value in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child_value) for child_value in value)
    return value


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, (list, tuple)) else value


def thaw_contract_v0(value: Any) -> Any:
    """Return a detached JSON-compatible representation of immutable values."""

    if isinstance(value, BaseModel):
        return {
            field_name: thaw_contract_v0(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {
            key: thaw_contract_v0(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [thaw_contract_v0(child_value) for child_value in value]
    return value


def _audit_contract_object_graph(
    value: Any,
    *,
    path: tuple[str, ...] = ("contract",),
    seen: set[int] | None = None,
) -> None:
    if not isinstance(value, (BaseModel, Mapping, list, tuple)):
        return

    visited = seen if seen is not None else set()
    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)

    if isinstance(value, BaseModel):
        declared_fields = frozenset(type(value).model_fields)
        actual_fields = frozenset(value.__dict__)

        fields_set = value.__pydantic_fields_set__
        if not isinstance(fields_set, (set, frozenset)):
            raise ContractValidationError(
                f"invalid __pydantic_fields_set__ at {'.'.join(path)}"
            )

        extra = value.__pydantic_extra__
        if extra is not None:
            raise ContractValidationError(
                f"forbidden __pydantic_extra__ at {'.'.join(path)}"
            )

        unknown_fields = (
            actual_fields
            | frozenset(fields_set)
        ) - declared_fields
        if unknown_fields:
            names = ", ".join(sorted(repr(field) for field in unknown_fields))
            raise ContractValidationError(
                f"unknown contract field at {'.'.join(path)}: {names}"
            )

        for field_name in declared_fields & actual_fields:
            _audit_contract_object_graph(
                value.__dict__[field_name],
                path=path + (field_name,),
                seen=visited,
            )
        return

    if isinstance(value, Mapping):
        for key, child_value in value.items():
            _audit_contract_object_graph(
                key,
                path=path + ("<key>",),
                seen=visited,
            )
            _audit_contract_object_graph(
                child_value,
                path=path + (f"[{key!r}]",),
                seen=visited,
            )
        return

    for index, child_value in enumerate(value):
        _audit_contract_object_graph(
            child_value,
            path=path + (f"[{index}]",),
            seen=visited,
        )


def _untrusted_round_trip_input(value: Any) -> Any:
    _audit_contract_object_graph(value)
    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="python",
            round_trip=True,
            serialize_as_any=True,
            warnings="error",
        )
    return value


class FixedPointV0(_ContractModel):
    """An exact decimal represented as a canonical integer and base-10 scale."""

    atoms: str
    scale: int = Field(ge=0, le=18)

    @field_validator("atoms")
    @classmethod
    def _canonical_atoms(cls, value: str) -> str:
        if not isinstance(value, str) or _CANONICAL_ATOMS_RE.fullmatch(value) is None:
            raise ValueError("atoms must be a canonical signed base-10 integer")
        return value

    @classmethod
    def from_value(cls, value: int | str | Decimal) -> "FixedPointV0":
        """Create an exact fixed-point value without accepting binary floats.

        Plain decimal strings preserve their declared number of fractional
        places.  Scientific/exponent notation is intentionally outside v0.
        """

        if isinstance(value, float):
            raise ValueError("binary float is forbidden in canonical contracts")
        if isinstance(value, bool):
            raise ValueError("boolean is not a fixed-point value")
        if isinstance(value, int):
            return cls(atoms=str(value), scale=0)
        if isinstance(value, str):
            if "e" in value.lower():
                raise ValueError("exponent notation is forbidden")
            if _PLAIN_DECIMAL_RE.fullmatch(value) is None:
                raise ValueError("value must be a plain canonical decimal string")
            negative = value.startswith("-")
            unsigned = value[1:] if negative else value
            whole, separator, fraction = unsigned.partition(".")
            scale = len(fraction) if separator else 0
            digits = (whole + fraction).lstrip("0") or "0"
            atoms = f"-{digits}" if negative and digits != "0" else digits
            return cls(atoms=atoms, scale=scale)
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError("fixed-point Decimal must be finite")
            textual = str(value)
            exponent = value.as_tuple().exponent
            if "e" in textual.lower() or not isinstance(exponent, int) or exponent > 0:
                raise ValueError("exponent notation is forbidden")
            digits = "".join(str(digit) for digit in value.as_tuple().digits) or "0"
            digits = digits.lstrip("0") or "0"
            atoms = f"-{digits}" if value.is_signed() and digits != "0" else digits
            return cls(atoms=atoms, scale=-exponent)
        raise TypeError("FixedPointV0 accepts only int, str, or Decimal")

    def to_decimal(self) -> Decimal:
        """Return the represented value without a context-sensitive operation."""

        negative = self.atoms.startswith("-")
        unsigned = self.atoms[1:] if negative else self.atoms
        digits = tuple(int(character) for character in unsigned)
        return Decimal((1 if negative else 0, digits, -self.scale))


def _validate_utc_timestamp(value: str) -> str:
    if _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError("timestamp must be RFC3339 UTC with a terminal Z")
    try:
        datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError("timestamp is not a valid RFC3339 instant") from error
    return value


def _validate_non_blank(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("value must be non-empty and have no surrounding whitespace")
    return value


Sha256V0 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$"),
]
EventIdV0 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^evt_[0-9a-f]{64}$"),
]
UtcTimestampV0 = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_validate_utc_timestamp),
]
NonBlankStr = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_validate_non_blank),
]
VenueSlugV0 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-z0-9][a-z0-9._-]*$"),
]
CompetitionIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^cmp_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
GameIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^game_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
ParticipantIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^participant_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
VenueEventIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^venue_event_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
MarketIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^market_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
OutcomeIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^outcome_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
ConditionIdV0 = Annotated[
    str, StringConstraints(strict=True, pattern=r"^condition_[A-Za-z0-9][A-Za-z0-9._:-]*$")
]

PayloadHashV0 = Sha256V0
RawObjectHashV0 = Sha256V0
RuleSnapshotRefV0 = Sha256V0

ExperimentIdV0 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^X-[0-9]{2,}$"),
]
ExperimentIdV1 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^X-[0-9]{2,}$"),
]
EventTypeV0 = Literal[
    "raw_observation",
    "normalized_observation",
    "model_output",
    "label",
    "signal",
    "simulated_order",
    "simulated_fill",
    "simulated_pnl",
]
QualityFlagV0 = Literal[
    "clock_jump",
    "crossed_book",
    "duplicate_event",
    "gap_detected",
    "missing_initial_snapshot",
    "missing_side",
    "negative_time_delta",
    "non_positive_size",
    "out_of_order",
    "pause_observed",
    "preliminary_rules",
    "reconstruction_divergence",
    "source_clock_unverified",
    "stale_as_of_join",
    "tick_size_change",
]
SportV0 = Literal["nba", "nfl", "soccer", "mlb", "f1"]
GameStateObservationModeV0 = Literal[
    "live_pit",
    "offline_reconstruction",
    "synthetic_fixture",
]
MarketRelationV0 = Literal[
    "identity",
    "subset",
    "superset",
    "overlap",
    "mutex",
    "exhaustive",
    "incompatible",
]
StartTimeCancelPolicyV0 = Literal[
    "cancel_all_at_game_start",
    "cancel_all_with_schedule_change_exception",
    "preserve_orders_at_game_start",
]
OrderTypeV0 = Literal[
    "DAY", "FAK", "FOK", "GTC", "GTD", "IOC", "LIMIT", "MARKET", "POST_ONLY"
]
FeeFormulaV0 = Literal["C_X_RATE_X_P_ONE_MINUS_P_POW_EXPONENT"]


class EventSourceV0(_ContractModel):
    system: NonBlankStr
    stream: NonBlankStr
    venue: VenueSlugV0 | None = None
    sequence: int | NonBlankStr | None = None
    capture_session_id: NonBlankStr | None = None
    record_ordinal: int | None = Field(default=None, ge=0)

    @field_validator("sequence")
    @classmethod
    def _sequence_is_nonnegative(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, int) and value < 0:
            raise ValueError("sequence must be non-negative")
        return value


class EventTimeV0(_ContractModel):
    receive_at: UtcTimestampV0
    receive_basis: Literal["local_recorder", "upstream_exporter"]
    source_at: UtcTimestampV0 | None = None
    publish_at: UtcTimestampV0 | None = None
    exchange_at: UtcTimestampV0 | None = None


class CanonicalReferencesV0(_ContractModel):
    competition_id: CompetitionIdV0 | None
    game_id: GameIdV0 | None
    participant_ids: tuple[ParticipantIdV0, ...]
    venue_event_id: VenueEventIdV0 | None
    market_id: MarketIdV0 | None
    outcome_id: OutcomeIdV0 | None
    condition_id: ConditionIdV0 | None

    @field_validator("participant_ids", mode="before")
    @classmethod
    def _participant_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("participant_ids")
    @classmethod
    def _participants_are_a_set(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("participant_ids must be unique")
        return tuple(sorted(value))


class NativeReferenceV0(_ContractModel):
    namespace: NonBlankStr
    native_id: NonBlankStr


class EventLineageV0(_ContractModel):
    raw_object_hash: Sha256V0 | None = None
    raw_record_ordinal: int | None = Field(default=None, ge=0)
    parent_event_ids: tuple[EventIdV0, ...] = ()

    @field_validator("parent_event_ids", mode="before")
    @classmethod
    def _parent_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("parent_event_ids")
    @classmethod
    def _parents_are_a_set(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("parent_event_ids must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _exactly_one_lineage_branch(self) -> "EventLineageV0":
        has_raw_hash = self.raw_object_hash is not None
        has_raw_ordinal = self.raw_record_ordinal is not None
        if has_raw_hash != has_raw_ordinal:
            raise ValueError("raw lineage requires both raw_object_hash and raw_record_ordinal")
        has_raw = has_raw_hash and has_raw_ordinal
        has_parents = bool(self.parent_event_ids)
        if has_raw == has_parents:
            raise ValueError("lineage requires raw object hash+ordinal OR parent event IDs")
        return self


# Only these complete structural paths have set semantics. Payload paths never
# match them, even when a payload property reuses a structural field name.
_SET_SEMANTIC_PATHS = frozenset(
    {
        ("event_envelope", "canonical_refs", "participant_ids"),
        ("event_envelope", "native_refs"),
        ("event_envelope", "lineage", "parent_event_ids"),
        ("event_envelope", "quality_flags"),
        ("model_output", "state_space"),
        ("model_output", "quality_flags"),
        ("static_dataset_manifest", "lineage", "source_object_refs"),
        ("market_metadata_snapshot", "participants"),
        ("market_metadata_snapshot", "canonical_refs", "participant_ids"),
        ("market_metadata_snapshot", "quality_flags"),
        ("venue_rule_snapshot", "order_types_supported"),
    }
)


def _canonical_root(value: Any) -> tuple[str, ...]:
    model_name = type(value).__name__ if isinstance(value, BaseModel) else ""
    if model_name == "EventEnvelopeV0":
        return ("event_envelope",)
    if model_name == "ModelOutputV1":
        return ("model_output",)
    if model_name == "StaticDatasetManifestV0":
        return ("static_dataset_manifest",)
    if model_name == "MarketMetadataSnapshotV0":
        return ("market_metadata_snapshot",)
    if model_name == "VenueRuleSnapshotV0":
        return ("venue_rule_snapshot",)
    if isinstance(value, Mapping):
        if value.get("envelope_version") == "v0" and "event_type" in value:
            return ("event_envelope",)
        if value.get("contract_version") == "v1" and "probabilities" in value:
            return ("model_output",)
        if value.get("manifest_version") == "v0" and "object_kind" in value:
            return ("static_dataset_manifest",)
        if value.get("snapshot_version") == "v0" and "native_token_id" in value:
            return ("market_metadata_snapshot",)
        if "condition_id" in value and "raw_response_hash" in value:
            return ("venue_rule_snapshot",)
    return ("value",)


def _canonical_value(value: Any, *, path: tuple[str, ...]) -> Any:
    if isinstance(value, BaseModel):
        value = {
            field_name: getattr(value, field_name)
            for field_name in type(value).model_fields
        }
    if isinstance(value, float):
        raise ValueError("binary float is forbidden in canonical contracts")
    if isinstance(value, Decimal):
        raise ValueError("Decimal must be encoded as FixedPointV0")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise ValueError("canonical object keys must be strings")
            normalized[child_key] = _canonical_value(
                child_value, path=path + (child_key,)
            )
        return normalized
    if isinstance(value, (list, tuple)):
        normalized_list = [
            _canonical_value(item, path=path + ("[]",)) for item in value
        ]
        if path in _SET_SEMANTIC_PATHS:
            normalized_list.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return normalized_list
    raise ValueError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for a canonical contract value."""

    normalized = _canonical_value(value, path=_canonical_root(value))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = _canonical_value(payload, path=("payload",))
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _event_hash_material(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        material = value.model_dump(mode="python")
    else:
        material = dict(value)
        material.setdefault("experiment_id", None)
        material.setdefault("rule_snapshot_ref", None)

        nested_defaults: dict[str, dict[str, Any]] = {
            "source": {
                "venue": None,
                "sequence": None,
                "capture_session_id": None,
                "record_ordinal": None,
            },
            "time": {
                "source_at": None,
                "publish_at": None,
                "exchange_at": None,
            },
            "lineage": {
                "raw_object_hash": None,
                "raw_record_ordinal": None,
                "parent_event_ids": [],
            },
        }
        for object_key, defaults in nested_defaults.items():
            child = material.get(object_key)
            if isinstance(child, Mapping):
                expanded_child = dict(child)
                for child_key, default in defaults.items():
                    expanded_child.setdefault(child_key, default)
                material[object_key] = expanded_child
    material.pop("event_id", None)
    return material


def event_id_for(value: Mapping[str, Any] | BaseModel) -> str:
    digest = hashlib.sha256(canonical_json_bytes(_event_hash_material(value))).hexdigest()
    return f"evt_{digest}"


# Explicit aliases used by consumers that prefer verb-first helper names.
compute_payload_sha256 = payload_sha256
compute_event_id = event_id_for


def _timestamp_order_value(value: str) -> tuple[int, int, int, int, int, int, Decimal]:
    _validate_utc_timestamp(value)
    date_part, time_part = value[:-1].split("T", maxsplit=1)
    year, month, day = (int(part) for part in date_part.split("-"))
    hour_text, minute_text, second_and_fraction = time_part.split(":")
    second_text, separator, fraction_text = second_and_fraction.partition(".")
    fraction = Decimal(f"0.{fraction_text}") if separator else Decimal(0)
    return (
        year,
        month,
        day,
        int(hour_text),
        int(minute_text),
        int(second_text),
        fraction,
    )


def _nulls_last(
    value: Any, transform: Callable[[Any], Any]
) -> tuple[int, Any]:
    return (1, "") if value is None else (0, transform(value))


def _require_envelope(value: Any) -> "EventEnvelopeV0":
    if not isinstance(value, EventEnvelopeV0):
        raise TypeError("ordering and framing require a validated EventEnvelopeV0")
    return EventEnvelopeV0.model_validate(_untrusted_round_trip_input(value))


def replay_order_key(value: "EventEnvelopeV0") -> tuple[Any, ...]:
    """Return the collision-free v0 total-order key for a validated envelope."""

    envelope = _require_envelope(value)
    return (
        _timestamp_order_value(envelope.time.receive_at),
        _nulls_last(envelope.time.source_at, _timestamp_order_value),
        _nulls_last(envelope.canonical_refs.market_id, str),
        _nulls_last(envelope.canonical_refs.outcome_id, str),
        envelope.payload_sha256,
        envelope.event_id,
    )


def level2_stream_frame(
    events: Iterable["EventEnvelopeV0"],
) -> bytes:
    """Sort validated envelopes and frame their digests without ambiguity."""

    envelopes = tuple(_require_envelope(event) for event in events)
    ordered = tuple(sorted(envelopes, key=replay_order_key))
    count = len(ordered)
    return (
        LEVEL2_STREAM_DOMAIN_TAG
        + count.to_bytes(8, byteorder="big", signed=False)
        + b"".join(
            bytes.fromhex(envelope.event_id.removeprefix("evt_"))
            for envelope in ordered
        )
    )


def level2_stream_sha256(
    events: Iterable["EventEnvelopeV0"],
) -> str:
    return f"sha256:{hashlib.sha256(level2_stream_frame(events)).hexdigest()}"


class EventEnvelopeV0(_ContractModel):
    envelope_version: Literal["v0"]
    event_id: EventIdV0
    event_type: EventTypeV0
    payload_schema_version: Literal["v0", "v1"]
    source: EventSourceV0
    time: EventTimeV0
    canonical_refs: CanonicalReferencesV0
    native_refs: tuple[NativeReferenceV0, ...]
    lineage: EventLineageV0
    experiment_id: ExperimentIdV0 | None = None
    rule_snapshot_ref: Sha256V0 | None = None
    quality_flags: tuple[QualityFlagV0, ...]
    payload: Any
    payload_sha256: Sha256V0

    @field_validator("native_refs", mode="before")
    @classmethod
    def _native_ref_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("native_refs")
    @classmethod
    def _native_refs_are_a_set(
        cls, value: tuple[NativeReferenceV0, ...]
    ) -> tuple[NativeReferenceV0, ...]:
        keys = [(reference.namespace, reference.native_id) for reference in value]
        if len(keys) != len(set(keys)):
            raise ValueError("native_refs must be unique")
        return tuple(
            sorted(value, key=lambda reference: (reference.namespace, reference.native_id))
        )

    @field_validator("quality_flags", mode="before")
    @classmethod
    def _quality_flag_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("quality_flags")
    @classmethod
    def _quality_flags_are_a_set(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("quality_flags must be unique")
        return tuple(sorted(value))

    @field_validator("payload", mode="before")
    @classmethod
    def _payload_is_canonical_json(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("payload must be an object")
        _canonical_value(value, path=("payload",))
        return _freeze(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Any) -> dict[str, Any]:
        return thaw_contract_v0(value)

    @model_validator(mode="after")
    def _validate_hash_and_conditional_contract(self) -> "EventEnvelopeV0":
        required_payload_version = "v1" if self.event_type == "model_output" else "v0"
        if self.payload_schema_version != required_payload_version:
            raise ValueError(
                f"{self.event_type} requires payload_schema_version="
                f"{required_payload_version}"
            )
        expected_payload_hash = payload_sha256(self.payload)
        if self.payload_sha256 != expected_payload_hash:
            raise ValueError(
                f"payload_sha256 mismatch: expected {expected_payload_hash}"
            )

        if self.event_type == "raw_observation":
            if self.source.capture_session_id is None or self.source.record_ordinal is None:
                raise ValueError(
                    "raw observation requires capture_session_id and record_ordinal"
                )
            if not self.native_refs:
                raise ValueError("raw observation requires at least one native reference")
            if self.lineage.raw_object_hash is None:
                raise ValueError("raw observation requires raw lineage")
            if self.lineage.raw_record_ordinal != self.source.record_ordinal:
                raise ValueError("raw lineage ordinal must match source record_ordinal")
        else:
            if not self.lineage.parent_event_ids:
                raise ValueError("derived event requires parent event IDs")
            if self.experiment_id is None:
                raise ValueError("derived event requires a registered experiment ID")
        if self.event_type in SIMULATED_EVENT_TYPES and self.rule_snapshot_ref is None:
            raise ValueError("simulated event requires rule_snapshot_ref")

        expected_event_id = event_id_for(self)
        if self.event_id != expected_event_id:
            raise ValueError(f"event_id mismatch: expected {expected_event_id}")
        return self

    @classmethod
    def create(cls, **values: Any) -> "EventEnvelopeV0":
        """Build an envelope while deriving both content hashes deterministically."""

        material = dict(values)
        payload = material.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be an object")
        material["payload_sha256"] = payload_sha256(payload)
        material["event_id"] = event_id_for(material)
        return cls.model_validate(material)


class StaticDatasetLineageV0(_ContractModel):
    source_object_refs: tuple[Sha256V0, ...]
    query_sha256: Sha256V0 | None

    @field_validator("source_object_refs", mode="before")
    @classmethod
    def _source_object_refs_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("source_object_refs")
    @classmethod
    def _source_object_refs_are_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("source_object_refs must be unique")
        return tuple(sorted(value))


def _content_hash_excluding(value: Any, hash_field: str) -> str:
    if isinstance(value, BaseModel):
        material = thaw_contract_v0(value)
    elif isinstance(value, Mapping):
        material = dict(value)
    else:
        raise TypeError("content hash input must be a mapping or contract model")
    material.pop(hash_field, None)
    return canonical_sha256(material)


def static_dataset_manifest_sha256(value: Any) -> str:
    """Hash every manifest field except ``manifest_sha256`` itself."""

    return _content_hash_excluding(value, "manifest_sha256")


class StaticDatasetManifestV0(_ContractModel):
    """Immutable publication record for one static source object or extract."""

    manifest_version: Literal["v0"]
    dataset_id: Annotated[
        str,
        StringConstraints(strict=True, pattern=r"^DS-[A-Z0-9][A-Z0-9-]*$"),
    ]
    object_kind: Literal["byte_exact_original", "source_derived_extract"]
    source_url: NonBlankStr
    source_request: Mapping[str, Any]
    source_cursor: NonBlankStr | None
    fetched_at: UtcTimestampV0
    coverage: NonBlankStr
    etag: NonBlankStr | None
    last_modified: NonBlankStr | None
    byte_length: int = Field(ge=0)
    object_sha256: Sha256V0
    native_object_path: NonBlankStr
    media_type: NonBlankStr
    schema_fingerprint: Sha256V0
    license_ref: Annotated[
        str,
        StringConstraints(strict=True, pattern=r"^[RIO]-[0-9]{3}$"),
    ]
    license_status: Literal[
        "approved", "research_only", "pending", "unknown", "blocked"
    ]
    upstream_partition: NonBlankStr
    lineage: StaticDatasetLineageV0
    manifest_sha256: Sha256V0

    @field_validator("source_url")
    @classmethod
    def _source_url_is_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("source_url must use HTTPS")
        return value

    @field_validator("source_request", mode="before")
    @classmethod
    def _source_request_is_canonical(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("source_request must be an object")
        _canonical_value(value, path=("static_dataset_manifest", "source_request"))
        return _freeze(value)

    @field_serializer("source_request")
    def _serialize_source_request(self, value: Any) -> dict[str, Any]:
        return thaw_contract_v0(value)

    @field_validator("native_object_path")
    @classmethod
    def _native_object_path_is_canonical(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("native_object_path must be a canonical relative POSIX path")
        return value

    @model_validator(mode="after")
    def _lineage_and_hash_are_consistent(self) -> "StaticDatasetManifestV0":
        refs = self.lineage.source_object_refs
        query_hash = self.lineage.query_sha256
        if self.object_kind == "source_derived_extract":
            if not refs or query_hash is None:
                raise ValueError(
                    "source_derived_extract requires source_object_refs and query_sha256"
                )
        elif refs or query_hash is not None:
            raise ValueError(
                "byte_exact_original cannot claim source-derived lineage"
            )
        expected = static_dataset_manifest_sha256(self)
        if self.manifest_sha256 != expected:
            raise ValueError(f"manifest_sha256 mismatch: expected {expected}")
        return self


def validate_static_dataset_manifest_v0(
    program_root: str | Path,
    instance: Any,
) -> StaticDatasetManifestV0:
    """Validate a static manifest and its dataset/license registry bindings."""

    validated = StaticDatasetManifestV0.model_validate(
        _untrusted_round_trip_input(instance)
    )
    from prediction_market.program_audit import (
        ResearchRegistryError,
        load_dataset_registry,
    )

    try:
        datasets = {
            row.dataset_id: row for row in load_dataset_registry(program_root)
        }
    except ResearchRegistryError as exc:
        raise ContractValidationError(
            f"dataset registry is invalid: {exc}"
        ) from exc
    dataset = datasets.get(validated.dataset_id)
    if dataset is None:
        raise ContractValidationError(
            f"dataset {validated.dataset_id} is not registered"
        )
    if validated.license_ref != dataset.license_review_id:
        raise ContractValidationError(
            f"license_ref {validated.license_ref} does not match dataset "
            f"{validated.dataset_id} license review {dataset.license_review_id}"
        )
    if validated.license_status != dataset.license_status:
        raise ContractValidationError(
            f"license_status {validated.license_status} does not match "
            f"dataset {validated.dataset_id}"
        )
    return validated


def market_metadata_snapshot_sha256(value: Any) -> str:
    """Hash every metadata snapshot field except ``snapshot_sha256`` itself."""

    return _content_hash_excluding(value, "snapshot_sha256")


class MarketMetadataSnapshotV0(_ContractModel):
    """Point-in-time venue metadata for one outcome/token mapping."""

    snapshot_version: Literal["v0"]
    venue: VenueSlugV0
    native_event_id: NonBlankStr
    native_market_id: NonBlankStr
    native_condition_id: NonBlankStr
    native_outcome_id: NonBlankStr
    native_token_id: NonBlankStr
    canonical_refs: CanonicalReferencesV0
    sport: NonBlankStr
    competition: NonBlankStr
    participants: tuple[NonBlankStr, ...]
    game_start_at: UtcTimestampV0
    rules: NonBlankStr
    resolution: NonBlankStr | None
    closed: bool
    resolved: bool
    captured_at: UtcTimestampV0
    source_updated_at: UtcTimestampV0
    raw_object_hash: Sha256V0
    quality_flags: tuple[QualityFlagV0, ...]
    snapshot_sha256: Sha256V0

    @field_validator("participants", "quality_flags", mode="before")
    @classmethod
    def _set_inputs_are_tuples(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("participants")
    @classmethod
    def _participants_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("participants must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("participants must be unique")
        return tuple(sorted(value))

    @field_validator("quality_flags")
    @classmethod
    def _metadata_quality_flags_are_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("quality_flags must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _point_in_time_and_hash_are_consistent(
        self,
    ) -> "MarketMetadataSnapshotV0":
        if (
            self.canonical_refs.competition_id is None
            or self.canonical_refs.game_id is None
            or not self.canonical_refs.participant_ids
            or self.canonical_refs.venue_event_id is None
            or self.canonical_refs.market_id is None
            or self.canonical_refs.outcome_id is None
            or self.canonical_refs.condition_id is None
        ):
            raise ValueError(
                "market metadata snapshot requires all canonical join mappings"
            )
        if _timestamp_order_value(self.source_updated_at) > _timestamp_order_value(
            self.captured_at
        ):
            raise ValueError("source_updated_at cannot be after captured_at")
        if self.resolved and (not self.closed or self.resolution is None):
            raise ValueError("resolved metadata requires closed=true and a resolution")
        if not self.resolved and self.resolution is not None:
            raise ValueError("unresolved metadata cannot declare a resolution")
        expected = market_metadata_snapshot_sha256(self)
        if self.snapshot_sha256 != expected:
            raise ValueError(f"snapshot_sha256 mismatch: expected {expected}")
        return self


TransitionUnitV1 = Literal["possession", "drive", "five_minute_interval"]


class ModelOutputV1(_ContractModel):
    """A point-in-time state-transition distribution."""

    contract_version: Literal["v1"]
    model_id: NonBlankStr
    model_version: NonBlankStr
    experiment_id: ExperimentIdV1
    run_id: NonBlankStr
    game_id: GameIdV0
    state_event_id: EventIdV0
    pit_cutoff_at: UtcTimestampV0
    output_kind: Literal["state_transition"]
    transition_unit: TransitionUnitV1
    state_space: tuple[NonBlankStr, ...]
    horizon: Literal["next_state_transition"]
    probabilities: Mapping[str, FixedPointV0]
    feature_sha256: Sha256V0
    data_sha256: Sha256V0
    config_sha256: Sha256V0
    quality_flags: tuple[QualityFlagV0, ...]

    @field_validator("state_space", mode="before")
    @classmethod
    def _state_space_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("state_space")
    @classmethod
    def _state_space_is_a_set(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("state_space must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("state_space must be unique")
        return tuple(sorted(value))

    @field_validator("probabilities")
    @classmethod
    def _probabilities_are_immutable(
        cls, value: Mapping[str, FixedPointV0]
    ) -> FrozenDict[str, FixedPointV0]:
        return FrozenDict(value)

    @field_serializer("probabilities")
    def _serialize_probabilities(
        self, value: Mapping[str, FixedPointV0]
    ) -> dict[str, Any]:
        return thaw_contract_v0(value)

    @field_validator("quality_flags", mode="before")
    @classmethod
    def _model_quality_flag_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("quality_flags")
    @classmethod
    def _model_quality_flags_are_a_set(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("quality_flags must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _probabilities_form_distribution(self) -> "ModelOutputV1":
        if not self.probabilities:
            raise ValueError("probabilities must not be empty")
        if set(self.probabilities) != set(self.state_space):
            raise ValueError("probability keys must exactly match state_space")
        values = [probability.to_decimal() for probability in self.probabilities.values()]
        if any(probability < 0 or probability > 1 for probability in values):
            raise ValueError("each probability must be in [0, 1]")
        if sum(values, start=Decimal(0)) != Decimal(1):
            raise ValueError("probabilities must sum exactly to 1")
        return self


class GameStateStepV0(_ContractModel):
    """Content-addressed evidence for one sport-specific state transition."""

    step_version: Literal["v0"]
    sport: SportV0
    game_id: GameIdV0
    sequence: Annotated[int, Field(strict=True, ge=0)]
    terminal: bool
    reducer_id: NonBlankStr
    reducer_version: NonBlankStr
    state_schema_id: Annotated[
        str,
        StringConstraints(
            strict=True,
            pattern=r"^urn:saf:game-state:(?:nba|nfl|soccer|mlb|f1):v[0-9]+$",
        ),
    ]
    event_schema_id: Annotated[
        str,
        StringConstraints(
            strict=True,
            pattern=r"^urn:saf:game-event:(?:nba|nfl|soccer|mlb|f1):v[0-9]+$",
        ),
    ]
    event_id: EventIdV0
    previous_state_sha256: Sha256V0
    event_sha256: Sha256V0
    next_state_sha256: Sha256V0
    observation_mode: GameStateObservationModeV0
    quality_flags: tuple[QualityFlagV0, ...]
    step_sha256: Sha256V0

    @field_validator("quality_flags", mode="before")
    @classmethod
    def _quality_flags_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("quality_flags")
    @classmethod
    def _quality_flags_are_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("quality_flags must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _step_hash_is_content_addressed(self) -> "GameStateStepV0":
        expected = game_state_step_sha256(self)
        if self.step_sha256 != expected:
            raise ValueError(f"step_sha256 mismatch: expected {expected}")
        return self


def game_state_step_sha256(
    value: Mapping[str, Any] | GameStateStepV0,
) -> str:
    """Hash a game-state step without its self-referential digest."""

    if isinstance(value, GameStateStepV0):
        material = value.model_dump(mode="python")
    elif isinstance(value, Mapping):
        material = dict(value)
    else:
        raise TypeError("game-state step must be a mapping or GameStateStepV0")
    material.pop("step_sha256", None)
    return canonical_sha256(material)


class VenueRuleSnapshotV0(_ContractModel):
    """One append-only, per-condition observation of venue execution rules."""

    venue: VenueSlugV0
    condition_id: ConditionIdV0
    fetched_at: UtcTimestampV0
    effective_from: UtcTimestampV0
    game_start_time: UtcTimestampV0
    seconds_delay: FixedPointV0
    cancel_during_delay: bool
    start_time_cancel_policy: StartTimeCancelPolicyV0
    fees_enabled: bool
    fee_rate: FixedPointV0
    fee_exponent: FixedPointV0
    taker_only: bool
    maker_fee_rate: FixedPointV0
    minimum_tick_size: FixedPointV0
    minimum_order_size: FixedPointV0
    order_types_supported: tuple[OrderTypeV0, ...]
    source_document_version: NonBlankStr
    raw_response_hash: Sha256V0

    @field_validator("venue")
    @classmethod
    def _venue_is_observed(cls, value: str) -> str:
        if value == "unknown":
            raise ValueError("unknown venue is not an observed rule")
        return value

    @field_validator("source_document_version")
    @classmethod
    def _source_document_is_observed(cls, value: str) -> str:
        if value.lower() == "unknown":
            raise ValueError("source_document_version must identify the observation")
        return value

    @field_validator("order_types_supported", mode="before")
    @classmethod
    def _order_type_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("order_types_supported")
    @classmethod
    def _order_types_are_a_nonempty_set(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("order_types_supported must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("order_types_supported must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _execution_numbers_are_observed_and_valid(self) -> "VenueRuleSnapshotV0":
        nonnegative = {
            "seconds_delay": self.seconds_delay,
            "fee_rate": self.fee_rate,
            "fee_exponent": self.fee_exponent,
            "maker_fee_rate": self.maker_fee_rate,
        }
        for field_name, fixed_point in nonnegative.items():
            if fixed_point.to_decimal() < 0:
                raise ValueError(f"{field_name} must be non-negative")
        positive = {
            "minimum_tick_size": self.minimum_tick_size,
            "minimum_order_size": self.minimum_order_size,
        }
        for field_name, fixed_point in positive.items():
            if fixed_point.to_decimal() <= 0:
                raise ValueError(f"{field_name} must be positive")
        return self


class TcaMarkoutV0(_ContractModel):
    """One executable-bid X-07 markout with point-in-time exit costs."""

    horizon_microseconds: int = Field(gt=0)
    local_receive_at: UtcTimestampV0
    executable_bid_vwap: FixedPointV0
    gross_markout_per_unit: FixedPointV0
    exit_fee: FixedPointV0
    net_markout_per_unit: FixedPointV0
    exit_levels_consumed: int = Field(gt=0)
    rule_snapshot_ref: Sha256V0
    book_snapshot_ref: Sha256V0

    @model_validator(mode="after")
    def _markout_values_are_executable(self) -> "TcaMarkoutV0":
        bid = self.executable_bid_vwap.to_decimal()
        if not Decimal(0) <= bid <= Decimal(1):
            raise ValueError("executable_bid_vwap must be in [0, 1]")
        if self.exit_fee.to_decimal() < 0:
            raise ValueError("exit_fee must be non-negative")
        return self


class TcaRecordV0(_ContractModel):
    """Canonical PRELIMINARY-only TCA output for the X-07 taker simulator."""

    tca_version: Literal["v0"]
    order_id: Annotated[
        str,
        StringConstraints(
            strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
        ),
    ]
    experiment_id: Literal["X-07"]
    result_label: Literal["PRELIMINARY"]
    fee_formula: FeeFormulaV0
    venue: VenueSlugV0
    market_id: MarketIdV0
    condition_id: ConditionIdV0
    created_at: UtcTimestampV0
    executed_at: UtcTimestampV0
    filled_quantity: FixedPointV0
    entry_vwap: FixedPointV0
    gross_entry_cost: FixedPointV0
    entry_fee: FixedPointV0
    total_entry_cost: FixedPointV0
    entry_levels_consumed: int = Field(gt=0)
    own_delay_microseconds: int = Field(ge=0)
    venue_delay_microseconds: int = Field(ge=0)
    entry_rule_snapshot_ref: Sha256V0
    entry_book_snapshot_ref: Sha256V0
    markouts: tuple[TcaMarkoutV0, ...]

    @field_validator("markouts", mode="before")
    @classmethod
    def _markout_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("markouts")
    @classmethod
    def _markouts_are_nonempty_unique_and_ordered(
        cls, value: tuple[TcaMarkoutV0, ...]
    ) -> tuple[TcaMarkoutV0, ...]:
        if not value:
            raise ValueError("markouts must not be empty")
        horizons = tuple(markout.horizon_microseconds for markout in value)
        if horizons != tuple(sorted(horizons)) or len(horizons) != len(set(horizons)):
            raise ValueError("markout horizons must be unique and ascending")
        return value

    @model_validator(mode="after")
    def _costs_and_markouts_are_exact(self) -> "TcaRecordV0":
        quantity = self.filled_quantity.to_decimal()
        entry_vwap = self.entry_vwap.to_decimal()
        gross_cost = self.gross_entry_cost.to_decimal()
        entry_fee = self.entry_fee.to_decimal()
        total_cost = self.total_entry_cost.to_decimal()
        if quantity <= 0:
            raise ValueError("filled_quantity must be positive")
        if not Decimal(0) <= entry_vwap <= Decimal(1):
            raise ValueError("entry_vwap must be in [0, 1]")
        if gross_cost < 0 or entry_fee < 0 or total_cost < 0:
            raise ValueError("entry costs must be non-negative")
        if gross_cost != quantity * entry_vwap:
            raise ValueError("gross_entry_cost does not match quantity times VWAP")
        if total_cost != gross_cost + entry_fee:
            raise ValueError("total_entry_cost does not match gross cost plus fee")

        created = datetime.fromisoformat(self.created_at[:-1] + "+00:00")
        executed = datetime.fromisoformat(self.executed_at[:-1] + "+00:00")
        if executed < created:
            raise ValueError("executed_at cannot precede created_at")
        for markout in self.markouts:
            observed = datetime.fromisoformat(
                markout.local_receive_at[:-1] + "+00:00"
            )
            elapsed_microseconds = (observed - executed) // timedelta(
                microseconds=1
            )
            if elapsed_microseconds < markout.horizon_microseconds:
                raise ValueError("markout observation precedes its horizon")
            bid = markout.executable_bid_vwap.to_decimal()
            gross_markout = bid - entry_vwap
            if markout.gross_markout_per_unit.to_decimal() != gross_markout:
                raise ValueError("gross markout does not match executable prices")
            net_markout = (
                quantity * bid
                - markout.exit_fee.to_decimal()
                - total_cost
            ) / quantity
            if markout.net_markout_per_unit.to_decimal() != net_markout:
                raise ValueError("net markout does not match entry and exit costs")
        return self


EntityTypeV0 = Literal[
    "competition",
    "game",
    "participant",
    "venue_event",
    "market",
    "outcome",
    "condition",
]
CanonicalDomainIdV0 = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^(?:cmp|game|participant|venue_event|market|outcome|condition)_"
            r"[A-Za-z0-9][A-Za-z0-9._:-]*$"
        ),
    ),
]
MarketOutcomeConditionIdV0 = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^(?:market|outcome|condition)_[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]

_DOMAIN_PREFIX_BY_TYPE = {
    "competition": "cmp_",
    "game": "game_",
    "participant": "participant_",
    "venue_event": "venue_event_",
    "market": "market_",
    "outcome": "outcome_",
    "condition": "condition_",
}


class _EvidenceAssertionV0(_ContractModel):
    evidence_refs: tuple[NonBlankStr, ...]

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def _evidence_input_is_tuple(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_is_a_nonempty_set(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("evidence_refs must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("evidence_refs must be unique")
        return tuple(sorted(value))


class EntityAssertionV0(_EvidenceAssertionV0):
    assertion_version: Literal["v0"]
    entity_type: EntityTypeV0
    canonical_id: CanonicalDomainIdV0
    asserted_at: UtcTimestampV0
    asserted_by: NonBlankStr

    @model_validator(mode="after")
    def _canonical_prefix_matches_type(self) -> "EntityAssertionV0":
        expected_prefix = _DOMAIN_PREFIX_BY_TYPE[self.entity_type]
        if not self.canonical_id.startswith(expected_prefix):
            raise ValueError("canonical_id prefix must match entity_type")
        return self


class NativeAssertionV0(_EvidenceAssertionV0):
    assertion_version: Literal["v0"]
    canonical_id: CanonicalDomainIdV0
    entity_type: EntityTypeV0
    native_namespace: NonBlankStr
    native_id: NonBlankStr
    valid_from: UtcTimestampV0
    valid_to: UtcTimestampV0 | None = None
    asserted_at: UtcTimestampV0

    @model_validator(mode="after")
    def _canonical_prefix_matches_type(self) -> "NativeAssertionV0":
        expected_prefix = _DOMAIN_PREFIX_BY_TYPE[self.entity_type]
        if not self.canonical_id.startswith(expected_prefix):
            raise ValueError("canonical_id prefix must match entity_type")
        return self


class RelationAssertionV0(_EvidenceAssertionV0):
    assertion_version: Literal["v0"]
    left_id: MarketOutcomeConditionIdV0
    relation: MarketRelationV0
    right_id: MarketOutcomeConditionIdV0
    asserted_at: UtcTimestampV0


_MODEL_BY_SCHEMA_NAME: dict[str, type[BaseModel]] = {
    "game-state-step/v0.schema.yaml": GameStateStepV0,
    "id-registry/v0/entity.schema.yaml": EntityAssertionV0,
    "id-registry/v0/native-assertion.schema.yaml": NativeAssertionV0,
    "id-registry/v0/relation-assertion.schema.yaml": RelationAssertionV0,
    "market-metadata-snapshot/v0.schema.yaml": MarketMetadataSnapshotV0,
    "tca/v0.schema.yaml": TcaRecordV0,
    "venue-rule-snapshot/v0.schema.yaml": VenueRuleSnapshotV0,
}
_PROGRAM_ROOT_V0_SCHEMAS = frozenset(
    {
        "event-envelope/v0.schema.yaml",
        "static-dataset-manifest/v0.schema.yaml",
    }
)
_V1_MODEL_BY_SCHEMA_NAME: dict[str, type[BaseModel]] = {
    "model-output/v1.schema.yaml": ModelOutputV1,
}
_QUALITY_FLAG_ADAPTER = TypeAdapter(QualityFlagV0)
_MARKET_RELATION_ADAPTER = TypeAdapter(MarketRelationV0)


def validate_contract_v0(schema_name: str, instance: Any) -> Any:
    """Run the normative v0 semantic validator for one contract schema."""

    if schema_name in _PROGRAM_ROOT_V0_SCHEMAS:
        raise ValueError(
            f"{schema_name} requires its dedicated program-root validator"
        )
    model = _MODEL_BY_SCHEMA_NAME.get(schema_name)
    if model is not None:
        return model.model_validate(_untrusted_round_trip_input(instance))
    if schema_name == "quality-flags/v0.yaml":
        return _QUALITY_FLAG_ADAPTER.validate_python(instance, strict=True)
    if schema_name == "market-relations/v0.yaml":
        return _MARKET_RELATION_ADAPTER.validate_python(instance, strict=True)
    raise ValueError(f"unknown v0 contract schema: {schema_name}")


def validate_contract_v1(
    program_root: str | Path, schema_name: str, instance: Any
) -> Any:
    """Validate one v1 contract including all runtime registry foreign keys."""

    model = _V1_MODEL_BY_SCHEMA_NAME.get(schema_name)
    if model is None:
        raise ValueError(f"unknown v1 contract schema: {schema_name}")
    validated = model.model_validate(_untrusted_round_trip_input(instance))
    if isinstance(validated, ModelOutputV1):
        from prediction_market.experiments import (
            ExperimentRegistryError,
            load_experiment_registry,
            require_execution_authorized,
        )
        from prediction_market.program_audit import (
            ResearchRegistryError,
            load_model_registry,
        )

        try:
            registered = load_experiment_registry(program_root)
            models = {
                row.model_id: row for row in load_model_registry(program_root)
            }
        except (ExperimentRegistryError, ResearchRegistryError) as exc:
            raise ContractValidationError(
                f"model/experiment registry is invalid: {exc}"
            ) from exc
        card = registered.get(validated.experiment_id)
        if card is None:
            raise ContractValidationError(
                f"experiment {validated.experiment_id} is not registered"
            )
        try:
            require_execution_authorized(card)
        except ExperimentRegistryError as exc:
            raise ContractValidationError(str(exc)) from exc
        model_row = models.get(validated.model_id)
        if model_row is None:
            raise ContractValidationError(
                f"model {validated.model_id} is not registered"
            )
        if model_row.model_version != validated.model_version:
            raise ContractValidationError(
                f"model {validated.model_id} version does not match registry"
            )
        if model_row.experiment_id != validated.experiment_id:
            raise ContractValidationError(
                f"model {validated.model_id} is owned by experiment "
                f"{model_row.experiment_id}, not {validated.experiment_id}"
            )
        if model_row.horizon != validated.horizon:
            raise ContractValidationError(
                f"model {validated.model_id} horizon does not match registry"
            )
        if tuple(sorted(model_row.state_space)) != validated.state_space:
            raise ContractValidationError(
                f"model {validated.model_id} state_space does not match registry"
            )
        output_contract = card.get("output_contract")
        if not isinstance(output_contract, dict):
            raise ContractValidationError(
                f"experiment {validated.experiment_id} has no model output contract"
            )
        expected_output = {
            "contract": "model-output/v1.schema.yaml",
            "output_kind": validated.output_kind,
            "transition_unit": validated.transition_unit,
            "state_space": list(validated.state_space),
        }
        actual_output = {
            "contract": output_contract.get("contract"),
            "output_kind": output_contract.get("output_kind"),
            "transition_unit": output_contract.get("transition_unit"),
            "state_space": sorted(output_contract.get("state_space", [])),
        }
        if actual_output != expected_output:
            raise ContractValidationError(
                f"model output transition contract does not match "
                f"experiment {validated.experiment_id}"
            )
        authorized_bindings = [
            scope["input_binding"]
            for scope in card["authorization_scopes"].values()
            if scope["authorized"] is True
            and not scope.get("permanent_no_go", False)
            and isinstance(scope.get("input_binding"), dict)
        ]
        matching_bindings = [
            binding
            for binding in authorized_bindings
            if validated.model_id in binding["model_ids"]
        ]
        if not matching_bindings:
            raise ContractValidationError(
                f"model {validated.model_id} is not bound to a currently "
                f"authorized scope for experiment {validated.experiment_id}"
            )
        if all(
            binding["result_class"] == "synthetic"
            for binding in matching_bindings
        ):
            fixture_hashes = {
                binding["synthetic_data_sha256"]
                for binding in matching_bindings
            }
            if validated.data_sha256 not in fixture_hashes:
                raise ContractValidationError(
                    f"model {validated.model_id} synthetic data_sha256 does "
                    "not match its authorized fixture"
                )
    return validated


def validate_event_envelope_v0(
    program_root: str | Path,
    instance: Any,
) -> EventEnvelopeV0:
    """Validate an event envelope and all program-root-backed payload FKs."""

    envelope = EventEnvelopeV0.model_validate(
        _untrusted_round_trip_input(instance)
    )
    if envelope.experiment_id is not None:
        from prediction_market.experiments import (
            ExperimentRegistryError,
            load_experiment_registry,
        )

        try:
            registered = load_experiment_registry(program_root)
        except ExperimentRegistryError as exc:
            raise ContractValidationError(
                f"experiment registry is invalid: {exc}"
            ) from exc
        if envelope.experiment_id not in registered:
            raise ContractValidationError(
                f"experiment {envelope.experiment_id} is not registered"
            )
    if envelope.event_type == "model_output":
        payload = validate_contract_v1(
            program_root,
            "model-output/v1.schema.yaml",
            thaw_contract_v0(envelope.payload),
        )
        if payload.experiment_id != envelope.experiment_id:
            raise ContractValidationError(
                "model output payload experiment does not match event envelope"
            )
    return envelope


# Short aliases keep consumers from inventing parallel v0 names.
SourceV0 = EventSourceV0
TimeV0 = EventTimeV0
CanonicalRefsV0 = CanonicalReferencesV0
NativeRefV0 = NativeReferenceV0
LineageV0 = EventLineageV0


__all__ = [
    "CanonicalReferencesV0",
    "CanonicalRefsV0",
    "ConditionIdV0",
    "ContractValidationError",
    "DERIVED_EVENT_TYPES",
    "EVENT_TYPES",
    "EntityAssertionV0",
    "EventEnvelopeV0",
    "EventIdV0",
    "EventLineageV0",
    "EventSourceV0",
    "EventTimeV0",
    "FeeFormulaV0",
    "FixedPointV0",
    "FrozenDict",
    "GameStateObservationModeV0",
    "GameStateStepV0",
    "LineageV0",
    "LEVEL2_STREAM_DOMAIN_TAG",
    "MARKET_RELATIONS",
    "MarketMetadataSnapshotV0",
    "ModelOutputV1",
    "NativeAssertionV0",
    "NativeRefV0",
    "NativeReferenceV0",
    "ORDER_TYPES",
    "QUALITY_FLAGS",
    "REGISTERED_EXPERIMENT_IDS",
    "RelationAssertionV0",
    "SIMULATED_EVENT_TYPES",
    "START_TIME_CANCEL_POLICIES",
    "Sha256V0",
    "SourceV0",
    "StaticDatasetLineageV0",
    "StaticDatasetManifestV0",
    "SportV0",
    "TimeV0",
    "TcaMarkoutV0",
    "TcaRecordV0",
    "UtcTimestampV0",
    "VenueRuleSnapshotV0",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "compute_event_id",
    "compute_payload_sha256",
    "game_state_step_sha256",
    "event_id_for",
    "level2_stream_frame",
    "level2_stream_sha256",
    "market_metadata_snapshot_sha256",
    "payload_sha256",
    "replay_order_key",
    "static_dataset_manifest_sha256",
    "validate_static_dataset_manifest_v0",
    "thaw_contract_v0",
    "validate_contract_v0",
    "validate_contract_v1",
    "validate_event_envelope_v0",
]
