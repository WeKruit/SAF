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
from datetime import datetime
from decimal import Decimal
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
    f"X-{number:02d}" for number in range(1, 11)
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


def _untrusted_round_trip_input(value: Any) -> Any:
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

ExperimentIdV0 = Literal[
    "X-01",
    "X-02",
    "X-03",
    "X-04",
    "X-05",
    "X-06",
    "X-07",
    "X-08",
    "X-09",
    "X-10",
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
        ("venue_rule_snapshot", "order_types_supported"),
    }
)


def _canonical_root(value: Any) -> tuple[str, ...]:
    model_name = type(value).__name__ if isinstance(value, BaseModel) else ""
    if model_name == "EventEnvelopeV0":
        return ("event_envelope",)
    if model_name == "ModelOutputV0":
        return ("model_output",)
    if model_name == "VenueRuleSnapshotV0":
        return ("venue_rule_snapshot",)
    if isinstance(value, Mapping):
        if value.get("envelope_version") == "v0" and "event_type" in value:
            return ("event_envelope",)
        if value.get("contract_version") == "v0" and "probabilities" in value:
            return ("model_output",)
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
    payload_schema_version: Literal["v0"]
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


class ModelOutputV0(_ContractModel):
    """A point-in-time state-transition distribution."""

    contract_version: Literal["v0"]
    model_id: NonBlankStr
    model_version: NonBlankStr
    experiment_id: ExperimentIdV0
    run_id: NonBlankStr
    game_id: GameIdV0
    state_event_id: EventIdV0
    pit_cutoff_at: UtcTimestampV0
    state_space: tuple[NonBlankStr, ...]
    horizon: NonBlankStr
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
    def _probabilities_form_distribution(self) -> "ModelOutputV0":
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
    "event-envelope/v0.schema.yaml": EventEnvelopeV0,
    "id-registry/v0/entity.schema.yaml": EntityAssertionV0,
    "id-registry/v0/native-assertion.schema.yaml": NativeAssertionV0,
    "id-registry/v0/relation-assertion.schema.yaml": RelationAssertionV0,
    "model-output/v0.schema.yaml": ModelOutputV0,
    "venue-rule-snapshot/v0.schema.yaml": VenueRuleSnapshotV0,
}
_QUALITY_FLAG_ADAPTER = TypeAdapter(QualityFlagV0)
_MARKET_RELATION_ADAPTER = TypeAdapter(MarketRelationV0)


def validate_contract_v0(schema_name: str, instance: Any) -> Any:
    """Run the normative v0 semantic validator for one contract schema."""

    model = _MODEL_BY_SCHEMA_NAME.get(schema_name)
    if model is not None:
        return model.model_validate(_untrusted_round_trip_input(instance))
    if schema_name == "quality-flags/v0.yaml":
        return _QUALITY_FLAG_ADAPTER.validate_python(instance, strict=True)
    if schema_name == "market-relations/v0.yaml":
        return _MARKET_RELATION_ADAPTER.validate_python(instance, strict=True)
    raise ValueError(f"unknown v0 contract schema: {schema_name}")


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
    "DERIVED_EVENT_TYPES",
    "EVENT_TYPES",
    "EntityAssertionV0",
    "EventEnvelopeV0",
    "EventIdV0",
    "EventLineageV0",
    "EventSourceV0",
    "EventTimeV0",
    "FixedPointV0",
    "FrozenDict",
    "LineageV0",
    "LEVEL2_STREAM_DOMAIN_TAG",
    "MARKET_RELATIONS",
    "ModelOutputV0",
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
    "TimeV0",
    "UtcTimestampV0",
    "VenueRuleSnapshotV0",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "compute_event_id",
    "compute_payload_sha256",
    "event_id_for",
    "level2_stream_frame",
    "level2_stream_sha256",
    "payload_sha256",
    "replay_order_key",
    "thaw_contract_v0",
    "validate_contract_v0",
]
