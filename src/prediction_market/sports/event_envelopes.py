"""Governed event envelopes for immutable, static sports observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from prediction_market.contracts import (
    EventEnvelopeV0,
    validate_event_envelope_v0,
)


@dataclass(frozen=True, slots=True)
class StaticSportObservationEnvelopePair:
    """The raw pointer and its normalized, experiment-bound observation."""

    raw: EventEnvelopeV0
    normalized: EventEnvelopeV0


@dataclass(frozen=True, slots=True)
class StaticSportObservationEnvelopeBundle:
    """One normalized observation and every raw row used to derive it."""

    raw: tuple[EventEnvelopeV0, ...]
    normalized: EventEnvelopeV0


def _canonical_refs(
    *,
    competition_id: str,
    game_id: str,
    participant_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "competition_id": competition_id,
        "game_id": game_id,
        "participant_ids": participant_ids,
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }


def build_static_sport_observation_bundle(
    *,
    program_root: str | Path,
    experiment_id: str,
    dataset_id: str,
    source_system: str,
    source_stream: str,
    raw_object_hash: str,
    raw_record_ordinals: tuple[int, ...],
    partition: str,
    fetched_at: str,
    source_at: str | None,
    competition_id: str,
    game_id: str,
    participant_ids: tuple[str, ...],
    native_namespace: str,
    native_ids: tuple[str, ...],
    normalized_source_sequence: int,
    normalized_payload: Mapping[str, Any],
    quality_flags: tuple[str, ...] = (),
) -> StaticSportObservationEnvelopeBundle:
    """Build a complete derived observation with exact multi-row lineage."""

    if (
        type(raw_record_ordinals) is not tuple
        or not raw_record_ordinals
        or any(type(value) is not int or value < 0 for value in raw_record_ordinals)
        or len(set(raw_record_ordinals)) != len(raw_record_ordinals)
    ):
        raise ValueError(
            "raw_record_ordinals must be unique non-negative integers"
        )
    if (
        type(native_ids) is not tuple
        or len(native_ids) != len(raw_record_ordinals)
        or any(type(value) is not str or not value for value in native_ids)
        or len(set(native_ids)) != len(native_ids)
    ):
        raise ValueError(
            "native_ids must be unique nonempty strings aligned to raw rows"
        )
    if type(normalized_source_sequence) is not int or normalized_source_sequence < 1:
        raise ValueError("normalized_source_sequence must be a positive integer")
    if normalized_payload.get("game_id") != game_id:
        raise ValueError("normalized payload game_id must match canonical game_id")
    if normalized_payload.get("sequence") != normalized_source_sequence:
        raise ValueError(
            "normalized payload sequence must match normalized source sequence"
        )
    payload_flags = normalized_payload.get("quality_flags")
    if (
        not isinstance(payload_flags, (list, tuple))
        or tuple(sorted(payload_flags)) != tuple(sorted(quality_flags))
    ):
        raise ValueError(
            "normalized payload quality_flags must match envelope quality_flags"
        )

    canonical_refs = _canonical_refs(
        competition_id=competition_id,
        game_id=game_id,
        participant_ids=participant_ids,
    )
    event_time = {
        "receive_at": fetched_at,
        "receive_basis": "upstream_exporter",
        "source_at": source_at,
        "publish_at": None,
        "exchange_at": None,
    }
    raw_envelopes: list[EventEnvelopeV0] = []
    for raw_record_ordinal, native_id in zip(
        raw_record_ordinals,
        native_ids,
        strict=True,
    ):
        raw = EventEnvelopeV0.create(
            envelope_version="v0",
            event_type="raw_observation",
            payload_schema_version="v0",
            source={
                "system": source_system,
                "stream": source_stream,
                "venue": None,
                "sequence": raw_record_ordinal,
                "capture_session_id": f"static:{raw_object_hash}",
                "record_ordinal": raw_record_ordinal,
            },
            time=event_time,
            canonical_refs=canonical_refs,
            native_refs=(
                {
                    "namespace": native_namespace,
                    "native_id": native_id,
                },
            ),
            lineage={
                "raw_object_hash": raw_object_hash,
                "raw_record_ordinal": raw_record_ordinal,
                "parent_event_ids": (),
            },
            experiment_id=None,
            rule_snapshot_ref=None,
            quality_flags=quality_flags,
            payload={
                "dataset_id": dataset_id,
                "partition": partition,
                "raw_object_hash": raw_object_hash,
                "raw_record_ordinal": raw_record_ordinal,
            },
        )
        raw_envelopes.append(validate_event_envelope_v0(program_root, raw))

    normalized = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": source_system,
            "stream": f"{source_stream}.normalized",
            "venue": None,
            "sequence": normalized_source_sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=tuple(
            {
                "namespace": native_namespace,
                "native_id": native_id,
            }
            for native_id in native_ids
        ),
        lineage={
            "raw_object_hash": None,
            "raw_record_ordinal": None,
            "parent_event_ids": tuple(
                parent.event_id for parent in raw_envelopes
            ),
        },
        experiment_id=experiment_id,
        rule_snapshot_ref=None,
        quality_flags=quality_flags,
        payload=normalized_payload,
    )
    normalized = validate_event_envelope_v0(program_root, normalized)
    return StaticSportObservationEnvelopeBundle(
        raw=tuple(raw_envelopes),
        normalized=normalized,
    )


def validate_static_sport_observation_bundle(
    program_root: str | Path,
    envelope: EventEnvelopeV0,
    *,
    raw_parents: tuple[EventEnvelopeV0, ...],
    expected_experiment_id: str,
    expected_dataset_id: str,
    expected_source_system: str,
    expected_source_stream: str,
    expected_native_namespace: str,
) -> EventEnvelopeV0:
    """Fail closed unless a normalized sport envelope is fully raw-bound."""

    if not isinstance(envelope, EventEnvelopeV0):
        raise TypeError("envelope must be a validated EventEnvelopeV0")
    envelope = validate_event_envelope_v0(program_root, envelope)
    if type(raw_parents) is not tuple or not raw_parents:
        raise ValueError("parent raw lineage requires at least one raw envelope")
    validated_parents: list[EventEnvelopeV0] = []
    for parent in raw_parents:
        if not isinstance(parent, EventEnvelopeV0):
            raise TypeError("raw parents must be validated EventEnvelopeV0 objects")
        validated_parents.append(
            validate_event_envelope_v0(program_root, parent)
        )
    raw_parents = tuple(validated_parents)

    if envelope.event_type != "normalized_observation":
        raise ValueError("sport adapter requires a normalized_observation envelope")
    if envelope.experiment_id != expected_experiment_id:
        raise ValueError("normalized envelope experiment binding is invalid")
    if (
        envelope.source.system != expected_source_system
        or envelope.source.stream != f"{expected_source_stream}.normalized"
    ):
        raise ValueError("normalized envelope source binding is invalid")

    parent_ids = tuple(sorted(parent.event_id for parent in raw_parents))
    if envelope.lineage.parent_event_ids != parent_ids:
        raise ValueError("normalized envelope parent raw lineage is incomplete")
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("parent raw lineage must not contain duplicates")

    expected_native_refs: set[tuple[str, str]] = set()
    for parent in raw_parents:
        if parent.event_type != "raw_observation":
            raise ValueError("parent raw lineage must contain raw observations")
        if (
            parent.source.system != expected_source_system
            or parent.source.stream != expected_source_stream
        ):
            raise ValueError("parent raw lineage source binding is invalid")
        if parent.canonical_refs != envelope.canonical_refs:
            raise ValueError("parent raw lineage canonical identity is invalid")
        if (
            len(parent.native_refs) != 1
            or parent.native_refs[0].namespace != expected_native_namespace
        ):
            raise ValueError("parent raw lineage native identity is invalid")
        expected_native_refs.add(
            (
                parent.native_refs[0].namespace,
                parent.native_refs[0].native_id,
            )
        )
        payload = parent.payload
        if payload.get("dataset_id") != expected_dataset_id:
            raise ValueError("parent raw lineage dataset binding is invalid")
        if (
            payload.get("raw_object_hash") != parent.lineage.raw_object_hash
            or payload.get("raw_record_ordinal")
            != parent.lineage.raw_record_ordinal
            or parent.source.record_ordinal
            != parent.lineage.raw_record_ordinal
        ):
            raise ValueError("parent raw lineage payload is inconsistent")

    actual_native_refs = {
        (reference.namespace, reference.native_id)
        for reference in envelope.native_refs
    }
    if actual_native_refs != expected_native_refs:
        raise ValueError("normalized envelope native identity is invalid")
    payload = envelope.payload
    if payload.get("game_id") != envelope.canonical_refs.game_id:
        raise ValueError("normalized envelope game identity is invalid")
    if payload.get("sequence") != envelope.source.sequence:
        raise ValueError("normalized envelope sequence binding is invalid")
    payload_flags = payload.get("quality_flags")
    if (
        not isinstance(payload_flags, tuple)
        or tuple(sorted(payload_flags)) != envelope.quality_flags
    ):
        raise ValueError("normalized envelope quality flags are invalid")
    return envelope


def build_static_sport_observation_envelopes(
    *,
    program_root: str | Path,
    experiment_id: str,
    dataset_id: str,
    source_system: str,
    source_stream: str,
    raw_object_hash: str,
    raw_record_ordinal: int,
    partition: str,
    fetched_at: str,
    source_at: str | None,
    competition_id: str,
    game_id: str,
    participant_ids: tuple[str, ...],
    native_namespace: str,
    native_id: str,
    normalized_payload: Mapping[str, Any],
    quality_flags: tuple[str, ...] = (),
) -> StaticSportObservationEnvelopePair:
    """Create a validated raw pointer and normalized sports observation.

    Static source rows remain byte-addressed by the immutable source object and
    ordinal.  Normalized content is a distinct derived event whose lineage
    points to that raw observation and whose experiment foreign key is checked
    against the program registry.
    """

    canonical_refs = _canonical_refs(
        competition_id=competition_id,
        game_id=game_id,
        participant_ids=participant_ids,
    )
    native_refs = (
        {
            "namespace": native_namespace,
            "native_id": native_id,
        },
    )
    event_time = {
        "receive_at": fetched_at,
        "receive_basis": "upstream_exporter",
        "source_at": source_at,
        "publish_at": None,
        "exchange_at": None,
    }

    raw = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="raw_observation",
        payload_schema_version="v0",
        source={
            "system": source_system,
            "stream": source_stream,
            "venue": None,
            "sequence": raw_record_ordinal,
            "capture_session_id": f"static:{raw_object_hash}",
            "record_ordinal": raw_record_ordinal,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=native_refs,
        lineage={
            "raw_object_hash": raw_object_hash,
            "raw_record_ordinal": raw_record_ordinal,
            "parent_event_ids": (),
        },
        experiment_id=None,
        rule_snapshot_ref=None,
        quality_flags=quality_flags,
        payload={
            "dataset_id": dataset_id,
            "partition": partition,
            "raw_object_hash": raw_object_hash,
            "raw_record_ordinal": raw_record_ordinal,
        },
    )
    raw = validate_event_envelope_v0(program_root, raw)

    normalized = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": source_system,
            "stream": f"{source_stream}.normalized",
            "venue": None,
            "sequence": raw_record_ordinal,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=native_refs,
        lineage={
            "raw_object_hash": None,
            "raw_record_ordinal": None,
            "parent_event_ids": (raw.event_id,),
        },
        experiment_id=experiment_id,
        rule_snapshot_ref=None,
        quality_flags=quality_flags,
        payload=normalized_payload,
    )
    normalized = validate_event_envelope_v0(program_root, normalized)

    return StaticSportObservationEnvelopePair(
        raw=raw,
        normalized=normalized,
    )


__all__ = [
    "StaticSportObservationEnvelopeBundle",
    "StaticSportObservationEnvelopePair",
    "build_static_sport_observation_bundle",
    "build_static_sport_observation_envelopes",
    "validate_static_sport_observation_bundle",
]
