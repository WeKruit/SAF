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

    canonical_refs = {
        "competition_id": competition_id,
        "game_id": game_id,
        "participant_ids": participant_ids,
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }
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
    "StaticSportObservationEnvelopePair",
    "build_static_sport_observation_envelopes",
]
