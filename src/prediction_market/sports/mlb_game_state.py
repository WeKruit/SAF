"""Immutable, deterministic MLB game-state primitives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from prediction_market.contracts import (
    EventEnvelopeV0,
    canonical_json,
    canonical_sha256,
    validate_event_envelope_v0,
)
from prediction_market.sports.game_state import (
    advance_state,
    canonical_state_sha256,
)
from prediction_market.static_store import read_verified_static_object


MLB_SPORT = "mlb"
CHADWICK_CWEVENT_VERSION = "0.10.0"
RETROSHEET_DATASET_ID = "DS-RETROSHEET"
RETROSHEET_2025_RAW_OBJECT_SHA256 = (
    "sha256:30592120e807f7b2e073ddee544ac615"
    "a7194db43e551956e22f4a98bf722b7a"
)
RETROSHEET_2025_MANIFEST_SHA256 = (
    "sha256:15eda81cfe06c8c742536fd9a8df671"
    "7215998d529755ff90049007315b6bc91"
)
RETROSHEET_2025_MANIFEST_RELATIVE_PATH = (
    "manifests/source=retrosheet/dataset=DS-RETROSHEET/"
    "version=2025-release/partition=regular-season-2025/"
    "15eda81cfe06c8c742536fd9a8df6717215998d529755ff90049007315b6bc91"
    ".manifest.json"
)
RETROSHEET_FROZEN_GAME_ID = "ANA202504040"
RETROSHEET_FROZEN_EVENT_FILE = "2025ANA.EVA"
RETROSHEET_FROZEN_AWAY_TEAM = "CLE"
RETROSHEET_FROZEN_HOME_TEAM = "ANA"
_EVENT_ID_RE = re.compile(r"evt_[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"game_[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_RETROSHEET_GAME_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z"
)
_CWEVENT_VERSION_RE = re.compile(
    r"Chadwick expanded event descriptor, version ([0-9]+\.[0-9]+\.[0-9]+)"
)
CWEVENT_FIELD_MAP: Mapping[str, tuple[str, int]] = MappingProxyType(
    {
        "game_id": ("GAME_ID", 0),
        "inning": ("INN_CT", 2),
        "batting_home": ("BAT_HOME_ID", 3),
        "outs_before": ("OUTS_CT", 4),
        "balls_before": ("BALLS_CT", 5),
        "strikes_before": ("STRIKES_CT", 6),
        "away_score_before": ("AWAY_SCORE_CT", 8),
        "home_score_before": ("HOME_SCORE_CT", 9),
        "batter_id": ("BAT_ID", 10),
        "pitcher_id": ("PIT_ID", 14),
        "runner_on_first": ("BASE1_RUN_ID", 26),
        "runner_on_second": ("BASE2_RUN_ID", 27),
        "runner_on_third": ("BASE3_RUN_ID", 28),
        "event_text": ("EVENT_TX", 29),
        "lineup_slot": ("BAT_LINEUP_ID", 33),
        "event_code": ("EVENT_CD", 34),
        "batter_event": ("BAT_EVENT_FL", 35),
        "outs_on_play": ("EVENT_OUTS_CT", 40),
        "batter_destination": ("BAT_DEST_ID", 58),
        "runner_on_first_destination": ("RUN1_DEST_ID", 59),
        "runner_on_second_destination": ("RUN2_DEST_ID", 60),
        "runner_on_third_destination": ("RUN3_DEST_ID", 61),
        "end_game": ("GAME_END_FL", 79),
        "event_index": ("EVENT_ID", 96),
    }
)
CWEVENT_FIELD_NAMES = tuple(
    field_name
    for field_name, _ in sorted(CWEVENT_FIELD_MAP.values(), key=lambda item: item[1])
)
CWEVENT_FIELD_ARGUMENT = ",".join(
    str(index)
    for _, index in sorted(CWEVENT_FIELD_MAP.values(), key=lambda item: item[1])
)
CWEVENT_FIELD_MAP_SHA256 = canonical_sha256(
    {
        logical_name: [field_name, index]
        for logical_name, (field_name, index) in CWEVENT_FIELD_MAP.items()
    }
)
HalfInning = Literal["top", "bottom"]
BaseState = tuple[str | None, str | None, str | None]
MLBObservationMode = Literal["offline_reconstruction", "synthetic_fixture"]


class MLBGameStateError(ValueError):
    """A parsed MLB observation cannot produce a trustworthy game state."""


def _require_text(value: object, field_name: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise MLBGameStateError(f"{field_name} must be a canonical nonempty string")


def _require_int(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise MLBGameStateError(f"{field_name} is outside its legal range")


def _require_sha256(value: object, field_name: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MLBGameStateError(f"{field_name} must be lowercase sha256")


def _require_event_id(value: object, field_name: str) -> None:
    if type(value) is not str or _EVENT_ID_RE.fullmatch(value) is None:
        raise MLBGameStateError(f"{field_name} must be an EventEnvelopeV0 ID")


def _require_utc_timestamp(value: object, field_name: str) -> None:
    if type(value) is not str or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise MLBGameStateError(f"{field_name} must be an RFC3339 UTC timestamp")


def _validate_bases(value: object, field_name: str) -> BaseState:
    if type(value) is not tuple or len(value) != 3:
        raise MLBGameStateError(f"{field_name} bases must be a three-item tuple")
    for runner in value:
        if runner is not None:
            _require_text(runner, f"{field_name} runner")
    runners = tuple(runner for runner in value if runner is not None)
    if len(set(runners)) != len(runners):
        raise MLBGameStateError(f"{field_name} runner IDs must be unique")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CweventRuntime:
    """Byte-exact identity of the Chadwick binary used for one replay."""

    executable: str
    version: str
    binary_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.executable, "cwevent executable")
        if not Path(self.executable).is_absolute():
            raise MLBGameStateError("cwevent executable must be an absolute path")
        if self.version != CHADWICK_CWEVENT_VERSION:
            raise MLBGameStateError(
                "Chadwick cwevent version does not match the declared field map"
            )
        _require_sha256(self.binary_sha256, "cwevent binary_sha256")


@dataclass(frozen=True, slots=True)
class MLBOfflineStateProvenance:
    """The exact static row cutoff supporting an offline reconstructed state."""

    observation_mode: Literal["offline_reconstruction"]
    source_dataset_id: str
    raw_object_sha256: str
    source_manifest_sha256: str
    source_fetched_at: str
    cwevent_output_sha256: str
    source_envelope_id: str
    source_row_ordinal: int
    source_row_sha256: str
    reconstruction_cutoff_basis: Literal["cwevent_output_row_ordinal"]
    reconstruction_cutoff_ordinal: int
    cwevent_executable: str
    cwevent_version: str
    cwevent_binary_sha256: str
    cwevent_command_sha256: str
    cwevent_field_map_sha256: str

    def __post_init__(self) -> None:
        if self.observation_mode != "offline_reconstruction":
            raise MLBGameStateError("MLB state provenance must be offline")
        if self.source_dataset_id != RETROSHEET_DATASET_ID:
            raise MLBGameStateError("MLB state provenance dataset is invalid")
        for value, name in (
            (self.raw_object_sha256, "raw_object_sha256"),
            (self.source_manifest_sha256, "source_manifest_sha256"),
            (self.cwevent_output_sha256, "cwevent_output_sha256"),
            (self.source_row_sha256, "source_row_sha256"),
            (self.cwevent_binary_sha256, "cwevent_binary_sha256"),
            (self.cwevent_command_sha256, "cwevent_command_sha256"),
            (self.cwevent_field_map_sha256, "cwevent_field_map_sha256"),
        ):
            _require_sha256(value, name)
        _require_utc_timestamp(self.source_fetched_at, "source_fetched_at")
        _require_event_id(self.source_envelope_id, "source_envelope_id")
        _require_int(self.source_row_ordinal, "source_row_ordinal", minimum=1)
        if self.reconstruction_cutoff_basis != "cwevent_output_row_ordinal":
            raise MLBGameStateError("MLB state cutoff basis is invalid")
        if self.reconstruction_cutoff_ordinal != self.source_row_ordinal:
            raise MLBGameStateError(
                "MLB state cutoff must equal its source row ordinal"
            )
        _require_text(self.cwevent_executable, "cwevent_executable")
        if not Path(self.cwevent_executable).is_absolute():
            raise MLBGameStateError("cwevent executable must be absolute")
        if self.cwevent_version != CHADWICK_CWEVENT_VERSION:
            raise MLBGameStateError("MLB state cwevent version is invalid")
        if self.cwevent_field_map_sha256 != CWEVENT_FIELD_MAP_SHA256:
            raise MLBGameStateError("MLB state cwevent field map is invalid")


@dataclass(frozen=True, slots=True)
class MLBOfflineEventProvenance:
    """Both rows and the later-row cutoff used by an offline play adapter."""

    observation_mode: Literal["offline_reconstruction"]
    source_dataset_id: str
    raw_object_sha256: str
    source_manifest_sha256: str
    source_fetched_at: str
    cwevent_output_sha256: str
    play_envelope_id: str
    play_row_ordinal: int
    play_row_sha256: str
    post_event_envelope_id: str | None
    post_event_row_ordinal: int | None
    post_event_row_sha256: str | None
    reconstruction_cutoff_basis: Literal["cwevent_output_row_ordinal"]
    reconstruction_cutoff_ordinal: int
    cwevent_executable: str
    cwevent_version: str
    cwevent_binary_sha256: str
    cwevent_command_sha256: str
    cwevent_field_map_sha256: str

    def __post_init__(self) -> None:
        if self.observation_mode != "offline_reconstruction":
            raise MLBGameStateError("MLB event provenance must be offline")
        if self.source_dataset_id != RETROSHEET_DATASET_ID:
            raise MLBGameStateError("MLB event provenance dataset is invalid")
        for value, name in (
            (self.raw_object_sha256, "raw_object_sha256"),
            (self.source_manifest_sha256, "source_manifest_sha256"),
            (self.cwevent_output_sha256, "cwevent_output_sha256"),
            (self.play_row_sha256, "play_row_sha256"),
            (self.cwevent_binary_sha256, "cwevent_binary_sha256"),
            (self.cwevent_command_sha256, "cwevent_command_sha256"),
            (self.cwevent_field_map_sha256, "cwevent_field_map_sha256"),
        ):
            _require_sha256(value, name)
        _require_utc_timestamp(self.source_fetched_at, "source_fetched_at")
        _require_event_id(self.play_envelope_id, "play_envelope_id")
        _require_int(self.play_row_ordinal, "play_row_ordinal", minimum=1)
        post_values = (
            self.post_event_envelope_id,
            self.post_event_row_ordinal,
            self.post_event_row_sha256,
        )
        if all(value is None for value in post_values):
            if self.reconstruction_cutoff_ordinal != self.play_row_ordinal:
                raise MLBGameStateError(
                    "terminal event cutoff must equal its play row"
                )
        elif any(value is None for value in post_values):
            raise MLBGameStateError(
                "post-event provenance must be entirely present or absent"
            )
        else:
            _require_event_id(
                self.post_event_envelope_id,
                "post_event_envelope_id",
            )
            _require_int(
                self.post_event_row_ordinal,
                "post_event_row_ordinal",
                minimum=1,
            )
            _require_sha256(
                self.post_event_row_sha256,
                "post_event_row_sha256",
            )
            if self.post_event_row_ordinal != self.play_row_ordinal + 1:
                raise MLBGameStateError(
                    "post-event row must immediately follow the play row"
                )
            if self.reconstruction_cutoff_ordinal != self.post_event_row_ordinal:
                raise MLBGameStateError(
                    "event cutoff must equal the post-event row ordinal"
                )
        if self.reconstruction_cutoff_basis != "cwevent_output_row_ordinal":
            raise MLBGameStateError("MLB event cutoff basis is invalid")
        _require_text(self.cwevent_executable, "cwevent_executable")
        if not Path(self.cwevent_executable).is_absolute():
            raise MLBGameStateError("cwevent executable must be absolute")
        if self.cwevent_version != CHADWICK_CWEVENT_VERSION:
            raise MLBGameStateError("MLB event cwevent version is invalid")
        if self.cwevent_field_map_sha256 != CWEVENT_FIELD_MAP_SHA256:
            raise MLBGameStateError("MLB event cwevent field map is invalid")


@dataclass(frozen=True, slots=True)
class MLBRowEnvelopeEvidence:
    """Canonical primitive-only commitment to one validated raw envelope."""

    event_id: str
    envelope_json: str
    envelope_sha256: str
    payload_sha256: str
    stream_identity_sha256: str
    source_row_ordinal: int
    source_row_sha256: str

    def __post_init__(self) -> None:
        _require_event_id(self.event_id, "row envelope event_id")
        _require_text(self.envelope_json, "row envelope_json")
        _require_sha256(self.envelope_sha256, "row envelope_sha256")
        _require_sha256(self.payload_sha256, "row payload_sha256")
        _require_sha256(
            self.stream_identity_sha256,
            "row stream_identity_sha256",
        )
        _require_int(
            self.source_row_ordinal,
            "row envelope ordinal",
            minimum=1,
        )
        _require_sha256(self.source_row_sha256, "row source_row_sha256")
        try:
            parsed = json.loads(self.envelope_json)
            if self.envelope_json != canonical_json(parsed):
                raise MLBGameStateError("row envelope_json is not canonical")
            envelope = EventEnvelopeV0.model_validate(parsed)
        except (TypeError, ValueError) as exc:
            raise MLBGameStateError("row envelope evidence is invalid") from exc
        if (
            envelope.event_type != "raw_observation"
            or envelope.experiment_id is not None
            or envelope.event_id != self.event_id
            or envelope.payload_sha256 != self.payload_sha256
            or canonical_sha256(parsed) != self.envelope_sha256
            or envelope.source.record_ordinal != self.source_row_ordinal
            or envelope.lineage.raw_record_ordinal != self.source_row_ordinal
            or envelope.payload["source_row_sha256"] != self.source_row_sha256
        ):
            raise MLBGameStateError("row envelope evidence identity is invalid")


def _offline_stream_identity_sha256(
    *,
    source_dataset_id: str,
    raw_object_sha256: str,
    source_manifest_sha256: str,
    source_fetched_at: str,
    cwevent_output_sha256: str,
    cwevent_executable: str,
    cwevent_version: str,
    cwevent_binary_sha256: str,
    cwevent_command_sha256: str,
    cwevent_field_map_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "source_dataset_id": source_dataset_id,
            "raw_object_sha256": raw_object_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_fetched_at": source_fetched_at,
            "cwevent_output_sha256": cwevent_output_sha256,
            "cwevent_executable": cwevent_executable,
            "cwevent_version": cwevent_version,
            "cwevent_binary_sha256": cwevent_binary_sha256,
            "cwevent_command_sha256": cwevent_command_sha256,
            "cwevent_field_map_sha256": cwevent_field_map_sha256,
        }
    )


def _row_envelope_evidence(
    envelope: EventEnvelopeV0,
) -> MLBRowEnvelopeEvidence:
    payload = envelope.payload
    envelope_json = canonical_json(
        envelope.model_dump(mode="json", round_trip=True)
    )
    return MLBRowEnvelopeEvidence(
        event_id=envelope.event_id,
        envelope_json=envelope_json,
        envelope_sha256=canonical_sha256(json.loads(envelope_json)),
        payload_sha256=envelope.payload_sha256,
        stream_identity_sha256=_offline_stream_identity_sha256(
            source_dataset_id=payload["dataset_id"],
            raw_object_sha256=payload["raw_object_sha256"],
            source_manifest_sha256=payload["source_manifest_sha256"],
            source_fetched_at=payload["source_fetched_at"],
            cwevent_output_sha256=payload["cwevent_output_sha256"],
            cwevent_executable=payload["cwevent_executable"],
            cwevent_version=payload["cwevent_version"],
            cwevent_binary_sha256=payload["cwevent_binary_sha256"],
            cwevent_command_sha256=payload["cwevent_command_sha256"],
            cwevent_field_map_sha256=payload["cwevent_field_map_sha256"],
        ),
        source_row_ordinal=payload["source_row_ordinal"],
        source_row_sha256=payload["source_row_sha256"],
    )


def _require_envelope_matches_offline_row(
    evidence: MLBRowEnvelopeEvidence | None,
    *,
    source_dataset_id: str,
    raw_object_sha256: str,
    source_manifest_sha256: str,
    source_fetched_at: str,
    cwevent_output_sha256: str,
    cwevent_executable: str,
    cwevent_version: str,
    cwevent_binary_sha256: str,
    cwevent_command_sha256: str,
    cwevent_field_map_sha256: str,
    expected_envelope_id: str,
    expected_row_ordinal: int,
    expected_row_sha256: str,
    label: str,
) -> None:
    if not isinstance(evidence, MLBRowEnvelopeEvidence):
        raise MLBGameStateError(f"{label} requires validated envelope evidence")
    expected_stream_identity = _offline_stream_identity_sha256(
        source_dataset_id=source_dataset_id,
        raw_object_sha256=raw_object_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_fetched_at=source_fetched_at,
        cwevent_output_sha256=cwevent_output_sha256,
        cwevent_executable=cwevent_executable,
        cwevent_version=cwevent_version,
        cwevent_binary_sha256=cwevent_binary_sha256,
        cwevent_command_sha256=cwevent_command_sha256,
        cwevent_field_map_sha256=cwevent_field_map_sha256,
    )
    if (
        evidence.event_id != expected_envelope_id
        or evidence.source_row_ordinal != expected_row_ordinal
        or evidence.source_row_sha256 != expected_row_sha256
        or evidence.stream_identity_sha256 != expected_stream_identity
    ):
        raise MLBGameStateError(f"{label} envelope provenance identity is invalid")


def _require_state_envelope_identity(
    provenance: MLBOfflineStateProvenance,
    evidence: MLBRowEnvelopeEvidence | None,
) -> None:
    _require_envelope_matches_offline_row(
        evidence,
        source_dataset_id=provenance.source_dataset_id,
        raw_object_sha256=provenance.raw_object_sha256,
        source_manifest_sha256=provenance.source_manifest_sha256,
        source_fetched_at=provenance.source_fetched_at,
        cwevent_output_sha256=provenance.cwevent_output_sha256,
        cwevent_executable=provenance.cwevent_executable,
        cwevent_version=provenance.cwevent_version,
        cwevent_binary_sha256=provenance.cwevent_binary_sha256,
        cwevent_command_sha256=provenance.cwevent_command_sha256,
        cwevent_field_map_sha256=provenance.cwevent_field_map_sha256,
        expected_envelope_id=provenance.source_envelope_id,
        expected_row_ordinal=provenance.source_row_ordinal,
        expected_row_sha256=provenance.source_row_sha256,
        label="offline state",
    )


def _require_event_envelope_identity(
    provenance: MLBOfflineEventProvenance,
    play_evidence: MLBRowEnvelopeEvidence | None,
    post_event_evidence: MLBRowEnvelopeEvidence | None,
) -> None:
    common = {
        "source_dataset_id": provenance.source_dataset_id,
        "raw_object_sha256": provenance.raw_object_sha256,
        "source_manifest_sha256": provenance.source_manifest_sha256,
        "source_fetched_at": provenance.source_fetched_at,
        "cwevent_output_sha256": provenance.cwevent_output_sha256,
        "cwevent_executable": provenance.cwevent_executable,
        "cwevent_version": provenance.cwevent_version,
        "cwevent_binary_sha256": provenance.cwevent_binary_sha256,
        "cwevent_command_sha256": provenance.cwevent_command_sha256,
        "cwevent_field_map_sha256": provenance.cwevent_field_map_sha256,
    }
    _require_envelope_matches_offline_row(
        play_evidence,
        **common,
        expected_envelope_id=provenance.play_envelope_id,
        expected_row_ordinal=provenance.play_row_ordinal,
        expected_row_sha256=provenance.play_row_sha256,
        label="offline play",
    )
    if provenance.post_event_envelope_id is None:
        if post_event_evidence is not None:
            raise MLBGameStateError(
                "terminal offline event cannot contain a post-event envelope"
            )
        return
    assert provenance.post_event_row_ordinal is not None
    assert provenance.post_event_row_sha256 is not None
    _require_envelope_matches_offline_row(
        post_event_evidence,
        **common,
        expected_envelope_id=provenance.post_event_envelope_id,
        expected_row_ordinal=provenance.post_event_row_ordinal,
        expected_row_sha256=provenance.post_event_row_sha256,
        label="offline post-event",
    )


@dataclass(frozen=True, slots=True)
class MLBScore:
    away: int
    home: int

    def __post_init__(self) -> None:
        _require_int(self.away, "away score", minimum=0)
        _require_int(self.home, "home score", minimum=0)


@dataclass(frozen=True, slots=True)
class RunnerAdvance:
    runner_id: str
    start_base: int
    destination: int

    def __post_init__(self) -> None:
        _require_text(self.runner_id, "runner_id")
        if type(self.start_base) is not int or not 0 <= self.start_base <= 3:
            raise MLBGameStateError("runner start base must be in 0..3")
        if type(self.destination) is not int or not 0 <= self.destination <= 4:
            raise MLBGameStateError("runner destination must be in 0..4")
        if self.destination not in {0, 4} and self.destination < self.start_base:
            raise MLBGameStateError("runner destination cannot move backward")


@dataclass(frozen=True, slots=True)
class InningTransition:
    inning: int
    half: HalfInning
    batting_team: str
    fielding_team: str
    batter_id: str
    pitcher_id: str
    lineup_slot: int
    balls: int
    strikes: int

    def __post_init__(self) -> None:
        _require_int(self.inning, "transition inning", minimum=1)
        if self.half not in {"top", "bottom"}:
            raise MLBGameStateError("transition half must be top or bottom")
        _require_text(self.batting_team, "transition batting_team")
        _require_text(self.fielding_team, "transition fielding_team")
        if self.batting_team == self.fielding_team:
            raise MLBGameStateError("transition teams must differ")
        _require_text(self.batter_id, "transition batter_id")
        _require_text(self.pitcher_id, "transition pitcher_id")
        _require_int(self.lineup_slot, "transition lineup slot", minimum=1, maximum=9)
        _require_int(self.balls, "transition balls", minimum=0, maximum=3)
        _require_int(self.strikes, "transition strikes", minimum=0, maximum=2)


@dataclass(frozen=True, slots=True)
class MLBPlayEvent:
    sport: str
    game_id: str
    sequence: int
    event_id: str
    inning: int
    half: HalfInning
    outs_before: int
    bases_before: BaseState
    score_before: MLBScore
    batting_team: str
    fielding_team: str
    batter_id: str
    pitcher_id: str
    balls_before: int
    strikes_before: int
    lineup_slot_before: int
    play_type: str
    runs: tuple[str, ...]
    outs: tuple[str, ...]
    runner_destinations: tuple[RunnerAdvance, ...]
    next_batter_id: str | None
    next_pitcher_id: str | None
    next_balls: int | None
    next_strikes: int | None
    next_lineup_slot: int | None
    inning_transition: InningTransition | None = None
    terminal: bool = False
    observation_mode: MLBObservationMode = "synthetic_fixture"
    source_provenance: MLBOfflineEventProvenance | None = None
    play_envelope_evidence: MLBRowEnvelopeEvidence | None = None
    post_event_envelope_evidence: MLBRowEnvelopeEvidence | None = None
    source_parser: str = "parsed_observation"
    source_parser_version: str = "v1"
    source_event_index: int | None = None

    def __post_init__(self) -> None:
        if self.sport != MLB_SPORT:
            raise MLBGameStateError("event sport must be mlb")
        if type(self.game_id) is not str or _GAME_ID_RE.fullmatch(self.game_id) is None:
            raise MLBGameStateError("event game_id must be canonical")
        _require_int(self.sequence, "event sequence", minimum=1)
        if type(self.event_id) is not str or _EVENT_ID_RE.fullmatch(self.event_id) is None:
            raise MLBGameStateError("event_id must be an external evt_<lowercase sha256>")
        _require_int(self.inning, "event inning", minimum=1)
        if self.half not in {"top", "bottom"}:
            raise MLBGameStateError("event half must be top or bottom")
        _require_int(self.outs_before, "event outs", minimum=0, maximum=2)
        _validate_bases(self.bases_before, "event")
        if not isinstance(self.score_before, MLBScore):
            raise MLBGameStateError("event score must be MLBScore")
        for value, name in (
            (self.batting_team, "event batting_team"),
            (self.fielding_team, "event fielding_team"),
            (self.batter_id, "event batter_id"),
            (self.pitcher_id, "event pitcher_id"),
            (self.play_type, "event play_type"),
        ):
            _require_text(value, name)
        _require_int(self.balls_before, "event balls", minimum=0, maximum=3)
        _require_int(self.strikes_before, "event strikes", minimum=0, maximum=2)
        _require_int(
            self.lineup_slot_before,
            "event lineup slot",
            minimum=1,
            maximum=9,
        )
        if type(self.runs) is not tuple or type(self.outs) is not tuple:
            raise MLBGameStateError("event runs and outs must be tuples")
        for runner in (*self.runs, *self.outs):
            _require_text(runner, "event run/out runner")
        if type(self.runner_destinations) is not tuple or any(
            not isinstance(advance, RunnerAdvance)
            for advance in self.runner_destinations
        ):
            raise MLBGameStateError(
                "runner_destinations must be a tuple of RunnerAdvance"
            )
        if type(self.terminal) is not bool:
            raise MLBGameStateError("event terminal must be boolean")
        if self.observation_mode not in {
            "offline_reconstruction",
            "synthetic_fixture",
        }:
            raise MLBGameStateError("event observation_mode is invalid")
        if self.observation_mode == "offline_reconstruction":
            if not isinstance(
                self.source_provenance,
                MLBOfflineEventProvenance,
            ):
                raise MLBGameStateError(
                    "offline event requires complete source provenance"
                )
            if self.event_id != self.source_provenance.play_envelope_id:
                raise MLBGameStateError(
                    "offline event_id must equal the validated play envelope ID"
                )
            _require_event_envelope_identity(
                self.source_provenance,
                self.play_envelope_evidence,
                self.post_event_envelope_evidence,
            )
        elif (
            self.source_provenance is not None
            or self.play_envelope_evidence is not None
            or self.post_event_envelope_evidence is not None
        ):
            raise MLBGameStateError(
                "synthetic event cannot contain offline source evidence"
            )
        _require_text(self.source_parser, "source_parser")
        _require_text(self.source_parser_version, "source_parser_version")
        if self.source_event_index is not None:
            _require_int(
                self.source_event_index,
                "source_event_index",
                minimum=1,
            )
        next_values = (
            self.next_batter_id,
            self.next_pitcher_id,
            self.next_balls,
            self.next_strikes,
            self.next_lineup_slot,
        )
        if self.terminal or self.inning_transition is not None:
            if any(value is not None for value in next_values):
                raise MLBGameStateError(
                    "terminal/transition event cannot contain next plate appearance"
                )
        else:
            if any(value is None for value in next_values):
                raise MLBGameStateError(
                    "nonterminal event requires the next plate appearance"
                )
            _require_text(self.next_batter_id, "next_batter_id")
            _require_text(self.next_pitcher_id, "next_pitcher_id")
            _require_int(self.next_balls, "next balls", minimum=0, maximum=3)
            _require_int(self.next_strikes, "next strikes", minimum=0, maximum=2)
            _require_int(
                self.next_lineup_slot,
                "next lineup slot",
                minimum=1,
                maximum=9,
            )
        advances_by_runner = {
            advance.runner_id: advance for advance in self.runner_destinations
        }
        if len(advances_by_runner) != len(self.runner_destinations):
            raise MLBGameStateError("runner destinations must have unique runner IDs")
        destination_runs = {
            advance.runner_id
            for advance in self.runner_destinations
            if advance.destination == 4
        }
        if set(self.runs) != destination_runs or len(set(self.runs)) != len(self.runs):
            raise MLBGameStateError("explicit runs must match home destinations")
        destination_outs = {
            advance.runner_id
            for advance in self.runner_destinations
            if advance.destination == 0
        }
        if set(self.outs) != destination_outs or len(set(self.outs)) != len(self.outs):
            raise MLBGameStateError("explicit outs must match out destinations")


@dataclass(frozen=True, slots=True)
class MLBGameState:
    sport: str
    game_id: str
    sequence: int
    inning: int
    half: HalfInning
    outs: int
    bases: BaseState
    score: MLBScore
    away_team: str
    home_team: str
    batting_team: str
    fielding_team: str
    batter_id: str
    pitcher_id: str
    balls: int
    strikes: int
    lineup_slot: int
    terminal: bool = False
    observation_mode: MLBObservationMode = "synthetic_fixture"
    source_provenance: MLBOfflineStateProvenance | None = None
    source_envelope_evidence: MLBRowEnvelopeEvidence | None = None

    def __post_init__(self) -> None:
        if self.sport != MLB_SPORT:
            raise MLBGameStateError("state sport must be mlb")
        if type(self.game_id) is not str or _GAME_ID_RE.fullmatch(self.game_id) is None:
            raise MLBGameStateError("state game_id must be canonical")
        _require_int(self.sequence, "state sequence", minimum=0)
        _require_int(self.inning, "state inning", minimum=1)
        if self.half not in {"top", "bottom"}:
            raise MLBGameStateError("state half must be top or bottom")
        if type(self.terminal) is not bool:
            raise MLBGameStateError("state terminal must be boolean")
        if self.observation_mode not in {
            "offline_reconstruction",
            "synthetic_fixture",
        }:
            raise MLBGameStateError("state observation_mode is invalid")
        if self.observation_mode == "offline_reconstruction":
            if not isinstance(
                self.source_provenance,
                MLBOfflineStateProvenance,
            ):
                raise MLBGameStateError(
                    "offline state requires complete source provenance"
                )
            _require_state_envelope_identity(
                self.source_provenance,
                self.source_envelope_evidence,
            )
        elif (
            self.source_provenance is not None
            or self.source_envelope_evidence is not None
        ):
            raise MLBGameStateError(
                "synthetic state cannot contain offline source evidence"
            )
        _require_int(
            self.outs,
            "state outs",
            minimum=0,
            maximum=3 if self.terminal else 2,
        )
        _validate_bases(self.bases, "state")
        if not isinstance(self.score, MLBScore):
            raise MLBGameStateError("state score must be MLBScore")
        for value, name in (
            (self.away_team, "away_team"),
            (self.home_team, "home_team"),
            (self.batting_team, "batting_team"),
            (self.fielding_team, "fielding_team"),
            (self.batter_id, "batter_id"),
            (self.pitcher_id, "pitcher_id"),
        ):
            _require_text(value, name)
        if self.away_team == self.home_team:
            raise MLBGameStateError("away and home teams must differ")
        expected_batting = self.away_team if self.half == "top" else self.home_team
        expected_fielding = self.home_team if self.half == "top" else self.away_team
        if self.batting_team != expected_batting:
            raise MLBGameStateError("batting team is inconsistent with half inning")
        if self.fielding_team != expected_fielding:
            raise MLBGameStateError("fielding team is inconsistent with half inning")
        if not self.terminal and self.batter_id in self.bases:
            raise MLBGameStateError("current batter cannot also be a base runner")
        _require_int(self.balls, "state balls", minimum=0, maximum=3)
        _require_int(self.strikes, "state strikes", minimum=0, maximum=2)
        _require_int(self.lineup_slot, "state lineup slot", minimum=1, maximum=9)


def initial_state(
    *,
    game_id: str,
    away_team: str,
    home_team: str,
    batter_id: str,
    pitcher_id: str,
    lineup_slot: int = 1,
) -> MLBGameState:
    """Create the state immediately before the first plate appearance."""

    return MLBGameState(
        sport=MLB_SPORT,
        game_id=game_id,
        sequence=0,
        inning=1,
        half="top",
        outs=0,
        bases=(None, None, None),
        score=MLBScore(away=0, home=0),
        away_team=away_team,
        home_team=home_team,
        batting_team=away_team,
        fielding_team=home_team,
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        balls=0,
        strikes=0,
        lineup_slot=lineup_slot,
        terminal=False,
    )


def retrosheet_game_id(native_game_id: str) -> str:
    """Return the common canonical ID for one Retrosheet game."""

    _require_text(native_game_id, "Retrosheet game_id")
    if native_game_id.startswith("game_"):
        if _GAME_ID_RE.fullmatch(native_game_id) is None:
            raise MLBGameStateError("Retrosheet game_id must be canonical")
        return native_game_id
    if _RETROSHEET_GAME_ID_RE.fullmatch(native_game_id) is None:
        raise MLBGameStateError("Retrosheet game_id contains unsupported characters")
    return f"game_retrosheet_{native_game_id}"


def require_cwevent_runtime(executable: str = "cwevent") -> CweventRuntime:
    """Inspect the actual executable, exact version, and binary content hash."""

    _require_text(executable, "cwevent executable")
    resolved = shutil.which(executable)
    if resolved is None:
        raise MLBGameStateError("Chadwick cwevent executable is not installed")
    try:
        actual_path = Path(resolved).resolve(strict=True)
        binary_sha256 = (
            "sha256:" + hashlib.sha256(actual_path.read_bytes()).hexdigest()
        )
    except OSError as exc:
        raise MLBGameStateError("Chadwick cwevent binary cannot be hashed") from exc
    try:
        completed = subprocess.run(
            [str(actual_path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MLBGameStateError("Chadwick cwevent version cannot be inspected") from exc
    banner = completed.stdout + completed.stderr
    match = _CWEVENT_VERSION_RE.search(banner)
    if match is None:
        raise MLBGameStateError("Chadwick cwevent version banner is missing")
    version = match.group(1)
    if version != CHADWICK_CWEVENT_VERSION:
        raise MLBGameStateError(
            "Chadwick cwevent version does not match the declared field map"
        )
    return CweventRuntime(
        executable=str(actual_path),
        version=version,
        binary_sha256=binary_sha256,
    )


def require_cwevent_version(executable: str = "cwevent") -> str:
    """Require the exact live binary version used by the declared field map."""

    return require_cwevent_runtime(executable).version


def _cwevent_value(row: Mapping[str, object], logical_name: str) -> object:
    if not isinstance(row, Mapping):
        raise MLBGameStateError("cwevent observation must be a mapping")
    field_name = CWEVENT_FIELD_MAP[logical_name][0]
    if field_name not in row:
        raise MLBGameStateError(f"cwevent observation lacks {field_name}")
    return row[field_name]


def _cwevent_text(
    row: Mapping[str, object],
    logical_name: str,
    *,
    optional: bool = False,
) -> str | None:
    value = _cwevent_value(row, logical_name)
    if optional and value == "":
        return None
    _require_text(value, f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]}")
    assert type(value) is str
    return value


def _cwevent_int(
    row: Mapping[str, object],
    logical_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = _cwevent_value(row, logical_name)
    if type(value) is int:
        parsed = value
    elif type(value) is str and re.fullmatch(r"-?[0-9]+", value):
        parsed = int(value)
    else:
        raise MLBGameStateError(
            f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]} must be an integer"
        )
    _require_int(
        parsed,
        f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]}",
        minimum=minimum,
        maximum=maximum,
    )
    return parsed


def _cwevent_flag(row: Mapping[str, object], logical_name: str) -> bool:
    value = _cwevent_value(row, logical_name)
    if type(value) is bool:
        return value
    if type(value) is int and value in {0, 1}:
        return bool(value)
    if type(value) is str and value in {"T", "F"}:
        return value == "T"
    raise MLBGameStateError(
        f"cwevent {CWEVENT_FIELD_MAP[logical_name][0]} must be T or F"
    )


def _cwevent_snapshot(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
) -> dict[str, object]:
    native_game_id = _cwevent_text(row, "game_id")
    assert native_game_id is not None
    batting_home = _cwevent_int(
        row,
        "batting_home",
        minimum=0,
        maximum=1,
    )
    half: HalfInning = "bottom" if batting_home else "top"
    return {
        "game_id": retrosheet_game_id(native_game_id),
        "inning": _cwevent_int(row, "inning", minimum=1),
        "half": half,
        "outs": _cwevent_int(row, "outs_before", minimum=0, maximum=2),
        "bases": (
            _cwevent_text(row, "runner_on_first", optional=True),
            _cwevent_text(row, "runner_on_second", optional=True),
            _cwevent_text(row, "runner_on_third", optional=True),
        ),
        "score": MLBScore(
            away=_cwevent_int(row, "away_score_before", minimum=0),
            home=_cwevent_int(row, "home_score_before", minimum=0),
        ),
        "batting_team": home_team if batting_home else away_team,
        "fielding_team": away_team if batting_home else home_team,
        "batter_id": _cwevent_text(row, "batter_id"),
        "pitcher_id": _cwevent_text(row, "pitcher_id"),
        "balls": _cwevent_int(row, "balls_before", minimum=0, maximum=3),
        "strikes": _cwevent_int(row, "strikes_before", minimum=0, maximum=2),
        "lineup_slot": _cwevent_int(
            row,
            "lineup_slot",
            minimum=1,
            maximum=9,
        ),
    }


def _require_snapshot_matches_state(
    snapshot: Mapping[str, object],
    state: MLBGameState,
) -> None:
    for field_name in (
        "game_id",
        "inning",
        "half",
        "outs",
        "bases",
        "score",
        "batting_team",
        "fielding_team",
        "batter_id",
        "pitcher_id",
        "balls",
        "strikes",
        "lineup_slot",
    ):
        if snapshot[field_name] != getattr(state, field_name):
            raise MLBGameStateError(
                f"cwevent {field_name} does not match the supplied state"
            )


def cwevent_row_sha256(row: Mapping[str, object]) -> str:
    """Hash one decoded cwevent row under the frozen field order."""

    if not isinstance(row, Mapping):
        raise MLBGameStateError("cwevent row must be a mapping")
    if set(row) != set(CWEVENT_FIELD_NAMES):
        raise MLBGameStateError("cwevent row fields do not match the frozen map")
    return canonical_sha256(
        {field_name: row[field_name] for field_name in CWEVENT_FIELD_NAMES}
    )


def cwevent_command(
    runtime: CweventRuntime,
    *,
    native_game_id: str,
    event_file_name: str,
) -> tuple[str, ...]:
    """Return the only cwevent command authorized by the frozen adapter."""

    if not isinstance(runtime, CweventRuntime):
        raise MLBGameStateError("cwevent command requires a verified runtime")
    _require_text(native_game_id, "native_game_id")
    _require_text(event_file_name, "event_file_name")
    if PurePosixPath(event_file_name).name != event_file_name:
        raise MLBGameStateError("event_file_name must be a safe basename")
    return (
        runtime.executable,
        "-q",
        "-n",
        "-i",
        native_game_id,
        "-y",
        "2025",
        "-f",
        CWEVENT_FIELD_ARGUMENT,
        event_file_name,
    )


def _validate_cwevent_command(
    value: object,
    *,
    runtime: CweventRuntime,
    native_game_id: str,
) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise MLBGameStateError("cwevent command must be a nonempty tuple")
    for argument in value:
        _require_text(argument, "cwevent command argument")
    if value[0] != runtime.executable:
        raise MLBGameStateError("cwevent command executable is not the verified binary")
    if value.count("-f") != 1:
        raise MLBGameStateError("cwevent command must contain one field map")
    field_position = value.index("-f")
    if field_position + 1 >= len(value) or value[field_position + 1] != (
        CWEVENT_FIELD_ARGUMENT
    ):
        raise MLBGameStateError("cwevent command field map is not frozen")
    if value.count("-i") != 1:
        raise MLBGameStateError("cwevent command must contain one game filter")
    game_position = value.index("-i")
    if game_position + 1 >= len(value) or value[game_position + 1] != native_game_id:
        raise MLBGameStateError("cwevent command game filter is inconsistent")
    if "-q" not in value or "-n" not in value:
        raise MLBGameStateError("cwevent command must be quiet and emit a header")
    return value


def build_cwevent_row_envelope(
    *,
    program_root: str | Path,
    row: Mapping[str, object],
    row_ordinal: int,
    raw_object_sha256: str,
    source_manifest_sha256: str,
    source_fetched_at: str,
    cwevent_output_sha256: str,
    cwevent_command: tuple[str, ...],
    cwevent_executable: str,
    away_team: str,
    home_team: str,
) -> EventEnvelopeV0:
    """Bind one deterministic cwevent output row to the immutable source."""

    _require_int(row_ordinal, "cwevent row ordinal", minimum=1)
    _require_sha256(raw_object_sha256, "raw_object_sha256")
    _require_sha256(source_manifest_sha256, "source_manifest_sha256")
    _require_sha256(cwevent_output_sha256, "cwevent_output_sha256")
    _require_utc_timestamp(source_fetched_at, "source_fetched_at")
    _require_text(away_team, "away_team")
    _require_text(home_team, "home_team")
    if away_team == home_team:
        raise MLBGameStateError("away and home teams must differ")
    runtime = require_cwevent_runtime(cwevent_executable)
    native_game_id = _cwevent_text(row, "game_id")
    assert native_game_id is not None
    source_event_index = _cwevent_int(row, "event_index", minimum=1)
    if source_event_index != row_ordinal:
        raise MLBGameStateError(
            "frozen game cwevent EVENT_ID must equal output row ordinal"
        )
    command = _validate_cwevent_command(
        cwevent_command,
        runtime=runtime,
        native_game_id=native_game_id,
    )
    row_sha256 = cwevent_row_sha256(row)
    command_sha256 = canonical_sha256(list(command))
    canonical_game_id = retrosheet_game_id(native_game_id)
    envelope = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="raw_observation",
        payload_schema_version="v0",
        source={
            "system": "retrosheet.chadwick",
            "stream": "cwevent.offline_reconstruction",
            "venue": None,
            "sequence": source_event_index,
            "capture_session_id": f"derived:{cwevent_output_sha256}",
            "record_ordinal": row_ordinal,
        },
        time={
            "receive_at": source_fetched_at,
            "receive_basis": "upstream_exporter",
            "source_at": None,
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs={
            "competition_id": "cmp_mlb_2025",
            "game_id": canonical_game_id,
            "participant_ids": (
                f"participant_{away_team}",
                f"participant_{home_team}",
            ),
            "venue_event_id": None,
            "market_id": None,
            "outcome_id": None,
            "condition_id": None,
        },
        native_refs=(
            {
                "namespace": "retrosheet.cwevent_event",
                "native_id": f"{native_game_id}:{source_event_index}",
            },
        ),
        lineage={
            "raw_object_hash": cwevent_output_sha256,
            "raw_record_ordinal": row_ordinal,
            "parent_event_ids": (),
        },
        experiment_id=None,
        rule_snapshot_ref=None,
        quality_flags=(),
        payload={
            "dataset_id": RETROSHEET_DATASET_ID,
            "observation_mode": "offline_reconstruction",
            "raw_object_sha256": raw_object_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_fetched_at": source_fetched_at,
            "cwevent_output_sha256": cwevent_output_sha256,
            "source_row_ordinal": row_ordinal,
            "source_row_sha256": row_sha256,
            "cwevent_executable": runtime.executable,
            "cwevent_version": runtime.version,
            "cwevent_binary_sha256": runtime.binary_sha256,
            "cwevent_command": command,
            "cwevent_command_sha256": command_sha256,
            "cwevent_field_map_sha256": CWEVENT_FIELD_MAP_SHA256,
        },
    )
    try:
        return validate_event_envelope_v0(program_root, envelope)
    except Exception as exc:
        raise MLBGameStateError(
            "cwevent row EventEnvelopeV0 validation failed"
        ) from exc


def _validated_cwevent_row_envelope(
    *,
    program_root: str | Path,
    row: Mapping[str, object],
    row_envelope: EventEnvelopeV0,
    runtime: CweventRuntime,
    away_team: str,
    home_team: str,
) -> EventEnvelopeV0:
    try:
        envelope = validate_event_envelope_v0(program_root, row_envelope)
    except Exception as exc:
        raise MLBGameStateError(
            "cwevent row EventEnvelopeV0 validation failed"
        ) from exc
    if envelope.event_type != "raw_observation" or envelope.experiment_id is not None:
        raise MLBGameStateError(
            "MLB has no registered experiment; only raw_observation envelopes "
            "with experiment_id null are accepted"
        )
    if (
        envelope.source.system != "retrosheet.chadwick"
        or envelope.source.stream != "cwevent.offline_reconstruction"
    ):
        raise MLBGameStateError("cwevent envelope source is invalid")
    payload = envelope.payload
    expected_payload_keys = {
        "dataset_id",
        "observation_mode",
        "raw_object_sha256",
        "source_manifest_sha256",
        "source_fetched_at",
        "cwevent_output_sha256",
        "source_row_ordinal",
        "source_row_sha256",
        "cwevent_executable",
        "cwevent_version",
        "cwevent_binary_sha256",
        "cwevent_command",
        "cwevent_command_sha256",
        "cwevent_field_map_sha256",
    }
    if set(payload) != expected_payload_keys:
        raise MLBGameStateError("cwevent envelope payload scope is invalid")
    row_ordinal = payload["source_row_ordinal"]
    _require_int(row_ordinal, "source_row_ordinal", minimum=1)
    if (
        envelope.source.record_ordinal != row_ordinal
        or envelope.lineage.raw_record_ordinal != row_ordinal
        or envelope.source.sequence != _cwevent_int(row, "event_index", minimum=1)
    ):
        raise MLBGameStateError("cwevent envelope row ordinal is inconsistent")
    native_game_id = _cwevent_text(row, "game_id")
    assert native_game_id is not None
    expected_game_id = retrosheet_game_id(native_game_id)
    if envelope.canonical_refs.game_id != expected_game_id:
        raise MLBGameStateError("cwevent envelope game_id is inconsistent")
    if set(envelope.canonical_refs.participant_ids) != {
        f"participant_{away_team}",
        f"participant_{home_team}",
    }:
        raise MLBGameStateError("cwevent envelope participants are inconsistent")
    expected_native = (
        "retrosheet.cwevent_event",
        f"{native_game_id}:{_cwevent_int(row, 'event_index', minimum=1)}",
    )
    observed_native = tuple(
        (reference.namespace, reference.native_id)
        for reference in envelope.native_refs
    )
    if observed_native != (expected_native,):
        raise MLBGameStateError("cwevent envelope native reference is inconsistent")
    if (
        payload["dataset_id"] != RETROSHEET_DATASET_ID
        or payload["observation_mode"] != "offline_reconstruction"
        or payload["source_row_sha256"] != cwevent_row_sha256(row)
        or payload["cwevent_output_sha256"] != envelope.lineage.raw_object_hash
        or payload["source_fetched_at"] != envelope.time.receive_at
        or payload["cwevent_executable"] != runtime.executable
        or payload["cwevent_version"] != runtime.version
        or payload["cwevent_binary_sha256"] != runtime.binary_sha256
        or payload["cwevent_field_map_sha256"] != CWEVENT_FIELD_MAP_SHA256
    ):
        raise MLBGameStateError("cwevent envelope hash/runtime binding is invalid")
    command = tuple(payload["cwevent_command"])
    command = _validate_cwevent_command(
        command,
        runtime=runtime,
        native_game_id=native_game_id,
    )
    if payload["cwevent_command_sha256"] != canonical_sha256(list(command)):
        raise MLBGameStateError("cwevent envelope command hash is invalid")
    return envelope


def _state_provenance_from_envelope(
    envelope: EventEnvelopeV0,
) -> MLBOfflineStateProvenance:
    payload = envelope.payload
    return MLBOfflineStateProvenance(
        observation_mode="offline_reconstruction",
        source_dataset_id=payload["dataset_id"],
        raw_object_sha256=payload["raw_object_sha256"],
        source_manifest_sha256=payload["source_manifest_sha256"],
        source_fetched_at=payload["source_fetched_at"],
        cwevent_output_sha256=payload["cwevent_output_sha256"],
        source_envelope_id=envelope.event_id,
        source_row_ordinal=payload["source_row_ordinal"],
        source_row_sha256=payload["source_row_sha256"],
        reconstruction_cutoff_basis="cwevent_output_row_ordinal",
        reconstruction_cutoff_ordinal=payload["source_row_ordinal"],
        cwevent_executable=payload["cwevent_executable"],
        cwevent_version=payload["cwevent_version"],
        cwevent_binary_sha256=payload["cwevent_binary_sha256"],
        cwevent_command_sha256=payload["cwevent_command_sha256"],
        cwevent_field_map_sha256=payload["cwevent_field_map_sha256"],
    )


def state_from_cwevent_row(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
    row_envelope: EventEnvelopeV0,
    program_root: str | Path,
    cwevent_executable: str = "cwevent",
    sequence: int = 0,
) -> MLBGameState:
    """Build the exact state observed immediately before one cwevent row."""

    runtime = require_cwevent_runtime(cwevent_executable)
    envelope = _validated_cwevent_row_envelope(
        program_root=program_root,
        row=row,
        row_envelope=row_envelope,
        runtime=runtime,
        away_team=away_team,
        home_team=home_team,
    )
    _require_int(sequence, "state sequence", minimum=0)
    snapshot = _cwevent_snapshot(
        row,
        away_team=away_team,
        home_team=home_team,
    )
    return MLBGameState(
        sport=MLB_SPORT,
        sequence=sequence,
        away_team=away_team,
        home_team=home_team,
        terminal=False,
        observation_mode="offline_reconstruction",
        source_provenance=_state_provenance_from_envelope(envelope),
        source_envelope_evidence=_row_envelope_evidence(envelope),
        **snapshot,
    )


def _canonical_destination(value: int) -> int:
    if not 0 <= value <= 6:
        raise MLBGameStateError("cwevent runner destination must be in 0..6")
    return 4 if value >= 4 else value


def event_from_cwevent_rows(
    state: MLBGameState,
    play_row: Mapping[str, object],
    next_row: Mapping[str, object] | None,
    *,
    play_envelope: EventEnvelopeV0,
    next_envelope: EventEnvelopeV0 | None,
    program_root: str | Path,
    cwevent_executable: str = "cwevent",
) -> MLBPlayEvent:
    """Adapt governed cwevent rows as an explicit offline reconstruction."""

    if not isinstance(state, MLBGameState):
        raise MLBGameStateError("cwevent adapter requires MLBGameState")
    if (
        state.observation_mode != "offline_reconstruction"
        or state.source_provenance is None
    ):
        raise MLBGameStateError(
            "cwevent adapter requires an offline state from a validated row envelope"
        )
    runtime = require_cwevent_runtime(cwevent_executable)
    validated_play_envelope = _validated_cwevent_row_envelope(
        program_root=program_root,
        row=play_row,
        row_envelope=play_envelope,
        runtime=runtime,
        away_team=state.away_team,
        home_team=state.home_team,
    )
    expected_state_provenance = _state_provenance_from_envelope(
        validated_play_envelope
    )
    if state.source_provenance != expected_state_provenance:
        raise MLBGameStateError(
            "cwevent play envelope does not match the supplied state provenance"
        )
    if (next_row is None) != (next_envelope is None):
        raise MLBGameStateError(
            "next cwevent row and its EventEnvelopeV0 must be present together"
        )
    validated_next_envelope = (
        None
        if next_row is None or next_envelope is None
        else _validated_cwevent_row_envelope(
            program_root=program_root,
            row=next_row,
            row_envelope=next_envelope,
            runtime=runtime,
            away_team=state.away_team,
            home_team=state.home_team,
        )
    )
    play_payload = validated_play_envelope.payload
    next_payload = (
        None if validated_next_envelope is None else validated_next_envelope.payload
    )
    if next_payload is not None:
        for common_field in (
            "dataset_id",
            "raw_object_sha256",
            "source_manifest_sha256",
            "source_fetched_at",
            "cwevent_output_sha256",
            "cwevent_executable",
            "cwevent_version",
            "cwevent_binary_sha256",
            "cwevent_command_sha256",
            "cwevent_field_map_sha256",
        ):
            if next_payload[common_field] != play_payload[common_field]:
                raise MLBGameStateError(
                    "play and post-event envelopes have inconsistent provenance"
                )
        if next_payload["source_row_ordinal"] != (
            play_payload["source_row_ordinal"] + 1
        ):
            raise MLBGameStateError(
                "post-event envelope must be the immediate next output row"
            )
    pre = _cwevent_snapshot(
        play_row,
        away_team=state.away_team,
        home_team=state.home_team,
    )
    _require_snapshot_matches_state(pre, state)

    advances: list[RunnerAdvance] = []
    if _cwevent_flag(play_row, "batter_event"):
        advances.append(
            RunnerAdvance(
                runner_id=state.batter_id,
                start_base=0,
                destination=_canonical_destination(
                    _cwevent_int(
                        play_row,
                        "batter_destination",
                        minimum=0,
                        maximum=6,
                    )
                ),
            )
        )
    elif _cwevent_int(
        play_row,
        "batter_destination",
        minimum=0,
        maximum=6,
    ) != 0:
        raise MLBGameStateError(
            "cwevent non-batter event cannot advance the batter"
        )

    runner_destination_fields = (
        "runner_on_first_destination",
        "runner_on_second_destination",
        "runner_on_third_destination",
    )
    for start_base, (runner_id, destination_name) in enumerate(
        zip(state.bases, runner_destination_fields, strict=True),
        start=1,
    ):
        destination = _canonical_destination(
            _cwevent_int(
                play_row,
                destination_name,
                minimum=0,
                maximum=6,
            )
        )
        if runner_id is None:
            if destination != 0:
                raise MLBGameStateError(
                    "cwevent cannot advance a runner from an empty base"
                )
            continue
        advances.append(
            RunnerAdvance(
                runner_id=runner_id,
                start_base=start_base,
                destination=destination,
            )
        )

    runs = tuple(
        advance.runner_id for advance in advances if advance.destination == 4
    )
    outs = tuple(
        advance.runner_id for advance in advances if advance.destination == 0
    )
    observed_outs = _cwevent_int(
        play_row,
        "outs_on_play",
        minimum=0,
        maximum=3,
    )
    if len(outs) != observed_outs:
        raise MLBGameStateError(
            "cwevent EVENT_OUTS_CT does not match runner destinations"
        )

    terminal = _cwevent_flag(play_row, "end_game")
    if terminal and next_row is not None:
        raise MLBGameStateError("terminal cwevent row must not have a next row")
    if not terminal and next_row is None:
        raise MLBGameStateError("nonterminal cwevent row requires the next row")
    next_snapshot = (
        None
        if next_row is None
        else _cwevent_snapshot(
            next_row,
            away_team=state.away_team,
            home_team=state.home_team,
        )
    )
    outs_after = state.outs + len(outs)
    transition = (
        InningTransition(
            inning=int(next_snapshot["inning"]),
            half=next_snapshot["half"],  # type: ignore[arg-type]
            batting_team=str(next_snapshot["batting_team"]),
            fielding_team=str(next_snapshot["fielding_team"]),
            batter_id=str(next_snapshot["batter_id"]),
            pitcher_id=str(next_snapshot["pitcher_id"]),
            lineup_slot=int(next_snapshot["lineup_slot"]),
            balls=int(next_snapshot["balls"]),
            strikes=int(next_snapshot["strikes"]),
        )
        if next_snapshot is not None and outs_after == 3
        else None
    )
    event_text = _cwevent_text(play_row, "event_text")
    event_code = _cwevent_int(play_row, "event_code", minimum=0)
    source_event_index = _cwevent_int(
        play_row,
        "event_index",
        minimum=1,
    )
    event_provenance = MLBOfflineEventProvenance(
        observation_mode="offline_reconstruction",
        source_dataset_id=play_payload["dataset_id"],
        raw_object_sha256=play_payload["raw_object_sha256"],
        source_manifest_sha256=play_payload["source_manifest_sha256"],
        source_fetched_at=play_payload["source_fetched_at"],
        cwevent_output_sha256=play_payload["cwevent_output_sha256"],
        play_envelope_id=validated_play_envelope.event_id,
        play_row_ordinal=play_payload["source_row_ordinal"],
        play_row_sha256=play_payload["source_row_sha256"],
        post_event_envelope_id=(
            None
            if validated_next_envelope is None
            else validated_next_envelope.event_id
        ),
        post_event_row_ordinal=(
            None if next_payload is None else next_payload["source_row_ordinal"]
        ),
        post_event_row_sha256=(
            None if next_payload is None else next_payload["source_row_sha256"]
        ),
        reconstruction_cutoff_basis="cwevent_output_row_ordinal",
        reconstruction_cutoff_ordinal=(
            play_payload["source_row_ordinal"]
            if next_payload is None
            else next_payload["source_row_ordinal"]
        ),
        cwevent_executable=play_payload["cwevent_executable"],
        cwevent_version=play_payload["cwevent_version"],
        cwevent_binary_sha256=play_payload["cwevent_binary_sha256"],
        cwevent_command_sha256=play_payload["cwevent_command_sha256"],
        cwevent_field_map_sha256=play_payload["cwevent_field_map_sha256"],
    )
    event = MLBPlayEvent(
        sport=MLB_SPORT,
        game_id=state.game_id,
        sequence=state.sequence + 1,
        event_id=validated_play_envelope.event_id,
        inning=state.inning,
        half=state.half,
        outs_before=state.outs,
        bases_before=state.bases,
        score_before=state.score,
        batting_team=state.batting_team,
        fielding_team=state.fielding_team,
        batter_id=state.batter_id,
        pitcher_id=state.pitcher_id,
        balls_before=state.balls,
        strikes_before=state.strikes,
        lineup_slot_before=state.lineup_slot,
        play_type=f"{event_code}:{event_text}",
        runs=runs,
        outs=outs,
        runner_destinations=tuple(advances),
        next_batter_id=(
            None
            if terminal or transition is not None
            else str(next_snapshot["batter_id"])
        ),
        next_pitcher_id=(
            None
            if terminal or transition is not None
            else str(next_snapshot["pitcher_id"])
        ),
        next_balls=(
            None
            if terminal or transition is not None
            else int(next_snapshot["balls"])
        ),
        next_strikes=(
            None
            if terminal or transition is not None
            else int(next_snapshot["strikes"])
        ),
        next_lineup_slot=(
            None
            if terminal or transition is not None
            else int(next_snapshot["lineup_slot"])
        ),
        inning_transition=transition,
        terminal=terminal,
        observation_mode="offline_reconstruction",
        source_provenance=event_provenance,
        play_envelope_evidence=_row_envelope_evidence(
            validated_play_envelope
        ),
        post_event_envelope_evidence=(
            None
            if validated_next_envelope is None
            else _row_envelope_evidence(validated_next_envelope)
        ),
        source_parser="chadwick.cwevent",
        source_parser_version=runtime.version,
        source_event_index=source_event_index,
    )
    reduced = reduce_mlb_state(state, event)
    if next_snapshot is not None:
        _require_snapshot_matches_state(next_snapshot, reduced)
    return event


def _post_event_state_provenance(
    event: MLBPlayEvent,
) -> MLBOfflineStateProvenance | None:
    if event.observation_mode == "synthetic_fixture":
        return None
    provenance = event.source_provenance
    if provenance is None:
        raise MLBGameStateError("offline event lacks source provenance")
    uses_post_row = provenance.post_event_row_ordinal is not None
    source_envelope_id = (
        provenance.post_event_envelope_id
        if uses_post_row
        else provenance.play_envelope_id
    )
    source_row_ordinal = (
        provenance.post_event_row_ordinal
        if uses_post_row
        else provenance.play_row_ordinal
    )
    source_row_sha256 = (
        provenance.post_event_row_sha256
        if uses_post_row
        else provenance.play_row_sha256
    )
    assert source_envelope_id is not None
    assert source_row_ordinal is not None
    assert source_row_sha256 is not None
    return MLBOfflineStateProvenance(
        observation_mode="offline_reconstruction",
        source_dataset_id=provenance.source_dataset_id,
        raw_object_sha256=provenance.raw_object_sha256,
        source_manifest_sha256=provenance.source_manifest_sha256,
        source_fetched_at=provenance.source_fetched_at,
        cwevent_output_sha256=provenance.cwevent_output_sha256,
        source_envelope_id=source_envelope_id,
        source_row_ordinal=source_row_ordinal,
        source_row_sha256=source_row_sha256,
        reconstruction_cutoff_basis="cwevent_output_row_ordinal",
        reconstruction_cutoff_ordinal=source_row_ordinal,
        cwevent_executable=provenance.cwevent_executable,
        cwevent_version=provenance.cwevent_version,
        cwevent_binary_sha256=provenance.cwevent_binary_sha256,
        cwevent_command_sha256=provenance.cwevent_command_sha256,
        cwevent_field_map_sha256=provenance.cwevent_field_map_sha256,
    )


def _post_event_state_envelope_evidence(
    event: MLBPlayEvent,
) -> MLBRowEnvelopeEvidence | None:
    if event.observation_mode == "synthetic_fixture":
        return None
    return (
        event.post_event_envelope_evidence
        if event.post_event_envelope_evidence is not None
        else event.play_envelope_evidence
    )


def reduce_mlb_state(state: MLBGameState, event: MLBPlayEvent) -> MLBGameState:
    """Apply one already-parsed play observation without mutating prior state."""

    if not isinstance(state, MLBGameState) or not isinstance(event, MLBPlayEvent):
        raise MLBGameStateError("reducer requires MLBGameState and MLBPlayEvent")
    if state.terminal:
        raise MLBGameStateError("terminal state cannot accept another event")
    if state.sport != MLB_SPORT or event.sport != MLB_SPORT:
        raise MLBGameStateError("sport must be mlb")
    if event.observation_mode != state.observation_mode:
        raise MLBGameStateError("event and state observation modes must match")
    if state.observation_mode == "offline_reconstruction":
        state_provenance = state.source_provenance
        event_provenance = event.source_provenance
        if state_provenance is None or event_provenance is None:
            raise MLBGameStateError(
                "offline event provenance does not continue the state cutoff"
            )
        state_stream_identity = (
            state_provenance.source_dataset_id,
            state_provenance.raw_object_sha256,
            state_provenance.source_manifest_sha256,
            state_provenance.source_fetched_at,
            state_provenance.cwevent_output_sha256,
            state_provenance.cwevent_executable,
            state_provenance.cwevent_version,
            state_provenance.cwevent_binary_sha256,
            state_provenance.cwevent_command_sha256,
            state_provenance.cwevent_field_map_sha256,
            state_provenance.source_envelope_id,
            state_provenance.source_row_ordinal,
            state_provenance.source_row_sha256,
            state_provenance.reconstruction_cutoff_basis,
            state_provenance.reconstruction_cutoff_ordinal,
            state.source_envelope_evidence,
        )
        event_play_identity = (
            event_provenance.source_dataset_id,
            event_provenance.raw_object_sha256,
            event_provenance.source_manifest_sha256,
            event_provenance.source_fetched_at,
            event_provenance.cwevent_output_sha256,
            event_provenance.cwevent_executable,
            event_provenance.cwevent_version,
            event_provenance.cwevent_binary_sha256,
            event_provenance.cwevent_command_sha256,
            event_provenance.cwevent_field_map_sha256,
            event_provenance.play_envelope_id,
            event_provenance.play_row_ordinal,
            event_provenance.play_row_sha256,
            event_provenance.reconstruction_cutoff_basis,
            event_provenance.play_row_ordinal,
            event.play_envelope_evidence,
        )
        if state_stream_identity != event_play_identity:
            raise MLBGameStateError(
                "offline event provenance identity does not continue the "
                "complete immutable state stream/runtime identity"
            )
    if event.game_id != state.game_id:
        raise MLBGameStateError("event game_id does not match state game_id")
    if event.sequence != state.sequence + 1:
        raise MLBGameStateError("event sequence must be contiguous")
    if event.inning != state.inning:
        raise MLBGameStateError("event inning does not match state inning")
    if event.half != state.half:
        raise MLBGameStateError("event half does not match state half")
    if event.outs_before != state.outs:
        raise MLBGameStateError("event outs_before does not match state outs")
    if event.bases_before != state.bases:
        raise MLBGameStateError("event bases_before does not match state bases")
    if event.score_before != state.score:
        raise MLBGameStateError("event score_before does not match state score")
    if event.batting_team != state.batting_team:
        raise MLBGameStateError("event batting_team does not match state")
    if event.fielding_team != state.fielding_team:
        raise MLBGameStateError("event fielding_team does not match state")
    if event.batter_id != state.batter_id:
        raise MLBGameStateError("event batter_id does not match state batter")
    if event.pitcher_id != state.pitcher_id:
        raise MLBGameStateError("event pitcher_id does not match state pitcher")
    if event.balls_before != state.balls:
        raise MLBGameStateError("event balls_before does not match state balls")
    if event.strikes_before != state.strikes:
        raise MLBGameStateError("event strikes_before does not match state strikes")
    if event.lineup_slot_before != state.lineup_slot:
        raise MLBGameStateError("event lineup slot does not match state lineup")

    start_bases = [advance.start_base for advance in event.runner_destinations]
    if len(set(start_bases)) != len(start_bases):
        raise MLBGameStateError("runner origins must be unique")
    occupied_destinations = [
        advance.destination
        for advance in event.runner_destinations
        if 1 <= advance.destination <= 3
    ]
    if len(set(occupied_destinations)) != len(occupied_destinations):
        raise MLBGameStateError("runner base destinations must be unique")
    moving_bases = {base for base in start_bases if base}
    for advance in event.runner_destinations:
        if advance.start_base == 0:
            if advance.runner_id != state.batter_id:
                raise MLBGameStateError("batter runner origin does not match state")
        elif state.bases[advance.start_base - 1] != advance.runner_id:
            raise MLBGameStateError("base runner origin does not match state")
        if (
            1 <= advance.destination <= 3
            and state.bases[advance.destination - 1] is not None
            and advance.destination not in moving_bases
        ):
            raise MLBGameStateError("runner destination is already occupied")

    outs_after = state.outs + len(event.outs)
    if outs_after > 3:
        raise MLBGameStateError("a play cannot produce more than three outs")
    if event.inning_transition is not None and outs_after != 3:
        raise MLBGameStateError("inning transition requires the third out")
    if outs_after == 3 and event.inning_transition is None and not event.terminal:
        raise MLBGameStateError("third out requires an inning transition or terminal")
    if event.inning_transition is not None and event.terminal:
        raise MLBGameStateError("terminal event cannot also transition innings")
    if event.inning_transition is not None:
        transition = event.inning_transition
        expected_transition = (
            (
                state.inning,
                "bottom",
                state.home_team,
                state.away_team,
            )
            if state.half == "top"
            else (
                state.inning + 1,
                "top",
                state.away_team,
                state.home_team,
            )
        )
        observed_transition = (
            transition.inning,
            transition.half,
            transition.batting_team,
            transition.fielding_team,
        )
        if observed_transition != expected_transition:
            raise MLBGameStateError(
                "inning transition must advance to the immediate next half"
            )

    bases = list(state.bases)
    for advance in event.runner_destinations:
        if advance.start_base:
            bases[advance.start_base - 1] = None
    for advance in event.runner_destinations:
        if 1 <= advance.destination <= 3:
            bases[advance.destination - 1] = advance.runner_id
    score = (
        MLBScore(away=state.score.away + len(event.runs), home=state.score.home)
        if state.half == "top"
        else MLBScore(away=state.score.away, home=state.score.home + len(event.runs))
    )
    next_state_provenance = _post_event_state_provenance(event)
    next_state_envelope_evidence = _post_event_state_envelope_evidence(event)
    if event.terminal:
        if state.inning < 9:
            raise MLBGameStateError("terminal event cannot occur before inning nine")
        if score.away == score.home:
            raise MLBGameStateError("terminal event cannot leave a tied score")
        if state.half == "top" and (
            outs_after != 3 or score.home <= score.away
        ):
            raise MLBGameStateError(
                "top-half terminal event requires third out with home leading"
            )
        if (
            state.half == "bottom"
            and outs_after < 3
            and score.home <= score.away
        ):
            raise MLBGameStateError(
                "bottom-half terminal event before three outs must be a walkoff"
            )
        return replace(
            state,
            sequence=event.sequence,
            outs=state.outs + len(event.outs),
            bases=(bases[0], bases[1], bases[2]),
            score=score,
            terminal=True,
            source_provenance=next_state_provenance,
            source_envelope_evidence=next_state_envelope_evidence,
        )
    if event.inning_transition is not None:
        transition = event.inning_transition
        return replace(
            state,
            sequence=event.sequence,
            inning=transition.inning,
            half=transition.half,
            outs=0,
            bases=(None, None, None),
            score=score,
            batting_team=transition.batting_team,
            fielding_team=transition.fielding_team,
            batter_id=transition.batter_id,
            pitcher_id=transition.pitcher_id,
            balls=transition.balls,
            strikes=transition.strikes,
            lineup_slot=transition.lineup_slot,
            source_provenance=next_state_provenance,
            source_envelope_evidence=next_state_envelope_evidence,
        )
    return replace(
        state,
        sequence=event.sequence,
        outs=state.outs + len(event.outs),
        bases=(bases[0], bases[1], bases[2]),
        score=score,
        batter_id=event.next_batter_id,
        pitcher_id=event.next_pitcher_id,
        balls=event.next_balls,
        strikes=event.next_strikes,
        lineup_slot=event.next_lineup_slot,
        source_provenance=next_state_provenance,
        source_envelope_evidence=next_state_envelope_evidence,
    )


reduce = reduce_mlb_state


@dataclass(frozen=True, slots=True)
class MLBReplayEvidence:
    """Auditable result of one frozen, deterministic offline replay."""

    raw_object_sha256: str
    source_manifest_sha256: str
    native_game_id: str
    canonical_game_id: str
    cwevent_runtime: CweventRuntime
    cwevent_command: tuple[str, ...]
    cwevent_command_sha256: str
    cwevent_field_map_sha256: str
    cwevent_output_sha256: str
    source_row_sha256s: tuple[str, ...]
    event_envelope_ids: tuple[str, ...]
    transition_trace_sha256s: tuple[str, ...]
    state_step_sha256s: tuple[str, ...]
    events: int
    adapter_vs_next_observation_comparisons: int
    field_mismatches: int
    final_state_sha256: str
    replay_sha256: str
    initial_state: MLBGameState
    play_events: tuple[MLBPlayEvent, ...]
    final_state: MLBGameState


def _safe_extract_retrosheet_archive(
    object_bytes: bytes,
    destination: Path,
) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(object_bytes)) as archive:
            for member in archive.infolist():
                pure = PurePosixPath(member.filename)
                if (
                    pure.is_absolute()
                    or not pure.parts
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or "\\" in member.filename
                ):
                    raise MLBGameStateError(
                        "Retrosheet archive contains an unsafe member path"
                    )
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise MLBGameStateError("Retrosheet archive cannot be extracted") from exc


def _run_cwevent(
    command: tuple[str, ...],
    *,
    cwd: Path,
) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MLBGameStateError("Chadwick cwevent execution failed") from exc
    if completed.returncode != 0:
        raise MLBGameStateError("Chadwick cwevent returned a nonzero status")
    if completed.stderr:
        raise MLBGameStateError("quiet Chadwick cwevent emitted stderr")
    if not completed.stdout:
        raise MLBGameStateError("Chadwick cwevent emitted no rows")
    return completed.stdout


def _decode_cwevent_rows(output: bytes) -> tuple[dict[str, object], ...]:
    try:
        text = output.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != CWEVENT_FIELD_NAMES:
            raise MLBGameStateError("cwevent output header differs from frozen map")
        rows: list[dict[str, object]] = []
        for row in reader:
            if None in row or set(row) != set(CWEVENT_FIELD_NAMES):
                raise MLBGameStateError("cwevent output row has an invalid width")
            rows.append(dict(row))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise MLBGameStateError("cwevent output is not canonical CSV") from exc
    if not rows:
        raise MLBGameStateError("cwevent output contains no data rows")
    for ordinal, row in enumerate(rows, start=1):
        if _cwevent_int(row, "event_index", minimum=1) != ordinal:
            raise MLBGameStateError("cwevent EVENT_ID sequence is not contiguous")
    return tuple(rows)


def run_frozen_retrosheet_2025_game_replay(
    *,
    program_root: str | Path,
    cwevent_executable: str = "cwevent",
) -> MLBReplayEvidence:
    """Replay the one frozen MLB engineering game with byte-exact lineage."""

    root = Path(program_root)
    store_root = root / "var/raw"
    manifest_path = store_root / RETROSHEET_2025_MANIFEST_RELATIVE_PATH
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=root,
        )
    except Exception as exc:
        raise MLBGameStateError(
            "frozen Retrosheet object or manifest verification failed"
        ) from exc
    manifest = verified.record.manifest
    if (
        manifest.dataset_id != RETROSHEET_DATASET_ID
        or manifest.object_sha256 != RETROSHEET_2025_RAW_OBJECT_SHA256
        or manifest.manifest_sha256 != RETROSHEET_2025_MANIFEST_SHA256
        or manifest.license_status != "research_only"
    ):
        raise MLBGameStateError("Retrosheet frozen source binding is invalid")
    runtime = require_cwevent_runtime(cwevent_executable)
    command = cwevent_command(
        runtime,
        native_game_id=RETROSHEET_FROZEN_GAME_ID,
        event_file_name=RETROSHEET_FROZEN_EVENT_FILE,
    )
    with tempfile.TemporaryDirectory(prefix="saf-mlb-replay-") as temporary:
        workdir = Path(temporary)
        _safe_extract_retrosheet_archive(verified.object_bytes, workdir)
        if not (workdir / RETROSHEET_FROZEN_EVENT_FILE).is_file():
            raise MLBGameStateError("frozen Retrosheet event file is missing")
        cwevent_output = _run_cwevent(command, cwd=workdir)
    cwevent_output_sha256 = (
        "sha256:" + hashlib.sha256(cwevent_output).hexdigest()
    )
    rows = _decode_cwevent_rows(cwevent_output)
    if any(
        _cwevent_text(row, "game_id") != RETROSHEET_FROZEN_GAME_ID
        for row in rows
    ):
        raise MLBGameStateError("cwevent output contains a different game")
    envelopes = tuple(
        build_cwevent_row_envelope(
            program_root=root,
            row=row,
            row_ordinal=ordinal,
            raw_object_sha256=manifest.object_sha256,
            source_manifest_sha256=manifest.manifest_sha256,
            source_fetched_at=manifest.fetched_at,
            cwevent_output_sha256=cwevent_output_sha256,
            cwevent_command=command,
            cwevent_executable=runtime.executable,
            away_team=RETROSHEET_FROZEN_AWAY_TEAM,
            home_team=RETROSHEET_FROZEN_HOME_TEAM,
        )
        for ordinal, row in enumerate(rows, start=1)
    )
    initial_replay_state = state_from_cwevent_row(
        rows[0],
        away_team=RETROSHEET_FROZEN_AWAY_TEAM,
        home_team=RETROSHEET_FROZEN_HOME_TEAM,
        row_envelope=envelopes[0],
        program_root=root,
        cwevent_executable=runtime.executable,
    )
    state = initial_replay_state
    play_events: list[MLBPlayEvent] = []
    transition_trace_sha256s: list[str] = []
    state_step_sha256s: list[str] = []
    for position, (row, envelope) in enumerate(
        zip(rows, envelopes, strict=True)
    ):
        next_row = rows[position + 1] if position + 1 < len(rows) else None
        next_envelope = (
            envelopes[position + 1]
            if position + 1 < len(envelopes)
            else None
        )
        event = event_from_cwevent_rows(
            state,
            row,
            next_row,
            play_envelope=envelope,
            next_envelope=next_envelope,
            program_root=root,
            cwevent_executable=runtime.executable,
        )
        play_events.append(event)
        trace = advance_state(MLB_GAME_STATE_REDUCER, state, event)
        step = trace.to_contract(
            state_schema_id="urn:saf:game-state:mlb:v0",
            event_schema_id="urn:saf:game-event:mlb:v0",
            observation_mode="offline_reconstruction",
            quality_flags=(),
        )
        transition_trace_sha256s.append(trace.trace_sha256)
        state_step_sha256s.append(step.step_sha256)
        state = trace.next_state
    if not state.terminal:
        raise MLBGameStateError("frozen MLB replay did not reach a terminal state")
    command_sha256 = canonical_sha256(list(command))
    output_sha256 = cwevent_output_sha256
    row_sha256s = tuple(cwevent_row_sha256(row) for row in rows)
    final_state_sha256 = canonical_state_sha256(state)
    replay_material = {
        "raw_object_sha256": manifest.object_sha256,
        "source_manifest_sha256": manifest.manifest_sha256,
        "native_game_id": RETROSHEET_FROZEN_GAME_ID,
        "canonical_game_id": state.game_id,
        "cwevent_executable": runtime.executable,
        "cwevent_version": runtime.version,
        "cwevent_binary_sha256": runtime.binary_sha256,
        "cwevent_command_sha256": command_sha256,
        "cwevent_field_map_sha256": CWEVENT_FIELD_MAP_SHA256,
        "cwevent_output_sha256": output_sha256,
        "source_row_sha256s": row_sha256s,
        "event_envelope_ids": tuple(envelope.event_id for envelope in envelopes),
        "transition_trace_sha256s": tuple(transition_trace_sha256s),
        "state_step_sha256s": tuple(state_step_sha256s),
        "final_state_sha256": final_state_sha256,
    }
    return MLBReplayEvidence(
        raw_object_sha256=manifest.object_sha256,
        source_manifest_sha256=manifest.manifest_sha256,
        native_game_id=RETROSHEET_FROZEN_GAME_ID,
        canonical_game_id=state.game_id,
        cwevent_runtime=runtime,
        cwevent_command=command,
        cwevent_command_sha256=command_sha256,
        cwevent_field_map_sha256=CWEVENT_FIELD_MAP_SHA256,
        cwevent_output_sha256=output_sha256,
        source_row_sha256s=row_sha256s,
        event_envelope_ids=tuple(envelope.event_id for envelope in envelopes),
        transition_trace_sha256s=tuple(transition_trace_sha256s),
        state_step_sha256s=tuple(state_step_sha256s),
        events=len(rows),
        adapter_vs_next_observation_comparisons=len(rows) - 1,
        field_mismatches=0,
        final_state_sha256=final_state_sha256,
        replay_sha256=canonical_sha256(replay_material),
        initial_state=initial_replay_state,
        play_events=tuple(play_events),
        final_state=state,
    )


@dataclass(frozen=True, slots=True)
class MLBGameStateReducer:
    """Common-protocol adapter for the pure MLB reducer."""

    sport: str = field(default=MLB_SPORT, init=False)
    reducer_id: str = field(
        default="mlb.retrosheet.play-reducer",
        init=False,
    )
    reducer_version: str = field(default="v1", init=False)

    def reduce(
        self,
        state: MLBGameState,
        event: MLBPlayEvent,
    ) -> MLBGameState:
        return reduce_mlb_state(state, event)


MLB_GAME_STATE_REDUCER = MLBGameStateReducer()


__all__ = [
    "CHADWICK_CWEVENT_VERSION",
    "CWEVENT_FIELD_ARGUMENT",
    "CWEVENT_FIELD_MAP",
    "CWEVENT_FIELD_MAP_SHA256",
    "CWEVENT_FIELD_NAMES",
    "CweventRuntime",
    "EventEnvelopeV0",
    "InningTransition",
    "MLB_GAME_STATE_REDUCER",
    "MLB_SPORT",
    "MLBGameState",
    "MLBGameStateError",
    "MLBGameStateReducer",
    "MLBOfflineEventProvenance",
    "MLBOfflineStateProvenance",
    "MLBPlayEvent",
    "MLBReplayEvidence",
    "MLBRowEnvelopeEvidence",
    "MLBScore",
    "RETROSHEET_2025_MANIFEST_RELATIVE_PATH",
    "RETROSHEET_2025_MANIFEST_SHA256",
    "RETROSHEET_2025_RAW_OBJECT_SHA256",
    "RETROSHEET_DATASET_ID",
    "RETROSHEET_FROZEN_AWAY_TEAM",
    "RETROSHEET_FROZEN_EVENT_FILE",
    "RETROSHEET_FROZEN_GAME_ID",
    "RETROSHEET_FROZEN_HOME_TEAM",
    "RunnerAdvance",
    "build_cwevent_row_envelope",
    "cwevent_command",
    "cwevent_row_sha256",
    "event_from_cwevent_rows",
    "initial_state",
    "reduce",
    "reduce_mlb_state",
    "require_cwevent_runtime",
    "require_cwevent_version",
    "retrosheet_game_id",
    "run_frozen_retrosheet_2025_game_replay",
    "state_from_cwevent_row",
]
