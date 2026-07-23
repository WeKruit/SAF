"""Point-in-time NFL baselines and the frozen fastrmodels feature seam."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path
from typing import Literal

from prediction_market.contracts import EventEnvelopeV0, canonical_sha256
from prediction_market.sports.event_envelopes import (
    validate_static_sport_observation_bundle,
)
from prediction_market.sports.game_state import canonical_state_sha256
from prediction_market.sports.nfl_game_state import (
    NFLGameState,
    NFLPlayEvent,
    event_from_nflverse_envelope,
)


FASTRMODELS_COMMIT = "75c7b68bc49535370236c38c9826265da075bd71"
NFLFASTR_HELPER_COMMIT = "ead5e2f9641490f692d923c04835bd3b90275b4e"
FASTRMODELS_NO_SPREAD_MODEL_SHA256 = (
    "sha256:ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c"
)
FASTRMODELS_SPREAD_MODEL_SHA256 = (
    "sha256:a8efe70cf64f459187ef06ebaae08b7a1012661b3446e1336a0a5ece2ba86322"
)

FASTRMODELS_NO_SPREAD_FEATURES = (
    "receive_2h_ko",
    "home",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "Diff_Time_Ratio",
    "score_differential",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
)
FASTRMODELS_SPREAD_FEATURES = (
    "receive_2h_ko",
    "spread_time",
    *FASTRMODELS_NO_SPREAD_FEATURES[1:],
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NFLVERSE_GAME_ID_RE = re.compile(
    r"game_nflverse_([0-9]{4})_([0-9]{2})_"
    r"([A-Za-z0-9]+)_([A-Za-z0-9]+)\Z"
)
_SOURCE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
FASTRMODELS_PIT_STATUSES = (
    "live_pit",
    "offline_reconstruction_not_live_PIT",
    "PIT_UNPROVEN",
    "synthetic_fixture",
)


class NFLModelInputError(ValueError):
    """An NFL state cannot support the frozen official model feature seam."""


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NFLModelInputError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise NFLModelInputError(f"{field_name} must be finite")
    return number


def _utc_datetime(value: object, field_name: str) -> datetime:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise NFLModelInputError(
            f"{field_name} must be a timezone-aware UTC datetime"
        )
    return value


def _canonical_utc(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise NFLModelInputError(f"{field_name} must be a SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class NFLWinProbabilityContext:
    """Atomic external facts and lineage not derivable from reducer state."""

    state_event_envelope: EventEnvelopeV0
    second_half_receiver: str
    second_half_receiver_source_ref: str
    second_half_receiver_observed_at: datetime
    home_spread_line: float | None
    spread_observed_at: datetime | None
    source_object_sha256: str
    pit_status: Literal[
        "live_pit",
        "offline_reconstruction_not_live_PIT",
        "PIT_UNPROVEN",
        "synthetic_fixture",
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.state_event_envelope, EventEnvelopeV0):
            raise NFLModelInputError(
                "state_event_envelope must be an EventEnvelopeV0"
            )
        if (
            type(self.second_half_receiver) is not str
            or not self.second_half_receiver
            or self.second_half_receiver.strip() != self.second_half_receiver
        ):
            raise NFLModelInputError(
                "second_half_receiver must be a canonical team identifier"
            )
        if (
            type(self.second_half_receiver_source_ref) is not str
            or _SOURCE_REF_RE.fullmatch(
                self.second_half_receiver_source_ref
            )
            is None
        ):
            raise NFLModelInputError(
                "second_half_receiver_source_ref must be canonical"
            )
        _utc_datetime(
            self.second_half_receiver_observed_at,
            "second_half_receiver_observed_at",
        )
        if (self.home_spread_line is None) != (
            self.spread_observed_at is None
        ):
            raise NFLModelInputError(
                "home_spread_line and spread_observed_at must be supplied together"
            )
        if self.home_spread_line is not None:
            object.__setattr__(
                self,
                "home_spread_line",
                _finite_number(self.home_spread_line, "home_spread_line"),
            )
            _utc_datetime(self.spread_observed_at, "spread_observed_at")
        _sha256(self.source_object_sha256, "source_object_sha256")
        if self.pit_status not in FASTRMODELS_PIT_STATUSES:
            raise NFLModelInputError(
                "pit_status must explicitly describe point-in-time eligibility"
            )


def _parse_utc_timestamp(value: str, field_name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise NFLModelInputError(f"{field_name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise NFLModelInputError(
            f"{field_name} must be a canonical UTC timestamp"
        ) from exc
    return _utc_datetime(parsed, field_name)


def _envelope_hash_material(
    envelope: EventEnvelopeV0,
) -> dict[str, object]:
    return {
        "content_sha256": canonical_sha256(envelope),
        "event_id": envelope.event_id,
        "payload_sha256": envelope.payload_sha256,
        "source": {
            "system": envelope.source.system,
            "stream": envelope.source.stream,
            "sequence": envelope.source.sequence,
            "record_ordinal": envelope.source.record_ordinal,
        },
        "time": {
            "receive_at": envelope.time.receive_at,
            "receive_basis": envelope.time.receive_basis,
            "source_at": envelope.time.source_at,
        },
        "lineage": {
            "raw_object_hash": envelope.lineage.raw_object_hash,
            "raw_record_ordinal": envelope.lineage.raw_record_ordinal,
            "parent_event_ids": list(envelope.lineage.parent_event_ids),
        },
    }


def _context_material(context: NFLWinProbabilityContext) -> dict[str, object]:
    return {
        "home_spread_line": context.home_spread_line,
        "pit_status": context.pit_status,
        "second_half_receiver": context.second_half_receiver,
        "second_half_receiver_observed_at": _canonical_utc(
            context.second_half_receiver_observed_at
        ),
        "second_half_receiver_source_ref": (
            context.second_half_receiver_source_ref
        ),
        "source_object_sha256": context.source_object_sha256,
        "state_event_envelope": _envelope_hash_material(
            context.state_event_envelope
        ),
        "spread_observed_at": (
            None
            if context.spread_observed_at is None
            else _canonical_utc(context.spread_observed_at)
        ),
    }


def _model_artifact_sha256(variant: str) -> str:
    artifact_sha256 = (
        FASTRMODELS_NO_SPREAD_MODEL_SHA256
        if variant == "no_spread"
        else FASTRMODELS_SPREAD_MODEL_SHA256
        if variant == "spread"
        else None
    )
    if artifact_sha256 is None:
        raise NFLModelInputError("unknown fastrmodels variant")
    return _sha256(artifact_sha256, "model_artifact_sha256")


def _validate_post_state_projection(
    state: NFLGameState,
    event: NFLPlayEvent,
    envelope: EventEnvelopeV0,
) -> None:
    canonical_game_id = envelope.canonical_refs.game_id
    game_match = (
        None
        if canonical_game_id is None
        else _NFLVERSE_GAME_ID_RE.fullmatch(canonical_game_id)
    )
    if game_match is None:
        raise NFLModelInputError(
            "state event envelope game_id must expose nflverse team roles"
        )
    expected_away_team = game_match.group(3)
    expected_home_team = game_match.group(4)
    if state.away_team != expected_away_team:
        raise NFLModelInputError(
            "state away_team does not match the raw-bound nflverse game role"
        )
    if state.home_team != expected_home_team:
        raise NFLModelInputError(
            "state home_team does not match the raw-bound nflverse game role"
        )

    expected_participant_ids = tuple(
        sorted(
            (
                f"participant_nflverse_{state.away_team}",
                f"participant_nflverse_{state.home_team}",
            )
        )
    )
    if envelope.canonical_refs.participant_ids != expected_participant_ids:
        raise NFLModelInputError(
            "state event envelope canonical_refs.participant_ids do not "
            "match state teams"
        )
    if (
        state.suspended
        or event.lifecycle_action != "none"
        or event.clock_carry_forward
        or event.carry_forward_context
        or event.timeout_kind == "administrative"
    ):
        raise NFLModelInputError(
            "lifecycle, administrative, context-carry, or suspended state "
            "is not an eligible independently observed pre-snap model cutoff"
        )

    expected = {
        "sport": event.sport,
        "game_id": event.game_id,
        "sequence": event.sequence,
        "terminal": event.terminal,
        "season_type": event.season_type,
        "period": event.period,
        "period_seconds_remaining": event.period_seconds_remaining,
        "game_seconds_remaining": event.game_seconds_remaining,
        "source_play_id": event.next_source_play_id,
        "source_order_sequence": event.next_source_order_sequence,
        "context_source_play_id": event.context_source_play_id,
        "context_source_order_sequence": (
            event.context_source_order_sequence
        ),
        "suspended": event.lifecycle_action == "suspend",
        "drive_id": event.next_drive_id,
        "play_clock_seconds": event.next_play_clock_seconds,
        "possession_team": event.possession_team,
        "down": event.down,
        "distance": event.distance,
        "yardline_100": event.yardline_100,
        "goal_to_go": event.goal_to_go,
        "home_score": event.home_score,
        "away_score": event.away_score,
        "home_timeouts_remaining": event.home_timeouts_remaining,
        "away_timeouts_remaining": event.away_timeouts_remaining,
        "last_event_id": event.event_id,
    }
    for field_name, expected_value in expected.items():
        if getattr(state, field_name) != expected_value:
            raise NFLModelInputError(
                "state event post-state projection mismatch at "
                f"{field_name}"
            )


def _feature_digest(
    *,
    variant: str,
    names: tuple[str, ...],
    values: tuple[float, ...],
    source_state_sha256: str,
    context: NFLWinProbabilityContext,
    state_cutoff_at: datetime,
    state_event_raw_parents: tuple[EventEnvelopeV0, ...],
    model_artifact_sha256: str,
) -> str:
    material = {
        "context": _context_material(context),
        "feature_definition": "fastrmodels_regulation_wp_v1",
        "fastrmodels_commit": FASTRMODELS_COMMIT,
        "model_artifact_sha256": model_artifact_sha256,
        "nflfastR_helper_commit": NFLFASTR_HELPER_COMMIT,
        "names": list(names),
        "state_cutoff_at": _canonical_utc(state_cutoff_at),
        "state_cutoff_basis": "nflverse_native_event_time",
        "state_event_raw_parents": [
            _envelope_hash_material(parent)
            for parent in state_event_raw_parents
        ],
        "source_state_sha256": source_state_sha256,
        "values": list(values),
        "variant": variant,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class FastrmodelsFeatureVector:
    """One exact ordered vector consumed by an official regulation booster."""

    variant: Literal["no_spread", "spread"]
    names: tuple[str, ...]
    values: tuple[float, ...]
    source_state_sha256: str
    context: NFLWinProbabilityContext
    state_cutoff_at: datetime
    state_event_raw_parents: tuple[EventEnvelopeV0, ...]
    model_artifact_sha256: str = field(init=False)
    feature_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected_names = (
            FASTRMODELS_NO_SPREAD_FEATURES
            if self.variant == "no_spread"
            else FASTRMODELS_SPREAD_FEATURES
            if self.variant == "spread"
            else None
        )
        if expected_names is None:
            raise NFLModelInputError("unknown fastrmodels variant")
        if self.names != expected_names:
            raise NFLModelInputError(
                "feature names or order differ from the frozen official model"
            )
        if type(self.values) is not tuple or len(self.values) != len(self.names):
            raise NFLModelInputError(
                "feature values must match the frozen feature order"
            )
        values = tuple(
            _finite_number(value, f"values[{index}]")
            for index, value in enumerate(self.values)
        )
        object.__setattr__(self, "values", values)
        if (
            type(self.source_state_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_state_sha256) is None
        ):
            raise NFLModelInputError(
                "source_state_sha256 must be a SHA-256 digest"
            )
        if not isinstance(self.context, NFLWinProbabilityContext):
            raise NFLModelInputError(
                "context must be an NFLWinProbabilityContext"
            )
        state_cutoff_at = _utc_datetime(
            self.state_cutoff_at,
            "state_cutoff_at",
        )
        source_at = self.context.state_event_envelope.time.source_at
        if source_at is None or _parse_utc_timestamp(
            source_at,
            "state_event_envelope.time.source_at",
        ) != state_cutoff_at:
            raise NFLModelInputError(
                "state_cutoff_at must equal state_event_envelope.time.source_at"
            )
        if (
            type(self.state_event_raw_parents) is not tuple
            or len(self.state_event_raw_parents) < 2
            or any(
                not isinstance(parent, EventEnvelopeV0)
                for parent in self.state_event_raw_parents
            )
        ):
            raise NFLModelInputError(
                "state_event_raw_parents must contain the complete "
                "EventEnvelopeV0 source window"
            )
        if self.variant == "spread" and self.context.home_spread_line is None:
            raise NFLModelInputError(
                "spread variant requires a provenance-bound home spread line"
            )
        model_artifact_sha256 = _model_artifact_sha256(self.variant)
        object.__setattr__(
            self,
            "model_artifact_sha256",
            model_artifact_sha256,
        )
        object.__setattr__(
            self,
            "feature_sha256",
            _feature_digest(
                variant=self.variant,
                names=self.names,
                values=values,
                source_state_sha256=self.source_state_sha256,
                context=self.context,
                state_cutoff_at=state_cutoff_at,
                state_event_raw_parents=self.state_event_raw_parents,
                model_artifact_sha256=model_artifact_sha256,
            ),
        )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


def fastrmodels_feature_vector(
    state: NFLGameState,
    *,
    context: NFLWinProbabilityContext,
    variant: Literal["no_spread", "spread"],
    program_root: str | Path,
    state_event_raw_parents: tuple[EventEnvelopeV0, ...],
) -> FastrmodelsFeatureVector:
    """Project the provenance-bound observed state into official features."""

    if not isinstance(state, NFLGameState):
        raise NFLModelInputError("state must be an NFLGameState")
    if not isinstance(context, NFLWinProbabilityContext):
        raise NFLModelInputError(
            "context must be an NFLWinProbabilityContext"
        )
    try:
        validated_envelope = validate_static_sport_observation_bundle(
            program_root,
            context.state_event_envelope,
            raw_parents=state_event_raw_parents,
            expected_experiment_id="X-11",
            expected_dataset_id="DS-NFLVERSE",
            expected_source_system="nflverse",
            expected_source_stream="play_by_play",
            expected_native_namespace="nflverse.play",
        )
        state_event = event_from_nflverse_envelope(
            validated_envelope,
            program_root=program_root,
            raw_parents=state_event_raw_parents,
        )
    except (TypeError, ValueError) as exc:
        raise NFLModelInputError(
            f"state event envelope failed strict validation: {exc}"
        ) from exc

    if any(
        parent.time != validated_envelope.time
        for parent in state_event_raw_parents
    ):
        raise NFLModelInputError(
            "state event raw parent time must equal normalized envelope time"
        )
    raw_object_hashes = {
        parent.lineage.raw_object_hash
        for parent in state_event_raw_parents
    }
    if None in raw_object_hashes or len(raw_object_hashes) != 1:
        raise NFLModelInputError(
            "state event raw parents must share a single raw_object_hash"
        )
    raw_object_hash = next(iter(raw_object_hashes))
    if context.source_object_sha256 != raw_object_hash:
        raise NFLModelInputError(
            "context source_object_sha256 does not match the verified "
            "state event raw_object_hash"
        )
    _validate_post_state_projection(
        state,
        state_event,
        validated_envelope,
    )
    if validated_envelope.source.sequence != state.sequence:
        raise NFLModelInputError(
            "state event envelope sequence does not match state sequence"
        )
    if validated_envelope.time.receive_basis != "upstream_exporter":
        raise NFLModelInputError(
            "offline nflverse state event requires upstream_exporter receive basis"
        )
    if context.pit_status == "live_pit":
        raise NFLModelInputError(
            "offline nflverse state event cannot use live_pit pit_status"
        )
    source_at = validated_envelope.time.source_at
    if source_at is None:
        raise NFLModelInputError(
            "state_event_envelope.time.source_at is required as the "
            "offline state cutoff"
        )
    state_cutoff_at = _parse_utc_timestamp(
        source_at,
        "state_event_envelope.time.source_at",
    )
    season_match = _NFLVERSE_GAME_ID_RE.fullmatch(state.game_id)
    if season_match is None:
        raise NFLModelInputError(
            "state game_id must expose the nflverse season"
        )
    season = int(season_match.group(1))
    if state_cutoff_at.year not in {season, season + 1}:
        raise NFLModelInputError(
            "state_event_envelope.time.source_at year must match the "
            "nflverse season or its postseason calendar year"
        )
    if context.second_half_receiver_observed_at > state_cutoff_at:
        raise NFLModelInputError(
            "second_half_receiver_observed_at cannot follow the state cutoff"
        )
    if (
        context.spread_observed_at is not None
        and context.spread_observed_at > state_cutoff_at
    ):
        raise NFLModelInputError(
            "spread_observed_at cannot follow the state cutoff"
        )
    if state.terminal:
        raise NFLModelInputError(
            "terminal state is outside the regulation model"
        )
    if state.period not in {1, 2, 3, 4}:
        raise NFLModelInputError(
            "fastrmodels feature projection is regulation-only"
        )
    if context.second_half_receiver not in {
        state.home_team,
        state.away_team,
    }:
        raise NFLModelInputError(
            "second-half receiver is not a game participant"
        )
    if (
        state.possession_team is None
        or state.down is None
        or state.distance is None
        or state.yardline_100 is None
    ):
        raise NFLModelInputError(
            "official features require possession, down, distance, and yardline"
        )
    if variant not in {"no_spread", "spread"}:
        raise NFLModelInputError("unknown fastrmodels variant")

    possession_is_home = state.possession_team == state.home_team
    score_differential = (
        state.home_score - state.away_score
        if possession_is_home
        else state.away_score - state.home_score
    )
    posteam_timeouts = (
        state.home_timeouts_remaining
        if possession_is_home
        else state.away_timeouts_remaining
    )
    defteam_timeouts = (
        state.away_timeouts_remaining
        if possession_is_home
        else state.home_timeouts_remaining
    )
    half_seconds_remaining = state.period_seconds_remaining + (
        900 if state.period in {1, 3} else 0
    )
    elapsed_share = (3600.0 - state.game_seconds_remaining) / 3600.0
    time_decay = math.exp(-4.0 * elapsed_share)
    receive_second_half_kickoff = float(
        state.period <= 2
        and state.possession_team == context.second_half_receiver
    )
    common = {
        "receive_2h_ko": receive_second_half_kickoff,
        "home": float(possession_is_home),
        "half_seconds_remaining": float(half_seconds_remaining),
        "game_seconds_remaining": float(state.game_seconds_remaining),
        "Diff_Time_Ratio": float(score_differential) / time_decay,
        "score_differential": float(score_differential),
        "down": float(state.down),
        "ydstogo": float(state.distance),
        "yardline_100": float(state.yardline_100),
        "posteam_timeouts_remaining": float(posteam_timeouts),
        "defteam_timeouts_remaining": float(defteam_timeouts),
    }
    names = FASTRMODELS_NO_SPREAD_FEATURES
    values_by_name = common
    if variant == "spread":
        if context.home_spread_line is None:
            raise NFLModelInputError(
                "spread variant requires a provenance-bound home spread line"
            )
        posteam_spread = (
            context.home_spread_line
            if possession_is_home
            else -context.home_spread_line
        )
        values_by_name = {
            "receive_2h_ko": receive_second_half_kickoff,
            "spread_time": posteam_spread * time_decay,
            **{key: value for key, value in common.items() if key != "receive_2h_ko"},
        }
        names = FASTRMODELS_SPREAD_FEATURES
    values = tuple(float(values_by_name[name]) for name in names)
    ordered_raw_parents = tuple(
        sorted(
            state_event_raw_parents,
            key=lambda parent: (
                parent.lineage.raw_record_ordinal,
                parent.event_id,
            ),
        )
    )
    return FastrmodelsFeatureVector(
        variant=variant,
        names=names,
        values=values,
        source_state_sha256=canonical_state_sha256(state),
        context=context,
        state_cutoff_at=state_cutoff_at,
        state_event_raw_parents=ordered_raw_parents,
    )


def nfl_logistic_features(
    *,
    score_differential: int | float,
    seconds_remaining: int | float,
    possession_is_home: bool,
    home_timeouts: int,
    away_timeouts: int,
) -> dict[str, float]:
    values = (
        float(score_differential),
        float(seconds_remaining),
        float(home_timeouts),
        float(away_timeouts),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NFL state features must be finite")
    if values[1] < 0 or home_timeouts < 0 or away_timeouts < 0:
        raise ValueError("remaining time and timeouts must be nonnegative")
    if type(possession_is_home) is not bool:
        raise ValueError("possession_is_home must be boolean")
    return {
        "score_differential": values[0],
        "seconds_remaining": values[1],
        "possession_is_home": 1.0 if possession_is_home else 0.0,
        "home_timeouts": values[2],
        "away_timeouts": values[3],
    }


__all__ = [
    "FASTRMODELS_COMMIT",
    "FASTRMODELS_NO_SPREAD_FEATURES",
    "FASTRMODELS_NO_SPREAD_MODEL_SHA256",
    "FASTRMODELS_PIT_STATUSES",
    "FASTRMODELS_SPREAD_FEATURES",
    "FASTRMODELS_SPREAD_MODEL_SHA256",
    "FastrmodelsFeatureVector",
    "NFLFASTR_HELPER_COMMIT",
    "NFLModelInputError",
    "NFLWinProbabilityContext",
    "fastrmodels_feature_vector",
    "nfl_logistic_features",
]
