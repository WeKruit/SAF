"""PIT-audited Stage A NFL reference values from the frozen official model."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Final, Literal, Protocol

import numpy as np
import pandas as pd

from prediction_market.models.nfl_fastrmodels import (
    FEATURE_NAMES,
    NoSpreadModelInput,
    OfficialModelInputError,
)


EXPERIMENT_ID: Final[str] = "X-15"
CLAIM_BOUNDARY: Final[str] = (
    "RETROSPECTIVE_DIAGNOSTIC; frozen official no-spread model; "
    "not live-PIT latency, causality, execution, or alpha evidence"
)

ReferenceStatus = Literal[
    "SUPPORTED",
    "EVENT_INELIGIBLE",
    "MODEL_SUPPORT_UNPROVEN",
    "MISSING_PRE_STATE",
    "MISSING_POST_STATE",
    "MISSING_AS_OF_RECEIVER",
    "COMPOSITE_TRANSITION",
    "FUTURE_FEATURE_REJECTED",
    "ORDER_AMBIGUOUS",
]

_REFERENCE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "SUPPORTED",
        "EVENT_INELIGIBLE",
        "MODEL_SUPPORT_UNPROVEN",
        "MISSING_PRE_STATE",
        "MISSING_POST_STATE",
        "MISSING_AS_OF_RECEIVER",
        "COMPOSITE_TRANSITION",
        "FUTURE_FEATURE_REJECTED",
        "ORDER_AMBIGUOUS",
    }
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED_EVENT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "game_id",
        "event_id",
        "raw_play_id",
        "atomic_information_episode_id",
        "information_status",
        "stage_b_information_event_eligible",
        "final_sports_outcome_eligible",
        "source_interval_start",
        "source_interval_end",
        "known_at",
        "order_sequence",
        "quarter",
        "game_seconds_remaining",
        "home_team",
        "away_team",
        "actor_team",
        "possession_before",
        "pre_home_score",
        "pre_away_score",
        "down",
        "distance",
        "yardline_100",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
        "primary_action",
        "row_disposition",
        "review_result",
        "pbp_source_sha256",
    }
)

STATE_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "experiment_id",
    "claim_boundary",
    "game_id",
    "state_event_id",
    "state_raw_play_id",
    "state_atomic_information_episode_id",
    "order_sequence",
    "state_known_at",
    "quarter",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "possession_team",
    "p_home",
    "reference_status",
    "exclusion_reasons",
    "state_input_sha256",
    "model_id",
    "model_version",
    "model_sha256",
    "model_manifest_sha256",
    "pbp_source_sha256",
)
FEATURE_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "experiment_id",
    "game_id",
    "state_event_id",
    "state_raw_play_id",
    "state_known_at",
    "feature_name",
    "feature_value",
    "feature_known_at",
    "source_event_id",
    "source_raw_play_id",
    "source_row_id",
    "source_hash",
    "PIT_status",
)
REFERENCE_OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "experiment_id",
    "claim_boundary",
    "game_id",
    "event_id",
    "raw_play_id",
    "atomic_information_episode_id",
    "source_interval_start",
    "source_interval_end",
    "pre_state_event_id",
    "pre_state_raw_play_id",
    "pre_state_known_at",
    "post_state_event_id",
    "post_state_raw_play_id",
    "post_state_known_at",
    "p_before_home",
    "p_after_home",
    "reference_delta_home",
    "bridge_p_after_home",
    "bridge_delta_home",
    "intervening_episode_ids",
    "reference_status",
    "exclusion_reasons",
    "pre_state_input_sha256",
    "post_state_input_sha256",
    "model_id",
    "model_sha256",
    "model_manifest_sha256",
    "pbp_source_sha256",
)


class ReferenceValueBuildError(ValueError):
    """Canonical facts cannot support a trustworthy Stage A result."""


class HomeProbabilityPredictor(Protocol):
    def predict_home(self, model_input: NoSpreadModelInput) -> float: ...


def _require_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ReferenceValueBuildError(f"{field} must be canonical non-empty text")
    return value


def _require_sha(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ReferenceValueBuildError(f"{field} must be a sha256 digest")
    return value


def _optional_text(value: object) -> str | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    if type(value) is not str or not value.strip():
        return None
    return value.strip()


def _finite_number(value: object) -> float | None:
    if value is None or value is pd.NA or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _strict_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _timestamp(value: object, *, field: str) -> pd.Timestamp:
    text = _optional_text(value)
    if text is None:
        raise ReferenceValueBuildError(f"{field} must be a timestamp")
    try:
        result = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise ReferenceValueBuildError(f"{field} must be a timestamp") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ReferenceValueBuildError(f"{field} must be timezone-aware UTC")
    return result


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ReferenceValueBuildError(
                "canonical material cannot contain non-finite numbers"
            )
        return number
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    raise ReferenceValueBuildError(
        f"canonical material contains unsupported {type(value).__name__}"
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class ReferenceModelBindingV1:
    model_id: str
    model_version: str
    model_sha256: str
    manifest_sha256: str
    feature_spec_commit: str

    def __post_init__(self) -> None:
        _require_text(self.model_id, field="model_id")
        _require_text(self.model_version, field="model_version")
        _require_sha(self.model_sha256, field="model_sha256")
        _require_sha(self.manifest_sha256, field="manifest_sha256")
        if (
            type(self.feature_spec_commit) is not str
            or re.fullmatch(r"[0-9a-f]{40}", self.feature_spec_commit) is None
        ):
            raise ReferenceValueBuildError(
                "feature_spec_commit must be a git commit"
            )


@dataclass(frozen=True, slots=True)
class ReferenceFeatureProvenanceV1:
    game_id: str
    state_event_id: str
    state_raw_play_id: str
    state_known_at: str
    feature_name: str
    feature_value: float
    feature_known_at: str
    source_event_id: str
    source_raw_play_id: str
    source_row_id: str
    source_hash: str
    PIT_status: Literal["PIT_VERIFIED"] = "PIT_VERIFIED"
    schema_version: Literal["ReferenceFeatureProvenanceV1"] = (
        "ReferenceFeatureProvenanceV1"
    )
    experiment_id: Literal["X-15"] = "X-15"

    def __post_init__(self) -> None:
        for field in (
            "game_id",
            "state_event_id",
            "state_raw_play_id",
            "source_event_id",
            "source_raw_play_id",
            "source_row_id",
        ):
            _require_text(getattr(self, field), field=field)
        if self.feature_name not in FEATURE_NAMES:
            raise ReferenceValueBuildError("feature_name is not in frozen order")
        number = _finite_number(self.feature_value)
        if number is None:
            raise ReferenceValueBuildError("feature_value must be finite")
        object.__setattr__(self, "feature_value", number)
        state_time = _timestamp(self.state_known_at, field="state_known_at")
        feature_time = _timestamp(
            self.feature_known_at,
            field="feature_known_at",
        )
        if feature_time > state_time:
            raise ReferenceValueBuildError(
                "feature_known_at cannot be after state_known_at"
            )
        _require_sha(self.source_hash, field="source_hash")
        if self.PIT_status != "PIT_VERIFIED":
            raise ReferenceValueBuildError("PIT_status must be PIT_VERIFIED")

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReferenceValueObservationV1:
    game_id: str
    event_id: str
    raw_play_id: str
    atomic_information_episode_id: str
    source_interval_start: str | None
    source_interval_end: str | None
    pre_state_event_id: str | None
    pre_state_raw_play_id: str | None
    pre_state_known_at: str | None
    post_state_event_id: str | None
    post_state_raw_play_id: str | None
    post_state_known_at: str | None
    p_before_home: float | None
    p_after_home: float | None
    reference_delta_home: float | None
    bridge_p_after_home: float | None
    bridge_delta_home: float | None
    intervening_episode_ids: tuple[str, ...]
    reference_status: ReferenceStatus
    exclusion_reasons: tuple[str, ...]
    pre_state_input_sha256: str | None
    post_state_input_sha256: str | None
    model_id: str
    model_sha256: str
    model_manifest_sha256: str
    pbp_source_sha256: str
    schema_version: Literal["ReferenceValueObservationV1"] = (
        "ReferenceValueObservationV1"
    )
    experiment_id: Literal["X-15"] = "X-15"
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for field in (
            "game_id",
            "event_id",
            "raw_play_id",
            "atomic_information_episode_id",
            "model_id",
        ):
            _require_text(getattr(self, field), field=field)
        if self.reference_status not in _REFERENCE_STATUSES:
            raise ReferenceValueBuildError("reference_status is unsupported")
        _require_sha(self.model_sha256, field="model_sha256")
        _require_sha(self.model_manifest_sha256, field="model_manifest_sha256")
        _require_sha(self.pbp_source_sha256, field="pbp_source_sha256")
        for field in ("pre_state_input_sha256", "post_state_input_sha256"):
            value = getattr(self, field)
            if value is not None:
                _require_sha(value, field=field)
        for field in (
            "p_before_home",
            "p_after_home",
            "bridge_p_after_home",
        ):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ReferenceValueBuildError(
                    f"{field} must be a probability or null"
                )
        for field in ("reference_delta_home", "bridge_delta_home"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
            ):
                raise ReferenceValueBuildError(
                    f"{field} must be a probability delta or null"
                )
        if self.reference_status == "SUPPORTED" and (
            self.p_before_home is None
            or self.p_after_home is None
            or self.reference_delta_home is None
            or self.pre_state_input_sha256 is None
            or self.post_state_input_sha256 is None
        ):
            raise ReferenceValueBuildError(
                "supported reference requires probabilities and input hashes"
            )
        if self.reference_status == "SUPPORTED":
            if (
                self.pre_state_known_at is None
                or self.post_state_known_at is None
                or _timestamp(
                    self.post_state_known_at,
                    field="post_state_known_at",
                )
                < _timestamp(
                    self.pre_state_known_at,
                    field="pre_state_known_at",
                )
            ):
                raise ReferenceValueBuildError(
                    "supported reference requires monotonic state known_at"
                )
        if self.reference_status == "COMPOSITE_TRANSITION" and (
            self.p_after_home is not None
            or self.reference_delta_home is not None
            or not self.intervening_episode_ids
        ):
            raise ReferenceValueBuildError(
                "composite transition cannot publish an atomic delta"
            )
        if self.reference_status == "ORDER_AMBIGUOUS" and any(
            value is not None
            for value in (
                self.p_after_home,
                self.reference_delta_home,
                self.bridge_p_after_home,
                self.bridge_delta_home,
            )
        ):
            raise ReferenceValueBuildError(
                "order-ambiguous reference cannot publish a post-state delta"
            )
        if (
            self.p_before_home is not None
            and self.p_after_home is not None
            and self.reference_delta_home is not None
            and not math.isclose(
                self.p_after_home - self.p_before_home,
                self.reference_delta_home,
                abs_tol=1e-12,
            )
        ):
            raise ReferenceValueBuildError(
                "reference_delta_home does not match probabilities"
            )

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["intervening_episode_ids"] = _json_tuple(
            self.intervening_episode_ids
        )
        record["exclusion_reasons"] = _json_tuple(self.exclusion_reasons)
        return record


@dataclass(frozen=True, slots=True)
class GameReferenceTables:
    game_id: str
    second_half_receiver: str | None
    opening_kickoff_event_id: str | None
    opening_kickoff_known_at: str | None
    state_predictions: pd.DataFrame
    feature_provenance: pd.DataFrame
    reference_observations: pd.DataFrame


def _status_record(
    row: pd.Series,
    *,
    model: ReferenceModelBindingV1,
    status: ReferenceStatus,
    reasons: tuple[str, ...],
    p_home: float | None = None,
    state_input_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "ReferenceStatePredictionV1",
        "experiment_id": EXPERIMENT_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "game_id": str(row["game_id"]),
        "state_event_id": str(row["event_id"]),
        "state_raw_play_id": str(row["raw_play_id"]),
        "state_atomic_information_episode_id": str(
            row["atomic_information_episode_id"]
        ),
        "order_sequence": int(row["order_sequence"]),
        "state_known_at": _optional_text(row["known_at"]),
        "quarter": _integer(row["quarter"]),
        "game_seconds_remaining": _integer(row["game_seconds_remaining"]),
        "home_team": str(row["home_team"]),
        "away_team": str(row["away_team"]),
        "possession_team": _optional_text(row["possession_before"]),
        "p_home": p_home,
        "reference_status": status,
        "exclusion_reasons": _json_tuple(reasons),
        "state_input_sha256": state_input_sha256,
        "model_id": model.model_id,
        "model_version": model.model_version,
        "model_sha256": model.model_sha256,
        "model_manifest_sha256": model.manifest_sha256,
        "pbp_source_sha256": str(row["pbp_source_sha256"]),
    }


def _opening_kickoff(
    ordered: pd.DataFrame,
) -> tuple[str | None, pd.Series | None]:
    candidates = ordered.loc[
        ordered["quarter"].map(_integer).eq(1)
        & ordered["primary_action"]
        .map(_optional_text)
        .fillna("")
        .str.upper()
        .eq("KICKOFF")
    ]
    if candidates.empty:
        return None, None
    opening = candidates.iloc[0]
    home = _optional_text(opening["home_team"])
    away = _optional_text(opening["away_team"])
    receiver = _optional_text(opening["possession_before"])
    kicker = _optional_text(opening["actor_team"])
    if (
        home is None
        or away is None
        or receiver not in {home, away}
        or kicker not in {home, away}
        or receiver == kicker
        or _optional_text(opening["known_at"]) is None
    ):
        return None, opening
    return kicker, opening


def _event_is_stage_a_eligible(row: pd.Series) -> bool:
    status = (_optional_text(row["information_status"]) or "").upper()
    disposition = (_optional_text(row["row_disposition"]) or "").upper()
    review = (_optional_text(row["review_result"]) or "").lower()
    return (
        status == "FINAL"
        and _strict_bool(row["final_sports_outcome_eligible"])
        and "NO_PLAY" not in disposition
        and "NULLIFIED" not in disposition
        and "REVERSED" not in disposition
        and "reversed" not in review
    )


def _information_event_is_independent(row: pd.Series) -> bool:
    return _strict_bool(row["stage_b_information_event_eligible"]) or _strict_bool(
        row["final_sports_outcome_eligible"]
    )


def _state_projection(
    row: pd.Series,
    *,
    second_half_receiver: str | None,
    opening: pd.Series | None,
    predictor: HomeProbabilityPredictor,
    model: ReferenceModelBindingV1,
    expected_pbp_source_sha256: str,
) -> tuple[dict[str, object], tuple[ReferenceFeatureProvenanceV1, ...]]:
    period = _integer(row["quarter"])
    if period is None or period not in {1, 2, 3, 4}:
        return (
            _status_record(
                row,
                model=model,
                status="MODEL_SUPPORT_UNPROVEN",
                reasons=("REGULATION_ONLY",),
            ),
            (),
        )
    state_known_text = _optional_text(row["known_at"])
    if state_known_text is None:
        return (
            _status_record(
                row,
                model=model,
                status="MISSING_PRE_STATE",
                reasons=("MISSING_STATE_KNOWN_AT",),
            ),
            (),
        )
    if second_half_receiver is None or opening is None:
        return (
            _status_record(
                row,
                model=model,
                status="MISSING_AS_OF_RECEIVER",
                reasons=("MISSING_OPENING_KICKOFF_EVIDENCE",),
            ),
            (),
        )
    opening_known_text = _optional_text(opening["known_at"])
    if opening_known_text is None:
        return (
            _status_record(
                row,
                model=model,
                status="MISSING_AS_OF_RECEIVER",
                reasons=("MISSING_OPENING_KICKOFF_KNOWN_AT",),
            ),
            (),
        )
    state_known = _timestamp(state_known_text, field="state known_at")
    opening_known = _timestamp(
        opening_known_text,
        field="opening kickoff known_at",
    )
    if opening_known > state_known:
        return (
            _status_record(
                row,
                model=model,
                status="FUTURE_FEATURE_REJECTED",
                reasons=("RECEIVE_2H_KO_KNOWN_AFTER_STATE",),
            ),
            (),
        )
    home_team = _optional_text(row["home_team"])
    away_team = _optional_text(row["away_team"])
    possession = _optional_text(row["possession_before"])
    game_seconds = _finite_number(row["game_seconds_remaining"])
    home_score = _finite_number(row["pre_home_score"])
    away_score = _finite_number(row["pre_away_score"])
    down = _finite_number(row["down"])
    distance = _finite_number(row["distance"])
    yardline = _finite_number(row["yardline_100"])
    home_timeouts = _finite_number(row["home_timeouts_remaining"])
    away_timeouts = _finite_number(row["away_timeouts_remaining"])
    if (
        home_team is None
        or away_team is None
        or home_team == away_team
        or possession not in {home_team, away_team}
        or any(
            value is None
            for value in (
                game_seconds,
                home_score,
                away_score,
                down,
                distance,
                yardline,
                home_timeouts,
                away_timeouts,
            )
        )
    ):
        return (
            _status_record(
                row,
                model=model,
                status="MISSING_PRE_STATE",
                reasons=("INCOMPLETE_OFFICIAL_MODEL_STATE",),
            ),
            (),
        )
    assert game_seconds is not None
    assert home_score is not None
    assert away_score is not None
    assert down is not None
    assert distance is not None
    assert yardline is not None
    assert home_timeouts is not None
    assert away_timeouts is not None
    possession_is_home = possession == home_team
    score_differential = (
        home_score - away_score
        if possession_is_home
        else away_score - home_score
    )
    half_seconds = (
        game_seconds - 1800.0 if period <= 2 else game_seconds
    )
    elapsed_share = (3600.0 - game_seconds) / 3600.0
    posteam_timeouts = home_timeouts if possession_is_home else away_timeouts
    defteam_timeouts = away_timeouts if possession_is_home else home_timeouts
    feature_values = (
        float(period <= 2 and possession == second_half_receiver),
        float(possession_is_home),
        half_seconds,
        game_seconds,
        score_differential / math.exp(-4.0 * elapsed_share),
        score_differential,
        down,
        distance,
        yardline,
        posteam_timeouts,
        defteam_timeouts,
    )
    try:
        model_input = NoSpreadModelInput(
            feature_names=FEATURE_NAMES,
            feature_values=tuple(float(value) for value in feature_values),
            posteam=possession,
            home_team=home_team,
            away_team=away_team,
            period=period,
        )
    except OfficialModelInputError:
        return (
            _status_record(
                row,
                model=model,
                status="MISSING_PRE_STATE",
                reasons=("INVALID_OFFICIAL_MODEL_STATE",),
            ),
            (),
        )
    state_event_id = str(row["event_id"])
    state_raw_play_id = str(row["raw_play_id"])
    current_source_hash = str(row["pbp_source_sha256"])
    provenance: list[ReferenceFeatureProvenanceV1] = []
    for name, value in zip(
        FEATURE_NAMES,
        model_input.feature_values,
        strict=True,
    ):
        source = opening if name == "receive_2h_ko" else row
        provenance.append(
            ReferenceFeatureProvenanceV1(
                game_id=str(row["game_id"]),
                state_event_id=state_event_id,
                state_raw_play_id=state_raw_play_id,
                state_known_at=state_known_text,
                feature_name=name,
                feature_value=value,
                feature_known_at=(
                    opening_known_text
                    if name == "receive_2h_ko"
                    else state_known_text
                ),
                source_event_id=str(source["event_id"]),
                source_raw_play_id=str(source["raw_play_id"]),
                source_row_id=str(source["event_id"]),
                source_hash=(
                    expected_pbp_source_sha256
                    if name == "receive_2h_ko"
                    else current_source_hash
                ),
            )
        )
    input_material = {
        "model": asdict(model),
        "game_id": str(row["game_id"]),
        "state_event_id": state_event_id,
        "state_raw_play_id": state_raw_play_id,
        "state_known_at": state_known_text,
        "feature_names": list(FEATURE_NAMES),
        "feature_values": list(model_input.feature_values),
        "feature_provenance": [item.to_record() for item in provenance],
    }
    input_sha = _canonical_sha256(input_material)
    probability = predictor.predict_home(model_input)
    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ReferenceValueBuildError(
            "official predictor returned an invalid home probability"
        )
    return (
        _status_record(
            row,
            model=model,
            status="SUPPORTED",
            reasons=(),
            p_home=float(probability),
            state_input_sha256=input_sha,
        ),
        tuple(provenance),
    )


def build_game_reference_tables(
    canonical_events: pd.DataFrame,
    *,
    predictor: HomeProbabilityPredictor,
    model: ReferenceModelBindingV1,
    expected_pbp_source_sha256: str,
) -> GameReferenceTables:
    """Build one game without trusting physical row order or diagnostic WP."""

    if not isinstance(canonical_events, pd.DataFrame) or canonical_events.empty:
        raise ReferenceValueBuildError(
            "canonical_events must be a non-empty DataFrame"
        )
    missing = sorted(_REQUIRED_EVENT_COLUMNS.difference(canonical_events.columns))
    if missing:
        raise ReferenceValueBuildError(
            "canonical_events missing columns: " + ", ".join(missing)
        )
    expected_source = _require_sha(
        expected_pbp_source_sha256,
        field="expected_pbp_source_sha256",
    )
    ordered = canonical_events.copy()
    order_values = pd.to_numeric(ordered["order_sequence"], errors="coerce")
    if order_values.isna().any() or (order_values % 1 != 0).any():
        raise ReferenceValueBuildError(
            "order_sequence must be complete integral chronology"
        )
    ordered["_order"] = order_values.astype("int64")
    ordered["_raw_sort"] = ordered["raw_play_id"].astype("string")
    if ordered["_raw_sort"].isna().any() or ordered["_raw_sort"].eq("").any():
        raise ReferenceValueBuildError("raw_play_id must be complete")
    if ordered.duplicated(["_order", "_raw_sort"]).any():
        raise ReferenceValueBuildError(
            "canonical state chronology contains duplicate sort keys"
        )
    ordered = ordered.sort_values(
        ["_order", "_raw_sort"],
        kind="mergesort",
    ).reset_index(drop=True)
    games = tuple(ordered["game_id"].astype(str).unique())
    if len(games) != 1:
        raise ReferenceValueBuildError(
            "build_game_reference_tables requires exactly one game"
        )
    game_id = games[0]
    teams = ordered[["home_team", "away_team"]].drop_duplicates()
    if len(teams) != 1:
        raise ReferenceValueBuildError("game team identity is inconsistent")
    source_hashes = set(ordered["pbp_source_sha256"].astype(str))
    if source_hashes != {expected_source}:
        raise ReferenceValueBuildError(
            "canonical event PBP hashes do not match the governed source"
        )

    second_half_receiver, opening = _opening_kickoff(ordered)
    state_records: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        state_record, provenance = _state_projection(
            row,
            second_half_receiver=second_half_receiver,
            opening=opening,
            predictor=predictor,
            model=model,
            expected_pbp_source_sha256=expected_source,
        )
        state_records.append(state_record)
        feature_rows.extend(item.to_record() for item in provenance)
    states = pd.DataFrame(state_records, columns=STATE_PREDICTION_COLUMNS)
    features = pd.DataFrame(
        feature_rows,
        columns=FEATURE_PROVENANCE_COLUMNS,
    )

    observations: list[dict[str, object]] = []
    for index, event in ordered.iterrows():
        pre_state = states.iloc[index]
        later_supported = states.iloc[index + 1 :].loc[
            states.iloc[index + 1 :]["reference_status"].eq("SUPPORTED")
        ]
        post_index = (
            None if later_supported.empty else int(later_supported.index[0])
        )
        post_state = None if post_index is None else states.iloc[post_index]
        between = (
            ordered.iloc[0:0]
            if post_index is None
            else ordered.iloc[index + 1 : post_index]
        )
        intervening = tuple(
            dict.fromkeys(
                str(row["atomic_information_episode_id"])
                for _, row in between.iterrows()
                if _information_event_is_independent(row)
            )
        )

        status: ReferenceStatus
        reasons: tuple[str, ...]
        p_before: float | None = None
        p_after: float | None = None
        delta: float | None = None
        bridge_after: float | None = None
        bridge_delta: float | None = None
        if not _event_is_stage_a_eligible(event):
            status = "EVENT_INELIGIBLE"
            reasons = ("NOT_FINAL_EFFECTIVE_SPORTS_EVENT",)
        elif pre_state["reference_status"] != "SUPPORTED":
            status = str(pre_state["reference_status"])  # type: ignore[assignment]
            reasons = tuple(
                json.loads(str(pre_state["exclusion_reasons"]))
            )
        elif post_state is None:
            status = "MISSING_POST_STATE"
            reasons = ("NO_LATER_VALID_CANONICAL_PRE_STATE",)
            p_before = float(pre_state["p_home"])
        else:
            p_before = float(pre_state["p_home"])
            pre_known = _timestamp(
                str(pre_state["state_known_at"]),
                field="pre_state_known_at",
            )
            post_known = _timestamp(
                str(post_state["state_known_at"]),
                field="post_state_known_at",
            )
            if post_known < pre_known:
                status = "ORDER_AMBIGUOUS"
                reasons = ("POST_STATE_KNOWN_BEFORE_PRE_STATE",)
            else:
                bridge_after = float(post_state["p_home"])
                bridge_delta = bridge_after - p_before
                if intervening:
                    status = "COMPOSITE_TRANSITION"
                    reasons = (
                        "INDEPENDENT_INFORMATION_EVENT_BETWEEN_STATES",
                    )
                else:
                    status = "SUPPORTED"
                    reasons = ()
                    p_after = bridge_after
                    delta = bridge_delta
        observation = ReferenceValueObservationV1(
            game_id=game_id,
            event_id=str(event["event_id"]),
            raw_play_id=str(event["raw_play_id"]),
            atomic_information_episode_id=str(
                event["atomic_information_episode_id"]
            ),
            source_interval_start=_optional_text(event["source_interval_start"]),
            source_interval_end=_optional_text(event["source_interval_end"]),
            pre_state_event_id=str(pre_state["state_event_id"]),
            pre_state_raw_play_id=str(pre_state["state_raw_play_id"]),
            pre_state_known_at=_optional_text(pre_state["state_known_at"]),
            post_state_event_id=(
                None
                if post_state is None
                else str(post_state["state_event_id"])
            ),
            post_state_raw_play_id=(
                None
                if post_state is None
                else str(post_state["state_raw_play_id"])
            ),
            post_state_known_at=(
                None
                if post_state is None
                else _optional_text(post_state["state_known_at"])
            ),
            p_before_home=p_before,
            p_after_home=p_after,
            reference_delta_home=delta,
            bridge_p_after_home=bridge_after,
            bridge_delta_home=bridge_delta,
            intervening_episode_ids=intervening,
            reference_status=status,
            exclusion_reasons=reasons,
            pre_state_input_sha256=_optional_text(
                pre_state["state_input_sha256"]
            ),
            post_state_input_sha256=(
                None
                if post_state is None
                else _optional_text(post_state["state_input_sha256"])
            ),
            model_id=model.model_id,
            model_sha256=model.model_sha256,
            model_manifest_sha256=model.manifest_sha256,
            pbp_source_sha256=expected_source,
        )
        observations.append(observation.to_record())
    observation_frame = pd.DataFrame(
        observations,
        columns=REFERENCE_OBSERVATION_COLUMNS,
    )
    return GameReferenceTables(
        game_id=game_id,
        second_half_receiver=second_half_receiver,
        opening_kickoff_event_id=(
            None if opening is None else str(opening["event_id"])
        ),
        opening_kickoff_known_at=(
            None if opening is None else _optional_text(opening["known_at"])
        ),
        state_predictions=states,
        feature_provenance=features,
        reference_observations=observation_frame,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "EXPERIMENT_ID",
    "FEATURE_PROVENANCE_COLUMNS",
    "GameReferenceTables",
    "HomeProbabilityPredictor",
    "REFERENCE_OBSERVATION_COLUMNS",
    "ReferenceFeatureProvenanceV1",
    "ReferenceModelBindingV1",
    "ReferenceValueBuildError",
    "ReferenceValueObservationV1",
    "STATE_PREDICTION_COLUMNS",
    "build_game_reference_tables",
]
