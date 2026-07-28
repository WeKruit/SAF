"""Pinned no-spread reference values and local-projection regressions for X-13.

The module intentionally contains no model-training code.  A caller injects a
specific, content-addressed nflfastR no-spread XGBoost predictor, and this
module records the exact feature payloads supplied to it.  Its regression
interface is likewise in-memory only: it estimates market reactions from
provided observations and uses game-cluster resampling for uncertainty draws.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from prediction_market.models.nfl_fastrmodels import (
    FEATURE_NAMES,
    NoSpreadModelInput,
    OfficialModelInputError,
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
# This is deliberately an alias, not a locally maintained approximation.  It
# keeps the reference interface bound to the official pinned model contract.
NO_SPREAD_FEATURE_ALLOWLIST_V2 = FEATURE_NAMES
ReferenceSupportStatusV2 = Literal[
    "SUPPORTED",
    "UNSUPPORTED_UNFINALIZED",
    "UNSUPPORTED_RECEIVE_2H_KO_AFTER_STATE",
    "EXCLUDED_OVERTIME",
    "EXCLUDED_NULLIFIED",
    "EXCLUDED_REVERSED",
    "EXCLUDED_MISSING_POST_STATE",
]
RegressionStatusV2 = Literal["ESTIMATED", "UNSUPPORTED"]


class ReferenceRegressionError(ValueError):
    """Inputs cannot support a publication-ready V2 reference estimate."""


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ReferenceRegressionError(f"{field_name} must be non-empty text")
    return value.strip()


def _sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ReferenceRegressionError(f"{field_name} must be a SHA-256 digest")
    return value


def _integer(value: object, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ReferenceRegressionError(f"{field_name} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ReferenceRegressionError(f"{field_name} must be at least {minimum}")
    return number


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReferenceRegressionError(f"{field_name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ReferenceRegressionError(f"{field_name} must be finite")
    return number


def _probability(value: object, *, field_name: str) -> float:
    number = _finite(value, field_name=field_name)
    if not 0.0 <= number <= 1.0:
        raise ReferenceRegressionError(
            f"{field_name} must be between zero and one"
        )
    return number


def _canonical_value(value: object) -> object:
    """Return only deterministic, JSON-compatible feature material."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        material: dict[str, object] = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if type(key) is not str:
                raise ReferenceRegressionError(
                    "canonical feature payload mapping keys must be strings"
                )
            material[key] = _canonical_value(child)
        return material
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    if type(value) is bool or value is None or type(value) in {int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReferenceRegressionError(
                "canonical feature payload cannot contain non-finite floats"
            )
        return value
    raise ReferenceRegressionError(
        "canonical feature payload contains unsupported "
        f"{type(value).__name__}"
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reject_diagnostic_only_controls(
    value: object,
    *,
    path: str = "state",
    context: str = "no-spread reference payload",
) -> None:
    """Reject diagnostic-only material from formal features and controls."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            name = _text(key, field_name=f"{path} feature name").lower()
            if any(
                token in name
                for token in (
                    "spread",
                    "vegas",
                    "diagnostic",
                    "reference",
                    "precomputed",
                    "model_output",
                )
            ):
                raise ReferenceRegressionError(
                    f"{context} cannot contain diagnostic-only field {path}.{key}"
                )
            _reject_diagnostic_only_controls(
                child,
                path=f"{path}.{key}",
                context=context,
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_diagnostic_only_controls(
                child,
                path=f"{path}[{index}]",
                context=context,
            )


def _controls(
    value: Mapping[str, object], *, field_name: str
) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        raise ReferenceRegressionError(f"{field_name} must be a mapping")
    _reject_diagnostic_only_controls(
        value,
        path=field_name,
        context="formal control",
    )
    entries = tuple(
        sorted(
            (
                _text(name, field_name=f"{field_name} name"),
                _finite(number, field_name=f"{field_name}.{name}"),
            )
            for name, number in value.items()
        )
    )
    if len({name for name, _number in entries}) != len(entries):
        raise ReferenceRegressionError(f"{field_name} names must be unique")
    return entries


@dataclass(frozen=True, slots=True)
class NoSpreadFeatureAsOfV2:
    """One frozen no-spread model input and its source-time attestation."""

    name: str
    value: float | int | bool
    as_of_ns: int
    as_of_provenance_sha256: str

    def __post_init__(self) -> None:
        _text(self.name, field_name="feature name")
        if type(self.value) is not bool:
            _finite(self.value, field_name=f"feature {self.name}")
        _integer(self.as_of_ns, field_name=f"feature {self.name} as_of_ns", minimum=0)
        _sha256(
            self.as_of_provenance_sha256,
            field_name=f"feature {self.name} as_of_provenance_sha256",
        )


@dataclass(frozen=True, slots=True)
class NoSpreadFeatureStateV2:
    """The complete, ordered no-spread feature vector for one canonical state."""

    features: tuple[NoSpreadFeatureAsOfV2, ...]

    def __post_init__(self) -> None:
        if type(self.features) is not tuple or not all(
            isinstance(feature, NoSpreadFeatureAsOfV2) for feature in self.features
        ):
            raise ReferenceRegressionError(
                "features must be a tuple of NoSpreadFeatureAsOfV2"
            )


@dataclass(frozen=True, slots=True)
class PinnedNoSpreadXGBoostPredictorV2:
    """A loaded, injected predictor; this class never trains or reloads it."""

    asset_id: str
    asset_version: str
    asset_sha256: str
    probability_outcome: Literal["home", "focal"]
    predict_probability: Callable[[NoSpreadModelInput], object]

    def __post_init__(self) -> None:
        _text(self.asset_id, field_name="asset_id")
        _text(self.asset_version, field_name="asset_version")
        _sha256(self.asset_sha256, field_name="asset_sha256")
        if self.probability_outcome not in {"home", "focal"}:
            raise ReferenceRegressionError(
                "probability_outcome must be 'home' or 'focal'"
            )
        if not callable(self.predict_probability):
            raise ReferenceRegressionError("predict_probability must be callable")


@dataclass(frozen=True, slots=True)
class FinalizedEpisodeReferenceInputV2:
    """One finalized canonical episode and the exact source-time provenance."""

    game_id: str
    episode_id: str
    home_team: str
    away_team: str
    focal_team: str
    quarter: int
    pre_state: NoSpreadFeatureStateV2
    post_state: NoSpreadFeatureStateV2 | None
    pre_state_time_ns: int
    post_state_time_ns: int
    receive_2h_ko_ns: int
    finalized: bool
    nullified: bool = False
    reversed: bool = False
    focal_outcome: Literal["win"] = "win"


@dataclass(frozen=True, slots=True)
class NoSpreadReferenceObservationV2:
    """Auditable prediction pair, always oriented to the focal outcome."""

    game_id: str
    episode_id: str
    home_team: str
    away_team: str
    focal_team: str
    focal_outcome: Literal["win"]
    quarter: int
    pre_state_time_ns: int
    post_state_time_ns: int
    receive_2h_ko_ns: int
    predictor_asset_id: str
    predictor_asset_version: str
    predictor_asset_sha256: str
    probability_outcome: Literal["home", "focal"]
    pre_feature_payload: Mapping[str, object] | None
    post_feature_payload: Mapping[str, object] | None
    pre_feature_sha256: str | None
    post_feature_sha256: str | None
    p_before: float | None
    p_after: float | None
    reference_delta: float | None
    support_status: ReferenceSupportStatusV2
    exclusion_reason: str | None
    observation_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "NoSpreadReferenceObservationV2",
            "game_id": self.game_id,
            "episode_id": self.episode_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "focal_team": self.focal_team,
            "focal_outcome": self.focal_outcome,
            "quarter": self.quarter,
            "pre_state_time_ns": self.pre_state_time_ns,
            "post_state_time_ns": self.post_state_time_ns,
            "receive_2h_ko_ns": self.receive_2h_ko_ns,
            "predictor_asset_id": self.predictor_asset_id,
            "predictor_asset_version": self.predictor_asset_version,
            "predictor_asset_sha256": self.predictor_asset_sha256,
            "probability_outcome": self.probability_outcome,
            "pre_feature_payload": (
                None
                if self.pre_feature_payload is None
                else _canonical_value(self.pre_feature_payload)
            ),
            "post_feature_payload": (
                None
                if self.post_feature_payload is None
                else _canonical_value(self.post_feature_payload)
            ),
            "pre_feature_sha256": self.pre_feature_sha256,
            "post_feature_sha256": self.post_feature_sha256,
            "p_before": self.p_before,
            "p_after": self.p_after,
            "reference_delta": self.reference_delta,
            "support_status": self.support_status,
            "exclusion_reason": self.exclusion_reason,
            "observation_sha256": self.observation_sha256,
        }


def _validate_episode(
    episode: FinalizedEpisodeReferenceInputV2,
) -> tuple[str, str, str, str, str, int, int, int, int]:
    game_id = _text(episode.game_id, field_name="game_id")
    episode_id = _text(episode.episode_id, field_name="episode_id")
    home_team = _text(episode.home_team, field_name="home_team")
    away_team = _text(episode.away_team, field_name="away_team")
    focal_team = _text(episode.focal_team, field_name="focal_team")
    if home_team == away_team or focal_team not in {home_team, away_team}:
        raise ReferenceRegressionError("episode teams do not support focal orientation")
    quarter = _integer(episode.quarter, field_name="quarter", minimum=1)
    pre_time = _integer(
        episode.pre_state_time_ns, field_name="pre_state_time_ns", minimum=0
    )
    post_time = _integer(
        episode.post_state_time_ns, field_name="post_state_time_ns", minimum=0
    )
    receive_time = _integer(
        episode.receive_2h_ko_ns,
        field_name="receive_2h_ko_ns",
        minimum=0,
    )
    if post_time < pre_time:
        raise ReferenceRegressionError(
            "post_state_time_ns must not precede pre_state_time_ns"
        )
    for flag, name in (
        (episode.finalized, "finalized"),
        (episode.nullified, "nullified"),
        (episode.reversed, "reversed"),
    ):
        if type(flag) is not bool:
            raise ReferenceRegressionError(f"{name} must be boolean")
    return (
        game_id,
        episode_id,
        home_team,
        away_team,
        focal_team,
        quarter,
        pre_time,
        post_time,
        receive_time,
    )


def _feature_payload(
    *,
    episode: FinalizedEpisodeReferenceInputV2,
    predictor: PinnedNoSpreadXGBoostPredictorV2,
    state: NoSpreadFeatureStateV2,
    state_time_ns: int,
) -> tuple[dict[str, object], NoSpreadModelInput]:
    if not isinstance(state, NoSpreadFeatureStateV2):
        raise ReferenceRegressionError(
            "canonical state must be NoSpreadFeatureStateV2"
        )
    feature_names = tuple(feature.name for feature in state.features)
    if feature_names != NO_SPREAD_FEATURE_ALLOWLIST_V2:
        raise ReferenceRegressionError(
            "canonical state must contain the exact frozen no-spread feature "
            "allowlist in canonical order"
        )
    feature_values: dict[str, float | int | bool] = {}
    feature_as_of: dict[str, dict[str, object]] = {}
    for feature in state.features:
        if feature.as_of_ns > state_time_ns:
            raise ReferenceRegressionError(
                f"feature {feature.name} as-of provenance is after state time"
            )
        feature_values[feature.name] = feature.value
        feature_as_of[feature.name] = {
            "as_of_ns": feature.as_of_ns,
            "as_of_provenance_sha256": feature.as_of_provenance_sha256,
        }
    home_feature = float(feature_values["home"])
    posteam = episode.home_team if home_feature == 1.0 else episode.away_team
    try:
        official_input = NoSpreadModelInput(
            feature_names=FEATURE_NAMES,
            feature_values=tuple(
                float(feature_values[name]) for name in FEATURE_NAMES
            ),
            posteam=posteam,
            home_team=episode.home_team,
            away_team=episode.away_team,
            period=episode.quarter,
        )
    except OfficialModelInputError as error:
        raise ReferenceRegressionError(
            "canonical state fails the official no-spread feature adapter"
        ) from error
    return (
        {
            "schema_version": "NflNoSpreadXGBoostFeaturePayloadV2",
            "predictor_asset_id": predictor.asset_id,
            "predictor_asset_version": predictor.asset_version,
            "predictor_asset_sha256": predictor.asset_sha256,
            "probability_outcome": predictor.probability_outcome,
            "game_id": episode.game_id,
            "episode_id": episode.episode_id,
            "home_team": episode.home_team,
            "away_team": episode.away_team,
            "focal_team": episode.focal_team,
            "focal_outcome": episode.focal_outcome,
            "state_time_ns": state_time_ns,
            "feature_names": official_input.feature_names,
            "features": dict(
                zip(
                    official_input.feature_names,
                    official_input.feature_values,
                    strict=True,
                )
            ),
            "feature_as_of_provenance": feature_as_of,
        },
        official_input,
    )


def _freeze_canonical_snapshot(value: object) -> object:
    material = _canonical_value(value)
    if isinstance(material, Mapping):
        return MappingProxyType(
            {
                key: _freeze_canonical_snapshot(child)
                for key, child in material.items()
            }
        )
    if isinstance(material, list):
        return tuple(_freeze_canonical_snapshot(child) for child in material)
    return material


def _focal_probability(
    predictor: PinnedNoSpreadXGBoostPredictorV2,
    model_input: NoSpreadModelInput,
    *,
    focal_team: str,
    home_team: str,
) -> float:
    if not isinstance(model_input, NoSpreadModelInput):
        raise ReferenceRegressionError(
            "predictor must receive a validated official no-spread model input"
        )
    raw_probability = _probability(
        predictor.predict_probability(model_input),
        field_name="predictor probability",
    )
    if predictor.probability_outcome == "focal":
        return raw_probability
    return raw_probability if focal_team == home_team else 1.0 - raw_probability


def build_no_spread_reference_observations(
    finalized_episodes: Sequence[FinalizedEpisodeReferenceInputV2],
    *,
    predictor: PinnedNoSpreadXGBoostPredictorV2,
) -> tuple[NoSpreadReferenceObservationV2, ...]:
    """Evaluate a pinned no-spread predictor for eligible episode boundaries.

    The model is used only by invoking the supplied callable.  No XGBoost
    training, asset selection, or use of ``vegas_wp`` exists in this path.
    """

    if not isinstance(predictor, PinnedNoSpreadXGBoostPredictorV2):
        raise ReferenceRegressionError("predictor must be pinned no-spread V2")
    if not isinstance(finalized_episodes, Sequence) or isinstance(
        finalized_episodes, (str, bytes, bytearray)
    ):
        raise ReferenceRegressionError(
            "finalized_episodes must be a sequence of V2 episode inputs"
        )
    observations: list[NoSpreadReferenceObservationV2] = []
    seen: set[tuple[str, str]] = set()
    for episode in finalized_episodes:
        if not isinstance(episode, FinalizedEpisodeReferenceInputV2):
            raise ReferenceRegressionError(
                "every finalized episode must be a V2 episode input"
            )
        (
            game_id,
            episode_id,
            home_team,
            away_team,
            focal_team,
            quarter,
            pre_time,
            post_time,
            receive_time,
        ) = _validate_episode(episode)
        if episode.focal_outcome != "win":
            raise ReferenceRegressionError("focal_outcome must be 'win'")
        grain = (game_id, episode_id)
        if grain in seen:
            raise ReferenceRegressionError("duplicate game/episode grain")
        seen.add(grain)

        support_status: ReferenceSupportStatusV2
        reason: str | None
        if not episode.finalized:
            support_status, reason = "UNSUPPORTED_UNFINALIZED", "episode_not_finalized"
        elif episode.nullified:
            support_status, reason = "EXCLUDED_NULLIFIED", "episode_nullified"
        elif episode.reversed:
            support_status, reason = "EXCLUDED_REVERSED", "episode_reversed"
        elif quarter > 4:
            support_status, reason = "EXCLUDED_OVERTIME", "overtime"
        elif episode.post_state is None:
            support_status, reason = "EXCLUDED_MISSING_POST_STATE", "missing_post_state"
        elif receive_time > pre_time or receive_time > post_time:
            support_status, reason = (
                "UNSUPPORTED_RECEIVE_2H_KO_AFTER_STATE",
                "receive_2h_ko_after_state",
            )
        else:
            support_status, reason = "SUPPORTED", None

        pre_payload: Mapping[str, object] | None = None
        post_payload: Mapping[str, object] | None = None
        pre_hash: str | None = None
        post_hash: str | None = None
        p_before: float | None = None
        p_after: float | None = None
        reference_delta: float | None = None
        if support_status == "SUPPORTED":
            pre_payload_material, pre_model_input = _feature_payload(
                episode=episode,
                predictor=predictor,
                state=episode.pre_state,
                state_time_ns=pre_time,
            )
            post_payload_material, post_model_input = _feature_payload(
                episode=episode,
                predictor=predictor,
                state=episode.post_state,
                state_time_ns=post_time,
            )
            pre_payload = _freeze_canonical_snapshot(
                pre_payload_material
            )
            post_payload = _freeze_canonical_snapshot(
                post_payload_material
            )
            if not isinstance(pre_payload, Mapping) or not isinstance(
                post_payload, Mapping
            ):
                raise AssertionError("frozen feature payload must be a mapping")
            pre_hash = _canonical_sha256(pre_payload)
            post_hash = _canonical_sha256(post_payload)
            p_before = _focal_probability(
                predictor,
                pre_model_input,
                focal_team=focal_team,
                home_team=home_team,
            )
            p_after = _focal_probability(
                predictor,
                post_model_input,
                focal_team=focal_team,
                home_team=home_team,
            )
            reference_delta = p_after - p_before

        material = {
            "game_id": game_id,
            "episode_id": episode_id,
            "home_team": home_team,
            "away_team": away_team,
            "focal_team": focal_team,
            "focal_outcome": episode.focal_outcome,
            "quarter": quarter,
            "pre_state_time_ns": pre_time,
            "post_state_time_ns": post_time,
            "receive_2h_ko_ns": receive_time,
            "predictor_asset_id": predictor.asset_id,
            "predictor_asset_version": predictor.asset_version,
            "predictor_asset_sha256": predictor.asset_sha256,
            "probability_outcome": predictor.probability_outcome,
            "pre_feature_payload": pre_payload,
            "post_feature_payload": post_payload,
            "pre_feature_sha256": pre_hash,
            "post_feature_sha256": post_hash,
            "p_before": p_before,
            "p_after": p_after,
            "reference_delta": reference_delta,
            "support_status": support_status,
            "exclusion_reason": reason,
        }
        observations.append(
            NoSpreadReferenceObservationV2(
                **material,
                observation_sha256=_canonical_sha256(
                    {
                        "schema_version": "NoSpreadReferenceObservationV2",
                        **material,
                    }
                ),
            )
        )
    return tuple(sorted(observations, key=lambda item: (item.game_id, item.episode_id)))


@dataclass(frozen=True, slots=True)
class ReferenceMarketProjectionInputV2:
    """Signed focal-market change at a venue and local-projection horizon."""

    game_id: str
    episode_id: str
    venue: str
    horizon_seconds: int
    pre_market_probability: float
    post_market_probability: float
    target_team: str
    target_outcome: str
    held_out: bool = False
    state_controls: Mapping[str, object] = field(default_factory=dict)
    liquidity_controls: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReferenceMarketProjectionObservationV2:
    game_id: str
    episode_id: str
    venue: str
    horizon_seconds: int
    target_team: str
    target_outcome: Literal["win"]
    pre_market_probability: float
    post_market_probability: float
    market_delta: float
    reference_delta: float | None
    completion_fraction: float | None
    completion_fraction_reason: str | None
    materiality_threshold: float
    held_out: bool
    reference_support_status: ReferenceSupportStatusV2
    state_controls: tuple[tuple[str, float], ...]
    liquidity_controls: tuple[tuple[str, float], ...]
    reference_observation_sha256: str


def build_reference_market_projection_inputs(
    reference_observations: Sequence[NoSpreadReferenceObservationV2],
    market_rows: Sequence[ReferenceMarketProjectionInputV2],
    *,
    materiality_threshold: float,
) -> tuple[ReferenceMarketProjectionObservationV2, ...]:
    """Join focal market changes to references and gate completion fractions."""

    threshold = _finite(materiality_threshold, field_name="materiality_threshold")
    if not 0.0 < threshold <= 1.0:
        raise ReferenceRegressionError("materiality_threshold must be in (0, 1]")
    observations = {
        (item.game_id, item.episode_id): item for item in reference_observations
    }
    if len(observations) != len(reference_observations):
        raise ReferenceRegressionError("reference observations have duplicate grain")
    output: list[ReferenceMarketProjectionObservationV2] = []
    seen: set[tuple[str, str, str, int]] = set()
    for row in market_rows:
        if not isinstance(row, ReferenceMarketProjectionInputV2):
            raise ReferenceRegressionError("every market row must be a V2 projection input")
        game_id = _text(row.game_id, field_name="game_id")
        episode_id = _text(row.episode_id, field_name="episode_id")
        venue = _text(row.venue, field_name="venue")
        horizon = _integer(
            row.horizon_seconds, field_name="horizon_seconds", minimum=1
        )
        if type(row.held_out) is not bool:
            raise ReferenceRegressionError("held_out must be boolean")
        grain = (game_id, episode_id, venue, horizon)
        if grain in seen:
            raise ReferenceRegressionError("duplicate market projection grain")
        seen.add(grain)
        reference = observations.get((game_id, episode_id))
        if reference is None:
            raise ReferenceRegressionError("market row references an unknown episode")
        target_team = _text(row.target_team, field_name="target_team")
        target_outcome = _text(row.target_outcome, field_name="target_outcome")
        if (
            target_team != reference.focal_team
            or target_outcome != reference.focal_outcome
        ):
            raise ReferenceRegressionError(
                "market target and outcome must match reference focal orientation"
            )
        before = _probability(
            row.pre_market_probability, field_name="pre_market_probability"
        )
        after = _probability(
            row.post_market_probability, field_name="post_market_probability"
        )
        reference_delta = reference.reference_delta
        if reference.support_status != "SUPPORTED" or reference_delta is None:
            completion_fraction = None
            completion_reason = "reference_unsupported"
        elif abs(reference_delta) <= threshold:
            completion_fraction = None
            completion_reason = "reference_below_materiality"
        else:
            completion_fraction = (after - before) / reference_delta
            completion_reason = None
        output.append(
            ReferenceMarketProjectionObservationV2(
                game_id=game_id,
                episode_id=episode_id,
                venue=venue,
                horizon_seconds=horizon,
                target_team=target_team,
                target_outcome="win",
                pre_market_probability=before,
                post_market_probability=after,
                market_delta=after - before,
                reference_delta=reference_delta,
                completion_fraction=completion_fraction,
                completion_fraction_reason=completion_reason,
                materiality_threshold=threshold,
                held_out=row.held_out,
                reference_support_status=reference.support_status,
                state_controls=_controls(row.state_controls, field_name="state_controls"),
                liquidity_controls=_controls(
                    row.liquidity_controls, field_name="liquidity_controls"
                ),
                reference_observation_sha256=reference.observation_sha256,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.venue,
                item.horizon_seconds,
                item.game_id,
                item.episode_id,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class RobustRegressionDiagnosticsV2:
    covariance_estimator: str | None
    standard_errors: tuple[float | None, ...]
    training_observation_count: int
    held_out_observation_count: int
    game_cluster_count: int
    design_rank: int
    residual_degrees_of_freedom: int
    r_squared: float | None


@dataclass(frozen=True, slots=True)
class LocalProjectionResidualV2:
    game_id: str
    episode_id: str
    held_out: bool
    market_delta: float
    predicted_market_delta: float | None
    residual: float | None
    reference_delta: float
    centered_pre_market_probability: float | None


@dataclass(frozen=True, slots=True)
class GameClusterBootstrapDrawV2:
    draw_index: int
    sampled_game_ids: tuple[str, ...]
    beta_reference_delta: float


@dataclass(frozen=True, slots=True)
class LocalProjectionRegressionResultV2:
    venue: str
    horizon_seconds: int
    status: RegressionStatusV2
    unsupported_reason: str | None
    game_count: int
    intercept: float | None
    beta_reference_delta: float | None
    beta_centered_pre_market_probability: float | None
    beta_optional_controls: tuple[tuple[str, float], ...]
    pre_market_probability_center: float | None
    robust_diagnostics: RobustRegressionDiagnosticsV2
    residuals: tuple[LocalProjectionResidualV2, ...]
    bootstrap_cluster_unit: Literal["game_id"]
    bootstrap_seed: int
    bootstrap_draws: tuple[GameClusterBootstrapDrawV2, ...]


def _fit_ols(
    design: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int] | None:
    coefficients, _sum_squared, rank, _singular = np.linalg.lstsq(
        design, outcome, rcond=None
    )
    if int(rank) != design.shape[1]:
        return None
    return coefficients, outcome - design @ coefficients, int(rank)


def _cluster_standard_errors(
    design: np.ndarray,
    residuals: np.ndarray,
    game_ids: Sequence[str],
) -> tuple[float | None, ...]:
    observation_count, parameter_count = design.shape
    games = sorted(set(game_ids))
    if len(games) < 2 or observation_count <= parameter_count:
        return tuple(None for _ in range(parameter_count))
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for game_id in games:
        positions = [index for index, value in enumerate(game_ids) if value == game_id]
        score = design[positions].T @ residuals[positions]
        meat += np.outer(score, score)
    correction = (len(games) / (len(games) - 1)) * (
        (observation_count - 1) / (observation_count - parameter_count)
    )
    covariance = correction * bread @ meat @ bread
    diagonal = np.diag(covariance)
    return tuple(
        math.sqrt(max(0.0, float(value))) if math.isfinite(float(value)) else None
        for value in diagonal
    )


def _bootstrap_draws(
    design: np.ndarray,
    outcome: np.ndarray,
    game_ids: Sequence[str],
    *,
    draw_count: int,
    seed: int,
) -> tuple[GameClusterBootstrapDrawV2, ...]:
    games = tuple(sorted(set(game_ids)))
    if not games:
        raise ReferenceRegressionError("game-cluster bootstrap requires training games")
    rng = np.random.default_rng(seed)
    game_positions = {
        game_id: np.asarray(
            [index for index, value in enumerate(game_ids) if value == game_id],
            dtype=int,
        )
        for game_id in games
    }
    draws: list[GameClusterBootstrapDrawV2] = []
    attempts = 0
    max_attempts = draw_count * 1_000
    while len(draws) < draw_count and attempts < max_attempts:
        attempts += 1
        selected = tuple(
            games[int(index)]
            for index in rng.integers(0, len(games), len(games))
        )
        positions = np.concatenate([game_positions[game_id] for game_id in selected])
        fitted = _fit_ols(design[positions], outcome[positions])
        if fitted is None or not math.isfinite(float(fitted[0][1])):
            continue
        draws.append(
            GameClusterBootstrapDrawV2(
                draw_index=len(draws),
                sampled_game_ids=selected,
                beta_reference_delta=float(fitted[0][1]),
            )
        )
    if len(draws) != draw_count:
        raise ReferenceRegressionError(
            "game-cluster bootstrap could not produce the requested number of "
            "valid coefficient draws within the deterministic retry budget"
        )
    return tuple(draws)


def _unsupported_result(
    *,
    venue: str,
    horizon_seconds: int,
    reason: str,
    residuals: tuple[LocalProjectionResidualV2, ...],
    training_count: int,
    held_out_count: int,
    game_count: int,
    control_count: int,
    bootstrap_seed: int,
) -> LocalProjectionRegressionResultV2:
    return LocalProjectionRegressionResultV2(
        venue=venue,
        horizon_seconds=horizon_seconds,
        status="UNSUPPORTED",
        unsupported_reason=reason,
        game_count=game_count,
        intercept=None,
        beta_reference_delta=None,
        beta_centered_pre_market_probability=None,
        beta_optional_controls=(),
        pre_market_probability_center=None,
        robust_diagnostics=RobustRegressionDiagnosticsV2(
            covariance_estimator=None,
            standard_errors=tuple(None for _ in range(3 + control_count)),
            training_observation_count=training_count,
            held_out_observation_count=held_out_count,
            game_cluster_count=game_count,
            design_rank=0,
            residual_degrees_of_freedom=0,
            r_squared=None,
        ),
        residuals=residuals,
        bootstrap_cluster_unit="game_id",
        bootstrap_seed=bootstrap_seed,
        bootstrap_draws=(),
    )


def fit_reference_local_projection_regressions(
    projection_inputs: Sequence[ReferenceMarketProjectionObservationV2],
    *,
    bootstrap_draw_count: int,
    bootstrap_seed: int,
) -> tuple[LocalProjectionRegressionResultV2, ...]:
    """Fit per-venue/horizon signed-change regressions without model retraining.

    The fitted equation is ``market_delta = intercept + beta * reference_delta
    + gamma * centered_pre_market_probability + optional controls``.  Every
    bootstrap draw resamples whole ``game_id`` clusters and refits this equation;
    it is not a parametric Beta distribution.
    """

    draw_count = _integer(
        bootstrap_draw_count, field_name="bootstrap_draw_count", minimum=1
    )
    seed = _integer(bootstrap_seed, field_name="bootstrap_seed", minimum=0)
    grouped: defaultdict[tuple[str, int], list[ReferenceMarketProjectionObservationV2]] = defaultdict(list)
    for item in projection_inputs:
        if not isinstance(item, ReferenceMarketProjectionObservationV2):
            raise ReferenceRegressionError(
                "projection_inputs must contain V2 projection observations"
            )
        grouped[(item.venue, item.horizon_seconds)].append(item)
    results: list[LocalProjectionRegressionResultV2] = []
    for group_index, ((venue, horizon), all_rows) in enumerate(sorted(grouped.items())):
        eligible = [
            row
            for row in all_rows
            if row.reference_support_status == "SUPPORTED"
            and row.reference_delta is not None
        ]
        train = [row for row in eligible if not row.held_out]
        held_out = [row for row in eligible if row.held_out]
        control_names = tuple(
            [f"state.{name}" for name, _value in eligible[0].state_controls]
            + [f"liquidity.{name}" for name, _value in eligible[0].liquidity_controls]
        ) if eligible else ()
        expected_control_names = (
            tuple(name for name, _value in eligible[0].state_controls),
            tuple(name for name, _value in eligible[0].liquidity_controls),
        ) if eligible else ((), ())
        if any(
            (
                tuple(name for name, _value in row.state_controls),
                tuple(name for name, _value in row.liquidity_controls),
            )
            != expected_control_names
            for row in eligible
        ):
            raise ReferenceRegressionError(
                "every venue/horizon regression needs the same control names"
            )
        train_game_ids = tuple(sorted({row.game_id for row in train}))
        residual_shells = tuple(
            LocalProjectionResidualV2(
                game_id=row.game_id,
                episode_id=row.episode_id,
                held_out=row.held_out,
                market_delta=row.market_delta,
                predicted_market_delta=None,
                residual=None,
                reference_delta=float(row.reference_delta),
                centered_pre_market_probability=None,
            )
            for row in eligible
        )
        if not eligible:
            results.append(
                _unsupported_result(
                    venue=venue,
                    horizon_seconds=horizon,
                    reason="no_supported_reference_observations",
                    residuals=(),
                    training_count=0,
                    held_out_count=0,
                    game_count=0,
                    control_count=0,
                    bootstrap_seed=seed,
                )
            )
            continue
        if len(train_game_ids) < 2:
            results.append(
                _unsupported_result(
                    venue=venue,
                    horizon_seconds=horizon,
                    reason="fewer_than_two_training_game_clusters",
                    residuals=residual_shells,
                    training_count=len(train),
                    held_out_count=len(held_out),
                    game_count=len(train_game_ids),
                    control_count=len(control_names),
                    bootstrap_seed=seed,
                )
            )
            continue
        parameter_count = 3 + len(control_names)
        if len(train) <= parameter_count:
            results.append(
                _unsupported_result(
                    venue=venue,
                    horizon_seconds=horizon,
                    reason="insufficient_training_observations",
                    residuals=residual_shells,
                    training_count=len(train),
                    held_out_count=len(held_out),
                    game_count=len(train_game_ids),
                    control_count=len(control_names),
                    bootstrap_seed=seed,
                )
            )
            continue
        center = float(np.mean([row.pre_market_probability for row in train]))

        def row_design(row: ReferenceMarketProjectionObservationV2) -> list[float]:
            assert row.reference_delta is not None
            return [
                1.0,
                float(row.reference_delta),
                row.pre_market_probability - center,
                *[value for _name, value in row.state_controls],
                *[value for _name, value in row.liquidity_controls],
            ]

        train_design = np.asarray([row_design(row) for row in train], dtype=float)
        train_outcome = np.asarray([row.market_delta for row in train], dtype=float)
        fitted = _fit_ols(train_design, train_outcome)
        if fitted is None:
            results.append(
                _unsupported_result(
                    venue=venue,
                    horizon_seconds=horizon,
                    reason="rank_deficient_design",
                    residuals=residual_shells,
                    training_count=len(train),
                    held_out_count=len(held_out),
                    game_count=len(train_game_ids),
                    control_count=len(control_names),
                    bootstrap_seed=seed,
                )
            )
            continue
        coefficients, train_residuals, rank = fitted
        all_design = np.asarray([row_design(row) for row in eligible], dtype=float)
        predictions = all_design @ coefficients
        residuals = tuple(
            LocalProjectionResidualV2(
                game_id=row.game_id,
                episode_id=row.episode_id,
                held_out=row.held_out,
                market_delta=row.market_delta,
                predicted_market_delta=float(prediction),
                residual=float(row.market_delta - prediction),
                reference_delta=float(row.reference_delta),
                centered_pre_market_probability=row.pre_market_probability - center,
            )
            for row, prediction in zip(eligible, predictions, strict=True)
        )
        total_sum_squares = float(
            np.square(train_outcome - np.mean(train_outcome)).sum()
        )
        residual_sum_squares = float(np.square(train_residuals).sum())
        r_squared = (
            None
            if total_sum_squares == 0.0
            else 1.0 - residual_sum_squares / total_sum_squares
        )
        standard_errors = _cluster_standard_errors(
            train_design,
            train_residuals,
            [row.game_id for row in train],
        )
        group_seed = int(
            np.random.SeedSequence([seed, group_index]).generate_state(1)[0]
        )
        draws = _bootstrap_draws(
            train_design,
            train_outcome,
            [row.game_id for row in train],
            draw_count=draw_count,
            seed=group_seed,
        )
        results.append(
            LocalProjectionRegressionResultV2(
                venue=venue,
                horizon_seconds=horizon,
                status="ESTIMATED",
                unsupported_reason=None,
                game_count=len(train_game_ids),
                intercept=float(coefficients[0]),
                beta_reference_delta=float(coefficients[1]),
                beta_centered_pre_market_probability=float(coefficients[2]),
                beta_optional_controls=tuple(
                    (name, float(value))
                    for name, value in zip(control_names, coefficients[3:], strict=True)
                ),
                pre_market_probability_center=center,
                robust_diagnostics=RobustRegressionDiagnosticsV2(
                    covariance_estimator="CR1_GAME_CLUSTER",
                    standard_errors=standard_errors,
                    training_observation_count=len(train),
                    held_out_observation_count=len(held_out),
                    game_cluster_count=len(train_game_ids),
                    design_rank=rank,
                    residual_degrees_of_freedom=len(train) - parameter_count,
                    r_squared=r_squared,
                ),
                residuals=residuals,
                bootstrap_cluster_unit="game_id",
                bootstrap_seed=seed,
                bootstrap_draws=draws,
            )
        )
    return tuple(results)


__all__ = [
    "FinalizedEpisodeReferenceInputV2",
    "GameClusterBootstrapDrawV2",
    "LocalProjectionRegressionResultV2",
    "LocalProjectionResidualV2",
    "NO_SPREAD_FEATURE_ALLOWLIST_V2",
    "NoSpreadFeatureAsOfV2",
    "NoSpreadFeatureStateV2",
    "NoSpreadReferenceObservationV2",
    "PinnedNoSpreadXGBoostPredictorV2",
    "ReferenceMarketProjectionInputV2",
    "ReferenceMarketProjectionObservationV2",
    "ReferenceRegressionError",
    "ReferenceSupportStatusV2",
    "RegressionStatusV2",
    "RobustRegressionDiagnosticsV2",
    "build_no_spread_reference_observations",
    "build_reference_market_projection_inputs",
    "fit_reference_local_projection_regressions",
]
