"""Offline NFL diagnostic reference values from pinned precomputed WP fields.

This module deliberately does not download, train, or ensemble a model.  It
consumes canonical episode/play feature rows whose ``vegas_wp_focal`` values
were already produced by a version-pinned nflfastR/fastrmodels asset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from numbers import Integral, Real
from typing import Any, Literal


SourceClass = Literal[
    "open_model",
    "provider_probability",
    "devig_odds",
    "exchange_reference",
]
ExecutionMode = Literal["PRECOMPUTED_FIELD", "MODEL_BYTES_VERIFIED"]
OutputModelBytesStatus = Literal[
    "OUTPUT_MODEL_BYTES_UNVERIFIED",
    "MODEL_BYTES_VERIFIED",
]
ReferenceSupportStatus = Literal[
    "SUPPORTED",
    "EPISODE_INELIGIBLE",
    "MODEL_SUPPORT_UNPROVEN",
    "MISSING_PRE_REFERENCE",
    "MISSING_POST_REFERENCE",
]
ComparisonStatus = Literal[
    "ELIGIBLE",
    "BELOW_MATERIALITY",
    "REFERENCE_NOT_SUPPORTED",
]

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_CLASSES = frozenset(
    {
        "open_model",
        "provider_probability",
        "devig_odds",
        "exchange_reference",
    }
)
_SUPPORT_STATUS_ORDER = (
    "SUPPORTED",
    "EPISODE_INELIGIBLE",
    "MODEL_SUPPORT_UNPROVEN",
    "MISSING_PRE_REFERENCE",
    "MISSING_POST_REFERENCE",
)


class ReferenceValueBuildError(ValueError):
    """Canonical inputs cannot produce an auditable reference-value row."""


def _canonical_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, set):
        return sorted(_canonical_value(child) for child in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReferenceValueBuildError(
                "canonical material cannot contain non-finite numbers"
            )
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ReferenceValueBuildError(
        f"canonical material contains unsupported {type(value).__name__}"
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ReferenceValueBuildError(f"{field} must be a sha256 digest")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ReferenceValueBuildError(f"{field} must be non-empty text")
    return value.strip()


def _probability(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReferenceValueBuildError(f"{field} must be a probability or null")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ReferenceValueBuildError(f"{field} must be between zero and one")
    return number


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReferenceValueBuildError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ReferenceValueBuildError(f"{field} must be finite")
    return number


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ReferenceValueBuildError(f"{field} must be an integer")
    return int(value)


def _truthy_indicator(value: object) -> bool:
    if value is None:
        return False
    if type(value) is bool:
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value) == 1.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _record(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise ReferenceValueBuildError("every feature row must be mapping-like")


def _records(value: object) -> list[dict[str, Any]]:
    if hasattr(value, "rows") and not isinstance(value, Mapping):
        rows = getattr(value, "rows")
        if isinstance(rows, Sequence):
            value = rows
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        try:
            converted = value.to_dict(orient="records")
        except TypeError:
            converted = None
        if isinstance(converted, list):
            value = converted
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ReferenceValueBuildError("feature rows must be a record sequence")
    return [_record(row) for row in value]


def _source_hash_pairs(
    source_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    pairs = (
        list(source_hashes.items())
        if isinstance(source_hashes, Mapping)
        else list(source_hashes)
    )
    canonical = tuple(
        sorted(
            (
                _text(name, field="source hash name"),
                _digest(digest, field="source hash"),
            )
            for name, digest in pairs
        )
    )
    if not canonical:
        raise ReferenceValueBuildError("input source hashes are required")
    if len({name for name, _digest_value in canonical}) != len(canonical):
        raise ReferenceValueBuildError("input source hash names must be unique")
    return canonical


def _source_hash_values(
    source_hashes: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    return tuple(sorted({digest for _name, digest in source_hashes}))


@dataclass(frozen=True, slots=True)
class ReferenceModelProvenanceV1:
    """Frozen model identity even when only precomputed output is available."""

    model_id: str
    model_version: str
    model_sha256: str
    source_class: SourceClass
    pit_status: str
    license_status: str
    calibration_status: str
    execution_mode: ExecutionMode
    precomputed_field: str | None
    model_bytes_verified: bool
    output_model_bytes_status: OutputModelBytesStatus
    formal_registry_gate_passed: bool

    def __post_init__(self) -> None:
        _text(self.model_id, field="model_id")
        _text(self.model_version, field="model_version")
        _digest(self.model_sha256, field="model_sha256")
        if self.source_class not in _SOURCE_CLASSES:
            raise ReferenceValueBuildError("source_class is unsupported")
        for value, field in (
            (self.pit_status, "pit_status"),
            (self.license_status, "license_status"),
            (self.calibration_status, "calibration_status"),
        ):
            _text(value, field=field)
        if self.execution_mode not in {
            "PRECOMPUTED_FIELD",
            "MODEL_BYTES_VERIFIED",
        }:
            raise ReferenceValueBuildError("execution_mode is unsupported")
        if type(self.model_bytes_verified) is not bool:
            raise ReferenceValueBuildError(
                "model_bytes_verified must be boolean"
            )
        if type(self.formal_registry_gate_passed) is not bool:
            raise ReferenceValueBuildError(
                "formal_registry_gate_passed must be boolean"
            )
        if self.execution_mode == "PRECOMPUTED_FIELD":
            _text(self.precomputed_field, field="precomputed_field")
            if self.model_bytes_verified:
                raise ReferenceValueBuildError(
                    "precomputed mode cannot claim model bytes were verified"
                )
            if (
                self.output_model_bytes_status
                != "OUTPUT_MODEL_BYTES_UNVERIFIED"
            ):
                raise ReferenceValueBuildError(
                    "precomputed mode must explicitly mark output model "
                    "bytes unverified"
                )
        else:
            _text(self.precomputed_field, field="precomputed_field")
            if not self.model_bytes_verified:
                raise ReferenceValueBuildError(
                    "verified model-bytes mode requires verified bytes"
                )
            if self.output_model_bytes_status != "MODEL_BYTES_VERIFIED":
                raise ReferenceValueBuildError(
                    "verified mode must mark model bytes verified"
                )

    @classmethod
    def precomputed(
        cls,
        *,
        model_id: str,
        model_version: str,
        declared_output_model_sha256: str,
        field_name: str,
        source_class: SourceClass,
        pit_status: str,
        license_status: str,
        calibration_status: str,
        formal_registry_gate_passed: bool,
    ) -> ReferenceModelProvenanceV1:
        return cls(
            model_id=model_id,
            model_version=model_version,
            model_sha256=declared_output_model_sha256,
            source_class=source_class,
            pit_status=pit_status,
            license_status=license_status,
            calibration_status=calibration_status,
            execution_mode="PRECOMPUTED_FIELD",
            precomputed_field=field_name,
            model_bytes_verified=False,
            output_model_bytes_status="OUTPUT_MODEL_BYTES_UNVERIFIED",
            formal_registry_gate_passed=formal_registry_gate_passed,
        )

    @classmethod
    def from_model_bytes(
        cls,
        *,
        model_id: str,
        model_version: str,
        model_sha256: str,
        model_bytes: bytes,
        source_class: SourceClass,
        pit_status: str,
        license_status: str,
        calibration_status: str,
        formal_registry_gate_passed: bool,
        field_name: str = "vegas_wp_focal",
    ) -> ReferenceModelProvenanceV1:
        if type(model_bytes) is not bytes or not model_bytes:
            raise ReferenceValueBuildError("model_bytes must be non-empty bytes")
        observed = "sha256:" + hashlib.sha256(model_bytes).hexdigest()
        expected = _digest(model_sha256, field="model_sha256")
        if observed != expected:
            raise ReferenceValueBuildError(
                "model bytes SHA-256 does not match the pinned digest"
            )
        return cls(
            model_id=model_id,
            model_version=model_version,
            model_sha256=expected,
            source_class=source_class,
            pit_status=pit_status,
            license_status=license_status,
            calibration_status=calibration_status,
            execution_mode="MODEL_BYTES_VERIFIED",
            precomputed_field=field_name,
            model_bytes_verified=True,
            output_model_bytes_status="MODEL_BYTES_VERIFIED",
            formal_registry_gate_passed=formal_registry_gate_passed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ReferenceModelProvenanceV1",
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }


@dataclass(frozen=True, slots=True)
class ReferenceValueObservationV1:
    """One finalized episode with pre/post diagnostic win probabilities."""

    game_id: str
    episode_id: str
    episode_play_ids: tuple[str, ...]
    pre_reference_play_id: str
    post_reference_play_id: str | None
    home_team: str
    away_team: str
    focal_team: str
    quarter: int
    pre_probability_home: float | None
    post_probability_home: float | None
    reference_delta_home: float | None
    pre_probability_focal: float | None
    post_probability_focal: float | None
    reference_delta_focal: float | None
    source_class: SourceClass
    model_id: str
    model_version: str
    model_sha256: str
    execution_mode: ExecutionMode
    probability_field: str
    model_bytes_verified: bool
    output_model_bytes_status: OutputModelBytesStatus
    formal_registry_gate_passed: bool
    model_input_sha256: str
    pit_status: str
    license_status: str
    calibration_status: str
    support_status: ReferenceSupportStatus
    formal_comparison_eligible: bool
    exclusion_reasons: tuple[str, ...]
    source_hashes: tuple[str, ...]
    observation_sha256: str
    claim_boundary: str = "DIAGNOSTIC_REFERENCE_NOT_GROUND_TRUTH"

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.game_id, "game_id"),
            (self.episode_id, "episode_id"),
            (self.pre_reference_play_id, "pre_reference_play_id"),
            (self.home_team, "home_team"),
            (self.away_team, "away_team"),
            (self.focal_team, "focal_team"),
            (self.model_id, "model_id"),
            (self.model_version, "model_version"),
            (self.probability_field, "probability_field"),
            (self.pit_status, "pit_status"),
            (self.license_status, "license_status"),
            (self.calibration_status, "calibration_status"),
            (self.claim_boundary, "claim_boundary"),
        ):
            _text(value, field=field_name)
        if self.home_team == self.away_team or self.focal_team not in {
            self.home_team,
            self.away_team,
        }:
            raise ReferenceValueBuildError(
                "reference observation team orientation is invalid"
            )
        if not self.episode_play_ids or len(set(self.episode_play_ids)) != len(
            self.episode_play_ids
        ):
            raise ReferenceValueBuildError(
                "episode_play_ids must be non-empty and unique"
            )
        for play_id in self.episode_play_ids:
            _text(play_id, field="episode_play_id")
        if self.post_reference_play_id is not None:
            _text(
                self.post_reference_play_id,
                field="post_reference_play_id",
            )
        if (
            isinstance(self.quarter, bool)
            or not isinstance(self.quarter, int)
            or self.quarter <= 0
        ):
            raise ReferenceValueBuildError("quarter must be positive")
        for value, field_name in (
            (self.pre_probability_home, "pre_probability_home"),
            (self.post_probability_home, "post_probability_home"),
            (self.pre_probability_focal, "pre_probability_focal"),
            (self.post_probability_focal, "post_probability_focal"),
        ):
            _probability(value, field=field_name)
        for before, after, delta, field_name in (
            (
                self.pre_probability_home,
                self.post_probability_home,
                self.reference_delta_home,
                "reference_delta_home",
            ),
            (
                self.pre_probability_focal,
                self.post_probability_focal,
                self.reference_delta_focal,
                "reference_delta_focal",
            ),
        ):
            if before is None or after is None:
                if delta is not None:
                    raise ReferenceValueBuildError(
                        f"{field_name} requires pre and post probabilities"
                    )
            else:
                observed_delta = _finite(delta, field=field_name)
                if not math.isclose(
                    observed_delta,
                    after - before,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ReferenceValueBuildError(
                        f"{field_name} disagrees with pre/post probabilities"
                    )
        if self.source_class not in _SOURCE_CLASSES:
            raise ReferenceValueBuildError("source_class is unsupported")
        if self.execution_mode not in {
            "PRECOMPUTED_FIELD",
            "MODEL_BYTES_VERIFIED",
        }:
            raise ReferenceValueBuildError("execution_mode is unsupported")
        if type(self.model_bytes_verified) is not bool:
            raise ReferenceValueBuildError(
                "model_bytes_verified must be boolean"
            )
        if self.output_model_bytes_status not in {
            "OUTPUT_MODEL_BYTES_UNVERIFIED",
            "MODEL_BYTES_VERIFIED",
        }:
            raise ReferenceValueBuildError(
                "output_model_bytes_status is unsupported"
            )
        if self.model_bytes_verified != (
            self.output_model_bytes_status == "MODEL_BYTES_VERIFIED"
        ):
            raise ReferenceValueBuildError(
                "output model bytes status disagrees with verification flag"
            )
        if type(self.formal_registry_gate_passed) is not bool:
            raise ReferenceValueBuildError(
                "formal_registry_gate_passed must be boolean"
            )
        _digest(self.model_sha256, field="model_sha256")
        _digest(self.model_input_sha256, field="model_input_sha256")
        if self.support_status not in _SUPPORT_STATUS_ORDER:
            raise ReferenceValueBuildError("support_status is unsupported")
        if type(self.formal_comparison_eligible) is not bool:
            raise ReferenceValueBuildError(
                "formal_comparison_eligible must be boolean"
            )
        if self.formal_comparison_eligible != (
            self.support_status == "SUPPORTED"
            and self.formal_registry_gate_passed
        ):
            raise ReferenceValueBuildError(
                "formal comparison eligibility disagrees with support status"
            )
        if self.exclusion_reasons != tuple(sorted(set(self.exclusion_reasons))):
            raise ReferenceValueBuildError(
                "exclusion_reasons must be sorted and unique"
            )
        for reason in self.exclusion_reasons:
            _text(reason, field="exclusion_reason")
        if self.source_hashes != tuple(sorted(set(self.source_hashes))):
            raise ReferenceValueBuildError(
                "source_hashes must be sorted and unique"
            )
        for digest in self.source_hashes:
            _digest(digest, field="source_hash")
        _digest(self.observation_sha256, field="observation_sha256")
        material = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "observation_sha256"
        }
        if self.observation_sha256 != _observation_sha256(material):
            raise ReferenceValueBuildError(
                "observation_sha256 does not match canonical material"
            )

    def to_dict(self) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        result["schema_version"] = "ReferenceValueObservationV1"
        result["episode_play_ids"] = list(self.episode_play_ids)
        result["exclusion_reasons"] = list(self.exclusion_reasons)
        result["source_hashes"] = list(self.source_hashes)
        return result


@dataclass(frozen=True, slots=True)
class ReferenceValueSummaryV1:
    total_episodes: int
    formal_comparison_eligible: int
    support_status_counts: tuple[tuple[str, int], ...]
    game_count: int
    model_sha256: str
    input_source_hashes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ReferenceValueSummaryV1",
            "total_episodes": self.total_episodes,
            "formal_comparison_eligible": self.formal_comparison_eligible,
            "support_status_counts": {
                name: count for name, count in self.support_status_counts
            },
            "game_count": self.game_count,
            "model_sha256": self.model_sha256,
            "input_source_hashes": {
                name: digest for name, digest in self.input_source_hashes
            },
        }


@dataclass(frozen=True, slots=True)
class ReferenceValueBundleV1:
    observations: tuple[ReferenceValueObservationV1, ...]
    summary: ReferenceValueSummaryV1
    model: ReferenceModelProvenanceV1
    observations_sha256: str

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.observations]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ReferenceValueBundleV1",
            "model": self.model.to_dict(),
            "summary": self.summary.to_dict(),
            "observations_sha256": self.observations_sha256,
            "observations": self.to_records(),
        }


@dataclass(frozen=True, slots=True)
class ReferenceMarketDiagnosticV1:
    game_id: str
    episode_id: str
    logical_market_id: str
    venue: str
    horizon_seconds: int
    target_team: str
    market_delta: float
    reference_delta_target: float | None
    reference_gap: float | None
    completion_fraction: float | None
    materiality_threshold: float
    comparison_status: ComparisonStatus
    reference_observation_sha256: str
    source_hashes: tuple[str, ...]
    diagnostic_sha256: str
    claim_boundary: str = "DIAGNOSTIC_REFERENCE_NOT_GROUND_TRUTH"

    def to_dict(self) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        result["schema_version"] = "ReferenceMarketDiagnosticV1"
        result["source_hashes"] = list(self.source_hashes)
        return result


@dataclass(frozen=True, slots=True)
class ReferenceMarketDiagnosticBundleV1:
    rows: tuple[ReferenceMarketDiagnosticV1, ...]
    rows_sha256: str
    materiality_threshold: float

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]


def _row_identity(row: Mapping[str, object]) -> tuple[str, int, str]:
    return (
        _text(row.get("game_id"), field="game_id"),
        _integer(row.get("order_sequence"), field="order_sequence"),
        _text(row.get("play_id"), field="play_id"),
    )


def _episode_ineligible(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    reasons: set[str] = set()
    for row in rows:
        event_flags = row.get("event_flags", ())
        flags = (
            {str(value).lower() for value in event_flags}
            if isinstance(event_flags, Sequence)
            and not isinstance(event_flags, (str, bytes, bytearray))
            else set()
        )
        play_type = str(row.get("play_type") or "").strip().lower()
        review_result = str(
            row.get("review_result")
            or row.get("replay_or_challenge_result")
            or ""
        ).strip().lower()
        if play_type == "no_play" or _truthy_indicator(row.get("no_play")):
            reasons.add("no_play")
        if (
            _truthy_indicator(row.get("episode_nullified"))
            or _truthy_indicator(row.get("nullified"))
        ):
            reasons.add("nullified")
        if (
            _truthy_indicator(row.get("episode_audit_only"))
            or _truthy_indicator(row.get("audit_only"))
        ):
            reasons.add("audit_only")
        if "reversed" in review_result or "reversed" in flags:
            reasons.add("reversed")
    return tuple(sorted(reasons))


def _home_probability(
    row: Mapping[str, object],
    *,
    field_name: str,
) -> float | None:
    probability = _probability(row.get(field_name), field=field_name)
    if probability is None:
        return None
    home = _text(row.get("home_team"), field="home_team")
    away = _text(row.get("away_team"), field="away_team")
    focal = _text(row.get("focal_team"), field="focal_team")
    if home == away or focal not in {home, away}:
        raise ReferenceValueBuildError("feature row team orientation is invalid")
    return probability if focal == home else 1.0 - probability


def _observation_sha256(material: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "schema_version": "ReferenceValueObservationV1",
            **dict(material),
        }
    )


def build_reference_value_bundle(
    feature_rows: object,
    *,
    model: ReferenceModelProvenanceV1,
    input_source_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
) -> ReferenceValueBundleV1:
    """Build episode-boundary reference values without executing a model."""

    if not isinstance(model, ReferenceModelProvenanceV1):
        raise ReferenceValueBuildError("model provenance must be V1")
    if model.precomputed_field is None:
        raise ReferenceValueBuildError(
            "model provenance must name the frozen probability field"
        )
    source_pairs = _source_hash_pairs(input_source_hashes)
    source_values = _source_hash_values(source_pairs)
    raw_rows = _records(feature_rows)
    if not raw_rows:
        raise ReferenceValueBuildError("feature rows must not be empty")
    ordered = sorted(raw_rows, key=_row_identity)
    grains = [
        (
            _text(row.get("game_id"), field="game_id"),
            _text(row.get("play_id"), field="play_id"),
        )
        for row in ordered
    ]
    if len(set(grains)) != len(grains):
        raise ReferenceValueBuildError("duplicate game/play grain")

    grouped: list[list[dict[str, Any]]] = []
    group_keys: list[tuple[str, str]] = []
    group_index: dict[tuple[str, str], int] = {}
    for row in ordered:
        game_id = _text(row.get("game_id"), field="game_id")
        episode_id = _text(row.get("episode_id"), field="episode_id")
        key = (game_id, episode_id)
        index = group_index.get(key)
        if index is None:
            index = len(grouped)
            group_index[key] = index
            group_keys.append(key)
            grouped.append([])
        grouped[index].append(row)

    observations: list[ReferenceValueObservationV1] = []
    for index, members in enumerate(grouped):
        game_id, episode_id = group_keys[index]
        first = members[0]
        home_team = _text(first.get("home_team"), field="home_team")
        away_team = _text(first.get("away_team"), field="away_team")
        focal_team = _text(first.get("focal_team"), field="focal_team")
        if home_team == away_team or focal_team not in {home_team, away_team}:
            raise ReferenceValueBuildError("episode team orientation is invalid")
        for member in members:
            if (
                _text(member.get("home_team"), field="home_team") != home_team
                or _text(member.get("away_team"), field="away_team") != away_team
            ):
                raise ReferenceValueBuildError(
                    "episode members disagree on game teams"
                )
        quarter = _integer(first.get("quarter"), field="quarter")
        if quarter <= 0:
            raise ReferenceValueBuildError("quarter must be positive")
        ineligible_reasons = _episode_ineligible(members)

        post_row: Mapping[str, object] | None = None
        for next_members in grouped[index + 1 :]:
            if _text(next_members[0].get("game_id"), field="game_id") != game_id:
                break
            if _episode_ineligible(next_members):
                continue
            candidate = next_members[0]
            if (
                _home_probability(candidate, field_name=model.precomputed_field)
                is not None
            ):
                post_row = candidate
                break

        pre_home = _home_probability(first, field_name=model.precomputed_field)
        post_home = (
            None
            if post_row is None
            else _home_probability(post_row, field_name=model.precomputed_field)
        )
        post_quarter = (
            None
            if post_row is None
            else _integer(post_row.get("quarter"), field="post quarter")
        )
        if ineligible_reasons:
            support_status: ReferenceSupportStatus = "EPISODE_INELIGIBLE"
            pre_home = None
            post_home = None
        elif quarter > 4 or (post_quarter is not None and post_quarter > 4):
            support_status = "MODEL_SUPPORT_UNPROVEN"
        elif pre_home is None:
            support_status = "MISSING_PRE_REFERENCE"
        elif post_home is None:
            support_status = "MISSING_POST_REFERENCE"
        else:
            support_status = "SUPPORTED"
        formal_eligible = (
            support_status == "SUPPORTED"
            and model.formal_registry_gate_passed
        )
        reference_delta_home = (
            None
            if pre_home is None or post_home is None
            else post_home - pre_home
        )
        pre_focal = (
            None
            if pre_home is None
            else (pre_home if focal_team == home_team else 1.0 - pre_home)
        )
        post_focal = (
            None
            if post_home is None
            else (post_home if focal_team == home_team else 1.0 - post_home)
        )
        reference_delta_focal = (
            None
            if pre_focal is None or post_focal is None
            else post_focal - pre_focal
        )
        member_material = [
            {
                "game_id": game_id,
                "episode_id": episode_id,
                "play_id": _text(member.get("play_id"), field="play_id"),
                "order_sequence": _integer(
                    member.get("order_sequence"), field="order_sequence"
                ),
                "focal_team": _text(
                    member.get("focal_team"), field="focal_team"
                ),
                model.precomputed_field: _probability(
                    member.get(model.precomputed_field),
                    field=model.precomputed_field,
                ),
            }
            for member in members
        ]
        model_input_sha256 = _canonical_sha256(
            {
                "model_sha256": model.model_sha256,
                "precomputed_field": model.precomputed_field,
                "input_source_hashes": [
                    {"name": name, "sha256": digest}
                    for name, digest in source_pairs
                ],
                "episode_members": member_material,
                "post_reference": (
                    None
                    if post_row is None
                    else {
                        "play_id": _text(
                            post_row.get("play_id"), field="play_id"
                        ),
                        model.precomputed_field: _probability(
                            post_row.get(model.precomputed_field),
                            field=model.precomputed_field,
                        ),
                    }
                ),
            }
        )
        material: dict[str, object] = {
            "game_id": game_id,
            "episode_id": episode_id,
            "episode_play_ids": tuple(
                _text(member.get("play_id"), field="play_id")
                for member in members
            ),
            "pre_reference_play_id": _text(
                first.get("play_id"), field="play_id"
            ),
            "post_reference_play_id": (
                None
                if post_row is None
                else _text(post_row.get("play_id"), field="play_id")
            ),
            "home_team": home_team,
            "away_team": away_team,
            "focal_team": focal_team,
            "quarter": quarter,
            "pre_probability_home": pre_home,
            "post_probability_home": post_home,
            "reference_delta_home": reference_delta_home,
            "pre_probability_focal": pre_focal,
            "post_probability_focal": post_focal,
            "reference_delta_focal": reference_delta_focal,
            "source_class": model.source_class,
            "model_id": model.model_id,
            "model_version": model.model_version,
            "model_sha256": model.model_sha256,
            "execution_mode": model.execution_mode,
            "probability_field": model.precomputed_field,
            "model_bytes_verified": model.model_bytes_verified,
            "output_model_bytes_status": (
                model.output_model_bytes_status
            ),
            "formal_registry_gate_passed": (
                model.formal_registry_gate_passed
            ),
            "model_input_sha256": model_input_sha256,
            "pit_status": model.pit_status,
            "license_status": model.license_status,
            "calibration_status": model.calibration_status,
            "support_status": support_status,
            "formal_comparison_eligible": formal_eligible,
            "exclusion_reasons": ineligible_reasons,
            "source_hashes": source_values,
            "claim_boundary": "DIAGNOSTIC_REFERENCE_NOT_GROUND_TRUTH",
        }
        observations.append(
            ReferenceValueObservationV1(
                **material,
                observation_sha256=_observation_sha256(material),
            )
        )

    # ``grouped`` is created from rows already sorted by the canonical
    # game/order-sequence/play key, so preserving group order also supports
    # non-numeric provider play IDs without inventing another ordering rule.
    canonical_observations = tuple(observations)
    counts = Counter(row.support_status for row in canonical_observations)
    summary = ReferenceValueSummaryV1(
        total_episodes=len(canonical_observations),
        formal_comparison_eligible=sum(
            row.formal_comparison_eligible for row in canonical_observations
        ),
        support_status_counts=tuple(
            (status, counts[status])
            for status in _SUPPORT_STATUS_ORDER
            if counts[status]
        ),
        game_count=len({row.game_id for row in canonical_observations}),
        model_sha256=model.model_sha256,
        input_source_hashes=source_pairs,
    )
    observations_sha256 = _canonical_sha256(
        {
            "schema_version": "ReferenceValueBundleV1",
            "model": model.to_dict(),
            "summary": summary.to_dict(),
            "observations": [row.to_dict() for row in canonical_observations],
        }
    )
    return ReferenceValueBundleV1(
        observations=canonical_observations,
        summary=summary,
        model=model,
        observations_sha256=observations_sha256,
    )


def build_reference_market_diagnostics(
    reference_bundle: ReferenceValueBundleV1,
    market_reaction_rows: object,
    *,
    materiality_threshold: float,
) -> ReferenceMarketDiagnosticBundleV1:
    """Join explicitly oriented market deltas to diagnostic references."""

    if not isinstance(reference_bundle, ReferenceValueBundleV1):
        raise ReferenceValueBuildError("reference_bundle must be V1")
    threshold = _finite(
        materiality_threshold, field="materiality_threshold"
    )
    if threshold <= 0.0 or threshold > 1.0:
        raise ReferenceValueBuildError(
            "materiality_threshold must be in (0, 1]"
        )
    observations = {
        (row.game_id, row.episode_id): row
        for row in reference_bundle.observations
    }
    reaction_records = _records(market_reaction_rows)
    output: list[ReferenceMarketDiagnosticV1] = []
    seen: set[tuple[str, str, str, str, int, str]] = set()
    for source in reaction_records:
        game_id = _text(source.get("game_id"), field="game_id")
        episode_id = _text(source.get("episode_id"), field="episode_id")
        logical_market_id = _text(
            source.get("logical_market_id"), field="logical_market_id"
        )
        venue = _text(source.get("venue"), field="venue")
        horizon = _integer(
            source.get("horizon_seconds"), field="horizon_seconds"
        )
        if horizon <= 0:
            raise ReferenceValueBuildError("horizon_seconds must be positive")
        target_team = _text(source.get("target_team"), field="target_team")
        market_delta = _finite(source.get("market_delta"), field="market_delta")
        key = (
            game_id,
            episode_id,
            logical_market_id,
            venue,
            horizon,
            target_team,
        )
        if key in seen:
            raise ReferenceValueBuildError(
                "duplicate reference diagnostic grain"
            )
        seen.add(key)
        observation = observations.get((game_id, episode_id))
        if observation is None:
            raise ReferenceValueBuildError(
                "market reaction references an unknown episode"
            )
        if target_team not in {observation.home_team, observation.away_team}:
            raise ReferenceValueBuildError(
                "market target_team is not one of the game teams"
            )
        reference_delta_target = observation.reference_delta_home
        if (
            reference_delta_target is not None
            and target_team == observation.away_team
        ):
            reference_delta_target = -reference_delta_target
        reference_supported = (
            observation.support_status == "SUPPORTED"
            and reference_delta_target is not None
        )
        if not observation.formal_comparison_eligible:
            comparison_status: ComparisonStatus = "REFERENCE_NOT_SUPPORTED"
        elif (
            reference_delta_target is None
            or abs(reference_delta_target) < threshold
        ):
            comparison_status = "BELOW_MATERIALITY"
        else:
            comparison_status = "ELIGIBLE"
        reference_gap = (
            market_delta - reference_delta_target
            if reference_supported
            else None
        )
        if comparison_status == "ELIGIBLE":
            assert reference_delta_target is not None
            completion_fraction = market_delta / reference_delta_target
        else:
            completion_fraction = None
        reaction_hashes_raw = source.get("source_hashes")
        if not isinstance(reaction_hashes_raw, Sequence) or isinstance(
            reaction_hashes_raw, (str, bytes, bytearray)
        ):
            raise ReferenceValueBuildError(
                "market reaction source_hashes are required"
            )
        reaction_hashes = tuple(
            _digest(value, field="market source hash")
            for value in reaction_hashes_raw
        )
        if not reaction_hashes:
            raise ReferenceValueBuildError(
                "market reaction source_hashes are required"
            )
        combined_hashes = tuple(
            sorted({*observation.source_hashes, *reaction_hashes})
        )
        material: dict[str, object] = {
            "game_id": game_id,
            "episode_id": episode_id,
            "logical_market_id": logical_market_id,
            "venue": venue,
            "horizon_seconds": horizon,
            "target_team": target_team,
            "market_delta": market_delta,
            "reference_delta_target": reference_delta_target,
            "reference_gap": reference_gap,
            "completion_fraction": completion_fraction,
            "materiality_threshold": threshold,
            "comparison_status": comparison_status,
            "reference_observation_sha256": observation.observation_sha256,
            "source_hashes": combined_hashes,
            "claim_boundary": "DIAGNOSTIC_REFERENCE_NOT_GROUND_TRUTH",
        }
        output.append(
            ReferenceMarketDiagnosticV1(
                **material,
                diagnostic_sha256=_canonical_sha256(
                    {
                        "schema_version": "ReferenceMarketDiagnosticV1",
                        **material,
                    }
                ),
            )
        )
    canonical_rows = tuple(
        sorted(
            output,
            key=lambda row: (
                row.game_id,
                row.episode_id,
                row.logical_market_id,
                row.venue,
                row.horizon_seconds,
                row.target_team,
            ),
        )
    )
    return ReferenceMarketDiagnosticBundleV1(
        rows=canonical_rows,
        rows_sha256=_canonical_sha256(
            {
                "schema_version": "ReferenceMarketDiagnosticBundleV1",
                "materiality_threshold": threshold,
                "rows": [row.to_dict() for row in canonical_rows],
            }
        ),
        materiality_threshold=threshold,
    )


__all__ = [
    "ReferenceMarketDiagnosticBundleV1",
    "ReferenceMarketDiagnosticV1",
    "ReferenceModelProvenanceV1",
    "ReferenceValueBuildError",
    "ReferenceValueBundleV1",
    "ReferenceValueObservationV1",
    "ReferenceValueSummaryV1",
    "build_reference_market_diagnostics",
    "build_reference_value_bundle",
]
