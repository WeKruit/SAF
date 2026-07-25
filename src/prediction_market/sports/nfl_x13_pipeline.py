"""Verified end-to-end integration and publication for NFL X-13.

The public entry point deliberately re-opens every evidence root.  It never
turns caller-supplied ``PASS`` flags into proof and never weakens the
``PRELIMINARY_SOURCE_TIME_ONLY`` claim boundary.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import resource
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from prediction_market.compliance import (
    ComplianceRegistryError,
    load_data_license_register,
)
from prediction_market.experiments import (
    ExperimentRegistryError,
    load_experiment_registry,
)
from prediction_market.sports.nfl_x13_association import (
    AssociationResultV1,
    AssociationEvidenceError,
    X13_REGISTERED_ANALYSIS_LOCK_V1,
    X13AssociationStreamV1,
    _semantic_orientation_key,
    audit_layer_m,
    iter_x13_associations,
    prepare_x13_association_stream,
)
from prediction_market.sports.nfl_x13_batch import (
    AuxiliaryArtifactV1,
    X13BatchPublishError,
    publish_x13_batch,
    verify_published_batch,
)
from prediction_market.sports.nfl_x13_capture import (
    BatchCaptureResult,
    CaptureReceipt,
    CaptureTarget,
    CapturePointer,
    GameCaptureResult,
    ImmutableRequestManifest,
    POLYMARKET_TRADE_LIMIT,
    _plans_sha256,
)
from prediction_market.sports.nfl_x13_game_build import (
    X13GameBuildError,
    X13GameStateBuild,
    verify_x13_game_state_publication,
    x13_game_state_manifest_semantic_sha256,
)
from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
    X13MarketError,
    _observation_material,
    deduplicate_observations,
    derive_complement,
    normalize_kalshi_candle_bbo,
    normalize_kalshi_trade,
    normalize_polymarket_trade,
)
from prediction_market.sports.nfl_x13_replay import X13GameLedgerV1
from prediction_market.sports.nfl_x13_spec import (
    X13_BATCH_SPEC,
    X13_EXPERIMENT_ID,
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
    X13_STATUS,
    X13_VENUE_FAMILY_CATALOG,
    MarketContractV1,
    canonical_sha256,
)
from prediction_market.sports.nfl_x13_universe import (
    CompositeClassificationV1,
    CompositeLegV1,
    ExcludedContractV1,
    KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1,
    KalshiSeriesRegistryProofV1,
    PolymarketTokenBindingV1,
    SourceTimeWindowV1,
    UniverseContractV1,
    UniverseGameV1,
    X13UniverseBatchV1,
)
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BUILDER_VERSION = "nfl-x13-pipeline-v2"
_REQUIRED_SCOPE = "preliminary_source_time_only"
_REQUIRED_LICENSE_REFS = ("I-018", "O-001", "O-003", "O-009")
_X13_PREREGISTERED_DATASET_IDS = (
    "DS-KALSHI-HISTORICAL",
    "DS-NFLVERSE",
    "DS-NFLVERSE-PARTICIPATION",
    "DS-POLYMARKET-PUBLIC",
)
X13_EVALUATION_CODE_PATHS = (
    "src/prediction_market/experiments.py",
    "src/prediction_market/sports/nfl_x13_association.py",
    "src/prediction_market/sports/nfl_x13_batch.py",
    "src/prediction_market/sports/nfl_x13_capture.py",
    "src/prediction_market/sports/nfl_x13_dataset.py",
    "src/prediction_market/sports/nfl_x13_game_build.py",
    "src/prediction_market/sports/nfl_x13_market.py",
    "src/prediction_market/sports/nfl_x13_pipeline.py",
    "src/prediction_market/sports/nfl_x13_replay.py",
    "src/prediction_market/sports/nfl_x13_spec.py",
    "src/prediction_market/sports/nfl_x13_universe.py",
)
_RESEARCH_OPERATIONAL_USE = frozenset({"APPROVED", "RESEARCH_ONLY"})
_POLYMARKET_DATA_TRADES_PATH = "/trades"
_KALSHI_HISTORICAL_MARKETS_PATH = "/historical/markets"
_KALSHI_HISTORICAL_TRADES_PATH = "/historical/trades"
_KALSHI_HISTORICAL_CUTOFF_PATH = "/historical/cutoff"
_FIELD_COLORS = ("#2563eb", "#ef4444")
_FROZEN_UNIVERSE_ID = (
    "sha256:01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
)
_FROZEN_ASSOCIATION_COUNT = 48_624_912
_ASSOCIATION_ROW_GROUP_SIZE = 65_536
_ASSOCIATION_PILOT_ROW_LIMIT = 65_536
_PRESENTATION_ROW_LIMIT_PER_GAME = 5_000
_ASSOCIATION_P95_UPLIFT_NUMERATOR = 3
_ASSOCIATION_P95_UPLIFT_DENOMINATOR = 2
_ASSOCIATION_MIN_BYTES_PER_ROW = 32
_GIB = 1024**3
_TEMP_HEADROOM_BYTES = 1 * _GIB
_UNTOUCHED_RESERVE_BYTES = 5 * _GIB
_PARTITION_VALUE_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_ASSOCIATION_SCHEMA_FIELDS = (
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
_ASSOCIATION_SCHEMA_FINGERPRINT = canonical_sha256(
    {
        "schema": "nfl_x13_association_candidate_parquet_v1",
        "fields": _ASSOCIATION_SCHEMA_FIELDS,
    }
)
_NFL_TEAM_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ARI": ("ARI", "Arizona", "Arizona Cardinals", "Cardinals"),
        "ATL": ("ATL", "Atlanta", "Atlanta Falcons", "Falcons"),
        "BAL": ("BAL", "Baltimore", "Baltimore Ravens", "Ravens"),
        "BUF": ("BUF", "Buffalo", "Buffalo Bills", "Bills"),
        "CAR": ("CAR", "Carolina", "Carolina Panthers", "Panthers"),
        "CHI": ("CHI", "Chicago", "Chicago Bears", "Bears"),
        "CIN": ("CIN", "Cincinnati", "Cincinnati Bengals", "Bengals"),
        "CLE": ("CLE", "Cleveland", "Cleveland Browns", "Browns"),
        "DAL": ("DAL", "Dallas", "Dallas Cowboys", "Cowboys"),
        "DEN": ("DEN", "Denver", "Denver Broncos", "Broncos"),
        "DET": ("DET", "Detroit", "Detroit Lions", "Lions"),
        "GB": ("GB", "Green Bay", "Green Bay Packers", "Packers"),
        "HOU": ("HOU", "Houston", "Houston Texans", "Texans"),
        "JAX": (
            "JAC",
            "JAX",
            "Jacksonville",
            "Jacksonville Jaguars",
            "Jaguars",
        ),
        "KC": ("KC", "Kansas City", "Kansas City Chiefs", "Chiefs"),
        "LA": ("LA", "Los Angeles R", "Los Angeles Rams", "Rams"),
        "LAC": (
            "LAC",
            "Los Angeles C",
            "Los Angeles Chargers",
            "Chargers",
        ),
        "LV": ("LV", "Las Vegas", "Las Vegas Raiders", "Raiders"),
        "MIA": ("MIA", "Miami", "Miami Dolphins", "Dolphins"),
        "MIN": ("MIN", "Minnesota", "Minnesota Vikings", "Vikings"),
        "NE": ("NE", "New England", "New England Patriots", "Patriots"),
        "NO": ("NO", "New Orleans", "New Orleans Saints", "Saints"),
        "NYJ": ("NYJ", "New York J", "New York Jets", "Jets"),
        "PHI": ("PHI", "Philadelphia", "Philadelphia Eagles", "Eagles"),
        "SEA": ("SEA", "Seattle", "Seattle Seahawks", "Seahawks"),
        "SF": ("SF", "San Francisco", "San Francisco 49ers", "49ers"),
        "TB": (
            "TB",
            "Tampa Bay",
            "Tampa Bay Buccaneers",
            "Buccaneers",
            "Bucs",
        ),
    }
)


class X13PipelineError(ValueError):
    """The evidence graph cannot support an X-13 batch publication."""


@dataclass(frozen=True, slots=True)
class X13PipelineAuthorizationV1:
    experiment_id: str
    scope: str
    registration_head_sha256: str
    required_lock_ids: tuple[str, ...]
    resolved_lock_ids: tuple[str, ...]
    analysis_lock_sha256: str
    license_operational_use: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.experiment_id != X13_EXPERIMENT_ID:
            raise X13PipelineError("authorization is not bound to X-13")
        if self.scope != _REQUIRED_SCOPE:
            raise X13PipelineError("authorization scope is not source-time-only")
        for field_name in (
            "registration_head_sha256",
            "analysis_lock_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in ("required_lock_ids", "resolved_lock_ids"):
            values = getattr(self, field_name)
            if (
                type(values) is not tuple
                or not values
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
                or any(type(item) is not str or not item for item in values)
            ):
                raise X13PipelineError(
                    f"{field_name} must be a unique sorted tuple"
                )
        if self.required_lock_ids != self.resolved_lock_ids:
            unresolved = sorted(
                set(self.required_lock_ids) - set(self.resolved_lock_ids)
            )
            raise X13PipelineError(
                "unresolved registration locks: " + ",".join(unresolved)
            )
        if (
            type(self.license_operational_use) is not tuple
            or tuple(item[0] for item in self.license_operational_use)
            != _REQUIRED_LICENSE_REFS
        ):
            raise X13PipelineError(
                "authorization must bind exact Polymarket and Kalshi licenses"
            )
        for license_ref, operational_use in self.license_operational_use:
            if (
                license_ref not in _REQUIRED_LICENSE_REFS
                or operational_use not in _RESEARCH_OPERATIONAL_USE
            ):
                raise X13PipelineError(
                    "license is unknown or blocked for research use"
                )


@dataclass(frozen=True, slots=True)
class VerifiedCaptureDocumentV1:
    game_id: str | None
    resource: str
    manifest_sha256: str
    object_sha256: str
    byte_length: int
    source_url: str
    source_request: Mapping[str, object]
    source_cursor: str | None
    coverage: str
    dataset_id: str
    license_ref: str
    license_status: str
    payload: object


@dataclass(frozen=True, slots=True)
class VerifiedX13CaptureEvidenceV1:
    capture_sha256: str
    raw_manifest_sha256s: tuple[str, ...]
    documents_by_game: Mapping[str, tuple[VerifiedCaptureDocumentV1, ...]]
    batch_documents: tuple[VerifiedCaptureDocumentV1, ...]
    terminal_proofs_verified: bool
    manifest_hashes_verified: bool


@dataclass(frozen=True, slots=True)
class X13PipelineResultV1:
    batch_path: Path
    batch_manifest: Mapping[str, object]
    game_payloads: tuple[Mapping[str, object], ...]
    runtime: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AssociationStoragePreflightV1:
    available_free_bytes: int
    already_written_bytes: int
    remaining_association_rows: int
    empirical_bytes_per_row: int
    planned_remaining_p95_bytes: int
    temporary_headroom_bytes: int
    untouched_reserve_bytes: int
    required_free_bytes: int


@dataclass(frozen=True, slots=True)
class AssociationStoragePilotV1:
    row_limit: int
    row_count: int
    byte_length: int
    empirical_bytes_per_row: int
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class _PreparedGameV1:
    game_state: Any
    universe_game: UniverseGameV1
    contracts: tuple[MarketContractV1, ...]
    observations: tuple[MarketObservation, ...]
    ledger: X13GameLedgerV1
    association_stream: X13AssociationStreamV1
    raw_manifest_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _StreamedGameV1:
    presentation_rows: tuple[Mapping[str, object], ...]
    presentation_eligible_count: int
    presentation_omitted_count: int
    auxiliary_artifacts: tuple[AuxiliaryArtifactV1, ...]
    actual_row_count: int
    contaminated_episode_ids: tuple[str, ...]
    elapsed_ns: int


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise X13PipelineError(f"{field} must be a canonical SHA-256")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise X13PipelineError("payload is not canonical JSON") from error


def _strict_json(
    raw: bytes,
    *,
    context: str,
    exact_decimal_numbers: bool = False,
) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise X13PipelineError(
                    f"{context} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_float=Decimal if exact_decimal_numbers else float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                X13PipelineError(
                    f"{context} contains non-finite JSON {value}"
                )
            ),
        )
    except X13PipelineError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise X13PipelineError(f"{context} is not strict JSON") from error


def _parse_utc_text(value: object, *, context: str) -> datetime:
    if type(value) is not str or not value:
        raise X13PipelineError(f"{context} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise X13PipelineError(f"{context} is not a UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise X13PipelineError(f"{context} is not a UTC timestamp")
    return parsed


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise X13PipelineError("datetime is not UTC")
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(child)
            for key, child in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise X13PipelineError(
        f"unsupported canonical payload type: {type(value).__name__}"
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def preflight_x13_association_storage(
    *,
    available_free_bytes: int,
    already_written_bytes: int,
    remaining_association_rows: int,
    empirical_bytes_per_row: int,
) -> AssociationStoragePreflightV1:
    """Apply the frozen 1 GiB temporary + 5 GiB untouched disk gate."""

    for field_name, value in (
        ("available_free_bytes", available_free_bytes),
        ("already_written_bytes", already_written_bytes),
        ("remaining_association_rows", remaining_association_rows),
        ("empirical_bytes_per_row", empirical_bytes_per_row),
    ):
        if type(value) is not int or value < 0:
            raise X13PipelineError(
                f"{field_name} must be a nonnegative integer"
            )
    if empirical_bytes_per_row == 0 and remaining_association_rows:
        raise X13PipelineError(
            "empirical bytes/row must be measured before disk preflight"
        )
    planning_bytes_per_row = max(
        _ASSOCIATION_MIN_BYTES_PER_ROW,
        empirical_bytes_per_row,
    )
    planned_remaining = math.ceil(
        remaining_association_rows
        * planning_bytes_per_row
        * _ASSOCIATION_P95_UPLIFT_NUMERATOR
        / _ASSOCIATION_P95_UPLIFT_DENOMINATOR
    )
    required = (
        planned_remaining
        + _TEMP_HEADROOM_BYTES
        + _UNTOUCHED_RESERVE_BYTES
    )
    result = AssociationStoragePreflightV1(
        available_free_bytes=available_free_bytes,
        already_written_bytes=already_written_bytes,
        remaining_association_rows=remaining_association_rows,
        empirical_bytes_per_row=empirical_bytes_per_row,
        planned_remaining_p95_bytes=planned_remaining,
        temporary_headroom_bytes=_TEMP_HEADROOM_BYTES,
        untouched_reserve_bytes=_UNTOUCHED_RESERVE_BYTES,
        required_free_bytes=required,
    )
    if available_free_bytes < required:
        raise X13PipelineError(
            "association storage preflight failed: preserve the required "
            "5 GiB untouched reserve after planned p95 derived output and "
            "1 GiB temporary headroom"
        )
    return result


def expected_analysis_lock_sha256(
    *,
    registration_head_sha256: str,
    game_state_build_id: str,
    universe_id: str,
    capture_sha256: str,
) -> str:
    """Bind the preregistration head and all three immutable input roots."""

    for field, value in (
        ("registration_head_sha256", registration_head_sha256),
        ("game_state_build_id", game_state_build_id),
        ("universe_id", universe_id),
        ("capture_sha256", capture_sha256),
    ):
        _require_sha256(value, field)
    return canonical_sha256(
        {
            "schema": "nfl_x13_analysis_lock_v1",
            "experiment_id": X13_EXPERIMENT_ID,
            "scope": _REQUIRED_SCOPE,
            "plan_id": X13_BATCH_SPEC.plan_id,
            "registration_head_sha256": registration_head_sha256,
            "game_state_build_id": game_state_build_id,
            "universe_id": universe_id,
            "capture_sha256": capture_sha256,
        }
    )


def expected_source_manifest_bundle_sha256(
    *,
    plan_id: str,
    game_state_build_id: str,
    universe_id: str,
    capture_sha256: str,
) -> str:
    """Bind every content-addressed source root required by X-13."""

    for field, value in (
        ("plan_id", plan_id),
        ("game_state_build_id", game_state_build_id),
        ("universe_id", universe_id),
        ("capture_sha256", capture_sha256),
    ):
        _require_sha256(value, field)
    return canonical_sha256(
        {
            "schema": "nfl_x13_source_manifest_bundle_v1",
            "experiment_id": X13_EXPERIMENT_ID,
            "plan_id": plan_id,
            "game_state_build_id": game_state_build_id,
            "universe_id": universe_id,
            "capture_sha256": capture_sha256,
        }
    )


def expected_x13_evaluation_code_bundle_sha256(
    program_root: str | Path,
) -> str:
    """Hash the exact frozen X-13 governance and evaluation code files."""

    unresolved_root = Path(program_root).absolute()
    if unresolved_root == Path(unresolved_root.anchor):
        raise X13PipelineError("program_root cannot be a filesystem root")
    root_cursor = Path(unresolved_root.anchor)
    for component in unresolved_root.parts[1:]:
        root_cursor /= component
        if root_cursor.is_symlink():
            raise X13PipelineError(
                "X-13 evaluation code root contains a symlink component"
            )
    try:
        root = unresolved_root.resolve(strict=True)
    except OSError as error:
        raise X13PipelineError(
            "X-13 evaluation code root is missing or unsafe"
        ) from error
    if not root.is_dir():
        raise X13PipelineError(
            "X-13 evaluation code root is not a directory"
        )
    files: list[dict[str, str]] = []
    for relative_path in X13_EVALUATION_CODE_PATHS:
        if type(relative_path) is not str or not relative_path:
            raise X13PipelineError(
                "X-13 evaluation code path must be canonical relative POSIX"
            )
        canonical_path = PurePosixPath(relative_path)
        if (
            canonical_path.is_absolute()
            or canonical_path.as_posix() != relative_path
            or "\\" in relative_path
            or any(
                component in {"", ".", ".."}
                for component in canonical_path.parts
            )
        ):
            raise X13PipelineError(
                "X-13 evaluation code path must be canonical relative POSIX"
            )
        path = root
        for component in canonical_path.parts:
            path /= component
            if path.is_symlink():
                raise X13PipelineError(
                    "X-13 evaluation code path contains a symlink component"
                )
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise X13PipelineError(
                f"X-13 evaluation code file escapes or is missing: "
                f"{relative_path}"
            ) from error
        if not resolved_path.is_file():
            raise X13PipelineError(
                f"X-13 evaluation code file is missing or unsafe: "
                f"{relative_path}"
            )
        files.append(
            {
                "path": relative_path,
                "sha256": "sha256:"
                + hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
            }
        )
    return canonical_sha256(
        {
            "schema": "nfl_x13_evaluation_code_bundle_v1",
            "experiment_id": X13_EXPERIMENT_ID,
            "files": files,
        }
    )


def expected_x13_source_manifest_amendment_changes(
    *,
    program_root: str | Path,
    game_state_build_id: str,
    universe_id: str,
    capture_sha256: str,
) -> dict[str, object]:
    """Return the exact post-capture lock and input preregistration change."""

    data_sha256 = expected_source_manifest_bundle_sha256(
        plan_id=X13_BATCH_SPEC.plan_id,
        game_state_build_id=game_state_build_id,
        universe_id=universe_id,
        capture_sha256=capture_sha256,
    )
    return {
        "resolve_locks": [
            {
                "lock_id": "source_manifest_bundle",
                "evidence_ref": data_sha256,
            }
        ],
        "preregistered_inputs": [
            {
                "scope": _REQUIRED_SCOPE,
                "code_sha256": (
                    expected_x13_evaluation_code_bundle_sha256(program_root)
                ),
                "data_sha256": data_sha256,
                "dataset_ids": list(_X13_PREREGISTERED_DATASET_IDS),
                "model_ids": [],
            }
        ],
    }


def load_x13_pipeline_authorization(
    program_root: str | Path,
    *,
    game_state_build_id: str,
    universe_id: str,
    capture_sha256: str,
) -> X13PipelineAuthorizationV1:
    """Load the registered X-13 scope and reject every unresolved lock."""

    try:
        card = load_experiment_registry(program_root)[X13_EXPERIMENT_ID]
        reviews = {
            row.catalog_item_id: row
            for row in load_data_license_register(program_root)
        }
    except (
        ExperimentRegistryError,
        ComplianceRegistryError,
        KeyError,
    ) as error:
        raise X13PipelineError(
            "X-13 registration or license registry is invalid"
        ) from error
    if (
        card.get("status") != "registered"
        or card.get("execution_authorized") is not True
        or card.get("source_time_only") is not True
        or card.get("causal_or_execution_claims_authorized") is not False
    ):
        raise X13PipelineError("X-13 source-time execution is not authorized")
    scopes = card.get("authorization_scopes")
    if not isinstance(scopes, Mapping):
        raise X13PipelineError("X-13 authorization scopes are malformed")
    scope = scopes.get(_REQUIRED_SCOPE)
    if (
        not isinstance(scope, Mapping)
        or scope.get("authorized") is not True
        or scope.get("required_result_label") != "PRELIMINARY"
    ):
        raise X13PipelineError("X-13 preliminary scope is not authorized")
    required = scope.get("required_lock_ids")
    locks = card.get("registration_locks")
    if (
        type(required) is not list
        or not isinstance(locks, Sequence)
        or isinstance(locks, (str, bytes))
    ):
        raise X13PipelineError("X-13 registration locks are malformed")
    lock_status: dict[str, str] = {}
    lock_evidence: dict[str, object] = {}
    for lock in locks:
        if (
            not isinstance(lock, Mapping)
            or type(lock.get("id")) is not str
            or type(lock.get("status")) is not str
        ):
            raise X13PipelineError("X-13 registration lock is malformed")
        lock_id = str(lock["id"])
        lock_status[lock_id] = str(lock["status"])
        lock_evidence[lock_id] = lock.get("evidence_ref")
    unresolved = sorted(
        lock_id
        for lock_id in required
        if lock_status.get(lock_id) != "resolved"
    )
    if unresolved:
        raise X13PipelineError(
            "unresolved registration locks: " + ",".join(unresolved)
        )
    expected_changes = expected_x13_source_manifest_amendment_changes(
        program_root=program_root,
        game_state_build_id=game_state_build_id,
        universe_id=universe_id,
        capture_sha256=capture_sha256,
    )
    expected_source_bundle = expected_changes["resolve_locks"][0][
        "evidence_ref"
    ]
    if (
        lock_evidence.get("source_manifest_bundle")
        != expected_source_bundle
    ):
        raise X13PipelineError(
            "source_manifest_bundle evidence does not bind the active "
            "plan, game-state build, universe, and capture"
        )
    preregistered_inputs = card.get("preregistered_inputs")
    expected_input = expected_changes["preregistered_inputs"][0]
    actual_input = (
        preregistered_inputs.get(_REQUIRED_SCOPE)
        if isinstance(preregistered_inputs, Mapping)
        else None
    )
    if (
        not isinstance(actual_input, Mapping)
        or set(actual_input)
        != {
            "code_sha256",
            "data_sha256",
            "dataset_ids",
            "model_ids",
            "registered_at",
        }
        or any(
            actual_input.get(field) != expected_input[field]
            for field in (
                "code_sha256",
                "data_sha256",
                "dataset_ids",
                "model_ids",
            )
        )
        or type(actual_input.get("registered_at")) is not str
        or not actual_input["registered_at"]
    ):
        raise X13PipelineError(
            "X-13 preregistered_inputs do not bind the frozen code and "
            "source manifest bundles"
        )
    license_rows: list[tuple[str, str]] = []
    for license_ref in _REQUIRED_LICENSE_REFS:
        review = reviews.get(license_ref)
        if (
            review is None
            or review.status == "NOT_GREEN_BLOCKED"
            or review.operational_use not in _RESEARCH_OPERATIONAL_USE
        ):
            raise X13PipelineError(
                f"license {license_ref} is unknown or blocked"
            )
        license_rows.append((license_ref, review.operational_use))
    registration_head = _require_sha256(
        card.get("registration_head_sha256"),
        "registration_head_sha256",
    )
    analysis_lock = expected_analysis_lock_sha256(
        registration_head_sha256=registration_head,
        game_state_build_id=game_state_build_id,
        universe_id=universe_id,
        capture_sha256=capture_sha256,
    )
    return X13PipelineAuthorizationV1(
        experiment_id=X13_EXPERIMENT_ID,
        scope=_REQUIRED_SCOPE,
        registration_head_sha256=registration_head,
        required_lock_ids=tuple(sorted(required)),
        resolved_lock_ids=tuple(sorted(required)),
        analysis_lock_sha256=analysis_lock,
        license_operational_use=tuple(license_rows),
    )


def _verify_game_state_evidence(
    build: X13GameStateBuild,
    publication: str | Path,
) -> None:
    if not isinstance(build, X13GameStateBuild):
        raise X13PipelineError("game-state build has an unknown type")
    if tuple(sorted(build.games)) != X13_GAME_IDS:
        raise X13PipelineError("game-state build must cover the exact frozen 20")
    try:
        verified = verify_x13_game_state_publication(publication)
    except (X13GameBuildError, OSError) as error:
        raise X13PipelineError("game-state publication verification failed") from error
    build_id = _require_sha256(build.manifest.get("build_id"), "game build_id")
    if verified.get("build_id") != build_id:
        raise X13PipelineError(
            "game-state publication does not match the supplied build"
        )
    root = Path(publication).resolve()
    manifest = _strict_json(
        (root / "build_manifest.json").read_bytes(),
        context="game-state build manifest",
    )
    if not isinstance(manifest, Mapping):
        raise X13PipelineError(
            "game-state publication manifest is malformed"
        )
    try:
        publication_semantic_sha256 = (
            x13_game_state_manifest_semantic_sha256(manifest)
        )
        build_semantic_sha256 = x13_game_state_manifest_semantic_sha256(
            build.manifest
        )
    except X13GameBuildError as error:
        raise X13PipelineError(
            "game-state semantic manifest verification failed"
        ) from error
    if publication_semantic_sha256 != build_semantic_sha256:
        raise X13PipelineError(
            "game-state in-memory semantic manifest differs from publication"
        )
    for game_id in X13_GAME_IDS:
        game_manifest = _strict_json(
            (root / "games" / game_id / "game_manifest.json").read_bytes(),
            context=f"{game_id} game manifest",
        )
        game = build.games[game_id]
        if (
            not isinstance(game_manifest, Mapping)
            or game_manifest.get("events_sha256") != game.events_sha256
            or game_manifest.get("ledger_sha256") != game.ledger_sha256
            or game_manifest.get("final_score") != dict(game.final_score)
        ):
            raise X13PipelineError(
                f"{game_id} in-memory replay differs from publication"
            )


def _verify_game_source_manifests(
    build: X13GameStateBuild,
    *,
    program_root: Path,
    raw_store_root: Path,
) -> tuple[VerifiedCaptureDocumentV1, ...]:
    expected = {
        build.source.pbp_manifest_sha256: (
            "DS-NFLVERSE",
            "I-018",
            "nflverse_pbp",
        ),
        build.source.participation_manifest_sha256: (
            "DS-NFLVERSE-PARTICIPATION",
            "O-009",
            "nflverse_participation",
        ),
    }
    declared = build.manifest.get("source_manifest_hashes")
    if (
        type(declared) is not list
        or set(declared) != set(expected)
        or len(declared) != len(expected)
    ):
        raise X13PipelineError(
            "game-state build does not bind the exact NFL source manifests"
        )
    documents: list[VerifiedCaptureDocumentV1] = []
    for manifest_sha256, (
        dataset_id,
        license_ref,
        resource_name,
    ) in sorted(expected.items()):
        _require_sha256(manifest_sha256, "game-state source manifest")
        matches = list(
            raw_store_root.glob(
                "manifests/**/"
                f"{manifest_sha256.removeprefix('sha256:')}.manifest.json"
            )
        )
        if len(matches) != 1:
            raise X13PipelineError(
                "game-state source manifest cannot be resolved uniquely"
            )
        try:
            verified = read_verified_static_object(
                matches[0],
                store_root=raw_store_root,
                program_root=program_root,
            )
        except (StaticStoreError, OSError) as error:
            raise X13PipelineError(
                "game-state source raw object verification failed"
            ) from error
        manifest = verified.record.manifest
        if (
            manifest.manifest_sha256 != manifest_sha256
            or manifest.dataset_id != dataset_id
            or manifest.license_ref != license_ref
            or manifest.license_status in {"unknown", "blocked"}
        ):
            raise X13PipelineError(
                "game-state source license, dataset, or hash is invalid"
            )
        documents.append(
            VerifiedCaptureDocumentV1(
                game_id=None,
                resource=resource_name,
                manifest_sha256=manifest.manifest_sha256,
                object_sha256=manifest.object_sha256,
                byte_length=manifest.byte_length,
                source_url=manifest.source_url,
                source_request=MappingProxyType(
                    dict(manifest.source_request)
                ),
                source_cursor=manifest.source_cursor,
                coverage=manifest.coverage,
                dataset_id=manifest.dataset_id,
                license_ref=manifest.license_ref,
                license_status=manifest.license_status,
                payload=None,
            )
        )
    return tuple(documents)


def _team_alias_key(value: str) -> str:
    return "".join(
        character
        for character in value.casefold()
        if character.isalnum()
    )


def _game_binding(game_id: str) -> Any:
    matches = [
        binding
        for binding in X13_GAME_BINDINGS
        if binding.native_game_id == game_id
    ]
    if len(matches) != 1:
        raise X13PipelineError("game has no unique frozen team binding")
    return matches[0]


def _is_frozen_winner_contract(
    game: UniverseGameV1,
    contract: UniverseContractV1,
) -> bool:
    return (
        contract.family == "moneyline"
        and contract.period == "game"
        and contract.measure == "winner"
        and contract.logical_market_id
        == f"{contract.venue}:{game.game_id}:winner"
    )


def _canonical_winner_outcome(
    game: UniverseGameV1,
    contract: UniverseContractV1,
    native_outcome: str,
) -> str:
    """Map a frozen winner token label through an explicit team registry."""

    if not _is_frozen_winner_contract(game, contract):
        raise X13PipelineError(
            "canonical winner orientation requested for a non-winner contract"
        )
    binding = _game_binding(game.game_id)
    candidates = (binding.away_team, binding.home_team)
    native_key = _team_alias_key(native_outcome)
    matches = [
        team
        for team in candidates
        if native_key
        in {
            _team_alias_key(alias)
            for alias in _NFL_TEAM_ALIASES.get(team, ())
        }
    ]
    if len(matches) != 1:
        raise X13PipelineError(
            "winner outcome cannot be oriented by the explicit team registry"
        )
    return matches[0]


def _is_structured_game_total(
    contract: UniverseContractV1,
) -> bool:
    """Return whether native fields prove a full-game combined-points total."""

    return (
        contract.family == "total"
        and contract.period == "game"
        and contract.measure == "combined_points"
        and contract.comparator == "gt"
        and contract.line is not None
        and {
            outcome.casefold()
            for outcome in contract.outcomes
        }
        in ({"yes", "no"}, {"over", "under"})
    )


def _canonical_market_outcome(
    game: UniverseGameV1,
    contract: UniverseContractV1,
    native_outcome: str,
) -> str:
    """Orient only propositions whose structured fields prove equivalence."""

    if _is_frozen_winner_contract(game, contract):
        return _canonical_winner_outcome(game, contract, native_outcome)
    if _is_structured_game_total(contract):
        normalized = {
            "yes": "Over",
            "over": "Over",
            "no": "Under",
            "under": "Under",
        }.get(native_outcome.casefold())
        if normalized is None:
            raise X13PipelineError(
                "structured game-total outcome orientation is unresolved"
            )
        return normalized
    return native_outcome


def normalize_x13_contracts(
    game: UniverseGameV1,
) -> tuple[MarketContractV1, ...]:
    """Convert the universe proposition registry into Layer M contracts."""

    if not isinstance(game, UniverseGameV1):
        raise X13PipelineError("universe game has an unknown type")
    known_families = {
        *X13_VENUE_FAMILY_CATALOG.primitive_families,
        X13_VENUE_FAMILY_CATALOG.same_game_composite_family,
        X13_VENUE_FAMILY_CATALOG.cross_game_inventory_family,
    }
    values: list[MarketContractV1] = []
    seen_contract_ids: set[str] = set()
    seen_logical_ids: set[str] = set()
    for contract in game.contracts:
        if not isinstance(contract, UniverseContractV1):
            raise X13PipelineError("universe contract has an unknown type")
        if contract.family not in known_families:
            raise X13PipelineError(
                f"unknown contract family: {contract.family}"
            )
        if (
            contract.analysis_eligible
            and contract.dependency_game_ids != (game.game_id,)
        ):
            raise X13PipelineError(
                "cross-game contract entered single-game analysis"
            )
        if (
            game.game_id not in contract.dependency_game_ids
            or contract.contract_id in seen_contract_ids
            or contract.logical_market_id in seen_logical_ids
        ):
            raise X13PipelineError("contract inventory identity is inconsistent")
        seen_contract_ids.add(contract.contract_id)
        seen_logical_ids.add(contract.logical_market_id)
        subject = f"SINGLE_VENUE:{contract.contract_id}"
        outcomes = contract.outcomes
        if _is_frozen_winner_contract(game, contract):
            binding = _game_binding(game.game_id)
            mapped = tuple(
                _canonical_winner_outcome(game, contract, outcome)
                for outcome in contract.outcomes
            )
            if (
                len(mapped) != 2
                or set(mapped)
                != {binding.away_team, binding.home_team}
            ):
                raise X13PipelineError(
                    "winner contract does not prove both frozen teams"
                )
            subject = game.game_id
            outcomes = (binding.away_team, binding.home_team)
        elif _is_structured_game_total(contract):
            subject = game.game_id
            outcomes = ("Over", "Under")
        try:
            values.append(
                MarketContractV1(
                    contract_id=contract.contract_id,
                    logical_market_id=contract.logical_market_id,
                    venue=contract.venue,
                    family=contract.family,
                    period=contract.period,
                    subject=subject,
                    measure=contract.measure,
                    comparator=contract.comparator,
                    line=contract.line,
                    outcomes=outcomes,
                    rule_sha256=contract.rule_sha256,
                    dependency_game_ids=contract.dependency_game_ids,
                    kind=contract.kind,
                    analysis_eligible=contract.analysis_eligible,
                )
            )
        except (TypeError, ValueError) as error:
            raise X13PipelineError(
                f"contract normalization failed: {contract.contract_id}"
            ) from error
    if not values:
        raise X13PipelineError("game has no normalized contracts")
    return tuple(sorted(values, key=lambda value: value.contract_id))


def verify_x13_universe_artifact(
    universe: X13UniverseBatchV1,
    artifact: str | Path,
) -> None:
    """Re-open the canonical universe artifact and verify its full shape."""

    if not isinstance(universe, X13UniverseBatchV1):
        raise X13PipelineError("universe has an unknown type")
    if (
        universe.game_ids != X13_GAME_IDS
        or tuple(plan.game_id for plan in universe.capture_plans)
        != X13_GAME_IDS
        or len(universe.games) != 20
    ):
        raise X13PipelineError("universe must cover the exact frozen 20")
    if universe.universe_id != _FROZEN_UNIVERSE_ID:
        raise X13PipelineError(
            "universe_id does not match the frozen X-13 contract universe"
        )
    expected_series = tuple(
        spec.series_ticker
        for spec in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
    )
    if (
        universe.series_registry_proof.series_tickers != expected_series
        or universe.schema_version != "nfl-x13-universe-v1"
    ):
        raise X13PipelineError("universe series registry proof is invalid")
    for game, plan in zip(universe.games, universe.capture_plans, strict=True):
        if (
            game.capture_plan != plan
            or game.unknown_contract_ids
            or game.game_id != plan.game_id
        ):
            raise X13PipelineError(
                f"{game.game_id} universe contains an unresolved identity"
            )
        normalize_x13_contracts(game)
        for digest in game.metadata_manifest_sha256s:
            _require_sha256(digest, "universe metadata manifest")
    path = Path(artifact)
    if path.is_symlink() or not path.is_file():
        raise X13PipelineError("universe artifact is missing or unsafe")
    raw = path.read_bytes()
    payload = _strict_json(raw, context="universe artifact")
    expected = universe.canonical_payload
    if payload != expected or raw != _json_bytes(expected) + b"\n":
        raise X13PipelineError(
            "universe artifact does not match the canonical in-memory plan"
        )
    if payload.get("universe_id") != universe.universe_id:
        raise X13PipelineError("universe artifact identity is inconsistent")


def _artifact_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise X13PipelineError(f"{context} must be an object")
    return value


def _artifact_list(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise X13PipelineError(f"{context} must be an array")
    return value


def _load_capture_target(value: object) -> CaptureTarget:
    row = _artifact_mapping(value, "universe capture target")
    try:
        return CaptureTarget(
            contract_id=str(row["contract_id"]),
            venue=str(row["venue"]),
            venue_market_id=str(row["venue_market_id"]),
            condition_id=(
                None
                if row.get("condition_id") is None
                else str(row["condition_id"])
            ),
            venue_title=str(row["venue_title"]),
            kalshi_series_ticker=(
                None
                if row.get("kalshi_series_ticker") is None
                else str(row["kalshi_series_ticker"])
            ),
            kalshi_event_ticker=(
                None
                if row.get("kalshi_event_ticker") is None
                else str(row["kalshi_event_ticker"])
            ),
            family=str(row["family"]),
            dependency_game_ids=tuple(
                str(item)
                for item in _artifact_list(
                    row["dependency_game_ids"],
                    "capture target dependency_game_ids",
                )
            ),
            kind=str(row["kind"]),
            analysis_eligible=row["analysis_eligible"] is True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise X13PipelineError("universe capture target is malformed") from error


def _load_universe_contract(value: object) -> UniverseContractV1:
    row = _artifact_mapping(value, "universe contract")
    outcomes = tuple(
        str(item)
        for item in _artifact_list(row.get("outcomes"), "contract outcomes")
    )
    coverage = _artifact_mapping(
        row.get("outcome_coverage"), "contract outcome_coverage"
    )
    line = row.get("line")
    try:
        return UniverseContractV1(
            contract_id=str(row["contract_id"]),
            venue_market_id=str(row["venue_market_id"]),
            raw_contract_ids=tuple(
                str(item)
                for item in _artifact_list(
                    row["raw_contract_ids"], "contract raw_contract_ids"
                )
            ),
            condition_id=(
                None
                if row.get("condition_id") is None
                else str(row["condition_id"])
            ),
            logical_market_id=str(row["logical_market_id"]),
            venue=str(row["venue"]),
            family=str(row["family"]),
            period=str(row["period"]),
            subject=str(row["subject"]),
            measure=str(row["measure"]),
            comparator=str(row["comparator"]),
            line=None if line is None else Decimal(str(line)),
            outcomes=outcomes,
            raw_outcome_labels=tuple(
                tuple(str(part) for part in _artifact_list(item, "outcome label"))
                for item in _artifact_list(
                    row.get("raw_outcome_labels"),
                    "contract raw_outcome_labels",
                )
            ),
            rule_sha256=str(row["rule_sha256"]),
            dependency_game_ids=tuple(
                str(item)
                for item in _artifact_list(
                    row["dependency_game_ids"],
                    "contract dependency_game_ids",
                )
            ),
            kind=str(row["kind"]),
            analysis_eligible=row["analysis_eligible"] is True,
            outcome_coverage=tuple(
                (outcome, str(coverage[outcome])) for outcome in outcomes
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise X13PipelineError("universe contract is malformed") from error


def _load_composite(value: object) -> CompositeClassificationV1:
    row = _artifact_mapping(value, "cross-game composite")
    try:
        legs = tuple(
            CompositeLegV1(
                event_ticker=str(leg["event_ticker"]),
                market_ticker=str(leg["market_ticker"]),
                side=str(leg["side"]),
                game_id=str(leg["game_id"]),
            )
            for leg_value in _artifact_list(row["legs"], "composite legs")
            for leg in (_artifact_mapping(leg_value, "composite leg"),)
        )
        return CompositeClassificationV1(
            venue_market_id=str(row["venue_market_id"]),
            venue_title=str(row["venue_title"]),
            series_ticker=str(row["series_ticker"]),
            event_ticker=str(row["event_ticker"]),
            legs=legs,
            dependency_game_ids=tuple(
                str(item)
                for item in _artifact_list(
                    row["dependency_game_ids"],
                    "composite dependency_game_ids",
                )
            ),
            kind=str(row["kind"]),
            analysis_eligible=row["analysis_eligible"] is True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise X13PipelineError("cross-game composite is malformed") from error


def load_x13_universe_artifact(
    artifact: str | Path,
) -> X13UniverseBatchV1:
    """Reconstruct the exact typed capture plan from its canonical artifact."""

    path = Path(artifact)
    if path.is_symlink() or not path.is_file():
        raise X13PipelineError("universe artifact is missing or unsafe")
    raw = path.read_bytes()
    payload = _strict_json(raw, context="universe artifact")
    root = _artifact_mapping(payload, "universe artifact")
    games: list[UniverseGameV1] = []
    try:
        for raw_game in _artifact_list(root["games"], "universe games"):
            game = _artifact_mapping(raw_game, "universe game")
            plan = _artifact_mapping(game["capture_plan"], "capture plan")
            targets = tuple(
                _load_capture_target(item)
                for item in _artifact_list(plan["targets"], "capture targets")
            )
            target_by_id = {target.contract_id: target for target in targets}
            if len(target_by_id) != len(targets):
                raise X13PipelineError(
                    "universe capture targets contain duplicate identity"
                )
            inventory = _artifact_mapping(game["inventory"], "game inventory")
            primitive_ids = tuple(
                str(item)
                for item in _artifact_list(
                    inventory["primitive_contract_ids"],
                    "primitive contract IDs",
                )
            )
            composite_ids = tuple(
                str(item)
                for item in _artifact_list(
                    inventory["same_game_composite_contract_ids"],
                    "same-game composite contract IDs",
                )
            )
            if set((*primitive_ids, *composite_ids)) != set(target_by_id):
                raise X13PipelineError(
                    "universe inventory does not bind every capture target"
                )
            source_window = _artifact_mapping(
                game["source_window"], "source window"
            )
            poly = _artifact_mapping(
                game["polymarket_event"], "Polymarket event"
            )
            excluded = tuple(
                ExcludedContractV1(
                    contract_id=str(item["contract_id"]),
                    venue_market_id=str(item["venue_market_id"]),
                    series_ticker=str(item["series_ticker"]),
                    event_ticker=str(item["event_ticker"]),
                    title=str(item["title"]),
                    reason=str(item["reason"]),
                )
                for raw_item in _artifact_list(
                    game["excluded_contracts"], "excluded contracts"
                )
                for item in (
                    _artifact_mapping(raw_item, "excluded contract"),
                )
            )
            token_bindings = tuple(
                PolymarketTokenBindingV1(
                    condition_id=str(item["condition_id"]),
                    outcomes=tuple(
                        str(outcome)
                        for outcome in _artifact_list(
                            item["outcomes"], "token outcomes"
                        )
                    ),
                    token_ids=tuple(
                        str(token)
                        for token in _artifact_list(
                            item["token_ids"], "token IDs"
                        )
                    ),
                )
                for raw_item in _artifact_list(
                    game["polymarket_token_bindings"],
                    "Polymarket token bindings",
                )
                for item in (
                    _artifact_mapping(raw_item, "token binding"),
                )
            )
            games.append(
                UniverseGameV1(
                    game_id=str(game["game_id"]),
                    source_window=SourceTimeWindowV1(
                        int(source_window["start_ts"]),
                        int(source_window["end_ts"]),
                    ),
                    polymarket_event_id=str(poly["id"]),
                    polymarket_event_slug=str(poly["slug"]),
                    polymarket_event_title=str(poly["title"]),
                    primitive_targets=tuple(
                        target_by_id[contract_id]
                        for contract_id in primitive_ids
                    ),
                    same_game_composite_targets=tuple(
                        target_by_id[contract_id]
                        for contract_id in composite_ids
                    ),
                    cross_game_inventory=tuple(
                        _load_composite(item)
                        for item in _artifact_list(
                            game["cross_game_inventory"],
                            "cross-game inventory",
                        )
                    ),
                    excluded_contracts=excluded,
                    unknown_contract_ids=tuple(
                        str(item)
                        for item in _artifact_list(
                            inventory["unknown_contract_ids"],
                            "unknown contract IDs",
                        )
                    ),
                    contracts=tuple(
                        _load_universe_contract(item)
                        for item in _artifact_list(
                            game["contracts"], "universe contracts"
                        )
                    ),
                    polymarket_token_bindings=token_bindings,
                    kalshi_series_tickers=tuple(
                        str(item)
                        for item in _artifact_list(
                            game["kalshi_series_tickers"],
                            "Kalshi series tickers",
                        )
                    ),
                    metadata_manifest_sha256s=tuple(
                        str(item)
                        for item in _artifact_list(
                            game["metadata_manifest_sha256s"],
                            "metadata manifest hashes",
                        )
                    ),
                )
            )
        proof = _artifact_mapping(
            root["series_registry_proof"], "series registry proof"
        )
        batch = X13UniverseBatchV1(
            games=tuple(games),
            capture_plans=tuple(game.capture_plan for game in games),
            series_registry_proof=KalshiSeriesRegistryProofV1(
                series_tickers=tuple(
                    str(item)
                    for item in _artifact_list(
                        proof["series_tickers"], "series registry tickers"
                    )
                ),
                manifest_sha256=str(proof["manifest_sha256"]),
                terminal_proof=str(proof["terminal_proof"]),
            ),
            schema_version=str(root["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, X13PipelineError):
            raise
        raise X13PipelineError("universe artifact is malformed") from error
    verify_x13_universe_artifact(batch, path)
    return batch


def _resource(source_url: str) -> str:
    try:
        parsed = urlsplit(source_url)
        port = parsed.port
    except ValueError as error:
        raise X13PipelineError(
            f"unknown captured resource URL: {source_url}"
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise X13PipelineError(
            f"unknown captured resource URL: {source_url}"
        )
    host = parsed.hostname
    path = parsed.path
    if (
        host == "gamma-api.polymarket.com"
        and (path == "/events" or path.startswith("/events/"))
    ):
        return "gamma_event"
    if (
        host == "data-api.polymarket.com"
        and path == _POLYMARKET_DATA_TRADES_PATH
    ):
        return "polymarket_trades"
    if host != "external-api.kalshi.com":
        raise X13PipelineError(
            f"unknown captured resource URL: {source_url}"
        )
    if path.endswith(_KALSHI_HISTORICAL_CUTOFF_PATH):
        return "kalshi_cutoff"
    if path.endswith("/candlesticks"):
        return "kalshi_candlesticks"
    if path.endswith(_KALSHI_HISTORICAL_TRADES_PATH):
        return "kalshi_trades"
    if path.endswith(_KALSHI_HISTORICAL_MARKETS_PATH):
        return "kalshi_markets"
    if path.endswith("/series"):
        return "kalshi_series"
    raise X13PipelineError(f"unknown captured resource URL: {source_url}")


def _verify_pointer(
    capture: BatchCaptureResult,
    *,
    universe: X13UniverseBatchV1,
    program_root: Path,
    raw_store_root: Path,
) -> None:
    pointer = capture.pointer
    receipt = capture.receipt
    if not isinstance(pointer, CapturePointer):
        raise X13PipelineError("capture pointer has an unknown type")
    if not isinstance(receipt, CaptureReceipt):
        raise X13PipelineError("capture receipt has an unknown type")
    if pointer.game_ids != X13_GAME_IDS:
        raise X13PipelineError("capture pointer must cover the exact frozen 20")
    if (
        capture.preflight.raw_store_root.resolve() != raw_store_root
        or capture.preflight.program_root.resolve() != program_root
    ):
        raise X13PipelineError(
            "capture preflight roots differ from verification roots"
        )
    hashes = tuple(
        manifest_receipt.manifest_sha256
        for manifest_receipt in capture.manifests
    )
    if (
        tuple(sorted(hashes)) != pointer.raw_manifest_sha256s
        or len(set(hashes)) != len(hashes)
    ):
        raise X13PipelineError(
            "capture pointer manifest hash set does not match receipts"
        )
    expected_plan_sha256 = _plans_sha256(universe.capture_plans)
    if (
        pointer.universe_id != universe.universe_id
        or receipt.universe_id != universe.universe_id
        or pointer.plan_sha256 != expected_plan_sha256
        or receipt.plan_sha256 != expected_plan_sha256
    ):
        raise X13PipelineError(
            "capture receipt is not bound to the frozen universe plan"
        )
    if (
        receipt.preflight != capture.preflight
        or receipt.kalshi_cutoff != capture.kalshi_cutoff
        or receipt.games != capture.games
        or receipt.manifests != capture.manifests
    ):
        raise X13PipelineError(
            "capture result differs from its immutable receipt"
        )
    receipt_sha256 = receipt.receipt_sha256
    if pointer.receipt_sha256 != receipt_sha256:
        raise X13PipelineError(
            "capture pointer does not bind the immutable receipt"
        )
    material = {
        "game_ids": list(pointer.game_ids),
        "plan_sha256": pointer.plan_sha256,
        "raw_manifest_sha256s": list(pointer.raw_manifest_sha256s),
        "receipt_sha256": pointer.receipt_sha256,
        "universe_id": pointer.universe_id,
        "version": "nfl-x13-capture-pointer-v2",
    }
    expected_hash = "sha256:" + hashlib.sha256(_json_bytes(material)).hexdigest()
    if pointer.capture_sha256 != expected_hash:
        raise X13PipelineError("capture pointer content hash is invalid")
    receipt_path = capture.receipt_path
    expected_receipt_path = (
        raw_store_root
        / "capture-receipts"
        / f"{receipt_sha256.removeprefix('sha256:')}.json"
    )
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.resolve() != expected_receipt_path.resolve()
    ):
        raise X13PipelineError("capture receipt is missing or unsafe")
    receipt_raw = receipt_path.read_bytes()
    if (
        _strict_json(receipt_raw, context="capture receipt")
        != receipt.canonical_payload
        or receipt_raw != _json_bytes(receipt.canonical_payload) + b"\n"
    ):
        raise X13PipelineError("capture receipt bytes do not match its identity")
    path = capture.pointer_path
    expected_pointer_path = (
        raw_store_root
        / "capture-pointers"
        / f"{pointer.capture_sha256.removeprefix('sha256:')}.json"
    )
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != expected_pointer_path.resolve()
    ):
        raise X13PipelineError("capture pointer is missing or unsafe")
    raw = path.read_bytes()
    if (
        _strict_json(raw, context="capture pointer") != pointer.canonical_payload
        or raw != _json_bytes(pointer.canonical_payload) + b"\n"
    ):
        raise X13PipelineError("capture pointer bytes do not match its identity")


def _receipt_document(
    receipt: ImmutableRequestManifest,
    *,
    game_id: str | None,
    program_root: Path,
    raw_store_root: Path,
) -> VerifiedCaptureDocumentV1:
    if not isinstance(receipt, ImmutableRequestManifest):
        raise X13PipelineError("capture receipt has an unknown type")
    try:
        verified = read_verified_static_object(
            receipt.manifest_path,
            store_root=raw_store_root,
            program_root=program_root,
        )
    except (StaticStoreError, OSError) as error:
        raise X13PipelineError(
            "capture static manifest or raw object verification failed"
        ) from error
    manifest = verified.record.manifest
    if (
        manifest.manifest_sha256 != receipt.manifest_sha256
        or manifest.object_sha256 != receipt.object_sha256
        or manifest.byte_length != receipt.byte_length
        or verified.record.object_path.resolve() != receipt.object_path.resolve()
        or verified.record.manifest_path.resolve()
        != receipt.manifest_path.resolve()
    ):
        raise X13PipelineError("capture receipt does not bind verified raw bytes")
    if (
        manifest.license_ref not in _REQUIRED_LICENSE_REFS
        or manifest.license_status in {"unknown", "blocked"}
    ):
        raise X13PipelineError("capture manifest has an unknown or blocked license")
    resource = _resource(manifest.source_url)
    payload = _strict_json(
        verified.object_bytes,
        context=f"capture {receipt.manifest_sha256}",
        exact_decimal_numbers=resource == "polymarket_trades",
    )
    return VerifiedCaptureDocumentV1(
        game_id=game_id,
        resource=resource,
        manifest_sha256=manifest.manifest_sha256,
        object_sha256=manifest.object_sha256,
        byte_length=manifest.byte_length,
        source_url=manifest.source_url,
        source_request=MappingProxyType(dict(manifest.source_request)),
        source_cursor=manifest.source_cursor,
        coverage=manifest.coverage,
        dataset_id=manifest.dataset_id,
        license_ref=manifest.license_ref,
        license_status=manifest.license_status,
        payload=payload,
    )


def _cursor_chain(
    documents: Sequence[VerifiedCaptureDocumentV1],
    *,
    item_field: str,
    context: str,
) -> list[Mapping[str, object]]:
    by_cursor: dict[str | None, VerifiedCaptureDocumentV1] = {}
    for document in documents:
        params = dict(document.source_request.get("params", {}))
        cursor = params.get("cursor")
        if cursor is not None and type(cursor) is not str:
            raise X13PipelineError(f"{context} request cursor is malformed")
        if cursor in by_cursor:
            raise X13PipelineError(f"{context} has duplicate request cursor")
        by_cursor[cursor] = document
    current: str | None = None
    seen: set[str | None] = set()
    rows: list[Mapping[str, object]] = []
    while True:
        if current in seen or current not in by_cursor:
            raise X13PipelineError(f"{context} cursor chain is incomplete")
        seen.add(current)
        payload = by_cursor[current].payload
        if (
            type(payload) is not dict
            or set(payload) != {item_field, "cursor"}
            or type(payload[item_field]) is not list
            or type(payload["cursor"]) is not str
        ):
            raise X13PipelineError(f"{context} page is malformed")
        for row in payload[item_field]:
            if not isinstance(row, Mapping):
                raise X13PipelineError(f"{context} item is malformed")
            rows.append(row)
        next_cursor = payload["cursor"]
        if next_cursor == "":
            break
        current = next_cursor
    if seen != set(by_cursor):
        raise X13PipelineError(f"{context} has an unbound pagination page")
    return rows


def _validate_game_capture_proofs(
    result: GameCaptureResult,
    game: UniverseGameV1,
    documents: tuple[VerifiedCaptureDocumentV1, ...],
) -> None:
    plan = game.capture_plan
    poly_targets = tuple(
        target for target in plan.targets if target.venue == "polymarket"
    )
    kalshi_targets = tuple(
        target for target in plan.targets if target.venue == "kalshi"
    )
    by_hash = {document.manifest_sha256: document for document in documents}
    gamma = tuple(
        document for document in documents if document.resource == "gamma_event"
    )
    if len(gamma) != 1 or type(gamma[0].payload) is not list:
        raise X13PipelineError(f"{game.game_id} Gamma metadata proof is missing")
    gamma_payload = gamma[0].payload
    if (
        len(gamma_payload) != 1
        or type(gamma_payload[0]) is not dict
        or gamma_payload[0].get("id") != game.polymarket_event_id
        or type(gamma_payload[0].get("markets")) is not list
        or result.polymarket_market_count != len(gamma_payload[0]["markets"])
        or result.polymarket_market_count != len(poly_targets)
    ):
        raise X13PipelineError(f"{game.game_id} Gamma inventory changed")

    windows_by_condition: dict[str, list[Any]] = {}
    for window in result.polymarket_terminal_windows:
        if (
            window.terminal_proof != "unsaturated_time_window"
            or type(window.item_count) is not int
            or window.item_count < 0
            or window.item_count >= POLYMARKET_TRADE_LIMIT
            or window.manifest_sha256 not in by_hash
        ):
            raise X13PipelineError(
                f"{game.game_id} Polymarket terminal proof is missing or "
                "saturated"
            )
        document = by_hash[window.manifest_sha256]
        params = dict(document.source_request.get("params", {}))
        if (
            document.resource != "polymarket_trades"
            or params.get("market") != window.condition_id
            or params.get("start") != window.start_ts
            or params.get("end") != window.end_ts
            or type(document.payload) is not list
            or len(document.payload) != window.item_count
        ):
            raise X13PipelineError(
                f"{game.game_id} Polymarket terminal receipt changed"
            )
        for trade in document.payload:
            if (
                type(trade) is not dict
                or trade.get("conditionId") != window.condition_id
                or type(trade.get("timestamp")) is not int
                or not window.start_ts
                <= trade["timestamp"]
                <= window.end_ts
            ):
                raise X13PipelineError(
                    f"{game.game_id} Polymarket trade escaped terminal window"
                )
        windows_by_condition.setdefault(window.condition_id, []).append(window)
    expected_conditions = {
        target.condition_id for target in poly_targets if target.condition_id
    }
    if set(windows_by_condition) != expected_conditions:
        raise X13PipelineError(
            f"{game.game_id} Polymarket conditions lack terminal proof"
        )
    for condition_id, windows in windows_by_condition.items():
        ordered = sorted(windows, key=lambda item: item.start_ts)
        if (
            ordered[0].start_ts != plan.start_ts
            or ordered[-1].end_ts != plan.end_ts
            or any(
                previous.end_ts + 1 != following.start_ts
                for previous, following in zip(ordered, ordered[1:])
            )
        ):
            raise X13PipelineError(
                f"{game.game_id} Polymarket window coverage is incomplete"
            )
    if result.polymarket_trade_count != sum(
        window.item_count for window in result.polymarket_terminal_windows
    ):
        raise X13PipelineError(
            f"{game.game_id} Polymarket trade count proof changed"
        )

    market_documents = tuple(
        document
        for document in documents
        if document.resource == "kalshi_markets"
    )
    target_series = {
        target.kalshi_series_ticker for target in kalshi_targets
    }
    selected_markets: dict[str, Mapping[str, object]] = {}
    for series in sorted(value for value in target_series if value is not None):
        series_documents = tuple(
            document
            for document in market_documents
            if dict(document.source_request.get("params", {})).get(
                "series_ticker"
            )
            == series
        )
        rows = _cursor_chain(
            series_documents,
            item_field="markets",
            context=f"{game.game_id} {series} market",
        )
        for row in rows:
            ticker = row.get("ticker")
            if type(ticker) is str:
                selected_markets[ticker] = row
    expected_tickers = {
        target.venue_market_id for target in kalshi_targets
    }
    if (
        set(selected_markets) & expected_tickers != expected_tickers
        or result.kalshi_market_count != len(expected_tickers)
    ):
        raise X13PipelineError(
            f"{game.game_id} Kalshi market cursor proof changed"
        )

    proofs = {proof.ticker: proof for proof in result.kalshi_trade_proofs}
    if set(proofs) != expected_tickers:
        raise X13PipelineError(
            f"{game.game_id} Kalshi trade proofs are incomplete"
        )
    for ticker, proof in proofs.items():
        if (
            proof.terminal_proof != "explicit_empty_cursor"
            or proof.terminal_cursor != ""
        ):
            raise X13PipelineError(
                f"{game.game_id} Kalshi trade terminal proof is missing"
            )
        trade_documents = tuple(
            document
            for document in documents
            if document.resource == "kalshi_trades"
            and dict(document.source_request.get("params", {})).get("ticker")
            == ticker
        )
        rows = _cursor_chain(
            trade_documents,
            item_field="trades",
            context=f"{game.game_id} {ticker} trades",
        )
        canonical_by_id: dict[str, bytes] = {}
        duplicate_count = 0
        for row in rows:
            trade_id = row.get("trade_id")
            if (
                type(trade_id) is not str
                or not trade_id
                or row.get("ticker") != ticker
            ):
                raise X13PipelineError(
                    f"{game.game_id} Kalshi trade identity changed"
                )
            canonical = _json_bytes(dict(row))
            prior = canonical_by_id.get(trade_id)
            if prior is None:
                canonical_by_id[trade_id] = canonical
            elif prior == canonical:
                duplicate_count += 1
            else:
                raise X13PipelineError(
                    f"{game.game_id} Kalshi stable trade ID collided"
                )
        if (
            proof.page_count != len(trade_documents)
            or proof.raw_item_count != len(rows)
            or proof.unique_item_count != len(canonical_by_id)
            or proof.exact_duplicate_count != duplicate_count
        ):
            raise X13PipelineError(
                f"{game.game_id} Kalshi trade count proof changed"
            )

    candles = {item.ticker: item for item in result.kalshi_candlesticks}
    eligible_tickers = {
        target.venue_market_id
        for target in kalshi_targets
        if target.analysis_eligible
    }
    if set(candles) != eligible_tickers:
        raise X13PipelineError(
            f"{game.game_id} Kalshi candle proofs are incomplete"
        )
    for ticker, proof in candles.items():
        if proof.period_interval != 1 or proof.manifest_sha256 not in by_hash:
            raise X13PipelineError(
                f"{game.game_id} Kalshi candle proof is invalid"
            )
        document = by_hash[proof.manifest_sha256]
        payload = document.payload
        if (
            document.resource != "kalshi_candlesticks"
            or type(payload) is not dict
            or set(payload) != {"ticker", "candlesticks"}
            or payload["ticker"] != ticker
            or type(payload["candlesticks"]) is not list
            or len(payload["candlesticks"]) != proof.candlestick_count
        ):
            raise X13PipelineError(
                f"{game.game_id} Kalshi candle receipt changed"
            )
        ends = [row.get("end_period_ts") for row in payload["candlesticks"]]
        if (
            any(type(value) is not int for value in ends)
            or (
                ends
                and (
                    ends[0] != proof.first_end_period_ts
                    or ends[-1] != proof.last_end_period_ts
                    or ends != sorted(set(ends))
                )
            )
            or (not ends and proof.first_end_period_ts is not None)
            or (not ends and proof.last_end_period_ts is not None)
        ):
            raise X13PipelineError(
                f"{game.game_id} Kalshi candle coverage changed"
            )


def verify_x13_capture_evidence(
    capture: BatchCaptureResult,
    *,
    universe: X13UniverseBatchV1,
    program_root: str | Path,
    raw_store_root: str | Path,
) -> VerifiedX13CaptureEvidenceV1:
    """Re-open pointer, manifests, raw bytes, and all terminal proofs."""

    if not isinstance(capture, BatchCaptureResult):
        raise X13PipelineError("capture has an unknown type")
    if not isinstance(universe, X13UniverseBatchV1):
        raise X13PipelineError("universe has an unknown type")
    program = Path(program_root).resolve()
    raw_root = Path(raw_store_root).resolve()
    _verify_pointer(
        capture,
        universe=universe,
        program_root=program,
        raw_store_root=raw_root,
    )
    if (
        tuple(game.game_id for game in capture.games) != X13_GAME_IDS
        or len(capture.games) != 20
    ):
        raise X13PipelineError("capture results must cover the exact frozen 20")
    receipt_game: dict[str, str] = {}
    for result in capture.games:
        for receipt in result.manifests:
            if receipt.manifest_sha256 in receipt_game:
                raise X13PipelineError(
                    "capture receipt belongs to multiple games"
                )
            receipt_game[receipt.manifest_sha256] = result.game_id
    documents: list[VerifiedCaptureDocumentV1] = []
    for receipt in capture.manifests:
        documents.append(
            _receipt_document(
                receipt,
                game_id=receipt_game.get(receipt.manifest_sha256),
                program_root=program,
                raw_store_root=raw_root,
            )
        )
    if {
        document.manifest_sha256 for document in documents
    } != set(capture.pointer.raw_manifest_sha256s):
        raise X13PipelineError(
            "verified raw manifest set differs from capture pointer"
        )
    by_game: dict[str, tuple[VerifiedCaptureDocumentV1, ...]] = {}
    universe_by_game = {game.game_id: game for game in universe.games}
    capture_by_game = {game.game_id: game for game in capture.games}
    for game_id in X13_GAME_IDS:
        game_documents = tuple(
            document for document in documents if document.game_id == game_id
        )
        _validate_game_capture_proofs(
            capture_by_game[game_id],
            universe_by_game[game_id],
            game_documents,
        )
        by_game[game_id] = game_documents
    batch_documents = tuple(
        document for document in documents if document.game_id is None
    )
    cutoff = [
        document
        for document in batch_documents
        if document.resource == "kalshi_cutoff"
    ]
    if (
        len(cutoff) != 1
        or cutoff[0].manifest_sha256
        != capture.kalshi_cutoff.manifest_sha256
    ):
        raise X13PipelineError("Kalshi historical cutoff proof is missing")
    cutoff_payload = cutoff[0].payload
    cutoff_fields = {
        "market_positions_last_updated_ts": (
            capture.kalshi_cutoff.market_positions_last_updated_ts
        ),
        "market_settled_ts": capture.kalshi_cutoff.market_settled_ts,
        "orders_updated_ts": capture.kalshi_cutoff.orders_updated_ts,
        "trades_created_ts": capture.kalshi_cutoff.trades_created_ts,
    }
    if (
        type(cutoff_payload) is not dict
        or set(cutoff_payload) != set(cutoff_fields)
        or any(
            _parse_utc_text(
                cutoff_payload[field],
                context=f"Kalshi cutoff {field}",
            )
            != expected
            for field, expected in cutoff_fields.items()
        )
    ):
        raise X13PipelineError(
            "Kalshi historical cutoff fields differ from immutable raw"
        )
    historical_end = min(
        capture.kalshi_cutoff.market_settled_ts,
        capture.kalshi_cutoff.trades_created_ts,
    ).timestamp()
    if any(
        plan.end_ts >= historical_end for plan in universe.capture_plans
    ):
        raise X13PipelineError(
            "capture window is not wholly before the frozen historical cutoff"
        )
    if any(
        document.resource != "kalshi_cutoff"
        for document in batch_documents
    ):
        raise X13PipelineError("capture contains an unbound batch receipt")
    return VerifiedX13CaptureEvidenceV1(
        capture_sha256=capture.pointer.capture_sha256,
        raw_manifest_sha256s=capture.pointer.raw_manifest_sha256s,
        documents_by_game=MappingProxyType(by_game),
        batch_documents=batch_documents,
        terminal_proofs_verified=True,
        manifest_hashes_verified=True,
    )


def _verify_universe_manifest_hashes(
    universe: X13UniverseBatchV1,
    capture_evidence: VerifiedX13CaptureEvidenceV1,
    *,
    program_root: Path,
    raw_store_root: Path,
) -> tuple[VerifiedCaptureDocumentV1, ...]:
    required = {
        universe.series_registry_proof.manifest_sha256,
        *(
            digest
            for game in universe.games
            for digest in game.metadata_manifest_sha256s
        ),
    }
    known = {
        document.manifest_sha256: document
        for documents in capture_evidence.documents_by_game.values()
        for document in documents
    }
    known.update(
        {
            document.manifest_sha256: document
            for document in capture_evidence.batch_documents
        }
    )
    external: list[VerifiedCaptureDocumentV1] = []
    for digest in sorted(required):
        _require_sha256(digest, "universe metadata manifest")
        if digest in known:
            continue
        matches = list(
            raw_store_root.glob(
                f"manifests/**/{digest.removeprefix('sha256:')}.manifest.json"
            )
        )
        if len(matches) != 1:
            raise X13PipelineError(
                "universe metadata manifest cannot be resolved uniquely"
            )
        try:
            verified = read_verified_static_object(
                matches[0],
                store_root=raw_store_root,
                program_root=program_root,
            )
        except (StaticStoreError, OSError) as error:
            raise X13PipelineError(
                "universe metadata raw object verification failed"
            ) from error
        manifest = verified.record.manifest
        if (
            manifest.manifest_sha256 != digest
            or manifest.license_ref not in _REQUIRED_LICENSE_REFS
            or manifest.license_status in {"unknown", "blocked"}
        ):
            raise X13PipelineError(
                "universe metadata license or hash is invalid"
            )
        external.append(
            VerifiedCaptureDocumentV1(
                game_id=None,
                resource=_resource(manifest.source_url),
                manifest_sha256=manifest.manifest_sha256,
                object_sha256=manifest.object_sha256,
                byte_length=manifest.byte_length,
                source_url=manifest.source_url,
                source_request=MappingProxyType(dict(manifest.source_request)),
                source_cursor=manifest.source_cursor,
                coverage=manifest.coverage,
                dataset_id=manifest.dataset_id,
                license_ref=manifest.license_ref,
                license_status=manifest.license_status,
                payload=_strict_json(
                    verified.object_bytes,
                    context="universe metadata",
                ),
            )
        )
    return tuple(external)


def _raw_contract_map(
    game: UniverseGameV1,
) -> dict[str, UniverseContractV1]:
    result: dict[str, UniverseContractV1] = {}
    for contract in game.contracts:
        for raw_id in contract.raw_contract_ids:
            if raw_id in result:
                raise X13PipelineError("raw market ID maps to multiple contracts")
            result[raw_id] = contract
        if contract.venue == "polymarket" and contract.condition_id:
            result[contract.condition_id] = contract
    return result


def _kalshi_yes_outcome(
    contract: UniverseContractV1,
    *,
    ticker: str,
    game_id: str,
) -> str:
    binding = next(
        binding
        for binding in X13_GAME_BINDINGS
        if binding.native_game_id == game_id
    )
    if contract.family == "moneyline" and set(contract.raw_contract_ids) == {
        binding.kalshi_away_ticker,
        binding.kalshi_home_ticker,
    }:
        return (
            binding.away_team
            if ticker == binding.kalshi_away_ticker
            else binding.home_team
        )
    yes = [outcome for outcome in contract.outcomes if outcome.casefold() == "yes"]
    if len(yes) != 1:
        raise X13PipelineError(
            f"Kalshi YES outcome is ambiguous for {contract.contract_id}"
        )
    return yes[0]


def _normalize_observations(
    game: UniverseGameV1,
    result: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
) -> tuple[MarketObservation, ...]:
    raw_map = _raw_contract_map(game)
    normalized_by_id = {
        contract.contract_id: contract
        for contract in normalize_x13_contracts(game)
    }
    by_hash = {document.manifest_sha256: document for document in documents}
    token_map: dict[str, tuple[str, str]] = {}
    for binding in game.polymarket_token_bindings:
        for outcome, token_id in zip(
            binding.outcomes, binding.token_ids, strict=True
        ):
            token_map[token_id] = (binding.condition_id, outcome)
    observations: list[MarketObservation] = []
    terminal_hashes = {
        window.manifest_sha256
        for window in result.polymarket_terminal_windows
    }
    for manifest_hash in sorted(terminal_hashes):
        document = by_hash[manifest_hash]
        assert isinstance(document.payload, list)
        for raw in document.payload:
            if type(raw) is not dict:
                raise X13PipelineError("Polymarket trade is malformed")
            token = raw.get("asset")
            token_binding = token_map.get(str(token))
            if (
                token_binding is None
                or raw.get("conditionId") != token_binding[0]
                or raw.get("outcome") != token_binding[1]
            ):
                raise X13PipelineError(
                    "Polymarket token/outcome orientation is unresolved"
                )
            contract = raw_map.get(token_binding[0])
            if contract is None:
                raise X13PipelineError(
                    "Polymarket trade references an unknown contract"
                )
            canonical_outcome = _canonical_market_outcome(
                game,
                contract,
                token_binding[1],
            )
            normalized_contract = normalized_by_id[contract.contract_id]
            try:
                observed = normalize_polymarket_trade(
                    raw,
                    raw_market_id=contract.contract_id,
                    logical_market_id=contract.logical_market_id,
                    outcome=canonical_outcome,
                )
                observations.append(observed)
                if len(normalized_contract.outcomes) == 2:
                    other = next(
                        outcome
                        for outcome in normalized_contract.outcomes
                        if outcome != observed.outcome
                    )
                    observations.append(
                        derive_complement(observed, outcome=other)
                    )
            except X13MarketError as error:
                raise X13PipelineError(
                    "Polymarket trade normalization failed"
                ) from error

    for document in documents:
        if document.resource not in {
            "kalshi_trades",
            "kalshi_candlesticks",
        }:
            continue
        params = dict(document.source_request.get("params", {}))
        ticker = params.get("ticker")
        if document.resource == "kalshi_candlesticks":
            if type(document.payload) is not dict:
                raise X13PipelineError("Kalshi candle response is malformed")
            ticker = document.payload.get("ticker")
        if type(ticker) is not str or ticker not in raw_map:
            raise X13PipelineError("Kalshi observation contract is unresolved")
        contract = raw_map[ticker]
        normalized_contract = normalized_by_id[contract.contract_id]
        yes_outcome = _canonical_market_outcome(
            game,
            contract,
            _kalshi_yes_outcome(
                contract,
                ticker=ticker,
                game_id=game.game_id,
            ),
        )
        try:
            if document.resource == "kalshi_trades":
                if (
                    type(document.payload) is not dict
                    or type(document.payload.get("trades")) is not list
                ):
                    raise X13PipelineError(
                        "Kalshi trade response is malformed"
                    )
                for raw in document.payload["trades"]:
                    observed = normalize_kalshi_trade(
                        raw,
                        ticker=ticker,
                        outcome=yes_outcome,
                        logical_market_id=contract.logical_market_id,
                    )
                    observations.append(observed)
                    if (
                        len(normalized_contract.outcomes) == 2
                        and len(contract.raw_contract_ids) == 1
                    ):
                        other = next(
                            outcome
                            for outcome in normalized_contract.outcomes
                            if outcome != yes_outcome
                        )
                        observations.append(
                            derive_complement(observed, outcome=other)
                        )
            else:
                candles = document.payload.get("candlesticks")
                if type(candles) is not list:
                    raise X13PipelineError(
                        "Kalshi candle response is malformed"
                    )
                for raw in candles:
                    observations.append(
                        normalize_kalshi_candle_bbo(
                            raw,
                            ticker=ticker,
                            outcome=yes_outcome,
                            logical_market_id=contract.logical_market_id,
                        )
                    )
        except X13MarketError as error:
            raise X13PipelineError(
                "Kalshi observation normalization failed"
            ) from error
    try:
        return deduplicate_observations(observations)
    except X13MarketError as error:
        raise X13PipelineError("normalized observation identity collided") from error


def _observation_contracts(
    observations: Sequence[MarketObservation],
    game: UniverseGameV1,
) -> Mapping[str, UniverseContractV1]:
    native: dict[tuple[str, str], UniverseContractV1] = {}
    logical: dict[tuple[str, str], UniverseContractV1] = {}
    for contract in game.contracts:
        for identity in {
            contract.contract_id,
            *contract.raw_contract_ids,
        }:
            key = (contract.venue, identity)
            if key in native and native[key] != contract:
                raise X13PipelineError(
                    "normalized observation native market is ambiguous"
                )
            native[key] = contract
        logical_key = (contract.venue, contract.logical_market_id)
        if logical_key in logical:
            raise X13PipelineError(
                "normalized observation logical market is ambiguous"
            )
        logical[logical_key] = contract
    result: dict[str, UniverseContractV1] = {}
    for observation in observations:
        contract = native.get(
            (observation.venue, observation.raw_market_id)
        )
        if contract is None and observation.logical_market_id is not None:
            contract = logical.get(
                (observation.venue, observation.logical_market_id)
            )
        if contract is None:
            raise X13PipelineError(
                "normalized observation logical market is ambiguous"
            )
        result[observation.observation_id] = contract
    return MappingProxyType(result)


def _contract_payload(
    contract: UniverseContractV1,
    normalized: MarketContractV1,
    observations: Sequence[MarketObservation],
) -> dict[str, object]:
    coverage = {
        outcome: (
            "available"
            if any(
                observation.outcome == outcome
                and observation.price is not None
                and observation.provenance == "observed"
                for observation in observations
            )
            else "unavailable_no_trades"
        )
        for outcome in normalized.outcomes
    }
    return {
        "contract_id": contract.contract_id,
        "condition_id": contract.condition_id,
        "venue_market_id": contract.venue_market_id,
        "raw_contract_ids": list(contract.raw_contract_ids),
        "logical_market_id": contract.logical_market_id,
        "venue": contract.venue,
        "family": contract.family,
        "period": contract.period,
        "subject": normalized.subject,
        "native_subject": contract.subject,
        "measure": normalized.measure,
        "comparator": normalized.comparator,
        "line": (
            None if normalized.line is None else format(normalized.line, "f")
        ),
        "kind": normalized.kind,
        "dependency_game_ids": list(normalized.dependency_game_ids),
        "analysis_eligible": normalized.analysis_eligible,
        "outcomes": list(normalized.outcomes),
        "native_outcomes": list(contract.outcomes),
        "outcome_coverage": coverage,
        "rule_sha256": contract.rule_sha256,
    }


def _observation_payload(
    observation: MarketObservation,
    contract: UniverseContractV1,
) -> dict[str, object]:
    if observation.logical_market_id != contract.logical_market_id:
        raise X13PipelineError(
            "observation logical market differs from contract inventory"
        )
    material = _observation_material(
        venue=observation.venue,
        raw_market_id=observation.raw_market_id,
        logical_market_id=observation.logical_market_id,
        outcome=observation.outcome,
        kind=observation.kind,
        source_interval=observation.source_interval,
        price=observation.price,
        size=observation.size,
        bid=observation.bid,
        ask=observation.ask,
        provenance=observation.provenance,
        native_id=observation.native_id,
        derived_from_observation_id=(
            observation.derived_from_observation_id
        ),
        native_source_time=observation.native_source_time,
        is_block_trade=observation.is_block_trade,
        taker_side=observation.taker_side,
        taker_outcome_side=observation.taker_outcome_side,
        yes_price_dollars=observation.yes_price_dollars,
        no_price_dollars=observation.no_price_dollars,
        primary_path_eligible=observation.primary_path_eligible,
        yes_bid_ohlc=observation.yes_bid_ohlc,
        yes_ask_ohlc=observation.yes_ask_ohlc,
        volume=observation.volume,
        open_interest=observation.open_interest,
    )
    return {
        "observation_id": observation.observation_id,
        "render_semantics": observation.render_semantics,
        **material,
    }


def _association_payload(
    row: AssociationResultV1,
    *,
    game_id: str,
    contract_by_id: Mapping[str, UniverseContractV1],
    episode_by_id: Mapping[str, object],
    interval_by_key: Mapping[tuple[str, int], object],
) -> dict[str, object]:
    if (
        row.contract_id not in contract_by_id
        or row.episode_id not in episode_by_id
        or (row.episode_id, row.delay_scenario_seconds) not in interval_by_key
    ):
        raise X13PipelineError("streamed association identity is unresolved")
    metrics = dict(row.candidate.metrics)
    contract = contract_by_id[row.contract_id]
    episode = episode_by_id[row.episode_id]
    episode_type = getattr(episode, "episode_type", None)
    if type(episode_type) is not str or not episode_type:
        raise X13PipelineError("streamed association episode is malformed")
    interval = interval_by_key[
        (row.episode_id, row.delay_scenario_seconds)
    ]
    start = getattr(interval, "start", None)
    end = getattr(interval, "end", None)
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise X13PipelineError("streamed association interval is malformed")
    return {
        "game_id": game_id,
        "episode_id": row.episode_id,
        "episode_type": episode_type,
        "contract_id": row.contract_id,
        "logical_market_id": contract.logical_market_id,
        "venue": row.venue,
        "family": contract.family,
        "outcome": row.outcome,
        "source_time_start_utc": _json_value(start),
        "source_time_end_utc": _json_value(end),
        "delay_scenario_seconds": row.delay_scenario_seconds,
        "horizon_seconds": row.horizon_seconds,
        "pre_event_actual_trade": _json_value(
            metrics["pre_event_actual_trade"]
        ),
        "first_post_event_trade": _json_value(
            metrics["first_post_event_trade"]
        ),
        "vwap": _json_value(metrics["vwap"]),
        "signed_price_change": _json_value(metrics["signed_price_change"]),
        "maximum_excursion": _json_value(metrics["maximum_excursion"]),
        "net_change_60s": _json_value(metrics["net_60_second_change"]),
        "trade_count": metrics["trade_count"],
        "volume": _json_value(metrics["volume"]),
        "staleness_seconds": _json_value(metrics["staleness_seconds"]),
        "overshoot_candidate": metrics["overshoot_candidate"],
        "reversal_candidate": metrics["reversal_candidate"],
        "two_venue_direction_consistency": metrics[
            "two_venue_direction_consistency"
        ],
        "order_ambiguous": row.order_ambiguous,
        "contaminated": row.contaminated,
        "validity_status": row.status,
    }


def _presentation_payload(
    row: Mapping[str, object],
) -> dict[str, object]:
    excluded = {"game_id", "contract_id", "venue", "family"}
    return {
        key: value
        for key, value in row.items()
        if key not in excluded
    }


class _AssociationPreviewSampler:
    """Build a bounded deterministic preview without canonical-order bias."""

    def __init__(self, *, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise X13PipelineError(
                "association preview limit must be a positive integer"
            )
        self._limit = limit
        self._reservoir_capacity = limit * 2
        self._first_by_market: dict[str, Mapping[str, object]] = {}
        self._seen_rows: set[str] = set()
        self._seen_strata: set[tuple[object, ...]] = set()
        self._stratum_reservoir: list[
            tuple[int, str, str, Mapping[str, object]]
        ] = []
        self._row_reservoir: list[
            tuple[int, str, str, Mapping[str, object]]
        ] = []

    @staticmethod
    def _stratum(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            row.get("logical_market_id"),
            row.get("delay_scenario_seconds"),
            row.get("horizon_seconds"),
            row.get("episode_type"),
            row.get("validity_status"),
        )

    def _offer(
        self,
        reservoir: list[tuple[int, str, str, Mapping[str, object]]],
        *,
        score: int,
        ordering_key: str,
        row_key: str,
        row: Mapping[str, object],
    ) -> None:
        entry = (-score, ordering_key, row_key, row)
        if len(reservoir) < self._reservoir_capacity:
            heapq.heappush(reservoir, entry)
        elif entry > reservoir[0]:
            heapq.heapreplace(reservoir, entry)

    def add(self, row: Mapping[str, object]) -> None:
        value = MappingProxyType(dict(row))
        logical_market_id = value.get("logical_market_id")
        if type(logical_market_id) is not str or not logical_market_id:
            raise X13PipelineError(
                "association preview row lacks logical market identity"
            )
        self._first_by_market.setdefault(logical_market_id, value)

        row_key = canonical_sha256(dict(value))
        if row_key in self._seen_rows:
            return
        self._seen_rows.add(row_key)
        row_score = int(row_key.removeprefix("sha256:"), 16)
        self._offer(
            self._row_reservoir,
            score=row_score,
            ordering_key=row_key,
            row_key=row_key,
            row=value,
        )

        stratum = self._stratum(value)
        if stratum in self._seen_strata:
            return
        self._seen_strata.add(stratum)
        stratum_key = json.dumps(
            stratum,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        stratum_sha256 = canonical_sha256(list(stratum))
        self._offer(
            self._stratum_reservoir,
            score=int(stratum_sha256.removeprefix("sha256:"), 16),
            ordering_key=stratum_key,
            row_key=row_key,
            row=value,
        )

    @staticmethod
    def _ordered(
        reservoir: Sequence[
            tuple[int, str, str, Mapping[str, object]]
        ],
    ) -> list[Mapping[str, object]]:
        return [
            row
            for negative_score, ordering_key, row_key, row in sorted(
                reservoir,
                key=lambda item: (-item[0], item[1], item[2]),
            )
        ]

    def finalize(self) -> tuple[Mapping[str, object], ...]:
        if len(self._first_by_market) > self._limit:
            raise X13PipelineError(
                "association preview cannot cover every eligible market "
                "within its frozen row limit"
            )
        selected: list[Mapping[str, object]] = []
        selected_ids: set[str] = set()

        def append(row: Mapping[str, object]) -> None:
            row_id = canonical_sha256(dict(row))
            if row_id in selected_ids or len(selected) >= self._limit:
                return
            selected_ids.add(row_id)
            selected.append(row)

        for market_id in sorted(self._first_by_market):
            append(self._first_by_market[market_id])
        for row in self._ordered(self._stratum_reservoir):
            append(row)
        for row in self._ordered(self._row_reservoir):
            append(row)
        return tuple(selected)


def _has_actual_market_evidence(row: Mapping[str, object]) -> bool:
    return (
        row.get("first_post_event_trade") is not None
        or (
            type(row.get("trade_count")) is int
            and int(row["trade_count"]) > 0
        )
    )


def _association_arrow_schema() -> object:
    try:
        import pyarrow as pa
    except ImportError as error:  # pragma: no cover - workspace dependency
        raise X13PipelineError(
            "pyarrow is required for the full association table"
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
            for name, type_name, nullable in _ASSOCIATION_SCHEMA_FIELDS
        ],
        metadata={
            b"schema": b"nfl_x13_association_candidate_parquet_v1",
            b"schema_fingerprint": _ASSOCIATION_SCHEMA_FINGERPRINT.encode(),
            b"experiment_id": X13_EXPERIMENT_ID.encode(),
            b"claim_scope": b"PRELIMINARY_SOURCE_TIME_ONLY",
        },
    )


def _partition_value(value: str, field: str) -> str:
    if _PARTITION_VALUE_RE.fullmatch(value) is None:
        raise X13PipelineError(
            f"{field} cannot be represented as a canonical partition"
        )
    return value


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            byte_length += len(chunk)
    return "sha256:" + digest.hexdigest(), byte_length


class _AssociationWriteDiskGate:
    """Serialize partition flushes around the five-GiB reserve check."""

    def __init__(self, staging_root: Path) -> None:
        self._staging_root = staging_root
        self._lock = threading.Lock()
        self._flush_count = 0

    @property
    def flush_count(self) -> int:
        return self._flush_count

    def write(self, operation: Callable[[], None]) -> None:
        with self._lock:
            before = shutil.disk_usage(self._staging_root).free
            if before < _UNTOUCHED_RESERVE_BYTES + _TEMP_HEADROOM_BYTES:
                raise X13PipelineError(
                    "association running disk gate failed before partition "
                    "flush"
                )
            operation()
            self._flush_count += 1
            after = shutil.disk_usage(self._staging_root).free
            if after < _UNTOUCHED_RESERVE_BYTES:
                raise X13PipelineError(
                    "association running disk gate consumed untouched reserve"
                )


class _AssociationParquetSink:
    def __init__(
        self,
        staging_root: Path,
        game_id: str,
        *,
        disk_gate: _AssociationWriteDiskGate,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - workspace dependency
            raise X13PipelineError(
                "pyarrow is required for the full association table"
            ) from error
        self._pq = pq
        self._staging_root = staging_root
        self._game_id = game_id
        self._disk_gate = disk_gate
        self._schema = _association_arrow_schema()
        self._writers: dict[tuple[str, str], object] = {}
        self._paths: dict[tuple[str, str], tuple[str, Path]] = {}
        self._buffers: dict[
            tuple[str, str], list[Mapping[str, object]]
        ] = {}
        self._counts: Counter[tuple[str, str]] = Counter()

    def _open(self, key: tuple[str, str]) -> object:
        venue, family = key
        relative_path = (
            "data/associations/"
            f"game_id={self._game_id}/"
            f"venue={_partition_value(venue, 'venue')}/"
            f"family={_partition_value(family, 'family')}/"
            "part-00000.parquet"
        )
        path = self._staging_root.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        writers: list[object] = []
        self._disk_gate.write(
            lambda: writers.append(
                self._pq.ParquetWriter(
                    path,
                    self._schema,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=True,
                    write_statistics=True,
                    version="2.6",
                    data_page_version="1.0",
                )
            )
        )
        writer = writers[0]
        self._writers[key] = writer
        self._paths[key] = (relative_path, path)
        self._buffers[key] = []
        return writer

    def _flush(self, key: tuple[str, str]) -> None:
        buffer = self._buffers[key]
        if not buffer:
            return
        try:
            import pyarrow as pa
        except ImportError as error:  # pragma: no cover
            raise X13PipelineError("pyarrow disappeared during execution") from error
        table = pa.Table.from_pylist(buffer, schema=self._schema)
        self._disk_gate.write(
            lambda: self._writers[key].write_table(
                table,
                row_group_size=_ASSOCIATION_ROW_GROUP_SIZE,
            )
        )
        buffer.clear()

    def append(self, row: Mapping[str, object]) -> None:
        key = (str(row["venue"]), str(row["family"]))
        if key not in self._writers:
            self._open(key)
        self._buffers[key].append(row)
        self._counts[key] += 1
        if len(self._buffers[key]) >= _ASSOCIATION_ROW_GROUP_SIZE:
            self._flush(key)

    def close(self) -> tuple[AuxiliaryArtifactV1, ...]:
        values: list[AuxiliaryArtifactV1] = []
        for key in sorted(self._writers):
            self._flush(key)
            self._disk_gate.write(self._writers[key].close)
            relative_path, path = self._paths[key]
            digest, byte_length = _file_identity(path)
            values.append(
                AuxiliaryArtifactV1(
                    relative_path=relative_path,
                    source_path=path,
                    role="association_candidate_table",
                    media_type="application/vnd.apache.parquet",
                    schema_fingerprint=_ASSOCIATION_SCHEMA_FINGERPRINT,
                    row_count=self._counts[key],
                    sha256=digest,
                    byte_length=byte_length,
                )
            )
        return tuple(values)


def _association_payload_inputs(
    prepared: _PreparedGameV1,
) -> tuple[
    Mapping[str, UniverseContractV1],
    Mapping[str, object],
    Mapping[tuple[str, int], object],
]:
    stream = prepared.association_stream
    return (
        {
            contract.contract_id: contract
            for contract in prepared.universe_game.contracts
        },
        {
            episode.episode_id: episode
            for episode in prepared.ledger.episodes
        },
        {
            (interval.episode_id, interval.delay_scenario_seconds): (
                interval.source_interval
            )
            for interval in stream.layer_g_audit.episode_intervals
        },
    )


def _run_association_storage_pilot(
    prepared: _PreparedGameV1,
    *,
    staging_root: Path,
    disk_gate: _AssociationWriteDiskGate,
) -> AssociationStoragePilotV1:
    started = time.perf_counter_ns()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - workspace dependency
        raise X13PipelineError(
            "pyarrow is required for the association storage pilot"
        ) from error
    contract_by_id, episode_by_id, interval_by_key = (
        _association_payload_inputs(prepared)
    )
    rows: list[Mapping[str, object]] = []
    for association in iter_x13_associations(
        prepared.association_stream
    ):
        rows.append(
            _association_payload(
                association,
                game_id=prepared.universe_game.game_id,
                contract_by_id=contract_by_id,
                episode_by_id=episode_by_id,
                interval_by_key=interval_by_key,
            )
        )
        if len(rows) == _ASSOCIATION_PILOT_ROW_LIMIT:
            break
    if not rows:
        raise X13PipelineError("association storage pilot produced no rows")
    pilot_directory = staging_root / "pilot"
    pilot_directory.mkdir(parents=True, exist_ok=True)
    pilot_path = pilot_directory / "association-storage-pilot.parquet"
    table = pa.Table.from_pylist(rows, schema=_association_arrow_schema())
    disk_gate.write(
        lambda: pq.write_table(
            table,
            pilot_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
            row_group_size=_ASSOCIATION_ROW_GROUP_SIZE,
        )
    )
    byte_length = pilot_path.stat().st_size
    pilot_path.unlink()
    pilot_directory.rmdir()
    return AssociationStoragePilotV1(
        row_limit=_ASSOCIATION_PILOT_ROW_LIMIT,
        row_count=len(rows),
        byte_length=byte_length,
        empirical_bytes_per_row=max(
            _ASSOCIATION_MIN_BYTES_PER_ROW,
            math.ceil(byte_length / len(rows)),
        ),
        elapsed_ns=max(1, time.perf_counter_ns() - started),
    )


def _stream_game_associations(
    prepared: _PreparedGameV1,
    *,
    staging_root: Path,
    disk_gate: _AssociationWriteDiskGate,
) -> _StreamedGameV1:
    started = time.perf_counter_ns()
    stream = prepared.association_stream
    contract_by_id, episode_by_id, interval_by_key = (
        _association_payload_inputs(prepared)
    )
    sink = _AssociationParquetSink(
        staging_root,
        prepared.universe_game.game_id,
        disk_gate=disk_gate,
    )
    presentation_sampler = _AssociationPreviewSampler(
        limit=_PRESENTATION_ROW_LIMIT_PER_GAME
    )
    presentation_eligible_count = 0
    contaminated: set[str] = set()
    actual_count = 0
    try:
        for association in iter_x13_associations(stream):
            row = _association_payload(
                association,
                game_id=prepared.universe_game.game_id,
                contract_by_id=contract_by_id,
                episode_by_id=episode_by_id,
                interval_by_key=interval_by_key,
            )
            sink.append(row)
            actual_count += 1
            if association.contaminated:
                contaminated.add(association.episode_id)
            if _has_actual_market_evidence(row) and (
                association.status == "OBSERVED"
                or association.order_ambiguous
                or association.contaminated
            ):
                presentation_eligible_count += 1
                presentation_sampler.add(_presentation_payload(row))
        artifacts = sink.close()
    except Exception:
        for writer in sink._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        raise
    if actual_count != stream.expected_association_count:
        raise X13PipelineError(
            f"{stream.game_id} streamed association cardinality mismatch"
        )
    if sum(
        artifact.row_count or 0 for artifact in artifacts
    ) != actual_count:
        raise X13PipelineError(
            f"{stream.game_id} Parquet partition cardinality mismatch"
        )
    presentation = presentation_sampler.finalize()
    expected_presentation_count = min(
        presentation_eligible_count,
        _PRESENTATION_ROW_LIMIT_PER_GAME,
    )
    if len(presentation) != expected_presentation_count:
        raise X13PipelineError(
            "association stratified preview did not fill its frozen row limit"
        )
    return _StreamedGameV1(
        presentation_rows=presentation,
        presentation_eligible_count=presentation_eligible_count,
        presentation_omitted_count=(
            presentation_eligible_count - len(presentation)
        ),
        auxiliary_artifacts=artifacts,
        actual_row_count=actual_count,
        contaminated_episode_ids=tuple(sorted(contaminated)),
        elapsed_ns=max(1, time.perf_counter_ns() - started),
    )


def _episode_payloads(
    ledger: X13GameLedgerV1,
    contaminated_episode_ids: Sequence[str],
) -> list[dict[str, object]]:
    event_by_id = {event.play_id: event for event in ledger.events}
    contaminated = set(contaminated_episode_ids)
    values: list[dict[str, object]] = []
    for episode in ledger.episodes:
        first = event_by_id[episode.play_ids[0]]
        last = event_by_id[episode.play_ids[-1]]
        row = _json_value(episode)
        assert isinstance(row, dict)
        row.update(
            {
                "source_time_start_utc": first.source_time_start_utc,
                "source_time_end_utc": last.source_time_end_utc,
                "contaminated": episode.episode_id in contaminated,
            }
        )
        values.append(row)
    return values


def _game_payload(
    *,
    prepared: _PreparedGameV1,
    streamed: _StreamedGameV1,
    authorization: X13PipelineAuthorizationV1,
) -> dict[str, object]:
    game_state = prepared.game_state
    universe_game = prepared.universe_game
    observations = prepared.observations
    ledger = prepared.ledger
    association_stream = prepared.association_stream
    associations = list(streamed.presentation_rows)
    binding = next(
        binding
        for binding in X13_GAME_BINDINGS
        if binding.native_game_id == universe_game.game_id
    )
    observation_contracts = _observation_contracts(
        observations, universe_game
    )
    observation_payloads = [
        _observation_payload(
            observation,
            observation_contracts[observation.observation_id],
        )
        for observation in observations
    ]
    observations_by_logical: dict[str, list[MarketObservation]] = {}
    for observation in observations:
        contract = observation_contracts[observation.observation_id]
        observations_by_logical.setdefault(
            contract.logical_market_id, []
        ).append(observation)
    normalized_contract_by_id = {
        contract.contract_id: contract for contract in prepared.contracts
    }
    contracts_payload = [
        _contract_payload(
            contract,
            normalized_contract_by_id[contract.contract_id],
            observations_by_logical.get(contract.logical_market_id, ()),
        )
        for contract in universe_game.contracts
    ]
    personnel = dict(game_state.personnel_summary)
    return {
        "schema": "nfl_x13_game_bundle_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": X13_STATUS,
        "game_id": universe_game.game_id,
        "teams": {
            "away": binding.away_team,
            "home": binding.home_team,
            "away_color": _FIELD_COLORS[0],
            "home_color": _FIELD_COLORS[1],
        },
        "final_score": {
            "away": binding.away_score,
            "home": binding.home_score,
        },
        "events": [_json_value(event) for event in ledger.events],
        "episodes": _episode_payloads(
            ledger, streamed.contaminated_episode_ids
        ),
        "personnel": {
            **_json_value(personnel),
            "coverage": "research_only",
            "complete_11v11_rows": personnel.get(
                "present_complete_11v11_rows", 0
            ),
        },
        "stat_ledger": [_json_value(entry) for entry in ledger.stat_ledger],
        "contracts": contracts_payload,
        "observations": observation_payloads,
        "associations": associations,
        "association_coverage": {
            "expected_full_rows": (
                association_stream.expected_association_count
            ),
            "actual_full_rows": streamed.actual_row_count,
            "presentation_row_count": len(associations),
            "presentation_eligible_count": (
                streamed.presentation_eligible_count
            ),
            "presentation_omitted_count": (
                streamed.presentation_omitted_count
            ),
            "presentation_row_limit": _PRESENTATION_ROW_LIMIT_PER_GAME,
            "presentation_selection_order": (
                "deterministic_market_coverage_then_hashed_stratum_reservoir_v1"
            ),
            "presentation_filter": (
                "actual_market_evidence_and_ambiguous_or_contaminated_status"
            ),
            "full_table_format": "partitioned_parquet",
            "full_table_partitions": [
                {
                    "relative_path": artifact.relative_path,
                    "row_count": artifact.row_count,
                    "sha256": artifact.sha256,
                    "schema_fingerprint": artifact.schema_fingerprint,
                }
                for artifact in streamed.auxiliary_artifacts
            ],
        },
        "market_coverage": {
            "polymarket_historical_bbo": "unavailable",
            "kalshi_historical_bbo": "one_minute_candle_only",
            "robinhood_historical_venue": (
                "not_counted_as_third_venue_underlying_exchange_or_fcm_only"
            ),
        },
        "market_observation_audit": {
            "total_observation_count": len(observations),
            "observed_count": sum(
                observation.provenance == "observed"
                for observation in observations
            ),
            "derived_complement_count": sum(
                observation.provenance == "derived_complement"
                for observation in observations
            ),
            "block_trade_count": sum(
                observation.is_block_trade is True
                and observation.provenance == "observed"
                for observation in observations
            ),
            "primary_path_observation_count": sum(
                observation.primary_path_eligible
                for observation in observations
            ),
            "primary_path_exclusion_policy": (
                "exclude_block_trades_and_all_derived_complements"
            ),
        },
        "audit": {
            "replay": "PASS",
            "personnel": "PASS_RESEARCH_ONLY",
            "layer_g": association_stream.layer_g_audit.status,
            "layer_m": association_stream.layer_m_audit.status,
            "layer_a": "PASS_SOURCE_TIME_ONLY",
            "manifest_hashes_verified": True,
            "unresolved": [
                "Team I terms review remains NOT_GREEN_OPEN; research-only"
            ],
        },
        "lineage": {
            "raw_manifest_hashes": list(prepared.raw_manifest_hashes),
            "builder_version": _BUILDER_VERSION,
            "analysis_lock": authorization.analysis_lock_sha256,
            "game_state_ledger_sha256": game_state.ledger_sha256,
        },
    }


def _write_auxiliary_json(
    staging_root: Path,
    *,
    relative_path: str,
    role: str,
    payload: Mapping[str, object],
    row_count: int | None,
) -> AuxiliaryArtifactV1:
    path = staging_root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return AuxiliaryArtifactV1(
        relative_path=relative_path,
        source_path=path,
        role=role,
        media_type="application/json",
        schema_fingerprint=canonical_sha256(
            {
                "schema": payload.get("schema"),
                "version": "canonical-json-v1",
            }
        ),
        row_count=row_count,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
    )


def _robinhood_provenance_inventory() -> dict[str, object]:
    return {
        "schema": "nfl_x13_robinhood_provenance_inventory_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": X13_STATUS,
        "classification": (
            "underlying_exchange_or_fcm_provenance_only"
        ),
        "independent_historical_venue_count": 0,
        "price_capture_performed": False,
        "double_count_with_kalshi_or_polymarket_permitted": False,
        "coverage_status": (
            "NOT_CAPTURED_NO_CONTRACT_LEVEL_EXCHANGE_OR_FCM_EVIDENCE"
        ),
        "records": [
            {
                "platform": "Robinhood",
                "underlying_exchange_or_fcm": None,
                "provenance_status": (
                    "requires_contract_level_disclosure_evidence"
                ),
                "historical_price_series": None,
                "analysis_inclusion": False,
            }
        ],
        "claim_boundary": (
            "Robinhood is not treated as a third historical venue and "
            "contributes no price observation"
        ),
    }


def _cross_game_mve_inventory(
    universe: X13UniverseBatchV1,
) -> dict[str, object]:
    records_by_id: dict[str, dict[str, object]] = {}
    listed_under: dict[str, set[str]] = {}
    for game in universe.games:
        for composite in game.cross_game_inventory:
            record = _json_value(composite)
            if not isinstance(record, dict):
                raise X13PipelineError(
                    "cross-game MVE inventory record is malformed"
                )
            market_id = composite.venue_market_id
            prior = records_by_id.get(market_id)
            if prior is not None and prior != record:
                raise X13PipelineError(
                    "cross-game MVE identity has conflicting definitions"
                )
            records_by_id[market_id] = record
            listed_under.setdefault(market_id, set()).add(game.game_id)
    records = [
        {
            **records_by_id[market_id],
            "listed_under_game_ids": sorted(listed_under[market_id]),
            "single_game_price_path_inclusion": False,
        }
        for market_id in sorted(records_by_id)
    ]
    return {
        "schema": "nfl_x13_cross_game_mve_inventory_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": X13_STATUS,
        "record_count": len(records),
        "records": records,
        "analysis_policy": (
            "inventory_only; cross-game contracts never enter a single-game "
            "price path or association"
        ),
    }


def _validation_report(
    *,
    prepared_games: Sequence[_PreparedGameV1],
    capture_evidence: VerifiedX13CaptureEvidenceV1,
    semantic_overlap: Mapping[str, object],
    expected_association_count: int,
    actual_association_count: int,
) -> dict[str, object]:
    games = [
        {
            "game_id": prepared.universe_game.game_id,
            "layer_g": prepared.association_stream.layer_g_audit.status,
            "layer_m_primary_path": (
                prepared.association_stream.layer_m_audit.status
            ),
            "replay": "PASS",
            "personnel": "PASS_RESEARCH_ONLY",
            "block_trade_count": sum(
                observation.is_block_trade is True
                and observation.provenance == "observed"
                for observation in prepared.observations
            ),
            "primary_path_observation_count": sum(
                observation.primary_path_eligible
                for observation in prepared.observations
            ),
        }
        for prepared in prepared_games
    ]
    passed = (
        len(games) == 20
        and all(
            game["layer_g"] == "PASS"
            and game["layer_m_primary_path"] == "PASS"
            for game in games
        )
        and capture_evidence.terminal_proofs_verified
        and capture_evidence.manifest_hashes_verified
        and expected_association_count == actual_association_count
        and semantic_overlap.get(
            "unapproved_cross_venue_orientation_count"
        )
        == 0
    )
    if not passed:
        raise X13PipelineError("final X-13 validation report did not pass")
    return {
        "schema": "nfl_x13_validation_report_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": "PASS_PRELIMINARY_SOURCE_TIME_ONLY",
        "game_count": len(games),
        "raw_manifest_hashes_verified": True,
        "terminal_pagination_proofs_verified": True,
        "association_expected_rows": expected_association_count,
        "association_actual_rows": actual_association_count,
        "approved_cross_venue_orientation_count": semantic_overlap[
            "approved_cross_venue_orientation_count"
        ],
        "unapproved_cross_venue_orientation_count": 0,
        "games": games,
        "claim_boundary": (
            "no causal, live latency, execution, or tradable-alpha claim"
        ),
    }


def _lineage_report(
    *,
    prepared_games: Sequence[_PreparedGameV1],
    authorization: X13PipelineAuthorizationV1,
    game_state_build: X13GameStateBuild,
    universe: X13UniverseBatchV1,
    capture: BatchCaptureResult,
) -> dict[str, object]:
    return {
        "schema": "nfl_x13_lineage_report_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": X13_STATUS,
        "builder_version": _BUILDER_VERSION,
        "plan_id": X13_BATCH_SPEC.plan_id,
        "game_state_build_id": game_state_build.manifest["build_id"],
        "universe_id": universe.universe_id,
        "capture_sha256": capture.pointer.capture_sha256,
        "capture_receipt_sha256": capture.pointer.receipt_sha256,
        "analysis_lock_sha256": authorization.analysis_lock_sha256,
        "raw_manifest_sha256s": list(
            capture.pointer.raw_manifest_sha256s
        ),
        "games": [
            {
                "game_id": prepared.universe_game.game_id,
                "game_state_ledger_sha256": (
                    prepared.game_state.ledger_sha256
                ),
                "raw_manifest_sha256s": list(
                    prepared.raw_manifest_hashes
                ),
            }
            for prepared in prepared_games
        ],
    }


def _contract_semantic_overlap_audit(
    prepared_games: Sequence[_PreparedGameV1],
) -> dict[str, object]:
    winner_records: list[dict[str, object]] = []
    matched_total_records: list[dict[str, object]] = []
    structured_total_contract_count = 0
    semantic_venues: dict[tuple[object, ...], set[str]] = {}
    approved_semantic_keys: set[tuple[object, ...]] = set()
    for prepared in prepared_games:
        game_id = prepared.universe_game.game_id
        binding = _game_binding(game_id)
        for contract in prepared.contracts:
            approved = (
                contract.family == "moneyline"
                and contract.period == "game"
                and contract.subject == game_id
                and contract.measure == "winner"
            ) or (
                contract.family == "total"
                and contract.period == "game"
                and contract.subject == game_id
                and contract.measure == "combined_points"
                and contract.comparator == "gt"
                and contract.line is not None
                and contract.outcomes == ("Over", "Under")
            )
            for outcome in contract.outcomes:
                semantic_key = (
                    game_id,
                    *_semantic_orientation_key(contract, outcome),
                )
                semantic_venues.setdefault(semantic_key, set()).add(
                    contract.venue
                )
                if approved:
                    approved_semantic_keys.add(semantic_key)
        winners = [
            contract
            for contract in prepared.contracts
            if contract.logical_market_id
            == f"{contract.venue}:{game_id}:winner"
        ]
        if (
            len(winners) != 2
            or {contract.venue for contract in winners}
            != {"polymarket", "kalshi"}
        ):
            raise X13PipelineError(
                f"{game_id} canonical winner overlap is incomplete"
            )
        orientation_records: list[dict[str, object]] = []
        for outcome in (binding.away_team, binding.home_team):
            keys = {
                (
                    contract.family,
                    contract.period,
                    contract.subject,
                    contract.measure,
                    contract.comparator,
                    contract.line,
                    outcome,
                )
                for contract in winners
            }
            if len(keys) != 1:
                raise X13PipelineError(
                    f"{game_id} winner semantic orientation is divergent"
                )
            orientation_records.append(
                {
                    "canonical_outcome": outcome,
                    "venues": ["kalshi", "polymarket"],
                    "semantic_key_sha256": canonical_sha256(
                        _json_value(next(iter(keys)))
                    ),
                }
            )
        winner_records.append(
            {
                "game_id": game_id,
                "status": "PASS",
                "canonical_subject": game_id,
                "orientation_count": 2,
                "orientations": orientation_records,
            }
        )
        totals_by_key: dict[
            tuple[object, ...],
            list[MarketContractV1],
        ] = {}
        for contract in prepared.contracts:
            if not (
                contract.family == "total"
                and contract.period == "game"
                and contract.subject == game_id
                and contract.measure == "combined_points"
                and contract.comparator == "gt"
                and contract.line is not None
                and contract.outcomes == ("Over", "Under")
            ):
                continue
            structured_total_contract_count += 1
            key = (
                contract.family,
                contract.period,
                contract.subject,
                contract.measure,
                contract.comparator,
                contract.line,
            )
            totals_by_key.setdefault(key, []).append(contract)
        for key, contracts in sorted(
            totals_by_key.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        ):
            venues = tuple(sorted(contract.venue for contract in contracts))
            if len(venues) != len(set(venues)):
                raise X13PipelineError(
                    f"{game_id} structured total has duplicate venue identity"
                )
            if venues != ("kalshi", "polymarket"):
                continue
            line = contracts[0].line
            assert line is not None
            orientations = []
            for outcome in ("Over", "Under"):
                orientations.append(
                    {
                        "canonical_outcome": outcome,
                        "semantic_key_sha256": canonical_sha256(
                            {
                                "family": key[0],
                                "period": key[1],
                                "subject": key[2],
                                "measure": key[3],
                                "comparator": key[4],
                                "line": format(line, "f"),
                                "orientation": (
                                    game_id
                                    if outcome == "Over"
                                    else f"NOT:{game_id}"
                                ),
                            }
                        ),
                    }
                )
            matched_total_records.append(
                {
                    "game_id": game_id,
                    "line": format(line, "f"),
                    "venues": list(venues),
                    "contract_ids": sorted(
                        contract.contract_id for contract in contracts
                    ),
                    "orientation_count": len(orientations),
                    "orientations": orientations,
                }
            )
    actual_cross_venue = {
        key
        for key, venues in semantic_venues.items()
        if venues == {"kalshi", "polymarket"}
    }
    approved_cross_venue = actual_cross_venue & approved_semantic_keys
    unapproved_cross_venue = actual_cross_venue - approved_semantic_keys
    if actual_cross_venue != approved_cross_venue:
        raise X13PipelineError(
            "unapproved raw-string semantic identity entered cross-venue "
            f"association: {len(unapproved_cross_venue)} orientations"
        )
    return {
        "schema": "nfl_x13_contract_semantic_overlap_audit_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": "PASS",
        "winner_game_count": len(winner_records),
        "winner_orientation_count": sum(
            int(record["orientation_count"]) for record in winner_records
        ),
        "winner_expected_game_count": 20,
        "winner_expected_orientation_count": 40,
        "structured_total_contract_count": structured_total_contract_count,
        "matched_total_line_count": len(matched_total_records),
        "matched_total_orientation_count": sum(
            int(record["orientation_count"])
            for record in matched_total_records
        ),
        "actual_cross_venue_orientation_count": len(actual_cross_venue),
        "approved_cross_venue_orientation_count": len(
            approved_cross_venue
        ),
        "unapproved_cross_venue_orientation_count": len(
            unapproved_cross_venue
        ),
        "approved_cross_venue_semantic_key_sha256s": sorted(
            canonical_sha256(_json_value(list(key)))
            for key in approved_cross_venue
        ),
        "winner_records": winner_records,
        "matched_total_records": matched_total_records,
        "unresolved_family_policy": (
            "team, player, period, spread, and composite propositions remain "
            "SINGLE_VENUE unless structured native fields prove the same "
            "subject, period, comparator, line, and orientation; titles are "
            "not parsed"
        ),
    }


def _exploration_readiness_report(
    *,
    association_paths: Sequence[Path],
    expected_candidate_row_count: int,
    input_root_analysis_lock_sha256: str,
) -> dict[str, object]:
    """Count independent candidate units and refuse an unfrozen estimand."""

    if (
        not association_paths
        or type(expected_candidate_row_count) is not int
        or expected_candidate_row_count <= 0
    ):
        raise X13PipelineError(
            "exploration readiness requires full association evidence"
        )
    _require_sha256(
        input_root_analysis_lock_sha256,
        "input_root_analysis_lock_sha256",
    )
    resolved_paths: list[str] = []
    partition_sha256s: list[str] = []
    for path in association_paths:
        if not isinstance(path, Path):
            raise X13PipelineError(
                "exploration association path must be a Path"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise X13PipelineError(
                "exploration association evidence is missing"
            ) from error
        if not resolved.is_file() or resolved.suffix != ".parquet":
            raise X13PipelineError(
                "exploration association evidence must be Parquet"
            )
        resolved_paths.append(str(resolved))
        partition_sha256s.append(_file_identity(resolved)[0])
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - direct dependency
        raise X13PipelineError(
            "duckdb is required for exploration readiness audit"
        ) from error

    connection = duckdb.connect(database=":memory:")
    try:
        connection.read_parquet(resolved_paths).create_view(
            "association_candidates"
        )
        candidate_row_count = int(
            connection.execute(
                "SELECT count(*) FROM association_candidates"
            ).fetchone()[0]
        )
        if candidate_row_count != expected_candidate_row_count:
            raise X13PipelineError(
                "exploration candidate cardinality changed"
            )
        connection.execute(
            """
            CREATE TEMP VIEW observed_candidates AS
            SELECT *
            FROM association_candidates
            WHERE validity_status = 'OBSERVED'
              AND NOT contaminated
              AND NOT order_ambiguous
            """
        )
        (
            observed_row_count,
            unique_episode_count,
            episode_venue_unit_count,
            observed_game_count,
        ) = (
            int(value)
            for value in connection.execute(
                """
                SELECT
                    count(*),
                    count(DISTINCT (game_id, episode_id)),
                    count(DISTINCT (game_id, episode_id, venue)),
                    count(DISTINCT game_id)
                FROM observed_candidates
                """
            ).fetchone()
        )
        unit_distribution = connection.execute(
            """
            WITH units AS (
                SELECT game_id, episode_id, venue, count(*) AS unit_rows
                FROM observed_candidates
                GROUP BY game_id, episode_id, venue
            )
            SELECT
                count(*),
                count_if(unit_rows > 1),
                coalesce(quantile_disc(unit_rows, 0.50), 0),
                coalesce(quantile_disc(unit_rows, 0.95), 0),
                coalesce(max(unit_rows), 0)
            FROM units
            """
        ).fetchone()
        by_venue = [
            {
                "venue": str(venue),
                "observed_row_count": int(row_count),
                "unique_episode_count": int(episode_count),
                "game_count": int(game_count),
            }
            for venue, row_count, episode_count, game_count in (
                connection.execute(
                    """
                    SELECT
                        venue,
                        count(*),
                        count(DISTINCT (game_id, episode_id)),
                        count(DISTINCT game_id)
                    FROM observed_candidates
                    GROUP BY venue
                    ORDER BY venue
                    """
                ).fetchall()
            )
        ]
        by_game = [
            {
                "game_id": str(game_id),
                "unique_episode_count": int(episode_count),
            }
            for game_id, episode_count in connection.execute(
                """
                SELECT game_id, count(DISTINCT episode_id)
                FROM observed_candidates
                GROUP BY game_id
                ORDER BY game_id
                """
            ).fetchall()
        ]
    finally:
        connection.close()

    venue_by_id = {row["venue"]: row for row in by_venue}
    nominal_count_threshold_met = (
        unique_episode_count >= 300
        and observed_game_count >= 16
        and all(
            venue_by_id.get(venue, {}).get("unique_episode_count", 0)
            >= 100
            for venue in ("kalshi", "polymarket")
        )
    )
    maximum_game_episode_count = max(
        (
            int(row["unique_episode_count"])
            for row in by_game
        ),
        default=0,
    )
    maximum_game_contribution = (
        None
        if unique_episode_count == 0
        else format(
            (
                Decimal(maximum_game_episode_count)
                / Decimal(unique_episode_count)
            ).quantize(Decimal("0.000001")),
            "f",
        )
    )
    registered_whitelist = []
    for hypothesis in X13_REGISTERED_ANALYSIS_LOCK_V1.hypotheses:
        if hypothesis.hypothesis_id == "event_superclass":
            status = "NOT_RUN_PRIMARY_PROJECTION_NOT_FROZEN"
        elif hypothesis.hypothesis_id == "sequence":
            status = "NOT_RUN_OPERATIONAL_SEQUENCE_DEFINITION_NOT_FROZEN"
        elif hypothesis.hypothesis_id == "fair_delta_by_pre_event_liquidity":
            status = (
                "NOT_RUN_FAIR_DELTA_AND_PRE_EVENT_LIQUIDITY_NOT_SUPPLIED"
            )
        else:
            status = "NOT_RUN_PIT_UNPROVEN_FAIR_DELTA_NOT_SUPPLIED"
        registered_whitelist.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_family": hypothesis.hypothesis_family,
                "analysis_kind": hypothesis.analysis_kind,
                "approved_groups": list(hypothesis.approved_groups),
                "estimand": hypothesis.estimand,
                "status": status,
            }
        )
    return {
        "schema": "nfl_x13_exploration_readiness_report_v1",
        "experiment_id": X13_EXPERIMENT_ID,
        "status": X13_STATUS,
        "analysis_status": "NOT_RUN_UNRESOLVED_ESTIMAND",
        "source_binding": {
            "registered_analysis_spec_lock_id": (
                X13_REGISTERED_ANALYSIS_LOCK_V1.lock_id
            ),
            "input_root_analysis_lock_sha256": (
                input_root_analysis_lock_sha256
            ),
            "association_partition_sha256s": sorted(
                partition_sha256s
            ),
        },
        "candidate_evidence": {
            "candidate_row_count": candidate_row_count,
            "observed_row_count": observed_row_count,
            "unique_episode_count": unique_episode_count,
            "episode_venue_unit_count": episode_venue_unit_count,
            "repeated_episode_venue_unit_count": int(
                unit_distribution[1]
            ),
            "episode_venue_rows_p50": int(unit_distribution[2]),
            "episode_venue_rows_p95": int(unit_distribution[3]),
            "episode_venue_rows_max": int(unit_distribution[4]),
            "game_count": observed_game_count,
            "by_venue": by_venue,
            "by_game": by_game,
            "maximum_single_game_contribution": (
                maximum_game_contribution
            ),
        },
        "candidate_support_envelope": {
            "thresholds": {
                "eligible_episode_count": 300,
                "game_count": 16,
                "per_venue_episode_count": 100,
            },
            "nominal_support_threshold_met": (
                nominal_count_threshold_met
            ),
            "gate_evaluated": False,
            "category_concentration_not_evaluated": True,
            "status": (
                "NOT_RUN_PRIMARY_PROJECTION_NOT_FROZEN"
            ),
            "reason": (
                "candidate rows repeat episode-venue units across contract, "
                "outcome, delay, and horizon; no frozen one-row projection "
                "or estimand exists"
            ),
        },
        "registered_whitelist": registered_whitelist,
        "inference_controls": {
            "game_cluster_bootstrap": (
                "NOT_RUN_BOOTSTRAP_REPLICATE_COUNT_NOT_FROZEN"
            ),
            "leave_one_game_out": (
                "NOT_RUN_PRIMARY_PROJECTION_NOT_FROZEN"
            ),
            "multiple_testing_correction": (
                "NOT_RUN_NO_EFFECT_TESTS_EXECUTED"
            ),
        },
        "required_before_inference": [
            "freeze one delay and horizon projection",
            "freeze logical market outcome and orientation projection",
            "freeze one episode-venue estimand",
            "freeze sequence operational definitions",
            "freeze bootstrap replicate count",
        ],
        "claim_boundary": {
            "association_candidates_only": True,
            "causality": False,
            "real_latency": False,
            "execution": False,
            "tradeable_alpha": False,
        },
    }


def _prepare_game(
    *,
    game_state: Any,
    universe_game: UniverseGameV1,
    capture_result: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    raw_manifest_hashes: tuple[str, ...],
) -> _PreparedGameV1:
    contracts = normalize_x13_contracts(universe_game)
    observations = _normalize_observations(
        universe_game, capture_result, documents
    )
    ledger = X13GameLedgerV1(
        events=game_state.events,
        episodes=game_state.episodes,
        stat_ledger=game_state.stat_ledger,
        events_sha256=game_state.events_sha256,
        artifact_sha256=game_state.ledger_sha256,
    )
    full_layer_m_audit = audit_layer_m(
        contracts,
        observations,
        game_id=universe_game.game_id,
    )
    if full_layer_m_audit.status != "PASS":
        raise X13PipelineError(
            f"{universe_game.game_id} complete Layer M audit failed"
        )
    primary_path_observations = tuple(
        observation
        for observation in observations
        if observation.primary_path_eligible
    )
    try:
        stream = prepare_x13_association_stream(
            ledger,
            contracts,
            primary_path_observations,
        )
    except AssociationEvidenceError as error:
        raise X13PipelineError(
            f"{universe_game.game_id} Layer G/M audit failed"
        ) from error
    return _PreparedGameV1(
        game_state=game_state,
        universe_game=universe_game,
        contracts=contracts,
        observations=observations,
        ledger=ledger,
        association_stream=stream,
        raw_manifest_hashes=raw_manifest_hashes,
    )


def execute_x13_pipeline(
    *,
    game_state_build: X13GameStateBuild,
    game_state_publication: str | Path,
    universe: X13UniverseBatchV1,
    universe_artifact: str | Path,
    capture: BatchCaptureResult,
    program_root: str | Path,
    raw_store_root: str | Path,
    output_root: str | Path,
) -> X13PipelineResultV1:
    """Verify, normalize, associate, and atomically publish all twenty games."""

    started = time.perf_counter_ns()
    program = Path(program_root).resolve()
    raw_root = Path(raw_store_root).resolve()
    _verify_game_state_evidence(game_state_build, game_state_publication)
    game_source_documents = _verify_game_source_manifests(
        game_state_build,
        program_root=program,
        raw_store_root=raw_root,
    )
    verify_x13_universe_artifact(universe, universe_artifact)
    capture_evidence = verify_x13_capture_evidence(
        capture,
        universe=universe,
        program_root=program,
        raw_store_root=raw_root,
    )
    external_universe_documents = _verify_universe_manifest_hashes(
        universe,
        capture_evidence,
        program_root=program,
        raw_store_root=raw_root,
    )
    active_authorization = load_x13_pipeline_authorization(
        program,
        game_state_build_id=str(game_state_build.manifest["build_id"]),
        universe_id=universe.universe_id,
        capture_sha256=capture.pointer.capture_sha256,
    )
    if not isinstance(active_authorization, X13PipelineAuthorizationV1):
        raise X13PipelineError("pipeline authorization has an unknown type")
    expected_lock = expected_analysis_lock_sha256(
        registration_head_sha256=(
            active_authorization.registration_head_sha256
        ),
        game_state_build_id=str(game_state_build.manifest["build_id"]),
        universe_id=universe.universe_id,
        capture_sha256=capture.pointer.capture_sha256,
    )
    if active_authorization.analysis_lock_sha256 != expected_lock:
        raise X13PipelineError(
            "analysis lock does not bind the verified input roots"
        )
    operational_use = dict(active_authorization.license_operational_use)
    all_documents = [
        document
        for documents in capture_evidence.documents_by_game.values()
        for document in documents
    ] + list(capture_evidence.batch_documents) + list(
        external_universe_documents
    ) + list(game_source_documents)
    if any(
        operational_use.get(document.license_ref)
        not in _RESEARCH_OPERATIONAL_USE
        for document in all_documents
    ):
        raise X13PipelineError("verified raw object has an unknown license")

    universe_by_game = {game.game_id: game for game in universe.games}
    capture_by_game = {game.game_id: game for game in capture.games}
    source_hashes = (
        game_state_build.source.pbp_manifest_sha256,
        game_state_build.source.participation_manifest_sha256,
    )
    def prepare_game(game_id: str) -> _PreparedGameV1:
        game_documents = capture_evidence.documents_by_game[game_id]
        lineage_hashes = tuple(
            sorted(
                {
                    *source_hashes,
                    *(document.manifest_sha256 for document in game_documents),
                    *universe_by_game[game_id].metadata_manifest_sha256s,
                    universe.series_registry_proof.manifest_sha256,
                }
            )
        )
        return _prepare_game(
            game_state=game_state_build.games[game_id],
            universe_game=universe_by_game[game_id],
            capture_result=capture_by_game[game_id],
            documents=game_documents,
            raw_manifest_hashes=lineage_hashes,
        )

    prepared_by_game: dict[str, _PreparedGameV1] = {}
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="x13-prepare",
    ) as executor:
        futures = {
            executor.submit(prepare_game, game_id): game_id
            for game_id in X13_GAME_IDS
        }
        for future in as_completed(futures):
            game_id = futures[future]
            prepared_by_game[game_id] = future.result()
    prepared_games = [
        prepared_by_game[game_id] for game_id in X13_GAME_IDS
    ]
    expected_association_count = sum(
        prepared.association_stream.expected_association_count
        for prepared in prepared_games
    )
    if expected_association_count != _FROZEN_ASSOCIATION_COUNT:
        raise X13PipelineError(
            "frozen universe association cardinality changed: expected "
            f"{_FROZEN_ASSOCIATION_COUNT}, observed "
            f"{expected_association_count}"
        )

    output = Path(output_root).resolve()
    if output == Path(output.anchor):
        raise X13PipelineError("output_root cannot be a filesystem root")
    output.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".x13-pipeline-", suffix=".tmp", dir=output)
    )
    streamed_by_game: dict[str, _StreamedGameV1] = {}
    auxiliary: list[AuxiliaryArtifactV1] = []
    payloads: list[dict[str, object]] = []
    try:
        disk_gate = _AssociationWriteDiskGate(staging_root)
        storage_pilot = _run_association_storage_pilot(
            prepared_games[0],
            staging_root=staging_root,
            disk_gate=disk_gate,
        )
        storage_preflight = preflight_x13_association_storage(
            available_free_bytes=shutil.disk_usage(staging_root).free,
            already_written_bytes=0,
            remaining_association_rows=expected_association_count,
            empirical_bytes_per_row=(
                storage_pilot.empirical_bytes_per_row
            ),
        )
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="x13-association",
        ) as executor:
            futures = {
                executor.submit(
                    _stream_game_associations,
                    prepared,
                    staging_root=staging_root,
                    disk_gate=disk_gate,
                ): prepared.universe_game.game_id
                for prepared in prepared_games
            }
            for future in as_completed(futures):
                game_id = futures[future]
                streamed_by_game[game_id] = future.result()
        for game_id in X13_GAME_IDS:
            streamed = streamed_by_game[game_id]
            auxiliary.extend(streamed.auxiliary_artifacts)
        actual_association_count = sum(
            streamed.actual_row_count
            for streamed in streamed_by_game.values()
        )
        if actual_association_count != expected_association_count:
            raise X13PipelineError(
                "batch association cardinality differs from prepared streams"
            )
        for prepared in prepared_games:
            payloads.append(
                _game_payload(
                    prepared=prepared,
                    streamed=streamed_by_game[
                        prepared.universe_game.game_id
                    ],
                    authorization=active_authorization,
                )
            )

        partition_descriptors = [
            artifact.descriptor
            for artifact in auxiliary
            if artifact.role == "association_candidate_table"
        ]
        cardinality_payload = {
            "schema": "nfl_x13_association_cardinality_report_v1",
            "experiment_id": X13_EXPERIMENT_ID,
            "status": X13_STATUS,
            "frozen_universe_id": _FROZEN_UNIVERSE_ID,
            "frozen_expected_rows": _FROZEN_ASSOCIATION_COUNT,
            "active_universe_id": universe.universe_id,
            "expected_rows": expected_association_count,
            "actual_rows": actual_association_count,
            "schema_fingerprint": _ASSOCIATION_SCHEMA_FINGERPRINT,
            "partition_count": len(partition_descriptors),
            "partitions": partition_descriptors,
            "storage_pilot": _json_value(storage_pilot),
            "storage_preflight": _json_value(storage_preflight),
            "running_disk_gate_flush_count": disk_gate.flush_count,
            "compute_worker_count": 2,
        }
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path=(
                    "reports/association-cardinality-report-v1.json"
                ),
                role="association_cardinality_report",
                payload=cardinality_payload,
                row_count=len(partition_descriptors),
            )
        )
        exploration_readiness_payload = _exploration_readiness_report(
            association_paths=tuple(
                artifact.source_path
                for artifact in auxiliary
                if artifact.role == "association_candidate_table"
            ),
            expected_candidate_row_count=actual_association_count,
            input_root_analysis_lock_sha256=(
                active_authorization.analysis_lock_sha256
            ),
        )
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path=(
                    "reports/exploration-readiness-report-v1.json"
                ),
                role="exploration_readiness_report",
                payload=exploration_readiness_payload,
                row_count=int(
                    exploration_readiness_payload[
                        "candidate_evidence"
                    ]["unique_episode_count"]
                ),
            )
        )
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path=(
                    "inventory/robinhood-provenance-inventory-v1.json"
                ),
                role="robinhood_provenance_inventory",
                payload=_robinhood_provenance_inventory(),
                row_count=1,
            )
        )
        cross_game_inventory_payload = _cross_game_mve_inventory(universe)
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path="inventory/cross-game-mve-inventory-v1.json",
                role="cross_game_mve_inventory",
                payload=cross_game_inventory_payload,
                row_count=int(cross_game_inventory_payload["record_count"]),
            )
        )
        semantic_overlap_payload = _contract_semantic_overlap_audit(
            prepared_games
        )
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path=(
                    "reports/contract-semantic-overlap-audit-v1.json"
                ),
                role="contract_semantic_overlap_audit",
                payload=semantic_overlap_payload,
                row_count=(
                    int(semantic_overlap_payload["winner_orientation_count"])
                    + int(
                        semantic_overlap_payload[
                            "matched_total_orientation_count"
                        ]
                    )
                ),
            )
        )
        validation_payload = _validation_report(
            prepared_games=prepared_games,
            capture_evidence=capture_evidence,
            semantic_overlap=semantic_overlap_payload,
            expected_association_count=expected_association_count,
            actual_association_count=actual_association_count,
        )
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path="reports/validation-report-v1.json",
                role="validation_report",
                payload=validation_payload,
                row_count=20,
            )
        )
        lineage_payload = _lineage_report(
            prepared_games=prepared_games,
            authorization=active_authorization,
            game_state_build=game_state_build,
            universe=universe,
            capture=capture,
        )
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path="reports/lineage-report-v1.json",
                role="lineage_report",
                payload=lineage_payload,
                row_count=20,
            )
        )
        runtime_before_publish = {
            "schema": "nfl_x13_association_runtime_rss_report_v1",
            "experiment_id": X13_EXPERIMENT_ID,
            "status": X13_STATUS,
            "measurement_boundary": "through_association_streaming",
            "compute_worker_count": 2,
            "storage_pilot_elapsed_ns": storage_pilot.elapsed_ns,
            "running_disk_gate_flush_count": disk_gate.flush_count,
            "elapsed_ns": max(1, time.perf_counter_ns() - started),
            "peak_rss_bytes": max(1, _peak_rss_bytes()),
            "game_count": len(payloads),
            "association_count": actual_association_count,
            "association_presentation_count": sum(
                len(payload["associations"]) for payload in payloads
            ),
            "per_game_elapsed_ns": {
                game_id: streamed_by_game[game_id].elapsed_ns
                for game_id in X13_GAME_IDS
            },
        }
        auxiliary.append(
            _write_auxiliary_json(
                staging_root,
                relative_path="reports/runtime-rss-v1.json",
                role="association_runtime_rss_report",
                payload=runtime_before_publish,
                row_count=20,
            )
        )
        batch_path = publish_x13_batch(
            payloads,
            output_root=output_root,
            plan_id=X13_BATCH_SPEC.plan_id,
            builder_version=_BUILDER_VERSION,
            auxiliary_artifacts=tuple(auxiliary),
        )
        verify_published_batch(batch_path)
        manifest_payload = _strict_json(
            (batch_path / "batch_manifest.json").read_bytes(),
            context="published batch manifest",
        )
        if not isinstance(manifest_payload, Mapping):
            raise X13PipelineError(
                "published batch manifest is not an object"
            )
        batch_manifest = dict(manifest_payload)
    except (X13BatchPublishError, AssociationEvidenceError, OSError) as error:
        raise X13PipelineError("atomic batch publication failed") from error
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    runtime = {
        "elapsed_ns": max(1, time.perf_counter_ns() - started),
        "peak_rss_bytes": max(1, _peak_rss_bytes()),
        "compute_worker_count": 2,
        "storage_pilot_row_count": storage_pilot.row_count,
        "running_disk_gate_flush_count": disk_gate.flush_count,
        "game_count": len(payloads),
        "contract_count": sum(
            len(payload["contracts"]) for payload in payloads
        ),
        "observation_count": sum(
            len(payload["observations"]) for payload in payloads
        ),
        "association_count": expected_association_count,
        "association_presentation_count": sum(
            len(payload["associations"]) for payload in payloads
        ),
    }
    return X13PipelineResultV1(
        batch_path=batch_path,
        batch_manifest=MappingProxyType(dict(batch_manifest)),
        game_payloads=tuple(MappingProxyType(payload) for payload in payloads),
        runtime=MappingProxyType(runtime),
    )


def run_x13_full_batch(
    *,
    game_state_build: X13GameStateBuild,
    game_state_publication: str | Path,
    universe_artifact: str | Path,
    capture: BatchCaptureResult,
    program_root: str | Path,
    raw_store_root: str | Path,
    output_root: str | Path,
) -> X13PipelineResultV1:
    """Run the formal batch from a complete live or reloaded capture result."""

    if not isinstance(capture, BatchCaptureResult):
        raise X13PipelineError(
            "run_x13_full_batch requires the complete typed capture result; "
            "a pointer alone cannot prove terminal pagination"
        )
    universe = load_x13_universe_artifact(universe_artifact)
    return execute_x13_pipeline(
        game_state_build=game_state_build,
        game_state_publication=game_state_publication,
        universe=universe,
        universe_artifact=universe_artifact,
        capture=capture,
        program_root=program_root,
        raw_store_root=raw_store_root,
        output_root=output_root,
    )


__all__ = [
    "AssociationStoragePreflightV1",
    "X13_EVALUATION_CODE_PATHS",
    "VerifiedCaptureDocumentV1",
    "VerifiedX13CaptureEvidenceV1",
    "X13PipelineAuthorizationV1",
    "X13PipelineError",
    "X13PipelineResultV1",
    "execute_x13_pipeline",
    "expected_analysis_lock_sha256",
    "expected_x13_evaluation_code_bundle_sha256",
    "expected_x13_source_manifest_amendment_changes",
    "expected_source_manifest_bundle_sha256",
    "load_x13_universe_artifact",
    "load_x13_pipeline_authorization",
    "normalize_x13_contracts",
    "preflight_x13_association_storage",
    "run_x13_full_batch",
    "verify_x13_capture_evidence",
    "verify_x13_universe_artifact",
]
