"""Verified StatsBomb adapter and real-data Dixon-Coles POC for X-12."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from prediction_market.models.validation import (
    ValidationInputError,
    evaluate_multiclass_probabilities,
)
from prediction_market.sports.soccer_transition_model import (
    DynamicIntensityModel,
    SoccerTransitionFeatures,
    SoccerTransitionModelError,
    TemperatureCalibration,
    apply_multiclass_temperature,
    fit_dynamic_intensity,
    fit_multiclass_temperature,
    predict_transition_distribution,
)
from prediction_market.sports.statsbomb import (
    STATSBOMB_COMMIT,
    STATSBOMB_EXPECTED_MATCHES,
    StatsBombSourceError,
    inspect_statsbomb_event,
    inspect_statsbomb_match_index,
)
from prediction_market.static_store import read_verified_static_object


X12_DATASET_ID = "DS-STATSBOMB-OPEN"
X12_SOURCE = "statsbomb"
X12_STATSBOMB_VERSION = STATSBOMB_COMMIT
X12_SEED = 20260722
X12_RESULT_LABEL = "PRELIMINARY"
X12_EXPERIMENT_ID = "X-12"
X12_AUTHORIZATION_SCOPE = (
    "team_h_soccer_dynamic_transition_reproduction_v1"
)
X12_MODEL_ID = "MODEL-SOCCER-DYNAMIC-INTENSITY"
X12_MODEL_VERSION = "v1"
X12_DYNAMIC_EVIDENCE_RELATIVE_PATH = Path(
    "artifacts/game-state/soccer/x12_dynamic_transition_poc_v1.json"
)
X12_HISTORICAL_1X2_RELATIVE_PATH = Path(
    "artifacts/game-state/soccer/x12_real_data_poc_v0.json"
)
X12_HISTORICAL_1X2_FILE_SHA256 = (
    "sha256:34b80e999885f3c44c0a9366b7595ed24080623e42fc2e4e4ed2c264fceb61eb"
)

OUTCOME_CLASSES = ("home_win", "draw", "away_win")
TRANSITION_CLASSES = ("home_goal", "away_goal", "no_goal")
TRANSITION_HORIZON_SECONDS = 300
TRANSITION_SNAPSHOTS = tuple(
    (period, period_minute)
    for period in (1, 2)
    for period_minute in range(0, 45, 5)
)
KICKOFF_TIME_BASIS = "source_naive_attached_UTC_for_deterministic_order_only"
OFFLINE_PIT_STATUS = "offline_reconstruction_not_live_PIT"

_TIMESTAMP_RE = re.compile(
    r"^(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):"
    r"(?P<second>[0-9]{2})\.(?P<fraction>[0-9]{3,6})$"
)
_HASH_PREFIX = "sha256:"
_INVALID_LIKELIHOOD_OBJECTIVE = 1e100
_BOUND_ROUNDOFF_ULPS = 8
_KKT_BOUNDARY_ABS_TOLERANCE = 1e-10
_DYNAMIC_INTENSITY_L2_PENALTY = 1.0
_MINIMUM_TRANSITION_CALIBRATION_MATCHES = 20


class X12DataError(ValueError):
    """Verified inputs cannot support the frozen X-12 POC."""


@dataclass(frozen=True, slots=True)
class X12ManifestInventory:
    partition: str
    manifest_path: str
    manifest_sha256: str
    object_sha256: str
    schema_fingerprint: str
    byte_length: int
    event_count: int | None


@dataclass(frozen=True, slots=True)
class X12InputInventory:
    dataset_id: str
    source: str
    version: str
    competition_id: int
    season_id: int
    match_count: int
    event_partition_count: int
    total_events: int
    team_count: int
    first_played_at: str
    last_played_at: str
    license_ref: str
    license_status: str
    kickoff_time_basis: str
    manifests: tuple[X12ManifestInventory, ...]
    manifest_paths: tuple[str, ...]
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class X12LoadedDataset:
    inventory: X12InputInventory
    matches: pd.DataFrame
    goals: pd.DataFrame
    dismissals: pd.DataFrame
    chronology_sha256: str
    goal_timeline_sha256: str
    dismissal_timeline_sha256: str
    source_time_regressions: int


@dataclass(frozen=True, slots=True)
class DixonColesModel:
    team_ids: tuple[int, ...]
    reference_team_id: int
    parameters: tuple[float, ...]
    home_advantage: float
    rho: float
    initial_objective: float
    initial_projected_gradient_inf_norm: float
    objective: float
    objective_improvement: float
    parameter_displacement: float
    projected_gradient_inf_norm: float
    iterations: int
    optimizer_status: str


@dataclass(frozen=True, slots=True)
class X12TransitionSplitAudit:
    method: str
    base_fit_first_date: str
    base_fit_last_date: str
    calibration_first_date: str
    calibration_last_date: str
    final_test_first_date: str
    final_test_last_date: str
    final_test_evaluated_first_date: str
    final_test_evaluated_last_date: str
    base_fit_date_count: int
    calibration_date_count: int
    final_test_date_count: int
    final_test_evaluated_date_count: int
    base_fit_match_count: int
    calibration_match_count: int
    final_test_match_count: int
    final_test_evaluated_match_count: int
    dynamic_fit_evaluation_cutoff: pd.Timestamp
    temperature_fit_evaluation_cutoff: pd.Timestamp
    base_fit_max_outcome_available_at: pd.Timestamp
    calibration_max_label_available_at: pd.Timestamp
    dixon_coles_parameter_sha256: str
    dynamic_parameter_sha256: str
    raw_transition_parameter_sha256: str
    temperature_parameter_sha256: str
    calibrated_transition_parameter_sha256: str


@dataclass(frozen=True, slots=True)
class X12DynamicTransitionEvaluation:
    experiment_id: str
    model_id: str
    model_version: str
    authorization_scope: str
    result_label: str
    is_formal_result: bool
    seed: int
    transition_predictions: pd.DataFrame
    transition_split: X12TransitionSplitAudit
    temperature_calibration: TemperatureCalibration
    transition_metrics: dict[str, object]
    evaluation_match_limit: int | None
    bootstrap_samples: int
    minimum_valid_bootstrap_samples: int
    confidence_level: float
    optimizer_max_iterations: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return _HASH_PREFIX + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_HASH_PREFIX)
        or len(value) != len(_HASH_PREFIX) + 64
    ):
        raise X12DataError(f"{field} must be a SHA-256 digest")
    try:
        int(value.removeprefix(_HASH_PREFIX), 16)
    except ValueError as error:
        raise X12DataError(f"{field} must be a SHA-256 digest") from error
    return value


def _json_ready(value: object) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise X12DataError("evidence cannot contain non-finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise X12DataError(
        f"evidence contains unsupported value type: {type(value).__name__}"
    )


def _strict_json_array(payload: bytes, *, context: str) -> list[dict[str, Any]]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise X12DataError(f"{context} contains duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                X12DataError(f"{context} contains non-finite value {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise X12DataError(f"{context} is not strict UTF-8 JSON") from error
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise X12DataError(f"{context} must be an array of objects")
    return value


def _inventory_material(inventory: X12InputInventory) -> dict[str, object]:
    return {
        "dataset_id": inventory.dataset_id,
        "source": inventory.source,
        "version": inventory.version,
        "competition_id": inventory.competition_id,
        "season_id": inventory.season_id,
        "match_count": inventory.match_count,
        "event_partition_count": inventory.event_partition_count,
        "total_events": inventory.total_events,
        "team_count": inventory.team_count,
        "first_played_at": inventory.first_played_at,
        "last_played_at": inventory.last_played_at,
        "license_ref": inventory.license_ref,
        "license_status": inventory.license_status,
        "kickoff_time_basis": inventory.kickoff_time_basis,
        "manifests": [asdict(item) for item in inventory.manifests],
        "manifest_paths": list(inventory.manifest_paths),
    }


def inventory_sha256(inventory: X12InputInventory) -> str:
    """Recompute the frozen inventory self-hash."""

    if not isinstance(inventory, X12InputInventory):
        raise TypeError("inventory must be an X12InputInventory")
    return _sha256(_inventory_material(inventory))


def _frame_timestamp(value: object) -> str:
    if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
        raise X12DataError("chronology contains a non-UTC timestamp")
    if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise X12DataError("chronology timestamps must be UTC")
    return value.isoformat().replace("+00:00", "Z")


def chronology_sha256(matches: pd.DataFrame) -> str:
    """Hash the canonical ordered match chronology and its event lineage."""

    required = (
        "match_id",
        "played_at",
        "outcome_available_at",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "outcome",
        "event_manifest_sha256",
        "event_object_sha256",
        "event_schema_fingerprint",
        "goal_timeline_sha256",
        "dismissal_timeline_sha256",
        "period_2_global_offset_ms",
    )
    if not isinstance(matches, pd.DataFrame) or matches.empty:
        raise X12DataError("chronology requires nonempty canonical matches")
    missing = sorted(set(required) - set(matches.columns))
    if missing:
        raise X12DataError(f"chronology columns are missing: {missing}")
    documents: list[dict[str, object]] = []
    for row in matches.sort_values(
        ["played_at", "match_id"], kind="mergesort"
    ).itertuples(index=False):
        documents.append(
            {
                "match_id": int(row.match_id),
                "played_at": _frame_timestamp(row.played_at),
                "outcome_available_at": _frame_timestamp(
                    row.outcome_available_at
                ),
                "home_team_id": int(row.home_team_id),
                "away_team_id": int(row.away_team_id),
                "home_score": int(row.home_score),
                "away_score": int(row.away_score),
                "outcome": str(row.outcome),
                "event_manifest_sha256": _validate_digest(
                    row.event_manifest_sha256,
                    "event_manifest_sha256",
                ),
                "event_object_sha256": _validate_digest(
                    row.event_object_sha256,
                    "event_object_sha256",
                ),
                "event_schema_fingerprint": _validate_digest(
                    row.event_schema_fingerprint,
                    "event_schema_fingerprint",
                ),
                "goal_timeline_sha256": _validate_digest(
                    row.goal_timeline_sha256,
                    "goal_timeline_sha256",
                ),
                "dismissal_timeline_sha256": _validate_digest(
                    row.dismissal_timeline_sha256,
                    "dismissal_timeline_sha256",
                ),
                "period_2_global_offset_ms": int(
                    row.period_2_global_offset_ms
                ),
            }
        )
    return _sha256(documents)


def _canonical_goal_documents(goals: pd.DataFrame) -> list[dict[str, object]]:
    required = (
        "match_id",
        "period",
        "period_clock_ms",
        "global_elapsed_ms",
        "source_clock_ms",
        "event_index",
        "native_event_id",
        "scoring_side",
    )
    if not isinstance(goals, pd.DataFrame):
        raise X12DataError("goal timeline must be a dataframe")
    missing = sorted(set(required) - set(goals.columns))
    if missing:
        raise X12DataError(f"goal timeline columns are missing: {missing}")
    documents: list[dict[str, object]] = []
    ordered = goals.sort_values(
        ["match_id", "global_elapsed_ms", "event_index"],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        numeric_values = (
            row.match_id,
            row.period,
            row.period_clock_ms,
            row.global_elapsed_ms,
            row.source_clock_ms,
            row.event_index,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in numeric_values
        ):
            raise X12DataError("goal timeline identities and times must be integers")
        match_id = int(row.match_id)
        period = int(row.period)
        period_clock_ms = int(row.period_clock_ms)
        global_elapsed_ms = int(row.global_elapsed_ms)
        source_clock_ms = int(row.source_clock_ms)
        event_index = int(row.event_index)
        native_event_id = str(row.native_event_id)
        scoring_side = str(row.scoring_side)
        if match_id <= 0:
            raise X12DataError("goal timeline match_id must be positive")
        if not 1 <= period <= 5:
            raise X12DataError("goal timeline period is invalid")
        if (
            period_clock_ms < 0
            or global_elapsed_ms < 0
            or source_clock_ms < 0
        ):
            raise X12DataError("goal timeline clocks must be nonnegative")
        if event_index <= 0:
            raise X12DataError("goal timeline event_index must be positive")
        if not native_event_id:
            raise X12DataError("goal timeline native_event_id must be nonempty")
        if scoring_side not in {"home_goal", "away_goal"}:
            raise X12DataError("goal timeline scoring_side is invalid")
        documents.append(
            {
                "match_id": match_id,
                "period": period,
                "period_clock_ms": period_clock_ms,
                "global_elapsed_ms": global_elapsed_ms,
                "source_clock_ms": source_clock_ms,
                "event_index": event_index,
                "native_event_id": native_event_id,
                "scoring_side": scoring_side,
            }
        )
    identities = [
        (int(item["match_id"]), int(item["event_index"])) for item in documents
    ]
    if len(set(identities)) != len(identities):
        raise X12DataError("goal timeline contains duplicate event identity")
    return documents


def goal_timeline_sha256(goals: pd.DataFrame) -> str:
    """Hash the actual canonical goal rows consumed by transition labels."""

    return _sha256(_canonical_goal_documents(goals))


def _validate_goal_timeline_binding(
    matches: pd.DataFrame,
    goals: pd.DataFrame,
    *,
    expected_sha256: str,
) -> None:
    documents = _canonical_goal_documents(goals)
    actual_sha256 = _sha256(documents)
    if actual_sha256 != _validate_digest(
        expected_sha256,
        "goal_timeline_sha256",
    ):
        raise X12DataError("frozen goal timeline SHA-256 is invalid")
    known_match_ids = set(matches["match_id"].astype(int))
    observed_match_ids = {int(item["match_id"]) for item in documents}
    if not observed_match_ids <= known_match_ids:
        raise X12DataError("goal timeline contains an unknown match")
    by_match: dict[int, list[dict[str, object]]] = {
        match_id: [] for match_id in known_match_ids
    }
    for item in documents:
        by_match[int(item["match_id"])].append(
            {
                "period": int(item["period"]),
                "period_clock_ms": int(item["period_clock_ms"]),
                "global_elapsed_ms": int(item["global_elapsed_ms"]),
                "source_clock_ms": int(item["source_clock_ms"]),
                "event_index": int(item["event_index"]),
                "native_event_id": str(item["native_event_id"]),
                "scoring_side": str(item["scoring_side"]),
            }
        )
    for row in matches.itertuples(index=False):
        expected = _validate_digest(
            row.goal_timeline_sha256,
            "match.goal_timeline_sha256",
        )
        if _sha256(by_match[int(row.match_id)]) != expected:
            raise X12DataError(
                f"goal timeline does not match frozen match {int(row.match_id)}"
            )


def _canonical_dismissal_documents(
    dismissals: pd.DataFrame,
) -> list[dict[str, object]]:
    required = (
        "match_id",
        "period",
        "period_clock_ms",
        "global_elapsed_ms",
        "source_clock_ms",
        "event_index",
        "native_event_id",
        "player_id",
        "dismissal_side",
        "card",
    )
    if not isinstance(dismissals, pd.DataFrame):
        raise X12DataError("dismissal timeline must be a dataframe")
    missing = sorted(set(required) - set(dismissals.columns))
    if missing:
        raise X12DataError(
            f"dismissal timeline columns are missing: {missing}"
        )
    documents: list[dict[str, object]] = []
    ordered = dismissals.sort_values(
        ["match_id", "global_elapsed_ms", "event_index", "player_id"],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        numeric_values = (
            row.match_id,
            row.period,
            row.period_clock_ms,
            row.global_elapsed_ms,
            row.source_clock_ms,
            row.event_index,
            row.player_id,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in numeric_values
        ):
            raise X12DataError(
                "dismissal timeline identities and times must be integers"
            )
        match_id = int(row.match_id)
        period = int(row.period)
        period_clock_ms = int(row.period_clock_ms)
        global_elapsed_ms = int(row.global_elapsed_ms)
        source_clock_ms = int(row.source_clock_ms)
        event_index = int(row.event_index)
        native_event_id = str(row.native_event_id)
        player_id = int(row.player_id)
        dismissal_side = str(row.dismissal_side)
        card = str(row.card)
        if match_id <= 0 or event_index <= 0 or player_id <= 0:
            raise X12DataError(
                "dismissal timeline identities must be positive"
            )
        if not 1 <= period <= 5:
            raise X12DataError("dismissal timeline period is invalid")
        if (
            period_clock_ms < 0
            or global_elapsed_ms < 0
            or source_clock_ms < 0
        ):
            raise X12DataError("dismissal timeline clocks must be nonnegative")
        if not native_event_id:
            raise X12DataError(
                "dismissal timeline native_event_id must be nonempty"
            )
        if dismissal_side not in {
            "home_dismissal",
            "away_dismissal",
        }:
            raise X12DataError("dismissal timeline side is invalid")
        if card not in {"Red Card", "Second Yellow"}:
            raise X12DataError("dismissal timeline card is invalid")
        documents.append(
            {
                "match_id": match_id,
                "period": period,
                "period_clock_ms": period_clock_ms,
                "global_elapsed_ms": global_elapsed_ms,
                "source_clock_ms": source_clock_ms,
                "event_index": event_index,
                "native_event_id": native_event_id,
                "player_id": player_id,
                "dismissal_side": dismissal_side,
                "card": card,
            }
        )
    identities = [
        (int(item["match_id"]), int(item["player_id"]))
        for item in documents
    ]
    if len(set(identities)) != len(identities):
        raise X12DataError(
            "dismissal timeline contains duplicate match-player identity"
        )
    return documents


def dismissal_timeline_sha256(dismissals: pd.DataFrame) -> str:
    """Hash the canonical state-affecting dismissal timeline."""

    return _sha256(_canonical_dismissal_documents(dismissals))


def _validate_dismissal_timeline_binding(
    matches: pd.DataFrame,
    dismissals: pd.DataFrame,
    *,
    expected_sha256: str,
) -> None:
    documents = _canonical_dismissal_documents(dismissals)
    actual_sha256 = _sha256(documents)
    if actual_sha256 != _validate_digest(
        expected_sha256,
        "dismissal_timeline_sha256",
    ):
        raise X12DataError("frozen dismissal timeline SHA-256 is invalid")
    known_match_ids = set(matches["match_id"].astype(int))
    observed_match_ids = {int(item["match_id"]) for item in documents}
    if not observed_match_ids <= known_match_ids:
        raise X12DataError("dismissal timeline contains an unknown match")
    by_match: dict[int, list[dict[str, object]]] = {
        match_id: [] for match_id in known_match_ids
    }
    for item in documents:
        by_match[int(item["match_id"])].append(
            {
                "period": int(item["period"]),
                "period_clock_ms": int(item["period_clock_ms"]),
                "global_elapsed_ms": int(item["global_elapsed_ms"]),
                "source_clock_ms": int(item["source_clock_ms"]),
                "event_index": int(item["event_index"]),
                "native_event_id": str(item["native_event_id"]),
                "player_id": int(item["player_id"]),
                "dismissal_side": str(item["dismissal_side"]),
                "card": str(item["card"]),
            }
        )
    for row in matches.itertuples(index=False):
        expected = _validate_digest(
            row.dismissal_timeline_sha256,
            "match.dismissal_timeline_sha256",
        )
        if _sha256(by_match[int(row.match_id)]) != expected:
            raise X12DataError(
                "dismissal timeline does not match frozen match "
                f"{int(row.match_id)}"
            )


def evidence_sha256(evidence: dict[str, object]) -> str:
    """Recompute the evidence hash, excluding its self-hash field."""

    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")
    material = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    return _sha256(_json_ready(material))


def _discover_manifest_paths(store_root: Path) -> tuple[Path, ...]:
    base = (
        store_root
        / "manifests"
        / f"source={X12_SOURCE}"
        / f"dataset={X12_DATASET_ID}"
        / f"version={X12_STATSBOMB_VERSION}"
    )
    index_paths = tuple(sorted((base / "partition=matches-2-27").glob("*.manifest.json")))
    if len(index_paths) != 1:
        raise X12DataError(
            "X-12 requires exactly one matches-2-27 manifest; "
            f"found {len(index_paths)}"
        )
    event_paths: list[Path] = []
    event_directories = tuple(sorted(base.glob("partition=events-*")))
    if len(event_directories) != STATSBOMB_EXPECTED_MATCHES:
        raise X12DataError(
            "X-12 requires exactly 380 event partitions; "
            f"found {len(event_directories)}"
        )
    for directory in event_directories:
        paths = tuple(sorted(directory.glob("*.manifest.json")))
        if len(paths) != 1:
            raise X12DataError(
                f"X-12 requires one manifest in {directory.name}; found {len(paths)}"
            )
        event_paths.append(paths[0])
    return (index_paths[0], *event_paths)


def _manifest_reference(path: Path, store_root: Path) -> str:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(store_root))
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        return absolute.as_posix()


def _manifest_inventory(
    verified: Any,
    *,
    manifest_path: Path,
    store_root: Path,
    event_count: int | None,
) -> X12ManifestInventory:
    record = verified.record
    manifest = record.manifest
    if (
        record.source != X12_SOURCE
        or record.dataset != X12_DATASET_ID
        or record.version != X12_STATSBOMB_VERSION
        or manifest.dataset_id != X12_DATASET_ID
    ):
        raise X12DataError("verified object is outside the frozen StatsBomb source")
    if manifest.object_kind != "byte_exact_original":
        raise X12DataError("X-12 requires byte-exact original source objects")
    if manifest.license_ref != "O-004" or manifest.license_status != "research_only":
        raise X12DataError("X-12 requires the O-004 research-only license binding")
    if manifest.upstream_partition != record.partition:
        raise X12DataError("manifest partition binding is inconsistent")
    return X12ManifestInventory(
        partition=str(record.partition),
        manifest_path=_manifest_reference(manifest_path, store_root),
        manifest_sha256=_validate_digest(
            manifest.manifest_sha256, "manifest_sha256"
        ),
        object_sha256=_validate_digest(manifest.object_sha256, "object_sha256"),
        schema_fingerprint=_validate_digest(
            manifest.schema_fingerprint, "schema_fingerprint"
        ),
        byte_length=len(verified.object_bytes),
        event_count=event_count,
    )


def _played_at(match: dict[str, Any]) -> pd.Timestamp:
    match_date = match.get("match_date")
    kickoff = match.get("kick_off")
    if type(match_date) is not str or type(kickoff) is not str:
        raise X12DataError("match date and kickoff must be source strings")
    value = pd.to_datetime(
        f"{match_date}T{kickoff}",
        format="ISO8601",
        errors="coerce",
        utc=True,
    )
    if pd.isna(value):
        raise X12DataError("match kickoff cannot be deterministically parsed")
    return pd.Timestamp(value)


def _team(match: dict[str, Any], side: str) -> tuple[int, str]:
    value = match.get(f"{side}_team")
    if type(value) is not dict:
        raise X12DataError(f"{side}_team must be an object")
    identifier = value.get(f"{side}_team_id")
    name = value.get(f"{side}_team_name")
    if type(identifier) is not int or identifier <= 0:
        raise X12DataError(f"{side}_team_id must be a positive integer")
    if type(name) is not str or not name.strip():
        raise X12DataError(f"{side}_team_name must be nonempty")
    return identifier, name


def _score(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise X12DataError(f"{field} must be a nonnegative integer")
    return value


def _event_team_id(event: dict[str, Any]) -> int:
    team = event.get("team")
    if type(team) is not dict or type(team.get("id")) is not int:
        raise X12DataError("event team identity is invalid")
    return int(team["id"])


def _event_type(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if type(event_type) is not dict or type(event_type.get("name")) is not str:
        raise X12DataError("event type is invalid")
    return str(event_type["name"])


def _validate_native_event_time(
    event: dict[str, Any],
) -> tuple[int, int, int]:
    minute = event.get("minute")
    second = event.get("second")
    period = event.get("period")
    timestamp = event.get("timestamp")
    if type(minute) is not int or not 0 <= minute <= 130:
        raise X12DataError("event minute is outside the POC match horizon")
    if type(second) is not int or not 0 <= second <= 59:
        raise X12DataError("event second is outside [0, 59]")
    if type(period) is not int or not 1 <= period <= 5:
        raise X12DataError("event period is outside [1, 5]")
    if type(timestamp) is not str:
        raise X12DataError("event timestamp must be a source string")
    match = _TIMESTAMP_RE.fullmatch(timestamp)
    if (
        match is None
        or int(match.group("minute")) > 59
        or int(match.group("second")) > 59
    ):
        raise X12DataError("event timestamp is not a valid native clock")
    timestamp_second = int(match.group("second"))
    if timestamp_second != second:
        raise X12DataError("event timestamp second must match event second")
    millisecond = int(match.group("fraction")[:3].ljust(3, "0"))
    period_clock_ms = (
        (
            int(match.group("hour")) * 60 * 60
            + int(match.group("minute")) * 60
            + timestamp_second
        )
        * 1_000
        + millisecond
    )
    source_clock_ms = (minute * 60 + second) * 1_000 + millisecond
    return period, period_clock_ms, source_clock_ms


def _goal_side(
    event: dict[str, Any],
    *,
    home_team_id: int,
    away_team_id: int,
) -> str | None:
    event_type = _event_type(event)
    team_id = _event_team_id(event)
    if event_type == "Shot":
        shot = event.get("shot")
        if type(shot) is not dict:
            return None
        outcome = shot.get("outcome")
        if type(outcome) is not dict or outcome.get("name") != "Goal":
            return None
        return "home_goal" if team_id == home_team_id else "away_goal"
    if event_type == "Own Goal Against":
        return "away_goal" if team_id == home_team_id else "home_goal"
    return None


def _dismissal(
    event: dict[str, Any],
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[int, str, str] | None:
    event_type = _event_type(event)
    container_name = (
        "bad_behaviour"
        if event_type == "Bad Behaviour"
        else "foul_committed"
        if event_type == "Foul Committed"
        else None
    )
    if container_name is None:
        return None
    container = event.get(container_name)
    if type(container) is not dict:
        return None
    card_value = container.get("card")
    if type(card_value) is not dict:
        return None
    card = card_value.get("name")
    if card not in {"Red Card", "Second Yellow"}:
        return None
    player = event.get("player")
    if (
        type(player) is not dict
        or type(player.get("id")) is not int
        or int(player["id"]) <= 0
    ):
        raise X12DataError(
            "state-affecting dismissal requires a positive player id"
        )
    team_id = _event_team_id(event)
    dismissal_side = (
        "home_dismissal"
        if team_id == home_team_id
        else "away_dismissal"
        if team_id == away_team_id
        else None
    )
    if dismissal_side is None:
        raise X12DataError("dismissal team is outside the current match")
    return int(player["id"]), dismissal_side, str(card)


def _adapt_events(
    payload: bytes,
    *,
    match_id: int,
    home_team_id: int,
    away_team_id: int,
    expected_home_score: int,
    expected_away_score: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    int,
    dict[int, int],
]:
    try:
        inspect_statsbomb_event(payload, match_id=match_id)
    except StatsBombSourceError as error:
        raise X12DataError(f"events-{match_id} failed native validation") from error
    events = _strict_json_array(payload, context=f"events-{match_id}")
    goals: list[dict[str, object]] = []
    dismissal_by_player: dict[int, dict[str, object]] = {}
    prior_source_clock_ms = -1
    maximum_period_clock_ms: dict[int, int] = {}
    time_regressions = 0
    for event in events:
        event_index = event.get("index")
        if type(event_index) is not int or event_index <= 0:
            raise X12DataError("event index must be a positive integer")
        period, period_clock_ms, source_clock_ms = (
            _validate_native_event_time(event)
        )
        if source_clock_ms < prior_source_clock_ms:
            time_regressions += 1
        prior_source_clock_ms = source_clock_ms
        maximum_period_clock_ms[period] = max(
            maximum_period_clock_ms.get(period, -1),
            period_clock_ms,
        )
        native_event_id = event.get("id")
        if type(native_event_id) is not str or not native_event_id:
            raise X12DataError("event id must be a nonempty source string")
        team_id = _event_team_id(event)
        if team_id not in {home_team_id, away_team_id}:
            raise X12DataError(f"events-{match_id} contains an unknown team")
        scoring_side = _goal_side(
            event,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        if scoring_side is not None:
            goals.append(
                {
                    "match_id": match_id,
                    "period": period,
                    "period_clock_ms": period_clock_ms,
                    "source_clock_ms": source_clock_ms,
                    "event_index": event_index,
                    "native_event_id": native_event_id,
                    "scoring_side": scoring_side,
                }
            )
        dismissal = _dismissal(
            event,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        if dismissal is not None:
            player_id, dismissal_side, card = dismissal
            candidate: dict[str, object] = {
                "match_id": match_id,
                "period": period,
                "period_clock_ms": period_clock_ms,
                "source_clock_ms": source_clock_ms,
                "event_index": event_index,
                "native_event_id": native_event_id,
                "player_id": player_id,
                "dismissal_side": dismissal_side,
                "card": card,
            }
            prior = dismissal_by_player.get(player_id)
            if prior is None or (
                int(candidate["period"]),
                int(candidate["period_clock_ms"]),
                int(candidate["event_index"]),
            ) < (
                int(prior["period"]),
                int(prior["period_clock_ms"]),
                int(prior["event_index"]),
            ):
                dismissal_by_player[player_id] = candidate
    if not {1, 2} <= set(maximum_period_clock_ms):
        raise X12DataError("regulation match requires periods 1 and 2")
    period_offsets_ms: dict[int, int] = {}
    next_offset_ms = 0
    for period in sorted(maximum_period_clock_ms):
        period_offsets_ms[period] = next_offset_ms
        next_offset_ms += maximum_period_clock_ms[period] + 1
    for item in (*goals, *dismissal_by_player.values()):
        item["global_elapsed_ms"] = (
            period_offsets_ms[int(item["period"])]
            + int(item["period_clock_ms"])
        )
    goals.sort(
        key=lambda value: (
            value["global_elapsed_ms"],
            value["event_index"],
        )
    )
    dismissals = sorted(
        dismissal_by_player.values(),
        key=lambda value: (
            value["global_elapsed_ms"],
            value["event_index"],
            value["player_id"],
        ),
    )
    observed = Counter(str(goal["scoring_side"]) for goal in goals)
    if (
        observed["home_goal"] != expected_home_score
        or observed["away_goal"] != expected_away_score
    ):
        raise X12DataError(
            f"events-{match_id} goal timeline does not reconcile to final score"
        )
    return goals, dismissals, time_regressions, period_offsets_ms


def load_x12_dataset(
    *,
    store_root: str | Path,
    program_root: str | Path,
    manifest_paths: Iterable[str | Path] | None = None,
) -> X12LoadedDataset:
    """Verify and adapt the exact 380-match StatsBomb X-12 source snapshot."""

    store = Path(store_root)
    selected = (
        list(_discover_manifest_paths(store))
        if manifest_paths is None
        else [Path(path) for path in manifest_paths]
    )
    if len(selected) != STATSBOMB_EXPECTED_MATCHES + 1:
        raise X12DataError("X-12 requires one match index and 380 event manifests")
    spellings = [os.fspath(path) for path in selected]
    if len(set(spellings)) != len(spellings):
        raise X12DataError("X-12 manifest_paths contain a duplicate")

    verified_objects = [
        (
            path,
            read_verified_static_object(
                path,
                store_root=store,
                program_root=program_root,
            ),
        )
        for path in selected
    ]
    index_candidates = [
        item
        for item in verified_objects
        if item[1].record.partition == "matches-2-27"
    ]
    if len(index_candidates) != 1:
        raise X12DataError("X-12 requires exactly one verified match index")
    index_path, verified_index = index_candidates[0]
    try:
        index_audit = inspect_statsbomb_match_index(verified_index.object_bytes)
    except StatsBombSourceError as error:
        raise X12DataError("StatsBomb match index failed native validation") from error
    native_matches = _strict_json_array(
        verified_index.object_bytes,
        context="matches-2-27",
    )
    matches_by_id = {int(match["match_id"]): match for match in native_matches}
    if set(matches_by_id) != set(index_audit.match_ids):
        raise X12DataError("match index identity changed during adaptation")

    events_by_match: dict[int, tuple[Path, Any]] = {}
    for path, verified in verified_objects:
        partition = str(verified.record.partition)
        if partition == "matches-2-27":
            continue
        if not partition.startswith("events-"):
            raise X12DataError(f"unexpected X-12 partition: {partition}")
        raw_match_id = partition.removeprefix("events-")
        if not raw_match_id.isdigit() or str(int(raw_match_id)) != raw_match_id:
            raise X12DataError("event partition has a noncanonical match_id")
        match_id = int(raw_match_id)
        if match_id in events_by_match:
            raise X12DataError(f"duplicate event partition for match {match_id}")
        events_by_match[match_id] = (path, verified)
    if set(events_by_match) != set(index_audit.match_ids):
        missing = sorted(set(index_audit.match_ids) - set(events_by_match))
        extra = sorted(set(events_by_match) - set(index_audit.match_ids))
        raise X12DataError(
            f"event partitions do not exactly cover match index; "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    index_inventory = _manifest_inventory(
        verified_index,
        manifest_path=index_path,
        store_root=store,
        event_count=None,
    )
    match_rows: list[dict[str, object]] = []
    goal_rows: list[dict[str, object]] = []
    dismissal_rows: list[dict[str, object]] = []
    event_inventory: list[X12ManifestInventory] = []
    source_time_regressions = 0
    for match_id in index_audit.chronological_match_ids:
        match = matches_by_id[match_id]
        home_team_id, home_team_name = _team(match, "home")
        away_team_id, away_team_name = _team(match, "away")
        if home_team_id == away_team_id:
            raise X12DataError("home and away team must differ")
        home_score = _score(match.get("home_score"), "home_score")
        away_score = _score(match.get("away_score"), "away_score")
        played_at = _played_at(match)
        event_path, verified_event = events_by_match[match_id]
        try:
            event_audit = inspect_statsbomb_event(
                verified_event.object_bytes,
                match_id=match_id,
            )
        except StatsBombSourceError as error:
            raise X12DataError(
                f"events-{match_id} failed native validation"
            ) from error
        (
            adapted_goals,
            adapted_dismissals,
            regressions,
            period_offsets_ms,
        ) = _adapt_events(
            verified_event.object_bytes,
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            expected_home_score=home_score,
            expected_away_score=away_score,
        )
        source_time_regressions += regressions
        goal_rows.extend(adapted_goals)
        dismissal_rows.extend(adapted_dismissals)
        event_manifest = _manifest_inventory(
            verified_event,
            manifest_path=event_path,
            store_root=store,
            event_count=event_audit.event_count,
        )
        if event_manifest.partition != f"events-{match_id}":
            raise X12DataError("event manifest does not bind to its match")
        event_inventory.append(event_manifest)
        outcome = (
            "home_win"
            if home_score > away_score
            else "away_win"
            if away_score > home_score
            else "draw"
        )
        goal_material = [
            {
                "period": int(goal["period"]),
                "period_clock_ms": int(goal["period_clock_ms"]),
                "global_elapsed_ms": int(goal["global_elapsed_ms"]),
                "source_clock_ms": int(goal["source_clock_ms"]),
                "event_index": int(goal["event_index"]),
                "native_event_id": str(goal["native_event_id"]),
                "scoring_side": str(goal["scoring_side"]),
            }
            for goal in adapted_goals
        ]
        dismissal_material = [
            {
                "period": int(dismissal["period"]),
                "period_clock_ms": int(
                    dismissal["period_clock_ms"]
                ),
                "global_elapsed_ms": int(
                    dismissal["global_elapsed_ms"]
                ),
                "source_clock_ms": int(dismissal["source_clock_ms"]),
                "event_index": int(dismissal["event_index"]),
                "native_event_id": str(
                    dismissal["native_event_id"]
                ),
                "player_id": int(dismissal["player_id"]),
                "dismissal_side": str(dismissal["dismissal_side"]),
                "card": str(dismissal["card"]),
            }
            for dismissal in adapted_dismissals
        ]
        match_rows.append(
            {
                "match_id": match_id,
                "match_date": played_at.normalize(),
                "played_at": played_at,
                "feature_available_at": played_at - pd.Timedelta(microseconds=1),
                "outcome_available_at": played_at + pd.Timedelta(hours=3),
                "home_team_id": home_team_id,
                "home_team_name": home_team_name,
                "away_team_id": away_team_id,
                "away_team_name": away_team_name,
                "home_score": home_score,
                "away_score": away_score,
                "outcome": outcome,
                "event_count": event_audit.event_count,
                "event_manifest_path": event_manifest.manifest_path,
                "event_manifest_sha256": event_manifest.manifest_sha256,
                "event_object_sha256": event_manifest.object_sha256,
                "event_schema_fingerprint": event_manifest.schema_fingerprint,
                "goal_timeline_sha256": _sha256(goal_material),
                "dismissal_timeline_sha256": _sha256(
                    dismissal_material
                ),
                "period_2_global_offset_ms": period_offsets_ms[2],
                "source_time_regressions": regressions,
            }
        )
    matches = (
        pd.DataFrame(match_rows)
        .sort_values(["played_at", "match_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if matches["match_id"].nunique() != STATSBOMB_EXPECTED_MATCHES:
        raise X12DataError("canonical match table does not contain 380 unique games")
    for timestamp_column in (
        "match_date",
        "played_at",
        "feature_available_at",
        "outcome_available_at",
    ):
        if matches[timestamp_column].isna().any():
            raise X12DataError(f"{timestamp_column} contains missing timestamps")
        if str(matches[timestamp_column].dt.tz) != "UTC":
            raise X12DataError(f"{timestamp_column} must be timezone-aware UTC")
    if not (matches["feature_available_at"] < matches["played_at"]).all():
        raise X12DataError("pregame feature availability is not strictly PIT")

    goals = pd.DataFrame(
        goal_rows,
        columns=(
            "match_id",
            "period",
            "period_clock_ms",
            "global_elapsed_ms",
            "source_clock_ms",
            "event_index",
            "native_event_id",
            "scoring_side",
        ),
    )
    if not goals.empty:
        goals = goals.sort_values(
            ["match_id", "global_elapsed_ms", "event_index"],
            kind="mergesort",
        ).reset_index(drop=True)
    dismissals = pd.DataFrame(
        dismissal_rows,
        columns=(
            "match_id",
            "period",
            "period_clock_ms",
            "global_elapsed_ms",
            "source_clock_ms",
            "event_index",
            "native_event_id",
            "player_id",
            "dismissal_side",
            "card",
        ),
    )
    if not dismissals.empty:
        dismissals = dismissals.sort_values(
            ["match_id", "global_elapsed_ms", "event_index", "player_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    team_ids = set(matches["home_team_id"]) | set(matches["away_team_id"])
    ordered_inventory = (
        index_inventory,
        *sorted(
            event_inventory,
            key=lambda item: index_audit.chronological_match_ids.index(
                int(item.partition.removeprefix("events-"))
            ),
        ),
    )
    inventory_without_hash = X12InputInventory(
        dataset_id=X12_DATASET_ID,
        source=X12_SOURCE,
        version=X12_STATSBOMB_VERSION,
        competition_id=2,
        season_id=27,
        match_count=len(matches),
        event_partition_count=len(event_inventory),
        total_events=sum(
            int(item.event_count or 0) for item in event_inventory
        ),
        team_count=len(team_ids),
        first_played_at=_frame_timestamp(matches["played_at"].min()),
        last_played_at=_frame_timestamp(matches["played_at"].max()),
        license_ref="O-004",
        license_status="research_only",
        kickoff_time_basis=KICKOFF_TIME_BASIS,
        manifests=ordered_inventory,
        manifest_paths=tuple(item.manifest_path for item in ordered_inventory),
        inventory_sha256="",
    )
    inventory = replace(
        inventory_without_hash,
        inventory_sha256=inventory_sha256(inventory_without_hash),
    )
    return X12LoadedDataset(
        inventory=inventory,
        matches=matches,
        goals=goals,
        dismissals=dismissals,
        chronology_sha256=chronology_sha256(matches),
        goal_timeline_sha256=goal_timeline_sha256(goals),
        dismissal_timeline_sha256=dismissal_timeline_sha256(
            dismissals
        ),
        source_time_regressions=source_time_regressions,
    )


def _unpack_parameters(
    parameters: np.ndarray,
    *,
    team_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    attacks = np.zeros(team_count, dtype=float)
    attacks[1:] = parameters[: team_count - 1]
    defenses = parameters[team_count - 1 : 2 * team_count - 1]
    home_advantage = float(parameters[-2])
    rho = float(parameters[-1])
    return attacks, defenses, home_advantage, rho


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: np.ndarray,
    away_rate: np.ndarray,
    rho: float,
) -> np.ndarray:
    result = np.ones(len(home_goals), dtype=float)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)
    result[zero_zero] = 1.0 - home_rate[zero_zero] * away_rate[zero_zero] * rho
    result[zero_one] = 1.0 + home_rate[zero_one] * rho
    result[one_zero] = 1.0 + away_rate[one_zero] * rho
    result[one_one] = 1.0 - rho
    return result


def _invalid_likelihood_gradient(
    parameters: np.ndarray,
    *,
    parameter_count: int,
) -> np.ndarray:
    vector = np.asarray(parameters, dtype=float)
    if vector.shape != (parameter_count,):
        vector = np.zeros(parameter_count, dtype=float)
    safe = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    gradient = np.sign(safe)
    gradient[gradient == 0] = 1.0
    return gradient


def _is_valid_dixon_coles_likelihood_point(
    parameters: np.ndarray,
    *,
    home_index: np.ndarray,
    away_index: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    team_count: int,
) -> bool:
    """Independently validate that a point is inside the likelihood domain."""

    vector = np.asarray(parameters, dtype=float)
    parameter_count = 2 * team_count + 1
    if vector.shape != (parameter_count,) or not np.all(np.isfinite(vector)):
        return False
    if (
        home_index.shape != away_index.shape
        or home_index.shape != home_goals.shape
        or home_index.shape != away_goals.shape
        or len(home_index) == 0
        or np.any(home_index < 0)
        or np.any(home_index >= team_count)
        or np.any(away_index < 0)
        or np.any(away_index >= team_count)
        or np.any(home_goals < 0)
        or np.any(away_goals < 0)
    ):
        return False
    attacks, defenses, home_advantage, rho = _unpack_parameters(
        vector,
        team_count=team_count,
    )
    home_linear = home_advantage + attacks[home_index] + defenses[away_index]
    away_linear = attacks[away_index] + defenses[home_index]
    if (
        np.any(~np.isfinite(home_linear))
        or np.any(~np.isfinite(away_linear))
        or np.any(np.abs(home_linear) > 10)
        or np.any(np.abs(away_linear) > 10)
    ):
        return False
    home_rate = np.exp(home_linear)
    away_rate = np.exp(away_linear)
    correction = _tau(
        home_goals,
        away_goals,
        home_rate,
        away_rate,
        rho,
    )
    if (
        np.any(~np.isfinite(home_rate))
        or np.any(~np.isfinite(away_rate))
        or np.any(home_rate <= 0)
        or np.any(away_rate <= 0)
        or np.any(~np.isfinite(correction))
        or np.any(correction <= 0)
    ):
        return False
    log_likelihood = (
        home_goals * home_linear
        - home_rate
        - gammaln(home_goals + 1)
        + away_goals * away_linear
        - away_rate
        - gammaln(away_goals + 1)
        + np.log(correction)
    )
    objective = -float(log_likelihood.sum()) + 0.001 * float(
        np.square(vector[:-1]).sum()
    )
    return (
        np.all(np.isfinite(log_likelihood))
        and math.isfinite(objective)
        and objective < _INVALID_LIKELIHOOD_OBJECTIVE
    )


def _dixon_coles_objective_and_gradient(
    parameters: np.ndarray,
    *,
    home_index: np.ndarray,
    away_index: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    team_count: int,
) -> tuple[float, np.ndarray]:
    """Return the regularized negative log likelihood and analytic gradient."""

    vector = np.asarray(parameters, dtype=float)
    parameter_count = 2 * team_count + 1
    if vector.shape != (parameter_count,) or not np.all(np.isfinite(vector)):
        return (
            _INVALID_LIKELIHOOD_OBJECTIVE,
            _invalid_likelihood_gradient(
                vector,
                parameter_count=parameter_count,
            ),
        )
    attacks, defenses, home_advantage, rho = _unpack_parameters(
        vector,
        team_count=team_count,
    )
    home_linear = home_advantage + attacks[home_index] + defenses[away_index]
    away_linear = attacks[away_index] + defenses[home_index]
    if np.any(np.abs(home_linear) > 10) or np.any(np.abs(away_linear) > 10):
        return (
            _INVALID_LIKELIHOOD_OBJECTIVE,
            _invalid_likelihood_gradient(
                vector,
                parameter_count=parameter_count,
            ),
        )
    home_rate = np.exp(home_linear)
    away_rate = np.exp(away_linear)
    correction = _tau(
        home_goals,
        away_goals,
        home_rate,
        away_rate,
        rho,
    )
    if np.any(~np.isfinite(correction)) or np.any(correction <= 0):
        return (
            _INVALID_LIKELIHOOD_OBJECTIVE,
            _invalid_likelihood_gradient(
                vector,
                parameter_count=parameter_count,
            ),
        )

    log_likelihood = (
        home_goals * home_linear
        - home_rate
        - gammaln(home_goals + 1)
        + away_goals * away_linear
        - away_rate
        - gammaln(away_goals + 1)
        + np.log(correction)
    )
    objective = -float(log_likelihood.sum()) + 0.001 * float(
        np.square(vector[:-1]).sum()
    )
    if not math.isfinite(objective):
        return (
            _INVALID_LIKELIHOOD_OBJECTIVE,
            _invalid_likelihood_gradient(
                vector,
                parameter_count=parameter_count,
            ),
        )

    home_linear_gradient = home_rate - home_goals
    away_linear_gradient = away_rate - away_goals
    rho_gradient = np.zeros(len(home_goals), dtype=float)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)

    home_linear_gradient[zero_zero] += (
        home_rate[zero_zero]
        * away_rate[zero_zero]
        * rho
        / correction[zero_zero]
    )
    away_linear_gradient[zero_zero] += (
        home_rate[zero_zero]
        * away_rate[zero_zero]
        * rho
        / correction[zero_zero]
    )
    rho_gradient[zero_zero] = (
        home_rate[zero_zero]
        * away_rate[zero_zero]
        / correction[zero_zero]
    )
    home_linear_gradient[zero_one] -= (
        home_rate[zero_one] * rho / correction[zero_one]
    )
    rho_gradient[zero_one] = -home_rate[zero_one] / correction[zero_one]
    away_linear_gradient[one_zero] -= (
        away_rate[one_zero] * rho / correction[one_zero]
    )
    rho_gradient[one_zero] = -away_rate[one_zero] / correction[one_zero]
    rho_gradient[one_one] = 1.0 / correction[one_one]

    attack_gradient = np.zeros(team_count, dtype=float)
    defense_gradient = np.zeros(team_count, dtype=float)
    np.add.at(attack_gradient, home_index, home_linear_gradient)
    np.add.at(attack_gradient, away_index, away_linear_gradient)
    np.add.at(defense_gradient, away_index, home_linear_gradient)
    np.add.at(defense_gradient, home_index, away_linear_gradient)
    gradient = np.zeros(parameter_count, dtype=float)
    gradient[: team_count - 1] = attack_gradient[1:]
    gradient[team_count - 1 : 2 * team_count - 1] = defense_gradient
    gradient[-2] = float(home_linear_gradient.sum())
    gradient[-1] = float(rho_gradient.sum())
    gradient[:-1] += 0.002 * vector[:-1]
    if not np.all(np.isfinite(gradient)):
        return (
            _INVALID_LIKELIHOOD_OBJECTIVE,
            _invalid_likelihood_gradient(
                vector,
                parameter_count=parameter_count,
            ),
        )
    return objective, gradient


def _projected_gradient_inf_norm(
    parameters: np.ndarray,
    gradient: np.ndarray,
    bounds: Sequence[tuple[float, float]],
) -> float:
    vector = np.asarray(parameters, dtype=float)
    projected = np.asarray(gradient, dtype=float).copy()
    if (
        vector.ndim != 1
        or projected.ndim != 1
        or vector.shape != projected.shape
        or len(bounds) != len(vector)
    ):
        raise X12DataError(
            "projected-gradient parameters, gradient, and bounds must align"
        )
    _validate_parameter_bounds(
        vector,
        bounds,
        context="projected-gradient",
    )
    for index, (lower, upper) in enumerate(bounds):
        value = float(vector[index])
        at_lower_bound = (
            abs(value - lower) <= _KKT_BOUNDARY_ABS_TOLERANCE
        )
        at_upper_bound = (
            abs(value - upper) <= _KKT_BOUNDARY_ABS_TOLERANCE
        )
        if (at_lower_bound and projected[index] > 0) or (
            at_upper_bound and projected[index] < 0
        ):
            projected[index] = 0.0
    if not np.all(np.isfinite(projected)):
        raise X12DataError("projected-gradient input must be finite")
    return float(np.max(np.abs(projected)))


def _bound_machine_tolerance(lower: float, upper: float) -> float:
    """Return the explicit floating-point roundoff allowance for one bound."""

    return float(
        _BOUND_ROUNDOFF_ULPS
        * max(math.ulp(float(lower)), math.ulp(float(upper)))
    )


def _validate_parameter_bounds(
    parameters: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    context: str,
) -> None:
    vector = np.asarray(parameters, dtype=float)
    if vector.ndim != 1 or len(vector) != len(bounds):
        raise X12DataError(f"{context} parameters do not match frozen bounds")
    for index, (value_raw, bound) in enumerate(zip(vector, bounds, strict=True)):
        lower, upper = (float(bound[0]), float(bound[1]))
        value = float(value_raw)
        if (
            not math.isfinite(value)
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
        ):
            raise X12DataError(
                f"{context} parameter[{index}] or bound is non-finite/invalid"
            )
        tolerance = _bound_machine_tolerance(lower, upper)
        below_by = lower - value
        above_by = value - upper
        if below_by > tolerance or above_by > tolerance:
            raise X12DataError(
                f"{context} parameter[{index}]={value} is outside bounds "
                f"[{lower}, {upper}] beyond {_BOUND_ROUNDOFF_ULPS} ULP "
                "machine-roundoff tolerance"
            )


def _fit_dixon_coles(
    train: pd.DataFrame,
    *,
    team_ids: tuple[int, ...],
    optimizer_max_iterations: int,
    initial_parameters: Sequence[float] | None,
) -> DixonColesModel:
    if len(team_ids) < 2:
        raise X12DataError("Dixon-Coles requires at least two teams")
    team_index = {team_id: index for index, team_id in enumerate(team_ids)}
    try:
        home_index = np.asarray(
            [team_index[int(value)] for value in train["home_team_id"]],
            dtype=int,
        )
        away_index = np.asarray(
            [team_index[int(value)] for value in train["away_team_id"]],
            dtype=int,
        )
    except KeyError as error:
        raise X12DataError("training match contains an unknown team") from error
    if set(home_index) | set(away_index) != set(range(len(team_ids))):
        raise X12DataError("every frozen league team must appear before model fitting")
    home_goals = train["home_score"].to_numpy(dtype=int)
    away_goals = train["away_score"].to_numpy(dtype=int)
    parameter_count = 2 * len(team_ids) + 1
    bounds = [
        *[(-3.0, 3.0)] * (parameter_count - 2),
        (-1.5, 1.5),
        (-0.2, 0.2),
    ]
    if initial_parameters is None:
        initial = np.zeros(parameter_count, dtype=float)
        mean_home = max(float(home_goals.mean()), 0.1)
        mean_away = max(float(away_goals.mean()), 0.1)
        initial[-2] = float(np.clip(math.log(mean_home / mean_away), -0.5, 0.5))
        initial[-1] = -0.05
    else:
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.shape != (parameter_count,) or not np.all(np.isfinite(initial)):
            raise X12DataError("warm-start parameters do not match the frozen league")
    _validate_parameter_bounds(
        initial,
        bounds,
        context="initial",
    )
    if not _is_valid_dixon_coles_likelihood_point(
        initial,
        home_index=home_index,
        away_index=away_index,
        home_goals=home_goals,
        away_goals=away_goals,
        team_count=len(team_ids),
    ):
        raise X12DataError(
            "Dixon-Coles initial point is outside the valid likelihood domain"
        )

    def objective(parameters: np.ndarray) -> float:
        return _dixon_coles_objective_and_gradient(
            parameters,
            home_index=home_index,
            away_index=away_index,
            home_goals=home_goals,
            away_goals=away_goals,
            team_count=len(team_ids),
        )[0]

    def gradient(parameters: np.ndarray) -> np.ndarray:
        return _dixon_coles_objective_and_gradient(
            parameters,
            home_index=home_index,
            away_index=away_index,
            home_goals=home_goals,
            away_goals=away_goals,
            team_count=len(team_ids),
        )[1]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        options={
            "maxiter": optimizer_max_iterations,
            "ftol": 1e-12,
        },
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise X12DataError(
            "Dixon-Coles optimizer failed closed: "
            f"status={result.status} message={result.message}"
        )
    final_parameters = np.asarray(result.x, dtype=float)
    _validate_parameter_bounds(
        final_parameters,
        bounds,
        context="final",
    )
    if (
        float(result.fun) >= _INVALID_LIKELIHOOD_OBJECTIVE
        or not _is_valid_dixon_coles_likelihood_point(
            final_parameters,
            home_index=home_index,
            away_index=away_index,
            home_goals=home_goals,
            away_goals=away_goals,
            team_count=len(team_ids),
        )
    ):
        raise X12DataError(
            "Dixon-Coles optimizer returned a point outside the valid "
            "likelihood domain"
        )
    initial_objective, initial_gradient = _dixon_coles_objective_and_gradient(
        initial,
        home_index=home_index,
        away_index=away_index,
        home_goals=home_goals,
        away_goals=away_goals,
        team_count=len(team_ids),
    )
    initial_projected_gradient_inf_norm = _projected_gradient_inf_norm(
        initial,
        initial_gradient,
        bounds,
    )
    final_objective, final_gradient = _dixon_coles_objective_and_gradient(
        final_parameters,
        home_index=home_index,
        away_index=away_index,
        home_goals=home_goals,
        away_goals=away_goals,
        team_count=len(team_ids),
    )
    objective_improvement = initial_objective - final_objective
    parameter_displacement = float(
        np.linalg.norm(final_parameters - initial)
    )
    minimum_improvement = max(1e-8, abs(initial_objective) * 1e-10)
    if final_objective > initial_objective + minimum_improvement:
        raise X12DataError(
            "Dixon-Coles optimizer objective worsened: "
            f"initial={initial_objective} final={final_objective}"
        )
    if not math.isclose(
        float(result.fun),
        final_objective,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise X12DataError(
            "Dixon-Coles optimizer reported an inconsistent objective"
        )
    made_progress = (
        objective_improvement > minimum_improvement
        and parameter_displacement > 1e-8
    )
    started_converged = initial_projected_gradient_inf_norm <= 1e-4
    if not made_progress and not started_converged:
        raise X12DataError(
            "Dixon-Coles optimizer failed progress checks: "
            f"objective_improvement={objective_improvement} "
            f"parameter_displacement={parameter_displacement} "
            "initial_projected_gradient_inf_norm="
            f"{initial_projected_gradient_inf_norm}"
        )
    projected_gradient_inf_norm = _projected_gradient_inf_norm(
        final_parameters,
        final_gradient,
        bounds,
    )
    if projected_gradient_inf_norm > 1e-4:
        raise X12DataError(
            "Dixon-Coles optimizer failed gradient convergence check: "
            f"projected_gradient_inf_norm={projected_gradient_inf_norm}"
        )
    _, _, home_advantage, rho = _unpack_parameters(
        final_parameters,
        team_count=len(team_ids),
    )
    return DixonColesModel(
        team_ids=team_ids,
        reference_team_id=team_ids[0],
        parameters=tuple(float(value) for value in final_parameters),
        home_advantage=home_advantage,
        rho=rho,
        initial_objective=initial_objective,
        initial_projected_gradient_inf_norm=(
            initial_projected_gradient_inf_norm
        ),
        objective=final_objective,
        objective_improvement=objective_improvement,
        parameter_displacement=parameter_displacement,
        projected_gradient_inf_norm=projected_gradient_inf_norm,
        iterations=int(result.nit),
        optimizer_status=str(result.message),
    )


def _expected_goals(
    model: DixonColesModel,
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[float, float]:
    team_index = {team_id: index for index, team_id in enumerate(model.team_ids)}
    if home_team_id not in team_index or away_team_id not in team_index:
        raise X12DataError("prediction contains a team outside the fitted model")
    parameters = np.asarray(model.parameters, dtype=float)
    attacks, defenses, home_advantage, _ = _unpack_parameters(
        parameters,
        team_count=len(model.team_ids),
    )
    home_index = team_index[home_team_id]
    away_index = team_index[away_team_id]
    home_rate = math.exp(
        home_advantage + attacks[home_index] + defenses[away_index]
    )
    away_rate = math.exp(attacks[away_index] + defenses[home_index])
    if (
        not math.isfinite(home_rate)
        or not math.isfinite(away_rate)
        or home_rate <= 0
        or away_rate <= 0
    ):
        raise X12DataError("Dixon-Coles emitted invalid expected goals")
    return home_rate, away_rate


def _timeline_by_match(
    frame: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    return {
        int(match_id): rows.sort_values(
            ["global_elapsed_ms", "event_index"],
            kind="mergesort",
        )
        for match_id, rows in frame.groupby("match_id", sort=False)
    }


def _events_at_or_before_cutoff(
    frame: pd.DataFrame,
    *,
    period: int,
    period_clock_ms: int,
) -> pd.DataFrame:
    return frame.loc[
        (frame["period"] < period)
        | (
            (frame["period"] == period)
            & (frame["period_clock_ms"] <= period_clock_ms)
        )
    ].sort_values(["global_elapsed_ms", "event_index"], kind="mergesort")


def _events_in_transition_window(
    frame: pd.DataFrame,
    *,
    period: int,
    cutoff_period_clock_ms: int,
) -> pd.DataFrame:
    end_period_clock_ms = (
        cutoff_period_clock_ms + TRANSITION_HORIZON_SECONDS * 1_000
    )
    return frame.loc[
        (frame["period"] == period)
        & (frame["period_clock_ms"] > cutoff_period_clock_ms)
        & (frame["period_clock_ms"] <= end_period_clock_ms)
    ].sort_values(["period_clock_ms", "event_index"], kind="mergesort")


def _state_at_cutoff(
    goals: pd.DataFrame,
    dismissals: pd.DataFrame,
    *,
    period: int,
    period_clock_ms: int,
) -> tuple[int, int, int, int]:
    known_goals = _events_at_or_before_cutoff(
        goals,
        period=period,
        period_clock_ms=period_clock_ms,
    )
    known_dismissals = _events_at_or_before_cutoff(
        dismissals,
        period=period,
        period_clock_ms=period_clock_ms,
    )
    return (
        int((known_goals["scoring_side"] == "home_goal").sum()),
        int((known_goals["scoring_side"] == "away_goal").sum()),
        int(
            (
                known_dismissals["dismissal_side"]
                == "home_dismissal"
            ).sum()
        ),
        int(
            (
                known_dismissals["dismissal_side"]
                == "away_dismissal"
            ).sum()
        ),
    )


def _empty_goal_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "match_id",
            "period",
            "period_clock_ms",
            "global_elapsed_ms",
            "source_clock_ms",
            "event_index",
            "native_event_id",
            "scoring_side",
        )
    )


def _empty_dismissal_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "match_id",
            "period",
            "period_clock_ms",
            "global_elapsed_ms",
            "source_clock_ms",
            "event_index",
            "native_event_id",
            "player_id",
            "dismissal_side",
            "card",
        )
    )


def _snapshot_global_elapsed_ms(
    match: Any,
    *,
    period: int,
    period_clock_ms: int,
) -> int:
    if period == 1:
        return period_clock_ms
    if period == 2:
        offset = int(match.period_2_global_offset_ms)
        if offset <= 45 * 60 * 1_000:
            raise X12DataError(
                "second-period global offset must follow regulation minute 45"
            )
        return offset + period_clock_ms
    raise X12DataError("transition snapshots are limited to periods 1 and 2")


def _dynamic_training_rows(
    train: pd.DataFrame,
    goals: pd.DataFrame,
    dismissals: pd.DataFrame,
    *,
    model: DixonColesModel,
) -> pd.DataFrame:
    goals_by_match = _timeline_by_match(goals)
    dismissals_by_match = _timeline_by_match(dismissals)
    rows: list[dict[str, object]] = []
    for match in train.itertuples(index=False):
        match_id = int(match.match_id)
        home_rate, away_rate = _expected_goals(
            model,
            home_team_id=int(match.home_team_id),
            away_team_id=int(match.away_team_id),
        )
        match_goals = goals_by_match.get(
            match_id,
            _empty_goal_timeline(),
        )
        match_dismissals = dismissals_by_match.get(
            match_id,
            _empty_dismissal_timeline(),
        )
        for period, period_minute in TRANSITION_SNAPSHOTS:
            period_clock_ms = period_minute * 60 * 1_000
            global_elapsed_ms = _snapshot_global_elapsed_ms(
                match,
                period=period,
                period_clock_ms=period_clock_ms,
            )
            (
                home_score,
                away_score,
                home_dismissals,
                away_dismissals,
            ) = _state_at_cutoff(
                match_goals,
                match_dismissals,
                period=period,
                period_clock_ms=period_clock_ms,
            )
            future = _events_in_transition_window(
                match_goals,
                period=period,
                cutoff_period_clock_ms=period_clock_ms,
            )
            feature_available_at = match.played_at + pd.Timedelta(
                milliseconds=global_elapsed_ms
            )
            label_available_at = match.played_at + pd.Timedelta(
                milliseconds=(
                    global_elapsed_ms
                    + TRANSITION_HORIZON_SECONDS * 1_000
                )
            )
            home_score_difference = home_score - away_score
            home_dismissal_difference = (
                home_dismissals - away_dismissals
            )
            common: dict[str, object] = {
                "match_id": match_id,
                "exposure_seconds": TRANSITION_HORIZON_SECONDS,
                "second_half": int(period == 2),
                "feature_available_at": feature_available_at,
                "label_available_at": label_available_at,
            }
            rows.extend(
                (
                    {
                        **common,
                        "side": "home",
                        "base_goals_per_90": home_rate,
                        "goal_count": int(
                            (
                                future["scoring_side"]
                                == "home_goal"
                            ).sum()
                        ),
                        "score_difference": home_score_difference,
                        "dismissal_difference": (
                            home_dismissal_difference
                        ),
                    },
                    {
                        **common,
                        "side": "away",
                        "base_goals_per_90": away_rate,
                        "goal_count": int(
                            (
                                future["scoring_side"]
                                == "away_goal"
                            ).sum()
                        ),
                        "score_difference": -home_score_difference,
                        "dismissal_difference": (
                            -home_dismissal_difference
                        ),
                    },
                )
            )
    return pd.DataFrame(rows)


def _source_state_sha256(
    match: Any,
    goals: pd.DataFrame,
    dismissals: pd.DataFrame,
    *,
    period: int,
    period_clock_ms: int,
    home_score: int,
    away_score: int,
    home_dismissals: int,
    away_dismissals: int,
) -> str:
    expected_state = _state_at_cutoff(
        goals,
        dismissals,
        period=period,
        period_clock_ms=period_clock_ms,
    )
    supplied_state = (
        home_score,
        away_score,
        home_dismissals,
        away_dismissals,
    )
    if expected_state != supplied_state:
        raise X12DataError("cutoff state does not match its observed prefix")
    known_goals = _events_at_or_before_cutoff(
        goals,
        period=period,
        period_clock_ms=period_clock_ms,
    )
    known_dismissals = _events_at_or_before_cutoff(
        dismissals,
        period=period,
        period_clock_ms=period_clock_ms,
    )
    goal_prefix = [
        {
            "period": int(row.period),
            "period_clock_ms": int(row.period_clock_ms),
            "event_index": int(row.event_index),
            "native_event_id": str(row.native_event_id),
            "scoring_side": str(row.scoring_side),
        }
        for row in known_goals.itertuples(index=False)
    ]
    dismissal_prefix = [
        {
            "period": int(row.period),
            "period_clock_ms": int(row.period_clock_ms),
            "event_index": int(row.event_index),
            "native_event_id": str(row.native_event_id),
            "player_id": int(row.player_id),
            "dismissal_side": str(row.dismissal_side),
            "card": str(row.card),
        }
        for row in known_dismissals.itertuples(index=False)
    ]
    return _sha256(
        {
            "state_kind": (
                "offline_statsbomb_observed_prefix_snapshot"
            ),
            "match_id": int(match.match_id),
            "period": period,
            "period_clock_ms": period_clock_ms,
            "home_team_id": int(match.home_team_id),
            "away_team_id": int(match.away_team_id),
            "home_score": home_score,
            "away_score": away_score,
            "home_dismissals": home_dismissals,
            "away_dismissals": away_dismissals,
            "goal_prefix": goal_prefix,
            "dismissal_prefix": dismissal_prefix,
        }
    )


def _transition_rows(
    test: pd.DataFrame,
    goals: pd.DataFrame,
    dismissals: pd.DataFrame,
    *,
    model: DixonColesModel,
    dynamic_model: DynamicIntensityModel,
    inventory_sha256_value: str,
    dixon_coles_parameter_sha256: str,
    raw_transition_parameter_sha256: str,
    calibrated_transition_parameter_sha256: str | None,
    temperature_calibration: TemperatureCalibration | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    static_comparator = DynamicIntensityModel(
        coefficients=(0.0, 0.0, 0.0),
        l2_penalty=0.0,
        objective=0.0,
        iterations=0,
        optimizer_status="fixed_zero_dynamic_coefficients",
    )
    goals_by_match = _timeline_by_match(goals)
    dismissals_by_match = _timeline_by_match(dismissals)
    for match in test.itertuples(index=False):
        home_rate, away_rate = _expected_goals(
            model,
            home_team_id=int(match.home_team_id),
            away_team_id=int(match.away_team_id),
        )
        match_goals = goals_by_match.get(
            int(match.match_id),
            _empty_goal_timeline(),
        )
        match_dismissals = dismissals_by_match.get(
            int(match.match_id),
            _empty_dismissal_timeline(),
        )
        for period, period_minute in TRANSITION_SNAPSHOTS:
            period_clock_ms = period_minute * 60 * 1_000
            global_elapsed_ms = _snapshot_global_elapsed_ms(
                match,
                period=period,
                period_clock_ms=period_clock_ms,
            )
            future = _events_in_transition_window(
                match_goals,
                period=period,
                cutoff_period_clock_ms=period_clock_ms,
            )
            observed = (
                str(future.iloc[0]["scoring_side"])
                if not future.empty
                else "no_goal"
            )
            prediction_at = match.played_at + pd.Timedelta(
                milliseconds=global_elapsed_ms
            )
            (
                home_score,
                away_score,
                home_dismissals,
                away_dismissals,
            ) = _state_at_cutoff(
                match_goals,
                match_dismissals,
                period=period,
                period_clock_ms=period_clock_ms,
            )
            source_state_sha256 = _source_state_sha256(
                match,
                match_goals,
                match_dismissals,
                period=period,
                period_clock_ms=period_clock_ms,
                home_score=home_score,
                away_score=away_score,
                home_dismissals=home_dismissals,
                away_dismissals=away_dismissals,
            )
            features = SoccerTransitionFeatures(
                game_id=f"game_{int(match.match_id)}",
                home_team_id=int(match.home_team_id),
                away_team_id=int(match.away_team_id),
                elapsed_seconds=global_elapsed_ms / 1_000.0,
                second_half=int(period == 2),
                home_score=home_score,
                away_score=away_score,
                home_dismissals=home_dismissals,
                away_dismissals=away_dismissals,
                home_score_difference=home_score - away_score,
                home_dismissal_difference=(
                    home_dismissals - away_dismissals
                ),
                source_state_sha256=source_state_sha256,
            )
            try:
                distribution = predict_transition_distribution(
                    dynamic_model,
                    base_home_goals=home_rate,
                    base_away_goals=away_rate,
                    features=features,
                )
                static_distribution = predict_transition_distribution(
                    static_comparator,
                    base_home_goals=home_rate,
                    base_away_goals=away_rate,
                    features=features,
                )
            except SoccerTransitionModelError as error:
                raise X12DataError(
                    "dynamic transition prediction failed closed"
                ) from error
            row: dict[str, object] = {
                "match_id": int(match.match_id),
                "period": period,
                "period_minute": period_minute,
                "period_clock_ms": period_clock_ms,
                "global_elapsed_ms": global_elapsed_ms,
                "prediction_at": prediction_at,
                "feature_available_at": prediction_at,
                "horizon_seconds": TRANSITION_HORIZON_SECONDS,
                "home_score_at_cutoff": home_score,
                "away_score_at_cutoff": away_score,
                "home_dismissals_at_cutoff": home_dismissals,
                "away_dismissals_at_cutoff": away_dismissals,
                "observed_transition": observed,
                "pit_status": OFFLINE_PIT_STATUS,
                "inventory_sha256": inventory_sha256_value,
                "model_parameter_sha256": (
                    calibrated_transition_parameter_sha256
                    if temperature_calibration is not None
                    else raw_transition_parameter_sha256
                ),
                "dixon_coles_parameter_sha256": (
                    dixon_coles_parameter_sha256
                ),
                "raw_transition_parameter_sha256": (
                    raw_transition_parameter_sha256
                ),
                "dynamic_parameter_sha256": (
                    dynamic_model.parameter_sha256
                ),
                "temperature_parameter_sha256": (
                    temperature_calibration.parameter_sha256
                    if temperature_calibration is not None
                    else None
                ),
                "source_state_sha256": source_state_sha256,
                "source_feature_sha256": features.feature_sha256,
                "manifest_sha256": match.event_manifest_sha256,
                "object_sha256": match.event_object_sha256,
                "schema_fingerprint": match.event_schema_fingerprint,
            }
            for index, label in enumerate(TRANSITION_CLASSES):
                row[f"raw_probability_{label}"] = float(
                    distribution.probabilities[index]
                )
                row[f"static_probability_{label}"] = float(
                    static_distribution.probabilities[index]
                )
            rows.append(row)
    raw_probability_matrix = np.asarray(
        [
            [
                float(row[f"raw_probability_{label}"])
                for label in TRANSITION_CLASSES
            ]
            for row in rows
        ],
        dtype=float,
    )
    calibrated_probability_matrix = (
        apply_multiclass_temperature(
            raw_probability_matrix,
            temperature=temperature_calibration.temperature,
        )
        if temperature_calibration is not None
        else raw_probability_matrix
    )
    for row, probabilities in zip(
        rows,
        calibrated_probability_matrix,
        strict=True,
    ):
        for index, label in enumerate(TRANSITION_CLASSES):
            row[f"probability_{label}"] = float(probabilities[index])
    return rows


def _transition_metrics(
    transitions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    minimum_valid_bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, object]:
    def evaluate(
        *,
        probability_prefix: str,
        probability_variant: str,
    ) -> dict[str, object]:
        columns = [
            f"{probability_prefix}probability_{label}"
            for label in TRANSITION_CLASSES
        ]
        try:
            metrics = evaluate_multiclass_probabilities(
                transitions["observed_transition"].to_numpy(dtype=object),
                transitions[columns].to_numpy(dtype=float),
                classes=TRANSITION_CLASSES,
                groups=transitions["match_id"].to_numpy(dtype=object),
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                minimum_valid_samples=minimum_valid_bootstrap_samples,
                seed=X12_SEED,
                prediction_at=transitions["prediction_at"],
                feature_available_at=transitions["feature_available_at"],
                prior_probabilities=transitions[static_columns].to_numpy(
                    dtype=float
                ),
                prior_available_at=transitions["feature_available_at"],
            )
        except ValidationInputError as error:
            raise X12DataError(
                "X-12 transition evaluation failed closed"
            ) from error
        comparison = dict(metrics.pop("prior_comparison"))
        comparison["static_metrics"] = comparison.pop("prior_metrics")
        comparison["delta_definition"] = (
            f"{probability_variant}_model_minus_static_comparator"
        )
        comparison["comparator"] = (
            "pregame_dixon_coles_competing_poisson"
        )
        metrics["static_comparison"] = comparison
        metrics["probability_variant"] = probability_variant
        metrics["target_definition"] = (
            "first scoring side in (cutoff, cutoff+300s]; "
            "no_goal if none"
        )
        metrics["seed"] = X12_SEED
        return metrics

    static_columns = [
        f"static_probability_{label}" for label in TRANSITION_CLASSES
    ]
    calibrated_metrics = evaluate(
        probability_prefix="",
        probability_variant="temperature_calibrated",
    )
    raw_metrics = evaluate(
        probability_prefix="raw_",
        probability_variant="uncalibrated",
    )
    calibrated_metrics["raw_model_metrics"] = raw_metrics
    return calibrated_metrics


def _validate_dynamic_run_parameters(
    *,
    evaluation_match_limit: int | None,
    bootstrap_samples: int,
    minimum_valid_bootstrap_samples: int,
    confidence_level: float,
    optimizer_max_iterations: int,
) -> None:
    if evaluation_match_limit is not None and (
        type(evaluation_match_limit) is not int or evaluation_match_limit < 20
    ):
        raise X12DataError(
            "evaluation_match_limit must be None or an integer >= 20"
        )
    if type(bootstrap_samples) is not int or bootstrap_samples < 20:
        raise X12DataError("bootstrap_samples must be an integer >= 20")
    if (
        type(minimum_valid_bootstrap_samples) is not int
        or minimum_valid_bootstrap_samples < 20
        or minimum_valid_bootstrap_samples > bootstrap_samples
    ):
        raise X12DataError(
            "minimum_valid_bootstrap_samples must be between 20 and requested"
        )
    if type(confidence_level) is not float or not 0 < confidence_level < 1:
        raise X12DataError("confidence_level must be a float in (0, 1)")
    if type(optimizer_max_iterations) is not int or optimizer_max_iterations < 20:
        raise X12DataError("optimizer_max_iterations must be an integer >= 20")


def _whole_date_group_limit(
    matches: pd.DataFrame,
    *,
    match_limit: int | None,
) -> pd.DataFrame:
    if match_limit is None:
        return matches.copy()
    selected_dates: list[pd.Timestamp] = []
    selected_match_count = 0
    for match_date, group in matches.groupby(
        "match_date",
        sort=True,
    ):
        group_match_count = len(group)
        if selected_match_count + group_match_count > match_limit:
            break
        selected_dates.append(pd.Timestamp(match_date))
        selected_match_count += group_match_count
    selected = matches.loc[
        matches["match_date"].isin(selected_dates)
    ].copy()
    if selected["match_id"].nunique() < 20:
        raise X12DataError(
            "bounded transition evaluation requires at least 20 matches "
            "in complete date groups"
        )
    return selected


def _frozen_transition_split(
    matches: pd.DataFrame,
    *,
    evaluation_match_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = [
        pd.Timestamp(value)
        for value in matches["match_date"].drop_duplicates()
    ]
    if len(dates) < 4:
        raise X12DataError(
            "transition holdout requires at least four chronological date groups"
        )
    base_date_count = len(dates) // 2
    calibration_date_count = len(dates) // 4
    if base_date_count == 0 or calibration_date_count == 0:
        raise X12DataError(
            "transition holdout cannot form nonempty 50/25/25 partitions"
        )
    base_dates = dates[:base_date_count]
    calibration_dates = dates[
        base_date_count : base_date_count + calibration_date_count
    ]
    final_dates = dates[base_date_count + calibration_date_count :]
    base = matches.loc[matches["match_date"].isin(base_dates)].copy()
    calibration = matches.loc[
        matches["match_date"].isin(calibration_dates)
    ].copy()
    final_full = matches.loc[matches["match_date"].isin(final_dates)].copy()
    final_evaluated = _whole_date_group_limit(
        final_full,
        match_limit=evaluation_match_limit,
    )
    partitions = (base, calibration, final_full)
    match_id_sets = [
        set(frame["match_id"].astype(int))
        for frame in partitions
    ]
    if (
        any(frame.empty for frame in partitions)
        or match_id_sets[0] & match_id_sets[1]
        or match_id_sets[0] & match_id_sets[2]
        or match_id_sets[1] & match_id_sets[2]
        or set().union(*match_id_sets)
        != set(matches["match_id"].astype(int))
    ):
        raise X12DataError(
            "transition holdout partitions must be disjoint and exhaustive"
        )
    if len(calibration) < _MINIMUM_TRANSITION_CALIBRATION_MATCHES:
        raise X12DataError(
            "transition calibration interval has insufficient matches"
        )
    if not (
        base["played_at"].max() < calibration["played_at"].min()
        and calibration["played_at"].max() < final_full["played_at"].min()
    ):
        raise X12DataError(
            "transition holdout chronology is not strictly ordered"
        )
    return base, calibration, final_full, final_evaluated


def run_x12_dynamic_transition(
    loaded: X12LoadedDataset,
    *,
    evaluation_match_limit: int | None = None,
    bootstrap_samples: int = 200,
    minimum_valid_bootstrap_samples: int = 100,
    confidence_level: float = 0.95,
    optimizer_max_iterations: int = 250,
) -> X12DynamicTransitionEvaluation:
    """Run the frozen date-grouped dynamic-transition reproduction."""

    _validate_dynamic_run_parameters(
        evaluation_match_limit=evaluation_match_limit,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
        optimizer_max_iterations=optimizer_max_iterations,
    )
    if not isinstance(loaded, X12LoadedDataset):
        raise TypeError("loaded must be an X12LoadedDataset")
    if inventory_sha256(loaded.inventory) != loaded.inventory.inventory_sha256:
        raise X12DataError("input inventory self-hash is invalid")
    if chronology_sha256(loaded.matches) != loaded.chronology_sha256:
        raise X12DataError("frozen match chronology SHA-256 is invalid")
    _validate_goal_timeline_binding(
        loaded.matches,
        loaded.goals,
        expected_sha256=loaded.goal_timeline_sha256,
    )
    _validate_dismissal_timeline_binding(
        loaded.matches,
        loaded.dismissals,
        expected_sha256=loaded.dismissal_timeline_sha256,
    )
    matches = loaded.matches.sort_values(
        ["played_at", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(matches) != STATSBOMB_EXPECTED_MATCHES:
        raise X12DataError("X-12 run requires all 380 frozen matches")
    team_ids = tuple(
        sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    )
    if len(team_ids) != loaded.inventory.team_count:
        raise X12DataError("team inventory changed after source adaptation")

    (
        transition_base,
        transition_calibration,
        transition_final_full,
        transition_final_evaluated,
    ) = _frozen_transition_split(
        matches,
        evaluation_match_limit=evaluation_match_limit,
    )
    dynamic_fit_evaluation_cutoff = transition_calibration[
        "played_at"
    ].min()
    temperature_fit_evaluation_cutoff = transition_final_full[
        "played_at"
    ].min()
    if not (
        transition_base["outcome_available_at"].max()
        < dynamic_fit_evaluation_cutoff
    ):
        raise X12DataError(
            "transition base outcomes are not available before calibration"
        )
    transition_dixon_coles = _fit_dixon_coles(
        transition_base,
        team_ids=team_ids,
        optimizer_max_iterations=optimizer_max_iterations,
        initial_parameters=None,
    )
    transition_dixon_coles_parameter_sha256 = _sha256(
        list(transition_dixon_coles.parameters)
    )
    dynamic_training_rows = _dynamic_training_rows(
        transition_base,
        loaded.goals,
        loaded.dismissals,
        model=transition_dixon_coles,
    )
    held_out_transition_match_ids = frozenset(
        set(transition_calibration["match_id"].astype(int))
        | set(transition_final_full["match_id"].astype(int))
    )
    try:
        dynamic_model = fit_dynamic_intensity(
            dynamic_training_rows,
            evaluation_cutoff=dynamic_fit_evaluation_cutoff,
            held_out_match_ids=held_out_transition_match_ids,
            l2_penalty=_DYNAMIC_INTENSITY_L2_PENALTY,
            optimizer_max_iterations=optimizer_max_iterations,
        )
    except SoccerTransitionModelError as error:
        raise X12DataError(
            "dynamic-intensity base fit failed closed"
        ) from error
    raw_transition_parameter_sha256 = _sha256(
        {
            "dixon_coles_parameter_sha256": (
                transition_dixon_coles_parameter_sha256
            ),
            "dynamic_parameter_sha256": dynamic_model.parameter_sha256,
            "model_id": X12_MODEL_ID,
            "model_version": X12_MODEL_VERSION,
            "probability_variant": "uncalibrated",
        }
    )
    calibration_rows = _transition_rows(
        transition_calibration,
        loaded.goals,
        loaded.dismissals,
        model=transition_dixon_coles,
        dynamic_model=dynamic_model,
        inventory_sha256_value=loaded.inventory.inventory_sha256,
        dixon_coles_parameter_sha256=(
            transition_dixon_coles_parameter_sha256
        ),
        raw_transition_parameter_sha256=(
            raw_transition_parameter_sha256
        ),
        calibrated_transition_parameter_sha256=None,
        temperature_calibration=None,
    )
    calibration_predictions = pd.DataFrame(calibration_rows).sort_values(
        ["prediction_at", "match_id", "period", "period_clock_ms"],
        kind="mergesort",
    )
    calibration_max_label_available_at = (
        calibration_predictions["prediction_at"].max()
        + pd.Timedelta(seconds=TRANSITION_HORIZON_SECONDS)
    )
    if not (
        calibration_max_label_available_at
        < temperature_fit_evaluation_cutoff
    ):
        raise X12DataError(
            "calibration labels are not available before final test"
        )
    raw_columns = [
        f"raw_probability_{label}" for label in TRANSITION_CLASSES
    ]
    try:
        temperature_calibration = fit_multiclass_temperature(
            calibration_predictions[
                "observed_transition"
            ].to_numpy(dtype=object),
            calibration_predictions[raw_columns].to_numpy(dtype=float),
            groups=calibration_predictions[
                "match_id"
            ].to_numpy(dtype=object),
            minimum_matches=_MINIMUM_TRANSITION_CALIBRATION_MATCHES,
            optimizer_max_iterations=optimizer_max_iterations,
        )
    except SoccerTransitionModelError as error:
        raise X12DataError(
            "transition temperature calibration failed closed"
        ) from error
    calibrated_transition_parameter_sha256 = _sha256(
        {
            "model_id": X12_MODEL_ID,
            "model_version": X12_MODEL_VERSION,
            "raw_transition_parameter_sha256": (
                raw_transition_parameter_sha256
            ),
            "temperature_parameter_sha256": (
                temperature_calibration.parameter_sha256
            ),
            "probability_variant": "temperature_calibrated",
        }
    )
    transition_rows = _transition_rows(
        transition_final_evaluated,
        loaded.goals,
        loaded.dismissals,
        model=transition_dixon_coles,
        dynamic_model=dynamic_model,
        inventory_sha256_value=loaded.inventory.inventory_sha256,
        dixon_coles_parameter_sha256=(
            transition_dixon_coles_parameter_sha256
        ),
        raw_transition_parameter_sha256=(
            raw_transition_parameter_sha256
        ),
        calibrated_transition_parameter_sha256=(
            calibrated_transition_parameter_sha256
        ),
        temperature_calibration=temperature_calibration,
    )
    transition_predictions = (
        pd.DataFrame(transition_rows)
        .sort_values(
            ["prediction_at", "match_id", "period", "period_clock_ms"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    transition_split = X12TransitionSplitAudit(
        method="frozen_chronological_date_group_holdout_50_25_25",
        base_fit_first_date=_frame_timestamp(
            transition_base["match_date"].min()
        ),
        base_fit_last_date=_frame_timestamp(
            transition_base["match_date"].max()
        ),
        calibration_first_date=_frame_timestamp(
            transition_calibration["match_date"].min()
        ),
        calibration_last_date=_frame_timestamp(
            transition_calibration["match_date"].max()
        ),
        final_test_first_date=_frame_timestamp(
            transition_final_full["match_date"].min()
        ),
        final_test_last_date=_frame_timestamp(
            transition_final_full["match_date"].max()
        ),
        final_test_evaluated_first_date=_frame_timestamp(
            transition_final_evaluated["match_date"].min()
        ),
        final_test_evaluated_last_date=_frame_timestamp(
            transition_final_evaluated["match_date"].max()
        ),
        base_fit_date_count=transition_base["match_date"].nunique(),
        calibration_date_count=(
            transition_calibration["match_date"].nunique()
        ),
        final_test_date_count=(
            transition_final_full["match_date"].nunique()
        ),
        final_test_evaluated_date_count=(
            transition_final_evaluated["match_date"].nunique()
        ),
        base_fit_match_count=transition_base["match_id"].nunique(),
        calibration_match_count=(
            transition_calibration["match_id"].nunique()
        ),
        final_test_match_count=(
            transition_final_full["match_id"].nunique()
        ),
        final_test_evaluated_match_count=(
            transition_final_evaluated["match_id"].nunique()
        ),
        dynamic_fit_evaluation_cutoff=dynamic_fit_evaluation_cutoff,
        temperature_fit_evaluation_cutoff=(
            temperature_fit_evaluation_cutoff
        ),
        base_fit_max_outcome_available_at=transition_base[
            "outcome_available_at"
        ].max(),
        calibration_max_label_available_at=(
            calibration_max_label_available_at
        ),
        dixon_coles_parameter_sha256=(
            transition_dixon_coles_parameter_sha256
        ),
        dynamic_parameter_sha256=dynamic_model.parameter_sha256,
        raw_transition_parameter_sha256=(
            raw_transition_parameter_sha256
        ),
        temperature_parameter_sha256=(
            temperature_calibration.parameter_sha256
        ),
        calibrated_transition_parameter_sha256=(
            calibrated_transition_parameter_sha256
        ),
    )

    return X12DynamicTransitionEvaluation(
        experiment_id=X12_EXPERIMENT_ID,
        model_id=X12_MODEL_ID,
        model_version=X12_MODEL_VERSION,
        authorization_scope=X12_AUTHORIZATION_SCOPE,
        result_label=X12_RESULT_LABEL,
        is_formal_result=False,
        seed=X12_SEED,
        transition_predictions=transition_predictions,
        transition_split=transition_split,
        temperature_calibration=temperature_calibration,
        transition_metrics=_transition_metrics(
            transition_predictions,
            bootstrap_samples=bootstrap_samples,
            minimum_valid_bootstrap_samples=(
                minimum_valid_bootstrap_samples
            ),
            confidence_level=confidence_level,
        ),
        evaluation_match_limit=evaluation_match_limit,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
        optimizer_max_iterations=optimizer_max_iterations,
    )


def _validate_x12_reproduction_preflight(
    program_root: str | Path,
) -> dict[str, object]:
    from prediction_market.experiments import (
        ExperimentRegistryError,
        load_experiment_registry,
    )

    root = Path(program_root)
    try:
        card = load_experiment_registry(root)[X12_EXPERIMENT_ID]
    except (ExperimentRegistryError, KeyError, OSError) as error:
        raise X12DataError(
            "X-12 reproduction registration preflight failed"
        ) from error
    scopes = card.get("authorization_scopes")
    if (
        type(scopes) is not dict
        or X12_AUTHORIZATION_SCOPE not in scopes
    ):
        raise X12DataError(
            "X-12 reproduction registration is missing"
        )
    scope = scopes[X12_AUTHORIZATION_SCOPE]
    expected_binding = {
        "result_class": "poc",
        "dataset_ids": [X12_DATASET_ID],
        "model_ids": [
            "MODEL-SOCCER-DIXON-COLES",
            X12_MODEL_ID,
        ],
        "synthetic_data_sha256": None,
    }
    if (
        type(scope) is not dict
        or scope.get("authorized") is not True
        or scope.get("required_result_label") != X12_RESULT_LABEL
        or scope.get("input_binding") != expected_binding
    ):
        raise X12DataError(
            "X-12 reproduction registration binding is invalid"
        )
    preregistered_inputs = card.get("preregistered_inputs")
    registered_input = (
        preregistered_inputs.get(X12_AUTHORIZATION_SCOPE)
        if type(preregistered_inputs) is dict
        else None
    )
    if (
        type(registered_input) is not dict
        or registered_input.get("dataset_ids") != [X12_DATASET_ID]
        or registered_input.get("model_ids")
        != [
            "MODEL-SOCCER-DIXON-COLES",
            X12_MODEL_ID,
        ]
        or type(registered_input.get("registered_at")) is not str
    ):
        raise X12DataError(
            "X-12 reproduction registered inputs are missing"
        )
    code_sha256 = _validate_digest(
        registered_input.get("code_sha256"),
        "reproduction code_sha256",
    )
    data_sha256 = _validate_digest(
        registered_input.get("data_sha256"),
        "reproduction data_sha256",
    )
    required_lock_ids = scope.get("required_lock_ids")
    registration_locks = card.get("registration_locks")
    if (
        type(required_lock_ids) is not list
        or not required_lock_ids
        or type(registration_locks) is not list
    ):
        raise X12DataError(
            "X-12 reproduction registration locks are invalid"
        )
    lock_by_id = {
        str(lock.get("id")): lock
        for lock in registration_locks
        if type(lock) is dict
    }
    unresolved = [
        str(lock_id)
        for lock_id in required_lock_ids
        if (
            str(lock_id) not in lock_by_id
            or lock_by_id[str(lock_id)].get("status") != "resolved"
        )
    ]
    reproduction_lock_ids = [
        str(lock_id)
        for lock_id in required_lock_ids
        if str(lock_id).startswith("reproduction:")
    ]
    if len(reproduction_lock_ids) != 1:
        raise X12DataError(
            "X-12 reproduction registration lock is missing"
        )
    if unresolved:
        raise X12DataError(
            "X-12 reproduction registration has unresolved locks: "
            + ", ".join(unresolved)
        )
    resolved_locks = [
        {
            "id": str(lock_id),
            "evidence_ref": _validate_digest(
                lock_by_id[str(lock_id)].get("evidence_ref"),
                f"registration lock {lock_id} evidence_ref",
            ),
        }
        for lock_id in required_lock_ids
    ]
    amendments = card.get("amendments")
    if type(amendments) is not list:
        raise X12DataError(
            "X-12 reproduction amendment chain is invalid"
        )
    registration_head_sha256 = _validate_digest(
        (
            amendments[-1].get("amendment_sha256")
            if amendments
            else card.get("registration_record_sha256")
        ),
        "registration_head_sha256",
    )
    return {
        "experiment_id": X12_EXPERIMENT_ID,
        "scope": X12_AUTHORIZATION_SCOPE,
        "result_label": X12_RESULT_LABEL,
        "dataset_ids": [X12_DATASET_ID],
        "model_ids": [
            "MODEL-SOCCER-DIXON-COLES",
            X12_MODEL_ID,
        ],
        "required_lock_ids": [str(value) for value in required_lock_ids],
        "resolved_locks": resolved_locks,
        "reproduction_lock_id": reproduction_lock_ids[0],
        "reproduction_spec_sha256": lock_by_id[
            reproduction_lock_ids[0]
        ]["evidence_ref"],
        "code_sha256": code_sha256,
        "data_sha256": data_sha256,
        "registered_at": registered_input["registered_at"],
        "registration_head_sha256": registration_head_sha256,
        "status": "resolved",
    }


def _historical_x12_1x2_reference(
    program_root: str | Path,
) -> dict[str, object]:
    path = Path(program_root) / X12_HISTORICAL_1X2_RELATIVE_PATH
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X12DataError(
            "historical X-12 1X2 artifact is unavailable"
        ) from error
    file_sha256 = (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    if file_sha256 != X12_HISTORICAL_1X2_FILE_SHA256:
        raise X12DataError(
            "historical X-12 1X2 artifact file hash changed"
        )
    if (
        type(document) is not dict
        or document.get("artifact_type")
        != "x12_real_data_dixon_coles_poc_v0"
        or document.get("experiment_id") != X12_EXPERIMENT_ID
        or document.get("authorization_scope") != "poc_result"
        or document.get("result_label")
        != "POC_NO_PIT_MARKET_PRIOR"
        or document.get("evidence_sha256")
        != evidence_sha256(document)
    ):
        raise X12DataError(
            "historical X-12 1X2 artifact identity is invalid"
        )
    return {
        "path": X12_HISTORICAL_1X2_RELATIVE_PATH.as_posix(),
        "file_sha256": file_sha256,
        "evidence_sha256": document["evidence_sha256"],
        "artifact_type": document["artifact_type"],
        "result_label": document["result_label"],
        "authorization_scope": document["authorization_scope"],
        "model_id": "MODEL-SOCCER-DIXON-COLES",
        "referenced_component": "outcome_evaluation",
        "reuse_policy": "read_only_reference",
        "recomputed_for_v1": False,
        "metrics_migrated_to_v1": False,
        "historical_transition_output_reused": False,
    }


def build_x12_evidence(
    loaded: X12LoadedDataset,
    evaluation: X12DynamicTransitionEvaluation,
    *,
    program_root: str | Path,
    execution_mode: str,
) -> dict[str, object]:
    """Build a self-hashed X-12 POC artifact without upgrading its status."""

    if not isinstance(loaded, X12LoadedDataset):
        raise TypeError("loaded must be an X12LoadedDataset")
    if not isinstance(evaluation, X12DynamicTransitionEvaluation):
        raise TypeError(
            "evaluation must be an X12DynamicTransitionEvaluation"
        )
    if execution_mode not in {"bounded_smoke", "full"}:
        raise X12DataError("execution_mode must be bounded_smoke or full")
    registration_preflight = _validate_x12_reproduction_preflight(
        program_root
    )
    historical_1x2_reference = _historical_x12_1x2_reference(
        program_root
    )
    if (
        evaluation.experiment_id != X12_EXPERIMENT_ID
        or evaluation.model_id != X12_MODEL_ID
        or evaluation.model_version != X12_MODEL_VERSION
        or evaluation.authorization_scope != X12_AUTHORIZATION_SCOPE
        or evaluation.result_label != X12_RESULT_LABEL
        or evaluation.is_formal_result
    ):
        raise X12DataError("X-12 evidence cannot upgrade beyond the POC scope")
    if inventory_sha256(loaded.inventory) != loaded.inventory.inventory_sha256:
        raise X12DataError("input inventory self-hash is invalid")
    if (
        evaluation.transition_predictions.empty
        or not evaluation.transition_predictions[
            "inventory_sha256"
        ].eq(loaded.inventory.inventory_sha256).all()
    ):
        raise X12DataError(
            "transition evaluation is not bound to its inventory"
        )
    transition_documents = []
    for row in evaluation.transition_predictions.itertuples(index=False):
        probabilities = {
            label: float(getattr(row, f"probability_{label}"))
            for label in TRANSITION_CLASSES
        }
        raw_probabilities = {
            label: float(getattr(row, f"raw_probability_{label}"))
            for label in TRANSITION_CLASSES
        }
        if not math.isclose(
            sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            sum(raw_probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise X12DataError("transition evidence is not exactly normalized")
        transition_documents.append(
            {
                "match_id": int(row.match_id),
                "period": int(row.period),
                "period_clock_ms": int(row.period_clock_ms),
                "global_elapsed_ms": int(row.global_elapsed_ms),
                "prediction_at": _frame_timestamp(row.prediction_at),
                "feature_available_at": _frame_timestamp(
                    row.feature_available_at
                ),
                "horizon_seconds": int(row.horizon_seconds),
                "state": {
                    "home_score": int(row.home_score_at_cutoff),
                    "away_score": int(row.away_score_at_cutoff),
                    "home_dismissals": int(
                        row.home_dismissals_at_cutoff
                    ),
                    "away_dismissals": int(
                        row.away_dismissals_at_cutoff
                    ),
                },
                "observed_transition": str(row.observed_transition),
                "probabilities": probabilities,
                "raw_probabilities": raw_probabilities,
                "static_comparator_probabilities": {
                    label: float(
                        getattr(row, f"static_probability_{label}")
                    )
                    for label in TRANSITION_CLASSES
                },
                "contract_output": None,
                "contract_output_status": {
                    "available": False,
                    "reason": (
                        "offline_snapshot_not_bound_to_reducer_event_envelope"
                    ),
                },
                "pit_status": str(row.pit_status),
                "lineage": {
                    "inventory_sha256": row.inventory_sha256,
                    "manifest_sha256": row.manifest_sha256,
                    "object_sha256": row.object_sha256,
                    "schema_fingerprint": row.schema_fingerprint,
                    "source_state_sha256": row.source_state_sha256,
                    "source_feature_sha256": (
                        row.source_feature_sha256
                    ),
                    "dixon_coles_parameter_sha256": (
                        row.dixon_coles_parameter_sha256
                    ),
                    "raw_transition_parameter_sha256": (
                        row.raw_transition_parameter_sha256
                    ),
                    "dynamic_parameter_sha256": (
                        row.dynamic_parameter_sha256
                    ),
                    "temperature_parameter_sha256": (
                        row.temperature_parameter_sha256
                    ),
                    "calibrated_transition_parameter_sha256": (
                        row.model_parameter_sha256
                    ),
                },
            }
        )
    inventory_document = {
        **_inventory_material(loaded.inventory),
        "manifest_count": len(loaded.inventory.manifests),
        "inventory_sha256": loaded.inventory.inventory_sha256,
    }
    evidence_without_hash: dict[str, object] = {
        "artifact_type": (
            "x12_real_data_dixon_coles_dynamic_transition_poc_v1"
        ),
        "experiment_id": X12_EXPERIMENT_ID,
        "model_id": X12_MODEL_ID,
        "model_version": X12_MODEL_VERSION,
        "authorization_scope": X12_AUTHORIZATION_SCOPE,
        "result_label": X12_RESULT_LABEL,
        "execution_mode": execution_mode,
        "is_formal_result": False,
        "formal_result_eligible": False,
        "promotion_decision": "POC_ONLY",
        "registration_preflight": registration_preflight,
        "historical_1x2_reference": historical_1x2_reference,
        "input_inventory": inventory_document,
        "chronology_sha256": loaded.chronology_sha256,
        "goal_timeline_sha256": loaded.goal_timeline_sha256,
        "dismissal_timeline_sha256": (
            loaded.dismissal_timeline_sha256
        ),
        "kickoff_time_basis": KICKOFF_TIME_BASIS,
        "source_time_audit": {
            "native_index_time_regressions_observed": loaded.source_time_regressions,
            "treatment": (
                "reported_not_silently_reordered_for_PIT; period and "
                "period-local native timestamp are preserved; global elapsed "
                "ordering uses non-colliding observed-period offsets"
            ),
        },
        "market_prior": {
            "available": False,
            "imputed": False,
            "reason": "no_point_in_time_market_prior",
        },
        "model": {
            "name": "state_conditioned_dynamic_soccer_transition",
            "model_id": X12_MODEL_ID,
            "model_version": X12_MODEL_VERSION,
            "optimizer_max_iterations": evaluation.optimizer_max_iterations,
            "seed": X12_SEED,
            "transition_model": {
                "methodology": (
                    "Maia-family dynamic-covariate adaptation with frozen "
                    "Dixon-Coles base-rate offset"
                ),
                "reproduction_scope": (
                    "not a complete Maia or Cox reproduction"
                ),
                "features": [
                    "second_half",
                    "score_difference",
                    "dismissal_difference",
                ],
                "side_orientation": (
                    "shared coefficients with home/away signed covariates"
                ),
                "base_rate_offset": (
                    "single base-fit-interval Dixon-Coles goals per 90"
                ),
                "base_rate_model": {
                    "model_id": "MODEL-SOCCER-DIXON-COLES",
                    "role": "transition_offset_only",
                    "new_1x2_output_produced": False,
                    "reference_team_rule": (
                        "lowest numeric team_id attack fixed to zero"
                    ),
                    "outcome_availability_rule": (
                        "kickoff_plus_3_hours precedes calibration"
                    ),
                    "regularization": 0.001,
                    "rho_bounds": [-0.2, 0.2],
                    "optimizer": "SLSQP",
                    "optimizer_gradient": "analytic",
                    "parameter_bound_machine_roundoff_tolerance_ulps": (
                        _BOUND_ROUNDOFF_ULPS
                    ),
                    "kkt_boundary_absolute_tolerance": (
                        _KKT_BOUNDARY_ABS_TOLERANCE
                    ),
                    "optimizer_fail_closed_checks": [
                        (
                            "initial_and_final_parameter_bounds_with_8_ulp_"
                            "machine_roundoff_tolerance"
                        ),
                        "valid_likelihood_domain",
                        "finite_objective_and_gradient",
                        "objective_non_regression",
                        "objective_improvement",
                        "parameter_displacement",
                        "projected_gradient_inf_norm_lte_1e-4",
                    ],
                },
                "training_windows_per_match": len(
                    TRANSITION_SNAPSHOTS
                ),
                "window_seconds": TRANSITION_HORIZON_SECONDS,
                "l2_penalty": _DYNAMIC_INTENSITY_L2_PENALTY,
                "optimizer": "L-BFGS-B",
                "optimizer_gradient": "analytic",
                "optimizer_max_iterations": (
                    evaluation.optimizer_max_iterations
                ),
                "pit_rule": (
                    "state consumes events at or before cutoff; labels use "
                    "the first goal in (cutoff,cutoff+300s]; calibration and "
                    "final-test match ids are rejected from base fitting"
                ),
                "evaluation_protocol": (
                    "frozen chronological date-group holdout; earliest 50% "
                    "base fit, next 25% temperature calibration, final 25% test"
                ),
                "split": _json_ready(asdict(evaluation.transition_split)),
                "temperature_calibration": _json_ready(
                    asdict(evaluation.temperature_calibration)
                ),
                "output_probability_variants": {
                    "primary": "temperature_calibrated",
                    "diagnostic": "uncalibrated",
                },
            },
        },
        "transition_output": {
            "state_space": list(TRANSITION_CLASSES),
            "horizon_seconds": TRANSITION_HORIZON_SECONDS,
            "boundary_rule": (
                "current score and deduplicated dismissal state consume "
                "events at or before cutoff; first goal in "
                "(cutoff,cutoff+300s] defines the label"
            ),
            "evaluation_protocol": (
                "frozen_chronological_date_group_holdout_50_25_25"
            ),
            "primary_probability_variant": "temperature_calibrated",
            "diagnostic_probability_variant": "uncalibrated",
            "availability_status": OFFLINE_PIT_STATUS,
            "distributions": transition_documents,
            "metrics": _json_ready(evaluation.transition_metrics),
        },
        "bootstrap": {
            "seed": X12_SEED,
            "unit": "match_cluster",
            "samples_requested": evaluation.bootstrap_samples,
            "minimum_valid_samples": evaluation.minimum_valid_bootstrap_samples,
            "confidence_level": evaluation.confidence_level,
            "transition_samples_valid": evaluation.transition_metrics[
                "bootstrap_samples_valid"
            ],
        },
        "registration_locks": registration_preflight[
            "required_lock_ids"
        ],
        "open_gates": [
            "point-in-time market prior unavailable",
            "StatsBomb O-004 remains research-only",
            "offline event availability is not a live PIT feed",
            "transition snapshots lack real EventEnvelope state_event_id",
            "formal promotion unauthorized",
        ],
        "no_go_attestation": {
            "real_money_execution": False,
            "maker": False,
            "multi_venue_live_arbitrage": False,
            "live_copy_trading": False,
            "llm_hot_path": False,
            "reinforcement_learning": False,
            "unregistered_backtest": False,
            "readme_returns_as_evidence": False,
        },
    }
    evidence = _json_ready(evidence_without_hash)
    evidence["evidence_sha256"] = evidence_sha256(evidence)
    return evidence


def write_x12_evidence(
    *,
    program_root: str | Path,
    evidence: dict[str, object],
) -> Path:
    """Persist only the registered append-only dynamic-transition artifact."""

    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")
    if evidence.get("evidence_sha256") != evidence_sha256(evidence):
        raise X12DataError("evidence self-hash is invalid")
    registration_preflight = _validate_x12_reproduction_preflight(
        program_root
    )
    if evidence.get("registration_preflight") != registration_preflight:
        raise X12DataError(
            "evidence registration preflight is stale or mismatched"
        )
    historical_1x2_reference = _historical_x12_1x2_reference(
        program_root
    )
    if (
        evidence.get("historical_1x2_reference")
        != historical_1x2_reference
    ):
        raise X12DataError(
            "historical X-12 1X2 reference is stale or mismatched"
        )
    if (
        evidence.get("experiment_id") != X12_EXPERIMENT_ID
        or evidence.get("model_id") != X12_MODEL_ID
        or evidence.get("model_version") != X12_MODEL_VERSION
        or evidence.get("authorization_scope") != X12_AUTHORIZATION_SCOPE
        or evidence.get("result_label") != X12_RESULT_LABEL
    ):
        raise X12DataError(
            "evidence identity does not match the registered reproduction"
        )
    destination = (
        Path(program_root) / X12_DYNAMIC_EVIDENCE_RELATIVE_PATH
    )
    payload = _canonical_bytes(evidence) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == payload:
            return destination
        raise X12DataError("refusing to overwrite different X-12 evidence")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "DixonColesModel",
    "KICKOFF_TIME_BASIS",
    "OFFLINE_PIT_STATUS",
    "OUTCOME_CLASSES",
    "TRANSITION_CLASSES",
    "TRANSITION_HORIZON_SECONDS",
    "X12DataError",
    "X12DynamicTransitionEvaluation",
    "X12InputInventory",
    "X12LoadedDataset",
    "X12ManifestInventory",
    "X12TransitionSplitAudit",
    "X12_AUTHORIZATION_SCOPE",
    "X12_DATASET_ID",
    "X12_DYNAMIC_EVIDENCE_RELATIVE_PATH",
    "X12_EXPERIMENT_ID",
    "X12_HISTORICAL_1X2_FILE_SHA256",
    "X12_HISTORICAL_1X2_RELATIVE_PATH",
    "X12_MODEL_ID",
    "X12_MODEL_VERSION",
    "X12_RESULT_LABEL",
    "X12_SEED",
    "X12_SOURCE",
    "X12_STATSBOMB_VERSION",
    "build_x12_evidence",
    "chronology_sha256",
    "evidence_sha256",
    "goal_timeline_sha256",
    "inventory_sha256",
    "load_x12_dataset",
    "run_x12_dynamic_transition",
    "write_x12_evidence",
]
