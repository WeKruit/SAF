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

from prediction_market import contracts
from prediction_market.models.validation import (
    ValidationInputError,
    evaluate_multiclass_probabilities,
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
X12_RESULT_LABEL = "POC_NO_PIT_MARKET_PRIOR"
X12_CONTRACT_RESULT_LABEL = "PRELIMINARY"
X12_EXPERIMENT_ID = "X-12"
X12_AUTHORIZATION_SCOPE = "poc_result"

OUTCOME_CLASSES = ("home_win", "draw", "away_win")
TRANSITION_CLASSES = ("home_goal", "away_goal", "no_goal")
TRANSITION_HORIZON_SECONDS = 300
TRANSITION_SNAPSHOT_MINUTES = tuple(range(0, 90, 5))
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
    chronology_sha256: str
    goal_timeline_sha256: str
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
class X12FoldAudit:
    test_date: str
    test_min_played_at: pd.Timestamp
    test_max_played_at: pd.Timestamp
    train_max_played_at: pd.Timestamp
    train_max_outcome_available_at: pd.Timestamp
    train_match_count: int
    test_match_count: int
    optimizer_iterations: int
    optimizer_objective: float
    optimizer_initial_projected_gradient_inf_norm: float
    optimizer_objective_improvement: float
    optimizer_parameter_displacement: float
    optimizer_projected_gradient_inf_norm: float
    parameter_sha256: str


@dataclass(frozen=True, slots=True)
class X12Evaluation:
    experiment_id: str
    authorization_scope: str
    result_label: str
    contract_result_label: str
    is_formal_result: bool
    seed: int
    predictions: pd.DataFrame
    transition_predictions: pd.DataFrame
    folds: tuple[X12FoldAudit, ...]
    outcome_metrics: dict[str, object]
    transition_metrics: dict[str, object]
    minimum_train_matches: int
    evaluation_match_limit: int | None
    bootstrap_samples: int
    minimum_valid_bootstrap_samples: int
    confidence_level: float
    optimizer_max_iterations: int
    goal_grid_max: int


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
            }
        )
    return _sha256(documents)


def _canonical_goal_documents(goals: pd.DataFrame) -> list[dict[str, object]]:
    required = ("match_id", "elapsed_seconds", "event_index", "scoring_side")
    if not isinstance(goals, pd.DataFrame):
        raise X12DataError("goal timeline must be a dataframe")
    missing = sorted(set(required) - set(goals.columns))
    if missing:
        raise X12DataError(f"goal timeline columns are missing: {missing}")
    documents: list[dict[str, object]] = []
    ordered = goals.sort_values(
        ["match_id", "elapsed_seconds", "event_index"],
        kind="mergesort",
    )
    for row in ordered.itertuples(index=False):
        numeric_values = (row.match_id, row.elapsed_seconds, row.event_index)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in numeric_values
        ):
            raise X12DataError("goal timeline identities and times must be integers")
        match_id = int(row.match_id)
        elapsed_seconds = int(row.elapsed_seconds)
        event_index = int(row.event_index)
        scoring_side = str(row.scoring_side)
        if match_id <= 0:
            raise X12DataError("goal timeline match_id must be positive")
        if not 0 <= elapsed_seconds <= 130 * 60 + 59:
            raise X12DataError("goal timeline elapsed_seconds is invalid")
        if event_index <= 0:
            raise X12DataError("goal timeline event_index must be positive")
        if scoring_side not in {"home_goal", "away_goal"}:
            raise X12DataError("goal timeline scoring_side is invalid")
        documents.append(
            {
                "match_id": match_id,
                "elapsed_seconds": elapsed_seconds,
                "event_index": event_index,
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
                "elapsed_seconds": int(item["elapsed_seconds"]),
                "event_index": int(item["event_index"]),
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


def _validate_native_event_time(event: dict[str, Any]) -> int:
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
    return minute * 60 + second


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


def _adapt_events(
    payload: bytes,
    *,
    match_id: int,
    home_team_id: int,
    away_team_id: int,
    expected_home_score: int,
    expected_away_score: int,
) -> tuple[list[dict[str, object]], int]:
    try:
        inspect_statsbomb_event(payload, match_id=match_id)
    except StatsBombSourceError as error:
        raise X12DataError(f"events-{match_id} failed native validation") from error
    events = _strict_json_array(payload, context=f"events-{match_id}")
    goals: list[dict[str, object]] = []
    prior_elapsed = -1
    time_regressions = 0
    for event in events:
        event_index = event.get("index")
        if type(event_index) is not int or event_index <= 0:
            raise X12DataError("event index must be a positive integer")
        elapsed = _validate_native_event_time(event)
        if elapsed < prior_elapsed:
            time_regressions += 1
        prior_elapsed = elapsed
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
                    "elapsed_seconds": elapsed,
                    "event_index": event_index,
                    "scoring_side": scoring_side,
                }
            )
    goals.sort(key=lambda value: (value["elapsed_seconds"], value["event_index"]))
    observed = Counter(str(goal["scoring_side"]) for goal in goals)
    if (
        observed["home_goal"] != expected_home_score
        or observed["away_goal"] != expected_away_score
    ):
        raise X12DataError(
            f"events-{match_id} goal timeline does not reconcile to final score"
        )
    return goals, time_regressions


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
        adapted_goals, regressions = _adapt_events(
            verified_event.object_bytes,
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            expected_home_score=home_score,
            expected_away_score=away_score,
        )
        source_time_regressions += regressions
        goal_rows.extend(adapted_goals)
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
                "elapsed_seconds": int(goal["elapsed_seconds"]),
                "event_index": int(goal["event_index"]),
                "scoring_side": str(goal["scoring_side"]),
            }
            for goal in adapted_goals
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
        columns=("match_id", "elapsed_seconds", "event_index", "scoring_side"),
    )
    if not goals.empty:
        goals = goals.sort_values(
            ["match_id", "elapsed_seconds", "event_index"],
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
        chronology_sha256=chronology_sha256(matches),
        goal_timeline_sha256=goal_timeline_sha256(goals),
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


def _poisson_mass(rate: float, goal_grid_max: int) -> np.ndarray:
    goals = np.arange(goal_grid_max + 1, dtype=float)
    return np.exp(-rate + goals * math.log(rate) - gammaln(goals + 1))


def _outcome_probabilities(
    model: DixonColesModel,
    *,
    home_team_id: int,
    away_team_id: int,
    goal_grid_max: int,
) -> tuple[np.ndarray, float, float]:
    home_rate, away_rate = _expected_goals(
        model,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )
    matrix = np.outer(
        _poisson_mass(home_rate, goal_grid_max),
        _poisson_mass(away_rate, goal_grid_max),
    )
    matrix[0, 0] *= 1.0 - home_rate * away_rate * model.rho
    matrix[0, 1] *= 1.0 + home_rate * model.rho
    matrix[1, 0] *= 1.0 + away_rate * model.rho
    matrix[1, 1] *= 1.0 - model.rho
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise X12DataError("Dixon-Coles score grid contains invalid probability mass")
    probabilities = np.asarray(
        [
            np.tril(matrix, k=-1).sum(),
            np.trace(matrix),
            np.triu(matrix, k=1).sum(),
        ],
        dtype=float,
    )
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        raise X12DataError("Dixon-Coles outcome mass cannot be normalized")
    probabilities /= total
    probabilities[-1] = 1.0 - probabilities[0] - probabilities[1]
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not math.isclose(
            float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise X12DataError("Dixon-Coles outcome probabilities are not normalized")
    return probabilities, home_rate, away_rate


def _empirical_baseline(train: pd.DataFrame) -> np.ndarray:
    counts = train["outcome"].value_counts()
    smoothed = np.asarray(
        [float(counts.get(label, 0)) + 1.0 for label in OUTCOME_CLASSES],
        dtype=float,
    )
    probabilities = smoothed / float(smoothed.sum())
    probabilities[-1] = 1.0 - probabilities[0] - probabilities[1]
    return probabilities


def _competing_goal_probabilities(
    home_rate: float,
    away_rate: float,
) -> np.ndarray:
    total_rate = home_rate + away_rate
    if not math.isfinite(total_rate) or total_rate <= 0:
        raise X12DataError("five-minute transition intensity is invalid")
    no_goal = math.exp(
        -total_rate * TRANSITION_HORIZON_SECONDS / (90.0 * 60.0)
    )
    scoring = 1.0 - no_goal
    home_goal = scoring * home_rate / total_rate
    away_goal = 1.0 - home_goal - no_goal
    probabilities = np.asarray([home_goal, away_goal, no_goal], dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not math.isclose(
            float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise X12DataError("five-minute transition probabilities are not normalized")
    return probabilities


def _transition_rows(
    test: pd.DataFrame,
    goals: pd.DataFrame,
    *,
    model: DixonColesModel,
    inventory_sha256_value: str,
    model_parameter_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    goals_by_match = {
        int(match_id): frame.sort_values(
            ["elapsed_seconds", "event_index"],
            kind="mergesort",
        )
        for match_id, frame in goals.groupby("match_id", sort=False)
    }
    for match in test.itertuples(index=False):
        home_rate, away_rate = _expected_goals(
            model,
            home_team_id=int(match.home_team_id),
            away_team_id=int(match.away_team_id),
        )
        probabilities = _competing_goal_probabilities(home_rate, away_rate)
        match_goals = goals_by_match.get(
            int(match.match_id),
            pd.DataFrame(
                columns=("elapsed_seconds", "event_index", "scoring_side")
            ),
        )
        for snapshot_minute in TRANSITION_SNAPSHOT_MINUTES:
            start = snapshot_minute * 60
            end = start + TRANSITION_HORIZON_SECONDS
            known = match_goals.loc[match_goals["elapsed_seconds"] < start]
            future = match_goals.loc[
                (match_goals["elapsed_seconds"] >= start)
                & (match_goals["elapsed_seconds"] < end)
            ]
            observed = (
                str(future.iloc[0]["scoring_side"])
                if not future.empty
                else "no_goal"
            )
            prediction_at = match.played_at + pd.Timedelta(seconds=start)
            row: dict[str, object] = {
                "match_id": int(match.match_id),
                "snapshot_minute": snapshot_minute,
                "prediction_at": prediction_at,
                "feature_available_at": prediction_at,
                "horizon_seconds": TRANSITION_HORIZON_SECONDS,
                "home_score_at_cutoff": int(
                    (known["scoring_side"] == "home_goal").sum()
                ),
                "away_score_at_cutoff": int(
                    (known["scoring_side"] == "away_goal").sum()
                ),
                "observed_transition": observed,
                "pit_status": OFFLINE_PIT_STATUS,
                "inventory_sha256": inventory_sha256_value,
                "model_parameter_sha256": model_parameter_sha256,
                "manifest_sha256": match.event_manifest_sha256,
                "object_sha256": match.event_object_sha256,
                "schema_fingerprint": match.event_schema_fingerprint,
            }
            for index, label in enumerate(TRANSITION_CLASSES):
                row[f"probability_{label}"] = float(probabilities[index])
            rows.append(row)
    return rows


def _multiclass_point_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: tuple[str, ...],
) -> tuple[float, float]:
    class_index = {label: index for index, label in enumerate(classes)}
    try:
        encoded = np.asarray(
            [class_index[str(value)] for value in targets],
            dtype=int,
        )
    except KeyError as error:
        raise X12DataError("transition target is outside the frozen state space") from error
    indicator = np.eye(len(classes), dtype=float)[encoded]
    selected = np.clip(
        probabilities[np.arange(len(encoded)), encoded],
        1e-15,
        1.0,
    )
    return (
        float(np.mean(np.sum(np.square(probabilities - indicator), axis=1))),
        float(-np.mean(np.log(selected))),
    )


def _transition_metrics(
    transitions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, object]:
    columns = [f"probability_{label}" for label in TRANSITION_CLASSES]
    probabilities = transitions[columns].to_numpy(dtype=float)
    targets = transitions["observed_transition"].to_numpy(dtype=object)
    brier, log_loss = _multiclass_point_metrics(
        targets,
        probabilities,
        classes=TRANSITION_CLASSES,
    )
    groups = transitions["match_id"].to_numpy(dtype=object)
    unique_groups = tuple(dict.fromkeys(groups.tolist()))
    if len(unique_groups) < 2:
        raise X12DataError("transition bootstrap requires at least two matches")
    by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(X12_SEED)
    brier_samples: list[float] = []
    log_loss_samples: list[float] = []
    for _ in range(bootstrap_samples):
        selected = rng.choice(
            len(unique_groups),
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate(
            [by_group[unique_groups[int(index)]] for index in selected]
        )
        sampled = _multiclass_point_metrics(
            targets[indices],
            probabilities[indices],
            classes=TRANSITION_CLASSES,
        )
        brier_samples.append(sampled[0])
        log_loss_samples.append(sampled[1])
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "classes": TRANSITION_CLASSES,
        "target_definition": (
            "first scoring side in [cutoff, cutoff+300s); no_goal if none"
        ),
        "brier": brier,
        "brier_definition": "mean_sum_squared_class_error",
        "log_loss": log_loss,
        "bootstrap_ci": {
            "brier": (
                float(np.quantile(brier_samples, alpha)),
                float(np.quantile(brier_samples, 1.0 - alpha)),
            ),
            "log_loss": (
                float(np.quantile(log_loss_samples, alpha)),
                float(np.quantile(log_loss_samples, 1.0 - alpha)),
            ),
        },
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_valid": bootstrap_samples,
        "confidence_level": confidence_level,
        "clusters": len(unique_groups),
        "observations": len(transitions),
        "seed": X12_SEED,
    }


def _validate_run_parameters(
    *,
    minimum_train_matches: int,
    evaluation_match_limit: int | None,
    bootstrap_samples: int,
    minimum_valid_bootstrap_samples: int,
    confidence_level: float,
    optimizer_max_iterations: int,
    goal_grid_max: int,
) -> None:
    if type(minimum_train_matches) is not int or minimum_train_matches < 30:
        raise X12DataError("minimum_train_matches must be an integer >= 30")
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
    if type(goal_grid_max) is not int or goal_grid_max < 8:
        raise X12DataError("goal_grid_max must be an integer >= 8")


def run_x12_walk_forward(
    loaded: X12LoadedDataset,
    *,
    minimum_train_matches: int = 100,
    evaluation_match_limit: int | None = None,
    bootstrap_samples: int = 200,
    minimum_valid_bootstrap_samples: int = 100,
    confidence_level: float = 0.95,
    optimizer_max_iterations: int = 250,
    goal_grid_max: int = 10,
) -> X12Evaluation:
    """Run date-grouped expanding-window Dixon-Coles 1X2 evaluation."""

    _validate_run_parameters(
        minimum_train_matches=minimum_train_matches,
        evaluation_match_limit=evaluation_match_limit,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
        optimizer_max_iterations=optimizer_max_iterations,
        goal_grid_max=goal_grid_max,
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
    matches = loaded.matches.sort_values(
        ["played_at", "match_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(matches) != STATSBOMB_EXPECTED_MATCHES:
        raise X12DataError("X-12 run requires all 380 frozen matches")
    if set(matches["outcome"]) != set(OUTCOME_CLASSES):
        raise X12DataError("frozen season must contain every 1X2 outcome class")
    team_ids = tuple(
        sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    )
    if len(team_ids) != loaded.inventory.team_count:
        raise X12DataError("team inventory changed after source adaptation")

    candidate_dates: list[pd.Timestamp] = []
    for test_date in matches["match_date"].drop_duplicates():
        train = matches.loc[matches["match_date"] < test_date]
        if len(train) >= minimum_train_matches:
            candidate_dates.append(pd.Timestamp(test_date))
    selected_match_ids: set[int] | None = None
    if evaluation_match_limit is not None:
        candidates = matches.loc[matches["match_date"].isin(candidate_dates)].head(
            evaluation_match_limit
        )
        selected_match_ids = set(candidates["match_id"].astype(int))
        candidate_dates = [
            pd.Timestamp(value)
            for value in candidates["match_date"].drop_duplicates()
        ]
    if not candidate_dates:
        raise X12DataError("no expanding-window evaluation dates are available")

    prediction_frames: list[pd.DataFrame] = []
    transition_rows: list[dict[str, object]] = []
    fold_audits: list[X12FoldAudit] = []
    warm_start: tuple[float, ...] | None = None
    for test_date in candidate_dates:
        train = matches.loc[matches["match_date"] < test_date].copy()
        test = matches.loc[matches["match_date"] == test_date].copy()
        if selected_match_ids is not None:
            test = test.loc[test["match_id"].isin(selected_match_ids)].copy()
        if test.empty:
            continue
        if len(train) < minimum_train_matches:
            raise X12DataError("expanding fold has insufficient prior matches")
        if not train["played_at"].max() < test["played_at"].min():
            raise X12DataError("training kickoff is not strictly before test kickoff")
        if not train["outcome_available_at"].max() < test["played_at"].min():
            raise X12DataError(
                "training outcome availability is not strictly before prediction"
            )
        model = _fit_dixon_coles(
            train,
            team_ids=team_ids,
            optimizer_max_iterations=optimizer_max_iterations,
            initial_parameters=warm_start,
        )
        warm_start = model.parameters
        model_parameter_sha256 = _sha256(list(model.parameters))
        baseline = _empirical_baseline(train)
        feature_available_at = train["outcome_available_at"].max()
        for class_index, label in enumerate(OUTCOME_CLASSES):
            test[f"baseline_probability_{label}"] = float(baseline[class_index])
        home_rates: list[float] = []
        away_rates: list[float] = []
        outcome_matrices: list[np.ndarray] = []
        for row in test.itertuples(index=False):
            probabilities, home_rate, away_rate = _outcome_probabilities(
                model,
                home_team_id=int(row.home_team_id),
                away_team_id=int(row.away_team_id),
                goal_grid_max=goal_grid_max,
            )
            outcome_matrices.append(probabilities)
            home_rates.append(home_rate)
            away_rates.append(away_rate)
        probability_matrix = np.vstack(outcome_matrices)
        for class_index, label in enumerate(OUTCOME_CLASSES):
            test[f"probability_{label}"] = probability_matrix[:, class_index]
        test["expected_home_goals"] = home_rates
        test["expected_away_goals"] = away_rates
        test["prediction_at"] = test["played_at"]
        test["model_feature_available_at"] = feature_available_at
        test["baseline_available_at"] = feature_available_at
        test["inventory_sha256"] = loaded.inventory.inventory_sha256
        test["result_label"] = X12_RESULT_LABEL
        prediction_frames.append(test)
        transition_rows.extend(
            _transition_rows(
                test,
                loaded.goals,
                model=model,
                inventory_sha256_value=loaded.inventory.inventory_sha256,
                model_parameter_sha256=model_parameter_sha256,
            )
        )
        fold_audits.append(
            X12FoldAudit(
                test_date=_frame_timestamp(test_date),
                test_min_played_at=test["played_at"].min(),
                test_max_played_at=test["played_at"].max(),
                train_max_played_at=train["played_at"].max(),
                train_max_outcome_available_at=train[
                    "outcome_available_at"
                ].max(),
                train_match_count=len(train),
                test_match_count=len(test),
                optimizer_iterations=model.iterations,
                optimizer_objective=model.objective,
                optimizer_initial_projected_gradient_inf_norm=(
                    model.initial_projected_gradient_inf_norm
                ),
                optimizer_objective_improvement=model.objective_improvement,
                optimizer_parameter_displacement=model.parameter_displacement,
                optimizer_projected_gradient_inf_norm=(
                    model.projected_gradient_inf_norm
                ),
                parameter_sha256=model_parameter_sha256,
            )
        )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["played_at", "match_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if predictions["match_id"].nunique() < 20:
        raise X12DataError("bounded X-12 evaluation requires at least 20 matches")
    transition_predictions = (
        pd.DataFrame(transition_rows)
        .sort_values(["prediction_at", "match_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    outcome_columns = [f"probability_{label}" for label in OUTCOME_CLASSES]
    baseline_columns = [
        f"baseline_probability_{label}" for label in OUTCOME_CLASSES
    ]
    try:
        outcome_metrics = evaluate_multiclass_probabilities(
            predictions["outcome"].to_numpy(dtype=object),
            predictions[outcome_columns].to_numpy(dtype=float),
            classes=OUTCOME_CLASSES,
            groups=predictions["match_id"].to_numpy(dtype=object),
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            minimum_valid_samples=minimum_valid_bootstrap_samples,
            seed=X12_SEED,
            prediction_at=predictions["prediction_at"],
            feature_available_at=predictions["model_feature_available_at"],
            prior_probabilities=predictions[baseline_columns].to_numpy(dtype=float),
            prior_available_at=predictions["baseline_available_at"],
        )
    except ValidationInputError as error:
        raise X12DataError("X-12 multiclass evaluation failed closed") from error
    comparison = dict(outcome_metrics.pop("prior_comparison"))
    comparison["baseline_metrics"] = comparison.pop("prior_metrics")
    comparison["delta_definition"] = "model_minus_simple_baseline"
    comparison["comparator"] = "expanding_empirical_1x2_laplace_alpha_1"
    outcome_metrics["simple_baseline_comparison"] = comparison
    outcome_metrics["market_prior"] = {
        "available": False,
        "reason": "no_point_in_time_market_prior",
    }

    return X12Evaluation(
        experiment_id=X12_EXPERIMENT_ID,
        authorization_scope=X12_AUTHORIZATION_SCOPE,
        result_label=X12_RESULT_LABEL,
        contract_result_label=X12_CONTRACT_RESULT_LABEL,
        is_formal_result=False,
        seed=X12_SEED,
        predictions=predictions,
        transition_predictions=transition_predictions,
        folds=tuple(fold_audits),
        outcome_metrics=outcome_metrics,
        transition_metrics=_transition_metrics(
            transition_predictions,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        ),
        minimum_train_matches=minimum_train_matches,
        evaluation_match_limit=evaluation_match_limit,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
        optimizer_max_iterations=optimizer_max_iterations,
        goal_grid_max=goal_grid_max,
    )


_REGISTRATION_LOCK_IDS = (
    "statsbomb_manifest_and_version",
    "statsbomb_research_license",
    "pit_feature_contract",
    "dixon_coles_config_and_seed",
    "expanding_window_split",
    "transition_definition",
    "h_split_approval",
)


def _fixed_point_probabilities(
    probabilities: dict[str, float],
) -> dict[str, dict[str, object]]:
    scale = 18
    denominator = 10**scale
    atoms: dict[str, int] = {}
    remaining = denominator
    for label in TRANSITION_CLASSES[:-1]:
        value = int(round(probabilities[label] * denominator))
        if value < 0 or value > remaining:
            raise X12DataError("fixed-point transition probability is invalid")
        atoms[label] = value
        remaining -= value
    atoms[TRANSITION_CLASSES[-1]] = remaining
    if any(value < 0 for value in atoms.values()) or sum(atoms.values()) != denominator:
        raise X12DataError("fixed-point transition probabilities are not exact")
    return {
        label: {"atoms": str(atoms[label]), "scale": scale}
        for label in TRANSITION_CLASSES
    }


def _transition_contract_output(
    row: Any,
    *,
    evaluation: X12Evaluation,
    program_root: str | Path,
) -> dict[str, object]:
    probabilities = {
        label: float(getattr(row, f"probability_{label}"))
        for label in TRANSITION_CLASSES
    }
    cutoff = _frame_timestamp(row.prediction_at)
    feature_material = {
        "match_id": int(row.match_id),
        "pit_cutoff_at": cutoff,
        "home_score": int(row.home_score_at_cutoff),
        "away_score": int(row.away_score_at_cutoff),
        "event_object_sha256": row.object_sha256,
        "offline_pit_status": row.pit_status,
    }
    config_sha256 = _sha256(
        {
            "model_id": "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
            "model_version": "v1",
            "parameter_sha256": row.model_parameter_sha256,
            "transition_horizon_seconds": TRANSITION_HORIZON_SECONDS,
            "transition_boundary": "[cutoff,cutoff+300s)",
            "rate_basis": "Dixon-Coles pregame expected goals competing hazards",
            "seed": evaluation.seed,
        }
    )
    event_digest = hashlib.sha256(
        _canonical_bytes(
            {
                "experiment_id": X12_EXPERIMENT_ID,
                "match_id": int(row.match_id),
                "pit_cutoff_at": cutoff,
                "horizon_seconds": TRANSITION_HORIZON_SECONDS,
                "data_sha256": row.inventory_sha256,
                "config_sha256": config_sha256,
            }
        )
    ).hexdigest()
    document: dict[str, object] = {
        "contract_version": "v1",
        "model_id": "MODEL-SOCCER-FIVE-MINUTE-TRANSITION",
        "model_version": "v1",
        "experiment_id": X12_EXPERIMENT_ID,
        "run_id": (
            "run_x12_"
            + str(row.inventory_sha256).removeprefix(_HASH_PREFIX)[:16]
            + "_"
            + config_sha256.removeprefix(_HASH_PREFIX)[:16]
        ),
        "game_id": f"game_{int(row.match_id)}",
        "state_event_id": f"evt_{event_digest}",
        "pit_cutoff_at": cutoff,
        "output_kind": "state_transition",
        "transition_unit": "five_minute_interval",
        "state_space": list(TRANSITION_CLASSES),
        "horizon": "next_state_transition",
        "probabilities": _fixed_point_probabilities(probabilities),
        "feature_sha256": _sha256(feature_material),
        "data_sha256": row.inventory_sha256,
        "config_sha256": config_sha256,
        "quality_flags": [
            "preliminary_rules",
            "source_clock_unverified",
        ],
    }
    try:
        validated = contracts.validate_contract_v1(
            program_root,
            "model-output/v1.schema.yaml",
            document,
        )
    except ValueError as error:
        raise X12DataError(
            "five-minute transition failed model-output v1 validation"
        ) from error
    if not isinstance(validated, contracts.ModelOutputV1):
        raise X12DataError(
            "five-minute transition validator returned an invalid contract type"
        )
    return validated.model_dump(mode="json")


def build_x12_evidence(
    loaded: X12LoadedDataset,
    evaluation: X12Evaluation,
    *,
    program_root: str | Path,
    execution_mode: str,
) -> dict[str, object]:
    """Build a self-hashed X-12 POC artifact without upgrading its status."""

    if not isinstance(loaded, X12LoadedDataset):
        raise TypeError("loaded must be an X12LoadedDataset")
    if not isinstance(evaluation, X12Evaluation):
        raise TypeError("evaluation must be an X12Evaluation")
    if execution_mode not in {"bounded_smoke", "full"}:
        raise X12DataError("execution_mode must be bounded_smoke or full")
    if (
        evaluation.experiment_id != X12_EXPERIMENT_ID
        or evaluation.authorization_scope != X12_AUTHORIZATION_SCOPE
        or evaluation.result_label != X12_RESULT_LABEL
        or evaluation.contract_result_label != X12_CONTRACT_RESULT_LABEL
        or evaluation.is_formal_result
    ):
        raise X12DataError("X-12 evidence cannot upgrade beyond the POC scope")
    if inventory_sha256(loaded.inventory) != loaded.inventory.inventory_sha256:
        raise X12DataError("input inventory self-hash is invalid")
    for frame in (
        evaluation.predictions,
        evaluation.transition_predictions,
    ):
        if (
            frame.empty
            or not frame["inventory_sha256"]
            .eq(loaded.inventory.inventory_sha256)
            .all()
        ):
            raise X12DataError("evaluation output is not bound to its inventory")

    prediction_documents = []
    for row in evaluation.predictions.itertuples(index=False):
        prediction_documents.append(
            {
                "match_id": int(row.match_id),
                "prediction_at": _frame_timestamp(row.prediction_at),
                "observed_outcome": str(row.outcome),
                "probabilities": {
                    label: float(getattr(row, f"probability_{label}"))
                    for label in OUTCOME_CLASSES
                },
                "simple_baseline_probabilities": {
                    label: float(
                        getattr(row, f"baseline_probability_{label}")
                    )
                    for label in OUTCOME_CLASSES
                },
                "expected_goals": {
                    "home": float(row.expected_home_goals),
                    "away": float(row.expected_away_goals),
                },
                "lineage": {
                    "inventory_sha256": row.inventory_sha256,
                    "manifest_sha256": row.event_manifest_sha256,
                    "object_sha256": row.event_object_sha256,
                    "schema_fingerprint": row.event_schema_fingerprint,
                },
            }
        )
    transition_documents = []
    for row in evaluation.transition_predictions.itertuples(index=False):
        probabilities = {
            label: float(getattr(row, f"probability_{label}"))
            for label in TRANSITION_CLASSES
        }
        if not math.isclose(
            sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise X12DataError("transition evidence is not exactly normalized")
        transition_documents.append(
            {
                "match_id": int(row.match_id),
                "prediction_at": _frame_timestamp(row.prediction_at),
                "feature_available_at": _frame_timestamp(
                    row.feature_available_at
                ),
                "horizon_seconds": int(row.horizon_seconds),
                "state": {
                    "home_score": int(row.home_score_at_cutoff),
                    "away_score": int(row.away_score_at_cutoff),
                },
                "observed_transition": str(row.observed_transition),
                "probabilities": probabilities,
                "contract_output": _transition_contract_output(
                    row,
                    evaluation=evaluation,
                    program_root=program_root,
                ),
                "pit_status": str(row.pit_status),
                "lineage": {
                    "inventory_sha256": row.inventory_sha256,
                    "manifest_sha256": row.manifest_sha256,
                    "object_sha256": row.object_sha256,
                    "schema_fingerprint": row.schema_fingerprint,
                },
            }
        )
    inventory_document = {
        **_inventory_material(loaded.inventory),
        "manifest_count": len(loaded.inventory.manifests),
        "inventory_sha256": loaded.inventory.inventory_sha256,
    }
    evidence_without_hash: dict[str, object] = {
        "artifact_type": "x12_real_data_dixon_coles_poc_v0",
        "experiment_id": X12_EXPERIMENT_ID,
        "authorization_scope": X12_AUTHORIZATION_SCOPE,
        "result_label": X12_RESULT_LABEL,
        "contract_result_label": X12_CONTRACT_RESULT_LABEL,
        "execution_mode": execution_mode,
        "is_formal_result": False,
        "formal_result_eligible": False,
        "promotion_decision": "POC_ONLY",
        "input_inventory": inventory_document,
        "chronology_sha256": loaded.chronology_sha256,
        "goal_timeline_sha256": loaded.goal_timeline_sha256,
        "kickoff_time_basis": KICKOFF_TIME_BASIS,
        "source_time_audit": {
            "native_index_time_regressions_observed": loaded.source_time_regressions,
            "treatment": (
                "reported_not_silently_reordered_for_PIT; goal labels sort by "
                "native minute,second,index; result remains offline POC"
            ),
        },
        "market_prior": {
            "available": False,
            "imputed": False,
            "reason": "no_point_in_time_market_prior",
        },
        "model": {
            "name": "low_score_adjusted_dixon_coles",
            "training_rule": (
                "all complete matches on calendar dates strictly before test date"
            ),
            "outcome_availability_rule": (
                "conservative kickoff_plus_3_hours must precede prediction"
            ),
            "reference_team_rule": "lowest numeric team_id attack fixed to zero",
            "regularization": 0.001,
            "rho_bounds": [-0.2, 0.2],
            "parameter_bound_machine_roundoff_tolerance_ulps": (
                _BOUND_ROUNDOFF_ULPS
            ),
            "kkt_boundary_absolute_tolerance": (
                _KKT_BOUNDARY_ABS_TOLERANCE
            ),
            "goal_grid_max": evaluation.goal_grid_max,
            "optimizer": "SLSQP",
            "optimizer_gradient": "analytic",
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
            "optimizer_max_iterations": evaluation.optimizer_max_iterations,
            "seed": X12_SEED,
        },
        "walk_forward": {
            "minimum_train_matches": evaluation.minimum_train_matches,
            "evaluation_match_limit": evaluation.evaluation_match_limit,
            "folds": [_json_ready(asdict(fold)) for fold in evaluation.folds],
        },
        "outcome_evaluation": {
            "classes": list(OUTCOME_CLASSES),
            "predictions": prediction_documents,
            "metrics": _json_ready(evaluation.outcome_metrics),
        },
        "transition_output": {
            "state_space": list(TRANSITION_CLASSES),
            "horizon_seconds": TRANSITION_HORIZON_SECONDS,
            "boundary_rule": (
                "first goal in [cutoff,cutoff+300s); current state uses goals "
                "strictly before cutoff"
            ),
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
            "outcome_samples_valid": evaluation.outcome_metrics[
                "bootstrap_samples_valid"
            ],
            "transition_samples_valid": evaluation.transition_metrics[
                "bootstrap_samples_valid"
            ],
        },
        "registration_locks": [
            {
                "id": lock_id,
                "status": "registry_unresolved",
            }
            for lock_id in _REGISTRATION_LOCK_IDS
        ],
        "open_gates": [
            "Team H lock approval",
            "point-in-time market prior unavailable",
            "StatsBomb O-004 remains research-only",
            "offline event availability is not a live PIT feed",
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


def write_x12_evidence(path: str | Path, evidence: dict[str, object]) -> None:
    """Persist canonical evidence without silently replacing different bytes."""

    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")
    if evidence.get("evidence_sha256") != evidence_sha256(evidence):
        raise X12DataError("evidence self-hash is invalid")
    destination = Path(path)
    payload = _canonical_bytes(evidence) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == payload:
            return
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


__all__ = [
    "DixonColesModel",
    "KICKOFF_TIME_BASIS",
    "OFFLINE_PIT_STATUS",
    "OUTCOME_CLASSES",
    "TRANSITION_CLASSES",
    "TRANSITION_HORIZON_SECONDS",
    "X12DataError",
    "X12Evaluation",
    "X12FoldAudit",
    "X12InputInventory",
    "X12LoadedDataset",
    "X12ManifestInventory",
    "X12_AUTHORIZATION_SCOPE",
    "X12_CONTRACT_RESULT_LABEL",
    "X12_DATASET_ID",
    "X12_EXPERIMENT_ID",
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
    "run_x12_walk_forward",
    "write_x12_evidence",
]
