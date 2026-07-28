"""Publish the reaction-closed NFL Factor Lab V2 expansion registry.

The builder reads only the published reaction-blind candidate scan, its
content-addressed temporal audit, and the two identity-only legacy cohort
locks.  It never opens a market-reaction artifact and emits no market value or
direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SCAN = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/candidate-coverage"
    / "nfl_2025_dual_venue_candidate_scan_v0.json"
)
TEMPORAL_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/candidate-coverage"
    / "temporal-all-dual-v1/manifests"
    / "6549c5003b6419fc422dbd27fb6a2663ae22f8bc9f0584274c3321a47666e2ac"
    ".manifest.json"
)
LEGACY_RUN_INDEX = (
    PROJECT_ROOT / "registries/factors/nfl_factor_lab_run_index_v2.json"
)
LEGACY_HOLDOUT_SELECTION = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/historical-holdout"
    / "sha256-6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
    / "6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
    ".holdout-selection.json"
)

EXPECTED_CANDIDATE_SCAN_SHA256 = (
    "sha256:9fb96bfa10f641f410eda001803aa28dc3908ee5f957f196800b9e8adcddeaef"
)
EXPECTED_TEMPORAL_MANIFEST_SHA256 = (
    "sha256:6549c5003b6419fc422dbd27fb6a2663ae22f8bc9f0584274c3321a47666e2ac"
)
EXPECTED_TEMPORAL_OBJECT_SHA256 = (
    "sha256:432277924ccc8249333b48b96060662cf630bf1a72616e79d9e186bdb2de1588"
)
EXPECTED_LEGACY_HOLDOUT_SELECTION_ID = (
    "sha256:6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
)

FACTOR_REGISTRY_VERSION = "v2"
HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60)
TIME_BUCKET_DEFINITIONS = (
    {
        "label": "Q1",
        "predicate": {
            "all": [{"field": "quarter", "operator": "EQ", "value": 1}]
        },
    },
    {
        "label": "Q2_PRE_2M",
        "predicate": {
            "all": [
                {"field": "quarter", "operator": "EQ", "value": 2},
                {
                    "field": "quarter_seconds_remaining",
                    "operator": "GT",
                    "value": 120,
                },
            ]
        },
    },
    {
        "label": "Q2_FINAL_2M",
        "predicate": {
            "all": [
                {"field": "quarter", "operator": "EQ", "value": 2},
                {
                    "field": "quarter_seconds_remaining",
                    "operator": "GTE",
                    "value": 0,
                },
                {
                    "field": "quarter_seconds_remaining",
                    "operator": "LTE",
                    "value": 120,
                },
            ]
        },
    },
    {
        "label": "Q3",
        "predicate": {
            "all": [{"field": "quarter", "operator": "EQ", "value": 3}]
        },
    },
    {
        "label": "Q4_PRE_5M",
        "predicate": {
            "all": [
                {"field": "quarter", "operator": "EQ", "value": 4},
                {
                    "field": "quarter_seconds_remaining",
                    "operator": "GT",
                    "value": 300,
                },
            ]
        },
    },
    {
        "label": "Q4_FINAL_5M",
        "predicate": {
            "all": [
                {"field": "quarter", "operator": "EQ", "value": 4},
                {
                    "field": "quarter_seconds_remaining",
                    "operator": "GTE",
                    "value": 0,
                },
                {
                    "field": "quarter_seconds_remaining",
                    "operator": "LTE",
                    "value": 300,
                },
            ]
        },
    },
    {
        "label": "OT",
        "predicate": {
            "all": [{"field": "quarter", "operator": "GTE", "value": 5}]
        },
    },
)
EXACT_TIME_FIELDS = (
    "source_time",
    "quarter",
    "game_clock",
    "quarter_seconds_remaining",
    "game_seconds_remaining",
)
TIME_ELIGIBILITY = {
    "required_fields": [
        "quarter",
        "game_clock",
        "quarter_seconds_remaining",
    ],
    "quarter": {
        "type": "INTEGER",
        "minimum": 1,
        "minimum_inclusive": True,
    },
    "game_clock": {"type": "NON_EMPTY_STRING"},
    "quarter_seconds_remaining": {
        "type": "FINITE_NUMBER",
        "minimum": 0,
        "maximum": 900,
        "bounds_inclusive": True,
    },
    "otherwise": "NOT_ELIGIBLE",
}
PROBABILITY_DISPLAY_BIN_DEFINITIONS = (
    {
        "label": "LOW",
        "predicate": {
            "field": "pre_event_probability",
            "minimum": 0,
            "minimum_inclusive": True,
            "maximum": 0.33,
            "maximum_inclusive": False,
        },
    },
    {
        "label": "MID",
        "predicate": {
            "field": "pre_event_probability",
            "minimum": 0.33,
            "minimum_inclusive": True,
            "maximum": 0.67,
            "maximum_inclusive": True,
        },
    },
    {
        "label": "HIGH",
        "predicate": {
            "field": "pre_event_probability",
            "minimum": 0.67,
            "minimum_inclusive": False,
            "maximum": 1,
            "maximum_inclusive": True,
        },
    },
)
PROBABILITY_DISPLAY_ELIGIBILITY = {
    "field": "pre_event_probability",
    "type": "FINITE_NUMBER",
    "minimum": 0,
    "maximum": 1,
    "bounds_inclusive": True,
    "otherwise": "NOT_ELIGIBLE",
}
DEVELOPMENT_WEEKS = tuple(range(1, 13))
FINAL_HOLDOUT_WEEKS = tuple(range(13, 19))
DEVELOPMENT_GAME_COUNT = 153
FINAL_HOLDOUT_GAME_COUNT = 81
EXPANSION_GAME_COUNT = 234
LEGACY_GAME_COUNT = 40
FOLD_ASSIGNMENT_SEED = 20260726

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"2025_(0[1-9]|1[0-9]|20)_[A-Z]+_[A-Z]+\Z")
_FORBIDDEN_INPUT_KEY_TOKENS = frozenset(
    {
        "ask",
        "bid",
        "buy",
        "change",
        "delta",
        "direction",
        "movement",
        "price",
        "reaction",
        "score",
        "sell",
        "side",
        "size",
        "volume",
        "vwap",
    }
)


class ExpansionGovernanceError(RuntimeError):
    """The expansion governance contract failed closed."""


def _classify_time_bucket(
    quarter: object,
    game_clock: object,
    quarter_seconds_remaining: object,
) -> str:
    if (
        type(quarter) is not int
        or quarter < 1
        or type(game_clock) is not str
        or not game_clock
        or type(quarter_seconds_remaining) not in (int, float)
        or not math.isfinite(quarter_seconds_remaining)
        or quarter_seconds_remaining < 0
        or quarter_seconds_remaining > 900
    ):
        return "NOT_ELIGIBLE"
    if quarter == 1:
        return "Q1"
    if quarter == 2:
        return (
            "Q2_PRE_2M"
            if quarter_seconds_remaining > 120
            else "Q2_FINAL_2M"
        )
    if quarter == 3:
        return "Q3"
    if quarter == 4:
        return (
            "Q4_PRE_5M"
            if quarter_seconds_remaining > 300
            else "Q4_FINAL_5M"
        )
    return "OT"


def _classify_probability_display_bin(probability: object) -> str:
    if (
        type(probability) not in (int, float)
        or not math.isfinite(probability)
        or probability < 0
        or probability > 1
    ):
        return "NOT_ELIGIBLE"
    if probability < 0.33:
        return "LOW"
    if probability <= 0.67:
        return "MID"
    return "HIGH"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExpansionGovernanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> None:
    raise ExpansionGovernanceError(f"non-JSON numeric literal: {value}")


def _load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExpansionGovernanceError(f"{label} is not strict JSON") from error
    if type(value) is not dict:
        raise ExpansionGovernanceError(f"{label} must be a JSON object")
    return value


def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        before = os.stat(resolved, follow_symlinks=False)
        if resolved.is_symlink() or not resolved.is_file():
            raise OSError("not a regular file")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise OSError("byte length outside bounded input limit")
        with resolved.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        after = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise ExpansionGovernanceError(
            f"{label} is not a bounded stable regular file: {resolved}"
        ) from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ExpansionGovernanceError(f"{label} changed while being read")
    return payload


def _repo_path(path: Path) -> str:
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ExpansionGovernanceError(
            f"input path escapes the project root: {resolved}"
        ) from error


def _source_binding(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": _repo_path(path),
        "sha256": _sha256(payload),
        "byte_length": len(payload),
    }


def _reject_market_value_keys(value: object, *, label: str, path: str = "$") -> None:
    if type(value) is dict:
        for raw_key, child in value.items():
            if type(raw_key) is not str:
                raise ExpansionGovernanceError(
                    f"{label} has a non-string key at {path}"
                )
            tokens = set(filter(None, re.split(r"[^a-z0-9]+", raw_key.lower())))
            forbidden = sorted(tokens & _FORBIDDEN_INPUT_KEY_TOKENS)
            if forbidden:
                raise ExpansionGovernanceError(
                    f"{label} exposes prohibited market-value key "
                    f"{path}.{raw_key}: {forbidden}"
                )
            _reject_market_value_keys(
                child,
                label=label,
                path=f"{path}.{raw_key}",
            )
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_market_value_keys(
                child,
                label=label,
                path=f"{path}[{index}]",
            )


def _require_game_map(
    rows: object,
    *,
    label: str,
    expected_count: int,
) -> dict[str, int]:
    if type(rows) is not list or len(rows) != expected_count:
        raise ExpansionGovernanceError(
            f"{label} must contain exactly {expected_count} games"
        )
    games: dict[str, int] = {}
    for row in rows:
        if type(row) is not dict:
            raise ExpansionGovernanceError(f"{label} game row is malformed")
        game_id = row.get("game_id")
        week = row.get("week")
        if (
            type(game_id) is not str
            or _GAME_ID_RE.fullmatch(game_id) is None
            or type(week) is not int
            or int(game_id.split("_")[1]) != week
            or game_id in games
        ):
            raise ExpansionGovernanceError(
                f"{label} has an invalid or duplicate game identity"
            )
        games[game_id] = week
    return games


def _verify_candidate_scan(
    path: Path,
) -> tuple[dict[str, int], bytes, dict[str, object]]:
    payload = _read_regular(
        path,
        label="candidate scan",
        max_bytes=2 * 1024 * 1024,
    )
    digest = _sha256(payload)
    if digest != EXPECTED_CANDIDATE_SCAN_SHA256:
        raise ExpansionGovernanceError("candidate scan hash is not authoritative")
    scan = _load_json(payload, label="candidate scan")
    _reject_market_value_keys(scan, label="candidate scan")
    if (
        scan.get("schema") != "nfl_2025_dual_venue_metadata_candidates_v0"
        or scan.get("selection_basis")
        != "metadata_only_no_score_price_volume_or_reaction"
        or scan.get("seed") != FOLD_ASSIGNMENT_SEED
        or scan.get("dual_venue_metadata_count") != EXPANSION_GAME_COUNT
    ):
        raise ExpansionGovernanceError("candidate scan is not the frozen blind scan")
    all_games = scan.get("games")
    if type(all_games) is not list:
        raise ExpansionGovernanceError("candidate scan games are missing")
    selected_rows = [
        row
        for row in all_games
        if type(row) is dict and row.get("dual_venue_metadata") is True
    ]
    games = _require_game_map(
        selected_rows,
        label="candidate scan dual-venue cohort",
        expected_count=EXPANSION_GAME_COUNT,
    )
    return games, payload, _source_binding(path, payload)


def _verify_temporal_audit(
    manifest_path: Path,
    *,
    candidate_scan_sha256: str,
) -> tuple[dict[str, int], bytes, bytes, dict[str, object], dict[str, object]]:
    manifest_bytes = _read_regular(
        manifest_path,
        label="temporal audit manifest",
        max_bytes=64 * 1024,
    )
    if _sha256(manifest_bytes) != EXPECTED_TEMPORAL_MANIFEST_SHA256:
        raise ExpansionGovernanceError("temporal manifest hash is not authoritative")
    manifest = _load_json(manifest_bytes, label="temporal audit manifest")
    if (
        manifest.get("schema")
        != "nfl_2025_candidate_temporal_coverage_manifest_v1"
        or manifest.get("scope") != "all-dual"
        or manifest.get("candidate_game_count") != EXPANSION_GAME_COUNT
        or manifest.get("object_sha256") != EXPECTED_TEMPORAL_OBJECT_SHA256
        or type(manifest.get("object_path")) is not str
        or type(manifest.get("byte_length")) is not int
    ):
        raise ExpansionGovernanceError("temporal audit manifest is malformed")
    object_path = PROJECT_ROOT / manifest["object_path"]
    object_bytes = _read_regular(
        object_path,
        label="temporal audit object",
        max_bytes=2 * 1024 * 1024,
    )
    if (
        len(object_bytes) != manifest["byte_length"]
        or _sha256(object_bytes) != EXPECTED_TEMPORAL_OBJECT_SHA256
    ):
        raise ExpansionGovernanceError("temporal audit object binding failed")
    audit = _load_json(object_bytes, label="temporal audit object")
    _reject_market_value_keys(audit, label="temporal audit object")
    if (
        audit.get("schema") != "nfl_2025_candidate_temporal_coverage_v1"
        or audit.get("builder_version") != "v1-mixed-iso-source-time"
        or audit.get("scope") != "all-dual"
        or audit.get("selection_basis")
        != (
            "source_timestamp_and_trade_presence_only;"
            "no_score_price_size_volume_or_direction_retained"
        )
        or audit.get("candidate_scan_sha256") != candidate_scan_sha256
        or audit.get("horizons_seconds") != list(HORIZONS_SECONDS)
        or audit.get("candidate_game_count") != EXPANSION_GAME_COUNT
    ):
        raise ExpansionGovernanceError("temporal audit is outside frozen scope")
    rows = audit.get("games")
    games = _require_game_map(
        rows,
        label="temporal audit",
        expected_count=EXPANSION_GAME_COUNT,
    )
    expected_horizons = {str(value) for value in HORIZONS_SECONDS}
    assert type(rows) is list
    for row in rows:
        assert type(row) is dict
        coverage = row.get("horizon_coverage")
        if type(coverage) is not dict or set(coverage) != expected_horizons:
            raise ExpansionGovernanceError(
                "temporal audit does not cover every formal horizon"
            )
    return (
        games,
        manifest_bytes,
        object_bytes,
        _source_binding(manifest_path, manifest_bytes),
        _source_binding(object_path, object_bytes),
    )


def _verify_legacy_cohort(
    run_index_path: Path,
    holdout_path: Path,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, object],
    dict[str, object],
]:
    run_index_bytes = _read_regular(
        run_index_path,
        label="legacy V2 run index",
        max_bytes=1024 * 1024,
    )
    run_index = _load_json(run_index_bytes, label="legacy V2 run index")
    game_ids = run_index.get("game_ids")
    if (
        run_index.get("schema") != "NFLFactorLabRunIndexV2"
        or run_index.get("holdout_reaction_accessed") is not False
        or type(game_ids) is not list
        or len(game_ids) != 20
        or len(set(game_ids)) != 20
        or any(
            type(game_id) is not str
            or _GAME_ID_RE.fullmatch(game_id) is None
            for game_id in game_ids
        )
    ):
        raise ExpansionGovernanceError("legacy V2 run index is malformed")
    discovery_ids = tuple(sorted(game_ids))

    holdout_bytes = _read_regular(
        holdout_path,
        label="legacy holdout selection",
        max_bytes=128 * 1024,
    )
    holdout = _load_json(holdout_bytes, label="legacy holdout selection")
    selected = holdout.get("selected_game_ids")
    boundary = holdout.get("claim_boundary")
    if (
        holdout.get("artifact_type")
        != "nfl_x13_historical_holdout_selection"
        or holdout.get("selection_id")
        != EXPECTED_LEGACY_HOLDOUT_SELECTION_ID
        or holdout.get("status") != "FROZEN"
        or type(selected) is not list
        or len(selected) != 20
        or len(set(selected)) != 20
        or type(boundary) is not dict
        or boundary.get("reaction_used") is not False
        or boundary.get("trade_price_used") is not False
        or any(
            type(game_id) is not str
            or _GAME_ID_RE.fullmatch(game_id) is None
            for game_id in selected
        )
    ):
        raise ExpansionGovernanceError("legacy holdout selection is malformed")
    holdout_ids = tuple(sorted(selected))
    if set(discovery_ids) & set(holdout_ids):
        raise ExpansionGovernanceError("legacy discovery and holdout overlap")
    return (
        discovery_ids,
        holdout_ids,
        _source_binding(run_index_path, run_index_bytes),
        _source_binding(holdout_path, holdout_bytes),
    )


def _development_fold_order(game_ids: list[str]) -> list[str]:
    return sorted(
        game_ids,
        key=lambda game_id: (
            hashlib.sha256(
                f"{FOLD_ASSIGNMENT_SEED}|development|{game_id}".encode("ascii")
            ).hexdigest(),
            game_id,
        ),
    )


def _build_registry(
    *,
    candidate_scan_path: Path,
    temporal_manifest_path: Path,
    legacy_run_index_path: Path,
    legacy_holdout_path: Path,
) -> dict[str, object]:
    candidate_games, candidate_bytes, candidate_binding = _verify_candidate_scan(
        candidate_scan_path
    )
    (
        temporal_games,
        _temporal_manifest_bytes,
        _temporal_object_bytes,
        temporal_manifest_binding,
        temporal_object_binding,
    ) = _verify_temporal_audit(
        temporal_manifest_path,
        candidate_scan_sha256=_sha256(candidate_bytes),
    )
    if candidate_games != temporal_games:
        raise ExpansionGovernanceError(
            "candidate scan and temporal audit game identities differ"
        )
    (
        legacy_discovery_ids,
        legacy_holdout_ids,
        legacy_run_binding,
        legacy_holdout_binding,
    ) = _verify_legacy_cohort(legacy_run_index_path, legacy_holdout_path)

    expansion_ids = set(temporal_games)
    legacy_ids = set(legacy_discovery_ids) | set(legacy_holdout_ids)
    if len(legacy_ids) != LEGACY_GAME_COUNT or expansion_ids & legacy_ids:
        raise ExpansionGovernanceError(
            "the 234-game expansion is not disjoint from the legacy 40"
        )

    assignments = [
        {
            "game_id": game_id,
            "week": week,
            "cohort": (
                "development"
                if week in DEVELOPMENT_WEEKS
                else "final_holdout"
                if week in FINAL_HOLDOUT_WEEKS
                else "INVALID"
            ),
        }
        for game_id, week in sorted(temporal_games.items())
    ]
    if any(row["cohort"] == "INVALID" for row in assignments):
        raise ExpansionGovernanceError("expansion game escaped weeks 1 through 18")
    development_ids = [
        str(row["game_id"])
        for row in assignments
        if row["cohort"] == "development"
    ]
    final_holdout_ids = [
        str(row["game_id"])
        for row in assignments
        if row["cohort"] == "final_holdout"
    ]
    if (
        len(development_ids) != DEVELOPMENT_GAME_COUNT
        or len(final_holdout_ids) != FINAL_HOLDOUT_GAME_COUNT
    ):
        raise ExpansionGovernanceError(
            "out-of-time split is not exactly 153 development / 81 final holdout"
        )

    builder_bytes = _read_regular(
        Path(__file__),
        label="registry builder",
        max_bytes=1024 * 1024,
    )
    return {
        "schema": "nfl_factor_expansion_registry_v2",
        "experiment_id": "X-13",
        "factor_registry_version": FACTOR_REGISTRY_VERSION,
        "status": "FROZEN",
        "source_bindings": {
            "candidate_scan": candidate_binding,
            "temporal_audit_manifest": temporal_manifest_binding,
            "temporal_audit_object": temporal_object_binding,
            "legacy_v2_run_index": legacy_run_binding,
            "legacy_holdout_selection": legacy_holdout_binding,
            "builder": _source_binding(Path(__file__), builder_bytes),
        },
        "analysis_scope": {
            "market_policy": "MONEYLINE_FIRST",
            "authorized_market_families": ["moneyline"],
            "market_value_or_direction_accessed": False,
            "market_value_or_direction_emitted": False,
        },
        "factor_axes": {
            "horizons_seconds": list(HORIZONS_SECONDS),
            "timeline": {
                "predicate_language": "CONJUNCTIVE_COMPARISON_V1",
                "eligibility": TIME_ELIGIBILITY,
                "formal_buckets": list(TIME_BUCKET_DEFINITIONS),
                "preserve_exact_fields": list(EXACT_TIME_FIELDS),
            },
            "probability": {
                "formal_field": "pre_event_probability",
                "formal_representation": "CONTINUOUS",
                "formal_domain": {
                    "minimum": 0,
                    "maximum": 1,
                    "bounds_inclusive": True,
                },
                "display_bin_eligibility": PROBABILITY_DISPLAY_ELIGIBILITY,
                "display_only_bins": list(
                    PROBABILITY_DISPLAY_BIN_DEFINITIONS
                ),
                "display_labels_are_formal_values": False,
            },
        },
        "split_lock": {
            "method": "OUT_OF_TIME_BY_WEEK",
            "development": {
                "weeks": list(DEVELOPMENT_WEEKS),
                "game_count": len(development_ids),
                "reaction_access": False,
                "access_state": "CLOSED",
            },
            "final_holdout": {
                "weeks": list(FINAL_HOLDOUT_WEEKS),
                "game_count": len(final_holdout_ids),
                "reaction_access": False,
                "access_state": "CLOSED",
            },
            "development_internal_resampling": {
                "cluster_unit": "game_id",
                "fold_assignment_seed": FOLD_ASSIGNMENT_SEED,
                "seed_scope": (
                    "DEVELOPMENT_INTERNAL_GAME_CLUSTER_FOLD_AND_TIE_BREAK_ONLY"
                ),
                "seed_selects_development_or_final_holdout": False,
                "deterministic_cluster_order": _development_fold_order(
                    development_ids
                ),
            },
            "game_assignments": assignments,
        },
        "legacy_cohort": {
            "status": "LEGACY_CONSUMED",
            "game_count": len(legacy_ids),
            "discovery_game_ids": list(legacy_discovery_ids),
            "holdout_game_ids": list(legacy_holdout_ids),
            "mutually_exclusive_with_expansion": True,
        },
        "claim_boundary": {
            "candidate_selection_reaction_blind": True,
            "development_reaction_access": False,
            "final_holdout_reaction_access": False,
            "market_value_or_direction_read": False,
            "market_value_or_direction_output": False,
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ExpansionGovernanceError(
                f"content-addressed path conflicts: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish(
    registry: dict[str, object],
    *,
    output_root: Path,
) -> dict[str, object]:
    object_bytes = _canonical_bytes(registry)
    object_sha256 = _sha256(object_bytes)
    object_hex = object_sha256.removeprefix("sha256:")
    object_relative_path = (
        Path("objects") / "sha256" / object_hex[:2] / f"{object_hex}.json"
    )
    object_path = output_root / object_relative_path
    split_lock = registry["split_lock"]
    assert type(split_lock) is dict
    development = split_lock["development"]
    final_holdout = split_lock["final_holdout"]
    assert type(development) is dict and type(final_holdout) is dict
    manifest = {
        "schema": "nfl_factor_expansion_registry_manifest_v2",
        "factor_registry_version": FACTOR_REGISTRY_VERSION,
        "object_path": object_relative_path.as_posix(),
        "object_sha256": object_sha256,
        "byte_length": len(object_bytes),
        "candidate_game_count": EXPANSION_GAME_COUNT,
        "development_game_count": development["game_count"],
        "final_holdout_game_count": final_holdout["game_count"],
        "development_reaction_access": False,
        "final_holdout_reaction_access": False,
        "market_policy": "MONEYLINE_FIRST",
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    manifest_hex = manifest_sha256.removeprefix("sha256:")
    manifest_relative_path = (
        Path("manifests") / f"{manifest_hex}.manifest.json"
    )
    manifest_path = output_root / manifest_relative_path
    _atomic_write(object_path, object_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return {
        **manifest,
        "manifest_path": manifest_relative_path.as_posix(),
        "manifest_sha256": manifest_sha256,
    }


def _bounded_validate(
    registry: dict[str, object],
    publication: dict[str, object],
    *,
    output_root: Path,
) -> list[str]:
    checks: list[str] = []
    split_lock = registry.get("split_lock")
    axes = registry.get("factor_axes")
    scope = registry.get("analysis_scope")
    legacy = registry.get("legacy_cohort")
    if not all(type(value) is dict for value in (split_lock, axes, scope, legacy)):
        raise ExpansionGovernanceError("registry top-level contracts are malformed")
    assert type(split_lock) is dict
    assert type(axes) is dict
    assert type(scope) is dict
    assert type(legacy) is dict

    assignments = split_lock.get("game_assignments")
    if type(assignments) is not list or len(assignments) != EXPANSION_GAME_COUNT:
        raise ExpansionGovernanceError("bounded validation: assignment count")
    for row in assignments:
        if type(row) is not dict or set(row) != {"game_id", "week", "cohort"}:
            raise ExpansionGovernanceError(
                "bounded validation: unsafe game assignment projection"
            )
    checks.append("234_IDENTITY_ONLY_GAME_ASSIGNMENTS")

    development = split_lock.get("development")
    final_holdout = split_lock.get("final_holdout")
    if (
        type(development) is not dict
        or type(final_holdout) is not dict
        or development.get("weeks") != list(DEVELOPMENT_WEEKS)
        or development.get("game_count") != DEVELOPMENT_GAME_COUNT
        or development.get("reaction_access") is not False
        or final_holdout.get("weeks") != list(FINAL_HOLDOUT_WEEKS)
        or final_holdout.get("game_count") != FINAL_HOLDOUT_GAME_COUNT
        or final_holdout.get("reaction_access") is not False
    ):
        raise ExpansionGovernanceError("bounded validation: split lock")
    checks.append("OUT_OF_TIME_153_81_REACTION_CLOSED")

    if (
        axes.get("horizons_seconds") != list(HORIZONS_SECONDS)
        or type(axes.get("timeline")) is not dict
        or axes["timeline"].get("predicate_language")
        != "CONJUNCTIVE_COMPARISON_V1"
        or axes["timeline"].get("eligibility") != TIME_ELIGIBILITY
        or axes["timeline"].get("formal_buckets")
        != list(TIME_BUCKET_DEFINITIONS)
        or axes["timeline"].get("preserve_exact_fields")
        != list(EXACT_TIME_FIELDS)
        or type(axes.get("probability")) is not dict
        or axes["probability"].get("formal_representation") != "CONTINUOUS"
        or axes["probability"].get("display_bin_eligibility")
        != PROBABILITY_DISPLAY_ELIGIBILITY
        or axes["probability"].get("display_only_bins")
        != list(PROBABILITY_DISPLAY_BIN_DEFINITIONS)
        or axes["probability"].get("display_labels_are_formal_values") is not False
    ):
        raise ExpansionGovernanceError("bounded validation: factor V2 axes")
    checks.append("FACTOR_V2_AXES_FROZEN")

    time_boundary_cases = (
        ((1, "15:00", 900), "Q1"),
        ((2, "02:01", 121), "Q2_PRE_2M"),
        ((2, "02:00", 120), "Q2_FINAL_2M"),
        ((2, "00:00", 0), "Q2_FINAL_2M"),
        ((3, "15:00", 900), "Q3"),
        ((4, "05:01", 301), "Q4_PRE_5M"),
        ((4, "05:00", 300), "Q4_FINAL_5M"),
        ((4, "00:00", 0), "Q4_FINAL_5M"),
        ((5, "10:00", 600), "OT"),
        ((6, "15:00", 900), "OT"),
        ((None, "02:00", 120), "NOT_ELIGIBLE"),
        ((2, None, 120), "NOT_ELIGIBLE"),
        ((2, "", 120), "NOT_ELIGIBLE"),
        ((2, "02:00", None), "NOT_ELIGIBLE"),
        ((0, "02:00", 120), "NOT_ELIGIBLE"),
        ((2, "00:00", -1), "NOT_ELIGIBLE"),
        ((2, "15:01", 901), "NOT_ELIGIBLE"),
    )
    if any(
        _classify_time_bucket(*inputs) != expected
        for inputs, expected in time_boundary_cases
    ):
        raise ExpansionGovernanceError(
            "bounded validation: time bucket predicate boundary"
        )
    checks.append("TIME_BUCKET_MACHINE_PREDICATES_AND_BOUNDARIES")

    probability_boundary_cases = (
        (0, "LOW"),
        (0.329999999999, "LOW"),
        (0.33, "MID"),
        (0.67, "MID"),
        (0.670000000001, "HIGH"),
        (1, "HIGH"),
        (None, "NOT_ELIGIBLE"),
        (-0.000000000001, "NOT_ELIGIBLE"),
        (1.000000000001, "NOT_ELIGIBLE"),
        (float("nan"), "NOT_ELIGIBLE"),
        (float("inf"), "NOT_ELIGIBLE"),
    )
    if any(
        _classify_probability_display_bin(value) != expected
        for value, expected in probability_boundary_cases
    ):
        raise ExpansionGovernanceError(
            "bounded validation: probability display boundary"
        )
    checks.append("PROBABILITY_DISPLAY_PREDICATES_AND_BOUNDARIES")

    if (
        scope.get("market_policy") != "MONEYLINE_FIRST"
        or scope.get("authorized_market_families") != ["moneyline"]
        or scope.get("market_value_or_direction_accessed") is not False
        or scope.get("market_value_or_direction_emitted") is not False
    ):
        raise ExpansionGovernanceError("bounded validation: market scope")
    checks.append("MONEYLINE_FIRST_NO_MARKET_VALUE_OR_DIRECTION")

    expansion_ids = {str(row["game_id"]) for row in assignments}
    legacy_ids = set(legacy.get("discovery_game_ids", [])) | set(
        legacy.get("holdout_game_ids", [])
    )
    if (
        legacy.get("status") != "LEGACY_CONSUMED"
        or len(legacy_ids) != LEGACY_GAME_COUNT
        or expansion_ids & legacy_ids
    ):
        raise ExpansionGovernanceError("bounded validation: legacy separation")
    checks.append("LEGACY_40_CONSUMED_AND_DISJOINT")

    object_relative = publication.get("object_path")
    manifest_relative = publication.get("manifest_path")
    if type(object_relative) is not str or type(manifest_relative) is not str:
        raise ExpansionGovernanceError("bounded validation: publication paths")
    object_path = output_root / object_relative
    manifest_path = output_root / manifest_relative
    object_bytes = _read_regular(
        object_path,
        label="published registry object",
        max_bytes=2 * 1024 * 1024,
    )
    manifest_bytes = _read_regular(
        manifest_path,
        label="published registry manifest",
        max_bytes=64 * 1024,
    )
    if (
        object_bytes != _canonical_bytes(registry)
        or _sha256(object_bytes) != publication.get("object_sha256")
        or len(object_bytes) != publication.get("byte_length")
        or _sha256(manifest_bytes) != publication.get("manifest_sha256")
    ):
        raise ExpansionGovernanceError("bounded validation: hash round trip")
    checks.append("CONTENT_ADDRESS_ROUND_TRIP")

    repeated = _publish(registry, output_root=output_root)
    if repeated != publication:
        raise ExpansionGovernanceError("bounded validation: nondeterministic publish")
    checks.append("IDEMPOTENT_DETERMINISTIC_REPUBLISH")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the 234-game, reaction-closed Factor V2 expansion registry"
        )
    )
    parser.add_argument(
        "--candidate-scan",
        type=Path,
        default=CANDIDATE_SCAN,
    )
    parser.add_argument(
        "--temporal-manifest",
        type=Path,
        default=TEMPORAL_MANIFEST,
    )
    parser.add_argument(
        "--legacy-run-index",
        type=Path,
        default=LEGACY_RUN_INDEX,
    )
    parser.add_argument(
        "--legacy-holdout-selection",
        type=Path,
        default=LEGACY_HOLDOUT_SELECTION,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bounded-validation", action="store_true")
    args = parser.parse_args()

    registry = _build_registry(
        candidate_scan_path=args.candidate_scan,
        temporal_manifest_path=args.temporal_manifest,
        legacy_run_index_path=args.legacy_run_index,
        legacy_holdout_path=args.legacy_holdout_selection,
    )
    publication = _publish(registry, output_root=args.output_root)
    if args.bounded_validation:
        checks = _bounded_validate(
            registry,
            publication,
            output_root=args.output_root,
        )
    else:
        checks = []
    print(
        json.dumps(
            {
                **publication,
                "bounded_validation": (
                    {"status": "PASS", "checks": checks}
                    if args.bounded_validation
                    else {"status": "NOT_REQUESTED", "checks": []}
                ),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
