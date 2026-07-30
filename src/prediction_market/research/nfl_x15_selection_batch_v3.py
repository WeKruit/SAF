"""Exact-45 X-15 selection batch over immutable V4 OOF shards.

The V4 shard format remains the canonical computation checkpoint.  This
module publishes a narrower selection authority that excludes unsupported D4
features while preserving byte- and semantic-level verification through the
public V4 shard loader.

No function in this module resolves or reads final-holdout reaction data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prediction_market.immutable_object_store import sha256_path
from prediction_market.research.nfl_x15_model_selection import (
    EXPECTED_DEVELOPMENT_GAME_COUNT,
    EXPECTED_FOLDS,
    FrozenDevelopmentAuthority,
    ModelSelectionError,
    bind_frozen_development_authority,
)
from prediction_market.research.nfl_x15_models import (
    X15ModelRun,
)
from prediction_market.research.nfl_x15_oof_publication import (
    SELECTION_PROJECTION_COLUMNS,
    X15OOFPublicationError,
    _read_shard_document as _read_canonical_v4_shard_document,
    load_x15_oof_shard,
)


SCHEMA: Final[str] = "nfl_x15_selection_batch_v3"
BUILDER_VERSION: Final[str] = "nfl-x15-selection-batch-v3"
SHARD_SCHEMA: Final[str] = "nfl_x15_oof_shard_publication_v4"
SHARD_BUILDER_VERSION: Final[str] = "nfl-x15-oof-publication-v4"
PANEL_SCHEMA: Final[str] = "nfl_x15_development_panel_batch_index_v1"
PANEL_BUILDER_VERSION: Final[str] = "nfl-x15-development-panel-v2"
SUPPORT_AUDIT_SCHEMA: Final[str] = (
    "nfl_x15_stage_a_decision_support_audit_v1"
)
SUPPORT_AUDIT_BUILDER_VERSION: Final[str] = (
    "nfl-x15-stage-a-decision-support-audit-v1"
)
EXPECTED_DECISION_ELIGIBLE_ROWS: Final[int] = 1_050_084
REQUIRED_ZERO_NONNULL_FIELDS: Final[tuple[str, ...]] = (
    "p_before_home",
    "p_after_home",
    "reference_delta_home",
    "reference_gap_at_landmark",
)
SUPPORT_AUDIT_CLAIM_BOUNDARY: Final[str] = (
    "DEVELOPMENT_DECISION_TIME_FEATURE_SUPPORT_AUDIT_"
    "NOT_MARKET_REACTION_RESULT"
)
UNSUPPORTED_D4_REASON: Final[str] = (
    "ZERO_NON_NULL_STAGE_A_REFERENCE_FEATURES_ON_DECISION_ELIGIBLE_ROWS"
)
EXACT_DESIGN_MATRIX_V3: Final[tuple[tuple[str, str], ...]] = (
    ("b0_empirical_v1", "D0"),
    ("regularized_logistic_v1", "D0"),
    ("regularized_logistic_v1", "D1"),
    ("regularized_logistic_v1", "D2"),
    ("regularized_logistic_v1", "D3"),
    ("shallow_xgboost_v1", "D0"),
    ("shallow_xgboost_v1", "D1"),
    ("shallow_xgboost_v1", "D2"),
    ("shallow_xgboost_v1", "D3"),
)
SELECTABLE_CANDIDATES_V3: Final[tuple[tuple[str, str], ...]] = (
    ("regularized_logistic_v1", "D1"),
    ("regularized_logistic_v1", "D2"),
    ("regularized_logistic_v1", "D3"),
    ("shallow_xgboost_v1", "D1"),
    ("shallow_xgboost_v1", "D2"),
    ("shallow_xgboost_v1", "D3"),
)
SOURCE_PROVENANCE_ANCHOR_SCHEMA: Final[str] = (
    "nfl_x15_selection_source_provenance_anchor_v1"
)
SOURCE_PROVENANCE_HASH_COLUMNS: Final[tuple[str, ...]] = (
    "training_data_sha256",
    "preprocessor_training_sha256",
    "s_h_model_training_sha256",
    "o_h_given_s_model_training_sha256",
    "direction_model_training_sha256",
    "s_h_calibration_training_sha256",
    "o_h_given_s_calibration_training_sha256",
    "direction_calibration_training_sha256",
    "feature_block_sha256",
    "model_spec_sha256",
    "fold_sha256",
)
SOURCE_PROVENANCE_CALIBRATION_COLUMNS: Final[tuple[str, ...]] = (
    "calibrator_fit_game_ids_s_h",
    "calibrator_fit_game_ids_o_h_given_s",
    "calibrator_fit_game_ids_direction",
)
SOURCE_PROVENANCE_ANCHOR_COLUMN_TYPES: Final[
    tuple[tuple[str, str], ...]
] = (
    ("fold_id", "utf8"),
    ("model_id", "utf8"),
    ("feature_block_id", "utf8"),
    ("shard_manifest_path", "utf8"),
    ("shard_manifest_file_sha256", "sha256"),
    ("shard_bundle_sha256", "sha256"),
    ("oof_predictions_object_sha256", "sha256"),
    ("selection_fold_arrays_sha256", "sha256"),
    ("cohort_authority_sha256", "sha256"),
    ("authority_metadata_sha256", "sha256"),
    ("cohort_mapping_sha256", "sha256"),
    ("run_config_sha256", "sha256"),
    ("run_config_fold_ids", "list<utf8>"),
    ("run_config_model_ids", "list<utf8>"),
    ("run_config_feature_block_ids", "list<utf8>"),
    ("train_weeks", "list<int64>"),
    ("validation_weeks", "list<int64>"),
    ("training_game_ids", "list<utf8>"),
    ("validation_game_ids", "list<utf8>"),
    ("preprocessor_fit_game_ids", "list<utf8>"),
    *tuple((column, "sha256") for column in SOURCE_PROVENANCE_HASH_COLUMNS),
    *tuple(
        (column, "list<utf8>")
        for column in SOURCE_PROVENANCE_CALIBRATION_COLUMNS
    ),
)
SOURCE_PROVENANCE_ANCHOR_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column, _ in SOURCE_PROVENANCE_ANCHOR_COLUMN_TYPES
)
V3_SELECTION_PROJECTION_COLUMNS: Final[tuple[str, ...]] = (
    *SELECTION_PROJECTION_COLUMNS,
    *SOURCE_PROVENANCE_HASH_COLUMNS,
    *SOURCE_PROVENANCE_CALIBRATION_COLUMNS,
)
EXPECTED_FOLD_IDS: Final[tuple[str, ...]] = tuple(
    fold_id for fold_id, _, _ in EXPECTED_FOLDS
)
EXACT_SELECTION_CELLS_V3: Final[
    tuple[tuple[str, str, str], ...]
] = tuple(
    (fold_id, model_id, feature_block_id)
    for fold_id in EXPECTED_FOLD_IDS
    for model_id, feature_block_id in EXACT_DESIGN_MATRIX_V3
)
_LEGACY_D4_CELLS: Final[frozenset[tuple[str, str, str]]] = frozenset(
    (fold_id, model_id, "D4")
    for fold_id in EXPECTED_FOLD_IDS
    for model_id in (
        "regularized_logistic_v1",
        "shallow_xgboost_v1",
    )
)
_SHA256_PREFIX: Final[str] = "sha256:"
_EXPECTED_TRAINING_ONLY_GAME_COUNT: Final[int] = 30
_EXPECTED_OOF_VALIDATION_GAME_COUNT: Final[int] = 123


class X15SelectionBatchV3Error(RuntimeError):
    """An exact-45 selection artifact failed a governed invariant."""


@dataclass(frozen=True, slots=True)
class X15SelectionBatchV3Publication:
    manifest_path: Path
    manifest_sha256: str
    batch_sha256: str
    selection_contract_sha256: str
    observed_execution_cell_count: int
    selection_ready: bool
    holdout_reaction_accessed: bool = False


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(child)
            for key, child in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        children = list(value)
        if isinstance(value, set):
            children = sorted(children, key=repr)
        return [_canonical(child) for child in children]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return str(value)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _model_sha256(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_bytes(payload: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_SHA256_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise X15SelectionBatchV3Error(
            f"{field} must be a SHA-256 digest"
        )
    return value


def _strict_json(payload: bytes, *, field: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise X15SelectionBatchV3Error(
                    f"{field} contains duplicate JSON key {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise X15SelectionBatchV3Error(
            f"{field} is not valid JSON"
        ) from error
    if type(value) is not dict:
        raise X15SelectionBatchV3Error(
            f"{field} must be a JSON object"
        )
    return value


def _resolve_under(
    root: Path, path: str | Path, *, field: str
) -> Path:
    root = root.resolve()
    candidate = (
        Path(path).resolve()
        if Path(path).is_absolute()
        else (root / path).resolve()
    )
    if candidate != root and root not in candidate.parents:
        raise X15SelectionBatchV3Error(f"{field} escapes its root")
    return candidate


def _require_output_root_contract(
    *,
    project_root: Path,
    output_root: Path,
) -> dict[str, object]:
    project = project_root.resolve()
    output = output_root.resolve()
    if output != project and project not in output.parents:
        raise X15SelectionBatchV3Error(
            "output_root must be within or equal to project_root"
        )
    relative = (
        "."
        if output == project
        else output.relative_to(project).as_posix()
    )
    material = {
        "artifact_root_semantics": "PROJECT_ROOT",
        "output_root": relative,
        "reference_path_semantics": (
            "PROJECT_ROOT_RELATIVE_EXTERNAL_AND_OUTPUT_ROOT_RELATIVE_SHARDS"
        ),
    }
    return {
        **material,
        "root_contract_sha256": _canonical_sha256(material),
    }


def _verify_root_contract(
    *,
    document: Mapping[str, object],
    project_root: Path,
    output_root: Path,
) -> dict[str, object]:
    expected = _require_output_root_contract(
        project_root=project_root, output_root=output_root
    )
    observed = document.get("root_contract")
    if not isinstance(observed, dict) or _canonical(
        observed
    ) != _canonical(expected):
        raise X15SelectionBatchV3Error(
            "selection batch root contract mismatch"
        )
    return expected


def _verify_content_addressed_path(
    path: Path,
    *,
    expected_file_sha256: str,
    suffix: str,
    field: str,
) -> tuple[str, int]:
    digest = _require_sha256(
        expected_file_sha256, field=f"{field}.file_sha256"
    ).removeprefix(_SHA256_PREFIX)
    if (
        not path.name.endswith(suffix)
        or path.name[: -len(suffix)] != digest
        or path.parent.name != digest[:2]
        or path.parent.parent.name != "sha256"
    ):
        raise X15SelectionBatchV3Error(
            f"{field} path is not content-addressed"
        )
    if path.is_symlink() or not path.is_file():
        raise X15SelectionBatchV3Error(
            f"{field} is not a regular file"
        )
    observed, byte_length = sha256_path(path)
    if observed != expected_file_sha256:
        raise X15SelectionBatchV3Error(f"{field} SHA-256 mismatch")
    return observed, byte_length


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise X15SelectionBatchV3Error(
                f"content-addressed collision: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise X15SelectionBatchV3Error(
                    f"content-addressed publish race: {path}"
                )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_manifest(
    *,
    output_root: Path,
    material: dict[str, object],
) -> tuple[Path, str, str]:
    batch_sha256 = _canonical_sha256(material)
    document = {**material, "batch_sha256": batch_sha256}
    payload = _canonical_bytes(document)
    file_sha256 = _sha256_bytes(payload)
    digest = file_sha256.removeprefix(_SHA256_PREFIX)
    path = (
        output_root
        / "selection-v3"
        / "batches"
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.batch-index.json"
    )
    _atomic_publish(path, payload)
    return path, file_sha256, batch_sha256


def _canonical_authority(
    authority: FrozenDevelopmentAuthority,
) -> FrozenDevelopmentAuthority:
    if not isinstance(authority, FrozenDevelopmentAuthority):
        raise X15SelectionBatchV3Error(
            "authority must be FrozenDevelopmentAuthority"
        )
    frame = pd.DataFrame(
        authority.development_games,
        columns=("game_id", "nfl_week", "kickoff_utc", "batch_sha256"),
    )
    frame["cohort_authority_sha256"] = (
        authority.cohort_authority_sha256
    )
    try:
        rebound = bind_frozen_development_authority(
            frame,
            cohort_authority_sha256=authority.cohort_authority_sha256,
        )
    except ModelSelectionError as error:
        raise X15SelectionBatchV3Error(str(error)) from error
    if rebound != authority:
        raise X15SelectionBatchV3Error("authority is not canonical")
    game_weeks = authority.game_weeks
    if (
        authority.game_count != EXPECTED_DEVELOPMENT_GAME_COUNT
        or sum(
            week in (1, 2) for week in game_weeks.values()
        )
        != _EXPECTED_TRAINING_ONLY_GAME_COUNT
        or sum(
            week in range(3, 13) for week in game_weeks.values()
        )
        != _EXPECTED_OOF_VALIDATION_GAME_COUNT
    ):
        raise X15SelectionBatchV3Error(
            "authority is not the exact 153-game development cohort"
        )
    return rebound


def _authority_from_document(
    document: Mapping[str, object],
) -> FrozenDevelopmentAuthority:
    games = document.get("authority_development_games")
    if not isinstance(games, list):
        raise X15SelectionBatchV3Error(
            "batch authority development games are malformed"
        )
    normalized: list[tuple[object, ...]] = []
    for value in games:
        if not isinstance(value, list) or len(value) != 4:
            raise X15SelectionBatchV3Error(
                "batch authority development games are malformed"
            )
        normalized.append(tuple(value))
    cohort_sha = _require_sha256(
        document.get("cohort_authority_sha256"),
        field="batch.cohort_authority_sha256",
    )
    frame = pd.DataFrame(
        normalized,
        columns=("game_id", "nfl_week", "kickoff_utc", "batch_sha256"),
    )
    frame["cohort_authority_sha256"] = cohort_sha
    try:
        authority = bind_frozen_development_authority(
            frame, cohort_authority_sha256=cohort_sha
        )
    except ModelSelectionError as error:
        raise X15SelectionBatchV3Error(str(error)) from error
    if (
        authority.development_games != tuple(normalized)
        or authority.metadata_sha256
        != document.get("authority_metadata_sha256")
        or authority.game_count != document.get("authority_game_count")
    ):
        raise X15SelectionBatchV3Error(
            "batch authority metadata is inconsistent"
        )
    return _canonical_authority(authority)


def _read_panel_binding(
    *,
    project_root: Path,
    panel_batch_manifest_path: str | Path,
    panel_batch_file_sha256: str,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
) -> tuple[dict[str, object], Path, str]:
    path = _resolve_under(
        project_root,
        panel_batch_manifest_path,
        field="panel batch manifest",
    )
    file_sha = _require_sha256(
        panel_batch_file_sha256,
        field="panel_batch_file_sha256",
    )
    _verify_content_addressed_path(
        path,
        expected_file_sha256=file_sha,
        suffix=".batch-index.json",
        field="panel batch manifest",
    )
    document = _strict_json(
        path.read_bytes(), field="panel batch manifest"
    )
    material = dict(document)
    batch_sha = _require_sha256(
        material.pop("batch_sha256", None),
        field="panel batch.batch_sha256",
    )
    if _canonical_sha256(material) != batch_sha:
        raise X15SelectionBatchV3Error(
            "panel batch semantic SHA-256 mismatch"
        )
    if (
        document.get("schema") != PANEL_SCHEMA
        or document.get("builder_version") != PANEL_BUILDER_VERSION
        or document.get("cohort") != "development"
        or document.get("cohort_authority_sha256")
        != authority.cohort_authority_sha256
        or document.get("cohort_mapping_sha256")
        != cohort_mapping_sha256
        or document.get("published_game_count") != 153
        or document.get("verified_development_game_count") != 153
        or document.get("partial_publication") is not False
        or document.get("publication_gate") != "PASS"
        or document.get("holdout_reaction_accessed") is not False
    ):
        raise X15SelectionBatchV3Error(
            "panel batch contract mismatch"
        )
    return document, path, file_sha


def _read_support_audit(
    *,
    project_root: Path,
    feature_support_audit_path: str | Path,
    feature_support_audit_file_sha256: str,
    panel_document: Mapping[str, object],
    panel_path: Path,
    panel_file_sha256: str,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
) -> tuple[dict[str, object], Path, str]:
    path = _resolve_under(
        project_root,
        feature_support_audit_path,
        field="feature support audit",
    )
    file_sha = _require_sha256(
        feature_support_audit_file_sha256,
        field="feature_support_audit_file_sha256",
    )
    _verify_content_addressed_path(
        path,
        expected_file_sha256=file_sha,
        suffix=".stage-a-support-audit.json",
        field="feature support audit",
    )
    document = _strict_json(
        path.read_bytes(), field="feature support audit"
    )
    material = dict(document)
    audit_sha = _require_sha256(
        material.pop("audit_sha256", None),
        field="feature support audit.audit_sha256",
    )
    if _model_sha256(material) != audit_sha:
        raise X15SelectionBatchV3Error(
            "feature support audit semantic SHA-256 mismatch"
        )
    expected_panel_relative = panel_path.relative_to(
        project_root
    ).as_posix()
    nonnull_counts = document.get("nonnull_counts")
    if not isinstance(nonnull_counts, dict):
        raise X15SelectionBatchV3Error(
            "feature support audit nonnull_counts are malformed"
        )
    feature_status = document.get("feature_block_status")
    if not isinstance(feature_status, dict):
        raise X15SelectionBatchV3Error(
            "feature support audit feature_block_status is malformed"
        )
    selection_eligible = document.get("selection_eligible")
    support_reason = document.get("support_reason")
    if not isinstance(selection_eligible, dict) or not isinstance(
        support_reason, dict
    ):
        raise X15SelectionBatchV3Error(
            "feature support audit D4 eligibility evidence is malformed"
        )
    if (
        document.get("schema") != SUPPORT_AUDIT_SCHEMA
        or document.get("builder_version")
        != SUPPORT_AUDIT_BUILDER_VERSION
        or document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or document.get("panel_batch_manifest_path")
        != expected_panel_relative
        or document.get("panel_batch_file_sha256") != panel_file_sha256
        or document.get("panel_batch_sha256")
        != panel_document.get("batch_sha256")
        or document.get("cohort_authority_sha256")
        != authority.cohort_authority_sha256
        or document.get("cohort_mapping_sha256")
        != cohort_mapping_sha256
        or document.get("decision_eligible_rows")
        != EXPECTED_DECISION_ELIGIBLE_ROWS
        or any(
            nonnull_counts.get(field) != 0
            for field in REQUIRED_ZERO_NONNULL_FIELDS
        )
        or set(nonnull_counts) != set(REQUIRED_ZERO_NONNULL_FIELDS)
        or feature_status.get("D4") != "UNSUPPORTED_FEATURE_BLOCK"
        or selection_eligible != {"D4": False}
        or support_reason != {"D4": UNSUPPORTED_D4_REASON}
        or document.get("claim_boundary")
        != SUPPORT_AUDIT_CLAIM_BOUNDARY
        or document.get("holdout_reaction_accessed") is not False
    ):
        if feature_status.get("D4") != "UNSUPPORTED_FEATURE_BLOCK":
            raise X15SelectionBatchV3Error(
                "D4 must be UNSUPPORTED_FEATURE_BLOCK"
            )
        raise X15SelectionBatchV3Error(
            "feature support audit contract mismatch"
        )
    return document, path, file_sha


def _fold_authority(
    authority: FrozenDevelopmentAuthority,
    fold_id: str,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    try:
        _, train_weeks, validation_weeks = next(
            fold for fold in EXPECTED_FOLDS if fold[0] == fold_id
        )
    except StopIteration as error:
        raise X15SelectionBatchV3Error(
            f"unknown fold_id: {fold_id}"
        ) from error
    game_weeks = authority.game_weeks
    training_game_ids = tuple(
        sorted(
            game_id
            for game_id, week in game_weeks.items()
            if week in train_weeks
        )
    )
    validation_game_ids = tuple(
        sorted(
            game_id
            for game_id, week in game_weeks.items()
            if week in validation_weeks
        )
    )
    return (
        tuple(train_weeks),
        tuple(validation_weeks),
        training_game_ids,
        validation_game_ids,
    )


def _tuple(value: object) -> tuple[object, ...] | None:
    return tuple(value) if isinstance(value, (tuple, list)) else None


def _read_v4_manifest(
    *,
    manifest_path: str | Path,
    output_root: Path,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    allow_unsupported_d4: bool = False,
) -> tuple[
    tuple[str, str, str],
    dict[str, object],
    Path,
    str,
]:
    path = _resolve_under(
        output_root, manifest_path, field="V4 shard manifest"
    )
    try:
        document, loaded_tables = _read_canonical_v4_shard_document(
            manifest_path=path,
            output_root=output_root,
            load_tables=False,
        )
    except X15OOFPublicationError as error:
        raise X15SelectionBatchV3Error(
            "canonical V4 shard bytes/metadata verification failed: "
            f"{error}"
        ) from error
    if loaded_tables:
        raise X15SelectionBatchV3Error(
            "canonical V4 metadata verification materialized tables"
        )
    observed_file_sha, _ = sha256_path(path)
    fold_id = str(document.get("fold_id", ""))
    model_id = str(document.get("model_id", ""))
    feature_block_id = str(document.get("feature_block_id", ""))
    cell = (fold_id, model_id, feature_block_id)
    digest = observed_file_sha.removeprefix(_SHA256_PREFIX)
    expected_manifest_path = (
        output_root
        / "shards"
        / fold_id
        / model_id
        / feature_block_id
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.shard-manifest.json"
    )
    if path != expected_manifest_path:
        raise X15SelectionBatchV3Error(
            "V4 shard manifest namespace mismatch"
        )
    if feature_block_id == "D4" and not allow_unsupported_d4:
        raise X15SelectionBatchV3Error(
            "D4 is outside the exact-45 selection contract"
        )
    if cell not in EXACT_SELECTION_CELLS_V3 and not (
        allow_unsupported_d4 and cell in _LEGACY_D4_CELLS
    ):
        raise X15SelectionBatchV3Error(
            "V4 shard is outside the exact-45 execution matrix"
        )
    run_config = document.get("run_config")
    if not isinstance(run_config, dict):
        raise X15SelectionBatchV3Error(
            "V4 shard run config is malformed"
        )
    expected_authority = _fold_authority(authority, fold_id)
    if (
        document.get("schema") != SHARD_SCHEMA
        or document.get("builder_version") != SHARD_BUILDER_VERSION
        or document.get("cohort") != "development"
        or document.get("holdout_reaction_accessed") is not False
        or document.get("cohort_authority_sha256")
        != authority.cohort_authority_sha256
        or document.get("authority_metadata_sha256")
        != authority.metadata_sha256
        or document.get("authority_game_count") != 153
        or document.get("cohort_mapping_sha256")
        != cohort_mapping_sha256
        or document.get("authority_coverage_complete") is not True
        or _tuple(document.get("train_weeks")) != expected_authority[0]
        or _tuple(document.get("validation_weeks"))
        != expected_authority[1]
        or _tuple(document.get("training_game_ids"))
        != expected_authority[2]
        or _tuple(document.get("validation_game_ids"))
        != expected_authority[3]
        or _tuple(document.get("observed_validation_game_ids"))
        != expected_authority[3]
        or _tuple(run_config.get("fold_ids")) != (fold_id,)
        or _tuple(run_config.get("model_ids")) != (model_id,)
        or _tuple(run_config.get("feature_block_ids"))
        != (feature_block_id,)
        or run_config.get("include_magnitude") is not False
    ):
        raise X15SelectionBatchV3Error(
            f"{fold_id}/{model_id}/{feature_block_id} authority, identity, "
            "or singleton coordinate mismatch"
        )
    return cell, document, path, observed_file_sha


def _require_text_sequence(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise X15SelectionBatchV3Error(
            f"{field} must be a tuple/list"
        )
    result = tuple(value)
    if not allow_empty and not result:
        raise X15SelectionBatchV3Error(f"{field} must not be empty")
    if not all(
        type(item) is str and bool(item.strip()) for item in result
    ):
        raise X15SelectionBatchV3Error(
            f"{field} must contain nonempty text"
        )
    if len(result) != len(set(result)):
        raise X15SelectionBatchV3Error(
            f"{field} must not contain duplicates"
        )
    return result


def _require_week_sequence(
    value: object, *, field: str
) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise X15SelectionBatchV3Error(
            f"{field} must be a nonempty tuple/list"
        )
    if not all(
        isinstance(item, (int, np.integer))
        and not isinstance(item, (bool, np.bool_))
        for item in value
    ):
        raise X15SelectionBatchV3Error(
            f"{field} must contain integers"
        )
    result = tuple(int(item) for item in value)
    if len(result) != len(set(result)):
        raise X15SelectionBatchV3Error(
            f"{field} must not contain duplicates"
        )
    return result


def _oof_predictions_descriptor(
    document: Mapping[str, object],
) -> dict[str, object]:
    tables = document.get("tables")
    if not isinstance(tables, list):
        raise X15SelectionBatchV3Error(
            "V4 shard provenance tables are malformed"
        )
    matches = [
        value
        for value in tables
        if isinstance(value, dict)
        and value.get("name") == "oof_predictions"
    ]
    if len(matches) != 1:
        raise X15SelectionBatchV3Error(
            "V4 shard provenance lacks one oof_predictions object"
        )
    return matches[0]


def _native_polymarket_mask_arrow(batch) -> object:
    mask = pc.equal(batch.column("venue"), "polymarket")
    for column, expected in (
        ("training_venue", "polymarket"),
        ("calibration_venue", "polymarket"),
        ("transport_mode", "VENUE_SPECIFIC"),
    ):
        mask = pc.and_(
            mask, pc.equal(batch.column(column), expected)
        )
    return mask


def _unique_oof_provenance_from_arrow(
    *,
    document: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    descriptor = _oof_predictions_descriptor(document)
    object_path = _resolve_under(
        output_root,
        str(descriptor.get("object_path", "")),
        field="oof_predictions provenance object",
    )
    required = (
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        *SOURCE_PROVENANCE_HASH_COLUMNS,
        *SOURCE_PROVENANCE_CALIBRATION_COLUMNS,
    )
    try:
        parquet = pq.ParquetFile(object_path)
    except Exception as error:
        raise X15SelectionBatchV3Error(
            "oof_predictions provenance Parquet is unreadable"
        ) from error
    missing = sorted(set(required).difference(parquet.schema_arrow.names))
    if missing:
        raise X15SelectionBatchV3Error(
            f"oof_predictions provenance columns are missing: {missing}"
        )
    unique: dict[str, set[object]] = {
        column: set()
        for column in (
            *SOURCE_PROVENANCE_HASH_COLUMNS,
            *SOURCE_PROVENANCE_CALIBRATION_COLUMNS,
        )
    }
    native_row_count = 0
    try:
        batches = parquet.iter_batches(
            batch_size=1024,
            columns=list(required),
            use_threads=False,
        )
        for batch in batches:
            filtered = batch.filter(
                _native_polymarket_mask_arrow(batch)
            )
            if filtered.num_rows == 0:
                continue
            native_row_count += filtered.num_rows
            for column in SOURCE_PROVENANCE_HASH_COLUMNS:
                values = pc.unique(filtered.column(column)).to_pylist()
                unique[column].update(values)
                if len(unique[column]) > 1:
                    raise X15SelectionBatchV3Error(
                        "oof_predictions provenance has multiple "
                        f"native Polymarket values for {column}"
                    )
            for column in SOURCE_PROVENANCE_CALIBRATION_COLUMNS:
                values = filtered.column(column).to_pylist()
                unique[column].update(
                    tuple(value) if isinstance(value, list) else value
                    for value in values
                )
                if len(unique[column]) > 1:
                    raise X15SelectionBatchV3Error(
                        "oof_predictions provenance has multiple "
                        f"native Polymarket values for {column}"
                    )
    except X15SelectionBatchV3Error:
        raise
    except Exception as error:
        raise X15SelectionBatchV3Error(
            "oof_predictions provenance scan failed"
        ) from error
    if native_row_count == 0:
        raise X15SelectionBatchV3Error(
            "oof_predictions provenance has no native Polymarket rows"
        )
    result: dict[str, object] = {}
    for column in SOURCE_PROVENANCE_HASH_COLUMNS:
        if len(unique[column]) != 1:
            raise X15SelectionBatchV3Error(
                "oof_predictions provenance requires one value for "
                f"{column}"
            )
        result[column] = _require_sha256(
            next(iter(unique[column])),
            field=f"oof_predictions provenance.{column}",
        )
    for column in SOURCE_PROVENANCE_CALIBRATION_COLUMNS:
        if len(unique[column]) != 1:
            raise X15SelectionBatchV3Error(
                "oof_predictions provenance requires one value for "
                f"{column}"
            )
        result[column] = _require_text_sequence(
            next(iter(unique[column])),
            field=f"oof_predictions provenance.{column}",
        )
    return result


def _unique_oof_provenance_from_frame(
    frame: pd.DataFrame,
) -> dict[str, object]:
    required = {
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        *SOURCE_PROVENANCE_HASH_COLUMNS,
        *SOURCE_PROVENANCE_CALIBRATION_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise X15SelectionBatchV3Error(
            f"loaded OOF provenance columns are missing: {missing}"
        )
    native = frame.loc[
        frame["venue"].eq("polymarket")
        & frame["training_venue"].eq("polymarket")
        & frame["calibration_venue"].eq("polymarket")
        & frame["transport_mode"].eq("VENUE_SPECIFIC")
    ]
    if native.empty:
        raise X15SelectionBatchV3Error(
            "loaded OOF provenance has no native Polymarket rows"
        )
    result: dict[str, object] = {}
    for column in SOURCE_PROVENANCE_HASH_COLUMNS:
        values = set(native[column].tolist())
        if len(values) != 1:
            raise X15SelectionBatchV3Error(
                f"loaded OOF provenance has multiple values for {column}"
            )
        result[column] = _require_sha256(
            next(iter(values)),
            field=f"loaded OOF provenance.{column}",
        )
    for column in SOURCE_PROVENANCE_CALIBRATION_COLUMNS:
        values = {
            tuple(value) if isinstance(value, (list, np.ndarray)) else value
            for value in native[column].tolist()
        }
        if len(values) != 1:
            raise X15SelectionBatchV3Error(
                f"loaded OOF provenance has multiple values for {column}"
            )
        result[column] = _require_text_sequence(
            next(iter(values)),
            field=f"loaded OOF provenance.{column}",
        )
    return result


def _source_provenance_anchor_row(
    *,
    document: Mapping[str, object],
    path: Path,
    file_sha256: str,
    output_root: Path,
    oof_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    fold_id = str(document.get("fold_id", ""))
    model_id = str(document.get("model_id", ""))
    feature_block_id = str(document.get("feature_block_id", ""))
    run_config = document.get("run_config")
    fold_arrays = document.get("selection_fold_arrays")
    if not isinstance(run_config, dict) or not isinstance(
        fold_arrays, dict
    ):
        raise X15SelectionBatchV3Error(
            "V4 shard provenance identities are malformed"
        )
    descriptor = _oof_predictions_descriptor(document)
    values = dict(
        oof_provenance
        if oof_provenance is not None
        else _unique_oof_provenance_from_arrow(
            document=document, output_root=output_root
        )
    )
    row: dict[str, object] = {
        "fold_id": fold_id,
        "model_id": model_id,
        "feature_block_id": feature_block_id,
        "shard_manifest_path": path.relative_to(
            output_root
        ).as_posix(),
        "shard_manifest_file_sha256": _require_sha256(
            file_sha256, field="shard manifest file SHA-256"
        ),
        "shard_bundle_sha256": _require_sha256(
            document.get("bundle_sha256"),
            field="shard bundle_sha256",
        ),
        "oof_predictions_object_sha256": _require_sha256(
            descriptor.get("object_sha256"),
            field="oof_predictions object_sha256",
        ),
        "selection_fold_arrays_sha256": _require_sha256(
            document.get("selection_fold_arrays_sha256"),
            field="selection_fold_arrays_sha256",
        ),
        "cohort_authority_sha256": _require_sha256(
            document.get("cohort_authority_sha256"),
            field="cohort_authority_sha256",
        ),
        "authority_metadata_sha256": _require_sha256(
            document.get("authority_metadata_sha256"),
            field="authority_metadata_sha256",
        ),
        "cohort_mapping_sha256": _require_sha256(
            document.get("cohort_mapping_sha256"),
            field="cohort_mapping_sha256",
        ),
        "run_config_sha256": _require_sha256(
            document.get("run_config_sha256"),
            field="run_config_sha256",
        ),
        "run_config_fold_ids": _require_text_sequence(
            run_config.get("fold_ids"),
            field="run_config.fold_ids",
        ),
        "run_config_model_ids": _require_text_sequence(
            run_config.get("model_ids"),
            field="run_config.model_ids",
        ),
        "run_config_feature_block_ids": _require_text_sequence(
            run_config.get("feature_block_ids"),
            field="run_config.feature_block_ids",
        ),
        "train_weeks": _require_week_sequence(
            fold_arrays.get("train_weeks"), field="train_weeks"
        ),
        "validation_weeks": _require_week_sequence(
            fold_arrays.get("validation_weeks"),
            field="validation_weeks",
        ),
        "training_game_ids": _require_text_sequence(
            fold_arrays.get("training_game_ids"),
            field="training_game_ids",
        ),
        "validation_game_ids": _require_text_sequence(
            fold_arrays.get("validation_game_ids"),
            field="validation_game_ids",
        ),
        "preprocessor_fit_game_ids": _require_text_sequence(
            fold_arrays.get("preprocessor_fit_game_ids"),
            field="preprocessor_fit_game_ids",
        ),
        **values,
    }
    if set(row) != set(SOURCE_PROVENANCE_ANCHOR_COLUMNS):
        raise X15SelectionBatchV3Error(
            "source provenance anchor row has wrong columns"
        )
    return row


def _normalize_source_provenance_anchor_row(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(
        SOURCE_PROVENANCE_ANCHOR_COLUMNS
    ):
        raise X15SelectionBatchV3Error(
            "source provenance anchor row has wrong columns"
        )
    row = dict(value)
    for column in (
        "fold_id",
        "model_id",
        "feature_block_id",
        "shard_manifest_path",
    ):
        if type(row[column]) is not str or not row[column].strip():
            raise X15SelectionBatchV3Error(
                f"source provenance anchor {column} must be text"
            )
    for column, logical_type in SOURCE_PROVENANCE_ANCHOR_COLUMN_TYPES:
        if logical_type == "sha256":
            row[column] = _require_sha256(
                row[column],
                field=f"source provenance anchor.{column}",
            )
        elif logical_type == "list<utf8>":
            row[column] = _require_text_sequence(
                row[column],
                field=f"source provenance anchor.{column}",
            )
        elif logical_type == "list<int64>":
            row[column] = _require_week_sequence(
                row[column],
                field=f"source provenance anchor.{column}",
            )
    cell = (
        row["fold_id"],
        row["model_id"],
        row["feature_block_id"],
    )
    if cell not in EXACT_SELECTION_CELLS_V3:
        raise X15SelectionBatchV3Error(
            "source provenance anchor row leaves exact-45"
        )
    return row


def _build_source_provenance_anchor(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    normalized = [
        _normalize_source_provenance_anchor_row(dict(row))
        for row in rows
    ]
    by_cell: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in normalized:
        cell = (
            str(row["fold_id"]),
            str(row["model_id"]),
            str(row["feature_block_id"]),
        )
        if cell in by_cell:
            raise X15SelectionBatchV3Error(
                f"source provenance anchor duplicates {cell}"
            )
        by_cell[cell] = row
    ordered = tuple(
        by_cell[cell]
        for cell in EXACT_SELECTION_CELLS_V3
        if cell in by_cell
    )
    if len(ordered) != len(by_cell):
        raise X15SelectionBatchV3Error(
            "source provenance anchor cannot be canonically ordered"
        )
    material: dict[str, object] = {
        "schema": SOURCE_PROVENANCE_ANCHOR_SCHEMA,
        "column_types": SOURCE_PROVENANCE_ANCHOR_COLUMN_TYPES,
        "row_count": len(ordered),
        "rows": ordered,
    }
    return {
        **material,
        "table_sha256": _canonical_sha256(material),
    }


def _verify_source_provenance_anchor(
    *,
    observed: object,
    rederived_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(observed, dict):
        raise X15SelectionBatchV3Error(
            "source provenance anchor is missing or malformed"
        )
    material = dict(observed)
    observed_sha = _require_sha256(
        material.pop("table_sha256", None),
        field="source provenance anchor.table_sha256",
    )
    if _canonical_sha256(material) != observed_sha:
        raise X15SelectionBatchV3Error(
            "source provenance anchor typed-table SHA-256 mismatch"
        )
    rebuilt_observed = _build_source_provenance_anchor(
        material.get("rows", ())
        if isinstance(material.get("rows"), list)
        else ()
    )
    if _canonical(rebuilt_observed) != _canonical(observed):
        raise X15SelectionBatchV3Error(
            "source provenance anchor typed schema or order drifted"
        )
    rederived = _build_source_provenance_anchor(rederived_rows)
    if _canonical(rederived) != _canonical(observed):
        raise X15SelectionBatchV3Error(
            "source provenance anchor differs from immutable V4 source"
        )
    return rederived


def verify_x15_selection_support_v3(
    *,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    project_root: str | Path,
    panel_batch_manifest_path: str | Path,
    panel_batch_file_sha256: str,
    feature_support_audit_path: str | Path,
    feature_support_audit_file_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify PanelV2 and the required D4 support audit before fitting."""

    project = Path(project_root).resolve()
    canonical_authority = _canonical_authority(authority)
    mapping_sha = _require_sha256(
        cohort_mapping_sha256, field="cohort_mapping_sha256"
    )
    panel, panel_path, panel_file_sha = _read_panel_binding(
        project_root=project,
        panel_batch_manifest_path=panel_batch_manifest_path,
        panel_batch_file_sha256=panel_batch_file_sha256,
        authority=canonical_authority,
        cohort_mapping_sha256=mapping_sha,
    )
    audit, _, _ = _read_support_audit(
        project_root=project,
        feature_support_audit_path=feature_support_audit_path,
        feature_support_audit_file_sha256=(
            feature_support_audit_file_sha256
        ),
        panel_document=panel,
        panel_path=panel_path,
        panel_file_sha256=panel_file_sha,
        authority=canonical_authority,
        cohort_mapping_sha256=mapping_sha,
    )
    return panel, audit


def _shard_reference(
    *,
    document: Mapping[str, object],
    path: Path,
    file_sha256: str,
    output_root: Path,
) -> dict[str, object]:
    tables = document.get("tables")
    if not isinstance(tables, list):
        raise X15SelectionBatchV3Error(
            "V4 shard table descriptors are malformed"
        )
    return {
        "fold_id": document["fold_id"],
        "model_id": document["model_id"],
        "feature_block_id": document["feature_block_id"],
        "manifest_path": path.relative_to(output_root).as_posix(),
        "manifest_sha256": file_sha256,
        "shard_bundle_sha256": document["bundle_sha256"],
        "run_config_sha256": document["run_config_sha256"],
        "shared_run_config_sha256": document[
            "shared_run_config_sha256"
        ],
        "effective_seed_contract_sha256": document[
            "effective_seed_contract_sha256"
        ],
        "selection_fold_arrays_sha256": document[
            "selection_fold_arrays_sha256"
        ],
        "tables": [
            {
                field: table[field]
                for field in (
                    "name",
                    "object_path",
                    "object_sha256",
                    "byte_length",
                    "row_count",
                    "schema_columns",
                    "schema_fingerprint",
                    "semantic_rows_sha256",
                    "format",
                )
            }
            for table in tables
            if isinstance(table, dict)
        ],
    }


def _selection_contract_material(
    *,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    shared_run_config_sha256: str,
    effective_seed_contract_sha256: str,
    panel_binding: Mapping[str, object],
    support_binding: Mapping[str, object],
    root_contract: Mapping[str, object],
) -> dict[str, object]:
    return {
        "contract": "X15_EXACT45_SELECTION_V3",
        "expected_execution_cells": EXACT_SELECTION_CELLS_V3,
        "authority_development_games": authority.development_games,
        "authority_metadata_sha256": authority.metadata_sha256,
        "cohort_authority_sha256": authority.cohort_authority_sha256,
        "cohort_mapping_sha256": cohort_mapping_sha256,
        "shared_run_config_sha256": shared_run_config_sha256,
        "effective_seed_contract_sha256": (
            effective_seed_contract_sha256
        ),
        "panel_batch": dict(panel_binding),
        "feature_support_audit": dict(support_binding),
        "root_contract": dict(root_contract),
        "unsupported_feature_blocks": {
            "D4": "UNSUPPORTED_FEATURE_BLOCK"
        },
        "holdout_reaction_accessed": False,
    }


def publish_x15_selection_batch_v3(
    *,
    shard_manifest_paths: Sequence[str | Path],
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    output_root: str | Path,
    project_root: str | Path,
    panel_batch_manifest_path: str | Path,
    panel_batch_file_sha256: str,
    feature_support_audit_path: str | Path,
    feature_support_audit_file_sha256: str,
) -> X15SelectionBatchV3Publication:
    """Publish a partial or exact-45 batch after full V4 re-verification."""

    root = Path(output_root).resolve()
    project = Path(project_root).resolve()
    root_contract = _require_output_root_contract(
        project_root=project, output_root=root
    )
    canonical_authority = _canonical_authority(authority)
    mapping_sha = _require_sha256(
        cohort_mapping_sha256, field="cohort_mapping_sha256"
    )
    panel, panel_path, panel_file_sha = _read_panel_binding(
        project_root=project,
        panel_batch_manifest_path=panel_batch_manifest_path,
        panel_batch_file_sha256=panel_batch_file_sha256,
        authority=canonical_authority,
        cohort_mapping_sha256=mapping_sha,
    )
    audit, audit_path, audit_file_sha = _read_support_audit(
        project_root=project,
        feature_support_audit_path=feature_support_audit_path,
        feature_support_audit_file_sha256=(
            feature_support_audit_file_sha256
        ),
        panel_document=panel,
        panel_path=panel_path,
        panel_file_sha256=panel_file_sha,
        authority=canonical_authority,
        cohort_mapping_sha256=mapping_sha,
    )
    if not shard_manifest_paths:
        raise X15SelectionBatchV3Error(
            "at least one V4 shard manifest is required"
        )

    records: dict[
        tuple[str, str, str],
        tuple[dict[str, object], Path, str],
    ] = {}
    for manifest_path in shard_manifest_paths:
        cell, document, path, file_sha = _read_v4_manifest(
            manifest_path=manifest_path,
            output_root=root,
            authority=canonical_authority,
            cohort_mapping_sha256=mapping_sha,
        )
        if cell in records:
            raise X15SelectionBatchV3Error(
                f"duplicate execution coordinate: {cell}"
            )
        records[cell] = (document, path, file_sha)

    observed_cells = tuple(
        cell for cell in EXACT_SELECTION_CELLS_V3 if cell in records
    )
    if len(observed_cells) != len(records):
        raise X15SelectionBatchV3Error(
            "batch contains cells outside exact-45"
        )
    missing_cells = tuple(
        cell for cell in EXACT_SELECTION_CELLS_V3 if cell not in records
    )
    shared_hashes = {
        str(document["shared_run_config_sha256"])
        for document, _, _ in records.values()
    }
    seed_hashes = {
        str(document["effective_seed_contract_sha256"])
        for document, _, _ in records.values()
    }
    if len(shared_hashes) != 1:
        raise X15SelectionBatchV3Error(
            "V4 shards disagree on shared run config"
        )
    if len(seed_hashes) != 1:
        raise X15SelectionBatchV3Error(
            "V4 shards disagree on effective seed contract"
        )
    shared_hash = next(iter(shared_hashes))
    seed_hash = next(iter(seed_hashes))
    references = [
        _shard_reference(
            document=records[cell][0],
            path=records[cell][1],
            file_sha256=records[cell][2],
            output_root=root,
        )
        for cell in observed_cells
    ]
    provenance_anchor = _build_source_provenance_anchor(
        tuple(
            _source_provenance_anchor_row(
                document=records[cell][0],
                path=records[cell][1],
                file_sha256=records[cell][2],
                output_root=root,
            )
            for cell in observed_cells
        )
    )
    panel_binding = {
        "manifest_path": panel_path.relative_to(project).as_posix(),
        "file_sha256": panel_file_sha,
        "batch_sha256": panel["batch_sha256"],
    }
    support_binding = {
        "manifest_path": audit_path.relative_to(project).as_posix(),
        "file_sha256": audit_file_sha,
        "audit_sha256": audit["audit_sha256"],
        "decision_eligible_rows": EXPECTED_DECISION_ELIGIBLE_ROWS,
        "nonnull_counts": {
            field: 0 for field in REQUIRED_ZERO_NONNULL_FIELDS
        },
        "feature_block_status": {
            "D4": "UNSUPPORTED_FEATURE_BLOCK"
        },
        "selection_eligible": {"D4": False},
        "support_reason": {"D4": UNSUPPORTED_D4_REASON},
        "claim_boundary": SUPPORT_AUDIT_CLAIM_BOUNDARY,
        "holdout_reaction_accessed": False,
    }
    selection_contract = _selection_contract_material(
        authority=canonical_authority,
        cohort_mapping_sha256=mapping_sha,
        shared_run_config_sha256=shared_hash,
        effective_seed_contract_sha256=seed_hash,
        panel_binding=panel_binding,
        support_binding=support_binding,
        root_contract=root_contract,
    )
    selection_contract_sha = _canonical_sha256(selection_contract)
    selection_ready = not missing_cells
    material: dict[str, object] = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-15",
        "cohort": "development",
        "authority_game_count": canonical_authority.game_count,
        "authority_development_games": (
            canonical_authority.development_games
        ),
        "authority_metadata_sha256": (
            canonical_authority.metadata_sha256
        ),
        "cohort_authority_sha256": (
            canonical_authority.cohort_authority_sha256
        ),
        "cohort_mapping_sha256": mapping_sha,
        "panel_batch": panel_binding,
        "feature_support_audit": support_binding,
        "root_contract": root_contract,
        "shared_run_config_sha256": shared_hash,
        "effective_seed_contract_sha256": seed_hash,
        "selection_contract_sha256": selection_contract_sha,
        "expected_fold_ids": EXPECTED_FOLD_IDS,
        "exact_design_matrix": EXACT_DESIGN_MATRIX_V3,
        "selectable_candidates": SELECTABLE_CANDIDATES_V3,
        "unsupported_feature_blocks": {
            "D4": "UNSUPPORTED_FEATURE_BLOCK"
        },
        "expected_execution_cell_count": 45,
        "observed_execution_cell_count": len(observed_cells),
        "missing_execution_cell_count": len(missing_cells),
        "expected_execution_cells": EXACT_SELECTION_CELLS_V3,
        "observed_execution_cells": observed_cells,
        "missing_execution_cells": missing_cells,
        "selection_ready": selection_ready,
        "publication_status": (
            "SELECTION_READY"
            if selection_ready
            else "PARTIAL_DIAGNOSTIC_ONLY"
        ),
        "holdout_reaction_accessed": False,
        "shards": references,
        "source_provenance_anchor": provenance_anchor,
    }
    path, manifest_sha, batch_sha = _publish_manifest(
        output_root=root, material=material
    )
    return X15SelectionBatchV3Publication(
        manifest_path=path,
        manifest_sha256=manifest_sha,
        batch_sha256=batch_sha,
        selection_contract_sha256=selection_contract_sha,
        observed_execution_cell_count=len(observed_cells),
        selection_ready=selection_ready,
    )


def _execution_cells(
    value: object, *, field: str
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        raise X15SelectionBatchV3Error(f"{field} must be an array")
    cells: list[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            raise X15SelectionBatchV3Error(
                f"{field} contains a malformed coordinate"
            )
        cells.append(tuple(map(str, item)))
    return tuple(cells)


def _read_verified_batch(
    *,
    batch_manifest_path: str | Path,
    output_root: Path,
    project_root: Path,
) -> tuple[
    dict[str, object],
    Path,
    str,
    FrozenDevelopmentAuthority,
]:
    path = _resolve_under(
        output_root, batch_manifest_path, field="selection batch manifest"
    )
    observed_file_sha, _ = sha256_path(path)
    _verify_content_addressed_path(
        path,
        expected_file_sha256=observed_file_sha,
        suffix=".batch-index.json",
        field="selection batch manifest",
    )
    document = _strict_json(
        path.read_bytes(), field="selection batch manifest"
    )
    material = dict(document)
    batch_sha = _require_sha256(
        material.pop("batch_sha256", None),
        field="selection batch.batch_sha256",
    )
    if _canonical_sha256(material) != batch_sha:
        raise X15SelectionBatchV3Error(
            "selection batch semantic SHA-256 mismatch"
        )
    if (
        document.get("schema") != SCHEMA
        or document.get("builder_version") != BUILDER_VERSION
        or document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or document.get("selection_ready") is not True
        or document.get("publication_status") != "SELECTION_READY"
        or document.get("holdout_reaction_accessed") is not False
        or document.get("expected_execution_cell_count") != 45
        or document.get("observed_execution_cell_count") != 45
        or document.get("missing_execution_cell_count") != 0
    ):
        raise X15SelectionBatchV3Error(
            "verified load requires an exact-45 selection-ready batch"
        )
    authority = _authority_from_document(document)
    root_contract = _verify_root_contract(
        document=document,
        project_root=project_root,
        output_root=output_root,
    )
    mapping_sha = _require_sha256(
        document.get("cohort_mapping_sha256"),
        field="selection batch.cohort_mapping_sha256",
    )
    panel_binding = document.get("panel_batch")
    support_binding = document.get("feature_support_audit")
    if not isinstance(panel_binding, dict) or not isinstance(
        support_binding, dict
    ):
        raise X15SelectionBatchV3Error(
            "selection batch external bindings are malformed"
        )
    panel, panel_path, panel_file_sha = _read_panel_binding(
        project_root=project_root,
        panel_batch_manifest_path=str(
            panel_binding.get("manifest_path", "")
        ),
        panel_batch_file_sha256=str(
            panel_binding.get("file_sha256", "")
        ),
        authority=authority,
        cohort_mapping_sha256=mapping_sha,
    )
    if (
        panel.get("batch_sha256")
        != panel_binding.get("batch_sha256")
    ):
        raise X15SelectionBatchV3Error(
            "selection batch panel binding drifted"
        )
    audit, audit_path, audit_file_sha = _read_support_audit(
        project_root=project_root,
        feature_support_audit_path=str(
            support_binding.get("manifest_path", "")
        ),
        feature_support_audit_file_sha256=str(
            support_binding.get("file_sha256", "")
        ),
        panel_document=panel,
        panel_path=panel_path,
        panel_file_sha256=panel_file_sha,
        authority=authority,
        cohort_mapping_sha256=mapping_sha,
    )
    expected_support_binding = {
        "manifest_path": audit_path.relative_to(
            project_root
        ).as_posix(),
        "file_sha256": audit_file_sha,
        "audit_sha256": audit["audit_sha256"],
        "decision_eligible_rows": EXPECTED_DECISION_ELIGIBLE_ROWS,
        "nonnull_counts": {
            field: 0 for field in REQUIRED_ZERO_NONNULL_FIELDS
        },
        "feature_block_status": {
            "D4": "UNSUPPORTED_FEATURE_BLOCK"
        },
        "selection_eligible": {"D4": False},
        "support_reason": {"D4": UNSUPPORTED_D4_REASON},
        "claim_boundary": SUPPORT_AUDIT_CLAIM_BOUNDARY,
        "holdout_reaction_accessed": False,
    }
    if _canonical(support_binding) != _canonical(
        expected_support_binding
    ):
        raise X15SelectionBatchV3Error(
            "selection batch feature support binding drifted"
        )
    expected_panel_binding = {
        "manifest_path": panel_path.relative_to(
            project_root
        ).as_posix(),
        "file_sha256": panel_file_sha,
        "batch_sha256": panel["batch_sha256"],
    }
    if _canonical(panel_binding) != _canonical(expected_panel_binding):
        raise X15SelectionBatchV3Error(
            "selection batch panel binding drifted"
        )
    if (
        _execution_cells(
            document.get("expected_execution_cells"),
            field="expected_execution_cells",
        )
        != EXACT_SELECTION_CELLS_V3
        or _execution_cells(
            document.get("observed_execution_cells"),
            field="observed_execution_cells",
        )
        != EXACT_SELECTION_CELLS_V3
        or _execution_cells(
            document.get("missing_execution_cells"),
            field="missing_execution_cells",
        )
        != ()
        or _tuple(document.get("expected_fold_ids"))
        != EXPECTED_FOLD_IDS
        or _canonical(document.get("exact_design_matrix"))
        != _canonical(EXACT_DESIGN_MATRIX_V3)
        or _canonical(document.get("selectable_candidates"))
        != _canonical(SELECTABLE_CANDIDATES_V3)
        or _canonical(document.get("unsupported_feature_blocks"))
        != _canonical({"D4": "UNSUPPORTED_FEATURE_BLOCK"})
    ):
        raise X15SelectionBatchV3Error(
            "selection batch exact-45 matrix drifted"
        )
    shared_hash = _require_sha256(
        document.get("shared_run_config_sha256"),
        field="selection batch.shared_run_config_sha256",
    )
    seed_hash = _require_sha256(
        document.get("effective_seed_contract_sha256"),
        field="selection batch.effective_seed_contract_sha256",
    )
    contract = _selection_contract_material(
        authority=authority,
        cohort_mapping_sha256=mapping_sha,
        shared_run_config_sha256=shared_hash,
        effective_seed_contract_sha256=seed_hash,
        panel_binding=expected_panel_binding,
        support_binding=expected_support_binding,
        root_contract=root_contract,
    )
    if _canonical_sha256(contract) != document.get(
        "selection_contract_sha256"
    ):
        raise X15SelectionBatchV3Error(
            "selection contract SHA-256 mismatch"
        )
    references = document.get("shards")
    if not isinstance(references, list) or len(references) != 45:
        raise X15SelectionBatchV3Error(
            "exact-45 batch must reference 45 V4 shards"
        )
    indexed_cells: list[tuple[str, str, str]] = []
    rederived_provenance_rows: list[dict[str, object]] = []
    for reference in references:
        if not isinstance(reference, dict):
            raise X15SelectionBatchV3Error(
                "selection shard reference is malformed"
            )
        cell, shard_document, shard_path, shard_file_sha = (
            _read_v4_manifest(
                manifest_path=str(reference.get("manifest_path", "")),
                output_root=output_root,
                authority=authority,
                cohort_mapping_sha256=mapping_sha,
            )
        )
        expected_reference = _shard_reference(
            document=shard_document,
            path=shard_path,
            file_sha256=shard_file_sha,
            output_root=output_root,
        )
        if _canonical(reference) != _canonical(expected_reference):
            raise X15SelectionBatchV3Error(
                f"selection shard reference drifted for {cell}"
            )
        if (
            reference.get("shared_run_config_sha256") != shared_hash
            or reference.get("effective_seed_contract_sha256")
            != seed_hash
        ):
            raise X15SelectionBatchV3Error(
                f"selection shard config/seed drifted for {cell}"
            )
        indexed_cells.append(cell)
        rederived_provenance_rows.append(
            _source_provenance_anchor_row(
                document=shard_document,
                path=shard_path,
                file_sha256=shard_file_sha,
                output_root=output_root,
            )
        )
    if tuple(indexed_cells) != EXACT_SELECTION_CELLS_V3:
        raise X15SelectionBatchV3Error(
            "selection shard index is not exact-45 canonical order"
        )
    _verify_source_provenance_anchor(
        observed=document.get("source_provenance_anchor"),
        rederived_rows=rederived_provenance_rows,
    )
    return document, path, observed_file_sha, authority


def _selection_output_root_from_batch(
    *,
    batch_manifest_path: str | Path,
    artifact_root: Path,
) -> Path:
    path = _resolve_under(
        artifact_root,
        batch_manifest_path,
        field="selection batch manifest",
    )
    for parent in path.parents:
        if parent.name == "selection-v3":
            return parent.parent
    raise X15SelectionBatchV3Error(
        "selection batch path is outside the selection-v3 namespace"
    )


def load_verified_x15_selection_batch_v3(
    *,
    batch_manifest_path: str | Path,
    artifact_root: str | Path,
) -> tuple[
    dict[str, object],
    Path,
    str,
    FrozenDevelopmentAuthority,
]:
    """Load and fully re-verify one exact-45 selection batch."""

    artifacts = Path(artifact_root).resolve()
    return _read_verified_batch(
        batch_manifest_path=batch_manifest_path,
        output_root=_selection_output_root_from_batch(
            batch_manifest_path=batch_manifest_path,
            artifact_root=artifacts,
        ),
        project_root=artifacts,
    )


def load_x15_selection_projection_v3(
    *,
    batch_manifest_path: str | Path,
    artifact_root: str | Path,
    candidate_model_id: str,
    candidate_feature_block_id: str,
) -> X15ModelRun:
    """Load baseline plus one supported candidate across all five folds."""

    candidate = (candidate_model_id, candidate_feature_block_id)
    if candidate not in SELECTABLE_CANDIDATES_V3:
        raise X15SelectionBatchV3Error(
            "candidate must be a supported D1-D3 selectable design"
        )
    artifacts = Path(artifact_root).resolve()
    root = _selection_output_root_from_batch(
        batch_manifest_path=batch_manifest_path,
        artifact_root=artifacts,
    )
    document, batch_path, batch_file_sha, _ = _read_verified_batch(
        batch_manifest_path=batch_manifest_path,
        output_root=root,
        project_root=artifacts,
    )
    references = document["shards"]
    if not isinstance(references, list):
        raise X15SelectionBatchV3Error(
            "selection shard references are malformed"
        )
    by_cell = {
        (
            str(reference["fold_id"]),
            str(reference["model_id"]),
            str(reference["feature_block_id"]),
        ): reference
        for reference in references
        if isinstance(reference, dict)
    }
    batch_anchor = document.get("source_provenance_anchor")
    if not isinstance(batch_anchor, dict) or not isinstance(
        batch_anchor.get("rows"), list
    ):
        raise X15SelectionBatchV3Error(
            "selection projection lacks source provenance anchor"
        )
    anchor_by_cell = {
        (
            str(row["fold_id"]),
            str(row["model_id"]),
            str(row["feature_block_id"]),
        ): _normalize_source_provenance_anchor_row(row)
        for row in batch_anchor["rows"]
        if isinstance(row, dict)
    }
    frames: list[pd.DataFrame] = []
    source_shards: list[dict[str, object]] = []
    selected_provenance_rows: list[dict[str, object]] = []
    common_run_config: dict[str, object] | None = None
    for fold_id in EXPECTED_FOLD_IDS:
        for model_id, feature_block_id in (
            ("b0_empirical_v1", "D0"),
            candidate,
        ):
            cell = (fold_id, model_id, feature_block_id)
            reference = by_cell.get(cell)
            if reference is None:
                raise X15SelectionBatchV3Error(
                    f"selection projection lacks {cell}"
                )
            run = load_x15_oof_shard(
                manifest_path=str(reference["manifest_path"]),
                output_root=root,
            )
            anchor_row = anchor_by_cell.get(cell)
            if anchor_row is None:
                raise X15SelectionBatchV3Error(
                    f"selection provenance anchor lacks {cell}"
                )
            loaded_provenance = _unique_oof_provenance_from_frame(
                run.oof_predictions
            )
            expected_loaded_provenance = {
                column: anchor_row[column]
                for column in (
                    *SOURCE_PROVENANCE_HASH_COLUMNS,
                    *SOURCE_PROVENANCE_CALIBRATION_COLUMNS,
                )
            }
            if _canonical(loaded_provenance) != _canonical(
                expected_loaded_provenance
            ):
                raise X15SelectionBatchV3Error(
                    f"loaded OOF provenance differs from anchor for {cell}"
                )
            if (
                run.run_config_sha256
                != anchor_row["run_config_sha256"]
                or _tuple(run.run_config.get("fold_ids"))
                != tuple(anchor_row["run_config_fold_ids"])
                or _tuple(run.run_config.get("model_ids"))
                != tuple(anchor_row["run_config_model_ids"])
                or _tuple(run.run_config.get("feature_block_ids"))
                != tuple(
                    anchor_row["run_config_feature_block_ids"]
                )
            ):
                raise X15SelectionBatchV3Error(
                    f"loaded OOF run provenance differs for {cell}"
                )
            shard_common = dict(run.run_config)
            for execution_field in (
                "fold_ids",
                "model_ids",
                "feature_block_ids",
            ):
                shard_common.pop(execution_field, None)
            # A projection contains native Polymarket rows only.  Transport
            # pairs are therefore intentionally empty even though the source
            # computation shard also produced a Kalshi transport diagnostic.
            shard_common["transport_pairs"] = ()
            if common_run_config is None:
                common_run_config = shard_common
            elif _canonical(common_run_config) != _canonical(shard_common):
                raise X15SelectionBatchV3Error(
                    "projected V4 shards disagree on common run config"
                )
            frame = run.oof_predictions
            mask = (
                frame["venue"].eq("polymarket")
                & frame["training_venue"].eq("polymarket")
                & frame["calibration_venue"].eq("polymarket")
                & frame["transport_mode"].eq("VENUE_SPECIFIC")
            )
            projection = frame.loc[
                mask, list(V3_SELECTION_PROJECTION_COLUMNS)
            ].copy()
            if projection.empty:
                raise X15SelectionBatchV3Error(
                    f"selection projection has no native Polymarket rows {cell}"
                )
            frames.append(projection)
            selected_provenance_rows.append(anchor_row)
            source_shards.append(
                {
                    "fold_id": fold_id,
                    "model_id": model_id,
                    "feature_block_id": feature_block_id,
                    "manifest_sha256": reference["manifest_sha256"],
                    "shard_bundle_sha256": reference[
                        "shard_bundle_sha256"
                    ],
                    "oof_predictions_sha256": next(
                        table["object_sha256"]
                        for table in reference["tables"]
                        if table["name"] == "oof_predictions"
                    ),
                    "source_provenance_anchor_row": anchor_row,
                }
            )
    predictions = pd.concat(frames, ignore_index=True)
    if common_run_config is None:
        raise X15SelectionBatchV3Error(
            "selection projection has no common V4 run config"
        )
    projected_provenance_anchor = _build_source_provenance_anchor(
        selected_provenance_rows
    )
    for source_shard in source_shards:
        source_shard["source_provenance_anchor_schema"] = (
            SOURCE_PROVENANCE_ANCHOR_SCHEMA
        )
        source_shard["source_provenance_anchor_table_sha256"] = (
            projected_provenance_anchor["table_sha256"]
        )
    run_config = {
        **common_run_config,
        "source_batch_manifest_path": batch_path.relative_to(
            root
        ).as_posix(),
        "source_batch_manifest_sha256": batch_file_sha,
        "source_batch_sha256": document["batch_sha256"],
        "selection_contract_sha256": document[
            "selection_contract_sha256"
        ],
        "fold_ids": EXPECTED_FOLD_IDS,
        "model_ids": ("b0_empirical_v1", candidate_model_id),
        "feature_block_ids": ("D0", candidate_feature_block_id),
        "source_shards": tuple(source_shards),
        "selection_projection_schema": (
            "nfl_x15_selection_projection_v3"
        ),
        "holdout_reaction_accessed": False,
    }
    return X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256=_model_sha256(run_config),
        run_config=run_config,
    )


__all__ = [
    "BUILDER_VERSION",
    "EXACT_DESIGN_MATRIX_V3",
    "EXACT_SELECTION_CELLS_V3",
    "SCHEMA",
    "SELECTABLE_CANDIDATES_V3",
    "SOURCE_PROVENANCE_ANCHOR_COLUMN_TYPES",
    "SOURCE_PROVENANCE_ANCHOR_COLUMNS",
    "SOURCE_PROVENANCE_ANCHOR_SCHEMA",
    "SOURCE_PROVENANCE_HASH_COLUMNS",
    "SOURCE_PROVENANCE_CALIBRATION_COLUMNS",
    "V3_SELECTION_PROJECTION_COLUMNS",
    "X15SelectionBatchV3Error",
    "X15SelectionBatchV3Publication",
    "load_verified_x15_selection_batch_v3",
    "load_x15_selection_projection_v3",
    "publish_x15_selection_batch_v3",
    "verify_x15_selection_support_v3",
]
