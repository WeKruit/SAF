"""Content-addressed, resumable publication for NFL X-15 OOF shards.

Each ``fold × model × feature-block`` probability shard is published
independently from an ``X15ModelRun``.  Those shard manifests are diagnostic
checkpoints.  A batch becomes selection-ready only after all 55 frozen
execution cells prove exact alignment with the governed 153-game development
authority.  Weeks 1--2 are training-only; the exact OOF validation union is
the 123 games in weeks 3--12.

This module never resolves or reads final-holdout reaction artifacts.
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
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.immutable_object_store import sha256_path
from prediction_market.research.nfl_x15_model_selection import (
    EXPECTED_DEVELOPMENT_GAME_COUNT,
    EXPECTED_FOLDS,
    FrozenDevelopmentAuthority,
)
from prediction_market.research.nfl_x15_models import (
    DIAGNOSTIC_ANALYSIS_SCOPE,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY,
    DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS,
    DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT,
    DIAGNOSTIC_SCHEMA_VERSION,
    DIAGNOSTIC_TARGET_CONTRACT,
    DIAGNOSTIC_VENUE_TICK_SUPPORT,
    X15ModelRun,
)


SHARD_SCHEMA: Final[str] = "nfl_x15_oof_shard_publication_v2"
BATCH_SCHEMA: Final[str] = "nfl_x15_oof_batch_index_v2"
BUILDER_VERSION: Final[str] = "nfl-x15-oof-publication-v2"
OOF_TABLE_NAMES: Final[tuple[str, ...]] = (
    "oof_predictions",
    "conditional_quantiles",
    "fold_metrics",
    "support_audit",
    "weight_audit",
)
SELECTION_PROJECTION_COLUMNS: Final[tuple[str, ...]] = (
    "source_row_id",
    "game_id",
    "nfl_week",
    "atomic_information_episode_id",
    "venue",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
    "fold_id",
    "cohort_authority_sha256",
    "target_contract",
    "claim_boundary",
    "direction_threshold_probability",
    "direction_threshold_semantics",
    "venue_tick_support",
    "market_continuity_support",
    "schema_version",
    "analysis_scope",
    "claim_eligible",
    "calibration_venue",
    "transport_mode",
    "training_venue",
    "model_id",
    "feature_block_id",
    "s_h_truth",
    "o_h_given_s_truth",
    "direction_truth",
    "s_h_calibrated_probability",
    "o_h_given_s_calibrated_probability",
    "direction_calibrated_prob_down",
    "direction_calibrated_prob_no_move",
    "direction_calibrated_prob_up",
    "train_weeks",
    "validation_weeks",
    "training_game_ids",
    "validation_game_ids",
    "preprocessor_fit_game_ids",
)
_FOLD_ARRAY_COLUMNS: Final[tuple[str, ...]] = (
    "train_weeks",
    "validation_weeks",
    "training_game_ids",
    "validation_game_ids",
    "preprocessor_fit_game_ids",
)
_SELECTION_PARQUET_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in SELECTION_PROJECTION_COLUMNS
    if column not in _FOLD_ARRAY_COLUMNS
)
_SELECTION_FILTERS: Final[list[tuple[str, str, str]]] = [
    ("venue", "=", "polymarket"),
    ("training_venue", "=", "polymarket"),
    ("transport_mode", "=", "VENUE_SPECIFIC"),
]
_PREDICTION_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *SELECTION_PROJECTION_COLUMNS,
        "run_config_sha256",
    }
)
_SELECTION_PAIR_GRAIN: Final[tuple[str, ...]] = (
    "source_row_id",
    "game_id",
    "nfl_week",
    "atomic_information_episode_id",
    "venue",
    "training_venue",
    "calibration_venue",
    "transport_mode",
    "actual_home_contract_id",
    "landmark_seconds",
    "endpoint_seconds",
)
_IDENTITY_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "atomic_information_episode_id",
    "venue",
    "training_venue",
    "calibration_venue",
    "transport_mode",
    "actual_home_contract_id",
    "fold_id",
    "model_id",
    "feature_block_id",
    "cohort_authority_sha256",
    "target_contract",
    "claim_boundary",
    "direction_threshold_semantics",
    "venue_tick_support",
    "market_continuity_support",
    "schema_version",
    "analysis_scope",
)
_BINARY_TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    "s_h_truth",
    "o_h_given_s_truth",
)
_BINARY_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "s_h_calibrated_probability",
    "o_h_given_s_calibrated_probability",
)
_DIRECTION_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "direction_calibrated_prob_down",
    "direction_calibrated_prob_no_move",
    "direction_calibrated_prob_up",
)
_DIRECTION_CLASSES: Final[frozenset[str]] = frozenset(
    {"DOWN", "NO_MOVE", "UP"}
)
_EFFECTIVE_SEED_FORMULA: Final[str] = (
    "base_random_state + fold_index*1000 + unit_index*100 "
    "+ feature_block_index*10 + model_index"
)
_SHA256_PREFIX: Final[str] = "sha256:"
_EXPECTED_TRAINING_ONLY_GAME_COUNT: Final[int] = 30
_EXPECTED_OOF_VALIDATION_GAME_COUNT: Final[int] = 123
BASELINE_DESIGN_CELL: Final[tuple[str, str]] = (
    "b0_empirical_v1",
    "D0",
)
NEGATIVE_CONTROL_DESIGN_MATRIX: Final[
    tuple[tuple[str, str], ...]
] = (
    ("regularized_logistic_v1", "D0"),
    ("shallow_xgboost_v1", "D0"),
)
SELECTABLE_CANDIDATE_DESIGN_MATRIX: Final[
    tuple[tuple[str, str], ...]
] = (
    ("regularized_logistic_v1", "D1"),
    ("regularized_logistic_v1", "D2"),
    ("regularized_logistic_v1", "D3"),
    ("regularized_logistic_v1", "D4"),
    ("shallow_xgboost_v1", "D1"),
    ("shallow_xgboost_v1", "D2"),
    ("shallow_xgboost_v1", "D3"),
    ("shallow_xgboost_v1", "D4"),
)
PROBABILITY_DESIGN_MATRIX: Final[tuple[tuple[str, str], ...]] = (
    BASELINE_DESIGN_CELL,
    NEGATIVE_CONTROL_DESIGN_MATRIX[0],
    *SELECTABLE_CANDIDATE_DESIGN_MATRIX[:4],
    NEGATIVE_CONTROL_DESIGN_MATRIX[1],
    *SELECTABLE_CANDIDATE_DESIGN_MATRIX[4:],
)
_FOLD_BY_ID: Final[
    dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
] = {
    fold_id: (tuple(train_weeks), tuple(validation_weeks))
    for fold_id, train_weeks, validation_weeks in EXPECTED_FOLDS
}
_DIAGNOSTIC_CONFIG: Final[dict[str, object]] = {
    "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    "target_contract": DIAGNOSTIC_TARGET_CONTRACT,
    "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
    "analysis_scope": DIAGNOSTIC_ANALYSIS_SCOPE,
    "direction_threshold_probability": (
        DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
    ),
    "direction_threshold_semantics": (
        DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS
    ),
    "venue_tick_support": DIAGNOSTIC_VENUE_TICK_SUPPORT,
    "market_continuity_support": DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT,
    "claim_eligible": False,
}
_PREDICTION_FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    "model_id",
    "feature_block_id",
    "game_id",
    "cohort_authority_sha256",
    "train_weeks",
    "validation_weeks",
    "training_game_ids",
    "validation_game_ids",
    "preprocessor_fit_game_ids",
)


class X15OOFPublicationError(RuntimeError):
    """An OOF shard or batch failed a governed publication invariant."""


def _selection_role(model_id: str, feature_block_id: str) -> str:
    cell = (model_id, feature_block_id)
    if cell == BASELINE_DESIGN_CELL:
        return "BASELINE"
    if cell in NEGATIVE_CONTROL_DESIGN_MATRIX:
        return "NEGATIVE_CONTROL_TIME_GRID_ONLY_NOT_SELECTABLE"
    if cell in SELECTABLE_CANDIDATE_DESIGN_MATRIX:
        return "SELECTABLE_CANDIDATE"
    raise X15OOFPublicationError(
        "execution cell is outside the frozen probability design matrix"
    )


@dataclass(frozen=True, slots=True)
class X15OOFShardPublication:
    fold_id: str
    model_id: str
    feature_block_id: str
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    run_config_sha256: str
    authority_coverage_complete: bool
    holdout_reaction_accessed: bool = False


@dataclass(frozen=True, slots=True)
class X15OOFBatchPublication:
    manifest_path: Path
    manifest_sha256: str
    batch_sha256: str
    observed_fold_ids: tuple[str, ...]
    selection_ready: bool
    holdout_reaction_accessed: bool = False


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_SHA256_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise X15OOFPublicationError(f"{field} must be a SHA-256 digest")
    return value


def _canonical(value: object) -> object:
    """Match the X-15 model runner's canonical run-config encoding."""

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


def _model_sha256(value: object) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


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


def _sha256_bytes(payload: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _strict_json(payload: bytes, *, field: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise X15OOFPublicationError(
                    f"{field} contains duplicate JSON key {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise X15OOFPublicationError(f"{field} is not valid JSON") from error
    if type(value) is not dict:
        raise X15OOFPublicationError(f"{field} must be a JSON object")
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
        raise X15OOFPublicationError(f"{field} escapes output_root")
    return candidate


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise X15OOFPublicationError(
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
                raise X15OOFPublicationError(
                    f"content-addressed publish race: {path}"
                )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _content_addressed_path(
    *,
    output_root: Path,
    namespace: Path,
    payload_sha256: str,
    suffix: str,
) -> Path:
    digest = _require_sha256(
        payload_sha256, field="payload_sha256"
    ).removeprefix(_SHA256_PREFIX)
    return (
        output_root
        / namespace
        / "sha256"
        / digest[:2]
        / f"{digest}{suffix}"
    )


def _verify_content_addressed_filename(
    path: Path,
    *,
    suffix: str,
    field: str,
) -> str:
    if not path.name.endswith(suffix):
        raise X15OOFPublicationError(
            f"{field} path is not content-addressed"
        )
    hexadecimal = path.name[: -len(suffix)]
    if (
        len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
        or path.parent.name != hexadecimal[:2]
        or path.parent.parent.name != "sha256"
    ):
        raise X15OOFPublicationError(
            f"{field} path is not content-addressed"
        )
    return _SHA256_PREFIX + hexadecimal


def _validate_authority(
    authority: FrozenDevelopmentAuthority,
) -> dict[str, int]:
    if not isinstance(authority, FrozenDevelopmentAuthority):
        raise X15OOFPublicationError(
            "authority must be FrozenDevelopmentAuthority"
        )
    if authority.game_count != EXPECTED_DEVELOPMENT_GAME_COUNT:
        raise X15OOFPublicationError(
            "OOF publication requires exact 153-game development authority"
        )
    _require_sha256(
        authority.cohort_authority_sha256,
        field="authority.cohort_authority_sha256",
    )
    if _model_sha256(authority.development_games) != authority.metadata_sha256:
        raise X15OOFPublicationError("authority metadata SHA-256 mismatch")
    game_weeks = authority.game_weeks
    if (
        len(game_weeks) != EXPECTED_DEVELOPMENT_GAME_COUNT
        or set(game_weeks.values()) != set(range(1, 13))
    ):
        raise X15OOFPublicationError(
            "authority must contain unique games across weeks 1..12"
        )
    training_only_count = sum(
        week in (1, 2) for week in game_weeks.values()
    )
    validation_count = sum(
        week in range(3, 13) for week in game_weeks.values()
    )
    if (
        training_only_count != _EXPECTED_TRAINING_ONLY_GAME_COUNT
        or validation_count != _EXPECTED_OOF_VALIDATION_GAME_COUNT
    ):
        raise X15OOFPublicationError(
            "exact-153 authority must bind 30 weeks1-2 training-only "
            "games and 123 weeks3-12 OOF validation games"
        )
    for game_id, week, kickoff_utc, batch_sha256 in (
        authority.development_games
    ):
        if (
            not game_id
            or not kickoff_utc
            or week not in range(1, 13)
        ):
            raise X15OOFPublicationError(
                "authority game metadata is malformed"
            )
        _require_sha256(
            batch_sha256, field=f"authority.{game_id}.batch_sha256"
        )
    return game_weeks


def _tuple_value(value: object) -> tuple[object, ...] | None:
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(value)
    return None


def _execution_cells(
    value: object, *, field: str
) -> tuple[tuple[str, str, str], ...]:
    sequence = _tuple_value(value)
    if sequence is None:
        raise X15OOFPublicationError(f"{field} must be an array")
    cells: list[tuple[str, str, str]] = []
    for item in sequence:
        coordinate = _tuple_value(item)
        if coordinate is None or len(coordinate) != 3:
            raise X15OOFPublicationError(
                f"{field} contains a malformed execution cell"
            )
        cells.append(tuple(map(str, coordinate)))
    return tuple(cells)


def _fold_authority(
    *,
    authority: FrozenDevelopmentAuthority,
    fold_id: str,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    game_weeks = _validate_authority(authority)
    if fold_id not in _FOLD_BY_ID:
        raise X15OOFPublicationError(f"unknown frozen fold_id: {fold_id}")
    train_weeks, validation_weeks = _FOLD_BY_ID[fold_id]
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
        train_weeks,
        validation_weeks,
        training_game_ids,
        validation_game_ids,
    )


def _effective_seed_contract(
    run_config: Mapping[str, object],
) -> dict[str, object]:
    random_state = run_config.get("random_state")
    if (
        not isinstance(random_state, (int, np.integer))
        or isinstance(random_state, (bool, np.bool_))
    ):
        raise X15OOFPublicationError(
            "run config random_state must be an integer"
        )
    base_random_state = int(random_state)
    if not 0 <= base_random_state <= 2**32 - 1:
        raise X15OOFPublicationError(
            "run config random_state is outside the uint32 seed range"
        )
    return {
        "base_random_state": base_random_state,
        "execution_unit_seed_stride": 100,
        "formula": _EFFECTIVE_SEED_FORMULA,
        "shard_fold_index": 0,
        "shard_feature_block_index": 0,
        "shard_model_index": 0,
    }


def _validate_run_config(
    model_run: X15ModelRun,
    *,
    fold_id: str,
    model_id: str,
    feature_block_id: str,
    authority: FrozenDevelopmentAuthority,
) -> tuple[dict[str, object], str]:
    if not isinstance(model_run, X15ModelRun):
        raise X15OOFPublicationError("model_run must be X15ModelRun")
    run_config_sha256 = _require_sha256(
        model_run.run_config_sha256,
        field="model_run.run_config_sha256",
    )
    if not isinstance(model_run.run_config, Mapping):
        raise X15OOFPublicationError(
            "model_run must carry its stamped diagnostic run_config"
        )
    run_config = dict(model_run.run_config)
    if _model_sha256(run_config) != run_config_sha256:
        raise X15OOFPublicationError("run config SHA-256 mismatch")
    for field, expected in _DIAGNOSTIC_CONFIG.items():
        if run_config.get(field) != expected:
            raise X15OOFPublicationError(
                f"diagnostic run config drifted on {field}"
            )
    if _tuple_value(run_config.get("fold_ids")) != (fold_id,):
        raise X15OOFPublicationError(
            "one shard requires run_config.fold_ids=(fold_id,)"
        )
    if _tuple_value(run_config.get("model_ids")) != (model_id,):
        raise X15OOFPublicationError(
            "one shard requires run_config.model_ids=(model_id,)"
        )
    if _tuple_value(run_config.get("feature_block_ids")) != (
        feature_block_id,
    ):
        raise X15OOFPublicationError(
            "one shard requires "
            "run_config.feature_block_ids=(feature_block_id,)"
        )
    if (model_id, feature_block_id) not in PROBABILITY_DESIGN_MATRIX:
        raise X15OOFPublicationError(
            "shard model/feature-block pair is outside the frozen "
            "probability design matrix"
        )
    if run_config.get("include_magnitude") is not False:
        raise X15OOFPublicationError(
            "probability shard requires include_magnitude=False"
        )
    if (
        run_config.get("cohort_authority_sha256")
        != authority.cohort_authority_sha256
    ):
        raise X15OOFPublicationError(
            "run config cohort authority does not match publication authority"
        )
    _effective_seed_contract(run_config)
    shared_config = dict(run_config)
    for execution_field in (
        "fold_ids",
        "model_ids",
        "feature_block_ids",
    ):
        shared_config.pop(execution_field, None)
    return run_config, _model_sha256(shared_config)


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _validate_prediction_semantics(
    predictions: pd.DataFrame,
    *,
    authority: FrozenDevelopmentAuthority,
    fold_id: str,
    model_id: str,
    feature_block_id: str,
    validation_game_ids: tuple[str, ...],
) -> None:
    missing = sorted(
        _PREDICTION_REQUIRED_COLUMNS.difference(predictions.columns)
    )
    if missing:
        raise X15OOFPublicationError(
            f"oof_predictions missing governed columns: {missing}"
        )
    for column in _IDENTITY_TEXT_COLUMNS:
        if not predictions[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise X15OOFPublicationError(
                f"oof_predictions {column} must be nonempty text"
            )
    for column, expected in (
        ("fold_id", fold_id),
        ("model_id", model_id),
        ("feature_block_id", feature_block_id),
        ("target_contract", DIAGNOSTIC_TARGET_CONTRACT),
        ("claim_boundary", DIAGNOSTIC_CLAIM_BOUNDARY),
        (
            "direction_threshold_semantics",
            DIAGNOSTIC_DIRECTION_THRESHOLD_SEMANTICS,
        ),
        ("venue_tick_support", DIAGNOSTIC_VENUE_TICK_SUPPORT),
        (
            "market_continuity_support",
            DIAGNOSTIC_MARKET_CONTINUITY_SUPPORT,
        ),
        ("schema_version", DIAGNOSTIC_SCHEMA_VERSION),
        ("analysis_scope", DIAGNOSTIC_ANALYSIS_SCOPE),
    ):
        if not predictions[column].eq(expected).all():
            raise X15OOFPublicationError(
                f"oof_predictions identity drifted on {column}"
            )
    if not predictions["calibration_venue"].eq(
        predictions["training_venue"]
    ).all():
        raise X15OOFPublicationError(
            "oof_predictions calibration venue must equal training venue"
        )
    if not predictions["claim_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and not bool(value)
    ).all():
        raise X15OOFPublicationError(
            "oof_predictions must remain claim_eligible=False"
        )
    threshold = pd.to_numeric(
        predictions["direction_threshold_probability"],
        errors="coerce",
    )
    if (
        threshold.isna().any()
        or not np.isfinite(threshold.to_numpy(dtype=float)).all()
        or not threshold.eq(
            DIAGNOSTIC_DIRECTION_THRESHOLD_PROBABILITY
        ).all()
    ):
        raise X15OOFPublicationError(
            "oof_predictions direction probability threshold drifted"
        )

    game_weeks = authority.game_weeks
    governed_weeks = predictions["game_id"].map(game_weeks)
    observed_weeks = pd.to_numeric(
        predictions["nfl_week"], errors="coerce"
    )
    if (
        governed_weeks.isna().any()
        or observed_weeks.isna().any()
        or not np.equal(
            observed_weeks.to_numpy(dtype=float),
            governed_weeks.to_numpy(dtype=float),
        ).all()
    ):
        raise X15OOFPublicationError(
            "oof_predictions game week disagrees with authority"
        )
    for column in ("source_row_id", "landmark_seconds", "endpoint_seconds"):
        numeric = pd.to_numeric(predictions[column], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if (
            numeric.isna().any()
            or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise X15OOFPublicationError(
                f"oof_predictions {column} must be integral"
            )
    if (
        pd.to_numeric(predictions["source_row_id"]).lt(0).any()
        or pd.to_numeric(predictions["landmark_seconds"]).lt(0).any()
        or pd.to_numeric(predictions["endpoint_seconds"]).le(
            pd.to_numeric(predictions["landmark_seconds"])
        ).any()
    ):
        raise X15OOFPublicationError(
            "oof_predictions L/H/source-row identity is invalid"
        )
    if predictions.duplicated(
        list(_SELECTION_PAIR_GRAIN), keep=False
    ).any():
        raise X15OOFPublicationError(
            "oof_predictions contains duplicate selection pair grain"
        )

    for column in _BINARY_TRUTH_COLUMNS:
        if not predictions[column].map(
            lambda value: _is_missing_scalar(value)
            or isinstance(value, (bool, np.bool_))
        ).all():
            raise X15OOFPublicationError(
                f"oof_predictions {column} has invalid binary truth"
            )
    if not predictions["direction_truth"].map(
        lambda value: _is_missing_scalar(value)
        or (
            isinstance(value, str)
            and value in _DIRECTION_CLASSES
        )
    ).all():
        raise X15OOFPublicationError(
            "oof_predictions has invalid direction truth"
        )
    s_h_true = predictions["s_h_truth"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and bool(value)
    )
    o_h_missing = predictions["o_h_given_s_truth"].map(
        _is_missing_scalar
    )
    if (
        (~s_h_true & ~o_h_missing).any()
        or (s_h_true & o_h_missing).any()
    ):
        raise X15OOFPublicationError(
            "oof_predictions O_H truth requires S_H truth"
        )
    o_h_true = predictions["o_h_given_s_truth"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and bool(value)
    )
    direction_missing = predictions["direction_truth"].map(
        _is_missing_scalar
    )
    if (
        (~o_h_true & ~direction_missing).any()
        or (o_h_true & direction_missing).any()
    ):
        raise X15OOFPublicationError(
            "oof_predictions direction truth requires O_H truth"
        )
    for column in (
        *_BINARY_PROBABILITY_COLUMNS,
        *_DIRECTION_PROBABILITY_COLUMNS,
    ):
        numeric = pd.to_numeric(predictions[column], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if (
            numeric.isna().any()
            or not np.isfinite(values).all()
            or not numeric.between(0.0, 1.0, inclusive="both").all()
        ):
            raise X15OOFPublicationError(
                f"oof_predictions {column} probability is outside [0,1]"
            )
    direction_sum = predictions[
        list(_DIRECTION_PROBABILITY_COLUMNS)
    ].astype(float).sum(axis=1).to_numpy(dtype=float)
    if not np.isclose(
        direction_sum,
        np.ones(len(direction_sum), dtype=float),
        rtol=0.0,
        atol=1e-10,
    ).all():
        raise X15OOFPublicationError(
            "oof_predictions direction probabilities must sum to one"
        )

    native_polymarket = predictions.loc[
        predictions["venue"].eq("polymarket")
        & predictions["training_venue"].eq("polymarket")
        & predictions["transport_mode"].eq("VENUE_SPECIFIC")
    ]
    if not set(native_polymarket["game_id"].astype(str)).issubset(
        validation_game_ids
    ):
        raise X15OOFPublicationError(
            "native Polymarket OOF rows leave fold validation authority"
        )


def _native_polymarket_fold_arrays(
    predictions: pd.DataFrame,
) -> dict[str, tuple[object, ...]]:
    native = predictions.loc[
        predictions["venue"].eq("polymarket")
        & predictions["training_venue"].eq("polymarket")
        & predictions["transport_mode"].eq("VENUE_SPECIFIC")
    ]
    if native.empty:
        return {column: () for column in _FOLD_ARRAY_COLUMNS}
    values: dict[str, tuple[object, ...]] = {}
    for column in _FOLD_ARRAY_COLUMNS:
        first = _tuple_value(native.iloc[0][column])
        if first is None or not native[column].map(
            lambda value: _tuple_value(value) == first
        ).all():
            raise X15OOFPublicationError(
                "native Polymarket fold arrays must be constant within shard"
            )
        values[column] = first
    return values


def _validate_frame_identity(
    frame: pd.DataFrame,
    *,
    name: str,
    fold_id: str,
    model_id: str,
    feature_block_id: str,
    run_config_sha256: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise X15OOFPublicationError(f"{name} must be a DataFrame")
    if frame.columns.duplicated().any():
        raise X15OOFPublicationError(f"{name} has duplicate columns")
    if frame.empty:
        return
    if "fold_id" not in frame:
        raise X15OOFPublicationError(f"{name} lacks fold_id")
    if not frame["fold_id"].eq(fold_id).all():
        raise X15OOFPublicationError(f"{name} crosses fold boundaries")
    for column, expected in (
        ("model_id", model_id),
        ("feature_block_id", feature_block_id),
    ):
        if column in frame and not frame[column].eq(expected).all():
            raise X15OOFPublicationError(
                f"{name} crosses shard {column} boundaries"
            )
    if "run_config_sha256" in frame and not frame[
        "run_config_sha256"
    ].eq(run_config_sha256).all():
        raise X15OOFPublicationError(
            f"{name} run_config_sha256 does not match model run"
        )
    if "claim_boundary" in frame and not frame["claim_boundary"].eq(
        DIAGNOSTIC_CLAIM_BOUNDARY
    ).all():
        raise X15OOFPublicationError(
            f"{name} diagnostic claim boundary drifted"
        )
    if "claim_eligible" in frame and not frame["claim_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and not bool(value)
    ).all():
        raise X15OOFPublicationError(
            f"{name} must remain diagnostic claim_eligible=False"
        )


def _validate_shard_run(
    *,
    model_run: X15ModelRun,
    authority: FrozenDevelopmentAuthority,
    fold_id: str,
    model_id: str,
    feature_block_id: str,
) -> tuple[
    dict[str, object],
    str,
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool,
]:
    run_config, shared_config_sha256 = _validate_run_config(
        model_run,
        fold_id=fold_id,
        model_id=model_id,
        feature_block_id=feature_block_id,
        authority=authority,
    )
    (
        train_weeks,
        validation_weeks,
        training_game_ids,
        validation_game_ids,
    ) = _fold_authority(authority=authority, fold_id=fold_id)
    for name in OOF_TABLE_NAMES:
        _validate_frame_identity(
            getattr(model_run, name),
            name=name,
            fold_id=fold_id,
            model_id=model_id,
            feature_block_id=feature_block_id,
            run_config_sha256=model_run.run_config_sha256,
        )
    if not model_run.conditional_quantiles.empty:
        raise X15OOFPublicationError(
            "probability-only shard cannot publish conditional quantiles"
        )
    predictions = model_run.oof_predictions
    if predictions.empty:
        raise X15OOFPublicationError("oof_predictions cannot be empty")
    missing = sorted(
        set(_PREDICTION_FOLD_COLUMNS).difference(predictions.columns)
    )
    if missing:
        raise X15OOFPublicationError(
            f"oof_predictions missing fold authority columns: {missing}"
        )
    _validate_prediction_semantics(
        predictions,
        authority=authority,
        fold_id=fold_id,
        model_id=model_id,
        feature_block_id=feature_block_id,
        validation_game_ids=validation_game_ids,
    )
    if not predictions["cohort_authority_sha256"].eq(
        authority.cohort_authority_sha256
    ).all():
        raise X15OOFPublicationError(
            "OOF rows disagree with cohort authority"
        )
    for column, expected in (
        ("train_weeks", train_weeks),
        ("validation_weeks", validation_weeks),
    ):
        if not predictions[column].map(
            lambda value: _tuple_value(value) == tuple(expected)
        ).all():
            raise X15OOFPublicationError(
                f"oof_predictions {column} disagrees with fold authority"
            )

    expected_training = set(training_game_ids)
    expected_validation = set(validation_game_ids)

    def valid_subset(value: object, expected: set[str]) -> bool:
        sequence = _tuple_value(value)
        return (
            sequence is not None
            and bool(sequence)
            and len(sequence) == len(set(sequence))
            and tuple(sorted(map(str, sequence))) == tuple(sequence)
            and set(map(str, sequence)).issubset(expected)
        )

    if not predictions["training_game_ids"].map(
        lambda value: valid_subset(value, expected_training)
    ).all():
        raise X15OOFPublicationError(
            "oof_predictions training_game_ids leave fold training authority"
        )
    if not predictions["validation_game_ids"].map(
        lambda value: valid_subset(value, expected_validation)
    ).all():
        raise X15OOFPublicationError(
            "oof_predictions validation_game_ids leave fold validation authority"
        )
    if not predictions.apply(
        lambda row: _tuple_value(row["preprocessor_fit_game_ids"])
        == _tuple_value(row["training_game_ids"]),
        axis=1,
    ).all():
        raise X15OOFPublicationError(
            "preprocessor fit games must equal the row training games"
        )
    if not predictions.apply(
        lambda row: str(row["game_id"])
        in set(map(str, _tuple_value(row["validation_game_ids"]) or ())),
        axis=1,
    ).all():
        raise X15OOFPublicationError(
            "OOF row game must belong to its venue validation games"
        )
    native_polymarket = predictions.loc[
        predictions["venue"].eq("polymarket")
        & predictions["training_venue"].eq("polymarket")
        & predictions["transport_mode"].eq("VENUE_SPECIFIC")
    ]
    observed_validation_game_ids = tuple(
        sorted(set(native_polymarket["game_id"].astype(str)))
    )
    if not set(observed_validation_game_ids).issubset(
        validation_game_ids
    ):
        raise X15OOFPublicationError(
            "OOF predictions contain games outside fold validation weeks"
        )
    authority_coverage_complete = (
        observed_validation_game_ids == validation_game_ids
    )
    return (
        run_config,
        shared_config_sha256,
        train_weeks,
        validation_weeks,
        training_game_ids,
        validation_game_ids,
        observed_validation_game_ids,
        authority_coverage_complete,
    )


def _parquet_payload(
    frame: pd.DataFrame,
) -> tuple[bytes, pa.Schema]:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        sink = pa.BufferOutputStream()
        pq.write_table(
            table,
            sink,
            compression="zstd",
            version="2.6",
            use_dictionary=True,
            write_statistics=True,
        )
        payload = sink.getvalue().to_pybytes()
        persisted_schema = pq.ParquetFile(
            pa.BufferReader(payload)
        ).schema_arrow
    except Exception as error:
        raise X15OOFPublicationError(
            "model output cannot be encoded as Parquet"
        ) from error
    return payload, persisted_schema


def _publish_table(
    *,
    output_root: Path,
    fold_id: str,
    model_id: str,
    feature_block_id: str,
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload, schema = _parquet_payload(frame)
    object_sha256 = _sha256_bytes(payload)
    path = _content_addressed_path(
        output_root=output_root,
        namespace=(
            Path("shards")
            / fold_id
            / model_id
            / feature_block_id
            / "objects"
        ),
        payload_sha256=object_sha256,
        suffix=".parquet",
    )
    _atomic_publish(path, payload)
    return {
        "name": name,
        "object_path": path.relative_to(output_root).as_posix(),
        "object_sha256": object_sha256,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": _sha256_bytes(
            schema.serialize().to_pybytes()
        ),
        "semantic_rows_sha256": _canonical_sha256(
            frame.to_dict("records")
        ),
        "format": "parquet",
    }


def _publish_manifest(
    *,
    output_root: Path,
    namespace: Path,
    suffix: str,
    material: dict[str, object],
    digest_field: str,
) -> tuple[Path, str, str]:
    digest = _canonical_sha256(material)
    document = dict(material)
    document[digest_field] = digest
    payload = _canonical_bytes(document)
    manifest_sha256 = _sha256_bytes(payload)
    path = _content_addressed_path(
        output_root=output_root,
        namespace=namespace,
        payload_sha256=manifest_sha256,
        suffix=suffix,
    )
    _atomic_publish(path, payload)
    return path, manifest_sha256, digest


def publish_x15_oof_shard(
    *,
    model_run: X15ModelRun,
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    output_root: str | Path,
) -> X15OOFShardPublication:
    """Publish one diagnostic X-15 probability shard atomically."""

    root = Path(output_root).resolve()
    mapping_sha256 = _require_sha256(
        cohort_mapping_sha256, field="cohort_mapping_sha256"
    )
    if not isinstance(model_run.run_config, Mapping):
        raise X15OOFPublicationError(
            "model_run must carry one execution shard in run_config"
        )
    configured: dict[str, tuple[object, ...]] = {}
    for field in ("fold_ids", "model_ids", "feature_block_ids"):
        values = _tuple_value(model_run.run_config.get(field))
        if values is None or len(values) != 1:
            raise X15OOFPublicationError(
                f"one shard requires exactly one configured {field}"
            )
        configured[field] = values
    fold_id = str(configured["fold_ids"][0])
    model_id = str(configured["model_ids"][0])
    feature_block_id = str(configured["feature_block_ids"][0])
    if (model_id, feature_block_id) not in PROBABILITY_DESIGN_MATRIX:
        raise X15OOFPublicationError(
            "configured shard is outside the frozen probability design matrix"
        )
    (
        run_config,
        shared_config_sha256,
        train_weeks,
        validation_weeks,
        training_game_ids,
        validation_game_ids,
        observed_validation_game_ids,
        authority_coverage_complete,
    ) = _validate_shard_run(
        model_run=model_run,
        authority=authority,
        fold_id=fold_id,
        model_id=model_id,
        feature_block_id=feature_block_id,
    )
    effective_seed_contract = _effective_seed_contract(run_config)
    selection_fold_arrays = _native_polymarket_fold_arrays(
        model_run.oof_predictions
    )
    tables = [
        _publish_table(
            output_root=root,
            fold_id=fold_id,
            model_id=model_id,
            feature_block_id=feature_block_id,
            name=name,
            frame=getattr(model_run, name),
        )
        for name in OOF_TABLE_NAMES
    ]
    material: dict[str, object] = {
        "schema": SHARD_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-15",
        "cohort": "development",
        "fold_id": fold_id,
        "model_id": model_id,
        "feature_block_id": feature_block_id,
        "selection_role": _selection_role(model_id, feature_block_id),
        "include_magnitude": False,
        "train_weeks": train_weeks,
        "validation_weeks": validation_weeks,
        "training_game_ids": training_game_ids,
        "validation_game_ids": validation_game_ids,
        "observed_validation_game_ids": observed_validation_game_ids,
        "authority_game_count": authority.game_count,
        "authority_metadata_sha256": authority.metadata_sha256,
        "authority_coverage_complete": authority_coverage_complete,
        "cohort_authority_sha256": authority.cohort_authority_sha256,
        "cohort_mapping_sha256": mapping_sha256,
        "run_config": run_config,
        "run_config_sha256": model_run.run_config_sha256,
        "shared_run_config_sha256": shared_config_sha256,
        "effective_seed_contract": effective_seed_contract,
        "effective_seed_contract_sha256": _model_sha256(
            effective_seed_contract
        ),
        "selection_fold_arrays": selection_fold_arrays,
        "selection_fold_arrays_sha256": _model_sha256(
            selection_fold_arrays
        ),
        "diagnostic_claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "diagnostic_target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "claim_eligible": False,
        "selection_ready": False,
        "publication_status": "SHARD_DIAGNOSTIC_ONLY",
        "holdout_reaction_accessed": False,
        "tables": tables,
    }
    manifest_path, manifest_sha256, bundle_sha256 = _publish_manifest(
        output_root=root,
        namespace=(
            Path("shards")
            / fold_id
            / model_id
            / feature_block_id
            / "manifests"
        ),
        suffix=".shard-manifest.json",
        material=material,
        digest_field="bundle_sha256",
    )
    return X15OOFShardPublication(
        fold_id=fold_id,
        model_id=model_id,
        feature_block_id=feature_block_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        bundle_sha256=bundle_sha256,
        run_config_sha256=model_run.run_config_sha256,
        authority_coverage_complete=authority_coverage_complete,
    )


def _read_shard_document(
    *,
    manifest_path: str | Path,
    output_root: Path,
    load_tables: bool,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    path = _resolve_under(
        output_root, manifest_path, field="shard manifest"
    )
    if path.is_symlink() or not path.is_file():
        raise X15OOFPublicationError("shard manifest is not a regular file")
    expected_manifest_sha256 = _verify_content_addressed_filename(
        path,
        suffix=".shard-manifest.json",
        field="shard manifest",
    )
    observed_manifest_sha256, _ = sha256_path(path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise X15OOFPublicationError("shard manifest SHA-256 mismatch")
    document = _strict_json(
        path.read_bytes(), field="shard manifest"
    )
    material = dict(document)
    bundle_sha256 = _require_sha256(
        material.pop("bundle_sha256", None),
        field="shard manifest.bundle_sha256",
    )
    if _canonical_sha256(material) != bundle_sha256:
        raise X15OOFPublicationError(
            "shard manifest bundle SHA-256 mismatch"
        )
    if (
        document.get("schema") != SHARD_SCHEMA
        or document.get("builder_version") != BUILDER_VERSION
        or document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or document.get("diagnostic_claim_boundary")
        != DIAGNOSTIC_CLAIM_BOUNDARY
        or document.get("diagnostic_target_contract")
        != DIAGNOSTIC_TARGET_CONTRACT
        or document.get("claim_eligible") is not False
        or document.get("selection_ready") is not False
        or document.get("holdout_reaction_accessed") is not False
    ):
        raise X15OOFPublicationError("shard manifest contract mismatch")
    fold_id = str(document.get("fold_id", ""))
    if fold_id not in _FOLD_BY_ID:
        raise X15OOFPublicationError("shard manifest fold_id is invalid")
    model_id = str(document.get("model_id", ""))
    feature_block_id = str(document.get("feature_block_id", ""))
    if (model_id, feature_block_id) not in PROBABILITY_DESIGN_MATRIX:
        raise X15OOFPublicationError(
            "shard manifest model/feature-block identity is invalid"
        )
    if document.get("selection_role") != _selection_role(
        model_id, feature_block_id
    ):
        raise X15OOFPublicationError(
            "shard manifest selection role is invalid"
        )
    if document.get("include_magnitude") is not False:
        raise X15OOFPublicationError(
            "shard manifest must remain probability-only"
        )
    run_config = document.get("run_config")
    if not isinstance(run_config, dict):
        raise X15OOFPublicationError("shard manifest run_config is invalid")
    run_config_sha256 = _require_sha256(
        document.get("run_config_sha256"),
        field="shard manifest.run_config_sha256",
    )
    if _model_sha256(run_config) != run_config_sha256:
        raise X15OOFPublicationError(
            "shard manifest run config SHA-256 mismatch"
        )
    expected_seed_contract = _effective_seed_contract(run_config)
    seed_contract = document.get("effective_seed_contract")
    if (
        not isinstance(seed_contract, dict)
        or seed_contract != expected_seed_contract
        or _model_sha256(seed_contract)
        != _require_sha256(
            document.get("effective_seed_contract_sha256"),
            field="shard manifest.effective_seed_contract_sha256",
        )
    ):
        raise X15OOFPublicationError(
            "shard manifest effective seed contract mismatch"
        )
    selection_fold_arrays = document.get("selection_fold_arrays")
    if (
        not isinstance(selection_fold_arrays, dict)
        or set(selection_fold_arrays) != set(_FOLD_ARRAY_COLUMNS)
        or _model_sha256(selection_fold_arrays)
        != _require_sha256(
            document.get("selection_fold_arrays_sha256"),
            field="shard manifest.selection_fold_arrays_sha256",
        )
    ):
        raise X15OOFPublicationError(
            "shard manifest selection fold arrays mismatch"
        )
    normalized_fold_arrays: dict[str, tuple[object, ...]] = {}
    for column in _FOLD_ARRAY_COLUMNS:
        sequence = _tuple_value(selection_fold_arrays[column])
        if sequence is None:
            raise X15OOFPublicationError(
                "shard manifest selection fold arrays are malformed"
            )
        normalized_fold_arrays[column] = sequence
    if normalized_fold_arrays["validation_game_ids"]:
        if (
            normalized_fold_arrays["train_weeks"]
            != _tuple_value(document.get("train_weeks"))
            or normalized_fold_arrays["validation_weeks"]
            != _tuple_value(document.get("validation_weeks"))
            or not set(
                map(str, normalized_fold_arrays["training_game_ids"])
            ).issubset(set(map(str, document["training_game_ids"])))
            or not set(
                map(str, normalized_fold_arrays["validation_game_ids"])
            ).issubset(set(map(str, document["validation_game_ids"])))
            or normalized_fold_arrays["preprocessor_fit_game_ids"]
            != normalized_fold_arrays["training_game_ids"]
        ):
            raise X15OOFPublicationError(
                "shard manifest selection fold arrays leave authority"
            )
    if (
        _tuple_value(run_config.get("fold_ids")) != (fold_id,)
        or _tuple_value(run_config.get("model_ids")) != (model_id,)
        or _tuple_value(run_config.get("feature_block_ids"))
        != (feature_block_id,)
        or run_config.get("include_magnitude") is not False
    ):
        raise X15OOFPublicationError(
            "shard manifest run config identity mismatch"
        )
    tables = document.get("tables")
    if not isinstance(tables, list) or len(tables) != len(OOF_TABLE_NAMES):
        raise X15OOFPublicationError(
            "shard manifest table set is incomplete"
        )
    by_name: dict[str, dict[str, object]] = {}
    for value in tables:
        if not isinstance(value, dict):
            raise X15OOFPublicationError("shard table descriptor is invalid")
        name = str(value.get("name", ""))
        if name not in OOF_TABLE_NAMES or name in by_name:
            raise X15OOFPublicationError(
                "shard table names are invalid or duplicated"
            )
        by_name[name] = value
    if tuple(sorted(by_name)) != tuple(sorted(OOF_TABLE_NAMES)):
        raise X15OOFPublicationError("shard manifest table names differ")

    loaded: dict[str, pd.DataFrame] = {}
    for name in OOF_TABLE_NAMES:
        descriptor = by_name[name]
        object_path = _resolve_under(
            output_root,
            str(descriptor.get("object_path", "")),
            field=f"{name}.object_path",
        )
        if object_path.is_symlink() or not object_path.is_file():
            raise X15OOFPublicationError(
                f"{name} object is not a regular file"
            )
        expected_sha256 = _require_sha256(
            descriptor.get("object_sha256"),
            field=f"{name}.object_sha256",
        )
        if _verify_content_addressed_filename(
            object_path, suffix=".parquet", field=f"{name} object"
        ) != expected_sha256:
            raise X15OOFPublicationError(
                f"{name} content-addressed path disagrees with descriptor"
            )
        observed_sha256, observed_bytes = sha256_path(object_path)
        if observed_bytes != descriptor.get("byte_length"):
            raise X15OOFPublicationError(f"{name} byte length mismatch")
        if observed_sha256 != expected_sha256:
            raise X15OOFPublicationError(f"{name} SHA-256 mismatch")
        try:
            parquet = pq.ParquetFile(object_path)
            schema = parquet.schema_arrow
            row_count = (
                parquet.metadata.num_rows
                if parquet.metadata is not None
                else None
            )
        except Exception as error:
            raise X15OOFPublicationError(
                f"{name} Parquet metadata is unreadable"
            ) from error
        if row_count != descriptor.get("row_count"):
            raise X15OOFPublicationError(f"{name} row count mismatch")
        if _sha256_bytes(
            schema.serialize().to_pybytes()
        ) != descriptor.get("schema_fingerprint"):
            raise X15OOFPublicationError(
                f"{name} schema fingerprint mismatch"
            )
        if list(schema.names) != descriptor.get("schema_columns"):
            raise X15OOFPublicationError(f"{name} schema columns mismatch")
        if not load_tables:
            continue
        try:
            frame = pq.read_table(object_path).to_pandas()
        except Exception as error:
            raise X15OOFPublicationError(
                f"{name} Parquet object is unreadable"
            ) from error
        object_columns = [
            column
            for column, dtype in frame.dtypes.items()
            if dtype == object
        ]
        for column in object_columns:
            frame[column] = frame[column].map(
                lambda value: (
                    tuple(value)
                    if isinstance(value, (list, np.ndarray))
                    else value
                )
            )
        if _canonical_sha256(
            frame.to_dict("records")
        ) != descriptor.get("semantic_rows_sha256"):
            raise X15OOFPublicationError(
                f"{name} semantic rows SHA-256 mismatch"
            )
        loaded[name] = frame
    return document, loaded


def load_x15_oof_shard(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
) -> X15ModelRun:
    """Recover one byte-verified shard checkpoint as ``X15ModelRun``."""

    document, tables = _read_shard_document(
        manifest_path=manifest_path,
        output_root=Path(output_root).resolve(),
        load_tables=True,
    )
    return X15ModelRun(
        oof_predictions=tables["oof_predictions"],
        conditional_quantiles=tables["conditional_quantiles"],
        fold_metrics=tables["fold_metrics"],
        support_audit=tables["support_audit"],
        weight_audit=tables["weight_audit"],
        run_config_sha256=str(document["run_config_sha256"]),
        run_config=dict(document["run_config"]),
    )


def _shard_reference(
    *,
    document: Mapping[str, object],
    manifest_path: Path,
    output_root: Path,
) -> dict[str, object]:
    manifest_sha256, _ = sha256_path(manifest_path)
    tables = document["tables"]
    if not isinstance(tables, list):
        raise X15OOFPublicationError(
            "verified shard table references are malformed"
        )
    return {
        "fold_id": document["fold_id"],
        "model_id": document["model_id"],
        "feature_block_id": document["feature_block_id"],
        "selection_role": document["selection_role"],
        "train_weeks": document["train_weeks"],
        "validation_weeks": document["validation_weeks"],
        "training_game_count": len(document["training_game_ids"]),
        "validation_game_count": len(document["validation_game_ids"]),
        "observed_validation_game_count": len(
            document["observed_validation_game_ids"]
        ),
        "authority_coverage_complete": document[
            "authority_coverage_complete"
        ],
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
        "shard_bundle_sha256": document["bundle_sha256"],
        "manifest_path": manifest_path.relative_to(output_root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "tables": [
            {
                "name": table["name"],
                "object_sha256": table["object_sha256"],
                "schema_fingerprint": table["schema_fingerprint"],
                "semantic_rows_sha256": table[
                    "semantic_rows_sha256"
                ],
                "row_count": table["row_count"],
            }
            for table in tables
        ],
    }


def publish_x15_oof_batch(
    *,
    shard_manifest_paths: Sequence[str | Path],
    authority: FrozenDevelopmentAuthority,
    cohort_mapping_sha256: str,
    output_root: str | Path,
) -> X15OOFBatchPublication:
    """Publish a partial index or the exact 55-cell selection gate.

    The batch verifier reads only manifests, file digests, Parquet metadata,
    and schema fingerprints.  It never concatenates prediction tables.
    """

    root = Path(output_root).resolve()
    _validate_authority(authority)
    mapping_sha256 = _require_sha256(
        cohort_mapping_sha256, field="cohort_mapping_sha256"
    )
    if not shard_manifest_paths:
        raise X15OOFPublicationError(
            "at least one shard manifest is required"
        )
    records: dict[
        tuple[str, str, str], tuple[dict[str, object], Path]
    ] = {}
    for manifest_path in shard_manifest_paths:
        path = _resolve_under(
            root, manifest_path, field="shard manifest"
        )
        document, _ = _read_shard_document(
            manifest_path=path,
            output_root=root,
            load_tables=False,
        )
        fold_id = str(document["fold_id"])
        model_id = str(document["model_id"])
        feature_block_id = str(document["feature_block_id"])
        cell = (fold_id, model_id, feature_block_id)
        if cell in records:
            raise X15OOFPublicationError(
                "duplicate execution shard: "
                f"{fold_id}/{model_id}/{feature_block_id}"
            )
        if (
            document.get("cohort_authority_sha256")
            != authority.cohort_authority_sha256
            or document.get("cohort_mapping_sha256") != mapping_sha256
            or document.get("authority_metadata_sha256")
            != authority.metadata_sha256
            or document.get("authority_game_count")
            != EXPECTED_DEVELOPMENT_GAME_COUNT
        ):
            raise X15OOFPublicationError(
                f"{fold_id} authority or cohort mapping mismatch"
            )
        expected = _fold_authority(authority=authority, fold_id=fold_id)
        for field, value in (
            ("train_weeks", expected[0]),
            ("validation_weeks", expected[1]),
            ("training_game_ids", expected[2]),
            ("validation_game_ids", expected[3]),
        ):
            if _tuple_value(document.get(field)) != tuple(value):
                raise X15OOFPublicationError(
                    f"{fold_id} {field} differs from exact authority"
                )
        records[cell] = (document, path)

    expected_fold_ids = tuple(fold[0] for fold in EXPECTED_FOLDS)
    expected_cells = tuple(
        (fold_id, model_id, feature_block_id)
        for fold_id in expected_fold_ids
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX
    )
    expected_cell_set = set(expected_cells)
    if set(records).difference(expected_cell_set):
        raise X15OOFPublicationError(
            "batch contains execution cells outside the frozen design matrix"
        )
    observed_cells = tuple(
        cell for cell in expected_cells if cell in records
    )
    missing_cells = tuple(
        cell for cell in expected_cells if cell not in records
    )
    observed_fold_ids = tuple(
        fold_id
        for fold_id in expected_fold_ids
        if any(cell[0] == fold_id for cell in observed_cells)
    )
    missing_fold_ids = tuple(
        fold_id
        for fold_id in expected_fold_ids
        if not any(cell[0] == fold_id for cell in observed_cells)
    )
    shared_run_config_sha256s = {
        str(document["shared_run_config_sha256"])
        for document, _ in records.values()
    }
    if len(shared_run_config_sha256s) != 1:
        raise X15OOFPublicationError(
            "shard run config differs beyond fold/model/block execution "
            "coordinates"
        )
    effective_seed_contract_sha256s = {
        str(document["effective_seed_contract_sha256"])
        for document, _ in records.values()
    }
    if len(effective_seed_contract_sha256s) != 1:
        raise X15OOFPublicationError(
            "shard effective seed contracts differ"
        )
    selection_ready = not missing_cells
    if selection_ready and not all(
        document.get("authority_coverage_complete") is True
        for document, _ in records.values()
    ):
        raise X15OOFPublicationError(
            "selection-ready batch requires exact shard authority coverage"
        )
    publication_status = (
        "SELECTION_READY"
        if selection_ready
        else "PARTIAL_DIAGNOSTIC_ONLY"
    )
    game_weeks = authority.game_weeks
    training_only_game_ids = tuple(
        sorted(
            game_id
            for game_id, week in game_weeks.items()
            if week in (1, 2)
        )
    )
    oof_validation_game_ids = tuple(
        sorted(
            game_id
            for game_id, week in game_weeks.items()
            if week in range(3, 13)
        )
    )
    published_validation_game_ids = tuple(
        sorted(
            {
                str(game_id)
                for document, _ in records.values()
                for game_id in document["observed_validation_game_ids"]
            }
        )
    )
    if (
        selection_ready
        and published_validation_game_ids != oof_validation_game_ids
    ):
        raise X15OOFPublicationError(
            "selection-ready batch must cover the exact 123-game "
            "weeks3-12 OOF validation union"
        )
    shard_references = [
        _shard_reference(
            document=records[cell][0],
            manifest_path=records[cell][1],
            output_root=root,
        )
        for cell in observed_cells
    ]
    table_row_counts = {
        name: sum(
            int(table["row_count"])
            for reference in shard_references
            for table in reference["tables"]
            if table["name"] == name
        )
        for name in OOF_TABLE_NAMES
    }
    first_run_config = dict(
        records[observed_cells[0]][0]["run_config"]
    )
    for execution_field in (
        "fold_ids",
        "model_ids",
        "feature_block_ids",
    ):
        first_run_config.pop(execution_field, None)
    source_shards = tuple(
        {
            "fold_id": reference["fold_id"],
            "model_id": reference["model_id"],
            "feature_block_id": reference["feature_block_id"],
            "manifest_sha256": reference["manifest_sha256"],
            "shard_bundle_sha256": reference["shard_bundle_sha256"],
        }
        for reference in shard_references
    )
    batch_run_config = {
        **first_run_config,
        "fold_ids": observed_fold_ids,
        "model_ids": tuple(
            dict.fromkeys(cell[1] for cell in observed_cells)
        ),
        "feature_block_ids": tuple(
            dict.fromkeys(cell[2] for cell in observed_cells)
        ),
        "source_shards": source_shards,
    }
    material: dict[str, object] = {
        "schema": BATCH_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-15",
        "cohort": "development",
        "authority_game_count": authority.game_count,
        "training_only_game_count": len(training_only_game_ids),
        "oof_validation_game_count": len(oof_validation_game_ids),
        "published_oof_validation_game_count": len(
            published_validation_game_ids
        ),
        "oof_validation_game_ids_sha256": _model_sha256(
            oof_validation_game_ids
        ),
        "authority_metadata_sha256": authority.metadata_sha256,
        "cohort_authority_sha256": authority.cohort_authority_sha256,
        "cohort_mapping_sha256": mapping_sha256,
        "expected_fold_ids": expected_fold_ids,
        "observed_fold_ids": observed_fold_ids,
        "missing_fold_ids": missing_fold_ids,
        "baseline_design_cell": BASELINE_DESIGN_CELL,
        "negative_control_design_matrix": (
            NEGATIVE_CONTROL_DESIGN_MATRIX
        ),
        "negative_control_semantics": (
            "TIME_GRID_ONLY_MODEL_COMPLEXITY_CONTROL_NOT_SELECTABLE"
        ),
        "selectable_candidate_design_matrix": (
            SELECTABLE_CANDIDATE_DESIGN_MATRIX
        ),
        "expected_execution_cell_count": len(expected_cells),
        "observed_execution_cell_count": len(observed_cells),
        "missing_execution_cell_count": len(missing_cells),
        "expected_execution_cells": expected_cells,
        "observed_execution_cells": observed_cells,
        "missing_execution_cells": missing_cells,
        "shared_run_config_sha256": next(
            iter(shared_run_config_sha256s)
        ),
        "effective_seed_contract_sha256": next(
            iter(effective_seed_contract_sha256s)
        ),
        "batch_run_config": batch_run_config,
        "batch_run_config_sha256": _model_sha256(batch_run_config),
        "diagnostic_claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "diagnostic_target_contract": DIAGNOSTIC_TARGET_CONTRACT,
        "claim_eligible": False,
        "selection_ready": selection_ready,
        "publication_status": publication_status,
        "holdout_reaction_accessed": False,
        "table_row_counts": table_row_counts,
        "shards": shard_references,
    }
    manifest_path, manifest_sha256, batch_sha256 = _publish_manifest(
        output_root=root,
        namespace=Path("batches") / "manifests",
        suffix=".batch-index.json",
        material=material,
        digest_field="batch_sha256",
    )
    return X15OOFBatchPublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        batch_sha256=batch_sha256,
        observed_fold_ids=observed_fold_ids,
        selection_ready=selection_ready,
    )


def _read_selection_ready_batch(
    *,
    batch_manifest_path: str | Path,
    output_root: Path,
) -> tuple[dict[str, object], Path, str]:
    path = _resolve_under(
        output_root, batch_manifest_path, field="batch manifest"
    )
    if path.is_symlink() or not path.is_file():
        raise X15OOFPublicationError("batch manifest is not a regular file")
    expected_manifest_sha256 = _verify_content_addressed_filename(
        path,
        suffix=".batch-index.json",
        field="batch manifest",
    )
    observed_manifest_sha256, _ = sha256_path(path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise X15OOFPublicationError("batch manifest SHA-256 mismatch")
    document = _strict_json(path.read_bytes(), field="batch manifest")
    material = dict(document)
    batch_sha256 = _require_sha256(
        material.pop("batch_sha256", None),
        field="batch manifest.batch_sha256",
    )
    if _canonical_sha256(material) != batch_sha256:
        raise X15OOFPublicationError(
            "batch manifest semantic SHA-256 mismatch"
        )
    if (
        document.get("schema") != BATCH_SCHEMA
        or document.get("builder_version") != BUILDER_VERSION
        or document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or document.get("diagnostic_claim_boundary")
        != DIAGNOSTIC_CLAIM_BOUNDARY
        or document.get("diagnostic_target_contract")
        != DIAGNOSTIC_TARGET_CONTRACT
        or document.get("claim_eligible") is not False
        or document.get("selection_ready") is not True
        or document.get("publication_status") != "SELECTION_READY"
        or document.get("holdout_reaction_accessed") is not False
        or document.get("authority_game_count")
        != EXPECTED_DEVELOPMENT_GAME_COUNT
        or document.get("training_only_game_count")
        != _EXPECTED_TRAINING_ONLY_GAME_COUNT
        or document.get("oof_validation_game_count")
        != _EXPECTED_OOF_VALIDATION_GAME_COUNT
        or document.get("expected_execution_cell_count") != 55
        or document.get("observed_execution_cell_count") != 55
        or document.get("missing_execution_cell_count") != 0
    ):
        raise X15OOFPublicationError(
            "selection projection requires a verified exact-55 batch"
        )
    batch_run_config = document.get("batch_run_config")
    if not isinstance(batch_run_config, dict):
        raise X15OOFPublicationError(
            "batch manifest lacks its synthesized run config"
        )
    if _model_sha256(batch_run_config) != _require_sha256(
        document.get("batch_run_config_sha256"),
        field="batch manifest.batch_run_config_sha256",
    ):
        raise X15OOFPublicationError(
            "batch synthesized run config SHA-256 mismatch"
        )
    batch_seed_contract_sha256 = _require_sha256(
        document.get("effective_seed_contract_sha256"),
        field="batch manifest.effective_seed_contract_sha256",
    )
    expected_fold_ids = tuple(fold[0] for fold in EXPECTED_FOLDS)
    expected_cells = tuple(
        (fold_id, model_id, feature_block_id)
        for fold_id in expected_fold_ids
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX
    )
    if (
        _tuple_value(document.get("expected_fold_ids"))
        != expected_fold_ids
        or _tuple_value(document.get("observed_fold_ids"))
        != expected_fold_ids
        or _tuple_value(document.get("missing_fold_ids")) != ()
    ):
        raise X15OOFPublicationError(
            "selection-ready batch does not bind all five frozen folds"
        )
    if (
        _canonical(document.get("baseline_design_cell"))
        != _canonical(BASELINE_DESIGN_CELL)
        or _canonical(document.get("negative_control_design_matrix"))
        != _canonical(NEGATIVE_CONTROL_DESIGN_MATRIX)
        or document.get("negative_control_semantics")
        != "TIME_GRID_ONLY_MODEL_COMPLEXITY_CONTROL_NOT_SELECTABLE"
        or _canonical(document.get("selectable_candidate_design_matrix"))
        != _canonical(SELECTABLE_CANDIDATE_DESIGN_MATRIX)
    ):
        raise X15OOFPublicationError(
            "selection-ready batch candidate/negative-control matrix drifted"
        )
    if (
        _execution_cells(
            document.get("expected_execution_cells"),
            field="batch expected_execution_cells",
        )
        != expected_cells
        or _execution_cells(
            document.get("observed_execution_cells"),
            field="batch observed_execution_cells",
        )
        != expected_cells
        or _execution_cells(
            document.get("missing_execution_cells"),
            field="batch missing_execution_cells",
        )
        != ()
    ):
        raise X15OOFPublicationError(
            "selection-ready batch execution matrix is incomplete"
        )
    shards = document.get("shards")
    if not isinstance(shards, list) or len(shards) != 55:
        raise X15OOFPublicationError(
            "selection-ready batch must index 55 execution shards"
        )
    indexed_cells: list[tuple[str, str, str]] = []
    for reference in shards:
        if not isinstance(reference, dict):
            raise X15OOFPublicationError(
                "selection-ready batch shard reference is malformed"
            )
        if (
            _require_sha256(
                reference.get("effective_seed_contract_sha256"),
                field="batch shard effective_seed_contract_sha256",
            )
            != batch_seed_contract_sha256
        ):
            raise X15OOFPublicationError(
                "selection-ready batch shard seed contract drifted"
            )
        _require_sha256(
            reference.get("selection_fold_arrays_sha256"),
            field="batch shard selection_fold_arrays_sha256",
        )
        indexed_cells.append(
            (
                str(reference.get("fold_id", "")),
                str(reference.get("model_id", "")),
                str(reference.get("feature_block_id", "")),
            )
        )
    if tuple(indexed_cells) != expected_cells:
        raise X15OOFPublicationError(
            "selection-ready batch shard index differs from frozen matrix"
        )
    return document, path, observed_manifest_sha256


def _selection_shard_frame(
    *,
    reference: Mapping[str, object],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = _resolve_under(
        output_root,
        str(reference.get("manifest_path", "")),
        field="selection shard manifest",
    )
    document, _ = _read_shard_document(
        manifest_path=manifest_path,
        output_root=output_root,
        load_tables=False,
    )
    observed_manifest_sha256, _ = sha256_path(manifest_path)
    for field in (
        "fold_id",
        "model_id",
        "feature_block_id",
        "effective_seed_contract_sha256",
        "selection_fold_arrays_sha256",
    ):
        if document.get(field) != reference.get(field):
            raise X15OOFPublicationError(
                f"selection shard reference disagrees on {field}"
            )
    if (
        observed_manifest_sha256
        != reference.get("manifest_sha256")
        or document.get("bundle_sha256")
        != reference.get("shard_bundle_sha256")
    ):
        raise X15OOFPublicationError(
            "selection shard reference hash mismatch"
        )
    tables = document.get("tables")
    if not isinstance(tables, list):
        raise X15OOFPublicationError(
            "selection shard table descriptors are malformed"
        )
    prediction = next(
        (
            descriptor
            for descriptor in tables
            if isinstance(descriptor, dict)
            and descriptor.get("name") == "oof_predictions"
        ),
        None,
    )
    if prediction is None:
        raise X15OOFPublicationError(
            "selection shard lacks oof_predictions"
        )
    object_path = _resolve_under(
        output_root,
        str(prediction.get("object_path", "")),
        field="selection oof_predictions object",
    )
    try:
        frame = pq.read_table(
            object_path,
            columns=list(_SELECTION_PARQUET_COLUMNS),
            filters=_SELECTION_FILTERS,
        ).to_pandas()
    except Exception as error:
        raise X15OOFPublicationError(
            "selection projection columns are unavailable"
        ) from error
    if frame.empty:
        raise X15OOFPublicationError(
            "selection shard has no native Polymarket OOF rows"
        )
    if (
        not frame["venue"].eq("polymarket").all()
        or not frame["training_venue"].eq("polymarket").all()
        or not frame["calibration_venue"].eq("polymarket").all()
        or not frame["transport_mode"].eq("VENUE_SPECIFIC").all()
    ):
        raise X15OOFPublicationError(
            "selection projection must contain native Polymarket rows only"
        )
    selection_fold_arrays = document["selection_fold_arrays"]
    if not isinstance(selection_fold_arrays, dict):
        raise X15OOFPublicationError(
            "selection shard fold arrays are malformed"
        )
    fold_arrays = {
        column: tuple(selection_fold_arrays[column])
        for column in _FOLD_ARRAY_COLUMNS
    }
    for column, value in fold_arrays.items():
        frame[column] = pd.Series(
            [value] * len(frame),
            index=frame.index,
            dtype=object,
        )
    frame = frame.loc[:, list(SELECTION_PROJECTION_COLUMNS)]
    if tuple(frame.columns) != SELECTION_PROJECTION_COLUMNS:
        raise X15OOFPublicationError(
            "selection projection column order drifted"
        )
    return frame, document


def load_x15_oof_selection_projection(
    *,
    batch_manifest_path: str | Path,
    output_root: str | Path,
    candidate_model_id: str,
    candidate_feature_block_id: str,
) -> X15ModelRun:
    """Load only baseline + candidate projected OOF columns.

    The method reads ten ``oof_predictions`` projections (two model/block
    cells across five folds).  It does not load conditional quantiles, fold
    metrics, support audits, weight audits, or unrelated wide columns.
    """

    candidate = (candidate_model_id, candidate_feature_block_id)
    baseline = BASELINE_DESIGN_CELL
    if candidate not in SELECTABLE_CANDIDATE_DESIGN_MATRIX:
        raise X15OOFPublicationError(
            "selection candidate must be in the frozen selectable matrix; "
            "logistic/XGBoost D0 are negative controls only"
        )
    root = Path(output_root).resolve()
    document, batch_path, batch_manifest_sha256 = (
        _read_selection_ready_batch(
            batch_manifest_path=batch_manifest_path,
            output_root=root,
        )
    )
    shard_values = document["shards"]
    if not isinstance(shard_values, list):
        raise X15OOFPublicationError("batch shard index is malformed")
    by_cell: dict[tuple[str, str, str], dict[str, object]] = {}
    for value in shard_values:
        if not isinstance(value, dict):
            raise X15OOFPublicationError(
                "batch shard reference is malformed"
            )
        cell = (
            str(value.get("fold_id", "")),
            str(value.get("model_id", "")),
            str(value.get("feature_block_id", "")),
        )
        if cell in by_cell:
            raise X15OOFPublicationError(
                "batch shard index contains duplicate execution cells"
            )
        by_cell[cell] = value

    frames: list[pd.DataFrame] = []
    source_shards: list[dict[str, object]] = []
    expected_fold_ids = tuple(fold[0] for fold in EXPECTED_FOLDS)
    for fold_id in expected_fold_ids:
        for model_id, feature_block_id in (baseline, candidate):
            cell = (fold_id, model_id, feature_block_id)
            reference = by_cell.get(cell)
            if reference is None:
                raise X15OOFPublicationError(
                    "selection batch lacks required projected shard "
                    f"{fold_id}/{model_id}/{feature_block_id}"
                )
            frame, shard_document = _selection_shard_frame(
                reference=reference,
                output_root=root,
            )
            frames.append(frame)
            source_shards.append(
                {
                    "fold_id": fold_id,
                    "model_id": model_id,
                    "feature_block_id": feature_block_id,
                    "manifest_sha256": reference["manifest_sha256"],
                    "shard_bundle_sha256": shard_document["bundle_sha256"],
                    "oof_predictions_sha256": next(
                        table["object_sha256"]
                        for table in shard_document["tables"]
                        if table["name"] == "oof_predictions"
                    ),
                }
            )
    predictions = pd.concat(frames, ignore_index=True)
    if tuple(
        sorted(set(predictions["fold_id"].astype(str)))
    ) != expected_fold_ids:
        raise X15OOFPublicationError(
            "selection projection does not cover all five folds"
        )

    batch_run_config = dict(document["batch_run_config"])
    for field in ("fold_ids", "model_ids", "feature_block_ids"):
        batch_run_config.pop(field, None)
    batch_run_config["fold_ids"] = expected_fold_ids
    batch_run_config["model_ids"] = (
        baseline[0],
        candidate_model_id,
    )
    batch_run_config["feature_block_ids"] = (
        baseline[1],
        candidate_feature_block_id,
    )
    batch_run_config["source_shards"] = tuple(source_shards)
    batch_run_config["source_batch_manifest_path"] = (
        batch_path.relative_to(root).as_posix()
    )
    batch_run_config["source_batch_manifest_sha256"] = (
        batch_manifest_sha256
    )
    batch_run_config["source_batch_sha256"] = document["batch_sha256"]
    run_config_sha256 = _model_sha256(batch_run_config)
    return X15ModelRun(
        oof_predictions=predictions,
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256=run_config_sha256,
        run_config=batch_run_config,
    )


__all__ = [
    "BASELINE_DESIGN_CELL",
    "BATCH_SCHEMA",
    "BUILDER_VERSION",
    "NEGATIVE_CONTROL_DESIGN_MATRIX",
    "OOF_TABLE_NAMES",
    "PROBABILITY_DESIGN_MATRIX",
    "SELECTION_PROJECTION_COLUMNS",
    "SELECTABLE_CANDIDATE_DESIGN_MATRIX",
    "SHARD_SCHEMA",
    "X15OOFBatchPublication",
    "X15OOFShardPublication",
    "X15OOFPublicationError",
    "load_x15_oof_selection_projection",
    "load_x15_oof_shard",
    "publish_x15_oof_batch",
    "publish_x15_oof_shard",
]
