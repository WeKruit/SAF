"""Development-only venue validation and metadata-only pre-holdout audit.

There is intentionally no final-holdout prediction or reaction-read API in
this module.  Task 5 may bind the frozen 153/81 cohort metadata, inspect its
target-specific exposure declarations, validate genuine Task4 development OOF
transport, and lock a metadata-only ledger.  Reaction data remains unreachable
here, and prior X-11 sports-outcome exposure forbids Stage-A outcome validation.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from prediction_market.research.nfl_x15_models import X15ModelRun
from prediction_market.research.nfl_x15_model_selection import (
    ANCHOR_ENDPOINT_SECONDS,
    ANCHOR_LANDMARK_SECONDS,
    BASELINE_FEATURE_BLOCK_ID,
    BASELINE_MODEL_ID,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DIRECTION_THRESHOLD_PROBABILITY,
    DIRECTION_THRESHOLD_SEMANTICS,
    FrozenDevelopmentAuthority,
    HISTORICAL_ANALYSIS_SCOPE,
    HISTORICAL_CLAIM_BOUNDARY,
    HISTORICAL_SCHEMA_VERSION,
    HISTORICAL_TARGET_CONTRACT,
    LOSS_IMPROVEMENT_SIGN_SEMANTICS,
    MARKET_CONTINUITY_SUPPORT,
    ModelSelectionError,
    ModelSelectionResult,
    SURVIVAL_PROBABILITY_CONTRACT,
    StageBModelSelectionResult,
    VENUE_TICK_SUPPORT,
    bind_frozen_development_authority,
    verify_stage_b_v3_selection_result,
)
from prediction_market.research.nfl_x15_selection_batch_v3 import (
    X15SelectionBatchV3Error,
    load_x15_selection_projection_v3,
)
from prediction_market.research.nfl_x15_statistics import (
    DEFAULT_MAX_GAME_CONTRIBUTION,
    DEFAULT_MAX_Q_VALUE,
    DEFAULT_MIN_LOO_SAME_SIGN,
)


EXPECTED_DEVELOPMENT_GAMES: Final[int] = 153
EXPECTED_HOLDOUT_GAMES: Final[int] = 81
SOURCE_VENUE: Final[str] = "polymarket"
TARGET_VENUE: Final[str] = "kalshi"
VENUE_SPECIFIC_MODE: Final[str] = "VENUE_SPECIFIC"
TRANSPORT_MODE: Final[str] = "NO_TARGET_RECALIBRATION"
MIN_TRANSPORT_PAIRED_GAMES: Final[int] = 30
MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD: Final[int] = 30
MIN_TRANSPORT_EVALUATED_ROWS_PER_HEAD: Final[int] = 30
MAX_LOG_LOSS_DEGRADATION: Final[float] = 0.25
MAX_BRIER_DEGRADATION: Final[float] = 0.10
FACTOR_PREDICTIVE_UTILITY_SCHEMA_VERSION: Final[str] = (
    "NFLFactorPredictiveUtilityShortlistV1"
)
X11_SPORTS_OUTCOME_EVIDENCE_SHA256: Final[str] = (
    "sha256:1d0c033459c69778e265be3fca16ae2c87f650d5003a61ffdea4c020a4fd0b05"
)
X11_HOLDOUT_DRIVE_OUTCOME_COUNT: Final[int] = 1_683
_REVIEWED_EXPOSURE_ARTIFACT_PATH: Final[str] = (
    "artifacts/market-observation/nfl/x13/historical-holdout/"
    "sha256-6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294/"
    "6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
    ".holdout-selection.json"
)
_REVIEWED_EXPOSURE_ARTIFACT_SHA256: Final[str] = (
    "sha256:1685ee5084d99670abeb623bba16c9dd310b972c1b7778c0a967d171dd679034"
)
_REACTION_EXPOSURE_ARTIFACT_PATH: Final[str] = (
    "artifacts/market-observation/nfl/x13/dense-reaction-v3/"
    "kalshi-native-time-v3/exact-153/manifests/sha256/a2/"
    "a2aa20c764523d9c117e0087eb319cc50bbf736924009c9f7b8850fb471a3ea4"
    ".manifest.json"
)
_REACTION_EXPOSURE_ARTIFACT_SHA256: Final[str] = (
    "sha256:a2aa20c764523d9c117e0087eb319cc50bbf736924009c9f7b8850fb471a3ea4"
)
_FACTOR_REGISTRY_PATH: Final[str] = (
    "registries/factors/nfl_factor_registry_v4.json"
)
_FACTOR_REGISTRY_FILE_SHA256: Final[str] = (
    "sha256:92e5001d92afa0748731b5310dae8289ff6930b26a141e981ed910d2c761575f"
)
_FACTOR_REGISTRY_SEMANTIC_SHA256: Final[str] = (
    "sha256:527a084317ec4a728e5567feea756c1541b65bb814fcf96900b6cfbfd223ead8"
)
_FACTOR_FACTS_AUTHORITY_MANIFEST_PATH: Final[str] = (
    "artifacts/market-observation/nfl/x13/exact-153-facts-v4/"
    "batches/manifests/sha256/5d/"
    "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    ".batch-index.json"
)
_FACTOR_FACTS_AUTHORITY_MANIFEST_FILE_SHA256: Final[str] = (
    "sha256:5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
)
_FACTOR_FACTS_AUTHORITY_BATCH_SHA256: Final[str] = (
    "sha256:b097f35c30312068ca46e43a0d97e692f30f51a9dcdb89fc1ee604d1be98a082"
)
_FACTOR_MEMBERSHIP_ROOT: Final[str] = (
    "artifacts/market-observation/nfl/x15/"
    "historical-trades-only-development-panel-v2"
)
_FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_PATH: Final[str] = (
    f"{_FACTOR_MEMBERSHIP_ROOT}/batches/manifests/sha256/39/"
    "39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3"
    ".batch-index.json"
)
_FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_FILE_SHA256: Final[str] = (
    "sha256:39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3"
)
_FACTOR_MEMBERSHIP_AUTHORITY_BATCH_SHA256: Final[str] = (
    "sha256:d0cb73d381eeb39a7cf5d4cb2ebf24f05037be3e25905ac5e24977c72b3baba8"
)
_FACTOR_MEMBERSHIP_COHORT_AUTHORITY_SHA256: Final[str] = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
_FACTOR_MEMBERSHIP_EPISODE_COUNT: Final[int] = 25_408
_FACTOR_MEMBERSHIP_ROW_COUNT: Final[int] = 83_659
_FACTOR_MEMBERSHIP_ROWS_SHA256: Final[str] = (
    "sha256:d2fc72c3d81720bcf2bf2a7550272f734544923abbfec72d2165e12cc634a874"
)
_FACTOR_MEMBERSHIP_ARTIFACT_BINDINGS_SHA256: Final[str] = (
    "sha256:0725380c27e0353a0f6c92bef482b72b757970981cf6c31102f93fb6c64047c4"
)
_FACTOR_MEMBERSHIP_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "atomic_information_episode_id",
    "factor_id",
    "factor_version",
    "registry_sha256",
)

_AUTHORITY_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "game_id",
        "cohort",
        "nfl_week",
        "kickoff_utc",
        "batch_sha256",
        "cohort_authority_sha256",
        "market_reaction_exposure",
        "sports_outcome_exposure",
        "reaction_read_count",
    }
)
_TRANSPORT_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "nfl_week",
    "atomic_information_episode_id",
    "landmark_seconds",
    "endpoint_seconds",
    "fold_id",
    "model_id",
    "feature_block_id",
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
)
_MODEL_COMPARISON_PAIR_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in _TRANSPORT_PAIR_COLUMNS
    if column not in {"model_id", "feature_block_id"}
)
_CROSS_VENUE_FACTOR_PAIR_GRAIN: Final[tuple[str, ...]] = (
    "game_id",
    "nfl_week",
    "atomic_information_episode_id",
    "factor_id",
    "factor_version",
    "landmark_seconds",
    "endpoint_seconds",
    "fold_id",
    "candidate_model_id",
    "candidate_feature_block_id",
    "baseline_model_id",
    "baseline_feature_block_id",
)
_TRANSPORT_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "source_row_id",
        *_TRANSPORT_PAIR_COLUMNS,
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        "actual_home_contract_id",
        "train_weeks",
        "validation_weeks",
        "training_game_ids",
        "validation_game_ids",
        "preprocessor_fit_game_ids",
        "calibrator_fit_game_ids_s_h",
        "calibrator_fit_game_ids_o_h_given_s",
        "calibrator_fit_game_ids_direction",
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
        "s_h_calibrated_probability",
        "o_h_given_s_calibrated_probability",
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
        "s_h_truth",
        "o_h_given_s_truth",
        "direction_truth",
    }
)
_HASH_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
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
_SEQUENCE_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "train_weeks",
    "validation_weeks",
    "training_game_ids",
    "validation_game_ids",
    "preprocessor_fit_game_ids",
    "calibrator_fit_game_ids_s_h",
    "calibrator_fit_game_ids_o_h_given_s",
    "calibrator_fit_game_ids_direction",
)
_SOURCE_ONLY_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    *_HASH_PROVENANCE_COLUMNS,
    *_SEQUENCE_PROVENANCE_COLUMNS,
)
_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "s_h_calibrated_probability",
    "o_h_given_s_calibrated_probability",
    "direction_calibrated_prob_down",
    "direction_calibrated_prob_no_move",
    "direction_calibrated_prob_up",
)
_ALLOWED_METADATA_ACCESS: Final[frozenset[str]] = frozenset(
    {
        "authority_manifest_read",
        "batch_manifest_read",
        "cohort_metadata_read",
        "prior_exposure_audit",
    }
)


class KalshiValidationError(ValueError):
    """A row or audit operation violates the sealed Task 5 contract."""


@dataclass(frozen=True, slots=True)
class FrozenHistoricalExposureEvidence:
    """IDs derived from two fixed, hash-verified tracked audit artifacts."""

    reviewed_artifact_path: str
    reviewed_artifact_sha256: str
    reviewed_game_ids: tuple[str, ...]
    reviewed_game_ids_sha256: str
    reaction_artifact_path: str
    reaction_artifact_sha256: str
    reaction_game_ids: tuple[str, ...]
    reaction_game_ids_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenFactorMembershipEvidence:
    """Membership decoded from one fixed, fully hash-bound 153-game batch."""

    authority_manifest_path: str
    authority_manifest_file_sha256: str
    authority_batch_sha256: str
    facts_authority_manifest_path: str
    facts_authority_manifest_file_sha256: str
    facts_authority_batch_sha256: str
    cohort_authority_sha256: str
    registry_path: str
    registry_file_sha256: str
    registry_semantic_sha256: str
    membership_rows_sha256: str
    membership_artifact_bindings_sha256: str
    membership_rows: pd.DataFrame


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise KalshiValidationError(f"{field} must be a sha256 digest")
    return str(value)


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_game_id_evidence(game_ids: tuple[str, ...]) -> str:
    """Hash a canonical authoritative game-ID evidence set."""

    if not isinstance(game_ids, (tuple, list)):
        raise KalshiValidationError(
            "game-ID evidence must be a tuple/list"
        )
    raw_ids = tuple(game_ids)
    if (
        not all(
            isinstance(game_id, str) and bool(game_id.strip())
            for game_id in raw_ids
        )
        or len(raw_ids) != len(set(raw_ids))
    ):
        raise KalshiValidationError(
            "game-ID evidence must contain unique nonempty IDs"
        )
    normalized = tuple(sorted(raw_ids))
    return _canonical_sha256(
        {
            "schema": "authoritative_historical_game_id_evidence_v1",
            "game_ids": normalized,
        }
    )


def load_frozen_historical_exposure_evidence(
) -> FrozenHistoricalExposureEvidence:
    """Read the fixed reviewed/reaction audits and derive their game IDs."""

    repository_root = Path(__file__).resolve().parents[3]
    reviewed_path = repository_root / _REVIEWED_EXPOSURE_ARTIFACT_PATH
    reaction_path = repository_root / _REACTION_EXPOSURE_ARTIFACT_PATH
    try:
        reviewed_bytes = reviewed_path.read_bytes()
        reaction_bytes = reaction_path.read_bytes()
        reviewed_payload = json.loads(reviewed_bytes)
        reaction_payload = json.loads(reaction_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise KalshiValidationError(
            "fixed historical exposure audit artifacts are unreadable"
        ) from exc
    reviewed_file_hash = "sha256:" + hashlib.sha256(
        reviewed_bytes
    ).hexdigest()
    reaction_file_hash = "sha256:" + hashlib.sha256(
        reaction_bytes
    ).hexdigest()
    if reviewed_file_hash != _REVIEWED_EXPOSURE_ARTIFACT_SHA256:
        raise KalshiValidationError(
            "fixed reviewed exposure artifact SHA-256 mismatch"
        )
    if reaction_file_hash != _REACTION_EXPOSURE_ARTIFACT_SHA256:
        raise KalshiValidationError(
            "fixed reaction exposure artifact SHA-256 mismatch"
        )
    if (
        reviewed_payload.get("artifact_type")
        != "nfl_x13_historical_holdout_selection"
        or reviewed_payload.get("status") != "FROZEN"
        or reviewed_payload.get("eligible_game_count") != 20
        or reviewed_payload.get("claim_boundary")
        != {
            "event_result_used": False,
            "reaction_used": False,
            "trade_price_used": False,
            "volume_used": False,
        }
    ):
        raise KalshiValidationError(
            "fixed reviewed exposure artifact contract drifted"
        )
    reviewed_ids = tuple(
        sorted(reviewed_payload.get("eligible_game_ids", ()))
    )
    if (
        len(reviewed_ids) != 20
        or tuple(sorted(reviewed_payload.get("selected_game_ids", ())))
        != reviewed_ids
    ):
        raise KalshiValidationError(
            "fixed reviewed exposure artifact game IDs drifted"
        )
    if (
        reaction_payload.get("schema")
        != "nfl_x13_dense_reaction_publication_v1"
        or reaction_payload.get("cohort") != "development"
        or reaction_payload.get("game_count") != 153
        or reaction_payload.get("holdout_reaction_accessed") is not False
        or reaction_payload.get("final_holdout_access") != "CLOSED"
    ):
        raise KalshiValidationError(
            "fixed reaction exposure artifact contract drifted"
        )
    games = reaction_payload.get("games")
    if not isinstance(games, list):
        raise KalshiValidationError(
            "fixed reaction exposure artifact games are invalid"
        )
    reaction_ids = tuple(
        sorted(
            game.get("game_id")
            for game in games
            if isinstance(game, dict)
        )
    )
    if len(reaction_ids) != 153:
        raise KalshiValidationError(
            "fixed reaction exposure artifact game IDs drifted"
        )
    return FrozenHistoricalExposureEvidence(
        reviewed_artifact_path=_REVIEWED_EXPOSURE_ARTIFACT_PATH,
        reviewed_artifact_sha256=reviewed_file_hash,
        reviewed_game_ids=reviewed_ids,
        reviewed_game_ids_sha256=hash_game_id_evidence(reviewed_ids),
        reaction_artifact_path=_REACTION_EXPOSURE_ARTIFACT_PATH,
        reaction_artifact_sha256=reaction_file_hash,
        reaction_game_ids=reaction_ids,
        reaction_game_ids_sha256=hash_game_id_evidence(reaction_ids),
    )


def _factor_membership_rows_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, list(_FACTOR_MEMBERSHIP_COLUMNS)].sort_values(
        [
            "game_id",
            "atomic_information_episode_id",
            "factor_id",
            "factor_version",
        ],
        kind="mergesort",
    )
    records = [
        {
            column: str(row[column])
            for column in _FACTOR_MEMBERSHIP_COLUMNS
        }
        for row in ordered.to_dict("records")
    ]
    return _canonical_sha256(
        {
            "schema": "nfl_x15_frozen_factor_membership_v1",
            "rows": records,
        }
    )


def _path_under(root: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise KalshiValidationError(f"{label} path is invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise KalshiValidationError(
            f"{label} path escapes its fixed artifact root"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise KalshiValidationError(f"{label} artifact is not a regular file")
    return resolved


def _sha256_file_bytes(path: Path, *, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise KalshiValidationError(f"{label} artifact is unreadable") from exc
    return payload


def _validate_factor_registry_contract(
    registry: object,
) -> None:
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema") != "NFLFactorRegistryV4"
        or registry.get("version") != "v4"
        or registry.get("status") != "AUTHORITATIVE"
        or _canonical_sha256(registry)
        != _FACTOR_REGISTRY_SEMANTIC_SHA256
    ):
        raise KalshiValidationError(
            "frozen factor registry semantic contract drifted"
        )
    definitions = registry.get("factors")
    if (
        not isinstance(definitions, list)
        or len(definitions) != 59
        or any(
            not isinstance(definition, dict)
            or definition.get("status") not in {"ACTIVE", "DATA_GAP"}
            for definition in definitions
        )
    ):
        raise KalshiValidationError(
            "frozen factor registry definitions are invalid"
        )


def _validate_factor_facts_authority_contract(
    authority: object,
) -> None:
    if not isinstance(authority, Mapping):
        raise KalshiValidationError(
            "frozen V4 facts authority contract drifted"
        )
    games = authority.get("games")
    registry = authority.get("factor_registry")
    cohort_authority = authority.get("authority")
    if (
        authority.get("schema")
        != "nfl_x13_exact153_fact_batch_index_v4"
        or authority.get("builder_version")
        != "nfl_x13_exact153_fact_publication_v4"
        or authority.get("publication_id") != "exact-153-facts-v4"
        or authority.get("experiment_id") != "X-13"
        or authority.get("cohort") != "development"
        or authority.get("game_count") != EXPECTED_DEVELOPMENT_GAMES
        or authority.get("holdout_reaction_accessed") is not False
        or authority.get("market_data_read") is not False
        or authority.get("publication_gate") != "PASS"
        or authority.get("batch_sha256")
        != _FACTOR_FACTS_AUTHORITY_BATCH_SHA256
        or not isinstance(games, list)
        or len(games) != EXPECTED_DEVELOPMENT_GAMES
        or not isinstance(registry, Mapping)
        or registry.get("schema") != "NFLFactorRegistryV4"
        or registry.get("version") != "v4"
        or registry.get("status") != "AUTHORITATIVE"
        or registry.get("factor_count") != 59
        or registry.get("file_sha256")
        != _FACTOR_REGISTRY_FILE_SHA256
        or registry.get("semantic_sha256")
        != _FACTOR_REGISTRY_SEMANTIC_SHA256
        or not isinstance(cohort_authority, Mapping)
        or cohort_authority.get("development_game_count")
        != EXPECTED_DEVELOPMENT_GAMES
        or cohort_authority.get("final_holdout_game_count")
        != EXPECTED_HOLDOUT_GAMES
        or cohort_authority.get("object_sha256")
        != _FACTOR_MEMBERSHIP_COHORT_AUTHORITY_SHA256
    ):
        raise KalshiValidationError(
            "frozen V4 facts authority contract drifted"
        )


def _validate_factor_membership_authority_contract(
    authority: object,
) -> None:
    if not isinstance(authority, Mapping):
        raise KalshiValidationError(
            "frozen factor membership authority contract drifted"
        )
    games = authority.get("games")
    source_batches = authority.get("source_batch_file_sha256s")
    if (
        authority.get("schema")
        != "nfl_x15_development_panel_batch_index_v1"
        or authority.get("builder_version")
        != "nfl-x15-development-panel-v2"
        or authority.get("cohort") != "development"
        or authority.get("verified_development_game_count")
        != EXPECTED_DEVELOPMENT_GAMES
        or authority.get("published_game_count")
        != EXPECTED_DEVELOPMENT_GAMES
        or authority.get("partial_publication") is not False
        or authority.get("holdout_reaction_accessed") is not False
        or authority.get("publication_gate") != "PASS"
        or authority.get("cohort_authority_sha256")
        != _FACTOR_MEMBERSHIP_COHORT_AUTHORITY_SHA256
        or authority.get("batch_sha256")
        != _FACTOR_MEMBERSHIP_AUTHORITY_BATCH_SHA256
        or not isinstance(source_batches, Mapping)
        or source_batches.get("facts")
        != _FACTOR_FACTS_AUTHORITY_MANIFEST_FILE_SHA256
        or not isinstance(games, list)
        or len(games) != EXPECTED_DEVELOPMENT_GAMES
    ):
        raise KalshiValidationError(
            "frozen factor membership authority contract drifted"
        )


def load_frozen_factor_membership_evidence(
) -> FrozenFactorMembershipEvidence:
    """Decode exact factor tags from the fixed complete X15 batch only."""

    repository_root = Path(__file__).resolve().parents[3]
    registry_path = repository_root / _FACTOR_REGISTRY_PATH
    facts_authority_path = (
        repository_root / _FACTOR_FACTS_AUTHORITY_MANIFEST_PATH
    )
    authority_path = (
        repository_root / _FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_PATH
    )
    registry_bytes = _sha256_file_bytes(
        registry_path, label="frozen factor registry"
    )
    facts_authority_bytes = _sha256_file_bytes(
        facts_authority_path, label="frozen V4 facts authority"
    )
    authority_bytes = _sha256_file_bytes(
        authority_path, label="frozen factor membership authority"
    )
    registry_file_sha256 = (
        "sha256:" + hashlib.sha256(registry_bytes).hexdigest()
    )
    facts_authority_file_sha256 = (
        "sha256:" + hashlib.sha256(facts_authority_bytes).hexdigest()
    )
    authority_file_sha256 = (
        "sha256:" + hashlib.sha256(authority_bytes).hexdigest()
    )
    if registry_file_sha256 != _FACTOR_REGISTRY_FILE_SHA256:
        raise KalshiValidationError(
            "frozen factor registry file SHA-256 mismatch"
        )
    if (
        facts_authority_file_sha256
        != _FACTOR_FACTS_AUTHORITY_MANIFEST_FILE_SHA256
    ):
        raise KalshiValidationError(
            "frozen V4 facts authority SHA-256 mismatch"
        )
    if (
        authority_file_sha256
        != _FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_FILE_SHA256
    ):
        raise KalshiValidationError(
            "frozen factor membership authority SHA-256 mismatch"
        )
    try:
        registry = json.loads(registry_bytes)
        facts_authority = json.loads(facts_authority_bytes)
        authority = json.loads(authority_bytes)
    except json.JSONDecodeError as exc:
        raise KalshiValidationError(
            "frozen factor registry/facts/membership authority is invalid JSON"
        ) from exc
    _validate_factor_registry_contract(registry)
    _validate_factor_facts_authority_contract(facts_authority)
    _validate_factor_membership_authority_contract(authority)
    games = authority.get("games")
    cohort_authority_sha256 = _require_sha256(
        authority.get("cohort_authority_sha256"),
        field="factor membership cohort_authority_sha256",
    )
    authority_batch_sha256 = _require_sha256(
        authority.get("batch_sha256"),
        field="factor membership authority batch_sha256",
    )
    if (
        cohort_authority_sha256
        != _FACTOR_MEMBERSHIP_COHORT_AUTHORITY_SHA256
        or authority_batch_sha256
        != _FACTOR_MEMBERSHIP_AUTHORITY_BATCH_SHA256
    ):
        raise KalshiValidationError(
            "frozen factor membership embedded authority drifted"
        )
    facts_games = facts_authority.get("games")
    if not isinstance(facts_games, list):
        raise KalshiValidationError(
            "frozen V4 facts authority game descriptors are invalid"
        )
    facts_manifest_sha256_by_game: dict[str, str] = {}
    for facts_game in facts_games:
        if not isinstance(facts_game, Mapping):
            raise KalshiValidationError(
                "frozen V4 facts authority game descriptor is invalid"
            )
        facts_game_id = facts_game.get("game_id")
        if (
            not isinstance(facts_game_id, str)
            or not facts_game_id.strip()
            or facts_game_id in facts_manifest_sha256_by_game
        ):
            raise KalshiValidationError(
                "frozen V4 facts authority game identity is invalid"
            )
        facts_manifest_sha256_by_game[facts_game_id] = _require_sha256(
            facts_game.get("manifest_sha256"),
            field=f"V4 facts {facts_game_id} manifest_sha256",
        )
    fixed_root = repository_root / _FACTOR_MEMBERSHIP_ROOT
    frames: list[pd.DataFrame] = []
    artifact_bindings: list[dict[str, str]] = []
    observed_game_ids: set[str] = set()
    expected_schema_columns = [
        "game_id",
        "atomic_information_episode_id",
        "factor_id",
        "factor_version",
        "registry_sha256",
        "pbp_source_sha256",
        "predicate_evidence",
    ]
    for game in games:
        if not isinstance(game, dict):
            raise KalshiValidationError(
                "factor membership game descriptor is invalid"
            )
        game_id = game.get("game_id")
        if (
            not isinstance(game_id, str)
            or not game_id.strip()
            or game_id in observed_game_ids
        ):
            raise KalshiValidationError(
                "factor membership game identity is invalid"
            )
        observed_game_ids.add(game_id)
        manifest_sha256 = _require_sha256(
            game.get("manifest_sha256"),
            field=f"factor membership {game_id} manifest_sha256",
        )
        manifest_path = _path_under(
            fixed_root,
            game.get("manifest_path"),
            label=f"factor membership {game_id} manifest",
        )
        manifest_bytes = _sha256_file_bytes(
            manifest_path,
            label=f"factor membership {game_id} manifest",
        )
        if (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
            != manifest_sha256
        ):
            raise KalshiValidationError(
                f"factor membership {game_id} manifest SHA-256 mismatch"
            )
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise KalshiValidationError(
                f"factor membership {game_id} manifest is invalid JSON"
            ) from exc
        tables = manifest.get("tables")
        sources = manifest.get("sources")
        if (
            manifest.get("schema")
            != "nfl_x15_development_game_panel_manifest_v1"
            or manifest.get("game_id") != game_id
            or manifest.get("cohort") != "development"
            or manifest.get("holdout_reaction_accessed") is not False
            or not isinstance(tables, list)
            or not isinstance(sources, dict)
            or sources.get("cohort_authority_sha256")
            != cohort_authority_sha256
            or sources.get("facts_manifest_sha256")
            != facts_manifest_sha256_by_game.get(game_id)
        ):
            raise KalshiValidationError(
                f"factor membership {game_id} manifest contract drifted"
            )
        membership_tables = [
            table
            for table in tables
            if isinstance(table, dict)
            and table.get("name") == "factor_membership"
        ]
        if len(membership_tables) != 1:
            raise KalshiValidationError(
                f"factor membership {game_id} table binding is ambiguous"
            )
        descriptor = membership_tables[0]
        object_sha256 = _require_sha256(
            descriptor.get("object_sha256"),
            field=f"factor membership {game_id} object_sha256",
        )
        semantic_rows_sha256 = _require_sha256(
            descriptor.get("semantic_rows_sha256"),
            field=f"factor membership {game_id} semantic_rows_sha256",
        )
        if descriptor.get("schema_columns") != expected_schema_columns:
            raise KalshiValidationError(
                f"factor membership {game_id} schema drifted"
            )
        object_path = _path_under(
            fixed_root,
            descriptor.get("object_path"),
            label=f"factor membership {game_id} object",
        )
        object_bytes = _sha256_file_bytes(
            object_path,
            label=f"factor membership {game_id} object",
        )
        if (
            "sha256:" + hashlib.sha256(object_bytes).hexdigest()
            != object_sha256
        ):
            raise KalshiValidationError(
                f"factor membership {game_id} object SHA-256 mismatch"
            )
        try:
            frame = pd.read_parquet(object_path)
        except (OSError, ValueError) as exc:
            raise KalshiValidationError(
                f"factor membership {game_id} object is unreadable"
            ) from exc
        if (
            list(frame.columns) != expected_schema_columns
            or len(frame) != descriptor.get("row_count")
            or not frame["game_id"].eq(game_id).all()
        ):
            raise KalshiValidationError(
                f"factor membership {game_id} object contract drifted"
            )
        frames.append(
            frame.loc[:, list(_FACTOR_MEMBERSHIP_COLUMNS)].copy()
        )
        artifact_bindings.append(
            {
                "game_id": game_id,
                "game_manifest_sha256": manifest_sha256,
                "membership_object_sha256": object_sha256,
                "membership_semantic_rows_sha256": (
                    semantic_rows_sha256
                ),
            }
        )
    if observed_game_ids != set(facts_manifest_sha256_by_game):
        raise KalshiValidationError(
            "factor membership game set differs from frozen V4 facts"
        )
    membership = pd.concat(frames, ignore_index=True)
    for column in _FACTOR_MEMBERSHIP_COLUMNS:
        if not membership[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise KalshiValidationError(
                f"frozen factor membership {column} is invalid"
            )
    grain = [
        "game_id",
        "atomic_information_episode_id",
        "factor_id",
        "factor_version",
    ]
    if membership.duplicated(grain, keep=False).any():
        raise KalshiValidationError(
            "frozen factor membership grain is not unique"
        )
    if not membership["registry_sha256"].eq(
        _FACTOR_REGISTRY_SEMANTIC_SHA256
    ).all():
        raise KalshiValidationError(
            "frozen factor membership registry binding drifted"
        )
    definitions = registry.get("factors")
    if not isinstance(definitions, list):
        raise KalshiValidationError(
            "frozen factor registry definitions are invalid"
        )
    registered = {
        (definition.get("factor_id"), definition.get("version"))
        for definition in definitions
        if isinstance(definition, dict)
        and definition.get("status") == "ACTIVE"
    }
    observed = set(
        map(
            tuple,
            membership[
                ["factor_id", "factor_version"]
            ].drop_duplicates().to_numpy(),
        )
    )
    if not observed.issubset(registered):
        raise KalshiValidationError(
            "frozen membership contains unregistered factor identities"
        )
    membership = membership.sort_values(
        grain, kind="mergesort"
    ).reset_index(drop=True)
    membership_rows_sha256 = _factor_membership_rows_sha256(
        membership
    )
    bindings_sha256 = _canonical_sha256(
        {
            "schema": "nfl_x15_factor_membership_artifact_bindings_v1",
            "batch_manifest_file_sha256": authority_file_sha256,
            "tables": artifact_bindings,
        }
    )
    episode_count = membership[
        ["game_id", "atomic_information_episode_id"]
    ].drop_duplicates().shape[0]
    if (
        membership["game_id"].nunique() != EXPECTED_DEVELOPMENT_GAMES
        or episode_count != _FACTOR_MEMBERSHIP_EPISODE_COUNT
        or len(membership) != _FACTOR_MEMBERSHIP_ROW_COUNT
        or membership_rows_sha256 != _FACTOR_MEMBERSHIP_ROWS_SHA256
        or bindings_sha256
        != _FACTOR_MEMBERSHIP_ARTIFACT_BINDINGS_SHA256
    ):
        raise KalshiValidationError(
            "frozen factor membership complete authority drifted"
        )
    return FrozenFactorMembershipEvidence(
        authority_manifest_path=(
            _FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_PATH
        ),
        authority_manifest_file_sha256=authority_file_sha256,
        authority_batch_sha256=authority_batch_sha256,
        facts_authority_manifest_path=(
            _FACTOR_FACTS_AUTHORITY_MANIFEST_PATH
        ),
        facts_authority_manifest_file_sha256=(
            facts_authority_file_sha256
        ),
        facts_authority_batch_sha256=(
            _FACTOR_FACTS_AUTHORITY_BATCH_SHA256
        ),
        cohort_authority_sha256=cohort_authority_sha256,
        registry_path=_FACTOR_REGISTRY_PATH,
        registry_file_sha256=registry_file_sha256,
        registry_semantic_sha256=(
            _FACTOR_REGISTRY_SEMANTIC_SHA256
        ),
        membership_rows_sha256=membership_rows_sha256,
        membership_artifact_bindings_sha256=bindings_sha256,
        membership_rows=membership,
    )


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KalshiValidationError(f"{field} must be a nonempty string")
    return value


def _integer_series(
    frame: pd.DataFrame, column: str, *, lower: int, upper: int
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if (
        values.isna().any()
        or not np.equal(
            values.to_numpy(dtype=float),
            values.astype(int).to_numpy(dtype=float),
        ).all()
        or not values.between(lower, upper).all()
    ):
        raise KalshiValidationError(
            f"{column} must be integral in {lower}..{upper}"
        )
    return values.astype(int)


def _canonical_metadata_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    ordered = frame.sort_values(
        ["cohort", "game_id"], kind="mergesort"
    )
    records: list[dict[str, object]] = []
    for row in ordered.to_dict("records"):
        records.append(
            {
                "game_id": str(row["game_id"]),
                "cohort": str(row["cohort"]),
                "nfl_week": int(row["nfl_week"]),
                "kickoff_utc": str(row["kickoff_utc"]),
                "batch_sha256": str(row["batch_sha256"]),
                "cohort_authority_sha256": str(
                    row["cohort_authority_sha256"]
                ),
                "market_reaction_exposure": str(
                    row["market_reaction_exposure"]
                ),
                "sports_outcome_exposure": str(
                    row["sports_outcome_exposure"]
                ),
                "reaction_read_count": int(row["reaction_read_count"]),
            }
        )
    return records


def _recompute_authority_metadata_sha256(
    authority_metadata: FrozenAuthorityMetadata,
) -> str:
    """Rebind the current mutable DataFrame contents at the lock boundary."""

    development = authority_metadata.development
    holdout = authority_metadata.holdout
    if (
        not isinstance(development, pd.DataFrame)
        or not isinstance(holdout, pd.DataFrame)
        or len(development) != EXPECTED_DEVELOPMENT_GAMES
        or len(holdout) != EXPECTED_HOLDOUT_GAMES
        or development["game_id"].nunique()
        != EXPECTED_DEVELOPMENT_GAMES
        or holdout["game_id"].nunique() != EXPECTED_HOLDOUT_GAMES
        or set(development["game_id"]).intersection(holdout["game_id"])
        or not development["cohort"].eq("development").all()
        or not holdout["cohort"].eq("holdout").all()
    ):
        raise KalshiValidationError(
            "current authority metadata frame/count contract drifted"
        )
    frame = pd.concat([development, holdout], ignore_index=True)
    if not _AUTHORITY_REQUIRED.issubset(frame.columns):
        raise KalshiValidationError(
            "current authority metadata columns drifted"
        )
    historical_evidence = load_frozen_historical_exposure_evidence()
    holdout_ids = tuple(sorted(holdout["game_id"].astype(str)))
    return _canonical_sha256(
        {
            "cohort_metadata": _canonical_metadata_records(frame),
            "historical_reviewed_game_ids": tuple(
                sorted(historical_evidence.reviewed_game_ids)
            ),
            "historical_reviewed_game_ids_sha256": (
                authority_metadata.historical_reviewed_game_ids_sha256
            ),
            "historical_reviewed_artifact_path": (
                authority_metadata.historical_reviewed_artifact_path
            ),
            "historical_reviewed_artifact_sha256": (
                authority_metadata.historical_reviewed_artifact_sha256
            ),
            "historical_reaction_game_ids": tuple(
                sorted(historical_evidence.reaction_game_ids)
            ),
            "historical_reaction_game_ids_sha256": (
                authority_metadata.historical_reaction_game_ids_sha256
            ),
            "historical_reaction_artifact_path": (
                authority_metadata.historical_reaction_artifact_path
            ),
            "historical_reaction_artifact_sha256": (
                authority_metadata.historical_reaction_artifact_sha256
            ),
            "sports_outcome_exposed_game_ids": holdout_ids,
            "sports_outcome_exposed_game_ids_sha256": (
                authority_metadata.sports_outcome_exposed_game_ids_sha256
            ),
            "sports_outcome_source_evidence_sha256": (
                authority_metadata.sports_outcome_source_evidence_sha256
            ),
            "sports_outcome_observation_count": (
                authority_metadata.sports_outcome_observation_count
            ),
            "reaction_blind_metadata_game_ids": holdout_ids,
            "reaction_blind_metadata_game_ids_sha256": (
                authority_metadata
                .reaction_blind_metadata_game_ids_sha256
            ),
            "stage_a_outcome_validation_eligible": (
                authority_metadata.stage_a_outcome_validation_eligible
            ),
            "stage_b_market_reaction_validation_eligible": (
                authority_metadata
                .stage_b_market_reaction_validation_eligible
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenAuthorityMetadata:
    """Exact cohort metadata that can be inspected without reaction rows."""

    cohort_authority_sha256: str
    development: pd.DataFrame
    holdout: pd.DataFrame
    overlap_game_ids: tuple[str, ...]
    holdout_reviewed_overlap_game_ids: tuple[str, ...]
    holdout_reaction_overlap_game_ids: tuple[str, ...]
    historical_reviewed_artifact_path: str
    historical_reviewed_artifact_sha256: str
    historical_reaction_artifact_path: str
    historical_reaction_artifact_sha256: str
    historical_reviewed_game_ids_sha256: str
    historical_reaction_game_ids_sha256: str
    sports_outcome_exposed_game_ids_sha256: str
    sports_outcome_source_evidence_sha256: str
    sports_outcome_observation_count: int
    reaction_blind_metadata_game_ids_sha256: str
    prior_exposure_audit: pd.DataFrame
    holdout_reaction_read_count: int
    stage_a_outcome_validation_eligible: bool
    stage_b_market_reaction_validation_eligible: bool
    metadata_sha256: str
    selection_authority: FrozenDevelopmentAuthority


def bind_frozen_authority_metadata(
    cohort_metadata: pd.DataFrame,
    *,
    cohort_authority_sha256: str,
    sports_outcome_exposed_game_ids: tuple[str, ...],
    sports_outcome_exposed_game_ids_sha256: str,
    sports_outcome_source_evidence_sha256: str,
    sports_outcome_observation_count: int,
    reaction_blind_metadata_game_ids: tuple[str, ...],
    reaction_blind_metadata_game_ids_sha256: str,
) -> FrozenAuthorityMetadata:
    """Bind exact cohorts and target-specific holdout exposure evidence."""

    authority = _require_sha256(
        cohort_authority_sha256, field="cohort_authority_sha256"
    )
    historical_evidence = load_frozen_historical_exposure_evidence()
    reviewed_ids = historical_evidence.reviewed_game_ids
    reaction_ids = historical_evidence.reaction_game_ids
    sports_outcome_ids = tuple(sports_outcome_exposed_game_ids)
    reaction_blind_ids = tuple(reaction_blind_metadata_game_ids)
    reviewed_hash = historical_evidence.reviewed_game_ids_sha256
    reaction_hash = historical_evidence.reaction_game_ids_sha256
    sports_outcome_hash = _require_sha256(
        sports_outcome_exposed_game_ids_sha256,
        field="sports_outcome_exposed_game_ids_sha256",
    )
    sports_outcome_source_hash = _require_sha256(
        sports_outcome_source_evidence_sha256,
        field="sports_outcome_source_evidence_sha256",
    )
    reaction_blind_hash = _require_sha256(
        reaction_blind_metadata_game_ids_sha256,
        field="reaction_blind_metadata_game_ids_sha256",
    )
    if (
        hash_game_id_evidence(sports_outcome_ids)
        != sports_outcome_hash
    ):
        raise KalshiValidationError(
            "sports outcome game-ID evidence hash mismatch"
        )
    if (
        sports_outcome_source_hash
        != X11_SPORTS_OUTCOME_EVIDENCE_SHA256
    ):
        raise KalshiValidationError(
            "sports outcome source evidence must bind the frozen X-11 "
            "artifact SHA-256"
        )
    if (
        isinstance(sports_outcome_observation_count, bool)
        or not isinstance(sports_outcome_observation_count, Integral)
        or int(sports_outcome_observation_count)
        != X11_HOLDOUT_DRIVE_OUTCOME_COUNT
    ):
        raise KalshiValidationError(
            "sports outcome evidence must record exactly 1,683 exposed "
            "holdout drive outcomes"
        )
    if hash_game_id_evidence(reaction_blind_ids) != reaction_blind_hash:
        raise KalshiValidationError(
            "reaction-blind metadata game-ID evidence hash mismatch"
        )
    if not isinstance(cohort_metadata, pd.DataFrame):
        raise KalshiValidationError(
            "cohort_metadata must be a DataFrame"
        )
    if cohort_metadata.columns.duplicated().any():
        raise KalshiValidationError(
            "cohort_metadata has duplicate columns"
        )
    missing = sorted(
        _AUTHORITY_REQUIRED.difference(cohort_metadata.columns)
    )
    if missing:
        raise KalshiValidationError(
            f"cohort_metadata missing required columns: {missing}"
        )
    frame = cohort_metadata.loc[
        :, sorted(_AUTHORITY_REQUIRED)
    ].copy()
    for column in (
        "game_id",
        "cohort",
        "kickoff_utc",
        "batch_sha256",
        "cohort_authority_sha256",
        "market_reaction_exposure",
        "sports_outcome_exposure",
    ):
        if not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise KalshiValidationError(
                f"cohort_metadata {column} must be a nonempty string"
            )
    if frame["game_id"].duplicated().any():
        raise KalshiValidationError(
            "development and holdout game IDs must be globally unique"
        )
    if not frame["cohort"].isin(("development", "holdout")).all():
        raise KalshiValidationError(
            "cohort must be development or holdout"
        )
    if not frame["cohort_authority_sha256"].eq(authority).all():
        raise KalshiValidationError(
            "cohort metadata does not match cohort_authority_sha256"
        )
    if not frame["batch_sha256"].map(_is_sha256).all():
        raise KalshiValidationError(
            "every cohort batch_sha256 must be a sha256 digest"
        )

    kickoff = pd.to_datetime(
        frame["kickoff_utc"], utc=True, errors="coerce"
    )
    explicitly_utc = frame["kickoff_utc"].str.endswith(
        ("Z", "+00:00")
    )
    if kickoff.isna().any() or not explicitly_utc.all():
        raise KalshiValidationError(
            "kickoff_utc must be an explicit UTC timestamp"
        )
    frame["kickoff_utc"] = frame["kickoff_utc"].astype(str)

    development = frame.loc[
        frame["cohort"].eq("development")
    ].copy()
    holdout = frame.loc[frame["cohort"].eq("holdout")].copy()
    if len(development) != EXPECTED_DEVELOPMENT_GAMES:
        raise KalshiValidationError(
            "authority metadata must contain exactly 153 development games"
        )
    if len(holdout) != EXPECTED_HOLDOUT_GAMES:
        raise KalshiValidationError(
            "authority metadata must contain exactly 81 holdout games"
        )
    development["nfl_week"] = _integer_series(
        development, "nfl_week", lower=1, upper=12
    )
    holdout["nfl_week"] = _integer_series(
        holdout, "nfl_week", lower=13, upper=18
    )

    reaction_counts = pd.to_numeric(
        frame["reaction_read_count"], errors="coerce"
    )
    if (
        reaction_counts.isna().any()
        or not np.equal(
            reaction_counts.to_numpy(dtype=float),
            reaction_counts.astype(int).to_numpy(dtype=float),
        ).all()
        or (reaction_counts < 0).any()
    ):
        raise KalshiValidationError(
            "reaction_read_count must be a nonnegative integer"
        )
    frame["reaction_read_count"] = reaction_counts.astype(int)
    development["reaction_read_count"] = frame.loc[
        development.index, "reaction_read_count"
    ]
    holdout["reaction_read_count"] = frame.loc[
        holdout.index, "reaction_read_count"
    ]
    holdout_read_count = int(holdout["reaction_read_count"].sum())
    if holdout_read_count != 0:
        raise KalshiValidationError(
            "holdout reaction access counter must equal zero"
        )
    if not holdout["market_reaction_exposure"].eq(
        "SEALED_UNREAD"
    ).all():
        raise KalshiValidationError(
            "all 81 holdout games must declare "
            "market_reaction_exposure=SEALED_UNREAD"
        )
    if not holdout["sports_outcome_exposure"].eq(
        "PRIOR_EXPOSED_X11"
    ).all():
        raise KalshiValidationError(
            "all 81 holdout games must declare "
            "sports_outcome_exposure=PRIOR_EXPOSED_X11"
        )

    overlap = tuple(
        sorted(
            set(development["game_id"]).intersection(
                holdout["game_id"]
            )
        )
    )
    if overlap:
        raise KalshiValidationError(
            "development and holdout game IDs are not disjoint"
        )
    holdout_ids = set(holdout["game_id"].astype(str))
    reviewed_overlap = tuple(
        sorted(holdout_ids.intersection(reviewed_ids))
    )
    reaction_overlap = tuple(
        sorted(holdout_ids.intersection(reaction_ids))
    )
    if reviewed_overlap or reaction_overlap:
        raise KalshiValidationError(
            "authoritative historical reviewed/reaction evidence has "
            "holdout game overlap"
        )
    if set(sports_outcome_ids) != holdout_ids:
        raise KalshiValidationError(
            "authoritative sports outcome evidence must cover the "
            "exact 81-game holdout"
        )
    if set(reaction_blind_ids) != holdout_ids:
        raise KalshiValidationError(
            "authoritative reaction-blind metadata evidence must cover "
            "the exact 81-game holdout"
        )
    declaration_exposure = (
        frame.groupby(
            [
                "cohort",
                "market_reaction_exposure",
                "sports_outcome_exposure",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            game_count=("game_id", "nunique"),
            reaction_read_count=("reaction_read_count", "sum"),
        )
        .reset_index(drop=True)
    )
    declaration_exposure["evidence_kind"] = declaration_exposure[
        "cohort"
    ].map(lambda cohort: f"TARGET_SPECIFIC_METADATA:{cohort.upper()}")
    evidence_rows = pd.DataFrame(
        [
            {
                "cohort": "historical",
                "market_reaction_exposure": "REVIEWED_ID_SET",
                "sports_outcome_exposure": "NOT_APPLICABLE",
                "game_count": len(reviewed_ids),
                "reaction_read_count": 0,
                "evidence_kind": "HISTORICAL_REVIEWED_GAME_IDS",
                "artifact_path": (
                    historical_evidence.reviewed_artifact_path
                ),
                "source_evidence_sha256": (
                    historical_evidence.reviewed_artifact_sha256
                ),
            },
            {
                "cohort": "historical",
                "market_reaction_exposure": "REACTION_EXPOSED_ID_SET",
                "sports_outcome_exposure": "NOT_APPLICABLE",
                "game_count": len(reaction_ids),
                "reaction_read_count": len(reaction_ids),
                "evidence_kind": "HISTORICAL_REACTION_GAME_IDS",
                "artifact_path": (
                    historical_evidence.reaction_artifact_path
                ),
                "source_evidence_sha256": (
                    historical_evidence.reaction_artifact_sha256
                ),
            },
            {
                "cohort": "holdout",
                "market_reaction_exposure": "NOT_APPLICABLE",
                "sports_outcome_exposure": "PRIOR_EXPOSED_X11",
                "game_count": len(sports_outcome_ids),
                "reaction_read_count": 0,
                "evidence_kind": "SPORTS_OUTCOME_EXPOSED_GAME_IDS",
                "game_id_evidence_sha256": sports_outcome_hash,
                "source_evidence_sha256": sports_outcome_source_hash,
                "observation_count": int(
                    sports_outcome_observation_count
                ),
            },
            {
                "cohort": "holdout",
                "market_reaction_exposure": (
                    "REACTION_BLIND_METADATA_ONLY"
                ),
                "sports_outcome_exposure": "NOT_APPLICABLE",
                "game_count": len(reaction_blind_ids),
                "reaction_read_count": 0,
                "evidence_kind": "REACTION_BLIND_METADATA_GAME_IDS",
            },
        ]
    )
    exposure = pd.concat(
        [declaration_exposure, evidence_rows], ignore_index=True
    )
    metadata_hash = _canonical_sha256(
        {
            "cohort_metadata": _canonical_metadata_records(frame),
            "historical_reviewed_game_ids": tuple(
                sorted(reviewed_ids)
            ),
            "historical_reviewed_game_ids_sha256": reviewed_hash,
            "historical_reviewed_artifact_path": (
                historical_evidence.reviewed_artifact_path
            ),
            "historical_reviewed_artifact_sha256": (
                historical_evidence.reviewed_artifact_sha256
            ),
            "historical_reaction_game_ids": tuple(
                sorted(reaction_ids)
            ),
            "historical_reaction_game_ids_sha256": reaction_hash,
            "historical_reaction_artifact_path": (
                historical_evidence.reaction_artifact_path
            ),
            "historical_reaction_artifact_sha256": (
                historical_evidence.reaction_artifact_sha256
            ),
            "sports_outcome_exposed_game_ids": tuple(
                sorted(sports_outcome_ids)
            ),
            "sports_outcome_exposed_game_ids_sha256": (
                sports_outcome_hash
            ),
            "sports_outcome_source_evidence_sha256": (
                sports_outcome_source_hash
            ),
            "sports_outcome_observation_count": int(
                sports_outcome_observation_count
            ),
            "reaction_blind_metadata_game_ids": tuple(
                sorted(reaction_blind_ids)
            ),
            "reaction_blind_metadata_game_ids_sha256": (
                reaction_blind_hash
            ),
            "stage_a_outcome_validation_eligible": False,
            "stage_b_market_reaction_validation_eligible": True,
        }
    )
    selection_authority = bind_frozen_development_authority(
        development,
        cohort_authority_sha256=authority,
    )
    return FrozenAuthorityMetadata(
        cohort_authority_sha256=authority,
        development=development.sort_values(
            ["nfl_week", "kickoff_utc", "game_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        holdout=holdout.sort_values(
            ["nfl_week", "kickoff_utc", "game_id"],
            kind="mergesort",
        ).reset_index(drop=True),
        overlap_game_ids=overlap,
        holdout_reviewed_overlap_game_ids=reviewed_overlap,
        holdout_reaction_overlap_game_ids=reaction_overlap,
        historical_reviewed_artifact_path=(
            historical_evidence.reviewed_artifact_path
        ),
        historical_reviewed_artifact_sha256=(
            historical_evidence.reviewed_artifact_sha256
        ),
        historical_reaction_artifact_path=(
            historical_evidence.reaction_artifact_path
        ),
        historical_reaction_artifact_sha256=(
            historical_evidence.reaction_artifact_sha256
        ),
        historical_reviewed_game_ids_sha256=reviewed_hash,
        historical_reaction_game_ids_sha256=reaction_hash,
        sports_outcome_exposed_game_ids_sha256=sports_outcome_hash,
        sports_outcome_source_evidence_sha256=(
            sports_outcome_source_hash
        ),
        sports_outcome_observation_count=int(
            sports_outcome_observation_count
        ),
        reaction_blind_metadata_game_ids_sha256=reaction_blind_hash,
        prior_exposure_audit=exposure,
        holdout_reaction_read_count=0,
        stage_a_outcome_validation_eligible=False,
        stage_b_market_reaction_validation_eligible=True,
        metadata_sha256=metadata_hash,
        selection_authority=selection_authority,
    )


def _as_text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise KalshiValidationError(
            f"{field} must be a tuple/list from Task4 provenance"
        )
    result = tuple(value)
    if not all(
        isinstance(item, str) and bool(item.strip()) for item in result
    ):
        raise KalshiValidationError(
            f"{field} must contain nonempty game IDs"
        )
    if len(result) != len(set(result)):
        raise KalshiValidationError(
            f"{field} must not contain duplicate game IDs"
        )
    return result


def _as_week_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise KalshiValidationError(
            f"{field} must be a nonempty Task4 week tuple/list"
        )
    if not all(
        isinstance(item, (Integral, np.integer))
        and not isinstance(item, (bool, np.bool_))
        for item in value
    ):
        raise KalshiValidationError(
            f"{field} must contain integer weeks"
        )
    result = tuple(int(item) for item in value)
    if len(result) != len(set(result)) or not all(
        1 <= item <= 12 for item in result
    ):
        raise KalshiValidationError(
            f"{field} must contain unique development weeks 1..12"
        )
    return result


def _validate_fold_local_source_rows(
    frame: pd.DataFrame,
    *,
    development_ids: frozenset[str],
) -> pd.DataFrame:
    work = frame.copy()
    for column in _HASH_PROVENANCE_COLUMNS:
        if not work[column].map(_is_sha256).all():
            raise KalshiValidationError(
                f"Task4 {column} must contain sha256 provenance"
            )
    normalized: dict[str, list[tuple[object, ...]]] = {
        column: [] for column in _SEQUENCE_PROVENANCE_COLUMNS
    }
    for row in work.itertuples(index=False):
        train_weeks = _as_week_tuple(
            getattr(row, "train_weeks"), field="train_weeks"
        )
        validation_weeks = _as_week_tuple(
            getattr(row, "validation_weeks"),
            field="validation_weeks",
        )
        if max(train_weeks) >= min(validation_weeks):
            raise KalshiValidationError(
                "Task4 transport must use chronological fold-local "
                "source training"
            )
        if int(getattr(row, "nfl_week")) not in validation_weeks:
            raise KalshiValidationError(
                "Task4 row nfl_week is outside its validation fold"
            )
        training_ids = _as_text_tuple(
            getattr(row, "training_game_ids"),
            field="training_game_ids",
        )
        validation_ids = _as_text_tuple(
            getattr(row, "validation_game_ids"),
            field="validation_game_ids",
        )
        if set(training_ids).intersection(validation_ids):
            raise KalshiValidationError(
                "Task4 training and validation game IDs overlap"
            )
        if (
            not set(training_ids).issubset(development_ids)
            or not set(validation_ids).issubset(development_ids)
            or str(getattr(row, "game_id")) not in validation_ids
        ):
            raise KalshiValidationError(
                "Task4 fold provenance must bind governed development games"
            )
        preprocessor_ids = _as_text_tuple(
            getattr(row, "preprocessor_fit_game_ids"),
            field="preprocessor_fit_game_ids",
        )
        if set(preprocessor_ids) != set(training_ids):
            raise KalshiValidationError(
                "preprocessor must fit exactly the fold-local source "
                "training games"
            )
        calibration_id_fields = (
            "calibrator_fit_game_ids_s_h",
            "calibrator_fit_game_ids_o_h_given_s",
            "calibrator_fit_game_ids_direction",
        )
        calibration_ids: dict[str, tuple[str, ...]] = {}
        for field in calibration_id_fields:
            ids = _as_text_tuple(getattr(row, field), field=field)
            if not set(ids).issubset(training_ids):
                raise KalshiValidationError(
                    "calibrators may fit only fold-local source "
                    "training games"
                )
            calibration_ids[field] = ids
        normalized["train_weeks"].append(train_weeks)
        normalized["validation_weeks"].append(validation_weeks)
        normalized["training_game_ids"].append(training_ids)
        normalized["validation_game_ids"].append(validation_ids)
        normalized["preprocessor_fit_game_ids"].append(
            preprocessor_ids
        )
        for field, ids in calibration_ids.items():
            normalized[field].append(ids)
    for column, values in normalized.items():
        work[column] = values
    return work


def _transport_source_provenance_sha256(frame: pd.DataFrame) -> str:
    required = {
        *_TRANSPORT_PAIR_COLUMNS,
        *_HASH_PROVENANCE_COLUMNS,
        *_SEQUENCE_PROVENANCE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KalshiValidationError(
            "transport source provenance columns are missing: "
            f"{missing}"
        )
    records: list[dict[str, object]] = []
    ordered = frame.sort_values(
        list(_TRANSPORT_PAIR_COLUMNS), kind="mergesort"
    )
    for row in ordered.to_dict("records"):
        record: dict[str, object] = {}
        for column in (
            *_TRANSPORT_PAIR_COLUMNS,
            *_HASH_PROVENANCE_COLUMNS,
            *_SEQUENCE_PROVENANCE_COLUMNS,
        ):
            value = row[column]
            record[column] = (
                tuple(value)
                if column in _SEQUENCE_PROVENANCE_COLUMNS
                else value
            )
        records.append(record)
    return _canonical_sha256(
        {
            "schema": "nfl_x16_stage_b_source_provenance_v1",
            "rows": records,
        }
    )


def _require_exact_frozen_stage_b_source(
    source: pd.DataFrame,
    *,
    selection: ModelSelectionResult,
    verified_source_run: X15ModelRun,
) -> None:
    identity = (
        "source_row_id",
        *_MODEL_COMPARISON_PAIR_COLUMNS,
        "actual_home_contract_id",
    )
    winner_rows = selection.paired_rows
    for model_id, feature_block_id, suffix in (
        (
            selection.spec.candidate_model_id,
            selection.spec.candidate_feature_block_id,
            "candidate",
        ),
        (
            selection.spec.baseline_model_id,
            selection.spec.baseline_feature_block_id,
            "b0",
        ),
    ):
        observed = source.loc[
            source["model_id"].eq(model_id)
            & source["feature_block_id"].eq(feature_block_id)
        ].copy()
        if observed.empty or observed.duplicated(
            list(identity), keep=False
        ).any():
            raise KalshiValidationError(
                "transport source does not bind exact frozen Stage-B "
                "winner provenance"
            )
        expected = winner_rows.copy()
        expected_columns = {
            f"train_weeks_{suffix}": "train_weeks",
            f"validation_weeks_{suffix}": "validation_weeks",
            f"training_game_ids_{suffix}": "training_game_ids",
            f"validation_game_ids_{suffix}": "validation_game_ids",
            f"preprocessor_fit_game_ids_{suffix}": (
                "preprocessor_fit_game_ids"
            ),
            f"s_h_truth_{suffix}": "s_h_truth",
            f"o_h_given_s_truth_{suffix}": "o_h_given_s_truth",
            f"direction_truth_{suffix}": "direction_truth",
            f"s_h_calibrated_probability_{suffix}": (
                "s_h_calibrated_probability"
            ),
            f"o_h_given_s_calibrated_probability_{suffix}": (
                "o_h_given_s_calibrated_probability"
            ),
            f"direction_calibrated_prob_down_{suffix}": (
                "direction_calibrated_prob_down"
            ),
            f"direction_calibrated_prob_no_move_{suffix}": (
                "direction_calibrated_prob_no_move"
            ),
            f"direction_calibrated_prob_up_{suffix}": (
                "direction_calibrated_prob_up"
            ),
        }
        expected = expected[
            [*identity, *expected_columns]
        ].rename(columns=expected_columns)
        pair = observed.merge(
            expected,
            on=list(identity),
            how="left",
            suffixes=("_observed", "_frozen"),
            validate="one_to_one",
            indicator=True,
        )
        if not pair["_merge"].eq("both").all():
            raise KalshiValidationError(
                "transport source does not bind exact frozen Stage-B "
                "winner provenance"
            )
        for column in expected_columns.values():
            left = pair[f"{column}_observed"]
            right = pair[f"{column}_frozen"]
            if column in _SEQUENCE_PROVENANCE_COLUMNS:
                matches = all(
                    tuple(left_value) == tuple(right_value)
                    for left_value, right_value in zip(
                        left, right, strict=True
                    )
                )
            elif column in {
                "s_h_truth",
                "o_h_given_s_truth",
                "direction_truth",
            }:
                matches = _nullable_truth_equal(left, right)
            else:
                numeric_left = pd.to_numeric(left, errors="coerce")
                numeric_right = pd.to_numeric(right, errors="coerce")
                matches = bool(
                    np.isclose(
                        numeric_left.to_numpy(dtype=float),
                        numeric_right.to_numpy(dtype=float),
                        atol=0.0,
                        rtol=0.0,
                        equal_nan=True,
                    ).all()
                )
            if not matches:
                raise KalshiValidationError(
                    "transport source does not bind exact frozen Stage-B "
                    "winner provenance"
                )

    anchor_columns = (
        "source_row_id",
        *_TRANSPORT_PAIR_COLUMNS,
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        "actual_home_contract_id",
        *_HASH_PROVENANCE_COLUMNS,
        *_SEQUENCE_PROVENANCE_COLUMNS,
        *_PROBABILITY_COLUMNS,
        "s_h_truth",
        "o_h_given_s_truth",
        "direction_truth",
    )
    anchor = verified_source_run.oof_predictions
    missing_anchor = sorted(set(anchor_columns).difference(anchor.columns))
    if (
        not isinstance(anchor, pd.DataFrame)
        or anchor.empty
        or missing_anchor
    ):
        raise KalshiValidationError(
            "independently verified exact45 source lacks frozen "
            f"winner provenance columns: {missing_anchor}"
        )
    anchor = anchor.loc[
        anchor["venue"].eq(SOURCE_VENUE)
        & anchor["training_venue"].eq(SOURCE_VENUE)
        & anchor["calibration_venue"].eq(SOURCE_VENUE)
        & anchor["transport_mode"].eq(VENUE_SPECIFIC_MODE),
        list(anchor_columns),
    ].copy()
    expected_designs = {
        (
            selection.spec.baseline_model_id,
            selection.spec.baseline_feature_block_id,
        ),
        (
            selection.spec.candidate_model_id,
            selection.spec.candidate_feature_block_id,
        ),
    }
    observed_designs = set(
        zip(
            anchor["model_id"].astype(str),
            anchor["feature_block_id"].astype(str),
            strict=True,
        )
    )
    identity = list(_TRANSPORT_PAIR_COLUMNS)
    if (
        observed_designs != expected_designs
        or anchor.duplicated(identity, keep=False).any()
    ):
        raise KalshiValidationError(
            "independently verified exact45 source has wrong winner "
            "or fold identity"
        )
    observed = source.loc[:, list(anchor_columns)].copy()
    pair = observed.merge(
        anchor,
        on=identity,
        how="left",
        suffixes=("_observed", "_verified"),
        validate="one_to_one",
        indicator=True,
    )
    if not pair["_merge"].eq("both").all():
        raise KalshiValidationError(
            "transport source does not match independently verified "
            "exact45 winner provenance"
        )
    compared_columns = [
        column for column in anchor_columns if column not in identity
    ]
    for column in compared_columns:
        left = pair[f"{column}_observed"]
        right = pair[f"{column}_verified"]
        if column in _SEQUENCE_PROVENANCE_COLUMNS:
            matches = all(
                tuple(left_value) == tuple(right_value)
                for left_value, right_value in zip(
                    left, right, strict=True
                )
            )
        elif column in {
            "s_h_truth",
            "o_h_given_s_truth",
            "direction_truth",
        }:
            matches = _nullable_truth_equal(left, right)
        elif column in {
            "source_row_id",
            *_PROBABILITY_COLUMNS,
        }:
            numeric_left = pd.to_numeric(left, errors="coerce")
            numeric_right = pd.to_numeric(right, errors="coerce")
            matches = bool(
                np.isclose(
                    numeric_left.to_numpy(dtype=float),
                    numeric_right.to_numpy(dtype=float),
                    atol=0.0,
                    rtol=0.0,
                    equal_nan=True,
                ).all()
            )
        else:
            matches = left.astype(str).equals(right.astype(str))
        if not matches:
            raise KalshiValidationError(
                "transport source does not match independently verified "
                "exact45 winner provenance"
            )


def _validate_task4_transport_run(
    model_run: X15ModelRun,
    *,
    authority_metadata: FrozenAuthorityMetadata,
    selection: ModelSelectionResult,
    winner_evidence_sha256: str,
    verified_source_run: X15ModelRun,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(model_run, X15ModelRun):
        raise KalshiValidationError(
            "venue transport validation requires an actual Task4 "
            "X15ModelRun"
        )
    _require_sha256(
        model_run.run_config_sha256, field="run_config_sha256"
    )
    if not isinstance(model_run.run_config, Mapping):
        raise KalshiValidationError(
            "transport validation requires the stamped Task4 "
            "diagnostic run_config"
        )
    if (
        _canonical_sha256(model_run.run_config)
        != model_run.run_config_sha256
    ):
        raise KalshiValidationError(
            "run_config_sha256 does not bind the canonical Task4 "
            "diagnostic run_config"
        )
    expected_config = {
        "schema_version": HISTORICAL_SCHEMA_VERSION,
        "survival_probability_contract": (
            SURVIVAL_PROBABILITY_CONTRACT
        ),
        "target_contract": HISTORICAL_TARGET_CONTRACT,
        "claim_boundary": HISTORICAL_CLAIM_BOUNDARY,
        "analysis_scope": HISTORICAL_ANALYSIS_SCOPE,
        "direction_threshold_probability": (
            DIRECTION_THRESHOLD_PROBABILITY
        ),
        "direction_threshold_semantics": (
            DIRECTION_THRESHOLD_SEMANTICS
        ),
        "venue_tick_support": VENUE_TICK_SUPPORT,
        "market_continuity_support": MARKET_CONTINUITY_SUPPORT,
        "claim_eligible": False,
    }
    for field, expected in expected_config.items():
        if model_run.run_config.get(field) != expected:
            raise KalshiValidationError(
                f"Task4 diagnostic run_config drifted on {field}"
            )
    source_run_config_sha256 = model_run.run_config.get(
        "stage_b_source_run_config_sha256"
    )
    source_provenance_sha256 = model_run.run_config.get(
        "stage_b_source_provenance_sha256"
    )
    declared_winner_evidence_sha256 = model_run.run_config.get(
        "stage_b_winner_model_selection_result_sha256"
    )
    source_config = dict(model_run.run_config)
    for field in (
        "stage_b_source_run_config_sha256",
        "stage_b_source_provenance_sha256",
        "stage_b_winner_model_selection_result_sha256",
    ):
        source_config.pop(field, None)
    source_config["transport_pairs"] = ()
    if (
        source_run_config_sha256
        != verified_source_run.run_config_sha256
        or not _is_sha256(source_provenance_sha256)
        or declared_winner_evidence_sha256 != winner_evidence_sha256
        or _canonical_sha256(source_config)
        != verified_source_run.run_config_sha256
    ):
        raise KalshiValidationError(
            "transport run_config does not bind exact frozen Stage-B "
            "winner provenance"
        )
    transport_pairs = model_run.run_config.get("transport_pairs")
    if (
        not isinstance(transport_pairs, (tuple, list))
        or (SOURCE_VENUE, TARGET_VENUE)
        not in {tuple(pair) for pair in transport_pairs}
    ):
        raise KalshiValidationError(
            "Task4 run_config must predeclare Polymarket-to-Kalshi "
            "development transport"
        )
    predictions = model_run.oof_predictions
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise KalshiValidationError(
            "X15ModelRun.oof_predictions must be nonempty"
        )
    missing = sorted(_TRANSPORT_REQUIRED.difference(predictions.columns))
    if missing:
        raise KalshiValidationError(
            f"Task4 transport OOF missing required columns: {missing}"
        )
    weeks = pd.to_numeric(predictions["nfl_week"], errors="coerce")
    if (
        weeks.isna().any()
        or not np.equal(
            weeks.to_numpy(dtype=float),
            weeks.astype(int).to_numpy(dtype=float),
        ).all()
        or not weeks.between(1, 12).all()
    ):
        raise KalshiValidationError(
            "development transport nfl_week must be integral in 1..12"
        )
    predictions = predictions.copy()
    predictions["nfl_week"] = weeks.astype(int)
    if not predictions["cohort_authority_sha256"].eq(
        authority_metadata.cohort_authority_sha256
    ).all():
        raise KalshiValidationError(
            "Task4 transport OOF cohort_authority_sha256 does not "
            "match frozen authority metadata"
        )
    threshold = pd.to_numeric(
        predictions["direction_threshold_probability"],
        errors="coerce",
    )
    if (
        threshold.isna().any()
        or not np.isfinite(threshold.to_numpy(dtype=float)).all()
        or not threshold.eq(DIRECTION_THRESHOLD_PROBABILITY).all()
    ):
        raise KalshiValidationError(
            "direction_threshold_probability is frozen at 0.01"
        )
    if not predictions["direction_threshold_semantics"].eq(
        DIRECTION_THRESHOLD_SEMANTICS
    ).all():
        raise KalshiValidationError(
            "transport direction semantics must remain research "
            "materiality, not a venue tick"
        )
    if not predictions["venue_tick_support"].eq(
        VENUE_TICK_SUPPORT
    ).all():
        raise KalshiValidationError(
            "transport must preserve venue_tick_support=UNSUPPORTED"
        )
    if not predictions["market_continuity_support"].eq(
        MARKET_CONTINUITY_SUPPORT
    ).all():
        raise KalshiValidationError(
            "transport must preserve market_continuity_support=UNKNOWN"
        )
    if not predictions["schema_version"].eq(
        HISTORICAL_SCHEMA_VERSION
    ).all():
        raise KalshiValidationError(
            "transport requires HistoricalTradesOnlyProbabilityPanelV2"
        )
    if not predictions["analysis_scope"].eq(
        HISTORICAL_ANALYSIS_SCOPE
    ).all():
        raise KalshiValidationError(
            "transport must preserve the exact historical source-time "
            "diagnostic analysis_scope"
        )
    if not predictions["claim_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and not bool(value)
    ).all():
        raise KalshiValidationError(
            "transport must preserve claim_eligible=False"
        )
    if not predictions["target_contract"].eq(
        HISTORICAL_TARGET_CONTRACT
    ).all():
        raise KalshiValidationError(
            "transport must preserve the exact historical trades-only "
            "target_contract"
        )
    if not predictions["claim_boundary"].eq(
        HISTORICAL_CLAIM_BOUNDARY
    ).all():
        raise KalshiValidationError(
            "transport must preserve the exact historical source-time "
            "probability diagnostic claim_boundary"
        )

    source = predictions.loc[
        predictions["venue"].eq(SOURCE_VENUE)
        & predictions["training_venue"].eq(SOURCE_VENUE)
        & predictions["calibration_venue"].eq(SOURCE_VENUE)
        & predictions["transport_mode"].eq(VENUE_SPECIFIC_MODE)
    ].copy()
    target = predictions.loc[
        predictions["venue"].eq(TARGET_VENUE)
        & predictions["training_venue"].eq(SOURCE_VENUE)
        & predictions["calibration_venue"].eq(SOURCE_VENUE)
        & predictions["transport_mode"].eq(TRANSPORT_MODE)
    ].copy()
    native_kalshi = predictions.loc[
        predictions["venue"].eq(TARGET_VENUE)
        & predictions["training_venue"].eq(TARGET_VENUE)
        & predictions["calibration_venue"].eq(TARGET_VENUE)
        & predictions["transport_mode"].eq(VENUE_SPECIFIC_MODE)
    ].copy()
    if source.empty or target.empty or native_kalshi.empty:
        raise KalshiValidationError(
            "Task4 run must contain genuine Polymarket source OOF and "
            "Kalshi development transport/native OOF"
        )
    development_ids = frozenset(
        authority_metadata.development["game_id"].astype(str)
    )
    source = _validate_fold_local_source_rows(
        source, development_ids=development_ids
    )
    target = _validate_fold_local_source_rows(
        target, development_ids=development_ids
    )
    native_kalshi = _validate_fold_local_source_rows(
        native_kalshi, development_ids=development_ids
    )
    _require_exact_frozen_stage_b_source(
        source,
        selection=selection,
        verified_source_run=verified_source_run,
    )
    if (
        _transport_source_provenance_sha256(source)
        != source_provenance_sha256
    ):
        raise KalshiValidationError(
            "transport source does not bind exact frozen Stage-B "
            "winner provenance"
        )
    return source, target, native_kalshi


def _nullable_truth_equal(
    left: pd.Series, right: pd.Series
) -> bool:
    for left_value, right_value in zip(left, right, strict=True):
        left_missing = bool(pd.isna(left_value))
        right_missing = bool(pd.isna(right_value))
        if left_missing or right_missing:
            if left_missing and right_missing:
                continue
            return False
        if left_value != right_value:
            return False
    return True


def _valid_binary_truth(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if (
        isinstance(value, (Integral, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) in (0, 1)
    ):
        return int(value)
    raise KalshiValidationError(
        "Kalshi target binary truth must be 0/1 or missing"
    )


def _paired_binary_scores(
    pairs: pd.DataFrame,
    *,
    head: str,
    truth_column: str,
    probability_column: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for _, row in pairs.iterrows():
        truth = _valid_binary_truth(row[f"{truth_column}_transport"])
        if truth is None:
            continue
        transport_probability = float(
            row[f"{probability_column}_transport"]
        )
        native_probability = float(
            row[f"{probability_column}_native"]
        )
        if not (
            np.isfinite(transport_probability)
            and np.isfinite(native_probability)
        ):
            continue
        if not (
            0 < transport_probability < 1
            and 0 < native_probability < 1
        ):
            raise KalshiValidationError(
                f"{head} calibrated probabilities must be in (0, 1)"
            )
        rows.append(
            {
                "game_id": str(row["game_id"]),
                "head": head,
                "truth": truth,
                "transport_probability": transport_probability,
                "native_probability": native_probability,
                "transport_log_loss": -(
                    truth * np.log(transport_probability)
                    + (1 - truth)
                    * np.log(1 - transport_probability)
                ),
                "native_log_loss": -(
                    truth * np.log(native_probability)
                    + (1 - truth) * np.log(1 - native_probability)
                ),
                "transport_brier": (
                    transport_probability - truth
                )
                ** 2,
                "native_brier": (native_probability - truth) ** 2,
            }
        )
    if rows:
        score_frame = pd.DataFrame(rows)
        calibration_rows.append(
            {
                "head": head,
                "class_label": "BINARY_POSITIVE",
                "evaluated_row_count": len(score_frame),
                "evaluated_game_count": int(
                    score_frame["game_id"].nunique()
                ),
                "observed_rate": float(score_frame["truth"].mean()),
                "transport_mean_probability": float(
                    score_frame["transport_probability"].mean()
                ),
                "native_kalshi_mean_probability": float(
                    score_frame["native_probability"].mean()
                ),
                "transport_calibration_gap": float(
                    abs(
                        score_frame["transport_probability"].mean()
                        - score_frame["truth"].mean()
                    )
                ),
                "native_kalshi_calibration_gap": float(
                    abs(
                        score_frame["native_probability"].mean()
                        - score_frame["truth"].mean()
                    )
                ),
            }
        )
    return rows, calibration_rows


def _paired_direction_scores(
    pairs: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    classes = ("DOWN", "NO_MOVE", "UP")
    probability_columns = (
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    )
    rows: list[dict[str, object]] = []
    for _, row in pairs.iterrows():
        truth = row["direction_truth_transport"]
        if pd.isna(truth):
            continue
        if str(truth) not in classes:
            raise KalshiValidationError(
                "Kalshi target direction truth is invalid"
            )
        transport = np.asarray(
            [
                row[f"{column}_transport"]
                for column in probability_columns
            ],
            dtype=float,
        )
        native = np.asarray(
            [
                row[f"{column}_native"]
                for column in probability_columns
            ],
            dtype=float,
        )
        if not np.isfinite(transport).all() or not np.isfinite(native).all():
            continue
        if (
            (transport <= 0).any()
            or (native <= 0).any()
            or not np.isclose(transport.sum(), 1.0, atol=1e-6)
            or not np.isclose(native.sum(), 1.0, atol=1e-6)
        ):
            raise KalshiValidationError(
                "direction probabilities must be positive and sum to one"
            )
        truth_index = classes.index(str(truth))
        one_hot = np.zeros(3, dtype=float)
        one_hot[truth_index] = 1.0
        record: dict[str, object] = {
            "game_id": str(row["game_id"]),
            "head": "DIRECTION",
            "truth": str(truth),
            "transport_log_loss": -np.log(transport[truth_index]),
            "native_log_loss": -np.log(native[truth_index]),
            "transport_brier": float(
                np.square(transport - one_hot).sum()
            ),
            "native_brier": float(
                np.square(native - one_hot).sum()
            ),
        }
        for index, class_label in enumerate(classes):
            record[f"transport_probability_{class_label}"] = transport[
                index
            ]
            record[f"native_probability_{class_label}"] = native[index]
            record[f"truth_{class_label}"] = int(
                truth_index == index
            )
        rows.append(record)
    calibration_rows: list[dict[str, object]] = []
    if rows:
        frame = pd.DataFrame(rows)
        for class_label in classes:
            observed = frame[f"truth_{class_label}"].mean()
            transport_mean = frame[
                f"transport_probability_{class_label}"
            ].mean()
            native_mean = frame[
                f"native_probability_{class_label}"
            ].mean()
            calibration_rows.append(
                {
                    "head": "DIRECTION",
                    "class_label": class_label,
                    "evaluated_row_count": len(frame),
                    "evaluated_game_count": int(
                        frame["game_id"].nunique()
                    ),
                    "observed_rate": float(observed),
                    "transport_mean_probability": float(
                        transport_mean
                    ),
                    "native_kalshi_mean_probability": float(
                        native_mean
                    ),
                    "transport_calibration_gap": float(
                        abs(transport_mean - observed)
                    ),
                    "native_kalshi_calibration_gap": float(
                        abs(native_mean - observed)
                    ),
                }
            )
    return rows, calibration_rows


def _equal_game_score_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    head = str(frame["head"].iloc[0])
    records: list[dict[str, object]] = []
    for metric, transport_column, native_column in (
        ("LOG_LOSS", "transport_log_loss", "native_log_loss"),
        ("BRIER", "transport_brier", "native_brier"),
    ):
        games = (
            frame.groupby("game_id", sort=True, as_index=False)
            .agg(
                transport_score=(transport_column, "mean"),
                native_kalshi_score=(native_column, "mean"),
            )
            .reset_index(drop=True)
        )
        transport_score = float(games["transport_score"].mean())
        native_score = float(games["native_kalshi_score"].mean())
        records.append(
            {
                "head": head,
                "metric": metric,
                "transport_score": transport_score,
                "native_kalshi_score": native_score,
                "score_degradation": (
                    transport_score - native_score
                ),
                "evaluated_game_count": len(games),
                "evaluated_row_count": len(frame),
                "aggregation": "EQUAL_GAME",
            }
        )
    return records


def _score_transport_against_native(
    target_native_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_records: list[dict[str, object]] = []
    calibration_records: list[dict[str, object]] = []
    for (model_id, feature_block_id), model_rows in (
        target_native_pairs.groupby(
            ["model_id", "feature_block_id"],
            sort=True,
            observed=True,
        )
    ):
        model_scores: list[dict[str, object]] = []
        model_calibration: list[dict[str, object]] = []
        for head, truth_column, probability_column in (
            (
                "S_H",
                "s_h_truth",
                "s_h_calibrated_probability",
            ),
            (
                "O_H_GIVEN_S",
                "o_h_given_s_truth",
                "o_h_given_s_calibrated_probability",
            ),
        ):
            rows, calibration = _paired_binary_scores(
                model_rows,
                head=head,
                truth_column=truth_column,
                probability_column=probability_column,
            )
            model_scores.extend(_equal_game_score_summary(rows))
            model_calibration.extend(calibration)
        direction_rows, direction_calibration = (
            _paired_direction_scores(model_rows)
        )
        model_scores.extend(
            _equal_game_score_summary(direction_rows)
        )
        model_calibration.extend(direction_calibration)
        for record in (*model_scores, *model_calibration):
            record["model_id"] = str(model_id)
            record["feature_block_id"] = str(feature_block_id)
        score_records.extend(model_scores)
        calibration_records.extend(model_calibration)
    return (
        pd.DataFrame(score_records),
        pd.DataFrame(calibration_records),
    )


def _merge_exact_candidate_b0_rows(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    layer: str,
) -> pd.DataFrame:
    for frame, label in ((candidate, "candidate"), (baseline, "B0")):
        if frame.empty:
            raise KalshiValidationError(
                f"{layer} requires frozen {label} Kalshi truth rows"
            )
        if frame.duplicated(
            list(_MODEL_COMPARISON_PAIR_COLUMNS), keep=False
        ).any():
            raise KalshiValidationError(
                f"{layer} {label} comparison identities are not unique"
            )
    pairs = candidate.merge(
        baseline,
        on=list(_MODEL_COMPARISON_PAIR_COLUMNS),
        how="outer",
        suffixes=("_candidate", "_b0"),
        validate="one_to_one",
        indicator=True,
    )
    if not pairs["_merge"].eq("both").all():
        raise KalshiValidationError(
            f"{layer} candidate/B0 must pair on exact Kalshi truth rows"
        )
    pairs = pairs.drop(columns="_merge")
    for truth_column in (
        "s_h_truth",
        "o_h_given_s_truth",
        "direction_truth",
    ):
        if not _nullable_truth_equal(
            pairs[f"{truth_column}_candidate"],
            pairs[f"{truth_column}_b0"],
        ):
            raise KalshiValidationError(
                f"{layer} candidate/B0 disagree on {truth_column}"
            )
    if not pairs["actual_home_contract_id_candidate"].eq(
        pairs["actual_home_contract_id_b0"]
    ).all():
        raise KalshiValidationError(
            f"{layer} candidate/B0 disagree on Kalshi contract ID"
        )
    return pairs


def _candidate_b0_integrated_log_loss(
    pairs: pd.DataFrame,
    *,
    layer: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    direction_columns = (
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    )
    direction_classes = ("DOWN", "NO_MOVE", "UP")
    for _, row in pairs.iterrows():
        candidate_losses: list[float] = []
        baseline_losses: list[float] = []
        for truth_column, probability_column in (
            ("s_h_truth", "s_h_calibrated_probability"),
            (
                "o_h_given_s_truth",
                "o_h_given_s_calibrated_probability",
            ),
        ):
            truth = _valid_binary_truth(
                row[f"{truth_column}_candidate"]
            )
            if truth is None:
                continue
            candidate_probability = float(
                row[f"{probability_column}_candidate"]
            )
            baseline_probability = float(
                row[f"{probability_column}_b0"]
            )
            if not (
                np.isfinite(candidate_probability)
                and np.isfinite(baseline_probability)
                and 0 < candidate_probability < 1
                and 0 < baseline_probability < 1
            ):
                raise KalshiValidationError(
                    f"{layer} binary probabilities must be finite in (0, 1)"
                )
            candidate_losses.append(
                -(
                    truth * np.log(candidate_probability)
                    + (1 - truth) * np.log(1 - candidate_probability)
                )
            )
            baseline_losses.append(
                -(
                    truth * np.log(baseline_probability)
                    + (1 - truth) * np.log(1 - baseline_probability)
                )
            )
        direction_truth = row["direction_truth_candidate"]
        if not pd.isna(direction_truth):
            if str(direction_truth) not in direction_classes:
                raise KalshiValidationError(
                    f"{layer} direction truth is invalid"
                )
            candidate_direction = np.asarray(
                [
                    row[f"{column}_candidate"]
                    for column in direction_columns
                ],
                dtype=float,
            )
            baseline_direction = np.asarray(
                [row[f"{column}_b0"] for column in direction_columns],
                dtype=float,
            )
            if (
                not np.isfinite(candidate_direction).all()
                or not np.isfinite(baseline_direction).all()
                or (candidate_direction <= 0).any()
                or (baseline_direction <= 0).any()
                or not np.isclose(
                    candidate_direction.sum(), 1.0, atol=1e-6
                )
                or not np.isclose(
                    baseline_direction.sum(), 1.0, atol=1e-6
                )
            ):
                raise KalshiValidationError(
                    f"{layer} direction probabilities are invalid"
                )
            truth_index = direction_classes.index(str(direction_truth))
            candidate_losses.append(
                float(-np.log(candidate_direction[truth_index]))
            )
            baseline_losses.append(
                float(-np.log(baseline_direction[truth_index]))
            )
        if not candidate_losses:
            raise KalshiValidationError(
                f"{layer} exact row has no evaluable Kalshi truth head"
            )
        candidate_loss = float(np.sum(candidate_losses))
        baseline_loss = float(np.sum(baseline_losses))
        records.append(
            {
                **{
                    column: row[column]
                    for column in _MODEL_COMPARISON_PAIR_COLUMNS
                },
                "comparison_layer": layer,
                "candidate_model_id": str(row["model_id_candidate"]),
                "candidate_feature_block_id": str(
                    row["feature_block_id_candidate"]
                ),
                "baseline_model_id": str(row["model_id_b0"]),
                "baseline_feature_block_id": str(
                    row["feature_block_id_b0"]
                ),
                "actual_home_contract_id": str(
                    row["actual_home_contract_id_candidate"]
                ),
                "available_head_count": len(candidate_losses),
                "s_h_truth": row["s_h_truth_candidate"],
                "o_h_given_s_truth": row[
                    "o_h_given_s_truth_candidate"
                ],
                "direction_truth": row["direction_truth_candidate"],
                "candidate_integrated_log_loss": candidate_loss,
                "b0_integrated_log_loss": baseline_loss,
                "loss_improvement": baseline_loss - candidate_loss,
            }
        )
    return pd.DataFrame(records)


def _candidate_b0_head_scores(
    pairs: pd.DataFrame,
    *,
    layer: str,
) -> pd.DataFrame:
    """Score transported candidate and B0 on each exact Kalshi truth head."""

    records: list[dict[str, object]] = []
    direction_columns = (
        "direction_calibrated_prob_down",
        "direction_calibrated_prob_no_move",
        "direction_calibrated_prob_up",
    )
    direction_classes = ("DOWN", "NO_MOVE", "UP")
    for _, row in pairs.iterrows():
        identity = {
            column: row[column]
            for column in _MODEL_COMPARISON_PAIR_COLUMNS
        }
        for head, truth_column, probability_column in (
            ("S_H", "s_h_truth", "s_h_calibrated_probability"),
            (
                "O_H_GIVEN_S",
                "o_h_given_s_truth",
                "o_h_given_s_calibrated_probability",
            ),
        ):
            truth = _valid_binary_truth(
                row[f"{truth_column}_candidate"]
            )
            if truth is None:
                continue
            candidate_probability = float(
                row[f"{probability_column}_candidate"]
            )
            baseline_probability = float(
                row[f"{probability_column}_b0"]
            )
            if not (
                np.isfinite(candidate_probability)
                and np.isfinite(baseline_probability)
                and 0 < candidate_probability < 1
                and 0 < baseline_probability < 1
            ):
                raise KalshiValidationError(
                    f"{layer} binary probabilities must be finite in (0, 1)"
                )
            candidate_loss = float(
                -(
                    truth * np.log(candidate_probability)
                    + (1 - truth) * np.log(1 - candidate_probability)
                )
            )
            baseline_loss = float(
                -(
                    truth * np.log(baseline_probability)
                    + (1 - truth) * np.log(1 - baseline_probability)
                )
            )
            records.append(
                {
                    **identity,
                    "comparison_layer": layer,
                    "head": head,
                    "metric": "LOG_LOSS",
                    "candidate_score": candidate_loss,
                    "b0_score": baseline_loss,
                    "score_improvement": baseline_loss - candidate_loss,
                }
            )

        direction_truth = row["direction_truth_candidate"]
        if pd.isna(direction_truth):
            continue
        if str(direction_truth) not in direction_classes:
            raise KalshiValidationError(
                f"{layer} direction truth is invalid"
            )
        candidate_direction = np.asarray(
            [
                row[f"{column}_candidate"]
                for column in direction_columns
            ],
            dtype=float,
        )
        baseline_direction = np.asarray(
            [row[f"{column}_b0"] for column in direction_columns],
            dtype=float,
        )
        if (
            not np.isfinite(candidate_direction).all()
            or not np.isfinite(baseline_direction).all()
            or (candidate_direction <= 0).any()
            or (baseline_direction <= 0).any()
            or not np.isclose(candidate_direction.sum(), 1.0, atol=1e-6)
            or not np.isclose(baseline_direction.sum(), 1.0, atol=1e-6)
        ):
            raise KalshiValidationError(
                f"{layer} direction probabilities are invalid"
            )
        truth_index = direction_classes.index(str(direction_truth))
        one_hot = np.zeros(3, dtype=float)
        one_hot[truth_index] = 1.0
        for metric, candidate_score, baseline_score in (
            (
                "LOG_LOSS",
                float(-np.log(candidate_direction[truth_index])),
                float(-np.log(baseline_direction[truth_index])),
            ),
            (
                "BRIER",
                float(np.square(candidate_direction - one_hot).sum()),
                float(np.square(baseline_direction - one_hot).sum()),
            ),
        ):
            records.append(
                {
                    **identity,
                    "comparison_layer": layer,
                    "head": "DIRECTION",
                    "metric": metric,
                    "candidate_score": candidate_score,
                    "b0_score": baseline_score,
                    "score_improvement": baseline_score - candidate_score,
                }
            )
    return pd.DataFrame(records)


def _bootstrap_game_mean(
    values: np.ndarray,
) -> tuple[float, float, float]:
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selections = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    draws = values[selections].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high), float(np.mean(draws > 0.0))


def _head_gate_summary(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize exact transported-candidate versus transported-B0 heads."""

    records: list[dict[str, object]] = []
    for (head, metric), rows in diagnostics.groupby(
        ["head", "metric"], sort=True, observed=True
    ):
        games = (
            rows.groupby("game_id", sort=True, as_index=False)
            .agg(score_improvement=("score_improvement", "mean"))
            .reset_index(drop=True)
        )
        values = games["score_improvement"].to_numpy(dtype=float)
        low, high, probability = _bootstrap_game_mean(values)
        anchor = rows.loc[
            rows["landmark_seconds"].eq(ANCHOR_LANDMARK_SECONDS)
            & rows["endpoint_seconds"].eq(ANCHOR_ENDPOINT_SECONDS)
        ]
        anchor_games = (
            anchor.groupby("game_id", sort=True)["score_improvement"]
            .mean()
            .to_numpy(dtype=float)
        )
        overall_mean = (
            float(values.mean()) if len(values) else float("nan")
        )
        anchor_mean = (
            float(anchor_games.mean())
            if len(anchor_games)
            else float("nan")
        )
        if str(head) in {"S_H", "O_H_GIVEN_S"}:
            gate = bool(
                str(metric) == "LOG_LOSS"
                and len(values) >= MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD
                and len(anchor_games)
                >= MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD
                and np.isfinite(overall_mean)
                and np.isfinite(anchor_mean)
                and overall_mean >= 0.0
                and anchor_mean >= 0.0
            )
            gate_semantics = (
                "OVERALL_AND_L3_H30_MEAN_LOG_LOSS_IMPROVEMENT_NONNEGATIVE"
            )
        else:
            gate = bool(
                str(head) == "DIRECTION"
                and str(metric) in {"LOG_LOSS", "BRIER"}
                and len(values) >= MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD
                and len(anchor_games)
                >= MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD
                and np.isfinite(overall_mean)
                and np.isfinite(anchor_mean)
                and np.isfinite(low)
                and float(low) > 0.0
                and anchor_mean >= 0.0
            )
            gate_semantics = (
                "OVERALL_GAME_CLUSTER_LOWER95_POSITIVE_AND_L3_H30_"
                "MEAN_NONNEGATIVE"
            )
        records.append(
            {
                "head": str(head),
                "metric": str(metric),
                "evaluated_row_count": len(rows),
                "evaluated_game_count": len(games),
                "overall_mean_improvement": overall_mean,
                "overall_ci_low": low,
                "overall_ci_high": high,
                "bootstrap_probability_improved": probability,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "anchor_landmark_seconds": ANCHOR_LANDMARK_SECONDS,
                "anchor_endpoint_seconds": ANCHOR_ENDPOINT_SECONDS,
                "anchor_game_count": len(anchor_games),
                "anchor_mean_improvement": anchor_mean,
                "gate_semantics": gate_semantics,
                "gate_passed": gate,
            }
        )
    return pd.DataFrame(records)


def _native_noninferiority_summary(
    native_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap transport-minus-native degradation by game and score head."""

    score_rows: list[dict[str, object]] = []
    for head, truth_column, probability_column in (
        ("S_H", "s_h_truth", "s_h_calibrated_probability"),
        (
            "O_H_GIVEN_S",
            "o_h_given_s_truth",
            "o_h_given_s_calibrated_probability",
        ),
    ):
        paired_rows, _ = _paired_binary_scores(
            native_pairs,
            head=head,
            truth_column=truth_column,
            probability_column=probability_column,
        )
        for row in paired_rows:
            score_rows.extend(
                (
                    {
                        "game_id": row["game_id"],
                        "head": head,
                        "metric": "LOG_LOSS",
                        "score_degradation": (
                            row["transport_log_loss"]
                            - row["native_log_loss"]
                        ),
                    },
                    {
                        "game_id": row["game_id"],
                        "head": head,
                        "metric": "BRIER",
                        "score_degradation": (
                            row["transport_brier"]
                            - row["native_brier"]
                        ),
                    },
                )
            )
    direction_rows, _ = _paired_direction_scores(native_pairs)
    for row in direction_rows:
        score_rows.extend(
            (
                {
                    "game_id": row["game_id"],
                    "head": "DIRECTION",
                    "metric": "LOG_LOSS",
                    "score_degradation": (
                        row["transport_log_loss"]
                        - row["native_log_loss"]
                    ),
                },
                {
                    "game_id": row["game_id"],
                    "head": "DIRECTION",
                    "metric": "BRIER",
                    "score_degradation": (
                        row["transport_brier"]
                        - row["native_brier"]
                    ),
                },
            )
        )
    frame = pd.DataFrame(score_rows)
    records: list[dict[str, object]] = []
    for (head, metric), rows in frame.groupby(
        ["head", "metric"], sort=True, observed=True
    ):
        games = (
            rows.groupby("game_id", sort=True, as_index=False)
            .agg(score_degradation=("score_degradation", "mean"))
            .reset_index(drop=True)
        )
        values = games["score_degradation"].to_numpy(dtype=float)
        if len(values) < 2:
            low = high = float("nan")
        else:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            selections = rng.integers(
                0,
                len(values),
                size=(BOOTSTRAP_SAMPLES, len(values)),
            )
            draws = values[selections].mean(axis=1)
            low, high = np.quantile(draws, (0.025, 0.975))
        margin = (
            MAX_LOG_LOSS_DEGRADATION
            if str(metric) == "LOG_LOSS"
            else MAX_BRIER_DEGRADATION
        )
        gate = bool(
            len(games) >= MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD
            and np.isfinite(high)
            and float(high) <= margin
        )
        records.append(
            {
                "head": str(head),
                "metric": str(metric),
                "evaluated_row_count": len(rows),
                "evaluated_game_count": len(games),
                "mean_score_degradation": (
                    float(values.mean()) if len(values) else float("nan")
                ),
                "degradation_ci_low": float(low),
                "degradation_ci_high": float(high),
                "noninferiority_margin": margin,
                "upper95_within_margin": gate,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "aggregation": "EQUAL_GAME_GAME_CLUSTER_BOOTSTRAP",
            }
        )
    return pd.DataFrame(records)


def _game_cluster_improvement_summary(
    diagnostics: pd.DataFrame,
) -> dict[str, object]:
    games = (
        diagnostics.groupby("game_id", sort=True, as_index=False)
        .agg(loss_improvement=("loss_improvement", "mean"))
        .reset_index(drop=True)
    )
    values = games["loss_improvement"].to_numpy(dtype=float)
    if len(values) < 2:
        low = high = probability = float("nan")
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        selections = rng.integers(
            0,
            len(values),
            size=(BOOTSTRAP_SAMPLES, len(values)),
        )
        draws = values[selections].mean(axis=1)
        low, high = np.quantile(draws, (0.025, 0.975))
        probability = float(np.mean(draws > 0.0))
    anchor = diagnostics.loc[
        diagnostics["landmark_seconds"].eq(ANCHOR_LANDMARK_SECONDS)
        & diagnostics["endpoint_seconds"].eq(ANCHOR_ENDPOINT_SECONDS)
    ]
    anchor_games = (
        anchor.groupby("game_id", sort=True)["loss_improvement"]
        .mean()
        .to_numpy(dtype=float)
    )
    mean_improvement = float(values.mean()) if len(values) else float("nan")
    anchor_mean = (
        float(anchor_games.mean()) if len(anchor_games) else float("nan")
    )
    gate = bool(
        len(values) >= MIN_TRANSPORT_PAIRED_GAMES
        and len(anchor_games) >= MIN_TRANSPORT_PAIRED_GAMES
        and np.isfinite(mean_improvement)
        and np.isfinite(low)
        and np.isfinite(anchor_mean)
        and mean_improvement >= 0.0
        and float(low) > 0.0
        and anchor_mean >= 0.0
    )
    return {
        "paired_row_count": len(diagnostics),
        "paired_game_count": len(games),
        "mean_improvement": mean_improvement,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_probability_improved": probability,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "anchor_game_count": len(anchor_games),
        "anchor_mean_improvement": anchor_mean,
        "improvement_gate_passed": gate,
        "aggregation": "EQUAL_GAME_GAME_CLUSTER_BOOTSTRAP",
    }


@dataclass(frozen=True, slots=True)
class DevelopmentVenueValidation:
    """Exact development pair diagnostics from a genuine Task4 run."""

    source_venue: str
    target_venue: str
    cohort: str
    cohort_authority_sha256: str
    run_config_sha256: str
    schema_version: str
    analysis_scope: str
    target_contract: str
    claim_boundary: str
    candidate_model_id: str
    candidate_feature_block_id: str
    diagnostic_status: str
    execution_claim_eligible: bool
    tick_claim_eligible: bool
    continuity_claim_eligible: bool
    target_recalibration_applied: bool
    paired_identity_count: int
    paired_game_count: int
    unmatched_source_rows: int
    unmatched_target_rows: int
    native_comparator_paired_rows: int
    native_comparator_unmatched_transport_rows: int
    native_comparator_unmatched_native_rows: int
    coverage_gate_passed: bool
    score_gate_passed: bool
    catastrophic_degradation: bool
    transport_b0_paired_rows: int
    transport_b0_paired_games: int
    transport_b0_mean_improvement: float
    transport_b0_ci_low: float
    transport_b0_ci_high: float
    transport_b0_bootstrap_probability_improved: float
    transport_b0_bootstrap_samples: int
    transport_b0_bootstrap_seed: int
    transport_b0_anchor_game_count: int
    transport_b0_anchor_mean_improvement: float
    transport_b0_improvement_gate_passed: bool
    binary_nuisance_gate_passed: bool
    direction_gate_passed: bool
    native_noninferiority_gate_passed: bool
    transport_gate_passed: bool
    coverage_audit: pd.DataFrame
    attrition: pd.DataFrame
    exact_pair_diagnostics: pd.DataFrame
    native_comparison_diagnostics: pd.DataFrame
    transport_b0_pair_diagnostics: pd.DataFrame
    native_b0_pair_diagnostics: pd.DataFrame
    native_b0_score_summary: pd.DataFrame
    transport_b0_head_diagnostics: pd.DataFrame
    transport_b0_head_gate_summary: pd.DataFrame
    native_noninferiority_summary: pd.DataFrame
    score_summary: pd.DataFrame
    calibration_summary: pd.DataFrame


def validate_development_venue_transport(
    model_run: X15ModelRun,
    *,
    authority_metadata: FrozenAuthorityMetadata,
    stage_b_selection: StageBModelSelectionResult,
    stage_b_batch_manifest_path: str | Path,
    stage_b_artifact_root: str | Path,
) -> DevelopmentVenueValidation:
    """Validate the frozen Stage-B Poly winner on Kalshi development OOF."""

    if not isinstance(authority_metadata, FrozenAuthorityMetadata):
        raise KalshiValidationError(
            "authority_metadata must be FrozenAuthorityMetadata"
        )
    selection = _require_frozen_stage_b_v3_winner(
        stage_b_selection,
        authority=authority_metadata.selection_authority,
    )
    if (
        selection.cohort_authority_sha256
        != authority_metadata.cohort_authority_sha256
    ):
        raise KalshiValidationError(
            "Stage-B winner authority differs from Kalshi development "
            "authority"
        )
    spec = selection.spec
    try:
        verified_source_run = load_x15_selection_projection_v3(
            batch_manifest_path=stage_b_batch_manifest_path,
            artifact_root=stage_b_artifact_root,
            candidate_model_id=spec.candidate_model_id,
            candidate_feature_block_id=(
                spec.candidate_feature_block_id
            ),
        )
    except (OSError, ValueError, X15SelectionBatchV3Error) as exc:
        raise KalshiValidationError(
            "Stage-B winner source failed immutable exact45 verification"
        ) from exc
    if (
        verified_source_run.run_config_sha256
        != selection.run_config_sha256
    ):
        raise KalshiValidationError(
            "Stage-B winner does not match independently verified "
            "exact45 source evidence"
        )
    _require_exact_frozen_stage_b_source(
        verified_source_run.oof_predictions,
        selection=selection,
        verified_source_run=verified_source_run,
    )
    winner_audit = stage_b_selection.candidate_audit.loc[
        stage_b_selection.candidate_audit["final_winner"].eq(True)
    ]
    if len(winner_audit) != 1:
        raise KalshiValidationError(
            "Stage-B V3 winner audit is not unique"
        )
    winner_evidence_sha256 = _require_sha256(
        winner_audit.iloc[0]["model_selection_result_sha256"],
        field="Stage-B winner model_selection_result_sha256",
    )
    all_source, all_target, all_native_kalshi = (
        _validate_task4_transport_run(
            model_run,
            authority_metadata=authority_metadata,
            selection=selection,
            winner_evidence_sha256=winner_evidence_sha256,
            verified_source_run=verified_source_run,
        )
    )

    def _model_rows(
        frame: pd.DataFrame, *, model_id: str, feature_block_id: str
    ) -> pd.DataFrame:
        return frame.loc[
            frame["model_id"].eq(model_id)
            & frame["feature_block_id"].eq(feature_block_id)
        ].copy()

    source = _model_rows(
        all_source,
        model_id=spec.candidate_model_id,
        feature_block_id=spec.candidate_feature_block_id,
    )
    target = _model_rows(
        all_target,
        model_id=spec.candidate_model_id,
        feature_block_id=spec.candidate_feature_block_id,
    )
    native_kalshi = _model_rows(
        all_native_kalshi,
        model_id=spec.candidate_model_id,
        feature_block_id=spec.candidate_feature_block_id,
    )
    transported_b0 = _model_rows(
        all_target,
        model_id=BASELINE_MODEL_ID,
        feature_block_id=BASELINE_FEATURE_BLOCK_ID,
    )
    native_b0 = _model_rows(
        all_native_kalshi,
        model_id=BASELINE_MODEL_ID,
        feature_block_id=BASELINE_FEATURE_BLOCK_ID,
    )
    for frame, label in (
        (source, "source"),
        (target, "target"),
        (native_kalshi, "native Kalshi"),
    ):
        duplicate = frame.duplicated(
            list(_TRANSPORT_PAIR_COLUMNS), keep=False
        )
        if duplicate.any():
            raise KalshiValidationError(
                f"{label} Task4 transport identities are not unique"
            )
    pairs = source.merge(
        target,
        on=list(_TRANSPORT_PAIR_COLUMNS),
        how="outer",
        suffixes=("_source", "_target"),
        validate="one_to_one",
        indicator=True,
    )
    unmatched_source_rows = int(pairs["_merge"].eq("left_only").sum())
    unmatched_target_rows = int(pairs["_merge"].eq("right_only").sum())
    paired = pairs.loc[pairs["_merge"].eq("both")].drop(
        columns="_merge"
    )
    if paired.empty:
        raise KalshiValidationError(
            "Task4 source and transport have no exact "
            "game/episode/fold/model/block/L/H development intersection"
        )
    for column in _SOURCE_ONLY_PROVENANCE_COLUMNS:
        left = paired[f"{column}_source"]
        right = paired[f"{column}_target"]
        if column in _SEQUENCE_PROVENANCE_COLUMNS:
            equal = [
                tuple(source_value) == tuple(target_value)
                for source_value, target_value in zip(
                    left, right, strict=True
                )
            ]
            matches = all(equal)
        else:
            matches = bool(left.eq(right).all())
        if not matches:
            raise KalshiValidationError(
                "Kalshi development transport must preserve source-only "
                f"{column}"
            )

    native_pairs_outer = target.merge(
        native_kalshi,
        on=list(_TRANSPORT_PAIR_COLUMNS),
        how="outer",
        suffixes=("_transport", "_native"),
        validate="one_to_one",
        indicator=True,
    )
    native_unmatched_transport = int(
        native_pairs_outer["_merge"].eq("left_only").sum()
    )
    native_unmatched_native = int(
        native_pairs_outer["_merge"].eq("right_only").sum()
    )
    native_pairs = native_pairs_outer.loc[
        native_pairs_outer["_merge"].eq("both")
    ].drop(columns="_merge")
    if native_pairs.empty:
        raise KalshiValidationError(
            "transported and native Kalshi OOF have no exact target pairs"
        )
    for truth_column in (
        "s_h_truth",
        "o_h_given_s_truth",
        "direction_truth",
    ):
        if not _nullable_truth_equal(
            native_pairs[f"{truth_column}_transport"],
            native_pairs[f"{truth_column}_native"],
        ):
            raise KalshiValidationError(
                f"transport/native Kalshi pairs disagree on {truth_column}"
            )
    if not native_pairs["actual_home_contract_id_transport"].eq(
        native_pairs["actual_home_contract_id_native"]
    ).all():
        raise KalshiValidationError(
            "transport/native Kalshi pairs disagree on target contract ID"
        )
    score_summary, calibration_summary = (
        _score_transport_against_native(native_pairs)
    )
    transport_b0_pairs = _merge_exact_candidate_b0_rows(
        target,
        transported_b0,
        layer="TRANSPORT_CANDIDATE_VS_TRANSPORT_B0",
    )
    transport_b0_diagnostics = _candidate_b0_integrated_log_loss(
        transport_b0_pairs,
        layer="TRANSPORT_CANDIDATE_VS_TRANSPORT_B0",
    )
    transport_b0_head_diagnostics = _candidate_b0_head_scores(
        transport_b0_pairs,
        layer="TRANSPORT_CANDIDATE_VS_TRANSPORT_B0",
    )
    transport_b0_head_gate_summary = _head_gate_summary(
        transport_b0_head_diagnostics
    )
    transport_b0_summary = _game_cluster_improvement_summary(
        transport_b0_diagnostics
    )
    native_noninferiority_summary = _native_noninferiority_summary(
        native_pairs
    )
    native_b0_pairs = _merge_exact_candidate_b0_rows(
        native_kalshi,
        native_b0,
        layer="NATIVE_KALSHI_CANDIDATE_VS_NATIVE_KALSHI_B0",
    )
    population = transport_b0_pairs[
        [
            *_MODEL_COMPARISON_PAIR_COLUMNS,
            "actual_home_contract_id_candidate",
            "s_h_truth_candidate",
            "o_h_given_s_truth_candidate",
            "direction_truth_candidate",
        ]
    ].merge(
        native_b0_pairs[
            [
                *_MODEL_COMPARISON_PAIR_COLUMNS,
                "actual_home_contract_id_candidate",
                "s_h_truth_candidate",
                "o_h_given_s_truth_candidate",
                "direction_truth_candidate",
            ]
        ],
        on=list(_MODEL_COMPARISON_PAIR_COLUMNS),
        how="outer",
        suffixes=("_transport", "_native"),
        validate="one_to_one",
        indicator=True,
    )
    if (
        not population["_merge"].eq("both").all()
        or not population[
            "actual_home_contract_id_candidate_transport"
        ].eq(
            population[
                "actual_home_contract_id_candidate_native"
            ]
        ).all()
        or not _nullable_truth_equal(
            population["s_h_truth_candidate_transport"],
            population["s_h_truth_candidate_native"],
        )
        or not _nullable_truth_equal(
            population["o_h_given_s_truth_candidate_transport"],
            population["o_h_given_s_truth_candidate_native"],
        )
        or not _nullable_truth_equal(
            population["direction_truth_candidate_transport"],
            population["direction_truth_candidate_native"],
        )
    ):
        raise KalshiValidationError(
            "transported and native candidate-vs-B0 layers require an "
            "identical Kalshi truth/identity population"
        )
    native_b0_diagnostics = _candidate_b0_integrated_log_loss(
        native_b0_pairs,
        layer="NATIVE_KALSHI_CANDIDATE_VS_NATIVE_KALSHI_B0",
    )
    native_b0_summary = _game_cluster_improvement_summary(
        native_b0_diagnostics
    )
    native_b0_score_summary = pd.DataFrame(
        [
            {
                "comparison_layer": (
                    "NATIVE_KALSHI_CANDIDATE_VS_NATIVE_KALSHI_B0"
                ),
                **native_b0_summary,
            }
        ]
    )

    metadata_weeks = authority_metadata.development.set_index(
        "game_id"
    )["nfl_week"]
    for game_id, week in zip(
        paired["game_id"], paired["nfl_week"], strict=True
    ):
        if (
            str(game_id) not in metadata_weeks.index
            or int(metadata_weeks.loc[str(game_id)]) != int(week)
        ):
            raise KalshiValidationError(
                "transport game/week is not bound to development "
                "authority metadata 1..12"
            )

    diagnostics = paired.loc[
        :,
        [
            *_TRANSPORT_PAIR_COLUMNS,
            "source_row_id_source",
            "source_row_id_target",
            "actual_home_contract_id_source",
            "actual_home_contract_id_target",
        ],
    ].copy()
    for column in _SOURCE_ONLY_PROVENANCE_COLUMNS:
        diagnostics[column] = paired[f"{column}_source"]
    for column in _PROBABILITY_COLUMNS:
        source_probability = pd.to_numeric(
            paired[f"{column}_source"], errors="coerce"
        )
        target_probability = pd.to_numeric(
            paired[f"{column}_target"], errors="coerce"
        )
        diagnostics[f"{column}_source"] = source_probability
        diagnostics[f"{column}_target"] = target_probability
        diagnostics[f"{column}_transport_delta"] = (
            target_probability - source_probability
        )
    diagnostics["training_venue"] = SOURCE_VENUE
    diagnostics["calibration_venue"] = SOURCE_VENUE
    diagnostics["transport_mode"] = TRANSPORT_MODE
    diagnostics["target_recalibration_applied"] = False
    native_diagnostics = native_pairs.loc[
        :,
        [
            *_TRANSPORT_PAIR_COLUMNS,
            "source_row_id_transport",
            "source_row_id_native",
            "actual_home_contract_id_transport",
            "actual_home_contract_id_native",
            "s_h_truth_transport",
            "o_h_given_s_truth_transport",
            "direction_truth_transport",
        ],
    ].copy()
    for column in _PROBABILITY_COLUMNS:
        native_diagnostics[f"{column}_transport"] = pd.to_numeric(
            native_pairs[f"{column}_transport"], errors="coerce"
        )
        native_diagnostics[f"{column}_native_kalshi"] = pd.to_numeric(
            native_pairs[f"{column}_native"], errors="coerce"
        )

    coverage_keys = [
        "landmark_seconds",
        "endpoint_seconds",
        "model_id",
        "feature_block_id",
    ]

    def _coverage_counts(
        frame: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        return (
            frame.groupby(coverage_keys, sort=True, as_index=False)
            .size()
            .rename(columns={"size": column})
        )

    coverage = _coverage_counts(source, "source_identity_count")
    coverage = coverage.merge(
        _coverage_counts(target, "target_identity_count"),
        on=coverage_keys,
        how="outer",
        validate="one_to_one",
    ).merge(
        _coverage_counts(paired, "paired_identity_count"),
        on=coverage_keys,
        how="outer",
        validate="one_to_one",
    )
    count_columns = [
        "source_identity_count",
        "target_identity_count",
        "paired_identity_count",
    ]
    coverage[count_columns] = (
        coverage[count_columns].fillna(0).astype(int)
    )
    coverage["unmatched_source_count"] = (
        coverage["source_identity_count"]
        - coverage["paired_identity_count"]
    )
    coverage["unmatched_target_count"] = (
        coverage["target_identity_count"]
        - coverage["paired_identity_count"]
    )
    coverage["paired_over_source_coverage"] = np.where(
        coverage["source_identity_count"].gt(0),
        coverage["paired_identity_count"]
        / coverage["source_identity_count"],
        np.nan,
    )
    coverage["paired_over_target_coverage"] = np.where(
        coverage["target_identity_count"].gt(0),
        coverage["paired_identity_count"]
        / coverage["target_identity_count"],
        np.nan,
    )
    coverage["intersection_evaluable"] = coverage[
        "paired_identity_count"
    ].gt(0)

    attrition_rows = pairs.loc[
        ~pairs["_merge"].eq("both"),
        [
            "game_id",
            "atomic_information_episode_id",
            "landmark_seconds",
            "endpoint_seconds",
            "model_id",
            "feature_block_id",
            "_merge",
        ],
    ].copy()
    attrition_rows["attrition_side"] = attrition_rows["_merge"].map(
        {
            "left_only": "UNMATCHED_SOURCE",
            "right_only": "UNMATCHED_TARGET",
        }
    )
    attrition_rows["attrition_reason"] = (
        "NO_EXACT_CROSS_VENUE_IDENTITY"
    )
    attrition = (
        attrition_rows.groupby(
            [
                "attrition_side",
                "attrition_reason",
                "landmark_seconds",
                "endpoint_seconds",
                "model_id",
                "feature_block_id",
            ],
            sort=True,
            observed=True,
            as_index=False,
        )
        .agg(
            row_count=("game_id", "size"),
            game_count=("game_id", "nunique"),
            episode_count=(
                "atomic_information_episode_id",
                "nunique",
            ),
        )
        .reset_index(drop=True)
    )
    paired_game_count = int(paired["game_id"].nunique())
    clean_anchor_present = bool(
        (
            paired["landmark_seconds"].eq(3)
            & paired["endpoint_seconds"].eq(30)
        ).any()
    )
    coverage_gate = (
        paired_game_count >= MIN_TRANSPORT_PAIRED_GAMES
        and clean_anchor_present
    )
    expected_score_pairs = {
        ("S_H", "LOG_LOSS"),
        ("S_H", "BRIER"),
        ("O_H_GIVEN_S", "LOG_LOSS"),
        ("O_H_GIVEN_S", "BRIER"),
        ("DIRECTION", "LOG_LOSS"),
        ("DIRECTION", "BRIER"),
    }
    expected_model_blocks = set(
        map(
            tuple,
            native_pairs[
                ["model_id", "feature_block_id"]
            ].drop_duplicates().to_numpy(),
        )
    )
    observed_model_blocks: set[tuple[object, object]] = set()
    complete_score_groups = True
    if not score_summary.empty:
        for model_block, group in score_summary.groupby(
            ["model_id", "feature_block_id"],
            sort=True,
            observed=True,
        ):
            observed_model_blocks.add(tuple(model_block))
            observed_pairs = set(
                zip(
                    group["head"],
                    group["metric"],
                    strict=True,
                )
            )
            complete_score_groups = (
                complete_score_groups
                and observed_pairs == expected_score_pairs
                and len(group) == len(expected_score_pairs)
            )
    else:
        complete_score_groups = False
    score_gate = bool(
        observed_model_blocks == expected_model_blocks
        and complete_score_groups
        and np.isfinite(
            score_summary[
                [
                    "transport_score",
                    "native_kalshi_score",
                    "score_degradation",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and score_summary["evaluated_game_count"]
        .ge(MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD)
        .all()
        and score_summary["evaluated_row_count"]
        .ge(MIN_TRANSPORT_EVALUATED_ROWS_PER_HEAD)
        .all()
    )
    catastrophic = False
    if not score_summary.empty:
        log_loss_bad = score_summary["metric"].eq("LOG_LOSS") & (
            score_summary["score_degradation"]
            > MAX_LOG_LOSS_DEGRADATION
        )
        brier_bad = score_summary["metric"].eq("BRIER") & (
            score_summary["score_degradation"]
            > MAX_BRIER_DEGRADATION
        )
        catastrophic = bool((log_loss_bad | brier_bad).any())
    nuisance_rows = transport_b0_head_gate_summary.loc[
        transport_b0_head_gate_summary["head"].isin(
            ("S_H", "O_H_GIVEN_S")
        )
        & transport_b0_head_gate_summary["metric"].eq("LOG_LOSS")
    ]
    direction_rows = transport_b0_head_gate_summary.loc[
        transport_b0_head_gate_summary["head"].eq("DIRECTION")
        & transport_b0_head_gate_summary["metric"].isin(
            ("LOG_LOSS", "BRIER")
        )
    ]
    binary_nuisance_gate = bool(
        len(nuisance_rows) == 2
        and nuisance_rows["gate_passed"].all()
    )
    direction_gate = bool(
        len(direction_rows) == 2
        and direction_rows["gate_passed"].all()
    )
    native_noninferiority_gate = bool(
        len(native_noninferiority_summary) == 6
        and native_noninferiority_summary[
            "upper95_within_margin"
        ].all()
    )
    transport_gate = (
        coverage_gate
        and score_gate
        and bool(
            transport_b0_summary["improvement_gate_passed"]
        )
        and binary_nuisance_gate
        and direction_gate
        and native_noninferiority_gate
    )
    return DevelopmentVenueValidation(
        source_venue=SOURCE_VENUE,
        target_venue=TARGET_VENUE,
        cohort="development",
        cohort_authority_sha256=(
            authority_metadata.cohort_authority_sha256
        ),
        run_config_sha256=model_run.run_config_sha256,
        schema_version=HISTORICAL_SCHEMA_VERSION,
        analysis_scope=HISTORICAL_ANALYSIS_SCOPE,
        target_contract=str(paired["target_contract"].iloc[0]),
        claim_boundary=str(paired["claim_boundary"].iloc[0]),
        candidate_model_id=spec.candidate_model_id,
        candidate_feature_block_id=spec.candidate_feature_block_id,
        diagnostic_status=(
            "HISTORICAL_PREDICTIVE_UTILITY_CANDIDATE"
            if transport_gate
            else "HISTORICAL_PREDICTIVE_UTILITY_REJECTED"
        ),
        execution_claim_eligible=False,
        tick_claim_eligible=False,
        continuity_claim_eligible=False,
        target_recalibration_applied=False,
        paired_identity_count=len(paired),
        paired_game_count=paired_game_count,
        unmatched_source_rows=unmatched_source_rows,
        unmatched_target_rows=unmatched_target_rows,
        native_comparator_paired_rows=len(native_pairs),
        native_comparator_unmatched_transport_rows=(
            native_unmatched_transport
        ),
        native_comparator_unmatched_native_rows=(
            native_unmatched_native
        ),
        coverage_gate_passed=coverage_gate,
        score_gate_passed=score_gate,
        catastrophic_degradation=catastrophic,
        transport_b0_paired_rows=int(
            transport_b0_summary["paired_row_count"]
        ),
        transport_b0_paired_games=int(
            transport_b0_summary["paired_game_count"]
        ),
        transport_b0_mean_improvement=float(
            transport_b0_summary["mean_improvement"]
        ),
        transport_b0_ci_low=float(transport_b0_summary["ci_low"]),
        transport_b0_ci_high=float(transport_b0_summary["ci_high"]),
        transport_b0_bootstrap_probability_improved=float(
            transport_b0_summary["bootstrap_probability_improved"]
        ),
        transport_b0_bootstrap_samples=int(
            transport_b0_summary["bootstrap_samples"]
        ),
        transport_b0_bootstrap_seed=int(
            transport_b0_summary["bootstrap_seed"]
        ),
        transport_b0_anchor_game_count=int(
            transport_b0_summary["anchor_game_count"]
        ),
        transport_b0_anchor_mean_improvement=float(
            transport_b0_summary["anchor_mean_improvement"]
        ),
        transport_b0_improvement_gate_passed=bool(
            transport_b0_summary["improvement_gate_passed"]
        ),
        binary_nuisance_gate_passed=binary_nuisance_gate,
        direction_gate_passed=direction_gate,
        native_noninferiority_gate_passed=(
            native_noninferiority_gate
        ),
        transport_gate_passed=transport_gate,
        coverage_audit=coverage.sort_values(
            coverage_keys, kind="mergesort"
        ).reset_index(drop=True),
        attrition=attrition,
        exact_pair_diagnostics=diagnostics.sort_values(
            list(_TRANSPORT_PAIR_COLUMNS), kind="mergesort"
        ).reset_index(drop=True),
        native_comparison_diagnostics=native_diagnostics.sort_values(
            list(_TRANSPORT_PAIR_COLUMNS), kind="mergesort"
        ).reset_index(drop=True),
        transport_b0_pair_diagnostics=(
            transport_b0_diagnostics.sort_values(
                list(_MODEL_COMPARISON_PAIR_COLUMNS),
                kind="mergesort",
            ).reset_index(drop=True)
        ),
        native_b0_pair_diagnostics=native_b0_diagnostics.sort_values(
            list(_MODEL_COMPARISON_PAIR_COLUMNS),
            kind="mergesort",
        ).reset_index(drop=True),
        native_b0_score_summary=native_b0_score_summary,
        transport_b0_head_diagnostics=(
            transport_b0_head_diagnostics.sort_values(
                [
                    "head",
                    "metric",
                    *_MODEL_COMPARISON_PAIR_COLUMNS,
                ],
                kind="mergesort",
            ).reset_index(drop=True)
        ),
        transport_b0_head_gate_summary=(
            transport_b0_head_gate_summary.sort_values(
                ["head", "metric"], kind="mergesort"
            ).reset_index(drop=True)
        ),
        native_noninferiority_summary=(
            native_noninferiority_summary.sort_values(
                ["head", "metric"], kind="mergesort"
            ).reset_index(drop=True)
        ),
        score_summary=score_summary.sort_values(
            ["model_id", "feature_block_id", "head", "metric"],
            kind="mergesort",
        ).reset_index(drop=True),
        calibration_summary=calibration_summary.sort_values(
            [
                "model_id",
                "feature_block_id",
                "head",
                "class_label",
            ],
            kind="mergesort",
        ).reset_index(drop=True),
    )


def _require_frozen_stage_b_v3_winner(
    stage_b_selection: object,
    *,
    authority: FrozenDevelopmentAuthority,
) -> ModelSelectionResult:
    try:
        verified = verify_stage_b_v3_selection_result(
            stage_b_selection,
            authority=authority,
        )
    except ModelSelectionError as exc:
        raise KalshiValidationError(
            "Stage-B V3 selection evidence failed complete "
            "authoritative verification"
        ) from exc
    winner = verified.winner
    if (
        verified.decision_status != "MODEL_ADVANCE"
        or not isinstance(winner, ModelSelectionResult)
        or not winner.selected
        or winner.spec.selection_venue != SOURCE_VENUE
    ):
        raise KalshiValidationError(
            "Stage-B V3 selection evidence is not one frozen "
            "Polymarket winner"
        )
    return winner


def _verify_preholdout_metadata_lock(
    lock: object,
    *,
    cohort_authority_sha256: str,
    authority_metadata: FrozenAuthorityMetadata,
) -> Mapping[str, object]:
    if not isinstance(lock, Mapping):
        raise KalshiValidationError(
            "factor-conditioned utility requires a verified pre-holdout "
            "metadata lock"
        )
    required = {
        "lock_event",
        "cohort_authority_sha256",
        "metadata_sha256",
        "current_metadata_sha256",
        "chain_sha256",
        "metadata_access_count",
        "development_game_count",
        "holdout_game_count",
        "holdout_reaction_read_count",
        "market_reaction_exposure",
        "sports_outcome_exposure",
        "sports_outcome_source_evidence_sha256",
        "sports_outcome_observation_count",
        "stage_a_outcome_validation_eligible",
        "stage_b_market_reaction_validation_eligible",
        "lock_sha256",
    }
    if set(lock) != required:
        raise KalshiValidationError(
            "factor-conditioned utility requires a verified pre-holdout "
            "metadata lock"
        )
    if not isinstance(authority_metadata, FrozenAuthorityMetadata):
        raise KalshiValidationError(
            "factor-conditioned utility requires current authority "
            "metadata"
        )
    try:
        current_metadata_sha256 = (
            _recompute_authority_metadata_sha256(authority_metadata)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KalshiValidationError(
            "factor-conditioned utility requires current authority "
            "metadata"
        ) from exc
    payload = {key: lock[key] for key in lock if key != "lock_sha256"}
    if (
        lock["lock_sha256"] != _canonical_sha256(payload)
        or lock["lock_event"] != "PRE_SHORTLIST_METADATA_ONLY_LOCK"
        or lock["cohort_authority_sha256"]
        != cohort_authority_sha256
        or authority_metadata.cohort_authority_sha256
        != cohort_authority_sha256
        or authority_metadata.metadata_sha256
        != lock["metadata_sha256"]
        or current_metadata_sha256 != lock["metadata_sha256"]
        or lock["metadata_sha256"] != lock["current_metadata_sha256"]
        or not _is_sha256(lock["metadata_sha256"])
        or not _is_sha256(lock["chain_sha256"])
        or lock["development_game_count"] != EXPECTED_DEVELOPMENT_GAMES
        or lock["holdout_game_count"] != EXPECTED_HOLDOUT_GAMES
        or lock["holdout_reaction_read_count"] != 0
        or lock["market_reaction_exposure"] != "SEALED_UNREAD"
        or lock["sports_outcome_exposure"] != "PRIOR_EXPOSED_X11"
        or lock["sports_outcome_source_evidence_sha256"]
        != X11_SPORTS_OUTCOME_EVIDENCE_SHA256
        or lock["sports_outcome_observation_count"]
        != X11_HOLDOUT_DRIVE_OUTCOME_COUNT
        or authority_metadata.holdout_reaction_read_count != 0
        or not authority_metadata.holdout[
            "reaction_read_count"
        ].eq(0).all()
        or not authority_metadata.holdout[
            "market_reaction_exposure"
        ].eq("SEALED_UNREAD").all()
        or not authority_metadata.holdout[
            "sports_outcome_exposure"
        ].eq("PRIOR_EXPOSED_X11").all()
        or authority_metadata.stage_a_outcome_validation_eligible
        is not False
        or (
            authority_metadata.stage_b_market_reaction_validation_eligible
            is not True
        )
        or (
            authority_metadata.sports_outcome_source_evidence_sha256
            != X11_SPORTS_OUTCOME_EVIDENCE_SHA256
        )
        or (
            authority_metadata.sports_outcome_observation_count
            != X11_HOLDOUT_DRIVE_OUTCOME_COUNT
        )
        or lock["stage_a_outcome_validation_eligible"] is not False
        or lock["stage_b_market_reaction_validation_eligible"] is not True
        or isinstance(lock["metadata_access_count"], bool)
        or not isinstance(lock["metadata_access_count"], Integral)
        or int(lock["metadata_access_count"]) < 0
    ):
        raise KalshiValidationError(
            "factor-conditioned utility requires a verified pre-holdout "
            "metadata lock"
        )
    return lock


def build_cross_venue_predictive_utility_shortlist(
    stage_b_selection: StageBModelSelectionResult,
    *,
    transport_validation: DevelopmentVenueValidation,
    factor_membership_evidence: FrozenFactorMembershipEvidence,
    authority_metadata: FrozenAuthorityMetadata,
    preholdout_metadata_lock: Mapping[str, object],
    min_support_games: int = 30,
    min_support_episodes: int = 20,
) -> pd.DataFrame:
    """Condition frozen predictive scores on exact V4 factor membership.

    This is a model-utility audit, not a price-reaction or trading claim.
    Candidate identity remains frozen and cannot be reranked here.
    """

    selection = _require_frozen_stage_b_v3_winner(
        stage_b_selection,
        authority=authority_metadata.selection_authority,
    )
    metadata_lock = _verify_preholdout_metadata_lock(
        preholdout_metadata_lock,
        cohort_authority_sha256=selection.cohort_authority_sha256,
        authority_metadata=authority_metadata,
    )
    if not isinstance(
        transport_validation, DevelopmentVenueValidation
    ):
        raise KalshiValidationError(
            "transport_validation must be a DevelopmentVenueValidation"
        )
    if not isinstance(
        factor_membership_evidence, FrozenFactorMembershipEvidence
    ):
        raise KalshiValidationError(
            "factor membership must come from the frozen artifact evidence"
        )
    if (
        factor_membership_evidence.authority_manifest_path
        != _FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_PATH
        or factor_membership_evidence.authority_manifest_file_sha256
        != _FACTOR_MEMBERSHIP_AUTHORITY_MANIFEST_FILE_SHA256
        or factor_membership_evidence.authority_batch_sha256
        != _FACTOR_MEMBERSHIP_AUTHORITY_BATCH_SHA256
        or factor_membership_evidence.facts_authority_manifest_path
        != _FACTOR_FACTS_AUTHORITY_MANIFEST_PATH
        or (
            factor_membership_evidence
            .facts_authority_manifest_file_sha256
            != _FACTOR_FACTS_AUTHORITY_MANIFEST_FILE_SHA256
        )
        or factor_membership_evidence.facts_authority_batch_sha256
        != _FACTOR_FACTS_AUTHORITY_BATCH_SHA256
        or factor_membership_evidence.cohort_authority_sha256
        != _FACTOR_MEMBERSHIP_COHORT_AUTHORITY_SHA256
        or factor_membership_evidence.registry_path
        != _FACTOR_REGISTRY_PATH
        or factor_membership_evidence.registry_file_sha256
        != _FACTOR_REGISTRY_FILE_SHA256
        or factor_membership_evidence.registry_semantic_sha256
        != _FACTOR_REGISTRY_SEMANTIC_SHA256
        or factor_membership_evidence.membership_rows_sha256
        != _FACTOR_MEMBERSHIP_ROWS_SHA256
        or (
            factor_membership_evidence
            .membership_artifact_bindings_sha256
            != _FACTOR_MEMBERSHIP_ARTIFACT_BINDINGS_SHA256
        )
        or len(factor_membership_evidence.membership_rows)
        != _FACTOR_MEMBERSHIP_ROW_COUNT
        or factor_membership_evidence.membership_rows[
            "game_id"
        ].nunique()
        != EXPECTED_DEVELOPMENT_GAMES
        or factor_membership_evidence.membership_rows[
            ["game_id", "atomic_information_episode_id"]
        ].drop_duplicates().shape[0]
        != _FACTOR_MEMBERSHIP_EPISODE_COUNT
        or _factor_membership_rows_sha256(
            factor_membership_evidence.membership_rows
        )
        != _FACTOR_MEMBERSHIP_ROWS_SHA256
    ):
        raise KalshiValidationError(
            "factor membership evidence does not bind the frozen artifacts"
        )
    if (
        isinstance(min_support_games, bool)
        or not isinstance(min_support_games, Integral)
        or int(min_support_games) != 30
        or isinstance(min_support_episodes, bool)
        or not isinstance(min_support_episodes, Integral)
        or int(min_support_episodes) != 20
    ):
        raise KalshiValidationError(
            "factor-conditioned support contract is frozen at "
            "30 games and 20 episodes"
        )
    if (
        selection.spec.candidate_model_id
        != transport_validation.candidate_model_id
        or selection.spec.candidate_feature_block_id
        != transport_validation.candidate_feature_block_id
    ):
        raise KalshiValidationError(
            "Polymarket selection and Kalshi transport candidate differ"
        )
    if (
        selection.cohort_authority_sha256
        != transport_validation.cohort_authority_sha256
        or selection.target_contract
        != transport_validation.target_contract
        or selection.claim_boundary
        != transport_validation.claim_boundary
        or selection.schema_version
        != transport_validation.schema_version
        or selection.analysis_scope
        != transport_validation.analysis_scope
        or selection.cohort_authority_sha256
        != factor_membership_evidence.cohort_authority_sha256
    ):
        raise KalshiValidationError(
            "Polymarket selection and Kalshi transport authority differ"
        )
    if transport_validation.target_recalibration_applied:
        raise KalshiValidationError(
            "factor shortlist forbids target recalibration"
        )

    membership = factor_membership_evidence.membership_rows.loc[
        :,
        [
            "game_id",
            "atomic_information_episode_id",
            "factor_id",
            "factor_version",
        ],
    ].copy()

    poly_required = {
        "game_id",
        "nfl_week",
        "atomic_information_episode_id",
        "venue",
        "landmark_seconds",
        "endpoint_seconds",
        "fold_id",
        "model_id_candidate",
        "feature_block_id_candidate",
        "model_id_b0",
        "feature_block_id_b0",
        "loss_improvement",
    }
    kalshi_required = {
        "game_id",
        "nfl_week",
        "atomic_information_episode_id",
        "landmark_seconds",
        "endpoint_seconds",
        "fold_id",
        "comparison_layer",
        "candidate_model_id",
        "candidate_feature_block_id",
        "baseline_model_id",
        "baseline_feature_block_id",
        "s_h_truth",
        "o_h_given_s_truth",
        "direction_truth",
        "loss_improvement",
    }
    missing_poly = sorted(
        poly_required.difference(selection.paired_rows.columns)
    )
    missing_kalshi = sorted(
        kalshi_required.difference(
            transport_validation.transport_b0_pair_diagnostics.columns
        )
    )
    if missing_poly or missing_kalshi:
        raise KalshiValidationError(
            "factor shortlist requires exact clean-anchor evidence "
            f"(Polymarket missing={missing_poly}, Kalshi missing={missing_kalshi})"
        )

    poly_anchor = selection.paired_rows.loc[
        selection.paired_rows["landmark_seconds"].eq(
            selection.spec.anchor_landmark_seconds
        )
        & selection.paired_rows["endpoint_seconds"].eq(
            selection.spec.anchor_endpoint_seconds
        ),
        sorted(poly_required),
    ].rename(
        columns={
            "model_id_candidate": "candidate_model_id",
            "feature_block_id_candidate": (
                "candidate_feature_block_id"
            ),
            "model_id_b0": "baseline_model_id",
            "feature_block_id_b0": "baseline_feature_block_id",
        }
    )
    kalshi_anchor = (
        transport_validation.transport_b0_pair_diagnostics.loc[
            lambda frame: frame["landmark_seconds"].eq(
                selection.spec.anchor_landmark_seconds
            )
            & frame["endpoint_seconds"].eq(
                selection.spec.anchor_endpoint_seconds
            ),
            sorted(kalshi_required),
        ].copy()
    )
    if poly_anchor.empty or not poly_anchor["venue"].eq(
        SOURCE_VENUE
    ).all():
        raise KalshiValidationError(
            "Polymarket factor evidence must use the clean source anchor"
        )
    if kalshi_anchor.empty:
        raise KalshiValidationError(
            "Kalshi factor evidence has no exact clean-anchor truth rows"
        )
    for frame, label in (
        (poly_anchor, "Polymarket"),
        (kalshi_anchor, "Kalshi"),
    ):
        if (
            not frame["candidate_model_id"].eq(
                selection.spec.candidate_model_id
            ).all()
            or not frame["candidate_feature_block_id"].eq(
                selection.spec.candidate_feature_block_id
            ).all()
            or not frame["baseline_model_id"].eq(
                selection.spec.baseline_model_id
            ).all()
            or not frame["baseline_feature_block_id"].eq(
                selection.spec.baseline_feature_block_id
            ).all()
        ):
            raise KalshiValidationError(
                f"{label} factor evidence has non-frozen model identity"
            )
        improvement = pd.to_numeric(
            frame["loss_improvement"], errors="coerce"
        )
        if not np.isfinite(improvement.to_numpy(dtype=float)).all():
            raise KalshiValidationError(
                f"{label} factor loss improvements must be finite"
            )
        frame["loss_improvement"] = improvement.astype(float)
    if not kalshi_anchor["comparison_layer"].eq(
        "TRANSPORT_CANDIDATE_VS_TRANSPORT_B0"
    ).all():
        raise KalshiValidationError(
            "Kalshi factor evidence is not transported candidate-vs-B0"
        )

    episode_identity = [
        "game_id",
        "atomic_information_episode_id",
    ]
    poly_joined = membership.merge(
        poly_anchor,
        on=episode_identity,
        how="inner",
        validate="many_to_one",
    )
    kalshi_joined = membership.merge(
        kalshi_anchor,
        on=episode_identity,
        how="inner",
        validate="many_to_one",
    )
    if poly_joined.empty and kalshi_joined.empty:
        raise KalshiValidationError(
            "frozen factor membership matches no clean-anchor evidence"
        )
    pair_grain = list(_CROSS_VENUE_FACTOR_PAIR_GRAIN)
    for frame, label in (
        (poly_joined, "Polymarket"),
        (kalshi_joined, "Kalshi"),
    ):
        if frame.duplicated(pair_grain, keep=False).any():
            raise KalshiValidationError(
                f"{label} factor pair grain is not unique"
            )
    pair_outer = poly_joined[
        [*pair_grain, "loss_improvement"]
    ].merge(
        kalshi_joined[[*pair_grain, "loss_improvement"]],
        on=pair_grain,
        how="outer",
        suffixes=("_polymarket", "_kalshi"),
        validate="one_to_one",
        indicator=True,
    )
    shared_pairs = pair_outer.loc[
        pair_outer["_merge"].eq("both")
    ].drop(columns="_merge")
    polymarket_only = pair_outer.loc[
        pair_outer["_merge"].eq("left_only")
    ].drop(columns="_merge")
    kalshi_only = pair_outer.loc[
        pair_outer["_merge"].eq("right_only")
    ].drop(columns="_merge")
    versions = (
        pd.concat(
            [
                poly_joined[["factor_id", "factor_version"]],
                kalshi_joined[["factor_id", "factor_version"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(
            ["factor_id", "factor_version"], kind="mergesort"
        )
        .reset_index(drop=True)
    )

    shared_support: dict[tuple[str, str], dict[str, object]] = {}
    for raw_key, group in shared_pairs.groupby(
        ["factor_id", "factor_version"], sort=True
    ):
        key = (str(raw_key[0]), str(raw_key[1]))
        episode_count = len(
            group[
                ["game_id", "atomic_information_episode_id"]
            ].drop_duplicates()
        )
        game_count = int(group["game_id"].nunique())
        shared_support[key] = {
            "shared_pair_row_count": len(group),
            "shared_game_count": game_count,
            "shared_episode_count": episode_count,
            "shared_support_status": (
                "SUPPORTED"
                if game_count >= int(min_support_games)
                and episode_count >= int(min_support_episodes)
                else "INSUFFICIENT_SUPPORT"
            ),
        }

    utility_records: list[dict[str, object]] = []
    if not shared_pairs.empty:
        for (
            factor_id,
            factor_version,
        ), factor_rows in shared_pairs.groupby(
            ["factor_id", "factor_version"],
            sort=True,
            observed=True,
        ):
            for venue, improvement_column in (
                (SOURCE_VENUE, "loss_improvement_polymarket"),
                (TARGET_VENUE, "loss_improvement_kalshi"),
            ):
                game_units = (
                    factor_rows.groupby(
                        "game_id", sort=True, as_index=False
                    )
                    .agg(
                        log_score_improvement=(
                            improvement_column,
                            "mean",
                        )
                    )
                    .reset_index(drop=True)
                )
                values = game_units[
                    "log_score_improvement"
                ].to_numpy(dtype=float)
                low, high, probability = _bootstrap_game_mean(values)
                if len(values) >= 2:
                    rng = np.random.default_rng(BOOTSTRAP_SEED)
                    selections = rng.integers(
                        0,
                        len(values),
                        size=(BOOTSTRAP_SAMPLES, len(values)),
                    )
                    draws = values[selections].mean(axis=1)
                    p_value = float(
                        min(
                            1.0,
                            2.0
                            * min(
                                np.mean(draws <= 0.0),
                                np.mean(draws >= 0.0),
                            ),
                        )
                    )
                    leave_one_out = np.asarray(
                        [
                            np.delete(values, index).mean()
                            for index in range(len(values))
                        ],
                        dtype=float,
                    )
                    loo_positive_rate = float(
                        np.mean(leave_one_out > 0.0)
                    )
                else:
                    p_value = float("nan")
                    loo_positive_rate = float("nan")
                absolute_sum = float(np.abs(values).sum())
                max_contribution = (
                    float(np.abs(values).max() / absolute_sum)
                    if len(values) and absolute_sum > 0.0
                    else float("nan")
                )
                utility_records.append(
                    {
                        "factor_id": str(factor_id),
                        "factor_version": str(factor_version),
                        "venue": venue,
                        "support_games": len(game_units),
                        "support_episodes": len(
                            factor_rows[
                                [
                                    "game_id",
                                    "atomic_information_episode_id",
                                ]
                            ].drop_duplicates()
                        ),
                        "mean_log_score_improvement": (
                            float(values.mean())
                            if len(values)
                            else float("nan")
                        ),
                        "median_log_score_improvement": (
                            float(np.median(values))
                            if len(values)
                            else float("nan")
                        ),
                        "log_score_improvement_ci_lower": low,
                        "log_score_improvement_ci_upper": high,
                        "bootstrap_probability_improved": probability,
                        "bootstrap_two_sided_p_value": p_value,
                        "leave_one_game_out_positive_rate": (
                            loo_positive_rate
                        ),
                        "max_single_game_absolute_contribution_ratio": (
                            max_contribution
                        ),
                    }
                )
    utility = pd.DataFrame(
        utility_records,
        columns=[
            "factor_id",
            "factor_version",
            "venue",
            "support_games",
            "support_episodes",
            "mean_log_score_improvement",
            "median_log_score_improvement",
            "log_score_improvement_ci_lower",
            "log_score_improvement_ci_upper",
            "bootstrap_probability_improved",
            "bootstrap_two_sided_p_value",
            "leave_one_game_out_positive_rate",
            "max_single_game_absolute_contribution_ratio",
        ],
    )
    if not utility.empty:
        utility["bh_q_value"] = float("nan")
        for venue, indices in utility.groupby(
            "venue", sort=True
        ).groups.items():
            del venue
            ordered = utility.loc[
                indices, "bootstrap_two_sided_p_value"
            ].sort_values(kind="mergesort")
            finite = ordered.loc[np.isfinite(ordered)]
            if finite.empty:
                continue
            raw = (
                finite.to_numpy(dtype=float)
                * len(finite)
                / np.arange(1, len(finite) + 1, dtype=float)
            )
            adjusted = np.minimum.accumulate(raw[::-1])[::-1]
            utility.loc[finite.index, "bh_q_value"] = np.minimum(
                adjusted, 1.0
            )
        utility["predictive_utility_gate_passed"] = (
            utility["support_games"].ge(int(min_support_games))
            & utility["support_episodes"].ge(int(min_support_episodes))
            & utility["mean_log_score_improvement"].gt(0.0)
            & utility["log_score_improvement_ci_lower"].gt(0.0)
            & utility["bh_q_value"].le(DEFAULT_MAX_Q_VALUE)
            & utility["leave_one_game_out_positive_rate"].ge(
                DEFAULT_MIN_LOO_SAME_SIGN
            )
            & utility[
                "max_single_game_absolute_contribution_ratio"
            ].le(DEFAULT_MAX_GAME_CONTRIBUTION)
        )
    venue_metrics = (
        "support_games",
        "support_episodes",
        "mean_log_score_improvement",
        "median_log_score_improvement",
        "log_score_improvement_ci_lower",
        "log_score_improvement_ci_upper",
        "bootstrap_probability_improved",
        "bootstrap_two_sided_p_value",
        "bh_q_value",
        "leave_one_game_out_positive_rate",
        "max_single_game_absolute_contribution_ratio",
        "predictive_utility_gate_passed",
    )
    records: list[dict[str, object]] = []
    for version_row in versions.itertuples(index=False):
        factor_key = (
            str(version_row.factor_id),
            str(version_row.factor_version),
        )
        factor_rows = utility.loc[
            utility["factor_id"].eq(version_row.factor_id)
            & utility["factor_version"].eq(
                version_row.factor_version
            )
        ] if not utility.empty else utility
        support = shared_support.get(
            factor_key,
            {
                "shared_pair_row_count": 0,
                "shared_game_count": 0,
                "shared_episode_count": 0,
                "shared_support_status": "NO_EXACT_SHARED_EVIDENCE",
            },
        )
        record: dict[str, object] = {
            "factor_id": str(version_row.factor_id),
            "factor_version": str(version_row.factor_version),
            **support,
        }
        for attrition_frame, prefix in (
            (polymarket_only, "polymarket_only"),
            (kalshi_only, "kalshi_only"),
        ):
            attrition_rows = attrition_frame.loc[
                attrition_frame["factor_id"].eq(
                    version_row.factor_id
                )
                & attrition_frame["factor_version"].eq(
                    version_row.factor_version
                )
            ]
            identities = tuple(
                "|".join(
                    f"{column}={row[column]}"
                    for column in _CROSS_VENUE_FACTOR_PAIR_GRAIN
                )
                for row in attrition_rows.sort_values(
                    pair_grain, kind="mergesort"
                ).to_dict("records")
            )
            record[f"{prefix}_pair_row_count"] = len(
                attrition_rows
            )
            record[f"{prefix}_game_count"] = int(
                attrition_rows["game_id"].nunique()
            )
            record[f"{prefix}_episode_count"] = len(
                attrition_rows[
                    [
                        "game_id",
                        "atomic_information_episode_id",
                    ]
                ].drop_duplicates()
            )
            record[f"{prefix}_pair_identities"] = identities
        for venue, prefix in (
            (SOURCE_VENUE, "polymarket"),
            (TARGET_VENUE, "kalshi"),
        ):
            venue_rows = factor_rows.loc[
                factor_rows["venue"].eq(venue)
            ]
            if len(venue_rows) > 1:
                raise KalshiValidationError(
                    "factor-conditioned utility produced duplicate venue rows"
                )
            if venue_rows.empty:
                for metric in venue_metrics:
                    if metric == "predictive_utility_gate_passed":
                        value = False
                    elif metric in {
                        "support_games",
                        "support_episodes",
                    }:
                        value = 0
                    else:
                        value = float("nan")
                    record[f"{prefix}_{metric}"] = value
            else:
                venue_row = venue_rows.iloc[0]
                for metric in venue_metrics:
                    record[f"{prefix}_{metric}"] = venue_row[
                        metric
                    ]
        record["cross_venue_both_positive"] = bool(
            len(factor_rows) == 2
            and factor_rows["mean_log_score_improvement"].gt(0.0).all()
        )
        factor_gate = bool(
            record["shared_support_status"] == "SUPPORTED"
            and record[
                "polymarket_predictive_utility_gate_passed"
            ]
            and record["kalshi_predictive_utility_gate_passed"]
            and record["cross_venue_both_positive"]
        )
        record["factor_conditioned_dual_venue_gate_passed"] = (
            factor_gate
        )
        record["polymarket_selection_gate_passed"] = bool(
            selection.selected
        )
        record["global_transport_gate_passed"] = bool(
            transport_validation.transport_gate_passed
        )
        registered = bool(
            selection.selected
            and transport_validation.transport_gate_passed
            and factor_gate
        )
        record["registered_shortlist_gate_passed"] = registered
        record["diagnostic_status"] = (
            "HISTORICAL_PREDICTIVE_UTILITY_CANDIDATE"
            if registered
            else "HISTORICAL_PREDICTIVE_UTILITY_REJECTED"
        )
        record["shortlist_schema_version"] = (
            FACTOR_PREDICTIVE_UTILITY_SCHEMA_VERSION
        )
        record["claim_landmark_seconds"] = (
            selection.spec.anchor_landmark_seconds
        )
        record["claim_endpoint_seconds"] = (
            selection.spec.anchor_endpoint_seconds
        )
        record["loss_improvement_sign_semantics"] = (
            LOSS_IMPROVEMENT_SIGN_SEMANTICS
        )
        record["target_contract"] = selection.target_contract
        record["claim_boundary"] = selection.claim_boundary
        record["source_schema_version"] = selection.schema_version
        record["analysis_scope"] = selection.analysis_scope
        record["exact_pair_grain"] = (
            _CROSS_VENUE_FACTOR_PAIR_GRAIN
        )
        record["factor_membership_authority_manifest_path"] = (
            factor_membership_evidence.authority_manifest_path
        )
        record[
            "factor_membership_authority_manifest_file_sha256"
        ] = (
            factor_membership_evidence
            .authority_manifest_file_sha256
        )
        record["factor_membership_authority_batch_sha256"] = (
            factor_membership_evidence.authority_batch_sha256
        )
        record["factor_facts_authority_manifest_path"] = (
            factor_membership_evidence.facts_authority_manifest_path
        )
        record["factor_facts_authority_manifest_file_sha256"] = (
            factor_membership_evidence
            .facts_authority_manifest_file_sha256
        )
        record["factor_facts_authority_batch_sha256"] = (
            factor_membership_evidence.facts_authority_batch_sha256
        )
        record["factor_membership_cohort_authority_sha256"] = (
            factor_membership_evidence.cohort_authority_sha256
        )
        record["factor_membership_rows_sha256"] = (
            factor_membership_evidence.membership_rows_sha256
        )
        record["factor_membership_artifact_bindings_sha256"] = (
            factor_membership_evidence
            .membership_artifact_bindings_sha256
        )
        record["factor_registry_path"] = (
            factor_membership_evidence.registry_path
        )
        record["factor_registry_file_sha256"] = (
            factor_membership_evidence.registry_file_sha256
        )
        record["factor_registry_semantic_sha256"] = (
            factor_membership_evidence.registry_semantic_sha256
        )
        record["execution_claim_eligible"] = False
        record["tick_claim_eligible"] = False
        record["continuity_claim_eligible"] = False
        record["preholdout_metadata_lock_sha256"] = metadata_lock[
            "lock_sha256"
        ]
        record["preholdout_metadata_sha256"] = metadata_lock[
            "metadata_sha256"
        ]
        record["holdout_reaction_read_count"] = metadata_lock[
            "holdout_reaction_read_count"
        ]
        record["holdout_status"] = metadata_lock[
            "market_reaction_exposure"
        ]
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["factor_id", "factor_version"], kind="mergesort"
    ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class MetadataAccessRecord:
    sequence: int
    access_kind: str
    source_sha256: str
    previous_chain_sha256: str | None
    chain_sha256: str


@dataclass(frozen=True, slots=True)
class PrelockMetadataLedger:
    """Immutable metadata-only ledger; it has no reaction-read transition."""

    cohort_authority_sha256: str
    metadata_sha256: str
    records: tuple[MetadataAccessRecord, ...]
    holdout_reaction_read_count: int = 0


def _record_hash_payload(
    *,
    sequence: int,
    access_kind: str,
    source_sha256: str,
    previous_chain_sha256: str | None,
    cohort_authority_sha256: str,
    metadata_sha256: str,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "access_kind": access_kind,
        "source_sha256": source_sha256,
        "previous_chain_sha256": previous_chain_sha256,
        "cohort_authority_sha256": cohort_authority_sha256,
        "metadata_sha256": metadata_sha256,
        "holdout_reaction_read_count": 0,
    }


def _verify_metadata_ledger(ledger: PrelockMetadataLedger) -> None:
    if not isinstance(ledger, PrelockMetadataLedger):
        raise KalshiValidationError(
            "ledger must be a PrelockMetadataLedger"
        )
    authority = _require_sha256(
        ledger.cohort_authority_sha256,
        field="ledger.cohort_authority_sha256",
    )
    metadata_hash = _require_sha256(
        ledger.metadata_sha256, field="ledger.metadata_sha256"
    )
    if ledger.holdout_reaction_read_count != 0:
        raise KalshiValidationError(
            "metadata-only ledger reaction counter must remain zero"
        )
    if not ledger.records:
        raise KalshiValidationError(
            "metadata-only ledger must contain genesis"
        )
    previous: str | None = None
    for sequence, record in enumerate(ledger.records):
        if (
            not isinstance(record, MetadataAccessRecord)
            or record.sequence != sequence
            or record.previous_chain_sha256 != previous
        ):
            raise KalshiValidationError(
                "metadata-only ledger sequence/chain is invalid"
            )
        _require_sha256(
            record.source_sha256,
            field="metadata access source_sha256",
        )
        if sequence == 0:
            allowed = record.access_kind == "metadata_ledger_genesis"
        else:
            allowed = record.access_kind in _ALLOWED_METADATA_ACCESS
        if not allowed:
            raise KalshiValidationError(
                "prelock ledger is metadata-only"
            )
        payload = _record_hash_payload(
            sequence=sequence,
            access_kind=record.access_kind,
            source_sha256=record.source_sha256,
            previous_chain_sha256=record.previous_chain_sha256,
            cohort_authority_sha256=authority,
            metadata_sha256=metadata_hash,
        )
        if record.chain_sha256 != _canonical_sha256(payload):
            raise KalshiValidationError(
                "metadata-only ledger hash is invalid"
            )
        previous = record.chain_sha256


def begin_prelock_metadata_ledger(
    authority_metadata: FrozenAuthorityMetadata,
) -> PrelockMetadataLedger:
    """Begin a ledger that can record metadata reads and nothing else."""

    if not isinstance(authority_metadata, FrozenAuthorityMetadata):
        raise KalshiValidationError(
            "authority_metadata must be FrozenAuthorityMetadata"
        )
    payload = _record_hash_payload(
        sequence=0,
        access_kind="metadata_ledger_genesis",
        source_sha256=authority_metadata.metadata_sha256,
        previous_chain_sha256=None,
        cohort_authority_sha256=(
            authority_metadata.cohort_authority_sha256
        ),
        metadata_sha256=authority_metadata.metadata_sha256,
    )
    genesis = MetadataAccessRecord(
        sequence=0,
        access_kind="metadata_ledger_genesis",
        source_sha256=authority_metadata.metadata_sha256,
        previous_chain_sha256=None,
        chain_sha256=_canonical_sha256(payload),
    )
    return PrelockMetadataLedger(
        cohort_authority_sha256=(
            authority_metadata.cohort_authority_sha256
        ),
        metadata_sha256=authority_metadata.metadata_sha256,
        records=(genesis,),
    )


def record_metadata_access(
    ledger: PrelockMetadataLedger,
    *,
    access_kind: str,
    source_sha256: str,
) -> PrelockMetadataLedger:
    """Append one allow-listed metadata access; reaction reads are impossible."""

    _verify_metadata_ledger(ledger)
    if access_kind not in _ALLOWED_METADATA_ACCESS:
        raise KalshiValidationError(
            "prelock ledger is metadata-only; reaction access is forbidden"
        )
    source_hash = _require_sha256(
        source_sha256, field="metadata source_sha256"
    )
    previous = ledger.records[-1].chain_sha256
    sequence = len(ledger.records)
    payload = _record_hash_payload(
        sequence=sequence,
        access_kind=access_kind,
        source_sha256=source_hash,
        previous_chain_sha256=previous,
        cohort_authority_sha256=ledger.cohort_authority_sha256,
        metadata_sha256=ledger.metadata_sha256,
    )
    record = MetadataAccessRecord(
        sequence=sequence,
        access_kind=access_kind,
        source_sha256=source_hash,
        previous_chain_sha256=previous,
        chain_sha256=_canonical_sha256(payload),
    )
    return PrelockMetadataLedger(
        cohort_authority_sha256=ledger.cohort_authority_sha256,
        metadata_sha256=ledger.metadata_sha256,
        records=(*ledger.records, record),
    )


def lock_preholdout_metadata_audit(
    ledger: PrelockMetadataLedger,
    *,
    authority_metadata: FrozenAuthorityMetadata,
) -> dict[str, object]:
    """Issue the metadata proof while the reaction counter remains zero."""

    _verify_metadata_ledger(ledger)
    if not isinstance(authority_metadata, FrozenAuthorityMetadata):
        raise KalshiValidationError(
            "authority_metadata must be FrozenAuthorityMetadata"
        )
    try:
        current_metadata_sha256 = (
            _recompute_authority_metadata_sha256(authority_metadata)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KalshiValidationError(
            "current authority metadata cannot be recomputed"
        ) from exc
    if (
        ledger.cohort_authority_sha256
        != authority_metadata.cohort_authority_sha256
        or ledger.metadata_sha256 != authority_metadata.metadata_sha256
        or current_metadata_sha256 != authority_metadata.metadata_sha256
        or authority_metadata.holdout_reaction_read_count != 0
        or authority_metadata.stage_a_outcome_validation_eligible is not False
        or (
            authority_metadata.stage_b_market_reaction_validation_eligible
            is not True
        )
        or authority_metadata.sports_outcome_source_evidence_sha256
        != X11_SPORTS_OUTCOME_EVIDENCE_SHA256
        or authority_metadata.sports_outcome_observation_count
        != X11_HOLDOUT_DRIVE_OUTCOME_COUNT
        or not authority_metadata.holdout[
            "market_reaction_exposure"
        ].eq("SEALED_UNREAD").all()
        or not authority_metadata.holdout[
            "sports_outcome_exposure"
        ].eq("PRIOR_EXPOSED_X11").all()
    ):
        raise KalshiValidationError(
            "metadata ledger does not bind the current authority metadata"
        )
    latest = ledger.records[-1].chain_sha256
    lock_payload = {
        "lock_event": "PRE_SHORTLIST_METADATA_ONLY_LOCK",
        "cohort_authority_sha256": ledger.cohort_authority_sha256,
        "metadata_sha256": ledger.metadata_sha256,
        "current_metadata_sha256": current_metadata_sha256,
        "development_game_count": len(authority_metadata.development),
        "holdout_game_count": len(authority_metadata.holdout),
        "chain_sha256": latest,
        "metadata_access_count": len(ledger.records) - 1,
        "holdout_reaction_read_count": 0,
        "market_reaction_exposure": "SEALED_UNREAD",
        "sports_outcome_exposure": "PRIOR_EXPOSED_X11",
        "sports_outcome_source_evidence_sha256": (
            authority_metadata.sports_outcome_source_evidence_sha256
        ),
        "sports_outcome_observation_count": (
            authority_metadata.sports_outcome_observation_count
        ),
        "stage_a_outcome_validation_eligible": False,
        "stage_b_market_reaction_validation_eligible": True,
    }
    return {
        **lock_payload,
        "lock_sha256": _canonical_sha256(lock_payload),
    }


__all__ = [
    "DevelopmentVenueValidation",
    "EXPECTED_DEVELOPMENT_GAMES",
    "EXPECTED_HOLDOUT_GAMES",
    "FrozenAuthorityMetadata",
    "FrozenFactorMembershipEvidence",
    "FrozenHistoricalExposureEvidence",
    "KalshiValidationError",
    "MetadataAccessRecord",
    "PrelockMetadataLedger",
    "X11_HOLDOUT_DRIVE_OUTCOME_COUNT",
    "X11_SPORTS_OUTCOME_EVIDENCE_SHA256",
    "begin_prelock_metadata_ledger",
    "bind_frozen_authority_metadata",
    "build_cross_venue_predictive_utility_shortlist",
    "lock_preholdout_metadata_audit",
    "load_frozen_factor_membership_evidence",
    "load_frozen_historical_exposure_evidence",
    "record_metadata_access",
    "validate_development_venue_transport",
]
