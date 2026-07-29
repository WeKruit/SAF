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
from typing import Final

import numpy as np
import pandas as pd

from prediction_market.research.nfl_x15_models import X15ModelRun
from prediction_market.research.nfl_x15_model_selection import (
    DIRECTION_THRESHOLD_PROBABILITY,
    DIRECTION_THRESHOLD_SEMANTICS,
    FrozenDevelopmentAuthority,
    HISTORICAL_ANALYSIS_SCOPE,
    HISTORICAL_CLAIM_BOUNDARY,
    HISTORICAL_SCHEMA_VERSION,
    HISTORICAL_TARGET_CONTRACT,
    MARKET_CONTINUITY_SUPPORT,
    VENUE_TICK_SUPPORT,
    bind_frozen_development_authority,
)


EXPECTED_DEVELOPMENT_GAMES: Final[int] = 153
EXPECTED_HOLDOUT_GAMES: Final[int] = 81
SOURCE_VENUE: Final[str] = "polymarket"
TARGET_VENUE: Final[str] = "kalshi"
VENUE_SPECIFIC_MODE: Final[str] = "VENUE_SPECIFIC"
TRANSPORT_MODE: Final[str] = "NO_TARGET_RECALIBRATION"
MIN_TRANSPORT_PAIRED_GAMES: Final[int] = 30
MIN_TRANSPORT_EVALUATED_GAMES_PER_HEAD: Final[int] = 20
MIN_TRANSPORT_EVALUATED_ROWS_PER_HEAD: Final[int] = 20
MAX_LOG_LOSS_DEGRADATION: Final[float] = 0.25
MAX_BRIER_DEGRADATION: Final[float] = 0.10
X11_SPORTS_OUTCOME_EVIDENCE_SHA256: Final[str] = (
    "sha256:1d0c033459c69778e265be3fca16ae2c87f650d5003a61ffdea4c020a4fd0b05"
)
X11_HOLDOUT_DRIVE_OUTCOME_COUNT: Final[int] = 1_683

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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


@dataclass(frozen=True, slots=True)
class FrozenAuthorityMetadata:
    """Exact cohort metadata that can be inspected without reaction rows."""

    cohort_authority_sha256: str
    development: pd.DataFrame
    holdout: pd.DataFrame
    overlap_game_ids: tuple[str, ...]
    holdout_reviewed_overlap_game_ids: tuple[str, ...]
    holdout_reaction_overlap_game_ids: tuple[str, ...]
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
    historical_reviewed_game_ids: tuple[str, ...],
    historical_reviewed_game_ids_sha256: str,
    historical_reaction_game_ids: tuple[str, ...],
    historical_reaction_game_ids_sha256: str,
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
    reviewed_ids = tuple(historical_reviewed_game_ids)
    reaction_ids = tuple(historical_reaction_game_ids)
    sports_outcome_ids = tuple(sports_outcome_exposed_game_ids)
    reaction_blind_ids = tuple(reaction_blind_metadata_game_ids)
    reviewed_hash = _require_sha256(
        historical_reviewed_game_ids_sha256,
        field="historical_reviewed_game_ids_sha256",
    )
    reaction_hash = _require_sha256(
        historical_reaction_game_ids_sha256,
        field="historical_reaction_game_ids_sha256",
    )
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
    if hash_game_id_evidence(reviewed_ids) != reviewed_hash:
        raise KalshiValidationError(
            "historical reviewed game-ID evidence hash mismatch"
        )
    if hash_game_id_evidence(reaction_ids) != reaction_hash:
        raise KalshiValidationError(
            "historical reaction game-ID evidence hash mismatch"
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
            },
            {
                "cohort": "historical",
                "market_reaction_exposure": "REACTION_EXPOSED_ID_SET",
                "sports_outcome_exposure": "NOT_APPLICABLE",
                "game_count": len(reaction_ids),
                "reaction_read_count": len(reaction_ids),
                "evidence_kind": "HISTORICAL_REACTION_GAME_IDS",
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
            "historical_reaction_game_ids": tuple(
                sorted(reaction_ids)
            ),
            "historical_reaction_game_ids_sha256": reaction_hash,
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


def _validate_task4_transport_run(
    model_run: X15ModelRun,
    *,
    authority_metadata: FrozenAuthorityMetadata,
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
    expected_config = {
        "schema_version": HISTORICAL_SCHEMA_VERSION,
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
            "transport requires HistoricalTradesOnlyProbabilityPanelV1"
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
    transport_gate_passed: bool
    coverage_audit: pd.DataFrame
    attrition: pd.DataFrame
    exact_pair_diagnostics: pd.DataFrame
    native_comparison_diagnostics: pd.DataFrame
    score_summary: pd.DataFrame
    calibration_summary: pd.DataFrame


def validate_development_venue_transport(
    model_run: X15ModelRun,
    *,
    authority_metadata: FrozenAuthorityMetadata,
) -> DevelopmentVenueValidation:
    """Validate Task4 Poly-trained -> Kalshi development OOF transport."""

    if not isinstance(authority_metadata, FrozenAuthorityMetadata):
        raise KalshiValidationError(
            "authority_metadata must be FrozenAuthorityMetadata"
        )
    source, target, native_kalshi = _validate_task4_transport_run(
        model_run, authority_metadata=authority_metadata
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
    transport_gate = (
        coverage_gate and score_gate and not catastrophic
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
        diagnostic_status=(
            "HISTORICAL_SIGNAL_CANDIDATE"
            if transport_gate
            else "HISTORICAL_SIGNAL_REJECTED"
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
    if (
        ledger.cohort_authority_sha256
        != authority_metadata.cohort_authority_sha256
        or ledger.metadata_sha256 != authority_metadata.metadata_sha256
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
            "metadata ledger does not bind the frozen authority"
        )
    latest = ledger.records[-1].chain_sha256
    lock_payload = {
        "lock_event": "PRE_SHORTLIST_METADATA_ONLY_LOCK",
        "cohort_authority_sha256": ledger.cohort_authority_sha256,
        "metadata_sha256": ledger.metadata_sha256,
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
    "KalshiValidationError",
    "MetadataAccessRecord",
    "PrelockMetadataLedger",
    "X11_HOLDOUT_DRIVE_OUTCOME_COUNT",
    "X11_SPORTS_OUTCOME_EVIDENCE_SHA256",
    "begin_prelock_metadata_ledger",
    "bind_frozen_authority_metadata",
    "lock_preholdout_metadata_audit",
    "record_metadata_access",
    "validate_development_venue_transport",
]
