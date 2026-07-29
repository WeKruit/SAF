"""Development-only venue validation and metadata-only pre-holdout audit.

There is intentionally no final-holdout prediction or reaction-read API in
this module.  Task 5 may bind the frozen 153/81 cohort metadata, inspect its
prior-exposure declarations, validate genuine Task4 development OOF transport,
and lock a metadata-only ledger.  Reaction data remains unreachable here.
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
    HISTORICAL_ANALYSIS_SCOPE,
    HISTORICAL_CLAIM_BOUNDARY,
    HISTORICAL_SCHEMA_VERSION,
    HISTORICAL_TARGET_CONTRACT,
    MARKET_CONTINUITY_SUPPORT,
    VENUE_TICK_SUPPORT,
)


EXPECTED_DEVELOPMENT_GAMES: Final[int] = 153
EXPECTED_HOLDOUT_GAMES: Final[int] = 81
SOURCE_VENUE: Final[str] = "polymarket"
TARGET_VENUE: Final[str] = "kalshi"
VENUE_SPECIFIC_MODE: Final[str] = "VENUE_SPECIFIC"
TRANSPORT_MODE: Final[str] = "NO_TARGET_RECALIBRATION"

_AUTHORITY_REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "game_id",
        "cohort",
        "nfl_week",
        "kickoff_utc",
        "batch_sha256",
        "cohort_authority_sha256",
        "prior_exposure_status",
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
                "prior_exposure_status": str(
                    row["prior_exposure_status"]
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
    prior_exposure_audit: pd.DataFrame
    holdout_reaction_read_count: int
    metadata_sha256: str


def bind_frozen_authority_metadata(
    cohort_metadata: pd.DataFrame,
    *,
    cohort_authority_sha256: str,
) -> FrozenAuthorityMetadata:
    """Validate and bind exact 153/81 ID, week, kickoff, batch metadata."""

    authority = _require_sha256(
        cohort_authority_sha256, field="cohort_authority_sha256"
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
        "prior_exposure_status",
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
    if not holdout["prior_exposure_status"].eq(
        "PRISTINE_NO_PRIOR_REACTION_READ"
    ).all():
        raise KalshiValidationError(
            "all 81 holdout games must declare pristine prior exposure"
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
    exposure = (
        frame.groupby(
            ["cohort", "prior_exposure_status"],
            sort=True,
            as_index=False,
        )
        .agg(
            game_count=("game_id", "nunique"),
            reaction_read_count=("reaction_read_count", "sum"),
        )
        .reset_index(drop=True)
    )
    metadata_hash = _canonical_sha256(
        _canonical_metadata_records(frame)
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
        prior_exposure_audit=exposure,
        holdout_reaction_read_count=0,
        metadata_sha256=metadata_hash,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if source.empty or target.empty:
        raise KalshiValidationError(
            "Task4 run must contain genuine Polymarket source OOF and "
            "Kalshi development transport OOF"
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
    return source, target


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
    coverage_gate_passed: bool
    coverage_audit: pd.DataFrame
    attrition: pd.DataFrame
    exact_pair_diagnostics: pd.DataFrame


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
    source, target = _validate_task4_transport_run(
        model_run, authority_metadata=authority_metadata
    )
    for frame, label in ((source, "source"), (target, "target")):
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
    coverage_gate = paired_game_count >= 2 and clean_anchor_present
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
            if coverage_gate
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
        coverage_gate_passed=coverage_gate,
        coverage_audit=coverage.sort_values(
            coverage_keys, kind="mergesort"
        ).reset_index(drop=True),
        attrition=attrition,
        exact_pair_diagnostics=diagnostics.sort_values(
            list(_TRANSPORT_PAIR_COLUMNS), kind="mergesort"
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
    "begin_prelock_metadata_ledger",
    "bind_frozen_authority_metadata",
    "lock_preholdout_metadata_audit",
    "record_metadata_access",
    "validate_development_venue_transport",
]
