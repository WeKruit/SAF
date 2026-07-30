"""Governed magnitude follow-up for the frozen NFL X-15 Stage-B winner.

This module is deliberately downstream of the exact-45 probability batch and
the independently reproducible Stage-B selection authority.  It never resolves
or reads final-holdout reaction data.  Magnitude models are fitted only for D0
and the selected feature block, on the same five folds and venue transport pair
used by the exact-45 computation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.immutable_object_store import sha256_path
from prediction_market.research.nfl_x15_development_panel import (
    VerifiedDiagnosticPanelPartition,
    iter_verified_diagnostic_panel_partitions,
)
from prediction_market.research.nfl_x15_distribution import (
    PRIMARY_QUANTILES,
    magnitude_distribution_metrics,
)
from prediction_market.research.nfl_x15_model_selection import (
    EXPECTED_DEVELOPMENT_GAME_COUNT,
    EXPECTED_FOLDS,
    STAGE_B_CANDIDATE_SUITE,
    FrozenDevelopmentAuthority,
    StageBModelSelectionResult,
    bind_frozen_development_authority,
    select_stage_b_v3_winner,
)
from prediction_market.research.nfl_x15_models import (
    QUANTILE_MODEL_ID,
    X15ModelRun,
    X15PreparedDiagnosticPanel,
    prepare_x15_historical_trades_diagnostic_partitions,
    run_x15_historical_trades_diagnostic_walk_forward,
)
from prediction_market.research.nfl_x15_selection_batch_v3 import (
    X15SelectionBatchV3Error,
    load_verified_x15_selection_batch_v3,
    load_x15_selection_projection_v3,
)


SCHEMA: Final[str] = "nfl_x15_magnitude_followup_v1"
BUILDER_VERSION: Final[str] = "nfl-x15-magnitude-followup-v1"
SELECTION_SCHEMA: Final[str] = (
    "nfl_x15_stage_b_v3_selection_manifest_v1"
)
COMPLETE_STATUS: Final[str] = "MAGNITUDE_DIAGNOSTIC_COMPLETE"
NOT_APPLICABLE: Final[str] = "NOT_APPLICABLE"
BOOTSTRAP_SAMPLES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 20260729
CONFIDENCE_LEVEL: Final[float] = 0.95
ANCHOR_LANDMARK_SECONDS: Final[int] = 3
ANCHOR_ENDPOINT_SECONDS: Final[int] = 30
SOURCE_TRANSPORT_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("polymarket", "kalshi"),
)
EXPECTED_FOLD_IDS: Final[tuple[str, ...]] = tuple(
    fold_id for fold_id, _, _ in EXPECTED_FOLDS
)
QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
)
CLAIM_BOUNDARY: Final[str] = (
    "HISTORICAL_TRADES_ONLY_SOURCE_TIME_CONDITIONAL_MAGNITUDE_"
    "DIAGNOSTIC_NOT_EXECUTION_NOT_ALPHA"
)
_SHA_PREFIX: Final[str] = "sha256:"


class MagnitudeFollowupError(RuntimeError):
    """A governed magnitude input, computation, or artifact failed closed."""


@dataclass(frozen=True, slots=True)
class VerifiedMagnitudeInputs:
    """Exact probability and selection authorities reverified for follow-up."""

    exact45_batch_document: Mapping[str, object]
    exact45_batch_path: Path
    exact45_batch_file_sha256: str
    authority: FrozenDevelopmentAuthority
    selection_document: Mapping[str, object]
    selection_path: Path
    selection_file_sha256: str
    decision_status: str
    winner_model_id: str | None
    winner_feature_block_id: str | None
    source_transport_pairs: tuple[tuple[str, str], ...]


class MagnitudeFollowupPublication(NamedTuple):
    manifest_path: Path
    manifest_sha256: str
    status: str
    winner_feature_block_id: str | None
    holdout_reaction_accessed: bool


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
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return _SHA_PREFIX + hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_SHA_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise MagnitudeFollowupError(f"{field} must be a SHA-256 digest")
    return value


def _strict_json(payload: bytes, *, field: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise MagnitudeFollowupError(
                    f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise MagnitudeFollowupError(f"{field} is not strict JSON") from error
    if not isinstance(value, dict):
        raise MagnitudeFollowupError(f"{field} must be a JSON object")
    return value


def _resolve_under(
    root: Path, value: str | Path, *, field: str
) -> Path:
    base = root.resolve()
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise MagnitudeFollowupError(
            f"{field} escapes the artifact root"
        ) from error
    if not resolved.is_file():
        raise MagnitudeFollowupError(f"{field} does not exist")
    return resolved


def _verify_content_addressed_json(
    path: Path,
    *,
    expected_file_sha256: str,
    suffix: str,
    field: str,
) -> dict[str, object]:
    expected = _require_sha256(expected_file_sha256, field=f"{field} SHA")
    observed, _ = sha256_path(path)
    if observed != expected:
        raise MagnitudeFollowupError(f"{field} SHA-256 mismatch")
    digest = expected.removeprefix(_SHA_PREFIX)
    if path.name != f"{digest}{suffix}":
        raise MagnitudeFollowupError(
            f"{field} path is not content-addressed by the verified bytes"
        )
    return _strict_json(path.read_bytes(), field=field)


def _winner_document(
    selection: StageBModelSelectionResult,
) -> dict[str, object] | None:
    winner = selection.winner
    if winner is None:
        return None
    return {
        "model_id": winner.spec.candidate_model_id,
        "feature_block_id": winner.spec.candidate_feature_block_id,
        "joint_integrated_mean_improvement": (
            winner.integrated_mean_improvement
        ),
        "joint_integrated_ci": (
            winner.integrated_ci_low,
            winner.integrated_ci_high,
        ),
        "direction_log_loss_integrated_mean_improvement": (
            winner.direction_log_loss_integrated_mean_improvement
        ),
        "direction_log_loss_integrated_ci": (
            winner.direction_log_loss_integrated_ci_low,
            winner.direction_log_loss_integrated_ci_high,
        ),
        "direction_brier_integrated_mean_improvement": (
            winner.direction_brier_integrated_mean_improvement
        ),
        "direction_brier_integrated_ci": (
            winner.direction_brier_integrated_ci_low,
            winner.direction_brier_integrated_ci_high,
        ),
        "joint_anchor_game_count": winner.anchor_game_count,
        "joint_anchor_mean_improvement": winner.anchor_mean_improvement,
        "direction_anchor_game_count": (
            winner.direction_anchor_game_count
        ),
        "direction_anchor_log_loss_mean_improvement": (
            winner.direction_anchor_log_loss_mean_improvement
        ),
        "direction_anchor_brier_mean_improvement": (
            winner.direction_anchor_brier_mean_improvement
        ),
    }


def _selection_expected_fields(
    selection: StageBModelSelectionResult,
    runs: Sequence[X15ModelRun],
    *,
    batch_document: Mapping[str, object],
    batch_path: Path,
    batch_file_sha256: str,
    authority: FrozenDevelopmentAuthority,
) -> dict[str, object]:
    return {
        "schema": SELECTION_SCHEMA,
        "experiment_id": "X-15",
        "cohort": "development",
        "selection_venue": "polymarket",
        "input_selection_batch": {
            "manifest_path": str(batch_path),
            "manifest_file_sha256": batch_file_sha256,
            "batch_sha256": batch_document["batch_sha256"],
        },
        "candidate_suite": STAGE_B_CANDIDATE_SUITE,
        "candidate_run_config_sha256s": tuple(
            {
                "model_id": model_id,
                "feature_block_id": feature_block_id,
                "run_config_sha256": run.run_config_sha256,
            }
            for (model_id, feature_block_id), run in zip(
                STAGE_B_CANDIDATE_SUITE, runs, strict=True
            )
        ),
        "decision_status": selection.decision_status,
        "winner": _winner_document(selection),
        "best_integrated_mean_improvement": (
            selection.best_integrated_mean_improvement
        ),
        "best_standard_error": selection.best_standard_error,
        "one_se_threshold": selection.one_se_threshold,
        "cohort_authority_sha256": selection.cohort_authority_sha256,
        "authority_metadata_sha256": authority.metadata_sha256,
        "shared_run_config_sha256": selection.shared_run_config_sha256,
        "suite_run_config_sha256": selection.suite_run_config_sha256,
        "candidate_audit_sha256": selection.candidate_audit_sha256,
        "winner_rule_sha256": selection.winner_rule_sha256,
        "selection_contract_version": (
            selection.selection_contract_version
        ),
        "schema_version": selection.schema_version,
        "survival_probability_contract": (
            selection.survival_probability_contract
        ),
        "holdout_reaction_accessed": False,
    }


def _selection_output_root(batch_path: Path) -> Path:
    for parent in batch_path.parents:
        if parent.name == "selection-v3":
            return parent.parent
    raise MagnitudeFollowupError(
        "exact45 batch is outside the governed selection-v3 namespace"
    )


def _source_transport_pairs(
    batch_document: Mapping[str, object],
    *,
    batch_path: Path,
) -> tuple[tuple[str, str], ...]:
    references = batch_document.get("shards")
    if not isinstance(references, list) or len(references) != 45:
        raise MagnitudeFollowupError(
            "exact45 batch has no canonical 45-shard source"
        )
    root = _selection_output_root(batch_path)
    observed: set[tuple[tuple[str, str], ...]] = set()
    for reference in references:
        if not isinstance(reference, dict):
            raise MagnitudeFollowupError(
                "exact45 source shard reference is malformed"
            )
        path = _resolve_under(
            root,
            str(reference.get("manifest_path", "")),
            field="exact45 source shard manifest",
        )
        document = _strict_json(
            path.read_bytes(), field="exact45 source shard manifest"
        )
        config = document.get("run_config")
        if not isinstance(config, dict):
            raise MagnitudeFollowupError(
                "exact45 source shard lacks run_config"
            )
        pairs = config.get("transport_pairs")
        if not isinstance(pairs, list):
            raise MagnitudeFollowupError(
                "exact45 source shard transport_pairs are malformed"
            )
        normalized: list[tuple[str, str]] = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise MagnitudeFollowupError(
                    "exact45 source shard has malformed transport pair"
                )
            normalized.append((str(pair[0]), str(pair[1])))
        observed.add(tuple(normalized))
    if observed != {SOURCE_TRANSPORT_PAIRS}:
        raise MagnitudeFollowupError(
            "exact45 shards do not share the frozen Polymarket-to-Kalshi "
            "transport pair"
        )
    return SOURCE_TRANSPORT_PAIRS


def verify_magnitude_followup_inputs(
    *,
    exact45_batch_manifest_path: str | Path,
    exact45_batch_manifest_file_sha256: str,
    selection_manifest_path: str | Path,
    selection_manifest_file_sha256: str,
    artifact_root: str | Path,
) -> VerifiedMagnitudeInputs:
    """Re-derive the selection decision from a fully verified exact45 batch."""

    root = Path(artifact_root).resolve()
    expected_batch_sha = _require_sha256(
        exact45_batch_manifest_file_sha256,
        field="exact45 batch manifest file SHA",
    )
    try:
        (
            batch_document,
            batch_path,
            observed_batch_sha,
            authority,
        ) = load_verified_x15_selection_batch_v3(
            batch_manifest_path=exact45_batch_manifest_path,
            artifact_root=root,
        )
    except X15SelectionBatchV3Error as error:
        raise MagnitudeFollowupError(str(error)) from error
    if observed_batch_sha != expected_batch_sha:
        raise MagnitudeFollowupError(
            "exact45 batch bytes differ from the requested authority"
        )
    if batch_document.get("holdout_reaction_accessed") is not False:
        raise MagnitudeFollowupError(
            "exact45 magnitude source must remain holdout blind"
        )
    selection_path = _resolve_under(
        root, selection_manifest_path, field="selection manifest"
    )
    selection_file_sha = _require_sha256(
        selection_manifest_file_sha256,
        field="selection manifest file SHA",
    )
    selection_document = _verify_content_addressed_json(
        selection_path,
        expected_file_sha256=selection_file_sha,
        suffix=".stage-b-v3-selection.json",
        field="selection manifest",
    )
    if selection_document.get("holdout_reaction_accessed") is not False:
        raise MagnitudeFollowupError(
            "magnitude selection authority must remain holdout blind"
        )

    runs: list[X15ModelRun] = []
    for model_id, feature_block_id in STAGE_B_CANDIDATE_SUITE:
        try:
            runs.append(
                load_x15_selection_projection_v3(
                    batch_manifest_path=batch_path,
                    artifact_root=root,
                    candidate_model_id=model_id,
                    candidate_feature_block_id=feature_block_id,
                )
            )
        except X15SelectionBatchV3Error as error:
            raise MagnitudeFollowupError(str(error)) from error
    selection = select_stage_b_v3_winner(tuple(runs), authority=authority)
    expected = _selection_expected_fields(
        selection,
        runs,
        batch_document=batch_document,
        batch_path=batch_path,
        batch_file_sha256=expected_batch_sha,
        authority=authority,
    )
    if _canonical(selection_document) != _canonical(expected):
        raise MagnitudeFollowupError(
            "selection manifest differs from independently rederived "
            "exact45 authority"
        )
    winner = selection.winner
    return VerifiedMagnitudeInputs(
        exact45_batch_document=batch_document,
        exact45_batch_path=batch_path,
        exact45_batch_file_sha256=expected_batch_sha,
        authority=authority,
        selection_document=selection_document,
        selection_path=selection_path,
        selection_file_sha256=selection_file_sha,
        decision_status=selection.decision_status,
        winner_model_id=(
            winner.spec.candidate_model_id if winner is not None else None
        ),
        winner_feature_block_id=(
            winner.spec.candidate_feature_block_id
            if winner is not None
            else None
        ),
        source_transport_pairs=_source_transport_pairs(
            batch_document, batch_path=batch_path
        ),
    )


def _explicit_utc(value: pd.Timestamp) -> str:
    if value.tzinfo is None:
        raise MagnitudeFollowupError(
            "panel authority source time must be timezone-aware"
        )
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _prepare_verified_panel(
    *,
    verified: VerifiedMagnitudeInputs,
    artifact_root: Path,
) -> X15PreparedDiagnosticPanel:
    binding = verified.exact45_batch_document.get("panel_batch")
    if not isinstance(binding, dict):
        raise MagnitudeFollowupError(
            "exact45 batch panel binding is malformed"
        )
    authority_rows: list[dict[str, object]] = []

    def observe() -> Iterator[VerifiedDiagnosticPanelPartition]:
        # This is consumed as an iterator by the existing bounded preparer.
        for partition in iter_verified_diagnostic_panel_partitions(
            project_root=artifact_root,
            batch_manifest_path=str(binding.get("manifest_path", "")),
            batch_manifest_file_sha256=str(
                binding.get("file_sha256", "")
            ),
            expected_game_count=EXPECTED_DEVELOPMENT_GAME_COUNT,
        ):
            panel = partition.panel
            weeks = tuple(
                sorted(
                    set(
                        pd.to_numeric(
                            panel["nfl_week"], errors="raise"
                        ).astype(int)
                    )
                )
            )
            source_times = pd.to_datetime(
                panel["source_interval_start"],
                utc=True,
                errors="coerce",
            ).dropna()
            if len(weeks) != 1 or source_times.empty:
                raise MagnitudeFollowupError(
                    f"{partition.game_id} lacks governed week/source time"
                )
            authority_rows.append(
                {
                    "game_id": partition.game_id,
                    "nfl_week": weeks[0],
                    "kickoff_utc": _explicit_utc(source_times.min()),
                    "batch_sha256": partition.game_manifest_sha256,
                    "cohort_authority_sha256": (
                        partition.cohort_authority_sha256
                    ),
                }
            )
            yield partition

    prepared = prepare_x15_historical_trades_diagnostic_partitions(
        observe()
    )
    observed_authority = bind_frozen_development_authority(
        pd.DataFrame(authority_rows),
        cohort_authority_sha256=prepared.cohort_authority_sha256,
    )
    if observed_authority != verified.authority:
        raise MagnitudeFollowupError(
            "prepared panel authority differs from exact45 authority"
        )
    if (
        prepared.batch_manifest_file_sha256
        != binding.get("file_sha256")
        or prepared.batch_sha256 != binding.get("batch_sha256")
    ):
        raise MagnitudeFollowupError(
            "prepared panel lineage differs from exact45 panel binding"
        )
    return prepared


def validate_magnitude_cell(
    run: X15ModelRun,
    *,
    fold_id: str,
    feature_block_id: str,
    cohort_authority_sha256: str,
    source_transport_pairs: tuple[tuple[str, str], ...],
) -> None:
    """Fail closed if a cell drifts from the exact probability design."""

    if not isinstance(run, X15ModelRun) or not isinstance(
        run.run_config, Mapping
    ):
        raise MagnitudeFollowupError(
            "magnitude cell must be a governed X15ModelRun"
        )
    config = run.run_config
    if tuple(config.get("fold_ids", ())) != (fold_id,):
        raise MagnitudeFollowupError("magnitude cell fold differs")
    if tuple(config.get("feature_block_ids", ())) != (feature_block_id,):
        raise MagnitudeFollowupError(
            "magnitude cell feature block differs"
        )
    observed_pairs = tuple(
        tuple(map(str, pair))
        for pair in config.get("transport_pairs", ())
    )
    if observed_pairs != source_transport_pairs:
        raise MagnitudeFollowupError(
            "magnitude cell transport pair differs from exact45"
        )
    if config.get("include_magnitude") is not True:
        raise MagnitudeFollowupError(
            "magnitude cell did not enable governed quantile fitting"
        )
    if (
        str(config.get("cohort_authority_sha256"))
        != cohort_authority_sha256
    ):
        raise MagnitudeFollowupError(
            "magnitude cell cohort authority differs"
        )
    if config.get("holdout_reaction_accessed") is True:
        raise MagnitudeFollowupError(
            "magnitude cell accessed holdout reaction data"
        )
    frame = run.conditional_quantiles
    required = {
        "fold_id",
        "feature_block_id",
        "model_id",
        "direction_condition",
        "direction_truth",
        "conditional_magnitude_truth",
        "support_status",
        "magnitude_weight",
        "cohort_authority_sha256",
        "landmark_seconds",
        "endpoint_seconds",
        *QUANTILE_COLUMNS,
    }
    if frame.empty or not required.issubset(frame.columns):
        raise MagnitudeFollowupError(
            "magnitude cell lacks conditional quantile rows"
        )
    if (
        not frame["fold_id"].eq(fold_id).all()
        or not frame["feature_block_id"].eq(feature_block_id).all()
        or not frame["model_id"].eq(QUANTILE_MODEL_ID).all()
        or not frame["cohort_authority_sha256"].eq(
            cohort_authority_sha256
        ).all()
    ):
        raise MagnitudeFollowupError(
            "magnitude cell row identity differs from requested cell"
        )


def _bootstrap_mean(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    if len(values) < 2:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(bootstrap_seed)
    draws = values[
        rng.integers(
            0, len(values), size=(bootstrap_samples, len(values))
        )
    ].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high), float(np.mean(draws > 0.0))


def _truth_conditioned_rows(quantiles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "fold_id",
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        "feature_block_id",
        "direction_condition",
        "direction_truth",
        "conditional_magnitude_truth",
        "magnitude_weight",
        "support_status",
        "landmark_seconds",
        "endpoint_seconds",
        *QUANTILE_COLUMNS,
    }
    if not isinstance(quantiles, pd.DataFrame) or not required.issubset(
        quantiles.columns
    ):
        raise MagnitudeFollowupError(
            "conditional quantiles lack governed metric columns"
        )
    rows = quantiles.loc[
        quantiles["conditional_magnitude_truth"].notna()
        & quantiles["direction_condition"].eq(
            quantiles["direction_truth"]
        )
    ].copy()
    if rows.empty:
        raise MagnitudeFollowupError(
            "magnitude run has no truth-conditioned UP/DOWN rows"
        )
    truth = pd.to_numeric(
        rows["conditional_magnitude_truth"], errors="coerce"
    )
    weights = pd.to_numeric(rows["magnitude_weight"], errors="coerce")
    quantile_values = rows.loc[:, QUANTILE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    eligible = (
        rows["support_status"].eq("SUPPORTED")
        & truth.notna()
        & truth.ge(0)
        & weights.notna()
        & weights.gt(0)
        & quantile_values.notna().all(axis=1)
    )
    if eligible.any():
        values = quantile_values.loc[eligible].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(np.diff(values, axis=1) < -1e-12)
        ):
            raise MagnitudeFollowupError(
                "supported magnitude quantiles must be finite, "
                "nonnegative, and noncrossing"
            )
    rows["metric_eligible"] = eligible
    rows["metric_exclusion_reason"] = np.where(
        eligible,
        "ELIGIBLE",
        np.where(
            ~rows["support_status"].eq("SUPPORTED"),
            "UNSUPPORTED_DIRECTION_FIT",
            "MISSING_OR_INVALID_TRUTH_WEIGHT_QUANTILE",
        ),
    )
    grain = [
        "fold_id",
        "venue",
        "training_venue",
        "calibration_venue",
        "transport_mode",
        "feature_block_id",
        "game_id",
        "atomic_information_episode_id",
        "landmark_seconds",
        "endpoint_seconds",
        "direction_truth",
    ]
    available_grain = [column for column in grain if column in rows]
    if rows.duplicated(available_grain, keep=False).any():
        raise MagnitudeFollowupError(
            "truth-conditioned magnitude rows duplicate governed grain"
        )
    return rows.sort_values(
        available_grain, kind="mergesort"
    ).reset_index(drop=True)


_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "venue",
    "training_venue",
    "calibration_venue",
    "transport_mode",
    "feature_block_id",
    "direction_truth",
)
_GAME_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "fold_id",
    *_GROUP_COLUMNS,
)


def _game_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.Series]] = [
        ("ALL_HORIZONS", pd.Series(True, index=rows.index))
    ]
    pairs = (
        rows.loc[:, ["landmark_seconds", "endpoint_seconds"]]
        .drop_duplicates()
        .sort_values(
            ["landmark_seconds", "endpoint_seconds"], kind="mergesort"
        )
    )
    for pair in pairs.itertuples(index=False):
        landmark = int(pair.landmark_seconds)
        endpoint = int(pair.endpoint_seconds)
        scope = (
            "ANCHOR_L3_H30"
            if (
                landmark == ANCHOR_LANDMARK_SECONDS
                and endpoint == ANCHOR_ENDPOINT_SECONDS
            )
            else f"PAIR_L{landmark}_H{endpoint}"
        )
        scopes.append(
            (
                scope,
                rows["landmark_seconds"].eq(landmark)
                & rows["endpoint_seconds"].eq(endpoint),
            )
        )
    for scope, scope_mask in scopes:
        eligible = rows.loc[scope_mask & rows["metric_eligible"]]
        for key, group in eligible.groupby(
            [*_GAME_GROUP_COLUMNS, "game_id"], sort=True
        ):
            metrics = magnitude_distribution_metrics(
                group["conditional_magnitude_truth"].to_numpy(dtype=float),
                group.loc[:, QUANTILE_COLUMNS],
                sample_weight=group["magnitude_weight"].to_numpy(
                    dtype=float
                ),
            )
            base = {
                **dict(
                    zip([*_GAME_GROUP_COLUMNS, "game_id"], key, strict=True)
                ),
                "scope": scope,
                "row_count": len(group),
            }
            output.extend(
                {
                    **base,
                    "metric_name": metric_name,
                    "metric_value": value,
                }
                for metric_name, value in metrics.items()
            )
    columns = [
        *_GAME_GROUP_COLUMNS,
        "game_id",
        "scope",
        "row_count",
        "metric_name",
        "metric_value",
    ]
    result = pd.DataFrame(output, columns=columns)
    return result.sort_values(
        [*_GAME_GROUP_COLUMNS, "scope", "metric_name", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _summary_metric_rows(
    games: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    keys = [*_GROUP_COLUMNS, "scope", "metric_name"]
    for offset, (key, group) in enumerate(games.groupby(keys, sort=True)):
        values = group["metric_value"].to_numpy(dtype=float)
        low, high, probability_positive = _bootstrap_mean(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
        output.append(
            {
                **dict(zip(keys, key, strict=True)),
                "metric_value": float(values.mean()),
                "game_count": group["game_id"].nunique(),
                "row_count": int(group["row_count"].sum()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_probability_positive": probability_positive,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed + offset,
                "bootstrap_cluster": "game_id",
                "support_status": (
                    "SUPPORTED"
                    if group["game_id"].nunique() >= 2
                    else "INSUFFICIENT_GAME_CLUSTERS"
                ),
            }
        )
    return pd.DataFrame(output).sort_values(
        keys, kind="mergesort"
    ).reset_index(drop=True)


def build_magnitude_metric_tables(
    conditional_quantiles: pd.DataFrame,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build truth-conditioned q10--q90, per-game, and cluster summaries."""

    if (
        type(bootstrap_samples) is not int
        or bootstrap_samples < 1
        or type(bootstrap_seed) is not int
    ):
        raise MagnitudeFollowupError(
            "bootstrap samples/seed must be fixed integers"
        )
    rows = _truth_conditioned_rows(conditional_quantiles)
    games = _game_metric_rows(rows)
    summaries = _summary_metric_rows(
        games,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return rows, games, summaries


def compare_winner_to_d0(
    game_metrics: pd.DataFrame,
    *,
    winner_feature_block_id: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Paired game-cluster comparison; positive means winner-block is better."""

    if winner_feature_block_id not in {"D1", "D2", "D3"}:
        raise MagnitudeFollowupError(
            "magnitude winner feature block must be D1..D3"
        )
    required = {
        *_GAME_GROUP_COLUMNS,
        "game_id",
        "scope",
        "metric_name",
        "metric_value",
    }
    if not isinstance(game_metrics, pd.DataFrame) or not required.issubset(
        game_metrics.columns
    ):
        raise MagnitudeFollowupError(
            "game metrics lack governed comparison columns"
        )
    frame = game_metrics.loc[
        game_metrics["feature_block_id"].isin(
            ("D0", winner_feature_block_id)
        )
        & game_metrics["metric_name"].isin(
            (
                "pinball_q10",
                "pinball_q25",
                "pinball_q50",
                "pinball_q75",
                "pinball_q90",
                "approximate_crps",
                "interval_80_coverage",
            )
        )
    ].copy()
    identity = [
        column
        for column in _GAME_GROUP_COLUMNS
        if column != "feature_block_id"
    ] + ["scope", "metric_name", "game_id"]
    baseline = frame.loc[
        frame["feature_block_id"].eq("D0"), identity + ["metric_value"]
    ].rename(columns={"metric_value": "d0_metric_value"})
    candidate = frame.loc[
        frame["feature_block_id"].eq(winner_feature_block_id),
        identity + ["metric_value"],
    ].rename(columns={"metric_value": "winner_metric_value"})
    paired = baseline.merge(
        candidate,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise MagnitudeFollowupError(
            "winner and D0 have no paired game magnitude metrics"
        )
    coverage = paired["metric_name"].eq("interval_80_coverage")
    paired["improvement"] = (
        paired["d0_metric_value"] - paired["winner_metric_value"]
    )
    paired.loc[coverage, "improvement"] = (
        (paired.loc[coverage, "d0_metric_value"] - 0.80).abs()
        - (paired.loc[coverage, "winner_metric_value"] - 0.80).abs()
    )
    output: list[dict[str, object]] = []
    group_keys = [
        column
        for column in identity
        if column not in {"fold_id", "game_id"}
    ]
    for offset, (key, group) in enumerate(
        paired.groupby(group_keys, sort=True)
    ):
        values = group["improvement"].to_numpy(dtype=float)
        low, high, probability_positive = _bootstrap_mean(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
        output.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "baseline_feature_block_id": "D0",
                "winner_feature_block_id": winner_feature_block_id,
                "mean_improvement": float(values.mean()),
                "paired_game_count": group["game_id"].nunique(),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_probability_improved": probability_positive,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed + offset,
                "bootstrap_cluster": "game_id",
                "improvement_semantics": (
                    "ABS_COVERAGE_ERROR_D0_MINUS_WINNER"
                    if key[group_keys.index("metric_name")]
                    == "interval_80_coverage"
                    else "LOSS_D0_MINUS_WINNER"
                ),
                "support_status": (
                    "SUPPORTED"
                    if group["game_id"].nunique() >= 2
                    else "INSUFFICIENT_GAME_CLUSTERS"
                ),
            }
        )
    return pd.DataFrame(output).sort_values(
        group_keys, kind="mergesort"
    ).reset_index(drop=True)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise MagnitudeFollowupError(
                f"content-addressed collision: {path}"
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publication_material(
    *,
    verified: VerifiedMagnitudeInputs,
    status: str,
    reason: str | None,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-15",
        "cohort": "development",
        "status": status,
        "reason": reason,
        "input_exact45": {
            "manifest_path": str(verified.exact45_batch_path),
            "manifest_file_sha256": (
                verified.exact45_batch_file_sha256
            ),
            "batch_sha256": verified.exact45_batch_document[
                "batch_sha256"
            ],
            "selection_contract_sha256": (
                verified.exact45_batch_document[
                    "selection_contract_sha256"
                ]
            ),
        },
        "input_selection": {
            "manifest_path": str(verified.selection_path),
            "manifest_file_sha256": verified.selection_file_sha256,
            "decision_status": verified.decision_status,
        },
        "cohort_authority_sha256": (
            verified.authority.cohort_authority_sha256
        ),
        "authority_metadata_sha256": verified.authority.metadata_sha256,
        "authority_game_count": verified.authority.game_count,
        "winner_model_id": verified.winner_model_id,
        "winner_feature_block_id": verified.winner_feature_block_id,
        "feature_blocks": (
            ("D0", verified.winner_feature_block_id)
            if verified.winner_feature_block_id is not None
            else ()
        ),
        "fold_ids": EXPECTED_FOLD_IDS,
        "source_transport_pairs": verified.source_transport_pairs,
        "quantiles": PRIMARY_QUANTILES,
        "anchor": {
            "landmark_seconds": ANCHOR_LANDMARK_SECONDS,
            "endpoint_seconds": ANCHOR_ENDPOINT_SECONDS,
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "game_id",
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "holdout_reaction_accessed": False,
    }


def _publish_manifest(
    *,
    output_root: Path,
    material: Mapping[str, object],
) -> tuple[Path, str]:
    document = dict(material)
    document["bundle_sha256"] = _canonical_sha256(material)
    payload = _canonical_bytes(document)
    file_sha = _sha256_bytes(payload)
    digest = file_sha.removeprefix(_SHA_PREFIX)
    path = (
        output_root.resolve()
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.magnitude-followup.json"
    )
    _atomic_write(path, payload)
    return path, file_sha


def publish_not_applicable(
    *,
    verified: VerifiedMagnitudeInputs,
    output_root: str | Path,
) -> MagnitudeFollowupPublication:
    """Publish an explicit terminal NOT_APPLICABLE result without fitting."""

    if (
        verified.decision_status != "NO_MODEL_ADVANCE"
        or verified.winner_feature_block_id is not None
        or verified.winner_model_id is not None
    ):
        raise MagnitudeFollowupError(
            "NOT_APPLICABLE requires verified NO_MODEL_ADVANCE"
        )
    material = _publication_material(
        verified=verified,
        status=NOT_APPLICABLE,
        reason="NO_MODEL_ADVANCE",
    )
    path, digest = _publish_manifest(
        output_root=Path(output_root), material=material
    )
    return MagnitudeFollowupPublication(
        manifest_path=path,
        manifest_sha256=digest,
        status=NOT_APPLICABLE,
        winner_feature_block_id=None,
        holdout_reaction_accessed=False,
    )


def _table_descriptor(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    name: str,
) -> dict[str, object]:
    root = output_root.resolve()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    temporary_dir = root / ".tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=temporary_dir, prefix=f".{name}.", suffix=".parquet"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        object_sha, byte_length = sha256_path(temporary)
        digest = object_sha.removeprefix(_SHA_PREFIX)
        destination = (
            root
            / "objects"
            / "sha256"
            / digest[:2]
            / f"{digest}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            observed, _ = sha256_path(destination)
            if observed != object_sha:
                raise MagnitudeFollowupError(
                    f"content-addressed table collision: {destination}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        semantic_sha = _canonical_sha256(table.to_pylist())
        return {
            "name": name,
            "format": "parquet",
            "object_path": destination.relative_to(root).as_posix(),
            "object_sha256": object_sha,
            "byte_length": byte_length,
            "row_count": table.num_rows,
            "schema_columns": table.schema.names,
            "schema_fingerprint": _sha256_bytes(
                table.schema.serialize().to_pybytes()
            ),
            "semantic_rows_sha256": semantic_sha,
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_cell(
    prepared: X15PreparedDiagnosticPanel,
    *,
    fold_id: str,
    feature_block_id: str,
    verified: VerifiedMagnitudeInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = run_x15_historical_trades_diagnostic_walk_forward(
        prepared,
        model_ids=("b0_empirical_v1",),
        feature_block_ids=(feature_block_id,),
        fold_ids=(fold_id,),
        transport_pairs=verified.source_transport_pairs,
        include_magnitude=True,
    )
    validate_magnitude_cell(
        run,
        fold_id=fold_id,
        feature_block_id=feature_block_id,
        cohort_authority_sha256=(
            verified.authority.cohort_authority_sha256
        ),
        source_transport_pairs=verified.source_transport_pairs,
    )
    rows, games, summaries = build_magnitude_metric_tables(
        run.conditional_quantiles
    )
    support = run.support_audit.loc[
        run.support_audit["model_id"].eq(QUANTILE_MODEL_ID)
    ].copy()
    if support.empty:
        raise MagnitudeFollowupError(
            "magnitude cell lacks quantile support audit"
        )
    lineage = {
        "exact45_batch_manifest_file_sha256": (
            verified.exact45_batch_file_sha256
        ),
        "selection_manifest_file_sha256": (
            verified.selection_file_sha256
        ),
        "authority_metadata_sha256": verified.authority.metadata_sha256,
        "magnitude_run_config_sha256": run.run_config_sha256,
        "holdout_reaction_accessed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for frame in (rows, games, summaries, support):
        for column, value in lineage.items():
            frame[column] = value
    return rows, games, summaries, support


def run_magnitude_followup(
    *,
    exact45_batch_manifest_path: str | Path,
    exact45_batch_manifest_file_sha256: str,
    selection_manifest_path: str | Path,
    selection_manifest_file_sha256: str,
    artifact_root: str | Path,
    output_root: str | Path,
) -> MagnitudeFollowupPublication:
    """Verify, run D0/winner magnitude OOF cells, and publish one bundle."""

    artifacts = Path(artifact_root).resolve()
    output = Path(output_root).resolve()
    verified = verify_magnitude_followup_inputs(
        exact45_batch_manifest_path=exact45_batch_manifest_path,
        exact45_batch_manifest_file_sha256=(
            exact45_batch_manifest_file_sha256
        ),
        selection_manifest_path=selection_manifest_path,
        selection_manifest_file_sha256=selection_manifest_file_sha256,
        artifact_root=artifacts,
    )
    if verified.decision_status == "NO_MODEL_ADVANCE":
        return publish_not_applicable(
            verified=verified, output_root=output
        )
    if (
        verified.decision_status != "MODEL_ADVANCE"
        or verified.winner_feature_block_id not in {"D1", "D2", "D3"}
    ):
        raise MagnitudeFollowupError(
            "selection authority has an unknown terminal decision"
        )
    prepared = _prepare_verified_panel(
        verified=verified, artifact_root=artifacts
    )
    all_games: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_support: list[pd.DataFrame] = []
    cell_descriptors: list[dict[str, object]] = []
    for fold_id in EXPECTED_FOLD_IDS:
        for feature_block_id in (
            "D0",
            verified.winner_feature_block_id,
        ):
            rows, games, summaries, support = _run_cell(
                prepared,
                fold_id=fold_id,
                feature_block_id=feature_block_id,
                verified=verified,
            )
            descriptors = tuple(
                _table_descriptor(frame, output_root=output, name=name)
                for name, frame in (
                    ("conditional_quantiles", rows),
                    ("game_metrics", games),
                    ("summary_metrics", summaries),
                    ("support_audit", support),
                )
            )
            cell_descriptors.append(
                {
                    "fold_id": fold_id,
                    "feature_block_id": feature_block_id,
                    "tables": descriptors,
                    "cell_sha256": _canonical_sha256(
                        {
                            "fold_id": fold_id,
                            "feature_block_id": feature_block_id,
                            "tables": descriptors,
                        }
                    ),
                }
            )
            all_games.append(games)
            all_summaries.append(summaries)
            all_support.append(support)
            del rows, games, summaries, support
            gc.collect()
    game_metrics = pd.concat(all_games, ignore_index=True)
    summary_metrics = pd.concat(all_summaries, ignore_index=True)
    support_audit = pd.concat(all_support, ignore_index=True)
    comparison = compare_winner_to_d0(
        game_metrics,
        winner_feature_block_id=verified.winner_feature_block_id,
    )
    cross_descriptors = tuple(
        _table_descriptor(frame, output_root=output, name=name)
        for name, frame in (
            ("game_metrics", game_metrics),
            ("summary_metrics", summary_metrics),
            ("winner_vs_d0", comparison),
            ("support_audit", support_audit),
        )
    )
    material = {
        **_publication_material(
            verified=verified,
            status=COMPLETE_STATUS,
            reason=None,
        ),
        "cell_count": len(cell_descriptors),
        "cells": cell_descriptors,
        "cross_game_tables": cross_descriptors,
    }
    path, digest = _publish_manifest(
        output_root=output, material=material
    )
    return MagnitudeFollowupPublication(
        manifest_path=path,
        manifest_sha256=digest,
        status=COMPLETE_STATUS,
        winner_feature_block_id=verified.winner_feature_block_id,
        holdout_reaction_accessed=False,
    )


__all__ = [
    "ANCHOR_ENDPOINT_SECONDS",
    "ANCHOR_LANDMARK_SECONDS",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "COMPLETE_STATUS",
    "MagnitudeFollowupError",
    "MagnitudeFollowupPublication",
    "NOT_APPLICABLE",
    "QUANTILE_COLUMNS",
    "SCHEMA",
    "VerifiedMagnitudeInputs",
    "build_magnitude_metric_tables",
    "compare_winner_to_d0",
    "publish_not_applicable",
    "run_magnitude_followup",
    "validate_magnitude_cell",
    "verify_magnitude_followup_inputs",
]
