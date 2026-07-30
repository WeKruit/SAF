"""Verified pre-event decision context for NFL X-15 Stage B.

This module intentionally has only two upstream inputs:

* the authoritative exact-153 Facts V4 batch; and
* the frozen Stage A state/reference batch.

It does not read market data, final-holdout reactions, or any upstream network
source.  ``reference_observations`` is used only to identify the pre-state ID
for an information episode.  Probability, support status, known-at time, and
model/input provenance always come from that exact row in
``state_predictions``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.research.nfl_x15_development_panel import (
    DEFAULT_FACTS_BATCH,
    DEFAULT_FACTS_BATCH_FILE_SHA256,
    DEFAULT_STAGE_A_BATCH,
    DEFAULT_STAGE_A_BATCH_FILE_SHA256,
    DevelopmentPanelError,
    _read_verified_table,
    _resolve_under,
    _table_descriptor,
    _verify_batch,
    _verify_game_manifest,
)


BUILDER_VERSION = "nfl-x15-stage-b-decision-context-v1"
CONTEXT_SCHEMA = "EventPrestateContextV1"
MANIFEST_SCHEMA = "EventPrestateContextManifestV1"
AUDIT_SCHEMA = "EventPrestateContextAuditV1"
CLAIM_BOUNDARY = (
    "RETROSPECTIVE_EVENT_PRESTATE_CONTEXT_ONLY;"
    "NO_MARKET_DATA;NO_HOLDOUT_REACTION;NO_POST_STATE_REFERENCE;"
    "OVERTIME_MODEL_SUPPORT_UNPROVEN;"
    "FINALIZED_EPISODE_PANEL_NOT_YET_BUILT"
)
EXPECTED_GAME_COUNT = 153
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/event-prestate-context-v1"
)
_FACTS_BATCH_SCHEMA = "nfl_x13_exact153_fact_batch_index_v4"
_STAGE_A_BATCH_SCHEMA = "nfl_x15_stage_a_batch_index_v1"
_SHA_PREFIX = "sha256:"
_GRAIN = ("game_id", "atomic_information_episode_id")


class DecisionContextError(RuntimeError):
    """A source, join, temporal, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class DecisionContextSourceSpec:
    facts_batch_path: Path = DEFAULT_FACTS_BATCH
    facts_batch_file_sha256: str = DEFAULT_FACTS_BATCH_FILE_SHA256
    stage_a_batch_path: Path = DEFAULT_STAGE_A_BATCH
    stage_a_batch_file_sha256: str = DEFAULT_STAGE_A_BATCH_FILE_SHA256
    expected_game_count: int = EXPECTED_GAME_COUNT


@dataclass(frozen=True, slots=True)
class PreparedDecisionContext:
    context: pd.DataFrame
    audit_counts: Mapping[str, int]
    facts_batch_file_sha256: str
    facts_batch_sha256: str
    stage_a_batch_file_sha256: str
    stage_a_batch_sha256: str
    game_count: int


@dataclass(frozen=True, slots=True)
class PublishedDecisionContext:
    output_root: Path
    context_path: Path
    context_sha256: str
    audit_path: Path
    audit_sha256: str
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    row_count: int
    game_count: int
    regulation_rows: int
    overtime_rows: int
    supported_prestate_rows: int


def _sha256_bytes(payload: bytes) -> str:
    return _SHA_PREFIX + hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_SHA_PREFIX)
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise DecisionContextError(f"{label} is not a canonical SHA-256")
    return value


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool,)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise DecisionContextError(
        f"unsupported canonical value {type(value).__name__}"
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DecisionContextError(f"{label} must be a DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DecisionContextError(
            f"{label} missing columns: {', '.join(missing)}"
        )


def _row_source_binding(row: pd.Series) -> str:
    return _canonical_sha256(
        {
            "facts_manifest_sha256": row["facts_manifest_sha256"],
            "facts_object_sha256": row["facts_object_sha256"],
            "stage_a_manifest_sha256": row["stage_a_manifest_sha256"],
            "state_predictions_object_sha256": (
                row["state_predictions_object_sha256"]
            ),
            "feature_provenance_object_sha256": (
                row["feature_provenance_object_sha256"]
            ),
            "reference_observations_object_sha256": (
                row["reference_observations_object_sha256"]
            ),
            "pbp_source_sha256": row["pbp_source_sha256"],
            "state_input_sha256": row["state_input_sha256"],
            "feature_provenance_semantic_sha256": (
                row["feature_provenance_semantic_sha256"]
            ),
            "model_sha256": row["model_sha256"],
            "model_manifest_sha256": row["model_manifest_sha256"],
        }
    )


def build_decision_context_frame(
    *,
    facts: pd.DataFrame,
    state_predictions: pd.DataFrame,
    feature_provenance: pd.DataFrame,
    reference_observations: pd.DataFrame,
    facts_object_sha256: str,
    state_predictions_object_sha256: str,
    feature_provenance_object_sha256: str,
    reference_observations_object_sha256: str,
    facts_manifest_sha256: str,
    stage_a_manifest_sha256: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build event-level pre-state rows with direct state identity joins."""

    for label, value in (
        ("facts_object_sha256", facts_object_sha256),
        ("state_predictions_object_sha256", state_predictions_object_sha256),
        (
            "feature_provenance_object_sha256",
            feature_provenance_object_sha256,
        ),
        (
            "reference_observations_object_sha256",
            reference_observations_object_sha256,
        ),
        ("facts_manifest_sha256", facts_manifest_sha256),
        ("stage_a_manifest_sha256", stage_a_manifest_sha256),
    ):
        _require_sha256(value, label=label)

    _require_columns(
        facts,
        {
            "game_id",
            "event_id",
            "raw_play_id",
            "atomic_information_episode_id",
            "quarter",
            "source_interval_end",
            "stage_b_information_event_eligible",
            "pbp_source_sha256",
        },
        label="facts",
    )
    _require_columns(
        reference_observations,
        {
            "game_id",
            "event_id",
            "raw_play_id",
            "atomic_information_episode_id",
            "pre_state_event_id",
            "pre_state_raw_play_id",
            "pbp_source_sha256",
        },
        label="reference_observations",
    )
    _require_columns(
        state_predictions,
        {
            "game_id",
            "state_event_id",
            "state_raw_play_id",
            "state_atomic_information_episode_id",
            "state_known_at",
            "p_home",
            "reference_status",
            "state_input_sha256",
            "model_id",
            "model_version",
            "model_sha256",
            "model_manifest_sha256",
            "pbp_source_sha256",
        },
        label="state_predictions",
    )
    _require_columns(
        feature_provenance,
        {
            "game_id",
            "state_event_id",
            "state_raw_play_id",
            "state_known_at",
            "feature_name",
            "feature_value",
            "feature_known_at",
            "source_event_id",
            "source_raw_play_id",
            "source_row_id",
            "source_hash",
            "PIT_status",
        },
        label="feature_provenance",
    )
    if facts.empty:
        raise DecisionContextError("facts cannot be empty")
    if facts[list(_GRAIN)].isna().any().any():
        raise DecisionContextError("facts grain cannot contain nulls")
    if facts.duplicated(list(_GRAIN)).any():
        raise DecisionContextError("facts contain duplicate decision grain")
    strong_key = [
        "game_id",
        "event_id",
        "raw_play_id",
        "atomic_information_episode_id",
    ]
    if reference_observations[strong_key].isna().any().any():
        raise DecisionContextError("reference mapping key cannot contain nulls")
    if reference_observations.duplicated(strong_key).any():
        raise DecisionContextError("reference mapping contains duplicate keys")
    state_key = [
        "game_id",
        "state_event_id",
        "state_raw_play_id",
        "state_atomic_information_episode_id",
    ]
    if state_predictions[state_key].isna().any().any():
        raise DecisionContextError("state prediction key cannot contain nulls")
    if state_predictions.duplicated(state_key).any():
        raise DecisionContextError("state predictions contain duplicate keys")

    fact_projection = facts[
        [
            "game_id",
            "event_id",
            "raw_play_id",
            "atomic_information_episode_id",
            "quarter",
            "source_interval_end",
            "stage_b_information_event_eligible",
            "pbp_source_sha256",
        ]
    ].rename(columns={"pbp_source_sha256": "fact_pbp_source_sha256"})
    reference_projection = reference_observations[
        strong_key
        + [
            "pre_state_event_id",
            "pre_state_raw_play_id",
            "pbp_source_sha256",
        ]
    ].rename(columns={"pbp_source_sha256": "reference_pbp_source_sha256"})
    identity_check = fact_projection.merge(
        reference_projection,
        on=strong_key,
        how="left",
        validate="one_to_one",
        indicator="_reference_join",
    )
    if not identity_check["_reference_join"].eq("both").all():
        raise DecisionContextError(
            "facts-to-reference identity cross-check is incomplete"
        )
    if not (
        identity_check["pre_state_event_id"].eq(identity_check["event_id"])
        & identity_check["pre_state_raw_play_id"].eq(
            identity_check["raw_play_id"]
        )
    ).all():
        raise DecisionContextError(
            "reference identity disagrees with the event pre-state identity"
        )

    state_projection = state_predictions[
        [
            "game_id",
            "state_event_id",
            "state_raw_play_id",
            "state_atomic_information_episode_id",
            "state_known_at",
            "p_home",
            "reference_status",
            "state_input_sha256",
            "model_id",
            "model_version",
            "model_sha256",
            "model_manifest_sha256",
            "pbp_source_sha256",
        ]
    ].rename(
        columns={
            "state_event_id": "event_id",
            "state_raw_play_id": "raw_play_id",
            "state_atomic_information_episode_id": (
                "atomic_information_episode_id"
            ),
            "state_known_at": "pre_state_known_at",
            "p_home": "_state_p_home",
            "reference_status": "prestate_support_status",
            "pbp_source_sha256": "state_pbp_source_sha256",
        }
    )
    joined = fact_projection.merge(
        state_projection,
        on=strong_key,
        how="left",
        validate="one_to_one",
        indicator="_prestate_join",
    )
    if not joined["_prestate_join"].eq("both").all():
        raise DecisionContextError("direct event-to-pre-state join is incomplete")

    quarter = pd.to_numeric(joined["quarter"], errors="coerce")
    if quarter.isna().any() or (~quarter.between(1, 5)).any():
        raise DecisionContextError("quarter must be an integer in [1, 5]")
    if not (quarter % 1).eq(0).all():
        raise DecisionContextError("quarter must be integral")
    joined["quarter"] = quarter.astype("int64")
    joined["is_regulation"] = joined["quarter"].le(4)
    joined["stage_b_information_event_eligible"] = joined[
        "stage_b_information_event_eligible"
    ].fillna(False).astype(bool)

    status = joined["prestate_support_status"].astype("string")
    if status.isna().any() or status.str.strip().eq("").any():
        raise DecisionContextError("pre-state support status is missing")
    supported = status.eq("SUPPORTED")
    probability = pd.to_numeric(joined["_state_p_home"], errors="coerce")
    if probability.loc[supported].isna().any() or (
        ~probability.loc[supported].between(0.0, 1.0)
    ).any():
        raise DecisionContextError(
            "supported pre-state has invalid home probability"
        )
    if probability.loc[~supported].notna().any():
        raise DecisionContextError(
            "unsupported pre-state carries a home probability"
        )
    if joined.loc[supported, "pre_state_known_at"].isna().any():
        raise DecisionContextError("supported pre-state has no known-at time")
    if joined.loc[supported, "state_input_sha256"].isna().any():
        raise DecisionContextError("supported pre-state has no state input hash")
    for value in joined.loc[supported, "state_input_sha256"].astype(str):
        _require_sha256(value, label="state_input_sha256")
    if (supported & ~joined["is_regulation"]).any():
        raise DecisionContextError(
            "overtime event cannot have a supported Stage A pre-state"
        )
    eligible = joined["stage_b_information_event_eligible"]
    if joined.loc[eligible, "source_interval_end"].isna().any():
        raise DecisionContextError("eligible event has no interval end")

    joined = joined.merge(
        identity_check[
            strong_key
            + ["reference_pbp_source_sha256"]
        ],
        on=strong_key,
        how="left",
        validate="one_to_one",
    )
    pbp_columns = (
        "fact_pbp_source_sha256",
        "reference_pbp_source_sha256",
        "state_pbp_source_sha256",
    )
    for column in pbp_columns:
        joined[column] = joined[column].astype("string")
        for value in joined[column].dropna().astype(str).unique():
            _require_sha256(value, label=column)
    if not (
        joined[pbp_columns[0]].eq(joined[pbp_columns[1]])
        & joined[pbp_columns[0]].eq(joined[pbp_columns[2]])
    ).all():
        raise DecisionContextError("PBP source hashes disagree across inputs")

    for column in ("model_sha256", "model_manifest_sha256"):
        for value in joined[column].dropna().astype(str).unique():
            _require_sha256(value, label=column)
    if joined[["model_id", "model_version", "model_sha256",
               "model_manifest_sha256"]].isna().any().any():
        raise DecisionContextError("model binding is incomplete")

    feature_key = ["game_id", "state_event_id", "feature_name"]
    if not feature_provenance.empty and (
        feature_provenance[feature_key].isna().any().any()
        or feature_provenance.duplicated(feature_key).any()
    ):
        raise DecisionContextError("feature provenance key is invalid")
    state_id_set = set(
        zip(
            joined["game_id"].astype(str),
            joined["event_id"].astype(str),
        )
    )
    provenance_id_set = set(
        zip(
            feature_provenance["game_id"].astype(str),
            feature_provenance["state_event_id"].astype(str),
        )
    )
    if not provenance_id_set.issubset(state_id_set):
        raise DecisionContextError("feature provenance contains orphan states")
    provenance_groups = {
        (str(game_id), str(state_event_id)): group.sort_values(
            "feature_name", kind="mergesort"
        )
        for (game_id, state_event_id), group in feature_provenance.groupby(
            ["game_id", "state_event_id"],
            sort=False,
        )
    }
    feature_counts: list[int] = []
    feature_hashes: list[object] = []
    for row in joined.itertuples(index=False):
        key = (str(row.game_id), str(row.event_id))
        group = provenance_groups.get(key)
        if row.prestate_support_status == "SUPPORTED":
            if group is None or len(group) != 11:
                raise DecisionContextError(
                    "supported pre-state must have exactly 11 feature rows"
                )
            if (
                not group["PIT_status"].eq("PIT_VERIFIED").all()
                or not group["state_known_at"].astype("string").eq(
                    str(row.pre_state_known_at)
                ).all()
            ):
                raise DecisionContextError(
                    "supported feature provenance is not PIT verified"
                )
            feature_known = pd.to_datetime(
                group["feature_known_at"],
                utc=True,
                errors="coerce",
                format="mixed",
            )
            state_known = pd.to_datetime(
                row.pre_state_known_at,
                utc=True,
                errors="coerce",
                format="mixed",
            )
            if (
                feature_known.isna().any()
                or pd.isna(state_known)
                or (feature_known > state_known).any()
            ):
                raise DecisionContextError(
                    "feature provenance is not known by the pre-state time "
                    f"for {row.game_id}/{row.event_id}"
                )
            if not group["source_hash"].astype("string").eq(
                str(row.state_pbp_source_sha256)
            ).all():
                raise DecisionContextError(
                    "feature provenance source hash disagrees with state"
                )
            feature_counts.append(11)
            feature_hashes.append(
                _canonical_sha256(group.to_dict("records"))
            )
        else:
            if group is not None and not group.empty:
                raise DecisionContextError(
                    "unsupported pre-state cannot carry model feature provenance"
                )
            feature_counts.append(0)
            feature_hashes.append(pd.NA)

    context = pd.DataFrame(
        {
            "schema_version": CONTEXT_SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "game_id": joined["game_id"].astype("string"),
            "atomic_information_episode_id": joined[
                "atomic_information_episode_id"
            ].astype("string"),
            "event_id": joined["event_id"].astype("string"),
            "quarter": joined["quarter"].astype("int64"),
            "is_regulation": joined["is_regulation"].astype(bool),
            "event_interval_end": joined["source_interval_end"].astype("string"),
            "stage_b_information_event_eligible": eligible.astype(bool),
            "pre_state_event_id": joined["event_id"].astype("string"),
            "pre_state_known_at": joined["pre_state_known_at"].astype("string"),
            "p_before_home": probability.where(supported).astype("Float64"),
            "prestate_support_status": status,
            "state_input_sha256": joined["state_input_sha256"].astype("string"),
            "feature_provenance_row_count": pd.Series(
                feature_counts, dtype="Int64"
            ),
            "feature_provenance_semantic_sha256": pd.Series(
                feature_hashes, dtype="string"
            ),
            "model_id": joined["model_id"].astype("string"),
            "model_version": joined["model_version"].astype("string"),
            "model_sha256": joined["model_sha256"].astype("string"),
            "model_manifest_sha256": joined[
                "model_manifest_sha256"
            ].astype("string"),
            "pbp_source_sha256": joined[
                "fact_pbp_source_sha256"
            ].astype("string"),
            "facts_manifest_sha256": facts_manifest_sha256,
            "facts_object_sha256": facts_object_sha256,
            "stage_a_manifest_sha256": stage_a_manifest_sha256,
            "state_predictions_object_sha256": (
                state_predictions_object_sha256
            ),
            "feature_provenance_object_sha256": (
                feature_provenance_object_sha256
            ),
            "reference_observations_object_sha256": (
                reference_observations_object_sha256
            ),
        }
    )
    context["source_binding_sha256"] = context.apply(
        _row_source_binding,
        axis=1,
    ).astype("string")
    context = context.sort_values(
        ["game_id", "atomic_information_episode_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if context.duplicated(list(_GRAIN)).any():
        raise DecisionContextError("published context grain is not unique")

    finalized_regulation = eligible & joined["is_regulation"]
    audit = {
        "context_rows": len(context),
        "game_count": int(context["game_id"].nunique()),
        "reference_identity_expected_rows": len(facts),
        "reference_identity_matched_rows": len(identity_check),
        "prestate_join_expected_rows": len(facts),
        "prestate_join_matched_rows": len(joined),
        "eligible_rows": int(eligible.sum()),
        "regulation_rows": int(context["is_regulation"].sum()),
        "overtime_rows": int((~context["is_regulation"]).sum()),
        "finalized_regulation_anchor_rows": int(
            finalized_regulation.sum()
        ),
        "finalized_regulation_supported_rows": int(
            (finalized_regulation & supported).sum()
        ),
        "finalized_regulation_missing_prestate_rows": int(
            (
                finalized_regulation
                & status.eq("MISSING_PRE_STATE")
            ).sum()
        ),
        "finalized_regulation_model_support_unproven_rows": int(
            (
                finalized_regulation
                & status.eq("MODEL_SUPPORT_UNPROVEN")
            ).sum()
        ),
        "supported_prestate_rows": int(supported.sum()),
        "unsupported_prestate_rows": int((~supported).sum()),
        "supported_feature_provenance_rows": int(sum(feature_counts)),
        "supported_states_with_11_feature_rows": int(
            sum(count == 11 for count in feature_counts)
        ),
        "feature_state_known_at_mismatch_rows": 0,
        "pbp_source_hash_mismatch_rows": 0,
        "supported_overtime_rows": int(
            (supported & ~joined["is_regulation"]).sum()
        ),
    }
    for value, count in status.value_counts(dropna=False).items():
        audit[f"prestate_status_{value}_rows"] = int(count)
    return context, audit


def _descriptor(
    manifest: Mapping[str, Any],
    table_name: str,
) -> Mapping[str, Any]:
    try:
        return _table_descriptor(
            manifest,
            table_name=table_name,
            market_style=False,
            label=str(manifest.get("game_id", "game")),
        )
    except DevelopmentPanelError as exc:
        raise DecisionContextError(str(exc)) from exc


def prepare_exact153_decision_context(
    *,
    project_root: Path,
    source_spec: DecisionContextSourceSpec | None = None,
) -> PreparedDecisionContext:
    """Verify the complete two-source cohort and build the small context."""

    project = Path(project_root).resolve()
    spec = source_spec or DecisionContextSourceSpec()
    if (
        type(spec.expected_game_count) is not int
        or spec.expected_game_count != EXPECTED_GAME_COUNT
    ):
        raise DecisionContextError("decision context requires exactly 153 games")
    try:
        facts_batch = _verify_batch(
            project_root=project,
            path=spec.facts_batch_path,
            expected_file_sha256=spec.facts_batch_file_sha256,
            expected_schema=_FACTS_BATCH_SCHEMA,
            expected_game_count=spec.expected_game_count,
            label="Facts V4",
            market_style=False,
            facts_v4=True,
        )
        stage_a_batch = _verify_batch(
            project_root=project,
            path=spec.stage_a_batch_path,
            expected_file_sha256=spec.stage_a_batch_file_sha256,
            expected_schema=_STAGE_A_BATCH_SCHEMA,
            expected_game_count=spec.expected_game_count,
            label="Stage A",
            market_style=False,
        )
    except DevelopmentPanelError as exc:
        raise DecisionContextError(str(exc)) from exc
    if set(facts_batch.games) != set(stage_a_batch.games):
        raise DecisionContextError("Facts V4 and Stage A game sets differ")

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, int]] = []
    for game_id in sorted(facts_batch.games):
        try:
            facts_manifest, _ = _verify_game_manifest(
                batch=facts_batch,
                descriptor=facts_batch.games[game_id],
                label="Facts V4",
            )
            stage_manifest, _ = _verify_game_manifest(
                batch=stage_a_batch,
                descriptor=stage_a_batch.games[game_id],
                label="Stage A",
            )
            facts = _read_verified_table(
                batch=facts_batch,
                manifest=facts_manifest,
                table_name="canonical_factor_events",
                market_style=False,
                label=f"Facts V4.{game_id}",
            )
            states = _read_verified_table(
                batch=stage_a_batch,
                manifest=stage_manifest,
                table_name="state_predictions",
                market_style=False,
                label=f"Stage A.{game_id}",
            )
            features = _read_verified_table(
                batch=stage_a_batch,
                manifest=stage_manifest,
                table_name="feature_provenance",
                market_style=False,
                label=f"Stage A.{game_id}",
            )
            references = _read_verified_table(
                batch=stage_a_batch,
                manifest=stage_manifest,
                table_name="reference_observations",
                market_style=False,
                label=f"Stage A.{game_id}",
            )
        except DevelopmentPanelError as exc:
            raise DecisionContextError(str(exc)) from exc
        batch_model = stage_a_batch.document.get("model")
        if (
            type(batch_model) is not dict
            or stage_manifest.get("model") != batch_model
        ):
            raise DecisionContextError(
                f"{game_id} Stage A model manifest binding mismatch"
            )
        model_columns = {
            "model_id": "model_id",
            "model_version": "model_version",
            "model_sha256": "model_sha256",
            "model_manifest_sha256": "manifest_sha256",
        }
        for column, model_field in model_columns.items():
            if (
                column not in states
                or not states[column]
                .astype("string")
                .eq(str(batch_model[model_field]))
                .all()
            ):
                raise DecisionContextError(
                    f"{game_id} state predictions disagree with model binding"
                )
        facts_descriptor = _descriptor(
            facts_manifest, "canonical_factor_events"
        )
        state_descriptor = _descriptor(stage_manifest, "state_predictions")
        feature_descriptor = _descriptor(stage_manifest, "feature_provenance")
        reference_descriptor = _descriptor(
            stage_manifest, "reference_observations"
        )
        context, audit = build_decision_context_frame(
            facts=facts,
            state_predictions=states,
            feature_provenance=features,
            reference_observations=references,
            facts_object_sha256=str(facts_descriptor["object_sha256"]),
            state_predictions_object_sha256=str(
                state_descriptor["object_sha256"]
            ),
            feature_provenance_object_sha256=str(
                feature_descriptor["object_sha256"]
            ),
            reference_observations_object_sha256=str(
                reference_descriptor["object_sha256"]
            ),
            facts_manifest_sha256=str(
                facts_batch.games[game_id]["manifest_sha256"]
            ),
            stage_a_manifest_sha256=str(
                stage_a_batch.games[game_id]["manifest_sha256"]
            ),
        )
        if set(context["game_id"].astype(str)) != {game_id}:
            raise DecisionContextError(f"{game_id} context crossed game boundary")
        frames.append(context)
        audits.append(audit)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(list(_GRAIN), kind="mergesort").reset_index(
        drop=True
    )
    if (
        combined["game_id"].nunique() != spec.expected_game_count
        or combined.duplicated(list(_GRAIN)).any()
    ):
        raise DecisionContextError("exact-153 combined context is incomplete")
    aggregate: dict[str, int] = {}
    for audit in audits:
        for key, value in audit.items():
            if key == "game_count":
                continue
            aggregate[key] = aggregate.get(key, 0) + int(value)
    aggregate["game_count"] = int(combined["game_id"].nunique())
    if (
        aggregate["reference_identity_expected_rows"]
        != aggregate["reference_identity_matched_rows"]
        or aggregate["prestate_join_expected_rows"]
        != aggregate["prestate_join_matched_rows"]
        or aggregate["supported_overtime_rows"] != 0
    ):
        raise DecisionContextError("exact-153 join or overtime audit failed")
    return PreparedDecisionContext(
        context=combined,
        audit_counts=aggregate,
        facts_batch_file_sha256=spec.facts_batch_file_sha256,
        facts_batch_sha256=str(facts_batch.document["batch_sha256"]),
        stage_a_batch_file_sha256=spec.stage_a_batch_file_sha256,
        stage_a_batch_sha256=str(stage_a_batch.document["batch_sha256"]),
        game_count=spec.expected_game_count,
    )


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise DecisionContextError("context cannot be encoded as Parquet") from exc
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
    )
    return (
        sink.getvalue().to_pybytes(),
        _sha256_bytes(table.schema.remove_metadata().serialize().to_pybytes()),
    )


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise DecisionContextError(f"content-addressed collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise DecisionContextError(f"content-addressed publish race: {path}")


def _content_path(
    output_root: Path,
    *,
    directory: str,
    sha256: str,
    suffix: str,
) -> Path:
    hexadecimal = sha256.removeprefix(_SHA_PREFIX)
    return (
        output_root
        / directory
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}{suffix}"
    )


def publish_decision_context_tables(
    *,
    output_root: Path,
    context: pd.DataFrame,
    audit_counts: Mapping[str, int],
    facts_batch_file_sha256: str,
    facts_batch_sha256: str,
    stage_a_batch_file_sha256: str,
    stage_a_batch_sha256: str,
    expected_game_count: int = EXPECTED_GAME_COUNT,
) -> PublishedDecisionContext:
    """Publish one immutable Parquet, audit JSON, and manifest."""

    for label, value in (
        ("facts_batch_file_sha256", facts_batch_file_sha256),
        ("facts_batch_sha256", facts_batch_sha256),
        ("stage_a_batch_file_sha256", stage_a_batch_file_sha256),
        ("stage_a_batch_sha256", stage_a_batch_sha256),
    ):
        _require_sha256(value, label=label)
    if (
        not isinstance(context, pd.DataFrame)
        or context.empty
        or context.duplicated(list(_GRAIN)).any()
        or context["game_id"].nunique() != expected_game_count
    ):
        raise DecisionContextError("context publication grain/count is invalid")
    if int(audit_counts.get("supported_overtime_rows", -1)) != 0:
        raise DecisionContextError("supported overtime rows cannot be published")

    output = Path(output_root).resolve()
    context_payload, schema_fingerprint = _parquet_bytes(context)
    context_sha = _sha256_bytes(context_payload)
    context_path = _content_path(
        output,
        directory="objects",
        sha256=context_sha,
        suffix=".parquet",
    )
    _atomic_publish(context_path, context_payload)
    semantic_sha = _canonical_sha256(context.to_dict("records"))

    audit_material: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "stage_b_panel_status": "FINALIZED_EPISODE_PANEL_NOT_YET_BUILT",
        "stage_b_promotion_gate": "BLOCKED_PENDING_EPISODE_AGGREGATION",
        "overtime_scope": (
            "QUARTER_GT_4_RETAINED_WITH_IS_REGULATION_FALSE;"
            "STAGE_A_MODEL_SUPPORT_UNPROVEN;P_BEFORE_HOME_NULL"
        ),
        "counts": {key: int(value) for key, value in sorted(audit_counts.items())},
        "context_semantic_rows_sha256": semantic_sha,
    }
    audit_payload = _canonical_bytes(audit_material)
    audit_sha = _sha256_bytes(audit_payload)
    audit_path = _content_path(
        output,
        directory="audits",
        sha256=audit_sha,
        suffix=".audit.json",
    )
    _atomic_publish(audit_path, audit_payload)

    material: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "cohort": "development",
        "publication_gate": "PASS",
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "stage_b_panel_status": "FINALIZED_EPISODE_PANEL_NOT_YET_BUILT",
        "stage_b_promotion_gate": "BLOCKED_PENDING_EPISODE_AGGREGATION",
        "game_count": int(context["game_id"].nunique()),
        "audit_counts": {
            key: int(value) for key, value in sorted(audit_counts.items())
        },
        "sources": {
            "facts_v4_batch_file_sha256": facts_batch_file_sha256,
            "facts_v4_batch_sha256": facts_batch_sha256,
            "stage_a_batch_file_sha256": stage_a_batch_file_sha256,
            "stage_a_batch_sha256": stage_a_batch_sha256,
        },
        "context": {
            "schema": CONTEXT_SCHEMA,
            "grain": list(_GRAIN),
            "object_path": context_path.relative_to(output).as_posix(),
            "object_sha256": context_sha,
            "byte_length": len(context_payload),
            "row_count": len(context),
            "schema_columns": list(context.columns),
            "schema_fingerprint": schema_fingerprint,
            "semantic_rows_sha256": semantic_sha,
        },
        "audit": {
            "schema": AUDIT_SCHEMA,
            "object_path": audit_path.relative_to(output).as_posix(),
            "object_sha256": audit_sha,
            "byte_length": len(audit_payload),
        },
    }
    bundle_sha = _canonical_sha256(material)
    manifest = {**material, "bundle_sha256": bundle_sha}
    manifest_payload = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    manifest_path = _content_path(
        output,
        directory="manifests",
        sha256=manifest_sha,
        suffix=".manifest.json",
    )
    _atomic_publish(manifest_path, manifest_payload)
    return PublishedDecisionContext(
        output_root=output,
        context_path=context_path,
        context_sha256=context_sha,
        audit_path=audit_path,
        audit_sha256=audit_sha,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        bundle_sha256=bundle_sha,
        row_count=len(context),
        game_count=int(context["game_id"].nunique()),
        regulation_rows=int(audit_counts["regulation_rows"]),
        overtime_rows=int(audit_counts["overtime_rows"]),
        supported_prestate_rows=int(audit_counts["supported_prestate_rows"]),
    )


def publish_exact153_decision_context(
    *,
    project_root: Path,
    output_root: Path | None = None,
    source_spec: DecisionContextSourceSpec | None = None,
) -> PublishedDecisionContext:
    """Prepare and publish the complete exact-153 context."""

    project = Path(project_root).resolve()
    prepared = prepare_exact153_decision_context(
        project_root=project,
        source_spec=source_spec,
    )
    try:
        output = _resolve_under(
            project,
            output_root or DEFAULT_OUTPUT_ROOT,
            label="decision context output",
        )
    except DevelopmentPanelError as exc:
        raise DecisionContextError(str(exc)) from exc
    return publish_decision_context_tables(
        output_root=output,
        context=prepared.context,
        audit_counts=prepared.audit_counts,
        facts_batch_file_sha256=prepared.facts_batch_file_sha256,
        facts_batch_sha256=prepared.facts_batch_sha256,
        stage_a_batch_file_sha256=prepared.stage_a_batch_file_sha256,
        stage_a_batch_sha256=prepared.stage_a_batch_sha256,
        expected_game_count=prepared.game_count,
    )


__all__ = [
    "AUDIT_SCHEMA",
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "CONTEXT_SCHEMA",
    "DEFAULT_OUTPUT_ROOT",
    "DecisionContextError",
    "DecisionContextSourceSpec",
    "MANIFEST_SCHEMA",
    "PreparedDecisionContext",
    "PublishedDecisionContext",
    "build_decision_context_frame",
    "prepare_exact153_decision_context",
    "publish_decision_context_tables",
    "publish_exact153_decision_context",
]
