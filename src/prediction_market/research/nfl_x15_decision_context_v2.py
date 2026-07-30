"""PIA-grained, PIT-audited pre-event context for NFL X-15 Panel V4.

The only semantic inputs are:

* the verified ``ProvisionalInformationAnchorV2`` publication;
* Exact-153 Facts V4 pre-event fields; and
* frozen Stage-A state predictions plus their feature provenance.

The output grain is ``game_id + information_anchor_id``.  It deliberately
contains no post-event or finalized-outcome columns and reads neither market
data nor final-holdout reactions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Final, Mapping
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
    _sha256_file,
    _table_descriptor,
    _verify_batch,
    _verify_game_manifest,
)


CONTEXT_SCHEMA: Final[str] = "EventPrestateContextV2"
AUDIT_SCHEMA: Final[str] = "EventPrestateContextAuditV2"
MANIFEST_SCHEMA: Final[str] = "EventPrestateContextManifestV2"
BUILDER_VERSION: Final[str] = "nfl-x15-event-prestate-context-v2"
EXPERIMENT_ID: Final[str] = "X-15"
CLAIM_BOUNDARY: Final[str] = (
    "RETROSPECTIVE_PIA_PRESTATE_CONTEXT_ONLY;"
    "NO_POST_OR_FINAL_OUTCOME_FEATURES;NO_MARKET_DATA;NO_HOLDOUT_REACTION;"
    "OVERTIME_REFERENCE_SUPPORT_UNPROVEN"
)
EXPECTED_GAME_COUNT: Final[int] = 153
EXPECTED_ANCHOR_COUNT: Final[int] = 25_250
DEFAULT_INFORMATION_ANCHOR_MANIFEST: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/"
    "finalized-episodes-regulation-v2/manifests/sha256/fa/"
    "fac9db018900fae919fd94fd2fec064f5eb3fcae24255fbedcc913099ad2b208"
    ".manifest.json"
)
DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256: Final[str] = (
    "sha256:fac9db018900fae919fd94fd2fec064f5eb3fcae24255fbedcc913099ad2b208"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/event-prestate-context-v2"
)

_PIA_MANIFEST_SCHEMA: Final[str] = "nfl_x15_finalized_episode_batch_v2"
_PIA_TABLE_SCHEMA: Final[str] = "ProvisionalInformationAnchorV2"
_FACTS_BATCH_SCHEMA: Final[str] = "nfl_x13_exact153_fact_batch_index_v4"
_STAGE_A_BATCH_SCHEMA: Final[str] = "nfl_x15_stage_a_batch_index_v1"
_SHA_PREFIX: Final[str] = "sha256:"
_GRAIN: Final[tuple[str, str]] = ("game_id", "information_anchor_id")
_SPORTS_FIELDS: Final[tuple[str, ...]] = (
    "quarter",
    "game_clock",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "possession_before",
    "possession_is_home",
    "pre_home_score",
    "pre_away_score",
    "score_margin_home",
    "down",
    "distance",
    "yardline_100",
    "goal_to_go",
    "pre_red_zone",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
)
_PUBLISHED_SPORTS_FIELDS: Final[tuple[str, ...]] = (
    "quarter",
    "game_clock",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "possession_team",
    "possession_is_home",
    "pre_home_score",
    "pre_away_score",
    "score_margin_home",
    "down",
    "distance",
    "yardline_100",
    "goal_to_go",
    "pre_red_zone",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
)
_FORBIDDEN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "p_after_home",
        "reference_delta_home",
        "bridge_p_after_home",
        "bridge_delta_home",
        "reference_gap",
        "post_home_score",
        "post_away_score",
        "final_home_score",
        "final_away_score",
        "final_home_win",
    }
)


class DecisionContextV2Error(RuntimeError):
    """A source, identity, temporal, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class DecisionContextV2SourceSpec:
    information_anchor_manifest_path: Path = (
        DEFAULT_INFORMATION_ANCHOR_MANIFEST
    )
    information_anchor_manifest_file_sha256: str = (
        DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256
    )
    facts_batch_path: Path = DEFAULT_FACTS_BATCH
    facts_batch_file_sha256: str = DEFAULT_FACTS_BATCH_FILE_SHA256
    stage_a_batch_path: Path = DEFAULT_STAGE_A_BATCH
    stage_a_batch_file_sha256: str = DEFAULT_STAGE_A_BATCH_FILE_SHA256
    expected_game_count: int = EXPECTED_GAME_COUNT
    expected_anchor_count: int = EXPECTED_ANCHOR_COUNT


@dataclass(frozen=True, slots=True)
class PreparedDecisionContextV2:
    context: pd.DataFrame
    audit_counts: Mapping[str, int]
    information_anchor_manifest_file_sha256: str
    information_anchor_bundle_sha256: str
    facts_batch_file_sha256: str
    facts_batch_sha256: str
    stage_a_batch_file_sha256: str
    stage_a_batch_sha256: str
    game_count: int
    anchor_count: int


@dataclass(frozen=True, slots=True)
class PublishedDecisionContextV2:
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
    primary_selection_rows: int
    censor_boundary_rows: int
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
        raise DecisionContextV2Error(f"{label} is not a canonical SHA-256")
    return value


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise DecisionContextV2Error(
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
        raise DecisionContextV2Error(f"{label} must be a DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DecisionContextV2Error(
            f"{label} missing columns: {', '.join(missing)}"
        )


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.map(lambda value: pd.isna(value) or math.isfinite(float(value))))


def _row_source_binding(row: pd.Series) -> str:
    return _canonical_sha256(
        {
            "information_anchor_manifest_sha256": (
                row["information_anchor_manifest_sha256"]
            ),
            "information_anchor_object_sha256": (
                row["information_anchor_object_sha256"]
            ),
            "facts_manifest_sha256": row["facts_manifest_sha256"],
            "facts_object_sha256": row["facts_object_sha256"],
            "stage_a_manifest_sha256": row["stage_a_manifest_sha256"],
            "state_predictions_object_sha256": (
                row["state_predictions_object_sha256"]
            ),
            "feature_provenance_object_sha256": (
                row["feature_provenance_object_sha256"]
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


def compute_prestate_context_sha256(row: Mapping[str, object]) -> str:
    """Hash the complete formal pre-state feature row and its provenance."""

    required = {
        "information_anchor_id",
        "information_anchor",
        *_PUBLISHED_SPORTS_FIELDS,
        "p_before_home",
        "p_before_home_missing",
        "prestate_support_status",
        "pre_state_known_at",
        "sports_prestate_known_at",
        "sports_prestate_known_at_basis",
        "source_binding_sha256",
        "state_input_sha256",
        "model_id",
        "model_version",
        "model_sha256",
        "model_manifest_sha256",
        "feature_provenance_row_count",
        "feature_provenance_semantic_sha256",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise DecisionContextV2Error(
            f"prestate context hash missing fields: {', '.join(missing)}"
        )
    material = {
        "schema": "NFLPrestateContextSnapshotV1",
        "information_anchor_id": row["information_anchor_id"],
        "information_anchor": row["information_anchor"],
        "sports_prestate": {
            field: row[field] for field in _PUBLISHED_SPORTS_FIELDS
        },
        "reference_prestate": {
            "p_before_home": row["p_before_home"],
            "p_before_home_missing": row["p_before_home_missing"],
            "prestate_support_status": row["prestate_support_status"],
            "pre_state_known_at": row["pre_state_known_at"],
            "sports_prestate_known_at": row["sports_prestate_known_at"],
            "sports_prestate_known_at_basis": (
                row["sports_prestate_known_at_basis"]
            ),
        },
        "provenance": {
            "source_binding_sha256": row["source_binding_sha256"],
            "state_input_sha256": row["state_input_sha256"],
            "model_id": row["model_id"],
            "model_version": row["model_version"],
            "model_sha256": row["model_sha256"],
            "model_manifest_sha256": row["model_manifest_sha256"],
            "feature_provenance_row_count": (
                row["feature_provenance_row_count"]
            ),
            "feature_provenance_semantic_sha256": (
                row["feature_provenance_semantic_sha256"]
            ),
        },
    }
    return _canonical_sha256(material)


def build_decision_context_v2_frame(
    *,
    information_anchors: pd.DataFrame,
    facts: pd.DataFrame,
    state_predictions: pd.DataFrame,
    feature_provenance: pd.DataFrame,
    information_anchor_object_sha256: str,
    facts_object_sha256: str,
    state_predictions_object_sha256: str,
    feature_provenance_object_sha256: str,
    information_anchor_manifest_sha256: str,
    facts_manifest_sha256: str,
    stage_a_manifest_sha256: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one game's complete PIA-grained pre-event context."""

    for label, value in (
        ("information_anchor_object_sha256", information_anchor_object_sha256),
        ("facts_object_sha256", facts_object_sha256),
        ("state_predictions_object_sha256", state_predictions_object_sha256),
        ("feature_provenance_object_sha256", feature_provenance_object_sha256),
        ("information_anchor_manifest_sha256", information_anchor_manifest_sha256),
        ("facts_manifest_sha256", facts_manifest_sha256),
        ("stage_a_manifest_sha256", stage_a_manifest_sha256),
    ):
        _require_sha256(value, label=label)

    _require_columns(
        information_anchors,
        {
            "schema_version",
            "game_id",
            "information_anchor_id",
            "episode_id",
            "event_id",
            "raw_play_id",
            "order_sequence",
            "source_interval_start",
            "source_interval_end",
            "information_anchor",
            "provisional_known_at",
            "provisional_primary_event_id",
            "provisional_feature_support_status",
            "primary_selection_eligible",
            "censor_boundary_eligible",
            "pbp_source_sha256",
        },
        label="ProvisionalInformationAnchorV2",
    )
    _require_columns(
        facts,
        {
            "game_id",
            "event_id",
            "raw_play_id",
            "source_interval_start",
            "source_interval_end",
            "known_at",
            "order_sequence",
            "pbp_source_sha256",
            *_SPORTS_FIELDS,
        },
        label="Facts V4",
    )
    _require_columns(
        state_predictions,
        {
            "game_id",
            "state_event_id",
            "state_raw_play_id",
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
        label="Stage A state_predictions",
    )
    _require_columns(
        feature_provenance,
        {
            "game_id",
            "state_event_id",
            "state_raw_play_id",
            "state_known_at",
            "feature_name",
            "feature_known_at",
            "source_hash",
            "PIT_status",
        },
        label="Stage A feature_provenance",
    )
    if information_anchors.empty:
        raise DecisionContextV2Error("information anchors cannot be empty")
    if information_anchors["schema_version"].astype(str).ne(
        _PIA_TABLE_SCHEMA
    ).any():
        raise DecisionContextV2Error("information-anchor schema mismatch")
    if (
        information_anchors[list(_GRAIN)].isna().any().any()
        or information_anchors.duplicated(list(_GRAIN)).any()
        or information_anchors["event_id"].astype(str).duplicated().any()
    ):
        raise DecisionContextV2Error("information-anchor grain is invalid")
    if facts[["game_id", "event_id", "raw_play_id"]].isna().any().any():
        raise DecisionContextV2Error("Facts V4 identity is incomplete")
    if facts["event_id"].astype(str).duplicated().any():
        raise DecisionContextV2Error("Facts V4 event identity is duplicate")
    state_key = ["game_id", "state_event_id", "state_raw_play_id"]
    if (
        state_predictions[state_key].isna().any().any()
        or state_predictions.duplicated(state_key).any()
    ):
        raise DecisionContextV2Error("state prediction identity is invalid")

    game_sets = (
        set(information_anchors["game_id"].astype(str)),
        set(facts["game_id"].astype(str)),
        set(state_predictions["game_id"].astype(str)),
    )
    if len(game_sets[0]) != 1 or not game_sets[0].issubset(
        game_sets[1] & game_sets[2]
    ):
        raise DecisionContextV2Error("single-game source identities disagree")

    anchors = information_anchors.copy()
    if not anchors["event_id"].astype(str).eq(
        anchors["provisional_primary_event_id"].astype(str)
    ).all():
        raise DecisionContextV2Error(
            "PIA event and provisional primary event identities disagree"
        )
    anchors = anchors.rename(
        columns={
            "source_interval_start": "pia_source_interval_start",
            "source_interval_end": "pia_source_interval_end",
            "pbp_source_sha256": "pia_pbp_source_sha256",
            "episode_id": "parent_episode_id_audit_only",
        }
    )
    fact_projection = facts[
        [
            "game_id",
            "event_id",
            "raw_play_id",
            "source_interval_start",
            "source_interval_end",
            "known_at",
            "order_sequence",
            "pbp_source_sha256",
            *_SPORTS_FIELDS,
        ]
    ].rename(
        columns={
            "source_interval_start": "fact_source_interval_start",
            "source_interval_end": "fact_source_interval_end",
            "known_at": "sports_prestate_known_at",
            "order_sequence": "fact_order_sequence",
            "pbp_source_sha256": "fact_pbp_source_sha256",
        }
    )
    joined = anchors.merge(
        fact_projection,
        on=["game_id", "event_id", "raw_play_id"],
        how="left",
        validate="one_to_one",
        indicator="_facts_join",
    )
    if not joined["_facts_join"].eq("both").all():
        raise DecisionContextV2Error(
            "information-anchor to Facts V4 join is incomplete"
        )

    state_projection = state_predictions[
        [
            "game_id",
            "state_event_id",
            "state_raw_play_id",
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
            "state_known_at": "pre_state_known_at",
            "p_home": "_state_p_home",
            "reference_status": "prestate_support_status",
            "pbp_source_sha256": "state_pbp_source_sha256",
        }
    )
    joined = joined.merge(
        state_projection,
        on=["game_id", "event_id", "raw_play_id"],
        how="left",
        validate="one_to_one",
        indicator="_state_join",
    )
    if not joined["_state_join"].eq("both").all():
        raise DecisionContextV2Error(
            "information-anchor to Stage A pre-state join is incomplete"
        )

    timestamp_columns = (
        "pia_source_interval_start",
        "pia_source_interval_end",
        "information_anchor",
        "fact_source_interval_start",
        "fact_source_interval_end",
        "sports_prestate_known_at",
        "pre_state_known_at",
    )
    parsed: dict[str, pd.Series] = {
        column: pd.to_datetime(
            joined[column],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        for column in timestamp_columns
    }
    required_time_columns = (
        "pia_source_interval_start",
        "pia_source_interval_end",
        "information_anchor",
        "fact_source_interval_start",
        "fact_source_interval_end",
    )
    if any(parsed[column].isna().any() for column in required_time_columns):
        raise DecisionContextV2Error("required PIA/Facts timestamp is missing")
    sports_known_was_missing = parsed["sports_prestate_known_at"].isna()
    parsed["sports_prestate_known_at"] = parsed[
        "sports_prestate_known_at"
    ].fillna(parsed["fact_source_interval_end"])
    if not (
        parsed["pia_source_interval_start"].eq(
            parsed["fact_source_interval_start"]
        )
        & parsed["pia_source_interval_end"].eq(
            parsed["fact_source_interval_end"]
        )
        & parsed["information_anchor"].eq(
            parsed["pia_source_interval_end"]
        )
        & joined["order_sequence"].astype("Int64").eq(
            joined["fact_order_sequence"].astype("Int64")
        )
    ).all():
        raise DecisionContextV2Error(
            "PIA and Facts source interval/order identities disagree"
        )
    if (
        (parsed["sports_prestate_known_at"] > parsed["information_anchor"]).any()
        or (
            parsed["pre_state_known_at"].notna()
            & (
                parsed["pre_state_known_at"]
                > parsed["information_anchor"]
            )
        ).any()
    ):
        raise DecisionContextV2Error(
            "sports or model pre-state is known after information anchor"
        )

    for column in (
        "pia_pbp_source_sha256",
        "fact_pbp_source_sha256",
        "state_pbp_source_sha256",
    ):
        for value in joined[column].dropna().astype(str).unique():
            _require_sha256(value, label=column)
    if not (
        joined["pia_pbp_source_sha256"].astype(str).eq(
            joined["fact_pbp_source_sha256"].astype(str)
        )
        & joined["pia_pbp_source_sha256"].astype(str).eq(
            joined["state_pbp_source_sha256"].astype(str)
        )
    ).all():
        raise DecisionContextV2Error("PBP source hashes disagree")

    quarter = pd.to_numeric(joined["quarter"], errors="coerce")
    if (
        quarter.isna().any()
        or (~quarter.between(1, 5)).any()
        or (~(quarter % 1).eq(0)).any()
    ):
        raise DecisionContextV2Error("quarter must be an integer in [1, 5]")
    joined["quarter"] = quarter.astype("int64")
    joined["is_overtime"] = joined["quarter"].gt(4)

    support_status = joined["prestate_support_status"].astype("string")
    if support_status.isna().any() or support_status.str.strip().eq("").any():
        raise DecisionContextV2Error("pre-state support status is missing")
    supported = support_status.eq("SUPPORTED")
    probability = pd.to_numeric(joined["_state_p_home"], errors="coerce")
    if (
        probability.loc[supported].isna().any()
        or (~probability.loc[supported].between(0.0, 1.0)).any()
    ):
        raise DecisionContextV2Error(
            "supported pre-state has invalid home probability"
        )
    if probability.loc[~supported].notna().any():
        raise DecisionContextV2Error(
            "unsupported pre-state carries a home probability"
        )
    if (
        joined.loc[supported, "pre_state_known_at"].isna().any()
        or joined.loc[supported, "state_input_sha256"].isna().any()
    ):
        raise DecisionContextV2Error(
            "supported pre-state lacks known-at or state-input hash"
        )
    if (supported & joined["is_overtime"]).any():
        raise DecisionContextV2Error(
            "overtime cannot carry supported Stage A probability"
        )
    for value in joined.loc[supported, "state_input_sha256"].astype(str):
        _require_sha256(value, label="state_input_sha256")
    for column in ("model_sha256", "model_manifest_sha256"):
        if joined[column].isna().any():
            raise DecisionContextV2Error("model binding is incomplete")
        for value in joined[column].astype(str).unique():
            _require_sha256(value, label=column)

    feature_key = ["game_id", "state_event_id", "feature_name"]
    if not feature_provenance.empty and (
        feature_provenance[feature_key].isna().any().any()
        or feature_provenance.duplicated(feature_key).any()
    ):
        raise DecisionContextV2Error("feature provenance key is invalid")
    provenance_groups = {
        (str(game_id), str(event_id)): group.sort_values(
            "feature_name", kind="mergesort"
        )
        for (game_id, event_id), group in feature_provenance.groupby(
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
                raise DecisionContextV2Error(
                    "supported pre-state must have exactly 11 PIT feature rows"
                )
            known = pd.to_datetime(
                group["feature_known_at"],
                utc=True,
                errors="coerce",
                format="mixed",
            )
            state_known = pd.Timestamp(row.pre_state_known_at)
            if (
                not group["PIT_status"].eq("PIT_VERIFIED").all()
                or known.isna().any()
                or (known > state_known).any()
                or not group["source_hash"].astype(str).eq(
                    str(row.pia_pbp_source_sha256)
                ).all()
            ):
                raise DecisionContextV2Error(
                    "supported feature provenance is not PIT/source verified"
                )
            feature_counts.append(11)
            feature_hashes.append(
                _canonical_sha256(group.to_dict("records"))
            )
        else:
            if group is not None and not group.empty:
                raise DecisionContextV2Error(
                    "unsupported pre-state carries model feature provenance"
                )
            feature_counts.append(0)
            feature_hashes.append(pd.NA)

    context = pd.DataFrame(
        {
            "schema_version": CONTEXT_SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "game_id": joined["game_id"].astype("string"),
            "information_anchor_id": joined[
                "information_anchor_id"
            ].astype("string"),
            "parent_episode_id_audit_only": joined[
                "parent_episode_id_audit_only"
            ].astype("string"),
            "event_id": joined["event_id"].astype("string"),
            "raw_play_id": joined["raw_play_id"].astype("string"),
            "order_sequence": joined["order_sequence"].astype("Int64"),
            "source_interval_start": parsed[
                "pia_source_interval_start"
            ].astype("string"),
            "source_interval_end": parsed[
                "pia_source_interval_end"
            ].astype("string"),
            "information_anchor": parsed["information_anchor"].astype("string"),
            "primary_selection_eligible": joined[
                "primary_selection_eligible"
            ].fillna(False).astype(bool),
            "censor_boundary_eligible": joined[
                "censor_boundary_eligible"
            ].fillna(False).astype(bool),
            "sports_prestate_known_at": parsed[
                "sports_prestate_known_at"
            ].astype("string"),
            "sports_prestate_known_at_basis": pd.Series(
                [
                    (
                        "SOURCE_INTERVAL_END_BOUND"
                        if missing
                        else "FACT_KNOWN_AT"
                    )
                    for missing in sports_known_was_missing
                ],
                dtype="string",
            ),
            "quarter": joined["quarter"].astype("int64"),
            "is_overtime": joined["is_overtime"].astype(bool),
            "game_clock": joined["game_clock"].astype("string"),
            "game_seconds_remaining": _finite_numeric(
                joined["game_seconds_remaining"]
            ).astype("Float64"),
            "home_team": joined["home_team"].astype("string"),
            "away_team": joined["away_team"].astype("string"),
            "possession_team": joined["possession_before"].astype("string"),
            "possession_is_home": joined["possession_is_home"].astype(
                "boolean"
            ),
            "pre_home_score": _finite_numeric(
                joined["pre_home_score"]
            ).astype("Float64"),
            "pre_away_score": _finite_numeric(
                joined["pre_away_score"]
            ).astype("Float64"),
            "score_margin_home": _finite_numeric(
                joined["score_margin_home"]
            ).astype("Float64"),
            "down": _finite_numeric(joined["down"]).astype("Float64"),
            "distance": _finite_numeric(joined["distance"]).astype("Float64"),
            "yardline_100": _finite_numeric(
                joined["yardline_100"]
            ).astype("Float64"),
            "goal_to_go": joined["goal_to_go"].astype("boolean"),
            "pre_red_zone": joined["pre_red_zone"].astype("boolean"),
            "home_timeouts_remaining": _finite_numeric(
                joined["home_timeouts_remaining"]
            ).astype("Float64"),
            "away_timeouts_remaining": _finite_numeric(
                joined["away_timeouts_remaining"]
            ).astype("Float64"),
            "p_before_home": probability.where(supported).astype("Float64"),
            "p_before_home_missing": (~supported).astype(bool),
            "prestate_support_status": support_status,
            "pre_state_known_at": parsed[
                "pre_state_known_at"
            ].astype("string"),
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
                "pia_pbp_source_sha256"
            ].astype("string"),
            "information_anchor_manifest_sha256": (
                information_anchor_manifest_sha256
            ),
            "information_anchor_object_sha256": (
                information_anchor_object_sha256
            ),
            "facts_manifest_sha256": facts_manifest_sha256,
            "facts_object_sha256": facts_object_sha256,
            "stage_a_manifest_sha256": stage_a_manifest_sha256,
            "state_predictions_object_sha256": (
                state_predictions_object_sha256
            ),
            "feature_provenance_object_sha256": (
                feature_provenance_object_sha256
            ),
        }
    )
    context["source_binding_sha256"] = context.apply(
        _row_source_binding, axis=1
    ).astype("string")
    context["prestate_context_sha256"] = context.apply(
        lambda row: compute_prestate_context_sha256(row.to_dict()),
        axis=1,
    ).astype("string")
    context = context.sort_values(
        ["game_id", "order_sequence", "information_anchor_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if (
        context.duplicated(list(_GRAIN)).any()
        or len(context) != len(information_anchors)
    ):
        raise DecisionContextV2Error("published PIA context grain is invalid")
    if _FORBIDDEN_COLUMNS.intersection(context.columns) or any(
        column.startswith(("post_", "final_")) for column in context.columns
    ):
        raise DecisionContextV2Error(
            "post-event or finalized outcome leaked into ContextV2"
        )

    audit = {
        "context_rows": len(context),
        "game_count": int(context["game_id"].nunique()),
        "information_anchor_expected_rows": len(information_anchors),
        "information_anchor_matched_rows": len(context),
        "primary_selection_rows": int(
            context["primary_selection_eligible"].sum()
        ),
        "censor_boundary_rows": int(
            context["censor_boundary_eligible"].sum()
        ),
        "regulation_rows": int((~context["is_overtime"]).sum()),
        "overtime_rows": int(context["is_overtime"].sum()),
        "supported_prestate_rows": int(supported.sum()),
        "unsupported_prestate_rows": int((~supported).sum()),
        "supported_feature_provenance_rows": int(sum(feature_counts)),
        "known_after_anchor_rows": 0,
        "sports_prestate_fact_known_at_rows": int(
            (~sports_known_was_missing).sum()
        ),
        "sports_prestate_source_interval_end_bound_rows": int(
            sports_known_was_missing.sum()
        ),
        "interval_identity_mismatch_rows": 0,
        "pbp_source_hash_mismatch_rows": 0,
        "duplicate_information_anchor_rows": 0,
        "silent_information_anchor_loss_rows": 0,
    }
    for value, count in support_status.value_counts(dropna=False).items():
        audit[f"prestate_status_{value}_rows"] = int(count)
    for field in _SPORTS_FIELDS:
        if field in context:
            audit[f"missing_{field}_rows"] = int(context[field].isna().sum())
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
        raise DecisionContextV2Error(str(exc)) from exc


def _read_information_anchor_publication(
    *,
    project_root: Path,
    path: Path,
    expected_file_sha256: str,
    expected_anchor_count: int,
    expected_game_count: int,
) -> tuple[pd.DataFrame, Mapping[str, Any], str]:
    expected = _require_sha256(
        expected_file_sha256,
        label="information_anchor_manifest_file_sha256",
    )
    try:
        resolved = _resolve_under(
            project_root,
            path,
            label="ProvisionalInformationAnchorV2 manifest",
        )
    except DevelopmentPanelError as exc:
        raise DecisionContextV2Error(str(exc)) from exc
    if not resolved.is_file() or _sha256_file(resolved) != expected:
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 manifest hash mismatch"
        )
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 manifest unreadable"
        ) from exc
    material = dict(manifest) if isinstance(manifest, dict) else {}
    declared_bundle = material.pop("bundle_sha256", None)
    if (
        manifest.get("schema") != _PIA_MANIFEST_SCHEMA
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("market_data_read") is not False
        or manifest.get("holdout_reaction_accessed") is not False
        or declared_bundle != _canonical_sha256(material)
        or manifest.get("counts", {}).get("information_anchor_count")
        != expected_anchor_count
    ):
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 manifest contract mismatch"
        )
    descriptors = [
        row
        for row in manifest.get("tables", [])
        if isinstance(row, dict)
        and row.get("name") == "provisional_information_anchors"
    ]
    if len(descriptors) != 1:
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 table descriptor missing"
        )
    descriptor = descriptors[0]
    object_sha = _require_sha256(
        descriptor.get("object_sha256"),
        label="provisional_information_anchors.object_sha256",
    )
    dataset_root = resolved.parents[3]
    candidates = (
        dataset_root / str(descriptor.get("object_path", "")),
        dataset_root.parent / str(descriptor.get("object_path", "")),
    )
    object_path = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate.resolve().is_file()
        ),
        None,
    )
    if (
        object_path is None
        or project_root.resolve() not in object_path.parents
        or object_path.stat().st_size != descriptor.get("byte_length")
        or _sha256_file(object_path) != object_sha
    ):
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 object verification failed"
        )
    try:
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != descriptor.get("row_count"):
            raise DecisionContextV2Error(
                "ProvisionalInformationAnchorV2 row-count mismatch"
            )
        anchors = parquet.read().to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 parquet unreadable"
        ) from exc
    if (
        list(anchors.columns) != descriptor.get("schema_columns")
        or len(anchors) != expected_anchor_count
        or anchors["game_id"].nunique() != expected_game_count
        or anchors["information_anchor_id"].duplicated().any()
    ):
        raise DecisionContextV2Error(
            "ProvisionalInformationAnchorV2 schema/cohort mismatch"
        )
    return anchors, manifest, object_sha


def prepare_exact153_decision_context_v2(
    *,
    project_root: Path,
    source_spec: DecisionContextV2SourceSpec | None = None,
) -> PreparedDecisionContextV2:
    """Verify all sources and build the exact-153 PIA context."""

    project = Path(project_root).resolve()
    spec = source_spec or DecisionContextV2SourceSpec()
    if (
        spec.expected_game_count != EXPECTED_GAME_COUNT
        or spec.expected_anchor_count != EXPECTED_ANCHOR_COUNT
    ):
        raise DecisionContextV2Error(
            "ContextV2 requires the frozen 153-game/25,250-anchor cohort"
        )
    anchors, pia_manifest, pia_object_sha = (
        _read_information_anchor_publication(
            project_root=project,
            path=spec.information_anchor_manifest_path,
            expected_file_sha256=(
                spec.information_anchor_manifest_file_sha256
            ),
            expected_anchor_count=spec.expected_anchor_count,
            expected_game_count=spec.expected_game_count,
        )
    )
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
        raise DecisionContextV2Error(str(exc)) from exc
    anchor_game_set = set(anchors["game_id"].astype(str))
    if not (
        anchor_game_set
        == set(facts_batch.games)
        == set(stage_a_batch.games)
    ):
        raise DecisionContextV2Error("ContextV2 source game sets differ")

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, int]] = []
    for game_id in sorted(anchor_game_set):
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
        except DevelopmentPanelError as exc:
            raise DecisionContextV2Error(str(exc)) from exc
        batch_model = stage_a_batch.document.get("model")
        if (
            not isinstance(batch_model, dict)
            or stage_manifest.get("model") != batch_model
        ):
            raise DecisionContextV2Error(
                f"{game_id} Stage A model binding mismatch"
            )
        for column, model_field in (
            ("model_id", "model_id"),
            ("model_version", "model_version"),
            ("model_sha256", "model_sha256"),
            ("model_manifest_sha256", "manifest_sha256"),
        ):
            if states[column].astype(str).ne(
                str(batch_model[model_field])
            ).any():
                raise DecisionContextV2Error(
                    f"{game_id} state prediction model binding mismatch"
                )
        context, audit = build_decision_context_v2_frame(
            information_anchors=anchors.loc[
                anchors["game_id"].astype(str).eq(game_id)
            ].copy(),
            facts=facts,
            state_predictions=states,
            feature_provenance=features,
            information_anchor_object_sha256=pia_object_sha,
            facts_object_sha256=str(
                _descriptor(
                    facts_manifest, "canonical_factor_events"
                )["object_sha256"]
            ),
            state_predictions_object_sha256=str(
                _descriptor(
                    stage_manifest, "state_predictions"
                )["object_sha256"]
            ),
            feature_provenance_object_sha256=str(
                _descriptor(
                    stage_manifest, "feature_provenance"
                )["object_sha256"]
            ),
            information_anchor_manifest_sha256=(
                spec.information_anchor_manifest_file_sha256
            ),
            facts_manifest_sha256=str(
                facts_batch.games[game_id]["manifest_sha256"]
            ),
            stage_a_manifest_sha256=str(
                stage_a_batch.games[game_id]["manifest_sha256"]
            ),
        )
        frames.append(context)
        audits.append(audit)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["game_id", "order_sequence", "information_anchor_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if (
        len(combined) != spec.expected_anchor_count
        or combined["game_id"].nunique() != spec.expected_game_count
        or combined.duplicated(list(_GRAIN)).any()
    ):
        raise DecisionContextV2Error(
            "exact-153 combined ContextV2 grain/count is invalid"
        )
    aggregate: dict[str, int] = {}
    for audit in audits:
        for key, value in audit.items():
            if key == "game_count":
                continue
            aggregate[key] = aggregate.get(key, 0) + int(value)
    aggregate["game_count"] = int(combined["game_id"].nunique())
    if (
        aggregate["information_anchor_expected_rows"]
        != spec.expected_anchor_count
        or aggregate["information_anchor_matched_rows"]
        != spec.expected_anchor_count
        or aggregate["silent_information_anchor_loss_rows"] != 0
        or aggregate["known_after_anchor_rows"] != 0
        or aggregate["pbp_source_hash_mismatch_rows"] != 0
    ):
        raise DecisionContextV2Error(
            "exact-153 ContextV2 reconciliation failed"
        )
    return PreparedDecisionContextV2(
        context=combined,
        audit_counts=aggregate,
        information_anchor_manifest_file_sha256=(
            spec.information_anchor_manifest_file_sha256
        ),
        information_anchor_bundle_sha256=str(pia_manifest["bundle_sha256"]),
        facts_batch_file_sha256=spec.facts_batch_file_sha256,
        facts_batch_sha256=str(facts_batch.document["batch_sha256"]),
        stage_a_batch_file_sha256=spec.stage_a_batch_file_sha256,
        stage_a_batch_sha256=str(stage_a_batch.document["batch_sha256"]),
        game_count=spec.expected_game_count,
        anchor_count=spec.expected_anchor_count,
    )


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise DecisionContextV2Error(
            "ContextV2 cannot be encoded as Parquet"
        ) from exc
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
            raise DecisionContextV2Error(
                f"content-addressed collision: {path}"
            )
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
        raise DecisionContextV2Error(
            f"content-addressed publish race: {path}"
        )


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


def publish_decision_context_v2_tables(
    *,
    output_root: Path,
    context: pd.DataFrame,
    audit_counts: Mapping[str, int],
    information_anchor_manifest_file_sha256: str,
    information_anchor_bundle_sha256: str,
    facts_batch_file_sha256: str,
    facts_batch_sha256: str,
    stage_a_batch_file_sha256: str,
    stage_a_batch_sha256: str,
    expected_game_count: int = EXPECTED_GAME_COUNT,
    expected_anchor_count: int = EXPECTED_ANCHOR_COUNT,
) -> PublishedDecisionContextV2:
    """Publish immutable ContextV2 Parquet, audit, and manifest objects."""

    for label, value in (
        (
            "information_anchor_manifest_file_sha256",
            information_anchor_manifest_file_sha256,
        ),
        ("information_anchor_bundle_sha256", information_anchor_bundle_sha256),
        ("facts_batch_file_sha256", facts_batch_file_sha256),
        ("facts_batch_sha256", facts_batch_sha256),
        ("stage_a_batch_file_sha256", stage_a_batch_file_sha256),
        ("stage_a_batch_sha256", stage_a_batch_sha256),
    ):
        _require_sha256(value, label=label)
    if (
        not isinstance(context, pd.DataFrame)
        or context.empty
        or len(context) != expected_anchor_count
        or context["game_id"].nunique() != expected_game_count
        or context.duplicated(list(_GRAIN)).any()
        or context["schema_version"].astype(str).ne(CONTEXT_SCHEMA).any()
    ):
        raise DecisionContextV2Error(
            "ContextV2 publication grain/count is invalid"
        )
    if _FORBIDDEN_COLUMNS.intersection(context.columns) or any(
        column.startswith(("post_", "final_")) for column in context.columns
    ):
        raise DecisionContextV2Error(
            "forbidden future/final column cannot be published"
        )
    if (
        int(audit_counts.get("information_anchor_matched_rows", -1))
        != expected_anchor_count
        or int(audit_counts.get("known_after_anchor_rows", -1)) != 0
        or int(audit_counts.get("silent_information_anchor_loss_rows", -1))
        != 0
    ):
        raise DecisionContextV2Error(
            "ContextV2 publication audit is not green"
        )

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
        "panel_v4_status": "BLOCKED_PENDING_CONTEXT_REVIEW",
        "counts": {
            key: int(value) for key, value in sorted(audit_counts.items())
        },
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
        "experiment_id": EXPERIMENT_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "cohort": "development",
        "publication_gate": "PASS",
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "panel_v4_status": "BLOCKED_PENDING_CONTEXT_REVIEW",
        "game_count": int(context["game_id"].nunique()),
        "information_anchor_count": len(context),
        "audit_counts": {
            key: int(value) for key, value in sorted(audit_counts.items())
        },
        "model_feature_boundary": {
            "allowed_identity_unit": "information_anchor_id",
            "parent_episode_column": "parent_episode_id_audit_only",
            "parent_episode_predictor_allowed": False,
            "known_at_rule": (
                "sports_prestate_known_at_and_pre_state_known_at"
                "<=information_anchor"
            ),
            "forbidden_column_prefixes": ["post_", "final_"],
            "forbidden_columns": sorted(_FORBIDDEN_COLUMNS),
            "row_feature_lock": {
                "column": "prestate_context_sha256",
                "algorithm": "CANONICAL_JSON_SHA256",
                "coverage": (
                    "information_anchor_id+information_anchor+all_sports_"
                    "prestate+p_before/support/known_at+source/state/model/"
                    "feature_provenance_hashes"
                ),
            },
        },
        "sources": {
            "provisional_information_anchor_v2": {
                "manifest_file_sha256": (
                    information_anchor_manifest_file_sha256
                ),
                "bundle_sha256": information_anchor_bundle_sha256,
            },
            "facts_v4": {
                "batch_file_sha256": facts_batch_file_sha256,
                "batch_sha256": facts_batch_sha256,
            },
            "stage_a": {
                "batch_file_sha256": stage_a_batch_file_sha256,
                "batch_sha256": stage_a_batch_sha256,
            },
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
    return PublishedDecisionContextV2(
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
        primary_selection_rows=int(
            audit_counts["primary_selection_rows"]
        ),
        censor_boundary_rows=int(audit_counts["censor_boundary_rows"]),
        supported_prestate_rows=int(
            audit_counts["supported_prestate_rows"]
        ),
    )


def publish_exact153_decision_context_v2(
    *,
    project_root: Path,
    output_root: Path | None = None,
    source_spec: DecisionContextV2SourceSpec | None = None,
) -> PublishedDecisionContextV2:
    """Prepare and atomically publish the exact-153 ContextV2."""

    project = Path(project_root).resolve()
    prepared = prepare_exact153_decision_context_v2(
        project_root=project,
        source_spec=source_spec,
    )
    try:
        output = _resolve_under(
            project,
            output_root or DEFAULT_OUTPUT_ROOT,
            label="ContextV2 output",
        )
    except DevelopmentPanelError as exc:
        raise DecisionContextV2Error(str(exc)) from exc
    return publish_decision_context_v2_tables(
        output_root=output,
        context=prepared.context,
        audit_counts=prepared.audit_counts,
        information_anchor_manifest_file_sha256=(
            prepared.information_anchor_manifest_file_sha256
        ),
        information_anchor_bundle_sha256=(
            prepared.information_anchor_bundle_sha256
        ),
        facts_batch_file_sha256=prepared.facts_batch_file_sha256,
        facts_batch_sha256=prepared.facts_batch_sha256,
        stage_a_batch_file_sha256=prepared.stage_a_batch_file_sha256,
        stage_a_batch_sha256=prepared.stage_a_batch_sha256,
        expected_game_count=prepared.game_count,
        expected_anchor_count=prepared.anchor_count,
    )


__all__ = [
    "AUDIT_SCHEMA",
    "BUILDER_VERSION",
    "CLAIM_BOUNDARY",
    "CONTEXT_SCHEMA",
    "DEFAULT_INFORMATION_ANCHOR_MANIFEST",
    "DEFAULT_INFORMATION_ANCHOR_MANIFEST_FILE_SHA256",
    "DEFAULT_OUTPUT_ROOT",
    "DecisionContextV2Error",
    "DecisionContextV2SourceSpec",
    "MANIFEST_SCHEMA",
    "PreparedDecisionContextV2",
    "PublishedDecisionContextV2",
    "build_decision_context_v2_frame",
    "compute_prestate_context_sha256",
    "prepare_exact153_decision_context_v2",
    "publish_decision_context_v2_tables",
    "publish_exact153_decision_context_v2",
]
