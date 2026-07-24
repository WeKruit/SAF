"""Frozen X-11 reproduction for the official nflverse no-spread booster.

The runner is deliberately offline and preliminary.  It does not train or
calibrate a model, read a spread, join prediction-market data, or claim PIT.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.experiments import (
    load_experiment_registry,
    validate_result_ref,
)
from prediction_market.models.nfl_fastrmodels import (
    ASSET_ID,
    ARCHIVE_TAG_COMMIT,
    FEATURE_NAMES,
    NoSpreadModelInput,
    OfficialNoSpreadPredictor,
    load_official_no_spread_predictor,
)
from prediction_market.models.validation import evaluate_probabilities
from prediction_market.sports import nfl_game_state as nfl_state
from prediction_market.sports import nfl_season_census as season_census
from prediction_market.sports import x11 as x11_pipeline
from prediction_market.static_store import read_verified_static_object


EXPERIMENT_ID: Final = "X-11"
REPRODUCTION_ID: Final = "REPRO-X11-NFL-FASTRMODELS-V2"
REPRODUCTION_SCOPE: Final = "team_h_nfl_fastrmodels_reproduction_v2"
MODEL_ID: Final = "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1"
DATASET_IDS: Final = ("DS-NFL-FASTRMODELS", "DS-NFLVERSE")
SEASONS: Final = (2021, 2022, 2023, 2024, 2025)
SEASON_TYPES: Final = ("POST", "REG")
REGULATION_PERIODS: Final = (1, 2, 3, 4)
BOOTSTRAP_SAMPLES: Final = 200
BOOTSTRAP_SEED: Final = 20260723
CONFIDENCE_LEVEL: Final = 0.95
MINIMUM_VALID_BOOTSTRAP_SAMPLES: Final = 100
RELIABILITY_EDGES: Final = tuple(index / 10 for index in range(11))

EXPECTED_REGISTRATION_HEAD_SHA256: Final = (
    "sha256:"
    "0dcd4a1a62c7790967023b2383a2cb93eaf35b25e3e4d64baabe8decb8f45960"
)
EXPECTED_V1_REGISTRATION_HEAD_SHA256: Final = (
    "sha256:"
    "ad594d2aa06ff7ecc99ba4389d53c2973f1aba8bc922bba45a7c0cedc3ed6177"
)
EXPECTED_AMENDMENT_AT: Final = "2026-07-24T00:41:31Z"
EXPECTED_CODE_SHA256: Final = (
    "sha256:"
    "3c1b92679df15dd6a8bce1d4c4bddbcf5c81944f59b3fe85ded7e0e315161e75"
)
EXPECTED_DATA_SHA256: Final = (
    "sha256:"
    "6342fceb33fba8b7f2f3b601f85e11a8e201898419013d0e1a46f9f66623fbc4"
)
EXPECTED_PROTOCOL_SHA256: Final = (
    "sha256:"
    "3d75a366ed6e9627d50b4831ace348c63890160e2ead8241fddb0ce0fb917bdd"
)
EXPECTED_REPRODUCTION_SPEC_SHA256: Final = (
    "sha256:"
    "82d81413d3fe851003b64dd014d45a83fcb881ce04487134596fe342dfa4e1d1"
)
EXPECTED_MODEL_RECORD_SHA256: Final = (
    "sha256:"
    "df098fb7c050669f445bf7115c4575d3409fa591e57ea7eddff511d84f2cf3d3"
)
EXPECTED_CENSUS: Final = {
    2021: {
        "raw_rows": 49922,
        "games": 285,
        "eligible_rows": 41342,
        "eligible_non_tie_games": 284,
    },
    2022: {
        "raw_rows": 49434,
        "games": 284,
        "eligible_rows": 40937,
        "eligible_non_tie_games": 282,
    },
    2023: {
        "raw_rows": 49665,
        "games": 285,
        "eligible_rows": 41698,
        "eligible_non_tie_games": 285,
    },
    2024: {
        "raw_rows": 49492,
        "games": 285,
        "eligible_rows": 41269,
        "eligible_non_tie_games": 285,
    },
    2025: {
        "raw_rows": 48771,
        "games": 285,
        "eligible_rows": 40361,
        "eligible_non_tie_games": 284,
    },
}
EXPECTED_TOTAL_CENSUS: Final = {
    "raw_rows": 247284,
    "games": 1424,
    "eligible_rows": 205607,
    "eligible_non_tie_games": 1420,
}

FULL_PATH_INCLUDED_STAGES: Final = (
    "normalized_event_construction",
    "state_event_transition",
    "official_feature_projection",
    "preloaded_official_booster",
    "probability_output_validation",
)
LATENCY_EXCLUDED_STAGES: Final = (
    "io",
    "market_join",
    "network",
    "registry_loading",
)

_ELIGIBLE_REQUIRED_FIELDS: Final = (
    "ep",
    "score_differential",
    "play_type",
    "posteam",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
)
_EVALUATION_COLUMNS: Final = (
    "game_id",
    "season",
    "season_type",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "qtr",
    "order_sequence",
    "ep",
    "play_type",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "home_opening_kickoff",
    "result",
    "home_wp",
)
_METRIC_ID_COLUMNS: Final = (
    "season",
    "game_id",
    "order_sequence",
    "raw_record_ordinal",
    "row_key",
    "final_outcome",
    "home_win",
    "home_probability",
)
_PROBABILITY_EPSILON: Final = 1e-9


class X11FastrmodelsError(ValueError):
    """The frozen official no-spread reproduction failed closed."""


@dataclass(frozen=True, slots=True)
class ReproductionGate:
    evaluation_started_at: datetime
    amended_at: datetime
    registration_head_sha256: str
    scope: str
    code_sha256: str
    data_sha256: str
    dataset_ids: tuple[str, ...]
    model_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedPartition:
    frame: pd.DataFrame
    census: Mapping[str, int]
    manifest_sha256: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class FullPathFixture:
    prior_state: nfl_state.NFLGameState
    source_row: Mapping[str, object]
    successor_rows: tuple[Mapping[str, object], ...]
    sequence: int
    raw_object_sha256: str
    source_version: str
    second_half_receiver: str


@dataclass(frozen=True, slots=True)
class LoadedEvaluation:
    frame: pd.DataFrame
    census: Mapping[str, object]
    input_manifest_bindings: tuple[Mapping[str, object], ...]
    evaluation_input_sha256: str
    full_path_fixture: FullPathFixture


def canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def metrics_bytes(value: Mapping[str, object]) -> bytes:
    return _canonical_bytes(dict(value))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_utc(value: datetime) -> str:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset().total_seconds() != 0
    ):
        raise X11FastrmodelsError(
            "evaluation_started_at must be timezone-aware UTC"
        )
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise X11FastrmodelsError(f"{field} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise X11FastrmodelsError(f"{field} must be canonical UTC") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise X11FastrmodelsError(f"{field} must be canonical UTC")
    return parsed


def verify_reproduction_gate(
    program_root: str | Path,
    *,
    evaluation_started_at: datetime,
) -> ReproductionGate:
    """Validate the exact committed Team H scope before reading evaluation rows."""

    started_at_text = _canonical_utc(evaluation_started_at)
    card = load_experiment_registry(program_root)[EXPERIMENT_ID]
    if len(card["amendments"]) < 2:
        raise X11FastrmodelsError(
            "X-11 V2 reproduction requires its two frozen registration amendments"
        )
    v1_amendment, amendment = card["amendments"][:2]
    if (
        v1_amendment["amendment_sha256"]
        != EXPECTED_V1_REGISTRATION_HEAD_SHA256
        or amendment["prior_sha256"]
        != EXPECTED_V1_REGISTRATION_HEAD_SHA256
    ):
        raise X11FastrmodelsError("X-11 V1 to V2 amendment chain differs")
    if (
        amendment["amendment_sha256"]
        != EXPECTED_REGISTRATION_HEAD_SHA256
        or amendment["amended_at"] != EXPECTED_AMENDMENT_AT
        or amendment["approved_by"] != "H"
    ):
        raise X11FastrmodelsError("X-11 Team H amendment identity differs")
    supersession = amendment["changes"].get("supersede_reproduction")
    if (
        not isinstance(supersession, dict)
        or supersession.get("supersedes_reproduction_id")
        != "REPRO-X11-NFL-FASTRMODELS-V1"
    ):
        raise X11FastrmodelsError("X-11 V1 supersession identity differs")
    registration = supersession.get("registration")
    if not isinstance(registration, dict):
        raise X11FastrmodelsError("X-11 V2 reproduction registration is absent")
    expected_registration = {
        "reproduction_id": REPRODUCTION_ID,
        "scope": REPRODUCTION_SCOPE,
        "result_class": "poc",
        "dataset_ids": list(DATASET_IDS),
        "model_bindings": [
            {
                "model_id": MODEL_ID,
                "model_version": "v1",
                "model_record_sha256": EXPECTED_MODEL_RECORD_SHA256,
            }
        ],
        "code_paths": [
            "src/prediction_market/models/nfl.py",
            "src/prediction_market/models/nfl_fastrmodels.py",
        ],
        "code_sha256": EXPECTED_CODE_SHA256,
        "data_sha256": EXPECTED_DATA_SHA256,
        "protocol_path": (
            "registries/protocols/x11_fastrmodels_no_spread_v1.json"
        ),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "reproduction_spec_sha256": EXPECTED_REPRODUCTION_SPEC_SHA256,
    }
    if registration != expected_registration:
        raise X11FastrmodelsError("X-11 reproduction registration differs")
    expected_scope = {
        "authorized": True,
        "required_result_label": "PRELIMINARY",
        "required_lock_ids": [f"reproduction:{REPRODUCTION_ID}"],
        "input_binding": {
            "result_class": "poc",
            "dataset_ids": list(DATASET_IDS),
            "model_ids": [MODEL_ID],
            "synthetic_data_sha256": None,
        },
    }
    scope = card["authorization_scopes"].get(REPRODUCTION_SCOPE)
    if scope != expected_scope:
        raise X11FastrmodelsError("X-11 effective reproduction scope differs")
    v1_scope = card["authorization_scopes"].get(
        "team_h_nfl_fastrmodels_reproduction_v1"
    )
    if not isinstance(v1_scope, dict) or v1_scope.get("authorized") is not False:
        raise X11FastrmodelsError("X-11 V1 reproduction remains authorized")
    expected_inputs = {
        "code_sha256": EXPECTED_CODE_SHA256,
        "data_sha256": EXPECTED_DATA_SHA256,
        "dataset_ids": list(DATASET_IDS),
        "model_ids": [MODEL_ID],
        "registered_at": EXPECTED_AMENDMENT_AT,
    }
    if card["preregistered_inputs"].get(REPRODUCTION_SCOPE) != expected_inputs:
        raise X11FastrmodelsError("X-11 preregistered input binding differs")
    amended_at = _parse_utc(amendment["amended_at"], "amended_at")
    if evaluation_started_at <= amended_at:
        raise X11FastrmodelsError(
            "evaluation must start strictly after the Team H amendment"
        )
    if evaluation_started_at > datetime.now(timezone.utc):
        raise X11FastrmodelsError("evaluation_started_at cannot be future-dated")
    if started_at_text <= amendment["amended_at"]:
        raise X11FastrmodelsError(
            "evaluation timestamp does not follow the Team H amendment"
        )
    return ReproductionGate(
        evaluation_started_at=evaluation_started_at,
        amended_at=amended_at,
        registration_head_sha256=EXPECTED_REGISTRATION_HEAD_SHA256,
        scope=REPRODUCTION_SCOPE,
        code_sha256=EXPECTED_CODE_SHA256,
        data_sha256=EXPECTED_DATA_SHA256,
        dataset_ids=DATASET_IDS,
        model_ids=(MODEL_ID,),
    )


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float, na_value=np.nan)
        result.loc[~np.isfinite(values), column] = np.nan
    return result


def _require_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise X11FastrmodelsError(
            "evaluation partition is missing columns: " + ", ".join(missing)
        )


def prepare_partition_frame(
    native: pd.DataFrame,
    *,
    season: int,
    manifest_sha256: str,
    object_sha256: str,
) -> PreparedPartition:
    """Filter one verified native season using the preregistered row contract."""

    if not isinstance(native, pd.DataFrame) or native.empty:
        raise X11FastrmodelsError("native evaluation partition must be nonempty")
    if season not in SEASONS:
        raise X11FastrmodelsError("evaluation season is outside 2021-2025")
    _require_columns(
        native,
        (*_EVALUATION_COLUMNS, "_raw_record_ordinal"),
    )
    frame = native.copy()
    raw_rows = len(frame)
    if frame["game_id"].isna().any():
        raise X11FastrmodelsError("native game_id is missing")
    frame["game_id"] = frame["game_id"].astype(str)
    if (frame["game_id"].str.strip() == "").any():
        raise X11FastrmodelsError("native game_id is empty")
    games = int(frame["game_id"].nunique())
    numeric_columns = (
        "season",
        "qtr",
        "order_sequence",
        "_raw_record_ordinal",
        "ep",
        "score_differential",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "home_opening_kickoff",
        "result",
        "home_wp",
    )
    frame = _finite_numeric(frame, numeric_columns)
    if frame["season"].isna().any() or not frame["season"].eq(season).all():
        raise X11FastrmodelsError("native season differs from partition")
    if frame["_raw_record_ordinal"].isna().any():
        raise X11FastrmodelsError("raw record ordinal is required")
    ordinals = frame["_raw_record_ordinal"]
    if (
        (ordinals < 0).any()
        or not ordinals.map(float.is_integer).all()
        or ordinals.duplicated().any()
    ):
        raise X11FastrmodelsError(
            "raw record ordinals must be unique nonnegative integers"
        )
    structural = (
        "home_team",
        "away_team",
        "home_opening_kickoff",
        "result",
        "order_sequence",
    )
    if frame[list(structural)].isna().any().any():
        raise X11FastrmodelsError("native game/model identity is incomplete")
    if not set(frame["season_type"].dropna().unique()) <= set(SEASON_TYPES):
        raise X11FastrmodelsError("native season_type is outside REG/POST")
    opening = frame["home_opening_kickoff"]
    if (
        not opening.isin([0.0, 1.0]).all()
        or not opening.map(float.is_integer).all()
    ):
        raise X11FastrmodelsError(
            "home_opening_kickoff must be a binary native observation"
        )
    for column in (
        "season",
        "season_type",
        "home_team",
        "away_team",
        "home_opening_kickoff",
        "result",
    ):
        if not frame.groupby("game_id", sort=False)[column].nunique(
            dropna=False
        ).eq(1).all():
            raise X11FastrmodelsError(
                f"native {column} must be constant within game"
            )

    required_present = frame[list(_ELIGIBLE_REQUIRED_FIELDS)].notna().all(
        axis=1
    )
    required_text = (
        frame["play_type"].map(
            lambda value: type(value) is str
            and bool(value)
            and value == value.strip()
        )
        & frame["posteam"].map(lambda value: type(value) is str and bool(value))
    )
    regulation = frame["qtr"].isin(REGULATION_PERIODS)
    participant = (
        (
            (frame["posteam"] == frame["home_team"])
            & (frame["defteam"] == frame["away_team"])
        )
        | (
            (frame["posteam"] == frame["away_team"])
            & (frame["defteam"] == frame["home_team"])
        )
    )
    eligible = required_present & required_text & regulation & participant
    selected = frame.loc[eligible].copy()
    if selected.empty:
        raise X11FastrmodelsError("partition has no eligible evaluation rows")
    selected["season"] = selected["season"].astype(int)
    selected["qtr"] = selected["qtr"].astype(int)
    selected["order_sequence"] = selected["order_sequence"].astype(int)
    selected["raw_record_ordinal"] = selected["_raw_record_ordinal"].astype(
        int
    )
    selected["final_outcome"] = "tie"
    selected.loc[selected["result"] > 0, "final_outcome"] = "home_win"
    selected.loc[selected["result"] < 0, "final_outcome"] = "away_win"
    selected["home_win"] = pd.Series(
        [
            1
            if outcome == "home_win"
            else 0
            if outcome == "away_win"
            else pd.NA
            for outcome in selected["final_outcome"]
        ],
        index=selected.index,
        dtype="Int64",
    )
    selected["manifest_sha256"] = manifest_sha256
    selected["object_sha256"] = object_sha256
    selected["row_key"] = [
        f"{season}:{game_id}:{order}:{ordinal}"
        for game_id, order, ordinal in zip(
            selected["game_id"],
            selected["order_sequence"],
            selected["raw_record_ordinal"],
            strict=True,
        )
    ]
    selected = selected.sort_values(
        [
            "season",
            "game_id",
            "order_sequence",
            "raw_record_ordinal",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    if selected["row_key"].duplicated().any():
        raise X11FastrmodelsError("eligible row keys must be unique")
    non_tie_games = int(
        selected.loc[
            selected["final_outcome"] != "tie", "game_id"
        ].nunique()
    )
    non_tie_rows = int((selected["final_outcome"] != "tie").sum())
    census = {
        "raw_rows": raw_rows,
        "games": games,
        "eligible_rows": non_tie_rows,
        "eligible_non_tie_games": non_tie_games,
    }
    return PreparedPartition(
        frame=selected,
        census=census,
        manifest_sha256=manifest_sha256,
        object_sha256=object_sha256,
    )


def _mapping_value(row: Mapping[str, object], field: str) -> object:
    if field not in row:
        raise X11FastrmodelsError(f"model row is missing {field}")
    value = row[field]
    as_py = getattr(value, "as_py", None)
    return as_py() if callable(as_py) else value


def _float_value(row: Mapping[str, object], field: str) -> float:
    value = _mapping_value(row, field)
    if isinstance(value, bool):
        raise X11FastrmodelsError(f"{field} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise X11FastrmodelsError(f"{field} must be finite numeric") from error
    if not math.isfinite(number):
        raise X11FastrmodelsError(f"{field} must be finite numeric")
    return number


def project_no_spread_input(
    row: Mapping[str, object],
) -> NoSpreadModelInput:
    """Project one eligible native state with the pinned helper definition."""

    home_team = str(_mapping_value(row, "home_team"))
    away_team = str(_mapping_value(row, "away_team"))
    posteam = str(_mapping_value(row, "posteam"))
    qtr = int(_float_value(row, "qtr"))
    opening = int(_float_value(row, "home_opening_kickoff"))
    if opening not in {0, 1}:
        raise X11FastrmodelsError("home_opening_kickoff must be binary")
    second_half_receiver = away_team if opening == 1 else home_team
    receive_second_half_kickoff = float(
        qtr <= 2 and posteam == second_half_receiver
    )
    game_seconds_remaining = _float_value(
        row, "game_seconds_remaining"
    )
    score_differential = _float_value(row, "score_differential")
    elapsed_share = (3600.0 - game_seconds_remaining) / 3600.0
    diff_time_ratio = score_differential / math.exp(-4.0 * elapsed_share)
    values = (
        receive_second_half_kickoff,
        float(posteam == home_team),
        _float_value(row, "half_seconds_remaining"),
        game_seconds_remaining,
        diff_time_ratio,
        score_differential,
        _float_value(row, "down"),
        _float_value(row, "ydstogo"),
        _float_value(row, "yardline_100"),
        _float_value(row, "posteam_timeouts_remaining"),
        _float_value(row, "defteam_timeouts_remaining"),
    )
    return NoSpreadModelInput(
        feature_names=FEATURE_NAMES,
        feature_values=values,
        posteam=posteam,
        home_team=home_team,
        away_team=away_team,
        period=qtr,
    )


def _input_from_state(
    state: nfl_state.NFLGameState,
    *,
    second_half_receiver: str,
) -> NoSpreadModelInput:
    if (
        state.period not in REGULATION_PERIODS
        or state.terminal
        or state.suspended
        or state.possession_team is None
        or state.down is None
        or state.distance is None
        or state.yardline_100 is None
    ):
        raise X11FastrmodelsError(
            "reducer state is not eligible for official regulation features"
        )
    possession_is_home = state.possession_team == state.home_team
    score_differential = (
        state.home_score - state.away_score
        if possession_is_home
        else state.away_score - state.home_score
    )
    posteam_timeouts = (
        state.home_timeouts_remaining
        if possession_is_home
        else state.away_timeouts_remaining
    )
    defteam_timeouts = (
        state.away_timeouts_remaining
        if possession_is_home
        else state.home_timeouts_remaining
    )
    half_seconds_remaining = state.period_seconds_remaining + (
        900 if state.period in {1, 3} else 0
    )
    elapsed_share = (3600.0 - state.game_seconds_remaining) / 3600.0
    score = float(score_differential)
    values = (
        float(
            state.period <= 2
            and state.possession_team == second_half_receiver
        ),
        float(possession_is_home),
        float(half_seconds_remaining),
        float(state.game_seconds_remaining),
        score / math.exp(-4.0 * elapsed_share),
        score,
        float(state.down),
        float(state.distance),
        float(state.yardline_100),
        float(posteam_timeouts),
        float(defteam_timeouts),
    )
    return NoSpreadModelInput(
        feature_names=FEATURE_NAMES,
        feature_values=values,
        posteam=state.possession_team,
        home_team=state.home_team,
        away_team=state.away_team,
        period=state.period,
    )


def _validate_probability(value: object) -> float:
    if isinstance(value, bool):
        raise X11FastrmodelsError("official probability must be finite")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise X11FastrmodelsError(
            "official probability must be finite"
        ) from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise X11FastrmodelsError(
            "official probability must be finite in [0, 1]"
        )
    return probability


def predict_frame(
    frame: pd.DataFrame,
    predictor: OfficialNoSpreadPredictor,
    *,
    batch_size: int = 8192,
) -> pd.DataFrame:
    if type(batch_size) is not int or batch_size < 1:
        raise X11FastrmodelsError("batch_size must be positive")
    result = frame.copy()
    probabilities: list[float] = []
    records = result.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        inputs = tuple(
            project_no_spread_input(row)
            for row in records[start : start + batch_size]
        )
        possession_probabilities = predictor.predict_possession_batch(inputs)
        for model_input, possession_probability in zip(
            inputs,
            possession_probabilities,
            strict=True,
        ):
            oriented = (
                possession_probability
                if model_input.posteam == model_input.home_team
                else 1.0 - possession_probability
            )
            probabilities.append(_validate_probability(oriented))
    if len(probabilities) != len(result):
        raise X11FastrmodelsError("official prediction count mismatch")
    result["home_probability"] = probabilities
    return result


def binary_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "final_outcome" not in frame or "home_win" not in frame:
        raise X11FastrmodelsError("evaluation frame lacks final result labels")
    return frame.loc[frame["final_outcome"] != "tie"].copy()


def binary_labels(frame: pd.DataFrame) -> pd.Series:
    metric_frame = binary_metric_frame(frame)
    return metric_frame["home_win"].astype(int)


def _game_macro_point(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    clipped = np.clip(probability, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
    losses = pd.DataFrame(
        {
            "game_id": groups,
            "brier": (probability - y) ** 2,
            "log_loss": -(
                y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
            ),
        }
    )
    games = losses.groupby("game_id", sort=True)[["brier", "log_loss"]].mean()
    return (
        {
            "brier": float(games["brier"].mean()),
            "log_loss": float(games["log_loss"].mean()),
            "clusters": len(games),
        },
        games,
    )


def _game_macro_bootstrap(
    games: pd.DataFrame,
) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = games[["brier", "log_loss"]].to_numpy(dtype=float)
    samples: dict[str, list[float]] = {"brier": [], "log_loss": []}
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = rng.choice(len(values), size=len(values), replace=True)
        sampled = values[selected].mean(axis=0)
        samples["brier"].append(float(sampled[0]))
        samples["log_loss"].append(float(sampled[1]))
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "bootstrap_ci": {
            key: [
                float(np.quantile(values_, alpha)),
                float(np.quantile(values_, 1.0 - alpha)),
            ]
            for key, values_ in samples.items()
        },
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_samples_requested": BOOTSTRAP_SAMPLES,
        "bootstrap_samples_valid": BOOTSTRAP_SAMPLES,
    }


def _reliability(
    y: np.ndarray,
    probability: np.ndarray,
) -> list[dict[str, object]]:
    bins: list[dict[str, object]] = []
    for index, (lower, upper) in enumerate(
        zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:], strict=True)
    ):
        selected = (
            (probability >= lower)
            & (
                probability <= upper
                if index == len(RELIABILITY_EDGES) - 2
                else probability < upper
            )
        )
        count = int(selected.sum())
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": (
                    None
                    if count == 0
                    else float(probability[selected].mean())
                ),
                "observed_home_win_rate": (
                    None if count == 0 else float(y[selected].mean())
                ),
            }
        )
    return bins


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise X11FastrmodelsError("artifact cannot contain non-finite numbers")
    return value


def _metric_scope(frame: pd.DataFrame) -> dict[str, object]:
    y = binary_labels(frame).to_numpy(dtype=int)
    metric = binary_metric_frame(frame)
    probability = metric["home_probability"].to_numpy(dtype=float)
    groups = metric["game_id"].to_numpy(dtype=object)
    row_micro = evaluate_probabilities(
        y,
        probability,
        groups=groups,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        minimum_valid_samples=MINIMUM_VALID_BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    game_macro, games = _game_macro_point(y, probability, groups)
    game_macro.update(_game_macro_bootstrap(games))
    return {
        "row_micro": _json_ready(row_micro),
        "game_macro": game_macro,
        "reliability": _reliability(y, probability),
    }


def _source_home_wp_diagnostic(frame: pd.DataFrame) -> dict[str, object]:
    source = pd.to_numeric(frame["home_wp"], errors="coerce").to_numpy(
        dtype=float
    )
    model = frame["home_probability"].to_numpy(dtype=float)
    valid = np.isfinite(source)
    delta = np.abs(model[valid] - source[valid])
    if len(delta) == 0:
        raise X11FastrmodelsError("source home_wp diagnostic has no valid rows")
    return {
        "role": "diagnostic_only_not_accuracy_oracle",
        "observations": len(delta),
        "missing_source_rows": int((~valid).sum()),
        "mean_absolute_delta": float(delta.mean()),
        "quantiles": {
            "p50": float(np.quantile(delta, 0.50)),
            "p95": float(np.quantile(delta, 0.95)),
            "p99": float(np.quantile(delta, 0.99)),
        },
        "minimum": float(delta.min()),
        "maximum": float(delta.max()),
    }


def evaluate_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Compute only the preregistered game-grouped accuracy diagnostics."""

    _require_columns(
        frame,
        (
            "season",
            "game_id",
            "final_outcome",
            "home_win",
            "home_probability",
            "home_wp",
        ),
    )
    if frame["home_probability"].isna().any():
        raise X11FastrmodelsError("official home probability is missing")
    metric = binary_metric_frame(frame)
    ties = frame.loc[frame["final_outcome"] == "tie"]
    seasons = {
        str(season): _metric_scope(
            metric.loc[metric["season"] == season].copy()
        )
        for season in sorted(int(value) for value in metric["season"].unique())
    }
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_cluster_unit": "game_id",
        "aggregate": _metric_scope(metric),
        "per_season": seasons,
        "ties": {
            "games": int(ties["game_id"].nunique()),
            "rows": len(ties),
            "game_ids": sorted(str(value) for value in ties["game_id"].unique()),
            "excluded_from_binary_metrics": True,
        },
        "source_home_wp_absolute_delta": _source_home_wp_diagnostic(frame),
    }


def _stream_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for record in frame.loc[:, list(columns)].to_dict(orient="records"):
        digest.update(_canonical_bytes(record))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _asset_manifest_path(store_root: Path) -> Path:
    directory = (
        store_root
        / "manifests"
        / "source=nflverse"
        / "dataset=DS-NFL-FASTRMODELS"
        / f"version={ARCHIVE_TAG_COMMIT}"
        / f"partition=asset-{ASSET_ID}"
    )
    paths = tuple(sorted(directory.glob("*.manifest.json")))
    if len(paths) != 1:
        raise X11FastrmodelsError(
            "official no-spread asset manifest must be unique"
        )
    return paths[0]


def _verified_partition(
    *,
    manifest_path: Path,
    store_root: Path,
    program_root: Path,
    year: int,
) -> tuple[bytes, Mapping[str, object]]:
    verified = read_verified_static_object(
        manifest_path,
        store_root=store_root,
        program_root=program_root,
    )
    record = verified.record
    manifest = record.manifest
    expected_object, expected_schema = (
        x11_pipeline.X11_FROZEN_PARTITION_ALLOWLIST[year]
    )
    if (
        record.source != "nflverse"
        or record.dataset != "DS-NFLVERSE"
        or record.version != x11_pipeline.X11_NFLVERSE_VERSION
        or record.partition != f"season-{year}"
        or record.extension != "parquet"
        or manifest.object_sha256 != expected_object
        or manifest.schema_fingerprint != expected_schema
        or not x11_pipeline._matches_frozen_partition_source(
            manifest,
            year=year,
        )
    ):
        raise X11FastrmodelsError(
            f"verified season-{year} object differs from frozen X-11"
        )
    return verified.object_bytes, {
        "season": year,
        "manifest_sha256": manifest.manifest_sha256,
        "object_sha256": manifest.object_sha256,
        "schema_fingerprint": manifest.schema_fingerprint,
        "partition": record.partition,
    }


def _latency_native_rows(table: pa.Table) -> tuple[Mapping[str, object], ...]:
    frame = table.to_pandas()
    frame["_raw_record_ordinal"] = range(len(frame))
    return tuple(frame.to_dict(orient="records"))


def _second_half_receiver(row: Mapping[str, object]) -> str:
    opening = int(float(row["home_opening_kickoff"]))
    if opening not in {0, 1}:
        raise X11FastrmodelsError("home opening kickoff must be binary")
    return (
        str(row["away_team"])
        if opening == 1
        else str(row["home_team"])
    )


def build_full_path_fixture(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_object_sha256: str,
    source_version: str,
) -> FullPathFixture:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["game_id"]), []).append(row)
    for game_id in sorted(grouped):
        ordered = sorted(
            grouped[game_id],
            key=lambda row: (
                int(float(row["order_sequence"])),
                int(float(row["_raw_record_ordinal"])),
            ),
        )
        orders = [int(float(row["order_sequence"])) for row in ordered]
        if any(later <= earlier for earlier, later in zip(orders, orders[1:])):
            continue
        if len(ordered) < 2:
            continue
        try:
            state = nfl_state.state_from_nflverse_row(ordered[0])
        except (TypeError, ValueError):
            continue
        second_half_receiver = _second_half_receiver(ordered[0])
        for sequence, (source_row, post_row) in enumerate(
            zip(ordered, ordered[1:]),
            start=1,
        ):
            try:
                successor_rows = season_census._causal_successor_window(
                    ordered,
                    source_row=source_row,
                    post_index=sequence,
                )
                fixture = FullPathFixture(
                    prior_state=state,
                    source_row=source_row,
                    successor_rows=successor_rows,
                    sequence=sequence,
                    raw_object_sha256=raw_object_sha256,
                    source_version=source_version,
                    second_half_receiver=second_half_receiver,
                )
                event = season_census._actual_event(
                    state,
                    source_row,
                    post_row,
                    successor_rows=successor_rows,
                    sequence=sequence,
                    raw_object_sha256=raw_object_sha256,
                    source_version=source_version,
                )
                next_state = nfl_state.reduce(state, event)
            except (TypeError, ValueError):
                break
            try:
                _input_from_state(
                    next_state,
                    second_half_receiver=second_half_receiver,
                )
            except (TypeError, ValueError):
                state = next_state
                continue
            return fixture
    raise X11FastrmodelsError(
        "frozen 2025 rows contain no complete full-path latency fixture"
    )


def _full_path_prediction(
    fixture: FullPathFixture,
    predictor: OfficialNoSpreadPredictor,
) -> float:
    post_row = fixture.successor_rows[0]
    event = season_census._actual_event(
        fixture.prior_state,
        fixture.source_row,
        post_row,
        successor_rows=fixture.successor_rows,
        sequence=fixture.sequence,
        raw_object_sha256=fixture.raw_object_sha256,
        source_version=fixture.source_version,
    )
    state = nfl_state.reduce(fixture.prior_state, event)
    model_input = _input_from_state(
        state,
        second_half_receiver=fixture.second_half_receiver,
    )
    return _validate_probability(predictor.predict_home(model_input))


def measure_full_path_latency(
    fixture: FullPathFixture,
    predictor: OfficialNoSpreadPredictor,
    *,
    warmup: int = 50,
    samples: int = 1000,
) -> dict[str, object]:
    if type(warmup) is not int or warmup < 1:
        raise X11FastrmodelsError("latency warmup must be positive")
    if type(samples) is not int or samples < 20:
        raise X11FastrmodelsError("latency samples must be at least 20")
    probability = 0.0
    for _ in range(warmup):
        probability = _full_path_prediction(fixture, predictor)
    durations: list[int] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        probability = _full_path_prediction(fixture, predictor)
        durations.append(time.perf_counter_ns() - started)
    values = np.asarray(durations, dtype=np.int64)
    mean_ns = float(values.mean())
    report = {
        "measurement_scope": "full_path",
        "included_stages": list(FULL_PATH_INCLUDED_STAGES),
        "excluded_stages": list(LATENCY_EXCLUDED_STAGES),
        "warmup_calls": warmup,
        "samples": samples,
        "unit": "nanoseconds",
        "minimum_ns": int(values.min()),
        "p50_ns": int(np.quantile(values, 0.50)),
        "p95_ns": int(np.quantile(values, 0.95)),
        "p99_ns": int(np.quantile(values, 0.99)),
        "maximum_ns": int(values.max()),
        "mean_ns": mean_ns,
        "transitions_per_second_from_mean": 1_000_000_000.0 / mean_ns,
        "validated_probability": probability,
        "fixture": {
            "game_id": fixture.prior_state.game_id,
            "source_order_sequence": (
                fixture.prior_state.source_order_sequence
            ),
            "sequence": fixture.sequence,
            "raw_object_sha256": fixture.raw_object_sha256,
        },
    }
    report["latency_sha256"] = canonical_sha256(report)
    return report


def load_frozen_evaluation(
    *,
    program_root: str | Path,
    store_root: str | Path,
    gate: ReproductionGate,
) -> LoadedEvaluation:
    """Read verified 2021-2025 rows only after the exact gate is established."""

    if not isinstance(gate, ReproductionGate):
        raise X11FastrmodelsError("evaluation rows require a reproduction gate")
    program = Path(program_root).resolve()
    store = Path(store_root).resolve()
    manifest_paths = x11_pipeline._discover_manifest_paths(store)
    by_year = {
        int(path.parent.name.removeprefix("partition=season-")): path
        for path in manifest_paths
    }
    frames: list[pd.DataFrame] = []
    census_by_season: dict[str, Mapping[str, int]] = {}
    bindings: list[Mapping[str, object]] = []
    full_path_fixture: FullPathFixture | None = None
    latency_columns = tuple(
        dict.fromkeys(
            (
                *season_census._CENSUS_COLUMNS,
                "home_opening_kickoff",
            )
        )
    )
    for year in SEASONS:
        payload, binding = _verified_partition(
            manifest_path=by_year[year],
            store_root=store,
            program_root=program,
            year=year,
        )
        columns = (
            tuple(dict.fromkeys((*_EVALUATION_COLUMNS, *latency_columns)))
            if year == 2025
            else _EVALUATION_COLUMNS
        )
        try:
            table = pq.ParquetFile(BytesIO(payload)).read(columns=list(columns))
        except (pa.ArrowException, OSError) as error:
            raise X11FastrmodelsError(
                f"verified season-{year} Parquet cannot be read"
            ) from error
        native = table.select(list(_EVALUATION_COLUMNS)).to_pandas()
        native["_raw_record_ordinal"] = range(len(native))
        prepared = prepare_partition_frame(
            native,
            season=year,
            manifest_sha256=str(binding["manifest_sha256"]),
            object_sha256=str(binding["object_sha256"]),
        )
        if dict(prepared.census) != EXPECTED_CENSUS[year]:
            raise X11FastrmodelsError(
                f"season-{year} census differs from preregistration"
            )
        frames.append(prepared.frame)
        census_by_season[str(year)] = dict(prepared.census)
        bindings.append(binding)
        if year == 2025:
            full_path_fixture = build_full_path_fixture(
                _latency_native_rows(table.select(list(latency_columns))),
                raw_object_sha256=str(binding["object_sha256"]),
                source_version=x11_pipeline.X11_NFLVERSE_VERSION,
            )
    frame = pd.concat(frames, ignore_index=True)
    total = {
        key: sum(int(value[key]) for value in census_by_season.values())
        for key in EXPECTED_TOTAL_CENSUS
    }
    if total != EXPECTED_TOTAL_CENSUS:
        raise X11FastrmodelsError(
            "aggregate census differs from preregistration"
        )
    if full_path_fixture is None:
        raise X11FastrmodelsError("full-path latency fixture is absent")
    input_columns = (
        "season",
        "game_id",
        "season_type",
        "qtr",
        "order_sequence",
        "raw_record_ordinal",
        "posteam",
        "defteam",
        "home_team",
        "away_team",
        "home_opening_kickoff",
        "score_differential",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "result",
        "home_wp",
        "manifest_sha256",
        "object_sha256",
    )
    return LoadedEvaluation(
        frame=frame,
        census={"seasons": census_by_season, "total": total},
        input_manifest_bindings=tuple(bindings),
        evaluation_input_sha256=_stream_frame_sha256(frame, input_columns),
        full_path_fixture=full_path_fixture,
    )


def _run_metrics_once(
    loaded: LoadedEvaluation,
    predictor: OfficialNoSpreadPredictor,
) -> dict[str, object]:
    predictions = predict_frame(loaded.frame, predictor)
    metrics = evaluate_metrics(predictions)
    prediction_sha256 = _stream_frame_sha256(
        predictions,
        _METRIC_ID_COLUMNS,
    )
    return {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_variant": "official_no_spread",
        "model_training": "official_model_not_retrained",
        "calibrator": "none_fitted",
        "census": copy.deepcopy(dict(loaded.census)),
        "input_manifest_bindings": [
            dict(value) for value in loaded.input_manifest_bindings
        ],
        "evaluation_input_sha256": loaded.evaluation_input_sha256,
        "predictions_sha256": prediction_sha256,
        "evaluation": metrics,
    }


def _runner_sha256() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _markdown(document: Mapping[str, object]) -> str:
    metrics = document["metrics"]
    assert isinstance(metrics, dict)
    evaluation = metrics["evaluation"]
    assert isinstance(evaluation, dict)
    aggregate = evaluation["aggregate"]
    assert isinstance(aggregate, dict)
    row_micro = aggregate["row_micro"]
    game_macro = aggregate["game_macro"]
    assert isinstance(row_micro, dict) and isinstance(game_macro, dict)
    latency = document["latency"]
    assert isinstance(latency, dict)
    census = metrics["census"]
    assert isinstance(census, dict)
    total = census["total"]
    assert isinstance(total, dict)
    return "\n".join(
        (
            "# NFL official no-spread reproduction v2",
            "",
            "- Status: **PRELIMINARY**",
            "- PIT status: **PIT_UNPROVEN**",
            "- Observation mode: `offline_reconstruction_not_live_PIT`",
            "- Calibrator: none fitted",
            "- Prediction-market alignment: none",
            "- Prediction-market symmetry: not evaluated",
            "- Alpha evidence: none",
            "",
            "## Frozen evaluation",
            "",
            f"- Eligible rows: {total['eligible_rows']}",
            f"- Eligible non-tie games: {total['eligible_non_tie_games']}",
            f"- Row-micro Brier: {row_micro['brier']:.12f}",
            f"- Row-micro log loss: {row_micro['log_loss']:.12f}",
            f"- Calibration slope: {row_micro['calibration_slope']:.12f}",
            f"- Calibration intercept: {row_micro['calibration_intercept']:.12f}",
            f"- Game-macro Brier: {game_macro['brier']:.12f}",
            f"- Game-macro log loss: {game_macro['log_loss']:.12f}",
            "",
            "## Current full-path latency",
            "",
            (
                "- Scope: event construction → state transition → official "
                "feature projection → preloaded booster → probability validation"
            ),
            f"- p50: {latency['p50_ns']} ns",
            f"- p95: {latency['p95_ns']} ns",
            f"- p99: {latency['p99_ns']} ns",
            (
                "- Excludes I/O, registry loading, network, market joins, "
                "and any market/alpha interpretation."
            ),
            "",
        )
    )


def write_reproduction_artifacts(
    document: Mapping[str, object],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(_markdown(document), encoding="utf-8")


def read_reproduction_artifact(path: str | Path) -> dict[str, object]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X11FastrmodelsError("reproduction artifact is unreadable") from error
    if not isinstance(document, dict):
        raise X11FastrmodelsError("reproduction artifact must be an object")
    material = copy.deepcopy(document)
    observed = material.pop("artifact_sha256", None)
    if observed != canonical_sha256(material):
        raise X11FastrmodelsError("reproduction artifact self-hash mismatch")
    return document


def run_frozen_reproduction(
    *,
    program_root: str | Path,
    store_root: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
    latency_warmup: int = 50,
    latency_samples: int = 1000,
) -> dict[str, object]:
    """Execute two deterministic metric runs plus separately bound latency."""

    program = Path(program_root).resolve()
    store = Path(store_root).resolve()
    predictor = load_official_no_spread_predictor(
        program_root=program,
        store_root=store,
        manifest_path=_asset_manifest_path(store),
    )
    evaluation_started_at = datetime.now(timezone.utc)
    gate = verify_reproduction_gate(
        program,
        evaluation_started_at=evaluation_started_at,
    )
    loaded = load_frozen_evaluation(
        program_root=program,
        store_root=store,
        gate=gate,
    )
    first_metrics = _run_metrics_once(loaded, predictor)
    second_metrics = _run_metrics_once(loaded, predictor)
    first_bytes = metrics_bytes(first_metrics)
    second_bytes = metrics_bytes(second_metrics)
    if first_bytes != second_bytes:
        raise X11FastrmodelsError(
            "two complete official metric runs are not byte-identical"
        )
    metrics_sha256 = (
        "sha256:" + hashlib.sha256(first_bytes).hexdigest()
    )
    latency = measure_full_path_latency(
        loaded.full_path_fixture,
        predictor,
        warmup=latency_warmup,
        samples=latency_samples,
    )
    evaluation_started_at_text = _canonical_utc(evaluation_started_at)
    result_ref = validate_result_ref(
        program,
        EXPERIMENT_ID,
        {
            "scope": gate.scope,
            "result_label": "PRELIMINARY",
            "evaluation_started_at": evaluation_started_at_text,
            "code_sha256": gate.code_sha256,
            "data_sha256": gate.data_sha256,
            "result_sha256": metrics_sha256,
            "registration_head_sha256": gate.registration_head_sha256,
            "dataset_ids": list(gate.dataset_ids),
            "model_ids": list(gate.model_ids),
        },
    )
    material: dict[str, object] = {
        "artifact_id": "NFL-FASTRMODELS-NO-SPREAD-REPRODUCTION-V2",
        "artifact_version": "v2",
        "experiment_id": EXPERIMENT_ID,
        "reproduction_id": REPRODUCTION_ID,
        "status": "PRELIMINARY",
        "pit_status": "PIT_UNPROVEN",
        "observation_mode": "offline_reconstruction_not_live_PIT",
        "evaluation_started_at": evaluation_started_at_text,
        "registration": {
            **asdict(gate),
            "evaluation_started_at": evaluation_started_at_text,
            "amended_at": _canonical_utc(gate.amended_at),
        },
        "result_ref_candidate": result_ref,
        "runner_sha256": _runner_sha256(),
        "metrics": first_metrics,
        "determinism": {
            "complete_runs": 2,
            "metrics_bytes_identical": True,
            "metrics_byte_length": len(first_bytes),
            "metrics_sha256": metrics_sha256,
            "latency_excluded_from_metrics_bytes": True,
        },
        "latency": latency,
        "evidence_boundaries": {
            "alpha_evidence": "none",
            "calibrator": "none_fitted",
            "market_data_used": False,
            "prediction_market_alignment": "none",
            "prediction_market_symmetry": "not_evaluated",
            "training": "official_model_not_retrained",
        },
    }
    material["artifact_sha256"] = canonical_sha256(material)
    write_reproduction_artifacts(
        material,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    return material


__all__ = [
    "BOOTSTRAP_SEED",
    "EXPECTED_CENSUS",
    "EXPECTED_TOTAL_CENSUS",
    "FULL_PATH_INCLUDED_STAGES",
    "LATENCY_EXCLUDED_STAGES",
    "LoadedEvaluation",
    "PreparedPartition",
    "ReproductionGate",
    "X11FastrmodelsError",
    "binary_labels",
    "binary_metric_frame",
    "canonical_sha256",
    "evaluate_metrics",
    "load_frozen_evaluation",
    "measure_full_path_latency",
    "metrics_bytes",
    "predict_frame",
    "prepare_partition_frame",
    "project_no_spread_input",
    "read_reproduction_artifact",
    "run_frozen_reproduction",
    "verify_reproduction_gate",
    "write_reproduction_artifacts",
]
