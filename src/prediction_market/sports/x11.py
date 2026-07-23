"""Verified real-data adapter and preliminary evaluator for NFL X-11."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prediction_market import contracts
from prediction_market.models.validation import (
    ValidationInputError,
    evaluate_model_vs_prior,
    evaluate_probabilities,
)
from prediction_market.sports.nflverse import (
    NFLVERSE_RELEASE_ID,
    NFLVERSE_RELEASE_VERSION,
    NFLVERSE_YEAR_ASSET_IDS,
    NFLVerseSourceError,
    inspect_nflverse_partition,
    nflverse_partition_url,
)
from prediction_market.static_store import read_verified_static_object


X11_YEARS = tuple(range(2015, 2026))
X11_NFLVERSE_VERSION = NFLVERSE_RELEASE_VERSION
X11_DATASET_ID = "DS-NFLVERSE"
X11_SOURCE = "nflverse"
X11_LICENSE_REF = "I-018"
X11_LICENSE_STATUS = "approved"
X11_SEED = 20260722
X11_RESULT_LABEL = "PRELIMINARY"
X11_PIT_STATUS = "PIT_UNPROVEN"
X11_EXPERIMENT_ID = "X-11"
X11_TRANSITION_MODEL_ID = "MODEL-NFL-DRIVE-TRANSITION"
X11_TRANSITION_MODEL_VERSION = "v1"
X11_FROZEN_PARTITION_ALLOWLIST = MappingProxyType({
    2015: (
        "sha256:01b5ae7d06633a2e66b418404461a3d4ff055ad2f3a87d453cebcc8d3fda7ed9",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2016: (
        "sha256:95eba04e2145e3c1c8ca502f2a3a76cfb0a5990680c3fb480f02a74a45f54a3b",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2017: (
        "sha256:84eacd963c1fdd45965f6222c62e9329a7f3412f029d92c1ac5e34a1bb4d3710",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2018: (
        "sha256:2e6f2dce7c7ebd46e985cabe0c17eb72b39a77f98cb4478409294f50b5820150",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2019: (
        "sha256:60c3067017db2d28a78f66a79b657268be8578d9a5288e6a827efdcd7fe42540",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2020: (
        "sha256:73b7dbf66fa8cb9356f58bf6b1f15a0fee197ecc10cf4983b640cb9679b15cb4",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2021: (
        "sha256:333ad34378e5339d5172717cc83378e908daf02c8699416ab3e17c2ec10f78d8",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2022: (
        "sha256:931121d8897779d7944e2a293e92ed8799c8e5cceef84096ac42339003fedc09",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2023: (
        "sha256:bd3484731408def6b0ec93225bba2bd7b2c65769ca707a2b9444d891abdc6776",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2024: (
        "sha256:6d432dd4308329bfddaef633309ea119f9ca46d52cbb3c09f47172a2e8efcd01",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
    2025: (
        "sha256:3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a",
        "sha256:879e24069b394aeb76267515b2bd8201b75927293e5e0a5c99f0b017648d7c1f",
    ),
})

TRANSITION_CLASSES = (
    "touchdown",
    "field_goal",
    "punt",
    "turnover",
    "other",
)
NATIVE_STATE_FEATURES = (
    "home_score_differential",
    "game_seconds_remaining",
    "possession_home",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
)
GAME_STATE_FEATURES = (*NATIVE_STATE_FEATURES, "spread_prior")

_NATIVE_ADAPTER_COLUMNS = (
    "play_id",
    "game_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "score_differential",
    "game_seconds_remaining",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "spread_line",
    "home_wp",
    "fixed_drive",
    "fixed_drive_result",
    "home_score",
    "away_score",
)
_HASH_PREFIX = "sha256:"


class X11DataError(ValueError):
    """Verified inputs cannot support the frozen X-11 data contract."""


@dataclass(frozen=True, slots=True)
class X11PartitionInventory:
    year: int
    partition: str
    manifest_sha256: str
    object_sha256: str
    schema_fingerprint: str
    rows: int
    games: int
    season_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class X11InputInventory:
    dataset_id: str
    source: str
    version: str
    years: tuple[int, ...]
    partitions: tuple[X11PartitionInventory, ...]
    total_rows: int
    total_games: int
    season_types: tuple[str, ...]
    inventory_sha256: str


@dataclass(frozen=True, slots=True)
class X11AdapterAudit:
    native_drives: int
    canonical_drive_starts: int
    excluded_drives_without_complete_state: int
    games: int
    ties: int


@dataclass(frozen=True, slots=True)
class X11LoadedDataset:
    inventory: X11InputInventory
    drive_starts: pd.DataFrame
    chronology_sha256: str
    normalized_frame_sha256: str
    adapter_audit: X11AdapterAudit


@dataclass(frozen=True, slots=True)
class X11FoldAudit:
    test_game_id: str
    pit_cutoff: pd.Timestamp
    train_max_game_date: pd.Timestamp
    train_game_count: int
    binary_train_game_count: int
    train_drive_rows: int
    test_game_count: int
    test_drive_rows: int
    prior_train_game_count: int


@dataclass(frozen=True, slots=True)
class X11Evaluation:
    result_label: str
    seed: int
    predictions: pd.DataFrame
    transition_predictions: pd.DataFrame
    folds: tuple[X11FoldAudit, ...]
    outcome_metrics: dict[str, dict[str, object]]
    transition_metrics: dict[str, object]
    tie_report: dict[str, object]
    season_stability: dict[str, object]
    model_features: dict[str, tuple[str, ...]]
    normalized_frame_sha256: str
    evaluation_input_sha256: str
    predictions_sha256: str
    transition_predictions_sha256: str
    transition_outputs: tuple[dict[str, object], ...]
    transition_outputs_sha256: str
    evaluation_sha256: str
    bootstrap_samples: int
    minimum_valid_bootstrap_samples: int
    confidence_level: float
    evaluation_game_limit: int | None
    minimum_prior_train_games: int
    gbdt_max_iter: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_ready(value: object) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise X11DataError("evidence cannot contain non-finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise X11DataError(
        f"evidence contains unsupported value type: {type(value).__name__}"
    )


def _sha256(value: object) -> str:
    return _HASH_PREFIX + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _inventory_material(inventory: X11InputInventory) -> dict[str, object]:
    return {
        "dataset_id": inventory.dataset_id,
        "source": inventory.source,
        "version": inventory.version,
        "years": list(inventory.years),
        "partitions": [
            {
                **asdict(partition),
                "season_types": list(partition.season_types),
            }
            for partition in inventory.partitions
        ],
        "total_rows": inventory.total_rows,
        "total_games": inventory.total_games,
        "season_types": list(inventory.season_types),
    }


def inventory_sha256(inventory: X11InputInventory) -> str:
    """Recompute the inventory's self-hash, excluding its hash field."""

    if not isinstance(inventory, X11InputInventory):
        raise TypeError("inventory must be an X11InputInventory")
    return _sha256(_inventory_material(inventory))


def evidence_sha256(evidence: dict[str, object]) -> str:
    """Recompute the evidence self-hash, excluding its hash field."""

    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")
    material = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    return _sha256(_json_ready(material))


def evaluation_sha256(evaluation: X11Evaluation) -> str:
    """Hash every frozen X-11 evaluation result and its content hashes."""

    if not isinstance(evaluation, X11Evaluation):
        raise TypeError("evaluation must be an X11Evaluation")
    material = {
        "result_label": evaluation.result_label,
        "seed": evaluation.seed,
        "normalized_frame_sha256": evaluation.normalized_frame_sha256,
        "evaluation_input_sha256": evaluation.evaluation_input_sha256,
        "predictions_sha256": evaluation.predictions_sha256,
        "transition_predictions_sha256": (
            evaluation.transition_predictions_sha256
        ),
        "transition_outputs_sha256": evaluation.transition_outputs_sha256,
        "folds": [asdict(fold) for fold in evaluation.folds],
        "outcome_metrics": evaluation.outcome_metrics,
        "transition_metrics": evaluation.transition_metrics,
        "tie_report": evaluation.tie_report,
        "season_stability": evaluation.season_stability,
        "model_features": {
            name: list(features)
            for name, features in evaluation.model_features.items()
        },
        "bootstrap_samples": evaluation.bootstrap_samples,
        "minimum_valid_bootstrap_samples": (
            evaluation.minimum_valid_bootstrap_samples
        ),
        "confidence_level": evaluation.confidence_level,
        "evaluation_game_limit": evaluation.evaluation_game_limit,
        "minimum_prior_train_games": evaluation.minimum_prior_train_games,
        "gbdt_max_iter": evaluation.gbdt_max_iter,
    }
    return _sha256(_json_ready(material))


def expected_nflverse_source_cursor(year: int) -> str:
    """Return the frozen official release/asset cursor for one X-11 season."""

    if year not in X11_YEARS:
        raise X11DataError("X-11 source cursor year must be in 2015-2025")
    return (
        f"github_release_id:{NFLVERSE_RELEASE_ID};"
        f"asset_id:{NFLVERSE_YEAR_ASSET_IDS[year]}"
    )


def _matches_frozen_partition_source(
    manifest: Any,
    *,
    year: int,
) -> bool:
    expected_object, expected_schema = X11_FROZEN_PARTITION_ALLOWLIST[year]
    get = manifest.get if isinstance(manifest, dict) else lambda key: getattr(
        manifest, key, None
    )
    return (
        get("license_ref") == X11_LICENSE_REF
        and get("license_status") == X11_LICENSE_STATUS
        and get("dataset_id") == X11_DATASET_ID
        and get("upstream_partition") == f"season-{year}"
        and get("object_kind") == "byte_exact_original"
        and get("source_url") == nflverse_partition_url(year)
        and get("source_cursor") == expected_nflverse_source_cursor(year)
        and get("object_sha256") == expected_object
        and get("schema_fingerprint") == expected_schema
    )


def _discover_manifest_paths(store_root: Path) -> tuple[Path, ...]:
    base = (
        store_root
        / "manifests"
        / f"source={X11_SOURCE}"
        / f"dataset={X11_DATASET_ID}"
        / f"version={X11_NFLVERSE_VERSION}"
    )
    paths: list[Path] = []
    for year in X11_YEARS:
        observed = tuple(
            sorted((base / f"partition=season-{year}").glob("*.manifest.json"))
        )
        matches: list[Path] = []
        for path in observed:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise X11DataError(
                    f"cannot classify X-11 manifest observation: {path}"
                ) from error
            if not isinstance(document, dict):
                raise X11DataError(
                    f"X-11 manifest observation must be an object: {path}"
                )
            if _matches_frozen_partition_source(document, year=year):
                matches.append(path)
        if len(matches) != 1:
            raise X11DataError(
                "X-11 requires exactly one governed "
                f"{X11_LICENSE_REF}/{X11_LICENSE_STATUS} manifest for "
                f"season-{year}; found {len(matches)} among {len(observed)} "
                "append-only observations"
            )
        paths.append(matches[0])
    return tuple(paths)


def _validate_digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_HASH_PREFIX)
        or len(value) != len(_HASH_PREFIX) + 64
    ):
        raise X11DataError(f"{field} must be a SHA-256 digest")
    try:
        int(value.removeprefix(_HASH_PREFIX), 16)
    except ValueError as error:
        raise X11DataError(f"{field} must be a SHA-256 digest") from error
    return value


def _partition_year(partition: object) -> int:
    if type(partition) is not str or not partition.startswith("season-"):
        raise X11DataError("verified partition is not an X-11 season")
    try:
        year = int(partition.removeprefix("season-"))
    except ValueError as error:
        raise X11DataError("verified partition is not an X-11 season") from error
    if year not in X11_YEARS or partition != f"season-{year}":
        raise X11DataError("verified partition year is outside 2015-2025")
    return year


def _constant_by_game(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        counts = frame.groupby("game_id", sort=False)[column].nunique(dropna=False)
        if not bool((counts == 1).all()):
            raise X11DataError(f"native {column} must be constant within every game")


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[
            ~result[column].map(
                lambda value: pd.notna(value) and math.isfinite(float(value))
            ),
            column,
        ] = float("nan")
    return result


def _transition_label(native: object) -> str:
    if not isinstance(native, str):
        return "other"
    normalized = " ".join(native.strip().lower().split())
    if normalized == "touchdown":
        return "touchdown"
    if normalized == "field goal":
        return "field_goal"
    if normalized == "punt":
        return "punt"
    if normalized in {"turnover", "turnover on downs"}:
        return "turnover"
    return "other"


def _canonical_partition_frame(
    object_bytes: bytes,
    *,
    year: int,
    manifest_sha256: str,
    object_sha256: str,
    schema_fingerprint: str,
) -> tuple[pd.DataFrame, int]:
    try:
        parquet = pq.ParquetFile(BytesIO(object_bytes))
        table = parquet.read(columns=list(_NATIVE_ADAPTER_COLUMNS))
    except (pa.ArrowException, OSError) as error:
        raise X11DataError("verified NFL object cannot be adapted") from error
    native = table.to_pandas()
    if native.empty:
        raise X11DataError("verified NFL partition is empty")
    native["_native_row"] = range(len(native))
    if (
        native["game_id"].isna().any()
        or (native["game_id"].astype(str).str.strip() == "").any()
    ):
        raise X11DataError("native game_id must be present")
    native["game_id"] = native["game_id"].astype(str)
    native["game_date"] = pd.to_datetime(native["game_date"], errors="coerce", utc=True)
    if native["game_date"].isna().any():
        raise X11DataError("native game_date must be parseable as UTC")
    numeric = (
        "play_id",
        "season",
        "week",
        "score_differential",
        "game_seconds_remaining",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
        "spread_line",
        "home_wp",
        "fixed_drive",
        "home_score",
        "away_score",
    )
    native = _finite_numeric(native, numeric)
    identity_numeric = (
        "play_id",
        "season",
        "week",
        "fixed_drive",
        "home_score",
        "away_score",
    )
    if native[list(identity_numeric)].isna().any().any():
        raise X11DataError("native game/drive identity contains invalid numbers")
    if not (native["season"] == year).all():
        raise X11DataError(f"native season mismatch while adapting season-{year}")
    fixed_drive = native["fixed_drive"]
    if (fixed_drive <= 0).any() or not fixed_drive.map(float.is_integer).all():
        raise X11DataError("fixed_drive must contain positive integers")
    native["drive_number"] = fixed_drive.astype(int)
    _constant_by_game(
        native,
        (
            "season",
            "season_type",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
        ),
    )
    if not set(native["season_type"].unique()) <= {"REG", "POST"}:
        raise X11DataError("native season_type must be REG or POST")
    if (
        native["home_team"].isna().any()
        or native["away_team"].isna().any()
        or (native["home_team"] == native["away_team"]).any()
    ):
        raise X11DataError("native home/away teams are invalid")
    native_drive_count = int(
        native[["game_id", "drive_number"]].drop_duplicates().shape[0]
    )
    state_numeric = (
        "score_differential",
        "game_seconds_remaining",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
        "spread_line",
        "home_wp",
    )
    valid_state = native[list(state_numeric)].notna().all(axis=1) & (
        (native["posteam"] == native["home_team"])
        | (native["posteam"] == native["away_team"])
    )
    candidates = native.loc[valid_state].copy()
    if candidates.empty:
        raise X11DataError(f"season-{year} contains no complete drive-start states")
    candidates = candidates.sort_values(
        ["game_date", "game_id", "drive_number", "play_id", "_native_row"],
        kind="mergesort",
    )
    selected = candidates.drop_duplicates(
        ["game_id", "drive_number"], keep="first"
    ).copy()
    if set(selected["game_id"]) != set(native["game_id"]):
        raise X11DataError(
            f"season-{year} contains a game with no complete drive state"
        )
    home_possession = selected["posteam"] == selected["home_team"]
    selected["home_score_differential"] = selected["score_differential"].where(
        home_possession, -selected["score_differential"]
    )
    selected["possession_home"] = home_possession.astype(float)
    selected["final_outcome"] = "tie"
    selected.loc[selected["home_score"] > selected["away_score"], "final_outcome"] = (
        "home_win"
    )
    selected.loc[selected["home_score"] < selected["away_score"], "final_outcome"] = (
        "away_win"
    )
    selected["home_win"] = pd.Series(
        [
            1 if outcome == "home_win" else 0 if outcome == "away_win" else pd.NA
            for outcome in selected["final_outcome"]
        ],
        index=selected.index,
        dtype="Int64",
    )
    selected["next_drive_outcome"] = selected["fixed_drive_result"].map(
        _transition_label
    )
    selected["manifest_sha256"] = manifest_sha256
    selected["object_sha256"] = object_sha256
    selected["schema_fingerprint"] = schema_fingerprint
    canonical_columns = (
        "game_id",
        "season",
        "season_type",
        "week",
        "game_date",
        "home_team",
        "away_team",
        "drive_number",
        "play_id",
        "home_score_differential",
        "game_seconds_remaining",
        "possession_home",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
        "spread_line",
        "home_wp",
        "final_outcome",
        "home_win",
        "next_drive_outcome",
        "manifest_sha256",
        "object_sha256",
        "schema_fingerprint",
    )
    return selected.loc[:, canonical_columns], native_drive_count


def _chronology_material(frame: pd.DataFrame) -> list[dict[str, object]]:
    games = (
        frame[
            [
                "game_id",
                "season",
                "season_type",
                "game_date",
                "final_outcome",
            ]
        ]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"], kind="mergesort")
    )
    return [
        {
            "game_id": row.game_id,
            "season": int(row.season),
            "season_type": row.season_type,
            "game_date": row.game_date.strftime("%Y-%m-%d"),
            "final_outcome": row.final_outcome,
        }
        for row in games.itertuples(index=False)
    ]


def chronology_sha256(frame: pd.DataFrame) -> str:
    """Hash the frozen date/game ordering and outcome identity."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise X11DataError("chronology frame must be a nonempty DataFrame")
    required = {
        "game_id",
        "season",
        "season_type",
        "game_date",
        "final_outcome",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise X11DataError(f"chronology frame is missing columns: {missing}")
    return _sha256(_chronology_material(frame))


_NORMALIZED_FRAME_COLUMNS = (
    "game_order",
    "game_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "drive_number",
    "play_id",
    "home_score_differential",
    "game_seconds_remaining",
    "possession_home",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "spread_line",
    "home_wp",
    "final_outcome",
    "home_win",
    "next_drive_outcome",
    "manifest_sha256",
    "object_sha256",
    "schema_fingerprint",
)


def _frame_scalar(value: object) -> object:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return _json_ready(value)


def _frame_sha256(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    context: str,
) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise X11DataError(f"{context} must be a nonempty DataFrame")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise X11DataError(f"{context} is missing columns: {missing}")
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"columns": list(columns)}))
    digest.update(b"\n")
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        digest.update(_canonical_bytes([_frame_scalar(value) for value in row]))
        digest.update(b"\n")
    return _HASH_PREFIX + digest.hexdigest()


def normalized_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash every canonical normalized row consumed by the X-11 runner."""

    return _frame_sha256(
        frame,
        columns=_NORMALIZED_FRAME_COLUMNS,
        context="normalized X-11 frame",
    )


_EVALUATION_INPUT_COLUMNS = (
    *_NORMALIZED_FRAME_COLUMNS,
    "spread_prior",
    "prior_train_game_count",
    "prior_train_max_game_date",
    "prior_pit_cutoff",
    "prior_method",
    "prior_pit_status",
)


def evaluation_input_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact normalized plus derived PIT frame consumed by model fitting."""

    return _frame_sha256(
        frame,
        columns=_EVALUATION_INPUT_COLUMNS,
        context="X-11 evaluation input frame",
    )


def attach_point_in_time_spread_prior(
    frame: pd.DataFrame,
    *,
    minimum_train_games: int,
) -> pd.DataFrame:
    """Fit each game's spread prior from strictly earlier non-tie games."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise X11DataError("drive frame must be a nonempty DataFrame")
    if type(minimum_train_games) is not int or minimum_train_games < 2:
        raise X11DataError("minimum_train_games must be an integer >= 2")
    required = {
        "game_id",
        "game_date",
        "spread_line",
        "final_outcome",
        "home_win",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise X11DataError(f"spread prior frame is missing columns: {missing}")
    result = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(result["game_date"]):
        raise X11DataError("game_date must be a timezone-aware UTC datetime")
    if (
        result["game_date"].dt.tz is None
        or str(result["game_date"].dt.tz) != "UTC"
        or result["game_date"].isna().any()
    ):
        raise X11DataError("game_date must be a timezone-aware UTC datetime")
    _constant_by_game(
        result,
        ("game_date", "spread_line", "final_outcome", "home_win"),
    )
    if not set(result["final_outcome"].unique()) <= {
        "home_win",
        "away_win",
        "tie",
    }:
        raise X11DataError("final_outcome must be home_win, away_win, or tie")
    games = (
        result[
            [
                "game_id",
                "game_date",
                "spread_line",
                "final_outcome",
                "home_win",
            ]
        ]
        .drop_duplicates("game_id")
        .sort_values(["game_date", "game_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    games["spread_line"] = pd.to_numeric(games["spread_line"], errors="coerce")
    if (
        games["spread_line"].isna().any()
        or not games["spread_line"].map(lambda value: math.isfinite(float(value))).all()
    ):
        raise X11DataError("spread_line must be finite for every game")
    binary = games["final_outcome"] != "tie"
    expected_home_win = games["final_outcome"].map({"home_win": 1, "away_win": 0})
    if (
        not (
            games.loc[binary, "home_win"].astype(int).to_numpy()
            == expected_home_win.loc[binary].astype(int).to_numpy()
        ).all()
        or games.loc[~binary, "home_win"].notna().any()
    ):
        raise X11DataError("home_win does not match final_outcome")

    games["spread_prior"] = float("nan")
    games["prior_train_game_count"] = 0
    games["prior_train_max_game_date"] = pd.Series(
        pd.NaT,
        index=games.index,
        dtype=games["game_date"].dtype,
    )
    games["prior_pit_cutoff"] = games["game_date"]
    games["prior_method"] = "logistic_spread_line_strict_prior_game_dates"
    games["prior_pit_status"] = X11_PIT_STATUS
    for cutoff in games["game_date"].drop_duplicates().sort_values():
        train = games.loc[
            (games["game_date"] < cutoff) & (games["final_outcome"] != "tie")
        ]
        test_index = games.index[games["game_date"] == cutoff]
        games.loc[test_index, "prior_train_game_count"] = len(train)
        if not train.empty:
            games.loc[test_index, "prior_train_max_game_date"] = train[
                "game_date"
            ].max()
        if len(train) < minimum_train_games or train["home_win"].nunique() != 2:
            continue
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=X11_SEED,
        )
        model.fit(
            train[["spread_line"]].to_numpy(dtype=float),
            train["home_win"].to_numpy(dtype=int),
        )
        games.loc[test_index, "spread_prior"] = model.predict_proba(
            games.loc[test_index, ["spread_line"]].to_numpy(dtype=float)
        )[:, 1]
    additions = games[
        [
            "game_id",
            "spread_prior",
            "prior_train_game_count",
            "prior_train_max_game_date",
            "prior_pit_cutoff",
            "prior_method",
            "prior_pit_status",
        ]
    ]
    result = result.drop(
        columns=[
            column
            for column in additions.columns
            if column != "game_id" and column in result.columns
        ]
    ).merge(additions, on="game_id", how="left", validate="many_to_one")
    result = result.sort_values(
        ["game_date", "game_id", "drive_number", "play_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return result


def _outcome_models(
    train: pd.DataFrame,
    *,
    gbdt_max_iter: int,
) -> tuple[Pipeline, HistGradientBoostingClassifier]:
    binary_train = train.loc[train["home_win"].notna()]
    if binary_train["home_win"].nunique() != 2:
        raise X11DataError(
            "strictly earlier binary training games require both outcomes"
        )
    features = binary_train.loc[:, GAME_STATE_FEATURES].to_numpy(dtype=float)
    target = binary_train["home_win"].to_numpy(dtype=int)
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=X11_SEED,
                ),
            ),
        ]
    ).fit(features, target)
    gbdt = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=gbdt_max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=X11_SEED,
    ).fit(features, target)
    return logistic, gbdt


def _transition_model(train: pd.DataFrame) -> Pipeline:
    observed = set(train["next_drive_outcome"].unique())
    if observed != set(TRANSITION_CLASSES):
        raise X11DataError(
            "strictly earlier transition training is missing a frozen class"
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=X11_SEED,
                ),
            ),
        ]
    ).fit(
        train.loc[:, GAME_STATE_FEATURES].to_numpy(dtype=float),
        train["next_drive_outcome"].to_numpy(dtype=object),
    )


def _positive_class_probability(model: Any, features: np.ndarray) -> np.ndarray:
    classes = tuple(int(value) for value in model.classes_)
    try:
        home_index = classes.index(1)
    except ValueError as error:
        raise X11DataError("binary model did not retain the home-win class") from error
    return np.asarray(model.predict_proba(features)[:, home_index], dtype=float)


def _fixed_transition_probabilities(
    model: Pipeline, features: np.ndarray
) -> np.ndarray:
    observed_classes = tuple(str(value) for value in model.classes_)
    if set(observed_classes) != set(TRANSITION_CLASSES):
        raise X11DataError("transition model classes differ from frozen state space")
    raw = np.asarray(model.predict_proba(features), dtype=float)
    ordered = np.column_stack(
        [raw[:, observed_classes.index(label)] for label in TRANSITION_CLASSES]
    )
    row_sums = ordered.sum(axis=1)
    if not np.all(np.isfinite(ordered)) or np.any(ordered < 0) or np.any(row_sums <= 0):
        raise X11DataError("transition model emitted invalid probabilities")
    normalized = ordered / row_sums[:, None]
    if not np.allclose(
        normalized.sum(axis=1),
        np.ones(len(normalized)),
        rtol=0,
        atol=1e-12,
    ):
        raise X11DataError("transition probabilities are not normalized")
    return normalized


def _transition_point_metrics(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    class_index = {label: index for index, label in enumerate(TRANSITION_CLASSES)}
    try:
        encoded = np.asarray([class_index[str(value)] for value in targets], dtype=int)
    except KeyError as error:
        raise X11DataError("transition target is outside frozen state space") from error
    selected = np.clip(
        probabilities[np.arange(len(encoded)), encoded],
        1e-15,
        1.0,
    )
    indicator = np.eye(len(TRANSITION_CLASSES), dtype=float)[encoded]
    return (
        float(np.mean(np.sum((probabilities - indicator) ** 2, axis=1))),
        float(-np.mean(np.log(selected))),
    )


def _evaluate_transitions(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, object]:
    probability_columns = [f"probability_{label}" for label in TRANSITION_CLASSES]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    targets = predictions["next_drive_outcome"].to_numpy(dtype=object)
    brier, log_loss = _transition_point_metrics(targets, probabilities)
    groups = predictions["game_id"].to_numpy(dtype=object)
    unique_groups = tuple(dict.fromkeys(groups.tolist()))
    if len(unique_groups) < 2:
        raise X11DataError(
            "transition game-cluster bootstrap requires at least two games"
        )
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(X11_SEED)
    sampled_brier: list[float] = []
    sampled_log_loss: list[float] = []
    for _ in range(bootstrap_samples):
        selected_groups = rng.choice(
            len(unique_groups), size=len(unique_groups), replace=True
        )
        indices = np.concatenate(
            [indices_by_group[unique_groups[int(index)]] for index in selected_groups]
        )
        sampled = _transition_point_metrics(targets[indices], probabilities[indices])
        sampled_brier.append(sampled[0])
        sampled_log_loss.append(sampled[1])
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "classes": TRANSITION_CLASSES,
        "brier": brier,
        "brier_definition": "mean_sum_squared_class_error",
        "log_loss": log_loss,
        "bootstrap_ci": {
            "brier": (
                float(np.quantile(sampled_brier, alpha)),
                float(np.quantile(sampled_brier, 1.0 - alpha)),
            ),
            "log_loss": (
                float(np.quantile(sampled_log_loss, alpha)),
                float(np.quantile(sampled_log_loss, 1.0 - alpha)),
            ),
        },
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_valid": bootstrap_samples,
        "confidence_level": confidence_level,
        "clusters": len(unique_groups),
        "observations": len(predictions),
        "seed": X11_SEED,
    }


def _dataframe_content_sha256(frame: pd.DataFrame, context: str) -> str:
    return _frame_sha256(
        frame,
        columns=tuple(str(column) for column in frame.columns),
        context=context,
    )


def _fixed_point_probabilities(
    probabilities: dict[str, float],
    *,
    scale: int = 15,
) -> dict[str, dict[str, object]]:
    if set(probabilities) != set(TRANSITION_CLASSES):
        raise X11DataError("fixed-point probabilities must match transition classes")
    values = np.asarray(
        [probabilities[label] for label in TRANSITION_CLASSES],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise X11DataError("transition probabilities cannot be fixed-point encoded")
    denominator = 10**scale
    scaled = values * denominator
    atoms = np.floor(scaled).astype(object)
    remainder = denominator - int(sum(int(value) for value in atoms))
    order = sorted(
        range(len(TRANSITION_CLASSES)),
        key=lambda index: (-(scaled[index] - math.floor(scaled[index])), index),
    )
    for index in order[:remainder]:
        atoms[index] = int(atoms[index]) + 1
    if sum(int(value) for value in atoms) != denominator:
        raise X11DataError("fixed-point transition probabilities are not exact")
    return {
        label: {"atoms": str(int(atoms[index])), "scale": scale}
        for index, label in enumerate(TRANSITION_CLASSES)
    }


def _nominal_pit_cutoff_at(row: Any) -> str:
    remaining = float(row.game_seconds_remaining)
    if not math.isfinite(remaining) or not 0 <= remaining <= 3600:
        raise X11DataError("game_seconds_remaining cannot define a PIT cutoff")
    game_date = pd.Timestamp(row.game_date)
    if game_date.tzinfo is None or str(game_date.tz) != "UTC":
        raise X11DataError("transition game_date must be timezone-aware UTC")
    cutoff = game_date + pd.Timedelta(seconds=3600.0 - remaining)
    return cutoff.isoformat().replace("+00:00", "Z")


def _transition_config_sha256(*, minimum_prior_train_games: int) -> str:
    return _sha256(
        {
            "model_id": X11_TRANSITION_MODEL_ID,
            "model_version": X11_TRANSITION_MODEL_VERSION,
            "estimator": "StandardScaler+LogisticRegression",
            "solver": "lbfgs",
            "max_iter": 1000,
            "features": list(GAME_STATE_FEATURES),
            "state_space": list(TRANSITION_CLASSES),
            "training_rule": "game_date_strictly_less_than_test_game_date",
            "minimum_prior_train_games": minimum_prior_train_games,
            "seed": X11_SEED,
        }
    )


def _transition_contract_document(
    row: Any,
    *,
    data_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    pit_cutoff_at = str(row.pit_cutoff_at)
    features = {
        name: float(getattr(row, name))
        for name in GAME_STATE_FEATURES
    }
    feature_sha256 = _sha256(
        {
            "native_game_id": str(row.game_id),
            "drive_number": int(row.drive_number),
            "play_id": float(row.play_id),
            "pit_cutoff_at": pit_cutoff_at,
            "features": features,
            "manifest_sha256": str(row.manifest_sha256),
            "object_sha256": str(row.object_sha256),
            "schema_fingerprint": str(row.schema_fingerprint),
        }
    )
    event_digest = hashlib.sha256(
        _canonical_bytes(
            {
                "experiment_id": X11_EXPERIMENT_ID,
                "native_game_id": str(row.game_id),
                "drive_number": int(row.drive_number),
                "play_id": float(row.play_id),
                "pit_cutoff_at": pit_cutoff_at,
                "data_sha256": data_sha256,
                "config_sha256": config_sha256,
            }
        )
    ).hexdigest()
    probabilities = {
        label: float(getattr(row, f"probability_{label}"))
        for label in TRANSITION_CLASSES
    }
    return {
        "contract_version": "v1",
        "model_id": X11_TRANSITION_MODEL_ID,
        "model_version": X11_TRANSITION_MODEL_VERSION,
        "experiment_id": X11_EXPERIMENT_ID,
        "run_id": (
            "run_x11_"
            + data_sha256.removeprefix(_HASH_PREFIX)[:16]
            + "_"
            + config_sha256.removeprefix(_HASH_PREFIX)[:16]
        ),
        "game_id": f"game_nflverse_{row.game_id}",
        "state_event_id": f"evt_{event_digest}",
        "pit_cutoff_at": pit_cutoff_at,
        "output_kind": "state_transition",
        "transition_unit": "drive",
        "state_space": list(TRANSITION_CLASSES),
        "horizon": "next_state_transition",
        "probabilities": _fixed_point_probabilities(probabilities),
        "feature_sha256": feature_sha256,
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "quality_flags": [
            "preliminary_rules",
            "source_clock_unverified",
        ],
    }


def _transition_contract_output(
    row: Any,
    *,
    program_root: Path,
    data_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    document = _transition_contract_document(
        row,
        data_sha256=data_sha256,
        config_sha256=config_sha256,
    )
    try:
        validated = contracts.validate_contract_v1(
            program_root,
            "model-output/v1.schema.yaml",
            document,
        )
    except (TypeError, ValueError) as error:
        raise X11DataError(
            "drive transition failed the registry-backed model-output v1 contract"
        ) from error
    if not isinstance(validated, contracts.ModelOutputV1):
        raise X11DataError("model-output v1 validator returned an invalid type")
    return validated.model_dump(mode="json")


def _season_stability(
    predictions: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    minimum_valid_bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, object]:
    model_columns = {
        "spread_prior": "spread_prior_probability",
        "logistic": "logistic_probability",
        "gbdt": "gbdt_probability",
        "nflfastr_home_wp": "nflfastr_home_wp_probability",
    }
    reports: dict[str, object] = {}
    for season in sorted(int(value) for value in predictions["season"].unique()):
        season_predictions = predictions.loc[predictions["season"] == season]
        binary = season_predictions.loc[
            season_predictions["home_win"].notna()
        ].copy()
        if binary["game_id"].nunique() < 2 or binary["home_win"].nunique() != 2:
            raise X11DataError(
                f"season {season} cannot support registered stability metrics"
            )
        y_true = binary["home_win"].to_numpy(dtype=int)
        groups = binary["game_id"].to_numpy(dtype=object)
        prior = binary["spread_prior_probability"].to_numpy(dtype=float)
        models: dict[str, object] = {}
        try:
            for model_name, column in model_columns.items():
                probabilities = binary[column].to_numpy(dtype=float)
                report = evaluate_probabilities(
                    y_true,
                    probabilities,
                    groups=groups,
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    minimum_valid_samples=minimum_valid_bootstrap_samples,
                    seed=X11_SEED,
                )
                if model_name != "spread_prior":
                    report["paired_model_minus_prior"] = evaluate_model_vs_prior(
                        y_true,
                        probabilities,
                        prior,
                        groups=groups,
                        bootstrap_samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        minimum_valid_samples=minimum_valid_bootstrap_samples,
                        seed=X11_SEED,
                    )
                models[model_name] = report
        except ValidationInputError as error:
            raise X11DataError(
                f"season {season} stability evaluation failed closed"
            ) from error
        season_transitions = transitions.loc[transitions["season"] == season]
        transition_report = _evaluate_transitions(
            season_transitions,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        )
        transition_report["games"] = int(
            season_transitions["game_id"].nunique()
        )
        reports[str(season)] = {
            "outcome_models": models,
            "binary_games": int(binary["game_id"].nunique()),
            "binary_observations": len(binary),
            "transition": transition_report,
        }
    return reports


def _outcome_metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    minimum_valid_bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, dict[str, object]]:
    binary = predictions.loc[predictions["home_win"].notna()].copy()
    if binary["game_id"].nunique() < 2 or binary["home_win"].nunique() != 2:
        raise X11DataError(
            "evaluation must contain at least two games and both outcomes"
        )
    y_true = binary["home_win"].to_numpy(dtype=int)
    groups = binary["game_id"].to_numpy(dtype=object)
    model_columns = {
        "spread_prior": "spread_prior_probability",
        "logistic": "logistic_probability",
        "gbdt": "gbdt_probability",
        "nflfastr_home_wp": "nflfastr_home_wp_probability",
    }
    prior_probability = binary["spread_prior_probability"].to_numpy(dtype=float)
    metrics: dict[str, dict[str, object]] = {}
    try:
        for model_name, column in model_columns.items():
            probabilities = binary[column].to_numpy(dtype=float)
            report = evaluate_probabilities(
                y_true,
                probabilities,
                groups=groups,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                minimum_valid_samples=minimum_valid_bootstrap_samples,
                seed=X11_SEED,
            )
            if model_name != "spread_prior":
                report["paired_model_minus_prior"] = evaluate_model_vs_prior(
                    y_true,
                    probabilities,
                    prior_probability,
                    groups=groups,
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    minimum_valid_samples=minimum_valid_bootstrap_samples,
                    seed=X11_SEED,
                )
            metrics[model_name] = report
    except ValidationInputError as error:
        raise X11DataError("X-11 calibration evaluation failed closed") from error
    return metrics


def _tie_report_from_predictions(
    predictions: pd.DataFrame,
) -> dict[str, object]:
    tie_ids = tuple(
        sorted(
            predictions.loc[
                predictions["final_outcome"] == "tie",
                "game_id",
            ].unique()
        )
    )
    return {
        "games_reported": len(tie_ids),
        "game_ids": tie_ids,
        "drive_rows_excluded": int(
            (predictions["final_outcome"] == "tie").sum()
        ),
        "excluded_from_binary_calibration": True,
    }


def _validate_walk_forward_parameters(
    *,
    evaluation_game_limit: int | None,
    minimum_prior_train_games: int,
    bootstrap_samples: int,
    minimum_valid_bootstrap_samples: int,
    confidence_level: float,
    gbdt_max_iter: int,
) -> None:
    if evaluation_game_limit is not None and (
        type(evaluation_game_limit) is not int or evaluation_game_limit < 2
    ):
        raise X11DataError("evaluation_game_limit must be None or an integer >= 2")
    if type(minimum_prior_train_games) is not int or minimum_prior_train_games < 2:
        raise X11DataError("minimum_prior_train_games must be an integer >= 2")
    if type(bootstrap_samples) is not int or bootstrap_samples < 20:
        raise X11DataError("bootstrap_samples must be an integer >= 20")
    if (
        type(minimum_valid_bootstrap_samples) is not int
        or minimum_valid_bootstrap_samples < 20
        or minimum_valid_bootstrap_samples > bootstrap_samples
    ):
        raise X11DataError(
            "minimum_valid_bootstrap_samples must be between 20 and bootstrap_samples"
        )
    if type(confidence_level) is not float or not 0 < confidence_level < 1:
        raise X11DataError("confidence_level must be a float in (0, 1)")
    if type(gbdt_max_iter) is not int or gbdt_max_iter < 1:
        raise X11DataError("gbdt_max_iter must be a positive integer")


def run_x11_walk_forward(
    loaded: X11LoadedDataset,
    *,
    program_root: str | Path,
    evaluation_game_limit: int | None = None,
    minimum_prior_train_games: int = 32,
    bootstrap_samples: int = 200,
    minimum_valid_bootstrap_samples: int = 100,
    confidence_level: float = 0.95,
    gbdt_max_iter: int = 50,
) -> X11Evaluation:
    """Run game-grouped 2020-2025 folds using strict prior game dates."""

    _validate_walk_forward_parameters(
        evaluation_game_limit=evaluation_game_limit,
        minimum_prior_train_games=minimum_prior_train_games,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
        gbdt_max_iter=gbdt_max_iter,
    )
    if not isinstance(loaded, X11LoadedDataset):
        raise TypeError("loaded must be an X11LoadedDataset")
    if inventory_sha256(loaded.inventory) != loaded.inventory.inventory_sha256:
        raise X11DataError("input inventory self-hash is invalid")
    if _validate_digest(
        loaded.chronology_sha256, "chronology_sha256"
    ) != chronology_sha256(loaded.drive_starts):
        raise X11DataError("frozen game chronology SHA-256 is invalid")
    if _validate_digest(
        loaded.normalized_frame_sha256,
        "normalized_frame_sha256",
    ) != normalized_frame_sha256(loaded.drive_starts):
        raise X11DataError("frozen normalized frame SHA-256 is invalid")
    frame = attach_point_in_time_spread_prior(
        loaded.drive_starts,
        minimum_train_games=minimum_prior_train_games,
    )
    missing_features = sorted(set(GAME_STATE_FEATURES) - set(frame.columns))
    if missing_features:
        raise X11DataError(
            f"canonical drive frame is missing features: {missing_features}"
        )
    for feature in GAME_STATE_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    evaluation_input_hash = evaluation_input_sha256(frame)
    evaluation_games = (
        frame.loc[frame["season"].between(2020, 2025)][["game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"], kind="mergesort")
    )
    if evaluation_game_limit is not None:
        evaluation_games = evaluation_games.head(evaluation_game_limit)
    if len(evaluation_games) < 2:
        raise X11DataError("X-11 evaluation requires at least two games")

    outcome_predictions: list[pd.DataFrame] = []
    transition_predictions: list[pd.DataFrame] = []
    folds: list[X11FoldAudit] = []
    selected_game_ids = set(evaluation_games["game_id"])
    selected = frame.loc[frame["game_id"].isin(selected_game_ids)].copy()
    for cutoff, date_games in evaluation_games.groupby("game_date", sort=True):
        train = frame.loc[
            (frame["game_date"] < cutoff) & frame["spread_prior"].notna()
        ].copy()
        if train.empty:
            raise X11DataError(
                "no strictly earlier complete games have an available spread prior"
            )
        if train[list(GAME_STATE_FEATURES)].isna().any().any():
            raise X11DataError("strictly earlier training features are incomplete")
        train_max = train["game_date"].max()
        if not train_max < cutoff:
            raise X11DataError("walk-forward training is not strictly earlier")
        logistic, gbdt = _outcome_models(
            train,
            gbdt_max_iter=gbdt_max_iter,
        )
        transition_model = _transition_model(train)
        date_game_ids = set(date_games["game_id"])
        date_test = selected.loc[selected["game_id"].isin(date_game_ids)].copy()
        if (
            date_test["spread_prior"].isna().any()
            or date_test[list(GAME_STATE_FEATURES)].isna().any().any()
        ):
            raise X11DataError("evaluation game has no point-in-time spread prior")
        features = date_test.loc[:, GAME_STATE_FEATURES].to_numpy(dtype=float)
        date_test["logistic_probability"] = _positive_class_probability(
            logistic, features
        )
        date_test["gbdt_probability"] = _positive_class_probability(gbdt, features)
        date_test["spread_prior_probability"] = date_test["spread_prior"].astype(float)
        date_test["nflfastr_home_wp_probability"] = date_test["home_wp"].astype(float)
        date_test["pit_cutoff"] = cutoff
        date_test["inventory_sha256"] = loaded.inventory.inventory_sha256
        outcome_predictions.append(date_test)

        transition_matrix = _fixed_transition_probabilities(transition_model, features)
        transition = date_test[
            [
                "game_id",
                "season",
                "season_type",
                "game_date",
                "drive_number",
                "play_id",
                *GAME_STATE_FEATURES,
                "next_drive_outcome",
                "manifest_sha256",
                "object_sha256",
                "schema_fingerprint",
                "inventory_sha256",
            ]
        ].copy()
        transition["pit_cutoff_at"] = [
            _nominal_pit_cutoff_at(row)
            for row in transition.itertuples(index=False)
        ]
        transition["pit_cutoff_basis"] = (
            "native_preplay_state_at_play_id;training_game_date_strictly_less"
        )
        transition["pit_status"] = X11_PIT_STATUS
        for class_index, label in enumerate(TRANSITION_CLASSES):
            transition[f"probability_{label}"] = transition_matrix[:, class_index]
        transition_predictions.append(transition)

        train_game_count = int(
            frame.loc[frame["game_date"] < cutoff, "game_id"].nunique()
        )
        binary_train_game_count = int(
            train.loc[train["home_win"].notna(), "game_id"].nunique()
        )
        for game_id in date_games["game_id"]:
            game_test = date_test.loc[date_test["game_id"] == game_id]
            folds.append(
                X11FoldAudit(
                    test_game_id=str(game_id),
                    pit_cutoff=cutoff,
                    train_max_game_date=train_max,
                    train_game_count=train_game_count,
                    binary_train_game_count=binary_train_game_count,
                    train_drive_rows=len(train),
                    test_game_count=1,
                    test_drive_rows=len(game_test),
                    prior_train_game_count=int(
                        game_test["prior_train_game_count"].iloc[0]
                    ),
                )
            )
    predictions = (
        pd.concat(outcome_predictions, ignore_index=True)
        .sort_values(
            ["game_date", "game_id", "drive_number", "play_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    transitions = (
        pd.concat(transition_predictions, ignore_index=True)
        .sort_values(
            ["game_date", "game_id", "drive_number", "play_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    transition_config_hash = _transition_config_sha256(
        minimum_prior_train_games=minimum_prior_train_games
    )
    transition_outputs = tuple(
        _transition_contract_output(
            row,
            program_root=Path(program_root),
            data_sha256=evaluation_input_hash,
            config_sha256=transition_config_hash,
        )
        for row in transitions.itertuples(index=False)
    )
    predictions_hash = _dataframe_content_sha256(
        predictions,
        "X-11 outcome predictions",
    )
    transition_predictions_hash = _dataframe_content_sha256(
        transitions,
        "X-11 transition predictions",
    )
    transition_outputs_hash = _sha256(list(transition_outputs))
    metrics = _outcome_metrics_from_predictions(
        predictions,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
    )
    tie_report = _tie_report_from_predictions(predictions)
    season_stability = _season_stability(
        predictions,
        transitions,
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
    )
    evaluation_without_hash = X11Evaluation(
        result_label=X11_RESULT_LABEL,
        seed=X11_SEED,
        predictions=predictions,
        transition_predictions=transitions,
        folds=tuple(folds),
        outcome_metrics=metrics,
        transition_metrics=_evaluate_transitions(
            transitions,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        ),
        tie_report=tie_report,
        season_stability=season_stability,
        model_features={
            "logistic": GAME_STATE_FEATURES,
            "gbdt": GAME_STATE_FEATURES,
            "drive_transition": GAME_STATE_FEATURES,
        },
        normalized_frame_sha256=loaded.normalized_frame_sha256,
        evaluation_input_sha256=evaluation_input_hash,
        predictions_sha256=predictions_hash,
        transition_predictions_sha256=transition_predictions_hash,
        transition_outputs=transition_outputs,
        transition_outputs_sha256=transition_outputs_hash,
        evaluation_sha256="",
        bootstrap_samples=bootstrap_samples,
        minimum_valid_bootstrap_samples=minimum_valid_bootstrap_samples,
        confidence_level=confidence_level,
        evaluation_game_limit=evaluation_game_limit,
        minimum_prior_train_games=minimum_prior_train_games,
        gbdt_max_iter=gbdt_max_iter,
    )
    return replace(
        evaluation_without_hash,
        evaluation_sha256=evaluation_sha256(evaluation_without_hash),
    )


_REGISTRATION_LOCK_IDS = (
    "nfl_data_manifest_and_version",
    "spread_prior_manifest",
    "pit_feature_contract",
    "model_config_and_seed",
    "bootstrap_parameters",
    "tie_policy",
    "h_split_approval",
)


def _validate_evaluation_integrity(
    loaded: X11LoadedDataset,
    evaluation: X11Evaluation,
    *,
    program_root: Path,
) -> None:
    if inventory_sha256(loaded.inventory) != loaded.inventory.inventory_sha256:
        raise X11DataError("input inventory self-hash is invalid")
    if chronology_sha256(loaded.drive_starts) != loaded.chronology_sha256:
        raise X11DataError("loaded chronology integrity check failed")
    if normalized_frame_sha256(loaded.drive_starts) != (
        loaded.normalized_frame_sha256
    ):
        raise X11DataError("loaded normalized-frame integrity check failed")
    if evaluation.result_label != X11_RESULT_LABEL:
        raise X11DataError("evaluation result label differs from registration")
    if evaluation.normalized_frame_sha256 != loaded.normalized_frame_sha256:
        raise X11DataError("evaluation is not bound to the normalized frame")

    derived = attach_point_in_time_spread_prior(
        loaded.drive_starts,
        minimum_train_games=evaluation.minimum_prior_train_games,
    )
    for feature in GAME_STATE_FEATURES:
        derived[feature] = pd.to_numeric(derived[feature], errors="coerce")
    if evaluation_input_sha256(derived) != evaluation.evaluation_input_sha256:
        raise X11DataError("evaluation input integrity check failed")
    if _dataframe_content_sha256(
        evaluation.predictions,
        "X-11 outcome predictions",
    ) != evaluation.predictions_sha256:
        raise X11DataError("outcome prediction integrity check failed")
    if _dataframe_content_sha256(
        evaluation.transition_predictions,
        "X-11 transition predictions",
    ) != evaluation.transition_predictions_sha256:
        raise X11DataError("transition prediction integrity check failed")
    if _sha256(list(evaluation.transition_outputs)) != (
        evaluation.transition_outputs_sha256
    ):
        raise X11DataError("transition output integrity check failed")
    if len(evaluation.transition_outputs) != len(
        evaluation.transition_predictions
    ):
        raise X11DataError("transition output count integrity check failed")
    config_sha256 = _transition_config_sha256(
        minimum_prior_train_games=evaluation.minimum_prior_train_games
    )
    expected_outputs = tuple(
        contracts.ModelOutputV1.model_validate(
            _transition_contract_document(
                row,
                data_sha256=evaluation.evaluation_input_sha256,
                config_sha256=config_sha256,
            )
        ).model_dump(mode="json")
        for row in evaluation.transition_predictions.itertuples(index=False)
    )
    if expected_outputs != evaluation.transition_outputs:
        raise X11DataError(
            "transition model-output v1 documents fail integrity comparison"
        )
    try:
        registry_validated = contracts.validate_contract_v1(
            program_root,
            "model-output/v1.schema.yaml",
            evaluation.transition_outputs[0],
        )
    except (TypeError, ValueError) as error:
        raise X11DataError(
            "transition output registry binding integrity check failed"
        ) from error
    if registry_validated.model_dump(mode="json") != (
        evaluation.transition_outputs[0]
    ):
        raise X11DataError("transition output registry validation changed data")
    expected_outcome_metrics = _outcome_metrics_from_predictions(
        evaluation.predictions,
        bootstrap_samples=evaluation.bootstrap_samples,
        minimum_valid_bootstrap_samples=(
            evaluation.minimum_valid_bootstrap_samples
        ),
        confidence_level=evaluation.confidence_level,
    )
    if _json_ready(expected_outcome_metrics) != _json_ready(
        evaluation.outcome_metrics
    ):
        raise X11DataError("outcome metric integrity check failed")
    expected_transition_metrics = _evaluate_transitions(
        evaluation.transition_predictions,
        bootstrap_samples=evaluation.bootstrap_samples,
        confidence_level=evaluation.confidence_level,
    )
    if _json_ready(expected_transition_metrics) != _json_ready(
        evaluation.transition_metrics
    ):
        raise X11DataError("transition metric integrity check failed")
    if _json_ready(_tie_report_from_predictions(evaluation.predictions)) != (
        _json_ready(evaluation.tie_report)
    ):
        raise X11DataError("tie-report integrity check failed")
    expected_season_stability = _season_stability(
        evaluation.predictions,
        evaluation.transition_predictions,
        bootstrap_samples=evaluation.bootstrap_samples,
        minimum_valid_bootstrap_samples=(
            evaluation.minimum_valid_bootstrap_samples
        ),
        confidence_level=evaluation.confidence_level,
    )
    if _json_ready(expected_season_stability) != _json_ready(
        evaluation.season_stability
    ):
        raise X11DataError("season-stability integrity check failed")
    expected_game_ids = tuple(
        evaluation.predictions[
            ["game_id", "game_date"]
        ]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"], kind="mergesort")["game_id"]
        .astype(str)
    )
    if tuple(fold.test_game_id for fold in evaluation.folds) != expected_game_ids:
        raise X11DataError("walk-forward fold integrity check failed")
    if evaluation.model_features != {
        "logistic": GAME_STATE_FEATURES,
        "gbdt": GAME_STATE_FEATURES,
        "drive_transition": GAME_STATE_FEATURES,
    }:
        raise X11DataError("model feature-contract integrity check failed")
    if evaluation_sha256(evaluation) != evaluation.evaluation_sha256:
        raise X11DataError("evaluation result integrity check failed")


def build_x11_evidence(
    loaded: X11LoadedDataset,
    evaluation: X11Evaluation,
    *,
    program_root: str | Path,
    execution_mode: str,
) -> dict[str, object]:
    """Build non-formal, self-hashed evidence for a real-data pipeline run."""

    if not isinstance(loaded, X11LoadedDataset):
        raise TypeError("loaded must be an X11LoadedDataset")
    if not isinstance(evaluation, X11Evaluation):
        raise TypeError("evaluation must be an X11Evaluation")
    if execution_mode not in {"bounded_smoke", "full"}:
        raise X11DataError("execution_mode must be bounded_smoke or full")
    try:
        _validate_evaluation_integrity(
            loaded,
            evaluation,
            program_root=Path(program_root),
        )
    except (TypeError, ValueError) as error:
        raise X11DataError(
            "X-11 evaluation integrity validation failed"
        ) from error
    for frame in (
        evaluation.predictions,
        evaluation.transition_predictions,
    ):
        if (
            frame.empty
            or not frame["inventory_sha256"].eq(loaded.inventory.inventory_sha256).all()
        ):
            raise X11DataError(
                "evaluation predictions do not bind to the input inventory"
            )

    if execution_mode == "full":
        expected_games = set(
            loaded.drive_starts.loc[
                loaded.drive_starts["season"].between(2020, 2025),
                "game_id",
            ].unique()
        )
        evaluated_games = {fold.test_game_id for fold in evaluation.folds}
        observed_seasons = set(
            int(value) for value in evaluation.predictions["season"].unique()
        )
        if (
            evaluation.evaluation_game_limit is not None
            or evaluated_games != expected_games
            or observed_seasons != set(range(2020, 2026))
        ):
            raise X11DataError(
                "full X-11 evidence requires every 2020-2025 game and season"
            )
    inventory_document = {
        **_inventory_material(loaded.inventory),
        "inventory_sha256": loaded.inventory.inventory_sha256,
    }
    fold_documents = [_json_ready(asdict(fold)) for fold in evaluation.folds]
    evidence_without_hash: dict[str, object] = {
        "artifact_type": "x11_real_data_pipeline_evidence_v0",
        "experiment_id": "X-11",
        "result_label": X11_RESULT_LABEL,
        "execution_mode": execution_mode,
        "is_formal_result": False,
        "formal_result_eligible": False,
        "authorization_scope": "preregistered_pipeline",
        "seed": X11_SEED,
        "input_inventory": inventory_document,
        "chronology_sha256": loaded.chronology_sha256,
        "normalized_frame_sha256": loaded.normalized_frame_sha256,
        "evaluation_input_sha256": evaluation.evaluation_input_sha256,
        "predictions_sha256": evaluation.predictions_sha256,
        "transition_predictions_sha256": (
            evaluation.transition_predictions_sha256
        ),
        "transition_outputs_sha256": evaluation.transition_outputs_sha256,
        "evaluation_sha256": evaluation.evaluation_sha256,
        "adapter_audit": asdict(loaded.adapter_audit),
        "pit_assessment": {
            "spread_observation_timestamp_proven": False,
            "reason": (
                "nflverse spread_line has no exact prior observation timestamp; "
                "the train-only mapping is chronological, but source availability "
                "before each game cannot be proven from this object"
            ),
            "method_status": X11_PIT_STATUS,
            "prior_method": ("logistic_spread_line_strict_prior_game_dates"),
            "same_date_games_allowed_in_training": False,
            "transition_cutoff_semantics": (
                "nominal UTC game_date midnight plus elapsed regulation-clock "
                "seconds; not an observed wall-clock timestamp"
            ),
            "transition_cutoff_quality_flag": "source_clock_unverified",
        },
        "registration_locks": [
            {"id": lock_id, "status": "registry_unresolved"}
            for lock_id in _REGISTRATION_LOCK_IDS
        ],
        "feature_contract": {
            "logistic": list(GAME_STATE_FEATURES),
            "gbdt": list(GAME_STATE_FEATURES),
            "drive_transition": list(GAME_STATE_FEATURES),
            "spread_prior_is_model_input": True,
            "nflfastr_home_wp_role": "comparator_only",
            "prohibited_as_features": [
                "home_score",
                "away_score",
                "fixed_drive_result",
                "final_outcome",
                "home_win",
                "next_drive_outcome",
                "home_wp",
            ],
        },
        "model_configuration": {
            "logistic": {
                "estimator": "StandardScaler+LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 1000,
                "seed": X11_SEED,
            },
            "gbdt": {
                "estimator": "HistGradientBoostingClassifier",
                "learning_rate": 0.05,
                "max_iter": evaluation.gbdt_max_iter,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                "seed": X11_SEED,
            },
            "drive_transition": {
                "estimator": "StandardScaler+LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 1000,
                "state_space": list(TRANSITION_CLASSES),
                "seed": X11_SEED,
            },
        },
        "walk_forward": {
            "warmup_seasons": [2015, 2016, 2017, 2018, 2019],
            "evaluation_seasons": [2020, 2021, 2022, 2023, 2024, 2025],
            "training_rule": (
                "complete games with game_date strictly less than test game_date"
            ),
            "test_unit": "one complete game",
            "same_date_training_excluded": True,
            "minimum_prior_train_games": evaluation.minimum_prior_train_games,
            "evaluation_game_limit": evaluation.evaluation_game_limit,
            "evaluated_games": len(evaluation.folds),
            "folds": fold_documents,
            "seed": evaluation.seed,
        },
        "bootstrap": {
            "cluster_unit": "game_id",
            "confidence_level": evaluation.confidence_level,
            "samples_requested": evaluation.bootstrap_samples,
            "minimum_valid_samples": (evaluation.minimum_valid_bootstrap_samples),
            "seed": evaluation.seed,
        },
        "outcome_evaluation": {
            "models": _json_ready(evaluation.outcome_metrics),
            "ties": _json_ready(evaluation.tie_report),
            "binary_drive_rows": int(evaluation.predictions["home_win"].notna().sum()),
            "binary_games": int(
                evaluation.predictions.loc[
                    evaluation.predictions["home_win"].notna(), "game_id"
                ].nunique()
            ),
        },
        "season_stability": _json_ready(evaluation.season_stability),
        "transition_evaluation": {
            "state_space": list(TRANSITION_CLASSES),
            "output_contract": "model-output/v1.schema.yaml",
            "metrics": _json_ready(evaluation.transition_metrics),
            "model_outputs": _json_ready(list(evaluation.transition_outputs)),
            "observed_targets": [
                {
                    "state_event_id": output["state_event_id"],
                    "native_game_id": row.game_id,
                    "drive_number": int(row.drive_number),
                    "play_id": float(row.play_id),
                    "observed_next_drive_outcome": row.next_drive_outcome,
                    "lineage": {
                        "inventory_sha256": row.inventory_sha256,
                        "manifest_sha256": row.manifest_sha256,
                        "object_sha256": row.object_sha256,
                        "schema_fingerprint": row.schema_fingerprint,
                    },
                }
                for row, output in zip(
                    evaluation.transition_predictions.itertuples(index=False),
                    evaluation.transition_outputs,
                    strict=True,
                )
            ],
        },
    }
    ready = _json_ready(evidence_without_hash)
    ready["evidence_sha256"] = evidence_sha256(ready)
    return ready


def write_x11_evidence(
    path: str | Path,
    evidence: dict[str, object],
) -> Path:
    """Persist one canonical machine-readable X-11 evidence document."""

    destination = Path(path)
    ready = _json_ready(evidence)
    expected_hash = ready.get("evidence_sha256")
    if expected_hash != evidence_sha256(ready):
        raise X11DataError("evidence self-hash is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            ready,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_x11_dataset(
    *,
    store_root: str | Path,
    program_root: str | Path,
    manifest_paths: Iterable[str | Path] | None = None,
) -> X11LoadedDataset:
    """Verify exactly 11 governed partitions and adapt drive-start states."""

    store = Path(store_root)
    program = Path(program_root)
    paths = (
        tuple(Path(path) for path in manifest_paths)
        if manifest_paths is not None
        else _discover_manifest_paths(store)
    )
    if len(paths) != len(X11_YEARS):
        raise X11DataError(f"X-11 requires 11 manifests; received {len(paths)}")
    if len(set(paths)) != len(paths):
        raise X11DataError("X-11 manifest paths must be unique")

    partitions: list[X11PartitionInventory] = []
    frames: list[pd.DataFrame] = []
    native_drives = 0
    seen_years: set[int] = set()
    seen_manifest_hashes: set[str] = set()
    seen_object_hashes: set[str] = set()
    for path in paths:
        try:
            verified = read_verified_static_object(
                path,
                store_root=store,
                program_root=program,
            )
            record = verified.record
            year = _partition_year(record.partition)
            audit = inspect_nflverse_partition(
                verified.object_bytes,
                expected_year=year,
            )
        except NFLVerseSourceError as error:
            raise X11DataError("verified NFL partition failed native audit") from error
        if year in seen_years:
            raise X11DataError(f"duplicate X-11 season manifest: {year}")
        if (
            record.source != X11_SOURCE
            or record.dataset != X11_DATASET_ID
            or record.version != X11_NFLVERSE_VERSION
            or record.extension != "parquet"
        ):
            raise X11DataError("verified object is outside the frozen X-11 source")
        manifest = record.manifest
        if not _matches_frozen_partition_source(manifest, year=year):
            raise X11DataError(
                f"verified season-{year} manifest does not match the frozen source "
                "URL, release asset, object SHA-256, schema SHA-256, and license"
            )
        manifest_hash = _validate_digest(manifest.manifest_sha256, "manifest_sha256")
        object_hash = _validate_digest(manifest.object_sha256, "object_sha256")
        schema_hash = _validate_digest(
            manifest.schema_fingerprint, "schema_fingerprint"
        )
        if (
            manifest.dataset_id != X11_DATASET_ID
            or manifest.upstream_partition != f"season-{year}"
            or manifest.object_kind != "byte_exact_original"
        ):
            raise X11DataError("verified manifest identity is not frozen X-11")
        if audit.object_sha256 != object_hash:
            raise X11DataError("native audit and manifest object hashes differ")
        if audit.schema_fingerprint != schema_hash:
            raise X11DataError("native audit and manifest schema hashes differ")
        if audit.season_types != ("POST", "REG"):
            raise X11DataError(f"season-{year} must contain both REG and POST")
        if manifest_hash in seen_manifest_hashes:
            raise X11DataError("duplicate manifest SHA-256 in X-11 inventory")
        if object_hash in seen_object_hashes:
            raise X11DataError("duplicate object SHA-256 in X-11 inventory")
        frame, partition_native_drives = _canonical_partition_frame(
            verified.object_bytes,
            year=year,
            manifest_sha256=manifest_hash,
            object_sha256=object_hash,
            schema_fingerprint=schema_hash,
        )
        partitions.append(
            X11PartitionInventory(
                year=year,
                partition=f"season-{year}",
                manifest_sha256=manifest_hash,
                object_sha256=object_hash,
                schema_fingerprint=schema_hash,
                rows=audit.row_count,
                games=audit.game_count,
                season_types=audit.season_types,
            )
        )
        frames.append(frame)
        native_drives += partition_native_drives
        seen_years.add(year)
        seen_manifest_hashes.add(manifest_hash)
        seen_object_hashes.add(object_hash)
    if seen_years != set(X11_YEARS):
        missing = sorted(set(X11_YEARS) - seen_years)
        unexpected = sorted(seen_years - set(X11_YEARS))
        raise X11DataError(
            f"X-11 year inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    ordered_partitions = tuple(sorted(partitions, key=lambda value: value.year))
    inventory_without_hash = X11InputInventory(
        dataset_id=X11_DATASET_ID,
        source=X11_SOURCE,
        version=X11_NFLVERSE_VERSION,
        years=X11_YEARS,
        partitions=ordered_partitions,
        total_rows=sum(value.rows for value in ordered_partitions),
        total_games=sum(value.games for value in ordered_partitions),
        season_types=("POST", "REG"),
        inventory_sha256="",
    )
    inventory = replace(
        inventory_without_hash,
        inventory_sha256=inventory_sha256(inventory_without_hash),
    )
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(
        ["game_date", "game_id", "drive_number", "play_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    game_order = {
        item["game_id"]: index for index, item in enumerate(_chronology_material(frame))
    }
    frame.insert(0, "game_order", frame["game_id"].map(game_order))
    observed_games = int(frame["game_id"].nunique())
    if observed_games != inventory.total_games:
        raise X11DataError("canonical game count differs from verified inventory")
    tie_games = int(frame.loc[frame["final_outcome"] == "tie", "game_id"].nunique())
    chronology_hash = chronology_sha256(frame)
    normalized_hash = normalized_frame_sha256(frame)
    return X11LoadedDataset(
        inventory=inventory,
        drive_starts=frame,
        chronology_sha256=chronology_hash,
        normalized_frame_sha256=normalized_hash,
        adapter_audit=X11AdapterAudit(
            native_drives=native_drives,
            canonical_drive_starts=len(frame),
            excluded_drives_without_complete_state=native_drives - len(frame),
            games=observed_games,
            ties=tie_games,
        ),
    )


__all__ = [
    "GAME_STATE_FEATURES",
    "NATIVE_STATE_FEATURES",
    "TRANSITION_CLASSES",
    "X11AdapterAudit",
    "X11DataError",
    "X11InputInventory",
    "X11Evaluation",
    "X11FoldAudit",
    "X11LoadedDataset",
    "X11PartitionInventory",
    "X11_DATASET_ID",
    "X11_EXPERIMENT_ID",
    "X11_FROZEN_PARTITION_ALLOWLIST",
    "X11_LICENSE_REF",
    "X11_LICENSE_STATUS",
    "X11_NFLVERSE_VERSION",
    "X11_PIT_STATUS",
    "X11_RESULT_LABEL",
    "X11_SEED",
    "X11_TRANSITION_MODEL_ID",
    "X11_TRANSITION_MODEL_VERSION",
    "X11_YEARS",
    "attach_point_in_time_spread_prior",
    "build_x11_evidence",
    "chronology_sha256",
    "evaluation_input_sha256",
    "evaluation_sha256",
    "evidence_sha256",
    "expected_nflverse_source_cursor",
    "inventory_sha256",
    "load_x11_dataset",
    "normalized_frame_sha256",
    "run_x11_walk_forward",
    "write_x11_evidence",
]
