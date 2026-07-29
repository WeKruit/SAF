"""Governed X-15 Stage A inference, evaluation, and publication."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.models import nfl_fastrmodels as _frozen
from prediction_market.program_audit import (
    ResearchRegistryError,
    load_dataset_registry,
)
from prediction_market.sports.nfl_reference_value import (
    CLAIM_BOUNDARY,
    EXPERIMENT_ID,
    GameReferenceTables,
    ReferenceModelBindingV1,
    ReferenceValueBuildError,
    build_game_reference_tables,
)
from prediction_market.sports.nfl_historical_sports_clean import (
    HistoricalSportsCleanError,
    VerifiedSource,
    discover_main_checkout,
    verify_frozen_2025_sources,
)
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)


BUILDER_VERSION: Final[str] = "nfl_x15_stage_a_reference_v1"
EXPECTED_X13_GAME_COUNT: Final[int] = 153
EXPECTED_X13_BATCH_INDEX_SHA256: Final[str] = (
    "sha256:"
    "db2fb125ea4d5e7844e22b27d6643e6f1c1ddc48db527d347ca789faacb38acd"
)
DEFAULT_X13_BATCH_INDEX: Final[Path] = Path(
    "artifacts/market-observation/nfl/x13/exact-153-facts-v3/"
    "batches/manifests/sha256/db/"
    "db2fb125ea4d5e7844e22b27d6643e6f1c1ddc48db527d347ca789faacb38acd"
    ".batch-index.json"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/stage-a-reference-v1"
)
DEFAULT_MODEL_MANIFEST: Final[Path] = Path(
    "var/raw/manifests/source=nflverse/"
    "dataset=DS-NFL-FASTRMODELS/"
    "version=9f2495fdb4943087ca663d96706eb5df7973aff4/"
    "partition=asset-253928623/"
    "080d98f34495fe59a532b7c24e17536f471700e92ac8415b682234d7241fe3cb"
    ".manifest.json"
)
BOOTSTRAP_SEED: Final[int] = 20260729


class StageAReferenceError(RuntimeError):
    """A Stage A governance, inference, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class StageACalibrationTables:
    summary: pd.DataFrame
    reliability: pd.DataFrame
    breakdowns: pd.DataFrame
    bootstrap: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PublishedStageAGame:
    game_id: str
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    table_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PublishedStageABatch:
    game_count: int
    index_path: Path
    index_sha256: str
    batch_sha256: str
    games: tuple[PublishedStageAGame, ...]
    table_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedX13Game:
    game_id: str
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    canonical_events_path: Path
    canonical_events_object_sha256: str
    canonical_events_semantic_sha256: str
    canonical_events_byte_length: int
    canonical_events_row_count: int
    canonical_events_schema_columns: tuple[str, ...]

    def input_binding(
        self,
        *,
        batch_index_sha256: str,
        batch_sha256: str,
        semantic_batch_sha256: str,
    ) -> dict[str, object]:
        return {
            "experiment_id": "X-13",
            "batch_index_sha256": batch_index_sha256,
            "batch_sha256": batch_sha256,
            "semantic_batch_sha256": semantic_batch_sha256,
            "game_manifest_sha256": self.manifest_sha256,
            "game_bundle_sha256": self.bundle_sha256,
            "canonical_events_object_sha256": (
                self.canonical_events_object_sha256
            ),
            "canonical_events_semantic_sha256": (
                self.canonical_events_semantic_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class _VerifiedX13Batch:
    root: Path
    index_path: Path
    index_sha256: str
    batch_sha256: str
    semantic_batch_sha256: str
    sources: dict[str, object]
    games: tuple[_VerifiedX13Game, ...]


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    raise StageAReferenceError(
        f"canonical material contains unsupported {type(value).__name__}"
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


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise StageAReferenceError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise StageAReferenceError(f"{label} must be a JSON object")
    return value


def _resolve_under(root: Path, relative: str | Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if (
        candidate != resolved_root
        and resolved_root not in candidate.parents
    ):
        raise StageAReferenceError(f"{label} escapes its declared root")
    return candidate


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise StageAReferenceError(
                f"content-addressed object collision: {path}"
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
                raise StageAReferenceError(
                    f"content-addressed publish race: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            str(column): _canonical_value(value)
            for column, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise StageAReferenceError(
            "Stage A table cannot be represented as Parquet"
        ) from exc
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )
    return (
        sink.getvalue().to_pybytes(),
        _sha256_bytes(table.schema.remove_metadata().serialize().to_pybytes()),
    )


def _publish_table(
    *,
    output_root: Path,
    relative_parent: Path,
    name: str,
    frame: pd.DataFrame,
) -> tuple[dict[str, object], Path]:
    payload, schema_fingerprint = _parquet_bytes(frame)
    object_sha = _sha256_bytes(payload)
    digest = object_sha.removeprefix("sha256:")
    relative = (
        relative_parent
        / "objects"
        / "sha256"
        / digest[:2]
        / f"{digest}.parquet"
    )
    target = _resolve_under(output_root, relative, label=f"{name} object")
    _atomic_publish(target, payload)
    descriptor = {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": schema_fingerprint,
        "semantic_rows_sha256": _canonical_sha256(
            {
                "name": name,
                "schema_columns": list(frame.columns),
                "rows": _frame_records(frame),
            }
        ),
    }
    return descriptor, target


def _require_x15_registry_binding(program_root: Path) -> None:
    try:
        matches = [
            row
            for row in load_dataset_registry(program_root)
            if row.dataset_id == _frozen.DATASET_ID
        ]
    except ResearchRegistryError as exc:
        raise StageAReferenceError(
            "dataset registry cannot prove the X-15 official model asset"
        ) from exc
    if len(matches) != 1:
        raise StageAReferenceError(
            "dataset registry must contain exactly one official model asset"
        )
    row = matches[0]
    if (
        row.catalog_item_ids != ("I-018",)
        or row.canonical_url != _frozen.ASSET_URL
        or row.license != "MIT"
        or row.license_status != "approved"
        or row.allowed_experiments != ("X-11", "X-13", "X-15")
        or row.manifest_sha256 != _frozen.ASSET_MANIFEST_SHA256
        or row.status != "registered"
    ):
        raise StageAReferenceError(
            "dataset registry differs from the current X-15 asset binding"
        )


def load_x15_official_no_spread_predictor(
    *,
    program_root: str | Path,
    store_root: str | Path,
    manifest_path: str | Path,
) -> _frozen.OfficialNoSpreadPredictor:
    """Load frozen bytes through the current X-15 registry authorization."""

    program = Path(program_root).resolve()
    _require_x15_registry_binding(program)
    try:
        _frozen._require_runtime_version()
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=program,
        )
        _frozen._require_verified_asset_identity(verified)
    except (StaticStoreError, _frozen.OfficialModelAssetError) as exc:
        raise StageAReferenceError(
            "official no-spread model bytes failed verification"
        ) from exc
    booster = _frozen.xgboost.Booster()
    try:
        booster.load_model(bytearray(verified.object_bytes))
    except _frozen.xgboost.core.XGBoostError as exc:
        raise StageAReferenceError(
            "verified official asset cannot be loaded as XGBoost UBJSON"
        ) from exc
    if (
        booster.num_features() != _frozen.EXPECTED_FEATURE_COUNT
        or booster.num_boosted_rounds() != _frozen.EXPECTED_BOOSTED_ROUNDS
    ):
        raise StageAReferenceError(
            "official booster shape differs from the frozen contract"
        )
    booster.set_param({"nthread": 1})
    return _frozen.OfficialNoSpreadPredictor(booster)


def official_model_binding() -> ReferenceModelBindingV1:
    return ReferenceModelBindingV1(
        model_id=_frozen.MODEL_ID,
        model_version=_frozen.ARCHIVE_TAG_COMMIT,
        model_sha256=_frozen.ASSET_SHA256,
        manifest_sha256=_frozen.ASSET_MANIFEST_SHA256,
        feature_spec_commit=_frozen.FEATURE_SPEC_COMMIT,
    )


def _verify_x13_fact_batch(
    *,
    project_root: str | Path,
    batch_index_path: str | Path,
) -> _VerifiedX13Batch:
    project = Path(project_root).resolve()
    path = Path(batch_index_path).resolve()
    if project != path and project not in path.parents:
        raise StageAReferenceError("X-13 batch index escapes project root")
    if not path.is_file() or path.is_symlink():
        raise StageAReferenceError("X-13 batch index is missing or unsafe")
    index_sha = _sha256_file(path)
    if index_sha != EXPECTED_X13_BATCH_INDEX_SHA256:
        raise StageAReferenceError("X-13 batch index bytes/hash mismatch")
    index = _read_json(path, label="X-13 batch index")
    if (
        index.get("schema") != "nfl_x13_exact153_fact_batch_index_v1"
        or index.get("experiment_id") != "X-13"
        or index.get("cohort") != "development"
        or index.get("game_count") != EXPECTED_X13_GAME_COUNT
        or index.get("market_data_read") is not False
        or index.get("holdout_reaction_accessed") is not False
        or index.get("publication_gate") != "PASS"
    ):
        raise StageAReferenceError("X-13 batch contract mismatch")
    authority = index.get("authority")
    sources = index.get("sources")
    games_payload = index.get("games")
    if (
        type(authority) is not dict
        or authority.get("experiment_id") != "X-13"
        or authority.get("development_game_count") != EXPECTED_X13_GAME_COUNT
        or type(sources) is not dict
        or type(sources.get("pbp")) is not dict
        or sources["pbp"].get("dataset_id") != "DS-NFLVERSE"
        or type(sources.get("participation")) is not dict
        or sources["participation"].get("dataset_id")
        != "DS-NFLVERSE-PARTICIPATION"
        or type(games_payload) is not list
        or len(games_payload) != EXPECTED_X13_GAME_COUNT
    ):
        raise StageAReferenceError("X-13 batch governance binding mismatch")
    batch_sha = index.get("batch_sha256")
    material = dict(index)
    material.pop("batch_sha256", None)
    if (
        type(batch_sha) is not str
        or batch_sha != _canonical_sha256(material)
        or type(index.get("semantic_batch_sha256")) is not str
    ):
        raise StageAReferenceError("X-13 batch content identity mismatch")
    if len(path.parents) < 5:
        raise StageAReferenceError("X-13 batch path cannot identify its root")
    x13_root = path.parents[4]
    verified_games: list[_VerifiedX13Game] = []
    seen: set[str] = set()
    for descriptor in games_payload:
        if (
            type(descriptor) is not dict
            or type(descriptor.get("game_id")) is not str
            or descriptor["game_id"] in seen
            or type(descriptor.get("manifest_path")) is not str
            or type(descriptor.get("manifest_sha256")) is not str
            or type(descriptor.get("bundle_sha256")) is not str
        ):
            raise StageAReferenceError("X-13 game descriptor is malformed")
        game_id = str(descriptor["game_id"])
        seen.add(game_id)
        manifest_path = _resolve_under(
            x13_root,
            str(descriptor["manifest_path"]),
            label=f"{game_id} X-13 manifest",
        )
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or _sha256_file(manifest_path)
            != descriptor["manifest_sha256"]
        ):
            raise StageAReferenceError(
                f"X-13 game manifest hash mismatch: {game_id}"
            )
        manifest = _read_json(
            manifest_path,
            label=f"{game_id} X-13 manifest",
        )
        manifest_material = dict(manifest)
        manifest_bundle_sha = manifest_material.pop("bundle_sha256", None)
        if (
            manifest.get("schema")
            != "nfl_x13_exact153_single_game_fact_manifest_v1"
            or manifest.get("experiment_id") != "X-13"
            or manifest.get("game_id") != game_id
            or manifest.get("cohort") != "development"
            or manifest.get("market_data_read") is not False
            or manifest.get("holdout_reaction_accessed") is not False
            or manifest.get("publication_gate") != "PASS"
            or manifest.get("authority") != authority
            or manifest.get("sources") != sources
            or manifest_bundle_sha != descriptor["bundle_sha256"]
            or manifest_bundle_sha != _canonical_sha256(manifest_material)
        ):
            raise StageAReferenceError(
                f"X-13 game manifest contract mismatch: {game_id}"
            )
        tables = manifest.get("tables")
        if type(tables) is not list:
            raise StageAReferenceError(
                f"X-13 game table inventory is absent: {game_id}"
            )
        canonical_matches = [
            row
            for row in tables
            if type(row) is dict
            and row.get("name") == "canonical_factor_events"
        ]
        if len(canonical_matches) != 1:
            raise StageAReferenceError(
                f"X-13 canonical events table is ambiguous: {game_id}"
            )
        canonical = canonical_matches[0]
        object_path = _resolve_under(
            x13_root,
            str(canonical.get("object_path", "")),
            label=f"{game_id} canonical events",
        )
        if (
            not object_path.is_file()
            or object_path.is_symlink()
            or object_path.stat().st_size != canonical.get("byte_length")
            or _sha256_file(object_path) != canonical.get("object_sha256")
        ):
            raise StageAReferenceError(
                f"X-13 canonical events object hash mismatch: {game_id}"
            )
        try:
            parquet = pq.ParquetFile(object_path)
            schema_columns = tuple(parquet.schema_arrow.names)
        except (OSError, pa.ArrowException) as exc:
            raise StageAReferenceError(
                f"X-13 canonical events are unreadable: {game_id}"
            ) from exc
        if (
            parquet.metadata.num_rows != canonical.get("row_count")
            or schema_columns != tuple(canonical.get("schema_columns", ()))
            or canonical.get("semantic_rows_sha256")
            != descriptor.get("table_semantic_sha256", {}).get(
                "canonical_factor_events"
            )
        ):
            raise StageAReferenceError(
                f"X-13 canonical events schema/count mismatch: {game_id}"
            )
        verified_games.append(
            _VerifiedX13Game(
                game_id=game_id,
                manifest_path=manifest_path,
                manifest_sha256=str(descriptor["manifest_sha256"]),
                bundle_sha256=str(descriptor["bundle_sha256"]),
                canonical_events_path=object_path,
                canonical_events_object_sha256=str(
                    canonical["object_sha256"]
                ),
                canonical_events_semantic_sha256=str(
                    canonical["semantic_rows_sha256"]
                ),
                canonical_events_byte_length=int(canonical["byte_length"]),
                canonical_events_row_count=int(canonical["row_count"]),
                canonical_events_schema_columns=schema_columns,
            )
        )
    return _VerifiedX13Batch(
        root=x13_root,
        index_path=path,
        index_sha256=index_sha,
        batch_sha256=str(batch_sha),
        semantic_batch_sha256=str(index["semantic_batch_sha256"]),
        sources=dict(sources),
        games=tuple(verified_games),
    )


def _builder_code_sha256() -> str:
    source_paths: set[Path] = {Path(__file__).resolve()}
    for item in (build_game_reference_tables, _frozen.NoSpreadModelInput):
        source = inspect.getsourcefile(item)
        if source is None:
            raise StageAReferenceError(
                f"cannot bind builder source for {item.__name__}"
            )
        source_paths.add(Path(source).resolve())
    return _canonical_sha256(
        [
            {
                "path_name": path.name,
                "source_file_sha256": _sha256_file(path),
            }
            for path in sorted(source_paths, key=lambda value: value.as_posix())
        ]
    )


def _require_source_binding(
    descriptor: object,
    verified: VerifiedSource,
) -> None:
    if type(descriptor) is not dict:
        raise StageAReferenceError(
            f"X-13 source binding is absent: {verified.dataset_id}"
        )
    expected = {
        "dataset_id": verified.dataset_id,
        "logical_manifest_sha256": verified.manifest_logical_sha256,
        "manifest_file_sha256": verified.manifest_file_sha256,
        "object_sha256": verified.object_sha256,
        "byte_length": verified.byte_length,
        "license_status": verified.license_status,
    }
    if descriptor != expected:
        raise StageAReferenceError(
            f"X-13 source binding differs from frozen bytes: "
            f"{verified.dataset_id}"
        )


def _load_verified_game_events(game: _VerifiedX13Game) -> pd.DataFrame:
    try:
        payload = game.canonical_events_path.read_bytes()
    except OSError as exc:
        raise StageAReferenceError(
            f"cannot read X-13 canonical events: {game.game_id}"
        ) from exc
    if (
        len(payload) != game.canonical_events_byte_length
        or _sha256_bytes(payload) != game.canonical_events_object_sha256
    ):
        raise StageAReferenceError(
            f"X-13 canonical events changed after verification: {game.game_id}"
        )
    try:
        table = pq.read_table(pa.BufferReader(payload), use_threads=False)
    except (OSError, pa.ArrowException) as exc:
        raise StageAReferenceError(
            f"cannot decode verified X-13 canonical events: {game.game_id}"
        ) from exc
    if (
        table.num_rows != game.canonical_events_row_count
        or tuple(table.schema.names) != game.canonical_events_schema_columns
    ):
        raise StageAReferenceError(
            f"decoded X-13 canonical events changed shape: {game.game_id}"
        )
    frame = table.to_pandas()
    if (
        "game_id" not in frame
        or tuple(frame["game_id"].astype(str).unique()) != (game.game_id,)
    ):
        raise StageAReferenceError(
            f"X-13 canonical events have wrong game identity: {game.game_id}"
        )
    return frame


def _canonical_final_score(
    canonical_events: pd.DataFrame,
) -> tuple[int, int]:
    required = {
        "game_id",
        "order_sequence",
        "raw_play_id",
        "post_home_score",
        "post_away_score",
    }
    missing = sorted(required.difference(canonical_events.columns))
    if missing or canonical_events.empty:
        raise StageAReferenceError(
            "canonical events cannot establish final score"
        )
    order = pd.to_numeric(
        canonical_events["order_sequence"],
        errors="coerce",
    )
    raw_play = canonical_events["raw_play_id"].astype("string")
    if (
        order.isna().any()
        or (order % 1 != 0).any()
        or raw_play.isna().any()
        or raw_play.eq("").any()
        or pd.DataFrame(
            {"order": order.astype("int64"), "raw_play": raw_play}
        ).duplicated().any()
    ):
        raise StageAReferenceError(
            "canonical chronology cannot establish final score"
        )
    ordered = canonical_events.assign(
        _order=order.astype("int64"),
        _raw_sort=raw_play,
    ).sort_values(["_order", "_raw_sort"], kind="mergesort")
    last = ordered.iloc[-1]
    scores: list[int] = []
    for field in ("post_home_score", "post_away_score"):
        value = pd.to_numeric(pd.Series([last[field]]), errors="coerce").iloc[0]
        if (
            pd.isna(value)
            or not math.isfinite(float(value))
            or not float(value).is_integer()
            or float(value) < 0
        ):
            raise StageAReferenceError(
                "canonical final score must be nonnegative integral values"
            )
        scores.append(int(value))
    return scores[0], scores[1]


def _attach_final_outcome(
    state_predictions: pd.DataFrame,
    canonical_events: pd.DataFrame,
) -> pd.DataFrame:
    if (
        not isinstance(state_predictions, pd.DataFrame)
        or state_predictions.empty
        or "game_id" not in state_predictions
        or "reference_status" not in state_predictions
    ):
        raise StageAReferenceError(
            "state predictions cannot be bound to final outcome"
        )
    event_games = tuple(canonical_events["game_id"].astype(str).unique())
    state_games = tuple(state_predictions["game_id"].astype(str).unique())
    if len(event_games) != 1 or state_games != event_games:
        raise StageAReferenceError(
            "state predictions and canonical outcome identify different games"
        )
    home_score, away_score = _canonical_final_score(canonical_events)
    final_home_win: int | None
    if home_score == away_score:
        final_home_win = None
    else:
        final_home_win = int(home_score > away_score)
    result = state_predictions.copy()
    result["final_home_score"] = home_score
    result["final_away_score"] = away_score
    result["final_home_win"] = pd.Series(
        [final_home_win] * len(result),
        dtype="Int64",
        index=result.index,
    )
    eligible = result["reference_status"].eq("SUPPORTED") & result[
        "final_home_win"
    ].notna()
    result["evaluation_eligible"] = eligible.astype(bool)
    eligible_count = int(eligible.sum())
    result["equal_game_weight"] = 0.0
    if eligible_count > 0:
        result.loc[eligible, "equal_game_weight"] = 1.0 / eligible_count
    return result


def _publish_game_reference_bundle(
    *,
    output_root: Path,
    tables: GameReferenceTables,
    x13_input_binding: dict[str, object],
    model: ReferenceModelBindingV1,
    builder_code_sha256: str,
) -> PublishedStageAGame:
    table_frames = (
        ("state_predictions", tables.state_predictions),
        ("feature_provenance", tables.feature_provenance),
        ("reference_observations", tables.reference_observations),
    )
    descriptors: list[dict[str, object]] = []
    paths: list[Path] = []
    for name, frame in table_frames:
        descriptor, path = _publish_table(
            output_root=output_root,
            relative_parent=Path("single-game") / tables.game_id,
            name=name,
            frame=frame,
        )
        descriptors.append(descriptor)
        paths.append(path)
    material: dict[str, object] = {
        "schema": "nfl_x15_stage_a_single_game_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "cohort": "development",
        "game_id": tables.game_id,
        "claim_boundary": CLAIM_BOUNDARY,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "x13_input": dict(x13_input_binding),
        "model": {
            "model_id": model.model_id,
            "model_version": model.model_version,
            "model_sha256": model.model_sha256,
            "manifest_sha256": model.manifest_sha256,
            "feature_spec_commit": model.feature_spec_commit,
        },
        "opening_kickoff_evidence": {
            "second_half_receiver": tables.second_half_receiver,
            "event_id": tables.opening_kickoff_event_id,
            "known_at": tables.opening_kickoff_known_at,
        },
        "builder_version": BUILDER_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "tables": descriptors,
        "publication_gate": "PASS",
    }
    bundle_sha = _canonical_sha256(material)
    manifest = {**material, "bundle_sha256": bundle_sha}
    encoded = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(encoded)
    digest = manifest_sha.removeprefix("sha256:")
    relative = (
        Path("single-game")
        / tables.game_id
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    path = _resolve_under(
        output_root,
        relative,
        label=f"{tables.game_id} Stage A manifest",
    )
    _atomic_publish(path, encoded)
    if _sha256_file(path) != manifest_sha:
        raise StageAReferenceError(
            f"Stage A manifest verification failed: {tables.game_id}"
        )
    return PublishedStageAGame(
        game_id=tables.game_id,
        manifest_path=path,
        manifest_sha256=manifest_sha,
        bundle_sha256=bundle_sha,
        table_paths=tuple(paths),
    )


def _weighted_point_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    y = frame["final_home_win"].to_numpy(dtype=float)
    probability = frame["p_home"].to_numpy(dtype=float)
    weights = frame["_game_weight"].to_numpy(dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise StageAReferenceError("calibration has no positive game weight")
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    brier = float(np.sum(weights * (probability - y) ** 2) / weight_sum)
    log_loss = float(
        -np.sum(
            weights
            * (y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
        )
        / weight_sum
    )
    slope, intercept = _weighted_calibration(y, clipped, weights)
    return {
        "brier": brier,
        "log_loss": log_loss,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _weighted_calibration(
    y: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(np.unique(y)) != 2:
        return None, None
    logit = np.log(probability / (1.0 - probability))
    if float(np.ptp(logit)) <= 1e-12:
        return None, None
    design = np.column_stack([np.ones(len(logit)), logit])
    beta = np.zeros(2, dtype=float)
    for _ in range(100):
        linear = np.clip(design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        variance = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        score = design.T @ (weights * (y - fitted))
        information = design.T @ (
            design * (weights * variance)[:, np.newaxis]
        )
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return None, None
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            return float(beta[1]), float(beta[0])
    return None, None


def _with_equal_game_weight(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts = result.groupby("game_id", sort=False)["state_event_id"].transform(
        "size"
    )
    result["_game_weight"] = 1.0 / counts.astype(float)
    return result


def _reliability_table(frame: pd.DataFrame) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, 11)
    bins = pd.cut(
        frame["p_home"],
        bins=edges,
        include_lowest=True,
        right=True,
        labels=False,
    )
    rows: list[dict[str, object]] = []
    for index in range(10):
        group = frame.loc[bins.eq(index)]
        if group.empty:
            rows.append(
                {
                    "bin": index,
                    "probability_low": float(edges[index]),
                    "probability_high": float(edges[index + 1]),
                    "state_count": 0,
                    "game_count": 0,
                    "total_game_weight": 0.0,
                    "mean_probability": None,
                    "observed_home_win_rate": None,
                }
            )
            continue
        weight = group["_game_weight"].to_numpy(dtype=float)
        total = float(weight.sum())
        rows.append(
            {
                "bin": index,
                "probability_low": float(edges[index]),
                "probability_high": float(edges[index + 1]),
                "state_count": len(group),
                "game_count": int(group["game_id"].nunique()),
                "total_game_weight": total,
                "mean_probability": float(
                    np.average(group["p_home"].to_numpy(dtype=float), weights=weight)
                ),
                "observed_home_win_rate": float(
                    np.average(
                        group["final_home_win"].to_numpy(dtype=float),
                        weights=weight,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_breakdowns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_quarter_bucket"] = work["quarter"].map(
        lambda value: f"Q{int(value)}"
    )
    seconds = work["game_seconds_remaining"].astype(float)
    work["_time_bucket"] = pd.cut(
        seconds,
        bins=[-np.inf, 120.0, 300.0, 600.0, 900.0, np.inf],
        labels=["LE_2M", "2_5M", "5_10M", "10_15M", "GT_15M"],
        right=True,
    ).astype(str)
    work["_probability_bucket"] = pd.cut(
        work["p_home"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    ).astype(str)
    dimensions = (
        ("quarter", "_quarter_bucket"),
        ("game_time", "_time_bucket"),
        ("probability", "_probability_bucket"),
    )
    rows: list[dict[str, object]] = []
    for dimension, column in dimensions:
        for bucket, raw_group in work.groupby(column, sort=True):
            group = _with_equal_game_weight(raw_group)
            metrics = _weighted_point_metrics(group)
            rows.append(
                {
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "state_count": len(group),
                    "game_count": int(group["game_id"].nunique()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_table(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if type(bootstrap_samples) is not int or bootstrap_samples < 20:
        raise StageAReferenceError("bootstrap_samples must be an integer >= 20")
    if type(bootstrap_seed) is not int:
        raise StageAReferenceError("bootstrap_seed must be an integer")
    clipped = np.clip(frame["p_home"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    work = frame.assign(
        _brier_loss=(
            frame["p_home"].to_numpy(dtype=float)
            - frame["final_home_win"].to_numpy(dtype=float)
        )
        ** 2,
        _log_loss=(
            -frame["final_home_win"].to_numpy(dtype=float) * np.log(clipped)
            - (1.0 - frame["final_home_win"].to_numpy(dtype=float))
            * np.log(1.0 - clipped)
        ),
    )
    game_losses = (
        work.groupby("game_id", sort=True)[["_brier_loss", "_log_loss"]]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(game_losses) < 2:
        raise StageAReferenceError(
            "game-cluster bootstrap requires at least two games"
        )
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(game_losses),
        size=(bootstrap_samples, len(game_losses)),
    )
    sampled = game_losses[draws].mean(axis=1)
    rows: list[dict[str, object]] = []
    for offset, metric in enumerate(("brier", "log_loss")):
        distribution = sampled[:, offset]
        rows.append(
            {
                "metric": metric,
                "estimate": float(game_losses[:, offset].mean()),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "requested_samples": bootstrap_samples,
                "valid_samples": len(distribution),
                "seed": bootstrap_seed,
                "cluster_unit": "game_id",
                "weighting": "equal_total_weight_per_game",
            }
        )
    return pd.DataFrame(rows)


def calculate_stage_a_calibration(
    state_predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> StageACalibrationTables:
    """Evaluate unique supported regulation states with equal game weight."""

    required = {
        "game_id",
        "state_event_id",
        "reference_status",
        "p_home",
        "final_home_win",
        "quarter",
        "game_seconds_remaining",
    }
    if not isinstance(state_predictions, pd.DataFrame):
        raise StageAReferenceError("state_predictions must be a DataFrame")
    missing = sorted(required.difference(state_predictions.columns))
    if missing:
        raise StageAReferenceError(
            "state_predictions missing columns: " + ", ".join(missing)
        )
    if state_predictions.duplicated(["game_id", "state_event_id"]).any():
        raise StageAReferenceError(
            "state_predictions contain duplicate state grain"
        )
    supported = state_predictions.loc[
        state_predictions["reference_status"].eq("SUPPORTED")
    ].copy()
    tie_states = int(supported["final_home_win"].isna().sum())
    evaluated = supported.loc[supported["final_home_win"].notna()].copy()
    if evaluated.empty:
        raise StageAReferenceError("no supported non-tie states to evaluate")
    probability = pd.to_numeric(evaluated["p_home"], errors="coerce")
    target = pd.to_numeric(evaluated["final_home_win"], errors="coerce")
    if (
        probability.isna().any()
        or (~probability.between(0.0, 1.0)).any()
        or target.isna().any()
        or (~target.isin([0, 1])).any()
    ):
        raise StageAReferenceError(
            "supported probabilities or final outcomes are invalid"
        )
    evaluated["p_home"] = probability.astype(float)
    evaluated["final_home_win"] = target.astype(int)
    evaluated = _with_equal_game_weight(evaluated)
    metrics = _weighted_point_metrics(evaluated)
    summary = pd.DataFrame(
        [
            {
                "schema_version": "StageACalibrationSummaryV1",
                "experiment_id": EXPERIMENT_ID,
                "claim_boundary": CLAIM_BOUNDARY,
                "evaluated_games": int(evaluated["game_id"].nunique()),
                "evaluated_states": len(evaluated),
                "excluded_tie_states": tie_states,
                "total_game_weight": float(evaluated["_game_weight"].sum()),
                "weighting": "equal_total_weight_per_game",
                **metrics,
            }
        ]
    )
    return StageACalibrationTables(
        summary=summary,
        reliability=_reliability_table(evaluated),
        breakdowns=_metric_breakdowns(evaluated),
        bootstrap=_bootstrap_table(
            evaluated,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
    )


def _publish_stage_a_batch_index(
    *,
    output_root: Path,
    x13_batch: _VerifiedX13Batch,
    games: tuple[PublishedStageAGame, ...],
    calibration: StageACalibrationTables,
    model: ReferenceModelBindingV1,
    builder_code_sha256: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    state_status_counts: dict[str, int],
    observation_status_counts: dict[str, int],
) -> PublishedStageABatch:
    governed_tables: tuple[tuple[str, str, pd.DataFrame], ...] = (
        (
            "calibration_summary",
            "StageACalibrationSummaryV1",
            calibration.summary,
        ),
        (
            "calibration_reliability",
            "StageACalibrationReliabilityV1",
            calibration.reliability,
        ),
        (
            "calibration_breakdowns",
            "StageACalibrationBreakdownV1",
            calibration.breakdowns,
        ),
        (
            "calibration_bootstrap",
            "StageACalibrationBootstrapV1",
            calibration.bootstrap,
        ),
    )
    table_descriptors: list[dict[str, object]] = []
    table_paths: list[Path] = []
    for name, schema_version, source in governed_tables:
        frame = source.copy()
        if "schema_version" not in frame:
            frame.insert(0, "schema_version", schema_version)
        if "experiment_id" not in frame:
            frame.insert(1, "experiment_id", EXPERIMENT_ID)
        if "claim_boundary" not in frame:
            frame.insert(2, "claim_boundary", CLAIM_BOUNDARY)
        descriptor, table_path = _publish_table(
            output_root=output_root,
            relative_parent=Path("batches"),
            name=name,
            frame=frame,
        )
        table_descriptors.append(descriptor)
        table_paths.append(table_path)

    game_descriptors = [
        {
            "game_id": game.game_id,
            "manifest_path": game.manifest_path.relative_to(
                output_root
            ).as_posix(),
            "manifest_sha256": game.manifest_sha256,
            "bundle_sha256": game.bundle_sha256,
        }
        for game in games
    ]
    material: dict[str, object] = {
        "schema": "nfl_x15_stage_a_batch_index_v1",
        "experiment_id": EXPERIMENT_ID,
        "cohort": "development",
        "game_count": len(games),
        "claim_boundary": CLAIM_BOUNDARY,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "x13_input": {
            "experiment_id": "X-13",
            "batch_index_sha256": x13_batch.index_sha256,
            "batch_sha256": x13_batch.batch_sha256,
            "semantic_batch_sha256": x13_batch.semantic_batch_sha256,
        },
        "sources": dict(x13_batch.sources),
        "model": {
            "model_id": model.model_id,
            "model_version": model.model_version,
            "model_sha256": model.model_sha256,
            "manifest_sha256": model.manifest_sha256,
            "feature_spec_commit": model.feature_spec_commit,
        },
        "builder_version": BUILDER_VERSION,
        "builder_code_sha256": builder_code_sha256,
        "bootstrap": {
            "cluster_unit": "game_id",
            "requested_samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "weighting": "equal_total_weight_per_game",
        },
        "counts": {
            "state_status": dict(sorted(state_status_counts.items())),
            "observation_status": dict(
                sorted(observation_status_counts.items())
            ),
        },
        "calibration_tables": table_descriptors,
        "games": game_descriptors,
        "publication_gate": "PASS",
    }
    batch_sha = _canonical_sha256(material)
    index = {**material, "batch_sha256": batch_sha}
    encoded = _canonical_bytes(index)
    index_sha = _sha256_bytes(encoded)
    digest = index_sha.removeprefix("sha256:")
    relative = (
        Path("batches")
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.batch-index.json"
    )
    index_path = _resolve_under(
        output_root,
        relative,
        label="Stage A batch index",
    )
    _atomic_publish(index_path, encoded)
    if _sha256_file(index_path) != index_sha:
        raise StageAReferenceError("Stage A batch index verification failed")
    return PublishedStageABatch(
        game_count=len(games),
        index_path=index_path,
        index_sha256=index_sha,
        batch_sha256=batch_sha,
        games=games,
        table_paths=tuple(table_paths),
    )


def publish_stage_a_reference_batch(
    *,
    project_root: str | Path,
    output_root: str | Path | None = None,
    x13_batch_index_path: str | Path | None = None,
    model_manifest_path: str | Path | None = None,
    workers: int = 1,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> PublishedStageABatch:
    """Publish the verified 153-game Stage A batch."""

    if type(workers) is not int or workers != 1:
        raise StageAReferenceError("workers must equal 1")
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise StageAReferenceError("project_root must be an existing directory")
    output = (
        _resolve_under(project, DEFAULT_OUTPUT_ROOT, label="Stage A output")
        if output_root is None
        else _resolve_under(project, output_root, label="Stage A output")
    )
    x13_index = (
        _resolve_under(
            project,
            DEFAULT_X13_BATCH_INDEX,
            label="X-13 batch index",
        )
        if x13_batch_index_path is None
        else _resolve_under(
            project,
            x13_batch_index_path,
            label="X-13 batch index",
        )
    )
    x13_batch = _verify_x13_fact_batch(
        project_root=project,
        batch_index_path=x13_index,
    )
    if len(x13_batch.games) != EXPECTED_X13_GAME_COUNT:
        raise StageAReferenceError("X-13 verified game count is not 153")

    try:
        data_root = discover_main_checkout(project)
        pbp_source, participation_source = verify_frozen_2025_sources(
            data_root
        )
    except HistoricalSportsCleanError as exc:
        raise StageAReferenceError(
            "frozen 2025 sports sources failed verification"
        ) from exc
    _require_source_binding(x13_batch.sources.get("pbp"), pbp_source)
    _require_source_binding(
        x13_batch.sources.get("participation"),
        participation_source,
    )

    store_root = (data_root / "var/raw").resolve()
    manifest = (
        (data_root / DEFAULT_MODEL_MANIFEST).resolve()
        if model_manifest_path is None
        else _resolve_under(
            data_root,
            model_manifest_path,
            label="official model manifest",
        )
    )
    if manifest != store_root and store_root not in manifest.parents:
        raise StageAReferenceError(
            "official model manifest must belong to the verified static store"
        )
    predictor = load_x15_official_no_spread_predictor(
        program_root=project,
        store_root=store_root,
        manifest_path=manifest,
    )
    model = official_model_binding()
    builder_code_sha256 = _builder_code_sha256()

    published_games: list[PublishedStageAGame] = []
    calibration_frames: list[pd.DataFrame] = []
    state_status_counts: dict[str, int] = {}
    observation_status_counts: dict[str, int] = {}
    for game in x13_batch.games:
        events = _load_verified_game_events(game)
        try:
            raw_tables = build_game_reference_tables(
                events,
                predictor=predictor,
                model=model,
                expected_pbp_source_sha256=pbp_source.object_sha256,
            )
            enriched_states = _attach_final_outcome(
                raw_tables.state_predictions,
                events,
            )
        except ReferenceValueBuildError as exc:
            raise StageAReferenceError(
                f"Stage A inference failed for {game.game_id}"
            ) from exc
        tables = GameReferenceTables(
            game_id=raw_tables.game_id,
            second_half_receiver=raw_tables.second_half_receiver,
            opening_kickoff_event_id=raw_tables.opening_kickoff_event_id,
            opening_kickoff_known_at=raw_tables.opening_kickoff_known_at,
            state_predictions=enriched_states,
            feature_provenance=raw_tables.feature_provenance,
            reference_observations=raw_tables.reference_observations,
        )
        published_games.append(
            _publish_game_reference_bundle(
                output_root=output,
                tables=tables,
                x13_input_binding=game.input_binding(
                    batch_index_sha256=x13_batch.index_sha256,
                    batch_sha256=x13_batch.batch_sha256,
                    semantic_batch_sha256=x13_batch.semantic_batch_sha256,
                ),
                model=model,
                builder_code_sha256=builder_code_sha256,
            )
        )
        for status, count in enriched_states[
            "reference_status"
        ].value_counts().items():
            state_status_counts[str(status)] = (
                state_status_counts.get(str(status), 0) + int(count)
            )
        for status, count in tables.reference_observations[
            "reference_status"
        ].value_counts().items():
            observation_status_counts[str(status)] = (
                observation_status_counts.get(str(status), 0) + int(count)
            )
        calibration_frames.append(
            enriched_states[
                [
                    "game_id",
                    "state_event_id",
                    "reference_status",
                    "p_home",
                    "final_home_win",
                    "quarter",
                    "game_seconds_remaining",
                ]
            ].copy()
        )
        del events, raw_tables, tables, enriched_states
        gc.collect()

    calibration_input = pd.concat(calibration_frames, ignore_index=True)
    calibration = calculate_stage_a_calibration(
        calibration_input,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    result = _publish_stage_a_batch_index(
        output_root=output,
        x13_batch=x13_batch,
        games=tuple(published_games),
        calibration=calibration,
        model=model,
        builder_code_sha256=builder_code_sha256,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        state_status_counts=state_status_counts,
        observation_status_counts=observation_status_counts,
    )
    del calibration_frames, calibration_input, calibration
    gc.collect()
    return result


__all__ = [
    "BOOTSTRAP_SEED",
    "BUILDER_VERSION",
    "DEFAULT_MODEL_MANIFEST",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_X13_BATCH_INDEX",
    "EXPECTED_X13_BATCH_INDEX_SHA256",
    "EXPECTED_X13_GAME_COUNT",
    "PublishedStageABatch",
    "PublishedStageAGame",
    "StageACalibrationTables",
    "StageAReferenceError",
    "calculate_stage_a_calibration",
    "load_x15_official_no_spread_predictor",
    "official_model_binding",
    "publish_stage_a_reference_batch",
]
