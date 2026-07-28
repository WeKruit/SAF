"""Immutable publication for X-13 historical empirical distributions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_x13_historical_study import (
    CLAIM_BOUNDARY,
    HistoricalFactorStudy,
)
from prediction_market.sports.nfl_x13_dense_factor_summary import (
    build_dense_factor_summary,
)


SCHEMA = "nfl_x13_historical_study_publication_v1"
BUILDER_VERSION = "nfl-x13-historical-study-publication-v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TABLE_NAMES = (
    "summary",
    "histogram",
    "ecdf",
    "bootstrap_histogram",
    "cases",
    "game_effects",
    "state_breakdown",
    "path_probabilities",
    "venue_pairs",
    "venue_diagnostics",
)


class HistoricalStudyPublicationError(RuntimeError):
    """A historical study object or manifest failed closed."""


@dataclass(frozen=True, slots=True)
class HistoricalStudyPublication:
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _inside(root: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HistoricalStudyPublicationError(f"{field} path is invalid")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HistoricalStudyPublicationError(
            f"{field} path escapes source root"
        ) from error
    return candidate


def _verified_json(
    path: Path,
    *,
    expected_sha256: object | None,
    field: str,
) -> dict[str, object]:
    if not path.is_file():
        raise HistoricalStudyPublicationError(f"{field} is missing")
    payload = path.read_bytes()
    actual = _sha(payload)
    if expected_sha256 is not None and actual != expected_sha256:
        raise HistoricalStudyPublicationError(f"{field} SHA-256 mismatch")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise HistoricalStudyPublicationError(f"{field} must be a JSON object")
    return value


def _verified_parquet(
    root: Path,
    reference: object,
    *,
    field: str,
) -> pd.DataFrame:
    if not isinstance(reference, dict):
        raise HistoricalStudyPublicationError(f"{field} reference is invalid")
    required = {"object_path", "object_sha256", "byte_length", "row_count"}
    if not required.issubset(reference):
        raise HistoricalStudyPublicationError(f"{field} reference is incomplete")
    path = _inside(root, reference["object_path"], field=field)
    if (
        not path.is_file()
        or path.stat().st_size != reference["byte_length"]
        or _file_sha(path) != reference["object_sha256"]
    ):
        raise HistoricalStudyPublicationError(f"{field} SHA-256 mismatch")
    frame = pq.read_table(path).to_pandas()
    if len(frame) != reference["row_count"]:
        raise HistoricalStudyPublicationError(f"{field} row count mismatch")
    return frame


def _game_manifest_map(
    batch: dict[str, object],
    *,
    root: Path,
    expected_game_count: int,
    field: str,
) -> dict[str, tuple[dict[str, object], str]]:
    if batch.get("game_count") != expected_game_count:
        raise HistoricalStudyPublicationError(f"{field} game count mismatch")
    if batch.get("final_holdout_access") not in (None, "CLOSED"):
        raise HistoricalStudyPublicationError(f"{field} opens the final holdout")
    if batch.get("holdout_reaction_accessed") not in (None, False):
        raise HistoricalStudyPublicationError(f"{field} accessed holdout reaction")
    games = batch.get("games")
    if not isinstance(games, list) or len(games) != expected_game_count:
        raise HistoricalStudyPublicationError(f"{field} game references are invalid")
    result: dict[str, tuple[dict[str, object], str]] = {}
    for reference in games:
        if not isinstance(reference, dict):
            raise HistoricalStudyPublicationError(f"{field} game reference is invalid")
        game_id = str(reference.get("game_id", "")).strip()
        if not game_id or game_id in result:
            raise HistoricalStudyPublicationError(f"{field} game identity is invalid")
        path = _inside(
            root,
            reference.get("manifest_path"),
            field=f"{field}.{game_id}",
        )
        expected = reference.get("manifest_sha256")
        manifest = _verified_json(
            path,
            expected_sha256=expected,
            field=f"{field}.{game_id}",
        )
        result[game_id] = (manifest, str(expected))
    return result


def load_development_eligible_paths(
    *,
    market_root: str | Path,
    market_batch_index_path: str | Path,
    dense_root: str | Path,
    dense_batch_manifest_path: str | Path,
    expected_game_count: int = 153,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Verify sealed development batches and join factor membership to dense paths."""

    if type(expected_game_count) is not int or expected_game_count < 1:
        raise HistoricalStudyPublicationError(
            "expected_game_count must be positive"
        )
    market_root_path = Path(market_root).resolve()
    dense_root_path = Path(dense_root).resolve()
    market_batch_path = Path(market_batch_index_path).resolve()
    dense_batch_path = Path(dense_batch_manifest_path).resolve()
    market_batch = _verified_json(
        market_batch_path,
        expected_sha256=None,
        field="market batch",
    )
    dense_batch = _verified_json(
        dense_batch_path,
        expected_sha256=None,
        field="dense batch",
    )
    market_games = _game_manifest_map(
        market_batch,
        root=market_root_path,
        expected_game_count=expected_game_count,
        field="market batch",
    )
    dense_games = _game_manifest_map(
        dense_batch,
        root=dense_root_path,
        expected_game_count=expected_game_count,
        field="dense batch",
    )
    if set(market_games) != set(dense_games):
        raise HistoricalStudyPublicationError(
            "market and dense development games differ"
        )
    source_hashes = {
        _file_sha(market_batch_path),
        _file_sha(dense_batch_path),
    }
    eligible_parts: list[pd.DataFrame] = []
    membership_keys = [
        "factor_id",
        "factor_version",
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
    ]
    for game_id in sorted(market_games):
        market_manifest, market_manifest_sha = market_games[game_id]
        dense_manifest, dense_manifest_sha = dense_games[game_id]
        stages = market_manifest.get("stages")
        objects = dense_manifest.get("objects")
        if not isinstance(stages, dict) or not isinstance(objects, dict):
            raise HistoricalStudyPublicationError(
                f"{game_id} source manifest is malformed"
            )
        factor = _verified_parquet(
            market_root_path,
            stages.get("factor_observations"),
            field=f"{game_id}.factor_observations",
        )
        paths = _verified_parquet(
            dense_root_path,
            objects.get("paths"),
            field=f"{game_id}.dense_paths",
        )
        result = build_dense_factor_summary(factor, paths)
        if result.eligible.empty:
            continue
        if "pre_quarter" not in factor.columns:
            raise HistoricalStudyPublicationError(
                f"{game_id} factor observations lack pre_quarter"
            )
        quarter = factor[[*membership_keys, "pre_quarter"]].drop_duplicates()
        if quarter.duplicated(membership_keys).any():
            raise HistoricalStudyPublicationError(
                f"{game_id} quarter differs across registered horizons"
            )
        enriched = result.eligible.merge(
            quarter.rename(columns={"pre_quarter": "quarter"}),
            on=membership_keys,
            how="left",
            validate="many_to_one",
        )
        if enriched["quarter"].isna().any():
            raise HistoricalStudyPublicationError(
                f"{game_id} eligible rows lack quarter"
            )
        eligible_parts.append(enriched)
        source_hashes.update(
            (
                market_manifest_sha,
                dense_manifest_sha,
                str(stages["factor_observations"]["object_sha256"]),
                str(objects["paths"]["object_sha256"]),
            )
        )
    if not eligible_parts:
        raise HistoricalStudyPublicationError(
            "development sources produced no eligible paths"
        )
    eligible = pd.concat(eligible_parts, ignore_index=True)
    if set(eligible["game_id"]) - set(market_games):
        raise HistoricalStudyPublicationError("eligible game identity escaped batch")
    return eligible, tuple(sorted(source_hashes))


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    if not isinstance(frame, pd.DataFrame):
        raise HistoricalStudyPublicationError("study table must be a DataFrame")
    buffer = BytesIO()
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        buffer,
        compression="zstd",
        version="2.6",
    )
    return buffer.getvalue()


def _publish_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise HistoricalStudyPublicationError(
                f"immutable object collision at {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)


def _publish_table(
    root: Path,
    *,
    logical_name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload = _parquet_bytes(frame)
    digest = _sha(payload)
    hexadecimal = digest.removeprefix("sha256:")
    relative = Path("objects") / "sha256" / hexadecimal[:2] / f"{hexadecimal}.parquet"
    _publish_immutable(root / relative, payload)
    return {
        "logical_name": logical_name,
        "object_path": relative.as_posix(),
        "object_sha256": digest,
        "byte_length": len(payload),
        "row_count": len(frame),
        "format": "parquet",
    }


def publish_historical_study_tables(
    tables: HistoricalFactorStudy,
    *,
    output_root: str | Path,
    source_manifest_sha256s: tuple[str, ...],
    development_game_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> HistoricalStudyPublication:
    """Publish derived tables without reading a source or holdout."""

    if not isinstance(tables, HistoricalFactorStudy):
        raise HistoricalStudyPublicationError(
            "tables must be HistoricalFactorStudy"
        )
    if (
        type(source_manifest_sha256s) is not tuple
        or not source_manifest_sha256s
        or any(_SHA_RE.fullmatch(value) is None for value in source_manifest_sha256s)
    ):
        raise HistoricalStudyPublicationError(
            "source_manifest_sha256s must be non-empty SHA-256 values"
        )
    if type(development_game_count) is not int or development_game_count < 1:
        raise HistoricalStudyPublicationError(
            "development_game_count must be positive"
        )
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise HistoricalStudyPublicationError("bootstrap_samples must be positive")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise HistoricalStudyPublicationError(
            "bootstrap_seed must be nonnegative"
        )
    root = Path(output_root).resolve()
    objects = [
        _publish_table(
            root,
            logical_name=name,
            frame=getattr(tables, name),
        )
        for name in _TABLE_NAMES
    ]
    material = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "cohort": "development",
        "development_game_count": development_game_count,
        "source_manifest_sha256s": sorted(source_manifest_sha256s),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "claim_boundary": CLAIM_BOUNDARY,
        "objects": sorted(objects, key=lambda value: str(value["logical_name"])),
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
    }
    bundle_sha = _sha(_canonical_bytes(material))
    manifest = {**material, "bundle_sha256": bundle_sha}
    payload = _canonical_bytes(manifest)
    manifest_sha = _sha(payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        root
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    _publish_immutable(manifest_path, payload)
    return HistoricalStudyPublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        bundle_sha256=bundle_sha,
    )


def verify_historical_study_publication(
    manifest_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Verify manifest and object bytes before exposing any table."""

    path = Path(manifest_path).resolve()
    payload = path.read_bytes()
    digest = _sha(payload)
    if path.stem.removesuffix(".manifest") != digest.removeprefix("sha256:"):
        raise HistoricalStudyPublicationError("manifest SHA-256 path mismatch")
    manifest = json.loads(payload)
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("experiment_id") != "X-13"
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
        or manifest.get("final_holdout_access") != "CLOSED"
        or manifest.get("holdout_reaction_accessed") is not False
    ):
        raise HistoricalStudyPublicationError("publication manifest is invalid")
    object_root = Path(root).resolve()
    frames: dict[str, pd.DataFrame] = {}
    for reference in manifest.get("objects", ()):
        object_path = object_root / str(reference["object_path"])
        if (
            not object_path.is_file()
            or _file_sha(object_path) != reference["object_sha256"]
            or object_path.stat().st_size != reference["byte_length"]
        ):
            raise HistoricalStudyPublicationError(
                f"{reference.get('logical_name')} object SHA-256 mismatch"
            )
        frame = pq.read_table(object_path).to_pandas()
        if len(frame) != reference["row_count"]:
            raise HistoricalStudyPublicationError(
                f"{reference.get('logical_name')} row count mismatch"
            )
        frames[str(reference["logical_name"])] = frame
    if set(frames) != set(_TABLE_NAMES):
        raise HistoricalStudyPublicationError("publication object set is incomplete")
    return frames


__all__ = [
    "BUILDER_VERSION",
    "HistoricalStudyPublication",
    "HistoricalStudyPublicationError",
    "SCHEMA",
    "publish_historical_study_tables",
    "verify_historical_study_publication",
]
