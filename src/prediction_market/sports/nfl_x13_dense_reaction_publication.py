"""Development-only, content-addressed X-13 dense-reaction publication.

This module consumes only the already sealed development market and sports
feature batches.  It verifies every selected manifest and Parquet object by
byte length and SHA-256 before deriving the dense source-time tables.  It does
not open raw capture documents, reconstruct a game, or access holdout reaction
data.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_x13_dense_reaction import (
    CLAIM_BOUNDARY,
    DenseReactionError,
    build_dense_reaction_tables_v2,
)


SCHEMA = "nfl_x13_dense_reaction_publication_v1"
GAME_SCHEMA = "nfl_x13_dense_reaction_game_publication_v1"
BUILDER_VERSION = "nfl-x13-dense-reaction-publication-v3"
EXPECTED_EXACT153_GAMES = 153
_SHA256_PREFIX = "sha256:"
_SALIENCE_FIELDS = (
    "episode_is_score",
    "episode_is_turnover",
    "penalty",
    "replay_or_challenge",
    "timeout",
    "safety",
    "punt_blocked",
    "return_touchdown",
    "fumble_lost",
    "interception",
    "fourth_down_failed",
)
_FEATURE_REQUIRED = {
    "game_id",
    "play_id",
    "episode_id",
    "order_sequence",
    "source_time_utc",
    "game_seconds_remaining",
    "score_margin_focal",
    "yardline_100",
    "possession_team",
    "focal_team",
    "down",
    "distance",
    "event_eligible",
    "episode_audit_only",
    *_SALIENCE_FIELDS,
}


class DenseReactionPublicationError(RuntimeError):
    """A sealed dense-reaction input or output invariant failed closed."""


@dataclass(frozen=True, slots=True)
class DenseReactionPublication:
    """One immutable development-batch dense-reaction publication."""

    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    game_ids: tuple[str, ...]
    holdout_reaction_accessed: bool = False


@dataclass(frozen=True, slots=True)
class _MarketGameSource:
    game_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest_bundle_sha256: str
    stages: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class _FeatureGameSource:
    game_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest_bundle_sha256: str
    feature_view: Mapping[str, object]


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
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return _SHA256_PREFIX + digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_SHA256_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise DenseReactionPublicationError(f"{field} is not a SHA-256")
    return value


def _safe_relative(root: Path, relative: object, *, field: str) -> Path:
    if type(relative) is not str or not relative:
        raise DenseReactionPublicationError(f"{field} must be a relative path")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise DenseReactionPublicationError(f"{field} escapes its sealed root")
    return candidate


def _content_addressed_sha(path: Path, *, suffix: str, field: str) -> str:
    filename = path.name
    if not filename.endswith(suffix):
        raise DenseReactionPublicationError(f"{field} path is not content-addressed")
    hexadecimal = filename[: -len(suffix)]
    if len(hexadecimal) != 64 or any(
        character not in "0123456789abcdef" for character in hexadecimal
    ):
        raise DenseReactionPublicationError(f"{field} path is not content-addressed")
    if path.parent.name != hexadecimal[:2] or path.parent.parent.name != "sha256":
        raise DenseReactionPublicationError(f"{field} path is not content-addressed")
    return _SHA256_PREFIX + hexadecimal


def _read_verified_json(
    path: Path,
    *,
    expected_sha256: str,
    field: str,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DenseReactionPublicationError(f"{field} is not a regular file")
    if _sha256_file(path) != expected_sha256:
        raise DenseReactionPublicationError(f"{field} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseReactionPublicationError(f"{field} is not valid JSON") from error
    if type(value) is not dict:
        raise DenseReactionPublicationError(f"{field} must be a JSON object")
    return value


def _verify_bundle(
    value: Mapping[str, object],
    *,
    field: str,
    digest_field: str = "bundle_sha256",
) -> str:
    material = dict(value)
    bundle_sha256 = _require_sha256(
        material.pop(digest_field, None), field=f"{field}.{digest_field}"
    )
    if _sha256_bytes(_canonical_bytes(material)) != bundle_sha256:
        raise DenseReactionPublicationError(f"{field} bundle SHA-256 mismatch")
    return bundle_sha256


def _descriptor(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise DenseReactionPublicationError(f"{field} descriptor is missing")
    descriptor = dict(value)
    _require_sha256(descriptor.get("object_sha256"), field=f"{field}.object_sha256")
    if (
        type(descriptor.get("object_path")) is not str
        or type(descriptor.get("byte_length")) is not int
        or descriptor["byte_length"] < 0
        or type(descriptor.get("row_count")) is not int
        or descriptor["row_count"] < 0
    ):
        raise DenseReactionPublicationError(f"{field} descriptor is malformed")
    if descriptor.get("format", "parquet") != "parquet":
        raise DenseReactionPublicationError(f"{field} is not a Parquet descriptor")
    return descriptor


def _read_verified_parquet(
    root: Path,
    descriptor: Mapping[str, object],
    *,
    field: str,
) -> pd.DataFrame:
    path = _safe_relative(root, descriptor.get("object_path"), field=f"{field}.object_path")
    if path.is_symlink() or not path.is_file():
        raise DenseReactionPublicationError(f"{field} object is not a regular file")
    if path.stat().st_size != descriptor["byte_length"]:
        raise DenseReactionPublicationError(f"{field} byte length mismatch")
    if _sha256_file(path) != descriptor["object_sha256"]:
        raise DenseReactionPublicationError(f"{field} SHA-256 mismatch")
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata is None or parquet.metadata.num_rows != descriptor["row_count"]:
            raise DenseReactionPublicationError(f"{field} row count mismatch")
        return pq.read_table(path).to_pandas()
    except DenseReactionPublicationError:
        raise
    except Exception as error:
        raise DenseReactionPublicationError(f"{field} Parquet is unreadable") from error


def _load_market_sources(
    *,
    root: Path,
    batch_index_path: Path,
    expected_game_count: int,
) -> tuple[str, str, dict[str, _MarketGameSource]]:
    batch_path = batch_index_path.resolve()
    if not batch_path.is_relative_to(root.resolve()):
        raise DenseReactionPublicationError("market batch index escapes market root")
    batch_sha256 = _content_addressed_sha(
        batch_path, suffix=".batch-index.json", field="market batch index"
    )
    batch = _read_verified_json(
        batch_path, expected_sha256=batch_sha256, field="market batch index"
    )
    verified_batch_sha = _verify_bundle(
        batch, field="market batch index", digest_field="batch_sha256"
    )
    if (
        batch.get("schema") != "nfl_expansion_development_market_batch_index_v1"
        or batch.get("experiment_id") != "X-13"
        or batch.get("cohort") != "development"
        or batch.get("game_count") != expected_game_count
        or batch.get("publication_gate") != "PASS"
        or batch.get("final_holdout_access") != "CLOSED"
    ):
        raise DenseReactionPublicationError("market batch is not a sealed development cohort")
    game_rows = batch.get("games")
    if type(game_rows) is not list or len(game_rows) != expected_game_count:
        raise DenseReactionPublicationError("market batch has incomplete game coverage")
    sources: dict[str, _MarketGameSource] = {}
    for value in game_rows:
        if type(value) is not dict or type(value.get("game_id")) is not str:
            raise DenseReactionPublicationError("market batch game reference is malformed")
        game_id = value["game_id"]
        if game_id in sources:
            raise DenseReactionPublicationError("market batch has duplicate game IDs")
        manifest_sha256 = _require_sha256(
            value.get("manifest_sha256"), field=f"{game_id}.market_manifest_sha256"
        )
        manifest_path = _safe_relative(
            root, value.get("manifest_path"), field=f"{game_id}.market_manifest_path"
        )
        if _content_addressed_sha(
            manifest_path, suffix=".manifest.json", field=f"{game_id} market manifest"
        ) != manifest_sha256:
            raise DenseReactionPublicationError(f"{game_id} market manifest path/hash mismatch")
        manifest = _read_verified_json(
            manifest_path,
            expected_sha256=manifest_sha256,
            field=f"{game_id} market manifest",
        )
        manifest_bundle_sha = _verify_bundle(manifest, field=f"{game_id} market manifest")
        if (
            manifest.get("schema")
            != "nfl_expansion_development_market_single_game_manifest_v1"
            or manifest.get("experiment_id") != "X-13"
            or manifest.get("cohort") != "development"
            or manifest.get("game_id") != game_id
            or manifest.get("publication_gate") != "PASS"
            or manifest_bundle_sha != value.get("bundle_sha256")
        ):
            raise DenseReactionPublicationError(f"{game_id} market manifest contract mismatch")
        stages = manifest.get("stages")
        if type(stages) is not dict:
            raise DenseReactionPublicationError(f"{game_id} market stages are missing")
        needed = {
            stage: _descriptor(stages.get(stage), field=f"{game_id}.{stage}")
            for stage in (
                "contract_inventory",
                "actual_market_observations",
                "layer_g_source_timeline",
            )
        }
        sources[game_id] = _MarketGameSource(
            game_id=game_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            manifest_bundle_sha256=manifest_bundle_sha,
            stages=needed,
        )
    return batch_sha256, verified_batch_sha, sources


def _load_feature_sources(
    *,
    root: Path,
    batch_index_path: Path,
    expected_game_count: int,
) -> tuple[str, str, dict[str, _FeatureGameSource]]:
    batch_path = batch_index_path.resolve()
    if not batch_path.is_relative_to(root.resolve()):
        raise DenseReactionPublicationError("feature batch index escapes feature root")
    batch_sha256 = _content_addressed_sha(
        batch_path, suffix=".batch-index.json", field="feature batch index"
    )
    batch = _read_verified_json(
        batch_path, expected_sha256=batch_sha256, field="feature batch index"
    )
    verified_batch_sha = _verify_bundle(
        batch, field="feature batch index", digest_field="batch_sha256"
    )
    if (
        batch.get("schema") != "nfl_expansion_development_feature_batch_index_v1"
        or batch.get("scope") != f"development-{expected_game_count}"
        or batch.get("cohort") != "development"
        or batch.get("requested_game_count") != expected_game_count
        or batch.get("published_game_count") != expected_game_count
        or batch.get("failed_game_count") != 0
        or batch.get("publication_gate") != "PASS"
        or batch.get("reaction_access") != "CLOSED"
        or batch.get("market_data_read") is not False
    ):
        raise DenseReactionPublicationError("feature batch is not sealed sports-only development data")
    game_rows = batch.get("games")
    if type(game_rows) is not list or len(game_rows) != expected_game_count:
        raise DenseReactionPublicationError("feature batch has incomplete game coverage")
    sources: dict[str, _FeatureGameSource] = {}
    for value in game_rows:
        if type(value) is not dict or type(value.get("game_id")) is not str:
            raise DenseReactionPublicationError("feature batch game reference is malformed")
        game_id = value["game_id"]
        if game_id in sources:
            raise DenseReactionPublicationError("feature batch has duplicate game IDs")
        manifest_sha256 = _require_sha256(
            value.get("manifest_sha256"), field=f"{game_id}.feature_manifest_sha256"
        )
        manifest_path = _safe_relative(
            root, value.get("manifest_path"), field=f"{game_id}.feature_manifest_path"
        )
        if _content_addressed_sha(
            manifest_path, suffix=".manifest.json", field=f"{game_id} feature manifest"
        ) != manifest_sha256:
            raise DenseReactionPublicationError(f"{game_id} feature manifest path/hash mismatch")
        manifest = _read_verified_json(
            manifest_path,
            expected_sha256=manifest_sha256,
            field=f"{game_id} feature manifest",
        )
        manifest_bundle_sha = _verify_bundle(manifest, field=f"{game_id} feature manifest")
        if (
            manifest.get("schema")
            != "nfl_expansion_development_single_game_feature_manifest_v1"
            or manifest.get("cohort") != "development"
            or manifest.get("game_id") != game_id
            or manifest.get("publication_gate") != "PASS"
            or manifest.get("reaction_access") != "CLOSED"
            or manifest.get("market_data_read") is not False
            or manifest_bundle_sha != value.get("bundle_sha256")
        ):
            raise DenseReactionPublicationError(f"{game_id} feature manifest contract mismatch")
        feature_view = _descriptor(
            manifest.get("feature_view"), field=f"{game_id}.feature_view"
        )
        if (
            feature_view["object_sha256"] != value.get("object_sha256")
            or feature_view["row_count"] != value.get("row_count")
        ):
            raise DenseReactionPublicationError(f"{game_id} feature object reference mismatch")
        sources[game_id] = _FeatureGameSource(
            game_id=game_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            manifest_bundle_sha256=manifest_bundle_sha,
            feature_view=feature_view,
        )
    return batch_sha256, verified_batch_sha, sources


def _required_columns(frame: pd.DataFrame, *, columns: set[str], field: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise DenseReactionPublicationError(
            f"{field} is missing required columns: {', '.join(missing)}"
        )


def _boolean(frame: pd.DataFrame, field: str, *, label: str) -> pd.Series:
    value = frame[field]
    valid = value.map(lambda child: isinstance(child, bool))
    if not valid.all():
        raise DenseReactionPublicationError(f"{label}.{field} must be boolean")
    return value.astype(bool)


def _canonical_episode_features(features: pd.DataFrame, *, game_id: str) -> pd.DataFrame:
    _required_columns(features, columns=_FEATURE_REQUIRED, field=f"{game_id} sports features")
    if features["game_id"].isna().any() or set(features["game_id"].astype(str)) != {game_id}:
        raise DenseReactionPublicationError(f"{game_id} sports feature identity mismatch")
    if features["episode_id"].isna().any() or features["episode_id"].astype(str).str.strip().eq("").any():
        raise DenseReactionPublicationError(f"{game_id} sports features have a blank episode ID")
    source = features.copy()
    source["_source_time"] = pd.to_datetime(
        source["source_time_utc"], utc=True, errors="coerce", format="mixed"
    )
    if source["order_sequence"].isna().any() or source["play_id"].isna().any():
        raise DenseReactionPublicationError(f"{game_id} sports feature ordering is incomplete")
    # A finalized episode may include a scoring play followed by a try, a
    # penalty/review administrative row, or another live row.  Its covariates
    # must come from the *first* row's pre-state; using the final row would
    # leak a post-event state into a pre-event factor/control.  The published
    # Layer G contract, however, anchors a finalized episode at its final
    # timed row (for example, TD + PAT), so we retain that endpoint strictly
    # for source-time joining.  Episode semantics are an OR across all
    # constituent rows.
    for field in ("event_eligible", "episode_audit_only", *_SALIENCE_FIELDS):
        source[field] = _boolean(source, field, label=f"{game_id} sports features")

    rows: list[pd.Series] = []
    for _, child in source.groupby("episode_id", sort=True, dropna=False):
        timed = child[child["_source_time"].notna()]
        ordered = child.sort_values(
            ["_source_time", "order_sequence", "play_id"],
            kind="mergesort",
            na_position="first",
        )
        selected = ordered.iloc[0].copy()
        endpoint = (timed if not timed.empty else ordered).sort_values(
            ["_source_time", "order_sequence", "play_id"],
            kind="mergesort",
            na_position="first",
        ).iloc[-1]
        selected["source_time_utc"] = endpoint["source_time_utc"]
        # A multi-row episode is analysis-eligible when it contains a live
        # event; it is audit-only only when every constituent row is audit
        # only.  Salience is deliberately conservative: any score, turnover,
        # review, penalty, etc. ends a routine control window.
        selected["event_eligible"] = bool(child["event_eligible"].any())
        selected["episode_audit_only"] = bool(child["episode_audit_only"].all())
        for field in _SALIENCE_FIELDS:
            selected[field] = bool(child[field].any())
        rows.append(selected)
    result = pd.DataFrame(rows).reset_index(drop=True)
    for field in ("event_eligible", "episode_audit_only", *_SALIENCE_FIELDS):
        result[field] = _boolean(result, field, label=f"{game_id} sports features")
    return result


def _episode_inputs(
    *,
    game_id: str,
    layer_g: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    _required_columns(
        layer_g,
        columns={
            "game_id",
            "episode_id",
            "delay_seconds",
            "source_time_utc",
            "interval_start_utc",
            "interval_end_utc",
            "layer",
        },
        field=f"{game_id} Layer G",
    )
    source = layer_g[
        layer_g["layer"].eq("G") & layer_g["delay_seconds"].eq(0)
    ].copy()
    if source.empty:
        raise DenseReactionPublicationError(f"{game_id} Layer G has no zero-delay rows")
    if set(source["game_id"].astype(str)) != {game_id}:
        raise DenseReactionPublicationError(f"{game_id} Layer G identity mismatch")
    if source.duplicated(["game_id", "episode_id"]).any():
        raise DenseReactionPublicationError(f"{game_id} Layer G has duplicate episodes")
    for field in ("source_time_utc", "interval_start_utc", "interval_end_utc"):
        source[field] = pd.to_datetime(
            source[field], utc=True, errors="coerce", format="mixed"
        )
    if source[["source_time_utc", "interval_start_utc", "interval_end_utc"]].isna().any().any():
        raise DenseReactionPublicationError(f"{game_id} Layer G has a source-time gap")
    if source["interval_end_utc"].lt(source["interval_start_utc"]).any():
        raise DenseReactionPublicationError(f"{game_id} Layer G has an inverted interval")
    canonical = _canonical_episode_features(features, game_id=game_id)
    if set(canonical["episode_id"].astype(str)) != set(source["episode_id"].astype(str)):
        raise DenseReactionPublicationError(f"{game_id} Layer G and sports episodes differ")
    joined = source.merge(
        canonical,
        on=["game_id", "episode_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_g", "_feature"),
    )
    if len(joined) != len(source):
        raise DenseReactionPublicationError(f"{game_id} has a missing sports episode")
    feature_source = pd.to_datetime(
        joined["source_time_utc_feature"], utc=True, errors="coerce", format="mixed"
    )
    if feature_source.isna().any() or feature_source.ne(joined["source_time_utc_g"]).any():
        raise DenseReactionPublicationError(f"{game_id} Layer G/sports source time mismatch")
    salient = joined[list(_SALIENCE_FIELDS)].any(axis=1)
    routine = joined["event_eligible"] & ~joined["episode_audit_only"] & ~salient
    next_salient: list[pd.Timestamp | None] = []
    starts = joined["interval_start_utc"]
    ends = joined["interval_end_utc"]
    salient_starts = sorted(starts[salient].tolist())
    for end in ends:
        next_salient.append(next((value for value in salient_starts if value > end), None))
    possession = joined["possession_team"]
    focal = joined["focal_team"]
    if possession.isna().any() or focal.isna().any() or possession.astype(str).str.strip().eq("").any() or focal.astype(str).str.strip().eq("").any():
        raise DenseReactionPublicationError(f"{game_id} sports possession orientation is incomplete")
    episodes = pd.DataFrame(
        {
            "game_id": joined["game_id"].astype(str),
            "episode_id": joined["episode_id"].astype(str),
            "game_time_seconds": joined["game_seconds_remaining"],
            "signed_margin": joined["score_margin_focal"],
            "field_position": joined["yardline_100"],
            "possession": possession.astype(str),
            "actor_beneficiary": pd.Series(
                [
                    "FOCAL_POSSESSION" if actor == beneficiary else "OPPONENT_POSSESSION"
                    for actor, beneficiary in zip(possession.astype(str), focal.astype(str), strict=True)
                ],
                index=joined.index,
            ),
            "down": joined["down"],
            "distance": joined["distance"],
        }
    )
    timeline = pd.DataFrame(
        {
            "game_id": joined["game_id"].astype(str),
            "episode_id": joined["episode_id"].astype(str),
            "source_time_start_utc": joined["interval_start_utc"],
            "source_time_end_utc": joined["interval_end_utc"],
            "next_salient_source_time_utc": next_salient,
        }
    )
    routine_ids = tuple(sorted(joined.loc[routine, "episode_id"].astype(str)))
    return episodes, timeline, routine_ids


def _actual_with_family(
    *,
    game_id: str,
    actual: pd.DataFrame,
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    join_keys = ["game_id", "logical_market_id", "outcome", "venue"]
    _required_columns(actual, columns=set(join_keys), field=f"{game_id} actual market")
    _required_columns(contracts, columns={*join_keys, "family"}, field=f"{game_id} contract inventory")
    contract_families = contracts[[*join_keys, "family"]].copy()
    if contract_families["family"].isna().any() or contract_families["family"].astype(str).str.strip().eq("").any():
        raise DenseReactionPublicationError(f"{game_id} contract inventory has a blank family")
    if contract_families.duplicated(join_keys).any():
        raise DenseReactionPublicationError(f"{game_id} contract family join is ambiguous")
    result = actual.merge(contract_families, on=join_keys, how="left", validate="many_to_one")
    if result["family"].isna().any():
        raise DenseReactionPublicationError(f"{game_id} actual market lacks a contract family")
    result["market_family"] = result["family"].astype(str)
    return result.drop(columns="family")


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        sink,
        compression="zstd",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise DenseReactionPublicationError(f"immutable content-addressed collision: {path}")
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _publish_object(
    *,
    output_root: Path,
    logical_name: str,
    payload: bytes,
    suffix: str,
    row_count: int,
) -> dict[str, object]:
    digest = _sha256_bytes(payload)
    hexadecimal = digest.removeprefix(_SHA256_PREFIX)
    path = output_root / "objects" / "sha256" / hexadecimal[:2] / f"{hexadecimal}.{suffix}"
    _publish_immutable(path, payload)
    return {
        "logical_name": logical_name,
        "object_path": str(path.relative_to(output_root)),
        "object_sha256": digest,
        "byte_length": len(payload),
        "row_count": row_count,
        "format": suffix,
    }


def _source_record(
    *,
    kind: str,
    root: Path,
    path: Path,
    sha256: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "path": str(path.relative_to(root)),
        "sha256": sha256,
        "byte_verified": True,
    }


def _publish_game(
    *,
    game_id: str,
    output_root: Path,
    market_root: Path,
    feature_root: Path,
    market: _MarketGameSource,
    feature: _FeatureGameSource,
) -> dict[str, object]:
    contracts = _read_verified_parquet(
        market_root, market.stages["contract_inventory"], field=f"{game_id}.contract_inventory"
    )
    actual = _read_verified_parquet(
        market_root,
        market.stages["actual_market_observations"],
        field=f"{game_id}.actual_market_observations",
    )
    layer_g = _read_verified_parquet(
        market_root,
        market.stages["layer_g_source_timeline"],
        field=f"{game_id}.layer_g_source_timeline",
    )
    sports_features = _read_verified_parquet(
        feature_root, feature.feature_view, field=f"{game_id}.sports_feature_view"
    )
    episodes, timeline, routine_ids = _episode_inputs(
        game_id=game_id, layer_g=layer_g, features=sports_features
    )
    actual = _actual_with_family(game_id=game_id, actual=actual, contracts=contracts)
    try:
        tables = build_dense_reaction_tables_v2(episodes, actual, timeline)
    except DenseReactionError as error:
        raise DenseReactionPublicationError(f"{game_id} dense reaction build failed: {error}") from error
    source_inputs = [
        _source_record(kind="market_manifest", root=market_root, path=market.manifest_path, sha256=market.manifest_sha256),
        _source_record(kind="sports_feature_manifest", root=feature_root, path=feature.manifest_path, sha256=feature.manifest_sha256),
    ]
    for stage in ("contract_inventory", "actual_market_observations", "layer_g_source_timeline"):
        descriptor = market.stages[stage]
        source_inputs.append(
            {
                "kind": stage,
                "path": str(descriptor["object_path"]),
                "sha256": descriptor["object_sha256"],
                "byte_verified": True,
            }
        )
    source_inputs.append(
        {
            "kind": "sports_feature_view",
            "path": str(feature.feature_view["object_path"]),
            "sha256": feature.feature_view["object_sha256"],
            "byte_verified": True,
        }
    )
    objects = {
        "paths": _publish_object(output_root=output_root, logical_name=f"{game_id}.paths", payload=_parquet_bytes(tables.paths), suffix="parquet", row_count=len(tables.paths)),
        "path_summaries": _publish_object(output_root=output_root, logical_name=f"{game_id}.path_summaries", payload=_parquet_bytes(tables.path_summaries), suffix="parquet", row_count=len(tables.path_summaries)),
        "attrition": _publish_object(output_root=output_root, logical_name=f"{game_id}.attrition", payload=_parquet_bytes(tables.attrition), suffix="parquet", row_count=len(tables.attrition)),
        "routine_control_episode_ids": _publish_object(output_root=output_root, logical_name=f"{game_id}.routine_control_episode_ids", payload=_canonical_bytes({"game_id": game_id, "episode_ids": list(routine_ids)}), suffix="json", row_count=len(routine_ids)),
        "lineage": _publish_object(output_root=output_root, logical_name=f"{game_id}.lineage", payload=_canonical_bytes({"game_id": game_id, "inputs": source_inputs, "claim_boundary": CLAIM_BOUNDARY}), suffix="json", row_count=len(source_inputs)),
    }
    material = {
        "schema": GAME_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "cohort": "development",
        "game_id": game_id,
        "claim_boundary": CLAIM_BOUNDARY,
        "market_manifest_sha256": market.manifest_sha256,
        "market_bundle_sha256": market.manifest_bundle_sha256,
        "feature_manifest_sha256": feature.manifest_sha256,
        "feature_bundle_sha256": feature.manifest_bundle_sha256,
        "objects": objects,
        "episode_count": len(episodes),
        "routine_control_episode_count": len(routine_ids),
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
    }
    bundle_sha256 = _sha256_bytes(_canonical_bytes(material))
    manifest = {**material, "bundle_sha256": bundle_sha256}
    payload = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(payload)
    hexadecimal = manifest_sha256.removeprefix(_SHA256_PREFIX)
    manifest_path = output_root / "single-game" / game_id / "manifests" / "sha256" / hexadecimal[:2] / f"{hexadecimal}.manifest.json"
    _publish_immutable(manifest_path, payload)
    return {
        "game_id": game_id,
        "manifest_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": manifest_sha256,
        "bundle_sha256": bundle_sha256,
        "objects": objects,
        "episode_count": len(episodes),
        "routine_control_episode_count": len(routine_ids),
    }


def publish_dense_reaction_batch(
    *,
    market_root: str | Path,
    market_batch_index_path: str | Path,
    feature_root: str | Path,
    feature_batch_index_path: str | Path,
    output_root: str | Path,
    expected_game_count: int = EXPECTED_EXACT153_GAMES,
) -> DenseReactionPublication:
    """Publish dense V2 tables for one sealed development batch.

    ``expected_game_count`` defaults to the governed exact-153 cohort.  The
    parameter exists solely to permit small synthetic sealed-batch tests; a
    production call must leave it at 153.
    """

    if type(expected_game_count) is not int or expected_game_count < 1:
        raise DenseReactionPublicationError("expected_game_count must be positive")
    market_root_path = Path(market_root).resolve()
    feature_root_path = Path(feature_root).resolve()
    output_root_path = Path(output_root).resolve()
    if not market_root_path.is_dir() or not feature_root_path.is_dir():
        raise DenseReactionPublicationError("sealed source roots must be directories")
    market_index_sha, market_bundle_sha, market_sources = _load_market_sources(
        root=market_root_path,
        batch_index_path=Path(market_batch_index_path),
        expected_game_count=expected_game_count,
    )
    feature_index_sha, feature_bundle_sha, feature_sources = _load_feature_sources(
        root=feature_root_path,
        batch_index_path=Path(feature_batch_index_path),
        expected_game_count=expected_game_count,
    )
    if set(market_sources) != set(feature_sources):
        raise DenseReactionPublicationError("market and sports batches cover different games")
    games = [
        _publish_game(
            game_id=game_id,
            output_root=output_root_path,
            market_root=market_root_path,
            feature_root=feature_root_path,
            market=market_sources[game_id],
            feature=feature_sources[game_id],
        )
        for game_id in sorted(market_sources)
    ]
    material = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": expected_game_count,
        "market_batch_index_sha256": market_index_sha,
        "market_batch_sha256": market_bundle_sha,
        "feature_batch_index_sha256": feature_index_sha,
        "feature_batch_sha256": feature_bundle_sha,
        "claim_boundary": CLAIM_BOUNDARY,
        "games": games,
        "final_holdout_access": "CLOSED",
        "holdout_reaction_accessed": False,
    }
    bundle_sha256 = _sha256_bytes(_canonical_bytes(material))
    manifest = {**material, "bundle_sha256": bundle_sha256}
    payload = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(payload)
    hexadecimal = manifest_sha256.removeprefix(_SHA256_PREFIX)
    manifest_path = output_root_path / "manifests" / "sha256" / hexadecimal[:2] / f"{hexadecimal}.manifest.json"
    _publish_immutable(manifest_path, payload)
    return DenseReactionPublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        bundle_sha256=bundle_sha256,
        game_ids=tuple(game["game_id"] for game in games),
    )


__all__ = [
    "BUILDER_VERSION",
    "DenseReactionPublication",
    "DenseReactionPublicationError",
    "EXPECTED_EXACT153_GAMES",
    "GAME_SCHEMA",
    "SCHEMA",
    "publish_dense_reaction_batch",
]
