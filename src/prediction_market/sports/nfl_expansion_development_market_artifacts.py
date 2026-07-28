"""Strict artifact I/O for the NFL expansion development market analysis.

This module owns no market semantics.  It verifies the frozen access decision,
capture, sports-feature, and factor-registry artifacts and exposes only the
153-game development cohort.  The final 81-game holdout is rejected before
any game-level file is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import pandas as pd
import pyarrow.parquet as pq

from prediction_market import contracts


EXPECTED_UNLOCK_OBJECT_SHA256 = (
    "sha256:bf85722240985f5a9eca206bfae0f1b3ff2a6947687d4c87bdf7d082049cd4f3"
)
EXPECTED_UNLOCK_MANIFEST_SHA256 = (
    "sha256:185e2275fd50c68e80d9f8dc56e875f767f723e9994eda82a1851a9cbc0afa1b"
)
EXPECTED_CAPTURE_BATCH_SHA256 = (
    "sha256:ce8e509657b6b9eb0e9cb116758242a1462b45090fc8a668c8af6de2c2a8c061"
)
EXPECTED_CAPTURE_PLAN_SHA256 = (
    "sha256:5fa54dd135deb4c207acfed68fa0b963904f17fa1b5cff0eeca596ca0ad2e712"
)
EXPECTED_FEATURE_BATCH_FILE_SHA256 = (
    "sha256:3a09538aa3013d4ddab9aa5e24299c91914b98a41701854037c353326ccf4866"
)
EXPECTED_FEATURE_BATCH_SHA256 = (
    "sha256:91679220e8f379dde18c5a4859bef11e6a130ae09d9f253339f67e43aa50d4f9"
)
EXPECTED_FACTOR_REGISTRY_SHA256 = (
    "sha256:295e4e48237f8523fe3153f28b5b28adc9f6e1d3a219568047da5c250ed243c1"
)
EXPECTED_GOVERNANCE_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
EXPECTED_SPORTS_CLEAN_BATCH_SHA256 = (
    "sha256:4ce0759112084e45a7670c4436e240568215bc825c3133c40827061094d2c934"
)
EXPECTED_DEVELOPMENT_COUNT = 153
EXPECTED_FINAL_HOLDOUT_COUNT = 81
EXPECTED_CAPTURE_COUNT = 234
EXPECTED_FACTOR_COUNT = 104
EXPECTED_FORMAL_FACTOR_COUNT = 101
EXPECTED_HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60)
DISCOVERY_DELAY_SECONDS = 0
DISCOVERY_BOOTSTRAP_SEED = 20260725
DISCOVERY_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_SINGLE_GAME_STAGES = frozenset(
    {
        "contract_inventory",
        "actual_market_observations",
        "layer_m_intervals",
        "layer_g_source_timeline",
        "reaction_paths",
        "factor_observations",
        "factor_exclusion_audit",
        "single_game_results",
    }
)

DEFAULT_UNLOCK_OBJECT = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "reaction-access-amendment/development-open-v1/objects/sha256/bf/"
    "bf85722240985f5a9eca206bfae0f1b3ff2a6947687d4c87bdf7d082049cd4f3"
    ".json"
)
DEFAULT_UNLOCK_MANIFEST = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "reaction-access-amendment/development-open-v1/manifests/"
    "185e2275fd50c68e80d9f8dc56e875f767f723e9994eda82a1851a9cbc0afa1b"
    ".manifest.json"
)
DEFAULT_CAPTURE_BATCH = Path(
    "artifacts/market-observation/nfl/x13/expansion-capture/"
    "moneyline-reaction-closed-v1/batch/manifests/"
    "ce8e509657b6b9eb0e9cb116758242a1462b45090fc8a668c8af6de2c2a8c061"
    ".manifest.json"
)
DEFAULT_FEATURE_BATCH = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-development-sports-features/batches/manifests/sha256/3a/"
    "3a09538aa3013d4ddab9aa5e24299c91914b98a41701854037c353326ccf4866"
    ".batch-index.json"
)
DEFAULT_FACTOR_REGISTRY = Path("registries/factors/nfl_factor_registry_v2.json")

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"2025_(0[1-9]|1[0-8])_[A-Z]{2,3}_[A-Z]{2,3}\Z")
_STAGE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class DevelopmentArtifactError(RuntimeError):
    """A frozen artifact or development-only access invariant failed."""


@dataclass(frozen=True, slots=True)
class VerifiedUnlockAccess:
    object_path: Path
    object_sha256: str
    manifest_path: Path
    manifest_sha256: str
    development_game_ids: frozenset[str]
    final_holdout_game_ids: frozenset[str]
    document: Mapping[str, Any]

    def require_development(self, game_id: str) -> None:
        """Reject holdout access before a game-level path can be resolved."""

        if game_id in self.final_holdout_game_ids:
            raise DevelopmentArtifactError(
                f"final holdout access is prohibited: {game_id}"
            )
        if game_id not in self.development_game_ids:
            raise DevelopmentArtifactError(
                f"game is outside the development unlock: {game_id}"
            )


@dataclass(frozen=True, slots=True)
class VerifiedCaptureGameRef:
    game_id: str
    index_path: Path
    index_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    index_document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedFeatureGameRef:
    game_id: str
    manifest_path: Path
    manifest_sha256: str
    object_path: Path
    object_sha256: str
    rows_sha256: str
    row_count: int
    manifest_document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedRawDocument:
    game_id: str
    source_url: str
    request_sha256: str
    manifest_path: Path
    manifest_sha256: str
    object_path: Path
    object_sha256: str
    object_bytes: bytes
    manifest_document: Mapping[str, Any]
    value: object


@dataclass(frozen=True, slots=True)
class VerifiedDevelopmentInputs:
    project_root: Path
    raw_root: Path
    unlock: VerifiedUnlockAccess
    capture_batch_path: Path
    capture_batch_sha256: str
    capture_batch_document: Mapping[str, Any]
    development_capture_games: Mapping[str, VerifiedCaptureGameRef]
    feature_batch_path: Path
    feature_batch_file_sha256: str
    feature_batch_sha256: str
    feature_batch_document: Mapping[str, Any]
    development_feature_games: Mapping[str, VerifiedFeatureGameRef]
    factor_registry_path: Path
    factor_registry_sha256: str
    factor_registry_document: Mapping[str, Any]
    formal_factor_definitions: tuple[Mapping[str, Any], ...]

    def require_development(self, game_id: str) -> None:
        self.unlock.require_development(game_id)


@dataclass(frozen=True, slots=True)
class PublishedStageRef:
    stage: str
    object_path: Path
    object_sha256: str
    byte_length: int
    row_count: int
    column_count: int
    schema_fingerprint: str


@dataclass(frozen=True, slots=True)
class PublishedSingleGameBundle:
    game_id: str
    input_fingerprint: str
    bundle_sha256: str
    manifest_path: Path
    manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    stages: Mapping[str, PublishedStageRef]


@dataclass(frozen=True, slots=True)
class PublishedDevelopmentBatch:
    input_fingerprint: str
    batch_sha256: str
    index_path: Path
    index_sha256: str
    game_count: int
    merged_single_game_results: PublishedStageRef
    games: tuple[PublishedSingleGameBundle, ...]


@dataclass(frozen=True, slots=True)
class PublishedCrossGameDiscovery:
    batch_sha256: str
    object_path: Path
    object_sha256: str
    row_count: int
    manifest_path: Path
    manifest_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DevelopmentArtifactError(
            "publication material is not canonical JSON"
        ) from error
    return encoded + b"\n"


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise DevelopmentArtifactError(f"{field} is not a SHA-256 identifier")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> None:
    raise DevelopmentArtifactError(f"non-JSON numeric value: {value}")


def strict_load_json_bytes(payload: bytes, *, label: str) -> object:
    """Load JSON without losing decimal precision or accepting duplicate keys."""

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_reject_non_json_number,
        )
    except DevelopmentArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentArtifactError(f"invalid JSON: {label}") from error


def _require_object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise DevelopmentArtifactError(f"{label} must be a JSON object")
    return value


def _resolve_under(root: Path, path: str | Path, *, label: str) -> Path:
    root = root.resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if ".." in candidate.parts:
            raise DevelopmentArtifactError(f"{label} contains '..'")
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DevelopmentArtifactError(f"{label} escapes its frozen root") from error
    return resolved


def _read_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_length: int | None = None,
) -> bytes:
    expected_sha256 = (
        _require_sha256(expected_sha256, field=f"{label}.sha256")
        if expected_sha256 is not None
        else None
    )
    try:
        before = os.stat(path, follow_symlinks=False)
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise OSError("outside bounded byte limit")
        if expected_length is not None and before.st_size != expected_length:
            raise OSError("byte length mismatch")
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise DevelopmentArtifactError(
            f"{label} is not a stable bounded regular file: {path}"
        ) from error
    before_id = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_id = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_id != after_id or len(payload) != before.st_size:
        raise DevelopmentArtifactError(f"{label} changed while reading")
    actual_sha256 = _sha256_bytes(payload)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise DevelopmentArtifactError(f"{label} SHA-256 mismatch")
    return payload


def _verify_regular_file_streaming(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_length: int,
) -> None:
    expected_sha256 = _require_sha256(
        expected_sha256, field=f"{label}.sha256"
    )
    try:
        before = os.stat(path, follow_symlinks=False)
        if (
            path.is_symlink()
            or not path.is_file()
            or before.st_size != expected_length
            or expected_length < 0
        ):
            raise OSError("not the expected regular file")
        actual_sha256 = _sha256_file(path)
        after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise DevelopmentArtifactError(
            f"{label} is not a stable expected regular file"
        ) from error
    before_id = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_id = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_id != after_id or actual_sha256 != expected_sha256:
        raise DevelopmentArtifactError(f"{label} hash or identity changed")


def _read_json_object(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_length: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(
        path,
        label=label,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
        expected_length=expected_length,
    )
    return (
        _require_object(strict_load_json_bytes(payload, label=label), label=label),
        payload,
    )


def _default_raw_root(project_root: Path) -> Path:
    if project_root.parent.name == ".worktrees":
        return project_root.parent.parent / "var/raw"
    return project_root / "var/raw"


def _verify_unlock(
    *,
    project_root: Path,
    object_path: str | Path,
    manifest_path: str | Path,
) -> VerifiedUnlockAccess:
    object_path = _resolve_under(project_root, object_path, label="unlock object")
    manifest_path = _resolve_under(
        project_root, manifest_path, label="unlock manifest"
    )
    manifest, _ = _read_json_object(
        manifest_path,
        label="unlock manifest",
        max_bytes=2 * 1024 * 1024,
        expected_sha256=EXPECTED_UNLOCK_MANIFEST_SHA256,
    )
    document, object_bytes = _read_json_object(
        object_path,
        label="unlock object",
        max_bytes=2 * 1024 * 1024,
        expected_sha256=EXPECTED_UNLOCK_OBJECT_SHA256,
    )
    if (
        manifest.get("schema")
        != "nfl_expansion_reaction_access_amendment_manifest_v1"
        or manifest.get("object_sha256") != EXPECTED_UNLOCK_OBJECT_SHA256
        or manifest.get("byte_length") != len(object_bytes)
        or manifest.get("development_reaction_access")
        != "OPEN_FOR_REGISTERED_ANALYSIS"
        or manifest.get("final_holdout_reaction_access") != "CLOSED"
        or manifest.get("factor_registry_sha256")
        != EXPECTED_FACTOR_REGISTRY_SHA256
        or manifest.get("governance_object_sha256")
        != EXPECTED_GOVERNANCE_SHA256
        or manifest.get("sports_batch_sha256")
        != EXPECTED_SPORTS_CLEAN_BATCH_SHA256
        or document.get("schema")
        != "nfl_expansion_reaction_access_amendment_v1"
        or document.get("status") != "FROZEN"
    ):
        raise DevelopmentArtifactError("unlock artifact contract mismatch")
    split = document.get("split_access")
    if type(split) is not dict:
        raise DevelopmentArtifactError("unlock split_access is missing")
    development = split.get("development")
    final_holdout = split.get("final_holdout")
    if type(development) is not dict or type(final_holdout) is not dict:
        raise DevelopmentArtifactError("unlock cohorts are missing")
    development_ids = development.get("game_ids")
    final_ids = final_holdout.get("game_ids")
    if type(development_ids) is not list or type(final_ids) is not list:
        raise DevelopmentArtifactError("unlock game IDs are missing")
    development_set = frozenset(development_ids)
    final_set = frozenset(final_ids)
    if (
        len(development_ids) != EXPECTED_DEVELOPMENT_COUNT
        or len(development_set) != EXPECTED_DEVELOPMENT_COUNT
        or development.get("game_count") != EXPECTED_DEVELOPMENT_COUNT
        or len(final_ids) != EXPECTED_FINAL_HOLDOUT_COUNT
        or len(final_set) != EXPECTED_FINAL_HOLDOUT_COUNT
        or final_holdout.get("game_count") != EXPECTED_FINAL_HOLDOUT_COUNT
        or development_set & final_set
        or any(
            type(game_id) is not str
            or _GAME_ID_RE.fullmatch(game_id) is None
            for game_id in development_set | final_set
        )
        or development.get("reaction_access")
        != "OPEN_FOR_REGISTERED_ANALYSIS"
        or final_holdout.get("reaction_access") != "CLOSED"
    ):
        raise DevelopmentArtifactError("unlock 153/81 cohort contract mismatch")
    return VerifiedUnlockAccess(
        object_path=object_path,
        object_sha256=EXPECTED_UNLOCK_OBJECT_SHA256,
        manifest_path=manifest_path,
        manifest_sha256=EXPECTED_UNLOCK_MANIFEST_SHA256,
        development_game_ids=development_set,
        final_holdout_game_ids=final_set,
        document=MappingProxyType(document),
    )


def _verify_capture(
    *,
    project_root: Path,
    batch_path: str | Path,
    access: VerifiedUnlockAccess,
) -> tuple[Path, dict[str, Any], Mapping[str, VerifiedCaptureGameRef]]:
    batch_path = _resolve_under(project_root, batch_path, label="capture batch")
    batch, _ = _read_json_object(
        batch_path,
        label="capture batch",
        max_bytes=4 * 1024 * 1024,
        expected_sha256=EXPECTED_CAPTURE_BATCH_SHA256,
    )
    if (
        batch.get("schema") != "nfl_moneyline_expansion_batch_manifest_v1"
        or batch.get("game_count") != EXPECTED_CAPTURE_COUNT
        or batch.get("batch_plan_sha256") != EXPECTED_CAPTURE_PLAN_SHA256
        or batch.get("governance_sha256") != EXPECTED_GOVERNANCE_SHA256
        or batch.get("capture_scope") != "moneyline-only"
        or batch.get("reaction_access") != "CLOSED"
    ):
        raise DevelopmentArtifactError("capture batch contract mismatch")
    entries = batch.get("game_indexes")
    if type(entries) is not list or len(entries) != EXPECTED_CAPTURE_COUNT:
        raise DevelopmentArtifactError("capture batch game index is incomplete")
    capture_root = batch_path.parents[2]
    all_ids = access.development_game_ids | access.final_holdout_game_ids
    verified_all: dict[str, VerifiedCaptureGameRef] = {}
    for entry in entries:
        if type(entry) is not dict:
            raise DevelopmentArtifactError("capture game reference is not an object")
        game_id = entry.get("game_id")
        if type(game_id) is not str or game_id not in all_ids:
            raise DevelopmentArtifactError("capture game is outside frozen cohorts")
        if game_id in verified_all:
            raise DevelopmentArtifactError(f"duplicate capture game: {game_id}")
        index_sha256 = _require_sha256(
            entry.get("index_sha256"), field=f"{game_id}.index_sha256"
        )
        index_path = _resolve_under(
            capture_root,
            entry.get("index_path", ""),
            label=f"{game_id} capture index",
        )
        index, _ = _read_json_object(
            index_path,
            label=f"{game_id} capture index",
            max_bytes=8 * 1024 * 1024,
            expected_sha256=index_sha256,
        )
        checkpoint_path = _resolve_under(
            capture_root,
            Path("checkpoints") / f"{game_id}.json",
            label=f"{game_id} capture checkpoint",
        )
        checkpoint, checkpoint_bytes = _read_json_object(
            checkpoint_path,
            label=f"{game_id} capture checkpoint",
            max_bytes=1024 * 1024,
        )
        if (
            checkpoint.get("schema")
            != "nfl_moneyline_game_capture_checkpoint_v1"
            or checkpoint.get("game_id") != game_id
            or checkpoint.get("index_path") != entry.get("index_path")
            or checkpoint.get("index_sha256") != index_sha256
            or index.get("schema") != "nfl_moneyline_game_capture_index_v1"
            or index.get("game_id") != game_id
            or index.get("capture_scope") != "moneyline-only"
            or index.get("reaction_access") != "CLOSED"
            or entry.get("polymarket_trade_count")
            != index.get("polymarket_trades", {}).get("raw_item_count")
            or entry.get("kalshi_trade_count")
            != sum(
                row.get("raw_item_count", -1)
                for row in index.get("kalshi_trades", [])
                if type(row) is dict
            )
            or entry.get("raw_manifest_count")
            != len(index.get("raw_manifests", []))
        ):
            raise DevelopmentArtifactError(
                f"{game_id} capture index/checkpoint mismatch"
            )
        raw_refs = index.get("raw_manifests")
        if type(raw_refs) is not list or not raw_refs:
            raise DevelopmentArtifactError(f"{game_id} has no raw manifest refs")
        for raw_ref in raw_refs:
            if (
                type(raw_ref) is not dict
                or type(raw_ref.get("manifest_path")) is not str
                or type(raw_ref.get("object_path")) is not str
                or type(raw_ref.get("byte_length")) is not int
                or raw_ref.get("byte_length", 0) <= 0
            ):
                raise DevelopmentArtifactError(
                    f"{game_id} raw manifest reference is invalid"
                )
            _require_sha256(
                raw_ref.get("manifest_sha256"),
                field=f"{game_id}.raw_manifest_sha256",
            )
            _require_sha256(
                raw_ref.get("object_sha256"),
                field=f"{game_id}.raw_object_sha256",
            )
            _require_sha256(
                raw_ref.get("request_sha256"),
                field=f"{game_id}.raw_request_sha256",
            )
        verified_all[game_id] = VerifiedCaptureGameRef(
            game_id=game_id,
            index_path=index_path,
            index_sha256=index_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
            index_document=MappingProxyType(index),
        )
    if frozenset(verified_all) != all_ids:
        raise DevelopmentArtifactError("capture batch does not cover exact 234 games")
    development_only = {
        game_id: verified_all[game_id]
        for game_id in sorted(access.development_game_ids)
    }
    return batch_path, batch, MappingProxyType(development_only)


def _verify_features(
    *,
    project_root: Path,
    batch_path: str | Path,
    access: VerifiedUnlockAccess,
) -> tuple[
    Path,
    dict[str, Any],
    Mapping[str, VerifiedFeatureGameRef],
]:
    batch_path = _resolve_under(project_root, batch_path, label="feature batch")
    batch, _ = _read_json_object(
        batch_path,
        label="feature batch",
        max_bytes=4 * 1024 * 1024,
        expected_sha256=EXPECTED_FEATURE_BATCH_FILE_SHA256,
    )
    if (
        batch.get("schema")
        != "nfl_expansion_development_feature_batch_index_v1"
        or batch.get("scope") != "development-153"
        or batch.get("cohort") != "development"
        or batch.get("batch_sha256") != EXPECTED_FEATURE_BATCH_SHA256
        or batch.get("governance_object_sha256") != EXPECTED_GOVERNANCE_SHA256
        or batch.get("factor_registry_sha256")
        != EXPECTED_FACTOR_REGISTRY_SHA256
        or batch.get("sports_clean_batch_sha256")
        != EXPECTED_SPORTS_CLEAN_BATCH_SHA256
        or batch.get("requested_game_count") != EXPECTED_DEVELOPMENT_COUNT
        or batch.get("published_game_count") != EXPECTED_DEVELOPMENT_COUNT
        or batch.get("failed_game_count") != 0
        or batch.get("publication_gate") != "PASS"
        or batch.get("reaction_access") != "CLOSED"
        or batch.get("market_data_read") is not False
    ):
        raise DevelopmentArtifactError("feature batch contract mismatch")
    entries = batch.get("games")
    if type(entries) is not list or len(entries) != EXPECTED_DEVELOPMENT_COUNT:
        raise DevelopmentArtifactError("feature batch is incomplete")
    feature_root = batch_path.parents[4]
    verified: dict[str, VerifiedFeatureGameRef] = {}
    aggregate_rows = 0
    for entry in entries:
        if type(entry) is not dict:
            raise DevelopmentArtifactError("feature game reference is invalid")
        game_id = entry.get("game_id")
        access.require_development(game_id)
        if game_id in verified:
            raise DevelopmentArtifactError(f"duplicate feature game: {game_id}")
        manifest_sha256 = _require_sha256(
            entry.get("manifest_sha256"),
            field=f"{game_id}.feature_manifest_sha256",
        )
        manifest_path = _resolve_under(
            feature_root,
            entry.get("manifest_path", ""),
            label=f"{game_id} feature manifest",
        )
        manifest, _ = _read_json_object(
            manifest_path,
            label=f"{game_id} feature manifest",
            max_bytes=2 * 1024 * 1024,
            expected_sha256=manifest_sha256,
        )
        descriptor = manifest.get("feature_view")
        if type(descriptor) is not dict:
            raise DevelopmentArtifactError(f"{game_id} feature descriptor missing")
        object_sha256 = _require_sha256(
            descriptor.get("object_sha256"),
            field=f"{game_id}.feature_object_sha256",
        )
        rows_sha256 = _require_sha256(
            descriptor.get("rows_sha256"),
            field=f"{game_id}.feature_rows_sha256",
        )
        object_path = _resolve_under(
            feature_root,
            descriptor.get("object_path", ""),
            label=f"{game_id} feature object",
        )
        _read_regular(
            object_path,
            label=f"{game_id} feature object",
            max_bytes=256 * 1024 * 1024,
            expected_sha256=object_sha256,
            expected_length=descriptor.get("byte_length"),
        )
        row_count = descriptor.get("row_count")
        try:
            parquet_row_count = pq.ParquetFile(object_path).metadata.num_rows
        except Exception as error:
            raise DevelopmentArtifactError(
                f"{game_id} feature Parquet is unreadable"
            ) from error
        if (
            manifest.get("schema")
            != "nfl_expansion_development_single_game_feature_manifest_v1"
            or manifest.get("game_id") != game_id
            or manifest.get("cohort") != "development"
            or manifest.get("market_data_read") is not False
            or manifest.get("reaction_access") != "CLOSED"
            or manifest.get("publication_gate") != "PASS"
            or manifest.get("bundle_sha256") != entry.get("bundle_sha256")
            or object_sha256 != entry.get("object_sha256")
            or row_count != entry.get("row_count")
            or type(row_count) is not int
            or row_count <= 0
            or row_count != parquet_row_count
        ):
            raise DevelopmentArtifactError(f"{game_id} feature manifest mismatch")
        aggregate_rows += row_count
        verified[game_id] = VerifiedFeatureGameRef(
            game_id=game_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            object_path=object_path,
            object_sha256=object_sha256,
            rows_sha256=rows_sha256,
            row_count=row_count,
            manifest_document=MappingProxyType(manifest),
        )
    if (
        frozenset(verified) != access.development_game_ids
        or aggregate_rows != batch.get("aggregate_feature_row_count")
    ):
        raise DevelopmentArtifactError(
            "feature batch does not cover exact development cohort"
        )
    return batch_path, batch, MappingProxyType(dict(sorted(verified.items())))


def _verify_factor_registry(
    *,
    project_root: Path,
    registry_path: str | Path,
) -> tuple[Path, dict[str, Any], tuple[Mapping[str, Any], ...]]:
    registry_path = _resolve_under(
        project_root, registry_path, label="factor registry"
    )
    registry, _ = _read_json_object(
        registry_path,
        label="factor registry",
        max_bytes=4 * 1024 * 1024,
        expected_sha256=EXPECTED_FACTOR_REGISTRY_SHA256,
    )
    definitions = registry.get("factor_definitions")
    formal = (
        tuple(
            MappingProxyType(definition)
            for definition in definitions
            if type(definition) is dict
            and definition.get("analysis_role") == "FORMAL_ANALYSIS"
        )
        if type(definitions) is list
        else ()
    )
    factor_ids = {
        definition.get("factor_id")
        for definition in definitions or ()
        if type(definition) is dict
    }
    if (
        registry.get("schema") != "nfl_factor_registry_v2"
        or registry.get("status") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or registry.get("horizons_seconds")
        != list(EXPECTED_HORIZONS_SECONDS)
        or type(definitions) is not list
        or len(definitions) != EXPECTED_FACTOR_COUNT
        or len(factor_ids) != EXPECTED_FACTOR_COUNT
        or len(formal) != EXPECTED_FORMAL_FACTOR_COUNT
        or any(
            definition.get("horizons_seconds")
            != list(EXPECTED_HORIZONS_SECONDS)
            for definition in definitions
            if type(definition) is dict
        )
    ):
        raise DevelopmentArtifactError("factor registry contract mismatch")
    return registry_path, registry, formal


def verify_development_inputs(
    *,
    project_root: Path,
    raw_root: Path | None = None,
    unlock_object_path: str | Path = DEFAULT_UNLOCK_OBJECT,
    unlock_manifest_path: str | Path = DEFAULT_UNLOCK_MANIFEST,
    capture_batch_path: str | Path = DEFAULT_CAPTURE_BATCH,
    feature_batch_path: str | Path = DEFAULT_FEATURE_BATCH,
    factor_registry_path: str | Path = DEFAULT_FACTOR_REGISTRY,
) -> VerifiedDevelopmentInputs:
    """Verify every frozen input and return development-only references."""

    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise DevelopmentArtifactError("project_root is not a directory")
    raw_root = (
        _default_raw_root(project_root)
        if raw_root is None
        else Path(raw_root).resolve()
    )
    if not raw_root.is_dir():
        raise DevelopmentArtifactError(f"raw_root is not a directory: {raw_root}")
    unlock = _verify_unlock(
        project_root=project_root,
        object_path=unlock_object_path,
        manifest_path=unlock_manifest_path,
    )
    capture_path, capture_batch, capture_games = _verify_capture(
        project_root=project_root,
        batch_path=capture_batch_path,
        access=unlock,
    )
    feature_path, feature_batch, feature_games = _verify_features(
        project_root=project_root,
        batch_path=feature_batch_path,
        access=unlock,
    )
    registry_path, registry, formal = _verify_factor_registry(
        project_root=project_root,
        registry_path=factor_registry_path,
    )
    return VerifiedDevelopmentInputs(
        project_root=project_root,
        raw_root=raw_root,
        unlock=unlock,
        capture_batch_path=capture_path,
        capture_batch_sha256=EXPECTED_CAPTURE_BATCH_SHA256,
        capture_batch_document=MappingProxyType(capture_batch),
        development_capture_games=capture_games,
        feature_batch_path=feature_path,
        feature_batch_file_sha256=EXPECTED_FEATURE_BATCH_FILE_SHA256,
        feature_batch_sha256=EXPECTED_FEATURE_BATCH_SHA256,
        feature_batch_document=MappingProxyType(feature_batch),
        development_feature_games=feature_games,
        factor_registry_path=registry_path,
        factor_registry_sha256=EXPECTED_FACTOR_REGISTRY_SHA256,
        factor_registry_document=MappingProxyType(registry),
        formal_factor_definitions=formal,
    )


def load_development_capture_game(
    inputs: VerifiedDevelopmentInputs,
    game_id: str,
) -> VerifiedCaptureGameRef:
    """Return one verified capture reference after enforcing the split."""

    inputs.require_development(game_id)
    try:
        return inputs.development_capture_games[game_id]
    except KeyError as error:
        raise DevelopmentArtifactError(
            f"verified capture reference missing: {game_id}"
        ) from error


def load_development_capture_raw_documents(
    inputs: VerifiedDevelopmentInputs,
    game_id: str,
    *,
    source_url: str | None = None,
) -> tuple[VerifiedRawDocument, ...]:
    """Load byte-exact raw responses and manifest/request lineage for one game."""

    capture = load_development_capture_game(inputs, game_id)
    refs = capture.index_document.get("raw_manifests")
    if type(refs) is not list:
        raise DevelopmentArtifactError(f"{game_id} raw refs are missing")
    documents: list[VerifiedRawDocument] = []
    for ordinal, ref in enumerate(refs):
        if type(ref) is not dict:
            raise DevelopmentArtifactError(f"{game_id} raw ref is invalid")
        manifest_path = _resolve_under(
            inputs.raw_root,
            ref.get("manifest_path", ""),
            label=f"{game_id} raw manifest {ordinal}",
        )
        manifest, _ = _read_json_object(
            manifest_path,
            label=f"{game_id} raw manifest {ordinal}",
            max_bytes=4 * 1024 * 1024,
        )
        try:
            logical_manifest_sha256 = (
                contracts.static_dataset_manifest_sha256(manifest)
            )
        except (TypeError, ValueError) as error:
            raise DevelopmentArtifactError(
                f"{game_id} raw manifest is not canonical"
            ) from error
        manifest_source_url = manifest.get("source_url")
        if type(manifest_source_url) is not str:
            raise DevelopmentArtifactError(
                f"{game_id} raw manifest source_url is invalid"
            )
        if source_url is not None and manifest_source_url != source_url:
            continue
        if (
            manifest.get("manifest_sha256") != ref.get("manifest_sha256")
            or logical_manifest_sha256 != ref.get("manifest_sha256")
            or manifest.get("object_sha256") != ref.get("object_sha256")
            or manifest.get("native_object_path") != ref.get("object_path")
            or manifest.get("byte_length") != ref.get("byte_length")
        ):
            raise DevelopmentArtifactError(
                f"{game_id} raw manifest lineage mismatch"
            )
        object_path = _resolve_under(
            inputs.raw_root,
            ref.get("object_path", ""),
            label=f"{game_id} raw object {ordinal}",
        )
        object_bytes = _read_regular(
            object_path,
            label=f"{game_id} raw object {ordinal}",
            max_bytes=64 * 1024 * 1024,
            expected_sha256=ref.get("object_sha256"),
            expected_length=ref.get("byte_length"),
        )
        value = strict_load_json_bytes(
            object_bytes, label=f"{game_id} raw object {ordinal}"
        )
        documents.append(
            VerifiedRawDocument(
                game_id=game_id,
                source_url=manifest_source_url,
                request_sha256=_require_sha256(
                    ref.get("request_sha256"),
                    field=f"{game_id}.raw_request_sha256",
                ),
                manifest_path=manifest_path,
                manifest_sha256=ref["manifest_sha256"],
                object_path=object_path,
                object_sha256=ref["object_sha256"],
                object_bytes=object_bytes,
                manifest_document=MappingProxyType(manifest),
                value=value,
            )
        )
    if source_url is not None and not documents:
        raise DevelopmentArtifactError(
            f"{game_id} has no raw documents for source_url={source_url}"
        )
    return tuple(documents)


def load_development_feature_game(
    inputs: VerifiedDevelopmentInputs,
    game_id: str,
    *,
    columns: Sequence[str] | None = None,
) -> tuple[VerifiedFeatureGameRef, pd.DataFrame]:
    """Load one verified sports feature Parquet without opening holdout data."""

    inputs.require_development(game_id)
    try:
        reference = inputs.development_feature_games[game_id]
    except KeyError as error:
        raise DevelopmentArtifactError(
            f"verified feature reference missing: {game_id}"
        ) from error
    try:
        frame = pd.read_parquet(
            reference.object_path,
            columns=None if columns is None else list(columns),
        )
    except Exception as error:
        raise DevelopmentArtifactError(
            f"feature Parquet load failed: {game_id}"
        ) from error
    if len(frame) != reference.row_count:
        raise DevelopmentArtifactError(f"feature row count drift: {game_id}")
    if "game_id" in frame.columns and set(frame["game_id"].dropna()) != {game_id}:
        raise DevelopmentArtifactError(f"feature game identity drift: {game_id}")
    return reference, frame


def _relative_path(path: Path, root: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise DevelopmentArtifactError(f"{label} escapes publication root") from error


def _publish_immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_regular(
            path,
            label=label,
            max_bytes=max(len(payload), 1),
        )
        if existing != payload:
            raise DevelopmentArtifactError(
                f"immutable publication target differs: {path}"
            )
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_stage(
    *,
    output_root: Path,
    game_id: str,
    stage: str,
    frame: pd.DataFrame,
) -> PublishedStageRef:
    if _STAGE_RE.fullmatch(stage) is None:
        raise DevelopmentArtifactError(f"invalid stage name: {stage!r}")
    if type(frame) is not pd.DataFrame:
        raise DevelopmentArtifactError(f"{stage} must be a pandas DataFrame")
    if any(type(column) is not str for column in frame.columns):
        raise DevelopmentArtifactError(f"{stage} columns must be strings")
    if frame.columns.duplicated().any():
        raise DevelopmentArtifactError(f"{stage} has duplicate columns")
    if "game_id" in frame.columns:
        observed = set(frame["game_id"].dropna())
        if observed and observed != {game_id}:
            raise DevelopmentArtifactError(
                f"{stage} contains another game identity"
            )
    work_root = output_root / ".work" / game_id
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = work_root / f".{stage}.{uuid.uuid4().hex}.parquet"
    try:
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        object_sha256 = _sha256_file(temporary)
        digest = object_sha256.removeprefix("sha256:")
        object_path = (
            output_root
            / "single-game"
            / game_id
            / "objects"
            / "sha256"
            / digest[:2]
            / f"{digest}.parquet"
        )
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if object_path.exists():
            if (
                object_path.stat().st_size != temporary.stat().st_size
                or _sha256_file(object_path) != object_sha256
            ):
                raise DevelopmentArtifactError(
                    f"immutable stage object differs: {object_path}"
                )
        else:
            os.replace(temporary, object_path)
        try:
            metadata = pq.ParquetFile(object_path).metadata
            schema = pq.ParquetFile(object_path).schema_arrow
        except Exception as error:
            raise DevelopmentArtifactError(
                f"published stage is not readable Parquet: {stage}"
            ) from error
        if metadata.num_rows != len(frame):
            raise DevelopmentArtifactError(
                f"published stage row count differs: {stage}"
            )
        return PublishedStageRef(
            stage=stage,
            object_path=object_path,
            object_sha256=object_sha256,
            byte_length=object_path.stat().st_size,
            row_count=len(frame),
            column_count=len(frame.columns),
            schema_fingerprint=_sha256_bytes(schema.serialize().to_pybytes()),
        )
    finally:
        temporary.unlink(missing_ok=True)


def publish_single_game_stages(
    inputs: VerifiedDevelopmentInputs,
    *,
    output_root: Path,
    game_id: str,
    stages: Mapping[str, pd.DataFrame],
    input_fingerprint: str,
    builder_version: str,
    source_bindings: Mapping[str, object],
    claim_boundary: str,
) -> PublishedSingleGameBundle:
    """Publish one immutable development game bundle and resumable checkpoint."""

    inputs.require_development(game_id)
    input_fingerprint = _require_sha256(
        input_fingerprint, field="input_fingerprint"
    )
    if (
        type(builder_version) is not str
        or not builder_version
        or type(claim_boundary) is not str
        or not claim_boundary
        or not isinstance(source_bindings, Mapping)
        or frozenset(stages) != EXPECTED_SINGLE_GAME_STAGES
    ):
        raise DevelopmentArtifactError(
            "single-game publication requires the exact eight registered stages"
        )
    output_root = _resolve_under(
        inputs.project_root, output_root, label="development output_root"
    )
    published = {
        stage: _publish_stage(
            output_root=output_root,
            game_id=game_id,
            stage=stage,
            frame=stages[stage],
        )
        for stage in sorted(stages)
    }
    stage_material = {
        stage: {
            "object_path": _relative_path(
                ref.object_path, output_root, label=f"{stage} object"
            ),
            "object_sha256": ref.object_sha256,
            "byte_length": ref.byte_length,
            "row_count": ref.row_count,
            "column_count": ref.column_count,
            "schema_fingerprint": ref.schema_fingerprint,
            "format": "parquet",
        }
        for stage, ref in published.items()
    }
    material = {
        "schema": "nfl_expansion_development_market_single_game_manifest_v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_id": game_id,
        "input_fingerprint": input_fingerprint,
        "builder_version": builder_version,
        "source_bindings": dict(source_bindings),
        "stages": stage_material,
        "claim_boundary": claim_boundary,
        "publication_gate": "PASS",
    }
    bundle_sha256 = _sha256_bytes(_canonical_bytes(material))
    manifest = dict(material)
    manifest["bundle_sha256"] = bundle_sha256
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    manifest_digest = manifest_sha256.removeprefix("sha256:")
    manifest_path = (
        output_root
        / "single-game"
        / game_id
        / "manifests"
        / "sha256"
        / manifest_digest[:2]
        / f"{manifest_digest}.manifest.json"
    )
    _publish_immutable_bytes(
        manifest_path,
        manifest_bytes,
        label=f"{game_id} single-game manifest",
    )
    checkpoint = {
        "schema": "nfl_expansion_development_market_checkpoint_v1",
        "game_id": game_id,
        "input_fingerprint": input_fingerprint,
        "bundle_sha256": bundle_sha256,
        "manifest_path": _relative_path(
            manifest_path, output_root, label="single-game manifest"
        ),
        "manifest_sha256": manifest_sha256,
        "stage_count": len(published),
    }
    checkpoint_bytes = _canonical_bytes(checkpoint)
    checkpoint_path = output_root / "checkpoints" / f"{game_id}.json"
    _atomic_checkpoint(checkpoint_path, checkpoint_bytes)
    return PublishedSingleGameBundle(
        game_id=game_id,
        input_fingerprint=input_fingerprint,
        bundle_sha256=bundle_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
        stages=MappingProxyType(published),
    )


def load_single_game_checkpoint(
    inputs: VerifiedDevelopmentInputs,
    *,
    output_root: Path,
    game_id: str,
    expected_input_fingerprint: str | None = None,
) -> PublishedSingleGameBundle | None:
    """Verify a resumable checkpoint, its manifest, and every stage object."""

    inputs.require_development(game_id)
    if expected_input_fingerprint is not None:
        expected_input_fingerprint = _require_sha256(
            expected_input_fingerprint, field="expected_input_fingerprint"
        )
    output_root = _resolve_under(
        inputs.project_root, output_root, label="development output_root"
    )
    checkpoint_path = output_root / "checkpoints" / f"{game_id}.json"
    if not checkpoint_path.exists():
        return None
    checkpoint, checkpoint_bytes = _read_json_object(
        checkpoint_path,
        label=f"{game_id} development checkpoint",
        max_bytes=1024 * 1024,
    )
    if (
        checkpoint.get("schema")
        != "nfl_expansion_development_market_checkpoint_v1"
        or checkpoint.get("game_id") != game_id
    ):
        raise DevelopmentArtifactError(f"{game_id} checkpoint contract mismatch")
    input_fingerprint = _require_sha256(
        checkpoint.get("input_fingerprint"),
        field=f"{game_id}.input_fingerprint",
    )
    if (
        expected_input_fingerprint is not None
        and input_fingerprint != expected_input_fingerprint
    ):
        return None
    manifest_sha256 = _require_sha256(
        checkpoint.get("manifest_sha256"),
        field=f"{game_id}.manifest_sha256",
    )
    manifest_path = _resolve_under(
        output_root,
        checkpoint.get("manifest_path", ""),
        label=f"{game_id} development manifest",
    )
    manifest, _ = _read_json_object(
        manifest_path,
        label=f"{game_id} development manifest",
        max_bytes=8 * 1024 * 1024,
        expected_sha256=manifest_sha256,
    )
    stages = manifest.get("stages")
    if (
        manifest.get("schema")
        != "nfl_expansion_development_market_single_game_manifest_v1"
        or manifest.get("game_id") != game_id
        or manifest.get("cohort") != "development"
        or manifest.get("input_fingerprint") != input_fingerprint
        or manifest.get("bundle_sha256") != checkpoint.get("bundle_sha256")
        or manifest.get("publication_gate") != "PASS"
        or type(stages) is not dict
        or len(stages) != checkpoint.get("stage_count")
    ):
        raise DevelopmentArtifactError(f"{game_id} development manifest mismatch")
    manifest_material = dict(manifest)
    observed_bundle_sha256 = manifest_material.pop("bundle_sha256", None)
    if (
        _require_sha256(
            observed_bundle_sha256, field=f"{game_id}.bundle_sha256"
        )
        != _sha256_bytes(_canonical_bytes(manifest_material))
    ):
        raise DevelopmentArtifactError(f"{game_id} bundle hash mismatch")
    published: dict[str, PublishedStageRef] = {}
    for stage, descriptor in sorted(stages.items()):
        if _STAGE_RE.fullmatch(stage) is None or type(descriptor) is not dict:
            raise DevelopmentArtifactError(f"{game_id} stage descriptor invalid")
        object_sha256 = _require_sha256(
            descriptor.get("object_sha256"),
            field=f"{game_id}.{stage}.object_sha256",
        )
        object_path = _resolve_under(
            output_root,
            descriptor.get("object_path", ""),
            label=f"{game_id}.{stage} object",
        )
        byte_length = descriptor.get("byte_length")
        if type(byte_length) is not int:
            raise DevelopmentArtifactError(
                f"{game_id}.{stage} byte_length is invalid"
            )
        _verify_regular_file_streaming(
            object_path,
            label=f"{game_id}.{stage} object",
            expected_sha256=object_sha256,
            expected_length=byte_length,
        )
        try:
            metadata = pq.ParquetFile(object_path).metadata
            schema = pq.ParquetFile(object_path).schema_arrow
        except Exception as error:
            raise DevelopmentArtifactError(
                f"{game_id}.{stage} Parquet is unreadable"
            ) from error
        if (
            metadata.num_rows != descriptor.get("row_count")
            or len(schema) != descriptor.get("column_count")
            or _sha256_bytes(schema.serialize().to_pybytes())
            != descriptor.get("schema_fingerprint")
        ):
            raise DevelopmentArtifactError(
                f"{game_id}.{stage} descriptor differs from Parquet"
            )
        published[stage] = PublishedStageRef(
            stage=stage,
            object_path=object_path,
            object_sha256=object_sha256,
            byte_length=int(descriptor["byte_length"]),
            row_count=int(descriptor["row_count"]),
            column_count=int(descriptor["column_count"]),
            schema_fingerprint=str(descriptor["schema_fingerprint"]),
        )
    return PublishedSingleGameBundle(
        game_id=game_id,
        input_fingerprint=input_fingerprint,
        bundle_sha256=str(checkpoint["bundle_sha256"]),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
        stages=MappingProxyType(published),
    )


def _publish_cross_game_results(
    *,
    output_root: Path,
    frame: pd.DataFrame,
    expected_game_ids: frozenset[str],
) -> PublishedStageRef:
    stage = "single_game_results"
    if (
        type(frame) is not pd.DataFrame
        or "game_id" not in frame.columns
        or set(frame["game_id"].dropna()) != expected_game_ids
        or any(type(column) is not str for column in frame.columns)
        or frame.columns.duplicated().any()
    ):
        raise DevelopmentArtifactError(
            "cross-game results must cover the exact development cohort"
        )
    work_root = output_root / ".work" / "cross-game"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = work_root / f".{stage}.{uuid.uuid4().hex}.parquet"
    try:
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        object_sha256 = _sha256_file(temporary)
        digest = object_sha256.removeprefix("sha256:")
        object_path = (
            output_root
            / "cross-game"
            / "objects"
            / "sha256"
            / digest[:2]
            / f"{digest}.parquet"
        )
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if object_path.exists():
            if (
                object_path.stat().st_size != temporary.stat().st_size
                or _sha256_file(object_path) != object_sha256
            ):
                raise DevelopmentArtifactError(
                    f"immutable cross-game object differs: {object_path}"
                )
        else:
            os.replace(temporary, object_path)
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != len(frame):
            raise DevelopmentArtifactError(
                "cross-game result row count differs"
            )
        schema = parquet.schema_arrow
        return PublishedStageRef(
            stage=stage,
            object_path=object_path,
            object_sha256=object_sha256,
            byte_length=object_path.stat().st_size,
            row_count=len(frame),
            column_count=len(frame.columns),
            schema_fingerprint=_sha256_bytes(schema.serialize().to_pybytes()),
        )
    finally:
        temporary.unlink(missing_ok=True)


def publish_development_batch_index(
    inputs: VerifiedDevelopmentInputs,
    *,
    output_root: Path,
    games: Sequence[PublishedSingleGameBundle],
    input_fingerprint: str,
    builder_version: str,
    source_bindings: Mapping[str, object],
    claim_boundary: str,
) -> PublishedDevelopmentBatch:
    """Atomically publish an index only after all 153 games verify."""

    input_fingerprint = _require_sha256(
        input_fingerprint, field="batch.input_fingerprint"
    )
    output_root = _resolve_under(
        inputs.project_root, output_root, label="development output_root"
    )
    by_game = {game.game_id: game for game in games}
    if (
        len(games) != EXPECTED_DEVELOPMENT_COUNT
        or len(by_game) != EXPECTED_DEVELOPMENT_COUNT
        or frozenset(by_game) != inputs.unlock.development_game_ids
    ):
        raise DevelopmentArtifactError(
            "batch publication requires exact verified 153-game cohort"
        )
    verified: list[PublishedSingleGameBundle] = []
    for game_id in sorted(by_game):
        resumed = load_single_game_checkpoint(
            inputs,
            output_root=output_root,
            game_id=game_id,
        )
        if (
            resumed is None
            or resumed.bundle_sha256 != by_game[game_id].bundle_sha256
            or resumed.manifest_sha256 != by_game[game_id].manifest_sha256
        ):
            raise DevelopmentArtifactError(
                f"batch game checkpoint does not verify: {game_id}"
            )
        verified.append(resumed)
    result_frames: list[pd.DataFrame] = []
    for game in verified:
        result_ref = game.stages.get("single_game_results")
        if result_ref is None:
            raise DevelopmentArtifactError(
                f"single_game_results stage missing: {game.game_id}"
            )
        try:
            result_frame = pd.read_parquet(result_ref.object_path)
        except Exception as error:
            raise DevelopmentArtifactError(
                f"single_game_results is unreadable: {game.game_id}"
            ) from error
        if (
            "game_id" not in result_frame.columns
            or set(result_frame["game_id"].dropna()) != {game.game_id}
        ):
            raise DevelopmentArtifactError(
                f"single_game_results identity differs: {game.game_id}"
            )
        result_frames.append(result_frame)
    merged_results = pd.concat(result_frames, ignore_index=True)
    merged_ref = _publish_cross_game_results(
        output_root=output_root,
        frame=merged_results,
        expected_game_ids=inputs.unlock.development_game_ids,
    )
    game_rows = [
        {
            "game_id": game.game_id,
            "input_fingerprint": game.input_fingerprint,
            "bundle_sha256": game.bundle_sha256,
            "manifest_path": _relative_path(
                game.manifest_path,
                output_root,
                label=f"{game.game_id} manifest",
            ),
            "manifest_sha256": game.manifest_sha256,
            "checkpoint_sha256": game.checkpoint_sha256,
            "stage_count": len(game.stages),
        }
        for game in verified
    ]
    material = {
        "schema": "nfl_expansion_development_market_batch_index_v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": EXPECTED_DEVELOPMENT_COUNT,
        "input_fingerprint": input_fingerprint,
        "builder_version": builder_version,
        "source_bindings": dict(source_bindings),
        "games": game_rows,
        "merged_single_game_results": {
            "object_path": _relative_path(
                merged_ref.object_path,
                output_root,
                label="merged single_game_results",
            ),
            "object_sha256": merged_ref.object_sha256,
            "byte_length": merged_ref.byte_length,
            "row_count": merged_ref.row_count,
            "column_count": merged_ref.column_count,
            "schema_fingerprint": merged_ref.schema_fingerprint,
            "format": "parquet",
        },
        "claim_boundary": claim_boundary,
        "publication_gate": "PASS",
        "final_holdout_access": "CLOSED",
    }
    batch_sha256 = _sha256_bytes(_canonical_bytes(material))
    index = dict(material)
    index["batch_sha256"] = batch_sha256
    index_bytes = _canonical_bytes(index)
    index_sha256 = _sha256_bytes(index_bytes)
    digest = index_sha256.removeprefix("sha256:")
    index_path = (
        output_root
        / "batches"
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.batch-index.json"
    )
    _publish_immutable_bytes(
        index_path,
        index_bytes,
        label="development market batch index",
    )
    return PublishedDevelopmentBatch(
        input_fingerprint=input_fingerprint,
        batch_sha256=batch_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
        game_count=EXPECTED_DEVELOPMENT_COUNT,
        merged_single_game_results=merged_ref,
        games=tuple(verified),
    )


def load_development_batch_index(
    inputs: VerifiedDevelopmentInputs,
    *,
    output_root: Path,
    index_path: Path,
) -> PublishedDevelopmentBatch:
    """Fully verify and reopen an already-published exact-153 batch."""

    output_root = _resolve_under(
        inputs.project_root, output_root, label="development output_root"
    )
    index_path = _resolve_under(
        output_root, index_path, label="development batch index"
    )
    match = re.fullmatch(r"([0-9a-f]{64})\.batch-index\.json", index_path.name)
    if match is None or index_path.parent.name != match.group(1)[:2]:
        raise DevelopmentArtifactError(
            "batch index path is not content-addressed"
        )
    index_sha256 = "sha256:" + match.group(1)
    index, _ = _read_json_object(
        index_path,
        label="development batch index",
        max_bytes=16 * 1024 * 1024,
        expected_sha256=index_sha256,
    )
    batch_sha256 = _require_sha256(
        index.get("batch_sha256"), field="batch.batch_sha256"
    )
    material = dict(index)
    material.pop("batch_sha256")
    if (
        index.get("schema")
        != "nfl_expansion_development_market_batch_index_v1"
        or index.get("experiment_id") != "X-13"
        or index.get("cohort") != "development"
        or index.get("game_count") != EXPECTED_DEVELOPMENT_COUNT
        or index.get("publication_gate") != "PASS"
        or index.get("final_holdout_access") != "CLOSED"
        or batch_sha256 != _sha256_bytes(_canonical_bytes(material))
    ):
        raise DevelopmentArtifactError("development batch index contract mismatch")
    input_fingerprint = _require_sha256(
        index.get("input_fingerprint"), field="batch.input_fingerprint"
    )
    game_rows = index.get("games")
    if type(game_rows) is not list or len(game_rows) != EXPECTED_DEVELOPMENT_COUNT:
        raise DevelopmentArtifactError(
            "development batch game references are incomplete"
        )
    verified: list[PublishedSingleGameBundle] = []
    observed_ids: set[str] = set()
    for row in game_rows:
        if type(row) is not dict or type(row.get("game_id")) is not str:
            raise DevelopmentArtifactError("batch game reference is malformed")
        game_id = row["game_id"]
        inputs.require_development(game_id)
        if game_id in observed_ids:
            raise DevelopmentArtifactError(f"duplicate batch game: {game_id}")
        observed_ids.add(game_id)
        game_input_fingerprint = _require_sha256(
            row.get("input_fingerprint"),
            field=f"{game_id}.input_fingerprint",
        )
        reopened = load_single_game_checkpoint(
            inputs,
            output_root=output_root,
            game_id=game_id,
            expected_input_fingerprint=game_input_fingerprint,
        )
        if (
            reopened is None
            or reopened.bundle_sha256 != row.get("bundle_sha256")
            or reopened.manifest_sha256 != row.get("manifest_sha256")
            or reopened.checkpoint_sha256 != row.get("checkpoint_sha256")
            or len(reopened.stages) != row.get("stage_count")
            or _relative_path(
                reopened.manifest_path,
                output_root,
                label=f"{game_id} manifest",
            )
            != row.get("manifest_path")
        ):
            raise DevelopmentArtifactError(
                f"batch game reference does not verify: {game_id}"
            )
        verified.append(reopened)
    if observed_ids != inputs.unlock.development_game_ids:
        raise DevelopmentArtifactError(
            "batch game references do not cover exact development cohort"
        )

    descriptor = index.get("merged_single_game_results")
    if type(descriptor) is not dict:
        raise DevelopmentArtifactError(
            "merged single_game_results descriptor is missing"
        )
    merged_sha256 = _require_sha256(
        descriptor.get("object_sha256"),
        field="merged_single_game_results.object_sha256",
    )
    merged_path = _resolve_under(
        output_root,
        descriptor.get("object_path", ""),
        label="merged single_game_results object",
    )
    byte_length = descriptor.get("byte_length")
    if type(byte_length) is not int:
        raise DevelopmentArtifactError(
            "merged single_game_results byte_length is invalid"
        )
    _verify_regular_file_streaming(
        merged_path,
        label="merged single_game_results object",
        expected_sha256=merged_sha256,
        expected_length=byte_length,
    )
    try:
        parquet = pq.ParquetFile(merged_path)
        merged = pd.read_parquet(merged_path)
    except Exception as error:
        raise DevelopmentArtifactError(
            "merged single_game_results is unreadable"
        ) from error
    schema_fingerprint = _sha256_bytes(
        parquet.schema_arrow.serialize().to_pybytes()
    )
    if (
        descriptor.get("format") != "parquet"
        or parquet.metadata.num_rows != descriptor.get("row_count")
        or len(parquet.schema_arrow) != descriptor.get("column_count")
        or schema_fingerprint != descriptor.get("schema_fingerprint")
        or "game_id" not in merged.columns
        or set(merged["game_id"].dropna())
        != inputs.unlock.development_game_ids
    ):
        raise DevelopmentArtifactError(
            "merged single_game_results descriptor differs from object"
        )
    expected_frames: list[pd.DataFrame] = []
    for game in verified:
        stage = game.stages.get("single_game_results")
        if stage is None:
            raise DevelopmentArtifactError(
                f"single_game_results stage missing: {game.game_id}"
            )
        expected_frames.append(pd.read_parquet(stage.object_path))
    expected_merged = pd.concat(expected_frames, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            merged.reset_index(drop=True),
            expected_merged.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise DevelopmentArtifactError(
            "merged single_game_results differs from verified game stages"
        ) from error
    merged_ref = PublishedStageRef(
        stage="single_game_results",
        object_path=merged_path,
        object_sha256=merged_sha256,
        byte_length=byte_length,
        row_count=int(descriptor["row_count"]),
        column_count=int(descriptor["column_count"]),
        schema_fingerprint=schema_fingerprint,
    )
    return PublishedDevelopmentBatch(
        input_fingerprint=input_fingerprint,
        batch_sha256=batch_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
        game_count=EXPECTED_DEVELOPMENT_COUNT,
        merged_single_game_results=merged_ref,
        games=tuple(verified),
    )


def publish_cross_game_discovery(
    inputs: VerifiedDevelopmentInputs,
    *,
    output_root: Path,
    batch: PublishedDevelopmentBatch,
    discovery: pd.DataFrame,
    expected_factor_horizon_grid: Sequence[tuple[str, int]],
    builder_version: str,
    source_bindings: Mapping[str, object],
    claim_boundary: str,
) -> PublishedCrossGameDiscovery:
    """Publish the frozen delay-zero discovery table and immutable lineage."""

    output_root = _resolve_under(
        inputs.project_root, output_root, label="development output_root"
    )
    if (
        batch.game_count != EXPECTED_DEVELOPMENT_COUNT
        or len(batch.games) != EXPECTED_DEVELOPMENT_COUNT
        or {game.game_id for game in batch.games}
        != inputs.unlock.development_game_ids
    ):
        raise DevelopmentArtifactError(
            "discovery requires the exact verified 153-game batch"
        )
    batch_index, _ = _read_json_object(
        batch.index_path,
        label="development batch index",
        max_bytes=16 * 1024 * 1024,
        expected_sha256=batch.index_sha256,
    )
    if (
        batch_index.get("schema")
        != "nfl_expansion_development_market_batch_index_v1"
        or batch_index.get("batch_sha256") != batch.batch_sha256
        or batch_index.get("game_count") != EXPECTED_DEVELOPMENT_COUNT
        or batch_index.get("publication_gate") != "PASS"
        or batch_index.get("final_holdout_access") != "CLOSED"
    ):
        raise DevelopmentArtifactError(
            "discovery batch index binding does not verify"
        )
    required_columns = {
        "factor_id",
        "horizon_seconds",
        "delay_seconds",
    }
    if (
        type(discovery) is not pd.DataFrame
        or discovery.empty
        or not required_columns.issubset(discovery.columns)
        or any(type(column) is not str for column in discovery.columns)
        or discovery.columns.duplicated().any()
    ):
        raise DevelopmentArtifactError(
            "discovery table does not expose its registered grid"
        )
    grid_columns = ["factor_id", "horizon_seconds", "delay_seconds"]
    if discovery.duplicated(grid_columns).any():
        raise DevelopmentArtifactError(
            "discovery factor/horizon/delay grain is duplicated"
        )
    formal_factor_ids = {
        str(definition["factor_id"])
        for definition in inputs.formal_factor_definitions
    }
    supplied_grid: list[tuple[str, int]] = []
    for row in expected_factor_horizon_grid:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            or row[0] not in formal_factor_ids
            or row[1] not in EXPECTED_HORIZONS_SECONDS
        ):
            raise DevelopmentArtifactError(
                "expected discovery grid is not registered"
            )
        supplied_grid.append(row)
    expected_grid = frozenset(supplied_grid)
    if not expected_grid or len(expected_grid) != len(supplied_grid):
        raise DevelopmentArtifactError(
            "expected discovery grid is empty or duplicated"
        )
    observed_grid_rows: list[tuple[str, int]] = []
    for row in discovery.itertuples():
        if (
            type(row.factor_id) is not str
            or not isinstance(row.horizon_seconds, Integral)
            or isinstance(row.horizon_seconds, bool)
            or not isinstance(row.delay_seconds, Integral)
            or isinstance(row.delay_seconds, bool)
            or int(row.delay_seconds) != DISCOVERY_DELAY_SECONDS
        ):
            raise DevelopmentArtifactError(
                "discovery grid contains invalid values"
            )
        observed_grid_rows.append(
            (row.factor_id, int(row.horizon_seconds))
        )
    observed_grid = frozenset(observed_grid_rows)
    if (
        observed_grid != expected_grid
        or len(observed_grid_rows) != len(observed_grid)
    ):
        raise DevelopmentArtifactError(
            "discovery table differs from the supplied registered grid"
        )
    if (
        type(builder_version) is not str
        or not builder_version
        or not isinstance(source_bindings, Mapping)
        or type(claim_boundary) is not str
        or not claim_boundary
    ):
        raise DevelopmentArtifactError("discovery publication metadata invalid")

    work_root = output_root / ".work" / "cross-game-discovery"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = work_root / f".discovery.{uuid.uuid4().hex}.parquet"
    try:
        discovery.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        object_sha256 = _sha256_file(temporary)
        object_digest = object_sha256.removeprefix("sha256:")
        object_path = (
            output_root
            / "cross-game"
            / "discovery"
            / "objects"
            / "sha256"
            / object_digest[:2]
            / f"{object_digest}.parquet"
        )
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if object_path.exists():
            if (
                object_path.stat().st_size != temporary.stat().st_size
                or _sha256_file(object_path) != object_sha256
            ):
                raise DevelopmentArtifactError(
                    f"immutable discovery object differs: {object_path}"
                )
        else:
            os.replace(temporary, object_path)
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != len(discovery):
            raise DevelopmentArtifactError(
                "published discovery row count differs"
            )
        schema_fingerprint = _sha256_bytes(
            parquet.schema_arrow.serialize().to_pybytes()
        )
    finally:
        temporary.unlink(missing_ok=True)

    grid_material = [
        {"factor_id": factor_id, "horizon_seconds": horizon}
        for factor_id, horizon in sorted(expected_grid)
    ]
    manifest = {
        "schema": "nfl_expansion_development_cross_game_discovery_manifest_v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "batch": {
            "batch_sha256": batch.batch_sha256,
            "index_path": _relative_path(
                batch.index_path,
                output_root,
                label="development batch index",
            ),
            "index_sha256": batch.index_sha256,
            "game_count": EXPECTED_DEVELOPMENT_COUNT,
        },
        "discovery_parameters": {
            "delay_seconds": DISCOVERY_DELAY_SECONDS,
            "bootstrap_seed": DISCOVERY_BOOTSTRAP_SEED,
            "bootstrap_samples": DISCOVERY_BOOTSTRAP_SAMPLES,
        },
        "registered_grid": {
            "factor_horizon_count": len(grid_material),
            "factor_horizons": grid_material,
            "sha256": _sha256_bytes(_canonical_bytes(grid_material)),
        },
        "discovery": {
            "object_path": _relative_path(
                object_path,
                output_root,
                label="cross-game discovery object",
            ),
            "object_sha256": object_sha256,
            "byte_length": object_path.stat().st_size,
            "row_count": len(discovery),
            "column_count": len(discovery.columns),
            "schema_fingerprint": schema_fingerprint,
            "grain": "one registered factor_id x horizon_seconds x delay_seconds",
            "format": "parquet",
        },
        "builder_version": builder_version,
        "source_bindings": dict(source_bindings),
        "claim_boundary": claim_boundary,
        "publication_gate": "PASS",
        "final_holdout_access": "CLOSED",
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    manifest_digest = manifest_sha256.removeprefix("sha256:")
    manifest_path = (
        output_root
        / "cross-game"
        / "discovery"
        / "manifests"
        / "sha256"
        / manifest_digest[:2]
        / f"{manifest_digest}.manifest.json"
    )
    _publish_immutable_bytes(
        manifest_path,
        manifest_bytes,
        label="cross-game discovery manifest",
    )
    return PublishedCrossGameDiscovery(
        batch_sha256=batch.batch_sha256,
        object_path=object_path,
        object_sha256=object_sha256,
        row_count=len(discovery),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "DISCOVERY_BOOTSTRAP_SAMPLES",
    "DISCOVERY_BOOTSTRAP_SEED",
    "DISCOVERY_DELAY_SECONDS",
    "DevelopmentArtifactError",
    "EXPECTED_SINGLE_GAME_STAGES",
    "PublishedCrossGameDiscovery",
    "PublishedDevelopmentBatch",
    "PublishedSingleGameBundle",
    "PublishedStageRef",
    "VerifiedCaptureGameRef",
    "VerifiedDevelopmentInputs",
    "VerifiedFeatureGameRef",
    "VerifiedRawDocument",
    "VerifiedUnlockAccess",
    "load_development_capture_game",
    "load_development_capture_raw_documents",
    "load_development_feature_game",
    "load_development_batch_index",
    "load_single_game_checkpoint",
    "publish_development_batch_index",
    "publish_cross_game_discovery",
    "publish_single_game_stages",
    "strict_load_json_bytes",
    "verify_development_inputs",
]
