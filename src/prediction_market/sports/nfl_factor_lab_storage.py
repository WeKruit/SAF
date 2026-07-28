"""Fail-closed Supabase S3 storage for the NFL X-13 Factor Lab V2.

This module is intentionally orchestration-only.  Byte identity, atomic
hydration, and remote readback are delegated to ``immutable_object_store``.
No function in this module can fetch an upstream sports or market source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from prediction_market.immutable_object_store import (
    ArtifactRefV1,
    ImmutableObjectStore,
    ImmutableObjectStoreError,
    content_addressed_key,
    hydrate_artifact,
    mirror_artifact,
    sha256_path,
)
from prediction_market.pmxt.data_store import configured_object_store
from prediction_market.sports.nfl_factor_lab_bundles import (
    bundle_artifact_refs,
    verify_cross_game_bundle,
    verify_single_game_bundle,
)
from prediction_market.sports.nfl_factor_lab_run_index import (
    VerifiedFactorLabRunIndexV2,
    verify_factor_lab_run_index,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


_BASE_PREFIX = "prediction-market"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_INVENTORY_SCHEMA = "NFLFactorLabRemoteInventoryV2"
_CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"


class FactorLabStorageError(RuntimeError):
    """Factor Lab storage identity could not be proven."""


@dataclass(frozen=True, slots=True)
class FactorLabStorageRefsV2:
    """All immutable objects referenced by one verified X-13 V2 run."""

    run_index_path: Path
    run_index_sha256: str
    s3_bucket: str
    s3_prefix: str
    single_game_refs: tuple[ArtifactRefV1, ...]
    cross_game_refs: tuple[ArtifactRefV1, ...]
    source_refs: tuple[ArtifactRefV1, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_index_path, Path)
            or _SHA256_RE.fullmatch(self.run_index_sha256) is None
        ):
            raise FactorLabStorageError("run-index identity is invalid")
        if not isinstance(self.s3_bucket, str) or not self.s3_bucket.strip():
            raise FactorLabStorageError("S3 bucket must be explicit")
        if self.s3_prefix != _BASE_PREFIX:
            raise FactorLabStorageError(
                f"S3 prefix must be {_BASE_PREFIX}"
            )
        if (
            not self.single_game_refs
            or not self.cross_game_refs
            or not self.source_refs
        ):
            raise FactorLabStorageError(
                "single-game, cross-game, and source refs are required"
            )
        seen: dict[str, tuple[str, int, str]] = {}
        for ref in self.all_refs:
            if ref.s3_bucket != self.s3_bucket:
                raise FactorLabStorageError("artifact bucket differs")
            if not ref.s3_key.startswith(f"{self.s3_prefix}/"):
                raise FactorLabStorageError("artifact prefix differs")
            identity = (ref.sha256, ref.byte_length, ref.media_type)
            prior = seen.setdefault(ref.s3_key, identity)
            if prior != identity:
                raise FactorLabStorageError(
                    "one S3 key has conflicting immutable identities"
                )

    @property
    def all_refs(self) -> tuple[ArtifactRefV1, ...]:
        return tuple(
            sorted(
                (
                    *self.single_game_refs,
                    *self.cross_game_refs,
                    *self.source_refs,
                ),
                key=lambda ref: (ref.s3_key, ref.local_relative_path),
            )
        )


@dataclass(frozen=True, slots=True)
class FactorLabStorageStatusRowV2:
    scope: str
    ref: ArtifactRefV1
    local_status: str
    remote_status: str
    state: str


@dataclass(frozen=True, slots=True)
class FactorLabStorageStatusV2:
    rows: tuple[FactorLabStorageStatusRowV2, ...]
    local_verified_count: int
    remote_verified_count: int
    conflict_count: int


@dataclass(frozen=True, slots=True)
class VerifiedFactorLabRemoteInventoryV2:
    path: Path
    inventory_ref: ArtifactRefV1
    run_index_sha256: str
    object_refs: tuple[ArtifactRefV1, ...]
    total_byte_length: int
    s3_bucket: str
    s3_prefix: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _relative_to_project(path: Path, root: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise FactorLabStorageError(
            f"{label} is outside project_root"
        ) from error


def _source_suffix_and_media_type(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if not suffix:
        return ".bin", "application/octet-stream"
    media_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".parquet": "application/x-parquet",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
    }
    return suffix, media_types.get(suffix, "application/octet-stream")


def _source_ref(
    *,
    name: str,
    path: Path,
    sha256: str,
    byte_length: int,
    project_root: Path,
    s3_bucket: str,
) -> ArtifactRefV1:
    if _SOURCE_NAME_RE.fullmatch(name) is None:
        raise FactorLabStorageError("source artifact name is invalid")
    observed_sha, observed_length = sha256_path(path)
    if observed_sha != sha256 or observed_length != byte_length:
        raise FactorLabStorageError(
            f"source artifact identity differs: {name}"
        )
    suffix, media_type = _source_suffix_and_media_type(path)
    return ArtifactRefV1(
        sha256=sha256,
        byte_length=byte_length,
        media_type=media_type,
        local_relative_path=_relative_to_project(
            path, project_root, label=f"source artifact {name}"
        ),
        s3_bucket=s3_bucket,
        s3_key=content_addressed_key(
            prefix=_BASE_PREFIX,
            namespace=f"raw/{name}",
            sha256=sha256,
            suffix=suffix,
        ),
    )


def _verified_run_index(
    value: VerifiedFactorLabRunIndexV2 | str | Path,
    *,
    project_root: Path,
) -> VerifiedFactorLabRunIndexV2:
    path = value.path if isinstance(value, VerifiedFactorLabRunIndexV2) else Path(value)
    return verify_factor_lab_run_index(path, project_root=project_root)


def build_factor_lab_storage_refs(
    run_index: VerifiedFactorLabRunIndexV2 | str | Path,
    *,
    project_root: str | Path,
) -> FactorLabStorageRefsV2:
    """Generate source/single/cross refs from one verified 20-game run."""

    root = Path(project_root)
    verified = _verified_run_index(run_index, project_root=root)
    if verified.game_ids != X13_GAME_IDS:
        raise FactorLabStorageError(
            "storage refs require the frozen X-13 20-game set"
        )
    prefix = verified.s3_prefix.strip("/")
    if prefix != _BASE_PREFIX:
        raise FactorLabStorageError(
            f"S3 prefix must be {_BASE_PREFIX}"
        )
    single_refs: list[ArtifactRefV1] = []
    for manifest in verified.single_game_manifests:
        bundle = verify_single_game_bundle(manifest)
        single_refs.extend(
            bundle_artifact_refs(
                bundle,
                project_root=root,
                s3_bucket=verified.s3_bucket,
                s3_prefix=prefix,
            )
        )
    cross_bundle = verify_cross_game_bundle(verified.cross_game_manifest)
    cross_refs = bundle_artifact_refs(
        cross_bundle,
        project_root=root,
        s3_bucket=verified.s3_bucket,
        s3_prefix=prefix,
    )
    source_refs = tuple(
        _source_ref(
            name=name,
            path=path,
            sha256=sha256,
            byte_length=byte_length,
            project_root=root,
            s3_bucket=verified.s3_bucket,
        )
        for name, path, sha256, byte_length in verified.source_artifacts
    )
    run_index_sha, _run_index_length = sha256_path(verified.path)
    return FactorLabStorageRefsV2(
        run_index_path=verified.path,
        run_index_sha256=run_index_sha,
        s3_bucket=verified.s3_bucket,
        s3_prefix=prefix,
        single_game_refs=tuple(
            sorted(
                single_refs,
                key=lambda ref: (ref.s3_key, ref.local_relative_path),
            )
        ),
        cross_game_refs=tuple(
            sorted(
                cross_refs,
                key=lambda ref: (ref.s3_key, ref.local_relative_path),
            )
        ),
        source_refs=tuple(
            sorted(
                source_refs,
                key=lambda ref: (ref.s3_key, ref.local_relative_path),
            )
        ),
    )


def _scoped_refs(
    refs: FactorLabStorageRefsV2,
) -> tuple[tuple[str, ArtifactRefV1], ...]:
    return tuple(
        sorted(
            (
                *(("single_game", ref) for ref in refs.single_game_refs),
                *(("cross_game", ref) for ref in refs.cross_game_refs),
                *(("source", ref) for ref in refs.source_refs),
            ),
            key=lambda item: (
                item[0],
                item[1].s3_key,
                item[1].local_relative_path,
            ),
        )
    )


def _local_path_for_ref(root: Path, ref: ArtifactRefV1) -> Path:
    path = root / ref.local_relative_path
    _relative_to_project(path, root, label="artifact")
    return path


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FactorLabStorageError(
                "content-addressed inventory path conflicts"
            )
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inventory_payload(refs: FactorLabStorageRefsV2) -> dict[str, object]:
    scoped = _scoped_refs(refs)
    return {
        "schema": _INVENTORY_SCHEMA,
        "experiment_id": "X-13",
        "claim_boundary": _CLAIM_BOUNDARY,
        "s3_bucket": refs.s3_bucket,
        "s3_prefix": refs.s3_prefix,
        "run_index": {
            "local_relative_path": refs.run_index_path.name,
            "sha256": refs.run_index_sha256,
        },
        "object_count": len(scoped),
        "total_byte_length": sum(ref.byte_length for _scope, ref in scoped),
        "objects": [
            {
                "scope": scope,
                "readback_verified": True,
                "ref": ref.to_dict(),
            }
            for scope, ref in scoped
        ],
    }


def mirror_factor_lab_storage(
    refs: FactorLabStorageRefsV2,
    *,
    project_root: str | Path,
    inventory_root: str | Path,
    object_store: ImmutableObjectStore,
    max_workers: int = 4,
) -> VerifiedFactorLabRemoteInventoryV2:
    """Mirror all refs with readback proof, then publish one immutable inventory."""

    if not isinstance(refs, FactorLabStorageRefsV2):
        raise FactorLabStorageError("storage refs are invalid")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise FactorLabStorageError("max_workers must be a positive integer")
    root = Path(project_root)
    run_index_sha, _run_index_length = sha256_path(refs.run_index_path)
    if run_index_sha != refs.run_index_sha256:
        raise FactorLabStorageError("run-index identity differs")

    unique_remote_refs: list[ArtifactRefV1] = []
    seen_keys: set[str] = set()
    for ref in refs.all_refs:
        if ref.s3_key not in seen_keys:
            unique_remote_refs.append(ref)
            seen_keys.add(ref.s3_key)

    def mirror_one(ref: ArtifactRefV1) -> ArtifactRefV1:
        try:
            return mirror_artifact(
                _local_path_for_ref(root, ref),
                ref=ref,
                object_store=object_store,
            )
        except ImmutableObjectStoreError as error:
            raise FactorLabStorageError(str(error)) from error

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="nfl-factor-lab-s3",
    ) as executor:
        futures = [
            executor.submit(mirror_one, ref) for ref in unique_remote_refs
        ]
        try:
            mirrored = tuple(future.result() for future in futures)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    if mirrored != tuple(unique_remote_refs):
        raise FactorLabStorageError("parallel mirror result order differs")

    payload = _canonical_json(_inventory_payload(refs))
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    inventory_base = Path(inventory_root)
    _relative_to_project(inventory_base, root, label="inventory_root")
    inventory_path = (
        inventory_base
        / f"sha256-{digest.removeprefix('sha256:')}"
        / "remote_inventory_v2.json"
    )
    _atomic_publish(inventory_path, payload)
    inventory_ref = ArtifactRefV1(
        sha256=digest,
        byte_length=len(payload),
        media_type="application/json",
        local_relative_path=_relative_to_project(
            inventory_path, root, label="remote inventory"
        ),
        s3_bucket=refs.s3_bucket,
        s3_key=content_addressed_key(
            prefix=refs.s3_prefix,
            namespace="research/nfl/x13/remote-inventory",
            sha256=digest,
            suffix=".json",
        ),
    )
    try:
        mirror_artifact(
            inventory_path,
            ref=inventory_ref,
            object_store=object_store,
        )
    except ImmutableObjectStoreError as error:
        raise FactorLabStorageError(str(error)) from error
    return verify_factor_lab_remote_inventory(
        inventory_path,
        project_root=root,
        expected_refs=refs,
    )


def _parse_inventory_objects(
    value: dict[str, object],
) -> tuple[tuple[str, ArtifactRefV1], ...]:
    raw_objects = value.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise FactorLabStorageError("remote inventory has no objects")
    parsed: list[tuple[str, ArtifactRefV1]] = []
    for item in raw_objects:
        if (
            not isinstance(item, dict)
            or item.get("scope")
            not in {"single_game", "cross_game", "source"}
            or item.get("readback_verified") is not True
        ):
            raise FactorLabStorageError(
                "remote inventory object proof is invalid"
            )
        try:
            ref = ArtifactRefV1.from_dict(item.get("ref"))
        except ImmutableObjectStoreError as error:
            raise FactorLabStorageError(str(error)) from error
        parsed.append((str(item["scope"]), ref))
    return tuple(
        sorted(
            parsed,
            key=lambda item: (
                item[0],
                item[1].s3_key,
                item[1].local_relative_path,
            ),
        )
    )


def verify_factor_lab_remote_inventory(
    path: str | Path,
    *,
    project_root: str | Path,
    expected_refs: FactorLabStorageRefsV2 | None = None,
) -> VerifiedFactorLabRemoteInventoryV2:
    """Verify canonical inventory bytes and every embedded ArtifactRefV1."""

    inventory_path = Path(path)
    try:
        raw = inventory_path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabStorageError(
            "remote inventory is unreadable"
        ) from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise FactorLabStorageError("remote inventory is not canonical")
    if (
        value.get("schema") != _INVENTORY_SCHEMA
        or value.get("experiment_id") != "X-13"
        or value.get("claim_boundary") != _CLAIM_BOUNDARY
        or value.get("s3_prefix") != _BASE_PREFIX
        or not isinstance(value.get("s3_bucket"), str)
        or not str(value["s3_bucket"]).strip()
    ):
        raise FactorLabStorageError("remote inventory governance differs")
    scoped = _parse_inventory_objects(value)
    object_refs = tuple(ref for _scope, ref in scoped)
    if value.get("object_count") != len(scoped) or value.get(
        "total_byte_length"
    ) != sum(ref.byte_length for ref in object_refs):
        raise FactorLabStorageError("remote inventory totals differ")
    run_index = value.get("run_index")
    if (
        not isinstance(run_index, dict)
        or _SHA256_RE.fullmatch(str(run_index.get("sha256"))) is None
    ):
        raise FactorLabStorageError("remote inventory run index differs")
    if expected_refs is not None:
        if (
            str(value["s3_bucket"]) != expected_refs.s3_bucket
            or value["s3_prefix"] != expected_refs.s3_prefix
            or run_index.get("sha256") != expected_refs.run_index_sha256
            or scoped != _scoped_refs(expected_refs)
        ):
            raise FactorLabStorageError(
                "remote inventory differs from expected refs"
            )
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    root = Path(project_root)
    inventory_ref = ArtifactRefV1(
        sha256=digest,
        byte_length=len(raw),
        media_type="application/json",
        local_relative_path=_relative_to_project(
            inventory_path, root, label="remote inventory"
        ),
        s3_bucket=str(value["s3_bucket"]),
        s3_key=content_addressed_key(
            prefix=_BASE_PREFIX,
            namespace="research/nfl/x13/remote-inventory",
            sha256=digest,
            suffix=".json",
        ),
    )
    return VerifiedFactorLabRemoteInventoryV2(
        path=inventory_path,
        inventory_ref=inventory_ref,
        run_index_sha256=str(run_index["sha256"]),
        object_refs=object_refs,
        total_byte_length=int(value["total_byte_length"]),
        s3_bucket=str(value["s3_bucket"]),
        s3_prefix=_BASE_PREFIX,
    )


def hydrate_factor_lab_ref(
    ref: ArtifactRefV1,
    *,
    object_store: ImmutableObjectStore,
    cache_root: str | Path,
) -> Path:
    """Hydrate exactly one ArtifactRefV1; no upstream fallback exists."""

    if not isinstance(ref, ArtifactRefV1):
        raise FactorLabStorageError("artifact ref is invalid")
    try:
        return hydrate_artifact(
            ref=ref,
            object_store=object_store,
            cache_root=cache_root,
        )
    except ImmutableObjectStoreError as error:
        raise FactorLabStorageError(str(error)) from error


def _scope_for_ref(
    refs: FactorLabStorageRefsV2,
    ref: ArtifactRefV1,
) -> str:
    if ref in refs.single_game_refs:
        return "single_game"
    if ref in refs.cross_game_refs:
        return "cross_game"
    return "source"


def _local_status(path: Path, ref: ArtifactRefV1) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_symlink() or not path.is_file():
        return "CONFLICT"
    try:
        digest, byte_length = sha256_path(path)
    except OSError:
        return "CONFLICT"
    return (
        "VERIFIED"
        if digest == ref.sha256 and byte_length == ref.byte_length
        else "CONFLICT"
    )


def _remote_status(
    ref: ArtifactRefV1,
    object_store: ImmutableObjectStore,
) -> str:
    head = object_store.head_object(ref.s3_key)
    if head is None:
        return "MISSING"
    metadata = head.get("metadata")
    if (
        head.get("byte_size") == ref.byte_length
        and isinstance(metadata, dict)
        and metadata.get("sha256") == ref.sha256
    ):
        return "IDENTITY_METADATA_VERIFIED"
    return "CONFLICT"


def factor_lab_storage_status(
    refs: FactorLabStorageRefsV2,
    *,
    project_root: str | Path,
    object_store: ImmutableObjectStore,
) -> FactorLabStorageStatusV2:
    """Report local byte verification and remote identity metadata."""

    root = Path(project_root)
    rows: list[FactorLabStorageStatusRowV2] = []
    for ref in refs.all_refs:
        local = _local_status(_local_path_for_ref(root, ref), ref)
        remote = _remote_status(ref, object_store)
        if "CONFLICT" in {local, remote}:
            state = "CONFLICT"
        elif local == "VERIFIED" and remote == "IDENTITY_METADATA_VERIFIED":
            state = "LOCAL_REMOTE_VERIFIED"
        elif local == "VERIFIED":
            state = "LOCAL_ONLY"
        elif remote == "IDENTITY_METADATA_VERIFIED":
            state = "REMOTE_ONLY"
        else:
            state = "MISSING"
        rows.append(
            FactorLabStorageStatusRowV2(
                scope=_scope_for_ref(refs, ref),
                ref=ref,
                local_status=local,
                remote_status=remote,
                state=state,
            )
        )
    result = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.scope,
                row.ref.s3_key,
                row.ref.local_relative_path,
            ),
        )
    )
    return FactorLabStorageStatusV2(
        rows=result,
        local_verified_count=sum(
            row.local_status == "VERIFIED" for row in result
        ),
        remote_verified_count=sum(
            row.remote_status == "IDENTITY_METADATA_VERIFIED"
            for row in result
        ),
        conflict_count=sum(row.state == "CONFLICT" for row in result),
    )


def configured_factor_lab_object_store(
    *,
    env_file: str | Path | None = None,
) -> tuple[ImmutableObjectStore, str]:
    """Return the existing configured S3 client without exposing credentials."""

    object_store, prefix = configured_object_store(env_file=env_file)
    normalized = prefix.strip("/")
    if normalized != _BASE_PREFIX:
        raise FactorLabStorageError(
            f"configured S3 prefix must be {_BASE_PREFIX}"
        )
    return object_store, normalized


__all__ = [
    "FactorLabStorageError",
    "FactorLabStorageRefsV2",
    "FactorLabStorageStatusRowV2",
    "FactorLabStorageStatusV2",
    "VerifiedFactorLabRemoteInventoryV2",
    "build_factor_lab_storage_refs",
    "configured_factor_lab_object_store",
    "factor_lab_storage_status",
    "hydrate_factor_lab_ref",
    "mirror_factor_lab_storage",
    "verify_factor_lab_remote_inventory",
]
