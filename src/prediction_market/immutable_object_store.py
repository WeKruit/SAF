"""Fail-closed content-addressed storage shared by research artifacts.

The object store is deliberately small: callers own capture semantics and
manifests, while this module proves local/remote byte identity and performs
atomic hydration.  It never falls back to an upstream data provider.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Protocol


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MEDIA_TYPE_RE = re.compile(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+\Z")
_CHUNK_SIZE = 1024 * 1024


class ImmutableObjectStoreError(RuntimeError):
    """An immutable object could not be proven byte-identical."""


class ImmutableObjectStore(Protocol):
    def head_object(self, key: str) -> dict[str, object] | None: ...

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None: ...

    def read_object_chunks(
        self, key: str, *, chunk_size: int = _CHUNK_SIZE
    ) -> Iterator[bytes]: ...


def _relative_posix(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ImmutableObjectStoreError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ImmutableObjectStoreError(f"{label} must be a relative POSIX path")
    return path


def _require_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ImmutableObjectStoreError(
            "artifact hash must be sha256:<64 lowercase hex>"
        )
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRefV1:
    """Stable local/S3 locator for one byte-exact object."""

    sha256: str
    byte_length: int
    media_type: str
    local_relative_path: str
    s3_bucket: str
    s3_key: str

    def __post_init__(self) -> None:
        _require_sha256(self.sha256)
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ImmutableObjectStoreError("byte_length must be nonnegative")
        if (
            not isinstance(self.media_type, str)
            or _MEDIA_TYPE_RE.fullmatch(self.media_type) is None
        ):
            raise ImmutableObjectStoreError("media_type is invalid")
        _relative_posix(self.local_relative_path, label="local_relative_path")
        if not isinstance(self.s3_bucket, str) or not self.s3_bucket.strip():
            raise ImmutableObjectStoreError("s3_bucket must be nonempty")
        _relative_posix(self.s3_key, label="s3_key")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ArtifactRefV1",
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "media_type": self.media_type,
            "local_relative_path": self.local_relative_path,
            "s3_bucket": self.s3_bucket,
            "s3_key": self.s3_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactRefV1":
        if not isinstance(value, dict) or value.get("schema") != "ArtifactRefV1":
            raise ImmutableObjectStoreError("artifact reference schema is invalid")
        try:
            return cls(
                sha256=value["sha256"],
                byte_length=value["byte_length"],
                media_type=value["media_type"],
                local_relative_path=value["local_relative_path"],
                s3_bucket=value["s3_bucket"],
                s3_key=value["s3_key"],
            )
        except KeyError as error:
            raise ImmutableObjectStoreError(
                "artifact reference is incomplete"
            ) from error


def content_addressed_key(
    *,
    prefix: str,
    namespace: str,
    sha256: str,
    suffix: str,
) -> str:
    """Return the frozen ``objects/sha256/<aa>/<sha>`` key."""

    prefix_path = _relative_posix(prefix, label="prefix")
    namespace_path = _relative_posix(namespace, label="namespace")
    digest = _require_sha256(sha256).removeprefix("sha256:")
    if (
        not isinstance(suffix, str)
        or not suffix.startswith(".")
        or "/" in suffix
        or "\\" in suffix
    ):
        raise ImmutableObjectStoreError("suffix must be one filename extension")
    return (
        f"{prefix_path}/{namespace_path}/objects/sha256/"
        f"{digest[:2]}/{digest}{suffix}"
    )


def sha256_path(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_length = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
            byte_length += len(chunk)
    return "sha256:" + digest.hexdigest(), byte_length


def _verify_local(path: Path, ref: ArtifactRefV1, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ImmutableObjectStoreError(f"{label} object is not a regular file")
    observed_sha, observed_bytes = sha256_path(path)
    if observed_sha != ref.sha256 or observed_bytes != ref.byte_length:
        raise ImmutableObjectStoreError(
            f"{label} object hash or byte length differs"
        )


def mirror_artifact(
    local_path: str | Path,
    *,
    ref: ArtifactRefV1,
    object_store: ImmutableObjectStore,
) -> ArtifactRefV1:
    """Upload and read back an object, proving the final remote hash."""

    path = Path(local_path)
    _verify_local(path, ref, label="local")
    head = object_store.head_object(ref.s3_key)
    if head is None:
        object_store.put_file(
            ref.s3_key,
            path,
            metadata={
                "sha256": ref.sha256,
                "byte_length": str(ref.byte_length),
                "media_type": ref.media_type,
            },
        )
    else:
        metadata = head.get("metadata")
        if (
            head.get("byte_size") != ref.byte_length
            or not isinstance(metadata, dict)
            or metadata.get("sha256") != ref.sha256
        ):
            raise ImmutableObjectStoreError(
                "remote object identity conflicts with immutable reference"
            )

    digest = hashlib.sha256()
    byte_length = 0
    try:
        chunks = object_store.read_object_chunks(ref.s3_key)
        for chunk in chunks:
            if not chunk:
                continue
            digest.update(chunk)
            byte_length += len(chunk)
    except Exception as error:
        raise ImmutableObjectStoreError("remote readback failed") from error
    if (
        "sha256:" + digest.hexdigest() != ref.sha256
        or byte_length != ref.byte_length
    ):
        raise ImmutableObjectStoreError(
            "remote object hash or byte length differs"
        )
    return ref


def _validated_target(cache_root: Path, relative: str) -> Path:
    relative_path = _relative_posix(relative, label="local_relative_path")
    root = cache_root.resolve()
    current = root
    for part in relative_path.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ImmutableObjectStoreError("local cache path contains a symlink")
        if current.exists() and not current.is_dir():
            raise ImmutableObjectStoreError(
                "local cache path component is not a directory"
            )
    target = root.joinpath(*relative_path.parts)
    if target.is_symlink():
        raise ImmutableObjectStoreError("local cache target is a symlink")
    return target


def hydrate_artifact(
    *,
    ref: ArtifactRefV1,
    object_store: ImmutableObjectStore,
    cache_root: str | Path,
) -> Path:
    """Hydrate a missing local cache object from S3 and fail closed."""

    target = _validated_target(Path(cache_root), ref.local_relative_path)
    if target.exists():
        _verify_local(target, ref, label="local")
        return target
    head = object_store.head_object(ref.s3_key)
    if head is None:
        raise ImmutableObjectStoreError("remote object is missing")
    metadata = head.get("metadata")
    if (
        head.get("byte_size") != ref.byte_length
        or not isinstance(metadata, dict)
        or metadata.get("sha256") != ref.sha256
    ):
        raise ImmutableObjectStoreError("remote object metadata differs")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.hydrate-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_length = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            try:
                chunks = object_store.read_object_chunks(ref.s3_key)
                for chunk in chunks:
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_length += len(chunk)
            except Exception as error:
                raise ImmutableObjectStoreError("remote object read failed") from error
            handle.flush()
            os.fsync(handle.fileno())
        if (
            "sha256:" + digest.hexdigest() != ref.sha256
            or byte_length != ref.byte_length
        ):
            raise ImmutableObjectStoreError(
                "remote object hash or byte length differs"
            )
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    _verify_local(target, ref, label="local")
    return target


__all__ = [
    "ArtifactRefV1",
    "ImmutableObjectStore",
    "ImmutableObjectStoreError",
    "content_addressed_key",
    "hydrate_artifact",
    "mirror_artifact",
    "sha256_path",
]
