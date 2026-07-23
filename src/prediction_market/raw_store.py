"""Append-only, content-addressed storage for byte-exact raw captures."""

from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import io
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard


_PATH_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CAPTURE_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OBJECT_NAME_RE = re.compile(r"^(?P<digest>[0-9a-f]{64})\.jsonl\.zst$")
_MANIFEST_NAME_RE = re.compile(r"^(?P<digest>[0-9a-f]{64})\.manifest\.json$")

_RECORD_KEYS = frozenset(
    {
        "capture_version",
        "capture_session_id",
        "record_ordinal",
        "receive_at",
        "payload_base64",
        "payload_sha256",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "manifest_version",
        "source",
        "stream",
        "capture_session_id",
        "partition_date",
        "partition_hour",
        "record_count",
        "first_record_ordinal",
        "last_record_ordinal",
        "first_receive_at",
        "last_receive_at",
        "object_path",
        "object_sha256",
        "object_size_bytes",
        "compression",
        "record_encoding",
        "sealed_at",
        "manifest_sha256",
    }
)


class RawStoreError(ValueError):
    """The raw store cannot prove that an operation is safe or valid."""


class RawStorePathError(RawStoreError):
    """A storage path is malformed, escapes its root, or traverses a symlink."""


class ImmutableSegmentError(RawStoreError):
    """An operation would reopen or overwrite an immutable segment."""


class PartitionBoundaryError(RawStoreError):
    """A segment attempted to span more than one UTC hour partition."""


@dataclass(frozen=True, slots=True)
class SegmentManifest:
    """Validated runtime view of an immutable sidecar manifest."""

    path: Path
    object_path: Path
    manifest_version: str
    source: str
    stream: str
    capture_session_id: str
    partition_date: str
    partition_hour: str
    record_count: int
    first_record_ordinal: int | None
    last_record_ordinal: int | None
    first_receive_at: str | None
    last_receive_at: str | None
    object_sha256: str
    object_size_bytes: int
    compression: str
    record_encoding: str
    sealed_at: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SegmentVerification:
    """Fail-closed verification result for a manifest and its exact object."""

    valid: bool
    errors: tuple[str, ...]
    manifest: SegmentManifest | None = None


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if chunk == b"":
            break
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _manifest_sha256(document: dict[str, Any]) -> str:
    hash_input = dict(document)
    hash_input.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(hash_input))


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    if type(value) is not str or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid canonical UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be UTC")
    return parsed


def _utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_component(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise RawStorePathError(f"{field} is not a safe path component")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise RawStorePathError(f"{field} is not a safe path component")
    return value


def _relative_parts(relative: Path) -> tuple[str, ...]:
    if relative.is_absolute() or not relative.parts:
        raise RawStorePathError("storage path must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RawStorePathError("storage path contains an unsafe component")
    return relative.parts


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_read_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _lexical_absolute_path(value: str | Path, field: str) -> Path:
    candidate = Path(value)
    if any(part == ".." for part in candidate.parts):
        raise RawStorePathError(f"{field} contains parent traversal")
    return Path(os.path.abspath(candidate))


def _fsync_directory(
    directory: int | Path, *, context: str | None = None
) -> None:
    del context
    if type(directory) is int:
        os.fsync(directory)
        return
    descriptor = _open_directory_absolute(
        _lexical_absolute_path(directory, "storage directory"),
        create=False,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_child(parent_fd: int, name: str, *, create: bool) -> int:
    try:
        return os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise RawStorePathError(f"storage directory is missing: {name}") from None
        created = False
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise RawStorePathError(
                f"cannot create storage directory: {name}"
            ) from exc
        if created:
            _fsync_directory(parent_fd, context="directory_create")
        try:
            return os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        except OSError as exc:
            raise RawStorePathError(
                f"storage directory became unsafe: {name}"
            ) from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RawStorePathError(
                f"storage path traverses a symlink or non-directory: {name}"
            ) from exc
        raise RawStorePathError(f"cannot open storage directory: {name}") from exc


def _open_directory_absolute(path: Path, *, create: bool) -> int:
    if not path.is_absolute():
        raise RawStorePathError("storage directory must be absolute")
    try:
        descriptor = os.open(path.anchor, _directory_open_flags())
    except OSError as exc:
        raise RawStorePathError("cannot open filesystem root safely") from exc
    try:
        for part in path.parts[1:]:
            child = _open_directory_child(descriptor, part, create=create)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RawStorePathError("store root must be a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_relative_directory(
    root_fd: int, relative: Path, *, create: bool
) -> int:
    parts = _relative_parts(relative)
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = _open_directory_child(descriptor, part, create=create)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _safe_relative_path(relative_text: str, field: str) -> PurePosixPath:
    if type(relative_text) is not str or "\\" in relative_text:
        raise RawStorePathError(f"{field} is not a safe relative path")
    pure = PurePosixPath(relative_text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RawStorePathError(f"{field} is not a safe relative path")
    return pure


def _open_relative_file(
    root_fd: int, relative_text: str, field: str
) -> tuple[int, int, str]:
    pure = _safe_relative_path(relative_text, field)
    parent_relative = Path(*pure.parts[:-1])
    parent_fd = _open_relative_directory(
        root_fd, parent_relative, create=False
    )
    name = pure.parts[-1]
    try:
        descriptor = os.open(name, _file_read_flags(), dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RawStorePathError(f"{field} is not a regular file")
        return descriptor, parent_fd, name
    except OSError as exc:
        os.close(parent_fd)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RawStorePathError(f"{field} traverses a symlink") from exc
        raise RawStorePathError(f"{field} is missing or unsafe") from exc


def _resolve_store_root(root: str | Path) -> tuple[Path, int]:
    lexical = _lexical_absolute_path(root, "store root")
    descriptor = _open_directory_absolute(lexical, create=True)
    return lexical, descriptor


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RawStoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                RawStoreError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RawStoreError) as exc:
        raise RawStoreError(f"{context} is not strict JSON") from exc
    if type(value) is not dict:
        raise RawStoreError(f"{context} must be a JSON object")
    return value


def _stat_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _unlink_if_same_inode(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
) -> bool:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_identity(current) != _stat_identity(expected):
            return False
        os.unlink(name, dir_fd=directory_fd)
        return True
    except FileNotFoundError:
        return False


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short write while sealing raw manifest")
        offset += written


class RawSegmentWriter:
    """Single-writer staging object that can be sealed exactly once."""

    def __init__(
        self,
        root: str | Path,
        *,
        source: str,
        stream: str,
        capture_session_id: str | None = None,
    ) -> None:
        self.source = _validate_component(source, "source", _PATH_COMPONENT_RE)
        self.stream = _validate_component(stream, "stream", _PATH_COMPONENT_RE)
        generated_session = "capture-" + uuid.uuid4().hex
        self.capture_session_id = _validate_component(
            capture_session_id if capture_session_id is not None else generated_session,
            "capture_session_id",
            _CAPTURE_SESSION_RE,
        )
        self._root, self._root_fd = _resolve_store_root(root)
        self._staging_fd = -1
        self._staging_name = ""
        try:
            for relative in (Path("raw"), Path("manifests"), Path(".staging")):
                directory_fd = _open_relative_directory(
                    self._root_fd, relative, create=True
                )
                os.close(directory_fd)
            staging_root_fd = _open_relative_directory(
                self._root_fd, Path(".staging"), create=False
            )
            try:
                for _ in range(10):
                    candidate = "raw-segment-" + uuid.uuid4().hex
                    try:
                        os.mkdir(
                            candidate,
                            mode=0o700,
                            dir_fd=staging_root_fd,
                        )
                    except FileExistsError:
                        continue
                    self._staging_name = candidate
                    _fsync_directory(
                        staging_root_fd, context="staging_directory_create"
                    )
                    self._staging_fd = _open_directory_child(
                        staging_root_fd, candidate, create=False
                    )
                    break
                else:
                    raise RawStoreError("cannot allocate unique staging directory")
            finally:
                os.close(staging_root_fd)
        except Exception:
            os.close(self._root_fd)
            self._root_fd = -1
            raise

        self._staging_path = self._root / ".staging" / self._staging_name
        self._staging_object = self._staging_path / "segment.jsonl.zst"
        try:
            object_descriptor = os.open(
                "segment.jsonl.zst",
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self._staging_fd,
            )
            self._object_handle = os.fdopen(
                object_descriptor, "wb", closefd=True
            )
        except OSError as exc:
            self._cleanup_staging()
            raise RawStoreError("cannot create staging object") from exc
        self._compressor = zstandard.ZstdCompressor(
            level=3,
            write_checksum=True,
            write_content_size=False,
            threads=0,
        ).stream_writer(self._object_handle, closefd=False)
        self._state = "open"
        self._record_count = 0
        self._first_receive_at: str | None = None
        self._last_receive_at: str | None = None
        self._partition: tuple[str, str] | None = None
        self._opened_at = _utc_now_text()

    @property
    def staging_path(self) -> Path:
        return self._staging_path

    def _require_open(self) -> None:
        if self._state != "open":
            raise ImmutableSegmentError(
                f"segment is {self._state}; a sealed segment cannot be reopened"
            )

    def append(self, payload: bytes, *, receive_at: str) -> int:
        """Append one exact payload and return its immutable record ordinal."""

        self._require_open()
        if type(payload) is not bytes:
            raise TypeError("payload must be exact bytes")
        instant = _parse_utc_timestamp(receive_at, "receive_at")
        partition = (instant.date().isoformat(), f"{instant.hour:02d}")
        if self._partition is not None and partition != self._partition:
            raise PartitionBoundaryError(
                "one raw segment cannot cross a UTC hour partition"
            )

        ordinal = self._record_count
        record = {
            "capture_version": "v0",
            "capture_session_id": self.capture_session_id,
            "record_ordinal": ordinal,
            "receive_at": receive_at,
            "payload_base64": base64.b64encode(payload).decode("ascii"),
            "payload_sha256": _sha256_bytes(payload),
        }
        encoded = _canonical_json_bytes(record) + b"\n"
        try:
            self._compressor.write(encoded)
        except Exception as exc:
            self._state = "failed"
            raise RawStoreError("cannot append to staging object") from exc

        self._partition = partition
        if self._first_receive_at is None:
            self._first_receive_at = receive_at
        self._last_receive_at = receive_at
        self._record_count += 1
        return ordinal

    def _close_staging_object(self) -> None:
        try:
            self._compressor.close()
            self._object_handle.flush()
            os.fsync(self._object_handle.fileno())
            self._object_handle.close()
        except Exception as exc:
            self._state = "failed"
            raise RawStoreError("cannot durably finalize staging object") from exc

    def _cleanup_staging(self) -> None:
        if self._staging_fd < 0:
            return
        staging_metadata = os.fstat(self._staging_fd)
        for name in ("segment.manifest.json", "segment.jsonl.zst"):
            try:
                os.unlink(name, dir_fd=self._staging_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        os.close(self._staging_fd)
        self._staging_fd = -1
        try:
            staging_root_fd = _open_relative_directory(
                self._root_fd, Path(".staging"), create=False
            )
            try:
                current = os.stat(
                    self._staging_name,
                    dir_fd=staging_root_fd,
                    follow_symlinks=False,
                )
                if _stat_identity(current) == _stat_identity(staging_metadata):
                    os.rmdir(self._staging_name, dir_fd=staging_root_fd)
                    _fsync_directory(
                        staging_root_fd, context="staging_directory_cleanup"
                    )
            finally:
                os.close(staging_root_fd)
        except OSError:
            pass
        finally:
            if self._root_fd >= 0:
                os.close(self._root_fd)
                self._root_fd = -1

    def seal(self) -> SegmentManifest:
        """Durably publish an immutable object, then its manifest commit point."""

        self._require_open()
        self._state = "sealing"
        raw_linked = False
        manifest_linked = False
        raw_directory_fd = -1
        manifest_directory_fd = -1
        staging_object_fd = -1
        staging_manifest_fd = -1
        final_object_name = ""
        final_manifest_name = ""
        staging_object_metadata: os.stat_result | None = None
        staging_manifest_metadata: os.stat_result | None = None
        try:
            self._close_staging_object()
            staging_object_fd = os.open(
                "segment.jsonl.zst",
                _file_read_flags(),
                dir_fd=self._staging_fd,
            )
            object_sha256 = _sha256_fd(staging_object_fd)
            digest = object_sha256.removeprefix("sha256:")
            staging_object_metadata = os.fstat(staging_object_fd)
            object_size = staging_object_metadata.st_size

            partition_instant = _parse_utc_timestamp(
                self._first_receive_at or self._opened_at,
                "partition timestamp",
            )
            partition_date = partition_instant.date().isoformat()
            partition_hour = f"{partition_instant.hour:02d}"
            partition_relative = Path(
                f"source={self.source}",
                f"stream={self.stream}",
                f"date={partition_date}",
                f"hour={partition_hour}",
            )
            raw_relative = Path("raw") / partition_relative
            manifest_relative = Path("manifests") / partition_relative
            raw_directory_fd = _open_relative_directory(
                self._root_fd, raw_relative, create=True
            )
            manifest_directory_fd = _open_relative_directory(
                self._root_fd, manifest_relative, create=True
            )
            final_object_name = f"{digest}.jsonl.zst"
            final_manifest_name = f"{digest}.manifest.json"
            final_object = self._root / raw_relative / final_object_name
            final_manifest = self._root / manifest_relative / final_manifest_name

            document = {
                "manifest_version": "v0",
                "source": self.source,
                "stream": self.stream,
                "capture_session_id": self.capture_session_id,
                "partition_date": partition_date,
                "partition_hour": partition_hour,
                "record_count": self._record_count,
                "first_record_ordinal": 0 if self._record_count else None,
                "last_record_ordinal": self._record_count - 1
                if self._record_count
                else None,
                "first_receive_at": self._first_receive_at,
                "last_receive_at": self._last_receive_at,
                "object_path": final_object.relative_to(self._root).as_posix(),
                "object_sha256": object_sha256,
                "object_size_bytes": object_size,
                "compression": "zstd",
                "record_encoding": "canonical-json-lines-v0",
                "sealed_at": _utc_now_text(),
            }
            document["manifest_sha256"] = _manifest_sha256(document)
            staging_manifest_fd = os.open(
                "segment.manifest.json",
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self._staging_fd,
            )
            _write_all(
                staging_manifest_fd,
                _canonical_json_bytes(document) + b"\n",
            )
            os.fsync(staging_manifest_fd)
            os.fchmod(staging_object_fd, stat_mode_read_only())
            os.fchmod(staging_manifest_fd, stat_mode_read_only())
            os.fsync(staging_object_fd)
            os.fsync(staging_manifest_fd)
            staging_manifest_metadata = os.fstat(staging_manifest_fd)
            result = _manifest_from_document(
                self._root,
                final_manifest,
                final_object,
                document,
            )

            os.link(
                "segment.jsonl.zst",
                final_object_name,
                src_dir_fd=self._staging_fd,
                dst_dir_fd=raw_directory_fd,
                follow_symlinks=False,
            )
            raw_linked = True
            _fsync_directory(raw_directory_fd, context="raw_commit")
            os.link(
                "segment.manifest.json",
                final_manifest_name,
                src_dir_fd=self._staging_fd,
                dst_dir_fd=manifest_directory_fd,
                follow_symlinks=False,
            )
            manifest_linked = True
            _fsync_directory(
                manifest_directory_fd, context="manifest_commit"
            )

            self._state = "sealed"
            return result
        except Exception as error:
            if (
                manifest_linked
                and manifest_directory_fd >= 0
                and staging_manifest_metadata is not None
                and _unlink_if_same_inode(
                    manifest_directory_fd,
                    final_manifest_name,
                    staging_manifest_metadata,
                )
            ):
                _fsync_directory(
                    manifest_directory_fd, context="manifest_rollback"
                )
                manifest_linked = False
            if (
                raw_linked
                and raw_directory_fd >= 0
                and staging_object_metadata is not None
                and _unlink_if_same_inode(
                    raw_directory_fd,
                    final_object_name,
                    staging_object_metadata,
                )
            ):
                _fsync_directory(raw_directory_fd, context="raw_rollback")
                raw_linked = False
            self._state = "failed"
            if isinstance(error, FileExistsError):
                raise ImmutableSegmentError(
                    "sealed segment final path already exists; overwrite is forbidden"
                ) from error
            raise
        finally:
            for descriptor in (
                staging_manifest_fd,
                staging_object_fd,
                manifest_directory_fd,
                raw_directory_fd,
            ):
                if descriptor >= 0:
                    os.close(descriptor)
            self._cleanup_staging()


def stat_mode_read_only() -> int:
    """Owner/group/world-readable mode for published immutable artifacts."""

    return 0o444


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], context: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise RawStoreError(
            f"{context} fields mismatch; missing={missing!r}, extra={extra!r}"
        )


def _require_exact_type(value: Any, expected: type, field: str) -> None:
    if type(value) is not expected:
        raise RawStoreError(f"{field} has invalid type")


def _manifest_from_document(
    root: Path,
    manifest_path: Path,
    object_path: Path,
    document: dict[str, Any],
) -> SegmentManifest:
    del root
    return SegmentManifest(
        path=manifest_path,
        object_path=object_path,
        manifest_version=document["manifest_version"],
        source=document["source"],
        stream=document["stream"],
        capture_session_id=document["capture_session_id"],
        partition_date=document["partition_date"],
        partition_hour=document["partition_hour"],
        record_count=document["record_count"],
        first_record_ordinal=document["first_record_ordinal"],
        last_record_ordinal=document["last_record_ordinal"],
        first_receive_at=document["first_receive_at"],
        last_receive_at=document["last_receive_at"],
        object_sha256=document["object_sha256"],
        object_size_bytes=document["object_size_bytes"],
        compression=document["compression"],
        record_encoding=document["record_encoding"],
        sealed_at=document["sealed_at"],
        manifest_sha256=document["manifest_sha256"],
    )


def _validate_manifest_document(
    root: Path, manifest_path: Path, document: dict[str, Any]
) -> tuple[SegmentManifest, list[str]]:
    _require_exact_keys(document, _MANIFEST_KEYS, "manifest")
    string_fields = (
        "manifest_version",
        "source",
        "stream",
        "capture_session_id",
        "partition_date",
        "partition_hour",
        "object_path",
        "object_sha256",
        "compression",
        "record_encoding",
        "sealed_at",
        "manifest_sha256",
    )
    for field in string_fields:
        _require_exact_type(document[field], str, field)
    _require_exact_type(document["record_count"], int, "record_count")
    _require_exact_type(document["object_size_bytes"], int, "object_size_bytes")
    if _SHA256_RE.fullmatch(document["manifest_sha256"]) is None:
        raise RawStoreError("manifest_sha256 is invalid")
    if document["manifest_sha256"] != _manifest_sha256(document):
        raise RawStoreError("manifest SHA-256 mismatch")

    if document["manifest_version"] != "v0":
        raise RawStoreError("manifest_version must be v0")
    source = _validate_component(document["source"], "source", _PATH_COMPONENT_RE)
    stream = _validate_component(document["stream"], "stream", _PATH_COMPONENT_RE)
    session = _validate_component(
        document["capture_session_id"],
        "capture_session_id",
        _CAPTURE_SESSION_RE,
    )
    try:
        parsed_date = date.fromisoformat(document["partition_date"])
    except ValueError as exc:
        raise RawStoreError("partition_date is invalid") from exc
    if parsed_date.isoformat() != document["partition_date"]:
        raise RawStoreError("partition_date is non-canonical")
    if re.fullmatch(r"(?:[01][0-9]|2[0-3])", document["partition_hour"]) is None:
        raise RawStoreError("partition_hour is invalid")
    if _SHA256_RE.fullmatch(document["object_sha256"]) is None:
        raise RawStoreError("object_sha256 is invalid")
    if document["compression"] != "zstd":
        raise RawStoreError("compression must be zstd")
    if document["record_encoding"] != "canonical-json-lines-v0":
        raise RawStoreError("record_encoding is invalid")
    _parse_utc_timestamp(document["sealed_at"], "sealed_at")
    if document["record_count"] < 0:
        raise RawStoreError("record_count must be non-negative")
    if document["object_size_bytes"] <= 0:
        raise RawStoreError("object_size_bytes must be positive")

    count = document["record_count"]
    ordinal_fields = ("first_record_ordinal", "last_record_ordinal")
    time_fields = ("first_receive_at", "last_receive_at")
    if count == 0:
        if any(document[field] is not None for field in ordinal_fields + time_fields):
            raise RawStoreError("empty manifest bounds must be null")
    else:
        for field in ordinal_fields:
            _require_exact_type(document[field], int, field)
        for field in time_fields:
            _require_exact_type(document[field], str, field)
            instant = _parse_utc_timestamp(document[field], field)
            if (
                instant.date().isoformat() != document["partition_date"]
                or f"{instant.hour:02d}" != document["partition_hour"]
            ):
                raise RawStoreError(f"{field} is outside manifest partition")
        if document["first_record_ordinal"] != 0:
            raise RawStoreError("first_record_ordinal must be zero")
        if document["last_record_ordinal"] != count - 1:
            raise RawStoreError("last_record_ordinal does not match record_count")

    digest = document["object_sha256"].removeprefix("sha256:")
    expected_object_relative = (
        f"raw/source={source}/stream={stream}/"
        f"date={document['partition_date']}/hour={document['partition_hour']}/"
        f"{digest}.jsonl.zst"
    )
    if document["object_path"] != expected_object_relative:
        raise RawStorePathError(
            "manifest object_path is not the canonical content-addressed path"
        )
    expected_manifest_relative = (
        f"manifests/source={source}/stream={stream}/"
        f"date={document['partition_date']}/hour={document['partition_hour']}/"
        f"{digest}.manifest.json"
    )
    try:
        actual_manifest_relative = manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RawStorePathError("manifest path escapes store root") from exc
    if actual_manifest_relative != expected_manifest_relative:
        raise RawStorePathError("manifest path does not match its content address")

    object_path = root.joinpath(
        *_safe_relative_path(document["object_path"], "object_path").parts
    )
    manifest = _manifest_from_document(root, manifest_path, object_path, document)
    return manifest, []


def _derive_root_and_validate_manifest_path(
    manifest_path: str | Path, root: str | Path | None
) -> tuple[Path, int, Path, int, int, str]:
    lexical_manifest = _lexical_absolute_path(manifest_path, "manifest path")
    parents = lexical_manifest.parents
    if len(parents) < 6 or parents[4].name != "manifests":
        raise RawStorePathError("manifest path does not follow partition convention")
    lexical_root = parents[5]
    if root is not None:
        supplied_root = _lexical_absolute_path(root, "store root")
        if supplied_root != lexical_root:
            raise RawStorePathError("manifest path escapes supplied store root")

    try:
        relative = lexical_manifest.relative_to(lexical_root)
    except ValueError as exc:
        raise RawStorePathError("manifest path escapes store root") from exc
    if len(relative.parts) != 6 or relative.parts[0] != "manifests":
        raise RawStorePathError("manifest path does not follow partition convention")
    if _MANIFEST_NAME_RE.fullmatch(lexical_manifest.name) is None:
        raise RawStorePathError("manifest filename is not content addressed")
    root_fd = _open_directory_absolute(lexical_root, create=False)
    try:
        manifest_fd, parent_fd, name = _open_relative_file(
            root_fd,
            relative.as_posix(),
            "manifest path",
        )
    except Exception:
        os.close(root_fd)
        raise
    return (
        lexical_root,
        root_fd,
        lexical_manifest,
        manifest_fd,
        parent_fd,
        name,
    )


def _read_fd_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if chunk == b"":
            return b"".join(chunks)
        chunks.append(chunk)


def _fd_and_path_unchanged(
    descriptor: int,
    before: os.stat_result,
    parent_fd: int,
    name: str,
) -> bool:
    after = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        _stat_signature(before) == _stat_signature(after)
        and _stat_identity(before) == _stat_identity(current)
        and stat.S_ISREG(current.st_mode)
    )


def _validate_record(
    record: dict[str, Any],
    *,
    canonical_line: bytes,
    manifest: SegmentManifest,
    expected_ordinal: int,
) -> str:
    _require_exact_keys(record, _RECORD_KEYS, f"record {expected_ordinal}")
    for field in (
        "capture_version",
        "capture_session_id",
        "receive_at",
        "payload_base64",
        "payload_sha256",
    ):
        _require_exact_type(record[field], str, f"record {expected_ordinal} {field}")
    _require_exact_type(
        record["record_ordinal"], int, f"record {expected_ordinal} record_ordinal"
    )
    if _canonical_json_bytes(record) != canonical_line:
        raise RawStoreError(f"record {expected_ordinal} is not canonical JSON")
    if record["capture_version"] != "v0":
        raise RawStoreError(f"record {expected_ordinal} capture_version is invalid")
    if record["capture_session_id"] != manifest.capture_session_id:
        raise RawStoreError(f"record {expected_ordinal} capture_session_id mismatch")
    if record["record_ordinal"] != expected_ordinal:
        raise RawStoreError(f"record {expected_ordinal} ordinal is not contiguous")
    instant = _parse_utc_timestamp(
        record["receive_at"], f"record {expected_ordinal} receive_at"
    )
    if (
        instant.date().isoformat() != manifest.partition_date
        or f"{instant.hour:02d}" != manifest.partition_hour
    ):
        raise RawStoreError(f"record {expected_ordinal} crosses manifest partition")
    if _SHA256_RE.fullmatch(record["payload_sha256"]) is None:
        raise RawStoreError(f"record {expected_ordinal} payload_sha256 is invalid")
    try:
        payload = base64.b64decode(record["payload_base64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RawStoreError(f"record {expected_ordinal} payload_base64 is invalid") from exc
    if base64.b64encode(payload).decode("ascii") != record["payload_base64"]:
        raise RawStoreError(f"record {expected_ordinal} payload_base64 is non-canonical")
    if _sha256_bytes(payload) != record["payload_sha256"]:
        raise RawStoreError(f"record {expected_ordinal} payload SHA-256 mismatch")
    return record["receive_at"]


def verify_segment(
    manifest_path: str | Path, *, root: str | Path | None = None
) -> SegmentVerification:
    """Verify path safety, manifest invariants, object hash, and every raw record."""

    errors: list[str] = []
    manifest: SegmentManifest | None = None
    root_fd = -1
    manifest_fd = -1
    manifest_parent_fd = -1
    object_fd = -1
    object_parent_fd = -1
    try:
        (
            resolved_root,
            root_fd,
            resolved_manifest,
            manifest_fd,
            manifest_parent_fd,
            manifest_name,
        ) = _derive_root_and_validate_manifest_path(manifest_path, root)
        manifest_before = os.fstat(manifest_fd)
        manifest_bytes = _read_fd_bytes(manifest_fd)
        if not _fd_and_path_unchanged(
            manifest_fd,
            manifest_before,
            manifest_parent_fd,
            manifest_name,
        ):
            raise RawStoreError("manifest changed during verification")
        document = _load_json_object(manifest_bytes, "segment manifest")
        if manifest_bytes != _canonical_json_bytes(document) + b"\n":
            raise RawStoreError("segment manifest is not canonical JSON")
        manifest, _ = _validate_manifest_document(
            resolved_root, resolved_manifest, document
        )

        object_relative = manifest.object_path.relative_to(resolved_root).as_posix()
        object_fd, object_parent_fd, object_name = _open_relative_file(
            root_fd,
            object_relative,
            "object_path",
        )
        object_before = os.fstat(object_fd)
        actual_size = object_before.st_size
        if actual_size != manifest.object_size_bytes:
            errors.append(
                "object_size_bytes mismatch: "
                f"manifest={manifest.object_size_bytes}, actual={actual_size}"
            )
        actual_sha256 = _sha256_fd(object_fd)
        if actual_sha256 != manifest.object_sha256:
            errors.append(
                "object SHA-256 mismatch: "
                f"manifest={manifest.object_sha256}, actual={actual_sha256}"
            )

        receive_times: list[str] = []
        record_count = 0
        try:
            os.lseek(object_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(object_fd), "rb", closefd=True) as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    buffered = io.BufferedReader(reader)
                    while True:
                        line = buffered.readline()
                        if line == b"":
                            break
                        if not line.endswith(b"\n"):
                            raise RawStoreError(
                                f"record {record_count} lacks canonical newline framing"
                            )
                        canonical_line = line[:-1]
                        record = _load_json_object(
                            canonical_line, f"record {record_count}"
                        )
                        receive_times.append(
                            _validate_record(
                                record,
                                canonical_line=canonical_line,
                                manifest=manifest,
                                expected_ordinal=record_count,
                            )
                        )
                        record_count += 1
        except zstandard.ZstdError as exc:
            errors.append(f"object is not a valid Zstandard stream: {exc}")
        except RawStoreError as exc:
            errors.append(str(exc))

        if not _fd_and_path_unchanged(
            object_fd,
            object_before,
            object_parent_fd,
            object_name,
        ):
            errors.append("object changed during verification")

        if record_count != manifest.record_count:
            errors.append(
                "record_count mismatch: "
                f"manifest={manifest.record_count}, actual={record_count}"
            )
        actual_first = receive_times[0] if receive_times else None
        actual_last = receive_times[-1] if receive_times else None
        if actual_first != manifest.first_receive_at:
            errors.append("first_receive_at mismatch")
        if actual_last != manifest.last_receive_at:
            errors.append("last_receive_at mismatch")
    except (OSError, RawStoreError, ValueError) as exc:
        errors.append(str(exc))
    finally:
        for descriptor in (
            object_fd,
            object_parent_fd,
            manifest_fd,
            manifest_parent_fd,
            root_fd,
        ):
            if descriptor >= 0:
                os.close(descriptor)
    return SegmentVerification(valid=not errors, errors=tuple(errors), manifest=manifest)


__all__ = [
    "ImmutableSegmentError",
    "PartitionBoundaryError",
    "RawSegmentWriter",
    "RawStoreError",
    "RawStorePathError",
    "SegmentManifest",
    "SegmentVerification",
    "verify_segment",
]
