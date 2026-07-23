"""Immutable, content-addressed storage for governed static source objects."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from prediction_market import contracts


_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_DATASET_RE = re.compile(r"^DS-[A-Z0-9][A-Z0-9-]{0,124}$")
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXTENSION_RE = re.compile(
    r"(?=.{1,32}\Z)"
    r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?)*\Z"
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_NAME_RE = re.compile(
    r"^(?P<digest>[0-9a-f]{64})\.manifest\.json$"
)


class StaticStoreError(ValueError):
    """The static store cannot prove that an operation is safe or valid."""


class StaticStorePathError(StaticStoreError):
    """A storage path is malformed, escapes its root, or traverses a symlink."""


class ImmutableStaticObjectError(StaticStoreError):
    """An operation would overwrite an immutable static object or manifest."""


@dataclass(frozen=True, slots=True)
class StaticObjectRecord:
    """Validated paths and governed manifest for one immutable static object."""

    store_root: Path
    object_path: Path
    manifest_path: Path
    source: str
    dataset: str
    version: str
    partition: str
    extension: str
    manifest: contracts.StaticDatasetManifestV0


@dataclass(frozen=True, slots=True)
class VerifiedStaticObject:
    """A manifest plus bytes verified through the same safe object handle."""

    record: StaticObjectRecord
    object_bytes: bytes


def _validate_component(
    value: object, field: str, pattern: re.Pattern[str]
) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise StaticStorePathError(f"{field} is not a safe path component")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise StaticStorePathError(f"{field} is not a safe path component")
    return value


def _validate_https_url(value: object) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in value
        )
    ):
        raise StaticStoreError("source_url must be a canonical HTTPS URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise StaticStoreError(
            "source_url must be a canonical HTTPS URL"
        ) from error
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise StaticStoreError("source_url must be a canonical HTTPS URL")
    return value


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_read_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _lexical_absolute_path(
    value: str | Path,
    field: str,
    *,
    require_canonical_spelling: bool = False,
) -> Path:
    raw = os.fspath(value)
    candidate = Path(raw)
    if require_canonical_spelling and (
        raw != str(candidate)
        or (candidate.is_absolute() and candidate.anchor != os.sep)
    ):
        raise StaticStorePathError(f"{field} is not canonically spelled")
    if any(part == ".." for part in candidate.parts):
        raise StaticStorePathError(f"{field} contains parent traversal")
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
            raise StaticStorePathError(
                f"storage directory is missing: {name}"
            ) from None
        created = False
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as error:
            raise StaticStorePathError(
                f"cannot create storage directory: {name}"
            ) from error
        if created:
            _fsync_directory(parent_fd, context="directory_create")
        try:
            return os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        except OSError as error:
            raise StaticStorePathError(
                f"storage directory became unsafe: {name}"
            ) from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise StaticStorePathError(
                f"storage path traverses a symlink or non-directory: {name}"
            ) from error
        raise StaticStorePathError(
            f"cannot open storage directory: {name}"
        ) from error


def _open_directory_absolute(path: Path, *, create: bool) -> int:
    if not path.is_absolute():
        raise StaticStorePathError("storage directory must be absolute")
    try:
        descriptor = os.open(path.anchor, _directory_open_flags())
    except OSError as error:
        raise StaticStorePathError("cannot open filesystem root safely") from error
    try:
        for part in path.parts[1:]:
            child = _open_directory_child(descriptor, part, create=create)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise StaticStorePathError("store root must be a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _relative_parts(relative: Path) -> tuple[str, ...]:
    if relative.is_absolute() or not relative.parts:
        raise StaticStorePathError(
            "storage path must be a non-empty relative path"
        )
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise StaticStorePathError("storage path contains an unsafe component")
    return relative.parts


def _open_relative_directory(
    root_fd: int, relative: Path, *, create: bool
) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in _relative_parts(relative):
            child = _open_directory_child(descriptor, part, create=create)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short write while preserving static object")
        offset += written


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


def _read_existing_regular_at(
    directory_fd: int, name: str, field: str
) -> bytes:
    try:
        descriptor = os.open(name, _file_read_flags(), dir_fd=directory_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise StaticStorePathError(f"{field} traverses a symlink") from error
        raise StaticStorePathError(f"{field} is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StaticStorePathError(f"{field} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if chunk == b"":
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise StaticStoreError(f"{field} changed while being read") from error
        if (
            _stat_signature(before) != _stat_signature(after)
            or (before.st_dev, before.st_ino)
            != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(current.st_mode)
        ):
            raise StaticStoreError(f"{field} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_relative_path(relative_text: object, field: str) -> PurePosixPath:
    if type(relative_text) is not str or "\\" in relative_text:
        raise StaticStorePathError(f"{field} is not a safe relative path")
    pure = PurePosixPath(relative_text)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative_text
    ):
        raise StaticStorePathError(f"{field} is not a safe relative path")
    return pure


def _open_relative_file(
    root_fd: int, relative_text: str, field: str
) -> tuple[int, int, str]:
    pure = _safe_relative_path(relative_text, field)
    parent_fd = _open_relative_directory(
        root_fd, Path(*pure.parts[:-1]), create=False
    )
    name = pure.parts[-1]
    descriptor = -1
    try:
        descriptor = os.open(name, _file_read_flags(), dir_fd=parent_fd)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StaticStorePathError(f"{field} is not a regular file")
        return descriptor, parent_fd, name
    except Exception as error:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
        if isinstance(error, StaticStorePathError):
            raise
        if isinstance(error, OSError) and error.errno in {
            errno.ELOOP,
            errno.ENOTDIR,
        }:
            raise StaticStorePathError(f"{field} traverses a symlink") from error
        raise StaticStorePathError(f"{field} is missing or unsafe") from error


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
        and (before.st_dev, before.st_ino)
        == (current.st_dev, current.st_ino)
        and stat.S_ISREG(current.st_mode)
    )


def _require_canonical_path_bindings(
    root: Path,
    root_fd: int,
    bindings: tuple[tuple[Path, int, int, str, str], ...],
) -> None:
    try:
        current_root_fd = _open_directory_absolute(root, create=False)
    except (OSError, StaticStoreError) as error:
        raise StaticStoreError(
            "store root changed during verification"
        ) from error
    try:
        opened_root = os.fstat(root_fd)
        current_root = os.fstat(current_root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (
            current_root.st_dev,
            current_root.st_ino,
        ):
            raise StaticStoreError("store root changed during verification")

        for relative_parent, parent_fd, file_fd, name, field in bindings:
            try:
                current_parent_fd = _open_relative_directory(
                    current_root_fd,
                    relative_parent,
                    create=False,
                )
            except (OSError, StaticStoreError) as error:
                raise StaticStoreError(
                    f"{field} directory ancestry changed during verification"
                ) from error
            try:
                opened_parent = os.fstat(parent_fd)
                current_parent = os.fstat(current_parent_fd)
                if (opened_parent.st_dev, opened_parent.st_ino) != (
                    current_parent.st_dev,
                    current_parent.st_ino,
                ):
                    raise StaticStoreError(
                        f"{field} directory ancestry changed during verification"
                    )
                try:
                    current_file = os.stat(
                        name,
                        dir_fd=current_parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise StaticStoreError(
                        f"{field} changed during verification"
                    ) from error
                if (
                    _stat_signature(os.fstat(file_fd))
                    != _stat_signature(current_file)
                    or not stat.S_ISREG(current_file.st_mode)
                ):
                    raise StaticStoreError(
                        f"{field} changed during verification"
                    )
            finally:
                os.close(current_parent_fd)
    finally:
        os.close(current_root_fd)


def _json_object_no_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StaticStoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                StaticStoreError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, StaticStoreError) as error:
        raise StaticStoreError(f"{context} is not strict JSON") from error
    if type(value) is not dict:
        raise StaticStoreError(f"{context} must be a JSON object")
    return value


def _prefixed_component(
    value: str,
    prefix: str,
    field: str,
    pattern: re.Pattern[str],
) -> str:
    if not value.startswith(prefix):
        raise StaticStorePathError(f"{field} path prefix is invalid")
    return _validate_component(value.removeprefix(prefix), field, pattern)


def _unlink_if_same_inode(
    directory_fd: int, name: str, expected: os.stat_result
) -> bool:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            return False
        os.unlink(name, dir_fd=directory_fd)
        return True
    except FileNotFoundError:
        return False


def _build_record(
    *,
    store_root: Path,
    source: str,
    dataset: str,
    version: str,
    partition: str,
    extension: str,
    manifest: contracts.StaticDatasetManifestV0,
) -> StaticObjectRecord:
    object_path = store_root.joinpath(
        *PurePosixPath(manifest.native_object_path).parts
    )
    manifest_digest = manifest.manifest_sha256.removeprefix("sha256:")
    manifest_path = (
        store_root
        / "manifests"
        / f"source={source}"
        / f"dataset={dataset}"
        / f"version={version}"
        / f"partition={partition}"
        / f"{manifest_digest}.manifest.json"
    )
    return StaticObjectRecord(
        store_root=store_root,
        object_path=object_path,
        manifest_path=manifest_path,
        source=source,
        dataset=dataset,
        version=version,
        partition=partition,
        extension=extension,
        manifest=manifest,
    )


def preserve_static_object(
    store_root: str | Path,
    object_bytes: bytes,
    *,
    program_root: str | Path,
    source: str,
    dataset: str,
    version: str,
    partition: str,
    extension: str,
    source_url: str,
    source_request: Mapping[str, Any],
    source_cursor: str | None,
    fetched_at: str,
    coverage: str,
    etag: str | None,
    last_modified: str | None,
    media_type: str,
    schema_fingerprint: str,
    license_ref: str,
    license_status: Literal[
        "approved", "research_only", "pending", "unknown", "blocked"
    ],
    upstream_partition: str,
    object_kind: Literal["byte_exact_original", "source_derived_extract"],
    lineage: Mapping[str, Any],
) -> StaticObjectRecord:
    """Durably publish exact upstream bytes before their governed manifest."""

    if type(object_bytes) is not bytes:
        raise TypeError("object_bytes must be exact bytes")
    source = _validate_component(source, "source", _SOURCE_RE)
    dataset = _validate_component(dataset, "dataset", _DATASET_RE)
    version = _validate_component(version, "version", _COMPONENT_RE)
    partition = _validate_component(partition, "partition", _COMPONENT_RE)
    extension = _validate_component(extension, "extension", _EXTENSION_RE)
    source_url = _validate_https_url(source_url)
    if type(upstream_partition) is not str or upstream_partition != partition:
        raise StaticStoreError(
            "upstream_partition must equal the canonical storage partition"
        )

    object_digest = hashlib.sha256(object_bytes).hexdigest()
    object_relative = (
        Path("raw")
        / f"source={source}"
        / f"dataset={dataset}"
        / f"version={version}"
        / f"partition={partition}"
        / f"{object_digest}.{extension}"
    )
    manifest_material: dict[str, Any] = {
        "manifest_version": "v0",
        "dataset_id": dataset,
        "object_kind": object_kind,
        "source_url": source_url,
        "source_request": source_request,
        "source_cursor": source_cursor,
        "fetched_at": fetched_at,
        "coverage": coverage,
        "etag": etag,
        "last_modified": last_modified,
        "byte_length": len(object_bytes),
        "object_sha256": f"sha256:{object_digest}",
        "native_object_path": object_relative.as_posix(),
        "media_type": media_type,
        "schema_fingerprint": schema_fingerprint,
        "license_ref": license_ref,
        "license_status": license_status,
        "upstream_partition": upstream_partition,
        "lineage": lineage,
    }
    try:
        manifest_material["manifest_sha256"] = (
            contracts.static_dataset_manifest_sha256(manifest_material)
        )
    except (TypeError, ValueError) as error:
        raise StaticStoreError(
            "static dataset manifest material is not canonical"
        ) from error
    validator = getattr(
        contracts, "validate_static_dataset_manifest_v0", None
    )
    if validator is None:
        raise StaticStoreError(
            "normative static dataset manifest validator is unavailable"
        )
    try:
        manifest = validator(program_root, manifest_material)
    except (TypeError, ValueError) as error:
        raise StaticStoreError("static dataset manifest is invalid") from error
    if not isinstance(manifest, contracts.StaticDatasetManifestV0):
        raise StaticStoreError(
            "normative static dataset manifest validator returned invalid data"
        )

    lexical_root = _lexical_absolute_path(store_root, "store root")
    root_fd = _open_directory_absolute(lexical_root, create=True)
    staging_root_fd = -1
    staging_fd = -1
    raw_directory_fd = -1
    manifest_directory_fd = -1
    staging_name = ""
    manifest_linked = False
    staging_manifest_metadata: os.stat_result | None = None
    record = _build_record(
        store_root=lexical_root,
        source=source,
        dataset=dataset,
        version=version,
        partition=partition,
        extension=extension,
        manifest=manifest,
    )
    try:
        for relative in (Path("raw"), Path("manifests"), Path(".staging")):
            descriptor = _open_relative_directory(
                root_fd, relative, create=True
            )
            os.close(descriptor)
        raw_directory_fd = _open_relative_directory(
            root_fd, object_relative.parent, create=True
        )
        manifest_relative = record.manifest_path.relative_to(lexical_root)
        manifest_directory_fd = _open_relative_directory(
            root_fd, manifest_relative.parent, create=True
        )
        try:
            existing_manifest_metadata = os.stat(
                record.manifest_path.name,
                dir_fd=manifest_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing_manifest_metadata = None
        except OSError as error:
            raise StaticStorePathError(
                "committed manifest path is unsafe"
            ) from error
        if existing_manifest_metadata is not None:
            if not stat.S_ISREG(existing_manifest_metadata.st_mode):
                raise StaticStorePathError(
                    "committed manifest path is not a regular file"
                )
            try:
                existing_record = verify_static_object(
                    record.manifest_path,
                    store_root=lexical_root,
                    program_root=program_root,
                )
            except StaticStoreError as error:
                raise ImmutableStaticObjectError(
                    "committed manifest is invalid and cannot be replaced "
                    "with canonical bytes"
                ) from error
            if existing_record != record:
                raise ImmutableStaticObjectError(
                    "committed manifest content address collides with "
                    "different canonical bytes"
                )
            _fsync_directory(
                manifest_directory_fd, context="manifest_existing"
            )
            return existing_record
        staging_root_fd = _open_relative_directory(
            root_fd, Path(".staging"), create=False
        )
        for _ in range(10):
            candidate = "static-object-" + uuid.uuid4().hex
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=staging_root_fd)
            except FileExistsError:
                continue
            staging_name = candidate
            _fsync_directory(
                staging_root_fd, context="staging_directory_create"
            )
            staging_fd = _open_directory_child(
                staging_root_fd, staging_name, create=False
            )
            break
        else:
            raise StaticStoreError("cannot allocate unique staging directory")

        object_fd = os.open(
            "object",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=staging_fd,
        )
        try:
            _write_all(object_fd, object_bytes)
            os.fsync(object_fd)
            os.fchmod(object_fd, 0o444)
            os.fsync(object_fd)
        finally:
            os.close(object_fd)

        object_already_existed = False
        try:
            os.link(
                "object",
                record.object_path.name,
                src_dir_fd=staging_fd,
                dst_dir_fd=raw_directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            object_already_existed = True
            existing_object = _read_existing_regular_at(
                raw_directory_fd,
                record.object_path.name,
                "static object",
            )
            if existing_object != object_bytes:
                raise ImmutableStaticObjectError(
                    "existing static object does not match exact source bytes"
                )
        _fsync_directory(
            raw_directory_fd,
            context="raw_existing" if object_already_existed else "raw_commit",
        )

        manifest_bytes = contracts.canonical_json_bytes(manifest) + b"\n"
        manifest_fd = os.open(
            "manifest.json",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=staging_fd,
        )
        try:
            _write_all(manifest_fd, manifest_bytes)
            os.fsync(manifest_fd)
            os.fchmod(manifest_fd, 0o444)
            os.fsync(manifest_fd)
            staging_manifest_metadata = os.fstat(manifest_fd)
        finally:
            os.close(manifest_fd)

        try:
            os.link(
                "manifest.json",
                record.manifest_path.name,
                src_dir_fd=staging_fd,
                dst_dir_fd=manifest_directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing_manifest = _read_existing_regular_at(
                manifest_directory_fd,
                record.manifest_path.name,
                "static manifest",
            )
            if existing_manifest != manifest_bytes:
                raise ImmutableStaticObjectError(
                    "existing static manifest does not match canonical bytes"
                )
            _fsync_directory(
                manifest_directory_fd, context="manifest_existing"
            )
            return verify_static_object(
                record.manifest_path,
                store_root=lexical_root,
                program_root=program_root,
            )
        manifest_linked = True
        _fsync_directory(manifest_directory_fd, context="manifest_commit")
        return verify_static_object(
            record.manifest_path,
            store_root=lexical_root,
            program_root=program_root,
        )
    except Exception:
        if (
            manifest_linked
            and manifest_directory_fd >= 0
            and staging_manifest_metadata is not None
            and _unlink_if_same_inode(
                manifest_directory_fd,
                record.manifest_path.name,
                staging_manifest_metadata,
            )
        ):
            _fsync_directory(
                manifest_directory_fd, context="manifest_rollback"
            )
        raise
    finally:
        if staging_fd >= 0:
            for name in ("manifest.json", "object"):
                try:
                    os.unlink(name, dir_fd=staging_fd)
                except FileNotFoundError:
                    pass
            os.close(staging_fd)
        if staging_root_fd >= 0 and staging_name:
            try:
                os.rmdir(staging_name, dir_fd=staging_root_fd)
                _fsync_directory(
                    staging_root_fd, context="staging_directory_cleanup"
                )
            except FileNotFoundError:
                pass
        for descriptor in (
            manifest_directory_fd,
            raw_directory_fd,
            staging_root_fd,
            root_fd,
        ):
            if descriptor >= 0:
                os.close(descriptor)


def _inspect_static_object(
    manifest_path: str | Path,
    *,
    store_root: str | Path,
    program_root: str | Path,
    collect_bytes: bool,
) -> tuple[StaticObjectRecord, bytes | None]:
    lexical_root = _lexical_absolute_path(store_root, "store root")
    lexical_manifest = _lexical_absolute_path(
        manifest_path,
        "manifest path",
        require_canonical_spelling=True,
    )
    try:
        manifest_relative = lexical_manifest.relative_to(lexical_root)
    except ValueError as error:
        raise StaticStorePathError(
            "manifest path escapes the supplied store root"
        ) from error
    manifest_parts = manifest_relative.parts
    if len(manifest_parts) != 6 or manifest_parts[0] != "manifests":
        raise StaticStorePathError(
            "manifest path does not follow the static partition convention"
        )
    source = _prefixed_component(
        manifest_parts[1], "source=", "source", _SOURCE_RE
    )
    dataset = _prefixed_component(
        manifest_parts[2], "dataset=", "dataset", _DATASET_RE
    )
    version = _prefixed_component(
        manifest_parts[3], "version=", "version", _COMPONENT_RE
    )
    partition = _prefixed_component(
        manifest_parts[4], "partition=", "partition", _COMPONENT_RE
    )
    manifest_name_match = _MANIFEST_NAME_RE.fullmatch(manifest_parts[5])
    if manifest_name_match is None:
        raise StaticStorePathError(
            "manifest filename is not a canonical content address"
        )

    root_fd = _open_directory_absolute(lexical_root, create=False)
    manifest_fd = -1
    manifest_parent_fd = -1
    object_fd = -1
    object_parent_fd = -1
    try:
        manifest_fd, manifest_parent_fd, manifest_name = _open_relative_file(
            root_fd, manifest_relative.as_posix(), "manifest path"
        )
        manifest_before = os.fstat(manifest_fd)
        manifest_bytes = _read_fd_bytes(manifest_fd)
        if not _fd_and_path_unchanged(
            manifest_fd,
            manifest_before,
            manifest_parent_fd,
            manifest_name,
        ):
            raise StaticStoreError("manifest changed during verification")
        document = _load_json_object(manifest_bytes, "static manifest")
        try:
            canonical_manifest_bytes = (
                contracts.canonical_json_bytes(document) + b"\n"
            )
        except (TypeError, ValueError) as error:
            raise StaticStoreError(
                "static manifest is not canonical JSON"
            ) from error
        if manifest_bytes != canonical_manifest_bytes:
            raise StaticStoreError("static manifest is not canonical JSON")

        validator = getattr(
            contracts, "validate_static_dataset_manifest_v0", None
        )
        if validator is None:
            raise StaticStoreError(
                "normative static dataset manifest validator is unavailable"
            )
        try:
            manifest = validator(program_root, document)
        except (TypeError, ValueError) as error:
            raise StaticStoreError(
                "static dataset manifest is invalid"
            ) from error
        if not isinstance(manifest, contracts.StaticDatasetManifestV0):
            raise StaticStoreError(
                "normative static dataset manifest validator returned invalid data"
            )

        manifest_digest = manifest.manifest_sha256.removeprefix("sha256:")
        if manifest_name_match.group("digest") != manifest_digest:
            raise StaticStorePathError(
                "manifest path does not match its manifest SHA-256"
            )
        if manifest.dataset_id != dataset:
            raise StaticStorePathError(
                "manifest dataset_id does not match its storage partition"
            )
        if manifest.upstream_partition != partition:
            raise StaticStorePathError(
                "manifest upstream_partition does not match its storage partition"
            )

        object_relative = _safe_relative_path(
            manifest.native_object_path, "native_object_path"
        )
        object_parts = object_relative.parts
        if len(object_parts) != 6 or object_parts[0] != "raw":
            raise StaticStorePathError(
                "native_object_path does not follow the static partition convention"
            )
        object_source = _prefixed_component(
            object_parts[1], "source=", "object source", _SOURCE_RE
        )
        object_dataset = _prefixed_component(
            object_parts[2], "dataset=", "object dataset", _DATASET_RE
        )
        object_version = _prefixed_component(
            object_parts[3], "version=", "object version", _COMPONENT_RE
        )
        object_partition = _prefixed_component(
            object_parts[4],
            "partition=",
            "object partition",
            _COMPONENT_RE,
        )
        if (
            object_source,
            object_dataset,
            object_version,
            object_partition,
        ) != (source, dataset, version, partition):
            raise StaticStorePathError(
                "manifest and object source/dataset/version/partition differ"
            )
        object_digest = manifest.object_sha256.removeprefix("sha256:")
        object_prefix = f"{object_digest}."
        object_name = object_parts[5]
        if not object_name.startswith(object_prefix):
            raise StaticStorePathError(
                "object filename does not match its object SHA-256"
            )
        extension = _validate_component(
            object_name.removeprefix(object_prefix),
            "extension",
            _EXTENSION_RE,
        )
        if (
            _SHA256_HEX_RE.fullmatch(object_digest) is None
            or object_name != f"{object_digest}.{extension}"
        ):
            raise StaticStorePathError(
                "object filename is not a canonical content address"
            )

        object_fd, object_parent_fd, opened_object_name = _open_relative_file(
            root_fd, object_relative.as_posix(), "native_object_path"
        )
        object_before = os.fstat(object_fd)
        digest = hashlib.sha256()
        byte_length = 0
        chunks: list[bytes] | None = [] if collect_bytes else None
        while True:
            chunk = os.read(object_fd, 1024 * 1024)
            if chunk == b"":
                break
            digest.update(chunk)
            byte_length += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        if not _fd_and_path_unchanged(
            object_fd,
            object_before,
            object_parent_fd,
            opened_object_name,
        ):
            raise StaticStoreError("object changed during verification")
        actual_sha256 = "sha256:" + digest.hexdigest()
        if actual_sha256 != manifest.object_sha256:
            raise StaticStoreError(
                "object SHA-256 does not match the governed manifest"
            )
        if byte_length != manifest.byte_length:
            raise StaticStoreError(
                "object byte length does not match the governed manifest"
            )
        if not _fd_and_path_unchanged(
            manifest_fd,
            manifest_before,
            manifest_parent_fd,
            manifest_name,
        ):
            raise StaticStoreError("manifest changed during verification")

        _require_canonical_path_bindings(
            lexical_root,
            root_fd,
            (
                (
                    manifest_relative.parent,
                    manifest_parent_fd,
                    manifest_fd,
                    manifest_name,
                    "manifest",
                ),
                (
                    Path(*object_relative.parts[:-1]),
                    object_parent_fd,
                    object_fd,
                    opened_object_name,
                    "object",
                ),
            ),
        )

        record = _build_record(
            store_root=lexical_root,
            source=source,
            dataset=dataset,
            version=version,
            partition=partition,
            extension=extension,
            manifest=manifest,
        )
        return record, b"".join(chunks) if chunks is not None else None
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


def verify_static_object(
    manifest_path: str | Path,
    *,
    store_root: str | Path,
    program_root: str | Path,
) -> StaticObjectRecord:
    """Fail closed unless one canonical manifest proves its exact object."""

    record, _ = _inspect_static_object(
        manifest_path,
        store_root=store_root,
        program_root=program_root,
        collect_bytes=False,
    )
    return record


def read_verified_static_object(
    manifest_path: str | Path,
    *,
    store_root: str | Path,
    program_root: str | Path,
) -> VerifiedStaticObject:
    """Return exact bytes hashed and validated from one no-follow file handle."""

    record, object_bytes = _inspect_static_object(
        manifest_path,
        store_root=store_root,
        program_root=program_root,
        collect_bytes=True,
    )
    if object_bytes is None:
        raise StaticStoreError("verified object bytes were not retained")
    return VerifiedStaticObject(record=record, object_bytes=object_bytes)


__all__ = [
    "ImmutableStaticObjectError",
    "StaticObjectRecord",
    "StaticStoreError",
    "StaticStorePathError",
    "VerifiedStaticObject",
    "preserve_static_object",
    "read_verified_static_object",
    "verify_static_object",
]
