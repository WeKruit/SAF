"""Deterministic dashboard projections for a controlled output root.

Publication provides atomic no-replace for normal cooperative writers in one
controlled process. It does not defend against adversarial same-UID path
replacement after rename. A post-publish verification failure leaves the
published tree untouched for manual inspection; there is no latest pointer.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from prediction_market.sports.nfl_x13_batch import verify_published_batch
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_SOURCE_HTML_BYTES = 64 * 1024 * 1024
MAX_BATCH_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_CHECKSUM_LEDGER_BYTES = 1 * 1024 * 1024
MAX_BATCH_SNAPSHOT_FILE_BYTES = 512 * 1024 * 1024
MAX_BATCH_SNAPSHOT_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
DASHBOARD_BUILDER_VERSION = "nfl-x13-dashboard-exporter-v1"
_PRESENTATION_SCHEMA = "nfl_x13_html_presentation_v1"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.=-]+")


class X13DashboardExportError(ValueError):
    """The source batch or dashboard publication failed a closed gate."""


@dataclass(frozen=True, slots=True)
class DashboardAssetV1:
    role: str
    relative_path: str
    media_type: str
    schema: str
    builder_version: str
    byte_length: int
    object_sha256: str
    source_sha256: str
    source_artifacts: tuple[dict[str, str], ...]
    game_id: str | None = None
    logical_market_id: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardExportV1:
    batch_id: str
    publish_root: Path
    manifest_path: Path
    publish_manifest_sha256: str
    game_count: int
    maximum_asset_bytes: int
    assets: tuple[DashboardAssetV1, ...]
    asset_count: int


class _X13DataScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.scripts: list[dict[str, object]] = []
        self._active: int | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "script":
            return
        attributes = {name.lower(): value for name, value in attrs}
        if attributes.get("id") != "x13-data":
            return
        self.scripts.append(
            {"type": attributes.get("type"), "chunks": []}
        )
        self._active = len(self.scripts) - 1

    def handle_data(self, data: str) -> None:
        if self._active is not None:
            chunks = self.scripts[self._active]["chunks"]
            if isinstance(chunks, list):
                chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._active is not None:
            self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._active is not None:
            self.handle_data(f"&#{name};")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._active is not None:
            self._active = None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise X13DashboardExportError(
            "dashboard projection is not canonical JSON"
        ) from error
    return encoded + b"\n"


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise X13DashboardExportError(f"{label} must be an object")
    return dict(value)


def _require_sequence(value: object, label: str) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise X13DashboardExportError(f"{label} must be an array")
    return list(value)


def _read_open_regular_file(
    descriptor: int,
    *,
    before: os.stat_result,
    max_bytes: int,
    label: str,
) -> bytes:
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
    ):
        raise X13DashboardExportError(f"{label} changed before secure read")
    if opened.st_size > max_bytes:
        raise X13DashboardExportError(f"{label} exceeds its size limit")
    chunks: list[bytes] = []
    remaining = max_bytes
    while True:
        chunk = os.read(descriptor, min(1 << 20, remaining + 1))
        if not chunk:
            break
        if len(chunk) > remaining:
            raise X13DashboardExportError(f"{label} exceeds its size limit")
        chunks.append(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise X13DashboardExportError(f"{label} changed during read")
    return b"".join(chunks)


def _read_regular_file_nofollow(
    path: Path, *, max_bytes: int
) -> bytes:
    try:
        before = os.lstat(path)
    except FileNotFoundError as error:
        raise X13DashboardExportError(f"source file is missing: {path}") from error
    if stat.S_ISLNK(before.st_mode):
        raise X13DashboardExportError(f"source path is a symlink: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise X13DashboardExportError(
            f"source path is not a regular file: {path}"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise X13DashboardExportError("O_NOFOLLOW is required")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise X13DashboardExportError(
            f"source could not be opened without following links: {path}"
        ) from error
    try:
        return _read_open_regular_file(
            descriptor,
            before=before,
            max_bytes=max_bytes,
            label=f"source file {path}",
        )
    finally:
        os.close(descriptor)


def _read_regular_file_at(
    root_fd: int,
    relative_path: str,
    *,
    max_bytes: int,
) -> bytes:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise X13DashboardExportError("anchored source path is unsafe")
    directory_fd = os.dup(root_fd)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        name = relative.parts[-1]
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            return _read_open_regular_file(
                descriptor,
                before=before,
                max_bytes=max_bytes,
                label=f"anchored source file {relative_path}",
            )
        finally:
            os.close(descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        if isinstance(error, X13DashboardExportError):
            raise
        raise X13DashboardExportError(
            f"anchored source file is unavailable: {relative_path}"
        ) from error
    finally:
        os.close(directory_fd)


def _copy_snapshot_file(
    source_parent_fd: int,
    destination_parent_fd: int,
    name: str,
    *,
    remaining_total: int,
) -> tuple[int, str]:
    before = os.stat(name, dir_fd=source_parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise X13DashboardExportError(
            "batch snapshot source is not a regular file"
        )
    if before.st_size > MAX_BATCH_SNAPSHOT_FILE_BYTES:
        raise X13DashboardExportError(
            "batch snapshot file exceeds its size limit"
        )
    if before.st_size > remaining_total:
        raise X13DashboardExportError(
            "batch snapshot exceeds its total size limit"
        )
    digest = hashlib.sha256()
    copied = 0
    with ExitStack() as descriptors:
        source_fd = os.open(
            name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=source_parent_fd,
        )
        descriptors.callback(os.close, source_fd)
        destination_fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
            dir_fd=destination_parent_fd,
        )
        descriptors.callback(os.close, destination_fd)
        opened = os.fstat(source_fd)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise X13DashboardExportError(
                "batch snapshot source changed before capture"
            )
        remaining_file = MAX_BATCH_SNAPSHOT_FILE_BYTES
        while True:
            chunk = os.read(source_fd, min(1 << 20, remaining_file + 1))
            if not chunk:
                break
            if len(chunk) > remaining_file:
                raise X13DashboardExportError(
                    "batch snapshot file exceeds its size limit"
                )
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
            digest.update(chunk)
            copied += len(chunk)
            remaining_file -= len(chunk)
        after = os.fstat(source_fd)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or copied != opened.st_size
        ):
            raise X13DashboardExportError(
                "batch snapshot source changed during capture"
            )
        os.fsync(destination_fd)
    return copied, f"sha256:{digest.hexdigest()}"


def _capture_batch_snapshot(
    source_fd: int,
    destination_fd: int,
    *,
    remaining_total: int,
) -> int:
    copied_total = 0
    for name in sorted(os.listdir(source_fd)):
        status = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISLNK(status.st_mode):
            raise X13DashboardExportError(
                "batch snapshot source contains a symlink"
            )
        if stat.S_ISDIR(status.st_mode):
            os.mkdir(name, 0o700, dir_fd=destination_fd)
            with ExitStack() as child_descriptors:
                source_child = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=source_fd,
                )
                child_descriptors.callback(os.close, source_child)
                destination_child = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=destination_fd,
                )
                child_descriptors.callback(os.close, destination_child)
                child_total = _capture_batch_snapshot(
                    source_child,
                    destination_child,
                    remaining_total=(
                        remaining_total - copied_total
                    ),
                )
                os.fsync(destination_child)
                os.fchmod(destination_child, 0o500)
            copied_total += child_total
        elif stat.S_ISREG(status.st_mode):
            copied, _digest = _copy_snapshot_file(
                source_fd,
                destination_fd,
                name,
                remaining_total=remaining_total - copied_total,
            )
            copied_total += copied
        else:
            raise X13DashboardExportError(
                "batch snapshot source contains a special file"
            )
    if copied_total > remaining_total:
        raise X13DashboardExportError(
            "batch snapshot exceeds its total size limit"
        )
    return copied_total


def extract_presentation_html(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_game_id: str,
) -> dict[str, Any]:
    """Extract the already-published bounded payload without recomputation."""

    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise X13DashboardExportError("expected source SHA-256 is malformed")
    raw = _read_regular_file_nofollow(
        Path(path), max_bytes=MAX_SOURCE_HTML_BYTES
    )
    return _extract_presentation_bytes(
        raw,
        expected_sha256=expected_sha256,
        expected_game_id=expected_game_id,
    )


def _extract_presentation_bytes(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_game_id: str,
) -> dict[str, Any]:
    if _sha256(raw) != expected_sha256:
        raise X13DashboardExportError("HTML source hash does not match ledger")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise X13DashboardExportError("HTML source is not UTF-8") from error
    parser = _X13DataScriptParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as error:
        raise X13DashboardExportError("HTML source is malformed") from error
    if len(parser.scripts) != 1:
        raise X13DashboardExportError(
            "HTML source must contain exactly one x13-data script"
        )
    script = parser.scripts[0]
    if script["type"] != "application/json":
        raise X13DashboardExportError(
            "x13-data script type must be application/json"
        )
    chunks = script["chunks"]
    if not isinstance(chunks, list):
        raise X13DashboardExportError("x13-data script payload is malformed")
    try:
        value = json.loads("".join(str(chunk) for chunk in chunks))
    except json.JSONDecodeError as error:
        raise X13DashboardExportError("x13-data is not valid JSON") from error
    presentation = _require_mapping(value, "x13-data")
    if presentation.get("schema") != _PRESENTATION_SCHEMA:
        raise X13DashboardExportError("x13-data schema is not approved")
    if presentation.get("game_id") != expected_game_id:
        raise X13DashboardExportError("x13-data game identity is foreign")
    if presentation.get("experiment_id") != "X-13":
        raise X13DashboardExportError("x13-data experiment identity is foreign")
    return presentation


def _checksum_ledger(batch_root_fd: int) -> dict[str, str]:
    raw = _read_regular_file_at(
        batch_root_fd,
        "checksums.sha256",
        max_bytes=MAX_CHECKSUM_LEDGER_BYTES,
    )
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise X13DashboardExportError("batch checksum ledger is not ASCII") from error
    entries: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise X13DashboardExportError("batch checksum ledger is malformed")
        digest, relative_text = line[:64], line[66:]
        relative = PurePosixPath(relative_text)
        if (
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative_text in entries
        ):
            raise X13DashboardExportError("batch checksum ledger is unsafe")
        entries[relative_text] = f"sha256:{digest}"
    return entries


def _safe_component(value: object, label: str) -> str:
    text = str(value)
    if not text or _SAFE_COMPONENT_RE.fullmatch(text) is None:
        raise X13DashboardExportError(f"{label} is not a safe path component")
    return text


def _market_identity_component(logical_market_id: str) -> str:
    digest = hashlib.sha256(logical_market_id.encode("utf-8")).hexdigest()
    return f"market-id=sha256-{digest}"


def _source(relative_path: str, sha256: str) -> dict[str, str]:
    return {"relative_path": relative_path, "sha256": sha256}


def _asset_document(
    *,
    schema: str,
    batch_id: str,
    source_artifacts: list[dict[str, str]],
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": schema,
        "experiment_id": "X-13",
        "batch_id": batch_id,
        "builder_version": DASHBOARD_BUILDER_VERSION,
        "source_artifacts": source_artifacts,
        "payload": dict(payload),
    }


def _asset_descriptor(
    *,
    logical_role: str,
    relative_path: str,
    schema: str,
    raw: bytes,
    source_artifacts: list[dict[str, str]],
    game_id: str | None = None,
    logical_market_id: str | None = None,
) -> dict[str, object]:
    if not source_artifacts:
        raise X13DashboardExportError("dashboard asset has no source SHA-256")
    descriptor: dict[str, object] = {
        "logical_role": logical_role,
        "relative_path": relative_path,
        "media_type": "application/json",
        "schema": schema,
        "builder_version": DASHBOARD_BUILDER_VERSION,
        "byte_length": len(raw),
        "object_sha256": _sha256(raw),
        "source_sha256": source_artifacts[0]["sha256"],
        "source_artifacts": source_artifacts,
    }
    if game_id is not None:
        descriptor["game_id"] = game_id
    if logical_market_id is not None:
        descriptor["logical_market_id"] = logical_market_id
    return descriptor


def _typed_asset(descriptor: Mapping[str, object]) -> DashboardAssetV1:
    sources = tuple(
        {
            "relative_path": str(source["relative_path"]),
            "sha256": str(source["sha256"]),
        }
        for source in (
            _require_mapping(value, "dashboard asset source")
            for value in _require_sequence(
                descriptor.get("source_artifacts"),
                "dashboard asset source_artifacts",
            )
        )
    )
    return DashboardAssetV1(
        role=str(descriptor["logical_role"]),
        relative_path=str(descriptor["relative_path"]),
        media_type=str(descriptor["media_type"]),
        schema=str(descriptor["schema"]),
        builder_version=str(descriptor["builder_version"]),
        byte_length=int(descriptor["byte_length"]),
        object_sha256=str(descriptor["object_sha256"]),
        source_sha256=str(descriptor["source_sha256"]),
        source_artifacts=sources,
        game_id=(
            str(descriptor["game_id"])
            if descriptor.get("game_id") is not None
            else None
        ),
        logical_market_id=(
            str(descriptor["logical_market_id"])
            if descriptor.get("logical_market_id") is not None
            else None
        ),
    )


def _write_asset(
    stage_fd: int,
    *,
    directory: PurePosixPath,
    logical_role: str,
    schema: str,
    document: Mapping[str, object],
    source_artifacts: list[dict[str, str]],
    game_id: str | None = None,
    logical_market_id: str | None = None,
) -> dict[str, object]:
    raw = _canonical_json_bytes(document)
    if len(raw) > MAX_ASSET_BYTES:
        raise X13DashboardExportError(
            "dashboard asset exceeds the 8 MiB payload limit"
        )
    digest = _sha256(raw)
    filename = digest.replace(":", "-") + ".json"
    relative = directory / filename
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(_SAFE_COMPONENT_RE.fullmatch(part) is None for part in relative.parts)
    ):
        raise X13DashboardExportError("dashboard asset path is unsafe")
    parent_fd = os.dup(stage_fd)
    try:
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, 0o755, dir_fd=parent_fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        name = relative.parts[-1]
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                name, flags, 0o644, dir_fd=parent_fd
            )
        except OSError as error:
            raise X13DashboardExportError(
                f"dashboard asset could not be created: {relative}"
            ) from error
    finally:
        os.close(parent_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _asset_descriptor(
        logical_role=logical_role,
        relative_path=relative.as_posix(),
        schema=schema,
        raw=raw,
        source_artifacts=source_artifacts,
        game_id=game_id,
        logical_market_id=logical_market_id,
    )


def _directory_anchor(descriptor: int) -> tuple[int, int]:
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        raise X13DashboardExportError("directory anchor is not a directory")
    return status.st_dev, status.st_ino


def _assert_directory_anchor(
    descriptor: int, expected: tuple[int, int], label: str
) -> None:
    if _directory_anchor(descriptor) != expected:
        raise X13DashboardExportError(f"{label} inode anchor changed")


def _open_existing_directory(path: Path) -> tuple[int, tuple[int, int]]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise X13DashboardExportError(
            f"directory cannot be opened without following links: {path}"
        ) from error
    return descriptor, _directory_anchor(descriptor)


def _open_or_create_directory(path: Path) -> tuple[int, tuple[int, int]]:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(
        "/",
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise X13DashboardExportError(
                    "dashboard output traverses a symlink or non-directory"
                ) from error
            os.close(descriptor)
            descriptor = next_fd
        return descriptor, _directory_anchor(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _assert_path_matches_anchor(
    path: Path, expected: tuple[int, int], label: str
) -> None:
    descriptor, observed = _open_existing_directory(path)
    try:
        if observed != expected:
            raise X13DashboardExportError(f"{label} path anchor changed")
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace(
    directory_fd: int,
    source_name: str,
    target_name: str,
    *,
    expected_source_anchor: tuple[int, int] | None = None,
) -> None:
    _safe_component(source_name, "atomic source name")
    _safe_component(target_name, "atomic target name")
    if expected_source_anchor is not None:
        source_status = os.stat(
            source_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            (source_status.st_dev, source_status.st_ino)
            != expected_source_anchor
        ):
            raise X13DashboardExportError(
                "atomic source name no longer maps to its inode anchor"
            )
    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source_name)
    target_raw = os.fsencode(target_name)
    if hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            directory_fd,
            source_raw,
            directory_fd,
            target_raw,
            0x00000004,
        )
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            directory_fd, source_raw, directory_fd, target_raw, 1
        )
    else:
        raise X13DashboardExportError(
            "atomic no-replace publication is unsupported"
        )
    if result == 0:
        return
    observed = ctypes.get_errno()
    if observed in (errno.EEXIST, errno.ENOTEMPTY):
        raise X13DashboardExportError("dashboard publish target already exists")
    raise X13DashboardExportError(
        f"atomic dashboard publication failed: errno={observed}"
    )


def _sync_tree_fd(root_fd: int) -> None:
    for name in os.listdir(root_fd):
        status = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISDIR(status.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=root_fd,
            )
            try:
                _sync_tree_fd(child)
            finally:
                os.close(child)
        elif not stat.S_ISREG(status.st_mode):
            raise X13DashboardExportError(
                "staged dashboard contains an unsafe entry"
            )
    os.fsync(root_fd)


def _clear_directory_fd(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(status.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                child_anchor = _directory_anchor(child)
                _clear_directory_fd(child)
                current = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != child_anchor:
                    raise X13DashboardExportError(
                        "cleanup directory inode changed"
                    )
                os.rmdir(name, dir_fd=descriptor)
            finally:
                os.close(child)
        else:
            os.unlink(name, dir_fd=descriptor)


def _remove_anchored_tree(
    parent_fd: int,
    name: str,
    expected_anchor: tuple[int, int],
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        _assert_directory_anchor(descriptor, expected_anchor, "cleanup target")
        _clear_directory_fd(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != expected_anchor:
        raise X13DashboardExportError("cleanup target inode changed")
    os.rmdir(name, dir_fd=parent_fd)


def _regular_file_inventory(
    root_fd: int, prefix: PurePosixPath = PurePosixPath()
) -> set[str]:
    inventory: set[str] = set()
    for name in os.listdir(root_fd):
        status = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        relative = prefix / name
        if stat.S_ISLNK(status.st_mode):
            raise X13DashboardExportError(
                "published dashboard asset path is a symlink"
            )
        if stat.S_ISDIR(status.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=root_fd,
            )
            try:
                inventory.update(_regular_file_inventory(child, relative))
            finally:
                os.close(child)
        elif stat.S_ISREG(status.st_mode):
            inventory.add(relative.as_posix())
        else:
            raise X13DashboardExportError(
                "published dashboard contains a non-regular asset"
            )
    return inventory


def _verify_published_projection(
    target_fd: int,
    *,
    manifest_name: str,
    manifest_raw: bytes,
    assets: Sequence[Mapping[str, object]],
) -> None:
    _directory_anchor(target_fd)
    expected = {
        str(descriptor["relative_path"]): descriptor for descriptor in assets
    }
    expected_manifest = f"manifests/{manifest_name}"
    actual = _regular_file_inventory(target_fd)
    if actual != set(expected) | {expected_manifest}:
        raise X13DashboardExportError(
            "published dashboard asset set differs from its manifest"
        )
    for relative, descriptor in expected.items():
        raw = _read_regular_file_at(
            target_fd,
            relative,
            max_bytes=MAX_ASSET_BYTES,
        )
        if (
            len(raw) != descriptor["byte_length"]
            or _sha256(raw) != descriptor["object_sha256"]
        ):
            raise X13DashboardExportError(
                f"published dashboard asset identity mismatch: {relative}"
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise X13DashboardExportError(
                f"published dashboard asset is not JSON: {relative}"
            ) from error
        if raw != _canonical_json_bytes(value):
            raise X13DashboardExportError(
                f"published dashboard asset is not canonical: {relative}"
            )
    observed_manifest = _read_regular_file_at(
        target_fd,
        f"manifests/{manifest_name}",
        max_bytes=MAX_ASSET_BYTES,
    )
    if observed_manifest != manifest_raw:
        raise X13DashboardExportError("published dashboard manifest changed")


def _game_assets(
    *,
    stage_fd: int,
    batch_id: str,
    game_id: str,
    presentation: Mapping[str, Any],
    html_source: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_artifacts = [html_source]
    game_directory = PurePosixPath("games") / game_id
    assets: list[dict[str, object]] = []
    contracts = [
        _require_mapping(row, f"{game_id}.contract")
        for row in _require_sequence(
            presentation.get("contracts"), f"{game_id}.contracts"
        )
    ]
    contract_by_market: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        _safe_component(contract.get("venue"), "market venue")
        _safe_component(contract.get("family"), "market family")
        logical_market_id = str(contract.get("logical_market_id", ""))
        if not logical_market_id or logical_market_id in contract_by_market:
            raise X13DashboardExportError(
                f"{game_id} contracts have a missing or duplicate logical market"
            )
        contract_by_market[logical_market_id] = contract
    series_by_market: dict[str, list[dict[str, Any]]] = {
        logical_market_id: [] for logical_market_id in contract_by_market
    }
    for value in _require_sequence(
        presentation.get("market_series"), f"{game_id}.market_series"
    ):
        row = _require_mapping(value, f"{game_id}.market_series row")
        logical_market_id = str(row.get("logical_market_id", ""))
        if logical_market_id not in series_by_market:
            raise X13DashboardExportError(
                f"{game_id} market series references an unknown contract"
            )
        contract = contract_by_market[logical_market_id]
        if (
            row.get("venue") != contract.get("venue")
            or row.get("family") != contract.get("family")
        ):
            raise X13DashboardExportError(
                f"{game_id} market series identity differs from its contract"
            )
        provenance = row.get("provenance")
        if provenance not in {"observed", "derived_complement"}:
            raise X13DashboardExportError(
                f"{game_id} market series provenance is unknown"
            )
        series_kind = row.get("series_kind")
        if series_kind not in {"trade_range", "bbo_candle"}:
            raise X13DashboardExportError(
                f"{game_id} market series kind is unknown"
            )
        if series_kind == "bbo_candle" and (
            contract.get("venue") == "polymarket"
            or provenance != "observed"
        ):
            raise X13DashboardExportError(
                f"{game_id} Polymarket or derived series cannot invent BBO"
            )
        series_by_market[logical_market_id].append(row)

    market_refs: dict[str, dict[str, object]] = {}
    for logical_market_id, contract in sorted(contract_by_market.items()):
        venue = _safe_component(contract.get("venue"), "market venue")
        family = _safe_component(contract.get("family"), "market family")
        schema = "nfl_x13_dashboard_market_series_v1"
        document = _asset_document(
            schema=schema,
            batch_id=batch_id,
            source_artifacts=source_artifacts,
            payload={
                "game_id": game_id,
                "logical_market_id": logical_market_id,
                "contract": contract,
                "series": series_by_market[logical_market_id],
            },
        )
        descriptor = _write_asset(
            stage_fd,
            directory=(
                game_directory
                / "markets"
                / venue
                / family
                / _market_identity_component(logical_market_id)
            ),
            logical_role="market_series",
            schema=schema,
            document=document,
            source_artifacts=source_artifacts,
            game_id=game_id,
            logical_market_id=logical_market_id,
        )
        assets.append(descriptor)
        market_refs[logical_market_id] = {
            "relative_path": descriptor["relative_path"],
            "object_sha256": descriptor["object_sha256"],
        }

    contract_rows = []
    for logical_market_id, contract in sorted(contract_by_market.items()):
        row = dict(contract)
        row["market_asset"] = market_refs[logical_market_id]
        contract_rows.append(row)
    contracts_schema = "nfl_x13_dashboard_contracts_v1"
    contracts_document = _asset_document(
        schema=contracts_schema,
        batch_id=batch_id,
        source_artifacts=source_artifacts,
        payload={"game_id": game_id, "contracts": contract_rows},
    )
    contracts_descriptor = _write_asset(
        stage_fd,
        directory=game_directory / "contracts",
        logical_role="game_contracts",
        schema=contracts_schema,
        document=contracts_document,
        source_artifacts=source_artifacts,
        game_id=game_id,
    )
    assets.append(contracts_descriptor)

    association_schema = "nfl_x13_dashboard_association_preview_v1"
    association_document = _asset_document(
        schema=association_schema,
        batch_id=batch_id,
        source_artifacts=source_artifacts,
        payload={
            "game_id": game_id,
            "association_preview": _require_sequence(
                presentation.get("association_preview"),
                f"{game_id}.association_preview",
            ),
            "association_coverage": _require_mapping(
                presentation.get("association_coverage"),
                f"{game_id}.association_coverage",
            ),
        },
    )
    association_descriptor = _write_asset(
        stage_fd,
        directory=game_directory / "associations",
        logical_role="association_preview",
        schema=association_schema,
        document=association_document,
        source_artifacts=source_artifacts,
        game_id=game_id,
    )
    assets.append(association_descriptor)

    core_schema = "nfl_x13_dashboard_game_core_v1"
    core_payload = {
        key: presentation.get(key)
        for key in (
            "status",
            "game_id",
            "teams",
            "final_score",
            "events",
            "episodes",
            "stat_ledger",
            "personnel",
            "observation_coverage",
            "presentation_limits",
            "market_coverage",
            "audit",
            "lineage",
        )
    }
    core_document = _asset_document(
        schema=core_schema,
        batch_id=batch_id,
        source_artifacts=source_artifacts,
        payload=core_payload,
    )
    core_descriptor = _write_asset(
        stage_fd,
        directory=game_directory / "core",
        logical_role="game_core",
        schema=core_schema,
        document=core_document,
        source_artifacts=source_artifacts,
        game_id=game_id,
    )
    assets.append(core_descriptor)

    teams = _require_mapping(presentation.get("teams"), f"{game_id}.teams")
    score = _require_mapping(
        presentation.get("final_score"), f"{game_id}.final_score"
    )
    game_catalog = {
        "game_id": game_id,
        "away_team": teams.get("away"),
        "home_team": teams.get("home"),
        "away_score": score.get("away"),
        "home_score": score.get("home"),
        "event_count": len(
            _require_sequence(presentation.get("events"), f"{game_id}.events")
        ),
        "episode_count": len(
            _require_sequence(
                presentation.get("episodes"), f"{game_id}.episodes"
            )
        ),
        "contract_count": len(contracts),
        "market_asset_count": len(market_refs),
        "venues": sorted(
            {
                str(contract.get("venue"))
                for contract in contracts
                if contract.get("venue") is not None
            }
        ),
        "status": presentation.get("status"),
        "audit": presentation.get("audit"),
        "assets": {
            "core": {
                "relative_path": core_descriptor["relative_path"],
                "object_sha256": core_descriptor["object_sha256"],
            },
            "contracts": {
                "relative_path": contracts_descriptor["relative_path"],
                "object_sha256": contracts_descriptor["object_sha256"],
            },
            "associations": {
                "relative_path": association_descriptor["relative_path"],
                "object_sha256": association_descriptor["object_sha256"],
            },
        },
    }
    return assets, game_catalog


def _prepare_batch_snapshot(
    source_batch: Path,
) -> tuple[Path, int, tuple[int, int], ExitStack]:
    resources = ExitStack()
    try:
        source_fd, source_anchor = _open_existing_directory(source_batch)
        resources.callback(os.close, source_fd)
        snapshot_root = resources.enter_context(
            tempfile.TemporaryDirectory(
                prefix=".x13-dashboard-batch-snapshot-"
            )
        )
        snapshot_batch = Path(snapshot_root) / source_batch.name
        snapshot_batch.mkdir(mode=0o700)
        snapshot_fd, snapshot_anchor = _open_existing_directory(
            snapshot_batch
        )
        resources.callback(os.close, snapshot_fd)
        _capture_batch_snapshot(
            source_fd,
            snapshot_fd,
            remaining_total=MAX_BATCH_SNAPSHOT_TOTAL_BYTES,
        )
        _assert_directory_anchor(
            source_fd, source_anchor, "source batch root"
        )
        os.fsync(snapshot_fd)
        os.fchmod(snapshot_fd, 0o500)
        return snapshot_batch, snapshot_fd, snapshot_anchor, resources
    except BaseException:
        resources.close()
        raise


def export_x13_dashboard(
    batch_path: str | Path, output_root: str | Path
) -> DashboardExportV1:
    """Publish once into a controlled, cooperative-writer output root.

    The no-replace rename is atomic. Verification after that commit point is
    observational only: any failure leaves the published path untouched and
    requires manual inspection. Same-UID adversarial replacement is outside
    the guarantee, and this exporter does not maintain a latest pointer.
    """

    source_batch = Path(batch_path)
    output = Path(output_root)
    batch, batch_fd, batch_anchor, snapshot_resources = (
        _prepare_batch_snapshot(source_batch)
    )
    output_fd: int | None = None
    stage_name: str | None = None
    stage_anchor: tuple[int, int] | None = None
    stage_fd: int | None = None
    target_name: str | None = None
    published = False
    try:
        verification = verify_published_batch(batch)
        _assert_directory_anchor(batch_fd, batch_anchor, "batch root")
        if (
            verification.get("game_count") != len(X13_GAME_IDS)
            or verification.get("batch_id") is None
        ):
            raise X13DashboardExportError(
                "published batch verification is incomplete"
            )
        batch_id = str(verification["batch_id"])
        if _SHA256_RE.fullmatch(batch_id) is None:
            raise X13DashboardExportError("verified batch_id is malformed")

        ledger = _checksum_ledger(batch_fd)
        manifest_relative = "batch_manifest.json"
        if manifest_relative not in ledger:
            raise X13DashboardExportError(
                "batch manifest lacks a source hash"
            )
        manifest_raw = _read_regular_file_at(
            batch_fd,
            manifest_relative,
            max_bytes=MAX_BATCH_MANIFEST_BYTES,
        )
        if _sha256(manifest_raw) != ledger[manifest_relative]:
            raise X13DashboardExportError(
                "batch manifest source hash mismatch"
            )
        try:
            batch_manifest = _require_mapping(
                json.loads(manifest_raw), "batch manifest"
            )
        except json.JSONDecodeError as error:
            raise X13DashboardExportError(
                "batch manifest is not JSON"
            ) from error
        if (
            batch_manifest.get("batch_id") != batch_id
            or batch_manifest.get("game_ids") != list(X13_GAME_IDS)
        ):
            raise X13DashboardExportError(
                "batch manifest identity is inconsistent"
            )

        output_fd, output_anchor = _open_or_create_directory(output)
        _assert_directory_anchor(output_fd, output_anchor, "output root")
        target_name = f"batch={batch_id.replace(':', '-')}"
        try:
            os.stat(
                target_name,
                dir_fd=output_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise X13DashboardExportError(
                "dashboard publish target already exists"
            )
        stage_name = f".{target_name}.stage-{secrets.token_hex(16)}"
        os.mkdir(stage_name, 0o755, dir_fd=output_fd)
        stage_status = os.stat(
            stage_name, dir_fd=output_fd, follow_symlinks=False
        )
        stage_anchor = (stage_status.st_dev, stage_status.st_ino)
        stage_fd = os.open(
            stage_name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=output_fd,
        )
        _assert_directory_anchor(stage_fd, stage_anchor, "staging root")

        assets: list[dict[str, object]] = []
        game_catalog: list[dict[str, object]] = []
        html_sources: list[dict[str, str]] = []
        for game_id in X13_GAME_IDS:
            relative = f"games/{game_id}.html"
            expected = ledger.get(relative)
            if expected is None:
                raise X13DashboardExportError(
                    f"{game_id} HTML lacks a source hash"
                )
            source = _source(relative, expected)
            html_sources.append(source)
            html_raw = _read_regular_file_at(
                batch_fd,
                relative,
                max_bytes=MAX_SOURCE_HTML_BYTES,
            )
            presentation = _extract_presentation_bytes(
                html_raw,
                expected_sha256=expected,
                expected_game_id=game_id,
            )
            game_assets, catalog_row = _game_assets(
                stage_fd=stage_fd,
                batch_id=batch_id,
                game_id=game_id,
                presentation=presentation,
                html_source=source,
            )
            assets.extend(game_assets)
            game_catalog.append(catalog_row)

        catalog_sources = [
            _source(manifest_relative, ledger[manifest_relative]),
            *html_sources,
        ]
        catalog_schema = "nfl_x13_dashboard_catalog_v1"
        catalog_document = _asset_document(
            schema=catalog_schema,
            batch_id=batch_id,
            source_artifacts=catalog_sources,
            payload={
                "status": "PRELIMINARY_SOURCE_TIME_ONLY",
                "publication_gate": batch_manifest.get("publication_gate"),
                "game_count": len(game_catalog),
                "games": game_catalog,
            },
        )
        catalog_descriptor = _write_asset(
            stage_fd,
            directory=PurePosixPath("catalog"),
            logical_role="catalog",
            schema=catalog_schema,
            document=catalog_document,
            source_artifacts=catalog_sources,
        )
        assets.append(catalog_descriptor)

        assets.sort(key=lambda row: str(row["relative_path"]))
        manifest_document = {
            "schema": "nfl_x13_dashboard_publish_manifest_v1",
            "experiment_id": "X-13",
            "batch_id": batch_id,
            "builder_version": DASHBOARD_BUILDER_VERSION,
            "status": "PRELIMINARY_SOURCE_TIME_ONLY",
            "claim_boundary": {
                "causality": False,
                "real_latency": False,
                "execution": False,
                "tradable_alpha": False,
                "purpose": "human_replay_query_and_candidate_exploration",
            },
            "source_batch_manifest": _source(
                manifest_relative, ledger[manifest_relative]
            ),
            "game_count": len(game_catalog),
            "asset_count": len(assets),
            "assets": assets,
        }
        manifest_raw = _canonical_json_bytes(manifest_document)
        if len(manifest_raw) > MAX_ASSET_BYTES:
            raise X13DashboardExportError(
                "dashboard publish manifest exceeds the 8 MiB payload limit"
            )
        manifest_sha256 = _sha256(manifest_raw)
        os.mkdir("manifests", 0o755, dir_fd=stage_fd)
        manifest_directory_fd = os.open(
            "manifests",
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=stage_fd,
        )
        manifest_name = manifest_sha256.replace(":", "-") + ".json"
        try:
            descriptor = os.open(
                manifest_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o644,
                dir_fd=manifest_directory_fd,
            )
            try:
                view = memoryview(manifest_raw)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(manifest_directory_fd)
        _sync_tree_fd(stage_fd)
        _verify_published_projection(
            stage_fd,
            manifest_name=manifest_name,
            manifest_raw=manifest_raw,
            assets=assets,
        )
        _assert_directory_anchor(output_fd, output_anchor, "output root")
        _assert_directory_anchor(batch_fd, batch_anchor, "batch root")
        stage_current = os.stat(
            stage_name, dir_fd=output_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(stage_current.st_mode)
            or (stage_current.st_dev, stage_current.st_ino)
            != stage_anchor
        ):
            raise X13DashboardExportError(
                "staging name no longer maps to its inode anchor"
            )
        _atomic_rename_noreplace(
            output_fd,
            stage_name,
            target_name,
            expected_source_anchor=stage_anchor,
        )
        published = True
        try:
            target_fd = os.open(
                target_name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=output_fd,
            )
        except OSError as error:
            raise X13DashboardExportError(
                "published target cannot be opened as the staged directory"
            ) from error
        try:
            observed_target_anchor = _directory_anchor(target_fd)
            if observed_target_anchor != stage_anchor:
                raise X13DashboardExportError(
                    "published target inode differs from staged inode"
                )
            _verify_published_projection(
                target_fd,
                manifest_name=manifest_name,
                manifest_raw=manifest_raw,
                assets=assets,
            )
        finally:
            os.close(target_fd)
        _assert_directory_anchor(output_fd, output_anchor, "output root")
        try:
            _assert_path_matches_anchor(output, output_anchor, "output")
        except X13DashboardExportError as error:
            raise X13DashboardExportError(
                "output path anchor changed"
            ) from error
        os.fsync(output_fd)

        target = output / target_name
        manifest_path = target / "manifests" / manifest_name
        typed_assets = tuple(
            _typed_asset(descriptor) for descriptor in assets
        )
        return DashboardExportV1(
            batch_id=batch_id,
            publish_root=target,
            manifest_path=manifest_path,
            publish_manifest_sha256=manifest_sha256,
            game_count=len(game_catalog),
            maximum_asset_bytes=max(
                len(manifest_raw),
                *(asset.byte_length for asset in typed_assets),
            ),
            assets=typed_assets,
            asset_count=len(assets),
        )
    except BaseException as error:
        if published:
            raise X13DashboardExportError(
                "dashboard published but verification failed; "
                "manual inspection required"
            ) from error
        if output_fd is not None:
            if stage_name is not None and stage_anchor is not None:
                try:
                    current = os.stat(
                        stage_name,
                        dir_fd=output_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    if (
                        current.st_dev,
                        current.st_ino,
                    ) == stage_anchor:
                        _remove_anchored_tree(
                            output_fd, stage_name, stage_anchor
                        )
        raise
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        if output_fd is not None:
            os.close(output_fd)
        snapshot_resources.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export bounded dashboard projections from a verified X-13 batch."
    )
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = export_x13_dashboard(arguments.batch, arguments.output)
    print(
        json.dumps(
            {
                "batch_id": result.batch_id,
                "publish_root": str(result.publish_root),
                "manifest_path": str(result.manifest_path),
                "publish_manifest_sha256": (
                    result.publish_manifest_sha256
                ),
                "game_count": result.game_count,
                "maximum_asset_bytes": result.maximum_asset_bytes,
                "asset_count": result.asset_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DASHBOARD_BUILDER_VERSION",
    "MAX_ASSET_BYTES",
    "DashboardAssetV1",
    "DashboardExportV1",
    "X13DashboardExportError",
    "export_x13_dashboard",
    "extract_presentation_html",
    "main",
]
