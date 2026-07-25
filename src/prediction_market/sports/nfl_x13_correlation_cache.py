"""Deterministic descriptive cache for audited NFL X-13 market reactions.

This module deliberately stops before inference.  It projects the exact
audited OBSERVED subset into a queryable Parquet file and binds each row to
canonical episode state and structured contract orientation.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import resource
import shutil
import stat
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_x13_batch import (
    X13_ASSOCIATION_SCHEMA_FINGERPRINT,
    X13_FROZEN_UNIVERSE_ID,
    association_arrow_schema,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_BATCH_SPEC,
    X13_DELAY_SCENARIOS_SECONDS,
    X13_GAME_IDS,
    X13_HORIZONS_SECONDS,
    canonical_sha256,
)


X13_FROZEN_BATCH_ID = (
    "sha256:3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
)
X13_APPROVED_CACHE_ID = (
    "sha256:d6e98965c0985a4f4769480f8d40bdabac1375e5f6b3bd66bc7bf654d02529c5"
)
X13_FROZEN_CANDIDATE_ROW_COUNT = 48_624_912
X13_EXPECTED_OBSERVED_ROW_COUNT = 333_340
X13_EXPECTED_ASSOCIATION_PARTITIONS = 306
X13_EXPECTED_CROSS_VENUE_ORIENTATIONS = 154
X13_CACHE_SCHEMA = "nfl_x13_reaction_observed_cache_v1"
X13_COVERAGE_SCHEMA = "nfl_x13_reaction_coverage_matrix_v1"
_MAX_CAPTURE_BYTES = 64 << 20
_READ_CHUNK_BYTES = 1 << 20
_SOURCE_FILTER = (
    "validity_status='OBSERVED' AND contaminated=false "
    "AND order_ambiguous=false"
)
_OBSERVED_CACHE_FIELDS = (
    ("game_id", "string"),
    ("episode_id", "string"),
    ("episode_type", "string"),
    ("contract_id", "string"),
    ("logical_market_id", "string"),
    ("venue", "string"),
    ("family", "string"),
    ("outcome", "string"),
    ("source_time_start_utc", "string"),
    ("source_time_end_utc", "string"),
    ("delay_scenario_seconds", "int32"),
    ("horizon_seconds", "int32"),
    ("pre_event_actual_trade", "string"),
    ("first_post_event_trade", "string"),
    ("vwap", "string"),
    ("signed_price_change", "string"),
    ("maximum_excursion", "string"),
    ("net_change_60s", "string"),
    ("trade_count", "int64"),
    ("volume", "string"),
    ("staleness_seconds", "string"),
    ("overshoot_candidate", "bool"),
    ("reversal_candidate", "bool"),
    ("two_venue_direction_consistency", "string"),
    ("order_ambiguous", "bool"),
    ("contaminated", "bool"),
    ("validity_status", "string"),
    ("episode_superclass", "string"),
    ("episode_is_score", "bool"),
    ("episode_is_turnover", "bool"),
    ("signal_event_eligible", "bool"),
    ("signal_exclusion_reason", "string"),
    ("signal_focal_team", "string"),
    ("scoring_team", "string"),
    ("pre_offense_team", "string"),
    ("pre_defense_team", "string"),
    ("pre_game_seconds_remaining", "int32"),
    ("pre_quarter", "int32"),
    ("pre_score_margin_home", "int32"),
    ("pre_possession_team", "string"),
    ("pre_down", "int32"),
    ("pre_distance", "int32"),
    ("pre_yardline_100", "int32"),
    ("pre_goal_to_go", "bool"),
    ("contract_subject", "string"),
    ("contract_period", "string"),
    ("contract_line", "string"),
    ("contract_measure", "string"),
    ("contract_comparator", "string"),
    ("semantic_orientation_id", "string"),
    ("cross_venue_eligible", "bool"),
    ("pre_trade_age_seconds", "double"),
    ("pre_trade_age_status", "string"),
    ("pre_trade_fresh_le_60s", "bool"),
    ("signal_candidate_eligible", "bool"),
)
_COVERAGE_FIELDS = (
    ("game_id", "string"),
    ("episode_id", "string"),
    ("episode_superclass", "string"),
    ("venue", "string"),
    ("family", "string"),
    ("validity_status", "string"),
    ("order_ambiguous", "bool"),
    ("contaminated", "bool"),
    ("row_count", "int64"),
    ("distinct_contract_outcome_count", "int64"),
    ("delay_horizon_cell_count", "int64"),
    ("parameter_repetition_row_count", "int64"),
)


class X13CorrelationCacheError(ValueError):
    """The descriptive X-13 cache cannot prove its inputs or joins."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise X13CorrelationCacheError(
            f"cache metadata is not canonical JSON: {error}"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_relative(value: PurePosixPath) -> None:
    if (
        value.is_absolute()
        or not value.parts
        or ".." in value.parts
        or any(part in {"", "."} for part in value.parts)
    ):
        raise X13CorrelationCacheError("input path is unsafe")


def _open_bound_file(root_fd: int, relative: PurePosixPath) -> int:
    _safe_relative(relative)
    parent = os.dup(root_fd)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            os.close(parent)
            parent = next_fd
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
    except OSError as error:
        raise X13CorrelationCacheError(
            f"input is a symlink, missing, or unreadable: {relative}"
        ) from error
    finally:
        os.close(parent)
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise X13CorrelationCacheError(
            f"input is not a regular file: {relative}"
        )
    return descriptor


def _copy_bound_input(
    root_fd: int,
    relative: PurePosixPath,
    target: Path,
    *,
    expected_sha256: str,
    expected_byte_length: int | None = None,
) -> tuple[str, int]:
    """Copy one no-follow source handle and bind it to its frozen digest."""

    descriptor = _open_bound_file(root_fd, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    output_fd: int | None = None
    try:
        before = os.fstat(descriptor)
        output_fd = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        digest = hashlib.sha256()
        byte_length = 0
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            digest.update(chunk)
            byte_length += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_fd, view)
                view = view[written:]
        os.fsync(output_fd)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise X13CorrelationCacheError(
                f"input changed while being copied: {relative}"
            )
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise X13CorrelationCacheError(
                f"input digest mismatch: {relative}"
            )
        if (
            expected_byte_length is not None
            and byte_length != expected_byte_length
        ):
            raise X13CorrelationCacheError(
                f"input byte length mismatch: {relative}"
            )
        return observed, byte_length
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(descriptor)


def _hash_bound_input(
    root_fd: int, relative: PurePosixPath
) -> tuple[str, int]:
    descriptor = _open_bound_file(root_fd, relative)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        byte_length = 0
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            digest.update(chunk)
            byte_length += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise X13CorrelationCacheError(
                f"input changed while being hashed: {relative}"
            )
        return digest.hexdigest(), byte_length
    finally:
        os.close(descriptor)


def _read_bound_bytes(
    root_fd: int, relative: PurePosixPath, *, maximum_bytes: int
) -> bytes:
    descriptor = _open_bound_file(root_fd, relative)
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum_bytes:
            raise X13CorrelationCacheError(
                f"bounded input is too large: {relative}"
            )
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes + 1)):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise X13CorrelationCacheError(
                    f"bounded input is too large: {relative}"
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise X13CorrelationCacheError(
                f"input changed while being read: {relative}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _walk_bound_files(root_fd: int) -> set[str]:
    found: set[str] = set()

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        for name in os.listdir(directory_fd):
            status = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            relative = (*prefix, name)
            if stat.S_ISLNK(status.st_mode):
                raise X13CorrelationCacheError(
                    f"batch contains a symlink: {'/'.join(relative)}"
                )
            if stat.S_ISDIR(status.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_fd,
                )
                try:
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(status.st_mode):
                found.add("/".join(relative))
            else:
                raise X13CorrelationCacheError(
                    f"batch contains a special file: {'/'.join(relative)}"
                )

    visit(root_fd, ())
    return found


def _parse_checksum_ledger(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise X13CorrelationCacheError(
            "checksum ledger is not ASCII"
        ) from error
    entries: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise X13CorrelationCacheError("checksum ledger is malformed")
        digest, relative_text = line[:64], line[66:]
        relative = PurePosixPath(relative_text)
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or relative_text in entries
        ):
            raise X13CorrelationCacheError("checksum ledger is malformed")
        _safe_relative(relative)
        entries[relative_text] = digest
    return entries


def _sha256_canonical_game_file(path: Path) -> str:
    digest = hashlib.sha256()
    last = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            material = last + chunk
            if len(material) > 1:
                digest.update(material[:-1])
                last = material[-1:]
            else:
                last = material
    if last != b"\n":
        raise X13CorrelationCacheError(
            f"canonical game JSON lacks one terminal newline: {path.name}"
        )
    return f"sha256:{digest.hexdigest()}"


def _verify_association_snapshot(
    path: Path, descriptor: dict[str, object]
) -> None:
    parquet = pq.ParquetFile(path)
    if (
        parquet.metadata.num_rows != descriptor.get("row_count")
        or parquet.schema_arrow != association_arrow_schema()
        or descriptor.get("schema_fingerprint")
        != X13_ASSOCIATION_SCHEMA_FINGERPRINT
    ):
        raise X13CorrelationCacheError(
            f"association snapshot schema/row count mismatch: {path.name}"
        )


def _snapshot_and_verify_frozen_batch(
    batch_path: Path, snapshot_root: Path
) -> Path:
    """Strictly verify once and snapshot every input consumed downstream."""

    if (
        batch_path.is_symlink()
        or batch_path.name != X13_FROZEN_BATCH_ID.replace(":", "-")
    ):
        raise X13CorrelationCacheError(
            "only the frozen formal X-13 batch may be cached"
        )
    try:
        root_fd = os.open(
            batch_path,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise X13CorrelationCacheError(
            "frozen batch root is not a no-follow directory"
        ) from error
    snapshot = snapshot_root / batch_path.name
    snapshot.mkdir(parents=True)
    try:
        ledger_raw = _read_bound_bytes(
            root_fd,
            PurePosixPath("checksums.sha256"),
            maximum_bytes=16 << 20,
        )
        ledger = _parse_checksum_ledger(ledger_raw)
        actual = _walk_bound_files(root_fd)
        if actual != set(ledger) | {"checksums.sha256"}:
            raise X13CorrelationCacheError(
                "batch checksum ledger file set differs"
            )
        manifest_raw = _read_bound_bytes(
            root_fd,
            PurePosixPath("batch_manifest.json"),
            maximum_bytes=16 << 20,
        )
        try:
            manifest = json.loads(manifest_raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise X13CorrelationCacheError(
                "batch manifest is unreadable"
            ) from error
        if (
            type(manifest) is not dict
            or manifest_raw != _canonical_json_bytes(manifest)
            or manifest.get("schema") != "nfl_x13_batch_manifest_v1"
            or manifest.get("batch_id") != X13_FROZEN_BATCH_ID
            or manifest.get("game_ids") != list(X13_GAME_IDS)
            or manifest.get("game_count") != 20
            or manifest.get("plan_id") != X13_BATCH_SPEC.plan_id
            or manifest.get("universe_id") != X13_FROZEN_UNIVERSE_ID
            or manifest.get("publication_gate") != "PASS"
            or manifest.get("association_expected_row_count")
            != X13_FROZEN_CANDIDATE_ROW_COUNT
        ):
            raise X13CorrelationCacheError(
                "batch manifest does not prove the frozen batch"
            )
        identity_material = {
            field: manifest[field]
            for field in (
                "experiment_id",
                "status",
                "plan_id",
                "universe_id",
                "builder_version",
                "game_ids",
                "game_hashes",
                "raw_manifest_hashes",
                "auxiliary_artifacts",
                "association_expected_row_count",
            )
        }
        if canonical_sha256(identity_material) != X13_FROZEN_BATCH_ID:
            raise X13CorrelationCacheError(
                "batch manifest identity material does not match batch_id"
            )
        game_hashes = manifest.get("game_hashes")
        descriptors = manifest.get("auxiliary_artifacts")
        if (
            type(game_hashes) is not dict
            or set(game_hashes) != set(X13_GAME_IDS)
            or type(descriptors) is not list
        ):
            raise X13CorrelationCacheError(
                "batch manifest source identities are incomplete"
            )
        association_descriptors = [
            item
            for item in descriptors
            if type(item) is dict
            and item.get("role") == "association_candidate_table"
        ]
        if (
            len(association_descriptors)
            != X13_EXPECTED_ASSOCIATION_PARTITIONS
            or sum(int(item["row_count"]) for item in association_descriptors)
            != X13_FROZEN_CANDIDATE_ROW_COUNT
        ):
            raise X13CorrelationCacheError(
                "batch association descriptor universe is incomplete"
            )
        audit_relative = (
            "reports/contract-semantic-overlap-audit-v1.json"
        )
        consumed = {
            "batch_manifest.json",
            audit_relative,
            *(f"games/{game_id}.json" for game_id in X13_GAME_IDS),
            *(
                str(item["relative_path"])
                for item in association_descriptors
            ),
        }
        descriptor_by_path = {
            str(item["relative_path"]): item
            for item in descriptors
            if type(item) is dict and "relative_path" in item
        }
        if audit_relative not in descriptor_by_path:
            raise X13CorrelationCacheError(
                "semantic overlap audit is not manifest-bound"
            )
        for relative_text, expected in sorted(ledger.items()):
            relative = PurePosixPath(relative_text)
            descriptor = descriptor_by_path.get(relative_text)
            expected_length = (
                int(descriptor["byte_length"])
                if descriptor is not None
                else None
            )
            if relative_text in consumed:
                _copy_bound_input(
                    root_fd,
                    relative,
                    snapshot.joinpath(*relative.parts),
                    expected_sha256=expected,
                    expected_byte_length=expected_length,
                )
            else:
                observed, byte_length = _hash_bound_input(root_fd, relative)
                if observed != expected:
                    raise X13CorrelationCacheError(
                        f"batch checksum mismatch: {relative_text}"
                    )
                if (
                    expected_length is not None
                    and byte_length != expected_length
                ):
                    raise X13CorrelationCacheError(
                        f"batch byte length mismatch: {relative_text}"
                    )
        for game_id, expected in game_hashes.items():
            if _sha256_canonical_game_file(
                snapshot / "games" / f"{game_id}.json"
            ) != expected:
                raise X13CorrelationCacheError(
                    f"canonical game identity mismatch: {game_id}"
                )
        for descriptor in association_descriptors:
            relative = PurePosixPath(str(descriptor["relative_path"]))
            _verify_association_snapshot(
                snapshot.joinpath(*relative.parts), descriptor
            )
        return snapshot
    finally:
        os.close(root_fd)


def _root_arrays(
    path: Path,
    targets: frozenset[str],
) -> dict[str, list[object]]:
    """Capture selected root arrays without materializing giant sibling arrays."""

    if path.is_symlink() or not path.is_file():
        raise X13CorrelationCacheError(
            f"canonical game JSON is not a regular file: {path.name}"
        )
    found: dict[str, bytes] = {}
    object_depth = 0
    array_depth = 0
    in_string = False
    escaped = False
    key_buffer: bytearray | None = None
    pending_key: str | None = None
    expect_root_key = False
    capture_key: str | None = None
    capture: bytearray | None = None

    with path.open("rb") as stream:
        complete = False
        while chunk := stream.read(_READ_CHUNK_BYTES):
            for byte in chunk:
                char = chr(byte)
                if capture is not None:
                    capture.append(byte)
                    if len(capture) > _MAX_CAPTURE_BYTES:
                        raise X13CorrelationCacheError(
                            f"{capture_key} exceeds bounded JSON capture limit"
                        )

                if in_string:
                    if key_buffer is not None:
                        key_buffer.append(byte)
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                        if key_buffer is not None:
                            try:
                                pending_key = json.loads(
                                    bytes(key_buffer).decode("utf-8")
                                )
                            except (UnicodeError, json.JSONDecodeError) as error:
                                raise X13CorrelationCacheError(
                                    "canonical game JSON root key is invalid"
                                ) from error
                            key_buffer = None
                    continue

                if char == '"':
                    in_string = True
                    escaped = False
                    if (
                        object_depth == 1
                        and array_depth == 0
                        and expect_root_key
                    ):
                        key_buffer = bytearray(b'"')
                    continue
                if char == "{":
                    object_depth += 1
                    if object_depth == 1 and array_depth == 0:
                        expect_root_key = True
                    continue
                if char == "}":
                    object_depth -= 1
                    if object_depth < 0:
                        raise X13CorrelationCacheError(
                            "canonical game JSON object nesting is invalid"
                        )
                    continue
                if char == "[":
                    if (
                        object_depth == 1
                        and array_depth == 0
                        and pending_key in targets
                    ):
                        if pending_key in found:
                            raise X13CorrelationCacheError(
                                f"duplicate root array: {pending_key}"
                            )
                        capture_key = pending_key
                        capture = bytearray(b"[")
                    array_depth += 1
                    pending_key = None
                    continue
                if char == "]":
                    array_depth -= 1
                    if array_depth < 0:
                        raise X13CorrelationCacheError(
                            "canonical game JSON array nesting is invalid"
                        )
                    if capture is not None and array_depth == 0:
                        assert capture_key is not None
                        found[capture_key] = bytes(capture)
                        capture = None
                        capture_key = None
                        if set(found) == set(targets):
                            complete = True
                            break
                    continue
                if object_depth == 1 and array_depth == 0:
                    if char == ":" and pending_key is not None:
                        expect_root_key = False
                    elif char == ",":
                        pending_key = None
                        expect_root_key = True
            if complete:
                break

    if not complete and (
        in_string or object_depth != 0 or array_depth != 0 or capture is not None
    ):
        raise X13CorrelationCacheError(
            "canonical game JSON is truncated or structurally invalid"
        )
    if set(found) != set(targets):
        missing = ",".join(sorted(set(targets) - set(found)))
        raise X13CorrelationCacheError(
            f"canonical game JSON lacks required root arrays: {missing}"
        )
    decoded: dict[str, list[object]] = {}
    for key, raw in found.items():
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise X13CorrelationCacheError(
                f"canonical game JSON {key} array is invalid"
            ) from error
        if type(value) is not list:
            raise X13CorrelationCacheError(f"{key} must be a JSON array")
        decoded[key] = value
    return decoded


def _iter_root_object_array(path: Path, target: str) -> Iterator[dict[str, object]]:
    """Stream object items from one canonical root array."""

    object_depth = 0
    array_depth = 0
    in_string = False
    escaped = False
    key_buffer: bytearray | None = None
    pending_key: str | None = None
    expect_root_key = False
    in_target = False
    item: bytearray | None = None
    item_object_depth: int | None = None
    found = False
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            for byte in chunk:
                char = chr(byte)
                if item is not None:
                    item.append(byte)
                    if len(item) > _MAX_CAPTURE_BYTES:
                        raise X13CorrelationCacheError(
                            f"{target} item exceeds bounded JSON capture limit"
                        )
                if in_string:
                    if key_buffer is not None:
                        key_buffer.append(byte)
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                        if key_buffer is not None:
                            pending_key = json.loads(
                                bytes(key_buffer).decode("utf-8")
                            )
                            key_buffer = None
                    continue
                if char == '"':
                    in_string = True
                    if (
                        object_depth == 1
                        and array_depth == 0
                        and expect_root_key
                    ):
                        key_buffer = bytearray(b'"')
                    continue
                if char == "{":
                    object_depth += 1
                    if object_depth == 1 and array_depth == 0:
                        expect_root_key = True
                    elif in_target and item is None and array_depth == 1:
                        item = bytearray(b"{")
                        item_object_depth = object_depth
                    continue
                if char == "}":
                    if (
                        item is not None
                        and item_object_depth is not None
                        and object_depth == item_object_depth
                    ):
                        try:
                            value = json.loads(item)
                        except json.JSONDecodeError as error:
                            raise X13CorrelationCacheError(
                                f"{target} item is invalid JSON"
                            ) from error
                        if type(value) is not dict:
                            raise X13CorrelationCacheError(
                                f"{target} item is not an object"
                            )
                        yield value
                        item = None
                        item_object_depth = None
                    object_depth -= 1
                    continue
                if char == "[":
                    if (
                        object_depth == 1
                        and array_depth == 0
                        and pending_key == target
                    ):
                        if found:
                            raise X13CorrelationCacheError(
                                f"duplicate root array: {target}"
                            )
                        found = True
                        in_target = True
                    array_depth += 1
                    pending_key = None
                    continue
                if char == "]":
                    if in_target and array_depth == 1:
                        if item is not None:
                            raise X13CorrelationCacheError(
                                f"{target} array ends inside an item"
                            )
                        return
                    array_depth -= 1
                    continue
                if object_depth == 1 and array_depth == 0:
                    if char == ":" and pending_key is not None:
                        expect_root_key = False
                    elif char == ",":
                        pending_key = None
                        expect_root_key = True
    if not found:
        raise X13CorrelationCacheError(f"missing root array: {target}")
    raise X13CorrelationCacheError(f"truncated root array: {target}")


def _episode_superclass(episode: dict[str, object]) -> str:
    score_points = episode.get("score_points")
    turnover = episode.get("turnover")
    if type(score_points) is not int or type(turnover) is not bool:
        raise X13CorrelationCacheError(
            "episode score/turnover classification is invalid"
        )
    if score_points > 0:
        return "score"
    if turnover:
        return "turnover"
    return "routine"


def _signal_event_status(
    episode: dict[str, object],
) -> tuple[bool, str | None]:
    episode_type = episode.get("episode_type")
    score_points = episode.get("score_points")
    turnover = episode.get("turnover")
    if (
        type(episode_type) is not str
        or type(score_points) is not int
        or type(turnover) is not bool
    ):
        raise X13CorrelationCacheError(
            "episode signal classification is invalid"
        )
    if episode_type == "score_turnover":
        return False, "score_turnover_excluded"
    if score_points > 0:
        if episode_type in {"touchdown", "field_goal", "safety"}:
            return True, None
        return False, "unsupported_score_type"
    if turnover:
        if episode_type in {"pass", "run", "sack"}:
            return True, None
        return False, "unsupported_turnover_type"
    if episode_type in {"pass", "run", "sack"}:
        return True, None
    return False, f"{episode_type}_excluded"


def _orientation_value(subject: str, outcome: str) -> str:
    lowered = outcome.casefold()
    if lowered in {"yes", "over"}:
        return subject
    if lowered in {"no", "under"}:
        return f"NOT:{subject}"
    return outcome


def _semantic_orientation_id(
    game_id: str,
    contract: dict[str, object],
    outcome: str,
) -> str:
    return canonical_sha256(
        [
            game_id,
            contract["family"],
            contract["period"],
            contract["measure"],
            contract["comparator"],
            contract["line"],
            _orientation_value(str(contract["subject"]), outcome),
        ]
    )


def _extract_game_dimensions(
    path: Path,
    *,
    approved_orientation_ids: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    arrays = _root_arrays(path, frozenset({"contracts", "episodes"}))
    episodes: list[dict[str, object]] = []
    episode_ids: set[tuple[str, str]] = set()
    for raw in arrays["episodes"]:
        if type(raw) is not dict:
            raise X13CorrelationCacheError("episode dimension row is invalid")
        game_id = raw.get("game_id")
        episode_id = raw.get("episode_id")
        pre_state = raw.get("pre_state")
        if type(game_id) is not str or type(episode_id) is not str:
            raise X13CorrelationCacheError(
                "episode dimension identity is invalid"
            )
        key = (game_id, episode_id)
        if key in episode_ids:
            raise X13CorrelationCacheError(
                "episode dimension contains duplicate identity"
            )
        episode_ids.add(key)
        if pre_state is None:
            state: dict[str, object] = {}
            pre_state_present = False
        elif type(pre_state) is dict:
            state = pre_state
            pre_state_present = True
        else:
            raise X13CorrelationCacheError("episode pre_state is invalid")
        home_score = state.get("home_score")
        away_score = state.get("away_score")
        if pre_state_present and (
            type(home_score) is not int or type(away_score) is not int
        ):
            raise X13CorrelationCacheError(
                "episode pre_state score is invalid"
            )
        score_points = raw.get("score_points")
        turnover = raw.get("turnover")
        signal_event_eligible, signal_exclusion_reason = (
            _signal_event_status(raw)
        )
        if signal_event_eligible:
            if type(score_points) is int and score_points > 0:
                signal_focal_team = raw.get("scoring_team")
            elif turnover:
                signal_focal_team = state.get("defense_team")
            else:
                signal_focal_team = state.get("offense_team")
            if type(signal_focal_team) is not str:
                signal_event_eligible = False
                signal_exclusion_reason = "missing_signal_focal_team"
                signal_focal_team = None
        else:
            signal_focal_team = None
        episodes.append(
            {
                "game_id": game_id,
                "episode_id": episode_id,
                "episode_superclass": _episode_superclass(raw),
                "episode_is_score": (
                    type(score_points) is int and score_points > 0
                ),
                "episode_is_turnover": turnover,
                "signal_event_eligible": signal_event_eligible,
                "signal_exclusion_reason": signal_exclusion_reason,
                "signal_focal_team": signal_focal_team,
                "scoring_team": raw.get("scoring_team"),
                "pre_offense_team": state.get("offense_team"),
                "pre_defense_team": state.get("defense_team"),
                "pre_state_present": pre_state_present,
                "pre_game_seconds_remaining": state.get(
                    "game_seconds_remaining"
                ),
                "pre_quarter": state.get("quarter"),
                "pre_score_margin_home": (
                    home_score - away_score
                    if type(home_score) is int and type(away_score) is int
                    else None
                ),
                "pre_possession_team": state.get("possession_team"),
                "pre_down": state.get("down"),
                "pre_distance": state.get("distance"),
                "pre_yardline_100": state.get("yardline_100"),
                "pre_goal_to_go": state.get("goal_to_go"),
            }
        )

    contracts: list[dict[str, object]] = []
    contract_orientations: set[tuple[str, str, str]] = set()
    for raw in arrays["contracts"]:
        if type(raw) is not dict:
            raise X13CorrelationCacheError("contract dimension row is invalid")
        contract_id = raw.get("contract_id")
        dependencies = raw.get("dependency_game_ids")
        outcomes = raw.get("outcomes")
        if (
            type(contract_id) is not str
            or type(dependencies) is not list
            or len(dependencies) != 1
            or type(dependencies[0]) is not str
            or type(outcomes) is not list
            or not outcomes
            or any(type(outcome) is not str for outcome in outcomes)
        ):
            raise X13CorrelationCacheError(
                "contract dimension identity is invalid"
            )
        game_id = dependencies[0]
        required_strings = ("family", "period", "measure", "subject", "venue")
        if any(type(raw.get(field)) is not str for field in required_strings):
            raise X13CorrelationCacheError(
                "contract structured semantics are incomplete"
            )
        comparator = raw.get("comparator")
        line = raw.get("line")
        if comparator is not None and type(comparator) is not str:
            raise X13CorrelationCacheError("contract comparator is invalid")
        if line is not None and type(line) is not str:
            raise X13CorrelationCacheError("contract line is invalid")
        for outcome_value in outcomes:
            outcome = str(outcome_value)
            key = (game_id, contract_id, outcome)
            if key in contract_orientations:
                raise X13CorrelationCacheError(
                    "contract orientation contains duplicate identity"
                )
            contract_orientations.add(key)
            orientation_id = _semantic_orientation_id(
                game_id, raw, outcome
            )
            contracts.append(
                {
                    "game_id": game_id,
                    "contract_id": contract_id,
                    "outcome": outcome,
                    "contract_subject": raw["subject"],
                    "contract_period": raw["period"],
                    "contract_line": line,
                    "contract_measure": raw["measure"],
                    "contract_comparator": comparator,
                    "contract_family": raw["family"],
                    "contract_venue": raw["venue"],
                    "semantic_orientation_id": orientation_id,
                    "cross_venue_eligible": (
                        orientation_id in approved_orientation_ids
                    ),
                }
            )
    return episodes, contracts


_EPISODE_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("episode_superclass", pa.string(), nullable=False),
        pa.field("episode_is_score", pa.bool_(), nullable=False),
        pa.field("episode_is_turnover", pa.bool_(), nullable=False),
        pa.field("signal_event_eligible", pa.bool_(), nullable=False),
        pa.field("signal_exclusion_reason", pa.string()),
        pa.field("signal_focal_team", pa.string()),
        pa.field("scoring_team", pa.string()),
        pa.field("pre_offense_team", pa.string()),
        pa.field("pre_defense_team", pa.string()),
        pa.field("pre_state_present", pa.bool_(), nullable=False),
        pa.field("pre_game_seconds_remaining", pa.int32()),
        pa.field("pre_quarter", pa.int32()),
        pa.field("pre_score_margin_home", pa.int32()),
        pa.field("pre_possession_team", pa.string()),
        pa.field("pre_down", pa.int32()),
        pa.field("pre_distance", pa.int32()),
        pa.field("pre_yardline_100", pa.int32()),
        pa.field("pre_goal_to_go", pa.bool_()),
    ]
)
_CONTRACT_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("contract_id", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("contract_subject", pa.string(), nullable=False),
        pa.field("contract_period", pa.string(), nullable=False),
        pa.field("contract_line", pa.string()),
        pa.field("contract_measure", pa.string(), nullable=False),
        pa.field("contract_comparator", pa.string()),
        pa.field("contract_family", pa.string(), nullable=False),
        pa.field("contract_venue", pa.string(), nullable=False),
        pa.field("semantic_orientation_id", pa.string(), nullable=False),
        pa.field("cross_venue_eligible", pa.bool_(), nullable=False),
    ]
)
_MONEYLINE_TRADE_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("contract_id", pa.string(), nullable=False),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("source_start", pa.string(), nullable=False),
        pa.field("source_end", pa.string(), nullable=False),
        pa.field("price", pa.string(), nullable=False),
        pa.field("observation_id", pa.string(), nullable=False),
    ]
)


def _load_manifest(batch_path: Path) -> dict[str, object]:
    try:
        raw = (batch_path / "batch_manifest.json").read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X13CorrelationCacheError("batch manifest is unreadable") from error
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise X13CorrelationCacheError("batch manifest is not canonical JSON")
    return value


def _association_paths(
    batch_path: Path, manifest: dict[str, object]
) -> tuple[list[Path], int]:
    descriptors = manifest.get("auxiliary_artifacts")
    if type(descriptors) is not list:
        raise X13CorrelationCacheError(
            "batch manifest lacks auxiliary artifacts"
        )
    paths: list[Path] = []
    candidate_rows = 0
    for descriptor in descriptors:
        if type(descriptor) is not dict:
            raise X13CorrelationCacheError(
                "batch auxiliary descriptor is invalid"
            )
        if descriptor.get("role") != "association_candidate_table":
            continue
        relative_text = descriptor.get("relative_path")
        row_count = descriptor.get("row_count")
        if type(relative_text) is not str or type(row_count) is not int:
            raise X13CorrelationCacheError(
                "association descriptor is incomplete"
            )
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise X13CorrelationCacheError(
                "association descriptor path is unsafe"
            )
        paths.append(batch_path.joinpath(*relative.parts))
        candidate_rows += row_count
    if not paths:
        raise X13CorrelationCacheError(
            "batch has no association candidate partitions"
        )
    return sorted(paths), candidate_rows


def _approved_orientation_ids(batch_path: Path) -> frozenset[str]:
    path = batch_path / "reports/contract-semantic-overlap-audit-v1.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X13CorrelationCacheError(
            "contract semantic overlap audit is unreadable"
        ) from error
    if (
        type(value) is not dict
        or raw != _canonical_json_bytes(value)
        or value.get("schema")
        != "nfl_x13_contract_semantic_overlap_audit_v1"
        or value.get("status") != "PASS"
        or value.get("unapproved_cross_venue_orientation_count") != 0
    ):
        raise X13CorrelationCacheError(
            "contract semantic overlap audit is not passing"
        )
    ids = value.get("approved_cross_venue_semantic_key_sha256s")
    if (
        type(ids) is not list
        or any(
            type(item) is not str
            or not item.startswith("sha256:")
            or len(item) != 71
            for item in ids
        )
        or len(ids) != len(set(ids))
    ):
        raise X13CorrelationCacheError(
            "approved semantic orientation identities are invalid"
        )
    return frozenset(ids)


def _write_dimensions(
    batch_path: Path,
    stage: Path,
    approved_orientation_ids: frozenset[str],
) -> tuple[Path, Path, Path]:
    episode_path = stage / ".episode-dimension.parquet"
    contract_path = stage / ".contract-dimension.parquet"
    moneyline_trade_path = stage / ".moneyline-trades.parquet"
    episode_writer = pq.ParquetWriter(
        episode_path, _EPISODE_SCHEMA, compression="zstd"
    )
    contract_writer = pq.ParquetWriter(
        contract_path, _CONTRACT_SCHEMA, compression="zstd"
    )
    trade_writer = pq.ParquetWriter(
        moneyline_trade_path, _MONEYLINE_TRADE_SCHEMA, compression="zstd"
    )
    try:
        game_paths = sorted((batch_path / "games").glob("*.json"))
        if not game_paths:
            raise X13CorrelationCacheError(
                "batch contains no canonical game JSON"
            )
        for game_path in game_paths:
            episodes, contracts = _extract_game_dimensions(
                game_path,
                approved_orientation_ids=approved_orientation_ids,
            )
            episode_writer.write_table(
                pa.Table.from_pylist(episodes, schema=_EPISODE_SCHEMA)
            )
            contract_writer.write_table(
                pa.Table.from_pylist(contracts, schema=_CONTRACT_SCHEMA)
            )
            arrays = _root_arrays(game_path, frozenset({"contracts"}))
            raw_to_contract: dict[str, tuple[str, frozenset[str]]] = {}
            for raw_contract in arrays["contracts"]:
                if (
                    type(raw_contract) is not dict
                    or raw_contract.get("family") != "moneyline"
                ):
                    continue
                contract_id = raw_contract.get("contract_id")
                raw_ids = raw_contract.get("raw_contract_ids")
                outcomes = raw_contract.get("outcomes")
                if (
                    type(contract_id) is not str
                    or type(raw_ids) is not list
                    or type(outcomes) is not list
                    or any(type(item) is not str for item in raw_ids)
                    or any(type(item) is not str for item in outcomes)
                ):
                    raise X13CorrelationCacheError(
                        "moneyline contract binding is incomplete"
                    )
                for raw_id in raw_ids:
                    if raw_id in raw_to_contract:
                        raise X13CorrelationCacheError(
                            "moneyline raw market identity is ambiguous"
                        )
                    raw_to_contract[raw_id] = (
                        contract_id,
                        frozenset(str(item) for item in outcomes),
                    )
            trade_rows: list[dict[str, object]] = []
            for observation in _iter_root_object_array(
                game_path, "observations"
            ):
                if not (
                    observation.get("kind") == "trade"
                    and observation.get("provenance") == "observed"
                    and observation.get("price") is not None
                    and observation.get("primary_path_eligible") is True
                ):
                    continue
                binding = raw_to_contract.get(
                    str(observation.get("raw_market_id"))
                )
                if binding is None:
                    continue
                contract_id, contract_outcomes = binding
                outcome = observation.get("outcome")
                if type(outcome) is not str or outcome not in contract_outcomes:
                    raise X13CorrelationCacheError(
                        "moneyline observation outcome is not canonical"
                    )
                row = {
                    "game_id": game_path.stem,
                    "contract_id": contract_id,
                    "outcome": outcome,
                    "source_start": observation.get("source_start"),
                    "source_end": observation.get("source_end"),
                    "price": observation.get("price"),
                    "observation_id": observation.get("observation_id"),
                }
                if any(type(value) is not str for value in row.values()):
                    raise X13CorrelationCacheError(
                        "moneyline observation is incomplete"
                    )
                trade_rows.append(row)
                if len(trade_rows) >= 20_000:
                    trade_writer.write_table(
                        pa.Table.from_pylist(
                            trade_rows, schema=_MONEYLINE_TRADE_SCHEMA
                        )
                    )
                    trade_rows.clear()
            if trade_rows:
                trade_writer.write_table(
                    pa.Table.from_pylist(
                        trade_rows, schema=_MONEYLINE_TRADE_SCHEMA
                    )
                )
    finally:
        episode_writer.close()
        contract_writer.close()
        trade_writer.close()
    return episode_path, contract_path, moneyline_trade_path


def _duckdb_connection(stage: Path) -> duckdb.DuckDBPyConnection:
    scratch = stage / ".duckdb-tmp"
    scratch.mkdir()
    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET temp_directory = ?", [str(scratch)])
    return connection


def _materialize_verified_cache(
    batch_path: Path,
    stage: Path,
    *,
    expected_observed_rows: int,
) -> dict[str, object]:
    """Materialize from an already verified immutable batch into ``stage``."""

    manifest = _load_manifest(batch_path)
    paths, candidate_rows = _association_paths(batch_path, manifest)
    approved = _approved_orientation_ids(batch_path)
    episode_path, contract_path, moneyline_trade_path = _write_dimensions(
        batch_path, stage, approved
    )
    output_path = stage / "x13_reaction_observed_cache_v1.parquet"
    coverage_parquet = stage / "x13_reaction_coverage_matrix_v1.parquet"
    connection = _duckdb_connection(stage)
    try:
        connection.from_parquet(
            [str(path) for path in paths]
        ).create_view("associations")
        connection.from_parquet(str(episode_path)).create_view("episodes")
        connection.from_parquet(str(contract_path)).create_view("contracts")
        connection.from_parquet(str(moneyline_trade_path)).create_view(
            "moneyline_trades"
        )
        observed_count = connection.execute(
            """
            SELECT count(*)
            FROM associations
            WHERE validity_status = 'OBSERVED'
              AND contaminated = false
              AND order_ambiguous = false
            """
        ).fetchone()[0]
        if observed_count != expected_observed_rows:
            raise X13CorrelationCacheError(
                "observed association row count differs from frozen "
                f"expectation: {observed_count} != {expected_observed_rows}"
            )
        join_audit = connection.execute(
            """
            SELECT
              count(*) FILTER (WHERE e.episode_id IS NULL),
              count(*) FILTER (WHERE c.contract_id IS NULL),
              count(*) FILTER (
                WHERE e.episode_id IS NOT NULL
                  AND NOT e.pre_state_present
              ),
              count(*) FILTER (
                WHERE c.contract_id IS NOT NULL
                  AND (
                    a.family != c.contract_family
                    OR a.venue != c.contract_venue
                  )
              )
            FROM associations a
            LEFT JOIN episodes e USING (game_id, episode_id)
            LEFT JOIN contracts c USING (game_id, contract_id, outcome)
            """
        ).fetchone()
        if any(int(value) != 0 for value in join_audit):
            raise X13CorrelationCacheError(
                "episode/contract referential integrity failed: "
                + ",".join(str(value) for value in join_audit)
            )
        connection.execute(
            """
            CREATE TEMP TABLE moneyline_pre AS
            WITH keys AS (
              SELECT DISTINCT
                game_id,
                episode_id,
                contract_id,
                outcome,
                delay_scenario_seconds,
                source_time_start_utc,
                pre_event_actual_trade
              FROM associations
              WHERE validity_status = 'OBSERVED'
                AND contaminated = false
                AND order_ambiguous = false
                AND family = 'moneyline'
            ),
            latest AS (
              SELECT
                k.*,
                max(t.source_start) AS latest_source_start
              FROM keys k
              INNER JOIN moneyline_trades t
                ON t.game_id = k.game_id
               AND t.contract_id = k.contract_id
               AND t.outcome = k.outcome
               AND CAST(t.source_end AS TIMESTAMPTZ)
                   <= CAST(k.source_time_start_utc AS TIMESTAMPTZ)
              GROUP BY ALL
            )
            SELECT
              l.game_id,
              l.episode_id,
              l.contract_id,
              l.outcome,
              l.delay_scenario_seconds,
              l.source_time_start_utc,
              l.pre_event_actual_trade,
              count(*)::BIGINT AS latest_trade_count,
              min(t.price) AS reconstructed_pre_trade,
              (
                epoch_us(CAST(l.source_time_start_utc AS TIMESTAMPTZ))
                - epoch_us(CAST(min(t.source_end) AS TIMESTAMPTZ))
              )::BIGINT AS pre_trade_age_microseconds
            FROM latest l
            INNER JOIN moneyline_trades t
              ON t.game_id = l.game_id
             AND t.contract_id = l.contract_id
             AND t.outcome = l.outcome
             AND t.source_start = l.latest_source_start
             AND CAST(t.source_end AS TIMESTAMPTZ)
                 <= CAST(l.source_time_start_utc AS TIMESTAMPTZ)
            GROUP BY
              l.game_id,
              l.episode_id,
              l.contract_id,
              l.outcome,
              l.delay_scenario_seconds,
              l.source_time_start_utc,
              l.pre_event_actual_trade
            """
        )
        pre_age_audit = connection.execute(
            """
            WITH keys AS (
              SELECT DISTINCT
                game_id,
                episode_id,
                contract_id,
                outcome,
                delay_scenario_seconds,
                source_time_start_utc,
                pre_event_actual_trade
              FROM associations
              WHERE validity_status = 'OBSERVED'
                AND contaminated = false
                AND order_ambiguous = false
                AND family = 'moneyline'
            )
            SELECT
              count(*) FILTER (WHERE p.contract_id IS NULL),
              count(*) FILTER (WHERE p.latest_trade_count != 1),
              count(*) FILTER (
                WHERE CAST(p.reconstructed_pre_trade AS DECIMAL(18, 8))
                      != CAST(k.pre_event_actual_trade AS DECIMAL(18, 8))
              ),
              count(*) FILTER (
                WHERE p.pre_trade_age_microseconds < 0
              )
            FROM keys k
            LEFT JOIN moneyline_pre p USING (
              game_id,
              episode_id,
              contract_id,
              outcome,
              delay_scenario_seconds,
              source_time_start_utc,
              pre_event_actual_trade
            )
            """
        ).fetchone()
        if any(int(value) != 0 for value in pre_age_audit):
            raise X13CorrelationCacheError(
                "moneyline pre-trade age reconstruction failed: "
                + ",".join(str(value) for value in pre_age_audit)
            )
        copy_query = """
            COPY (
              SELECT
                a.*,
                e.episode_superclass,
                e.episode_is_score,
                e.episode_is_turnover,
                e.signal_event_eligible,
                e.signal_exclusion_reason,
                e.signal_focal_team,
                e.scoring_team,
                e.pre_offense_team,
                e.pre_defense_team,
                e.pre_game_seconds_remaining,
                e.pre_quarter,
                e.pre_score_margin_home,
                e.pre_possession_team,
                e.pre_down,
                e.pre_distance,
                e.pre_yardline_100,
                e.pre_goal_to_go,
                c.contract_subject,
                c.contract_period,
                c.contract_line,
                c.contract_measure,
                c.contract_comparator,
                c.semantic_orientation_id,
                c.cross_venue_eligible,
                CASE
                  WHEN a.family = 'moneyline' AND p.contract_id IS NOT NULL
                  THEN p.pre_trade_age_microseconds / 1000000.0
                  ELSE NULL
                END AS pre_trade_age_seconds,
                CASE
                  WHEN a.validity_status != 'OBSERVED'
                    OR a.contaminated
                    OR a.order_ambiguous
                  THEN 'NOT_APPLICABLE_INVALID_CELL'
                  WHEN a.family = 'moneyline' AND p.contract_id IS NOT NULL
                  THEN 'RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS'
                  ELSE 'UNKNOWN_NOT_PERSISTED_IN_FROZEN_ASSOCIATION'
                END AS pre_trade_age_status,
                CASE
                  WHEN a.validity_status = 'OBSERVED'
                    AND NOT a.contaminated
                    AND NOT a.order_ambiguous
                    AND a.family = 'moneyline'
                    AND p.contract_id IS NOT NULL
                  THEN p.pre_trade_age_microseconds <= 60000000
                  ELSE NULL
                END AS pre_trade_fresh_le_60s,
                COALESCE((
                  a.validity_status = 'OBSERVED'
                  AND NOT a.contaminated
                  AND NOT a.order_ambiguous
                  AND e.signal_event_eligible
                  AND a.family = 'moneyline'
                  AND p.pre_trade_age_microseconds <= 60000000
                ), false) AS signal_candidate_eligible
              FROM associations a
              INNER JOIN episodes e USING (game_id, episode_id)
              INNER JOIN contracts c USING (game_id, contract_id, outcome)
              LEFT JOIN moneyline_pre p USING (
                game_id,
                episode_id,
                contract_id,
                outcome,
                delay_scenario_seconds,
                source_time_start_utc,
                pre_event_actual_trade
              )
              ORDER BY
                a.game_id,
                a.episode_id,
                a.venue,
                a.family,
                a.contract_id,
                a.outcome,
                a.delay_scenario_seconds,
                a.horizon_seconds
            ) TO ? (
              FORMAT PARQUET,
              COMPRESSION ZSTD,
              ROW_GROUP_SIZE 100000
            )
        """
        connection.execute(copy_query, [str(output_path)])
        connection.from_parquet(str(output_path)).create_view(
            "observed_cache"
        )
        coverage_query = """
            COPY (
              SELECT
                game_id,
                episode_id,
                episode_superclass,
                venue,
                family,
                validity_status,
                order_ambiguous,
                contaminated,
                count(*)::BIGINT AS row_count,
                count(
                  DISTINCT (contract_id, outcome)
                )::BIGINT AS distinct_contract_outcome_count,
                count(
                  DISTINCT (delay_scenario_seconds, horizon_seconds)
                )::BIGINT AS delay_horizon_cell_count,
                (
                  count(*) - count(DISTINCT (contract_id, outcome))
                )::BIGINT AS parameter_repetition_row_count
              FROM observed_cache
              GROUP BY
                game_id,
                episode_id,
                episode_superclass,
                venue,
                family,
                validity_status,
                order_ambiguous,
                contaminated
              ORDER BY
                game_id,
                episode_id,
                venue,
                family,
                validity_status,
                order_ambiguous,
                contaminated
            ) TO ? (
              FORMAT PARQUET,
              COMPRESSION ZSTD,
              ROW_GROUP_SIZE 100000
            )
        """
        connection.execute(coverage_query, [str(coverage_parquet)])
        counts = connection.execute(
            """
            SELECT
              count(DISTINCT game_id),
              count(DISTINCT (game_id, episode_id)),
              count(DISTINCT (game_id, episode_id, venue)),
              count(
                DISTINCT (game_id, episode_id, venue, contract_id, outcome)
              ),
              count(DISTINCT venue),
              count(DISTINCT contract_id),
              count(DISTINCT semantic_orientation_id),
              count(*) FILTER (WHERE cross_venue_eligible),
              count(*) FILTER (
                WHERE pre_trade_age_status =
                  'RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS'
              ),
              count(*) FILTER (WHERE signal_candidate_eligible)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()
        grids = connection.execute(
            """
            SELECT
              list(DISTINCT delay_scenario_seconds ORDER BY delay_scenario_seconds),
              list(DISTINCT horizon_seconds ORDER BY horizon_seconds)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()
        coverage_rows = connection.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(coverage_parquet)],
        ).fetchone()[0]
        published_cell_rows = connection.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(output_path)],
        ).fetchone()[0]
    finally:
        connection.close()
        shutil.rmtree(stage / ".duckdb-tmp", ignore_errors=True)
        episode_path.unlink(missing_ok=True)
        contract_path.unlink(missing_ok=True)
        moneyline_trade_path.unlink(missing_ok=True)

    reaction_units = int(counts[3])
    coverage = {
        "batch_id": manifest.get("batch_id"),
        "candidate_row_count": candidate_rows,
        "claim_scope": "DESCRIPTIVE_CACHE_ONLY",
        "cross_venue_eligible_row_count": int(counts[7]),
        "delay_scenarios_seconds": list(grids[0]),
        "experiment_id": "X-13",
        "game_count": int(counts[0]),
        "game_episode_venue_unit_count": int(counts[2]),
        "horizons_seconds": list(grids[1]),
        "independent_sports_event_count": int(counts[1]),
        "inference_result_count": 0,
        "pre_trade_age_known_row_count": int(counts[8]),
        "pre_trade_age_policy": (
            "moneyline age is reconstructed from frozen normalized observed "
            "trades and price-matched to the association; all other families "
            "remain UNKNOWN and cannot receive a signal candidate label"
        ),
        "observed_row_count": observed_count,
        "published_cell_row_count": int(published_cell_rows),
        "parameter_repetition_row_count": observed_count - reaction_units,
        "reaction_unit_count": reaction_units,
        "schema": X13_COVERAGE_SCHEMA,
        "semantic_orientation_count": int(counts[6]),
        "signal_candidate_eligible_row_count": int(counts[9]),
        "source_filter": (
            "validity_status='OBSERVED' AND contaminated=false "
            "AND order_ambiguous=false"
        ),
        "venue_count": int(counts[4]),
        "contract_count": int(counts[5]),
        "coverage_matrix_row_count": int(coverage_rows),
        "row_unit_warning": (
            "association candidate and observed rows repeat registered "
            "delay/horizon parameter cells; they are not independent samples"
        ),
    }
    coverage_json = stage / "x13_reaction_coverage_matrix_v1.json"
    coverage_json.write_bytes(_canonical_json_bytes(coverage))
    return {
        "cache_parquet_sha256": _sha256_file(output_path),
        "cache_parquet_bytes": output_path.stat().st_size,
        "coverage_parquet_sha256": _sha256_file(coverage_parquet),
        "coverage_parquet_bytes": coverage_parquet.stat().st_size,
        "coverage_json_sha256": _sha256_file(coverage_json),
        "coverage_json_bytes": coverage_json.stat().st_size,
        "observed_row_count": observed_count,
        "published_cell_row_count": int(published_cell_rows),
        "candidate_row_count": candidate_rows,
        "input_partition_count": len(paths),
        "coverage_matrix_row_count": int(coverage_rows),
        "delay_scenarios_seconds": list(grids[0]),
        "horizons_seconds": list(grids[1]),
    }


def _build_cache_manifest(
    identities: dict[str, object],
    *,
    batch_id: str,
    candidate_row_count: int,
    cross_venue_orientation_count: int,
) -> dict[str, object]:
    artifacts = {
        "observed_cache": {
            "byte_length": identities["cache_parquet_bytes"],
            "relative_path": "x13_reaction_observed_cache_v1.parquet",
            "row_count": identities["published_cell_row_count"],
            "schema": X13_CACHE_SCHEMA,
            "sha256": identities["cache_parquet_sha256"],
        },
        "coverage_matrix_parquet": {
            "byte_length": identities["coverage_parquet_bytes"],
            "relative_path": "x13_reaction_coverage_matrix_v1.parquet",
            "row_count": identities["coverage_matrix_row_count"],
            "schema": X13_COVERAGE_SCHEMA,
            "sha256": identities["coverage_parquet_sha256"],
        },
        "coverage_matrix_json": {
            "byte_length": identities["coverage_json_bytes"],
            "relative_path": "x13_reaction_coverage_matrix_v1.json",
            "row_count": 1,
            "schema": X13_COVERAGE_SCHEMA,
            "sha256": identities["coverage_json_sha256"],
        },
    }
    manifest: dict[str, object] = {
        "artifacts": artifacts,
        "batch_id": batch_id,
        "candidate_row_count": candidate_row_count,
        "claim_scope": "DESCRIPTIVE_CACHE_ONLY",
        "cross_venue_orientation_count": cross_venue_orientation_count,
        "experiment_id": "X-13",
        "inference_result_count": 0,
        "observed_row_count": identities["observed_row_count"],
        "schema": "nfl_x13_correlation_cache_manifest_v1",
        "source_filter": _SOURCE_FILTER,
    }
    manifest["cache_id"] = canonical_sha256(manifest)
    return manifest


def _expected_arrow_schema(
    fields: tuple[tuple[str, str], ...],
) -> pa.Schema:
    types = {
        "string": pa.string(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
        "double": pa.float64(),
    }
    return pa.schema([pa.field(name, types[type_name]) for name, type_name in fields])


def _verify_cache_directory(
    path: Path, *, require_frozen: bool = True
) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise X13CorrelationCacheError(
            "immutable cache directory is incomplete"
        )
    children = list(path.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise X13CorrelationCacheError(
            "immutable cache contains an unsafe entry"
        )
    manifest_path = path / "manifest.json"
    ledger_path = path / "checksums.sha256"
    if not manifest_path.is_file() or not ledger_path.is_file():
        raise X13CorrelationCacheError(
            "immutable cache directory is incomplete"
        )
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X13CorrelationCacheError("cache manifest is unreadable") from error
    required_manifest_fields = {
        "artifacts",
        "batch_id",
        "cache_id",
        "candidate_row_count",
        "claim_scope",
        "cross_venue_orientation_count",
        "experiment_id",
        "inference_result_count",
        "observed_row_count",
        "schema",
        "source_filter",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != required_manifest_fields
        or manifest_raw != _canonical_json_bytes(manifest)
        or manifest.get("schema") != "nfl_x13_correlation_cache_manifest_v1"
        or manifest.get("claim_scope") != "DESCRIPTIVE_CACHE_ONLY"
        or manifest.get("experiment_id") != "X-13"
        or manifest.get("inference_result_count") != 0
        or manifest.get("source_filter") != _SOURCE_FILTER
    ):
        raise X13CorrelationCacheError("cache manifest structure is invalid")
    supplied_cache_id = manifest.get("cache_id")
    identity_material = {
        key: value for key, value in manifest.items() if key != "cache_id"
    }
    recomputed_cache_id = canonical_sha256(identity_material)
    if (
        supplied_cache_id != recomputed_cache_id
        or path.name != recomputed_cache_id.replace(":", "-")
    ):
        raise X13CorrelationCacheError(
            "cache_id does not match complete manifest material"
        )
    if require_frozen and supplied_cache_id != X13_APPROVED_CACHE_ID:
        raise X13CorrelationCacheError(
            "cache_id is not the approved cache identity"
        )
    if require_frozen and (
        manifest.get("batch_id") != X13_FROZEN_BATCH_ID
        or manifest.get("candidate_row_count")
        != X13_FROZEN_CANDIDATE_ROW_COUNT
        or manifest.get("observed_row_count")
        != X13_EXPECTED_OBSERVED_ROW_COUNT
        or manifest.get("cross_venue_orientation_count")
        != X13_EXPECTED_CROSS_VENUE_ORIENTATIONS
    ):
        raise X13CorrelationCacheError(
            "cache manifest does not bind the frozen formal projection"
        )
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "observed_cache": (
            "x13_reaction_observed_cache_v1.parquet",
            X13_CACHE_SCHEMA,
        ),
        "coverage_matrix_parquet": (
            "x13_reaction_coverage_matrix_v1.parquet",
            X13_COVERAGE_SCHEMA,
        ),
        "coverage_matrix_json": (
            "x13_reaction_coverage_matrix_v1.json",
            X13_COVERAGE_SCHEMA,
        ),
    }
    if type(artifacts) is not dict or set(artifacts) != set(expected_artifacts):
        raise X13CorrelationCacheError(
            "cache manifest artifact set is invalid"
        )
    artifact_paths: dict[str, Path] = {}
    for role, (relative, schema_name) in expected_artifacts.items():
        descriptor = artifacts.get(role)
        if (
            type(descriptor) is not dict
            or set(descriptor)
            != {
                "byte_length",
                "relative_path",
                "row_count",
                "schema",
                "sha256",
            }
            or descriptor.get("relative_path") != relative
            or descriptor.get("schema") != schema_name
            or type(descriptor.get("byte_length")) is not int
            or int(descriptor["byte_length"]) < 0
            or type(descriptor.get("row_count")) is not int
            or int(descriptor["row_count"]) < 0
            or type(descriptor.get("sha256")) is not str
            or len(str(descriptor["sha256"])) != 71
        ):
            raise X13CorrelationCacheError(
                f"cache manifest artifact descriptor is invalid: {role}"
            )
        artifact_paths[role] = path / relative
    expected_files = {
        "manifest.json",
        "checksums.sha256",
        *(target.name for target in artifact_paths.values()),
    }
    if {child.name for child in children} != expected_files:
        raise X13CorrelationCacheError(
            "cache directory file set differs from manifest"
        )
    ledger = _parse_checksum_ledger(ledger_path.read_bytes())
    ledger_files = expected_files - {"checksums.sha256"}
    if set(ledger) != ledger_files:
        raise X13CorrelationCacheError(
            "cache checksum ledger file set differs"
        )
    for relative, expected in ledger.items():
        target = path / relative
        if _sha256_file(target) != f"sha256:{expected}":
            raise X13CorrelationCacheError(
                f"cache checksum mismatch: {relative}"
            )
    for role, target in artifact_paths.items():
        descriptor = artifacts[role]
        if (
            target.stat().st_size != descriptor["byte_length"]
            or _sha256_file(target) != descriptor["sha256"]
        ):
            raise X13CorrelationCacheError(
                f"cache artifact SHA/length mismatch: {role}"
            )
    observed_path = artifact_paths["observed_cache"]
    coverage_path = artifact_paths["coverage_matrix_parquet"]
    observed_parquet = pq.ParquetFile(observed_path)
    coverage_parquet = pq.ParquetFile(coverage_path)
    if (
        observed_parquet.metadata.num_rows
        != artifacts["observed_cache"]["row_count"]
        or observed_parquet.schema_arrow
        != _expected_arrow_schema(_OBSERVED_CACHE_FIELDS)
    ):
        raise X13CorrelationCacheError(
            "observed cache schema/row_count is invalid"
        )
    if (
        coverage_parquet.metadata.num_rows
        != artifacts["coverage_matrix_parquet"]["row_count"]
        or coverage_parquet.schema_arrow
        != _expected_arrow_schema(_COVERAGE_FIELDS)
    ):
        raise X13CorrelationCacheError(
            "coverage parquet schema/row_count is invalid"
        )
    coverage_json_path = artifact_paths["coverage_matrix_json"]
    try:
        coverage_raw = coverage_json_path.read_bytes()
        coverage = json.loads(coverage_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X13CorrelationCacheError(
            "coverage JSON is unreadable"
        ) from error
    if (
        artifacts["coverage_matrix_json"]["row_count"] != 1
        or type(coverage) is not dict
        or coverage_raw != _canonical_json_bytes(coverage)
        or coverage.get("schema") != X13_COVERAGE_SCHEMA
        or coverage.get("batch_id") != manifest.get("batch_id")
        or coverage.get("candidate_row_count")
        != manifest.get("candidate_row_count")
        or coverage.get("observed_row_count")
        != manifest.get("observed_row_count")
        or coverage.get("coverage_matrix_row_count")
        != artifacts["coverage_matrix_parquet"]["row_count"]
    ):
        raise X13CorrelationCacheError(
            "coverage JSON schema/row_count is invalid"
        )
    connection = duckdb.connect()
    try:
        audit = connection.execute(
            """
            SELECT
              count(*),
              count(*) FILTER (
                WHERE validity_status = 'OBSERVED'
                  AND NOT contaminated
                  AND NOT order_ambiguous
              ),
              list(DISTINCT delay_scenario_seconds ORDER BY delay_scenario_seconds),
              list(DISTINCT horizon_seconds ORDER BY horizon_seconds),
              count(DISTINCT game_id)
            FROM read_parquet(?)
            """,
            [str(observed_path)],
        ).fetchone()
    finally:
        connection.close()
    if (
        audit[0] != artifacts["observed_cache"]["row_count"]
        or audit[1] != manifest.get("observed_row_count")
        or (
            require_frozen
            and (
                audit[2] != list(X13_DELAY_SCENARIOS_SECONDS)
                or audit[3] != list(X13_HORIZONS_SECONDS)
                or audit[4] != 20
            )
        )
    ):
        raise X13CorrelationCacheError(
            "observed cache content gates are invalid"
        )
    return manifest


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace(parent: Path, source: str, target: str) -> None:
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        source_raw = os.fsencode(source)
        target_raw = os.fsencode(target)
        if hasattr(libc, "renameatx_np"):
            rename = libc.renameatx_np
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            result = rename(
                parent_fd, source_raw, parent_fd, target_raw, 0x00000004
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
            result = rename(parent_fd, source_raw, parent_fd, target_raw, 1)
        else:
            raise X13CorrelationCacheError(
                "atomic no-replace publication is unsupported"
            )
        if result != 0:
            observed = ctypes.get_errno()
            if observed in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(target)
            raise X13CorrelationCacheError(
                f"atomic cache publication failed: errno={observed}"
            )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _publish_bundle(
    stage: Path,
    *,
    output_root: Path,
    manifest: dict[str, object],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise X13CorrelationCacheError(
            "cache output root must be a regular directory"
        )
    cache_id = str(manifest["cache_id"])
    target_name = cache_id.replace(":", "-")
    target = output_root / target_name
    if target.exists():
        existing = _verify_cache_directory(target, require_frozen=False)
        if existing != manifest:
            raise X13CorrelationCacheError(
                "immutable cache target conflicts with current projection"
            )
        return target
    container = Path(
        tempfile.mkdtemp(prefix=".x13-publish.", dir=output_root)
    )
    bundle = container / target_name
    bundle.mkdir()
    try:
        expected_data = {
            "x13_reaction_observed_cache_v1.parquet",
            "x13_reaction_coverage_matrix_v1.parquet",
            "x13_reaction_coverage_matrix_v1.json",
        }
        stage_files = {
            child.name for child in stage.iterdir() if child.is_file()
        }
        if stage_files != expected_data:
            raise X13CorrelationCacheError(
                "projection stage file set is invalid"
            )
        for name in sorted(expected_data):
            shutil.copyfile(stage / name, bundle / name)
        (bundle / "manifest.json").write_bytes(
            _canonical_json_bytes(manifest)
        )
        files = sorted(
            child
            for child in bundle.iterdir()
            if child.is_file() and child.name != "checksums.sha256"
        )
        (bundle / "checksums.sha256").write_text(
            "".join(
                f"{_sha256_file(child).removeprefix('sha256:')}  {child.name}\n"
                for child in files
            ),
            "ascii",
        )
        for child in bundle.iterdir():
            if child.is_file():
                _fsync_file(child)
        _fsync_directory(bundle)
        _verify_cache_directory(bundle, require_frozen=False)
        source_name = f"{container.name}/{target_name}"
        try:
            _atomic_rename_noreplace(
                output_root, source_name, target_name
            )
        except FileExistsError:
            existing = _verify_cache_directory(
                target, require_frozen=False
            )
            if existing != manifest:
                raise X13CorrelationCacheError(
                    "immutable cache target conflicts after publish race"
                )
        _fsync_directory(output_root)
        return target
    finally:
        shutil.rmtree(container, ignore_errors=True)


def _write_external_runtime_report(
    output_root: Path,
    *,
    cache_id: str,
    elapsed_seconds: float,
    input_partition_count: int,
    observed_row_count: int,
) -> Path:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Linux":
        peak *= 1024
    report = {
        "bounded_memory_claim": False,
        "cache_id": cache_id,
        "claim_scope": "DESCRIPTIVE_CACHE_ONLY",
        "duckdb_memory_limit_configuration": "1GB",
        "duckdb_threads": 1,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "input_partition_count": input_partition_count,
        "observed_row_count": observed_row_count,
        "process_peak_rss_bytes": peak,
        "schema": "nfl_x13_correlation_cache_runtime_v2",
        "storage_scope": "OUTSIDE_CONTENT_ADDRESSED_BUNDLE",
        "verification_strategy": (
            "streaming no-follow digest snapshot; no retained game payloads"
        ),
    }
    raw = _canonical_json_bytes(report)
    digest = hashlib.sha256(raw).hexdigest()
    directory = (
        output_root
        / "runtime-reports"
        / cache_id.replace(":", "-")
    )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"sha256-{digest}.json"
    if not target.exists():
        temporary = directory / f".{target.name}.{os.getpid()}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temporary, target)
        _fsync_directory(directory)
    return target


def verify_x13_correlation_cache(
    cache_path: str | Path,
) -> dict[str, object]:
    """Verify an immutable cache and return its manifest."""

    return _verify_cache_directory(Path(cache_path), require_frozen=True)


def build_x13_correlation_cache(
    batch_path: str | Path,
    *,
    output_root: str | Path,
) -> Path:
    """Snapshot verified inputs, build, verify, then atomically publish."""

    started = time.monotonic()
    batch = Path(batch_path)
    root = Path(output_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise X13CorrelationCacheError(
            "cache output root must be a regular directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".x13-work.", dir=root))
    try:
        snapshot_root = work / "snapshot"
        snapshot_root.mkdir()
        try:
            snapshot = _snapshot_and_verify_frozen_batch(
                batch, snapshot_root
            )
        except Exception as error:
            raise X13CorrelationCacheError(
                f"frozen batch strict verification failed: {error}"
            ) from error
        stage = work / "projection"
        stage.mkdir()
        identities = _materialize_verified_cache(
            snapshot,
            stage,
            expected_observed_rows=X13_EXPECTED_OBSERVED_ROW_COUNT,
        )
        if (
            identities["candidate_row_count"]
            != X13_FROZEN_CANDIDATE_ROW_COUNT
            or identities["input_partition_count"]
            != X13_EXPECTED_ASSOCIATION_PARTITIONS
            or identities["delay_scenarios_seconds"]
            != list(X13_DELAY_SCENARIOS_SECONDS)
            or identities["horizons_seconds"]
            != list(X13_HORIZONS_SECONDS)
        ):
            raise X13CorrelationCacheError(
                "cache does not preserve the frozen candidate universe/grid"
            )
        approved = _approved_orientation_ids(snapshot)
        if len(approved) != X13_EXPECTED_CROSS_VENUE_ORIENTATIONS:
            raise X13CorrelationCacheError(
                "cross-venue orientation audit is not the frozen 154-set"
            )
        manifest = _build_cache_manifest(
            identities,
            batch_id=X13_FROZEN_BATCH_ID,
            candidate_row_count=X13_FROZEN_CANDIDATE_ROW_COUNT,
            cross_venue_orientation_count=(
                X13_EXPECTED_CROSS_VENUE_ORIENTATIONS
            ),
        )
        target = _publish_bundle(
            stage, output_root=root, manifest=manifest
        )
        verify_x13_correlation_cache(target)
        _write_external_runtime_report(
            root,
            cache_id=str(manifest["cache_id"]),
            elapsed_seconds=time.monotonic() - started,
            input_partition_count=int(
                identities["input_partition_count"]
            ),
            observed_row_count=int(identities["observed_row_count"]),
        )
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the descriptive, precomputed X-13 observed-reaction cache"
        )
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = build_x13_correlation_cache(
        args.batch, output_root=args.output_root
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
