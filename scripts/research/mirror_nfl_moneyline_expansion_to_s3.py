#!/usr/bin/env python3
"""Mirror a completed NFL moneyline raw capture to immutable S3 objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from prediction_market.immutable_object_store import (
    ArtifactRefV1,
    ImmutableObjectStoreError,
    content_addressed_key,
    mirror_artifact,
)
from prediction_market.pmxt.data_store import (
    BotoS3ObjectStore,
    DataStoreError,
    s3_config_from_env,
)
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)
from prediction_market.sports.nfl_historical_moneyline_capture import (
    HistoricalMoneylineCaptureError,
    MoneylineExpansionSpec,
    dry_run_moneyline_expansion,
    load_moneyline_expansion_spec,
)


_BASE_PREFIX = "prediction-market"
_REMOTE_NAMESPACE = "research/nfl/x13/moneyline-raw-v1"
_RAW_VERSION = "nfl2025-moneyline-expansion-v1"
_BUILDER_VERSION = "nfl-historical-moneyline-capture-v1"
_CAPTURE_SCOPE = "moneyline-only"
_REACTION_ACCESS = "CLOSED"
_GAME_COUNT = 234
_VALIDATION_FIELDS = ["identity", "timestamp", "count", "cursor"]
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"2025_[0-9]{2}_[A-Z0-9]{2,3}_[A-Z0-9]{2,3}\Z")
_POINTER_FIELDS = {
    "batch_plan_sha256",
    "manifest_path",
    "manifest_sha256",
    "schema",
}
_BATCH_FIELDS = {
    "batch_plan_sha256",
    "builder_version",
    "candidate_scan_sha256",
    "capture_scope",
    "game_count",
    "game_indexes",
    "governance_sha256",
    "network_workers",
    "reaction_access",
    "schema",
    "temporal_audit_sha256",
    "validation_fields",
}
_BATCH_INDEX_FIELDS = {
    "game_id",
    "index_path",
    "index_sha256",
    "kalshi_trade_count",
    "polymarket_trade_count",
    "raw_manifest_count",
}
_GAME_INDEX_FIELDS = {
    "builder_version",
    "candidate_scan_sha256",
    "capture_scope",
    "captured_at_utc",
    "game_id",
    "identity",
    "kalshi_event_markets",
    "kalshi_trades",
    "plan_sha256",
    "polymarket_event_metadata",
    "polymarket_trades",
    "raw_manifests",
    "reaction_access",
    "schema",
    "temporal_audit_sha256",
    "validation_fields",
}
_RAW_FIELDS = {
    "byte_length",
    "manifest_path",
    "manifest_sha256",
    "object_path",
    "object_sha256",
    "request_sha256",
}


class MoneylineMirrorError(RuntimeError):
    """The local or remote mirror identity could not be proven."""


@dataclass(frozen=True, slots=True)
class RawProvenance:
    cohort: str
    game_id: str
    manifest_path: str
    manifest_sha256: str
    request_sha256: str
    week: int


@dataclass(frozen=True, slots=True)
class LocalRawObject:
    path: Path
    relative_path: str
    sha256: str
    byte_length: int
    provenance: tuple[RawProvenance, ...]


@dataclass(frozen=True, slots=True)
class CompletedBatch:
    pointer_bytes: bytes
    pointer_sha256: str
    manifest_sha256: str
    batch_plan_sha256: str
    candidate_scan_sha256: str
    governance_sha256: str
    temporal_audit_sha256: str
    game_ids: tuple[str, ...]
    raw_manifest_count: int
    raw_objects: tuple[LocalRawObject, ...]

    @property
    def total_byte_length(self) -> int:
        return sum(item.byte_length for item in self.raw_objects)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MoneylineMirrorError("value is not canonical JSON") from error
    return encoded + b"\n"


def _strict_json(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MoneylineMirrorError(f"{label} has duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                MoneylineMirrorError(f"{label} has a non-finite JSON value")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MoneylineMirrorError(
            f"{label} is not strict UTF-8 JSON"
        ) from error


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MoneylineMirrorError(f"{field} is not a canonical SHA-256")
    return value


def _require_count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise MoneylineMirrorError(f"{field} is not a nonnegative integer")
    return value


def _relative_path(value: object, *, field: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise MoneylineMirrorError(f"{field} is not a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MoneylineMirrorError(f"{field} is not a relative POSIX path")
    return path


def _existing_root(value: Path, *, field: str) -> Path:
    if value.is_symlink() or not value.is_dir():
        raise MoneylineMirrorError(f"{field} must be an existing directory")
    return value.resolve()


def _existing_file(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> Path:
    lexical = root.joinpath(*relative.parts)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise MoneylineMirrorError(f"{label} is missing or escapes its root") from error
    if resolved != lexical or resolved.is_symlink() or not resolved.is_file():
        raise MoneylineMirrorError(f"{label} is not a no-follow regular file")
    return resolved


def _read_json(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> tuple[Path, bytes, dict[str, object]]:
    path = _existing_file(root, relative, label=label)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise MoneylineMirrorError(f"{label} is unreadable") from error
    value = _strict_json(payload, label=label)
    if type(value) is not dict or payload != _canonical_json(value):
        raise MoneylineMirrorError(f"{label} is not canonical JSON")
    return path, payload, value


def _expected_manifest_path(sha256: str) -> str:
    digest = sha256.removeprefix("sha256:")
    return f"batch/manifests/{digest}.manifest.json"


def _expected_index_path(game_id: str, sha256: str) -> str:
    digest = sha256.removeprefix("sha256:")
    return f"games/{game_id}/objects/sha256/{digest[:2]}/{digest}.json"


def _verify_formal_capture(
    *,
    candidate_scan: Path,
    temporal_audit: Path,
    governance: Path,
    capture_root: Path,
    raw_root: Path,
    program_root: Path,
) -> MoneylineExpansionSpec:
    try:
        spec = load_moneyline_expansion_spec(
            candidate_scan,
            temporal_audit,
            governance,
        )
        proof = dry_run_moneyline_expansion(
            spec,
            raw_root=raw_root,
            program_root=program_root,
            output_root=capture_root,
        )
    except HistoricalMoneylineCaptureError as error:
        raise MoneylineMirrorError(
            "formal capture replay verification failed"
        ) from error
    expected_ids = [plan.game_id for plan in spec.plans]
    if (
        proof.get("schema") != "nfl_moneyline_expansion_dry_run_v1"
        or proof.get("batch_plan_sha256") != spec.batch_plan_sha256
        or proof.get("governance_sha256") != spec.governance_sha256
        or proof.get("network_accessed") is not False
        or proof.get("game_count") != _GAME_COUNT
        or proof.get("pending_game_count") != 0
        or proof.get("pending_game_ids") != []
        or proof.get("resumed_game_count") != _GAME_COUNT
        or proof.get("resumed_game_ids") != expected_ids
    ):
        raise MoneylineMirrorError(
            "formal capture replay did not prove 234 completed games"
        )
    return spec


def _verified_raw_object(
    row: object,
    *,
    game_id: str,
    cohort: str,
    week: int,
    raw_root: Path,
    program_root: Path,
) -> tuple[LocalRawObject, RawProvenance]:
    if type(row) is not dict or set(row) != _RAW_FIELDS:
        raise MoneylineMirrorError(f"{game_id} raw reference is malformed")
    request_sha = _require_sha256(
        row.get("request_sha256"),
        field=f"{game_id} raw request_sha256",
    )
    manifest_sha = _require_sha256(
        row.get("manifest_sha256"),
        field=f"{game_id} raw manifest_sha256",
    )
    object_sha = _require_sha256(
        row.get("object_sha256"),
        field=f"{game_id} raw object_sha256",
    )
    byte_length = _require_count(
        row.get("byte_length"),
        field=f"{game_id} raw byte_length",
    )
    manifest_relative = _relative_path(
        row.get("manifest_path"),
        field=f"{game_id} raw manifest_path",
    )
    object_relative = _relative_path(
        row.get("object_path"),
        field=f"{game_id} raw object_path",
    )
    manifest_path = _existing_file(
        raw_root,
        manifest_relative,
        label=f"{game_id} raw manifest",
    )
    object_path = _existing_file(
        raw_root,
        object_relative,
        label=f"{game_id} raw object",
    )
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=raw_root,
            program_root=program_root,
        )
    except StaticStoreError as error:
        raise MoneylineMirrorError(
            f"{game_id} raw static verification failed"
        ) from error
    manifest = verified.record.manifest
    if (
        verified.record.manifest_path != manifest_path
        or verified.record.object_path != object_path
        or verified.record.version != _RAW_VERSION
        or verified.record.extension != "json"
        or manifest.manifest_sha256 != manifest_sha
        or manifest.object_sha256 != object_sha
        or manifest.byte_length != byte_length
        or len(verified.object_bytes) != byte_length
    ):
        raise MoneylineMirrorError(
            f"{game_id} raw reference differs from verified bytes"
        )
    provenance = RawProvenance(
        cohort=cohort,
        game_id=game_id,
        manifest_path=manifest_relative.as_posix(),
        manifest_sha256=manifest_sha,
        request_sha256=request_sha,
        week=week,
    )
    return (
        LocalRawObject(
            path=object_path,
            relative_path=object_relative.as_posix(),
            sha256=object_sha,
            byte_length=byte_length,
            provenance=(provenance,),
        ),
        provenance,
    )


def _add_raw_object(
    objects: dict[str, LocalRawObject],
    item: LocalRawObject,
    provenance: RawProvenance,
) -> None:
    prior = objects.get(item.sha256)
    if prior is None:
        objects[item.sha256] = item
        return
    if prior.byte_length != item.byte_length:
        raise MoneylineMirrorError(
            "one object hash has conflicting immutable metadata"
        )
    source = min((prior, item), key=lambda value: value.relative_path)
    objects[item.sha256] = LocalRawObject(
        path=source.path,
        relative_path=source.relative_path,
        sha256=item.sha256,
        byte_length=item.byte_length,
        provenance=tuple(
            sorted(
                {*prior.provenance, provenance},
                key=lambda value: (
                    value.manifest_sha256,
                    value.request_sha256,
                ),
            )
        ),
    )


def _index_trade_counts(
    index: dict[str, object],
    *,
    game_id: str,
) -> tuple[int, int]:
    polymarket = index.get("polymarket_trades")
    kalshi = index.get("kalshi_trades")
    if type(polymarket) is not dict or type(kalshi) is not list:
        raise MoneylineMirrorError(f"{game_id} trade summary is malformed")
    polymarket_count = _require_count(
        polymarket.get("raw_item_count"),
        field=f"{game_id} Polymarket trade count",
    )
    kalshi_count = 0
    for item in kalshi:
        if type(item) is not dict:
            raise MoneylineMirrorError(
                f"{game_id} Kalshi trade summary is malformed"
            )
        kalshi_count += _require_count(
            item.get("raw_item_count"),
            field=f"{game_id} Kalshi trade count",
        )
    return kalshi_count, polymarket_count


def _verify_completed_batch(
    *,
    candidate_scan: Path,
    temporal_audit: Path,
    governance: Path,
    capture_root: Path,
    raw_root: Path,
    program_root: Path,
) -> CompletedBatch:
    _pointer_path, pointer_bytes, pointer = _read_json(
        capture_root,
        PurePosixPath("batch/latest.json"),
        label="completion batch pointer",
    )
    if (
        set(pointer) != _POINTER_FIELDS
        or pointer.get("schema")
        != "nfl_moneyline_expansion_batch_pointer_v1"
    ):
        raise MoneylineMirrorError("completion batch pointer is malformed")
    manifest_sha = _require_sha256(
        pointer.get("manifest_sha256"),
        field="batch manifest_sha256",
    )
    batch_plan_sha = _require_sha256(
        pointer.get("batch_plan_sha256"),
        field="batch plan_sha256",
    )
    manifest_relative = _relative_path(
        pointer.get("manifest_path"),
        field="batch manifest_path",
    )
    if manifest_relative.as_posix() != _expected_manifest_path(manifest_sha):
        raise MoneylineMirrorError(
            "batch pointer is not content addressed"
        )
    _manifest_path, manifest_bytes, manifest = _read_json(
        capture_root,
        manifest_relative,
        label="completion batch manifest",
    )
    if _sha256(manifest_bytes) != manifest_sha:
        raise MoneylineMirrorError("batch manifest hash differs")
    candidate_sha = _require_sha256(
        manifest.get("candidate_scan_sha256"),
        field="candidate scan_sha256",
    )
    temporal_sha = _require_sha256(
        manifest.get("temporal_audit_sha256"),
        field="temporal audit_sha256",
    )
    governance_sha = _require_sha256(
        manifest.get("governance_sha256"),
        field="governance_sha256",
    )
    rows = manifest.get("game_indexes")
    if (
        set(manifest) != _BATCH_FIELDS
        or manifest.get("schema")
        != "nfl_moneyline_expansion_batch_manifest_v1"
        or manifest.get("builder_version") != _BUILDER_VERSION
        or manifest.get("capture_scope") != _CAPTURE_SCOPE
        or manifest.get("reaction_access") != _REACTION_ACCESS
        or manifest.get("batch_plan_sha256") != batch_plan_sha
        or manifest.get("game_count") != _GAME_COUNT
        or manifest.get("network_workers") != 2
        or manifest.get("validation_fields") != _VALIDATION_FIELDS
        or type(rows) is not list
        or len(rows) != _GAME_COUNT
    ):
        raise MoneylineMirrorError("batch manifest is not complete")

    spec = _verify_formal_capture(
        candidate_scan=candidate_scan,
        temporal_audit=temporal_audit,
        governance=governance,
        capture_root=capture_root,
        raw_root=raw_root,
        program_root=program_root,
    )
    if (
        candidate_sha != spec.candidate_scan_sha256
        or temporal_sha != spec.temporal_audit_sha256
        or governance_sha != spec.governance_sha256
        or batch_plan_sha != spec.batch_plan_sha256
    ):
        raise MoneylineMirrorError(
            "batch manifest differs from the pinned formal capture"
        )
    expected_game_ids = [plan.game_id for plan in spec.plans]
    assignments = {
        game_id: (cohort, week)
        for game_id, cohort, week in spec.game_assignments
    }
    game_ids: list[str] = []
    raw_objects: dict[str, LocalRawObject] = {}
    request_shas: set[str] = set()
    manifest_shas: set[str] = set()
    raw_manifest_count = 0
    for row in rows:
        if type(row) is not dict or set(row) != _BATCH_INDEX_FIELDS:
            raise MoneylineMirrorError("batch game index is malformed")
        game_id = row.get("game_id")
        if type(game_id) is not str or _GAME_ID_RE.fullmatch(game_id) is None:
            raise MoneylineMirrorError("batch game_id is malformed")
        game_ids.append(game_id)
        index_sha = _require_sha256(
            row.get("index_sha256"),
            field=f"{game_id} index_sha256",
        )
        index_relative = _relative_path(
            row.get("index_path"),
            field=f"{game_id} index_path",
        )
        if index_relative.as_posix() != _expected_index_path(
            game_id,
            index_sha,
        ):
            raise MoneylineMirrorError(
                f"{game_id} index is not content addressed"
            )
        _checkpoint_path, _checkpoint_bytes, checkpoint = _read_json(
            capture_root,
            PurePosixPath(f"checkpoints/{game_id}.json"),
            label=f"{game_id} checkpoint",
        )
        if (
            checkpoint.get("index_path") != index_relative.as_posix()
            or checkpoint.get("index_sha256") != index_sha
        ):
            raise MoneylineMirrorError(
                f"{game_id} batch index differs from its verified checkpoint"
            )
        _index_path, index_bytes, index = _read_json(
            capture_root,
            index_relative,
            label=f"{game_id} capture index",
        )
        raw_rows = index.get("raw_manifests")
        if (
            _sha256(index_bytes) != index_sha
            or set(index) != _GAME_INDEX_FIELDS
            or index.get("schema")
            != "nfl_moneyline_game_capture_index_v1"
            or index.get("builder_version") != _BUILDER_VERSION
            or index.get("capture_scope") != _CAPTURE_SCOPE
            or index.get("reaction_access") != _REACTION_ACCESS
            or index.get("candidate_scan_sha256") != candidate_sha
            or index.get("temporal_audit_sha256") != temporal_sha
            or index.get("game_id") != game_id
            or index.get("validation_fields") != _VALIDATION_FIELDS
            or type(raw_rows) is not list
            or not raw_rows
        ):
            raise MoneylineMirrorError(f"{game_id} index identity differs")
        expected_raw_count = _require_count(
            row.get("raw_manifest_count"),
            field=f"{game_id} raw manifest count",
        )
        expected_kalshi_count = _require_count(
            row.get("kalshi_trade_count"),
            field=f"{game_id} Kalshi trade count",
        )
        expected_polymarket_count = _require_count(
            row.get("polymarket_trade_count"),
            field=f"{game_id} Polymarket trade count",
        )
        kalshi_count, polymarket_count = _index_trade_counts(
            index,
            game_id=game_id,
        )
        if (
            len(raw_rows) != expected_raw_count
            or kalshi_count != expected_kalshi_count
            or polymarket_count != expected_polymarket_count
        ):
            raise MoneylineMirrorError(
                f"{game_id} batch counts differ from its verified index"
            )
        assignment = assignments.get(game_id)
        if assignment is None:
            raise MoneylineMirrorError(
                f"{game_id} has no pinned eligibility assignment"
            )
        cohort, week = assignment
        previous_request: str | None = None
        for raw_row in raw_rows:
            item, provenance = _verified_raw_object(
                raw_row,
                game_id=game_id,
                cohort=cohort,
                week=week,
                raw_root=raw_root,
                program_root=program_root,
            )
            if (
                previous_request is not None
                and provenance.request_sha256 <= previous_request
            ):
                raise MoneylineMirrorError(
                    f"{game_id} raw requests are not strictly ordered"
                )
            if (
                provenance.request_sha256 in request_shas
                or provenance.manifest_sha256 in manifest_shas
            ):
                raise MoneylineMirrorError(
                    "completed batch duplicates raw evidence"
                )
            previous_request = provenance.request_sha256
            request_shas.add(provenance.request_sha256)
            manifest_shas.add(provenance.manifest_sha256)
            _add_raw_object(raw_objects, item, provenance)
            raw_manifest_count += 1

    if (
        game_ids != expected_game_ids
        or game_ids != sorted(game_ids)
        or len(set(game_ids)) != _GAME_COUNT
    ):
        raise MoneylineMirrorError(
            "batch does not contain 234 unique sorted games"
        )
    completed = CompletedBatch(
        pointer_bytes=pointer_bytes,
        pointer_sha256=_sha256(pointer_bytes),
        manifest_sha256=manifest_sha,
        batch_plan_sha256=batch_plan_sha,
        candidate_scan_sha256=candidate_sha,
        governance_sha256=governance_sha,
        temporal_audit_sha256=temporal_sha,
        game_ids=tuple(game_ids),
        raw_manifest_count=raw_manifest_count,
        raw_objects=tuple(
            sorted(raw_objects.values(), key=lambda item: item.sha256)
        ),
    )
    _assert_pointer_unchanged(completed, capture_root=capture_root)
    return completed


def _assert_pointer_unchanged(
    batch: CompletedBatch,
    *,
    capture_root: Path,
) -> None:
    _path, current, _value = _read_json(
        capture_root,
        PurePosixPath("batch/latest.json"),
        label="completion batch pointer",
    )
    if current != batch.pointer_bytes:
        raise MoneylineMirrorError(
            "completion batch pointer changed during verification"
        )


def _raw_ref(item: LocalRawObject, *, bucket: str) -> ArtifactRefV1:
    return ArtifactRefV1(
        sha256=item.sha256,
        byte_length=item.byte_length,
        media_type="application/json",
        local_relative_path=item.relative_path,
        s3_bucket=bucket,
        s3_key=content_addressed_key(
            prefix=_BASE_PREFIX,
            namespace=_REMOTE_NAMESPACE,
            sha256=item.sha256,
            suffix=".json",
        ),
    )


def _audit_payload(
    batch: CompletedBatch,
    *,
    bucket: str,
    refs: tuple[ArtifactRefV1, ...],
) -> bytes:
    if len(refs) != len(batch.raw_objects):
        raise MoneylineMirrorError("remote readback set is incomplete")
    return _canonical_json(
        {
            "schema": "nfl_moneyline_expansion_remote_mirror_audit_v1",
            "source_batch": {
                "batch_manifest_sha256": batch.manifest_sha256,
                "batch_plan_sha256": batch.batch_plan_sha256,
                "batch_pointer_sha256": batch.pointer_sha256,
                "candidate_scan_sha256": batch.candidate_scan_sha256,
                "game_count": len(batch.game_ids),
                "governance_sha256": batch.governance_sha256,
                "temporal_audit_sha256": batch.temporal_audit_sha256,
            },
            "eligibility_split": {
                "development_game_count": sum(
                    game_id.split("_")[1] <= "12"
                    for game_id in batch.game_ids
                ),
                "final_holdout_game_count": sum(
                    game_id.split("_")[1] >= "13"
                    for game_id in batch.game_ids
                ),
                "lineage": "pinned_governance_game_assignment",
            },
            "destination": {
                "namespace": _REMOTE_NAMESPACE,
                "s3_bucket": bucket,
                "s3_prefix": _BASE_PREFIX,
                "s3_key_prefix": f"{_BASE_PREFIX}/{_REMOTE_NAMESPACE}",
            },
            "local_relative_path_root": "raw_root",
            "raw_manifest_count": batch.raw_manifest_count,
            "unique_object_count": len(refs),
            "total_unique_byte_length": sum(ref.byte_length for ref in refs),
            "objects": [
                {
                    "readback_verified": True,
                    "ref": ref.to_dict(),
                    "source_manifests": [
                        {
                            "cohort": source.cohort,
                            "game_id": source.game_id,
                            "local_relative_path": source.manifest_path,
                            "manifest_sha256": source.manifest_sha256,
                            "request_sha256": source.request_sha256,
                            "week": source.week,
                        }
                        for source in item.provenance
                    ],
                }
                for item, ref in zip(batch.raw_objects, refs, strict=True)
            ],
        }
    )


def _audit_root(value: Path, *, program_root: Path) -> Path:
    if any(part == ".." for part in value.parts):
        raise MoneylineMirrorError("audit_root contains parent traversal")
    absolute = Path(os.path.abspath(value))
    if absolute.resolve(strict=False) != absolute:
        raise MoneylineMirrorError("audit_root traverses a symlink")
    try:
        absolute.relative_to(program_root)
    except ValueError as error:
        raise MoneylineMirrorError(
            "audit_root must be inside program_root"
        ) from error
    return absolute


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise MoneylineMirrorError(
                "content-addressed audit path conflicts"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
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
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise MoneylineMirrorError(
                    "content-addressed audit path conflicts"
                ) from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _mirror(
    batch: CompletedBatch,
    *,
    capture_root: Path,
    program_root: Path,
    audit_root: Path,
    env_file: Path | None,
    workers: int,
) -> dict[str, object]:
    try:
        config = s3_config_from_env(env_file=env_file)
    except DataStoreError as error:
        raise MoneylineMirrorError(str(error)) from error
    if config.prefix.strip("/") != _BASE_PREFIX:
        raise MoneylineMirrorError(f"S3 prefix must be {_BASE_PREFIX}")
    try:
        store = BotoS3ObjectStore.from_config(config)
        def mirror_one(item: LocalRawObject) -> ArtifactRefV1:
            ref = _raw_ref(item, bucket=config.bucket)
            mirror_artifact(item.path, ref=ref, object_store=store)
            return ref

        with ThreadPoolExecutor(max_workers=workers) as executor:
            refs = list(executor.map(mirror_one, batch.raw_objects))
    except ImmutableObjectStoreError as error:
        raise MoneylineMirrorError(str(error)) from error
    except Exception as error:
        raise MoneylineMirrorError("remote object mirror failed") from error

    _assert_pointer_unchanged(batch, capture_root=capture_root)
    audit_bytes = _audit_payload(
        batch,
        bucket=config.bucket,
        refs=tuple(refs),
    )
    audit_sha = _sha256(audit_bytes)
    digest = audit_sha.removeprefix("sha256:")
    audit_path = (
        audit_root
        / "objects"
        / "sha256"
        / digest[:2]
        / f"{digest}.remote-mirror-audit.json"
    )
    _publish_immutable(audit_path, audit_bytes)
    audit_ref = ArtifactRefV1(
        sha256=audit_sha,
        byte_length=len(audit_bytes),
        media_type="application/json",
        local_relative_path=audit_path.relative_to(program_root).as_posix(),
        s3_bucket=config.bucket,
        s3_key=content_addressed_key(
            prefix=_BASE_PREFIX,
            namespace=f"{_REMOTE_NAMESPACE}/audits",
            sha256=audit_sha,
            suffix=".json",
        ),
    )
    try:
        mirror_artifact(audit_path, ref=audit_ref, object_store=store)
    except ImmutableObjectStoreError as error:
        raise MoneylineMirrorError(str(error)) from error
    except Exception as error:
        raise MoneylineMirrorError("remote audit mirror failed") from error
    _assert_pointer_unchanged(batch, capture_root=capture_root)
    return {
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "batch_manifest_sha256": batch.manifest_sha256,
        "game_count": len(batch.game_ids),
        "raw_manifest_count": batch.raw_manifest_count,
        "status": "remote_mirror_readback_verified",
        "total_unique_byte_length": batch.total_byte_length,
        "unique_object_count": len(batch.raw_objects),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scan", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--governance", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--program-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        capture_root = _existing_root(
            args.capture_root,
            field="capture_root",
        )
        raw_root = _existing_root(args.raw_root, field="raw_root")
        program_root = _existing_root(
            args.program_root,
            field="program_root",
        )
        batch = _verify_completed_batch(
            candidate_scan=args.candidate_scan,
            temporal_audit=args.temporal_audit,
            governance=args.governance,
            capture_root=capture_root,
            raw_root=raw_root,
            program_root=program_root,
        )
        if args.dry_run:
            result = {
                "batch_manifest_sha256": batch.manifest_sha256,
                "batch_plan_sha256": batch.batch_plan_sha256,
                "dry_run": True,
                "game_count": len(batch.game_ids),
                "raw_manifest_count": batch.raw_manifest_count,
                "remote_write_attempted": False,
                "status": "complete_local_batch_verified",
                "total_unique_byte_length": batch.total_byte_length,
                "unique_object_count": len(batch.raw_objects),
            }
        else:
            if args.audit_root is None:
                raise MoneylineMirrorError(
                    "--audit-root is required unless --dry-run is set"
                )
            if args.workers < 1 or args.workers > 32:
                raise MoneylineMirrorError(
                    "--workers must be between 1 and 32"
                )
            result = _mirror(
                batch,
                capture_root=capture_root,
                program_root=program_root,
                audit_root=_audit_root(
                    args.audit_root,
                    program_root=program_root,
                ),
                env_file=args.env_file,
                workers=args.workers,
            )
    except MoneylineMirrorError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("ERROR: local batch verification failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
