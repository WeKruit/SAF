"""Independent private object-store publication for X-13 dashboard assets."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Protocol

from prediction_market.sports.nfl_x13_dashboard import (
    X13DashboardExportError,
    verify_x13_dashboard_export,
)


_CHUNK_SIZE = 1024 * 1024
_MAX_OBJECT_BYTES = 8 * 1024 * 1024
_MAX_POINTER_BYTES = 64 * 1024
_CONDITIONAL_CREATE_ATTEMPTS = 3
_DEFAULT_ROOT_PREFIX = "prediction-market"
_DASHBOARD_PREFIX = "dashboard/x13"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ENV_NAMES = (
    "PM_DATA_S3_ENDPOINT",
    "PM_DATA_S3_REGION",
    "PM_DATA_S3_BUCKET",
    "PM_DATA_S3_ACCESS_KEY_ID",
    "PM_DATA_S3_SECRET_ACCESS_KEY",
)


class DashboardStoreError(RuntimeError):
    """Raised when dashboard publication cannot prove byte identity."""


class DashboardObjectAlreadyExists(DashboardStoreError):
    """A conditional immutable-object create lost a concurrent race."""


class DashboardConditionalRequestConflict(DashboardStoreError):
    """A conditional create conflicted before the object became visible."""


@dataclass(frozen=True, slots=True)
class DashboardS3Config:
    endpoint: str
    region: str
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    prefix: str = _DEFAULT_ROOT_PREFIX


@dataclass(frozen=True, slots=True)
class DashboardPublishResult:
    batch_id: str
    manifest_sha256: str
    manifest_key: str
    pointer_key: str
    pointer_sha256: str
    verified_object_count: int
    total_verified_bytes: int


@dataclass(frozen=True, slots=True)
class DashboardVerifyResult:
    batch_id: str
    manifest_sha256: str
    manifest_key: str
    pointer_key: str
    pointer_sha256: str
    verified_object_count: int
    total_verified_bytes: int


class DashboardObjectStore(Protocol):
    def head_object(self, key: str) -> dict[str, object] | None: ...

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        metadata: dict[str, str],
        create_only: bool,
    ) -> None: ...

    def read_object_chunks(
        self, key: str, *, chunk_size: int = _CHUNK_SIZE
    ) -> Iterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class _LocalObject:
    sha256: str
    payload: bytes


def _canonical_json(value: object) -> bytes:
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


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DashboardStoreError(
            f"{label} must be sha256:<64 lowercase hex>"
        )
    return value


def _normalized_root_prefix(prefix: str) -> str:
    if not isinstance(prefix, str):
        raise DashboardStoreError("dashboard S3 prefix must be text")
    if prefix == "":
        return ""
    path = PurePosixPath(prefix)
    if (
        "\\" in prefix
        or path.is_absolute()
        or path.as_posix() != prefix
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DashboardStoreError("dashboard S3 prefix would escape its root")
    return prefix


def _join_key(prefix: str, suffix: str) -> str:
    root = _normalized_root_prefix(prefix)
    return f"{root}/{suffix}" if root else suffix


def dashboard_object_key(prefix: str, object_sha256: str) -> str:
    digest = _require_sha256(
        object_sha256, "dashboard object hash"
    ).removeprefix("sha256:")
    return _join_key(
        prefix,
        f"{_DASHBOARD_PREFIX}/objects/sha256/{digest[:2]}/{digest}.json",
    )


def dashboard_latest_pointer_key(prefix: str) -> str:
    return _join_key(prefix, f"{_DASHBOARD_PREFIX}/published/latest.json")


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DashboardStoreError(f"S3 env file is unavailable: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_dashboard_s3_config(
    *,
    env_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> DashboardS3Config:
    material: dict[str, str] = {}
    if env_file is not None:
        material.update(_parse_env_file(Path(env_file)))
    material.update(dict(os.environ if env is None else env))
    missing = [name for name in _ENV_NAMES if not material.get(name)]
    if missing:
        raise DashboardStoreError(
            "missing S3 environment values: " + ", ".join(missing)
        )
    endpoint = material["PM_DATA_S3_ENDPOINT"]
    if not endpoint.startswith("https://"):
        raise DashboardStoreError("PM_DATA_S3_ENDPOINT must be HTTPS")
    prefix = material.get("PM_DATA_S3_PREFIX", _DEFAULT_ROOT_PREFIX)
    _normalized_root_prefix(prefix)
    return DashboardS3Config(
        endpoint=endpoint,
        region=material["PM_DATA_S3_REGION"],
        bucket=material["PM_DATA_S3_BUCKET"],
        access_key_id=material["PM_DATA_S3_ACCESS_KEY_ID"],
        secret_access_key=material["PM_DATA_S3_SECRET_ACCESS_KEY"],
        prefix=prefix,
    )


class BotoDashboardObjectStore:
    """Minimal boto3 adapter kept independent from the PMXT store."""

    def __init__(self, client: Any, *, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def from_config(
        cls, config: DashboardS3Config
    ) -> "BotoDashboardObjectStore":
        import boto3
        from botocore.config import Config

        session = boto3.session.Session(
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name=config.region,
        )
        client = session.client(
            "s3",
            endpoint_url=config.endpoint,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        return cls(client, bucket=config.bucket)

    def head_object(self, key: str) -> dict[str, object] | None:
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(
                Bucket=self._bucket, Key=key
            )
        except ClientError as error:
            code = str(
                error.response.get("Error", {}).get("Code", "")
            )
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return {
            "key": key,
            "byte_size": int(response["ContentLength"]),
            "metadata": dict(response.get("Metadata", {})),
        }

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        metadata: dict[str, str],
        create_only: bool,
    ) -> None:
        from botocore.exceptions import ClientError

        sha256 = _require_sha256(
            metadata.get("sha256"), "dashboard upload checksum"
        )
        arguments: dict[str, object] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": payload,
            "Metadata": metadata,
            "ContentType": "application/json",
            "ChecksumSHA256": base64.b64encode(
                bytes.fromhex(sha256.removeprefix("sha256:"))
            ).decode("ascii"),
        }
        if create_only:
            arguments["IfNoneMatch"] = "*"
        try:
            self._client.put_object(**arguments)
        except ClientError as error:
            code = str(
                error.response.get("Error", {}).get("Code", "")
            )
            status = error.response.get(
                "ResponseMetadata", {}
            ).get("HTTPStatusCode")
            if create_only and (
                code in {"412", "PreconditionFailed"} or status == 412
            ):
                raise DashboardObjectAlreadyExists(key) from error
            if create_only and (
                code == "ConditionalRequestConflict" or status == 409
            ):
                raise DashboardConditionalRequestConflict(key) from error
            raise

    def read_object_chunks(
        self, key: str, *, chunk_size: int = _CHUNK_SIZE
    ) -> Iterator[bytes]:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(
                Bucket=self._bucket, Key=key
            )
        except ClientError as error:
            code = str(
                error.response.get("Error", {}).get("Code", "")
            )
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise KeyError(key) from error
            raise
        body = response["Body"]
        try:
            yield from body.iter_chunks(chunk_size=chunk_size)
        finally:
            body.close()


def _validate_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise DashboardStoreError("dashboard asset path must be text")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DashboardStoreError(
            "dashboard asset path would escape the export root"
        )
    return path


def _resolve_batch_root(export_root: Path) -> Path:
    if not export_root.is_dir() or export_root.is_symlink():
        raise DashboardStoreError(
            "dashboard export root must be a real directory"
        )
    if (export_root / "manifests").is_dir():
        return export_root
    batches = [
        path
        for path in export_root.iterdir()
        if path.name.startswith("batch=")
        and path.is_dir()
        and not path.is_symlink()
    ]
    if len(batches) != 1:
        raise DashboardStoreError(
            "dashboard export root must contain exactly one batch"
        )
    return batches[0]


def _load_local_export(
    export_root: str | Path,
) -> tuple[Path, dict[str, object], _LocalObject, tuple[_LocalObject, ...]]:
    batch_root = _resolve_batch_root(Path(export_root))
    try:
        verified = verify_x13_dashboard_export(batch_root)
    except X13DashboardExportError as error:
        raise DashboardStoreError(
            "local export is not a verified canonical Task 2 projection"
        ) from error
    objects: dict[str, _LocalObject] = {}
    for asset in verified.assets:
        entry = asset.descriptor
        expected_sha256 = _require_sha256(
            entry.get("object_sha256"), "dashboard asset hash"
        )
        payload = asset.payload
        prior = objects.get(expected_sha256)
        if prior is not None and prior.payload != payload:
            raise DashboardStoreError(
                "dashboard object hash maps to conflicting local bytes"
            )
        objects[expected_sha256] = _LocalObject(
            sha256=expected_sha256, payload=payload
        )
    manifest_object = _LocalObject(
        sha256=verified.manifest_sha256,
        payload=verified.manifest_payload,
    )
    prior_manifest = objects.get(verified.manifest_sha256)
    if (
        prior_manifest is not None
        and prior_manifest.payload != verified.manifest_payload
    ):
        raise DashboardStoreError(
            "dashboard manifest hash conflicts with an asset"
        )
    objects[verified.manifest_sha256] = manifest_object
    return (
        batch_root,
        dict(verified.manifest),
        manifest_object,
        tuple(objects[key] for key in sorted(objects)),
    )


def _read_remote(
    object_store: DashboardObjectStore,
    key: str,
    *,
    max_bytes: int,
) -> bytes:
    payload = bytearray()
    try:
        chunks = object_store.read_object_chunks(key)
        for block in chunks:
            if not isinstance(block, (bytes, bytearray)):
                raise DashboardStoreError(
                    "remote dashboard object returned non-bytes"
                )
            payload.extend(block)
            if len(payload) > max_bytes:
                raise DashboardStoreError(
                    "remote dashboard object exceeds its size limit"
                )
    except KeyError as error:
        raise DashboardStoreError(
            f"remote dashboard object is missing: {key}"
        ) from error
    except DashboardStoreError:
        raise
    except Exception as error:
        raise DashboardStoreError(
            f"failed to read remote dashboard object: {key}"
        ) from error
    return bytes(payload)


def _verify_remote_object(
    object_store: DashboardObjectStore,
    *,
    key: str,
    expected_sha256: str,
    expected_size: int,
) -> None:
    remote = _read_remote(
        object_store, key, max_bytes=_MAX_OBJECT_BYTES
    )
    if len(remote) != expected_size or _sha256(remote) != expected_sha256:
        raise DashboardStoreError(
            f"remote dashboard object hash mismatch: {key}"
        )


def _publish_immutable_object(
    object_store: DashboardObjectStore,
    *,
    key: str,
    item: _LocalObject,
) -> bool:
    try:
        head = object_store.head_object(key)
    except Exception as error:
        raise DashboardStoreError(
            f"failed to inspect remote dashboard object: {key}"
        ) from error
    if head is not None:
        metadata = head.get("metadata")
        if (
            head.get("byte_size") != len(item.payload)
            or not isinstance(metadata, dict)
            or metadata.get("sha256") != item.sha256
        ):
            raise DashboardStoreError(
                f"immutable dashboard object conflict: {key}"
            )
        _verify_remote_object(
            object_store,
            key=key,
            expected_sha256=item.sha256,
            expected_size=len(item.payload),
        )
        return False
    for attempt in range(_CONDITIONAL_CREATE_ATTEMPTS):
        try:
            object_store.put_bytes(
                key,
                item.payload,
                metadata={"sha256": item.sha256},
                create_only=True,
            )
        except DashboardObjectAlreadyExists:
            _verify_remote_object(
                object_store,
                key=key,
                expected_sha256=item.sha256,
                expected_size=len(item.payload),
            )
            return False
        except DashboardConditionalRequestConflict as error:
            try:
                concurrent_head = object_store.head_object(key)
            except Exception as head_error:
                raise DashboardStoreError(
                    f"failed to inspect 409 dashboard object: {key}"
                ) from head_error
            if concurrent_head is not None:
                _verify_remote_object(
                    object_store,
                    key=key,
                    expected_sha256=item.sha256,
                    expected_size=len(item.payload),
                )
                return False
            if attempt + 1 == _CONDITIONAL_CREATE_ATTEMPTS:
                raise DashboardStoreError(
                    "dashboard conditional create conflict remained "
                    f"unresolved after {_CONDITIONAL_CREATE_ATTEMPTS} "
                    f"attempts: {key}"
                ) from error
            continue
        except Exception as error:
            raise DashboardStoreError(
                f"dashboard content upload failed: {key}"
            ) from error
        _verify_remote_object(
            object_store,
            key=key,
            expected_sha256=item.sha256,
            expected_size=len(item.payload),
        )
        return True
    raise AssertionError("conditional create attempt loop is unreachable")


def _pointer_document(
    manifest: Mapping[str, object],
    *,
    manifest_sha256: str,
    manifest_key: str,
) -> bytes:
    return _canonical_json(
        {
            "schema": "x13_dashboard_latest_pointer_v1",
            "experiment_id": "X-13",
            "batch_id": manifest["batch_id"],
            "manifest_sha256": manifest_sha256,
            "manifest_key": manifest_key,
        }
    )


def _publish_pointer(
    object_store: DashboardObjectStore,
    *,
    key: str,
    payload: bytes,
) -> str:
    expected_sha256 = _sha256(payload)
    try:
        head = object_store.head_object(key)
    except Exception as error:
        raise DashboardStoreError(
            "failed to inspect dashboard latest pointer"
        ) from error
    if head is not None:
        current = _read_remote(
            object_store, key, max_bytes=_MAX_POINTER_BYTES
        )
        if current == payload:
            return expected_sha256
    try:
        object_store.put_bytes(
            key,
            payload,
            metadata={"sha256": expected_sha256},
            create_only=False,
        )
    except Exception as error:
        try:
            actual = _read_remote(
                object_store, key, max_bytes=_MAX_POINTER_BYTES
            )
        except DashboardStoreError as read_error:
            raise DashboardStoreError(
                "dashboard latest pointer has an indeterminate commit; "
                "manual inspection required"
            ) from read_error
        if actual == payload:
            return expected_sha256
        raise DashboardStoreError(
            "dashboard latest pointer has an indeterminate commit; "
            "manual inspection required"
        ) from error
    try:
        actual = _read_remote(
            object_store, key, max_bytes=_MAX_POINTER_BYTES
        )
    except DashboardStoreError as error:
        raise DashboardStoreError(
            "dashboard latest pointer has an indeterminate commit; "
            "manual inspection required"
        ) from error
    if actual != payload or _sha256(actual) != expected_sha256:
        raise DashboardStoreError(
            "dashboard latest pointer has an indeterminate commit; "
            "manual inspection required"
        )
    return expected_sha256


def _store_for(
    config: DashboardS3Config,
    object_store: DashboardObjectStore | None,
) -> DashboardObjectStore:
    return (
        object_store
        if object_store is not None
        else BotoDashboardObjectStore.from_config(config)
    )


def publish_dashboard_export(
    export_root: str | Path,
    config: DashboardS3Config,
    *,
    object_store: DashboardObjectStore | None = None,
) -> DashboardPublishResult:
    """Publish immutable content first and update the latest pointer last."""

    _normalized_root_prefix(config.prefix)
    _, manifest, manifest_object, objects = _load_local_export(export_root)
    store = _store_for(config, object_store)
    total_bytes = 0
    for item in objects:
        key = dashboard_object_key(config.prefix, item.sha256)
        _publish_immutable_object(store, key=key, item=item)
        total_bytes += len(item.payload)
    manifest_key = dashboard_object_key(
        config.prefix, manifest_object.sha256
    )
    pointer_key = dashboard_latest_pointer_key(config.prefix)
    pointer = _pointer_document(
        manifest,
        manifest_sha256=manifest_object.sha256,
        manifest_key=manifest_key,
    )
    pointer_sha256 = _publish_pointer(
        store, key=pointer_key, payload=pointer
    )
    return DashboardPublishResult(
        batch_id=str(manifest["batch_id"]),
        manifest_sha256=manifest_object.sha256,
        manifest_key=manifest_key,
        pointer_key=pointer_key,
        pointer_sha256=pointer_sha256,
        verified_object_count=len(objects),
        total_verified_bytes=total_bytes,
    )


def _parse_remote_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise DashboardStoreError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise DashboardStoreError(f"{label} must be an object")
    return value


def verify_dashboard_publication(
    config: DashboardS3Config,
    *,
    object_store: DashboardObjectStore | None = None,
) -> DashboardVerifyResult:
    """Re-read the pointer, manifest, and every referenced remote object."""

    _normalized_root_prefix(config.prefix)
    store = _store_for(config, object_store)
    pointer_key = dashboard_latest_pointer_key(config.prefix)
    pointer_raw = _read_remote(
        store, pointer_key, max_bytes=_MAX_POINTER_BYTES
    )
    pointer = _parse_remote_json(
        pointer_raw,
        "dashboard latest pointer",
    )
    if pointer.get("schema") != "x13_dashboard_latest_pointer_v1":
        raise DashboardStoreError("dashboard latest pointer schema is invalid")
    manifest_sha256 = _require_sha256(
        pointer.get("manifest_sha256"), "dashboard manifest hash"
    )
    manifest_key = dashboard_object_key(
        config.prefix, manifest_sha256
    )
    if pointer.get("manifest_key") != manifest_key:
        raise DashboardStoreError(
            "dashboard latest pointer manifest key would escape its prefix"
        )
    manifest_raw = _read_remote(
        store, manifest_key, max_bytes=_MAX_OBJECT_BYTES
    )
    if _sha256(manifest_raw) != manifest_sha256:
        raise DashboardStoreError(
            "remote dashboard manifest hash mismatch"
        )
    manifest = _parse_remote_json(
        manifest_raw, "remote dashboard manifest"
    )
    if (
        manifest.get("schema")
        != "nfl_x13_dashboard_publish_manifest_v1"
        or manifest.get("batch_id") != pointer.get("batch_id")
    ):
        raise DashboardStoreError(
            "remote dashboard manifest identity is inconsistent"
        )
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise DashboardStoreError(
            "remote dashboard manifest assets are invalid"
        )
    objects: dict[str, int] = {manifest_sha256: len(manifest_raw)}
    for entry in assets:
        if not isinstance(entry, dict):
            raise DashboardStoreError(
                "remote dashboard asset descriptor is invalid"
            )
        _validate_relative_path(entry.get("relative_path"))
        object_sha256 = _require_sha256(
            entry.get("object_sha256"), "remote dashboard asset hash"
        )
        byte_length = entry.get("byte_length")
        if not isinstance(byte_length, int) or byte_length < 0:
            raise DashboardStoreError(
                "remote dashboard asset byte length is invalid"
            )
        prior = objects.get(object_sha256)
        if prior is not None and prior != byte_length:
            raise DashboardStoreError(
                "remote dashboard object has conflicting lengths"
            )
        objects[object_sha256] = byte_length
    total_bytes = 0
    for object_sha256, byte_length in sorted(objects.items()):
        _verify_remote_object(
            store,
            key=dashboard_object_key(config.prefix, object_sha256),
            expected_sha256=object_sha256,
            expected_size=byte_length,
        )
        total_bytes += byte_length
    final_pointer_raw = _read_remote(
        store, pointer_key, max_bytes=_MAX_POINTER_BYTES
    )
    if final_pointer_raw != pointer_raw:
        raise DashboardStoreError(
            "dashboard latest pointer changed during verification"
        )
    return DashboardVerifyResult(
        batch_id=str(manifest["batch_id"]),
        manifest_sha256=manifest_sha256,
        manifest_key=manifest_key,
        pointer_key=pointer_key,
        pointer_sha256=_sha256(pointer_raw),
        verified_object_count=len(objects),
        total_verified_bytes=total_bytes,
    )


__all__ = [
    "BotoDashboardObjectStore",
    "DashboardConditionalRequestConflict",
    "DashboardObjectAlreadyExists",
    "DashboardObjectStore",
    "DashboardPublishResult",
    "DashboardS3Config",
    "DashboardStoreError",
    "DashboardVerifyResult",
    "dashboard_latest_pointer_key",
    "dashboard_object_key",
    "load_dashboard_s3_config",
    "publish_dashboard_export",
    "verify_dashboard_publication",
]
