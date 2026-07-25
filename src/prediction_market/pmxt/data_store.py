"""Supabase S3 mirror utilities for immutable PMXT Parquet objects."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


_SHA256_HEX_LENGTH = 64
_CHUNK_SIZE = 1024 * 1024
_CANONICAL_SUFFIX = ".parquet"
_DEFAULT_PREFIX = "prediction-market"
_ENV_ENDPOINT = "PM_DATA_S3_ENDPOINT"
_ENV_REGION = "PM_DATA_S3_REGION"
_ENV_BUCKET = "PM_DATA_S3_BUCKET"
_ENV_ACCESS_KEY_ID = "PM_DATA_S3_ACCESS_KEY_ID"
_ENV_SECRET_ACCESS_KEY = "PM_DATA_S3_SECRET_ACCESS_KEY"
_ENV_PREFIX = "PM_DATA_S3_PREFIX"
_PRUNE_CONFIRMATION = "NO_LOCAL_PMXT_READERS"


class DataStoreError(RuntimeError):
    """Raised when PMXT data-store mirroring cannot prove byte identity."""


@dataclass(frozen=True, slots=True)
class LocalPMXTObject:
    sha256: str
    byte_size: int
    path: Path
    relative_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "path": str(self.path),
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True, slots=True)
class RemoteObject:
    key: str
    byte_size: int
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class S3Config:
    endpoint: str
    region: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    prefix: str = _DEFAULT_PREFIX


class PMXTObjectStore(Protocol):
    def head_object(self, key: str) -> dict[str, object] | None: ...

    def put_file(self, key: str, path: Path, *, metadata: dict[str, str]) -> None: ...

    def list_objects(self, prefix: str) -> Any: ...

    def read_object_chunks(
        self, key: str, *, chunk_size: int = _CHUNK_SIZE
    ) -> Any: ...


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _require_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + _SHA256_HEX_LENGTH
    ):
        raise DataStoreError("PMXT object hash must be sha256:<64 lowercase hex>")
    digest = value.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in digest):
        raise DataStoreError("PMXT object hash must be sha256:<64 lowercase hex>")
    return digest


def _normalized_prefix(prefix: str | None) -> str:
    value = (prefix or _DEFAULT_PREFIX).strip("/")
    if not value:
        raise DataStoreError("S3 PMXT prefix cannot be empty")
    return value


def pmxt_canonical_prefix(prefix: str | None = _DEFAULT_PREFIX) -> str:
    return f"{_normalized_prefix(prefix)}/raw/pmxt-v2/sha256"


def pmxt_canonical_key(prefix: str | None, sha256: str) -> str:
    digest = _require_sha256(sha256)
    return f"{pmxt_canonical_prefix(prefix)}/{digest[:2]}/{digest}.parquet"


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
            byte_size += len(block)
    return "sha256:" + digest.hexdigest(), byte_size


def scan_local_pmxt_objects(raw_root: str | Path) -> tuple[LocalPMXTObject, ...]:
    root = Path(raw_root)
    canonical_root = root / "pmxt-v2" / "sha256"
    if not canonical_root.exists():
        return ()
    objects: list[LocalPMXTObject] = []
    for path in sorted(canonical_root.glob("*/*.parquet")):
        digest = path.name.removesuffix(_CANONICAL_SUFFIX)
        if (
            len(digest) != _SHA256_HEX_LENGTH
            or path.parent.name != digest[:2]
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DataStoreError(f"invalid PMXT canonical object path: {path}")
        expected = "sha256:" + digest
        actual, byte_size = _sha256_path(path)
        if actual != expected:
            raise DataStoreError(
                f"local PMXT hash mismatch: {path} contains {actual}; expected {expected}"
            )
        objects.append(
            LocalPMXTObject(
                sha256=actual,
                byte_size=byte_size,
                path=path,
                relative_path=path.relative_to(root).as_posix(),
            )
        )
    return tuple(objects)


def upload_pmxt(
    raw_root: str | Path,
    object_store: PMXTObjectStore,
    *,
    prefix: str | None = _DEFAULT_PREFIX,
) -> dict[str, object]:
    objects = scan_local_pmxt_objects(raw_root)
    results: list[dict[str, object]] = []
    uploaded_count = 0
    skipped_count = 0
    failure_count = 0
    for item in objects:
        key = pmxt_canonical_key(prefix, item.sha256)
        head = object_store.head_object(key)
        if (
            head is not None
            and head.get("byte_size") == item.byte_size
            and isinstance(head.get("metadata"), dict)
            and head["metadata"].get("sha256") == item.sha256
        ):
            status = "skipped_existing"
            skipped_count += 1
        else:
            try:
                object_store.put_file(key, item.path, metadata={"sha256": item.sha256})
            except Exception as exc:
                status = "upload_failed"
                failure_count += 1
                results.append(
                    {
                        **item.to_dict(),
                        "key": key,
                        "status": status,
                        "error": str(exc),
                    }
                )
                continue
            uploaded_count += 1
            status = "uploaded"
        results.append({**item.to_dict(), "key": key, "status": status})
    return {
        "audit_kind": "pmxt_s3_upload_v1",
        "generated_at": _utc_now(),
        "prefix": _normalized_prefix(prefix),
        "summary": {
            "local_object_count": len(objects),
            "uploaded_count": uploaded_count,
            "skipped_count": skipped_count,
            "failure_count": failure_count,
            "total_bytes": sum(item.byte_size for item in objects),
        },
        "objects": results,
    }


def _sha256_remote_chunks(chunks: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    for block in chunks:
        if not block:
            continue
        digest.update(block)
        byte_size += len(block)
    return "sha256:" + digest.hexdigest(), byte_size


def _sha256_from_pmxt_key(key: str, *, prefix: str | None) -> str | None:
    key_path = PurePosixPath(key)
    expected_prefix = PurePosixPath(pmxt_canonical_prefix(prefix))
    try:
        suffix = key_path.relative_to(expected_prefix)
    except ValueError:
        return None
    if len(suffix.parts) != 2:
        return None
    shard, filename = suffix.parts
    if not filename.endswith(_CANONICAL_SUFFIX):
        return None
    digest = filename.removesuffix(_CANONICAL_SUFFIX)
    if shard != digest[:2]:
        return None
    try:
        _require_sha256("sha256:" + digest)
    except DataStoreError:
        return None
    return "sha256:" + digest


def verify_pmxt_remote(
    object_store: PMXTObjectStore,
    *,
    prefix: str | None = _DEFAULT_PREFIX,
) -> dict[str, object]:
    remote_prefix = pmxt_canonical_prefix(prefix)
    objects: list[dict[str, object]] = []
    verified_count = 0
    failure_count = 0
    total_bytes = 0
    for remote in object_store.list_objects(remote_prefix):
        expected_sha = _sha256_from_pmxt_key(remote.key, prefix=prefix)
        if expected_sha is None:
            objects.append(
                {
                    "key": remote.key,
                    "byte_size": remote.byte_size,
                    "sha256": None,
                    "status": "invalid_key",
                }
            )
            failure_count += 1
            continue
        actual_sha, actual_size = _sha256_remote_chunks(
            object_store.read_object_chunks(remote.key)
        )
        status = "verified"
        if actual_sha != expected_sha:
            status = "hash_mismatch"
        elif actual_size != remote.byte_size:
            status = "size_mismatch"
        if status == "verified":
            verified_count += 1
            total_bytes += actual_size
        else:
            failure_count += 1
        objects.append(
            {
                "key": remote.key,
                "sha256": expected_sha,
                "actual_sha256": actual_sha,
                "byte_size": actual_size,
                "listed_byte_size": remote.byte_size,
                "status": status,
            }
        )
    return {
        "audit_kind": "pmxt_s3_verify_v1",
        "generated_at": _utc_now(),
        "prefix": _normalized_prefix(prefix),
        "summary": {
            "remote_object_count": len(objects),
            "verified_count": verified_count,
            "failure_count": failure_count,
            "total_verified_bytes": total_bytes,
        },
        "objects": objects,
    }


def _walk_json(value: object) -> Any:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _is_pmxt_v2_object_path(value: str) -> bool:
    parts = PurePosixPath(value).parts
    return (
        "source=pmxt" in parts
        and "dataset=DS-PMXT-V2" in parts
        and value.endswith(_CANONICAL_SUFFIX)
    ) or (
        len(parts) == 4
        and parts[0] == "pmxt-v2"
        and parts[1] == "sha256"
        and parts[3].endswith(_CANONICAL_SUFFIX)
    )


def scan_manifest_pmxt_references(
    manifest_roots: tuple[Path, ...] | list[Path],
) -> tuple[dict[str, str], ...]:
    refs: list[dict[str, str]] = []
    for root in manifest_roots:
        if not root.exists():
            continue
        candidates = sorted([root] if root.is_file() else root.rglob("*.json"))
        for path in candidates:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for node in _walk_json(document):
                sha256 = node.get("object_sha256")
                if not isinstance(sha256, str):
                    continue
                try:
                    _require_sha256(sha256)
                except DataStoreError:
                    continue
                for path_field in ("object_path", "native_object_path"):
                    object_path = node.get(path_field)
                    if isinstance(object_path, str) and _is_pmxt_v2_object_path(
                        object_path
                    ):
                        refs.append(
                            {
                                "sha256": sha256,
                                "object_path": object_path,
                                "manifest_path": str(path),
                            }
                        )
    return tuple(sorted(refs, key=lambda item: (item["sha256"], item["object_path"])))


def load_remote_inventory(path: str | Path) -> dict[str, object]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataStoreError(f"remote inventory is not readable JSON: {path}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("objects"), list):
        raise DataStoreError("remote inventory must contain an objects array")
    return document


def _verified_remote_shas(remote_inventory: dict[str, object]) -> set[str]:
    objects = remote_inventory.get("objects")
    if not isinstance(objects, list):
        raise DataStoreError("remote inventory must contain an objects array")
    shas: set[str] = set()
    for item in objects:
        if not isinstance(item, dict):
            continue
        sha256 = item.get("sha256")
        if item.get("status") == "verified" and isinstance(sha256, str):
            _require_sha256(sha256)
            shas.add(sha256)
    return shas


def status_pmxt(
    raw_root: str | Path,
    *,
    prefix: str | None = _DEFAULT_PREFIX,
    manifest_roots: tuple[Path, ...] | list[Path] = (),
    remote_inventory: dict[str, object] | None = None,
) -> dict[str, object]:
    local_objects = scan_local_pmxt_objects(raw_root)
    if remote_inventory is None:
        raise DataStoreError("status requires a verified remote inventory")
    manifest_refs = scan_manifest_pmxt_references(tuple(manifest_roots))
    local_shas = {item.sha256 for item in local_objects}
    remote_verified_shas = _verified_remote_shas(remote_inventory)
    manifest_shas = {item["sha256"] for item in manifest_refs}
    local_missing_remote = sorted(local_shas - remote_verified_shas)
    manifest_missing_local = sorted(manifest_shas - local_shas)
    manifest_missing_remote = sorted(manifest_shas - remote_verified_shas)
    remote_missing_local = sorted(remote_verified_shas - local_shas)
    return {
        "audit_kind": "pmxt_s3_status_v1",
        "generated_at": _utc_now(),
        "prefix": _normalized_prefix(prefix),
        "summary": {
            "local_object_count": len(local_objects),
            "remote_verified_object_count": len(remote_verified_shas),
            "manifest_reference_count": len(manifest_shas),
            "local_missing_remote_count": len(local_missing_remote),
            "manifest_missing_local_count": len(manifest_missing_local),
            "manifest_missing_remote_count": len(manifest_missing_remote),
            "remote_missing_local_count": len(remote_missing_local),
        },
        "local_missing_remote": local_missing_remote,
        "manifest_missing_local": manifest_missing_local,
        "manifest_missing_remote": manifest_missing_remote,
        "remote_missing_local": remote_missing_local,
        "local_objects": [item.to_dict() for item in local_objects],
        "manifest_references": list(manifest_refs),
    }


def _validate_relative_object_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DataStoreError(
            "PMXT hydrate path must be a canonical relative POSIX path"
        )
    return path


def _safe_target_path(raw_root: str | Path, relative: str) -> Path:
    relative_path = _validate_relative_object_path(relative)
    root = Path(raw_root).resolve(strict=True)
    if not root.is_dir():
        raise DataStoreError("PMXT raw root must be a directory")
    current = root
    for part in relative_path.parts[:-1]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            current.mkdir(mode=0o755)
            mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise DataStoreError(f"PMXT hydrate path contains a symlink: {relative}")
        if not stat.S_ISDIR(mode):
            raise DataStoreError(
                f"PMXT hydrate path component is not a directory: {relative}"
            )
    target = current / relative_path.name
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return target
    if stat.S_ISLNK(mode):
        raise DataStoreError(f"PMXT hydrate target is a symlink: {relative}")
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def hydrate_pmxt_object(
    raw_root: str | Path,
    relative_object_path: str,
    object_sha256: str,
    object_store: PMXTObjectStore,
    *,
    prefix: str | None = _DEFAULT_PREFIX,
) -> Path:
    expected = "sha256:" + _require_sha256(object_sha256)
    target = _safe_target_path(raw_root, relative_object_path)
    if target.exists():
        actual, _ = _sha256_path(target)
        if actual == expected:
            return target
        raise DataStoreError(
            f"existing PMXT object SHA-256 mismatch: {relative_object_path}"
        )
    key = pmxt_canonical_key(prefix, expected)
    staging = target.parent / f".{target.name}.{secrets.token_hex(16)}.partial"
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with staging.open("xb") as handle:
            try:
                chunks = object_store.read_object_chunks(key)
                for block in chunks:
                    if not block:
                        continue
                    handle.write(block)
                    digest.update(block)
                    byte_size += len(block)
            except KeyError as exc:
                raise DataStoreError(f"remote PMXT object is missing: {key}") from exc
            except Exception as exc:
                raise DataStoreError(
                    f"failed to hydrate remote PMXT object: {key}: {exc}"
                ) from exc
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        actual = "sha256:" + digest.hexdigest()
        if actual != expected:
            raise DataStoreError(
                f"hydrated PMXT SHA-256 mismatch: {relative_object_path}"
            )
        try:
            os.link(staging, target)
        except FileExistsError:
            pass
        _fsync_directory(target.parent)
        verified, _ = _sha256_path(target)
        if verified != expected:
            raise DataStoreError(
                f"published PMXT SHA-256 mismatch: {relative_object_path}"
            )
        return target
    finally:
        staging.unlink(missing_ok=True)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def default_env_file(cwd: str | Path = ".") -> Path | None:
    root = Path(cwd)
    for name in (".env.s3", ".env"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def s3_config_from_env(
    *,
    env_file: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> S3Config:
    material: dict[str, str] = {}
    selected_env_file = Path(env_file) if env_file is not None else default_env_file()
    if selected_env_file is not None:
        material.update(_parse_env_file(selected_env_file))
    material.update(env or os.environ)
    missing = [
        name
        for name in (
            _ENV_ENDPOINT,
            _ENV_REGION,
            _ENV_BUCKET,
            _ENV_ACCESS_KEY_ID,
            _ENV_SECRET_ACCESS_KEY,
        )
        if not material.get(name)
    ]
    if missing:
        raise DataStoreError(f"missing S3 environment values: {', '.join(missing)}")
    endpoint = material[_ENV_ENDPOINT]
    if not endpoint.startswith("https://"):
        raise DataStoreError("PM_DATA_S3_ENDPOINT must be HTTPS")
    return S3Config(
        endpoint=endpoint,
        region=material[_ENV_REGION],
        bucket=material[_ENV_BUCKET],
        access_key_id=material[_ENV_ACCESS_KEY_ID],
        secret_access_key=material[_ENV_SECRET_ACCESS_KEY],
        prefix=material.get(_ENV_PREFIX, _DEFAULT_PREFIX),
    )


def configured_object_store(
    *,
    env_file: str | Path | None = None,
) -> tuple["BotoS3ObjectStore", str]:
    config = s3_config_from_env(env_file=env_file)
    return BotoS3ObjectStore.from_config(config), config.prefix


def configured_object_store_if_available() -> tuple["BotoS3ObjectStore", str] | None:
    try:
        return configured_object_store()
    except DataStoreError:
        return None


class BotoS3ObjectStore:
    def __init__(self, client: Any, *, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def from_config(cls, config: S3Config) -> "BotoS3ObjectStore":
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

    def head_bucket(self) -> None:
        self._client.head_bucket(Bucket=self._bucket)

    def head_object(self, key: str) -> dict[str, object] | None:
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return {
            "key": key,
            "byte_size": int(response["ContentLength"]),
            "metadata": dict(response.get("Metadata", {})),
        }

    def put_file(self, key: str, path: Path, *, metadata: dict[str, str]) -> None:
        from boto3.s3.transfer import TransferConfig

        self._client.upload_file(
            Filename=str(path),
            Bucket=self._bucket,
            Key=key,
            ExtraArgs={"Metadata": metadata},
            Config=TransferConfig(
                multipart_threshold=64 * 1024 * 1024,
                multipart_chunksize=64 * 1024 * 1024,
                max_concurrency=4,
                use_threads=True,
            ),
        )

    def put_bytes(self, key: str, payload: bytes, *, metadata: dict[str, str]) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=payload,
            Metadata=metadata,
        )

    def list_objects(self, prefix: str) -> Any:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                head = self.head_object(item["Key"])
                yield RemoteObject(
                    key=item["Key"],
                    byte_size=int(item["Size"]),
                    metadata=(
                        dict(head.get("metadata", {}))
                        if isinstance(head, dict)
                        else {}
                    ),
                )

    def read_object_chunks(
        self, key: str, *, chunk_size: int = _CHUNK_SIZE
    ) -> Any:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise KeyError(key) from exc
            raise
        body = response["Body"]
        yield from body.iter_chunks(chunk_size=chunk_size)

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)


def _remote_inventory_all_verified(remote_inventory: dict[str, object]) -> None:
    objects = remote_inventory.get("objects")
    if not isinstance(objects, list) or not objects:
        raise DataStoreError("remote inventory has no verified PMXT objects")
    for item in objects:
        if not isinstance(item, dict) or item.get("status") != "verified":
            raise DataStoreError("remote inventory is not all verified")


def prune_pmxt_local(
    raw_root: str | Path,
    *,
    remote_inventory: dict[str, object],
    execute: bool,
    confirm_no_local_readers: str | None,
) -> dict[str, object]:
    _remote_inventory_all_verified(remote_inventory)
    if execute and confirm_no_local_readers != _PRUNE_CONFIRMATION:
        raise DataStoreError(
            f"prune requires --confirm-no-local-readers {_PRUNE_CONFIRMATION}"
        )
    verified = _verified_remote_shas(remote_inventory)
    root = Path(raw_root)
    candidates: list[Path] = []
    candidates.extend(item.path for item in scan_local_pmxt_objects(root))
    alias_root = root / "raw" / "source=pmxt" / "dataset=DS-PMXT-V2" / "version=v2"
    if alias_root.exists():
        candidates.extend(sorted(alias_root.rglob("*.parquet")))

    deletable: list[dict[str, object]] = []
    for path in sorted(set(candidates)):
        actual, byte_size = _sha256_path(path)
        if actual not in verified:
            raise DataStoreError(f"local PMXT object is not verified remote: {path}")
        deletable.append(
            {"path": str(path), "sha256": actual, "byte_size": byte_size}
        )
    if execute:
        for item in deletable:
            Path(str(item["path"])).unlink()
    return {
        "audit_kind": "pmxt_local_prune_v1",
        "generated_at": _utc_now(),
        "executed": execute,
        "summary": {
            "candidate_count": len(deletable),
            "deleted_count": len(deletable) if execute else 0,
            "candidate_bytes": sum(int(item["byte_size"]) for item in deletable),
        },
        "objects": deletable,
    }
