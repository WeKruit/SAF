from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from prediction_market.immutable_object_store import (
    ArtifactRefV1,
    ImmutableObjectStoreError,
    content_addressed_key,
    hydrate_artifact,
    mirror_artifact,
)


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def head_object(self, key: str) -> dict[str, object] | None:
        payload = self.objects.get(key)
        if payload is None:
            return None
        return {
            "byte_size": len(payload),
            "metadata": self.metadata[key],
        }

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None:
        self.objects[key] = path.read_bytes()
        self.metadata[key] = dict(metadata)

    def read_object_chunks(
        self, key: str, *, chunk_size: int = 1024 * 1024
    ):
        payload = self.objects[key]
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _ref(payload: bytes) -> ArtifactRefV1:
    digest = _digest(payload)
    return ArtifactRefV1(
        sha256=digest,
        byte_length=len(payload),
        media_type="application/x-parquet",
        local_relative_path="x13/cache/example.parquet",
        s3_bucket="prediction-market-data",
        s3_key=content_addressed_key(
            prefix="prediction-market",
            namespace="research/nfl/x13/single-game/2025_14_DAL_DET",
            sha256=digest,
            suffix=".parquet",
        ),
    )


def test_mirror_then_hydrate_proves_byte_identity(tmp_path: Path) -> None:
    payload = b"immutable-x13"
    source = tmp_path / "source.parquet"
    source.write_bytes(payload)
    ref = _ref(payload)
    store = MemoryStore()

    mirrored = mirror_artifact(source, ref=ref, object_store=store)
    hydrated = hydrate_artifact(
        ref=ref,
        object_store=store,
        cache_root=tmp_path / "cache",
    )

    assert mirrored == ref
    assert hydrated.read_bytes() == payload
    assert store.metadata[ref.s3_key]["sha256"] == ref.sha256


def test_hydrate_fails_closed_for_local_or_remote_hash_mismatch(
    tmp_path: Path,
) -> None:
    payload = b"expected"
    ref = _ref(payload)
    store = MemoryStore()
    store.objects[ref.s3_key] = b"tampered"
    store.metadata[ref.s3_key] = {"sha256": ref.sha256}

    with pytest.raises(ImmutableObjectStoreError, match="remote"):
        hydrate_artifact(
            ref=ref,
            object_store=store,
            cache_root=tmp_path / "remote",
        )

    local = tmp_path / "local" / ref.local_relative_path
    local.parent.mkdir(parents=True)
    local.write_bytes(b"tampered")
    with pytest.raises(ImmutableObjectStoreError, match="local"):
        hydrate_artifact(
            ref=ref,
            object_store=store,
            cache_root=tmp_path / "local",
        )


def test_artifact_ref_rejects_escaping_paths() -> None:
    payload = b"x"
    with pytest.raises(ImmutableObjectStoreError, match="relative"):
        ArtifactRefV1(
            sha256=_digest(payload),
            byte_length=1,
            media_type="application/octet-stream",
            local_relative_path="../escape",
            s3_bucket="prediction-market-data",
            s3_key="prediction-market/object",
        )
