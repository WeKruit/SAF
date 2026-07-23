from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction_market.contracts import (  # noqa: E402
    StaticDatasetManifestV0,
    canonical_json_bytes,
)
from prediction_market.static_store import (  # noqa: E402
    ImmutableStaticObjectError,
    StaticStoreError,
    StaticStorePathError,
    preserve_static_object,
    read_verified_static_object,
    verify_static_object,
)


FETCHED_AT = "2026-07-22T14:03:04.123456Z"
SCHEMA_FINGERPRINT = "sha256:" + "7" * 64


def _preserve(
    store_root: Path,
    object_bytes: bytes = b"PAR1\x00exact upstream parquet bytes\xff",
    **overrides: object,
):
    values: dict[str, object] = {
        "program_root": PROJECT_ROOT,
        "source": "huggingface",
        "dataset": "DS-POLYMARKET-V1",
        "version": "66a1d6ddfc3cdab9e2087c1e2e855bab272d3404",
        "partition": "2024",
        "extension": "parquet",
        "source_url": (
            "https://huggingface.co/datasets/TimeSeventeen/"
            "Polymarket-v1/resolve/main/data/2024.parquet"
        ),
        "source_request": {
            "method": "GET",
            "headers": {"Accept": "application/vnd.apache.parquet"},
        },
        "source_cursor": "revision:66a1d6d;file:data/2024.parquet",
        "fetched_at": FETCHED_AT,
        "coverage": "calendar_year=2024",
        "etag": '"fixture-etag"',
        "last_modified": "Tue, 21 Jul 2026 23:59:59 GMT",
        "media_type": "application/vnd.apache.parquet",
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "license_ref": "O-009",
        "license_status": "approved",
        "upstream_partition": "2024",
        "object_kind": "byte_exact_original",
        "lineage": {"source_object_refs": [], "query_sha256": None},
    }
    values.update(overrides)
    return preserve_static_object(store_root, object_bytes, **values)


def test_preserves_exact_external_bytes_at_canonical_content_address(
    tmp_path: Path,
) -> None:
    object_bytes = b"\x00PAR1\nexact bytes; no decoding or normalization\xff"
    digest = hashlib.sha256(object_bytes).hexdigest()

    stored = _preserve(tmp_path, object_bytes)

    assert stored.object_path.relative_to(tmp_path).as_posix() == (
        "raw/source=huggingface/dataset=DS-POLYMARKET-V1/"
        "version=66a1d6ddfc3cdab9e2087c1e2e855bab272d3404/"
        f"partition=2024/{digest}.parquet"
    )
    assert stored.object_path.read_bytes() == object_bytes
    assert isinstance(stored.manifest, StaticDatasetManifestV0)
    assert stored.manifest.native_object_path == (
        stored.object_path.relative_to(tmp_path).as_posix()
    )
    assert stored.manifest_path.read_bytes() == (
        canonical_json_bytes(stored.manifest) + b"\n"
    )


def test_exact_retry_is_idempotent_and_does_not_rewrite_files(
    tmp_path: Path,
) -> None:
    first = _preserve(tmp_path)
    object_before = first.object_path.stat()
    manifest_before = first.manifest_path.stat()

    second = _preserve(tmp_path)

    assert second == first
    assert second.object_path.stat().st_ino == object_before.st_ino
    assert second.object_path.stat().st_mtime_ns == object_before.st_mtime_ns
    assert second.manifest_path.stat().st_ino == manifest_before.st_ino
    assert (
        second.manifest_path.stat().st_mtime_ns
        == manifest_before.st_mtime_ns
    )
    assert len(list((tmp_path / "raw").rglob("*.parquet"))) == 1
    assert (
        len(list((tmp_path / "manifests").rglob("*.manifest.json"))) == 1
    )


def test_same_bytes_with_distinct_observation_metadata_share_one_object(
    tmp_path: Path,
) -> None:
    first = _preserve(tmp_path)
    second = _preserve(
        tmp_path,
        fetched_at="2026-07-22T15:03:04.123456Z",
        etag='"fixture-etag-second-observation"',
    )

    assert second.object_path == first.object_path
    assert second.manifest_path != first.manifest_path
    assert second.manifest.manifest_sha256 != first.manifest.manifest_sha256
    assert len(list((tmp_path / "raw").rglob("*.parquet"))) == 1
    assert (
        len(list((tmp_path / "manifests").rglob("*.manifest.json"))) == 2
    )


def test_object_is_durable_before_manifest_commit_and_failed_commit_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import static_store

    original_fsync_directory = static_store._fsync_directory
    commit_events: list[str] = []

    def fail_manifest_commit_once(
        directory: int | Path, *, context: str | None = None
    ) -> None:
        if context in {"raw_commit", "manifest_commit"}:
            commit_events.append(context)
        if context == "manifest_commit":
            raw_objects = list((tmp_path / "raw").rglob("*.parquet"))
            assert len(raw_objects) == 1
            with raw_objects[0].open("rb") as published:
                os.fsync(published.fileno())
            raise OSError("injected manifest commit failure")
        original_fsync_directory(directory, context=context)

    monkeypatch.setattr(
        static_store, "_fsync_directory", fail_manifest_commit_once
    )

    with pytest.raises(OSError, match="injected manifest commit failure"):
        _preserve(tmp_path)

    assert commit_events == ["raw_commit", "manifest_commit"]
    assert list((tmp_path / "manifests").rglob("*.manifest.json")) == []
    assert len(list((tmp_path / "raw").rglob("*.parquet"))) == 1

    monkeypatch.setattr(
        static_store, "_fsync_directory", original_fsync_directory
    )
    retried = _preserve(tmp_path)
    assert retried.manifest_path.is_file()


def test_verify_returns_frozen_governed_record(tmp_path: Path) -> None:
    stored = _preserve(tmp_path)

    verified = verify_static_object(
        stored.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )

    assert verified == stored
    assert isinstance(verified.manifest, StaticDatasetManifestV0)
    with pytest.raises(FrozenInstanceError):
        verified.partition = "corrected"  # type: ignore[misc]


def test_verified_read_returns_exact_bytes_from_one_safe_object_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import static_store

    object_bytes = b"\x00single handle verified read\xff"
    stored = _preserve(tmp_path, object_bytes)
    original_open = static_store.os.open
    object_open_count = 0

    def tracked_open(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal object_open_count
        if path == stored.object_path.name:
            object_open_count += 1
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(static_store.os, "open", tracked_open)

    verified = read_verified_static_object(
        stored.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )

    assert verified.record == stored
    assert verified.object_bytes == object_bytes
    assert object_open_count == 1
    with pytest.raises(FrozenInstanceError):
        verified.object_bytes = b"corrected"  # type: ignore[misc]


def test_object_tampering_fails_closed_and_retry_does_not_correct_it(
    tmp_path: Path,
) -> None:
    stored = _preserve(tmp_path)
    stored.object_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    tampered = bytearray(stored.object_path.read_bytes())
    tampered[-1] ^= 0x01
    stored.object_path.write_bytes(tampered)

    with pytest.raises(StaticStoreError, match="SHA-256"):
        verify_static_object(
            stored.manifest_path,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )
    with pytest.raises(
        ImmutableStaticObjectError,
        match="committed manifest|exact source bytes",
    ):
        _preserve(tmp_path)
    assert stored.object_path.read_bytes() == bytes(tampered)


@pytest.mark.parametrize("recompute_self_hash", [False, True])
def test_manifest_tampering_fails_closed_even_if_self_hash_is_recomputed(
    tmp_path: Path, recompute_self_hash: bool
) -> None:
    from prediction_market import contracts as governed_contracts

    stored = _preserve(tmp_path)
    stored.manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    document = json.loads(stored.manifest_path.read_bytes())
    document["coverage"] = "tampered-after-publication"
    if recompute_self_hash:
        document["manifest_sha256"] = (
            governed_contracts.static_dataset_manifest_sha256(document)
        )
    tampered_bytes = canonical_json_bytes(document) + b"\n"
    stored.manifest_path.write_bytes(tampered_bytes)

    with pytest.raises(StaticStoreError):
        verify_static_object(
            stored.manifest_path,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )
    with pytest.raises(
        ImmutableStaticObjectError, match="canonical bytes"
    ):
        _preserve(tmp_path)
    assert stored.manifest_path.read_bytes() == tampered_bytes


def test_missing_object_fails_closed_and_is_not_repaired_in_place(
    tmp_path: Path,
) -> None:
    stored = _preserve(tmp_path)
    stored.object_path.unlink()

    with pytest.raises(StaticStoreError, match="missing|unsafe"):
        verify_static_object(
            stored.manifest_path,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )
    with pytest.raises(
        ImmutableStaticObjectError, match="committed manifest"
    ):
        _preserve(tmp_path)
    assert stored.object_path.exists() is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "../outside"),
        ("source", "hugging/face"),
        ("dataset", "../DS-POLYMARKET-V1"),
        ("version", "../../revision"),
        ("partition", "/absolute"),
        ("extension", "../parquet"),
        ("extension", ".parquet"),
        ("extension", "parquet."),
        ("extension", "tar..gz"),
        ("source_url", "http://example.test/data.parquet"),
        ("source_url", "https://"),
        ("fetched_at", "2026-07-22T14:03:04+00:00"),
        ("source_request", {"unsafe_float": 0.1}),
    ],
)
def test_unsafe_components_extensions_urls_and_timestamps_are_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(StaticStoreError):
        _preserve(tmp_path, **{field: value})

    assert list(tmp_path.iterdir()) == []


def test_symlinked_store_prefix_and_root_ancestor_cannot_escape(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    os.symlink(outside, store / "raw")

    with pytest.raises(StaticStorePathError, match="symlink"):
        _preserve(store)

    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StaticStorePathError, match="symlink"):
        _preserve(alias / "escaped-store")

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("target_kind", ["object", "manifest"])
def test_verification_never_follows_object_or_manifest_symlinks(
    tmp_path: Path, target_kind: str
) -> None:
    stored = _preserve(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = (
        stored.object_path
        if target_kind == "object"
        else stored.manifest_path
    )
    outside_copy = outside / target.name
    outside_copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside_copy)

    with pytest.raises(StaticStorePathError, match="symlink|regular"):
        verify_static_object(
            stored.manifest_path,
            store_root=tmp_path / "store",
            program_root=PROJECT_ROOT,
        )


def test_source_derived_extract_requires_and_preserves_explicit_lineage(
    tmp_path: Path,
) -> None:
    with pytest.raises(StaticStoreError, match="manifest"):
        _preserve(
            tmp_path,
            object_kind="source_derived_extract",
            lineage={"source_object_refs": [], "query_sha256": None},
        )
    assert list(tmp_path.iterdir()) == []

    source_ref = "sha256:" + "8" * 64
    query_sha256 = "sha256:" + "9" * 64
    derived = _preserve(
        tmp_path,
        object_kind="source_derived_extract",
        lineage={
            "source_object_refs": [source_ref],
            "query_sha256": query_sha256,
        },
    )

    assert derived.manifest.object_kind == "source_derived_extract"
    assert derived.manifest.lineage.source_object_refs == (source_ref,)
    assert derived.manifest.lineage.query_sha256 == query_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("license_ref", "O-006"),
        ("license_status", "research_only"),
        ("dataset", "DS-NOT-REGISTERED"),
    ],
)
def test_preservation_uses_normative_dataset_foreign_key_validator(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(StaticStoreError, match="manifest"):
        _preserve(tmp_path, **{field: value})

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("component_index", "replacement"),
    [
        (1, "source=other-source"),
        (2, "dataset=DS-NFLVERSE"),
        (3, "version=other-version"),
        (4, "partition=2025"),
    ],
)
def test_manifest_path_must_match_object_source_dataset_version_and_partition(
    tmp_path: Path, component_index: int, replacement: str
) -> None:
    stored = _preserve(tmp_path)
    relative_parts = list(stored.manifest_path.relative_to(tmp_path).parts)
    relative_parts[component_index] = replacement
    moved_manifest = tmp_path.joinpath(*relative_parts)
    moved_manifest.parent.mkdir(parents=True, exist_ok=True)
    stored.manifest_path.rename(moved_manifest)

    with pytest.raises(StaticStorePathError, match="match|differ"):
        verify_static_object(
            moved_manifest,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )


def test_manifest_byte_length_is_verified_independently_of_content_hash(
    tmp_path: Path,
) -> None:
    from prediction_market import contracts as governed_contracts

    stored = _preserve(tmp_path)
    document = json.loads(stored.manifest_path.read_bytes())
    document["byte_length"] += 1
    document["manifest_sha256"] = (
        governed_contracts.static_dataset_manifest_sha256(document)
    )
    forged_name = (
        document["manifest_sha256"].removeprefix("sha256:")
        + ".manifest.json"
    )
    forged_manifest = stored.manifest_path.with_name(forged_name)
    forged_manifest.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(StaticStoreError, match="byte length"):
        verify_static_object(
            forged_manifest,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize(
    "noncanonical_path",
    [
        (
            "raw//source=huggingface/dataset=DS-POLYMARKET-V1/"
            "version=66a1d6ddfc3cdab9e2087c1e2e855bab272d3404/"
            "partition=2024/{name}"
        ),
        (
            "raw/./source=huggingface/dataset=DS-POLYMARKET-V1/"
            "version=66a1d6ddfc3cdab9e2087c1e2e855bab272d3404/"
            "partition=2024/{name}"
        ),
    ],
)
def test_verification_rejects_noncanonical_equivalent_object_paths(
    tmp_path: Path, noncanonical_path: str
) -> None:
    from prediction_market import contracts as governed_contracts

    stored = _preserve(tmp_path)
    document = json.loads(stored.manifest_path.read_bytes())
    document["native_object_path"] = noncanonical_path.format(
        name=stored.object_path.name
    )
    document["manifest_sha256"] = (
        governed_contracts.static_dataset_manifest_sha256(document)
    )
    forged_name = (
        document["manifest_sha256"].removeprefix("sha256:")
        + ".manifest.json"
    )
    forged_manifest = stored.manifest_path.with_name(forged_name)
    forged_manifest.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(StaticStorePathError, match="canonical|safe"):
        verify_static_object(
            forged_manifest,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize(
    "operation",
    [verify_static_object, read_verified_static_object],
    ids=["verify", "verified-read"],
)
@pytest.mark.parametrize(
    "alias_fragment",
    ["manifests/./", "manifests//"],
    ids=["dot", "double-separator"],
)
@pytest.mark.parametrize("spelling", ["absolute", "relative"])
def test_static_object_apis_reject_noncanonical_manifest_path_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation,
    alias_fragment: str,
    spelling: str,
) -> None:
    stored = _preserve(tmp_path)
    manifest_path = str(stored.manifest_path)
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
        manifest_path = stored.manifest_path.relative_to(tmp_path).as_posix()
    aliased_path = manifest_path.replace("manifests/", alias_fragment, 1)

    with pytest.raises(StaticStorePathError, match="canonical"):
        operation(
            aliased_path,
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize(
    "operation",
    [verify_static_object, read_verified_static_object],
    ids=["verify", "verified-read"],
)
@pytest.mark.parametrize("target_kind", ["object", "manifest"])
def test_static_object_apis_reject_replaced_directory_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation,
    target_kind: str,
) -> None:
    from prediction_market import static_store

    store_root = tmp_path / "store"
    object_bytes = b"verified bytes from the original object handle"
    stored = _preserve(store_root, object_bytes)
    target_path = (
        stored.object_path
        if target_kind == "object"
        else stored.manifest_path
    )
    target_prefix_name = "raw" if target_kind == "object" else "manifests"
    target_prefix = store_root / target_prefix_name
    target_metadata = target_path.stat()
    target_identity = (target_metadata.st_dev, target_metadata.st_ino)

    outside_prefix = tmp_path / f"outside-{target_prefix_name}"
    replacement_file = outside_prefix / target_path.relative_to(target_prefix)
    replacement_file.parent.mkdir(parents=True)
    replacement_file.write_bytes(b"unverified replacement bytes")

    original_read = static_store.os.read
    swapped = False

    def read_then_replace_ancestor(
        descriptor: int, byte_count: int
    ) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, byte_count)
        metadata = os.fstat(descriptor)
        if (
            not swapped
            and chunk
            and (metadata.st_dev, metadata.st_ino) == target_identity
        ):
            target_prefix.rename(
                store_root / f"{target_prefix_name}-original"
            )
            os.symlink(
                outside_prefix,
                target_prefix,
                target_is_directory=True,
            )
            swapped = True
        return chunk

    monkeypatch.setattr(static_store.os, "read", read_then_replace_ancestor)

    with pytest.raises(StaticStoreError, match="changed|unsafe|symlink"):
        operation(
            stored.manifest_path,
            store_root=store_root,
            program_root=PROJECT_ROOT,
        )

    assert swapped is True
    assert target_path.resolve().is_relative_to(store_root.resolve()) is False
