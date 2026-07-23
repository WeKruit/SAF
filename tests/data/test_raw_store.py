from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest
import yaml
import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction_market.raw_store import (  # noqa: E402
    ImmutableSegmentError,
    PartitionBoundaryError,
    RawSegmentWriter,
    RawStorePathError,
    read_verified_segment,
    verify_segment,
)


UTC_TIME = "2026-07-22T14:03:04.123456Z"


def _records(object_path: Path) -> list[dict[str, object]]:
    with object_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            raw = reader.read()
    return [json.loads(line) for line in raw.splitlines()]


def _write_canonical_manifest(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _manifest_sha256(document: dict[str, object]) -> str:
    hash_input = dict(document)
    hash_input.pop("manifest_sha256", None)
    encoded = json.dumps(
        hash_input,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _refresh_manifest_sha256(document: dict[str, object]) -> None:
    document["manifest_sha256"] = _manifest_sha256(document)


def _make_segment(
    root: Path,
    *,
    capture_session_id: str = "capture-test",
    payload: bytes = b'{"event":"book"}',
):
    writer = RawSegmentWriter(
        root,
        source="polymarket",
        stream="market",
        capture_session_id=capture_session_id,
    )
    ordinal = writer.append(payload, receive_at=UTC_TIME)
    return writer, ordinal, writer.seal()


def test_sealed_segment_preserves_exact_payload_and_hash(tmp_path: Path) -> None:
    payload = b"\x00\xff\nexact websocket frame\x00"
    writer = RawSegmentWriter(
        tmp_path,
        source="polymarket",
        stream="market",
        capture_session_id="capture-exact",
    )

    assert writer.append(payload, receive_at=UTC_TIME) == 0
    manifest = writer.seal()
    verification = verify_segment(manifest.path)

    assert verification.valid is True
    assert verification.errors == ()
    assert manifest.record_count == 1
    assert manifest.object_sha256 == (
        "sha256:" + hashlib.sha256(manifest.object_path.read_bytes()).hexdigest()
    )
    record = _records(manifest.object_path)[0]
    assert base64.b64decode(record["payload_base64"], validate=True) == payload
    assert record["payload_sha256"] == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    assert record["capture_session_id"] == "capture-exact"
    assert record["record_ordinal"] == 0
    assert record["receive_at"] == UTC_TIME


def test_sealed_segment_cannot_be_reopened_or_resealed(tmp_path: Path) -> None:
    writer = RawSegmentWriter(tmp_path, source="polymarket", stream="market")
    manifest = writer.seal()

    with pytest.raises(ImmutableSegmentError, match="sealed"):
        writer.append(b"late", receive_at=UTC_TIME)
    with pytest.raises(ImmutableSegmentError, match="sealed"):
        writer.seal()
    assert verify_segment(manifest.path).valid


def test_final_paths_are_content_addressed_and_hour_partitioned(
    tmp_path: Path,
) -> None:
    writer, _, manifest = _make_segment(tmp_path)
    digest = manifest.object_sha256.removeprefix("sha256:")

    assert manifest.object_path.relative_to(tmp_path).as_posix() == (
        "raw/source=polymarket/stream=market/date=2026-07-22/hour=14/"
        f"{digest}.jsonl.zst"
    )
    assert manifest.path.relative_to(tmp_path).as_posix() == (
        "manifests/source=polymarket/stream=market/date=2026-07-22/hour=14/"
        f"{digest}.manifest.json"
    )
    assert writer.staging_path.exists() is False


def test_staging_is_never_visible_under_raw_or_manifests(tmp_path: Path) -> None:
    writer = RawSegmentWriter(tmp_path, source="kalshi", stream="orderbook")
    writer.append(b"snapshot", receive_at=UTC_TIME)

    assert writer.staging_path.is_relative_to(tmp_path / ".staging")
    assert not list((tmp_path / "raw").rglob("*")) if (tmp_path / "raw").exists() else True
    assert (
        not list((tmp_path / "manifests").rglob("*"))
        if (tmp_path / "manifests").exists()
        else True
    )

    manifest = writer.seal()
    assert manifest.path.is_file()
    assert manifest.object_path.is_file()
    assert writer.staging_path.exists() is False


def test_manifest_is_strict_sidecar_for_the_exact_object(tmp_path: Path) -> None:
    _, _, manifest = _make_segment(tmp_path)
    document = json.loads(manifest.path.read_text(encoding="utf-8"))

    assert set(document) == {
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
    assert document["object_path"] == manifest.object_path.relative_to(tmp_path).as_posix()
    assert document["object_sha256"] == manifest.object_sha256
    assert document["compression"] == "zstd"
    assert document["record_encoding"] == "canonical-json-lines-v0"
    assert document["manifest_sha256"] == _manifest_sha256(document)
    assert manifest.manifest_sha256 == document["manifest_sha256"]


def test_empty_segment_is_valid_and_has_explicit_empty_bounds(tmp_path: Path) -> None:
    writer = RawSegmentWriter(
        tmp_path,
        source="polymarket",
        stream="market",
        capture_session_id="capture-empty",
    )
    manifest = writer.seal()

    assert manifest.record_count == 0
    assert manifest.first_record_ordinal is None
    assert manifest.last_record_ordinal is None
    assert manifest.first_receive_at is None
    assert manifest.last_receive_at is None
    assert verify_segment(manifest.path).valid


def test_segment_cannot_cross_an_hour_partition(tmp_path: Path) -> None:
    writer = RawSegmentWriter(tmp_path, source="polymarket", stream="market")
    writer.append(b"first", receive_at="2026-07-22T14:59:59Z")

    with pytest.raises(PartitionBoundaryError, match="UTC hour"):
        writer.append(b"next-hour", receive_at="2026-07-22T15:00:00Z")

    manifest = writer.seal()
    assert manifest.record_count == 1
    assert verify_segment(manifest.path).valid


def test_duplicate_finalization_never_overwrites_existing_raw(
    tmp_path: Path,
) -> None:
    _, _, first = _make_segment(tmp_path, capture_session_id="duplicate")
    original_object = first.object_path.read_bytes()
    original_manifest = first.path.read_bytes()

    second = RawSegmentWriter(
        tmp_path,
        source="polymarket",
        stream="market",
        capture_session_id="duplicate",
    )
    second.append(b'{"event":"book"}', receive_at=UTC_TIME)
    with pytest.raises(ImmutableSegmentError, match="already exists"):
        second.seal()

    assert first.object_path.read_bytes() == original_object
    assert first.path.read_bytes() == original_manifest
    assert verify_segment(first.path).valid
    assert len(list((tmp_path / "raw").rglob("*.jsonl.zst"))) == 1
    assert len(list((tmp_path / "manifests").rglob("*.manifest.json"))) == 1


def test_manifest_directory_fsync_failure_rolls_back_commit_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import raw_store

    writer = RawSegmentWriter(
        tmp_path,
        source="polymarket",
        stream="market",
        capture_session_id="fsync-failure",
    )
    writer.append(b"exact", receive_at=UTC_TIME)
    original_fsync_directory = raw_store._fsync_directory
    failure_injected = False

    def fail_manifest_commit_once(directory, *args, **kwargs) -> None:
        nonlocal failure_injected
        context = kwargs.get("context")
        is_manifest_path = isinstance(directory, (str, Path)) and (
            "manifests" in Path(directory).parts
        )
        if not failure_injected and (
            context == "manifest_commit" or is_manifest_path
        ):
            failure_injected = True
            raise OSError("injected manifest-directory fsync failure")
        original_fsync_directory(directory, *args, **kwargs)

    monkeypatch.setattr(raw_store, "_fsync_directory", fail_manifest_commit_once)

    with pytest.raises(OSError, match="injected manifest-directory fsync failure"):
        writer.seal()

    assert failure_injected is True
    assert list((tmp_path / "manifests").rglob("*.manifest.json")) == []
    assert list((tmp_path / "raw").rglob("*.jsonl.zst")) == []


def test_raw_object_tampering_fails_verification(tmp_path: Path) -> None:
    _, _, manifest = _make_segment(tmp_path)
    manifest.object_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    data = bytearray(manifest.object_path.read_bytes())
    data[-1] ^= 0x01
    manifest.object_path.write_bytes(data)

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("SHA-256" in error for error in verification.errors)


def test_manifest_tampering_fails_verification_even_with_valid_json(
    tmp_path: Path,
) -> None:
    _, _, manifest = _make_segment(tmp_path)
    manifest.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    document = json.loads(manifest.path.read_text(encoding="utf-8"))
    document["record_count"] = 999
    _refresh_manifest_sha256(document)
    _write_canonical_manifest(manifest.path, document)

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("record_count" in error for error in verification.errors)


def test_manifest_canonical_hash_detects_any_field_tampering(tmp_path: Path) -> None:
    _, _, manifest = _make_segment(tmp_path)
    manifest.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    document = json.loads(manifest.path.read_text(encoding="utf-8"))
    document["sealed_at"] = "2030-01-01T00:00:00Z"
    _write_canonical_manifest(manifest.path, document)

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("manifest SHA-256" in error for error in verification.errors)


def test_manifest_byte_changes_cannot_hide_in_json_whitespace(tmp_path: Path) -> None:
    _, _, manifest = _make_segment(tmp_path)
    manifest.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    manifest.path.write_bytes(manifest.path.read_bytes() + b"\n")

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("canonical JSON" in error for error in verification.errors)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "../outside"),
        ("source", "/absolute"),
        ("source", "poly/market"),
        ("source", "poly\\market"),
        ("stream", "../outside"),
        ("capture_session_id", "../outside"),
        ("capture_session_id", "session/child"),
    ],
)
def test_untrusted_path_components_are_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    arguments = {
        "source": "polymarket",
        "stream": "market",
        "capture_session_id": "capture-safe",
    }
    arguments[field] = value

    with pytest.raises(RawStorePathError, match=field):
        RawSegmentWriter(tmp_path, **arguments)


def test_symlinked_raw_prefix_cannot_escape_store_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "store"
    root.mkdir()
    os.symlink(outside, root / "raw")

    with pytest.raises(RawStorePathError, match="symlink"):
        RawSegmentWriter(root, source="polymarket", stream="market")

    assert list(outside.iterdir()) == []


def test_store_root_rejects_symlinked_ancestor_and_parent_traversal(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RawStorePathError, match="symlink"):
        RawSegmentWriter(alias / "store", source="polymarket", stream="market")
    with pytest.raises(RawStorePathError, match="parent traversal"):
        RawSegmentWriter(
            tmp_path / "intended" / ".." / "escaped",
            source="polymarket",
            stream="market",
        )

    assert list(outside.iterdir()) == []


def test_writer_uses_nofollow_dirfds_for_traversal_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import raw_store

    original_open = raw_store.os.open
    original_link = raw_store.os.link
    directory_opens: list[tuple[int, int | None]] = []
    link_dirfds: list[tuple[int | None, int | None, bool]] = []

    def observe_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_DIRECTORY:
            directory_opens.append((flags, dir_fd))
        return descriptor

    def observe_link(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ) -> None:
        link_dirfds.append((src_dir_fd, dst_dir_fd, follow_symlinks))
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(raw_store.os, "open", observe_open)
    monkeypatch.setattr(raw_store.os, "link", observe_link)

    _, _, manifest = _make_segment(tmp_path)

    assert verify_segment(manifest.path).valid
    assert directory_opens
    assert all(flags & os.O_NOFOLLOW for flags, _ in directory_opens)
    assert all(flags & os.O_DIRECTORY for flags, _ in directory_opens)
    assert any(directory_fd is not None for _, directory_fd in directory_opens)
    assert len(link_dirfds) == 2
    assert all(
        isinstance(source_fd, int)
        and isinstance(destination_fd, int)
        and follow_symlinks is False
        for source_fd, destination_fd, follow_symlinks in link_dirfds
    )


def test_verifier_rejects_store_prefixes_replaced_by_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _, _, manifest = _make_segment(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "raw").rename(outside / "raw")
    (root / "manifests").rename(outside / "manifests")
    os.symlink(outside / "raw", root / "raw")
    os.symlink(outside / "manifests", root / "manifests")

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("symlink" in error for error in verification.errors)


def test_manifest_object_path_escape_fails_closed(tmp_path: Path) -> None:
    _, _, manifest = _make_segment(tmp_path)
    manifest.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    document = json.loads(manifest.path.read_text(encoding="utf-8"))
    document["object_path"] = "../../outside-secret"
    _refresh_manifest_sha256(document)
    _write_canonical_manifest(manifest.path, document)

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("object_path" in error for error in verification.errors)


def test_verifier_opens_object_once_with_nofollow_and_fstats_same_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import raw_store

    _, _, manifest = _make_segment(tmp_path)
    original_open = raw_store.os.open
    original_fstat = raw_store.os.fstat
    object_descriptors: set[int] = set()
    object_open_flags: list[int] = []
    object_fstats = 0

    def observe_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if os.fspath(path) == manifest.object_path.name:
            object_descriptors.add(descriptor)
            object_open_flags.append(flags)
        return descriptor

    def observe_fstat(descriptor: int):
        nonlocal object_fstats
        if descriptor in object_descriptors:
            object_fstats += 1
        return original_fstat(descriptor)

    monkeypatch.setattr(raw_store.os, "open", observe_open)
    monkeypatch.setattr(raw_store.os, "fstat", observe_fstat)

    verification = verify_segment(manifest.path)

    assert verification.valid is True
    assert len(object_open_flags) == 1
    assert object_open_flags[0] & os.O_NOFOLLOW
    assert object_fstats >= 2


def test_verifier_rejects_path_inode_replacement_between_hash_and_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import raw_store

    _, _, original = _make_segment(
        tmp_path,
        capture_session_id="same-session",
        payload=b"original",
    )
    _, _, replacement = _make_segment(
        tmp_path,
        capture_session_id="same-session",
        payload=b"replacement",
    )
    real_sha256 = hashlib.sha256
    replaced = False

    class ReplacingDigest:
        def __init__(self, data: bytes = b"") -> None:
            self._digest = real_sha256(data)
            self._object_candidate = data == b""

        def update(self, data: bytes) -> None:
            self._digest.update(data)

        def hexdigest(self) -> str:
            nonlocal replaced
            value = self._digest.hexdigest()
            if self._object_candidate and not replaced:
                os.replace(replacement.object_path, original.object_path)
                replaced = True
            return value

        def __getattr__(self, name: str):
            return getattr(self._digest, name)

    monkeypatch.setattr(raw_store.hashlib, "sha256", ReplacingDigest)

    verification = verify_segment(original.path)

    assert replaced is True
    assert verification.valid is False
    assert any("changed during verification" in error for error in verification.errors)


def test_verifier_rechecks_manifest_inode_after_object_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market import raw_store

    _, _, manifest = _make_segment(tmp_path)
    replacement = manifest.path.parent / "replacement.manifest.json"
    replacement.write_bytes(b'{"replacement":true}\n')
    original_sha256_fd = raw_store._sha256_fd
    replaced = False

    def replace_manifest_then_hash(descriptor: int) -> str:
        nonlocal replaced
        if not replaced:
            os.replace(replacement, manifest.path)
            replaced = True
        return original_sha256_fd(descriptor)

    monkeypatch.setattr(raw_store, "_sha256_fd", replace_manifest_then_hash)

    verification = verify_segment(manifest.path, root=tmp_path)

    assert replaced is True
    assert verification.valid is False
    assert any(
        "manifest changed during verification" in error
        for error in verification.errors
    )


def test_verified_segment_reader_returns_exact_validated_payloads(
    tmp_path: Path,
) -> None:
    payloads = (b"first exact payload", b"\x00second\xff")
    writer = RawSegmentWriter(
        tmp_path,
        source="polymarket",
        stream="market",
        capture_session_id="verified-reader",
    )
    for payload in payloads:
        writer.append(payload, receive_at=UTC_TIME)
    manifest = writer.seal()

    verified = read_verified_segment(manifest.path, root=tmp_path)

    assert verified.manifest == manifest
    assert verified.payloads == payloads


def test_record_ordinals_are_contiguous_and_payload_duplicates_are_preserved(
    tmp_path: Path,
) -> None:
    writer = RawSegmentWriter(tmp_path, source="polymarket", stream="market")
    assert writer.append(b"duplicate", receive_at=UTC_TIME) == 0
    assert writer.append(b"duplicate", receive_at=UTC_TIME) == 1
    manifest = writer.seal()

    records = _records(manifest.object_path)
    assert [record["record_ordinal"] for record in records] == [0, 1]
    assert records[0]["payload_sha256"] == records[1]["payload_sha256"]
    assert verify_segment(manifest.path).valid


@pytest.mark.parametrize(
    "receive_at",
    [
        "2026-07-22 14:03:04Z",
        "2026-07-22T14:03:04+00:00",
        "2026-W30-3T14:03:04Z",
        "not-a-time",
    ],
)
def test_receive_time_requires_canonical_utc(
    tmp_path: Path, receive_at: str
) -> None:
    writer = RawSegmentWriter(tmp_path, source="polymarket", stream="market")
    with pytest.raises(ValueError, match="UTC"):
        writer.append(b"frame", receive_at=receive_at)


def test_raw_capture_contract_and_data_boundaries_are_published() -> None:
    schema_path = PROJECT_ROOT / "contracts" / "raw-capture" / "v0.schema.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    assert schema["contract_version"] == "v0"
    assert schema["$defs"]["raw_record"]["additionalProperties"] is False
    assert schema["$defs"]["segment_manifest"]["additionalProperties"] is False
    assert "manifest_sha256" in schema["$defs"]["segment_manifest"]["required"]
    assert schema["x-storage-discipline"] == {
        "append_only": True,
        "content_addressed": True,
        "manifest_is_commit_point": True,
        "overwrite_existing_final_path": False,
    }

    raw_readme = (PROJECT_ROOT / "data" / "raw" / "README.md").read_text(
        encoding="utf-8"
    )
    manifest_readme = (
        PROJECT_ROOT / "data" / "manifests" / "README.md"
    ).read_text(encoding="utf-8")
    normalized_readme = (
        PROJECT_ROOT / "data" / "normalized" / "README.md"
    ).read_text(encoding="utf-8")
    assert "append-only" in raw_readme
    assert "commit point" in manifest_readme
    assert "reconstruct" in normalized_readme
    boundary_text = "\n".join((raw_readme, manifest_readme, normalized_readme))
    assert "not a WORM control" in boundary_text
    assert "Object Lock" in boundary_text
    assert "retention" in boundary_text
    assert "deny-delete" in boundary_text
    assert "independent anchor" in boundary_text
    assert "OPEN" in boundary_text
