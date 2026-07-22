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
    }
    assert document["object_path"] == manifest.object_path.relative_to(tmp_path).as_posix()
    assert document["object_sha256"] == manifest.object_sha256
    assert document["compression"] == "zstd"
    assert document["record_encoding"] == "canonical-json-lines-v0"


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
    _write_canonical_manifest(manifest.path, document)

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("record_count" in error for error in verification.errors)


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
    _write_canonical_manifest(manifest.path, document)

    verification = verify_segment(manifest.path)

    assert verification.valid is False
    assert any("object_path" in error for error in verification.errors)


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
