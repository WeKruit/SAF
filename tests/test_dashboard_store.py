from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from prediction_market import dashboard_store
from prediction_market.cli import publish_x13_dashboard
from prediction_market.dashboard_store import (
    BotoDashboardObjectStore,
    DashboardConditionalRequestConflict,
    DashboardObjectAlreadyExists,
    DashboardS3Config,
    DashboardStoreError,
    load_dashboard_s3_config,
    publish_dashboard_export,
    verify_dashboard_publication,
)
from prediction_market.sports import nfl_x13_dashboard


_TASK2_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_dashboard_store_task2_fixtures",
    Path(__file__).parent / "sports/test_nfl_x13_dashboard.py",
)
assert _TASK2_FIXTURE_SPEC is not None
assert _TASK2_FIXTURE_SPEC.loader is not None
task2_fixtures = importlib.util.module_from_spec(_TASK2_FIXTURE_SPEC)
sys.modules[_TASK2_FIXTURE_SPEC.name] = task2_fixtures
_TASK2_FIXTURE_SPEC.loader.exec_module(task2_fixtures)


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


@dataclass(frozen=True)
class PutCall:
    key: str
    checksum_sha256: str | None
    create_only: bool


class FakeS3:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config = DashboardS3Config(
            endpoint="https://s3.example.test",
            region="us-east-1",
            bucket="private-test",
            access_key_id="test-access",
            secret_access_key="test-secret",
            prefix="",
        )
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.put_calls: list[PutCall] = []
        self.fail_key: str | None = None
        self.corrupt_key: str | None = None
        self.lose_response_key: str | None = None
        self.race_key: str | None = None
        self.race_payload: bytes | None = None
        self.read_counts: dict[str, int] = {}
        self.replace_on_second_read_key: str | None = None
        self.second_read_payload: bytes | None = None
        self.conditional_conflict_key: str | None = None
        self.conditional_conflicts_remaining = 0
        self.conditional_conflict_payload: bytes | None = None
        self.upload_delay_seconds = 0.0
        self.active_content_puts = 0
        self.peak_content_puts = 0
        self.read_delay_seconds = 0.0
        self.active_content_reads = 0
        self.peak_content_reads = 0

    def head_object(self, key: str) -> dict[str, object] | None:
        with self.lock:
            payload = self.objects.get(key)
            if payload is None:
                return None
            return {
                "key": key,
                "byte_size": len(payload),
                "metadata": dict(self.metadata.get(key, {})),
            }

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        metadata: dict[str, str],
        create_only: bool,
    ) -> None:
        if create_only and self.upload_delay_seconds:
            with self.lock:
                self.active_content_puts += 1
                self.peak_content_puts = max(
                    self.peak_content_puts, self.active_content_puts
                )
            try:
                time.sleep(self.upload_delay_seconds)
            finally:
                with self.lock:
                    self.active_content_puts -= 1
        with self.lock:
            self.put_calls.append(
                PutCall(
                    key=key,
                    checksum_sha256=metadata.get("sha256"),
                    create_only=create_only,
                )
            )
            if (
                create_only
                and key == self.conditional_conflict_key
                and self.conditional_conflicts_remaining > 0
            ):
                self.conditional_conflicts_remaining -= 1
                if self.conditional_conflict_payload is not None:
                    self.objects[key] = self.conditional_conflict_payload
                    self.metadata[key] = {
                        "sha256": _sha256(
                            self.conditional_conflict_payload
                        )
                    }
                raise DashboardConditionalRequestConflict(key)
            if create_only and key == self.race_key:
                assert self.race_payload is not None
                self.objects[key] = self.race_payload
                self.metadata[key] = {
                    "sha256": _sha256(self.race_payload)
                }
                raise DashboardObjectAlreadyExists(key)
            if create_only and key in self.objects:
                raise DashboardObjectAlreadyExists(key)
            if key == self.fail_key:
                raise OSError("injected upload failure")
            self.objects[key] = (
                b"remote corruption" if key == self.corrupt_key else payload
            )
            self.metadata[key] = dict(metadata)
            if key == self.lose_response_key:
                raise OSError("injected lost response after commit")

    def read_object_chunks(
        self, key: str, *, chunk_size: int = 1024 * 1024
    ) -> Iterator[bytes]:
        content_read = "/objects/sha256/" in key
        if content_read and self.read_delay_seconds:
            with self.lock:
                self.active_content_reads += 1
                self.peak_content_reads = max(
                    self.peak_content_reads, self.active_content_reads
                )
            try:
                time.sleep(self.read_delay_seconds)
            finally:
                with self.lock:
                    self.active_content_reads -= 1
        with self.lock:
            self.read_counts[key] = self.read_counts.get(key, 0) + 1
            try:
                payload = self.objects[key]
            except KeyError as error:
                raise KeyError(key) from error
            if (
                key == self.replace_on_second_read_key
                and self.read_counts[key] == 2
            ):
                assert self.second_read_payload is not None
                self.objects[key] = self.second_read_payload
                payload = self.second_read_payload
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]


def _write_forged_export(tmp_path: Path) -> Path:
    output = tmp_path / "export"
    batch = output / ("batch=sha256-" + "1" * 64)
    asset_directory = batch / "catalog"
    asset_directory.mkdir(parents=True)
    assets: list[dict[str, object]] = []
    for role, payload in (
        ("catalog", {"schema": "catalog_v1", "games": []}),
        ("core", {"schema": "core_v1", "events": []}),
    ):
        raw = _canonical_json(payload)
        object_sha256 = _sha256(raw)
        relative = (
            f"catalog/{object_sha256.replace(':', '-')}-{role}.json"
        )
        (batch / relative).write_bytes(raw)
        assets.append(
            {
                "relative_path": relative,
                "role": role,
                "media_type": "application/json",
                "schema": str(payload["schema"]),
                "source_sha256": "sha256:" + "2" * 64,
                "object_sha256": object_sha256,
                "byte_length": len(raw),
            }
        )
    manifest = {
        "schema": "nfl_x13_dashboard_publish_manifest_v1",
        "experiment_id": "X-13",
        "batch_id": "sha256:" + "1" * 64,
        "builder_version": "test-builder-v1",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "claim_boundary": {
            "causality": False,
            "real_latency": False,
            "execution": False,
            "tradable_alpha": False,
        },
        "game_count": 20,
        "asset_count": len(assets),
        "assets": assets,
    }
    manifest_raw = _canonical_json(manifest)
    manifest_sha256 = _sha256(manifest_raw)
    manifest_directory = batch / "manifests"
    manifest_directory.mkdir()
    (
        manifest_directory
        / f"{manifest_sha256.replace(':', '-')}.json"
    ).write_bytes(manifest_raw)
    return output


def _write_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    batch = task2_fixtures._write_batch(tmp_path / "source")
    task2_fixtures._stub_batch_verifier(monkeypatch)
    result = nfl_x13_dashboard.export_x13_dashboard(
        batch, tmp_path / "export"
    )
    return result.publish_root.parent


def _content_keys(fake_s3: FakeS3) -> list[str]:
    return sorted(
        key
        for key in fake_s3.objects
        if key != "dashboard/x13/published/latest.json"
    )


def test_publish_uploads_content_objects_before_latest_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)

    result = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    assert fake_s3.put_calls[-1].key == (
        "dashboard/x13/published/latest.json"
    )
    assert all(
        call.checksum_sha256 for call in fake_s3.put_calls[:-1]
    )
    assert all(call.create_only for call in fake_s3.put_calls[:-1])
    assert fake_s3.put_calls[-1].create_only is False
    assert result.verified_object_count == len(fake_s3.put_calls) - 1
    assert result.pointer_key == "dashboard/x13/published/latest.json"
    pointer = json.loads(fake_s3.objects[result.pointer_key])
    assert pointer["manifest_key"] in _content_keys(fake_s3)
    assert pointer["manifest_sha256"] == result.manifest_sha256
    assert result.pointer_sha256 == _sha256(
        fake_s3.objects[result.pointer_key]
    )


def test_content_publication_is_bounded_parallel_and_pointer_remains_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    fake_s3.upload_delay_seconds = 0.01
    export_root = _write_export(tmp_path, monkeypatch)

    result = publish_dashboard_export(
        export_root,
        fake_s3.config,
        object_store=fake_s3,
        max_workers=4,
    )

    assert 1 < fake_s3.peak_content_puts <= 4
    assert fake_s3.put_calls[-1].key == result.pointer_key
    assert fake_s3.put_calls[-1].create_only is False
    assert result.verified_object_count == len(_content_keys(fake_s3))
    assert result.total_verified_bytes == sum(
        len(fake_s3.objects[key]) for key in _content_keys(fake_s3)
    )


def test_parallel_content_failure_never_writes_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    fake_s3.upload_delay_seconds = 0.005
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    failed_sha = manifest["assets"][0]["object_sha256"]
    fake_s3.fail_key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, failed_sha
    )

    with pytest.raises(DashboardStoreError, match="upload"):
        publish_dashboard_export(
            export_root,
            fake_s3.config,
            object_store=fake_s3,
            max_workers=4,
        )

    assert "dashboard/x13/published/latest.json" not in fake_s3.objects
    assert fake_s3.active_content_puts == 0


@pytest.mark.parametrize("workers", [0, -1, 33, True])
def test_invalid_worker_count_is_rejected_before_publication(
    tmp_path: Path,
    workers: int,
) -> None:
    fake_s3 = FakeS3()
    with pytest.raises(DashboardStoreError, match="workers"):
        publish_dashboard_export(
            tmp_path / "unused",
            fake_s3.config,
            object_store=fake_s3,
            max_workers=workers,
        )
    assert fake_s3.put_calls == []


def test_existing_identical_content_and_pointer_are_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    first = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )
    original_objects = dict(fake_s3.objects)
    fake_s3.put_calls.clear()

    second = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    assert second.manifest_sha256 == first.manifest_sha256
    assert second.verified_object_count == first.verified_object_count
    assert fake_s3.put_calls == []
    assert fake_s3.objects == original_objects


def test_conflicting_immutable_object_fails_without_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    first_sha = manifest["assets"][0]["object_sha256"]
    key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, first_sha
    )
    fake_s3.objects[key] = b"foreign bytes"
    fake_s3.metadata[key] = {"sha256": first_sha}

    with pytest.raises(DashboardStoreError, match="conflict|mismatch"):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert "dashboard/x13/published/latest.json" not in fake_s3.objects


def test_remote_hash_mismatch_after_upload_has_no_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    first_sha = manifest["assets"][0]["object_sha256"]
    fake_s3.corrupt_key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, first_sha
    )

    with pytest.raises(DashboardStoreError, match="hash mismatch"):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert "dashboard/x13/published/latest.json" not in fake_s3.objects


def test_failed_content_upload_has_no_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    first_sha = manifest["assets"][0]["object_sha256"]
    fake_s3.fail_key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, first_sha
    )

    with pytest.raises(DashboardStoreError, match="upload"):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert "dashboard/x13/published/latest.json" not in fake_s3.objects


def test_create_only_race_with_identical_object_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    asset = manifest["assets"][0]
    key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, asset["object_sha256"]
    )
    payload = (batch / asset["relative_path"]).read_bytes()
    fake_s3.race_key = key
    fake_s3.race_payload = payload

    result = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    assert result.verified_object_count > 0
    assert fake_s3.objects[key] == payload
    assert fake_s3.put_calls[-1].key == (
        "dashboard/x13/published/latest.json"
    )


def test_create_only_race_with_conflicting_object_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    asset = manifest["assets"][0]
    key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, asset["object_sha256"]
    )
    foreign = b"concurrent foreign bytes"
    fake_s3.race_key = key
    fake_s3.race_payload = foreign

    with pytest.raises(DashboardStoreError, match="conflict|hash mismatch"):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert fake_s3.objects[key] == foreign
    assert "dashboard/x13/published/latest.json" not in fake_s3.objects


def test_409_race_with_identical_object_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    asset = manifest["assets"][0]
    key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, asset["object_sha256"]
    )
    payload = (batch / asset["relative_path"]).read_bytes()
    fake_s3.conditional_conflict_key = key
    fake_s3.conditional_conflicts_remaining = 1
    fake_s3.conditional_conflict_payload = payload

    result = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    assert result.verified_object_count > 0
    assert fake_s3.objects[key] == payload
    assert [
        call.create_only
        for call in fake_s3.put_calls
        if call.key == key
    ] == [True]


def test_409_race_with_conflicting_object_fails_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    asset = manifest["assets"][0]
    key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, asset["object_sha256"]
    )
    foreign = b"409 concurrent foreign bytes"
    fake_s3.conditional_conflict_key = key
    fake_s3.conditional_conflicts_remaining = 1
    fake_s3.conditional_conflict_payload = foreign

    with pytest.raises(DashboardStoreError, match="conflict|hash mismatch"):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert fake_s3.objects[key] == foreign
    assert "dashboard/x13/published/latest.json" not in fake_s3.objects


def test_409_not_yet_visible_retries_create_only_with_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    asset = manifest["assets"][0]
    key = dashboard_store.dashboard_object_key(
        fake_s3.config.prefix, asset["object_sha256"]
    )
    fake_s3.conditional_conflict_key = key
    fake_s3.conditional_conflicts_remaining = 2

    publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    calls = [call for call in fake_s3.put_calls if call.key == key]
    assert len(calls) == 3
    assert all(call.create_only for call in calls)
    assert all(call.checksum_sha256 == asset["object_sha256"] for call in calls)


def test_pointer_lost_response_committed_exact_payload_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    pointer_key = "dashboard/x13/published/latest.json"
    fake_s3.lose_response_key = pointer_key

    result = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    assert result.pointer_key == pointer_key
    assert json.loads(fake_s3.objects[pointer_key])["manifest_sha256"] == (
        result.manifest_sha256
    )


def test_corrupt_pointer_commit_is_explicitly_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    pointer_key = "dashboard/x13/published/latest.json"
    fake_s3.corrupt_key = pointer_key

    with pytest.raises(
        DashboardStoreError, match="indeterminate.*pointer|pointer.*indeterminate"
    ):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert fake_s3.objects[pointer_key] == b"remote corruption"


def test_prefix_escape_is_rejected_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest_path = next((batch / "manifests").glob("*.json"))
    manifest = json.loads(manifest_path.read_bytes())
    manifest["assets"][0]["relative_path"] = "../outside.json"
    manifest_path.unlink()
    replacement = _canonical_json(manifest)
    replacement_sha = _sha256(replacement)
    (
        manifest_path.parent
        / f"{replacement_sha.replace(':', '-')}.json"
    ).write_bytes(replacement)

    with pytest.raises(
        DashboardStoreError, match="Task 2|projection"
    ):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert fake_s3.put_calls == []


def test_s3_prefix_escape_is_rejected_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    config = DashboardS3Config(
        endpoint=fake_s3.config.endpoint,
        region=fake_s3.config.region,
        bucket=fake_s3.config.bucket,
        access_key_id="test-access",
        secret_access_key="test-secret",
        prefix="../raw/pmxt-v2",
    )

    with pytest.raises(DashboardStoreError, match="prefix.*escape"):
        publish_dashboard_export(
            export_root, config, object_store=fake_s3
        )

    assert fake_s3.put_calls == []


def test_local_asset_hash_mismatch_is_rejected_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    batch = next(export_root.glob("batch=*"))
    manifest = json.loads(next((batch / "manifests").glob("*.json")).read_bytes())
    asset = batch / manifest["assets"][0]["relative_path"]
    asset.write_bytes(b"x" * len(asset.read_bytes()))

    with pytest.raises(
        DashboardStoreError, match="Task 2|projection"
    ):
        publish_dashboard_export(
            export_root, fake_s3.config, object_store=fake_s3
        )

    assert fake_s3.put_calls == []


def test_missing_s3_environment_values_fail_closed() -> None:
    with pytest.raises(
        DashboardStoreError,
        match="PM_DATA_S3_SECRET_ACCESS_KEY",
    ):
        load_dashboard_s3_config(
            env={
                "PM_DATA_S3_ENDPOINT": "https://s3.example.test",
                "PM_DATA_S3_REGION": "us-east-1",
                "PM_DATA_S3_BUCKET": "private",
                "PM_DATA_S3_ACCESS_KEY_ID": "access",
            }
        )


def test_verify_rereads_pointer_manifest_and_every_remote_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    published = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )

    verified = verify_dashboard_publication(
        fake_s3.config, object_store=fake_s3
    )

    assert verified.manifest_sha256 == published.manifest_sha256
    assert verified.verified_object_count == published.verified_object_count
    assert verified.pointer_sha256 == published.pointer_sha256

    fake_s3.objects[_content_keys(fake_s3)[0]] = b"tampered remote"
    with pytest.raises(DashboardStoreError, match="hash mismatch"):
        verify_dashboard_publication(
            fake_s3.config, object_store=fake_s3
        )


def test_remote_verification_is_bounded_parallel_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    published = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )
    fake_s3.read_delay_seconds = 0.01
    fake_s3.peak_content_reads = 0

    verified = verify_dashboard_publication(
        fake_s3.config,
        object_store=fake_s3,
        max_workers=4,
    )

    assert 1 < fake_s3.peak_content_reads <= 4
    assert verified.verified_object_count == (
        published.verified_object_count
    )
    assert verified.total_verified_bytes == (
        published.total_verified_bytes
    )
    assert verified.pointer_sha256 == published.pointer_sha256


def test_parallel_remote_verification_failure_waits_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )
    fake_s3.objects[_content_keys(fake_s3)[0]] = b"tampered"
    fake_s3.read_delay_seconds = 0.005

    with pytest.raises(DashboardStoreError, match="hash mismatch"):
        verify_dashboard_publication(
            fake_s3.config,
            object_store=fake_s3,
            max_workers=4,
        )

    assert fake_s3.active_content_reads == 0


@pytest.mark.parametrize("workers", [0, -1, 33, True])
def test_invalid_verify_worker_count_is_rejected_before_remote_read(
    workers: int,
) -> None:
    fake_s3 = FakeS3()
    with pytest.raises(DashboardStoreError, match="workers"):
        verify_dashboard_publication(
            fake_s3.config,
            object_store=fake_s3,
            max_workers=workers,
        )
    assert fake_s3.read_counts == {}


def test_verify_fails_if_pointer_changes_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_s3 = FakeS3()
    export_root = _write_export(tmp_path, monkeypatch)
    published = publish_dashboard_export(
        export_root, fake_s3.config, object_store=fake_s3
    )
    fake_s3.read_counts.clear()
    fake_s3.replace_on_second_read_key = published.pointer_key
    fake_s3.second_read_payload = _canonical_json(
        {
            "schema": "x13_dashboard_latest_pointer_v1",
            "experiment_id": "X-13",
            "batch_id": "sha256:" + "f" * 64,
            "manifest_sha256": "sha256:" + "e" * 64,
            "manifest_key": "dashboard/x13/objects/sha256/ee/"
            + "e" * 64
            + ".json",
        }
    )

    with pytest.raises(DashboardStoreError, match="pointer.*changed"):
        verify_dashboard_publication(
            fake_s3.config, object_store=fake_s3
        )


def test_forged_self_consistent_projection_is_rejected(
    tmp_path: Path,
) -> None:
    fake_s3 = FakeS3()
    forged = _write_forged_export(tmp_path)

    with pytest.raises(
        DashboardStoreError, match="Task 2|projection|builder|descriptor"
    ):
        publish_dashboard_export(
            forged, fake_s3.config, object_store=fake_s3
        )

    assert fake_s3.put_calls == []


def test_boto_create_only_uses_precondition_and_stream_body_is_closed() -> None:
    class Body:
        def __init__(self) -> None:
            self.closed = False

        def iter_chunks(self, *, chunk_size: int) -> Iterator[bytes]:
            assert chunk_size > 0
            yield b"payload"

        def close(self) -> None:
            self.closed = True

    class Client:
        def __init__(self) -> None:
            self.body = Body()
            self.put_kwargs: dict[str, object] = {}

        def put_object(self, **kwargs: object) -> None:
            self.put_kwargs = kwargs

        def get_object(self, **kwargs: object) -> dict[str, object]:
            return {"Body": self.body}

    client = Client()
    store = BotoDashboardObjectStore(client, bucket="private")
    payload = b"payload"
    store.put_bytes(
        "dashboard/x13/object.json",
        payload,
        metadata={"sha256": _sha256(payload)},
        create_only=True,
    )
    assert client.put_kwargs["IfNoneMatch"] == "*"
    assert client.put_kwargs["ChecksumSHA256"]
    assert b"".join(
        store.read_object_chunks("dashboard/x13/object.json")
    ) == payload
    assert client.body.closed is True


def test_boto_precondition_failure_maps_to_concurrent_create() -> None:
    from botocore.exceptions import ClientError

    class Client:
        def put_object(self, **kwargs: object) -> None:
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )

    store = BotoDashboardObjectStore(Client(), bucket="private")
    payload = b"payload"
    with pytest.raises(DashboardObjectAlreadyExists):
        store.put_bytes(
            "dashboard/x13/object.json",
            payload,
            metadata={"sha256": _sha256(payload)},
            create_only=True,
        )


def test_boto_409_maps_to_conditional_request_conflict() -> None:
    from botocore.exceptions import ClientError

    class Client:
        def put_object(self, **kwargs: object) -> None:
            raise ClientError(
                {
                    "Error": {"Code": "ConditionalRequestConflict"},
                    "ResponseMetadata": {"HTTPStatusCode": 409},
                },
                "PutObject",
            )

    store = BotoDashboardObjectStore(Client(), bucket="private")
    payload = b"payload"
    with pytest.raises(DashboardConditionalRequestConflict):
        store.put_bytes(
            "dashboard/x13/object.json",
            payload,
            metadata={"sha256": _sha256(payload)},
            create_only=True,
        )


def test_cli_has_exact_three_subcommands() -> None:
    parser = publish_x13_dashboard._build_parser()
    subparser_actions = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    ]
    assert len(subparser_actions) == 1
    assert set(subparser_actions[0].choices) == {
        "export",
        "publish-s3",
        "verify-s3",
    }
    arguments = parser.parse_args(
        [
            "publish-s3",
            "--export-root",
            "output",
            "--workers",
            "6",
        ]
    )
    assert arguments.workers == 6
    verify_arguments = parser.parse_args(
        ["verify-s3", "--workers", "5"]
    )
    assert verify_arguments.workers == 5


def test_project_registers_dashboard_cli() -> None:
    program_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (program_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["scripts"]["publish-x13-dashboard"] == (
        "prediction_market.cli.publish_x13_dashboard:main"
    )


def test_dashboard_store_has_no_pmxt_store_dependency() -> None:
    program_root = Path(__file__).resolve().parents[1]
    source = (
        program_root / "src/prediction_market/dashboard_store.py"
    ).read_text(encoding="utf-8")
    assert "prediction_market.pmxt" not in source
    assert "/raw/pmxt" not in source


def test_migration_has_exact_three_rls_tables_and_no_policy(
) -> None:
    program_root = Path(__file__).resolve().parents[1]
    sql = (
        program_root
        / "supabase/migrations/202607250001_x13_dashboard_index.sql"
    ).read_text(encoding="utf-8")
    tables = re.findall(
        r"create\s+table\s+([a-z_]+)", sql, flags=re.IGNORECASE
    )
    assert tables == [
        "dashboard_batches",
        "dashboard_games",
        "dashboard_assets",
    ]
    assert len(
        re.findall(
            r"alter\s+table\s+dashboard_(?:batches|games|assets)"
            r"\s+enable\s+row\s+level\s+security",
            sql,
            flags=re.IGNORECASE,
        )
    ) == 3
    assert "create policy" not in sql.lower()
    assert "anon" not in sql.lower()
