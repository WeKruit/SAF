from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _FakeObjectStore:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.metadata: dict[str, dict[str, str]] = {}
        self.puts: list[tuple[str, Path, dict[str, str]]] = []

    def head_object(self, key: str):
        payload = self.objects.get(key)
        if payload is None:
            return None
        return {
            "key": key,
            "byte_size": len(payload),
            "metadata": dict(self.metadata.get(key, {})),
        }

    def put_file(self, key: str, path: Path, *, metadata: dict[str, str]) -> None:
        self.puts.append((key, path, dict(metadata)))
        self.objects[key] = path.read_bytes()
        self.metadata[key] = dict(metadata)

    def list_objects(self, prefix: str):
        from prediction_market.pmxt.data_store import RemoteObject

        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield RemoteObject(
                    key=key,
                    byte_size=len(self.objects[key]),
                    metadata=dict(self.metadata.get(key, {})),
                )

    def read_object_chunks(self, key: str, *, chunk_size: int = 1024 * 1024):
        payload = self.objects[key]
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]


def test_upload_pmxt_uploads_only_local_canonical_objects(tmp_path: Path) -> None:
    from prediction_market.pmxt import data_store

    payload = b"canonical parquet bytes"
    sha256 = _digest(payload)
    digest_hex = sha256.removeprefix("sha256:")
    canonical = (
        tmp_path
        / "pmxt-v2"
        / "sha256"
        / digest_hex[:2]
        / f"{digest_hex}.parquet"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(payload)
    alias = (
        tmp_path
        / "raw"
        / "source=pmxt"
        / "dataset=DS-PMXT-V2"
        / "version=v2"
        / "partition=hour-2026-01-01T00"
        / f"{digest_hex}.parquet"
    )
    alias.parent.mkdir(parents=True)
    alias.write_bytes(payload)
    store = _FakeObjectStore()

    report = data_store.upload_pmxt(
        tmp_path, store, prefix="prediction-market"
    )

    expected_key = (
        f"prediction-market/raw/pmxt-v2/sha256/{digest_hex[:2]}/"
        f"{digest_hex}.parquet"
    )
    assert report["summary"] == {
        "local_object_count": 1,
        "uploaded_count": 1,
        "skipped_count": 0,
        "failure_count": 0,
        "total_bytes": len(payload),
    }
    assert store.puts == [(expected_key, canonical, {"sha256": sha256})]
    assert store.objects[expected_key] == payload


def test_verify_pmxt_remote_recomputes_sha256_and_reports_mismatch() -> None:
    from prediction_market.pmxt import data_store

    payload = b"good remote bytes"
    good_sha = _digest(payload)
    good_key = data_store.pmxt_canonical_key("prediction-market", good_sha)
    bad_sha = "sha256:" + "b" * 64
    bad_key = data_store.pmxt_canonical_key("prediction-market", bad_sha)
    store = _FakeObjectStore({good_key: payload, bad_key: b"wrong bytes"})

    report = data_store.verify_pmxt_remote(store, prefix="prediction-market")

    assert report["summary"]["remote_object_count"] == 2
    assert report["summary"]["verified_count"] == 1
    assert report["summary"]["failure_count"] == 1
    by_sha = {item["sha256"]: item for item in report["objects"]}
    assert by_sha[good_sha]["status"] == "verified"
    assert by_sha[bad_sha]["status"] == "hash_mismatch"


def test_status_pmxt_compares_local_remote_and_manifest_references(
    tmp_path: Path,
) -> None:
    from prediction_market.pmxt import data_store

    local_payload = b"local bytes"
    local_sha = _digest(local_payload)
    local_hex = local_sha.removeprefix("sha256:")
    local_path = (
        tmp_path
        / "pmxt-v2"
        / "sha256"
        / local_hex[:2]
        / f"{local_hex}.parquet"
    )
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(local_payload)
    missing_sha = "sha256:" + "c" * 64
    native_missing_sha = "sha256:" + "a" * 64
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    (manifest_root / "day.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_path": (
                            "raw/source=pmxt/dataset=DS-PMXT-V2/version=v2/"
                            "partition=hour-2026-01-01T00/a.parquet"
                        ),
                        "object_sha256": local_sha,
                    },
                    {
                        "object_path": (
                            "raw/source=pmxt/dataset=DS-PMXT-V2/version=v2/"
                            "partition=hour-2026-01-01T01/missing.parquet"
                        ),
                        "object_sha256": missing_sha,
                    },
                    {
                        "object_path": "raw/source=kalshi/other.parquet",
                        "object_sha256": "sha256:" + "e" * 64,
                    },
                    {
                        "native_object_path": (
                            "raw/source=pmxt/dataset=DS-PMXT-V2/version=v2/"
                            "partition=hour-2026-01-01T02/native-missing.parquet"
                        ),
                        "object_sha256": native_missing_sha,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    remote_inventory = {
        "objects": [
            {
                "key": data_store.pmxt_canonical_key("prediction-market", local_sha),
                "sha256": local_sha,
                "byte_size": len(local_payload),
                "status": "verified",
            }
        ]
    }

    report = data_store.status_pmxt(
        tmp_path,
        prefix="prediction-market",
        manifest_roots=(manifest_root,),
        remote_inventory=remote_inventory,
    )

    assert report["summary"] == {
        "local_object_count": 1,
        "remote_verified_object_count": 1,
        "manifest_reference_count": 3,
        "local_missing_remote_count": 0,
        "manifest_missing_local_count": 2,
        "manifest_missing_remote_count": 2,
        "remote_missing_local_count": 0,
    }
    assert report["manifest_missing_local"] == [native_missing_sha, missing_sha]
    assert report["manifest_missing_remote"] == [native_missing_sha, missing_sha]


def test_status_pmxt_requires_verified_remote_inventory(tmp_path: Path) -> None:
    from prediction_market.pmxt import data_store

    with pytest.raises(data_store.DataStoreError, match="verified remote inventory"):
        data_store.status_pmxt(tmp_path, prefix="prediction-market")


def test_upload_pmxt_rejects_local_content_address_mismatch(tmp_path: Path) -> None:
    from prediction_market.pmxt import data_store

    wrong_hex = "d" * 64
    path = tmp_path / "pmxt-v2" / "sha256" / "dd" / f"{wrong_hex}.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the declared digest")

    with pytest.raises(data_store.DataStoreError, match="local PMXT hash mismatch"):
        data_store.upload_pmxt(tmp_path, _FakeObjectStore(), prefix="prediction-market")


def test_pm_data_store_cli_upload_writes_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market.cli import pm_data_store

    payload = b"canonical parquet bytes"
    sha256 = _digest(payload)
    digest_hex = sha256.removeprefix("sha256:")
    canonical = (
        tmp_path
        / "pmxt-v2"
        / "sha256"
        / digest_hex[:2]
        / f"{digest_hex}.parquet"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(payload)
    store = _FakeObjectStore()
    monkeypatch.setattr(
        pm_data_store,
        "_store_and_prefix",
        lambda _args: (store, "prediction-market"),
    )
    output = tmp_path / "upload-report.json"

    exit_code = pm_data_store.main(
        ["upload-pmxt", "--raw-root", str(tmp_path), "--output", str(output)]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["audit_kind"] == "pmxt_s3_upload_v1"
    assert report["summary"]["uploaded_count"] == 1
    assert len(store.puts) == 1


def test_pm_data_store_cli_returns_nonzero_when_status_has_missing_remote(
    tmp_path: Path,
) -> None:
    from prediction_market.cli import pm_data_store

    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    missing_sha = "sha256:" + "f" * 64
    (manifest_root / "day.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_path": (
                            "raw/source=pmxt/dataset=DS-PMXT-V2/version=v2/"
                            "partition=hour-2026-01-01T00/missing.parquet"
                        ),
                        "object_sha256": missing_sha,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    remote_inventory = tmp_path / "remote.json"
    remote_inventory.write_text(json.dumps({"objects": []}), encoding="utf-8")
    output = tmp_path / "status-report.json"

    exit_code = pm_data_store.main(
        [
            "status-pmxt",
            "--raw-root",
            str(tmp_path),
            "--manifest-root",
            str(manifest_root),
            "--remote-inventory",
            str(remote_inventory),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["summary"][
        "manifest_missing_remote_count"
    ] == 1


def test_prune_pmxt_local_dry_run_does_not_delete_verified_objects(
    tmp_path: Path,
) -> None:
    from prediction_market.pmxt import data_store

    payload = b"verified local bytes"
    sha256 = _digest(payload)
    digest_hex = sha256.removeprefix("sha256:")
    canonical = (
        tmp_path
        / "pmxt-v2"
        / "sha256"
        / digest_hex[:2]
        / f"{digest_hex}.parquet"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(payload)
    remote_inventory = {"objects": [{"sha256": sha256, "status": "verified"}]}

    report = data_store.prune_pmxt_local(
        tmp_path,
        remote_inventory=remote_inventory,
        execute=False,
        confirm_no_local_readers=None,
    )

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["deleted_count"] == 0
    assert canonical.exists()


def test_prune_pmxt_local_execute_requires_literal_confirmation(
    tmp_path: Path,
) -> None:
    from prediction_market.pmxt import data_store

    payload = b"verified local bytes"
    sha256 = _digest(payload)
    digest_hex = sha256.removeprefix("sha256:")
    canonical = (
        tmp_path
        / "pmxt-v2"
        / "sha256"
        / digest_hex[:2]
        / f"{digest_hex}.parquet"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(payload)
    remote_inventory = {"objects": [{"sha256": sha256, "status": "verified"}]}

    with pytest.raises(data_store.DataStoreError, match="NO_LOCAL_PMXT_READERS"):
        data_store.prune_pmxt_local(
            tmp_path,
            remote_inventory=remote_inventory,
            execute=True,
            confirm_no_local_readers="yes",
        )

    assert canonical.exists()

    report = data_store.prune_pmxt_local(
        tmp_path,
        remote_inventory=remote_inventory,
        execute=True,
        confirm_no_local_readers="NO_LOCAL_PMXT_READERS",
    )

    assert report["summary"]["deleted_count"] == 1
    assert not canonical.exists()
