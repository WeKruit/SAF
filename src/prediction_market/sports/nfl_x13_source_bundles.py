from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA = "nfl_x13_dataset_manifest_bundle_v1"
_EXPERIMENT_ID = "X-13"
_SHA256_PREFIX = "sha256:"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_DATASET_PATH_MARKERS = {
    "DS-KALSHI-HISTORICAL": (
        "source=kalshi/dataset=DS-KALSHI-HISTORICAL/"
    ),
    "DS-POLYMARKET-PUBLIC": (
        "source=polymarket/dataset=DS-POLYMARKET-PUBLIC/"
    ),
}
_FROZEN_SOURCE_VERSIONS = {
    "DS-KALSHI-HISTORICAL": "official-rest-20260723",
    "DS-POLYMARKET-PUBLIC": "public-api-20260723",
}
_FROZEN_CAPTURE_SHA256 = (
    "sha256:"
    "d78eaf7d203df942959748b942bdf7215f5cd8ae83e4967774843fad1953d516"
)
_FROZEN_CAPTURE_RECEIPT_SHA256 = (
    "sha256:"
    "f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d"
)
_FROZEN_BUNDLE_EXPECTATIONS = {
    "DS-KALSHI-HISTORICAL": (
        9345,
        "sha256:"
        "40a886ddb9463720a5f54ad8e3a212216a9a0098ae2d17c8230b0d8f33ad2530",
    ),
    "DS-POLYMARKET-PUBLIC": (
        840,
        "sha256:"
        "7714b0132a89a1f8e812e5a8bc78a20e2fb6dfd5c1689c08cee9ec9c6dc376d8",
    ),
}
_REQUIRED_KEYS = {
    "schema",
    "experiment_id",
    "dataset_id",
    "source_version",
    "capture_sha256",
    "capture_receipt_sha256",
    "raw_manifest_sha256s",
}


@dataclass(frozen=True, slots=True)
class X13SourceManifestBundleV1:
    schema: str
    experiment_id: str
    dataset_id: str
    source_version: str
    capture_sha256: str
    capture_receipt_sha256: str
    raw_manifest_sha256s: tuple[str, ...]

    @property
    def artifact_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_payload()))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "source_version": self.source_version,
            "capture_sha256": self.capture_sha256,
            "capture_receipt_sha256": self.capture_receipt_sha256,
            "raw_manifest_sha256s": list(self.raw_manifest_sha256s),
        }


def _sha256(raw: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a lowercase canonical sha256 digest"
        )
    return value


def _load_capture_receipt(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("capture receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("capture receipt must be a JSON object")
    manifests = receipt.get("manifests")
    if not isinstance(manifests, list):
        raise ValueError("capture receipt manifests must be a list")
    observed_paths: dict[str, set[str]] = {
        "manifest_path": set(),
        "object_path": set(),
    }
    for item in manifests:
        if not isinstance(item, dict):
            raise ValueError("capture receipt manifest entry must be an object")
        for field, seen in observed_paths.items():
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"capture receipt manifest has invalid {field}"
                )
            if value in seen:
                raise ValueError(f"capture receipt has duplicate {field}")
            seen.add(value)
    receipt_sha256 = _sha256(_canonical_bytes(receipt))
    expected_name = f"{receipt_sha256.removeprefix(_SHA256_PREFIX)}.json"
    if path.name != expected_name:
        raise ValueError("capture receipt path does not match its identity")
    return receipt, receipt_sha256


def _validate_bundle_shape(bundle: object) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != _REQUIRED_KEYS:
        raise ValueError("source bundle has invalid fields")
    if bundle["schema"] != _SCHEMA:
        raise ValueError("source bundle has invalid schema")
    if bundle["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError("source bundle has invalid experiment_id")
    dataset_id = bundle["dataset_id"]
    if dataset_id not in _DATASET_PATH_MARKERS:
        raise ValueError("source bundle has unsupported dataset_id")
    if not isinstance(bundle["source_version"], str) or not bundle[
        "source_version"
    ]:
        raise ValueError("source bundle has invalid source_version")
    _require_sha256(bundle["capture_sha256"], field="capture_sha256")
    _require_sha256(
        bundle["capture_receipt_sha256"],
        field="capture_receipt_sha256",
    )
    manifests = bundle["raw_manifest_sha256s"]
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("source bundle must contain raw manifests")
    for manifest_sha256 in manifests:
        _require_sha256(
            manifest_sha256, field="raw_manifest_sha256s entry"
        )
    if manifests != sorted(set(manifests)):
        raise ValueError("raw manifests must be sorted and unique")
    return bundle


def _build_x13_dataset_manifest_bundle(
    capture_receipt_path: Path,
    *,
    dataset_id: str,
    source_version: str,
    capture_sha256: str,
) -> dict[str, Any]:
    marker = _DATASET_PATH_MARKERS.get(dataset_id)
    if marker is None:
        raise ValueError(f"unsupported X-13 dataset: {dataset_id}")
    _require_sha256(capture_sha256, field="capture_sha256")
    receipt, receipt_sha256 = _load_capture_receipt(capture_receipt_path)
    manifests: list[str] = []
    observed_versions: set[str] = set()
    for item in receipt["manifests"]:
        if not isinstance(item, dict):
            raise ValueError("capture receipt manifest entry must be an object")
        manifest_path = item.get("manifest_path")
        if not isinstance(manifest_path, str) or marker not in manifest_path:
            continue
        manifest_sha256 = _require_sha256(
            item.get("manifest_sha256"), field="manifest_sha256"
        )
        manifests.append(manifest_sha256)
        version_marker = "/version="
        version_tail = manifest_path.split(version_marker, maxsplit=1)
        if len(version_tail) != 2 or "/" not in version_tail[1]:
            raise ValueError("capture manifest path has no source version")
        observed_versions.add(version_tail[1].split("/", maxsplit=1)[0])
    if observed_versions != {source_version}:
        raise ValueError(
            "source_version does not match capture receipt: "
            f"{sorted(observed_versions)!r}"
        )
    bundle = {
        "schema": _SCHEMA,
        "experiment_id": _EXPERIMENT_ID,
        "dataset_id": dataset_id,
        "source_version": source_version,
        "capture_sha256": capture_sha256,
        "capture_receipt_sha256": receipt_sha256,
        "raw_manifest_sha256s": sorted(set(manifests)),
    }
    return _validate_bundle_shape(bundle)


def canonical_bundle_sha256(bundle: object) -> str:
    validated = _bundle_payload(bundle)
    return _sha256(_canonical_bytes(validated))


def _bundle_payload(bundle: object) -> dict[str, Any]:
    if isinstance(bundle, X13SourceManifestBundleV1):
        return _validate_bundle_shape(bundle.to_payload())
    return _validate_bundle_shape(bundle)


def build_x13_source_manifest_bundles(
    capture_receipt_path: Path,
) -> dict[str, X13SourceManifestBundleV1]:
    _, receipt_sha256 = _load_capture_receipt(capture_receipt_path)
    if receipt_sha256 != _FROZEN_CAPTURE_RECEIPT_SHA256:
        raise ValueError(
            "receipt is not the frozen capture receipt "
            f"{_FROZEN_CAPTURE_RECEIPT_SHA256}"
        )
    bundles: dict[str, X13SourceManifestBundleV1] = {}
    for dataset_id, source_version in sorted(
        _FROZEN_SOURCE_VERSIONS.items()
    ):
        payload = _build_x13_dataset_manifest_bundle(
            capture_receipt_path,
            dataset_id=dataset_id,
            source_version=source_version,
            capture_sha256=_FROZEN_CAPTURE_SHA256,
        )
        expected_count, expected_sha256 = _FROZEN_BUNDLE_EXPECTATIONS[
            dataset_id
        ]
        if len(payload["raw_manifest_sha256s"]) != expected_count:
            raise ValueError(
                f"{dataset_id}: frozen raw manifest count mismatch"
            )
        if canonical_bundle_sha256(payload) != expected_sha256:
            raise ValueError(f"{dataset_id}: frozen bundle hash mismatch")
        bundles[dataset_id] = X13SourceManifestBundleV1(
            schema=payload["schema"],
            experiment_id=payload["experiment_id"],
            dataset_id=payload["dataset_id"],
            source_version=payload["source_version"],
            capture_sha256=payload["capture_sha256"],
            capture_receipt_sha256=payload["capture_receipt_sha256"],
            raw_manifest_sha256s=tuple(
                payload["raw_manifest_sha256s"]
            ),
        )
    return bundles


def _read_regular_file_nofollow(path: Path) -> bytes | None:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(
            "content-addressed artifact path must not be a symlink"
        )
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(
            "content-addressed artifact path must be a regular file"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("O_NOFOLLOW is required for artifact verification")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ValueError(
            "content-addressed artifact could not be opened without following"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise ValueError(
                "content-addressed artifact changed before verification"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def publish_x13_dataset_manifest_bundle(
    bundle: object, output_root: Path
) -> Path:
    validated = _bundle_payload(bundle)
    raw = _canonical_bytes(validated)
    digest = _sha256(raw)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = (
        output_root
        / f"sha256-{digest.removeprefix(_SHA256_PREFIX)}.json"
    )
    existing = _read_regular_file_nofollow(destination)
    if existing is not None:
        if existing != raw:
            raise ValueError(
                "content-addressed path contains a conflicting artifact"
            )
        return destination
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_root, prefix=".bundle-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            winner = _read_regular_file_nofollow(destination)
            if winner is None:
                raise ValueError(
                    "content-addressed publish race has no stable winner"
                )
            if winner != raw:
                raise ValueError(
                    "content-addressed publish race produced a "
                    "conflicting artifact"
                )
        else:
            directory_fd = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_x13_dataset_manifest_bundle(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    _require_sha256(expected_sha256, field="expected_sha256")
    raw = path.read_bytes()
    actual_sha256 = _sha256(raw)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "source bundle hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("source bundle is not valid JSON") from exc
    validated = _validate_bundle_shape(bundle)
    if raw != _canonical_bytes(validated):
        raise ValueError("source bundle bytes are not canonical")
    return validated


def _verify_x13_dataset_manifest_bundle(
    bundle: object, capture_receipt_path: Path
) -> None:
    validated = _bundle_payload(bundle)
    rebuilt = _build_x13_dataset_manifest_bundle(
        capture_receipt_path,
        dataset_id=validated["dataset_id"],
        source_version=validated["source_version"],
        capture_sha256=validated["capture_sha256"],
    )
    if rebuilt != validated:
        raise ValueError("source bundle does not match capture receipt")


def verify_x13_dataset_manifest_bundle(
    bundle: object, capture_receipt_path: Path
) -> None:
    validated = _bundle_payload(bundle)
    if validated["capture_sha256"] != _FROZEN_CAPTURE_SHA256:
        raise ValueError("bundle does not use the frozen capture_sha256")
    if (
        validated["capture_receipt_sha256"]
        != _FROZEN_CAPTURE_RECEIPT_SHA256
    ):
        raise ValueError(
            "bundle does not use the frozen capture_receipt_sha256"
        )
    frozen = build_x13_source_manifest_bundles(capture_receipt_path)
    expected = frozen.get(validated["dataset_id"])
    if expected is None or expected.to_payload() != validated:
        raise ValueError("bundle does not match frozen source evidence")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish exact X-13 per-source manifest bundles."
    )
    parser.add_argument(
        "--receipt",
        required=True,
        type=Path,
        help="Path to the immutable X-13 capture receipt.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for content-addressed bundle JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    bundles = build_x13_source_manifest_bundles(arguments.receipt)
    for dataset_id, bundle in bundles.items():
        path = publish_x13_dataset_manifest_bundle(
            bundle, arguments.output
        )
        expected_sha256 = _FROZEN_BUNDLE_EXPECTATIONS[dataset_id][1]
        reopened = load_x13_dataset_manifest_bundle(
            path, expected_sha256=expected_sha256
        )
        if reopened != bundle.to_payload():
            raise ValueError(f"{dataset_id}: published bundle reopen mismatch")
        verify_x13_dataset_manifest_bundle(reopened, arguments.receipt)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
