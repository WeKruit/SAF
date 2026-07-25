from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from prediction_market.sports import nfl_x13_source_bundles as source_bundles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_REPO_ROOT = (
    PROJECT_ROOT.parents[1]
    if PROJECT_ROOT.parent.name == ".worktrees"
    else PROJECT_ROOT
)
CAPTURE_RECEIPT = (
    PRIMARY_REPO_ROOT
    / "var"
    / "raw"
    / "capture-receipts"
    /
    "f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d.json"
)
CAPTURE_SHA256 = (
    "sha256:"
    "d78eaf7d203df942959748b942bdf7215f5cd8ae83e4967774843fad1953d516"
)
CAPTURE_RECEIPT_SHA256 = (
    "sha256:"
    "f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d"
)
EXPECTED = {
    "DS-KALSHI-HISTORICAL": {
        "source_version": "official-rest-20260723",
        "manifest_count": 9345,
        "bundle_sha256": (
            "sha256:"
            "40a886ddb9463720a5f54ad8e3a212216a9a0098ae2d17c8230b0d8f33ad2530"
        ),
    },
    "DS-POLYMARKET-PUBLIC": {
        "source_version": "public-api-20260723",
        "manifest_count": 840,
        "bundle_sha256": (
            "sha256:"
            "7714b0132a89a1f8e812e5a8bc78a20e2fb6dfd5c1689c08cee9ec9c6dc376d8"
        ),
    },
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_receipt(tmp_path: Path, value: object) -> Path:
    receipt_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    path = tmp_path / f"{receipt_sha256}.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _write_synthetic_receipt(tmp_path: Path) -> Path:
    return _write_receipt(
        tmp_path,
        {
        "manifests": [
            {
                "manifest_path": (
                    "manifests/source=polymarket/"
                    "dataset=DS-POLYMARKET-PUBLIC/"
                    "version=public-api-test/partition=a/"
                    f"{digit * 64}.manifest.json"
                ),
                "manifest_sha256": f"sha256:{digit * 64}",
                "object_path": (
                    "raw/source=polymarket/"
                    "dataset=DS-POLYMARKET-PUBLIC/"
                    "version=public-api-test/partition=a/"
                    f"{digit * 64}.json"
                ),
            }
            for digit in ("1", "2")
        ]
        },
    )


def test_x13_source_bundle_module_is_published() -> None:
    assert importlib.util.find_spec(
        "prediction_market.sports.nfl_x13_source_bundles"
    ) is not None


def test_frozen_source_bundle_api_returns_typed_exact_mapping() -> None:
    if not CAPTURE_RECEIPT.is_file():
        pytest.skip("exact immutable X-13 capture receipt is not present")

    bundles = source_bundles.build_x13_source_manifest_bundles(
        CAPTURE_RECEIPT
    )

    assert set(bundles) == set(EXPECTED)
    for dataset_id, bundle in bundles.items():
        assert isinstance(
            bundle, source_bundles.X13SourceManifestBundleV1
        )
        assert bundle.dataset_id == dataset_id
        assert bundle.source_version == EXPECTED[dataset_id]["source_version"]
        assert len(bundle.raw_manifest_sha256s) == (
            EXPECTED[dataset_id]["manifest_count"]
        )
        assert bundle.artifact_sha256 == (
            EXPECTED[dataset_id]["bundle_sha256"]
        )
        assert source_bundles.canonical_bundle_sha256(bundle) == (
            EXPECTED[dataset_id]["bundle_sha256"]
        )


@pytest.mark.parametrize(("dataset_id", "expected"), EXPECTED.items())
@pytest.mark.skipif(
    not CAPTURE_RECEIPT.is_file(),
    reason="exact immutable X-13 capture receipt is not present",
)
def test_build_source_bundle_is_exact_and_capture_bound(
    dataset_id: str, expected: dict[str, object]
) -> None:
    bundle = source_bundles._build_x13_dataset_manifest_bundle(
        CAPTURE_RECEIPT,
        dataset_id=dataset_id,
        source_version=str(expected["source_version"]),
        capture_sha256=CAPTURE_SHA256,
    )

    assert bundle == {
        "schema": "nfl_x13_dataset_manifest_bundle_v1",
        "experiment_id": "X-13",
        "dataset_id": dataset_id,
        "source_version": expected["source_version"],
        "capture_sha256": CAPTURE_SHA256,
        "capture_receipt_sha256": CAPTURE_RECEIPT_SHA256,
        "raw_manifest_sha256s": bundle["raw_manifest_sha256s"],
    }
    manifests = bundle["raw_manifest_sha256s"]
    assert isinstance(manifests, list)
    assert len(manifests) == expected["manifest_count"]
    assert manifests == sorted(set(manifests))
    assert source_bundles.canonical_bundle_sha256(bundle) == (
        expected["bundle_sha256"]
    )


def test_source_bundle_round_trip_is_content_addressed(
    tmp_path: Path,
) -> None:
    receipt = _write_synthetic_receipt(tmp_path)
    bundle = source_bundles._build_x13_dataset_manifest_bundle(
        receipt,
        dataset_id="DS-POLYMARKET-PUBLIC",
        source_version="public-api-test",
        capture_sha256=CAPTURE_SHA256,
    )
    expected_sha256 = source_bundles.canonical_bundle_sha256(bundle)

    path = source_bundles.publish_x13_dataset_manifest_bundle(
        bundle, tmp_path / "published"
    )

    assert path.name == (
        f"sha256-{expected_sha256.removeprefix('sha256:')}.json"
    )
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        == expected_sha256
    )
    assert source_bundles.load_x13_dataset_manifest_bundle(
        path, expected_sha256=expected_sha256
    ) == bundle
    source_bundles._verify_x13_dataset_manifest_bundle(
        bundle, receipt
    )


def test_source_bundle_reopen_fails_closed_after_byte_mutation(
    tmp_path: Path,
) -> None:
    receipt = _write_synthetic_receipt(tmp_path)
    bundle = source_bundles._build_x13_dataset_manifest_bundle(
        receipt,
        dataset_id="DS-POLYMARKET-PUBLIC",
        source_version="public-api-test",
        capture_sha256=CAPTURE_SHA256,
    )
    expected_sha256 = source_bundles.canonical_bundle_sha256(bundle)
    path = source_bundles.publish_x13_dataset_manifest_bundle(
        bundle, tmp_path / "published"
    )
    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["source_version"] = "tampered"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        source_bundles.load_x13_dataset_manifest_bundle(
            path, expected_sha256=expected_sha256
        )


def test_source_bundle_rejects_uppercase_sha256(tmp_path: Path) -> None:
    receipt = _write_synthetic_receipt(tmp_path)

    with pytest.raises(ValueError, match="sha256 digest"):
        source_bundles._build_x13_dataset_manifest_bundle(
            receipt,
            dataset_id="DS-POLYMARKET-PUBLIC",
            source_version="public-api-test",
            capture_sha256="sha256:" + "A" * 64,
        )


def test_publish_race_never_overwrites_conflicting_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_synthetic_receipt(tmp_path)
    bundle = source_bundles._build_x13_dataset_manifest_bundle(
        receipt,
        dataset_id="DS-POLYMARKET-PUBLIC",
        source_version="public-api-test",
        capture_sha256=CAPTURE_SHA256,
    )
    digest = source_bundles.canonical_bundle_sha256(bundle)
    output = tmp_path / "published"
    destination = (
        output / f"sha256-{digest.removeprefix('sha256:')}.json"
    )
    real_link = os.link

    def competing_link(source: object, target: object) -> None:
        destination.write_bytes(b"competing-writer")
        real_link(source, target)

    monkeypatch.setattr(os, "link", competing_link)

    with pytest.raises(ValueError, match="conflicting artifact"):
        source_bundles.publish_x13_dataset_manifest_bundle(bundle, output)
    assert destination.read_bytes() == b"competing-writer"


@pytest.mark.parametrize("race_winner", [False, True])
def test_publish_rejects_symlink_even_when_target_bytes_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_winner: bool,
) -> None:
    receipt = _write_synthetic_receipt(tmp_path)
    bundle = source_bundles._build_x13_dataset_manifest_bundle(
        receipt,
        dataset_id="DS-POLYMARKET-PUBLIC",
        source_version="public-api-test",
        capture_sha256=CAPTURE_SHA256,
    )
    canonical = source_bundles.publish_x13_dataset_manifest_bundle(
        bundle, tmp_path / "canonical"
    )
    target = tmp_path / "external-target.json"
    target.write_bytes(canonical.read_bytes())
    target_before = target.read_bytes()
    digest = source_bundles.canonical_bundle_sha256(bundle)
    output = tmp_path / "published"
    output.mkdir()
    destination = (
        output / f"sha256-{digest.removeprefix('sha256:')}.json"
    )
    if race_winner:
        real_link = os.link

        def competing_link(source: object, link_target: object) -> None:
            destination.symlink_to(target)
            real_link(source, link_target)

        monkeypatch.setattr(os, "link", competing_link)
    else:
        destination.symlink_to(target)

    with pytest.raises(ValueError, match="regular file|symlink"):
        source_bundles.publish_x13_dataset_manifest_bundle(bundle, output)
    assert target.read_bytes() == target_before


@pytest.mark.parametrize("duplicate_field", ["manifest_path", "object_path"])
def test_source_bundle_rejects_duplicate_capture_paths(
    tmp_path: Path, duplicate_field: str
) -> None:
    receipt = json.loads(
        _write_synthetic_receipt(tmp_path).read_text(encoding="utf-8")
    )
    receipt["manifests"][1][duplicate_field] = receipt["manifests"][0][
        duplicate_field
    ]
    duplicate_receipt = _write_receipt(tmp_path, receipt)

    with pytest.raises(ValueError, match=f"duplicate {duplicate_field}"):
        source_bundles._build_x13_dataset_manifest_bundle(
            duplicate_receipt,
            dataset_id="DS-POLYMARKET-PUBLIC",
            source_version="public-api-test",
            capture_sha256=CAPTURE_SHA256,
        )


def test_frozen_builder_and_cli_reject_synthetic_receipt(
    tmp_path: Path,
) -> None:
    manifests = []
    for source, dataset_id, version, digit in (
        (
            "kalshi",
            "DS-KALSHI-HISTORICAL",
            "official-rest-20260723",
            "1",
        ),
        (
            "polymarket",
            "DS-POLYMARKET-PUBLIC",
            "public-api-20260723",
            "2",
        ),
    ):
        manifests.append(
            {
                "manifest_path": (
                    f"manifests/source={source}/dataset={dataset_id}/"
                    f"version={version}/partition=a/"
                    f"{digit * 64}.manifest.json"
                ),
                "manifest_sha256": f"sha256:{digit * 64}",
                "object_path": (
                    f"raw/source={source}/dataset={dataset_id}/"
                    f"version={version}/partition=a/{digit * 64}.json"
                ),
            }
        )
    receipt = _write_receipt(tmp_path, {"manifests": manifests})
    with pytest.raises(ValueError, match="frozen capture receipt"):
        source_bundles.build_x13_source_manifest_bundles(receipt)
    with pytest.raises(ValueError, match="frozen capture receipt"):
        source_bundles.main(
            [
                "--receipt",
                str(receipt),
                "--output",
                str(tmp_path / "published"),
            ]
        )


@pytest.mark.skipif(
    not CAPTURE_RECEIPT.is_file(),
    reason="exact immutable X-13 capture receipt is not present",
)
def test_module_cli_publishes_and_reopens_exact_frozen_bundles(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"

    assert source_bundles.main(
        ["--receipt", str(CAPTURE_RECEIPT), "--output", str(output)]
    ) == 0

    assert {
        path.name for path in output.glob("sha256-*.json")
    } == {
        f"sha256-{str(value['bundle_sha256']).removeprefix('sha256:')}.json"
        for value in EXPECTED.values()
    }
    for expected in EXPECTED.values():
        digest = str(expected["bundle_sha256"])
        source_bundles.load_x13_dataset_manifest_bundle(
            output / f"sha256-{digest.removeprefix('sha256:')}.json",
            expected_sha256=digest,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("capture_sha256", "frozen capture_sha256"),
        ("capture_receipt_sha256", "frozen capture_receipt_sha256"),
    ],
)
def test_public_verifier_rejects_claimed_nonfrozen_capture_digest(
    tmp_path: Path, field: str, message: str
) -> None:
    expected_sha256 = str(
        EXPECTED["DS-POLYMARKET-PUBLIC"]["bundle_sha256"]
    )
    bundle = source_bundles.load_x13_dataset_manifest_bundle(
        PROJECT_ROOT
        / "registries"
        / "dataset_manifest_bundles"
        / f"sha256-{expected_sha256.removeprefix('sha256:')}.json",
        expected_sha256=expected_sha256,
    )
    bundle[field] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match=message):
        source_bundles.verify_x13_dataset_manifest_bundle(
            bundle, tmp_path / "receipt-must-not-be-read.json"
        )


@pytest.mark.parametrize(("dataset_id", "expected"), EXPECTED.items())
def test_published_source_bundle_reopens_and_matches_registry_target(
    dataset_id: str, expected: dict[str, object]
) -> None:
    expected_sha256 = str(expected["bundle_sha256"])
    path = (
        PROJECT_ROOT
        / "registries"
        / "dataset_manifest_bundles"
        / f"sha256-{expected_sha256.removeprefix('sha256:')}.json"
    )

    bundle = source_bundles.load_x13_dataset_manifest_bundle(
        path, expected_sha256=expected_sha256
    )

    assert bundle["dataset_id"] == dataset_id
    assert bundle["source_version"] == expected["source_version"]
    assert len(bundle["raw_manifest_sha256s"]) == expected["manifest_count"]
    if CAPTURE_RECEIPT.is_file():
        source_bundles.verify_x13_dataset_manifest_bundle(
            bundle, CAPTURE_RECEIPT
        )
