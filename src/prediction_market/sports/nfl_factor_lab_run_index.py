"""Verified progress/run index for the single authoritative Factor Lab notebook."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from prediction_market.sports.nfl_factor_lab_bundles import (
    PublishedCrossGameBundleV2,
    PublishedSingleGameBundleV2,
    verify_cross_game_bundle,
    verify_single_game_bundle,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_IDS,
    canonical_sha256,
)


class FactorLabRunIndexError(ValueError):
    """The Factor Lab V2 run index does not match its artifacts."""


_MATERIALIZER_CODE_LOCK_FILES = (
    "src/prediction_market/contracts.py",
    "src/prediction_market/immutable_object_store.py",
    "src/prediction_market/program_audit.py",
    "src/prediction_market/sports/nfl_factor_lab.py",
    "src/prediction_market/sports/nfl_factor_lab_analysis.py",
    "src/prediction_market/sports/nfl_factor_lab_artifacts.py",
    "src/prediction_market/sports/nfl_factor_lab_bundles.py",
    "src/prediction_market/sports/nfl_factor_lab_cleaning.py",
    "src/prediction_market/sports/nfl_factor_lab_factor_observations.py",
    "src/prediction_market/sports/nfl_factor_lab_features.py",
    "src/prediction_market/sports/nfl_factor_lab_materialize.py",
    "src/prediction_market/sports/nfl_factor_lab_run_index.py",
    "src/prediction_market/sports/nfl_factor_lab_types.py",
    "src/prediction_market/sports/nfl_x13_association.py",
    "src/prediction_market/sports/nfl_x13_batch.py",
    "src/prediction_market/sports/nfl_x13_capture.py",
    "src/prediction_market/sports/nfl_x13_market.py",
    "src/prediction_market/sports/nfl_x13_market_cleaning.py",
    "src/prediction_market/sports/nfl_x13_pipeline.py",
    "src/prediction_market/sports/nfl_x13_replay.py",
    "src/prediction_market/sports/nfl_x13_spec.py",
    "src/prediction_market/sports/nfl_x13_universe.py",
    "src/prediction_market/static_store.py",
)


@dataclass(frozen=True, slots=True)
class StageManifestEntryV2:
    scope: str
    game_id: str | None
    stage: str
    grain: tuple[str, ...]
    row_count: int
    semantic_rows_sha256: str
    artifact_path: str
    artifact_sha256: str
    status: str = "PUBLISHED_VERIFIED"

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "game_id": self.game_id,
            "stage": self.stage,
            "grain": list(self.grain),
            "row_count": self.row_count,
            "semantic_rows_sha256": self.semantic_rows_sha256,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class VerifiedFactorLabRunIndexV2:
    path: Path
    game_ids: tuple[str, ...]
    single_game_manifests: tuple[Path, ...]
    cross_game_manifest: Path
    source_artifacts: tuple[tuple[str, Path, str, int], ...]
    stage_manifest: tuple[StageManifestEntryV2, ...]
    s3_bucket: str
    s3_prefix: str
    holdout_reaction_accessed: bool
    code_lock_sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            length += len(chunk)
    return "sha256:" + digest.hexdigest(), length


def factor_lab_materializer_code_lock_sha256(
    project_root: str | Path,
) -> str:
    """Compute the only authoritative code lock accepted by formal V2 runs."""

    root = Path(project_root).resolve()
    files: list[dict[str, str]] = []
    for relative in _MATERIALIZER_CODE_LOCK_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FactorLabRunIndexError(
                f"authoritative code-lock file is missing: {relative}"
            )
        digest, _length = _sha256_path(path)
        files.append({"path": relative, "sha256": digest})
    return canonical_sha256(
        {
            "schema": "NFLFactorLabMaterializerCodeLockV2",
            "files": files,
        }
    )


def factor_lab_materializer_code_lock_paths() -> tuple[str, ...]:
    """Return the complete ordered source inventory bound by the V2 lock."""

    return _MATERIALIZER_CODE_LOCK_FILES


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise FactorLabRunIndexError(
            "run-index artifact is outside project_root"
        ) from error


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    digest, length = _sha256_path(path)
    return {
        "path": _relative(path, root),
        "sha256": digest,
        "byte_length": length,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabRunIndexError("bundle manifest is unreadable") from error
    if not isinstance(value, dict):
        raise FactorLabRunIndexError("bundle manifest root is invalid")
    return value


def _stage_entries(
    single_bundles: Sequence[PublishedSingleGameBundleV2],
    cross_bundle: PublishedCrossGameBundleV2,
    *,
    project_root: Path,
) -> tuple[StageManifestEntryV2, ...]:
    entries: list[StageManifestEntryV2] = []
    for bundle in single_bundles:
        manifest = _read_manifest(bundle.manifest_path)
        descriptors = manifest.get("stages")
        if not isinstance(descriptors, dict):
            raise FactorLabRunIndexError("single-game stages are missing")
        for name, path in sorted(bundle.stage_paths.items()):
            descriptor = descriptors.get(name)
            if not isinstance(descriptor, dict):
                raise FactorLabRunIndexError("single-game stage descriptor differs")
            digest, _length = _sha256_path(path)
            entries.append(
                StageManifestEntryV2(
                    scope="single_game",
                    game_id=bundle.game_id,
                    stage=name,
                    grain=tuple(descriptor["grain"]),
                    row_count=int(descriptor["row_count"]),
                    semantic_rows_sha256=str(
                        descriptor["semantic_rows_sha256"]
                    ),
                    artifact_path=_relative(path, project_root),
                    artifact_sha256=digest,
                )
            )
    manifest = _read_manifest(cross_bundle.manifest_path)
    descriptors = manifest.get("stages")
    if not isinstance(descriptors, dict):
        raise FactorLabRunIndexError("cross-game stages are missing")
    for name, path in sorted(cross_bundle.stage_paths.items()):
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, dict):
            raise FactorLabRunIndexError("cross-game stage descriptor differs")
        digest, _length = _sha256_path(path)
        entries.append(
            StageManifestEntryV2(
                scope="cross_game",
                game_id=None,
                stage=name,
                grain=tuple(descriptor["grain"]),
                row_count=int(descriptor["row_count"]),
                semantic_rows_sha256=str(
                    descriptor["semantic_rows_sha256"]
                ),
                artifact_path=_relative(path, project_root),
                artifact_sha256=digest,
            )
        )
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.scope,
                entry.game_id or "",
                entry.stage,
            ),
        )
    )


def build_factor_lab_run_index(
    *,
    single_game_manifests: Sequence[str | Path],
    cross_game_manifest: str | Path,
    source_artifacts: Mapping[str, str | Path],
    project_root: str | Path,
    s3_bucket: str,
    s3_prefix: str,
    output_path: str | Path,
) -> VerifiedFactorLabRunIndexV2:
    """Atomically publish the deterministic V2 notebook run index."""

    root = Path(project_root)
    authoritative_code_lock = factor_lab_materializer_code_lock_sha256(root)
    singles = tuple(
        verify_single_game_bundle(path) for path in single_game_manifests
    )
    if not singles:
        raise FactorLabRunIndexError("single-game bundles are required")
    cross = verify_cross_game_bundle(cross_game_manifest)
    game_ids = tuple(sorted(bundle.game_id for bundle in singles))
    if game_ids != X13_GAME_IDS or game_ids != cross.game_ids:
        raise FactorLabRunIndexError(
            "run index must exactly reference the frozen X-13 20-game set"
        )
    if (
        cross.code_lock_sha256 != authoritative_code_lock
        or any(
            bundle.code_lock_sha256 != authoritative_code_lock
            for bundle in singles
        )
    ):
        raise FactorLabRunIndexError(
            "bundles differ from the authoritative current code lock"
        )
    if not source_artifacts:
        raise FactorLabRunIndexError("source artifacts are required")
    source_descriptors: dict[str, dict[str, object]] = {}
    for name, source in sorted(source_artifacts.items()):
        if not isinstance(name, str) or not name:
            raise FactorLabRunIndexError("source artifact name is invalid")
        path = Path(source)
        if not path.is_file():
            raise FactorLabRunIndexError(f"source artifact is missing: {name}")
        source_descriptors[name] = _descriptor(path, root)
    stages = _stage_entries(singles, cross, project_root=root)
    payload = {
        "schema": "NFLFactorLabRunIndexV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "holdout_reaction_accessed": False,
        "shortlist_lock_path": None,
        "code_lock_sha256": cross.code_lock_sha256,
        "s3": {"bucket": s3_bucket, "prefix": s3_prefix},
        "game_ids": list(game_ids),
        "single_game_bundles": [
            {
                "game_id": bundle.game_id,
                "bundle_sha256": bundle.bundle_sha256,
                "manifest": _descriptor(bundle.manifest_path, root),
            }
            for bundle in sorted(singles, key=lambda item: item.game_id)
        ],
        "cross_game_bundle": {
            "bundle_sha256": cross.bundle_sha256,
            "manifest": _descriptor(cross.manifest_path, root),
        },
        "source_artifacts": source_descriptors,
        "stage_manifest": [entry.to_dict() for entry in stages],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return verify_factor_lab_run_index(path, project_root=root)


def _verified_descriptor(
    descriptor: object,
    *,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(descriptor, dict):
        raise FactorLabRunIndexError(f"{label} descriptor is invalid")
    relative = descriptor.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise FactorLabRunIndexError(f"{label} path is invalid")
    path = root / relative
    if not path.is_file():
        raise FactorLabRunIndexError(f"{label} is missing")
    digest, length = _sha256_path(path)
    if digest != descriptor.get("sha256") or length != descriptor.get(
        "byte_length"
    ):
        raise FactorLabRunIndexError(f"{label} hash or length differs")
    return path


def verify_factor_lab_run_index(
    path: str | Path,
    *,
    project_root: str | Path,
) -> VerifiedFactorLabRunIndexV2:
    index_path = Path(path)
    try:
        raw = index_path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabRunIndexError("run index is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise FactorLabRunIndexError("run index is not canonical")
    if (
        value.get("schema") != "NFLFactorLabRunIndexV2"
        or value.get("experiment_id") != "X-13"
        or value.get("claim_boundary") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or value.get("holdout_reaction_accessed") is not False
        or value.get("shortlist_lock_path") is not None
    ):
        raise FactorLabRunIndexError("run index governance differs")
    root = Path(project_root)
    authoritative_code_lock = factor_lab_materializer_code_lock_sha256(root)
    single_values = value.get("single_game_bundles")
    if not isinstance(single_values, list) or not single_values:
        raise FactorLabRunIndexError("run index has no single-game bundles")
    singles: list[PublishedSingleGameBundleV2] = []
    for record in single_values:
        if not isinstance(record, dict):
            raise FactorLabRunIndexError("single-game record is invalid")
        manifest_path = _verified_descriptor(
            record.get("manifest"), root=root, label="single-game manifest"
        )
        bundle = verify_single_game_bundle(manifest_path)
        if (
            bundle.game_id != record.get("game_id")
            or bundle.bundle_sha256 != record.get("bundle_sha256")
        ):
            raise FactorLabRunIndexError("single-game identity differs")
        singles.append(bundle)
    cross_value = value.get("cross_game_bundle")
    if not isinstance(cross_value, dict):
        raise FactorLabRunIndexError("cross-game record is invalid")
    cross_path = _verified_descriptor(
        cross_value.get("manifest"), root=root, label="cross-game manifest"
    )
    cross = verify_cross_game_bundle(cross_path)
    if cross.bundle_sha256 != cross_value.get("bundle_sha256"):
        raise FactorLabRunIndexError("cross-game identity differs")
    game_ids = tuple(sorted(bundle.game_id for bundle in singles))
    if (
        value.get("game_ids") != list(game_ids)
        or game_ids != X13_GAME_IDS
        or game_ids != cross.game_ids
    ):
        raise FactorLabRunIndexError("run-index game coverage differs")
    if (
        value.get("code_lock_sha256") != authoritative_code_lock
        or cross.code_lock_sha256 != authoritative_code_lock
        or any(
            bundle.code_lock_sha256 != authoritative_code_lock
            for bundle in singles
        )
    ):
        raise FactorLabRunIndexError(
            "run index differs from the authoritative current code lock"
        )
    source_values = value.get("source_artifacts")
    if not isinstance(source_values, dict) or not source_values:
        raise FactorLabRunIndexError("source artifacts are missing")
    sources: list[tuple[str, Path, str, int]] = []
    for name, descriptor in sorted(source_values.items()):
        source_path = _verified_descriptor(
            descriptor, root=root, label=f"source artifact {name}"
        )
        assert isinstance(descriptor, dict)
        sources.append(
            (
                name,
                source_path,
                str(descriptor["sha256"]),
                int(descriptor["byte_length"]),
            )
        )
    expected_stages = _stage_entries(singles, cross, project_root=root)
    stage_values = value.get("stage_manifest")
    if (
        not isinstance(stage_values, list)
        or stage_values != [entry.to_dict() for entry in expected_stages]
    ):
        raise FactorLabRunIndexError("stage manifest differs")
    s3 = value.get("s3")
    if (
        not isinstance(s3, dict)
        or not isinstance(s3.get("bucket"), str)
        or not isinstance(s3.get("prefix"), str)
    ):
        raise FactorLabRunIndexError("S3 public configuration differs")
    return VerifiedFactorLabRunIndexV2(
        path=index_path,
        game_ids=game_ids,
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in sorted(
                singles, key=lambda item: item.game_id
            )
        ),
        cross_game_manifest=cross.manifest_path,
        source_artifacts=tuple(sources),
        stage_manifest=expected_stages,
        s3_bucket=s3["bucket"],
        s3_prefix=s3["prefix"],
        holdout_reaction_accessed=False,
        code_lock_sha256=cross.code_lock_sha256,
    )


__all__ = [
    "FactorLabRunIndexError",
    "StageManifestEntryV2",
    "VerifiedFactorLabRunIndexV2",
    "build_factor_lab_run_index",
    "factor_lab_materializer_code_lock_paths",
    "factor_lab_materializer_code_lock_sha256",
    "verify_factor_lab_run_index",
]
