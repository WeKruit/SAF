"""Narrow, offline loaders for the NFL Factor Lab master notebook.

The notebook remains the visible workflow.  This module only provides
fail-closed file verification, bounded table reads, and uniform presentation
metadata.  It deliberately has no ``run_all``-style orchestration and no
network client.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from prediction_market.sports.nfl_factor_lab_run_index import (
    verify_factor_lab_run_index,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_DELAY_SCENARIOS_SECONDS,
    X13_GAME_IDS,
)


RUN_INDEX_ENV = "SAF_FACTOR_LAB_RUN_INDEX"
BATCH_MANIFEST_ENV = "SAF_FACTOR_LAB_BATCH_MANIFEST"
STAGE_MANIFEST_ENV = "SAF_FACTOR_LAB_STAGE_MANIFEST"
SINGLE_GAME_BUNDLE_ENV = "SAF_FACTOR_LAB_SINGLE_GAME_BUNDLE"
CROSS_GAME_BUNDLE_ENV = "SAF_FACTOR_LAB_CROSS_GAME_BUNDLE"
DEEP_VERIFY_ENV = "SAF_FACTOR_LAB_DEEP_VERIFY"
DEFAULT_RUN_INDEX = Path("registries/factors/nfl_factor_lab_run_index_v1.json")
PREFERRED_RUN_INDEX = Path("registries/factors/nfl_factor_lab_run_index_v2.json")
VALID_MODES = frozenset({"single_game_audit", "twenty_game_published"})
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_BUNDLE_BUILDER_VERSION = "nfl-factor-lab-bundle-v2"

NOTEBOOK_SINGLE_STAGE_GRAINS: dict[str, tuple[str, ...]] = {
    "sports_row_disposition": ("game_id", "raw_play_id"),
    "canonical_events": ("game_id", "play_id"),
    "finalized_episodes": ("game_id", "episode_id"),
    "stat_ledger": ("game_id", "through_play_id"),
    "pit_features": ("game_id", "play_id"),
    "contract_inventory": ("game_id", "logical_market_id", "outcome"),
    "market_observations": ("game_id", "observation_id"),
    "market_cleaning_audit": ("game_id", "audit_row_id"),
    "game_time_intervals": ("game_id", "interval_id"),
    "market_time_intervals": ("game_id", "interval_id"),
    "source_timeline": ("game_id", "timeline_id"),
    "timeline_audit": ("game_id", "timeline_audit_id"),
    "association_attrition": ("game_id", "association_id"),
    "reaction_paths": (
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "horizon_seconds",
    ),
    "factor_observations": (
        "factor_id",
        "game_id",
        "episode_id",
        "logical_market_id",
        "outcome",
        "venue",
        "horizon_seconds",
    ),
    "factor_exclusion_audit": ("game_id", "audit_id"),
    "single_game_factor_results": (
        "factor_id",
        "game_id",
        "market_family",
        "horizon_seconds",
    ),
}
NOTEBOOK_CROSS_STAGE_GRAINS: dict[str, tuple[str, ...]] = {
    "cross_game_factor_results": (
        "factor_id",
        "market_family",
        "horizon_seconds",
    ),
    "game_contribution": (
        "factor_id",
        "market_family",
        "horizon_seconds",
        "game_id",
    ),
    "leave_one_game_out": (
        "factor_id",
        "market_family",
        "horizon_seconds",
        "game_id",
    ),
    "venue_consistency": (
        "factor_id",
        "market_family",
        "horizon_seconds",
        "venue",
    ),
    "expert_review_queue": (
        "factor_id",
        "market_family",
        "horizon_seconds",
    ),
}


class ArtifactVerificationError(ValueError):
    """A notebook artifact reference failed closed."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    name: str
    path: Path
    sha256: str
    byte_length: int
    row_count: int | None = None
    schema: str | None = None


@dataclass(frozen=True, slots=True)
class NotebookStageManifestEntryV2:
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
class ShallowSingleGameBundleV2:
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    game_id: str
    stage_paths: dict[str, Path]
    source_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShallowCrossGameBundleV2:
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    game_ids: tuple[str, ...]
    stage_paths: dict[str, Path]
    holdout_reaction_accessed: bool


@dataclass(frozen=True, slots=True)
class ShallowFactorLabRunIndexV2:
    path: Path
    run_index_sha256: str
    game_ids: tuple[str, ...]
    single_game_manifests: tuple[Path, ...]
    cross_game_manifest: Path
    source_artifacts: tuple[tuple[str, Path, str, int], ...]
    stage_manifest: tuple[NotebookStageManifestEntryV2, ...]
    s3_bucket: str
    s3_prefix: str
    holdout_reaction_accessed: bool


@dataclass(frozen=True, slots=True)
class FactorLabNotebookRunV2:
    run_index: ShallowFactorLabRunIndexV2
    single_game_bundles: tuple[ShallowSingleGameBundleV2, ...]
    cross_game_bundle: ShallowCrossGameBundleV2
    verification_mode: str
    stage_content_verification: str
    latest_deep_batch_status: str
    s3_status: str
    verification_elapsed_seconds: float

    def single_game_bundle(self, game_id: str) -> ShallowSingleGameBundleV2:
        matches = [
            bundle
            for bundle in self.single_game_bundles
            if bundle.game_id == game_id
        ]
        if len(matches) != 1:
            raise ArtifactVerificationError(
                f"run index contains no unique single-game bundle: {game_id}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class V2BatchAttritionSummary:
    """Compact, batch-wide association funnel and exclusion reasons."""

    funnel: pd.DataFrame
    reasons: pd.DataFrame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactVerificationError(f"{label} SHA-256 is invalid")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactVerificationError(f"{label} is invalid")
    return value


def _resolve_repo_path(
    raw_path: object,
    *,
    project_root: Path,
    label: str,
    allow_parent_parts: bool = False,
    base: Path | None = None,
) -> Path:
    if type(raw_path) is not str or not raw_path:
        raise ArtifactVerificationError(f"{label} path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or (
        not allow_parent_parts and ".." in relative.parts
    ):
        raise ArtifactVerificationError(f"{label} path is invalid")
    unresolved = (base or project_root) / relative
    if unresolved.is_symlink():
        raise ArtifactVerificationError(f"{label} path cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ArtifactVerificationError(
            f"{label} must remain inside project root"
        ) from error
    if not resolved.is_file():
        raise ArtifactVerificationError(f"{label} is missing")
    return resolved


def _verify_committed_manifest_descriptor(
    descriptor: object,
    *,
    project_root: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(descriptor, Mapping):
        raise ArtifactVerificationError(f"{label} descriptor is invalid")
    path = _resolve_repo_path(
        descriptor.get("path"),
        project_root=project_root,
        label=label,
    )
    expected_sha = _require_sha256(
        descriptor.get("sha256"),
        label=label,
    )
    expected_length = _require_nonnegative_int(
        descriptor.get("byte_length"),
        label=f"{label} byte length",
    )
    if path.stat().st_size != expected_length or _file_sha256(path) != expected_sha:
        raise ArtifactVerificationError(f"{label} hash or length differs")
    return path, expected_sha


def _verify_stage_descriptors_shallow(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
    project_root: Path,
    expected_grains: Mapping[str, tuple[str, ...]],
    scope: str,
    game_id: str | None,
) -> tuple[
    dict[str, Path],
    tuple[NotebookStageManifestEntryV2, ...],
]:
    descriptors = manifest.get("stages")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(
        expected_grains
    ):
        raise ArtifactVerificationError(
            f"{scope} bundle stage set differs"
        )
    stage_paths: dict[str, Path] = {}
    rows: list[NotebookStageManifestEntryV2] = []
    for stage, expected_grain in sorted(expected_grains.items()):
        descriptor = descriptors.get(stage)
        if not isinstance(descriptor, Mapping):
            raise ArtifactVerificationError(
                f"{scope} stage descriptor is invalid: {stage}"
            )
        raw_path = descriptor.get("path")
        path = _resolve_repo_path(
            raw_path,
            project_root=project_root,
            base=manifest_path.parent,
            label=f"{scope} stage",
        )
        expected_length = _require_nonnegative_int(
            descriptor.get("byte_length"),
            label=f"{scope} stage byte length",
        )
        if path.stat().st_size != expected_length:
            raise ArtifactVerificationError(
                f"{scope} stage byte length differs: {stage}"
            )
        digest = _require_sha256(
            descriptor.get("sha256"),
            label=f"{scope} stage",
        )
        semantic_digest = _require_sha256(
            descriptor.get("semantic_rows_sha256"),
            label=f"{scope} stage semantic rows",
        )
        _require_sha256(
            descriptor.get("schema_fingerprint"),
            label=f"{scope} stage schema",
        )
        grain_value = descriptor.get("grain")
        if not isinstance(grain_value, list) or tuple(grain_value) != tuple(
            expected_grain
        ):
            raise ArtifactVerificationError(
                f"{scope} stage grain differs: {stage}"
            )
        row_count = _require_nonnegative_int(
            descriptor.get("row_count"),
            label=f"{scope} stage row count",
        )
        stage_paths[stage] = path
        rows.append(
            NotebookStageManifestEntryV2(
                scope=scope,
                game_id=game_id,
                stage=stage,
                grain=tuple(expected_grain),
                row_count=row_count,
                semantic_rows_sha256=semantic_digest,
                artifact_path=path.relative_to(project_root).as_posix(),
                artifact_sha256=digest,
            )
        )
    return stage_paths, tuple(rows)


def _verify_single_manifest_shallow(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    expected_game_id: str,
    expected_bundle_sha256: str,
    project_root: Path,
) -> tuple[ShallowSingleGameBundleV2, tuple[NotebookStageManifestEntryV2, ...]]:
    manifest = read_json_object(
        manifest_path,
        expected_schema="SingleGameAnalysisBundleV2",
    )
    if (
        manifest.get("experiment_id") != "X-13"
        or manifest.get("claim_boundary") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or manifest.get("builder_version") != _BUNDLE_BUILDER_VERSION
        or manifest.get("game_id") != expected_game_id
        or manifest.get("bundle_sha256") != expected_bundle_sha256
        or manifest_path.parent.name
        != expected_bundle_sha256.replace(":", "-")
    ):
        raise ArtifactVerificationError("single-game bundle governance differs")
    source_hashes = manifest.get("source_hashes")
    if (
        not isinstance(source_hashes, list)
        or not source_hashes
        or source_hashes != sorted(set(source_hashes))
    ):
        raise ArtifactVerificationError(
            "single-game bundle source hashes are invalid"
        )
    for value in source_hashes:
        _require_sha256(value, label="single-game source")
    stage_paths, stage_rows = _verify_stage_descriptors_shallow(
        manifest,
        manifest_path=manifest_path,
        project_root=project_root,
        expected_grains=NOTEBOOK_SINGLE_STAGE_GRAINS,
        scope="single_game",
        game_id=expected_game_id,
    )
    return (
        ShallowSingleGameBundleV2(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            bundle_sha256=expected_bundle_sha256,
            game_id=expected_game_id,
            stage_paths=stage_paths,
            source_hashes=tuple(source_hashes),
        ),
        stage_rows,
    )


def _verify_cross_manifest_shallow(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    expected_bundle_sha256: str,
    singles: Sequence[ShallowSingleGameBundleV2],
    project_root: Path,
) -> tuple[ShallowCrossGameBundleV2, tuple[NotebookStageManifestEntryV2, ...]]:
    manifest = read_json_object(
        manifest_path,
        expected_schema="CrossGameAnalysisBundleV2",
    )
    if (
        manifest.get("experiment_id") != "X-13"
        or manifest.get("claim_boundary") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or manifest.get("builder_version") != _BUNDLE_BUILDER_VERSION
        or manifest.get("bundle_sha256") != expected_bundle_sha256
        or manifest.get("holdout_reaction_accessed") is not False
        or manifest_path.parent.name
        != expected_bundle_sha256.replace(":", "-")
    ):
        raise ArtifactVerificationError("cross-game bundle governance differs")
    if manifest.get("game_ids") != list(X13_GAME_IDS):
        raise ArtifactVerificationError("cross-game game coverage differs")
    expected_singles = {bundle.game_id: bundle for bundle in singles}
    references = manifest.get("single_game_bundles")
    if not isinstance(references, list) or len(references) != len(
        expected_singles
    ):
        raise ArtifactVerificationError(
            "cross-game single-game references differ"
        )
    observed_games: list[str] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ArtifactVerificationError(
                "cross-game single-game reference is invalid"
            )
        game_id = reference.get("game_id")
        if type(game_id) is not str or game_id not in expected_singles:
            raise ArtifactVerificationError(
                "cross-game single-game identity differs"
            )
        expected = expected_singles[game_id]
        resolved = _resolve_repo_path(
            reference.get("manifest_relative_path"),
            project_root=project_root,
            base=manifest_path.parent.parent,
            allow_parent_parts=True,
            label="cross-game single-game manifest",
        )
        if (
            resolved != expected.manifest_path
            or reference.get("bundle_sha256") != expected.bundle_sha256
            or reference.get("manifest_sha256") != expected.manifest_sha256
        ):
            raise ArtifactVerificationError(
                "cross-game single-game reference differs"
            )
        observed_games.append(game_id)
    if tuple(sorted(observed_games)) != X13_GAME_IDS:
        raise ArtifactVerificationError(
            "cross-game single-game references do not cover 20 games"
        )
    stage_paths, stage_rows = _verify_stage_descriptors_shallow(
        manifest,
        manifest_path=manifest_path,
        project_root=project_root,
        expected_grains=NOTEBOOK_CROSS_STAGE_GRAINS,
        scope="cross_game",
        game_id=None,
    )
    return (
        ShallowCrossGameBundleV2(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            bundle_sha256=expected_bundle_sha256,
            game_ids=X13_GAME_IDS,
            stage_paths=stage_paths,
            holdout_reaction_accessed=False,
        ),
        stage_rows,
    )


def _verify_source_descriptors_shallow(
    values: object,
    *,
    project_root: Path,
) -> tuple[tuple[str, Path, str, int], ...]:
    if not isinstance(values, Mapping) or not values:
        raise ArtifactVerificationError("run index source artifacts are missing")
    result: list[tuple[str, Path, str, int]] = []
    for name, descriptor in sorted(values.items()):
        if type(name) is not str or not name or not isinstance(
            descriptor, Mapping
        ):
            raise ArtifactVerificationError(
                "run index source artifact descriptor is invalid"
            )
        path = _resolve_repo_path(
            descriptor.get("path"),
            project_root=project_root,
            label=f"source artifact {name}",
        )
        digest = _require_sha256(
            descriptor.get("sha256"),
            label=f"source artifact {name}",
        )
        byte_length = _require_nonnegative_int(
            descriptor.get("byte_length"),
            label=f"source artifact {name} byte length",
        )
        if path.stat().st_size != byte_length:
            raise ArtifactVerificationError(
                f"source artifact {name} byte length differs"
            )
        result.append((name, path, digest, byte_length))
    return tuple(result)


def _verify_run_index_shallow(
    path: Path,
    *,
    project_root: Path,
) -> tuple[
    ShallowFactorLabRunIndexV2,
    tuple[ShallowSingleGameBundleV2, ...],
    ShallowCrossGameBundleV2,
]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError("run index is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise ArtifactVerificationError("run index is not canonical")
    if (
        value.get("schema") != "NFLFactorLabRunIndexV2"
        or value.get("experiment_id") != "X-13"
        or value.get("claim_boundary") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or value.get("holdout_reaction_accessed") is not False
        or value.get("shortlist_lock_path") is not None
        or value.get("game_ids") != list(X13_GAME_IDS)
    ):
        raise ArtifactVerificationError("run index governance differs")

    records = value.get("single_game_bundles")
    if not isinstance(records, list) or len(records) != len(X13_GAME_IDS):
        raise ArtifactVerificationError(
            "run index must contain exactly 20 single-game bundles"
        )
    singles: list[ShallowSingleGameBundleV2] = []
    stage_rows: list[NotebookStageManifestEntryV2] = []
    observed_games: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactVerificationError(
                "single-game bundle record is invalid"
            )
        game_id = record.get("game_id")
        bundle_sha = _require_sha256(
            record.get("bundle_sha256"),
            label="single-game bundle",
        )
        if type(game_id) is not str or game_id not in X13_GAME_IDS:
            raise ArtifactVerificationError(
                "single-game bundle game identity differs"
            )
        manifest_path, manifest_sha = _verify_committed_manifest_descriptor(
            record.get("manifest"),
            project_root=project_root,
            label="single-game manifest",
        )
        single, rows = _verify_single_manifest_shallow(
            manifest_path,
            manifest_sha256=manifest_sha,
            expected_game_id=game_id,
            expected_bundle_sha256=bundle_sha,
            project_root=project_root,
        )
        singles.append(single)
        stage_rows.extend(rows)
        observed_games.append(game_id)
    if tuple(sorted(observed_games)) != X13_GAME_IDS:
        raise ArtifactVerificationError(
            "single-game bundle records do not cover 20 games"
        )
    singles = sorted(singles, key=lambda bundle: bundle.game_id)

    cross_record = value.get("cross_game_bundle")
    if not isinstance(cross_record, Mapping):
        raise ArtifactVerificationError(
            "cross-game bundle record is invalid"
        )
    cross_bundle_sha = _require_sha256(
        cross_record.get("bundle_sha256"),
        label="cross-game bundle",
    )
    cross_manifest_path, cross_manifest_sha = (
        _verify_committed_manifest_descriptor(
            cross_record.get("manifest"),
            project_root=project_root,
            label="cross-game manifest",
        )
    )
    cross, cross_rows = _verify_cross_manifest_shallow(
        cross_manifest_path,
        manifest_sha256=cross_manifest_sha,
        expected_bundle_sha256=cross_bundle_sha,
        singles=singles,
        project_root=project_root,
    )
    stage_rows.extend(cross_rows)
    canonical_stage_rows = tuple(
        sorted(
            stage_rows,
            key=lambda row: (
                row.scope,
                row.game_id or "",
                row.stage,
            ),
        )
    )
    expected_stage_count = (
        len(X13_GAME_IDS) * len(NOTEBOOK_SINGLE_STAGE_GRAINS)
        + len(NOTEBOOK_CROSS_STAGE_GRAINS)
    )
    raw_stage_manifest = value.get("stage_manifest")
    if (
        len(canonical_stage_rows) != expected_stage_count
        or expected_stage_count != 345
        or not isinstance(raw_stage_manifest, list)
        or raw_stage_manifest
        != [row.to_dict() for row in canonical_stage_rows]
    ):
        raise ArtifactVerificationError(
            "run index 345-stage manifest differs"
        )
    sources = _verify_source_descriptors_shallow(
        value.get("source_artifacts"),
        project_root=project_root,
    )
    s3 = value.get("s3")
    if (
        not isinstance(s3, Mapping)
        or type(s3.get("bucket")) is not str
        or not s3.get("bucket")
        or type(s3.get("prefix")) is not str
        or not s3.get("prefix")
    ):
        raise ArtifactVerificationError("S3 configuration is invalid")
    run_index = ShallowFactorLabRunIndexV2(
        path=path,
        run_index_sha256=_file_sha256(path),
        game_ids=X13_GAME_IDS,
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in singles
        ),
        cross_game_manifest=cross.manifest_path,
        source_artifacts=sources,
        stage_manifest=canonical_stage_rows,
        s3_bucket=str(s3["bucket"]),
        s3_prefix=str(s3["prefix"]),
        holdout_reaction_accessed=False,
    )
    return run_index, tuple(singles), cross


def verify_factor_lab_notebook_run(
    path: str | Path,
    *,
    project_root: str | Path,
    deep_verify: bool | None = None,
) -> FactorLabNotebookRunV2:
    """Open the canonical V2 run without re-reading 3.58 GiB by default.

    The default verifies the canonical run-index bytes, hashes every committed
    bundle manifest, validates all local path and byte-length commitments, and
    reconciles the exact 20-game/345-stage graph.  Stage Parquet bytes and
    semantic row hashes are deliberately not read in this mode.

    Set ``SAF_FACTOR_LAB_DEEP_VERIFY=1`` (or pass ``deep_verify=True``) to run
    the existing full content and semantic verifiers.
    """

    started = time.perf_counter()
    root = Path(project_root).resolve()
    index_path = Path(path)
    if not index_path.is_absolute():
        index_path = root / index_path
    if index_path.is_symlink():
        raise ArtifactVerificationError("run index cannot be a symlink")
    index_path = index_path.resolve()
    try:
        index_path.relative_to(root)
    except ValueError as error:
        raise ArtifactVerificationError(
            "run index must remain inside project root"
        ) from error
    shallow, singles, cross = _verify_run_index_shallow(
        index_path,
        project_root=root,
    )

    if deep_verify is None:
        configured = os.environ.get(DEEP_VERIFY_ENV, "0")
        if configured not in {"0", "1"}:
            raise ArtifactVerificationError(
                f"{DEEP_VERIFY_ENV} must be 0 or 1"
            )
        deep_verify = configured == "1"
    elif type(deep_verify) is not bool:
        raise ArtifactVerificationError("deep_verify must be boolean")
    if deep_verify:
        verify_factor_lab_run_index(index_path, project_root=root)
        mode = "DEEP_CONTENT_VERIFIED"
        stage_content = "FULL_SHA256_AND_SEMANTIC"
        latest_deep = "VERIFIED_CURRENT_KERNEL"
    else:
        mode = "SHALLOW_COMMIT_VERIFIED"
        stage_content = "NOT_READ"
        latest_deep = "NOT_RECORDED_IN_RUN_INDEX"
    return FactorLabNotebookRunV2(
        run_index=shallow,
        single_game_bundles=singles,
        cross_game_bundle=cross,
        verification_mode=mode,
        stage_content_verification=stage_content,
        latest_deep_batch_status=latest_deep,
        s3_status="CONFIGURED_NOT_REMOTE_VERIFIED",
        verification_elapsed_seconds=time.perf_counter() - started,
    )


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the checkout containing the governed X-13 registry."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (
                (candidate / PREFERRED_RUN_INDEX).is_file()
                or (candidate / DEFAULT_RUN_INDEX).is_file()
            )
        ):
            return candidate
    raise ArtifactVerificationError("cannot locate the Factor Lab project root")


def read_json_object(
    path: str | Path,
    *,
    expected_schema: str | None = None,
) -> dict[str, object]:
    """Read one local JSON object and optionally enforce its schema."""

    source = Path(path)
    try:
        value = json.loads(source.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError(f"cannot read JSON object: {source}") from error
    if type(value) is not dict:
        raise ArtifactVerificationError(f"JSON root must be an object: {source}")
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise ArtifactVerificationError(
            f"JSON schema differs for {source}: expected {expected_schema}"
        )
    return value


def load_run_index(
    project_root: str | Path,
    *,
    path: str | Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Load the strict local run index without resolving its data tables."""

    root = Path(project_root).resolve()
    configured = path or os.environ.get(RUN_INDEX_ENV)
    source = (
        Path(configured).expanduser()
        if configured is not None
        else (
            root / PREFERRED_RUN_INDEX
            if (root / PREFERRED_RUN_INDEX).is_file()
            else root / DEFAULT_RUN_INDEX
        )
    )
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ArtifactVerificationError(
            "run index must remain inside project root"
        ) from error
    value = read_json_object(source)
    if value.get("schema") not in {
        "NFLFactorLabRunIndexV1",
        "NFLFactorLabRunIndexV2",
    }:
        raise ArtifactVerificationError("run index schema is unsupported")
    if value.get("holdout_reaction_accessed") is not False:
        raise ArtifactVerificationError("run index does not prove sealed holdout")
    return source, value


def _descriptor(
    manifest: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value: object = manifest
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ArtifactVerificationError(f"artifact descriptor is missing: {key}")
        value = value[part]
    if not isinstance(value, Mapping):
        raise ArtifactVerificationError(f"artifact descriptor is invalid: {key}")
    return value


def resolve_artifact_ref(
    manifest: Mapping[str, object],
    key: str,
    *,
    project_root: str | Path,
) -> ArtifactRef:
    """Resolve and byte-verify one repo-relative manifest descriptor."""

    descriptor = _descriptor(manifest, key)
    raw_path = descriptor.get("path", descriptor.get("relative_path"))
    digest = descriptor.get("sha256")
    byte_length = descriptor.get("byte_length")
    row_count = descriptor.get("row_count")
    schema = descriptor.get("schema")
    if type(raw_path) is not str or not raw_path:
        raise ArtifactVerificationError(f"artifact path is invalid: {key}")
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactVerificationError(f"artifact SHA-256 is invalid: {key}")
    if isinstance(byte_length, bool) or not isinstance(byte_length, int):
        raise ArtifactVerificationError(f"artifact byte length is invalid: {key}")
    if byte_length < 0:
        raise ArtifactVerificationError(f"artifact byte length is negative: {key}")
    if row_count is not None and (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 0
    ):
        raise ArtifactVerificationError(f"artifact row count is invalid: {key}")
    if schema is not None and type(schema) is not str:
        raise ArtifactVerificationError(f"artifact schema is invalid: {key}")

    root = Path(project_root).resolve()
    source = Path(raw_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ArtifactVerificationError(
            f"artifact must remain inside project root: {key}"
        ) from error
    if not source.is_file():
        raise ArtifactVerificationError(f"artifact file is missing: {source}")
    if source.stat().st_size != byte_length:
        raise ArtifactVerificationError(f"artifact byte length differs: {key}")
    if _file_sha256(source) != digest:
        raise ArtifactVerificationError(f"artifact SHA-256 differs: {key}")
    return ArtifactRef(
        name=key,
        path=source,
        sha256=digest,
        byte_length=byte_length,
        row_count=row_count,
        schema=schema,
    )


def _filter_expression(
    filters: Mapping[str, object] | None,
) -> ds.Expression | None:
    expression: ds.Expression | None = None
    for field, value in (filters or {}).items():
        if type(field) is not str or not field:
            raise ArtifactVerificationError("Parquet filter field is invalid")
        clause = ds.field(field).isin(list(value)) if isinstance(
            value, (tuple, list, set, frozenset)
        ) else ds.field(field) == value
        expression = clause if expression is None else expression & clause
    return expression


def read_parquet_slice(
    reference: ArtifactRef,
    *,
    filters: Mapping[str, object] | None = None,
    columns: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read a projected and filtered Parquet slice.

    ``limit`` is applied by Arrow's scanner, not after constructing a full
    Pandas DataFrame.  The source row count comes from Parquet metadata.
    """

    if not isinstance(reference, ArtifactRef):
        raise ArtifactVerificationError("reference must be an ArtifactRef")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ArtifactVerificationError("Parquet row limit must be positive")
    selected_columns = None if columns is None else list(columns)
    if selected_columns is not None and (
        not selected_columns
        or any(type(item) is not str or not item for item in selected_columns)
    ):
        raise ArtifactVerificationError("Parquet columns are invalid")
    dataset = ds.dataset(reference.path, format="parquet")
    expression = _filter_expression(filters)
    try:
        table = (
            dataset.head(
                limit,
                columns=selected_columns,
                filter=expression,
            )
            if limit is not None
            else dataset.to_table(
                columns=selected_columns,
                filter=expression,
            )
        )
        source_row_count = pq.ParquetFile(reference.path).metadata.num_rows
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ArtifactVerificationError(
            f"cannot read bounded Parquet table: {reference.name}"
        ) from error
    frame = table.to_pandas()
    frame.attrs.update(
        {
            "status": "AVAILABLE",
            "source_path": str(reference.path),
            "source_sha256": reference.sha256,
            "source_byte_length": reference.byte_length,
            "source_row_count": source_row_count,
            "loaded_row_limit": limit,
            "filters": dict(filters or {}),
        }
    )
    return frame


def data_gap_frame(
    *,
    stage: str,
    grain: str,
    reason: str,
    expected_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return an empty, explicitly labelled frame for an unavailable stage."""

    if not stage or not grain or not reason:
        raise ArtifactVerificationError("data-gap metadata must be nonempty")
    columns = list(expected_columns)
    if any(type(item) is not str or not item for item in columns):
        raise ArtifactVerificationError("data-gap columns are invalid")
    frame = pd.DataFrame(columns=columns)
    frame.attrs.update(
        {
            "status": "DATA_GAP",
            "stage": stage,
            "grain": grain,
            "reason": reason,
            "source_sha256": None,
            "source_path": None,
        }
    )
    return frame


def frame_audit(
    frame: pd.DataFrame,
    *,
    name: str,
    grain: str,
    exclusions: int = 0,
) -> pd.DataFrame:
    """Summarize shape, missingness, exclusions, and provenance uniformly."""

    if not isinstance(frame, pd.DataFrame):
        raise ArtifactVerificationError("frame audit requires a DataFrame")
    if not name or not grain:
        raise ArtifactVerificationError("frame audit name and grain are required")
    if (
        isinstance(exclusions, bool)
        or not isinstance(exclusions, int)
        or exclusions < 0
    ):
        raise ArtifactVerificationError("frame exclusions must be nonnegative")
    missing_cells = int(frame.isna().sum().sum())
    cells = len(frame.index) * len(frame.columns)
    return pd.DataFrame(
        [
            {
                "dataframe": name,
                "grain": grain,
                "shape": f"{len(frame.index)} x {len(frame.columns)}",
                "row_count": len(frame.index),
                "column_count": len(frame.columns),
                "missing_cells": missing_cells,
                "missingness_pct": 0.0 if cells == 0 else 100 * missing_cells / cells,
                "excluded_rows": exclusions,
                "status": frame.attrs.get("status", "AVAILABLE"),
                "reason": frame.attrs.get("reason"),
                "source_row_count": frame.attrs.get("source_row_count"),
                "source_sha256": frame.attrs.get("source_sha256"),
                "source_path": frame.attrs.get("source_path"),
            }
        ]
    )


def manifest_frame(
    manifest: Mapping[str, object],
    *,
    manifest_path: str | Path,
    source_sha256: str | None = None,
) -> pd.DataFrame:
    """Flatten file descriptors from one verified local manifest."""

    rows: list[dict[str, object]] = []

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            raw_path = value.get("path", value.get("relative_path"))
            if type(raw_path) is str:
                rows.append(
                    {
                        "artifact": prefix,
                        "path": raw_path,
                        "schema": value.get("schema"),
                        "row_count": value.get("row_count"),
                        "byte_length": value.get("byte_length"),
                        "sha256": value.get("sha256"),
                    }
                )
                return
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{prefix}[{index}]", child)

    visit("", manifest)
    if not rows:
        rows.append(
            {
                "artifact": "manifest_header",
                "path": str(manifest_path),
                "schema": manifest.get("schema"),
                "row_count": manifest.get("game_count"),
                "byte_length": Path(manifest_path).stat().st_size,
                "sha256": source_sha256,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "status": "AVAILABLE",
            "source_path": str(manifest_path),
            "source_sha256": source_sha256,
            "source_row_count": len(frame),
        }
    )
    return frame


def discover_batch_manifest(
    project_root: str | Path,
    run_index: Mapping[str, object],
) -> Path | None:
    """Resolve the published batch deterministically, without directory scans."""

    root = Path(project_root).resolve()
    configured = os.environ.get(BATCH_MANIFEST_ENV)
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()
    if "batch_manifest" in run_index:
        return resolve_artifact_ref(
            run_index,
            "batch_manifest",
            project_root=root,
        ).path
    try:
        cache_reference = resolve_artifact_ref(
            run_index,
            "observed_cache",
            project_root=root,
        )
    except ArtifactVerificationError:
        return None
    cache_manifest_path = cache_reference.path.parent / "manifest.json"
    if not cache_manifest_path.is_file():
        return None
    cache_manifest = read_json_object(cache_manifest_path)
    batch_id = cache_manifest.get("batch_id")
    if type(batch_id) is not str or _SHA256_RE.fullmatch(batch_id) is None:
        return None
    candidate = (
        root
        / "artifacts/market-observation/nfl/x13/batches"
        / batch_id.replace(":", "-")
        / "batch_manifest.json"
    )
    return candidate if candidate.is_file() else None


def _manifest_candidate(
    project_root: Path,
    value: object,
) -> Path | None:
    if type(value) is str:
        candidate = Path(value)
    elif isinstance(value, Mapping) and type(value.get("path")) is str:
        candidate = Path(str(value["path"]))
    else:
        return None
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise ArtifactVerificationError(
            "V2 bundle manifest must remain inside project root"
        ) from error
    return candidate if candidate.is_file() else None


def discover_v2_bundle_manifests(
    project_root: str | Path,
    run_index: Mapping[str, object],
    *,
    game_id: str,
) -> tuple[Path | None, Path | None]:
    """Prefer explicit V2 bundle refs while keeping V1 runs readable.

    Accepted run-index shapes are deliberately narrow:

    - ``single_game_bundle_manifests[game_id]``
    - ``single_game_bundle_manifest``
    - ``cross_game_bundle_manifest``
    - the same keys under a top-level ``v2`` object

    Environment variables override those local references for notebook
    parameterization.  Missing files return ``None``; malformed/out-of-root
    paths fail closed.
    """

    root = Path(project_root).resolve()
    nested = run_index.get("v2")
    v2 = nested if isinstance(nested, Mapping) else run_index

    single_value: object = os.environ.get(SINGLE_GAME_BUNDLE_ENV)
    if single_value is None:
        records = v2.get("single_game_bundles")
        if isinstance(records, list):
            matching = [
                record.get("manifest")
                for record in records
                if isinstance(record, Mapping) and record.get("game_id") == game_id
            ]
            single_value = matching[0] if len(matching) == 1 else None
        else:
            by_game = v2.get("single_game_bundle_manifests")
            single_value = (
                by_game.get(game_id)
                if isinstance(by_game, Mapping)
                else v2.get("single_game_bundle_manifest")
            )
    cross_value: object = os.environ.get(CROSS_GAME_BUNDLE_ENV)
    if cross_value is None:
        cross_record = v2.get("cross_game_bundle")
        cross_value = (
            cross_record.get("manifest")
            if isinstance(cross_record, Mapping)
            else v2.get("cross_game_bundle_manifest")
        )
    return (
        _manifest_candidate(root, single_value),
        _manifest_candidate(root, cross_value),
    )


def read_bundle_stage(
    bundle: object,
    stage: str,
    *,
    filters: Mapping[str, object] | None = None,
    columns: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read one named stage from an already verified V2 bundle."""

    manifest_path = getattr(bundle, "manifest_path", None)
    stage_paths = getattr(bundle, "stage_paths", None)
    if not isinstance(manifest_path, Path) or not isinstance(stage_paths, Mapping):
        raise ArtifactVerificationError("bundle must be a verified V2 publication")
    if stage not in stage_paths:
        raise ArtifactVerificationError(f"V2 bundle stage is missing: {stage}")
    manifest = read_json_object(manifest_path)
    descriptor = _descriptor(manifest, f"stages.{stage}")
    reference = ArtifactRef(
        name=stage,
        path=Path(stage_paths[stage]),
        sha256=str(descriptor["sha256"]),
        byte_length=int(descriptor["byte_length"]),
        row_count=int(descriptor["row_count"]),
        schema=str(descriptor.get("schema_fingerprint")),
    )
    frame = read_parquet_slice(
        reference,
        filters=filters,
        columns=columns,
        limit=limit,
    )
    frame.attrs["grain"] = tuple(descriptor.get("grain", ()))
    frame.attrs["semantic_rows_sha256"] = descriptor.get(
        "semantic_rows_sha256"
    )
    frame.attrs["bundle_manifest_path"] = str(manifest_path)
    frame.attrs["bundle_sha256"] = getattr(bundle, "bundle_sha256", None)
    return frame


def load_v2_batch_attrition_summary(
    notebook_run: FactorLabNotebookRunV2,
) -> V2BatchAttritionSummary:
    """Load the verified 20-game V2 funnel without scanning attrition rows.

    The delay-zero projection and reaction-candidate counts come from the
    reconciled run-index stage manifest.  Only the compact ``reaction_paths``
    stages are content-verified and projected to four columns so actual
    observations, the strict analysis gate, and reasons can be audited.
    """

    if not isinstance(notebook_run, FactorLabNotebookRunV2):
        raise ArtifactVerificationError(
            "batch attrition summary requires a verified V2 notebook run"
        )
    if (
        notebook_run.run_index.game_ids != X13_GAME_IDS
        or tuple(
            bundle.game_id for bundle in notebook_run.single_game_bundles
        )
        != X13_GAME_IDS
    ):
        raise ArtifactVerificationError(
            "batch attrition summary requires the frozen 20-game coverage"
        )

    relevant_entries = [
        entry
        for entry in notebook_run.run_index.stage_manifest
        if entry.scope == "single_game"
        and entry.stage in {"association_attrition", "reaction_paths"}
    ]
    entries_by_stage: dict[
        str, dict[str, NotebookStageManifestEntryV2]
    ] = {}
    for stage in ("association_attrition", "reaction_paths"):
        rows = [entry for entry in relevant_entries if entry.stage == stage]
        by_game = {
            str(entry.game_id): entry
            for entry in rows
            if entry.game_id is not None
        }
        if (
            len(rows) != len(X13_GAME_IDS)
            or tuple(sorted(by_game)) != X13_GAME_IDS
            or any(
                entry.status != "PUBLISHED_VERIFIED" for entry in rows
            )
        ):
            raise ArtifactVerificationError(
                f"batch attrition stage coverage differs: {stage}"
            )
        entries_by_stage[stage] = by_game

    association_projection_rows = sum(
        entry.row_count
        for entry in entries_by_stage["association_attrition"].values()
    )
    reaction_manifest_rows = sum(
        entry.row_count
        for entry in entries_by_stage["reaction_paths"].values()
    )
    compact_columns = (
        "game_id",
        "actual_observation",
        "analysis_eligible",
        "analysis_exclusion_reason",
    )
    compact_frames: list[pd.DataFrame] = []
    for bundle in notebook_run.single_game_bundles:
        entry = entries_by_stage["reaction_paths"][bundle.game_id]
        stage_path = bundle.stage_paths.get("reaction_paths")
        if (
            not isinstance(stage_path, Path)
            or _file_sha256(stage_path) != entry.artifact_sha256
        ):
            raise ArtifactVerificationError(
                "reaction_paths compact stage SHA-256 differs: "
                f"{bundle.game_id}"
            )
        frame = read_bundle_stage(
            bundle,
            "reaction_paths",
            columns=compact_columns,
        )
        if (
            len(frame) != entry.row_count
            or frame.attrs.get("source_row_count") != entry.row_count
            or not frame["game_id"].eq(bundle.game_id).all()
            or frame["actual_observation"].isna().any()
            or frame["analysis_eligible"].isna().any()
        ):
            raise ArtifactVerificationError(
                "reaction_paths compact stage differs from its manifest: "
                f"{bundle.game_id}"
            )
        eligible = frame["analysis_eligible"].eq(True)
        reasons = frame["analysis_exclusion_reason"].astype("string")
        has_reason = reasons.notna() & reasons.str.strip().ne("")
        if (
            not reasons.loc[eligible].eq("ELIGIBLE").fillna(False).all()
            or not has_reason.loc[~eligible].all()
            or reasons.loc[~eligible].eq("ELIGIBLE").any()
        ):
            raise ArtifactVerificationError(
                "reaction_paths analysis reason contract differs: "
                f"{bundle.game_id}"
            )
        actual_observation = frame["actual_observation"].eq(True)
        if (eligible & ~actual_observation).any():
            raise ArtifactVerificationError(
                "reaction_paths actual-observation contract differs: "
                f"{bundle.game_id}"
            )
        compact_frames.append(frame)

    compact = pd.concat(compact_frames, ignore_index=True)
    missing_candidate_rows = (
        association_projection_rows - reaction_manifest_rows
    )
    if len(compact) != reaction_manifest_rows or missing_candidate_rows < 0:
        raise ArtifactVerificationError(
            "reaction_paths compact rows do not reconcile to the run index"
        )
    actual_observation_rows = int(
        compact["actual_observation"].eq(True).sum()
    )
    strict_eligible_rows = int(
        compact["analysis_eligible"].eq(True).sum()
    )
    reason_counts = (
        compact.groupby(
            "analysis_exclusion_reason",
            dropna=False,
            sort=False,
        )
        .size()
        .rename("rows")
        .reset_index()
        .rename(columns={"analysis_exclusion_reason": "reason"})
    )
    reason_counts["disposition"] = reason_counts["reason"].map(
        lambda reason: "ELIGIBLE" if reason == "ELIGIBLE" else "EXCLUDED"
    )
    reason_counts = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "reason": "NO_REACTION_PATH_CANDIDATE",
                        "rows": missing_candidate_rows,
                        "disposition": "EXCLUDED",
                    }
                ]
            ),
            reason_counts,
        ],
        ignore_index=True,
    )
    reason_counts = reason_counts.sort_values(
        ["rows", "reason"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    if int(reason_counts["rows"].sum()) != association_projection_rows:
        raise ArtifactVerificationError(
            "batch attrition reasons do not reconcile to delay=0 projection"
        )

    funnel = pd.DataFrame(
        [
            {
                "stage": "association cells",
                "rows": association_projection_rows
                * len(X13_DELAY_SCENARIOS_SECONDS),
                "evidence": (
                    "20-bundle association_attrition manifest rows × "
                    f"{len(X13_DELAY_SCENARIOS_SECONDS)} registered delays"
                ),
            },
            {
                "stage": "delay=0 projection",
                "rows": association_projection_rows,
                "evidence": (
                    "20-bundle association_attrition manifest row_count"
                ),
            },
            {
                "stage": "reaction path candidates",
                "rows": reaction_manifest_rows,
                "evidence": "20-bundle reaction_paths manifest row_count",
            },
            {
                "stage": "actual observations",
                "rows": actual_observation_rows,
                "evidence": (
                    "SHA-256 verified reaction_paths actual_observation"
                ),
            },
            {
                "stage": "strict analysis eligible",
                "rows": strict_eligible_rows,
                "evidence": (
                    "SHA-256 verified compact reaction_paths analysis gate"
                ),
            },
        ]
    )
    provenance = {
        "status": "OFFICIAL_V2_BATCH_VERIFIED",
        "source_path": str(notebook_run.run_index.path),
        "source_sha256": notebook_run.run_index.run_index_sha256,
        "run_index_sha256": notebook_run.run_index.run_index_sha256,
        "game_count": len(X13_GAME_IDS),
        "reaction_manifest_rows_reconciled": True,
        "compact_stage_content_verification": "FULL_SHA256",
    }
    funnel.attrs.update(
        {
            **provenance,
            "source_row_count": len(
                notebook_run.run_index.stage_manifest
            ),
            "grain": "one V2 batch attrition stage",
        }
    )
    reason_counts.attrs.update(
        {
            **provenance,
            "source_row_count": association_projection_rows,
            "compact_source_row_count": reaction_manifest_rows,
            "actual_observation_rows": actual_observation_rows,
            "grain": "one V2 analysis eligibility reason",
        }
    )
    return V2BatchAttritionSummary(
        funnel=funnel,
        reasons=reason_counts,
    )


def read_v2_sports_source_slice(
    bundle: object,
    *,
    limit: int = 10_000,
) -> pd.DataFrame:
    """Expose a bounded raw-row lineage view from two verified V2 stages.

    The V2 bundle deliberately does not duplicate the seasonal raw Parquet.
    ``sports_row_disposition`` preserves every selected raw row and
    ``canonical_events`` carries the source fields that survived adjudication.
    Joining those already-published stages gives the notebook a transparent
    raw-to-canonical view without loading or revalidating the full season.
    """

    dispositions = read_bundle_stage(
        bundle,
        "sports_row_disposition",
        limit=limit,
    )
    canonical = read_bundle_stage(
        bundle,
        "canonical_events",
        limit=limit,
    )
    disposition_keys = {"game_id", "raw_play_id", "canonical_play_id"}
    canonical_keys = {"game_id", "play_id"}
    if not disposition_keys.issubset(dispositions.columns):
        raise ArtifactVerificationError(
            "sports disposition stage lacks raw/canonical play identity"
        )
    if not canonical_keys.issubset(canonical.columns):
        raise ArtifactVerificationError(
            "canonical event stage lacks game/play identity"
        )
    if dispositions.duplicated(["game_id", "raw_play_id"]).any():
        raise ArtifactVerificationError(
            "sports disposition stage duplicates a raw play identity"
        )
    if canonical.duplicated(["game_id", "play_id"]).any():
        raise ArtifactVerificationError(
            "canonical event stage duplicates a canonical play identity"
        )

    left = dispositions.copy()
    right = canonical.copy()
    left["_canonical_join_id"] = left["canonical_play_id"].astype("string")
    right["_canonical_join_id"] = right["play_id"].astype("string")
    overlapping = (
        set(left.columns)
        & set(right.columns)
        - {"game_id", "_canonical_join_id"}
    )
    right = right.rename(
        columns={name: f"canonical_{name}" for name in sorted(overlapping)}
    )
    result = left.merge(
        right,
        on=["game_id", "_canonical_join_id"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["_canonical_join_id"])
    if len(result) != len(dispositions):
        raise ArtifactVerificationError(
            "V2 sports source lineage changed the raw-row count"
        )
    result.attrs.update(
        {
            "status": "OFFICIAL_V2_VERIFIED",
            "source_path": str(getattr(bundle, "manifest_path", "")),
            "source_sha256": getattr(bundle, "bundle_sha256", None),
            "source_row_count": dispositions.attrs.get(
                "source_row_count",
                len(dispositions),
            ),
            "loaded_row_limit": limit,
            "grain": ("game_id", "raw_play_id"),
            "lineage_stages": (
                "sports_row_disposition",
                "canonical_events",
            ),
        }
    )
    return result


def load_game_projection(
    batch_manifest_path: str | Path,
    game_id: str,
) -> tuple[dict[str, object], Path, str]:
    """Load one game projection after checking its registered semantic hash."""

    if type(game_id) is not str or not game_id:
        raise ArtifactVerificationError("game_id must be nonempty")
    manifest_path = Path(batch_manifest_path)
    manifest = read_json_object(
        manifest_path,
        expected_schema="nfl_x13_batch_manifest_v1",
    )
    game_ids = manifest.get("game_ids")
    hashes = manifest.get("game_hashes")
    if (
        type(game_ids) is not list
        or game_id not in game_ids
        or not isinstance(hashes, Mapping)
        or type(hashes.get(game_id)) is not str
    ):
        raise ArtifactVerificationError("game is absent from the published batch")
    source = manifest_path.parent / "games" / f"{game_id}.json"
    payload = read_json_object(source)
    from prediction_market.sports.nfl_x13_batch import canonical_payload_hash

    observed_hash = canonical_payload_hash(payload)
    expected_hash = str(hashes[game_id])
    if observed_hash != expected_hash:
        raise ArtifactVerificationError("game projection semantic hash differs")
    return payload, source, observed_hash


def projection_frame(
    payload: Mapping[str, object],
    key: str,
    *,
    source_path: str | Path,
    source_sha256: str,
) -> pd.DataFrame:
    """Expose one named list/object from a verified game projection."""

    value = payload.get(key)
    if isinstance(value, list):
        frame = pd.json_normalize(value)
    elif isinstance(value, Mapping):
        frame = pd.json_normalize([value])
    else:
        return data_gap_frame(
            stage=key,
            grain=f"published game projection field {key}",
            reason=f"game projection does not contain a record collection named {key}",
        )
    frame.attrs.update(
        {
            "status": "AVAILABLE",
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_row_count": len(frame),
        }
    )
    return frame


__all__ = [
    "ArtifactRef",
    "ArtifactVerificationError",
    "BATCH_MANIFEST_ENV",
    "CROSS_GAME_BUNDLE_ENV",
    "DEEP_VERIFY_ENV",
    "DEFAULT_RUN_INDEX",
    "FactorLabNotebookRunV2",
    "NOTEBOOK_CROSS_STAGE_GRAINS",
    "NOTEBOOK_SINGLE_STAGE_GRAINS",
    "PREFERRED_RUN_INDEX",
    "RUN_INDEX_ENV",
    "ShallowCrossGameBundleV2",
    "ShallowFactorLabRunIndexV2",
    "ShallowSingleGameBundleV2",
    "SINGLE_GAME_BUNDLE_ENV",
    "STAGE_MANIFEST_ENV",
    "VALID_MODES",
    "V2BatchAttritionSummary",
    "data_gap_frame",
    "discover_batch_manifest",
    "discover_v2_bundle_manifests",
    "find_project_root",
    "frame_audit",
    "load_v2_batch_attrition_summary",
    "load_game_projection",
    "load_run_index",
    "manifest_frame",
    "projection_frame",
    "read_bundle_stage",
    "read_json_object",
    "read_parquet_slice",
    "read_v2_sports_source_slice",
    "resolve_artifact_ref",
    "verify_factor_lab_notebook_run",
]
