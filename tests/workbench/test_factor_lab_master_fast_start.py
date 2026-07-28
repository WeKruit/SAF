from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import pandas as pd
import pytest

import prediction_market.workbench.factor_lab_master as master
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


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


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    root: Path,
    *,
    game_id: str,
    bundle_sha256: str,
) -> tuple[Path, list[dict[str, object]]]:
    directory = (
        root
        / "artifacts"
        / "single-game"
        / game_id
        / bundle_sha256.replace(":", "-")
    )
    directory.mkdir(parents=True)
    stages: dict[str, dict[str, object]] = {}
    stage_rows: list[dict[str, object]] = []
    for stage, grain in master.NOTEBOOK_SINGLE_STAGE_GRAINS.items():
        stage_path = directory / f"{stage}.parquet"
        stage_path.write_bytes(f"{game_id}:{stage}".encode())
        stage_sha = "sha256:" + hashlib.sha256(
            f"committed:{game_id}:{stage}".encode()
        ).hexdigest()
        semantic_sha = "sha256:" + hashlib.sha256(
            f"semantic:{game_id}:{stage}".encode()
        ).hexdigest()
        descriptor = {
            "path": stage_path.name,
            "sha256": stage_sha,
            "byte_length": stage_path.stat().st_size,
            "row_count": 1,
            "schema_fingerprint": "sha256:" + "3" * 64,
            "semantic_rows_sha256": semantic_sha,
            "grain": list(grain),
        }
        stages[stage] = descriptor
        stage_rows.append(
            {
                "scope": "single_game",
                "game_id": game_id,
                "stage": stage,
                "grain": list(grain),
                "row_count": 1,
                "semantic_rows_sha256": semantic_sha,
                "artifact_path": stage_path.relative_to(root).as_posix(),
                "artifact_sha256": stage_sha,
                "status": "PUBLISHED_VERIFIED",
            }
        )
    manifest = {
        "schema": "SingleGameAnalysisBundleV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "builder_version": "nfl-factor-lab-bundle-v2",
        "bundle_sha256": bundle_sha256,
        "game_id": game_id,
        "source_hashes": ["sha256:" + "4" * 64],
        "stages": stages,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest_path, stage_rows


def _fixture_run_index(tmp_path: Path) -> Path:
    singles: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    single_manifests: list[tuple[str, str, Path]] = []
    for index, game_id in enumerate(X13_GAME_IDS, start=1):
        bundle_sha = "sha256:" + f"{index:064x}"
        manifest_path, rows = _write_manifest(
            tmp_path,
            game_id=game_id,
            bundle_sha256=bundle_sha,
        )
        single_manifests.append((game_id, bundle_sha, manifest_path))
        stage_rows.extend(rows)
        singles.append(
            {
                "game_id": game_id,
                "bundle_sha256": bundle_sha,
                "manifest": {
                    "path": manifest_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha256(manifest_path),
                    "byte_length": manifest_path.stat().st_size,
                },
            }
        )

    cross_bundle_sha = "sha256:" + "f" * 64
    cross_dir = (
        tmp_path
        / "artifacts"
        / "cross-game"
        / cross_bundle_sha.replace(":", "-")
    )
    cross_dir.mkdir(parents=True)
    cross_stages: dict[str, dict[str, object]] = {}
    for stage, grain in master.NOTEBOOK_CROSS_STAGE_GRAINS.items():
        stage_path = cross_dir / f"{stage}.parquet"
        stage_path.write_bytes(f"cross:{stage}".encode())
        stage_sha = "sha256:" + hashlib.sha256(
            f"committed:cross:{stage}".encode()
        ).hexdigest()
        semantic_sha = "sha256:" + hashlib.sha256(
            f"semantic:cross:{stage}".encode()
        ).hexdigest()
        cross_stages[stage] = {
            "path": stage_path.name,
            "sha256": stage_sha,
            "byte_length": stage_path.stat().st_size,
            "row_count": 1,
            "schema_fingerprint": "sha256:" + "5" * 64,
            "semantic_rows_sha256": semantic_sha,
            "grain": list(grain),
        }
        stage_rows.append(
            {
                "scope": "cross_game",
                "game_id": None,
                "stage": stage,
                "grain": list(grain),
                "row_count": 1,
                "semantic_rows_sha256": semantic_sha,
                "artifact_path": stage_path.relative_to(tmp_path).as_posix(),
                "artifact_sha256": stage_sha,
                "status": "PUBLISHED_VERIFIED",
            }
        )
    cross_manifest = {
        "schema": "CrossGameAnalysisBundleV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "builder_version": "nfl-factor-lab-bundle-v2",
        "bundle_sha256": cross_bundle_sha,
        "game_ids": list(X13_GAME_IDS),
        "holdout_reaction_accessed": False,
        "single_game_bundles": [
            {
                "game_id": game_id,
                "bundle_sha256": bundle_sha,
                "manifest_relative_path": "",
                "manifest_sha256": _sha256(manifest_path),
            }
            for game_id, bundle_sha, manifest_path in single_manifests
        ],
        "stages": cross_stages,
    }
    # Cross references in production are relative to ``path.parent.parent``.
    for record, (_, _, manifest_path) in zip(
        cross_manifest["single_game_bundles"],
        single_manifests,
        strict=True,
    ):
        record["manifest_relative_path"] = Path(
            __import__("os").path.relpath(
                manifest_path,
                cross_dir.parent,
            )
        ).as_posix()
    cross_manifest_path = cross_dir / "manifest.json"
    cross_manifest_path.write_bytes(_canonical_json(cross_manifest))

    source = tmp_path / "source.json"
    source.write_bytes(b"source\n")
    run_index = {
        "schema": "NFLFactorLabRunIndexV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "holdout_reaction_accessed": False,
        "shortlist_lock_path": None,
        "s3": {"bucket": "test-bucket", "prefix": "test-prefix"},
        "game_ids": list(X13_GAME_IDS),
        "single_game_bundles": singles,
        "cross_game_bundle": {
            "bundle_sha256": cross_bundle_sha,
            "manifest": {
                "path": cross_manifest_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(cross_manifest_path),
                "byte_length": cross_manifest_path.stat().st_size,
            },
        },
        "source_artifacts": {
            "source": {
                "path": source.relative_to(tmp_path).as_posix(),
                "sha256": _sha256(source),
                "byte_length": source.stat().st_size,
            }
        },
        "stage_manifest": sorted(
            stage_rows,
            key=lambda row: (
                str(row["scope"]),
                str(row["game_id"] or ""),
                str(row["stage"]),
            ),
        ),
    }
    run_index_path = tmp_path / "run-index.json"
    run_index_path.write_bytes(_canonical_json(run_index))
    return run_index_path


def test_shallow_startup_verifies_commit_without_hashing_stage_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_index_path = _fixture_run_index(tmp_path)
    hashed_paths: list[Path] = []
    original = master._file_sha256

    def tracked(path: Path) -> str:
        hashed_paths.append(Path(path))
        return original(Path(path))

    monkeypatch.setattr(master, "_file_sha256", tracked)

    verified = master.verify_factor_lab_notebook_run(
        run_index_path,
        project_root=tmp_path,
        deep_verify=False,
    )

    assert verified.verification_mode == "SHALLOW_COMMIT_VERIFIED"
    assert verified.stage_content_verification == "NOT_READ"
    assert verified.latest_deep_batch_status == "NOT_RECORDED_IN_RUN_INDEX"
    assert verified.s3_status == "CONFIGURED_NOT_REMOTE_VERIFIED"
    assert len(verified.run_index.game_ids) == 20
    assert len(verified.run_index.stage_manifest) == 345
    assert len(verified.single_game_bundles) == 20
    assert all(path.suffix == ".json" for path in hashed_paths)
    assert not any(path.suffix == ".parquet" for path in hashed_paths)


def test_deep_verifier_runs_only_for_explicit_environment_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_index_path = _fixture_run_index(tmp_path)
    calls: list[Path] = []

    def fake_deep(path: Path, *, project_root: Path) -> object:
        calls.append(Path(path))
        return object()

    monkeypatch.setattr(master, "verify_factor_lab_run_index", fake_deep)
    monkeypatch.delenv(master.DEEP_VERIFY_ENV, raising=False)
    shallow = master.verify_factor_lab_notebook_run(
        run_index_path,
        project_root=tmp_path,
    )
    assert calls == []
    assert shallow.verification_mode == "SHALLOW_COMMIT_VERIFIED"

    monkeypatch.setenv(master.DEEP_VERIFY_ENV, "1")
    deep = master.verify_factor_lab_notebook_run(
        run_index_path,
        project_root=tmp_path,
    )
    assert calls == [run_index_path]
    assert deep.verification_mode == "DEEP_CONTENT_VERIFIED"
    assert deep.stage_content_verification == "FULL_SHA256_AND_SEMANTIC"
    assert deep.latest_deep_batch_status == "VERIFIED_CURRENT_KERNEL"


def test_shallow_startup_fails_closed_on_manifest_tampering(
    tmp_path: Path,
) -> None:
    run_index_path = _fixture_run_index(tmp_path)
    value = json.loads(run_index_path.read_bytes())
    manifest_path = tmp_path / value["single_game_bundles"][0]["manifest"]["path"]
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(
        master.ArtifactVerificationError,
        match="single-game manifest hash or length differs",
    ):
        master.verify_factor_lab_notebook_run(
            run_index_path,
            project_root=tmp_path,
            deep_verify=False,
        )


def test_shallow_startup_rejects_stage_path_escape(
    tmp_path: Path,
) -> None:
    run_index_path = _fixture_run_index(tmp_path)
    value = json.loads(run_index_path.read_bytes())
    record = value["single_game_bundles"][0]
    manifest_path = tmp_path / record["manifest"]["path"]
    manifest = json.loads(manifest_path.read_bytes())
    manifest["stages"]["canonical_events"]["path"] = "../escape.parquet"
    manifest_path.write_bytes(_canonical_json(manifest))
    record["manifest"]["sha256"] = _sha256(manifest_path)
    record["manifest"]["byte_length"] = manifest_path.stat().st_size
    run_index_path.write_bytes(_canonical_json(value))

    with pytest.raises(
        master.ArtifactVerificationError,
        match="stage path is invalid",
    ):
        master.verify_factor_lab_notebook_run(
            run_index_path,
            project_root=tmp_path,
            deep_verify=False,
        )


def test_v2_batch_attrition_summary_uses_all_verified_compact_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    notebook_run = master.verify_factor_lab_notebook_run(
        project_root / master.PREFERRED_RUN_INDEX,
        project_root=project_root,
        deep_verify=False,
    )
    calls: list[tuple[str, str, tuple[str, ...] | None]] = []
    original = master.read_bundle_stage

    def tracked_read(
        bundle: object,
        stage: str,
        *,
        filters: dict[str, object] | None = None,
        columns: tuple[str, ...] | None = None,
        limit: int | None = None,
    ):
        calls.append((str(getattr(bundle, "game_id", "")), stage, columns))
        return original(
            bundle,
            stage,
            filters=filters,
            columns=columns,
            limit=limit,
        )

    monkeypatch.setattr(master, "read_bundle_stage", tracked_read)
    started = time.perf_counter()

    summary = master.load_v2_batch_attrition_summary(notebook_run)

    loader_source = inspect.getsource(master.load_v2_batch_attrition_summary)
    for hardcoded_count in (
        "48624912",
        "8104152",
        "64058",
        "63772",
        "2508",
    ):
        assert hardcoded_count not in loader_source.replace("_", "")
    assert summary.funnel[["stage", "rows"]].to_dict(orient="records") == [
        {"stage": "association cells", "rows": 48_624_912},
        {"stage": "delay=0 projection", "rows": 8_104_152},
        {"stage": "reaction path candidates", "rows": 64_058},
        {"stage": "actual observations", "rows": 63_772},
        {"stage": "strict analysis eligible", "rows": 2_508},
    ]
    assert summary.reasons[["reason", "rows"]].to_dict(orient="records") == [
        {"reason": "NO_REACTION_PATH_CANDIDATE", "rows": 8_040_094},
        {"reason": "SIGNAL_CANDIDATE_INELIGIBLE", "rows": 58_665},
        {"reason": "NON_FOCAL_OUTCOME", "rows": 2_599},
        {"reason": "ELIGIBLE", "rows": 2_508},
        {"reason": "ORDER_AMBIGUOUS", "rows": 286},
    ]
    assert int(summary.reasons["rows"].sum()) == 8_104_152
    assert calls == [
        (
            game_id,
            "reaction_paths",
            (
                "game_id",
                "actual_observation",
                "analysis_eligible",
                "analysis_exclusion_reason",
            ),
        )
        for game_id in X13_GAME_IDS
    ]
    assert summary.funnel.attrs["game_count"] == 20
    assert summary.funnel.attrs["run_index_sha256"] == (
        notebook_run.run_index.run_index_sha256
    )
    assert summary.funnel.attrs["reaction_manifest_rows_reconciled"] is True
    assert time.perf_counter() - started < 10.0


def test_v2_batch_attrition_summary_separates_candidates_and_actual_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_sha = "sha256:" + "a" * 64
    stage_manifest = tuple(
        master.NotebookStageManifestEntryV2(
            scope="single_game",
            game_id=game_id,
            stage=stage,
            grain=("game_id",),
            row_count=4 if stage == "association_attrition" else 3,
            semantic_rows_sha256="sha256:" + "b" * 64,
            artifact_path=f"{game_id}/{stage}.parquet",
            artifact_sha256=artifact_sha,
        )
        for game_id in X13_GAME_IDS
        for stage in ("association_attrition", "reaction_paths")
    )
    bundles = tuple(
        master.ShallowSingleGameBundleV2(
            manifest_path=tmp_path / f"{game_id}.json",
            manifest_sha256="sha256:" + "c" * 64,
            bundle_sha256="sha256:" + "d" * 64,
            game_id=game_id,
            stage_paths={
                "reaction_paths": tmp_path / f"{game_id}.parquet"
            },
            source_hashes=("sha256:" + "e" * 64,),
        )
        for game_id in X13_GAME_IDS
    )
    run_index = master.ShallowFactorLabRunIndexV2(
        path=tmp_path / "run-index.json",
        run_index_sha256="sha256:" + "f" * 64,
        game_ids=X13_GAME_IDS,
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in bundles
        ),
        cross_game_manifest=tmp_path / "cross.json",
        source_artifacts=(),
        stage_manifest=stage_manifest,
        s3_bucket="test",
        s3_prefix="test",
        holdout_reaction_accessed=False,
    )
    notebook_run = master.FactorLabNotebookRunV2(
        run_index=run_index,
        single_game_bundles=bundles,
        cross_game_bundle=master.ShallowCrossGameBundleV2(
            manifest_path=tmp_path / "cross.json",
            manifest_sha256="sha256:" + "1" * 64,
            bundle_sha256="sha256:" + "2" * 64,
            game_ids=X13_GAME_IDS,
            stage_paths={},
            holdout_reaction_accessed=False,
        ),
        verification_mode="SHALLOW_COMMIT_VERIFIED",
        stage_content_verification="NOT_READ",
        latest_deep_batch_status="NOT_RECORDED_IN_RUN_INDEX",
        s3_status="CONFIGURED_NOT_REMOTE_VERIFIED",
        verification_elapsed_seconds=0.0,
    )
    ineligible_reason = {"value": "STALE"}
    eligible_reason = {"value": "ELIGIBLE"}
    gap_actual_observation = {"value": False}
    gap_analysis_eligible = {"value": False}
    gap_reason = {"value": "ORDER_AMBIGUOUS"}

    def compact_stage(
        bundle: object,
        stage: str,
        **_: object,
    ) -> object:
        frame = pd.DataFrame(
            {
                "game_id": [bundle.game_id] * 3,
                "actual_observation": [
                    True,
                    True,
                    gap_actual_observation["value"],
                ],
                "analysis_eligible": [
                    True,
                    False,
                    gap_analysis_eligible["value"],
                ],
                "analysis_exclusion_reason": [
                    eligible_reason["value"],
                    ineligible_reason["value"],
                    gap_reason["value"],
                ],
            }
        )
        frame.attrs["source_row_count"] = 3
        return frame

    monkeypatch.setattr(master, "_file_sha256", lambda _: artifact_sha)
    monkeypatch.setattr(master, "read_bundle_stage", compact_stage)

    summary = master.load_v2_batch_attrition_summary(notebook_run)

    assert summary.funnel[["stage", "rows"]].to_dict(orient="records") == [
        {"stage": "association cells", "rows": 480},
        {"stage": "delay=0 projection", "rows": 80},
        {"stage": "reaction path candidates", "rows": 60},
        {"stage": "actual observations", "rows": 40},
        {"stage": "strict analysis eligible", "rows": 20},
    ]
    assert summary.reasons[["reason", "rows"]].to_dict(
        orient="records"
    ) == [
        {"reason": "ELIGIBLE", "rows": 20},
        {"reason": "NO_REACTION_PATH_CANDIDATE", "rows": 20},
        {"reason": "ORDER_AMBIGUOUS", "rows": 20},
        {"reason": "STALE", "rows": 20},
    ]

    ineligible_reason["value"] = ""
    with pytest.raises(
        master.ArtifactVerificationError,
        match="reaction_paths analysis reason contract differs",
    ):
        master.load_v2_batch_attrition_summary(notebook_run)

    ineligible_reason["value"] = "STALE"
    eligible_reason["value"] = None
    with pytest.raises(
        master.ArtifactVerificationError,
        match="reaction_paths analysis reason contract differs",
    ):
        master.load_v2_batch_attrition_summary(notebook_run)

    eligible_reason["value"] = "ELIGIBLE"
    gap_analysis_eligible["value"] = True
    gap_reason["value"] = "ELIGIBLE"
    with pytest.raises(
        master.ArtifactVerificationError,
        match="reaction_paths actual-observation contract differs",
    ):
        master.load_v2_batch_attrition_summary(notebook_run)

    gap_analysis_eligible["value"] = False
    gap_reason["value"] = "ORDER_AMBIGUOUS"
    gap_actual_observation["value"] = None
    with pytest.raises(
        master.ArtifactVerificationError,
        match="reaction_paths compact stage differs from its manifest",
    ):
        master.load_v2_batch_attrition_summary(notebook_run)
