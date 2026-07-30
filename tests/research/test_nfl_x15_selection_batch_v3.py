from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

import prediction_market.research.nfl_x15_selection_batch_v3 as selection_v3
from prediction_market.research.nfl_x15_oof_publication import (
    publish_x15_oof_shard,
)
from prediction_market.research.nfl_x15_model_selection import (
    _stage_b_projection_identity,
)
from prediction_market.research.nfl_x15_selection_batch_v3 import (
    EXACT_SELECTION_CELLS_V3,
    X15SelectionBatchV3Error,
    load_verified_x15_selection_batch_v3,
    load_x15_selection_projection_v3,
    publish_x15_selection_batch_v3,
)


_HELPER_PATH = Path(__file__).with_name(
    "test_nfl_x15_oof_publication.py"
)
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_x15_oof_test_helpers_v3", _HELPER_PATH
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)
AUTHORITY_SHA = _HELPERS.AUTHORITY_SHA
MAPPING_SHA = _HELPERS.MAPPING_SHA
_authority = _HELPERS._authority
_fold_run = _HELPERS._fold_run


def _canonical_bytes(value: object) -> bytes:
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


def _publish_json(
    root: Path,
    *,
    namespace: str,
    suffix: str,
    material: dict[str, object],
    digest_field: str,
    semantic_newline: bool = True,
) -> tuple[Path, str, str]:
    semantic_payload = (
        _canonical_bytes(material)
        if semantic_newline
        else json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    semantic_sha = "sha256:" + hashlib.sha256(
        semantic_payload
    ).hexdigest()
    document = {**material, digest_field: semantic_sha}
    payload = _canonical_bytes(document)
    file_sha = "sha256:" + hashlib.sha256(payload).hexdigest()
    digest = file_sha.removeprefix("sha256:")
    path = (
        root
        / namespace
        / "sha256"
        / digest[:2]
        / f"{digest}{suffix}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, file_sha, semantic_sha


def _publish_panel_and_support_audit(
    root: Path,
) -> tuple[Path, str, Path, str]:
    panel_material = {
        "schema": "nfl_x15_development_panel_batch_index_v1",
        "builder_version": "nfl-x15-development-panel-v2",
        "cohort": "development",
        "cohort_authority_sha256": AUTHORITY_SHA,
        "cohort_mapping_sha256": MAPPING_SHA,
        "published_game_count": 153,
        "verified_development_game_count": 153,
        "partial_publication": False,
        "publication_gate": "PASS",
        "holdout_reaction_accessed": False,
    }
    panel_path, panel_file_sha, panel_batch_sha = _publish_json(
        root,
        namespace="panel/batches/manifests",
        suffix=".batch-index.json",
        material=panel_material,
        digest_field="batch_sha256",
    )
    audit_material = {
        "schema": "nfl_x15_stage_a_decision_support_audit_v1",
        "builder_version": (
            "nfl-x15-stage-a-decision-support-audit-v1"
        ),
        "experiment_id": "X-15",
        "cohort": "development",
        "panel_batch_manifest_path": panel_path.relative_to(root).as_posix(),
        "panel_batch_file_sha256": panel_file_sha,
        "panel_batch_sha256": panel_batch_sha,
        "cohort_authority_sha256": AUTHORITY_SHA,
        "cohort_mapping_sha256": MAPPING_SHA,
        "decision_eligible_rows": 1_050_084,
        "nonnull_counts": {
            "p_before_home": 0,
            "p_after_home": 0,
            "reference_delta_home": 0,
            "reference_gap_at_landmark": 0,
        },
        "feature_block_status": {
            "D4": "UNSUPPORTED_FEATURE_BLOCK",
        },
        "selection_eligible": {"D4": False},
        "support_reason": {
            "D4": (
                "ZERO_NON_NULL_STAGE_A_REFERENCE_FEATURES_ON_"
                "DECISION_ELIGIBLE_ROWS"
            )
        },
        "claim_boundary": (
            "DEVELOPMENT_DECISION_TIME_FEATURE_SUPPORT_AUDIT_"
            "NOT_MARKET_REACTION_RESULT"
        ),
        "holdout_reaction_accessed": False,
    }
    audit_path, audit_file_sha, _ = _publish_json(
        root,
        namespace="support-audit/manifests",
        suffix=".stage-a-support-audit.json",
        material=audit_material,
        digest_field="audit_sha256",
        semantic_newline=False,
    )
    return panel_path, panel_file_sha, audit_path, audit_file_sha


def _publish_shards(
    root: Path,
    cells: tuple[tuple[str, str, str], ...],
) -> tuple[Path, ...]:
    authority = _authority()
    paths: list[Path] = []
    for fold_id, model_id, feature_block_id in cells:
        run = _fold_run(
            fold_id,
            authority=authority,
            model_id=model_id,
            feature_block_id=feature_block_id,
        )
        frame = run.oof_predictions.copy()
        calibration_games = tuple(
            frame.iloc[0]["training_game_ids"]
        )[:2]
        for column in (
            "calibrator_fit_game_ids_s_h",
            "calibrator_fit_game_ids_o_h_given_s",
            "calibrator_fit_game_ids_direction",
        ):
            frame[column] = [calibration_games] * len(frame)
        for column in (
            "training_data_sha256",
            "preprocessor_training_sha256",
            "s_h_model_training_sha256",
            "o_h_given_s_model_training_sha256",
            "direction_model_training_sha256",
            "s_h_calibration_training_sha256",
            "o_h_given_s_calibration_training_sha256",
            "direction_calibration_training_sha256",
            "feature_block_sha256",
            "model_spec_sha256",
            "fold_sha256",
        ):
            digest = "sha256:" + hashlib.sha256(
                (
                    f"{fold_id}/{model_id}/{feature_block_id}/{column}"
                ).encode("utf-8")
            ).hexdigest()
            frame[column] = digest
        run = replace(run, oof_predictions=frame)
        publication = publish_x15_oof_shard(
            model_run=run,
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=root,
        )
        paths.append(publication.manifest_path)
    return tuple(paths)


def _republish_self_hashed_shard(
    root: Path,
    source: Path,
    *,
    field: str,
    value: object,
) -> Path:
    document = json.loads(source.read_text())
    material = dict(document)
    material.pop("bundle_sha256")
    material[field] = value
    bundle_sha = "sha256:" + hashlib.sha256(
        _canonical_bytes(material)
    ).hexdigest()
    payload = _canonical_bytes(
        {**material, "bundle_sha256": bundle_sha}
    )
    file_sha = "sha256:" + hashlib.sha256(payload).hexdigest()
    digest = file_sha.removeprefix("sha256:")
    path = (
        root
        / "shards"
        / str(material["fold_id"])
        / str(material["model_id"])
        / str(material["feature_block_id"])
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.shard-manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture(scope="module")
def exact45_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("exact45")
    paths = _publish_shards(root, EXACT_SELECTION_CELLS_V3)
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(root)
    )
    return root, paths, panel_path, panel_sha, audit_path, audit_sha


def test_exact45_batch_is_selection_ready_and_projection_is_verified(
    exact45_source,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    authority = _authority()

    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )

    assert publication.selection_ready is True
    document, path, file_sha, loaded_authority = (
        load_verified_x15_selection_batch_v3(
            batch_manifest_path=publication.manifest_path,
            artifact_root=root,
        )
    )
    assert path == publication.manifest_path
    assert file_sha == publication.manifest_sha256
    assert loaded_authority == authority
    assert document["expected_execution_cell_count"] == 45
    assert document["observed_execution_cell_count"] == 45
    assert document["missing_execution_cell_count"] == 0
    assert document["selection_ready"] is True
    assert document["holdout_reaction_accessed"] is False
    assert document["feature_support_audit"]["file_sha256"] == audit_sha
    anchor = document["source_provenance_anchor"]
    assert anchor["schema"] == (
        "nfl_x15_selection_source_provenance_anchor_v1"
    )
    assert anchor["row_count"] == 45
    assert len(anchor["rows"]) == 45
    assert len(anchor["column_types"]) == len(anchor["rows"][0])
    assert anchor["rows"][0]["fold_id"] == "fold_01"
    assert anchor["rows"][0]["model_id"] == "b0_empirical_v1"
    assert anchor["rows"][0]["feature_block_id"] == "D0"
    assert anchor["table_sha256"].startswith("sha256:")

    projection = load_x15_selection_projection_v3(
        batch_manifest_path=publication.manifest_path,
        artifact_root=root,
        candidate_model_id="regularized_logistic_v1",
        candidate_feature_block_id="D2",
    )
    assert set(
        zip(
            projection.oof_predictions["model_id"],
            projection.oof_predictions["feature_block_id"],
            strict=True,
        )
    ) == {
        ("b0_empirical_v1", "D0"),
        ("regularized_logistic_v1", "D2"),
    }
    assert projection.oof_predictions["fold_id"].nunique() == 5
    assert {
        "training_data_sha256",
        "preprocessor_training_sha256",
        "s_h_model_training_sha256",
        "o_h_given_s_model_training_sha256",
        "direction_model_training_sha256",
        "s_h_calibration_training_sha256",
        "o_h_given_s_calibration_training_sha256",
        "direction_calibration_training_sha256",
        "feature_block_sha256",
        "model_spec_sha256",
        "fold_sha256",
        "calibrator_fit_game_ids_s_h",
        "calibrator_fit_game_ids_o_h_given_s",
        "calibrator_fit_game_ids_direction",
    }.issubset(projection.oof_predictions.columns)
    source_shards = projection.run_config["source_shards"]
    assert len(source_shards) == 10
    assert len(
        {
            shard["source_provenance_anchor_table_sha256"]
            for shard in source_shards
        }
    ) == 1
    assert all(
        shard["source_provenance_anchor_schema"]
        == "nfl_x15_selection_source_provenance_anchor_v1"
        and isinstance(
            shard["source_provenance_anchor_row"], dict
        )
        for shard in source_shards
    )
    assert _stage_b_projection_identity(projection)[0] == (
        "regularized_logistic_v1",
        "D2",
    )


def test_batch_publication_and_load_never_materialize_v4_tables(
    exact45_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )

    def forbid_full_load(**_: object):
        raise AssertionError("full V4 pandas loader was called")

    monkeypatch.setattr(
        selection_v3, "load_x15_oof_shard", forbid_full_load
    )
    monkeypatch.setattr(selection_v3.pq, "read_table", forbid_full_load)
    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    loaded = load_verified_x15_selection_batch_v3(
        batch_manifest_path=publication.manifest_path,
        artifact_root=root,
    )
    assert loaded[0]["observed_execution_cell_count"] == 45


def test_projection_full_loads_only_baseline_and_selected_candidate(
    exact45_source,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    original = selection_v3.load_x15_oof_shard
    calls: list[str] = []

    def observe_full_load(**kwargs):
        calls.append(str(kwargs["manifest_path"]))
        return original(**kwargs)

    monkeypatch.setattr(
        selection_v3, "load_x15_oof_shard", observe_full_load
    )
    load_x15_selection_projection_v3(
        batch_manifest_path=publication.manifest_path,
        artifact_root=root,
        candidate_model_id="shallow_xgboost_v1",
        candidate_feature_block_id="D3",
    )
    assert len(calls) == 10
    assert all(
        "/b0_empirical_v1/D0/" in path
        or "/shallow_xgboost_v1/D3/" in path
        for path in calls
    )


def test_44_of_45_publishes_partial_but_verified_loader_rejects(
    exact45_source,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    partial = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths[:-1],
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    document = json.loads(partial.manifest_path.read_text())
    assert partial.selection_ready is False
    assert document["observed_execution_cell_count"] == 44
    assert document["missing_execution_cell_count"] == 1
    with pytest.raises(
        X15SelectionBatchV3Error, match="exact-45"
    ):
        load_verified_x15_selection_batch_v3(
            batch_manifest_path=partial.manifest_path,
            artifact_root=root,
        )


def test_d4_shard_is_rejected_even_for_partial_publication(
    tmp_path: Path,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    d4 = _publish_shards(
        tmp_path,
        (("fold_01", "regularized_logistic_v1", "D4"),),
    )
    with pytest.raises(
        X15SelectionBatchV3Error, match="D4|exact-45"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=d4,
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_duplicate_execution_coordinate_is_rejected(
    tmp_path: Path,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    path = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )[0]
    with pytest.raises(
        X15SelectionBatchV3Error, match="duplicate execution"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(path, path),
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_conflicting_manifests_for_same_coordinate_are_rejected(
    tmp_path: Path,
) -> None:
    authority = _authority()
    first = publish_x15_oof_shard(
        model_run=_fold_run(
            "fold_01",
            authority=authority,
            model_id="b0_empirical_v1",
            feature_block_id="D0",
        ),
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )
    second = publish_x15_oof_shard(
        model_run=_fold_run(
            "fold_01",
            authority=authority,
            model_id="b0_empirical_v1",
            feature_block_id="D0",
            config_override={"random_state": 20260729},
        ),
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )
    assert first.manifest_path != second.manifest_path
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    with pytest.raises(
        X15SelectionBatchV3Error, match="duplicate execution"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(
                first.manifest_path,
                second.manifest_path,
            ),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_metadata_verifier_rejects_tampered_v4_object(
    tmp_path: Path,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    shard = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )[0]
    document = json.loads(shard.read_text())
    prediction = next(
        value
        for value in document["tables"]
        if value["name"] == "oof_predictions"
    )
    object_path = tmp_path / prediction["object_path"]
    object_path.write_bytes(object_path.read_bytes() + b"tampered")

    with pytest.raises(
        X15SelectionBatchV3Error, match="bytes|SHA-256"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(shard,),
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("experiment_id", "FORGED"),
        ("diagnostic_claim_boundary", "FORGED"),
        ("claim_eligible", True),
        ("selection_role", "FORGED"),
        ("include_magnitude", True),
    ),
)
def test_metadata_verifier_rejects_self_hashed_v4_contract_forgery(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    shard = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )[0]
    forged = _republish_self_hashed_shard(
        tmp_path,
        shard,
        field=field,
        value=value,
    )

    with pytest.raises(
        X15SelectionBatchV3Error, match="canonical V4"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(forged,),
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_verified_loader_rejects_self_hashed_forged_provenance_anchor(
    exact45_source,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    document = json.loads(publication.manifest_path.read_text())
    material = dict(document)
    material.pop("batch_sha256")
    anchor = material["source_provenance_anchor"]
    anchor["rows"][0]["fold_sha256"] = "sha256:" + "f" * 64
    anchor_material = {
        key: value
        for key, value in anchor.items()
        if key != "table_sha256"
    }
    anchor["table_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(anchor_material)
    ).hexdigest()
    forged, _, _ = _publish_json(
        root,
        namespace="selection-v3/batches/manifests",
        suffix=".batch-index.json",
        material=material,
        digest_field="batch_sha256",
    )

    with pytest.raises(
        X15SelectionBatchV3Error,
        match="provenance anchor.*source|source.*provenance anchor",
    ):
        load_verified_x15_selection_batch_v3(
            batch_manifest_path=forged,
            artifact_root=root,
        )


def test_verified_loader_rejects_missing_provenance_anchor(
    exact45_source,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    document = json.loads(publication.manifest_path.read_text())
    material = dict(document)
    material.pop("batch_sha256")
    material.pop("source_provenance_anchor")
    forged, _, _ = _publish_json(
        root,
        namespace="selection-v3/batches/manifests",
        suffix=".batch-index.json",
        material=material,
        digest_field="batch_sha256",
    )

    with pytest.raises(
        X15SelectionBatchV3Error, match="provenance anchor"
    ):
        load_verified_x15_selection_batch_v3(
            batch_manifest_path=forged,
            artifact_root=root,
        )


def test_verified_loader_rejects_tampered_provenance_table_hash(
    exact45_source,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    document = json.loads(publication.manifest_path.read_text())
    material = dict(document)
    material.pop("batch_sha256")
    material["source_provenance_anchor"]["table_sha256"] = (
        "sha256:" + "e" * 64
    )
    forged, _, _ = _publish_json(
        root,
        namespace="selection-v3/batches/manifests",
        suffix=".batch-index.json",
        material=material,
        digest_field="batch_sha256",
    )

    with pytest.raises(
        X15SelectionBatchV3Error,
        match="provenance anchor.*SHA-256",
    ):
        load_verified_x15_selection_batch_v3(
            batch_manifest_path=forged,
            artifact_root=root,
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "multi_value"),
)
def test_publication_rejects_invalid_oof_provenance(
    tmp_path: Path,
    mutation: str,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    authority = _authority()
    run = _fold_run("fold_01", authority=authority)
    frame = run.oof_predictions.copy()
    calibration_games = tuple(
        frame.iloc[0]["training_game_ids"]
    )[:2]
    for column in (
        "calibrator_fit_game_ids_s_h",
        "calibrator_fit_game_ids_o_h_given_s",
        "calibrator_fit_game_ids_direction",
    ):
        frame[column] = [calibration_games] * len(frame)
    for column in (
        "training_data_sha256",
        "preprocessor_training_sha256",
        "s_h_model_training_sha256",
        "o_h_given_s_model_training_sha256",
        "direction_model_training_sha256",
        "s_h_calibration_training_sha256",
        "o_h_given_s_calibration_training_sha256",
        "direction_calibration_training_sha256",
        "feature_block_sha256",
        "model_spec_sha256",
        "fold_sha256",
    ):
        frame[column] = "sha256:" + hashlib.sha256(
            column.encode("utf-8")
        ).hexdigest()
    if mutation == "missing":
        frame = frame.drop(columns=["training_data_sha256"])
    else:
        frame.loc[frame.index[0], "fold_sha256"] = (
            "sha256:" + "d" * 64
        )
    run = replace(run, oof_predictions=frame)
    shard = publish_x15_oof_shard(
        model_run=run,
        authority=authority,
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=tmp_path,
    )

    with pytest.raises(
        X15SelectionBatchV3Error,
        match="provenance.*missing|provenance.*multiple",
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(shard.manifest_path,),
            authority=authority,
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_selection_batch_rejects_path_escape(
    tmp_path: Path,
) -> None:
    other = tmp_path.parent / f"{tmp_path.name}-other"
    other.mkdir()
    external = _publish_shards(
        other, (("fold_01", "b0_empirical_v1", "D0"),)
    )[0]
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    with pytest.raises(X15SelectionBatchV3Error, match="escapes"):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(external,),
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_output_root_must_be_within_project_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    project_root.mkdir()
    external_root.mkdir()
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(project_root)
    )
    shards = _publish_shards(
        external_root, (("fold_01", "b0_empirical_v1", "D0"),)
    )
    with pytest.raises(X15SelectionBatchV3Error, match="output_root"):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=shards,
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=external_root,
            project_root=project_root,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_verified_loader_rejects_incoherent_root_contract(
    exact45_source,
) -> None:
    root, paths, panel_path, panel_sha, audit_path, audit_sha = (
        exact45_source
    )
    publication = publish_x15_selection_batch_v3(
        shard_manifest_paths=paths,
        authority=_authority(),
        cohort_mapping_sha256=MAPPING_SHA,
        output_root=root,
        project_root=root,
        panel_batch_manifest_path=panel_path,
        panel_batch_file_sha256=panel_sha,
        feature_support_audit_path=audit_path,
        feature_support_audit_file_sha256=audit_sha,
    )
    document = json.loads(publication.manifest_path.read_text())
    material = dict(document)
    material.pop("batch_sha256")
    material["root_contract"] = {
        **material["root_contract"],
        "output_root": "external",
    }
    forged_path, _, _ = _publish_json(
        root,
        namespace="selection-v3/batches/manifests",
        suffix=".batch-index.json",
        material=material,
        digest_field="batch_sha256",
    )
    with pytest.raises(
        X15SelectionBatchV3Error, match="root contract"
    ):
        load_verified_x15_selection_batch_v3(
            batch_manifest_path=forged_path,
            artifact_root=root,
        )


def test_shard_manifest_must_use_exact_coordinate_namespace(
    tmp_path: Path,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    shard = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )[0]
    digest = shard.name.removesuffix(".shard-manifest.json")
    forged = (
        tmp_path
        / "forged"
        / "manifests"
        / "sha256"
        / digest[:2]
        / shard.name
    )
    forged.parent.mkdir(parents=True)
    forged.write_bytes(shard.read_bytes())
    with pytest.raises(
        X15SelectionBatchV3Error, match="namespace"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=(forged,),
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=audit_path,
            feature_support_audit_file_sha256=audit_sha,
        )


def test_audit_must_prove_d4_is_unsupported(
    tmp_path: Path,
) -> None:
    panel_path, panel_sha, audit_path, audit_sha = (
        _publish_panel_and_support_audit(tmp_path)
    )
    document = json.loads(audit_path.read_text())
    material = dict(document)
    material.pop("audit_sha256")
    material["feature_block_status"] = {"D4": "SUPPORTED"}
    forged_path, forged_sha, _ = _publish_json(
        tmp_path,
        namespace="forged-audit/manifests",
        suffix=".stage-a-support-audit.json",
        material=material,
        digest_field="audit_sha256",
        semantic_newline=False,
    )
    shard = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )
    with pytest.raises(
        X15SelectionBatchV3Error, match="UNSUPPORTED_FEATURE_BLOCK"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=shard,
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=forged_path,
            feature_support_audit_file_sha256=forged_sha,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selection_eligible", {"D4": True}),
        ("support_reason", {"D4": "CONTRADICTORY"}),
        ("claim_boundary", "WRONG"),
    ),
)
def test_audit_rejects_contradictory_production_support_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    panel_path, panel_sha, audit_path, _ = (
        _publish_panel_and_support_audit(tmp_path)
    )
    document = json.loads(audit_path.read_text())
    material = dict(document)
    material.pop("audit_sha256")
    material[field] = value
    forged_path, forged_sha, _ = _publish_json(
        tmp_path,
        namespace=f"forged-{field}/manifests",
        suffix=".stage-a-support-audit.json",
        material=material,
        digest_field="audit_sha256",
        semantic_newline=False,
    )
    shard = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )
    with pytest.raises(
        X15SelectionBatchV3Error, match="feature support audit"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=shard,
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=forged_path,
            feature_support_audit_file_sha256=forged_sha,
        )


def test_audit_rejects_missing_required_production_support_field(
    tmp_path: Path,
) -> None:
    panel_path, panel_sha, audit_path, _ = (
        _publish_panel_and_support_audit(tmp_path)
    )
    document = json.loads(audit_path.read_text())
    material = dict(document)
    material.pop("audit_sha256")
    material.pop("selection_eligible")
    forged_path, forged_sha, _ = _publish_json(
        tmp_path,
        namespace="forged-missing/manifests",
        suffix=".stage-a-support-audit.json",
        material=material,
        digest_field="audit_sha256",
        semantic_newline=False,
    )
    shard = _publish_shards(
        tmp_path, (("fold_01", "b0_empirical_v1", "D0"),)
    )
    with pytest.raises(
        X15SelectionBatchV3Error, match="feature support audit"
    ):
        publish_x15_selection_batch_v3(
            shard_manifest_paths=shard,
            authority=_authority(),
            cohort_mapping_sha256=MAPPING_SHA,
            output_root=tmp_path,
            project_root=tmp_path,
            panel_batch_manifest_path=panel_path,
            panel_batch_file_sha256=panel_sha,
            feature_support_audit_path=forged_path,
            feature_support_audit_file_sha256=forged_sha,
        )
