from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import pytest

import prediction_market.workbench.factor_lab_master as master
from prediction_market.workbench.factor_lab_master import (
    ArtifactVerificationError,
    data_gap_frame,
    frame_audit,
    read_parquet_slice,
    read_v2_sports_source_slice,
    resolve_artifact_ref,
)


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT / "notebooks" / "nfl-factor-lab" / "NFL_Factor_Lab_Master.ipynb"
)

SECTION_TITLES = (
    "Claim boundary and parameters",
    "Run, S3, and stage manifests",
    "Raw sports DataFrames",
    "Sports cleaning reconciliation",
    "Point-in-time rolling features",
    "Raw and clean market DataFrames",
    "G/M source-time intervals",
    "Unified source timeline",
    "Association attrition and reaction paths",
    "Factor registry and coverage",
    "Full A–G factor universe",
    "Row-level factor construction",
    "Single-game results and cases",
    "20-game aggregation",
    "Method research mapping",
    "Expert review and shortlist gate",
    "Export and progress",
)


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict[str, object]) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return "\n".join(
        _source(cell)
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


def test_master_notebook_preserves_core_sections_and_registered_extensions() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    headings = [
        match.groups()
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        for match in [re.match(r"^## (\d+)\. (.+)$", _source(cell).splitlines()[0])]
        if match is not None
    ]
    assert headings[:17] == [
        (str(number), title)
        for number, title in enumerate(SECTION_TITLES)
    ]
    assert [number for number, _title in headings[17:]] == ["17", "18", "19", "20"]
    assert not any(
        "drilldown" in _source(cell).lower()
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def test_master_notebook_has_one_explicit_parameter_cell() -> None:
    notebook = _load_notebook()
    parameter_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "parameters" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(parameter_cells) == 1
    parameter_code = _source(parameter_cells[0])
    assert parameter_cells[0]["id"] == "parameters"
    expected_defaults = {
        "MODE": '"single_game_audit"',
        "GAME_ID": '"2025_14_DAL_DET"',
        "FACTOR_ID": '"NFL.EVENT.EXPLOSIVE_PLAY"',
        "MARKET_FAMILY": '"moneyline"',
        "VENUE": '"all"',
        "DELAY_S": "0",
        "HORIZON_S": "10",
    }
    for name, value in expected_defaults.items():
        assert re.search(rf"^{name}\s*=\s*{re.escape(value)}$", parameter_code, re.M)


def test_master_notebook_is_a_transparent_sequential_dataframe_workflow() -> None:
    notebook = _load_notebook()
    cells_by_id = {
        str(cell.get("id")): _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    required_stage_frames = {
        "manifest-stage": ("run_manifest_df", "s3_manifest_df", "stage_manifest_df"),
        "raw-sports-stage": ("raw_sports_df", "raw_sports_audit_df"),
        "sports-clean-stage": (
            "sports_clean_df",
            "sports_reconciliation_df",
            "sports_clean_audit_df",
        ),
        "pit-rolling-stage": ("pit_rolling_df", "pit_rolling_audit_df"),
        "market-stage": (
            "raw_market_df",
            "clean_market_df",
            "market_semantics_df",
        ),
        "interval-stage": ("g_intervals_df", "m_intervals_df"),
        "timeline-stage": ("source_timeline_df", "receive_time_gap_df"),
        "association-stage": ("association_attrition_df", "reaction_paths_df"),
        "registry-stage": ("factor_registry_df", "factor_coverage_df"),
        "universe-stage": ("factor_universe_df", "factor_data_gaps_df"),
        "row-factor-stage": ("row_factor_df", "row_factor_audit_df"),
        "single-game-stage": ("single_game_results_df", "single_game_cases_df"),
        "aggregate-stage": ("twenty_game_results_df", "per_game_effects_df"),
        "methods-stage": ("method_cards_df", "method_research_mapping_df"),
        "review-stage": ("expert_review_df", "shortlist_gate_df"),
        "export-stage": ("export_manifest_df", "progress_df"),
    }
    for cell_id, frame_names in required_stage_frames.items():
        assert cell_id in cells_by_id
        for frame_name in frame_names:
            assert frame_name in cells_by_id[cell_id]
        assert "frame_audit(" in cells_by_id[cell_id] or cell_id == "manifest-stage"
        assert "display(" in cells_by_id[cell_id]


def test_master_notebook_is_offline_and_keeps_holdout_sealed_unread() -> None:
    notebook = _load_notebook()
    governance = notebook["metadata"]["saf_factor_lab"]
    assert governance == {
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "experiment_id": "X-13",
        "holdout_reaction_access": "SEALED_UNREAD",
        "network_access": "forbidden",
        "official_computation": "verified_stage_functions_or_publications_only",
        "schema": "NFLFactorLabMasterNotebookV2",
    }
    code = _code(notebook)
    ast.parse(code)
    for forbidden in (
        "requests.",
        "httpx.",
        "urlopen(",
        "read_html(",
        "run_all(",
        "run_factor_discovery(",
        "run_locked_historical_holdout(",
        "freeze_factor_shortlist(",
        "fit(",
        "train(",
    ):
        assert forbidden not in code
    assert "MODE == \"twenty_game_published\"" in code
    assert "association_candidate_table" not in code
    assert "read_verified_publications_only" in code
    assert "SHORTLIST_NOT_FROZEN_HOLDOUT_SEALED" in code
    assert '"holdout_reaction_accessed": False' in code
    for forbidden in (
        "historical-holdout-results",
        "holdout_results_v1.parquet",
        "holdout_factor_observations_v1.parquet",
        "holdout_results_df",
        "holdout_observations_df",
    ):
        assert forbidden not in code
    assert not any(
        output.get("output_type") == "error"
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    )


def test_master_notebook_states_market_time_and_attrition_truth() -> None:
    notebook = _load_notebook()
    prose = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for required in (
        "actual trades",
        "never forward-filled",
        "source time",
        "receive time",
        "G0",
        "G3",
        "M0",
        "M3",
        "SEALED_UNREAD",
        "no model training",
        "grain",
        "missingness",
        "SHA-256",
    ):
        assert required in prose
    for superseded in ("333,340", "39,873", "34,374"):
        assert superseded not in prose


def test_master_section_eight_uses_verified_batch_attrition_summary() -> None:
    notebook = _load_notebook()
    imports = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("id") == "imports"
    )
    association = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("id") == "association-stage"
    )

    assert "load_v2_batch_attrition_summary" in imports
    assert (
        "v2_batch_attrition = "
        "load_v2_batch_attrition_summary(notebook_run)"
    ) in association
    assert "association_attrition_df = v2_batch_attrition.funnel" in association
    assert (
        "association_exclusion_reasons_df = v2_batch_attrition.reasons"
        in association
    )
    assert 'read_bundle_stage(\n        single_v2_bundle,\n        "reaction_paths"' in association
    for superseded in (
        "prior_attrition_baseline_df",
        "coverage_payload",
        'read_bundle_stage(single_v2_bundle, "association_attrition"',
    ):
        assert superseded not in association


def test_master_notebook_maps_all_required_research_methods_from_cards() -> None:
    notebook = _load_notebook()
    code = _code(notebook)
    prose = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for required in (
        "nflfastR",
        "fastrmodels",
        "xpass",
        "CP/CPOE",
        "XYAC",
        "nfl4th",
        "Expected Field Position",
        "Big Data Bowl",
        "Kaggle",
    ):
        assert required in prose
    assert "method_catalog" in code
    assert "README.md" not in code
    assert "ADOPTED" in code
    assert "REFERENCE_ONLY" in code
    assert "DATA_GAP" in code


def test_master_notebook_uses_fast_v2_commit_verification_by_default() -> None:
    notebook = _load_notebook()
    code = _code(notebook)
    assert "verify_factor_lab_notebook_run" in code
    assert "SAF_FACTOR_LAB_DEEP_VERIFY=1" in code
    assert "notebook_run.verification_mode" in code
    assert "notebook_run.stage_content_verification" in code
    assert "notebook_run.latest_deep_batch_status" in code
    assert "notebook_run.s3_status" in code
    assert "CONFIGURED_NOT_REMOTE_VERIFIED" in code
    assert "verify_factor_lab_run_index(" not in code
    assert "verify_single_game_bundle(" not in code
    assert "verify_cross_game_bundle(" not in code
    assert "read_bundle_stage" in code
    for source_key in (
        "factor_registry_v1",
        "method_catalog_v1",
        "x13_signal_manifest_v1",
        "x13_correlation_cache_manifest_v1",
        "x13_reaction_paths_v1",
        "x13_prior_sports_feature_view_manifest_v1",
        "x13_prior_discovery_manifest_v1",
        "x13_batch_manifest_v1",
        "nflverse_2022_manifest",
        "nflverse_2023_manifest",
        "nflverse_2024_manifest",
        "nflverse_2025_manifest",
    ):
        assert f'"{source_key}"' in code
    assert "required_v2_source_keys <= observed_v2_source_keys" in code
    assert "observed_v2_source_keys != required_v2_source_keys" not in code
    batch_attrition_loader = inspect.getsource(
        master.load_v2_batch_attrition_summary
    )
    for stage in (
        "sports_row_disposition",
        "canonical_events",
        "finalized_episodes",
        "stat_ledger",
        "pit_features",
        "contract_inventory",
        "market_observations",
        "market_cleaning_audit",
        "game_time_intervals",
        "market_time_intervals",
        "source_timeline",
        "timeline_audit",
        "association_attrition",
        "reaction_paths",
        "factor_observations",
        "factor_exclusion_audit",
        "single_game_factor_results",
        "cross_game_factor_results",
        "game_contribution",
        "leave_one_game_out",
        "venue_consistency",
        "expert_review_queue",
    ):
        source = (
            batch_attrition_loader
            if stage in {"association_attrition", "reaction_paths"}
            else code
        )
        assert f'"{stage}"' in source
    assert "SHALLOW_COMMIT_VERIFIED" in code
    assert "NOT_RECORDED_IN_RUN_INDEX" in code
    assert "single_v2_bundle = notebook_run.single_game_bundle(GAME_ID)" in code
    assert "cross_v2_bundle = notebook_run.cross_game_bundle" in code


def test_master_notebook_does_not_scan_seasonal_pbp_when_opened() -> None:
    notebook = _load_notebook()
    cells_by_id = {
        str(cell.get("id")): _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    imports = cells_by_id["imports"]
    raw_sports = cells_by_id["raw-sports-stage"]
    assert "verify_and_load_x13_inputs" not in imports
    assert "NFL_FACTOR_LAB_PBP_COLUMNS" not in imports
    assert "pyarrow.parquet" not in imports
    assert "verify_and_load_x13_inputs" not in raw_sports
    assert "pq.read_table" not in raw_sports
    assert "read_v2_sports_source_slice" in raw_sports
    assert "single_v2_bundle is not None" in raw_sports
    assert "V2 single-game bundle is unavailable" in raw_sports


def test_master_notebook_reads_formal_factor_stages_without_recomputing_predicates() -> None:
    notebook = _load_notebook()
    row_factor = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("id") == "row-factor-stage"
    )
    for forbidden in (
        "def evaluate_clause",
        "predicate_mask",
        "exclusion_mask",
        'definition["predicate"]',
        'definition["exclusions"]',
        "factor_active",
    ):
        assert forbidden not in row_factor
    assert 'read_bundle_stage(single_v2_bundle, "factor_observations"' in row_factor
    assert 'read_bundle_stage(single_v2_bundle, "factor_exclusion_audit"' in row_factor
    assert "factor_flow_df" in row_factor
    assert "candidate_observation_count" in row_factor
    assert "excluded_observation_count" in row_factor
    assert "formal_observation_count" in row_factor
    assert 'factor_exclusion_audit_df["decision"].eq("EXCLUDE")' in row_factor
    assert 'factor_exclusion_audit_df["decision"].eq("INCLUDE")' in row_factor
    assert "formal V2 factor stages disagree on included row count" in row_factor
    assert "V2 bundle unavailable; row-level factor construction is DATA_GAP" in row_factor


def test_single_game_case_section_does_not_reconstruct_cases_from_v1() -> None:
    notebook = _load_notebook()
    single_game = next(
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("id") == "single-game-stage"
    )
    assert "active_episode_ids" not in single_game
    assert "per_game_effects_json" not in single_game
    assert "selected_cases_df" not in single_game
    assert "single-game V2 factor stages are unavailable" in single_game
    assert "PRIOR_EVIDENCE_V1" in single_game


def test_v2_sports_source_slice_uses_only_bounded_verified_bundle_stages(
    tmp_path: Path,
) -> None:
    dispositions = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "raw_play_id": "10",
                "canonical_play_id": "10",
                "disposition": "factor_eligible",
                "source_hash": "sha256:" + "1" * 64,
            },
            {
                "game_id": "g1",
                "raw_play_id": "20",
                "canonical_play_id": None,
                "disposition": "administrative_only",
                "source_hash": "sha256:" + "2" * 64,
            },
        ]
    )
    canonical = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "play_id": "10",
                "source_time_utc": "2025-01-01T00:00:00Z",
                "description": "pass complete",
                "posteam": "A",
                "defteam": "B",
            }
        ]
    )
    factor_observations = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "play_id": "10",
                "episode_id": "e1",
                "factor_id": "NFL.EVENT.TEST",
                "logical_market_id": "m1",
                "outcome": "A",
                "venue": "polymarket",
                "market_family": "moneyline",
                "horizon_seconds": 10,
            }
        ]
    )
    factor_exclusion_audit = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "audit_id": "a1",
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "venue": "polymarket",
                "decision": "INCLUDE",
            },
            {
                "game_id": "g1",
                "audit_id": "a2",
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "venue": "polymarket",
                "decision": "EXCLUDE",
            },
        ]
    )
    stage_paths: dict[str, Path] = {}
    stages: dict[str, object] = {}
    for name, frame, grain in (
        (
            "sports_row_disposition",
            dispositions,
            ["game_id", "raw_play_id"],
        ),
        ("canonical_events", canonical, ["game_id", "play_id"]),
        (
            "factor_observations",
            factor_observations,
            [
                "factor_id",
                "game_id",
                "episode_id",
                "logical_market_id",
                "outcome",
                "venue",
                "horizon_seconds",
            ],
        ),
        (
            "factor_exclusion_audit",
            factor_exclusion_audit,
            [
                "factor_id",
                "game_id",
                "episode_id",
                "logical_market_id",
                "outcome",
                "venue",
                "horizon_seconds",
            ],
        ),
    ):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        stage_paths[name] = path
        stages[name] = {
            "path": path.name,
            "sha256": digest,
            "byte_length": path.stat().st_size,
            "row_count": len(frame),
            "schema_fingerprint": "sha256:" + "3" * 64,
            "grain": grain,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"stages": stages}), encoding="utf-8")
    bundle = SimpleNamespace(
        manifest_path=manifest_path,
        stage_paths=stage_paths,
        bundle_sha256="sha256:" + "4" * 64,
    )

    loaded = read_v2_sports_source_slice(bundle, limit=10)

    observed = loaded[
        [
            "game_id",
            "raw_play_id",
            "canonical_play_id",
            "description",
            "disposition",
        ]
    ].to_dict(orient="records")
    assert observed[0] == {
        "game_id": "g1",
        "raw_play_id": "10",
        "canonical_play_id": "10",
        "description": "pass complete",
        "disposition": "factor_eligible",
    }
    assert observed[1]["game_id"] == "g1"
    assert observed[1]["raw_play_id"] == "20"
    assert pd.isna(observed[1]["canonical_play_id"])
    assert pd.isna(observed[1]["description"])
    assert observed[1]["disposition"] == "administrative_only"
    assert loaded.attrs["status"] == "OFFICIAL_V2_VERIFIED"
    assert loaded.attrs["loaded_row_limit"] == 10
    assert loaded.attrs["source_row_count"] == 2

    cells_by_id = {
        str(cell.get("id")): _source(cell)
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "code"
    }
    raw_namespace = {
        "MODE": "single_game_audit",
        "single_v2_bundle": bundle,
        "read_v2_sports_source_slice": read_v2_sports_source_slice,
        "data_gap_frame": data_gap_frame,
        "frame_audit": frame_audit,
        "display": lambda *_args: None,
    }
    exec(cells_by_id["raw-sports-stage"], raw_namespace)
    assert len(raw_namespace["raw_sports_df"]) == 2

    factor_namespace = {
        "MODE": "single_game_audit",
        "single_v2_bundle": bundle,
        "FACTOR_ID": "NFL.EVENT.TEST",
        "MARKET_FAMILY": "moneyline",
        "HORIZON_S": 10,
        "VENUE": "polymarket",
        "read_bundle_stage": __import__(
            "prediction_market.workbench.factor_lab_master",
            fromlist=["read_bundle_stage"],
        ).read_bundle_stage,
        "data_gap_frame": data_gap_frame,
        "frame_audit": frame_audit,
        "display": lambda *_args: None,
        "pd": pd,
    }
    exec(cells_by_id["row-factor-stage"], factor_namespace)
    assert factor_namespace["factor_flow_df"].iloc[0].to_dict() == {
        "factor_id": "NFL.EVENT.TEST",
        "market_family": "moneyline",
        "horizon_seconds": 10,
        "candidate_observation_count": 2,
        "excluded_observation_count": 1,
        "formal_observation_count": 1,
        "flow_status": "OFFICIAL_V2_STAGE_QUERY",
    }


def test_artifact_loader_verifies_hash_and_reads_only_a_bounded_slice(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rows.parquet"
    pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g2"],
            "value": [1, 2, 3, 4],
            "unused": ["a", "b", "c", "d"],
        }
    ).to_parquet(source, index=False)
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "table": {
            "path": source.name,
            "sha256": digest,
            "byte_length": source.stat().st_size,
        }
    }
    reference = resolve_artifact_ref(
        manifest,
        "table",
        project_root=tmp_path,
    )
    loaded = read_parquet_slice(
        reference,
        filters={"game_id": "g1"},
        columns=["game_id", "value"],
        limit=2,
    )
    assert loaded.to_dict(orient="records") == [
        {"game_id": "g1", "value": 1},
        {"game_id": "g1", "value": 2},
    ]
    assert loaded.attrs["source_sha256"] == digest
    assert loaded.attrs["source_row_count"] == 4
    assert loaded.attrs["loaded_row_limit"] == 2


def test_artifact_loader_fails_closed_on_escape_or_tampering(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.parquet"
    outside.write_bytes(b"not parquet")
    with pytest.raises(ArtifactVerificationError, match="inside project root"):
        resolve_artifact_ref(
            {
                "table": {
                    "path": "../outside.parquet",
                    "sha256": "sha256:" + "0" * 64,
                    "byte_length": 11,
                }
            },
            "table",
            project_root=tmp_path,
        )

    source = tmp_path / "rows.parquet"
    source.write_bytes(b"tampered")
    with pytest.raises(ArtifactVerificationError, match="SHA-256"):
        resolve_artifact_ref(
            {
                "table": {
                    "path": "rows.parquet",
                    "sha256": "sha256:" + "0" * 64,
                    "byte_length": len(b"tampered"),
                }
            },
            "table",
            project_root=tmp_path,
        )


def test_gap_and_audit_frames_never_invent_data_rows() -> None:
    gap = data_gap_frame(
        stage="raw_market",
        grain="one native market record",
        reason="stage manifest has no raw-market table",
        expected_columns=("venue", "native_id"),
    )
    assert gap.empty
    assert list(gap) == ["venue", "native_id"]
    assert gap.attrs["status"] == "DATA_GAP"
    assert gap.attrs["reason"] == "stage manifest has no raw-market table"

    audit = frame_audit(gap, name="raw_market_df", grain="one native market record")
    assert audit.loc[0, "row_count"] == 0
    assert audit.loc[0, "status"] == "DATA_GAP"
    assert audit.loc[0, "reason"] == "stage manifest has no raw-market table"
