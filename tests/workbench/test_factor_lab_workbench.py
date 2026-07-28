from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = ROOT / "notebooks" / "nfl-factor-lab"
DRILLDOWN_ROOT = NOTEBOOK_ROOT / "drilldowns"
REPORT_ROOT = ROOT / "reports" / "nfl-factor-lab"
NOTEBOOK_STEMS = (
    "00_data_dictionary",
    "01_twenty_game_coverage",
    "02_replay_and_time_audit",
    "03_play_call_factors",
    "04_turnover_factors",
    "05_scoring_factors",
    "06_field_position_special_teams",
    "07_penalty_review_timeout",
    "08_team_player_rolling_strength",
    "09_market_reaction",
    "10_combination_explorer",
    "11_human_review_queue",
    "12_candidate_decision_and_holdout",
)
FORMAL_MODULE = "prediction_market.sports.nfl_factor_lab"
FORBIDDEN_CODE = (
    "requests.",
    "urllib.",
    "http://",
    "https://",
    "read_html(",
    "read_csv(",
    "read_parquet(",
    "to_csv(",
    "to_parquet(",
    "bootstrap",
    "multipletests",
)


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_all_thirteen_notebooks_are_thin_clean_kernel_clients() -> None:
    paths = sorted(
        path
        for path in DRILLDOWN_ROOT.glob("*.ipynb")
        if path.stem in NOTEBOOK_STEMS
    )
    assert [path.stem for path in paths] == list(NOTEBOOK_STEMS)

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        governance = notebook["metadata"]["saf_factor_lab"]
        assert governance["experiment_id"] == "X-13"
        assert governance["claim_boundary"] == "PRELIMINARY_SOURCE_TIME_ONLY"
        assert governance["network_access"] == "forbidden"
        assert governance["official_computation"] == "formal_python_interfaces_only"

        cells = notebook["cells"]
        assert cells[0]["cell_type"] == "markdown"
        assert path.stem in _source(cells[0])
        code = "\n".join(
            _source(cell) for cell in cells if cell["cell_type"] == "code"
        )
        assert FORMAL_MODULE in code
        assert "verify_and_load_x13_inputs" in code
        assert "claim_boundary" in code
        assert not any(token in code for token in FORBIDDEN_CODE)
        assert len(cells) <= 8
        for cell in cells:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_holdout_notebook_does_not_execute_before_a_shortlist_lock() -> None:
    notebook = json.loads(
        (DRILLDOWN_ROOT / "12_candidate_decision_and_holdout.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "freeze_factor_shortlist" in code
    assert "run_locked_historical_holdout" in code
    assert "if shortlist_lock is not None:" in code


def test_unavailable_feature_and_discovery_batches_are_guarded() -> None:
    for stem in (
        "03_play_call_factors",
        "04_turnover_factors",
        "05_scoring_factors",
        "06_field_position_special_teams",
        "07_penalty_review_timeout",
        "08_team_player_rolling_strength",
    ):
        notebook = json.loads(
            (DRILLDOWN_ROOT / f"{stem}.ipynb").read_text(encoding="utf-8")
        )
        code = "\n".join(
            _source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "verified_row_inputs = None" in code
        assert "if verified_row_inputs is not None:" in code

    for stem in (
        "09_market_reaction",
        "10_combination_explorer",
        "11_human_review_queue",
    ):
        notebook = json.loads(
            (DRILLDOWN_ROOT / f"{stem}.ipynb").read_text(encoding="utf-8")
        )
        code = "\n".join(
            _source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "feature_view = None" in code
        assert "if feature_view is not None:" in code


def test_quarto_project_is_parameterized_self_contained_and_offline() -> None:
    config = yaml.safe_load(
        (REPORT_ROOT / "_quarto.yml").read_text(encoding="utf-8")
    )
    html = config["format"]["html"]
    assert html["embed-resources"] is True
    assert html["self-contained-math"] is True
    assert html["toc"] is True
    assert config["execute"]["freeze"] is True

    report = (REPORT_ROOT / "factor-lab.qmd").read_text(encoding="utf-8")
    assert "#| tags: [parameters]" in report
    for parameter in (
        "game_id",
        "factor_id",
        "market_family",
        "venue",
        "horizon",
        "review_status",
    ):
        assert f"{parameter} =" in report
        assert f'"{parameter}": {parameter}' in report
    assert "prediction_market.workbench.factor_lab_master" in report
    assert "verify_factor_lab_notebook_run" in report
    assert "load_v2_batch_attrition_summary" in report
    assert "read_bundle_stage" in report
    assert "nfl_factor_lab_run_index_v2.json" in report
    assert "NFL_Factor_Lab_Master.ipynb" in report
    assert "NFLFactorLabRunIndexV1" not in report
    assert "nfl_factor_lab_run_index_v1.json" not in report
    assert "verify_and_load_x13_inputs" not in report
    assert '"association_attrition"' not in report
    assert "http://" not in report
    assert "https://" not in report


def test_render_matrix_declares_every_required_report_family() -> None:
    matrix = yaml.safe_load(
        (REPORT_ROOT / "render-matrix.yml").read_text(encoding="utf-8")
    )
    assert matrix["experiment_id"] == "X-13"
    assert matrix["claim_boundary"] == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert matrix["game_reports"]["source"] == "frozen_discovery_games"
    assert matrix["game_reports"]["expected_count"] == 20
    assert set(matrix["report_families"]) == {
        "shortlisted_factor",
        "discovery_summary",
        "holdout_summary",
        "method_catalog",
        "data_gap",
    }
    assert matrix["holdout_summary"]["gate"] == "shortlist_lock_required"


def test_quarto_render_status_explicitly_blocks_stale_notebook_output() -> None:
    status = json.loads(
        (REPORT_ROOT / "render-status.json").read_text(encoding="utf-8")
    )
    assert status["schema"] == "NFLFactorLabQuartoRenderStatusV2"
    assert status["claim_boundary"] == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert status["holdout_reaction_accessed"] is False
    assert status["render_status"] == "STALE_BLOCKED_NOTEBOOK_CHANGED"
    assert status["source_validation"] == "STALE_BLOCKED_NOTEBOOK_CHANGED"
    assert status["block_reason"] == (
        "QUARTO_RERENDER_NOT_PERFORMED_FOR_CURRENT_NOTEBOOK"
    )
    assert status["quarto_version"] == "1.9.38"
    assert status["quarto_archive_sha256"] == (
        "sha256:"
        "47089a5020cfb41981ba0d4b46e110edfa608722aea45ef248e14efba6d6b18a"
    )
    assert status["authority"]["notebook"]["path"] == (
        "notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb"
    )
    assert status["authority"]["run_index"]["path"] == (
        "registries/factors/nfl_factor_lab_run_index_v2.json"
    )
    assert status["authority"]["cross_game_bundle_sha256"] == (
        "sha256:"
        "ac3af1adbc3dabe73e4b8d2a3a9d6bd19639e23108dc419453ea08e1ec38dccb"
    )
    assert status["authority"]["single_game_bundle_count"] == 20
    assert status["external_runtime_resources"] == []
    assert status["rendered_outputs"]

    compact_execution = status["compact_clean_kernel_execution"]
    assert compact_execution["source_code_cells"] == 10
    assert compact_execution["executed_code_cells"] == 10
    assert compact_execution["failed_code_cells"] == 0
    assert compact_execution["source_sha256"] == (
        status["source_files"]["factor-lab.qmd"]["sha256"]
    )
    assert compact_execution["status"] == "PASS"

    master_execution = status["master_clean_kernel_execution"]
    assert master_execution["source_code_cells"] == 19
    assert master_execution["executed_code_cells"] == 19
    assert master_execution["failed_code_cells"] == 0
    assert master_execution["source_sha256"] == (
        status["authority"]["notebook"]["sha256"]
    )
    assert master_execution["status"] == "STALE_LAST_SUCCESSFUL_EXECUTION"

    rendered_html: dict[str, str] = {}
    for descriptor in status["rendered_outputs"]:
        assert descriptor["status"] == "STALE_LAST_SUCCESSFUL_RENDER"
        output = REPORT_ROOT / descriptor["path"]
        rendered = output.read_bytes()
        assert len(rendered) == descriptor["byte_length"]
        assert (
            "sha256:" + hashlib.sha256(rendered).hexdigest()
            == descriptor["sha256"]
        )
        html = rendered.decode("utf-8")
        rendered_html[descriptor["path"]] = html
        assert "2025_14_DAL_DET" in html
        assert "PRELIMINARY_SOURCE_TIME_ONLY" in html
        assert re.search(
            r"""<(?:script|link|img|iframe|video|audio|source)\b[^>]*"""
            r"""(?:src|href)=["'](?:https?:)?//""",
            html,
            flags=re.IGNORECASE,
        ) is None

    compact = rendered_html["_rendered/factor-lab.html"]
    assert (
        "ac3af1adbc3dabe73e4b8d2a3a9d6bd19639e23108dc419453ea08e1ec38dccb"
        in compact
    )
    assert "NFLFactorLabRunIndexV2" in compact
    assert "13 个旧 drilldown" not in compact
    assert "_rendered/NFL_Factor_Lab_Master.html" in rendered_html


def test_quarto_render_status_distinguishes_current_from_last_rendered_notebook() -> None:
    status = json.loads(
        (REPORT_ROOT / "render-status.json").read_text(encoding="utf-8")
    )
    sources = (
        (
            REPORT_ROOT / "factor-lab.qmd",
            status["source_files"]["factor-lab.qmd"],
        ),
        (
            NOTEBOOK_ROOT / "NFL_Factor_Lab_Master.ipynb",
            status["current_notebook"],
        ),
    )

    for source, descriptor in sources:
        payload = source.read_bytes()
        assert descriptor["byte_length"] == len(payload)
        assert descriptor["sha256"] == (
            "sha256:" + hashlib.sha256(payload).hexdigest()
        )
    assert status["current_notebook"]["status"] == "CURRENT_SOURCE_UNRENDERED"
    assert status["authority"]["notebook"]["status"] == "LAST_RENDERED_SOURCE"
    assert (
        status["current_notebook"]["sha256"]
        != status["authority"]["notebook"]["sha256"]
    )


def test_formal_rendered_reports_show_the_verified_twenty_game_batch_funnel() -> None:
    expected_funnel = (
        ("association cells", 48_624_912),
        ("delay=0 projection", 8_104_152),
        ("actual reaction paths", 64_058),
        ("strict analysis eligible", 2_518),
    )
    funnel_pattern = r".*?".join(
        rf"<td>{re.escape(stage)}</td>\s*<td>{rows}</td>"
        for stage, rows in expected_funnel
    )

    for relative_path in (
        "_rendered/factor-lab.html",
        "_rendered/NFL_Factor_Lab_Master.html",
    ):
        html = (REPORT_ROOT / relative_path).read_text(encoding="utf-8")
        assert re.search(funnel_pattern, html, flags=re.DOTALL)
        assert re.search(
            r"<td>association cells</td>\s*<td>770832</td>",
            html,
        ) is None
