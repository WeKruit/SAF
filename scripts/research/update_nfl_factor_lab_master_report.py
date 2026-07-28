#!/usr/bin/env python3
"""Attach the exact-153 report entry point to the single Master Notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "nfl-factor-lab"
    / "NFL_Factor_Lab_Master.ipynb"
)
SECTION_ID = "nfl_x13_exact153_report_v1"


def _markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {"saf_section_id": SECTION_ID},
        "source": source.splitlines(keepends=True),
    }


def _code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"saf_section_id": SECTION_ID},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("saf_section_id") != SECTION_ID
    ]
    cells.extend(
        [
            _markdown_cell(
                """## 19. Exact-153 中文总报告与当前研究 State

这里不重新计算 153 场，也不读取 81 场 final holdout。它只核对并展示由
Section 18 同一份 exact-153 discovery object 生成的离线报告、状态和 hash。

颜色口径统一为：**蓝色上升、红色下降**。报告中的 PASS 仅代表样本支持门，
所有 113 个机械候选仍是 `NEEDS_HUMAN_REVIEW`，不是 signal 或 alpha。
"""
            ),
            _code_cell(
                """exact153_report_dir = ROOT / "reports/nfl-factor-lab/exact-153"
exact153_report_html = exact153_report_dir / "NFL_X13_153_GAME_REPORT.html"
exact153_report_markdown = exact153_report_dir / "NFL_X13_153_GAME_REPORT_ZH.md"
exact153_report_state_path = exact153_report_dir / "NFL_X13_153_GAME_STATE.json"
exact153_report_manifest_path = exact153_report_dir / "manifest.json"

exact153_report_state = json.loads(
    exact153_report_state_path.read_text(encoding="utf-8")
)
exact153_report_manifest = json.loads(
    exact153_report_manifest_path.read_text(encoding="utf-8")
)
if exact153_report_state["game_count"] != 153:
    raise ValueError("Exact-153 report state no longer binds 153 games")
if exact153_report_state["holdout_access"] != "CLOSED":
    raise ValueError("Final holdout must remain CLOSED")
for descriptor in exact153_report_manifest["outputs"].values():
    output_path = exact153_report_dir / descriptor["path"]
    observed = "sha256:" + hashlib.sha256(output_path.read_bytes()).hexdigest()
    if observed != descriptor["sha256"]:
        raise ValueError(f"Report output hash drift: {output_path}")

exact153_report_state_df = pd.DataFrame(
    [
        {"metric": key, "value": value}
        for key, value in exact153_report_state.items()
        if key
        in {
            "game_count",
            "factor_count",
            "factor_horizon_rows",
            "actual_market_observations",
            "reaction_paths",
            "factor_observations",
            "support_gate_pass_count",
            "mechanical_candidate_count",
            "mechanical_candidate_factor_count",
            "promotion_status",
            "holdout_access",
        }
    ]
)
exact153_report_files_df = pd.DataFrame(
    [
        {
            "format": name,
            "absolute_path": str(exact153_report_dir / descriptor["path"]),
            "sha256": descriptor["sha256"],
            "byte_length": descriptor["byte_length"],
        }
        for name, descriptor in exact153_report_manifest["outputs"].items()
    ]
)
display(exact153_report_state_df, exact153_report_files_df)
display(
    HTML(
        "<p><strong>打开完整彩色报告：</strong> "
        f"<a href='file://{exact153_report_html}' target='_blank'>"
        "NFL_X13_153_GAME_REPORT.html</a></p>"
    )
)
"""
            ),
        ]
    )
    notebook["cells"] = cells

    # Keep the notebook heatmap direction consistent with the audited report.
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            'endpoint = (178, 24, 43) if float(value) > 0 else (33, 102, 172)',
            'endpoint = (69, 117, 180) if float(value) > 0 else (215, 48, 39)',
        )
        source = source.replace(
            "红/蓝仅表示方向和幅度，不表示可交易性",
            "蓝升/红降仅表示方向和幅度，不表示可交易性",
        )
        source = source.replace(
            "from IPython.display import SVG, display",
            "from IPython.display import HTML, SVG, display",
        )
        source = source.replace(
            "if HORIZON_S not in {1, 2, 5, 10, 30, 60}:",
            "if HORIZON_S not in {1, 2, 5, 10, 20, 30, 60}:",
        )
        source = source.replace(
            '"holdout_reaction_access": "COMPLETED_AFTER_SHORTLIST_LOCK"',
            '"holdout_reaction_access": "CLOSED_ZERO_READ"',
        )
        cell["source"] = source.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            "The frozen shortlist and historical holdout have now completed; "
            "there is still no model training.",
            "The shortlist is not frozen and the 81-game final holdout remains "
            "CLOSED; there is still no model training.",
        )
        cell["source"] = source.splitlines(keepends=True)

    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK)
    print(f"cells={len(notebook['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
