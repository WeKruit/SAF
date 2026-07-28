#!/usr/bin/env python3
"""Execute only the report-facing cells of the NFL Factor Lab notebook.

The expensive upstream pipeline is intentionally not rerun.  This script
executes the shared setup plus Sections 17–19 in a clean kernel, then copies
their rendered outputs back into the authoritative Master Notebook.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "nfl-factor-lab"
    / "NFL_Factor_Lab_Master.ipynb"
)
REPORT_SECTION_PREFIXES = (
    "## 17.",
    "## 18.",
    "## 19.",
)


def _section_code_indices(notebook: nbformat.NotebookNode) -> list[int]:
    section_indices: list[int] = []
    active = False
    for index, cell in enumerate(notebook.cells):
        source = cell.source.lstrip()
        if cell.cell_type == "markdown" and source.startswith(
            REPORT_SECTION_PREFIXES
        ):
            active = True
            continue
        if cell.cell_type == "markdown" and source.startswith("## "):
            active = False
        if active and cell.cell_type == "code":
            section_indices.append(index)
    return section_indices


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    report_indices = _section_code_indices(notebook)
    if not report_indices:
        raise ValueError("No Section 17–19 code cells found")

    # The first two code cells contain the canonical imports and ROOT lookup.
    setup_indices = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    ][:2]
    selected_indices = setup_indices + report_indices
    mini = nbformat.v4.new_notebook(
        cells=[deepcopy(notebook.cells[index]) for index in selected_indices],
        metadata=deepcopy(notebook.metadata),
    )
    executed = NotebookClient(
        mini,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    for mini_index, source_index in enumerate(selected_indices):
        if source_index not in report_indices:
            continue
        notebook.cells[source_index].execution_count = (
            executed.cells[mini_index].execution_count
        )
        notebook.cells[source_index].outputs = deepcopy(
            executed.cells[mini_index].outputs
        )

    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)
    print(f"executed_report_cells={len(report_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
