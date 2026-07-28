#!/usr/bin/env python3
"""Seal holdout access and attach nine verified workbench entry zones."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb"
SECTION_ID = "nfl_x13_research_workbench_v2"
REPLACED_SECTION_IDS = {
    "nfl_x13_historical_research_workbench_v1",
    SECTION_ID,
}
ZONES = (
    (
        "Tick Inventory",
        "ticks",
        "Verified tick inventory；没有独立 artifact 时显示 NOT_AVAILABLE。",
    ),
    (
        "Historical Distribution",
        "historical",
        "Verified development-only empirical distribution and histogram rows。",
    ),
    (
        "Reference Completion",
        "reference",
        "Retrospective reference completion；global exclusions 按 support status 单列。",
    ),
    (
        "Calibration/Bias",
        "calibration",
        "Verified factor-bound calibration or bias rows；缺失时不推断。",
    ),
    (
        "Poly×Kalshi paired",
        "paired",
        "只展示 SAME_EPISODE_PAIRED 和 venue_consistency。",
    ),
    (
        "PMXT L2 method audit",
        "l2",
        "只允许 METHOD_VALIDATION_ONLY_NON_NFL_ALPHA 边界；无 L2 时阻断。",
    ),
    (
        "Shortlist/Holdout",
        "shortlist",
        "Shortlist 未冻结；final holdout 只显示 SEALED_UNREAD 元数据。",
    ),
    (
        "Shadow metrics",
        "shadow",
        "Verified factor-bound shadow rows；不构成执行或交易结论。",
    ),
    (
        "Audit",
        "audit",
        "Global governance、manifest/object descriptor 与对象字节 SHA-256。",
    ),
)


def _metadata(zone: str, anchor: str) -> dict[str, str]:
    return {
        "saf_section_id": SECTION_ID,
        "workbench_zone": zone,
        "workbench_anchor": anchor,
    }


def _markdown_cell(
    source: str,
    *,
    zone: str,
    anchor: str,
) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": _metadata(zone, anchor),
        "source": source.splitlines(keepends=True),
    }


def _code_cell(
    source: str,
    *,
    zone: str,
    anchor: str,
) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": _metadata(zone, anchor),
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _cell_by_id(notebook: dict[str, object], cell_id: str) -> dict[str, object]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    matches = [cell for cell in cells if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one notebook cell id={cell_id}")
    return matches[0]


def _seal_holdout(notebook: dict[str, object]) -> None:
    metadata = notebook.setdefault("metadata", {}).setdefault(
        "saf_factor_lab", {}
    )
    metadata["holdout_reaction_access"] = "SEALED_UNREAD"

    section = _cell_by_id(notebook, "section-15")
    section["source"] = (
        """## 15. Expert review and shortlist gate

Statistical discovery exposes the expert review queue, but no candidate has
completed the full freeze gate. The shortlist is not frozen and final holdout
reaction data remains `SEALED_UNREAD`. This section reads zero holdout rows and
shows only review evidence plus sealed governance metadata.
""".splitlines(keepends=True)
    )

    review = _cell_by_id(notebook, "review-stage")
    review["source"] = (
        """review_queue = read_json_object(discovery_publication.review_queue_path, expected_schema="HumanFactorReviewQueueV1")
if cross_v2_bundle is not None:
    expert_review_df = read_bundle_stage(cross_v2_bundle, "expert_review_queue", limit=5_000)
else:
    expert_review_df = pd.json_normalize(review_queue.get("reviews", []))
    expert_review_df.attrs.update({"status": "PRIOR_EVIDENCE_V1", "reason": "V2 bundle unavailable; V1 publication fallback", "source_path": str(discovery_publication.review_queue_path), "source_sha256": discovery_manifest["files"]["review_queue"]["sha256"], "source_row_count": len(expert_review_df)})

shortlist_gate_df = pd.DataFrame([{
    "shortlist_locked": False,
    "accepted_candidates": 0,
    "holdout_reaction_accessed": False,
    "holdout_status": "SEALED_UNREAD",
    "gate": "SHORTLIST_NOT_FROZEN_HOLDOUT_SEALED",
}])
shortlist_gate_df.attrs.update({"status": "SEALED_UNREAD", "source_path": str(discovery_publication.review_queue_path), "source_sha256": discovery_manifest["files"]["review_queue"]["sha256"], "source_row_count": 1})
display(
    frame_audit(expert_review_df, name="expert_review_df", grain="one expert review queue item"),
    expert_review_df,
    frame_audit(shortlist_gate_df, name="shortlist_gate_df", grain="one shortlist/holdout gate assertion"),
    shortlist_gate_df,
)
""".splitlines(keepends=True)
    )
    review["execution_count"] = None
    review["outputs"] = []

    export = _cell_by_id(notebook, "export-stage")
    export_source = "".join(export.get("source", []))
    for forbidden_line in (
        '    "holdout_results": holdout_results_df,\n',
        '    "holdout_observations": holdout_observations_df,\n',
    ):
        export_source = export_source.replace(forbidden_line, "")
    export["source"] = export_source.splitlines(keepends=True)
    export["execution_count"] = None
    export["outputs"] = []

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            '"holdout_reaction_access": "CLOSED_ZERO_READ"',
            '"holdout_reaction_access": "SEALED_UNREAD"',
        )
        cell["source"] = source.splitlines(keepends=True)


def _zone_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    setup = """import hashlib
from IPython.display import HTML, display
from prediction_market.reports.nfl_x13_research_workbench import (
    load_research_workbench_evidence,
    research_workbench_zone_frame,
)

research_workbench_root = ROOT / "reports/nfl-factor-lab/exact-153/research-workbench-v5"
research_workbench_paths = sorted(research_workbench_root.glob("sha256/*/*.html"))
if len(research_workbench_paths) != 1:
    raise ValueError(
        "Expected exactly one content-addressed research workbench artifact; "
        f"found {len(research_workbench_paths)}"
    )
research_workbench_html = research_workbench_paths[0]
observed_workbench_sha = "sha256:" + hashlib.sha256(
    research_workbench_html.read_bytes()
).hexdigest()
if research_workbench_html.stem != observed_workbench_sha.removeprefix("sha256:"):
    raise ValueError("Research workbench content-address mismatch")

study_manifest = ROOT / "artifacts/market-observation/nfl/x13/historical-study-v2/kalshi-native-time-v3/exact-153/manifests/sha256/50/506a8f5651fca1c5cca0a4fa2be84ad3d0d19e189d6c9555e5eeef507d1c4770.manifest.json"
reference_manifest = ROOT / "artifacts/market-observation/nfl/x13/historical-reference-v2/kalshi-native-time-v3/exact-153/manifests/sha256/c5/c5e60bc106a62003dcf77dde28a159483b27c921fd9610a86f40f90e86f5c952.manifest.json"
matched_manifest = ROOT / "artifacts/market-observation/nfl/x13/matched-reaction-v2/kalshi-native-time-v3/exact-153/manifests/sha256/67/6742fdd550ec9b0890df75879aec3c831d6ae2a633fc02a279a12a8cd3db7ed5.manifest.json"
pmxt_l2_method_audit = ROOT / "artifacts/data-audit/pmxt-l2-method-audit-v1/sha256/92/92a80cd64f7cecd79990d2cd0e484743475afc86b867e711fbeb3c700aa29253.json"
calibration_manifest = ROOT / "artifacts/market-observation/nfl/x13/market-calibration-v1/kalshi-native-time-v3/exact-153/manifests/sha256/10/10b8b443f663a0b4eeb3b69d9b5227f12c2a1f47749dc0b1bc3712e74e3bca3a.manifest.json"
tick_inventory_manifest = ROOT / "artifacts/data-audit/tick-data-inventory-v1/manifests/sha256/c8/c82eb668ba6af9cfc863d4e38b1ed776edae12fb1169dea223fd8a3b1ea5fffc.manifest.json"
workbench_artifact_sources = {
    "historical_distribution": {
        "artifact_sha256": "sha256:506a8f5651fca1c5cca0a4fa2be84ad3d0d19e189d6c9555e5eeef507d1c4770",
        "artifact_path": str(study_manifest),
    },
    "same_episode_paired": {
        "artifact_sha256": "sha256:506a8f5651fca1c5cca0a4fa2be84ad3d0d19e189d6c9555e5eeef507d1c4770",
        "artifact_path": str(study_manifest),
    },
    "reference_completion": {
        "artifact_sha256": "sha256:c5e60bc106a62003dcf77dde28a159483b27c921fd9610a86f40f90e86f5c952",
        "artifact_path": str(reference_manifest),
    },
    "venue_consistency": {
        "artifact_sha256": "sha256:6742fdd550ec9b0890df75879aec3c831d6ae2a633fc02a279a12a8cd3db7ed5",
        "artifact_path": str(matched_manifest),
    },
    "pmxt_l2_method_audit": {
        "artifact_sha256": "sha256:92a80cd64f7cecd79990d2cd0e484743475afc86b867e711fbeb3c700aa29253",
        "artifact_path": str(pmxt_l2_method_audit),
    },
    "calibration_bias": {
        "artifact_sha256": "sha256:10b8b443f663a0b4eeb3b69d9b5227f12c2a1f47749dc0b1bc3712e74e3bca3a",
        "artifact_path": str(calibration_manifest),
    },
    "tick_inventory": {
        "artifact_sha256": "sha256:c82eb668ba6af9cfc863d4e38b1ed776edae12fb1169dea223fd8a3b1ea5fffc",
        "artifact_path": str(tick_inventory_manifest),
    },
}
workbench_evidence = load_research_workbench_evidence(workbench_artifact_sources)

def workbench_zone_link(title, fragment):
    return HTML(
        "<p><strong>打开离线工作台：</strong> "
        f"<a href='file://{research_workbench_html}{fragment}' target='_blank'>"
        f"{title}</a></p>"
    )

tick_inventory_df = research_workbench_zone_frame(
    workbench_evidence, "tick_inventory"
)
display(tick_inventory_df)
display(workbench_zone_link("Tick Inventory", "#ticks"))
"""
    zone_slugs = (
        "tick_inventory",
        "historical_distribution",
        "reference_completion",
        "calibration_bias",
        "poly_kalshi_paired",
        "pmxt_l2_method_audit",
        "shortlist_holdout",
        "shadow_metrics",
        "audit",
    )
    for index, ((zone, anchor, description), zone_slug) in enumerate(
        zip(ZONES, zone_slugs, strict=True),
        start=1,
    ):
        heading = "## 20." if index == 1 else f"### 20.{index}."
        cells.append(
            _markdown_cell(
                f"{heading} {zone}\n\n{description}\n",
                zone=zone,
                anchor=anchor,
            )
        )
        code = (
            setup
            if index == 1
            else (
                f"{zone_slug}_df = research_workbench_zone_frame(\n"
                f'    workbench_evidence, "{zone_slug}"\n'
                ")\n"
                f"display({zone_slug}_df)\n"
                f'display(workbench_zone_link("{zone}", "#{anchor}"))\n'
            )
        )
        cells.append(_code_cell(code, zone=zone, anchor=anchor))
    return cells


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    _seal_holdout(notebook)
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("saf_section_id")
        not in REPLACED_SECTION_IDS
    ]
    notebook["cells"].extend(_zone_cells())
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK)
    print(f"cells={len(notebook['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
