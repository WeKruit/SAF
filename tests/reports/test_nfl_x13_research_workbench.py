from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pytest

import prediction_market.reports.nfl_x13_research_workbench as workbench
from prediction_market.reports.nfl_x13_research_workbench import (
    ResearchWorkbenchError,
    render_research_workbench,
)


ZONE_NAMES = (
    "Tick Inventory",
    "Historical Distribution",
    "Reference Completion",
    "Calibration/Bias",
    "Poly×Kalshi paired",
    "PMXT L2 method audit",
    "Shortlist/Holdout",
    "Shadow metrics",
    "Audit",
)


def _plain_records(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict("records")
    return [dict(row) for row in rows]


def _rows_sha256(
    tables: Mapping[str, pd.DataFrame | Sequence[Mapping[str, object]]],
) -> str:
    payload = {
        name: _plain_records(rows)
        for name, rows in sorted(tables.items())
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _publication_source(
    root: Path,
    name: str,
    tables: Mapping[str, pd.DataFrame | Sequence[Mapping[str, object]]],
    *,
    objects_as_mapping: bool = False,
) -> dict[str, object]:
    publication_root = root / name
    objects: list[dict[str, object]] = []
    for logical_name, rows in sorted(tables.items()):
        frame = (
            rows.copy()
            if isinstance(rows, pd.DataFrame)
            else pd.DataFrame([dict(row) for row in rows])
        )
        temporary = publication_root / f".{logical_name}.parquet"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(temporary, index=False)
        object_bytes = temporary.read_bytes()
        temporary.unlink()
        object_digest = hashlib.sha256(object_bytes).hexdigest()
        object_path = (
            publication_root
            / "objects"
            / "sha256"
            / object_digest[:2]
            / f"{object_digest}.parquet"
        )
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(object_bytes)
        objects.append(
            {
                "logical_name": logical_name,
                "format": "parquet",
                "object_path": object_path.relative_to(publication_root).as_posix(),
                "object_sha256": "sha256:" + object_digest,
                "byte_length": len(object_bytes),
                "row_count": len(frame),
            }
        )
    manifest_objects: object = (
        {
            str(descriptor["logical_name"]): {
                key: value
                for key, value in descriptor.items()
                if key != "logical_name"
            }
            for descriptor in objects
        }
        if objects_as_mapping
        else objects
    )
    manifest_bytes = json.dumps(
        {
            "schema": "TestResearchWorkbenchPublicationV1",
            "holdout_reaction_accessed": False,
            "objects": manifest_objects,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    path = (
        publication_root
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest_bytes)
    return {
        "artifact_sha256": "sha256:" + digest,
        "artifact_path": str(path),
    }


def _caller_claim(
    source: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame | Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    return {**source, "rows_sha256": _rows_sha256(tables)}


def _tick_inventory_source(
    root: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    publication_root = root / "tick-inventory"
    object_bytes = json.dumps(
        {
            "schema_version": "TickDataInventoryV1",
            "holdout_access": "CLOSED_UNREAD",
            "rows": [dict(row) for row in rows],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    object_digest = hashlib.sha256(object_bytes).hexdigest()
    object_path = (
        publication_root
        / "objects"
        / "sha256"
        / object_digest[:2]
        / f"{object_digest}.json"
    )
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(object_bytes)
    manifest_bytes = json.dumps(
        {
            "schema_version": "TickDataInventoryManifestV1",
            "holdout_access": "CLOSED_UNREAD",
            "rows": {
                "path": object_path.relative_to(publication_root).as_posix(),
                "row_count": len(rows),
                "sha256": "sha256:" + object_digest,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = (
        publication_root
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    return {
        "artifact_sha256": "sha256:" + digest,
        "artifact_path": str(manifest_path),
    }


def _fixtures(tmp_path: Path) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, dict[str, object]],
]:
    summary = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.TEST",
                "factor_version": "v1",
                "market_family": "moneyline",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "observed_n": 20,
                "game_count": 12,
                "mean": 0.02,
                "median": 0.01,
            }
        ]
    )
    histogram = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "bin_left_pp": 1.0,
                "bin_right_pp": 2.0,
                "count": 20,
            }
        ]
    )
    venue_diagnostics = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "comparison_mode": "RAW",
                "effect_mean": -0.5,
                "row_marker": "RAW_SINGLE_VENUE_NOT_PAIRED",
            },
            {
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "venue": "polymarket_minus_kalshi",
                "horizon_seconds": 10,
                "comparison_mode": "SAME_EPISODE_PAIRED",
                "effect_mean": 0.004,
                "row_marker": "PAIRED_VERIFIED",
            },
        ]
    )
    reference_summary = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.TEST",
                "venue": "polymarket",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "reference_delta_mean": 0.03,
                "completion_fraction_median": 0.67,
            }
        ]
    )
    reference_exclusions = pd.DataFrame(
        [
            {
                "support_status": "MODEL_SUPPORT_UNPROVEN",
                "exclusion_reason": "overtime",
                "episode_count": 2,
                "game_count": 1,
            }
        ]
    )
    venue_consistency = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "kalshi_point_estimate": 0.01,
                "polymarket_point_estimate": 0.02,
                "venue_sign_consistent": True,
                "both_venues_support_pass": False,
            }
        ]
    )
    cross_game_results = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.TEST",
                "market_family": "moneyline",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "analysis_status": "SINGLE_VENUE_MATCHED_SHOULD_NOT_BE_PAIRED",
            }
        ]
    )
    study = {
        "summary": summary,
        "histogram": histogram,
        "venue_diagnostics": venue_diagnostics,
    }
    reference = {
        "reference_summary": reference_summary,
        "reference_exclusions": reference_exclusions,
    }
    matched = {
        "venue_consistency": venue_consistency,
        "cross_game_results": cross_game_results,
    }
    same_episode = venue_diagnostics.loc[
        venue_diagnostics["comparison_mode"].eq("SAME_EPISODE_PAIRED")
    ].reset_index(drop=True)
    study_source = _publication_source(
        tmp_path,
        "study",
        {
            "summary": summary,
            "histogram": histogram,
            "venue_diagnostics": venue_diagnostics,
        },
    )
    reference_source = _publication_source(
        tmp_path,
        "reference",
        {
            "reference_summary": reference_summary,
            "reference_exclusions": reference_exclusions,
        },
    )
    matched_source = _publication_source(
        tmp_path,
        "matched",
        {
            "venue_consistency": venue_consistency,
            "cross_game_results": cross_game_results,
        },
    )
    sources = {
        "historical_distribution": _caller_claim(
            study_source, {"summary": summary, "histogram": histogram}
        ),
        "reference_completion": _caller_claim(
            reference_source,
            {
                "reference_summary": reference_summary,
                "reference_exclusions": reference_exclusions,
            },
        ),
        "same_episode_paired": _caller_claim(
            study_source, {"same_episode_paired": same_episode}
        ),
        "venue_consistency": _caller_claim(
            matched_source, {"venue_consistency": venue_consistency}
        ),
    }
    # Present only so the pre-fix implementation reaches the intended RED assertions.
    sources["poly_kalshi_paired"] = sources["venue_consistency"]
    return study, reference, matched, sources


def _render(
    tmp_path: Path,
    *,
    study: Mapping[str, pd.DataFrame],
    reference: Mapping[str, pd.DataFrame],
    matched: Mapping[str, pd.DataFrame],
    sources: Mapping[str, Mapping[str, object]],
    **kwargs: object,
) -> Path:
    return render_research_workbench(
        study_tables=study,
        reference_tables=reference,
        matched_tables=matched,
        notebook_path=Path("notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb"),
        output_root=tmp_path / "output",
        audit={
            "holdout_status": "SEALED_UNREAD",
            "shortlist_status": "NOT_FROZEN",
        },
        artifact_sources=sources,
        **kwargs,
    )


def test_formal_loader_reads_verified_publication_objects_for_all_zones(
    tmp_path: Path,
) -> None:
    _study, _reference, _matched, sources = _fixtures(tmp_path)

    evidence = workbench.load_research_workbench_evidence(sources)

    for zone in (
        "tick_inventory",
        "historical_distribution",
        "reference_completion",
        "calibration_bias",
        "poly_kalshi_paired",
        "pmxt_l2_method_audit",
        "shortlist_holdout",
        "shadow_metrics",
        "audit",
    ):
        frame = workbench.research_workbench_zone_frame(evidence, zone)
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty


def test_workbench_uses_real_paired_tables_and_strict_factor_filter(
    tmp_path: Path,
) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)

    path = _render(
        tmp_path,
        study=study,
        reference=reference,
        matched=matched,
        sources=sources,
    )

    html = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.stem
    for zone in ZONE_NAMES:
        assert zone in html
    assert "SAME_EPISODE_PAIRED" in html
    assert "venue_sign_consistent" in html
    assert "SINGLE_VENUE_MATCHED_SHOULD_NOT_BE_PAIRED" not in html
    assert "RAW_SINGLE_VENUE_NOT_PAIRED" not in html
    assert "function factorRows(rows){return rows.filter(r=>r.factor_id===S.factor)" in html
    assert "r.factor_id==null" not in html
    assert "Global reference exclusions by support_status" in html
    assert "Global Audit" in html
    assert "https://" not in html
    for renderer in (
        "renderTimePath",
        "renderSignedHeatmap",
        "renderDistribution",
    ):
        assert f"function {renderer}" in html
    assert "<svg" in html


def test_workbench_rejects_a_missing_artifact_path(tmp_path: Path) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)
    broken = {name: dict(value) for name, value in sources.items()}
    digest = str(broken["historical_distribution"]["artifact_sha256"]).removeprefix(
        "sha256:"
    )
    broken["historical_distribution"]["artifact_path"] = str(
        tmp_path / "missing" / "sha256" / digest[:2] / f"{digest}.manifest.json"
    )

    with pytest.raises(ResearchWorkbenchError, match="artifact_path does not exist"):
        _render(
            tmp_path,
            study=study,
            reference=reference,
            matched=matched,
            sources=broken,
        )


def test_workbench_rejects_tampered_rows_even_with_recomputed_caller_hash(
    tmp_path: Path,
) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)
    tampered_study = dict(study)
    tampered_study["summary"] = study["summary"].copy()
    tampered_study["summary"].loc[0, "mean"] = 0.99
    broken = {name: dict(value) for name, value in sources.items()}
    broken["historical_distribution"]["rows_sha256"] = _rows_sha256(
        {
            "summary": tampered_study["summary"],
            "histogram": tampered_study["histogram"],
        }
    )

    with pytest.raises(
        ResearchWorkbenchError,
        match="caller rows differ from verified publication object",
    ):
        _render(
            tmp_path,
            study=tampered_study,
            reference=reference,
            matched=matched,
            sources=broken,
        )


def test_tick_inventory_uses_source_venue_data_grade_grain_without_factor_id(
    tmp_path: Path,
) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)
    ticks = [
        {
            "source": "kalshi-native-time-v3",
            "venue": "kalshi",
            "data_grade": "SOURCE_TIME_NATIVE",
            "tick_count": 473460,
        }
    ]
    sources["tick_inventory"] = _caller_claim(
        _tick_inventory_source(tmp_path, ticks),
        {"tick_inventory": ticks},
    )

    html = _render(
        tmp_path,
        study=study,
        reference=reference,
        matched=matched,
        sources=sources,
        tick_inventory_rows=ticks,
    ).read_text()

    assert "SOURCE_TIME_NATIVE" in html
    assert "table(inventoryRows(D.tick_inventory_rows))" in html
    assert "factorRows(D.tick_inventory_rows)" not in html


def test_calibration_loads_metrics_reliability_and_bias_without_factor_id(
    tmp_path: Path,
) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)
    calibration = {
        "metrics": [
            {
                "scope": "overall",
                "scope_value": "all",
                "forecast_count": 81230,
                "brier_score": 0.1423,
            }
        ],
        "reliability": [
            {
                "scope": "venue",
                "scope_value": "kalshi",
                "bin_index": 4,
                "mean_forecast": 0.45,
                "realized_rate": 0.41,
            }
        ],
        "bias_proxies": [
            {
                "proxy_family": "home_away",
                "proxy_value": "HOME",
                "status": "COMPUTED_PROXY",
                "brier_score": 0.1422,
            }
        ],
    }
    sources["calibration_bias"] = _caller_claim(
        _publication_source(
            tmp_path,
            "calibration",
            calibration,
            objects_as_mapping=True,
        ),
        calibration,
    )

    html = _render(
        tmp_path,
        study=study,
        reference=reference,
        matched=matched,
        sources=sources,
        calibration_tables=calibration,
    ).read_text()

    for value in ("brier_score", "mean_forecast", "COMPUTED_PROXY"):
        assert value in html
    assert "table(calibrationRows(D.calibration_metrics))" in html
    assert "table(calibrationRows(D.calibration_reliability))" in html
    assert "table(D.calibration_bias_proxies)" in html

    tampered = {name: list(rows) for name, rows in calibration.items()}
    tampered["metrics"] = [{**tampered["metrics"][0], "brier_score": 0.99}]
    with pytest.raises(
        ResearchWorkbenchError,
        match="caller rows differ from verified publication object",
    ):
        _render(
            tmp_path,
            study=study,
            reference=reference,
            matched=matched,
            sources=sources,
            calibration_tables=tampered,
        )


def test_l2_rows_require_method_only_boundary_and_reject_alpha_claims(
    tmp_path: Path,
) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)
    invalid_l2 = [
        {
            "factor_id": "NFL.EVENT.TEST",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "method_status": "FORWARD_CAPTURE_ONLY",
        }
    ]
    sources["pmxt_l2_method_audit"] = _caller_claim(
        _publication_source(
            tmp_path, "l2", {"pmxt_l2_method_audit": invalid_l2}
        ),
        {"pmxt_l2_method_audit": invalid_l2},
    )

    with pytest.raises(ResearchWorkbenchError, match="METHOD_VALIDATION_ONLY_NON_NFL_ALPHA"):
        _render(
            tmp_path,
            study=study,
            reference=reference,
            matched=matched,
            sources=sources,
            l2_audit_rows=invalid_l2,
        )

    with pytest.raises(ResearchWorkbenchError, match="findings.*alpha claim"):
        _render(
            tmp_path,
            study=study,
            reference=reference,
            matched=matched,
            sources=sources,
            findings=("This is an NFL alpha claim",),
        )


def test_valid_external_rows_are_independently_content_bound(tmp_path: Path) -> None:
    study, reference, matched, sources = _fixtures(tmp_path)
    ticks = [
        {
            "source_id": "verified-source",
            "venue": "kalshi",
            "data_grade": "TRADE_TICK",
        }
    ]
    l2 = [
        {
            "factor_id": "NFL.EVENT.TEST",
            "claim_boundary": "METHOD_VALIDATION_ONLY_NON_NFL_ALPHA",
            "method_status": "FORWARD_CAPTURE_ONLY",
        }
    ]
    shadow = [{"factor_id": "NFL.EVENT.TEST", "shadow_metric": "OBSERVATION_ONLY"}]
    sources["tick_inventory"] = _caller_claim(
        _tick_inventory_source(tmp_path, ticks),
        {"tick_inventory": ticks},
    )
    for name, rows in (
        ("pmxt_l2_method_audit", l2),
        ("shadow_metrics", shadow),
    ):
        sources[name] = _caller_claim(
            _publication_source(tmp_path, name, {name: rows}),
            {name: rows},
        )

    html = _render(
        tmp_path,
        study=study,
        reference=reference,
        matched=matched,
        sources=sources,
        tick_inventory_rows=ticks,
        l2_audit_rows=l2,
        shadow_rows=shadow,
    ).read_text()

    for value in (
        "TRADE_TICK",
        "FORWARD_CAPTURE_ONLY",
        "OBSERVATION_ONLY",
    ):
        assert value in html


def test_master_notebook_has_zero_holdout_data_paths_or_reads() -> None:
    notebook = json.loads(
        Path("notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb").read_text()
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert notebook["metadata"]["saf_factor_lab"]["holdout_reaction_access"] == (
        "SEALED_UNREAD"
    )
    for forbidden in (
        "historical-holdout-results",
        "holdout_results_v1.parquet",
        "holdout_factor_observations_v1.parquet",
        "holdout_results_df",
        "holdout_observations_df",
        '"holdout_reaction_accessed": True',
    ):
        assert forbidden not in code
    assert "SEALED_UNREAD" in code
    assert '"holdout_reaction_accessed": False' in code


def test_master_notebook_has_nine_clean_markdown_code_zone_entries() -> None:
    notebook = json.loads(
        Path("notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb").read_text()
    )
    cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("saf_section_id")
        == "nfl_x13_research_workbench_v2"
    ]

    assert len(cells) == 18
    for index, zone in enumerate(ZONE_NAMES):
        markdown, code = cells[index * 2 : index * 2 + 2]
        assert [markdown["cell_type"], code["cell_type"]] == ["markdown", "code"]
        assert markdown["metadata"]["workbench_zone"] == zone
        assert code["metadata"]["workbench_zone"] == zone
        assert zone in "".join(markdown["source"])
        assert code["execution_count"] is None
        assert code["outputs"] == []
        assert f'#{code["metadata"]["workbench_anchor"]}' in "".join(code["source"])
        assert "research_workbench_zone_frame" in "".join(code["source"])
        assert "display(" in "".join(code["source"])
    all_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell["cell_type"] == "code"
    )
    assert 'glob("sha256/*/*.html")' in all_code
    assert "content-address mismatch" in all_code
    assert "load_research_workbench_evidence" in all_code
