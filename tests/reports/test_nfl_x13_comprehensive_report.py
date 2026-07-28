from __future__ import annotations

import pytest

from prediction_market.reports.nfl_x13_comprehensive_report import (
    ComprehensiveReportError,
    build_comprehensive_report_data,
    render_comprehensive_html,
)


HASH = "sha256:" + "a" * 64
REGISTRY_HASH = "sha256:295e4e48237f8523fe3153f28b5b28adc9f6e1d3a219568047da5c250ed243c1"


def _evidence_descriptors() -> dict[str, dict[str, str]]:
    return {
        section: {
            "experiment_id": "X-13",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "artifact_sha256": HASH,
            "factor_registry_sha256": REGISTRY_HASH,
        }
        for section in (
            "dense_paths",
            "matched_effects",
            "reference_regressions",
            "cross_venue_rows",
        )
    }


def _dense_paths() -> list[dict[str, object]]:
    return [
        {
            "factor_id": "NFL.EVENT.SACK",
            "horizon_seconds": 5,
            "mean_probability_change": -0.012,
            "median_probability_change": -0.010,
            "material_probability": 0.63,
            "support_count": 42,
            "available": True,
            "balanced": False,
            "shape_distribution": {"IMMEDIATE": 9, "GRADUAL": 33},
        },
        {
            "factor_id": "NFL.SCORE.PASS_TD",
            "horizon_seconds": 5,
            "mean_probability_change": 0.020,
            "median_probability_change": 0.018,
            "material_probability": 0.79,
            "support_count": 24,
            "available": True,
            "balanced": True,
            "shape_distribution": {"IMMEDIATE": 20, "GRADUAL": 4},
        },
    ]


def test_report_renders_only_supplied_evidence_with_source_time_boundary() -> None:
    state = build_comprehensive_report_data(
        dense_paths=_dense_paths(),
        matched_effects=[
            {
                "factor_id": "NFL.EVENT.SACK",
                "horizon_seconds": 5,
                "matched_mean_effect": -0.008,
                "beta_draws_summary": {
                    "draw_count": 10_000,
                    "mean": -0.008,
                    "p05": -0.015,
                    "p95": -0.001,
                },
            }
        ],
        reference_regressions=[
            {
                "factor_id": "NFL.EVENT.SACK",
                "coefficient": -0.004,
                "support_count": 42,
                "status": "DIAGNOSTIC_ONLY",
            }
        ],
        cross_venue_rows=[
            {
                "factor_id": "NFL.EVENT.SACK",
                "horizon_seconds": 5,
                "paired_count": 17,
                "kalshi_mean": -0.007,
                "polymarket_mean": -0.010,
                "paired_difference": -0.003,
            }
        ],
        evidence_descriptors=_evidence_descriptors(),
    )

    html = render_comprehensive_html(state)

    assert "NFL X-13 综合证据报告" in html
    assert "均值 / 中位数 / 实质变化概率 / 支持" in html
    assert "可用 / 平衡" in html
    assert "IMMEDIATE" in html and "GRADUAL" in html
    assert "10,000" in html
    assert "配对场馆差异" in html
    assert "历史 source-time" in html
    assert "不构成 Alpha、延迟或执行主张" in html
    assert "background-color:" in html
    assert "https://" not in html
    assert "http://" not in html


def test_report_keeps_empty_sections_empty_instead_of_inventing_values() -> None:
    state = build_comprehensive_report_data(
        dense_paths=[],
        matched_effects=[],
        reference_regressions=[],
        cross_venue_rows=[],
        evidence_descriptors=_evidence_descriptors(),
    )

    html = render_comprehensive_html(state)

    assert state["input_counts"] == {
        "dense_paths": 0,
        "matched_effects": 0,
        "reference_regressions": 0,
        "cross_venue_rows": 0,
    }
    assert html.count("暂无已产出数据；未补造数值。") == 4
    assert "0.00pp" not in html


def test_dense_path_rows_require_available_and_balanced_evidence_flags() -> None:
    with pytest.raises(ComprehensiveReportError, match="available"):
        build_comprehensive_report_data(
            dense_paths=[
                {
                    "factor_id": "NFL.EVENT.SACK",
                    "horizon_seconds": 5,
                    "mean_probability_change": -0.012,
                    "median_probability_change": -0.010,
                    "material_probability": 0.63,
                    "support_count": 42,
                    "balanced": False,
                    "shape_distribution": {},
                }
            ],
            matched_effects=[],
            reference_regressions=[],
            cross_venue_rows=[],
            evidence_descriptors=_evidence_descriptors(),
        )


def test_report_data_rejects_an_experiment_other_than_x13() -> None:
    with pytest.raises(ComprehensiveReportError, match="experiment_id"):
        build_comprehensive_report_data(
            experiment_id="X-99",
            dense_paths=[],
            matched_effects=[],
            reference_regressions=[],
            cross_venue_rows=[],
            evidence_descriptors=_evidence_descriptors(),
        )


def test_renderer_rejects_directly_forged_non_x13_report_data() -> None:
    data = build_comprehensive_report_data(
        dense_paths=[],
        matched_effects=[],
        reference_regressions=[],
        cross_venue_rows=[],
        evidence_descriptors=_evidence_descriptors(),
    )
    data["experiment_id"] = "X-99"

    with pytest.raises(ComprehensiveReportError, match="experiment_id"):
        render_comprehensive_html(data)


def test_report_requires_section_evidence_and_rejects_fabricated_cross_venue_difference() -> None:
    with pytest.raises(ComprehensiveReportError, match="paired_difference"):
        build_comprehensive_report_data(
            dense_paths=[],
            matched_effects=[],
            reference_regressions=[],
            cross_venue_rows=[
                {
                    "factor_id": "NFL.EVENT.SACK",
                    "horizon_seconds": 5,
                    "paired_count": 12,
                    "kalshi_mean": -0.01,
                    "polymarket_mean": -0.02,
                    "paired_difference": 0.77,
                }
            ],
            evidence_descriptors=_evidence_descriptors(),
        )


def test_report_rejects_impossible_change_and_forged_section_scope() -> None:
    bad_dense = _dense_paths()
    bad_dense[0]["mean_probability_change"] = 1.5
    with pytest.raises(ComprehensiveReportError, match="probability_change"):
        build_comprehensive_report_data(
            dense_paths=bad_dense,
            matched_effects=[],
            reference_regressions=[],
            cross_venue_rows=[],
            evidence_descriptors=_evidence_descriptors(),
        )

    forged = _evidence_descriptors()
    forged["dense_paths"]["experiment_id"] = "X-99"
    with pytest.raises(ComprehensiveReportError, match="evidence"):
        build_comprehensive_report_data(
            dense_paths=[],
            matched_effects=[],
            reference_regressions=[],
            cross_venue_rows=[],
            evidence_descriptors=forged,
        )
