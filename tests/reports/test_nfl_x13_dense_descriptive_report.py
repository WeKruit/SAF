from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.reports.nfl_x13_dense_descriptive_report import (
    DenseDescriptiveReportError,
    render_dense_descriptive_html,
    render_dense_descriptive_markdown,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.SACK",
                "factor_version": "v2",
                "market_family": "moneyline",
                "venue": "kalshi",
                "horizon_seconds": 1,
                "observed_n": 40,
                "game_count": 20,
                "mean": 0.012,
                "median": 0.010,
                "p10": -0.002,
                "p90": 0.030,
                "positive_rate": 0.70,
                "negative_rate": 0.20,
                "jump_dominant_share": 0.60,
                "gradual_share": 0.25,
                "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            },
            {
                "factor_id": "NFL.TURNOVER.INTERCEPTION",
                "factor_version": "v2",
                "market_family": "moneyline",
                "venue": "kalshi",
                "horizon_seconds": 10,
                "observed_n": 32,
                "game_count": 18,
                "mean": -0.016,
                "median": -0.012,
                "p10": -0.040,
                "p90": 0.001,
                "positive_rate": 0.15,
                "negative_rate": 0.75,
                "jump_dominant_share": 0.40,
                "gradual_share": 0.45,
                "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            },
        ]
    )


def _evidence() -> dict[str, object]:
    return {
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "source_sha256": "sha256:" + "a" * 64,
        "bundle_sha256": "sha256:" + "b" * 64,
        "attrition_counts": {
            "ELIGIBLE": 72,
            "FORWARD_FILLED": 4,
            "NOT_OBSERVED": 3,
        },
    }


def test_dense_descriptive_renderers_show_inline_heatmap_distribution_and_evidence() -> None:
    html = render_dense_descriptive_html(_summary(), _evidence())
    markdown = render_dense_descriptive_markdown(_summary(), _evidence())

    assert "NFL X-13：Dense Descriptive Report" in html
    assert "background-color:#1b7f5c" in html
    assert "background-color:#b42318" in html
    assert "background-color:#f3f4f6" in html
    assert "60s" in html
    assert "样本 n" in html and "P10" in html and "P90" in html
    assert "Jump 主导" in html and "Gradual" in html
    assert "FORWARD_FILLED" in html
    assert "sha256:" + "a" * 64 in html
    assert "https://" not in html and "http://" not in html
    assert "历史 source-time 描述性结果" in markdown
    assert "| NFL.EVENT.SACK | v2 | moneyline | kalshi | 1s |" in markdown
    assert "ELIGIBLE" in markdown


def test_dense_descriptive_renderers_fail_closed_without_exact_x13_evidence() -> None:
    invalid_experiment = _evidence()
    invalid_experiment["experiment_id"] = "X-99"
    with pytest.raises(DenseDescriptiveReportError, match="experiment_id"):
        render_dense_descriptive_html(_summary(), invalid_experiment)

    missing_bundle = _evidence()
    missing_bundle.pop("bundle_sha256")
    with pytest.raises(DenseDescriptiveReportError, match="bundle_sha256"):
        render_dense_descriptive_markdown(_summary(), missing_bundle)

    invalid_summary = _summary()
    invalid_summary.loc[0, "claim_boundary"] = "LIVE_ALPHA"
    with pytest.raises(DenseDescriptiveReportError, match="claim_boundary"):
        render_dense_descriptive_html(invalid_summary, _evidence())

