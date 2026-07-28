from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from prediction_market.reports.nfl_x13_report import (
    build_report_state,
    render_exact153_html,
    render_exact153_markdown,
)


def _fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_id, values in {
        "NFL.EVENT.EXPLOSIVE_PLAY": {1: 0.002, 5: 0.010, 10: 0.015},
        "NFL.EVENT.SACK": {1: -0.003, 5: -0.011, 10: -0.020},
        "NFL.STATE.QUARTER_4": {1: -0.001, 5: -0.004, 10: -0.008},
    }.items():
        for horizon, estimate in values.items():
            rows.append(
                {
                    "factor_id": factor_id,
                    "factor_version": "v2",
                    "market_family": "moneyline",
                    "horizon_seconds": horizon,
                    "delay_seconds": 0,
                    "episode_count": 100,
                    "game_count": 60,
                    "observation_count": 100,
                    "support_gate_pass": True,
                    "point_estimate": estimate,
                    "ci_low": estimate - 0.001,
                    "ci_high": estimate + 0.001,
                    "bh_q_value": 0.01,
                    "loo_sign_stability_rate": 1.0,
                    "maximum_single_game_contribution": 0.10,
                    "kalshi_point_estimate": estimate * 0.8,
                    "polymarket_point_estimate": estimate * 1.1,
                    "venue_sign_consistent": True,
                    "support_status": "PASS",
                    "claim_boundary": "PRELIMINARY source-time only",
                }
            )
    return pd.DataFrame(rows)


def test_report_state_distinguishes_mechanical_candidates_from_claims() -> None:
    state = build_report_state(
        _fixture(),
        game_count=153,
        actual_market_observations=4_773_299,
        reaction_paths=610_932,
        factor_observations=496_996,
        source_sha256="sha256:" + "a" * 64,
    )

    assert state["game_count"] == 153
    assert state["factor_count"] == 3
    assert state["horizons_seconds"] == [1, 5, 10]
    assert state["mechanical_candidate_count"] == 9
    assert state["mechanical_candidate_factor_count"] == 3
    assert state["support_status_counts"] == {"PASS": 9}
    assert state["promotion_status"] == "NEEDS_HUMAN_REVIEW"
    assert state["holdout_access"] == "CLOSED"
    assert "tradable alpha" in state["claim_boundary"]


def test_html_has_true_inline_heatmap_colors_legend_and_full_state() -> None:
    state = build_report_state(
        _fixture(),
        game_count=153,
        actual_market_observations=4_773_299,
        reaction_paths=610_932,
        factor_observations=496_996,
        source_sha256="sha256:" + "a" * 64,
    )
    html = render_exact153_html(_fixture(), state)

    assert "<!doctype html>" in html.lower()
    assert "background-color:" in html
    assert "#d73027" in html
    assert "#4575b4" in html
    assert "蓝色表示上升" in html
    assert "红色表示下降" in html
    assert "153 场" in html
    assert "4,773,299" in html
    assert "NFL.EVENT.EXPLOSIVE_PLAY" in html
    assert "NFL.STATE.QUARTER_4" in html
    assert "按比赛相对时间与状态" in html
    assert "上升方向（未匹配）" in html
    assert "下降方向（未匹配）" in html
    assert "PASS 只表示样本支持门通过" in html
    assert "INSUFFICIENT_SUPPORT" in html
    assert "DATA_GAP" in html
    assert "N/R" in html
    assert "不是因果、真实延迟或可交易 Alpha" in html
    assert "Exact-153 historical aggregate" in html
    assert "CODE_GREEN / FULL_BUNDLE_NOT_RUN" in html
    assert "INTERFACE_GREEN / NO_LIVE_SESSION" in html
    assert "CLOSED / ZERO_READ" in html
    assert "https://" not in html


def test_markdown_and_state_are_deterministic_and_content_addressable(
    tmp_path: Path,
) -> None:
    state = build_report_state(
        _fixture(),
        game_count=153,
        actual_market_observations=4_773_299,
        reaction_paths=610_932,
        factor_observations=496_996,
        source_sha256="sha256:" + "a" * 64,
    )
    first = render_exact153_markdown(_fixture(), state)
    second = render_exact153_markdown(_fixture(), state)
    assert first == second
    assert "当前 State" in first
    assert "主要上升方向（未匹配）" in first
    assert "主要下降方向（未匹配）" in first
    assert "Holdout" in first

    output = tmp_path / "state.json"
    payload = (
        json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    output.write_bytes(payload)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == hashlib.sha256(
        payload
    ).hexdigest()
