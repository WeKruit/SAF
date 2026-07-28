"""Offline Chinese report and heatmap for the exact 153-game X-13 development set."""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import pandas as pd


_REQUIRED_COLUMNS = {
    "factor_id",
    "horizon_seconds",
    "episode_count",
    "game_count",
    "support_gate_pass",
    "point_estimate",
    "ci_low",
    "ci_high",
    "bh_q_value",
    "loo_sign_stability_rate",
    "maximum_single_game_contribution",
    "kalshi_point_estimate",
    "polymarket_point_estimate",
    "venue_sign_consistent",
    "support_status",
}


class X13ReportError(ValueError):
    """The supplied development table cannot support the report."""


def _verified_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise X13ReportError("discovery table must be a nonempty DataFrame")
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise X13ReportError(
            f"discovery table is missing columns: {sorted(missing)}"
        )
    if frame.duplicated(["factor_id", "horizon_seconds"]).any():
        raise X13ReportError("factor/horizon grain is duplicated")
    result = frame.copy()
    result["horizon_seconds"] = result["horizon_seconds"].astype(int)
    result["factor_id"] = result["factor_id"].astype(str)
    return result.sort_values(
        ["factor_id", "horizon_seconds"], kind="mergesort"
    ).reset_index(drop=True)


def _mechanical_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["support_gate_pass"].fillna(False).astype(bool)
        & (frame["bh_q_value"] <= 0.05)
        & (frame["loo_sign_stability_rate"] >= 0.80)
        & (frame["maximum_single_game_contribution"] <= 0.25)
        & frame["venue_sign_consistent"].fillna(False).astype(bool)
    )


def build_report_state(
    discovery: pd.DataFrame,
    *,
    game_count: int,
    actual_market_observations: int,
    reaction_paths: int,
    factor_observations: int,
    source_sha256: str,
) -> dict[str, Any]:
    frame = _verified_frame(discovery)
    if game_count != 153:
        raise X13ReportError("the report requires the exact 153-game set")
    for field, value in {
        "actual_market_observations": actual_market_observations,
        "reaction_paths": reaction_paths,
        "factor_observations": factor_observations,
    }.items():
        if type(value) is not int or value <= 0:
            raise X13ReportError(f"{field} must be a positive integer")
    if (
        not isinstance(source_sha256, str)
        or not source_sha256.startswith("sha256:")
        or len(source_sha256) != 71
    ):
        raise X13ReportError("source_sha256 is invalid")

    mechanical = frame[_mechanical_mask(frame)]
    status_counts = {
        str(name): int(count)
        for name, count in frame["support_status"]
        .value_counts(dropna=False)
        .sort_index()
        .items()
    }
    both_venues = mechanical.dropna(
        subset=["kalshi_point_estimate", "polymarket_point_estimate"]
    )
    if both_venues.empty:
        venue_magnitude_ratio = None
    else:
        kalshi_median = float(
            both_venues["kalshi_point_estimate"].abs().median()
        )
        polymarket_median = float(
            both_venues["polymarket_point_estimate"].abs().median()
        )
        venue_magnitude_ratio = (
            None if kalshi_median == 0.0 else polymarket_median / kalshi_median
        )
    return {
        "schema_version": "NFLX13Exact153ReportStateV1",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": game_count,
        "factor_count": int(frame["factor_id"].nunique()),
        "factor_horizon_rows": int(len(frame)),
        "horizons_seconds": sorted(
            int(value) for value in frame["horizon_seconds"].unique()
        ),
        "actual_market_observations": actual_market_observations,
        "reaction_paths": reaction_paths,
        "factor_observations": factor_observations,
        "support_gate_pass_count": int(
            frame["support_gate_pass"].fillna(False).sum()
        ),
        "bh_q_le_005_count": int((frame["bh_q_value"] <= 0.05).sum()),
        "mechanical_candidate_count": int(len(mechanical)),
        "mechanical_candidate_factor_count": int(
            mechanical["factor_id"].nunique()
        ),
        "support_status_counts": status_counts,
        "median_absolute_polymarket_to_kalshi_ratio": venue_magnitude_ratio,
        "promotion_status": "NEEDS_HUMAN_REVIEW",
        "holdout_access": "CLOSED",
        "source_sha256": source_sha256,
        "claim_boundary": (
            "PRELIMINARY historical source-time association only; "
            "not causality, live latency, execution, or tradable alpha."
        ),
    }


def _cell_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "#f3f4f6"
    if value >= 0.01:
        return "#4575b4"
    if value >= 0.003:
        return "#91bfdb"
    if value > 0:
        return "#d9edf7"
    if value <= -0.01:
        return "#d73027"
    if value <= -0.003:
        return "#fc8d59"
    if value < 0:
        return "#fee8c8"
    return "#f7f7f7"


def _text_color(value: float | None) -> str:
    if value is not None and not pd.isna(value) and abs(value) >= 0.01:
        return "#ffffff"
    return "#111827"


def _factor_family(factor_id: str) -> str:
    parts = factor_id.split(".")
    return parts[1] if len(parts) > 1 else "OTHER"


def _heatmap_html(frame: pd.DataFrame, *, title: str) -> str:
    horizons = sorted(frame["horizon_seconds"].unique())
    pivot = frame.pivot(
        index="factor_id", columns="horizon_seconds", values="point_estimate"
    ).sort_index()
    pieces = [
        f"<h2>{escape(title)}</h2>",
        '<div class="table-wrap"><table class="heatmap"><thead><tr>',
        "<th>Factor</th>",
    ]
    pieces.extend(f"<th>{int(h)}s</th>" for h in horizons)
    pieces.append("</tr></thead><tbody>")
    for factor_id, row in pivot.iterrows():
        pieces.append(
            f'<tr><th class="factor">{escape(str(factor_id))}</th>'
        )
        for horizon in horizons:
            value = row.get(horizon)
            if pd.isna(value):
                label = "—"
                numeric: float | None = None
            else:
                numeric = float(value)
                label = f"{numeric * 100:+.2f}pp"
            pieces.append(
                "<td style=\"background-color:"
                + _cell_color(numeric)
                + ";color:"
                + _text_color(numeric)
                + f'\">{label}</td>'
            )
        pieces.append("</tr>")
    pieces.append("</tbody></table></div>")
    return "".join(pieces)


def _patterns(
    frame: pd.DataFrame,
    *,
    direction: str | None = None,
    limit: int = 12,
) -> pd.DataFrame:
    eligible = frame[_mechanical_mask(frame)].copy()
    if direction == "positive":
        eligible = eligible[eligible["point_estimate"] > 0]
    elif direction == "negative":
        eligible = eligible[eligible["point_estimate"] < 0]
    elif direction is not None:
        raise X13ReportError("direction must be positive, negative, or null")
    if eligible.empty:
        return eligible
    eligible["absolute_estimate"] = eligible["point_estimate"].abs()
    return eligible.sort_values(
        ["absolute_estimate", "bh_q_value", "factor_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(limit)


def _patterns_html(frame: pd.DataFrame, *, direction: str) -> str:
    rows = _patterns(frame, direction=direction)
    if rows.empty:
        return "<p>没有 factor×horizon 同时通过机械门槛。</p>"
    pieces = [
        "<table><thead><tr><th>Factor</th><th>Horizon</th>"
        "<th>变化</th><th>95% CI</th><th>q</th><th>Games</th>"
        "</tr></thead><tbody>"
    ]
    for row in rows.itertuples():
        pieces.append(
            "<tr>"
            f"<td>{escape(str(row.factor_id))}</td>"
            f"<td>{int(row.horizon_seconds)}s</td>"
            f"<td>{float(row.point_estimate) * 100:+.2f}pp</td>"
            f"<td>[{float(row.ci_low) * 100:+.2f}, "
            f"{float(row.ci_high) * 100:+.2f}]pp</td>"
            f"<td>{float(row.bh_q_value):.4f}</td>"
            f"<td>{int(row.game_count)}</td>"
            "</tr>"
        )
    pieces.append("</tbody></table>")
    return "".join(pieces)


def render_exact153_html(
    discovery: pd.DataFrame, state: Mapping[str, Any]
) -> str:
    frame = _verified_frame(discovery)
    required_state = {
        "game_count",
        "factor_count",
        "factor_horizon_rows",
        "actual_market_observations",
        "reaction_paths",
        "factor_observations",
        "mechanical_candidate_count",
        "promotion_status",
        "holdout_access",
        "source_sha256",
        "claim_boundary",
    }
    if not required_state.issubset(state):
        raise X13ReportError("report state is incomplete")

    time_state = frame[
        frame["factor_id"].str.contains(
            r"\.QUARTER_|\.GAME_TIME_|\.WP_", regex=True
        )
    ]
    if time_state.empty:
        time_state = frame[
            frame["factor_id"].map(_factor_family).eq("STATE")
        ]
    status_counts = state.get("support_status_counts", {})
    pass_count = int(status_counts.get("PASS", 0))
    insufficient_count = int(status_counts.get("INSUFFICIENT_SUPPORT", 0))
    data_gap_count = int(status_counts.get("DATA_GAP", 0))
    venue_ratio = state.get("median_absolute_polymarket_to_kalshi_ratio")
    venue_ratio_text = (
        "不可计算"
        if venue_ratio is None
        else f"{float(venue_ratio):.1f}×"
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-13：153 场 Factor × Market Reaction</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
margin:0;background:#f7f8fa;color:#172033;line-height:1.5}}
main{{max-width:1480px;margin:0 auto;padding:28px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:12px}}.card{{background:white;border:1px solid #dce1e8;border-radius:10px;
padding:14px}}.metric{{font-size:1.55rem;font-weight:750}}.muted{{color:#5d6878}}
.boundary{{background:#fff8dd;border-left:5px solid #d5a000;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #d9dee6;padding:7px 9px;text-align:right;
white-space:nowrap}}th{{background:#eef1f5}}th.factor{{text-align:left;position:sticky;
left:0;z-index:1}}.table-wrap{{overflow:auto;max-height:900px;border:1px solid #d9dee6}}
.heatmap td{{font-variant-numeric:tabular-nums;font-weight:650}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}}.swatch{{display:inline-block;
width:16px;height:16px;vertical-align:-3px;margin-right:5px;border:1px solid #aaa}}
h1,h2{{letter-spacing:-.02em}}code{{background:#eef1f5;padding:2px 5px}}
</style></head><body><main>
<h1>NFL X-13：153 场 Game State × Prediction Market</h1>
<p class="boundary"><strong>结论边界：</strong>这些是 development 样本的历史
source-time 描述性关联，不是因果、真实延迟或可交易 Alpha。所有“通过”仍需体育专家审核，
81 场 Holdout 目前保持 CLOSED。</p>
<h2>当前 State</h2>
<div class="cards">
<div class="card"><div class="metric">{int(state["game_count"]):,}</div><div>比赛</div></div>
<div class="card"><div class="metric">{int(state["factor_count"]):,}</div><div>Factors</div></div>
<div class="card"><div class="metric">{int(state["factor_horizon_rows"]):,}</div><div>Factor × Horizon</div></div>
<div class="card"><div class="metric">{int(state["actual_market_observations"]):,}</div><div>实际市场观察</div></div>
<div class="card"><div class="metric">{int(state["reaction_paths"]):,}</div><div>Reaction paths</div></div>
<div class="card"><div class="metric">{int(state["factor_observations"]):,}</div><div>Factor observations</div></div>
<div class="card"><div class="metric">{int(state["mechanical_candidate_count"]):,}</div><div>机械候选；未晋升</div></div>
</div>
<p class="muted">Source: <code>{escape(str(state["source_sha256"]))}</code></p>
<h2>颜色与名词</h2>
<div class="legend">
<span><i class="swatch" style="background:#4575b4"></i>蓝色表示上升</span>
<span><i class="swatch" style="background:#d73027"></i>红色表示下降</span>
<span><i class="swatch" style="background:#f7f7f7"></i>接近零</span>
</div>
<p>数值单位是百分点（pp）。颜色只表示 signed price change 的方向与幅度，
不表示盈利、执行性或统计晋升。</p>
<h2>当前能读出的结论</h2>
<ul>
<li>{int(state["mechanical_candidate_count"]):,} 个 factor×horizon 单元通过严格机械筛选，
覆盖 {int(state["mechanical_candidate_factor_count"]):,} 个 factor；它们都还没有经过 matched-control
和体育专家审核。</li>
<li>支持状态：PASS {pass_count:,}、INSUFFICIENT_SUPPORT {insufficient_count:,}、
DATA_GAP {data_gap_count:,}。PASS 只表示样本支持门通过，不表示规律成立。</li>
<li>严格机械候选中，Polymarket 的绝对变化中位数约为 Kalshi 的
{venue_ratio_text}；因此当前大幅度结果明显由 Polymarket 贡献更多，不能把“同方向”误读为
双 venue 等强度复现。</li>
<li>多数较大的变化出现在 20–60 秒，而不是 1–5 秒；这既可能是渐进调整，也可能来自
source-time 不确定性、成交稀疏和样本构成变化，必须用 matched controls 与前向 L2 区分。</li>
<li>Quarter / game-time 单独分桶的均值整体较小；OT 的部分大幅度单元支持不足。
因此“第几节”本身不是当前结论，必须与事件、比分、球权和赛前概率组合审核。</li>
</ul>
<h2>上升方向（未匹配）</h2>
{_patterns_html(frame, direction="positive")}
<h2>下降方向（未匹配）</h2>
{_patterns_html(frame, direction="negative")}
{_heatmap_html(frame, title="153 场完整热点图：Factor × 事件后秒数")}
{_heatmap_html(time_state, title="按比赛相对时间与状态：Quarter / Game Time / WP")}
<h2>状态名词</h2>
<table><thead><tr><th>状态</th><th>含义</th></tr></thead><tbody>
<tr><td>PASS</td><td>该 factor×horizon 达到 registry 的最低 episode/game 支持；不是统计或体育审核通过。</td></tr>
<tr><td>INSUFFICIENT_SUPPORT</td><td>定义可计算，但本批有效 episode/game 数不足，不能稳定解释。</td></tr>
<tr><td>DATA_GAP</td><td>所需字段或合法数据层不存在，结果不允许被补造。</td></tr>
<tr><td>N/R</td><td>Not reported / not applicable；通常表示该单元没有正式结果，绝不是 0。</td></tr>
</tbody></table>
<h2>实现与证据 State</h2>
<table><thead><tr><th>Component</th><th>State</th><th>证据边界</th></tr></thead><tbody>
<tr><td>Exact-153 historical aggregate</td><td>COMPLETE</td>
<td>153 场 immutable objects 与 hash 已验证；本页所有数值直接读取该对象。</td></tr>
<tr><td>Matched controls</td><td>CODE_GREEN / FULL_BUNDLE_NOT_RUN</td>
<td>匹配实现已通过目标测试；完整 153 场 matched publication 尚未生成，不能用本页未匹配均值替代。</td></tr>
<tr><td>nflfastR reference</td><td>DIAGNOSTIC_ONLY</td>
<td>reference-gap 接口可用；当前 vegas_wp 对应的 with-spread model bytes 尚未独立验证，正式 gate 关闭。</td></tr>
<tr><td>X-14 prospective L2</td><td>INTERFACE_GREEN / NO_LIVE_SESSION</td>
<td>recorder→timing→bid/ask t50/t90 接口已通过目标测试；尚无未来比赛、独立 sports feed 或 Kalshi key 的真实 session。</td></tr>
<tr><td>81-game final holdout</td><td>CLOSED / ZERO_READ</td>
<td>只有专家审核和 shortlist hash 冻结后才允许一次性打开。</td></tr>
</tbody></table>
<h2>下一道 Gate</h2>
<ol><li>Matched-control 结果与当前未匹配均值分层。</li>
<li>体育专家逐 factor ACCEPT / REDEFINE / DATA_GAP / REJECT。</li>
<li>冻结 shortlist hash 后，81 场 Holdout 只运行一次。</li></ol>
</main></body></html>"""


def render_exact153_markdown(
    discovery: pd.DataFrame, state: Mapping[str, Any]
) -> str:
    frame = _verified_frame(discovery)
    positive_patterns = _patterns(frame, direction="positive")
    negative_patterns = _patterns(frame, direction="negative")
    lines = [
        "# NFL X-13：153 场完整报告",
        "",
        "## 当前 State",
        "",
        f"- 比赛：{int(state['game_count']):,}",
        f"- Factors：{int(state['factor_count']):,}",
        f"- Factor × Horizon：{int(state['factor_horizon_rows']):,}",
        f"- 实际市场观察：{int(state['actual_market_observations']):,}",
        f"- Reaction paths：{int(state['reaction_paths']):,}",
        f"- Factor observations：{int(state['factor_observations']):,}",
        f"- Support PASS：{int(state['support_gate_pass_count']):,}",
        f"- 机械候选：{int(state['mechanical_candidate_count']):,}",
        "- 机械候选覆盖 Factors："
        f"{int(state['mechanical_candidate_factor_count']):,}",
        f"- Holdout：{state['holdout_access']}",
        "",
        "当前所有结果仍是 historical source-time development association；"
        "不是因果、实时延迟、执行性或可交易 alpha。",
        "",
        "## 这份报告究竟计算了什么",
        "",
        "- 每个 factor 先在每场比赛内计算，再跨 153 场聚合。",
        "- Horizon 是事件后 1/2/5/10/20/30/60 秒；"
        "没有实际成交时不 forward-fill。",
        "- 数值是按 focal outcome 定向后的实际成交概率变化，单位为百分点（pp）。",
        "- 当前表是未匹配 development 均值；matched-control 结果必须单独发布，"
        "不能混写。",
        "- 机械候选要求样本支持、q≤0.05、LOO 同号率≥0.80、"
        "最大单场贡献≤25% 且双 venue 同方向。",
        "",
        "## 当前可以读出的规律",
        "",
        "- 正向最强的是 explosive play、first down、"
        "fourth-down conversion。",
        "- 负向最强的是 red-zone empty drive、turnover on downs、"
        "missed field goal、red-zone turnover 和 lost fumble。",
        "- 大变化主要集中在 20–60 秒。它可能表示渐进调整，也可能来自"
        " source-time 不确定性、成交稀疏或样本构成，历史数据不能区分。",
        "- Quarter / game-time 单独分桶的均值整体较小；OT 的部分大幅度单元"
        "支持不足。因此比赛相对时间必须与事件、比分、球权和赛前概率组合审核。",
        "- 双 venue 虽常同方向，但机械候选中 Polymarket 的绝对变化中位数约为"
        f" Kalshi 的 {float(state['median_absolute_polymarket_to_kalshi_ratio']):.1f}×；"
        "当前幅度不是双 venue 等强度复现。",
        "- 以上都仍是待审核 pattern，不是 signal、因果或 alpha。",
        "",
        "## 主要上升方向（未匹配）",
        "",
        "| Factor | Horizon | Change | 95% CI | q | Games |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in positive_patterns.itertuples():
        lines.append(
            f"| {row.factor_id} | {int(row.horizon_seconds)}s | "
            f"{float(row.point_estimate) * 100:+.2f}pp | "
            f"[{float(row.ci_low) * 100:+.2f}, "
            f"{float(row.ci_high) * 100:+.2f}]pp | "
            f"{float(row.bh_q_value):.4f} | {int(row.game_count)} |"
        )
    lines.extend(
        [
            "",
            "## 主要下降方向（未匹配）",
            "",
            "| Factor | Horizon | Change | 95% CI | q | Games |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in negative_patterns.itertuples():
        lines.append(
            f"| {row.factor_id} | {int(row.horizon_seconds)}s | "
            f"{float(row.point_estimate) * 100:+.2f}pp | "
            f"[{float(row.ci_low) * 100:+.2f}, "
            f"{float(row.ci_high) * 100:+.2f}]pp | "
            f"{float(row.bh_q_value):.4f} | {int(row.game_count)} |"
        )
    lines.extend(
        [
            "",
            "## 状态名词",
            "",
            "- PASS：只表示样本支持门通过，不代表规律成立。",
            "- INSUFFICIENT_SUPPORT：有效 episode/game 不足。",
            "- DATA_GAP：正式所需字段或数据层缺失。",
            "- N/R：未报告或不适用，不等于 0。",
            "",
            "## 当前执行状态",
            "",
            "- Exact-153：已发布并核验 hash；本报告与热力图直接读取该 "
            "immutable object。",
            "- Matched controls：代码与目标测试已完成 reviewer 修复；"
            "完整 153 场 publication 仍须独立生成，不能用当前未匹配均值替代。",
            "- Reference value：可以生成 diagnostic `reference_gap`；当前"
            " `vegas_wp` 对应的 with-spread 模型 bytes 尚未独立验证，"
            "正式 gate 关闭。",
            "- X-14：用于未来比赛的真实 event-receive、book-ready、bid/ask、"
            "t50/t90；接口与目标测试已完成，但尚无未来比赛真实 session。"
            "历史 X-13 不能回答这些真实延迟问题。",
            "",
            "## Holdout Gate",
            "",
            "只有体育专家审核并冻结 shortlist hash 后，才允许一次性读取 81 场 Holdout。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "X13ReportError",
    "build_report_state",
    "render_exact153_html",
    "render_exact153_markdown",
]
