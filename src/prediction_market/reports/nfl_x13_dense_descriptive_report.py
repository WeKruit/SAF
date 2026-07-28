"""Pure Chinese renderers for verified X-13 dense descriptive summaries."""

from __future__ import annotations

from html import escape
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping

import pandas as pd


_CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
_DENSE_HORIZONS_SECONDS = (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUMMARY_COLUMNS = {
    "factor_id",
    "factor_version",
    "market_family",
    "venue",
    "horizon_seconds",
    "observed_n",
    "game_count",
    "mean",
    "median",
    "p10",
    "p90",
    "positive_rate",
    "negative_rate",
    "jump_dominant_share",
    "gradual_share",
    "claim_boundary",
}
_EVIDENCE_FIELDS = {
    "experiment_id",
    "claim_boundary",
    "source_sha256",
    "bundle_sha256",
    "attrition_counts",
}
_GRAIN = (
    "factor_id",
    "factor_version",
    "market_family",
    "venue",
    "horizon_seconds",
)


class DenseDescriptiveReportError(ValueError):
    """Dense summary or evidence descriptor cannot support this report."""


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DenseDescriptiveReportError(f"{field} must be non-empty text")
    text = value.strip()
    if "http://" in text.lower() or "https://" in text.lower():
        raise DenseDescriptiveReportError(f"{field} must not contain a remote URL")
    return text


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DenseDescriptiveReportError(f"{field} must be a SHA-256 digest")
    return value


def _integer(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DenseDescriptiveReportError(f"{field} must be an integer")
    number = int(value)
    if number < minimum:
        raise DenseDescriptiveReportError(f"{field} must be at least {minimum}")
    return number


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DenseDescriptiveReportError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise DenseDescriptiveReportError(f"{field} must be finite")
    return number


def _share(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise DenseDescriptiveReportError(f"{field} must be between zero and one")
    return number


def _verified_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DenseDescriptiveReportError("evidence descriptor must be a mapping")
    if set(value) != _EVIDENCE_FIELDS:
        missing = sorted(_EVIDENCE_FIELDS - set(value))
        if missing:
            raise DenseDescriptiveReportError(
                "evidence descriptor is missing " + ", ".join(missing)
            )
        raise DenseDescriptiveReportError("evidence descriptor has unknown fields")
    if value["experiment_id"] != "X-13":
        raise DenseDescriptiveReportError("experiment_id must be X-13")
    if value["claim_boundary"] != _CLAIM_BOUNDARY:
        raise DenseDescriptiveReportError(
            "claim_boundary must be PRELIMINARY_SOURCE_TIME_ONLY"
        )
    attrition = value["attrition_counts"]
    if not isinstance(attrition, Mapping) or not attrition:
        raise DenseDescriptiveReportError("attrition_counts must be a non-empty mapping")
    counts: dict[str, int] = {}
    for reason, count in attrition.items():
        counts[_text(reason, field="attrition reason")] = _integer(
            count,
            field="attrition count",
            minimum=0,
        )
    return {
        "source_sha256": _digest(value["source_sha256"], field="source_sha256"),
        "bundle_sha256": _digest(value["bundle_sha256"], field="bundle_sha256"),
        "attrition_counts": dict(sorted(counts.items())),
    }


def _verified_summary(frame: object) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DenseDescriptiveReportError("dense summary must be a non-empty DataFrame")
    missing = sorted(_SUMMARY_COLUMNS - set(frame.columns))
    if missing:
        raise DenseDescriptiveReportError(
            "dense summary is missing columns: " + ", ".join(missing)
        )
    result = frame.copy()
    for field in ("factor_id", "factor_version", "market_family", "venue"):
        result[field] = result[field].map(lambda value: _text(value, field=field))
    result["horizon_seconds"] = result["horizon_seconds"].map(
        lambda value: _integer(value, field="horizon_seconds", minimum=1)
    )
    if not result["horizon_seconds"].isin(_DENSE_HORIZONS_SECONDS).all():
        raise DenseDescriptiveReportError(
            "dense summary horizon_seconds is not registered"
        )
    if result.duplicated(list(_GRAIN)).any():
        raise DenseDescriptiveReportError("dense summary has duplicate factor grain")
    for field in ("observed_n", "game_count"):
        result[field] = result[field].map(
            lambda value, name=field: _integer(value, field=name, minimum=1)
        )
    for field in ("mean", "median", "p10", "p90"):
        result[field] = result[field].map(
            lambda value, name=field: _finite(value, field=name)
        )
    for field in (
        "positive_rate",
        "negative_rate",
        "jump_dominant_share",
        "gradual_share",
    ):
        result[field] = result[field].map(
            lambda value, name=field: _share(value, field=name)
        )
    if not result["claim_boundary"].eq(_CLAIM_BOUNDARY).all():
        raise DenseDescriptiveReportError(
            "dense summary claim_boundary must be PRELIMINARY_SOURCE_TIME_ONLY"
        )
    return result.sort_values(list(_GRAIN), kind="mergesort").reset_index(drop=True)


def _heat_color(value: float | None) -> str:
    if value is None:
        return "#f3f4f6"
    if value > 0:
        return "#1b7f5c"
    if value < 0:
        return "#b42318"
    return "#f3f4f6"


def _heat_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}pp"


def _row_label(row: Mapping[str, Any]) -> str:
    return " · ".join(
        (
            str(row["factor_id"]),
            str(row["factor_version"]),
            str(row["market_family"]),
            str(row["venue"]),
        )
    )


def _heatmap_html(summary: pd.DataFrame) -> str:
    dimensions = ["factor_id", "factor_version", "market_family", "venue"]
    pieces = [
        '<div class="table-wrap"><table class="heatmap"><thead><tr>',
        "<th>Factor · Version · Market · Venue</th>",
        *[f"<th>{horizon}s</th>" for horizon in _DENSE_HORIZONS_SECONDS],
        "</tr></thead><tbody>",
    ]
    for key, child in summary.groupby(dimensions, sort=True, dropna=False):
        row = dict(zip(dimensions, key, strict=True))
        lookup = {
            int(item.horizon_seconds): float(item.mean)
            for item in child.itertuples(index=False)
        }
        pieces.append(f"<tr><th>{escape(_row_label(row))}</th>")
        for horizon in _DENSE_HORIZONS_SECONDS:
            value = lookup.get(horizon)
            text_color = "#ffffff" if value is not None and abs(value) >= 0.01 else "#172033"
            pieces.append(
                f'<td style="background-color:{_heat_color(value)};color:{text_color}">'
                f"{_heat_text(value)}</td>"
            )
        pieces.append("</tr>")
    pieces.append("</tbody></table></div>")
    return "".join(pieces)


def _distribution_html(summary: pd.DataFrame) -> str:
    pieces = [
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Factor</th><th>Horizon</th><th>样本 n</th><th>比赛</th>"
        "<th>均值</th><th>中位数</th><th>P10</th><th>P90</th>"
        "<th>正/负号率</th><th>Jump 主导</th><th>Gradual</th>"
        "</tr></thead><tbody>"
    ]
    for row in summary.to_dict("records"):
        pieces.append(
            "<tr>"
            f"<th>{escape(_row_label(row))}</th>"
            f"<td>{int(row['horizon_seconds'])}s</td>"
            f"<td>{int(row['observed_n']):,}</td>"
            f"<td>{int(row['game_count']):,}</td>"
            f"<td>{_heat_text(float(row['mean']))}</td>"
            f"<td>{_heat_text(float(row['median']))}</td>"
            f"<td>{_heat_text(float(row['p10']))}</td>"
            f"<td>{_heat_text(float(row['p90']))}</td>"
            f"<td>{float(row['positive_rate']) * 100:.1f}% / {float(row['negative_rate']) * 100:.1f}%</td>"
            f"<td>{float(row['jump_dominant_share']) * 100:.1f}%</td>"
            f"<td>{float(row['gradual_share']) * 100:.1f}%</td>"
            "</tr>"
        )
    pieces.append("</tbody></table></div>")
    return "".join(pieces)


def _attrition_html(attrition: Mapping[str, int]) -> str:
    rows = "".join(
        f"<tr><th>{escape(reason)}</th><td>{count:,}</td></tr>"
        for reason, count in attrition.items()
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>路径状态</th>'
        "<th>行数</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></div>"
    )


def _heatmap_markdown(summary: pd.DataFrame) -> list[str]:
    headers = ["Factor · Version · Market · Venue"] + [
        f"{horizon}s" for horizon in _DENSE_HORIZONS_SECONDS
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    dimensions = ["factor_id", "factor_version", "market_family", "venue"]
    for key, child in summary.groupby(dimensions, sort=True, dropna=False):
        row = dict(zip(dimensions, key, strict=True))
        lookup = {
            int(item.horizon_seconds): float(item.mean)
            for item in child.itertuples(index=False)
        }
        cells = [_row_label(row)] + [
            _heat_text(lookup.get(horizon)) for horizon in _DENSE_HORIZONS_SECONDS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _distribution_markdown(summary: pd.DataFrame) -> list[str]:
    headers = [
        "Factor",
        "Version",
        "Market",
        "Venue",
        "Horizon",
        "n",
        "比赛",
        "均值",
        "中位数",
        "P10",
        "P90",
        "正号率",
        "负号率",
        "Jump 主导",
        "Gradual",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in summary.to_dict("records"):
        cells = [
            str(row["factor_id"]),
            str(row["factor_version"]),
            str(row["market_family"]),
            str(row["venue"]),
            f"{int(row['horizon_seconds'])}s",
            f"{int(row['observed_n']):,}",
            f"{int(row['game_count']):,}",
            _heat_text(float(row["mean"])),
            _heat_text(float(row["median"])),
            _heat_text(float(row["p10"])),
            _heat_text(float(row["p90"])),
            f"{float(row['positive_rate']) * 100:.1f}%",
            f"{float(row['negative_rate']) * 100:.1f}%",
            f"{float(row['jump_dominant_share']) * 100:.1f}%",
            f"{float(row['gradual_share']) * 100:.1f}%",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_dense_descriptive_html(
    summary: pd.DataFrame,
    evidence: Mapping[str, object],
) -> str:
    """Render standalone Chinese HTML from already-produced dense summaries."""

    verified_summary = _verified_summary(summary)
    verified_evidence = _verified_evidence(evidence)
    attrition = verified_evidence["attrition_counts"]
    assert isinstance(attrition, Mapping)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-13：Dense Descriptive Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f8fa;color:#172033;line-height:1.5}}
main{{max-width:1500px;margin:0 auto;padding:28px}}.boundary{{background:#fff8dd;border-left:5px solid #b78103;padding:12px 16px}}
.muted{{color:#596579}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #d9dee6;padding:7px 9px;text-align:right;white-space:nowrap}}
th{{background:#eef1f5;text-align:left}}.table-wrap{{overflow:auto;border:1px solid #d9dee6}}.heatmap td{{font-variant-numeric:tabular-nums;font-weight:650}}
</style></head><body><main>
<h1>NFL X-13：Dense Descriptive Report</h1>
<p class="boundary"><strong>结论边界：</strong>本页是历史 source-time 描述性结果；不构成因果、Alpha、延迟或执行主张。</p>
<p class="muted">证据 source：<code>{escape(str(verified_evidence['source_sha256']))}</code>；dense bundle：<code>{escape(str(verified_evidence['bundle_sha256']))}</code></p>
<h2>Factor × 注册 dense horizons 热图（均值）</h2>
<p class="muted">绿色表示正均值，红色表示负均值，灰色表示零值或该 horizon 无已产出汇总。</p>
{_heatmap_html(verified_summary)}
<h2>分布与路径形状</h2>
{_distribution_html(verified_summary)}
<h2>路径状态与排除</h2>
{_attrition_html(attrition)}
</main></body></html>"""


def render_dense_descriptive_markdown(
    summary: pd.DataFrame,
    evidence: Mapping[str, object],
) -> str:
    """Render the same verified dense descriptive evidence as Markdown."""

    verified_summary = _verified_summary(summary)
    verified_evidence = _verified_evidence(evidence)
    attrition = verified_evidence["attrition_counts"]
    assert isinstance(attrition, Mapping)
    lines = [
        "# NFL X-13：Dense Descriptive Report",
        "",
        "历史 source-time 描述性结果；不构成因果、Alpha、延迟或执行主张。",
        "",
        f"- Source SHA-256：`{verified_evidence['source_sha256']}`",
        f"- Dense bundle SHA-256：`{verified_evidence['bundle_sha256']}`",
        "",
        "## Factor × 注册 dense horizons 热图（均值）",
        "",
        *_heatmap_markdown(verified_summary),
        "",
        "## 分布与路径形状",
        "",
        *_distribution_markdown(verified_summary),
        "",
        "## 路径状态与排除",
        "",
        "| 原因 | 行数 |",
        "|---|---:|",
    ]
    lines.extend(f"| {reason} | {count:,} |" for reason, count in attrition.items())
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DenseDescriptiveReportError",
    "render_dense_descriptive_html",
    "render_dense_descriptive_markdown",
]
