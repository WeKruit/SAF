"""Offline render primitives for already-produced NFL X-13 evidence rows.

The module deliberately does not estimate an effect, infer missing rows, or
promote factors.  It only validates and renders evidence supplied by the
upstream dense-path, matched, reference, and cross-venue publications.
"""

from __future__ import annotations

from html import escape
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping, Sequence


_EMPTY_SECTION = "暂无已产出数据；未补造数值。"
_CLAIM_BOUNDARY = "PRELIMINARY_SOURCE_TIME_ONLY"
_FACTOR_REGISTRY_V2_SHA256 = (
    "sha256:295e4e48237f8523fe3153f28b5b28adc9f6e1d3a219568047da5c250ed243c1"
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EVIDENCE_SECTIONS = (
    "dense_paths",
    "matched_effects",
    "reference_regressions",
    "cross_venue_rows",
)


class ComprehensiveReportError(ValueError):
    """Rows do not have the minimum evidence needed for safe rendering."""


def _records(value: object, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ComprehensiveReportError(f"{field} must be a row sequence")
    result: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ComprehensiveReportError(f"{field} rows must be mappings")
        result.append(dict(row))
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComprehensiveReportError(f"{field} must be non-empty text")
    text = value.strip()
    if "http://" in text.lower() or "https://" in text.lower():
        raise ComprehensiveReportError(f"{field} must not contain a remote URL")
    return text


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ComprehensiveReportError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ComprehensiveReportError(f"{field} must be finite")
    return number


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ComprehensiveReportError(f"{field} must be an integer")
    number = int(value)
    if number < minimum:
        raise ComprehensiveReportError(f"{field} must be at least {minimum}")
    return number


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ComprehensiveReportError(f"{field} must be boolean")
    return value


def _probability(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ComprehensiveReportError(f"{field} must be between zero and one")
    return number


def _probability_change(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if not -1.0 <= number <= 1.0:
        raise ComprehensiveReportError(f"{field} must be a probability_change in [-1, 1]")
    return number


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ComprehensiveReportError(f"{field} must be a SHA-256 digest")
    return value


def _evidence_descriptors(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ComprehensiveReportError("evidence_descriptors must be a mapping")
    if set(value) != set(_EVIDENCE_SECTIONS):
        raise ComprehensiveReportError("evidence_descriptors must cover every report section")
    result: dict[str, dict[str, str]] = {}
    for section in _EVIDENCE_SECTIONS:
        descriptor = value.get(section)
        if not isinstance(descriptor, Mapping):
            raise ComprehensiveReportError(f"{section} evidence descriptor is invalid")
        if descriptor.get("experiment_id") != "X-13":
            raise ComprehensiveReportError(f"{section} evidence experiment_id must be X-13")
        if descriptor.get("claim_boundary") != _CLAIM_BOUNDARY:
            raise ComprehensiveReportError(f"{section} evidence claim boundary is invalid")
        artifact_sha = _sha256(
            descriptor.get("artifact_sha256"),
            field=f"{section} evidence artifact_sha256",
        )
        registry_sha = _sha256(
            descriptor.get("factor_registry_sha256"),
            field=f"{section} evidence factor_registry_sha256",
        )
        if registry_sha != _FACTOR_REGISTRY_V2_SHA256:
            raise ComprehensiveReportError(f"{section} evidence registry hash is not frozen X-13")
        result[section] = {
            "experiment_id": "X-13",
            "claim_boundary": _CLAIM_BOUNDARY,
            "artifact_sha256": artifact_sha,
            "factor_registry_sha256": registry_sha,
        }
    return result


def _required(row: Mapping[str, Any], name: str) -> object:
    if name not in row:
        raise ComprehensiveReportError(f"row is missing {name}")
    return row[name]


def _shape_distribution(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ComprehensiveReportError("shape_distribution must be a mapping")
    result: dict[str, int] = {}
    for name, count in value.items():
        label = _text(name, field="shape_distribution label")
        result[label] = _integer(
            count,
            field="shape_distribution count",
            minimum=0,
        )
    return dict(sorted(result.items()))


def _dense_path_rows(value: object) -> list[dict[str, Any]]:
    rows = _records(value, field="dense_paths")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        factor_id = _text(_required(row, "factor_id"), field="factor_id")
        horizon = _integer(
            _required(row, "horizon_seconds"),
            field="horizon_seconds",
            minimum=1,
        )
        key = (factor_id, horizon)
        if key in seen:
            raise ComprehensiveReportError("dense_paths factor/horizon grain is duplicated")
        seen.add(key)
        result.append(
            {
                "factor_id": factor_id,
                "horizon_seconds": horizon,
                "mean_probability_change": _probability_change(
                    _required(row, "mean_probability_change"),
                    field="mean_probability_change",
                ),
                "median_probability_change": _probability_change(
                    _required(row, "median_probability_change"),
                    field="median_probability_change",
                ),
                "material_probability": _probability(
                    _required(row, "material_probability"),
                    field="material_probability",
                ),
                "support_count": _integer(
                    _required(row, "support_count"),
                    field="support_count",
                ),
                "available": _boolean(
                    _required(row, "available"),
                    field="available",
                ),
                "balanced": _boolean(
                    _required(row, "balanced"),
                    field="balanced",
                ),
                "shape_distribution": _shape_distribution(
                    _required(row, "shape_distribution")
                ),
            }
        )
    return sorted(result, key=lambda row: (row["factor_id"], row["horizon_seconds"]))


def _beta_summary(value: object) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or not value:
        raise ComprehensiveReportError("beta_draws_summary must be a non-empty mapping")
    result: dict[str, float | int] = {}
    for name, number in value.items():
        label = _text(name, field="beta_draws_summary label")
        if label == "draw_count":
            result[label] = _integer(number, field="beta_draws_summary draw_count", minimum=1)
        else:
            result[label] = _finite(number, field="beta_draws_summary value")
    return dict(sorted(result.items()))


def _matched_effect_rows(value: object) -> list[dict[str, Any]]:
    rows = _records(value, field="matched_effects")
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "factor_id": _text(_required(row, "factor_id"), field="factor_id"),
                "horizon_seconds": _integer(
                    _required(row, "horizon_seconds"),
                    field="horizon_seconds",
                    minimum=1,
                ),
                "matched_mean_effect": _probability_change(
                    _required(row, "matched_mean_effect"),
                    field="matched_mean_effect",
                ),
                "beta_draws_summary": _beta_summary(
                    _required(row, "beta_draws_summary")
                ),
            }
        )
    return sorted(result, key=lambda row: (row["factor_id"], row["horizon_seconds"]))


def _reference_regression_rows(value: object) -> list[dict[str, Any]]:
    rows = _records(value, field="reference_regressions")
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "factor_id": _text(_required(row, "factor_id"), field="factor_id"),
                "coefficient": _finite(
                    _required(row, "coefficient"), field="coefficient"
                ),
                "support_count": _integer(
                    _required(row, "support_count"), field="support_count"
                ),
                "status": _text(_required(row, "status"), field="status"),
            }
        )
    return sorted(result, key=lambda row: row["factor_id"])


def _cross_venue_rows(value: object) -> list[dict[str, Any]]:
    rows = _records(value, field="cross_venue_rows")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        factor_id = _text(_required(row, "factor_id"), field="factor_id")
        horizon = _integer(
            _required(row, "horizon_seconds"),
            field="horizon_seconds",
            minimum=1,
        )
        key = (factor_id, horizon)
        if key in seen:
            raise ComprehensiveReportError("cross_venue_rows factor/horizon grain is duplicated")
        seen.add(key)
        kalshi_mean = _probability_change(
            _required(row, "kalshi_mean"), field="kalshi_mean"
        )
        polymarket_mean = _probability_change(
            _required(row, "polymarket_mean"), field="polymarket_mean"
        )
        paired_difference = _probability_change(
            _required(row, "paired_difference"), field="paired_difference"
        )
        expected_difference = polymarket_mean - kalshi_mean
        if not math.isclose(
            paired_difference,
            expected_difference,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ComprehensiveReportError(
                "paired_difference must equal polymarket_mean minus kalshi_mean"
            )
        result.append(
            {
                "factor_id": factor_id,
                "horizon_seconds": horizon,
                "paired_count": _integer(
                    _required(row, "paired_count"), field="paired_count"
                ),
                "kalshi_mean": kalshi_mean,
                "polymarket_mean": polymarket_mean,
                "paired_difference": paired_difference,
            }
        )
    return sorted(result, key=lambda row: (row["factor_id"], row["horizon_seconds"]))


def build_comprehensive_report_data(
    *,
    experiment_id: str = "X-13",
    dense_paths: object,
    matched_effects: object,
    reference_regressions: object,
    cross_venue_rows: object,
    evidence_descriptors: object,
) -> dict[str, object]:
    """Validate only completed rows and retain explicit empty evidence sections."""

    if experiment_id != "X-13":
        raise ComprehensiveReportError("experiment_id must be X-13")
    evidence = _evidence_descriptors(evidence_descriptors)
    dense = _dense_path_rows(dense_paths)
    matched = _matched_effect_rows(matched_effects)
    reference = _reference_regression_rows(reference_regressions)
    venue = _cross_venue_rows(cross_venue_rows)
    return {
        "schema_version": "NFLX13ComprehensiveReportDataV1",
        "experiment_id": experiment_id,
        "claim_boundary": _CLAIM_BOUNDARY,
        "evidence_descriptors": evidence,
        "input_counts": {
            "dense_paths": len(dense),
            "matched_effects": len(matched),
            "reference_regressions": len(reference),
            "cross_venue_rows": len(venue),
        },
        "dense_paths": dense,
        "matched_effects": matched,
        "reference_regressions": reference,
        "cross_venue_rows": venue,
    }


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def _heat_color(value: float) -> str:
    if value >= 0.01:
        return "#2166ac"
    if value > 0:
        return "#d1e5f0"
    if value <= -0.01:
        return "#b2182b"
    if value < 0:
        return "#fddbc7"
    return "#f7f7f7"


def _heat_text_color(value: float) -> str:
    return "#ffffff" if abs(value) >= 0.01 else "#172033"


def _empty_or_table(headers: Sequence[str], rows: Sequence[str]) -> str:
    if not rows:
        return f"<p class=\"empty\">{_EMPTY_SECTION}</p>"
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header_html
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _dense_html(rows: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        mean = float(row["mean_probability_change"])
        color = _heat_color(mean)
        rendered.append(
            "<tr>"
            f"<th>{escape(str(row['factor_id']))}</th>"
            f"<td>{int(row['horizon_seconds'])}s</td>"
            f'<td style="background-color:{color};color:{_heat_text_color(mean)}">{_pp(mean)}</td>'
            f"<td>{_pp(float(row['median_probability_change']))}</td>"
            f"<td>{float(row['material_probability']) * 100:.1f}%</td>"
            f"<td>{int(row['support_count']):,}</td>"
            f"<td>{'可用' if row['available'] else '不可用'} / {'平衡' if row['balanced'] else '未平衡'}</td>"
            "</tr>"
        )
    return _empty_or_table(
        [
            "Factor",
            "Horizon",
            "均值",
            "中位数",
            "实质变化概率",
            "支持",
            "可用 / 平衡",
        ],
        rendered,
    )


def _shape_html(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">暂无 dense path 输入；未补造形状分布。</p>'
    rendered: list[str] = []
    for row in rows:
        distribution = row["shape_distribution"]
        assert isinstance(distribution, Mapping)
        values = "；".join(
            f"{escape(str(name))} {int(count):,}" for name, count in distribution.items()
        ) or "—"
        rendered.append(
            "<tr>"
            f"<th>{escape(str(row['factor_id']))}</th>"
            f"<td>{int(row['horizon_seconds'])}s</td>"
            f"<td>{values}</td>"
            "</tr>"
        )
    return _empty_or_table(["Factor", "Horizon", "形状分布"], rendered)


def _matched_html(rows: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        summary = row["beta_draws_summary"]
        assert isinstance(summary, Mapping)
        beta = "；".join(
            f"{escape(str(name))}="
            + (f"{int(value):,}" if name == "draw_count" else _pp(float(value)))
            for name, value in summary.items()
        )
        rendered.append(
            "<tr>"
            f"<th>{escape(str(row['factor_id']))}</th>"
            f"<td>{int(row['horizon_seconds'])}s</td>"
            f"<td>{_pp(float(row['matched_mean_effect']))}</td>"
            f"<td>{beta}</td>"
            "</tr>"
        )
    return _empty_or_table(
        ["Factor", "Horizon", "匹配效应均值", "Beta draws 摘要"],
        rendered,
    )


def _reference_html(rows: Sequence[Mapping[str, Any]]) -> str:
    rendered = [
        "<tr>"
        f"<th>{escape(str(row['factor_id']))}</th>"
        f"<td>{_pp(float(row['coefficient']))}</td>"
        f"<td>{int(row['support_count']):,}</td>"
        f"<td>{escape(str(row['status']))}</td>"
        "</tr>"
        for row in rows
    ]
    return _empty_or_table(["Factor", "回归系数", "支持", "状态"], rendered)


def _venue_html(rows: Sequence[Mapping[str, Any]]) -> str:
    rendered = [
        "<tr>"
        f"<th>{escape(str(row['factor_id']))}</th>"
        f"<td>{int(row['horizon_seconds'])}s</td>"
        f"<td>{int(row['paired_count']):,}</td>"
        f"<td>{_pp(float(row['kalshi_mean']))}</td>"
        f"<td>{_pp(float(row['polymarket_mean']))}</td>"
        f"<td>{_pp(float(row['paired_difference']))}</td>"
        "</tr>"
        for row in rows
    ]
    return _empty_or_table(
        ["Factor", "Horizon", "配对数", "Kalshi 均值", "Polymarket 均值", "配对差异"],
        rendered,
    )


def render_comprehensive_html(data: Mapping[str, object]) -> str:
    """Render an entirely self-contained Chinese HTML report from report data."""

    if not isinstance(data, Mapping):
        raise ComprehensiveReportError("report data must be a mapping")
    if data.get("schema_version") != "NFLX13ComprehensiveReportDataV1":
        raise ComprehensiveReportError("report data schema_version is unsupported")
    if data.get("experiment_id") != "X-13":
        raise ComprehensiveReportError("report data experiment_id must be X-13")
    if data.get("claim_boundary") != _CLAIM_BOUNDARY:
        raise ComprehensiveReportError("report data claim boundary is unsupported")
    _evidence_descriptors(data.get("evidence_descriptors"))
    dense = _dense_path_rows(data.get("dense_paths"))
    matched = _matched_effect_rows(data.get("matched_effects"))
    reference = _reference_regression_rows(data.get("reference_regressions"))
    venue = _cross_venue_rows(data.get("cross_venue_rows"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-13 综合证据报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;
background:#f7f8fa;color:#172033;line-height:1.5}}main{{max-width:1320px;margin:0 auto;padding:28px}}
.boundary{{background:#fff8dd;border-left:5px solid #b78103;padding:12px 16px}}.muted{{color:#596579}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #d9dee6;padding:7px 9px;text-align:right;white-space:nowrap}}
th{{background:#eef1f5;text-align:left}}.table-wrap{{overflow:auto;border:1px solid #d9dee6}}.empty{{color:#596579;font-style:italic}}
</style></head><body><main>
<h1>NFL X-13 综合证据报告</h1>
<p class="boundary"><strong>结论边界：</strong>本页仅呈现已经产出的历史 source-time 证据；不构成 Alpha、延迟或执行主张。没有输入行的章节保持为空，绝不填造数据。</p>
<p class="muted">报告状态：{escape(_CLAIM_BOUNDARY)}。颜色仅编码 dense path 均值的方向与幅度。</p>
<h2>Dense paths 热图：均值 / 中位数 / 实质变化概率 / 支持</h2>
{_dense_html(dense)}
<h2>形状分布</h2>
{_shape_html(dense)}
<h2>Matched effects 与 Beta draws 摘要</h2>
{_matched_html(matched)}
<h2>Reference regression</h2>
{_reference_html(reference)}
<h2>配对场馆差异</h2>
{_venue_html(venue)}
</main></body></html>"""


__all__ = [
    "ComprehensiveReportError",
    "build_comprehensive_report_data",
    "render_comprehensive_html",
]
