"""Offline X-15 development workbench from verified publication objects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class X15WorkbenchError(RuntimeError):
    """The X-15 report input or immutable output is invalid."""


@dataclass(frozen=True, slots=True)
class X15Workbench:
    report_path: Path
    report_sha256: str
    source_manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ready(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ready(item) for item in value]
    return value


def _load_object(
    root: Path,
    manifest: dict[str, object],
    logical_name: str,
) -> pd.DataFrame:
    descriptor = next(
        (
            item
            for item in manifest.get("objects", [])
            if item.get("logical_name") == logical_name
        ),
        None,
    )
    if not isinstance(descriptor, dict):
        raise X15WorkbenchError(f"missing object: {logical_name}")
    relative = Path(str(descriptor["object_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise X15WorkbenchError(f"unsafe object path: {logical_name}")
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != descriptor["byte_length"]
        or _sha256(path) != descriptor["object_sha256"]
    ):
        raise X15WorkbenchError(f"object identity mismatch: {logical_name}")
    frame = pd.read_parquet(path)
    if len(frame) != descriptor["row_count"]:
        raise X15WorkbenchError(f"object row count mismatch: {logical_name}")
    return frame


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise X15WorkbenchError("content-addressed report collision")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise X15WorkbenchError("content-addressed report publish race")
    finally:
        temporary.unlink(missing_ok=True)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return _ready(frame.to_dict("records"))  # type: ignore[return-value]


def _require_frame_columns(
    frame: pd.DataFrame,
    *,
    logical_name: str,
    columns: set[str],
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise X15WorkbenchError(
            f"{logical_name} missing required columns: {missing}"
        )


def _prepare_workbench_tables(
    *,
    decision_landmarks: pd.DataFrame,
    signals: pd.DataFrame,
    signal_statistics: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build v2 workbench tables without treating censored rows as effects."""

    _require_frame_columns(
        decision_landmarks,
        logical_name="decision_landmarks",
        columns={
            "game_id",
            "episode_id",
            "venue",
            "landmark_seconds",
            "endpoint_seconds",
            "decision_eligible",
            "target_eligible",
        },
    )
    _require_frame_columns(
        signals,
        logical_name="signals",
        columns={
            "game_id",
            "episode_id",
            "venue",
            "model_id",
            "factor_id",
            "landmark_seconds",
            "endpoint_seconds",
            "gross_markout",
            "evaluation_status",
        },
    )
    _require_frame_columns(
        signal_statistics,
        logical_name="signal_statistics",
        columns={
            "venue",
            "model_id",
            "factor_id",
            "individual_statistical_gate",
            "cross_venue_same_sign",
            "passes_development_gate",
        },
    )
    _require_frame_columns(
        model_summary,
        logical_name="model_summary",
        columns={"venue", "model_id"},
    )
    statuses = set(signals["evaluation_status"].dropna().astype(str))
    allowed_statuses = {"EVALUATED", "CENSORED_EVALUATION"}
    if not statuses <= allowed_statuses:
        raise X15WorkbenchError(
            "signals contain unknown evaluation_status values"
        )

    signal_coverage = (
        signals.assign(
            _evaluated=signals["evaluation_status"].eq("EVALUATED"),
            _censored=signals["evaluation_status"].eq(
                "CENSORED_EVALUATION"
            ),
        )
        .groupby(["venue", "model_id"], as_index=False)
        .agg(
            total_signals=("episode_id", "size"),
            evaluated_signals=("_evaluated", "sum"),
            censored_signals=("_censored", "sum"),
            total_games=("game_id", "nunique"),
        )
        .sort_values(["venue", "model_id"], kind="mergesort")
    )
    evaluated = signals.loc[
        signals["evaluation_status"].eq("EVALUATED")
    ].copy()
    if not evaluated.empty:
        markout = pd.to_numeric(evaluated["gross_markout"], errors="coerce")
        if not np.isfinite(markout.to_numpy(dtype=float)).all():
            raise X15WorkbenchError(
                "EVALUATED signals require finite gross_markout"
            )
        evaluated["gross_markout"] = markout

    signal_event = (
        evaluated.groupby(
            ["venue", "model_id", "factor_id"], as_index=False
        )
        .agg(
            evaluated_signals=("episode_id", "size"),
            games=("game_id", "nunique"),
            hit_rate=(
                "gross_markout",
                lambda values: float((values > 0).mean()),
            ),
            mean_markout=("gross_markout", "mean"),
            median_markout=("gross_markout", "median"),
            p10_markout=(
                "gross_markout",
                lambda values: values.quantile(0.10),
            ),
            p90_markout=(
                "gross_markout",
                lambda values: values.quantile(0.90),
            ),
        )
        .sort_values(
            ["evaluated_signals", "games"],
            ascending=False,
            kind="mergesort",
        )
    )
    signal_time = (
        evaluated.groupby(
            [
                "venue",
                "model_id",
                "landmark_seconds",
                "endpoint_seconds",
            ],
            as_index=False,
        )
        .agg(
            evaluated_signals=("episode_id", "size"),
            games=("game_id", "nunique"),
            hit_rate=(
                "gross_markout",
                lambda values: float((values > 0).mean()),
            ),
            mean_markout=("gross_markout", "mean"),
            median_markout=("gross_markout", "median"),
        )
    )
    signal_distribution = (
        evaluated.groupby(["venue", "model_id"], as_index=False)
        .agg(
            evaluated_signals=("episode_id", "size"),
            games=("game_id", "nunique"),
            mean=("gross_markout", "mean"),
            standard_deviation=("gross_markout", "std"),
            p05=(
                "gross_markout",
                lambda values: values.quantile(0.05),
            ),
            p10=(
                "gross_markout",
                lambda values: values.quantile(0.10),
            ),
            p25=(
                "gross_markout",
                lambda values: values.quantile(0.25),
            ),
            median=("gross_markout", "median"),
            p75=(
                "gross_markout",
                lambda values: values.quantile(0.75),
            ),
            p90=(
                "gross_markout",
                lambda values: values.quantile(0.90),
            ),
            p95=(
                "gross_markout",
                lambda values: values.quantile(0.95),
            ),
            probability_positive=(
                "gross_markout",
                lambda values: float((values > 0).mean()),
            ),
            probability_negative=(
                "gross_markout",
                lambda values: float((values < 0).mean()),
            ),
            probability_abs_ge_1pp=(
                "gross_markout",
                lambda values: float((values.abs() >= 0.01).mean()),
            ),
            probability_abs_ge_2pp=(
                "gross_markout",
                lambda values: float((values.abs() >= 0.02).mean()),
            ),
            probability_abs_ge_5pp=(
                "gross_markout",
                lambda values: float((values.abs() >= 0.05).mean()),
            ),
        )
    )
    game_markouts = (
        evaluated.groupby(
            ["venue", "model_id", "game_id"], as_index=False
        )
        .agg(
            evaluated_signals=("episode_id", "size"),
            game_mean_markout=("gross_markout", "mean"),
        )
    )
    per_game_distribution = (
        game_markouts.groupby(["venue", "model_id"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            median_game_effect=("game_mean_markout", "median"),
            p10_game_effect=(
                "game_mean_markout",
                lambda values: values.quantile(0.10),
            ),
            p90_game_effect=(
                "game_mean_markout",
                lambda values: values.quantile(0.90),
            ),
            positive_game_fraction=(
                "game_mean_markout",
                lambda values: float((values > 0).mean()),
            ),
        )
    )
    histogram_rows: list[dict[str, object]] = []
    histogram_edges = np.linspace(-1.0, 1.0, 41)
    for (venue, model_id), group in evaluated.groupby(
        ["venue", "model_id"], sort=True
    ):
        counts, edges = np.histogram(
            group["gross_markout"].clip(-1.0, 1.0),
            bins=histogram_edges,
        )
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            histogram_rows.append(
                {
                    "venue": venue,
                    "model_id": model_id,
                    "bin_left": left,
                    "bin_right": right,
                    "count": count,
                }
            )
    signal_histogram = pd.DataFrame(
        histogram_rows,
        columns=[
            "venue",
            "model_id",
            "bin_left",
            "bin_right",
            "count",
        ],
    )
    landmark_coverage = (
        decision_landmarks.groupby(
            ["venue", "landmark_seconds", "endpoint_seconds"],
            as_index=False,
        )
        .agg(
            decision_rows=("episode_id", "size"),
            games=("game_id", "nunique"),
            target_eligible_rows=("target_eligible", "sum"),
        )
    )
    return {
        "model_summary": model_summary,
        "signal_coverage": signal_coverage,
        "signal_statistics": signal_statistics,
        "signal_event": signal_event,
        "signal_time": signal_time,
        "signal_distribution": signal_distribution,
        "signal_histogram": signal_histogram,
        "per_game_distribution": per_game_distribution,
        "landmark_coverage": landmark_coverage,
    }


def render_x15_workbench(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    notebook_url: str = (
        "http://127.0.0.1:8890/lab/tree/"
        "notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb"
    ),
) -> X15Workbench:
    """Render one self-contained HTML from verified X-15 artifacts."""

    manifest_file = Path(manifest_path)
    manifest_sha = _sha256(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "nfl_x15_development_publication_v2"
        or manifest.get("experiment_id") != "X-15"
        or manifest.get("development_game_count") != 153
        or manifest.get("holdout_state") != "SEALED_UNREAD"
        or manifest.get("holdout_reaction_accessed") is not False
    ):
        raise X15WorkbenchError("X-15 development/holdout contract mismatch")
    source_root = manifest_file.parents[3]
    decision_landmarks = _load_object(
        source_root, manifest, "decision_landmarks"
    )
    signals = _load_object(source_root, manifest, "signals")
    signal_statistics = _load_object(
        source_root, manifest, "signal_statistics"
    )
    model_summary = _load_object(source_root, manifest, "model_summary")
    tables = _prepare_workbench_tables(
        decision_landmarks=decision_landmarks,
        signals=signals,
        signal_statistics=signal_statistics,
        model_summary=model_summary,
    )
    payload = {
        "audit": {
            key: value
            for key, value in manifest.items()
            if key
            in {
                "experiment_id",
                "result_label",
                "claim_boundary",
                "development_game_count",
                "landmark_rows",
                "decision_landmark_rows",
                "target_eligible_landmark_rows",
                "decision_episode_count",
                "target_eligible_episode_count",
                "oof_prediction_rows",
                "signal_rows",
                "evaluated_signal_rows",
                "censored_signal_rows",
                "holdout_game_count",
                "holdout_state",
                "holdout_reaction_accessed",
                "batch_index_sha256",
                "reference_manifest_sha256",
                "matched_manifest_sha256",
                "feature_columns",
            }
        },
        "manifest_path": str(manifest_file.resolve()),
        "manifest_sha256": manifest_sha,
        "notebook_url": notebook_url,
        **{
            logical_name: _records(frame)
            for logical_name, frame in tables.items()
        },
    }
    data = json.dumps(
        _ready(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-15 153场 Taker Research Workbench</title><style>
:root{{--bg:#f2eee5;--paper:#fffdf8;--ink:#17231d;--muted:#66736b;--line:#d6d0c3;--green:#12634b;--red:#a94137;--amber:#9a6818}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}}header{{padding:28px 36px;background:#e5dfd2;border-bottom:1px solid var(--line)}}h1{{font:700 30px/1.15 Georgia,serif;margin:6px 0}}main{{padding:20px 30px 60px}}.boundary{{border-left:4px solid var(--amber);padding:11px 14px;background:#fff5dc;max-width:1100px}}.controls{{position:sticky;top:0;z-index:3;background:rgba(242,238,229,.96);display:flex;gap:12px;flex-wrap:wrap;padding:12px 0;border-bottom:1px solid var(--line)}}select{{padding:7px;background:var(--paper);border:1px solid var(--line)}}.tabs{{display:flex;overflow:auto;border-bottom:1px solid var(--line);margin:18px 0}}button{{border:0;border-bottom:3px solid transparent;background:transparent;padding:10px 13px;font-weight:700;color:var(--muted)}}button.active{{color:var(--green);border-color:var(--green)}}section{{display:none}}section.active{{display:block}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card,.panel{{background:var(--paper);border:1px solid var(--line);padding:14px}}.value{{font:700 24px Georgia,serif}}.table{{max-height:680px;overflow:auto;border:1px solid var(--line);background:var(--paper)}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:7px 9px;border-bottom:1px solid #e6e0d5;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#ece7dc;font-size:11px;text-transform:uppercase}}th:first-child,td:first-child{{text-align:left}}.heat td{{text-align:center;min-width:66px}}.small{{color:var(--muted);font-size:12px}}a{{color:var(--green)}}code{{font-size:11px}}
</style><header><div>NFL FACTOR LAB · X-15 · DEVELOPMENT ONLY</div><h1>153 场 Multi-Landmark Taker Research</h1><p>真实历史成交 × 1/2/3/5/10 秒观察点 × 5–60 秒终点 × venue-separated walk-forward。</p><p><a id="notebook" href="{notebook_url}">打开 Master Notebook</a></p></header>
<main><div class="boundary"><strong>不是可执行回测：</strong>历史成交 mark 不是当时可成交 bid/ask；未计 fees/slippage；81 场 holdout 仍未读取。所有 effect 与 distribution 只使用 EVALUATED 信号，CENSORED_EVALUATION 仅计覆盖、不进入效果。</div>
<div class="controls">Venue <select id="venue"></select>Model <select id="model"></select></div>
<nav class="tabs"><button class="active" data-tab="overview">Overview</button><button data-tab="factors">Event/Factor</button><button data-tab="time">Landmark×Endpoint</button><button data-tab="distribution">Distribution</button><button data-tab="diagnostics">Diagnostics</button><button data-tab="audit">Audit</button></nav>
<section id="overview" class="active"></section><section id="factors"></section><section id="time"></section><section id="distribution"></section><section id="diagnostics"></section><section id="audit"></section></main>
<script id="data" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('data').textContent),S={{venue:'kalshi',model:'multinomial_logistic_v1'}},$=x=>document.querySelector(x),uniq=a=>[...new Set(a)].sort();
function fmt(v){{if(v===null||v===undefined)return '—';if(typeof v==='number')return Math.abs(v)<1?v.toFixed(4):v.toFixed(2);return String(v)}}function table(rows){{if(!rows.length)return '<p class="small">无符合行。</p>';const cols=Object.keys(rows[0]);return '<div class="table"><table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+fmt(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}}
function rows(x){{return x.filter(r=>(!r.venue||r.venue===S.venue)&&(!r.model_id||r.model_id===S.model))}}function color(v,m){{if(v===null||!Number.isFinite(+v))return '#eee';const a=Math.min(.9,Math.abs(+v)/(m||1));return +v>=0?`rgba(18,99,75,${{.12+a*.62}})`:`rgba(169,65,55,${{.12+a*.62}})`}}
function heat(data,value){{const r=rows(data),ls=uniq(r.map(x=>+x.landmark_seconds)),hs=uniq(r.map(x=>+x.endpoint_seconds)).sort((a,b)=>a-b),m=Math.max(...r.map(x=>Math.abs(+x[value]||0)),.001);return '<div class="table"><table class="heat"><thead><tr><th>L\\\\H</th>'+hs.map(h=>`<th>${{h}}s</th>`).join('')+'</tr></thead><tbody>'+ls.map(l=>'<tr><td>'+l+'s</td>'+hs.map(h=>{{const x=r.find(y=>+y.landmark_seconds===l&&+y.endpoint_seconds===h),v=x?.[value];return `<td style="background:${{color(v,m)}}" title="evaluated n=${{x?.evaluated_signals||0}}">${{v==null?'—':(+v*100).toFixed(2)+'pp'}}</td>`}}).join('')+'</tr>').join('')+'</tbody></table></div>'}}
function histogram(){{const r=rows(D.signal_histogram),w=920,h=280,p=36,m=Math.max(...r.map(x=>+x.count),1),bw=(w-p*2)/Math.max(r.length,1);return `<svg viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="gross markout histogram" style="width:100%;background:#fffdf8;border:1px solid #d6d0c3">${{r.map((x,i)=>{{const bh=(+x.count/m)*(h-p*2),xx=p+i*bw,yy=h-p-bh,c=(+x.bin_right<=0?'#a94137':'#12634b');return `<rect x="${{xx}}" y="${{yy}}" width="${{Math.max(1,bw-1)}}" height="${{bh}}" fill="${{c}}"><title>${{(+x.bin_left*100).toFixed(1)}}pp…${{(+x.bin_right*100).toFixed(1)}}pp: ${{x.count}}</title></rect>`}}).join('')}}<line x1="${{p}}" y1="${{h-p}}" x2="${{w-p}}" y2="${{h-p}}" stroke="#66736b"/><text x="${{p}}" y="${{h-8}}" font-size="12">−100pp</text><text x="${{w/2-14}}" y="${{h-8}}" font-size="12">0</text><text x="${{w-p-34}}" y="${{h-8}}" font-size="12">+100pp</text></svg>`}}
function render(){{const s=rows(D.model_summary),a=D.audit;$('#overview').innerHTML='<h2>关键结论表</h2>'+table(s)+'<div class="cards">'+[['Decision landmarks',a.decision_landmark_rows],['Total signals',a.signal_rows],['Evaluated',a.evaluated_signal_rows],['Censored',a.censored_signal_rows],['Holdout',a.holdout_state]].map(x=>`<div class="card">${{x[0]}}<div class="value">${{fmt(x[1])}}</div></div>`).join('')+'<h2>Signal coverage</h2>'+table(rows(D.signal_coverage));$('#factors').innerHTML='<h2>仅 EVALUATED 的 event/factor effect</h2><p class="small">hit/markout 是 OOF development actual-trade mark；不是 fill P&L。Censored 不参与任何效果或分布。</p>'+table(rows(D.signal_event))+'<h2>Development statistical gates</h2><p class="small">明确展示 individual_statistical_gate、cross_venue_same_sign 与 passes_development_gate。</p>'+table(rows(D.signal_statistics));$('#time').innerHTML='<h2>仅 EVALUATED 的 mean markout 热点图</h2>'+heat(D.signal_time,'mean_markout')+'<h2>精确数值</h2>'+table(rows(D.signal_time))+'<h2>Decision / target availability（覆盖，不是 effect）</h2>'+table(rows(D.landmark_coverage));$('#distribution').innerHTML='<h2>仅 EVALUATED 的 gross markout 经验分布</h2><p class="small">横轴为信号方向下的后续实际成交变化；绿色为正、红色为负。端点压缩在 ±100pp。</p>'+histogram()+'<h2>Episode-level quantiles / threshold probabilities</h2>'+table(rows(D.signal_distribution))+'<h2>Per-game distribution</h2>'+table(rows(D.per_game_distribution));$('#diagnostics').innerHTML='<h2>Model summary</h2>'+table(rows(D.model_summary))+'<h2>Total / evaluated / censored</h2>'+table(rows(D.signal_coverage))+'<h2>Statistical gate details</h2>'+table(rows(D.signal_statistics));$('#audit').innerHTML='<h2>Lineage / Claim boundary</h2><pre>'+JSON.stringify(D.audit,null,2)+'</pre><p><code>'+D.manifest_sha256+'</code><br><code>'+D.manifest_path+'</code></p><h2>Decision landmark coverage</h2>'+table(rows(D.landmark_coverage))}}
const venues=uniq(D.model_summary.map(x=>x.venue)),models=uniq(D.model_summary.map(x=>x.model_id));$('#venue').innerHTML=venues.map(x=>`<option ${{x===S.venue?'selected':''}}>${{x}}</option>`).join('');$('#model').innerHTML=models.map(x=>`<option ${{x===S.model?'selected':''}}>${{x}}</option>`).join('');$('#venue').onchange=e=>{{S.venue=e.target.value;render()}};$('#model').onchange=e=>{{S.model=e.target.value;render()}};document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button[data-tab],section').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.tab).classList.add('active')}});render();
</script></html>""".encode("utf-8")
    digest = "sha256:" + hashlib.sha256(html).hexdigest()
    hexadecimal = digest.removeprefix("sha256:")
    target = (
        Path(output_root)
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.html"
    )
    _atomic(target, html)
    return X15Workbench(
        report_path=target,
        report_sha256=digest,
        source_manifest_sha256=manifest_sha,
    )


__all__ = [
    "X15Workbench",
    "X15WorkbenchError",
    "render_x15_workbench",
]
