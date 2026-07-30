#!/usr/bin/env python3
"""Render the canonical NFL X-15 regulation-model decision report.

The report is deliberately downstream-only.  It verifies and reads the
content-addressed Stage-A, PanelV4, exact-45 OOF, SelectionLock, and native
rule-audit publications.  It never reads holdout reaction data and never
re-fits a model.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pyarrow.parquet as pq

from prediction_market.research.nfl_x15_native_rule_audit import (
    verify_rule_audit_batch,
)
from prediction_market.research.nfl_x15_regulation_models import (
    POLYMARKET,
    discover_verified_regulation_oof_shards,
    load_verified_regulation_panel_batch,
    verify_regulation_selection_lock,
)


DEFAULT_OUTPUT_ROOT = Path(
    "reports/nfl-factor-lab/exact-153/regulation-decision-v1"
)
CLAIM_BOUNDARY = (
    "DEVELOPMENT_HISTORICAL_ACTUAL_TRADES_ONLY;"
    "SOURCE_TIME_PROBABILITY_DIAGNOSTIC;"
    "NO_LIVE_LATENCY;NO_CAUSALITY;NO_EXECUTION;NO_ALPHA_CLAIM"
)
NOTEBOOK_URL = (
    "http://127.0.0.1:8890/lab/tree/"
    "notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb"
)


class ReportError(RuntimeError):
    """A report input is not byte-exact or violates the frozen contract."""


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _load_json(
    project_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    schema: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_relative_to(project_root)
        or resolved.is_symlink()
        or not resolved.is_file()
        or _sha256_file(resolved) != expected_sha256
    ):
        raise ReportError(f"input failed byte verification: {resolved}")
    try:
        value = json.loads(resolved.read_text("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"input is not strict JSON: {resolved}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ReportError(f"input schema mismatch: {resolved}")
    if value.get("holdout_reaction_accessed") is not False:
        raise ReportError(f"holdout access is not false: {resolved}")
    return value


def _descriptor(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    descriptors = [
        item
        for item in document.get("calibration_tables", [])
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(descriptors) != 1:
        raise ReportError(f"missing unique table descriptor: {name}")
    return descriptors[0]


def _verified_parquet(
    root: Path,
    descriptor: Mapping[str, Any],
):
    path = (root / str(descriptor["object_path"])).resolve()
    if (
        not path.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != descriptor["object_sha256"]
    ):
        raise ReportError(f"Parquet failed byte verification: {path}")
    table = pq.read_table(path)
    if (
        table.num_rows != descriptor["row_count"]
        or table.column_names != descriptor["schema_columns"]
    ):
        raise ReportError(f"Parquet descriptor mismatch: {path}")
    return table.to_pandas()


def _atomic_content_addressed_write(
    output_root: Path,
    payload: bytes,
    suffix: str,
    *,
    namespace: str = "objects",
) -> tuple[Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    target = (
        output_root
        / namespace
        / "sha256"
        / digest[:2]
        / f"{digest}.{suffix}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise ReportError(f"existing content-addressed object differs: {target}")
        return target, f"sha256:{digest}"
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp-report-", dir=target.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target, f"sha256:{digest}"


def _number(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _signed(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def _heat(value: float, scale: float = 0.05) -> str:
    bounded = max(-1.0, min(1.0, value / scale))
    if bounded >= 0:
        alpha = 0.10 + 0.55 * bounded
        return f"background:rgba(45,155,96,{alpha:.3f})"
    alpha = 0.10 + 0.55 * abs(bounded)
    return f"background:rgba(207,70,65,{alpha:.3f})"


def _stage_a_summary(
    project_root: Path,
    stage_a_path: Path,
    stage_a_sha256: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    document = _load_json(
        project_root,
        stage_a_path,
        stage_a_sha256,
        schema="nfl_x15_stage_a_batch_index_v1",
    )
    if (
        document.get("publication_gate") != "PASS"
        or document.get("cohort") != "development"
        or document.get("game_count") != 153
        or document.get("market_data_read") is not False
    ):
        raise ReportError("Stage A is not the canonical development publication")
    output_root = stage_a_path.resolve().parents[4]
    frame = _verified_parquet(
        output_root, _descriptor(document, "calibration_summary")
    )
    if len(frame) != 1:
        raise ReportError("Stage A summary must contain one row")
    row = frame.iloc[0]
    bootstrap = _verified_parquet(
        output_root, _descriptor(document, "calibration_bootstrap")
    )
    bootstrap_by_metric = {
        str(item.metric): item for item in bootstrap.itertuples(index=False)
    }
    if set(bootstrap_by_metric) != {"brier", "log_loss"}:
        raise ReportError("Stage A bootstrap metric inventory is not exact")
    metrics = {
        "evaluated_games": int(row["evaluated_games"]),
        "evaluated_states": int(row["evaluated_states"]),
        "brier": float(row["brier"]),
        "brier_ci_low": float(bootstrap_by_metric["brier"].ci_low),
        "brier_ci_high": float(bootstrap_by_metric["brier"].ci_high),
        "log_loss": float(row["log_loss"]),
        "log_loss_ci_low": float(bootstrap_by_metric["log_loss"].ci_low),
        "log_loss_ci_high": float(bootstrap_by_metric["log_loss"].ci_high),
        "calibration_slope": float(row["calibration_slope"]),
        "calibration_intercept": float(row["calibration_intercept"]),
    }
    return document, metrics


def _fold_metric_improvements(
    oof_root: Path,
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    for shard in selection["input_shards"]:
        manifest_path = Path(str(shard["manifest_path"])).resolve()
        if (
            not manifest_path.is_relative_to(oof_root)
            or _sha256_file(manifest_path) != shard["manifest_file_sha256"]
        ):
            raise ReportError("SelectionLock shard inventory is not byte-exact")
        manifest = json.loads(manifest_path.read_text("ascii"))
        cell = manifest["cell"]
        key = (
            str(cell["fold_id"]),
            str(cell["model_id"]),
            str(cell["feature_block_id"]),
        )
        objects = [
            item
            for item in manifest["objects"]
            if item["logical_name"] == "fold_metrics"
        ]
        if len(objects) != 1:
            raise ReportError(f"{key} has no unique fold metrics")
        descriptor = objects[0]
        path = (oof_root / descriptor["object_path"]).resolve()
        if (
            _sha256_file(path) != descriptor["object_sha256"]
            or pq.read_metadata(path).num_rows != descriptor["row_count"]
        ):
            raise ReportError(f"{key} metric object failed verification")
        frame = pq.read_table(path).to_pandas()
        metrics[key] = {
            str(row.metric_name): float(row.metric_value)
            for row in frame.itertuples(index=False)
        }
    result: list[dict[str, Any]] = []
    candidates = [
        ("regularized_logistic_v1", block)
        for block in ("D1", "D2", "D3")
    ] + [
        ("shallow_xgboost_v1", block)
        for block in ("D1", "D2", "D3")
    ]
    for model_id, block in candidates:
        row: dict[str, Any] = {
            "model_id": model_id,
            "feature_block_id": block,
        }
        for fold in [f"fold_{number:02d}" for number in range(1, 6)]:
            baseline = metrics[(fold, "b0_empirical_v1", "D0")]
            candidate = metrics[(fold, model_id, block)]
            row[fold] = (
                baseline["joint_log_loss"] - candidate["joint_log_loss"]
            )
        result.append(row)
    return result


def _oof_sample_summary(
    selection: Mapping[str, Any],
) -> dict[str, int]:
    """Separate repeated model-cell rows from unique source-risk-set rows."""

    accumulated_rows = 0
    baseline_rows = 0
    baseline_folds: set[str] = set()
    for shard in selection["input_shards"]:
        manifest_path = Path(str(shard["manifest_path"])).resolve()
        if _sha256_file(manifest_path) != shard["manifest_file_sha256"]:
            raise ReportError("OOF sample summary shard is not byte-exact")
        manifest = json.loads(manifest_path.read_text("ascii"))
        cell = manifest.get("cell")
        objects = [
            item
            for item in manifest.get("objects", [])
            if item.get("logical_name") == "oof_predictions"
        ]
        if not isinstance(cell, dict) or len(objects) != 1:
            raise ReportError("OOF sample summary shard contract mismatch")
        row_count = int(objects[0]["row_count"])
        accumulated_rows += row_count
        if (
            cell.get("model_id") == "b0_empirical_v1"
            and cell.get("feature_block_id") == "D0"
        ):
            fold_id = str(cell["fold_id"])
            if fold_id in baseline_folds:
                raise ReportError("duplicate B0 fold in OOF sample summary")
            baseline_folds.add(fold_id)
            baseline_rows += row_count
    expected_folds = {f"fold_{number:02d}" for number in range(1, 6)}
    if baseline_folds != expected_folds:
        raise ReportError("OOF sample summary does not contain exact B0 folds")
    validation_games = {
        str(game_id)
        for fold in selection["fold_definitions"].values()
        for game_id in fold["validation_game_ids"]
    }
    anchor_game_counts = {
        int(candidate["anchor_game_count"])
        for candidate in selection["candidate_audit"]
    }
    if len(anchor_game_counts) != 1:
        raise ReportError("candidate anchor-game counts disagree")
    return {
        "accumulated_cell_rows": accumulated_rows,
        "unique_oof_source_rows": baseline_rows,
        "oof_game_count": len(validation_games),
        "anchor_gate_game_count": anchor_game_counts.pop(),
    }


def _candidate_table_rows(candidate_audit: list[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in candidate_audit:
        joint = float(item["integrated_joint_mean_improvement"])
        joint_low = float(item["integrated_joint_ci_low"])
        availability_low = float(item["availability_log_loss_ci_low"])
        direction_low = float(item["direction_log_loss_ci_low"])
        passed = bool(item["candidate_gate_passed"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['model_id']))}</td>"
            f"<td>{html.escape(str(item['feature_block_id']))}</td>"
            f"<td>{int(item['game_count'])}</td>"
            f"<td style='{_heat(joint)}'>{_signed(joint)}</td>"
            f"<td style='{_heat(joint_low)}'>{_signed(joint_low)}</td>"
            f"<td style='{_heat(availability_low)}'>"
            f"{_signed(availability_low)}</td>"
            f"<td style='{_heat(direction_low)}'>{_signed(direction_low)}</td>"
            f"<td>{'PASS' if item['anchor_gate_passed'] else 'FAIL'}</td>"
            f"<td class='gate {'pass' if passed else 'fail'}'>"
            f"{'PASS' if passed else 'FAIL'}</td>"
            "</tr>"
        )
    return "".join(rows)


def _fold_heatmap_rows(rows: list[Mapping[str, Any]]) -> str:
    output: list[str] = []
    for row in rows:
        cells = "".join(
            f"<td style='{_heat(float(row[fold]))}'>"
            f"{_signed(row[fold])}</td>"
            for fold in [f"fold_{number:02d}" for number in range(1, 6)]
        )
        output.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_id']))}</td>"
            f"<td>{html.escape(str(row['feature_block_id']))}</td>"
            f"{cells}</tr>"
        )
    return "".join(output)


def _render_html(
    *,
    stage_a_metrics: Mapping[str, float],
    panel_document: Mapping[str, Any],
    oof_document: Mapping[str, Any],
    selection: Mapping[str, Any],
    rule_document: Mapping[str, Any],
    fold_rows: list[Mapping[str, Any]],
    oof_summary: Mapping[str, int],
    input_hashes: Mapping[str, str],
) -> bytes:
    candidates = selection["candidate_audit"]
    candidate_rows = _candidate_table_rows(candidates)
    fold_heatmap = _fold_heatmap_rows(fold_rows)
    counts = panel_document["counts"]
    decision = str(selection["decision_status"])
    selection_status = str(selection["selection_result_status"])
    style = """
    :root{--ink:#18201d;--muted:#65716b;--paper:#f6f2e8;--card:#fffdf7;
      --line:#d8d0bf;--green:#267a50;--red:#a23c38;--amber:#9a6a16}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
      font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",
      sans-serif;line-height:1.45}.wrap{max-width:1500px;margin:auto;padding:28px}
    h1,h2,h3{font-family:Georgia,"Times New Roman",serif;margin:.2em 0 .6em}
    h1{font-size:2.25rem}.sub{color:var(--muted);max-width:1050px}
    .hero{display:grid;grid-template-columns:1.5fr 1fr;gap:18px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:20px;box-shadow:0 8px 24px rgba(40,35,20,.05)}
    .decision{font-size:1.45rem;font-weight:800;color:var(--red)}
    .chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
    .chip{border:1px solid var(--line);border-radius:999px;padding:5px 10px;
      font-size:.85rem;background:#fff}.chip.good{color:var(--green)}
    .chip.bad{color:var(--red)}.metrics{display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
    .metric{padding:12px;border-left:3px solid var(--green);background:#faf7ef}
    .metric b{display:block;font-size:1.25rem}.metric span{color:var(--muted);
      font-size:.8rem}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0 12px}
    button{border:1px solid var(--line);background:#fff;padding:9px 13px;
      border-radius:9px;cursor:pointer}button.active{background:var(--ink);color:white}
    .tab{display:none}.tab.active{display:block}table{width:100%;
      border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.88rem}
    th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:right}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
    th{position:sticky;top:0;background:#ede7d9;z-index:1}.scroll{overflow:auto;
      max-height:690px;border:1px solid var(--line);border-radius:10px}
    .gate{font-weight:800}.gate.pass{color:var(--green)}.gate.fail{color:var(--red)}
    .note{border-left:4px solid var(--amber);padding:10px 14px;background:#fff7df}
    .flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
    .flow div{padding:12px;border:1px solid var(--line);background:#fff;
      border-radius:9px;text-align:center}.hash{font-family:ui-monospace,SFMono-Regular,
      Menlo,monospace;font-size:.74rem;word-break:break-all}.two{display:grid;
      grid-template-columns:1fr 1fr;gap:16px}a{color:#195d82}
    @media(max-width:900px){.hero,.two{grid-template-columns:1fr}
      .metrics{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr}}
    """
    script = """
    const buttons=[...document.querySelectorAll('[data-tab]')];
    buttons.forEach(b=>b.addEventListener('click',()=>{
      buttons.forEach(x=>x.classList.toggle('active',x===b));
      document.querySelectorAll('.tab').forEach(x=>
        x.classList.toggle('active',x.id===b.dataset.tab));
    }));
    """
    input_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='hash'>{html.escape(value)}</td></tr>"
        for name, value in input_hashes.items()
    )
    document = f"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-15 153场两阶段概率研究</title><style>{style}</style></head><body>
<main class="wrap">
  <div class="hero">
    <section class="card"><div class="sub">NFL X-15 · exact-153 development</div>
      <h1>比赛事实 → 合理胜率 → 市场反应概率</h1>
      <p class="sub">这是当前 canonical、holdout-blind 的两阶段研究结论。
      Stage A 是冻结 no-spread 模型在 153 场上的 retrospective diagnostic；
      Stage B 才是按比赛等权、按周向前的 OOF 验证，预测事件后是否出现新成交，
      以及有成交时价格方向。</p>
      <div class="chips"><span class="chip good">153 games</span>
      <span class="chip good">{int(oof_summary['oof_game_count'])} OOF games</span>
      <span class="chip good">{int(oof_summary['unique_oof_source_rows']):,}
      unique OOF source rows</span>
      <span class="chip bad">Holdout reads: 0</span></div>
      <a href="{NOTEBOOK_URL}">打开 Master Notebook（localhost:8890）</a>
    </section>
    <aside class="card"><h2>正式决策</h2>
      <div class="decision">{html.escape(decision)}</div>
      <p><b>{html.escape(selection_status)}</b></p>
      <p>本次冻结的六个候选中 0 个同时通过 joint、availability、direction
      和 anchor 门槛；结论只是否定这六个候选在当前 target 上的晋级，
      不否定其他 factor 或未来预注册模型。81场 holdout 保持封存。</p>
    </aside>
  </div>

  <div class="tabs">
    <button data-tab="state" class="active">State / 当前状态</button>
    <button data-tab="selection">Model gates / 模型门</button>
    <button data-tab="folds">Fold heatmap / 逐折</button>
    <button data-tab="stagea">Stage A / 胜率</button>
    <button data-tab="kalshi">Kalshi / 规则与验证</button>
    <button data-tab="audit">Audit / 谱系</button>
  </div>

  <section id="state" class="tab active card">
    <h2>全链路状态</h2>
    <div class="flow"><div>153场 FactsV4<br><b>PASS</b></div>
      <div>PIA + Prestate<br><b>PASS</b></div>
      <div>PanelV4<br><b>PASS</b></div>
      <div>45-cell OOF<br><b>PASS</b></div>
      <div>Selection<br><b>NO ADVANCE</b></div></div>
    <h3>Panel 真实样本</h3>
    <div class="metrics">
      <div class="metric"><b>{int(counts['panel_row_count']):,}</b>
        <span>Panel rows（anchor × L × H 展开）</span></div>
      <div class="metric"><b>{int(counts['loss_eligible_count']):,}</b>
        <span>Loss eligible</span></div>
      <div class="metric"><b>{int(counts['availability_one_count']):,}</b>
        <span>Availability = 1</span></div>
      <div class="metric"><b>{int(counts['availability_zero_count']):,}</b>
        <span>Availability = 0</span></div>
    </div>
    <div class="note"><b>大白话：</b>我们已经能回答“这个事件之后，在给定时间窗
    内有没有新的实际成交”以及“有成交时价格向上/不变/向下的概率”。
    当前复杂模型主要学会了成交是否出现，却没有稳定提高方向判断。</div>
    <p><b>独立单位：</b>153 场 development cohort；其中
    {int(oof_summary['oof_game_count'])} 场进入 OOF 验证，
    {int(oof_summary['anchor_gate_game_count'])} 场进入 anchor gate。
    {int(oof_summary['accumulated_cell_rows']):,} 是 45 个模型 cell 的累计行数，
    同一 source-risk-set 会跨模型重复；去重后是
    {int(oof_summary['unique_oof_source_rows']):,} 条 OOF source rows，
    统计推断仍以 game 为 cluster。</p>
  </section>

  <section id="selection" class="tab card">
    <h2>六个候选的 10,000 次 game-cluster bootstrap 门</h2>
    <p>下表都是 <b>B0 loss − candidate loss</b>：正数代表候选更好。
    只有所有强制门均通过才可晋升。</p>
    <div class="scroll"><table><thead><tr><th>Model</th><th>Block</th>
      <th>Games</th><th>Joint mean</th><th>Joint CI low</th>
      <th>Avail LL CI low</th><th>Direction LL CI low</th>
      <th>Anchor</th><th>Final gate</th></tr></thead>
      <tbody>{candidate_rows}</tbody></table></div>
    <div class="two" style="margin-top:16px">
      <div class="note"><b>Logistic D1–D3：</b>joint 与 availability 确有改善，
      但 direction 的置信区间下界为负，说明跨比赛不够稳定。</div>
      <div class="note"><b>XGBoost：</b>D3 的 direction 有改善，但牺牲了
      availability 与整体 joint；D1/D2 更弱。</div>
    </div>
    <h3>Target 与 feature block 的精确定义</h3>
    <p><b>Availability：</b>已有 landmark actual trade、连续性有效且未被下一信息
    事件等 censor 后，区间 <code>(T+L, T+H]</code> 是否出现新的实际成交。
    <code>availability=0</code> 不是“盘口不可交易”。</p>
    <p><b>Direction：</b>仅在出现新实际成交时，比较 HOME_TEAM_OUTCOME 的
    最新同秒 size-weighted trade mark 与 landmark mark；变化 ≥+1pp 为 UP，
    ≤−1pp 为 DOWN，中间为 NO_MOVE。没有成交绝不记为 NO_MOVE。</p>
    <p><b>D0</b>：L/H 时间网格基线；<b>D1</b>：加入 landmark 价格、
    staleness 与 30/60 秒成交活跃度；<b>D2</b>：再加入比赛 pre-state 与
    <code>p_before</code>；<b>D3</b>：再加入 provisional event 语义。</p>
  </section>

  <section id="folds" class="tab card">
    <h2>逐折 joint log-loss 改善热力图</h2>
    <p>绿色为优于同折 B0，红色为差于 B0。逐折变化说明为什么不能只看总均值。</p>
    <div class="scroll"><table><thead><tr><th>Model</th><th>Block</th>
      <th>Fold 1</th><th>Fold 2</th><th>Fold 3</th><th>Fold 4</th>
      <th>Fold 5</th></tr></thead><tbody>{fold_heatmap}</tbody></table></div>
  </section>

  <section id="stagea" class="tab card">
    <h2>Stage A：参考胜率模型</h2>
    <div class="metrics">
      <div class="metric"><b>{_number(stage_a_metrics['brier'], 6)}</b>
        <span>Brier（95% CI {_number(stage_a_metrics['brier_ci_low'], 6)}
        – {_number(stage_a_metrics['brier_ci_high'], 6)}）</span></div>
      <div class="metric"><b>{_number(stage_a_metrics['log_loss'], 6)}</b>
        <span>Log loss（95% CI {_number(stage_a_metrics['log_loss_ci_low'], 6)}
        – {_number(stage_a_metrics['log_loss_ci_high'], 6)}）</span></div>
      <div class="metric"><b>{_number(stage_a_metrics['calibration_slope'], 4)}</b>
        <span>Calibration slope（理想 1）</span></div>
      <div class="metric"><b>{_signed(stage_a_metrics['calibration_intercept'], 4)}</b>
        <span>Calibration intercept（理想 0）</span></div>
    </div>
    <p>覆盖 {int(stage_a_metrics['evaluated_games'])} 场、
    {int(stage_a_metrics['evaluated_states']):,} 个支持状态；CI 来自 10,000 次
    game-cluster bootstrap。</p>
    <p>冻结的 nflfastR/fastrmodels no-spread XGBoost 是
    <b>retrospective diagnostic reference</b>，不是市场真值，也不是
    live-PIT latency 证据。它帮助把“同为 INT/TD 但比赛状态不同”的影响统一到
    胜率概率点上。</p>
  </section>

  <section id="kalshi" class="tab card">
    <h2>Kalshi：现在能说什么</h2>
    <div class="metrics">
      <div class="metric"><b>{int(rule_document['completed_non_tie_comparable_count'])}/153</b>
        <span>Completed non-tie 可比</span></div>
      <div class="metric"><b>{int(rule_document['strict_rule_equivalent_count'])}/153</b>
        <span>严格规则等价</span></div>
      <div class="metric"><b>0</b><span>Holdout reaction reads</span></div>
      <div class="metric"><b>NOT APPLICABLE</b><span>Candidate transport</span></div>
    </div>
    <p>已证明 153 场完成且非平局时，两 venue 的实际胜者命题可作
    <b>描述性对照</b>；但 152 场存在 tie-clause 差异，另 1 场规则仍
    unproven，所以不能声称严格合约等价。</p>
    <div class="note"><b>两件事分开：</b>Gate 2 因规则不等价而禁止 formal
    contract transport；更早一步，SelectionLock 没有 winner，所以本轮
    candidate transport 根本未运行（NOT_APPLICABLE），不能写成一次失败的
    Kalshi 验证，也不能把 Kalshi 变成第二个 development set。</div>
  </section>

  <section id="audit" class="tab card">
    <h2>不可变谱系与边界</h2>
    <table><thead><tr><th>Artifact</th><th>SHA-256</th></tr></thead>
      <tbody>{input_rows}</tbody></table>
    <p class="hash">Claim boundary: {html.escape(CLAIM_BOUNDARY)}</p>
    <p>不声称真实 latency、因果、bid/ask、L2、fill、执行性或可交易 alpha；
    不读取 81 场 final holdout；不进行真实资金、maker、hedging 或下单。</p>
  </section>
</main><script>{script}</script></body></html>"""
    return document.encode("utf-8")


def render_report(
    *,
    project_root: Path,
    stage_a_manifest: Path,
    stage_a_sha256: str,
    panel_manifest: Path,
    panel_sha256: str,
    oof_manifest: Path,
    oof_sha256: str,
    selection_manifest: Path,
    selection_sha256: str,
    rule_audit_manifest: Path,
    rule_audit_sha256: str,
    output_root: Path,
) -> tuple[Path, Path]:
    project = project_root.resolve()
    stage_a_document, stage_a_metrics = _stage_a_summary(
        project, stage_a_manifest.resolve(), stage_a_sha256
    )
    panel_document = _load_json(
        project,
        panel_manifest.resolve(),
        panel_sha256,
        schema="RegulationDecisionPanelBatchManifestV2",
    )
    oof_document = _load_json(
        project,
        oof_manifest.resolve(),
        oof_sha256,
        schema="RegulationStageBOOFBatchManifestV1",
    )
    selection_document = _load_json(
        project,
        selection_manifest.resolve(),
        selection_sha256,
        schema="RegulationStageBSelectionLockV2",
    )
    rule_document = _load_json(
        project,
        rule_audit_manifest.resolve(),
        rule_audit_sha256,
        schema="VenueNativeRuleAuditBatchManifestV1",
    )
    if (
        oof_document.get("run_status") != "COMPLETE"
        or oof_document.get("publication_gate") != "PASS"
        or oof_document.get("completed_cell_count") != 45
        or oof_document.get("selection_ready") is not True
        or selection_document.get("decision_status") != "NO_MODEL_ADVANCE"
        or selection_document.get("winner") is not None
        or selection_document.get("kalshi_transport_allowed") is not False
    ):
        raise ReportError("canonical Stage-B terminal state is inconsistent")

    verified_panel = load_verified_regulation_panel_batch(
        project_root=project,
        batch_manifest_path=panel_manifest.resolve(),
        batch_manifest_file_sha256=panel_sha256,
        venue_scope=(POLYMARKET,),
        expected_game_count=153,
    )
    oof_root = oof_manifest.resolve().parents[4]
    shards = discover_verified_regulation_oof_shards(
        project_root=project,
        output_root=oof_root,
        verified_panel=verified_panel,
    )
    if len(shards) != 45:
        raise ReportError("official verifier did not recover exact-45 shards")
    verified_selection = verify_regulation_selection_lock(
        project_root=project,
        output_root=oof_root,
        manifest_path=selection_manifest.resolve(),
        manifest_file_sha256=selection_sha256,
        verified_panel=verified_panel,
        shards=shards,
    )
    if (
        verified_selection.decision_status != "NO_MODEL_ADVANCE"
        or verified_selection.winner is not None
        or not verified_selection.formal_lock_verified
    ):
        raise ReportError("official SelectionLock verification failed")
    verify_rule_audit_batch(
        project_root=project,
        manifest_path=rule_audit_manifest.resolve(),
        manifest_file_sha256=rule_audit_sha256,
        expected_game_count=153,
    )

    fold_rows = _fold_metric_improvements(oof_root, selection_document)
    oof_summary = _oof_sample_summary(selection_document)
    hashes = {
        "Stage A": stage_a_sha256,
        "PanelV4": panel_sha256,
        "Exact-45 OOF": oof_sha256,
        "SelectionLock": selection_sha256,
        "Native rule audit": rule_audit_sha256,
        "Selection semantic": str(selection_document["selection_sha256"]),
        "Selection lock semantic": str(
            selection_document["selection_lock_sha256"]
        ),
    }
    payload = _render_html(
        stage_a_metrics=stage_a_metrics,
        panel_document=panel_document,
        oof_document=oof_document,
        selection=selection_document,
        rule_document=rule_document,
        fold_rows=fold_rows,
        oof_summary=oof_summary,
        input_hashes=hashes,
    )
    output = (project / output_root).resolve()
    html_path, html_sha = _atomic_content_addressed_write(
        output, payload, "html"
    )
    manifest = {
        "schema": "NFLX15RegulationDecisionReportManifestV1",
        "experiment_id": "X-15",
        "cohort": "development",
        "game_count": 153,
        "claim_boundary": CLAIM_BOUNDARY,
        "decision_status": "NO_MODEL_ADVANCE",
        "holdout_reaction_accessed": False,
        "kalshi_transport_allowed": False,
        "oof_sample_summary": oof_summary,
        "stage_a_metrics": dict(stage_a_metrics),
        "inputs": hashes,
        "html": {
            "path": str(html_path),
            "file_sha256": html_sha,
            "byte_length": len(payload),
        },
        "publication_gate": "PASS",
    }
    manifest_path, _ = _atomic_content_addressed_write(
        output,
        _canonical_bytes(manifest),
        "report-manifest.json",
        namespace="manifests",
    )
    return html_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage-a-manifest", type=Path, required=True)
    parser.add_argument("--stage-a-sha256", required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--oof-manifest", type=Path, required=True)
    parser.add_argument("--oof-sha256", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--rule-audit-manifest", type=Path, required=True)
    parser.add_argument("--rule-audit-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    html_path, manifest_path = render_report(
        project_root=args.project_root,
        stage_a_manifest=args.stage_a_manifest,
        stage_a_sha256=args.stage_a_sha256,
        panel_manifest=args.panel_manifest,
        panel_sha256=args.panel_sha256,
        oof_manifest=args.oof_manifest,
        oof_sha256=args.oof_sha256,
        selection_manifest=args.selection_manifest,
        selection_sha256=args.selection_sha256,
        rule_audit_manifest=args.rule_audit_manifest,
        rule_audit_sha256=args.rule_audit_sha256,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "html_path": str(html_path),
                "html_sha256": _sha256_file(html_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "holdout_reaction_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
