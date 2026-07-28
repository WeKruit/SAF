"""Verified exact-153 development publication for NFL X-15.

This module reads only the already-published X-13 development batch.  It
never opens the sealed 81-game holdout, fetches upstream data, or interprets
historical trades as executable quotes/fills.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.research.nfl_x15_landmarks import (
    build_nfl_x15_landmark_panel,
)
from prediction_market.research.nfl_x15_models import run_x15_walk_forward
from prediction_market.research.nfl_x15_policy import (
    build_trade_marked_shadow_curve,
    select_earliest_landmark_signals,
)
from prediction_market.research.nfl_x15_statistics import (
    summarize_development_signals,
)


EXPERIMENT_ID: Final[str] = "X-15"
RESULT_LABEL: Final[str] = "PRELIMINARY_NON_EXECUTABLE"
CLAIM_BOUNDARY: Final[str] = (
    "DEVELOPMENT_ONLY_HISTORICAL_ACTUAL_TRADES; no executable quote, fill, "
    "live latency, causality, tradeability, or holdout claim."
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "factor_id",
    "landmark_seconds",
    "endpoint_seconds",
    "reference_gap_at_landmark",
    "game_seconds_remaining",
    "score_margin",
    "pit_strength",
    "reference_delta",
    "mark_landmark_price",
)
_EXPECTED_GAME_COUNT = 153


class X15PublicationError(RuntimeError):
    """A development input or publication invariant failed closed."""


@dataclass(frozen=True, slots=True)
class X15Publication:
    manifest_path: Path
    manifest_sha256: str
    report_path: Path
    report_sha256: str
    game_count: int
    decision_landmark_rows: int
    target_eligible_landmark_rows: int
    signal_rows: int
    holdout_reaction_accessed: bool = False


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_ready(value: object) -> object:
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
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        sink,
        compression="zstd",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise X15PublicationError(f"content-addressed collision: {path}")
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
            raise X15PublicationError(f"content-addressed publish race: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _publish_object(
    root: Path,
    *,
    logical_name: str,
    payload: bytes,
    suffix: str,
    row_count: int,
) -> dict[str, object]:
    digest = _sha256_bytes(payload)
    hexadecimal = digest.removeprefix("sha256:")
    path = root / "objects" / "sha256" / hexadecimal[:2] / (
        hexadecimal + "." + suffix
    )
    _atomic_publish(path, payload)
    return {
        "logical_name": logical_name,
        "object_path": str(path.relative_to(root)),
        "object_sha256": digest,
        "byte_length": len(payload),
        "row_count": int(row_count),
        "format": suffix,
    }


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, label: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise X15PublicationError(f"{label} missing columns: {missing}")


def _is_true(value: object) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _clean_text(value: object, default: str = "") -> str:
    return default if pd.isna(value) else str(value).strip()


def _canonical_event_type(row: pd.Series) -> str:
    if _is_true(row.get("interception", False)):
        return "interception"
    if _is_true(row.get("fumble_lost", False)):
        return "lost_fumble"
    if _is_true(row.get("fourth_down_failed", False)):
        return "turnover_on_downs"
    if _is_true(row.get("pass_touchdown", False)):
        return "passing_touchdown"
    if _is_true(row.get("rush_touchdown", False)):
        return "rushing_touchdown"
    if _is_true(row.get("return_touchdown", False)):
        return "return_touchdown"
    field_goal_result = _clean_text(
        row.get("field_goal_result"), ""
    ).lower()
    if field_goal_result:
        return "field_goal_" + field_goal_result.replace(" ", "_")
    if _is_true(row.get("safety", False)):
        return "safety"
    if _is_true(row.get("punt", False)):
        return "punt"
    play_type = _clean_text(
        row.get("play_type"), "administrative"
    ).lower()
    return play_type.replace(" ", "_")


def build_episode_state_frame(
    factor_rows: pd.DataFrame,
    references: pd.DataFrame,
    *,
    matched_control_episode_ids: set[tuple[str, str]],
) -> pd.DataFrame:
    """Return one reference-oriented state row per finalized episode."""

    _require_columns(
        factor_rows,
        (
            "game_id",
            "episode_id",
            "event_interval_end_utc",
            "week",
            "play_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_seconds_remaining",
            "rolling_offense_success_diff",
            "factor_id",
            "horizon_seconds",
        ),
        label="factor_rows",
    )
    _require_columns(
        references,
        (
            "game_id",
            "episode_id",
            "focal_team",
            "p_after_ref",
            "reference_delta",
            "support_status",
        ),
        label="references",
    )
    supported = references.loc[
        references["support_status"].eq("SUPPORTED_RETROSPECTIVE")
    ].copy()
    if supported.empty:
        raise X15PublicationError("no supported reference observations")
    if supported.duplicated(["game_id", "episode_id"]).any():
        raise X15PublicationError("supported reference episode grain is not unique")

    sports = (
        factor_rows.sort_values(
            ["game_id", "episode_id", "factor_id", "horizon_seconds"],
            kind="mergesort",
        )
        .drop_duplicates(["game_id", "episode_id"], keep="first")
        .copy()
    )
    merged = supported.merge(
        sports,
        on=["game_id", "episode_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_ref", ""),
    )
    if merged.empty:
        raise X15PublicationError("supported reference and episode state do not join")

    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        home = str(row["home_team"])
        away = str(row["away_team"])
        focal = str(row["focal_team"])
        if focal not in {home, away}:
            raise X15PublicationError("reference focal team is not a game team")
        home_margin = float(row["home_score"]) - float(row["away_score"])
        event_type = _canonical_event_type(pd.Series(row))
        rows.append(
            {
                "game_id": str(row["game_id"]),
                "episode_id": str(row["episode_id"]),
                "beneficiary_team": focal,
                "finalized_interval_end_utc": row["event_interval_end_utc"],
                "event_type": event_type,
                "game_seconds_remaining": float(row["game_seconds_remaining"]),
                "score_margin": home_margin if focal == home else -home_margin,
                "reference_delta": float(row["reference_delta"]),
                "p_after_ref": float(row["p_after_ref"]),
                "factor_id": "NFL.X15.PRIMARY." + event_type.upper(),
                "nfl_week": int(row["week"]),
                "is_matched_control": (
                    (str(row["game_id"]), str(row["episode_id"]))
                    in matched_control_episode_ids
                ),
                "pit_strength": (
                    float(row["rolling_offense_success_diff"])
                    if pd.notna(row["rolling_offense_success_diff"])
                    else math.nan
                ),
                "orientation_resolved": True,
                "contaminated": False,
                "contamination_time_utc": None,
                "censored": False,
                "censor_time_utc": None,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["game_id", "finalized_interval_end_utc", "episode_id"],
        kind="mergesort",
    )
    result["finalized_interval_end_utc"] = pd.to_datetime(
        result["finalized_interval_end_utc"], utc=True, errors="raise"
    )
    result["contamination_time_utc"] = result.groupby("game_id")[
        "finalized_interval_end_utc"
    ].shift(-1)
    return result.reset_index(drop=True)


def normalize_actual_moneyline_trades(
    observations: pd.DataFrame,
    contract_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize exact observed winner trades using native exchange time."""

    _require_columns(
        observations,
        (
            "game_id",
            "observation_id",
            "venue",
            "logical_market_id",
            "outcome",
            "kind",
            "source_start_utc",
            "native_source_time_utc",
            "price",
            "provenance",
            "primary_path_eligible",
        ),
        label="observations",
    )
    _require_columns(
        contract_inventory,
        (
            "game_id",
            "logical_market_id",
            "outcome",
            "venue",
            "family",
            "analysis_eligible",
        ),
        label="contract_inventory",
    )
    contracts = contract_inventory.loc[
        contract_inventory["family"].eq("moneyline")
        & contract_inventory["analysis_eligible"].eq(True),
        [
            "game_id",
            "logical_market_id",
            "outcome",
            "venue",
            "family",
        ],
    ].drop_duplicates()
    filtered = observations.loc[
        observations["kind"].eq("trade")
        & observations["provenance"].eq("observed")
        & observations["primary_path_eligible"].eq(True)
    ].merge(
        contracts,
        on=["game_id", "logical_market_id", "outcome", "venue"],
        how="inner",
        validate="many_to_one",
    )
    native = pd.to_datetime(
        filtered["native_source_time_utc"], utc=True, errors="coerce"
    )
    fallback = pd.to_datetime(
        filtered["source_start_utc"], utc=True, errors="coerce"
    )
    source_time = native.fillna(fallback)
    if source_time.isna().any():
        raise X15PublicationError("actual trade source time is missing")
    result = pd.DataFrame(
        {
            "trade_id": filtered["observation_id"].astype(str),
            "game_id": filtered["game_id"].astype(str),
            "venue": filtered["venue"].astype(str),
            "logical_market_id": filtered["logical_market_id"].astype(str),
            "market_family": filtered["family"].astype(str),
            "outcome_team": filtered["outcome"].astype(str),
            "source_time_utc": source_time,
            "price": pd.to_numeric(filtered["price"], errors="raise"),
            "kind": "trade",
            "provenance": "observed",
            "orientation_resolved": True,
        }
    )
    if result["trade_id"].duplicated().any():
        raise X15PublicationError("actual trade IDs are not unique")
    return result.sort_values(
        ["source_time_utc", "trade_id"], kind="mergesort"
    ).reset_index(drop=True)


def _read_verified_json(path: Path) -> tuple[dict[str, object], str]:
    if path.is_symlink() or not path.is_file():
        raise X15PublicationError(f"manifest is missing: {path}")
    digest = _sha256_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8")), digest
    except (OSError, json.JSONDecodeError) as error:
        raise X15PublicationError(f"manifest is unreadable: {path}") from error


def _verified_stage(
    exact_root: Path,
    manifest: dict[str, object],
    stage_name: str,
) -> pd.DataFrame:
    stages = manifest.get("stages")
    if not isinstance(stages, dict) or not isinstance(stages.get(stage_name), dict):
        raise X15PublicationError(f"missing stage: {stage_name}")
    descriptor = stages[stage_name]
    relative = Path(str(descriptor["object_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise X15PublicationError(f"unsafe stage path: {stage_name}")
    path = exact_root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(descriptor["byte_length"])
        or _sha256_file(path) != descriptor["object_sha256"]
    ):
        raise X15PublicationError(f"stage identity mismatch: {stage_name}")
    frame = pd.read_parquet(path)
    if len(frame) != int(descriptor["row_count"]):
        raise X15PublicationError(f"stage row count mismatch: {stage_name}")
    return frame


def _load_reference(
    reference_manifest_path: Path,
) -> tuple[pd.DataFrame, str]:
    manifest, digest = _read_verified_json(reference_manifest_path)
    if (
        manifest.get("experiment_id") != "X-13"
        or manifest.get("cohort") != "development"
        or manifest.get("development_game_count") != _EXPECTED_GAME_COUNT
        or manifest.get("final_holdout_access") != "CLOSED"
        or manifest.get("holdout_reaction_accessed") is not False
    ):
        raise X15PublicationError("reference development/holdout contract mismatch")
    descriptor = next(
        (
            item
            for item in manifest.get("objects", [])
            if item.get("logical_name") == "reference_observations"
        ),
        None,
    )
    if not isinstance(descriptor, dict):
        raise X15PublicationError("reference observations are missing")
    root = reference_manifest_path.parents[3]
    path = root / str(descriptor["object_path"])
    if (
        _sha256_file(path) != descriptor["object_sha256"]
        or path.stat().st_size != descriptor["byte_length"]
    ):
        raise X15PublicationError("reference object identity mismatch")
    frame = pd.read_parquet(path)
    if len(frame) != descriptor["row_count"]:
        raise X15PublicationError("reference row count mismatch")
    return frame, digest


def _load_matched_control_ids(
    matched_manifest_path: Path,
) -> tuple[set[tuple[str, str]], str]:
    manifest, digest = _read_verified_json(matched_manifest_path)
    if (
        manifest.get("experiment_id") != "X-13"
        or manifest.get("cohort") != "development"
        or len(manifest.get("selected_game_ids", [])) != _EXPECTED_GAME_COUNT
        or manifest.get("final_holdout_access") != "CLOSED"
        or manifest.get("holdout_reaction_accessed") is not False
    ):
        raise X15PublicationError("matched-control holdout contract mismatch")
    descriptor = next(
        (
            item
            for item in manifest.get("objects", [])
            if item.get("logical_name") == "matched_controls"
        ),
        None,
    )
    if not isinstance(descriptor, dict):
        raise X15PublicationError("matched controls are missing")
    root = matched_manifest_path.parents[3]
    path = root / str(descriptor["object_path"])
    if (
        _sha256_file(path) != descriptor["object_sha256"]
        or path.stat().st_size != descriptor["byte_length"]
    ):
        raise X15PublicationError("matched-control object identity mismatch")
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT control_game_id, control_episode_id
            FROM read_parquet(?)
            WHERE balance_pass = TRUE
            """,
            [str(path)],
        ).fetchall()
    finally:
        connection.close()
    return {(str(game), str(episode)) for game, episode in rows}, digest


def _model_summary(
    fold_metrics: pd.DataFrame,
    signals: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    metric_summary = (
        fold_metrics.groupby(["venue", "model_id"], as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            validation_games=("validation_game_count", "sum"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            log_loss=("multiclass_log_loss", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mean_noise_threshold=("noise_threshold", "mean"),
        )
    )
    if signals.empty:
        signal_summary = pd.DataFrame(
            columns=[
                "venue",
                "model_id",
                "signal_count",
                "evaluated_signal_count",
                "censored_signal_count",
                "hit_rate",
                "mean_gross_markout",
                "median_gross_markout",
            ]
        )
    else:
        all_signal_counts = (
            signals.groupby(["venue", "model_id"], as_index=False)
            .agg(
                signal_count=("episode_id", "size"),
                evaluated_signal_count=(
                    "evaluation_status",
                    lambda values: int(values.eq("EVALUATED").sum()),
                ),
                censored_signal_count=(
                    "evaluation_status",
                    lambda values: int(
                        values.eq("CENSORED_EVALUATION").sum()
                    ),
                ),
            )
        )
        evaluated = signals.loc[
            signals["evaluation_status"].eq("EVALUATED")
        ]
        evaluated_effects = (
            evaluated.groupby(["venue", "model_id"], as_index=False)
            .agg(
                hit_rate=("gross_markout", lambda values: float((values > 0).mean())),
                mean_gross_markout=("gross_markout", "mean"),
                median_gross_markout=("gross_markout", "median"),
            )
        )
        signal_summary = all_signal_counts.merge(
            evaluated_effects,
            on=["venue", "model_id"],
            how="left",
        )
    result = metric_summary.merge(
        signal_summary, on=["venue", "model_id"], how="left"
    )
    if not curves.empty:
        curve_summary = (
            curves.groupby(["venue", "model_id"], as_index=False)
            .agg(
                terminal_shadow_nav=("nav", "last"),
                maximum_shadow_drawdown=("drawdown", "min"),
            )
        )
        result = result.merge(curve_summary, on=["venue", "model_id"], how="left")
    result["evidence_classification"] = RESULT_LABEL
    result["claim_boundary"] = CLAIM_BOUNDARY
    return result.sort_values(["venue", "model_id"], kind="mergesort")


def _report_html(
    summary: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    *,
    audit: dict[str, object],
) -> bytes:
    rows = summary.to_dict("records")
    exclusions = exclusion_summary.head(30).to_dict("records")
    payload = json.dumps(
        _json_ready(
            {"summary": rows, "exclusions": exclusions, "audit": audit}
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-15 153场多时间 Taker 研究</title>
<style>
body{{font:14px/1.5 system-ui;margin:0;background:#f4f1e9;color:#17231d}}header,main{{padding:24px 32px}}header{{background:#e5e0d4;border-bottom:1px solid #cbc5b8}}h1{{font-family:Georgia,serif}}.note{{border-left:4px solid #a66a13;background:#fff7e5;padding:12px}}table{{border-collapse:collapse;width:100%;background:#fffdf8;margin:14px 0}}th,td{{padding:8px;border-bottom:1px solid #e6e0d5;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#ece7dc;position:sticky;top:0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{background:#fffdf8;border:1px solid #d7d0c3;padding:14px}}code{{font-size:11px}}button{{padding:8px 12px;margin-right:8px}}section{{display:none}}section.active{{display:block}}
</style>
<header><div>NFL FACTOR LAB · X-15</div><h1>153 场多时间 Taker Signal（历史实际成交）</h1>
<p>结果是 development-only、non-executable。成交 mark 不是 bid/ask，也不是 fill。</p></header>
<main><div class="note">81 场 holdout 仍为 SEALED_UNREAD；没有 L2、手续费或可成交价格，因此这里不是正式回测收益。</div>
<p><button data-tab="models">模型</button><button data-tab="audit">数据审计</button><button data-tab="limits">结论边界</button></p>
<section id="models" class="active"><h2>Walk-forward 结果</h2><div id="summary"></div></section>
<section id="audit"><h2>Landmark 排除</h2><div id="exclusions"></div><h2>Lineage</h2><pre id="lineage"></pre></section>
<section id="limits"><h2>现在能回答什么</h2><ul><li>在 1/2/3/5/10 秒观察后，模型能否预测后续 5–60 秒实际成交方向。</li><li>不同 venue 必须分开训练。</li><li>只能称 historical signal candidate；不能称可执行 alpha。</li></ul></section>
</main><script id="data" type="application/json">{payload}</script><script>
const D=JSON.parse(document.getElementById('data').textContent);
function fmt(v){{if(v===null||v===undefined)return '—';if(typeof v==='number')return Math.abs(v)<1? v.toFixed(4):v.toFixed(2);return String(v)}}
function table(rows){{if(!rows.length)return '<p>无数据</p>';const cols=Object.keys(rows[0]);return '<table><thead><tr>'+cols.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+fmt(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}}
document.getElementById('summary').innerHTML=table(D.summary);document.getElementById('exclusions').innerHTML=table(D.exclusions);document.getElementById('lineage').textContent=JSON.stringify(D.audit,null,2);
document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('section').forEach(s=>s.classList.remove('active'));document.getElementById(b.dataset.tab).classList.add('active')}})
</script></html>"""
    return html.encode("utf-8")


def publish_x15_exact153(
    *,
    exact_root: str | Path,
    batch_index_path: str | Path,
    reference_manifest_path: str | Path,
    matched_manifest_path: str | Path,
    output_root: str | Path,
) -> X15Publication:
    """Run and atomically publish the X-15 development-only study."""

    exact = Path(exact_root)
    batch, batch_index_sha = _read_verified_json(Path(batch_index_path))
    games = batch.get("games")
    if (
        batch.get("experiment_id") != "X-13"
        or batch.get("cohort") != "development"
        or batch.get("game_count") != _EXPECTED_GAME_COUNT
        or batch.get("final_holdout_access") != "CLOSED"
        or batch.get("publication_gate") != "PASS"
        or not isinstance(games, list)
        or len(games) != _EXPECTED_GAME_COUNT
    ):
        raise X15PublicationError("exact-153 development batch contract mismatch")

    references, reference_manifest_sha = _load_reference(
        Path(reference_manifest_path)
    )
    matched_controls, matched_manifest_sha = _load_matched_control_ids(
        Path(matched_manifest_path)
    )
    output = Path(output_root)
    landmark_frames: list[pd.DataFrame] = []
    input_manifest_sha256s: list[str] = []
    for game in sorted(games, key=lambda item: str(item["game_id"])):
        manifest_path = exact / str(game["manifest_path"])
        manifest, manifest_sha = _read_verified_json(manifest_path)
        if (
            manifest_sha != game["manifest_sha256"]
            or manifest.get("game_id") != game["game_id"]
            or manifest.get("cohort") != "development"
            or manifest.get("publication_gate") != "PASS"
        ):
            raise X15PublicationError("single-game manifest identity mismatch")
        observations = _verified_stage(
            exact, manifest, "actual_market_observations"
        )
        inventory = _verified_stage(exact, manifest, "contract_inventory")
        factor_rows = _verified_stage(exact, manifest, "factor_observations")
        game_references = references.loc[
            references["game_id"].eq(game["game_id"])
        ]
        episodes = build_episode_state_frame(
            factor_rows,
            game_references,
            matched_control_episode_ids=matched_controls,
        )
        trades = normalize_actual_moneyline_trades(observations, inventory)
        landmark_frames.append(build_nfl_x15_landmark_panel(trades, episodes))
        input_manifest_sha256s.append(manifest_sha)

    landmarks = pd.concat(landmark_frames, ignore_index=True)
    decision_landmarks = landmarks.loc[
        landmarks["decision_eligible"].eq(True)
    ].reset_index(drop=True)
    target_eligible_landmarks = landmarks.loc[
        landmarks["target_eligible"].eq(True)
    ].reset_index(drop=True)
    if (
        decision_landmarks.empty
        or decision_landmarks["game_id"].nunique()
        != _EXPECTED_GAME_COUNT
    ):
        raise X15PublicationError(
            "decision landmark panel does not cover all 153 development games"
        )
    model_run = run_x15_walk_forward(
        landmarks,
        feature_columns=FEATURE_COLUMNS,
        random_state=20260728,
    )
    joined = model_run.oof_predictions.merge(
        landmarks.reset_index()
        .rename(columns={"index": "source_row_id"})
        [
            [
                "source_row_id",
                "logical_market_id",
                "beneficiary_team",
                "event_anchor_utc",
                "mark_landmark_price",
                "mark_endpoint_price",
            ]
        ],
        on="source_row_id",
        how="left",
        validate="many_to_one",
    ).rename(
        columns={
            "beneficiary_team": "outcome",
            "event_anchor_utc": "source_time_utc",
            "predicted_label": "predicted_class",
            "noise_threshold": "materiality_threshold",
            "mark_landmark_price": "mark_landmark",
            "mark_endpoint_price": "mark_endpoint",
        }
    )
    joined["dataset_partition"] = "development"
    signals, signal_audit = select_earliest_landmark_signals(joined)
    evaluated_signals = signals.loc[
        signals["evaluation_status"].eq("EVALUATED")
    ].reset_index(drop=True)
    signal_statistics = (
        summarize_development_signals(
            evaluated_signals,
            seed=20260728,
            n_bootstrap=10_000,
            min_support_games=30,
            min_support_signals=20,
        )
        if not evaluated_signals.empty
        else pd.DataFrame()
    )
    curves: list[pd.DataFrame] = []
    curve_audits: list[pd.DataFrame] = []
    if not signals.empty:
        for (venue, model_id), group in signals.groupby(
            ["venue", "model_id"], sort=True
        ):
            curve, audit = build_trade_marked_shadow_curve(
                group, fixed_contracts=1, initial_nav=1000.0
            )
            if not curve.empty:
                curve["venue"] = venue
                curve["model_id"] = model_id
            audit["venue"] = venue
            audit["model_id"] = model_id
            if not curve.empty:
                curves.append(curve)
            curve_audits.append(audit)
    curve_frame = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    curve_audit = (
        pd.concat(curve_audits, ignore_index=True)
        if curve_audits
        else pd.DataFrame()
    )
    exclusion_summary = (
        landmarks.assign(
            exclusion_reason=landmarks["exclusion_reasons"].map(
                lambda values: "|".join(values) if values else "ELIGIBLE"
            )
        )
        .groupby(
            ["venue", "landmark_seconds", "endpoint_seconds", "exclusion_reason"],
            as_index=False,
        )
        .agg(rows=("episode_id", "size"), games=("game_id", "nunique"))
        .sort_values(["rows", "games"], ascending=False, kind="mergesort")
    )
    summary = _model_summary(
        model_run.fold_metrics, signals, curve_frame
    ).reset_index(drop=True)
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "result_label": RESULT_LABEL,
        "claim_boundary": CLAIM_BOUNDARY,
        "development_game_count": int(
            decision_landmarks["game_id"].nunique()
        ),
        "landmark_rows": int(len(landmarks)),
        "decision_landmark_rows": int(len(decision_landmarks)),
        "target_eligible_landmark_rows": int(
            len(target_eligible_landmarks)
        ),
        "decision_episode_count": int(
            decision_landmarks["episode_id"].nunique()
        ),
        "target_eligible_episode_count": int(
            target_eligible_landmarks["episode_id"].nunique()
        ),
        "oof_prediction_rows": int(len(model_run.oof_predictions)),
        "signal_rows": int(len(signals)),
        "evaluated_signal_rows": int(len(evaluated_signals)),
        "censored_signal_rows": int(
            signals["evaluation_status"]
            .eq("CENSORED_EVALUATION")
            .sum()
        ),
        "holdout_game_count": 81,
        "holdout_state": "SEALED_UNREAD",
        "holdout_reaction_accessed": False,
        "batch_index_sha256": batch_index_sha,
        "reference_manifest_sha256": reference_manifest_sha,
        "matched_manifest_sha256": matched_manifest_sha,
        "single_game_manifest_sha256s": sorted(input_manifest_sha256s),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    objects = [
        _publish_object(
            output,
            logical_name="decision_landmarks",
            payload=_parquet_bytes(decision_landmarks),
            suffix="parquet",
            row_count=len(decision_landmarks),
        ),
        _publish_object(
            output,
            logical_name="target_eligible_landmarks",
            payload=_parquet_bytes(target_eligible_landmarks),
            suffix="parquet",
            row_count=len(target_eligible_landmarks),
        ),
        _publish_object(
            output,
            logical_name="landmark_exclusion_summary",
            payload=_parquet_bytes(exclusion_summary),
            suffix="parquet",
            row_count=len(exclusion_summary),
        ),
        _publish_object(
            output,
            logical_name="oof_predictions",
            payload=_parquet_bytes(model_run.oof_predictions),
            suffix="parquet",
            row_count=len(model_run.oof_predictions),
        ),
        _publish_object(
            output,
            logical_name="fold_metrics",
            payload=_parquet_bytes(model_run.fold_metrics),
            suffix="parquet",
            row_count=len(model_run.fold_metrics),
        ),
        _publish_object(
            output,
            logical_name="signals",
            payload=_parquet_bytes(signals),
            suffix="parquet",
            row_count=len(signals),
        ),
        _publish_object(
            output,
            logical_name="signal_statistics",
            payload=_parquet_bytes(signal_statistics),
            suffix="parquet",
            row_count=len(signal_statistics),
        ),
        _publish_object(
            output,
            logical_name="signal_audit",
            payload=_parquet_bytes(signal_audit),
            suffix="parquet",
            row_count=len(signal_audit),
        ),
        _publish_object(
            output,
            logical_name="shadow_curve",
            payload=_parquet_bytes(curve_frame),
            suffix="parquet",
            row_count=len(curve_frame),
        ),
        _publish_object(
            output,
            logical_name="shadow_curve_audit",
            payload=_parquet_bytes(curve_audit),
            suffix="parquet",
            row_count=len(curve_audit),
        ),
        _publish_object(
            output,
            logical_name="model_summary",
            payload=_parquet_bytes(summary),
            suffix="parquet",
            row_count=len(summary),
        ),
    ]
    report_payload = _report_html(summary, exclusion_summary, audit=audit)
    report_object = _publish_object(
        output,
        logical_name="offline_report",
        payload=report_payload,
        suffix="html",
        row_count=1,
    )
    objects.append(report_object)
    manifest = {
        "schema": "nfl_x15_development_publication_v2",
        **audit,
        "objects": sorted(objects, key=lambda item: str(item["logical_name"])),
    }
    manifest_payload = _canonical_json(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        output
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    _atomic_publish(manifest_path, manifest_payload)
    report_path = output / str(report_object["object_path"])
    return X15Publication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        report_path=report_path,
        report_sha256=str(report_object["object_sha256"]),
        game_count=_EXPECTED_GAME_COUNT,
        decision_landmark_rows=len(decision_landmarks),
        target_eligible_landmark_rows=len(target_eligible_landmarks),
        signal_rows=len(signals),
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "EXPERIMENT_ID",
    "FEATURE_COLUMNS",
    "RESULT_LABEL",
    "X15Publication",
    "X15PublicationError",
    "build_episode_state_frame",
    "normalize_actual_moneyline_trades",
    "publish_x15_exact153",
]
