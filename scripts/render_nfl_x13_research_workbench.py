#!/usr/bin/env python3
"""Render the completed exact-153 historical study as one offline workbench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from prediction_market.reports.nfl_x13_research_workbench import (
    render_research_workbench,
)
from prediction_market.sports.nfl_x13_historical_study_publication import (
    verify_historical_study_publication,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = (
    ROOT
    / "artifacts/market-observation/nfl/x13/historical-study-v2"
    / "kalshi-native-time-v3/exact-153"
)
REFERENCE_ROOT = (
    ROOT
    / "artifacts/market-observation/nfl/x13/historical-reference-v2"
    / "kalshi-native-time-v3/exact-153"
)
MATCHED_ROOT = (
    ROOT
    / "artifacts/market-observation/nfl/x13/matched-reaction-v2"
    / "kalshi-native-time-v3/exact-153"
)
PMXT_L2_METHOD_AUDIT = (
    ROOT
    / "artifacts/data-audit/pmxt-l2-method-audit-v1/sha256/92"
    / "92a80cd64f7cecd79990d2cd0e484743475afc86b867e711fbeb3c700aa29253.json"
)
CALIBRATION_MANIFEST = (
    ROOT
    / "artifacts/market-observation/nfl/x13/market-calibration-v1"
    / "kalshi-native-time-v3/exact-153/manifests/sha256/10"
    / "10b8b443f663a0b4eeb3b69d9b5227f12c2a1f47749dc0b1bc3712e74e3bca3a.manifest.json"
)
TICK_INVENTORY_MANIFEST = (
    ROOT
    / "artifacts/data-audit/tick-data-inventory-v1/manifests/sha256/c8"
    / "c82eb668ba6af9cfc863d4e38b1ed776edae12fb1169dea223fd8a3b1ea5fffc.manifest.json"
)
OUTPUT_ROOT = (
    ROOT
    / "reports/nfl-factor-lab/exact-153/research-workbench-v5"
)
NOTEBOOK = (
    ROOT / "notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _one_manifest(root: Path) -> Path:
    paths = sorted(root.glob("manifests/sha256/*/*.manifest.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one immutable manifest under {root}")
    path = paths[0]
    if path.name.split(".")[0] != _sha(path).removeprefix("sha256:"):
        raise RuntimeError(f"manifest path/hash mismatch: {path}")
    return path


def _artifact_source(manifest_path: Path) -> dict[str, str]:
    """Identify the publication; the renderer verifies its object bytes."""

    return {
        "artifact_sha256": _sha(manifest_path),
        "artifact_path": manifest_path.resolve().as_posix(),
    }


def _verified_tables(
    root: Path,
    manifest_path: Path,
    *,
    load_names: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("experiment_id") != "X-13"
        or manifest.get("cohort") != "development"
        or manifest.get("final_holdout_access") != "CLOSED"
        or manifest.get("holdout_reaction_accessed") is not False
    ):
        raise RuntimeError(f"governance mismatch: {manifest_path}")
    tables: dict[str, pd.DataFrame] = {}
    for descriptor in manifest.get("objects", ()):
        if descriptor.get("format") != "parquet":
            continue
        path = root / str(descriptor["object_path"])
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["byte_length"]
            or _sha(path) != descriptor["object_sha256"]
        ):
            raise RuntimeError(f"object hash mismatch: {path}")
        logical_name = str(descriptor["logical_name"])
        if load_names is not None and logical_name not in load_names:
            continue
        frame = pq.read_table(path).to_pandas()
        if len(frame) != descriptor["row_count"]:
            raise RuntimeError(f"object row-count mismatch: {path}")
        tables[logical_name] = frame
    return tables, manifest


def _reference_completion(
    reference_summary: pd.DataFrame,
    factor_id: str,
    horizon: int,
) -> float | None:
    row = reference_summary[
        reference_summary["factor_id"].eq(factor_id)
        & reference_summary["venue"].eq("polymarket")
        & reference_summary["market_family"].eq("moneyline")
        & reference_summary["horizon_seconds"].eq(horizon)
    ]
    return (
        None if row.empty else float(row.iloc[0]["completion_fraction_median"])
    )


def _object_row_count(manifest: dict[str, object], logical_name: str) -> int:
    matches = [
        descriptor
        for descriptor in manifest.get("objects", ())
        if descriptor.get("logical_name") == logical_name
    ]
    if len(matches) != 1 or type(matches[0].get("row_count")) is not int:
        raise RuntimeError(f"missing row_count for {logical_name}")
    return int(matches[0]["row_count"])


def main() -> None:
    study_manifest_path = _one_manifest(STUDY_ROOT)
    study = verify_historical_study_publication(
        study_manifest_path,
        root=STUDY_ROOT,
    )
    study_manifest = json.loads(study_manifest_path.read_text())
    reference_manifest_path = _one_manifest(REFERENCE_ROOT)
    reference, reference_manifest = _verified_tables(
        REFERENCE_ROOT,
        reference_manifest_path,
    )
    matched_manifest_path = _one_manifest(MATCHED_ROOT)
    matched, matched_manifest = _verified_tables(
        MATCHED_ROOT,
        matched_manifest_path,
        load_names={"cross_game_results", "match_balance", "venue_consistency"},
    )
    if matched_manifest.get("full_run_status") != "COMPLETED":
        raise RuntimeError("exact-153 matched-control run is not complete")
    summary = study["summary"]
    reference_summary = reference["reference_summary"]
    cross = matched["cross_game_results"]
    balance = matched["match_balance"]
    balance_pass = balance[balance["balance_pass"]]
    matched_pair_rows = _object_row_count(matched_manifest, "matched_controls")
    eligible_path_count = _object_row_count(study_manifest, "ecdf")
    reference_completion_rows = int(
        reference["reference_paths"]["completion_status"].eq("ELIGIBLE").sum()
    )
    development_game_count = int(study_manifest["development_game_count"])
    dual_venue_support_count = int(
        matched["venue_consistency"]["both_venues_support_pass"].sum()
    )
    calibration_manifest = json.loads(CALIBRATION_MANIFEST.read_text())
    calibration_objects = calibration_manifest["objects"]
    tick_inventory_manifest = json.loads(TICK_INVENTORY_MANIFEST.read_text())
    tick_inventory_rows = int(tick_inventory_manifest["rows"]["row_count"])
    findings = (
        f"{development_game_count} 场严格有效历史路径为 "
        f"{eligible_path_count:,} 条；经验分布不是只看均值，"
        "同时发布 median/MAD/quantiles、方向概率、阈值概率、game-cluster "
        "bootstrap、LOO 与最大单场贡献。",
        "Polymarket 的 explosive play completion median 从 "
        f"10s={_reference_completion(reference_summary, 'NFL.EVENT.EXPLOSIVE_PLAY', 10):.2f}、"
        f"30s={_reference_completion(reference_summary, 'NFL.EVENT.EXPLOSIVE_PLAY', 30):.2f} "
        f"升到 60s={_reference_completion(reference_summary, 'NFL.EVENT.EXPLOSIVE_PLAY', 60):.2f}；"
        "这是渐进式 source-time adjustment candidate。",
        "Sack 的 Polymarket completion median 从 "
        f"10s={_reference_completion(reference_summary, 'NFL.EVENT.SACK', 10):.2f}、"
        f"30s={_reference_completion(reference_summary, 'NFL.EVENT.SACK', 30):.2f} "
        f"升到 60s={_reference_completion(reference_summary, 'NFL.EVENT.SACK', 60):.2f}；"
        "负向冲击在 20–60 秒更明显。",
        "Kalshi 的历史实际成交覆盖明显更稀；raw venue 均值差不能解释为 venue 效应。"
        "只有 same-episode、matched balance 与 reference-adjusted 三层同时通过时才允许"
        "保留 venue-specific candidate。",
        f"Exact matcher 发布 {matched_pair_rows:,} 对 controls；"
        f"{len(balance):,} 个 covariate-balance groups 中只有 "
        f"{len(balance_pass):,} 个通过 SMD≤0.10，"
        f"Kalshi 通过数={int(balance_pass['venue'].eq('kalshi').sum())}。",
        f"双 venue support pass 数={dual_venue_support_count}；"
        "体育语义已 ACCEPT 的原子 factor 中，没有一个同时满足双 venue ≥30 场、"
        "matched balance、q/LOO/单场贡献全部门槛；因此本轮不冻结 shortlist，"
        "81 场 final holdout 继续 SEALED_UNREAD。",
        "PMXT L2 只发布 metadata attestation 与 fixture method validation："
        "不读取 upstream，不扫描 raw Parquet，也不声称完成全量 empirical metric scan。",
        "Market calibration 只展示正式 development-only metrics、reliability 与"
        " descriptive bias proxies；不读取 bootstrap object，不构成因果或 holdout 结论。",
        f"Tick Data Inventory 正式发布 {tick_inventory_rows} 个 source/venue/data-grade"
        " 能力边界；inventory row 不绑定 factor。",
    )
    artifact_sources = {
        "historical_distribution": _artifact_source(study_manifest_path),
        "reference_completion": _artifact_source(reference_manifest_path),
        "same_episode_paired": _artifact_source(study_manifest_path),
        "venue_consistency": _artifact_source(matched_manifest_path),
        "pmxt_l2_method_audit": _artifact_source(PMXT_L2_METHOD_AUDIT),
        "calibration_bias": _artifact_source(CALIBRATION_MANIFEST),
        "tick_inventory": _artifact_source(TICK_INVENTORY_MANIFEST),
    }
    path = render_research_workbench(
        study_tables=study,
        reference_tables=reference,
        matched_tables=matched,
        notebook_path=NOTEBOOK,
        output_root=OUTPUT_ROOT,
        audit={
            "development_game_count": development_game_count,
            "eligible_path_count": eligible_path_count,
            "distribution_result_rows": len(summary),
            "reference_episode_rows": len(reference["reference_observations"]),
            "reference_completion_rows": reference_completion_rows,
            "matched_result_rows": len(cross),
            "matched_pair_rows": matched_pair_rows,
            "balance_group_rows": len(balance),
            "balance_pass_rows": len(balance_pass),
            "dual_venue_support_pass_rows": dual_venue_support_count,
            "current_exact_153_holdout_status": "SEALED_UNREAD",
            "legacy_v1_20_game_shortlist_holdout_artifacts": (
                "LEGACY_NONCANONICAL_NOT_READ"
            ),
            "shortlist_status": "NOT_FROZEN_NO_FACTOR_PASSES_ALL_GATES",
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
            "study_manifest_sha256": _sha(study_manifest_path),
            "reference_manifest_sha256": _sha(reference_manifest_path),
            "matched_manifest_sha256": _sha(matched_manifest_path),
            "pmxt_l2_method_audit_sha256": _sha(PMXT_L2_METHOD_AUDIT),
            "calibration_manifest_sha256": _sha(CALIBRATION_MANIFEST),
            "calibration_metrics_rows": int(
                calibration_objects["metrics"]["row_count"]
            ),
            "calibration_reliability_rows": int(
                calibration_objects["reliability"]["row_count"]
            ),
            "calibration_bias_proxy_rows": int(
                calibration_objects["bias_proxies"]["row_count"]
            ),
            "tick_inventory_manifest_sha256": _sha(TICK_INVENTORY_MANIFEST),
            "tick_inventory_rows": tick_inventory_rows,
        },
        artifact_sources=artifact_sources,
        findings=findings,
    )
    print(path)


if __name__ == "__main__":
    main()
