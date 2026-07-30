#!/usr/bin/env python3
"""Render the verified NFL exact-153 two-stage research status report.

The renderer is deliberately read-only with respect to research artifacts.  It
accepts only content-addressed JSON manifests, verifies every byte digest, and
loads only content-addressed Parquet tables explicitly referenced by a verified
manifest.  Missing downstream results are reported as pending; they are never
inferred from partial shards.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
from typing import NamedTuple

import pyarrow.parquet as pq


DEFAULT_OUTPUT_ROOT = Path(
    "reports/nfl-factor-lab/exact-153/two-stage-v1"
)
_SHA_PATTERN = re.compile(r"^([0-9a-f]{64})\.")
_SHA_VALUE = re.compile(r"^sha256:[0-9a-f]{64}$")
_BLOCKER_ID = re.compile(r"^[A-Z][A-Z0-9_]+$")


class TwoStageReportError(RuntimeError):
    """A requested artifact is not verified or violates the report contract."""


class TwoStageReportPublication(NamedTuple):
    html_path: Path
    html_sha256: str
    manifest_path: Path
    manifest_sha256: str


class VerifiedJson(NamedTuple):
    path: Path
    file_sha256: str
    document: dict[str, object]


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
    ).encode("utf-8")


def _require_content_addressed_path(path: Path, payload: bytes) -> str:
    match = _SHA_PATTERN.match(path.name)
    if match is None:
        raise TwoStageReportError(
            f"artifact path is not content-addressed: {path}"
        )
    expected_hex = match.group(1)
    if (
        path.parent.name != expected_hex[:2]
        or path.parent.parent.name != "sha256"
    ):
        raise TwoStageReportError(
            f"artifact SHA-256 namespace is malformed: {path}"
        )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_hex:
        raise TwoStageReportError(
            f"artifact SHA-256 mismatch: {path}"
        )
    return f"sha256:{observed}"


def _load_verified_json(
    path: str | Path,
    *,
    role: str,
    allowed_schemas: set[str] | None = None,
) -> VerifiedJson:
    resolved = Path(path).resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise TwoStageReportError(
            f"{role} artifact cannot be read: {resolved}"
        ) from error
    file_sha = _require_content_addressed_path(resolved, payload)
    try:
        document = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise TwoStageReportError(
            f"{role} artifact is not valid JSON: {resolved}"
        ) from error
    if not isinstance(document, dict):
        raise TwoStageReportError(
            f"{role} artifact must be a JSON object"
        )
    schema = document.get("schema")
    if allowed_schemas is not None and schema not in allowed_schemas:
        raise TwoStageReportError(
            f"{role} artifact has unsupported schema: {schema!r}"
        )
    if document.get("holdout_reaction_accessed") is not False:
        raise TwoStageReportError(
            f"{role} artifact is not holdout-reaction blind"
        )
    return VerifiedJson(resolved, file_sha, document)


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TwoStageReportError(f"{field} must be an integer")
    return value


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_VALUE.fullmatch(value) is None:
        raise TwoStageReportError(f"{field} must be a SHA-256 digest")
    return value


def _verify_development_gate(
    gate: object,
    blockers: Sequence[object],
) -> tuple[str, tuple[str, ...]]:
    if gate not in {"OPEN", "BLOCKED"}:
        raise TwoStageReportError(
            "development_gate must be OPEN or BLOCKED"
        )
    normalized: list[str] = []
    for blocker in blockers:
        if (
            not isinstance(blocker, str)
            or _BLOCKER_ID.fullmatch(blocker) is None
        ):
            raise TwoStageReportError(
                "development blockers must be stable uppercase IDs"
            )
        if blocker not in normalized:
            normalized.append(blocker)
    if gate == "OPEN" and normalized:
        raise TwoStageReportError(
            "development_gate OPEN cannot carry blockers"
        )
    if gate == "BLOCKED" and not normalized:
        raise TwoStageReportError(
            "development_gate BLOCKED requires at least one blocker"
        )
    return str(gate), tuple(normalized)


def _verify_factor_manifest(artifact: VerifiedJson) -> None:
    document = artifact.document
    if (
        document.get("experiment_id") != "X-13"
        or document.get("cohort") != "development"
        or document.get("publication_gate") != "PASS"
        or document.get("market_data_read") is not False
        or _require_int(document.get("game_count"), field="factor game_count")
        != 153
    ):
        raise TwoStageReportError(
            "Factor V4 manifest does not describe the verified 153-game "
            "development facts publication"
        )
    registry = document.get("factor_registry")
    counts = document.get("aggregate_counts")
    if not isinstance(registry, dict) or not isinstance(counts, dict):
        raise TwoStageReportError(
            "Factor V4 registry/count authority is missing"
        )
    if (
        registry.get("version") != "v4"
        or _require_int(
            registry.get("factor_count"), field="factor registry count"
        )
        <= 0
        or _require_int(
            counts.get("raw_rows_silently_dropped"),
            field="raw_rows_silently_dropped",
        )
        != 0
    ):
        raise TwoStageReportError(
            "Factor V4 authority is not publication-safe"
        )


def _verify_stage_a_manifest(artifact: VerifiedJson) -> None:
    document = artifact.document
    if (
        document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or document.get("publication_gate") != "PASS"
        or document.get("market_data_read") is not False
        or _require_int(
            document.get("game_count"), field="Stage A game_count"
        )
        != 153
    ):
        raise TwoStageReportError(
            "Stage A manifest does not describe the verified 153-game "
            "development reference publication"
        )
    if not isinstance(document.get("model"), dict):
        raise TwoStageReportError("Stage A model authority is missing")


def _cell(value: object, *, field: str) -> tuple[str, str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(child, str) and child for child in value)
    ):
        raise TwoStageReportError(f"{field} has an invalid execution cell")
    return (value[0], value[1], value[2])


def _verify_exact45_manifest(
    artifact: VerifiedJson,
) -> tuple[
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    document = artifact.document
    if (
        document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or _require_int(
            document.get("authority_game_count"),
            field="exact45 authority_game_count",
        )
        != 153
    ):
        raise TwoStageReportError(
            "exact45 manifest has the wrong development authority"
        )
    _require_sha(
        document.get("authority_metadata_sha256"),
        field="exact45 authority metadata",
    )
    _require_sha(
        document.get("cohort_authority_sha256"),
        field="exact45 cohort authority",
    )
    expected_raw = document.get("expected_execution_cells")
    observed_raw = document.get("observed_execution_cells")
    missing_raw = document.get("missing_execution_cells")
    if not all(
        isinstance(value, list)
        for value in (expected_raw, observed_raw, missing_raw)
    ):
        raise TwoStageReportError(
            "exact45 execution cell inventories are missing"
        )
    expected = tuple(
        _cell(value, field="expected_execution_cells")
        for value in expected_raw
    )
    observed = tuple(
        _cell(value, field="observed_execution_cells")
        for value in observed_raw
    )
    missing = tuple(
        _cell(value, field="missing_execution_cells")
        for value in missing_raw
    )
    if (
        len(set(expected)) != len(expected)
        or len(set(observed)) != len(observed)
        or len(set(missing)) != len(missing)
        or set(observed).intersection(missing)
        or set(observed).union(missing) != set(expected)
        or _require_int(
            document.get("expected_execution_cell_count"),
            field="expected execution count",
        )
        != len(expected)
        or _require_int(
            document.get("observed_execution_cell_count"),
            field="observed execution count",
        )
        != len(observed)
        or _require_int(
            document.get("missing_execution_cell_count"),
            field="missing execution count",
        )
        != len(missing)
    ):
        raise TwoStageReportError(
            "exact45 execution inventory is internally inconsistent"
        )
    if len(expected) != 45:
        raise TwoStageReportError(
            "exact45 authority must contain exactly 45 frozen execution cells"
        )
    unsupported = document.get("unsupported_feature_blocks")
    if (
        not isinstance(unsupported, dict)
        or unsupported.get("D4") != "UNSUPPORTED_FEATURE_BLOCK"
    ):
        raise TwoStageReportError(
            "exact45 authority must preserve the D4 unsupported-feature gate"
        )
    ready = document.get("selection_ready")
    if ready is not (len(missing) == 0):
        raise TwoStageReportError(
            "exact45 selection_ready disagrees with execution inventory"
        )
    return expected, observed


def _stage_root(manifest_path: Path) -> Path:
    parts = manifest_path.parts
    indices = [index for index, value in enumerate(parts) if value == "batches"]
    if not indices:
        raise TwoStageReportError(
            f"batch manifest has no stage root: {manifest_path}"
        )
    return Path(*parts[: indices[-1]])


def _load_verified_parquet_rows(
    *,
    stage_manifest: VerifiedJson,
    table_reference: Mapping[str, object],
    role: str,
) -> list[dict[str, object]]:
    relative = table_reference.get("object_path")
    if not isinstance(relative, str) or not relative:
        raise TwoStageReportError(f"{role} object_path is missing")
    root = _stage_root(stage_manifest.path)
    object_path = (root / relative).resolve()
    try:
        object_path.relative_to(root)
    except ValueError as error:
        raise TwoStageReportError(
            f"{role} path escapes the verified stage root"
        ) from error
    try:
        payload = object_path.read_bytes()
    except OSError as error:
        raise TwoStageReportError(
            f"{role} object cannot be read: {object_path}"
        ) from error
    observed_sha = _require_content_addressed_path(object_path, payload)
    if observed_sha != _require_sha(
        table_reference.get("object_sha256"),
        field=f"{role} object_sha256",
    ):
        raise TwoStageReportError(f"{role} object digest drifted")
    expected_bytes = _require_int(
        table_reference.get("byte_length"), field=f"{role} byte_length"
    )
    if len(payload) != expected_bytes:
        raise TwoStageReportError(f"{role} byte length drifted")
    metadata = pq.ParquetFile(object_path).metadata
    expected_rows = _require_int(
        table_reference.get("row_count"), field=f"{role} row_count"
    )
    if metadata.num_rows != expected_rows:
        raise TwoStageReportError(f"{role} row count drifted")
    table = pq.read_table(object_path)
    expected_columns = table_reference.get("schema_columns")
    if (
        not isinstance(expected_columns, list)
        or list(table.column_names) != expected_columns
    ):
        raise TwoStageReportError(f"{role} schema columns drifted")
    return table.to_pylist()


def _stage_a_metrics(stage_a: VerifiedJson) -> dict[str, object]:
    references = stage_a.document.get("calibration_tables")
    if not isinstance(references, list):
        raise TwoStageReportError(
            "Stage A calibration table inventory is missing"
        )
    summary = next(
        (
            reference
            for reference in references
            if isinstance(reference, dict)
            and reference.get("name") == "calibration_summary"
        ),
        None,
    )
    if summary is None:
        raise TwoStageReportError(
            "Stage A calibration_summary table is missing"
        )
    rows = _load_verified_parquet_rows(
        stage_manifest=stage_a,
        table_reference=summary,
        role="Stage A calibration_summary",
    )
    if len(rows) != 1:
        raise TwoStageReportError(
            "Stage A calibration_summary must have exactly one row"
        )
    return rows[0]


def _verify_selection(
    artifact: VerifiedJson,
    *,
    exact45: VerifiedJson,
) -> None:
    document = artifact.document
    if (
        document.get("experiment_id") != "X-15"
        or document.get("cohort") != "development"
        or document.get("selection_venue") != "polymarket"
    ):
        raise TwoStageReportError(
            "selection result has the wrong experiment/cohort/venue"
        )
    source = document.get("input_selection_batch")
    if not isinstance(source, dict):
        raise TwoStageReportError(
            "selection result does not bind its exact45 input"
        )
    if (
        source.get("manifest_file_sha256") != exact45.file_sha256
        or source.get("batch_sha256")
        != exact45.document.get("batch_sha256")
    ):
        raise TwoStageReportError(
            "selection result does not bind the verified exact45 batch"
        )
    status = document.get("decision_status")
    winner = document.get("winner")
    if status not in {"MODEL_ADVANCE", "NO_MODEL_ADVANCE"}:
        raise TwoStageReportError(
            f"selection result has an invalid decision_status: {status!r}"
        )
    if (status == "MODEL_ADVANCE") is not isinstance(winner, dict):
        raise TwoStageReportError(
            "selection decision and winner payload disagree"
        )


def _verify_optional_downstream(
    artifact: VerifiedJson,
    *,
    role: str,
    selection: VerifiedJson,
) -> None:
    if artifact.document.get("cohort") != "development":
        raise TwoStageReportError(
            f"{role} result is not a development result"
        )
    declared = artifact.document.get(
        "selection_manifest_file_sha256"
    )
    if declared != selection.file_sha256:
        raise TwoStageReportError(
            f"{role} result does not bind the verified selection result"
        )


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: object, *, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    classes: str = "",
) -> str:
    head = "".join(f"<th>{_esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_esc(_fmt(item))}</td>" for item in row)
        + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{_esc(classes)}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _execution_heatmap(
    expected: tuple[tuple[str, str, str], ...],
    observed: tuple[tuple[str, str, str], ...],
) -> str:
    folds = sorted({fold for fold, _, _ in expected})
    designs = sorted({(model, block) for _, model, block in expected})
    observed_set = set(observed)
    header = "".join(f"<th>{_esc(fold)}</th>" for fold in folds)
    rows: list[str] = []
    for model, block in designs:
        cells = []
        for fold in folds:
            coordinate = (fold, model, block)
            if coordinate not in set(expected):
                css, value = "na", "N/A"
            elif coordinate in observed_set:
                css, value = "done", "完成"
            else:
                css, value = "pending", "待运行"
            cells.append(
                f'<td class="heat {css}" '
                f'title="{_esc("|".join(coordinate))}">{value}</td>'
            )
        rows.append(
            "<tr>"
            f"<th>{_esc(model)} · {_esc(block)}</th>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="heatmap">'
        f"<thead><tr><th>真实执行单元</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _feature_model_rows(
    expected: tuple[tuple[str, str, str], ...],
    observed: tuple[tuple[str, str, str], ...],
) -> list[tuple[object, ...]]:
    descriptions = {
        "D0": "L/H 时间控制",
        "D1": "D0 + 价格、staleness、30/60s 成交活跃度",
        "D2": "D1 + 比赛时钟、比分、球权、档数、距离、场地",
        "D3": "D2 + event/action 与 Factor V4 multi-hot",
        "D4": "事件后 Stage A 字段；decision time 不可用，禁止",
    }
    expected_counts: dict[tuple[str, str], int] = {}
    observed_counts: dict[tuple[str, str], int] = {}
    for _, model, block in expected:
        expected_counts[(model, block)] = (
            expected_counts.get((model, block), 0) + 1
        )
    for _, model, block in observed:
        observed_counts[(model, block)] = (
            observed_counts.get((model, block), 0) + 1
        )
    return [
        (
            model,
            block,
            descriptions.get(block, "artifact-defined feature block"),
            observed_counts.get((model, block), 0),
            count,
            (
                "COMPLETE"
                if observed_counts.get((model, block), 0) == count
                else "PARTIAL"
            ),
        )
        for (model, block), count in sorted(expected_counts.items())
    ]


def _selection_section(
    selection: VerifiedJson | None,
    *,
    development_blocked: bool,
) -> str:
    blocked_callout = (
        '<div class="callout fail"><b>'
        "Selection promotion gate：BLOCKED</b>"
        "<br>当前 exact45 只可作为诊断，不能进入 shortlist 或 holdout。"
        "</div>"
        if development_blocked
        else ""
    )
    if selection is None:
        return (
            blocked_callout
            + '<div class="callout pending"><b>'
            "Stage B selection artifact：PENDING</b>"
            "<br>selection result 尚未发布；本报告不从 partial shards "
            "推断 winner 或方向指标。</div>"
        )
    document = selection.document
    status = document["decision_status"]
    winner = document.get("winner")
    if not isinstance(winner, dict):
        return (
            blocked_callout
            + f'<div class="callout neutral"><b>'
            f"Stage B artifact：{_esc(status)}</b>"
            "<br>没有模型通过冻结的 joint + direction 选择门。</div>"
        )
    rows = [
        (
            "Joint integrated improvement",
            winner.get("joint_integrated_mean_improvement"),
            winner.get("joint_integrated_ci"),
        ),
        (
            "Direction log-loss improvement",
            winner.get(
                "direction_log_loss_integrated_mean_improvement"
            ),
            winner.get("direction_log_loss_integrated_ci"),
        ),
        (
            "Direction Brier improvement",
            winner.get(
                "direction_brier_integrated_mean_improvement"
            ),
            winner.get("direction_brier_integrated_ci"),
        ),
        (
            "L=3→H=30 joint anchor",
            winner.get("joint_anchor_mean_improvement"),
            f"{_fmt(winner.get('joint_anchor_game_count'))} games",
        ),
    ]
    diagnostic_css = "neutral" if development_blocked else "pass"
    diagnostic_label = (
        "Stage B artifact：MODEL_ADVANCE · DIAGNOSTIC ONLY"
        if development_blocked
        else "Stage B：MODEL_ADVANCE"
    )
    return (
        blocked_callout
        + f'<div class="callout {diagnostic_css}"><b>'
        f"{diagnostic_label}</b>"
        f"<br>Winner：{_esc(winner.get('model_id'))} / "
        f"{_esc(winner.get('feature_block_id'))}</div>"
        + _table(("方向/联合指标", "真实估计", "95% CI / 支持"), rows)
    )


def _kalshi_section(
    selection: VerifiedJson | None,
    kalshi: VerifiedJson | None,
    *,
    development_blocked: bool,
) -> str:
    if development_blocked:
        diagnostic = (
            "尚无 Kalshi artifact"
            if kalshi is None
            else (
                "verified artifact transport_gate_passed="
                f"{_fmt(kalshi.document.get('transport_gate_passed'))}"
            )
        )
        rows = (
            []
            if kalshi is None
            else [
                ("Model", kalshi.document.get("candidate_model_id")),
                (
                    "Feature block",
                    kalshi.document.get("candidate_feature_block_id"),
                ),
                (
                    "Paired games",
                    kalshi.document.get("paired_game_count"),
                ),
                (
                    "Transport vs B0 improvement",
                    kalshi.document.get(
                        "transport_b0_mean_improvement"
                    ),
                ),
            ]
        )
        return (
            '<div class="callout fail"><b>Kalshi gate：BLOCKED</b>'
            f"<br>{_esc(diagnostic)}；不得作为 cross-venue promotion。"
            "</div>"
            + (
                _table(("Kalshi diagnostic", "真实值"), rows)
                if rows
                else ""
            )
        )
    if (
        selection is not None
        and selection.document.get("decision_status") == "NO_MODEL_ADVANCE"
    ):
        return (
            '<div class="callout neutral"><b>'
            "Kalshi 验证：NOT_APPLICABLE</b>"
            "<br>Polymarket 没有冻结 winner，因此不能强行 transport。</div>"
        )
    if kalshi is None:
        return (
            '<div class="callout pending"><b>Kalshi 验证：PENDING</b>'
            "<br>尚无绑定当前 selection result 的 verified publication。</div>"
        )
    document = kalshi.document
    passed = document.get("transport_gate_passed")
    if passed is True:
        status = "TRANSPORT PASS"
        css = "pass"
    elif passed is False:
        status = "TRANSPORT FAIL"
        css = "fail"
    else:
        status = str(document.get("diagnostic_status", "VERIFIED RESULT"))
        css = "neutral"
    rows = [
        ("Model", document.get("candidate_model_id")),
        ("Feature block", document.get("candidate_feature_block_id")),
        ("Paired games", document.get("paired_game_count")),
        ("Paired identities", document.get("paired_identity_count")),
        (
            "Transport vs B0 improvement",
            document.get("transport_b0_mean_improvement"),
        ),
        (
            "Transport vs B0 95% CI",
            (
                document.get("transport_b0_ci_low"),
                document.get("transport_b0_ci_high"),
            ),
        ),
    ]
    return (
        f'<div class="callout {css}"><b>Kalshi：{_esc(status)}</b></div>'
        + _table(("Kalshi development 指标", "真实值"), rows)
    )


def _magnitude_section(
    selection: VerifiedJson | None,
    magnitude: VerifiedJson | None,
    *,
    development_blocked: bool,
) -> str:
    if development_blocked:
        return (
            '<div class="callout fail"><b>Magnitude gate：BLOCKED</b>'
            "<br>Probability development gate 未通过，不能把条件幅度"
            "结果晋升为正式 follow-up。</div>"
        )
    if (
        selection is not None
        and selection.document.get("decision_status") == "NO_MODEL_ADVANCE"
    ):
        return (
            '<div class="callout neutral"><b>'
            "Magnitude：NOT_APPLICABLE</b>"
            "<br>没有 probability winner，不能启动条件幅度 follow-up。</div>"
        )
    if magnitude is None:
        return (
            '<div class="callout pending"><b>Magnitude：PENDING</b>'
            "<br>条件幅度结果尚未发布；不补造 UP/DOWN 分布。</div>"
        )
    document = magnitude.document
    metrics = document.get("metrics", document.get("magnitude_metrics"))
    if not isinstance(metrics, list):
        return (
            '<div class="callout neutral"><b>Magnitude：'
            f"{_esc(document.get('status', 'VERIFIED'))}</b>"
            "<br>已验证发布物未包含可展示的 row-level metrics table。</div>"
        )
    rows: list[tuple[object, ...]] = []
    for record in metrics:
        if not isinstance(record, dict):
            raise TwoStageReportError(
                "magnitude metrics table contains a non-object row"
            )
        rows.append(
            (
                record.get("direction"),
                record.get("quantile"),
                record.get("estimate"),
                record.get("pinball_loss"),
                record.get("approx_crps"),
                record.get("interval_coverage"),
                record.get("support_games"),
            )
        )
    return (
        '<div class="callout pass"><b>Magnitude：'
        f"{_esc(document.get('status', 'VERIFIED'))}</b></div>"
        + _table(
            (
                "方向",
                "分位数",
                "估计",
                "Pinball",
                "Approx CRPS",
                "Coverage",
                "Games",
            ),
            rows,
        )
    )


def _artifact_rows(
    artifacts: Sequence[tuple[str, VerifiedJson | None]],
) -> list[tuple[object, ...]]:
    return [
        (
            label,
            "VERIFIED" if artifact is not None else "NOT_PUBLISHED",
            artifact.file_sha256 if artifact is not None else None,
            str(artifact.path) if artifact is not None else None,
            (
                artifact.document.get("schema")
                if artifact is not None
                else None
            ),
        )
        for label, artifact in artifacts
    ]


def _render_html(
    *,
    factor: VerifiedJson,
    stage_a: VerifiedJson,
    exact45: VerifiedJson,
    expected: tuple[tuple[str, str, str], ...],
    observed: tuple[tuple[str, str, str], ...],
    stage_a_metrics: Mapping[str, object],
    selection: VerifiedJson | None,
    kalshi: VerifiedJson | None,
    magnitude: VerifiedJson | None,
    development_gate: str,
    development_blockers: tuple[str, ...],
) -> str:
    factor_document = factor.document
    counts = factor_document["aggregate_counts"]
    registry = factor_document["factor_registry"]
    exact_document = exact45.document
    complete = len(observed) == len(expected)
    stage_b_status = (
        str(exact_document.get("publication_status"))
        if not complete
        else "EXACT45_COMPLETE"
    )
    development_blocked = development_gate == "BLOCKED"
    blocker_rows = [
        (
            blocker,
            (
                "OT 样本范围污染；旧 exact45 仅作诊断"
                if blocker == "OT_SCOPE_CONTAMINATION"
                else (
                    "约 70% 可得的 PIT p_before 未传播到 Stage B"
                    if blocker
                    == "PIT_PRESTATE_REFERENCE_NOT_PROPAGATED"
                    else "审核登记的 development blocker"
                )
            ),
        )
        for blocker in development_blockers
    ]
    factor_rows = [
        ("比赛", factor_document.get("game_count")),
        ("Raw PBP", counts.get("raw_pbp_rows")),
        (
            "Canonical factor events",
            counts.get("canonical_factor_events_rows"),
        ),
        ("Factor hits", counts.get("factor_hits_rows")),
        (
            "Coverage rows",
            counts.get("factor_coverage_audit_rows"),
        ),
        ("Silent loss", counts.get("raw_rows_silently_dropped")),
        ("Factor definitions", registry.get("factor_count")),
    ]
    stage_a_rows = [
        ("Model", stage_a.document["model"].get("model_id")),
        ("Model version", stage_a.document["model"].get("model_version")),
        ("Evaluated games", stage_a_metrics.get("evaluated_games")),
        ("Evaluated states", stage_a_metrics.get("evaluated_states")),
        ("Brier", stage_a_metrics.get("brier")),
        ("Log loss", stage_a_metrics.get("log_loss")),
        ("Calibration slope", stage_a_metrics.get("calibration_slope")),
        (
            "Calibration intercept",
            stage_a_metrics.get("calibration_intercept"),
        ),
    ]
    artifacts = (
        ("Factor V4", factor),
        ("Stage A", stage_a),
        ("exact45 authority", exact45),
        ("selection result", selection),
        ("Kalshi validation", kalshi),
        ("magnitude result", magnitude),
    )
    claim_rows = [
        (
            "Factor V4",
            factor.document.get("claim_boundary"),
        ),
        ("Stage A", stage_a.document.get("claim_boundary")),
        (
            "Stage B",
            "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC",
        ),
        (
            "明确排除",
            "不声称 live latency、因果、bid/ask execution、L2/OFI、"
            "queue、maker fill、可交易 alpha；holdout 尚未读取。",
        ),
    ]
    css = """
    :root{color-scheme:light;--ink:#17231f;--muted:#66746e;--line:#d9dfda;
    --paper:#f5f3ec;--panel:#fffdfa;--green:#167a55;--green2:#d9f1e7;
    --amber:#9a6714;--amber2:#fff0c9;--red:#a43c36;--red2:#f9dcd8;
    --blue:#315f9a;--blue2:#dfeaf8}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
    "Hiragino Sans GB","Microsoft YaHei",sans-serif}.page{max-width:1480px;
    margin:auto;padding:34px 42px 80px}h1{font:700 36px/1.15 Georgia,
    "Songti SC",serif;margin:0 0 10px}h2{font:700 23px/1.2 Georgia,
    "Songti SC",serif;margin:36px 0 14px}h3{font-size:17px;margin:22px 0 10px}
    .lede{color:var(--muted);max-width:920px}.badges{display:flex;gap:9px;
    flex-wrap:wrap;margin:20px 0}.badge{border:1px solid var(--line);background:
    var(--panel);padding:7px 11px;border-radius:999px;font-weight:650}.grid{
    display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{
    background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:20px;box-shadow:0 4px 16px rgba(24,38,32,.04)}.callout{
    border-left:5px solid var(--blue);background:var(--blue2);padding:13px 15px;
    margin:12px 0;border-radius:8px}.callout.pass{border-color:var(--green);
    background:var(--green2)}.callout.pending{border-color:var(--amber);
    background:var(--amber2)}.callout.fail{border-color:var(--red);
    background:var(--red2)}.callout.neutral{border-color:#78837e;background:#e9ecea}
    .table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
    background:var(--panel)}table{border-collapse:collapse;width:100%;min-width:680px}
    th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;
    vertical-align:top}thead th{background:#ecefe9;font-size:12px;text-transform:
    uppercase;letter-spacing:.04em}tbody tr:last-child td,tbody tr:last-child th{
    border-bottom:0}.heat{text-align:center!important;font-weight:700}.heat.done{
    background:var(--green2);color:var(--green)}.heat.pending{background:var(--amber2);
    color:var(--amber)}.heat.na{background:#eef0ee;color:#88918d}.path{font:12px/1.4
    ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}.foot{
    margin-top:36px;color:var(--muted);font-size:13px}
    @media(max-width:850px){.page{padding:24px 16px}.grid{grid-template-columns:1fr}}
    """
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL exact-153 两阶段研究状态报告</title><style>{css}</style></head>
<body><main class="page">
<h1>NFL exact-153 两阶段研究状态报告</h1>
<p class="lede">离线报告 · 只读取 SHA-256 验证的 content-addressed artifacts。
部分阶段未完成时只报告进度，不从 partial shards 推断结论。</p>
<div class="badges">
<span class="badge">Factor V4：PASS</span>
<span class="badge">Stage A：PASS</span>
<span class="badge">Stage B：{_esc(stage_b_status)}</span>
<span class="badge">Development Gate：{_esc(development_gate)}</span>
<span class="badge">153 场 development</span>
<span class="badge">Holdout：SEALED</span>
</div>

<h2>1. 当前 State / Progress</h2>
<div class="callout {'fail' if development_blocked else 'pass'}"><b>
Development Gate：{_esc(development_gate)}</b><br>
{'现有 exact45/selection/Kalshi 只能作为诊断，不得 shortlist 或读取 holdout。'
if development_blocked else
'当前没有登记的 development blocker；后续仍须遵守 selection/Kalshi/shortlist gates。'}
</div>
{_table(("Blocker ID","审核含义"), blocker_rows) if blocker_rows else ''}
<div class="grid">
<section class="card"><h3>153 场事实覆盖</h3>
{_table(("项目","真实值"), factor_rows)}</section>
<section class="card"><h3>Stage A：体育 reference</h3>
{_table(("指标","真实值"), stage_a_rows)}</section>
</div>
<section class="card" style="margin-top:18px"><h3>exact45 运行进度</h3>
<div class="callout {'pass' if complete else 'pending'}"><b>{_esc(stage_b_status)}</b>
<br>{len(observed):,} / {len(expected):,} 个冻结执行单元已发布并进入 authority。</div>
{_execution_heatmap(expected, observed)}</section>

<h2>2. 模型与 Feature Block</h2>
{_table(("模型","Block","PIT 输入","完成 folds","计划 folds","状态"),
_feature_model_rows(expected, observed))}
<div class="callout neutral"><b>D4：UNSUPPORTED_FEATURE_BLOCK</b><br>
Stage A 的 event-after reference 字段在 L=1/2/3/5/10 秒 decision rows 上
没有 PIT support，因此不进入 Stage B selection。</div>

<h2>3. Polymarket Stage B 方向概率</h2>
{_selection_section(selection, development_blocked=development_blocked)}

<h2>4. Kalshi 冻结 Winner 验证</h2>
{_kalshi_section(selection, kalshi,
development_blocked=development_blocked)}

<h2>5. 条件幅度 / Magnitude</h2>
{_magnitude_section(selection, magnitude,
development_blocked=development_blocked)}

<h2>6. Holdout Gate</h2>
<div class="callout {'fail' if development_blocked else 'pending'}"><b>
Holdout gate：{'BLOCKED' if development_blocked else 'SEALED'}</b><br>
81 场 holdout reaction read count 保持 0；{'development blockers 清除并重新发布 authority 前禁止读取。'
if development_blocked else '只有 verified shortlist lock 后才允许一次性读取。'}
</div>

<h2>7. Verified Artifacts / Lineage</h2>
{_table(("Artifact","状态","文件 SHA-256","绝对路径","Schema"),
_artifact_rows(artifacts), classes="path")}

<h2>8. Claim Boundary</h2>
{_table(("层","允许的结论"), claim_rows)}
<p class="foot">本页中的颜色热力图只编码 exact45 manifest 的真实
observed/missing execution cell table；不编码未发布的模型效果。</p>
</main></body></html>"""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise TwoStageReportError(
                f"content-addressed collision: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_two_stage_report(
    *,
    factor_v4_manifest_path: str | Path,
    stage_a_manifest_path: str | Path,
    exact45_manifest_path: str | Path,
    development_gate: str,
    development_blockers: Sequence[str],
    output_root: str | Path,
    selection_manifest_path: str | Path | None = None,
    kalshi_manifest_path: str | Path | None = None,
    magnitude_manifest_path: str | Path | None = None,
) -> TwoStageReportPublication:
    """Verify inputs and publish one content-addressed offline HTML report."""

    verified_gate, verified_blockers = _verify_development_gate(
        development_gate,
        development_blockers,
    )
    factor = _load_verified_json(
        factor_v4_manifest_path,
        role="Factor V4",
        allowed_schemas={"nfl_x13_exact153_fact_batch_index_v4"},
    )
    stage_a = _load_verified_json(
        stage_a_manifest_path,
        role="Stage A",
        allowed_schemas={"nfl_x15_stage_a_batch_index_v1"},
    )
    exact45 = _load_verified_json(
        exact45_manifest_path,
        role="exact45 authority",
        allowed_schemas={"nfl_x15_selection_batch_v3"},
    )
    _verify_factor_manifest(factor)
    _verify_stage_a_manifest(stage_a)
    expected, observed = _verify_exact45_manifest(exact45)
    calibration = _stage_a_metrics(stage_a)

    selection = (
        _load_verified_json(
            selection_manifest_path,
            role="Stage B selection",
            allowed_schemas={
                "nfl_x15_stage_b_v3_selection_manifest_v1"
            },
        )
        if selection_manifest_path is not None
        else None
    )
    if selection is not None:
        if exact45.document.get("selection_ready") is not True:
            raise TwoStageReportError(
                "selection result cannot be paired with a partial exact45 "
                "authority"
            )
        _verify_selection(selection, exact45=exact45)

    kalshi = (
        _load_verified_json(
            kalshi_manifest_path,
            role="Kalshi validation",
        )
        if kalshi_manifest_path is not None
        else None
    )
    magnitude = (
        _load_verified_json(
            magnitude_manifest_path,
            role="magnitude",
        )
        if magnitude_manifest_path is not None
        else None
    )
    for artifact, role in (
        (kalshi, "Kalshi validation"),
        (magnitude, "magnitude"),
    ):
        if artifact is not None:
            if (
                selection is None
                or selection.document.get("decision_status")
                != "MODEL_ADVANCE"
            ):
                raise TwoStageReportError(
                    f"{role} result requires a verified MODEL_ADVANCE "
                    "selection"
                )
            _verify_optional_downstream(
                artifact,
                role=role,
                selection=selection,
            )

    rendered = _render_html(
        factor=factor,
        stage_a=stage_a,
        exact45=exact45,
        expected=expected,
        observed=observed,
        stage_a_metrics=calibration,
        selection=selection,
        kalshi=kalshi,
        magnitude=magnitude,
        development_gate=verified_gate,
        development_blockers=verified_blockers,
    )
    html_payload = rendered.encode("utf-8")
    html_sha = _sha256(html_payload)
    digest = html_sha.removeprefix("sha256:")
    root = Path(output_root).resolve()
    html_path = (
        root / "objects" / "sha256" / digest[:2] / f"{digest}.html"
    )
    _atomic_write(html_path, html_payload)

    inputs = {
        "factor_v4": {
            "path": str(factor.path),
            "file_sha256": factor.file_sha256,
        },
        "stage_a": {
            "path": str(stage_a.path),
            "file_sha256": stage_a.file_sha256,
        },
        "exact45": {
            "path": str(exact45.path),
            "file_sha256": exact45.file_sha256,
        },
        "selection": (
            {
                "path": str(selection.path),
                "file_sha256": selection.file_sha256,
            }
            if selection is not None
            else None
        ),
        "kalshi": (
            {
                "path": str(kalshi.path),
                "file_sha256": kalshi.file_sha256,
            }
            if kalshi is not None
            else None
        ),
        "magnitude": (
            {
                "path": str(magnitude.path),
                "file_sha256": magnitude.file_sha256,
            }
            if magnitude is not None
            else None
        ),
    }
    report_manifest = {
        "schema": "nfl_x15_two_stage_report_manifest_v1",
        "experiment_id": "X-15",
        "cohort": "development",
        "game_count": 153,
        "development_gate": verified_gate,
        "development_blockers": list(verified_blockers),
        "inputs": inputs,
        "stage_status": {
            "factor_v4": "PASS",
            "stage_a": "PASS",
            "exact45": (
                "COMPLETE"
                if len(observed) == len(expected)
                else "PARTIAL"
            ),
            "selection": (
                "BLOCKED"
                if verified_gate == "BLOCKED"
                else (
                    selection.document["decision_status"]
                    if selection is not None
                    else "PENDING"
                )
            ),
            "kalshi": (
                "BLOCKED"
                if verified_gate == "BLOCKED"
                else (
                    "NOT_APPLICABLE"
                    if selection is not None
                    and selection.document["decision_status"]
                    == "NO_MODEL_ADVANCE"
                    else "PUBLISHED" if kalshi is not None else "PENDING"
                )
            ),
            "magnitude": (
                "BLOCKED"
                if verified_gate == "BLOCKED"
                else (
                    "NOT_APPLICABLE"
                    if selection is not None
                    and selection.document["decision_status"]
                    == "NO_MODEL_ADVANCE"
                    else (
                        "PUBLISHED" if magnitude is not None else "PENDING"
                    )
                )
            ),
            "holdout": (
                "BLOCKED" if verified_gate == "BLOCKED" else "SEALED"
            ),
        },
        "diagnostic_status": {
            "selection": (
                selection.document["decision_status"]
                if selection is not None
                else "NOT_PUBLISHED"
            ),
            "kalshi": (
                "PUBLISHED" if kalshi is not None else "NOT_PUBLISHED"
            ),
            "magnitude": (
                "PUBLISHED"
                if magnitude is not None
                else "NOT_PUBLISHED"
            ),
        },
        "html": {
            "path": str(html_path),
            "file_sha256": html_sha,
            "byte_length": len(html_payload),
        },
        "holdout_reaction_accessed": False,
        "claim_boundary": (
            "HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC; "
            "not live latency, causality, execution, L2, fill, or alpha "
            "evidence"
        ),
    }
    manifest_payload = _canonical_bytes(report_manifest)
    manifest_sha = _sha256(manifest_payload)
    manifest_digest = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        root
        / "manifests"
        / "sha256"
        / manifest_digest[:2]
        / f"{manifest_digest}.report-manifest.json"
    )
    _atomic_write(manifest_path, manifest_payload)
    return TwoStageReportPublication(
        html_path=html_path,
        html_sha256=html_sha,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-v4-manifest", type=Path, required=True)
    parser.add_argument("--stage-a-manifest", type=Path, required=True)
    parser.add_argument("--exact45-manifest", type=Path, required=True)
    parser.add_argument(
        "--development-gate",
        choices=("OPEN", "BLOCKED"),
        required=True,
        help=(
            "Current development promotion gate. BLOCKED requires at "
            "least one --development-blocker."
        ),
    )
    parser.add_argument(
        "--development-blocker",
        action="append",
        default=[],
        help=(
            "Stable blocker ID; repeat for multiple blockers, e.g. "
            "OT_SCOPE_CONTAMINATION."
        ),
    )
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--kalshi-manifest", type=Path)
    parser.add_argument("--magnitude-manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = render_two_stage_report(
        factor_v4_manifest_path=args.factor_v4_manifest,
        stage_a_manifest_path=args.stage_a_manifest,
        exact45_manifest_path=args.exact45_manifest,
        development_gate=args.development_gate,
        development_blockers=tuple(args.development_blocker),
        selection_manifest_path=args.selection_manifest,
        kalshi_manifest_path=args.kalshi_manifest,
        magnitude_manifest_path=args.magnitude_manifest,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "html_path": str(publication.html_path),
                "html_sha256": publication.html_sha256,
                "manifest_path": str(publication.manifest_path),
                "manifest_sha256": publication.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
