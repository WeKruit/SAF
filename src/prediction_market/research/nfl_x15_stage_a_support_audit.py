"""Publish the decision-time Stage-A feature-support audit for NFL X-15.

The audit is deliberately reaction-blind.  It reads only the already-published
development PanelV2 through its byte/semantic verifier and asks whether the
four Stage-A reference values are actually available on rows that may enter a
Stage-B decision model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Final

import pandas as pd

from prediction_market.research.nfl_x15_development_panel import (
    VerifiedDiagnosticPanelPartition,
    iter_verified_diagnostic_panel_partitions,
)


SCHEMA: Final[str] = "nfl_x15_stage_a_decision_support_audit_v1"
BUILDER_VERSION: Final[str] = (
    "nfl-x15-stage-a-decision-support-audit-v1"
)
REFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    "p_before_home",
    "p_after_home",
    "reference_delta_home",
    "reference_gap_at_landmark",
)
EXPECTED_GAME_COUNT: Final[int] = 153


class StageASupportAuditError(RuntimeError):
    """The verified development panel cannot support a truthful audit."""


@dataclass(frozen=True, slots=True)
class StageASupportAuditPublication:
    manifest_path: Path
    manifest_file_sha256: str
    audit_sha256: str
    decision_eligible_rows: int
    feature_block_status: str


def _canonical(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {
            str(key): _canonical(child)
            for key, child in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not pd.notna(value):
            return None
        return value
    return str(value)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in value[7:]
        )
    ):
        raise StageASupportAuditError(
            f"{field} must be a lowercase sha256 digest"
        )
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise StageASupportAuditError(
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


def _summarize_partitions(
    partitions: Iterable[VerifiedDiagnosticPanelPartition],
) -> dict[str, object]:
    total_rows = 0
    decision_rows = 0
    overall_nonnull = {column: 0 for column in REFERENCE_COLUMNS}
    decision_nonnull = {column: 0 for column in REFERENCE_COLUMNS}
    decision_status_counts: dict[str, int] = {}
    game_ids: set[str] = set()
    batch_file_hashes: set[str] = set()
    batch_hashes: set[str] = set()
    cohort_hashes: set[str] = set()
    mapping_hashes: set[str] = set()

    for partition in partitions:
        if not isinstance(partition, VerifiedDiagnosticPanelPartition):
            raise StageASupportAuditError(
                "audit partitions must be verified PanelV2 partitions"
            )
        frame = partition.panel
        required = {
            "decision_eligible",
            "stage_a_status",
            *REFERENCE_COLUMNS,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise StageASupportAuditError(
                f"{partition.game_id} is missing audit columns: {missing}"
            )
        if partition.game_id in game_ids:
            raise StageASupportAuditError(
                f"duplicate verified game partition: {partition.game_id}"
            )
        eligibility = frame["decision_eligible"]
        if not eligibility.map(lambda value: isinstance(value, bool)).all():
            raise StageASupportAuditError(
                f"{partition.game_id} decision_eligible is not boolean"
            )
        selected = frame.loc[eligibility]
        total_rows += len(frame)
        decision_rows += len(selected)
        for column in REFERENCE_COLUMNS:
            overall_nonnull[column] += int(frame[column].notna().sum())
            decision_nonnull[column] += int(
                selected[column].notna().sum()
            )
        for status, count in selected["stage_a_status"].value_counts(
            dropna=False
        ).items():
            key = "<MISSING>" if pd.isna(status) else str(status)
            decision_status_counts[key] = (
                decision_status_counts.get(key, 0) + int(count)
            )
        game_ids.add(partition.game_id)
        batch_file_hashes.add(partition.batch_manifest_file_sha256)
        batch_hashes.add(partition.batch_sha256)
        cohort_hashes.add(partition.cohort_authority_sha256)
        mapping_hashes.add(partition.cohort_mapping_sha256)

    if len(game_ids) != EXPECTED_GAME_COUNT:
        raise StageASupportAuditError(
            "Stage-A support audit requires exact-153 development games"
        )
    for values, field in (
        (batch_file_hashes, "panel batch file"),
        (batch_hashes, "panel batch"),
        (cohort_hashes, "cohort authority"),
        (mapping_hashes, "cohort mapping"),
    ):
        if len(values) != 1:
            raise StageASupportAuditError(
                f"verified partitions disagree on {field}"
            )
        _require_sha256(next(iter(values)), field=field)

    reference_supported = any(decision_nonnull.values())
    feature_block_status = (
        "SUPPORTED"
        if reference_supported
        else "UNSUPPORTED_FEATURE_BLOCK"
    )
    return {
        "verified_game_count": len(game_ids),
        "total_panel_rows": total_rows,
        "decision_eligible_rows": decision_rows,
        "overall_nonnull_counts": overall_nonnull,
        "nonnull_counts": decision_nonnull,
        "decision_stage_a_status_counts": dict(
            sorted(decision_status_counts.items())
        ),
        "feature_block_status": {"D4": feature_block_status},
        "selection_eligible": {"D4": reference_supported},
        "support_reason": {
            "D4": (
                "REFERENCE_VALUES_AVAILABLE_ON_DECISION_ROWS"
                if reference_supported
                else (
                    "ZERO_NON_NULL_STAGE_A_REFERENCE_FEATURES_ON_"
                    "DECISION_ELIGIBLE_ROWS"
                )
            )
        },
        "panel_batch_file_sha256": next(iter(batch_file_hashes)),
        "panel_batch_sha256": next(iter(batch_hashes)),
        "cohort_authority_sha256": next(iter(cohort_hashes)),
        "cohort_mapping_sha256": next(iter(mapping_hashes)),
    }


def publish_stage_a_decision_support_audit(
    *,
    project_root: str | Path,
    panel_batch_manifest_path: str | Path,
    panel_batch_file_sha256: str,
    output_root: str | Path,
) -> StageASupportAuditPublication:
    """Verify exact-153 PanelV2 and publish one immutable support audit."""

    root = Path(project_root).resolve()
    panel_path = Path(panel_batch_manifest_path)
    resolved_panel_path = (
        panel_path
        if panel_path.is_absolute()
        else root / panel_path
    ).resolve()
    try:
        relative_panel_path = resolved_panel_path.relative_to(root)
    except ValueError as error:
        raise StageASupportAuditError(
            "panel batch manifest must remain under project_root"
        ) from error
    requested_file_sha256 = _require_sha256(
        panel_batch_file_sha256,
        field="panel_batch_file_sha256",
    )
    summary = _summarize_partitions(
        iter_verified_diagnostic_panel_partitions(
            project_root=root,
            batch_manifest_path=resolved_panel_path,
            batch_manifest_file_sha256=requested_file_sha256,
            expected_game_count=EXPECTED_GAME_COUNT,
        )
    )
    if summary["panel_batch_file_sha256"] != requested_file_sha256:
        raise StageASupportAuditError(
            "verified partitions disagree with requested panel file SHA"
        )
    material = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-15",
        "cohort": "development",
        "panel_batch_manifest_path": relative_panel_path.as_posix(),
        **summary,
        "holdout_reaction_accessed": False,
        "claim_boundary": (
            "DEVELOPMENT_DECISION_TIME_FEATURE_SUPPORT_AUDIT_"
            "NOT_MARKET_REACTION_RESULT"
        ),
    }
    audit_sha256 = _canonical_sha256(material)
    document = {**material, "audit_sha256": audit_sha256}
    payload = _canonical_bytes(document)
    file_sha256 = _bytes_sha256(payload)
    hexadecimal = file_sha256.removeprefix("sha256:")
    manifest_path = (
        Path(output_root).resolve()
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.stage-a-support-audit.json"
    )
    _atomic_write(manifest_path, payload)
    return StageASupportAuditPublication(
        manifest_path=manifest_path,
        manifest_file_sha256=file_sha256,
        audit_sha256=audit_sha256,
        decision_eligible_rows=int(summary["decision_eligible_rows"]),
        feature_block_status=str(
            summary["feature_block_status"]["D4"]
        ),
    )


__all__ = [
    "BUILDER_VERSION",
    "EXPECTED_GAME_COUNT",
    "REFERENCE_COLUMNS",
    "SCHEMA",
    "StageASupportAuditError",
    "StageASupportAuditPublication",
    "publish_stage_a_decision_support_audit",
]
