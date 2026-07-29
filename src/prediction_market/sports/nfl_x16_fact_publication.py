"""Verified, content-addressed publication of one NFL sports-fact review.

This is deliberately a draft expert-review artifact.  It reads no market
observations and does not register or publish an experimental result.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from prediction_market.reports.nfl_x16_fact_review import render_game_fact_review
from prediction_market.sports.nfl_x16_fact_extraction import (
    GameFactTables,
    NFLFactExtractionError,
    build_game_fact_tables,
)


class NFLFactPublicationError(ValueError):
    """A source or derived object cannot be published without mutation."""


@dataclass(frozen=True, slots=True)
class PublishedGameFactReview:
    manifest_path: Path
    manifest_sha256: str
    report_path: Path
    object_paths: Mapping[str, Path]


def _sha_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_content_addressed_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise NFLFactPublicationError("content-addressed object collision")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise NFLFactPublicationError("content-addressed publish race")
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_verified_source(
    object_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_dataset_id: str,
) -> tuple[Path, dict[str, object], str]:
    object_file = Path(object_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not object_file.is_file() or not manifest_file.is_file():
        raise NFLFactPublicationError("source object and manifest must exist")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NFLFactPublicationError("source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise NFLFactPublicationError("source manifest must be an object")
    if manifest.get("dataset_id") != expected_dataset_id:
        raise NFLFactPublicationError("source manifest dataset_id mismatch")
    observed_sha = _sha_file(object_file)
    if manifest.get("object_sha256") != observed_sha:
        raise NFLFactPublicationError("source object SHA-256 mismatch")
    return object_file, manifest, _sha_file(manifest_file)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _publish_object(
    output_root: Path,
    *,
    name: str,
    extension: str,
    media_type: str,
    payload: bytes,
    row_count: int | None,
) -> tuple[dict[str, object], Path]:
    sha = _sha_bytes(payload)
    digest = sha.removeprefix("sha256:")
    relative = Path("objects") / "sha256" / digest[:2] / f"{digest}.{extension}"
    target = output_root / relative
    _atomic_content_addressed_write(target, payload)
    record: dict[str, object] = {
        "name": name,
        "path": relative.as_posix(),
        "object_sha256": sha,
        "byte_length": len(payload),
        "media_type": media_type,
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record, target


def publish_single_game_fact_review(
    *,
    game_id: str,
    pbp_object_path: str | Path,
    pbp_manifest_path: str | Path,
    participation_object_path: str | Path,
    participation_manifest_path: str | Path,
    factor_registry_path: str | Path,
    output_root: str | Path,
) -> PublishedGameFactReview:
    """Publish one deterministic fact-description bundle for expert review."""

    pbp_file, pbp_manifest, pbp_manifest_file_sha = _read_verified_source(
        pbp_object_path,
        pbp_manifest_path,
        expected_dataset_id="DS-NFLVERSE",
    )
    participation_file, participation_manifest, participation_manifest_file_sha = (
        _read_verified_source(
            participation_object_path,
            participation_manifest_path,
            expected_dataset_id="DS-NFLVERSE-PARTICIPATION",
        )
    )
    registry_file = Path(factor_registry_path).resolve()
    try:
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NFLFactPublicationError("factor registry is not valid JSON") from exc
    if not isinstance(registry, dict):
        raise NFLFactPublicationError("factor registry must be an object")
    if registry.get("status") != "DRAFT_EXPERT_REVIEW":
        raise NFLFactPublicationError("single-game review requires a draft registry")

    pbp = pd.read_parquet(
        pbp_file,
        filters=[("game_id", "==", game_id)],
    )
    participation = pd.read_parquet(
        participation_file,
        filters=[("nflverse_game_id", "==", game_id)],
    )
    if pbp.empty:
        raise NFLFactPublicationError(f"game not found in frozen PBP: {game_id}")
    try:
        tables = build_game_fact_tables(
            pbp,
            participation,
            factor_registry=registry,
            pbp_source_sha256=str(pbp_manifest["object_sha256"]),
            participation_source_sha256=str(
                participation_manifest["object_sha256"]
            ),
        )
    except NFLFactExtractionError as exc:
        raise NFLFactPublicationError("game fact extraction failed") from exc

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    table_specs: tuple[tuple[str, pd.DataFrame], ...] = (
        ("canonical_factor_events", tables.events),
        ("factor_event_tags", tables.tags),
        ("factor_event_players", tables.players),
        ("player_availability_events", tables.availability),
        ("injury_evidence", tables.injury_evidence),
        ("factor_hits", tables.factor_hits),
        ("factor_coverage_audit", tables.factor_coverage),
        ("sports_row_reconciliation", tables.reconciliation),
    )
    object_records: list[dict[str, object]] = []
    object_paths: dict[str, Path] = {}
    for name, frame in table_specs:
        record, target = _publish_object(
            root,
            name=name,
            extension="parquet",
            media_type="application/vnd.apache.parquet",
            payload=_parquet_bytes(frame),
            row_count=len(frame),
        )
        object_records.append(record)
        object_paths[name] = target

    report_payload = render_game_fact_review(tables).encode("utf-8")
    report_record, report_path = _publish_object(
        root,
        name="game_fact_review",
        extension="html",
        media_type="text/html; charset=utf-8",
        payload=report_payload,
        row_count=None,
    )
    object_records.append(report_record)
    object_paths["game_fact_review"] = report_path

    reconciliation = tables.reconciliation
    manifest = {
        "schema": "SingleGameFactReviewBundleV1",
        "artifact_status": "DRAFT_EXPERT_REVIEW",
        "formal_experiment_registered": False,
        "game_id": game_id,
        "claim_boundary": "SPORTS_FACT_DESCRIPTION_ONLY_NO_MARKET_REACTION",
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "source_row_count": len(pbp),
        "source_row_preserved_count": int(
            reconciliation["raw_row_preserved"].fillna(False).sum()
        ),
        "source_refs": {
            "pbp": {
                "dataset_id": pbp_manifest["dataset_id"],
                "object_sha256": pbp_manifest["object_sha256"],
                "declared_manifest_sha256": pbp_manifest.get("manifest_sha256"),
                "manifest_file_sha256": pbp_manifest_file_sha,
                "license_status": pbp_manifest.get("license_status"),
            },
            "participation": {
                "dataset_id": participation_manifest["dataset_id"],
                "object_sha256": participation_manifest["object_sha256"],
                "declared_manifest_sha256": participation_manifest.get(
                    "manifest_sha256"
                ),
                "manifest_file_sha256": participation_manifest_file_sha,
                "license_status": participation_manifest.get("license_status"),
            },
            "factor_registry": {
                "path": str(registry_file),
                "file_sha256": _sha_file(registry_file),
                "semantic_sha256": tables.registry_sha256,
                "status": registry["status"],
            },
        },
        "counts": {
            "canonical_events": len(tables.events),
            "event_tags": len(tables.tags),
            "event_players": len(tables.players),
            "availability_events": len(tables.availability),
            "injury_evidence": len(tables.injury_evidence),
            "factor_hits": len(tables.factor_hits),
            "registered_factors": len(tables.factor_coverage),
        },
        "objects": sorted(object_records, key=lambda item: str(item["name"])),
    }
    manifest_payload = _canonical_json(manifest)
    manifest_sha = _sha_bytes(manifest_payload)
    digest = manifest_sha.removeprefix("sha256:")
    manifest_path = (
        root
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    _atomic_content_addressed_write(manifest_path, manifest_payload)
    return PublishedGameFactReview(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        report_path=report_path,
        object_paths=object_paths,
    )


__all__ = [
    "NFLFactPublicationError",
    "PublishedGameFactReview",
    "publish_single_game_fact_review",
]
