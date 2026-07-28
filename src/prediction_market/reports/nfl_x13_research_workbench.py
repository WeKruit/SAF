"""Offline, content-addressed NFL X-13 research workbench."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class ResearchWorkbenchError(ValueError):
    """The supplied evidence cannot support the governed workbench."""


Rows = pd.DataFrame | Sequence[Mapping[str, object]]
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CANONICAL_ARTIFACT_RE = re.compile(
    r"(?:^|/)sha256/([0-9a-f]{2})/([0-9a-f]{64})(?:[./]|$)"
)
_L2_CLAIM_BOUNDARY = "METHOD_VALIDATION_ONLY_NON_NFL_ALPHA"
_REQUIRED_PUBLICATION_TABLES = {
    "historical_distribution": ("summary", "histogram"),
    "reference_completion": ("reference_summary", "reference_exclusions"),
    "same_episode_paired": ("venue_diagnostics",),
    "venue_consistency": ("venue_consistency",),
}
_OPTIONAL_PUBLICATION_TABLES = {
    "tick_inventory": ("tick_inventory",),
    "calibration_bias": ("metrics", "reliability", "bias_proxies"),
    "pmxt_l2_method_audit": ("pmxt_l2_method_audit",),
    "shadow_metrics": ("shadow_metrics",),
}


@dataclass(frozen=True)
class ResearchWorkbenchEvidence:
    """Tables loaded from verified publication objects plus their audit."""

    tables: Mapping[str, pd.DataFrame]
    sources: Mapping[str, Mapping[str, object]]
    manifests: Mapping[str, Mapping[str, object]]
    source_audit: pd.DataFrame


def _clean(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _records(frame: pd.DataFrame | None) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    return [
        {str(key): _clean(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _coerce_rows(rows: Rows | None, *, name: str) -> pd.DataFrame | None:
    if rows is None:
        return None
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ResearchWorkbenchError(
            f"{name} rows must be a DataFrame or row sequence"
        )
    if not all(isinstance(row, Mapping) for row in rows):
        raise ResearchWorkbenchError(f"{name} rows must contain mappings")
    return pd.DataFrame([dict(row) for row in rows])


def canonical_rows_sha256(tables: Mapping[str, Rows]) -> str:
    """Hash named row tables with stable JSON field and table ordering."""

    canonical: dict[str, list[dict[str, object]]] = {}
    for name, rows in sorted(tables.items()):
        frame = _coerce_rows(rows, name=name)
        assert frame is not None
        canonical[str(name)] = _records(frame)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _publication_root(manifest_path: Path, *, name: str) -> Path:
    indices = [
        index
        for index, part in enumerate(manifest_path.parts)
        if part == "manifests"
    ]
    if not indices:
        raise ResearchWorkbenchError(f"{name} manifest path has no manifests root")
    return Path(*manifest_path.parts[: indices[-1]])


def _verify_content_addressed_path(
    path: Path,
    *,
    digest: str,
    name: str,
) -> None:
    match = _CANONICAL_ARTIFACT_RE.search(path.as_posix())
    hex_digest = digest.removeprefix("sha256:")
    if (
        match is None
        or match.group(1) != hex_digest[:2]
        or match.group(2) != hex_digest
    ):
        raise ResearchWorkbenchError(f"{name} filename/path digest mismatch")


def _load_publication_source(
    source: Mapping[str, object],
    *,
    name: str,
    logical_names: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, object]]:
    digest = source.get("artifact_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ResearchWorkbenchError(f"{name} artifact_sha256 is invalid")
    raw_path = source.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ResearchWorkbenchError(f"{name} artifact_path is required")
    manifest_path = Path(raw_path).resolve()
    if not manifest_path.is_file():
        raise ResearchWorkbenchError(f"{name} artifact_path does not exist")
    if _file_sha256(manifest_path) != digest:
        raise ResearchWorkbenchError(f"{name} artifact bytes sha256 mismatch")
    _verify_content_addressed_path(
        manifest_path,
        digest=digest,
        name=f"{name} manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchWorkbenchError(f"{name} manifest is not valid JSON") from exc
    if (
        manifest.get("holdout_reaction_accessed") is True
        or manifest.get("holdout_reaction_rows_read") not in (None, 0)
    ):
        raise ResearchWorkbenchError(f"{name} manifest accessed holdout reaction")
    raw_descriptors = manifest.get("objects")
    if isinstance(raw_descriptors, list):
        descriptors = raw_descriptors
    elif isinstance(raw_descriptors, Mapping):
        descriptors = [
            {"logical_name": logical_name, **descriptor}
            for logical_name, descriptor in raw_descriptors.items()
            if isinstance(descriptor, Mapping)
        ]
        if len(descriptors) != len(raw_descriptors):
            raise ResearchWorkbenchError(f"{name} manifest object descriptor invalid")
    else:
        raise ResearchWorkbenchError(f"{name} manifest objects are required")
    publication_root = _publication_root(manifest_path, name=name).resolve()
    tables: dict[str, pd.DataFrame] = {}
    object_audit: list[dict[str, object]] = []
    for logical_name in logical_names:
        matches = [
            descriptor
            for descriptor in descriptors
            if isinstance(descriptor, Mapping)
            and descriptor.get("logical_name") == logical_name
        ]
        if len(matches) != 1:
            raise ResearchWorkbenchError(
                f"{name} manifest must bind one {logical_name} object"
            )
        descriptor = matches[0]
        if descriptor.get("format") != "parquet":
            raise ResearchWorkbenchError(f"{name} {logical_name} must be parquet")
        object_digest = descriptor.get("object_sha256")
        if not isinstance(object_digest, str) or not _SHA256_RE.fullmatch(
            object_digest
        ):
            raise ResearchWorkbenchError(
                f"{name} {logical_name} object_sha256 is invalid"
            )
        object_relative = descriptor.get("object_path")
        if not isinstance(object_relative, str) or not object_relative:
            raise ResearchWorkbenchError(f"{name} {logical_name} object_path missing")
        object_path = (publication_root / object_relative).resolve()
        if not object_path.is_relative_to(publication_root):
            raise ResearchWorkbenchError(f"{name} {logical_name} object_path escapes root")
        if not object_path.is_file():
            raise ResearchWorkbenchError(f"{name} {logical_name} object is missing")
        _verify_content_addressed_path(
            object_path,
            digest=object_digest,
            name=f"{name} {logical_name} object",
        )
        byte_length = descriptor.get("byte_length")
        if type(byte_length) is not int or byte_length != object_path.stat().st_size:
            raise ResearchWorkbenchError(
                f"{name} {logical_name} object byte_length mismatch"
            )
        if _file_sha256(object_path) != object_digest:
            raise ResearchWorkbenchError(
                f"{name} {logical_name} object bytes sha256 mismatch"
            )
        frame = pd.read_parquet(object_path)
        row_count = descriptor.get("row_count")
        if type(row_count) is not int or row_count != len(frame):
            raise ResearchWorkbenchError(
                f"{name} {logical_name} object row_count mismatch"
            )
        column_count = descriptor.get("column_count")
        if column_count is not None and (
            type(column_count) is not int or column_count != len(frame.columns)
        ):
            raise ResearchWorkbenchError(
                f"{name} {logical_name} object column_count mismatch"
            )
        tables[logical_name] = frame
        object_audit.append(
            {
                "logical_name": logical_name,
                "object_sha256": object_digest,
                "row_count": row_count,
            }
        )
    verified_source = {
        "artifact_sha256": digest,
        "artifact_path": manifest_path.as_posix(),
        "loaded_objects": object_audit,
    }
    return tables, manifest, verified_source


def _load_tick_inventory_source(
    source: Mapping[str, object],
    *,
    name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, object]]:
    digest = source.get("artifact_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ResearchWorkbenchError(f"{name} artifact_sha256 is invalid")
    raw_path = source.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ResearchWorkbenchError(f"{name} artifact_path is required")
    manifest_path = Path(raw_path).resolve()
    if not manifest_path.is_file():
        raise ResearchWorkbenchError(f"{name} artifact_path does not exist")
    if _file_sha256(manifest_path) != digest:
        raise ResearchWorkbenchError(f"{name} artifact bytes sha256 mismatch")
    _verify_content_addressed_path(
        manifest_path,
        digest=digest,
        name=f"{name} manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchWorkbenchError(f"{name} manifest is not valid JSON") from exc
    descriptor = manifest.get("rows")
    if (
        manifest.get("schema_version") != "TickDataInventoryManifestV1"
        or manifest.get("holdout_access") != "CLOSED_UNREAD"
        or not isinstance(descriptor, Mapping)
    ):
        raise ResearchWorkbenchError(f"{name} manifest boundary is invalid")
    object_digest = descriptor.get("sha256")
    object_relative = descriptor.get("path")
    if (
        not isinstance(object_digest, str)
        or not _SHA256_RE.fullmatch(object_digest)
        or not isinstance(object_relative, str)
        or not object_relative
    ):
        raise ResearchWorkbenchError(f"{name} rows descriptor is invalid")
    publication_root = _publication_root(manifest_path, name=name).resolve()
    object_path = (publication_root / object_relative).resolve()
    if not object_path.is_relative_to(publication_root):
        raise ResearchWorkbenchError(f"{name} rows object_path escapes root")
    if not object_path.is_file():
        raise ResearchWorkbenchError(f"{name} rows object is missing")
    _verify_content_addressed_path(
        object_path,
        digest=object_digest,
        name=f"{name} rows object",
    )
    if _file_sha256(object_path) != object_digest:
        raise ResearchWorkbenchError(f"{name} rows object bytes sha256 mismatch")
    try:
        payload = json.loads(object_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchWorkbenchError(f"{name} rows object is not valid JSON") from exc
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != "TickDataInventoryV1"
        or payload.get("holdout_access") != "CLOSED_UNREAD"
        or not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ResearchWorkbenchError(f"{name} rows object boundary is invalid")
    row_count = descriptor.get("row_count")
    if type(row_count) is not int or row_count != len(rows):
        raise ResearchWorkbenchError(f"{name} rows object row_count mismatch")
    frame = pd.DataFrame([dict(row) for row in rows])
    verified_source = {
        "artifact_sha256": digest,
        "artifact_path": manifest_path.as_posix(),
        "loaded_objects": [
            {
                "logical_name": "tick_inventory",
                "object_sha256": object_digest,
                "row_count": row_count,
            }
        ],
    }
    return {"tick_inventory": frame}, manifest, verified_source


def _load_pmxt_method_audit_source(
    source: Mapping[str, object],
    *,
    name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, object]]:
    digest = source.get("artifact_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ResearchWorkbenchError(f"{name} artifact_sha256 is invalid")
    raw_path = source.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ResearchWorkbenchError(f"{name} artifact_path is required")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ResearchWorkbenchError(f"{name} artifact_path does not exist")
    if _file_sha256(path) != digest:
        raise ResearchWorkbenchError(f"{name} artifact bytes sha256 mismatch")
    _verify_content_addressed_path(path, digest=digest, name=name)
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchWorkbenchError(f"{name} artifact is not valid JSON") from exc
    crosswalk = audit.get("crosswalk")
    remote = audit.get("remote_verification")
    gates = audit.get("method_gates")
    if (
        audit.get("audit_kind") != "pmxt_l2_method_audit_v1"
        or audit.get("data_access") != "local_fixture_and_metadata_only"
        or audit.get("raw_parquet_scanned") is not False
        or audit.get("upstream_accessed") is not False
        or not isinstance(crosswalk, Mapping)
        or not isinstance(remote, Mapping)
        or not isinstance(gates, Mapping)
        or not gates
        or any(
            not isinstance(gate, Mapping) or gate.get("status") != "PASS"
            for gate in gates.values()
        )
    ):
        raise ResearchWorkbenchError(f"{name} method-only boundary is invalid")
    attested_count = crosswalk.get("attested_remote_object_count")
    verified_count = remote.get("verified_object_count")
    if (
        type(attested_count) is not int
        or type(verified_count) is not int
        or verified_count != attested_count
        or crosswalk.get("unattested_remote_object_count") != 0
    ):
        raise ResearchWorkbenchError(f"{name} remote attestation counts mismatch")
    row = {
        "factor_id": "PMXT.L2.METHOD_FIXTURE",
        "claim_boundary": _L2_CLAIM_BOUNDARY,
        "audit_kind": audit["audit_kind"],
        "data_access": audit["data_access"],
        "raw_parquet_scanned": audit["raw_parquet_scanned"],
        "upstream_accessed": audit["upstream_accessed"],
        "verified_object_count": verified_count,
        "attested_remote_object_count": attested_count,
        "fixture_event_count": audit.get("fixture_event_count"),
        "tick_count": audit.get("tick_count"),
        "epoch_count": audit.get("epoch_count"),
        "method_gate_count": len(gates),
        "method_gate_status": "PASS",
        "empirical_scan_claim": "NOT_MADE",
    }
    frame = pd.DataFrame([row])
    verified_source = {
        "artifact_sha256": digest,
        "artifact_path": path.as_posix(),
        "loaded_objects": [
            {
                "logical_name": name,
                "object_sha256": digest,
                "row_count": 1,
            }
        ],
    }
    return {name: frame}, audit, verified_source


def load_research_workbench_evidence(
    artifact_sources: Mapping[str, Mapping[str, object]],
) -> ResearchWorkbenchEvidence:
    """Load all available workbench tables from verified publication objects."""

    requested = dict(_REQUIRED_PUBLICATION_TABLES)
    requested.update(
        {
            name: logical_names
            for name, logical_names in _OPTIONAL_PUBLICATION_TABLES.items()
            if name in artifact_sources
        }
    )
    loaded_by_source: dict[str, dict[str, pd.DataFrame]] = {}
    manifests: dict[str, Mapping[str, object]] = {}
    sources: dict[str, Mapping[str, object]] = {}
    audit_rows: list[dict[str, object]] = []
    for name, logical_names in requested.items():
        source = artifact_sources.get(name)
        if not isinstance(source, Mapping):
            raise ResearchWorkbenchError(
                f"{name} requires a content-addressed artifact source"
            )
        artifact_path = source.get("artifact_path")
        if name == "tick_inventory":
            tables, manifest, verified_source = _load_tick_inventory_source(
                source,
                name=name,
            )
        elif (
            name == "pmxt_l2_method_audit"
            and isinstance(artifact_path, str)
            and not artifact_path.endswith(".manifest.json")
        ):
            tables, manifest, verified_source = _load_pmxt_method_audit_source(
                source,
                name=name,
            )
        else:
            tables, manifest, verified_source = _load_publication_source(
                source,
                name=name,
                logical_names=logical_names,
            )
        loaded_by_source[name] = tables
        manifests[name] = manifest
        sources[name] = verified_source
        audit_rows.append(
            {
                "source": name,
                "artifact_sha256": verified_source["artifact_sha256"],
                "artifact_path": verified_source["artifact_path"],
                "logical_names": ",".join(logical_names),
                "verified_row_count": sum(len(frame) for frame in tables.values()),
                "status": "VERIFIED_PUBLICATION_OBJECTS",
            }
        )
    venue_diagnostics = loaded_by_source["same_episode_paired"][
        "venue_diagnostics"
    ]
    same_episode = venue_diagnostics.loc[
        venue_diagnostics["comparison_mode"].eq("SAME_EPISODE_PAIRED")
    ].reset_index(drop=True)
    if same_episode.empty:
        raise ResearchWorkbenchError("study same-episode paired rows are required")
    tables = {
        "summary": loaded_by_source["historical_distribution"]["summary"],
        "histogram": loaded_by_source["historical_distribution"]["histogram"],
        "reference_summary": loaded_by_source["reference_completion"][
            "reference_summary"
        ],
        "reference_exclusions": loaded_by_source["reference_completion"][
            "reference_exclusions"
        ],
        "same_episode_paired": same_episode,
        "venue_consistency": loaded_by_source["venue_consistency"][
            "venue_consistency"
        ],
    }
    for source_name, logical_names in _OPTIONAL_PUBLICATION_TABLES.items():
        for logical_name in logical_names:
            tables[logical_name] = (
                loaded_by_source[source_name][logical_name]
                if source_name in loaded_by_source
                else pd.DataFrame()
            )
    return ResearchWorkbenchEvidence(
        tables=tables,
        sources=sources,
        manifests=manifests,
        source_audit=pd.DataFrame(audit_rows),
    )


def _not_available_zone(
    evidence: ResearchWorkbenchEvidence,
    *,
    zone: str,
    status: str = "NOT_AVAILABLE",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone": zone,
                "status": status,
                "verified_manifest_count": len(evidence.sources),
                "evidence_basis": "VERIFIED_PUBLICATION_MANIFEST_AUDIT",
            }
        ]
    )


def research_workbench_zone_frame(
    evidence: ResearchWorkbenchEvidence,
    zone: str,
) -> pd.DataFrame:
    """Project one Notebook zone from already verified publication evidence."""

    direct = {
        "tick_inventory": "tick_inventory",
        "historical_distribution": "summary",
        "reference_completion": "reference_summary",
        "pmxt_l2_method_audit": "pmxt_l2_method_audit",
        "shadow_metrics": "shadow_metrics",
    }
    if zone in direct:
        frame = evidence.tables[direct[zone]].copy()
        if not frame.empty:
            return frame
        status = "DATA_GRADE_BLOCKED" if zone == "pmxt_l2_method_audit" else "NOT_AVAILABLE"
        return _not_available_zone(evidence, zone=zone, status=status)
    if zone == "calibration_bias":
        frame = pd.concat(
            [
                evidence.tables[logical_name].assign(
                    evidence_table=logical_name
                )
                for logical_name in ("metrics", "reliability", "bias_proxies")
            ],
            ignore_index=True,
            sort=False,
        )
        return (
            frame
            if not frame.empty
            else _not_available_zone(evidence, zone=zone)
        )
    if zone == "poly_kalshi_paired":
        return pd.concat(
            [
                evidence.tables["same_episode_paired"].assign(
                    evidence_table="same_episode_paired"
                ),
                evidence.tables["venue_consistency"].assign(
                    evidence_table="venue_consistency"
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    if zone == "shortlist_holdout":
        return pd.DataFrame(
            [
                {
                    "source": name,
                    "holdout_reaction_accessed": manifest.get(
                        "holdout_reaction_accessed", False
                    ),
                    "status": "SEALED_UNREAD",
                }
                for name, manifest in evidence.manifests.items()
            ]
        )
    if zone == "audit":
        return evidence.source_audit.copy()
    raise ResearchWorkbenchError(f"unknown workbench zone: {zone}")


def _require_factor_id(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty:
        return
    if "factor_id" not in frame.columns:
        raise ResearchWorkbenchError(f"{name} rows require factor_id")
    values = frame["factor_id"]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ResearchWorkbenchError(f"{name} rows require non-empty factor_id")


def _reject_alpha_claims(
    rows: pd.DataFrame,
    *,
    name: str,
    exclude_fields: frozenset[str] = frozenset(),
) -> None:
    for column in rows.columns:
        if str(column) in exclude_fields:
            continue
        if re.search(r"\balpha\b", str(column), flags=re.IGNORECASE):
            raise ResearchWorkbenchError(f"{name} contains prohibited alpha claim")
        for value in rows[column].dropna().tolist():
            if re.search(r"\balpha\b", str(value), flags=re.IGNORECASE):
                raise ResearchWorkbenchError(
                    f"{name} contains prohibited alpha claim"
                )


def _assert_caller_rows_match_verified(
    caller: pd.DataFrame,
    verified: pd.DataFrame,
    *,
    name: str,
) -> None:
    caller_digest = canonical_rows_sha256({name: caller})
    verified_digest = canonical_rows_sha256({name: verified})
    if caller_digest != verified_digest:
        raise ResearchWorkbenchError(
            f"{name} caller rows differ from verified publication object"
        )


def _payload(
    *,
    study_tables: Mapping[str, pd.DataFrame],
    reference_tables: Mapping[str, pd.DataFrame],
    matched_tables: Mapping[str, pd.DataFrame],
    notebook_path: Path,
    audit: Mapping[str, object],
    findings: Sequence[str],
    artifact_sources: Mapping[str, Mapping[str, object]],
    calibration_tables: Mapping[str, Rows] | None,
    tick_inventory_rows: Rows | None,
    l2_audit_rows: Rows | None,
    shadow_rows: Rows | None,
) -> dict[str, object]:
    caller_summary = study_tables.get("summary")
    if caller_summary is None:
        raise ResearchWorkbenchError("historical summary is required")
    caller_histogram = study_tables.get("histogram", pd.DataFrame())
    caller_venue_diagnostics = study_tables.get("venue_diagnostics")
    if (
        caller_venue_diagnostics is None
        or "comparison_mode" not in caller_venue_diagnostics
    ):
        raise ResearchWorkbenchError("study same-episode paired rows are required")
    caller_same_episode = caller_venue_diagnostics.loc[
        caller_venue_diagnostics["comparison_mode"].eq("SAME_EPISODE_PAIRED")
    ].reset_index(drop=True)
    if caller_same_episode.empty:
        raise ResearchWorkbenchError("study same-episode paired rows are required")
    caller_venue_consistency = matched_tables.get("venue_consistency")
    if caller_venue_consistency is None:
        raise ResearchWorkbenchError("matched venue_consistency is required")
    caller_reference_summary = reference_tables.get("reference_summary")
    if caller_reference_summary is None:
        raise ResearchWorkbenchError("reference_summary is required")
    caller_reference_exclusions = reference_tables.get(
        "reference_exclusions", pd.DataFrame()
    )
    caller_optional = {
        "tick_inventory": _coerce_rows(
            tick_inventory_rows, name="tick_inventory"
        ),
        "pmxt_l2_method_audit": _coerce_rows(
            l2_audit_rows, name="pmxt_l2_method_audit"
        ),
        "shadow_metrics": _coerce_rows(shadow_rows, name="shadow_metrics"),
    }
    caller_calibration: dict[str, pd.DataFrame] = {}
    if calibration_tables is not None:
        required_calibration = {"metrics", "reliability", "bias_proxies"}
        if set(calibration_tables) != required_calibration:
            raise ResearchWorkbenchError(
                "calibration_tables must contain metrics, reliability, bias_proxies"
            )
        for name, rows in calibration_tables.items():
            frame = _coerce_rows(rows, name=f"calibration {name}")
            assert frame is not None
            caller_calibration[name] = frame

    evidence = load_research_workbench_evidence(artifact_sources)
    summary = evidence.tables["summary"]
    histogram = evidence.tables["histogram"]
    same_episode = evidence.tables["same_episode_paired"]
    venue_consistency = evidence.tables["venue_consistency"]
    reference_summary = evidence.tables["reference_summary"]
    reference_exclusions = evidence.tables["reference_exclusions"]
    external_inputs = {
        name: evidence.tables[name] for name in caller_optional
    }
    calibration = {
        name: evidence.tables[name]
        for name in ("metrics", "reliability", "bias_proxies")
    }

    for caller, verified, name in (
        (caller_summary, summary, "summary"),
        (caller_histogram, histogram, "histogram"),
        (caller_same_episode, same_episode, "same_episode_paired"),
        (
            caller_venue_consistency,
            venue_consistency,
            "venue_consistency",
        ),
        (
            caller_reference_summary,
            reference_summary,
            "reference_summary",
        ),
        (
            caller_reference_exclusions,
            reference_exclusions,
            "reference_exclusions",
        ),
    ):
        _assert_caller_rows_match_verified(caller, verified, name=name)
    for name, caller in caller_optional.items():
        if caller is not None:
            _assert_caller_rows_match_verified(
                caller,
                external_inputs[name],
                name=name,
            )
    for name, caller in caller_calibration.items():
        _assert_caller_rows_match_verified(
            caller,
            calibration[name],
            name=name,
        )

    if summary.empty:
        raise ResearchWorkbenchError("historical summary is required")
    required_summary = {
        "factor_id",
        "venue",
        "market_family",
        "horizon_seconds",
        "mean",
        "median",
    }
    missing = sorted(required_summary.difference(summary.columns))
    if missing:
        raise ResearchWorkbenchError(
            "historical summary missing columns: " + ", ".join(missing)
        )
    if same_episode.empty:
        raise ResearchWorkbenchError("study same-episode paired rows are required")
    if venue_consistency.empty:
        raise ResearchWorkbenchError("matched venue_consistency is required")
    if reference_summary.empty:
        raise ResearchWorkbenchError("reference_summary is required")

    for name, frame in (
        ("historical summary", summary),
        ("historical histogram", histogram),
        ("same_episode_paired", same_episode),
        ("venue_consistency", venue_consistency),
        ("reference_summary", reference_summary),
    ):
        _require_factor_id(frame, name=name)

    for name in ("pmxt_l2_method_audit", "shadow_metrics"):
        frame = external_inputs[name]
        if frame.empty:
            continue
        _require_factor_id(frame, name=name)

    findings_frame = pd.DataFrame({"finding": [str(value) for value in findings]})
    _reject_alpha_claims(findings_frame, name="findings")
    l2 = external_inputs["pmxt_l2_method_audit"]
    if not l2.empty:
        if (
            "claim_boundary" not in l2.columns
            or not l2["claim_boundary"].eq(_L2_CLAIM_BOUNDARY).all()
        ):
            raise ResearchWorkbenchError(
                "pmxt_l2_method_audit claim_boundary must equal "
                + _L2_CLAIM_BOUNDARY
            )
        _reject_alpha_claims(
            l2,
            name="pmxt_l2_method_audit",
            exclude_fields=frozenset({"claim_boundary"}),
        )

    factor_exclusions = (
        reference_exclusions.loc[reference_exclusions["factor_id"].notna()].copy()
        if "factor_id" in reference_exclusions.columns
        else pd.DataFrame()
    )
    global_exclusions = (
        reference_exclusions.loc[reference_exclusions["factor_id"].isna()].copy()
        if "factor_id" in reference_exclusions.columns
        else reference_exclusions.copy()
    )
    return {
        "summary": _records(summary),
        "histogram": _records(histogram),
        "reference_summary": _records(reference_summary),
        "reference_factor_exclusions": _records(factor_exclusions),
        "reference_global_exclusions": _records(global_exclusions),
        "same_episode_paired": _records(same_episode),
        "venue_consistency": _records(venue_consistency),
        "calibration_metrics": _records(calibration["metrics"]),
        "calibration_reliability": _records(calibration["reliability"]),
        "calibration_bias_proxies": _records(calibration["bias_proxies"]),
        "tick_inventory_rows": _records(external_inputs["tick_inventory"]),
        "l2_audit_rows": _records(external_inputs["pmxt_l2_method_audit"]),
        "shadow_rows": _records(external_inputs["shadow_metrics"]),
        "artifact_sources": evidence.sources,
        "notebook_path": notebook_path.as_posix(),
        "audit": {str(key): _clean(value) for key, value in audit.items()},
        "findings": [str(value) for value in findings],
    }


_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><link rel="icon" href="data:,">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL X-13 · offline research workbench</title>
<style>
:root{--ink:#18221d;--muted:#68746c;--paper:#f3f0e8;--panel:#fffdf8;--line:#d5d0c4;--green:#116149;--red:#a43b32;--amber:#9b6814}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:30px 36px 24px;border-bottom:1px solid var(--line);background:#e8e3d8}h1{font:700 30px/1.1 ui-serif,Georgia,serif;margin:0 0 10px}.kicker{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--green);font-weight:800}.boundary{max-width:1050px;color:var(--muted)}.status{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.pill{border:1px solid var(--line);background:var(--panel);padding:5px 9px;font-size:12px}.shell{padding:20px 30px 50px}.controls{position:sticky;top:0;z-index:4;background:rgba(243,240,232,.96);padding:12px 0;border-bottom:1px solid var(--line);display:grid;grid-template-columns:minmax(260px,2fr) repeat(3,minmax(120px,1fr));gap:10px}label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}select{width:100%;margin-top:4px;padding:8px;border:1px solid var(--line);background:var(--panel)}.tabs{display:flex;overflow:auto;margin:20px 0;border-bottom:1px solid var(--line)}.tab{border:0;border-bottom:3px solid transparent;background:transparent;padding:10px 14px;font-weight:700;color:var(--muted);white-space:nowrap;cursor:pointer}.tab.active{color:var(--green);border-color:var(--green)}.pane{display:none}.pane.active{display:block}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{grid-column:span 3;background:var(--panel);border:1px solid var(--line);padding:15px;min-height:100px}.wide{grid-column:span 12}.half{grid-column:span 6}.metric{font:700 25px/1 ui-serif,Georgia,serif;margin-top:8px}.small{font-size:12px;color:var(--muted)}h2{font:700 21px/1.2 ui-serif,Georgia,serif;margin:0 0 12px}h3{margin:18px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.08em}.table-wrap{overflow:auto;max-height:650px;border:1px solid var(--line);background:var(--panel)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:7px 9px;border-bottom:1px solid #e7e2d7;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#ece8de;color:#4e5a53;font-size:11px;text-transform:uppercase}td:first-child,th:first-child{text-align:left}.empty{padding:24px;border:1px solid var(--line);background:var(--panel);color:var(--muted)}.blocked{border-left:4px solid var(--red);padding:12px 14px;background:#fff1f0}.callout{border-left:4px solid var(--amber);padding:10px 14px;background:#fff7e7;margin-bottom:14px}a{color:var(--green)}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.viz-grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,2fr);gap:14px;margin:18px 0}.viz{background:var(--panel);border:1px solid var(--line);padding:14px;overflow:auto}.viz svg{display:block;width:100%;min-width:620px;height:auto}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend-key::before{content:"";display:inline-block;width:18px;border-top:3px solid var(--key);margin-right:5px;vertical-align:middle}.legend-key.median::before{border-top-style:dashed}.heatmap td{min-width:70px;text-align:center}.heatmap .selected-factor{font-weight:800}.axis-label{font-size:11px;fill:#68746c}.point-label{font-size:9px;fill:#4e5a53}@media(max-width:900px){.controls{grid-template-columns:1fr 1fr}.card,.half{grid-column:span 12}.viz-grid{grid-template-columns:1fr}.shell{padding:16px}}
</style></head><body>
<header><div class="kicker">X-13 · PRELIMINARY SOURCE-TIME ONLY</div><h1>NFL offline research workbench</h1><div class="boundary">只显示经 publication manifest、object descriptor 与实际对象字节 SHA-256 验证的离线研究输入；final holdout 保持 SEALED_UNREAD。</div><p><a id="notebookTop" href="#">打开 Master Notebook</a></p><div class="status" id="status"></div></header>
<main class="shell"><div class="controls"><label>Factor<select id="factor"></select></label><label>Market family<select id="family"></select></label><label>Venue<select id="venue"><option value="all">All venues</option><option value="polymarket">Polymarket</option><option value="kalshi">Kalshi</option></select></label><label>Horizon<select id="horizon"></select></label></div>
<nav class="tabs"><button class="tab active" data-pane="ticks">Tick Inventory</button><button class="tab" data-pane="historical">Historical Distribution</button><button class="tab" data-pane="reference">Reference Completion</button><button class="tab" data-pane="calibration">Calibration/Bias</button><button class="tab" data-pane="paired">Poly×Kalshi paired</button><button class="tab" data-pane="l2">PMXT L2 method audit</button><button class="tab" data-pane="shortlist">Shortlist/Holdout</button><button class="tab" data-pane="shadow">Shadow metrics</button><button class="tab" data-pane="audit">Audit</button></nav>
<section id="ticks" class="pane active"></section><section id="historical" class="pane"></section><section id="reference" class="pane"></section><section id="calibration" class="pane"></section><section id="paired" class="pane"></section><section id="l2" class="pane"></section><section id="shortlist" class="pane"></section><section id="shadow" class="pane"></section><section id="audit" class="pane"></section></main>
<script id="dataset" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById("dataset").textContent),$=s=>document.querySelector(s),esc=s=>String(s??"—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])),uniq=a=>[...new Set(a)].sort((a,b)=>String(a).localeCompare(String(b))),S={factor:null,family:null,venue:"all",horizon:null};
function factorRows(rows){return rows.filter(r=>r.factor_id===S.factor)}
function zoneRows(rows){return factorRows(rows).filter(r=>(r.market_family==null||r.market_family===S.family)&&(S.venue==="all"||r.venue==null||r.venue===S.venue)&&(r.horizon_seconds==null||Number(r.horizon_seconds)===Number(S.horizon)))}
function pairedRows(rows){return factorRows(rows).filter(r=>(r.market_family==null||r.market_family===S.family)&&(r.horizon_seconds==null||Number(r.horizon_seconds)===Number(S.horizon)))}
function inventoryRows(rows){return rows.filter(r=>S.venue==="all"||r.venue==null||r.venue===S.venue)}
function calibrationRows(rows){return rows.filter(r=>S.venue==="all"||r.scope!=="venue"||r.scope_value===S.venue)}
function table(rows){if(!rows.length)return '<div class="empty">NOT_AVAILABLE — 当前范围没有经验证的行。</div>';const keys=uniq(rows.flatMap(Object.keys)).slice(0,14);return `<div class="table-wrap"><table><thead><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${keys.map(k=>`<td>${esc(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`}
function source(name){const s=D.artifact_sources[name],objects=s?.loaded_objects||[];return s?`<p class="small">manifest <code>${esc(s.artifact_sha256)}</code><br>verified objects ${objects.map(x=>`<code>${esc(x.logical_name)}:${esc(x.object_sha256)}</code>`).join(" · ")}<br><code>${esc(s.artifact_path)}</code></p>`:'<p class="small">NOT_AVAILABLE — 未提供 content-addressed artifact。</p>'}
function panel(title,name,rows,{blocked=false}={}){const visible=zoneRows(rows);const state=visible.length?'':`<div class="${blocked?'blocked':'empty'}">${blocked?'DATA_GRADE_BLOCKED':'NOT_AVAILABLE'} — ${blocked?'没有合规 PMXT L2 method-only 输入。':'没有可显示的经验证输入。'}</div>`;return `<h2>${title}</h2>${source(name)}${state}${visible.length?table(visible):''}`}
function renderTimePath(){const rows=factorRows(D.summary).filter(r=>r.market_family===S.family&&["polymarket","kalshi"].includes(r.venue)).sort((a,b)=>Number(a.horizon_seconds)-Number(b.horizon_seconds));if(!rows.length)return '<div class="empty">NOT_AVAILABLE — 无 verified time-path rows。</div>';const horizons=uniq(rows.map(r=>Number(r.horizon_seconds))).sort((a,b)=>a-b),values=rows.flatMap(r=>[Number(r.mean),Number(r.median)]).filter(Number.isFinite),lo=Math.min(0,...values),hi=Math.max(0,...values),span=hi-lo||1,w=760,h=300,pad={l:56,r:18,t:25,b:42},sx=x=>pad.l+(horizons.indexOf(x)/Math.max(1,horizons.length-1))*(w-pad.l-pad.r),sy=y=>pad.t+(hi-y)/span*(h-pad.t-pad.b),colors={polymarket:"#116149",kalshi:"#9b6814"};const axes=`<line x1="${pad.l}" y1="${sy(0)}" x2="${w-pad.r}" y2="${sy(0)}" stroke="#9a9488"/><line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h-pad.b}" stroke="#9a9488"/>${horizons.map(x=>`<text class="axis-label" x="${sx(x)}" y="${h-18}" text-anchor="middle">${x}s</text>`).join("")}`;const series=["polymarket","kalshi"].flatMap(venue=>["mean","median"].map(stat=>{const points=rows.filter(r=>r.venue===venue&&Number.isFinite(Number(r[stat]))),path=points.map(r=>`${sx(Number(r.horizon_seconds))},${sy(Number(r[stat]))}`).join(" "),dash=stat==="median"?' stroke-dasharray="6 4"':"";return `<polyline points="${path}" fill="none" stroke="${colors[venue]}" stroke-width="2.5"${dash}/>${points.map(r=>`<circle cx="${sx(Number(r.horizon_seconds))}" cy="${sy(Number(r[stat]))}" r="3.5" fill="${colors[venue]}"><title>${venue} ${stat}=${r[stat]} · n=${r.observed_n} · ${r.horizon_seconds}s</title></circle>`).join("")}`})).join("");const nRows=rows.map(r=>({venue:r.venue,horizon_seconds:r.horizon_seconds,mean:r.mean,median:r.median,observed_n:r.observed_n}));return `<div class="viz"><h3>Current factor · Poly/Kalshi horizon time-path</h3><div class="legend"><span class="legend-key" style="--key:#116149">Polymarket mean</span><span class="legend-key median" style="--key:#116149">Polymarket median</span><span class="legend-key" style="--key:#9b6814">Kalshi mean</span><span class="legend-key median" style="--key:#9b6814">Kalshi median</span></div><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Verified mean and median reaction by horizon">${axes}${series}</svg><h3>Point values and n</h3>${table(nRows)}</div>`}
function heatColor(value,maxAbs){const v=Number(value);if(!Number.isFinite(v))return "transparent";const alpha=.12+.68*Math.min(1,Math.abs(v)/(maxAbs||1));return v>0?`rgba(17,97,73,${alpha})`:v<0?`rgba(164,59,50,${alpha})`:"rgba(104,116,108,.12)"}
function renderSignedHeatmap(){const base=D.summary.filter(r=>r.market_family===S.family&&(S.venue==="all"||r.venue===S.venue)),venues=S.venue==="all"?["polymarket","kalshi"]:[S.venue];return venues.map(venue=>{const rows=base.filter(r=>r.venue===venue),horizons=uniq(rows.map(r=>Number(r.horizon_seconds))).sort((a,b)=>a-b),factors=uniq(rows.map(r=>r.factor_id)),maxAbs=Math.max(0,...rows.map(r=>Math.abs(Number(r.mean)||0)));if(!rows.length)return "";return `<div class="viz"><h3>${esc(venue)} · signed factor × horizon heatmap</h3><div class="table-wrap"><table class="heatmap"><thead><tr><th>factor</th>${horizons.map(x=>`<th>${x}s</th>`).join("")}</tr></thead><tbody>${factors.map(f=>`<tr class="${f===S.factor?'selected-factor':''}"><td>${esc(f)}</td>${horizons.map(x=>{const row=rows.find(r=>r.factor_id===f&&Number(r.horizon_seconds)===x),v=row?.mean;return `<td style="background:${heatColor(v,maxAbs)}" title="mean=${esc(v)} · n=${esc(row?.observed_n)}">${Number.isFinite(Number(v))?(Number(v)*100).toFixed(2)+"pp":"—"}</td>`}).join("")}</tr>`).join("")}</tbody></table></div><p class="small">绿色=正向，红色=负向；单元值为 verified mean（概率点）。加粗行为当前 factor。</p></div>`}).join("")}
function renderDistribution(){const rows=zoneRows(D.histogram);if(!rows.length)return '<div class="empty">NOT_AVAILABLE — 当前选择无 verified histogram bins。</div>';return uniq(rows.map(r=>r.venue)).map(venue=>{const bins=rows.filter(r=>r.venue===venue),maxCount=Math.max(1,...bins.map(r=>Number(r.count)||0)),w=760,h=240,p={l:48,r:16,t:22,b:46},bar=(w-p.l-p.r)/Math.max(1,bins.length);return `<div class="viz"><h3>${esc(venue)} · verified histogram</h3><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Verified empirical histogram">${bins.map((r,i)=>{const bh=(Number(r.count)||0)/maxCount*(h-p.t-p.b),x=p.l+i*bar+1,y=h-p.b-bh;return `<rect x="${x}" y="${y}" width="${Math.max(1,bar-2)}" height="${bh}" fill="#116149"><title>${r.bin_left_pp}–${r.bin_right_pp}pp · count=${r.count}</title></rect>`}).join("")}<line x1="${p.l}" y1="${h-p.b}" x2="${w-p.r}" y2="${h-p.b}" stroke="#9a9488"/><text class="axis-label" x="${w/2}" y="${h-14}" text-anchor="middle">published Δp bins · hover for count</text></svg>${table(bins)}</div>`}).join("")}
function renderTicks(){const rows=inventoryRows(D.tick_inventory_rows);$("#ticks").innerHTML=`<h2>Tick Inventory</h2>${source("tick_inventory")}${rows.length?table(inventoryRows(D.tick_inventory_rows)):'<div class="empty">NOT_AVAILABLE — 当前 venue 无 inventory row。</div>'}`}
function renderHistorical(){const rows=zoneRows(D.summary),hist=zoneRows(D.histogram),best=rows[0]||{};$("#historical").innerHTML=`<h2>Historical Distribution</h2>${source("historical_distribution")}<div class="grid"><div class="card"><div class="small">Mean Δp</div><div class="metric">${esc(best.mean)}</div></div><div class="card"><div class="small">Median Δp</div><div class="metric">${esc(best.median)}</div></div><div class="card"><div class="small">Observed n</div><div class="metric">${esc(best.observed_n)}</div></div><div class="card"><div class="small">Games</div><div class="metric">${esc(best.game_count)}</div></div></div><div class="viz-grid">${renderTimePath()}${renderDistribution()}</div>${renderSignedHeatmap()}<h3>Selected summary rows</h3>${table(rows)}<h3>Selected histogram bins</h3>${table(hist)}`}
function renderReference(){const globalStatus=D.reference_global_exclusions;$("#reference").innerHTML=`<h2>Reference Completion</h2>${source("reference_completion")}<div class="callout">Retrospective diagnostic；不是实时 PIT 或执行证据。</div>${table(zoneRows(D.reference_summary))}<h3>Factor-specific reference exclusions</h3>${table(zoneRows(D.reference_factor_exclusions))}<h3>Global reference exclusions by support_status</h3>${table(globalStatus)}`}
function renderCalibration(){$("#calibration").innerHTML=`<h2>Calibration/Bias</h2>${source("calibration_bias")}<div class="callout">Retrospective development-only；按 scope/scope_value 展示，不受 factor 过滤。</div><h3>Metrics</h3>${table(calibrationRows(D.calibration_metrics))}<h3>Reliability</h3>${table(calibrationRows(D.calibration_reliability))}<h3>Bias proxies</h3>${table(D.calibration_bias_proxies)}`}
function renderPaired(){$("#paired").innerHTML=`<h2>Poly×Kalshi paired</h2><div class="callout">仅展示 SAME_EPISODE_PAIRED 与跨 venue consistency；单 venue matched controls 不属于 paired。</div>${source("same_episode_paired")}<h3>Same-episode paired</h3>${table(pairedRows(D.same_episode_paired))}${source("venue_consistency")}<h3>Venue consistency</h3>${table(pairedRows(D.venue_consistency))}`}
function renderL2(){$("#l2").innerHTML=panel("PMXT L2 method audit","pmxt_l2_method_audit",D.l2_audit_rows,{blocked:true})}
function renderShortlist(){const rows=Object.entries(D.audit).filter(([k])=>/shortlist|holdout/i.test(k)).map(([metric,value])=>({metric,value}));$("#shortlist").innerHTML=`<h2>Shortlist/Holdout</h2><div class="callout">只显示 SEALED_UNREAD 元数据；不读取 holdout rows。</div>${table(rows)}`}
function renderShadow(){$("#shadow").innerHTML=panel("Shadow metrics","shadow_metrics",D.shadow_rows)}
function renderAudit(){const sourceRows=Object.entries(D.artifact_sources).map(([zone,value])=>({zone,...value}));const findings=D.findings.length?`<div class="callout"><strong>Findings</strong><ol>${D.findings.map(x=>`<li>${esc(x)}</li>`).join("")}</ol></div>`:"";$("#audit").innerHTML=`<h2>Global Audit</h2>${findings}<div class="grid"><div class="half"><h3>Governance</h3>${table(Object.entries(D.audit).map(([metric,value])=>({metric,value})))}</div><div class="half"><h3>Artifact sources</h3>${table(sourceRows)}<p><a href="${esc(D.notebook_path)}">打开 Master Notebook</a></p></div></div>`}
function render(){renderTicks();renderHistorical();renderReference();renderCalibration();renderPaired();renderL2();renderShortlist();renderShadow();renderAudit()}
function activatePane(id){const button=document.querySelector(`.tab[data-pane="${id}"]`);if(!button)return;document.querySelectorAll(".tab,.pane").forEach(x=>x.classList.remove("active"));button.classList.add("active");$("#"+id).classList.add("active")}
function init(){const factors=uniq(D.summary.map(r=>r.factor_id)),families=uniq(D.summary.map(r=>r.market_family)),horizons=uniq(D.summary.map(r=>Number(r.horizon_seconds))).sort((a,b)=>a-b);S.factor=factors.includes("NFL.EVENT.EXPLOSIVE_PLAY")?"NFL.EVENT.EXPLOSIVE_PLAY":factors[0];S.family=families.includes("moneyline")?"moneyline":families[0];S.horizon=horizons.includes(60)?60:horizons[0];$("#factor").innerHTML=factors.map(x=>`<option ${x===S.factor?'selected':''}>${esc(x)}</option>`).join("");$("#family").innerHTML=families.map(x=>`<option ${x===S.family?'selected':''}>${esc(x)}</option>`).join("");$("#horizon").innerHTML=horizons.map(x=>`<option value="${x}" ${x===S.horizon?'selected':''}>${x}s</option>`).join("");$("#notebookTop").href=D.notebook_path;$("#status").innerHTML=Object.entries(D.audit).map(([k,v])=>`<span class="pill">${esc(k)}: <strong>${esc(v)}</strong></span>`).join("");$("#factor").onchange=e=>{S.factor=e.target.value;render()};$("#family").onchange=e=>{S.family=e.target.value;render()};$("#venue").onchange=e=>{S.venue=e.target.value;render()};$("#horizon").onchange=e=>{S.horizon=Number(e.target.value);render()};document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>activatePane(b.dataset.pane));render();activatePane(location.hash.slice(1))}init();
</script></body></html>"""


def render_research_workbench(
    *,
    study_tables: Mapping[str, pd.DataFrame],
    reference_tables: Mapping[str, pd.DataFrame],
    matched_tables: Mapping[str, pd.DataFrame],
    notebook_path: str | Path,
    output_root: str | Path,
    audit: Mapping[str, object],
    artifact_sources: Mapping[str, Mapping[str, object]],
    findings: Sequence[str] = (),
    calibration_tables: Mapping[str, Rows] | None = None,
    tick_inventory_rows: Rows | None = None,
    l2_audit_rows: Rows | None = None,
    shadow_rows: Rows | None = None,
) -> Path:
    """Render one static workbench from verified, content-addressed evidence."""

    payload = _payload(
        study_tables=study_tables,
        reference_tables=reference_tables,
        matched_tables=matched_tables,
        notebook_path=Path(notebook_path).resolve(),
        audit=audit,
        findings=findings,
        artifact_sources=artifact_sources,
        calibration_tables=calibration_tables,
        tick_inventory_rows=tick_inventory_rows,
        l2_audit_rows=l2_audit_rows,
        shadow_rows=shadow_rows,
    )
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    encoded = _HTML.replace("__DATA__", data).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    root = Path(output_root).resolve()
    path = root / "sha256" / digest[:2] / f"{digest}.html"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ResearchWorkbenchError("content-addressed HTML collision")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)
    return path


__all__ = [
    "ResearchWorkbenchEvidence",
    "ResearchWorkbenchError",
    "canonical_rows_sha256",
    "load_research_workbench_evidence",
    "research_workbench_zone_frame",
    "render_research_workbench",
]
