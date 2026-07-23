"""Deterministic X-02 PMXT source/receive timestamp audit."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import date, timezone
from pathlib import Path
from typing import Any

import duckdb

from prediction_market.pmxt.archive import ArchiveEntry
from prediction_market.pmxt.full_day import (
    FullDayInputError,
    FullDayManifest,
    _verified_paths,
    validate_full_day_manifest,
)


class TimestampAuditError(ValueError):
    """The X-02 timestamp selection or locked inputs are invalid."""


@dataclass(frozen=True, slots=True)
class TimestampAuditReport:
    version: str
    day: str
    input_manifest_sha256: str
    row_count: int
    delta_count: int
    minimum_delta_ms: float
    maximum_delta_ms: float
    quantiles_ms: dict[str, float]
    negative_delta_count: int
    negative_delta_rate: float
    ordered_comparison_count: int
    out_of_order_count: int
    out_of_order_rate: float
    hourly_medians_ms: dict[str, float]
    hourly_median_drift_ms: float
    timestamp_semantics: str
    report_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_hash(report: TimestampAuditReport) -> str:
    material = asdict(report)
    material.pop("report_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _entry_utc_hour(entry: ArchiveEntry) -> tuple[date, int]:
    value = entry.hour
    if value.tzinfo is None:
        raise TimestampAuditError("archive inventory hours must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.minute or normalized.second or normalized.microsecond:
        raise TimestampAuditError("archive inventory entries must identify exact hours")
    return normalized.date(), normalized.hour


def select_timestamp_audit_days(
    entries: Iterable[ArchiveEntry],
    *,
    x01_day: date,
    additional_days: int,
    seed: int,
) -> tuple[date, ...]:
    """Return X-01 day plus a seeded sample of other exact complete days."""

    if not isinstance(x01_day, date):
        raise TimestampAuditError("x01_day must be a date")
    if type(additional_days) is not int or additional_days < 0:
        raise TimestampAuditError("additional_days must be a nonnegative integer")
    if type(seed) is not int:
        raise TimestampAuditError("seed must be an integer")
    observed: dict[date, Counter[int]] = defaultdict(Counter)
    for entry in entries:
        observed_day, hour = _entry_utc_hour(entry)
        observed[observed_day][hour] += 1
    expected_hours = set(range(24))
    complete = sorted(
        observed_day
        for observed_day, counts in observed.items()
        if set(counts) == expected_hours
        and all(counts[hour] == 1 for hour in expected_hours)
    )
    if x01_day not in complete:
        raise TimestampAuditError("X-01 day must be an exact complete UTC day")
    candidates = [candidate for candidate in complete if candidate != x01_day]
    if len(candidates) < additional_days:
        raise TimestampAuditError(
            "not enough additional complete UTC days for X-02 selection"
        )
    sampled = random.Random(seed).sample(candidates, additional_days)
    return (x01_day, *sorted(sampled))


def _audit_query(paths: list[str]) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]]:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET TimeZone = 'UTC'")
        summary = connection.execute(
            """
            WITH base AS (
                SELECT
                    epoch_ms(timestamp_received) AS receive_ms,
                    epoch_ms(timestamp) AS source_ms,
                    market,
                    asset_id
                FROM read_parquet(?, union_by_name = true)
            ), ordered AS (
                SELECT
                    *,
                    receive_ms - source_ms AS delta_ms,
                    lag(source_ms) OVER (
                        PARTITION BY market, asset_id
                        ORDER BY receive_ms, source_ms
                    ) AS prior_source_ms
                FROM base
            )
            SELECT
                count(*) AS row_count,
                count(delta_ms) AS delta_count,
                count(*) FILTER (
                    WHERE receive_ms IS NULL OR source_ms IS NULL
                       OR market IS NULL OR asset_id IS NULL
                ) AS invalid_required_rows,
                min(delta_ms) AS minimum_delta_ms,
                max(delta_ms) AS maximum_delta_ms,
                quantile_cont(delta_ms, 0.50) AS p50,
                quantile_cont(delta_ms, 0.95) AS p95,
                quantile_cont(delta_ms, 0.99) AS p99,
                count(*) FILTER (WHERE delta_ms < 0) AS negative_count,
                count(*) FILTER (WHERE prior_source_ms IS NOT NULL)
                    AS ordered_comparison_count,
                count(*) FILTER (WHERE source_ms < prior_source_ms)
                    AS out_of_order_count
            FROM ordered
            """,
            [paths],
        ).fetchone()
        hourly = connection.execute(
            """
            SELECT
                extract(hour FROM timestamp_received)::INTEGER AS utc_hour,
                median(
                    epoch_ms(timestamp_received) - epoch_ms(timestamp)
                ) AS median_delta_ms
            FROM read_parquet(?, union_by_name = true)
            WHERE timestamp_received IS NOT NULL AND timestamp IS NOT NULL
            GROUP BY utc_hour
            ORDER BY utc_hour
            """,
            [paths],
        ).fetchall()
    except duckdb.Error as exc:
        raise TimestampAuditError(f"PMXT timestamp audit failed: {exc}") from exc
    finally:
        connection.close()
    if summary is None:
        raise TimestampAuditError("PMXT timestamp audit produced no summary")
    return summary, hourly


def audit_full_day_timestamps(
    raw_root: str | Path, manifest: FullDayManifest
) -> TimestampAuditReport:
    """Audit one frozen day, retaining receive time as the replay clock."""

    try:
        validate_full_day_manifest(manifest)
        locked_paths = _verified_paths(raw_root, manifest)
    except FullDayInputError as exc:
        raise TimestampAuditError(f"invalid frozen full-day input: {exc}") from exc
    paths = [str(path.resolve()) for path in locked_paths]
    summary, hourly_rows = _audit_query(paths)
    (
        row_count,
        delta_count,
        invalid_required_rows,
        minimum_delta_ms,
        maximum_delta_ms,
        p50,
        p95,
        p99,
        negative_count,
        ordered_count,
        out_of_order_count,
    ) = summary
    if row_count <= 0:
        raise TimestampAuditError("PMXT timestamp audit day contains no rows")
    if invalid_required_rows or delta_count != row_count:
        raise TimestampAuditError(
            "PMXT timestamp audit contains NULL required fields"
        )
    hourly = {f"{int(hour):02d}": float(median) for hour, median in hourly_rows}
    if not hourly:
        raise TimestampAuditError("PMXT timestamp audit has no hourly medians")
    first_hour = min(hourly)
    last_hour = max(hourly)
    provisional = TimestampAuditReport(
        version="pmxt-timestamp-audit-v1",
        day=manifest.day,
        input_manifest_sha256=manifest.manifest_sha256,
        row_count=int(row_count),
        delta_count=int(delta_count),
        minimum_delta_ms=float(minimum_delta_ms),
        maximum_delta_ms=float(maximum_delta_ms),
        quantiles_ms={"p50": float(p50), "p95": float(p95), "p99": float(p99)},
        negative_delta_count=int(negative_count),
        negative_delta_rate=float(negative_count) / int(delta_count),
        ordered_comparison_count=int(ordered_count),
        out_of_order_count=int(out_of_order_count),
        out_of_order_rate=(
            float(out_of_order_count) / int(ordered_count) if ordered_count else 0.0
        ),
        hourly_medians_ms=hourly,
        hourly_median_drift_ms=hourly[last_hour] - hourly[first_hour],
        timestamp_semantics="receive_at_primary;source_at_secondary_audit_only",
        report_sha256="",
    )
    report = replace(provisional, report_sha256=_report_hash(provisional))
    try:
        post_read_paths = _verified_paths(raw_root, manifest)
    except FullDayInputError as exc:
        raise TimestampAuditError(f"frozen input changed during audit: {exc}") from exc
    if post_read_paths != locked_paths:
        raise TimestampAuditError("frozen input paths changed during timestamp audit")
    return report


__all__ = [
    "TimestampAuditError",
    "TimestampAuditReport",
    "audit_full_day_timestamps",
    "select_timestamp_audit_days",
]
