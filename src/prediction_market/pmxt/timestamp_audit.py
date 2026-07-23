"""Deterministic X-02 PMXT source/receive timestamp audit."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import date, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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
    absolute_p99_ms: float
    quantile_method: str
    negative_delta_count: int
    negative_delta_rate: float
    ordered_comparison_count: int
    out_of_order_count: int
    out_of_order_rate: float
    disorder_definition: str
    hourly_medians_ms: dict[str, float]
    hourly_median_drift_ms: float
    millisecond_research_eligible: bool
    downgrade_triggers: tuple[str, ...]
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


def _exact_frequency_quantile(
    counts: Counter[int], probability: float
) -> float:
    """Return exact ``quantile_cont`` from an integer frequency table."""

    total = sum(counts.values())
    if total <= 0:
        raise TimestampAuditError("timestamp audit histogram is empty")
    if not 0.0 <= probability <= 1.0:
        raise TimestampAuditError("quantile probability must be in [0, 1]")
    position = (total - 1) * probability
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)
    targets = {lower_rank, upper_rank}
    values: dict[int, int] = {}
    cumulative = 0
    for value, frequency in sorted(counts.items()):
        next_cumulative = cumulative + frequency
        for target in targets - values.keys():
            if cumulative <= target < next_cumulative:
                values[target] = value
        if len(values) == len(targets):
            break
        cumulative = next_cumulative
    if values.keys() != targets:
        raise TimestampAuditError("timestamp audit histogram ranks are incomplete")
    lower = float(values[lower_rank])
    upper = float(values[upper_rank])
    return lower + (position - lower_rank) * (upper - lower)


def _delta_histograms(
    paths: list[str],
) -> tuple[Counter[int], dict[int, Counter[int]]]:
    """Build exact integer-ms frequencies without retaining source rows."""

    combined: Counter[int] = Counter()
    hourly: dict[int, Counter[int]] = defaultdict(Counter)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        for path in paths:
            rows = connection.execute(
                """
                SELECT
                    extract(hour FROM timestamp_received)::INTEGER AS utc_hour,
                    epoch_ms(timestamp_received) - epoch_ms(timestamp) AS delta_ms,
                    count(*) AS frequency
                FROM read_parquet(?)
                GROUP BY utc_hour, delta_ms
                ORDER BY utc_hour, delta_ms
                """,
                [path],
            ).fetchall()
            for utc_hour, delta_ms, frequency in rows:
                if utc_hour is None or delta_ms is None:
                    raise TimestampAuditError(
                        "PMXT timestamp audit contains NULL required fields"
                    )
                hour = int(utc_hour)
                delta = int(delta_ms)
                count = int(frequency)
                if not 0 <= hour <= 23 or count <= 0:
                    raise TimestampAuditError(
                        "PMXT timestamp histogram contains invalid values"
                    )
                combined[delta] += count
                hourly[hour][delta] += count
    except duckdb.Error as exc:
        raise TimestampAuditError(f"PMXT timestamp audit failed: {exc}") from exc
    finally:
        connection.close()
    return combined, dict(hourly)


def _canonical_disorder_counts(
    paths: list[str], *, batch_size: int = 1_048_576
) -> tuple[int, int, int]:
    """Count source-clock regressions using PMXT's verified native ordering.

    PMXT v2 is physically ordered by ``(market, asset_id,
    timestamp_received)``.  Within a receive-time tie the canonical replay key
    orders by source timestamp, so no tie-internal regression is possible.  A
    regression can only occur from the maximum source timestamp in one receive
    run to the minimum source timestamp in the next run.  This streaming form
    is equivalent to the canonical window calculation but does not materialize
    or sort billions of rows.
    """

    if type(batch_size) is not int or batch_size <= 0:
        raise TimestampAuditError("batch_size must be a positive integer")
    latest: dict[tuple[bytes, str], tuple[int, int]] = {}
    row_count = 0
    comparison_count = 0
    disorder_count = 0
    columns = ["timestamp_received", "timestamp", "market", "asset_id"]

    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            prior_native_key: tuple[bytes, str] | None = None
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=columns,
                use_threads=True,
            ):
                size = batch.num_rows
                if size == 0:
                    continue
                if any(batch.column(index).null_count for index in range(4)):
                    raise TimestampAuditError(
                        "PMXT timestamp audit contains NULL required fields"
                    )
                row_count += size
                received = (
                    batch.column(0)
                    .to_numpy(zero_copy_only=False)
                    .astype("datetime64[ms]")
                    .astype(np.int64, copy=False)
                )
                source = (
                    batch.column(1)
                    .to_numpy(zero_copy_only=False)
                    .astype("datetime64[ms]")
                    .astype(np.int64, copy=False)
                )
                market = batch.column(2)
                asset = batch.column(3)

                if size > 1:
                    same_key = np.logical_and(
                        pc.equal(
                            market.slice(1), market.slice(0, size - 1)
                        ).to_numpy(zero_copy_only=False),
                        pc.equal(
                            asset.slice(1), asset.slice(0, size - 1)
                        ).to_numpy(zero_copy_only=False),
                    )
                    if np.any(same_key & (received[1:] < received[:-1])):
                        raise TimestampAuditError(
                            "native PMXT order regresses timestamp_received"
                        )
                    comparison_count += int(same_key.sum())
                    group_starts = np.concatenate(
                        (
                            np.array([0], dtype=np.int64),
                            np.flatnonzero(~same_key).astype(np.int64) + 1,
                        )
                    )
                    same_receive_run = same_key & (
                        received[1:] == received[:-1]
                    )
                    run_starts = np.concatenate(
                        (
                            np.array([0], dtype=np.int64),
                            np.flatnonzero(~same_receive_run).astype(np.int64)
                            + 1,
                        )
                    )
                else:
                    group_starts = np.array([0], dtype=np.int64)
                    run_starts = np.array([0], dtype=np.int64)

                run_min_source = np.minimum.reduceat(source, run_starts)
                run_max_source = np.maximum.reduceat(source, run_starts)
                run_receive = received[run_starts]
                group_run_positions = np.searchsorted(
                    run_starts, group_starts
                )
                last_run_positions = np.concatenate(
                    (
                        group_run_positions[1:] - 1,
                        np.array([len(run_starts) - 1], dtype=np.int64),
                    )
                )
                take_indices = pa.array(group_starts, type=pa.int64())
                market_keys = pc.take(market, take_indices).to_pylist()
                asset_keys = pc.take(asset, take_indices).to_pylist()

                effective_run_max = run_max_source.copy()
                keys: list[tuple[bytes, str]] = []
                for group_index, (market_key, asset_key) in enumerate(
                    zip(market_keys, asset_keys, strict=True)
                ):
                    if not isinstance(market_key, bytes) or not isinstance(
                        asset_key, str
                    ):
                        raise TimestampAuditError(
                            "PMXT market or asset identifier has invalid type"
                        )
                    key = (market_key, asset_key)
                    if prior_native_key is not None and key < prior_native_key:
                        raise TimestampAuditError(
                            "native PMXT order regresses market/asset key"
                        )
                    prior_native_key = key
                    keys.append(key)

                    first_run = int(group_run_positions[group_index])
                    prior = latest.get(key)
                    if prior is None:
                        continue
                    comparison_count += 1
                    prior_receive, prior_max_source = prior
                    first_receive = int(run_receive[first_run])
                    if first_receive < prior_receive:
                        raise TimestampAuditError(
                            "native PMXT order regresses timestamp_received"
                        )
                    if first_receive == prior_receive:
                        effective_run_max[first_run] = max(
                            int(effective_run_max[first_run]), prior_max_source
                        )
                    elif int(run_min_source[first_run]) < prior_max_source:
                        disorder_count += 1

                internal = np.ones(len(run_starts), dtype=bool)
                internal[group_run_positions] = False
                if len(run_starts) > 1:
                    previous_run_max = np.empty_like(effective_run_max)
                    previous_run_max[0] = effective_run_max[0]
                    previous_run_max[1:] = effective_run_max[:-1]
                    disorder_count += int(
                        np.sum(internal & (run_min_source < previous_run_max))
                    )

                for group_index, key in enumerate(keys):
                    last_run = int(last_run_positions[group_index])
                    latest[key] = (
                        int(run_receive[last_run]),
                        int(effective_run_max[last_run]),
                    )
    except TimestampAuditError:
        raise
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise TimestampAuditError(
            f"PMXT timestamp disorder scan failed: {exc}"
        ) from exc

    expected_comparisons = row_count - len(latest)
    if comparison_count != expected_comparisons:
        raise TimestampAuditError(
            "canonical disorder comparison count violates stream invariant"
        )
    return row_count, comparison_count, disorder_count


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
    row_count, ordered_count, out_of_order_count = _canonical_disorder_counts(
        paths
    )
    histogram, hourly_histograms = _delta_histograms(paths)
    delta_count = sum(histogram.values())
    if row_count <= 0:
        raise TimestampAuditError("PMXT timestamp audit day contains no rows")
    if delta_count != row_count:
        raise TimestampAuditError(
            "PMXT timestamp audit contains NULL required fields"
        )
    if set(hourly_histograms) != set(range(24)):
        raise TimestampAuditError(
            "PMXT timestamp audit must cover all 24 receive-time hours"
        )
    hourly = {
        f"{hour:02d}": _exact_frequency_quantile(counts, 0.50)
        for hour, counts in sorted(hourly_histograms.items())
    }
    negative_count = sum(
        frequency for delta, frequency in histogram.items() if delta < 0
    )
    p50 = _exact_frequency_quantile(histogram, 0.50)
    p95 = _exact_frequency_quantile(histogram, 0.95)
    p99 = _exact_frequency_quantile(histogram, 0.99)
    absolute_histogram: Counter[int] = Counter()
    for delta, frequency in histogram.items():
        absolute_histogram[abs(delta)] += frequency
    absolute_p99 = _exact_frequency_quantile(absolute_histogram, 0.99)
    negative_rate = float(negative_count) / delta_count
    downgrade_triggers: list[str] = []
    if negative_rate >= 0.001:
        downgrade_triggers.append("negative_delta_rate_ge_0.001")
    if absolute_p99 > 5_000.0:
        downgrade_triggers.append("absolute_p99_ms_gt_5000")
    first_hour = min(hourly)
    last_hour = max(hourly)
    provisional = TimestampAuditReport(
        version="pmxt-timestamp-audit-v1",
        day=manifest.day,
        input_manifest_sha256=manifest.manifest_sha256,
        row_count=int(row_count),
        delta_count=int(delta_count),
        minimum_delta_ms=float(min(histogram)),
        maximum_delta_ms=float(max(histogram)),
        quantiles_ms={"p50": float(p50), "p95": float(p95), "p99": float(p99)},
        absolute_p99_ms=float(absolute_p99),
        quantile_method="exact_frequency_quantile_cont_ms_v1",
        negative_delta_count=int(negative_count),
        negative_delta_rate=negative_rate,
        ordered_comparison_count=int(ordered_count),
        out_of_order_count=int(out_of_order_count),
        out_of_order_rate=(
            float(out_of_order_count) / int(ordered_count) if ordered_count else 0.0
        ),
        disorder_definition=(
            "per_market_asset_canonical_receive_source_adjacent_source_regression_v1"
        ),
        hourly_medians_ms=hourly,
        hourly_median_drift_ms=hourly[last_hour] - hourly[first_hour],
        millisecond_research_eligible=not downgrade_triggers,
        downgrade_triggers=tuple(downgrade_triggers),
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
