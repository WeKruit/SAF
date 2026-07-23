"""Deterministic X-02 PMXT source/receive timestamp audit."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
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


_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


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


@dataclass(frozen=True, slots=True)
class TimestampSampleAuditReport:
    version: str
    days: tuple[str, ...]
    input_bundle_sha256: str
    input_manifest_sha256s: tuple[str, ...]
    day_count: int
    object_count: int
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


@dataclass(frozen=True, slots=True)
class _TimestampMetrics:
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


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_hash(
    report: TimestampAuditReport | TimestampSampleAuditReport,
) -> str:
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
    # Each key retains the last *completed* receive-run maximum plus the
    # still-open receive run.  The latter cannot be compared with its
    # predecessor until its full minimum is known: one receive-time tie may
    # span Arrow batches (or adjacent hourly objects).
    latest: dict[
        tuple[bytes, str], tuple[int | None, int, int, int]
    ] = {}
    row_count = 0
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

                    first_run = int(group_run_positions[group_index])
                    last_run = int(last_run_positions[group_index])
                    run_count = last_run - first_run + 1
                    first_receive = int(run_receive[first_run])
                    first_min = int(run_min_source[first_run])
                    first_max = int(run_max_source[first_run])
                    completed_max: int | None = None
                    prior = latest.get(key)
                    if prior is not None:
                        (
                            completed_max,
                            prior_receive,
                            prior_min,
                            prior_max,
                        ) = prior
                        if first_receive < prior_receive:
                            raise TimestampAuditError(
                                "native PMXT order regresses timestamp_received"
                            )
                        if first_receive == prior_receive:
                            first_min = min(first_min, prior_min)
                            first_max = max(first_max, prior_max)
                        else:
                            if (
                                completed_max is not None
                                and prior_min < completed_max
                            ):
                                disorder_count += 1
                            completed_max = prior_max

                    # Every run except the last is complete because the next
                    # receive timestamp has already been observed.  Keep the
                    # last run open so a lower tied source timestamp in the
                    # next batch cannot be missed.
                    if run_count > 1:
                        if (
                            completed_max is not None
                            and first_min < completed_max
                        ):
                            disorder_count += 1
                        if run_count > 2:
                            previous_max = run_max_source[
                                first_run : last_run - 1
                            ].copy()
                            previous_max[0] = first_max
                            target_min = run_min_source[
                                first_run + 1 : last_run
                            ]
                            disorder_count += int(
                                np.sum(target_min < previous_max)
                            )
                        completed_max = (
                            first_max
                            if run_count == 2
                            else int(run_max_source[last_run - 1])
                        )

                    latest[key] = (
                        completed_max,
                        int(run_receive[last_run]),
                        (
                            first_min
                            if run_count == 1
                            else int(run_min_source[last_run])
                        ),
                        (
                            first_max
                            if run_count == 1
                            else int(run_max_source[last_run])
                        ),
                    )
    except TimestampAuditError:
        raise
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise TimestampAuditError(
            f"PMXT timestamp disorder scan failed: {exc}"
        ) from exc

    for completed_max, _, current_min, _ in latest.values():
        if completed_max is not None and current_min < completed_max:
            disorder_count += 1

    comparison_count = row_count - len(latest)
    if comparison_count < 0:
        raise TimestampAuditError(
            "canonical disorder comparison count violates stream invariant"
        )
    return row_count, comparison_count, disorder_count


def _compute_timestamp_metrics(paths: list[str]) -> _TimestampMetrics:
    if not paths:
        raise TimestampAuditError("PMXT timestamp audit requires source paths")
    row_count, ordered_count, out_of_order_count = _canonical_disorder_counts(
        paths
    )
    histogram, hourly_histograms = _delta_histograms(paths)
    delta_count = sum(histogram.values())
    if row_count <= 0:
        raise TimestampAuditError("PMXT timestamp audit sample contains no rows")
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
    return _TimestampMetrics(
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
    )


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
    metrics = _compute_timestamp_metrics(paths)
    provisional = TimestampAuditReport(
        version="pmxt-timestamp-audit-v1",
        day=manifest.day,
        input_manifest_sha256=manifest.manifest_sha256,
        **asdict(metrics),
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


def audit_timestamp_sample(
    raw_root: str | Path,
    manifests: Iterable[FullDayManifest],
    *,
    input_bundle_sha256: str,
) -> TimestampSampleAuditReport:
    """Audit the exact four-day X-02 sample as one preregistered unit."""

    frozen = tuple(manifests)
    if len(frozen) != 4:
        raise TimestampAuditError(
            "X-02 timestamp sample must contain exactly four UTC days"
        )
    if (
        type(input_bundle_sha256) is not str
        or _SHA256_PATTERN.fullmatch(input_bundle_sha256) is None
    ):
        raise TimestampAuditError(
            "input_bundle_sha256 must be a lowercase sha256: digest"
        )

    verified_by_day: list[tuple[Path, ...]] = []
    try:
        for manifest in frozen:
            validate_full_day_manifest(manifest)
            verified_by_day.append(_verified_paths(raw_root, manifest))
    except (FullDayInputError, TypeError) as exc:
        raise TimestampAuditError(
            f"invalid frozen timestamp sample input: {exc}"
        ) from exc

    days = tuple(manifest.day for manifest in frozen)
    if list(days) != sorted(days) or len(set(days)) != len(days):
        raise TimestampAuditError(
            "X-02 timestamp sample days must be unique and strictly increasing"
        )
    paths = tuple(path for day_paths in verified_by_day for path in day_paths)
    if len(paths) != 96:
        raise TimestampAuditError(
            "X-02 timestamp sample must bind exactly 96 hourly objects"
        )
    if len(set(paths)) != len(paths):
        raise TimestampAuditError(
            "X-02 timestamp sample contains duplicate hourly object paths"
        )

    metrics = _compute_timestamp_metrics(
        [str(path.resolve()) for path in paths]
    )
    provisional = TimestampSampleAuditReport(
        version="pmxt-timestamp-sample-audit-v1",
        days=days,
        input_bundle_sha256=input_bundle_sha256,
        input_manifest_sha256s=tuple(
            manifest.manifest_sha256 for manifest in frozen
        ),
        day_count=len(frozen),
        object_count=len(paths),
        **asdict(metrics),
        report_sha256="",
    )
    report = replace(provisional, report_sha256=_report_hash(provisional))

    try:
        post_verified = tuple(
            _verified_paths(raw_root, manifest) for manifest in frozen
        )
    except FullDayInputError as exc:
        raise TimestampAuditError(
            f"frozen timestamp sample changed during audit: {exc}"
        ) from exc
    if post_verified != tuple(verified_by_day):
        raise TimestampAuditError(
            "frozen input paths changed during timestamp sample audit"
        )
    return report


__all__ = [
    "TimestampAuditError",
    "TimestampAuditReport",
    "TimestampSampleAuditReport",
    "audit_full_day_timestamps",
    "audit_timestamp_sample",
    "select_timestamp_audit_days",
]
