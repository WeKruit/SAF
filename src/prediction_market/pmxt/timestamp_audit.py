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
    LockedHourlyObject,
    _resolve_locked_path,
    _sha256_path,
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
    code_sha256: str
    data_sha256: str
    input_bundle_path: str
    input_bundle_file_sha256: str
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


@dataclass(frozen=True, slots=True)
class _TimestampInputBundle:
    relative_path: str
    file_sha256: str
    bundle_sha256: str
    manifests: tuple[FullDayManifest, ...]


@dataclass(frozen=True, slots=True)
class _GovernedX02Input:
    bundle_path: str
    bundle_file_sha256: str
    bundle_sha256: str
    code_sha256: str
    data_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(payload: bytes, *, context: str) -> dict[str, object]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise TimestampAuditError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                TimestampAuditError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, TimestampAuditError) as exc:
        raise TimestampAuditError(f"{context} is not strict JSON") from exc
    if type(parsed) is not dict:
        raise TimestampAuditError(f"{context} must be a JSON object")
    return parsed


def _require_exact_keys(
    value: dict[str, object], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        raise TimestampAuditError(f"{context} fields do not match the v1 contract")


def _require_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise TimestampAuditError(f"{field} must be a non-empty string")
    return value


def _require_integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TimestampAuditError(f"{field} must be an integer")
    return value


def x02_runner_code_sha256(program_root: str | Path) -> str:
    """Hash the fixed X-02 execution source set and dependency lock."""

    import prediction_market.compliance as compliance_module
    import prediction_market.contracts as contracts_module
    import prediction_market.experiments as experiments_module
    import prediction_market.pmxt.full_day as full_day_module
    import prediction_market.program_audit as program_audit_module

    try:
        dependency_lock = _resolve_locked_path(Path(program_root), "uv.lock")
    except FullDayInputError as exc:
        raise TimestampAuditError(
            f"X-02 dependency lock is unavailable: {exc}"
        ) from exc
    source_paths = {
        "src/prediction_market/compliance.py": Path(compliance_module.__file__),
        "src/prediction_market/contracts.py": Path(contracts_module.__file__),
        "src/prediction_market/experiments.py": Path(experiments_module.__file__),
        "src/prediction_market/pmxt/full_day.py": Path(full_day_module.__file__),
        "src/prediction_market/pmxt/timestamp_audit.py": Path(__file__),
        "src/prediction_market/program_audit.py": Path(program_audit_module.__file__),
        "uv.lock": dependency_lock,
    }
    manifest = {
        relative: _sha256_path(path)
        for relative, path in sorted(source_paths.items())
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _load_governed_x02_input(program_root: str | Path) -> _GovernedX02Input:
    from prediction_market.experiments import (
        ExperimentRegistryError,
        load_experiment_registry,
    )

    try:
        registry = load_experiment_registry(program_root)
    except ExperimentRegistryError as exc:
        raise TimestampAuditError(
            f"X-02 governance validation failed before measurement: {exc}"
        ) from exc
    card = registry.get("X-02")
    if type(card) is not dict:
        raise TimestampAuditError("X-02 governance registration is missing")
    binding = card.get("timestamp_input_manifest_binding")
    if type(binding) is not dict:
        raise TimestampAuditError("X-02 governed input binding is missing")
    _require_exact_keys(
        binding,
        {"bundle_path", "bundle_file_sha256", "bundle_sha256"},
        context="X-02 governed input binding",
    )
    bundle_path = _require_string(
        binding["bundle_path"], field="X-02 governed bundle_path"
    )
    bundle_file_sha256 = _require_string(
        binding["bundle_file_sha256"],
        field="X-02 governed bundle_file_sha256",
    )
    bundle_sha256 = _require_string(
        binding["bundle_sha256"], field="X-02 governed bundle_sha256"
    )
    if any(
        _SHA256_PATTERN.fullmatch(value) is None
        for value in (bundle_file_sha256, bundle_sha256)
    ):
        raise TimestampAuditError("X-02 governed bundle hashes are invalid")

    preregistered = card.get("preregistered_inputs")
    if type(preregistered) is not dict:
        raise TimestampAuditError("X-02 governed preregistered inputs are missing")
    formal = preregistered.get("formal_result")
    if type(formal) is not dict:
        raise TimestampAuditError(
            "X-02 formal runner code and data hashes are not preregistered"
        )
    _require_exact_keys(
        formal,
        {
            "code_sha256",
            "data_sha256",
            "dataset_ids",
            "model_ids",
            "registered_at",
        },
        context="X-02 formal preregistered input",
    )
    code_sha256 = _require_string(
        formal["code_sha256"], field="X-02 preregistered code_sha256"
    )
    data_sha256 = _require_string(
        formal["data_sha256"], field="X-02 preregistered data_sha256"
    )
    if code_sha256 != x02_runner_code_sha256(program_root):
        raise TimestampAuditError(
            "X-02 runner source does not match the preregistered code SHA-256"
        )
    if data_sha256 != bundle_sha256:
        raise TimestampAuditError(
            "X-02 governed bundle does not match preregistered data SHA-256"
        )
    if formal["dataset_ids"] != ["DS-PMXT-V2"] or formal["model_ids"] != []:
        raise TimestampAuditError(
            "X-02 formal preregistered dataset/model binding is invalid"
        )
    return _GovernedX02Input(
        bundle_path=bundle_path,
        bundle_file_sha256=bundle_file_sha256,
        bundle_sha256=bundle_sha256,
        code_sha256=code_sha256,
        data_sha256=data_sha256,
    )


def _parse_full_day_manifest(
    payload: bytes, *, context: str
) -> FullDayManifest:
    document = _strict_json_object(payload, context=context)
    _require_exact_keys(
        document,
        {
            "version",
            "day",
            "inventory_sha256",
            "canonicalization_version",
            "objects",
            "manifest_sha256",
        },
        context=context,
    )
    raw_objects = document["objects"]
    if type(raw_objects) is not list:
        raise TimestampAuditError(f"{context}.objects must be a list")
    objects: list[LockedHourlyObject] = []
    for index, raw_object in enumerate(raw_objects):
        object_context = f"{context}.objects[{index}]"
        if type(raw_object) is not dict:
            raise TimestampAuditError(f"{object_context} must be an object")
        _require_exact_keys(
            raw_object,
            {
                "hour",
                "source_url",
                "object_path",
                "object_sha256",
                "static_manifest_sha256",
                "inventory_size_bytes",
            },
            context=object_context,
        )
        inventory_size = raw_object["inventory_size_bytes"]
        if inventory_size is not None and type(inventory_size) is not int:
            raise TimestampAuditError(
                f"{object_context}.inventory_size_bytes must be an integer or null"
            )
        objects.append(
            LockedHourlyObject(
                hour=_require_string(raw_object["hour"], field=f"{object_context}.hour"),
                source_url=_require_string(
                    raw_object["source_url"], field=f"{object_context}.source_url"
                ),
                object_path=_require_string(
                    raw_object["object_path"], field=f"{object_context}.object_path"
                ),
                object_sha256=_require_string(
                    raw_object["object_sha256"],
                    field=f"{object_context}.object_sha256",
                ),
                static_manifest_sha256=_require_string(
                    raw_object["static_manifest_sha256"],
                    field=f"{object_context}.static_manifest_sha256",
                ),
                inventory_size_bytes=inventory_size,
            )
        )
    manifest = FullDayManifest(
        version=_require_string(document["version"], field=f"{context}.version"),
        day=_require_string(document["day"], field=f"{context}.day"),
        inventory_sha256=_require_string(
            document["inventory_sha256"], field=f"{context}.inventory_sha256"
        ),
        canonicalization_version=_require_string(
            document["canonicalization_version"],
            field=f"{context}.canonicalization_version",
        ),
        objects=tuple(objects),
        manifest_sha256=_require_string(
            document["manifest_sha256"], field=f"{context}.manifest_sha256"
        ),
    )
    try:
        return validate_full_day_manifest(manifest)
    except FullDayInputError as exc:
        raise TimestampAuditError(f"{context} is invalid: {exc}") from exc


def _load_timestamp_input_bundle(
    program_root: str | Path, relative_path: str
) -> _TimestampInputBundle:
    try:
        bundle_path = _resolve_locked_path(Path(program_root), relative_path)
    except FullDayInputError as exc:
        raise TimestampAuditError(f"invalid X-02 input bundle path: {exc}") from exc
    bundle_payload = bundle_path.read_bytes()
    document = _strict_json_object(bundle_payload, context="X-02 input bundle")
    _require_exact_keys(
        document,
        {
            "additional_days",
            "bundle_sha256",
            "day_count",
            "day_manifests",
            "formal_result",
            "inventory_path",
            "inventory_sha256",
            "object_count",
            "purpose",
            "selection_procedure",
            "selection_seed",
            "version",
            "x01_day",
        },
        context="X-02 input bundle",
    )
    if document["version"] != "x02-timestamp-input-bundle-v1":
        raise TimestampAuditError("unsupported X-02 input bundle version")
    if document["formal_result"] is not False:
        raise TimestampAuditError("X-02 input bundle formal_result must be false")
    if document["purpose"] != "frozen_input_only_before_X02_evaluation":
        raise TimestampAuditError("X-02 input bundle purpose is invalid")
    if _require_integer(document["selection_seed"], field="selection_seed") != 20260722:
        raise TimestampAuditError("X-02 input bundle selection_seed is not locked")
    if document["x01_day"] != "2026-05-28":
        raise TimestampAuditError("X-02 input bundle x01_day is not locked")
    bundle_sha256 = _require_string(
        document["bundle_sha256"], field="bundle_sha256"
    )
    if _SHA256_PATTERN.fullmatch(bundle_sha256) is None:
        raise TimestampAuditError("bundle_sha256 must be a lowercase sha256: digest")
    material = dict(document)
    material.pop("bundle_sha256")
    expected_bundle_sha256 = (
        "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()
    )
    if bundle_sha256 != expected_bundle_sha256:
        raise TimestampAuditError("bundle_sha256 does not match bundle content")

    raw_entries = document["day_manifests"]
    if type(raw_entries) is not list:
        raise TimestampAuditError("day_manifests must be a list")
    day_count = _require_integer(document["day_count"], field="day_count")
    if day_count != 4 or len(raw_entries) != 4:
        raise TimestampAuditError(
            "X-02 timestamp sample must contain exactly four UTC days"
        )

    manifests: list[FullDayManifest] = []
    artifact_paths: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        context = f"day_manifests[{index}]"
        if type(raw_entry) is not dict:
            raise TimestampAuditError(f"{context} must be an object")
        _require_exact_keys(
            raw_entry,
            {
                "artifact_file_sha256",
                "day",
                "full_day_manifest_sha256",
                "object_count",
                "path",
            },
            context=context,
        )
        artifact_path = _require_string(raw_entry["path"], field=f"{context}.path")
        if artifact_path in artifact_paths:
            raise TimestampAuditError("day manifest artifact paths must be unique")
        artifact_paths.add(artifact_path)
        try:
            resolved = _resolve_locked_path(Path(program_root), artifact_path)
        except FullDayInputError as exc:
            raise TimestampAuditError(f"invalid {context}.path: {exc}") from exc
        payload = resolved.read_bytes()
        expected_file_sha256 = _require_string(
            raw_entry["artifact_file_sha256"],
            field=f"{context}.artifact_file_sha256",
        )
        if _SHA256_PATTERN.fullmatch(expected_file_sha256) is None:
            raise TimestampAuditError(
                f"{context}.artifact_file_sha256 must be a lowercase sha256: digest"
            )
        actual_file_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
        if actual_file_sha256 != expected_file_sha256:
            raise TimestampAuditError(
                f"{context} artifact file SHA-256 does not match the bundle"
            )
        manifest = _parse_full_day_manifest(payload, context=context)
        if manifest.day != raw_entry["day"]:
            raise TimestampAuditError(f"{context}.day does not match its manifest")
        if manifest.manifest_sha256 != raw_entry["full_day_manifest_sha256"]:
            raise TimestampAuditError(
                f"{context}.full_day_manifest_sha256 does not match its manifest"
            )
        if _require_integer(raw_entry["object_count"], field=f"{context}.object_count") != len(
            manifest.objects
        ):
            raise TimestampAuditError(f"{context}.object_count does not match")
        manifests.append(manifest)

    days = tuple(manifest.day for manifest in manifests)
    if list(days) != sorted(days) or len(set(days)) != len(days):
        raise TimestampAuditError(
            "X-02 timestamp sample days must be unique and strictly increasing"
        )
    if document["x01_day"] not in days:
        raise TimestampAuditError("X-02 timestamp sample does not contain x01_day")
    additional_days = document["additional_days"]
    expected_additional = [day for day in days if day != document["x01_day"]]
    if additional_days != expected_additional:
        raise TimestampAuditError("additional_days does not match bound day manifests")
    if any(
        manifest.inventory_sha256 != document["inventory_sha256"]
        for manifest in manifests
    ):
        raise TimestampAuditError("day manifest inventory SHA-256 differs from bundle")
    object_count = _require_integer(document["object_count"], field="object_count")
    if object_count != sum(len(manifest.objects) for manifest in manifests):
        raise TimestampAuditError("bundle object_count does not match day manifests")
    if object_count != 96:
        raise TimestampAuditError(
            "X-02 timestamp sample must bind exactly 96 hourly objects"
        )
    return _TimestampInputBundle(
        relative_path=relative_path,
        file_sha256=_sha256_path(bundle_path),
        bundle_sha256=bundle_sha256,
        manifests=tuple(manifests),
    )


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
    *,
    program_root: str | Path,
    input_bundle_path: str,
) -> TimestampSampleAuditReport:
    """Audit the exact four-day X-02 sample as one preregistered unit."""

    if type(input_bundle_path) is not str or not input_bundle_path:
        raise TimestampAuditError("input_bundle_path must be a non-empty string")
    governed = _load_governed_x02_input(program_root)
    if input_bundle_path != governed.bundle_path:
        raise TimestampAuditError(
            "input_bundle_path does not match the governed X-02 binding"
        )
    bundle = _load_timestamp_input_bundle(program_root, input_bundle_path)
    if (
        bundle.file_sha256 != governed.bundle_file_sha256
        or bundle.bundle_sha256 != governed.bundle_sha256
    ):
        raise TimestampAuditError(
            "X-02 input bundle does not match the governed file/self hashes"
        )
    frozen = bundle.manifests

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
        code_sha256=governed.code_sha256,
        data_sha256=governed.data_sha256,
        input_bundle_path=bundle.relative_path,
        input_bundle_file_sha256=bundle.file_sha256,
        input_bundle_sha256=bundle.bundle_sha256,
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
    post_bundle = _load_timestamp_input_bundle(program_root, input_bundle_path)
    if post_bundle != bundle:
        raise TimestampAuditError(
            "frozen input bundle or day manifests changed during timestamp audit"
        )
    post_governed = _load_governed_x02_input(program_root)
    if post_governed != governed:
        raise TimestampAuditError(
            "X-02 governance binding changed during timestamp audit"
        )
    return report


__all__ = [
    "TimestampAuditError",
    "TimestampAuditReport",
    "TimestampSampleAuditReport",
    "audit_full_day_timestamps",
    "audit_timestamp_sample",
    "select_timestamp_audit_days",
    "x02_runner_code_sha256",
]
