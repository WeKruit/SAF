"""Frozen full-UTC-day PMXT reconstruction for X-01.

The runner binds exactly 24 immutable hourly objects, verifies their bytes
before and after use, and reconstructs each native market across hour
boundaries.  It deliberately leaves X-01's independent price-and-size
comparison gate open: PMXT alone cannot satisfy that gate.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.pmxt.archive import (
    ArchiveEntry,
    ArchiveIntegrityError,
    read_parquet_events,
)
from prediction_market.pmxt.reconstructor import PMXTValidationError, reconstruct


_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_MARKET_PATTERN = re.compile(r"0x[0-9a-fA-F]{64}")


class FullDayInputError(ValueError):
    """Raised when a frozen full-day input cannot be verified exactly."""


@dataclass(frozen=True)
class HourlyObjectRef:
    """Reference to one already-preserved PMXT hourly object."""

    hour: datetime
    source_url: str
    object_path: str
    object_sha256: str
    static_manifest_sha256: str


@dataclass(frozen=True)
class LockedHourlyObject:
    """Canonical JSON-safe hourly object embedded in the day manifest."""

    hour: str
    source_url: str
    object_path: str
    object_sha256: str
    static_manifest_sha256: str
    inventory_size_bytes: int | None


@dataclass(frozen=True)
class FullDayManifest:
    """Self-hashed binding for all inputs to one X-01 UTC-day run."""

    version: str
    day: str
    inventory_sha256: str
    canonicalization_version: str
    objects: tuple[LockedHourlyObject, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class MarketReconstruction:
    """Deterministic reconstruction summary for one PMXT condition."""

    native_market_id: str
    stream_sha256: str
    counts: dict[str, int]
    quality_flags: tuple[str, ...]
    queue_fill_reconstructed: bool = False


@dataclass(frozen=True)
class FullDayReconstructionReport:
    """Bounded, comparison-gated output of the full-day reconstruction."""

    version: str
    day: str
    input_manifest_sha256: str
    market_count: int
    input_event_count: int
    semantic_event_count: int
    market_results: tuple[MarketReconstruction, ...]
    day_stream_sha256: str
    independent_comparison_performed: bool
    x01_formal_gate_passed: bool
    open_gate: str
    queue_fill_reconstructed: bool = False


@dataclass(frozen=True)
class FullDayPreflightReport:
    """Verified scale and coverage facts without claiming reconstruction."""

    version: str
    day: str
    input_manifest_sha256: str
    object_count: int
    compressed_bytes: int
    row_count: int
    market_count: int
    asset_count: int
    event_type_counts: dict[str, int]
    timestamp_received_min: str
    timestamp_received_max: str
    schema_fingerprint: str
    reconstruction_executed: bool
    x01_formal_gate_passed: bool
    open_gates: tuple[str, ...]
    queue_fill_reconstructed: bool
    report_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _schema_fingerprint(schema: pa.Schema) -> str:
    material = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return _sha256_bytes(_canonical_bytes(material))


def _utc_text(value: datetime, *, field: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FullDayInputError(f"{field} must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _require_sha256(value: str, *, field: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise FullDayInputError(f"{field} must be a lowercase sha256: digest")


def _utc_hour(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise FullDayInputError(f"{field} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if (
        normalized.minute != 0
        or normalized.second != 0
        or normalized.microsecond != 0
    ):
        raise FullDayInputError(f"{field} must identify an exact UTC hour")
    return normalized


def _hour_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def select_complete_utc_day(
    entries: Iterable[ArchiveEntry], *, day: date
) -> tuple[ArchiveEntry, ...]:
    """Select exactly one inventory entry for each of a UTC day's 24 hours."""

    if not isinstance(day, date) or isinstance(day, datetime):
        raise FullDayInputError("day must be a date")
    by_hour: dict[int, ArchiveEntry] = {}
    for entry in entries:
        normalized = _utc_hour(entry.hour, field="archive entry hour")
        if normalized.date() != day:
            continue
        if normalized.hour in by_hour:
            raise FullDayInputError(
                f"duplicate PMXT inventory hour {normalized.hour:02d} for {day}"
            )
        by_hour[normalized.hour] = entry
    missing = [hour for hour in range(24) if hour not in by_hour]
    if missing:
        rendered = ", ".join(f"{hour:02d}" for hour in missing)
        raise FullDayInputError(f"missing PMXT inventory hours for {day}: {rendered}")
    return tuple(by_hour[hour] for hour in range(24))


def _manifest_material(manifest: FullDayManifest) -> dict[str, Any]:
    material = asdict(manifest)
    material.pop("manifest_sha256", None)
    return material


def _validate_relative_object_path(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise FullDayInputError("object_path must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FullDayInputError(
            "object_path must be a canonical relative POSIX path without traversal"
        )


def _validate_locked_object(value: LockedHourlyObject, *, day: str) -> datetime:
    try:
        parsed = datetime.strptime(value.hour, "%Y-%m-%dT%H:00:00Z").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise FullDayInputError("object hour must be a canonical UTC hour") from exc
    if parsed.date().isoformat() != day:
        raise FullDayInputError("object hour is outside the manifest day")
    if not value.source_url.startswith("https://"):
        raise FullDayInputError("source_url must use HTTPS")
    _validate_relative_object_path(value.object_path)
    _require_sha256(value.object_sha256, field="object_sha256")
    _require_sha256(value.static_manifest_sha256, field="static_manifest_sha256")
    if value.inventory_size_bytes is not None and value.inventory_size_bytes <= 0:
        raise FullDayInputError("inventory_size_bytes must be positive when present")
    return parsed


def build_full_day_manifest(
    *,
    day: date,
    entries: Iterable[ArchiveEntry],
    objects: Iterable[HourlyObjectRef],
    inventory_sha256: str,
    canonicalization_version: str,
) -> FullDayManifest:
    """Bind a complete inventory selection to 24 immutable static objects."""

    selected = select_complete_utc_day(entries, day=day)
    _require_sha256(inventory_sha256, field="inventory_sha256")
    if not canonicalization_version:
        raise FullDayInputError("canonicalization_version is required")

    objects_by_hour: dict[int, HourlyObjectRef] = {}
    for item in objects:
        normalized = _utc_hour(item.hour, field="hourly object hour")
        if normalized.date() != day:
            raise FullDayInputError("hourly object is outside the requested day")
        if normalized.hour in objects_by_hour:
            raise FullDayInputError(
                f"duplicate hourly object for hour {normalized.hour:02d}"
            )
        objects_by_hour[normalized.hour] = item
    missing = [hour for hour in range(24) if hour not in objects_by_hour]
    if missing:
        rendered = ", ".join(f"{hour:02d}" for hour in missing)
        raise FullDayInputError(f"missing hourly objects: {rendered}")

    locked: list[LockedHourlyObject] = []
    for entry in selected:
        normalized = _utc_hour(entry.hour, field="archive entry hour")
        item = objects_by_hour[normalized.hour]
        if item.source_url != entry.url:
            raise FullDayInputError(
                f"hour {normalized.hour:02d} source URL differs from inventory"
            )
        locked.append(
            LockedHourlyObject(
                hour=_hour_text(normalized),
                source_url=item.source_url,
                object_path=item.object_path,
                object_sha256=item.object_sha256,
                static_manifest_sha256=item.static_manifest_sha256,
                inventory_size_bytes=entry.size_bytes,
            )
        )

    provisional = FullDayManifest(
        version="pmxt-full-day-manifest-v1",
        day=day.isoformat(),
        inventory_sha256=inventory_sha256,
        canonicalization_version=canonicalization_version,
        objects=tuple(locked),
        manifest_sha256="",
    )
    complete = replace(
        provisional,
        manifest_sha256=_sha256_bytes(_canonical_bytes(_manifest_material(provisional))),
    )
    return validate_full_day_manifest(complete)


def validate_full_day_manifest(manifest: FullDayManifest) -> FullDayManifest:
    """Validate structure, exact 24-hour coverage, order, and self-hash."""

    if manifest.version != "pmxt-full-day-manifest-v1":
        raise FullDayInputError("unsupported full-day manifest version")
    try:
        parsed_day = date.fromisoformat(manifest.day)
    except (TypeError, ValueError) as exc:
        raise FullDayInputError("manifest day must be ISO-8601") from exc
    _require_sha256(manifest.inventory_sha256, field="inventory_sha256")
    _require_sha256(manifest.manifest_sha256, field="manifest_sha256")
    if not manifest.canonicalization_version:
        raise FullDayInputError("canonicalization_version is required")
    if len(manifest.objects) != 24:
        raise FullDayInputError("manifest must contain exactly 24 hourly objects")

    hours = [
        _validate_locked_object(item, day=parsed_day.isoformat())
        for item in manifest.objects
    ]
    expected = [
        datetime(parsed_day.year, parsed_day.month, parsed_day.day, hour, tzinfo=timezone.utc)
        for hour in range(24)
    ]
    if hours != expected:
        raise FullDayInputError("manifest objects must cover ordered UTC hours 00-23")
    if len({item.object_path for item in manifest.objects}) != 24:
        raise FullDayInputError("manifest object paths must be unique")

    expected_hash = _sha256_bytes(_canonical_bytes(_manifest_material(manifest)))
    if manifest.manifest_sha256 != expected_hash:
        raise FullDayInputError("manifest_sha256 does not match manifest content")
    return manifest


def _resolve_locked_path(root: Path, relative: str) -> Path:
    _validate_relative_object_path(relative)
    root_resolved = root.resolve(strict=True)
    if not root_resolved.is_dir():
        raise FullDayInputError("raw root must be a directory")
    current = root_resolved
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FullDayInputError(f"locked object does not exist: {relative}") from exc
        if stat.S_ISLNK(mode):
            raise FullDayInputError(f"locked object path contains a symlink: {relative}")
    if not stat.S_ISREG(current.stat().st_mode):
        raise FullDayInputError(f"locked object is not a regular file: {relative}")
    try:
        current.relative_to(root_resolved)
    except ValueError as exc:
        raise FullDayInputError("locked object escapes the raw root") from exc
    return current


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verified_paths(
    raw_root: str | Path, manifest: FullDayManifest
) -> tuple[Path, ...]:
    root = Path(raw_root)
    paths: list[Path] = []
    for item in manifest.objects:
        path = _resolve_locked_path(root, item.object_path)
        if _sha256_path(path) != item.object_sha256:
            raise FullDayInputError(
                f"locked object SHA-256 mismatch: {item.object_path}"
            )
        paths.append(path)
    return tuple(paths)


def _native_markets(paths: Sequence[Path]) -> tuple[str, ...]:
    connection = duckdb.connect(database=":memory:")
    try:
        total_rows, null_market_rows = connection.execute(
            """
            SELECT
                count(*) AS total_rows,
                count(*) FILTER (WHERE market IS NULL) AS null_market_rows
            FROM read_parquet(?, union_by_name = true)
            """,
            [[str(path.resolve()) for path in paths]],
        ).fetchone()
        if total_rows == 0:
            raise FullDayInputError("PMXT full day contains no source rows")
        if null_market_rows:
            raise FullDayInputError(
                f"PMXT full day contains {null_market_rows} NULL market rows"
            )
        rows = connection.execute(
            """
            SELECT DISTINCT market
            FROM read_parquet(?, union_by_name = true)
            ORDER BY market
            """,
            [[str(path.resolve()) for path in paths]],
        ).fetchall()
    except duckdb.Error as exc:
        raise FullDayInputError(f"failed to enumerate PMXT markets: {exc}") from exc
    finally:
        connection.close()

    markets: list[str] = []
    for (raw_market,) in rows:
        if isinstance(raw_market, memoryview):
            raw_market = raw_market.tobytes()
        if isinstance(raw_market, bytes):
            try:
                market = raw_market.decode("ascii")
            except UnicodeDecodeError as exc:
                raise FullDayInputError("PMXT market identifier is not ASCII") from exc
        elif isinstance(raw_market, str):
            market = raw_market
        else:
            raise FullDayInputError("PMXT market identifier has an invalid type")
        if _MARKET_PATTERN.fullmatch(market) is None:
            raise FullDayInputError(f"invalid PMXT market identifier: {market!r}")
        # Preserve the byte-exact native spelling: the pushed-down Parquet
        # filter is binary equality, so normalizing hex case would silently
        # select zero rows for an uppercase source identifier.
        markets.append(market)
    return tuple(sorted(markets))


def preflight_full_day_inputs(
    raw_root: str | Path, manifest: FullDayManifest
) -> FullDayPreflightReport:
    """Verify all 24 objects and measure their real full-day scale.

    This deliberately does not call the semantic reconstructor.  It creates an
    evidence boundary between a complete, immutable input day and the much
    stronger all-contract reconstruction/comparison claims required by X-01.
    """

    validate_full_day_manifest(manifest)
    paths = _verified_paths(raw_root, manifest)
    if len(paths) != len(manifest.objects):
        raise FullDayInputError("verified object count differs from manifest")

    schema_fingerprints: set[str] = set()
    compressed_bytes = 0
    metadata_rows = 0
    for locked, path in zip(manifest.objects, paths, strict=True):
        expected_hour = datetime.strptime(
            locked.hour, "%Y-%m-%dT%H:00:00Z"
        ).replace(tzinfo=timezone.utc)
        expected_end = expected_hour + timedelta(hours=1)
        try:
            parquet = pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as exc:
            raise FullDayInputError(
                f"locked object is not readable Parquet: {locked.object_path}"
            ) from exc
        schema = parquet.schema_arrow
        timestamp_index = schema.get_field_index("timestamp_received")
        if timestamp_index < 0:
            raise FullDayInputError(
                f"locked object lacks timestamp_received: {locked.object_path}"
            )
        minima: list[datetime] = []
        maxima: list[datetime] = []
        for row_group in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(row_group).column(
                timestamp_index
            ).statistics
            if statistics is None or not statistics.has_min_max:
                raise FullDayInputError(
                    "timestamp_received row-group statistics are required for "
                    f"receive-time partition verification: {locked.object_path}"
                )
            if not isinstance(statistics.min, datetime) or not isinstance(
                statistics.max, datetime
            ):
                raise FullDayInputError(
                    "timestamp_received statistics must be timestamps"
                )
            minima.append(statistics.min)
            maxima.append(statistics.max)
        if not minima or parquet.metadata.num_rows <= 0:
            raise FullDayInputError(
                f"locked object contains no rows: {locked.object_path}"
            )
        observed_min = min(minima).astimezone(timezone.utc)
        observed_max = max(maxima).astimezone(timezone.utc)
        if not (
            expected_hour <= observed_min < expected_end
            and expected_hour <= observed_max < expected_end
        ):
            raise FullDayInputError(
                "locked object violates its receive-time partition: "
                f"{locked.object_path}"
            )
        schema_fingerprints.add(_schema_fingerprint(schema))
        compressed_bytes += path.stat().st_size
        metadata_rows += parquet.metadata.num_rows
    if len(schema_fingerprints) != 1:
        raise FullDayInputError("PMXT full day contains schema drift")

    connection = duckdb.connect(database=":memory:")
    try:
        resolved = [str(path.resolve()) for path in paths]
        summary = connection.execute(
            """
            SELECT
                count(*) AS row_count,
                count(*) FILTER (
                    WHERE timestamp_received IS NULL
                       OR timestamp IS NULL
                       OR market IS NULL
                       OR asset_id IS NULL
                       OR event_type IS NULL
                ) AS invalid_required_rows,
                count(DISTINCT market) AS market_count,
                count(DISTINCT asset_id) AS asset_count,
                epoch_ms(min(timestamp_received)) AS timestamp_received_min_ms,
                epoch_ms(max(timestamp_received)) AS timestamp_received_max_ms
            FROM read_parquet(?, union_by_name = true)
            """,
            [resolved],
        ).fetchone()
        event_type_rows = connection.execute(
            """
            SELECT event_type, count(*) AS event_count
            FROM read_parquet(?, union_by_name = true)
            GROUP BY event_type
            ORDER BY event_type
            """,
            [resolved],
        ).fetchall()
    except duckdb.Error as exc:
        raise FullDayInputError(f"PMXT full-day preflight failed: {exc}") from exc
    finally:
        connection.close()
    if summary is None:
        raise FullDayInputError("PMXT full-day preflight produced no summary")
    (
        row_count,
        invalid_required_rows,
        market_count,
        asset_count,
        received_min,
        received_max,
    ) = summary
    if row_count != metadata_rows:
        raise FullDayInputError(
            "Parquet metadata row count differs from the full-day scan"
        )
    if invalid_required_rows:
        raise FullDayInputError(
            f"PMXT full day contains {invalid_required_rows} NULL required rows"
        )
    if not market_count or not asset_count:
        raise FullDayInputError("PMXT full day contains no markets or assets")
    if isinstance(received_min, bool) or not isinstance(received_min, int):
        raise FullDayInputError("PMXT full day has invalid receive timestamps")
    if isinstance(received_max, bool) or not isinstance(received_max, int):
        raise FullDayInputError("PMXT full day has invalid receive timestamps")
    received_min_at = datetime.fromtimestamp(
        received_min / 1_000, tz=timezone.utc
    )
    received_max_at = datetime.fromtimestamp(
        received_max / 1_000, tz=timezone.utc
    )
    event_type_counts: dict[str, int] = {}
    for event_type, event_count in event_type_rows:
        if not isinstance(event_type, str) or not event_type:
            raise FullDayInputError("PMXT full day has an invalid event type")
        event_type_counts[event_type] = int(event_count)
    if sum(event_type_counts.values()) != row_count:
        raise FullDayInputError("PMXT event-type counts do not cover all rows")

    provisional = FullDayPreflightReport(
        version="pmxt-full-day-preflight-v1",
        day=manifest.day,
        input_manifest_sha256=manifest.manifest_sha256,
        object_count=len(paths),
        compressed_bytes=compressed_bytes,
        row_count=int(row_count),
        market_count=int(market_count),
        asset_count=int(asset_count),
        event_type_counts=dict(sorted(event_type_counts.items())),
        timestamp_received_min=_utc_text(
            received_min_at, field="timestamp_received_min"
        ),
        timestamp_received_max=_utc_text(
            received_max_at, field="timestamp_received_max"
        ),
        schema_fingerprint=next(iter(schema_fingerprints)),
        reconstruction_executed=False,
        x01_formal_gate_passed=False,
        open_gates=(
            "all_contract_full_day_semantic_reconstruction",
            "independent_price_and_size_comparison",
        ),
        queue_fill_reconstructed=False,
        report_sha256="",
    )
    material = asdict(provisional)
    material.pop("report_sha256")
    report = replace(
        provisional,
        report_sha256=_sha256_bytes(_canonical_bytes(material)),
    )
    if _verified_paths(raw_root, manifest) != paths:
        raise FullDayInputError("locked object paths changed during preflight")
    return report


def run_full_day_reconstruction(
    raw_root: str | Path, manifest: FullDayManifest
) -> FullDayReconstructionReport:
    """Reconstruct every market while keeping the independent-data gate open."""

    validate_full_day_manifest(manifest)
    paths = _verified_paths(raw_root, manifest)
    markets = _native_markets(paths)
    results: list[MarketReconstruction] = []
    try:
        for market in markets:
            reconstructed = reconstruct(read_parquet_events(paths, market=market))
            results.append(
                MarketReconstruction(
                    native_market_id=market,
                    stream_sha256=reconstructed.stream_sha256,
                    counts=dict(sorted(reconstructed.counts.items())),
                    quality_flags=reconstructed.quality_flags,
                    queue_fill_reconstructed=reconstructed.queue_fill_reconstructed,
                )
            )
    except (ArchiveIntegrityError, PMXTValidationError) as exc:
        raise FullDayInputError(f"PMXT full-day reconstruction failed: {exc}") from exc

    # Detect replacement or mutation that happened anywhere during enumeration/read.
    post_read_paths = _verified_paths(raw_root, manifest)
    if post_read_paths != paths:
        raise FullDayInputError("locked object paths changed during reconstruction")

    stream_material = [
        {
            "native_market_id": result.native_market_id,
            "stream_sha256": result.stream_sha256,
            "counts": result.counts,
            "quality_flags": list(result.quality_flags),
        }
        for result in results
    ]
    return FullDayReconstructionReport(
        version="pmxt-full-day-report-v1",
        day=manifest.day,
        input_manifest_sha256=manifest.manifest_sha256,
        market_count=len(results),
        input_event_count=sum(item.counts["input_events"] for item in results),
        semantic_event_count=sum(item.counts["semantic_events"] for item in results),
        market_results=tuple(results),
        day_stream_sha256=_sha256_bytes(_canonical_bytes(stream_material)),
        independent_comparison_performed=False,
        x01_formal_gate_passed=False,
        open_gate="independent_price_and_size_comparison",
        queue_fill_reconstructed=False,
    )
