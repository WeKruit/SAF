"""PMXT v2 archive inventory, immutable preservation, and Parquet audit."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import duckdb
import httpx
import pyarrow.parquet as pq


PMXT_V2_INDEX_URL = "https://archive.pmxt.dev/Polymarket/v2"
_FILENAME_PATTERN = re.compile(
    r"polymarket_orderbook_(?P<hour>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2})\.parquet"
)
_DOWNLOAD_URL_PATTERN = re.compile(
    r"https://[^\"'<>\\\s]+/polymarket_orderbook_"
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}\.parquet"
)
_SIZE_PATTERN = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>KB|MB|GB|B)\b")
_SIZE_MULTIPLIERS = {
    "B": Decimal(1),
    # The PMXT listing labels these units KB/MB/GB, but exact downloaded
    # objects confirm its formatter uses binary powers.
    "KB": Decimal(1_024),
    "MB": Decimal(1_048_576),
    "GB": Decimal(1_073_741_824),
}


EXPECTED_PMXT_V2_SCHEMA = {
    "timestamp_received": "timestamp[ms, tz=UTC]",
    "timestamp": "timestamp[ms, tz=UTC]",
    "market": "fixed_size_binary[66]",
    "event_type": "string",
    "asset_id": "string",
    "bids": "string",
    "asks": "string",
    "price": "decimal128(9, 4)",
    "size": "decimal128(18, 6)",
    "side": "string",
    "best_bid": "decimal128(9, 4)",
    "best_ask": "decimal128(9, 4)",
    "fee_rate_bps": "uint16",
    "transaction_hash": "string",
    "old_tick_size": "decimal128(9, 4)",
    "new_tick_size": "decimal128(9, 4)",
}


class ArchiveError(RuntimeError):
    """Base class for failures that prevent an evidence-backed PMXT audit."""


class ArchiveIntegrityError(ArchiveError):
    """Raised when immutable bytes do not match their content address."""


class ArchiveNetworkError(ArchiveError):
    """Raised when the public archive cannot be fetched exactly as requested."""


@dataclass(frozen=True)
class ArchiveEntry:
    filename: str
    url: str
    hour: datetime
    size_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "url": self.url,
            "hour": _utc_iso(self.hour),
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ArchivedFile:
    source_filename: str
    source_url: str | None
    sha256: str
    byte_size: int
    object_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "source_filename": self.source_filename,
            "source_url": self.source_url,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "object_path": str(self.object_path),
        }


@dataclass(frozen=True)
class ParquetAudit:
    path: Path
    original_sha256: str
    byte_size: int
    row_count: int
    row_group_count: int
    column_count: int
    schema: dict[str, str]
    nullable: dict[str, bool]
    null_counts: dict[str, int]
    schema_issues: tuple[str, ...]
    event_type_counts: dict[str, int]
    receive_minus_source_ms: dict[str, int | float | None]
    timestamp_range: dict[str, str | None]

    def to_dict(self) -> dict[str, object]:
        material = asdict(self)
        material["path"] = str(self.path)
        material["schema_issues"] = list(self.schema_issues)
        return material


def _utc_iso(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _hour_from_filename(filename: str) -> datetime:
    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ArchiveError(f"not a PMXT v2 hourly filename: {filename}")
    return datetime.strptime(match.group("hour"), "%Y-%m-%dT%H").replace(
        tzinfo=timezone.utc
    )


def _size_bytes(value: str, unit: str) -> int:
    return int(Decimal(value) * _SIZE_MULTIPLIERS[unit])


def parse_inventory_page(html: str, *, index_url: str) -> tuple[ArchiveEntry, ...]:
    """Parse one PMXT archive page without double-counting JSON-LD URLs."""

    entries: dict[str, ArchiveEntry] = {}
    anchor_pattern = re.compile(
        r'<a\s+[^>]*href=["\'](?P<url>'
        + _DOWNLOAD_URL_PATTERN.pattern
        + r')["\'][^>]*>.*?</a>(?P<tail>[^\r\n]{0,220})',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in anchor_pattern.finditer(html):
        url = match.group("url")
        filename = Path(urlparse(url).path).name
        size_match = _SIZE_PATTERN.search(re.sub(r"<[^>]+>", " ", match.group("tail")))
        size = (
            _size_bytes(size_match.group("value"), size_match.group("unit").upper())
            if size_match is not None
            else None
        )
        entries[filename] = ArchiveEntry(
            filename=filename,
            url=url,
            hour=_hour_from_filename(filename),
            size_bytes=size,
        )

    for url in _DOWNLOAD_URL_PATTERN.findall(html):
        filename = Path(urlparse(url).path).name
        entries.setdefault(
            filename,
            ArchiveEntry(
                filename=filename,
                url=url,
                hour=_hour_from_filename(filename),
                size_bytes=None,
            ),
        )
    if not entries:
        raise ArchiveError(f"no PMXT v2 Parquet objects found at {index_url}")
    return tuple(
        sorted(entries.values(), key=lambda entry: (entry.hour, entry.filename))
    )


def _page_url(index_url: str, page: int) -> str:
    parsed = urlparse(index_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if page == 1:
        query.pop("page", None)
    else:
        query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_archive_inventory(
    *,
    index_url: str = PMXT_V2_INDEX_URL,
    timeout_seconds: float = 30.0,
    max_pages: int | None = None,
) -> tuple[ArchiveEntry, ...]:
    """Fetch the real public inventory, following its numbered pages."""

    if max_pages is not None and max_pages <= 0:
        raise ArchiveError("max_pages must be positive")
    parsed_url = urlparse(index_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ArchiveError("PMXT inventory URL must be HTTPS")

    all_entries: dict[str, ArchiveEntry] = {}
    try:

        def fetch_page(page: int) -> str:
            # The archive varies on Next.js router headers.  A fresh, ordinary
            # document request per page avoids connection-local RSC responses.
            with httpx.Client(follow_redirects=True, timeout=timeout_seconds) as client:
                response = client.get(_page_url(index_url, page))
                response.raise_for_status()
                return response.text

        first_html = fetch_page(1)
        page_numbers = [
            int(number) for number in re.findall(r"[?&]page=([0-9]+)", first_html)
        ]
        visible_text = re.sub(r"<!--.*?-->|<[^>]+>", " ", first_html, flags=re.DOTALL)
        advertised_totals = [
            int(number)
            for number in re.findall(
                r"\bPage\s+[0-9]+\s+of\s+([0-9]+)\b",
                visible_text,
                flags=re.IGNORECASE,
            )
        ]
        last_page = max([1, *page_numbers, *advertised_totals])
        if last_page > 10_000:
            raise ArchiveNetworkError(
                f"PMXT inventory advertised an implausible {last_page} pages"
            )
        if max_pages is not None:
            last_page = min(last_page, max_pages)
        for page in range(1, last_page + 1):
            html = first_html if page == 1 else fetch_page(page)
            for entry in parse_inventory_page(
                html, index_url=_page_url(index_url, page)
            ):
                previous = all_entries.get(entry.filename)
                if previous is None or (
                    previous.size_bytes is None and entry.size_bytes is not None
                ):
                    all_entries[entry.filename] = entry
    except (httpx.HTTPError, OSError) as exc:
        raise ArchiveNetworkError(
            f"failed to fetch PMXT inventory from {index_url}: {exc}"
        ) from exc
    if not all_entries:
        raise ArchiveNetworkError(f"PMXT inventory at {index_url} was empty")
    return tuple(
        sorted(all_entries.values(), key=lambda entry: (entry.hour, entry.filename))
    )


def summarize_inventory(
    entries: tuple[ArchiveEntry, ...] | list[ArchiveEntry],
) -> dict[str, object]:
    """Quantify hourly coverage and byte-cost observations."""

    if not entries:
        raise ArchiveError("cannot summarize an empty PMXT inventory")
    ordered = sorted(entries, key=lambda entry: (entry.hour, entry.filename))
    hours = sorted({entry.hour for entry in ordered})
    expected: list[datetime] = []
    cursor = hours[0]
    while cursor <= hours[-1]:
        expected.append(cursor)
        cursor += timedelta(hours=1)
    missing = sorted(set(expected) - set(hours))
    sizes = [entry.size_bytes for entry in ordered if entry.size_bytes is not None]
    mean_size = int(sum(sizes) / len(sizes)) if sizes else None
    per_day: dict[str, int] = {}
    for hour in hours:
        day = hour.date().isoformat()
        per_day[day] = per_day.get(day, 0) + 1
    return {
        "source": "PMXT Polymarket v2 hourly Parquet index",
        "reported_size_unit_basis": "binary (KiB/MiB/GiB despite KB/MB/GB labels)",
        "first_hour": _utc_iso(hours[0]),
        "last_hour": _utc_iso(hours[-1]),
        "file_count": len(ordered),
        "covered_hours": len(hours),
        "expected_hours_inclusive": len(expected),
        "coverage_ratio": len(hours) / len(expected),
        "missing_hours": [_utc_iso(hour) for hour in missing],
        "hours_per_day": dict(sorted(per_day.items())),
        "files_with_reported_sizes": len(sizes),
        "reported_total_bytes": sum(sizes),
        "min_file_bytes": min(sizes) if sizes else None,
        "mean_file_bytes": mean_size,
        "max_file_bytes": max(sizes) if sizes else None,
        "projected_mean_gb_per_day": (
            round(mean_size * 24 / 1_000_000_000, 6) if mean_size is not None else None
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def preserve_parquet(
    source: str | Path,
    *,
    raw_root: str | Path,
    source_url: str | None = None,
    source_filename: str | None = None,
) -> ArchivedFile:
    """Copy exact bytes once into a SHA-256-addressed object path.

    Existing objects are verified and returned.  A mismatching object is never
    repaired in place because that would violate append-only raw retention.
    """

    source_path = Path(source)
    if not source_path.is_file():
        raise ArchiveIntegrityError(f"source Parquet does not exist: {source_path}")
    digest = sha256_file(source_path)
    digest_hex = digest.removeprefix("sha256:")
    object_directory = Path(raw_root) / "pmxt-v2" / "sha256" / digest_hex[:2]
    object_directory.mkdir(parents=True, exist_ok=True)
    target = object_directory / f"{digest_hex}.parquet"

    if target.exists():
        if not target.is_file() or sha256_file(target) != digest:
            raise ArchiveIntegrityError(
                f"content address {target} is occupied by different bytes; refusing to overwrite"
            )
    else:
        try:
            with target.open("xb") as destination, source_path.open("rb") as input_file:
                for block in iter(lambda: input_file.read(1024 * 1024), b""):
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
        except FileExistsError:
            if not target.is_file() or sha256_file(target) != digest:
                raise ArchiveIntegrityError(
                    f"content address {target} changed concurrently; refusing to overwrite"
                )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        if sha256_file(target) != digest:
            target.unlink(missing_ok=True)
            raise ArchiveIntegrityError(
                "archived Parquet failed post-write SHA-256 verification"
            )

    return ArchivedFile(
        source_filename=source_filename or source_path.name,
        source_url=source_url,
        sha256=digest,
        byte_size=source_path.stat().st_size,
        object_path=target,
    )


def download_and_preserve(
    url: str,
    *,
    raw_root: str | Path,
    max_bytes: int = 600_000_000,
    timeout_seconds: float = 120.0,
) -> ArchivedFile:
    """Download one real HTTPS object with a byte ceiling, then preserve it."""

    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or _FILENAME_PATTERN.fullmatch(filename) is None
    ):
        raise ArchiveNetworkError(
            "sample URL must be an HTTPS PMXT v2 hourly Parquet URL"
        )
    if max_bytes <= 0:
        raise ArchiveNetworkError("max_bytes must be positive")

    raw_path = Path(raw_root)
    raw_path.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pmxt-download-", suffix=".partial", dir=raw_path
    )
    temporary_path = Path(temporary_name)
    received = 0
    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=timeout_seconds
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise ArchiveNetworkError(
                        f"PMXT object is {content_length} bytes, above max_bytes={max_bytes}"
                    )
                for block in response.iter_bytes(chunk_size=1024 * 1024):
                    received += len(block)
                    if received > max_bytes:
                        raise ArchiveNetworkError(
                            f"PMXT object exceeded max_bytes={max_bytes} while streaming"
                        )
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
        return preserve_parquet(
            temporary_path,
            raw_root=raw_path,
            source_url=url,
            source_filename=filename,
        )
    except ArchiveError:
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise ArchiveNetworkError(
            f"failed to download PMXT sample {url}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _quoted_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _number(value: object) -> int | float | None:
    if value is None:
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _datetime_from_epoch_ms(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArchiveIntegrityError("DuckDB returned a non-integer timestamp epoch")
    return datetime.fromtimestamp(value / 1_000, tz=timezone.utc)


def _timestamp_text_from_epoch_ms(value: object) -> str | None:
    timestamp = _datetime_from_epoch_ms(value)
    return _utc_iso(timestamp) if timestamp is not None else None


def audit_parquet(path: str | Path) -> ParquetAudit:
    """Audit original PMXT schema, nullability, event mix, and clock deltas."""

    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise ArchiveIntegrityError(f"Parquet sample does not exist: {parquet_path}")
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as exc:
        raise ArchiveIntegrityError(
            f"invalid Parquet sample {parquet_path}: {exc}"
        ) from exc

    arrow_schema = parquet_file.schema_arrow
    schema = {field.name: str(field.type) for field in arrow_schema}
    nullable = {field.name: field.nullable for field in arrow_schema}
    issues: list[str] = []
    for name, expected_type in EXPECTED_PMXT_V2_SCHEMA.items():
        actual_type = schema.get(name)
        if actual_type is None:
            issues.append(f"missing column {name}")
        elif actual_type != expected_type:
            issues.append(f"{name}: expected {expected_type}, observed {actual_type}")
    for name in sorted(set(schema) - set(EXPECTED_PMXT_V2_SCHEMA)):
        issues.append(f"unexpected column {name}")

    path_parameter = str(parquet_path.resolve())
    connection = duckdb.connect(database=":memory:")
    try:
        null_select = ", ".join(
            f"count(*) FILTER (WHERE {_quoted_identifier(name)} IS NULL)"
            for name in schema
        )
        null_row = connection.execute(
            f"SELECT {null_select} FROM read_parquet(?)", [path_parameter]
        ).fetchone()
        assert null_row is not None
        null_counts = {name: int(value) for name, value in zip(schema, null_row)}
        event_rows = connection.execute(
            """
            SELECT event_type, count(*) AS event_count
            FROM read_parquet(?)
            GROUP BY event_type
            ORDER BY event_type ASC NULLS LAST
            """,
            [path_parameter],
        ).fetchall()
        event_type_counts = {
            "<NULL>" if event_type is None else str(event_type): int(count)
            for event_type, count in event_rows
        }
        lag_row = connection.execute(
            """
            WITH lags AS (
                SELECT date_diff('millisecond', timestamp, timestamp_received) AS lag_ms
                FROM read_parquet(?)
                WHERE timestamp IS NOT NULL AND timestamp_received IS NOT NULL
            )
            SELECT
                count(*),
                count(*) FILTER (WHERE lag_ms < 0),
                min(lag_ms),
                quantile_cont(lag_ms, 0.5),
                quantile_cont(lag_ms, 0.95),
                max(lag_ms)
            FROM lags
            """,
            [path_parameter],
        ).fetchone()
        assert lag_row is not None
        receive_minus_source_ms = {
            "count": int(lag_row[0]),
            "negative_count": int(lag_row[1]),
            "min": _number(lag_row[2]),
            "median": _number(lag_row[3]),
            "p95": _number(lag_row[4]),
            "max": _number(lag_row[5]),
        }
        range_row = connection.execute(
            """
            SELECT
                epoch_ms(min(timestamp_received)),
                epoch_ms(max(timestamp_received)),
                epoch_ms(min(timestamp)),
                epoch_ms(max(timestamp))
            FROM read_parquet(?)
            """,
            [path_parameter],
        ).fetchone()
        assert range_row is not None
        timestamp_range = {
            "timestamp_received_min": _timestamp_text_from_epoch_ms(range_row[0]),
            "timestamp_received_max": _timestamp_text_from_epoch_ms(range_row[1]),
            "timestamp_min": _timestamp_text_from_epoch_ms(range_row[2]),
            "timestamp_max": _timestamp_text_from_epoch_ms(range_row[3]),
        }
    except duckdb.Error as exc:
        raise ArchiveIntegrityError(f"failed to audit Parquet sample: {exc}") from exc
    finally:
        connection.close()

    metadata = parquet_file.metadata
    return ParquetAudit(
        path=parquet_path,
        original_sha256=sha256_file(parquet_path),
        byte_size=parquet_path.stat().st_size,
        row_count=metadata.num_rows,
        row_group_count=metadata.num_row_groups,
        column_count=len(schema),
        schema=schema,
        nullable=nullable,
        null_counts=null_counts,
        schema_issues=tuple(issues),
        event_type_counts=event_type_counts,
        receive_minus_source_ms=receive_minus_source_ms,
        timestamp_range=timestamp_range,
    )


def read_parquet_events(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    market: str | None = None,
) -> list[dict[str, Any]]:
    """Read Parquet with an optional pushed-down market scope and explicit order."""

    if not paths:
        raise ArchiveIntegrityError("at least one Parquet path is required")
    resolved = [str(Path(path).resolve()) for path in paths]
    missing = [path for path in resolved if not Path(path).is_file()]
    if missing:
        raise ArchiveIntegrityError(f"Parquet path does not exist: {missing[0]}")
    if market is not None and re.fullmatch(r"0x[0-9a-fA-F]{64}", market) is None:
        raise ArchiveIntegrityError("market must be a 0x-prefixed 32-byte identifier")

    where_clause = "WHERE market = ?" if market is not None else ""
    parameters: list[object] = [resolved]
    if market is not None:
        parameters.append(market.encode("ascii"))

    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute(
            f"""
            SELECT
                epoch_ms(timestamp_received) AS _timestamp_received_epoch_ms,
                epoch_ms(timestamp) AS _timestamp_epoch_ms,
                * EXCLUDE (timestamp_received, timestamp)
            FROM read_parquet(?, union_by_name = true)
            {where_clause}
            ORDER BY
                timestamp_received ASC NULLS LAST,
                timestamp ASC NULLS LAST,
                market ASC NULLS LAST,
                asset_id ASC NULLS LAST
            """,
            parameters,
        )
        columns = [description[0] for description in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]
    except duckdb.Error as exc:
        raise ArchiveIntegrityError(
            f"failed to read PMXT Parquet events: {exc}"
        ) from exc
    finally:
        connection.close()

    for event in events:
        received_epoch_ms = event.pop("_timestamp_received_epoch_ms")
        source_epoch_ms = event.pop("_timestamp_epoch_ms")
        event["timestamp_received"] = _datetime_from_epoch_ms(received_epoch_ms)
        event["timestamp"] = _datetime_from_epoch_ms(source_epoch_ms)

    from prediction_market.pmxt.reconstructor import canonical_event_sort_key

    return sorted(events, key=canonical_event_sort_key)
