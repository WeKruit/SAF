"""Byte-exact nflverse play-by-play acquisition for preregistered X-11 work."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prediction_market.static_store import StaticObjectRecord, preserve_static_object


NFLVERSE_RELEASE_ID = 58152862
NFLVERSE_RELEASE_VERSION = "github-release-58152862-20260212T102526Z"
NFLVERSE_YEAR_ASSET_IDS = {
    2015: 250646456,
    2016: 250647177,
    2017: 250648728,
    2018: 250648497,
    2019: 278181049,
    2020: 250664511,
    2021: 337880457,
    2022: 354728862,
    2023: 354728689,
    2024: 289147973,
    2025: 354718810,
}
NFLVERSE_REQUIRED_COLUMNS = frozenset(
    {
        "play_id",
        "game_id",
        "season",
        "season_type",
        "game_date",
        "home_team",
        "away_team",
        "posteam",
        "score_differential",
        "game_seconds_remaining",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
        "spread_line",
        "home_wp",
        "fixed_drive",
        "fixed_drive_result",
        "home_score",
        "away_score",
    }
)


class NFLVerseSourceError(ValueError):
    """An nflverse partition cannot be accepted as an X-11 source object."""


@dataclass(frozen=True, slots=True)
class NFLVersePartitionAudit:
    year: int
    row_count: int
    game_count: int
    season_types: tuple[str, ...]
    column_count: int
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class PreservedNFLVersePartition:
    record: StaticObjectRecord
    audit: NFLVersePartitionAudit


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _schema_fingerprint(schema: pa.Schema) -> str:
    material = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _validate_year(year: object) -> int:
    if type(year) is not int or year not in NFLVERSE_YEAR_ASSET_IDS:
        raise NFLVerseSourceError("X-11 nflverse year must be in 2015-2025")
    return year


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise NFLVerseSourceError("fetched_at must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def nflverse_partition_url(year: int) -> str:
    _validate_year(year)
    return (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"pbp/play_by_play_{year}.parquet"
    )


def inspect_nflverse_partition(
    object_bytes: bytes, *, expected_year: int
) -> NFLVersePartitionAudit:
    """Audit required native fields without normalizing the upstream bytes."""

    year = _validate_year(expected_year)
    if type(object_bytes) is not bytes or not object_bytes:
        raise NFLVerseSourceError("nflverse object must contain exact nonempty bytes")
    try:
        parquet = pq.ParquetFile(BytesIO(object_bytes))
    except (pa.ArrowException, OSError) as exc:
        raise NFLVerseSourceError("nflverse object is not readable Parquet") from exc
    schema = parquet.schema_arrow
    missing = sorted(NFLVERSE_REQUIRED_COLUMNS - set(schema.names))
    if missing:
        raise NFLVerseSourceError(
            "nflverse partition is missing required columns: " + ", ".join(missing)
        )
    try:
        identity = parquet.read(columns=["game_id", "season", "season_type"])
    except pa.ArrowException as exc:
        raise NFLVerseSourceError("nflverse identity columns cannot be read") from exc
    if identity.num_rows == 0:
        raise NFLVerseSourceError("nflverse partition must not be empty")
    game_ids = identity.column("game_id")
    seasons = identity.column("season")
    season_types = identity.column("season_type")
    if game_ids.null_count or seasons.null_count or season_types.null_count:
        raise NFLVerseSourceError("nflverse identity columns must not contain nulls")
    observed_years = {int(value.as_py()) for value in pc.unique(seasons)}
    if observed_years != {year}:
        raise NFLVerseSourceError(
            f"nflverse partition season mismatch: expected {year}, got {sorted(observed_years)}"
        )
    observed_types = tuple(
        sorted(str(value.as_py()) for value in pc.unique(season_types))
    )
    if not observed_types or not set(observed_types) <= {"REG", "POST"}:
        raise NFLVerseSourceError(
            "nflverse X-11 partition may contain REG and POST only"
        )
    games = {str(value.as_py()) for value in pc.unique(game_ids)}
    if "" in games:
        raise NFLVerseSourceError("nflverse game_id must be nonempty")
    return NFLVersePartitionAudit(
        year=year,
        row_count=parquet.metadata.num_rows,
        game_count=len(games),
        season_types=observed_types,
        column_count=len(schema),
        schema_fingerprint=_schema_fingerprint(schema),
        object_sha256="sha256:" + hashlib.sha256(object_bytes).hexdigest(),
    )


def _download_partition(
    year: int,
    *,
    client: httpx.Client,
    max_bytes: int,
) -> tuple[bytes, httpx.Headers]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise NFLVerseSourceError("max_bytes must be a positive integer")
    url = nflverse_partition_url(year)
    try:
        with client.stream(
            "GET",
            url,
            headers={"Accept": "application/vnd.apache.parquet"},
        ) as response:
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as exc:
                    raise NFLVerseSourceError(
                        "nflverse Content-Length is invalid"
                    ) from exc
                if declared_bytes > max_bytes:
                    raise NFLVerseSourceError("nflverse partition exceeds max_bytes")
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise NFLVerseSourceError("nflverse partition exceeds max_bytes")
                chunks.append(chunk)
            if declared is not None and received != declared_bytes:
                raise NFLVerseSourceError(
                    "nflverse Content-Length does not match received bytes"
                )
            payload = b"".join(chunks)
            headers = httpx.Headers(response.headers)
    except httpx.HTTPError as exc:
        raise NFLVerseSourceError(f"nflverse download failed: {exc}") from exc
    if not payload:
        raise NFLVerseSourceError("nflverse download returned an empty object")
    return payload, headers


def fetch_and_preserve_nflverse_year(
    year: int,
    *,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 100_000_000,
) -> PreservedNFLVersePartition:
    """Fetch, audit, and commit one exact release asset plus its manifest."""

    year = _validate_year(year)
    fetched_at_text = _utc_text(fetched_at)
    owns_client = client is None
    active_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    try:
        payload, headers = _download_partition(
            year,
            client=active_client,
            max_bytes=max_bytes,
        )
    finally:
        if owns_client:
            active_client.close()
    audit = inspect_nflverse_partition(payload, expected_year=year)
    partition = f"season-{year}"
    record = preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="nflverse",
        dataset="DS-NFLVERSE",
        version=NFLVERSE_RELEASE_VERSION,
        partition=partition,
        extension="parquet",
        source_url=nflverse_partition_url(year),
        source_request={
            "method": "GET",
            "headers": {"Accept": "application/vnd.apache.parquet"},
        },
        source_cursor=(
            f"github_release_id:{NFLVERSE_RELEASE_ID};"
            f"asset_id:{NFLVERSE_YEAR_ASSET_IDS[year]}"
        ),
        fetched_at=fetched_at_text,
        coverage=f"season={year};season_type=REG,POST",
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/octet-stream").split(
            ";", 1
        )[0],
        schema_fingerprint=audit.schema_fingerprint,
        license_ref="I-018",
        license_status="approved",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )
    if record.manifest.object_sha256 != audit.object_sha256:
        raise NFLVerseSourceError("preserved object hash differs from audited bytes")
    return PreservedNFLVersePartition(record=record, audit=audit)


__all__ = [
    "NFLVERSE_RELEASE_ID",
    "NFLVERSE_RELEASE_VERSION",
    "NFLVERSE_YEAR_ASSET_IDS",
    "NFLVersePartitionAudit",
    "NFLVerseSourceError",
    "PreservedNFLVersePartition",
    "fetch_and_preserve_nflverse_year",
    "inspect_nflverse_partition",
    "nflverse_partition_url",
]
