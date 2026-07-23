"""Bounded MLB and Formula 1 source inventories; no model training."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

import httpx

from prediction_market.static_store import StaticObjectRecord, preserve_static_object


RETROSHEET_2025_URL = "https://www.retrosheet.org/events/2025eve.zip"
JOLPICA_API_PREFIX = "https://api.jolpi.ca/ergast/f1/"
_RETROSHEET_EVENT_FILE_RE = re.compile(r"^2025[A-Z0-9]{3}\.EV[ANFR]$")
_RETROSHEET_GAME_ID_RE = re.compile(r"^[A-Z0-9]{3}2025[0-1][0-9][0-3][0-9][0-9]$")
JolpicaEndpointKind = Literal["results", "laps", "pitstops"]


class SportSourceInventoryError(ValueError):
    """A bounded sports source object cannot be inventoried safely."""


@dataclass(frozen=True, slots=True)
class Retrosheet2025Audit:
    season: int
    event_file_count: int
    game_count: int
    play_record_count: int
    record_type_counts: Mapping[str, int]
    first_game_id: str
    last_game_id: str
    schema_fingerprint: str
    object_sha256: str
    byte_length: int


@dataclass(frozen=True, slots=True)
class PreservedRetrosheet2025:
    record: StaticObjectRecord
    audit: Retrosheet2025Audit


@dataclass(frozen=True, slots=True)
class JolpicaAudit:
    endpoint_kind: JolpicaEndpointKind
    race_count: int
    native_record_count: int
    seasons: tuple[int, ...]
    rounds: tuple[tuple[int, int], ...]
    schema_fingerprint: str
    object_sha256: str
    byte_length: int


@dataclass(frozen=True, slots=True)
class PreservedJolpica:
    record: StaticObjectRecord
    audit: JolpicaAudit


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _schema_fingerprint(value: Any) -> str:
    observed: dict[str, set[str]] = defaultdict(set)

    def kind(child: Any) -> str:
        if child is None:
            return "null"
        if type(child) is bool:
            return "boolean"
        if type(child) is int:
            return "integer"
        if type(child) is float:
            return "number"
        if type(child) is str:
            return "string"
        if type(child) is list:
            return "array"
        if type(child) is dict:
            return "object"
        raise SportSourceInventoryError("unsupported JSON value type")

    def visit(child: Any, path: str) -> None:
        observed[path].add(kind(child))
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    material = {path: sorted(types) for path, types in sorted(observed.items())}
    return "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SportSourceInventoryError("fetched_at must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _strict_json(payload: bytes) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SportSourceInventoryError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                SportSourceInventoryError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, SportSourceInventoryError) as exc:
        raise SportSourceInventoryError(
            "Jolpica response must be strict UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise SportSourceInventoryError("Jolpica response must be a JSON object")
    return value


def inspect_retrosheet_2025(
    object_bytes: bytes,
    *,
    max_files: int = 512,
    max_uncompressed_bytes: int = 100_000_000,
) -> Retrosheet2025Audit:
    """Inventory exact 2025 event ZIP bytes without deriving model features."""

    if type(object_bytes) is not bytes or not object_bytes:
        raise SportSourceInventoryError("Retrosheet object must be nonempty bytes")
    if type(max_files) is not int or max_files <= 0:
        raise SportSourceInventoryError("max_files must be positive")
    if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes <= 0:
        raise SportSourceInventoryError("max_uncompressed_bytes must be positive")
    try:
        archive = zipfile.ZipFile(io.BytesIO(object_bytes))
        infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise SportSourceInventoryError("Retrosheet object is not a valid ZIP") from exc
    with archive:
        if not infos or len(infos) > max_files:
            raise SportSourceInventoryError("Retrosheet ZIP file count is invalid")
        names: set[str] = set()
        total_uncompressed = 0
        for info in infos:
            name = info.filename
            if (
                not name
                or name in names
                or info.is_dir()
                or "/" in name
                or "\\" in name
                or name in {".", ".."}
                or info.flag_bits & 0x1
            ):
                raise SportSourceInventoryError(
                    "Retrosheet ZIP contains an unsafe or duplicate member"
                )
            names.add(name)
            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise SportSourceInventoryError(
                    "Retrosheet ZIP exceeds the uncompressed byte limit"
                )

        event_infos = sorted(
            (info for info in infos if _RETROSHEET_EVENT_FILE_RE.fullmatch(info.filename)),
            key=lambda info: info.filename,
        )
        if not event_infos or "TEAM2025" not in names:
            raise SportSourceInventoryError(
                "Retrosheet ZIP lacks 2025 event files or TEAM2025"
            )
        counts: Counter[str] = Counter()
        field_counts: dict[str, set[int]] = defaultdict(set)
        game_ids: list[str] = []
        for info in event_infos:
            try:
                text = archive.read(info).decode("ascii")
            except (KeyError, UnicodeError, RuntimeError) as exc:
                raise SportSourceInventoryError(
                    f"Retrosheet member cannot be read as ASCII: {info.filename}"
                ) from exc
            for row in csv.reader(io.StringIO(text, newline="")):
                if not row or not row[0]:
                    raise SportSourceInventoryError(
                        f"Retrosheet member has an empty record: {info.filename}"
                    )
                record_type = row[0]
                counts[record_type] += 1
                field_counts[record_type].add(len(row) - 1)
                if record_type == "id":
                    if len(row) != 2 or _RETROSHEET_GAME_ID_RE.fullmatch(row[1]) is None:
                        raise SportSourceInventoryError(
                            "Retrosheet 2025 game ID is malformed"
                        )
                    try:
                        datetime.strptime(row[1][3:11], "%Y%m%d")
                    except ValueError as exc:
                        raise SportSourceInventoryError(
                            "Retrosheet 2025 game ID contains an invalid date"
                        ) from exc
                    game_ids.append(row[1])
                elif record_type == "play" and len(row) != 7:
                    raise SportSourceInventoryError(
                        "Retrosheet play record must have six native fields"
                    )
        if not game_ids or not counts["play"]:
            raise SportSourceInventoryError(
                "Retrosheet 2025 inventory requires games and play records"
            )
        if len(set(game_ids)) != len(game_ids):
            raise SportSourceInventoryError("Retrosheet game IDs are not unique")

    schema_material = {
        "encoding": "ASCII",
        "container": "ZIP",
        "event_filename_pattern": _RETROSHEET_EVENT_FILE_RE.pattern,
        "field_counts_by_record_type": {
            key: sorted(values) for key, values in sorted(field_counts.items())
        },
    }
    ordered_game_ids = sorted(game_ids)
    return Retrosheet2025Audit(
        season=2025,
        event_file_count=len(event_infos),
        game_count=len(game_ids),
        play_record_count=counts["play"],
        record_type_counts=dict(sorted(counts.items())),
        first_game_id=ordered_game_ids[0],
        last_game_id=ordered_game_ids[-1],
        schema_fingerprint=(
            "sha256:" + hashlib.sha256(_canonical_bytes(schema_material)).hexdigest()
        ),
        object_sha256="sha256:" + hashlib.sha256(object_bytes).hexdigest(),
        byte_length=len(object_bytes),
    )


def bevent_state_mapping() -> dict[str, Any]:
    """Declare the future Chadwick BEVENT-to-canonical state mapping."""

    return {
        "game_id": {"bevent_field": "GAME_ID"},
        "event_index": {"bevent_field": "EVENT_ID"},
        "inning": {"bevent_field": "INN_CT"},
        "batting_home": {"bevent_field": "BAT_HOME_ID"},
        "outs_before": {"bevent_field": "OUTS_CT"},
        "base_state_before": {"bevent_field": "START_BASES_CD"},
        "score_before": {
            "requires": ["AWAY_SCORE_CT", "HOME_SCORE_CT"],
        },
        "event_code": {"bevent_field": "EVENT_CD"},
        "batter_destination": {"bevent_field": "BAT_DEST_ID"},
        "runner_destinations": {
            "requires": ["RUN1_DEST_ID", "RUN2_DEST_ID", "RUN3_DEST_ID"],
        },
        "mapping_status": {
            "requires_chadwick_bevent": True,
            "executed_in_inventory": False,
        },
    }


def inspect_jolpica_response(
    object_bytes: bytes,
    *,
    endpoint_kind: JolpicaEndpointKind,
) -> JolpicaAudit:
    """Inspect native results/laps/pit-stop JSON without model claims."""

    native_key_by_kind = {
        "results": "Results",
        "laps": "Laps",
        "pitstops": "PitStops",
    }
    if endpoint_kind not in native_key_by_kind:
        raise SportSourceInventoryError(
            "endpoint_kind must be results, laps, or pitstops"
        )
    if type(object_bytes) is not bytes or not object_bytes:
        raise SportSourceInventoryError("Jolpica object must be nonempty bytes")
    parsed = _strict_json(object_bytes)
    mrdata = parsed.get("MRData")
    if type(mrdata) is not dict or mrdata.get("series") != "f1":
        raise SportSourceInventoryError("Jolpica MRData identity is invalid")
    race_table = mrdata.get("RaceTable")
    if type(race_table) is not dict or type(race_table.get("Races")) is not list:
        raise SportSourceInventoryError("Jolpica RaceTable.Races is required")
    races = race_table["Races"]
    native_key = native_key_by_kind[endpoint_kind]
    rounds: list[tuple[int, int]] = []
    native_count = 0
    for position, race in enumerate(races):
        if type(race) is not dict:
            raise SportSourceInventoryError(
                f"Jolpica Races[{position}] must be an object"
            )
        try:
            season = int(race["season"])
            round_number = int(race["round"])
            datetime.strptime(race["date"], "%Y-%m-%d")
        except (KeyError, TypeError, ValueError) as exc:
            raise SportSourceInventoryError(
                "Jolpica race season/round/date is invalid"
            ) from exc
        native = race.get(native_key)
        if type(native) is not list:
            raise SportSourceInventoryError(
                f"Jolpica race requires native {native_key} records"
            )
        if any(type(item) is not dict for item in native):
            raise SportSourceInventoryError(
                f"Jolpica {native_key} records must be objects"
            )
        rounds.append((season, round_number))
        native_count += len(native)
    if len(set(rounds)) != len(rounds):
        raise SportSourceInventoryError("Jolpica race rounds are duplicated")
    return JolpicaAudit(
        endpoint_kind=endpoint_kind,
        race_count=len(races),
        native_record_count=native_count,
        seasons=tuple(sorted({season for season, _ in rounds})),
        rounds=tuple(sorted(rounds)),
        schema_fingerprint=_schema_fingerprint(parsed),
        object_sha256="sha256:" + hashlib.sha256(object_bytes).hexdigest(),
        byte_length=len(object_bytes),
    )


def _download(
    url: str,
    *,
    accept: str,
    client: httpx.Client,
    max_bytes: int,
) -> tuple[bytes, httpx.Headers]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise SportSourceInventoryError("max_bytes must be positive")
    try:
        with client.stream(
            "GET",
            url,
            headers={"Accept": accept, "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding") not in {None, "identity"}:
                raise SportSourceInventoryError("source response is content-encoded")
            declared_text = response.headers.get("Content-Length")
            declared = None
            if declared_text is not None:
                try:
                    declared = int(declared_text)
                except ValueError as exc:
                    raise SportSourceInventoryError(
                        "source Content-Length is invalid"
                    ) from exc
                if declared > max_bytes:
                    raise SportSourceInventoryError("source object exceeds max_bytes")
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise SportSourceInventoryError("source object exceeds max_bytes")
                chunks.append(chunk)
            if declared is not None and declared != received:
                raise SportSourceInventoryError(
                    "source Content-Length differs from received bytes"
                )
            payload = b"".join(chunks)
            headers = httpx.Headers(response.headers)
    except httpx.HTTPError as exc:
        raise SportSourceInventoryError(f"source request failed: {exc}") from exc
    if not payload:
        raise SportSourceInventoryError("source returned an empty object")
    return payload, headers


def fetch_and_preserve_retrosheet_2025(
    *,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 10_000_000,
) -> PreservedRetrosheet2025:
    """Fetch and preserve the official 2025 regular-season event archive."""

    owns_client = client is None
    active = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        payload, headers = _download(
            RETROSHEET_2025_URL,
            accept="application/zip",
            client=active,
            max_bytes=max_bytes,
        )
    finally:
        if owns_client:
            active.close()
    audit = inspect_retrosheet_2025(payload)
    record = preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="retrosheet",
        dataset="DS-RETROSHEET",
        version="2025-release",
        partition="regular-season-2025",
        extension="zip",
        source_url=RETROSHEET_2025_URL,
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/zip",
                "Accept-Encoding": "identity",
            },
        },
        source_cursor="season:2025;kind:regular-season-event-files",
        fetched_at=_utc_text(fetched_at),
        coverage="2025 MLB regular-season Retrosheet event archive",
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/zip").split(";", 1)[0],
        schema_fingerprint=audit.schema_fingerprint,
        license_ref="O-007",
        license_status="research_only",
        upstream_partition="regular-season-2025",
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )
    if record.manifest.object_sha256 != audit.object_sha256:
        raise SportSourceInventoryError("Retrosheet preserved hash mismatch")
    return PreservedRetrosheet2025(record=record, audit=audit)


def fetch_and_preserve_jolpica(
    url: str,
    *,
    endpoint_kind: JolpicaEndpointKind,
    partition: str,
    coverage: str,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
    max_bytes: int = 50_000_000,
) -> PreservedJolpica:
    """Fetch one bounded Jolpica endpoint into the immutable research store."""

    if type(url) is not str or not url.startswith(JOLPICA_API_PREFIX):
        raise SportSourceInventoryError("Jolpica URL must use the official HTTPS API")
    parsed_url = urlsplit(url)
    if parsed_url.username or parsed_url.password or parsed_url.fragment:
        raise SportSourceInventoryError("Jolpica URL contains unsafe components")
    owns_client = client is None
    active = client or httpx.Client(follow_redirects=True, timeout=60)
    try:
        payload, headers = _download(
            url,
            accept="application/json",
            client=active,
            max_bytes=max_bytes,
        )
    finally:
        if owns_client:
            active.close()
    audit = inspect_jolpica_response(payload, endpoint_kind=endpoint_kind)
    query = {key: value for key, value in parse_qsl(parsed_url.query, keep_blank_values=True)}
    record = preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="jolpica",
        dataset="DS-F1-JOLPICA",
        version="api-observed-20260722",
        partition=partition,
        extension="json",
        source_url=url,
        source_request={
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            "params": query,
        },
        source_cursor=f"endpoint_kind:{endpoint_kind};partition:{partition}",
        fetched_at=_utc_text(fetched_at),
        coverage=coverage,
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=headers.get("Content-Type", "application/json").split(";", 1)[0],
        schema_fingerprint=audit.schema_fingerprint,
        license_ref="O-008",
        license_status="research_only",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )
    if record.manifest.object_sha256 != audit.object_sha256:
        raise SportSourceInventoryError("Jolpica preserved hash mismatch")
    return PreservedJolpica(record=record, audit=audit)


__all__ = [
    "JOLPICA_API_PREFIX",
    "RETROSHEET_2025_URL",
    "JolpicaAudit",
    "PreservedJolpica",
    "PreservedRetrosheet2025",
    "Retrosheet2025Audit",
    "SportSourceInventoryError",
    "bevent_state_mapping",
    "fetch_and_preserve_jolpica",
    "fetch_and_preserve_retrosheet_2025",
    "inspect_jolpica_response",
    "inspect_retrosheet_2025",
]
