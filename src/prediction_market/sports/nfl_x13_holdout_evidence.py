"""Byte-exact, metadata-only evidence for the X-13 holdout selector."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from prediction_market.static_store import (
    StaticObjectRecord,
    StaticStoreError,
    preserve_static_object,
    read_verified_static_object,
)


NFLVERSE_SCHEDULES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "schedules/games.csv"
)
POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE = (
    "https://gamma-api.polymarket.com/events/slug/{slug}"
)
KALSHI_HISTORICAL_MARKETS_URL = (
    "https://external-api.kalshi.com/trade-api/v2/historical/markets"
)
X13_HOLDOUT_EVIDENCE_VERSION = "x13-holdout-evidence-v0"

_GAME_ID_RE = re.compile(
    r"^2025_[0-9]{2}_(?P<away>[A-Z]+)_(?P<home>[A-Z]+)$"
)
_TEAM_RE = re.compile(r"^[A-Z]{2,3}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_KALSHI_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,127}$")
_CONDITION_RE = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_INDEX_KEY_TOKENS = (
    "price",
    "volume",
    "score",
    "result",
    "reaction",
)


class HoldoutEvidenceError(ValueError):
    """Metadata cannot prove an exact, reaction-blind dual-venue binding."""


@dataclass(frozen=True, slots=True)
class MetadataResponseV0:
    """One already-received HTTP response and its complete request identity."""

    object_bytes: bytes
    source_url: str
    request_params: Mapping[str, object]
    source_cursor: str | None
    received_at_utc: str
    coverage: str
    etag: str | None = None
    last_modified: str | None = None

    def __post_init__(self) -> None:
        if type(self.object_bytes) is not bytes or not self.object_bytes:
            raise HoldoutEvidenceError(
                "metadata response must contain exact nonempty bytes"
            )
        if type(self.source_url) is not str or not self.source_url.startswith(
            "https://"
        ):
            raise HoldoutEvidenceError("metadata source_url must be HTTPS")
        if not isinstance(self.request_params, Mapping):
            raise HoldoutEvidenceError("request_params must be a mapping")
        try:
            _canonical_json_bytes(dict(self.request_params))
        except (TypeError, ValueError) as error:
            raise HoldoutEvidenceError(
                "request_params must be canonical JSON data"
            ) from error
        if self.source_cursor is not None and (
            type(self.source_cursor) is not str or not self.source_cursor
        ):
            raise HoldoutEvidenceError(
                "source_cursor must be null or nonempty text"
            )
        _parse_utc(self.received_at_utc, "received_at_utc")
        if type(self.coverage) is not str or not self.coverage:
            raise HoldoutEvidenceError("coverage must be nonempty text")
        for field_name in ("etag", "last_modified"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not str:
                raise HoldoutEvidenceError(
                    f"{field_name} must be null or text"
                )


@dataclass(frozen=True, slots=True)
class HoldoutGameExpectationV0:
    """Reaction-blind exact identities frozen before evidence is inspected."""

    game_id: str
    away_team: str
    home_team: str
    polymarket_event_slug: str
    kalshi_event_ticker: str

    def __post_init__(self) -> None:
        match = (
            _GAME_ID_RE.fullmatch(self.game_id)
            if type(self.game_id) is str
            else None
        )
        if match is None:
            raise HoldoutEvidenceError("game_id is not a 2025 canonical game")
        if (
            type(self.away_team) is not str
            or type(self.home_team) is not str
            or _TEAM_RE.fullmatch(self.away_team) is None
            or _TEAM_RE.fullmatch(self.home_team) is None
            or match.group("away") != self.away_team
            or match.group("home") != self.home_team
        ):
            raise HoldoutEvidenceError(
                "game teams do not match the canonical game_id"
            )
        if (
            type(self.polymarket_event_slug) is not str
            or _SLUG_RE.fullmatch(self.polymarket_event_slug) is None
        ):
            raise HoldoutEvidenceError(
                "polymarket_event_slug is not canonical"
            )
        if (
            type(self.kalshi_event_ticker) is not str
            or _KALSHI_TICKER_RE.fullmatch(self.kalshi_event_ticker) is None
        ):
            raise HoldoutEvidenceError(
                "kalshi_event_ticker is not canonical"
            )


@dataclass(frozen=True, slots=True)
class HoldoutEvidenceBundleV0:
    """Pure-data index plus the immutable source records it references."""

    index: dict[str, object]
    index_bytes: bytes
    index_sha256: str
    records: tuple[StaticObjectRecord, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _parse_utc(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise HoldoutEvidenceError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise HoldoutEvidenceError(
            f"{field} must be an ISO-8601 UTC timestamp"
        ) from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise HoldoutEvidenceError(f"{field} must be UTC")
    return parsed.astimezone(UTC)


def _strict_json(payload: bytes, *, context: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise HoldoutEvidenceError(
                    f"{context} contains a duplicate JSON key"
                )
            value[key] = child
        return value

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HoldoutEvidenceError(
                    f"{context} contains non-finite {token}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HoldoutEvidenceError(
            f"{context} must be strict UTF-8 JSON"
        ) from error


def _json_schema_fingerprint(value: Any) -> str:
    observed: dict[str, set[str]] = {}

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
        raise HoldoutEvidenceError("JSON contains an unsupported value")

    def visit(child: Any, path: str) -> None:
        observed.setdefault(path, set()).add(kind(child))
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    return _canonical_sha256(
        {path: sorted(types) for path, types in sorted(observed.items())}
    )


def _schedule_rows(payload: bytes) -> dict[str, dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise HoldoutEvidenceError(
            "nflverse schedules must be UTF-8 CSV"
        ) from error
    reader = csv.DictReader(StringIO(text, newline=""))
    required = {
        "game_id",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
    }
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or not required <= set(reader.fieldnames)
    ):
        raise HoldoutEvidenceError(
            "nflverse schedules lack unique required metadata columns"
        )
    result: dict[str, dict[str, str]] = {}
    for row in reader:
        game_id = row.get("game_id")
        if type(game_id) is not str or not game_id:
            raise HoldoutEvidenceError("schedule game_id is missing")
        if game_id in result:
            raise HoldoutEvidenceError("schedule contains duplicate game_id")
        result[game_id] = {
            key: row.get(key, "") for key in sorted(required)
        }
    return result


def _scheduled_start(
    row: Mapping[str, str],
    expectation: HoldoutGameExpectationV0,
) -> str:
    if (
        row.get("game_id") != expectation.game_id
        or row.get("away_team") != expectation.away_team
        or row.get("home_team") != expectation.home_team
    ):
        raise HoldoutEvidenceError(
            f"{expectation.game_id} schedule identity does not match"
        )
    try:
        local = datetime.strptime(
            f"{row['gameday']} {row['gametime']}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=ZoneInfo("America/New_York"))
    except (KeyError, ValueError) as error:
        raise HoldoutEvidenceError(
            f"{expectation.game_id} schedule time is invalid"
        ) from error
    return (
        local.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _poly_metadata(
    value: Any, expectation: HoldoutGameExpectationV0
) -> tuple[str, list[dict[str, str]]]:
    if type(value) is not dict:
        raise HoldoutEvidenceError(
            f"{expectation.game_id} exact Polymarket response must be one event"
        )
    event = value
    event_id = event.get("id")
    if (
        type(event_id) is not str
        or not event_id
        or event.get("slug") != expectation.polymarket_event_slug
    ):
        raise HoldoutEvidenceError(
            f"{expectation.game_id} exact Polymarket event identity failed"
        )
    markets = event.get("markets")
    if type(markets) is not list or not markets:
        raise HoldoutEvidenceError(
            f"{expectation.game_id} exact Polymarket condition proof is missing"
        )
    refs: list[dict[str, str]] = []
    for market in markets:
        if type(market) is not dict:
            raise HoldoutEvidenceError(
                f"{expectation.game_id} Polymarket market is malformed"
            )
        market_id = market.get("id")
        condition_id = market.get("conditionId")
        if (
            type(market_id) is not str
            or not market_id
            or type(condition_id) is not str
            or _CONDITION_RE.fullmatch(condition_id) is None
        ):
            raise HoldoutEvidenceError(
                f"{expectation.game_id} exact Polymarket condition proof is invalid"
            )
        refs.append(
            {"condition_id": condition_id, "market_id": market_id}
        )
    ordered = sorted(refs, key=lambda item: (item["market_id"], item["condition_id"]))
    if len({(item["market_id"], item["condition_id"]) for item in ordered}) != len(
        ordered
    ):
        raise HoldoutEvidenceError(
            f"{expectation.game_id} Polymarket condition proof is duplicated"
        )
    return event_id, ordered


def _kalshi_metadata(
    value: Any, expectation: HoldoutGameExpectationV0
) -> list[str]:
    if type(value) is not dict or value.get("cursor") not in {"", None}:
        raise HoldoutEvidenceError(
            f"{expectation.game_id} Kalshi terminal page proof is missing"
        )
    markets = value.get("markets")
    if type(markets) is not list or not markets:
        raise HoldoutEvidenceError(
            f"{expectation.game_id} exact Kalshi markets proof is missing"
        )
    tickers: list[str] = []
    for market in markets:
        if (
            type(market) is not dict
            or market.get("event_ticker") != expectation.kalshi_event_ticker
            or type(market.get("ticker")) is not str
            or _KALSHI_TICKER_RE.fullmatch(market["ticker"]) is None
        ):
            raise HoldoutEvidenceError(
                f"{expectation.game_id} exact Kalshi market identity failed"
            )
        tickers.append(market["ticker"])
    if len(tickers) != len(set(tickers)):
        raise HoldoutEvidenceError(
            f"{expectation.game_id} Kalshi market ticker is duplicated"
        )
    return sorted(tickers)


def _request_sha(response: MetadataResponseV0) -> str:
    return _canonical_sha256(
        {
            "cursor": response.source_cursor,
            "method": "GET",
            "params": dict(response.request_params),
            "url": response.source_url,
        }
    )


def _preserve(
    response: MetadataResponseV0,
    *,
    store_root: str | Path,
    program_root: str | Path,
    source: str,
    dataset: str,
    version: str,
    partition: str,
    media_type: str,
    schema_fingerprint: str,
    license_ref: str,
    license_status: str,
) -> StaticObjectRecord:
    try:
        return preserve_static_object(
            store_root,
            response.object_bytes,
            program_root=program_root,
            source=source,
            dataset=dataset,
            version=version,
            partition=partition,
            extension="csv" if media_type == "text/csv" else "json",
            source_url=response.source_url,
            source_request={
                "headers": {
                    "Accept": media_type,
                    "Accept-Encoding": "identity",
                },
                "method": "GET",
                "params": dict(response.request_params),
                "request_identity_sha256": _request_sha(response),
            },
            source_cursor=response.source_cursor,
            fetched_at=response.received_at_utc,
            coverage=response.coverage,
            etag=response.etag,
            last_modified=response.last_modified,
            media_type=media_type,
            schema_fingerprint=schema_fingerprint,
            license_ref=license_ref,
            license_status=license_status,  # type: ignore[arg-type]
            upstream_partition=partition,
            object_kind="byte_exact_original",
            lineage={
                "query_sha256": None,
                "source_object_refs": [],
            },
        )
    except StaticStoreError as error:
        raise HoldoutEvidenceError(
            f"cannot preserve {source} static evidence"
        ) from error


def _record_ref(record: StaticObjectRecord) -> dict[str, str]:
    try:
        relative = record.manifest_path.relative_to(record.store_root)
    except ValueError as error:
        raise HoldoutEvidenceError(
            "static manifest is outside its store root"
        ) from error
    return {
        "manifest_relative_path": relative.as_posix(),
        "manifest_sha256": record.manifest.manifest_sha256,
        "object_sha256": record.manifest.object_sha256,
    }


def _validate_capture_requests(
    schedules: MetadataResponseV0,
    expectations: Sequence[HoldoutGameExpectationV0],
    polymarket: Mapping[str, MetadataResponseV0],
    kalshi: Mapping[str, MetadataResponseV0],
) -> None:
    if (
        schedules.source_url != NFLVERSE_SCHEDULES_URL
        or dict(schedules.request_params) != {}
        or schedules.source_cursor != "github_release_asset:schedules"
    ):
        raise HoldoutEvidenceError(
            "nflverse schedules request identity is not exact"
        )
    slugs = {item.polymarket_event_slug for item in expectations}
    game_ids = {item.game_id for item in expectations}
    if set(polymarket) != slugs:
        raise HoldoutEvidenceError(
            "exact Polymarket response set differs from frozen games"
        )
    if set(kalshi) != game_ids:
        raise HoldoutEvidenceError(
            "exact Kalshi response set differs from frozen games"
        )
    for expectation in expectations:
        poly = polymarket[expectation.polymarket_event_slug]
        expected_poly_url = (
            POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE.format(
                slug=expectation.polymarket_event_slug
            )
        )
        if (
            poly.source_url != expected_poly_url
            or dict(poly.request_params) != {}
            or poly.source_cursor != "request_cursor:ROOT"
        ):
            raise HoldoutEvidenceError(
                f"{expectation.game_id} exact Polymarket request identity failed"
            )
        kalshi_page = kalshi[expectation.game_id]
        params = dict(kalshi_page.request_params)
        if (
            kalshi_page.source_url != KALSHI_HISTORICAL_MARKETS_URL
            or params.get("event_ticker") != expectation.kalshi_event_ticker
            or type(params.get("limit")) is not int
            or params["limit"] <= 0
            or set(params) != {"event_ticker", "limit"}
            or kalshi_page.source_cursor != "request_cursor:ROOT"
        ):
            raise HoldoutEvidenceError(
                f"{expectation.game_id} exact Kalshi terminal request identity failed"
            )


def capture_x13_holdout_evidence(
    *,
    store_root: str | Path,
    program_root: str | Path,
    schedules_response: MetadataResponseV0,
    game_expectations: Sequence[HoldoutGameExpectationV0],
    polymarket_responses: Mapping[str, MetadataResponseV0],
    kalshi_responses: Mapping[str, MetadataResponseV0],
) -> HoldoutEvidenceBundleV0:
    """Preserve exact source bytes and return a reaction-blind evidence index."""

    if not isinstance(schedules_response, MetadataResponseV0):
        raise TypeError("schedules_response must be MetadataResponseV0")
    if (
        not isinstance(game_expectations, Sequence)
        or isinstance(game_expectations, (str, bytes))
        or not game_expectations
        or any(
            not isinstance(item, HoldoutGameExpectationV0)
            for item in game_expectations
        )
    ):
        raise TypeError(
            "game_expectations must contain HoldoutGameExpectationV0"
        )
    expectations = tuple(
        sorted(game_expectations, key=lambda item: item.game_id)
    )
    if len({item.game_id for item in expectations}) != len(expectations):
        raise HoldoutEvidenceError("holdout game_id is duplicated")
    if len({item.polymarket_event_slug for item in expectations}) != len(
        expectations
    ):
        raise HoldoutEvidenceError("Polymarket event slug is duplicated")
    if len({item.kalshi_event_ticker for item in expectations}) != len(
        expectations
    ):
        raise HoldoutEvidenceError("Kalshi event ticker is duplicated")
    if not isinstance(polymarket_responses, Mapping) or not isinstance(
        kalshi_responses, Mapping
    ):
        raise TypeError("venue responses must be mappings")
    _validate_capture_requests(
        schedules_response,
        expectations,
        polymarket_responses,
        kalshi_responses,
    )

    schedule_rows = _schedule_rows(schedules_response.object_bytes)
    missing_schedules = [
        item.game_id for item in expectations if item.game_id not in schedule_rows
    ]
    if missing_schedules:
        raise HoldoutEvidenceError(
            "nflverse schedules miss frozen games: "
            + ",".join(missing_schedules)
        )
    schedule_schema = _canonical_sha256(
        sorted(next(iter(schedule_rows.values())).keys())
    )
    schedule_record = _preserve(
        schedules_response,
        store_root=store_root,
        program_root=program_root,
        source="nflverse",
        dataset="DS-NFLVERSE",
        version=X13_HOLDOUT_EVIDENCE_VERSION,
        partition="schedule-2025",
        media_type="text/csv",
        schema_fingerprint=schedule_schema,
        license_ref="I-018",
        license_status="approved",
    )
    records: list[StaticObjectRecord] = [schedule_record]
    schedule_ref = _record_ref(schedule_record)
    games: list[dict[str, object]] = []
    for expectation in expectations:
        poly_response = polymarket_responses[
            expectation.polymarket_event_slug
        ]
        kalshi_response = kalshi_responses[expectation.game_id]
        poly_value = _strict_json(
            poly_response.object_bytes,
            context=f"{expectation.game_id} Polymarket",
        )
        kalshi_value = _strict_json(
            kalshi_response.object_bytes,
            context=f"{expectation.game_id} Kalshi",
        )
        poly_event_id, condition_refs = _poly_metadata(
            poly_value, expectation
        )
        kalshi_market_refs = _kalshi_metadata(kalshi_value, expectation)
        poly_record = _preserve(
            poly_response,
            store_root=store_root,
            program_root=program_root,
            source="polymarket",
            dataset="DS-POLYMARKET-PUBLIC",
            version=X13_HOLDOUT_EVIDENCE_VERSION,
            partition=f"poly-{expectation.game_id}",
            media_type="application/json",
            schema_fingerprint=_json_schema_fingerprint(poly_value),
            license_ref="O-001",
            license_status="research_only",
        )
        kalshi_record = _preserve(
            kalshi_response,
            store_root=store_root,
            program_root=program_root,
            source="kalshi",
            dataset="DS-KALSHI-HISTORICAL",
            version=X13_HOLDOUT_EVIDENCE_VERSION,
            partition=f"kalshi-{expectation.game_id}",
            media_type="application/json",
            schema_fingerprint=_json_schema_fingerprint(kalshi_value),
            license_ref="O-003",
            license_status="research_only",
        )
        records.extend((poly_record, kalshi_record))
        games.append(
            {
                "away_team": expectation.away_team,
                "game_id": expectation.game_id,
                "home_team": expectation.home_team,
                "kalshi": {
                    "event_ticker": expectation.kalshi_event_ticker,
                    "market_refs": kalshi_market_refs,
                    **_record_ref(kalshi_record),
                },
                "polymarket": {
                    "condition_refs": condition_refs,
                    "event_id": poly_event_id,
                    "event_slug": expectation.polymarket_event_slug,
                    **_record_ref(poly_record),
                },
                "schedule_evidence": dict(schedule_ref),
                "scheduled_start_utc": _scheduled_start(
                    schedule_rows[expectation.game_id], expectation
                ),
            }
        )
    material: dict[str, object] = {
        "claim_scope": "X13_HISTORICAL_HOLDOUT_SELECTOR_METADATA_ONLY",
        "experiment_id": "X-13",
        "game_count": len(games),
        "games": games,
        "prohibited_data": [
            "game_result",
            "price",
            "reaction",
            "volume",
        ],
        "schema": "nfl_x13_holdout_metadata_evidence_index_v0",
    }
    index_sha256 = _canonical_sha256(material)
    index = {**material, "index_sha256": index_sha256}
    index_bytes = _canonical_json_bytes(index) + b"\n"
    verified = verify_holdout_evidence_index(
        index_bytes,
        store_root=store_root,
        program_root=program_root,
    )
    if verified != index:
        raise HoldoutEvidenceError(
            "captured holdout evidence index did not self-verify"
        )
    return HoldoutEvidenceBundleV0(
        index=index,
        index_bytes=index_bytes,
        index_sha256=index_sha256,
        records=tuple(records),
    )


def _forbidden_index_key(value: object) -> str | None:
    if type(value) is dict:
        for key, child in value.items():
            normalized = str(key).casefold()
            if any(token in normalized for token in _FORBIDDEN_INDEX_KEY_TOKENS):
                return str(key)
            nested = _forbidden_index_key(child)
            if nested is not None:
                return nested
    elif type(value) is list:
        for child in value:
            nested = _forbidden_index_key(child)
            if nested is not None:
                return nested
    return None


def _verified_ref(
    reference: object,
    *,
    store_root: str | Path,
    program_root: str | Path,
    expected_source: str,
    expected_dataset: str,
) -> tuple[StaticObjectRecord, bytes]:
    if type(reference) is not dict or set(reference) != {
        "manifest_relative_path",
        "manifest_sha256",
        "object_sha256",
    }:
        raise HoldoutEvidenceError("evidence object reference is malformed")
    relative = PurePosixPath(reference["manifest_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise HoldoutEvidenceError("evidence manifest reference escapes root")
    manifest_path = Path(store_root).joinpath(*relative.parts)
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=program_root,
        )
    except StaticStoreError as error:
        raise HoldoutEvidenceError("static evidence verification failed") from error
    record = verified.record
    if (
        record.source != expected_source
        or record.dataset != expected_dataset
        or record.manifest.manifest_sha256 != reference["manifest_sha256"]
        or record.manifest.object_sha256 != reference["object_sha256"]
    ):
        raise HoldoutEvidenceError("static evidence identity differs")
    return record, verified.object_bytes


def verify_holdout_evidence_index(
    index_bytes: bytes,
    *,
    store_root: str | Path,
    program_root: str | Path,
) -> dict[str, object]:
    """Verify index identity and rederive every metadata binding from raw bytes."""

    if type(index_bytes) is not bytes or not index_bytes.endswith(b"\n"):
        raise HoldoutEvidenceError("evidence index is not canonical bytes")
    value = _strict_json(index_bytes[:-1], context="evidence index")
    if type(value) is not dict or index_bytes != _canonical_json_bytes(value) + b"\n":
        raise HoldoutEvidenceError("evidence index is not canonical JSON")
    required = {
        "claim_scope",
        "experiment_id",
        "game_count",
        "games",
        "index_sha256",
        "prohibited_data",
        "schema",
    }
    if (
        set(value) != required
        or value.get("schema")
        != "nfl_x13_holdout_metadata_evidence_index_v0"
        or value.get("experiment_id") != "X-13"
        or value.get("claim_scope")
        != "X13_HISTORICAL_HOLDOUT_SELECTOR_METADATA_ONLY"
        or type(value.get("index_sha256")) is not str
        or _SHA256_RE.fullmatch(value["index_sha256"]) is None
    ):
        raise HoldoutEvidenceError("evidence index structure is invalid")
    material = {key: child for key, child in value.items() if key != "index_sha256"}
    if _canonical_sha256(material) != value["index_sha256"]:
        raise HoldoutEvidenceError("evidence index self hash differs")
    forbidden = _forbidden_index_key(value)
    if forbidden is not None:
        raise HoldoutEvidenceError(
            f"evidence index contains prohibited field: {forbidden}"
        )
    games = value.get("games")
    if (
        type(games) is not list
        or not games
        or value.get("game_count") != len(games)
    ):
        raise HoldoutEvidenceError("evidence index games are invalid")
    if any(
        type(item) is not dict or type(item.get("game_id")) is not str
        for item in games
    ):
        raise HoldoutEvidenceError("evidence index games are invalid")
    game_ids = [item["game_id"] for item in games]
    if game_ids != sorted(game_ids):
        raise HoldoutEvidenceError("evidence index games are invalid")
    seen_games: set[str] = set()
    schedule_identity: tuple[str, str] | None = None
    for game in games:
        if type(game) is not dict or set(game) != {
            "away_team",
            "game_id",
            "home_team",
            "kalshi",
            "polymarket",
            "schedule_evidence",
            "scheduled_start_utc",
        }:
            raise HoldoutEvidenceError("evidence game record is malformed")
        polymarket = game["polymarket"]
        kalshi = game["kalshi"]
        schedule_evidence = game["schedule_evidence"]
        if (
            type(polymarket) is not dict
            or set(polymarket)
            != {
                "condition_refs",
                "event_id",
                "event_slug",
                "manifest_relative_path",
                "manifest_sha256",
                "object_sha256",
            }
            or type(kalshi) is not dict
            or set(kalshi)
            != {
                "event_ticker",
                "manifest_relative_path",
                "manifest_sha256",
                "market_refs",
                "object_sha256",
            }
            or type(schedule_evidence) is not dict
        ):
            raise HoldoutEvidenceError(
                "evidence game venue binding is malformed"
            )
        expectation = HoldoutGameExpectationV0(
            game_id=game["game_id"],
            away_team=game["away_team"],
            home_team=game["home_team"],
            polymarket_event_slug=polymarket["event_slug"],
            kalshi_event_ticker=kalshi["event_ticker"],
        )
        if expectation.game_id in seen_games:
            raise HoldoutEvidenceError("evidence index duplicates a game")
        seen_games.add(expectation.game_id)
        _parse_utc(game["scheduled_start_utc"], "scheduled_start_utc")

        schedule_record, schedule_bytes = _verified_ref(
            schedule_evidence,
            store_root=store_root,
            program_root=program_root,
            expected_source="nflverse",
            expected_dataset="DS-NFLVERSE",
        )
        if (
            schedule_record.manifest.source_url != NFLVERSE_SCHEDULES_URL
            or schedule_record.manifest.source_request["params"] != {}
            or schedule_record.manifest.source_cursor
            != "github_release_asset:schedules"
        ):
            raise HoldoutEvidenceError(
                "nflverse schedule static request identity differs"
            )
        current_schedule_identity = (
            schedule_record.manifest.object_sha256,
            schedule_record.manifest.manifest_sha256,
        )
        if schedule_identity is None:
            schedule_identity = current_schedule_identity
        elif current_schedule_identity != schedule_identity:
            raise HoldoutEvidenceError(
                "evidence index mixes schedule snapshots"
            )
        schedule_rows = _schedule_rows(schedule_bytes)
        if expectation.game_id not in schedule_rows or _scheduled_start(
            schedule_rows[expectation.game_id], expectation
        ) != game["scheduled_start_utc"]:
            raise HoldoutEvidenceError(
                "schedule evidence does not reproduce the index"
            )

        poly_ref = {
            key: polymarket[key]
            for key in (
                "manifest_relative_path",
                "manifest_sha256",
                "object_sha256",
            )
        }
        poly_record, poly_bytes = _verified_ref(
            poly_ref,
            store_root=store_root,
            program_root=program_root,
            expected_source="polymarket",
            expected_dataset="DS-POLYMARKET-PUBLIC",
        )
        expected_poly_url = (
            POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE.format(
                slug=expectation.polymarket_event_slug
            )
        )
        if (
            poly_record.manifest.source_url != expected_poly_url
            or poly_record.manifest.source_request["params"] != {}
            or poly_record.manifest.source_cursor != "request_cursor:ROOT"
        ):
            raise HoldoutEvidenceError(
                "Polymarket static request identity differs"
            )
        poly_event_id, condition_refs = _poly_metadata(
            _strict_json(
                poly_bytes, context=f"{expectation.game_id} Polymarket"
            ),
            expectation,
        )
        if (
            polymarket["event_id"] != poly_event_id
            or polymarket["condition_refs"] != condition_refs
        ):
            raise HoldoutEvidenceError(
                "Polymarket evidence does not reproduce the index"
            )

        kalshi_ref = {
            key: kalshi[key]
            for key in (
                "manifest_relative_path",
                "manifest_sha256",
                "object_sha256",
            )
        }
        kalshi_record, kalshi_bytes = _verified_ref(
            kalshi_ref,
            store_root=store_root,
            program_root=program_root,
            expected_source="kalshi",
            expected_dataset="DS-KALSHI-HISTORICAL",
        )
        kalshi_params = kalshi_record.manifest.source_request["params"]
        if (
            kalshi_record.manifest.source_url
            != KALSHI_HISTORICAL_MARKETS_URL
            or kalshi_params.get("event_ticker")
            != expectation.kalshi_event_ticker
            or set(kalshi_params) != {"event_ticker", "limit"}
            or kalshi_record.manifest.source_cursor != "request_cursor:ROOT"
        ):
            raise HoldoutEvidenceError("Kalshi static request identity differs")
        market_refs = _kalshi_metadata(
            _strict_json(
                kalshi_bytes, context=f"{expectation.game_id} Kalshi"
            ),
            expectation,
        )
        if kalshi["market_refs"] != market_refs:
            raise HoldoutEvidenceError(
                "Kalshi evidence does not reproduce the index"
            )
    return value


__all__ = [
    "KALSHI_HISTORICAL_MARKETS_URL",
    "NFLVERSE_SCHEDULES_URL",
    "POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE",
    "HoldoutEvidenceBundleV0",
    "HoldoutEvidenceError",
    "HoldoutGameExpectationV0",
    "MetadataResponseV0",
    "capture_x13_holdout_evidence",
    "verify_holdout_evidence_index",
]
