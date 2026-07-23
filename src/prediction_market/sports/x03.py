"""Fail-closed X-03 sports-market census over one frozen normalized input."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from statistics import median
from typing import Any


X03_EXPERIMENT_ID = "X-03"
X03_INPUT_VERSION = "x03-normalized-input/v0"
X03_STATUS = "POC_ONLY"
X03_WINDOW_DAYS = 28
X03_SAMPLE_INTERVAL_SECONDS = 3600

ALLOWED_VENUES = frozenset({"kalshi", "polymarket"})
ALLOWED_SPORTS = frozenset({"f1", "mlb", "nba", "nfl", "soccer"})
ALLOWED_STATUSES = frozenset({"suspended", "trading"})

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MARKET_ID_RE = re.compile(r"^market_[A-Za-z0-9][A-Za-z0-9._:-]*$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "document_version",
        "experiment_id",
        "manifest_sha256",
        "window",
        "sample_interval_seconds",
        "observations",
        "input_sha256",
    }
)
_WINDOW_FIELDS = frozenset({"start_at", "end_at"})
_OBSERVATION_FIELDS = frozenset(
    {
        "venue",
        "sport",
        "canonical_market_id",
        "observed_at",
        "best_bid",
        "best_ask",
        "depth",
        "status",
        "pit_metadata_present",
    }
)


class X03DataError(ValueError):
    """The supplied normalized document cannot support the X-03 POC."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise X03DataError("value cannot be represented as canonical JSON") from error


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TypeError("X-03 input must be bytes")
    if not payload:
        raise X03DataError("X-03 input bytes must not be empty")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise X03DataError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_float=Decimal,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                X03DataError(f"non-finite JSON value: {constant}")
            ),
        )
    except UnicodeError as error:
        raise X03DataError("X-03 input must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise X03DataError("X-03 input must be valid JSON") from error
    if type(document) is not dict:
        raise X03DataError("X-03 input must be a JSON object")
    return document


def _require_fields(
    value: object,
    expected: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise X03DataError(f"{context} must be an object")
    observed = set(value)
    if observed != expected:
        raise X03DataError(
            f"{context} fields differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _validate_digest(value: object, field: str) -> str:
    if type(value) is not str or _HASH_RE.fullmatch(value) is None:
        raise X03DataError(f"{field} must be a SHA-256 digest")
    return value


def _parse_utc(value: object, field: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise X03DataError(
            f"{field} must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise X03DataError(
            f"{field} must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"
        ) from error
    return parsed.replace(tzinfo=timezone.utc)


def _exact_number(value: object, field: str) -> Decimal:
    if type(value) not in {Decimal, int, float}:
        raise X03DataError(f"{field} must be a finite JSON number")
    if type(value) is Decimal:
        number = value
    elif type(value) is int:
        number = Decimal(value)
    else:
        if not math.isfinite(value):
            raise X03DataError(f"{field} must be a finite JSON number")
        number = Decimal(str(value))
    if not number.is_finite():
        raise X03DataError(f"{field} must be a finite JSON number")
    return number


def _canonical_float(value: Decimal, field: str) -> float:
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise X03DataError(f"{field} must be a finite JSON number") from error
    if not math.isfinite(number):
        raise X03DataError(f"{field} must be a finite JSON number")
    if Decimal(str(number)) != value:
        raise X03DataError(f"{field} has unsupported decimal precision")
    if value.is_zero():
        return 0.0
    return number


def _normalize_observation(
    value: object,
    *,
    index: int,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, object]:
    observation = _require_fields(
        value,
        _OBSERVATION_FIELDS,
        context=f"observations[{index}]",
    )
    venue = observation["venue"]
    if type(venue) is not str or venue not in ALLOWED_VENUES:
        raise X03DataError(f"observations[{index}].venue is unknown")
    sport = observation["sport"]
    if type(sport) is not str or sport not in ALLOWED_SPORTS:
        raise X03DataError(f"observations[{index}].sport is unknown")
    market_id = observation["canonical_market_id"]
    if type(market_id) is not str or _MARKET_ID_RE.fullmatch(market_id) is None:
        raise X03DataError(
            f"observations[{index}].canonical_market_id is not canonical"
        )
    observed_at_text = observation["observed_at"]
    observed_at = _parse_utc(
        observed_at_text,
        f"observations[{index}].observed_at",
    )
    interval = timedelta(seconds=X03_SAMPLE_INTERVAL_SECONDS)
    if observed_at < window_start or observed_at + interval > window_end:
        raise X03DataError(
            f"observations[{index}] is outside the locked 28-day window"
        )
    elapsed_seconds = int((observed_at - window_start).total_seconds())
    if elapsed_seconds % X03_SAMPLE_INTERVAL_SECONDS != 0:
        raise X03DataError(
            f"observations[{index}].observed_at is outside the hourly sample grid"
        )
    best_bid_exact = _exact_number(
        observation["best_bid"], f"observations[{index}].best_bid"
    )
    best_ask_exact = _exact_number(
        observation["best_ask"], f"observations[{index}].best_ask"
    )
    depth_exact = _exact_number(
        observation["depth"], f"observations[{index}].depth"
    )
    if best_bid_exact < 0:
        raise X03DataError(f"observations[{index}].best_bid must be non-negative")
    if best_ask_exact < best_bid_exact:
        raise X03DataError(
            f"observations[{index}].best_ask cannot be below best_bid"
        )
    if depth_exact < 0:
        raise X03DataError(f"observations[{index}].depth must be non-negative")
    best_bid = _canonical_float(
        best_bid_exact, f"observations[{index}].best_bid"
    )
    best_ask = _canonical_float(
        best_ask_exact, f"observations[{index}].best_ask"
    )
    depth = _canonical_float(depth_exact, f"observations[{index}].depth")
    status = observation["status"]
    if type(status) is not str or status not in ALLOWED_STATUSES:
        raise X03DataError(f"observations[{index}].status is unknown")
    pit_metadata_present = observation["pit_metadata_present"]
    if type(pit_metadata_present) is not bool:
        raise X03DataError(
            f"observations[{index}].pit_metadata_present must be boolean"
        )
    return {
        "venue": venue,
        "sport": sport,
        "canonical_market_id": market_id,
        "observed_at": observed_at_text,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "depth": depth,
        "status": status,
        "pit_metadata_present": pit_metadata_present,
    }


def _normalized_input_material(
    document: Mapping[str, object],
    *,
    require_hash: bool,
) -> tuple[dict[str, object], datetime, datetime]:
    if not isinstance(document, Mapping):
        raise TypeError("X-03 normalized document must be a mapping")
    plain = dict(document)
    expected = _TOP_LEVEL_FIELDS if require_hash else _TOP_LEVEL_FIELDS - {
        "input_sha256"
    }
    _require_fields(plain, expected, context="X-03 input")
    if plain["document_version"] != X03_INPUT_VERSION:
        raise X03DataError(
            f"document_version must be {X03_INPUT_VERSION}"
        )
    if plain["experiment_id"] != X03_EXPERIMENT_ID:
        raise X03DataError(f"experiment_id must be {X03_EXPERIMENT_ID}")
    manifest_sha256 = _validate_digest(
        plain["manifest_sha256"], "manifest_sha256"
    )
    window = _require_fields(
        plain["window"],
        _WINDOW_FIELDS,
        context="window",
    )
    start_at = _parse_utc(window["start_at"], "window.start_at")
    end_at = _parse_utc(window["end_at"], "window.end_at")
    if end_at - start_at != timedelta(days=X03_WINDOW_DAYS):
        raise X03DataError("window must be exactly 28 days")
    sample_interval = plain["sample_interval_seconds"]
    if (
        type(sample_interval) is not int
        or sample_interval != X03_SAMPLE_INTERVAL_SECONDS
    ):
        raise X03DataError("sample_interval_seconds must equal 3600")
    observations = plain["observations"]
    if type(observations) is not list or not observations:
        raise X03DataError("observations must be a nonempty array")
    normalized = [
        _normalize_observation(
            observation,
            index=index,
            window_start=start_at,
            window_end=end_at,
        )
        for index, observation in enumerate(observations)
    ]
    logical_keys: set[tuple[str, str, str]] = set()
    for observation in normalized:
        key = (
            str(observation["venue"]),
            str(observation["canonical_market_id"]),
            str(observation["observed_at"]),
        )
        if key in logical_keys:
            raise X03DataError(
                "duplicate observation for venue, market, and timestamp"
            )
        logical_keys.add(key)
    normalized.sort(
        key=lambda observation: (
            str(observation["sport"]),
            str(observation["venue"]),
            str(observation["canonical_market_id"]),
            str(observation["observed_at"]),
        )
    )
    material: dict[str, object] = {
        "document_version": X03_INPUT_VERSION,
        "experiment_id": X03_EXPERIMENT_ID,
        "manifest_sha256": manifest_sha256,
        "window": {
            "start_at": window["start_at"],
            "end_at": window["end_at"],
        },
        "sample_interval_seconds": X03_SAMPLE_INTERVAL_SECONDS,
        "observations": normalized,
    }
    return material, start_at, end_at


def normalized_input_sha256(document: Mapping[str, object]) -> str:
    """Hash one normalized input independent of observation row order."""

    has_hash = "input_sha256" in document
    material, _, _ = _normalized_input_material(
        document,
        require_hash=has_hash,
    )
    return _sha256(material)


def result_sha256(result: Mapping[str, object]) -> str:
    """Hash one X-03 result after excluding only its self-hash field."""

    if not isinstance(result, Mapping):
        raise TypeError("X-03 result must be a mapping")
    material = dict(result)
    material.pop("result_sha256", None)
    return _sha256(material)


def run_x03_census(payload: bytes) -> dict[str, object]:
    """Aggregate one locked 28-day normalized input into a POC-only census."""

    document = _strict_json_object(payload)
    material, start_at, end_at = _normalized_input_material(
        document,
        require_hash=True,
    )
    claimed_hash = _validate_digest(document["input_sha256"], "input_sha256")
    observed_hash = _sha256(material)
    if claimed_hash != observed_hash:
        raise X03DataError(
            f"input_sha256 mismatch: expected {observed_hash}"
        )
    observations = material["observations"]
    if not isinstance(observations, list):
        raise X03DataError("normalized observations are unavailable")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, dict):
            raise X03DataError("normalized observation is unavailable")
        grouped[
            (str(observation["sport"]), str(observation["venue"]))
        ].append(observation)

    census: list[dict[str, object]] = []
    for sport, venue in sorted(grouped):
        rows = grouped[(sport, venue)]
        eligible = [
            row for row in rows if row["pit_metadata_present"] is True
        ]
        trading = [row for row in eligible if row["status"] == "trading"]
        suspended = [row for row in eligible if row["status"] == "suspended"]
        spreads = [
            Decimal(str(row["best_ask"])) - Decimal(str(row["best_bid"]))
            for row in trading
        ]
        depths = [Decimal(str(row["depth"])) for row in trading]
        census.append(
            {
                "sport": sport,
                "venue": venue,
                "market_count": len(
                    {str(row["canonical_market_id"]) for row in rows}
                ),
                "in_play_hours": (
                    len(trading) * X03_SAMPLE_INTERVAL_SECONDS / 3600.0
                ),
                "median_executable_spread": (
                    float(median(spreads)) if spreads else None
                ),
                "median_depth": float(median(depths)) if depths else None,
                "pause_frequency": (
                    len(suspended) / len(eligible) if eligible else None
                ),
                "eligible_observation_count": len(eligible),
                "trading_observation_count": len(trading),
                "suspended_observation_count": len(suspended),
                "missing_pit_metadata_count": len(rows) - len(eligible),
            }
        )

    result_without_hash: dict[str, object] = {
        "experiment_id": X03_EXPERIMENT_ID,
        "status": X03_STATUS,
        "is_formal_result": False,
        "formal_result_eligible": False,
        "execution_mode": "frozen_normalized_input_poc",
        "input": {
            "manifest_sha256": material["manifest_sha256"],
            "normalized_input_sha256": observed_hash,
            "window": {
                "start_at": start_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_at": end_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_days": X03_WINDOW_DAYS,
            },
            "sample_interval_seconds": X03_SAMPLE_INTERVAL_SECONDS,
        },
        "metric_definitions": {
            "market_count": (
                "Distinct canonical_market_id values in the sport-venue group, "
                "including observations missing PIT metadata."
            ),
            "in_play_hours": (
                "PIT-present trading observations multiplied by the locked "
                "one-hour sample interval."
            ),
            "median_executable_spread": (
                "Median best_ask minus best_bid over PIT-present trading "
                "observations only."
            ),
            "median_depth": (
                "Median normalized executable depth over PIT-present trading "
                "observations only."
            ),
            "pause_frequency": (
                "PIT-present suspended observations divided by all PIT-present "
                "trading or suspended observations."
            ),
            "missing_pit_metadata_count": (
                "Observations excluded from every liquidity and pause metric "
                "because pit_metadata_present is false."
            ),
        },
        "totals": {
            "observation_count": len(observations),
            "market_count": len(
                {
                    (
                        str(row["venue"]),
                        str(row["canonical_market_id"]),
                    )
                    for row in observations
                }
            ),
            "missing_pit_metadata_count": sum(
                row["pit_metadata_present"] is False for row in observations
            ),
        },
        "sport_venue_census": census,
    }
    result_without_hash["result_sha256"] = result_sha256(result_without_hash)
    return result_without_hash


__all__ = [
    "ALLOWED_SPORTS",
    "ALLOWED_STATUSES",
    "ALLOWED_VENUES",
    "X03DataError",
    "X03_EXPERIMENT_ID",
    "X03_INPUT_VERSION",
    "X03_SAMPLE_INTERVAL_SECONDS",
    "X03_STATUS",
    "X03_WINDOW_DAYS",
    "normalized_input_sha256",
    "result_sha256",
    "run_x03_census",
]
