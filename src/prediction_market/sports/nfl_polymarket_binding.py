"""Fail-closed retrospective NFL-to-Polymarket canonical binding audit.

The audit discovers nflverse games from frozen Gamma text; no target game ID,
team pair, game date, home/away orientation, or cancellation flag is preseeded.
It produces identity metadata only—never quote, model, symmetry, alpha, or P&L.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from prediction_market.contracts import (
    MarketMetadataSnapshotV0,
    StaticDatasetManifestV0,
    canonical_sha256,
    market_metadata_snapshot_sha256,
    thaw_contract_v0,
    validate_static_dataset_manifest_v0,
)


class NFLPolymarketBindingError(ValueError):
    """Frozen binding evidence is absent, ambiguous, inconsistent, or tampered."""


@dataclass(frozen=True, slots=True)
class TeamAlias:
    full_name: str
    nfl_abbreviation: str


# The only sports identity registry used by this bounded audit.  Gamma must
# supply an exact alias and its exact full name in the resolution description.
TEAM_ALIAS_REGISTRY = MappingProxyType(
    {
        "Dolphins": TeamAlias("Miami Dolphins", "MIA"),
        "Patriots": TeamAlias("New England Patriots", "NE"),
        "Vikings": TeamAlias("Minnesota Vikings", "MIN"),
        "Packers": TeamAlias("Green Bay Packers", "GB"),
        "Steelers": TeamAlias("Pittsburgh Steelers", "PIT"),
        "Ravens": TeamAlias("Baltimore Ravens", "BAL"),
        "Bills": TeamAlias("Buffalo Bills", "BUF"),
        "Bengals": TeamAlias("Cincinnati Bengals", "CIN"),
    }
)
if len({item.full_name for item in TEAM_ALIAS_REGISTRY.values()}) != len(
    TEAM_ALIAS_REGISTRY
) or len(
    {item.nfl_abbreviation for item in TEAM_ALIAS_REGISTRY.values()}
) != len(TEAM_ALIAS_REGISTRY):
    raise RuntimeError("NFL team alias registry must be one-to-one")


@dataclass(frozen=True, slots=True)
class FrozenNFLPolymarketBinding:
    """Frozen source identity and digests—never the nflverse mapping answer."""

    gamma_market_id: str
    condition_id: str
    gamma_object_sha256: str
    gamma_manifest_sha256: str

    @property
    def gamma_object_path(self) -> str:
        digest = self.gamma_object_sha256.removeprefix("sha256:")
        return (
            "raw/source=polymarket/dataset=DS-POLYMARKET-PUBLIC/"
            "version=public-api-20260722/"
            f"partition=gamma-market-{self.gamma_market_id}/{digest}.json"
        )

    @property
    def gamma_manifest_path(self) -> str:
        digest = self.gamma_manifest_sha256.removeprefix("sha256:")
        return (
            "manifests/source=polymarket/dataset=DS-POLYMARKET-PUBLIC/"
            "version=public-api-20260722/"
            f"partition=gamma-market-{self.gamma_market_id}/"
            f"{digest}.manifest.json"
        )


@dataclass(frozen=True, slots=True)
class GammaGameIdentity:
    gamma_market_id: str
    gamma_event_id: str
    condition_id: str
    slug: str
    outcome_aliases: tuple[str, str]
    full_team_names: tuple[str, str]
    nfl_abbreviations: tuple[str, str]
    token_ids: tuple[str, str]
    game_date: str
    game_start_at: str
    description: str
    updated_at: str
    outcome_prices: tuple[str, str]
    closed: bool


@dataclass(frozen=True, slots=True)
class NFLVerseGameCandidate:
    native_game_id: str
    season: int
    week: int
    season_type: str
    away_team: str
    home_team: str
    game_date: str
    native_start_time: str
    row_count: int


@dataclass(frozen=True, slots=True)
class BoundedExtractEvidence:
    rows: tuple[dict[str, Any], ...]
    winning_outcome_by_condition: Mapping[str, str | None]


# Only source keys and verified digests are frozen.  Expected game IDs live in
# tests, not here.
_SOURCE_ROWS = (
    (
        "248292",
        "0x86a95921efba50773e078f7b935e0ce94b6bfb35f42fcab86ff63ff226d9237f",
        "sha256:55197647196c4acdd0187befe3db0ebb296b428355e783b894010fd28cc50e53",
        "sha256:25b00196b675a9e460df972e7c2e92e9b4140e9281fe351115ed315915aab068",
    ),
    (
        "248293",
        "0x9a7209897461d86a93aec03b1f731c338b43ef3a8306f2b8a2b1f295e04f193f",
        "sha256:4ba866d8991d8754b1e78878548f91549c174a32b9e0e9e0b1d2a48488b40029",
        "sha256:c5e0bc8b60f6063b154ee1cdb6bcdb368dad66aa10bec8dd76b865628179a8fe",
    ),
    (
        "248294",
        "0x5e5172329111b4ee0c810f2c96f30a5967e9dd283ed8a4f27fff5056246b2141",
        "sha256:5af3fc2db280791ef57237fcbbba12973099f7223f232ce15026ecce1eaeb43d",
        "sha256:8393fc627f640ac68cbbfd86da5f1a1ee9dd267aed44c71af9b40b5c58b766dd",
    ),
    (
        "248295",
        "0x80dcb010ea53350c0a67f018c861d11239e5638de7d7c779510a7177cf1a1974",
        "sha256:247df994a7ef00462e02adb8e329c6558b42002123545557f8d816ce628adec3",
        "sha256:61b2818cd6201ad8088dc566b8a7e698e2ecf66520f2d1cc158aa5376d268e3d",
    ),
)
FROZEN_BINDINGS = tuple(
    FrozenNFLPolymarketBinding(*row) for row in _SOURCE_ROWS
)
FROZEN_BINDING_BY_MARKET_ID = MappingProxyType(
    {item.gamma_market_id: item for item in FROZEN_BINDINGS}
)

NFLVERSE_OBJECT_SHA256 = (
    "sha256:931121d8897779d7944e2a293e92ed8799c8e5cceef84096ac42339003fedc09"
)
NFLVERSE_MANIFEST_SHA256 = (
    "sha256:d4ab4bd0dbf09d2fc199c0f9b8bb0c151182c3130fa17cae0599d08d74bd496d"
)
_NFLVERSE_ROOT = (
    "source=nflverse/dataset=DS-NFLVERSE/"
    "version=github-release-58152862-20260212T102526Z/partition=season-2022"
)
NFLVERSE_OBJECT_PATH = (
    f"raw/{_NFLVERSE_ROOT}/{NFLVERSE_OBJECT_SHA256[7:]}.parquet"
)
NFLVERSE_MANIFEST_PATH = (
    f"manifests/{_NFLVERSE_ROOT}/{NFLVERSE_MANIFEST_SHA256[7:]}.manifest.json"
)

POLYMARKET_V1_EXTRACT_OBJECT_SHA256 = (
    "sha256:709cb27a89df201215f75d77da91033844f5db30dcb4311f3e9e93c8994cdb6a"
)
POLYMARKET_V1_EXTRACT_MANIFEST_SHA256 = (
    "sha256:2d134425e0098c61a9401136a9ba1da7b5dc779db3ea0b5c1687c816f401473c"
)
_EXTRACT_ROOT = (
    "source=huggingface/dataset=DS-POLYMARKET-V1/"
    "version=66a1d6ddfc3cdab9e2087c1e2e855bab272d3404/"
    "partition=sports-extract-2023-01-01"
)
POLYMARKET_V1_EXTRACT_OBJECT_PATH = (
    f"raw/{_EXTRACT_ROOT}/{POLYMARKET_V1_EXTRACT_OBJECT_SHA256[7:]}.jsonl"
)
POLYMARKET_V1_EXTRACT_MANIFEST_PATH = (
    f"manifests/{_EXTRACT_ROOT}/"
    f"{POLYMARKET_V1_EXTRACT_MANIFEST_SHA256[7:]}.manifest.json"
)

_NFL_COLUMNS = (
    "game_id",
    "season",
    "home_team",
    "away_team",
    "game_date",
    "season_type",
    "week",
    "start_time",
)
_NFL_TIMEZONE = ZoneInfo("America/New_York")
_QUESTION_RE = re.compile(
    r"^NFL (?P<day>Sunday|Monday): "
    r"(?P<first>[A-Za-z]+) vs\. (?P<second>[A-Za-z]+)$"
)
_DESCRIPTION_DATE_RE = re.compile(
    r"^In the upcoming NFL game scheduled for "
    r"(?P<date>[A-Z][a-z]+ [0-9]{1,2}, [0-9]{4}):\n\n"
)
_NATIVE_GAME_ID_RE = re.compile(
    r"^(?P<season>[0-9]{4})_(?P<week>[0-9]{2})_"
    r"(?P<away>[A-Z0-9]+)_(?P<home>[A-Z0-9]+)$"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise NFLPolymarketBindingError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _json(payload: bytes, context: str) -> Any:
    if type(payload) is not bytes or not payload:
        raise NFLPolymarketBindingError(f"{context} must be exact nonempty bytes")
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NFLPolymarketBindingError(f"{context} is not strict JSON") from exc


def _two_strings(
    value: Any, field: str, *, require_unique: bool = True
) -> tuple[str, str]:
    if not isinstance(value, str):
        raise NFLPolymarketBindingError(f"Gamma {field} must be encoded JSON")
    decoded = _json(value.encode(), f"Gamma {field}")
    if (
        not isinstance(decoded, list)
        or len(decoded) != 2
        or any(not isinstance(item, str) or not item for item in decoded)
        or (require_unique and decoded[0] == decoded[1])
    ):
        raise NFLPolymarketBindingError(f"Gamma {field} must be two unique strings")
    return decoded[0], decoded[1]


def _utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise NFLPolymarketBindingError("naive timestamp is forbidden")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _gamma_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise NFLPolymarketBindingError("Gamma gameStartTime must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace(" ", "T"))
    except ValueError as exc:
        raise NFLPolymarketBindingError("invalid Gamma gameStartTime") from exc
    if parsed.tzinfo is None:
        raise NFLPolymarketBindingError("Gamma gameStartTime must carry timezone")
    return parsed.astimezone(timezone.utc)


def _nfl_time(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%m/%d/%y, %H:%M:%S")
    except ValueError as exc:
        raise NFLPolymarketBindingError("invalid nflverse start_time") from exc
    return parsed.replace(tzinfo=_NFL_TIMEZONE).astimezone(timezone.utc)


def verify_frozen_object(
    *,
    program_root: str | Path,
    payload: bytes,
    manifest: Mapping[str, Any],
    expected_object_sha256: str,
    expected_manifest_sha256: str,
) -> StaticDatasetManifestV0:
    """Reuse the governed manifest contract, then verify exact frozen bytes."""

    try:
        checked = validate_static_dataset_manifest_v0(program_root, manifest)
    except (TypeError, ValueError) as exc:
        raise NFLPolymarketBindingError("invalid frozen object manifest") from exc
    actual_sha = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if (
        checked.manifest_sha256 != expected_manifest_sha256
        or checked.object_sha256 != expected_object_sha256
        or actual_sha != expected_object_sha256
        or len(payload) != checked.byte_length
    ):
        raise NFLPolymarketBindingError("frozen object or manifest digest mismatch")
    return checked


def _read_verified(
    root: Path,
    object_path: str,
    manifest_path: str,
    object_sha: str,
    manifest_sha: str,
    context: str,
) -> tuple[bytes, StaticDatasetManifestV0]:
    store = root / "var/raw"
    payload = (store / object_path).read_bytes()
    manifest = _json((store / manifest_path).read_bytes(), f"{context} manifest")
    if not isinstance(manifest, dict):
        raise NFLPolymarketBindingError(f"{context} manifest must be an object")
    return payload, verify_frozen_object(
        program_root=root,
        payload=payload,
        manifest=manifest,
        expected_object_sha256=object_sha,
        expected_manifest_sha256=manifest_sha,
    )


def parse_gamma_market_identity(
    gamma: Mapping[str, Any],
    source: FrozenNFLPolymarketBinding,
) -> GammaGameIdentity:
    """Parse team/date identity independently from exact frozen Gamma text."""

    if (
        gamma.get("id") != source.gamma_market_id
        or gamma.get("conditionId") != source.condition_id
    ):
        raise NFLPolymarketBindingError("Gamma market/condition identity mismatch")
    aliases = _two_strings(gamma.get("outcomes"), "outcomes")
    tokens = _two_strings(gamma.get("clobTokenIds"), "clobTokenIds")
    prices = _two_strings(
        gamma.get("outcomePrices"), "outcomePrices", require_unique=False
    )
    try:
        teams = tuple(TEAM_ALIAS_REGISTRY[alias] for alias in aliases)
    except KeyError as exc:
        raise NFLPolymarketBindingError("unknown Gamma NFL team alias") from exc

    question = gamma.get("question")
    match = _QUESTION_RE.fullmatch(question) if isinstance(question, str) else None
    if match is None or (match["first"], match["second"]) != aliases:
        raise NFLPolymarketBindingError("Gamma question/outcome identity mismatch")
    slug_aliases = tuple(alias.lower() for alias in aliases)
    expected_slug = (
        f"nfl-{match['day'].lower()}-{slug_aliases[0]}-vs-{slug_aliases[1]}"
    )
    if gamma.get("slug") != expected_slug:
        raise NFLPolymarketBindingError("Gamma slug/question identity mismatch")

    description = gamma.get("description")
    if not isinstance(description, str):
        raise NFLPolymarketBindingError("Gamma resolution description is absent")
    date_match = _DESCRIPTION_DATE_RE.match(description)
    if date_match is None:
        raise NFLPolymarketBindingError("Gamma description game date is absent")
    try:
        game_date = datetime.strptime(
            date_match["date"], "%B %d, %Y"
        ).date().isoformat()
    except ValueError as exc:
        raise NFLPolymarketBindingError(
            "Gamma description game date is invalid"
        ) from exc
    for alias, team in zip(aliases, teams, strict=True):
        clause = (
            f'If the {team.full_name} win, this market will resolve to "{alias}".'
        )
        if description.count(clause) != 1:
            raise NFLPolymarketBindingError(
                "Gamma description does not verify team alias"
            )

    events = gamma.get("events")
    if (
        not isinstance(events, list)
        or len(events) != 1
        or not isinstance(events[0], Mapping)
        or not isinstance(events[0].get("id"), str)
    ):
        raise NFLPolymarketBindingError("Gamma event identity is absent")
    if gamma.get("closed") is not True or not isinstance(
        gamma.get("updatedAt"), str
    ):
        raise NFLPolymarketBindingError("Gamma lifecycle metadata is invalid")
    return GammaGameIdentity(
        gamma_market_id=source.gamma_market_id,
        gamma_event_id=events[0]["id"],
        condition_id=source.condition_id,
        slug=expected_slug,
        outcome_aliases=aliases,
        full_team_names=(teams[0].full_name, teams[1].full_name),
        nfl_abbreviations=(
            teams[0].nfl_abbreviation,
            teams[1].nfl_abbreviation,
        ),
        token_ids=tokens,
        game_date=game_date,
        game_start_at=_utc(_gamma_time(gamma.get("gameStartTime"))),
        description=description,
        updated_at=gamma["updatedAt"],
        outcome_prices=prices,
        closed=True,
    )


def find_unique_nflverse_candidate(
    rows: Sequence[Mapping[str, Any]],
    identity: GammaGameIdentity,
) -> NFLVerseGameCandidate | None:
    """Scan by unordered team pair and parsed date; never accept a target ID."""

    team_pair = frozenset(identity.nfl_abbreviations)
    matched_ids = {
        row.get("game_id")
        for row in rows
        if row.get("game_date") == identity.game_date
        and frozenset((row.get("away_team"), row.get("home_team"))) == team_pair
    }
    if None in matched_ids or any(not isinstance(item, str) for item in matched_ids):
        raise NFLPolymarketBindingError("matched nflverse game_id is invalid")
    if len(matched_ids) > 1:
        raise NFLPolymarketBindingError("team/date identity has multiple games")
    if not matched_ids:
        return None
    game_id = next(iter(matched_ids))
    game_rows = [row for row in rows if row.get("game_id") == game_id]
    exact_fields = (
        "season",
        "week",
        "season_type",
        "away_team",
        "home_team",
        "game_date",
        "start_time",
    )
    values = {
        field: {row.get(field) for row in game_rows} for field in exact_fields
    }
    if (
        any(len(field_values) != 1 for field_values in values.values())
        or {values["away_team"].copy().pop(), values["home_team"].copy().pop()}
        != team_pair
        or values["game_date"].copy().pop() != identity.game_date
    ):
        raise NFLPolymarketBindingError("nflverse candidate rows are inconsistent")
    season = next(iter(values["season"]))
    week = next(iter(values["week"]))
    season_type = next(iter(values["season_type"]))
    away = next(iter(values["away_team"]))
    home = next(iter(values["home_team"]))
    native_start = next(iter(values["start_time"]))
    if (
        type(season) is not int
        or type(week) is not int
        or not all(
            isinstance(value, str)
            for value in (season_type, away, home, native_start)
        )
    ):
        raise NFLPolymarketBindingError("nflverse candidate identity is invalid")
    candidate = NFLVerseGameCandidate(
        native_game_id=game_id,
        season=season,
        week=week,
        season_type=season_type,
        away_team=away,
        home_team=home,
        game_date=identity.game_date,
        native_start_time=native_start,
        row_count=len(game_rows),
    )
    validate_candidate_identity(identity, candidate)
    return candidate


def validate_candidate_identity(
    identity: GammaGameIdentity,
    candidate: NFLVerseGameCandidate,
) -> None:
    """Reject any caller-supplied candidate outside parsed team/date identity."""

    native = _NATIVE_GAME_ID_RE.fullmatch(candidate.native_game_id)
    if (
        native is None
        or int(native["season"]) != candidate.season
        or int(native["week"]) != candidate.week
        or native["away"] != candidate.away_team
        or native["home"] != candidate.home_team
        or frozenset((candidate.away_team, candidate.home_team))
        != frozenset(identity.nfl_abbreviations)
        or candidate.game_date != identity.game_date
    ):
        raise NFLPolymarketBindingError("candidate does not match Gamma identity")


def compare_source_starts(
    *,
    gamma_game_start_at: str,
    nflverse_start_time_native: str,
) -> dict[str, Any]:
    """Compare raw source times and reject unexplained large discrepancies."""

    gamma_start = _gamma_time(gamma_game_start_at)
    nfl_start = _nfl_time(nflverse_start_time_native)
    delta = int((gamma_start - nfl_start).total_seconds())
    anomaly = False
    if abs(delta) >= 3_600:
        if abs(delta + 43_200) > 120:
            raise NFLPolymarketBindingError("unexpected large source-time delta")
        anomaly = True
    return {
        "gamma_game_start_at": _utc(gamma_start),
        "nflverse_start_time_native": nflverse_start_time_native,
        "nflverse_start_timezone_interpretation": "America/New_York",
        "nflverse_start_at_utc": _utc(nfl_start),
        "gamma_minus_nflverse_seconds": delta,
        "anomaly_detected": anomaly,
        "anomaly_kind": (
            "gamma_game_start_approximately_12_hours_early" if anomaly else None
        ),
        "correction_applied": False,
    }


def gamma_terminal_winner(identity: GammaGameIdentity) -> str | None:
    """Resolve only exact 50-50 cancellation or explicit closed one-hot prices."""

    if not identity.closed:
        raise NFLPolymarketBindingError("Gamma terminal direction is unproven")
    try:
        prices = tuple(Decimal(value) for value in identity.outcome_prices)
    except InvalidOperation as exc:
        raise NFLPolymarketBindingError("Gamma outcome prices are invalid") from exc
    if (
        any(not price.is_finite() or price < 0 or price > 1 for price in prices)
        or sum(prices, Decimal(0)) != Decimal(1)
    ):
        raise NFLPolymarketBindingError("Gamma terminal prices are not exhaustive")
    if prices == (Decimal("0.5"), Decimal("0.5")):
        return None
    high = [index for index, price in enumerate(prices) if price >= Decimal("0.999")]
    low = [index for index, price in enumerate(prices) if price <= Decimal("0.001")]
    if len(high) != 1 or len(low) != 1 or high[0] == low[0]:
        raise NFLPolymarketBindingError("Gamma terminal winner is unproven")
    return identity.outcome_aliases[high[0]]


def validate_bounded_extract_against_gamma(
    payload: bytes,
    identities: Mapping[str, GammaGameIdentity],
) -> BoundedExtractEvidence:
    """Cross-check 17 fills against Gamma condition/outcome/token orientation."""

    lines = payload.splitlines()
    if len(lines) != 17:
        raise NFLPolymarketBindingError("bounded extract must contain 17 fills")
    parsed: list[dict[str, Any]] = []
    ordinals: set[int] = set()
    winners: dict[str, set[str | None]] = {
        condition_id: set() for condition_id in identities
    }
    for line_number, line in enumerate(lines, 1):
        item = _json(line, f"extract line {line_number}")
        if not isinstance(item, dict):
            raise NFLPolymarketBindingError("extract line must be an object")
        row, ordinal = item.get("row"), item.get("source_row_index")
        if (
            not isinstance(row, dict)
            or type(ordinal) is not int
            or ordinal < 0
            or ordinal in ordinals
        ):
            raise NFLPolymarketBindingError("extract row/ordinal is invalid")
        ordinals.add(ordinal)
        identity = identities.get(row.get("condition_id"))
        outcome_seq = row.get("outcome_seq")
        if (
            identity is None
            or type(outcome_seq) is not int
            or outcome_seq not in (1, 2)
        ):
            raise NFLPolymarketBindingError("extract condition/outcome_seq is invalid")
        offset = outcome_seq - 1
        expected = {
            "market_slug": identity.slug,
            "category_refined": "Sports",
            "asset_id": identity.token_ids[offset],
            "outcome_label": identity.outcome_aliases[offset],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise NFLPolymarketBindingError("extract outcome orientation mismatch")
        winner = row.get("winning_outcome_label")
        if winner is not None and winner not in identity.outcome_aliases:
            raise NFLPolymarketBindingError("extract winning outcome is invalid")
        winners[identity.condition_id].add(winner)
        parsed.append({"row": row, "source_row_index": ordinal})
    if {item["row"]["condition_id"] for item in parsed} != set(identities):
        raise NFLPolymarketBindingError("extract does not cover four conditions")
    if any(len(values) != 1 for values in winners.values()):
        raise NFLPolymarketBindingError("extract winning outcome is inconsistent")
    final_winners = {
        condition_id: next(iter(values))
        for condition_id, values in winners.items()
    }
    for condition_id, identity in identities.items():
        if final_winners[condition_id] != gamma_terminal_winner(identity):
            raise NFLPolymarketBindingError(
                "extract winner contradicts Gamma terminal price orientation"
            )
    return BoundedExtractEvidence(
        rows=tuple(sorted(parsed, key=lambda item: item["source_row_index"])),
        winning_outcome_by_condition=MappingProxyType(final_winners),
    )


def _snapshot(
    identity: GammaGameIdentity,
    candidate: NFLVerseGameCandidate,
    winner: str,
    manifest: StaticDatasetManifestV0,
    outcome_index: int,
    time_anomaly: bool,
) -> dict[str, Any]:
    outcome, token = (
        identity.outcome_aliases[outcome_index],
        identity.token_ids[outcome_index],
    )
    value: dict[str, Any] = {
        "snapshot_version": "v0",
        "venue": "polymarket",
        "native_event_id": identity.gamma_event_id,
        "native_market_id": identity.gamma_market_id,
        "native_condition_id": identity.condition_id,
        "native_outcome_id": outcome,
        "native_token_id": token,
        "canonical_refs": {
            "competition_id": "cmp_nfl",
            "game_id": f"game_nflverse_{candidate.native_game_id}",
            "participant_ids": [
                f"participant_nflverse_{candidate.away_team}",
                f"participant_nflverse_{candidate.home_team}",
            ],
            "venue_event_id": f"venue_event_polymarket_{identity.gamma_event_id}",
            "market_id": f"market_polymarket_{identity.gamma_market_id}",
            "outcome_id": f"outcome_polymarket_{token}",
            "condition_id": f"condition_polymarket_{identity.condition_id}",
        },
        "sport": "nfl",
        "competition": "NFL",
        "participants": list(identity.full_team_names),
        "game_start_at": identity.game_start_at,
        "rules": identity.description,
        "resolution": winner,
        "closed": True,
        "resolved": True,
        "captured_at": manifest.fetched_at,
        "source_updated_at": identity.updated_at,
        "raw_object_hash": manifest.object_sha256,
        "quality_flags": ["source_clock_unverified"] if time_anomaly else [],
    }
    value["snapshot_sha256"] = market_metadata_snapshot_sha256(value)
    try:
        return thaw_contract_v0(MarketMetadataSnapshotV0.model_validate(value))
    except ValueError as exc:
        raise NFLPolymarketBindingError("invalid retrospective snapshot") from exc


def _manifest_record(
    role: str, manifest: StaticDatasetManifestV0, manifest_path: str
) -> dict[str, Any]:
    return {
        "role": role,
        "dataset_id": manifest.dataset_id,
        "license_ref": manifest.license_ref,
        "license_status": manifest.license_status,
        "object_sha256": manifest.object_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "native_object_path": manifest.native_object_path,
        "manifest_path": manifest_path,
        "fetched_at": manifest.fetched_at,
    }


def _completed_binding(
    identity: GammaGameIdentity,
    candidate: NFLVerseGameCandidate,
    winner: str | None,
    gamma_manifest: StaticDatasetManifestV0,
    nfl_manifest: StaticDatasetManifestV0,
) -> dict[str, Any]:
    validate_candidate_identity(identity, candidate)
    gamma_winner = gamma_terminal_winner(identity)
    if (
        winner is None
        or winner not in identity.outcome_aliases
        or winner != gamma_winner
    ):
        raise NFLPolymarketBindingError("completed game lacks a valid winner")
    time_comparison = compare_source_starts(
        gamma_game_start_at=identity.game_start_at,
        nflverse_start_time_native=candidate.native_start_time,
    )
    return {
        "gamma_market_id": identity.gamma_market_id,
        "gamma_event_id": identity.gamma_event_id,
        "condition_id": identity.condition_id,
        "native_nflverse_game_id": candidate.native_game_id,
        "canonical_game_id": f"game_nflverse_{candidate.native_game_id}",
        "identity": {
            "gamma_outcome_aliases": list(identity.outcome_aliases),
            "gamma_full_team_names": list(identity.full_team_names),
            "gamma_nfl_abbreviations": list(identity.nfl_abbreviations),
            "away_team": candidate.away_team,
            "home_team": candidate.home_team,
            "game_date": candidate.game_date,
            "match_method": (
                "gamma_text_alias_parse_then_unique_unordered_team_pair_date_scan"
            ),
            "fuzzy_matching_used": False,
        },
        "nflverse_row_count": candidate.row_count,
        "source_time_comparison": time_comparison,
        "gamma_object_sha256": gamma_manifest.object_sha256,
        "gamma_manifest_sha256": gamma_manifest.manifest_sha256,
        "nflverse_object_sha256": nfl_manifest.object_sha256,
        "nflverse_manifest_sha256": nfl_manifest.manifest_sha256,
        "resolution_cross_check": {
            "gamma_outcome_prices": list(identity.outcome_prices),
            "gamma_terminal_winner": gamma_winner,
            "extract_winning_outcome": winner,
            "status": "PROVEN_RETROSPECTIVE",
        },
        "canonical_outcome_documents": [
            _snapshot(
                identity,
                candidate,
                winner,
                gamma_manifest,
                index,
                time_comparison["anomaly_detected"],
            )
            for index in range(2)
        ],
    }


def _cancelled_binding(
    identity: GammaGameIdentity, winner: str | None
) -> dict[str, Any]:
    gamma_winner = gamma_terminal_winner(identity)
    if (
        winner is not None
        or gamma_winner is not None
        or identity.closed is not True
        or identity.outcome_prices != ("0.5", "0.5")
    ):
        raise NFLPolymarketBindingError(
            "zero-candidate market lacks exact cancellation evidence"
        )
    return {
        "gamma_market_id": identity.gamma_market_id,
        "condition_id": identity.condition_id,
        "candidate_native_nflverse_game_id": None,
        "identity": {
            "gamma_outcome_aliases": list(identity.outcome_aliases),
            "gamma_full_team_names": list(identity.full_team_names),
            "nfl_team_abbreviations": list(identity.nfl_abbreviations),
            "game_date": identity.game_date,
            "home_away_orientation_known": False,
        },
        "status": "UNMATCHED_CANCELLED",
        "reason": (
            "Gamma is closed 50-50 with no winning outcome and the frozen "
            "nflverse season has zero rows for the parsed team/date identity"
        ),
        "nflverse_row_count": 0,
        "hard_binding_created": False,
        "resolution_cross_check": {
            "gamma_outcome_prices": list(identity.outcome_prices),
            "gamma_terminal_winner": gamma_winner,
            "extract_winning_outcome": winner,
            "status": "CANCELLED_50_50_RETROSPECTIVE",
        },
        "canonical_outcome_documents": [],
    }


def build_nfl_polymarket_binding_audit(
    program_root: str | Path,
) -> dict[str, Any]:
    """Build the deterministic audit from six frozen objects."""

    root = Path(program_root)
    nfl_payload, nfl_manifest = _read_verified(
        root,
        NFLVERSE_OBJECT_PATH,
        NFLVERSE_MANIFEST_PATH,
        NFLVERSE_OBJECT_SHA256,
        NFLVERSE_MANIFEST_SHA256,
        "nflverse",
    )
    try:
        nfl_rows = pq.read_table(
            io.BytesIO(nfl_payload), columns=list(_NFL_COLUMNS)
        ).to_pylist()
    except Exception as exc:
        raise NFLPolymarketBindingError("unreadable nflverse Parquet") from exc

    gamma: dict[
        str, tuple[GammaGameIdentity, StaticDatasetManifestV0]
    ] = {}
    gamma_records: list[dict[str, Any]] = []
    for source in FROZEN_BINDINGS:
        payload, manifest = _read_verified(
            root,
            source.gamma_object_path,
            source.gamma_manifest_path,
            source.gamma_object_sha256,
            source.gamma_manifest_sha256,
            f"Gamma {source.gamma_market_id}",
        )
        raw = _json(payload, f"Gamma {source.gamma_market_id}")
        if not isinstance(raw, dict):
            raise NFLPolymarketBindingError("Gamma object must be a mapping")
        identity = parse_gamma_market_identity(raw, source)
        gamma[identity.condition_id] = identity, manifest
        gamma_records.append(
            _manifest_record(
                f"gamma_market_{source.gamma_market_id}",
                manifest,
                source.gamma_manifest_path,
            )
        )

    extract_payload, extract_manifest = _read_verified(
        root,
        POLYMARKET_V1_EXTRACT_OBJECT_PATH,
        POLYMARKET_V1_EXTRACT_MANIFEST_PATH,
        POLYMARKET_V1_EXTRACT_OBJECT_SHA256,
        POLYMARKET_V1_EXTRACT_MANIFEST_SHA256,
        "Polymarket-v1 extract",
    )
    extract = validate_bounded_extract_against_gamma(
        extract_payload,
        {condition: value[0] for condition, value in gamma.items()},
    )

    exact, unmatched = [], []
    for condition_id, (identity, gamma_manifest) in gamma.items():
        winner = extract.winning_outcome_by_condition[condition_id]
        candidate = find_unique_nflverse_candidate(nfl_rows, identity)
        if candidate is None:
            unmatched.append(_cancelled_binding(identity, winner))
        else:
            exact.append(
                _completed_binding(
                    identity,
                    candidate,
                    winner,
                    gamma_manifest,
                    nfl_manifest,
                )
            )
    outcome_count = sum(len(item["canonical_outcome_documents"]) for item in exact)
    if (len(exact), len(unmatched), outcome_count) != (3, 1, 6):
        raise NFLPolymarketBindingError("frozen binding cardinality changed")

    evidence_as_of = max(
        manifest.fetched_at
        for manifest in (
            nfl_manifest,
            extract_manifest,
            *(item[1] for item in gamma.values()),
        )
    )
    audit: dict[str, Any] = {
        "artifact_id": "NFL_POLYMARKET_V1_GAME_BINDING_AUDIT",
        "artifact_version": "v0",
        "owner": "C+D2",
        "status": "RETROSPECTIVE_RESEARCH_ONLY",
        "evidence_as_of": evidence_as_of,
        "scope": {
            "sport": "NFL",
            "polymarket_v1_partition": "daily_aligned/2023-01-01.parquet",
            "nflverse_season": 2022,
            "binding_method": (
                "gamma_text_alias_parse_then_unique_unordered_team_pair_date_scan"
            ),
            "production_expected_game_ids": False,
            "fuzzy_matching_used": False,
            "time_correction_applied": False,
        },
        "input_objects": [
            _manifest_record(
                "polymarket_v1_bounded_nfl_fills",
                extract_manifest,
                POLYMARKET_V1_EXTRACT_MANIFEST_PATH,
            ),
            *gamma_records,
            _manifest_record(
                "nflverse_2022_play_by_play",
                nfl_manifest,
                NFLVERSE_MANIFEST_PATH,
            ),
        ],
        "extract_validation": {
            "row_count": len(extract.rows),
            "condition_count": len(extract.winning_outcome_by_condition),
            "condition_join": "exact_condition_id",
            "condition_outcome_token_orientation_validated": True,
            "contains_fills": True,
            "contains_level2": False,
        },
        "summary": {
            "frozen_condition_count": len(FROZEN_BINDINGS),
            "exact_game_binding_count": len(exact),
            "unmatched_cancelled_count": len(unmatched),
            "canonical_outcome_document_count": outcome_count,
            "model_output_count": 0,
            "matched_as_of_rows": 0,
        },
        "exact_game_bindings": exact,
        "unmatched_conditions": unmatched,
        "contract_assessment": {
            "contract": "MarketMetadataSnapshotV0",
            "safe_use": (
                "content-addressed venue metadata as captured in 2026, after the "
                "events; valid only for retrospective identity binding"
            ),
            "unsafe_use": (
                "must not be used as game-time point-in-time model or quote evidence"
            ),
            "all_six_documents_contract_validated": True,
        },
        "evidence_boundary": {
            "intended_use": "RETROSPECTIVE_RESEARCH_ONLY",
            "gamma_license_ref": "O-001",
            "gamma_license_status": "pending",
            "metadata_fetched_after_event": True,
            "point_in_time_for_game_state_or_model": False,
            "fills_are_level2": False,
            "local_receive_time_available": False,
            "executable_depth_available": False,
            "venue_rule_snapshot_available": False,
            "model_output_count": 0,
            "matched_as_of_rows": 0,
            "symmetry_status": "NOT_VERIFIED",
            "alpha_status": "NOT_VERIFIED",
            "profit_or_return_computed": False,
            "fill_prices_treated_as_two_sided_quotes": False,
            "forbidden_claims": [
                "historical L2 reconstruction",
                "same-time executable bid/ask symmetry",
                "lead-lag or market mispricing",
                "queue position or fill probability",
                "profit, return, or trading alpha",
            ],
        },
    }
    audit["artifact_sha256"] = nfl_polymarket_binding_audit_sha256(audit)
    return validate_nfl_polymarket_binding_audit(audit)


def nfl_polymarket_binding_audit_sha256(
    audit: Mapping[str, Any],
) -> str:
    """Hash canonical audit content while excluding the hash field itself."""

    material = dict(audit)
    material.pop("artifact_sha256", None)
    return canonical_sha256(material)


def validate_nfl_polymarket_binding_audit(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the top-level hash and all six snapshot self-hashes."""

    if not isinstance(audit, Mapping):
        raise NFLPolymarketBindingError("audit must be a mapping")
    if audit.get("artifact_sha256") != nfl_polymarket_binding_audit_sha256(
        audit
    ):
        raise NFLPolymarketBindingError("artifact_sha256 mismatch")
    bindings = audit.get("exact_game_bindings")
    unmatched = audit.get("unmatched_conditions")
    summary = audit.get("summary")
    if (
        not isinstance(bindings, list)
        or not isinstance(unmatched, list)
        or not isinstance(summary, Mapping)
    ):
        raise NFLPolymarketBindingError("audit binding collections are invalid")
    documents = [
        document
        for binding in bindings
        if isinstance(binding, Mapping)
        for document in binding.get("canonical_outcome_documents", [])
    ]
    try:
        for document in documents:
            MarketMetadataSnapshotV0.model_validate(document)
    except ValueError as exc:
        raise NFLPolymarketBindingError("snapshot self-hash is invalid") from exc
    expected_summary = (
        len(bindings),
        len(unmatched),
        len(documents),
        summary.get("model_output_count"),
        summary.get("matched_as_of_rows"),
    )
    if expected_summary != (3, 1, 6, 0, 0):
        raise NFLPolymarketBindingError("audit summary cardinality is invalid")
    return dict(audit)


def load_nfl_polymarket_binding_audit(
    path: str | Path,
) -> dict[str, Any]:
    """Load strict JSON and validate canonical top-level/snapshot hashes."""

    value = _json(Path(path).read_bytes(), "NFL Polymarket binding audit")
    if not isinstance(value, dict):
        raise NFLPolymarketBindingError("audit JSON must be an object")
    return validate_nfl_polymarket_binding_audit(value)


def canonical_audit_bytes(audit: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                audit,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise NFLPolymarketBindingError("audit is not canonical JSON") from exc
