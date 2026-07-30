"""Immutable native-rule evidence overlay for the NFL X-13 development cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market import contracts
from prediction_market.sports.nfl_expansion_development_market_artifacts import (
    strict_load_json_bytes,
)
from prediction_market.sports.nfl_historical_sports_clean import (
    discover_main_checkout,
)
from prediction_market.sports.nfl_x16_exact153 import (
    Exact153Authority,
    NFLExact153PublicationError,
    verify_exact153_authority,
)


VENUE_EVIDENCE_SCHEMA = "VenueNativeRuleEvidenceV1"
PAIR_AUDIT_SCHEMA = "VenueContractPairAuditV1"
BATCH_MANIFEST_SCHEMA = "VenueNativeRuleAuditBatchManifestV1"
BUILDER_VERSION = "nfl_x15_native_rule_audit_v1"
CLAIM_BOUNDARY = (
    "DEVELOPMENT_METADATA_AND_SPORTS_FACTS_ONLY; no trade, candle, reaction, "
    "holdout, execution, speed, causality, or alpha claim"
)
TEAM_ALIASES: Mapping[str, Mapping[str, str]] = {
    "KALSHI": {"JAX": "JAC"},
    "POLYMARKET": {},
}
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_METADATA_URLS = (
    "https://gamma-api.polymarket.com/events/slug/",
    "https://external-api.kalshi.com/trade-api/v2/historical/markets",
)
EXPECTED_CAPTURE_BATCH_FILE_SHA256 = (
    "sha256:ce8e509657b6b9eb0e9cb116758242a1462b45090fc8a668c8af6de2c2a8c061"
)
EXPECTED_FACTS_BATCH_FILE_SHA256 = (
    "sha256:5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
)
EXPECTED_GOVERNANCE_OBJECT_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
DEFAULT_CAPTURE_BATCH = Path(
    "artifacts/market-observation/nfl/x13/expansion-capture/"
    "moneyline-reaction-closed-v1/batch/manifests/"
    "ce8e509657b6b9eb0e9cb116758242a1462b45090fc8a668c8af6de2c2a8c061"
    ".manifest.json"
)
DEFAULT_FACTS_BATCH = Path(
    "artifacts/market-observation/nfl/x13/exact-153-facts-v4/batches/"
    "manifests/sha256/5d/"
    "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    ".batch-index.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x15/native-rule-audit-v1"
)


class RuleAuditError(RuntimeError):
    """A native-rule input, publication, or verification failed closed."""


@dataclass(frozen=True, slots=True)
class NativeRuleInput:
    venue: str
    contract_ids: tuple[str, ...]
    identity_material: Mapping[str, object]
    rule_texts: tuple[str, ...]
    resolution_source: str | None
    raw_manifest_sha256s: tuple[str, ...]
    raw_object_sha256s: tuple[str, ...]
    request_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuleAuditFrames:
    venue_evidence: pd.DataFrame
    contract_pairs: pd.DataFrame
    read_audit: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PublishedRuleAudit:
    output_root: Path
    manifest_path: Path
    manifest_file_sha256: str
    batch_sha256: str
    venue_evidence_row_count: int
    contract_pair_row_count: int


@dataclass(frozen=True, slots=True)
class VerifiedRuleAudit:
    output_root: Path
    manifest_path: Path
    manifest_file_sha256: str
    batch_sha256: str
    venue_evidence_row_count: int
    contract_pair_row_count: int
    completed_non_tie_comparable_count: int
    strict_rule_equivalent_count: int


@dataclass(frozen=True, slots=True)
class VerifiedRuleAuditSources:
    project_root: Path
    raw_root: Path
    authority: Exact153Authority
    capture_batch_path: Path
    capture_batch: Mapping[str, object]
    development_capture_entries: tuple[Mapping[str, object], ...]
    facts_batch_path: Path
    facts_batch: Mapping[str, object]
    development_fact_entries: Mapping[str, Mapping[str, object]]


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise RuleAuditError(f"{label} is not a canonical SHA-256")
    return value


def kalshi_team_code(canonical_team: str) -> str:
    team = str(canonical_team).upper()
    return TEAM_ALIASES["KALSHI"].get(team, team)


def canonical_team_code(team: str, *, venue: str) -> str:
    code = str(team).upper()
    venue_name = str(venue).upper()
    if venue_name not in TEAM_ALIASES:
        raise RuleAuditError(f"unsupported venue: {venue}")
    reverse = {
        native: canonical
        for canonical, native in TEAM_ALIASES[venue_name].items()
    }
    return reverse.get(code, code)


def filter_development_capture_entries(
    *,
    entries: Sequence[Mapping[str, object]],
    development_game_ids: Sequence[str],
    holdout_game_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    development = frozenset(str(value) for value in development_game_ids)
    holdout = frozenset(str(value) for value in holdout_game_ids)
    if development & holdout:
        raise RuleAuditError("development and holdout cohorts overlap")
    selected: dict[str, Mapping[str, object]] = {}
    seen_holdout: set[str] = set()
    for entry in entries:
        game_id = str(entry.get("game_id", ""))
        if game_id in holdout:
            seen_holdout.add(game_id)
            continue
        if game_id not in development:
            raise RuleAuditError(f"capture game is outside frozen cohorts: {game_id}")
        if game_id in selected:
            raise RuleAuditError(f"duplicate development capture game: {game_id}")
        selected[game_id] = entry
    if set(selected) != development or seen_holdout != holdout:
        raise RuleAuditError("capture entries do not cover the frozen cohorts")
    return tuple(selected[game_id] for game_id in sorted(selected))


def select_metadata_raw_refs(
    *,
    raw_refs: Sequence[Mapping[str, object]],
    manifest_loader: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    selected: list[Mapping[str, object]] = []
    venues: set[str] = set()
    for ref in raw_refs:
        manifest = manifest_loader(ref)
        source_url = str(manifest.get("source_url", ""))
        if source_url.startswith(_METADATA_URLS[0]):
            venue = "POLYMARKET"
        elif source_url == _METADATA_URLS[1]:
            venue = "KALSHI"
        else:
            continue
        if venue in venues:
            raise RuleAuditError(f"duplicate {venue} metadata object")
        selected.append(ref)
        venues.add(venue)
    if venues != {"POLYMARKET", "KALSHI"}:
        raise RuleAuditError("both venue metadata objects are required")
    return tuple(selected)


def _lineage_values(
    lineage: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        (
            _require_sha(
                lineage.get("manifest_sha256"),
                label="metadata.manifest_sha256",
            ),
        ),
        (
            _require_sha(
                lineage.get("object_sha256"),
                label="metadata.object_sha256",
            ),
        ),
        (
            _require_sha(
                lineage.get("request_sha256"),
                label="metadata.request_sha256",
            ),
        ),
    )


def extract_polymarket_rule_input(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    identity: Mapping[str, object],
    payload: Mapping[str, object],
    lineage: Mapping[str, object],
) -> NativeRuleInput:
    market_id = str(identity.get("polymarket_market_id", ""))
    condition_id = str(identity.get("polymarket_condition_id", ""))
    markets = payload.get("markets")
    teams = payload.get("teams")
    if type(markets) is not list or type(teams) is not list:
        raise RuleAuditError(f"{game_id} Polymarket metadata is incomplete")
    selected = [
        market
        for market in markets
        if type(market) is dict and str(market.get("id")) == market_id
    ]
    if len(selected) != 1:
        raise RuleAuditError(f"{game_id} Polymarket moneyline identity mismatch")
    market = selected[0]
    if str(market.get("conditionId", "")).lower() != condition_id.lower():
        raise RuleAuditError(f"{game_id} Polymarket condition identity mismatch")
    ordering: dict[str, Mapping[str, object]] = {}
    for team in teams:
        if type(team) is not dict or team.get("ordering") not in {"home", "away"}:
            continue
        ordering[str(team["ordering"])] = team
    if set(ordering) != {"home", "away"}:
        raise RuleAuditError(f"{game_id} Polymarket team ordering is missing")
    poly_home = canonical_team_code(
        str(ordering["home"].get("abbreviation", "")), venue="POLYMARKET"
    )
    poly_away = canonical_team_code(
        str(ordering["away"].get("abbreviation", "")), venue="POLYMARKET"
    )
    if (poly_home, poly_away) != (home_team, away_team):
        raise RuleAuditError(f"{game_id} Polymarket teams do not match Facts")
    try:
        outcomes = json.loads(str(market.get("outcomes", "")))
        tokens = json.loads(str(market.get("clobTokenIds", "")))
    except json.JSONDecodeError as exc:
        raise RuleAuditError(
            f"{game_id} Polymarket outcome identity is malformed"
        ) from exc
    if (
        type(outcomes) is not list
        or type(tokens) is not list
        or len(outcomes) != 2
        or len(tokens) != 2
        or any(type(value) is not str or not value for value in outcomes + tokens)
    ):
        raise RuleAuditError(
            f"{game_id} Polymarket outcome/token identity is invalid"
        )
    description = market.get("description")
    if type(description) is not str or not description.strip():
        raise RuleAuditError(f"{game_id} Polymarket native rule text is absent")
    manifests, objects, requests = _lineage_values(lineage)
    return NativeRuleInput(
        venue="POLYMARKET",
        contract_ids=(market_id,),
        identity_material={
            "game_id": game_id,
            "logical_market": "MONEYLINE_WINNER",
            "home_team": home_team,
            "away_team": away_team,
            "native_home_team": str(
                ordering["home"].get("abbreviation", "")
            ).upper(),
            "native_away_team": str(
                ordering["away"].get("abbreviation", "")
            ).upper(),
            "market_id": market_id,
            "condition_id": condition_id.lower(),
            "outcomes": outcomes,
            "clob_token_ids": tokens,
        },
        rule_texts=(description,),
        resolution_source=(
            str(market["resolutionSource"])
            if type(market.get("resolutionSource")) is str
            else None
        ),
        raw_manifest_sha256s=manifests,
        raw_object_sha256s=objects,
        request_sha256s=requests,
    )


def extract_kalshi_rule_input(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    identity: Mapping[str, object],
    payload: Mapping[str, object],
    lineage: Mapping[str, object],
) -> NativeRuleInput:
    event_ticker = str(identity.get("kalshi_event_ticker", ""))
    tickers = identity.get("kalshi_team_tickers")
    markets = payload.get("markets")
    if (
        type(tickers) is not list
        or len(tickers) != 2
        or any(type(value) is not str or not value for value in tickers)
        or type(markets) is not list
    ):
        raise RuleAuditError(f"{game_id} Kalshi identity/metadata is incomplete")
    expected_tickers = tuple(sorted(str(value) for value in tickers))
    selected = {
        str(market.get("ticker")): market
        for market in markets
        if type(market) is dict
        and str(market.get("ticker", "")) in expected_tickers
    }
    if tuple(sorted(selected)) != expected_tickers:
        raise RuleAuditError(f"{game_id} Kalshi contract identity mismatch")
    native_codes = {
        str(ticker).rsplit("-", 1)[-1] for ticker in expected_tickers
    }
    canonical_codes = {
        canonical_team_code(code, venue="KALSHI") for code in native_codes
    }
    if canonical_codes != {home_team, away_team}:
        raise RuleAuditError(f"{game_id} Kalshi teams do not match Facts")
    primary: list[str] = []
    secondary: list[str] = []
    for ticker in expected_tickers:
        market = selected[ticker]
        if str(market.get("event_ticker", "")) != event_ticker:
            raise RuleAuditError(f"{game_id} Kalshi event ticker mismatch")
        rule_primary = market.get("rules_primary")
        rule_secondary = market.get("rules_secondary")
        if type(rule_primary) is not str or not rule_primary.strip():
            raise RuleAuditError(f"{game_id} Kalshi primary rule is absent")
        primary.append(rule_primary)
        if type(rule_secondary) is str and rule_secondary.strip():
            secondary.append(rule_secondary)
    rule_texts = tuple(primary + sorted(set(secondary)))
    manifests, objects, requests = _lineage_values(lineage)
    return NativeRuleInput(
        venue="KALSHI",
        contract_ids=expected_tickers,
        identity_material={
            "game_id": game_id,
            "logical_market": "MONEYLINE_WINNER",
            "home_team": home_team,
            "away_team": away_team,
            "native_home_team": kalshi_team_code(home_team),
            "native_away_team": kalshi_team_code(away_team),
            "event_ticker": event_ticker,
            "team_tickers": list(expected_tickers),
        },
        rule_texts=rule_texts,
        resolution_source=None,
        raw_manifest_sha256s=manifests,
        raw_object_sha256s=objects,
        request_sha256s=requests,
    )


def derive_final_game_facts(
    *,
    game_id: str,
    events: pd.DataFrame,
    facts_manifest_sha256: str,
    facts_object_sha256: str,
) -> dict[str, object]:
    required = {
        "game_id",
        "order_sequence",
        "home_team",
        "away_team",
        "post_home_score",
        "post_away_score",
    }
    if required - set(events.columns) or events.empty:
        raise RuleAuditError(f"{game_id} canonical Facts state is incomplete")
    frame = events.loc[events["game_id"].astype(str).eq(game_id)].copy()
    frame = frame.dropna(subset=["post_home_score", "post_away_score"])
    if frame.empty:
        raise RuleAuditError(f"{game_id} has no final score state")
    teams = frame.loc[:, ["home_team", "away_team"]].astype(str).drop_duplicates()
    if len(teams) != 1:
        raise RuleAuditError(f"{game_id} Facts team identity is inconsistent")
    frame["_order"] = pd.to_numeric(frame["order_sequence"], errors="raise")
    final = frame.sort_values("_order", kind="mergesort").iloc[-1]
    home_score = int(final["post_home_score"])
    away_score = int(final["post_away_score"])
    if home_score == away_score:
        raise RuleAuditError(f"{game_id} is not a completed non-tie game")
    return {
        "game_id": game_id,
        "home_team": str(final["home_team"]),
        "away_team": str(final["away_team"]),
        "final_home_score": home_score,
        "final_away_score": away_score,
        "completed_non_tie": True,
        "facts_manifest_sha256": _require_sha(
            facts_manifest_sha256, label="facts_manifest_sha256"
        ),
        "facts_object_sha256": _require_sha(
            facts_object_sha256, label="facts_object_sha256"
        ),
    }


def _resolve_under(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuleAuditError(f"{label} escapes its authority root")
    return resolved


def _read_json_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RuleAuditError(f"{label} is not a regular file")
    payload = path.read_bytes()
    if expected_sha256 is not None and _sha256_bytes(payload) != expected_sha256:
        raise RuleAuditError(f"{label} hash mismatch")
    value = strict_load_json_bytes(payload, label=label)
    if type(value) is not dict:
        raise RuleAuditError(f"{label} must be a JSON object")
    return value, payload


def verify_exact153_rule_audit_sources(
    *,
    project_root: Path,
    raw_root: Path | None = None,
    capture_batch_path: Path = DEFAULT_CAPTURE_BATCH,
    facts_batch_path: Path = DEFAULT_FACTS_BATCH,
) -> VerifiedRuleAuditSources:
    project = Path(project_root).resolve()
    try:
        authority = verify_exact153_authority(project_root=project)
    except NFLExact153PublicationError as exc:
        raise RuleAuditError(str(exc)) from exc
    if authority.object_sha256 != EXPECTED_GOVERNANCE_OBJECT_SHA256:
        raise RuleAuditError("governance object authority drifted")
    raw = (
        discover_main_checkout(project) / "var" / "raw"
        if raw_root is None
        else Path(raw_root).resolve()
    )
    if not raw.is_dir():
        raise RuleAuditError(f"raw root is absent: {raw}")
    capture_path = _resolve_under(
        project, capture_batch_path, label="capture batch"
    )
    capture, _ = _read_json_file(
        capture_path,
        label="capture batch",
        expected_sha256=EXPECTED_CAPTURE_BATCH_FILE_SHA256,
    )
    if (
        capture.get("schema") != "nfl_moneyline_expansion_batch_manifest_v1"
        or capture.get("game_count") != 234
        or capture.get("governance_sha256")
        != EXPECTED_GOVERNANCE_OBJECT_SHA256
        or capture.get("capture_scope") != "moneyline-only"
        or capture.get("reaction_access") != "CLOSED"
        or type(capture.get("game_indexes")) is not list
    ):
        raise RuleAuditError("capture batch contract mismatch")
    development_entries = filter_development_capture_entries(
        entries=capture["game_indexes"],  # type: ignore[arg-type]
        development_game_ids=authority.development_game_ids,
        holdout_game_ids=authority.final_holdout_game_ids,
    )
    facts_path = _resolve_under(project, facts_batch_path, label="FactsV4 batch")
    facts, _ = _read_json_file(
        facts_path,
        label="FactsV4 batch",
        expected_sha256=EXPECTED_FACTS_BATCH_FILE_SHA256,
    )
    if (
        facts.get("schema") != "nfl_x13_exact153_fact_batch_index_v4"
        or facts.get("cohort") != "development"
        or facts.get("game_count") != 153
        or facts.get("market_data_read") is not False
        or facts.get("holdout_reaction_accessed") is not False
        or facts.get("authority", {}).get("object_sha256")
        != EXPECTED_GOVERNANCE_OBJECT_SHA256
        or type(facts.get("games")) is not list
    ):
        raise RuleAuditError("FactsV4 batch contract mismatch")
    fact_entries: dict[str, Mapping[str, object]] = {}
    for entry in facts["games"]:  # type: ignore[union-attr]
        if type(entry) is not dict or type(entry.get("game_id")) is not str:
            raise RuleAuditError("FactsV4 game reference is invalid")
        game_id = str(entry["game_id"])
        authority.require_development(game_id)
        if game_id in fact_entries:
            raise RuleAuditError(f"duplicate FactsV4 game: {game_id}")
        _require_sha(entry.get("manifest_sha256"), label=f"{game_id}.facts")
        fact_entries[game_id] = entry
    if set(fact_entries) != set(authority.development_game_ids):
        raise RuleAuditError("FactsV4 does not cover exact development cohort")
    return VerifiedRuleAuditSources(
        project_root=project,
        raw_root=raw,
        authority=authority,
        capture_batch_path=capture_path,
        capture_batch=capture,
        development_capture_entries=development_entries,
        facts_batch_path=facts_path,
        facts_batch=facts,
        development_fact_entries=dict(sorted(fact_entries.items())),
    )


def _load_capture_index(
    sources: VerifiedRuleAuditSources,
    entry: Mapping[str, object],
) -> Mapping[str, object]:
    game_id = str(entry.get("game_id", ""))
    sources.authority.require_development(game_id)
    expected = _require_sha(
        entry.get("index_sha256"), label=f"{game_id}.capture_index"
    )
    capture_root = sources.capture_batch_path.parents[2]
    path = _resolve_under(
        capture_root,
        str(entry.get("index_path", "")),
        label=f"{game_id} capture index",
    )
    value, _ = _read_json_file(
        path, label=f"{game_id} capture index", expected_sha256=expected
    )
    if (
        value.get("schema") != "nfl_moneyline_game_capture_index_v1"
        or value.get("game_id") != game_id
        or value.get("capture_scope") != "moneyline-only"
        or value.get("reaction_access") != "CLOSED"
        or type(value.get("identity")) is not dict
        or type(value.get("raw_manifests")) is not list
    ):
        raise RuleAuditError(f"{game_id} capture index contract mismatch")
    return value


def _load_raw_manifest(
    *,
    raw_root: Path,
    game_id: str,
    ref: Mapping[str, object],
) -> Mapping[str, object]:
    manifest_path = _resolve_under(
        raw_root,
        str(ref.get("manifest_path", "")),
        label=f"{game_id} raw manifest",
    )
    manifest, _ = _read_json_file(
        manifest_path, label=f"{game_id} raw manifest"
    )
    try:
        logical_sha = contracts.static_dataset_manifest_sha256(manifest)
    except (TypeError, ValueError) as exc:
        raise RuleAuditError(f"{game_id} raw manifest is noncanonical") from exc
    if (
        logical_sha != ref.get("manifest_sha256")
        or manifest.get("manifest_sha256") != ref.get("manifest_sha256")
        or manifest.get("object_sha256") != ref.get("object_sha256")
        or manifest.get("native_object_path") != ref.get("object_path")
        or manifest.get("byte_length") != ref.get("byte_length")
    ):
        raise RuleAuditError(f"{game_id} raw manifest lineage mismatch")
    return manifest


def _load_metadata_document(
    *,
    raw_root: Path,
    game_id: str,
    ref: Mapping[str, object],
    manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    source_url = str(manifest.get("source_url", ""))
    if not (
        source_url.startswith(_METADATA_URLS[0])
        or source_url == _METADATA_URLS[1]
    ):
        raise RuleAuditError(f"{game_id} attempted non-metadata object read")
    object_path = _resolve_under(
        raw_root,
        str(ref.get("object_path", "")),
        label=f"{game_id} metadata object",
    )
    expected_sha = _require_sha(
        ref.get("object_sha256"), label=f"{game_id}.metadata_object"
    )
    if object_path.is_symlink() or not object_path.is_file():
        raise RuleAuditError(f"{game_id} metadata object is absent")
    payload = object_path.read_bytes()
    if (
        _sha256_bytes(payload) != expected_sha
        or len(payload) != ref.get("byte_length")
    ):
        raise RuleAuditError(f"{game_id} metadata object hash mismatch")
    value = strict_load_json_bytes(payload, label=f"{game_id} metadata object")
    if type(value) is not dict:
        raise RuleAuditError(f"{game_id} metadata payload must be an object")
    lineage = {
        "manifest_sha256": _require_sha(
            ref.get("manifest_sha256"), label=f"{game_id}.metadata_manifest"
        ),
        "object_sha256": expected_sha,
        "request_sha256": _require_sha(
            ref.get("request_sha256"), label=f"{game_id}.metadata_request"
        ),
        "source_url": source_url,
    }
    return value, lineage


def _load_facts(
    sources: VerifiedRuleAuditSources, game_id: str
) -> dict[str, object]:
    sources.authority.require_development(game_id)
    entry = sources.development_fact_entries[game_id]
    facts_root = sources.facts_batch_path.parents[5]
    manifest_path = _resolve_under(
        facts_root,
        str(entry.get("manifest_path", "")),
        label=f"{game_id} FactsV4 manifest",
    )
    manifest_sha = _require_sha(
        entry.get("manifest_sha256"), label=f"{game_id}.facts_manifest"
    )
    manifest, _ = _read_json_file(
        manifest_path,
        label=f"{game_id} FactsV4 manifest",
        expected_sha256=manifest_sha,
    )
    if (
        manifest.get("schema")
        != "nfl_x13_exact153_single_game_fact_manifest_v4"
        or manifest.get("game_id") != game_id
        or manifest.get("cohort") != "development"
        or manifest.get("market_data_read") is not False
        or manifest.get("holdout_reaction_accessed") is not False
        or type(manifest.get("tables")) is not list
    ):
        raise RuleAuditError(f"{game_id} FactsV4 manifest contract mismatch")
    descriptors = [
        row
        for row in manifest["tables"]  # type: ignore[union-attr]
        if type(row) is dict and row.get("name") == "canonical_factor_events"
    ]
    if len(descriptors) != 1:
        raise RuleAuditError(f"{game_id} canonical Facts table is missing")
    descriptor = descriptors[0]
    object_sha = _require_sha(
        descriptor.get("object_sha256"), label=f"{game_id}.facts_object"
    )
    object_path = _resolve_under(
        facts_root,
        str(descriptor.get("object_path", "")),
        label=f"{game_id} canonical Facts object",
    )
    if (
        object_path.is_symlink()
        or not object_path.is_file()
        or object_path.stat().st_size != descriptor.get("byte_length")
        or _sha256_file(object_path) != object_sha
    ):
        raise RuleAuditError(f"{game_id} canonical Facts object hash mismatch")
    columns = [
        "game_id",
        "order_sequence",
        "home_team",
        "away_team",
        "post_home_score",
        "post_away_score",
    ]
    try:
        table = pq.read_table(object_path, columns=columns)
    except Exception as exc:
        raise RuleAuditError(f"{game_id} canonical Facts is unreadable") from exc
    if table.num_rows != descriptor.get("row_count"):
        raise RuleAuditError(f"{game_id} canonical Facts row count mismatch")
    return derive_final_game_facts(
        game_id=game_id,
        events=table.to_pandas(),
        facts_manifest_sha256=manifest_sha,
        facts_object_sha256=object_sha,
    )


def validate_exact153_native_rule_audit_config(
    *,
    project_root: Path,
    raw_root: Path | None = None,
) -> dict[str, object]:
    sources = verify_exact153_rule_audit_sources(
        project_root=project_root, raw_root=raw_root
    )
    return {
        "status": "VALIDATED_NO_WRITE",
        "game_count": len(sources.authority.development_game_ids),
        "expected_venue_evidence_rows": 306,
        "expected_pair_rows": 153,
        "governance_object_sha256": sources.authority.object_sha256,
        "capture_batch_file_sha256": EXPECTED_CAPTURE_BATCH_FILE_SHA256,
        "facts_batch_file_sha256": EXPECTED_FACTS_BATCH_FILE_SHA256,
        "holdout_game_index_reads": 0,
        "holdout_fact_object_reads": 0,
        "trade_object_reads": 0,
        "candle_object_reads": 0,
        "reaction_object_reads": 0,
        "write_performed": False,
    }


def publish_exact153_native_rule_audit(
    *,
    project_root: Path,
    raw_root: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PublishedRuleAudit:
    sources = verify_exact153_rule_audit_sources(
        project_root=project_root, raw_root=raw_root
    )
    evidence_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    raw_manifest_reads = 0
    for entry in sources.development_capture_entries:
        game_id = str(entry["game_id"])
        index = _load_capture_index(sources, entry)
        raw_refs = index["raw_manifests"]
        manifest_cache: dict[str, Mapping[str, object]] = {}

        def manifest_loader(ref: Mapping[str, object]) -> Mapping[str, object]:
            nonlocal raw_manifest_reads
            key = str(ref.get("manifest_sha256", ""))
            if key not in manifest_cache:
                manifest_cache[key] = _load_raw_manifest(
                    raw_root=sources.raw_root, game_id=game_id, ref=ref
                )
                raw_manifest_reads += 1
            return manifest_cache[key]

        metadata_refs = select_metadata_raw_refs(
            raw_refs=raw_refs,  # type: ignore[arg-type]
            manifest_loader=manifest_loader,
        )
        documents: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
        for ref in metadata_refs:
            manifest = manifest_loader(ref)
            payload, lineage = _load_metadata_document(
                raw_root=sources.raw_root,
                game_id=game_id,
                ref=ref,
                manifest=manifest,
            )
            source_url = str(lineage["source_url"])
            venue = (
                "POLYMARKET"
                if source_url.startswith(_METADATA_URLS[0])
                else "KALSHI"
            )
            documents[venue] = (payload, lineage)
        facts = _load_facts(sources, game_id)
        identity = index["identity"]
        assert type(identity) is dict
        polymarket = extract_polymarket_rule_input(
            game_id=game_id,
            home_team=str(facts["home_team"]),
            away_team=str(facts["away_team"]),
            identity=identity,
            payload=documents["POLYMARKET"][0],
            lineage=documents["POLYMARKET"][1],
        )
        kalshi = extract_kalshi_rule_input(
            game_id=game_id,
            home_team=str(facts["home_team"]),
            away_team=str(facts["away_team"]),
            identity=identity,
            payload=documents["KALSHI"][0],
            lineage=documents["KALSHI"][1],
        )
        evidence, pair = build_game_rule_audit(
            game_id=game_id,
            home_team=str(facts["home_team"]),
            away_team=str(facts["away_team"]),
            final_home_score=int(facts["final_home_score"]),
            final_away_score=int(facts["final_away_score"]),
            facts_manifest_sha256=str(facts["facts_manifest_sha256"]),
            facts_object_sha256=str(facts["facts_object_sha256"]),
            polymarket=polymarket,
            kalshi=kalshi,
        )
        evidence_rows.extend(evidence)
        pair_rows.append(pair)
    frames = RuleAuditFrames(
        venue_evidence=pd.DataFrame(evidence_rows),
        contract_pairs=pd.DataFrame(pair_rows),
        read_audit={
            "capture_index_reads": 153,
            "raw_manifest_reads": raw_manifest_reads,
            "metadata_object_reads": 306,
            "facts_object_reads": 153,
            "trade_object_reads": 0,
            "candle_object_reads": 0,
            "reaction_object_reads": 0,
            "holdout_game_index_reads": 0,
            "holdout_fact_object_reads": 0,
        },
    )
    return publish_rule_audit_frames(
        project_root=sources.project_root,
        output_root=output_root,
        frames=frames,
        source_bindings={
            "governance_manifest_file_sha256": (
                sources.authority.manifest_file_sha256
            ),
            "governance_object_sha256": sources.authority.object_sha256,
            "capture_batch_file_sha256": EXPECTED_CAPTURE_BATCH_FILE_SHA256,
            "facts_batch_file_sha256": EXPECTED_FACTS_BATCH_FILE_SHA256,
        },
        expected_game_count=153,
    )


def _normalized_clauses(native: NativeRuleInput) -> dict[str, str]:
    joined = " ".join(native.rule_texts).lower()
    if not native.rule_texts or not joined.strip():
        raise RuleAuditError(f"{native.venue} rule text is empty")
    winner_count = sum(
        ("win" in text.lower() and "resolv" in text.lower())
        for text in native.rule_texts
    )
    winner_required = 1 if native.venue == "POLYMARKET" else 2
    winner = "SUPPORTED" if winner_count >= winner_required else "UNPROVEN"
    postponed = (
        "SUPPORTED"
        if ("postpon" in joined or "delay" in joined)
        and ("remain open" in joined or "rescheduled" in joined)
        else "UNPROVEN"
    )
    cancellation = (
        "FIFTY_FIFTY"
        if ("cancel" in joined and ("50-50" in joined or "50/50" in joined))
        else "UNPROVEN"
    )
    tie = (
        "FIFTY_FIFTY"
        if ("tie" in joined and ("50-50" in joined or "50/50" in joined))
        else "UNSPECIFIED"
    )
    source = (
        "NFL_OFFICIAL"
        if native.resolution_source
        and "nfl.com" in native.resolution_source.lower()
        else "UNSPECIFIED"
    )
    return {
        "winner_clause": winner,
        "postponement_clause": postponed,
        "cancellation_clause": cancellation,
        "tie_clause": tie,
        "resolution_source": source,
    }


def _native_evidence_row(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    final_home_score: int,
    final_away_score: int,
    facts_manifest_sha256: str,
    facts_object_sha256: str,
    native: NativeRuleInput,
) -> dict[str, object]:
    venue = native.venue.upper()
    if venue not in TEAM_ALIASES or not native.contract_ids:
        raise RuleAuditError("native venue or contract identity is invalid")
    for label, values in (
        ("raw_manifest_sha256s", native.raw_manifest_sha256s),
        ("raw_object_sha256s", native.raw_object_sha256s),
        ("request_sha256s", native.request_sha256s),
    ):
        if not values:
            raise RuleAuditError(f"{venue} {label} is empty")
        for value in values:
            _require_sha(value, label=f"{venue}.{label}")
    clauses = _normalized_clauses(native)
    identity_material = {
        "venue": venue,
        "contract_ids": sorted(native.contract_ids),
        "identity": dict(native.identity_material),
    }
    rule_material = {
        "venue": venue,
        "contract_ids": sorted(native.contract_ids),
        "rule_texts": list(native.rule_texts),
        "resolution_source": native.resolution_source,
    }
    return {
        "schema": VENUE_EVIDENCE_SCHEMA,
        "game_id": game_id,
        "cohort": "development",
        "venue": venue,
        "home_team": home_team,
        "away_team": away_team,
        "venue_home_team": (
            kalshi_team_code(home_team) if venue == "KALSHI" else home_team
        ),
        "venue_away_team": (
            kalshi_team_code(away_team) if venue == "KALSHI" else away_team
        ),
        "contract_ids_json": json.dumps(
            sorted(native.contract_ids), separators=(",", ":")
        ),
        "contract_identity_sha256": _canonical_sha256(identity_material),
        "rule_text_sha256": _canonical_sha256(rule_material),
        "normalized_clauses_json": json.dumps(
            clauses, sort_keys=True, separators=(",", ":")
        ),
        **{f"{key}_status": value for key, value in clauses.items()},
        "raw_manifest_sha256s_json": json.dumps(
            sorted(native.raw_manifest_sha256s), separators=(",", ":")
        ),
        "raw_object_sha256s_json": json.dumps(
            sorted(native.raw_object_sha256s), separators=(",", ":")
        ),
        "request_sha256s_json": json.dumps(
            sorted(native.request_sha256s), separators=(",", ":")
        ),
        "raw_manifest_set_sha256": _canonical_sha256(
            sorted(native.raw_manifest_sha256s)
        ),
        "raw_object_set_sha256": _canonical_sha256(
            sorted(native.raw_object_sha256s)
        ),
        "request_set_sha256": _canonical_sha256(sorted(native.request_sha256s)),
        "facts_manifest_sha256": _require_sha(
            facts_manifest_sha256, label="facts_manifest_sha256"
        ),
        "facts_object_sha256": _require_sha(
            facts_object_sha256, label="facts_object_sha256"
        ),
        "final_home_score": int(final_home_score),
        "final_away_score": int(final_away_score),
        "completed_non_tie": bool(final_home_score != final_away_score),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_game_rule_audit(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    final_home_score: int,
    final_away_score: int,
    facts_manifest_sha256: str,
    facts_object_sha256: str,
    polymarket: NativeRuleInput,
    kalshi: NativeRuleInput,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if polymarket.venue != "POLYMARKET" or kalshi.venue != "KALSHI":
        raise RuleAuditError("native rule inputs must be Poly then Kalshi")
    evidence = [
        _native_evidence_row(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            final_home_score=final_home_score,
            final_away_score=final_away_score,
            facts_manifest_sha256=facts_manifest_sha256,
            facts_object_sha256=facts_object_sha256,
            native=native,
        )
        for native in (polymarket, kalshi)
    ]
    by_venue = {str(row["venue"]): row for row in evidence}
    poly_clauses = json.loads(
        str(by_venue["POLYMARKET"]["normalized_clauses_json"])
    )
    kalshi_clauses = json.loads(
        str(by_venue["KALSHI"]["normalized_clauses_json"])
    )
    compared = (
        "winner_clause",
        "postponement_clause",
        "cancellation_clause",
        "tie_clause",
    )
    mismatches = [
        clause
        for clause in compared
        if poly_clauses[clause] != kalshi_clauses[clause]
    ]
    completed_non_tie = final_home_score != final_away_score
    identity_ok = all(
        str(native.identity_material.get("home_team")) == home_team
        and str(native.identity_material.get("away_team")) == away_team
        for native in (polymarket, kalshi)
    )
    gate1 = (
        "COMPARABLE"
        if completed_non_tie
        and identity_ok
        and all(
            row["winner_clause_status"] == "SUPPORTED" for row in evidence
        )
        else "UNPROVEN"
    )
    gate2 = "NON_EQUIVALENT" if mismatches else "UNPROVEN"
    winner = home_team if final_home_score > final_away_score else away_team
    pair = {
        "schema": PAIR_AUDIT_SCHEMA,
        "game_id": game_id,
        "cohort": "development",
        "home_team": home_team,
        "away_team": away_team,
        "final_home_score": int(final_home_score),
        "final_away_score": int(final_away_score),
        "realized_winner_team": winner if completed_non_tie else None,
        "completed_non_tie": completed_non_tie,
        "polymarket_contract_identity_sha256": by_venue["POLYMARKET"][
            "contract_identity_sha256"
        ],
        "kalshi_contract_identity_sha256": by_venue["KALSHI"][
            "contract_identity_sha256"
        ],
        "polymarket_rule_text_sha256": by_venue["POLYMARKET"][
            "rule_text_sha256"
        ],
        "kalshi_rule_text_sha256": by_venue["KALSHI"]["rule_text_sha256"],
        "completed_non_tie_comparability_status": gate1,
        "completed_non_tie_comparability_evidence_sha256": _canonical_sha256(
            {
                "game_id": game_id,
                "facts": [facts_manifest_sha256, facts_object_sha256],
                "winner": winner if completed_non_tie else None,
                "identity_ok": identity_ok,
                "winner_clauses": [
                    row["winner_clause_status"] for row in evidence
                ],
            }
        ),
        "strict_rule_equivalence_status": gate2,
        "strict_mismatch_clauses_json": json.dumps(
            mismatches, separators=(",", ":")
        ),
        "strict_rule_equivalence_evidence_sha256": _canonical_sha256(
            {
                "polymarket": poly_clauses,
                "kalshi": kalshi_clauses,
                "mismatches": mismatches,
                "status": gate2,
            }
        ),
        "facts_manifest_sha256": facts_manifest_sha256,
        "facts_object_sha256": facts_object_sha256,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return evidence, pair


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        normalized: dict[str, object] = {}
        for key, value in row.items():
            if pd.isna(value):
                normalized[str(key)] = None
            elif isinstance(value, bool):
                normalized[str(key)] = bool(value)
            elif isinstance(value, int):
                normalized[str(key)] = int(value)
            elif isinstance(value, float):
                normalized[str(key)] = float(value)
            else:
                normalized[str(key)] = value
        records.append(normalized)
    return records


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", use_dictionary=True)
    payload = sink.getvalue().to_pybytes()
    schema = pq.read_schema(pa.BufferReader(payload))
    return payload, _sha256_bytes(
        schema.remove_metadata().serialize().to_pybytes()
    )


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuleAuditError(f"content-addressed collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise RuleAuditError(f"content-addressed collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _publish_table(
    output_root: Path, name: str, frame: pd.DataFrame
) -> dict[str, object]:
    payload, schema_fingerprint = _parquet_bytes(frame)
    object_sha = _sha256_bytes(payload)
    hexadecimal = object_sha.removeprefix("sha256:")
    relative = (
        Path("objects")
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.parquet"
    )
    path = (output_root / relative).resolve()
    if not path.is_relative_to(output_root.resolve()):
        raise RuleAuditError("output table escapes output root")
    _atomic_publish(path, payload)
    return {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": schema_fingerprint,
        "semantic_rows_sha256": _canonical_sha256(
            {"columns": list(frame.columns), "rows": _frame_records(frame)}
        ),
    }


def publish_rule_audit_frames(
    *,
    project_root: Path,
    output_root: Path,
    frames: RuleAuditFrames,
    source_bindings: Mapping[str, str],
    expected_game_count: int = 153,
) -> PublishedRuleAudit:
    project = Path(project_root).resolve()
    output = (project / output_root).resolve()
    if not output.is_relative_to(project):
        raise RuleAuditError("rule-audit output escapes project root")
    if (
        len(frames.venue_evidence) != expected_game_count * 2
        or len(frames.contract_pairs) != expected_game_count
        or frames.contract_pairs["game_id"].nunique() != expected_game_count
        or set(frames.venue_evidence["venue"]) != {"POLYMARKET", "KALSHI"}
    ):
        raise RuleAuditError("rule-audit frame counts are not exact")
    required_zero = (
        "trade_object_reads",
        "candle_object_reads",
        "reaction_object_reads",
        "holdout_game_index_reads",
        "holdout_fact_object_reads",
    )
    if any(int(frames.read_audit.get(key, -1)) != 0 for key in required_zero):
        raise RuleAuditError("forbidden market/holdout bytes were read")
    for key, value in source_bindings.items():
        _require_sha(value, label=key)
    evidence = frames.venue_evidence.sort_values(
        ["game_id", "venue"], kind="mergesort"
    ).reset_index(drop=True)
    pairs = frames.contract_pairs.sort_values(
        ["game_id"], kind="mergesort"
    ).reset_index(drop=True)
    tables = [
        _publish_table(output, "venue_native_rule_evidence", evidence),
        _publish_table(output, "venue_contract_pair_audit", pairs),
    ]
    comparable = int(
        pairs["completed_non_tie_comparability_status"].eq("COMPARABLE").sum()
    )
    equivalent = int(
        pairs["strict_rule_equivalence_status"].eq("EQUIVALENT").sum()
    )
    material: dict[str, object] = {
        "schema": BATCH_MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "cohort": "development",
        "game_count": expected_game_count,
        "venue_evidence_row_count": len(evidence),
        "contract_pair_row_count": len(pairs),
        "completed_non_tie_comparable_count": comparable,
        "strict_rule_equivalent_count": equivalent,
        "source_bindings": dict(sorted(source_bindings.items())),
        "read_audit": dict(sorted(frames.read_audit.items())),
        "tables": tables,
        "holdout_reaction_accessed": False,
        "market_reaction_accessed": False,
        "publication_gate": (
            "PASS"
            if comparable == expected_game_count and equivalent == 0
            else "FAIL"
        ),
    }
    batch_sha = _canonical_sha256(material)
    manifest = {**material, "batch_sha256": batch_sha}
    payload = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(payload)
    hexadecimal = manifest_sha.removeprefix("sha256:")
    relative = (
        Path("manifests")
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    manifest_path = (output / relative).resolve()
    _atomic_publish(manifest_path, payload)
    return PublishedRuleAudit(
        output_root=output,
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        batch_sha256=batch_sha,
        venue_evidence_row_count=len(evidence),
        contract_pair_row_count=len(pairs),
    )


def verify_rule_audit_batch(
    *,
    project_root: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
    expected_game_count: int = 153,
) -> VerifiedRuleAudit:
    project = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_relative_to(project) or path.is_symlink() or not path.is_file():
        raise RuleAuditError("rule-audit manifest path is invalid")
    _require_sha(manifest_file_sha256, label="manifest_file_sha256")
    if _sha256_file(path) != manifest_file_sha256:
        raise RuleAuditError("rule-audit manifest hash mismatch")
    try:
        document = json.loads(path.read_text("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuleAuditError("rule-audit manifest is unreadable") from exc
    batch_sha = document.get("batch_sha256")
    material = dict(document)
    material.pop("batch_sha256", None)
    read_audit = document.get("read_audit")
    if (
        document.get("schema") != BATCH_MANIFEST_SCHEMA
        or document.get("builder_version") != BUILDER_VERSION
        or document.get("claim_boundary") != CLAIM_BOUNDARY
        or document.get("cohort") != "development"
        or document.get("game_count") != expected_game_count
        or document.get("venue_evidence_row_count") != expected_game_count * 2
        or document.get("contract_pair_row_count") != expected_game_count
        or document.get("completed_non_tie_comparable_count")
        != expected_game_count
        or document.get("strict_rule_equivalent_count") != 0
        or document.get("holdout_reaction_accessed") is not False
        or document.get("market_reaction_accessed") is not False
        or document.get("publication_gate") != "PASS"
        or type(read_audit) is not dict
        or any(
            int(read_audit.get(key, -1)) != 0
            for key in (
                "trade_object_reads",
                "candle_object_reads",
                "reaction_object_reads",
                "holdout_game_index_reads",
                "holdout_fact_object_reads",
            )
        )
        or batch_sha != _canonical_sha256(material)
    ):
        raise RuleAuditError("rule-audit manifest contract mismatch")
    output = path.parents[3]
    tables = document.get("tables")
    if type(tables) is not list or len(tables) != 2:
        raise RuleAuditError("rule-audit table inventory mismatch")
    for descriptor in tables:
        if type(descriptor) is not dict:
            raise RuleAuditError("rule-audit table descriptor is invalid")
        object_path = (output / str(descriptor.get("object_path", ""))).resolve()
        if (
            not object_path.is_relative_to(output)
            or object_path.is_symlink()
            or not object_path.is_file()
            or _sha256_file(object_path) != descriptor.get("object_sha256")
        ):
            raise RuleAuditError("rule-audit table hash mismatch")
        table = pq.read_table(object_path)
        fingerprint = _sha256_bytes(
            table.schema.remove_metadata().serialize().to_pybytes()
        )
        frame = table.to_pandas()
        semantic = _canonical_sha256(
            {"columns": list(frame.columns), "rows": _frame_records(frame)}
        )
        if (
            table.num_rows != descriptor.get("row_count")
            or table.column_names != descriptor.get("schema_columns")
            or fingerprint != descriptor.get("schema_fingerprint")
            or semantic != descriptor.get("semantic_rows_sha256")
        ):
            raise RuleAuditError("rule-audit table contract mismatch")
    return VerifiedRuleAudit(
        output_root=output,
        manifest_path=path,
        manifest_file_sha256=manifest_file_sha256,
        batch_sha256=str(batch_sha),
        venue_evidence_row_count=int(document["venue_evidence_row_count"]),
        contract_pair_row_count=int(document["contract_pair_row_count"]),
        completed_non_tie_comparable_count=int(
            document["completed_non_tie_comparable_count"]
        ),
        strict_rule_equivalent_count=int(
            document["strict_rule_equivalent_count"]
        ),
    )


__all__ = [
    "NativeRuleInput",
    "PublishedRuleAudit",
    "RuleAuditError",
    "RuleAuditFrames",
    "VerifiedRuleAudit",
    "build_game_rule_audit",
    "canonical_team_code",
    "derive_final_game_facts",
    "extract_kalshi_rule_input",
    "extract_polymarket_rule_input",
    "filter_development_capture_entries",
    "kalshi_team_code",
    "publish_exact153_native_rule_audit",
    "publish_rule_audit_frames",
    "select_metadata_raw_refs",
    "validate_exact153_native_rule_audit_config",
    "verify_exact153_rule_audit_sources",
    "verify_rule_audit_batch",
]
