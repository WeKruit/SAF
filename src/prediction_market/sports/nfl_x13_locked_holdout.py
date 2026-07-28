"""Exact locked X-13 historical holdout executor.

This module is intentionally narrower than discovery materialization:

* it may run only after a verified shortlist exists;
* it reads only the frozen 20-game holdout selection;
* it captures only the moneyline trades required by the locked candidates;
* it reuses the canonical NFL replay and market trade normalizers;
* it never reads 2022--2025 rolling history because the locked candidates do
  not depend on rolling features.

The output remains historical source-time association evidence.  It is not a
latency, execution, causality, or tradeability result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pyarrow.parquet as pq

from prediction_market.sports.nfl_factor_lab_analysis import (
    run_locked_historical_holdout,
    verify_shortlist_lock,
)
from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
    deduplicate_observations,
    normalize_kalshi_trade,
    normalize_polymarket_trade,
)
from prediction_market.sports.nfl_x13_replay import (
    FinalizedEpisodeV1,
    X13GameLedgerV1,
    build_x13_game_ledger,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS, kalshi_team_code
from prediction_market.static_store import (
    preserve_static_object,
    read_verified_static_object,
)


POLY_TRADES_URL = "https://data-api.polymarket.com/trades"
KALSHI_TRADES_URL = (
    "https://api.elections.kalshi.com/trade-api/v2/historical/trades"
)
POLY_LIMIT = 10_000
KALSHI_LIMIT = 1_000
CLAIM_BOUNDARY = (
    "PRELIMINARY historical source-time association only; no causality, "
    "live latency, execution, or tradeability claim."
)
LOCKED_FACTOR_IDS = frozenset(
    {
        "NFL.COMBO.THIRD_FOURTH_DOWN_FAILURE",
        "NFL.EVENT.EXPLOSIVE_PLAY",
        "NFL.SEQUENCE.ANSWER_SCORE",
    }
)
_TEAM_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ARI": ("ARI", "Arizona", "Arizona Cardinals", "Cardinals"),
    "ATL": ("ATL", "Atlanta", "Atlanta Falcons", "Falcons"),
    "BAL": ("BAL", "Baltimore", "Baltimore Ravens", "Ravens"),
    "BUF": ("BUF", "Buffalo", "Buffalo Bills", "Bills"),
    "CAR": ("CAR", "Carolina", "Carolina Panthers", "Panthers"),
    "CHI": ("CHI", "Chicago", "Chicago Bears", "Bears"),
    "CIN": ("CIN", "Cincinnati", "Cincinnati Bengals", "Bengals"),
    "CLE": ("CLE", "Cleveland", "Cleveland Browns", "Browns"),
    "DAL": ("DAL", "Dallas", "Dallas Cowboys", "Cowboys"),
    "DEN": ("DEN", "Denver", "Denver Broncos", "Broncos"),
    "DET": ("DET", "Detroit", "Detroit Lions", "Lions"),
    "GB": ("GB", "Green Bay", "Green Bay Packers", "Packers"),
    "HOU": ("HOU", "Houston", "Houston Texans", "Texans"),
    "IND": ("IND", "Indianapolis", "Indianapolis Colts", "Colts"),
    "JAX": (
        "JAC",
        "JAX",
        "Jacksonville",
        "Jacksonville Jaguars",
        "Jaguars",
    ),
    "KC": ("KC", "Kansas City", "Kansas City Chiefs", "Chiefs"),
    "LA": ("LA", "Los Angeles Rams", "Rams"),
    "LAC": ("LAC", "Los Angeles Chargers", "Chargers"),
    "LV": ("LV", "Las Vegas", "Las Vegas Raiders", "Raiders"),
    "MIA": ("MIA", "Miami", "Miami Dolphins", "Dolphins"),
    "MIN": ("MIN", "Minnesota", "Minnesota Vikings", "Vikings"),
    "NE": ("NE", "New England", "New England Patriots", "Patriots"),
    "NO": ("NO", "New Orleans", "New Orleans Saints", "Saints"),
    "NYG": ("NYG", "New York Giants", "Giants"),
    "NYJ": ("NYJ", "New York Jets", "Jets"),
    "PHI": ("PHI", "Philadelphia", "Philadelphia Eagles", "Eagles"),
    "PIT": ("PIT", "Pittsburgh", "Pittsburgh Steelers", "Steelers"),
    "SEA": ("SEA", "Seattle", "Seattle Seahawks", "Seahawks"),
    "SF": ("SF", "San Francisco", "San Francisco 49ers", "49ers"),
    "TB": (
        "TB",
        "Tampa Bay",
        "Tampa Bay Buccaneers",
        "Buccaneers",
        "Bucs",
    ),
    "TEN": ("TEN", "Tennessee", "Tennessee Titans", "Titans"),
    "WAS": (
        "WAS",
        "Washington",
        "Washington Commanders",
        "Commanders",
    ),
}


class LockedHoldoutError(ValueError):
    """The frozen holdout cannot be executed without changing its lock."""


@dataclass(frozen=True, slots=True)
class LockedCandidate:
    factor_id: str
    horizon_seconds: int
    family: str
    factor_version: str
    support_kind: str


@dataclass(frozen=True, slots=True)
class GameMarketBinding:
    game_id: str
    away_team: str
    home_team: str
    scheduled_start_utc: str
    polymarket_event_id: str
    polymarket_condition_id: str
    polymarket_market_id: str
    polymarket_outcomes: tuple[tuple[str, str], ...]
    kalshi_event_ticker: str
    kalshi_team_tickers: tuple[tuple[str, str], ...]
    rule_sha256: str


@dataclass(frozen=True, slots=True)
class LockedHoldoutPublication:
    manifest_path: Path
    bundle_sha256: str
    observation_path: Path
    result_path: Path
    observation_count: int
    result_count: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _alias_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _canonical_team(value: str, teams: tuple[str, str]) -> str:
    key = _alias_key(value)
    matches = [
        team
        for team in teams
        if key in {_alias_key(alias) for alias in _TEAM_ALIASES[team]}
    ]
    if len(matches) != 1:
        raise LockedHoldoutError(
            f"moneyline outcome {value!r} is not uniquely team-oriented"
        )
    return matches[0]


def _strict_json(raw: bytes, *, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LockedHoldoutError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_float=Decimal,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LockedHoldoutError(f"{label} has non-finite {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LockedHoldoutError(f"{label} is not strict JSON") from error


def _schema_fingerprint(value: object) -> str:
    observed: dict[str, set[str]] = defaultdict(set)

    def visit(child: object, path: str) -> None:
        if child is None:
            kind = "null"
        elif type(child) is bool:
            kind = "boolean"
        elif type(child) is int:
            kind = "integer"
        elif type(child) in {float, Decimal}:
            kind = "number"
        elif type(child) is str:
            kind = "string"
        elif type(child) is list:
            kind = "array"
        elif type(child) is dict:
            kind = "object"
        else:
            raise LockedHoldoutError("JSON contains unsupported value")
        observed[path].add(kind)
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    return _sha256_bytes(
        _canonical_bytes(
            {path: sorted(kinds) for path, kinds in sorted(observed.items())}
        )
    )


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise LockedHoldoutError("timestamp must be UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LockedHoldoutError(f"invalid UTC timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise LockedHoldoutError("timestamp must be UTC")
    return parsed.astimezone(UTC)


def _data_root(project_root: Path) -> Path:
    if project_root.parent.name == ".worktrees":
        candidate = project_root.parent.parent
        if (candidate / ".git").is_dir():
            return candidate
    return project_root


def _load_shortlist(
    shortlist_manifest: Path,
) -> tuple[tuple[LockedCandidate, ...], list[dict[str, object]], str]:
    lock = verify_shortlist_lock(shortlist_manifest)
    payload = _strict_json(lock.manifest_path.read_bytes(), label="shortlist")
    candidates = tuple(
        LockedCandidate(
            factor_id=item["factor_id"],
            horizon_seconds=int(item["horizon_seconds"]),
            family=item["market_family"],
            factor_version=next(
                definition["version"]
                for definition in payload["factor_definitions"]
                if definition["factor_id"] == item["factor_id"]
            ),
            support_kind=next(
                definition["support_kind"]
                for definition in payload["factor_definitions"]
                if definition["factor_id"] == item["factor_id"]
            ),
        )
        for item in payload["locked_candidates"]
    )
    if (
        {item.factor_id for item in candidates} != LOCKED_FACTOR_IDS
        or any(item.family != "moneyline" for item in candidates)
    ):
        raise LockedHoldoutError("shortlist is not the frozen three-candidate lock")
    return candidates, payload["factor_definitions"], lock.shortlist_sha256


def _load_holdout_selection(
    project_root: Path,
) -> tuple[dict[str, object], str]:
    candidates = sorted(
        (
            project_root
            / "artifacts/market-observation/nfl/x13/historical-holdout"
        ).glob("sha256-*/*.holdout-selection.json")
    )
    if len(candidates) != 1:
        raise LockedHoldoutError("exactly one frozen holdout selection is required")
    payload = _strict_json(candidates[0].read_bytes(), label="holdout selection")
    if payload.get("status") != "FROZEN":
        raise LockedHoldoutError("historical holdout selection is not frozen")
    games = payload.get("selected_game_ids")
    if type(games) is not list or len(games) != 20 or len(set(games)) != 20:
        raise LockedHoldoutError("holdout selection is not an exact 20-game set")
    return payload, payload["selection_id"]


def _load_evidence_index(project_root: Path) -> dict[str, object]:
    candidates = sorted(
        (
            project_root
            / "artifacts/market-observation/nfl/x13/holdout-evidence"
        ).glob("sha256-*/*.evidence-index.json")
    )
    if len(candidates) != 1:
        raise LockedHoldoutError("exactly one holdout evidence index is required")
    payload = _strict_json(candidates[0].read_bytes(), label="evidence index")
    if payload.get("claim_scope") != "X13_HISTORICAL_HOLDOUT_SELECTOR_METADATA_ONLY":
        raise LockedHoldoutError("holdout evidence scope is invalid")
    return payload


def _verified_metadata(
    *,
    data_root: Path,
    project_root: Path,
    relative_manifest: str,
) -> Any:
    verified = read_verified_static_object(
        data_root / "var/raw" / relative_manifest,
        store_root=data_root / "var/raw",
        program_root=project_root,
    )
    return _strict_json(verified.object_bytes, label=relative_manifest)


def _build_bindings(
    *,
    project_root: Path,
    selected_game_ids: Sequence[str],
    evidence_index: Mapping[str, object],
) -> tuple[GameMarketBinding, ...]:
    data_root = _data_root(project_root)
    evidence_games = {
        game["game_id"]: game for game in evidence_index["games"]  # type: ignore[index]
    }
    bindings: list[GameMarketBinding] = []
    for game_id in sorted(selected_game_ids):
        evidence = evidence_games.get(game_id)
        if type(evidence) is not dict:
            raise LockedHoldoutError(f"missing evidence for {game_id}")
        away = evidence["away_team"]
        home = evidence["home_team"]
        teams = (away, home)
        poly_ref = evidence["polymarket"]
        kalshi_ref = evidence["kalshi"]
        event = _verified_metadata(
            data_root=data_root,
            project_root=project_root,
            relative_manifest=poly_ref["manifest_relative_path"],
        )
        if event.get("id") != poly_ref["event_id"]:
            raise LockedHoldoutError(f"Polymarket event identity differs: {game_id}")
        moneylines = [
            market
            for market in event.get("markets", [])
            if type(market) is dict
            and market.get("sportsMarketType") == "moneyline"
        ]
        if len(moneylines) != 1:
            raise LockedHoldoutError(
                f"{game_id} must have exactly one Polymarket moneyline"
            )
        market = moneylines[0]
        condition_refs = {
            (item["market_id"], item["condition_id"])
            for item in poly_ref["condition_refs"]
        }
        if (market.get("id"), market.get("conditionId")) not in condition_refs:
            raise LockedHoldoutError(
                f"{game_id} moneyline is outside frozen condition refs"
            )
        native_outcomes = json.loads(market["outcomes"])
        if type(native_outcomes) is not list or len(native_outcomes) != 2:
            raise LockedHoldoutError(f"{game_id} moneyline outcomes are invalid")
        oriented = tuple(
            (native, _canonical_team(native, teams))
            for native in native_outcomes
        )
        if {team for _native, team in oriented} != set(teams):
            raise LockedHoldoutError(
                f"{game_id} Polymarket moneyline does not cover both teams"
            )
        ticker_pairs: list[tuple[str, str]] = []
        for ticker in kalshi_ref["market_refs"]:
            matches = [
                team
                for team in teams
                if ticker.endswith("-" + kalshi_team_code(team))
            ]
            if len(matches) != 1:
                raise LockedHoldoutError(
                    f"{game_id} Kalshi ticker is not team-oriented: {ticker}"
                )
            ticker_pairs.append((matches[0], ticker))
        if {team for team, _ticker in ticker_pairs} != set(teams):
            raise LockedHoldoutError(
                f"{game_id} Kalshi moneyline does not cover both teams"
            )
        rule_sha256 = _sha256_bytes(
            _canonical_bytes(
                {
                    "family": "moneyline",
                    "game_id": game_id,
                    "kalshi_event_ticker": kalshi_ref["event_ticker"],
                    "polymarket_condition_id": market["conditionId"],
                    "polymarket_question": market["question"],
                    "polymarket_sports_market_type": market[
                        "sportsMarketType"
                    ],
                }
            )
        )
        bindings.append(
            GameMarketBinding(
                game_id=game_id,
                away_team=away,
                home_team=home,
                scheduled_start_utc=evidence["scheduled_start_utc"],
                polymarket_event_id=poly_ref["event_id"],
                polymarket_condition_id=market["conditionId"],
                polymarket_market_id=market["id"],
                polymarket_outcomes=oriented,
                kalshi_event_ticker=kalshi_ref["event_ticker"],
                kalshi_team_tickers=tuple(sorted(ticker_pairs)),
                rule_sha256=rule_sha256,
            )
        )
    return tuple(bindings)


def _pbp_paths(data_root: Path) -> tuple[Path, Path]:
    pbp = sorted(
        (
            data_root / "var/raw/raw/source=nflverse/dataset=DS-NFLVERSE"
        ).glob("version=*/partition=season-2025/*.parquet")
    )
    participation = sorted(
        (
            data_root
            / "var/raw/raw/source=nflverse/"
            "dataset=DS-NFLVERSE-PARTICIPATION"
        ).glob("version=*/partition=season-2025/*.parquet")
    )
    if len(pbp) != 1 or len(participation) != 1:
        raise LockedHoldoutError("frozen 2025 PBP/participation object is absent")
    for path in (*pbp, *participation):
        if _sha256_file(path) != "sha256:" + path.stem:
            raise LockedHoldoutError(f"source hash differs from object name: {path}")
    return pbp[0], participation[0]


def _load_game_ledger(
    *,
    pbp_path: Path,
    participation_path: Path,
    game_id: str,
) -> tuple[X13GameLedgerV1, dict[str, dict[str, object]]]:
    pbp_rows = pq.read_table(
        pbp_path, filters=[("game_id", "=", game_id)]
    ).to_pylist()
    participation_rows = pq.read_table(
        participation_path,
        filters=[("nflverse_game_id", "=", game_id)],
    ).to_pylist()
    if not pbp_rows:
        raise LockedHoldoutError(f"no frozen PBP rows for {game_id}")
    ledger = build_x13_game_ledger(pbp_rows, participation_rows)
    def play_id(value: object) -> str:
        if type(value) is int:
            return str(value)
        if type(value) is float and math.isfinite(value) and value.is_integer():
            return str(int(value))
        if type(value) is str and value:
            return value
        raise LockedHoldoutError(f"invalid PBP play_id in {game_id}")

    rows_by_play = {play_id(row["play_id"]): row for row in pbp_rows}
    if len(rows_by_play) != len(pbp_rows):
        raise LockedHoldoutError(f"duplicate PBP play_id in {game_id}")
    return ledger, rows_by_play


def _event_time(
    ledger: X13GameLedgerV1,
    episode: FinalizedEpisodeV1,
) -> datetime | None:
    events = {event.play_id: event for event in ledger.events}
    if not episode.play_ids:
        return None
    event = events[episode.play_ids[0]]
    return (
        None
        if event.source_time_start_utc is None
        else _parse_utc(event.source_time_start_utc)
    )


def _factor_hits(
    *,
    ledger: X13GameLedgerV1,
    rows_by_play: Mapping[str, Mapping[str, object]],
    candidates: Sequence[LockedCandidate],
) -> list[dict[str, object]]:
    candidate_by_id = {item.factor_id: item for item in candidates}
    live = [
        episode
        for episode in ledger.episodes
        if not episode.audit_only
        and not episode.nullified
        and episode.episode_type != "administrative"
        and _event_time(ledger, episode) is not None
    ]
    live.sort(key=lambda episode: (_event_time(ledger, episode), episode.episode_id))
    next_time = {
        episode.episode_id: (
            None if index + 1 == len(live) else _event_time(ledger, live[index + 1])
        )
        for index, episode in enumerate(live)
    }
    last_score_team: str | None = None
    hits: list[dict[str, object]] = []
    for episode in live:
        row = rows_by_play[episode.play_ids[0]]
        focal_team = row.get("posteam")
        if type(focal_team) is not str or not focal_team:
            continue
        nullified = bool(episode.nullified or episode.audit_only)
        pass_play = bool(row.get("pass_attempt") == 1)
        rush_play = bool(row.get("rush_attempt") == 1)
        sack = bool(row.get("sack") == 1)
        first_down = bool(row.get("first_down") == 1)
        yards = row.get("yards_gained")
        yards_number = (
            float(yards)
            if type(yards) in {int, float} and math.isfinite(float(yards))
            else 0.0
        )
        score = (
            bool(row.get("touchdown") == 1)
            or row.get("field_goal_result") == "made"
            or bool(row.get("safety") == 1)
            or episode.score_points > 0
        ) and not nullified
        scoring_team = (
            episode.scoring_team
            or (row.get("defteam") if row.get("safety") == 1 else focal_team)
        )
        flags = {
            "NFL.EVENT.EXPLOSIVE_PLAY": (
                (pass_play and yards_number >= 20)
                or (rush_play and yards_number >= 10)
            ),
            "NFL.COMBO.THIRD_FOURTH_DOWN_FAILURE": (
                row.get("down") in {3, 4}
                and (pass_play or rush_play or sack)
                and not first_down
                and not score
                and not nullified
            ),
            "NFL.SEQUENCE.ANSWER_SCORE": (
                score
                and type(scoring_team) is str
                and last_score_team is not None
                and scoring_team != last_score_team
            ),
        }
        if score and type(scoring_team) is str:
            last_score_team = scoring_team
        event_time = _event_time(ledger, episode)
        assert event_time is not None
        for factor_id, hit in flags.items():
            if not hit or factor_id not in candidate_by_id:
                continue
            candidate = candidate_by_id[factor_id]
            hits.append(
                {
                    "factor_id": factor_id,
                    "factor_version": candidate.factor_version,
                    "support_kind": candidate.support_kind,
                    "family": candidate.family,
                    "horizon_seconds": candidate.horizon_seconds,
                    "game_id": episode.game_id,
                    "episode_id": episode.episode_id,
                    "play_id": episode.play_ids[0],
                    "focal_team": focal_team,
                    "event_source_time_utc": _utc_text(event_time),
                    "next_salient_time_utc": (
                        None
                        if next_time[episode.episode_id] is None
                        else _utc_text(next_time[episode.episode_id])
                    ),
                }
            )
    return hits


def _fetch_bytes(
    client: httpx.Client,
    *,
    url: str,
    params: Mapping[str, object],
) -> tuple[bytes, Mapping[str, str]]:
    response = client.get(
        url,
        params=params,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
    )
    response.raise_for_status()
    if response.headers.get("Content-Encoding") not in {None, "identity"}:
        raise LockedHoldoutError("capture response is content-encoded")
    return response.content, response.headers


def _cached_response(
    *,
    data_root: Path,
    project_root: Path,
    source: str,
    dataset: str,
    partition: str,
    url: str,
    params: Mapping[str, object],
) -> tuple[bytes, str] | None:
    manifests = sorted(
        (
            data_root
            / "var/raw/manifests"
            / f"source={source}"
            / f"dataset={dataset}"
            / "version=x13-locked-holdout-20260726"
            / f"partition={partition}"
        ).glob("*.manifest.json")
    )
    if not manifests:
        return None
    if len(manifests) != 1:
        raise LockedHoldoutError(
            f"cached request has multiple manifests: {partition}"
        )
    verified = read_verified_static_object(
        manifests[0],
        store_root=data_root / "var/raw",
        program_root=project_root,
    )
    manifest = verified.record.manifest
    request = manifest.source_request
    if (
        manifest.source_url != url
        or request.get("method") != "GET"
        or request.get("params") != dict(sorted(params.items()))
    ):
        raise LockedHoldoutError(
            f"cached request identity differs: {partition}"
        )
    return verified.object_bytes, manifest.manifest_sha256


def _preserve_response(
    *,
    data_root: Path,
    project_root: Path,
    source: str,
    dataset: str,
    partition: str,
    url: str,
    params: Mapping[str, object],
    cursor: str,
    coverage: str,
    raw: bytes,
    headers: Mapping[str, str],
) -> str:
    value = _strict_json(raw, label=f"{source}:{partition}")
    record = preserve_static_object(
        data_root / "var/raw",
        raw,
        program_root=project_root,
        source=source,
        dataset=dataset,
        version="x13-locked-holdout-20260726",
        partition=partition,
        extension="json",
        source_url=url,
        source_request={
            "method": "GET",
            "params": dict(sorted(params.items())),
            "headers": {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        },
        source_cursor=cursor,
        fetched_at=_utc_text(datetime.now(UTC)),
        coverage=coverage,
        etag=headers.get("etag"),
        last_modified=headers.get("last-modified"),
        media_type="application/json",
        schema_fingerprint=_schema_fingerprint(value),
        license_ref="O-001" if source == "polymarket" else "O-003",
        license_status="research_only",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"query_sha256": None, "source_object_refs": []},
    )
    return record.manifest.manifest_sha256


def _capture_polymarket(
    *,
    client: httpx.Client,
    data_root: Path,
    project_root: Path,
    binding: GameMarketBinding,
    start_ts: int,
    end_ts: int,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    rows: list[dict[str, object]] = []
    manifests: list[str] = []

    def capture(left: int, right: int) -> None:
        params = {
            "market": binding.polymarket_condition_id,
            "start": left,
            "end": right,
            "limit": POLY_LIMIT,
            "offset": 0,
        }
        partition = f"holdout-{binding.game_id}-moneyline-{left}-{right}"
        cached = _cached_response(
            data_root=data_root,
            project_root=project_root,
            source="polymarket",
            dataset="DS-POLYMARKET-PUBLIC",
            partition=partition,
            url=POLY_TRADES_URL,
            params=params,
        )
        headers: Mapping[str, str] = {}
        if cached is None:
            raw, headers = _fetch_bytes(
                client, url=POLY_TRADES_URL, params=params
            )
            cached_manifest = None
        else:
            raw, cached_manifest = cached
        value = _strict_json(raw, label=f"poly trades {binding.game_id}")
        if type(value) is not list or len(value) > POLY_LIMIT:
            raise LockedHoldoutError("Polymarket trades response is malformed")
        manifest = cached_manifest or _preserve_response(
                data_root=data_root,
                project_root=project_root,
                source="polymarket",
                dataset="DS-POLYMARKET-PUBLIC",
                partition=partition,
                url=POLY_TRADES_URL,
                params=params,
                cursor=f"start:{left};end:{right};offset:0",
                coverage=(
                    "locked holdout moneyline inclusive trade window;"
                    f"game={binding.game_id};start={left};end={right}"
                ),
                raw=raw,
                headers=headers,
            )
        manifests.append(manifest)
        for row in value:
            if (
                type(row) is not dict
                or row.get("conditionId") != binding.polymarket_condition_id
                or type(row.get("timestamp")) is not int
                or not left <= row["timestamp"] <= right
            ):
                raise LockedHoldoutError("Polymarket trade escaped frozen scope")
        if len(value) < POLY_LIMIT:
            rows.extend(value)
            return
        if left == right:
            raise LockedHoldoutError("Polymarket saturated one-second leaf")
        middle = (left + right) // 2
        capture(left, middle)
        capture(middle + 1, right)

    capture(start_ts, end_ts)
    return rows, tuple(sorted(set(manifests)))


def _capture_kalshi_ticker(
    *,
    client: httpx.Client,
    data_root: Path,
    project_root: Path,
    binding: GameMarketBinding,
    ticker: str,
    start_ts: int,
    end_ts: int,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    rows: list[dict[str, object]] = []
    manifests: list[str] = []
    cursor: str | None = None
    seen: set[str] = set()
    for page in range(100):
        params: dict[str, object] = {
            "limit": KALSHI_LIMIT,
            "ticker": ticker,
            "min_ts": start_ts,
            "max_ts": end_ts,
        }
        if cursor is not None:
            params["cursor"] = cursor
        partition = (
            f"holdout-{binding.game_id}-moneyline-{ticker}-p{page:03d}"
        )
        cached = _cached_response(
            data_root=data_root,
            project_root=project_root,
            source="kalshi",
            dataset="DS-KALSHI-HISTORICAL",
            partition=partition,
            url=KALSHI_TRADES_URL,
            params=params,
        )
        headers: Mapping[str, str] = {}
        if cached is None:
            raw, headers = _fetch_bytes(
                client, url=KALSHI_TRADES_URL, params=params
            )
            cached_manifest = None
        else:
            raw, cached_manifest = cached
        value = _strict_json(raw, label=f"kalshi trades {ticker}")
        if type(value) is not dict or set(value) != {"trades", "cursor"}:
            raise LockedHoldoutError("Kalshi trades page is malformed")
        items = value["trades"]
        next_cursor = value["cursor"]
        if type(items) is not list or type(next_cursor) is not str:
            raise LockedHoldoutError("Kalshi trades cursor types are invalid")
        manifest = cached_manifest or _preserve_response(
                data_root=data_root,
                project_root=project_root,
                source="kalshi",
                dataset="DS-KALSHI-HISTORICAL",
                partition=partition,
                url=KALSHI_TRADES_URL,
                params=params,
                cursor=f"ticker:{ticker};request_cursor:{cursor or 'ROOT'}",
                coverage=(
                    "locked holdout stable-ID moneyline trade cursor chain;"
                    f"game={binding.game_id};ticker={ticker}"
                ),
                raw=raw,
                headers=headers,
            )
        manifests.append(manifest)
        for item in items:
            if type(item) is not dict or item.get("ticker") != ticker:
                raise LockedHoldoutError("Kalshi trade escaped frozen ticker")
        rows.extend(items)
        if next_cursor == "":
            return rows, tuple(sorted(set(manifests)))
        if next_cursor == cursor or next_cursor in seen:
            raise LockedHoldoutError("Kalshi cursor repeated")
        seen.add(next_cursor)
        cursor = next_cursor
    raise LockedHoldoutError("Kalshi terminal cursor proof is missing")


def _normalise_trades(
    *,
    binding: GameMarketBinding,
    poly_rows: Sequence[Mapping[str, object]],
    kalshi_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[MarketObservation, ...]:
    poly_outcomes = dict(binding.polymarket_outcomes)
    observations: list[MarketObservation] = []
    for row in poly_rows:
        native = row.get("outcome")
        if type(native) is not str or native not in poly_outcomes:
            raise LockedHoldoutError("Polymarket trade outcome is unbound")
        observations.append(
            normalize_polymarket_trade(
                row,
                raw_market_id=binding.polymarket_condition_id,
                logical_market_id=f"polymarket:{binding.game_id}:winner",
                outcome=poly_outcomes[native],
            )
        )
    for team, ticker in binding.kalshi_team_tickers:
        for row in kalshi_rows[ticker]:
            observations.append(
                normalize_kalshi_trade(
                    row,
                    ticker=ticker,
                    outcome=team,
                    logical_market_id=f"kalshi:{binding.game_id}:winner",
                )
            )
    return deduplicate_observations(observations)


def _measure_hit(
    *,
    hit: Mapping[str, object],
    binding: GameMarketBinding,
    observations: Sequence[MarketObservation],
    market_source_hashes: Sequence[str],
    sports_source_hashes: Sequence[str],
) -> list[dict[str, object]]:
    event_start = _parse_utc(hit["event_source_time_utc"])  # type: ignore[arg-type]
    event_end = event_start + timedelta(seconds=1)
    deadline = event_end + timedelta(seconds=int(hit["horizon_seconds"]))
    next_salient = (
        None
        if hit["next_salient_time_utc"] is None
        else _parse_utc(hit["next_salient_time_utc"])  # type: ignore[arg-type]
    )
    focal_team = hit["focal_team"]
    rows: list[dict[str, object]] = []
    venue_ids = {
        "polymarket": binding.polymarket_condition_id,
        "kalshi": dict(binding.kalshi_team_tickers)[focal_team],
    }
    for venue, raw_market_id in venue_ids.items():
        material = [
            observation
            for observation in observations
            if observation.venue == venue
            and observation.outcome == focal_team
            and observation.price is not None
            and observation.primary_path_eligible
        ]
        pre = [
            item
            for item in material
            if item.source_interval.end <= event_start
        ]
        latest_pre_start = (
            None
            if not pre
            else max(item.source_interval.start for item in pre)
        )
        latest_pre = [
            item for item in pre if item.source_interval.start == latest_pre_start
        ]
        overlap = [
            item
            for item in material
            if item.source_interval.start <= event_end
            and item.source_interval.end > event_start
        ]
        post = [
            item
            for item in material
            if item.source_interval.start > event_end
            and item.source_interval.start <= deadline
        ]
        post.sort(key=lambda item: (item.source_interval.start, item.observation_id))
        same_second = len({item.source_interval.start for item in post}) != len(post)
        order_ambiguous = bool(len(latest_pre) > 1 or overlap or same_second)
        contaminated = next_salient is not None and next_salient <= deadline
        if len(latest_pre) != 1 or not post or order_ambiguous or contaminated:
            continue
        pre_price = latest_pre[0].price
        assert pre_price is not None
        last_start = max(item.source_interval.start for item in post)
        last_items = [item for item in post if item.source_interval.start == last_start]
        first_start = min(item.source_interval.start for item in post)
        first_items = [
            item for item in post if item.source_interval.start == first_start
        ]
        if len(first_items) != 1 or len(last_items) != 1:
            continue
        first_price = first_items[0].price
        last_price = last_items[0].price
        assert first_price is not None and last_price is not None
        volume = sum((item.size for item in post), Decimal(0))
        if volume <= 0:
            continue
        vwap = sum(
            (item.price * item.size for item in post if item.price is not None),
            Decimal(0),
        ) / volume
        deltas = [
            item.price - pre_price for item in post if item.price is not None
        ]
        maximum_magnitude = max(abs(delta) for delta in deltas)
        extrema = {delta for delta in deltas if abs(delta) == maximum_magnitude}
        if len(extrema) != 1:
            continue
        maximum_excursion = next(iter(extrema))
        rows.append(
            {
                **hit,
                "contract_id": f"{venue}:{raw_market_id}",
                "logical_market_id": f"{venue}:{binding.game_id}:winner",
                "outcome": focal_team,
                "venue": venue,
                "pre_event_actual_trade": float(pre_price),
                "first_post_event_trade": float(first_price),
                "horizon_vwap": float(vwap),
                "horizon_last_trade": float(last_price),
                "signed_price_change": float(last_price - pre_price),
                "maximum_excursion": float(maximum_excursion),
                "trade_count": len(post),
                "volume": float(volume),
                "staleness_seconds": (
                    deadline - last_start
                ).total_seconds(),
                "actual_observation": True,
                "forward_filled": False,
                "order_ambiguous": False,
                "contaminated": False,
                "rule_sha256": binding.rule_sha256,
                "sports_source_hashes": tuple(sorted(sports_source_hashes)),
                "market_source_hashes": tuple(sorted(market_source_hashes)),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def _publish(
    *,
    output_root: Path,
    observations: pd.DataFrame,
    results: pd.DataFrame,
    capture_audit: Mapping[str, object],
    shortlist_sha256: str,
    holdout_selection_id: str,
    source_hashes: Sequence[str],
) -> LockedHoldoutPublication:
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".x13-locked-holdout-", dir=output_root))
    try:
        observation_path = stage / "holdout_factor_observations_v1.parquet"
        result_path = stage / "holdout_results_v1.parquet"
        audit_path = stage / "capture_audit_v1.json"
        observations.to_parquet(observation_path, index=False)
        results.to_parquet(result_path, index=False)
        audit_path.write_bytes(_canonical_bytes(capture_audit) + b"\n")
        files = {
            path.name: {
                "byte_length": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in (observation_path, result_path, audit_path)
        }
        material = {
            "schema": "nfl_x13_locked_historical_holdout_bundle_v1",
            "experiment_id": "X-13",
            "shortlist_sha256": shortlist_sha256,
            "holdout_selection_id": holdout_selection_id,
            "factor_ids": sorted(LOCKED_FACTOR_IDS),
            "game_ids": sorted(observations["game_id"].unique()),
            "observation_count": len(observations),
            "result_count": len(results),
            "source_hashes": sorted(set(source_hashes)),
            "files": files,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        bundle_sha256 = _sha256_bytes(_canonical_bytes(material))
        manifest = {
            **material,
            "bundle_sha256": bundle_sha256,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
        target = output_root / bundle_sha256.replace(":", "-")
        if target.exists():
            existing = target / "manifest.json"
            if not existing.is_file() or existing.read_bytes() != manifest_path.read_bytes():
                raise LockedHoldoutError("immutable holdout publication conflicts")
        else:
            os.rename(stage, target)
            stage = target
        return LockedHoldoutPublication(
            manifest_path=target / "manifest.json",
            bundle_sha256=bundle_sha256,
            observation_path=target / observation_path.name,
            result_path=target / result_path.name,
            observation_count=len(observations),
            result_count=len(results),
        )
    finally:
        if stage.exists() and stage.parent == output_root and stage.name.startswith("."):
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()


def run_locked_holdout(
    *,
    project_root: str | Path,
    shortlist_manifest: str | Path,
    output_root: str | Path | None = None,
) -> LockedHoldoutPublication:
    root = Path(project_root).resolve()
    shortlist_path = Path(shortlist_manifest).resolve()
    candidates, registry, shortlist_sha256 = _load_shortlist(shortlist_path)
    selection, selection_id = _load_holdout_selection(root)
    evidence = _load_evidence_index(root)
    selected_games = tuple(selection["selected_game_ids"])
    evidence_games = {game["game_id"] for game in evidence["games"]}
    if set(selected_games) != evidence_games:
        raise LockedHoldoutError("evidence and frozen holdout games differ")
    bindings = _build_bindings(
        project_root=root,
        selected_game_ids=selected_games,
        evidence_index=evidence,
    )
    data_root = _data_root(root)
    pbp_path, participation_path = _pbp_paths(data_root)
    sports_hashes = (_sha256_file(pbp_path), _sha256_file(participation_path))
    all_rows: list[dict[str, object]] = []
    audit_games: list[dict[str, object]] = []
    capture_hashes: set[str] = set()
    timeout = httpx.Timeout(60.0, connect=20.0)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for binding in bindings:
            ledger, rows_by_play = _load_game_ledger(
                pbp_path=pbp_path,
                participation_path=participation_path,
                game_id=binding.game_id,
            )
            hits = _factor_hits(
                ledger=ledger,
                rows_by_play=rows_by_play,
                candidates=candidates,
            )
            event_times = [
                _parse_utc(event.source_time_start_utc)
                for event in ledger.events
                if event.source_time_start_utc is not None
            ]
            start_ts = int(min(event_times).timestamp())
            end_ts = int(max(event_times).timestamp())
            poly_rows, poly_hashes = _capture_polymarket(
                client=client,
                data_root=data_root,
                project_root=root,
                binding=binding,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            kalshi_by_ticker: dict[str, list[dict[str, object]]] = {}
            kalshi_hashes: set[str] = set()
            for _team, ticker in binding.kalshi_team_tickers:
                rows, hashes = _capture_kalshi_ticker(
                    client=client,
                    data_root=data_root,
                    project_root=root,
                    binding=binding,
                    ticker=ticker,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                kalshi_by_ticker[ticker] = rows
                kalshi_hashes.update(hashes)
            source_hashes = tuple(sorted({*poly_hashes, *kalshi_hashes}))
            capture_hashes.update(source_hashes)
            observations = _normalise_trades(
                binding=binding,
                poly_rows=poly_rows,
                kalshi_rows=kalshi_by_ticker,
            )
            game_rows: list[dict[str, object]] = []
            for hit in hits:
                game_rows.extend(
                    _measure_hit(
                        hit=hit,
                        binding=binding,
                        observations=observations,
                        market_source_hashes=source_hashes,
                        sports_source_hashes=sports_hashes,
                    )
                )
            all_rows.extend(game_rows)
            audit_games.append(
                {
                    "game_id": binding.game_id,
                    "ledger_sha256": ledger.artifact_sha256,
                    "episode_count": len(ledger.episodes),
                    "factor_hit_count": len(hits),
                    "polymarket_trade_count": len(poly_rows),
                    "kalshi_trade_count": sum(
                        len(rows) for rows in kalshi_by_ticker.values()
                    ),
                    "strict_factor_observation_count": len(game_rows),
                    "capture_manifest_sha256s": list(source_hashes),
                }
            )
    observations = pd.DataFrame(all_rows)
    if observations.empty:
        raise LockedHoldoutError("holdout produced no strict factor observation")
    key_columns = ["factor_id", "family", "horizon_seconds"]
    observed_keys = set(
        observations[key_columns].itertuples(index=False, name=None)
    )
    expected_keys = {
        (item.factor_id, item.family, item.horizon_seconds)
        for item in candidates
    }
    if observed_keys != expected_keys:
        raise LockedHoldoutError(
            "holdout observations do not cover every exact locked candidate"
        )
    discovery_games = X13_GAME_IDS
    results = run_locked_historical_holdout(
        shortlist_manifest=shortlist_path,
        holdout_observations=observations,
        factor_registry=registry,
        holdout_selection_id=selection_id,
        holdout_game_ids=selected_games,
        discovery_game_ids=discovery_games,
        bootstrap_samples=10_000,
        seed=20260725,
    )
    audit = {
        "schema": "nfl_x13_locked_holdout_capture_audit_v1",
        "shortlist_sha256": shortlist_sha256,
        "holdout_selection_id": selection_id,
        "game_count": len(bindings),
        "games": audit_games,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return _publish(
        output_root=(
            Path(output_root)
            if output_root is not None
            else root
            / "artifacts/market-observation/nfl/x13/"
            "historical-holdout-results"
        ),
        observations=observations,
        results=results,
        capture_audit=audit,
        shortlist_sha256=shortlist_sha256,
        holdout_selection_id=selection_id,
        source_hashes=(
            shortlist_sha256,
            selection_id,
            *sports_hashes,
            *sorted(capture_hashes),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact shortlist-locked X-13 historical holdout."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--shortlist-manifest", required=True)
    parser.add_argument("--output-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_locked_holdout(
        project_root=args.project_root,
        shortlist_manifest=args.shortlist_manifest,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                **asdict(result),
                "manifest_path": str(result.manifest_path),
                "observation_path": str(result.observation_path),
                "result_path": str(result.result_path),
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
