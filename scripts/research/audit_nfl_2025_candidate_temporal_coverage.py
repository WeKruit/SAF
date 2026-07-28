"""Audit reaction-blind temporal coverage for NFL 2025 candidate games.

This audit uses only source timestamps and stable market identities.  It does
not retain or aggregate score, market price, size, volume, or price direction.
Its purpose is to prove whether an event-time study is observable before any
factor shortlist is exposed to candidate-game reactions.
"""

from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import httpx
import pandas as pd


POLY_TRADES_URL = "https://data-api.polymarket.com/trades"
KALSHI_TRADES_URL = (
    "https://api.elections.kalshi.com/trade-api/v2/historical/trades"
)
POLY_LIMIT = 10_000
KALSHI_LIMIT = 1_000
HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60)


class CoverageAuditError(RuntimeError):
    """A coverage input or public API response failed closed."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise CoverageAuditError("timestamp must be a nonempty UTC string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise CoverageAuditError("timestamp must carry timezone")
    return parsed.astimezone(timezone.utc)


def _source_events(
    pbp_path: Path,
    game_ids: set[str],
) -> dict[str, dict[str, object]]:
    frame = pd.read_parquet(
        pbp_path,
        columns=["game_id", "play_id", "time_of_day"],
        filters=[("game_id", "in", sorted(game_ids))],
    )
    if set(frame["game_id"].unique()) != game_ids:
        missing = sorted(game_ids - set(frame["game_id"].unique()))
        raise CoverageAuditError(f"candidate PBP games are missing: {missing}")
    output: dict[str, dict[str, object]] = {}
    for game_id, group in frame.groupby("game_id", sort=True):
        source_times = pd.to_datetime(
            group["time_of_day"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        observed = sorted(
            {
                int(value.timestamp())
                for value in source_times.dropna().dt.floor("s").tolist()
            }
        )
        if not observed:
            raise CoverageAuditError(f"{game_id} has no source-time events")
        output[str(game_id)] = {
            "raw_row_count": int(len(group)),
            "source_time_row_count": int(source_times.notna().sum()),
            "source_time_coverage_rate": float(source_times.notna().mean()),
            "event_seconds": observed,
            "first_event_second": observed[0],
            "last_event_second": observed[-1],
        }
    return output


async def _request_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    url: str,
    params: dict[str, object],
) -> object:
    last_status: int | None = None
    for attempt in range(10):
        async with semaphore:
            try:
                response = await client.get(url, params=params)
                last_status = response.status_code
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("retry-after")
                    delay = (
                        float(retry_after)
                        if retry_after is not None
                        else 0.75 * (2**attempt)
                    )
                else:
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
                if attempt == 9:
                    raise CoverageAuditError(
                        f"public API request failed: {url}; status={last_status}"
                    ) from error
                delay = min(0.75 * (2**attempt), 30.0)
        await asyncio.sleep(min(delay, 30.0))
    raise CoverageAuditError(
        f"public API retry budget exhausted: {url}; status={last_status}"
    )


async def _poly_trade_seconds(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    condition_id: str,
    left: int,
    right: int,
) -> list[int]:
    payload = await _request_json(
        client,
        semaphore,
        url=POLY_TRADES_URL,
        params={
            "market": condition_id,
            "start": left,
            "end": right,
            "limit": POLY_LIMIT,
            "offset": 0,
        },
    )
    if not isinstance(payload, list) or len(payload) > POLY_LIMIT:
        raise CoverageAuditError("Polymarket trade response is malformed")
    seconds: list[int] = []
    for row in payload:
        if (
            not isinstance(row, dict)
            or row.get("conditionId") != condition_id
            or not isinstance(row.get("timestamp"), int)
            or not left <= row["timestamp"] <= right
        ):
            raise CoverageAuditError("Polymarket trade escaped frozen scope")
        seconds.append(int(row["timestamp"]))
    if len(payload) < POLY_LIMIT:
        return seconds
    if left == right:
        raise CoverageAuditError("Polymarket saturated a one-second leaf")
    middle = (left + right) // 2
    earlier, later = await asyncio.gather(
        _poly_trade_seconds(
            client,
            semaphore,
            condition_id=condition_id,
            left=left,
            right=middle,
        ),
        _poly_trade_seconds(
            client,
            semaphore,
            condition_id=condition_id,
            left=middle + 1,
            right=right,
        ),
    )
    return earlier + later


async def _kalshi_trade_seconds(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    ticker: str,
    left: int,
    right: int,
) -> list[int]:
    seconds: list[int] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _page in range(100):
        params: dict[str, object] = {
            "ticker": ticker,
            "limit": KALSHI_LIMIT,
            "min_ts": left,
            "max_ts": right,
        }
        if cursor is not None:
            params["cursor"] = cursor
        payload = await _request_json(
            client,
            semaphore,
            url=KALSHI_TRADES_URL,
            params=params,
        )
        if not isinstance(payload, dict) or set(payload) != {"trades", "cursor"}:
            raise CoverageAuditError("Kalshi trade response is malformed")
        rows = payload["trades"]
        next_cursor = payload["cursor"]
        if not isinstance(rows, list) or not isinstance(next_cursor, str):
            raise CoverageAuditError("Kalshi cursor response is malformed")
        for row in rows:
            if not isinstance(row, dict) or row.get("ticker") != ticker:
                raise CoverageAuditError("Kalshi trade escaped frozen ticker")
            created = _parse_utc(row.get("created_time"))
            second = int(created.replace(microsecond=0).timestamp())
            if not left <= second <= right:
                raise CoverageAuditError("Kalshi trade escaped frozen time window")
            seconds.append(second)
        if next_cursor == "":
            return seconds
        if next_cursor == cursor or next_cursor in seen:
            raise CoverageAuditError("Kalshi cursor repeated")
        seen.add(next_cursor)
        cursor = next_cursor
    raise CoverageAuditError("Kalshi terminal cursor proof is missing")


def _events_with_post_trade(
    event_seconds: Iterable[int],
    trade_seconds: list[int],
    horizon: int,
) -> int:
    ordered = sorted(set(trade_seconds))
    count = 0
    for event_second in event_seconds:
        event_end = event_second + 1
        index = bisect_right(ordered, event_end)
        if index < len(ordered) and ordered[index] <= event_end + horizon:
            count += 1
    return count


async def _audit_games(
    games: list[dict[str, object]],
    sports: dict[str, dict[str, object]],
    *,
    concurrency: int,
    checkpoint_root: Path,
) -> list[dict[str, object]]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "SAF-X13-reaction-blind-temporal-coverage/1.0",
    }
    timeout = httpx.Timeout(45.0, connect=10.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=False,
    ) as client:
        async def audit(game: dict[str, object]) -> dict[str, object]:
            game_id = str(game["game_id"])
            checkpoint = checkpoint_root / f"{game_id}.json"
            if checkpoint.exists():
                cached = json.loads(checkpoint.read_bytes())
                if (
                    isinstance(cached, dict)
                    and cached.get("schema")
                    == "nfl_candidate_temporal_game_checkpoint_v1"
                    and cached.get("builder_version")
                    == "v1-mixed-iso-source-time"
                    and cached.get("game_id") == game_id
                    and isinstance(cached.get("result"), dict)
                ):
                    return cached["result"]
                raise CoverageAuditError(
                    f"{game_id} checkpoint is malformed or stale"
                )
            sports_row = sports[game_id]
            left = int(sports_row["first_event_second"]) - 300
            right = int(sports_row["last_event_second"]) + 300
            poly = game.get("polymarket_identity")
            kalshi = game.get("kalshi_identity")
            if not isinstance(poly, dict) or not isinstance(kalshi, dict):
                raise CoverageAuditError(f"{game_id} lacks exact venue identity")
            condition_id = poly.get("moneyline_condition_id")
            tickers = kalshi.get("market_refs")
            if not isinstance(condition_id, str) or not isinstance(tickers, list):
                raise CoverageAuditError(f"{game_id} venue identity is malformed")
            poly_task = _poly_trade_seconds(
                client,
                semaphore,
                condition_id=condition_id,
                left=left,
                right=right,
            )
            kalshi_tasks = [
                _kalshi_trade_seconds(
                    client,
                    semaphore,
                    ticker=str(ticker),
                    left=left,
                    right=right,
                )
                for ticker in tickers
            ]
            fetched = await asyncio.gather(poly_task, *kalshi_tasks)
            poly_seconds = fetched[0]
            kalshi_by_ticker = {
                str(ticker): fetched[index + 1]
                for index, ticker in enumerate(tickers)
            }
            kalshi_seconds = [
                second
                for values in kalshi_by_ticker.values()
                for second in values
            ]
            event_seconds = list(sports_row["event_seconds"])
            horizon_rows: dict[str, dict[str, object]] = {}
            for horizon in HORIZONS_SECONDS:
                poly_count = _events_with_post_trade(
                    event_seconds,
                    poly_seconds,
                    horizon,
                )
                kalshi_count = _events_with_post_trade(
                    event_seconds,
                    kalshi_seconds,
                    horizon,
                )
                horizon_rows[str(horizon)] = {
                    "polymarket_event_count_with_post_trade": poly_count,
                    "polymarket_event_coverage_rate": (
                        poly_count / len(event_seconds)
                    ),
                    "kalshi_event_count_with_post_trade": kalshi_count,
                    "kalshi_event_coverage_rate": (
                        kalshi_count / len(event_seconds)
                    ),
                }
            result = {
                "game_id": game_id,
                "week": int(game["week"]),
                "source_time_raw_row_count": sports_row["raw_row_count"],
                "source_time_observed_row_count": sports_row[
                    "source_time_row_count"
                ],
                "source_time_coverage_rate": sports_row[
                    "source_time_coverage_rate"
                ],
                "unique_source_event_second_count": len(event_seconds),
                "first_event_utc": datetime.fromtimestamp(
                    int(sports_row["first_event_second"]),
                    tz=timezone.utc,
                ).isoformat(),
                "last_event_utc": datetime.fromtimestamp(
                    int(sports_row["last_event_second"]),
                    tz=timezone.utc,
                ).isoformat(),
                "polymarket_in_window_trade_count": len(poly_seconds),
                "kalshi_in_window_trade_count": len(kalshi_seconds),
                "kalshi_in_window_trade_count_by_ticker": {
                    ticker: len(values)
                    for ticker, values in sorted(kalshi_by_ticker.items())
                },
                "horizon_coverage": horizon_rows,
            }
            checkpoint_payload = {
                "schema": "nfl_candidate_temporal_game_checkpoint_v1",
                "builder_version": "v1-mixed-iso-source-time",
                "game_id": game_id,
                "result": result,
            }
            _atomic_write(checkpoint, _canonical_bytes(checkpoint_payload))
            return result

        completed = await asyncio.gather(
            *(audit(game) for game in games),
            return_exceptions=True,
        )
        failures = [
            item for item in completed if isinstance(item, BaseException)
        ]
        if failures:
            examples = "; ".join(str(item) for item in failures[:3])
            raise CoverageAuditError(
                f"{len(failures)} game audits failed after checkpointing "
                f"successful games: {examples}"
            )
        return [item for item in completed if isinstance(item, dict)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", type=Path, required=True)
    parser.add_argument("--candidate-scan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--scope",
        choices=("priority", "all-dual"),
        default="priority",
    )
    args = parser.parse_args()

    candidate_scan_bytes = args.candidate_scan.read_bytes()
    candidate_scan = json.loads(candidate_scan_bytes)
    if candidate_scan.get("selection_basis") != (
        "metadata_only_no_score_price_volume_or_reaction"
    ):
        raise CoverageAuditError("candidate scan is not reaction-blind")
    games = candidate_scan.get("games")
    priority_ids = candidate_scan.get("priority_game_ids")
    if not isinstance(priority_ids, list) or not isinstance(games, list):
        raise CoverageAuditError("candidate scan is malformed")
    selected_ids = (
        {
            str(game["game_id"])
            for game in games
            if isinstance(game, dict)
            and game.get("dual_venue_metadata") is True
            and isinstance(game.get("game_id"), str)
        }
        if args.scope == "all-dual"
        else {str(game_id) for game_id in priority_ids}
    )
    selected = [
        game
        for game in games
        if isinstance(game, dict) and str(game.get("game_id")) in selected_ids
    ]
    if len(selected) != len(selected_ids):
        raise CoverageAuditError("candidate scan does not carry every identity")
    sports = _source_events(args.pbp, selected_ids)
    audited = asyncio.run(
        _audit_games(
            selected,
            sports,
            concurrency=args.concurrency,
            checkpoint_root=args.output_root / ".work" / "game-checkpoints",
        )
    )
    audited.sort(key=lambda row: str(row["game_id"]))

    output = {
        "schema": "nfl_2025_candidate_temporal_coverage_v1",
        "builder_version": "v1-mixed-iso-source-time",
        "scope": args.scope,
        "selection_basis": (
            "source_timestamp_and_trade_presence_only;"
            "no_score_price_size_volume_or_direction_retained"
        ),
        "candidate_scan_sha256": "sha256:"
        + hashlib.sha256(candidate_scan_bytes).hexdigest(),
        "horizons_seconds": list(HORIZONS_SECONDS),
        "candidate_game_count": len(audited),
        "games": audited,
    }
    encoded = _canonical_bytes(output)
    object_sha = hashlib.sha256(encoded).hexdigest()
    object_path = (
        args.output_root
        / "objects"
        / "sha256"
        / object_sha[:2]
        / f"{object_sha}.json"
    )
    manifest = {
        "schema": "nfl_2025_candidate_temporal_coverage_manifest_v1",
        "builder_version": output["builder_version"],
        "scope": args.scope,
        "object_path": str(object_path),
        "object_sha256": f"sha256:{object_sha}",
        "byte_length": len(encoded),
        "candidate_game_count": len(audited),
        "selection_basis": output["selection_basis"],
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = (
        args.output_root
        / "manifests"
        / f"{manifest_sha}.manifest.json"
    )
    _atomic_write(object_path, encoded)
    _atomic_write(manifest_path, manifest_bytes)
    print(
        json.dumps(
            {
                **manifest,
                "manifest_path": str(manifest_path),
                "manifest_sha256": f"sha256:{manifest_sha}",
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
