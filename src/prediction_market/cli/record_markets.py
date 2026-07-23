"""Bounded public market recorder; never fabricates capture evidence."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect as websocket_connect

from prediction_market.adapters.polymarket import (
    MARKET_WS_URL,
    DiscoveryError,
    discover_active_sports_assets,
)
from prediction_market.recording import CaptureResult, record_polymarket_with_reconnect


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


async def _market_heartbeat(websocket: Any) -> None:
    while True:
        await asyncio.sleep(10.0)
        await websocket.send("PING")


def _public_connector(*, open_timeout: float):
    @asynccontextmanager
    async def connection() -> AsyncIterator[Any]:
        async with websocket_connect(
            MARKET_WS_URL,
            open_timeout=open_timeout,
            close_timeout=5.0,
            ping_interval=None,
            max_size=16 * 1024 * 1024,
        ) as websocket:
            heartbeat = asyncio.create_task(_market_heartbeat(websocket))
            try:
                yield websocket
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat

    return connection


async def capture_public_polymarket(
    asset_ids: Sequence[str],
    raw_root: Path,
    *,
    max_frames: int,
    timeout_seconds: float,
    max_reconnects: int = 2,
) -> CaptureResult:
    """Run one overall-time-bounded public capture with bounded reconnects."""

    if not 0 <= max_reconnects <= 3:
        raise ValueError("max_reconnects must be between zero and three")
    backoff = (0.25, 0.5, 1.0)[:max_reconnects]
    connector = _public_connector(open_timeout=min(timeout_seconds, 10.0))
    async with asyncio.timeout(timeout_seconds):
        return await record_polymarket_with_reconnect(
            connector,
            asset_ids,
            raw_root,
            max_frames=max_frames,
            max_reconnects=max_reconnects,
            backoff_seconds=backoff,
            receive_timeout_seconds=min(timeout_seconds, 15.0),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _relative_artifact_path(path: Path, raw_root: Path) -> str:
    return path.resolve(strict=True).relative_to(raw_root.resolve(strict=True)).as_posix()


def _report(
    result: CaptureResult,
    *,
    asset_ids: Sequence[str],
    raw_root: Path,
    market_limit: int,
    max_assets: int,
    max_frames: int,
    max_reconnects: int,
    timeout_seconds: float,
) -> dict[str, object]:
    return {
        "report_version": "v0",
        "evidence_scope": "operational_observation_not_x08_evidence",
        "source": "polymarket",
        "endpoint": MARKET_WS_URL,
        "subscription": {
            "channel": "market",
            "asset_ids": list(asset_ids),
        },
        "configuration": {
            "discover_sports": True,
            "market_limit": market_limit,
            "max_assets": max_assets,
            "max_frames": max_frames,
            "max_reconnects": max_reconnects,
            "timeout_seconds": timeout_seconds,
        },
        "complete": result.complete,
        "terminal_reason": result.terminal_reason,
        "counters": {
            "frames": result.counters.frames,
            "parse_errors": result.counters.parse_errors,
            "reconnects": result.counters.reconnects,
            "gaps": result.counters.gaps,
            "continuity_unknown": result.counters.continuity_unknown,
            "out_of_order": result.counters.out_of_order,
        },
        "segments": [
            {
                "capture_session_id": manifest.capture_session_id,
                "manifest_path": _relative_artifact_path(manifest.path, raw_root),
                "manifest_sha256": _sha256_file(manifest.path),
                "object_path": _relative_artifact_path(manifest.object_path, raw_root),
                "object_sha256": manifest.object_sha256,
                "record_count": manifest.record_count,
            }
            for manifest in result.manifests
        ],
    }


def _write_report(report: dict[str, object], output: Path | None) -> None:
    rendered = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2 if output is not None else None,
            separators=None if output is not None else (",", ":"),
        )
        + "\n"
    )
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"wrote {output}")


async def _run_polymarket(args: argparse.Namespace) -> tuple[int, dict[str, object] | None]:
    if not args.discover_sports:
        raise ValueError("polymarket requires --discover-sports")
    assets = await discover_active_sports_assets(
        market_limit=args.market_limit,
        max_assets=args.max_assets,
        timeout_seconds=min(args.timeout_seconds, 15.0),
    )
    if not assets:
        raise DiscoveryError(
            "no active sports markets with public CLOB asset IDs were discovered"
        )
    result = await capture_public_polymarket(
        assets,
        args.raw_root,
        max_frames=args.max_frames,
        timeout_seconds=args.timeout_seconds,
        max_reconnects=args.max_reconnects,
    )
    report = _report(
        result,
        asset_ids=assets,
        raw_root=args.raw_root,
        market_limit=args.market_limit,
        max_assets=args.max_assets,
        max_frames=args.max_frames,
        max_reconnects=args.max_reconnects,
        timeout_seconds=args.timeout_seconds,
    )
    return (0 if result.complete else 2), report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record-markets",
        description="Capture real public venue frames into immutable raw segments.",
    )
    subparsers = parser.add_subparsers(dest="venue", required=True)
    polymarket = subparsers.add_parser(
        "polymarket", help="capture the unauthenticated public market channel"
    )
    polymarket.add_argument("--discover-sports", action="store_true")
    polymarket.add_argument("--max-frames", type=_positive_integer, required=True)
    polymarket.add_argument("--max-assets", type=_positive_integer, default=20)
    polymarket.add_argument("--market-limit", type=_positive_integer, default=100)
    polymarket.add_argument("--max-reconnects", type=int, default=2)
    polymarket.add_argument("--timeout-seconds", type=_positive_float, default=45.0)
    polymarket.add_argument("--raw-root", type=Path, required=True)
    polymarket.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        exit_code, report = asyncio.run(_run_polymarket(args))
    except (DiscoveryError, OSError, TimeoutError, ValueError) as exc:
        print(f"record-markets failed: {exc}", file=sys.stderr)
        return 2
    if report is not None:
        _write_report(report, args.output)
    if exit_code != 0:
        print(
            f"record-markets failed: {report['terminal_reason'] if report else 'unknown'}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
