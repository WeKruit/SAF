"""Bounded public market recorder; never fabricates capture evidence."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from prediction_market.recorder_supervisor import (
    SupervisorResult,
    build_polymarket_health_report,
    supervise_polymarket,
)


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
    run_seconds: float,
    max_frames: int | None,
    receive_timeout_seconds: float,
    max_reconnects: int | None = None,
) -> SupervisorResult:
    """Run one real-time-bounded public supervisor with durable segments."""

    connector = _public_connector(open_timeout=min(run_seconds, 10.0))
    return await supervise_polymarket(
        connector,
        asset_ids,
        raw_root,
        run_seconds=run_seconds,
        max_frames=max_frames,
        max_reconnects=max_reconnects,
        receive_timeout_seconds=receive_timeout_seconds,
    )


def _report(
    result: SupervisorResult,
    *,
    raw_root: Path,
    market_limit: int,
    max_assets: int,
) -> dict[str, object]:
    report = build_polymarket_health_report(
        result,
        raw_root=raw_root,
    )
    report["configuration"] = {
        "discover_sports": True,
        "market_limit": market_limit,
        "max_assets": max_assets,
        **report["configuration"],
    }
    return report


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
        timeout_seconds=min(args.run_seconds, 15.0),
    )
    if not assets:
        raise DiscoveryError(
            "no active sports markets with public CLOB asset IDs were discovered"
        )
    result = await capture_public_polymarket(
        assets,
        args.raw_root,
        run_seconds=args.run_seconds,
        max_frames=args.max_frames,
        receive_timeout_seconds=args.receive_timeout_seconds,
        max_reconnects=args.max_reconnects,
    )
    report = _report(
        result,
        raw_root=args.raw_root,
        market_limit=args.market_limit,
        max_assets=args.max_assets,
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
    polymarket.add_argument("--run-seconds", type=_positive_float, required=True)
    polymarket.add_argument("--max-frames", type=_positive_integer)
    polymarket.add_argument("--max-assets", type=_positive_integer, default=20)
    polymarket.add_argument("--market-limit", type=_positive_integer, default=100)
    polymarket.add_argument("--max-reconnects", type=int)
    polymarket.add_argument(
        "--receive-timeout-seconds",
        type=_positive_float,
        default=30.0,
    )
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
