"""Bounded supervisor for prospective Polymarket public WebSocket capture."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prediction_market.adapters.base import (
    CaptureCounters,
    ProtocolError,
    WebSocketTransport,
    exact_frame_bytes,
    utc_now_text,
)
from prediction_market.adapters.polymarket import (
    MARKET_WS_URL,
    build_market_subscription,
    parse_market_frame,
)
from prediction_market.raw_store import (
    PartitionBoundaryError,
    RawSegmentWriter,
    RawStoreError,
    SegmentManifest,
    verify_segment,
)


Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], str]
_X08_REQUIRED_ELAPSED_DAYS = 7


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    """Measured outcome from one genuinely elapsed, bounded supervisor run."""

    manifests: tuple[SegmentManifest, ...]
    counters: CaptureCounters
    connection_attempts: int
    complete: bool
    terminal_reason: str
    started_at: str
    finished_at: str
    requested_run_seconds: float
    asset_ids: tuple[str, ...]
    max_frames: int | None
    max_reconnects: int | None
    reconnect_backoff_seconds: tuple[float, ...]
    receive_timeout_seconds: float
    observed_elapsed_seconds: float
    connected_elapsed_seconds: float
    required_elapsed_days: int
    observed_elapsed_days: float
    duration_gate_met: bool
    formal_x08_result: bool = False

    @property
    def uptime_ratio(self) -> float:
        if self.observed_elapsed_seconds <= 0:
            return 0.0
        return min(
            1.0,
            max(0.0, self.connected_elapsed_seconds)
            / self.observed_elapsed_seconds,
        )


def _positive_seconds(value: float, field: str) -> float:
    if type(value) not in {int, float} or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def _frame_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError("max_frames must be a positive integer or null")
    return value


def _reconnect_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("max_reconnects must be a non-negative integer or null")
    return value


def _backoff_schedule(values: Sequence[float]) -> tuple[float, ...]:
    schedule = tuple(values)
    if not schedule:
        raise ValueError("reconnect_backoff_seconds must not be empty")
    if any(
        type(value) not in {int, float} or value < 0 for value in schedule
    ):
        raise ValueError(
            "reconnect_backoff_seconds must contain non-negative numbers"
        )
    return tuple(float(value) for value in schedule)


def _terminal_reason(exc: Exception) -> str:
    detail = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _new_writer(raw_root: str | Path) -> RawSegmentWriter:
    return RawSegmentWriter(
        raw_root,
        source="polymarket",
        stream="market",
    )


async def supervise_polymarket(
    connect: Callable[[], Any],
    asset_ids: Sequence[str],
    raw_root: str | Path,
    *,
    run_seconds: float,
    max_frames: int | None = None,
    max_reconnects: int | None = None,
    reconnect_backoff_seconds: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 5.0),
    receive_timeout_seconds: float = 30.0,
    receive_clock: Clock = utc_now_text,
    sleep: Sleep = asyncio.sleep,
) -> SupervisorResult:
    """Capture until a real deadline or explicit frame cap, sealing on every break.

    Every reconnect begins a new segment.  Missing traffic is never synthesized,
    and every frame is durably staged before protocol parsing.
    """

    requested_seconds = _positive_seconds(run_seconds, "run_seconds")
    timeout_seconds = _positive_seconds(
        receive_timeout_seconds, "receive_timeout_seconds"
    )
    frame_limit = _frame_limit(max_frames)
    reconnect_limit = _reconnect_limit(max_reconnects)
    backoffs = _backoff_schedule(reconnect_backoff_seconds)
    subscription = build_market_subscription(asset_ids)
    assets = tuple(asset_ids)

    loop = asyncio.get_running_loop()
    started_monotonic = loop.time()
    deadline = started_monotonic + requested_seconds
    started_at = utc_now_text()
    manifests: list[SegmentManifest] = []
    frames = 0
    parse_errors = 0
    reconnects = 0
    gaps = 0
    continuity_unknown = 0
    connection_attempts = 0
    connected_elapsed_seconds = 0.0
    terminal_reason = "capture did not start"
    complete = False

    while True:
        now = loop.time()
        if now >= deadline:
            complete = True
            terminal_reason = "requested run window elapsed"
            break
        if frame_limit is not None and frames >= frame_limit:
            complete = True
            terminal_reason = "max_frames reached"
            break

        connection_attempts += 1
        writer: RawSegmentWriter | None = None
        connection_started: float | None = None
        planned_stop = False
        connection_error: Exception | None = None
        try:
            async with connect() as websocket:
                connection_started = loop.time()
                await websocket.send(subscription)
                while True:
                    if frame_limit is not None and frames >= frame_limit:
                        complete = True
                        terminal_reason = "max_frames reached"
                        planned_stop = True
                        break
                    remaining_seconds = deadline - loop.time()
                    if remaining_seconds <= 0:
                        complete = True
                        terminal_reason = "requested run window elapsed"
                        planned_stop = True
                        break
                    try:
                        frame = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=min(timeout_seconds, remaining_seconds),
                        )
                    except TimeoutError as exc:
                        if loop.time() >= deadline:
                            complete = True
                            terminal_reason = "requested run window elapsed"
                            planned_stop = True
                        else:
                            connection_error = exc
                            terminal_reason = "TimeoutError: receive idle timeout"
                        break
                    except Exception as exc:
                        connection_error = exc
                        terminal_reason = _terminal_reason(exc)
                        break

                    payload = exact_frame_bytes(frame)
                    receive_at = receive_clock()
                    if writer is None:
                        writer = _new_writer(raw_root)
                    try:
                        writer.append(payload, receive_at=receive_at)
                    except PartitionBoundaryError:
                        manifests.append(writer.seal())
                        writer = _new_writer(raw_root)
                        writer.append(payload, receive_at=receive_at)
                    frames += 1
                    try:
                        parse_market_frame(payload)
                    except (ProtocolError, TypeError, ValueError):
                        parse_errors += 1
        except Exception as exc:
            connection_error = exc
            terminal_reason = _terminal_reason(exc)
        finally:
            if writer is not None:
                manifests.append(writer.seal())
            if connection_started is not None:
                connected_elapsed_seconds += max(
                    0.0,
                    min(loop.time(), deadline) - connection_started,
                )

        if planned_stop:
            break

        # An unplanned connection end creates an unknowable continuity boundary.
        # The next segment starts a new epoch; no delta is fabricated to bridge it.
        if connection_error is not None:
            gaps += 1
            continuity_unknown += 1
        if reconnect_limit is not None and reconnects >= reconnect_limit:
            break

        remaining_seconds = deadline - loop.time()
        if remaining_seconds <= 0:
            complete = True
            terminal_reason = "requested run window elapsed"
            break
        delay = backoffs[min(reconnects, len(backoffs) - 1)]
        await sleep(min(delay, remaining_seconds))
        if loop.time() >= deadline:
            complete = True
            terminal_reason = "requested run window elapsed"
            break
        reconnects += 1

    finished_monotonic = loop.time()
    observed_seconds = max(
        0.0,
        min(finished_monotonic, deadline) - started_monotonic,
    )
    observed_days = observed_seconds / 86_400
    return SupervisorResult(
        manifests=tuple(manifests),
        counters=CaptureCounters(
            frames=frames,
            parse_errors=parse_errors,
            reconnects=reconnects,
            gaps=gaps,
            continuity_unknown=continuity_unknown,
            out_of_order=0,
        ),
        connection_attempts=connection_attempts,
        complete=complete,
        terminal_reason=terminal_reason,
        started_at=started_at,
        finished_at=utc_now_text(),
        requested_run_seconds=requested_seconds,
        asset_ids=assets,
        max_frames=frame_limit,
        max_reconnects=reconnect_limit,
        reconnect_backoff_seconds=backoffs,
        receive_timeout_seconds=timeout_seconds,
        observed_elapsed_seconds=observed_seconds,
        connected_elapsed_seconds=min(
            connected_elapsed_seconds, observed_seconds
        ),
        required_elapsed_days=_X08_REQUIRED_ELAPSED_DAYS,
        observed_elapsed_days=observed_days,
        duration_gate_met=observed_days >= _X08_REQUIRED_ELAPSED_DAYS,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(
        root.resolve(strict=True)
    ).as_posix()


def build_polymarket_health_report(
    result: SupervisorResult,
    *,
    raw_root: str | Path,
) -> dict[str, Any]:
    """Build a hash-bound operational report without promoting X-08."""

    root = Path(raw_root)
    segments: list[dict[str, Any]] = []
    for manifest in result.manifests:
        verification = verify_segment(manifest.path, root=root)
        if not verification.valid:
            raise RawStoreError(
                "cannot report an unverified raw segment: "
                + "; ".join(verification.errors)
            )
        segments.append(
            {
                "capture_session_id": manifest.capture_session_id,
                "partition_date": manifest.partition_date,
                "partition_hour": manifest.partition_hour,
                "first_receive_at": manifest.first_receive_at,
                "last_receive_at": manifest.last_receive_at,
                "record_count": manifest.record_count,
                "manifest_path": _relative_path(manifest.path, root),
                "manifest_sha256": _sha256_file(manifest.path),
                "object_path": _relative_path(manifest.object_path, root),
                "object_sha256": manifest.object_sha256,
                "object_size_bytes": manifest.object_size_bytes,
            }
        )

    total_bytes = sum(
        manifest.object_size_bytes for manifest in result.manifests
    )
    gb_per_day = (
        total_bytes
        / 1_000_000_000
        * 86_400
        / result.observed_elapsed_seconds
        if result.observed_elapsed_seconds > 0
        else None
    )
    return {
        "report_version": "v1",
        "evidence_scope": "operational_observation_not_formal_x08_result",
        "formal_x08_result": False,
        "source": "polymarket",
        "endpoint": MARKET_WS_URL,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "complete": result.complete,
        "terminal_reason": result.terminal_reason,
        "subscription": {
            "channel": "market",
            "asset_ids": list(result.asset_ids),
        },
        "configuration": {
            "run_seconds": result.requested_run_seconds,
            "max_frames": result.max_frames,
            "receive_timeout_seconds": result.receive_timeout_seconds,
            "max_reconnects": result.max_reconnects,
            "reconnect_backoff_seconds": list(
                result.reconnect_backoff_seconds
            ),
        },
        "prospective_observation": {
            "required_elapsed_days": result.required_elapsed_days,
            "observed_elapsed_days": result.observed_elapsed_days,
            "observed_elapsed_seconds": result.observed_elapsed_seconds,
            "duration_gate_met": result.duration_gate_met,
            "fixtures_can_satisfy_elapsed_time": False,
        },
        "health": {
            "connection_attempts": result.connection_attempts,
            "connected_elapsed_seconds": result.connected_elapsed_seconds,
            "uptime_ratio": result.uptime_ratio,
            "frames": result.counters.frames,
            "parse_errors": result.counters.parse_errors,
            "reconnects": result.counters.reconnects,
            "gaps": result.counters.gaps,
            "continuity_unknown": result.counters.continuity_unknown,
            "out_of_order": result.counters.out_of_order,
            "sealed_segments": len(result.manifests),
            "utc_hour_partitions": sorted(
                {
                    f"{manifest.partition_date}T{manifest.partition_hour}"
                    for manifest in result.manifests
                }
            ),
            "compressed_bytes": total_bytes,
            "observed_gb_per_day": gb_per_day,
        },
        "segments": segments,
    }


__all__ = [
    "SupervisorResult",
    "build_polymarket_health_report",
    "supervise_polymarket",
]
