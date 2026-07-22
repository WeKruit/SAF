"""Shared, venue-neutral recorder types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from prediction_market.raw_store import SegmentManifest


class RecorderError(RuntimeError):
    """A recorder could not complete the explicitly bounded capture."""


class ProtocolError(ValueError):
    """A venue payload violates the documented wire protocol."""


class WebSocketTransport(Protocol):
    """Small transport boundary shared by real and test WebSockets."""

    async def send(self, message: str | bytes) -> object: ...

    async def recv(self) -> str | bytes: ...


@dataclass(frozen=True, slots=True)
class CaptureCounters:
    """Measured capture health; no field is inferred from undocumented data."""

    frames: int = 0
    parse_errors: int = 0
    reconnects: int = 0
    gaps: int = 0
    out_of_order: int = 0


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Immutable result for one bounded recorder run."""

    manifests: tuple[SegmentManifest, ...]
    counters: CaptureCounters
    complete: bool
    terminal_reason: str


class SequenceTracker:
    """Count missing and non-increasing native sequence numbers per stream."""

    def __init__(self) -> None:
        self._last_by_stream: dict[int, int] = {}
        self.gaps = 0
        self.out_of_order = 0

    def observe(self, stream_id: int, sequence: int) -> None:
        previous = self._last_by_stream.get(stream_id)
        if previous is None:
            self._last_by_stream[stream_id] = sequence
            return
        if sequence > previous + 1:
            self.gaps += sequence - previous - 1
            self._last_by_stream[stream_id] = sequence
            return
        if sequence <= previous:
            self.out_of_order += 1
            return
        self._last_by_stream[stream_id] = sequence


def exact_frame_bytes(frame: str | bytes) -> bytes:
    """Return the exact UTF-8 payload bytes exposed by the WebSocket API."""

    if type(frame) is bytes:
        return frame
    if type(frame) is str:
        return frame.encode("utf-8")
    raise ProtocolError("WebSocket frame must be text or bytes")


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "CaptureCounters",
    "CaptureResult",
    "ProtocolError",
    "RecorderError",
    "SequenceTracker",
    "WebSocketTransport",
    "exact_frame_bytes",
    "utc_now_text",
]
