from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prediction_market.raw_store import read_verified_segment, verify_segment
from prediction_market.recorder_supervisor import (
    build_polymarket_health_report,
    supervise_polymarket,
)


class FakeWebSocket:
    def __init__(
        self,
        frames: list[bytes | str],
        *,
        terminal_error: Exception | None = None,
    ) -> None:
        self.frames = list(frames)
        self.terminal_error = terminal_error
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> bytes | str:
        if self.frames:
            return self.frames.pop(0)
        if self.terminal_error is not None:
            raise self.terminal_error
        raise ConnectionError("fake websocket exhausted")


class BlockingWebSocket(FakeWebSocket):
    async def recv(self) -> bytes | str:
        await __import__("asyncio").sleep(60)
        raise AssertionError("the bounded supervisor must cancel this receive")


class FakeConnection:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_supervisor_rotates_on_utc_hour_and_commits_each_segment(
    tmp_path: Path,
) -> None:
    frames = [
        b'{"event_type":"book","asset_id":"asset-1","bids":[],"asks":[]}',
        b'{"event_type":"price_change","asset_id":"asset-1","price":"0.5"}',
    ]
    receive_times = iter(
        ["2026-07-22T14:59:59.999999Z", "2026-07-22T15:00:00.000000Z"]
    )

    result = await supervise_polymarket(
        lambda: FakeConnection(FakeWebSocket(frames)),
        ("asset-1",),
        tmp_path,
        run_seconds=60,
        max_frames=2,
        max_reconnects=0,
        receive_timeout_seconds=1,
        receive_clock=lambda: next(receive_times),
    )

    assert result.complete is True
    assert result.terminal_reason == "max_frames reached"
    assert result.counters.frames == 2
    assert [manifest.partition_hour for manifest in result.manifests] == ["14", "15"]
    assert [manifest.record_count for manifest in result.manifests] == [1, 1]
    assert all(
        verify_segment(manifest.path, root=tmp_path).valid
        for manifest in result.manifests
    )
    assert [
        verified.payloads[0]
        for verified in (
            read_verified_segment(manifest.path, root=tmp_path)
            for manifest in result.manifests
        )
    ] == frames


@pytest.mark.asyncio
async def test_reconnect_starts_new_segment_and_marks_unknown_continuity(
    tmp_path: Path,
) -> None:
    sockets = [
        FakeWebSocket(
            [b'{"event_type":"book","asset_id":"asset-1"}'],
            terminal_error=ConnectionError("first connection lost"),
        ),
        FakeWebSocket(
            [b'{"event_type":"price_change","asset_id":"asset-1"}']
        ),
    ]
    attempts = 0

    def connect() -> FakeConnection:
        nonlocal attempts
        connection = FakeConnection(sockets[attempts])
        attempts += 1
        return connection

    async def no_wait(_seconds: float) -> None:
        return None

    result = await supervise_polymarket(
        connect,
        ("asset-1",),
        tmp_path,
        run_seconds=60,
        max_frames=2,
        max_reconnects=1,
        reconnect_backoff_seconds=(0,),
        receive_timeout_seconds=1,
        receive_clock=lambda: "2026-07-22T14:03:04.123456Z",
        sleep=no_wait,
    )

    assert attempts == 2
    assert result.counters.frames == 2
    assert result.counters.reconnects == 1
    assert result.counters.gaps == 1
    assert result.counters.continuity_unknown == 1
    assert len(result.manifests) == 2
    assert (
        result.manifests[0].capture_session_id
        != result.manifests[1].capture_session_id
    )


@pytest.mark.asyncio
async def test_run_deadline_is_graceful_and_cannot_satisfy_seven_day_gate(
    tmp_path: Path,
) -> None:
    result = await supervise_polymarket(
        lambda: FakeConnection(BlockingWebSocket([])),
        ("asset-1",),
        tmp_path,
        run_seconds=0.02,
        max_frames=None,
        max_reconnects=0,
        receive_timeout_seconds=1,
    )

    assert result.complete is True
    assert result.terminal_reason == "requested run window elapsed"
    assert result.required_elapsed_days == 7
    assert 0 <= result.observed_elapsed_days < 7
    assert result.duration_gate_met is False
    assert result.formal_x08_result is False
    assert result.counters.frames == 0
    assert result.manifests == ()


@pytest.mark.asyncio
async def test_health_report_hashes_committed_objects_and_separates_duration(
    tmp_path: Path,
) -> None:
    result = await supervise_polymarket(
        lambda: FakeConnection(
            FakeWebSocket(
                [b'{"event_type":"book","asset_id":"asset-1"}']
            )
        ),
        ("asset-1",),
        tmp_path,
        run_seconds=60,
        max_frames=1,
        max_reconnects=0,
        receive_timeout_seconds=1,
        receive_clock=lambda: "2026-07-22T14:03:04.123456Z",
    )

    report = build_polymarket_health_report(
        result,
        raw_root=tmp_path,
    )
    segment = report["segments"][0]
    manifest_path = tmp_path / segment["manifest_path"]
    object_path = tmp_path / segment["object_path"]

    assert report["report_version"] == "v1"
    assert report["evidence_scope"] == (
        "operational_observation_not_formal_x08_result"
    )
    assert report["prospective_observation"]["required_elapsed_days"] == 7
    assert 0 <= report["prospective_observation"]["observed_elapsed_days"] < 7
    assert report["prospective_observation"]["duration_gate_met"] is False
    assert report["formal_x08_result"] is False
    assert report["configuration"] == {
        "run_seconds": 60.0,
        "max_frames": 1,
        "receive_timeout_seconds": 1.0,
        "max_reconnects": 0,
        "reconnect_backoff_seconds": [0.25, 0.5, 1.0, 2.0, 5.0],
    }
    assert segment["manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert segment["object_sha256"] == (
        "sha256:" + hashlib.sha256(object_path.read_bytes()).hexdigest()
    )
    assert json.loads(manifest_path.read_text())["first_receive_at"] == (
        "2026-07-22T14:03:04.123456Z"
    )
