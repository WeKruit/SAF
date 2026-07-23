from __future__ import annotations

import base64
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from prediction_market import recording as recording_module
from prediction_market.contracts import VenueRuleSnapshotV0
from prediction_market.raw_store import verify_segment
from prediction_market.recording import (
    FormalReplayRejected,
    capture_venue_rule_snapshot,
    record_kalshi,
    record_kalshi_with_reconnect,
    record_polymarket,
    record_polymarket_with_reconnect,
)


UTC_TIME = "2026-07-22T14:03:04.123456Z"


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


class FakeConnection:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *_args: object) -> None:
        return None


def _raw_payloads(object_path: Path) -> list[bytes]:
    with object_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            raw = reader.read()
    return [
        base64.b64decode(json.loads(line)["payload_base64"], validate=True)
        for line in raw.splitlines()
    ]


@pytest.mark.asyncio
async def test_polymarket_frames_are_written_before_parsing(tmp_path: Path) -> None:
    frame = b'{"event_type":"book","asset_id":"asset-1","bids":[],"asks":[]}'
    websocket = FakeWebSocket([frame])

    result = await record_polymarket(
        websocket,
        ["asset-1"],
        tmp_path,
        max_frames=1,
        clock=lambda: UTC_TIME,
    )

    assert result.complete is True
    assert result.counters.frames == 1
    assert result.counters.parse_errors == 0
    assert len(result.manifests) == 1
    assert verify_segment(result.manifests[0].path).valid
    assert _raw_payloads(result.manifests[0].object_path) == [frame]


@pytest.mark.asyncio
async def test_parse_failure_does_not_erase_raw_polymarket_frame(
    tmp_path: Path,
) -> None:
    frame = b"not-json"
    websocket = FakeWebSocket([frame])

    result = await record_polymarket(
        websocket,
        ["asset-1"],
        tmp_path,
        max_frames=1,
        clock=lambda: UTC_TIME,
    )

    assert result.complete is True
    assert result.counters.parse_errors == 1
    assert _raw_payloads(result.manifests[0].object_path) == [frame]


@pytest.mark.asyncio
async def test_connection_rotates_segments_without_losing_boundary_frame(
    tmp_path: Path,
) -> None:
    frames = [
        b'{"event_type":"book","asset_id":"asset-1"}',
        b'{"event_type":"price_change","asset_id":"asset-1"}',
    ]
    times = iter(["2026-07-22T14:59:59.999999Z", "2026-07-22T15:00:00Z"])

    result = await record_polymarket(
        FakeWebSocket(frames),
        ["asset-1"],
        tmp_path,
        max_frames=2,
        clock=lambda: next(times),
    )

    assert result.complete is True
    assert len(result.manifests) == 2
    stored = [
        payload
        for manifest in result.manifests
        for payload in _raw_payloads(manifest.object_path)
    ]
    assert stored == frames
    assert all(verify_segment(manifest.path).valid for manifest in result.manifests)


@pytest.mark.asyncio
async def test_polymarket_reconnect_is_bounded_and_counts_continuity_gaps(
    tmp_path: Path,
) -> None:
    sockets = [
        FakeWebSocket([], terminal_error=ConnectionError("first disconnect")),
        FakeWebSocket([b'{"event_type":"book","asset_id":"asset-1"}']),
    ]
    attempts = 0
    sleeps: list[float] = []

    def connect() -> FakeConnection:
        nonlocal attempts
        websocket = sockets[attempts]
        attempts += 1
        return FakeConnection(websocket)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = await record_polymarket_with_reconnect(
        connect,
        ["asset-1"],
        tmp_path,
        max_frames=1,
        max_reconnects=1,
        backoff_seconds=(0.25,),
        clock=lambda: UTC_TIME,
        sleep=sleep,
    )

    assert result.complete is True
    assert attempts == 2
    assert sleeps == [0.25]
    assert result.counters.reconnects == 1
    assert result.counters.gaps == 1
    assert result.counters.continuity_unknown == 1
    assert result.counters.frames == 1
    assert all(verify_segment(manifest.path).valid for manifest in result.manifests)


@pytest.mark.asyncio
async def test_polymarket_reconnect_budget_exhaustion_is_explicit(
    tmp_path: Path,
) -> None:
    attempts = 0

    def connect() -> FakeConnection:
        nonlocal attempts
        attempts += 1
        return FakeConnection(
            FakeWebSocket([], terminal_error=ConnectionError("offline"))
        )

    async def no_wait(_delay: float) -> None:
        return None

    result = await record_polymarket_with_reconnect(
        connect,
        ["asset-1"],
        tmp_path,
        max_frames=1,
        max_reconnects=2,
        backoff_seconds=(0.0, 0.0),
        clock=lambda: UTC_TIME,
        sleep=no_wait,
    )

    assert result.complete is False
    assert attempts == 3
    assert result.counters.reconnects == 2
    assert result.counters.gaps == 2
    assert "offline" in result.terminal_reason


@pytest.mark.asyncio
async def test_kalshi_frames_are_exact_and_sequence_gaps_are_counted(
    tmp_path: Path,
) -> None:
    frames = [
        b'{"type":"orderbook_snapshot","sid":7,"seq":40,'
        b'"msg":{"market_ticker":"KXNBA-ONE"}}',
        b'{"type":"orderbook_delta","sid":7,"seq":42,'
        b'"msg":{"market_ticker":"KXNBA-ONE","ts_ms":1234}}',
    ]
    websocket = FakeWebSocket(frames)

    result = await record_kalshi(
        websocket,
        ["KXNBA-ONE"],
        tmp_path,
        max_frames=2,
        clock=lambda: UTC_TIME,
    )

    assert result.complete is True
    assert result.counters.frames == 2
    assert result.counters.gaps == 1
    assert result.counters.continuity_unknown == 0
    assert result.counters.out_of_order == 0
    assert _raw_payloads(result.manifests[0].object_path) == frames


@pytest.mark.asyncio
async def test_kalshi_duplicate_or_backward_sequence_is_counted(
    tmp_path: Path,
) -> None:
    websocket = FakeWebSocket(
        [
            b'{"type":"orderbook_snapshot","sid":7,"seq":40,"msg":{}}',
            b'{"type":"orderbook_delta","sid":7,"seq":40,"msg":{}}',
        ]
    )

    result = await record_kalshi(
        websocket,
        ["KXNBA-ONE"],
        tmp_path,
        max_frames=2,
        clock=lambda: UTC_TIME,
    )

    assert result.counters.out_of_order == 1
    assert result.counters.gaps == 0


@pytest.mark.asyncio
async def test_kalshi_subscription_ack_is_preserved_without_parse_error(
    tmp_path: Path,
) -> None:
    frames = [
        b'{"type":"subscribed","id":1,"msg":{"channel":"orderbook_delta"}}',
        b'{"type":"orderbook_snapshot","sid":7,"seq":40,"msg":{}}',
    ]

    result = await record_kalshi(
        FakeWebSocket(frames),
        ["KXNBA-ONE"],
        tmp_path,
        max_frames=2,
        clock=lambda: UTC_TIME,
    )

    assert result.counters.parse_errors == 0
    assert result.counters.gaps == 0
    assert _raw_payloads(result.manifests[0].object_path) == frames


@pytest.mark.asyncio
async def test_kalshi_reconnect_starts_a_new_sequence_epoch_and_marks_boundary(
    tmp_path: Path,
) -> None:
    sockets = [
        FakeWebSocket(
            [b'{"type":"orderbook_snapshot","sid":7,"seq":40,"msg":{}}'],
            terminal_error=ConnectionError("disconnect"),
        ),
        FakeWebSocket(
            [b'{"type":"orderbook_delta","sid":7,"seq":100,"msg":{}}']
        ),
    ]
    attempts = 0

    def connect() -> FakeConnection:
        nonlocal attempts
        websocket = sockets[attempts]
        attempts += 1
        return FakeConnection(websocket)

    async def no_wait(_delay: float) -> None:
        return None

    result = await record_kalshi_with_reconnect(
        connect,
        ["KXNBA-ONE"],
        tmp_path,
        max_frames=2,
        max_reconnects=1,
        backoff_seconds=(0.0,),
        clock=lambda: UTC_TIME,
        sleep=no_wait,
    )

    assert result.complete is True
    assert attempts == 2
    assert result.counters.reconnects == 1
    assert result.counters.gaps == 1
    assert result.counters.continuity_unknown == 1
    assert result.counters.out_of_order == 0
    assert result.counters.frames == 2
    assert len(result.manifests) == 2


def _valid_rule_document() -> dict[str, object]:
    return {
        "effective_from": UTC_TIME,
        "game_start_time": "2026-07-22T19:00:00Z",
        "seconds_delay": 1,
        "cancel_during_delay": False,
        "start_time_cancel_policy": "cancel_all_with_schedule_change_exception",
        "fees_enabled": True,
        "fee_rate": "0.05",
        "fee_exponent": "2",
        "taker_only": True,
        "maker_fee_rate": "0",
        "minimum_tick_size": "0.01",
        "minimum_order_size": "5",
        "order_types_supported": ["GTC", "GTD", "FOK", "FAK"],
    }


def test_rule_response_is_sealed_before_normalization(tmp_path: Path) -> None:
    raw_response = json.dumps(
        _valid_rule_document(), separators=(",", ":")
    ).encode()
    parser_observation: list[Path] = []

    def parser(payload: bytes, manifest_path: Path) -> dict[str, object]:
        assert payload == raw_response
        assert verify_segment(manifest_path).valid
        parser_observation.append(manifest_path)
        return json.loads(payload)

    capture = capture_venue_rule_snapshot(
        raw_response,
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at=UTC_TIME,
        source_document_version=(
            "https://docs.polymarket.com/trading/orders/create#sports-markets"
            "@2026-07-22"
        ),
        parser=parser,
    )

    assert parser_observation == [capture.raw_manifest.path]
    assert capture.valid is True
    assert capture.quality_flags == ()
    assert capture.validation_errors == ()
    assert isinstance(capture.snapshot, VenueRuleSnapshotV0)
    assert capture.snapshot.seconds_delay.to_decimal() == Decimal("1")
    assert capture.snapshot.raw_response_hash == (
        "sha256:" + hashlib.sha256(raw_response).hexdigest()
    )
    assert _raw_payloads(capture.raw_manifest.object_path) == [raw_response]
    assert verify_segment(capture.canonical_manifest.path).valid
    canonical_record = json.loads(
        _raw_payloads(capture.canonical_manifest.object_path)[0]
    )
    assert canonical_record["valid"] is True
    assert canonical_record["quality_flags"] == []
    assert canonical_record["market_id"] == "market_rule_1"
    assert canonical_record["condition_id"] == "condition_rule_1"
    assert canonical_record["source_document_version"].endswith("@2026-07-22")
    assert canonical_record["snapshot"] == capture.snapshot.model_dump(mode="json")


def test_missing_rule_field_is_quality_marked_and_formal_replay_rejects(
    tmp_path: Path,
) -> None:
    document = _valid_rule_document()
    del document["fee_rate"]
    raw_response = json.dumps(document, separators=(",", ":")).encode()

    capture = capture_venue_rule_snapshot(
        raw_response,
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at=UTC_TIME,
        source_document_version="official-response@2026-07-22",
    )

    assert capture.valid is False
    assert capture.snapshot is None
    assert capture.quality_flags == ("preliminary_rules",)
    assert "MISSING_RULE_FIELD:fee_rate" in capture.validation_errors
    assert verify_segment(capture.canonical_manifest.path).valid
    canonical_record = json.loads(
        _raw_payloads(capture.canonical_manifest.object_path)[0]
    )
    assert canonical_record["valid"] is False
    assert canonical_record["quality_flags"] == ["preliminary_rules"]
    assert "MISSING_RULE_FIELD:fee_rate" in canonical_record["validation_errors"]
    assert verify_segment(capture.raw_manifest.path).valid
    assert _raw_payloads(capture.raw_manifest.object_path) == [raw_response]


def test_rule_numeric_fields_reject_binary_float_in_formal_snapshot(
    tmp_path: Path,
) -> None:
    document = _valid_rule_document()
    document["fee_rate"] = 0.05
    raw_response = json.dumps(document, separators=(",", ":")).encode()

    capture = capture_venue_rule_snapshot(
        raw_response,
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at=UTC_TIME,
        source_document_version="official-response@2026-07-22",
        parser=lambda payload, _manifest_path: json.loads(payload),
    )

    assert capture.valid is False
    assert capture.snapshot is None
    assert "INVALID_RULE_FIELD:fee_rate" in capture.validation_errors


def test_rule_timestamps_must_use_canonical_utc_form(tmp_path: Path) -> None:
    document = _valid_rule_document()
    document["effective_from"] = "2026-07-22 14:03:04Z"
    raw_response = json.dumps(document, separators=(",", ":")).encode()

    capture = capture_venue_rule_snapshot(
        raw_response,
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at=UTC_TIME,
        source_document_version="official-response@2026-07-22",
    )

    assert capture.valid is False
    assert capture.snapshot is None
    assert "INVALID_RULE_FIELD:effective_from" in capture.validation_errors


def test_rule_store_selects_latest_strict_as_of_snapshot(tmp_path: Path) -> None:
    first_document = _valid_rule_document()
    first = capture_venue_rule_snapshot(
        json.dumps(first_document, separators=(",", ":")).encode(),
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at="2026-07-22T14:00:00Z",
        source_document_version="official-response@2026-07-22T14:00:00Z",
    )
    second_document = _valid_rule_document()
    second_document["minimum_order_size"] = "10"
    second_document["effective_from"] = "2026-07-22T14:05:00Z"
    second = capture_venue_rule_snapshot(
        json.dumps(second_document, separators=(",", ":")).encode(),
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at="2026-07-22T14:05:00Z",
        source_document_version="official-response@2026-07-22T14:05:00Z",
    )
    future_document = _valid_rule_document()
    future_document["minimum_order_size"] = "15"
    future_document["effective_from"] = "2026-07-22T14:10:00Z"
    future = capture_venue_rule_snapshot(
        json.dumps(future_document, separators=(",", ":")).encode(),
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at="2026-07-22T14:05:30Z",
        source_document_version="official-response@2026-07-22T14:05:30Z",
    )
    store = recording_module.VenueRuleStore(tmp_path)

    before_change = store.require_as_of(
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        at="2026-07-22T14:04:00Z",
        max_age_seconds=600,
    )
    after_change = store.require_as_of(
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        at="2026-07-22T14:06:00Z",
        max_age_seconds=600,
    )

    assert before_change == first.snapshot
    assert after_change == second.snapshot
    assert before_change.minimum_order_size.to_decimal() == Decimal("5")
    assert after_change.minimum_order_size.to_decimal() == Decimal("10")
    assert first.canonical_manifest.path != second.canonical_manifest.path
    assert second.canonical_manifest.path != future.canonical_manifest.path
    assert first.canonical_manifest.path.exists()
    assert second.canonical_manifest.path.exists()
    assert future.canonical_manifest.path.exists()


def test_rule_store_rejects_missing_stale_and_latest_invalid_state(
    tmp_path: Path,
) -> None:
    valid_document = _valid_rule_document()
    capture_venue_rule_snapshot(
        json.dumps(valid_document, separators=(",", ":")).encode(),
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at="2026-07-22T14:00:00Z",
        source_document_version="official-response@2026-07-22T14:00:00Z",
    )
    store = recording_module.VenueRuleStore(tmp_path)

    with pytest.raises(FormalReplayRejected, match="missing"):
        store.require_as_of(
            venue="polymarket",
            market_id="market_missing",
            condition_id="condition_missing",
            at="2026-07-22T14:01:00Z",
            max_age_seconds=300,
        )
    with pytest.raises(FormalReplayRejected, match="stale"):
        store.require_as_of(
            venue="polymarket",
            market_id="market_rule_1",
            condition_id="condition_rule_1",
            at="2026-07-22T15:00:00Z",
            max_age_seconds=300,
        )

    invalid_document = _valid_rule_document()
    del invalid_document["fee_rate"]
    capture_venue_rule_snapshot(
        json.dumps(invalid_document, separators=(",", ":")).encode(),
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at="2026-07-22T14:02:00Z",
        source_document_version="official-response@2026-07-22T14:02:00Z",
    )

    with pytest.raises(FormalReplayRejected, match="fee_rate"):
        store.require_as_of(
            venue="polymarket",
            market_id="market_rule_1",
            condition_id="condition_rule_1",
            at="2026-07-22T14:03:00Z",
            max_age_seconds=300,
        )


def test_rule_store_never_reopens_verified_segment_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _valid_rule_document()
    capture = capture_venue_rule_snapshot(
        json.dumps(document, separators=(",", ":")).encode(),
        raw_root=tmp_path,
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        fetched_at="2026-07-22T14:00:00Z",
        source_document_version="official-response@2026-07-22T14:00:00Z",
    )
    forbidden = {
        capture.canonical_manifest.object_path,
        capture.raw_manifest.object_path,
    }
    original_open = Path.open

    def reject_verified_path_reopen(path: Path, *args: object, **kwargs: object):
        if path in forbidden:
            raise AssertionError("verified segment object was reopened by path")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_verified_path_reopen)

    selected = recording_module.VenueRuleStore(tmp_path).require_as_of(
        venue="polymarket",
        market_id="market_rule_1",
        condition_id="condition_rule_1",
        at="2026-07-22T14:04:00Z",
        max_age_seconds=300,
    )

    assert selected == capture.snapshot


def test_connectivity_artifacts_record_evidence_and_credential_blocker() -> None:
    project_root = Path(__file__).resolve().parents[1]
    matrix = (
        project_root / "artifacts/venue-connectivity/capability_matrix_v0.csv"
    ).read_text(encoding="utf-8")
    credentials = (
        project_root / "artifacts/venue-connectivity/kalshi_credentials.md"
    ).read_text(encoding="utf-8")
    smoke_report_path = (
        project_root
        / "artifacts/venue-connectivity/polymarket_public_smoke_v0.json"
    )
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    smoke_root = (
        project_root
        / "artifacts/venue-connectivity/polymarket-public-smoke-v0"
    )

    assert "https://docs.polymarket.com/market-data/websocket/market-channel" in matrix
    assert "https://docs.kalshi.com/getting_started/api_keys" in matrix
    assert "KALSHI_API_KEY_ID" in credentials
    assert "KALSHI_PRIVATE_KEY_PATH" in credentials
    assert "blocked" in credentials.lower()
    assert "BEGIN PRIVATE KEY" not in credentials
    assert "artifacts/venue-connectivity/polymarket_public_smoke_v0.json" in matrix
    assert smoke_report["evidence_scope"] == (
        "operational_observation_not_x08_evidence"
    )
    assert smoke_report["configuration"]["max_frames"] == 5
    assert smoke_report["counters"]["frames"] == 5
    assert smoke_report["counters"]["parse_errors"] == 0
    for segment in smoke_report["segments"]:
        manifest_path = smoke_root / segment["manifest_path"]
        object_path = smoke_root / segment["object_path"]
        assert verify_segment(manifest_path, root=smoke_root).valid
        assert segment["manifest_sha256"] == (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )
        assert segment["object_sha256"] == (
            "sha256:" + hashlib.sha256(object_path.read_bytes()).hexdigest()
        )
    committed_evidence = smoke_report_path.read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" not in committed_evidence
    assert "KALSHI-ACCESS-" not in committed_evidence


def test_polymarket_cli_fails_explicitly_when_no_active_sports_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from prediction_market.cli import record_markets

    async def no_assets(**_kwargs: object) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(record_markets, "discover_active_sports_assets", no_assets)
    raw_root = tmp_path / "must-not-contain-fabricated-data"

    exit_code = record_markets.main(
        [
            "polymarket",
            "--discover-sports",
            "--run-seconds",
            "1",
            "--max-frames",
            "1",
            "--raw-root",
            str(raw_root),
        ]
    )

    assert exit_code == 2
    assert "no active sports markets" in capsys.readouterr().err.lower()
    assert not raw_root.exists()


def test_polymarket_cli_writes_reproducible_operational_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from prediction_market.cli import record_markets

    captured: dict[str, object] = {}

    async def assets(**_kwargs: object) -> tuple[str, ...]:
        return ("asset-1",)

    async def capture(asset_ids: tuple[str, ...], raw_root: Path, **kwargs: object):
        captured.update(asset_ids=asset_ids, raw_root=raw_root, **kwargs)
        writer = recording_module.RawSegmentWriter(
            raw_root, source="polymarket", stream="market"
        )
        writer.append(
            b'{"event_type":"book","asset_id":"asset-1"}',
            receive_at=UTC_TIME,
        )
        manifest = writer.seal()
        return SimpleNamespace(
            complete=True,
            terminal_reason="max_frames reached",
            connection_attempts=1,
            started_at="2026-07-22T14:03:04.000000Z",
            finished_at="2026-07-22T14:03:05.000000Z",
            requested_run_seconds=3.0,
            asset_ids=("asset-1",),
            max_frames=1,
            max_reconnects=2,
            reconnect_backoff_seconds=(0.25, 0.5, 1.0, 2.0, 5.0),
            receive_timeout_seconds=2.0,
            observed_elapsed_seconds=1.0,
            connected_elapsed_seconds=1.0,
            required_elapsed_days=7,
            observed_elapsed_days=1 / 86_400,
            duration_gate_met=False,
            formal_x08_result=False,
            uptime_ratio=1.0,
            counters=SimpleNamespace(
                frames=1,
                parse_errors=0,
                reconnects=0,
                gaps=0,
                continuity_unknown=0,
                out_of_order=0,
            ),
            manifests=(manifest,),
        )

    monkeypatch.setattr(record_markets, "discover_active_sports_assets", assets)
    monkeypatch.setattr(record_markets, "capture_public_polymarket", capture)
    raw_root = tmp_path / "raw"
    output = tmp_path / "polymarket-public-smoke.json"

    exit_code = record_markets.main(
        [
            "polymarket",
            "--discover-sports",
            "--run-seconds",
            "3",
            "--max-frames",
            "1",
            "--max-assets",
            "2",
            "--receive-timeout-seconds",
            "2",
            "--max-reconnects",
            "2",
            "--raw-root",
            str(raw_root),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured["asset_ids"] == ("asset-1",)
    assert captured["raw_root"] == raw_root
    assert captured["max_frames"] == 1
    assert captured["run_seconds"] == 3.0
    assert captured["receive_timeout_seconds"] == 2.0
    assert report["report_version"] == "v1"
    assert report["evidence_scope"] == (
        "operational_observation_not_formal_x08_result"
    )
    assert report["formal_x08_result"] is False
    assert report["source"] == "polymarket"
    assert report["endpoint"] == (
        "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    )
    assert report["subscription"] == {
        "asset_ids": ["asset-1"],
        "channel": "market",
    }
    assert report["configuration"] == {
        "discover_sports": True,
        "market_limit": 100,
        "max_assets": 2,
        "run_seconds": 3.0,
        "max_frames": 1,
        "max_reconnects": 2,
        "receive_timeout_seconds": 2.0,
        "reconnect_backoff_seconds": [0.25, 0.5, 1.0, 2.0, 5.0],
    }
    assert report["prospective_observation"] == {
        "required_elapsed_days": 7,
        "observed_elapsed_days": 1 / 86_400,
        "observed_elapsed_seconds": 1.0,
        "duration_gate_met": False,
        "fixtures_can_satisfy_elapsed_time": False,
    }
    assert report["health"]["continuity_unknown"] == 0
    assert report["health"]["frames"] == 1
    assert report["health"]["gaps"] == 0
    assert report["health"]["parse_errors"] == 0
    assert report["health"]["reconnects"] == 0
    assert report["health"]["connection_attempts"] == 1
    assert report["segments"][0]["capture_session_id"].startswith("capture-")
    assert report["segments"][0]["record_count"] == 1
    manifest_path = raw_root / report["segments"][0]["manifest_path"]
    object_path = raw_root / report["segments"][0]["object_path"]
    assert report["segments"][0]["manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert report["segments"][0]["object_sha256"] == (
        "sha256:" + hashlib.sha256(object_path.read_bytes()).hexdigest()
    )
    assert "manifest_paths" not in report
    assert "wrote" in capsys.readouterr().out
