from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from prediction_market.adapters.base import CaptureCounters
from prediction_market.raw_store import RawSegmentWriter
from prediction_market.recorder_supervisor import SupervisorResult
from prediction_market.sports.nfl_x14_polymarket_recorder import (
    GammaHttpResponse,
    X14PolymarketRecorderError,
    build_parser,
    fetch_target_event,
    parse_target_universe,
    preserve_target_metadata,
    run_targeted_capture,
    upload_capture_segments,
    verify_session_report,
    verify_target_metadata,
)
from prediction_market.sports.nfl_x14_reaction import (
    ProspectiveGameBindingV1,
)
from prediction_market.static_store import StaticStoreError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_SLUG = "nfl-car-ari-2026-08-07"
MONEYLINE_CONDITION = (
    "0xd0733676277270d804b2c93701d476d85d62e1d0444369e62e9e38687fc8864f"
)
TOTAL_CONDITION = (
    "0xcf85c1718080501dbd44e0fd1197a80f7a4b8558b52c5fd561ad79a9f48d61f4"
)


def _binding() -> ProspectiveGameBindingV1:
    return ProspectiveGameBindingV1.from_registry_record(
        {
            "binding_id": "NFL-2026-CAR-ARI",
            "game_id": "2026_00_CAR_ARI",
            "capture_session_id": "nfl-2026-car-ari",
            "scheduled_start_utc": "2026-08-07T00:00:00Z",
            "away_team": "CAR",
            "home_team": "ARI",
            "sports_provider": "licensed-feed",
            "sports_provider_game_id": "provider-car-ari",
            "polymarket": {
                "event_id": "684046",
                "event_slug": EVENT_SLUG,
                "native_game_id": "19453",
                "condition_ids": [
                    MONEYLINE_CONDITION,
                    TOTAL_CONDITION,
                ],
                "token_outcomes": {
                    "101": "Panthers",
                    "102": "Cardinals",
                    "201": "Over",
                    "202": "Under",
                },
            },
            "kalshi": None,
            "capture_start": "market_creation",
            "capture_end": "market_resolution",
            "venue_rule_snapshot_refs": ["sha256:" + "1" * 64],
            "license_refs": ["O-001"],
            "s3_namespace": "prediction-market/research/nfl/x14",
        }
    )


def _event() -> dict[str, Any]:
    return {
        "id": "684046",
        "slug": EVENT_SLUG,
        "gameId": 19453,
        "seriesSlug": "nfl-2026",
        "createdAt": "2026-07-08T11:00:00Z",
        "startDate": "2026-07-09T12:03:01.935741Z",
        "startTime": "2026-08-07T00:00:00Z",
        "sport": {"sport": "nfl"},
        "teams": [
            {
                "abbreviation": "car",
                "ordering": "away",
                "alias": "Panthers",
            },
            {
                "abbreviation": "ari",
                "ordering": "home",
                "alias": "Cardinals",
            },
        ],
        "markets": [
            {
                "id": "2858750",
                "conditionId": MONEYLINE_CONDITION,
                "question": "Panthers vs. Cardinals",
                "outcomes": '["Panthers","Cardinals"]',
                "clobTokenIds": '["101","102"]',
                "sportsMarketType": "moneyline",
                "comboStatus": "disabled",
                "gameStartTime": "2026-08-07 00:00:00+00",
                "createdAt": "2026-07-09T12:06:00Z",
            },
            {
                "id": "2922959",
                "conditionId": TOTAL_CONDITION,
                "question": "Panthers vs. Cardinals: O/U 32.5",
                "outcomes": '["Over","Under"]',
                "clobTokenIds": '["201","202"]',
                "sportsMarketType": "totals",
                "comboStatus": "disabled",
                "gameStartTime": "2026-08-07 00:00:00+00",
                "createdAt": "2026-07-24T12:06:00Z",
            },
        ],
    }


def _payload(event: dict[str, Any] | None = None) -> bytes:
    return json.dumps(
        _event() if event is None else event,
        separators=(",", ":"),
    ).encode()


def _http_response(event: dict[str, Any] | None = None) -> GammaHttpResponse:
    payload = _payload(event)
    return GammaHttpResponse(
        payload=payload,
        fetched_at="2026-07-25T18:00:00Z",
        source_url=(
            "https://gamma-api.polymarket.com/events/slug/"
            f"{EVENT_SLUG}"
        ),
        request_params={},
        headers={
            "content-type": "application/json",
            "etag": '"fixture-etag"',
        },
    )


class _CaptureSpy:
    def __init__(self) -> None:
        self.asset_ids: tuple[str, ...] | None = None

    async def __call__(
        self,
        asset_ids: tuple[str, ...],
        raw_root: Path,
        *,
        run_seconds: float,
        max_frames: int | None,
        receive_timeout_seconds: float,
        max_reconnects: int | None,
    ) -> SupervisorResult:
        self.asset_ids = tuple(asset_ids)
        writer = RawSegmentWriter(
            raw_root,
            source="polymarket",
            stream="market",
            capture_session_id="nfl-2026-car-ari",
        )
        writer.append(
            b'{"event_type":"book","asset_id":"101","bids":[],"asks":[]}',
            receive_at="2026-07-25T18:01:00Z",
        )
        manifest = writer.seal()
        return SupervisorResult(
            manifests=(manifest,),
            counters=CaptureCounters(frames=1),
            connection_attempts=1,
            complete=True,
            terminal_reason="max_frames reached",
            started_at="2026-07-25T18:01:00Z",
            finished_at="2026-07-25T18:01:01Z",
            requested_run_seconds=run_seconds,
            asset_ids=tuple(asset_ids),
            max_frames=max_frames,
            max_reconnects=max_reconnects,
            reconnect_backoff_seconds=(0.25,),
            receive_timeout_seconds=receive_timeout_seconds,
            observed_elapsed_seconds=1.0,
            connected_elapsed_seconds=1.0,
            required_elapsed_days=7,
            observed_elapsed_days=1 / 86_400,
            duration_gate_met=False,
        )


class _FakeObjectStore:
    def __init__(
        self,
        *,
        corrupt_head: bool = False,
        corrupt_readback: bool = False,
    ) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.puts: list[str] = []
        self.corrupt_head = corrupt_head
        self.corrupt_readback = corrupt_readback

    def head_object(self, key: str) -> dict[str, object] | None:
        if key not in self.objects:
            return None
        return {
            "key": key,
            "byte_size": (
                len(self.objects[key]) + 1
                if self.corrupt_head
                else len(self.objects[key])
            ),
            "metadata": self.metadata[key],
        }

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None:
        self.puts.append(key)
        self.objects[key] = path.read_bytes()
        self.metadata[key] = dict(metadata)

    def read_object_chunks(
        self, key: str, *, chunk_size: int = 1024 * 1024
    ) -> Any:
        payload = self.objects[key]
        if self.corrupt_readback and payload:
            payload = bytes([payload[0] ^ 1]) + payload[1:]
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]


def test_get_is_exact_event_by_slug_and_preserves_response_bytes() -> None:
    expected = _payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/events/slug/{EVENT_SLUG}"
        assert dict(request.url.params) == {}
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            content=expected,
            headers={"Content-Type": "application/json"},
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://gamma-api.polymarket.com",
    ) as client:
        response = fetch_target_event(
            client=client,
            binding=_binding(),
            fetched_at=lambda: "2026-07-25T18:00:00Z",
        )

    assert response.payload == expected
    assert response.request_params == {}
    assert parse_target_universe(
        response, binding=_binding()
    ).event_id == "684046"


def test_slug_endpoint_rejects_legacy_list_shape() -> None:
    response = _http_response()
    legacy_list = replace(
        response,
        payload=json.dumps([_event()], separators=(",", ":")).encode(),
    )

    with pytest.raises(X14PolymarketRecorderError, match="single event object"):
        parse_target_universe(legacy_list, binding=_binding())


@pytest.mark.parametrize(
    "response",
    [
        replace(
            _http_response(),
            source_url="https://gamma-api.polymarket.com/events",
        ),
        replace(_http_response(), request_params={"slug": EVENT_SLUG}),
    ],
)
def test_parser_requires_exact_slug_request_identity(
    response: GammaHttpResponse,
) -> None:
    with pytest.raises(X14PolymarketRecorderError, match="request identity"):
        parse_target_universe(response, binding=_binding())


def test_universe_is_exact_and_builds_existing_x14_session_spec() -> None:
    universe = parse_target_universe(
        _http_response(), binding=_binding()
    )

    assert universe.game_identity_key == (
        "2026-08-07T00:00:00Z|CAR|ARI"
    )
    assert universe.market_ids == ("2858750", "2922959")
    assert universe.condition_ids == (
        MONEYLINE_CONDITION,
        TOTAL_CONDITION,
    )
    assert universe.token_ids == ("101", "102", "201", "202")
    assert universe.discovered_market_ids == universe.market_ids
    assert universe.selected_universe_equals_discovered is True
    assert universe.creation_at == "2026-07-08T11:00:00Z"
    assert universe.binding.polymarket_event_id == "684046"
    assert tuple(
        sorted(universe.binding.polymarket_token_outcomes)
    ) == universe.token_ids


def test_parser_uses_binding_identity_for_a_different_nfl_game() -> None:
    binding = replace(
        _binding(),
        binding_id="NFL-2026-BAL-BUF",
        game_id="2026_01_BAL_BUF",
        capture_session_id="nfl-2026-bal-buf",
        scheduled_start_utc="2026-09-10T17:00:00Z",
        away_team="BAL",
        home_team="BUF",
        sports_provider_game_id="provider-bal-buf",
        polymarket_event_id="990001",
        polymarket_event_slug="nfl-bal-buf-2026-09-10",
        polymarket_native_game_id="8801",
    )
    event = _event()
    event.update(
        id="990001",
        slug="nfl-bal-buf-2026-09-10",
        gameId=8801,
        startTime="2026-09-10T17:00:00Z",
        seriesSlug="nfl-2026",
    )
    event["teams"] = [
        {"abbreviation": "bal", "ordering": "away"},
        {"abbreviation": "buf", "ordering": "home"},
    ]
    for market in event["markets"]:
        market["gameStartTime"] = "2026-09-10T17:00:00Z"
    response = replace(
        _http_response(event),
        source_url=(
            "https://gamma-api.polymarket.com/events/slug/"
            "nfl-bal-buf-2026-09-10"
        ),
    )

    universe = parse_target_universe(response, binding=binding)

    assert universe.binding is binding
    assert universe.game_identity_key == (
        "2026-09-10T17:00:00Z|BAL|BUF"
    )


def test_production_recorder_has_no_game_specific_identity_constant() -> None:
    source = (
        PROJECT_ROOT
        / "src/prediction_market/sports/nfl_x14_polymarket_recorder.py"
    ).read_text(encoding="utf-8")

    assert "CAR" not in source
    assert "ARI" not in source
    assert "684046" not in source
    assert "2026-08-07" not in source


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda event: event.update(id="wrong"), "event.id"),
        (lambda event: event.update(gameId="wrong"), "gameId"),
        (lambda event: event.update(gameId=True), "gameId"),
        (lambda event: event.update(gameId=19454), "gameId"),
        (
            lambda event: event.update(startTime="2026-08-07T00:00:01Z"),
            "UTC start",
        ),
            (
                lambda event: event["teams"][0].update(abbreviation="buf"),
                "bound game",
            ),
    ],
)
def test_identity_mismatch_fails_closed(mutation: Any, message: str) -> None:
    event = _event()
    mutation(event)

    with pytest.raises(X14PolymarketRecorderError, match=message):
        parse_target_universe(
            _http_response(event), binding=_binding()
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda market: market.pop("clobTokenIds"),
            "token",
        ),
        (
            lambda market: market.update(sportsMarketType=None),
            "unknown",
        ),
        (
            lambda market: market.update(
                gameStartTime="2026-08-08T00:00:00Z"
            ),
            "cross-game",
        ),
        (
            lambda market: market.update(
                comboStatus="active",
                mveLegs=[{"eventId": "another-game"}],
            ),
            "MVE",
        ),
    ],
)
def test_missing_token_unknown_cross_game_and_mve_fail_closed(
    mutation: Any,
    message: str,
) -> None:
    event = _event()
    mutation(event["markets"][1])

    with pytest.raises(X14PolymarketRecorderError, match=message):
        parse_target_universe(
            _http_response(event), binding=_binding()
        )


def test_byte_exact_metadata_manifest_detects_tamper(tmp_path: Path) -> None:
    response = _http_response()
    record = preserve_target_metadata(
        response,
        binding=_binding(),
        raw_root=tmp_path,
        program_root=PROJECT_ROOT,
    )

    verified = verify_target_metadata(
        record.manifest_path,
        raw_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert verified.object_bytes == response.payload
    assert verified.record.manifest.source_url == (
        "https://gamma-api.polymarket.com/events/slug/"
        f"{EVENT_SLUG}"
    )
    assert verified.record.manifest.source_request["params"] == {}
    assert record.manifest.object_sha256 == (
        "sha256:" + hashlib.sha256(response.payload).hexdigest()
    )

    os.chmod(record.object_path, 0o644)
    record.object_path.write_bytes(b"tampered")
    with pytest.raises(StaticStoreError):
        verify_target_metadata(
            record.manifest_path,
            raw_root=tmp_path,
            program_root=PROJECT_ROOT,
        )


@pytest.mark.asyncio
async def test_identity_failure_preserves_raw_before_parse_and_never_captures(
    tmp_path: Path,
) -> None:
    event = _event()
    event["gameId"] = 19454
    response = _http_response(event)
    spy = _CaptureSpy()
    raw_root = tmp_path / "raw-store"
    report_root = tmp_path / "reports"

    with pytest.raises(X14PolymarketRecorderError, match="gameId"):
        await run_targeted_capture(
            binding=_binding(),
            gamma_response=response,
            raw_root=raw_root,
            report_root=report_root,
            program_root=PROJECT_ROOT,
            run_seconds=1,
            max_frames=1,
            receive_timeout_seconds=1,
            max_reconnects=0,
            capture=spy,
            object_store=None,
            s3_prefix=None,
        )

    manifests = list(raw_root.rglob("*.manifest.json"))
    assert len(manifests) == 1
    verified = verify_target_metadata(
        manifests[0],
        raw_root=raw_root,
        program_root=PROJECT_ROOT,
    )
    assert verified.object_bytes == response.payload
    assert spy.asset_ids is None
    assert not report_root.exists()


@pytest.mark.asyncio
async def test_bounded_run_wires_all_tokens_and_reports_partial_late_start(
    tmp_path: Path,
) -> None:
    spy = _CaptureSpy()
    result = await run_targeted_capture(
        binding=_binding(),
        gamma_response=_http_response(),
        raw_root=tmp_path / "raw-store",
        report_root=tmp_path / "reports",
        program_root=PROJECT_ROOT,
        run_seconds=1,
        max_frames=1,
        receive_timeout_seconds=1,
        max_reconnects=0,
        capture=spy,
        object_store=None,
        s3_prefix=None,
    )

    assert spy.asset_ids == ("101", "102", "201", "202")
    assert result.report["coverage"]["status"] == "PARTIAL_LATE_START"
    assert (
        result.report["coverage"]["market_creation_at"]
        == "2026-07-08T11:00:00Z"
    )
    assert result.report["coverage"]["complete_from_creation"] is False
    assert result.report["coverage"]["unobserved_pre_recorder_l2"] is True
    assert result.report["s3_upload"]["status"] == "NOT_CONFIGURED"
    assert result.report["gates"]["continuous_capture"] is False
    assert result.report["claims"]["creation_to_resolution_complete"] is False
    assert verify_session_report(result.report_record.path) == result.report
    assert (
        result.report_record.path.stem
        == result.report_record.sha256.removeprefix("sha256:")
    )


@pytest.mark.asyncio
async def test_bounded_run_subscribes_new_single_game_prop_not_in_binding(
    tmp_path: Path,
) -> None:
    event = _event()
    event["markets"].append(
        {
            "id": "player-prop-1",
            "conditionId": "0x" + "3" * 64,
            "question": "Will the CAR quarterback pass for 225+ yards?",
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["301","302"]',
            "sportsMarketType": "player_prop",
            "comboStatus": "disabled",
            "gameStartTime": "2026-08-07 00:00:00+00",
            "createdAt": "2026-07-25T12:06:00Z",
        }
    )
    spy = _CaptureSpy()

    result = await run_targeted_capture(
        binding=_binding(),
        gamma_response=_http_response(event),
        raw_root=tmp_path / "raw-store",
        report_root=tmp_path / "reports",
        program_root=PROJECT_ROOT,
        run_seconds=1,
        max_frames=1,
        receive_timeout_seconds=1,
        max_reconnects=0,
        capture=spy,
        object_store=None,
        s3_prefix=None,
    )

    assert spy.asset_ids == ("101", "102", "201", "202", "301", "302")
    assert result.universe.token_ids == spy.asset_ids


@pytest.mark.asyncio
async def test_verified_s3_upload_uses_segment_hash_shard(
    tmp_path: Path,
) -> None:
    spy = _CaptureSpy()
    store = _FakeObjectStore()
    result = await run_targeted_capture(
        binding=_binding(),
        gamma_response=_http_response(),
        raw_root=tmp_path / "raw-store",
        report_root=tmp_path / "reports",
        program_root=PROJECT_ROOT,
        run_seconds=1,
        max_frames=1,
        receive_timeout_seconds=1,
        max_reconnects=0,
        capture=spy,
        object_store=store,
        s3_prefix="prediction-market",
    )

    segment = result.supervisor_result.manifests[0]
    digest = segment.object_sha256.removeprefix("sha256:")
    expected_key = (
        "prediction-market/raw/polymarket-x14/sha256/"
        f"{digest[:2]}/{digest}.jsonl.zst"
    )
    assert store.puts == [expected_key]
    assert store.metadata[expected_key]["sha256"] == segment.object_sha256
    assert result.report["s3_upload"]["status"] == "VERIFIED"

    with pytest.raises(X14PolymarketRecorderError, match="HEAD"):
        upload_capture_segments(
            result.supervisor_result,
            object_store=_FakeObjectStore(corrupt_head=True),
            prefix="prediction-market",
        )


@pytest.mark.asyncio
async def test_same_size_s3_readback_corruption_is_not_verified(
    tmp_path: Path,
) -> None:
    result = await run_targeted_capture(
        binding=_binding(),
        gamma_response=_http_response(),
        raw_root=tmp_path / "raw-store",
        report_root=tmp_path / "reports",
        program_root=PROJECT_ROOT,
        run_seconds=1,
        max_frames=1,
        receive_timeout_seconds=1,
        max_reconnects=0,
        capture=_CaptureSpy(),
        object_store=_FakeObjectStore(corrupt_readback=True),
        s3_prefix="prediction-market",
    )

    assert result.report["s3_upload"]["status"] == "HEAD_CONFIRMED"
    assert result.report["s3_upload"]["objects"][0][
        "readback_sha256_matches"
    ] is False
    assert result.report["gates"]["continuous_capture"] is False


def test_cli_requires_an_explicit_positive_bounded_run() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--raw-root",
            "/tmp/x14-raw",
            "--report-root",
            "/tmp/x14-reports",
            "--binding",
            "/tmp/x14-binding.json",
            "--run-seconds",
            "1",
        ]
    )
    assert args.run_seconds == 1.0
    assert args.max_frames is None

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--raw-root",
                "/tmp/x14-raw",
                "--report-root",
                "/tmp/x14-reports",
                "--binding",
                "/tmp/x14-binding.json",
            ]
        )
