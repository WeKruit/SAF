from __future__ import annotations

import base64
import hashlib
import json
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import zstandard

from prediction_market.contracts import CaptureTimingV0
from prediction_market.raw_store import verify_segment
from prediction_market.sports.nfl_x14_live_recorder import (
    LiveCaptureState,
    UnknownMarketUniverseError,
    build_live_parser,
    inspect_target_refresh,
    parse_live_market_frame,
    run_live_capture,
    upload_live_artifacts,
)
from prediction_market.sports.nfl_x14_polymarket_recorder import (
    GammaHttpResponse,
    parse_target_universe,
)
from prediction_market.sports.nfl_x14_reaction import (
    CaptureClockContextV1,
    InformationEventV1,
    ProspectiveGameBindingV1,
    measure_reaction,
)


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
                    "101": "Yes",
                    "102": "No",
                    "201": "Yes",
                    "202": "No",
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


def _market(
    market_id: str,
    condition_id: str,
    tokens: tuple[str, str],
    market_type: str,
    *,
    closed: bool = False,
    outcome_prices: str = '["0.5","0.5"]',
) -> dict[str, Any]:
    return {
        "id": market_id,
        "conditionId": condition_id,
        "question": f"fixture {market_id}",
        "outcomes": '["Yes","No"]',
        "outcomePrices": outcome_prices,
        "clobTokenIds": json.dumps(list(tokens), separators=(",", ":")),
        "sportsMarketType": market_type,
        "comboStatus": "disabled",
        "gameStartTime": "2026-08-07 00:00:00+00",
        "createdAt": "2026-07-09T12:06:00Z",
        "closed": closed,
    }


def _event(*, closed: bool = False) -> dict[str, Any]:
    return {
        "id": "684046",
        "slug": EVENT_SLUG,
        "gameId": 19453,
        "seriesSlug": "nfl-2026",
        "createdAt": "2026-07-08T11:00:00Z",
        "startTime": "2026-08-07T00:00:00Z",
        "closed": closed,
        "sport": {"sport": "nfl"},
        "teams": [
            {"abbreviation": "car", "ordering": "away"},
            {"abbreviation": "ari", "ordering": "home"},
        ],
        "markets": [
            _market(
                "2858750",
                MONEYLINE_CONDITION,
                ("101", "102"),
                "moneyline",
                closed=closed,
                outcome_prices='["1","0"]' if closed else '["0.5","0.5"]',
            ),
            _market(
                "2922959",
                TOTAL_CONDITION,
                ("201", "202"),
                "totals",
                closed=closed,
                outcome_prices='["0","1"]' if closed else '["0.5","0.5"]',
            ),
        ],
    }


def _response(event: dict[str, Any]) -> GammaHttpResponse:
    return GammaHttpResponse(
        payload=json.dumps(event, separators=(",", ":")).encode(),
        fetched_at="2026-07-25T18:00:00Z",
        source_url=(
            "https://gamma-api.polymarket.com/events/slug/"
            f"{EVENT_SLUG}"
        ),
        request_params={},
        headers={"content-type": "application/json"},
    )


def _decoded_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            outer = [json.loads(row) for row in reader.read().splitlines()]
    return [
        json.loads(base64.b64decode(row["payload_base64"], validate=True))
        for row in outer
    ]


def test_refresh_accepts_single_game_player_prop_as_primitive() -> None:
    event = _event()
    event["markets"].append(
        _market(
            "future-player-prop",
            "0x" + "a" * 64,
            ("301", "302"),
            "player-prop",
        )
    )

    inspection = inspect_target_refresh(
        _response(event), binding=_binding()
    )

    assert inspection.universe is not None
    assert inspection.resolution_observed is False
    assert inspection.unknown_markets == ()
    assert inspection.universe.market_ids[-1] == "future-player-prop"


def test_cli_defaults_to_formal_until_resolution_mode() -> None:
    args = build_live_parser().parse_args(
        [
            "--raw-root",
            "/tmp/x14-live-raw",
            "--report-root",
            "/tmp/x14-live-reports",
            "--binding",
            "/tmp/x14-binding.json",
            "--host-id",
            "capture-host-a",
            "--boot-id",
            "boot-a",
            "--monotonic-clock-id",
            "CLOCK_MONOTONIC_RAW",
            "--clock-epoch-id",
            "clock-epoch-a",
            "--clock-sync-sample-id",
            "sync-a",
            "--pairing-uncertainty-ns",
            "100",
        ]
    )

    assert args.max_runtime_seconds is None
    assert args.gamma_refresh_seconds == 300.0


def test_refresh_proves_resolution_only_from_closed_binary_prices() -> None:
    open_inspection = inspect_target_refresh(
        _response(_event()), binding=_binding()
    )
    closed_inspection = inspect_target_refresh(
        _response(_event(closed=True)), binding=_binding()
    )

    assert open_inspection.universe == parse_target_universe(
        _response(_event()), binding=_binding()
    )
    assert open_inspection.resolution_observed is False
    assert closed_inspection.resolution_observed is True

    unresolved = _event(closed=True)
    unresolved["markets"][0]["outcomePrices"] = '["0.6","0.4"]'
    assert (
        inspect_target_refresh(
            _response(unresolved), binding=_binding()
        ).resolution_observed
        is False
    )


def test_frame_parser_emits_source_and_local_time_for_each_token() -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    frame = json.dumps(
        {
            "event_type": "price_change",
            "market": MONEYLINE_CONDITION,
            "timestamp": "1786060800123",
            "price_changes": [
                {"asset_id": "101", "price": "0.6", "size": "10"},
                {"asset_id": "102", "price": "0.4", "size": "10"},
            ],
        },
        separators=(",", ":"),
    ).encode()

    records = parse_live_market_frame(
        frame,
        universe=universe,
        epoch=3,
        local_received_time="2026-08-07T00:00:00.250000Z",
    )

    assert len(records) == 2
    assert {record.token_id for record in records} == {"101", "102"}
    assert {record.market_id for record in records} == {"2858750"}
    assert {record.condition_id for record in records} == {
        MONEYLINE_CONDITION
    }
    assert {record.event_kind for record in records} == {"market_delta"}
    assert {record.epoch for record in records} == {3}
    assert {record.source_received_time for record in records} == {
        "2026-08-07T00:00:00.123000Z"
    }
    assert {record.local_received_time for record in records} == {
        "2026-08-07T00:00:00.250000Z"
    }


@pytest.mark.parametrize(
    "payload",
    [
        b'{"event_type":"book","asset_id":"101"}',
        b'{"event_type":"book","asset_id":"999","timestamp":"1786060800123"}',
        b'{"event_type":"mystery","asset_id":"101","timestamp":"1786060800123"}',
    ],
)
def test_frame_parser_fails_closed_on_missing_time_token_or_kind(
    payload: bytes,
) -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    with pytest.raises(ValueError):
        parse_live_market_frame(
            payload,
            universe=universe,
            epoch=0,
            local_received_time="2026-08-07T00:00:00.250000Z",
        )


def test_capture_state_rotates_hour_and_epoch_and_requires_new_snapshot(
    tmp_path: Path,
) -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    state = LiveCaptureState(raw_root=tmp_path, universe=universe)

    first = parse_live_market_frame(
        b'{"event_type":"book","asset_id":"101",'
        b'"market":"' + MONEYLINE_CONDITION.encode() + b'",'
        b'"timestamp":"1786060799900","bids":[],"asks":[]}',
        universe=universe,
        epoch=0,
        local_received_time="2026-08-06T23:59:59.950000Z",
    )
    state.append_native_and_canonical(
        native_payload=first[0].native_payload,
        local_received_time=first[0].local_received_time,
        records=first,
    )

    second = parse_live_market_frame(
        b'{"event_type":"book","asset_id":"201",'
        b'"market":"' + TOTAL_CONDITION.encode() + b'",'
        b'"timestamp":"1786060800100","bids":[],"asks":[]}',
        universe=universe,
        epoch=0,
        local_received_time="2026-08-07T00:00:00.150000Z",
    )
    state.append_native_and_canonical(
        native_payload=second[0].native_payload,
        local_received_time=second[0].local_received_time,
        records=second,
    )
    state.start_new_epoch(
        reason="reconnect", required_tokens=universe.token_ids
    )

    delta = parse_live_market_frame(
        b'{"event_type":"price_change","asset_id":"101",'
        b'"market":"' + MONEYLINE_CONDITION.encode() + b'",'
        b'"timestamp":"1786060800200","price":"0.7"}',
        universe=universe,
        epoch=1,
        local_received_time="2026-08-07T00:00:00.250000Z",
    )
    with pytest.raises(ValueError, match="snapshot"):
        state.append_native_and_canonical(
            native_payload=delta[0].native_payload,
            local_received_time=delta[0].local_received_time,
            records=delta,
        )

    manifests = state.close()
    assert state.audit["epoch_count"] == 2
    assert state.audit["gap_count"] == 1
    assert state.audit["new_snapshot_required_count"] == 1
    assert state.audit["snapshot_coverage_complete"] is False
    assert "102" in state.audit["missing_snapshot_tokens_by_epoch"]["0"]
    assert {manifest.partition_hour for manifest in manifests} == {
        "00",
        "23",
    }
    assert all(verify_segment(row.path, root=tmp_path).valid for row in manifests)
    canonical = [
        record
        for manifest in manifests
        for record in _decoded_records(manifest.object_path)
        if record.get("capture_contract") == "nfl-x14-prospective/v0"
    ]
    assert {record["epoch"] for record in canonical} == {0}
    assert all("source_received_time" in record for record in canonical)
    assert all("local_received_time" in record for record in canonical)


def test_complete_continuity_requires_every_token_snapshot_and_no_gap(
    tmp_path: Path,
) -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    state = LiveCaptureState(raw_root=tmp_path, universe=universe)
    for token in universe.token_ids:
        market = next(
            row for row in universe.markets if token in row.token_ids
        )
        records = parse_live_market_frame(
            json.dumps(
                {
                    "event_type": "book",
                    "asset_id": token,
                    "market": market.condition_id,
                    "timestamp": "1786060800100",
                    "tick_size": "0.01",
                    "bids": [{"price": "0.49", "size": "10"}],
                    "asks": [{"price": "0.51", "size": "10"}],
                },
                separators=(",", ":"),
            ).encode(),
            universe=universe,
            epoch=0,
            local_received_time="2026-08-07T00:00:00.150000Z",
        )
        state.append_native_and_canonical(
            native_payload=records[0].native_payload,
            local_received_time=records[0].local_received_time,
            records=records,
        )

    assert state.audit["snapshot_coverage_complete"] is True
    assert state.audit["continuity_complete"] is True
    state.start_new_epoch(
        reason="reconnect",
        audit_reason="receive_idle_timeout",
        required_tokens=universe.token_ids,
    )
    assert state.audit["continuity_complete"] is False
    assert state.audit["snapshot_coverage_complete"] is False


def test_universe_expansion_rebinds_normalizer_before_new_epoch(
    tmp_path: Path,
) -> None:
    initial = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    expanded_event = _event()
    expanded_event["markets"].append(
        _market(
            "future-player-prop",
            "0x" + "a" * 64,
            ("301", "302"),
            "player-prop",
        )
    )
    expanded = parse_target_universe(
        _response(expanded_event), binding=_binding()
    )
    state = LiveCaptureState(raw_root=tmp_path, universe=initial)

    state.expand_universe(expanded)
    state.start_new_epoch(
        reason="reconnect",
        audit_reason="universe_expansion",
        required_tokens=expanded.token_ids,
    )
    updates = state.ingest_timed_market_frame(
        native_payload=json.dumps(
            {
                "event_type": "book",
                "asset_id": "301",
                "market": "0x" + "a" * 64,
                "timestamp": "1786060800100",
                "tick_size": "0.01",
                "bids": [{"price": "0.49", "size": "10"}],
                "asks": [{"price": "0.51", "size": "10"}],
            },
            separators=(",", ":"),
        ).encode(),
        local_received_time="2026-08-07T00:00:00.150000Z",
        clock_context=_clock_context(),
        monotonic_clock=_ticks([1, 2, 3, 4, 5]),
    )

    assert updates[0].token_or_ticker == "301"
    assert updates[0].logical_market_id == "future-player-prop"


def _clock_context() -> CaptureClockContextV1:
    return CaptureClockContextV1(
        host_id="capture-host-a",
        boot_id="boot-a",
        monotonic_clock_id="CLOCK_MONOTONIC_RAW",
        clock_epoch_id="clock-epoch-a",
        clock_sync_sample_id="sync-a",
        pairing_uncertainty_ns=100,
    )


def _ticks(values: list[int]) -> Any:
    iterator = iter(values)
    return lambda: next(iterator)


def test_recorder_to_reaction_e2e_uses_linked_timing_and_book_ready(
    tmp_path: Path,
) -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    state = LiveCaptureState(raw_root=tmp_path, universe=universe)

    def ingest(
        ordinal: int, receive_ns: int, bid: str, ask: str
    ) -> None:
        minute, second = divmod(
            receive_ns // 1_000_000_000, 60
        )
        payload = json.dumps(
            {
                "event_type": "book",
                "asset_id": "101",
                "market": MONEYLINE_CONDITION,
                "timestamp": str(
                    1_786_060_800_000 + receive_ns // 1_000_000
                ),
                "tick_size": "0.01",
                "bids": [{"price": bid, "size": "100"}],
                "asks": [{"price": ask, "size": "100"}],
                "hash": f"book-{ordinal}",
            },
            separators=(",", ":"),
        ).encode()
        state.ingest_timed_market_frame(
            native_payload=payload,
            local_received_time=(
                f"2026-08-07T00:{minute:02d}:"
                f"{second:02d}.000000Z"
            ),
            clock_context=_clock_context(),
            monotonic_clock=_ticks(
                [
                    receive_ns,
                    receive_ns + 10,
                    receive_ns + 20,
                    receive_ns + 30,
                    receive_ns + 100,
                ]
            ),
        )

    ingest(0, 9_000_000_000, ".39", ".41")
    ingest(1, 11_000_000_000, ".45", ".47")
    ingest(2, 13_000_000_000, ".49", ".51")
    ingest(3, 65_000_000_000, ".49", ".51")
    state.close()

    payload = b'{"event":"touchdown"}'
    timing = CaptureTimingV0(
        timing_version="v0",
        capture_session_id=_binding().capture_session_id,
        record_ordinal=99,
        payload_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        receive_boundary="application_callback",
        local_receive_utc_ns=1_786_060_810_000_000_000,
        local_receive_monotonic_ns=10_000_000_000,
        pairing_uncertainty_ns=100,
        host_id="capture-host-a",
        boot_id="boot-a",
        monotonic_clock_id="CLOCK_MONOTONIC_RAW",
        clock_epoch_id="clock-epoch-a",
        connection_epoch=0,
        clock_sync_sample_id="sync-a",
        raw_append_done_monotonic_ns=10_000_000_010,
        parse_done_monotonic_ns=10_000_000_020,
        normalize_done_monotonic_ns=10_000_000_030,
        state_or_book_ready_monotonic_ns=10_000_000_100,
    )
    event = InformationEventV1(
        event_id="episode-1",
        game_id=_binding().game_id,
        provider="licensed-feed",
        provider_event_id="play-1",
        revision=1,
        revision_status="FINALIZED",
        timing=timing,
        provider_publish_time_utc_ns=None,
        finalized_monotonic_ns=10_000_000_110,
        pre_state={"home_score": 0},
        post_state={"home_score": 7},
        episode_kind="touchdown",
        salient=True,
        review=False,
        penalty=False,
        no_play=False,
    )
    updates = tuple(
        row for row in state.market_updates if row.token_or_ticker == "101"
    )
    result = measure_reaction(
        event=event,
        updates=updates,
        venue_tick=Decimal(".01"),
        control_noise_95=Decimal(".01"),
        next_salient_event_monotonic_ns=None,
    )

    envelope_ids = {row.envelope_id for row in state.finalized_envelopes}
    assert envelope_ids
    assert {row.raw_envelope_id for row in updates} <= envelope_ids
    assert all(
        row.timing.state_or_book_ready_monotonic_ns
        > row.timing.local_receive_monotonic_ns
        for row in updates
    )
    assert result.first_any_market_update_monotonic_ns == 11_000_000_100
    assert result.bid_first_passage_t90_ns == 3_000_000_000
    assert result.ask_first_passage_t90_ns == 3_000_000_000


def test_tick_size_change_is_canonical_and_controls_current_book_tick(
    tmp_path: Path,
) -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    state = LiveCaptureState(raw_root=tmp_path, universe=universe)
    snapshot = json.dumps(
        {
            "event_type": "book",
            "asset_id": "101",
            "market": MONEYLINE_CONDITION,
            "timestamp": "1786060800100",
            "tick_size": "0.01",
            "bids": [{"price": "0.49", "size": "10"}],
            "asks": [{"price": "0.51", "size": "10"}],
        },
        separators=(",", ":"),
    ).encode()
    state.ingest_timed_market_frame(
        native_payload=snapshot,
        local_received_time="2026-08-07T00:00:00.150000Z",
        clock_context=_clock_context(),
        monotonic_clock=_ticks([1, 2, 3, 4, 5]),
    )
    tick_change = json.dumps(
        {
            "event_type": "tick_size_change",
            "asset_id": "101",
            "market": MONEYLINE_CONDITION,
            "timestamp": "1786060800200",
            "old_tick_size": "0.01",
            "new_tick_size": "0.05",
        },
        separators=(",", ":"),
    ).encode()

    updates = state.ingest_timed_market_frame(
        native_payload=tick_change,
        local_received_time="2026-08-07T00:00:00.250000Z",
        clock_context=_clock_context(),
        monotonic_clock=_ticks([6, 7, 8, 9, 10]),
    )
    state.close()

    assert updates[0].update_kind == "tick_size"
    assert updates[0].tick_size == Decimal("0.05")
    assert updates[0].raw_envelope_id in {
        row.envelope_id for row in state.finalized_envelopes
    }


class _ObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def head_object(self, key: str) -> dict[str, object] | None:
        if key not in self.objects:
            return None
        return {
            "byte_size": len(self.objects[key]),
            "metadata": self.metadata[key],
        }

    def put_file(
        self, key: str, path: Path, *, metadata: dict[str, str]
    ) -> None:
        self.objects[key] = path.read_bytes()
        self.metadata[key] = dict(metadata)

    def read_object_chunks(
        self, key: str, *, chunk_size: int = 1024 * 1024
    ) -> Any:
        value = self.objects[key]
        return (
            value[offset : offset + chunk_size]
            for offset in range(0, len(value), chunk_size)
        )


def test_s3_upload_covers_raw_sidecar_metadata_and_report(tmp_path: Path) -> None:
    universe = parse_target_universe(
        _response(_event()), binding=_binding()
    )
    state = LiveCaptureState(raw_root=tmp_path / "raw", universe=universe)
    records = parse_live_market_frame(
        b'{"event_type":"book","asset_id":"101",'
        b'"market":"' + MONEYLINE_CONDITION.encode() + b'",'
        b'"timestamp":"1786060800100","bids":[],"asks":[]}',
        universe=universe,
        epoch=0,
        local_received_time="2026-08-07T00:00:00.150000Z",
    )
    state.append_native_and_canonical(
        native_payload=records[0].native_payload,
        local_received_time=records[0].local_received_time,
        records=records,
    )
    manifests = state.close()
    metadata = tmp_path / "metadata.json"
    report = tmp_path / "report.json"
    metadata.write_text('{"metadata":true}\n', encoding="utf-8")
    report.write_text('{"report":true}\n', encoding="utf-8")
    store = _ObjectStore()

    result = upload_live_artifacts(
        manifests=manifests,
        metadata_paths=(metadata,),
        report_paths=(report,),
        object_store=store,
        prefix="prediction-market",
    )

    assert result["status"] == "VERIFIED"
    assert result["counts"] == {
        "raw": len(manifests),
        "sidecar_manifest": len(manifests),
        "metadata": 1,
        "report": 1,
    }
    assert any("/raw/polymarket-x14/" in key for key in store.objects)
    assert any("/manifests/polymarket-x14/" in key for key in store.objects)
    assert any("/metadata/polymarket-x14/" in key for key in store.objects)
    assert any("/reports/polymarket-x14/" in key for key in store.objects)


class _IdleSocket:
    def __init__(self, subscriptions: list[tuple[str, ...]]) -> None:
        self.subscriptions = subscriptions

    async def send(self, value: str) -> None:
        document = json.loads(value)
        self.subscriptions.append(tuple(document["assets_ids"]))

    async def recv(self) -> bytes:
        # The runner's refresh timer must interrupt this idle socket.
        await __import__("asyncio").sleep(60)
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_long_runner_refreshes_universe_reconnects_and_stops_on_resolution(
    tmp_path: Path,
) -> None:
    expanded = _event()
    expanded["markets"].append(
        _market(
            "3000000",
            "0x" + "b" * 64,
            ("301", "302"),
            "spreads",
        )
    )
    resolved = json.loads(json.dumps(expanded))
    resolved["closed"] = True
    for index, market in enumerate(resolved["markets"]):
        market["closed"] = True
        market["outcomePrices"] = (
            '["1","0"]' if index % 2 == 0 else '["0","1"]'
        )
    responses = iter(
        [_response(_event()), _response(expanded), _response(resolved)]
    )

    async def gamma_fetch() -> GammaHttpResponse:
        return next(responses)

    subscriptions: list[tuple[str, ...]] = []

    @asynccontextmanager
    async def connect(_tokens: tuple[str, ...]) -> Any:
        yield _IdleSocket(subscriptions)

    result = await run_live_capture(
        binding=_binding(),
        gamma_fetch=gamma_fetch,
        connect=connect,
        raw_root=tmp_path / "raw",
        report_root=tmp_path / "reports",
        program_root=PROJECT_ROOT,
        clock_context=_clock_context(),
        gamma_refresh_seconds=0.001,
        receive_timeout_seconds=1,
        reconnect_backoff_seconds=0,
    )

    assert result.resolution_observed is True
    assert result.terminal_reason == "Gamma resolution observed"
    assert subscriptions == [
        ("101", "102", "201", "202"),
        ("101", "102", "201", "202", "301", "302"),
    ]
    assert result.audit["epoch_count"] == 2
    assert result.audit["epochs"][1]["reason"] == "universe_expansion"
    assert len(result.metadata_records) == 3
    assert result.report_path.exists()
    report = json.loads(result.report_path.read_text())
    assert report["coverage"]["status"] == "PARTIAL_LATE_START"
    assert report["claims"]["creation_to_resolution_complete"] is False
    assert report["claims"]["local_start_to_resolution_complete"] is False


@pytest.mark.asyncio
async def test_long_runner_preserves_unknown_inventory_and_never_connects(
    tmp_path: Path,
) -> None:
    event = _event()
    event["markets"].append(
        _market(
            "future-composite",
            "0x" + "a" * 64,
            ("301", "302"),
            "player-prop",
        )
    )
    event["markets"][-1]["comboStatus"] = "active"
    event["markets"][-1]["mveLegs"] = [{"eventId": "another-game"}]

    async def gamma_fetch() -> GammaHttpResponse:
        return _response(event)

    connected = False

    @asynccontextmanager
    async def connect(_tokens: tuple[str, ...]) -> Any:
        nonlocal connected
        connected = True
        yield _IdleSocket([])

    with pytest.raises(UnknownMarketUniverseError) as captured:
        await run_live_capture(
            binding=_binding(),
            gamma_fetch=gamma_fetch,
            connect=connect,
            raw_root=tmp_path / "raw",
            report_root=tmp_path / "reports",
            program_root=PROJECT_ROOT,
            clock_context=_clock_context(),
            gamma_refresh_seconds=1,
        )

    assert connected is False
    assert captured.value.inventory_path is not None
    inventory = json.loads(captured.value.inventory_path.read_text())
    assert inventory["markets"][0]["market_id"] == "future-composite"
    assert list((tmp_path / "raw").rglob("*.manifest.json"))
