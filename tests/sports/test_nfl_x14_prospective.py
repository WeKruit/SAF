from __future__ import annotations

import base64
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
import zstandard

from prediction_market.experiments import load_experiment_registry
from prediction_market.raw_store import verify_segment
from prediction_market.sports.nfl_x14_prospective import (
    CaptureContinuityError,
    CaptureMode,
    ProspectiveCaptureRecord,
    ProspectiveCaptureSession,
    ProspectiveCaptureValidator,
    ProspectiveSessionSpec,
    validate_association_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _at(second: int, *, minute: int = 0, hour: int = 17) -> str:
    return (
        datetime(2026, 9, 10, hour, minute, second, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _market_record(
    *,
    kind: str,
    epoch: int = 0,
    sequence: int | None = None,
    second: int = 1,
    minute: int = 0,
    capture_mode: CaptureMode = CaptureMode.PROSPECTIVE_L2,
    local_received_time: str | None = None,
) -> ProspectiveCaptureRecord:
    return ProspectiveCaptureRecord(
        experiment_id="X-14",
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
        epoch=epoch,
        capture_mode=capture_mode,
        record_class="market",
        event_kind=kind,
        source="polymarket",
        source_received_time=_at(second, minute=minute),
        local_received_time=(
            _at(second + 1, minute=minute)
            if local_received_time is None
            else local_received_time
        ),
        source_sequence=sequence,
        market_id="0xcondition",
        condition_id="0xcondition",
        token_id="token-a",
        play_id=None,
        play_revision=None,
        native_payload=b'{"event":"native"}',
    )


def _sports_record(
    *,
    epoch: int = 0,
    revision: int = 0,
    second: int = 1,
) -> ProspectiveCaptureRecord:
    return ProspectiveCaptureRecord(
        experiment_id="X-14",
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
        epoch=epoch,
        capture_mode=CaptureMode.PROSPECTIVE_L2,
        record_class="sports",
        event_kind="play_revision",
        source="licensed-game-feed",
        source_received_time=_at(second),
        local_received_time=_at(second + 1),
        source_sequence=None,
        market_id=None,
        condition_id=None,
        token_id=None,
        play_id="play-0001",
        play_revision=revision,
        native_payload=b'{"down":1,"yards_to_go":10}',
    )


def _segment_records(object_path: Path) -> list[dict[str, object]]:
    with object_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            lines = reader.read().splitlines()
    outer = [json.loads(line) for line in lines]
    return [
        json.loads(base64.b64decode(row["payload_base64"], validate=True))
        for row in outer
    ]


def _freeze_spec(
    *,
    session_id: str = "nfl-2026-car-ari",
    game_id: str = "nfl-2026-08-07-car-ari",
    polymarket_condition_ids: tuple[str, ...] = ("condition-a",),
    discovered_single_game_condition_ids: tuple[str, ...] = ("condition-a",),
    polymarket_token_ids: tuple[str, ...] = ("token-a",),
    discovered_single_game_token_ids: tuple[str, ...] = ("token-a",),
    kalshi_credentials_available: bool = False,
    kalshi_market_ids: tuple[str, ...] = (),
) -> ProspectiveSessionSpec:
    return ProspectiveSessionSpec.freeze(
        session_id=session_id,
        game_id=game_id,
        scheduled_start_utc="2026-08-07T00:00:00Z",
        away_team="CAR",
        home_team="ARI",
        polymarket_event_id="684046",
        polymarket_event_slug="nfl-car-ari-2026-08-07",
        polymarket_native_game_id="19453",
        kalshi_event_id="KXNFLGAME-26AUG06CARARI",
        polymarket_condition_ids=polymarket_condition_ids,
        discovered_single_game_condition_ids=(
            discovered_single_game_condition_ids
        ),
        polymarket_token_ids=polymarket_token_ids,
        discovered_single_game_token_ids=discovered_single_game_token_ids,
        kalshi_credentials_available=kalshi_credentials_available,
        kalshi_market_ids=kalshi_market_ids,
    )


def test_x14_is_a_canonical_preliminary_registration() -> None:
    registry = load_experiment_registry(PROJECT_ROOT)
    card = registry["X-14"]

    assert card["status"] == "registered"
    assert card["execution_authorized"] is True
    assert card["completion_required_scopes"] == ["prospective_capture"]
    assert card["authorization_scopes"]["prospective_capture"][
        "required_result_label"
    ] == "PRELIMINARY"
    assert card["authorization_scopes"]["prospective_capture"][
        "input_binding"
    ]["dataset_ids"] == ["DS-POLYMARKET-PUBLIC"]
    kalshi_scope = card["authorization_scopes"][
        "kalshi_prospective_capture"
    ]
    assert kalshi_scope["authorized"] is True
    assert "kalshi_credentials" in kalshi_scope["required_lock_ids"]
    assert kalshi_scope["input_binding"]["dataset_ids"] == [
        "DS-KALSHI-LIVE-L2"
    ]
    assert "X-08" in card["dependencies"]
    assert card["source_time_only"] is False
    assert card["causal_or_execution_claims_authorized"] is False

    raw = yaml.safe_load(
        (PROJECT_ROOT / "registries/experiments/X-14.yaml").read_text(
            encoding="utf-8"
        )
    )["experiment_card"]
    locks = {item["id"]: item for item in raw["registration_locks"]}
    assert locks["polymarket_single_game_token_universe"]["status"] == "unresolved"
    assert locks["sports_feed_and_timestamp_contract"]["status"] == "unresolved"
    assert locks["kalshi_credentials"]["status"] == "unresolved"

    with (
        PROJECT_ROOT / "registries/experiment_amendment_ledger.csv"
    ).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    genesis = [row for row in rows if row["experiment_id"] == "X-14"]
    assert genesis == [
        {
            "experiment_id": "X-14",
            "sequence": "0",
            "record_sha256": raw["registration_record_sha256"],
            "prior_sha256": "",
            "amended_at": "",
            "approved_by": "",
            "reason": "",
        }
    ]


def test_session_spec_requires_exact_polymarket_single_game_universe() -> None:
    spec = _freeze_spec(
        polymarket_token_ids=("token-a", "token-b"),
        discovered_single_game_token_ids=("token-b", "token-a"),
    )

    assert spec.polymarket_token_ids == ("token-a", "token-b")
    assert spec.start_boundary == "market_creation"
    assert spec.end_boundary == "market_resolution"
    assert spec.kalshi_enabled is False
    assert spec.identity_key == (
        "2026-08-07T00:00:00Z|CAR|ARI"
    )
    assert spec.polymarket_event_id == "684046"
    assert spec.polymarket_event_slug == "nfl-car-ari-2026-08-07"
    assert spec.polymarket_native_game_id == "19453"
    assert spec.kalshi_event_id == "KXNFLGAME-26AUG06CARARI"
    assert spec.venue_date_suffix_is_identity is False

    with pytest.raises(ValueError, match="exactly equal"):
        _freeze_spec(
            polymarket_token_ids=("token-a",),
            discovered_single_game_token_ids=("token-a", "token-b"),
        )

    with pytest.raises(ValueError, match="condition universe"):
        _freeze_spec(
            polymarket_condition_ids=("condition-a",),
            discovered_single_game_condition_ids=(
                "condition-a",
                "condition-b",
            ),
        )


def test_kalshi_is_enabled_only_when_credentials_are_available() -> None:
    with pytest.raises(ValueError, match="credentials"):
        _freeze_spec(
            kalshi_credentials_available=False,
            kalshi_market_ids=("KXNFLGAME-26SEP10BALBUF-BAL",),
        )

    spec = _freeze_spec(
        kalshi_credentials_available=True,
        kalshi_market_ids=("KXNFLGAME-26SEP10BALBUF-BAL",),
    )
    assert spec.kalshi_enabled is True


@pytest.mark.parametrize(
    "kind", ["market_snapshot", "market_delta", "bbo", "trade"]
)
def test_market_records_preserve_required_kinds_and_both_times(kind: str) -> None:
    record = _market_record(kind=kind)
    document = json.loads(record.to_bytes())

    assert document["event_kind"] == kind
    assert document["source_received_time"] == _at(1)
    assert document["local_received_time"] == _at(2)
    assert document["condition_id"] == "0xcondition"
    assert document["token_id"] == "token-a"
    assert base64.b64decode(
        document["native_payload_base64"], validate=True
    ) == b'{"event":"native"}'

    with pytest.raises(ValueError, match="condition_id and token_id"):
        replace(record, token_id=None)


def test_sports_record_requires_play_revision_and_both_times() -> None:
    assert _sports_record(revision=2).play_revision == 2

    with pytest.raises(ValueError, match="play_revision"):
        ProspectiveCaptureRecord(
            experiment_id="X-14",
            session_id="nfl-2026-bal-buf",
            game_id="2026_01_BAL_BUF",
            epoch=0,
            capture_mode=CaptureMode.PROSPECTIVE_L2,
            record_class="sports",
            event_kind="play_revision",
            source="licensed-game-feed",
            source_received_time=_at(1),
            local_received_time=_at(2),
            source_sequence=None,
            market_id=None,
            condition_id=None,
            token_id=None,
            play_id="play-0001",
            play_revision=None,
            native_payload=b"{}",
        )

    with pytest.raises(ValueError, match="source_received_time"):
        ProspectiveCaptureRecord(
            experiment_id="X-14",
            session_id="nfl-2026-bal-buf",
            game_id="2026_01_BAL_BUF",
            epoch=0,
            capture_mode=CaptureMode.PROSPECTIVE_L2,
            record_class="sports",
            event_kind="play_revision",
            source="licensed-game-feed",
            source_received_time="",
            local_received_time=_at(2),
            source_sequence=None,
            market_id=None,
            condition_id=None,
            token_id=None,
            play_id="play-0001",
            play_revision=0,
            native_payload=b"{}",
        )


def test_sports_play_revisions_cannot_regress_within_an_epoch() -> None:
    validator = ProspectiveCaptureValidator(
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
    )
    validator.observe(_sports_record(revision=1))

    with pytest.raises(CaptureContinuityError, match="revision"):
        validator.observe(_sports_record(revision=0, second=3))
    with pytest.raises(CaptureContinuityError, match="blocked"):
        validator.observe(_sports_record(revision=2, second=5))


def test_gap_blocks_epoch_and_new_epoch_must_restart_from_snapshot() -> None:
    validator = ProspectiveCaptureValidator(
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
    )
    validator.observe(_market_record(kind="market_snapshot", sequence=10))
    validator.observe(_market_record(kind="market_delta", sequence=11))

    with pytest.raises(CaptureContinuityError, match="gap"):
        validator.observe(_market_record(kind="market_delta", sequence=13))
    with pytest.raises(CaptureContinuityError, match="blocked"):
        validator.observe(_market_record(kind="market_delta", sequence=14))

    assert validator.start_new_epoch(reason="sequence_gap") == 1
    with pytest.raises(CaptureContinuityError, match="snapshot"):
        validator.observe(
            _market_record(kind="market_delta", epoch=1, sequence=14)
        )
    with pytest.raises(CaptureContinuityError, match="blocked"):
        validator.observe(
            _market_record(kind="market_snapshot", epoch=1, sequence=14)
        )
    assert validator.start_new_epoch(reason="reconnect") == 2
    validator.observe(
        _market_record(kind="market_snapshot", epoch=2, sequence=14)
    )


def test_reconnect_always_starts_a_new_epoch() -> None:
    validator = ProspectiveCaptureValidator(
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
    )
    validator.observe(_market_record(kind="market_snapshot", sequence=1))

    assert validator.start_new_epoch(reason="reconnect") == 1
    with pytest.raises(CaptureContinuityError, match="epoch"):
        validator.observe(_market_record(kind="market_delta", sequence=2))


def test_historical_trades_and_prospective_l2_cannot_be_mixed() -> None:
    assert (
        validate_association_provenance(
            [
                _market_record(kind="trade"),
                _sports_record(),
            ]
        )
        is CaptureMode.PROSPECTIVE_L2
    )

    with pytest.raises(ValueError, match="cannot be mixed"):
        validate_association_provenance(
            [
                _market_record(kind="trade"),
                _market_record(
                    kind="trade",
                    capture_mode=CaptureMode.HISTORICAL_TRADES_ONLY,
                ),
            ]
        )


def test_capture_rotates_hourly_and_reconnect_seals_a_new_epoch(
    tmp_path: Path,
) -> None:
    spec = _freeze_spec(
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
    )
    capture = ProspectiveCaptureSession(
        raw_root=tmp_path,
        source="polymarket",
        stream="nfl-market",
        spec=spec,
    )

    capture.append(
        _market_record(
            kind="market_snapshot", sequence=1, minute=59, second=58
        )
    )
    capture.append(
        _market_record(
            kind="market_delta",
            sequence=2,
            minute=59,
            second=59,
            local_received_time=_at(0, minute=0, hour=18),
        )
    )
    assert capture.start_new_epoch(reason="reconnect") == 1
    capture.append(
        _market_record(
            kind="market_snapshot",
            epoch=1,
            sequence=9,
            minute=1,
            second=1,
        )
    )
    manifests = capture.close()

    assert len(manifests) == 3
    assert all(verify_segment(manifest.path).valid for manifest in manifests)
    assert [
        _segment_records(manifest.object_path)[0]["epoch"]
        for manifest in manifests
    ] == [
        0,
        0,
        1,
    ]
    assert all(
        manifest.capture_session_id == "nfl-2026-bal-buf"
        for manifest in manifests
    )


def test_invalid_continuity_frame_is_never_written_to_canonical_raw(
    tmp_path: Path,
) -> None:
    spec = _freeze_spec(
        session_id="nfl-2026-bal-buf",
        game_id="2026_01_BAL_BUF",
    )
    capture = ProspectiveCaptureSession(
        raw_root=tmp_path,
        source="polymarket",
        stream="nfl-market",
        spec=spec,
    )
    capture.append(_market_record(kind="market_snapshot", sequence=1))

    with pytest.raises(CaptureContinuityError, match="sequence gap"):
        capture.append(_market_record(kind="market_delta", sequence=3))

    manifests = capture.close()
    assert len(manifests) == 1
    records = _segment_records(manifests[0].object_path)
    assert [record["source_sequence"] for record in records] == [1]
