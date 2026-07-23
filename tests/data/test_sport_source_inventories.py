from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from prediction_market.sports.source_inventories import (
    SportSourceInventoryError,
    bevent_state_mapping,
    fetch_and_preserve_jolpica,
    fetch_and_preserve_retrosheet_2025,
    inspect_jolpica_response,
    inspect_retrosheet_2025,
)
from prediction_market.static_store import read_verified_static_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCHED_AT = datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc)


def _retrosheet_zip(*, unsafe_name: str | None = None) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            unsafe_name or "2025AAA.EVA",
            (
                "id,AAA202504010\r\n"
                "version,2\r\n"
                "info,visteam,BBB\r\n"
                "info,hometeam,AAA\r\n"
                "start,batte001,\"Example Batter\",0,1,8\r\n"
                "play,1,0,batte001,00,,S7\r\n"
                "play,1,1,batte002,00,,63\r\n"
            ).encode("ascii"),
        )
        archive.writestr("TEAM2025", b"AAA,A,Example Team\r\n")
    return target.getvalue()


def _response(payload: bytes, request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        content=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(payload)),
            "ETag": '"fixture"',
        },
        request=request,
    )


def test_retrosheet_inventory_and_bevent_mapping_are_explicit() -> None:
    audit = inspect_retrosheet_2025(_retrosheet_zip())

    assert audit.season == 2025
    assert audit.event_file_count == 1
    assert audit.game_count == 1
    assert audit.play_record_count == 2
    assert audit.record_type_counts["play"] == 2
    assert audit.first_game_id == "AAA202504010"
    assert audit.last_game_id == "AAA202504010"
    assert audit.object_sha256.startswith("sha256:")
    assert audit.schema_fingerprint.startswith("sha256:")

    mapping = bevent_state_mapping()
    assert mapping["game_id"]["bevent_field"] == "GAME_ID"
    assert mapping["outs_before"]["bevent_field"] == "OUTS_CT"
    assert mapping["base_state_before"]["bevent_field"] == "START_BASES_CD"
    assert mapping["score_before"]["requires"] == [
        "AWAY_SCORE_CT",
        "HOME_SCORE_CT",
    ]
    assert mapping["mapping_status"] == {
        "requires_chadwick_bevent": True,
        "executed_in_inventory": False,
    }


@pytest.mark.parametrize(
    "payload",
    [
        b"not-a-zip",
        _retrosheet_zip(unsafe_name="../escape.EVA"),
    ],
)
def test_retrosheet_inventory_fails_closed_on_invalid_archive(payload: bytes) -> None:
    with pytest.raises(SportSourceInventoryError):
        inspect_retrosheet_2025(payload)


def test_retrosheet_fetch_preserves_exact_release_bytes(tmp_path: Path) -> None:
    payload = _retrosheet_zip()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://www.retrosheet.org/events/2025eve.zip"
        assert request.headers["accept-encoding"] == "identity"
        return _response(payload, request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        preserved = fetch_and_preserve_retrosheet_2025(
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert preserved.record.manifest.dataset_id == "DS-RETROSHEET"
    assert preserved.record.manifest.license_ref == "O-007"
    assert preserved.record.manifest.license_status == "research_only"
    verified = read_verified_static_object(
        preserved.record.manifest_path,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
    )
    assert verified.object_bytes == payload


def _jolpica_payload(kind: str) -> bytes:
    race: dict[str, object] = {
        "season": "2025",
        "round": "1",
        "raceName": "Australian Grand Prix",
        "date": "2025-03-16",
    }
    if kind == "results":
        race["Results"] = [
            {
                "position": "1",
                "Driver": {"driverId": "norris"},
                "Constructor": {"constructorId": "mclaren"},
                "status": "Finished",
            }
        ]
    elif kind == "laps":
        race["Laps"] = [
            {
                "number": "1",
                "Timings": [
                    {"driverId": "norris", "position": "1", "time": "1:42.001"}
                ],
            }
        ]
    elif kind == "pitstops":
        race["PitStops"] = [
            {
                "driverId": "norris",
                "lap": "14",
                "stop": "1",
                "time": "15:01:02",
                "duration": "2.34",
            }
        ]
    else:
        raise AssertionError(kind)
    return json.dumps(
        {
            "MRData": {
                "xmlns": "",
                "series": "f1",
                "url": "https://api.jolpi.ca/ergast/f1/2025.json",
                "limit": "100",
                "offset": "0",
                "total": "1",
                "RaceTable": {"season": "2025", "Races": [race]},
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@pytest.mark.parametrize("kind", ["results", "laps", "pitstops"])
def test_jolpica_inventory_reports_native_granularity(kind: str) -> None:
    audit = inspect_jolpica_response(_jolpica_payload(kind), endpoint_kind=kind)

    assert audit.endpoint_kind == kind
    assert audit.race_count == 1
    assert audit.native_record_count == 1
    assert audit.seasons == (2025,)
    assert audit.rounds == ((2025, 1),)
    assert audit.schema_fingerprint.startswith("sha256:")


def test_jolpica_inventory_rejects_wrong_native_shape() -> None:
    value = json.loads(_jolpica_payload("results"))
    value["MRData"]["RaceTable"]["Races"][0].pop("Results")
    with pytest.raises(SportSourceInventoryError, match="Results"):
        inspect_jolpica_response(
            json.dumps(value).encode(), endpoint_kind="results"
        )


def test_jolpica_fetch_preserves_exact_public_response(tmp_path: Path) -> None:
    payload = _jolpica_payload("results")
    url = "https://api.jolpi.ca/ergast/f1/2025/results/1.json?limit=100"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == url
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "application/json"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        preserved = fetch_and_preserve_jolpica(
            url,
            endpoint_kind="results",
            partition="season-2025-results-position-1",
            coverage="2025 race winners",
            store_root=tmp_path,
            program_root=PROJECT_ROOT,
            fetched_at=FETCHED_AT,
            client=client,
        )

    assert preserved.record.manifest.dataset_id == "DS-F1-JOLPICA"
    assert preserved.record.manifest.license_ref == "O-008"
    assert preserved.record.manifest.license_status == "research_only"
    assert preserved.audit.native_record_count == 1


def test_real_inventory_cards_do_not_claim_models_or_live_readiness() -> None:
    mlb = json.loads(
        (PROJECT_ROOT / "artifacts/game-state/mlb/source_inventory_v0.json").read_text()
    )
    f1 = json.loads(
        (PROJECT_ROOT / "artifacts/game-state/f1/source_inventory_v0.json").read_text()
    )

    assert mlb["model_trained"] is False
    assert mlb["bevent_state_mapping"]["executed_in_inventory"] is False
    assert mlb["dataset"]["object_sha256"].startswith("sha256:")
    assert f1["model_trained"] is False
    assert f1["fastf1"]["license_status"] == "blocked"
    assert f1["fastf1"]["access_executed"] is False
    assert "live SLA" in " ".join(f1["limitations"])
