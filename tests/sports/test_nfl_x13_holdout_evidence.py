from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from prediction_market.sports.nfl_x13_holdout_evidence import (
    KALSHI_HISTORICAL_MARKETS_URL,
    NFLVERSE_SCHEDULES_URL,
    POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE,
    HoldoutEvidenceError,
    HoldoutGameExpectationV0,
    MetadataResponseV0,
    capture_x13_holdout_evidence,
    verify_holdout_evidence_index,
)


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
CONDITION_A = "0x" + "a" * 64
CONDITION_B = "0x" + "b" * 64


def _response(
    payload: bytes,
    *,
    url: str,
    params: dict[str, object],
    cursor: str | None,
    coverage: str,
    received_at: str = "2026-07-25T18:00:00.000000Z",
) -> MetadataResponseV0:
    return MetadataResponseV0(
        object_bytes=payload,
        source_url=url,
        request_params=params,
        source_cursor=cursor,
        received_at_utc=received_at,
        coverage=coverage,
        etag='"fixture-etag"',
        last_modified="Fri, 25 Jul 2026 18:00:00 GMT",
    )


def _fixtures() -> tuple[
    MetadataResponseV0,
    tuple[HoldoutGameExpectationV0, ...],
    dict[str, MetadataResponseV0],
    dict[str, MetadataResponseV0],
]:
    schedules = (
        b"game_id,gameday,gametime,away_team,away_score,home_team,home_score,result\n"
        b"2025_01_BAL_BUF,2025-09-07,20:20,BAL,40,BUF,41,BUF +1\n"
        b"2025_03_ARI_SF,2025-09-21,16:25,ARI,15,SF,16,SF +1\n"
    )
    expectations = (
        HoldoutGameExpectationV0(
            game_id="2025_01_BAL_BUF",
            away_team="BAL",
            home_team="BUF",
            polymarket_event_slug="nfl-bal-buf",
            kalshi_event_ticker="KXNFLGAME-25SEP07BALBUF",
        ),
        HoldoutGameExpectationV0(
            game_id="2025_03_ARI_SF",
            away_team="ARI",
            home_team="SF",
            polymarket_event_slug="nfl-ari-sf",
            kalshi_event_ticker="KXNFLGAME-25SEP21ARISF",
        ),
    )
    poly = {
        "nfl-bal-buf": _response(
            json.dumps(
                {
                    "id": "poly-event-bal-buf",
                    "slug": "nfl-bal-buf",
                    "volume": 999999,
                    "markets": [
                        {
                            "id": "poly-market-bal-buf",
                            "conditionId": CONDITION_A,
                            "outcomePrices": ["0.45", "0.55"],
                        }
                    ],
                },
                separators=(",", ":"),
            ).encode(),
            url=POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE.format(
                slug="nfl-bal-buf"
            ),
            params={},
            cursor="request_cursor:ROOT",
            coverage="exact Gamma event slug=nfl-bal-buf",
        ),
        "nfl-ari-sf": _response(
            json.dumps(
                {
                    "id": "poly-event-ari-sf",
                    "slug": "nfl-ari-sf",
                    "markets": [
                        {
                            "id": "poly-market-ari-sf",
                            "conditionId": CONDITION_B,
                        }
                    ],
                },
                separators=(",", ":"),
            ).encode(),
            url=POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE.format(
                slug="nfl-ari-sf"
            ),
            params={},
            cursor="request_cursor:ROOT",
            coverage="exact Gamma event slug=nfl-ari-sf",
        ),
    }
    kalshi = {
        "2025_01_BAL_BUF": _response(
            json.dumps(
                {
                    "cursor": "",
                    "markets": [
                        {
                            "event_ticker": "KXNFLGAME-25SEP07BALBUF",
                            "ticker": "KXNFLGAME-25SEP07BALBUF-BAL",
                            "volume": 12345,
                            "result": "yes",
                        },
                        {
                            "event_ticker": "KXNFLGAME-25SEP07BALBUF",
                            "ticker": "KXNFLGAME-25SEP07BALBUF-BUF",
                        },
                    ],
                },
                separators=(",", ":"),
            ).encode(),
            url=KALSHI_HISTORICAL_MARKETS_URL,
            params={"event_ticker": "KXNFLGAME-25SEP07BALBUF", "limit": 1000},
            cursor="request_cursor:ROOT",
            coverage="terminal historical markets event=KXNFLGAME-25SEP07BALBUF",
        ),
        "2025_03_ARI_SF": _response(
            json.dumps(
                {
                    "cursor": None,
                    "markets": [
                        {
                            "event_ticker": "KXNFLGAME-25SEP21ARISF",
                            "ticker": "KXNFLGAME-25SEP21ARISF-ARI",
                        },
                        {
                            "event_ticker": "KXNFLGAME-25SEP21ARISF",
                            "ticker": "KXNFLGAME-25SEP21ARISF-SF",
                        },
                    ],
                },
                separators=(",", ":"),
            ).encode(),
            url=KALSHI_HISTORICAL_MARKETS_URL,
            params={"event_ticker": "KXNFLGAME-25SEP21ARISF", "limit": 1000},
            cursor="request_cursor:ROOT",
            coverage="terminal historical markets event=KXNFLGAME-25SEP21ARISF",
        ),
    }
    return (
        _response(
            schedules,
            url=NFLVERSE_SCHEDULES_URL,
            params={},
            cursor="github_release_asset:schedules",
            coverage="2025 REG,POST schedule metadata",
        ),
        expectations,
        poly,
        kalshi,
    )


def _capture(tmp_path: Path):
    schedules, expectations, poly, kalshi = _fixtures()
    return capture_x13_holdout_evidence(
        store_root=tmp_path / "raw-store",
        program_root=PROGRAM_ROOT,
        schedules_response=schedules,
        game_expectations=expectations,
        polymarket_responses=poly,
        kalshi_responses=kalshi,
    )


def test_capture_preserves_exact_metadata_evidence_and_request_identity(
    tmp_path: Path,
) -> None:
    bundle = _capture(tmp_path)
    verified = verify_holdout_evidence_index(
        bundle.index_bytes,
        store_root=tmp_path / "raw-store",
        program_root=PROGRAM_ROOT,
    )

    assert verified == bundle.index
    assert bundle.index["index_sha256"] == bundle.index_sha256
    assert [game["game_id"] for game in bundle.index["games"]] == [
        "2025_01_BAL_BUF",
        "2025_03_ARI_SF",
    ]
    assert bundle.index["games"][0]["scheduled_start_utc"] == (
        "2025-09-08T00:20:00.000000Z"
    )
    assert bundle.index["games"][0]["polymarket"]["condition_refs"] == [
        {
            "condition_id": CONDITION_A,
            "market_id": "poly-market-bal-buf",
        }
    ]
    assert bundle.index["games"][0]["kalshi"]["market_refs"] == [
        "KXNFLGAME-25SEP07BALBUF-BAL",
        "KXNFLGAME-25SEP07BALBUF-BUF",
    ]

    forbidden = {"price", "prices", "volume", "score", "result", "reaction"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {
                *(str(key).casefold() for key in value),
                *(child for item in value.values() for child in keys(item)),
            }
        if isinstance(value, list):
            return {child for item in value for child in keys(item)}
        return set()

    assert not forbidden & keys(bundle.index)
    for record in bundle.records:
        manifest = record.manifest
        assert manifest.source_request["method"] == "GET"
        assert manifest.source_url.startswith("https://")
        assert manifest.source_cursor
        assert manifest.fetched_at == "2026-07-25T18:00:00.000000Z"
        assert manifest.coverage
        assert manifest.object_kind == "byte_exact_original"


def test_same_complete_capture_is_idempotent_and_changed_bytes_are_new(
    tmp_path: Path,
) -> None:
    first = _capture(tmp_path)
    second = _capture(tmp_path)
    assert first.index_bytes == second.index_bytes
    assert [record.object_path for record in first.records] == [
        record.object_path for record in second.records
    ]
    assert [record.manifest_path for record in first.records] == [
        record.manifest_path for record in second.records
    ]

    schedules, expectations, poly, kalshi = _fixtures()
    changed_payload = poly["nfl-bal-buf"].object_bytes.replace(
        b'"volume":999999',
        b'"updatedAt":"2026-07-25T18:01:00Z","volume":999999',
    )
    poly["nfl-bal-buf"] = _response(
        changed_payload,
        url=POLYMARKET_GAMMA_EVENT_BY_SLUG_URL_TEMPLATE.format(
            slug="nfl-bal-buf"
        ),
        params={},
        cursor="request_cursor:ROOT",
        coverage="exact Gamma event slug=nfl-bal-buf",
    )
    changed = capture_x13_holdout_evidence(
        store_root=tmp_path / "raw-store",
        program_root=PROGRAM_ROOT,
        schedules_response=schedules,
        game_expectations=expectations,
        polymarket_responses=poly,
        kalshi_responses=kalshi,
    )
    assert changed.index_sha256 != first.index_sha256
    assert (
        changed.index["games"][0]["polymarket"]["object_sha256"]
        != first.index["games"][0]["polymarket"]["object_sha256"]
    )


def test_missing_exact_dual_venue_or_terminal_proof_fails_closed(
    tmp_path: Path,
) -> None:
    schedules, expectations, poly, kalshi = _fixtures()
    del poly["nfl-ari-sf"]
    with pytest.raises(HoldoutEvidenceError, match="exact Polymarket"):
        capture_x13_holdout_evidence(
            store_root=tmp_path / "missing-poly",
            program_root=PROGRAM_ROOT,
            schedules_response=schedules,
            game_expectations=expectations,
            polymarket_responses=poly,
            kalshi_responses=kalshi,
        )

    schedules, expectations, poly, kalshi = _fixtures()
    kalshi["2025_01_BAL_BUF"] = _response(
        b'{"cursor":"NEXT","markets":[]}',
        url=KALSHI_HISTORICAL_MARKETS_URL,
        params={"event_ticker": "KXNFLGAME-25SEP07BALBUF", "limit": 1000},
        cursor="request_cursor:ROOT",
        coverage="not terminal",
    )
    with pytest.raises(HoldoutEvidenceError, match="terminal"):
        capture_x13_holdout_evidence(
            store_root=tmp_path / "nonterminal",
            program_root=PROGRAM_ROOT,
            schedules_response=schedules,
            game_expectations=expectations,
            polymarket_responses=poly,
            kalshi_responses=kalshi,
        )


def test_index_or_static_object_tamper_fails_closed(tmp_path: Path) -> None:
    bundle = _capture(tmp_path)
    tampered_index = bundle.index_bytes.replace(
        b"2025_01_BAL_BUF", b"2025_01_BAL_BUG", 1
    )
    with pytest.raises(HoldoutEvidenceError, match="index"):
        verify_holdout_evidence_index(
            tampered_index,
            store_root=tmp_path / "raw-store",
            program_root=PROGRAM_ROOT,
        )

    object_path = bundle.records[0].object_path
    os.chmod(object_path, 0o600)
    object_path.write_bytes(b"tampered")
    with pytest.raises(HoldoutEvidenceError, match="static evidence"):
        verify_holdout_evidence_index(
            bundle.index_bytes,
            store_root=tmp_path / "raw-store",
            program_root=PROGRAM_ROOT,
        )


def test_malformed_nested_index_fails_closed_with_domain_error(
    tmp_path: Path,
) -> None:
    bundle = _capture(tmp_path)
    malformed = json.loads(bundle.index_bytes)
    malformed["games"][0]["polymarket"] = []
    material = {
        key: value
        for key, value in malformed.items()
        if key != "index_sha256"
    }
    malformed["index_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                material,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    malformed_bytes = (
        json.dumps(
            malformed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )

    with pytest.raises(HoldoutEvidenceError, match="malformed"):
        verify_holdout_evidence_index(
            malformed_bytes,
            store_root=tmp_path / "raw-store",
            program_root=PROGRAM_ROOT,
        )
