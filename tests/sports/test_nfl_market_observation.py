from __future__ import annotations

import json
from pathlib import Path

from prediction_market.sports.nfl_market_observation import (
    build_nfl_market_mapping,
    write_nfl_market_mapping,
)


CONDITION_ID = "0xe6f492eb15583ae324d43e987e3ad1b7bd1a86690b3b62a2843162278ff7af0e"


def test_market_mapping_declares_venue_timestamps_without_reaction_claim() -> None:
    report = build_nfl_market_mapping(
        polymarket_event={
            "id": "89365",
            "slug": "nfl-dal-det-2025-12-04",
            "markets": [
                {
                    "id": "700742",
                    "conditionId": CONDITION_ID,
                    "question": "Cowboys vs. Lions",
                    "outcomes": '["Cowboys","Lions"]',
                    "clobTokenIds": '["cowboys-token","lions-token"]',
                }
            ],
        },
        polymarket_trade_count=3922,
        polymarket_manifest_refs={
            "event": "sha256:" + "a" * 64,
            "trades": "sha256:" + "b" * 64,
        },
        kalshi_markets=[
            {
                "ticker": "KXNFLGAME-25DEC04DALDET-DET",
                "title": "Dallas at Detroit Winner?",
                "yes_sub_title": "Detroit",
            },
            {
                "ticker": "KXNFLGAME-25DEC04DALDET-DAL",
                "title": "Dallas at Detroit Winner?",
                "yes_sub_title": "Dallas",
            },
        ],
        kalshi_trade_runs={
            "KXNFLGAME-25DEC04DALDET-DET": {"pages": 24, "trades": 23768, "complete": True},
            "KXNFLGAME-25DEC04DALDET-DAL": {"pages": 51, "trades": 50076, "complete": True},
        },
        kalshi_candle_runs={
            "KXNFLGAME-25DEC04DALDET-DET": {"interval_minutes": 1, "candles": 207},
            "KXNFLGAME-25DEC04DALDET-DAL": {"interval_minutes": 1, "candles": 207},
        },
    )

    assert report["canonical_game_id"] == "game_nflverse_2025_14_DAL_DET"
    assert report["venue_time_semantics"]["polymarket"] == "trade.timestamp_epoch_seconds"
    assert report["venue_time_semantics"]["kalshi"] == "trade.created_time_and_candle.end_period_ts"
    assert report["venues"]["polymarket"]["condition_id"] == CONDITION_ID
    assert report["venues"]["kalshi"]["outcome_contracts"][0]["team"] == "DET"
    assert report["evidence_boundary"]["reaction_measured"] is False
    assert report["evidence_boundary"]["cross_venue_comparison"] is False
    assert report["evidence_boundary"]["formal_result_eligible"] is False


def test_mapping_writer_emits_canonical_json(tmp_path: Path) -> None:
    report = {"artifact_type": "nfl_market_mapping", "artifact_version": "v0"}
    output = tmp_path / "mapping.json"

    write_nfl_market_mapping(report, output_path=output)

    assert json.loads(output.read_text()) == report
