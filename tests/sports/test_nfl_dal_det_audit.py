from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_market.sports.nfl_dal_det_audit import (
    DALDETAuditError,
    _kalshi_candle_summary,
    _kalshi_trade_summary,
    _polymarket_summary,
    build_dal_det_evidence_audit,
    write_dal_det_evidence_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        if (candidate / "var" / "raw" / "raw").is_dir():
            return candidate
    raise AssertionError("test requires the preserved one-game raw evidence")


def test_trade_and_candle_summaries_reject_duplicate_or_noncontiguous_data() -> None:
    with pytest.raises(DALDETAuditError, match="duplicated"):
        _polymarket_summary(
            [
                {"conditionId": "c", "timestamp": 10, "asset": "a"},
                {"conditionId": "c", "timestamp": 10, "asset": "a"},
            ],
            expected_condition_id="c",
        )
    with pytest.raises(DALDETAuditError, match="duplicate ID"):
        _kalshi_trade_summary(
            payloads=[
                {
                    "trades": [
                        {
                            "ticker": "T",
                            "trade_id": "same",
                            "created_time": "2025-12-05T01:00:00Z",
                        }
                    ]
                },
                {
                    "trades": [
                        {
                            "ticker": "T",
                            "trade_id": "same",
                            "created_time": "2025-12-05T01:01:00Z",
                        }
                    ]
                },
            ],
            ticker="T",
        )
    with pytest.raises(DALDETAuditError, match="contiguous"):
        _kalshi_candle_summary(
            {"candlesticks": [{"end_period_ts": 60}, {"end_period_ts": 180}]},
            ticker="T",
        )


def test_dal_det_audit_verifies_all_bound_objects_and_blocks_reaction_claims(
    tmp_path: Path,
) -> None:
    source_root = _source_program_root()
    mapping_path = (
        PROJECT_ROOT
        / "artifacts"
        / "market-observation"
        / "nfl"
        / "nfl_2025_14_dal_det_market_mapping_v0.json"
    )
    report = build_dal_det_evidence_audit(
        program_root=source_root, mapping=json.loads(mapping_path.read_text())
    )

    assert report["game_state"]["source_row_count"] == 192
    assert report["game_state"]["source_timestamp"]["present_transition_count"] == 187
    assert report["immutable_input_verification"]["verified_manifest_count"] == 81
    assert report["market_data"]["polymarket"]["trade_inventory"]["record_count"] == 3922
    assert (
        report["market_data"]["kalshi"]["contracts"]
        ["KXNFLGAME-25DEC04DALDET-DET"]["trade_inventory"]["record_count"]
        == 23768
    )
    assert report["time_join_gate"]["status"] == "BLOCKED"
    assert report["official_gamebook"]["status"] == "BLOCKED_NOT_INGESTED"
    assert report["formal_result_eligible"] is False

    output = tmp_path / "audit.json"
    write_dal_det_evidence_audit(report, output_path=output)
    assert json.loads(output.read_text()) == report


def test_checked_in_dal_det_audit_is_current() -> None:
    source_root = _source_program_root()
    mapping_path = (
        PROJECT_ROOT
        / "artifacts"
        / "market-observation"
        / "nfl"
        / "nfl_2025_14_dal_det_market_mapping_v0.json"
    )
    generated = build_dal_det_evidence_audit(
        program_root=source_root, mapping=json.loads(mapping_path.read_text())
    )
    checked_in = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "audit"
            / "nfl"
            / "nfl_2025_14_dal_det_evidence_audit_v0.json"
        ).read_text()
    )
    assert checked_in == generated
