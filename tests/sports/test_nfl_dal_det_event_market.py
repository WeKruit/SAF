from __future__ import annotations

import json
from pathlib import Path

from prediction_market.sports.nfl_dal_det_event_market import (
    build_dal_det_event_market_source_time_timeline,
    write_dal_det_event_market_source_time_timeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        if (candidate / "var" / "raw" / "raw").is_dir():
            return candidate
    raise AssertionError("test requires immutable DAL–DET inputs")


def test_event_market_timeline_is_full_and_chronological() -> None:
    mapping = json.loads((PROJECT_ROOT / "artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json").read_text())
    report = build_dal_det_event_market_source_time_timeline(program_root=_source_program_root(), mapping=mapping)
    assert report["counts"]["event_personnel_rows"] == 181
    assert report["counts"]["by_lane"]["polymarket_trade"] > 0
    assert report["counts"]["by_lane"]["kalshi_trade"] > 0
    assert report["counts"]["by_lane"]["kalshi_candle"] == 414
    assert report["merge_validation"]["status"] == "PASS_SOURCE_TIME_CHRONOLOGY_ONLY"
    assert report["evidence_boundary"]["event_to_market_causal_join"] is False
    assert report["evidence_boundary"]["reaction_latency_measured"] is False
    times = [row["source_time_ns"] for row in report["rows"]]
    assert times == sorted(times)
    event_orders = [row["source_sequence"] for row in report["rows"] if row["lane"] == "nfl_event_personnel"]
    assert event_orders == sorted(event_orders)


def test_checked_in_event_market_timeline_is_current(tmp_path: Path) -> None:
    mapping = json.loads((PROJECT_ROOT / "artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json").read_text())
    generated = build_dal_det_event_market_source_time_timeline(program_root=_source_program_root(), mapping=mapping)
    output = tmp_path / "timeline.json"
    write_dal_det_event_market_source_time_timeline(generated, output_path=output)
    assert json.loads(output.read_text()) == generated
    checked_in = json.loads((PROJECT_ROOT / "artifacts/audit/nfl/nfl_2025_14_dal_det_event_market_source_time_timeline_v1.json").read_text())
    assert checked_in == generated
