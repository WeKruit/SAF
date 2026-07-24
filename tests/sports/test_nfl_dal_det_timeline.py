from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_market.sports.nfl_dal_det_timeline import (
    DALDETTimelineError,
    _ordered,
    build_dal_det_source_timeline,
    write_dal_det_source_timeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        if (candidate / "var" / "raw" / "raw").is_dir():
            return candidate
    raise AssertionError("test requires preserved DAL–DET raw evidence")


def test_ordering_is_deterministic_without_implying_causality() -> None:
    rows = _ordered(
        [
            {"source_time_ns": 1, "lane": "kalshi_trade", "source_sequence": "a"},
            {"source_time_ns": 1, "lane": "game_state", "source_sequence": 2},
            {"source_time_ns": 1, "lane": "game_state", "source_sequence": 1},
        ]
    )
    assert [(row["lane"], row["source_sequence"]) for row in rows] == [
        ("game_state", 1),
        ("game_state", 2),
        ("kalshi_trade", "a"),
    ]


def test_dal_det_source_timeline_contains_all_lanes_and_scenarios(tmp_path: Path) -> None:
    mapping = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json"
        ).read_text()
    )
    report = build_dal_det_source_timeline(
        program_root=_source_program_root(),
        mapping=mapping,
        latency_scenarios_seconds=(0, 2, 10),
    )
    assert report["time_basis"] == "native_source_time_only"
    assert report["latency_scenario_seconds"] == [0, 2, 10]
    assert report["lane_counts"]["game_state"] == 187
    assert report["lane_counts"]["polymarket_trade"] > 0
    assert report["lane_counts"]["kalshi_trade"] > 0
    assert report["lane_counts"]["kalshi_candle"] == 414
    assert report["row_count"] == sum(report["lane_counts"].values())
    assert report["research_boundary"]["timeline_visualization_allowed"] is True
    assert report["research_boundary"]["measured_latency_allowed"] is False

    output = tmp_path / "timeline.json"
    write_dal_det_source_timeline(report, output_path=output)
    assert json.loads(output.read_text()) == report


def test_timeline_rejects_pretended_quantiles() -> None:
    mapping = {"canonical_game_id": "game_nflverse_2025_14_DAL_DET"}
    with pytest.raises(DALDETTimelineError, match="non-negative"):
        build_dal_det_source_timeline(
            program_root=_source_program_root(),
            mapping=mapping,
            latency_scenarios_seconds=(-1,),
        )


def test_checked_in_source_timeline_is_current() -> None:
    mapping = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json"
        ).read_text()
    )
    generated = build_dal_det_source_timeline(
        program_root=_source_program_root(), mapping=mapping
    )
    checked_in = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/audit/nfl/nfl_2025_14_dal_det_source_timeline_v0.json"
        ).read_text()
    )
    assert checked_in == generated
