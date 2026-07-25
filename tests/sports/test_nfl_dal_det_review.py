from __future__ import annotations

import json
from pathlib import Path

from prediction_market.sports.nfl_dal_det_review import (
    build_dal_det_first_layer_review,
    write_dal_det_first_layer_review,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        if (candidate / "var" / "raw" / "raw").is_dir():
            return candidate
    raise AssertionError("test requires immutable DAL–DET source objects")


def _mapping() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / "artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json").read_text())


def test_first_layer_review_proves_only_its_reachable_gates(tmp_path: Path) -> None:
    report = build_dal_det_first_layer_review(program_root=_source_program_root(), market_mapping=_mapping())
    assert report["verified"]["final_score"] == {"DAL": 30, "DET": 44}
    assert report["gates"]["G_game_state_replay"] == "PASS"
    assert report["gates"]["P_play_personnel_join"] == "PASS"
    assert report["gates"]["A_source_time_chronological_merge"] == "PASS"
    assert report["gates"]["R_event_to_market_reaction"] == "BLOCKED"
    assert report["no_go_preserved"]["reaction_or_alpha_claim"] is False
    output = tmp_path / "review.json"
    write_dal_det_first_layer_review(report, output_path=output)
    assert json.loads(output.read_text()) == report


def test_checked_in_first_layer_review_is_current() -> None:
    generated = build_dal_det_first_layer_review(program_root=_source_program_root(), market_mapping=_mapping())
    checked_in = json.loads((PROJECT_ROOT / "artifacts/audit/nfl/nfl_2025_14_dal_det_first_layer_review_v0.json").read_text())
    assert checked_in == generated
