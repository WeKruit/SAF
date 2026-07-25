from __future__ import annotations

import json
from pathlib import Path

from prediction_market.sports.nfl_dal_det_gamebook_crosscheck import (
    build_dal_det_gamebook_crosscheck,
    write_dal_det_gamebook_crosscheck,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        if (candidate / "var" / "raw" / "raw").is_dir():
            return candidate
    raise AssertionError("test requires frozen DAL–DET PBP")


def test_official_scorecard_path_crosschecks_the_frozen_replay(tmp_path: Path) -> None:
    report = build_dal_det_gamebook_crosscheck(program_root=_source_program_root())
    comparison = report["comparison"]
    assert comparison["final_score_match"] is True
    assert comparison["all_official_score_milestones_match"] is True
    assert comparison["event_by_event_gamebook_match"] is False
    assert report["official_reference"]["raw_ingested"] is False
    assert report["official_reference"]["license_status"] == "blocked"
    output = tmp_path / "crosscheck.json"
    write_dal_det_gamebook_crosscheck(report, output_path=output)
    assert json.loads(output.read_text()) == report


def test_checked_in_scorecard_crosscheck_is_current() -> None:
    generated = build_dal_det_gamebook_crosscheck(program_root=_source_program_root())
    checked_in = json.loads((PROJECT_ROOT / "artifacts/audit/nfl/nfl_2025_14_dal_det_official_scorecard_crosscheck_v0.json").read_text())
    assert checked_in == generated
