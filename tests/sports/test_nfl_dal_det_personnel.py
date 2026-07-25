from __future__ import annotations

import json
from pathlib import Path

from prediction_market.sports.nfl_dal_det_personnel import (
    build_dal_det_event_personnel_timeline,
    write_dal_det_event_personnel_timeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        if (candidate / "var" / "raw" / "raw").is_dir():
            return candidate
    raise AssertionError("test requires preserved PBP and participation objects")


def test_dal_det_personnel_join_is_complete_and_deterministic() -> None:
    first = build_dal_det_event_personnel_timeline(program_root=_source_program_root())
    second = build_dal_det_event_personnel_timeline(program_root=_source_program_root())
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["join"] == {
        "join_key": "nflverse game_id + play_id",
        "pbp_game_rows": 192,
        "participation_rows": 181,
        "joined_rows": 181,
        "full_11v11_rows": 181,
        "administrative_or_nonparticipation_pbp_rows": 11,
        "missing_pbp_for_participation": 0,
    }
    assert all(row["source_time_utc"] is not None for row in first["rows"])
    assert all(len(row["participation"]["offense_names"]) == 11 for row in first["rows"])
    assert all(len(row["participation"]["defense_names"]) == 11 for row in first["rows"])
    assert first["evidence_boundary"]["exact_substitution_timestamp_available"] is False


def test_checked_in_personnel_timeline_is_current(tmp_path: Path) -> None:
    generated = build_dal_det_event_personnel_timeline(program_root=_source_program_root())
    output = tmp_path / "timeline.json"
    write_dal_det_event_personnel_timeline(generated, output_path=output)
    assert json.loads(output.read_text()) == generated
    checked_in = json.loads((PROJECT_ROOT / "artifacts/audit/nfl/nfl_2025_14_dal_det_event_personnel_timeline_v1.json").read_text())
    assert checked_in == generated
