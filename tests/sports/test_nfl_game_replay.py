from __future__ import annotations

import json
from pathlib import Path

from prediction_market.sports.nfl_game_replay import (
    replay_frozen_nfl_game,
    write_frozen_nfl_game_replay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NATIVE_GAME_ID = "2025_14_DAL_DET"


def _source_program_root() -> Path:
    for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent.parent):
        raw_object_root = candidate / "var" / "raw" / "raw"
        if raw_object_root.is_dir() and not raw_object_root.is_symlink():
            return candidate
    raise AssertionError("a test run requires one real immutable raw root")


SOURCE_PROGRAM_ROOT = _source_program_root()


def test_dal_det_replay_is_ordered_deterministic_and_offline_only() -> None:
    first = replay_frozen_nfl_game(
        program_root=SOURCE_PROGRAM_ROOT,
        native_game_id=NATIVE_GAME_ID,
    )
    second = replay_frozen_nfl_game(
        program_root=SOURCE_PROGRAM_ROOT,
        native_game_id=NATIVE_GAME_ID,
    )

    assert first.trace_sha256 == second.trace_sha256
    assert first.transition_count > 100
    assert first.native_order_strictly_increasing is True
    assert first.observation_mode == "offline_reconstruction"
    assert first.source_time_present_count == 187
    assert first.source_time_missing_count == 5
    assert first.source_row_count == 192
    assert first.evidence_boundary["market_data_included"] is False
    assert first.evidence_boundary["market_alignment_performed"] is False
    assert first.evidence_boundary["pit_validated"] is False
    assert first.evidence_boundary["model_output_included"] is False
    assert first.evidence_boundary["alpha_or_execution_claim"] is False


def test_written_artifacts_match_the_replayed_hash_chain(tmp_path: Path) -> None:
    report = write_frozen_nfl_game_replay(
        program_root=SOURCE_PROGRAM_ROOT,
        native_game_id=NATIVE_GAME_ID,
        output_directory=tmp_path,
    )

    summary = json.loads((tmp_path / report.summary_filename).read_text())
    trace_lines = (tmp_path / report.trace_filename).read_text().splitlines()

    assert summary["trace_sha256"] == report.trace_sha256
    assert len(trace_lines) == report.transition_count
    assert summary["evidence_boundary"]["market_data_included"] is False
    assert summary["evidence_boundary"]["pit_validated"] is False


def test_checked_in_dal_det_evidence_is_current() -> None:
    generated = replay_frozen_nfl_game(
        program_root=SOURCE_PROGRAM_ROOT,
        native_game_id=NATIVE_GAME_ID,
    )
    summary_path = (
        PROJECT_ROOT
        / "artifacts"
        / "game-state"
        / "nfl"
        / generated.summary_filename
    )
    summary = json.loads(summary_path.read_text())
    assert summary["trace_sha256"] == generated.trace_sha256
