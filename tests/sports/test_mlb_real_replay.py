from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from prediction_market.contracts import canonical_sha256
from prediction_market.sports import mlb_game_state as mlb
from prediction_market.sports.game_state import canonical_state_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "var/raw"
    / mlb.RETROSHEET_2025_MANIFEST_RELATIVE_PATH
)


def _require_frozen_inputs() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip("frozen Retrosheet raw object is not present")
    if shutil.which("cwevent") is None:
        pytest.skip("Chadwick cwevent is not installed")


def test_frozen_retrosheet_game_replay_is_deterministic_and_auditable() -> None:
    _require_frozen_inputs()

    first = mlb.run_frozen_retrosheet_2025_game_replay(
        program_root=PROJECT_ROOT
    )
    second = mlb.run_frozen_retrosheet_2025_game_replay(
        program_root=PROJECT_ROOT
    )

    assert first == second
    assert first.raw_object_sha256 == mlb.RETROSHEET_2025_RAW_OBJECT_SHA256
    assert first.source_manifest_sha256 == mlb.RETROSHEET_2025_MANIFEST_SHA256
    assert first.native_game_id == "ANA202504040"
    assert first.canonical_game_id == "game_retrosheet_ANA202504040"
    assert first.events == 88
    assert first.adapter_vs_next_observation_comparisons == 87
    assert first.field_mismatches == 0
    assert len(first.source_row_sha256s) == 88
    assert len(set(first.event_envelope_ids)) == 88
    assert len(first.play_events) == 88
    assert len(first.transition_trace_sha256s) == 88
    assert len(first.state_step_sha256s) == 88
    assert first.final_state.terminal is True
    assert first.final_state.score == mlb.MLBScore(away=8, home=6)
    assert first.final_state.observation_mode == "offline_reconstruction"
    assert first.final_state.source_provenance is not None
    assert first.final_state.source_provenance.source_row_ordinal == 88
    assert (
        first.final_state.source_provenance.reconstruction_cutoff_ordinal
        == 88
    )
    assert first.final_state_sha256 == canonical_state_sha256(first.final_state)
    assert first.replay_sha256 == second.replay_sha256


def test_frozen_replay_records_actual_cwevent_binary_command_and_field_map() -> None:
    _require_frozen_inputs()

    result = mlb.run_frozen_retrosheet_2025_game_replay(
        program_root=PROJECT_ROOT
    )
    binary = Path(result.cwevent_runtime.executable)

    assert binary.is_absolute()
    assert binary.is_file()
    assert result.cwevent_runtime.version == "0.10.0"
    assert result.cwevent_runtime.binary_sha256 == (
        "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    assert result.cwevent_command == mlb.cwevent_command(
        result.cwevent_runtime,
        native_game_id=mlb.RETROSHEET_FROZEN_GAME_ID,
        event_file_name=mlb.RETROSHEET_FROZEN_EVENT_FILE,
    )
    assert result.cwevent_command_sha256 == canonical_sha256(
        list(result.cwevent_command)
    )
    assert result.cwevent_field_map_sha256 == mlb.CWEVENT_FIELD_MAP_SHA256
    assert result.cwevent_command[result.cwevent_command.index("-f") + 1] == (
        mlb.CWEVENT_FIELD_ARGUMENT
    )


def test_frozen_replay_fails_closed_without_governed_raw_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        mlb.MLBGameStateError,
        match="object or manifest verification",
    ):
        mlb.run_frozen_retrosheet_2025_game_replay(program_root=tmp_path)
