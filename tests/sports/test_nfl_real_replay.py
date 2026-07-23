from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_envelopes,
)
from prediction_market.sports.game_state import canonical_state_sha256
from prediction_market.sports import nfl_game_state as nfl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OBJECT_SHA256 = (
    "sha256:3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a"
)
NFLVERSE_2025 = (
    PROJECT_ROOT
    / "var"
    / "raw"
    / "raw"
    / "source=nflverse"
    / "dataset=DS-NFLVERSE"
    / "version=github-release-58152862-20260212T102526Z"
    / "partition=season-2025"
    / f"{RAW_OBJECT_SHA256.removeprefix('sha256:')}.parquet"
)
GAME_ID = "2025_01_ARI_NO"

NFLVERSE_REPLAY_COLUMNS = (
    "game_id",
    "play_id",
    "qtr",
    "quarter_seconds_remaining",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "fixed_drive",
    "goal_to_go",
    "play_clock",
    "posteam",
    "down",
    "ydstogo",
    "yardline_100",
    "total_home_score",
    "total_away_score",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "play_type",
    "desc",
    "first_down",
    "interception",
    "fumble_lost",
    "timeout",
    "timeout_team",
    "series_result",
)


def _event_id(game_id: str, ordinal: int, source_play_id: object) -> str:
    material = (
        f"{RAW_OBJECT_SHA256}:{game_id}:{ordinal}:{source_play_id}"
    ).encode()
    return "evt_" + hashlib.sha256(material).hexdigest()


def _real_game_rows() -> list[dict[str, Any]]:
    if not NFLVERSE_2025.is_file():
        pytest.skip("frozen nflverse 2025 raw object is not present")
    parquet = pytest.importorskip("pyarrow.parquet")
    table = parquet.read_table(
        NFLVERSE_2025,
        columns=list(NFLVERSE_REPLAY_COLUMNS),
        filters=[("game_id", "=", GAME_ID)],
    )
    rows = table.to_pylist()
    if len(rows) < 2:
        pytest.fail(f"frozen raw object did not contain complete game {GAME_ID}")
    return rows


def _replay_once() -> tuple[str, int]:
    rows = _real_game_rows()
    state = nfl.state_from_nflverse_row(rows[0], sequence=0)

    for sequence, (pre_row, post_row) in enumerate(
        zip(rows, rows[1:]),
        start=1,
    ):
        event_id = _event_id(GAME_ID, sequence, pre_row["play_id"])
        event = nfl.event_from_nflverse_rows(
            pre_row,
            post_row,
            event_id=event_id,
            sequence=sequence,
        )
        state = nfl.reduce(state, event)
        expected = nfl.state_from_nflverse_row(
            post_row,
            sequence=sequence,
            terminal=event.terminal,
            last_event_id=event_id,
        )
        assert asdict(state) == asdict(expected)

    return canonical_state_sha256(state), len(rows) - 1


def test_real_snapshot_flags_are_committed_only_when_transition_verifies_them() -> None:
    rows = _real_game_rows()

    first_down_before_quarter_end = nfl.event_from_nflverse_rows(
        rows[38],
        rows[39],
        event_id=_event_id(GAME_ID, 39, rows[38]["play_id"]),
        sequence=39,
    )
    assert rows[38]["first_down"] == 1
    assert rows[39]["desc"] == "END QUARTER 1"
    assert first_down_before_quarter_end.first_down is False


def test_future_drive_result_cannot_change_a_normalized_transition() -> None:
    rows = _real_game_rows()
    pre_row = rows[20]
    post_row = rows[21]
    event_id = _event_id(GAME_ID, 21, pre_row["play_id"])

    baseline = nfl.event_from_nflverse_rows(
        pre_row,
        post_row,
        event_id=event_id,
        sequence=21,
    )
    mutated_pre = {**pre_row, "series_result": "Future impossible outcome"}
    mutated_post = {**post_row, "series_result": "Another future outcome"}

    assert "series_result" in nfl.NFLVERSE_LEAKAGE_FIELDS
    assert nfl.event_from_nflverse_rows(
        mutated_pre,
        mutated_post,
        event_id=event_id,
        sequence=21,
    ) == baseline


def test_real_complete_game_replays_twice_to_identical_hash() -> None:
    first_hash, first_steps = _replay_once()
    second_hash, second_steps = _replay_once()

    assert first_steps == second_steps == 181
    assert first_hash == second_hash


def test_adapter_requires_an_external_event_envelope_id() -> None:
    rows = _real_game_rows()
    pair = build_static_sport_observation_envelopes(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash=RAW_OBJECT_SHA256,
        raw_record_ordinal=1,
        partition="season-2025",
        fetched_at="2026-07-23T03:30:50.704643Z",
        source_at=None,
        competition_id="cmp_nfl",
        game_id=f"game_nflverse_{GAME_ID}",
        participant_ids=("participant_ARI", "participant_NO"),
        native_namespace="nflverse.play",
        native_id=f"{GAME_ID}:{int(rows[0]['play_id'])}",
        normalized_payload={
            "sport": "nfl",
            "source_play_id": str(int(rows[0]["play_id"])),
        },
    )

    event = nfl.event_from_nflverse_rows(
        rows[0],
        rows[1],
        event_id=pair.normalized.event_id,
    )
    assert event.event_id == pair.normalized.event_id

    with pytest.raises(TypeError, match="event_id"):
        nfl.event_from_nflverse_rows(rows[0], rows[1])
