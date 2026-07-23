from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.experiments import load_experiment_registry
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


@lru_cache(maxsize=1)
def _assert_x11_registered() -> None:
    assert "X-11" in load_experiment_registry(PROJECT_ROOT)


@lru_cache(maxsize=1)
def _cached_registry() -> dict[str, dict[str, Any]]:
    return load_experiment_registry(PROJECT_ROOT)


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


def _event_for_rows(
    pre_row: dict[str, Any],
    post_row: dict[str, Any],
    *,
    sequence: int,
) -> nfl.NFLPlayEvent:
    payload = nfl.nflverse_transition_payload(
        pre_row,
        post_row,
        sequence=sequence,
    )
    _assert_x11_registered()
    canonical_refs = {
        "competition_id": "cmp_nfl",
        "game_id": f"game_nflverse_{GAME_ID}",
        "participant_ids": ("participant_ARI", "participant_NO"),
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }
    event_time = {
        "receive_at": "2026-07-23T03:30:50.704643Z",
        "receive_basis": "upstream_exporter",
        "source_at": None,
        "publish_at": None,
        "exchange_at": None,
    }
    raw_parents = tuple(
        EventEnvelopeV0.create(
            envelope_version="v0",
            event_type="raw_observation",
            payload_schema_version="v0",
            source={
                "system": "nflverse",
                "stream": "play_by_play",
                "venue": None,
                "sequence": ordinal,
                "capture_session_id": f"static:{RAW_OBJECT_SHA256}",
                "record_ordinal": ordinal,
            },
            time=event_time,
            canonical_refs=canonical_refs,
            native_refs=(
                {
                    "namespace": "nflverse.play",
                    "native_id": f"{GAME_ID}:{int(row['play_id'])}",
                },
            ),
            lineage={
                "raw_object_hash": RAW_OBJECT_SHA256,
                "raw_record_ordinal": ordinal,
                "parent_event_ids": (),
            },
            experiment_id=None,
            rule_snapshot_ref=None,
            quality_flags=(),
            payload={
                "dataset_id": "DS-NFLVERSE",
                "partition": "season-2025",
                "raw_object_hash": RAW_OBJECT_SHA256,
                "raw_record_ordinal": ordinal,
            },
        )
        for ordinal, row in (
            (sequence - 1, pre_row),
            (sequence, post_row),
        )
    )
    normalized = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": "nflverse",
            "stream": "play_by_play.normalized",
            "venue": None,
            "sequence": sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=tuple(
            parent.native_refs[0] for parent in raw_parents
        ),
        lineage={
            "raw_object_hash": None,
            "raw_record_ordinal": None,
            "parent_event_ids": tuple(
                parent.event_id for parent in raw_parents
            ),
        },
        experiment_id="X-11",
        rule_snapshot_ref=None,
        quality_flags=(),
        payload=payload,
    )
    with patch(
        "prediction_market.experiments.load_experiment_registry",
        return_value=_cached_registry(),
    ):
        return nfl.event_from_nflverse_envelope(
            normalized,
            program_root=PROJECT_ROOT,
            raw_parents=raw_parents,
        )


def _replay_once() -> tuple[str, int]:
    rows = _real_game_rows()
    state = nfl.state_from_nflverse_row(rows[0], sequence=0)

    for sequence, (pre_row, post_row) in enumerate(
        zip(rows, rows[1:]),
        start=1,
    ):
        event = _event_for_rows(
            pre_row,
            post_row,
            sequence=sequence,
        )
        state = nfl.reduce(state, event)
        expected = nfl.state_from_nflverse_row(
            post_row,
            sequence=sequence,
            terminal=event.terminal,
            last_event_id=event.event_id,
        )
        assert asdict(state) == asdict(expected)

    return canonical_state_sha256(state), len(rows) - 1


def test_real_snapshot_flags_are_committed_only_when_transition_verifies_them() -> None:
    rows = _real_game_rows()

    first_down_before_quarter_end = _event_for_rows(
        rows[38],
        rows[39],
        sequence=39,
    )
    assert rows[38]["first_down"] == 1
    assert rows[39]["desc"] == "END QUARTER 1"
    assert first_down_before_quarter_end.first_down is False


def test_future_drive_result_cannot_change_a_normalized_transition() -> None:
    rows = _real_game_rows()
    pre_row = rows[20]
    post_row = rows[21]
    baseline = _event_for_rows(
        pre_row,
        post_row,
        sequence=21,
    )
    mutated_pre = {**pre_row, "series_result": "Future impossible outcome"}
    mutated_post = {
        **post_row,
        "series_result": "Another future outcome",
        "play_type": "future play must not leak",
    }

    assert "series_result" in nfl.NFLVERSE_LEAKAGE_FIELDS
    assert _event_for_rows(
        mutated_pre,
        mutated_post,
        sequence=21,
    ) == baseline


def test_real_complete_game_replays_twice_to_identical_hash() -> None:
    first_hash, first_steps = _replay_once()
    second_hash, second_steps = _replay_once()

    assert first_steps == second_steps == 181
    assert first_hash == second_hash


def test_adapter_requires_a_fully_bound_event_envelope() -> None:
    rows = _real_game_rows()
    event = _event_for_rows(rows[0], rows[1], sequence=1)
    assert event.source_play_id == str(int(rows[0]["play_id"]))

    with pytest.raises(TypeError, match="EventEnvelopeV0"):
        nfl.event_from_nflverse_envelope(  # type: ignore[arg-type]
            rows[0],
            program_root=PROJECT_ROOT,
            raw_parents=(),
        )
