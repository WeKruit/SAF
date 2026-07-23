from __future__ import annotations

from pathlib import Path

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_envelopes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_static_sport_row_gets_raw_and_normalized_event_envelopes() -> None:
    pair = build_static_sport_observation_envelopes(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "1" * 64,
        raw_record_ordinal=17,
        partition="season-2025",
        fetched_at="2026-07-22T12:00:00Z",
        source_at="2025-09-04T20:20:00Z",
        competition_id="cmp_nfl",
        game_id="game_nflverse_2025_01_DAL_PHI",
        participant_ids=("participant_DAL", "participant_PHI"),
        native_namespace="nflverse.play",
        native_id="2025_01_DAL_PHI:101",
        normalized_payload={
            "sport": "nfl",
            "source_play_id": "101",
            "down": 1,
            "distance": 10,
        },
        quality_flags=("source_clock_unverified",),
    )

    assert pair.raw.event_type == "raw_observation"
    assert pair.normalized.event_type == "normalized_observation"
    assert pair.raw.event_id != pair.normalized.event_id
    assert pair.normalized.lineage.parent_event_ids == (pair.raw.event_id,)
    assert pair.normalized.experiment_id == "X-11"
    assert pair.normalized.canonical_refs.game_id == (
        "game_nflverse_2025_01_DAL_PHI"
    )
    assert pair.normalized.payload["down"] == 1


def test_static_envelope_rejects_binary_float_and_invalid_lineage() -> None:
    arguments = {
        "program_root": PROJECT_ROOT,
        "experiment_id": "X-11",
        "dataset_id": "DS-NFLVERSE",
        "source_system": "nflverse",
        "source_stream": "play_by_play",
        "raw_object_hash": "sha256:" + "1" * 64,
        "raw_record_ordinal": 17,
        "partition": "season-2025",
        "fetched_at": "2026-07-22T12:00:00Z",
        "source_at": None,
        "competition_id": "cmp_nfl",
        "game_id": "game_nflverse_2025_01_DAL_PHI",
        "participant_ids": ("participant_DAL", "participant_PHI"),
        "native_namespace": "nflverse.play",
        "native_id": "2025_01_DAL_PHI:101",
        "quality_flags": ("source_clock_unverified",),
    }
    with pytest.raises(ValueError, match="float"):
        build_static_sport_observation_envelopes(
            **arguments,
            normalized_payload={"yardline_100": 45.0},
        )

    with pytest.raises(Exception, match="raw_object_hash|sha256"):
        build_static_sport_observation_envelopes(
            **{
                **arguments,
                "raw_object_hash": "not-a-hash",
            },
            normalized_payload={"yardline_100": 45},
        )
