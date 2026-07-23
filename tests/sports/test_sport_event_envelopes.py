from __future__ import annotations

from pathlib import Path

import pytest

from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_bundle,
    build_static_sport_observation_envelopes,
    validate_static_sport_observation_bundle,
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


def test_multi_row_bundle_binds_complete_payload_to_every_raw_parent() -> None:
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "2" * 64,
        raw_record_ordinals=(17, 18),
        partition="season-2025",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=None,
        competition_id="cmp_nfl",
        game_id="game_nflverse_2025_01_DAL_PHI",
        participant_ids=("participant_DAL", "participant_PHI"),
        native_namespace="nflverse.play",
        native_ids=("2025_01_DAL_PHI:101", "2025_01_DAL_PHI:122"),
        normalized_source_sequence=18,
        normalized_payload={
            "sport": "nfl",
            "game_id": "game_nflverse_2025_01_DAL_PHI",
            "sequence": 18,
            "source_play_id": "101",
            "next_source_play_id": "122",
            "quality_flags": [],
        },
    )

    validated = validate_static_sport_observation_bundle(
        PROJECT_ROOT,
        bundle.normalized,
        raw_parents=bundle.raw,
        expected_experiment_id="X-11",
        expected_dataset_id="DS-NFLVERSE",
        expected_source_system="nflverse",
        expected_source_stream="play_by_play",
        expected_native_namespace="nflverse.play",
    )

    assert len(bundle.raw) == 2
    assert validated == bundle.normalized
    assert validated.lineage.parent_event_ids == tuple(
        sorted(parent.event_id for parent in bundle.raw)
    )

    with pytest.raises(ValueError, match="parent raw lineage"):
        validate_static_sport_observation_bundle(
            PROJECT_ROOT,
            bundle.normalized,
            raw_parents=(bundle.raw[0],),
            expected_experiment_id="X-11",
            expected_dataset_id="DS-NFLVERSE",
            expected_source_system="nflverse",
            expected_source_stream="play_by_play",
            expected_native_namespace="nflverse.play",
        )


def test_bundle_validation_fails_closed_on_registry_and_dataset(
    tmp_path: Path,
) -> None:
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-WRONG",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash="sha256:" + "3" * 64,
        raw_record_ordinals=(1, 2),
        partition="season-2025",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=None,
        competition_id="cmp_nfl",
        game_id="game_nflverse_2025_01_DAL_PHI",
        participant_ids=("participant_DAL", "participant_PHI"),
        native_namespace="nflverse.play",
        native_ids=("2025_01_DAL_PHI:101", "2025_01_DAL_PHI:122"),
        normalized_source_sequence=2,
        normalized_payload={
            "sport": "nfl",
            "game_id": "game_nflverse_2025_01_DAL_PHI",
            "sequence": 2,
            "source_play_id": "101",
            "next_source_play_id": "122",
            "quality_flags": [],
        },
    )

    with pytest.raises(ValueError, match="dataset"):
        validate_static_sport_observation_bundle(
            PROJECT_ROOT,
            bundle.normalized,
            raw_parents=bundle.raw,
            expected_experiment_id="X-11",
            expected_dataset_id="DS-NFLVERSE",
            expected_source_system="nflverse",
            expected_source_stream="play_by_play",
            expected_native_namespace="nflverse.play",
        )

    with pytest.raises(Exception, match="registry|experiment"):
        validate_static_sport_observation_bundle(
            tmp_path,
            bundle.normalized,
            raw_parents=bundle.raw,
            expected_experiment_id="X-11",
            expected_dataset_id="DS-WRONG",
            expected_source_system="nflverse",
            expected_source_stream="play_by_play",
            expected_native_namespace="nflverse.play",
        )
