from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.research.nfl_x15_finalized_episodes import (
    DEFAULT_FACTS_BATCH_FILE_SHA256,
    FinalizedEpisodeError,
    build_finalized_episode_tables,
    publish_finalized_episode_batch,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _row(
    play_id: int,
    *,
    game_id: str = "2025_14_DAL_DET",
    home_team: str = "DET",
    away_team: str = "DAL",
    beneficiary_team: str = "DET",
    action: str = "PASS",
    tags: tuple[str, ...] = ("PASS_ATTEMPT",),
    score_sequence_id: str | None = None,
    adjudication_sequence_id: str | None = None,
    row_disposition: str = "LIVE_PLAY",
    information_status: str = "FINAL",
    review_result: str | None = None,
    quarter: int = 4,
    start: str | None = None,
    end: str | None = None,
    pre_home: int = 10,
    pre_away: int = 7,
    post_home: int = 10,
    post_away: int = 7,
    stage_b_eligible: bool = True,
    final_outcome_eligible: bool = True,
) -> dict[str, object]:
    start = start or f"2025-12-05T03:00:{play_id:02d}.000000Z"
    end = end or f"2025-12-05T03:00:{play_id:02d}.500000Z"
    return {
        "schema_version": "EpisodeFactV3",
        "claim_boundary": "SPORTS_FACT_DESCRIPTION_ONLY_NO_MARKET_REACTION",
        "game_id": game_id,
        "event_id": f"{game_id}:episode:{play_id}",
        "raw_play_id": str(play_id),
        "play_id": str(play_id),
        "atomic_information_episode_id": f"{game_id}:episode:{play_id}",
        "score_sequence_id": score_sequence_id,
        "adjudication_sequence_id": adjudication_sequence_id,
        "information_status": information_status,
        "stage_b_information_event_eligible": stage_b_eligible,
        "final_sports_outcome_eligible": final_outcome_eligible,
        "source_interval_start": start,
        "source_interval_end": end,
        "source_resolution": "MICROSECOND",
        "source_interval_semantics": "[START,END)",
        "known_at": start,
        "order_sequence": play_id,
        "source_time_utc": start,
        "source_time_semantics": "NFLVERSE_SOURCE_POINT_NOT_LOCAL_RECEIVE_TIME",
        "quarter": quarter,
        "game_clock": "05:00",
        "game_seconds_remaining": 300,
        "home_team": home_team,
        "away_team": away_team,
        "actor_team": beneficiary_team,
        "beneficiary_team": beneficiary_team,
        "actor_is_home": True,
        "beneficiary_is_home": True,
        "possession_is_home": True,
        "beneficiary_resolution_status": "RESOLVED_FINAL_SPORTS_RULE",
        "possession_before": "DET",
        "posteam_semantics": "NFLVERSE_POSTEAM_IS_SCRIMMAGE_OFFENSE",
        "defense_team": "DAL",
        "next_observed_possession": "DET",
        "next_observed_possession_semantics": "AUDIT",
        "offense_direction": "LEFT",
        "transition_direction_semantics": "STRUCTURED",
        "field_orientation_semantics": "TEAM_NORMALIZED",
        "pre_home_score": pre_home,
        "pre_away_score": pre_away,
        "post_home_score": post_home,
        "post_away_score": post_away,
        "score_margin_home": pre_home - pre_away,
        "score_margin_offense": pre_home - pre_away,
        "down": 1,
        "distance": 10,
        "goal_to_go": False,
        "pre_yardline": "DAL 20",
        "post_yardline": "DAL 10",
        "pre_field_coordinate_0_100": 20.0,
        "post_field_coordinate_0_100": 10.0,
        "next_observed_state_play_id": str(play_id + 1),
        "next_observed_possession_state": "DET",
        "next_observed_yardline": "DAL 10",
        "next_observed_field_coordinate_0_100": 10.0,
        "visual_end_field_coordinate_0_100": 10.0,
        "visual_end_semantics": "STRUCTURED",
        "yardline_100": 20.0,
        "pre_red_zone": True,
        "tied": False,
        "one_score_game": True,
        "drive_id": "1",
        "series_id": "1",
        "primary_action": action,
        "outcome_tags": json.dumps(tags, separators=(",", ":")),
        "yards_gained": 10.0,
        "air_yards": 5.0,
        "yards_after_catch": 5.0,
        "return_yards": None,
        "pass_length": "short",
        "pass_location": "middle",
        "run_location": None,
        "run_gap": None,
        "kick_distance": None,
        "field_goal_result": None,
        "extra_point_result": None,
        "two_point_result": None,
        "penalty_team": None,
        "penalty_type": None,
        "penalty_yards": None,
        "review_result": review_result,
        "timeout_team": None,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "description": "not used for episode linkage",
        "participation_status": "PRESENT_COMPLETE_11V11",
        "factor_eligible": final_outcome_eligible,
        "row_disposition": row_disposition,
        "home_wp_before_diagnostic": 0.6,
        "home_wp_after_diagnostic": 0.7,
        "home_wp_delta_diagnostic": 0.1,
        "reference_semantics": "DIAGNOSTIC",
        "source_hashes": json.dumps([SHA_A, SHA_B]),
        "pbp_source_sha256": SHA_A,
        "participation_source_sha256": SHA_B,
    }


def test_touchdown_chain_publishes_two_anchors_without_duplicate_rows() -> None:
    sequence = "2025_14_DAL_DET:score:10"
    rows = [
        _row(
            10,
            action="PASS",
            tags=("PASS_TOUCHDOWN", "SCORE", "TOUCHDOWN"),
            score_sequence_id=sequence,
            post_home=16,
        ),
        _row(
            11,
            action="TRY",
            tags=("TWO_POINT_SUCCESS", "SCORE"),
            score_sequence_id=sequence,
            pre_home=16,
            post_home=18,
        ),
        # A malformed upstream score id must not pull the kickoff into the TD chain.
        _row(
            12,
            action="KICKOFF",
            tags=("KICKOFF_RETURN",),
            score_sequence_id=sequence,
            pre_home=18,
            post_home=18,
        ),
    ]

    result = build_finalized_episode_tables(pd.DataFrame(rows))

    assert len(result.episodes) == 2
    touchdown = result.episodes.loc[
        result.episodes["episode_type"].eq("TOUCHDOWN_CHAIN")
    ].iloc[0]
    assert touchdown["constituent_count"] == 2
    assert touchdown["final_score_points"] == 8
    assert touchdown["first_seen_interval_end"] == rows[0]["source_interval_end"]
    assert touchdown["finalized_anchor"] == rows[1]["source_interval_end"]
    assert touchdown["finalization_gap_seconds"] == pytest.approx(1.0)
    assert touchdown["residual_diagnostic_eligible"]
    assert "provisional_primary_action" not in result.episodes.columns
    assert touchdown["first_seen_interval_start"] == rows[0]["source_interval_start"]
    assert touchdown["final_known_interval_end"] == rows[1]["source_interval_end"]
    assert result.episodes["episode_id"].is_unique
    assert len(result.episodes) == result.episodes["episode_id"].nunique()
    anchors = result.information_anchors.set_index("event_id")
    assert len(anchors) == 3
    assert anchors.index.is_unique
    assert anchors.loc[rows[0]["event_id"], "episode_id"] == touchdown[
        "episode_id"
    ]
    assert anchors.loc[rows[1]["event_id"], "episode_id"] == touchdown[
        "episode_id"
    ]
    assert anchors.loc[
        rows[0]["event_id"], "provisional_score_points_observed"
    ] == 6
    assert anchors.loc[
        rows[0]["event_id"], "provisional_primary_action"
    ] == "PASS"
    assert anchors.loc[
        rows[0]["event_id"], "provisional_feature_support_status"
    ] == "SUPPORTED"
    assert anchors.loc[rows[0]["event_id"], "primary_selection_eligible"]
    assert anchors.loc[rows[0]["event_id"], "censor_boundary_eligible"]
    assert str(
        anchors.loc[rows[0]["event_id"], "provisional_snapshot_sha256"]
    ).startswith("sha256:")
    kickoff = result.episodes.loc[
        result.episodes["episode_type"].eq("KICKOFF")
    ].iloc[0]
    assert kickoff["constituent_count"] == 1
    roles = result.constituents.set_index("event_id")["constituent_role"]
    assert roles.loc[rows[0]["event_id"]] == "TOUCHDOWN"
    assert roles.loc[rows[1]["event_id"]] == "TRY"
    assert roles.loc[rows[2]["event_id"]] == "PRIMARY"


def test_touchdown_without_try_is_data_gap_without_explicit_structured_evidence() -> None:
    row = _row(
        20,
        action="RUN",
        tags=("RUSH_TOUCHDOWN", "SCORE", "TOUCHDOWN"),
        score_sequence_id="2025_14_DAL_DET:score:20",
        post_home=16,
    )

    result = build_finalized_episode_tables(pd.DataFrame([row]))

    episode = result.episodes.iloc[0]
    assert episode["episode_status"] == "DATA_GAP"
    assert episode["exclusion_reason"] == "MISSING_TRY_OR_GAME_END_EVIDENCE"
    assert not episode["stage_a_eligible"]
    assert not episode["residual_diagnostic_eligible"]
    anchor = result.information_anchors.iloc[0]
    assert anchor["primary_selection_eligible"]
    assert anchor["censor_boundary_eligible"]
    assert result.reconciliation.iloc[0]["data_gap_episode_count"] == 1

    evidenced = dict(row)
    evidenced["outcome_tags"] = json.dumps(
        ["RUSH_TOUCHDOWN", "SCORE", "TOUCHDOWN", "GAME_END_NO_TRY_CONFIRMED"]
    )
    accepted = build_finalized_episode_tables(pd.DataFrame([evidenced]))
    assert accepted.episodes.iloc[0]["episode_status"] == "FINAL"
    assert accepted.episodes.iloc[0]["stage_a_eligible"]

    nullified_try = _row(
        21,
        action="TRY",
        tags=("NO_PLAY", "EXTRA_POINT_GOOD"),
        score_sequence_id=str(row["score_sequence_id"]),
        row_disposition="NULLIFIED_NO_PLAY",
        final_outcome_eligible=False,
        pre_home=16,
        post_home=16,
    )
    still_missing = build_finalized_episode_tables(
        pd.DataFrame([row, nullified_try])
    )
    assert still_missing.episodes.iloc[0]["episode_status"] == "DATA_GAP"
    assert (
        still_missing.episodes.iloc[0]["exclusion_reason"]
        == "MISSING_TRY_OR_GAME_END_EVIDENCE"
    )


def test_provisional_snapshot_is_invariant_to_later_try_outcome() -> None:
    sequence = "2025_14_DAL_DET:score:22"
    touchdown = _row(
        22,
        action="PASS",
        tags=("PASS_TOUCHDOWN", "SCORE", "TOUCHDOWN"),
        score_sequence_id=sequence,
        post_home=16,
    )
    try_good = _row(
        23,
        action="TRY",
        tags=("EXTRA_POINT_GOOD", "SCORE"),
        score_sequence_id=sequence,
        pre_home=16,
        post_home=17,
    )
    try_failed = dict(try_good)
    try_failed["outcome_tags"] = json.dumps(["EXTRA_POINT_FAILED"])
    try_failed["post_home_score"] = 16

    good_tables = build_finalized_episode_tables(
        pd.DataFrame([touchdown, try_good])
    )
    failed_tables = build_finalized_episode_tables(
        pd.DataFrame([touchdown, try_failed])
    )
    good = good_tables.episodes.iloc[0]
    failed = failed_tables.episodes.iloc[0]
    good_anchor = good_tables.information_anchors.loc[
        good_tables.information_anchors["event_id"].eq(touchdown["event_id"])
    ].iloc[0]
    failed_anchor = failed_tables.information_anchors.loc[
        failed_tables.information_anchors["event_id"].eq(
            touchdown["event_id"]
        )
    ].iloc[0]

    assert good_anchor["provisional_snapshot_sha256"] == failed_anchor[
        "provisional_snapshot_sha256"
    ]
    assert good_anchor["provisional_score_points_observed"] == 6
    assert failed_anchor["provisional_score_points_observed"] == 6
    assert good_anchor["primary_selection_eligible"]
    assert failed_anchor["primary_selection_eligible"]
    assert good["final_score_points"] == 7
    assert failed["final_score_points"] == 6


def test_linked_reviewed_touchdown_keeps_final_score_and_no_play_is_nullified() -> None:
    # Exact frozen DET@GB regression chain: the reviewed ruling is recorded as
    # "reversed", but the final structured outcome is a DET TD plus PAT.
    score = "2025_01_DET_GB:score:3730"
    rows = [
        _row(
            3730,
            game_id="2025_01_DET_GB",
            home_team="GB",
            away_team="DET",
            beneficiary_team="DET",
            tags=(
                "PASS_TOUCHDOWN",
                "TOUCHDOWN",
                "SCORE",
                "REVIEWED",
                "REVIEW_REVERSED",
            ),
            score_sequence_id=score,
            review_result="reversed",
            stage_b_eligible=False,
            start="2025-09-07T23:08:05.643000Z",
            end="2025-09-07T23:08:05.644000Z",
            pre_home=27,
            pre_away=6,
            post_home=27,
            post_away=12,
        ),
        _row(
            3760,
            game_id="2025_01_DET_GB",
            home_team="GB",
            away_team="DET",
            beneficiary_team="DET",
            action="TRY",
            tags=("EXTRA_POINT_GOOD", "SCORE"),
            score_sequence_id=score,
            start="2025-09-07T23:10:53Z",
            end="2025-09-07T23:10:54Z",
            pre_home=27,
            pre_away=12,
            post_home=27,
            post_away=13,
        ),
        _row(
            32,
            game_id="2025_01_DET_GB",
            action="PENALTY",
            tags=("NO_PLAY", "TURNOVER", "TOUCHDOWN"),
            row_disposition="NULLIFIED_NO_PLAY",
            final_outcome_eligible=False,
        ),
    ]

    result = build_finalized_episode_tables(pd.DataFrame(rows))

    reviewed_score = result.episodes.loc[
        result.episodes["score_sequence_id"].eq(score)
    ].iloc[0]
    assert reviewed_score["episode_status"] == "FINAL"
    assert reviewed_score["episode_type"] == "TOUCHDOWN_CHAIN"
    assert reviewed_score["final_score_points"] == 7
    assert not reviewed_score["final_turnover"]
    assert "TOUCHDOWN" in json.loads(reviewed_score["final_outcome_tags"])
    assert reviewed_score["stage_a_eligible"]
    assert reviewed_score["residual_diagnostic_eligible"]
    reviewed_anchor = result.information_anchors.loc[
        result.information_anchors["event_id"].eq(rows[0]["event_id"])
    ].iloc[0]
    try_anchor = result.information_anchors.loc[
        result.information_anchors["event_id"].eq(rows[1]["event_id"])
    ].iloc[0]
    assert reviewed_anchor["provisional_score_points_observed"] == 6
    assert not reviewed_anchor["primary_selection_eligible"]
    assert reviewed_anchor["censor_boundary_eligible"]
    assert (
        reviewed_anchor["provisional_feature_support_status"]
        == "PRIMARY_FEATURE_SUPPORT_UNPROVEN"
    )
    assert (
        reviewed_anchor["provisional_feature_support_reason"]
        == "REVIEW_REVISION_CHRONOLOGY_UNAVAILABLE"
    )
    assert try_anchor["primary_selection_eligible"]
    assert try_anchor["censor_boundary_eligible"]
    assert reviewed_anchor["episode_id"] == try_anchor["episode_id"]
    assert reviewed_score["first_seen_interval_end"] == rows[0]["source_interval_end"]
    assert reviewed_score["finalized_anchor"] == rows[1]["source_interval_end"]
    no_play_episode_id = result.information_anchors.loc[
        result.information_anchors["event_id"].eq(rows[2]["event_id"]),
        "episode_id",
    ].iloc[0]
    no_play = result.episodes.loc[
        result.episodes["episode_id"].eq(no_play_episode_id)
    ].iloc[0]
    assert no_play["episode_status"] == "NULLIFIED"
    assert no_play["final_score_points"] == 0
    assert not no_play["final_turnover"]
    assert json.loads(no_play["final_outcome_tags"]) == []


def test_unlinked_reversal_is_data_gap_and_regulation_excludes_ot() -> None:
    rows = [
        _row(
            40,
            tags=("PASS_ATTEMPT", "REVIEW", "TOUCHDOWN", "TURNOVER"),
            review_result="reversed",
            post_home=16,
        ),
        _row(41, action="RUN", tags=("RUSH_ATTEMPT",), quarter=5),
    ]

    result = build_finalized_episode_tables(pd.DataFrame(rows))

    assert len(result.episodes) == 1
    gap = result.episodes.iloc[0]
    assert gap["episode_status"] == "DATA_GAP"
    assert gap["exclusion_reason"] == "UNLINKED_REVERSAL"
    assert not gap["stage_a_eligible"]
    assert not gap["residual_diagnostic_eligible"]
    anchor = result.information_anchors.iloc[0]
    assert not anchor["primary_selection_eligible"]
    assert anchor["censor_boundary_eligible"]
    assert gap["final_score_points"] == 0
    assert not gap["final_turnover"]
    assert json.loads(gap["final_outcome_tags"]) == []
    ot_mapping = result.constituents.loc[
        result.constituents["event_id"].eq(rows[1]["event_id"])
    ].iloc[0]
    assert ot_mapping["mapping_status"] == "EXCLUDED"
    assert ot_mapping["exclusion_reason"] == "OT_QUARTER_GE_5"
    audit = result.reconciliation.iloc[0]
    assert audit["raw_row_count"] == 2
    assert audit["mapped_row_count"] == 1
    assert audit["excluded_row_count"] == 1
    assert audit["raw_rows_silently_dropped"] == 0
    assert audit["publication_gate"] == "PASS"


def test_timeout_is_stage_b_event_even_when_stage_a_and_no_play_are_false() -> None:
    row = _row(
        3915,
        game_id="2025_01_ARI_NO",
        action="TIMEOUT",
        tags=("NO_PLAY", "TIMEOUT"),
        row_disposition="NULLIFIED_NO_PLAY",
        final_outcome_eligible=False,
        stage_b_eligible=True,
        start="2025-09-07T19:37:05Z",
        end="2025-09-07T19:37:06Z",
    )

    result = build_finalized_episode_tables(pd.DataFrame([row]))

    assert len(result.episodes) == 1
    episode = result.episodes.iloc[0]
    assert episode["episode_status"] == "FINAL"
    assert episode["episode_type"] == "TIMEOUT"
    assert not episode["stage_a_eligible"]
    assert episode["residual_diagnostic_eligible"]
    anchor = result.information_anchors.iloc[0]
    assert anchor["primary_selection_eligible"]
    assert anchor["censor_boundary_eligible"]
    assert anchor["provisional_primary_action"] == "TIMEOUT"
    assert anchor["provisional_feature_support_status"] == "SUPPORTED"
    assert episode["first_seen_interval_end"] == row["source_interval_end"]
    assert episode["finalized_anchor"] == row["source_interval_end"]
    assert episode["finalization_gap_seconds"] == pytest.approx(0.0)
    assert episode["final_score_points"] == 0
    assert not episode["final_turnover"]


def test_canonical_non_target_remains_censor_but_not_primary_target() -> None:
    row = _row(
        42,
        action="PASS",
        tags=("PASS_ATTEMPT",),
        stage_b_eligible=False,
    )

    result = build_finalized_episode_tables(pd.DataFrame([row]))

    anchor = result.information_anchors.iloc[0]
    assert anchor["provisional_feature_support_status"] == "SUPPORTED"
    assert not anchor["canonical_stage_b_information_event_eligible"]
    assert not anchor["primary_selection_eligible"]
    assert anchor["censor_boundary_eligible"]


def test_kickoff_return_touchdown_joins_try_but_ordinary_kickoff_does_not() -> None:
    # Exact frozen NE@MIA kickoff-return TD regression chain.
    score = "2025_02_NE_MIA:score:3418"
    rows = [
        _row(
            3418,
            game_id="2025_02_NE_MIA",
            home_team="MIA",
            away_team="NE",
            beneficiary_team="NE",
            action="KICKOFF",
            tags=("KICKOFF_RETURN", "RETURN_TOUCHDOWN", "TOUCHDOWN", "SCORE"),
            score_sequence_id=score,
            start="2025-09-14T19:39:10Z",
            end="2025-09-14T19:39:11Z",
            pre_home=27,
            pre_away=23,
            post_home=27,
            post_away=29,
        ),
        _row(
            3449,
            game_id="2025_02_NE_MIA",
            home_team="MIA",
            away_team="NE",
            beneficiary_team="NE",
            action="TRY",
            tags=("EXTRA_POINT_GOOD", "SCORE"),
            score_sequence_id=score,
            start="2025-09-14T19:40:15Z",
            end="2025-09-14T19:40:16Z",
            pre_home=27,
            pre_away=29,
            post_home=27,
            post_away=30,
        ),
        _row(
            45,
            game_id="2025_02_NE_MIA",
            action="KICKOFF",
            tags=("KICKOFF_RETURN",),
            score_sequence_id=score,
            pre_home=27,
            pre_away=30,
            post_home=27,
            post_away=30,
        ),
    ]

    result = build_finalized_episode_tables(pd.DataFrame(rows))

    assert len(result.episodes) == 2
    touchdown = result.episodes.loc[
        result.episodes["episode_type"].eq("TOUCHDOWN_CHAIN")
    ].iloc[0]
    assert touchdown["constituent_count"] == 2
    assert touchdown["final_score_points"] == 7
    return_anchor = result.information_anchors.loc[
        result.information_anchors["event_id"].eq(rows[0]["event_id"])
    ].iloc[0]
    try_anchor = result.information_anchors.loc[
        result.information_anchors["event_id"].eq(rows[1]["event_id"])
    ].iloc[0]
    assert return_anchor["provisional_score_points_observed"] == 6
    assert return_anchor["episode_id"] == try_anchor["episode_id"]
    assert return_anchor["information_anchor"] < try_anchor[
        "information_anchor"
    ]
    ordinary = result.episodes.loc[
        result.episodes["episode_type"].eq("KICKOFF")
    ].iloc[0]
    assert ordinary["constituent_count"] == 1


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_fake_exact_facts(
    root: Path, rows: list[dict[str, object]]
) -> tuple[Path, str]:
    dataset_root = root / "artifacts/market-observation/nfl/x13"
    frame = pd.DataFrame(rows)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    object_bytes = sink.getvalue().to_pybytes()
    object_sha = _sha(object_bytes)
    object_digest = object_sha.removeprefix("sha256:")
    object_rel = (
        Path("exact-153-facts-v4")
        / "single-game"
        / "2025_14_DAL_DET"
        / "objects"
        / "sha256"
        / object_digest[:2]
        / f"{object_digest}.parquet"
    )
    object_path = dataset_root / object_rel
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(object_bytes)
    game_material = {
        "schema": "nfl_x13_exact153_single_game_fact_manifest_v4",
        "publication_id": "exact-153-facts-v4",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_id": "2025_14_DAL_DET",
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "publication_gate": "PASS",
        "tables": [
            {
                "name": "canonical_factor_events",
                "object_path": object_rel.as_posix(),
                "object_sha256": object_sha,
                "byte_length": len(object_bytes),
                "row_count": len(rows),
            }
        ],
    }
    game_material["bundle_sha256"] = _sha(
        _canonical_bytes(game_material)
    )
    game_bytes = _canonical_bytes(game_material)
    game_sha = _sha(game_bytes)
    game_digest = game_sha.removeprefix("sha256:")
    game_rel = (
        Path("exact-153-facts-v4")
        / "single-game"
        / "2025_14_DAL_DET"
        / "manifests"
        / "sha256"
        / game_digest[:2]
        / f"{game_digest}.manifest.json"
    )
    game_path = dataset_root / game_rel
    game_path.parent.mkdir(parents=True)
    game_path.write_bytes(game_bytes)
    batch_material = {
        "schema": "nfl_x13_exact153_fact_batch_index_v4",
        "publication_id": "exact-153-facts-v4",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": 1,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "publication_gate": "PASS",
        "games": [
            {
                "game_id": "2025_14_DAL_DET",
                "manifest_path": game_rel.as_posix(),
                "manifest_sha256": game_sha,
                "bundle_sha256": game_material["bundle_sha256"],
            }
        ],
    }
    batch_material["batch_sha256"] = _sha(_canonical_bytes(batch_material))
    batch_bytes = _canonical_bytes(batch_material)
    batch_sha = _sha(batch_bytes)
    batch_digest = batch_sha.removeprefix("sha256:")
    batch_path = (
        dataset_root
        / "exact-153-facts-v4"
        / "batches"
        / "manifests"
        / "sha256"
        / batch_digest[:2]
        / f"{batch_digest}.batch-index.json"
    )
    batch_path.parent.mkdir(parents=True)
    batch_path.write_bytes(batch_bytes)
    return batch_path, batch_sha


def test_publish_is_content_addressed_and_bound_only_to_verified_facts(
    tmp_path: Path,
) -> None:
    rows = [_row(50, action="RUN", tags=("RUSH_ATTEMPT",))]
    batch_path, batch_sha = _write_fake_exact_facts(tmp_path, rows)
    output_root = tmp_path / "artifacts/market-observation/nfl/x15"

    publication = publish_finalized_episode_batch(
        project_root=tmp_path,
        facts_batch_path=batch_path,
        facts_batch_file_sha256=batch_sha,
        expected_game_count=1,
        output_root=output_root,
    )

    assert publication.game_count == 1
    assert publication.raw_row_count == 1
    assert publication.episode_count == 1
    assert publication.raw_rows_silently_dropped == 0
    assert publication.manifest_path.is_file()
    manifest = json.loads(publication.manifest_path.read_text())
    assert manifest["source_scope"] == "VERIFIED_EXACT_153_FACTS_V4_ONLY"
    assert manifest["market_data_read"] is False
    assert manifest["holdout_reaction_accessed"] is False
    assert manifest["schema"] == "nfl_x15_finalized_episode_batch_v2"
    assert manifest["publication_id"] == "finalized-episodes-regulation-v2"
    assert manifest["anchor_kinds"] == [
        {
            "analysis_role": "PRIMARY_SELECTION",
            "anchor_kind": "PROVISIONAL_FIRST_SEEN",
            "column": "information_anchor",
            "table": "provisional_information_anchors",
        },
        {
            "analysis_role": "RESIDUAL_CORRECTION_DIAGNOSTIC",
            "anchor_kind": "FINALIZED",
            "column": "finalized_anchor",
            "table": "finalized_episodes",
        },
    ]
    assert manifest["provisional_feature_contract"][
        "selection_rule"
    ] == "EACH_MAPPED_CONSTITUENT_SNAPSHOT_NO_FINAL_OUTCOME_DEPENDENCY"
    assert manifest["provisional_feature_contract"][
        "snapshot_sha256_column"
    ] == "provisional_snapshot_sha256"
    assert manifest["counts"]["information_anchor_count"] == 1
    assert manifest["counts"]["information_anchor_count"] == manifest[
        "counts"
    ]["mapped_row_count"]
    assert {item["name"] for item in manifest["tables"]} == {
        "finalized_episodes",
        "provisional_information_anchors",
        "episode_constituents",
        "episode_reconciliation",
    }
    rerun = publish_finalized_episode_batch(
        project_root=tmp_path,
        facts_batch_path=batch_path,
        facts_batch_file_sha256=batch_sha,
        expected_game_count=1,
        output_root=output_root,
    )
    assert rerun.manifest_sha256 == publication.manifest_sha256
    assert rerun.manifest_path == publication.manifest_path

    batch_path.write_bytes(batch_path.read_bytes() + b"\n")
    with pytest.raises(FinalizedEpisodeError, match="facts batch hash mismatch"):
        publish_finalized_episode_batch(
            project_root=tmp_path,
            facts_batch_path=batch_path,
            facts_batch_file_sha256=batch_sha,
            expected_game_count=1,
            output_root=output_root,
        )


def test_default_source_pin_is_the_exact_v4_batch() -> None:
    assert DEFAULT_FACTS_BATCH_FILE_SHA256 == (
        "sha256:5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    )
