from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from prediction_market.contracts import canonical_sha256
from prediction_market.sports import mlb_game_state as mlb
from prediction_market.sports import mlb_season_census as census


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "var/raw"
    / mlb.RETROSHEET_2025_MANIFEST_RELATIVE_PATH
)
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts/game-state/mlb/season_state_census_v0.json"
)


def _row(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "GAME_ID": "ANA202504040",
        "INN_CT": "1",
        "BAT_HOME_ID": "0",
        "OUTS_CT": "0",
        "BALLS_CT": "0",
        "STRIKES_CT": "0",
        "AWAY_SCORE_CT": "0",
        "HOME_SCORE_CT": "0",
        "BAT_ID": "batter001",
        "PIT_ID": "pitcher01",
        "BASE1_RUN_ID": "",
        "BASE2_RUN_ID": "",
        "BASE3_RUN_ID": "",
        "EVENT_TX": "HR",
        "BAT_LINEUP_ID": "1",
        "EVENT_CD": "23",
        "BAT_EVENT_FL": "T",
        "EVENT_OUTS_CT": "0",
        "BAT_DEST_ID": "4",
        "RUN1_DEST_ID": "0",
        "RUN2_DEST_ID": "0",
        "RUN3_DEST_ID": "0",
        "GAME_END_FL": "F",
        "EVENT_ID": "1",
    }
    values.update(overrides)
    assert set(values) == set(mlb.CWEVENT_FIELD_NAMES)
    return values


def test_current_row_result_mutation_is_caught_by_next_row_comparison() -> None:
    play = _row()
    following = _row(
        AWAY_SCORE_CT="1",
        BAT_ID="batter002",
        EVENT_TX="63/G",
        EVENT_CD="2",
        EVENT_OUTS_CT="1",
        BAT_DEST_ID="0",
        EVENT_ID="2",
    )

    valid = census.evaluate_transition(
        play,
        following,
        away_team="CLE",
        home_team="ANA",
    )
    assert valid.causal_core_mismatches == ()

    tampered = dict(play)
    tampered["BAT_DEST_ID"] = "1"
    detected = census.evaluate_transition(
        tampered,
        following,
        away_team="CLE",
        home_team="ANA",
    )
    assert detected.causal_core_mismatches == (
        "bases",
        "score.away",
    )


def test_next_row_fields_are_labeled_same_source_offline_not_oracle() -> None:
    following = _row(
        AWAY_SCORE_CT="1",
        BAT_ID="batter002",
        PIT_ID="pitcher02",
        BALLS_CT="2",
        STRIKES_CT="1",
        BAT_LINEUP_ID="2",
        EVENT_TX="63/G",
        EVENT_CD="2",
        EVENT_OUTS_CT="1",
        BAT_DEST_ID="0",
        EVENT_ID="2",
    )
    result = census.evaluate_transition(
        _row(),
        following,
        away_team="CLE",
        home_team="ANA",
    )

    assert census.EVIDENCE_SPLIT["causal_core"]["source"] == (
        "current_cwevent_row"
    )
    assert census.EVIDENCE_SPLIT["same_source_offline_observation"] == {
        "fields": [
            "batter_id",
            "pitcher_id",
            "balls",
            "strikes",
            "lineup_slot",
            "base_runner_ids",
        ],
        "source": "immediate_next_cwevent_play_row",
        "classification": "same_source_next_play_context",
        "independent_oracle": False,
        "permitted_claim": "same_source_offline_consistency_only",
    }
    assert result.same_source_offline_observation == {
        "batter_id": "batter002",
        "pitcher_id": "pitcher02",
        "balls": 2,
        "strikes": 1,
        "lineup_slot": 2,
        "base_runner_ids": [None, None, None],
    }
    assert result.base_runner_identity_changed_after_omitted_np is False
    assert result.independent_oracle is False


def test_inning_end_survivor_collision_requires_all_canonicalization_gates() -> None:
    play = _row(
        INN_CT="5",
        BAT_HOME_ID="1",
        OUTS_CT="2",
        BAT_ID="solej001",
        BASE1_RUN_ID="schan001",
        BASE2_RUN_ID="parik002",
        BASE3_RUN_ID="rengl001",
        EVENT_TX="5(2)/FO/G56.B-1",
        EVENT_CD="2",
        EVENT_OUTS_CT="1",
        BAT_DEST_ID="1",
        RUN1_DEST_ID="1",
        RUN2_DEST_ID="0",
        RUN3_DEST_ID="3",
    )
    following = _row(
        INN_CT="6",
        BAT_HOME_ID="0",
        OUTS_CT="0",
        BAT_ID="nextbat01",
        BASE1_RUN_ID="",
        BASE2_RUN_ID="",
        BASE3_RUN_ID="",
        EVENT_TX="63/G",
        EVENT_CD="2",
        EVENT_OUTS_CT="1",
        BAT_DEST_ID="0",
        EVENT_ID="2",
    )

    accepted = census.evaluate_transition(
        play,
        following,
        away_team="TOR",
        home_team="ANA",
    )
    assert accepted.causal_core_mismatches == ()
    assert accepted.inning_end_survivor_destinations_normalized == 3

    bad_text = dict(play)
    bad_text["EVENT_TX"] = "FO"
    with pytest.raises(
        census.MLBSeasonCensusError,
        match="unsupported inning-end survivor destination collision",
    ):
        census.evaluate_transition(
            bad_text,
            following,
            away_team="TOR",
            home_team="ANA",
        )

    wrong_half = dict(following)
    wrong_half["INN_CT"] = "5"
    wrong_half["BAT_HOME_ID"] = "1"
    with pytest.raises(
        census.MLBSeasonCensusError,
        match="unsupported inning-end survivor destination collision",
    ):
        census.evaluate_transition(
            play,
            wrong_half,
            away_team="TOR",
            home_team="ANA",
        )


def test_canonical_season_command_identity_ignores_executable_path() -> None:
    event_files = tuple(f"2025A{index:02d}.EVA" for index in range(30))
    binary_sha256 = "sha256:" + "1" * 64
    first_runtime = mlb.CweventRuntime(
        executable="/opt/homebrew/bin/cwevent",
        version=mlb.CHADWICK_CWEVENT_VERSION,
        binary_sha256=binary_sha256,
    )
    second_runtime = mlb.CweventRuntime(
        executable="/private/tmp/simulated/cwevent",
        version=mlb.CHADWICK_CWEVENT_VERSION,
        binary_sha256=binary_sha256,
    )

    first = census.canonical_season_command_identity(
        first_runtime,
        event_files=event_files,
    )
    second = census.canonical_season_command_identity(
        second_runtime,
        event_files=event_files,
    )

    assert first == second
    assert first == {
        "binary_sha256": binary_sha256,
        "tokens": [
            "cwevent",
            "-q",
            "-n",
            "-y",
            "2025",
            "-f",
            mlb.CWEVENT_FIELD_ARGUMENT,
            *event_files,
        ],
    }
    assert "/opt/homebrew" not in json.dumps(first)
    assert "/private/tmp" not in json.dumps(second)


def _require_frozen_inputs() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip("frozen Retrosheet raw object is not present")
    if shutil.which("cwevent") is None:
        pytest.skip("Chadwick cwevent is not installed")


@pytest.fixture(scope="module")
def full_season_run() -> dict[str, object]:
    _require_frozen_inputs()
    return census.run_frozen_retrosheet_2025_season_census(
        program_root=PROJECT_ROOT
    )


def test_full_2025_season_is_complete_deterministic_and_fail_closed(
    full_season_run: dict[str, object],
) -> None:
    first = full_season_run

    coverage = first["coverage"]
    assert coverage["season"] == 2025
    assert coverage["game_type"] == "regular"
    assert coverage["event_files"] == 30
    assert coverage["games"] == 2430
    assert coverage["games_fully_supported"] == 2213
    assert coverage["games_with_any_unsupported"] == 217
    assert coverage["native_play_records"] == 216845
    assert coverage["native_no_play_records"] == 27534
    assert coverage["omitted_np_gaps"] > 0
    assert coverage["events"] == 189311
    assert coverage["transitions_with_next_row"] == 186881
    assert coverage["terminal_events"] == 2430
    assert coverage["full_native_lifecycle_complete"] is False
    assert first["execution"]["cwevent_invocations"] == 1
    assert first["execution"]["cwevent_version"] == "0.10.0"
    assert first["execution"]["cwevent_field_map_sha256"] == (
        mlb.CWEVENT_FIELD_MAP_SHA256
    )
    assert first["lineage"]["raw_object_sha256"] == (
        mlb.RETROSHEET_2025_RAW_OBJECT_SHA256
    )
    assert first["lineage"]["source_manifest_sha256"] == (
        mlb.RETROSHEET_2025_MANIFEST_SHA256
    )
    assert first["validation"]["games_valid"] == 2213
    assert first["validation"]["games_failed"] == 217
    assert first["validation"]["events_scanned"] == 189311
    assert first["validation"]["events_reducer_validated"] == 188979
    assert first["validation"]["events_failed_closed"] == 932
    assert first["validation"]["failures_by_category"] == {
        "unsupported_automatic_runner_context": 600,
        "unsupported_cwevent_destination_code": 321,
        "unsupported_inning_end_destination_collision": 5,
        "unsupported_non_inning_destination_collision": 1,
        "unsupported_shortened_game_terminal_rule": 5,
    }
    assert first["validation"]["causal_core_mismatches_total"] == 600
    assert set(first["validation"]["causal_core_mismatches_by_field"]) == {
        "inning",
        "half",
        "outs",
        "bases",
        "score.away",
        "score.home",
    }
    assert first["validation"]["causal_core_mismatches_by_field"] == {
        "inning": 0,
        "half": 0,
        "outs": 0,
        "bases": 600,
        "score.away": 0,
        "score.home": 0,
    }
    assert first["validation"]["destination_collision_events_observed"] == 78
    assert first["validation"][
        "inning_end_collision_events_canonicalized"
    ] == 72
    assert first["validation"][
        "unsupported_destination_collision_events"
    ] == 6
    assert first["determinism"]["runs"] == 2
    assert first["determinism"]["canonical_hash_match"] is True
    assert len(set(first["determinism"]["canonical_hashes"])) == 1


def test_reducer_performance_scope_and_offline_evidence_are_explicit(
    full_season_run: dict[str, object],
) -> None:
    first = full_season_run
    performance = first["performance"]["reducer_only"]

    assert performance["samples"] == 188979
    assert performance["scope"] == (
        "reduce_mlb_state_call_only; excludes cwevent, CSV parsing, "
        "event construction, comparison, hashing, and model inference"
    )
    assert 0 < performance["latency_ns"]["p50"]
    assert performance["latency_ns"]["p50"] <= performance["latency_ns"]["p95"]
    assert performance["latency_ns"]["p95"] <= performance["latency_ns"]["p99"]
    assert performance["throughput_transitions_per_second"] > 0
    assert first["evidence_split"] == census.EVIDENCE_SPLIT
    assert first["claims"]["independent_oracle"] is False
    assert first["claims"]["model_trained"] is False
    assert first["claims"]["prediction_accuracy_reported"] is False
    assert first["claims"]["full_native_lifecycle_complete"] is False


def test_checked_in_artifact_omits_research_only_event_text() -> None:
    _require_frozen_inputs()
    serialized = ARTIFACT_PATH.read_text(encoding="utf-8")
    payload = json.loads(serialized)

    assert '"event_text"' not in serialized
    assert "EVENT_TX" not in serialized
    assert "5(2)/FO/G56.B-1" not in serialized
    assert "destination_collision_event_text_counts" not in serialized
    assert "cwevent_executable" not in payload["execution"]
    assert payload["execution"]["cwevent_command"][0] == "cwevent"
    assert payload["execution"]["cwevent_command_sha256"] == canonical_sha256(
        {
            "binary_sha256": payload["execution"]["cwevent_binary_sha256"],
            "tokens": payload["execution"]["cwevent_command"],
        }
    )
    assert "failed_events" not in payload["validation"]
    signatures = payload["validation"]["destination_collision_signatures"]
    assert sum(item["count"] for item in signatures) == 78
    assert all(
        set(item) == {"category", "collision_sha256", "count"}
        and re.fullmatch(r"sha256:[0-9a-f]{64}", item["collision_sha256"])
        for item in signatures
    )
    allowed_failure_fields = {
        "event_index",
        "row_sha256",
        "category",
        "collision_sha256",
    }
    for game in payload["game_manifest"]:
        if game["first_failure"] is not None:
            assert set(game["first_failure"]) <= allowed_failure_fields
        for failure in game["failures"]:
            assert set(failure) <= allowed_failure_fields


def test_checked_in_artifact_matches_fresh_canonical_run(
    full_season_run: dict[str, object],
) -> None:
    _require_frozen_inputs()
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    artifact_sha256 = payload.pop("artifact_sha256")

    assert artifact_sha256 == canonical_sha256(payload)
    assert payload["canonical_reconstruction_sha256"] == full_season_run[
        "canonical_reconstruction_sha256"
    ]
    assert payload["validation"]["game_manifest_sha256"] == full_season_run[
        "validation"
    ]["game_manifest_sha256"]
    assert payload["coverage"]["games"] == 2430
    assert payload["validation"]["causal_core_mismatches_total"] == 600
