from __future__ import annotations

import json
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "registries/factors/nfl_factor_registry_v1.json"
CODE_LOCK = (
    PROJECT_ROOT / "registries/factors/nfl_factor_lab_code_lock_v1.json"
)


def test_factor_registry_covers_frozen_universe_and_has_no_future_claim() -> None:
    payload = json.loads(REGISTRY.read_text("utf-8"))

    assert payload["schema"] == "nfl_factor_registry_v1"
    assert payload["experiment_id"] == "X-13"
    assert payload["status"] == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert payload["bootstrap_samples"] == 10_000
    assert payload["horizons_seconds"] == [1, 2, 5, 10, 30, 60]
    assert payload["third_order_policy"] == (
        "ONLY_HUMAN_ACCEPTED_TWO_DIMENSION_FACTORS"
    )
    definitions = payload["factor_definitions"]
    ids = [item["factor_id"] for item in definitions]
    assert ids == sorted(set(ids))
    assert len(ids) >= 30
    required = {
        "NFL.EVENT.PASS",
        "NFL.EVENT.RUSH",
        "NFL.EVENT.SCRAMBLE",
        "NFL.EVENT.SACK",
        "NFL.SCORE.PASS_TD",
        "NFL.SCORE.RUSH_TD",
        "NFL.SCORE.RETURN_TD",
        "NFL.SCORE.FG_MADE",
        "NFL.SCORE.FG_MISSED",
        "NFL.SCORE.FG_BLOCKED",
        "NFL.SCORE.PAT_MADE",
        "NFL.SCORE.TWO_POINT_SUCCESS",
        "NFL.TURNOVER.INTERCEPTION",
        "NFL.TURNOVER.FUMBLE_LOST",
        "NFL.TURNOVER.ON_DOWNS",
        "NFL.SPECIAL.PUNT",
        "NFL.SPECIAL.SAFETY",
        "NFL.ADMIN.PENALTY",
        "NFL.ADMIN.REVIEW",
        "NFL.ADMIN.TIMEOUT",
        "NFL.COMBO.LATE_CLOSE_SCORE",
        "NFL.COMBO.RED_ZONE_TURNOVER",
        "NFL.COMBO.DISTANCE_GT_10",
        "NFL.SEQUENCE.ANSWER_SCORE",
        "NFL.SEQUENCE.BACK_TO_BACK_TURNOVERS",
        "NFL.SEQUENCE.TWO_MINUTE_DRIVE",
        "NFL.VALUE.PASS_CALL_SURPRISE",
        "NFL.VALUE.COMPLETION_SURPRISE",
        "NFL.VALUE.YAC_SURPRISE",
        "NFL.VALUE.FOURTH_DOWN_DECISION_REGRET",
        "NFL.VALUE.PUNT_EXECUTION_SURPRISE",
    }
    assert required <= set(ids)
    assert "turnover_over_expected" not in REGISTRY.read_text("utf-8").lower()


def test_factor_registry_freezes_market_targets_and_support_gates() -> None:
    payload = json.loads(REGISTRY.read_text("utf-8"))
    allowed_focal = {"team", "player", "possession", "game"}
    for factor in payload["factor_definitions"]:
        assert factor["version"] == "v1"
        assert factor["focal_entity"] in allowed_focal
        assert factor["predicate"]
        assert set(factor["pit_fields"]) >= {
            clause["field"] for clause in factor["predicate"]
        }
        assert factor["market_families"]
        assert set(factor["horizons_seconds"]) <= {1, 2, 5, 10, 30, 60}
        assert factor["support_kind"] in {
            "primary",
            "binary",
            "named",
            "sequence",
            "descriptive",
        }
        assert factor["minimum_episodes"] > 0
        assert factor["minimum_games"] > 0
        assert factor["claim_boundary"] == "PRELIMINARY_SOURCE_TIME_ONLY"

    targets = payload["target_adapters"]
    assert targets["winner_spread"]["orientation"] == "focal_team"
    assert targets["total"]["orientation"] == "over"
    assert targets["team_total"]["orientation"] == "target_team_over"
    assert targets["player_prop"]["orientation"] == "proposition_subject"
    assert targets["composite"]["pooled_inference"] is False
    assert targets["cross_game_mve"]["analysis"] == "inventory_only"


def test_factor_registry_covers_structured_native_outcomes_without_guessing() -> None:
    payload = json.loads(REGISTRY.read_text("utf-8"))
    by_id = {
        item["factor_id"]: item for item in payload["factor_definitions"]
    }
    required = {
        "NFL.ADMIN.PENALTY_ACCEPTED",
        "NFL.ADMIN.PENALTY_DECLINED",
        "NFL.ADMIN.PENALTY_NO_PLAY",
        "NFL.EVENT.FOURTH_DOWN_CONVERSION",
        "NFL.EVENT.FUMBLE_RECOVERY",
        "NFL.PERSONNEL.BETWEEN_PLAY_ROSTER_CHANGE",
        "NFL.SCORE.DEFENSIVE_TWO_POINT_ATTEMPT",
        "NFL.SCORE.PAT_BLOCKED",
        "NFL.SCORE.PAT_MISSED",
        "NFL.SCORE.TWO_POINT_FAILURE",
        "NFL.SPECIAL.KICKOFF_OUT_OF_BOUNDS",
        "NFL.SPECIAL.KICKOFF_OWN_RECOVERY",
        "NFL.SPECIAL.KICKOFF_POSITIVE_RETURN",
        "NFL.SPECIAL.PUNT_BLOCKED",
        "NFL.SPECIAL.PUNT_DOWNED",
        "NFL.SPECIAL.PUNT_FAIR_CATCH",
        "NFL.SPECIAL.PUNT_OUT_OF_BOUNDS",
        "NFL.SPECIAL.PUNT_POSITIVE_RETURN",
        "NFL.SPECIAL.PUNT_TOUCHBACK",
    }
    assert required <= set(by_id)
    assert by_id["NFL.SCORE.PAT_MADE"]["predicate"] == [
        {"field": "extra_point_result", "operator": "eq", "value": "good"}
    ]
    assert by_id["NFL.SCORE.PAT_MISSED"]["predicate"] == [
        {"field": "extra_point_result", "operator": "eq", "value": "failed"}
    ]
    assert by_id["NFL.EVENT.FUMBLE_RECOVERY"]["predicate"] == [
        {
            "field": "fumble_recovery_team",
            "operator": "not_null",
            "value": None,
        }
    ]

    # SportsFeatureRowV1 does not expose a predicate-safe scalar `fumble`.
    # A recorded recovery proves a conservative subset, but event_flags is an
    # array and must not be treated as a scalar occurrence indicator.
    assert "NFL.EVENT.FUMBLE_OCCURRENCE" not in by_id
    assert all(
        clause["field"] != "event_flags"
        for factor in by_id.values()
        for clause in factor["predicate"]
    )

    interaction_events = payload["interaction_templates"][0][
        "event_factor_ids"
    ]
    assert interaction_events == sorted(set(interaction_events))
    assert {
        "NFL.EVENT.FOURTH_DOWN_CONVERSION",
        "NFL.EVENT.FUMBLE_RECOVERY",
        "NFL.SCORE.PAT_BLOCKED",
        "NFL.SCORE.PAT_MISSED",
        "NFL.SCORE.TWO_POINT_FAILURE",
        "NFL.SPECIAL.PUNT_BLOCKED",
        "NFL.SPECIAL.PUNT_POSITIVE_RETURN",
    } <= set(interaction_events)


def test_factor_lab_code_lock_binds_exact_files_and_unread_holdout() -> None:
    lock = json.loads(CODE_LOCK.read_text("utf-8"))
    assert lock["schema"] == "NFLFactorLabCodeLockV1"
    assert lock["holdout_reaction_accessed"] is False
    assert lock["bootstrap_samples"] == 10_000
    for item in lock["files"]:
        raw = (PROJECT_ROOT / item["path"]).read_bytes()
        assert item["sha256"] == (
            "sha256:" + hashlib.sha256(raw).hexdigest()
        )
    material = {
        "schema": "NFLFactorLabCodeLockV1",
        "files": lock["files"],
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert lock["code_sha256"] == (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
