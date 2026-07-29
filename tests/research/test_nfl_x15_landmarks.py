from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from prediction_market.research import nfl_x15_landmarks as landmarks


def _inputs() -> dict[str, object]:
    facts = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "event_id": "event-1",
                "atomic_information_episode_id": "episode-1",
                "source_interval_start": "2025-09-05T00:00:00Z",
                "source_interval_end": "2025-09-05T00:00:01Z",
                "known_at": "2025-09-05T00:00:01Z",
                "source_resolution": "SECOND",
                "stage_b_information_event_eligible": True,
                "home_team": "BBB",
                "away_team": "AAA",
                "pbp_source_sha256": "sha256:" + "c" * 64,
                "game_seconds_remaining": 900,
                "score_margin_home": -3,
                "possession_is_home": False,
                "down": 2,
                "distance": 7,
                "yardline_100": 32.0,
                "primary_action": "PASS",
                "outcome_tags": '["COMPLETE_PASS"]',
                "yards_gained": 12.0,
                "return_yards": 0.0,
                "actor_is_home": False,
                "beneficiary_is_home": False,
            }
        ]
    )
    references = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "atomic_information_episode_id": "episode-1",
                "reference_status": "SUPPORTED",
                "pre_state_known_at": "2025-09-05T00:00:00Z",
                "post_state_known_at": "2025-09-05T00:00:01Z",
                "p_before_home": 0.48,
                "p_after_home": 0.52,
                "reference_delta_home": 0.04,
            }
        ]
    )
    factor_hits = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "event_id": "event-1",
                "play_id": "play-1",
                "factor_id": "NFL.PASS.COMPLETE",
                "factor_version": "v1",
                "registry_sha256": "sha256:" + "d" * 64,
                "pbp_source_sha256": "sha256:" + "c" * 64,
                "predicate_evidence": '{"primary_action":"PASS"}',
            }
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "trade_id": "home-L",
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-home",
                "source_time_utc": "2025-09-05T00:00:01.500Z",
                "price": 0.50,
                "size": 10.0,
                "kind": "trade",
                "provenance": "observed",
            },
            {
                "trade_id": "home-H5",
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-home",
                "source_time_utc": "2025-09-05T00:00:05.500Z",
                "price": 0.52,
                "size": 7.0,
                "kind": "trade",
                "provenance": "observed",
            },
            {
                "trade_id": "away-derived-L",
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-away-complement",
                "source_time_utc": "2025-09-05T00:00:01.500Z",
                "price": 0.90,
                "size": 0.0,
                "kind": "derived_complement",
                "provenance": "derived",
            },
            {
                "trade_id": "away-derived-H5",
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-away-complement",
                "source_time_utc": "2025-09-05T00:00:05.500Z",
                "price": 0.10,
                "size": 0.0,
                "kind": "derived_complement",
                "provenance": "derived",
            },
        ]
    )
    contracts = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-home",
                "outcome_team": "BBB",
                "home_team": "BBB",
                "market_family": "moneyline",
                "contract_role": "ACTUAL_HOME_OUTCOME",
                "tick_rule_id": "kalshi-cent",
            },
            {
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-away-complement",
                "outcome_team": "AAA",
                "home_team": "BBB",
                "market_family": "moneyline",
                "contract_role": "DERIVED_AWAY_COMPLEMENT",
                "tick_rule_id": "kalshi-cent",
            },
        ]
    )
    tick_rules = pd.DataFrame(
        [
            {
                "venue": "kalshi",
                "tick_rule_id": "kalshi-cent",
                "effective_start_utc": "2025-01-01T00:00:00Z",
                "effective_end_utc": "2026-01-01T00:00:00Z",
                "tick_size": 0.01,
            }
        ]
    )
    continuity = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "atomic_information_episode_id": "episode-1",
                "next_salient_event_time_utc": None,
                "suspension_time_utc": None,
                "game_end_time_utc": None,
                "continuity_gap_time_utc": None,
                "continuity_verified_until_utc": "2025-09-05T00:01:05Z",
            }
        ]
    )
    cohort_metadata = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "nfl_week": 1,
                "cohort": "development",
                "authority_sha256": "sha256:" + "e" * 64,
            }
        ]
    )
    canonical_mapping = [
        {
            "cohort": "development",
            "game_id": "2025_01_AAA_BBB",
            "nfl_week": 1,
        }
    ]
    mapping_sha256 = "sha256:" + hashlib.sha256(
        json.dumps(
            canonical_mapping,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "episode_facts": facts,
        "stage_a_references": references,
        "factor_hits": factor_hits,
        "market_rows": trades,
        "contract_metadata": contracts,
        "tick_rules": tick_rules,
        "continuity": continuity,
        "cohort_metadata": cohort_metadata,
        "expected_cohort_authority_sha256": "sha256:" + "e" * 64,
        "expected_cohort_mapping_sha256": mapping_sha256,
    }


def _build(**changes: object):
    inputs = _inputs()
    inputs.update(changes)
    builder = getattr(landmarks, "build_venue_reaction_panel_v3")
    return builder(**inputs)


def _row(panel: pd.DataFrame, landmark: int, endpoint: int) -> pd.Series:
    return panel.loc[
        panel["landmark_seconds"].eq(landmark)
        & panel["endpoint_seconds"].eq(endpoint)
    ].iloc[0]


def test_v3_grid_is_stable_unique_and_uses_only_actual_home_contract() -> None:
    inputs = _inputs()
    first = _build()
    shuffled = {
        name: (
            frame.sample(frac=1, random_state=7).reset_index(drop=True)
            if isinstance(frame, pd.DataFrame)
            else frame
        )
        for name, frame in inputs.items()
    }
    second = _build(**shuffled)

    pd.testing.assert_frame_equal(first.panel, second.panel)
    pd.testing.assert_frame_equal(first.attrition, second.attrition)
    assert len(first.panel) == 57
    grain = [
        "game_id",
        "atomic_information_episode_id",
        "venue",
        "actual_home_contract_id",
        "landmark_seconds",
        "endpoint_seconds",
    ]
    assert not first.panel.duplicated(grain).any()
    assert set(first.panel["actual_home_contract_id"]) == {"kalshi-home"}
    assert set(first.complement_diagnostics["contract_id"]) == {
        "kalshi-away-complement"
    }
    row = _row(first.panel, 1, 5)
    assert row["delta_l_h"] == pytest.approx(0.02)
    assert row["target_orientation"] == "ACTUAL_HOME_OUTCOME"
    assert bool(row["factor__NFL.PASS.COMPLETE"]) is True
    assert bool(row["event_tag__COMPLETE_PASS"]) is True
    assert "factor_id" not in first.panel.columns
    decision = row["decision_features_json"]
    assert "NFL.PASS.COMPLETE" in decision
    assert "COMPLETE_PASS" in decision
    assert "play-1" in decision
    assert "sha256:" + "d" * 64 in decision


def test_multi_event_factor_membership_is_order_invariant() -> None:
    inputs = _inputs()
    second_fact = inputs["episode_facts"].iloc[0].copy()
    second_fact["event_id"] = "event-2"
    second_fact["atomic_information_episode_id"] = "episode-2"
    second_fact["source_interval_start"] = "2025-09-05T00:00:10Z"
    second_fact["source_interval_end"] = "2025-09-05T00:00:11Z"
    second_fact["known_at"] = "2025-09-05T00:00:11Z"
    second_fact["primary_action"] = "RUSH"
    second_fact["outcome_tags"] = '["RUSH_ATTEMPT"]'
    inputs["episode_facts"] = pd.concat(
        [inputs["episode_facts"], second_fact.to_frame().T],
        ignore_index=True,
    )
    second_reference = inputs["stage_a_references"].iloc[0].copy()
    second_reference["atomic_information_episode_id"] = "episode-2"
    second_reference["pre_state_known_at"] = "2025-09-05T00:00:10Z"
    second_reference["post_state_known_at"] = "2025-09-05T00:00:11Z"
    inputs["stage_a_references"] = pd.concat(
        [inputs["stage_a_references"], second_reference.to_frame().T],
        ignore_index=True,
    )
    second_hit = inputs["factor_hits"].iloc[0].copy()
    second_hit["event_id"] = "event-2"
    second_hit["play_id"] = "play-2"
    second_hit["factor_id"] = "NFL.RUSH.ATTEMPT"
    inputs["factor_hits"] = pd.concat(
        [inputs["factor_hits"], second_hit.to_frame().T],
        ignore_index=True,
    )
    second_continuity = inputs["continuity"].iloc[0].copy()
    second_continuity["atomic_information_episode_id"] = "episode-2"
    inputs["continuity"] = pd.concat(
        [inputs["continuity"], second_continuity.to_frame().T],
        ignore_index=True,
    )

    first = _build(**inputs).panel
    shuffled = {
        name: (
            frame.sample(frac=1, random_state=11).reset_index(drop=True)
            if isinstance(frame, pd.DataFrame)
            else frame
        )
        for name, frame in inputs.items()
    }
    second = _build(**shuffled).panel

    pd.testing.assert_frame_equal(first, second)
    by_episode = first.loc[
        first["landmark_seconds"].eq(1)
        & first["endpoint_seconds"].eq(5)
    ].set_index("atomic_information_episode_id")
    assert bool(by_episode.loc["episode-1", "factor__NFL.PASS.COMPLETE"])
    assert not bool(by_episode.loc["episode-1", "factor__NFL.RUSH.ATTEMPT"])
    assert bool(by_episode.loc["episode-2", "factor__NFL.RUSH.ATTEMPT"])
    assert not bool(by_episode.loc["episode-2", "factor__NFL.PASS.COMPLETE"])


def test_panel_publishes_frozen_week_and_separate_factor_membership() -> None:
    metadata = _inputs()["cohort_metadata"].copy()
    result = _build(cohort_metadata=metadata)

    assert set(result.panel["nfl_week"]) == {1}
    assert set(result.panel["cohort_authority_sha256"]) == {"sha256:" + "e" * 64}
    assert len(result.factor_membership) == 1
    membership = result.factor_membership.iloc[0]
    assert membership["game_id"] == "2025_01_AAA_BBB"
    assert membership["atomic_information_episode_id"] == "episode-1"
    assert membership["factor_id"] == "NFL.PASS.COMPLETE"
    assert membership["factor_version"] == "v1"
    assert membership["registry_sha256"] == "sha256:" + "d" * 64
    assert membership["pbp_source_sha256"] == "sha256:" + "c" * 64
    assert "factor_id" not in result.panel.columns

    inconsistent = pd.concat([metadata, metadata.copy()], ignore_index=True)
    inconsistent.loc[1, "game_id"] = "2025_02_CCC_DDD"
    error = getattr(landmarks, "VenueReactionPanelError")
    with pytest.raises(error, match="game set"):
        _build(cohort_metadata=inconsistent)

    wrong_week = metadata.copy()
    wrong_week.loc[0, "nfl_week"] = 12
    with pytest.raises(error, match="mapping does not match frozen authority"):
        _build(cohort_metadata=wrong_week)

    wrong_authority = metadata.copy()
    wrong_authority.loc[0, "authority_sha256"] = "sha256:" + "f" * 64
    with pytest.raises(error, match="does not match frozen authority"):
        _build(cohort_metadata=wrong_authority)


def test_survival_censor_and_missing_observation_are_distinct_and_never_filled() -> None:
    missing = _inputs()["market_rows"].loc[
        lambda frame: ~frame["trade_id"].isin(["home-H5", "away-derived-H5"])
    ]
    missing_row = _row(_build(market_rows=missing).panel, 1, 5)
    assert bool(missing_row["s_h"]) is True
    assert bool(missing_row["o_h_given_s"]) is False
    assert pd.isna(missing_row["delta_l_h"])
    assert missing_row["direction"] == "UNOBSERVED"
    assert missing_row["attrition_reason"] == "ENDPOINT_STALE"

    continuity = _inputs()["continuity"].copy()
    continuity.loc[0, "next_salient_event_time_utc"] = (
        "2025-09-05T00:00:04Z"
    )
    censored_row = _row(_build(continuity=continuity).panel, 1, 5)
    assert bool(censored_row["s_h"]) is False
    assert pd.isna(censored_row["o_h_given_s"])
    assert pd.isna(censored_row["delta_l_h"])
    assert censored_row["attrition_reason"] == "NEXT_SALIENT_EVENT_BEFORE_H"


def test_interval_overlap_and_same_timestamp_are_ambiguous_but_t_plus_one_is_valid() -> None:
    rows = _inputs()["market_rows"].copy()
    rows.loc[rows["trade_id"].eq("home-L"), "source_time_utc"] = (
        "2025-09-05T00:00:00.800Z"
    )
    overlap = _row(_build(market_rows=rows).panel, 1, 5)
    assert overlap["attrition_reason"] == "LANDMARK_ORDER_AMBIGUOUS"
    assert pd.isna(overlap["mark_l_price"])

    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "trade_id": "home-at-interval-end",
                        "game_id": "2025_01_AAA_BBB",
                        "venue": "kalshi",
                        "contract_id": "kalshi-home",
                        "source_time_utc": "2025-09-05T00:00:01Z",
                        "price": 0.49,
                        "size": 2.0,
                        "kind": "trade",
                        "provenance": "observed",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    valid = _row(_build(market_rows=rows).panel, 1, 5)
    assert valid["mark_l_trade_id"] == "home-at-interval-end"
    assert valid["mark_l_price"] == pytest.approx(0.49)

    duplicate = rows.copy()
    endpoint = duplicate.loc[duplicate["trade_id"].eq("home-H5")].copy()
    endpoint["trade_id"] = "home-H5-duplicate"
    duplicate = pd.concat([duplicate, endpoint], ignore_index=True)
    ambiguous = _row(_build(market_rows=duplicate).panel, 1, 5)
    assert pd.isna(ambiguous["o_h_given_s"]) or not bool(
        ambiguous["o_h_given_s"]
    )
    assert ambiguous["attrition_reason"] == "ENDPOINT_ORDER_AMBIGUOUS"


def test_trade_index_preserves_mark_and_activity_boundaries() -> None:
    market = landmarks._validate_market(_inputs()["market_rows"])
    actual = market.loc[
        market["kind"].eq("trade") & market["provenance"].eq("observed")
    ]
    index = landmarks._trade_index(actual)
    start = pd.Timestamp("2025-09-05T00:00:00Z")
    end = pd.Timestamp("2025-09-05T00:00:01Z")
    h5 = pd.Timestamp("2025-09-05T00:00:05.500Z")

    mark = landmarks._select_mark(
        index,
        interval_start=start,
        interval_end=end,
        target_time=h5,
    )
    assert mark.status == "OBSERVED"
    assert mark.trade_id == "home-H5"
    assert not index.times_ns.flags.writeable
    assert landmarks._activity(index, landmark_time=h5, seconds=4) == (1, 7.0)

    duplicate = pd.concat([actual, actual.loc[actual["trade_id"].eq("home-H5")]])
    ambiguous = landmarks._select_mark(
        landmarks._trade_index(duplicate),
        interval_start=start,
        interval_end=end,
        target_time=h5,
    )
    assert ambiguous.status == "ORDER_AMBIGUOUS"
    assert ambiguous.source_time == h5

    overlap = actual.loc[actual["trade_id"].eq("home-L")].copy()
    overlap.loc[:, "source_time_utc"] = pd.Timestamp("2025-09-05T00:00:00.500Z")
    overlap_mark = landmarks._select_mark(
        landmarks._trade_index(overlap),
        interval_start=start,
        interval_end=end,
        target_time=pd.Timestamp("2025-09-05T00:00:02Z"),
    )
    assert overlap_mark.status == "ORDER_AMBIGUOUS"


def test_tick_defines_direction_without_fold_local_noise_labels() -> None:
    rows = _inputs()["market_rows"].copy()
    extra = []
    for trade_id, second, price in (
        ("home-H10", 10.5, 0.49),
        ("home-H15", 15.5, 0.509),
    ):
        extra.append(
            {
                "trade_id": trade_id,
                "game_id": "2025_01_AAA_BBB",
                "venue": "kalshi",
                "contract_id": "kalshi-home",
                "source_time_utc": f"2025-09-05T00:00:{second:04.1f}Z",
                "price": price,
                "size": 1.0,
                "kind": "trade",
                "provenance": "observed",
            }
        )
    rows = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)
    panel = _build(market_rows=rows).panel

    up = _row(panel, 1, 5)
    down = _row(panel, 1, 10)
    no_move = _row(panel, 1, 15)
    assert up["direction"] == "UP"
    assert up["conditional_magnitude"] == pytest.approx(0.02)
    assert "matched_control_p95" not in panel.columns
    assert "abnormal_move" not in panel.columns
    assert down["direction"] == "DOWN"
    assert down["conditional_magnitude"] == pytest.approx(0.01)
    assert no_move["direction"] == "NO_MOVE"
    assert pd.isna(no_move["conditional_magnitude"])


def test_endpoint_tamper_does_not_change_decision_features_or_hash() -> None:
    first = _row(_build().panel, 1, 5)
    rows = _inputs()["market_rows"].copy()
    rows.loc[rows["trade_id"].eq("home-H5"), "price"] = 0.80
    second = _row(_build(market_rows=rows).panel, 1, 5)

    assert first["delta_l_h"] != second["delta_l_h"]
    assert first["decision_feature_sha256"] == second["decision_feature_sha256"]
    assert first["decision_features_json"] == second["decision_features_json"]
    assert "mark_h" not in first["decision_features_json"].lower()
    assert "delta_l_h" not in first["decision_features_json"].lower()


def test_attrition_is_complete_and_continuity_is_mandatory() -> None:
    result = _build()
    assert {
        "venue",
        "landmark_seconds",
        "endpoint_seconds",
        "attrition_reason",
        "row_count",
    }.issubset(result.attrition.columns)
    assert int(result.attrition["row_count"].sum()) == len(result.panel)
    assert set(result.attrition["venue"]) == {"kalshi"}

    error = getattr(landmarks, "VenueReactionPanelError")
    with pytest.raises(error, match="continuity"):
        _build(continuity=pd.DataFrame())

    continuity = _inputs()["continuity"].copy()
    continuity.loc[0, "continuity_verified_until_utc"] = (
        "2025-09-05T00:00:04Z"
    )
    unverified = _row(_build(continuity=continuity).panel, 1, 5)
    assert pd.isna(unverified["s_h"])
    assert unverified["attrition_reason"] == "CONTINUITY_UNVERIFIED_BEFORE_H"


def test_real_stage_a_contract_allows_unsupported_nulls_and_gates_b4_at_l() -> None:
    unsupported = _inputs()["stage_a_references"].copy()
    unsupported.loc[0, "reference_status"] = "MODEL_SUPPORT_UNPROVEN"
    unsupported.loc[
        0,
        [
            "post_state_known_at",
            "p_before_home",
            "p_after_home",
            "reference_delta_home",
        ],
    ] = None
    unsupported_row = _row(
        _build(stage_a_references=unsupported).panel,
        1,
        5,
    )
    assert unsupported_row["reference_status"] == "MODEL_SUPPORT_UNPROVEN"
    assert unsupported_row["stage_a_status"] == "UNAVAILABLE"
    assert pd.isna(unsupported_row["p_before_home"])
    assert pd.isna(unsupported_row["p_after_home"])

    future = _inputs()["stage_a_references"].copy()
    future.loc[0, "post_state_known_at"] = "2025-09-05T00:00:02.500Z"
    future_row = _row(_build(stage_a_references=future).panel, 1, 5)
    assert future_row["reference_status"] == "SUPPORTED"
    assert future_row["stage_a_status"] == "NOT_KNOWN_AT_L"
    assert pd.isna(future_row["p_after_home"])
    assert future_row["post_state_known_at"] == pd.Timestamp(
        "2025-09-05T00:00:02.500Z"
    )
    assert "2025-09-05T00:00:02.500000+00:00" not in future_row[
        "decision_features_json"
    ]

    invalid = _inputs()["stage_a_references"].copy()
    invalid.loc[0, "p_after_home"] = None
    error = getattr(landmarks, "VenueReactionPanelError")
    with pytest.raises(error, match="SUPPORTED"):
        _build(stage_a_references=invalid)


def test_future_raw_reference_status_cannot_change_decision_features() -> None:
    rows = []
    for reference_status in (
        "SUPPORTED",
        "COMPOSITE_TRANSITION",
        "MISSING_POST_STATE",
    ):
        reference = _inputs()["stage_a_references"].copy()
        reference.loc[0, "reference_status"] = reference_status
        reference.loc[0, "post_state_known_at"] = (
            "2025-09-05T00:00:02.500Z"
        )
        if reference_status != "SUPPORTED":
            reference.loc[
                0,
                ["p_after_home", "reference_delta_home"],
            ] = None
        row = _row(_build(stage_a_references=reference).panel, 1, 5)
        assert row["reference_status"] == reference_status
        assert row["stage_a_status"] == "NOT_KNOWN_AT_L"
        rows.append(row)

    assert len({row["decision_features_json"] for row in rows}) == 1
    assert len({row["decision_feature_sha256"] for row in rows}) == 1
    assert all(
        "reference_status" not in row["decision_features_json"]
        for row in rows
    )


def test_non_stage_b_and_missing_interval_facts_are_audited_not_panelled() -> None:
    facts = _inputs()["episode_facts"].copy()
    excluded = facts.iloc[0].copy()
    excluded["event_id"] = "event-admin"
    excluded["atomic_information_episode_id"] = "episode-admin"
    excluded["stage_b_information_event_eligible"] = False
    excluded["source_interval_start"] = None
    excluded["source_interval_end"] = None
    excluded["known_at"] = None
    facts = pd.concat([facts, excluded.to_frame().T], ignore_index=True)

    result = _build(episode_facts=facts)

    assert len(result.panel) == 57
    assert set(result.panel["atomic_information_episode_id"]) == {"episode-1"}
    audit = result.fact_attrition.set_index("atomic_information_episode_id")
    assert audit.loc["episode-admin", "fact_attrition_reason"] == (
        "NOT_STAGE_B_INFORMATION_EVENT"
    )
    assert not bool(audit.loc["episode-admin", "included_in_panel"])


def test_factor_hits_must_link_and_match_source_and_registry_provenance() -> None:
    error = getattr(landmarks, "VenueReactionPanelError")
    unlinked = _inputs()["factor_hits"].copy()
    unlinked.loc[0, "event_id"] = "event-missing"
    with pytest.raises(error, match="event_id"):
        _build(factor_hits=unlinked)

    wrong_source = _inputs()["factor_hits"].copy()
    wrong_source.loc[0, "pbp_source_sha256"] = "sha256:" + "e" * 64
    with pytest.raises(error, match="source"):
        _build(factor_hits=wrong_source)

    bad_registry = _inputs()["factor_hits"].copy()
    bad_registry.loc[0, "registry_sha256"] = "not-a-sha"
    with pytest.raises(error, match="registry"):
        _build(factor_hits=bad_registry)


def test_exactly_one_actual_home_contract_is_required_per_game_venue() -> None:
    contracts = _inputs()["contract_metadata"].copy()
    duplicate = contracts.loc[
        contracts["contract_role"].eq("ACTUAL_HOME_OUTCOME")
    ].copy()
    duplicate["contract_id"] = "kalshi-home-duplicate"
    contracts = pd.concat([contracts, duplicate], ignore_index=True)

    error = getattr(landmarks, "VenueReactionPanelError")
    with pytest.raises(error, match="exactly one actual home"):
        _build(contract_metadata=contracts)


def test_censor_at_exact_h_is_explicitly_order_ambiguous_and_not_clean() -> None:
    continuity = _inputs()["continuity"].copy()
    continuity.loc[0, "next_salient_event_time_utc"] = (
        "2025-09-05T00:00:06Z"
    )

    row = _row(_build(continuity=continuity).panel, 1, 5)

    assert bool(row["s_h"]) is False
    assert pd.isna(row["o_h_given_s"])
    assert row["attrition_reason"] == (
        "NEXT_SALIENT_EVENT_AT_H_ORDER_AMBIGUOUS"
    )


@pytest.mark.parametrize(
    ("input_name", "column", "value"),
    (
        ("episode_facts", "source_interval_start", "2025-09-05T00:00:00"),
        ("episode_facts", "known_at", "2025-09-05T02:00:01+02:00"),
        ("stage_a_references", "pre_state_known_at", "2025-09-05T00:00:00"),
        ("market_rows", "source_time_utc", "2025-09-05T00:00:01.500"),
        ("market_rows", "source_time_utc", "2025-09-05T01:00:01.500+01:00"),
        ("tick_rules", "effective_start_utc", "2025-01-01T00:00:00"),
        ("continuity", "continuity_verified_until_utc", "2025-09-05T00:01:05"),
    ),
)
def test_all_supplied_times_require_explicit_utc_offset(
    input_name: str,
    column: str,
    value: str,
) -> None:
    changed = _inputs()[input_name].copy()
    changed.loc[0, column] = value

    error = getattr(landmarks, "VenueReactionPanelError")
    with pytest.raises(error, match="UTC"):
        _build(**{input_name: changed})
