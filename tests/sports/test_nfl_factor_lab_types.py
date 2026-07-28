from __future__ import annotations

from dataclasses import replace

import pytest

from prediction_market.sports.nfl_factor_lab_types import (
    BetweenPlayRosterChangeV1,
    FactorDefinitionV1,
    FactorLabValidationError,
    FactorObservationV1,
    FactorResultV1,
    FrontendReferenceV1,
    PredicateClauseV1,
    SportsFeatureRowV1,
    SportsFeatureViewV1,
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _row(**overrides: object) -> SportsFeatureRowV1:
    values: dict[str, object] = {
        "game_id": "2025_01_AAA_BBB",
        "play_id": "10",
        "episode_id": "episode-10",
        "order_sequence": 10,
        "season": 2025,
        "season_type": "REG",
        "week": 1,
        "kickoff_at_utc": "2025-09-01T17:00:00Z",
        "source_time_utc": "2025-09-01T17:10:00Z",
        "description": "Pass complete for 12 yards.",
        "home_team": "BBB",
        "away_team": "AAA",
        "focal_team": "AAA",
        "possession_team": "AAA",
        "offense_team": "AAA",
        "defense_team": "BBB",
        "quarter": 1,
        "quarter_seconds_remaining": 600,
        "half_seconds_remaining": 1500,
        "game_seconds_remaining": 3300,
        "home_score": 0,
        "away_score": 0,
        "score_margin_focal": 0,
        "tied": True,
        "one_score_game": True,
        "multi_score_game": False,
        "down": 1,
        "distance": 10,
        "yardline_100": 75,
        "field_zone": "own",
        "red_zone": False,
        "goal_to_go": False,
        "drive_id": "1",
        "drive_play_index": 1,
        "drive_yards_before": 0.0,
        "drive_progression_yards_before": 0.0,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "play_type": "pass",
        "event_flags": ("pass", "success"),
        "event_eligible": True,
        "episode_nullified": False,
        "episode_audit_only": False,
        "player_roles": (),
        "formation": None,
        "offense_personnel": None,
        "defense_personnel": None,
        "offense_player_ids": (),
        "defense_player_ids": (),
        "offense_player_names": (),
        "defense_player_names": (),
        "shotgun": True,
        "no_huddle": False,
        "yards_gained": 12.0,
        "wp_focal": 0.45,
        "vegas_wp_focal": 0.47,
        "wpa_focal": 0.02,
        "realized_delta_wp_focal": 0.018,
        "epa": 0.8,
        "qb_epa": 0.8,
        "xpass": 0.7,
        "pass_oe": 0.3,
        "cp": 0.6,
        "cpoe": 0.4,
        "yac_epa": 0.2,
        "xyac_epa": 0.1,
        "yac_surprise": 0.1,
        "rolling_features": (),
    }
    values.update(overrides)
    return SportsFeatureRowV1(**values)


def test_factor_definition_uses_canonical_structured_predicates() -> None:
    definition = FactorDefinitionV1(
        factor_id="distance_gt_10",
        version="v1",
        sports_meaning="Long-yardage state before the finalized episode.",
        predicate=(
            PredicateClauseV1(field="distance", operator="gt", value=10),
        ),
        pit_fields=("distance",),
        focal_entity="possession",
        exclusions=(
            PredicateClauseV1(
                field="event_eligible",
                operator="eq",
                value=False,
            ),
        ),
        target="winner.actual_trade_delta",
        market_families=("winner",),
        horizons_seconds=(10, 30, 60),
        support_kind="binary",
        minimum_episodes=40,
        minimum_games=8,
        minimum_episodes_per_venue=20,
    )

    assert definition.definition_sha256.startswith("sha256:")
    assert definition.definition_sha256 == replace(definition).definition_sha256
    assert definition.to_dict()["predicate"] == [
        {"field": "distance", "operator": "gt", "value": 10}
    ]


def test_factor_definition_rejects_noncanonical_or_unproved_turnover_factor() -> None:
    with pytest.raises(FactorLabValidationError, match="canonical order"):
        FactorDefinitionV1(
            factor_id="bad",
            version="v1",
            sports_meaning="Bad order.",
            predicate=(
                PredicateClauseV1(field="quarter", operator="eq", value=4),
                PredicateClauseV1(field="distance", operator="gt", value=10),
            ),
            pit_fields=("quarter", "distance"),
            focal_entity="team",
            exclusions=(),
            target="winner.actual_trade_delta",
            market_families=("winner",),
            horizons_seconds=(10,),
            support_kind="named",
            minimum_episodes=20,
            minimum_games=6,
        )

    with pytest.raises(FactorLabValidationError, match="turnover_over_expected"):
        FactorDefinitionV1(
            factor_id="turnover_over_expected",
            version="v1",
            sports_meaning="Unavailable PIT turnover model.",
            predicate=(
                PredicateClauseV1(field="interception", operator="eq", value=True),
            ),
            pit_fields=("interception",),
            focal_entity="team",
            exclusions=(),
            target="winner.actual_trade_delta",
            market_families=("winner",),
            horizons_seconds=(10,),
            support_kind="named",
            minimum_episodes=20,
            minimum_games=6,
        )


def test_predicate_clause_rejects_nonfinite_and_invalid_null_comparisons() -> None:
    with pytest.raises(FactorLabValidationError, match="finite"):
        PredicateClauseV1(field="epa", operator="gt", value=float("nan"))
    with pytest.raises(FactorLabValidationError, match="null operator"):
        PredicateClauseV1(field="epa", operator="is_null", value=0)


def test_feature_view_is_deterministic_and_rejects_duplicate_grain() -> None:
    first = _row(play_id="2", episode_id="episode-2", order_sequence=2)
    second = _row(play_id="1", episode_id="episode-1", order_sequence=1)
    hashes = (("nflverse_2025", _hash("a")), ("episodes", _hash("b")))

    view_a = SportsFeatureViewV1.build(rows=(first, second), source_hashes=hashes)
    view_b = SportsFeatureViewV1.build(rows=(second, first), source_hashes=hashes)

    assert [row.play_id for row in view_a.rows] == ["1", "2"]
    assert view_a.rows_sha256 == view_b.rows_sha256
    assert view_a.to_records()[0]["event_flags"] == ["pass", "success"]

    with pytest.raises(FactorLabValidationError, match="duplicate"):
        SportsFeatureViewV1.build(
            rows=(first, replace(first, order_sequence=3)),
            source_hashes=hashes,
        )


def test_feature_row_rejects_future_source_time_and_invalid_probability() -> None:
    with pytest.raises(FactorLabValidationError, match="source_time_utc"):
        _row(source_time_utc="2025-08-31T17:00:00Z")
    with pytest.raises(FactorLabValidationError, match="wp_focal"):
        _row(wp_focal=1.1)


def test_between_play_roster_change_and_structured_native_fields_serialize() -> None:
    roster_change = BetweenPlayRosterChangeV1(
        offense_entering_names=("New Receiver",),
        offense_leaving_names=("Old Receiver",),
        defense_entering_names=("New Corner",),
        defense_leaving_names=("Old Corner",),
    )
    row = _row(
        defensive_two_point_attempt=True,
        defensive_two_point_conversion=True,
        kicker_player_id="K1",
        kicker_player_name="Kicker One",
        penalty_team="BBB",
        penalty_type="Defensive Holding",
        penalty_yards=5.0,
        penalty_status="accepted",
        timeout_team="AAA",
        punt_blocked=True,
        kickoff_out_of_bounds=True,
        return_team="AAA",
        return_yards=12.0,
        fumble_recovery_team="AAA",
        fumble_recovery_yards=8.0,
        between_play_roster_change=roster_change,
    )

    record = row.to_dict()
    assert record["defensive_two_point_conversion"] is True
    assert record["kicker_player_name"] == "Kicker One"
    assert record["penalty_type"] == "Defensive Holding"
    assert record["timeout_team"] == "AAA"
    assert record["punt_blocked"] is True
    assert record["kickoff_out_of_bounds"] is True
    assert record["fumble_recovery_yards"] == 8.0
    assert record["between_play_roster_change"] == {
        "offense_entering_names": ["New Receiver"],
        "offense_leaving_names": ["Old Receiver"],
        "defense_entering_names": ["New Corner"],
        "defense_leaving_names": ["Old Corner"],
        "semantics": (
            "set difference between consecutive observed participation rows; "
            "timing is bounded between plays"
        ),
    }

    with pytest.raises(FactorLabValidationError, match="canonical order"):
        BetweenPlayRosterChangeV1(
            offense_entering_names=("Zulu", "Alpha"),
            offense_leaving_names=(),
            defense_entering_names=(),
            defense_leaving_names=(),
        )


def test_factor_observation_result_and_frontend_reference_are_frozen_contracts() -> None:
    source_hashes = (_hash("a"), _hash("b"))
    observation = FactorObservationV1(
        experiment_id="X-13",
        factor_id="distance_gt_10",
        factor_version="v1",
        factor_definition_sha256=_hash("c"),
        game_id="2025_14_DAL_DET",
        episode_id="episode-10",
        logical_market_id="poly:dal-det:winner",
        outcome="DAL",
        venue="polymarket",
        market_family="winner",
        horizon_seconds=10,
        event_source_time_utc="2025-12-05T02:00:00Z",
        first_post_trade_time_utc="2025-12-05T02:00:03Z",
        pre_event_actual_trade=0.45,
        first_post_event_trade=0.48,
        horizon_vwap=0.49,
        horizon_last_trade=0.50,
        signed_price_change=0.05,
        maximum_excursion=0.07,
        continuation_10_to_60=None,
        trade_count=4,
        volume=125.0,
        staleness_seconds=1.0,
        order_ambiguous=False,
        contaminated=False,
        source_hashes=source_hashes,
    )
    result = FactorResultV1(
        experiment_id="X-13",
        factor_id="distance_gt_10",
        factor_version="v1",
        factor_definition_sha256=_hash("c"),
        market_family="winner",
        horizon_seconds=10,
        estimate=0.025,
        ci_lower=0.005,
        ci_upper=0.045,
        game_count=20,
        episode_count=60,
        venue_episode_counts=(("kalshi", 40), ("polymarket", 45)),
        bootstrap_samples_requested=10_000,
        bootstrap_samples_valid=9_998,
        leave_one_game_out_same_sign_rate=0.9,
        max_single_game_contribution=0.2,
        bh_q_value=0.08,
        venue_sign_consistent=True,
        support_status="eligible",
        source_hashes=source_hashes,
    )
    reference = FrontendReferenceV1(
        game_id="2025_14_DAL_DET",
        episode_id="episode-10",
        logical_market_id="poly:dal-det:winner",
        delay_seconds=2,
        horizon_seconds=10,
    )

    assert observation.to_dict()["schema_version"] == "FactorObservationV1"
    assert result.to_dict()["venue_episode_counts"] == {
        "kalshi": 40,
        "polymarket": 45,
    }
    assert reference.to_query() == {
        "game": "2025_14_DAL_DET",
        "episode": "episode-10",
        "market": "poly:dal-det:winner",
        "delay_s": "2",
        "horizon_s": "10",
    }

    with pytest.raises(FactorLabValidationError, match="horizon_seconds"):
        replace(observation, horizon_seconds=3)
    with pytest.raises(FactorLabValidationError, match="CI"):
        replace(result, ci_lower=0.05)
    with pytest.raises(FactorLabValidationError, match="delay_seconds"):
        replace(reference, delay_seconds=4)
