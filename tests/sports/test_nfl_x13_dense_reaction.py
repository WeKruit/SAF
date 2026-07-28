from __future__ import annotations

import math

import pandas as pd
import pytest

from prediction_market.sports.nfl_x13_dense_reaction import (
    DENSE_HORIZONS_SECONDS,
    DenseMatchedReactionSpecV2,
    adapt_dense_paths_to_matched_observations_v2,
    build_dense_matched_effects_v2,
    build_dense_reaction_tables_v2,
)
from prediction_market.sports.nfl_x13_matched_reaction import HORIZONS_SECONDS


def _episode(episode_id: str = "episode-1") -> dict[str, object]:
    return {
        "game_id": "2025_01_AAA_BBB",
        "episode_id": episode_id,
        "game_seconds_remaining": 1800,
        "signed_margin": 3,
        "down": 2,
        "distance": 6,
        "field_position": 42,
        "possession": "AAA",
        "actor_beneficiary": "AAA",
    }


def _timeline(
    episode_id: str = "episode-1",
    *,
    order_ambiguous: bool = False,
    contaminated: bool = False,
) -> dict[str, object]:
    return {
        "game_id": "2025_01_AAA_BBB",
        "episode_id": episode_id,
        "source_time_start_utc": "2025-09-01T00:00:00Z",
        "source_time_end_utc": "2025-09-01T00:00:00Z",
        "order_ambiguous": order_ambiguous,
        "contaminated": contaminated,
        "source_coverage_end_utc": "2025-09-01T00:01:05Z",
    }


def _trade(
    observation_id: str,
    timestamp: str,
    price: float,
    *,
    coverage_end: str = "2025-09-01T00:01:05Z",
    contaminated: bool = False,
    order_ambiguous: bool = False,
) -> dict[str, object]:
    return {
        "game_id": "2025_01_AAA_BBB",
        "observation_id": observation_id,
        "logical_market_id": "polymarket:2025_01_AAA_BBB:winner",
        "outcome": "AAA",
        "venue": "polymarket",
        "market_family": "moneyline",
        "outcome_orientation": "AAA",
        "kind": "trade",
        "price": price,
        "size": 10.0,
        "source_start_utc": timestamp,
        "source_end_utc": timestamp,
        "primary_path_eligible": True,
        "source_coverage_end_utc": coverage_end,
        "contaminated": contaminated,
        "order_ambiguous": order_ambiguous,
    }


def test_dense_path_uses_actual_trade_bins_without_forward_fill() -> None:
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("post-1", "2025-09-01T00:00:01Z", 0.55),
                _trade("post-5", "2025-09-01T00:00:05Z", 0.60),
            ]
        ),
        pd.DataFrame([_timeline()]),
    )

    path = result.paths.set_index("horizon_seconds")
    assert tuple(DENSE_HORIZONS_SECONDS) == (
        1,
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
    )
    assert set(HORIZONS_SECONDS).issubset(DENSE_HORIZONS_SECONDS)
    assert path.loc[1, "cumulative_change"] == pytest.approx(0.05)
    assert path.loc[1, "incremental_change"] == pytest.approx(0.05)
    assert pd.isna(path.loc[2, "cumulative_change"])
    assert pd.isna(path.loc[2, "incremental_change"])
    assert path.loc[2, "attrition_reason"] == "NO_ACTUAL_TRADE_IN_BIN"
    assert path.loc[2, "forward_filled"] == False
    assert path.loc[5, "cumulative_change"] == pytest.approx(0.10)
    assert path.loc[5, "incremental_change"] == pytest.approx(0.05)
    assert path.loc[5, "source_time_first_trade"] == pd.Timestamp(
        "2025-09-01T00:00:01Z"
    )
    assert result.attrition["forward_filled"].eq(False).all()


def test_dense_path_surfaces_same_second_ambiguity_and_sparse_shape() -> None:
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("post-a", "2025-09-01T00:00:01Z", 0.55),
                _trade("post-b", "2025-09-01T00:00:01Z", 0.56),
            ]
        ),
        pd.DataFrame([_timeline()]),
    )

    assert result.paths["order_ambiguous"].eq(True).all()
    assert result.paths["attrition_reason"].eq("ORDER_AMBIGUOUS").all()
    assert result.path_summaries.iloc[0]["shape_classification"] == (
        "UNIDENTIFIED_SPARSE"
    )
    assert pd.isna(result.path_summaries.iloc[0]["jump"])


def test_dense_path_keeps_post_interval_after_boundary_trade() -> None:
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("within-event", "2025-09-01T00:00:00.500Z", 0.51),
                # The game-source uncertainty ends at 00:00:01Z.  A trade
                # exactly at that half-open interval boundary is post-event.
                _trade("post", "2025-09-01T00:00:01Z", 0.55),
            ]
        ),
        pd.DataFrame(
            [
                {
                    **_timeline(),
                    "source_time_end_utc": "2025-09-01T00:00:01Z",
                }
            ]
        ),
    )

    paths = result.paths.set_index("horizon_seconds")
    assert paths["order_ambiguous"].eq(False).all()
    assert paths.loc[1, "attrition_reason"] == "OBSERVED"
    assert paths.loc[1, "cumulative_change"] == pytest.approx(0.05)


def test_dense_path_classifies_a_jump_dominant_actual_trade_path() -> None:
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("post-1", "2025-09-01T00:00:01Z", 0.60),
                _trade("post-60", "2025-09-01T00:01:00Z", 0.61),
            ]
        ),
        pd.DataFrame([_timeline()]),
    )

    summary = result.path_summaries.iloc[0]
    assert summary["jump"] == pytest.approx(0.10)
    assert summary["jump_share"] == pytest.approx(0.10 / 0.11)
    assert summary["shape_classification"] == "JUMP_DOMINANT"


@pytest.mark.parametrize("field", ["contaminated", "order_ambiguous"])
def test_dense_path_propagates_flagged_post_trade_to_attrition(
    field: str,
) -> None:
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade(
                    "flagged-post",
                    "2025-09-01T00:00:01Z",
                    0.55,
                    **{field: True},
                ),
            ]
        ),
        pd.DataFrame([_timeline()]),
    )

    expected_reason = "CONTAMINATED" if field == "contaminated" else "ORDER_AMBIGUOUS"
    assert result.paths["observed"].eq(False).all()
    assert result.paths["attrition_reason"].eq(expected_reason).all()
    assert result.paths[field].eq(True).all()


def test_dense_path_censoring_forces_sparse_shape() -> None:
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade(
                    "pre",
                    "2025-08-31T23:59:59Z",
                    0.50,
                    coverage_end="2025-09-01T00:00:05Z",
                ),
                _trade(
                    "post-1",
                    "2025-09-01T00:00:01Z",
                    0.60,
                    coverage_end="2025-09-01T00:00:05Z",
                ),
                _trade(
                    "post-5",
                    "2025-09-01T00:00:05Z",
                    0.61,
                    coverage_end="2025-09-01T00:00:05Z",
                ),
            ]
        ),
        pd.DataFrame(
            [
                {
                    **_timeline(),
                    "source_coverage_end_utc": "2025-09-01T00:00:05Z",
                }
            ]
        ),
    )

    assert result.paths.loc[
        result.paths["horizon_seconds"].eq(10), "censored"
    ].item() == True
    assert result.path_summaries.iloc[0]["shape_classification"] == (
        "UNIDENTIFIED_SPARSE"
    )
    assert pd.isna(result.path_summaries.iloc[0]["jump"])


def test_dense_path_censors_each_horizon_after_the_next_salient_event() -> None:
    timeline = pd.DataFrame(
        [
            {
                **_timeline(),
                "next_salient_source_time_utc": "2025-09-01T00:00:03Z",
            }
        ]
    )
    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("post-1", "2025-09-01T00:00:01Z", 0.55),
                _trade("post-5", "2025-09-01T00:00:05Z", 0.60),
            ]
        ),
        timeline,
    )

    path = result.paths.set_index("horizon_seconds")
    assert path.loc[1, "observed"] == True
    assert path.loc[2, "observed"] == False
    assert path.loc[5, "contaminated"] == True
    assert path.loc[5, "attrition_reason"] == "CONTAMINATED"
    assert result.path_summaries.iloc[0]["shape_classification"] == (
        "UNIDENTIFIED_SPARSE"
    )


def test_dense_path_reads_sealed_market_columns_and_m_timeline_identity() -> None:
    actual = pd.DataFrame(
        [
            {
                **{
                    key: value
                    for key, value in _trade(
                        observation_id, timestamp, price
                    ).items()
                    if key
                    not in {
                        "logical_market_id",
                        "market_family",
                        "source_start_utc",
                        "source_end_utc",
                    }
                },
                "contract_id": "contract-winner",
                "family": "moneyline",
                "source_start": timestamp,
                "source_end": timestamp,
                "provenance": "observed",
            }
            for observation_id, timestamp, price in (
                ("pre", "2025-08-31T23:59:59Z", 0.50),
                ("post", "2025-09-01T00:00:01Z", 0.55),
            )
        ]
    )
    timeline = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "layer": "G",
                "identity": "episode-1",
                "interval_start_utc": "2025-09-01T00:00:00Z",
                "interval_end_utc": "2025-09-01T00:00:00Z",
                "delay_seconds": 0,
                "order_ambiguous": False,
            },
            *[
                {
                    "game_id": "2025_01_AAA_BBB",
                    "layer": "M",
                    "identity": observation_id,
                    "logical_market_id": "polymarket:2025_01_AAA_BBB:winner",
                    "delay_seconds": 0,
                }
                for observation_id in ("pre", "post")
            ],
        ]
    )

    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]), actual, timeline
    )

    assert len(result.paths) == len(DENSE_HORIZONS_SECONDS)
    assert set(result.paths["logical_market_id"]) == {
        "polymarket:2025_01_AAA_BBB:winner"
    }
    assert result.paths.set_index("horizon_seconds").loc[
        1, "cumulative_change"
    ] == pytest.approx(0.05)


def test_dense_path_derives_ambiguity_when_sealed_trade_rows_lack_flags() -> None:
    rows = [
        _trade("pre", "2025-08-31T23:59:59Z", 0.50),
        _trade("post-a", "2025-09-01T00:00:01Z", 0.55),
        _trade("post-b", "2025-09-01T00:00:01Z", 0.56),
    ]
    for row in rows:
        row.pop("contaminated")
        row.pop("order_ambiguous")

    result = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(rows),
        pd.DataFrame([_timeline()]),
    )

    assert result.paths["order_ambiguous"].eq(True).all()
    assert result.paths["attrition_reason"].eq("ORDER_AMBIGUOUS").all()


def _match_observation(
    observation_id: str,
    *,
    effect: float,
    game_id: str = "2025_01_AAA_BBB",
    horizon: int = 10,
    control_eligible: bool = True,
    treated_eligible: bool = True,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "game_id": game_id,
        "episode_id": observation_id,
        "venue": "polymarket",
        "market_family": "moneyline",
        "outcome_orientation": "AAA",
        "horizon_seconds": horizon,
        "effect": effect,
        "pre_probability": 0.50,
        "game_time_seconds": 1800.0,
        "signed_margin": 3.0,
        "down": 2.0,
        "distance": 6.0,
        "field_position": 42.0,
        "possession": "AAA",
        "actor_beneficiary": "AAA",
        "staleness_seconds": 1.0,
        "liquidity": 100.0,
        "control_eligible": control_eligible,
        "treated_eligible": treated_eligible,
    }


def test_dense_matching_audit_has_exact_key_unmatched_reason() -> None:
    treated = pd.DataFrame([_match_observation("treated", effect=0.05)])
    controls = pd.DataFrame(
        [
            _match_observation(
                f"control-{index}",
                effect=0.01,
                game_id="2025_02_CCC_DDD",
            )
            for index in range(9)
        ]
    )

    result = build_dense_matched_effects_v2(
        treated,
        controls,
        spec=DenseMatchedReactionSpecV2(),
    )

    audit = result.match_audit.iloc[0]
    assert result.matched_effects.empty
    assert audit["match_status"] == "UNMATCHED"
    assert audit["unmatched_reason"] == "INSUFFICIENT_EXACT_CONTROLS"
    assert audit["exact_control_count"] == 9
    assert audit["selected_control_count"] == 0
    assert set(result.match_audit["exact_match_keys"].iloc[0]) == {
        "venue",
        "market_family",
        "outcome_orientation",
        "horizon_seconds",
    }


def test_dense_matching_rejects_an_ineligible_treated_observation() -> None:
    treated = pd.DataFrame(
        [
            _match_observation(
                "treated",
                effect=0.05,
                control_eligible=True,
                treated_eligible=False,
            )
        ]
    )
    controls = pd.DataFrame(
        [
            _match_observation(
                f"control-{index}",
                effect=0.01,
                game_id="2025_02_CCC_DDD",
            )
            for index in range(10)
        ]
    )

    result = build_dense_matched_effects_v2(treated, controls)

    assert result.matched_controls.empty
    assert result.matched_effects.empty
    audit = result.match_audit.iloc[0]
    assert audit["match_status"] == "UNMATCHED_INELIGIBLE_TREATED"
    assert audit["unmatched_reason"] == "TREATED_NOT_ELIGIBLE"


def test_dense_matching_exactly_ten_controls_outputs_matched_effect_and_audit() -> None:
    treated = pd.DataFrame([_match_observation("treated", effect=0.05)])
    controls = pd.DataFrame(
        [
            _match_observation(
                f"control-{index}",
                effect=0.01,
                game_id="2025_02_CCC_DDD",
            )
            for index in range(10)
        ]
    )

    result = build_dense_matched_effects_v2(treated, controls)

    assert len(result.matched_controls) == 10
    assert result.match_audit.iloc[0]["match_status"] == "MATCHED"
    assert result.match_audit.iloc[0]["selected_control_count"] == 10
    effect = result.matched_effects.iloc[0]
    assert effect["matched_control_effect"] == pytest.approx(0.01)
    assert effect["matched_effect"] == pytest.approx(0.04)
    assert effect["maximum_absolute_smd"] == pytest.approx(0.0)


def test_dense_matching_caps_more_than_twenty_controls() -> None:
    treated = pd.DataFrame([_match_observation("treated", effect=0.05)])
    controls = pd.DataFrame(
        [
            _match_observation(
                f"control-{index:02d}", effect=0.01, game_id="2025_02_CCC_DDD"
            )
            for index in range(21)
        ]
    )

    result = build_dense_matched_effects_v2(treated, controls)

    assert len(result.matched_controls) == 20
    assert result.match_audit.iloc[0]["selected_control_count"] == 20


def test_dense_matching_rejects_smd_above_the_frozen_threshold() -> None:
    treated = pd.DataFrame([_match_observation("treated", effect=0.05)])
    controls = pd.DataFrame(
        [
            {
                **_match_observation(
                    f"control-{index}",
                    effect=0.01,
                    game_id="2025_02_CCC_DDD",
                ),
                "pre_probability": 0.10,
            }
            for index in range(10)
        ]
    )

    result = build_dense_matched_effects_v2(treated, controls)

    assert result.matched_effects.empty
    assert result.match_audit.iloc[0]["match_status"] == "UNMATCHED"
    assert result.match_audit.iloc[0]["unmatched_reason"] == "SMD_EXCEEDS_0_10"
    assert result.match_balance.iloc[0]["maximum_absolute_smd"] == math.inf


def test_dense_adapter_derives_matching_effect_and_provenance_from_sealed_paths(
) -> None:
    paths = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("post", "2025-09-01T00:00:01Z", 0.55),
            ]
        ),
        pd.DataFrame([_timeline()]),
    ).paths

    matched_input = adapt_dense_paths_to_matched_observations_v2(
        paths,
        routine_control_episode_ids=("episode-1",),
    )

    first = matched_input.set_index("horizon_seconds").loc[1]
    assert first["effect"] == pytest.approx(0.05)
    assert first["source_path_observation_id"] == paths.set_index(
        "horizon_seconds"
    ).loc[1, "observation_id"]
    assert first["effect_source_column"] == "cumulative_change"
    assert first["control_eligible"] == True


def test_dense_adapter_keeps_a_valid_nonroutine_treated_path_matchable() -> None:
    paths = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade("post", "2025-09-01T00:00:01Z", 0.55),
            ]
        ),
        pd.DataFrame([_timeline()]),
    ).paths
    treated = adapt_dense_paths_to_matched_observations_v2(
        paths,
        routine_control_episode_ids=(),
    )
    controls = pd.DataFrame(
        [
            {
                **_match_observation(
                    f"control-{index}",
                    effect=0.01,
                    game_id="2025_02_CCC_DDD",
                    horizon=1,
                ),
                "staleness_seconds": 0.0,
                "liquidity": 10.0,
            }
            for index in range(10)
        ]
    )

    assert len(treated) == len(paths)
    treated_horizon_one = treated[treated["horizon_seconds"].eq(1)]
    assert treated_horizon_one.iloc[0]["treated_eligible"] == True
    assert treated_horizon_one.iloc[0]["control_eligible"] == False
    result = build_dense_matched_effects_v2(treated_horizon_one, controls)
    assert result.match_audit.iloc[0]["match_status"] == "MATCHED"
    assert result.matched_effects.iloc[0]["matched_effect"] == pytest.approx(0.04)


def test_dense_adapter_keeps_nonadmissible_treated_path_in_audit() -> None:
    paths = build_dense_reaction_tables_v2(
        pd.DataFrame([_episode()]),
        pd.DataFrame(
            [
                _trade("pre", "2025-08-31T23:59:59Z", 0.50),
                _trade(
                    "flagged-post",
                    "2025-09-01T00:00:01Z",
                    0.55,
                    contaminated=True,
                ),
            ]
        ),
        pd.DataFrame([_timeline()]),
    ).paths
    treated = adapt_dense_paths_to_matched_observations_v2(
        paths,
        routine_control_episode_ids=(),
    )
    controls = pd.DataFrame(
        [
            _match_observation(
                f"control-{index}",
                effect=0.01,
                game_id="2025_02_CCC_DDD",
                horizon=1,
            )
            for index in range(10)
        ]
    )

    assert len(treated) == len(paths)
    treated_horizon_one = treated[treated["horizon_seconds"].eq(1)]
    assert treated_horizon_one.iloc[0]["treated_eligible"] == False
    assert treated_horizon_one.iloc[0]["control_eligible"] == False
    result = build_dense_matched_effects_v2(treated_horizon_one, controls)
    assert result.matched_controls.empty
    assert result.match_audit.iloc[0]["match_status"] == (
        "UNMATCHED_INELIGIBLE_TREATED"
    )
