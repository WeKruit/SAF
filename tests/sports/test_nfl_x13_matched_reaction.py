from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports.nfl_x13_matched_reaction import (
    MatchedReactionError,
    MatchedReactionSpecV1,
    adapt_published_x13_factor_observations,
    build_matched_reaction_tables,
)


CLAIM = "PRELIMINARY_SOURCE_TIME_ONLY"
HASH = "sha256:" + "a" * 64


def _observation(
    observation_id: str,
    *,
    game_id: str = "2025_01_AAA_BBB",
    episode_id: str = "episode-treated",
    factor_id: str = "NFL.EVENT.TOUCHDOWN",
    factor_family: str = "scoring",
    venue: str = "polymarket",
    orientation: str = "AAA",
    horizon: int = 10,
    effect: float = 0.04,
    pre_probability: float = 0.50,
    relative_time: float = 900.0,
    score_margin: float = 0.0,
    staleness: float = 1.0,
    liquidity: float = 100.0,
    routine: bool = False,
    hit_factor_ids: tuple[str, ...] | None = None,
    delay_seconds: int = 0,
    maximum_excursion: float | None = None,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "factor_id": factor_id,
        "factor_version": "v1",
        "factor_family": factor_family,
        "game_id": game_id,
        "episode_id": episode_id,
        "logical_market_id": f"{venue}:{game_id}:winner",
        "venue": venue,
        "market_family": "moneyline",
        "outcome_orientation": orientation,
        "horizon_seconds": horizon,
        "delay_seconds": delay_seconds,
        "effect": effect,
        "maximum_excursion": (
            effect if maximum_excursion is None else maximum_excursion
        ),
        "pre_probability": pre_probability,
        "relative_time_seconds": relative_time,
        "score_margin": score_margin,
        "staleness_seconds": staleness,
        "liquidity": liquidity,
        "routine_control_eligible": routine,
        "hit_factor_ids": (
            (factor_id,) if hit_factor_ids is None else hit_factor_ids
        ),
        "claim_boundary": CLAIM,
        "source_hashes": (HASH,),
    }


def _routine_controls(
    *,
    count: int = 20,
    game_id: str = "2025_01_AAA_BBB",
    horizon: int = 10,
    effect: float = 0.01,
    venue: str = "polymarket",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _observation(
                f"control-{game_id}-{horizon}-{index:02d}",
                game_id=game_id,
                episode_id=f"routine-{index:02d}",
                factor_id="NFL.CONTROL.ROUTINE",
                factor_family="control",
                venue=venue,
                horizon=horizon,
                effect=effect,
                routine=True,
                hit_factor_ids=("NFL.EVENT.PASS",),
            )
            for index in range(count)
        ]
    )


def test_exact_scope_and_same_game_routine_controls_are_enforced() -> None:
    treated = pd.DataFrame([_observation("treated")])
    controls = pd.concat(
        [
            _routine_controls(count=20),
            _routine_controls(
                count=20,
                game_id="2025_02_CCC_DDD",
                effect=-0.20,
            ),
            _routine_controls(count=20, venue="kalshi", effect=-0.30),
        ],
        ignore_index=True,
    )

    result = build_matched_reaction_tables(
        treated,
        controls,
        spec=replace(
            MatchedReactionSpecV1(),
            bootstrap_samples=200,
        ),
    )

    matches = result.matched_controls
    assert len(matches) == 20
    assert set(matches["control_game_id"]) == {"2025_01_AAA_BBB"}
    assert set(matches["venue"]) == {"polymarket"}
    assert set(matches["market_family"]) == {"moneyline"}
    assert set(matches["outcome_orientation"]) == {"AAA"}
    assert matches["match_scope"].eq("SAME_GAME").all()
    assert matches["match_weight"].sum() == pytest.approx(1.0)
    assert matches["treated_source_hashes"].map(
        lambda value: tuple(value) == (HASH,)
    ).all()
    assert matches["control_source_hashes"].map(
        lambda value: tuple(value) == (HASH,)
    ).all()
    assert result.match_audit.iloc[0]["match_status"] == "MATCHED"
    assert matches["delay_seconds"].eq(0).all()
    assert matches["treated_maximum_excursion"].eq(0.04).all()
    assert matches["control_maximum_excursion"].eq(0.01).all()


def test_controls_exposed_to_the_treated_factor_are_excluded() -> None:
    treated = pd.DataFrame([_observation("treated")])
    eligible = _routine_controls(count=20)
    exposed = _routine_controls(count=20).assign(
        observation_id=lambda frame: "exposed-" + frame["observation_id"],
        hit_factor_ids=lambda frame: [
            ("NFL.EVENT.TOUCHDOWN", "NFL.EVENT.PASS")
            for _ in range(len(frame))
        ],
        effect=-0.50,
    )

    result = build_matched_reaction_tables(
        treated,
        pd.concat([eligible, exposed], ignore_index=True),
        spec=replace(MatchedReactionSpecV1(), bootstrap_samples=50),
    )

    assert len(result.matched_controls) == 20
    assert result.matched_controls["control_effect"].eq(0.01).all()
    assert result.matched_controls["control_hit_factor_ids"].map(
        lambda values: "NFL.EVENT.TOUCHDOWN" not in values
    ).all()
    assert result.match_audit.iloc[0]["factor_exposed_control_count"] == 20


def test_cross_game_fallback_and_minimum_control_gate_are_explicit() -> None:
    treated = pd.DataFrame([_observation("treated")])
    controls = pd.concat(
        [
            _routine_controls(count=9),
            _routine_controls(
                count=11,
                game_id="2025_02_CCC_DDD",
            ),
        ],
        ignore_index=True,
    )
    spec = replace(MatchedReactionSpecV1(), bootstrap_samples=200)

    fallback = build_matched_reaction_tables(
        treated,
        controls,
        spec=spec,
    )
    assert len(fallback.matched_controls) == 20
    assert fallback.matched_controls["match_scope"].eq(
        "CROSS_GAME_FALLBACK"
    ).all()

    unmatched = build_matched_reaction_tables(
        treated,
        controls.iloc[:9],
        spec=spec,
    )
    assert unmatched.matched_controls.empty
    assert unmatched.match_audit.iloc[0]["match_status"] == (
        "UNMATCHED_MIN_CONTROLS"
    )
    assert unmatched.cross_game_results.empty


def test_missing_matching_field_and_non_source_time_claim_fail_closed() -> None:
    treated = pd.DataFrame([_observation("treated")])
    controls = _routine_controls()

    with pytest.raises(
        MatchedReactionError,
        match="treated observations missing required columns: liquidity",
    ):
        build_matched_reaction_tables(
            treated.drop(columns=["liquidity"]),
            controls,
        )

    treated.loc[0, "claim_boundary"] = "REAL_LATENCY"
    with pytest.raises(
        MatchedReactionError,
        match="historical claim_boundary must be SOURCE_TIME_ONLY",
    ):
        build_matched_reaction_tables(treated, controls)


def test_balance_gate_marks_group_unmatched_when_smd_exceeds_limit() -> None:
    treated = pd.DataFrame(
        [
            _observation(
                "treated-a",
                game_id="2025_01_AAA_BBB",
                pre_probability=0.90,
            ),
            _observation(
                "treated-b",
                game_id="2025_02_CCC_DDD",
                episode_id="episode-treated-b",
                pre_probability=0.90,
            ),
        ]
    )
    controls = pd.concat(
        [
            _routine_controls(
                game_id="2025_01_AAA_BBB",
            ).assign(pre_probability=0.10),
            _routine_controls(
                game_id="2025_02_CCC_DDD",
            ).assign(pre_probability=0.10),
        ],
        ignore_index=True,
    )

    result = build_matched_reaction_tables(
        treated,
        controls,
        spec=replace(
            MatchedReactionSpecV1(),
            bootstrap_samples=200,
        ),
    )

    assert result.match_balance["maximum_absolute_smd"].gt(0.10).all()
    assert result.match_audit["match_status"].eq(
        "UNMATCHED_BALANCE"
    ).all()
    assert result.single_game_results.empty


def test_source_time_path_reports_noise_materiality_and_10_to_60_shape() -> None:
    treated = pd.DataFrame(
        [
            _observation(
                f"treated-{horizon}",
                horizon=horizon,
                effect=effect,
            )
            for horizon, effect in (
                (1, 0.010),
                (2, 0.012),
                (5, 0.018),
                (10, 0.030),
                (20, 0.040),
                (30, 0.045),
                (60, 0.020),
            )
        ]
    )
    controls = pd.concat(
        [
            _routine_controls(
                horizon=horizon,
                effect=0.005,
            )
            for horizon in (1, 2, 5, 10, 20, 30, 60)
        ],
        ignore_index=True,
    )

    result = build_matched_reaction_tables(
        treated,
        controls,
        spec=replace(
            MatchedReactionSpecV1(),
            bootstrap_samples=200,
        ),
    )

    path = result.source_time_paths.iloc[0]
    assert path["first_material_horizon_seconds"] == 5
    assert path["control_noise_floor"] == pytest.approx(0.005)
    assert path["matched_effect_10"] == pytest.approx(0.025)
    assert path["matched_effect_60"] == pytest.approx(0.015)
    assert path["continuation_10_to_60"] == pytest.approx(-0.010)
    assert path["path_shape_10_to_60"] == "REVERSAL"
    assert path["delay_seconds"] == 0
    assert path["maximum_excursion_10"] == pytest.approx(0.030)
    assert path["maximum_excursion_60"] == pytest.approx(0.020)
    assert path["claim_boundary"] == CLAIM


def test_game_first_inference_reports_bootstrap_loo_bh_and_venue_split() -> None:
    treated_rows: list[dict[str, object]] = []
    control_frames: list[pd.DataFrame] = []
    for game_index in range(4):
        game_id = f"2025_{game_index + 1:02d}_AAA_BBB"
        for venue in ("polymarket", "kalshi"):
            treated_rows.append(
                _observation(
                    f"treated-{game_id}-{venue}",
                    game_id=game_id,
                    episode_id=f"episode-{game_id}",
                    venue=venue,
                    effect=0.04 + game_index * 0.001,
                )
            )
            control_frames.append(
                _routine_controls(
                    game_id=game_id,
                    venue=venue,
                    effect=0.01,
                )
            )

    result = build_matched_reaction_tables(
        pd.DataFrame(treated_rows),
        pd.concat(control_frames, ignore_index=True),
        spec=replace(
            MatchedReactionSpecV1(),
            bootstrap_samples=500,
            bootstrap_seed=7,
        ),
    )

    assert len(result.single_game_results) == 8
    assert set(result.cross_game_results["venue"]) == {
        "kalshi",
        "polymarket",
    }
    assert result.cross_game_results["game_count"].eq(4).all()
    assert result.cross_game_results[
        "bootstrap_samples_valid"
    ].eq(500).all()
    assert result.cross_game_results["loo_sign_stability_rate"].eq(
        1.0
    ).all()
    assert result.cross_game_results[
        "maximum_single_game_contribution"
    ].between(0.0, 1.0).all()
    assert result.cross_game_results["bh_q_value"].between(
        0.0, 1.0
    ).all()
    assert not result.cross_game_results["analysis_status"].eq(
        "PROGRESSIVE_SOURCE_TIME_CANDIDATE"
    ).any()
    assert set(result.cross_game_results["analysis_status"]).issubset(
        {
            "SOURCE_TIME_ONLY_DESCRIPTIVE",
            "MATCHED_SOURCE_TIME_REVIEW",
        }
    )
    assert result.venue_consistency.iloc[0][
        "venue_sign_consistent"
    ] == np.bool_(True)
    assert result.manifest.iloc[0]["claim_boundary"] == CLAIM
    assert result.manifest["input_source_hash_count"].eq(1).all()
    assert result.manifest["input_source_hashes_sha256"].str.match(
        r"sha256:[0-9a-f]{64}"
    ).all()
    assert result.manifest["semantic_rows_sha256"].str.match(
        r"sha256:[0-9a-f]{64}"
    ).all()


def test_published_factor_adapter_reuses_rows_and_deduplicates_routine() -> None:
    base = {
        "factor_observation_id": "published-pass",
        "factor_id": "NFL.EVENT.PASS",
        "factor_version": "v2",
        "game_id": "2025_01_AAA_BBB",
        "episode_id": "episode-routine",
        "logical_market_id": "polymarket:2025_01_AAA_BBB:winner",
        "outcome": "AAA",
        "venue": "polymarket",
        "market_family": "moneyline",
        "horizon_seconds": 10,
        "delay_seconds": 0,
        "signed_price_change": 0.01,
        "maximum_excursion": 0.01,
        "pre_event_actual_trade": 0.55,
        "event_source_time_utc": "2025-09-01T00:01:00Z",
        "game_seconds_remaining": 1200,
        "score_margin_focal": 3,
        "staleness_seconds": 2,
        "event_eligible": True,
        "episode_audit_only": False,
        "episode_is_score": False,
        "episode_is_turnover": False,
        "penalty": False,
        "replay_or_challenge": False,
        "timeout": False,
        "safety": False,
        "punt_blocked": False,
        "return_touchdown": False,
        "fumble_lost": False,
        "interception": False,
        "fourth_down_failed": False,
        "claim_boundary": CLAIM,
        "sports_source_hashes": (HASH,),
        "market_source_hashes": ("sha256:" + "b" * 64,),
    }
    market = pd.DataFrame(
        [
            {
                "venue": "polymarket",
                "logical_market_id": (
                    "polymarket:2025_01_AAA_BBB:winner"
                ),
                "outcome": "AAA",
                "kind": "trade",
                "source_start_utc": timestamp,
                "size": size,
                "primary_path_eligible": True,
            }
            for timestamp, size in (
                ("2025-09-01T00:00:00Z", 10),
                ("2025-09-01T00:00:30Z", 20),
                ("2025-09-01T00:00:59Z", 30),
                ("2025-09-01T00:01:00Z", 40),
                ("2025-09-01T00:01:01Z", 100),
            )
        ]
    )
    duplicate_factor = {
        **base,
        "factor_observation_id": "published-distance",
        "factor_id": "NFL.STATE.DISTANCE_LONG",
    }
    salient = {
        **base,
        "factor_observation_id": "published-touchdown",
        "factor_id": "NFL.EVENT.TOUCHDOWN",
        "episode_id": "episode-score",
        "episode_is_score": True,
        "signed_price_change": 0.08,
    }

    adapted = adapt_published_x13_factor_observations(
        pd.DataFrame([base, duplicate_factor, salient]),
        market,
    )

    assert len(adapted.treated_observations) == 3
    assert len(adapted.routine_control_observations) == 1
    control = adapted.routine_control_observations.iloc[0]
    assert control["episode_id"] == "episode-routine"
    assert control["routine_control_eligible"] == np.bool_(True)
    assert control["pre_probability"] == pytest.approx(0.55)
    assert control["relative_time_seconds"] == pytest.approx(1200)
    assert control["liquidity"] == pytest.approx(60)
    assert control["outcome_orientation"] == "AAA"
    assert control["hit_factor_ids"] == (
        "NFL.EVENT.PASS",
        "NFL.STATE.DISTANCE_LONG",
    )
    assert control["delay_seconds"] == 0
    assert control["maximum_excursion"] == pytest.approx(0.01)
    assert set(control["source_hashes"]) == {
        HASH,
        "sha256:" + "b" * 64,
    }

    conflicting = pd.DataFrame(
        [base, {**duplicate_factor, "signed_price_change": 0.02}]
    )
    with pytest.raises(
        MatchedReactionError,
        match="routine duplicate rows disagree",
    ):
        adapt_published_x13_factor_observations(conflicting, market)

    with pytest.raises(
        MatchedReactionError,
        match="actual market observations missing required columns",
    ):
        adapt_published_x13_factor_observations(
            pd.DataFrame([base]),
            market.drop(columns=["source_start_utc"]),
        )
