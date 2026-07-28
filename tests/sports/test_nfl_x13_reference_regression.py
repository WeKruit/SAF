from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from prediction_market.models.nfl_fastrmodels import FEATURE_NAMES, NoSpreadModelInput
from prediction_market.sports.nfl_x13_reference_regression import (
    FinalizedEpisodeReferenceInputV2,
    NO_SPREAD_FEATURE_ALLOWLIST_V2,
    NoSpreadFeatureAsOfV2,
    NoSpreadFeatureStateV2,
    PinnedNoSpreadXGBoostPredictorV2,
    ReferenceMarketProjectionInputV2,
    ReferenceRegressionError,
    build_no_spread_reference_observations,
    build_reference_market_projection_inputs,
    fit_reference_local_projection_regressions,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _predictor(*, outcome: str = "focal") -> PinnedNoSpreadXGBoostPredictorV2:
    return PinnedNoSpreadXGBoostPredictorV2(
        asset_id="nflfastr-no-spread-xgboost",
        asset_version="exact153",
        asset_sha256=_sha("a"),
        probability_outcome=outcome,
        predict_probability=lambda model_input: model_input.feature_values[
            FEATURE_NAMES.index("score_differential")
        ],
    )


def _state(
    probability: float,
    *,
    as_of_ns: int = 100,
) -> NoSpreadFeatureStateV2:
    game_seconds_remaining = 3_000.0
    score_differential = probability
    values = {
        "receive_2h_ko": 0.0,
        "home": 0.0,
        "half_seconds_remaining": 1_200.0,
        "game_seconds_remaining": game_seconds_remaining,
        "Diff_Time_Ratio": score_differential
        / math.exp(-4.0 * ((3_600.0 - game_seconds_remaining) / 3_600.0)),
        "score_differential": score_differential,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 65,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
    }
    return NoSpreadFeatureStateV2(
        features=tuple(
            NoSpreadFeatureAsOfV2(
                name=name,
                value=values[name],
                as_of_ns=as_of_ns,
                as_of_provenance_sha256=_sha("b"),
            )
            for name in NO_SPREAD_FEATURE_ALLOWLIST_V2
        )
    )


def _episode(
    *,
    game_id: str = "2025_01_AAA_BBB",
    episode_id: str = "episode-1",
    focal_team: str = "AAA",
    quarter: int = 1,
    p_before: float = 0.40,
    p_after: float | None = 0.55,
    receive_2h_ko_ns: int = 100,
    pre_state_time_ns: int = 200,
    post_state_time_ns: int = 300,
    nullified: bool = False,
    reversed: bool = False,
) -> FinalizedEpisodeReferenceInputV2:
    return FinalizedEpisodeReferenceInputV2(
        game_id=game_id,
        episode_id=episode_id,
        home_team="BBB",
        away_team="AAA",
        focal_team=focal_team,
        quarter=quarter,
        pre_state=_state(p_before),
        post_state=(
            None if p_after is None else _state(p_after)
        ),
        pre_state_time_ns=pre_state_time_ns,
        post_state_time_ns=post_state_time_ns,
        receive_2h_ko_ns=receive_2h_ko_ns,
        finalized=True,
        nullified=nullified,
        reversed=reversed,
    )


def _projection(
    observation,
    *,
    before: float,
    after: float,
    venue: str = "kalshi",
    horizon_seconds: int = 10,
    held_out: bool = False,
) -> ReferenceMarketProjectionInputV2:
    return ReferenceMarketProjectionInputV2(
        game_id=observation.game_id,
        episode_id=observation.episode_id,
        venue=venue,
        horizon_seconds=horizon_seconds,
        pre_market_probability=before,
        post_market_probability=after,
        target_team=observation.focal_team,
        target_outcome="win",
        held_out=held_out,
    )


def test_reference_delta_is_oriented_toward_the_focal_outcome() -> None:
    observation = build_no_spread_reference_observations(
        (_episode(p_before=0.70, p_after=0.55),),
        predictor=_predictor(outcome="home"),
    )[0]

    assert observation.p_before == pytest.approx(0.30)
    assert observation.p_after == pytest.approx(0.45)
    assert observation.reference_delta == pytest.approx(0.15)
    assert observation.pre_feature_payload["focal_team"] == "AAA"
    assert observation.post_feature_payload["focal_team"] == "AAA"
    assert observation.pre_feature_sha256.startswith("sha256:")
    assert observation.post_feature_sha256.startswith("sha256:")
    assert observation.support_status == "SUPPORTED"


def test_reference_uses_the_official_fastrmodels_feature_adapter_contract() -> None:
    received_inputs: list[NoSpreadModelInput] = []

    def fixture_predictor(model_input: NoSpreadModelInput) -> float:
        assert isinstance(model_input, NoSpreadModelInput)
        received_inputs.append(model_input)
        return model_input.feature_values[FEATURE_NAMES.index("score_differential")]

    episode = _episode(p_before=0.40, p_after=0.55)
    observation = build_no_spread_reference_observations(
        (episode,),
        predictor=replace(_predictor(), predict_probability=fixture_predictor),
    )[0]
    pre_values = observation.pre_feature_payload["features"]
    model_input = NoSpreadModelInput(
        feature_names=FEATURE_NAMES,
        feature_values=tuple(float(pre_values[name]) for name in FEATURE_NAMES),
        posteam="AAA",
        home_team="BBB",
        away_team="AAA",
        period=1,
    )

    assert NO_SPREAD_FEATURE_ALLOWLIST_V2 == FEATURE_NAMES
    assert observation.pre_feature_payload["feature_names"] == FEATURE_NAMES
    assert model_input.feature_names == FEATURE_NAMES
    assert observation.p_before == pytest.approx(0.40)
    assert all(item.feature_names == FEATURE_NAMES for item in received_inputs)


@pytest.mark.parametrize("quarter", (1, 2))
def test_q1_q2_future_receive_provenance_is_unsupported(quarter: int) -> None:
    observation = build_no_spread_reference_observations(
        (
            _episode(
                quarter=quarter,
                receive_2h_ko_ns=201,
                pre_state_time_ns=200,
            ),
        ),
        predictor=_predictor(),
    )[0]

    assert observation.support_status == "UNSUPPORTED_RECEIVE_2H_KO_AFTER_STATE"
    assert observation.reference_delta is None
    assert observation.exclusion_reason == "receive_2h_ko_after_state"


def test_overtime_is_excluded_from_formal_reference_estimation() -> None:
    observation = build_no_spread_reference_observations(
        (_episode(quarter=5),), predictor=_predictor()
    )[0]

    assert observation.support_status == "EXCLUDED_OVERTIME"
    assert observation.p_before is None
    assert observation.p_after is None
    assert observation.reference_delta is None


@pytest.mark.parametrize(
    ("episode", "status", "reason"),
    (
        (_episode(nullified=True), "EXCLUDED_NULLIFIED", "episode_nullified"),
        (_episode(reversed=True), "EXCLUDED_REVERSED", "episode_reversed"),
        (_episode(p_after=None), "EXCLUDED_MISSING_POST_STATE", "missing_post_state"),
    ),
)
def test_nonfinal_post_states_are_excluded_from_formal_estimation(
    episode: FinalizedEpisodeReferenceInputV2,
    status: str,
    reason: str,
) -> None:
    observation = build_no_spread_reference_observations(
        (episode,), predictor=_predictor()
    )[0]

    assert observation.support_status == status
    assert observation.exclusion_reason == reason
    assert observation.p_before is None
    assert observation.p_after is None
    assert observation.reference_delta is None


def _synthetic_projection_inputs():
    values = (
        ("game-1", 0.40, -0.20),
        ("game-2", 0.50, 0.10),
        ("game-3", 0.60, 0.20),
        ("game-4", 0.45, -0.10),
        ("game-5", 0.55, 0.15),
        ("game-6", 0.65, -0.05),
    )
    observations = build_no_spread_reference_observations(
        tuple(
            _episode(
                game_id=game_id,
                episode_id=f"episode-{index}",
                p_before=0.50,
                p_after=0.50 + reference_delta,
            )
            for index, (game_id, _before, reference_delta) in enumerate(values)
        ),
        predictor=_predictor(),
    )
    market = tuple(
        _projection(
            observation,
            before=before,
            after=before
            + 0.02
            + 0.75 * reference_delta
            + 0.30 * (before - 0.525),
            held_out=index == 5,
        )
        for index, (observation, (_game_id, before, reference_delta)) in enumerate(
            zip(observations, values, strict=True)
        )
    )
    return build_reference_market_projection_inputs(
        observations,
        market,
        materiality_threshold=0.01,
    )


def test_local_projection_coefficient_has_the_synthetic_reference_direction() -> None:
    result = fit_reference_local_projection_regressions(
        _synthetic_projection_inputs(),
        bootstrap_draw_count=30,
        bootstrap_seed=20260727,
    )[0]

    assert result.status == "ESTIMATED"
    assert result.beta_reference_delta == pytest.approx(0.75)
    assert result.robust_diagnostics.covariance_estimator == "CR1_GAME_CLUSTER"
    assert any(residual.held_out for residual in result.residuals)


def test_local_projection_accepts_varying_optional_control_values() -> None:
    inputs = tuple(
        replace(
            row,
            held_out=False,
            state_controls=(("score_margin", float(index)),),
            liquidity_controls=(("depth", float(index * index + 10)),),
        )
        for index, row in enumerate(_synthetic_projection_inputs())
    )

    result = fit_reference_local_projection_regressions(
        inputs,
        bootstrap_draw_count=10,
        bootstrap_seed=20260727,
    )[0]

    assert result.status == "ESTIMATED"
    assert tuple(name for name, _value in result.beta_optional_controls) == (
        "state.score_margin",
        "liquidity.depth",
    )


def test_game_cluster_bootstrap_draws_are_deterministic() -> None:
    first = fit_reference_local_projection_regressions(
        _synthetic_projection_inputs(),
        bootstrap_draw_count=30,
        bootstrap_seed=20260727,
    )[0]
    second = fit_reference_local_projection_regressions(
        _synthetic_projection_inputs(),
        bootstrap_draw_count=30,
        bootstrap_seed=20260727,
    )[0]

    assert first.bootstrap_cluster_unit == "game_id"
    assert first.bootstrap_draws == second.bootstrap_draws
    assert len(first.bootstrap_draws) == 30
    assert all(draw.beta_reference_delta is not None for draw in first.bootstrap_draws)
    assert all(
        len(draw.sampled_game_ids) == first.game_count
        and set(draw.sampled_game_ids).issubset(
            {residual.game_id for residual in first.residuals if not residual.held_out}
        )
        for draw in first.bootstrap_draws
    )


def test_completion_fraction_is_refused_for_an_immaterial_reference_delta() -> None:
    observation = build_no_spread_reference_observations(
        (_episode(p_before=0.50, p_after=0.51),), predictor=_predictor()
    )[0]
    projection = build_reference_market_projection_inputs(
        (observation,),
        (_projection(observation, before=0.50, after=0.55),),
        materiality_threshold=0.02,
    )[0]

    assert projection.reference_delta == pytest.approx(0.01)
    assert projection.completion_fraction is None
    assert projection.completion_fraction_reason == "reference_below_materiality"


def test_completion_fraction_is_refused_at_the_materiality_boundary() -> None:
    observation = build_no_spread_reference_observations(
        (_episode(p_before=0.50, p_after=0.625),), predictor=_predictor()
    )[0]
    projection = build_reference_market_projection_inputs(
        (observation,),
        (_projection(observation, before=0.50, after=0.55),),
        materiality_threshold=0.125,
    )[0]

    assert observation.reference_delta == pytest.approx(0.125)
    assert projection.completion_fraction is None
    assert projection.completion_fraction_reason == "reference_below_materiality"


@pytest.mark.parametrize(
    ("control_field", "control_name", "control_value"),
    (
        ("state_controls", "vegas_wp", 0.50),
        ("liquidity_controls", "vegas_wp_focal", 0.50),
        ("state_controls", "diagnostic_fair_delta", 0.02),
        ("liquidity_controls", "reference_delta", 0.02),
    ),
)
def test_formal_projection_rejects_precomputed_and_diagnostic_controls(
    control_field: str,
    control_name: str,
    control_value: float,
) -> None:
    observation = build_no_spread_reference_observations(
        (_episode(),), predictor=_predictor()
    )[0]
    market = replace(
        _projection(observation, before=0.50, after=0.55),
        **{control_field: {control_name: control_value}},
    )

    with pytest.raises(ReferenceRegressionError, match="formal control"):
        build_reference_market_projection_inputs(
            (observation,),
            (market,),
            materiality_threshold=0.01,
        )


@pytest.mark.parametrize(
    ("target_team", "target_outcome"),
    (("BBB", "win"), ("AAA", "loss")),
)
def test_market_target_and_outcome_must_match_reference_focal_orientation(
    target_team: str,
    target_outcome: str,
) -> None:
    observation = build_no_spread_reference_observations(
        (_episode(),), predictor=_predictor()
    )[0]
    market = replace(
        _projection(observation, before=0.50, after=0.55),
        target_team=target_team,
        target_outcome=target_outcome,
    )

    with pytest.raises(ReferenceRegressionError, match="focal orientation"):
        build_reference_market_projection_inputs(
            (observation,),
            (market,),
            materiality_threshold=0.01,
        )


@pytest.mark.parametrize(
    ("pre_state", "match"),
    (
        (
            NoSpreadFeatureStateV2(
                features=(
                    *_state(0.40).features,
                    NoSpreadFeatureAsOfV2(
                        name="final_home_score",
                        value=99,
                        as_of_ns=100,
                        as_of_provenance_sha256=_sha("b"),
                    ),
                )
            ),
            "exact frozen no-spread feature allowlist",
        ),
        (
            NoSpreadFeatureStateV2(
                features=(
                    replace(_state(0.40).features[0], as_of_ns=201),
                    *_state(0.40).features[1:],
                )
            ),
            "as-of provenance",
        ),
    ),
)
def test_reference_rejects_extra_or_non_asof_feature_state(
    pre_state: NoSpreadFeatureStateV2,
    match: str,
) -> None:
    episode = replace(_episode(), pre_state=pre_state)

    with pytest.raises(ReferenceRegressionError, match=match):
        build_no_spread_reference_observations((episode,), predictor=_predictor())


def test_predictor_receives_distinct_validated_inputs_and_cannot_mutate_audit_snapshot() -> None:
    received_inputs: list[NoSpreadModelInput] = []

    def strict_predictor(model_input: NoSpreadModelInput) -> float:
        assert isinstance(model_input, NoSpreadModelInput)
        received_inputs.append(model_input)
        return model_input.feature_values[FEATURE_NAMES.index("score_differential")]

    episode = _episode(p_before=0.40, p_after=0.55)
    baseline = build_no_spread_reference_observations(
        (episode,), predictor=_predictor()
    )[0]
    observation = build_no_spread_reference_observations(
        (episode,),
        predictor=replace(_predictor(), predict_probability=strict_predictor),
    )[0]

    assert observation.p_before == pytest.approx(0.40)
    assert observation.p_after == pytest.approx(0.55)
    assert observation.pre_feature_sha256 == baseline.pre_feature_sha256
    assert observation.post_feature_sha256 == baseline.post_feature_sha256
    assert observation.pre_feature_payload["features"]["score_differential"] == 0.40
    assert received_inputs[0] is not received_inputs[1]
    with pytest.raises(FrozenInstanceError):
        received_inputs[0].feature_values = ()
    with pytest.raises(TypeError):
        observation.pre_feature_payload["features"]["score_differential"] = 0.99


def test_local_projection_refuses_single_training_game_cluster() -> None:
    inputs = tuple(
        replace(row, game_id="single-game", held_out=False)
        for row in _synthetic_projection_inputs()
    )

    result = fit_reference_local_projection_regressions(
        inputs,
        bootstrap_draw_count=10,
        bootstrap_seed=20260727,
    )[0]

    assert result.status == "UNSUPPORTED"
    assert result.unsupported_reason == "fewer_than_two_training_game_clusters"
    assert result.beta_reference_delta is None
    assert result.bootstrap_draws == ()
