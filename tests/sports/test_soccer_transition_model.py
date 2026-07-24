from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports import soccer_transition_model as transition
from prediction_market.sports.soccer_game_state import SoccerCard, SoccerGameState


def _state(
    *,
    clock_ms: int = 60 * 60 * 1_000,
    home_score: int = 2,
    away_score: int = 1,
    cards: tuple[SoccerCard, ...] = (),
) -> SoccerGameState:
    return SoccerGameState(
        game_id="game_statsbomb_1",
        home_team_id=10,
        away_team_id=20,
        lifecycle="in_play",
        sequence=12,
        period=2,
        clock_ms=clock_ms,
        period_clock_ms=max(0, clock_ms - 45 * 60 * 1_000),
        home_score=home_score,
        away_score=away_score,
        possession_id=8,
        possession_team_id=10,
        last_action="Pass",
        last_event_id="evt_" + "1" * 64,
        cards=cards,
    )


def test_transition_features_are_projected_from_current_reducer_state() -> None:
    state = _state(
        cards=(
            SoccerCard(
                sequence=3,
                team_id=10,
                player_id=101,
                card="Yellow Card",
            ),
            SoccerCard(
                sequence=4,
                team_id=10,
                player_id=101,
                card="Second Yellow",
            ),
            SoccerCard(
                sequence=5,
                team_id=10,
                player_id=101,
                card="Red Card",
            ),
            SoccerCard(
                sequence=6,
                team_id=20,
                player_id=201,
                card="Red Card",
            ),
        )
    )

    features = transition.extract_transition_features(state)

    assert features.home_team_id == 10
    assert features.away_team_id == 20
    assert features.elapsed_seconds == 3600
    assert features.second_half == 1
    assert features.home_score == 2
    assert features.away_score == 1
    assert features.home_dismissals == 1
    assert features.away_dismissals == 1
    assert features.home_score_difference == 1
    assert features.home_dismissal_difference == 0
    assert features.source_state_sha256 == state.state_sha256
    assert features.feature_sha256.startswith("sha256:")


def test_state_conditioned_distribution_changes_and_is_exactly_normalized() -> None:
    model = transition.DynamicIntensityModel(
        coefficients=(-0.20, -0.45, -0.90),
        l2_penalty=1.0,
        objective=12.0,
        iterations=8,
        optimizer_status="synthetic_fixture",
    )
    tied = transition.extract_transition_features(
        _state(home_score=0, away_score=0, cards=())
    )
    leading = replace(tied, home_score=1, home_score_difference=1)
    home_short = replace(
        tied,
        home_dismissals=1,
        home_dismissal_difference=1,
    )

    tied_distribution = transition.predict_transition_distribution(
        model,
        base_home_goals=1.6,
        base_away_goals=1.2,
        features=tied,
    )
    leading_distribution = transition.predict_transition_distribution(
        model,
        base_home_goals=1.6,
        base_away_goals=1.2,
        features=leading,
    )
    short_distribution = transition.predict_transition_distribution(
        model,
        base_home_goals=1.6,
        base_away_goals=1.2,
        features=home_short,
    )

    for distribution in (
        tied_distribution,
        leading_distribution,
        short_distribution,
    ):
        assert distribution.classes == ("home_goal", "away_goal", "no_goal")
        assert np.isclose(
            sum(distribution.probabilities),
            1.0,
            rtol=0.0,
            atol=1e-15,
        )
        assert all(0.0 <= value <= 1.0 for value in distribution.probabilities)
    assert leading_distribution.probabilities != tied_distribution.probabilities
    assert leading.feature_sha256 != tied.feature_sha256
    assert leading_distribution.source_feature_sha256 == leading.feature_sha256
    assert short_distribution.probability_home_goal < (
        tied_distribution.probability_home_goal
    )
    assert short_distribution.probability_away_goal > (
        tied_distribution.probability_away_goal
    )


def _training_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origin = pd.Timestamp("2015-08-01T12:00:00Z")
    for match_index in range(24):
        for interval_index in range(6):
            start = origin + pd.Timedelta(
                days=match_index,
                minutes=interval_index * 5,
            )
            score_difference = (-1, 0, 1)[interval_index % 3]
            dismissal_difference = 1 if interval_index == 5 else 0
            for side in ("home", "away"):
                oriented_score = (
                    score_difference if side == "home" else -score_difference
                )
                oriented_dismissal = (
                    dismissal_difference
                    if side == "home"
                    else -dismissal_difference
                )
                rows.append(
                    {
                        "match_id": 1000 + match_index,
                        "side": side,
                        "base_goals_per_90": 1.45 if side == "home" else 1.10,
                        "exposure_seconds": 300,
                        "goal_count": int(
                            oriented_score < 0
                            and oriented_dismissal <= 0
                            and (match_index + interval_index) % 3 == 0
                        ),
                        "second_half": int(interval_index >= 3),
                        "score_difference": oriented_score,
                        "dismissal_difference": oriented_dismissal,
                        "feature_available_at": start,
                        "label_available_at": start + pd.Timedelta(minutes=5),
                    }
                )
    return pd.DataFrame(rows)


def test_dynamic_intensity_fit_is_deterministic_and_pit_fail_closed() -> None:
    rows = _training_rows()
    evaluation_cutoff = pd.Timestamp("2015-09-01T00:00:00Z")

    first = transition.fit_dynamic_intensity(
        rows,
        evaluation_cutoff=evaluation_cutoff,
        held_out_match_ids=frozenset({99_001, 99_002}),
        l2_penalty=1.0,
        optimizer_max_iterations=200,
    )
    second = transition.fit_dynamic_intensity(
        rows.sample(frac=1.0, random_state=7),
        evaluation_cutoff=evaluation_cutoff,
        held_out_match_ids=frozenset({99_001, 99_002}),
        l2_penalty=1.0,
        optimizer_max_iterations=200,
    )

    assert first.coefficients == second.coefficients
    assert first.parameter_sha256 == second.parameter_sha256
    assert first.objective < first.initial_objective
    assert first.projected_gradient_inf_norm <= 1e-5
    assert first.coefficient_names == (
        "second_half",
        "score_difference",
        "dismissal_difference",
    )

    leaked = rows.copy()
    leaked.loc[0, "label_available_at"] = evaluation_cutoff
    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="strictly before evaluation_cutoff",
    ):
        transition.fit_dynamic_intensity(
            leaked,
            evaluation_cutoff=evaluation_cutoff,
            held_out_match_ids=frozenset({99_001, 99_002}),
            l2_penalty=1.0,
            optimizer_max_iterations=200,
        )

    same_match = rows.copy()
    same_match.loc[same_match["match_id"] == 1000, "match_id"] = 99_001
    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="held-out match",
    ):
        transition.fit_dynamic_intensity(
            same_match,
            evaluation_cutoff=evaluation_cutoff,
            held_out_match_ids=frozenset({99_001, 99_002}),
            l2_penalty=1.0,
            optimizer_max_iterations=200,
        )


def test_dynamic_intensity_rejects_unidentified_feature_design() -> None:
    rows = _training_rows()
    rows["second_half"] = 0
    rows["score_difference"] = 0
    rows["dismissal_difference"] = 0

    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="full column rank",
    ):
        transition.fit_dynamic_intensity(
            rows,
            evaluation_cutoff=pd.Timestamp("2015-09-01T00:00:00Z"),
            held_out_match_ids=frozenset({99_001}),
            l2_penalty=1.0,
            optimizer_max_iterations=200,
        )


def test_multiclass_temperature_is_normalized_and_softens_confidence() -> None:
    raw = np.asarray(
        (
            (0.80, 0.10, 0.10),
            (0.05, 0.90, 0.05),
        ),
        dtype=float,
    )

    calibrated = transition.apply_multiclass_temperature(
        raw,
        temperature=2.0,
    )

    expected = np.sqrt(raw) / np.sqrt(raw).sum(axis=1, keepdims=True)
    assert np.allclose(calibrated, expected, rtol=0.0, atol=1e-15)
    assert np.allclose(
        calibrated.sum(axis=1),
        np.ones(2),
        rtol=0.0,
        atol=1e-15,
    )
    assert calibrated[0, 0] < raw[0, 0]
    assert calibrated[1, 1] < raw[1, 1]


def test_temperature_fit_is_deterministic_and_improves_calibration_loss() -> None:
    targets: list[str] = []
    probabilities: list[tuple[float, float, float]] = []
    groups: list[int] = []
    for index in range(60):
        target_index = index % len(transition.TRANSITION_CLASSES)
        predicted_index = (
            target_index if index % 3 != 0 else (target_index + 1) % 3
        )
        probability = [0.01, 0.01, 0.01]
        probability[predicted_index] = 0.98
        targets.append(transition.TRANSITION_CLASSES[target_index])
        probabilities.append(tuple(probability))
        groups.append(10_000 + index)
    raw = np.asarray(probabilities, dtype=float)
    order = np.random.default_rng(7).permutation(len(targets))

    first = transition.fit_multiclass_temperature(
        np.asarray(targets, dtype=object),
        raw,
        groups=np.asarray(groups, dtype=object),
        minimum_matches=20,
        optimizer_max_iterations=200,
    )
    second = transition.fit_multiclass_temperature(
        np.asarray(targets, dtype=object)[order],
        raw[order],
        groups=np.asarray(groups, dtype=object)[order],
        minimum_matches=20,
        optimizer_max_iterations=200,
    )

    assert first.temperature > 1.0
    assert first.objective < first.initial_objective
    assert first.temperature == second.temperature
    assert first.parameter_sha256 == second.parameter_sha256
    assert first.calibration_match_count == 60
    assert first.calibration_observation_count == 60
    with pytest.raises(FrozenInstanceError):
        first.temperature = 1.0  # type: ignore[misc]


def test_temperature_fit_fails_closed_without_enough_disjoint_matches() -> None:
    targets = np.asarray(
        [
            transition.TRANSITION_CLASSES[index % 3]
            for index in range(19)
        ],
        dtype=object,
    )
    raw = np.tile(np.asarray((0.20, 0.20, 0.60), dtype=float), (19, 1))

    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="at least 20 calibration matches",
    ):
        transition.fit_multiclass_temperature(
            targets,
            raw,
            groups=np.arange(19),
            minimum_matches=20,
            optimizer_max_iterations=200,
        )


def test_transition_features_reject_terminal_or_pregame_state() -> None:
    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="started",
    ):
        transition.extract_transition_features(
            SoccerGameState(
                game_id="game_statsbomb_1",
                home_team_id=10,
                away_team_id=20,
                lifecycle="not_started",
            )
        )
    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="started",
    ):
        transition.extract_transition_features(
            SoccerGameState(
                game_id="game_statsbomb_1",
                home_team_id=10,
                away_team_id=20,
                lifecycle="not_started",
                sequence=1,
                period=1,
                possession_id=1,
                possession_team_id=10,
                last_action="Starting XI",
                last_event_id="evt_" + "1" * 64,
            )
        )
    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="terminal",
    ):
        transition.extract_transition_features(
            replace(
                _state(),
                lifecycle="finished",
                period_complete=True,
            )
        )


def test_transition_features_reject_future_cards_and_non_full_horizons() -> None:
    future_card = SoccerCard(
        sequence=13,
        team_id=10,
        player_id=101,
        card="Red Card",
    )
    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="future card",
    ):
        transition.extract_transition_features(_state(cards=(future_card,)))

    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="regulation",
    ):
        transition.extract_transition_features(
            replace(
                _state(),
                period=3,
                clock_ms=90 * 60 * 1_000,
                period_clock_ms=0,
            )
        )

    with pytest.raises(
        transition.SoccerTransitionModelError,
        match="full five-minute",
    ):
        transition.extract_transition_features(
            replace(
                _state(),
                clock_ms=(45 + 41) * 60 * 1_000,
                period_clock_ms=41 * 60 * 1_000,
            )
        )
