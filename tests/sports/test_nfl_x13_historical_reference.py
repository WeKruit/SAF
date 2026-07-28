from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.models.nfl_fastrmodels import NoSpreadModelInput
from prediction_market.sports.nfl_x13_historical_reference import (
    build_reference_summary,
    build_retrospective_reference,
    join_reference_to_market_paths,
)


class _Predictor:
    def predict_home(self, model_input: NoSpreadModelInput) -> float:
        score = model_input.feature_values[5]
        possession_probability = min(0.99, max(0.01, 0.50 + score / 100.0))
        return (
            possession_probability
            if model_input.feature_values[1] == 1.0
            else 1.0 - possession_probability
        )


def _features() -> pd.DataFrame:
    rows = [
        {
            "game_id": "2025_01_AAA_BBB",
            "play_id": "10",
            "episode_id": "ep-1",
            "order_sequence": 10,
            "quarter": 1,
            "game_seconds_remaining": 3500,
            "half_seconds_remaining": 1700,
            "home_team": "BBB",
            "away_team": "AAA",
            "focal_team": "AAA",
            "possession_team": "AAA",
            "home_score": 0,
            "away_score": 0,
            "down": 1,
            "distance": 10,
            "yardline_100": 70,
            "home_timeouts_remaining": 3,
            "away_timeouts_remaining": 3,
            "episode_nullified": False,
            "episode_audit_only": False,
            "source_time_utc": "2025-09-01T00:00:00Z",
        },
        {
            "game_id": "2025_01_AAA_BBB",
            "play_id": "20",
            "episode_id": "ep-2",
            "order_sequence": 20,
            "quarter": 1,
            "game_seconds_remaining": 3480,
            "half_seconds_remaining": 1680,
            "home_team": "BBB",
            "away_team": "AAA",
            "focal_team": "AAA",
            "possession_team": "AAA",
            "home_score": 7,
            "away_score": 0,
            "down": 1,
            "distance": 10,
            "yardline_100": 75,
            "home_timeouts_remaining": 3,
            "away_timeouts_remaining": 3,
            "episode_nullified": False,
            "episode_audit_only": False,
            "source_time_utc": "2025-09-01T00:00:20Z",
        },
        {
            "game_id": "2025_01_AAA_BBB",
            "play_id": "30",
            "episode_id": "ep-q3",
            "order_sequence": 30,
            "quarter": 3,
            "game_seconds_remaining": 1800,
            "half_seconds_remaining": 1800,
            "home_team": "BBB",
            "away_team": "AAA",
            "focal_team": "BBB",
            "possession_team": "BBB",
            "home_score": 3,
            "away_score": 7,
            "down": 1,
            "distance": 10,
            "yardline_100": 75,
            "home_timeouts_remaining": 3,
            "away_timeouts_remaining": 3,
            "episode_nullified": False,
            "episode_audit_only": False,
            "source_time_utc": "2025-09-01T01:30:00Z",
        },
        {
            "game_id": "2025_01_AAA_BBB",
            "play_id": "40",
            "episode_id": "ep-ot",
            "order_sequence": 40,
            "quarter": 5,
            "game_seconds_remaining": 0,
            "half_seconds_remaining": 0,
            "home_team": "BBB",
            "away_team": "AAA",
            "focal_team": "AAA",
            "possession_team": "AAA",
            "home_score": 7,
            "away_score": 7,
            "down": 1,
            "distance": 10,
            "yardline_100": 75,
            "home_timeouts_remaining": 2,
            "away_timeouts_remaining": 2,
            "episode_nullified": False,
            "episode_audit_only": False,
            "source_time_utc": "2025-09-01T02:00:00Z",
        },
    ]
    return pd.DataFrame(rows)


def test_retrospective_reference_records_model_inputs_and_explicit_pit_boundary() -> None:
    result = build_retrospective_reference(
        _features(),
        predictor=_Predictor(),
        model_sha256="sha256:" + "a" * 64,
        source_object_sha256="sha256:" + "b" * 64,
    )

    first = result[result["episode_id"].eq("ep-1")].iloc[0]
    assert first["support_status"] == "SUPPORTED_RETROSPECTIVE"
    assert first["pit_status"] == "RETROSPECTIVE_INFERRED_NOT_LIVE_PIT"
    assert first["second_half_receiver"] == "BBB"
    assert first["p_before_ref"] == pytest.approx(0.50)
    assert first["p_after_ref"] == pytest.approx(0.43)
    assert first["reference_delta"] == pytest.approx(-0.07)
    assert first["pre_feature_sha256"].startswith("sha256:")
    assert first["post_feature_sha256"].startswith("sha256:")

    overtime = result[result["episode_id"].eq("ep-ot")].iloc[0]
    assert overtime["support_status"] == "MODEL_SUPPORT_UNPROVEN"
    assert overtime["exclusion_reason"] == "overtime"


def test_reference_join_computes_gap_and_materiality_gated_completion() -> None:
    reference = build_retrospective_reference(
        _features(),
        predictor=_Predictor(),
        model_sha256="sha256:" + "a" * 64,
        source_object_sha256="sha256:" + "b" * 64,
    )
    market = pd.DataFrame(
        [
            {
                "observation_id": "obs-1",
                "factor_id": "NFL.EVENT.TEST",
                "game_id": "2025_01_AAA_BBB",
                "episode_id": "ep-1",
                "outcome": "AAA",
                "venue": "polymarket",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "effect": -0.035,
            }
        ]
    )

    joined = join_reference_to_market_paths(
        market,
        reference,
        materiality_threshold=0.01,
    )

    row = joined.iloc[0]
    assert row["reference_gap"] == pytest.approx(0.035)
    assert row["completion_fraction"] == pytest.approx(0.5)
    assert row["completion_status"] == "ELIGIBLE"

    summary = build_reference_summary(joined)
    summary_row = summary.iloc[0]
    assert summary_row["observed_n"] == 1
    assert summary_row["reference_delta_mean"] == pytest.approx(-0.07)
    assert summary_row["market_delta_mean"] == pytest.approx(-0.035)
    assert summary_row["completion_fraction_median"] == pytest.approx(0.5)
    assert summary_row["p_partial_0_to_1"] == pytest.approx(1.0)
