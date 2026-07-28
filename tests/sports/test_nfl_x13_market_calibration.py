from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from prediction_market.sports.nfl_x13_market_calibration import (
    CalibrationPublicationError,
    build_calibration_tables,
    build_forecast_observations,
    publish_calibration,
)


def _paths() -> pd.DataFrame:
    rows = [
        # Same pre-event trade repeated at two horizons: one forecast, not two ticks.
        {
            "game_id": "2025_01_A_H",
            "episode_id": "e1",
            "logical_market_id": "v:g1:winner",
            "outcome": "H",
            "venue": "polymarket",
            "market_family": "moneyline",
            "horizon_seconds": horizon,
            "pre_probability": 0.70,
            "pre_event_observation_id": "trade-h-1",
        }
        for horizon in (1, 5)
    ] + [
        {
            "game_id": "2025_01_A_H",
            "episode_id": "e1",
            "logical_market_id": "v:g1:winner",
            "outcome": "A",
            "venue": "polymarket",
            "market_family": "moneyline",
            "horizon_seconds": 5,
            "pre_probability": 0.30,
            "pre_event_observation_id": "trade-a-1",
        },
        {
            "game_id": "2025_02_C_D",
            "episode_id": "e2",
            "logical_market_id": "v:g2:winner",
            "outcome": "C",
            "venue": "kalshi",
            "market_family": "moneyline",
            "horizon_seconds": 5,
            "pre_probability": 0.40,
            "pre_event_observation_id": "trade-c-1",
        },
        {
            "game_id": "2025_02_C_D",
            "episode_id": "e2",
            "logical_market_id": "v:g2:winner",
            "outcome": "D",
            "venue": "kalshi",
            "market_family": "moneyline",
            "horizon_seconds": 5,
            "pre_probability": 0.60,
            "pre_event_observation_id": "trade-d-1",
        },
        # Non-winner paths are not calibration forecasts.
        {
            "game_id": "2025_02_C_D",
            "episode_id": "e2",
            "logical_market_id": "v:g2:total",
            "outcome": "OVER",
            "venue": "kalshi",
            "market_family": "total",
            "horizon_seconds": 5,
            "pre_probability": 0.55,
            "pre_event_observation_id": "trade-over-1",
        },
    ]
    return pd.DataFrame(rows)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_H",
                "home_team": "H",
                "away_team": "A",
                "home_score": 24,
                "away_score": 17,
            },
            {
                "game_id": "2025_01_A_H",
                "home_team": "H",
                "away_team": "A",
                "home_score": 10,
                "away_score": 17,
            },
            {
                "game_id": "2025_02_C_D",
                "home_team": "D",
                "away_team": "C",
                "home_score": 20,
                "away_score": 21,
            },
        ]
    )


def test_build_forecast_observations_deduplicates_horizons_and_proves_orientation() -> None:
    observations, exclusions = build_forecast_observations(
        reaction_paths=_paths(),
        feature_rows=_features(),
        feature_object_sha256s={
            "2025_01_A_H": "sha256:" + "1" * 64,
            "2025_02_C_D": "sha256:" + "2" * 64,
        },
    )

    assert len(observations) == 4
    assert observations["pre_event_observation_id"].is_unique
    assert set(observations["orientation_proof"]) == {
        "OUTCOME_EQUALS_VERIFIED_HOME_TEAM",
        "OUTCOME_EQUALS_VERIFIED_AWAY_TEAM",
    }
    assert dict(
        zip(
            observations["outcome"],
            observations["realized_label"],
            strict=True,
        )
    ) == {"H": 1, "A": 0, "C": 1, "D": 0}
    assert exclusions.to_dict("records") == [
        {"reason": "NON_WINNER_MARKET_FAMILY", "row_count": 1}
    ]


def test_tied_game_and_unknown_outcome_fail_closed_from_binary_labels() -> None:
    tied = _features().copy()
    tied.loc[tied["game_id"] == "2025_01_A_H", ["home_score", "away_score"]] = 17
    observations, exclusions = build_forecast_observations(
        reaction_paths=_paths(),
        feature_rows=tied,
        feature_object_sha256s={
            "2025_01_A_H": "sha256:" + "1" * 64,
            "2025_02_C_D": "sha256:" + "2" * 64,
        },
    )
    assert set(observations["game_id"]) == {"2025_02_C_D"}
    assert {
        row["reason"]: row["row_count"]
        for row in exclusions.to_dict("records")
    } == {
        "BINARY_LABEL_UNDEFINED_TIE": 2,
        "NON_WINNER_MARKET_FAMILY": 1,
    }

    bad = _paths()
    bad.loc[0, "outcome"] = "ZZZ"
    with pytest.raises(CalibrationPublicationError, match="outcome orientation"):
        build_forecast_observations(
            reaction_paths=bad,
            feature_rows=_features(),
            feature_object_sha256s={
                "2025_01_A_H": "sha256:" + "1" * 64,
                "2025_02_C_D": "sha256:" + "2" * 64,
            },
        )


def test_calibration_uses_game_cluster_bootstrap_and_marks_popularity_gap() -> None:
    observations, _ = build_forecast_observations(
        reaction_paths=_paths(),
        feature_rows=_features(),
        feature_object_sha256s={
            "2025_01_A_H": "sha256:" + "1" * 64,
            "2025_02_C_D": "sha256:" + "2" * 64,
        },
    )
    tables = build_calibration_tables(
        observations,
        bootstrap_samples=200,
        bootstrap_seed=20260727,
        reliability_bin_count=5,
    )

    assert set(tables) == {
        "metrics",
        "reliability",
        "bootstrap",
        "bias_proxies",
    }
    overall = tables["metrics"].query("scope == 'overall'").iloc[0]
    assert overall["cluster_unit"] == "game_id"
    assert overall["game_count"] == 2
    assert overall["forecast_count"] == 4
    assert overall["bootstrap_samples_requested"] == 200
    assert 0 <= overall["brier_score"] <= 1
    assert overall["brier_ci_low"] <= overall["brier_score"] <= overall["brier_ci_high"]
    assert len(tables["bootstrap"].query("scope == 'overall'")) == 200
    popularity = tables["bias_proxies"].query(
        "proxy_family == 'popularity'"
    ).iloc[0]
    assert popularity["status"] == "DATA_GAP"
    assert popularity["claim_boundary"] == "NO_RELIABLE_POPULARITY_SOURCE_NOT_GUESSED"


def test_content_addressed_publication_is_deterministic_and_holdout_closed(
    tmp_path: Path,
) -> None:
    observations, exclusions = build_forecast_observations(
        reaction_paths=_paths(),
        feature_rows=_features(),
        feature_object_sha256s={
            "2025_01_A_H": "sha256:" + "1" * 64,
            "2025_02_C_D": "sha256:" + "2" * 64,
        },
    )
    tables = build_calibration_tables(
        observations,
        bootstrap_samples=25,
        bootstrap_seed=17,
        reliability_bin_count=5,
    )
    inputs = {
        "market_batch_index_sha256": "sha256:" + "a" * 64,
        "market_batch_sha256": "sha256:" + "b" * 64,
        "dense_manifest_sha256": "sha256:" + "c" * 64,
        "dense_bundle_sha256": "sha256:" + "d" * 64,
        "feature_batch_index_sha256": "sha256:" + "e" * 64,
        "feature_batch_sha256": "sha256:" + "f" * 64,
        "sports_clean_batch_index_sha256": "sha256:" + "0" * 64,
        "sports_clean_batch_sha256": "sha256:" + "9" * 64,
    }

    first = publish_calibration(
        output_root=tmp_path,
        observations=observations,
        exclusions=exclusions,
        tables=tables,
        input_hashes=inputs,
        expected_game_count=2,
        bootstrap_samples=25,
        bootstrap_seed=17,
    )
    second = publish_calibration(
        output_root=tmp_path,
        observations=observations,
        exclusions=exclusions,
        tables=tables,
        input_hashes=inputs,
        expected_game_count=2,
        bootstrap_samples=25,
        bootstrap_seed=17,
    )
    assert first == second

    payload = first.manifest_path.read_bytes()
    assert "sha256:" + hashlib.sha256(payload).hexdigest() == first.manifest_sha256
    manifest = json.loads(payload)
    assert manifest["holdout_reaction_access"] == "CLOSED"
    assert manifest["holdout_reaction_rows_read"] == 0
    assert manifest["claim_boundary"].startswith("RETROSPECTIVE")
    assert manifest["input_hashes"] == inputs
    for reference in manifest["objects"].values():
        path = tmp_path / reference["object_path"]
        assert path.is_file()
        assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == reference[
            "object_sha256"
        ]
        if path.suffix == ".parquet":
            assert pq.read_table(path).num_rows == reference["row_count"]
