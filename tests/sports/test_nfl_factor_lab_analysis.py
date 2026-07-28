from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports.nfl_factor_lab_analysis import (
    FactorLabError,
    HumanFactorReviewV1,
    build_factor_observations,
    freeze_factor_shortlist,
    run_factor_discovery,
    run_locked_historical_holdout,
    verify_shortlist_lock,
)


def _feature_rows(*, game_prefix: str, games: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_index in range(games):
        game_id = f"{game_prefix}_{game_index:02d}_AAA_BBB"
        for episode_index in range(30):
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": f"{game_id}:play:{episode_index}",
                    "episode_id": f"{game_id}:episode:{episode_index}",
                    "source_time_utc": (
                        f"2025-09-{game_index + 1:02d}T00:00:"
                        f"{episode_index:02d}Z"
                    ),
                    "event_interception": episode_index % 2 == 0,
                    "event_eligible": True,
                    "pre_distance": 12 if episode_index % 2 == 0 else 5,
                    "realized_delta_wp_focal": (
                        0.08 if episode_index % 2 == 0 else -0.01
                    ),
                    "feature_view_sha256": "sha256:" + "1" * 64,
                }
            )
    return pd.DataFrame(rows)


def _market_rows(*, game_prefix: str, games: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_index in range(games):
        game_id = f"{game_prefix}_{game_index:02d}_AAA_BBB"
        for episode_index in range(30):
            episode_id = f"{game_id}:episode:{episode_index}"
            positive = episode_index % 2 == 0
            for venue, venue_offset in (
                ("kalshi", 0.0002),
                ("polymarket", -0.0002),
            ):
                for horizon in (1, 2, 5, 10, 30, 60):
                    delta = (0.012 if positive else -0.002) + venue_offset
                    rows.append(
                        {
                            "game_id": game_id,
                            "episode_id": episode_id,
                            "logical_market_id": f"{game_id}:winner",
                            "contract_id": f"{venue}:{game_id}:AAA",
                            "outcome": "AAA",
                            "signal_focal_team": "AAA",
                            "venue": venue,
                            "family": "moneyline",
                            "contract_measure": "winner",
                            "horizon_seconds": horizon,
                            "pre_event_actual_trade": 0.50,
                            "first_post_event_trade": 0.50 + delta,
                            "signed_price_change": delta,
                            "maximum_excursion": abs(delta) + 0.002,
                            "net_change_60s": delta,
                            "trade_count": 2,
                            "volume": 10.0,
                            "staleness_seconds": 1.0,
                            "validity_status": "OBSERVED",
                            "actual_observation": True,
                            "forward_filled": False,
                            "orientation_audited": True,
                            "order_ambiguous": False,
                            "contaminated": False,
                            "signal_candidate_eligible": True,
                        }
                    )
    return pd.DataFrame(rows)


def _factor_registry() -> list[dict[str, object]]:
    return [
        {
            "factor_id": "NFL.INT.LONG_DISTANCE",
            "version": "v1",
            "sports_meaning": "Interception on a snap with more than ten yards to go.",
            "predicate": [
                {
                    "field": "event_interception",
                    "operator": "eq",
                    "value": True,
                },
                {"field": "pre_distance", "operator": "gt", "value": 10},
            ],
            "pit_fields": ["event_interception", "pre_distance"],
            "focal_entity": "team",
            "exclusions": [],
            "target": "signed_price_change",
            "support_kind": "primary",
            "market_families": ["moneyline"],
            "horizons_seconds": [10],
            "minimum_episodes": 300,
            "minimum_games": 16,
            "minimum_episodes_per_venue": 100,
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        }
    ]


def test_factor_observations_are_actual_trade_only_and_oriented() -> None:
    features = _feature_rows(game_prefix="2025_DISC", games=1)
    market = _market_rows(game_prefix="2025_DISC", games=1)
    invalid = market.iloc[[0]].copy()
    invalid["actual_observation"] = False
    invalid["forward_filled"] = True
    invalid["signed_price_change"] = 0.99
    observations = build_factor_observations(
        features,
        pd.concat([market, invalid], ignore_index=True),
        _factor_registry(),
    )

    assert set(observations["factor_id"]) == {"NFL.INT.LONG_DISTANCE"}
    assert set(observations["horizon_seconds"]) == {10}
    assert set(observations["venue"]) == {"kalshi", "polymarket"}
    assert observations["actual_observation"].all()
    assert not observations["forward_filled"].any()
    assert not observations["order_ambiguous"].any()
    assert not observations["contaminated"].any()
    assert observations["outcome"].eq(observations["signal_focal_team"]).all()


def test_discovery_requires_expert_review_before_shortlist(tmp_path: Path) -> None:
    observations = build_factor_observations(
        _feature_rows(game_prefix="2025_DISC"),
        _market_rows(game_prefix="2025_DISC"),
        _factor_registry(),
    )
    results = run_factor_discovery(
        observations,
        factor_registry=_factor_registry(),
        bootstrap_samples=10_000,
        seed=20260725,
    )

    row = results.iloc[0]
    assert row["factor_id"] == "NFL.INT.LONG_DISTANCE"
    assert row["game_count"] == 20
    assert row["episode_count"] == 300
    assert row["venue_sign_consistent"]
    assert row["bootstrap_samples_requested"] == 10_000
    assert row["bootstrap_samples_valid"] == 10_000
    assert row["discovery_status"] in {
        "DESCRIPTIVE",
        "AWAITING_EXPERT_REVIEW",
    }
    assert row["discovery_status"] != "SIGNAL_CANDIDATE"

    with pytest.raises(FactorLabError, match="expert review"):
        freeze_factor_shortlist(
            discovery_results=results,
            factor_registry=_factor_registry(),
            reviews=[],
            source_hashes=["sha256:" + "2" * 64],
            output_root=tmp_path,
        )


def test_shortlist_is_content_addressed_and_holdout_is_locked(
    tmp_path: Path,
) -> None:
    discovery_observations = build_factor_observations(
        _feature_rows(game_prefix="2025_DISC"),
        _market_rows(game_prefix="2025_DISC"),
        _factor_registry(),
    )
    discovery = run_factor_discovery(
        discovery_observations,
        factor_registry=_factor_registry(),
        bootstrap_samples=10_000,
        seed=20260725,
    )
    reviews = [
        HumanFactorReviewV1(
            factor_id="NFL.INT.LONG_DISTANCE",
            factor_version="v1",
            market_family="moneyline",
            horizon_seconds=10,
            decision="ACCEPT",
            sports_rationale="Mechanism is coherent enough for a locked holdout test.",
            anomaly_case_ids=(),
            priority="P1",
            reviewer="human:sports_expert",
            reviewed_at_utc="2026-07-25T20:00:00Z",
        )
    ]
    lock = freeze_factor_shortlist(
        discovery_results=discovery,
        factor_registry=_factor_registry(),
        reviews=reviews,
        source_hashes=["sha256:" + "2" * 64],
        output_root=tmp_path,
    )
    verified = verify_shortlist_lock(lock.manifest_path)

    assert verified.shortlist_sha256 == lock.shortlist_sha256
    assert verified.candidate_keys == (
        "NFL.INT.LONG_DISTANCE|moneyline|h=10",
    )
    assert lock.manifest_path.parent.name == (
        "sha256-" + lock.shortlist_sha256.removeprefix("sha256:")
    )

    holdout_observations = build_factor_observations(
        _feature_rows(game_prefix="2025_HOLD"),
        _market_rows(game_prefix="2025_HOLD"),
        _factor_registry(),
    )
    holdout = run_locked_historical_holdout(
        shortlist_manifest=lock.manifest_path,
        holdout_observations=holdout_observations,
        factor_registry=_factor_registry(),
        holdout_selection_id="sha256:" + "3" * 64,
        holdout_game_ids=sorted(holdout_observations["game_id"].unique()),
        discovery_game_ids=sorted(
            discovery_observations["game_id"].unique()
        ),
        bootstrap_samples=10_000,
        seed=20260725,
    )
    assert holdout.iloc[0]["holdout_status"] == "SIGNAL_CANDIDATE"
    assert holdout.iloc[0]["shortlist_sha256"] == lock.shortlist_sha256

    mutated_registry = _factor_registry()
    mutated_registry[0]["horizons_seconds"] = [60]
    with pytest.raises(FactorLabError, match="shortlist lock"):
        run_locked_historical_holdout(
            shortlist_manifest=lock.manifest_path,
            holdout_observations=holdout_observations,
            factor_registry=mutated_registry,
            holdout_selection_id="sha256:" + "3" * 64,
            holdout_game_ids=sorted(
                holdout_observations["game_id"].unique()
            ),
            discovery_game_ids=sorted(
                discovery_observations["game_id"].unique()
            ),
        )


def test_shortlist_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    discovery = run_factor_discovery(
        build_factor_observations(
            _feature_rows(game_prefix="2025_DISC"),
            _market_rows(game_prefix="2025_DISC"),
            _factor_registry(),
        ),
        factor_registry=_factor_registry(),
    )
    lock = freeze_factor_shortlist(
        discovery_results=discovery,
        factor_registry=_factor_registry(),
        reviews=[
            HumanFactorReviewV1(
                factor_id="NFL.INT.LONG_DISTANCE",
                factor_version="v1",
                market_family="moneyline",
                horizon_seconds=10,
                decision="ACCEPT",
                sports_rationale="Approved for test.",
                anomaly_case_ids=(),
                priority="P1",
                reviewer="human:test",
                reviewed_at_utc="2026-07-25T20:00:00Z",
            )
        ],
        source_hashes=["sha256:" + "2" * 64],
        output_root=tmp_path,
    )
    payload = json.loads(lock.manifest_path.read_text("utf-8"))
    payload["factor_definitions"][0]["target"] = "future_leak"
    lock.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FactorLabError, match="hash"):
        verify_shortlist_lock(lock.manifest_path)


def _registered_factor(
    *,
    factor_id: str,
    predicate: list[dict[str, object]],
    pit_fields: list[str],
    support_kind: str = "named",
    horizons_seconds: list[int] | None = None,
    minimum_episodes: int = 20,
    minimum_games: int = 6,
) -> dict[str, object]:
    return {
        "factor_id": factor_id,
        "version": "v1",
        "sports_meaning": f"Frozen test definition for {factor_id}.",
        "predicate": predicate,
        "pit_fields": pit_fields,
        "focal_entity": "team",
        "exclusions": [],
        "target": "signed_price_change",
        "support_kind": support_kind,
        "market_families": ["moneyline"],
        "horizons_seconds": horizons_seconds or [10],
        "minimum_episodes": minimum_episodes,
        "minimum_games": minimum_games,
        "minimum_episodes_per_venue": 0,
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
    }


def test_moneyline_baseline_is_fit_once_before_factor_expansion() -> None:
    features = _feature_rows(game_prefix="2025_BASE")
    episode_number = features["episode_id"].str.rsplit(":", n=1).str[-1].astype(int)
    features["subset_reweight"] = episode_number.mod(6).eq(0)
    markets = _market_rows(game_prefix="2025_BASE")
    market_episode = markets["episode_id"].str.rsplit(":", n=1).str[-1].astype(int)
    markets["signed_price_change"] += np.where(
        market_episode.mod(6).eq(0),
        0.05,
        0.0,
    )
    markets["first_post_event_trade"] = (
        markets["pre_event_actual_trade"] + markets["signed_price_change"]
    )
    all_factor = _registered_factor(
        factor_id="NFL.TEST.ALL",
        predicate=[{"field": "pre_distance", "operator": "gt", "value": 0}],
        pit_fields=["pre_distance"],
    )
    subset_factor = _registered_factor(
        factor_id="NFL.TEST.REWEIGHT",
        predicate=[
            {"field": "subset_reweight", "operator": "eq", "value": True}
        ],
        pit_fields=["subset_reweight"],
    )

    alone = build_factor_observations(features, markets, [all_factor])
    expanded = build_factor_observations(
        features,
        markets,
        [all_factor, subset_factor],
    )
    key = [
        "game_id",
        "episode_id",
        "contract_id",
        "outcome",
        "venue",
        "horizon_seconds",
    ]
    expected = (
        alone.loc[
            alone["factor_id"].eq("NFL.TEST.ALL"),
            key + ["diagnostic_baseline_prediction"],
        ]
        .sort_values(key)
        .reset_index(drop=True)
    )
    observed = (
        expanded.loc[
            expanded["factor_id"].eq("NFL.TEST.ALL"),
            key + ["diagnostic_baseline_prediction"],
        ]
        .sort_values(key)
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(observed, expected)


def test_binary_conditional_mean_cannot_enter_expert_shortlist() -> None:
    binary = _registered_factor(
        factor_id="NFL.TEST.BINARY",
        predicate=[
            {"field": "event_interception", "operator": "eq", "value": True}
        ],
        pit_fields=["event_interception"],
        support_kind="binary",
        minimum_episodes=40,
        minimum_games=8,
    )
    observations = build_factor_observations(
        _feature_rows(game_prefix="2025_BINARY"),
        _market_rows(game_prefix="2025_BINARY"),
        [binary],
    )

    result = run_factor_discovery(
        observations,
        factor_registry=[binary],
    ).iloc[0]

    assert result["estimand_type"] == "CONDITIONAL_MEAN_NO_COMPLEMENT"
    assert result["discovery_status"] == "DESCRIPTIVE_CONDITIONAL_MEAN"
    assert not bool(result["inferential_eligible"])


def test_discovery_emits_registered_zero_and_unsupported_grid_rows() -> None:
    features = _feature_rows(game_prefix="2025_GRID")
    episode_number = features["episode_id"].str.rsplit(":", n=1).str[-1].astype(int)
    features["never_observed"] = False
    features["rare_observed"] = episode_number.eq(0)
    observed = _registered_factor(
        factor_id="NFL.TEST.OBSERVED",
        predicate=[
            {"field": "event_interception", "operator": "eq", "value": True}
        ],
        pit_fields=["event_interception"],
        minimum_episodes=20,
        minimum_games=6,
    )
    no_data = _registered_factor(
        factor_id="NFL.TEST.NO_DATA",
        predicate=[
            {"field": "never_observed", "operator": "eq", "value": True}
        ],
        pit_fields=["never_observed"],
        horizons_seconds=[10, 60],
    )
    sparse = _registered_factor(
        factor_id="NFL.TEST.SPARSE",
        predicate=[
            {"field": "rare_observed", "operator": "eq", "value": True}
        ],
        pit_fields=["rare_observed"],
        minimum_episodes=300,
        minimum_games=16,
    )
    registry = [observed, no_data, sparse]
    observations = build_factor_observations(
        features,
        _market_rows(game_prefix="2025_GRID"),
        registry,
    )

    results = run_factor_discovery(
        observations,
        factor_registry=registry,
    )

    assert set(
        results[
            ["factor_id", "market_family", "horizon_seconds"]
        ].itertuples(index=False, name=None)
    ) == {
        ("NFL.TEST.NO_DATA", "moneyline", 10),
        ("NFL.TEST.NO_DATA", "moneyline", 60),
        ("NFL.TEST.OBSERVED", "moneyline", 10),
        ("NFL.TEST.SPARSE", "moneyline", 10),
    }
    no_data_rows = results[results["factor_id"].eq("NFL.TEST.NO_DATA")]
    assert no_data_rows["discovery_status"].eq("DATA_GAP").all()
    assert no_data_rows["support_status"].eq("DATA_GAP").all()
    assert no_data_rows["observation_count"].eq(0).all()
    sparse_row = results[results["factor_id"].eq("NFL.TEST.SPARSE")].iloc[0]
    assert sparse_row["discovery_status"] == "INSUFFICIENT_SUPPORT"
    assert sparse_row["support_status"] == "INSUFFICIENT_SUPPORT"
