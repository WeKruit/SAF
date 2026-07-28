from __future__ import annotations

import hashlib
import inspect
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports import (
    nfl_factor_lab_materialize as materialize_module,
)
from prediction_market.sports.nfl_factor_lab_bundles import (
    bundle_artifact_refs,
)
from prediction_market.sports.nfl_factor_lab_materialize import (
    ExistingX13StageProviderV2,
    FactorLabMaterializationError,
    FactorLabMaterializationPlanV2,
    _association_attrition,
    _arrow_safe_frame,
    _build_game_contribution,
    _build_leave_one_game_out,
    _partition_registered_factor_definitions,
    materialize_factor_lab_v2,
)
from prediction_market.sports.nfl_factor_lab_run_index import (
    factor_lab_materializer_code_lock_paths,
    factor_lab_materializer_code_lock_sha256,
    verify_factor_lab_run_index,
)
from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
    SourceInterval,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_IDS,
    canonical_sha256,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64
HASH_E = "sha256:" + "e" * 64
_AUTHORITY_FIXTURES = (
    (
        "catalog_registry_current",
        Path("charter/catalog_registry.csv"),
        b"current-catalog-registry",
    ),
    (
        "data_license_register_current",
        Path("registries/data_license_register.csv"),
        b"current-data-license-register",
    ),
    (
        "dataset_registry_current",
        Path("registries/dataset_registry.csv"),
        b"current-dataset-registry",
    ),
    (
        "experiment_registry_current",
        Path("registries/experiment_registry.csv"),
        b"current-experiment-registry",
    ),
)
_AUTHORITY_COMPONENT_SHA256S = {
    "charter/catalog_registry.csv": HASH_A,
    "registries/data_license_register.csv": HASH_B,
    "registries/dataset_registry.csv": HASH_E,
    "registries/experiment_registry.csv": HASH_D,
}
_AUTHORITY_SNAPSHOT_SHA256 = canonical_sha256(
    {
        "schema": "x13-current-dataset-authority-v2",
        "components": [
            {
                "path": path,
                "sha256": _AUTHORITY_COMPONENT_SHA256S[path],
            }
            for path in _AUTHORITY_COMPONENT_SHA256S
        ],
    }
)


def test_arrow_safe_frame_canonicalizes_parquet_ndarray_cells() -> None:
    frame = pd.DataFrame(
        {
            "direct": [np.asarray(["pass", "success"], dtype=object)],
            "nested": [{"players": np.asarray(["00-a", "00-b"])}],
        }
    )

    safe = _arrow_safe_frame(frame)

    assert safe.loc[0, "direct"] == '["pass","success"]'
    assert safe.loc[0, "nested"] == '{"players":["00-a","00-b"]}'


@pytest.mark.parametrize(
    "relative",
    (
        "src/prediction_market/sports/nfl_factor_lab.py",
        "src/prediction_market/sports/nfl_factor_lab_features.py",
    ),
)
def test_materializer_code_lock_binds_facade_and_feature_builder(
    tmp_path: Path,
    relative: str,
) -> None:
    for code_path in factor_lab_materializer_code_lock_paths():
        path = tmp_path / code_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(code_path.encode("utf-8"))
    before = factor_lab_materializer_code_lock_sha256(tmp_path)

    (tmp_path / relative).write_bytes(b"changed")

    assert factor_lab_materializer_code_lock_sha256(tmp_path) != before


def test_factor_definition_partition_runs_only_frozen_base_inventory() -> None:
    definitions = tuple(
        {
            "factor_id": factor_id,
            "version": "v1",
        }
        for factor_id in (
            *(
                f"NFL.EVENT.BASE_{index:03d}"
                for index in range(100)
            ),
            "NFL.COMBO.DISTANCE_GT_10",
            "NFL.COMBO.LATE_CLOSE_SCORE",
            "NFL.COMBO.RED_ZONE_TURNOVER",
            "NFL.COMBO.THIRD_FOURTH_DOWN_FAILURE",
            *(
                f"NFL.INTERACTION.CANDIDATE_{index:04d}"
                for index in range(1_368)
            ),
        )
    )

    base, interactions = _partition_registered_factor_definitions(definitions)

    assert len(base) == 104
    assert len(interactions) == 1_368
    assert all(".INTERACTION." not in row["factor_id"] for row in base)
    assert all(".INTERACTION." in row["factor_id"] for row in interactions)


def test_game_contribution_decomposes_the_pooled_effect_numerator() -> None:
    observations = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.PASS",
                "market_family": "moneyline",
                "horizon_seconds": 30,
                "game_id": "game-a",
                "episode_id": "episode-1",
                "venue": "polymarket",
                "signed_price_change": 1.0,
            },
            {
                "factor_id": "NFL.EVENT.PASS",
                "market_family": "moneyline",
                "horizon_seconds": 30,
                "game_id": "game-a",
                "episode_id": "episode-1",
                "venue": "kalshi",
                "signed_price_change": 3.0,
            },
            {
                "factor_id": "NFL.EVENT.PASS",
                "market_family": "moneyline",
                "horizon_seconds": 30,
                "game_id": "game-a",
                "episode_id": "episode-2",
                "venue": "polymarket",
                "signed_price_change": 1.0,
            },
            {
                "factor_id": "NFL.EVENT.PASS",
                "market_family": "moneyline",
                "horizon_seconds": 30,
                "game_id": "game-b",
                "episode_id": "episode-3",
                "venue": "polymarket",
                "signed_price_change": 1.0,
            },
        ]
    )

    contribution = _build_game_contribution(observations)

    by_game = contribution.set_index("game_id")
    assert by_game.loc["game-a", "effect_sum"] == 3.0
    assert by_game.loc["game-b", "effect_sum"] == 1.0
    assert by_game.loc["game-a", "contribution"] == pytest.approx(3 / 4)
    assert by_game.loc["game-b", "contribution"] == pytest.approx(1 / 4)

    result_keys = observations[
        ["factor_id", "market_family", "horizon_seconds"]
    ].drop_duplicates()
    loo = _build_leave_one_game_out(
        observations,
        result_keys,
        game_ids=("game-a", "game-b"),
    ).set_index("game_id")
    assert loo.loc["game-a", "point_estimate"] == 1.0
    assert loo.loc["game-b", "point_estimate"] == pytest.approx(1.5)
    assert loo.loc["game-b", "episode_count"] == 2


def test_zero_effect_contribution_is_explicitly_undefined() -> None:
    observations = pd.DataFrame(
        [
            {
                "factor_id": "NFL.EVENT.PASS",
                "market_family": "moneyline",
                "horizon_seconds": 1,
                "game_id": game_id,
                "episode_id": f"episode-{game_id}",
                "signed_price_change": 0.0,
            }
            for game_id in ("game-a", "game-b")
        ]
    )

    contribution = _build_game_contribution(observations)

    assert contribution["contribution"].isna().all()
    assert contribution["contribution_status"].eq(
        "ZERO_DENOMINATOR"
    ).all()


def test_association_attrition_requires_a_recleaned_actual_trade_series() -> None:
    projection = pd.DataFrame(
        [
            {
                "game_id": X13_GAME_IDS[0],
                "episode_id": "episode-1",
                "contract_id": "polymarket:0xcondition",
                "outcome": outcome,
                "venue": "polymarket",
                "horizon_seconds": 30,
                "actual_observation": True,
                "validity_status": "OBSERVED",
                "contaminated": False,
                "order_ambiguous": False,
                "staleness_seconds": 1,
                "orientation_audited": True,
                "analysis_eligible": True,
                "signal_candidate_eligible": True,
            }
            for outcome in ("HOME", "AWAY")
        ]
    )
    recleaned_observations = pd.DataFrame(
        [
            {
                "contract_id": "polymarket:0xcondition",
                "outcome": "HOME",
                "venue": "polymarket",
                "kind": "trade",
                "provenance": "observed",
                "primary_path_eligible": True,
                "price": "0.60",
            },
            {
                "contract_id": "polymarket:0xcondition",
                "outcome": "AWAY",
                "venue": "polymarket",
                "kind": "trade",
                "provenance": "derived_complement",
                "primary_path_eligible": False,
                "price": "0.40",
            },
        ]
    )

    attrition = _association_attrition(
        projection,
        logical_market_by_contract={
            "polymarket:0xcondition": "winner",
        },
        market_observations=recleaned_observations,
    ).set_index("outcome")

    assert bool(
        attrition.loc["HOME", "recleaned_actual_trade_series_available"]
    )
    assert bool(attrition.loc["HOME", "strict_analysis_eligible"])
    assert not bool(
        attrition.loc["AWAY", "recleaned_actual_trade_series_available"]
    )
    assert not bool(attrition.loc["AWAY", "strict_analysis_eligible"])
    assert (
        attrition.loc["AWAY", "exclusion_reason"]
        == "no_recleaned_actual_trade_series"
    )


def _tampered_reaction_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    game_id = X13_GAME_IDS[0]
    contract_id = "polymarket:0xcondition"
    projection = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "episode_id": "episode-1",
                "contract_id": contract_id,
                "outcome": "HOME",
                "signal_focal_team": "HOME",
                "venue": "polymarket",
                "family": "moneyline",
                "contract_measure": "winner",
                "semantic_orientation_id": "winner:HOME",
                "orientation_audited": True,
                "signal_candidate_eligible": True,
                "source_time_start_utc": "2025-09-01T00:00:10Z",
                "source_time_end_utc": "2025-09-01T00:00:11Z",
                "horizon_seconds": horizon,
                "pre_event_actual_trade": Decimal("0.50"),
                "first_post_event_trade": Decimal("0.60"),
                "vwap": Decimal("999"),
                "last_actual_trade": Decimal("998"),
                "signed_price_change": Decimal("997"),
                "maximum_excursion": Decimal("996"),
                "net_change_60s": Decimal("995"),
                "trade_count": 999,
                "volume": Decimal("999"),
                "staleness_seconds": Decimal("999"),
                "actual_observation": True,
                "validity_status": "OBSERVED",
                "analysis_eligible": True,
                "analysis_exclusion_reason": "ELIGIBLE",
                "forward_filled": False,
                "order_ambiguous": False,
                "contaminated": False,
            }
            for horizon in (10, 60)
        ]
    )

    def observation(
        observation_id: str,
        *,
        second: int,
        price: str,
        size: str,
        kind: str = "trade",
        provenance: str = "observed",
        primary_path_eligible: bool = True,
    ) -> dict[str, object]:
        start = datetime(
            2025,
            9,
            1,
            tzinfo=UTC,
        ) + timedelta(seconds=second)
        end = start + timedelta(seconds=1)
        return {
            "game_id": game_id,
            "contract_id": contract_id,
            "observation_id": observation_id,
            "logical_market_id": "winner",
            "outcome": "HOME",
            "venue": "polymarket",
            "kind": kind,
            "provenance": provenance,
            "primary_path_eligible": primary_path_eligible,
            "price": price,
            "size": size,
            "source_start": start.isoformat().replace("+00:00", "Z"),
            "source_end": end.isoformat().replace("+00:00", "Z"),
        }

    market_observations = pd.DataFrame(
        [
            observation(
                "actual-pre",
                second=8,
                price="0.50",
                size="1",
            ),
            observation(
                "actual-post-6",
                second=17,
                price="0.60",
                size="2",
            ),
            observation(
                "actual-post-9",
                second=20,
                price="0.70",
                size="1",
            ),
            observation(
                "actual-post-40",
                second=51,
                price="0.80",
                size="4",
            ),
            observation(
                "actual-post-50",
                second=61,
                price="0.45",
                size="1",
            ),
            observation(
                "derived-extreme",
                second=30,
                price="999",
                size="999",
                provenance="derived_complement",
                primary_path_eligible=False,
            ),
            observation(
                "candle-extreme",
                second=40,
                price="999",
                size="999",
                kind="kalshi_1m_candle_bbo",
                primary_path_eligible=False,
            ),
            observation(
                "non-primary-extreme",
                second=45,
                price="999",
                size="999",
                primary_path_eligible=False,
            ),
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "observation_id": observation_id,
                "included": True,
                "raw_manifest_sha256": HASH_A,
                "raw_object_sha256": HASH_B,
                "raw_request_sha256": HASH_C,
                "raw_row_fingerprint": HASH_D,
            }
            for observation_id in market_observations["observation_id"]
        ]
    )
    observed_cache = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "episode_id": "episode-1",
                "contract_id": contract_id,
                "outcome": "HOME",
                "venue": "polymarket",
                "horizon_seconds": horizon,
                "delay_scenario_seconds": 0,
                "logical_market_id": "winner",
                "overshoot_candidate": False,
                "reversal_candidate": False,
                "two_venue_direction_consistency": "TAMPERED",
            }
            for horizon in (10, 60)
        ]
    )
    return projection, market_observations, audit, observed_cache


def test_reaction_paths_recompute_all_metrics_from_recleaned_actual_trades() -> None:
    projection, observations, audit, _observed_cache = (
        _tampered_reaction_fixture()
    )

    paths = materialize_module._build_reaction_paths_v2(
        actual_projection=projection,
        market_observations=observations,
        market_audit=audit,
        logical_market_by_contract={
            "polymarket:0xcondition": "winner",
        },
        market_source_hashes=(HASH_E,),
    ).set_index("horizon_seconds")

    assert paths.loc[10, "pre_event_actual_trade"] == Decimal("0.50")
    assert paths.loc[10, "first_post_event_trade"] == Decimal("0.60")
    assert paths.loc[10, "horizon_vwap"] == Decimal("0.6333333333333333333333333333")
    assert paths.loc[10, "horizon_last_trade"] == Decimal("0.70")
    assert paths.loc[10, "signed_price_change"] == Decimal("0.20")
    assert paths.loc[10, "maximum_excursion"] == Decimal("0.20")
    assert paths.loc[10, "trade_count"] == 2
    assert paths.loc[10, "volume"] == Decimal("3")
    assert paths.loc[10, "staleness_seconds"] == Decimal("1")
    assert paths.loc[10, "metric_observation_ids"] == (
        "actual-pre",
        "actual-post-6",
        "actual-post-9",
    )

    assert paths.loc[60, "horizon_vwap"] == Decimal("0.69375")
    assert paths.loc[60, "horizon_last_trade"] == Decimal("0.45")
    assert paths.loc[60, "signed_price_change"] == Decimal("-0.05")
    assert paths.loc[60, "maximum_excursion"] == Decimal("0.30")
    assert paths.loc[60, "net_change_60s"] == Decimal("-0.05")
    assert paths.loc[60, "continuation_10_to_60"] == Decimal("-0.25")
    assert bool(paths.loc[60, "overshoot_candidate"])
    assert bool(paths.loc[60, "reversal_candidate"])
    assert paths.loc[60, "trade_count"] == 4
    assert paths.loc[60, "volume"] == Decimal("8")
    assert paths.loc[60, "staleness_seconds"] == Decimal("10")
    assert paths.loc[60, "metric_observation_ids"] == (
        "actual-pre",
        "actual-post-6",
        "actual-post-9",
        "actual-post-40",
        "actual-post-50",
    )
    assert paths.loc[60, "pre_event_observation_id"] == "actual-pre"
    assert (
        paths.loc[60, "first_post_observation_id"]
        == "actual-post-6"
    )
    assert (
        paths.loc[60, "horizon_last_trade_observation_id"]
        == "actual-post-50"
    )
    assert paths.loc[60, "metric_raw_manifest_sha256s"] == (HASH_A,)
    assert paths.loc[60, "metric_raw_object_sha256s"] == (HASH_B,)
    assert paths.loc[60, "metric_raw_request_sha256s"] == (HASH_C,)
    assert paths.loc[60, "metric_raw_row_fingerprints"] == (HASH_D,)
    assert paths.loc[60, "market_source_hashes"] == tuple(
        sorted((HASH_A, HASH_B, HASH_C, HASH_D, HASH_E))
    )
    assert (
        paths.loc[60, "reaction_time_verification_status"]
        == "RECOMPUTED_EXACT_ACTUAL_TRADE_WINDOW"
    )
    assert paths["reaction_calculation_semantics"].eq(
        "V2_RECLEANED_PRIMARY_ACTUAL_TRADE_CUMULATIVE_WINDOW"
    ).all()


def test_reaction_paths_reject_derived_candle_and_non_primary_metrics() -> None:
    projection, observations, audit, _observed_cache = (
        _tampered_reaction_fixture()
    )
    observations = observations[
        observations["observation_id"].isin(
            {
                "actual-pre",
                "derived-extreme",
                "candle-extreme",
                "non-primary-extreme",
            }
        )
    ].copy()

    paths = materialize_module._build_reaction_paths_v2(
        actual_projection=projection,
        market_observations=observations,
        market_audit=audit,
        logical_market_by_contract={
            "polymarket:0xcondition": "winner",
        },
        market_source_hashes=(HASH_E,),
    )

    assert not paths["actual_observation"].any()
    assert not paths["analysis_eligible"].any()
    assert paths["signed_price_change"].isna().all()
    assert paths["horizon_vwap"].isna().all()
    assert paths["trade_count"].eq(0).all()
    assert paths["volume"].map(lambda value: value == Decimal("0")).all()
    assert len(paths) == len(projection)
    assert paths["reaction_time_verification_status"].eq(
        "DATA_GAP_NO_POST_EVENT_ACTUAL_TRADE"
    ).all()
    assert paths["analysis_exclusion_reason"].eq(
        "MISSING_OBSERVATION"
    ).all()
    assert paths["metric_observation_ids"].map(
        lambda value: value == ("actual-pre",)
    ).all()
    assert not paths["metric_observation_ids"].map(
        lambda values: any("extreme" in value for value in values)
    ).any()


def test_reaction_paths_fail_closed_for_actual_trade_overlapping_game_interval() -> None:
    projection, observations, audit, _observed_cache = (
        _tampered_reaction_fixture()
    )
    overlap = observations.loc[
        observations["observation_id"].eq("actual-post-6")
    ].iloc[0].to_dict()
    overlap.update(
        {
            "observation_id": "actual-overlap",
            "price": "0.99",
            "source_start": "2025-09-01T00:00:10Z",
            "source_end": "2025-09-01T00:00:12Z",
        }
    )
    observations = pd.concat(
        [observations, pd.DataFrame([overlap])],
        ignore_index=True,
    )
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "observation_id": "actual-overlap",
                        "included": True,
                        "raw_manifest_sha256": HASH_A,
                        "raw_object_sha256": HASH_B,
                        "raw_request_sha256": HASH_C,
                        "raw_row_fingerprint": HASH_D,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    paths = materialize_module._build_reaction_paths_v2(
        actual_projection=projection,
        market_observations=observations,
        market_audit=audit,
        logical_market_by_contract={
            "polymarket:0xcondition": "winner",
        },
        market_source_hashes=(HASH_E,),
    )

    assert paths["order_ambiguous"].eq(True).all()
    assert not paths["actual_observation"].any()
    assert not paths["analysis_eligible"].any()
    assert paths["validity_status"].eq("ORDER_AMBIGUOUS").all()
    assert paths["reaction_time_verification_status"].eq(
        "ORDER_AMBIGUOUS_ACTUAL_TRADE_WINDOW"
    ).all()
    assert paths["overlapping_actual_trade_observation_ids"].map(
        lambda value: value == ("actual-overlap",)
    ).all()


def test_factor_definition_partition_fails_closed_when_inventory_drifts() -> None:
    definitions = tuple(
        {
            "factor_id": f"NFL.EVENT.BASE_{index:03d}",
            "version": "v1",
        }
        for index in range(103)
    )

    with pytest.raises(
        FactorLabMaterializationError,
        match="104 base and 1368 interaction",
    ):
        _partition_registered_factor_definitions(definitions)


def _single_frames(game_id: str) -> dict[str, pd.DataFrame]:
    common = {"game_id": [game_id]}
    return {
        "sports_row_disposition": pd.DataFrame(
            {
                **common,
                "raw_play_id": ["1"],
                "canonical_play_id": ["1"],
                "episode_id": ["e1"],
                "factor_eligible": [True],
            }
        ),
        "canonical_events": pd.DataFrame(
            {**common, "play_id": ["1"], "event_type": ["pass"]}
        ),
        "finalized_episodes": pd.DataFrame(
            {
                **common,
                "episode_id": ["e1"],
                "episode_type": ["pass"],
                "play_ids": [["1"]],
            }
        ),
        "stat_ledger": pd.DataFrame(
            {**common, "through_play_id": ["1"], "home_score": [0]}
        ),
        "pit_features": pd.DataFrame(
            {**common, "play_id": ["1"], "history_game_count": [8]}
        ),
        "contract_inventory": pd.DataFrame(
            {
                **common,
                "logical_market_id": ["winner"],
                "outcome": ["HOME"],
            }
        ),
        "market_observations": pd.DataFrame(
            {
                **common,
                "observation_id": ["o1"],
                "logical_market_id": ["winner"],
                "outcome": ["HOME"],
                "price": [0.5],
            }
        ),
        "market_cleaning_audit": pd.DataFrame(
            {
                **common,
                "audit_row_id": ["m1"],
                "observation_id": ["o1"],
                "included": [True],
                "decision": ["KEEP"],
            }
        ),
        "game_time_intervals": pd.DataFrame(
            {
                **common,
                "interval_id": ["gi1"],
                "episode_id": ["e1"],
                "delay_seconds": [0],
            }
        ),
        "market_time_intervals": pd.DataFrame(
            {
                **common,
                "interval_id": ["mi1"],
                "observation_id": ["o1"],
                "logical_market_id": ["winner"],
                "outcome": ["HOME"],
            }
        ),
        "source_timeline": pd.DataFrame(
            {
                **common,
                "timeline_id": ["t1"],
                "interval_id": ["gi1"],
                "layer": ["G"],
                "delay_seconds": [0],
            }
        ),
        "timeline_audit": pd.DataFrame(
            {
                **common,
                "timeline_audit_id": ["ta1"],
                "timeline_id": ["t1"],
                "status": ["PASS"],
            }
        ),
        "association_attrition": pd.DataFrame(
            {
                **common,
                "association_id": ["a1"],
                "episode_id": ["e1"],
                "logical_market_id": ["winner"],
                "outcome": ["HOME"],
                "status": ["OBSERVED"],
            }
        ),
        "reaction_paths": pd.DataFrame(
            {
                **common,
                "episode_id": ["e1"],
                "logical_market_id": ["winner"],
                "outcome": ["HOME"],
                "venue": ["polymarket"],
                "horizon_seconds": [30],
                "market_family": ["moneyline"],
            }
        ),
        "factor_observations": pd.DataFrame(
            {
                **common,
                "factor_id": ["NFL.EVENT.PASS"],
                "episode_id": ["e1"],
                "logical_market_id": ["winner"],
                "outcome": ["HOME"],
                "venue": ["polymarket"],
                "horizon_seconds": [30],
                "market_family": ["moneyline"],
            }
        ),
        "factor_exclusion_audit": pd.DataFrame(
            {**common, "audit_id": ["f1"], "decision": ["eligible"]}
        ),
        "single_game_factor_results": pd.DataFrame(
            {
                **common,
                "factor_id": ["NFL.EVENT.PASS"],
                "market_family": ["moneyline"],
                "horizon_seconds": [30],
                "point_estimate": [0.01],
            }
        ),
    }


def _cross_frames(game_ids: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {
        "cross_game_factor_results": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "point_estimate": 0.01,
                }
            ]
        ),
        "game_contribution": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "game_id": game_id,
                    "contribution": 1 / len(game_ids),
                }
                for game_id in game_ids
            ]
        ),
        "leave_one_game_out": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "game_id": game_id,
                    "point_estimate": 0.01,
                }
                for game_id in game_ids
            ]
        ),
        "venue_consistency": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "venue": "polymarket",
                    "point_estimate": 0.01,
                }
            ]
        ),
        "expert_review_queue": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "decision": "PENDING",
                }
            ]
        ),
    }


def _plan(tmp_path: Path, source: Path) -> FactorLabMaterializationPlanV2:
    for relative in factor_lab_materializer_code_lock_paths():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    pointer = (
        tmp_path
        / "var/raw/capture-pointers"
        / f"{HASH_C.removeprefix('sha256:')}.json"
    )
    receipt = (
        tmp_path
        / "var/raw/capture-receipts"
        / f"{HASH_D.removeprefix('sha256:')}.json"
    )
    for path, payload in (
        (pointer, b"capture-pointer"),
        (receipt, b"capture-receipt"),
        *(
            (tmp_path / relative, payload)
            for _name, relative, payload in _AUTHORITY_FIXTURES
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return FactorLabMaterializationPlanV2(
        project_root=tmp_path,
        single_game_output_root=tmp_path / "single",
        cross_game_output_root=tmp_path / "cross",
        run_index_path=tmp_path / "run-index-v2.json",
        game_ids=X13_GAME_IDS,
        source_hashes=(HASH_A, HASH_C, HASH_D),
        source_artifacts={
            "x13_market_universe_v1": source,
            "x13_market_capture_pointer_v2": pointer,
            "x13_market_capture_receipt_v2": receipt,
            **{
                name: tmp_path / relative
                for name, relative, _payload in _AUTHORITY_FIXTURES
            },
        },
        factor_registry_sha256=HASH_B,
        code_lock_sha256=factor_lab_materializer_code_lock_sha256(tmp_path),
        batch_spec_sha256=HASH_A,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
    )


def _fake_capture_index() -> SimpleNamespace:
    return SimpleNamespace(
        status="PASS",
        capture_sha256=HASH_C,
        capture_receipt_sha256=HASH_D,
        raw_manifest_count=10_185,
        raw_byte_length=2_387_000_000,
        terminal_proofs_verified=True,
        manifest_hashes_verified=True,
    )


def _install_indexed_reclean(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_game_id_for: str | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    index = _fake_capture_index()
    verify_calls: list[dict[str, object]] = []
    reclean_calls: list[str] = []

    def verify(**kwargs):
        verify_calls.append(kwargs)
        return index

    monkeypatch.setattr(
        materialize_module,
        "verify_x13_market_capture_v2",
        verify,
        raising=False,
    )

    def reclean(observed_index: object, game_id: str):
        assert observed_index is index
        reclean_calls.append(game_id)
        return SimpleNamespace(
            status="PASS",
            game_id=(
                "2025_99_WRONG_GAME"
                if wrong_game_id_for == game_id
                else game_id
            ),
            capture_sha256=HASH_C,
        )

    monkeypatch.setattr(
        materialize_module,
        "reclean_x13_verified_market_game_v2",
        reclean,
        raising=False,
    )
    return verify_calls, reclean_calls


def _formal_market_game(
    game_id: str,
    *,
    current_registry_sha256: str = HASH_E,
    authority_overrides: dict[str, str] | None = None,
) -> SimpleNamespace:
    start = datetime(2025, 9, 8, 0, 0, 1, tzinfo=UTC)
    end = start + timedelta(microseconds=1)
    observed = MarketObservation(
        observation_id="observation-observed",
        venue="polymarket",
        raw_market_id="0xcondition",
        logical_market_id="winner",
        outcome="HOME",
        kind="trade",
        source_interval=SourceInterval(
            start=start,
            end=end,
            end_inclusive=True,
        ),
        price=Decimal("0.60"),
        size=Decimal("4"),
        bid=None,
        ask=None,
        provenance="observed",
        render_semantics="solid",
        native_id="0xtrade",
        native_source_time=start,
        primary_path_eligible=True,
    )
    derived = MarketObservation(
        observation_id="observation-derived",
        venue="polymarket",
        raw_market_id="0xcondition",
        logical_market_id="winner",
        outcome="AWAY",
        kind="trade",
        source_interval=observed.source_interval,
        price=Decimal("0.40"),
        size=Decimal("4"),
        bid=None,
        ask=None,
        provenance="derived_complement",
        render_semantics="dashed",
        native_id="0xtrade",
        derived_from_observation_id=observed.observation_id,
        native_source_time=start,
        primary_path_eligible=False,
    )
    observations = (observed, derived)
    contract = {
        "game_id": game_id,
        "contract_id": "polymarket:0xcondition",
        "condition_id": "0xcondition",
        "venue_market_id": "0xcondition",
        "raw_contract_ids": ["0xcondition"],
        "logical_market_id": "winner",
        "venue": "polymarket",
        "family": "moneyline",
        "period": "full_game",
        "subject": "HOME",
        "native_subject": "HOME",
        "measure": "win",
        "comparator": "equals",
        "line": None,
        "kind": "primitive",
        "dependency_game_ids": [game_id],
        "analysis_eligible": True,
        "outcomes": ["HOME", "AWAY"],
        "native_outcomes": ["HOME", "AWAY"],
        "outcome_coverage": {
            "HOME": "available",
            "AWAY": "unavailable_no_trades",
        },
        "rule_sha256": HASH_A,
    }

    def observation_record(value: MarketObservation) -> dict[str, object]:
        return {
            "game_id": game_id,
            "contract_id": contract["contract_id"],
            "observation_id": value.observation_id,
            "venue": value.venue,
            "raw_market_id": value.raw_market_id,
            "logical_market_id": value.logical_market_id,
            "outcome": value.outcome,
            "kind": value.kind,
            "source_start": start.isoformat().replace("+00:00", "Z"),
            "source_end": end.isoformat().replace("+00:00", "Z"),
            "source_end_inclusive": True,
            "price": str(value.price),
            "size": str(value.size),
            "bid": None,
            "ask": None,
            "provenance": value.provenance,
            "render_semantics": value.render_semantics,
            "native_id": value.native_id,
            "derived_from_observation_id": (
                value.derived_from_observation_id
            ),
            "native_source_time": start.isoformat().replace("+00:00", "Z"),
            "is_block_trade": None,
            "taker_side": None,
            "taker_outcome_side": None,
            "yes_price_dollars": None,
            "no_price_dollars": None,
            "primary_path_eligible": value.primary_path_eligible,
            "yes_bid_ohlc": None,
            "yes_ask_ohlc": None,
            "volume": None,
            "open_interest": None,
            "stage_semantics": "RAW_CAPTURE_V2_RECLEAN_EQUIVALENT",
            "historical_l2_available": False,
            "local_receive_time_available": False,
        }

    observation_rows = tuple(
        observation_record(value) for value in observations
    )
    authority = {
        "current_authority_snapshot_sha256": (
            _AUTHORITY_SNAPSHOT_SHA256
        ),
        "current_catalog_registry_sha256": HASH_A,
        "current_data_license_registry_sha256": HASH_B,
        "current_dataset_registry_sha256": current_registry_sha256,
        "current_experiment_registry_sha256": HASH_D,
    }
    authority.update(authority_overrides or {})
    audit_rows = tuple(
        {
            "game_id": game_id,
            "audit_row_id": f"audit-{value.observation_id}",
            "observation_id": value.observation_id,
            "contract_id": contract["contract_id"],
            "logical_market_id": "winner",
            "venue": "polymarket",
            "outcome": value.outcome,
            "observation_kind": "trade",
            "provenance": value.provenance,
            "source_interval_start_utc": observation_rows[index][
                "source_start"
            ],
            "source_interval_end_utc": observation_rows[index][
                "source_end"
            ],
            "native_id": "0xtrade",
            "capture_sha256": HASH_C,
            "raw_manifest_sha256": HASH_A,
            "captured_license_status": "pending",
            "current_license_status": "research_only",
            **authority,
            "raw_object_sha256": HASH_B,
            "raw_request_sha256": HASH_D,
            "raw_row_index": 0,
            "raw_row_fingerprint": HASH_E,
            "raw_source_url": "https://example.invalid/trades",
            "raw_source_cursor": "condition:0xcondition;offset:0",
            "raw_page_index": 0,
            "raw_cursor_in": "ROOT",
            "raw_cursor_out": "",
            "terminal_proof": "unsaturated_time_window",
            "terminal_cursor": "",
            "terminal_manifest_sha256": HASH_A,
            "condition_id": "0xcondition",
            "token_id": "123",
            "ticker": None,
            "dedupe_decision": "KEEP",
            "exclusion_reason": None,
            "actual_observation": value.provenance == "observed",
            "actual_trade": value.provenance == "observed",
            "bbo_available": False,
            "bbo_tick": False,
            "executable": False,
            "included": True,
            "raw_request_row_available": True,
            "raw_token_or_ticker_binding_available": True,
            "v2_cleaning_equivalent": True,
            "stage_semantics": "RAW_CAPTURE_V2_RECLEAN_EQUIVALENT",
            "data_gap": None,
        }
        for index, value in enumerate(observations)
    )
    return SimpleNamespace(
        status="PASS",
        game_id=game_id,
        capture_sha256=HASH_C,
        normalized_observations=observations,
        contract_inventory_records=lambda: [dict(contract)],
        normalized_observation_records=lambda: [
            dict(row) for row in observation_rows
        ],
        market_cleaning_audit_records=lambda: [
            dict(row) for row in audit_rows
        ],
    )


def test_formal_market_inputs_preserve_v2_lineage_and_observation_semantics() -> None:
    market_game = _formal_market_game(X13_GAME_IDS[0])

    frames, typed_contracts, typed_observations, logical_by_contract = (
        materialize_module._formal_market_inputs_v2(
            game_id=X13_GAME_IDS[0],
            market_game=market_game,
            current_authority_snapshot_sha256=(
                _AUTHORITY_SNAPSHOT_SHA256
            ),
            current_authority_component_sha256s=(
                _AUTHORITY_COMPONENT_SHA256S
            ),
        )
    )

    contracts = frames["contract_inventory"]
    observations = frames["market_observations"]
    audit = frames["market_cleaning_audit"]
    assert len(contracts) == 2
    assert set(contracts["outcome"]) == {"HOME", "AWAY"}
    assert logical_by_contract == {
        "polymarket:0xcondition": "winner",
    }
    assert len(typed_contracts) == 1
    assert typed_observations == market_game.normalized_observations
    assert observations[
        ["observation_id", "provenance", "render_semantics"]
    ].to_dict("records") == [
        {
            "observation_id": "observation-observed",
            "provenance": "observed",
            "render_semantics": "solid",
        },
        {
            "observation_id": "observation-derived",
            "provenance": "derived_complement",
            "render_semantics": "dashed",
        },
    ]
    assert observations["stage_semantics"].eq(
        "RAW_CAPTURE_V2_RECLEAN_EQUIVALENT"
    ).all()
    assert not observations.astype(str).apply(
        lambda column: column.str.contains(
            "PRIOR_EVIDENCE_NOT_V2_EQUIVALENT",
            regex=False,
        )
    ).any().any()
    required_lineage = {
        "capture_sha256",
        "raw_manifest_sha256",
        "raw_object_sha256",
        "raw_request_sha256",
        "raw_row_index",
        "raw_row_fingerprint",
        "raw_source_url",
        "raw_source_cursor",
        "raw_page_index",
        "raw_cursor_in",
        "raw_cursor_out",
        "terminal_proof",
        "terminal_cursor",
        "terminal_manifest_sha256",
        "condition_id",
        "token_id",
        "ticker",
        "captured_license_status",
        "current_license_status",
        "current_authority_snapshot_sha256",
        "current_catalog_registry_sha256",
        "current_data_license_registry_sha256",
        "current_dataset_registry_sha256",
        "current_experiment_registry_sha256",
    }
    assert required_lineage.issubset(audit.columns)
    assert audit["raw_request_row_available"].eq(True).all()
    assert audit["raw_token_or_ticker_binding_available"].eq(True).all()
    assert audit["v2_cleaning_equivalent"].eq(True).all()
    assert audit["stage_semantics"].eq(
        "RAW_CAPTURE_V2_RECLEAN_EQUIVALENT"
    ).all()
    assert audit["data_gap"].isna().all()
    assert audit["current_authority_snapshot_sha256"].eq(
        _AUTHORITY_SNAPSHOT_SHA256
    ).all()
    assert audit["current_catalog_registry_sha256"].eq(HASH_A).all()
    assert audit["current_data_license_registry_sha256"].eq(HASH_B).all()
    assert audit["current_dataset_registry_sha256"].eq(HASH_E).all()
    assert audit["current_experiment_registry_sha256"].eq(HASH_D).all()
    assert (
        audit["token_id"].notna() | audit["ticker"].notna()
    ).all()


def test_formal_market_inputs_reject_stale_current_registry_binding() -> None:
    market_game = _formal_market_game(
        X13_GAME_IDS[0],
        current_registry_sha256=HASH_D,
    )

    with pytest.raises(
        FactorLabMaterializationError,
        match="current dataset registry",
    ):
        materialize_module._formal_market_inputs_v2(
            game_id=X13_GAME_IDS[0],
            market_game=market_game,
            current_authority_snapshot_sha256=(
                _AUTHORITY_SNAPSHOT_SHA256
            ),
            current_authority_component_sha256s=(
                _AUTHORITY_COMPONENT_SHA256S
            ),
        )


@pytest.mark.parametrize(
    "audit_field",
    (
        "current_authority_snapshot_sha256",
        "current_catalog_registry_sha256",
        "current_data_license_registry_sha256",
        "current_experiment_registry_sha256",
    ),
)
def test_formal_market_inputs_reject_stale_authority_binding(
    audit_field: str,
) -> None:
    market_game = _formal_market_game(
        X13_GAME_IDS[0],
        authority_overrides={audit_field: HASH_C},
    )

    with pytest.raises(
        FactorLabMaterializationError,
        match="current authority",
    ):
        materialize_module._formal_market_inputs_v2(
            game_id=X13_GAME_IDS[0],
            market_game=market_game,
            current_authority_snapshot_sha256=(
                _AUTHORITY_SNAPSHOT_SHA256
            ),
            current_authority_component_sha256s=(
                _AUTHORITY_COMPONENT_SHA256S
            ),
        )


def test_materialization_plan_requires_formal_market_source_artifacts(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "universe.json"
    universe.write_bytes(b"universe")
    plan = _plan(tmp_path, universe)

    with pytest.raises(
        FactorLabMaterializationError,
        match="market reclean source artifacts",
    ):
        replace(
            plan,
            source_artifacts={
                "x13_market_universe_v1": universe,
            },
        )


@pytest.mark.parametrize(
    "source_name",
    tuple(name for name, _relative, _payload in _AUTHORITY_FIXTURES),
)
def test_materialization_plan_requires_exact_authority_source_paths(
    tmp_path: Path,
    source_name: str,
) -> None:
    universe = tmp_path / "universe.json"
    universe.write_bytes(b"universe")
    plan = _plan(tmp_path, universe)

    with pytest.raises(
        FactorLabMaterializationError,
        match="authority source path differs",
    ):
        replace(
            plan,
            source_artifacts={
                **plan.source_artifacts,
                source_name: tmp_path / f"wrong-{source_name}.csv",
            },
        )


def test_materialization_rejects_wrong_indexed_game_before_its_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = tmp_path / "universe.json"
    universe.write_bytes(b"universe")
    target = X13_GAME_IDS[4]
    verify_calls, reclean_calls = _install_indexed_reclean(
        monkeypatch,
        wrong_game_id_for=target,
    )
    worker_calls: list[str] = []

    def build_single(
        game_id: str,
        _market_game: object | None = None,
    ) -> dict[str, pd.DataFrame]:
        worker_calls.append(game_id)
        return _single_frames(game_id)

    with pytest.raises(
        FactorLabMaterializationError,
        match="indexed reclean game identity",
    ):
        materialize_factor_lab_v2(
            _plan(tmp_path, universe),
            single_game_frame_builder=build_single,
            cross_game_frame_builder=_cross_frames,
        )

    assert len(verify_calls) == 1
    assert target in reclean_calls
    assert target not in worker_calls
    assert len(reclean_calls) == len(set(reclean_calls))


def test_legacy_market_projection_wrapper_is_absent_from_formal_provider() -> None:
    assert not hasattr(
        ExistingX13StageProviderV2,
        "_market_cleaning_audit",
    )
    parameters = inspect.signature(
        ExistingX13StageProviderV2.build_single_game_frames
    ).parameters
    assert tuple(parameters) == ("self", "game_id", "market_game")
    source = inspect.getsource(
        ExistingX13StageProviderV2.build_single_game_frames
    )
    assert 'payload["contracts"]' not in source
    assert 'payload["observations"]' not in source
    assert "PRIOR_EVIDENCE_NOT_V2_EQUIVALENT" not in source


def test_twenty_game_materialization_is_single_game_first_and_referential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "frozen-input.bin"
    source.write_bytes(b"frozen")
    calls: list[str] = []
    verify_calls, reclean_calls = _install_indexed_reclean(monkeypatch)
    first_wave = frozenset(X13_GAME_IDS[:4])
    four_workers = threading.Barrier(4, timeout=2)

    def build_single(
        game_id: str,
        market_game: object | None = None,
    ) -> dict[str, pd.DataFrame]:
        assert market_game is not None
        assert market_game.game_id == game_id
        assert len(verify_calls) == 1
        assert reclean_calls.count(game_id) == 1
        if game_id in first_wave:
            four_workers.wait()
        calls.append(game_id)
        return _single_frames(game_id)

    def build_cross(single_game_bundles) -> dict[str, pd.DataFrame]:
        calls.append("CROSS")
        game_ids = tuple(bundle.game_id for bundle in single_game_bundles)
        assert game_ids == X13_GAME_IDS
        assert set(calls[:-1]) == set(X13_GAME_IDS)
        assert all(bundle.manifest_path.is_file() for bundle in single_game_bundles)
        return _cross_frames(game_ids)

    result = materialize_factor_lab_v2(
        _plan(tmp_path, source),
        single_game_frame_builder=build_single,
        cross_game_frame_builder=build_cross,
    )

    assert len(verify_calls) == 1
    assert verify_calls[0] == {
        "universe_artifact": source,
        "capture_pointer": (
            tmp_path
            / "var/raw/capture-pointers"
            / f"{HASH_C.removeprefix('sha256:')}.json"
        ),
        "raw_store_root": tmp_path / "var/raw",
        "program_root": tmp_path,
    }
    assert len(reclean_calls) == len(X13_GAME_IDS)
    assert set(reclean_calls) == set(X13_GAME_IDS)
    assert len(reclean_calls) == len(set(reclean_calls))
    assert tuple(bundle.game_id for bundle in result.single_game_bundles) == (
        X13_GAME_IDS
    )
    assert result.cross_game_bundle.game_ids == X13_GAME_IDS
    verified = verify_factor_lab_run_index(
        result.run_index.path,
        project_root=tmp_path,
    )
    assert verified.game_ids == X13_GAME_IDS
    assert verified.holdout_reaction_accessed is False
    assert {
        name for name, _path, _digest, _length in verified.source_artifacts
    } == {
        "catalog_registry_current",
        "data_license_register_current",
        "dataset_registry_current",
        "experiment_registry_current",
        "x13_market_capture_pointer_v2",
        "x13_market_capture_receipt_v2",
        "x13_market_universe_v1",
    }
    source_descriptors = {
        name: (path, digest, length)
        for name, path, digest, length in verified.source_artifacts
    }
    for name, relative, payload in _AUTHORITY_FIXTURES:
        assert source_descriptors[name] == (
            tmp_path / relative,
            "sha256:" + hashlib.sha256(payload).hexdigest(),
            len(payload),
        )
    assert len(verified.stage_manifest) == 20 * 17 + 5
    refs = bundle_artifact_refs(
        result.single_game_bundles[0],
        project_root=tmp_path,
        s3_bucket=verified.s3_bucket,
        s3_prefix=verified.s3_prefix,
    )
    assert all(
        ref.s3_key.startswith(
            "prediction-market/research/nfl/x13/single-game/"
            f"{X13_GAME_IDS[0]}/"
        )
        for ref in refs
    )
    assert all(
        "/research/nfl/x13/research/nfl/x13/" not in ref.s3_key
        for ref in refs
    )


def test_materialization_refuses_a_partial_or_reordered_frozen_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "frozen-input.bin"
    source.write_bytes(b"frozen")
    _install_indexed_reclean(monkeypatch)
    plan = _plan(tmp_path, source)
    partial = FactorLabMaterializationPlanV2(
        project_root=plan.project_root,
        single_game_output_root=plan.single_game_output_root,
        cross_game_output_root=plan.cross_game_output_root,
        run_index_path=plan.run_index_path,
        game_ids=X13_GAME_IDS[:-1],
        source_hashes=plan.source_hashes,
        source_artifacts=plan.source_artifacts,
        factor_registry_sha256=plan.factor_registry_sha256,
        code_lock_sha256=plan.code_lock_sha256,
        batch_spec_sha256=plan.batch_spec_sha256,
        s3_bucket=plan.s3_bucket,
        s3_prefix=plan.s3_prefix,
    )

    with pytest.raises(
        FactorLabMaterializationError,
        match="exactly match the frozen X-13 discovery set",
    ):
        materialize_factor_lab_v2(
            partial,
            single_game_frame_builder=_single_frames,
            cross_game_frame_builder=_cross_frames,
        )


def test_materialization_rejects_a_non_authoritative_code_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "frozen-input.bin"
    source.write_bytes(b"frozen")
    verify_calls, reclean_calls = _install_indexed_reclean(monkeypatch)
    single_calls: list[str] = []

    def build_single(
        game_id: str,
        _market_game: object | None = None,
    ) -> dict[str, pd.DataFrame]:
        single_calls.append(game_id)
        return _single_frames(game_id)

    def build_cross(single_game_bundles) -> dict[str, pd.DataFrame]:
        return _cross_frames(
            tuple(bundle.game_id for bundle in single_game_bundles)
        )

    with pytest.raises(
        FactorLabMaterializationError,
        match="authoritative current code lock",
    ):
        materialize_factor_lab_v2(
            replace(_plan(tmp_path, source), code_lock_sha256=HASH_C),
            single_game_frame_builder=build_single,
            cross_game_frame_builder=build_cross,
        )

    assert single_calls == []
    assert verify_calls == []
    assert reclean_calls == []


def test_single_game_worker_failure_prevents_cross_game_and_run_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "frozen-input.bin"
    source.write_bytes(b"frozen")
    verify_calls, reclean_calls = _install_indexed_reclean(monkeypatch)
    cross_called = False

    def build_single(
        game_id: str,
        market_game: object | None = None,
    ) -> dict[str, pd.DataFrame]:
        assert market_game is not None
        if game_id == X13_GAME_IDS[4]:
            raise RuntimeError("deliberate worker failure")
        return _single_frames(game_id)

    def build_cross(_single_game_bundles) -> dict[str, pd.DataFrame]:
        nonlocal cross_called
        cross_called = True
        return _cross_frames(X13_GAME_IDS)

    with pytest.raises(RuntimeError, match="deliberate worker failure"):
        materialize_factor_lab_v2(
            _plan(tmp_path, source),
            single_game_frame_builder=build_single,
            cross_game_frame_builder=build_cross,
        )

    assert cross_called is False
    assert len(verify_calls) == 1
    assert len(reclean_calls) == len(set(reclean_calls))
    assert not (tmp_path / "run-index-v2.json").exists()


def test_cross_game_builder_cannot_omit_a_single_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "frozen-input.bin"
    source.write_bytes(b"frozen")
    _install_indexed_reclean(monkeypatch)

    def build_single(
        game_id: str,
        market_game: object | None = None,
    ) -> dict[str, pd.DataFrame]:
        assert market_game is not None
        return _single_frames(game_id)

    def broken_cross(single_game_bundles) -> dict[str, pd.DataFrame]:
        game_ids = tuple(bundle.game_id for bundle in single_game_bundles)
        frames = _cross_frames(game_ids)
        frames["game_contribution"] = frames["game_contribution"].iloc[:-1]
        return frames

    with pytest.raises(
        FactorLabMaterializationError,
        match="game_contribution does not cover the exact single-game set",
    ):
        materialize_factor_lab_v2(
            _plan(tmp_path, source),
            single_game_frame_builder=build_single,
            cross_game_frame_builder=broken_cross,
        )
