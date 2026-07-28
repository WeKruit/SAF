from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports.nfl_factor_lab_factor_observations import (
    FactorObservationBuildError,
    build_factor_observations_v2,
)


SPORTS_HASH = "sha256:" + "1" * 64
MARKET_HASH = "sha256:" + "2" * 64
RULE_HASH = "sha256:" + "3" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _factor_registry() -> list[dict[str, object]]:
    return [
        {
            "factor_id": "NFL.EVENT.PASS",
            "version": "v1",
            "sports_meaning": "A source-structured pass play.",
            "predicate": [
                {"field": "event_pass", "operator": "eq", "value": True}
            ],
            "pit_fields": ["event_pass"],
            "focal_entity": "team",
            "exclusions": [],
            "target": "signed_price_change",
            "market_families": [
                "moneyline",
                "period",
                "player_passing",
                "spread",
                "team_total",
                "total",
            ],
            "horizons_seconds": [30],
            "support_kind": "named",
            "minimum_episodes": 20,
            "minimum_games": 6,
            "minimum_episodes_per_venue": 0,
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        }
    ]


def _sports_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_AWY_HME",
                "play_id": "play-1",
                "episode_id": "episode-1",
                "order_sequence": 10,
                "source_time_utc": "2025-09-01T00:10:00Z",
                "event_eligible": True,
                "focal_team": "HME",
                "quarter": 2,
                "event_pass": True,
                "player_stat_progress": {
                    "player-1": {"passing_yards": 125.0}
                },
                "sports_source_hashes": (SPORTS_HASH,),
            }
        ]
    )


def _inventory_rows() -> pd.DataFrame:
    rows = [
        ("winner", "moneyline", "game", "2025_01_AWY_HME", "winner", "HME"),
        ("spread", "spread", "game", "HME", "margin", "HME"),
        ("total", "total", "game", "2025_01_AWY_HME", "points", "Over"),
        (
            "team-total",
            "team_total",
            "game",
            "HME",
            "points",
            "Over",
        ),
        (
            "player-pass",
            "player_passing",
            "game",
            "player-1",
            "passing_yards",
            "Over",
        ),
        (
            "first-half",
            "period",
            "first_half",
            "2025_01_AWY_HME",
            "winner",
            "HME",
        ),
        (
            "wrong-team-total",
            "team_total",
            "game",
            "AWY",
            "points",
            "Over",
        ),
        (
            "unbound-player",
            "player_passing",
            "game",
            "player-2",
            "passing_yards",
            "Over",
        ),
        (
            "wrong-period",
            "period",
            "second_half",
            "2025_01_AWY_HME",
            "winner",
            "HME",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_AWY_HME",
                "contract_id": f"kalshi:{logical_market_id}",
                "logical_market_id": logical_market_id,
                "venue": "kalshi",
                "family": family,
                "period": period,
                "subject": subject,
                "measure": measure,
                "outcome": outcome,
                "kind": "primitive",
                "analysis_eligible": True,
                "rule_sha256": RULE_HASH,
            }
            for (
                logical_market_id,
                family,
                period,
                subject,
                measure,
                outcome,
            ) in rows
        ]
    )


def _reaction_rows() -> pd.DataFrame:
    inventory = _inventory_rows()
    rows: list[dict[str, object]] = []
    for index, contract in inventory.iterrows():
        rows.append(
            {
                "game_id": contract["game_id"],
                "episode_id": "episode-1",
                "logical_market_id": contract["logical_market_id"],
                "outcome": contract["outcome"],
                "venue": contract["venue"],
                "horizon_seconds": 30,
                "event_source_time_utc": "2025-09-01T00:10:01Z",
                "first_post_trade_time_utc": "2025-09-01T00:10:03Z",
                "pre_event_actual_trade": 0.50,
                "first_post_event_trade": 0.54,
                "horizon_vwap": 0.53,
                "horizon_last_trade": 0.54,
                "signed_price_change": 0.04,
                "maximum_excursion": 0.05,
                "continuation_10_to_60": 0.01,
                "trade_count": 2,
                "volume": 10.0,
                "staleness_seconds": 1.0,
                "actual_observation": True,
                "analysis_eligible": True,
                "analysis_exclusion_reason": "ELIGIBLE",
                "forward_filled": False,
                "order_ambiguous": False,
                "contaminated": False,
                "market_source_hashes": (MARKET_HASH,),
                "reaction_path_id": f"path-{index}",
            }
        )
    return pd.DataFrame(rows)


def test_structured_target_adapter_includes_only_proven_targets() -> None:
    result = build_factor_observations_v2(
        _sports_rows(),
        _reaction_rows(),
        _inventory_rows(),
        _factor_registry(),
    )

    observations = result.factor_observations
    assert set(observations["logical_market_id"]) == {
        "winner",
        "spread",
        "total",
        "team-total",
        "player-pass",
        "first-half",
    }
    assert observations["factor_id"].eq("NFL.EVENT.PASS").all()
    assert observations["factor_version"].eq("v1").all()
    assert observations["factor_definition_sha256"].str.fullmatch(
        r"sha256:[0-9a-f]{64}"
    ).all()
    assert observations["predicate_hit"].eq(True).all()
    assert observations["exclusion_reasons"].map(
        lambda value: tuple(value) == ()
    ).all()
    assert observations["sports_source_hashes"].map(
        lambda value: tuple(value) == (SPORTS_HASH,)
    ).all()
    assert observations["market_source_hashes"].map(
        lambda value: tuple(value) == (MARKET_HASH,)
    ).all()
    assert observations["source_hashes"].map(
        lambda value: tuple(value)
        == tuple(sorted((SPORTS_HASH, MARKET_HASH, RULE_HASH)))
    ).all()
    assert observations["event_source_time_utc"].eq(
        "2025-09-01T00:10:01Z"
    ).all()
    assert observations["sports_play_source_time_utc"].eq(
        "2025-09-01T00:10:00Z"
    ).all()
    assert observations["first_post_trade_time_utc"].eq(
        "2025-09-01T00:10:03Z"
    ).all()
    player = observations[
        observations["logical_market_id"].eq("player-pass")
    ].iloc[0]
    assert player["subject_stat_progress"] == 125.0

    audit = result.factor_exclusion_audit.set_index("logical_market_id")
    assert audit.loc["wrong-team-total", "exclusion_reasons"] == (
        "team_total_subject_mismatch",
    )
    assert audit.loc["unbound-player", "exclusion_reasons"] == (
        "player_subject_or_stat_progress_unbound",
    )
    assert audit.loc["wrong-period", "exclusion_reasons"] == (
        "period_mismatch",
    )
    assert result.schema_version == "FactorObservationBuildV2"
    assert result.factor_observations_sha256.startswith("sha256:")
    assert result.factor_exclusion_audit_sha256.startswith("sha256:")
    assert result.semantic_rows_sha256.startswith("sha256:")


def test_period_adapter_accepts_integer_valued_float_from_nullable_join() -> None:
    sports = _sports_rows()
    sports["quarter"] = sports["quarter"].astype(float)
    inventory = _inventory_rows().query(
        "logical_market_id == 'first-half'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'first-half'"
    ).reset_index(drop=True)

    result = build_factor_observations_v2(
        sports,
        reactions,
        inventory,
        _factor_registry(),
    )

    assert len(result.factor_observations) == 1
    assert result.factor_exclusion_audit.iloc[0]["decision"] == "INCLUDE"


def test_decimal_valued_eligible_reaction_produces_an_observation() -> None:
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    decimal_metrics = {
        "pre_event_actual_trade": Decimal("0.50"),
        "first_post_event_trade": Decimal("0.54"),
        "horizon_vwap": Decimal("0.53"),
        "horizon_last_trade": Decimal("0.54"),
        "signed_price_change": Decimal("0.04"),
        "maximum_excursion": Decimal("0.05"),
        "continuation_10_to_60": Decimal("0.01"),
        "trade_count": Decimal("2"),
        "volume": Decimal("10"),
        "staleness_seconds": Decimal("1"),
    }
    for field, value in decimal_metrics.items():
        reactions[field] = pd.Series([value], dtype=object)

    result = build_factor_observations_v2(
        _sports_rows(), reactions, inventory, _factor_registry()
    )

    assert len(result.factor_observations) == 1
    assert result.factor_exclusion_audit.iloc[0]["decision"] == "INCLUDE"
    assert result.factor_observations.iloc[0][
        "signed_price_change"
    ] == Decimal("0.04")


@pytest.mark.parametrize(
    "invalid_metric",
    [
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ],
)
def test_nonfinite_or_boolean_reaction_metric_is_rejected(
    invalid_metric: object,
) -> None:
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions["signed_price_change"] = pd.Series(
        [invalid_metric], dtype=object
    )

    result = build_factor_observations_v2(
        _sports_rows(), reactions, inventory, _factor_registry()
    )

    assert result.factor_observations.empty
    assert result.factor_exclusion_audit.iloc[0]["decision"] == "EXCLUDE"
    assert "no_actual_trade" in result.factor_exclusion_audit.iloc[0][
        "exclusion_reasons"
    ]


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("actual_observation", False, "no_actual_trade"),
        ("first_post_event_trade", None, "no_actual_trade"),
        ("order_ambiguous", True, "order_ambiguous"),
        ("contaminated", True, "contaminated"),
        ("forward_filled", True, "forward_filled"),
        ("market_source_hashes", (), "market_lineage_missing"),
    ],
)
def test_market_evidence_gates_fail_rows_into_audit(
    column: str,
    value: object,
    reason: str,
) -> None:
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions.at[0, column] = value

    result = build_factor_observations_v2(
        _sports_rows(), reactions, inventory, _factor_registry()
    )

    assert result.factor_observations.empty
    assert result.factor_exclusion_audit.iloc[0]["decision"] == "EXCLUDE"
    assert reason in result.factor_exclusion_audit.iloc[0][
        "exclusion_reasons"
    ]


def test_analysis_ineligible_reaction_path_is_audited_not_published() -> None:
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions.loc[0, "analysis_eligible"] = False
    reactions.loc[0, "analysis_exclusion_reason"] = "NON_FOCAL_OUTCOME"

    result = build_factor_observations_v2(
        _sports_rows(), reactions, inventory, _factor_registry()
    )

    assert result.factor_observations.empty
    audit = result.factor_exclusion_audit.iloc[0]
    assert audit["decision"] == "EXCLUDE"
    assert not bool(audit["reaction_analysis_eligible"])
    assert (
        audit["reaction_analysis_exclusion_reason"]
        == "NON_FOCAL_OUTCOME"
    )
    assert audit["exclusion_reasons"] == (
        "reaction_path_analysis_ineligible",
    )


def test_predicate_and_registered_exclusion_are_row_level_audited() -> None:
    registry = deepcopy(_factor_registry())
    registry[0]["exclusions"] = [
        {"field": "quarter", "operator": "eq", "value": 2}
    ]
    registry[0]["pit_fields"] = ["event_pass", "quarter"]
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)

    excluded = build_factor_observations_v2(
        _sports_rows(), reactions, inventory, registry
    )

    assert excluded.factor_observations.empty
    row = excluded.factor_exclusion_audit.iloc[0]
    assert bool(row["predicate_hit"])
    assert row["exclusion_reasons"] == ("registered_exclusion",)

    sports = _sports_rows().assign(event_pass=False)
    missed = build_factor_observations_v2(
        sports, reactions, inventory, _factor_registry()
    )
    assert missed.factor_observations.empty
    assert missed.factor_exclusion_audit.iloc[0][
        "exclusion_reasons"
    ] == ("predicate_not_matched",)


def test_missing_pit_field_is_a_data_gap_not_a_guessed_factor() -> None:
    registry = deepcopy(_factor_registry())
    registry[0]["predicate"] = [
        {"field": "route_entropy", "operator": "gt", "value": 0.5}
    ]
    registry[0]["pit_fields"] = ["route_entropy"]
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)

    result = build_factor_observations_v2(
        _sports_rows(), reactions, inventory, registry
    )

    assert result.factor_observations.empty
    assert result.factor_exclusion_audit.iloc[0][
        "exclusion_reasons"
    ] == ("pit_field_unavailable:route_entropy",)


def test_an_eligible_unmatched_play_cannot_rescue_an_ineligible_factor_hit() -> None:
    sports = _sports_rows()
    sports.loc[0, "event_eligible"] = False
    unrelated = sports.iloc[0].copy()
    unrelated["play_id"] = "play-2"
    unrelated["order_sequence"] = 11
    unrelated["event_pass"] = False
    unrelated["event_eligible"] = True
    sports = pd.concat([sports, unrelated.to_frame().T], ignore_index=True)
    inventory = _inventory_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)
    reactions = _reaction_rows().query(
        "logical_market_id == 'winner'"
    ).reset_index(drop=True)

    result = build_factor_observations_v2(
        sports, reactions, inventory, _factor_registry()
    )

    assert result.factor_observations.empty
    assert result.factor_exclusion_audit.iloc[0][
        "exclusion_reasons"
    ] == ("sports_event_ineligible",)


def test_semantic_hashes_do_not_depend_on_input_row_order() -> None:
    first = build_factor_observations_v2(
        _sports_rows(),
        _reaction_rows(),
        _inventory_rows(),
        _factor_registry(),
    )
    second = build_factor_observations_v2(
        _sports_rows().sample(frac=1, random_state=3).reset_index(drop=True),
        _reaction_rows().sample(frac=1, random_state=4).reset_index(drop=True),
        _inventory_rows().sample(frac=1, random_state=5).reset_index(drop=True),
        _factor_registry(),
    )

    assert first.semantic_rows_sha256 == second.semantic_rows_sha256
    pd.testing.assert_frame_equal(
        first.factor_observations, second.factor_observations
    )
    pd.testing.assert_frame_equal(
        first.factor_exclusion_audit, second.factor_exclusion_audit
    )


def test_semantic_hash_normalizes_parquet_nested_numpy_arrays() -> None:
    parquet_shaped = _sports_rows()
    parquet_shaped["event_flags"] = [
        np.asarray(["pass", "success"], dtype=object)
    ]
    python_shaped = _sports_rows()
    python_shaped["event_flags"] = [["pass", "success"]]

    first = build_factor_observations_v2(
        parquet_shaped,
        _reaction_rows(),
        _inventory_rows(),
        _factor_registry(),
    )
    second = build_factor_observations_v2(
        python_shaped,
        _reaction_rows(),
        _inventory_rows(),
        _factor_registry(),
    )

    assert first.semantic_rows_sha256 == second.semantic_rows_sha256
    assert first.factor_observations_sha256 == (
        second.factor_observations_sha256
    )


def test_identity_and_grain_are_fail_closed() -> None:
    reactions = _reaction_rows().drop(columns=["logical_market_id"])
    with pytest.raises(
        FactorObservationBuildError,
        match="reaction paths missing columns: logical_market_id",
    ):
        build_factor_observations_v2(
            _sports_rows(),
            reactions,
            _inventory_rows(),
            _factor_registry(),
        )

    duplicate_inventory = pd.concat(
        [_inventory_rows(), _inventory_rows().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(
        FactorObservationBuildError,
        match="contract inventory is not unique",
    ):
        build_factor_observations_v2(
            _sports_rows(),
            _reaction_rows(),
            duplicate_inventory,
            _factor_registry(),
        )


def test_predicate_fields_must_be_declared_pit_inputs() -> None:
    registry = deepcopy(_factor_registry())
    registry[0]["pit_fields"] = ["quarter"]

    with pytest.raises(
        FactorObservationBuildError,
        match="predicate fields are not declared PIT inputs: event_pass",
    ):
        build_factor_observations_v2(
            _sports_rows(),
            _reaction_rows(),
            _inventory_rows(),
            registry,
        )


def test_checked_in_factor_registry_can_be_canonically_loaded() -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "registries/factors/nfl_factor_registry_v1.json"
        ).read_text(encoding="utf-8")
    )
    registry = payload["factor_definitions"]
    result = build_factor_observations_v2(
        _sports_rows(),
        _reaction_rows(),
        _inventory_rows(),
        registry,
    )

    assert not result.factor_exclusion_audit.empty
    assert set(result.factor_exclusion_audit["factor_id"]).issubset(
        {definition["factor_id"] for definition in registry}
    )
