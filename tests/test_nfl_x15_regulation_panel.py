from __future__ import annotations

import json

import pandas as pd

from prediction_market.research.nfl_x15_regulation_panel import (
    FEATURE_BLOCKS,
    SCHEMA_VERSION,
    build_regulation_game_panel,
)


_SHA = "sha256:" + "a" * 64


def _inputs(*, include_h_trade: bool = True):
    game_id = "2025_01_AWY_HME"
    anchors = [
        pd.Timestamp("2025-09-01T00:00:01Z"),
        pd.Timestamp("2025-09-01T00:00:08Z"),
    ]
    episodes = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "episode_id": f"final-{index}",
                "episode_status": "FINAL",
                "episode_type": "PASS",
                "constituent_count": 1,
                "primary_event_id": f"event-{index}",
                "quarter": 4,
                "first_seen_interval_start": anchor - pd.Timedelta(seconds=1),
                "first_seen_interval_end": anchor,
                "final_known_interval_start": anchor - pd.Timedelta(seconds=1),
                "final_known_interval_end": anchor,
                "episode_anchor": anchor,
                "stage_b_eligible": True,
                "home_team": "HME",
                "away_team": "AWY",
                "beneficiary_team": "HME",
                "pre_home_score": 20,
                "pre_away_score": 20,
                "post_home_score": 20,
                "post_away_score": 20,
                "score_points": 0,
                "turnover": False,
                "final_outcome_tags": json.dumps(
                    ["PASS_ATTEMPT", "COMPLETE_PASS"]
                ),
                "source_hashes": json.dumps([_SHA]),
            }
            for index, anchor in enumerate(anchors, start=1)
        ]
    )
    constituents = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "event_id": f"event-{index}",
                "episode_id": f"final-{index}",
                "mapping_status": "MAPPED",
                "order_sequence": index,
            }
            for index in (1, 2)
        ]
    )
    context = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "event_id": f"event-{index}",
                "is_regulation": True,
                "pre_state_known_at": anchor - pd.Timedelta(seconds=1),
                "p_before_home": 0.51,
                "prestate_support_status": "SUPPORTED",
                "source_binding_sha256": _SHA,
            }
            for index, anchor in enumerate(anchors, start=1)
        ]
    )
    facts = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "event_id": f"event-{index}",
                "quarter": 4,
                "game_clock": "02:00",
                "game_seconds_remaining": 120,
                "home_team": "HME",
                "away_team": "AWY",
                "actor_is_home": True,
                "beneficiary_is_home": True,
                "possession_is_home": True,
                "pre_home_score": 20,
                "pre_away_score": 20,
                "score_margin_home": 0,
                "down": 3,
                "distance": 7,
                "yardline_100": 35,
                "primary_action": "PASS",
                "yards_gained": 12,
                "return_yards": 0,
                "review_result": pd.NA,
                "outcome_tags": json.dumps(
                    ["PASS_ATTEMPT", "COMPLETE_PASS"]
                ),
            }
            for index in (1, 2)
        ]
    )
    trade_rows = [
        {
            "trade_id": "trade-a",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": pd.Timestamp("2025-09-01T00:00:01.500Z"),
            "price": 0.50,
            "size": 1.0,
        },
        {
            "trade_id": "trade-b",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": pd.Timestamp("2025-09-01T00:00:02Z"),
            "price": 0.52,
            "size": 1.0,
        },
        {
            "trade_id": "trade-c",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": pd.Timestamp("2025-09-01T00:00:02Z"),
            "price": 0.56,
            "size": 3.0,
        },
    ]
    if include_h_trade:
        trade_rows.append(
            {
                "trade_id": "trade-d",
                "game_id": game_id,
                "venue": "polymarket",
                "contract_id": "home-token",
                "source_time_utc": pd.Timestamp("2025-09-01T00:00:04Z"),
                "price": 0.565,
                "size": 2.0,
            }
        )
    market_rows = pd.DataFrame(trade_rows)
    contracts = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "venue": "polymarket",
                "contract_id": "home-token",
                "contract_role": "ACTUAL_HOME_OUTCOME",
            }
        ]
    )
    source_hashes = {
        "finalized_manifest_sha256": _SHA,
        "finalized_episode_object_sha256": _SHA,
        "finalized_constituent_object_sha256": _SHA,
        "context_manifest_sha256": _SHA,
        "context_object_sha256": _SHA,
        "facts_game_manifest_sha256": _SHA,
        "facts_object_sha256": _SHA,
        "market_game_manifest_sha256": _SHA,
        "market_observations_object_sha256": _SHA,
        "market_inventory_object_sha256": _SHA,
        "cohort_authority_sha256": _SHA,
    }
    return {
        "episodes": episodes,
        "constituents": constituents,
        "context": context,
        "facts": facts,
        "market_rows": market_rows,
        "contracts": contracts,
        "cohort_row": {
            "game_id": game_id,
            "nfl_week": 1,
            "authority_sha256": _SHA,
        },
        "source_hashes": source_hashes,
    }


def test_panel_uses_latest_same_time_vwap_and_new_post_l_trade():
    result = build_regulation_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5, 10),
    )
    panel = result.panel
    first = panel.loc[
        panel["finalized_episode_id"].eq("final-1")
        & panel["endpoint_seconds"].eq(5)
    ].iloc[0]
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["mark_l_price"] == 0.55
    assert first["mark_l_observation_count"] == 2
    assert first["availability_h"] == 1
    assert first["mark_h_price"] == 0.565
    assert first["direction_h"] == "UP"
    assert first["loss_eligible"]
    assert not first["censored"]


def test_censoring_is_not_encoded_as_no_trade():
    result = build_regulation_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5, 10),
    )
    row = result.panel.loc[
        result.panel["finalized_episode_id"].eq("final-1")
        & result.panel["endpoint_seconds"].eq(10)
    ].iloc[0]
    assert row["censored"]
    assert row["censor_reason"] == "NEXT_FINALIZED_EPISODE_IN_L_H_WINDOW"
    assert not row["loss_eligible"]
    assert pd.isna(row["availability_h"])
    assert pd.isna(row["direction_h"])


def test_clean_window_without_new_trade_is_availability_zero():
    result = build_regulation_game_panel(
        **_inputs(include_h_trade=False),
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    )
    row = result.panel.loc[
        result.panel["finalized_episode_id"].eq("final-1")
    ].iloc[0]
    assert row["decision_eligible"]
    assert row["loss_eligible"]
    assert row["availability_h"] == 0
    assert pd.isna(row["direction_h"])
    assert row["mark_h_status"] == "NO_ACTUAL_TRADE"


def test_feature_contract_is_orthogonal_and_source_bound():
    result = build_regulation_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    )
    panel = result.panel
    assert panel["source_row_id"].str.fullmatch(r"sha256:[0-9a-f]{64}").all()
    assert panel["source_binding_sha256"].str.fullmatch(
        r"sha256:[0-9a-f]{64}"
    ).all()
    assert not any(
        column.startswith(("factor__", "event_tag__"))
        for column in panel.columns
    )
    assert set(FEATURE_BLOCKS["D3"]).issubset(panel.columns)
    assert panel["is_overtime"].eq(False).all()
