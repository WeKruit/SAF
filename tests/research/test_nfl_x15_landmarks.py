from __future__ import annotations

import math

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_landmarks import (
    ENDPOINT_SECONDS,
    LANDMARK_SECONDS,
    MODEL_PASSTHROUGH_COLUMNS,
    NFLX15LandmarkError,
    build_nfl_x15_landmark_panel,
)


def _episodes(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "game_id": "2025_01_AAA_BBB",
        "episode_id": "episode-1",
        "beneficiary_team": "AAA",
        "finalized_interval_end_utc": "2025-09-05T00:00:00Z",
        "event_type": "touchdown",
        "game_seconds_remaining": 1200.0,
        "score_margin": 3.0,
        "reference_delta": 0.12,
        "p_after_ref": 0.52,
        "factor_id": "NFL.EVENT.TOUCHDOWN",
        "nfl_week": 1,
        "is_matched_control": False,
        "pit_strength": 0.64,
        "orientation_resolved": True,
        "contaminated": False,
        "contamination_time_utc": None,
        "censored": False,
        "censor_time_utc": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _trades(*points: tuple[str, float], **overrides: object) -> pd.DataFrame:
    rows = []
    for index, (time_value, price) in enumerate(points, start=1):
        row: dict[str, object] = {
            "trade_id": f"trade-{index}",
            "game_id": "2025_01_AAA_BBB",
            "venue": "polymarket",
            "logical_market_id": "poly:aaa-bbb:winner",
            "market_family": "moneyline",
            "outcome_team": "AAA",
            "source_time_utc": time_value,
            "price": price,
            "kind": "trade",
            "provenance": "observed",
            "orientation_resolved": True,
        }
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def _row(panel: pd.DataFrame, landmark: int, endpoint: int) -> pd.Series:
    selected = panel[
        panel["landmark_seconds"].eq(landmark)
        & panel["endpoint_seconds"].eq(endpoint)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_landmark_panel_uses_recent_unique_actual_trades_and_beneficiary_direction() -> None:
    trades = _trades(
        ("2025-09-05T00:00:00Z", 0.10),  # exactly T: not post-event
        ("2025-09-05T00:00:00.800000Z", 0.40),
        ("2025-09-05T00:00:01.800000Z", 0.45),
        ("2025-09-05T00:00:04.500000Z", 0.50),
        ("2025-09-05T00:00:09.500000Z", 0.57),
    )

    panel = build_nfl_x15_landmark_panel(trades, _episodes())

    assert LANDMARK_SECONDS == (1, 2, 3, 5, 10)
    assert ENDPOINT_SECONDS == tuple(range(5, 61, 5))
    assert len(panel) == 57
    assert not panel.duplicated(
        [
            "game_id",
            "episode_id",
            "venue",
            "logical_market_id",
            "beneficiary_team",
            "landmark_seconds",
            "endpoint_seconds",
        ]
    ).any()
    result = _row(panel, 2, 10)
    assert result["event_anchor_utc"] == pd.Timestamp(
        "2025-09-05T00:00:00Z"
    )
    assert result["reference_gap_at_landmark"] == pytest.approx(
        result["mark_landmark_price"] - 0.52
    )
    assert result["mark_landmark_trade_id"] == "trade-3"
    assert result["mark_landmark_price"] == pytest.approx(0.45)
    assert result["mark_endpoint_trade_id"] == "trade-5"
    assert result["mark_endpoint_price"] == pytest.approx(0.57)
    assert result["landmark_staleness_seconds"] == pytest.approx(0.2)
    assert result["endpoint_staleness_seconds"] == pytest.approx(0.5)
    assert result["target_delta"] == pytest.approx(0.12)
    assert result["target_orientation"] == "BENEFICIARY_OUTCOME"
    assert bool(result["decision_eligible"]) is True
    assert bool(result["target_observed"]) is True
    assert bool(result["target_eligible"]) is True
    assert bool(result["eligible"]) is True
    assert result["exclusion_reasons"] == ()
    assert result["reference_delta"] == pytest.approx(0.12)


def test_no_forward_fill_when_latest_trade_is_more_than_three_seconds_old() -> None:
    panel = build_nfl_x15_landmark_panel(
        _trades(("2025-09-05T00:00:00.500000Z", 0.4)),
        _episodes(),
    )

    result = _row(panel, 1, 5)
    assert result["mark_landmark_price"] == pytest.approx(0.4)
    assert math.isnan(result["mark_endpoint_price"])
    assert math.isnan(result["target_delta"])
    assert bool(result["decision_eligible"]) is True
    assert bool(result["target_observed"]) is False
    assert bool(result["target_eligible"]) is False
    assert bool(result["eligible"]) is False
    assert result["exclusion_reasons"] == ("endpoint_stale",)


def test_future_missing_trade_does_not_change_landmark_decision_eligibility() -> None:
    panel = build_nfl_x15_landmark_panel(
        _trades(("2025-09-05T00:00:01.500000Z", 0.4)),
        _episodes(),
    )

    result = _row(panel, 2, 10)
    assert bool(result["decision_eligible"]) is True
    assert bool(result["target_observed"]) is False
    assert bool(result["target_eligible"]) is False
    assert bool(result["eligible"]) is False
    assert math.isnan(result["target_delta"])
    assert result["exclusion_reasons"] == ("endpoint_stale",)


def test_future_contamination_does_not_change_landmark_decision_eligibility() -> None:
    panel = build_nfl_x15_landmark_panel(
        _trades(
            ("2025-09-05T00:00:01.500000Z", 0.4),
            ("2025-09-05T00:00:09.500000Z", 0.55),
        ),
        _episodes(contamination_time_utc="2025-09-05T00:00:07Z"),
    )

    result = _row(panel, 2, 10)
    assert bool(result["decision_eligible"]) is True
    assert bool(result["target_observed"]) is True
    assert bool(result["target_eligible"]) is False
    assert bool(result["eligible"]) is False
    assert math.isnan(result["target_delta"])
    assert result["exclusion_reasons"] == ("contaminated_before_endpoint",)


def test_future_endpoint_orientation_does_not_change_landmark_decision() -> None:
    trades = _trades(
        ("2025-09-05T00:00:01.500000Z", 0.4),
        ("2025-09-05T00:00:04.500000Z", 0.5),
    )
    trades.loc[trades["trade_id"].eq("trade-2"), "orientation_resolved"] = False

    result = _row(build_nfl_x15_landmark_panel(trades, _episodes()), 2, 5)

    assert bool(result["decision_eligible"]) is True
    assert bool(result["target_observed"]) is True
    assert bool(result["target_eligible"]) is False
    assert bool(result["eligible"]) is False
    assert math.isnan(result["target_delta"])
    assert result["exclusion_reasons"] == ("orientation_unresolved",)


def test_same_latest_source_timestamp_with_multiple_rows_is_order_ambiguous() -> None:
    trades = _trades(
        ("2025-09-05T00:00:01.500000Z", 0.40),
        ("2025-09-05T00:00:04.500000Z", 0.48),
        ("2025-09-05T00:00:04.500000Z", 0.49),
    )

    panel = build_nfl_x15_landmark_panel(trades, _episodes())

    result = _row(panel, 2, 5)
    assert result["mark_landmark_price"] == pytest.approx(0.4)
    assert math.isnan(result["mark_endpoint_price"])
    assert math.isnan(result["target_delta"])
    assert bool(result["decision_eligible"]) is True
    assert bool(result["target_observed"]) is False
    assert bool(result["target_eligible"]) is False
    assert bool(result["eligible"]) is False
    assert result["exclusion_reasons"] == ("endpoint_order_ambiguous",)


@pytest.mark.parametrize(
    ("episode_overrides", "trade_overrides", "expected_reason"),
    [
        ({"orientation_resolved": False}, {}, "orientation_unresolved"),
        ({}, {"orientation_resolved": False}, "orientation_unresolved"),
        ({"contaminated": True}, {}, "episode_contaminated"),
        (
            {"contamination_time_utc": "2025-09-05T00:00:07Z"},
            {},
            "contaminated_before_endpoint",
        ),
        ({"censored": True}, {}, "episode_censored"),
        (
            {"censor_time_utc": "2025-09-05T00:00:08Z"},
            {},
            "censored_before_endpoint",
        ),
    ],
)
def test_orientation_contamination_and_censoring_are_explicit_exclusions(
    episode_overrides: dict[str, object],
    trade_overrides: dict[str, object],
    expected_reason: str,
) -> None:
    trades = _trades(
        ("2025-09-05T00:00:01.500000Z", 0.40),
        ("2025-09-05T00:00:09.500000Z", 0.55),
        **trade_overrides,
    )
    panel = build_nfl_x15_landmark_panel(
        trades, _episodes(**episode_overrides)
    )

    result = _row(panel, 2, 10)
    assert bool(result["eligible"]) is False
    assert math.isnan(result["target_delta"])
    assert expected_reason in result["exclusion_reasons"]


def test_only_actual_beneficiary_moneyline_trades_define_a_market_panel() -> None:
    actual = _trades(
        ("2025-09-05T00:00:01.500000Z", 0.40),
        ("2025-09-05T00:00:04.500000Z", 0.50),
    )
    wrong_outcome = actual.assign(
        trade_id=["wrong-1", "wrong-2"], outcome_team="BBB"
    )
    other_family = actual.assign(
        trade_id=["total-1", "total-2"],
        logical_market_id="poly:aaa-bbb:total",
        market_family="total",
    )

    panel = build_nfl_x15_landmark_panel(
        pd.concat([actual, wrong_outcome, other_family], ignore_index=True),
        _episodes(),
    )

    assert set(panel["logical_market_id"]) == {"poly:aaa-bbb:winner"}
    assert set(panel["beneficiary_team"]) == {"AAA"}


def test_known_moneyline_without_beneficiary_trade_keeps_exclusion_rows() -> None:
    opponent_only = _trades(
        ("2025-09-05T00:00:01.500000Z", 0.60),
        ("2025-09-05T00:00:04.500000Z", 0.50),
        outcome_team="BBB",
    )

    panel = build_nfl_x15_landmark_panel(opponent_only, _episodes())

    assert len(panel) == 57
    result = _row(panel, 2, 5)
    assert bool(result["eligible"]) is False
    assert math.isnan(result["target_delta"])
    assert result["exclusion_reasons"] == (
        "landmark_no_actual_trade",
        "endpoint_no_actual_trade",
    )


def test_empty_market_scope_preserves_the_typed_output_schema() -> None:
    moneyline_panel = build_nfl_x15_landmark_panel(
        _trades(("2025-09-05T00:00:01Z", 0.4)),
        _episodes(),
    )
    no_moneyline = build_nfl_x15_landmark_panel(
        _trades(
            ("2025-09-05T00:00:01Z", 0.4),
            market_family="total",
        ),
        _episodes(),
    )

    assert no_moneyline.empty
    assert list(no_moneyline.columns) == list(moneyline_panel.columns)


def test_frozen_model_passthrough_columns_are_validated_and_preserved() -> None:
    assert MODEL_PASSTHROUGH_COLUMNS == (
        "nfl_week",
        "is_matched_control",
        "factor_id",
        "reference_gap_at_landmark",
        "game_seconds_remaining",
        "score_margin",
        "pit_strength",
    )
    panel = build_nfl_x15_landmark_panel(
        _trades(
            ("2025-09-05T00:00:01.500000Z", 0.40),
            ("2025-09-05T00:00:04.500000Z", 0.50),
        ),
        _episodes(
            nfl_week=7,
            is_matched_control=True,
            factor_id="NFL.CONTROL.ROUTINE",
            p_after_ref=0.445,
            game_seconds_remaining=777.0,
            score_margin=-4.0,
            pit_strength=0.73,
        ),
    )

    assert panel["nfl_week"].dtype == "int64"
    assert panel["is_matched_control"].dtype == "bool"
    assert set(panel["nfl_week"]) == {7}
    assert set(panel["is_matched_control"]) == {True}
    assert set(panel["factor_id"]) == {"NFL.CONTROL.ROUTINE"}
    eligible = panel[panel["mark_landmark_price"].notna()]
    assert eligible["reference_gap_at_landmark"].tolist() == pytest.approx(
        (eligible["mark_landmark_price"] - 0.445).tolist()
    )
    assert set(panel["game_seconds_remaining"]) == {777.0}
    assert set(panel["score_margin"]) == {-4.0}
    assert set(panel["pit_strength"]) == {0.73}


@pytest.mark.parametrize(
    "missing_column",
    tuple(
        column
        for column in MODEL_PASSTHROUGH_COLUMNS
        if column != "reference_gap_at_landmark"
    )
    + ("p_after_ref",),
)
def test_missing_frozen_model_passthrough_column_fails_closed(
    missing_column: str,
) -> None:
    episodes = _episodes().drop(columns=[missing_column])

    with pytest.raises(NFLX15LandmarkError, match=missing_column):
        build_nfl_x15_landmark_panel(
            _trades(("2025-09-05T00:00:01Z", 0.4)),
            episodes,
        )


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("nfl_week", 1.5, "nfl_week must contain integer"),
        ("nfl_week", 13, "weeks 1 through 12"),
        ("is_matched_control", "false", "is_matched_control must contain booleans"),
        ("factor_id", "", "factor_id must be non-empty"),
        (
            "p_after_ref",
            math.nan,
            "p_after_ref must be finite",
        ),
        ("pit_strength", "unknown", "pit_strength must be numeric"),
    ],
)
def test_model_passthrough_values_fail_closed(
    column: str,
    bad_value: object,
    message: str,
) -> None:
    with pytest.raises(NFLX15LandmarkError, match=message):
        build_nfl_x15_landmark_panel(
            _trades(("2025-09-05T00:00:01Z", 0.4)),
            _episodes(**{column: bad_value}),
        )


def test_contract_rejects_non_actual_trade_input_and_duplicate_episode_grain() -> None:
    with pytest.raises(NFLX15LandmarkError, match="actual observed trades"):
        build_nfl_x15_landmark_panel(
            _trades(
                ("2025-09-05T00:00:01Z", 0.4),
                provenance="derived",
            ),
            _episodes(),
        )
    with pytest.raises(NFLX15LandmarkError, match="episode grain"):
        build_nfl_x15_landmark_panel(
            _trades(("2025-09-05T00:00:01Z", 0.4)),
            pd.concat([_episodes(), _episodes()], ignore_index=True),
        )
