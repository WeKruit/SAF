from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_policy import (
    X15PolicyError,
    build_trade_marked_shadow_curve,
    select_earliest_landmark_signals,
)


def _prediction_rows() -> pd.DataFrame:
    source_time = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    for landmark, predictions in (
        (1, ((10, 0.006), (15, 0.007), (20, 0.0075))),
        (2, ((10, 0.012), (15, 0.018), (20, 0.019))),
        (3, ((10, 0.021), (15, 0.019), (20, 0.018))),
    ):
        for endpoint, predicted_delta in predictions:
            rows.append(
                {
                    "game_id": "2025_01_AAA_BBB",
                    "episode_id": "ep-1",
                    "venue": "polymarket",
                    "logical_market_id": "winner",
                    "outcome": "AAA",
                    "source_time_utc": source_time,
                    "model_id": "interpretable-v1",
                    "landmark_seconds": landmark,
                    "endpoint_seconds": endpoint,
                    "predicted_delta": predicted_delta,
                    "predicted_class": "UP",
                    "materiality_threshold": 0.01,
                    "target_delta": predicted_delta - 0.002,
                    "mark_landmark": 0.50,
                    "mark_endpoint": 0.50 + predicted_delta - 0.002,
                    "decision_eligible": True,
                    "target_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_selects_first_eligible_landmark_and_first_90_percent_endpoint() -> None:
    signals, audit = select_earliest_landmark_signals(_prediction_rows())

    assert len(signals) == 1
    signal = signals.iloc[0]
    assert signal["landmark_seconds"] == 2
    assert signal["endpoint_seconds"] == 15
    assert signal["direction"] == 1
    assert signal["gross_markout"] == pytest.approx(0.016)
    assert audit["selected"].sum() == 1
    assert set(audit["reason"]) >= {
        "BELOW_MATERIALITY",
        "SELECTED",
        "LATER_LANDMARK_AFTER_SELECTION",
    }


def test_rejects_inconsistent_adjacent_endpoint_directions() -> None:
    rows = _prediction_rows()
    rows.loc[rows["landmark_seconds"].eq(2), "predicted_class"] = [
        "UP",
        "DOWN",
        "UP",
    ]
    rows.loc[
        rows["landmark_seconds"].eq(2),
        "predicted_delta",
    ] = [0.012, -0.018, 0.019]

    signals, audit = select_earliest_landmark_signals(rows)

    assert signals.iloc[0]["landmark_seconds"] == 3
    assert "NO_ADJACENT_DIRECTION_CONFIRMATION" in set(audit["reason"])


def test_adjacent_confirmation_uses_all_decision_eligible_endpoints() -> None:
    rows = _prediction_rows()
    landmark_two = rows["landmark_seconds"].eq(2)
    rows.loc[
        landmark_two,
        "predicted_delta",
    ] = [0.009, 0.018, 0.005]

    signals, _ = select_earliest_landmark_signals(rows)

    assert signals.iloc[0]["landmark_seconds"] == 2
    assert signals.iloc[0]["endpoint_seconds"] == 15


def test_endpoint_selection_uses_all_decision_eligible_endpoints() -> None:
    rows = _prediction_rows()
    landmark_two = rows["landmark_seconds"].eq(2)
    rows.loc[
        landmark_two,
        "predicted_delta",
    ] = [0.00995, 0.011, 0.0105]

    signals, _ = select_earliest_landmark_signals(rows)

    assert signals.iloc[0]["landmark_seconds"] == 2
    assert signals.iloc[0]["endpoint_seconds"] == 10


def test_signal_selection_does_not_depend_on_future_target_availability() -> None:
    rows = _prediction_rows()
    chosen = rows["landmark_seconds"].eq(2) & rows["endpoint_seconds"].eq(15)
    rows.loc[chosen, "target_eligible"] = False
    rows.loc[chosen, ["target_delta", "mark_endpoint"]] = float("nan")

    signals, _ = select_earliest_landmark_signals(rows)

    assert len(signals) == 1
    signal = signals.iloc[0]
    assert signal["landmark_seconds"] == 2
    assert signal["endpoint_seconds"] == 15
    assert signal["evaluation_status"] == "CENSORED_EVALUATION"
    assert pd.isna(signal["gross_markout"])
    assert signal["evidence_classification"] == (
        "CENSORED_EVALUATION"
    )


def test_shadow_curve_skips_overlapping_positions_and_is_non_executable() -> None:
    first, _ = select_earliest_landmark_signals(_prediction_rows())
    second = first.copy()
    second["episode_id"] = "ep-2"
    second["source_time_utc"] = pd.to_datetime(
        second["source_time_utc"], utc=True
    ) + pd.Timedelta(seconds=10)
    second["outcome"] = "BBB"
    third = first.copy()
    third["episode_id"] = "ep-3"
    third["source_time_utc"] = pd.to_datetime(
        third["source_time_utc"], utc=True
    ) + pd.Timedelta(seconds=30)

    curve, audit = build_trade_marked_shadow_curve(
        pd.concat([first, second, third], ignore_index=True),
        fixed_contracts=100,
        initial_nav=10_000,
    )

    assert list(curve["episode_id"]) == ["ep-1", "ep-3"]
    assert list(curve["pnl"]) == pytest.approx([1.6, 1.6])
    assert curve.iloc[-1]["nav"] == pytest.approx(10_003.2)
    assert set(curve["evidence_classification"]) == {
        "NON_EXECUTABLE_TRADE_MARKED_DIAGNOSTIC"
    }
    assert audit.loc[audit["episode_id"].eq("ep-2"), "reason"].item() == (
        "OVERLAPPING_POSITION_SKIPPED"
    )


def test_shadow_curve_skips_censored_and_realizes_by_exit_from_initial_nav() -> None:
    selected, _ = select_earliest_landmark_signals(_prediction_rows())
    long_exit_gain = selected.copy()
    long_exit_gain["episode_id"] = "late-gain"
    long_exit_gain["logical_market_id"] = "winner-a"
    long_exit_gain["endpoint_seconds"] = 20
    long_exit_gain["gross_markout"] = 0.02

    short_exit_loss = selected.copy()
    short_exit_loss["episode_id"] = "early-loss"
    short_exit_loss["logical_market_id"] = "winner-b"
    short_exit_loss["source_time_utc"] = pd.to_datetime(
        short_exit_loss["source_time_utc"], utc=True
    ) + pd.Timedelta(seconds=1)
    short_exit_loss["endpoint_seconds"] = 10
    short_exit_loss["gross_markout"] = -0.01

    censored = selected.copy()
    censored["episode_id"] = "censored"
    censored["logical_market_id"] = "winner-c"
    censored["gross_markout"] = 0.50
    censored["evaluation_status"] = "CENSORED_EVALUATION"

    curve, audit = build_trade_marked_shadow_curve(
        pd.concat(
            [long_exit_gain, short_exit_loss, censored],
            ignore_index=True,
        ),
        fixed_contracts=100,
        initial_nav=1_000,
    )

    assert curve["episode_id"].tolist() == ["early-loss", "late-gain"]
    assert curve["nav"].tolist() == pytest.approx([999.0, 1_001.0])
    assert curve["peak_nav"].tolist() == pytest.approx([1_000.0, 1_001.0])
    assert curve["drawdown"].tolist() == pytest.approx([-0.001, 0.0])
    assert audit.loc[
        audit["episode_id"].eq("censored"), "reason"
    ].item() == "CENSORED_EVALUATION_SKIPPED"


def test_policy_rejects_holdout_rows() -> None:
    rows = _prediction_rows()
    rows["dataset_partition"] = "holdout"

    with pytest.raises(X15PolicyError, match="development"):
        select_earliest_landmark_signals(rows)
