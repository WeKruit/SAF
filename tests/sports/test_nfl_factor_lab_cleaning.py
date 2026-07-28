from __future__ import annotations

import pandas as pd

from prediction_market.sports.nfl_factor_lab_cleaning import (
    build_sports_row_disposition_ledger,
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _row(
    play_id: int,
    *,
    description: str,
    time_of_day: str | None = "2025-09-10T17:01:00Z",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "2025_01_AAA_BBB",
        "play_id": play_id,
        "order_sequence": play_id,
        "desc": description,
        "play_type": "pass",
        "pass_attempt": 1,
        "rush_attempt": 0,
        "sack": 0,
        "punt_attempt": 0,
        "field_goal_attempt": 0,
        "kickoff_attempt": 0,
        "extra_point_attempt": 0,
        "two_point_attempt": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "timeout": 0,
        "penalty": 0,
        "replay_or_challenge": 0,
        "replay_or_challenge_result": None,
        "play_deleted": 0,
        "time_of_day": time_of_day,
    }
    row.update(overrides)
    return row


def _episode(
    play_id: int,
    *,
    nullified: bool = False,
    audit_only: bool = False,
) -> dict[str, object]:
    return {
        "game_id": "2025_01_AAA_BBB",
        "episode_id": f"episode-{play_id}",
        "play_ids": [str(play_id)],
        "episode_type": "pass",
        "nullified": nullified,
        "audit_only": audit_only,
    }


def test_ledger_preserves_every_row_and_splits_live_admin_factor_eligibility() -> None:
    rows = [
        _row(1, description="live pass"),
        _row(
            2,
            description="timeout no play",
            play_type="no_play",
            pass_attempt=0,
            timeout=1,
        ),
        _row(3, description="deleted pass", play_deleted=1),
        _row(
            4,
            description="review reversed",
            pass_attempt=0,
            play_type="no_play",
            replay_or_challenge=1,
            replay_or_challenge_result="reversed",
        ),
        _row(5, description="live but untimed", time_of_day=None),
        _row(
            6,
            description="GAME",
            time_of_day=None,
            play_type=None,
            pass_attempt=0,
        ),
        _row(
            7,
            description="END Q2",
            time_of_day=None,
            play_type=None,
            pass_attempt=0,
        ),
        _row(
            8,
            description="OT END Q4",
            time_of_day=None,
            play_type=None,
            pass_attempt=0,
        ),
    ]
    episodes = [
        _episode(1),
        _episode(2, nullified=True),
        _episode(3, audit_only=True),
        _episode(4, nullified=True),
        _episode(5),
        _episode(6),
        _episode(7),
        _episode(8),
    ]

    ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=rows,
        finalized_episode_rows=episodes,
        source_hashes={"pbp": _hash("a"), "episodes": _hash("b")},
    )

    assert len(ledger.rows) == len(rows)
    records = {record["raw_play_id"]: record for record in ledger.to_records()}
    assert records["1"] == {
        "game_id": "2025_01_AAA_BBB",
        "raw_play_id": "1",
        "canonical_play_id": "1",
        "episode_id": "episode-1",
        "row_type": "live_play",
        "live_play_outcome_eligible": True,
        "administrative_event_eligible": False,
        "factor_eligible": True,
        "disposition": "factor_eligible",
        "exclusion_reasons": [],
        "source_hash": records["1"]["source_hash"],
    }
    assert records["2"]["row_type"] == "no_play"
    assert records["2"]["live_play_outcome_eligible"] is False
    assert records["2"]["administrative_event_eligible"] is True
    assert records["2"]["factor_eligible"] is False
    assert records["2"]["disposition"] == "administrative_only"
    assert {"no_play", "episode_nullified", "administrative"} <= set(
        records["2"]["exclusion_reasons"]
    )
    assert records["3"]["disposition"] == "audit_only"
    assert {"deleted", "episode_audit_only"} <= set(
        records["3"]["exclusion_reasons"]
    )
    assert records["4"]["administrative_event_eligible"] is True
    assert {"review", "reversed"} <= set(records["4"]["exclusion_reasons"])
    assert records["5"]["live_play_outcome_eligible"] is True
    assert records["5"]["factor_eligible"] is False
    assert records["5"]["exclusion_reasons"] == ["source_time_missing"]
    for play_id in ("6", "7", "8"):
        assert records[play_id]["row_type"] == "administrative"
        assert records[play_id]["administrative_event_eligible"] is True
        assert records[play_id]["factor_eligible"] is False
        assert "administrative" in records[play_id]["exclusion_reasons"]
    assert ledger.rows_sha256.startswith("sha256:")
    assert all(record["source_hash"].startswith("sha256:") for record in records.values())


def test_dataframe_nan_values_are_adjudicated_as_missing_source_values() -> None:
    raw = pd.DataFrame(
        [
            _row(
                1,
                description="GAME",
                play_type=float("nan"),
                pass_attempt=float("nan"),
                rush_attempt=float("nan"),
                sack=float("nan"),
                punt_attempt=float("nan"),
                field_goal_attempt=float("nan"),
                kickoff_attempt=float("nan"),
                extra_point_attempt=float("nan"),
                two_point_attempt=float("nan"),
                timeout=float("nan"),
                penalty=float("nan"),
                time_of_day=float("nan"),
            )
        ]
    )

    ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=raw,
        finalized_episode_rows=[_episode(1)],
        source_hashes={"pbp": _hash("a"), "episodes": _hash("b")},
    )

    [record] = ledger.to_records()
    assert record["row_type"] == "administrative"
    assert record["administrative_event_eligible"] is True
    assert record["factor_eligible"] is False
    assert set(record["exclusion_reasons"]) == {
        "administrative",
        "source_time_missing",
    }


def test_canonical_try_kneel_and_spike_families_remain_live_plays() -> None:
    rows = [
        _row(1, description="extra point", play_type="extra_point"),
        _row(2, description="quarterback kneel", play_type="qb_kneel"),
        _row(3, description="quarterback spike", play_type="qb_spike"),
    ]
    canonical = [
        {
            "game_id": "2025_01_AAA_BBB",
            "play_id": str(play_id),
            "play_family": family,
            "source_time_start_utc": "2025-09-10T17:01:00Z",
        }
        for play_id, family in ((1, "try"), (2, "kneel"), (3, "spike"))
    ]

    ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=rows,
        finalized_episode_rows=[_episode(1), _episode(2), _episode(3)],
        canonical_event_rows=canonical,
        source_hashes={"pbp": _hash("a"), "episodes": _hash("b")},
    )

    assert [row.row_type for row in ledger.rows] == [
        "live_play",
        "live_play",
        "live_play",
    ]
    assert all(row.factor_eligible for row in ledger.rows)
