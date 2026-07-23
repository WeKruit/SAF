from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from prediction_market.sports import nfl_season_census as census


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OBJECT_SHA256 = (
    "sha256:3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a"
)
SOURCE_VERSION = "github-release-58152862-20260212T102526Z"


def _row(
    *,
    order: int,
    play_id: int | None = None,
    game_id: str = "2025_01_AAA_BBB",
    season_type: str = "REG",
    qtr: int = 1,
    quarter_seconds: int | None = 900,
    game_seconds: int | None = 3600,
    home_team: str = "AAA",
    away_team: str = "BBB",
    posteam: str | None = "AAA",
    down: int | None = 1,
    ydstogo: int = 10,
    yardline_100: int | None = 75,
    fixed_drive: int = 1,
    play_clock: int | None = 40,
    goal_to_go: int = 0,
    home_score_before: int = 0,
    away_score_before: int = 0,
    home_score_after: int | None = None,
    away_score_after: int | None = None,
    home_timeouts: int = 3,
    away_timeouts: int = 3,
    play_type: str | None = "run",
    play_type_nfl: str | None = "RUSH",
    quarter_end: int | None = None,
    desc: str = "Synthetic play",
    first_down: int = 0,
    interception: int = 0,
    fumble_lost: int = 0,
    timeout: int = 0,
    timeout_team: str | None = None,
) -> dict[str, Any]:
    home_after = (
        home_score_before if home_score_after is None else home_score_after
    )
    away_after = (
        away_score_before if away_score_after is None else away_score_after
    )
    if posteam == home_team:
        posteam_score = home_score_before
        defteam_score = away_score_before
        posteam_score_post = home_after
        defteam_score_post = away_after
    elif posteam == away_team:
        posteam_score = away_score_before
        defteam_score = home_score_before
        posteam_score_post = away_after
        defteam_score_post = home_after
    else:
        posteam_score = None
        defteam_score = None
        posteam_score_post = None
        defteam_score_post = None
        down = None if down == 1 else down
        yardline_100 = None if down is None else yardline_100
    return {
        "game_id": game_id,
        "play_id": order if play_id is None else play_id,
        "order_sequence": order,
        "season_type": season_type,
        "qtr": qtr,
        "quarter_seconds_remaining": quarter_seconds,
        "game_seconds_remaining": game_seconds,
        "home_team": home_team,
        "away_team": away_team,
        "fixed_drive": fixed_drive,
        "goal_to_go": goal_to_go,
        "play_clock": play_clock,
        "posteam": posteam,
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "posteam_score": posteam_score,
        "defteam_score": defteam_score,
        "posteam_score_post": posteam_score_post,
        "defteam_score_post": defteam_score_post,
        "total_home_score": home_after,
        "total_away_score": away_after,
        "home_timeouts_remaining": home_timeouts,
        "away_timeouts_remaining": away_timeouts,
        "play_type": play_type,
        "play_type_nfl": play_type_nfl,
        "quarter_end": (
            int(play_type_nfl == "END_QUARTER")
            if quarter_end is None
            else quarter_end
        ),
        "desc": desc,
        "first_down": first_down,
        "interception": interception,
        "fumble_lost": fumble_lost,
        "timeout": timeout,
        "timeout_team": timeout_team,
    }


def _loaded(
    rows: list[dict[str, Any]],
    *,
    scan_runs: int = 2,
) -> census.NFLSeasonCensusReport:
    return census.census_loaded_nflverse_season(
        rows=rows,
        raw_object_sha256=RAW_OBJECT_SHA256,
        source_version=SOURCE_VERSION,
        scan_runs=scan_runs,
    )


def _audits(
    report: census.NFLSeasonCensusReport,
) -> dict[str, census.NFLFieldAudit]:
    return {audit.field: audit for audit in report.field_audits}


def test_synthetic_touchdown_pat_timeout_and_lifecycle_scan_cleanly() -> None:
    rows = [
        _row(order=10),
        _row(
            order=20,
            quarter_seconds=890,
            game_seconds=3590,
            down=2,
            home_score_after=6,
            play_type="pass",
            play_type_nfl="PASS",
            desc="Touchdown",
        ),
        _row(
            order=30,
            quarter_seconds=890,
            game_seconds=3590,
            down=None,
            yardline_100=None,
            home_score_before=6,
            home_score_after=7,
            play_type="extra_point",
            play_type_nfl="XP_KICK",
            desc="PAT good",
        ),
        _row(
            order=40,
            play_id=50,
            quarter_seconds=880,
            game_seconds=3580,
            posteam=None,
            home_score_before=7,
            home_timeouts=2,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #1 by AAA at 14:40.",
            timeout=1,
            timeout_team="AAA",
        ),
        _row(
            order=50,
            play_id=45,
            quarter_seconds=None,
            game_seconds=None,
            posteam=None,
            home_score_before=7,
            home_timeouts=2,
            play_type=None,
            play_type_nfl="COMMENT",
            desc="The game has been suspended. Weather.",
        ),
        _row(
            order=60,
            quarter_seconds=None,
            game_seconds=None,
            posteam=None,
            home_score_before=7,
            home_timeouts=2,
            play_type=None,
            play_type_nfl="COMMENT",
            desc="The game has resumed. Delay complete.",
        ),
        _row(
            order=70,
            quarter_seconds=870,
            game_seconds=3570,
            posteam="BBB",
            down=None,
            yardline_100=None,
            home_score_before=7,
            home_timeouts=2,
            play_type="kickoff",
            play_type_nfl="KICK_OFF",
            desc="Kickoff",
        ),
    ]

    report = _loaded(rows)

    assert report.deterministic is True
    assert report.scan_runs == 2
    assert report.games_total == report.completed_games == 1
    assert report.fail_closed_games == 0
    assert report.transitions == 6
    assert dict(report.lifecycle_counts) == {"resume": 1, "suspend": 1}
    assert all(audit.mismatches == 0 for audit in report.field_audits)
    assert _audits(report)["home_score"].comparisons == 7


def test_inserted_timeout_is_the_only_clock_correction_class() -> None:
    rows = [
        _row(
            order=100,
            play_id=100,
            qtr=4,
            quarter_seconds=100,
            game_seconds=100,
            home_score_before=3,
            home_timeouts=2,
        ),
        _row(
            order=110,
            play_id=200,
            qtr=4,
            quarter_seconds=110,
            game_seconds=110,
            posteam=None,
            home_score_before=3,
            home_timeouts=1,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #2 by AAA at 01:50.",
            timeout=1,
            timeout_team="AAA",
        ),
        _row(
            order=120,
            play_id=150,
            qtr=4,
            quarter_seconds=110,
            game_seconds=110,
            home_score_before=3,
            home_timeouts=1,
        ),
    ]

    report = _loaded(rows)

    assert report.fail_closed_games == 0
    assert dict(report.quality_flag_counts) == {
        "source_order_inserted_timeout": 1
    }
    correction = _audits(report)["clock_correction"]
    assert (correction.comparisons, correction.matches, correction.mismatches) == (
        2,
        2,
        0,
    )


def test_actual_adapter_does_not_call_or_inject_oracle_clock_classification() -> None:
    source = inspect.getsource(census._actual_event)

    assert "_clock_correction_expected" not in source
    assert "quality_flags=" not in source


def test_mutated_production_clock_correction_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(
            order=100,
            play_id=100,
            qtr=4,
            quarter_seconds=100,
            game_seconds=100,
            home_score_before=3,
            home_timeouts=2,
        ),
        _row(
            order=110,
            play_id=200,
            qtr=4,
            quarter_seconds=110,
            game_seconds=110,
            posteam=None,
            home_score_before=3,
            home_timeouts=1,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #2 by AAA at 01:50.",
            timeout=1,
            timeout_team="AAA",
        ),
        _row(
            order=120,
            play_id=150,
            qtr=4,
            quarter_seconds=110,
            game_seconds=110,
            home_score_before=3,
            home_timeouts=1,
        ),
    ]
    production_payload = census.nfl.nflverse_transition_payload
    mutated_calls = 0

    def mutated_payload(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal mutated_calls
        payload = production_payload(*args, **kwargs)
        if payload["clock_correction"]:
            mutated_calls += 1
            payload["clock_correction"] = False
            payload["quality_flags"] = [
                flag
                for flag in payload["quality_flags"]
                if flag != "source_order_inserted_timeout"
            ]
        return payload

    monkeypatch.setattr(
        census.nfl,
        "nflverse_transition_payload",
        mutated_payload,
    )

    report = _loaded(rows)

    assert mutated_calls == 2
    assert report.completed_games == 0
    assert report.fail_closed_games == 1
    assert report.failures[0].field == "clock_correction"
    assert report.failures[0].expected is True
    assert report.failures[0].actual is False
    assert _audits(report)["clock_correction"].mismatches == 1


def test_contextless_boundary_uses_first_complete_causal_successor() -> None:
    rows = [
        _row(
            order=10,
            qtr=1,
            quarter_seconds=10,
            game_seconds=2710,
            down=2,
            first_down=1,
            play_type="pass",
            play_type_nfl="PASS",
        ),
        _row(
            order=20,
            qtr=1,
            quarter_seconds=0,
            game_seconds=2700,
            posteam=None,
            play_type=None,
            play_type_nfl="END_QUARTER",
            desc="END QUARTER 1",
        ),
        _row(
            order=30,
            qtr=2,
            quarter_seconds=900,
            game_seconds=2700,
            posteam="BBB",
            down=None,
            yardline_100=None,
            play_type="kickoff",
            play_type_nfl="KICK_OFF",
        ),
        _row(
            order=40,
            qtr=2,
            quarter_seconds=895,
            game_seconds=2695,
            posteam="BBB",
            down=1,
            yardline_100=75,
        ),
    ]

    report = _loaded(rows)

    assert report.completed_games == 1
    assert report.fail_closed_games == 0
    assert all(audit.mismatches == 0 for audit in report.field_audits)
    assert _audits(report)["event_context_source_order_sequence"].matches == 3
    assert _audits(report)["source_window_order_sequences"].matches == 3


def test_observed_timeout_without_counter_charge_is_not_a_reducer_error() -> None:
    rows = [
        _row(
            order=10,
            qtr=4,
            quarter_seconds=120,
            game_seconds=120,
            home_score_before=20,
            away_score_before=12,
            home_timeouts=2,
            away_timeouts=0,
        ),
        _row(
            order=20,
            qtr=4,
            quarter_seconds=105,
            game_seconds=105,
            posteam=None,
            home_score_before=20,
            away_score_before=12,
            home_timeouts=2,
            away_timeouts=0,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #4 by BBB at 01:45.",
            timeout=1,
            timeout_team="BBB",
        ),
        _row(
            order=30,
            qtr=4,
            quarter_seconds=65,
            game_seconds=65,
            home_score_before=20,
            away_score_before=12,
            home_timeouts=2,
            away_timeouts=0,
        ),
    ]

    report = _loaded(rows)

    assert report.completed_games == 1
    assert report.fail_closed_games == 0
    assert _audits(report)["away_timeouts_remaining"].mismatches == 0
    assert _audits(report)["timeout_observed"].matches == 2
    assert _audits(report)["timeout_observed_team"].matches == 2
    assert _audits(report)["timeout_charge_team"].matches == 2


def test_regular_overtime_resets_to_two_timeouts() -> None:
    rows = [
        _row(
            order=1,
            qtr=4,
            quarter_seconds=0,
            game_seconds=0,
            play_type_nfl="END_QUARTER",
            home_score_before=20,
            away_score_before=20,
            home_timeouts=0,
            away_timeouts=0,
        ),
        _row(
            order=2,
            qtr=5,
            quarter_seconds=600,
            game_seconds=600,
            home_score_before=20,
            away_score_before=20,
            home_timeouts=2,
            away_timeouts=2,
        ),
    ]

    report = _loaded(rows)

    assert report.fail_closed_games == 0
    assert _audits(report)["home_timeouts_remaining"].mismatches == 0
    assert _audits(report)["away_timeouts_remaining"].mismatches == 0


def test_postseason_third_timeout_normalizes_native_minus_one_to_zero() -> None:
    game_id = "2025_20_AAA_BBB"
    common = {
        "game_id": game_id,
        "season_type": "POST",
        "home_score_before": 20,
        "away_score_before": 20,
    }
    rows = [
        _row(
            order=1,
            qtr=4,
            quarter_seconds=0,
            game_seconds=0,
            play_type_nfl="END_QUARTER",
            home_timeouts=0,
            away_timeouts=0,
            **common,
        ),
        _row(
            order=2,
            qtr=5,
            quarter_seconds=900,
            game_seconds=900,
            home_timeouts=2,
            away_timeouts=2,
            **common,
        ),
        _row(
            order=3,
            play_id=30,
            qtr=5,
            quarter_seconds=800,
            game_seconds=800,
            posteam=None,
            home_timeouts=1,
            away_timeouts=2,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #1 by AAA at 13:20.",
            timeout=1,
            timeout_team="AAA",
            **common,
        ),
        _row(
            order=4,
            play_id=25,
            qtr=5,
            quarter_seconds=800,
            game_seconds=800,
            home_timeouts=1,
            away_timeouts=2,
            **common,
        ),
        _row(
            order=5,
            play_id=50,
            qtr=5,
            quarter_seconds=700,
            game_seconds=700,
            posteam=None,
            home_timeouts=0,
            away_timeouts=2,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #2 by AAA at 11:40.",
            timeout=1,
            timeout_team="AAA",
            **common,
        ),
        _row(
            order=6,
            play_id=45,
            qtr=5,
            quarter_seconds=700,
            game_seconds=700,
            home_timeouts=0,
            away_timeouts=2,
            **common,
        ),
        _row(
            order=7,
            play_id=70,
            qtr=5,
            quarter_seconds=600,
            game_seconds=600,
            posteam=None,
            home_timeouts=-1,
            away_timeouts=2,
            play_type="no_play",
            play_type_nfl="TIMEOUT",
            desc="Timeout #3 by AAA at 10:00.",
            timeout=1,
            timeout_team="AAA",
            **common,
        ),
        _row(
            order=8,
            play_id=65,
            qtr=5,
            quarter_seconds=600,
            game_seconds=600,
            home_timeouts=-1,
            away_timeouts=2,
            **common,
        ),
    ]

    report = _loaded(rows)
    audit = _audits(report)["postseason_third_timeout_zero"]

    assert report.fail_closed_games == 0
    assert (audit.comparisons, audit.matches, audit.mismatches) == (1, 1, 0)


def test_rows_are_sorted_by_order_sequence_but_duplicates_fail_closed() -> None:
    first = _row(order=10)
    second = _row(order=20, quarter_seconds=890, game_seconds=3590)
    sorted_report = _loaded([second, first])
    assert sorted_report.fail_closed_games == 0

    duplicate = _loaded([first, {**second, "order_sequence": 10}])
    assert duplicate.completed_games == 0
    assert duplicate.fail_closed_games == 1
    assert duplicate.failures[0].field == "order_sequence"
    assert duplicate.failures[0].source_order_sequence == 10


def test_oracle_fails_closed_on_inconsistent_native_score_fields() -> None:
    first = _row(order=10, home_score_after=6)
    first["total_home_score"] = 5
    second = _row(
        order=20,
        quarter_seconds=890,
        game_seconds=3590,
        home_score_before=6,
    )

    report = _loaded([first, second])

    assert report.fail_closed_games == 1
    assert report.failures[0].field == "score_source_consistency"


def test_oracle_fails_closed_on_context_fields_without_possession() -> None:
    first = _row(order=10)
    second = _row(
        order=20,
        quarter_seconds=890,
        game_seconds=3590,
        posteam=None,
        down=2,
        ydstogo=7,
        yardline_100=60,
        play_type_nfl="END_QUARTER",
        desc="END QUARTER 1",
    )

    report = _loaded([first, second])

    assert report.fail_closed_games == 1
    assert report.failures[0].field == "context_carry_source"


def test_two_run_hash_is_deterministic_and_latency_is_reducer_only() -> None:
    rows = [
        _row(order=10),
        _row(order=20, quarter_seconds=890, game_seconds=3590, down=2),
    ]

    report = _loaded(rows)

    assert report.deterministic is True
    assert report.canonical_state_sha256.startswith("sha256:")
    assert report.reducer_latency["samples"] == 2
    assert report.reducer_latency["p50_ns"] > 0
    assert report.reducer_latency["p95_ns"] >= report.reducer_latency["p50_ns"]
    assert report.reducer_latency["p99_ns"] >= report.reducer_latency["p95_ns"]
    assert report.reducer_latency["max_ns"] >= report.reducer_latency["p99_ns"]
    assert report.reducer_latency["operations_per_second"] > 0


def test_oracle_code_does_not_call_reducer_adapter_helpers() -> None:
    source = "\n".join(
        (
            inspect.getsource(census._oracle_initial_state),
            inspect.getsource(census._oracle_transition),
        )
    )
    for forbidden in (
        "state_from_nflverse_row",
        "_snapshot_from_row",
        "nflverse_transition_payload",
        "reduce(",
    ):
        assert forbidden not in source


def test_loaded_census_requires_a_canonical_raw_digest() -> None:
    with pytest.raises(census.NFLSeasonCensusError, match="raw_object_sha256"):
        census.census_loaded_nflverse_season(
            rows=[_row(order=1), _row(order=2)],
            raw_object_sha256="not-a-digest",
            source_version=SOURCE_VERSION,
        )


def test_frozen_2025_season_has_zero_field_mismatches() -> None:
    report = census.run_frozen_nflverse_2025_census(program_root=PROJECT_ROOT)
    audits = _audits(report)

    assert report.dataset_id == "DS-NFLVERSE"
    assert report.source_version == SOURCE_VERSION
    assert report.reducer_version == "v3"
    assert report.rulebook_version == "2025"
    assert report.games_total == report.completed_games == 285
    assert report.fail_closed_games == 0, report.failures[:10]
    assert report.deterministic is True
    assert all(audit.mismatches == 0 for audit in report.field_audits)
    assert dict(report.lifecycle_counts) == {"resume": 2, "suspend": 2}
    assert dict(report.quality_flag_counts)[
        "source_order_inserted_timeout"
    ] == 1
    assert audits["postseason_third_timeout_zero"].comparisons == 1
    assert audits["postseason_third_timeout_zero"].mismatches == 0
