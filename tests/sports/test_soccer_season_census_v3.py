from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from prediction_market.sports.soccer_season_census import (
    FrozenStatsBombEventPartition,
    benchmark_loaded_statsbomb_season_reducer,
    census_loaded_statsbomb_season,
    summarize_reducer_latencies_ns,
)


def _match(match_id: int) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "home_team": {
            "home_team_id": match_id * 10 + 1,
            "home_team_name": f"Home {match_id}",
        },
        "away_team": {
            "away_team_id": match_id * 10 + 2,
            "away_team_name": f"Away {match_id}",
        },
        "home_score": 0,
        "away_score": 0,
    }


def _lineup(team_id: int) -> dict[str, Any]:
    return {
        "formation": 433,
        "lineup": [
            {
                "player": {
                    "id": team_id * 100 + player,
                    "name": f"Player {team_id}-{player}",
                },
                "position": {
                    "id": player,
                    "name": f"Position {player}",
                },
                "jersey_number": player,
            }
            for player in range(1, 12)
        ],
    }


def _row(
    *,
    match_id: int,
    index: int,
    action: str,
    period: int,
    period_clock_ms: int,
    possession: int,
    possession_team_id: int | None = None,
    team_id: int | None = None,
    off_camera: bool = False,
) -> dict[str, Any]:
    home = match_id * 10 + 1
    away = match_id * 10 + 2
    event_team = team_id or home
    possession_team = possession_team_id or home
    seconds, millisecond = divmod(period_clock_ms, 1_000)
    hour, seconds = divmod(seconds, 3_600)
    minute, second = divmod(seconds, 60)
    base = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}[period]
    result: dict[str, Any] = {
        "id": f"match-{match_id}-event-{index}",
        "index": index,
        "period": period,
        "timestamp": f"{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}",
        "minute": base + period_clock_ms // 60_000,
        "second": second,
        "type": {"id": index, "name": action},
        "team": {"id": event_team, "name": f"Team {event_team}"},
        "possession": possession,
        "possession_team": {
            "id": possession_team,
            "name": f"Team {possession_team}",
        },
    }
    if action == "Starting XI":
        result["tactics"] = _lineup(event_team)
    if action == "Pass":
        result["pass"] = {"end_location": [60.0, 40.0]}
    if off_camera:
        result["off_camera"] = True
    # Keep the otherwise-unused participant visible in every synthetic match.
    assert away != home
    return result


def _base_rows(match_id: int) -> list[dict[str, Any]]:
    home = match_id * 10 + 1
    away = match_id * 10 + 2
    specifications = (
        ("Starting XI", 1, 0, 1, home),
        ("Starting XI", 1, 0, 1, away),
        ("Half Start", 1, 0, 1, home),
        ("Half Start", 1, 0, 1, away),
    )
    return [
        _row(
            match_id=match_id,
            index=index,
            action=action,
            period=period,
            period_clock_ms=clock,
            possession=possession,
            team_id=team,
        )
        for index, (action, period, clock, possession, team) in enumerate(
            specifications,
            start=1,
        )
    ]


def _close_match(
    match_id: int,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    home = match_id * 10 + 1
    away = match_id * 10 + 2
    possession = int(rows[-1]["possession"])
    for action, period, clock, team in (
        ("Half End", 1, 2_700_000, home),
        ("Half End", 1, 2_700_000, away),
        ("Half Start", 2, 0, home),
        ("Half Start", 2, 0, away),
        ("Half End", 2, 2_700_000, home),
        ("Half End", 2, 2_700_000, away),
    ):
        rows.append(
            _row(
                match_id=match_id,
                index=len(rows) + 1,
                action=action,
                period=period,
                period_clock_ms=clock,
                possession=possession,
                team_id=team,
            )
        )
    return rows


def _partitions() -> tuple[
    tuple[dict[str, Any], ...],
    dict[int, FrozenStatsBombEventPartition],
]:
    matches = tuple(_match(match_id) for match_id in range(201, 206))
    partitions: dict[int, FrozenStatsBombEventPartition] = {}

    clean = _close_match(201, _base_rows(201))

    clock = _base_rows(202)
    clock.extend(
        (
            _row(
                match_id=202,
                index=5,
                action="Pass",
                period=1,
                period_clock_ms=10_000,
                possession=2,
            ),
            _row(
                match_id=202,
                index=6,
                action="Pass",
                period=1,
                period_clock_ms=9_000,
                possession=2,
            ),
        )
    )
    clock = _close_match(202, clock)

    possession = _base_rows(203)
    possession.extend(
        (
            _row(
                match_id=203,
                index=5,
                action="Pass",
                period=1,
                period_clock_ms=1_000,
                possession=2,
            ),
            _row(
                match_id=203,
                index=6,
                action="Pass",
                period=1,
                period_clock_ms=2_000,
                possession=1,
            ),
        )
    )
    possession = _close_match(203, possession)

    post_end = _close_match(204, _base_rows(204))
    post_end.append(
        _row(
            match_id=204,
            index=len(post_end) + 1,
            action="Pressure",
            period=2,
            period_clock_ms=2_700_001,
            possession=1,
        )
    )

    admin = _close_match(205, _base_rows(205))
    admin.append(
        _row(
            match_id=205,
            index=len(admin) + 1,
            action="Bad Behaviour",
            period=2,
            period_clock_ms=2_701_000,
            possession=1,
            off_camera=True,
        )
    )

    for offset, (match_id, rows) in enumerate(
        (
            (201, clean),
            (202, clock),
            (203, possession),
            (204, post_end),
            (205, admin),
        ),
        start=1,
    ):
        partitions[match_id] = FrozenStatsBombEventPartition(
            match_id=match_id,
            object_sha256="sha256:" + str(offset) * 64,
            fetched_at="2026-07-23T03:49:10.793545Z",
            events=tuple(rows),
        )
    return matches, partitions


@dataclass
class _Loader:
    partitions: Mapping[int, FrozenStatsBombEventPartition]
    calls: int = 0

    def __call__(self, match_id: int) -> FrozenStatsBombEventPartition:
        self.calls += 1
        return self.partitions[match_id]


def test_census_scans_every_game_twice_and_classifies_fail_closed_boundaries() -> None:
    matches, partitions = _partitions()
    loader = _Loader(partitions)

    report = census_loaded_statsbomb_season(
        matches=matches,
        load_events=loader,
        scan_runs=2,
    )

    assert report.census_version == "v1"
    assert report.reducer_version == "v3"
    assert report.scan_runs == 2
    assert report.deterministic is True
    assert report.run_summaries[0] == report.run_summaries[1]
    assert loader.calls == 10

    summary = report.run_summaries[0]
    assert summary.games_total == 5
    assert summary.completed_to_source_end == 2
    assert summary.fail_closed_games == 3
    assert summary.finished_games == 0
    assert summary.finalization_unproven == 2
    assert dict(summary.failure_category_counts) == {
        "clock_regression": 1,
        "possession_regression": 1,
        "post_period_end_active_event": 1,
    }
    assert dict(summary.native_anomaly_counts) == {
        "clock_regression": 1,
        "possession_regression": 1,
        "post_period_end_active_event": 1,
    }
    assert summary.source_events == sum(
        len(partition.events) for partition in partitions.values()
    )
    assert summary.events_adapted == summary.source_events
    assert summary.events_attempted == summary.events_reduced + 3
    assert summary.same_source_score_games_checked == 5
    assert summary.same_source_score_disagreements == 0
    assert summary.same_source_score_disagreement_match_ids == ()
    assert dict(summary.lifecycle_state_counts)["not_started"] > 0
    assert dict(summary.lifecycle_state_counts)["in_play"] > 0
    assert dict(summary.lifecycle_state_counts)["paused"] > 0
    assert dict(summary.lifecycle_state_counts).get("finished", 0) == 0
    assert summary.canonical_state_sha256.startswith("sha256:")
    assert "not an independent oracle" in report.score_evidence_boundary


def test_reducer_latency_summary_uses_nearest_rank_and_reducer_time_only() -> None:
    summary = summarize_reducer_latencies_ns((10, 20, 30, 40))

    assert summary.scope == "reduce_soccer_game_state only"
    assert summary.timer == "time.perf_counter_ns"
    assert summary.quantile_method == "nearest-rank"
    assert summary.sample_count == 4
    assert summary.total_ns == 100
    assert summary.p50_ns == 20
    assert summary.p95_ns == 40
    assert summary.p99_ns == 40
    assert summary.throughput_events_per_second == 40_000_000.0


def test_loaded_reducer_benchmark_times_every_successful_transition() -> None:
    matches, partitions = _partitions()
    clean_match = matches[0]
    clean_partition = partitions[201]

    summary = benchmark_loaded_statsbomb_season_reducer(
        matches=(clean_match,),
        load_events=_Loader({201: clean_partition}),
    )

    assert summary.scope == "reduce_soccer_game_state only"
    assert summary.sample_count == len(clean_partition.events)
    assert summary.total_ns >= summary.sample_count
    assert summary.min_ns <= summary.p50_ns <= summary.p95_ns
    assert summary.p95_ns <= summary.p99_ns <= summary.max_ns
    assert summary.throughput_events_per_second > 0
