from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from prediction_market.sports import soccer_season_census
from prediction_market.sports.event_envelopes import (
    build_static_sport_observation_bundle,
)
from prediction_market.sports.soccer_game_state import statsbomb_event_payload
from prediction_market.sports.soccer_season_census import (
    FrozenStatsBombEventPartition,
    _event_envelope_id,
    census_loaded_statsbomb_season,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _event(
    *,
    native_id: str,
    index: int,
    minute: int,
    second: int,
    action: str,
    team_id: int,
    shot_goal: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": native_id,
        "index": index,
        "period": 1,
        "timestamp": f"00:{minute:02d}:{second:02d}.000",
        "minute": minute,
        "second": second,
        "type": {"id": 1, "name": action},
        "team": {"id": team_id, "name": f"Team {team_id}"},
        "possession": 1,
        "possession_team": {
            "id": team_id,
            "name": f"Team {team_id}",
        },
    }
    if action == "Shot":
        value["shot"] = {
            "end_location": [120.0, 40.0, 1.0],
            "outcome": {
                "id": 97 if shot_goal else 96,
                "name": "Goal" if shot_goal else "Saved",
            },
            "type": {"id": 87, "name": "Open Play"},
        }
    return value


def _match(
    match_id: int,
    *,
    home_score: int,
    away_score: int,
) -> dict[str, Any]:
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
        "home_score": home_score,
        "away_score": away_score,
    }


@dataclass
class _Loader:
    partitions: Mapping[int, FrozenStatsBombEventPartition]
    calls: int = 0

    def __call__(self, match_id: int) -> FrozenStatsBombEventPartition:
        self.calls += 1
        return self.partitions[match_id]


def test_census_event_id_is_the_exact_governed_envelope_content_id() -> None:
    raw_event = _event(
        native_id="match-100-event-1",
        index=1,
        minute=0,
        second=1,
        action="Shot",
        team_id=1001,
        shot_goal=False,
    )
    object_sha256 = "sha256:" + "a" * 64
    fetched_at = "2026-07-23T03:49:10.793545Z"
    payload = statsbomb_event_payload(
        raw_event,
        game_id="game_statsbomb_100",
    )
    bundle = build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-12",
        dataset_id="DS-STATSBOMB-OPEN",
        source_system="statsbomb",
        source_stream="events",
        raw_object_hash=object_sha256,
        raw_record_ordinals=(0,),
        partition="events-100",
        fetched_at=fetched_at,
        source_at=None,
        competition_id="cmp_statsbomb_2",
        game_id="game_statsbomb_100",
        participant_ids=(
            "participant_statsbomb_1001",
            "participant_statsbomb_1002",
        ),
        native_namespace="statsbomb.event",
        native_ids=("match-100-event-1",),
        normalized_source_sequence=1,
        normalized_payload=payload,
    )

    assert _event_envelope_id(
        payload=payload,
        object_sha256=object_sha256,
        fetched_at=fetched_at,
        match_id=100,
        home_team_id=1001,
        away_team_id=1002,
        quality_flags=(),
    ) == bundle.normalized.event_id


def test_loaded_season_census_replays_every_game_twice_deterministically() -> None:
    first_match = _match(101, home_score=1, away_score=0)
    second_match = _match(102, home_score=0, away_score=0)
    first_home = first_match["home_team"]["home_team_id"]
    second_home = second_match["home_team"]["home_team_id"]
    partitions = {
        101: FrozenStatsBombEventPartition(
            match_id=101,
            object_sha256="sha256:" + "1" * 64,
            fetched_at="2026-07-23T03:49:10.793545Z",
            events=(
                _event(
                    native_id="match-101-start",
                    index=1,
                    minute=1,
                    second=0,
                    action="Shot",
                    team_id=first_home,
                    shot_goal=False,
                ),
                _event(
                    native_id="match-101-goal",
                    index=2,
                    minute=1,
                    second=1,
                    action="Shot",
                    team_id=first_home,
                    shot_goal=True,
                ),
            ),
        ),
        102: FrozenStatsBombEventPartition(
            match_id=102,
            object_sha256="sha256:" + "2" * 64,
            fetched_at="2026-07-23T03:49:10.793545Z",
            events=(
                _event(
                    native_id="match-102-later-clock",
                    index=1,
                    minute=1,
                    second=0,
                    action="Shot",
                    team_id=second_home,
                    shot_goal=False,
                ),
                _event(
                    native_id="match-102-clock-regression",
                    index=2,
                    minute=0,
                    second=59,
                    action="Shot",
                    team_id=second_home,
                    shot_goal=False,
                ),
            ),
        ),
    }
    loader = _Loader(partitions)

    report = census_loaded_statsbomb_season(
        matches=(first_match, second_match),
        load_events=loader,
        scan_runs=2,
    )

    assert report.scan_runs == 2
    assert report.deterministic is True
    assert report.run_summaries[0] == report.run_summaries[1]
    summary = report.run_summaries[0]
    assert summary.games_total == 2
    assert summary.completed_to_source_end == 0
    assert summary.finished_games == 0
    assert summary.finalization_unproven == 0
    assert summary.fail_closed_games == 2
    assert summary.source_events == 4
    assert summary.events_adapted == 4
    assert summary.events_attempted == 2
    assert summary.events_reduced == 0
    assert summary.same_source_score_agreements == 2
    assert summary.same_source_score_disagreements == 0
    assert dict(summary.failure_category_counts) == {
        "lifecycle_transition": 2,
    }
    assert dict(summary.native_anomaly_counts) == {
        "clock_regression": 1,
    }
    assert dict(summary.quality_flag_counts) == {}
    assert summary.canonical_state_sha256.startswith("sha256:")
    assert loader.calls == 4


def test_loaded_season_census_reports_score_mismatch_without_hiding_state() -> None:
    match = _match(103, home_score=2, away_score=0)
    home_id = match["home_team"]["home_team_id"]
    loader = _Loader(
        {
            103: FrozenStatsBombEventPartition(
                match_id=103,
                object_sha256="sha256:" + "3" * 64,
                fetched_at="2026-07-23T03:49:10.793545Z",
                events=(
                    _event(
                        native_id="match-103-one-goal",
                        index=1,
                        minute=1,
                        second=0,
                        action="Shot",
                        team_id=home_id,
                        shot_goal=True,
                    ),
                ),
            )
        }
    )

    report = census_loaded_statsbomb_season(
        matches=(match,),
        load_events=loader,
        scan_runs=2,
    )

    summary = report.run_summaries[0]
    assert summary.completed_to_source_end == 0
    assert summary.fail_closed_games == 1
    assert summary.same_source_score_disagreements == 1
    assert summary.same_source_score_disagreement_match_ids == (103,)


def test_cli_fails_closed_when_deterministic_census_has_score_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match = _match(104, home_score=2, away_score=0)
    home_id = match["home_team"]["home_team_id"]
    report = census_loaded_statsbomb_season(
        matches=(match,),
        load_events=_Loader(
            {
                104: FrozenStatsBombEventPartition(
                    match_id=104,
                    object_sha256="sha256:" + "4" * 64,
                    fetched_at="2026-07-23T03:49:10.793545Z",
                    events=(
                        _event(
                            native_id="match-104-one-goal",
                            index=1,
                            minute=1,
                            second=0,
                            action="Shot",
                            team_id=home_id,
                            shot_goal=True,
                        ),
                    ),
                )
            }
        ),
        scan_runs=2,
    )
    assert report.deterministic is True

    monkeypatch.setattr(
        soccer_season_census,
        "run_frozen_statsbomb_season_census",
        lambda **_: report,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["soccer_season_census", "--program-root", str(PROJECT_ROOT)],
    )

    assert soccer_season_census.main() == 1


def test_cli_fails_closed_when_deterministic_census_has_incomplete_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match = _match(105, home_score=0, away_score=0)
    home_id = match["home_team"]["home_team_id"]
    report = census_loaded_statsbomb_season(
        matches=(match,),
        load_events=_Loader(
            {
                105: FrozenStatsBombEventPartition(
                    match_id=105,
                    object_sha256="sha256:" + "5" * 64,
                    fetched_at="2026-07-23T03:49:10.793545Z",
                    events=(
                        _event(
                            native_id="match-105-event-1",
                            index=1,
                            minute=1,
                            second=0,
                            action="Shot",
                            team_id=home_id,
                            shot_goal=False,
                        ),
                        _event(
                            native_id="match-105-event-3",
                            index=3,
                            minute=1,
                            second=1,
                            action="Shot",
                            team_id=home_id,
                            shot_goal=False,
                        ),
                    ),
                )
            }
        ),
        scan_runs=2,
    )
    assert report.deterministic is True
    assert report.run_summaries[0].completed_to_source_end == 0
    assert report.run_summaries[0].fail_closed_games == 1

    monkeypatch.setattr(
        soccer_season_census,
        "run_frozen_statsbomb_season_census",
        lambda **_: report,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["soccer_season_census", "--program-root", str(PROJECT_ROOT)],
    )

    assert soccer_season_census.main() == 1
