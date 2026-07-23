from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prediction_market.sports import x12
from prediction_market.sports.statsbomb import (
    inspect_statsbomb_event,
    inspect_statsbomb_match_index,
)
from prediction_market.static_store import StaticStoreError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _match_specs() -> list[dict[str, object]]:
    first_day = date(2015, 8, 1)
    specs: list[dict[str, object]] = []
    for match_index in range(380):
        match_day = first_day + timedelta(days=7 * (match_index // 10))
        home_team_id = 1 + match_index % 20
        away_team_id = 1 + (match_index + 7 + match_index // 10) % 20
        if away_team_id == home_team_id:
            away_team_id = 1 + away_team_id % 20
        outcome_index = match_index % 3
        home_score, away_score = (
            (2, 0)
            if outcome_index == 0
            else (1, 1)
            if outcome_index == 1
            else (0, 1)
        )
        specs.append(
            {
                "match_id": 3750000 + match_index,
                "match_date": match_day.isoformat(),
                "kick_off": f"{12 + match_index % 4:02d}:00:00.000",
                "competition": {
                    "competition_id": 2,
                    "competition_name": "Premier League",
                },
                "season": {"season_id": 27, "season_name": "2015/2016"},
                "home_team": {
                    "home_team_id": home_team_id,
                    "home_team_name": f"Team {home_team_id:02d}",
                },
                "away_team": {
                    "away_team_id": away_team_id,
                    "away_team_name": f"Team {away_team_id:02d}",
                },
                "home_score": home_score,
                "away_score": away_score,
                "match_week": 1 + match_index // 10,
            }
        )
    return specs


def _match_index_bytes(specs: list[dict[str, object]] | None = None) -> bytes:
    return json.dumps(
        specs or _match_specs(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _event_bytes(
    match: dict[str, object],
    *,
    bad_second: bool = False,
    dismissal_side: str | None = None,
) -> bytes:
    home = match["home_team"]
    away = match["away_team"]
    assert isinstance(home, dict)
    assert isinstance(away, dict)
    events: list[dict[str, object]] = []

    def event(
        *,
        index: int,
        minute: int,
        second: int,
        event_type: str,
        team: dict[str, object],
        period: int | None = None,
        period_minute: int | None = None,
        goal: bool = False,
        player_id: int | None = None,
        card: str | None = None,
    ) -> dict[str, object]:
        native_period = period if period is not None else 1 if minute < 45 else 2
        native_period_minute = (
            period_minute
            if period_minute is not None
            else minute % 45
        )
        value: dict[str, object] = {
            "id": f"{match['match_id']}-{index}",
            "index": index,
            "period": native_period,
            "timestamp": (
                f"00:{native_period_minute:02d}:{second:02d}.000"
            ),
            "minute": minute,
            "second": second,
            "type": {"id": 16 if event_type == "Shot" else 35, "name": event_type},
            "team": {
                "id": team[
                    "home_team_id" if "home_team_id" in team else "away_team_id"
                ],
                "name": team[
                    "home_team_name"
                    if "home_team_name" in team
                    else "away_team_name"
                ],
            },
        }
        if event_type == "Shot":
            value["shot"] = {
                "outcome": {"id": 97 if goal else 96, "name": "Goal" if goal else "Saved"}
            }
        if player_id is not None:
            value["player"] = {
                "id": player_id,
                "name": f"Player {player_id}",
            }
        if event_type == "Bad Behaviour":
            assert card in {"Red Card", "Second Yellow"}
            value["bad_behaviour"] = {
                "card": {
                    "id": 5 if card == "Red Card" else 6,
                    "name": card,
                }
            }
        return value

    events.append(
        event(
            index=1,
            minute=0,
            second=99 if bad_second else 0,
            event_type="Starting XI",
            team=home,
        )
    )
    events.append(
        event(index=2, minute=0, second=0, event_type="Starting XI", team=away)
    )
    index = 3
    for goal_index in range(int(match["home_score"])):
        events.append(
            event(
                index=index,
                minute=10 + goal_index * 60,
                second=5,
                event_type="Shot",
                team=home,
                goal=True,
            )
        )
        index += 1
    for goal_index in range(int(match["away_score"])):
        events.append(
            event(
                index=index,
                minute=20 + goal_index * 55,
                second=10,
                event_type="Shot",
                team=away,
                goal=True,
            )
        )
        index += 1
    if dismissal_side is not None:
        assert dismissal_side in {"home", "away"}
        team = home if dismissal_side == "home" else away
        team_id_key = (
            "home_team_id" if dismissal_side == "home" else "away_team_id"
        )
        events.append(
            event(
                index=index,
                minute=35,
                second=30,
                event_type="Bad Behaviour",
                team=team,
                player_id=int(team[team_id_key]) * 100 + 1,
                card="Red Card",
            )
        )
        index += 1
    events.extend(
        (
            event(
                index=index,
                minute=45,
                second=0,
                event_type="Half End",
                team=home,
                period=1,
                period_minute=45,
            ),
            event(
                index=index + 1,
                minute=90,
                second=0,
                event_type="Half End",
                team=home,
                period=2,
                period_minute=45,
            ),
        )
    )
    events.sort(key=lambda value: int(value["index"]))
    return json.dumps(events, sort_keys=True, separators=(",", ":")).encode()


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _verified(
    *,
    partition: str,
    payload: bytes,
    match_id: int | None = None,
) -> SimpleNamespace:
    if match_id is None:
        audit = inspect_statsbomb_match_index(payload)
    else:
        audit = inspect_statsbomb_event(payload, match_id=match_id)
    manifest = SimpleNamespace(
        dataset_id=x12.X12_DATASET_ID,
        manifest_sha256=_digest(f"manifest-{partition}"),
        object_sha256=audit.object_sha256,
        schema_fingerprint=audit.schema_fingerprint,
        upstream_partition=partition,
        coverage=(
            "competition_id=2;season_id=27;Premier League 2015/16"
            if match_id is None
            else f"competition_id=2;season_id=27;match_id={match_id}"
        ),
        object_kind="byte_exact_original",
        license_ref="O-004",
        license_status="research_only",
    )
    return SimpleNamespace(
        record=SimpleNamespace(
            source=x12.X12_SOURCE,
            dataset=x12.X12_DATASET_ID,
            version=x12.X12_STATSBOMB_VERSION,
            partition=partition,
            extension="json",
            manifest=manifest,
        ),
        object_bytes=payload,
    )


def _install_reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bad_event_match_id: int | None = None,
) -> tuple[list[Path], list[Path]]:
    specs = _match_specs()
    index_path = Path("/fake/matches-2-27.manifest.json")
    paths = [index_path]
    objects = {
        index_path: _verified(
            partition="matches-2-27",
            payload=_match_index_bytes(specs),
        )
    }
    for match in specs:
        match_id = int(match["match_id"])
        path = Path(f"/fake/events-{match_id}.manifest.json")
        paths.append(path)
        objects[path] = _verified(
            partition=f"events-{match_id}",
            payload=_event_bytes(
                match,
                bad_second=match_id == bad_event_match_id,
                dismissal_side=(
                    "home"
                    if match_id % 31 == 0
                    else "away"
                    if match_id % 37 == 0
                    else None
                ),
            ),
            match_id=match_id,
        )
    calls: list[Path] = []

    def reader(
        manifest_path: str | Path,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> SimpleNamespace:
        del store_root, program_root
        path = Path(manifest_path)
        calls.append(path)
        return objects[path]

    monkeypatch.setattr(x12, "read_verified_static_object", reader)
    return paths, calls


def _load_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> x12.X12LoadedDataset:
    paths, _ = _install_reader(monkeypatch)
    return x12.load_x12_dataset(
        store_root=tmp_path,
        program_root=tmp_path,
        manifest_paths=paths,
    )


def test_inventory_requires_index_plus_all_380_verified_event_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, calls = _install_reader(monkeypatch)

    loaded = x12.load_x12_dataset(
        store_root=tmp_path,
        program_root=tmp_path,
        manifest_paths=reversed(paths),
    )

    assert len(calls) == 381
    assert loaded.inventory.match_count == 380
    assert loaded.inventory.event_partition_count == 380
    assert loaded.inventory.team_count == 20
    assert loaded.inventory.total_events > 760
    assert len(loaded.inventory.manifest_paths) == 381
    assert loaded.inventory.manifest_paths[0].endswith(
        "matches-2-27.manifest.json"
    )
    assert loaded.inventory.inventory_sha256 == x12.inventory_sha256(
        loaded.inventory
    )
    assert loaded.chronology_sha256 == x12.chronology_sha256(loaded.matches)
    assert loaded.goal_timeline_sha256 == x12.goal_timeline_sha256(
        loaded.goals
    )
    assert loaded.dismissal_timeline_sha256 == x12.dismissal_timeline_sha256(
        loaded.dismissals
    )
    assert set(loaded.dismissals["dismissal_side"]) == {
        "home_dismissal",
        "away_dismissal",
    }
    assert loaded.matches["match_id"].nunique() == 380
    assert loaded.matches["event_manifest_sha256"].nunique() == 380
    assert loaded.matches["played_at"].dt.tz is not None
    assert (
        loaded.matches["feature_available_at"] < loaded.matches["played_at"]
    ).all()
    with pytest.raises(FrozenInstanceError):
        loaded.inventory.match_count = 0  # type: ignore[misc]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "tamper", "bad_time"])
def test_loader_fails_closed_on_incomplete_duplicate_or_invalid_inputs(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_match = 3750000 if failure == "bad_time" else None
    paths, _ = _install_reader(monkeypatch, bad_event_match_id=bad_match)
    selected: list[Path] = paths
    if failure == "missing":
        selected = paths[:-1]
    elif failure == "duplicate":
        selected = [*paths, paths[-1]]
    elif failure == "tamper":
        def tampered_reader(
            manifest_path: str | Path,
            *,
            store_root: str | Path,
            program_root: str | Path,
        ) -> SimpleNamespace:
            del manifest_path, store_root, program_root
            raise StaticStoreError("object SHA-256 does not match manifest")

        monkeypatch.setattr(x12, "read_verified_static_object", tampered_reader)

    with pytest.raises((x12.X12DataError, StaticStoreError)):
        x12.load_x12_dataset(
            store_root=tmp_path,
            program_root=tmp_path,
            manifest_paths=selected,
        )


def test_loader_rejects_event_score_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _ = _install_reader(monkeypatch)
    original_reader = x12.read_verified_static_object
    bad_path = paths[1]
    bad = original_reader(bad_path, store_root=tmp_path, program_root=tmp_path)
    events = json.loads(bad.object_bytes)
    events = [
        event
        for event in events
        if not (
            event["type"]["name"] == "Shot"
            and event.get("shot", {}).get("outcome", {}).get("name") == "Goal"
        )
    ]
    payload = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    match_id = int(str(bad.record.partition).removeprefix("events-"))
    replacement = _verified(
        partition=bad.record.partition,
        payload=payload,
        match_id=match_id,
    )

    def reader(
        manifest_path: str | Path,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> SimpleNamespace:
        value = original_reader(
            manifest_path,
            store_root=store_root,
            program_root=program_root,
        )
        return replacement if Path(manifest_path) == bad_path else value

    monkeypatch.setattr(x12, "read_verified_static_object", reader)
    with pytest.raises(x12.X12DataError, match="score"):
        x12.load_x12_dataset(
            store_root=tmp_path,
            program_root=tmp_path,
            manifest_paths=paths,
        )


def _period_timeline_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    goals = pd.DataFrame(
        [
            {
                "match_id": 1,
                "period": 1,
                "period_clock_ms": 45 * 60 * 1_000 + 57_718,
                "global_elapsed_ms": 45 * 60 * 1_000 + 57_718,
                "source_clock_ms": 45 * 60 * 1_000 + 57_718,
                "event_index": 10,
                "native_event_id": "goal-first-half-stoppage",
                "scoring_side": "home_goal",
            },
            {
                "match_id": 1,
                "period": 2,
                "period_clock_ms": 0,
                "global_elapsed_ms": 45 * 60 * 1_000 + 57_719,
                "source_clock_ms": 45 * 60 * 1_000,
                "event_index": 20,
                "native_event_id": "goal-second-half-cutoff",
                "scoring_side": "away_goal",
            },
            {
                "match_id": 1,
                "period": 2,
                "period_clock_ms": 5 * 60 * 1_000,
                "global_elapsed_ms": 50 * 60 * 1_000 + 57_719,
                "source_clock_ms": 50 * 60 * 1_000,
                "event_index": 30,
                "native_event_id": "goal-second-half-window-end",
                "scoring_side": "home_goal",
            },
            {
                "match_id": 1,
                "period": 2,
                "period_clock_ms": 10 * 60 * 1_000,
                "global_elapsed_ms": 55 * 60 * 1_000 + 57_719,
                "source_clock_ms": 55 * 60 * 1_000,
                "event_index": 40,
                "native_event_id": "goal-future",
                "scoring_side": "away_goal",
            },
        ]
    )
    dismissals = pd.DataFrame(
        columns=(
            "match_id",
            "period",
            "period_clock_ms",
            "global_elapsed_ms",
            "source_clock_ms",
            "event_index",
            "native_event_id",
            "player_id",
            "dismissal_side",
            "card",
        )
    )
    return goals, dismissals


def test_period_local_cutoff_consumes_boundary_and_excludes_first_half_stoppage() -> None:
    goals, dismissals = _period_timeline_fixture()

    state = x12._state_at_cutoff(
        goals,
        dismissals,
        period=2,
        period_clock_ms=0,
    )
    window = x12._events_in_transition_window(
        goals,
        period=2,
        cutoff_period_clock_ms=0,
    )

    assert state == (1, 1, 0, 0)
    assert window["native_event_id"].tolist() == [
        "goal-second-half-window-end"
    ]


def test_prefix_state_hash_is_invariant_to_future_event_mutation() -> None:
    goals, dismissals = _period_timeline_fixture()
    match = SimpleNamespace(
        match_id=1,
        home_team_id=10,
        away_team_id=20,
    )
    before = x12._source_state_sha256(
        match,
        goals,
        dismissals,
        period=2,
        period_clock_ms=0,
        home_score=1,
        away_score=1,
        home_dismissals=0,
        away_dismissals=0,
    )
    mutated = goals.copy()
    mutated.loc[
        mutated["native_event_id"] == "goal-future",
        "scoring_side",
    ] = "home_goal"
    after = x12._source_state_sha256(
        match,
        mutated,
        dismissals,
        period=2,
        period_clock_ms=0,
        home_score=1,
        away_score=1,
        home_dismissals=0,
        away_dismissals=0,
    )

    assert after == before


def test_dynamic_transition_reports_disjoint_calibration_and_test_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)

    evaluation = x12.run_x12_dynamic_transition(
        loaded,
        evaluation_match_limit=90,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        optimizer_max_iterations=120,
    )

    assert evaluation.result_label == "PRELIMINARY"
    assert evaluation.experiment_id == "X-12"
    assert evaluation.model_id == "MODEL-SOCCER-DYNAMIC-INTENSITY"
    assert evaluation.model_version == "v1"
    assert (
        evaluation.authorization_scope
        == "team_h_soccer_dynamic_transition_reproduction_v1"
    )
    assert evaluation.is_formal_result is False
    assert not hasattr(x12, "run_x12_walk_forward")
    assert not hasattr(evaluation, "predictions")
    assert not hasattr(evaluation, "outcome_metrics")
    assert not hasattr(evaluation, "folds")

    transitions = evaluation.transition_predictions
    transition_columns = [
        f"probability_{label}" for label in x12.TRANSITION_CLASSES
    ]
    raw_transition_columns = [
        f"raw_probability_{label}" for label in x12.TRANSITION_CLASSES
    ]
    assert len(transitions) == 90 * 18
    assert transitions["horizon_seconds"].eq(300).all()
    assert transitions["pit_status"].eq("offline_reconstruction_not_live_PIT").all()
    assert transitions["model_parameter_sha256"].str.match(
        r"^sha256:[0-9a-f]{64}$"
    ).all()
    assert transitions["dynamic_parameter_sha256"].str.match(
        r"^sha256:[0-9a-f]{64}$"
    ).all()
    assert transitions["temperature_parameter_sha256"].str.match(
        r"^sha256:[0-9a-f]{64}$"
    ).all()
    assert transitions["source_state_sha256"].str.match(
        r"^sha256:[0-9a-f]{64}$"
    ).all()
    assert transitions["home_dismissals_at_cutoff"].ge(0).all()
    assert transitions["away_dismissals_at_cutoff"].ge(0).all()
    assert np.allclose(
        transitions[transition_columns].sum(axis=1).to_numpy(),
        np.ones(len(transitions)),
        rtol=0,
        atol=1e-12,
    )
    assert transitions[transition_columns].ge(0).all().all()
    assert np.allclose(
        transitions[raw_transition_columns].sum(axis=1).to_numpy(),
        np.ones(len(transitions)),
        rtol=0,
        atol=1e-12,
    )
    assert not np.allclose(
        transitions[transition_columns].to_numpy(dtype=float),
        transitions[raw_transition_columns].to_numpy(dtype=float),
        rtol=0,
        atol=1e-15,
    )
    assert set(transitions["observed_transition"]) == set(x12.TRANSITION_CLASSES)
    transition_metrics = evaluation.transition_metrics
    assert transition_metrics["bootstrap_samples_requested"] == 40
    assert transition_metrics["bootstrap_samples_valid"] >= 20
    assert transition_metrics["probability_variant"] == (
        "temperature_calibrated"
    )
    assert transition_metrics["raw_model_metrics"]["probability_variant"] == (
        "uncalibrated"
    )
    assert set(transition_metrics["ovr_calibration"]) == set(
        x12.TRANSITION_CLASSES
    )
    assert set(
        transition_metrics["ovr_calibration_bootstrap_ci"]
    ) == set(x12.TRANSITION_CLASSES)
    static_comparison = transition_metrics["static_comparison"]
    assert static_comparison["available"] is True
    assert static_comparison["comparator"] == (
        "pregame_dixon_coles_competing_poisson"
    )
    assert static_comparison["delta_definition"] == (
        "temperature_calibrated_model_minus_static_comparator"
    )
    assert set(static_comparison["delta_bootstrap_ci"]) == {
        "brier",
        "log_loss",
    }
    split = evaluation.transition_split
    assert split.method == "frozen_chronological_date_group_holdout_50_25_25"
    assert split.base_fit_match_count == 190
    assert split.calibration_match_count == 90
    assert split.final_test_match_count == 100
    assert split.final_test_evaluated_match_count == 90
    assert split.base_fit_last_date < split.calibration_first_date
    assert split.calibration_last_date < split.final_test_first_date
    assert (
        split.dixon_coles_parameter_sha256
        == transitions["dixon_coles_parameter_sha256"].iloc[0]
    )
    assert (
        split.dynamic_parameter_sha256
        == transitions["dynamic_parameter_sha256"].iloc[0]
    )
    assert (
        split.temperature_parameter_sha256
        == transitions["temperature_parameter_sha256"].iloc[0]
    )
    assert evaluation.temperature_calibration.calibration_match_count == 90
    assert (
        evaluation.temperature_calibration.calibration_observation_count
        == 90 * 18
    )
    assert (
        transitions[
            [
                "probability_home_goal",
                "probability_away_goal",
                "probability_no_goal",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        > 1
    )


def test_dixon_coles_analytic_gradient_matches_central_difference() -> None:
    home_index = np.asarray([0, 1, 2, 3, 0, 2, 1, 3], dtype=int)
    away_index = np.asarray([1, 2, 3, 0, 2, 0, 3, 1], dtype=int)
    home_goals = np.asarray([0, 0, 1, 1, 2, 3, 1, 2], dtype=int)
    away_goals = np.asarray([0, 1, 0, 1, 1, 0, 2, 3], dtype=int)
    parameters = np.asarray(
        [0.15, -0.08, 0.04, -0.12, 0.06, 0.02, -0.03, 0.18, -0.04],
        dtype=float,
    )

    objective, gradient = x12._dixon_coles_objective_and_gradient(
        parameters,
        home_index=home_index,
        away_index=away_index,
        home_goals=home_goals,
        away_goals=away_goals,
        team_count=4,
    )
    epsilon = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_objective, _ = x12._dixon_coles_objective_and_gradient(
            plus,
            home_index=home_index,
            away_index=away_index,
            home_goals=home_goals,
            away_goals=away_goals,
            team_count=4,
        )
        minus_objective, _ = x12._dixon_coles_objective_and_gradient(
            minus,
            home_index=home_index,
            away_index=away_index,
            home_goals=home_goals,
            away_goals=away_goals,
            team_count=4,
        )
        numerical[index] = (
            plus_objective - minus_objective
        ) / (2.0 * epsilon)

    assert np.isfinite(objective)
    assert np.allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_dixon_coles_rejects_optimizer_success_without_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    train = loaded.matches.head(100)
    team_ids = tuple(
        sorted(
            set(loaded.matches["home_team_id"])
            | set(loaded.matches["away_team_id"])
        )
    )

    def false_success(
        objective: object,
        initial: np.ndarray,
        **_: object,
    ) -> SimpleNamespace:
        assert callable(objective)
        value = float(objective(initial))
        return SimpleNamespace(
            success=True,
            status=0,
            message="false convergence",
            x=initial.copy(),
            fun=value,
            nit=1,
        )

    monkeypatch.setattr(x12, "minimize", false_success)
    with pytest.raises(x12.X12DataError, match="progress"):
        x12._fit_dixon_coles(
            train,
            team_ids=team_ids,
            optimizer_max_iterations=120,
            initial_parameters=None,
        )


def test_stationary_warm_start_cannot_accept_invalid_tau_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    train = loaded.matches.head(100)
    team_ids = tuple(
        sorted(
            set(loaded.matches["home_team_id"])
            | set(loaded.matches["away_team_id"])
        )
    )
    fitted = x12._fit_dixon_coles(
        train,
        team_ids=team_ids,
        optimizer_max_iterations=120,
        initial_parameters=None,
    )
    invalid = np.full(len(fitted.parameters), 2.0, dtype=float)
    invalid[-2] = 1.0
    invalid[-1] = -0.2
    team_index = {
        team_id: index for index, team_id in enumerate(team_ids)
    }
    invalid_objective, invalid_gradient = (
        x12._dixon_coles_objective_and_gradient(
            invalid,
            home_index=np.asarray(
                [team_index[int(value)] for value in train["home_team_id"]],
                dtype=int,
            ),
            away_index=np.asarray(
                [team_index[int(value)] for value in train["away_team_id"]],
                dtype=int,
            ),
            home_goals=train["home_score"].to_numpy(dtype=int),
            away_goals=train["away_score"].to_numpy(dtype=int),
            team_count=len(team_ids),
        )
    )
    assert invalid_objective == 1e100
    assert np.any(invalid_gradient != 0)

    def invalid_tau_success(
        objective: object,
        initial: np.ndarray,
        **_: object,
    ) -> SimpleNamespace:
        assert callable(objective)
        assert np.allclose(initial, np.asarray(fitted.parameters))
        assert float(objective(invalid)) == 1e100
        return SimpleNamespace(
            success=True,
            status=0,
            message="sentinel reported as stationary",
            x=invalid.copy(),
            fun=1e100,
            nit=1,
        )

    monkeypatch.setattr(x12, "minimize", invalid_tau_success)
    with pytest.raises(x12.X12DataError, match="likelihood domain"):
        x12._fit_dixon_coles(
            train,
            team_ids=team_ids,
            optimizer_max_iterations=120,
            initial_parameters=fitted.parameters,
        )


def test_warm_start_outside_rho_bound_fails_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    train = loaded.matches.head(100)
    team_ids = tuple(
        sorted(
            set(loaded.matches["home_team_id"])
            | set(loaded.matches["away_team_id"])
        )
    )
    fitted = x12._fit_dixon_coles(
        train,
        team_ids=team_ids,
        optimizer_max_iterations=120,
        initial_parameters=None,
    )
    outside = np.asarray(fitted.parameters)
    outside[-1] = -0.200003

    def optimizer_must_not_run(*_: object, **__: object) -> object:
        raise AssertionError("optimizer reached with an out-of-bounds warm start")

    monkeypatch.setattr(x12, "minimize", optimizer_must_not_run)
    with pytest.raises(
        x12.X12DataError,
        match=r"initial parameter\[[0-9]+\].*outside bounds",
    ):
        x12._fit_dixon_coles(
            train,
            team_ids=team_ids,
            optimizer_max_iterations=120,
            initial_parameters=outside,
        )


def test_mock_optimizer_success_outside_rho_bound_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    train = loaded.matches.head(100)
    team_ids = tuple(
        sorted(
            set(loaded.matches["home_team_id"])
            | set(loaded.matches["away_team_id"])
        )
    )

    def out_of_bounds_success(
        objective: object,
        initial: np.ndarray,
        **_: object,
    ) -> SimpleNamespace:
        assert callable(objective)
        candidate = initial.copy()
        candidate[-1] = -0.200003
        return SimpleNamespace(
            success=True,
            status=0,
            message="mock success outside rho bound",
            x=candidate,
            fun=float(objective(candidate)),
            nit=1,
        )

    monkeypatch.setattr(x12, "minimize", out_of_bounds_success)
    with pytest.raises(
        x12.X12DataError,
        match=r"final parameter\[[0-9]+\].*outside bounds",
    ):
        x12._fit_dixon_coles(
            train,
            team_ids=team_ids,
            optimizer_max_iterations=120,
            initial_parameters=None,
        )


def test_projected_gradient_requires_bound_validity_before_kkt_projection() -> None:
    bounds = [(-0.2, 0.2)]
    gradient = np.asarray([1.0])

    one_ulp_below = np.asarray([np.nextafter(-0.2, -np.inf)])
    assert (
        x12._projected_gradient_inf_norm(
            one_ulp_below,
            gradient,
            bounds,
        )
        == 0.0
    )
    assert (
        x12._projected_gradient_inf_norm(
            np.asarray([-0.2 + 2e-10]),
            gradient,
            bounds,
        )
        == 1.0
    )

    with pytest.raises(
        x12.X12DataError,
        match=r"projected-gradient parameter\[0\].*outside bounds",
    ):
        x12._projected_gradient_inf_norm(
            np.asarray([-0.200003]),
            gradient,
            bounds,
        )


def test_stationary_warm_start_rejects_objective_regression_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    train = loaded.matches.head(100)
    team_ids = tuple(
        sorted(
            set(loaded.matches["home_team_id"])
            | set(loaded.matches["away_team_id"])
        )
    )
    fitted = x12._fit_dixon_coles(
        train,
        team_ids=team_ids,
        optimizer_max_iterations=120,
        initial_parameters=None,
    )

    def worse_success(
        objective: object,
        initial: np.ndarray,
        **_: object,
    ) -> SimpleNamespace:
        assert callable(objective)
        candidates = []
        for direction in (-1.0, 1.0):
            candidate = initial.copy()
            candidate[0] += direction * 0.1
            candidates.append((float(objective(candidate)), candidate))
        value, selected = max(candidates, key=lambda item: item[0])
        assert value > float(objective(initial))
        return SimpleNamespace(
            success=True,
            status=0,
            message="worse stationary result",
            x=selected,
            fun=value,
            nit=1,
        )

    monkeypatch.setattr(x12, "minimize", worse_success)
    with pytest.raises(x12.X12DataError, match="objective worsened"):
        x12._fit_dixon_coles(
            train,
            team_ids=team_ids,
            optimizer_max_iterations=120,
            initial_parameters=fitted.parameters,
        )


def test_dynamic_transition_rejects_mutated_chronology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    mutated_matches = loaded.matches.copy()
    mutated_matches.loc[0, "played_at"] += np.timedelta64(1, "D")

    with pytest.raises(x12.X12DataError, match="chronology"):
        x12.run_x12_dynamic_transition(
            replace(loaded, matches=mutated_matches),
            evaluation_match_limit=30,
            bootstrap_samples=40,
            minimum_valid_bootstrap_samples=20,
            confidence_level=0.90,
            optimizer_max_iterations=60,
        )


@pytest.mark.parametrize(
    ("column", "mutator"),
    [
        (
            "scoring_side",
            lambda value: (
                "away_goal" if value == "home_goal" else "home_goal"
            ),
        ),
        ("period_clock_ms", lambda value: int(value) + 1),
        ("event_index", lambda value: int(value) + 100_000),
    ],
)
def test_dynamic_transition_rejects_mutated_goal_timeline_before_fitting(
    column: str,
    mutator: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    mutated_goals = loaded.goals.copy()
    assert callable(mutator)
    mutated_goals.loc[0, column] = mutator(mutated_goals.loc[0, column])

    def optimizer_must_not_run(*_: object, **__: object) -> object:
        raise AssertionError("optimizer reached before goal lineage validation")

    monkeypatch.setattr(x12, "_fit_dixon_coles", optimizer_must_not_run)
    with pytest.raises(x12.X12DataError, match="goal timeline"):
        x12.run_x12_dynamic_transition(
            replace(loaded, goals=mutated_goals),
            evaluation_match_limit=30,
            bootstrap_samples=40,
            minimum_valid_bootstrap_samples=20,
            confidence_level=0.90,
            optimizer_max_iterations=60,
        )


@pytest.mark.parametrize(
    ("column", "mutator"),
    [
        (
            "dismissal_side",
            lambda value: (
                "away_dismissal"
                if value == "home_dismissal"
                else "home_dismissal"
            ),
        ),
        ("period_clock_ms", lambda value: int(value) + 1),
        ("event_index", lambda value: int(value) + 100_000),
        ("player_id", lambda value: int(value) + 100_000),
    ],
)
def test_dynamic_transition_rejects_mutated_dismissal_timeline_before_fitting(
    column: str,
    mutator: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    mutated = loaded.dismissals.copy()
    assert not mutated.empty
    assert callable(mutator)
    mutated.loc[0, column] = mutator(mutated.loc[0, column])

    def optimizer_must_not_run(*_: object, **__: object) -> object:
        raise AssertionError("optimizer reached before dismissal lineage validation")

    monkeypatch.setattr(x12, "_fit_dixon_coles", optimizer_must_not_run)
    with pytest.raises(x12.X12DataError, match="dismissal timeline"):
        x12.run_x12_dynamic_transition(
            replace(loaded, dismissals=mutated),
            evaluation_match_limit=30,
            bootstrap_samples=40,
            minimum_valid_bootstrap_samples=20,
            confidence_level=0.90,
            optimizer_max_iterations=60,
        )


def test_evidence_is_self_hashed_and_cannot_claim_formal_or_market_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    evaluation = x12.run_x12_dynamic_transition(
        loaded,
        evaluation_match_limit=30,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        optimizer_max_iterations=120,
    )
    expected_path = (
        tmp_path
        / "artifacts"
        / "game-state"
        / "soccer"
        / "x12_dynamic_transition_poc_v1.json"
    )
    with pytest.raises(x12.X12DataError, match="reproduction registration"):
        x12.build_x12_evidence(
            loaded,
            evaluation,
            program_root=PROJECT_ROOT,
            execution_mode="bounded_smoke",
        )
    assert not (
        PROJECT_ROOT / x12.X12_DYNAMIC_EVIDENCE_RELATIVE_PATH
    ).exists()
    synthetic_preflight = {
        "experiment_id": "X-12",
        "scope": (
            "team_h_soccer_dynamic_transition_reproduction_v1"
        ),
        "result_label": "PRELIMINARY",
        "dataset_ids": ["DS-STATSBOMB-OPEN"],
        "model_ids": [
            "MODEL-SOCCER-DIXON-COLES",
            "MODEL-SOCCER-DYNAMIC-INTENSITY",
        ],
        "required_lock_ids": [
            "synthetic_fixture_lock",
            "reproduction:synthetic_fixture",
        ],
        "resolved_locks": [
            {
                "id": "synthetic_fixture_lock",
                "evidence_ref": "sha256:" + "a" * 64,
            },
            {
                "id": "reproduction:synthetic_fixture",
                "evidence_ref": "sha256:" + "b" * 64,
            },
        ],
        "reproduction_lock_id": "reproduction:synthetic_fixture",
        "reproduction_spec_sha256": "sha256:" + "b" * 64,
        "code_sha256": "sha256:" + "c" * 64,
        "data_sha256": "sha256:" + "d" * 64,
        "registered_at": "2026-07-23T00:00:02Z",
        "registration_head_sha256": "sha256:" + "e" * 64,
        "status": "resolved",
    }
    historical_reference = x12._historical_x12_1x2_reference(
        PROJECT_ROOT
    )
    monkeypatch.setattr(
        x12,
        "_validate_x12_reproduction_preflight",
        lambda _: synthetic_preflight,
    )
    monkeypatch.setattr(
        x12,
        "_historical_x12_1x2_reference",
        lambda _: historical_reference,
    )
    evidence = x12.build_x12_evidence(
        loaded,
        evaluation,
        program_root=tmp_path,
        execution_mode="bounded_smoke",
    )
    path = x12.write_x12_evidence(
        program_root=tmp_path,
        evidence=evidence,
    )

    assert path == expected_path
    assert json.loads(path.read_text()) == evidence
    assert evidence["experiment_id"] == "X-12"
    assert evidence["model_id"] == "MODEL-SOCCER-DYNAMIC-INTENSITY"
    assert evidence["model_version"] == "v1"
    assert evidence["authorization_scope"] == (
        "team_h_soccer_dynamic_transition_reproduction_v1"
    )
    assert evidence["result_label"] == "PRELIMINARY"
    assert evidence["is_formal_result"] is False
    assert evidence["formal_result_eligible"] is False
    assert evidence["market_prior"]["available"] is False
    assert "outcome_evaluation" not in evidence
    assert "walk_forward" not in evidence
    base_rate_model = evidence["model"]["transition_model"][
        "base_rate_model"
    ]
    assert base_rate_model["new_1x2_output_produced"] is False
    assert base_rate_model["optimizer"] == "SLSQP"
    assert base_rate_model["optimizer_gradient"] == "analytic"
    assert (
        base_rate_model["parameter_bound_machine_roundoff_tolerance_ulps"]
        == 8
    )
    assert base_rate_model["kkt_boundary_absolute_tolerance"] == 1e-10
    assert base_rate_model["optimizer_fail_closed_checks"] == [
        (
            "initial_and_final_parameter_bounds_with_8_ulp_"
            "machine_roundoff_tolerance"
        ),
        "valid_likelihood_domain",
        "finite_objective_and_gradient",
        "objective_non_regression",
        "objective_improvement",
        "parameter_displacement",
        "projected_gradient_inf_norm_lte_1e-4",
    ]
    assert evidence["historical_1x2_reference"] == historical_reference
    assert historical_reference["file_sha256"] == (
        x12.X12_HISTORICAL_1X2_FILE_SHA256
    )
    assert historical_reference["recomputed_for_v1"] is False
    assert historical_reference["metrics_migrated_to_v1"] is False
    assert historical_reference["referenced_component"] == (
        "outcome_evaluation"
    )
    assert (
        historical_reference["historical_transition_output_reused"]
        is False
    )
    assert evidence["input_inventory"]["manifest_count"] == 381
    assert len(evidence["input_inventory"]["manifest_paths"]) == 381
    assert evidence["evidence_sha256"] == x12.evidence_sha256(evidence)
    assert evidence["open_gates"] == [
        "point-in-time market prior unavailable",
        "StatsBomb O-004 remains research-only",
        "offline event availability is not a live PIT feed",
        "transition snapshots lack real EventEnvelope state_event_id",
        "formal promotion unauthorized",
    ]
    assert all(
        abs(sum(item["probabilities"].values()) - 1.0) <= 1e-12
        for item in evidence["transition_output"]["distributions"]
    )
    transition = evidence["transition_output"]["distributions"][0]
    assert set(transition["probabilities"]) == set(x12.TRANSITION_CLASSES)
    assert set(transition["raw_probabilities"]) == set(
        x12.TRANSITION_CLASSES
    )
    calibration = evidence["model"]["transition_model"][
        "temperature_calibration"
    ]
    assert calibration["parameter_sha256"] == (
        evaluation.temperature_calibration.parameter_sha256
    )
    assert evidence["model"]["transition_model"]["split"] == (
        x12._json_ready(asdict(evaluation.transition_split))
    )
    assert transition["contract_output"] is None
    assert transition["contract_output_status"] == {
        "available": False,
        "reason": "offline_snapshot_not_bound_to_reducer_event_envelope",
    }


def test_checked_in_real_x12_poc_is_nonconstant_and_poc_only() -> None:
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "game-state"
        / "soccer"
        / "x12_real_data_poc_v0.json"
    )
    evidence = json.loads(path.read_text())
    folds = evidence["walk_forward"]["folds"]
    predictions = evidence["outcome_evaluation"]["predictions"]
    transitions = evidence["transition_output"]["distributions"]

    assert evidence["evidence_sha256"] == x12.evidence_sha256(evidence)
    assert evidence["promotion_decision"] == "POC_ONLY"
    assert evidence["authorization_scope"] == "poc_result"
    assert evidence["is_formal_result"] is False
    assert evidence["formal_result_eligible"] is False
    assert evidence["market_prior"]["available"] is False
    assert evidence["model"]["optimizer"] == "SLSQP"
    assert evidence["model"]["optimizer_gradient"] == "analytic"
    assert len(folds) == 72
    assert len({fold["parameter_sha256"] for fold in folds}) == 72
    assert (
        len(
            {
                (
                    item["expected_goals"]["home"],
                    item["expected_goals"]["away"],
                )
                for item in predictions
            }
        )
        == 280
    )
    assert (
        len(
            {
                tuple(
                    item["probabilities"][label]
                    for label in x12.OUTCOME_CLASSES
                )
                for item in predictions
            }
        )
        == 280
    )
    assert (
        len(
            {
                tuple(
                    item["probabilities"][label]
                    for label in x12.TRANSITION_CLASSES
                )
                for item in transitions
            }
        )
        == 280
    )
    assert max(
        fold["optimizer_projected_gradient_inf_norm"] for fold in folds
    ) <= 1e-4
