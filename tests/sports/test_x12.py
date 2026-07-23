from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prediction_market import contracts, experiments, program_audit
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


def _event_bytes(match: dict[str, object], *, bad_second: bool = False) -> bytes:
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
        goal: bool = False,
    ) -> dict[str, object]:
        value: dict[str, object] = {
            "id": f"{match['match_id']}-{index}",
            "index": index,
            "period": 1 if minute < 45 else 2,
            "timestamp": f"00:{minute % 45:02d}:{second:02d}.000",
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


def test_expanding_dixon_coles_reports_calibration_baseline_ci_and_transitions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)

    evaluation = x12.run_x12_walk_forward(
        loaded,
        minimum_train_matches=100,
        evaluation_match_limit=90,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        optimizer_max_iterations=120,
    )

    assert evaluation.result_label == "POC_NO_PIT_MARKET_PRIOR"
    assert evaluation.contract_result_label == "PRELIMINARY"
    assert evaluation.experiment_id == "X-12"
    assert evaluation.authorization_scope == "poc_result"
    assert evaluation.is_formal_result is False
    assert len(evaluation.predictions) == 90
    assert all(
        fold.train_max_played_at < fold.test_min_played_at
        and fold.train_match_count >= 100
        for fold in evaluation.folds
    )
    probability_columns = [f"probability_{label}" for label in x12.OUTCOME_CLASSES]
    assert np.allclose(
        evaluation.predictions[probability_columns].sum(axis=1).to_numpy(),
        np.ones(len(evaluation.predictions)),
        rtol=0,
        atol=1e-12,
    )
    metrics = evaluation.outcome_metrics
    assert metrics["classes"] == x12.OUTCOME_CLASSES
    assert metrics["bootstrap_samples_requested"] == 40
    assert metrics["bootstrap_samples_valid"] >= 20
    assert set(metrics["ovr_calibration"]) == set(x12.OUTCOME_CLASSES)
    assert metrics["market_prior"] == {
        "available": False,
        "reason": "no_point_in_time_market_prior",
    }
    comparison = metrics["simple_baseline_comparison"]
    assert comparison["available"] is True
    assert comparison["delta_definition"] == "model_minus_simple_baseline"
    assert set(comparison["delta_bootstrap_ci"]) == {"brier", "log_loss"}

    transitions = evaluation.transition_predictions
    transition_columns = [
        f"probability_{label}" for label in x12.TRANSITION_CLASSES
    ]
    assert len(transitions) == 90 * 18
    assert transitions["horizon_seconds"].eq(300).all()
    assert transitions["pit_status"].eq("offline_reconstruction_not_live_PIT").all()
    assert transitions["model_parameter_sha256"].str.match(
        r"^sha256:[0-9a-f]{64}$"
    ).all()
    assert np.allclose(
        transitions[transition_columns].sum(axis=1).to_numpy(),
        np.ones(len(transitions)),
        rtol=0,
        atol=1e-12,
    )
    assert transitions[transition_columns].ge(0).all().all()
    assert set(transitions["observed_transition"]) == set(x12.TRANSITION_CLASSES)
    assert evaluation.transition_metrics["bootstrap_samples_requested"] == 40
    assert evaluation.transition_metrics["bootstrap_samples_valid"] == 40
    assert len({fold.parameter_sha256 for fold in evaluation.folds}) > 1
    assert (
        evaluation.predictions[
            ["expected_home_goals", "expected_away_goals"]
        ]
        .drop_duplicates()
        .shape[0]
        > 1
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
    assert all(
        fold.optimizer_projected_gradient_inf_norm <= 1e-4
        and (
            fold.optimizer_initial_projected_gradient_inf_norm <= 1e-4
            or (
                fold.optimizer_objective_improvement > 0
                and fold.optimizer_parameter_displacement > 0
            )
        )
        for fold in evaluation.folds
    )
    assert any(
        fold.optimizer_objective_improvement > 0
        and fold.optimizer_parameter_displacement > 0
        for fold in evaluation.folds
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


def test_walk_forward_rejects_mutated_chronology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _load_fixture(monkeypatch, tmp_path)
    mutated_matches = loaded.matches.copy()
    mutated_matches.loc[0, "played_at"] += np.timedelta64(1, "D")

    with pytest.raises(x12.X12DataError, match="chronology"):
        x12.run_x12_walk_forward(
            replace(loaded, matches=mutated_matches),
            minimum_train_matches=100,
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
        ("elapsed_seconds", lambda value: int(value) + 1),
        ("event_index", lambda value: int(value) + 100_000),
    ],
)
def test_walk_forward_rejects_mutated_goal_timeline_before_fitting(
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
        x12.run_x12_walk_forward(
            replace(loaded, goals=mutated_goals),
            minimum_train_matches=100,
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
    evaluation = x12.run_x12_walk_forward(
        loaded,
        minimum_train_matches=100,
        evaluation_match_limit=30,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        optimizer_max_iterations=120,
    )
    real_validate_contract_v1 = contracts.validate_contract_v1
    validation_calls: list[
        tuple[Path, str, dict[str, object], contracts.ModelOutputV1]
    ] = []

    def validate_contract_v1_spy(
        program_root: str | Path,
        schema_name: str,
        document: object,
    ) -> contracts.ModelOutputV1:
        validated = real_validate_contract_v1(
            program_root,
            schema_name,
            document,
        )
        assert isinstance(document, dict)
        assert isinstance(validated, contracts.ModelOutputV1)
        validation_calls.append(
            (Path(program_root), schema_name, document, validated)
        )
        return validated

    monkeypatch.setattr(
        contracts,
        "validate_contract_v1",
        validate_contract_v1_spy,
    )

    evidence = x12.build_x12_evidence(
        loaded,
        evaluation,
        program_root=PROJECT_ROOT,
        execution_mode="bounded_smoke",
    )
    path = tmp_path / "x12_evidence.json"
    x12.write_x12_evidence(path, evidence)

    assert json.loads(path.read_text()) == evidence
    assert evidence["experiment_id"] == "X-12"
    assert evidence["authorization_scope"] == "poc_result"
    assert evidence["result_label"] == "POC_NO_PIT_MARKET_PRIOR"
    assert evidence["contract_result_label"] == "PRELIMINARY"
    assert evidence["is_formal_result"] is False
    assert evidence["formal_result_eligible"] is False
    assert evidence["market_prior"]["available"] is False
    assert evidence["model"]["optimizer"] == "SLSQP"
    assert evidence["model"]["optimizer_gradient"] == "analytic"
    assert (
        evidence["model"]["parameter_bound_machine_roundoff_tolerance_ulps"]
        == 8
    )
    assert evidence["model"]["kkt_boundary_absolute_tolerance"] == 1e-10
    assert evidence["model"]["optimizer_fail_closed_checks"] == [
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
    assert evidence["input_inventory"]["manifest_count"] == 381
    assert len(evidence["input_inventory"]["manifest_paths"]) == 381
    assert evidence["evidence_sha256"] == x12.evidence_sha256(evidence)
    assert evidence["open_gates"] == [
        "Team H lock approval",
        "point-in-time market prior unavailable",
        "StatsBomb O-004 remains research-only",
        "offline event availability is not a live PIT feed",
        "formal promotion unauthorized",
    ]
    assert all(
        abs(sum(item["probabilities"].values()) - 1.0) <= 1e-12
        for item in evidence["transition_output"]["distributions"]
    )
    assert len(validation_calls) == len(evaluation.transition_predictions)
    assert all(
        program_root == PROJECT_ROOT
        and schema_name == "model-output/v1.schema.yaml"
        and validated.experiment_id == "X-12"
        and validated.model_id == "MODEL-SOCCER-FIVE-MINUTE-TRANSITION"
        for program_root, schema_name, _, validated in validation_calls
    )
    contract_output = evidence["transition_output"]["distributions"][0][
        "contract_output"
    ]
    validated = real_validate_contract_v1(
        PROJECT_ROOT,
        "model-output/v1.schema.yaml",
        contract_output,
    )
    assert isinstance(validated, contracts.ModelOutputV1)
    assert validated.model_id == "MODEL-SOCCER-FIVE-MINUTE-TRANSITION"
    assert validated.model_version == "v1"
    assert validated.experiment_id == "X-12"
    assert validated.transition_unit == "five_minute_interval"
    assert validated.data_sha256 == loaded.inventory.inventory_sha256
    assert sum(
        (
            Decimal(value["atoms"]).scaleb(-value["scale"])
            for value in contract_output["probabilities"].values()
        ),
        start=Decimal(0),
    ) == Decimal(1)
    assert contract_output["quality_flags"] == [
        "preliminary_rules",
        "source_clock_unverified",
    ]


def _contract_transition_row() -> SimpleNamespace:
    return SimpleNamespace(
        match_id=3750000,
        prediction_at=pd.Timestamp("2015-08-01T12:00:00Z"),
        home_score_at_cutoff=0,
        away_score_at_cutoff=0,
        probability_home_goal=0.10,
        probability_away_goal=0.05,
        probability_no_goal=0.85,
        object_sha256=_digest("event-object"),
        pit_status=x12.OFFLINE_PIT_STATUS,
        model_parameter_sha256=_digest("model-parameters"),
        inventory_sha256=_digest("inventory"),
    )


@pytest.mark.parametrize(
    ("missing_foreign_key", "expected_message"),
    [
        ("experiment", "experiment X-12 is not registered"),
        (
            "model",
            (
                "model MODEL-SOCCER-FIVE-MINUTE-TRANSITION "
                "is not registered"
            ),
        ),
    ],
)
def test_transition_contract_fails_closed_on_unknown_registry_foreign_key(
    missing_foreign_key: str,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if missing_foreign_key == "experiment":
        registered = experiments.load_experiment_registry(PROJECT_ROOT)
        monkeypatch.setattr(
            experiments,
            "load_experiment_registry",
            lambda _: {
                experiment_id: card
                for experiment_id, card in registered.items()
                if experiment_id != "X-12"
            },
        )
    else:
        registered_models = program_audit.load_model_registry(PROJECT_ROOT)
        monkeypatch.setattr(
            program_audit,
            "load_model_registry",
            lambda _: tuple(
                row
                for row in registered_models
                if row.model_id
                != "MODEL-SOCCER-FIVE-MINUTE-TRANSITION"
            ),
        )

    with pytest.raises(x12.X12DataError) as exc_info:
        x12._transition_contract_output(
            _contract_transition_row(),
            evaluation=SimpleNamespace(seed=x12.X12_SEED),
            program_root=PROJECT_ROOT,
        )

    assert isinstance(exc_info.value.__cause__, contracts.ContractValidationError)
    assert expected_message in str(exc_info.value.__cause__)


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
