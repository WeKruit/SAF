from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import prediction_market.sports.nfl_factor_lab as factor_lab
from prediction_market.sports.nfl_factor_lab import (
    FactorLabInterfaceError,
    build_cross_game_bundle,
    build_dashboard_url,
    build_single_game_bundle,
    render_quarto_reports,
    verify_and_load_x13_inputs,
)
from prediction_market.sports.nfl_factor_lab_features import (
    NFL_FACTOR_LAB_PBP_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_facade_reuses_the_feature_builder_pbp_projection() -> None:
    assert factor_lab.NFL_FACTOR_LAB_PBP_COLUMNS is NFL_FACTOR_LAB_PBP_COLUMNS
    assert not hasattr(factor_lab, "_PBP_COLUMNS")
    assert {
        "defensive_two_point_attempt",
        "kicker_player_id",
        "penalty_team",
        "timeout_team",
        "punt_out_of_bounds",
        "kickoff_fair_catch",
        "fumble_recovery_1_yards",
        "offense_personnel",
        "defense_personnel",
    } <= set(NFL_FACTOR_LAB_PBP_COLUMNS)


def test_facade_exposes_v2_single_and_cross_game_bundle_builders() -> None:
    assert build_single_game_bundle is factor_lab.build_single_game_bundle
    assert build_cross_game_bundle is factor_lab.build_cross_game_bundle
    assert build_single_game_bundle.__module__.endswith(
        "nfl_factor_lab_bundles"
    )
    assert build_cross_game_bundle.__module__.endswith(
        "nfl_factor_lab_bundles"
    )


def test_default_feature_build_passes_frozen_bindings_and_explicit_cutoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = verify_and_load_x13_inputs(PROJECT_ROOT)
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_builder(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "prediction_market.sports.nfl_factor_lab_features.build_sports_feature_view",
        fake_builder,
    )

    observed = factor_lab.build_sports_feature_view(inputs)

    assert observed is sentinel
    bindings = captured["frozen_game_bindings"]
    assert len(bindings) == 20  # type: ignore[arg-type]
    assert set(captured["kickoff_times_utc"]) == set(  # type: ignore[arg-type]
        inputs.discovery_game_ids
    )


def test_default_x13_inputs_are_hash_verified_without_loading_associations() -> None:
    started = time.monotonic()
    inputs = verify_and_load_x13_inputs(PROJECT_ROOT)
    elapsed = time.monotonic() - started

    assert elapsed < 10
    assert inputs.experiment_id == "X-13"
    assert inputs.claim_boundary == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert inputs.discovery_game_count == 20
    assert inputs.holdout_game_count == 20
    assert inputs.signal_bundle_sha256 == (
        "sha256:"
        "9e4afe95b6c9dd500cfd10221575e8a546592485eb5829d6027b61ad38b397be"
    )
    assert inputs.cache_sha256 == (
        "sha256:"
        "48d1f35e0e7fc63dc1a75611da6e81b2834c69d2d7fe6be4af4d9e3074e6d62e"
    )
    assert inputs.holdout_reaction_accessed is False
    assert inputs.actual_observations_path.is_file()
    assert inputs.observed_cache_path.is_file()
    assert inputs.factor_registry_path.is_file()
    assert len(inputs.factor_definitions) >= 30


def test_dashboard_url_is_exact_and_rejects_unregistered_horizon() -> None:
    url = build_dashboard_url(
        game_id="2025_14_DAL_DET",
        episode_id="2025_14_DAL_DET:episode:17",
        market="poly:condition/AAA winner",
        delay_s=2,
        horizon_s=10,
    )

    assert url.startswith(
        "https://saf-x13-research.windy-crab-9030.chatgpt.site/?"
    )
    assert "game=2025_14_DAL_DET" in url
    assert "episode=2025_14_DAL_DET%3Aepisode%3A17" in url
    assert "market=poly%3Acondition%2FAAA+winner" in url
    assert "delay_s=2" in url
    assert "horizon_s=10" in url

    with pytest.raises(FactorLabInterfaceError, match="horizon"):
        build_dashboard_url(
            game_id="2025_14_DAL_DET",
            episode_id="episode",
            market="market",
            delay_s=0,
            horizon_s=3,
        )


def test_render_requires_pinned_quarto_and_never_fakes_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = verify_and_load_x13_inputs(PROJECT_ROOT)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    with pytest.raises(FactorLabInterfaceError, match="Quarto 1.9.38"):
        render_quarto_reports(
            inputs,
            output_root=tmp_path,
            game_id="2025_14_DAL_DET",
            factor_id=None,
            market_family="moneyline",
            venue="all",
            horizon=10,
            review_status="all",
        )
    assert not list(tmp_path.rglob("*.html"))


def test_input_verifier_rejects_registry_tampering(tmp_path: Path) -> None:
    registry = json.loads(
        (
            PROJECT_ROOT / "registries/factors/nfl_factor_registry_v1.json"
        ).read_text("utf-8")
    )
    registry["factor_definitions"][0]["target"] = "future_result"
    fake_root = tmp_path / "program"
    (fake_root / "registries/factors").mkdir(parents=True)
    (fake_root / "registries/factors/nfl_factor_registry_v1.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    with pytest.raises(FactorLabInterfaceError):
        verify_and_load_x13_inputs(fake_root)
