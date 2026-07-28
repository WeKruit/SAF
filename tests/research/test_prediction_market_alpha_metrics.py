from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from prediction_market.research import prediction_market_alpha_metrics as metrics
from prediction_market.research.prediction_market_alpha_metrics import (
    MarketMetricDefinitionV1,
    MarketMetricObservationV1,
    MetricContractError,
    ShadowNAVObservationV1,
    ShadowStrategySpecV1,
    TickDataInventoryV1,
    benjamini_hochberg,
    compare_same_episode_venues,
    game_cluster_bootstrap_probability_delta,
    leave_one_game_out_probability_delta,
    summarize_historical_probability_delta,
    summarize_shadow_nav,
)


TIME = datetime(2026, 7, 1, tzinfo=UTC)
SHA_ONE = "sha256:" + "1" * 64
REAL_SHORTLIST_RELATIVE = Path(
    "artifacts/market-observation/nfl/x13/factor-shortlists/"
    "sha256-7a0088289e580c1fc85b5133402f00fca3999bf247d7b549eadb907f9dfa2929/"
    "7a0088289e580c1fc85b5133402f00fca3999bf247d7b549eadb907f9dfa2929"
    ".shortlist-lock.json"
)
ALL_METRICS = frozenset(
    {
        "probability_delta",
        "threshold_probability",
        "continuation",
        "reversal",
        "ofi",
        "depth",
        "shadow_nav",
        "maximum_drawdown",
        "sharpe_ratio",
    }
)


def _inventory(
    *,
    capabilities: frozenset[str] = frozenset({"trade_tick"}),
    venue: str = "kalshi",
    allowed_metrics: frozenset[str] = ALL_METRICS,
) -> TickDataInventoryV1:
    return TickDataInventoryV1(
        market_id="market-1",
        venue=venue,
        source_id="source-1",
        capabilities=capabilities,
        coverage_start=TIME - timedelta(days=1),
        coverage_end=TIME + timedelta(days=1),
        timestamp_domains=("source_time", "exchange_time"),
        direction_authority="venue_reported",
        storage_refs=("s3://research/market-1/trades.parquet",),
        object_hashes=(SHA_ONE,),
        gap_status="complete",
        gap_count=0,
        license_id="license-1",
        license_status="research_only",
        allowed_metrics=allowed_metrics,
        link_status="linked",
    )


def _definition(
    *,
    kind: str = "probability_delta",
    required_capabilities: frozenset[str] = frozenset({"trade_tick"}),
) -> MarketMetricDefinitionV1:
    return MarketMetricDefinitionV1(
        metric_id="probability-delta-60s",
        version="v1",
        layer="historical_market",
        metric_kind=kind,
        formula="p(t+60s)-p(t)",
        orientation="signed_to_outcome",
        window_seconds=60,
        exclusions=("stale", "unlinked"),
        grouping=("game_id", "episode_id"),
        claim_boundary="HISTORICAL_SOURCE_TIME_ONLY",
        required_capabilities=required_capabilities,
        threshold=0.02,
    )


def _observation(
    *,
    game_id: str,
    episode_id: str,
    probability_delta: float | None,
    continuation_delta: float | None,
    metric_kind: str = "probability_delta",
    value: float | None = None,
    venue: str = "kalshi",
    market_id: str = "market-1",
    outcome: str = "home_win",
    logical_proposition_id: str | None = None,
    market_family: str = "moneyline",
    period: str = "full_game",
    line: float | None = None,
    settlement_rule: str = "nfl_moneyline_v1",
    offset: int = 0,
    eligible: bool = True,
    exclusion_reasons: tuple[str, ...] = (),
    dataset_partition: str = "historical",
) -> MarketMetricObservationV1:
    anchor = TIME + timedelta(seconds=offset)
    return MarketMetricObservationV1(
        metric_id="probability-delta-60s",
        logical_contract_id="probability-delta-60s:v1",
        metric_kind=metric_kind,
        market_id=market_id,
        venue=venue,
        game_id=game_id,
        episode_id=episode_id,
        logical_proposition_id=(
            logical_proposition_id
            if logical_proposition_id is not None
            else f"proposition-{game_id}-{outcome}"
        ),
        market_family=market_family,
        period=period,
        line=line,
        settlement_rule=settlement_rule,
        outcome=outcome,
        reference_id=f"reference-{game_id}-{episode_id}",
        source_time=anchor,
        time_anchor=anchor,
        horizon_seconds=60,
        value=probability_delta if value is None else value,
        probability_delta=probability_delta,
        continuation_delta=continuation_delta,
        eligible=eligible,
        exclusion_reasons=exclusion_reasons,
        provenance_hashes=(SHA_ONE,),
        dataset_partition=dataset_partition,
    )


def _real_shortlist_manifest() -> tuple[Path, str]:
    path = Path(__file__).resolve().parents[2] / REAL_SHORTLIST_RELATIVE
    payload = json.loads(path.read_bytes())
    return path, payload["shortlist_sha256"]


def _formal_lock_with_holdout_accessed(tmp_path: Path) -> tuple[Path, str]:
    source, _ = _real_shortlist_manifest()
    payload = json.loads(source.read_bytes())
    payload["holdout_reaction_accessed"] = True
    payload.pop("shortlist_sha256")
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(material).hexdigest()
    payload["shortlist_sha256"] = digest
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    lock_root = tmp_path / f"sha256-{digest.removeprefix('sha256:')}"
    lock_root.mkdir()
    path = (
        lock_root
        / f"{digest.removeprefix('sha256:')}.shortlist-lock.json"
    )
    path.write_bytes(raw)
    return path, digest


def _strategy(
    manifest_path: Path,
    manifest_sha256: str,
) -> ShadowStrategySpecV1:
    return ShadowStrategySpecV1(
        strategy_id="shadow-60s",
        factor_id="NFL.EVENT.EXPLOSIVE_PLAY",
        factor_version="v1",
        entry_rule="abs(delta) >= 0.02",
        exit_rule="exit at frozen horizon",
        horizon_seconds=30,
        fixed_size=10.0,
        venue="kalshi",
        market_family="moneyline",
        max_staleness_seconds=5,
        fee_bps=10.0,
        max_concurrent_positions=2,
        shortlist_manifest_path=manifest_path,
        shortlist_manifest_sha256=manifest_sha256,
        initial_nav=100.0,
    )


def _nav(
    *,
    nav: float,
    episode_id: str,
    offset: int,
    price_basis: str,
    fill_status: str,
    fees: float,
    executable: bool = False,
) -> ShadowNAVObservationV1:
    return ShadowNAVObservationV1(
        strategy_id="shadow-60s",
        game_id="g1" if offset < 120 else "g2",
        episode_id=episode_id,
        source_time=TIME + timedelta(seconds=offset),
        nav=nav,
        price_basis=price_basis,
        fees=fees,
        fill_status=fill_status,
        executable=executable,
    )


def _calibration_observation(
    *,
    game_id: str,
    reference_id: str,
    forecast_probability: float,
    realized_label: int,
    favorite_underdog_proxy: str,
    home_away_proxy: str,
    popularity_proxy: str,
    forecast_offset: int = 0,
) -> object:
    forecast_time = TIME + timedelta(seconds=forecast_offset)
    return metrics.CalibrationForecastObservationV1(
        forecast_id=f"forecast-{game_id}-{reference_id}",
        logical_contract_id="home-win-calibration:v1",
        realized_label_contract_id="home-win-calibration:v1",
        game_id=game_id,
        reference_id=reference_id,
        outcome="home_win",
        forecast_time=forecast_time,
        label_time=forecast_time + timedelta(hours=4),
        forecast_probability=forecast_probability,
        realized_label=realized_label,
        favorite_underdog_proxy=favorite_underdog_proxy,
        home_away_proxy=home_away_proxy,
        popularity_proxy=popularity_proxy,
        provenance_hashes=(SHA_ONE,),
    )


def test_inventory_and_definition_capture_full_governed_contract() -> None:
    inventory = _inventory()
    definition = _definition()

    metrics.validate_metric_inventory(definition=definition, inventory=inventory)

    assert inventory.coverage_start < inventory.coverage_end
    assert inventory.timestamp_domains == ("source_time", "exchange_time")
    assert inventory.storage_refs == ("s3://research/market-1/trades.parquet",)
    assert inventory.object_hashes == (SHA_ONE,)
    assert inventory.gap_status == "complete"
    assert inventory.license_status == "research_only"
    assert inventory.link_status == "linked"
    assert definition.version == "v1"
    assert definition.layer == "historical_market"
    assert definition.formula == "p(t+60s)-p(t)"
    assert definition.orientation == "signed_to_outcome"
    assert definition.window_seconds == 60
    assert definition.grouping == ("game_id", "episode_id")


def test_capabilities_are_independent_not_linear_data_grades() -> None:
    trade_definition = _definition(
        required_capabilities=frozenset({"trade_tick"})
    )
    l2_definition = _definition(
        kind="ofi", required_capabilities=frozenset({"l2"})
    )

    with pytest.raises(MetricContractError, match="trade_tick"):
        metrics.validate_metric_inventory(
            definition=trade_definition,
            inventory=_inventory(capabilities=frozenset({"candle_l1"})),
        )
    with pytest.raises(MetricContractError, match="l2"):
        metrics.validate_metric_inventory(
            definition=l2_definition,
            inventory=_inventory(capabilities=frozenset({"execution"})),
        )

    metrics.validate_metric_inventory(
        definition=l2_definition,
        inventory=_inventory(capabilities=frozenset({"l2"})),
    )


def test_inventory_license_link_metric_and_gap_contracts_fail_closed() -> None:
    definition = _definition()
    inventory = _inventory()

    with pytest.raises(MetricContractError, match="license"):
        metrics.validate_metric_inventory(
            definition=definition,
            inventory=replace(inventory, license_status="pending"),
        )
    with pytest.raises(MetricContractError, match="link_status"):
        metrics.validate_metric_inventory(
            definition=definition,
            inventory=replace(inventory, link_status="unlinked"),
        )
    with pytest.raises(MetricContractError, match="allowed_metrics"):
        metrics.validate_metric_inventory(
            definition=definition,
            inventory=replace(inventory, allowed_metrics=frozenset({"depth"})),
        )
    with pytest.raises(MetricContractError, match="gap_count"):
        replace(inventory, gap_status="complete", gap_count=1)


def test_observation_canonical_fields_drive_historical_probability_metrics() -> None:
    observations = (
        _observation(
            game_id="g1",
            episode_id="e1",
            probability_delta=0.03,
            continuation_delta=0.01,
        ),
        _observation(
            game_id="g1",
            episode_id="e2",
            probability_delta=-0.04,
            continuation_delta=0.02,
        ),
        _observation(
            game_id="g2",
            episode_id="e3",
            probability_delta=0.01,
            continuation_delta=-0.02,
        ),
        _observation(
            game_id="g2",
            episode_id="excluded",
            probability_delta=0.99,
            continuation_delta=0.01,
            eligible=False,
            exclusion_reasons=("stale",),
        ),
    )

    summary = summarize_historical_probability_delta(
        observations=observations,
        definition=_definition(),
        inventory=_inventory(),
    )

    assert summary["claim_boundary"] == "HISTORICAL_SOURCE_TIME_ONLY"
    assert summary["probability_delta_distribution"] == (-0.04, 0.01, 0.03)
    assert summary["threshold_probability"] == pytest.approx(2 / 3)
    assert summary["continuation_probability"] == pytest.approx(1 / 3)
    assert summary["reversal_probability"] == pytest.approx(2 / 3)
    assert summary["excluded_observation_count"] == 1


def test_observation_aliases_contract_identity_horizon_coverage_and_holdout_fail_closed() -> None:
    observation = _observation(
        game_id="g1",
        episode_id="e1",
        probability_delta=0.03,
        continuation_delta=0.01,
    )

    with pytest.raises(MetricContractError, match="value.*probability_delta"):
        replace(observation, value=0.04)
    with pytest.raises(MetricContractError, match="time_anchor.*source_time"):
        replace(observation, time_anchor=TIME + timedelta(seconds=1))
    with pytest.raises(MetricContractError, match="logical_contract_id"):
        summarize_historical_probability_delta(
            observations=(replace(observation, logical_contract_id="other:v1"),),
            definition=_definition(),
            inventory=_inventory(),
        )
    with pytest.raises(MetricContractError, match="horizon_seconds"):
        summarize_historical_probability_delta(
            observations=(replace(observation, horizon_seconds=30),),
            definition=_definition(),
            inventory=_inventory(),
        )
    with pytest.raises(MetricContractError, match="coverage"):
        summarize_historical_probability_delta(
            observations=(replace(observation, source_time=TIME + timedelta(days=2), time_anchor=TIME + timedelta(days=2)),),
            definition=_definition(),
            inventory=_inventory(),
        )
    with pytest.raises(MetricContractError, match="holdout"):
        replace(observation, dataset_partition="holdout")


def test_ofi_and_depth_have_independent_non_probability_value_semantics() -> None:
    ofi = MarketMetricObservationV1(
        metric_id="ofi-5s",
        logical_contract_id="ofi-5s:v1",
        metric_kind="ofi",
        market_id="market-1",
        venue="kalshi",
        game_id="g1",
        episode_id="e1",
        logical_proposition_id="proposition-g1-home-win",
        market_family="moneyline",
        period="full_game",
        line=None,
        settlement_rule="nfl_moneyline_v1",
        outcome="home_win",
        reference_id="reference-g1-e1",
        source_time=TIME,
        time_anchor=TIME,
        horizon_seconds=5,
        value=12.5,
        probability_delta=None,
        continuation_delta=None,
        eligible=True,
        exclusion_reasons=(),
        provenance_hashes=(SHA_ONE,),
    )
    depth = replace(
        ofi,
        metric_id="depth-5s",
        logical_contract_id="depth-5s:v1",
        metric_kind="depth",
        value=250.0,
    )

    assert ofi.value == pytest.approx(12.5)
    assert ofi.probability_delta is None
    assert depth.value == pytest.approx(250.0)
    with pytest.raises(MetricContractError, match=r"probability.*\[-1, 1\]"):
        _observation(
            game_id="g1",
            episode_id="invalid-probability",
            probability_delta=1.1,
            continuation_delta=None,
        )
    with pytest.raises(MetricContractError, match="depth.*nonnegative"):
        replace(depth, value=-1.0)


def test_game_cluster_bootstrap_and_loo_operate_on_whole_eligible_games() -> None:
    observations = (
        _observation(
            game_id="g1",
            episode_id="e1",
            probability_delta=0.04,
            continuation_delta=None,
        ),
        _observation(
            game_id="g1",
            episode_id="e2",
            probability_delta=0.02,
            continuation_delta=None,
        ),
        _observation(
            game_id="g2",
            episode_id="e3",
            probability_delta=-0.02,
            continuation_delta=None,
        ),
        _observation(
            game_id="g3",
            episode_id="excluded",
            probability_delta=0.99,
            continuation_delta=None,
            eligible=False,
            exclusion_reasons=("unlinked",),
        ),
    )

    first = game_cluster_bootstrap_probability_delta(
        observations, draw_count=40, seed=7
    )
    second = game_cluster_bootstrap_probability_delta(
        observations, draw_count=40, seed=7
    )
    loo = leave_one_game_out_probability_delta(observations)

    assert first == second
    assert first["cluster_unit"] == "game_id"
    assert first["game_count"] == 2
    assert first["point_estimate"] == pytest.approx(0.005)
    assert len(first["bootstrap_draws"]) == 40
    assert {row["omitted_game_id"] for row in loo["estimates"]} == {"g1", "g2"}
    assert loo["estimates"][0]["point_estimate"] == pytest.approx(-0.02)


def test_bh_and_same_episode_dual_venue_pairing_are_inner_and_outcome_exact() -> None:
    q_values = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
    paired = compare_same_episode_venues(
        observations=(
            _observation(
                game_id="g1",
                episode_id="e1",
                probability_delta=0.02,
                continuation_delta=None,
                venue="kalshi",
            ),
            _observation(
                game_id="g1",
                episode_id="e1",
                probability_delta=0.05,
                continuation_delta=None,
                venue="polymarket",
                market_id="market-2",
            ),
            _observation(
                game_id="g1",
                episode_id="e2",
                probability_delta=0.04,
                continuation_delta=None,
                venue="kalshi",
            ),
            _observation(
                game_id="g1",
                episode_id="e2",
                probability_delta=-0.04,
                continuation_delta=None,
                venue="polymarket",
                market_id="market-2",
                outcome="away_win",
            ),
        ),
        left_venue="kalshi",
        right_venue="polymarket",
    )

    assert q_values == {
        "a": pytest.approx(0.03),
        "b": pytest.approx(0.04),
        "c": pytest.approx(0.04),
    }
    assert paired["pair_count"] == 1
    assert paired["right_minus_left_mean"] == pytest.approx(0.03)
    assert paired["pairs"][0]["game_id"] == "g1"
    assert paired["pairs"][0]["episode_id"] == "e1"
    assert paired["pairs"][0]["logical_proposition_id"] == (
        "proposition-g1-home_win"
    )


@pytest.mark.parametrize(
    ("field", "right_value"),
    (
        ("logical_proposition_id", "different-proposition"),
        ("market_family", "spread"),
        ("period", "first_half"),
        ("line", 1.5),
        ("settlement_rule", "different-settlement-v1"),
        ("outcome", "away_win"),
    ),
)
def test_same_episode_pairing_rejects_different_market_propositions(
    field: str,
    right_value: object,
) -> None:
    left = _observation(
        game_id="g1",
        episode_id="e1",
        probability_delta=0.02,
        continuation_delta=None,
        venue="kalshi",
    )
    right = replace(
        left,
        venue="polymarket",
        market_id="market-2",
        probability_delta=0.05,
        value=0.05,
        **{field: right_value},
    )

    paired = compare_same_episode_venues(
        observations=(left, right),
        left_venue="kalshi",
        right_venue="polymarket",
    )

    assert paired["pair_count"] == 0


def test_calibration_forecast_and_label_are_contract_bound_and_pit_safe() -> None:
    observation = _calibration_observation(
        game_id="g1",
        reference_id="r1",
        forecast_probability=0.7,
        realized_label=1,
        favorite_underdog_proxy="favorite",
        home_away_proxy="home",
        popularity_proxy="popular_team",
    )

    with pytest.raises(MetricContractError, match="label contract"):
        replace(
            observation,
            realized_label_contract_id="different-label:v1",
        )
    with pytest.raises(MetricContractError, match="PIT"):
        replace(observation, label_time=observation.forecast_time)
    with pytest.raises(MetricContractError, match="holdout"):
        replace(observation, dataset_partition="holdout")


def test_calibration_rejects_duplicate_forecasts_and_single_game_ticks() -> None:
    first = _calibration_observation(
        game_id="g1",
        reference_id="r1",
        forecast_probability=0.7,
        realized_label=1,
        favorite_underdog_proxy="favorite",
        home_away_proxy="home",
        popularity_proxy="popular_team",
    )
    second_tick = _calibration_observation(
        game_id="g1",
        reference_id="r2",
        forecast_probability=0.6,
        realized_label=1,
        favorite_underdog_proxy="favorite",
        home_away_proxy="home",
        popularity_proxy="popular_team",
        forecast_offset=1,
    )
    relabeled_same_reference_tick = replace(
        first,
        forecast_id="forecast-g1-r1-tick-2",
        forecast_time=first.forecast_time + timedelta(seconds=1),
        label_time=first.label_time + timedelta(seconds=1),
    )
    other_game = _calibration_observation(
        game_id="g2",
        reference_id="r2",
        forecast_probability=0.3,
        realized_label=0,
        favorite_underdog_proxy="underdog",
        home_away_proxy="away",
        popularity_proxy="other_team",
    )

    with pytest.raises(MetricContractError, match="duplicate forecast"):
        metrics.evaluate_game_clustered_calibration(
            (first, first),
            bin_count=5,
            bootstrap_draw_count=40,
            seed=7,
        )
    with pytest.raises(MetricContractError, match="duplicate forecast"):
        metrics.evaluate_game_clustered_calibration(
            (first, relabeled_same_reference_tick, other_game),
            bin_count=5,
            bootstrap_draw_count=40,
            seed=7,
        )
    with pytest.raises(MetricContractError, match="duplicate forecast"):
        metrics.evaluate_game_clustered_calibration(
            (first, second_tick),
            bin_count=5,
            bootstrap_draw_count=40,
            seed=7,
        )


def test_calibration_rejects_many_reference_ticks_for_one_game() -> None:
    repeated_game_ticks = tuple(
        _calibration_observation(
            game_id="g1",
            reference_id=f"reference-{index}",
            forecast_probability=0.55 + index / 100,
            realized_label=1,
            favorite_underdog_proxy="favorite",
            home_away_proxy="home",
            popularity_proxy="popular_team",
            forecast_offset=index,
        )
        for index in range(10)
    )
    other_games = (
        _calibration_observation(
            game_id="g2",
            reference_id="reference-g2",
            forecast_probability=0.4,
            realized_label=0,
            favorite_underdog_proxy="underdog",
            home_away_proxy="away",
            popularity_proxy="other_team",
        ),
        _calibration_observation(
            game_id="g3",
            reference_id="reference-g3",
            forecast_probability=0.6,
            realized_label=1,
            favorite_underdog_proxy="favorite",
            home_away_proxy="home",
            popularity_proxy="other_team",
        ),
    )

    with pytest.raises(MetricContractError, match="duplicate forecast"):
        metrics.evaluate_game_clustered_calibration(
            repeated_game_ticks + other_games,
            bin_count=5,
            bootstrap_draw_count=40,
            seed=7,
        )


def test_game_clustered_calibration_reports_metrics_bins_ci_and_proxy_breakdowns() -> None:
    observations = (
        _calibration_observation(
            game_id="g1",
            reference_id="r1",
            forecast_probability=0.8,
            realized_label=1,
            favorite_underdog_proxy="favorite",
            home_away_proxy="home",
            popularity_proxy="popular_team",
        ),
        _calibration_observation(
            game_id="g2",
            reference_id="r2",
            forecast_probability=0.6,
            realized_label=0,
            favorite_underdog_proxy="favorite",
            home_away_proxy="away",
            popularity_proxy="other_team",
        ),
        _calibration_observation(
            game_id="g3",
            reference_id="r3",
            forecast_probability=0.7,
            realized_label=1,
            favorite_underdog_proxy="underdog",
            home_away_proxy="home",
            popularity_proxy="other_team",
        ),
        _calibration_observation(
            game_id="g4",
            reference_id="r4",
            forecast_probability=0.3,
            realized_label=0,
            favorite_underdog_proxy="underdog",
            home_away_proxy="away",
            popularity_proxy="popular_team",
        ),
        _calibration_observation(
            game_id="g5",
            reference_id="r5",
            forecast_probability=0.4,
            realized_label=1,
            favorite_underdog_proxy="favorite",
            home_away_proxy="home",
            popularity_proxy="popular_team",
        ),
        _calibration_observation(
            game_id="g6",
            reference_id="r6",
            forecast_probability=0.2,
            realized_label=0,
            favorite_underdog_proxy="underdog",
            home_away_proxy="away",
            popularity_proxy="other_team",
        ),
    )

    report = metrics.evaluate_game_clustered_calibration(
        observations,
        bin_count=5,
        bootstrap_draw_count=80,
        seed=17,
    )

    expected_log_loss = -sum(
        label * math.log(probability)
        + (1 - label) * math.log(1 - probability)
        for probability, label in (
            (0.8, 1),
            (0.6, 0),
            (0.7, 1),
            (0.3, 0),
            (0.4, 1),
            (0.2, 0),
        )
    ) / 6
    assert report["cluster_unit"] == "game_id"
    assert report["game_count"] == 6
    assert report["forecast_count"] == 6
    assert report["brier_score"] == pytest.approx(0.98 / 6)
    assert report["log_loss"] == pytest.approx(expected_log_loss)
    assert math.isfinite(report["calibration_slope"])
    assert math.isfinite(report["calibration_intercept"])
    assert sum(row["forecast_count"] for row in report["reliability_bins"]) == 6
    assert set(report["game_cluster_bootstrap_ci"]) == {
        "brier_score",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
    }
    assert report["bootstrap_valid_draw_count"] > 0
    assert set(report["proxy_breakdowns"]) == {
        "favorite_underdog_proxy",
        "home_away_proxy",
        "popularity_proxy",
    }
    assert all(
        name.endswith("_proxy") for name in report["proxy_breakdowns"]
    )


def test_shadow_strategy_requires_formal_lock_and_binds_locked_candidate(
    tmp_path: Path,
) -> None:
    manifest_path, manifest_sha256 = _real_shortlist_manifest()

    strategy = _strategy(manifest_path, manifest_sha256)

    assert strategy.shortlist_manifest_path == manifest_path
    assert strategy.shortlist_manifest_sha256 == manifest_sha256
    assert strategy.factor_id == "NFL.EVENT.EXPLOSIVE_PLAY"
    assert strategy.factor_version == "v1"

    arbitrary_json = tmp_path / "arbitrary.json"
    arbitrary_json.write_text(
        '{"holdout_reaction_accessed":false}',
        encoding="utf-8",
    )
    with pytest.raises(MetricContractError, match="formal shortlist lock"):
        _strategy(arbitrary_json, SHA_ONE)

    with pytest.raises(MetricContractError, match="locked candidate"):
        replace(strategy, factor_id="NFL.EVENT.NOT_LOCKED")

    holdout_path, holdout_sha256 = _formal_lock_with_holdout_accessed(tmp_path)
    with pytest.raises(MetricContractError, match="holdout"):
        _strategy(holdout_path, holdout_sha256)


def test_trade_marked_shadow_nav_is_non_executable_but_reports_nav_mdd_and_sharpe(
) -> None:
    manifest_path, manifest_sha256 = _real_shortlist_manifest()
    strategy = _strategy(manifest_path, manifest_sha256)
    nav = (
        _nav(
            nav=100.0,
            episode_id="e1",
            offset=0,
            price_basis="ACTUAL_SUBSEQUENT_TRADE_MARK",
            fill_status="NO_FILL_OBSERVED_NON_EXECUTABLE",
            fees=0.1,
        ),
        _nav(
            nav=110.0,
            episode_id="e2",
            offset=60,
            price_basis="ACTUAL_SUBSEQUENT_TRADE_MARK",
            fill_status="NO_FILL_OBSERVED_NON_EXECUTABLE",
            fees=0.2,
        ),
        _nav(
            nav=99.0,
            episode_id="e3",
            offset=120,
            price_basis="ACTUAL_SUBSEQUENT_TRADE_MARK",
            fill_status="NO_FILL_OBSERVED_NON_EXECUTABLE",
            fees=0.3,
        ),
    )

    summary = summarize_shadow_nav(
        nav,
        strategy=strategy,
        inventories=(_inventory(capabilities=frozenset({"trade_tick"})),),
    )

    assert (
        summary["evidence_classification"]
        == "NON_EXECUTABLE_TRADE_MARKED_SHADOW"
    )
    assert summary["price_basis"] == "ACTUAL_SUBSEQUENT_TRADE_MARK"
    assert summary["fees"] == pytest.approx(0.6)
    assert summary["fill_status"] == "NO_FILL_OBSERVED_NON_EXECUTABLE"
    assert summary["executable"] is False
    assert summary["order_status"] == "NO_ORDERS_CREATED"
    assert summary["terminal_nav"] == pytest.approx(99.0)
    assert summary["maximum_drawdown"] == pytest.approx(0.1)
    assert summary["sharpe_ratio"] is not None


def test_execution_capability_is_separately_labeled_and_never_claims_live_orders(
) -> None:
    manifest_path, manifest_sha256 = _real_shortlist_manifest()
    strategy = _strategy(manifest_path, manifest_sha256)
    nav = (
        _nav(
            nav=100.0,
            episode_id="e1",
            offset=0,
            price_basis="EXECUTION_REPLAY_NO_ORDER",
            fill_status="NO_STRATEGY_FILL_OBSERVED",
            fees=0.1,
        ),
        _nav(
            nav=101.0,
            episode_id="e2",
            offset=60,
            price_basis="EXECUTION_REPLAY_NO_ORDER",
            fill_status="NO_STRATEGY_FILL_OBSERVED",
            fees=0.2,
        ),
        _nav(
            nav=100.0,
            episode_id="e3",
            offset=120,
            price_basis="EXECUTION_REPLAY_NO_ORDER",
            fill_status="NO_STRATEGY_FILL_OBSERVED",
            fees=0.2,
        ),
    )

    summary = summarize_shadow_nav(
        nav,
        strategy=strategy,
        inventories=(_inventory(capabilities=frozenset({"execution"})),),
    )

    assert summary["evidence_classification"] == "EXECUTION_GRADE_SHADOW"
    assert summary["price_basis"] == "EXECUTION_REPLAY_NO_ORDER"
    assert summary["fill_status"] == "NO_STRATEGY_FILL_OBSERVED"
    assert summary["executable"] is False
    assert summary["order_status"] == "NO_ORDERS_CREATED"


def test_trade_tick_rejects_fabricated_fill_strings(
) -> None:
    manifest_path, manifest_sha256 = _real_shortlist_manifest()
    strategy = _strategy(manifest_path, manifest_sha256)

    with pytest.raises(MetricContractError, match="price_basis|fill_status"):
        summarize_shadow_nav(
            (
                _nav(
                    nav=100.0,
                    episode_id="e1",
                    offset=0,
                    price_basis="execution_replay",
                    fill_status="ACTUAL_FILL",
                    fees=0.1,
                ),
            ),
            strategy=strategy,
            inventories=(
                _inventory(capabilities=frozenset({"trade_tick"})),
            ),
        )


@pytest.mark.parametrize(
    ("capability", "price_basis", "fill_status"),
    (
        (
            "trade_tick",
            "EXECUTION_REPLAY_NO_ORDER",
            "NO_STRATEGY_FILL_OBSERVED",
        ),
        (
            "trade_tick",
            "ACTUAL_SUBSEQUENT_TRADE_MARK",
            "NO_STRATEGY_FILL_OBSERVED",
        ),
        (
            "execution",
            "ACTUAL_SUBSEQUENT_TRADE_MARK",
            "NO_FILL_OBSERVED_NON_EXECUTABLE",
        ),
        (
            "execution",
            "EXECUTION_REPLAY_NO_ORDER",
            "NO_FILL_OBSERVED_NON_EXECUTABLE",
        ),
    ),
)
def test_shadow_nav_rejects_cross_capability_and_mixed_label_pairs(
    capability: str,
    price_basis: str,
    fill_status: str,
) -> None:
    manifest_path, manifest_sha256 = _real_shortlist_manifest()
    strategy = _strategy(manifest_path, manifest_sha256)
    observation = _nav(
        nav=100.0,
        episode_id="e1",
        offset=0,
        price_basis=price_basis,
        fill_status=fill_status,
        fees=0.1,
    )

    with pytest.raises(MetricContractError, match="capability-bound"):
        summarize_shadow_nav(
            (observation,),
            strategy=strategy,
            inventories=(
                _inventory(capabilities=frozenset({capability})),
            ),
        )


def test_shadow_nav_rejects_missing_nav_unsupported_capability_and_executable_claim(
) -> None:
    manifest_path, manifest_sha256 = _real_shortlist_manifest()
    strategy = _strategy(manifest_path, manifest_sha256)
    valid = _nav(
        nav=100.0,
        episode_id="e1",
        offset=0,
        price_basis="ACTUAL_SUBSEQUENT_TRADE_MARK",
        fill_status="NO_FILL_OBSERVED_NON_EXECUTABLE",
        fees=0.1,
    )

    with pytest.raises(MetricContractError, match="no NAV"):
        summarize_shadow_nav(
            (),
            strategy=strategy,
            inventories=(_inventory(capabilities=frozenset({"trade_tick"})),),
        )
    with pytest.raises(MetricContractError, match="trade_tick.*execution"):
        summarize_shadow_nav(
            (valid,),
            strategy=strategy,
            inventories=(_inventory(capabilities=frozenset({"candle_l1"})),),
        )
    with pytest.raises(MetricContractError, match="executable.*false"):
        replace(valid, executable=True)
